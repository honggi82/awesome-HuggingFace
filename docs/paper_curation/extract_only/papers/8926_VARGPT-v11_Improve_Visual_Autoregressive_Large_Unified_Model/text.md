# arXiv:2504.02949v1[cs.CV]3Apr2025

## VARGPT-v1.1: Improve Visual Autoregressive Large Unified Model via Iterative Instruction Tuning and Reinforcement Learning

Xianwei Zhuang1*, Yuxin Xie 1∗, Yufan Deng1∗, Dongchao Yang2, Liming Liang1, Jinghan Ru1, Yuguo Yin1, Yuexian Zou1 † 1School of Electronic and Computer Engineering, Peking University 2The Chinese University of Hong Kong

{xwzhuang, yuxinxie, dengyufan10}@stu.pku.edu.cn

Technical Report Project Page: https://vargpt1-1.github.io/ Codes: https://github.com/VARGPT-family/VARGPT-v1.1

[Figure 1]

Figure 1. Some generated 512×512 samples by VARGPT-v1.1. Our VARGPT-v1.1 supports text-and-image instructions from user and outputs both text-and-image mixed modal data simultaneously.

#### Abstract

In this work, we present VARGPT-v1.1, an advanced unified visual autoregressive model that builds upon our previous framework VARGPT. The model preserves the dual paradigm of next-token prediction for visual understanding and next-scale generation for image synthesis. Specifically, VARGPT-v1.1 integrates: (1) a novel training strategy combining iterative visual instruction tuning with re-

*Equal contribution †Corresponding author

inforcement learning through Direct Preference Optimization (DPO), (2) an expanded training corpus containing 8.3M visual-generative instruction pairs, (3) an upgraded language model backbone using Qwen2, (4) enhanced image generation resolution, and (5) emergent image editing capabilities without architectural modifications. These advancements enable VARGPT-v1.1 to achieve state-of-the-art performance in multimodal understanding and text-to-image instruction-following tasks, demonstrating significant improvements in both comprehension and generation metrics. Notably, through visual instruction tuning, the model ac-

VizWizQA POPE (adv.)

65.4 84.4

ScienceQA-IMG

83.6

91.6

56.8

72.1

53.5

MME

80.1

1684.1

67.9

1488.8

60.5 34.5

1480.1

1212.8

49.7

GQA

49.2

66.2

36.2502.8 33.5

61.2

62.3

36.0

48.8

62.7

54.1

67.6

53.4

81.0

MMBench(en)

MiniGPT-4

65.4

InstructBLIP

67.9

82.0

Qwen-VL

Qwen-VL-Chat

TextVQA

LLaVA

76.5

76.1

mPLUG-Owl2 LLaVA-1.5-hf Janus-Pro(7B)

78.4

80.4

SEED-Bench

VARGPT

VQAv2

VARGPT-v1.1(Ours)

- Figure 2. A comparative analysis of various MLLMs across multiple visual comprehension benchmarks is presented. The remaining metrics are derived from standard visual question-answering benchmarks and multi-modal comprehension benchmarks. Notably, our VARGPT-v1.1 model demonstrates significant superiority over the compared baselines across all comprehension benchmarks.

quires image editing functionality while maintaining architectural consistency with its predecessor, revealing the potential for unified visual understanding, generation, and editing. Our findings suggest that well-designed unified visual autoregressive models can effectively adopt flexible training strategies from large language models (LLMs), exhibiting promising scalability. The codebase and model weights are publicly available at https://github.com/VARGPTfamily/VARGPT-v1.1.

#### 1. Introduction

Recent advancements in multimodal AI have achieved dual breakthroughs in comprehension and generation. While Multimodal Large Language Models (MLLMs) [4, 9, 12, 48, 98] excel at cross-modal understanding, Denoising Diffusion Models [27, 61] dominate visual generation through iterative refinement. This progress drives the pursuit of unified frameworks (Figure 3), where three primary paradigms emerge: (1) Assembly Systems integrating LLMs with diffusion models [20, 93, 95], (2) Pure Autoregression architectures predicting visual tokens [10, 53, 80, 91], and (3) Dualdiffusion Models with parallel generation mechanisms [46]. Current implementations face representation conflicts between understanding and generation tasks, while TokenFlow [64] unifies tokenization, its visual generation and understanding pipeline remains decoupled. Emerging solutions [10, 24, 75, 78, 84, 89, 91, 92] address this through architectural innovations that preserve task-specific representations while enabling joint optimization

VARGPT [107] endeavors to unify visual generation and understanding within a visual autoregressive MLLM, naturally supporting mixed-modal input and output. VARGPT models understanding and generation as two distinct paradigms within a unified model: predicting the next token for visual understanding and predicting the next scale for visual generation, respectively. However, due to the limited scale of training data and relatively insufficient training iterations, such approaches exhibit certain limitations, including suboptimal performance in instruction-following capabilities for image generation and constrained coverage of image generation domains (primarily limited to ImageNet [13] categories). To address these limitations, we present VARGPTv1.1, which implements comprehensive enhancements over its predecessor while preserving the fundamental paradigm of VARGPT. Specifically, VARGPT-v1.1 achieves acrossthe-board improvements in both generative and comprehension capabilities through several key technical innovations:

VARGPT-v1.1 incorporates four fundamental technical innovations: (1) A multi-stage training paradigm combining iterative visual instruction tuning with reinforcement learning, where Direct Preference Optimization (DPO) [66] aligns model outputs with preferences through pairwise reward modeling. Our progressive resolution scaling (from 256×256 to 512×512) with alternating SFT-RL phases substantially enhances generation fidelity; (2) An expanded visual generation corpus containing 8.3 million instruction pairs (6× larger than v1.0), comprising 4.2M carefully curated real-world samples from LAION-COCO [70] and 4.1M synthetic data from Midjourney v6 [62] and Flux [34]; (3) Enhanced visual comprehension through migration to the Qwen2-7B [2] backbone, leveraging its improved tokenization efficiency and attention mechanisms; (4) Architecture-agnostic fine-tuning using constructed imageediting datasets, enabling VARGPT-v1.1 to acquire visual editing capabilities without structural modifications.

Through these improvements, our VARGPT-v1.1 demonstrates significant advancements in both comprehension and generation capabilities, outperforming comparable unified MLLMs in multimodal tasks. Extensive experiments reveal that VARGPT-v1.1 achieves state-of-the-art visual understanding performance (as illustrated in Figure 2) while surpassing existing unified models and multimodal large language models across various benchmarks. Furthermore, VARGPT-v1.1 exhibits three distinctive advantages: (1) enhanced visual generation quality with improved instruction adherence (Figure 1), (2) flexible image editing capabilities without requiring task-specific architectures, and (3) maintained generation diversity despite increased output resolution. These advancements are quantitatively validated through comprehensive human evaluations and automated metrics comparisons.

In summary, our main contributions are threefold:

Text

Image

Text

Image

Text

Text

Image

Text

Image

LLM (AR) or Diffusion

LLM (AR or AR+Diffusion)

LLM (AR)

LLM (AR)

LLM (AR + VAR)

Diffusion

Text

Image

Image

Text

Text Image

Text Image

(1) Understanding Only (2) Generation Only (3) Unified Model Architecture 1 (4) Unified Model Architecture 2 (5) VARGPT (Ours)

- Figure 3. Comparison of different model architectures, where, ‘AR’ denotes autoregressive, while ‘VAR’ signifies visual autoregressive. We present a comparative analysis of architectures designed for comprehension-only tasks, generation-only tasks, and unified comprehension and generation, alongside our proposed VARGPT-v1.1 an VARGPT [14] model. Our VARGPT-v1.1 and VARGPT are conceptualized as purely autoregressive multimodel model, achieving visual comprehension through next-token prediction and visual generation through next-scale prediction paradigms.

- • We propose VARGPT-v1.1, which comprehensively improves VARGPT in terms of data size, backbone model types, and training strategies, making significant progress in unifying visual understanding and generation.
- • We propose to train VARGPT-v1.1 by gradually increasing the resolution and iterating instruction tuning and reinforcement learning.
- • Extensive experiments and evaluations demonstrate that the model achieves more advanced visual understanding and image generation capabilities than the VARGPT model, and has certain image editing capabilities.

#### 2. Related Work

Visual Generation and Multimodal Learning. Modern visual generation frameworks include diffusion models [28, 73, 74], which reverse noise-to-image diffusion through optimized sampling [6, 57] and architectures [29, 63], and flow-based models [3] offering simplified alternatives. Autoregressive approaches [16, 100] adapt GPT-style token prediction, using visual tokenizers [14, 17, 36, 67, 75, 76, 89] for discrete image representations. Recent scale-predictive variants (VAR [84], HART [78]) align with scaling law principles [26, 32]. Concurrently, Multimodal Large Language Models (MLLMs) [85, 86] integrate visual encoders with text decoders through cross-modal connectors [35, 43]. Systems like LLaVA [49] and its successors [39, 40, 42, 50, 52, 103] enhance visual-language understanding via instructional tuning and high-quality datasets, while architectures such as Qwen-VL [4] and InstructBLIP [12] demonstrate advances in training paradigms and multimodal fusion. Our work unifies autoregressive generation through scale prediction while leveraging insights from these multimodal frameworks.

Unified Models For Visual Understanding and Generation. In recent years, researchers have been dedicated to unifying understanding and generation capabilities within a

model [15, 79, 97]. Most existing approaches [19, 77, 93] attempt to integrate pre-trained diffusion models with existing systems. However, these systems essentially treat diffusion models as external tools rather than incorporating them as intrinsic generative capabilities of MLLMs. Show-

- o [96] by combining autoregressive and (discrete) diffusion modeling, can adaptively handle inputs and outputs of vari-
- ous mixed modalities. Li et al. [46] employs a cross-modal maximum likelihood estimation framework, significantly improving upon existing diffusion-based multimodal models. [5, 83] have explored integrating image generation into large language models (LLMs) using autoregressive methods, achieving remarkable results. For instance, LWM [53] and Chameleon [81] utilize a VQ tokenizer [16, 84] to encode images, enabling simultaneous support for multimodal understanding and generation. Janus [91] further enhances model flexibility and performance by decoupling visual encoding into separate pathways, while Dual Diffusion [46] investigated the use of two diffusion models for understanding and generation. Liquid [92] learns images and text embeddings within the same space and implements autoregressive visual understanding and generation using the paradigm of predicting the next token. Building on these foundational efforts, several recent works have further explored the integration of visual understanding and generation. QLIP [105] proposes a text-aligned visual tokenization approach that unifies auto-regressive multimodal understanding and generation, highlighting the importance of aligning visual and textual representations. VILA-U [94] aims to develop a unified foundation model that integrates visual understanding and generation, focusing on improving the coherence between these two tasks. UniTok [59] introduces a unified tokenizer for visual generation and understanding, demonstrating the potential of a single tokenization framework for both tasks. MINT [90] explores multi-modal chain of thought in unified generative models, enhancing image gen-

Multi-Scale Image Tokenizer (VAE)

Text Tokenizer

Visual Encoder

Projector Gen Projector

|Large Language Model (Qwen2-7B)|
|---|

|Visual Decoder (2B)|
|---|

Text Detokenizer

- Figure 4. The illustration of the proposed VARGPT-v1.1 framework similar to VARGPT [107], which consists of (1) an LLM (Qwen2-7BInstruct [2, 87]), visual encoder and a understanding projector for visual understanding; (2) a visual decoder and dual generation projectors for visual generation. VARGPT-v1.1 employs causal attention in the LLM backbone while utilizing block causal attention in the visual decoder.

eration through more sophisticated reasoning mechanisms. Janus-Pro [10] scales up the data and model size for unified multimodal understanding and generation, showing that larger models and datasets can significantly improve performance. Different from all existing unified models, we propose modeling understanding and generation as two distinct paradigms within a unified model: predicting the next token for visual understanding and predicting the next scale for visual generation.

Reinforcement Learning Reinforcement learning (RL) has become a powerful technique to enhance the alignment, reasoning, and decision-making capabilities of large language models (LLMs). Recent works have explored using AI feedback, step-level verification, and policy optimization to improve model performance while reducing training costs. DeepSeek-R1 [82] significantly advances RL for LLMs by scaling reinforcement fine-tuning (RFT) with AI feedback, achieving high-quality preference alignment without human annotations. Math-Shepherd [88] and DeepSeek-Math [71] introduce step-level verification and group preference optimization to improve mathematical reasoning. Notably, Group Relative Policy Optimization (GRPO) [71] eliminates the need for a critic model in policy optimization, greatly reducing training overhead while maintaining superior performance. In parallel, LIMR [44] improves RL efficiency by training on high-impact samples, reducing data requirements by 84%. Additionally, works like Visual-RFT [56] and Grounding-LLM [7] extend RL to multimodal and interactive environments, improving decision-making and real-time

response. MM-RLHF [104] introduces a dataset containing 120,000 fine-grained, human-annotated preference comparison pairs, representing a substantial advancement over existing resources in size, diversity, annotation granularity, and quality. These advances demonstrate the critical role of RL in scaling and optimizing LLM capabilities.

#### 3. Methodology

##### 3.1. Model Architecture

Our VARGPT-v1.1 follows the model architecture design of VARGPT to unify visual understanding and generation, which architecture is shown in the Figure 4. Our VARGPTV1.1 architecture follows the paradigm of predicting the next token for visual understanding and adheres to the paradigm of predicting the next scale for image generation.

###### Visual understanding via next-token prediction

Following VARGPT, our model architecture of VARGPTv1.1 references the structure of Qwen2-VL [87], using Qwen2 [2] as the LLM θ, and employing a Vision Transformer (ViT) [65] as the visual encoder along with a twolayer linear network as the projector. Initially, the image Ximg for visual understanding undergoes processing through the vision encoder to produce embeddings Himg, which are then modified by the interface to align with the textual embeddings Htxt obtained by query Xquery. The combined data serves as input to the LLM, which autoregressively generates the textual output Ytxt as:

Yttxt ∼ pθ(Yttxt | Ximg,Xquery,Y<ttxt), (1)

[Figure 2]

[Figure 3]

[Figure 4]

Visual Encoder

Visual Encoder

Visual Encoder

Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua

Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua

Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua

[Figure 5]

Projector Projector

Projector Projector

[Figure 6]

Projector Projector

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

LLM (Qwen2-7B)

LLM (Qwen2-7B)

LLM (Qwen2-7B)

[Figure 13]

Projector

lm head

Projector

lm head

Projector

lm head

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

Visual Decoder

Visual Decoder

Visual Decoder

Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua

Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua

Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua

Stage 1. Pretraining Stage 2. Supervised Finetuning of Understanding

Stage 3. Iterative Training of Generation

- Figure 5. The three training stages of the VARGPT, including stage-1 pretraining, stage-2 visual instruction tuning and stage-3 iterative training.

Supervise Fine-tuning (256*256) 40K step

RLHF: DPO (256*256) 5K step

Supervise Fine-tuning (512*512) 30K step

RLHF: DPO (512*512) 3K step

Supervise Fine-tuning for Visual Editing (512*512)

Reference Model Reference Model

- Figure 6. The proposed iterative training strategy for the third stage gradually increases the resolution of the image, while using instruction fine-tuning and reinforcement learning iterative training. Finally, we use the instruction-follow dataset for image editing to stimulate the model’s visual editing ability.

where Yttxt represents the t-th token of Ytxt, and Y<ttxt refers to the sequence of tokens generated prior to the t-th step. To maintain the causal attention property of LLM, we apply a causal attention mask to all input LLM tokens, including those used for generating images.

Visual generation via next-scale prediction For visual generation, we follow most of the settings in Infinity [24] and VAR [84] and adopt a multi-scale image tokenizer with bitwise multi-scale residual quantizer [24] for visual token encoding and decoding. We construct two image generation projectors to transform the visual features used for generation at the input and output of the LLM. Additionally, we construct an extra 2B visual decoder ϕ with 32 layers of Transformers to decode visual features, which can avoid conflicts between knowledge in the text decoder and image generation knowledge. The image features obtained through the visual decoder will be further decoded through the multiscale VAE decoder to produce usable images. Unlike the text decoder (i.e., the LLM), the visual decoder uses attention that follows the block causal attention in Infinity [24] and

VAR [84] to support predicting tokens for the next scale. Unlike VARGPT, our VARGPT-v1.1 utilizes the infinite vocabulary classifier [24] technology from Infinity to compute training losses related to visual generation.

Prompt template for mixed-modal generation and Classifier-free guidance (CFG) To differentiate between tokens designated for textual generation and those for image synthesis, we use some special token markers similar to VARGPT to train our VARGPT-v1.1. In addition, we use CFG techniques similar to VARGPT to improve the quality of image generation.

##### 3.2. Training

The training of our VARGPT-v1.1 follows the three-stage training method of VARGPT, and the overall training process is shown in Figure 5. However, in the stage-3, we propose the methods of iterative instruction tuning and reinforcement learning to enhance the visual generation ability of the previous model. Specifically, the iterative training process in our third stage is as shown in Figure 6.

###### 3.2.1. Visual Instruction Tuning

The instruction tuning for visual generation mainly aims to endow VARGPT-v1.1 with the visual generation ability through supervised fine-tuning. At this stage, we unfreeze the visual decoder and the two projectors and freeze other parameters to perform SFT, as shown in Figure 6. We employ a training method that gradually increases the image resolution to train VARGPT-v1.1. Specifically, in the first SFT stage, the image resolution is set to 256×256, and the model is trained for 40,000 steps to endow it with the initial ability to generate images. In the second SFT stage, the image resolution is set to 512×512, and the model is trained for 30,000 steps to further enhance its high-resolution visual generation ability. The training data for the third stage includes 8.3M instruction pairs that are collected and constructed (for details, please refer to Section 4).

###### 3.2.2. Reinforcement Learning Human Feedback

We propose an iterative training framework that integrates instruction fine-tuning and reinforcement learning in visual autoregressive large visual-language models. We formulate the improvement of generation quality as a preference selection problem through preference learning. This approach incentivizes the model to favor higher-quality image outputs yw while rejecting inferior counterparts yl. During the preference learning phase, a reward model denoted as rˆis trained to reinforce the model’s propensity towards producing yw outputs. This reward model demonstrates the capability to assign differential scores to various responses, thereby accurately reflecting human preferences. Subsequently, the acquired reward model r is employed to provide feedback for guiding an additional fine-tuning stage, which learns a policy model πθ governed by human preference guidance.

Specifically, after each SFT phase, we implement direct preference optimization (DPO) [66] for images at various resolutions to enhance the model’s visual generation capabilities. For preference data construction, we sample from models at different SFT checkpoints and from the fully SFTtrained model. We sample 100k prompts from the Midjourney prompt dataset1 as input prompts x to obtain our model’s outputs yl. Concurrently, we utilize a robust commercial model (Midjourney v6 and Flux-1.1 pro) to generate images yw from these prompts. This sampling approach from both stronger and weaker models naturally creates a partially ordered dataset D = {(x,yl,yw)}. We leverage the post-SFT model as the reference model πref, and optimize the policy model using the following objective function:

1huggingface . co / datasets / vivym / midjourney prompts

Lrl(πθ;πref) = −E(x,y

w,yl)∼D

πθimg(yw|x) πrefimg(yw|x) − β log

πθimg(yl|x) πrefimg(yl|x)

(2)

[log σ(β log

)],

where, πθimg and πrefimg denote the policy model and reference model for the generated image tokens, log σ representing the

log sigmoid function and β is the hyperparameters related to rewards. Rather than explicitly training the reward model, Eq. 2 directly optimizes the policy model by implicitly representing the reward r through:

πθimg(y|x) πrefimg(y|x)

. (3)

rˆ(x,y) = β log

Therefore, we can formulate Eq. 2 to maximize the reward margin rˆ(x,yw) − rˆ(x,yl).

The proposed VARGPT-v1.1 is capable of simultaneously generating both text tokens and image tokens. During the reinforcement learning phase, exclusive focus is placed on the generated image tokens, where the image-specific policy models πθimg and πrefimg are formulated as:

π∗img = softmax(logit∗[img]),∗ ∈ {θ,ref}, (4)

where, logit∗[img] denotes the logits corresponding to the positions of generated image tokens.

###### 3.2.3. SFT for Visual Editing

Following multi-stage resolution-progressive iterations of SFT and DPO, we systematically construct an instructiontuning dataset comprising 11,325 samples from StyleBooth [25] 2 to enable visual editing capabilities in VARGPTv1.1. The pipeline processes target images through the visual encoder while utilizing editing instructions as textual prompts to supervise the model’s approximation of edited image distributions. Crucially, this approach achieves: (1) architecture-preserving adaptation through parameter space optimization, and (2) unified multimodal editing via joint text-image token prediction. All model parameters were unfrozen during SFT to maximize editing fidelity while maintaining generation diversity, following recent advances in full-parameter fine-tuning for multimodal systems.

#### 4. Instruction-following and Preference Data 4.1. Instruction-following Data

For the visual understanding instruction-following dataset, we maintain complete consistency with VARGPT’s training data composition from LLaVA-1.5-665K [51] and LLaVAOneVision [39, 40, 42, 52, 103], and same preprocessing

2https://huggingface.co/datasets/yzwang/X2I-mminstruction

Stage 3 SFT(256) DPO(256) SFT(512) DPO(512) Stage e

Hyperparameter Stage 1 Stage 2

batch size 1024 256 1024 384 512 384 512 lr 1e-3 5e-5 5e-5 1e-6 5e-5 1e-6 5e-5 lr schedule Cosine Cosine Cosine Cosine Cosine Cosine Cosine lr warmup ratio 0.1 0.1 0.1 0.1 0.1 0.1 0.1 weight decay 0 0 0 0 0 0 0 training step 2k 3.1k 40k 5k 30k 3k 2k optimizer AdamW AdamW AdamW AdamW AdamW AdamW AdamW DeepSpeed ZERO type 3 2 3 3 3 3 3

- Table 1. Main hyperparameters and training settings for training our VARGPT-v1.1 at various stages, where, (256) and (512) represent the image generation resolutions of 256 × 256 and 512 × 512, respectively

[Figure 18]

##### 4.2. Preference Instruction Data

- Stage-1 8.3M

- a. laion-coco-aesthetic 4.2M
- b. midjourney 4.1M

- Stage-2 1.2M

- a. LLaVA-1.5 0.665M
- b. LLaVA-OneVision 0.508M
- c. generate 50K

- Stage-3 8.3M

We employ iterative instruction tuning and reinforcement learning to train our VARGPT-v1.1 model, which necessitates the construction of partially ordered datasets for implementing DPO. We randomly sample 180k prompts from the Midjourney prompt dataset3. For each prompt, we generate "losing" images using our SFT model after initial training, while utilizing corresponding Midjourney and Fluxgenerated images as "winning" examples to construct the partially ordered dataset for DPO training. This approach leverages the prior that commercial generative models currently outperform our in-training model to automatically create preference data, effectively circumventing the need for labor-intensive manual annotation.

###### Data 9.5M

- a. laion-coco-aesthetic 4.2M
- b. midjourney 4.1M
- c. StyleBooth (Image Editing) 11K

- Figure 7. We present the data distribution we construct and collect, encompassing the proportional breakdown of data across the three training stages. Our composite dataset for stage-2 training is derived from LLaVA-1.5, LLaVA-OneVision.

pipeline. The visual generation dataset comprises 8.3M image-text pairs from two complementary sources: (1) 4.2M real-world images from LAION-COCO [1] filtered through aesthetic scoring, and (2) 4.1M synthetic samples generated by Midjourney collected from JourneyDB [62] and other public repositories. This 1:1 real-synthetic ratio leverages recent findings that synthetic data accelerates model convergence [10].

We implement a three-stage curriculum training strategy: For the Stage 1: Full 8.3M generation dataset for foundational capabilities. For the Stage 2: Hybrid training with 1.18M understanding samples (from VARGPT) and 50K generation samples from constructed 8.3M training data. For the Stage 3: Final refinement using the complete 8.3M generation set. The progressive training framework inherits VARGPT’s instruction construction methodology while introducing dynamic batch balancing between modalities.

##### 4.3. Image Editing Data

Our VARGPT-v1.1 demonstrates inherent advantages in visual editing capabilities due to its dual capacities in visual understanding and generation, requiring no architectural modifications for this specific task. Our system processes visual editing tasks by feeding input images through the vision encoder while incorporating user editing instructions to produce modified outputs.

For training the visual editing capacities, we construct an instruction tuning dataset using 11,325 samples from Stylebooth [25]4. All images were standardized at 512×512 resolution, with input images processed through our visual understanding preprocessing pipeline and output images handled through the dedicated generation preprocessing protocol.

POPE (acc.)

adv pop rand MME Chameleon (7B) from scratch ✓ 31.1 30.6 25.4 170

Methods LLM Gen. MMBench SEED MMMU

SEEDLLaMA (7B) Vicuna-7B ✓ 45.8 51.5 - - - - Show-o (1.5B) Phi-1.5-1.3B ✓ - - 25.1 - - - 948.4 VILA-U(7B) LLaMA-2-7B ✓ - 59.0 - - - - 1401.8

Liquid (7B) Gemma-7B ✓ - - - - - 81.1 1119.3 Janus-Pro(7B) DeepSeek-LLM × - 72.1 41.00 - - - 1567.1

65.17 69.73 79.67 581.7 InstructBLIP (8B) Vicuna-7B × 36.0 53.4 72.10 82.77 88.57 1212.8

MiniGPT-4 (7B) Vicuna-7B × 23.0 47.4

Qwen-VL (7B) Qwen-7B × 38.2 - - - - Qwen-VL-Chat (7B) Qwen-7B × 60.6 58.2 - - - 1487.5

-

LLaVA (7B) Vicuna-7B × 36.2 33.5 49.70 49.87 50.37 502.8

mPLUG-Owl2 (7B) LLaMA-2-7B × 64.5 57.8 - - - 1450.2 LLaVA-1.5-hf (7B) Vicuna-7B × 62.7 65.4 35.24 83.60 85.77 86.97 1480.1

VARGPT(7B+2B) Vicuna-7B ✓ 67.6 67.9 36.44 84.40 85.90 87.37 1488.8 VARGPT-v1.1(7B+2B) Qwen2-7B ✓ 81.01 76.08 48.56 84.40 88.36 89.17 1684.1

- Table 2. Zero-shot multi-modal evaluation on multi-modal benchmarks including MMMU, MME, MMBench, SEEDBench, and POPE (including different settings random, popular and adversarial ). The overall scores are reported for evaluation and we report test results for MMBench. Gen represents whether the method supports image generation capability. VARGPT-v1.1 achieves the best overall performance.

Methods LLM Gen. GQA TextVQA VQAv2 SciQA-img OKVQA VizWizQA MiniGPT-4 (7B) Vicuna-7B × 43.5 - 0.6 39.6 - InstructBLIP (8B) Vicuna-7B × 49.2 - - 60.5 - 34.5

Qwen-VL (7B) Qwen-7B × 59.3 50.1 - 67.1 - 35.2 Qwen-VL-Chat (7B) Qwen-7B × 57.5 61.5 78.2 68.2 56.6 38.9 mPLUG-Owl2 (7B) LLaMA-2-7B × 56.1 53.3 - 68.7 57.7 54.5 LLaVA-1.5-hf (7B) Vicuna-7B × 61.2 48.8 76.49 67.9 53.2 53.53

VARGPT(7B+2B) Vicuna-7B ✓ 62.3 54.1 78.4 80.1 55.8 56.8 VARGPT-v1.1(7B+2B) Qwen2-7B ✓ 66.2 82.0 80.4 91.6 57.8 65.4

- Table 3. Performance comparison on visual question answering tasks. We gray out the model has trained on the dataset. Gen represents whether the method supports image generation capability.

#### 5. Experiments

Details. For images utilized in generation tasks, we uniformly resize and crop them to 256×256 and 512×512 pixels, subsequently applying preprocessing techniques consistent with those employed in infinity [24, 84]. Regarding images for visual comprehension tasks, we adhere to the preprocessing protocol established in the Qwen2-VL [87] framework. Our language model, visual encoder, and visual feature mapper are initialized using the Qwen2VL-7B-

- 3huggingface . co / datasets / vivym / midjourney -

prompts

- 4https://huggingface.co/datasets/yzwang/X2I-mm-

instruction

Instruct 5 architecture. The visual decoder is initialized with Infinity-2B [24] parameters, encompassing approximately 2 billion model parameters. The visual generation feature mapper in VARGPT-v1.1 undergoes random initialization and is preliminarily updated during the first stage of pretraining. We adopt a multi-scale VQVAE [16] similar to Infinity [24, 84] for image tokenization, facilitating the nextscale prediction paradigm. A comprehensive summary of the training details across the three stages of our model is presented in Table 1. During image generation, the top-k and top-p sampling parameters of our VARGPT-v1.1 model are set to 900 and 0.95, respectively. Additionally, the classifier-

5https : / / huggingface . co / Qwen / Qwen2 - VL - 7B Instruct

Method Global↑ Entity↑ Attribute↑ Relation Other Overall↑

SDv1.5 74.63 74.23 75.39 73.49 67.81 63.18 PixArt-α 74.97 79.32 78.60 82.57 76.96 71.11 Lumina-Next 82.82 88.65 86.44 80.53 81.82 74.63 SDXL 83.27 82.43 80.91 86.76 80.41 74.65 Playground v2.5 83.06 82.59 81.20 84.08 83.50 75.47

###### VARGPT-v1.1(7B+2B) 84.83 82.80 84.95 88.13 87.70 78.59

Table 4. Performance comparison on the DPG-Bench benchmark.

Type Method Single Obj. Two Obj. Counting Colors Position Color Attr. Overall↑

LlamaGen 0.71 0.34 0.21 0.58 0.07 0.04 0.32 LDM 0.92 0.29 0.23 0.70 0.02 0.05 0.37 SDv1.5 0.97 0.38 0.35 0.76 0.04 0.06 0.43 PixArt-α 0.98 0.50 0.44 0.80 0.08 0.07 0.48

Gen. Only SDv2.1 0.98 0.51 0.44 0.85 0.07 0.17 0.50 DALL-E 2 0.94 0.66 0.49 0.77 0.10 0.19 0.52 SEED-X† 0.97 0.58 0.26 0.80 0.19 0.14 0.49 Show-o 0.95 0.52 0.49 0.82 0.11 0.28 0.53

Und. and Gen. LWM 0.93 0.41 0.46 0.79 0.09 0.15 0.47 Chameleon - - - - - - 0.39 VARGPT-v1.1 0.96 0.53 0.48 0.83 0.13 0.21 0.53

Table 5. Performance comparison on the GenEval benchmark.

free guidance (CFG) scale parameter is configured at 1.5.

Benchmarks. Following common settings of VARGPT [51, 52, 106], we evaluate the effectiveness of our VARGPT-v1.1 in visual understanding on a collection of both academic task-oriented benchmarks and recent benchmarks specifically proposed for instruction-following MLLMs, totaling 11 benchmarks:

(1) five multi-modal benchmarks for instructionfollowing MLLMs, including MMbench-dev (en) [55], SEED-bench [38], MMMU [101], POPE [45] and MME [18] benchmarks. For the POPE benchmark, we evaluate on random, popular and adversarial settings with accuracy as metrics; (2) six visual-centric question-answer benchmarks, including GQA [31], TextVQA [72], VQAv2 [22], SciQA-img [58], OKVQA [60] and VizWizQA [23]. For visual understanding benchmarks, we use the settings in lmms-eval [102] to achieve a unified evaluation.

For visual generation assessment, we adopt existing evaluation benchmarks for generative models, utilizing GenEval [21] and DPG-bench [30] 6 as the evaluation frameworks for assessing image generation capabilities. We employ the original prompts provided in these two benchmarks for evaluation purposes, strictly following all settings spec-

6https://github.com/TencentQQGYLab/ELLA

ified in the original benchmarks. For evaluations using the GenEval benchmark, we implement assessments following the prompt rewriting setup described in Infinity [24].

Baselines. For visual understanding, we conducted a comparative analysis of our VARGPT-v1.1 model against other multimodal large language models of similar scale. The comparison included prominent models such as LLaVA1.5 [48], MiniGPT-4 [9], and mPLUG-Owl2 [98], as well as InstructBLIP [12] and Qwen-VL [4]. Furthermore, we extended our comparative study to encompass unified models, including Chameleon [81], SEEDLLaMA [19], Show-o [95], and VILA-U [95]. This comprehensive comparison allowed us to evaluate the performance of VARGPT-v1.1 in relation to a diverse range of state-of-the-art models in the field. For visual generation, we compare VARGPT-v1.1 with existing pure generative models including LlamaGen [76], LDM [69], PixArt-α [8], SDv1.5 [69], SDv2.1 [69], DALL-E2 [68] , Playground v2.5 [41], SDXL [63] and Lumina-Next [108], as well as with state-of-the-art unified models such as SeedX [20], Show-o [95], Chameleon [81], LWM [54].

##### 5.1. Main Results

Evaluation on Multi-modal Benchmarks. We conduct comprehensive evaluations of VARGPT-v1.1 against various state-of-the-art multimodal models, with the results

2025/4/3 17:27 HBBjson0318-3

|1 / 2| | | | | | |
|---|---|---|---|---|---|---|
|[User]: Please render hyperdetailed oil pain portrait style.<br><br>[VARGPT1.1]: Here' hyperdetailed oil pain portrait style.<image_|a designed ting portray<br><br>s a crafted d ting portray gen>|photo of T s a realistic<br><br>drawing of s a realistic|his c frog in a<br><br>This c frog in a|[User weari style, [VAR<br><br>corgi art sty|]: I would like a crafted figure of A cute corgi [User]: Please sketch a designed image of A mother [User]: Can you design a crafted image of A ng a mermaid suit, drawn in colorful line art and baby under a full moon, portrayed with vibrant [User]: I need a sketched scene of A highly detailed, comfortable sweat-shirt with a beautiful flower is on a clear white background. colors and intricate details in a surrealistic style. photorealistic portrait shot of a Clark's nutcracker. illustration. GPT1.1]: Here's a rendered scene of A cute [VARGPT1.1]: Here's a crafted photo of A mother [VARGPT1.1]: Here's a sketched painting of A [VARGPT1.1]: Here's a rendered photo of A wearing a mermaid suit, drawn in colorful line and baby under a full moon, portrayed with vibrant highly detailed, photorealistic portrait shot of a comfortable sweat-shirt with a beautiful flower<br><br>colors and intricate details in a surrealistic style. Clark's nutcracker.<image_gen> le, is on a clear white background.<image_gen> illustration.<image_gen> <image_gen><br><br>[Figure 19]<br><br>[Figure 20]<br><br>[Figure 21]<br><br>[Figure 22]| |
|[User]: Please illustra portrait photograph w volumetric lighting. [VARGPT1.1]: I have<br><br>photoreal portrait pho features and volumetr|ate a scene ith hyper de<br><br>e rendered a tograph wit ic lighting.<|of A photor etailed featu<br><br>a drawing o th hyper det <image_gen|real ures and<br><br>f A tailed n>|[User kingfi the ra [VAR kingfi the ra|]: Can you produce a sketched photo of A isher catching fish in an Indian village during [User]: I need a sketched figure of Two gold stud [User]: Hello, can you generate a photo of A fractal [User]: I would like a rendered figure of A man in iny season. earrings with diamonds on them. flower with purple floral patterns. black shirt standing next to trees. GPT1.1]: Here's a designed figure of A [VARGPT1.1]: Here's a crafted scene of Two gold [VARGPT1.1]: I have designed a drawing of A [VARGPT1.1]: Here's a designed figure of A man in<br><br>isher catching fish in an Indian village during stud earrings with diamonds on them.<image_gen> fractal flower with purple floral patterns<image_gen> black shirt standing next to trees.<image_gen> iny season.<image_gen>| |
|[User]: I need a rende painting featuring Spi [VARGPT1.1]: I have<br><br>watercolor painting fe webs.<image_gen>|ered drawin derman wit e sketched a aturing Spi|g of A wat<br>h border w a photo of A derman wit<br>|ercolor ebs.<br><br>th border|[User album<br><br>a whi [VAR<br><br>powe illustr|]: Can you depict a scene of A power metal [User]: I need a rendered painting of A realistic [User]: I want to see a crafted scene of A cute and [User]: Please craft a visualized figure of A cute<br><br>cover featuring a fantasy-style illustration with image of a Malamute dressed in a Denver Broncos highly detailed 16 bit pixel art sticker with a fluffy unicorn with a unique character design drawn te falcon. NFL uniform, playing football. delightful design. in ink and watercolors. GPT1.1]: Here's a rendered painting of A [VARGPT1.1]: Here's a visualized scene of A [VARGPT1.1]: Here's a generated figure of A cute [VARGPT1.1]: Here's a sketched scene of A cute<br><br>r metal album cover featuring a fantasy-style realistic image of a Malamute dressed in a Denver and highly detailed 16 bit pixel art sticker with a fluffy unicorn with a unique character design drawn ration with a white falcon.<image_gen> Broncos NFL uniform, playing football.<image_gen> delightful design.<image_gen> in ink and watercolors.<image_gen>| |
|[User]: Can you prod beautifully lit mirror w|uce a sketc ith pretty f|hed image flowers in a|of A vase for|[User creep|]: Can you design a picture of A dark and [User]: Can you produce a rendered photo of An up- [User]: Please generate a crafted figure of Unknown [User]: Can you produce a sketched image of A Dark y alleyway with only one light illuminating the close view of the vibrant center of a flower pale rose flowers are blossoming from a pale blue Fantasy Castle situated on a hill with a lake of ice in| |
|Mother's Day, with ex [VARGPT1.1]: Here' beautifully lit mirror w Mother's Day, with ex|treme detai s a designed<br><br>ith pretty f treme detai|il. d photo of A<br><br>flowers in a il.<image_g|A<br><br>vase for en>|hyper [VAR and cr the hy|realistic surroundings. surrounded by green leaves and stems. eggshell. the background. GPT1.1]: I have produced a photo of A dark [VARGPT1.1]: Here's a crafted scene of An up-close [VARGPT1.1]: I have created a painting of Unknown [VARGPT1.1]: I have visualized an image of A Dark<br><br>reepy alleyway with only one light illuminating view of the vibrant center of a flower surrounded by pale rose flowers are blossoming from a pale blue Fantasy Castle situated on a hill with a lake of ice in perrealistic surroundings.<image_gen> green leaves and stems.<image_gen> eggshell.<image_gen> the background.<image_gen>| |

Figure 8. Some generated 512×512 samples by VARGPT-v1.1. Our VARGPT-v1.1 supports text-and-image instructions from user and outputs both text-and-image mixed modal data simultaneously.

presented in Table 2. Based on the experimental results, we have: (1) VARGPT-v1.1 demonstrates superior performance across all benchmarks, achieving state-of-the-art results with

81.01 on MMBench, 76.08 on SEED, and 48.56 on MMMU, surpassing previous models including LLaVA-1.5 [48], InstructBLIP [12] and Janus-pro [10]. (2) In comparison to

10

file:///C:/Users/DELL.DESKTOP-MM67BBG/Downloads/HBBjson0318-3.drawio.html 1/1

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[User]: Make the image <image> look like a Abstract Expressionism.

[VARGPT1.1]: The output image is: <image_gen>

[User]: Make the image <image> look like a Abstract Expressionism.

[VARGPT1.1]: The output image is: <image_gen>

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[User]: Make the image <image> look like a artstyle-constructivist.

[VARGPT1.1]: The output image is: <image_gen>

[User]: Make the image <image> look like a artstyle-constructivist.

[VARGPT1.1]: The output image is: <image_gen>

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[User]: Transform this image <image> to match the style of Neo-Futurism.

[VARGPT1.1]: The output image is: <image_gen>

[User]: <image> Let it in the style of saicomic book.

[VARGPT1.1]: The output image is: <image_gen>

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[User]: Make the image <image> look like a Bauhaus-Style Poster.

[VARGPT1.1]: The output image is: <image_gen>

[User]: Transform this image <image> to match the style of Bauhaus-Style Poster.

[VARGPT1.1]: The output image is: <image_gen>

Figure 9. Some visual display of image editing results from VARGPT-v1.1, with all images having a resolution of 512×512.

generation-capable models such as Chameleon [81] and SEEDLLaMA [19], VARGPT-v1.1 shows remarkable im-

provements across multiple metrics while maintaining generation capabilities. Furthermore, the consistent performance

###### Visual input example:

[Figure 39]

User Please explain the meme in detail.

LLaVA-v1.5-7B The image features a small brown dog lying on the floor, possibly on a wooden surface, with its head up and eyes open. The dog appears to be sleeping or resting, and it is positioned in a way that it seems to be looking up at the camera. The scene is set against a blue background, which adds a contrasting color to the image. The image is accompanied by a caption that reads "Just Monday...", suggesting that the dog is enjoying a lazy day at home. The caption is placed at the bottom of the image, with the dog taking up most of the frame.

VARGPT-v1.1 The meme features a small dog lying on its back on a wooden surface, with the words "MONDAY" and "JUST...MONDAY" written above and below it, respectively. The text humorously implies that the dog is expressing a common sentiment of fatigue or disinterest associated with Mondays, a day often associated with the start of the workweek.

Table 6. VARGPT demonstrates the capability to comprehend and interpret humorous elements within visual content.

improvements on the LLaVA-Bench benchmark, with partial results presented in Table 6, validate the effectiveness of our architectural choices and training strategy, establishing VARGPT-v1.1 as a robust and versatile multimodal model.

Evaluation on Visual Question Answering Benchmarks. We evaluate VARGPT-v1.1’s performance across multiple visual question answering datasets, comparing it with several state-of-the-art multimodal models. The results are presented in Table 3. Our experimental results demonstrate several notable findings: (1) VARGPT-v1.1 achieves superior performance across all VQA benchmarks, showing significant improvements over existing models. Specifically, it achieves 66.2% on GQA, 82.0% on TextVQA, 80.4% on VQAv2, and a remarkable 91.6% on SciQA-img, establishing new state-of-the-art results on these benchmarks. (2) Compared to its predecessor VARGPT (7B+2B), the v1.1 version shows substantial improvements across all metrics, with particularly notable gains in TextVQA (54.1% to 82.0%) and SciQA-img (80.1% to 91.6%), demonstrating the effectiveness of our architectural improvements and training strategy. These comprehensive results validate VARGPT v1.1’s superior capabilities in visual-language understanding across diverse question-answering scenarios.

###### Evaluation on Instruction-to-image Task.

To evaluate the visual generation capabilities of VARGPT,

we conducted comprehensive assessments using the widely adopted GenEval benchmark and DPG-Bench benchmark, with the quantitative results presented in Table 5 and Table 4 respectively. These datasets provide rigorous evaluation frameworks for text-to-image generation capabilities. Our experimental results demonstrate that VARGPT-v1.1 outperforms numerous specialized image-generation models, including diffusion-based architectures (e.g., SDv2.1) and autoregressive approaches (e.g., LlamaGen). Furthermore, when compared with existing unified model baselines, our approach achieves significant improvements in accurate textto-image generation. As visualized in Figure 8, we present representative image outputs and conversational interactions generated by VARGPT-v1.1. Qualitative analysis reveals that VARGPT-v1.1 consistently produces high-quality images that are closely aligned with given textual instructions.

Notably, VARGPT-v1.1 maintains the original VARGPT architecture’s capability to seamlessly integrate textual descriptions and visual generation within a single conversational flow. Compared with the image generation datasets used by other unified models (Show-o: 36M parameters, VILA-U: 15M parameters, Liquid: 30M image parameters), VARGPT-v1.1 achieves superior image generation performance with significantly fewer parameters (8.3M). This remarkable efficiency demonstrates VARGPT-v1.1’s scalability in achieving high-quality visual generation, along with its unique advantages in unifying visual understanding and generation tasks within a single framework.

Visualization of Visual Editing Results. Qualitative evaluation of our visual editing capabilities, as illustrated in Figure9, demonstrates that VARGPT-v1.1 achieves fundamental image manipulation competence. This capability emerges solely through training with visual editing instruction fine-tuning data, without requiring any architectural modifications. Furthermore, these observations substantiate the substantial potential of our unified model architecture in achieving generalized visual understanding, generation, and editing within a single framework.

#### 6. Conclusion, Limitation and Future Work

Conclusion. This work presents VARGPT-v1.1, an advanced extension of our unified visual autoregressive framework VARGPT, which achieves breakthrough performance in visual understanding and generation tasks through synergistic optimization of multi-stage training paradigms and large-scale data utilization. The model innovatively integrates iterative visual instruction tuning with DPO-based reinforcement learning, coupled with progressive resolution scaling strategies that significantly enhance generation fidelity. Notably, the framework realizes image editing capabilities through instruction tuning without structural modifications. These advancements demonstrate VARGPT-v1.1’s scalability through flexible adoption of training strategies

originally developed for MLLMs, while establishing new technical pathways for multimodal system architecture design.

Limitation. (1) There remains a discernible quality gap between VARGPT-v1.1 and certain commercial generative models, such as more advanced models like FLUX [33], which are pre-trained on extensive high-quality image datasets. This disparity is primarily attributable to differences in training data. Scaling on larger datasets and models can alleviate this difference. (2) Although VARGPTv1.1 has shown preliminary performance in visual editing, due to the limited nature of the visual editing dataset, some subtle details in the instructions may not be fully reflected in the generated images and can be used for more complex editing tasks. (3) Since the data used to train our model’s visual editing capabilities comes exclusively from Stylebooth [25], the current editing capabilities are thus limited to style transfer-type visual edits.

Future Work. (1) To achieve superior image generation quality and enable higher-resolution outputs, our technical roadmap includes three strategic initiatives: exploration of novel tokenizer architectures, systematic scaling of model parameters and training datasets, and implementation of largescale hybrid training paradigms. This multi-dimensional approach aims to establish new quality benchmarks while maintaining computational efficiency. (2) For subsequent iterations of VARGPT, we plan to investigate reinforcement learning integration within the autoregressive framework. This research direction specifically targets enhancing generation consistency in multi-turn dialogues and improved instruction adherence through reward modeling techniques.

## VARGPT-v1.1: Improve Visual Autoregressive Large Unified Model via Iterative Instruction Tuning and Reinforcement Learning

### Supplementary Material

#### 7. Qualitative Study for Understanding

To further evaluate our method’s effectiveness in visual understanding similar to VARGPT, we conducted experiments on LLaVA-Bench [47], which consists of 24 distinct images with expert-annotated descriptions and corresponding evaluation questions. In alignment with previous studies [11, 37, 99], we employed this benchmark for qualitative assessment. The visual results are presented in Table 8, 9, 7, 10, and 11.

#### 8. Qualitative Study for Generation

In Figures 10 and 11, we present additional results demonstrating our VARGPT-v1.1’s capability to output mixedmodality data from various datasets similar to VARGPT, as well as a quantitative assessment of the quality of instructiongenerated images. We observe that our approach naturally supports mixed-modal data, seamlessly embedding visual generation tokens within text-generated tokens, while also producing high-quality image content. These findings underscore the significant effectiveness of our VARGPT-v1.1 in unifying understanding and generation tasks.

Visual input example:

[Figure 40]

User What’s the personality of this character? Explain what elements or aspects of the character’s design may have contributed to its popularity.

LLaVA-v1.5-7B The character in the image is a young boy with a distinctive appearance, wearing glasses, a bow tie, and a suit. This unique combination of elements, along with his smiling expression, suggests that the character is friendly, approachable, and confident. The character’s design, which includes the bow tie and glasses, adds a touch of sophistication and charm to his overall look. These elements may have contributed to the character’s popularity, as they make him stand out and create a memorable impression. The character’s personality and design have likely resonated with audiences, making him a beloved and recognizable figure in popular culture.

VARGPT The character’s personality seems to be confident, intelligent, and perhaps a bit mischievous, as suggested by the slight smirk and the way the eyes are directed slightly upwards and to the side, giving an impression of a playful or slightly rebellious attitude. The glasses and bow tie suggest a formal or professional setting, which could imply that the character is a well-educated individual. The design elements that may have contributed to its popularity include the expressive eyes, the confident stance, and the stylish yet formal attire.

VARGPT-v1.1 The character’s personality appears to be that of a clever, observant detective, as suggested by the glasses and the serious expression. The large, expressive eyes and the confident smile could indicate a sense of intelligence and a quick wit. The character’s popularity might be attributed to its distinctive design, which includes the glasses and bow tie, making it easily recognizable and endearing to fans of the series.

Table 7. The case of visual understanding in VARGPT-v1.1 shows that our VARGPT-v1.1 has achieved better understanding performance.

#### References

- [1] laion-coco-aesthetic. https://huggingface.co/ datasets/guangyil/laion-coco-aesthetic. 7
- [2] Qwen2 technical report. 2024. 2, 4

2025/4/3 17:30 HBBjson0318-34

| | | | | |
|---|---|---|---|---|
|[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[User]: Can you depict a visualized scene of This is a cute chibi tooth fairy, depicted as a sweet female character with a kawaii style. [VARGPT1.1]: Here's a crafted figure of This is a<br><br>cute chibi tooth fairy, depicted as a sweet female character with a kawaii style.<image_gen><br><br>[User]: Create a figure of A full biome planet<br><br>enclosed inside a glass bottle, with intricate details of nature and lighting, presented in ultra high resolution 4k concept art.<br><br>[VARGPT1.1]: I have rendered a figure of A full biome planet enclosed inside a glass bottle, with intricate details of nature and lighting, presented in ultra high resolution 4k concept art.<image_gen><br><br>[User]: Can you produce a sketched figure of A big fluffy cloud serves as the bed in a blue sky made of smaller fluffy clouds.<br><br>[VARGPT1.1]: Here's a rendered painting of A big fluffy cloud serves as the bed in a blue sky made of smaller fluffy clouds.<image_gen><br><br>[User]: I want to see a rendered painting of An owl is perched on the branch in the woods. [VARGPT1.1]: Here's a sketched scene of An owl is perched on the branch in the woods.<image_gen><br><br>[User]: Please create a sketched figure of Phillip Fry listening to music in a realistic photo. [VARGPT1.1]: Here's a generated drawing of Phillip<br><br>Fry listening to music in a realistic photo. <image_gen><br><br>[User]: Can you depict a visualized painting of A vintage train boxcar seen from the side in a forest. [VARGPT1.1]: Here's a crafted image of A vintage<br><br>train boxcar seen from the side in a forest. <image_gen><br><br>[User]: Can you create a sketch of A cartoon baby is depicted on a white background, looking happy.<br><br>[VARGPT1.1]: Here's a generated drawing of A cartoon baby is depicted on a white background, looking happy.<image_gen><br><br>[User]: I would like a visualized image of This illustration features Jane Eyre in a watercolor style with pastel colors, depicted in sticker art on a white background. [VARGPT1.1]: Here's a visualized painting of This<br><br>illustration features Jane Eyre in a watercolor style with pastel colors, depicted in sticker art on a white background.<image_gen><br><br>[User]: Can you produce a rendered scene of An empty train compartment displays a minimalist 1960's aesthetic, while the passing countryside outside the window conveys a lonely atmosphere. [VARGPT1.1]: I have designed a figure of An empty<br><br>train compartment displays a minimalist 1960's aesthetic, while the passing countryside outside the window conveys a lonely atmosphere.<image_gen><br><br>[User]: I need a designed image of A castle ruins in a decadent forest, featuring blackish bright red roses and lots of butterflies [VARGPT1.1]: I generated a figure of A castle ruins<br><br>in a decadent forest, featuring blackish bright red roses and lots of butterflies<image_gen><br><br>[User]: Can you generate a sketched drawing of A young Elven Princess surrounded by spirit in a forest setting is depicted in an ultra-realistic watercolour portrait with a fantastic touch. [VARGPT1.1]: Here's a designed figure of A young<br><br>Elven Princess surrounded by spirit in a forest setting is depicted in an ultra-realistic watercolour portrait with a fantastic touch.<image_gen><br><br>[User]: I would like a crafted figure of A woman in a red dress walks down a dark path towards a distant bright future. [VARGPT1.1]: Here's a produced drawing of A<br><br>woman in a red dress walks down a dark path towards a distant bright future.<image_gen><br><br>[User]: Can you produce a sketched figure of A felted, cute and tiny white octopus with a blushing face and a cylindrical mouth. [VARGPT1.1]: The image shows A felted, cute and<br><br>tiny white octopus with a blushing face and a cylindrical mouth.<image_gen><br><br>[Figure 53]<br><br>[User]: Please design a painting of This is an artist's rendering of the front elevation for these traditional house plans. [VARGPT1.1]: I have rendered a scene of This is an<br><br>artist's rendering of the front elevation for these traditional house plans.<image_gen><br><br>[Figure 54]<br><br>[Figure 55]<br><br>[User]: I need a sketched figure of A white and brown ragdoll cat with blue eyes. [VARGPT1.1]: Here's a visualized figure of A white and brown ragdoll cat with blue eyes.<image_gen><br><br>[Figure 56]<br><br>[User]: Can you generate a sketched painting of The gatekeepers of Heaven's Gate stand guard, ensuring only the worthy can enter the divine realm. [VARGPT1.1]: I have generated a scene of The<br><br>gatekeepers of Heaven's Gate stand guard, ensuring only the worthy can enter the divine realm. <image_gen><br><br>[Figure 57]<br><br>[User]: Create a scene of A vibrant and natureinspired lily flower grows out of the streets of Brooklyn, with the iconic Brooklyn Bridge serving as a breathtaking backdrop. [VARGPT1.1]: I have sketched a scene of A vibrant<br><br>and nature-inspired lily flower grows out of the streets of Brooklyn, with the iconic Brooklyn Bridge serving as a breathtaking backdrop.<image_gen><br><br>[Figure 58]<br><br>[User]: I want to see a rendered drawing of Two red cabins sit next to each other on the snow. [VARGPT1.1]: I have generated a scene of Two red<br><br>cabins sit next to each other on the snow. <image_gen><br><br>[Figure 59]<br><br>[User]: I want to see a rendered drawing of A cute chibi anime pirate queen in Japanese style with bright colors. [VARGPT1.1]: Here's a visualized image of A cute<br><br>chibi anime pirate queen in Japanese style with bright colors.<image_gen><br><br>[Figure 60]<br><br>[User]: Can you create a crafted figure of A vector<br><br>illustration of a black rabbit with a cute necklace around its neck, standing on a white background. [VARGPT1.1]: Here's a crafted photo of A vector<br><br>illustration of a black rabbit with a cute necklace around its neck, standing on a white background. <image_gen>| | | | |

Figure 10. Some generated 512×512 samples by VARGPT-v1.1. Our VARGPT-v1.1 supports text-and-image instructions from user and outputs both text-and-image mixed modal data simultaneously.

- [3] Michael S. Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic interpolants, 2023. 3
- [4] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan

15

Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966,

file:///C:/Users/DELL.DESKTOP-MM67BBG/Downloads/HBBjson0318-34.drawio.html 1/1

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[User]: Change this <image> into a the style of Color Field Painting.

[VARGPT1.1]: The output image is: <image_gen>

[User]: <image> Let it in the style of Color Field Painting.

[VARGPT1.1]: The output image is: <image_gen>

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[User]: Make the image <image> look like a Color Field Painting.

[VARGPT1.1]: The output image is: <image_gen>

[User]: Transform this image <image> to match the style of sai-comic book.

[VARGPT1.1]: The output image is: <image_gen>

[User]: Change this <image> into a the style of Color Field Painting.

[VARGPT1.1]: The output image is: <image_gen>

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[User]: Change this <image> into a the style of Graffiti Art.

[VARGPT1.1]: The output image is: <image_gen>

[User]: Make the image <image> look like a futuristic-retro futurism.

[VARGPT1.1]: The output image is: <image_gen>

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[User]: Make the image <image> look like a game-retro arcade.

[VARGPT1.1]: The output image is: <image_gen>

[User]: Transform this image <image> to match the style of Neo-Futurism.

[VARGPT1.1]: The output image is: <image_gen>

Figure 11. Some visual display of image editing results from VARGPT-v1.1, with all images having a resolution of 512×512.

2023. 2, 3, 9

- [5] Yutong Bai, Xinyang Geng, Karttikeya Mangalam, Amir Bar, Alan Yuille, Trevor Darrell, Jitendra Malik, and Alexei A Efros. Sequential modeling enables scalable learning for large vision models, 2023. 3
- [6] Fan Bao, Chongxuan Li, Jun Zhu, and Bo Zhang. Analyticdpm: an analytic estimate of the optimal reverse variance in diffusion probabilistic models, 2022. 3

- [7] Thomas Carta, Clément Romac, Thomas Wolf, Sylvain Lamprier, Olivier Sigaud, and Pierre-Yves Oudeyer. Grounding large language models in interactive environments with online reinforcement learning, 2024. 4
- [8] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis,

[Figure 77]

[Figure 78]

User What are the possible reasons of the formation of this sight?

LLaVA-v1.5-7B The formation of the sight, which is a large, rocky mountain with a lush green hillside and a beach, can be attributed to several factors. Firstly, the mountain’s formation is likely due to tectonic forces, such as the collision or separation of tectonic plates, which have resulted in the formation of the mountain range. Secondly, the lush green hillside and the beach are the result of the local climate and geological processes. The presence of a beach suggests that the area is near a coastline, and the lush green hillside could be due to the presence of a water source, such as a river or a waterfall, which has created a fertile environment for plant growth. Additionally, the mountain’s unique shape and the presence of a road on its side could be the result of human intervention, such as the construction of roads and infrastructure to facilitate transportation and access to the area.

User Discuss how this creative twist on a classic work of art might be interpreted differently by various audiences.

LLaVA-v1.5-7B The creative twist on a classic work of art, featuring a dog wearing a Renaissance-style dress and a beret, can be interpreted differently by various audiences. Some might find it amusing and lighthearted, as it combines the seriousness of a historical painting with the playfulness of a dog dressed up in human attire. Others might view it as a commentary on the human-animal relationship, suggesting that dogs can be just as expressive and creative as humans. Additionally, some might perceive it as a critique of the traditional art world, as it challenges the norms of what is considered ärtby¨ incorporating a non-human subject and a more contemporary, humorous element. Overall, the interpretation of this creative twist on a classic work of art can vary greatly depending on the viewer’s perspective and personal experiences.

VARGPT The sight of a large mountain with a road running through it and a beach nearby is a result of geological processes that have shaped the landscape over time. Some possible reasons for the formation of this sight include:1. Tectonic forces: The movement of the Earth’s crust can lead to the formation of mountains, valleys, and other geological features. The mountain could have been formed through tectonic forces such as the collision of two tectonic plates or the uplift of the Earth’s crust.2. Weathering and erosion: The mountain could have been shaped by the forces of weathering and erosion, which break down the rock and soil, leaving behind a rugged landscape. The road running through the mountain could be a result of human intervention to connect different areas or to facilitate transportation.3. Coastal erosion: The beach and the surrounding area could have been shaped by the forces of coastal erosion, which is the process of the sea wearing away the land. The beach could have been formed by the deposition of sediments from the sea, while the surrounding area could have been shaped by the erosion of rock and soil.4. Climate and weather patterns: The vegetation on the mountain and the surrounding area could be influenced by the local climate and weather patterns, which can affect the growth and distribution of plants and animals in the region.

VARGPT This creative twist on a classic work of art, where a dog is dressed in a Mona Lisa-style outfit and painted in a similar style, could be interpreted in various ways by different audiences. Some might find it amusing and lighthearted, as it plays on the juxtaposition of a serious artwork with a playful, everyday subject. Others might appreciate the skillful use of artistic techniques to transform a simple image of a dog into a piece of art that can be appreciated on multiple levels. Some might even see it as a commentary on the universal appeal of art, which can be appreciated by both humans and animals. The interpretation would depend on the viewer’s personal experiences, artistic background, and sense of humor.

VARGPT-v1.1 This creative twist on a classic work of art, featuring a dog with a human-like face, can be interpreted differently by various audiences. Some may find it humorous and entertaining, appreciating the cleverness of the artist in blending two distinct elements. Others might view it as a commentary on the nature of identity and the human-animal relationship, suggesting that even animals can possess human-like qualities or emotions. Some may also see it as a form of modern art, exploring the boundaries between different genres and challenging traditional notions of beauty and representation. Ultimately, the interpretation of this artwork will vary based on individual perspectives and cultural backgrounds.

VARGPT-v1.1 The sight in the image is a volcanic crater, which is formed by the eruption of a volcano. Volcanic eruptions can create craters of varying sizes and shapes, depending on the type of eruption and the materials involved. The crater in the image is likely the result of a past volcanic eruption that excavated a large cavity in the Earth’s surface. Over time, the crater may have been shaped by erosion and weathering processes, and it may now be surrounded by vegetation and other geological features. The presence of the road and buildings around the crater suggests that it is a location of interest, possibly a tourist attraction or a natural landmark.

Table 8. The case of visual understanding in VARGPT-v1.1 shows that our VARGPT-v1.1 has achieved better understanding performance.

Table 9. The case of visual understanding in VARGPT-v1.1 shows that our VARGPT-v1.1 has achieved better understanding performance.

[Figure 79]

[Figure 80]

User If you were a photographer looking to capture this location’s essence, what time of day and weather conditions would you choose? Describe the reasons behind your choice.

User If someone were to recommend a new flavor or topping to the dish, describe the reason for this change and how it might alter the overall taste.

LLaVA-v1.5-7B To capture the essence of this location, which features a pier extending over a lake with mountains in the background, I would choose to photograph during the golden hour, which is the time just after sunrise or before sunset. This time of day provides soft, warm light that enhances the natural beauty of the scene, creating a serene and picturesque atmosphere. Additionally, the golden hour often results in softer shadows and more dramatic lighting, which can accentuate the pier’s structure and the surrounding landscape.As for the weather conditions, I would prefer a clear day with a gentle breeze, as it would provide a pleasant and comfortable environment for both the photographer and the subjects. A clear day with good visibility would allow for better composition and capture the stunning mountain backdrop in its full glory. A gentle breeze would also add a touch of dynamism to the scene, making it more visually engaging and interesting.

LLaVA-v1.5-7B If someone were to recommend a new flavor or topping to the dish, they might suggest adding a spicy element, such as chili flakes or crushed red pepper, to enhance the taste and make it more flavorful. The current dish consists of a bowl of noodles with chicken, vegetables, and sauce, which is already a delicious and hearty meal. However, adding a spicy component could elevate the dish by providing a contrasting flavor and a kick of heat, making it more exciting and satisfying for those who enjoy spicy foods. This change would alter the overall taste by introducing a new dimension of flavor and complexity to the dish, making it more enjoyable and memorable for the person trying it.

VARGPT If someone were to recommend a new flavor or topping to the dish, it could be due to a desire to add a new taste dimension or to cater to personal preferences. For example, they might suggest adding a slice of avocado to the dish, which would introduce a creamy, buttery texture and a mild, nutty flavor that complements the richness of the noodles and the savory broth. This addition would alter the overall taste by providing a contrasting texture and a new layer of flavor complexity, enhancing the dish’s overall appeal and making it more satisfying to eat.

VARGPT I would choose to capture this location’s essence during the early morning, specifically during the golden hour, which is the period shortly after sunrise or before sunset. This time of day is known for its soft, warm light and long shadows, which can enhance the tranquility and beauty of the scene. The weather conditions would be clear and calm, with a gentle breeze to create a sense of movement in the water and to soften the reflections on the water’s surface. The overcast sky would add a sense of depth and contrast to the scene, while the mist or fog would create a serene and ethereal atmosphere.

VARGPT-v1.1 If someone were to recommend adding a new flavor or topping to the dish, it could be a type of meat, such as chicken or beef, to add more protein and depth to the broth. This would alter the overall taste by introducing a new flavor profile and texture, potentially making the dish more savory and satisfying.

VARGPT-v1.1 I would choose to capture this location during the early morning or late afternoon, when the light is soft and the sky is clear, to create a serene and tranquil atmosphere. The calm water would reflect the light beautifully, enhancing the overall composition of the image. The weather should be clear with minimal cloud cover to ensure the best visibility of the distant mountains and the lush greenery. This would allow for a crisp and vibrant representation of the landscape, with the wooden dock and the lake’s surface acting as natural elements that draw the viewer’s eye towards the distant scenery.

Table 11. The case of visual understanding in VARGPT-v1.1 shows that our VARGPT-v1.1 has achieved better understanding performance.

2023. 9

- [9] Jun Chen, Deyao Zhu, Xiaoqian Shen, Xiang Li, Zechun Liu, Pengchuan Zhang, Raghuraman Krishnamoorthi, Vikas Chandra, Yunyang Xiong, and Mohamed Elhoseiny. Minigpt-v2: large language model as a unified interface for vision-language multi-task learning. ArXiv, abs/2310.09478,

2023. 2, 9

- [10] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Januspro: Unified multimodal understanding and generation with

Table 10. The case of visual understanding in VARGPT-v1.1 shows that our VARGPT-v1.1 has achieved better understanding performance.

- data and model scaling, 2025. 2, 4, 7, 10
- [11] Zhaorun Chen, Zhaorun Chen, Zhuokai Zhao, Hongyin Luo, Huaxiu Yao, Bo Li, and Jiawei Zhou. Halc: Object hallucination reduction via adaptive focal-contrast decoding. ArXiv, abs/2403.00425, 2024. 14
- [12] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning,

2023. 2, 3, 9, 10

- [13] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2009. 2
- [14] Ming Ding, Zhuoyi Yang, Wenyi Hong, Wendi Zheng, Chang Zhou, Da Yin, Junyang Lin, Xu Zou, Zhou Shao, Hongxia Yang, and Jie Tang. Cogview: Mastering text-toimage generation via transformers, 2021. 3
- [15] Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, Xiangwen Kong, Xiangyu Zhang, Kaisheng Ma, and Li Yi. Dreamllm: Synergistic multimodal comprehension and creation, 2024. 3
- [16] Patrick Esser, Robin Rombach, and Björn Ommer. Taming transformers for high-resolution image synthesis, 2021. 3, 8
- [17] Lijie Fan, Tianhong Li, Siyang Qin, Yuanzhen Li, Chen Sun, Michael Rubinstein, Deqing Sun, Kaiming He, and Yonglong Tian. Fluid: Scaling autoregressive text-to-image generative models with continuous tokens. arXiv preprint arXiv:2410.13863, 2024. 3
- [18] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models, 2024. 9
- [19] Yuying Ge, Sijie Zhao, Ziyun Zeng, Yixiao Ge, Chen Li, Xintao Wang, and Ying Shan. Making llama see and draw with seed tokenizer, 2023. 3, 9, 11
- [20] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396,

2024. 2, 9

- [21] Dhruba Ghosh, Hanna Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment, 2023. 9
- [22] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering, 2017. 9
- [23] Danna Gurari, Qing Li, Abigale J. Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P. Bigham. Vizwiz grand challenge: Answering visual questions from blind people, 2018. 9
- [24] Jian Han, Jinlai Liu, Yi Jiang, Bin Yan, Yuqi Zhang, Zehuan Yuan, Bingyue Peng, and Xiaobing Liu. Infinity: Scaling bitwise autoregressive modeling for high-resolution image synthesis, 2024. 2, 5, 8, 9

- [25] Zhen Han, Chaojie Mao, Zeyinzi Jiang, Yulin Pan, and Jingfeng Zhang. Stylebooth: Image style editing with multimodal instruction, 2024. 6, 7, 13
- [26] Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo Jun, Tom B Brown, Prafulla Dhariwal, Scott Gray, et al. Scaling laws for autoregressive generative modeling. arXiv preprint arXiv:2010.14701, 2020. 3
- [27] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2
- [28] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models, 2020. 3
- [29] Jonathan Ho, Chitwan Saharia, William Chan, David J. Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation, 2021. 3
- [30] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment, 2024. 9
- [31] Drew A. Hudson and Christopher D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering, 2019. 9
- [32] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361,

2020. 3

- [33] Black Forest Labs. Flux. https://github.com/

- black-forest-labs/flux, 2023. 13

[34] Black Forest Labs. Flux. https://github.com/

- black-forest-labs/flux, 2024. 2

- [35] Yann LeCun, Yoshua Bengio, and Geoffrey Hinton. Deep learning. nature, 521(7553):436–444, 2015. 3
- [36] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization, 2022. 3
- [37] Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Li Bing. Mitigating object hallucinations in large vision-language models through visual contrastive decoding. ArXiv, abs/2311.16922, 2023. 14
- [38] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension, 2023. 9
- [39] Bo Li, Hao Zhang, Kaichen Zhang, Dong Guo, Yuanhan Zhang, Renrui Zhang, Feng Li, Ziwei Liu, and Chunyuan Li. Llava-next: What else influences visual instruction tuning beyond data?, 2024. 3, 6
- [40] Bo Li, Kaichen Zhang, Hao Zhang, Dong Guo, Renrui Zhang, Feng Li, Yuanhan Zhang, Ziwei Liu, and Chunyuan Li. Llava-next: Stronger llms supercharge multimodal capabilities in the wild, 2024. 3, 6
- [41] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2.5: Three insights towards enhancing aesthetic quality in text-to-image generation, 2024. 9

- [42] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next: Tackling multi-image, video, and 3d in large multimodal models,

2024. 3, 6

- [43] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR, 2023. 3
- [44] Xuefeng Li, Haoyang Zou, and Pengfei Liu. Limr: Less is more for rl scaling, 2025. 4
- [45] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large visionlanguage models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 292–305, Singapore, 2023. Association for Computational Linguistics. 9
- [46] Zijie Li, Henry Li, Yichun Shi, Amir Barati Farimani, Yuval Kluger, Linjie Yang, and Peng Wang. Dual diffusion for unified image generation and understanding, 2024. 2, 3
- [47] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. ArXiv, abs/2310.03744, 2023. 14
- [48] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. ArXiv, abs/2304.08485, 2023. 2, 9, 10
- [49] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 3
- [50] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024. 3
- [51] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning, 2024. 6, 9
- [52] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 3, 6, 9
- [53] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with ringattention. arXiv preprint, 2024. 2, 3
- [54] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with blockwise ringattention, 2025. 9
- [55] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player?, 2024. 9
- [56] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning, 2025. 4
- [57] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps, 2022. 3
- [58] Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and

- Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering, 2022. 9
- [59] Chuofan Ma, Yi Jiang, Junfeng Wu, Jihan Yang, Xin Yu, Zehuan Yuan, Bingyue Peng, and Xiaojuan Qi. Unitok: A unified tokenizer for visual generation and understanding,

2025. 3

- [60] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge, 2019. 9
- [61] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International conference on machine learning, pages 8162–8171. PMLR,

2021. 2

- [62] Junting Pan, Keqiang Sun, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, Jifeng Dai, Yu Qiao, and Hongsheng Li. Journeydb: A benchmark for generative image understanding,

2023. 2, 7

- [63] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis, 2023. 3, 9
- [64] Liao Qu, Huichao Zhang, Yiheng Liu, Xu Wang, Yi Jiang, Yiming Gao, Hu Ye, Daniel K. Du, Zehuan Yuan, and Xinglong Wu. Tokenflow: Unified image tokenizer for multimodal understanding and generation, 2024. 2
- [65] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 4
- [66] Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model, 2024. 2, 6
- [67] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation, 2021. 3
- [68] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents, 2022. 9
- [69] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models, 2022. 9
- [70] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. Laion-5b: An open large-scale dataset for training next generation image-text models, 2022. 2
- [71] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. 4
- [72] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read, 2019. 9

- [73] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models, 2022. 3
- [74] Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution, 2020. 3
- [75] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 2, 3
- [76] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation, 2024. 3, 9
- [77] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Emu: Generative pretraining in multimodality, 2024. 3
- [78] Haotian Tang, Yecheng Wu, Shang Yang, Enze Xie, Junsong Chen, Junyu Chen, Zhuoyang Zhang, Han Cai, Yao Lu, and Song Han. Hart: Efficient visual generation with hybrid autoregressive transformer. arXiv preprint, 2024. 2, 3
- [79] Zineng Tang, Ziyi Yang, Chenguang Zhu, Michael Zeng, and Mohit Bansal. Any-to-any generation via composable diffusion, 2023. 3
- [80] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 2
- [81] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models, 2024. 3, 9, 11
- [82] DeepSeek-AI Team. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. 4
- [83] Gemini Team. Gemini: A family of highly capable multimodal models, 2024. 3
- [84] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. 2024. 2, 3, 5, 8
- [85] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, T. Lacroix, Baptiste R., Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 3
- [86] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 3
- [87] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 4, 8
- [88] Peiyi Wang, Lei Li, Zhihong Shao, R. X. Xu, Damai Dai, Yifei Li, Deli Chen, Y. Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations, 2024. 4

- [89] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 2, 3
- [90] Yi Wang, Mushui Liu, Wanggui He, Longxiang Zhang, Ziwei Huang, Guanghao Zhang, Fangxun Shu, Zhong Tao, Dong She, Zhelun Yu, Haoyuan Li, Weilong Dai, Mingli Song, Jie Song, and Hao Jiang. Mint: Multi-modal chain of thought in unified generative models for enhanced image generation, 2025. 3
- [91] Chengyue Wu, Xiaokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, Chong Ruan, and Ping Luo. Janus: Decoupling visual encoding for unified multimodal understanding and generation, 2024. 2, 3
- [92] Junfeng Wu, Yi Jiang, Chuofan Ma, Yuliang Liu, Hengshuang Zhao, Zehuan Yuan, Song Bai, and Xiang Bai. Liquid: Language models are scalable multi-modal generators. arXiv preprint arXiv:2412.04332, 2024. 2, 3
- [93] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal llm. arXiv preprint arXiv:2309.05519, 2023. 2, 3
- [94] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, Song Han, and Yao Lu. Vila-u: a unified foundation model integrating visual understanding and generation,

2025. 3

- [95] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024. 2, 9
- [96] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation, 2024. 3
- [97] Hanrong Ye, De-An Huang, Yao Lu, Zhiding Yu, Wei Ping, Andrew Tao, Jan Kautz, Song Han, Dan Xu, Pavlo Molchanov, and Hongxu Yin. X-vila: Cross-modality alignment for large language model, 2024. 3
- [98] Qinghao Ye, Haiyang Xu, Jiabo Ye, Mingshi Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl2: Revolutionizing multi-modal large language model with modality collaboration. ArXiv, abs/2311.04257, 2023. 2, 9
- [99] Shukang Yin, Chaoyou Fu, Sirui Zhao, Tong Xu, Hao Wang, Dianbo Sui, Yunhang Shen, Ke Li, Xingguo Sun, and Enhong Chen. Woodpecker: Hallucination correction for multimodal large language models. ArXiv, abs/2310.16045, 2023. 14
- [100] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. Scaling autoregressive models for content-rich text-to-image generation, 2022. 3

- [101] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi, 2024. 9
- [102] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Lmmseval: Reality check on the evaluation of large multimodal models, 2024. 9
- [103] Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llavanext: A strong zero-shot video understanding model, 2024. 3, 6
- [104] Yi-Fan Zhang, Tao Yu, Haochen Tian, Chaoyou Fu, Peiyan Li, Jianshu Zeng, Wulin Xie, Yang Shi, Huanyu Zhang, Junkang Wu, Xue Wang, Yibo Hu, Bin Wen, Fan Yang, Zhang Zhang, Tingting Gao, Di Zhang, Liang Wang, Rong Jin, and Tieniu Tan. Mm-rlhf: The next step forward in multimodal llm alignment, 2025. 4
- [105] Yue Zhao, Fuzhao Xue, Scott Reed, Linxi Fan, Yuke Zhu, Jan Kautz, Zhiding Yu, Philipp Krähenbühl, and De-An Huang. Qlip: Text-aligned visual tokenization unifies autoregressive multimodal understanding and generation, 2025. 3
- [106] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 9
- [107] Xianwei Zhuang, Yuxin Xie, Yufan Deng, Liming Liang, Jinghan Ru, Yuguo Yin, and Yuexian Zou. Vargpt: Unified understanding and generation in a visual autoregressive multimodal large language model, 2025. 2, 4
- [108] Le Zhuo, Ruoyi Du, Han Xiao, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, Lirui Zhao, Fu-Yun Wang, Zhanyu Ma, Xu Luo, Zehan Wang, Kaipeng Zhang, Xiangyang Zhu, Si Liu, Xiangyu Yue, Dingning Liu, Wanli Ouyang, Ziwei Liu, Yu Qiao, Hongsheng Li, and Peng Gao. Lumina-next: Making lumina-t2x stronger and faster with next-dit, 2024. 9

