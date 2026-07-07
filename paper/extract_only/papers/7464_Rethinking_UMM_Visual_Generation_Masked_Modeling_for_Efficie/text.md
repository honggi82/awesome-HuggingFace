# arXiv:2603.16139v1[cs.CV]17Mar2026

## Rethinking UMM Visual Generation: Masked Modeling for Efficient Image-Only Pre-training

Peng Sun1,3,* Jun Xie1,2,3,∗ Tao Lin3,†

1Zhejiang University 2Shanghai Innovation Institute 3Westlake University

sunpeng@westlake.edu.cn, junxiecs@zju.edu.cn, lintao@westlake.edu.cn

### Abstract

Unified Multimodal Models (UMMs) are often constrained by the pre-training of their visual generation components, which typically relies on inefficient paradigms and scarce, high-quality text-image paired data. In this paper, we systematically analyze pre-training recipes for UMM visual generation and identify these two issues as the major bottlenecks. To address them, we propose Image-Only Training for UMMs (IOMM), a data-efficient two-stage training framework. The first stage pre-trains the visual generative component exclusively using abundant unlabeled imageonly data, thereby removing the dependency on paired data for this costly phase. The second stage fine-tunes the model using a mixture of unlabeled images and a small curated set of text-image pairs, leading to improved instruction alignment and generative quality. Extensive experiments show that IOMM not only improves training efficiency but also achieves state-of-the-art (SOTA) performance. For example, our IOMM-B (3.6B) model was trained from scratch using only ∼1050 H800 GPU hours (with the vast majority, 1000 hours, dedicated to the efficient image-only pretraining stage). It achieves 0.89 on GenEval and 0.55 on WISE—surpassing strong baselines such as BAGEL-7B (0.82 & 0.55) and BLIP3-o-4B (0.84 & 0.50). Code is available https://github.com/LINs-lab/IOMM.

### 1. Introduction

Unifying deep semantic understanding with rich perceptual generation in a single model is a grand challenge in AI. These UMMs promise a synergy where comprehension and generation mutually enhance one another, unlocking applications from nuanced, dialogue-based image editing to context-aware content creation [16, 17, 37]. While recent UMMs demonstrate impressive generative capabili-

*Equal Contritbution †Corresponding author.

ties [9, 13, 38, 49], their development is often hampered by significant practical constraints.

However, current UMM training paradigms rely on vast, often proprietary, text-image datasets [9]. The prohibitive cost of curating this data impedes open and reproducible research. Moreover, the training procedures are notoriously inefficient, demanding immense computational resources. This raises a critical question: Can we develop a more dataand compute-efficient training paradigm for UMMs that reduces reliance on paired data while improving performance?

In this work, we address this question by deconstructing the pre-training of UMMs’ visual generative components. Our analysis reveals two primary bottlenecks: the dependency on scarce text-image pairs and the inefficiency of prevailing training objectives. We observe that many UMMs, particularly when fine-tuned on limited data, struggle to generate images that faithfully align with textual prompts. As shown in Fig. 7a, even a strong baseline like QwenImage [49] can produce outputs that lack detail and fidelity to the input prompt.

To surmount these limitations, we introduce IOMM, a novel, data-efficient two-stage training paradigm for constructing and refining UMMs. Our approach commences with an unsupervised pre-training phase that leverages unlabeled, image-only data, followed by a fine-tuning stage that employs a strategic mixture of image-only and high-quality paired data. This paradigm, as we empirically demonstrate, not only mitigates the reliance on paired data but also yields superior generative quality and instruction-following capabilities. In summary, our contributions are threefold:

(a) We introduce IOMM, a data- and compute-efficient framework built upon two key technical innovations: (1) a novel residual query adapter that efficiently adapts frozen Multimodal Large Language Models (MLLMs) for generative tasks with minimal parameter overhead, and (2) a masked image modeling objective that fosters a robust visual prior by framing pre-training as a sparseto-dense reconstruction task.

[Figure 1]

(a) Multi-resolution visualizations from our IOMM-XL.

|0.88<br><br>0.86<br><br>0.78<br><br>0.89<br><br>0.87<br><br>0.61<br><br>0.50 0.70 0.90<br><br>Pair + Mix<br><br>Pair + Pair<br><br>Pair + Image<br><br>Image + Mix<br><br>Image + Pair<br><br>Image + Image|
|---|

Pretrain Finetune Image Image

Pretrain Finetune Pair Image

- (1)
- (2)
- (3)

- (4)
- (5)
- (6)

Image Pair

Pair Pair

Image Mix

Pair Mix

(b) Overview of training recipes.

(c) GenEval performance comparison.

- Figure 1. An overview and validation of our proposed training paradigm. (a) Visual results of our IOMM-XL, demonstrating high-quality, multi-resolution image synthesis. Corresponding prompts are provided in App. C.7. (b) An illustration of the six training recipes we investigate. (c) Quantitative results of six training recipes on the GenEval benchmark.

(b) We present a systematic analysis of six distinct training recipes for UMMs, exploring various combinations of image-only, text-image pair, and mixed data across pretraining and fine-tuning. Under our framework IOMM, our central finding is that a two-stage paradigm—pretraining on image-only data followed by fine-tuning on a mixed dataset1—yields best performance (Fig. 1c).

1Concurrent work [55] explores a similar fine-tuning strategy on mixed

(c) Extensive experiments validate the efficacy and efficiency of IOMM. Our resulting models attain SOTA or comparable performance across diverse benchmarks, all while operating with substantially greater data and

data, but differs crucially: (1) they focus only on fine-tuning, while we study both pre-training and fine-tuning; (2) they use standard reconstruction, whereas we use masked image modeling; (3) they test on smaller models (e.g., BAGEL-7B), while we validate on both small and large-scale UMMs (e.g., Qwen-Image-20B).

compute efficiency (see Sec. 4).

Additionally, we establish that our proposed mixed-data finetuning strategy is a generalizable and effective technique for enhancing the instruction-following fidelity and image generation quality of existing powerful UMMs, which we validate on diverse models including Qwen-Image (Sec. 4.3).

### 2. Related Work

Text-to-image diffusion models. The field of text-toimage synthesis has seen rapid advancements, driven by innovations in diffusion model architectures and training methodologies. Foundational works, such as the initial Stable Diffusion series [40, 42], established the Latent Diffusion Model (LDM) as a dominant paradigm. A significant architectural evolution arrived with Stable Diffusion 3 [14], which introduced the Multimodal Diffusion Transformer (MM-DiT). This architecture employs separate transformerbased pathways to process image and text representations independently before fusing them, markedly improving textimage alignment. Following a similar design philosophy, FLUX.1 [25] also utilizes a dual-stream transformer architecture to enhance modality-specific encoding.

Concurrently, a parallel line of research has focused on optimizing training efficiency and data curation. For example, PixArt-α/σ [6, 7] demonstrated the ability to achieve SOTA performance with substantially reduced training costs. Similarly, Playground v2/v2.5 [26, 27] is distinguished by its high aesthetic quality, a result of meticulous data filtering and reinforcement learning from user preferences. More recent models, including SANA [54] and SANA-sprint [10], continue this trajectory, pushing the boundaries of performance through further architectural and training refinements. Notably, Lumos-T2I [33] presents a paradigm shift by demonstrating that high-quality text-to-image generation can be achieved through image-only pre-training, challenging the conventional reliance on paired text-image datasets.

However, these models are specialized for unidirectional text-to-image generation. They lack the inherent capacity for multimodal understanding, which precludes their direct application to complex, interactive tasks such as dialoguebased image editing [16, 49] that require a seamless blend of comprehension and generation.

Unified understanding and generation models. The pursuit of models that unify multimodal understanding and generation has led to two primary training paradigms: training end-to-end from scratch, and building upon pre-trained foundation models. Among those trained from scratch are Chameleon [45], Show-o [56], VILA-U [52], Janus [48],

JanusPro [11], JanusFlow [34], Transfusion [66], and Harmon [51]. These systems employ diverse architectures, including autoregressive (AR) and masked autoregressive (MAR) frameworks, to jointly handle both modalities.

The second paradigm leverages pre-trained components, integrating powerful Multimodal Large Language Models (MLLMs) with established diffusion backbones. Notable examples include DreamLLM [13], MetaQueries [38], BLIP3-

- o [9], UniWorld-V1 [28], Qwen-Image [49], and Bagel [12]. These approaches typically bridge the frozen MLLM and diffusion model using mechanisms like learnable queries
- or multi-stage training protocols [38] to harmonize understanding and generative processes. The resulting synergy of generation and comprehension enables these unified models to tackle a wide spectrum of tasks, including high-fidelity, instruction-guided image editing [16, 17].

Concurrently, UAE [58] and ViLex [47] explore modeling UMMs as auto-encoding tasks, which involve reconstructing the input image itself for improving understanding and generation in UMMs.

Despite these significant advances, a fundamental limitation persists across existing unified models. Current training paradigms depend heavily on meticulously curated, largescale datasets of high-quality image-text pairs to train their generative modules. This reliance on proprietary or difficultto-acquire data poses a significant barrier to open research and broader community-driven development.

Masked signal modeling. Masked signal modeling, pioneered by Masked Autoencoders (MAE) [19], has become a powerful self-supervised learning paradigm. The core principle involves training a model to learn robust representations by reconstructing randomly masked portions of an input signal. Initially applied to images, this “mask-andpredict” strategy has been successfully adapted to a diverse range of generative tasks. Notable adaptations include predicting masked visual tokens for non-autoregressive image synthesis [5], masking textual conditions to refine guidance in diffusion models [67], leveraging attention mechanisms to generate precise editing masks from user intent [69], and improving the data efficiency of Generative Adversarial Network (GAN) training [22]. The versatility of this approach underscores its potential as a flexible and potent tool for representation learning and generative modeling.

### 3. Methodology

We propose a novel framework for pre-training a generative model by leveraging a frozen Multimodal Large Language Model (MLLM) with an image-only dataset (see Sec. 3.2), entirely eschewing the need for paired text.

|| | |
|---|---|
| | |
| |0.88|
| |0.82<br><br>+0.38|
| | |
| | |
| | |
<br><br>0.44<br><br>0.40<br><br>0.45<br><br>0.50<br><br>0.55<br><br>0.60<br><br>0.65<br><br>0.70<br><br>0.75<br><br>0.80<br><br>0.85<br><br>0.90<br><br>Raw ⊕Residual Query Adapter<br><br>⊕Mask Image<br><br>+0.06<br><br>|
|---|

| |
|---|

| |
|---|

| |
|---|

[BOI] Token Text Token

[EOI] Token Image Token

Diffusion-based

Image

[Figure 2]

Generation Model

| |
|---|

| |
|---|

Masked Image Token

| | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | |[Figure 3]| | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

Other Output Gen. Condition

[Figure 4]

MLLM

[Figure 5]

[Figure 6]

Concatenate

[Figure 7]

[Figure 8]

[Figure 9]

Res. Query Adapter

Multimodal Input

[Figure 10]

|𝑡1|
|---|

|𝑡2|
|---|

|𝑡3|
|---|

|𝑡4|
|---|

|B.|
|---|

|𝑣1|
|---|

|𝑣2|
|---|

|𝑣3|
|---|

|𝑣4|
|---|

|E.|
|---|

MLLM Attn. Mask

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

[Figure 11]

[Figure 12]

|[Figure 13]| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|

Vision Transformer

Text Tokenizer

Image

Cross Attn. Key Padding Mask

Text: “The same as:”

(a) The architecture of our image-only pre-training stage.

(b) Component ablation study.

- Figure 2. Visualization of the IOMM framework. (a) The architecture of our proposed framework. (b) Ablation study demonstrating the effectiveness of architectural design choices, confirming that each component contributes positively to the final GenEval score. All variants utilize the same IOMM-XL architecture.

Our approach hinges on two key contributions. First, to adapt the MLLM’s representations for the generative task without costly fine-tuning, we introduce the Residual Query Adapter (see Sec. 3.3), a lightweight, parameter-efficient module that refines the visual condition. Second, to prevent the self-conditioning from collapsing to a trivial identity mapping, we employ a Masked Image Modeling strategy (see Sec. 3.4). This transforms training into a sparse-to-dense reconstruction task, compelling the model to learn a robust and compositional visual prior.

- 3.1. Preliminaries on Diffusion Models

###### 3.2. Image-Only Pre-training via Self-Conditioning

We hypothesize that explicit text is merely one possible modality for conveying the high-level semantic information necessary to guide image synthesis. The rich semantic content inherent in an image can itself serve as a sufficient conditioning signal. This principle allows us to design a training paradigm that relies exclusively on an unlabeled image corpus.

Our framework utilizes a pre-trained and frozen MLLM, which we denote as g. This MLLM includes a Vision Transformer (ViT) encoder, v, for processing visual inputs. To generate an image x, we first derive a conditioning signal directly from x.

Diffusion-based generative models transform a simple prior distribution, e.g., a standard Gaussian N(0,I), into a complex data distribution by learning to reverse a predefined noise-corruption process. In this paper, we focus on flow matching (FM) models [29], which have demonstrated strong performance in image generation [44, 54].

Forming the self-conditioning signal. Inspired by instruction-following models, we construct the initial condition by combining a generic, fixed textual prompt with the visual features of the image. Let caux ∈ RT×D be the token embeddings for an auxiliary prompt, such as “Generate an image that is identical to the reference image:”. The ViT encoder v processes the image x into a sequence of patch embeddings, cimg = v(x) ∈ RP

Flow matching models define a deterministic path from a data point x to a noise vector z ∼ N(0,I) via the interpolation xt = (1 − t) · x + t · z for t ∈ [0,1]. A neural network Fθ(xt,t,c) is then trained to learn the constant-velocity vector field z − x of this path. Formally, given a conditioning signal c, the objective is: L(θ) =

2×D, where P2 is the number of patches and D is the embedding dimension.

Ex,z,c,t ∥Fθ(xt,t,c) − (z − x)∥22 .

The complete conditioning sequence c is formed by concatenating these two components: c = concat(caux,cimg) ∈ R(T+P

For generation, one starts with a sample from the prior, x1 ∼ N(0,I), and integrates the learned vector field backward in time from t = 1 to t = 0. This is achieved by solving the probability flow ordinary differential equation (PF-ODE) [43]: dxt

2)×D. This sequence is then processed by the frozen MLLM g to produce the final latent condition h = g(c), which is used to guide the diffusion model Fθ.

dt = Fθ(xt,t,c). The solution at t = 0 yields the final generated sample x0.

###### 3.3. Residual Query Adapter

Directly using the output of a frozen MLLM, g(c), as a condition for the diffusion model yields suboptimal performance (see “Raw” in Fig. 2b). We attribute this to a domain mismatch: representations from an MLLM pre-trained for understanding-based tasks are not inherently optimized for the nuanced control required by a generative process.

While fine-tuning the entire MLLM (g) could in principle align its representations, this approach is fraught with two major challenges:

- (a) the immense computational cost associated with billions of parameters, where e.g. the MLLM in MetaQuery-XL has 7B parameters, versus 0.6B for the diffusion model [38].
- (b) the risk of catastrophic forgetting, where the powerful, pre-trained capabilities of the MLLM are degraded when fine-tuned on an image-only reconstruction task.

To circumvent these issues, we introduce the Residual Query Adapter (RQA), denoted qθ. The RQA is a lightweight (with only 29M parameters), trainable adapter module designed to preprocess the conditioning signal c before it enters the MLLM. Specifically, the RQA uses cross-attention [46] with 256 learned query tokens that learns a task-specific transformation. It generates a “residual query” that is appended to the original conditioning sequence: c ← concat(c,qθ(c)). The MLLM then processes this refined sequence, h = g(c). The RQA acts as a learnable “prompt”, guiding the frozen MLLM to extract features that are more salient for the downstream generative task without modifying any of the MLLM’s original weights.

This parameter-efficient approach effectively adapts the MLLM for generation at a fraction of the computational cost. The efficacy of the RQA is empirically validated in Fig. 2b and Sec. 4.4.

###### 3.4. Masked Image Modeling

A key feature of text-to-image training is the inherent sparsity of supervision: a short textual description provides only a high-level, incomplete specification of the corresponding image [25, 54]. This forces the model to learn a compositional understanding of scenes and objects to fill in the missing details. In contrast, our self-conditioning approach provides a dense, complete representation of the target image, which can encourage the model to learn a trivial identity mapping rather than a meaningful generative prior.

To emulate the benefits of sparse supervision, we introduce a Masked Image Modeling strategy inspired by masked autoencoders [19]. During training, we randomly mask a fraction of the image patch tokens cimg with a masking ratio r ∈ [0,1]. This is implemented by element-wise

Algorithm 1 Image-Only Pre-training for UMM Generation Require: Image dataset D; frozen pre-trained MLLM g;

frozen ViT encoder v; auxiliary prompt embeddings caux; mask ratio r.

Require: Randomly initialized diffusion network Fθ and

residual query adapter qθ.

- 1: repeat
- 2: Sample image x ∼ D, noise z ∼ N(0,I), time t ∼ U(0,1).
- 3: Compute noised image: xt = (1 − t) · x + t · z.
- 4: Extract image patch embeddings: cimg = v(x).
- 5: Generate random mask M with masking ratio r and apply it: cimg ← cimg ⊙ M.
- 6: Form the initial condition: c = concat(caux,cimg).
- 7: Refine condition with residual query adapter: c ← concat(c,qθ(c)).
- 8: Compute latent condition from frozen MLLM: h = g(c).
- 9: Compute loss: L(θ) = ∥Fθ(xt,t,h) − (z − x)∥22.
- 10: Update trainable parameters θ using gradients from L(θ).
- 11: until convergence

2×D, where entries are drawn from a Bernoulli distribution with parameter (1 − r): cimg ← cimg ⊙ M. This simple yet effective technique transforms the training objective from dense reconstruction to a more challenging sparse-to-dense task. The model is forced to infer the content of the masked patches from the visible ones, promoting the learning of robust, context-aware visual representations. As shown in our experiments (see Fig. 2b and Sec. 4.4), this significantly improves generation quality. Our complete training procedure is detailed in Alg. 1 and Fig. 2.

multiplication with a binary mask M ∈ {0,1}P

### 4. Experiment

We conduct comprehensive experiments to validate the efficacy of our proposed framework, IOMM. Our evaluation is designed to systematically assess its performance in text-to-image generation, analyze the impact of different training data compositions, and ablate its core architectural components.

###### 4.1. Experimental Setting

Datasets. Our pre-training corpus comprises the Megalith10M [35] and text-to-image-2M [18] datasets. For the finetuning stage, we leverage a curated collection of high-quality, instruction-following datasets, namely BLIP3-o-60K [9], Echo-4o-Image [59], and ShareGPT-4o-Image [8]. All images undergo a standardized preprocessing pipeline: we

- Table 1. Quantitative comparison on text-to-image generation benchmarks. The (↑) symbol indicates that higher scores are better. †Results obtained using rewritten prompts from the original GenEval benchmark. ∗Indicates the model was trained on an additional 30M proprietary image-text pairs.

GenEval

METHOD

DPGBench (↑) WISE (↑) Single Obj. Two Obj. Counting Colors Position Color Attri. Overall (↑)

Gen. Only

SDv1.5 [42] 0.97 0.38 0.35 0.76 0.04 0.06 0.43 63.18 0.32 SDv2.1 [42] 0.98 0.51 0.44 0.85 0.07 0.17 0.50 - 0.32 SD3-Medium [14] 0.99 0.94 0.72 0.89 0.33 0.60 0.74 84.08 0.42 SDXL [40] 0.98 0.74 0.39 0.85 0.15 0.23 0.55 74.65 0.43 PixArt-α [7] 0.98 0.50 0.44 0.80 0.08 0.07 0.48 71.11 0.47

- DALL-E 2 [41] 0.94 0.66 0.49 0.77 0.10 0.19 0.52 - -
- DALL-E 3 [1] 0.96 0.87 0.47 0.83 0.43 0.45 0.67 83.50 Lumos-T2I [33] 0.99 0.64 0.52 0.84 0.15 0.30 0.57 79.90 -

###### Unified Models

Chameleon [45] - - - - - - 0.39 - Show-o [56] 0.98 0.80 0.66 0.84 0.31 0.50 0.68 - 0.35 Show-o2-7B [57] 1.00 0.87 0.58 0.92 0.52 0.62 0.76† 86.14 0.39 Janus [48] 0.97 0.68 0.30 0.84 0.46 0.42 0.61 79.68 0.23 JanusFlow [34] 0.97 0.59 0.45 0.83 0.53 0.42 0.63 80.09 Janus-Pro-1B [11] 0.98 0.82 0.51 0.89 0.65 0.56 0.73 82.63 0.26 Janus-Pro-7B [11] 0.99 0.89 0.59 0.90 0.79 0.66 0.80 84.19 0.35 MetaQuery-B [38] - - - - - - 0.74† 80.04 0.46 MetaQuery-L [38] - - - - - - 0.78† 81.10 0.55 MetaQuery-XL [38] - - - - - - 0.80† 82.05 0.55 BLIP3-o-4B [9] - - - - - - 0.81 79.36 0.50 BLIP3-o-8B* [9] - - - - - - 0.84 81.60 0.62 BAGEL-7B [12] 0.98 0.95 0.84 0.95 0.78 0.77 0.88† - 0.52

Ours

IOMM-B 512 0.99 0.92 0.83 0.94 0.91 0.75 0.89 82.95 0.55 IOMM-B 1024 0.99 0.91 0.75 0.93 0.88 0.75 0.87 80.71 0.50 IOMM-L 512 0.99 0.91 0.82 0.94 0.85 0.72 0.87 76.09 0.53 IOMM-L 1024 1.00 0.91 0.71 0.92 0.78 0.78 0.85 72.26 0.48

apply a central crop and resize them to a resolution of either 512 × 512 or 1024 × 1024.

Neural network architectures. The core of our model adopts the Multi-Modal Diffusion Transformer (MM-DiT) architecture [14], as implemented in FLUX [24]. This design employs independent attention mechanisms for image and text modalities to facilitate robust cross-modal fusion. To investigate scaling properties, we instantiate three variants: IOMM-B (1.6B parameters), IOMM-L (2.7B parameters), and IOMM-XL, with the latter following the 6B parameter Z-Image framework [4]. For the auxiliary MLLM component, a frozen InternVL3-2B [68] is employed as a feature extractor, offering high-quality representations with a minimal computational footprint.

Implementation and evaluation. We implement our framework in PyTorch [39] and utilize the AdamW optimizer [31] for training of IOMM-B and IOMM-L and the

Muon optimizer [23] for IOMM-XL. Adhering to established practices in generative modeling [32, 62], we maintain an exponential moving average (EMA) of the model weights with a decay rate of 0.999. All reported results are derived from the EMA model weights to ensure stability and improved performance. For evaluation, we follow standard protocols established in prior works [11, 14, 38]. To assess generative quality and text-image alignment, we employ a suite of comprehensive benchmarks: GenEval [15], DPGBench [21], and WISE [36]. The image editing capabilities of our model are specifically evaluated using the ImgEditBench [60]. Further details regarding hyperparameters and the training infrastructure are available in App. B.

###### 4.2. Performance on Text-to-Image Generation

We benchmark IOMM against SOTA models in Tab. 1. Our base model, IOMM-B (512px) built on a 1.6B generative backbone, achieves a new SOTA score of 0.89 on GenEval. Notably, this performance surpasses strong baselines like BAGEL (0.88) and BLIP3-o-8B*(0.84, trained with an extra

- Table 2. Evaluating different fine-tuning strategies on various open-source UMMs. The notation A ⊕ B denotes applying fine-tuning method B to a pre-trained model A. The symbols ↓/↑ indicate the performance change relative to the baseline pre-trained model.

METHOD Res. NFE

GenEval

WISE (↑)

Single Obj. Two Obj. Counting Colors Position Color Attri. Overall (↑)

OpenUni-L [50] 512 20×2 0.99 0.91 0.77 0.90 0.75 0.76 0.85 0.52

⊕Image finetuning 512 20×2 1.00↑0.01 0.98↑0.07 0.22↓0.55 0.91↑0.01 0.60↓0.15 0.77↑0.01 0.74↓0.11 0.49↓0.03 ⊕Pair finetuning 512 20×2 0.99 0.94↑0.03 0.82↑0.05 0.91↑0.01 0.85↑0.10 0.76 0.88↑0.03 0.62↑0.10 ⊕Mix finetuning 512 20×2 0.99 0.91 0.78↑0.01 0.93↑0.03 0.87↑0.12 0.78↑0.02 0.88↑0.03 0.59↑0.07

Qwen-Image [49] 512 50×2 0.99 0.91 0.87 0.88 0.73 0.74 0.85 -

⊕Image finetuning 512 50×2 0.55↓0.44 0.51↓0.40 0.38↓0.49 0.43↓0.45 0.30↓0.43 0.37↓0.37 0.42↓0.43 0.41 ⊕Pair finetuning 512 50×2 1.00↑0.01 0.93↑0.02 0.88↑0.01 0.91↑0.03 0.82↑0.09 0.75↑0.01 0.88↑0.03 0.63 ⊕Mix finetuning 512 50×2 1.00↑0.01 0.92↑0.01 0.87 0.91↑0.03 0.82↑0.09 0.79↑0.05 0.89↑0.04 0.63

Qwen-Image [49] 1024 50×2 0.99 0.93 0.88 0.90 0.77 0.74 0.87 0.62

⊕Image finetuning 1024 50×2 0.54↓0.45 0.61↓0.32 0.47↓0.41 0.47↓0.43 0.28↓0.49 0.47↓0.27 0.47↓0.40 0.35↓0.27 ⊕Pair finetuning 1024 50×2 1.00↑0.01 0.93↑0.01 0.88 0.91↑0.01 0.82↑0.05 0.75↑0.01 0.88↑0.01 0.63↑0.01 ⊕Mix finetuning 1024 50×2 0.99 0.92↓0.01 0.90↑0.02 0.91↑0.01 0.81↑0.04 0.80↑0.06 0.89↑0.02 0.63↑0.01

- Table 3. Image editing benchmark results. Methods highlighted in red are trained on specific editing datasets. Our IOMM, highlighted in blue , is evaluated in a training-free setting without any training on editing data.

METHOD

ImgEdit-Bench Add Adjust Extract Replace Remove Background Style Hybrid Action Overall (↑)

Trained with editing data

MagicBrush [63] 2.84 1.58 1.51 1.97 1.58 1.75 2.38 1.62 1.22 1.90 Instruct-Pix2Pix [3] 2.45 1.83 1.44 2.01 1.50 1.44 3.55 1.20 1.46 1.88 AnyEdit [61] 3.18 2.95 1.88 2.47 2.23 2.24 2.85 1.56 2.65 2.45 UltraEdit [65] 3.44 2.81 2.13 2.96 1.45 2.83 3.76 1.91 2.98 2.70 OmniGen [53] 3.47 3.04 1.71 2.94 2.43 3.21 4.19 2.24 3.38 2.96 ICEdit [64] 3.58 3.39 1.73 3.15 2.93 3.08 3.84 2.04 3.68 3.05 Step1X-Edit [30] 3.88 3.14 1.76 3.40 2.41 3.16 4.63 2.64 2.52 3.06 BAGEL [12] 3.56 3.31 1.70 3.3 2.62 3.24 4.49 2.38 4.17 3.20

Ours (zero-shot)

IOMM-B (text-image pair pre-trained) 3.18 2.17 1.92 2.70 1.17 3.36 4.39 1.49 3.14 2.61 IOMM-B (image-only pre-trained) 3.84 2.37 2.12 2.60 1.30 3.14 4.41 1.80 3.78 2.82

30M proprietary image-text pairs), despite IOMM being trained exclusively on public datasets and with remarkable efficiency (1050 H800 GPU hours). Furthermore, IOMM-B attains a competitive score of 0.55 on the WISE benchmark, demonstrating that our approach effectively preserves world knowledge without degradation. Qualitative results in Fig. 1a showcase our model’s strong compositional abilities.

Analysis of model scaling. The lower performance of our larger IOMM-L model is an artifact of constrained training resources; it was trained for half the epochs of IOMM-B. When controlling for training duration (5 epochs), IOMM-L outperforms IOMM-B (0.87 vs. 0.86 on GenEval), confirming a positive scaling trend and suggesting potential for further gains with continued training.

- 4.3. Impact of Pre-training and Fine-tuning Data

0.90

Image-Only pre-training

0.88

Text-image pair pre-training

GenEval

0.86

0.84

0.82

0.80

0.78

Echo-4o-Image BLIP3-o-60K ShareGPT-4o-Image

Dataset

Figure 3. Analysis of different data paradigms. Fine-tuning performance comparison of models pre-trained on different data compositions (image-only, text-image pair) across distinct datasets.

data types: (a) image-only, (b) text-image pairs, and (c) a mixture of both. This section presents a systematic ablation study on the six possible combinations of these data types across the two stages, focusing on their efficacy for text-to-image generation.

The role of pre-training data. We first compare models pre-trained on image-only data versus those pre-trained on

We investigate the impact of data composition during the pre-training and fine-tuning stages. We define three distinct

0.90

80

|[Figure 14]| |
|---|---|
| | |
| | |
| | |
| | |
| | |

0.85

0.64 0.72 0.78 0.77 0.78

0.8

0.9

0.88

78

0.86

0.71 0.77 0.81 0.81 0.84

0.7

0.80

0.7

0.84

###### MixRatio

76

###### GenEval

###### GenEval

0.82

0.6

0.77 0.81 0.87 0.83 0.86

0.5

0.80

74

0.75

0.78

0.5

72

0.73 0.83 0.85 0.84 0.85

0.3

0.76

0.70

0.74

Residual Query Adapter

GenEval

0.4

70

0.77 0.76 0.86 0.84 0.85

0.1

0.72

MetaQuery

DPGBench

0.65

0.70

68

2k 4k 6k 8k 10k

2k 4k 6k 8k 10k

0.0 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95

###### Training Steps

Mask Ratio

Finetuning Steps

(a) Residual query adapter.

(b) Various mask ratio.

(c) Various mix ratio of data.

Figure 4. Ablation studies of key components in IOMM. These experiments analyze the impact of our primary design choices: (a) the residual query adapter, (b) the mask ratio for sparse reconstruction, and (c) the data mixture ratio during fine-tuning.

text-image pairs. As illustrated in Fig. 3 and Fig. 1c, the image-only pre-trained model consistently achieves superior or comparable performance to its text-image pair counterpart, irrespective of the fine-tuning data composition.

The role of fine-tuning data. Next, we analyze the effect of the fine-tuning data composition. Beyond using image-only or text-image pair data exclusively, we explore a mixed-data strategy. Remarkably, Fig. 1c reveals that for models pre-trained under both paradigms, fine-tuning with the mixed data yields the highest performance on GenEval. Conversely, fine-tuning with image-only data consistently results in the lowest scores.

Generalization to open-source UMMs. To validate the generalizability of our findings, we apply our fine-tuning strategies to prominent open-source UMMs: OpenUni-L3.6B [50] and Qwen-Image-20B [49]. For the larger QwenImage model, we employ LoRA [20] (with r = 64 and α = 64) for computational efficiency. The results, summarized in Tab. 2, corroborate our primary conclusion: the mixeddata fine-tuning approach consistently outperforms the other strategies on GenEval. For instance, it improves the GenEval score of OpenUni-L from a baseline of 0.85 to 0.88. Even for the powerful Qwen-Image model, this strategy yields notable gains, increasing scores from 0.85 to 0.89 (512px) and 0.87 to 0.89 (1024px).

Beyond generation quality, we evaluate world knowledge and reasoning using the WISE benchmark. As shown in the final column of Tab. 2, both text-image pair and mixeddata fine-tuning provide a substantial performance uplift for OpenUni-L (up to 0.10) and a modest improvement for Qwen-Image (0.01). In contrast, fine-tuning with image-only data proves detrimental across nearly all scenarios, significantly impairing the models’ prompt-following ability—an

effect particularly pronounced in larger models (see App. C.6 for a detailed analysis).

Emergent image editing capabilities. A surprising and significant finding is the emergence of strong image editing capabilities. Tab. 3 demonstrates that our model, when pre-trained on image-only data, achieves competitive performance on the ImgEdit-Bench benchmark. Crucially, this is accomplished in a zero-shot setting, without any fine-tuning on task-specific editing data. This training-free approach not only surpasses the performance of the same model pretrained on text-image pairs but also outperforms several strong baselines like UltraEdit [65] that are explicitly trained on editing datasets.

4.4. Ablation Studies on Key Components of IOMM Unless specified otherwise, all experiments in this section are conducted using the IOMM-XL model pre-trained exclusively on image-only data.

Efficacy of the residual query adapter. To further validate the efficacy of our proposed residual query adapter, we compare it against a strong baseline, MetaQuery [38], trained on identical data with the same 256 query tokens. The results, depicted in Fig. 4a, clearly demonstrate that our approach achieves a significantly faster convergence rate. Notably, extending the fine-tuning of MetaQuery by an additional 8K steps only yields a score of 0.82 on GenEval.

Impact of image token mask ratio. We investigate the impact of the mask ratio for image tokens, a key parameter in our sparse reconstruction objective. As shown in Fig. 4b, performance improves as the ratio increases, peaking at an impressive 0.88 GenEval score and a DPGBench score of

79.79 with a mask ratio of 0.45. This result validates the effectiveness of our learning paradigm. However, an excessively high ratio (e.g., 0.95) leads to a sharp performance degradation (a drop to 0.77 and 69.41, respectively), likely due to significant information loss that impairs the training guidance for the generation process.

Influence of data mixture ratio. We examine the effect of varying the proportion of image-only data versus textimage pairs during the fine-tuning stage. A mix ratio of 1.0 corresponds to pure image-only data, while 0.0 signifies pure text-image pairs. Fig. 4c reveals that performance initially increases with the mix ratio, reaching its optimum at 0.5. Furthermore, an optimal ratio of approximately 0.5 not only yields the best results but also demonstrates greater training stability, whereas lower ratios are prone to performance volatility in the later stages of fine-tuning.

### 5. Conclusion

We introduced IOMM, a novel and efficient framework for training UMM visual generation components using primarily image-only data, addressing the common paired-data bottleneck. Our two-stage approach—image-only pre-training followed by mixed-data fine-tuning—achieves SOTA performance with remarkable computational efficiency. Furthermore, we demonstrate that our mixed-data fine-tuning strategy is a generalizable technique that consistently enhances the performance of existing powerful UMMs. Detailed settings and results are in App. B.

### Acknowledgement

This work was supported in part by the National Science and Technology Major Project (No. 2022ZD0115101), NSFC under No. 62576285, the Research Center for Industries of the Future (RCIF) at Westlake University, and the Westlake Education Foundation.

### References

- [1] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023. 6
- [2] BlackForest. Flux. https://github.com/blackforest-labs/flux, 2024. 16
- [3] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023. 7

- [4] Huanqia Cai, Sihan Cao, Ruoyi Du, Peng Gao, Steven Hoi, Zhaohui Hou, Shijie Huang, Dengyang Jiang, Xin Jin, Liangchen Li, et al. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025. 6
- [5] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. Maskgit: Masked Generative Image Transformer. In Computer Vision and Pattern Recognition (CVPR), pages 11305–11315, 2022. 3
- [6] Junsong Chen, Chongjian Ge, Enze Xie, Yue Wu, Lewei Yao, Xiaozhe Ren, Zhongdao Wang, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-σ: Weak-to-Strong Training of Diffusion Transformer for 4k Text-to-Image Generation. In European Conference on Computer Vision (ECCV), pages 74–91, 2024. 3
- [7] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James T. Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. Pixart-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis. In The Twelfth International Conference on Learning Representations, 2024. 3, 6, 16, 17
- [8] Junying Chen, Zhenyang Cai, Pengcheng Chen, Shunian Chen, Ke Ji, Xidong Wang, Yunjin Yang, and Benyou Wang. Sharegpt-4o-Image: Aligning Multimodal Models with GPT4o-Level Image Generation. arXiv.org, abs/2506.18095, 2025. 5, 14
- [9] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, Le Xue, Caiming Xiong, and Ran Xu. Blip3-o: A Family of Fully Open Unified Multimodal Models-Architecture, Training and Dataset. arXiv.org, abs/2505.09568, 2025. 1, 3, 5, 6, 14, 16, 17
- [10] Junsong Chen, Shuchen Xue, Yuyang Zhao, Jincheng Yu, Sayak Paul, Junyu Chen, Han Cai, Enze Xie, and Song Han. Sana-sprint: One-step diffusion with continuous-time consistency distillation. arXiv preprint arXiv:2503.09641, 2025. 3
- [11] Xi-aokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and C. Ruan. Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling. arXiv.org, abs/2501.17811, 2025. 3, 6, 16, 17
- [12] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Shi Guang, and Haoqi Fan. Emerging Properties in Unified Multimodal Pretraining. arXiv.org, abs/2505.14683,

2025. 3, 6, 7, 17

- [13] Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, Xiangwen Kong, Xiangyu Zhang, Kaisheng Ma, and Li Yi. Dreamllm: Synergistic Multimodal Comprehension and Creation. In The Twelfth International Conference on Learning Representations, 2024. 1, 3
- [14] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first

- international conference on machine learning, 2024. 3, 6, 16, 17
- [15] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment. In Conference on Neural Information Processing Systems (NeurIPS), 2023. 6
- [16] Google. Experiment with gemini 2.0 flash native image generation, 2025. 1, 3
- [17] Google. Gemini 2.5 flash image: High-consistency image generation and editing, 2025. Official model page on Google AI Studio. Internal development code name: nano-banana. 1, 3
- [18] Jacky He and contributors. text-to-image-2M: A highquality, diverse text–image training dataset. https:// huggingface.co/datasets/jackyhate/textto-image-2M, 2024. 5, 14
- [19] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked Autoencoders Are Scalable Vision Learners. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2022. 3, 5
- [20] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3,

2022. 8, 14

- [21] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip Diffusion Models with LLM for Enhanced Semantic Alignment. arXiv.org, abs/2403.05135,

2024. 6

- [22] Jiaxing Huang, Kaiwen Cui, Dayan Guan, Aoran Xiao, Fangneng Zhan, Shijian Lu, Shengcai Liao, and Eric P. Xing. Masked Generative Adversarial Networks are Data-Efficient Generation Learners. In Conference on Neural Information Processing Systems (NeurIPS), 2022. 3
- [23] Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL https://kellerjordan. github. io/posts/muon, 6(3):4, 2024. 6
- [24] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. Flux.1 kontext: Flow matching for in-context image generation and editing in latent space, 2025. 6
- [25] Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742,

2025. 3, 5

- [26] Daiqing Li, Aleks Kamko, Ali Sabet, Ehsan Akhgari, Linmiao Xu, and Suhail Doshi. Playground v2. 3
- [27] Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi. Playground v2.5: Three Insights towards Enhancing Aesthetic Quality in Text-to-Image Generation. arXiv.org, abs/2402.17245, 2024. 3
- [28] Bin Lin, Zongjian Li, Xinhua Cheng, Yuwei Niu, Yang Ye, Xianyi He, Shenghai Yuan, Wangbo Yu, Shaodong Wang,

- Yunyang Ge, Yatian Pang, and Li Yuan. Uniworld-V1: HighResolution Semantic Encoders for Unified Visual Understanding and Generation. arXiv.org, abs/2506.03147, 2025. 3
- [29] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 4
- [30] Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025. 7
- [31] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 6
- [32] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024. 6
- [33] Shuailei Ma, Kecheng Zheng, Ying Wei, Wei Wu, Fan Lu, Yifei Zhang, Chen-Wei Xie, Biao Gong, Jiapeng Zhu, and Yujun Shen. Learning visual generative priors without text. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 8051–8061, 2025. 3, 6
- [34] Yiyang Ma, Xingchao Liu, Xi-aokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Xingkai Yu, Liang Zhao, Yisong Wang, Jiaying Liu, and C. Ruan. Janusflow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation. In Computer Vision and Pattern Recognition, pages 7739–7751, 2024. 3, 6
- [35] Ollin Matsubara and Draw Things AI Team. Megalith-10M: A dataset of 10 million public-domain photographs. https: //huggingface.co/datasets/madebyollin/ megalith - 10m, 2024. CC0/Flickr-Commons images; Florence-2 captions available in the *megalith-10mflorence2* variant. 5, 14
- [36] Yuwei Niu, Munan Ning, Mengren Zheng, Bin Lin, Peng Jin, Jiaqi Liao, Kun-Peng Ning, Bin Zhu, and Li Yuan. Wise: A World Knowledge-Informed Semantic Evaluation for Text-toImage Generation. arXiv.org, abs/2503.07265, 2025. 6
- [37] OpenAI. Introducing 4o image generation, 2025. 1
- [38] Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, Ji Hou, and Saining Xie. Transfer between Modalities with MetaQueries. arXiv.org, abs/2504.06256, 2025. 1, 3, 5, 6, 8, 16, 17
- [39] A Paszke. Pytorch: An imperative style, high-performance deep learning library. arXiv preprint arXiv:1912.01703, 2019. 6
- [40] Dustin Podell, Zion English, Kyle Lacey, A. Blattmann, Tim Dockhorn, Jonas Muller, Joe Penna, and Robin Rombach. Sdxl: Improving Latent Diffusion Models for HighResolution Image Synthesis. In The Twelfth International Conference on Learning Representations, 2024. 3, 6, 16, 17
- [41] A. Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical Text-Conditional Image Generation with CLIP Latents. arXiv.org, abs/2204.06125, 2022. 6
- [42] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image

- synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 3, 6, 16, 17
- [43] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 4
- [44] Peng Sun, Yi Jiang, and Tao Lin. Unified continuous generative models. arXiv preprint arXiv:2505.07447, 2025. 4
- [45] Chameleon Team, Mingda Chen, and Jacob Kahn. Chameleon: Mixed-Modal Early-Fusion Foundation Models. arXiv.org, abs/2405.09818, 2024. 3, 6
- [46] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 5
- [47] XuDong Wang, Xingyi Zhou, Alireza Fathi, Trevor Darrell, and Cordelia Schmid. Visual lexicon: Rich image features in language space. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19736–19747, 2025. 3
- [48] Chengyue Wu, Xi-aokang Chen, Zhiyu Wu, Yiyang Ma, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, C. Ruan, and Ping Luo. Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation. In Computer Vision and Pattern Recognition, pages 12966– 12977, 2024. 3, 6, 16, 17
- [49] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025. 1, 3, 7, 8, 18
- [50] Size Wu, Zhonghua Wu, Zerui Gong, Qi Tao, Sheng Jin, Qinyue Li, Wei Li, and Chen Change Loy. Openuni: A Simple Baseline for Unified Multimodal Understanding and Generation. arXiv.org, abs/2505.23661, 2025. 7, 8, 18
- [51] Size Wu, Wenwei Zhang, Lumin Xu, Sheng Jin, Zhonghua Wu, Qi Tao, Wentao Liu, Wei Li, and Chen Change Loy. Harmonizing Visual Representations for Unified Multimodal Understanding and Generation. arXiv.org, abs/2503.21979,

2025. 3

- [52] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, Song Han, and Yao Lu. Vila-U: a Unified Foundation Model Integrating Visual Understanding and Generation. In The Thirteenth International Conference on Learning Representations, 2025. 3
- [53] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Chaofan Li, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13294–13304, 2025. 7

- [54] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu, Yao Lu, et al. Sana: Efficient high-resolution image synthesis with linear diffusion transformers. arXiv preprint arXiv:2410.10629,

2024. 3, 4, 5

- [55] Ji Xie, Trevor Darrell, Luke Zettlemoyer, and XuDong Wang. Reconstruction alignment improves unified multimodal models. arXiv preprint arXiv:2509.07295, 2025. 2
- [56] Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One Single Transformer to Unify Multimodal Understanding and Generation. In The Thirteenth International Conference on Learning Representations, 2025. 3, 6, 17
- [57] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Showo2: Improved Native Unified Multimodal Models. arXiv.org, abs/2506.15564, 2025. 6
- [58] Zhiyuan Yan, Kaiqing Lin, Zongjian Li, Junyan Ye, Hui Han, Zhendong Wang, Hao Liu, Boyang Lin, Hui Li, Xiaodan Xu, and Xin Xiao. Unified multimodal model as auto-encoder,

2025. 3

- [59] Junyan Ye, Dongzhi Jiang, Zihao Wang, Leqi Zhu, Zhenghao Hu, Zilong Huang, Jun He, Zhiyuan Yan, Jinghua Yu, Hongsheng Li, et al. Echo-4o: Harnessing the power of gpt4o synthetic images for improved image generation. arXiv preprint arXiv:2508.09987, 2025. 5, 14
- [60] Yang Ye, Xianyi He, Zongjian Li, Bin Lin, Shenghai Yuan, Zhiyuan Yan, Bohan Hou, and Li Yuan. Imgedit: A Unified Image Editing Dataset and Benchmark. arXiv.org, abs/2505.20275, 2025. 6
- [61] Qifan Yu, Wei Chow, Zhongqi Yue, Kaihang Pan, Yang Wu, Xiaoyang Wan, Juncheng Li, Siliang Tang, Hanwang Zhang, and Yueting Zhuang. Anyedit: Mastering unified high-quality image editing for any idea. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26125– 26135, 2025. 7
- [62] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940,

2024. 6

- [63] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instructionguided image editing. Advances in Neural Information Processing Systems, 36:31428–31449, 2023. 7
- [64] Zechuan Zhang, Ji Xie, Yu Lu, Zongxin Yang, and Yi Yang. In-context edit: Enabling instructional image editing with incontext generation in large scale diffusion transformer. arXiv preprint arXiv:2504.20690, 2025. 7
- [65] Haozhe Zhao, Xiaojian Shawn Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. Ultraedit: Instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems, 37:3058–3093, 2024. 7, 8
- [66] Chunting Zhou, LILI YU, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict

- the Next Token and Diffuse Images with One Multi-Modal Model. In The Thirteenth International Conference on Learning Representations, 2025. 3
- [67] Yupeng Zhou, Daquan Zhou, Zuo-Liang Zhu, Yaxing Wang, Qibin Hou, and Jiashi Feng. Maskdiffusion: Boosting Textto-Image Consistency with Conditional Mask. International Journal of Computer Vision, abs/2309.04399, 2023. 3
- [68] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 6
- [69] Siyu Zou, Jiji Tang, Yiyi Zhou, Jing He, Chaoyi Zhao, Rongsheng Zhang, Zhipeng Hu, and Xiaoshuai Sun. Towards Efficient Diffusion-Based Image Editing with Instant Attention Masks. In AAAI Conference on Artificial Intelligence (AAAI), pages 7864–7872, 2024. 3

### Contents

- 1. Introduction 1
- 2. Related Work 3
- 3. Methodology 3

- 3.1. Preliminaries on Diffusion Models . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 3.2. Image-Only Pre-training via Self-Conditioning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 3.3. Residual Query Adapter . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 3.4. Masked Image Modeling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

- 4. Experiment 5

- 4.1. Experimental Setting . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 4.2. Performance on Text-to-Image Generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 4.3. Impact of Pre-training and Fine-tuning Data . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
- 4.4. Ablation Studies on Key Components of IOMM . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8

- 5. Conclusion 9

- A. Utilization of Large Language Models (LLMs) 14
- B. Detailed Experimental Settings 14 B.1. Pre-training Settings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14 B.2. Finetuning Settings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14 B.3. UMM Finetuning Settings . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- C. More Results 15

- C.1. DPGBench Evaluation Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- C.2. WISE Evaluation Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- C.3. Different training recipe . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- C.4. Image Editing Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- C.5. UMM finetune result . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- C.6. Generation results comparison of UMM finetuning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- C.7. Prompts details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

### A. Utilization of Large Language Models (LLMs)

In this study, Large Language Models (LLMs) are employed at the sentence level to assist in linguistic refinement. Their use was strictly confined to improving grammatical accuracy and overall readability of the manuscript. All research concepts, methodological designs, experimental processes, and analytical findings remain entirely original and have been solely contributed by the authors.

#### B. Detailed Experimental Settings This section elaborates on the experimental setup, including all relevant hyperparameter choices.

- B.1. Pre-training Settings

- The results presented in Tab. 1 are derived using the pre-training configurations outlined in Tab. 4. Due to computational resource constraints, Exponential Moving Average (EMA) decay was not applied during the training of IOMM-L and IOMMXL. All models were pre-trained on the Megalith-10M [35] and text-to-image-2M [18] datasets (except for IOMM-XL), comprising approximately 11 million images in total. Each image was resized so that its shorter edge was 512 pixels while preserving the original aspect ratio, then a central crop was applied to obtain a 512 × 512 image. Notably, since neither dataset provides images at a resolution of 1024 × 1024, we did not deploy high-resolution pre-training.

Table 4. Pre-training settings.

METHOD IOMM-B IOMM-L IOMM-XL Optimization

Optimizer AdamW Muon β (0.9,0.95) (0.9,0.95) Learning rate 1e-4 1e-4 Max gradient norm 1.0 1.0 Weight decay 0.0 0.0

Training Configuration

Generative Model Size 1.6B 2.7B 6B Training data type Image-only Image-only Image-only EMA decay 0.999 - Global batch size 1024 512 4096 Image token mask ratio r 0.85 0.85 0.45

B.2. Finetuning Settings

We fine-tuned the two models (B&L) at resolutions of 512 and 1024, respectively, using the pre-training settings specified in Tab. 4. The fine-tuning datasets include BLIP3o-60K [9], Echo-4o-Image [59], and ShareGPT-4o-Image [8], collectively comprising approximately 210,000 high-resolution images (except for IOMM-XL). All images in these datasets are at 1024 × 1024 resolution. For fine-tuning at both 512 and 1024 resolutions, we applied central cropping to resize images to the target resolution.

B.3. UMM Finetuning Settings

- The results presented in Tab. 2 were obtained using the fine-tuning configurations specified in Tab. 6. For OpenUni-L, we performed full fine-tuning on both the connector module and the generative model. In contrast, for Qwen-Image-20B, we applied Low-Rank Adaptation (LoRA) [20] to fine-tune the model. Both models utilized a frozen understanding module. Additionally, due to computational constraints, Exponential Moving Average (EMA) decay was not implemented for QwenImage-20B.

Table 5. Finetuning settings.

METHOD IOMM-B IOMM-L IOMM-XL Resolution 512 1024 512 1024 512

Optimization

Optimizer AdamW AdamW Muon β (0.9,0.95) (0.9,0.95) (0.9,0.95) Learning rate 1e-4 1e-4 1e-4 Max gradient norm 1.0 1.0 1.0 Weight decay 0.0 0.0 0.0 Generative Model Size 1.6B 1.6B 2.7B 2.7B 6B

Training Configuration

Training data type Mix Mix Mix Mix Mix EMA decay 0.999 0.999 - - Global batch size 256 96 256 96 256 Image token mask ratio r 0.85 0.85 0.85 0.85 0.45 Mix ratio λ 0.5 0.5 0.5 0.5 0.5

Table 6. UMM finetuning settings.

METHOD OpenUni-L Qwen-Image-20B

Optimization

Optimizer AdamW AdamW β (0.9,0.95) (0.9,0.95) Learning rate 1e-4 1e-4 Max gradient norm 1.0 1.0 Weight decay 0.0 0.0

Training Configuration

Training data type Mix/Image-only/Pair Mix/Image-only/Pair EMA decay 0.999 Global batch size 256 48 Epochs 12 5 Image token mask ratio r 0.85 0.85 Mix ratio λ 0.5 0.5

LoRA Configuration

LoRA rank - 64 LoRA alpha - 64 LoRA dropout - 0.0

### C. More Results

- C.1. DPGBench Evaluation Results

- The Tab. 7 shows the detailed results of the DPGBench evaluation shown in Tab. 1.

C.2. WISE Evaluation Results

- The Tab. 8 shows the detailed results of the WISE evaluation shown in Tab. 1.

Table 7. DPGBench evaluation results. Here BLIP3-o-8B* donates the model that is trained with an 30 million proprietary data.

METHOD Global Entity Attribute Relation Other Overall Gen. Only

SDv1.5 [42] 74.63 74.23 75.39 73.49 67.81 63.18 SD3-Medium [14] 87.90 91.01 88.83 80.70 88.68 84.08 SDXL [40] 83.27 82.43 80.91 86.76 80.41 74.65 PixArt-α [7] 74.97 79.32 78.60 82.57 76.96 71.11 FLUX.1-dev [2] 74.35 90.00 88.96 90.87 88.33 83.84

###### Unified Models

Janus [48] 82.33 87.38 87.70 85.46 86.41 79.68 Janus-Pro-1B [11] 87.58 88.63 88.17 88.98 88.30 82.63 Janus-Pro-7B [11] 86.90 88.90 89.40 89.32 89.48 84.19 MetaQuery-B [38] - - - - - 80.04 MetaQuery-L [38] - - - - - 81.10 MetaQuery-XL [38] - - - - - 82.05 BLIP3-o-4B [9] - - - - - 79.36 BLIP3-o-8B* [9] - - - - - 81.60

Ours

IOMM-B 512 91.33 89.39 90.07 86.89 87.78 82.95 IOMM-B 1024 86.20 88.39 87.69 90.11 87.05 80.71 IOMM-L 512 83.28 83.61 84.69 83.46 79.83 76.09 IOMM-L 1024 79.27 82.00 80.93 82.81 78.68 72.26

- C.3. Different training recipe

The results presented in Tab. 9 correspond to the training configurations depicted in Fig. 1b. All models underwent approximately 5 epochs of pre-training on a dataset comprising 11 million images, followed by 10 epochs of fine-tuning on a dataset of approximately 210,000 images. Notably, the model pre-trained exclusively on image-only data and fine-tuned on a mixed data achieved superior performance across most metrics in the GenEval benchmark.

- C.4. Image Editing Results

Fig. 5 compares the image editing capabilities of models pre-trained exclusively on image-only data (right) and those pretrained on image-text pairs (middle). The sole distinction between these models lies in their pre-training data type; all other hyperparameters and fine-tuning settings remain consistent. Despite in a zero-shot setting, the model pre-trained with image-only data demonstrates superior consistency with the original input image. For instance, in the first row, the right image closely resembles the raw input, while in the second and third rows, the right images maintain nearly identical gestures to the original.

- C.5. UMM finetune result Tab. 10 show the detailed WISE score of the UMM finetuning results shown in Tab. 2.
- C.6. Generation results comparison of UMM finetuning

As illustrated in Fig. 6, fine-tuning enhances the model’s performance on tasks requiring reasoning. Although the understanding module was frozen during fine-tuning, the model’s improved alignment between images and text enables more accurate generation of desired details. What’s more, a qualitative comparison between the original Qwen-Image model and our fine-tuned version. Our method enhances the model’s ability to generate images with richer visual detail and improved alignment to the textual prompt.

Table 8. WISE evaluation results. Here BLIP3-o-8B* donates the model that is trained with an 30 million proprietary data.

METHOD Cultural Time Space Biology Physics Chemistry Overall Gen. Only

SDv1.5 [42] 0.34 0.35 0.32 0.28 0.29 0.21 0.32 SDv2.1 [42] 0.30 0.38 0.35 0.33 0.34 0.21 0.32 SD3-Medium [14] 0.42 0.44 0.48 0.39 0.47 0.29 0.42 SDXL [40] 0.43 0.48 0.47 0.44 0.45 0.27 0.43 SD3.5-Large [14] 0.44 0.50 0.58 0.44 0.52 0.31 0.46 PixArt-α [7] 0.45 0.50 0.48 0.49 0.56 0.34 0.47 FLUX.1-dev [7] 0.48 0.58 0.62 0.42 0.51 0.35 0.50

###### Unified Models

Show-o [56] 0.28 0.40 0.48 0.30 0.46 0.30 0.35 Janus [48] 0.16 0.26 0.35 0.28 0.30 0.14 0.23 Janus-Pro-1B [11] 0.20 0.28 0.45 0.24 0.32 0.16 0.26 Janus-Pro-7B [11] 0.30 0.37 0.49 0.36 0.42 0.26 0.35 MetaQuery-B [38] 0.44 0.49 0.58 0.41 0.49 0.34 0.46 MetaQuery-L [38] 0.56 0.57 0.62 0.48 0.63 0.42 0.55 MetaQuery-XL [38] 0.56 0.55 0.62 0.49 0.63 0.41 0.55 BAGEL [12] 0.44 0.55 0.68 0.44 0.60 0.39 0.52 BLIP3-o-4B [9] - - - - - - 0.50 BLIP3-o-8B* [9] - - - - - - 0.62

Ours

IOMM-B 512 0.50 0.56 0.66 0.49 0.72 0.46 0.55 IOMM-B 1024 0.44 0.50 0.64 0.46 0.63 0.43 0.50 IOMM-L 512 0.48 0.56 0.63 0.49 0.64 0.51 0.53 IOMM-L 1024 0.44 0.48 0.59 0.43 0.58 0.44 0.48

Table 9. Training recipe comparison. The GenEval score of the models pre-trained with different training recipes. Bold denotes the best performance and underline denotes the second best performance.

Finetuning Recipe Single Obj. Two Obj. Counting Colors Position Color Attri. Overall (↑) Pre-trained with Text-Image Pair Data

Image 1.00 0.95 0.63 0.87 0.50 0.72 0.78 Pair 0.99 0.92 0.76 0.91 0.87 0.69 0.86

- Mix 0.99 0.91 0.80 0.92 0.90 0.75 0.88 Pre-trained with Image-Only Data

Image 0.99 0.84 0.24 0.75 0.37 0.45 0.61 Pair 0.99 0.91 0.77 0.93 0.87 0.75 0.87

- Mix 0.99 0.92 0.83 0.94 0.91 0.75 0.89

- C.7. Prompts details The prompts used in Fig. 1a are as follows, from left to right, top to bottom.

- • Hyper-detailed macro photograph of a mechanical hummingbird crafted from gold filigree and sapphire gears, sipping nectar from a chrome rose; studio lighting, 200 mm macro lens, razor-sharp focus with creamy bokeh.
- • A photo of a bear made entirely of autumn leaves.
- • A fox wearing a suit and tie reading a newspaper at a café.
- • a tiny astronaut hatching from an egg on the moon

Raw image Pair pretrain Image pretrain

[Figure 15]

[Figure 16]

[Figure 17]

In cartoon style.

[Figure 18]

[Figure 19]

[Figure 20]

##### The man wears a joker hat.

[Figure 21]

[Figure 22]

[Figure 23]

The woman with a necklace.

Figure 5. Image editing ability with different pre-training method.

Table 10. UMM finetuning WISE results. Notation A⊕B denotes the result obtained by combining methods A and B.

###### METHOD Res. NFEs Cultural Time Space Biology Physics Chemistry Overall

OpenUni-L [50] 512 20×2 0.51 0.45 0.58 0.39 0.50 0.30 0.52 ⊕Image finetuning 512 20×2 0.46 0.52 0.66 0.49 0.51 0.29 0.49 ⊕Pair finetuning 512 20×2 0.63 0.58 0.74 0.57 0.71 0.44 0.62 ⊕Mix finetuning 512 20×2 0.60 0.58 0.70 0.51 0.64 0.46 0.59

Qwen-Image [49] 512 50×2 - - - - - - -

⊕Image finetuning 512 50×2 0.39 0.42 0.56 0.32 0.50 0.28 0.41 ⊕Pair finetuning 512 50×2 0.62 0.62 0.76 0.56 0.74 0.36 0.62 ⊕Mix finetuning 512 50×2 0.62 0.64 0.81 0.56 0.70 0.36 0.63

Qwen-Image [49] 1024 50×2 0.62 0.63 0.77 0.57 0.75 0.40 0.62 ⊕Image finetuning 1024 50×2 0.28 0.35 0.52 0.40 0.40 0.28 0.35 ⊕Pair finetuning 1024 50×2 0.63 0.63 0.77 0.62 0.72 0.37 0.63 ⊕Mix finetuning 1024 50×2 0.64 0.63 0.78 0.57 0.73 0.38 0.63

• A man sipping coffee on a sunny balcony filled with potted plants, wearing linen clothes and sunglasses, basking in the morning light.

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

A fast-moving team sport popular in Russia, known for its

A large animal, a symbol of national pride in Thailand

A very popular sport in the US with an oval shaped ball

intense physicality

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

The most popular sport in the country where Sao Paulo is

A sail-like structure, an architectural icon on Sydney's harbor

A fruit known for its unique shape, with a star-like cross section when sliced, a popular snack in Southeast Asia"

located

- Figure 6. Generation results of OpenUni-L before and after finetuning. The left one is the image generated by the original OpenUni-L, while the right one is generated by the OpenUni-L after finetuning.

[Figure 36]

Cyberpunk neon rooftop skyline above, rainy alley street below: upper half sleek skyscrapers with holographic ads, neon signs, pulsating lighting; lower half slick alley with puddles, graffiti, cables, wet pavement. Between:

a fire escape or balcony ledge with steam rising. Top: “Image-Only” in neon tubing letters, center “Masked MULTIMODAL” half neon / half metal, bottom “IOMM” in icy chrome or LED panels. Vivid neon pinks, teal

blues vs muted dark tones; textures: metal, glass, rain wetness, reflections; high contrast lighting, cinematic night.

[Figure 37]

[Figure 38]

with in the .

[Figure 39]

[Figure 40]

[Figure 41]

A British shorthair

wearing sunglasses

[Figure 42]

Paper artwork, layered paper, colorful Chinese

dragon surrounded by clouds.

[Figure 43]

A cloud in the shape of two bunnies playing with

a ball. The ball is made of clouds too.

[Figure 44]

A cute human-like sheep wearing a pink rabbit hairpin in cartoon style.

(a) Baseline Qwen-Image generation.

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Cyberpunk neon rooftop skyline above, rainy alley street below: upper half sleek skyscrapers with holographic

ads, neon signs, pulsating lighting; lower half slick alley with puddles, graffiti, cables, wet pavement. Between: a fire escape or balcony ledge with steam rising. Top: “Image-Only” in neon tubing letters, center “Masked MULTIMODAL” half neon / half metal, bottom “IOMM” in icy chrome or LED panels. Vivid neon pinks, teal

blues vs muted dark tones; textures: metal, glass, rain wetness, reflections; high contrast lighting, cinematic

night.

[Figure 51]

with in the .

[Figure 52]

[Figure 53]

A British shorthair

wearing sunglasses

Paper artwork, layered paper, colorful Chinese

dragon surrounded by clouds.

A cloud in the shape of two bunnies playing with

a ball. The ball is made of clouds too.

A cute human-like sheep wearing a pink rabbit hairpin in cartoon style.

(b) Our fine-tuned Qwen-Image generation.

- Figure 7. (a, b) Qualitative comparison between the original Qwen-Image model and our fine-tuned version. Our method enhances the model’s ability to generate images with richer visual detail and improved alignment to the textual prompt.

- • A cloud in the shape of two bunnies playing with a ball. The ball is made of clouds too.
- • Portrait of a noble samurai android wearing lacquered carbon-fiber armor and cherry-blossom patterns; Rembrandt lighting, 50 mm f/1.2, hyperreal pores and brushed metal textures.
- • A hot air balloon in the shape of a heart. Grand Canyon
- • A captivating photograph of an exquisite wooden dragon sculpture, skillfully carved with intricate details and realistic scales. The dragon is poised on a tree branch, its grand wings spread wide, revealing a mesmerizing woodland landscape below. The sky is painted with a symphony of soft blues and yellows, as the sun casts its final rays beyond the horizon. The dragon’s glass eyes lend it a lifelike presence.
- • Close-up portrait of a young woman with light skin and long brown hair, looking directly at the camera. Her face is

- illuminated by dramatic, slatted sunlight casting shadows across her features, creating a pattern of light and shadow. Her eyes are a striking green, and her lips are slightly parted, with a natural pink hue. The background is a soft, dark gradient, enhancing the focus on her face. The lighting is warm and golden.
- • A lone figure in dark robes ascends worn stone steps toward a glowing light in an ancient temple entrance. Ornate arches, lush greenery, and intricate carvings adorn the scene, evoking a mystical, high-fantasy atmosphere reminiscent of works by artists like Randy Vargas, with cinematic lighting and epic storytelling.
- • A whimsical scene featuring a plush toy bear wearing a blue sweater, positioned in the foreground, holding a butterfly on its raised arm. The bear is surrounded by a field of vibrant blue flowers, likely nemophila, creating a lush and colorful foreground. In the background, Mount Fuji rises majestically, its snow-capped peak sharply contrasting against a clear blue sky. The mountain is framed by fluffy white clouds and a line of dark green trees at its base. The butterfly, with its intricate black and orange wings, adds a touch of realism to the playful composition.
- • A candid midday portrait of a young East Asian woman with dark braided hair, laughing softly at the camera while cradling a steaming mug of coffee. She wears a tattered band t-shirt with a faded punk logo, frayed gray collar, and missing sleeve button. The background shows peeling floral wallpaper and a rusted folding chair beneath a window with harsh noon sunlight. Shot as a grainy film photograph with high contrast and sharp focus on her animated expression.
- • professional portrait photo of an anthropomorphic cat wearing fancy gentleman hat and jacket walking in autumn forest.

