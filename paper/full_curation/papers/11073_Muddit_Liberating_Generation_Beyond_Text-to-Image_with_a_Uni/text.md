# arXiv:2505.23606v5[cs.LG]26May2026

MUDDIT: LIBERATING GENERATION BEYOND TEXTTO-IMAGE WITH A UNIFIED DISCRETE DIFFUSION MODEL

Qingyu Shi∗, Jinbin Bai∗†, Zhuoran Zhao, Wenhao Chai, Kaidong Yu, Jianzong Wu, Yunhai Tong, Xiangtai Li‡, Xuelong Li, Shuicheng Yan‡

∗Equal Contribution, †Project Lead, ‡Corresponding Authors

Model: https://huggingface.co/MeissonFlow/Muddit Code: https://github.com/M-E-AGI-Lab/Muddit

ABSTRACT

Unified generation models aim to handle diverse tasks across modalities—such as text generation, image generation, and vision-language reasoning—within a single architecture and decoding paradigm. Autoregressive unified models suffer from slow inference due to sequential decoding, and non-autoregressive unified models suffer from weak generalization due to limited pretrained backbones. We introduce the second-generation Meissonic: Muddit, a unified discrete diffusion transformer that enables fast and parallel generation across both text and image modalities. Unlike prior unified diffusion models trained from scratch, Muddit integrates strong visual priors from a pretrained text-to-image backbone with a lightweight text decoder, enabling flexible and high-quality multimodal generation under a unified architecture. Empirical results show that Muddit achieves competitive or superior performance compared to significantly larger autoregressive models in both quality and efficiency. The work highlights the potential of purely discrete diffusion, when equipped with strong visual priors, as a scalable and effective backbone for unified generation.

1 INTRODUCTION

Multimodal generative models capable of handling both text and images have rapidly advanced, typically relying on large autoregressive (AR) Transformers, also known as large language models (LLMs) (Touvron et al., 2023). These unified models represent both modalities as token sequences and generate outputs in a left-to-right autoregressive manner. However, this sequential decoding imposes a major inference bottleneck. For instance, in early unified transformers (Sun et al., 2024), as illustrated in Fig. 1(a), generating a single image requires sampling thousands of visual tokens one at a time. Despite strong correlation among adjacent image tokens, each token prediction triggers a full network forward, resulting in significant redundant computation. As a result, inference becomes extremely slow and compute-intensive. We refer to this as the first “dark cloud” over current unified generative models. Moreover, AR decoding enforces a rigid generation order. This prevents speed-quality trade-offs or flexible conditional generation like inpainting without fine-tuning, which severely limits practical applicability in interactive or real-time scenarios. To mitigate these limitations, some hybrid approaches (Chen et al., 2025a; Pan et al., 2025; Chen et al., 2025b), adopt AR language models paired with diffusion-based image synthesis heads (Fig. 1(b)). However, these “glue” architectures fall short of true unification, as they lack a shared generative modeling paradigm across modalities.

Recent work like Dual-Diffusion (Li et al., 2024c) and Diffuse Everything (Rojas et al., 2025) (Fig. 1(c)) claims to unify modalities under discrete diffusion, but they ultimately rely on continu-

1Peking University, 2National University of Singapore, 3Princeton University

: jinbin.bai@u.nus.edu

(a) (b) (c) (d)

Image Image Text

Text & Image

Text

Text & Image

Continuous/ Discrete Diff.

Continuous Diff.

Discrete Diff.

Autoregressive (AR) AR Discrete Diff.

Text & Image

Text & Image

Text & Image

Text & Image

Figure 1: Four types of unified generative models. More details can be found in Sec. A.

ous diffusion for image generation. This fundamental mismatch in generative principles undermines their claim of true unification. UniDisc (Swerdlow et al., 2025a) (Fig. 1(d)), takes a more promising step by applying discrete diffusion1 over unified token spaces. This allows parallel refinement of text and image tokens, improving inference efficiency and enabling more flexible conditioning. However, the overall generation quality of UniDisc remains far from satisfactory. For example, it struggles to produce high-resolution 1024×1024 images, fails to match the fidelity of early diffusion models such as Stable Diffusion 1.5, and lacks support for vision-language reasoning tasks such as visual question answering (VQA). These limitations expose the second “dark cloud”: the absence of strong pre-trained discrete diffusion backbone models: Unlike established unified autoregressive models that leverage powerful pretrained large language models, current unified discrete diffusion models are typically trained from scratch on mixed-modality tokens, which limits both their generative fidelity and transferability. Without modular components carrying rich pixel-level priors, these models face generalization and scalability bottlenecks.

Taken together, the two dark clouds: inefficient autoregressive sampling and the lack of strong pretrained foundations, highlight the need for a new generation of unified models. In this work, we present the second-generation Meissonic: Muddit, a MaskGIT-style unified discrete diffusion transformer equipped with a lightweight text decoder. By combining the strengths of parallel discrete diffusion and semantically rich image priors from a pre-trained Meissonic text-to-image backbone (Bai et al., 2025b), Muddit enables scalable, efficient, and flexible sampling while significantly improving alignment and quality across modalities and various tasks such as high-resolution text-toimage synthesis, image-to-text synthesis, and visual question answering. We systematically detail the training objective of unified discrete diffusion models, the masking strategy, and the shared inference sampling strategy across three tasks. Finally, we conduct comprehensive evaluations with current popular unified models on several benchmarks, including GenEval, CIDEr, VQAv2, MME, and GQA, demonstrating Muddit’s superior performance and efficiency, validating that the unexplored purely discrete diffusion approach can rival, or even surpass, much larger autoregressive-based unified models. While concurrent unified generation models (Yang et al., 2025b) often build upon a language modeling prior—leveraging pretrained dLLMs as the backbone—we instead take a visualfirst approach. Muddit is built upon an image generation prior, offering a new path toward unifying vision and language tasks within a discrete diffusion framework. We hope that this work inspires a new trend for unified generative modeling, grounded in discrete diffusion, beyond the boundaries of traditional text-to-image synthesis (Bai et al., 2025b) and text synthesis (Nie et al., 2025; Inception, 2025).

2 METHOD

- 2.1 DISCRETE DIFFUSION WITH UNIFIED IMAGE AND TEXT PERSPECTIVE

In discrete diffusion, a sample x ∈ X is treated as a one-hot vector x, where X = {1,...,N}. For language models, N equals the vocabulary size. While for image models, N is the number of discrete image token IDs obtained from a tokenizer or VQ codebook. At each diffusion step, we stochastically corrupt the tokens, gradually transforming the data distribution into a maximally

1MaskGIT, MaskAR, RandomAR, and Discrete Diffusion share significant conceptual and practical overlaps, often differing only in decoding order or architectural nuances. We elaborate on their connections in the next section. While Meissonic (Bai et al., 2025b) follows the naming convention of MaskGIT (Chang et al., 2022), we standardize terminology in this paper by referring to all such models under the umbrella of Discrete Diffusion.

entropic categorical prior; the generative model then learns to invert this corruption. Following recent works (Lou et al., 2023; Bai et al., 2025b) that cast token corruption as a continuous–time Markov chain (CTMC) over the finite alphabet X, we let

dpt dt

= Qt pt, (1)

where pt ∈ RN+1 is the distribution of xt, and the time–dependent matrix Qt transports the data distribution p0 ≈ pdata to the maximally entropic “noise” distribution p1 = pstationary. We adopt the absorbing-state (masked) diffusion variant that has proved particularly effective in text modelling: every symbol can jump to a dedicated mask token m = (0,...,0

,1) but never leaves it, i.e. m is an

N

absorbing class. Forward posterior. Marginalizing x gives

##### q(xt | x) = Cat xt | αtx + (1 − αt)m . (2)

Cat(·) denotes a categorical distribution; it returns a one-hot token sampled from the probability vector inside the parentheses. αt ∈ [0,1] is the survival probability, i.e. the probability that an individual token has not yet been masked by time t. Thus xt equals the original clean token with probability αt and equals the mask token m with probability 1 − αt.

Reverse process. For any 0 < s < t < 1, the CTMC induces an analytic posterior

 

Cat(xs | xt), xt ̸= m, Cat xs |

(3)

q(xs | xt,x) =

(1 − αs)m + (αs − αt)x 1 − αt



, xt = m,

xt and xs are the corrupted tokens at times t and s (s < t). If xt is already a real vocabulary token (xt ̸= m) it stays unchanged going backwards; otherwise, when xt = m, the distribution over xs is a convex combination of the mask and the clean token x, weighted by their respective survival probabilities αs and αt.

Training Objective. We employ a masked-token predictor xθ(xt,αt) ≈ x, which leads to the continuous-time negative ELBO (Ou et al., 2024; Sahoo et al., 2024; Shi et al., 2024)

1

log xθ(xt,αt)·x dt , (4)

LNELBO = Eq(x

t|x)

- 0

αt′

- 1 − αt

where αt′ = dα

dt and x is the one-hot vector of ground truth. xθ(xt,αt) ∈ RN+1 is the model’s predicted categorical probability vector for the clean token given the corrupted input (xt,αt).

t

During generation, we start from an all-mask sequence (t = 1) and integrate the reverse CTMC towards t = 0, repeatedly replacing every masked position with the model’s categorical prediction. Because the corruption schedule and objective are identical for any discrete alphabet X, the same diffusion backbone unifies text and image generation. In the following section, we present Muddit, a unified framework that leverages discrete diffusion to model the generation tasks for both text and image jointly.

- 2.2 MUDDIT

- 2.2.1 UNIFIED ARCHITECTURE

As shown in Fig. 2, our architecture comprises a text encoder Etxt, image encoder Eimg, transformer generator G, sampler S, text decoder Dtxt, and image decoder Dimg. The generator G is a single MM-DiT model, following the dual-/single-stream design of FLUX (Labs, 2024). Importantly, the generator G is initialized from Meissonic (Bai et al., 2025b), which has been extensively trained for high-resolution text-to-image generation. This initialization brings in a strong pretrained image prior, capturing rich spatial structures and semantic correlations across image and text tokens, which significantly enhances sample quality and accelerates convergence in the multimodal setting. Consequently, the same MM-DiT predicts the masked tokens for both modalities, which produces a shared generator for text and image synthesis.

(a) Training

###### (b) T2I Inference (c) I2T Inference

Detokenizer

The Canadian astronaut lands on the moon.

[Figure 1]

###### VQ Dec

[Figure 2]

###### Text Dec

Text Dec

[Figure 3]

𝒕𝜶 𝒕 𝟏−𝜶

Re-weighted Cross-Entropy

MM-DiT

MM-DiT

MM-DiT

𝒕𝟎𝒍𝒐𝒈(𝑮𝒙⋅𝒙)

… ×	𝑇 steps

Re-weighted Cross-Entropy

… ×	𝑇 steps

[Figure 4]

Text Enc

MM-DiT

Mask

The Canadian astronaut lands on the moon.

MM-DiT

[Figure 5]

VQ Enc

VQ Enc

Text Enc

Mask

Text Enc

[Figure 6]

[Figure 7]

Tokenizer

The Canadian astronaut lands on the moon.

The Canadian astronaut lands on the moon.

The Canadian astronaut lands on the moon.

[Figure 8]

[Figure 9]

Mask token Image token Text token Trainable Frozen

Figure 2: The training and inference architecture of Muddit. (a) During training, we randomly mask tokens from one of the two modalities. MM-DiT is trained to predict the masked tokens using a re-weighted cross-entropy loss, which jointly optimizes both the MM-DiT backbone and a lightweight text decoder. (b) In text-to-image inference, we initialize the image latent features using all-masked tokens and iteratively predict each latent token via MM-DiT. (c) In image-to-text inference, we similarly initialize all text tokens as masked and generate the text through the same iterative decoding process. Specifically for VQA tasks, we append mask token IDs to the end of the question and predict all masked token IDs as the final answer.

To reduce the computational cost of high-resolution imagery and lengthy captions, we quantize both modalities into a compact discrete space. A pre-trained VQ-VAE acts as the image encoder Eimg, mapping pixels to codebook indices, while the CLIP text model, as Etxt, provides the text token embeddings. The MM-DiT predicts clean tokens in this shared space, which a lightweight linear head Dtxt converts back to text tokens.

- 2.2.2 UNIFIED TRAINING

Masking strategy. We model the forward posterior in Eq. 2 of both modalities using time-dependent hyperparameters αt, with the mask ratio defined as γt = 1 − αt. While BERT (Devlin, 2018) employs a fixed mask ratio of 15%, this setting is suitable for token completion but insufficient for generation. To support generative tasks, the design of γt must satisfy the following criteria:

- 1. γt must be a continuous function, bounded between 0 and 1, for t ∈ [0,1].
- 2. γt should monotonically decrease with respect to t, with boundary conditions γ0 → 0 (initially clean data) and γ1 → 1 (masking all tokens).

Several strategies for masking and sampling have been proposed to meet these criteria (Chang et al., 2022). We adopt a cosine scheduling strategy. During training, a timestep t ∈ [0,1] is sampled from a truncated arccos distribution, with the density function:

2 π

- 1

- 2 . (5)

(1 − (1 − t)2)−

γt =

During training, a mask ratio γt ∈ [0,1) is randomly sampled for each modality x0 (either image or text tokens), and the forward process (Eq. 2) is applied by randomly replacing clean tokens with

mask tokens to obtain xt.

Unified training objective. Let c denote the conditioning: the text embedding when synthesizing an image, or the image embedding when generating a caption. We randomly sample a mask ratio by Eq. 5. Then we corrupt the target sequence x0 (image or text tokens) with the CTMC described in Eq. 1 and train a single masked-token predictor G(xt,αt,c) to reconstruct x0. Both directions—text → image and image→ text—share the identical continuous-time negative ELBO

1

log G(xt,αt,c)·x dt , (6)

Lunified = Eq(x

t|x)

- 0

αt′

- 1 − αt

where all symbols are as in Eq. 4 but the G now receives the cross-modal condition c as an additional input. Key point: switching from text→ image to image→ text merely changes the conditioning signal c; the loss Eq. 6 itself is unchanged. This symmetry keeps optimization identical across tasks and allows us to train a single parameter set jointly for both generation directions. During inference we again start from an all-mask sequence (t=1) and integrate the reverse CTMC towards t=0, feeding in the desired condition c to obtain either an image or a sentence from the same diffusion backbone.

- 2.2.3 UNIFIED INFERENCE Sampling strategy. During inference, we apply the time-reversed posterior as defined in Eq. 3.

 

Cat(xs | xt), xt ̸= m, Cat xs |

(7)

S(G,xt,t) = pθ(xs | xt) =

(1 − αs)m + (αs − αt)G(xt,αt,c) 1 − αt



, xt = m,

where θ denotes the parameters of G, c is the multimodal condition, and αt in Eq. 5 is applied sequentially with t taking values 1, TT−1,..., T1 , where T is the total number of reverse steps. At each timestep t, Muddit predicts a fraction γt+ 1

− γt of the masked tokens by G and updates the

T

masked tokens xt by S , continuing iteratively until all masked tokens are recovered. This dynamic approach offers several advantages over autoregressive methods, which require the model to learn

conditional probabilities P(xi | x<i) based on a fixed token ordering. In contrast, random masking with a variable ratio enables the model to learn P(xi | xΛ), where Λ denotes an arbitrary subset of observed tokens. This flexibility is essential for parallel sampling, allowing multiple tokens to be predicted simultaneously rather than sequentially.

Our Muddit supports three tasks with a single generator G and sampler S: (i) text→ image, (ii) image → text (captioning), and (iii) visual–question answering (VQA). The only change across tasks is the conditioning source c provided to G; the diffusion process and guidance logic are shared.

- (i) Text→image. Given a text prompt tp∈T , the text encoder Etxt produces a text token embedding ctxt = Etxt(tp). Starting from a fully masked sequence x1, the generator produces logits

lt = G(xt,αt,ctxt), xt− 1

T

= S(lt,xt,t), (8)

for k = 1, TT−1,... T1 . After T steps we obtain visual tokens x0, which the image decoder Dimg converts to a pixel-space image I = Dimg(x0).

- (ii) Image→text. For captioning, an input image I ∈ I is tokenized by the image encoder Eimg: cimg = Eimg(I). The generator now conditions on the visual tokens while progressively decoding text:

lt = G(xt,αt,cimg), xt− 1

T

= S(lt,xt,t), (9) yielding a text token sequence x0, which Dtxt maps to a caption = Detokenize(Dtxt(x0)).

- (iii) Image+question→answer (VQA). For visual–question answering we supply both an image and a question: cimg = Eimg(I) and ctxt = Etxt(q). They are concatenated and fed to the generator, which outputs logits over answer tokens xk:

= S(lt,xt,t), (10) until the full answer a is produced and decoded by a = Detokenize(Dtxt(x0)). Classifier-free guidance. At each decoding step, we apply the same guidance rule, independent of modality:

lt = G(xt,αt,[cimg,ctxt]), xt− 1

T

##### lk ← G(zk,αk,c) + λ G(zk,αk,c) − G(zk,αk,cneg) , (11)

where zk (image or text tokens) is the partial target sequence, c is the positive condition (prompt, image, or image+question), cneg is the corresponding negative condition, and λ is the guidance scale. Because the loss, decoding schedule, and guidance operator are identical in all three scenarios—only the conditioning signal changes—our framework realises a genuinely unified multimodal generator.

- 3 EXPERIMENT 3.1 EXPERIMENTAL SETUP

Implementation details. We build Muddit on top of the open-source Meissonic models (Bai et al., 2025b). The MM-DiT backbone is initialized with pretrained weights, and a lightweight linear head is added as a text decoder. Following Meissonic, we adopt the CLIP (Radford et al., 2021) as text encoder and VQ-VAE as image encoder and decoder, keeping them entirely frozen throughout all experiments. To support discrete denoising, we append a special <mask> token to CLIP’s vocabulary for text masking, while the image mask token is inherited directly from Meissonic’s initialization. We observe that, even without training, the <mask> embedding can already be predicted into a coherent sentence during training. Therefore, for simplicity, we freeze the <mask> embedding. During training, we use a constant learning rate of 1×10−4 and a weight decay of 1×10−2. Gradient accumulation is applied in both pretraining and supervised fine-tuning, resulting in an effective batch size of 1024. We train on 16 H100 GPUs for 5 days. During inference, we adopt the default Meissonic configuration, using cosine masking scheduling, 64 sampling steps, and a classifier-free guidance (CFG) scale of 9.0 and 1.5 for text-to-image and image-to-text generation, respectively.

Training data. We train Muddit in two stages using a combination of publicly available and internal datasets, including JourneyDB (Pan et al., 2023), LAION-Art (Schuhmann et al., 2022), CC12M (Changpinyo et al., 2021), and others. The final dataset is filtered based on aesthetic score, resolution, and aspect ratio, resulting in approximately 10 million image–text pairs. Both stages are optimized with the unified training objective defined in Eq. 6. Below, we describe the datasets and settings for each stage in detail.

- 1. Pretraining. We pretrain Muddit for 100K steps with a batch size of 1024, using the unified objective across both modalities. Text inputs are truncated to a maximum of 77 tokens, and images are resized to 512×512. The pretraining corpus consists of 8 million image–text pairs, recaptioned using Qwen2.5-VL-3B for improved consistency. Each batch is evenly split between text-to-image and image-to-text samples to enable joint training in both directions.
- 2. Instruction tuning. After pretraining, we fine-tune the model on a combination of 1 million instruction-following samples, including LLaVA-Instruct-150K, ALLaVA, SA-1B, and the VQAv2 training set. During this stage, only the answer portion of each prompt is masked. Additionally, we construct a curated dataset of 1 million high-quality image–text pairs to support multi-task training on VQA and image generation. Following the task instructions embedded in each sample, Muddit learns to produce long-form answers, concise replies, and image captions via task-specific prompting.

We present both quantitative and qualitative results for the T2I and I2T tasks in the following sections. Additional experiments and ablation studies are provided in the Appendix.

- 3.2 TEXT-TO-IMAGE GENERATION

Quantitative results. Following prior work, we evaluate our 512 × 512 model on GenEval (Ghosh et al., 2024) after supervised fine-tuning in Tab. 1. Muddit attains an overall accuracy of 0.61, surpassing prior discrete diffusion models such as Monetico (0.44) and Meissonic (0.54), and closely matching Stable Diffusion 3 (0.62) with only 1B parameters. It further shows strong compositional reasoning (0.72 on “Two Objects”, 0.54 on “Counting”), and benefits from joint multimodal training, which enhances T2I performance. These results demonstrate the effectiveness of Muddit as the first unified discrete diffusion model for both text and image modalities.

Qualitative results. We present diverse generations from our model conditioned on rich textual prompts in Fig. 3. The outputs exhibit strong text-image alignment, capturing fine details in both realistic and imaginative scenes. Our model effectively renders complex structures, lighting, and textures across various domains.

- 3.3 IMAGE-TO-TEXT GENERATION

We present a comprehensive comparison of our model Muddit against other multimodal models across four benchmarks: MS-COCO (image captioning) (Lin et al., 2014), VQAv2 (Antol et al.,

[Figure 10]

Human: A medieval town nestled on an expansive bay, cloaked in a blanket of snow and ice. Icy winds whip through cobblestone streets lined with half-timbered buildings. their stone walls adorned with intricate carvings. Steam rises from cozy hearths as villagers huddle around fires, their faces illuminated by lanterns. The sky is a deep indigo, flecked with stars as a full moon casts an ethereal glow over the scene.

[Figure 11]

Human: A towering, robotic whale sculpted with gears and cogs stands amidst a bustling futuristic cityscape bathed in neon light. Its colossal form is detailed with rusted brass accents and gleaming chrome fins that ripple as it glides across the shimmering water. Its eye stalks glow with an array of glowing LEDs.

[Figure 12]

Human: An organic, bioluminescent pavilion, crafted from gleaming chrome and polished metal, rises from the tranquil depths of a misty ocean. Sunbeams pierce the swirling fog, illuminating shimmering structures in intricate curves and forms. The water below reflects an ethereal sheen of ambient light, creating a hazy, captivating scene.

[Figure 13]

Human: A high-quality photograph capturing the majestic Plum Bayou Mound Architecture in a vibrant city of anthropomorphic mammals. The architecture is intricate and detailed, with towering structures made of natural materials like stone and wood, set against a backdrop of lush greenery and vibrant blue water reflecting the afternoon sun.

[Figure 14]

Human: A slice of creamy cheesecake, adorned with a vibrant yellow layer of lemon jelly and flecks of zesty lemon zest. The surface is smooth and inviting, captured in professional food photography with ultrarealistic detail. The lighting is soft, almost cinematic, highlighting the intricate texture and high definition sharpness.

[Figure 15]

Human: A 3D digitally rendered portrait in the style of "glass fragments" art. The face is depicted with a fractured, cracked surface reminiscent of shattered glass, using muted black and white tones. A split-toned effect adds depth and drama, highlighting the emotional intensity of the subject's expression. The image is ultra-high definition (4K) and rendered in Photoshop.

Figure 3: Samples of Text-to-Image Generation by Muddit.

Human: How many people are there in this image?

Human: What is the color of this mug?

[Figure 16]

[Figure 17]

Assistant: The color of the cup is white with blue picture on it.

Assistant: There are two people in this image.

Human: What is this statue made of?

Human: What is on the table?

[Figure 18]

[Figure 19]

Assistant: There are a

Assistant: The statue is made of marble.

cup, saucer, and a vase of flowers on the white table.

Figure 4: Samples of Visual Question Answering by Muddit.

2015), MME (Fu et al., 2023), GQA (Hudson & Manning, 2019), and MMMU (Yue et al., 2024) in Tab. 2. Notably, Muddit is the first unified model to employ discrete diffusion for both text-to-image and image-to-text generation, demonstrating that this approach is highly competitive.

Quantitative comparison. Muddit demonstrates strong performance across benchmarks despite its compact size (1B parameters). It achieves a CIDEr score of 59.9 on MS-COCO, surpassing diffusion-based baselines such as D-DiT (56.2). On VQAv2, it attains 68.2% accuracy, outperforming Show-O and D-DiT. Moreover, it reaches 1107.4 on MME and 57.5 on GQA, underscoring its competitiveness across multimodal reasoning tasks. These results highlight the effectiveness of Muddit as a unified diffusion-based model that balances efficiency with high-quality task performance.

Table 1: Evaluation of text-to-image generation performance on the GenEval (Ghosh et al., 2024).

Color ↑ Single Two Attribution

Objects ↑ Counting ↑Colors ↑Position ↑

Text Gen Arch

Image Gen Arch

Params (B)

Model

Overall ↑

PixArt-α (Chen et al., 2024b) - Diffusion 0.6 0.48 0.98 0.50 0.44 0.80 0.08 0.07 SD 2.1 (Rombach et al., 2022) - Diffusion 0.9 0.50 0.98 0.51 0.44 0.85 0.07 0.17

- DALL-E 2 (Ramesh et al., 2022) - Diffusion 6.5 0.52 0.94 0.66 0.49 0.77 0.10 0.19 SDXL (Podell et al., 2023) - Diffusion 2.6 0.55 0.98 0.74 0.39 0.85 0.15 0.23
- DALL-E 3 (Betker et al., 2023) - Diffusion - 0.67 0.96 0.87 0.47 0.83 0.43 0.45 SD 3 (Esser et al., 2024) - Diffusion 2 0.62 0.98 0.74 0.63 0.67 0.34 0.36

LWM (Liu et al., 2024c) AR AR 7 0.47 0.93 0.41 0.46 0.79 0.09 0.15 SEED-X (Ge et al., 2024) AR AR 17 0.49 0.97 0.58 0.26 0.80 0.19 0.14 Chameleon (Team, 2024) AR AR 7 0.39 - - - - - Show-O (Xie et al., 2024) AR Discrete Diff. 1.3 0.68 0.98 0.80 0.66 0.84 0.31 0.50 Transfusion (Zhou et al., 2024) AR Diffusion 8 0.67 - - - - - D-DiT (Li et al., 2024b) Discrete Diff. Diffusion 2 0.65 0.97 0.80 0.54 0.76 0.32 0.50 Monetico (512 × 512) (Bai et al., 2025b) - Discrete Diff. 1 0.44 0.92 0.48 0.26 0.78 0.06 0.13 Meissonic (1024 × 1024) (Bai et al., 2025b) - Discrete Diff. 1 0.54 0.99 0.66 0.42 0.86 0.10 0.22 UniDisc (512 × 512) (Swerdlow et al., 2025b)Discrete Diff.Discrete Diff. 1.4 0.42 0.92 0.47 0.15 0.67 0.13 0.19 Muddit (512 × 512) Discrete Diff.Discrete Diff. 1 0.61 0.98 0.72 0.54 0.82 0.19 0.41

[Figure 20]

[Figure 21]

Assistant: The image depicts a man standing in water, holding a black dog and smiling. The man is wearing a yellow and wet jacket, and he has a happy expression. The background consists of a calm body of water with possibly a lake.

Assistant: The image depicts two individuals standing in a lush, forested area. the person on the left is older and wearing a light blue shirt and dark pants, while the person on the right is dressed in a suit with a shirt and tie. both appear to be engaged in a discussion or discussing something on the documents, with the person on the left holding a tablet.

[Figure 22]

Assistant: The image depicts a group of sheep walking along a road near a body of water, possibly a lake or a body of water water, possibly a sea or sea. the road is lined with small rocks and a scattered vegetation, and a road is visible on the side of the road. in the background, there are rolling hills and a calm water with a fence.

[Figure 23]

Assistant: The image depicts a row of hot motorcycles parked in a parking lot. the motorcycle is a white model with a black and white color scheme, identifiable by its distinctive design and branding on the side. the motorcycle appears to be a modern model model, as evidenized by the branding and branding visible on the side. the background includes a residential building, some trees.

[Figure 24]

Assistant: The image depicts a brown horse standing amidst a lush, green forest. the horse appears to be a thoroughbred horse, characterized by its dense features and tail visible. it is equipped with a bridle and a decorative tag on its forehead. the background is filled with dense foliage and small plants, suggesting a natural, foreforested environment. the overall scene is serene.

Figure 5: Samples of Image-to-Text Generation by Muddit.

Qualitative results. We present example captions generated by our model across diverse scenarios in Fig. 5, including humans, animals, vehicles, and natural landscapes. The model demonstrates strong visual grounding and fine-grained descriptive ability, accurately capturing attributes such as clothing, expressions, background context, and object relationships. Fig. 4 illustrates our model’s ability to accurately answer visual questions across various domains, including object counting, color recognition, material identification, and compositional reasoning.

- 3.4 ABLATION STUDY AND ANALYSIS

Analysis of the inference timesteps. As shown in Tab. 5, performance generally improves with more diffusion steps, plateauing around T = 32. GenEval and CIDEr see large gains from T = 8 to T = 32, with diminishing returns afterward. VQAv2 remains stable across timesteps, indicating

- Table 2: Evaluation of image captioning, visual question answering on multimodal benchmarks.

Model

Params (B)

Text Gen Arch

Image Gen Arch

MS-COCO CIDEr ↑

VQAv2 Acc. ↑

MME Acc. ↑

GQA Acc. ↑

MMMU Acc. ↑ InternVL-2.0 (Chen et al., 2025e) 8 AR - - - 1648.1 61.0 49.3 LLaVA-Next (Liu et al., 2024d) 13 AR - - 82.8 1575.0 65.4 36.2 BLIP-2 (Li et al., 2023) 13 AR - - 65.0 1293.8 41.0 34.4 QWEN-VL (Bai et al., 2023) 7 AR - - 78.2 1487.5 57.5 35.9 OpenFlamingo (Awadalla et al., 2023) 9 AR - 65.5 43.5 - - 28.7 Flamingo (Alayrac et al., 2022) 9 AR - 79.4 51.8 - - Chameleon (Team, 2024) 7 AR AR 18.0 - - - LWM (Liu et al., 2024c) 7 AR AR - 55.8 - - Show-O (256×256) (Xie et al., 2024) 1.3 AR Discrete Diff. - 64.7 1014.9 54.2 Show-O (512×512) (Xie et al., 2024) 1.3 AR Discrete Diff. - 69.4 1097.2 58.0 27.4 Transfusion (Zhou et al., 2024) 7 AR Diffusion 29.0 - - - D-DiT (256×256) (Li et al., 2024b) 2 Discrete Diff. Diffusion - 59.5 897.5 55.1 D-DiT (512×512) (Li et al., 2024b) 2 Discrete Diff. Diffusion 56.2 60.1 1124.7 59.2 UniDisc (Swerdlow et al., 2025b) 1.4 Discrete Diff. Discrete Diff. 46.8 - - - -

Muddit (512×512) 1 Discrete Diff. Discrete Diff. 59.9 68.2 1107.4 57.5 27.6 Muddit (1024×1024) 1 Discrete Diff. Discrete Diff. 60.1 70.2 1139.2 57.8 28.7

- Table 3: Impact of text loss weight. We apply the same text loss weight during both pretraining and instruction tuning.

Table 4: Effect of joint training. We denote text-to-image as T2I and image-to-text as I2T, respectively.

Benchmark 0.2 0.4 0.6 0.8 1.0

GenEval 60.1 60.5 61.6 60.8 58.3 MS-COCO 51.4 52.1 59.9 58.8 59.4 VQAv2 62.7 66.2 68.2 68.4 69.2

Benchmark T2I only I2T only Joint training

GenEval 59.3 28.3 61.6 MS-COCO - 60.1 59.9 VQAv2 - 69.1 68.2

that fewer steps suffice for discriminative tasks. Overall, a moderate number of steps provides a good balance between accuracy and efficiency.

Analysis of the text loss weight. As shown in Tab. 3, moderate text loss weights (around 0.6) yield the best overall performance. CIDEr and GenEval peak near this value, suggesting that both insufficient and excessive text weighting can harm generation quality. VQAv2 continues to improve with stronger text supervision but begins to plateau beyond 0.6. Overall, while discriminative tasks benefit from heavier textual guidance, generative tasks require a balanced mix of visual and textual signals—highlighting the importance of grounding language in multimodal learning.

Table 5: Performance across different diffusion timesteps.

Sample steps GenEval CIDEr VQAv2

T=8 51.6 43.6 53.9 T=16 58.5 59.3 57.4 T=24 59.3 59.4 62.3 T=32 61.9 59.7 65.4 T=40 61.7 60.1 66.8 T=64 61.1 59.9 68.2

#### Analysis of joint training. Joint optimization

over both text-to-image (T2I) and image-to-text (I2T) objectives is essential. As shown in Tab. 4, joint training yields the highest GenEval score, outperforming both T2I-only and I2T-only variants. Notably, I2T-only causes GenEval to drop sharply from 61.6 to 28.3—more than a twofold decrease—while MS-COCO CIDEr remains nearly unchanged and VQAv2 declines only slightly. These results show that separating the objectives severely weakens cross-modal integration, underscoring the need for unified optimization to maintain strong multimodal coherence.

- 3.5 THE SCALABILITY OF MUDDIT

To demonstrate the scalability of our approach, we curate roughly 10 million image–text pairs from LAION-ART (Schuhmann et al., 2022), JourneyDB (Pan et al., 2023), CC12M (Changpinyo et al.,

- 2021) and internal datasets. We filter out samples with an aesthetic score below 7, a height or width under 512 pixels, or an aspect ratio above 2. All images are re-captioned using Qwen2.5-VL 7B (Bai et al., 2023). We pretrain Muddit on this dataset with a batch size of 512 and a resolution of 1024, applying random masking to both image and text modalities. The image and text loss weights are set to 1.0 and 0.3, respectively. Training runs for 100K steps.

Table 6: Training hyperparameters across different training stages.

Hyperparameters Stage-I (Pre-training) Stage-II (Instruction-tuning)

Learning Rate 1.0 × 10−4 1.0 × 10−4 LR Scheduler Constant Constant Weight Decay 0.01 0.01 Max Gradient Norm 10.0 10.0 Optimizer AdamW (β1 = 0.9, β2 = 0.999) Batch Size 512 512 Training Steps 100K 15K Training GPUs 16×H100 16×H100

Gen. Resolution 1024 1024 Under. Resolution 1024 1024

Table 7: Quantitative comparison with other unified models.

Model Params Base model Architecture Data scale Geneval w/ TTS VQAv2 MME MMMU

Lumina-DiMOO (Xin et al., 2025b) 8B LLaDA Discrete Diff. 80M 0.92 – 1534.2 58.6 MMaDA (512 × 512) (Yang et al., 2025a) 8B LLaDA Discrete Diff. Unknown 0.66 76.7 1410.7 30.2

Show-O (512×512) (Xie et al., 2024) 1.3B Phi-1.5 AR + Discrete Diff 35M – 69.4 1097.2 27.4 D-DiT (512×512) (Li et al., 2024b) 2B SD3-medium Discrete Diff. + Diff. 40M – 60.1 1124.7 – Muddit (512×512) 1B Meissonic Discrete Diff. 10M 0.64 68.2 1107.4 27.6 Muddit (1024×1024) 1B Meissonic Discrete Diff 16M 0.67 70.2 1139.2 28.7

For instruction tuning, we collect about 6M samples from LLAVA-Instruct-150K (Liu et al., 2024e), ALLaVA LAION (Chen et al., 2024a), SA-1B (Kirillov et al., 2023), ART500K (Mao et al., 2017), ScienceQA (Lu et al., 2022), Chart2Text (Kantharaj et al., 2022), and VQAv2 (Antol et al., 2015). Muddit is then trained with a batch size of 512 at a resolution of 1024, with masking applied only to the answer text. We also add a 2M high-quality image dataset for high-quality fine-tuning. Further training configurations are provided in Tab. 6. All experiments are conducted on 16 H100 GPUs.

We evaluate the scaled Muddit model against other comparably sized unified models and state-ofthe-art unified discrete diffusion models (Xin et al., 2025b; Yang et al., 2025a), as shown in Tab. 7. Across established benchmarks, Muddit exhibits consistent improvements in both image generation and image understanding, empirically validating the scalability of our model. Furthermore, we compare Muddit with unified models of similar parameter sizes, all of which rely on hybrid architectures. Despite being trained on substantially less data, Muddit achieves superior performance.

We attribute this data efficiency to two key factors. First, our visual prior naturally maintains strong text-following capability for text-to-image generation, enabling robust alignment between image and text modalities. From the perspective of unified modeling, we prioritize cross-modal alignment over isolated single-modality ability, which allows Muddit to reach higher performance with less training data. Second, Muddit adopts a fully unified modeling paradigm: the model learns by predicting mask tokens based on context across all tasks (text-to-image and image-to-text). In contrast, hybrid architectures must simultaneously handle next-token prediction alongside velocity or mask prediction, and often introduce additional special tokens (e.g., < soi >, < eoi >), which increases architectural complexity and hinders optimization.

- 4 CONCLUSION

In this work, we present the second-generation Meissonic: Muddit, a unified generative framework that employs discrete diffusion to bridge text and image modalities. By unifying image and text generation within a single model, Muddit demonstrates strong performance across text-to-image, image-to-text, and VQA tasks. Notably, it matches or outperforms the capabilities of significantly larger autoregressive models, while enabling fast, parallel inference. Our results validate the effectiveness of discrete diffusion as a general-purpose modeling strategy and highlight its potential to serve as a scalable backbone for future multimodal systems that are equipped with strong visual priors.

Acknowledgement. This work is supported by the National Key Research and Development Program of China (No. 2023YFC3807600) and the National Natural Science Foundation of China under Grant No. 62320106007. In addition, this work is supported in part by NUS Start-up Grant A-0010106-00-00.

REFERENCES

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Kar´en Simonyan. Flamingo: a visual language model for few-shot learning. In Adv. Neural Inform. Process. Syst., 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ 960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html.

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pp. 2425–2433, 2015.

Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, and Ludwig Schmidt. Openflamingo: An opensource framework for training large autoregressive vision-language models, 2023. URL https: //arxiv.org/abs/2308.01390.

Jinbin Bai, Yu Lei, Hecong Wu, Yuchen Zhu, Shufan Li, Yi Xin, Xiangtai Li, Molei Tao, Aditya Grover, and Ming-Hsuan Yang. From masks to worlds: A hitchhiker’s guide to world models. arXiv preprint arXiv:2510.20668, 2025a.

Jinbin Bai, Tian Ye, Wei Chow, Enxin Song, Qing-Guo Chen, Xiangtai Li, Zhen Dong, Lei Zhu, and Shuicheng Yan. Meissonic: Revitalizing masked generative transformers for efficient highresolution text-to-image synthesis. ICLR, 2025b.

Jinbin Bai, Yixuan Li, Yuchen Zhu, Yi Xin, Qingyu Shi, Aosong Feng, Xiaohong Liu, Molei Tao, Jianru Xue, Xiangtai Li, et al. Prism: Efficient test-time scaling via hierarchical search and selfverification for discrete diffusion language models. arXiv preprint arXiv:2602.01842, 2026.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023. URL https://arxiv.org/abs/2308.12966.

Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254, 2021.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science., 2:3, 2023.

Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11315–11325, 2022.

Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12M: Pushing webscale image-text pre-training to recognize long-tail visual concepts. In CVPR, 2021.

Guiming Hardy Chen, Shunian Chen, Ruifei Zhang, Junying Chen, Xiangbo Wu, Zhiyi Zhang, Zhihong Chen, Jianquan Li, Xiang Wan, and Benyou Wang. Allava: Harnessing gpt4v-synthesized data for a lite vision-language model, 2024a.

Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, Le Xue, Caiming Xiong, and Ran Xu. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset, 2025a. URL https: //arxiv.org/abs/2505.09568.

Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In ICLR, 2024b.

Liang Chen, Shuai Bai, Wenhao Chai, Weichu Xie, Haozhe Zhao, Leon Vinci, Junyang Lin, and Baobao Chang. Multimodal representation alignment for image generation: Text-image interleaved control is easier than you think. arXiv preprint arXiv:2502.20172, 2025b.

Sixiang Chen, Jinbin Bai, Zhuoran Zhao, Tian Ye, Qingyu Shi, Donghao Zhou, Wenhao Chai, Xin Lin, Jianzong Wu, Chao Tang, et al. An empirical study of gpt-4o image generation capabilities. arXiv preprint arXiv:2504.05979, 2025c.

Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025d.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang, Qingyun Li, Yimin Ren, Zixuan Chen, Jiapeng Luo, Jiahao Wang, Tan Jiang, Bo Wang, Conghui He, Botian Shi, Xingcheng Zhang, Han Lv, Yi Wang, Wenqi Shao, Pei Chu, Zhongying Tu, Tong He, Zhiyong Wu, Huipeng Deng, Jiaye Ge, Kai Chen, Kaipeng Zhang, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling, 2025e. URL https://arxiv.org/abs/2412.05271.

Jacob Devlin. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.

Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, et al. Dreamllm: Synergistic multimodal comprehension and creation. arXiv preprint arXiv:2309.11499, 2023.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024.

Lijie Fan, Tianhong Li, Siyang Qin, Yuanzhen Li, Chen Sun, Michael Rubinstein, Deqing Sun, Kaiming He, and Yonglong Tian. Fluid: Scaling autoregressive text-to-image generative models with continuous tokens. arXiv preprint arXiv:2410.13863, 2024.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.

Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396, 2024.

Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems (NeurIPS), 36, 2024.

Jianyuan Guo, Hanting Chen, Chengcheng Wang, Kai Han, Chang Xu, and Yunhe Wang. Vision superalignment: Weak-to-strong generalization for vision foundation models. arXiv preprint arXiv:2402.03749, 2024.

Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll´ar, and Ross Girshick. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), pp. 16000–16009, 2022.

Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6700–6709, 2019.

Inception. Inception labs. https://www.inceptionlabs.ai/, Feb 2025. Accessed: 202505-16.

Shankar Kantharaj, Rixie Tiffany Leong, Xiang Lin, Ahmed Masry, Megh Thakkar, Enamul Hoque, and Shafiq Joty. Chart-to-text: A large-scale benchmark for chart summarization. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 4005–4023. Association for Computational Linguistics, 2022.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chlo´e Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Doll´ar, and Ross B. Girshick. Segment anything. In ICCV, 2023.

Black Forest Labs. Announcing black forest labs, 2024. https://blackforestlabs.ai/ announcing-black-forest-labs/.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. BLIP-2: bootstrapping languageimage pre-training with frozen image encoders and large language models. In ICML, volume 202 of Proceedings of Machine Learning Research, pp. 19730–19742. PMLR, 2023. URL https: //proceedings.mlr.press/v202/li23q.html.

Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems, 37: 56424–56445, 2024a.

Zijie Li, Henry Li, Yichun Shi, Amir Barati Farimani, Yuval Kluger, Linjie Yang, and Peng Wang. Dual diffusion for unified image generation and understanding, 2024b. URL https://arxiv. org/abs/2501.00289.

Zijie Li, Henry Li, Yichun Shi, Amir Barati Farimani, Yuval Kluger, Linjie Yang, and Peng Wang. Dual diffusion for unified image generation and understanding. arXiv preprint arXiv:2501.00289, 2024c.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In ECV, 2014.

Dongyang Liu, Shitian Zhao, Le Zhuo, Weifeng Lin, Yu Qiao, Hongsheng Li, and Peng Gao. Lumina-mgpt: Illuminate flexible photorealistic text-to-image generation with multimodal generative pretraining, 2024a. URL https://arxiv.org/abs/2408.02657.

Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with ringattention. arXiv preprint arXiv:2402.08268, 2024b.

Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with ringattention. arXiv preprint, 2024c.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024d. URL https:// llava-vl.github.io/blog/2024-01-30-llava-next/.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. NeurIPS, 2024e.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023.

Pan Lu, Swaroop Mishra, Tony Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In NeurIPS, 2022.

Xinyin Ma, Runpeng Yu, Gongfan Fang, and Xinchao Wang. dkv-cache: The cache for diffusion language models. arXiv preprint arXiv:2505.15781, 2025.

Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Liang Zhao, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. arXiv preprint arXiv:2411.07975, 2024.

Hui Mao, Ming Cheung, and James She. Deepart: Learning joint representations of visual arts. In Proceedings of the 25th ACM international conference on Multimedia, pp. 1183–1191. ACM, 2017.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025.

OpenAI. Addendum to gpt-4o system card: 4o image generation, 2025. URL https://openai. com/index/gpt-4o-image-generation-system-card-addendum/. Accessed: 2025-04-02.

Jingyang Ou, Shen Nie, Kaiwen Xue, Fengqi Zhu, Jiacheng Sun, Zhenguo Li, and Chongxuan Li. Your absorbing discrete diffusion secretly models the conditional distributions of clean data. arXiv preprint arXiv:2406.03736, 2024.

Junting Pan, Keqiang Sun, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, Jifeng Dai, Yu Qiao, and Hongsheng Li. Journeydb: A benchmark for generative image understanding, 2023.

Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, et al. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.

Kevin Rojas, Yuchen Zhu, Sichen Zhu, Felix X-F Ye, and Molei Tao. Diffuse everything: Multimodal diffusion models on arbitrary state spaces. arXiv preprint arXiv:2506.07903, 2025.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In CVPR, 2022.

Subham Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems, 37:130136–130184, 2024.

Christoph Schuhmann, Richard Vencu Beaumont, Romain Gordon Vencu, Clayton Coombes, Arun Katta, Robin Kaczmarczyk, and Jenia Jitsev. Laion-5b: An open large-scale dataset for training next-generation image-text models. https://laion.ai/blog/laion-5b/, 2022. Accessed: 2025-09-25.

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis Titsias. Simplified and generalized masked diffusion for discrete data. Advances in neural information processing systems, 37: 103131–103167, 2024.

Qingyu Shi, Size Wu, Jinbin Bai, Kaidong Yu, Yujing Wang, Yunhai Tong, Xiangtai Li, and Xuelong Li. Rectok: Reconstruction distillation along rectified flow. arXiv preprint arXiv:2512.13421, 2025.

Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.

Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative pretraining in multimodality. arXiv preprint arXiv:2307.05222, 2023.

Alexander Swerdlow, Mihir Prabhudesai, Siddharth Gandhi, Deepak Pathak, and Katerina Fragki-

- adaki. Unified multimodal discrete diffusion. arXiv preprint arXiv:2503.20853, 2025a.

Alexander Swerdlow, Mihir Prabhudesai, Siddharth Gandhi, Deepak Pathak, and Katerina Fragki-

- adaki. Unified multimodal discrete diffusion. arXiv preprint arXiv:2503.20853, 2025b. doi: 10.48550/arXiv.2503.20853.

Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.

Shengbang Tong, David Fan, Jiachen Zhu, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, and Zhuang Liu. Metamorph: Multimodal understanding and generation via instruction tuning. arXiv preprint arXiv:2412.14164, 2024.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Chunwei Wang, Guansong Lu, Junwei Yang, Runhui Huang, Jianhua Han, Lu Hou, Wei Zhang, and Hang Xu. Illume: Illuminating your llms to see, draw, and self-enhance. arXiv preprint arXiv:2412.06673, 2024a.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024b.

Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024.

Jinheng Xie, Weijia Mao, Zechen Bai, David Junhao Zhang, Weihao Wang, Kevin Qinghong Lin, Yuchao Gu, Zhijie Chen, Zhenheng Yang, and Mike Zheng Shou. Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528, 2024.

Yi Xin, Siqi Luo, Qi Qin, Haoxing Chen, Kaiwen Zhu, Zhiwei Zhang, Yangfan He, Rongchao Zhang, Jinbin Bai, Shuo Cao, et al. dmllm-tts: Self-verified and efficient test-time scaling for diffusion multi-modal large language models. arXiv preprint arXiv:2512.19433, 2025a.

Yi Xin, Qi Qin, Siqi Luo, Kaiwen Zhu, Juncheng Yan, Yan Tai, Jiayi Lei, Yuewen Cao, Keqi Wang, Yibin Wang, et al. Lumina-dimoo: An omni diffusion large language model for multi-modal generation and understanding. arXiv preprint arXiv:2510.06308, 2025b.

Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada:

- Multimodal large diffusion language models. arXiv preprint arXiv:2505.15809, 2025a.

Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada:

- Multimodal large diffusion language models. arXiv preprint arXiv:2505.15809, 2025b.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In CVPR, 2024.

Chuyang Zhao, Yuxing Song, Wenhao Wang, Haocheng Feng, Errui Ding, Yifan Sun, Xinyan Xiao, and Jingdong Wang. Monoformer: One transformer for both diffusion and autoregression. arXiv preprint arXiv:2409.16280, 2024.

Mengyu Zheng, Yehui Tang, Zhiwei Hao, Kai Han, Yunhe Wang, and Chang Xu. Adapt without forgetting: Distill proximity from dual teachers in vision-language models. In European Conference on Computer Vision, pp. 109–125. Springer, 2024.

Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024.

## APPENDIX

APPENDIX OVERVIEW

This appendix provides additional discussions, results, and analyses to complement the main paper. It is organized as follows:

- • Related Work (Sec. A): We review unified multimodal models for understanding and generation, with a focus on autoregressive and diffusion-based paradigms, as well as recent advances in masked image modeling.
- • Additional Qualitative Results (Sec. B): We present extended visualizations for several tasks, including image captioning, text-to-image generation, visual question answering, and image-guided text editing.
- • Additional Experimental Results (Sec. C): We present more experimental results.
- • Additional Ablation Studies (Sec. D): We present extended ablation studies.
- • Inference Time Analysis (Sec. E): We analyze inference efficiency by comparing autoregressive decoding with discrete diffusion, providing FLOPs complexity and speed benchmarks.
- • Generated Results Step by Step (Sec. F): We illustrate the reverse discrete diffusion process in detail, showing intermediate decoding steps and examples of progressive generation.
- • Discussion (Sec. G): We reflect on the limitations of our approach and its broader impacts, including potential applications and risks of misuse.
- • Use of Large Language Models (Sec. H): We clarify the role of large language models during paper preparation.

- A RELATED WORK

- A.1 UNIFIED MODELS FOR GENERATION AND UNDERSTANDING

The success of LLMs in language modeling has inspired efforts to extend unified generation to multimodal domains. However, the divergence between autoregressive and diffusion-based paradigms presents fundamental architectural trade-offs. Autoregressive models naturally handle language, and several works (Sun et al., 2023; Wang et al., 2024a; Tong et al., 2024; Ge et al., 2024; Dong et al., 2023; Chen et al., 2025b) extend this by connecting vision modules to LLMs via adapters or instruction tuning, with LLMs serving as planning modules that produce intermediate representations for image generation. While effective to some extent, these paradigms often exhibit limited interaction between text and image modalities and struggle with content consistency, particularly in imageto-image generation and complex instruction-based synthesis. To address these limitations, recent research explores unified generation models that integrate understanding and generation within a single architecture. We categorize these into four major paradigms (see Fig. 1):

Fully Autoregressive: Both text and image are tokenized into discrete sequences and modeled with an AR Transformer (Liu et al., 2024b; Team, 2024; Wu et al., 2024; Wang et al., 2024b; Chen et al., 2025d; Liu et al., 2024a; Guo et al., 2024; Zheng et al., 2024). These models achieve strong cross-modal generation but suffer from high latency due to sequential decoding.

Text AR, Image Diffusion: LLMs generate text tokens while image synthesis is delegated to pretrained continuous diffusion backbones (Zhou et al., 2024; Zhao et al., 2024; Ma et al., 2024) or discrete diffusion (Xie et al., 2024). Though visually strong, these models are not truly unified, as they rely on separate architectures and token spaces.

Image Diffusion, Text Discrete Diffusion: Emerging models experiment with discrete diffusion for text and images (Li et al., 2024c), though many, like Dual-Diffusion (Li et al., 2024c), still use continuous diffusion for image synthesis, failing to realize true modality symmetry.

Fully Discrete Diffusion: Recent work like UniDisc (Swerdlow et al., 2025a) pioneers full-token discrete diffusion over shared Transformer backbones. These models support parallel sampling and native integration, but currently lag behind in generation fidelity and scale.

Among these, the GPT-4o (OpenAI, 2025) model represents a significant advance as a unified multimodal generative system. However, its closed-source nature obscures critical architectural and training details, and its success may be largely attributable to scale rather than architectural novelty (Chen et al., 2025c).

- A.2 MASKED IMAGE MODELING

Masked Image Modeling (MIM) has emerged as a powerful self-supervised learning paradigm in computer vision (Bai et al., 2025a), drawing inspiration from the success of Masked Language Modeling (MLM) in NLP, notably BERT (Devlin, 2018). The fundamental principle of MIM involves obscuring portions of an image, which could be raw pixels (MAE (He et al., 2022)), latent patches of pixels, or even discrete latent tokens (BEiT (Bao et al., 2021), MaskGIT (Chang et al., 2022)), and training a model, typically an autoencoder, to predict or reconstruct this missing information by leveraging the context provided by the visible parts.

MaskGIT (Chang et al., 2022) introduced parallel decoding via iterative token refinement, inspiring discrete diffusion models. Recent work such as RandomAR (Fan et al., 2024) and MAR (Li et al.,

- 2024a) formalize this as random-order or masked autoregressive generation, blending AR and MIM principles. The major conceptual difference between RandomAR/MAR and MaskGIT is in the scanning order at inference time.

This class of techniques forms the conceptual foundation of discrete diffusion over tokenized spaces (Shi et al., 2025) and plays a critical role in modern unified models. We will introduce discrete diffusion in the next section.

A.3 RELATIONSHIP TO CONCURRENT WORK

Our main contribution is to demonstrate that a unified, visual-prior fully discrete diffusion model can be both effective and data-efficient for image understanding tasks, rather than just text-to-image generation tasks. Regarding the distinction from concurrent discrete diffusion works (Yang et al.,

- 2025a; Bai et al., 2026; Xin et al., 2025b;a), we think that unified models should allow for multiple design choices. Our goal is to demonstrate that a visual-first, fully discrete diffusion backbone can be a practical and competitive alternative to the more common “LLM-first” unified paradigm, and we believe this is a fundamental design choice.

Specifically, prior unified discrete diffusion models, such as UniDisc (Swerdlow et al., 2025b), are trained from scratch on multimodal data and therefore lack strong visual priors. As a result, they significantly underperform early diffusion baselines such as Stable Diffusion 1.5 (Rombach et al.,

- 2022) and do not support visual question answering tasks (Antol et al., 2015). In contrast, Muddit is the first unified discrete diffusion model built on top of a pretrained high-resolution text-to-image backbone (Bai et al., 2025b), with a lightweight text decoder on top. This visual prior is not an implementation detail: it improves the scalability and generalization behavior of discrete diffusion through a visual backbone that is well aligned with text.

- B ADDITIONAL QUALITATIVE RESULTS

Image-to-text Generation. We present more examples for image-to-text generation in Fig. 6. Text-to-image Generation. We present more examples for text-to-image generation in Fig. 7.

Visual Question Answering. We present more examples for visual question answering in Fig. 8. Muddit reliably identifies fine-grained attributes (e.g., “blonde” hair), object categories (e.g., “beagle”), and physical affordances (e.g., answering “No” to crossing at a red light). Notably, it also handles commonsense reasoning and spatial localization, such as inferring traffic legality or locating vehicles on the street.

Image-guided text editing. Zero-shot text-guided image editing performance is already verified and presented in Meissonic (Bai et al., 2025b). As the successor to Meissonic, we present Muddit’s performance on the image-guided text editing task, where the model completes a masked sentence based on the input image. As shown in Fig. 9, given a partially masked caption and an image, Muddit fills in the blanks with semantically and visually grounded phrases.

- C ADDITIONAL EXPERIMENTAL RESULTS

We provide a detailed breakdown of the MME benchmark results in Tab. 8. Muddit demonstrates strong performance in existence, color, and scene understanding, while also exhibiting solid reasoning capabilities.

Table 8: Detailed MME results.

Category Task Score

Perception

Existence 135.00 Count 78.33 Position 53.33 Color 140.00 Posters 62.24 Celebrity 56.18 Scene 107.25 Landmark 94.50 Artwork 76.00 OCR 52.50 Total 855.34

Cognition

Commonsense Reasoning 78.57 Numerical Calculation 90.00 Text Translation 57.89 Code Reasoning 57.50 Total 283.97

- D ADDITIONAL ABLATION STUDIES D.1 ABLATION STUDY ON THE CFG FOR IMAGE-TO-TEXT GENERATION

As shown in Tab. 9, we report performance on MS-COCO captioning and VQAv2 benchmarks. Moderate CFG values (e.g., 1.5) yield the best results, while higher scales lead to degraded performance.

- E INFERENCE TIME ANALYSIS

As shown in Fig. 12, autoregressive multimodal models are inherently limited by token-by-token decoding, which constrains their inference speed. Muddit overcomes this bottleneck with a parallel discrete diffusion decoder, reducing average latency to just 1.49 seconds, achieving a 4× to 11× speedup over competitive baselines (4.2× faster than Qwen-2.5-VL, 5.6× than Show-o, 8.1× than BLIP-2, and 10.9× than LLaVA-1.6).

We also present detailed FLOPs comparison between Autoregressive and Discrete Diffusion. Autoregressive (AR) without KV Cache:

- • At step t, the model attends over t previous tokens.
- • Per-step attention FLOPs: O(t2D).
- • Total FLOPs: L

L

L(L + 1)(2L + 1) 6

O(t2D) = O D

t2 = O D ·

t=1

t=1

#### Autoregressive (AR) with KV Cache:

= O(L3D)

Table 9: Ablation study on the effect of classifier-free guidance (CFG) scale.

Dataset CFG = 1 CFG = 1.5 CFG = 2 CFG = 2.5 CFG = 3

MS-COCO 57.2 59.9 58.2 51.3 47.2 VQAv2 65.8 68.2 64.7 55.4 49.2

Table 10: Comparison of model efficiency across different resolutions and steps. We report throughput for both text-to-image generation (images per second) and image-to-text tasks (tokens per second). Muddit achieves the best overall balance, matching the highest text-to-image throughput while significantly outperforming others in image-to-text speed.

Model Image Res Steps Text-to-Image (img/s) Image-to-Text (token/s) Meissonic 1024 32 0.23 – UniDisc 512 32 0.89 79.36 Monetico 512 32 1.00 – D-DiT 512 28 0.62 26.89 Muddit 512 32 1.00 99.98

- • At step t, Q is computed for 1 token, and attends to t K/V keys.
- • Per-step attention FLOPs: O(tD).
- • Total FLOPs: L

L

L(L + 1) 2

t = O D ·

O(tD) = O D

t=1

t=1

#### Discrete Diffusion:

= O(L2D)

- • Each step updates the full sequence (length L) in parallel.
- • Per-step attention FLOPs: O(L2D).
- • Total FLOPs: T · O(L2D) = O(TL2D), T ≪ L

While discrete diffusion may appear less efficient than autoregressive (AR) models with KV caching in terms of theoretical FLOPs, it offers a significant advantage over AR without caching—achieving an L/T speedup by updating the full token sequence in parallel over T iterations. In practice, the higher degree of parallelism leads to competitive, and often faster, inference speed compared to AR models, especially when considering real-world GPU throughput. As KV cache techniques for discrete diffusion are rapidly evolving (Ma et al., 2025), we expect further acceleration in the near future, narrowing the theoretical speed gap even with KV-cache AR baselines.

In Tab. 10, we compared Muddit against other non-autoregressive models, running all tests on a single A800 80 GB GPU. Muddit demonstrated a clear advantage in both image and text generation.

- F GENERATED RESULTS STEP BY STEP

Muddit frames text generation as reverse discrete diffusion over a fixed-length sequence of 77 token indices. At inference time, the model performs 16 ≤ T ≤ 32 denoising steps, starting from a maximally entropic prior where every token is masked. At each step t, a parameter-shared transformer

- G predicts a categorical distribution over all positions in parallel, and a sampler S selects the next sequence:

xt−1 = S(G(xt,c,t),xt,t), t = T,...,1, (12)

where xt ∈ V77 is the token sequence at step t, and c denotes conditioning inputs. The logits can be tempered or top-k filtered before sampling each token independently. The resulting sequence xt−1 seeds the next step, enabling fast, parallel decoding without autoregressive constraints.

Because all positions are updated in parallel, Muddit preserves global syntactic and semantic structure throughout the reverse diffusion process—unlike left-to-right autoregressive models, which can only condition on past predictions. Empirically, as few as 16 ≤ T ≤ 32 steps are sufficient to approximate the natural language distribution with high fidelity. Thus, Muddit unifies diffusion generation with parallel decoding, effectively overcoming the serial bottleneck that limits conventional autoregressive multimodal models.

We present two examples in Fig. 10 and 11.

- G DISCUSSION

- G.1 LIMITATIONS

While Muddit advances discrete diffusion for unified multimodal generation, it still has several limitations. First, due to its token-level discrete representation, the model may underperform continuous diffusion models in generating photorealistic or high-resolution images. Second, Muddit is initialized from a pretrained text-to-image foundation model, which offers strong visual priors but limits its capacity for rich text understanding and generation compared to the latest large language models. This makes it less suitable for tasks that require long-form understanding and generation or deep linguistic reasoning.

- G.2 BROADER IMPACTS

Muddit explores a new paradigm in multimodal generation by leveraging a strong visual prior as the backbone, in contrast to the prevailing trend of scaling large language models. This offers a complementary path toward efficient, grounded multimodal generation, particularly in vision-centric applications. The model’s ability to generate aligned visual and textual outputs in a fast, parallel manner could benefit downstream tasks, especially in completion-based scenarios such as masked captioning, image editing, and code implementation. However, as with all generative models, there remains a risk of misuse in synthetic content creation.

- H THE USE OF LARGE LANGUAGE MODELS

During the preparation of this paper, large language models were used only for language polishing and minor editing. All research ideas, methods, and experimental results were carried out entirely by the human authors.

[Figure 25]

Assistant: The image depicts a charming street scene with a cobblestone pavement and tall buildings. A white dog is resting. The dog is lying down near the bicycle on the street. To the left of the street, there is an old bicycle parked to the side of the street.

[Figure 26]

Assistant: The image is a black - and - white photograph featuring three individuals sitting together on a bench. They appear to be in an urban setting, as indicated by the trees and buildings in the background. The individuals are dressed in police uniforms, with one person looking behind the first individual from the second individual.

[Figure 27]

Assistant: The image depicts an elderly man standing on a city street, wearing a red jacket and a black cap talking on cell phone to his ear. The background shows a crowd of people walking around, some of whom are handbags and others visible in the distance. The crowd appears to be the middle of a busy area.

[Figure 28]

Assistant: The image depicts a close - up of a large, round pizza placed on a wooden table. The pizza is topped with cheese, and there are several pieces of basil leaves. In the background, there are other people sitting around.

[Figure 29]

Assistant: The image depicts a tennis match on a blue court. the tennis

match is set against a backdrop of a blue court. The player, wearing a white shirt and white shorts, is in mid action and holding a tennis racket, possibly preparing for a shot. the court has a row of seats.

[Figure 30]

Assistant: The image depicts a vibrant and colorful arrangement of umbrellas hanging from the branches of trees. The umbrellas come in color and shape, including pink, red, and yellow, blue, and purple. They are arranged in a circular pattern across the umbrellas, creating a visually appealing display. The background features a mix of green trees and foliage, suggesting an outdoor.

[Figure 31]

Assistant: The image depicts a bustling urban street scene with several notable elements. In the foreground, there is a tall green clock tower with a clock face and a pole, which stands prominently on the street. The clock has a clock face on it, indicating the time. In the background, there are several parked cars.

[Figure 32]

Assistant: The image depicts a person surfing on a surfboard on a large wave. The surfer is in mid-air, with his arms bent, navigating the balance of the wave. The background shows a clear blue sky, adding depth to the overall scene. The surfer seems to be in the middle of the the wave.

[Figure 33]

Assistant: The image depicts a giraffe standing on a grassy field in a rural outdoor setting. The background features a clear blue sky with some scattered clouds, and there are trees in the distant distance. The giraffe appears to be looking towards a distant horizon.

[Figure 34]

Assistant: The image depicts a stop sign in front of a grassy field with

majestic mountains in the background. The sky is clear with a scattered clouds, and snow-capped mountains rise in the background. The scene appears to be in a rural setting.

Figure 6: Image-to-text generated results.

[Figure 35]

Human: An editorial fashion photo portrait of a striking avant-garde model with bubble details. Iridescent art and pop surrealism influence the image's ethereal feel. Pastelaesthetic hues of soft pink, tangerine, and green create a dreamy backdrop. Seamless pink background, studio lighting emphasizes the model's silhouette against the soft pastel palette.

[Figure 36]

Human: Dark, heavy rainclouds gather over a rocky mountain range,

obscuring the peaks. A weathered canvas tent flaps open in the wind, its silhouette stark against the gray backdrop. Water pours down in sheets, blurring the image and creating a misty effect.

[Figure 37]

Human: A quaint small town nestled amidst snow-capped hills, bathed in the soft, golden hues of dusk. The sky is a tapestry of twilight blue and

orange, casting long shadows across cobblestone streets as frosted trees stand bare against the fading light. A sense of quiet peace emanates from the scene.

[Figure 38]

Human: An old cyanotype photograph capturing a serene lake at twilight. A majestic heron stands in the tranquil waters, its long legs poised, as the last golden rays of sunlight paint the sky with hues of blue and purple. The distant mountain range glows warmly with the soft, crepuscular light. Sharp focus on the heron and water reflection. Wide-angle lens captures a panoramic scene.

[Figure 39]

Human: A vibrant Japanese garden, inspired by Van Gogh's swirling brushstrokes. Vivid reds, yellows, blues, and greens dominate the scene. Cherry blossoms bloom in full splendor against a backdrop of ancient stone lanterns, rendered with lush foliage and dappled sunlight. The effect is heightened by 4K resolution and cinematic depth.

Figure 7: Text-to-image generation results.

Human: Can someone legally walk across the street right now?

[Figure 40]

[Figure 41]

Human: What color is the kid‘s hair?

Assistant: The color of the kid’s hair is blonde.

Assistant: No.

[Figure 42]

[Figure 43]

Human: What brand of dogs are these?

Human: What is flying in the picture?

Assistant: The brand of dogs are beagle.

###### Assistant: Kite.

[Figure 44]

[Figure 45]

Human: What eating utensil

Human: Where are the

is pictured?

cars?

Assistant: The cars are on the street.

Assistant:Fork.

- Figure 8: Visual question answering results.

[Figure 46]

<mask> <mask> <mask> lying on a grassy surface. <mask> <mask> has a <mask> fur with darker patches on its face and ears, looking directly at

<mask> <mask>. the bear's mouth is slightly <mask>, revealing its teeth

and tongue. the background shows some green grass.

A curly bear lying on a grassy surface. the bear has a brown fur with

darker patches on its face and ears, looking directly at the camera. the

bear's mouth is slightly open, revealing its teeth and tongue. the background shows some green grass.

Muddit

Image-guided text editing

- Figure 9: Image-guided text editing results.

[Figure 47]

User: Please describe this image.

### T=1

<mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask>

<mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask>

<mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask>

<mask> <mask> depicts <mask> <mask> and whimsical <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask>, <mask> <mask> <mask> <mask> <mask> <mask> <mask>. <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask>

The <mask> depicts a <mask> and whimsical <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask>, <mask> <mask> <mask> <mask> <mask> <mask> <mask>. the <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask>, <mask> <mask> <mask> <mask> <mask> <mask>

The image depicts a surreal and whimsical scene in what appears to be a <mask> <mask>, possibly <mask> <mask> or a dining room. the floor is covered with <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask>, <mask> <mask> <mask> <mask> <mask>

The image depicts a surreal and whimsical scene in what appears to be a domestic setting, possibly a

room or a dining room. the floor is covered with numerous pink flowers, adding a touch of <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask>, <mask> to the <mask> <mask> <mask> <mask> <mask> <mask> <mask>

The image depicts a surreal and whimsical scene in what appears to be a domestic setting, possibly a room or a dining room. the floor is covered with numerous pink flowers, adding a touch of <mask>. the

petals are scattered throughout the room, adding to the dreamlike quality of the scene.

The image depicts a surreal and whimsical scene in what appears to be a domestic setting, possibly a room or a dining room. the floor is covered with numerous pink flowers, adding a touch of color. the petals are scattered throughout the room, adding to the dreamlike quality of the scene.

The image depicts a surreal and whimsical scene in what appears to be a domestic setting, possibly a room or a dining room. the floor is covered with numerous pink flowers, adding a touch of color. the petals are scattered throughout the room, adding to the dreamlike quality of the scene.

- T=0 Figure 10: Image-to-text generated results in each step.

[Figure 48]

User: Please describe this image.

- T=1

<mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask>

<mask> image <mask> <mask> bear <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask>

<mask> image depicts <mask> bear lying <mask> its <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> be <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask>.

<mask> image depicts <mask> bear lying <mask> its <mask> <mask> a <mask> <mask> <mask>

<mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> appears to be resting <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> or <mask>.

The image depicts a bear lying <mask> its back <mask> a <mask> surface <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask>. the bear appears to be sleeping <mask> with its head <mask> on <mask> <mask> surface <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> or <mask>.

the image depicts a bear lying on its back on a <mask> surface. the <mask> is <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask> <mask>. the bear appears to be sleeping. the <mask> suggests <mask> the <mask> <mask> <mask>, possibly near a <mask> or <mask>.

The image depicts a bear lying on its back on a concrete surface. the bear is lying down, with <mask> head <mask> <mask> its head resting <mask> <mask> <mask>. the bear appears to be sleeping. the setting suggests that the scene is outdoors, possibly near a park or <mask>.

The image depicts a bear lying on its back on a concrete surface. the bear is lying down, with its head resting on the surface. the bear appears to be sleeping. the setting suggests that the scene is outdoors, possibly near a park or garden.

T=0

Figure 11: Image-to-text generated results in each step.

[Figure 49]

###### Figure 12: Inference speed comparison. We use 32 inference steps for Muddit and fix the sequence length to 77 across all models.

