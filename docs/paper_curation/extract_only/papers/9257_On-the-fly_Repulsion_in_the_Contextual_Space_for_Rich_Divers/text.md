# arXiv:2603.28762v2[cs.CV]3Jun2026

## On-the-fly Repulsion in the Contextual Space for Rich Diversity in Diffusion Transformers

OMER DAHARY∗, Tel Aviv University, Israel and Snap Research, Israel BENAYA KOREN∗, Tel Aviv University, Israel DANIEL GARIBI, Tel Aviv University, Israel and Snap Research, Israel DANIEL COHEN-OR, Tel Aviv University, Israel and Snap Research, Israel

“People with 3D holograms”

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

OursFlux-dev

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Fig. 1. Example results of our Contextual Space repulsion framework using Flux-dev. The base model (top) typically converges on a narrow set of visual solutions. By applying semantic intervention within the internal multi-modal attention channels, our approach (bottom) produces a diverse set of images with minimal computational overhead.

of generative diversity. As advanced generative models are increasingly optimized for precision and human preference, they tend to converge on a narrow set of “typical” visual solutions, a phenomenon often described as typicality bias [Teotia et al. 2025]. Diversity is no longer a secondary metric; it has become a central research problem addressed by a growing body of work [Jalali et al. 2025; Morshed and Boddeti 2025; Um and Ye 2025]. This is because the utility of generative AI depends on its ability to act as a creative partner that explores the vast manifold of human imagination. It should function as a generative engine rather than merely a sophisticated retrieval mechanism.

Modern Text-to-Image (T2I) diffusion models have achieved remarkable semantic alignment, yet they often suffer from a significant lack of variety, converging on a narrow set of visual solutions for any given prompt. This typicality bias presents a challenge for creative applications that require a wide range of generative outcomes. We identify a fundamental trade-off in current approaches to diversity: modifying model inputs requires costly optimization to incorporate feedback from the generative path. In contrast, acting on spatially-committed intermediate latents tends to disrupt the forming visual structure, leading to artifacts. In this work, we propose to apply repulsion in the Contextual Space as a novel framework for achieving rich diversity in Diffusion Transformers. By intervening in the multimodal attention channels, we apply on-the-fly repulsion during the transformer’s forward pass, injecting the intervention between blocks where text conditioning is enriched with emergent image structure. This allows for redirecting the guidance trajectory after it is structurally informed but before the composition is fixed. Our results demonstrate that repulsion in the Contextual Space produces significantly richer diversity without sacrificing visual fidelity or semantic adherence. Furthermore, our method is uniquely efficient, imposing a small computational overhead while remaining effective even in modern “Turbo” and distilled models where traditional trajectory-based interventions typically fail. Project page: https://contextual-repulsion.github.io/.

The diversity problem is fundamentally difficult due to the structural tension between quality and variety. High-quality generation currently relies on strong conditioning signals, most notably Classifier-Free Guidance (CFG) [Ho and Salimans 2022], which effectively sharpens the probability distribution around a single mode by suppressing nearby semantically valid alternatives. Consequently, restoring diversity requires an efficient mechanism to overcome this bias without degrading the structural integrity of the image or losing semantic adherence.

Previous attempts to bridge the diversity-alignment gap can be categorized by their point of intervention within the denoising trajectory, as illustrated in Figure 2. Upstream methods (Figure 2a) attempt to solve the problem by altering initial conditions, such as noise seeds or prompt embeddings. However, these approaches are often decoupled from the actual generation process [Sadat et al. 2023]; to achieve semantic grounding, they must either rely on noisy intermediate estimates [Kim et al. 2025] or employ optimization that incurs significant computational overhead [Parmar et al. 2025;

1 Introduction

The rapid evolution of Text-to-Image (T2I) generative models has ushered in a new era of high-fidelity visual synthesis, where models now exhibit unprecedented alignment with complex textual prompts [Esser et al. 2024; Podell et al. 2023; Rombach et al. 2022]. However, this progress has come at a significant cost: the reduction

*Denotes equal contribution.

p i zti

p i zti

p i zti

Dual − stream block

Dual − stream block

Dual − stream block

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Dual − stream block

Dual − stream block

Dual − stream block

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

zt−1i

zt−1i

zt−1i

(a) Upstream

(b) Downstream

(c) Ours

Fig. 2. Conceptual comparison of diversity strategies in dual-stream DiT architectures. Here 𝑝(𝑖) denotes the prompt embedding for sample

𝑖, 𝑧𝑡(𝑖) denotes the latent at timestep 𝑡 for sample 𝑖, and the red doublearrow icon indicates the point of diversity manipulation. (a) Upstream:

Interventions on noise or prompt embeddings lack structural feedback from the emerging image. (b) Downstream: Repulsion in image latents acts on a fixed visual mode and can push samples off the data manifold, causing artifacts. (c) Ours: By applying on-the-fly repulsion within the Contextual Space (text-attention channels), we steer the model’s generative intent. This allows for a semantically driven intervention synchronized with the emergent visual structure.

Um and Ye 2025]. Conversely, downstream methods (Figure 2b) enforce repulsion in the image latent space during denoising [Corso et al. 2023; Jalali et al. 2025]. While these can force variance, they often push samples outside the learned data manifold, resulting in catastrophic drops in visual fidelity and unnatural visual artifacts.

The core difficulty lies in an interventional trade-off: early interventions lack structural feedback, while late interventions face a committed visual mode. This is particularly acute in few-step "Turbo" models, where the generative path is decided almost instantly. Upstream methods require slow optimization to search for diversity-inducing initial conditions, while downstream repulsion arrives too late to steer the composition.

In this work, we present a novel approach that bypasses this tradeoff by identifying and leveraging the Contextual Space (Figure 2c), which emerges inside the multimodal attention blocks of Diffusion Transformer (DiT) architectures [Esser et al. 2024; Labs 2024]. Unlike previous U-Net models where text conditioning remains a static external signal, these blocks facilitate a dynamic bidirectional exchange between text and image tokens, continuously updating the text representations in response to the evolving image. This interaction creates an “enriched” semantic representation that is both aware of the prompt and synchronized with emergent visual details [Helbling et al. 2025].

By leveraging these enriched textual representations, our approach steers the model’s generative intent to overcome the CFG mode collapse. By targeting these representations rather than raw pixels, we preserve samples within the learned data manifold, avoiding the artifacts common in downstream interventions. To achieve this, we apply repulsion to the tokens as they pass between multimodal attention blocks. This intervention is performed on-the-fly during the transformer’s forward pass, at a stage where the emergent representation is already structurally informed but the final

composition is not yet fixed. Intervening while the representation is still flexible allows for steering that remains semantically driven yet image-aware. This enables the model to explore diverse paths while maintaining natural, high-quality results.

To demonstrate the efficacy of our approach, we conduct extensive experiments across multiple DiT-based architectures. We evaluate our results on the COCO benchmark using metrics for both visual quality and distributional variety. Our results show that repulsion in the Contextual Space consistently produces richer diversity without the mode collapse or semantic misalignment characteristic of prior work. Furthermore, we demonstrate that our method is uniquely efficient, requiring only a small computational overhead and no additional memory, making it compatible with the rapid inference requirements of modern distilled models.

2 Related Work

Diffusion transformers. While foundational diffusion models predominantly utilized UNet-based architectures [Podell et al. 2023; Ramesh et al. 2022; Razzhigaev et al. 2023; Rombach et al. 2022; Saharia et al. 2022], contemporary state-of-the-art text-to-image systems have largely shifted toward Diffusion Transformers (DiTs) as their backbone [Esser et al. 2024; Kong et al. 2025; Labs 2024; Labs et al. 2025]. A key distinction lies in the conditioning mechanism: whereas UNets typically incorporate text via cross-attention layers, DiTs process text and image tokens concurrently within the transformer. This architecture employs multimodal attention blocks to facilitate bidirectional interaction, ensuring a unified integration of visual and textual information throughout the generation process. A growing body of research has successfully employed this architecture across diverse downstream tasks [Avrahami et al. 2025; Dalva et al. 2024; Garibi et al. 2025; Kamenetsky et al. 2025; Labs et al. 2025; Tan et al. 2025; Zarei et al. 2025]

Research addressing the diversity-alignment gap in Text-to-Image (T2I) models generally falls into two categories based on the stage and level of intervention: upstream methods, which modify conditions prior to or in the earliest stages of the generative process, and downstream methods, which manipulate the image latents throughout the denoising trajectory.

Upstream Interventions. Upstream methods attempt to induce diversity by optimizing input conditions, namely the initial noise or text conditioning, before a stable image structure emerges. Purely decoupled interventions like CADS [Sadat et al. 2023] inject promptagnostic noise into text embeddings, which often leads to semantic drifting due to a lack of structural feedback. To bridge this, meth-

ods like CNO [Kim et al. 2025] utilize the very first timestep’s 𝑥ˆ0 prediction to force divergence, yet these estimates are frequently structurally unformed at high noise levels, providing an unstable signal for conceptual variety. Similarly, optimization-based regimes such as MinorityPrompt [Um and Ye 2025] and Scalable Group Inference (SGI) [Parmar et al. 2025] seek diversity-inducing initial conditions through iterative search; however, their heavy computational overhead makes them increasingly impractical for real-time applications or integration with fast-inference distilled models.

Downstream Interventions. Downstream methods manipulate the latent trajectory throughout the denoising process, either through interacting particle systems or modified guidance schedules. The former, pioneered by Particle Guidance (PG) [Corso et al. 2023], uses kernel-based repulsion in the image latent space to force variance between samples, with subsequent works focusing on improving repulsion loss objectives [Askari Hemmat et al. 2024; Jalali et al. 2025; Morshed and Boddeti 2025] or applying it to image restoration [Cohen et al. 2023]. Despite these refinements, these methods operate on non-semantic representations, repelling low-level pixelspace features rather than semantic content. Importantly, semantic concepts in the image latent space are spatially entangled and not aligned across samples, so the same high-level attribute may correspond to different spatial locations and configurations in different generations. As a result, repulsion in this space often pushes samples outside the learned manifold, leading to unnatural artifacts. In addition, such approaches lack sufficient trajectory depth to remain effective in modern distilled “Turbo” models; since the generative path is decided almost instantly, the remaining denoising trajectory is insufficient for late-stage repulsion to steer the model toward diverse modes.

Specifically for distilled models, Diversity Distillation [Gandikota and Bau 2026] attempts to restore variety by matching the base model’s distribution; however, it is limited by the teacher’s own lack of diversity and requires a base model that may be inaccessible or costly at inference. Alternatively, scheduling-based approaches like Interval Guidance [Kynkäänniemi et al. 2024] preserve variety by modulating the CFG scale during denoising. However, because these rescaling schedules are fixed and independent of the model’s internal state, they often reduce the prompt’s influence before the model has sufficiently established semantic alignment to the prompt.

A recurring limitation of these approaches is that their steering signals, whether derived from raw latents or external encoders, lack the semantic coherence necessary for meaningful control during the critical early stages of denoising. This forces an unfavorable tradeoff: upstream intervention must incur significant computational overhead to find valid diversity-inducing paths, while downstream interventions occur on a committed visual mode where the composition is already fixed, often producing noise-level variance that pushes samples outside the learned manifold and results in unnatural artifacts. Our work departs from these by identifying a Contextual Space within Diffusion Transformers that is both semantically flexible and structurally informed. This allows us to redirect the guidance trajectory once the bidirectional exchange between text and image tokens has established a stable semantic signal, but before the model has fully converged on a specific generative outcome.

- 3 Method: Repulsion in the Contextual Space

In this section, we formalize our approach to generative diversity by shifting the intervention focus to the Contextual Space. As identified in Section 2, the core difficulty of existing methods lies in the timing and location of the repulsion: upstream methods act on unformed noise, while downstream methods act on a rigid latent manifold. Our central insight is that the Contextual Space, inherent to multimodal transformer architectures such as DiTs, provides an effective

environment for diversity interventions because it is structurally informed yet conceptually flexible.

- 3.1 Defining the Contextual Space

The Contextual Space is the high-dimensional manifold formed within the Multimodal Attention (MM-Attention) blocks of a DiT. Unlike the static text embeddings used in U-Net architectures, the DiT processing flow facilitates a bidirectional exchange between text features 𝑓𝑇 and image features 𝑓𝐼.

In each transformer block 𝑙, the resulting tokens undergo a structural transformation:

𝑓ˆ𝑇(𝑙), 𝑓ˆ𝐼(𝑙) = MM-Attn(𝑓𝑇(𝑙−1), 𝑓𝐼(𝑙−1)). (1) In thisinteraction, thetextfeatures 𝑓𝑇 guidetheimagetokens toward the prompt’s semantic requirements. Simultaneously, the image features 𝑓𝐼 provide feedback regarding the spatial composition and emerging visual details, which the text features absorb to become uniquely tied to the specific image being formed. We therefore identify the resulting enriched text tokens 𝑓ˆ𝑇(𝑙) as the primary elements of the Contextual Space.

A key advantage of this space is its inherent token ordering. Unlike the image latent space, where specific semantic content can shift spatially across different samples, the Contextual Space maintains a fixed semantic alignment across the sequence index. This facilitates a consistent representation where each token index generally represents the same conceptual component across the entire batch, largely independent of its realized placement in the emergent image structure.

- 3.2 The Mechanism of Contextual Repulsion

We illustrate the positioning of our intervention in Figure 2c. Our key insight is that applying repulsion within the Contextual Space allows for the manipulation of generative intent. By enforcing distance between batch samples in this space, we steer the model’s high-level planning before it commits to a specific visual mode. To achieve this, we adopt the particle guidance framework [Corso et al. 2023], which treats a batch of 𝐵 samples as interacting particles. However, unlike prior work that applies guidance to the image latents 𝑧𝑡 (Figure 2b), we apply the repulsive forces directly to the Contextual Space tokens 𝑓ˆ𝑇 (Figure 2c).

Since the conditioning for each sample is initialized from the same unmodified prompt encoding at every timestep, the intervention mitigates the risk of permanent semantic drift. This common starting point promotes a state where contextual features remain closely aligned to the original prompt and directly comparable across the batch throughout the trajectory, allowing the repulsion to act as a force that differentiates how the same prompt is visually realized.

A critical advantage of our approach is that these forces are computed on-the-fly. Because we intervene directly on the internal activations, the method does not require backpropagating through the model layers, making it significantly more computationally efficient than optimization-based methods. Within each transformer block, we apply 𝑀 inner-block iterations to iteratively refine the token positions. Following the gradient-based guidance formulation [Corso et al. 2023], the updated state of the contextual tokens

for a sample 𝑖 ∈ {1, . . .,𝐵} after each iteration is given by: 𝑓ˆ𝑇,𝑖(𝑙)′ = 𝑓ˆ𝑇,𝑖(𝑙) +

𝜂 𝑀 ∇𝑓ˆ(𝑙)

L𝑑𝑖𝑣({𝑓ˆ𝑇,𝑗(𝑙)}𝐵𝑗=1), (2)

𝑇,𝑖

where 𝜂 is the overall repulsion scale and L𝑑𝑖𝑣 is a diversity loss defined over the batch of 𝐵 samples. To maintain diversity throughout the trajectory, we apply this repulsion across all transformer MM-blocks. However, since the initial stages of the denoising trajectory are the most crucial for the eventual semantic meaning and global composition [Balaji et al. 2023; Cao et al. 2025; Dahary et al. 2025, 2024; Huberman et al. 2025; Patashnik et al. 2023; Yehezkel et al. 2025], and are also where strong guidance signals such as CFG most strongly bias the generative path, we restrict the intervention to a chosen interval of the first few timesteps.

3.3 Diversity Objective

The Contextual Space encodes global semantic intent shared across the batch, making diversity objectives based on batch-level similarity more appropriate than token-wise or local measures. While our framework is flexible and can adopt various diversity losses defined in prior work [Jalali et al. 2025; Morshed and Boddeti 2025], we specifically utilize the Vendi Score [Askari Hemmat et al. 2024; Friedman and Dieng 2022] as our primary objective. The Vendi Score provides a principled way to measure the effective number of distinct samples in a batch by considering the eigenvalues of a similarity matrix. Formally, it is defined as the exponent of the von Neumann entropy of that matrix.

For simplicity, we represent each sample 𝑖 at block 𝑙 as a single

vector c𝑖(𝑙) ∈ R𝑁𝐷 by flattening the sequence of 𝑁 contextual tokens, each of dimension 𝐷. For a batch of size 𝐵 represented by these

flattened contextual vectors {c𝑖(𝑙)}𝑖𝐵=1, we first define a kernel matrix K ∈ R𝐵×𝐵, where each entry 𝐾𝑖𝑗 represents the similarity between samples 𝑖 and 𝑗. In our work, we use the cosine similarity as our kernel:

⟨c𝑖(𝑙), c(𝑗𝑙)⟩ ∥c𝑖(𝑙)∥∥c(𝑗𝑙)∥

(3)

𝐾𝑖𝑗 =

To maximize diversity, we compute the eigenvalues {𝜆𝑘} of the normalized kernel K˜ = 𝐵1 K and define our loss L𝑑𝑖𝑣 as the negative von Neumann entropy:

∑︁𝐵

𝜆𝑘 log𝜆𝑘 (4)

L𝑑𝑖𝑣 = −

𝑘=1

This objective effectively pushes the tokens in the Contextual Space to span a higher-dimensional manifold, preventing the semantic collapse typically induced by CFG.

- 4 The Contextual Space

In this section, we empirically examine the properties of the Contextual Space by analyzing how internal representations behave under controlled interpolation and extrapolation. We focus on how semantic structure is preserved or degraded when steering representations in two internal spaces of the DiT: the VAE latent space and the contextual (enriched text) token space. The goal is to characterize how each of these spaces reflects semantic variation when multiple samples are generated from the same prompt, and to assess

“A mythical creature” Target Interpolation Source Extrapolation

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Contextual Space

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Latent Space

“A person with their pet” Target Interpolation Source Extrapolation

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Contextual Space

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Latent Space

Fig. 3. Comparison of interpolation and extrapolation between the internal representations of two images. Intermediate frames are generated by denoising the source image while linearly blending its internal features with those of the target; extrapolation extends this vector beyond the endpoints. While Latent Space interpolation leads to structural blurring and artifacts due to spatial misalignment, the Contextual Space maintains high visual fidelity. This demonstrates that the Contextual Space enables smooth semantic transitions by decoupling generative intent from fixed spatial structures.

their suitability for diversity control without introducing visual artifacts.

To examine this, we conduct an interpolation and extrapolation experiment across these two internal representation spaces. We consider two prompts, “a person with their pet” and “a mythical creature”. For each prompt, we generate two samples using different initial noise seeds, which we designate as a source image and a target image. Maintaining the initial noise of the source image, we intervene during the denoising process by replacing its internal representation with a linear combination of the source and target representations

h𝑖𝑛𝑡𝑒𝑟𝑝 = h𝑠𝑜𝑢𝑟𝑐𝑒 + 𝛼(h𝑡𝑎𝑟𝑔𝑒𝑡 − h𝑠𝑜𝑢𝑟𝑐𝑒), (5) where h represents the representation in a given space, and 𝛼 is the steering coefficient. We compare this behavior across two distinct spaces: the VAE Latent Space (𝑧𝑡) and our proposed Contextual Space (enriched text tokens 𝑓ˆ𝑇).

As illustrated in Figure 3, the results highlight a fundamental difference in how these spaces handle semantic information. In the VAE Latent Space, representations are tied to the specific spatial grid and pixel-level layout of the sample. Since the source and target

images are spatially unaligned (exhibiting different poses and compositions) interpolating between them results in a structural blur. The model attempts to resolve two conflicting geometries simultaneously, leading to incoherent overlays and ghostly artifacts. More critically, extrapolating in the VAE Latent Space quickly pushes the latents outside the learned data manifold, resulting in severe artifacts.

In contrast, performing the same operation within the Contextual Space yields a smooth semantic transition. Rather than blending pixels or geometries, the model reallocates visual elements in a coherent manner, gradually modifying appearance and composition while maintaining a sharp, high-fidelity structure. For instance, as we move from the source image toward the target, we observe a meaningful evolution in high-level appearance attributes of the subject, such as facial features and overall visual style, which shift naturally from the source toward the target. In the bottom example, this transition appliescoherentlytoeach subject independently, with both the woman and the accompanying pet undergoing meaningful semantic changes (e.g., the pet gradually shifting from a dog-like to a cat-like appearance). Throughout this interpolation, the pretrained weights retain the generated images on-manifold, preserving structural integrity and visual plausibility.

Furthermore, the Contextual Space maintains its integrity during extrapolation, where the shifts remain semantically consistent with the direction of the steering vector (h𝑡𝑎𝑟𝑔𝑒𝑡 − h𝑠𝑜𝑢𝑟𝑐𝑒). As shown in the right-most columns of Figure 3, applying extrapolation (𝛼 < 0) relative to the target does not lead to manifold collapse. Instead, it generates a semantically meaningful extrapolation: In the top example, extrapolation progressively removes the creature’s horns and beast-like features, producing a plausible semantic evolution rather than noise or collapse. In the bottom example, the woman’s features evolve toward a darker tone, effectively moving away from the characteristics of the reference. Simultaneously, the pet’s appearance is modified in a logically consistent manner, such as deepening the coat color and shifting the ears to a more drooping shape. These observations suggest that the Contextual Space encodes global semantic features independently of a fixed spatial grid. Intervening in this space enables the modification of high-level attributes while the transformer’s attention mechanisms maintain the structural coherence of the output.

- 5 Experiments

To evaluate the generality of our approach, we conduct experiments across three state-of-the-art Diffusion Transformer (DiT) architectures that span distinct design choices and sampling regimes: Fluxdev [Labs 2024], a guidance-distilled model; SD3.5-Turbo, distilled for high-speed, few-step inference; and SD3.5-Large [Esser et al. 2024], a standard non-distilled model. Together, these models cover a broad spectrum of modern DiT variants, allowing us to demonstrate that Contextual Space repulsion is broadly applicable and not tied to a specific architecture, training regime, or sampling budget.

We compare our Contextual Space repulsion against representative diversity-enhancing baselines, including upstream methods that modify initial conditions such as CADS [Sadat et al. 2023] and

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

OursFlux

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

“Kids with paper airplanes”

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

OursFlux

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

“A ballet dancer on stage” Fig. 4. Qualitative results. For each prompt, we compare the base model results (top) to our results (bottom).

SGI [Parmar et al. 2025], as well as downstream methods that intervene in the latent space, including Particle Guidance [Corso et al. 2023] and SPARKE [Jalali et al. 2025]. Full implementation details and hyperparameter settings are provided in Appendix A.

5.1 Qualitative Results

Flux-dev results. We compare our results with the base Flux-dev model in Figures 4 and 11; additional comparisons with Flux-dev, SD3.5-Large and SD3.5-Turbo are provided in Appendix B. For an objective comparison, all qualitative results use the same fixed seed to sample a batch of distinct initial noises. Despite this, the base model typically produces a narrow and repetitive range of outputs for many prompts. As shown in Figure 11, our method alleviates typicality biases, such as the barely visible or harsh lighting seen in the “musician” and “scientist” examples. Furthermore, it generates a diverse array of compositions, arrangements, and camera angles for the “painter” and “stadium” prompts.

Baseline comparisons. We present qualitative comparisons against the baselines in Figure 12. As illustrated, downstream methods like PG and SPARKE often introduce visual artifacts because they intervene directly in the VAE latent space. For instance, in the “red bus” example, PG fails to modify the image structure, while SPARKE succeeds in moving objects but leaves patterned “holes” in their original locations.

In contrast, upstream methods maintain higher image quality, though they face different trade-offs. CADS frequently leads to semantic drift, where diversity is achieved through weak prompt alignment (e.g., replacing “photographs” with people, or a “phoenix“

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

OursFluxKontext

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

Input Image “a person running a marathon”

Fig. 5. Integration with image editing models. We demonstrate that our method can be successfully integrated into Flux-Kontext to generate high-quality diverse results.

with a bonfire). SGI, which filters a large set of initial noise candidates through optimization, achieves both high quality and prompt adherence by minimizing intervention. However, SGI often struggles to produce high variation for prompts where the base model lacks inherent diversity, resulting in repetitive subject appearances and compositions (e.g., the “red bus”).

Our method achieves richer diversity even with challenging prompts, without sacrificing alignment or quality. Interestingly, the axes of variation adapt to each prompt: for the “phoenix,” the model alternates between artistic styles; for the “bus,” it varies weather and pose; and for the “camera with old photographs” and “wolf pack,” it generates unique compositions and object arrangements.

Example result on Flux-Kontext. In Figure 5, we demonstrate that

- our method generalizes beyond text-to-image generation and can be applied out of the box to image editing models, specifically Flux Kontext [Labs et al. 2025]. Perhaps surprisingly, this requires no modification to the model or to our intervention strategy: we apply the exact same Contextual Space repulsion within the editing instruction stream. While the base editing model produces nearly identical edits across different random seeds, our approach yields diverse yet coherent edit realizations, all while preserving the intended edit semantics and maintaining the visual integrity of the original image. This result highlights that contextual repulsion operates at a level of abstraction that is compatible with both generation and editing paradigms, despite being developed specifically for text-to-image models.

- 5.2 Quantitative Results

Diversity-Quality trade-off. We evaluated our method using 1,000 prompts sampled from the MS-COCO 2017 validation set, generating four images per prompt for a total of 4,000 images per configuration. To provide a holistic view of the diversity-quality trade-off, we utilize the Vendi Inception Score [Friedman and Dieng 2022; Szegedy

- et al. 2017] to measure high-level semantic diversity alongside three primary quality and alignment axes: ImageReward [Xu et al. 2023] for human preference, VQAScore [Lin et al. 2024] for fine-grained prompt adherence, and Kernel Inception Distance (KID) [Bińkowski
- et al. 2018] for distributional fidelity. By plotting the Pareto frontier

[Figure 66]

Fig. 6. Quantitative evaluation. Pareto frontiers comparing our method against baseline methods using Flux-dev. We evaluate the trade-off between semantic diversity (Vendi Score) and three performance axes: (Left) Human Preference [ImageReward ↑], (Middle) Prompt Alignment [VQAScore ↑], and (Right) Distributional Fidelity [KID ↓]. Our method (red) achieves a superior frontier across all metrics.

Table 1. Runtime comparison for generating a group of four images. Our method provides a significant speedup over optimization-based diversity methods like SGI while maintaining a low overhead relative to the base model.

Method SD3.5-Large SD3.5-Turbo Flux-dev

Base Model 13.83s 4.18s 10.34s Ours (Contextual) 18.12s 5.52s 12.80s

8 Candidates 66.79s 13.15s 47.47s 16 Candidates 76.79s 23.73s 56.32s 32 Candidates 101.44s 46.15s 75.39s 64 Candidates 145.14s 91.30s 113.99s

SGI

of the diversity score versus each of these metrics, we can analyze how effectively each method navigates the tension between generative variety and visual fidelity.

To map the Pareto frontiers, we systematically vary the control hyperparameters for each baseline: the guidance scale for PG and SPARKE, the noise intensity for CADS, and the number of initial noise candidates for SGI. Specific hyperparameter configurations are provided in Appendix A.

As shown in Figure 6, our method achieves a superior trade-off on Flux-dev. Notably, while our method exceeds the performance of SGI, the strongest baseline, it does so with drastically lower computational overhead (see Table 1). Results for additional models, including SD3.5-Turbo and SD3.5-Large, are provided in Appendix C.

Runtime. Many existing diversity methods rely on costly downstream signals, either through gradient-based optimization or by selecting from large pools of candidate latents. Both strategies impose substantial time overhead. By avoiding these mechanisms entirely, our approach provides a markedly more efficient solution, increasing runtime by only 20%–30% relative to the base model (Table 1).

User study. Standard quantitative metrics often fail to capture the nuances of generative diversity. These evaluators are typically trained on datasets dominated by common visual patterns, leading them to favor “typical” or average cases as more aesthetically pleasing or prompt-adherent. Consequently, methods that successfully

[Figure 67]

Fig. 7. Overall user preference comparison. Distribution of user choices comparing our method with five competing approaches. Bars indicate the percentage of cases in which users preferred our results (green), preferred competing methods (red), or rated both equally (gray).

push for greater diversity and creative interpretation may be unfairly penalized by these metrics, even when the resulting variations are highly desirable to human users. To address this limitation and provide a more meaningful assessment of our method, we conducted a user study.

We utilized ChatGPT to generate 40 diverse prompts across vari-

- ous categories. For each prompt, participants were presented with two batches of 8 images (16 images total): one batch generated by our method and the other by a competing method or the base model (Flux-dev). Participants were tasked with performing a side-by-side comparison to determine which batch: (i) Exhibited greater visual and semantic diversity; (ii) Maintained higher image quality; (iii) Demonstrated better prompt adherence; and (iv) Was preferred overall.

We collected 450 responses from 45 participants. Figure 7 reports the overall user preference results of this study, with the full preference table provided in Appendix C. Overall, our method achieves higher user preference than all competing approaches. The only exception is SGI, where preferences are closely matched, with a slight advantage for our method. Importantly, these gains are achieved with minimal runtime overhead, as demonstrated in Table 1.

- 5.3 Ablation Studies

We evaluate the impact of the repulsion scale and the specific representation space used for intervention below, with further hyperparameter analyses provided in Appendix D.

Repulsion scale ablation. In Figure 8, we ablate the effect of the repulsion scale 𝜂. The top row (𝜂 = 0) represents the base Flux-dev generations, which exhibit a narrow interpretation of the prompt; each image displays a similar-looking house in nearly identical environments. In each subsequent row, we show the results of our method with an increasing repulsion scale. As can be seen, higher values of 𝜂 generally yield greater diversity, introducing structural changes like adding a tower to the house, altering the landscape with a lake, or shifting the scene’s season.

Repulsion space ablation. To isolate the efficacy of intervening in the Contextual Space (𝑓ˆ𝑇), we compare our framework against an

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

192989059=====𝜂𝑒𝜂𝑒𝜂𝑒𝜂𝜂𝑒

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

“A breathtaking view of a distant house in beautiful scenery”

Fig. 8. Ablation of the repulsion scale 𝜂. We visualize the impact of the repulsion scale on our results. At𝜂 = 0 (top row), the base model exhibits low diversity, producing similar architectural styles and environments across multiple seeds. As 𝜂 increases, our Contextual Space repulsion introduces progressively larger variations, while maintaining high image quality and prompt alignment.

identical repulsion mechanism applied instead to the image attention tokens (𝑓ˆ𝐼) within the multimodal blocks (i.e., the dual-stream blocks in Flux). As illustrated in Figure 9, repulsion in the Contextual Space produces a significantly more robust Pareto frontier, yielding superior human preference (ImageReward), distributional fidelity (KID), and prompt alignment (VQAScore). Notably, while the image-token baseline exhibits sharp performance degradation as diversity increases, our method maintains a shallower decline across all metrics. This suggests that the Contextual Space is better suited for navigating semantic diversity while strictly preserving the integrity of samples within the learned conditional manifold.

Figure 10 provides qualitative examples. As can be seen, applying repulsion in the image token space (𝑓ˆ𝐼) often results in stagnant layouts due to its spatial rigidity; this forces the repulsion to artificially promote diversity by modifying local textures, leading to artifacts such as the sea blending unnaturally into the road in the “street” example. In contrast, intervening in the contextual space (𝑓ˆ𝑇) tends to promote varied compositions while maintaining alignment and quality.

6 Conclusions

At a high level, this work highlights the Contextual Space in Diffusion Transformers as a particularly effective place to intervene when aiming for diversity. The Contextual Space sits between text and image: the representations already encode rich semantic intent

[Figure 93]

- Fig. 9. Ablation of Repulsion Space. Pareto frontiers comparing repulsion applied to text attention tokens (Contextual Space, 𝑓ˆ𝑇 ) versus image attention tokens (𝑓ˆ𝐼) within the Flux-dev architecture. We evaluate the trade-off between semantic diversity (Vendi Score) and three performance axes: (Left) Human Preference [ImageReward ↑], (Middle) Prompt Alignment [VQAScore ↑], and (Right) Distributional Fidelity [KID ↓]. Our method (red) achieves a superior frontier across all metrics.

ImageContextual

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

“Two pieces of bread with a leafy green on top of it”

ImageContextual

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

“A city street scene with a green bus coming up a street, with ocean”

- Fig. 10. Qualitative Ablation of Repulsion Space. For each prompt, we compare repulsion applied in the image attention space (Image) versus our Contextual Space (Contextual). While image-space repulsion is limited by spatial rigidity, our method achieves more varied compositions.

ones. In addition, the intervention is focused on early to mid stages of generation; how to best coordinate it with later stages, or combine it with other control mechanisms, remains an open question.

Future directions. An interesting direction for future work is to investigate whether a user-provided textual cue, such as “color” or “size”, can be used to guide the repulsion along a specific semantic direction in the Contextual Space. Instead of encouraging diversity in an unconstrained manner, the idea would be to bias the repulsive forces so that samples spread primarily along attributes associated with the given word. This could enable a more controlled and interpretable form of diversity, where variation is focused on selected semantic aspects while other parts of the generation remain stable.

Acknowledgments

We would like to thank Or Patashnik, Yuval Alaluf, Nir Goren, Maya Vishnevsky, Sara Dorfman, Shelly Golan, Saar Huberman, and Jackson Wang for their early feedback and insightful discussions. We also thank the anonymous reviewers for their thorough and constructive comments, which helped improve this work.

This research was supported in part by the Israel Science Foundation (Grants No. 2492/20 and 1473/24) and by Len Blavatnik and the Blavatnik Family Foundation.

References

Reyhane Askari Hemmat, Melissa Hall, Alicia Sun, Candace Ross, Michal Drozdzal, and Adriana Romero-Soriano. 2024. Improving geo-diversity of generated images with contextualized vendi score guidance. In European Conference on Computer Vision. Springer, 213–229.

Omri Avrahami, Or Patashnik, Ohad Fried, Egor Nemchinov, Kfir Aberman, Dani Lischinski, and Daniel Cohen-Or. 2025. Stable Flow: Vital Layers for TrainingFree Image Editing. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 7877–7888. doi:10.1109/cvpr52734.2025.00738

shaped by the emerging image, yet they are not spatially locked in. Unlike image latents, this space is not tied to a spatial grid, so samples can be pushed apart semantically without tearing geometry or introducing visual artifacts. At the same time, unlike early text embeddings, it is structurally informed, meaning that interventions meaningfully influence what the model actually generates.

Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, Tero Karras, and Ming-Yu Liu. 2023. eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers. arXiv:2211.01324 [cs.CV] https://arxiv.org/abs/2211.01324 Mikołaj Bińkowski, Danica J Sutherland, Michael Arbel, and Arthur Gretton. 2018.

Demystifying mmd gans. arXiv preprint arXiv:1801.01401 (2018).

Yu Cao, Zengqun Zhao, Ioannis Patras, and Shaogang Gong. 2025. Temporal Score Analysis for Understanding and Correcting Diffusion Artifacts. arXiv:2503.16218 [cs.CV] https://arxiv.org/abs/2503.16218

Applying on-the-fly repulsion in this space allows diversity to be increased in a controlled way, without sacrificing visual quality or relying on heavy optimization with significant computational cost. More broadly, this points to the importance of intervening at the right representational level, where decisions are still flexible, but already grounded in the image being formed.

Noa Cohen, Hila Manor, Yuval Bahat, and Tomer Michaeli. 2023. From posterior sampling to meaningful diversity in image restoration. arXiv preprint arXiv:2310.16047

(2023). Gabriele Corso, Yilun Xu, Valentin De Bortoli, Regina Barzilay, and Tommi Jaakkola.

2023. Particle guidance: non-iid diverse sampling with diffusion models. arXiv preprint arXiv:2310.13102 (2023).

Omer Dahary, Yehonathan Cohen, Or Patashnik, Kfir Aberman, and Daniel Cohen-Or. 2025. Be Decisive: Noise-Induced Layouts for Multi-Subject Generation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers. 1–12.

Limitations. Contextual repulsion increases diversity but does not provide direct control over which attributes will vary, and may sometimes favor coarse semantic changes over fine, user-specified

Omer Dahary, Or Patashnik, Kfir Aberman, and Daniel Cohen-Or. 2024. Be yourself: Bounded attention for multi-subject text-to-image generation. In European Conference on Computer Vision. Springer, 432–448.

Yusuf Dalva, Kavana Venkatesh, and Pinar Yanardag. 2024. FluxSpace: Disentangled Semantic Editing in Rectified Flow Transformers. arXiv:2412.09611 [cs.CV] https: //arxiv.org/abs/2412.09611

Juechu Dong, Boyuan Feng, Driss Guessous, Yanbo Liang, and Horace He. 2024. Flex attention: A programming model for generating optimized attention kernels. arXiv preprint arXiv:2412.05496 2, 3 (2024), 4.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. 2024. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning.

Dan Friedman and Adji Bousso Dieng. 2022. The vendi score: A diversity evaluation metric for machine learning. arXiv preprint arXiv:2210.02410 (2022).

Rohit Gandikota and David Bau. 2026. Distilling diversity and control in diffusion models. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 1304–1313.

Daniel Garibi, Shahar Yadin, Roni Paiss, Omer Tov, Shiran Zada, Ariel Ephrat, Tomer Michaeli, Inbar Mosseri, and Tali Dekel. 2025. TokenVerse: Versatile Multi-concept Personalization in Token Modulation Space. arXiv:2501.12224 [cs.CV] https://arxiv. org/abs/2501.12224

Alec Helbling, Tuna Han Salih Meral, Ben Hoover, Pinar Yanardag, and Duen Horng Chau. 2025. Conceptattention: Diffusion transformers learn highly interpretable features. arXiv preprint arXiv:2502.04320 (2025).

Jonathan Ho and Tim Salimans. 2022. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 (2022). Saar Huberman, Or Patashnik, Omer Dahary, Ron Mokady, and Daniel Cohen-Or.

2025. Image Generation from Contextually-Contradictory Prompts. arXiv preprint arXiv:2506.01929 (2025).

Mohammad Jalali, LEI Haoyu, Amin Gohari, and Farzan Farnia. 2025. SPARKE: Scalable Prompt-Aware Diversity and Novelty Guidance in Diffusion Models via RKE Score. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Ronen Kamenetsky, Sara Dorfman, Daniel Garibi, Roni Paiss, Or Patashnik, and Daniel Cohen-Or. 2025. SAEdit: Token-level control for continuous image editing via Sparse AutoEncoder. arXiv:2510.05081 [cs.GR] https://arxiv.org/abs/2510.05081

Byungjun Kim, Soobin Um, and Jong Chul Ye. 2025. Diverse Text-to-Image Generation via Contrastive Noise Optimization. arXiv preprint arXiv:2510.03813 (2025).

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Junkun Yuan, Yanxin Long, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, Jie Jiang, and Caesar Zhong. 2025. HunyuanVideo: A Systematic Framework For Large Video Generative Models. arXiv:2412.03603 [cs.CV] https://arxiv.org/abs/2412.03603

Tuomas Kynkäänniemi, Miika Aittala, Tero Karras, Samuli Laine, Timo Aila, and Jaakko Lehtinen. 2024. Applying guidance in a limited interval improves sample and distribution quality in diffusion models. Advances in Neural Information Processing Systems 37 (2024), 122458–122483.

Black Forest Labs. 2024. FLUX. https://github.com/black-forest-labs/flux. Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. 2025. FLUX. 1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space. arXiv preprint arXiv:2506.15742 (2025).

Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. 2024. Evaluating text-to-visual generation with imageto-text generation. In European Conference on Computer Vision. Springer, 366–384.

Mashrur M Morshed and Vishnu Boddeti. 2025. DiverseFlow: Sample-Efficient Diverse Mode Coverage in Flows. In Proceedings of the Computer Vision and Pattern Recognition Conference. 23303–23312.

Gaurav Parmar, Or Patashnik, Daniil Ostashev, Kuan-Chieh Wang, Kfir Aberman, Srinivasa Narasimhan, and Jun-Yan Zhu. 2025. Scaling Group Inference for Diverse and High-Quality Generation. arXiv preprint arXiv:2508.15773 (2025).

Or Patashnik, Daniel Garibi, Idan Azuri, Hadar Averbuch-Elor, and Daniel Cohen-Or.

2023. Localizing Object-level Shape Variations with Text-to-Image Diffusion Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. 2023. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023).

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen.

2022. Hierarchical Text-Conditional Image Generation with CLIP Latents. arXiv:2204.06125 [cs.CV]

Anton Razzhigaev, Arseniy Shakhmatov, Anastasia Maltseva, Vladimir Arkhipkin, Igor Pavlov, Ilya Ryabov, Angelina Kuts, Alexander Panchenko, Andrey Kuznetsov, and Denis Dimitrov. 2023. Kandinsky: an Improved Text-to-Image Synthesis with Image Prior and Latent Diffusion. arXiv:2310.03502 [cs.CV] https://arxiv.org/abs/2310. 03502

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.

2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.

Seyedmorteza Sadat, Jakob Buhmann, Derek Bradley, Otmar Hilliges, and Romann M Weber. 2023. CADS: Unleashing the diversity of diffusion models through conditionannealed sampling. arXiv preprint arXiv:2310.17347 (2023).

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, and Mohammad Norouzi. 2022. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding. arXiv:2205.11487 [cs.CV]

Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, and Alexander Alemi. 2017. Inception-v4, inception-resnet and the impact of residual connections on learning. In Proceedings of the AAAI conference on artificial intelligence, Vol. 31.

Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang.

2025. OminiControl: Minimal and Universal Control for Diffusion Transformer. arXiv:2411.15098 [cs.CV] https://arxiv.org/abs/2411.15098

Revant Teotia, Candace Ross, Karen Ullrich, Sumit Chopra, Adriana Romero-Soriano, Melissa Hall, and Matthew Muckley. 2025. DIMCIM: A Quantitative Evaluation Framework for Default-mode Diversity and Generalization in Text-to-Image Generative Models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 16431–16440.

Soobin Um and Jong Chul Ye. 2025. Minority-Focused Text-to-Image Generation via Prompt Optimization. In Proceedings of the Computer Vision and Pattern Recognition Conference. 20926–20936.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. 2023. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems 36 (2023), 15903–15935.

Shai Yehezkel, Omer Dahary, Andrey Voynov, and Daniel Cohen-Or. 2025. Navigating with Annealing Guidance Scale in Diffusion Space. arXiv preprint arXiv:2506.24108

(2025).

Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, Ben Hutchinson, Wei Han, Zarana Parekh, Xin Li, Han Zhang, Jason Baldridge, and Yonghui Wu. 2022. Scaling Autoregressive Models for Content-Rich Text-to-Image Generation. arXiv:2206.10789 [cs.CV] https://arxiv.org/abs/2206.10789

Arman Zarei, Samyadeep Basu, Mobina Pournemat, Sayan Nag, Ryan Rossi, and Soheil Feizi. 2025. SliderEdit: Continuous Image Editing with Fine-Grained Instruction Control. arXiv:2511.09715 [cs.CV] https://arxiv.org/abs/2511.09715

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

OursFlux

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

“A jazz musician playing saxophone in a dimly lit club”

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

OursFlux

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

“An artist painting a landscape in an outdoor studio”

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

OursFlux

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

“A scientist in a modern laboratory ”

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

OursFlux

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

“A crowd cheering at a sports stadium”

- Fig. 11. Qualitative results. For each prompt, we compare the base model results (top) to our results (bottom). All batches were generated using the same random seed initialization. Additional results are provided in Appendix B.

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

CADSPGOursSGISPARKE

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

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

“A wolf pack howling at the moon”

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

SPARKEPGOursSGICADS

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

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

“A phoenix rising from ashes”

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

PGCADSSPARKEOursSGI

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

“A camera with old photographs”

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

CADSSPARKEPGOursSGI

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

“A red London double-decker bus”

### Fig. 12. Qualitative comparison of our Contextual Repulsion approach against baseline methods. Each quadrant displays four generated samples per method for a given prompt.

Appendix

- A Implementation Details

All experiments were conducted on an NVIDIA A100 GPU. Quantitative metrics and runtime evaluations were performed by generating groups of 4 images. Diversity metrics were calculated within each 4-image group and subsequently averaged across all groups.

The number of denoising steps was chosen based on the model architecture: 4 steps for SD3.5-Turbo [Esser et al. 2024], 20 steps for Flux-dev [Labs et al. 2025], and 28 steps for SD3.5-Large [Esser et al. 2024]. The guidance scale was set to 3.5 for both Flux-dev and SD3.5-Large, and 0.0 for SD3.5-Turbo.

For our proposed method, we employed 𝑀 = 100 gradient steps for the Stable Diffusion models and 𝑀 = 50 for Flux-dev. For all models, we apply repulsion to the text tokens in the multimodal attention blocks (dual-stream in Flux). For SD3.5-Large, which is not distilled for classifier-free guidance, the repulsion is applied to both the conditional and unconditional branches. For Flux-dev and Flux-Kontext, we additionally apply it to all tokens in the later single-stream blocks, which are specific to these architectures. The repulsion scale𝜂 was used to balance the trade-off between diversity and fidelity, with the intervention disabled after a fixed number of timesteps, denoted by 𝜏. The range of 𝜂 was tuned per model: 𝜂 ∈ [5 · 102, 5 · 106] with 𝜏 = 4 for SD3.5-Large; 𝜂 ∈ [1 · 105, 1 · 107] with 𝜏 = 1 for SD3.5-Turbo; and 𝜂 ∈ [5 · 106, 8 · 109] with 𝜏 = 1 for Flux-dev. For simplicity, 𝜂 remained constant throughout the intervention window.

We utilized official implementations for all baseline methods, where available. For baselines without compatible official implementations, we re-implemented them and tuned their hyperparameters to ensure competitive diversity levels. In addition to the shared guidance and step configurations, the following hyperparameters were used for the baselines:

- • PG [Corso et al. 2023]: Repulsion scales were varied between 10 and 100.
- • CADS [Sadat et al. 2023]: Scales were varied between 0.1 and 0.7, with 𝜏1 = 0.3,𝜏2 = 0.8, and 𝜓 = 1.
- • SPARKE [Jalali et al. 2025]: Scales were selected between 0.02 and 0.14, depending on the model.
- • SGI [Parmar et al. 2025]: Evaluated with initial candidate groups of 𝑁 ∈ {8, 16, 32, 64}, utilizing default hyperparameters from the official implementation. All qualitative comparisons and the user study results reported here were conducted with 𝑁 = 64.

- B Additional Qualitative Results

We present additional qualitative results of our method on SD3.5Large (Figure 15), SD3.5-Turbo (Figure 16) and Flux-dev (Figures 17, 18, and 19).

- C Additional Quantitative Results

Additional comparisons. We present additional quantitative comparisons on SD3.5-Large (Figure 13) and SD3.5-Turbo (Figure 14). Our method achieves competitive quality-diversity trade-offs at a

[Figure 254]

- Fig. 13. Quantitative evaluation on SD3.5-Large.

[Figure 255]

- Fig. 14. Quantitative evaluation on SD3.5-Turbo.

Table 2. Detailed metrics for the Flux-dev Pareto frontiers in Figure 6.

Method Vendi (↑) IR (↑) VQA (↑) KID ×10−4 (↓) Base Model 1.780 1.075 0.883

𝜂 = 5 · 106 1.810 1.102 0.884 0.066

- 𝜂 = 1 · 107 1.831 1.092 0.883 0.103
- 𝜂 = 1 · 108 1.869 1.075 0.883 0.157 𝜂 = 5 · 108 1.898 1.070 0.880 0.172

Ours

𝑠 = 10−20 1.908 0.377 0.719 0.558 𝑠 = 10−18 1.908 0.377 0.719 0.558 𝑠 = 10−12 1.910 0.303 0.699 0.530 𝑠 = 10−11 1.923 0.208 0.674 0.588

CADS

𝑠 = 1 1.753 0.991 0.871 0.555 𝑠 = 80 1.759 1.018 0.864 0.675 𝑠 = 150 1.787 0.846 0.848 2.650

PG

8 Candidates 1.778 1.152 0.875 0.440 16 Candidates 1.829 1.085 0.873 0.461 32 Candidates 1.860 1.063 0.872 0.289 64 Candidates 1.916 1.042 0.872 0.297

SGI

SPARKE 𝑠 = 0.01 1.790 1.094 0.884 0.057 𝑠 = 0.02 1.850 1.067 0.873 1.079

fraction of the computational cost required by SGI. Detailed metrics across all evaluated models are provided in Tables 2, 3, and 4.

User study table. We provide the full results of our user study in Table 5.

Evaluation on detailed prompts. While diversity is typically easier to achieve when prompts leave significant room for interpretation, we evaluate our method on the 100 longest prompts from the “Complex” and “Fine-Grained Detail” categories of PartiPrompts [Yu et al. 2022] using Flux-dev. Even under these highly constrained conditions, our method increases diversity and human preference scores with a negligible impact on prompt alignment. Specifically, we observe an increase in Vendi score (+0.08) and ImageReward (+0.05),

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

SD3.5-LargeOurs

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

“An abandoned carnival”

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

SD3.5-LargeOurs

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

“A couple stargazing”

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

SD3.5-LargeOurs

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

“Elephants at a waterhole”

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

SD3.5-LargeOurs

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

“A climber on a cliff”

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

OursSD3.5-Turbo

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

“A dragon guarding its treasure”

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

OursSD3.5-Turbo

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

“A picnic under cherry blossoms”

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

OursSD3.5-Turbo

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

“A french bakery at dawn”

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

OursSD3.5-Turbo

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

“A snowy village at night”

Fig. 15. Qualitative results on SD3.5-Large.

### Fig. 16. Qualitative results on SD3.5-Turbo.

“A family enjoying a traditional meal together at home”

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

OursFlux

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

“A beautiful Japanese garden with a koi pond and cherry blossoms”

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

OursFlux

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

“A jazz singer performing on stage with a vintage microphone”

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

OursFlux

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

“A bustling street market in Morocco with colorful spices”

“A group of students studying together in a university library”

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

OursFlux

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

“A futuristic warrior standing on the edge of a neon-lit cliff”

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

OursFlux

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

“A wedding couple sharing a romantic moment”

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

OursFlux

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

“A chef preparing a gourmet meal in a professional kitchen”

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

OursFlux

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

“An astronaut exploring the terrain of an alien planet”

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

OursFlux

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

“An astronaut floating in space with Earth in the background”

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

OursFlux

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

“A classic bicycle leaned against an old brick wall”

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

OursFlux

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

“A delicious breakfast spread served on a wooden table”

### Table 3. Detailed metrics for the SD3.5-Large Pareto frontiers in Fig-

- ure 13.

Method Vendi (↑) IR (↑) VQA (↑) KID ×10−4 (↓) Base Model 1.819 1.051 0.905

Ours

𝜂 = 5 · 102 1.851 1.018 0.904 0.619

- 𝜂 = 5 · 104 1.878 1.012 0.904 0.627
- 𝜂 = 5 · 105 1.941 0.988 0.900 0.625
- 𝜂 = 5 · 106 1.980 0.940 0.890 0.445

CADS

𝑠 = 10−12 2.004 0.131 0.717 0.941 𝑠 = 10−10 2.025 0.051 0.692 0.953 𝑠 = 10−08 2.018 0.066 0.692 0.953

PG

𝑠 = 1 1.900 0.783 0.878 1.521 𝑠 = 60 1.913 0.707 0.868 4.053 𝑠 = 80 1.924 0.632 0.861 5.930

SGI

8 Candidates 1.828 1.050 0.903 0.465 16 Candidates 1.862 1.025 0.902 0.455 32 Candidates 1.883 1.030 0.902 0.429 64 Candidates 1.915 1.004 0.901 0.421

SPARKE

- 𝑠 = 0.01 1.860 1.027 0.902 0.362
- 𝑠 = 0.02 1.887 0.999 0.901 0.770
- 𝑠 = 0.03 1.912 0.925 0.899 1.393
- 𝑠 = 0.04 1.989 0.735 0.882 2.918

Table 4. Detailed metrics for the SD3.5-Turbo Pareto frontiers in Fig-

- ure 14.

Method Vendi (↑) IR (↑) VQA (↑) KID ×10−4 (↓) Base Model 1.724 0.978 0.891

- 𝜂 = 1 · 105 1.819 0.914 0.887 1.796 𝜂 = 5 · 105 1.879 0.899 0.884 1.786
- 𝜂 = 1 · 106 1.914 0.864 0.876 1.897
- 𝜂 = 1 · 107 2.079 0.562 0.822 1.914

Ours

𝑠 = 0.1 1.808 0.551 0.772 0.158 𝑠 = 0.5 1.853 0.383 0.731 0.526

CADS

- 𝑠 = 0.8 1.911 0.180 0.683 1.319
- 𝑠 = 0.9 1.958 0.127 0.673 1.348

PG

𝑠 = 2 1.765 0.915 0.884 0.881

- 𝑠 = 10 1.857 0.638 0.859 2.285 𝑠 = 40 1.926 0.221 0.821 14.128

4 Candidates 1.707 0.962 0.888 0.078 8 Candidates 1.775 0.944 0.889 0.079 16 Candidates 1.829 0.933 0.883 0.005 32 Candidates 1.853 0.923 0.884 0.028 64 Candidates 1.879 0.913 0.886 0.120

SGI

𝑠 = 0.04 1.728 1.011 0.890 0.206 𝑠 = 0.08 1.763 0.928 0.885 0.744 𝑠 = 0.1 1.812 0.837 0.871 1.219 𝑠 = 0.12 1.869 0.629 0.850 2.742 𝑠 = 0.14 1.970 0.231 0.803 7.037

SPARKE

while VQAScore remains nearly constant (−0.01). These results demonstrate that intervening in the Contextual Space effectively identifies and navigates remaining semantic degrees of freedom, even in the presence of extensive conditioning.

- Table 5. User study results comparing our method against five competing approaches across four evaluation metrics. Values show the percentage of times users preferred our method (Ours), the competitor (Comp.), or rated both equally (Tie). Results are aggregated from 450 pairwise comparisons per metric.

Metric Choice Base Model CADS SGI PG SPARKE Average

Diversity

Ours 71.6 52.2 56.7 80.0 34.4 61.1 Comp. 12.9 30.0 11.1 14.4 53.1 22.0 Tie 15.5 17.8 32.2 5.6 12.5 16.9

Quality

Ours 49.1 67.8 15.6 82.2 85.9 58.0 Comp. 6.9 11.1 31.1 12.2 3.1 13.1 Tie 44.0 21.1 53.3 5.6 10.9 28.9

Adherence

Ours 25.0 74.4 13.3 67.8 79.7 48.9 Comp. 15.5 11.1 22.2 13.3 4.7 14.0 Tie 59.5 14.4 64.4 18.9 15.6 37.1

Overall

Ours 57.8 74.4 31.1 83.3 87.5 65.1 Comp. 13.8 15.6 27.8 10.0 9.4 15.6 Tie 28.4 10.0 41.1 6.7 3.1 19.3

All Metrics

Ours 50.9 67.2 29.2 78.3 71.9 58.3 Comp. 12.3 16.9 23.1 12.5 17.6 16.2 Tie 36.9 15.8 47.8 9.2 10.5 25.6

- Table 6. Scalability across batch sizes. Quantitative results on SD3.5Turbo for varying batch sizes. We report the average Vendi score per pair to normalize for batch size constraints.

Batch size Vendi Vendi (avg. pair) ImageReward

4 1.819 1.393 0.914 8 2.295 1.401 0.923 16 2.768 1.404 0.928

D Additional Ablation Studies

Attention score repulsion. We investigate the impact of applying the repulsion mechanism directly to the attention scores within Fluxdev. Despite utilizing FlexAttention [Dong et al. 2024] for optimization, this approach introduced significant computational overhead while paradoxically yielding a reduction in both semantic diversity and human preference scores (−0.075 Vendi, −0.178 ImageReward).

Batch size ablation. We examine the scalability of our method by evaluating performance across varying batch sizes on SD3.5-Turbo. To ensure a fair comparison across different sample counts, we report the average Vendi score per pair, as the raw Vendi score is inherently bounded by the batch size. As shown in Table 6, our method exhibits a consistent positive trend across all evaluated metrics as the batch size increases. This suggests that the repulsion mechanism scales effectively and benefits from the denser representation of the conditional manifold provided by larger batches.

Timestep ablation. We analyze the impact of the repulsion window across the diffusion trajectory by applying the intervention within specific timestep intervals while keeping all other hyperparameters constant. Table 7 summarizes these results. For both SD3.5-Large and SD3.5-Turbo, applying repulsion later in the trajectory typically improves ImageReward at the expense of diversity. Conversely, maintaining the intervention throughout the entire trajectory yields

- Table 7. Effect of the timestep interval on diversity and human preference. We evaluate different intervention windows during the diffusion trajectory for SD3.5-Large and SD3.5-Turbo.

Model Timestep interval Vendi ImageReward SD3.5-Turbo [0,1/4] 1.764 0.829

- [1/4,2/4] 1.776 0.811
- [2/4,3/4] 1.809 0.745
- [3/4,1] 1.988 0.660 [0,1] 2.064 0.501

SD3.5-Large [0,1/7] 1.849 0.942

- [1/7,2/7] 1.854 0.942
- [2/7,3/7] 1.849 0.946
- [3/7,4/7] 1.847 0.932
- [4/7,5/7] 1.848 0.954
- [5/7,6/7] 1.900 0.919
- [6/7,1] 1.960 0.852 [0,1] 2.135 0.535

- Table 8. Performance across different transformer block groups. Results are reported for interventions applied to the first, middle, or last third of the blocks for SD3.5-Large and SD3.5-Turbo.

SD3.5-Turbo SD3.5-Large Block group Vendi ImageReward Vendi ImageReward

First third 1.878 0.774 1.887 0.895 Middle third 1.947 0.844 1.947 0.902 Last third 1.765 0.913 1.835 0.985 All blocks 1.764 0.829 1.960 0.852

the highest diversity but results in a more pronounced decline in fidelity and alignment scores.

Transformer block ablation. We further investigate how the selection of transformer blocks influences performance by restricting the intervention to the first, middle, or last third of the architecture’s blocks. As reported in Table 8, applying repulsion to the middle blocks yields the strongest diversity among the partitioned groups, while preserving high preference scores for both SD3.5-Large and SD3.5-Turbo.

