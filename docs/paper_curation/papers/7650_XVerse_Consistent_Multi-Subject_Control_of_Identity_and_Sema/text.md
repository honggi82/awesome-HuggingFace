arXiv:2506.21416v1[cs.CV]26Jun2025

# XVerse: Consistent Multi-Subject Control of Identity and Semantic Attributes via DiT Modulation

Bowen Chen∗, Mengyi Zhao∗, Haomiao Sun∗, Li Chen∗,† Xu Wang, Kang Du, Xinglong Wu Intelligent Creation Team, ByteDance {chenbowen.cbw, zhaomengyi.pl, sunhaomiao, chenli.phd, wangxu.ailab, dukang.daniel, wuxinglong}@bytedance.com

[Figure 1]

- Figure 1: XVerse enables single/multi-subject personalization and the additional control of semantic attributes such as pose, style, and lighting. Input conditions are highlighted with red dots.

∗Equal contribution. †Corresponding author, project lead.

Preprint. Under review.

## Abstract

Achieving fine-grained control over subject identity and semantic attributes (pose, style, lighting) in text-to-image generation, particularly for multiple subjects, often undermines the editability and coherence of Diffusion Transformers (DiTs). Many approaches introduce artifacts or suffer from attribute entanglement. To overcome these challenges, we propose a novel multi-subject controlled generation model XVerse. By transforming reference images into offsets for tokenspecific text-stream modulation, XVerse allows for precise and independent control for specific subject without disrupting image latents or features. Consequently, XVerse offers high-fidelity, editable multi-subject image synthesis with robust control over individual subject characteristics and semantic attributes. This advancement significantly improves personalized and complex scene generation capabilities. Project Page: https://bytedance.github.io/XVerse; Github Link: https://github.com/bytedance/XVerse.

## 1 Introduction

The field of text-to-image generation has advanced remarkably [1; 2; 3; 4; 5; 6], enabling the creation of highly realistic and diverse images from textual descriptions. Initial breakthroughs in personalization focused on single subjects [7; 8; 9; 10]. These methods demonstrated strong control over individual subject appearances and showed good editability. However, the growing demand for more complex visual narratives and personalized content has spurred interest in extending these capabilities to scenarios involving multiple subjects within a single, coherent scene. This transition to multi-subject personalization presents substantial challenges, particularly in preserving individual identity fidelity and alleviating attribute entanglement.

Many current state-of-the-art multi-subject methods [11; 12; 13; 14] tried to leverage the attention mechanism in Diffusion Transformers (DiTs) [15] for injecting information from reference images. But this direct injection or strong reliance on image features can substantially impact the generation quality of the base model. This often leads to artifacts, distortions, attribute entanglement, and can compromise the overall structural integrity and coherence of the generated image. These limitations highlight a critical gap, underscoring the need for novel techniques that offer fine-grained, independent control over multiple subjects while preserving image quality, editability.

To address these limitations, this paper introduces XVerse, a novel method for consistent multi-subject control of identity and semantic attributes. We identify that the inherent modulation vectors within DiT blocks [15], which are typically employed for general conditioning, represent an underexplored yet highly promising pathway to achieve nuanced, subject-specific control. Building upon this insight, XVerse pioneers an approach centered on learning offsets within the text-stream modulation mechanism of DiTs. With the reference images provided, XVerse utilizes an adapter to transform them into share offsets and per-block offsets for token-specific text-stream modulation. This technique allows for condition injection from diverse reference images while preserving the image’s underlying structural integrity. To enhance fine-grained details, we incorporate VAE-encoded image features into the single block of FLUX [6]. Instead of being the main conditioning factor, the VAE-derived features play a supporting role in enhancing details for the backbone network. This strategy successfully minimizes the occurrence of artifacts and distortions, enabling XVerse to achieve exceptional multisubject controlled generation results (as shown in Figure 1). Extensive testing conducted on our benchmark validates XVerse’s exceptional performance in terms of both flexibility of editing and maintaining the appearance of the subject.

Our contributions are summarized as follows:

- • We propose XVerse, a novel framework for fine-grained multi-subject controllable generation. Our approach integrates the reference images into the text-stream modulation offsets. Additionally, we leverages VAE-encoded image features to enhance fine-grained details which are difficult to expresss in the semantic space. This allows XVerse to achieve a high degree of consistency with reference images, while preserving the original diffusion model’s editability by maintaining the image composition.

- • Through training on a high-quality datasets constructed by our data pipeline, XVerse achieves outstanding performance in generation tasks under multi-subject control. Moreover, due to the flexibility of text-stream modulation, XVerse also demonstrates strong generalization performance in tasks such as maintaining semantic information such as posture, lighting, and background.
- • We present a comprehensive benchmark XVerseBench, which evaluates both single-subject and multi-subject controlled image generation. This benchmark provides a rigorous methodology for assessing a model’s ability to support flexible editing, maintain subject characteristics, and preserve distinct identities.

## 2 Related Work

### 2.1 Subject-driven Generation

Subject-driven generation tasks aimed at synthesizing user-specific content, and have made significant progress in recent years. Early endeavors predominantly concentrated on single-subject personalization. These approaches can be broadly classified into two main categories: (1) Fine-tuning-based methods, such as Textual Inversion [7] and DreamBooth [8], which adapt pre-trained models to embed novel concepts from a few exemplar images of a single subject. (2) Tuning-free methods, including IP-Adapter [9] and Photoverse [10], leverage powerful vision encoders to inject subject identity directly into the generation process without test-time fine-tuning, thereby offering enhanced flexibility. However, extending personalization to accommodate multiple subjects within a single, cohesive image presents substantial challenges, particularly in preserving individual identity fidelity and mitigating attribute entanglement. Recent progress leveraging DiTs [15] has begun to address these complexities. For instance, OmniControl [11] demonstrated versatile control by conditioning DiT inputs. Concurrently, a line of research has focused on unified frameworks for complex multisubject image customization, exemplified by UniReal [12], UNO [13], and DreamO [14]. While these methods significantly advance multi-subject generation, their primary strategies involve conditioning the input token sequence, imposing explicit constraints, or modifying attention mechanisms to guide the generative process. XVerse introduces a distinct perspective, operating as a tuning-free method for multi-subject personalization. With the utilization of the text-stream modulation mechanism in DiTs, XVerse can enable precise, subject-specific conditioning while preserving the structural integrity of the generated image

### 2.2 Modulation in Generative Models

Modulation mechanisms have been instrumental in enhancing the controllability and expressiveness of generative models. This concept involves dynamically adjusting a model’s internal activations or parameters based on conditioning information. StyleGAN [16] pioneered the use of modulation by introducing adaptive instance normalization (AdaIN), which modulates features using a style vector. A key discovery associated with this approach was that even small modifications to these modulation parameters could induce smooth and semantically meaningful perturbations in the generated image. This insight spurred a wave of research focused on leveraging modulation layers, leading to significant advancements and successful applications in tasks such as image editing and manipulation [17; 18; 19; 20; 21].

Transformer-based models, such as DiTs, commonly employ modulation mechanisms like adaptive layer normalization (AdaLN) [22]. These mechanisms allow conditioning information (e.g., text embeddings, timesteps) to guide image generation by modulating normalization layers within transformer blocks. Tokenverse [23] explored modulation for subject preservation, they typically rely on fine-tuning strategies. Such methods necessitate extensive training on predefined datasets to learn subject-specific modulations, thus limiting their adaptability to novel, unseen subjects without retraining or inference-time optimization. In contrast, XVerse is designed to inject rich identity and semantic information for arbitrary subjects without such subject-specific fine-tuning.

###### Inject only in single-block

|a man is wearing a shirt sitting besides a cafe table| |
|---|---|
| |latent|

y

[Figure 2]

[Figure 3]

[Figure 4]

c x Linear

Linear

###### t

VAE encoder

| | |
|---|---|
| | |
| | |
| | |

| | |
|---|---|
| | |
| | |
| | |

T-ModAdapter

a man

[Figure 5]

Multi-Head Attention

Token-specific Modulation

Scale & Residual Connection

Scale & Residual Connection

a shirt

Attention

[Figure 6]

Feed Forward

Feed Forward Feed Forward

Modulated Image

DiT

Scale & Sum Scale & Sum

- Figure 2: Overview of the XVerse framework. The reference images are processed by a T-Mod Re-

- sampler and subsequently injected into the per-token modulation adapter. Additionally, to supplement image details, the VAE-encoded features of the reference image are also utilized as input to the single block of DiTs.

- 3 Method

### 3.1 Preliminaries

DiTs have introduced significant advancements in the quality, efficiency, and scalability of image synthesis, establishing DiTs as the foundational architecture for most state-of-the-art models. The Attention Block in DiTs process text and image tokens concurrently within their transformer layers. This attention mechanism provides a pathway for injecting control signals, such as features from control images, into the token representations. However, while injecting control signals through attention can achieve good similarity improvements, this approach can also cause the model’s sampling trajectory to deviate, leading to a reduction in image generation quality. In this work, We try to further explore the potential of the modulation mechanism in DiTs to achieve more precise subject control editing.

Modulation Mechanism in DiTs. The modulation mechanism in DiTs, adapted from techniques popularized by StyleGAN[16], refines neural network activations by applying learned scale factors and bias terms. For instance, models like Stable Diffusion 3 [15] and FLUX [6] utilize a Multi-Layer Perceptron (MLP) to process both the diffusion timestep t and a CLIP [24] embedding of the text prompt (e.g., fp = CLIP(p)), thereby deriving a conditioning vector y:

y = MLP(t,fp). (1)

This vector y is then further processed, typically through a linear layer, to generate a set of modulation parameters. In the DiT architecture, this yields twelve distinct parameters for each block. Six of these parameters modulate text features, and the remaining six modulate image features. The integration of these modulations is commonly achieved using techniques such as Adaptive Layer Normalization (AdaLN) and residual connections. AdaLN is implemented prior to the Attention Layer and Feed-Forward Layer, and can be expressed as:

x − µ(x) σ(x)

AdaLN(x,α,β) = α ·

+ β. (2)

where x is the input feature, µ(x) and σ(x) are its mean and standard deviation respectively, and α (scale) and β (bias) are modulation parameters derived from the conditioning vector y. Residual connections are added after the Attention layer and the Feed-Forward layer. They help stabilize training and enable deeper networks, typically taking the form:

xout = γF(xin) + xin. (3)

where xin is the input to a layer or block, F(xin) is the learned residual mapping, xout is the output, and γ is a scaling parameter applied to the residual term, also derived from the conditioning vector y.

These mechanisms facilitate fine-grained control over feature representations at various stages within the network. Such adaptive modulation significantly enhances the model’s capacity to align image generation with the conditioning inputs.

An essential aspect of this modulation mechanism is its ability to operate separately from the main data flow of the attention mechanism. This separation allows for the precise integration of visual feature representation into specific words without interfering with the denoising process. This has the potential to minimize errors and enhance the overall quality of the generated image. Furthermore, text-based features often offer clearer semantic directionality compared to image-based features. This indicates that manipulating text-stream modulation signals can offer a more straightforward and comprehensible method to control the generative process. Consequently, XVerse focuses on leveraging text-side feature modulation to attain precise control over image synthesis.

### 3.2 Multi-subject Controled Generation

The framework of XVerse is shown in Figure 2. We first leverage the modulation mechanism for its robust preservation of key attributes from the reference images. Subsequently, we inject fine-grained details with the VAE-encoded image features in single blocks of FLUX model. The subsequent sections will detail these enhancements.

Enhanced Text-Stream Modulation with Image Feature Control. To enhance fine-grained control over the generation process, we augment the existing text-stream modulation framework by introducing image features as an additional control signal. Specifically, given a conditioning image Ic, we first extract its deep features using the CLIP model, represented as fc = CLIP(Ic). Subsequently, we combine these image features with the CLIP-encoded features fp of the text prompt, using a perceiver resampler [25] as the text-stream modulation adapter (T-Mod Adapter). Here, the text features fp serve as the query vector, and the resampler outputs an offset ∆cross:

∆cross = Resampler(fp,fc). (4)

This output, ∆cross, encapsulates synergistic information from both the textual prompt and the conditioning image, acting as a corrective or refining signal. For example, if the text prompt p is

about a “handbag” and the accompanying image Ic shows a “brown leather handbag”, ∆cross helps the model adjust its representation from a generic handbag to the specific “brown leather handbag” with a series of visual characteristics (such as material, color, and style).

Crucially, ∆cross is formulated as an offset vector targeting specific tokens. This offset is added to the corresponding token embeddings injected into the model, enabling precise control over the semantics of each token while preserving the structure of the text-to-image result. The adjusted conditioning signal y∗ is thereby derived:

y∗ = MLP(t,fp) + ∆cross. (5) This adjusted conditioning signal y∗ is then utilized to adjust the original modulation parameters (e.g., scale and shift parameters) applied to the network’s activations, thereby refining the conditioning influence. Since our method primarily performs injection in the textual space, it can naturally generalize to the control of high-level semantic attributes such as pose, lighting, and style, without requiring additional differentiation for these conditions.

To further enhance the level of control over the final image output, we draw inspiration from StyleGAN’s expansion of the W latent space to W+, enabling each Transformer block i in our model to receive customized conditioning. This is achieved by decomposing the ∆cross signal. Instead of a single offset, we compute a shared component, ∆shared (applied across all DiT blocks), and individual components for each block i, denoted as ∆iper-block (which can be computed on a per-block or per-stage basis):

yi∗ = MLP(t,fp) + ∆shared + ∆iper-block. (6)

This structured decomposition of the image-derived text offset facilitates more precise and adaptive control over the influence of text conditioning at various levels of the generation process.

Refined Attention Module with Controlled VAE Integration. While using only text-stream modulation can achieve good editability and generation results, its ability to preserve detailed information is still limited. Inspired by OmniControl [11], we introduce VAE-encoded image features as an auxiliary module to further enhance the capability of our approach for maintaining consistency

[Figure 7]

[Figure 8]

Image Captioning

sidewalk

a woman walking down a sidewalk

Florence2 wearing a purple shirt and jeans. Phrase Filtering Face Extraction

[Figure 9]

[Figure 10]

Florence2 & SAM2

[Figure 11]

[Figure 12]

[Figure 13]

+

Phrase Grounding

a woman walking down a sidewalk wearing a purple shirt and jeans.

woman purple shirt jeans sidewalk woman

Figure 3: Training Data Construction Pipeline.

of fine-grained features. To avoid potential negative impacts of directly injecting image features (such as artifacts or degradation of image quality), we constrain the role of VAE features, making them primarily an auxiliary module for supplementing image details, rather than the dominant mechanism for feature injection. Specifically, we restrict the injection of VAE features only to single blocks within the FLUX model. To effectively distinguish different image patch regions within the latent space of the conditioning image, the position index for the latent is changed from (i,j) into individual index as UNO[13].

### 3.3 Regularizations

In modulation space M+, feature vectors derived from highly similar subjects often become entangled, leading to subject confusion and unintended fusion in generated results. To address this, we introduce two critical regularization techniques: region preservation loss and text-image attention loss, which facilitate multi-subject disentanglement.

Region Preservation Loss. We construct a new training sample by randomly selecting two existing samples and concatenating them in a side-by-side left-right configuration, with their captions combined into a unified overall caption. We randomly retain modulation injection from only one side (left or right), rather than both. For regions without modulation injection, we enforce consistency between the model’s output and the T2I branch’s output in those regions using L2 loss.

Lregion = Ex,y ||(1 − Mc) ⊙ (Vθ′(zt,t,y∗) − Vθ(zt,t,y))||22 (7)

where zt denotes the noisy latent at timestep t, Vθ(zt,t,y) denotes T2I branch’s output, Vθ′(zt,t,y∗) denotes the XVerse’s output, and 1 − Mc denotes the non-modulated mask region. This approach not only regularizes subject-specific features but also serves as a data augmentation strategy for multisubject datasets, enhancing the model’s ability to distinguish and preserve subject characteristics.

Text-Image Attention Loss. To maintain the compositional and editability properties of the T2I branch after modulation injection, we align the cross-attention dynamics between two branches. Specifically, we compute an L2 loss with normalization over the text-image cross-attention maps of the modulated model and the reference T2I branch. This encourages the modulated model to retain attention patterns that closely match those of the T2I branch, ensuring that semantic interactions between text and image regions remain consistent and editable.

### 3.4 Collection of Training data

High-quality multi-entity data remains scarce, and modulation injection necessitates knowledge of the corresponding text tokens in the target prompt for each conditional image. To address this, we introduce a high-quality general-purpose multi-entity data annotation pipeline.

Single-Image Data Construction. We curated a 1M-scale dataset of images with resolutions exceeding 512 pixels, constructing a universal multi-entity single-image dataset via the data workflow depicted in 3. Specifically, we first employ Florence2 [26] for joint image caption generation and phrase grounding, followed by large language model (LLM)-driven phrase filtering and classification to exclude non-entity terms (e.g., “sky”, “water surface”) that do not align with our definitional criteria. Subject segmentation is then performed using SAM2 [27], with additional face detection and extraction applied specifically for human entities. While open-source data includes diverse scenarios, challenges such as suboptimal aesthetic quality and scene complexity impede model learning. To mitigate this, we supplement the dataset with an additional 1M-scale corpus of high-aesthetic-quality images synthesized using FLUX [6].

Cross-Image Data Construction. For human-centric data, we harvested a 100K-scale single-person multi-view dataset from proprietary in-house collections, forming up to three image pairs per subject

[Figure 14]

[Figure 15]

[Figure 16]

- Figure 4: Data distribution and samples for XVerseBench. XVerseBench includes evaluations of single-subject, dual-subject, and triple-subject controlled image generation. The figure also illustrates the number of test samples allocated to each category.

ID. For general objects, we leveraged the Subject200K [11] dataset. The construction workflow mirrors that of single-image data, with domain-specific refinements: human data incorporates ID similarity thresholds, while general object data employs DINOv2 [28] similarity filtering—both applied independently to enforce cross-view consistency.

## 4 Experiments

### 4.1 Experiments Setting

Implementation Details. In XVerse, we design a three-stage training pipeline to achieve precise multi-subject controlled image generation. We first train the text-stream modulation adapter to establish foundational semantic alignment. It serves as the basis for conditional feature injection which ensures high-level consistency between injected images and generated results. Building upon the first stage, we introduce VAE-encoded features to enhance detail preservation. While maintaining the global structure and key characteristics obtained from stage one, this phase focuses on injecting fine-grained visual details through hierarchical feature fusion. The first two stages employ hybrid training data containing both single-subject and multi-subject samples from our single-image dataset. However, we found that such a reconstruction paradigm tends to produce copy-paste effects during inference, resulting in reduced diversity and compromised structural fidelity. To address this issue, we propose a third training phase incorporating cross-image data where the subject appearances differ between injected and target images. This extension enhances the capability of our approach to establish robust appearance mapping under heterogeneous visual conditions. Throughout all stages, we maintain mixed training on both single-subject and multi-subject datasets.

Based on FLUX.1-dev [6] text-to-image generation model, we employ LoRA [29] with a rank of 128 to efficiently fine-tune the model while maintaining its generalization capabilities. We utilize two three-layer resamplers with an intermediate dimension of 3072 to generate the shared offsets and per-block offsets for text-stream modulation. The model was trained for 70K, 150K, and 10K iterations per stage respectively. Both the text-stream modulation adapter and LoRA layers are optimized using a learning rate of 5e-6. The weight of region preservation loss is 10 and the weight of text-image attention loss is 0.01.

XVerseBench Details. Existing controlled image generation benchmarks often focus on either maintaining identity or object appearance consistency, rarely encompassing datasets that rigorously test both aspects. To comprehensively evaluate the model’s ability in single-subject and multi-subject condition generation and editing, we expanded the dataset based on DreamBench++ [30], adding 20 newly generated portrait images to construct a new benchmark.

Our resulting benchmark XVerseBench comprises 20 distinct human identities, 74 unique objects, and 45 different animal species/individuals. To thoroughly evaluate model effectiveness in subject-driven generation tasks, we developed test sets specifically for single-subject, dual-subject, and triple-subject control scenarios. This benchmark includes 300 unique test prompts covering diverse combinations of humans, objects, and animals. Figure 4 shows more detail information and samples for each

Table 1: Quantitative results of single-subject and multi-subject driven generation on XVerseBench.

Single-Subject Multi-Subject

Overall Method DPG ID-Sim IP-Sim AES AVG DPG ID-Sim IP-Sim AES AVG

MS-Diffusion [34] 96.89 6.52 55.71 59.63 54.69 87.21 3.77 46.21 55.91 48.28 51.49 MIP-Adapter [35] 87.56 39.59 71.97 52.12 62.81 84.56 24.58 57.00 51.81 54.49 58.65 OmniGen [36] 83.90 76.51 78.46 51.41 72.57 78.23 55.53 62.32 49.84 61.48 67.03 UNO [13] 89.65 47.91 80.40 55.90 68.47 85.28 31.82 67.00 54.24 59.59 64.03 OmniGen2 [37] 92.60 62.41 74.08 52.34 70.36 91.55 40.81 67.15 51.40 62.73 66.55 DreamO [14] 96.93 75.48 70.84 54.57 74.46 88.80 50.24 64.63 52.47 64.04 69.25 XVerse (Ours) 93.69 79.48 76.86 56.84 76.72 88.26 66.59 71.48 53.97 70.08 73.40

###### Inputs XVerse DreamO UNO OmniGen MIP MS-Diffusion

##### Inputs XVerse DreamO UNO OmniGen MIP MS-Diffusion

###### OmniGen2OmniGen2

[Figure 17]

[Figure 18]

|[Figure 19]|
|---|

|[Figure 20]|
|---|

[Figure 21]

[Figure 22]

[Figure 23]

a person standing in front of a hut

a person standing in front of a hut

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

|[Figure 28]|[Figure 29]|
|---|---|

|[Figure 30]|
|---|

|[Figure 31]|
|---|

A man and a woman standing side by side in a park.

A man and a woman standing side by side in a park.

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

|[Figure 37]|[Figure 38]|
|---|---|

|[Figure 39]|
|---|

|[Figure 40]|
|---|

An old man and a man standing together on the street.

An old man and a man standing together on the street.

[Figure 41]

[Figure 42]

|[Figure 43]|[Figure 44]|
|---|---|

|[Figure 45]|
|---|

|[Figure 46]|
|---|

[Figure 47]

[Figure 48]

A man is walking with a dog on the street.

A man is walking with a dog on the street.

|[Figure 49]|[Figure 50]| |[Figure 51]<br><br>[Figure 52]| |[Figure 53]|[Figure 54]|
|---|---|---|---|---|---|---|

[Figure 55]

[Figure 56]

[Figure 57]

A robin is perched on a branch above while a white tiger slowly approaches a curious kitten in a forest.

A robin is perched on a branch above while a white tiger slowly approaches a curious kitten in a forest.

Figure 5: Qualitative comparison with different methods on XVerseBench.

categories. For evaluation, we employ a suite of metrics to quantify different aspects of generation quality and control fidelity: including DPG score [31] to assess the model’s editing capability, Face ID similarity [32] and DINOv2 [28] similarity to assess the model’s preservation of human identity and objects, and Aesthetic Score [33] to measure to evaluate the aesthetics of the generated image. XVerseBench aims to provide a more challenging and holistic evaluation framework for state-of-the-art multi-subject controllable text-to-image generation models.

### 4.2 Comparisons with State-of-the-art Methods

We compared our proposed XVerse method with several leading multi-subject driven generation techniques, including MS-Diffusion [34], MIP-Adapter [35], OmniGen [36], UNO [13], OmniGen2 [37], and DreamO [14]. The generation results were evaluated for both single-subject and multisubject tasks, with quantitative findings presented in Table 1. Our XVerse method achieves the highest Overall score of 73.40, significantly outperforming all other compared methods. This clearly indicates a strong comprehensive advantage of our approach.

In the single-subject generation category, XVerse demonstrates exceptional performance, securing the top AVG score of 76.72. This underscores its robust capability in generating high-quality images focused on individual subjects. Notably, XVerse achieves the best similarity score of identity (IDSim, 79.48) , suggesting superior preservation of condition images. While DreamO leads in DPG with 96.93, XVerse’s strong average performance, bolstered by competitive scores in similarity and aesthetic, highlights its well-rounded excellence.

Meanwhile, XVerse truly excels in the more challenging multi-subject generation tasks, achieving a leading AVG score of 73.40. This remarkable performance can be attributed to XVerse’s novel approach of learning offsets within the text-stream modulation mechanism of DiT. This allows for precise conditioning from diverse image types while crucially preserving the image’s structural integrity. Furthermore, the careful integration of VAE-derived features for detail refinement, rather than dominant conditioning, effectively mitigates artifacts and distortions, which is particularly vital for maintaining clarity and attribute disentanglement in multi-subject scenarios.

- Figure 5 presents a qualitative comparison of our method with other state-of-the-art approaches. As illustrated, our model demonstrates a superior capability in maintaining the consistency and relevance between identities and associated objects within the generated images. This is a direct result of our refined modulation strategy. By carefully adjusting the modulation offsets, XVerse achieves enhanced text-image alignment, which is particularly evident in its accurate depiction of object quantities and the relationships between multiple subjects. Furthermore, when compared to existing methods, our model consistently produces images with a higher degree of naturalness and visual plausibility. This improved realism underscores the advantages of our approach in editing the text-stream modulation pathway, allowing for more faithful and aesthetically pleasing image synthesis.

[Figure 58]

- Figure 6: Effect of text-stream modulation resampler and VAE-encoded image features.

[Figure 59]

Figure 7: The control of sementic attributes, such as cloth, pose, lighting, and style.

- 4.3 Ablation Studies

To further evaluate the effects of different modules in XVerse, we conducted a series of ablation studies. Figure 6 shows the qualitative comparison in both dual-subject and triple-subject controlled generation tasks. The experimental results show that by converting the reference image to textstream modulation offset vectors, XVerse achieves the personalization of the condition subjects while maintaining the composition identical to the original T2I outcomes. Additionally, the VAE features play a supportive role by helping the model add specific details without compromising the overall structure and semantic coherence established by the text-stream modulation. This collaborative approach allows XVerse to maintain subject consistency while achieving a high degree of editability.

- 4.4 Applications

Figure 7 demonstrates XVerse’s capability to control semantic attributes of the generated image. XVerse exhibits precise control over attributes such as lighting, subject posture, clothing, and artistic style. By injecting reference images toward targeted words, XVerse can manipulate these semantic attributes without the need for extensive training data on specific attribute categories. These results further underscore the model’s exceptional ability to generalize and edit effectively.

- 5 Conclusion

In this paper, we introduce XVerse, an innovative framework designed to excel in the complex task of precise multi-subject control within DiTs. XVerse injects reference image features through modulation offsets and control the token-specific representation within the DiT blocks. This approach facilitates adaptive, per-subject conditioning by precisely governing how textual embeddings are transformed and integrated throughout the diffusion process. Consequently, XVerse effectively mitigates common generation issues such as attribute entanglement and artifacts, demonstrating particular excellence in both single-subject and multi-subject controlled generation tasks.

Limitations. While XVerse showcases significant progress in fine-grained multi-subject control, there are certain limitations that need to be addressed and opportunities for future research. One major challenge is the lack of high-quality, large-scale cross-image multi-subject datasets, which are essential for training and evaluating models that can understand and generate complex interactions between subjects. Additionally, our current research has mainly focused on the text-stream modulation pathway, leaving room for exploration in utilizing image-modulation techniques for precise pixel-level or region-specific control.

## References

- [1] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in neural information processing systems 27, 2014.
- [2] Diederik P Kingma and Max Welling. Auto-encoding variational Bayes. arXiv preprint arXiv:1312.6114, 2013.
- [3] Jascha Sohl-Dickstein, Eric A Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015.
- [4] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems, volume 33, pages 6840–6851, 2020.
- [5] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.
- [6] Black Forest Labs. Flux: Official inference repository for flux.1 models, 2024. Accessed: 2024-11-12.
- [7] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In The Eleventh International Conference on Learning Representations, 2023.
- [8] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500–22510, 2023.
- [9] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.
- [10] Li Chen, Mengyi Zhao, Yiheng Liu, Mingxu Ding, Yangyang Song, Shizun Wang, Xu Wang, Hao Yang, Jing Liu, Kang Du, et al. Photoverse: Tuning-free image customization with text-to-image diffusion models. arXiv preprint arXiv:2309.05793, 2023.
- [11] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 2024.
- [12] Xi Chen, Zhifei Zhang, He Zhang, Yuqian Zhou, Soo Ye Kim, Qing Liu, Yijun Li, Jianming Zhang, Nanxuan Zhao, Yilin Wang, et al. Unireal: Universal image generation and editing via learning real-world dynamics. arXiv preprint arXiv:2412.07774, 2024.
- [13] Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei Ding, and Qian He. Less-tomore generalization: Unlocking more controllability by in-context generation. arXiv preprint arXiv:2504.02160, 2025.
- [14] Chong Mou, Yanze Wu, Wenxu Wu, Zinan Guo, Pengze Zhang, Yufeng Cheng, Yiming Luo, Fei Ding, Shiwen Zhang, Xinghui Li, et al. Dreamo: A unified framework for image customization. arXiv preprint arXiv:2504.16915, 2025.

- [15] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023.
- [16] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019.
- [17] Omer Tov, Yuval Alaluf, Yotam Nitzan, Or Patashnik, and Daniel Cohen-Or. Designing an encoder for stylegan image manipulation. ACM Transactions on Graphics (TOG), 40(4):1–14, 2021.
- [18] Erik Härkönen, Aaron Hertzmann, Jaakko Lehtinen, and Sylvain Paris. Ganspace: Discovering interpretable gan controls. In Advances in neural information processing systems, volume 33, pages 9841–9850, 2020.
- [19] Elad Richardson, Yuval Alaluf, Or Patashnik, Yotam Nitzan, Yaniv Azar, Stav Shapiro, and Daniel Cohen-Or. Encoding in style: a stylegan encoder for image-to-image translation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2287–2296, 2021.
- [20] Daniel Roich, Ron Mokady, Amit H Bermano, and Daniel Cohen-Or. Pivotal tuning for latent-based editing of real images. ACM Transactions on graphics (TOG), 42(1):1–13, 2022.
- [21] Jiapeng Zhu, Yujun Shen, Deli Zhao, and Bolei Zhou. In-domain gan inversion for real image editing. In European conference on computer vision, pages 592–608, 2020.
- [22] Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.
- [23] Daniel Garibi, Shahar Yadin, Roni Paiss, Omer Tov, Shiran Zada, Ariel Ephrat, Tomer Michaeli, Inbar Mosseri, and Tali Dekel. Tokenverse: Versatile multi-concept personalization in token modulation space. arXiv preprint arXiv:2501.12224, 2025.
- [24] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763, 2021.
- [25] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35: 23716–23736, 2022.
- [26] Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan. Florence-2: Advancing a unified representation for a variety of vision tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4818–4829, 2024.
- [27] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.
- [28] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [29] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1

(2):3, 2022.

- [30] Yuang Peng, Yuxin Cui, Haomiao Tang, Zekun Qi, Runpei Dong, Jing Bai, Chunrui Han, Zheng Ge, Xiangyu Zhang, and Shu-Tao Xia. Dreambench++: A human-aligned benchmark for personalized image generation. In The Thirteenth International Conference on Learning Representations, 2025.

- [31] Xiwei Hu, Rui Wang, Yixiao Fang, Bin Fu, Pei Cheng, and Gang Yu. Ella: Equip diffusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135, 2024.
- [32] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019.
- [33] discus0434. Aesthetic predictor v2.5: Siglip-based aesthetic score predictor. https://github. com/discus0434/aesthetic-predictor-v2-5, 2024. Accessed: 2024-12-08.
- [34] Xierui Wang, Siming Fu, Qihan Huang, Wanggui He, and Hao Jiang. Ms-diffusion: Multisubject zero-shot image personalization with layout guidance. In The Thirteenth International Conference on Learning Representations, 2025.
- [35] Qihan Huang, Siming Fu, Jinlong Liu, Hao Jiang, Yipeng Yu, and Jie Song. Resolving multicondition confusion for finetuning-free personalized image generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 3707–3714, 2025.
- [36] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024.
- [37] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, Ze Liu, Ziyi Xia, Chaofan Li, Haoge Deng, Jiahao Wang, Kun Luo, Bo Zhang, Defu Lian, Xinlong Wang, Zhongyuan Wang, Tiejun Huang, and Zheng Liu. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025.

## A Samples for Training Dataset

[Figure 60]

Figure 8: Examples of training data for multi-subject controlled generation.

Figure 8 presents examples of our training data for multi-subject controlled generation. As illustrated, the dataset covers a diverse range of scenarios, including human-object interactions, human-animal compositions, and complex multi-person scenes. For human-centric data, we intentionally randomly select facial images or full-body images as reference inputs. This strategy can further enhance the model’s generalization performance. By utilizing this diverse and extensive dataset, which encompasses a wide range of scene variations and control types, our model is able to achieve impressive editing capabilities while maintaining high consistency with the reference images.

## B Impact of Prompt Variation on the Generated Image

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

|[Figure 65]|
|---|

| | |
|---|---|
| | |

[Figure 66]

a man is sitting on a tree stump playing a guitar

a xxxxx is sitting on a tree stump playing a guitar

a woman is sitting on a tree stump playing a guitar

[Figure 67]

[Figure 68]

a woman is sitting on a tree stump playing a guitar

a dog is sitting on a tree stump playing a guitar

a woman is sitting on a tree stump playing a guitar

a woman with long black hair wearing a scarf is sitting on a tree stump playing a guitar

- Figure 9: Impact of prompt variation on subject-controlled Image generation. The reference image and the initial output of our text-to-image generation model are shown on the left side. The right side illustrates the influence of different prompts on the generated output, with the prompt variances highlighted in red.

To evaluate the impact of prompt variation on subject-controlled image generation, we modified the injected words in the per-token text-modulation module while keeping the reference image constant. The results, shown in Figure 9, offer valuable insights. This visualization effectively illustrates that a more detailed prompt description improves the preservation of the subject’s identity in the generated image. Additionally, when the prompt closely matches the reference image, our model not only incorporates intricate image details but also maintains a high level of control over the subject’s attributes, even allowing for successful changes such as gender. On the other hand, if there is a significant semantic mismatch between the injected prompt and the reference image (e.g., trying to generate a person’s image from prompts like “a dog” or “a tree stump”), the injection process consistently fails. This highlights our model’s ability to accurately target and incorporate reference image features into specific words, enabling precise control over the generated output.

## C Comparison of the CLIP-T and DPG scores

When evaluating text-to-image generation models, the CLIP-T [24] score has been a prevalent metric in prior studies, assessing semantic consistency by leveraging CLIP’s image-text embeddings. However, our research highlights the superior efficacy of the DPG (Dense Prompt Graph) score, particularly for intricate prompts. While CLIP-T offers a broad measure of semantic alignment, the DPG score is specifically designed to evaluate a model’s capacity to interpret and execute detailed and complex textual instructions. It rigorously assesses editing abilities across multiple objects, diverse attributes, and intricate relationships, thereby capturing the nuanced and fine-grained semantic alignment crucial for advanced compositional generation. This provides a more comprehensive and robust evaluation for challenging scenarios.

## D Illustration of Region Preservation Loss

[Figure 69]

Figure 10: Illustration of the region preservation loss.

- Figure 10 shows the illustration of our region preservation loss. We form training samples by concatenating two existing samples, merging their captions, and randomly applying modulation to only one side. For the unmodulated regions, defined by Mc, we enforce consistency between our model’s output (Vθ′(zt,t,y∗)) and the text-to-image branch’s output (Vθ(zt,t,y)) via an L2 loss

(Eq. 1). By using this regularization, XVerse can better inject the reference image into specific areas without affecting the generation of irrelevant areas, thereby achieving more precise generation control.

## E Ablation Study for Text-Image Attention Loss

Generated Image A woman coffee cup pink suit

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

Text-to-image

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

XVerse w/o CAL

A woman holding a coffee cup, wearing a pink suit standing on the street

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

XVerse

Figure 11: The qualitative comparsion of Text-Image Attention Loss. This image shows the generated results and attention maps for "woman", "coffee cup", and "pink suit" for each method.

To validate the effectiveness of our text-image attention loss, we conducted an experiment where we excluded this regularization and examined the generated outputs along with their respective attention maps. The qualitative analysis presented in Figure 11 clearly demonstrates the significance of this method. It demonstrates our method’s ability to maintain the structural and editable characteristics of the T2I branch following modulation injection. Through ensuring L2 consistency between the cross-attention maps of the modulated model and the reference T2I branch, our approach ensures the reliable preservation of text-image semantic interactions. This ultimately enables precise control over semantics, as visually evidenced by the generated results and attention maps for specific prompts like "woman," "coffee cup," and "pink suit."

## F Broader Impacts

Our model, XVerse, marks a significant leap in multi-subject controllable text-to-image generation, leading to enhanced fidelity and editability. This breakthrough holds substantial positive societal impacts, particularly within the creative industries, where it can revolutionize the creation of personalized and complex visual content. Furthermore, XVerse can transform education and training by providing more engaging and tailored visual aids, and contribute to content inclusivity by enabling the representation of a wider range of individuals and scenarios.

However, this powerful technology also presents potential negative societal impacts. The improved generation capability could lead to misinformation and deepfakes, raise privacy concerns if used improperly, and potentially amplify biases present in training data. As foundational research, XVerse isn’t directly tied to deployment. Yet, we believe it’s crucial to acknowledge these risks. Future work will explore mitigation strategies like content detection and ethical guidelines, contributing to the responsible advancement of generative AI.

