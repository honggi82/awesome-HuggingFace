## arXiv:2511.06876v1[cs.CV]10Nov2025

### Generating an Image From 1,000 Words: Enhancing Text-to-Image With Structured Captions

Eyal Gutflaish * Eliran Kachlon * Hezi Zisman Tal Hacham Nimrod Sarid Alexander Visheratin Saar Huberman Gal Davidi Guy Bukchin Kfir Goldberg Ron Mokady †

BRIA AI

[Figure 1]

Figure 1. Generating an image from a long structured caption. Training with long structured captions substantially improves controllability and expressiveness. Unlike short prompts, each scene element is specified with rich attributes (e.g., position, size, texture, and relations), while global properties (e.g., background, lighting, composition, and photographic characteristics) are explicitly defined. The result is a complete, machine-readable description that enables faithful, controllable generation for professional use.

#### Abstract

Text-to-image models have rapidly evolved from casual creative tools to professional-grade systems, achieving unprecedented levels of image quality and realism. Yet, most models are trained to map short prompts into detailed images, creating a gap between sparse textual input and rich visual outputs. This mismatch reduces controllability, as models often fill in missing details arbitrarily, biasing toward average user preferences and limiting precision for professional use. We address this limitation by training the first open-source text-to-image model on long structured captions, where every training sample is annotated with the same set of fine-grained attributes. This design maximizes expressive coverage and enables disentangled control over visual factors. To process long captions efficiently, we pro-

pose DimFusion, a fusion mechanism that integrates intermediate tokens from a lightweight LLM without increasing token length. We also introduce the Text-as-a-Bottleneck Reconstruction (TaBR) evaluation protocol. By assessing how well real images can be reconstructed through a captioning–generation loop, TaBR directly measures controllability and expressiveness—even for very long captions where existing evaluation methods fail. Finally, we demonstrate our contributions by training the large-scale model FIBO, achieving state-of-the-art prompt alignment among open-source models. Model weights are publicly available at https://huggingface.co/briaai/FIBO to foster future research.

*Equal contribution. †Project lead.

#### 1. Introduction

Text-to-image models have transformed the way visual assets are created [33, 35, 36]. What began as casual tools for generating playful pictures has rapidly evolved into professional-grade systems delivering substantial creative and commercial value. Within only a few years, these models have achieved unprecedented levels of realism and visual fidelity [7, 11, 18, 43].

However, most existing models are trained to map short natural language prompts into highly detailed images. This asymmetric training creates a fundamental gap between the sparse representation of user input and the rich details required to produce a high-quality image. As a result, the model often fills in missing information arbitrarily, “guessing” the user’s intent [16, 29, 34, 44]. To mitigate this, many models are tuned toward “average” human preferences, optimizing for majority choices in user studies [17, 23, 41]. While this yields pleasing results for casual users, it limits both precision and creativity for professionals who require fine-grained control over composition, lighting, depth of field, and other visual factors. Indeed, the adage “an image is worth a thousand words” highlights the mismatch: short prompts cannot fully specify an image.

To address this limitation, we propose training text-toimage models with long structured captions. These captions encode significantly more visual detail than any prior open-source dataset, and their structured form ensures that each training image is paired with a precise textual description. Unlike existing approaches that bias toward human preference, our captions are designed to maximize expressive coverage, enabling the model to generate every plausible visual configuration. We show that models trained with long structured captions exhibit not only improved prompt adherence but also superior controllability: their disentangled representations allow modification of specific visual factors while leaving others unchanged. Since users may struggle to author such lengthy prompts, we employ a vision–language model (VLM) to bridge natural human intent and structured prompts, making detailed prompting practical in real-world use.

Yet, training with long captions introduces a new efficiency challenge. To address this, we propose DimFusion, a more efficient text–image fusion mechanism. Like prior methods, DimFusion leverages intermediate tokens from an LLM and integrates them directly with image tokens. Unlike them, it combines LLM layers without increasing the number of tokens. Remarkably, we find that even a relatively small LLM suffices for encoding structured prompts.

Beyond modeling, we propose a new evaluation paradigm, Text-as-a-Bottleneck Reconstruction (TaBR), specifically designed to assess alignment with extremely long captions. Since humans struggle to reliably evaluate prompts containing thousands of words, we anchor the eval-

[Figure 2]

[Figure 3]

Caption: “A knight and a horse”

Caption-to-JSON (VLM)

###### FIBO

[Figure 4]

[Figure 5]

Refiner (VLM)

###### FIBO

Refine: “Remove helmet and change horse to white”

Figure 2. Workflow. A short caption is expanded by a VLM into a detailed JSON used by FIBO to generate an image. The user can then refine the JSON, and FIBO produces a new image that reflects only the requested changes, demonstrating strong disentanglement between modified and preserved elements.

uation in images. TaBR integrates an image captioner into the pipeline and measures expressive power via a reconstruction protocol: given a real image, we caption it, regenerate it from the caption using the text-to-image model, and evaluate similarity to the original. While similarity can be measured automatically, we find human judgments to perform better. This image-grounded evaluation is more objective than standard user-preference scoring, reflects the expressive power of the caption–generation loop, and enables systematic optimization toward controllability rather than preference-driven popularity.

Finally, we introduce FIBO, a large-scale text-to-image model capable of generating an image from structured long captions. FIBO demonstrates the effectiveness of our approach, achieving state-of-the-art prompt adherence even for captions exceeding 1,000 words. Beyond fidelity, FIBO exhibits novel disentanglement capabilities, enabling intuitive and fine-grained control over individual image factors such as color, expression, or composition without unintended changes elsewhere. To foster community progress, we release FIBO’s model weights, code, and full implementation details. Our contributions are as follows:

- • We release FIBO, the first open-source text-to-image model trained entirely on long structured captions.
- • We introduce DimFusion, a novel mechanism to efficiently fuse intermediate LLM tokens into image generation models.
- • We propose TaBR, a new evaluation protocol that measures expressive power via caption–generation–reconstruction, shifting focus from user preference optimization to controllability.

#### 2. Related Work

Text-to-image models. Diffusion models have rapidly become the workhorse of text-to-image synthesis. Early systems, like GLIDE [27], Imagen [36], and DALL·E 2 [33], established the power of conditioning diffusion on strong language encoders. Latent diffusion [35] made large-scale training practical, and SDXL [30] pushed

UNet-based scaling. A newer wave pivots to transformer backbones and flow-matching objectives: Stable Diffusion 3 [11] adopts DiT-style architectures with bidirectional text–image mixing, while FLUX [18], HiDream-I1 [7], and Qwen-Image [43] further advance this paradigm. Together, these advances mark a shift toward models that must understand and accurately follow long prompts—the challenge our work is designed to tackle.

Synthetic captions. Early text-to-image models were trained on large-scale web datasets such as LAION [37], where captions derived from HTML alt text provided only coarse and noisy supervision. DALL·E 3 [4] demonstrated that augmenting training data with synthetically generated, descriptive captions improves both human and automated evaluations, allowing models to capture visual concepts that human annotators often omit. Subsequent works [11, 22] explored hybrid datasets that mix human-written and synthetic captions to balance linguistic diversity and descriptive precision. We extend this idea by using modern visionlanguage models to produce structured JSON captions that explicitly describe object attributes, spatial relations, composition, and photographic style—capturing fine-grained detail beyond prior open-source efforts. The same structured schema is used consistently during training and inference, enabling human-written prompts to be automatically converted into machine-readable structured descriptions.

LLM Fusion. Earlier text-to-image models relied on CLIP [31] or T5 [32] as text encoders [30, 35, 36], mapping raw text to a compact, semantically meaningful representation. Recent work explores leveraging LLMs for text encoding to exploit their richer semantics and compositional reasoning. Because different intermediate LLM layers encode complementary linguistic information [9, 10, 35], several methods, e.g., Playground v3 [22] and Tang et al. [39], fuse multiple layers into diffusion transformers via crossattention. However, these approaches typically lack bidirectional text–image mixing (i.e., joint attention), which has been shown to improve prompt adherence [11]. HiDreamI1 [7] use a dual-stream, decoder-only LLM but at high computational cost, since each attention block consumes tokens from both final and intermediate layers. Our proposed DimFusion improves efficiency by leveraging both intermediate and final LLM representations while keeping token length constant.

Cycle-consistency-based evaluation. Cycle consistency is the principle that translating an input from one domain to another and back should approximately reconstruct the original (e.g., image→text→image). A prominent example is the self-supervised objective in CycleGAN [47]. In textto-image generation, recent works use cycle consistency as

“A bear is performing a handstand in the park.”

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

“A woman writing with a dart.”

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

“A professional boxer does a split.”

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

FLUX HiDream Qwen-Image Fibo (Ours)

Figure 3. Contextual contradiction. We use prompts from ContraBench [16] and Whoops [5]. FIBO generates semantically consistent images that faithfully represent the correct relations, unlike other models that often default to typical co-occurrences.

a reward in reinforcement settings to improve prompt adherence [1, 25], and as an automatic metric for text–image alignment [15]. Meng et al. [25] further propose an automated evaluation based on image regeneration via captions, but without a direct pixel-level or perceptual comparison to the original image. In contrast, we introduce Text-as-aBottleneck Reconstruction (TaBR), which given real image, produce a detailed caption, regenerates the image from this caption, and then explicitly compares the reconstruction to the original. This image-grounded protocol provides a more direct measure of expressive power and controllability.

#### 3. Method

In this section, we present the main contributions of the paper. Section 3.1 introduces the motivation for training with long structured prompts and analyzes their effect on controllability and disentanglement. We also present our workflow, as demonstrated in Figure 2. Section 3.2 introduces DimFusion, our novel architecture for integrating an LLM with a text-to-image model under reduced computational requirements. Section 3.3 presents a novel evaluation protocol tailored for long captions, addressing the limitations of existing methods when handling inputs exceeding 1,000 words. Finally, Section 3.4 describes our large-scale training setup, resulting in our model, FIBO. Additional implementation details are provided in the Appendix.

“Shallow depth of field” “Deep depth of field”

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

FLUX FIBO (Ours) Flux FIBO (Ours)

“Three people: Left one holding a goose while screaming. Middle is deeply surprised. Third is crying while holding two piglets.”

[Figure 22]

[Figure 23]

FLUX FIBO (Ours)

Figure 4. Controllability examples. Top: Our model (FIBO) enables precise depth-of-field control, from shallow to deep, whereas FLUX [18] does not consistently produce deep depth of field. Bottom: FIBO allows simultaneous control of facial expressions for multiple subjects within the same scene.

Long structured captions training

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Short captions training

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Caption Type FID ↓ Long structured captions 19.01 Short captions 34.04

Figure 5. Training with long structured captions vs. short captions. Top: Qualitative comparison (256X256 resolution). Training with long captions produces more coherent and visually detailed images, showing faster convergence and better alignment. Bottom: Quantitative results on 30K COCO-2014 val images [20]: long captions yield lower (better) FID.

##### 3.1. Training with Long Structured Captions

While the prompt adherence of text-to-image models has steadily improved, even state-of-the-art models often struggle with complex scenes. For example, current models fail to generate contextual contradictions, reliably capture distinct facial expressions for multiple individuals, or to respect subtle specifications such as lighting conditions and camera parameters, as can be seen in Figures 3 and 4.

We attribute this to a fundamental limitation in training data: most captions lack detailed descriptions of finegrained visual factors. Recent works have demonstrated that training with synthetic captions outperforms the use of default human-written captions [11, 22]. However, synthetic captions are not exhaustive: some details are omitted depending on the judgment of the captioning model. As a result, crucial information is often absent during training, introducing ambiguity for the generator. Consequently, the model learns to ignore prompt instructions often, as it has become accustomed to incomplete conditioning.

To address this, we train with long structured captions, in which crucial details are always represented. By ensuring that essential aspects such as colors, lighting, or composition are consistently present, we remove ambiguity and establish a stronger alignment between text and image. This explicit conditioning encourages the generator to ground its outputs in the caption rather than relying on priors.

We observe that training with long structured captions results in faster convergence and improved image quality

compared to short captions. To validate this, we train a low resolution small model (1B parameters) for 100K steps using the same images, comparing between long structured captions and short captions. Both qualitative and quantitative comparisons are in Figure 5 demonstrate that training with long structured captions converges faster and achieves superior results compared to short captions. Complete experimental details are provided in the Appendix (Section C.

Surprisingly, disentanglement emerges natively from our structured representation in an unsupervised manner—the model never observes paired images during training. In standard text-to-image models, modifying a single prompt detail often induces unintended changes in unrelated factors. In contrast, our model enables precise editing of generated images: altering one attribute in the structured caption typically affects only the corresponding visual factor, without requiring additional editing techniques [14, 26]. This disentanglement arises naturally from conditioning on consistently detailed prompts and is illustrated in Figure 6. Notably, this is not an image-editing setup, as the model does not take an input image. We modify only the structured prompt while holding the random seed fixed.

The full schema of our structured captions is provided in the Appendix (Section A). Each caption begins with the most salient details of the image, including objects, background, and text. Objects are described with rich attributes such as size, position, shape, and color. Humans are also annotated with pose, expression, ethnicity, etc. We then incorporate information that is often missing in captions, includ-

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

black mug to white apple to orange add coffee to mug

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

blue sign to red make it evening set camera from above

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

change text to “water”

season is autumn add rain drops

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

remove helmet make sunny knight riding horse

- Figure 6. Structured captions encourage disentanglement. Our model allows iterative refinement: altering one attribute in the JSON typically affects only the corresponding visual factor. Starting from the left images, we demonstrate iteratively editing, where the label below describes the requested change in the JSON.

###### Avg. Median Std. Dev. Min Max #Tokens 1160.3 1176 316.4 192 1800

Table 1. Structured captions statistics.

ing lighting, depth of field, and composition. To generate these descriptions, we employ a state-of-the-art VLM [8] to produce the structured captions. We find that current VLMs excel at providing objective descriptions of visual content, though they are less reliable in subjective judgments, where they tend to produce flattering assessments. To address this, we complement our captions with scores from two dedicated aesthetic predictors [17, 19], which quantify the aesthetic quality of each image. Caption-length statistics are reported in Table 1, demonstrating that the vast majority are long and detailed.

While structured captions substantially enhance expressive power, we cannot expect users to author 1,000word prompts in a strict schema. To make it practical, we integrate a dedicated vision–language model (VLM) that supports three complementary operations: (1) Generate: expand a short prompt into a structured prompt. (2) Refine: modify an existing structured caption given editing instructions (see Figure 2). (3)Inspire: extract a structured cap-

tion from an input image and optionally edit it; the image serves as creative guidance. Although the VLM adds computational overhead, it makes the pipeline far more capable: users can start from either a brief prompt or an existing image, iteratively refine via simple text instructions, and benefit from the VLM’s world knowledge, reasoning ability, and multilingual support.

For the VLM, a state-of-the-art model such as Gemini 2.5 [8] is effective but costly. To provide an efficient alternative suitable for community use, we fine-tune an opensource model, Qwen-3 VL 4B [40]. We train on synthetically generated short prompts and editing instructions, using the same structured schema employed for our image model to ensure tight alignment. Training is performed on 8×H100 with a total of 2B tokens. To improve robustness, we decouple image-conditioned and text-only tasks during training and repeat each with different seeds, then final weights are produced via model merging [45].

##### 3.2. DimFusion Architecture

To fully benefit from our structured captions, the model must process long sequences of text tokens and fuse them with its internal image tokens. Recent works highlight the advantage of using LLMs as text encoders, leveraging the strong semantic and compositional understanding encoded in their intermediate layers [9, 10, 35]. To unlock this potential, text tokens can be fused with image tokens through bi-directional mixing [11], and information from both last and intermediate layers can be combined to maximize expressivity [7]. A key challenge in leveraging long captions is the prohibitive cost of processing thousands of text tokens. Naively combining intermediate layers by concatenating them along the sequence dimension increases token length, leading to substantial growth in attention computation. This challenge is especially pronounced in early pretraining stages, where models are trained on low-resolution images that produce far fewer image tokens than text tokens, causing the attention cost to scale with caption length.

We present DimFusion, an efficient mechanism for fusing intermediate LLM representations into a text-to-image model. Rather than extending the text sequence length, DimFusion concatenates intermediate layers along the embedding dimension, keeping the number of tokens fixed. This significantly reduces computational cost, particularly when handling long captions.

In more details, the embedding dimension dmodel is the size of each token’s feature vector—the last axis in tensors of shape (B,L,dmodel). A text encoding is formed by concatenating the last two layers of the LLM along the embedding dimension and projecting the result to fit D/2, where D is the transformer’s hidden dimension. At each subsequent block, hidden states from the corresponding LLM layer are projected to D/2 and concatenated with the cur-

Caption

Noisy Latents

LLM

Linear Linear

LLM Block

Linear

Double-stream block

LLM Block

Linear

Double-stream block

LLM Block

Linear

Single-stream block

LLM Block

Linear

Single-stream block

Linear

Output

- Figure 7. DimFusion architecture. In each layer, the text encoding is concatenated with the corresponding LLM hidden states along the embedding dimension. The resulting representation is then jointly processed with the noisy latents through bidirectional mixing in the Dual- and Single-stream blocks. After each block, the appended LLM hidden states are discarded, restoring the original text embedding dimension.

rent text encoding, restoring the dimension to D. After processing the block, the extra D/2 components are discarded, returning the text encoding to D/2. This design enables progressive fusion: each block integrates complementary information from intermediate LLM states while keeping token length fixed. Because half of each token is always preserved, DimFusion naturally supports dual-stream blocks followed by single-stream blocks, as in [11]. The overall architecture is illustrated in Figure 7.

We conducted an ablation study using a 1B-parameter transformer with SmolLM3-3B as the LLM text encoder. As a baseline, we trained with the final layer of T5XXL as the text encoding and no LLM fusion. Following HiDream [7], we then evaluated TokenFusion, where the text encoding is concatenated with the hidden states of the corresponding LLM block along the sequence dimension. Finally, we assessed our proposed DimFusion. All models were trained for 150k steps on 128 H200 GPUs with the same batch size (full training details appear in the appendix). Figure 8 reports the training loss, FID and average

###### Architecture FID ↓ Avg. Time per Step (sec) ↓

T5 36.46 1.28 TokenFusion 15.90 0.8 DimFusion 15.58 0.5

[Figure 48]

[Figure 49]

Figure 8. DimFusion vs. TokenFusion Top: FID on 30K COCO2014 val images [20]: both fusion strategies (TokenFusion, DimFusion) substantially outperform T5, with DimFusion achieving the best FID while maintaining lower time per step. Middle: Training loss comparison for T5 (baseline), SmolLM3-3B + TokenFusion, and SmolLM3-3B + DimFusion. Both fusion strategies outperform T5, with DimFusion closely matching TokenFusion. Bottom: Loss difference relative to TokenFusion zoomed in, showing DimFusion converges to a similar value while requiring significantly lower computational cost.

time per step for these configurations. Both TokenFusion and DimFusion substantially outperform the T5-XXL baseline, confirming the value of LLM representations. Notably, DimFusion achieves a slightly better FID than TokenFusion while reducing the average step time from 0.8 seconds to 0.5 seconds, which corresponds to a ∼1.6× reduction in wall-clock time, demonstrating that DimFusion retains the advantages of deep LLM fusion while being more computeefficient.

##### 3.3. Evaluating Long Captions

Most text-to-image models are evaluated through simple user-preference protocols, where participants are asked questions such as “which image do you prefer?” given outputs from different models, e.g., [11]. While this setting is effective for short prompts, it fails for long structured captions that often exceeding 1,000 words, since humans

Dual-stream Blocks

Single-stream Blocks

FFN Dimension

Attention Heads

Head dim (dt, dh, dw) 8 38 12288 24 128 (16, 56, 56)

Table 2. FIBO transformer architecture.

struggle to parse such detailed textual descriptions. Moreover, prior work has shown that preference-based evaluation tends to reward a characteristic “AI aesthetic”, marked by over-saturated colors and excessive focus on central subjects, a phenomenon recently termed bakeyness [3]. We argue that this evaluation paradigm hinders progress, as it incentivizes models to optimize for short prompts and average aesthetics, rather than expressive power and controllability.

To overcome this, we propose Text-as-a-Bottleneck Reconstruction (TaBR), a new evaluation protocol designed to suit long captions and generalize across domains and styles. Our key observation is that while humans find it difficult to digest long textual prompts, they can reliably compare complex images. TaBR leverages this asymmetry by introducing the image itself as the anchor for evaluation.

Inspired by cycle-consistency principles [1], TaBR begins with a real image, which is first captioned. This caption is then provided as the sole input to the generator, producing a reconstructed image. Finally, we compare the reconstructed outputs to the original. In practice, we generate two candidate reconstructions and ask annotators: “which image is more similar to the original?” Unlike standard preference scoring, user judgments in TaBR are guided by the original image, which conveys far richer detail than a short textual description. By controlling the set of real images used in evaluation, we prevent judgments from collapsing into personal taste and instead obtain a direct measure of the model’s expressive power. An example appears in Figure 9: our model is the only one that preserves the pose in the second and bottom rows, demonstrating superior controllability using text alone. Quantitative results are reported in Section 4.3.

##### 3.4. Large Scale Training

We demonstrate the effectiveness of our contributions through FIBO, a large-scale text-to-image model with 8B parameters trained entirely on long structured captions. FIBO adopts the DimFusion architecture introduced in Section 3.2, using SmolLM3-3B [2] as the fused LLM backbone. The model operates in the latent space [35] with the Wan 2.2 VAE [42] and a patch size of 1. Additional hyperparameters are summarized in Table 2.

Training is performed on 120M licensed image–caption pairs, where captions are long, structured JSONs generated by Gemini 2.5 [8], as described in Section 3.1. We employ progressive training, starting at low resolutions

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

Original Flux Qwen-Image FIBO (Ours)

Figure 9. Text-as-a-Bottleneck Reconstruction (TaBR). We caption the original image and use the resulting text as a bottleneck input to the generator, producing a reconstructed image. This evaluation allows humans to assess a model’s expressive power objectively without reading extremely long captions. As shown, our model achieves noticeably higher similarity to the original image.

and gradually increasing throughout. We initialize with REPA [46] using a coefficient of 0.1 for the first 300K lowresolution steps, followed by mixed-resolution training. After pretraining, we apply aesthetic finetuning with 3,000 hand-picked images, then DPO training [41] with dynamic beta [23] to improve text rendering. Additional implementation details, including optimization and data distribution, are provided in the Appendix.

#### 4. Experiments

In this section, we present a comprehensive evaluation of FIBO. Evaluation method are described in Section .4.1, Qualitative results are provided in Section 4.2, quantitative analyses in Section 4.3, and ablation studies examining the effect of individual design choices are discussed within the Method Section 3.

##### 4.1. Evaluation metrics.

We evaluate FIBO using four complementary protocols: PRISM-Bench-licensed [12], GenEval [13], TaBR (Section 3.3), and overall user preference—covering prompt adherence, compositional reasoning, aesthetics, and expressiveness. PRISM-Bench-licensed is an automatically de-

Imagination Text rendering Style Affection Composition Long text Overall

Model

Ali. Aes. Avg. Ali. Aes. Avg. Ali. Aes. Avg. Ali. Aes. Avg. Ali. Aes. Avg. Ali. Aes. Avg. Ali. Aes. Avg. SD3.5-Large [38] 75.4 75.2 75.3 56.1 68.3 62.2 78.3 81.0 79.7 87.7 86.4 87.0 86.6 83.1 84.9 67.8 59.2 63.5 78.8 77.6 78.2 FLUX.1-dev [18] 65.4 71.8 68.6 57.4 63.5 60.4 71.4 80.4 75.9 88.5 89.3 88.9 89.4 85.4 87.4 73.3 66.8 70.1 77.0 78.7 77.8 FLUX.1-Krea-dev [6] 71.4 73.3 72.4 52.6 62.6 57.6 79.4 84.3 81.9 86.8 89.6 88.2 90.0 86.6 88.3 77.1 65.9 71.5 79.7 79.7 79.7 HiDream-I1-Full [7] 74.1 74.4 74.2 64.8 72.2 68.5 79.3 88.8 84.1 90.8 87.8 89.3 90.3 86.7 88.5 64.0 53.7 58.8 80.0 79.1 79.5 Qwen-Image [43] 79.4 81.0 80.2 76.1 76.5 76.3 79.9 86.4 83.1 85.8 87.2 86.5 94.2 84.4 89.3 82.4 65.9 74.1 84.1 81.5 82.8 Gemini2.5-Flash-Image [8] 91.4 86.9 89.1 71.3 78.7 75.0 89.9 90.6 90.2 97.4 88.4 92.9 92.6 90.3 91.5 88.1 81.4 84.8 91.2 87.3 89.3 FIBO (Ours) 90.5 77.6 84.0 66.5 72.2 69.3 86.9 86.9 86.9 90.3 88.4 89.4 91.1 85.5 88.3 83.2 72.1 77.7 87.8 82.1 84.9 FIBO open-VLM 88.5 76.2 82.3 65.2 68.7 67.0 85.1 88.5 86.8 89.5 87.3 88.4 90.1 85.1 87.6 82.1 67.5 74.8 86.4 80.9 83.6

- Table 3. Quantitative results on PRISM-Bench-licensed: a licensed variant of PRISM [12]. We report alignment (Ali.), aesthetics (Aes.), and average (Avg.) across categories. Best results are in bold; second best are underlined. FIBO outperforms all open-source baselines.

Model Size (Parameters) Single object Two object Counting Colors Position Color attribution Overall

SD3.5-Large [38] 8B 1.00 0.91 0.79 0.81 0.23 0.54 0.71 FLUX.1-dev [18] 12B 0.99 0.80 0.75 0.77 0.19 0.49 0.66 FLUX.1-Krea-dev [6] 12B 0.99 0.91 0.70 0.88 0.31 0.69 0.75 HiDream-I1-Full [7] 17B 1.00 0.98 0.79 0.91 0.60 0.72 0.83 Qwen-Image [43] 20B 0.99 0.94 0.91 0.87 0.77 0.75 0.87

FIBO (Ours) 8B 1.00 0.88 0.79 0.90 0.80 0.71 0.85

- Table 4. Quantitative results on the GenEval benchmark. Best results are shown in bold, and second best are underlined. Unlike PRISM-Bench, GenEval focuses on relatively short captions. Accordingly, our model FIBO achieves comparable performance to other open-source models despite using fewer parameters.

rived variant of PRISM-Bench in which prompts referencing copyrighted or unlicensed concepts are filtered out, ensuring a fair comparison as FIBO is trained on 100% licensed data (see Section 6). PRISM-Bench was designed to expose alignment gaps that saturate on earlier benchmarks; for example, GenEval’s single-object category often reaches near-perfect scores, masking differences across models. For completeness, we still report GenEval, which probes object count, color attribution, and spatial relations with controlled prompts. We follow the official protocols: PRISM is scored automatically with GPT-4.1-Vision, and GenEval using detectors and CLIP-based similarity.

TaBR measures long-caption expressiveness via a caption→generation→reconstruction loop with human raters comparing reconstructions to the original image (Section 3.3). Annotators are shown the original and two reconstructions (from different models) and asked: “Which Image is more similar to the one above?” We perform this measurement on a test-set of 60 image that are not part of our training data. We also report overall user preference using blinded, side-by-side comparisons, asking: “Which

image do you prefer?” The evaluation covers a test set of 150 captions spanning diverse visual and stylistic domains. For both studies, image order and model identity are randomized per trial. We aggregate responses by computing, for each model, the percentage of questions won (win rate) and the percentage of ties (draw rate). In total, we collect over 5,000 votes to ensure a robust evaluation.

##### 4.2. Qualitative Results

Figure 9 shows qualitative examples from the TaBR evaluation set, with original reference images alongside reconstructions produced by FLUX, Qwen-Image, and FIBO. As illustrated, FIBO achieves the closest resemblance to the originals, preserving global structure and fine-grained details. In particular, it maintains pose, camera angle, and overall scene layout more faithfully than the baselines.

Figure 3 presents Contextual Contradictions drawn from ContraBench [16] and Whoops [5], which probe robustness to atypical relations, whereas prior models often default to plausible but incorrect configurations. FIBO preserves the full set of specified attributes and relations, e.g., bear per-

###### SD3.5 FLUX-dev FLUX-Krea HiDream Qwen

FIBO 90.5% 76.9% 66.4% 88.9% 59.2% Baseline 2.9% 8.0% 13.9% 4.5% 22.0% Draw 6.6% 15.1% 19.7% 6.6% 18.8%

- Table 5. Text-as-a-Bottleneck Reconstruction (TaBR) evaluation. Win rates of FIBO against each model. FIBO outperforms all open-source baselines, reflecting greater expressive power due to training on long structured captions.

forming handstand and women writing with a dart. We attribute this fidelity to training on structured captions that explicitly encode concepts and context. Additional visual results are provided in the Appendix, Figure 11, highlighting FIBO’s creative range. Figure 12 provide additional examples for controllable refinement.

##### 4.3. Quantitative Results

Table 3 reports results on PRISM-Bench-licensed. FIBO attains the highest alignment score among open-source models, substantially narrowing the prompt-alignment gap to closed systems such as Gemini 2.5 Flash Image. Despite operating at a moderate 8B parameter scale, FIBO rivals larger models in prompt understanding and text–image grounding, underscoring the impact of structured-caption supervision. We additionally report scores for FIBO when paired with a closed-source VLM [8] and with our finetuned open-source alternative. Surprisingly, the opensource variant achieves competitive results while relying on a significantly smaller and more cost-effective VLM to produce the structured prompts.

On GenEval (Table 4), FIBO delivers consistently strong results across tasks, with a pronounced gain in positional understanding, a known weakness for most models [13]. We attribute this improvement to training on structurally detailed captions that explicitly encode spatial relations.

Table 5 presents TaBR results. FIBO achieves the highest reconstruction fidelity across baselines, winning over 90% of comparisons against SD3.5-Large and 88.9% against HiDream-II-Full, while maintaining strong performance against FLUX and Qwen-Image. These findings indicate superior expressive power derived from long structured-caption training. Finally, in overall user preference (Table 6), FIBO is consistently favored over SD3.5, FLUX.1-dev, and FLUX.1-Krea-dev, and attains comparable ratings to Qwen-Image and HiDream-IIFull—demonstrating competitive perceptual quality alongside strong prompt alignment.

#### 5. Conclusions

Most prior work improves image generators under a fixed dataset regime. We take a different path: we scale the language itself by training on long structured captions that

SD3.5 FLUX-dev FLUX.1-Krea HiDream Qwen

FIBO 69.1% 58.6% 47.5% 42.3% 42.1% Baseline 19% 30.4% 35.1% 43.5% 40.2% Draw 11.9% 11% 17.4% 19.7% 17.7%

Table 6. Overall user preference. FIBO clearly surpasses SD3.5, FLUX.1-dev and Flux.1-Krea-dev, and achieves comparable performance to Qwen-Image and HiDream.

encode far richer visual and compositional detail than any prior open-source approach. This shift changes learning dynamics and behavior: long captions accelerate convergence and yield native disentanglement, so single-attribute edits (lighting, depth of field, expression) modify only the intended factor while the rest remains stable. To make this feasible, we introduced DimFusion, a novel architecture that integrates intermediate and final LLM representations without increasing token length. And because thousandword prompts outstrip human evaluators, we proposed Text-as-a-Bottleneck Reconstruction (TaBR)—an imagegrounded protocol to measure expressiveness and controllability. All components culminate in FIBO, a largescale text-to-image model that outperforms larger baselines, demonstrating that caption scale and structure are levers for controllable generation. Looking ahead, our results suggest a broader design space: we can build intermediate languages that unlock new user interfaces. Users continue to write natural language, which is translated into a structured intermediate dialect that enables new forms of creative control. This paves the way for predictable graphics primitives atop image generation models.

#### 6. Ethical Statement

Most text-to-image models are trained on data scraped online without permission or licenses from the rightful owners, a practice that violates intellectual property rights and yields models that cannot guarantee the copyright status of their outputs. In contrast, our model is trained solely on licensed data, substantially reducing the risk of infringement. We believe the AI ethics community should prioritize licensed, consent-based training, as responsible data use is essential for a sustainable AI ecosystem. Complementing this stance, our work investigates how structured synthetic captions and LLM fusion can improve fidelity and controllability: by strengthening prompt adherence and semantic grounding, our approach can help reduce unintended biases and misalignment. At the same time, greater controllability introduces dual-use risks, as it may also be exploited to generate misleading, or inappropriate content more reliably. We therefore advocate openly studying both the benefits and risks of these techniques to develop effective safeguards.

#### References

- [1] Hyojin Bahng, Caroline Chan, Fredo Durand, and Phillip Isola. Cycle consistency as reward: Learning imagetext alignment without human preferences. arXiv preprint arXiv:2506.02095, 2025. 3, 7
- [2] Elie Bakouch, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, Lewis Tunstall, Carlos Miguel Pati˜no, Edward Beeching, Aymeric Roucher, Aksel Joonas Reedi, Quentin Gallou´edec, Kashif Rasul, Nathan Habib, Cl´ementine Fourrier, Hynek Kydlicek, Guilherme Penedo, Hugo Larcher, Mathieu Morlon, Vaibhav Srivastav, Joshua Lochner, XuanSon Nguyen, Colin Raffel, Leandro von Werra, and Thomas Wolf. SmolLM3: smol, multilingual, long-context reasoner. https://huggingface.co/blog/smollm3, 2025. 7
- [3] Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv e-prints, pages arXiv–2506,

2025. 7

- [4] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023. 3
- [5] Nitzan Bitton-Guetta, Yonatan Bitton, Jack Hessel, Ludwig Schmidt, Yuval Elovici, Gabriel Stanovsky, and Roy Schwartz. Breaking common sense: Whoops! a vision-andlanguage benchmark of synthetic and compositional images. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), page 2616–2627. IEEE, 2023. 3, 8
- [6] BlackForest. Flux.1 krea, 2025. 8
- [7] Qi Cai, Jingwen Chen, Yang Chen, Yehao Li, Fuchen Long, Yingwei Pan, Zhaofan Qiu, Yiheng Zhang, Fengbin Gao, Peihan Xu, et al. Hidream-i1: A high-efficient image generative foundation model with sparse diffusion transformer. arXiv preprint arXiv:2505.22705, 2025. 2, 3, 5, 6, 8, 12
- [8] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 5, 7, 8, 9
- [9] Guy Dar, Mor Geva, Ankit Gupta, and Jonathan Berant. Analyzing transformers in embedding space. arXiv preprint arXiv:2209.02535, 2022. 3, 5
- [10] Nadir Durrani, Hassan Sajjad, Fahim Dalvi, and Yonatan Belinkov. Analyzing individual neurons in pre-trained language models. arXiv preprint arXiv:2010.02695, 2020. 3, 5
- [11] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning,

2024. 2, 3, 4, 5, 6, 12

- [12] Rongyao Fang, Aldrich Yu, Chengqi Duan, Linjiang Huang, Shuai Bai, Yuxuan Cai, Kun Wang, Si Liu, Xihui Liu, and Hongsheng Li. Flux-reason-6m & prism-bench: A millionscale text-to-image reasoning dataset and comprehensive benchmark. arXiv preprint arXiv:2509.09680, 2025. 7, 8
- [13] Dhruba Ghosh, Hannaneh Hajishirzi, and Ludwig Schmidt. Geneval: An object-focused framework for evaluating textto-image alignment. Advances in Neural Information Processing Systems, 36:52132–52152, 2023. 7, 9
- [14] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022. 4
- [15] Jia-Hong Huang, Hongyi Zhu, Yixian Shen, Stevan Rudinac, and Evangelos Kanoulas. Image2text2image: A novel framework for label-free evaluation of image-to-text generation with text-to-image diffusion models. In International Conference on Multimedia Modeling, pages 413–427. Springer, 2025. 3
- [16] Saar Huberman, Or Patashnik, Omer Dahary, Ron Mokady, and Daniel Cohen-Or. Image generation from contextuallycontradictory prompts. arXiv preprint arXiv:2506.01929,

2025. 2, 3, 8

- [17] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. Advances in neural information processing systems, 36:36652– 36663, 2023. 2, 5
- [18] Black Forest Labs. Flux. https://github.com/ black-forest-labs/flux, 2024. 2, 3, 4, 8
- [19] LAION-AI. Laion-aesthetics v1 (dataset documentation). https://projects.laion.ai/laiondatasets/laion- aesthetic.html, 2022. Accessed: 2023-11-10. 5
- [20] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014. 4, 6, 12
- [21] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling, 2023. 12
- [22] Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Chase Lambert, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving text-to-image alignment with deep-fusion large language models. arXiv preprint arXiv:2409.10695, 2024. 3, 4
- [23] Jie Liu, Gongye Liu, Jiajun Liang, Ziyang Yuan, Xiaokun Liu, Mingwu Zheng, Xiele Wu, Qiulin Wang, Wenyu Qin, Menghan Xia, et al. Improving video generation with human feedback. arXiv preprint arXiv:2501.13918, 2025. 2, 7, 12
- [24] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 12
- [25] Chutian Meng, Fan Ma, Jiaxu Miao, Chi Zhang, Yi Yang, and Yueting Zhuang. Image regeneration: Evaluating text-toimage model via generating identical image with multimodal

- large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 6090–6098, 2025. 3
- [26] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6038–6047, 2023. 4
- [27] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021. 2
- [28] Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 12
- [29] Hadas Orgad, Bahjat Kawar, and Yonatan Belinkov. Editing implicit assumptions in text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7053–7061, 2023. 2
- [30] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. 2, 3, 12
- [31] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020, 2021. 3
- [32] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J Liu, et al. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21

(140):1–67, 2020. 3, 12

- [33] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1

(2):3, 2022. 2

- [34] Royi Rassin, Aviv Slobodkin, Shauli Ravfogel, Yanai Elazar, and Yoav Goldberg. Grade: Quantifying sample diversity in text-to-image models. arXiv preprint arXiv:2410.22592,

2024. 2

- [35] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 3, 5, 7
- [36] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 2, 3
- [37] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo

- Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022. 3
- [38] Stability-AI. Stable diffusion 3.5, 2024. 8
- [39] Bingda Tang, Boyang Zheng, Sayak Paul, and Saining Xie. Exploring the deep fusion of large language models and diffusion transformers for text-to-image synthesis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28586–28595, 2025. 3
- [40] Qwen Team. Qwen3-vl: Sharper vision, deeper thought, broader action. https://qwen.ai/blog?id= 99f0335c4ad9ff6153e517418d48535ab6d8afef& from=research.latest- advancements- list. Accessed: 2025-10-29. 5
- [41] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8228–8238, 2024. 2, 7, 12
- [42] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 7
- [43] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report,

2025. 2, 3, 8

- [44] Yankun Wu, Yuta Nakashima, and Noa Garcia. Stable diffusion exposed: Gender bias from prompt to image. In Proceedings of the AAAI/ACM conference on AI, ethics, and society, pages 1648–1659, 2024. 2
- [45] Enneng Yang, Li Shen, Guibing Guo, Xingwei Wang, Xiaochun Cao, Jie Zhang, and Dacheng Tao. Model merging in llms, mllms, and beyond: Methods, theories, applications and opportunities. arXiv preprint arXiv:2408.07666, 2024. 5
- [46] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024. 7, 12
- [47] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycleconsistent adversarial networks. In Proceedings of the IEEE international conference on computer vision, pages 2223– 2232, 2017. 3

# Appendices

#### A. Structured Captions

In this section we describe our structured captioning strategy. Each caption is a long, detailed JSON that decomposes the image into interpretable fields, covering the objects, background, lighting, aesthetics, photographic attributes, style, and text. This structure guarantees full semantic coverage and improves controllability during training and inference. The structure of our JSON is described in Table 7.

Our structured captions are generated using Gemini 2.5 with a dedicated system prompt (see Table 8) and using Gemini’s structured outputs configured with our JSON schema (see Table 9) that enforce completeness, consistency, and adherence to the predefined schema.

#### B. Additional Training Details

The data distribution is illustrated in Figure 10.

We train the model using the AdamW optimizer [24] with weight decay of 1 × 10−4, β1 = 0.9, β2 = 0.999, and ϵ = 1 × 10−15. The learning rate is set to 1 × 10−4 with a constant schedule and a warmup of 10K steps, followed by an additional 2.5K warmup steps when increasing resolution. Training follows the flow-matching formulation [21], with a logit-normal noise schedule combined with resolution-dependent timestep shifting [11].

We train the model in stages of increasing resolution. The first stage is conducted at 2562 resolution for 652k steps, where the initial 300k steps employ REPA [46] using DINOv2-L encodings [28] and REPA coefficient of 0.1. Training then continues for an additional 100k steps at 5122 resolution, followed by 50k steps at 10242. At each resolution stage, the effective batch size starts at approximately 1K and is gradually increased to 2K as training progresses.

Post-training, we perform aesthetic finetuning with 3,000 hand-picked images, followed by DPO training [41] with dynamic beta [23] to improve text rendering.

Dual-stream Blocks

Single-stream Blocks

FFN Dimension

Attention Heads

Head dim(dt, dh, dw) 8 20 6144 24 64 (4, 30, 30)

Table 10. Model configuration for the 1B-parameter models used in the ablation studies. Each model is trained under identical conditions, differing only by caption type (long vs. short) or architectural variant (Baseline, DimFusion, TokenFusion).

In Section 3.1, we examine the effect of caption length using the DimFusion architecture with SmolLM3-3B as the LLM, trained once with long structured captions and once with short captions. In Section 3.2, we compare three architectures trained on long structured captions. The baseline follows [11] and uses T5 [32] as the text encoder. We further evaluate two large-language-model fusion variants that share the same hyperparameters as the baseline: (1) DimFusion with SmolLM3-3B as the LLM; and (2) TokenFusion, following HiDream [7], concatenates SmolLM3-3B hidden states with the text embeddings along the sequence dimension, effectively doubling the number of text tokens in each attention layer.

All models were trained on 70M licensed image-caption pairs, with the same optimizer, learning rate and noise scheduling described in Appendix B, and effective batch size of 1024. We also employed REPA [46] with DINOv2-L encodings [28] and a REPA coefficient of 0.5. For evaluation, FID was computed on 30K samples from the COCO2014 [20] validation split, using 50 inference steps and no classifier-free guidance.

#### D. Additional Samples from FIBO

Additional samples from FIBO in various aspect ratios appear in Figure 11, and additional examples for refinement and disentanglement appear in Figure 12.

#### C. Additional Ablation Studies Details

We provide implementation details for the ablation studies discussed in Section 3.1 (long structured captions vs. short captions) and Section 3.2 (convergence behavior across architectures). To reduce compute cost, all ablations were conducted under identical training and sampling conditions using 1B-parameter models trained with the SDXL VAE [30]. Table 10 lists the architectural hyperparameters used in all experiments.

###### Field Type Description

short description String A short summary of the image content. objects Array of Objects List of prominent objects. For each object, we record the following fields: a detailed description of the object, the object’s location, relative size, shape and color, texture (opt.), appearance details (opt.), relationship with other objects, and the object’s orientation (opt.). When a human is described, the following fields are also included: the human’s pose (opt.), expression (opt.), clothing (opt.), action (opt.), gender (opt.) and skin tone and texture (opt.). If the image is a cluster of objects then also includes the number of objects (opt.).

background setting String Description of the environment/background. lighting Object Illumination details, including conditions (e.g., daylight, studio),

direction (key-light orientation), and shadows (opt.).

aesthetics Object Global visual properties, including composition (e.g., rule of third, symmetrical), color scheme (dominant palette) and mood atmosphere (emotional tone).

photographic characteristics Object (optional) Camera/focus parameters: depth of field, focus, camera angle, and

lens focal length. style medium String Artistic/rendering medium (e.g., photograph, oil painting, 3D render). text render Array of Objects A list of prominent text renders in the image. For each text render it includes the

text, location, size, color, font and appearance details (opt.). context String Additional context that helps understand the image better. artistic style String High-level artistic/stylistic description (e.g., cinematic, minimalism).

Table 7. Structure of the JSON-based captions. Each field captures a distinct aspect of the image, ensuring semantic completeness and consistency.

People Objects Typography Nature Animals Food

Transportation Product Urban Indoors Logos

- 1%
- 2%

3% 4%

4%

5%

39%

6%

6%

10%

20%

Figure 10. Data distribution across categories. People and objects dominate the dataset (39% and 20%, respectively), followed by typography (10%), nature (6%), animals (6%), food (5%), transportation (4%), product (4%), urban scenes (3%), indoors (2%), and logos (1%).

You are a meticulous and perceptive Visual Art Director working for a leading Generative AI company. Your expertise lies in analyzing images and extracting detailed, structured information. Your primary task is to analyze provided images and generate a comprehensive JSON object describing them. Adhere strictly to the following structure and guidelines: The output MUST be ONLY a valid JSON object. Do not include any text before or after the JSON object (e.g., no "Here is the JSON:", no explanations, no apologies). IMPORTANT: When describing human body parts, positions, or actions, always describe them from the PERSON’S OWN PERSPECTIVE, not from the observer’s viewpoint. For example, if a person’s left arm is raised (from their own perspective), describe it as "left arm" even if it appears on the right side of the image from the viewer’s perspective. The JSON object must contain the following keys precisely:

- 1. ‘short description‘: (String) A concise summary of the image content, 200 words maximum.

- 2. ‘objects‘: (Array of Objects) List a maximum of 5 prominent objects if there are more than 5, list them in the background. For each object, include:

- * ‘description‘: (String) A detailed description of the object, 100 words maximum.
- * ‘location‘: (String) E.g., "center", "top-left", "bottom-right foreground".
- * ‘relative size‘: (String) E.g., "small", "medium", "large within frame".

- * ‘shape and color‘: (String) Describe the basic shape and dominant color.

- * ‘texture‘: (String) E.g., "smooth", "rough", "metallic", "furry".
- * ‘appearance details‘: (String) Any other notable visual details.

- * ‘relationship‘: (String) Describe the relationship between the object and the other objects in the image.
- * ‘orientation‘: (String) Describe the orientation or positioning of the object, e.g., "upright", "tilted 45

degrees", "horizontal", "vertical", "facing left", "facing right", "upside down", "lying on its side". If the object is a human or human-like entity, include:

- * ‘pose‘: (String) Describe the body position.
- * ‘expression‘: (String) Describe facial expression and emotion. E.g., "winking", "joyful", "serious",

"surprised", "calm".

- * ‘clothing‘: (String) Describe attire.
- * ‘action‘: (String) Describe the action of the human.
- * ‘gender‘: (String) Describe the gender of the human.
- * ‘skin tone and texture‘: (String) Describe the skin tone and texture.

If the object is a cluster of objects, include the following:

* ‘number of objects‘: (Integer) The number of objects in the cluster.

- 3. ‘background setting‘: (String) Describe the overall environment, setting, or background, including any notable background elements that are not part of the objects section.

- 4. ‘lighting‘: (Object)

- * ‘conditions‘: E.g., "bright daylight", "dim indoor", "studio lighting", "golden hour".
- * ‘direction‘: E.g., "front-lit", "backlit", "side-lit from left".
- * ‘shadows‘: Describe the presence of shadows.

- 5. ‘aesthetics‘: (Object)

- * ‘composition‘: E.g., "rule of thirds", "symmetrical", "centered", "leading lines".
- * ‘color scheme‘: E.g., "monochromatic blue", "warm complementary colors", "high contrast".

- * ‘mood atmosphere‘: E.g., "serene", "energetic", "mysterious", "joyful".

- 6. ‘photographic characteristics‘: (Object)

- * ‘depth of field‘: (String) E.g., "shallow", "deep", "bokeh background".

- * ‘focus‘: (String) E.g., "sharp focus on subject", "soft focus", "motion blur".
- * ‘camera angle‘: (String) E.g., "eye-level", "low angle", "high angle", "dutch angle".

- * ‘lens focal length‘: (String) E.g., "wide-angle", "telephoto", "macro", "fisheye".

- 7. ‘style medium‘: (String) Identify the artistic style or medium (e.g., "photograph", "oil painting", "watercolor", "3D render", "digital illustration", "pencil sketch") If the style is not "photograph", but artistic, please describe the specific artistic characteristics under ’artistic style’, 50 words maximum.

- 8. ‘artistic style‘: (String) Describe specific artistic characteristics, 3 words maximum.

- 9. ‘context‘: (String) Provide any additional context that helps understand the image better. This should include a general description of the type of image (e.g., Fashion Photography, Product Shot, Magazine Cover, Nature Photography, Art Piece, etc.), as well as any other relevant contextual information that situates the image within a broader category or intended use. For example: "This is a high-fashion editorial photograph intended for a magazine spread"
- 10. ‘text render‘: (Array of Objects) List of a maximum of 5 most prominent text renders in the image. For each text render, include:

- * ‘text‘: (String) The text content.
- * ‘location‘: (String) E.g., "center", "top-left", "bottom-right foreground".
- * ‘size‘: (String) E.g., "small", "medium", "large within frame".
- * ‘color‘: (String) E.g., "red", "blue", "green".
- * ‘font‘: (String) E.g., "realistic", "cartoonish", "minimalist".
- * ‘appearance details‘: (String) Any other notable visual details.

Ensure the information within the JSON is accurate, detailed where specified, and avoids redundancy between fields.

Table 8. System prompt used to generate structured captions.

{"type": "OBJECT", "properties": {

"short description": {"type": "STRING" }, "objects": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {

"description": {"type": "STRING" }, "location": {"type": "STRING" }, "relationship": {"type": "STRING" }, "relative size": {"type": "STRING" }, "shape and color": {"type": "STRING" }, "texture": {"type": "STRING", "nullable": true }, "appearance details": {"type": "STRING", "nullable": true }, "number of objects": {"type": "INTEGER", "nullable": true }, "pose": {"type": "STRING", "nullable": true }, "expression": {"type": "STRING", "nullable": true }, "clothing": {"type": "STRING", "nullable": true }, "action": {"type": "STRING", "nullable": true }, "gender": {"type": "STRING", "nullable": true }, "skin tone and texture": {"type": "STRING", "nullable": true }, "orientation": {"type": "STRING", "nullable": true }}, "required": [ "description", "location", "relationship", "relative size", "shape and color",

"texture", "appearance details", "number of objects", "pose", "expression", "clothing", "action", "gender", "skin tone and texture", "orientation" ] }},

"background setting": {"type": "STRING" }, "lighting": {"type": "OBJECT", "properties": {

"conditions": {"type": "STRING" }, "direction": {"type": "STRING" }, "shadows": {"type": "STRING", "nullable": true }}, "required": [ "conditions", "direction", "shadows" ] },

"aesthetics": {"type": "OBJECT", "properties": { "composition": {"type": "STRING" }, "color scheme": {"type": "STRING" }, "mood atmosphere": {"type": "STRING" }},

"required": [ "composition", "color scheme", "mood atmosphere" ] },

"photographic characteristics": {"type": "OBJECT", "properties": { "depth of field": {"type": "STRING" }, "focus": {"type": "STRING" }, "camera angle": {"type": "STRING" }, "lens focal length": {"type": "STRING" }}, "required": [ "depth of field", "focus", "camera angle", "lens focal length" ], "nullable": true },

"style medium": {"type": "STRING" }, "text render": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {

"text": {"type": "STRING" }, "location": {"type": "STRING" }, "size": {"type": "STRING" }, "color": {"type": "STRING" }, "font": {"type": "STRING" }, "appearance details": {"type": "STRING", "nullable": true }}, "required": [ "text", "location", "size", "color", "font", "appearance details" ] }},

"context": {"type": "STRING" }, "artistic style": {"type": "STRING" }},

"required": [ "short description", "objects", "background setting", "lighting", "aesthetics", "photographic characteristics", "style medium", "text render", "context", "artistic style" ] }

Table 9. Gemini’s Structured Outputs.

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

###### Figure 11. Additional samples from FIBO in various aspect ratios.

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

make snowing move to desert make night

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

make owl brown owl to lemur add jungle vegetation

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

blueberries to strawberries blue to pink add mint leaves

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

blue cup with flower ornaments add scattered daisies hand steering coffee with teaspoon

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

black dog light-green coat, hoodie on Add snow

Figure 12. Additional refinement examples.

