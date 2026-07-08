# arXiv:2508.08244v2[cs.CV]12Aug2025

## Cut2Next: Generating Next Shot via In-Context Tuning

Jingwen He1,2 Hongbo Liu2 Jiajun Li3 Ziqi Huang3 Yu Qiao2 Wanli Ouyang1† Ziwei Liu3† 1The Chinese University of Hong Kong, 2Shanghai Artificial Intelligence Laboratory, 3S-Lab, Nanyang Technological University †Corresponding authors. Project page: https://vchitect.github.io/Cut2Next-project/

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

###### Shot/Reverse Shot

###### Cutaway

Characters: <Woman in embroidered dress>, <Man in blue military coat>; Context: in a richly decorated room.

Characters: <Young girl with hood>, and <Polar Bear>. Context: in a snowy, mountainous landscape.

Cut2Next

Cut2Next

[Figure 5]

[Figure 6]

[Figure 7]

###### Cut-Out

###### Cut-Out

[Figure 8]

Characters: <Woman on the left with dark skin>, <Woman in the center with dark skin>, <Woman on the right with dark skin>. Context: on a stage with a dark background, bathed in stage lighting.

Characters: <Veiled Woman> (Black headpiece with veil and red lipstick)

Context: in an interior setting with red and warm tones.

Cut2Next

Cut2Next

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

###### Shot/Reverse Shot

###### Cutaway

Characters: <Man with Glasses> , <Woman with Blonde Hair>; Context: at a table in a high-end restaurant.

Characters: <Deena Jones>, <Vendor with brown hair>, <Vendor with short hair>. Context: at a vibrant outdoor market.

Cut2Next

Cut2Next

Figure 1: Cut2Next demonstrating versatile Next Shot Generation. The model produces cinematically coherent subsequent shots (bottom) adhering to diverse editing patterns (e.g., Shot/Reverse Shot, Cut-Out, Cutaway) specified alongside the input shots (upper).

Preprint.

### Abstract

Effective multi-shot generation demands purposeful, film-like transitions and strict cinematic continuity. Current methods, however, often prioritize basic visual consistency, neglecting crucial editing patterns (e.g., shot/reverse shot, cutaways) that drive narrative flow for compelling storytelling. This yields outputs that may be visually coherent but lack narrative sophistication and true cinematic integrity. To bridge this, we introduce Next Shot Generation (NSG): synthesizing a subsequent, high-quality shot that critically conforms to professional editing patterns while upholding rigorous cinematic continuity. Our framework, Cut2Next, leverages a Diffusion Transformer (DiT). It employs in-context tuning guided by a novel Hierarchical Multi-Prompting strategy. This strategy uses Relational Prompts to define overall context and inter-shot editing styles. Individual Prompts then specify per-shot content and cinematographic attributes. Together, these guide Cut2Next to generate cinematically appropriate next shots. Architectural innovations, ContextAware Condition Injection (CACI) and Hierarchical Attention Mask (HAM), further integrate these diverse signals without introducing new parameters. We construct RawCuts (large-scale) and CuratedCuts (refined) datasets, both with hierarchical prompts, and introduce CutBench for evaluation. Experiments show Cut2Next excels in visual consistency and text fidelity. Crucially, user studies reveal a strong preference for Cut2Next, particularly for its adherence to intended editing patterns and overall cinematic continuity, validating its ability to generate high-quality, narratively expressive, and cinematically coherent subsequent shots.

### 1 Introduction

Video generation has advanced remarkably. Models like Sora [2] and Kling [1] synthesize photorealistic single-shot videos, built upon large-scale data and scalable architectures like Diffusion Transformers (DiT) [35]. Following these successes, the academic community is increasingly focusing on narrative video generation [15, 37, 23, 48]. This aims to create videos composed of multiple interconnected shots, akin to films.

Several approaches address multi-shot video generation. One prominent strategy is storyboard generation [56, 55, 30, 43]: text-to-image models first create keyframes, which image-to-video models then animate to construct the full video. Another approach is to directly tackle multi-shot video generation by training models on extensive video datasets [15, 37, 23]. Both these research lines primarily aim for diverse content in long-form videos with rich plotlines. In contrast, a distinct research line prioritizes individual shot quality and basic consistency within sequences. For instance, IC-LoRA [18] leverages Flux [7]’s in-context generation ability to generate high-resolution storyboards with consistent environments and characters. CineVerse [36] enhanced IC-LoRA with detailed annotations for user-controlled shot scale. SynCamMaster [4] aims to synthesize 3Dconsistent multi-angle, multi-scale shots, which are then edited together to enrich visual expression. While significantly advancing shot quality and basic consistency, these approaches often overlook the explicit modeling and enforcement of complex editing patterns crucial to professional narrative filmmaking.

Our work addresses this crucial gap. We align with the pursuit of high-fidelity individual shots but place an emphasis on the fundamental cinematic concept of the "cut." In professionally edited movies, cuts are not arbitrary; they serve a distinct narrative purpose [34]. Building on this, as illustrated in Figure 2, we focus on canonical cut-driven sequences: Shot/Reverse Shot for dialogue, Cut-In/CutOut for emphasis, Cutaways for context, and Multi-Angle for perspective shifts. Such sequences are vital for enhancing narrative capability and inter-shot consistency. Traditionally, filming these sequences is difficult and expensive. It requires shooting different angles one by one, repeated actor performances, and careful resetting of the scene and lights.

Based on these considerations, we define a new task: Next Shot Generation (NSG). Given an existing shot, the goal is to generate a subsequent, highly coherent shot. This new shot must maintain character and environmental consistency while also adhering to established cinematic continuity principles and specific editing patterns. This task of Next Shot Generation presents significant challenges:

###### Cut-In/Cut-Out Cutaway Shot/Reverse Shot Multi-Angle

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Figure 2: Canonical cut-driven shot sequences (from CuratedCuts), their narrative functions, and the generation difficulties. Cut-In/Cut-Out: Emphasizes details or shifts focus; challenges models with drastic scale changes while maintaining subject consistency. Cutaway: Provides external or subjective context; demands generating novel yet semantically related content. Shot/Reverse Shot: Facilitates dialogue and reveals reactions; requires consistent character appearance and spatial logic across alternating viewpoints. Multi-Angle: Offers varied viewpoints; requires consistent rendering across significant visual transformations.

First, it demands comprehensive cinematic continuity from complex inputs. Unlike typical subjectdriven [42, 22, 9, 52, 13] tasks focused on isolated subject preservation, NSG requires consistency across numerous facets: character identity and appearance, spatial relationships, environmental details, lighting, color palettes, overall tone, and implied temporal progression. Achieving such multifaceted continuity from rich visual inputs is a foundational difficulty. Second, NSG aims to replicate diverse editing patterns, introducing significant visual diversity. Examples include Shot/Reverse Shot (shifting to a character’s converse view in dialogue), Cut-In (focusing on a detail for emphasis)/Cut-Out (returning to a wider context), Cutaway (providing external or subjective context), and Multi-Angle (cutting between different views of the same subject). The model must accurately synthesize these diverse visual outputs reflecting the specified editing pattern, while simultaneously upholding the comprehensive cinematic continuity. Next Shot Generation uniquely balances this required shot diversity with strict, multi-faceted continuity, making it more advanced than simpler consistency tasks.

To achieve the demanding balance of Next Shot Generation (NSG), we first construct a comprehensive data foundation. This includes RawCuts, a large-scale dataset of adjacent shot pairs for foundational learning of visual transitions. Complementing this, CuratedCuts, a smaller, meticulously humancurated set, exemplifies the strong cinematic continuity and professional editing techniques crucial for NSG, serving fine-grained refinement. To guide our model in navigating the dual demands of continuity and diversity, we employ a Hierarchical Prompt Annotation scheme. This provides Relational Prompt to dictate inter-shot relationships and editing patterns, and Individual Prompts to detail per-shot content and cinematographic attributes. Methodologically, we propose Cut2Next, built upon FLUX.1-dev [7] using in-context tuning. To specifically address NSG’s challenges, Cut2Next incorporates: (1) Our Context-Aware Condition Injection (CACI) mechanism for nuanced integration of diverse conditional inputs. (2) A Hierarchical Attention Mask (HAM) to orchestrate the intricate flow between visual and textual tokens. This approach allows Cut2Next to effectively utilize our rich annotations and generate high-quality, cinematically coherent next shots that masterfully balance diversity with continuity, all without introducing additional parameters to the base model.

To rigorously evaluate Cut2Next’s capabilities in balancing these NSG demands, we introduce CutBench. This new benchmark features diverse movie shot images, each paired with our hierarchical prompts designed to elicit specific cinematic continuations that test both continuity and varied editing. Our quantitative results demonstrate Cut2Next significantly outperforms baselines in generating visually coherent and textually aligned subsequent shots. Crucially, to evaluate the critical balance—cinematic continuity and intended editing patterns—we conducted extensive user studies. These studies not only corroborate our quantitative findings but, more importantly, indicate a

strong human preference for Cut2Next’s outputs in terms of overall cinematic continuity and faithful execution of specified editing.

### 2 Related Work

- 2.1 Multi-Shot Generation

Multi-shot generation aims to create sequences of related visual outputs, either as video or sets of images, emphasizing inter-shot consistency for narrative coherence or visual continuity. In multi-shot video generation, efforts focus on synthesizing continuous events [15, 37, 23, 55, 31, 3, 5, 10]. Some methods directly use keyframes [56, 55, 30, 43] for image-to-video (I2V) synthesis [49, 53, 6, 8, 26, 50]. VideoStudio [31] leverages entity embeddings for appearance. VGoT [55] uses identity-preserving embeddings for character consistency. MovieDreamer [54] renders visual tokens into I2V keyframes. SynCamMaster [4] generates 3D-consistent Multi-Angle shots for subsequent editing, while Long Context Tuning [15] expands pre-trained models’ context to learn scene-level consistency directly. Alternatively, multi-shot image generation [45] often produces "story-frames" for controlled narrative development. IC-LoRA [18] leverages in-context generation with LoRA tuning to achieve consistent environments and characters across frames. CineVerse [36] enhanced IC-LoRA by incorporating detailed annotations for user-controlled shot scale. While methods like StoryDiffusion [56] and One Prompt One Story [30] offer flexibility, they result in character-centric, repetitive perspectives that may weaken narrative impact. Despite advancements in character and environment consistency across these approaches, they often do not explicitly model the complex editing patterns and cinematic language crucial to professional narrative filmmaking.

- 2.2 Subject-driven Generation

Subject-driven generation aims to create new content featuring specific user-provided subjects. In image generation, methods like Dreambooth [40], Textual Inversion [12], and LoRA [17] enable subject customization via parameter tuning. Others, like IP-Adapter [51], utilized external image encoders to inject subject appearance without per-subject fine-tuning. For DiT architectures, methods [32, 18, 42, 28, 47] such as IC-LoRA and Ominicontrol, demonstrated the transformer’s inherent capacity for image referencing. Extending this to video customization, early works [16, 22, 46] largely focused on single-concept scenarios. While progress has been made, such as with ConceptMaster [20] in multi-concept video customization, challenges in robustly handling multiple concepts persist. Despite these advances in generating specified subjects, current methods often handle simple inputs and compositions, thus addressing limited consistency dimensions. Our work targets comprehensive cinematic continuity, encompassing consistent subjects, environments, color, lighting, and style across shot sequences.

- 3 Methodology

- 3.1 Dataset Construction and Annotation

To generate continuous and cinematically coherent shots, our data strategy employs a two-stage pipeline (Figure 3). First, RawCuts, a large-scale dataset, provides broad exposure to diverse visual transitions for foundational model learning. Second, CuratedCuts, a meticulously selected subset, enables fine-grained refinement of cinematic continuity. Rich textual annotations, via our Hierarchical Prompt Annotation scheme, are then applied to both datasets to capture inter-shot context and per-shot details. This comprehensive annotation strategy enables the model to learn robust visual-semantic correlations, ultimately guiding robust generation even from concise user inputs.

#### 3.1.1 Dataset Creation Pipeline

Our data creation process begins with an automated stage to build the large-scale RawCuts dataset, designed to provide Cut2Next with broad exposure to diverse visual content and transitional patterns. Initially, MovieNet [19] videos are preprocessed by resizing to 720p and removing black borders (ffmpeg [11]). Shot segmentation is performed using TransNetV2 [41], followed by uniform frame sampling per shot. To ensure quality, we compute aesthetic scores [27] and motion scores

Constructing RawCuts

Constructing CuratedCuts

HighQuality Shot Image Pairs

- - Aesthetics Filter
- - Quality Filter
- - OCR Filter
- - NSFW Filter
- - Pair Shot Images

- Manually Filter

- - Resize to 720p
- - Crop Black Borders
- - Shot Segmentation
- - Motion Filter

- - Sample Frames at Fixed FPS
- - Key Frame Selection

Shot Image Pairs

Shot Images

Single Shot Videos

Movies

Figure 3: The data construction pipeline for RawCuts and CuratedCuts.

(VMAF [11]). We retain only shots with low motion scores to ensure satisfactory per-frame image quality and composition. The frame with the highest aesthetic score from each shot is selected as a keyframe candidate. These candidates undergo further filtering based on aesthetics, image quality (MUSIQ [24]), absence of overlaid text (OCR [21]), and NSFW content [44]. Finally, RawCuts is formed by sequentially pairing adjacent keyframes, yielding over 200k shot pairs.

Subsequently, to facilitate the learning of cinematically sophisticated transitions, we manually curate a smaller, high-quality subset from RawCuts, termed CuratedCuts. This meticulous selection prioritizes thousands of shot pairs exhibiting strong adherence to cinematic continuity principles and diverse professional editing techniques (e.g., cut-ins, shot/reverse shots). Emphasis is placed on meaningful character interactions, expressive emotional cues, and varied cinematic styles, making CuratedCuts ideal for refining Cut2Next’s ability to generate cinematically coherent shots.

[Figure 25]

[Figure 26]

[Figure 27]

###### Description for Each Shot?

###### Relation Between Two Shots?

-Describe each shot separately: provide detailed visual and cinematographic descriptions using structured attributes.

-Determine scene and character consistency across shots; assign names; describe the transition and provide scene captions.

[Figure 28]

###### Response to Relation Between Two Shots:

Scene Consistency: Same scene — outdoor daytime setting, soft natural light, muted palette, shallow depth of field suggests unified visual style. Character Consistency: Consistent characters — <Belle> (Emma Watson) and <Maurice> (Man with wide-brimmed hat), appearance remains consistent; focus shifts between shots. Scene Caption: <Belle> and <Maurice> converse in an outdoor village setting. Transition Caption: Shot/reverse shot editing shows each character’s perspective in a conversation.

###### Response to Description for Each Shot :

###### [Shot 1]:

- - Description: <Belle> (Emma Watson) in medium close-up, slightly right of center, white blouse and blue vest, looking left. Foreground shows out-of-focus shoulder.
- - Cinematography: Framing: Over the shoulder, Dirty Single; Focus: Shallow — <Belle> in focus; Lighting: Soft natural daylight; Color: Muted, realistic tones; Camera Angle: High angle. [Shot 2]:
- - Description: <Maurice> (Man with wide-brimmed hat) in medium close-up, centered, grey hair and beard, blue jacket, looking left. Sky background blurred.
- - Cinematography: Framing: Single, Clean Reverse Shot; Focus: Shallow — <Maurice> in focus; Lighting: Natural daylight; Color: Naturalistic skin tones, muted colors; Camera Angle: Low Angle.

- Figure 4: Example of annotating one shot image pair by our Hierarchical Prompt Annotation.

[Figure 29]

input shot 𝑆 target shot 𝑆

relational prompt

individual prompts

visual tokens textual tokens (individual) textual tokens (relational)

[Figure 30]

|𝑃| |
|---|---|
| | |

|𝑃| |
|---|---|
| | |

|𝑃| |
|---|---|
| | |

Hierarchical Attention Mask (HAM)

Context-Aware Condition Injection (CACI)

|attn<br><br>t=0 t<br><br>…<br><br>…<br><br>…<br><br>…<br><br>…|
|---|

|| | | | | |
|---|---|---|---|---|
| | | | | |
|01|8007|9.jpg| | |
| | | | | |
| | | | | |
| | | | | |
|
|---|

Text encoder VAE encoder

+ noise

Cut2Next

…

…

…

…

…

- Figure 5: Architecture of Cut2Next. Individual prompts (Pcondind ,Ptgtind) and a relational prompt (Prel) are converted to textual embeddings by a shared text encoder. The conditional shot Scond is encoded by a VAE into clean latents, while the target shot Stgt is encoded and noised for training. These textual and visual tokens form the input to the Cut2Next (DiT-based) model. Our Context-Aware Condition Injection (CACI) module (center right) applies distinct conditioning to AdaLN layers based on token type. The Hierarchical Attention Mask (HAM) (far right) further refines information flow by defining specific attention patterns between different token segments.

#### 3.1.2 Hierarchical Prompt Annotation

To implement our Hierarchical Multi-Prompting strategy, we adopt an automated annotation process for each image pair (Scond,Stgt) in our datasets using the Gemini-2.0-flash [14]. This process generates two complementary types of prompts designed to provide comprehensive textual guidance: relational prompts (Prel) and individual prompts (Pind). The two-tiered prompting scheme captures both the overarching inter-shot narrative relationships and the fine-grained per-shot cinematic details.

Relational Prompts (Prel). Prel encapsulates the semantic and cinematic linkage between Scond and Stgt. It includes a high-level overview of their shared visual context (scene, key characters in transition), a narrative interpretation of the shot transition (including editing techniques like shot/reverse shot or cutaways), and evaluations of scene-level (spatial, temporal, stylistic) and character (appearance, attire) continuity. An example is shown in Figure 4.

Individual Prompts (Pind). For each shot (Scond and Stgt) independently, Pind offers a detailed description of its visual content and cinematographic characteristics [29]. This comprises: (1) a concise summary (subject, setting), a detailed description of visual content (appearance, posture, expression, costume, background); and (2) a structured set of cinematography attributes (e.g., shot size/framing, composition, camera angle, focal length). This schema provides a fine-grained representation of visual language. To enhance robustness against incomplete real-world descriptions, the detailed description and attributes are subject to a 20% dropout rate during training. An example is provided in Figure 4.

This two-stage dataset construction, coupled with our hierarchical annotation strategy, provides a robust foundation for training Cut2Next. Initial training on RawCuts fosters broad visual understanding, while subsequent refinement on CuratedCuts, guided by these detailed prompts, specializes the model for high-quality, cinematically sophisticated next shot generation.

#### 3.2 Cut2Next

To enable the generation of continuous and coherent multi-shot sequences, we leverage the strong generative capabilities of FLUX.1-dev [7], a state-of-the-art Diffusion Transformer (DiT) model. This foundation underpins our proposed method, Cut2Next, a novel framework for synthesizing subsequent shots. The design and operational principles of Cut2Next are elaborated in this section. The overview of Cut2Next is illustrated in Figure 5.

#### 3.2.1 Leveraging Flux via Parameter-Free Modification

Traditional Text-to-Image (T2I) models iteratively denoise Gaussian noise to generate images. The VAE encoder E(·) first maps the target image Stgt to its latent ztgt0 . Gaussian noise for timestep t is

then added, yielding the noisy latent ztgtt . This ztgtt is concatenated with the text conditioning c to form the DiT model’s input.

To incorporate the conditional shot Scond, we also use the pre-trained VAE encoder E(·) to map Scond to its latent zcond = E(Scond). This ensures zcond is in the same latent space as ztgt0 , maintaining compatibility without new encoders. zcond is then integrated with the noisy target latent ztgtt and the primary text conditioning c.

Furthermore, Cut2Next uses a Hierarchical Multi-Prompting strategy for textual guidance. All prompts are processed by a shared, pre-trained text encoder T (·) (e.g., T5 [39]): (1) A relational

prompt Prel is encoded to crel = T (Prel); (2) Individual prompts Pcondind and Ptgtind result in cindcond = T (Pcondind ) and cindtgt = T (Ptgtind). These textual embeddings and visual latents are concatenated to form the DiT model’s input sequence zmodel:

zmodel = concat(crel,cindcond,cindtgt ,zcond,ztgtt ). (1)

This rich input zmodel allows the DiT to condition on visual context from Scond and multi-level textual instructions, efficiently reusing existing encoders.

To adapt shared DiT blocks for this complex, multi-modal input, lightweight LoRA fine-tuning [17] is applied. This enables the model to learn the interplay between these conditioning signals without full-parameter updates.

#### 3.2.2 Context-Aware Condition Injection (CACI)

Standard Flux uses AdaLN-Zero [35] with synchronously noised visual tokens and a unified textual context (cpool), all conditioned by timestep t. Our Hierarchical Multi-Prompting, however, introduces heterogeneous inputs: noise-free conditional visual latents (zcond), noisy target visual latents (ztgtt ), and multiple distinct textual embeddings (crel,cindcond,cindtgt ). This departure from standard conditioning demands a nuanced approach.

We propose Context-Aware Condition Injection (CACI). CACI enables DiT blocks to be aware of each token segment’s context and role. It then tailors AdaLN-Zero inputs for each segment. Specifically, CACI differentiates visual latent conditioning: (1) Noise-free zcond tokens are modulated by AdaLN layers. These use t = 0 (reflecting their clean state) and context cindcond_pool (from Pcondind ). (2) Noisy ztgtt tokens use the diffusion timestep t and context cindtgt _pool (from Ptgtind). This allows specialized processing for zcond and ztgtt . CACI also applies context-awareness to textual tokens. Their AdaLN conditioning aligns with their semantic scope. Individual textual tokens cindcond and cindtgt mirror the timesteps of their visual counterparts (t = 0 for zcond; diffusion t for ztgtt ). For relational tokens crel, empirical results (Figure 7) showed that t = 0 yielded a lower initial loss than diffusion t, despite similar final convergence. Thus, we use t = 0 for crel, treating it as part of the "initial, clean context" with zcond. In essence, CACI’s context-aware, token-type-specific conditioning within AdaLN-Zero layers effectively manages our strategy’s heterogeneous conditioning.

#### 3.2.3 Hierarchical Attention Mask (HAM) for Structured Interactions

Our composite input zmodel requires structured information flow in DiT blocks. Standard full selfattention might undesirably mix distinct conditioning signals, diluting their influence. For example, individual prompts should guide their corresponding visual segments. The relational prompt should mediate between visual components. Direct cross-interference between disparate textual cues must be avoided.

We introduce the Hierarchical Attention Mask (HAM), a predefined, non-learnable binary mask for self-attention. HAM selectively controls attention between token types. This structures information exchange according to our Hierarchical Multi-Prompting and CACI, upholding textual prompt independence. HAM enforces specific attention pathways:

- - Visual Dynamics: Conditional (zcond) and target (ztgtt ) visual tokens mutually attend, besides intra-segment self-attention.
- - Localized Individual Prompts:

- (1) Conditional text (cindcond) attends only to itself and zcond; zcond also attends back to cindcond.
- (2) A similar isolated scope exists for target text (cindtgt ) and ztgtt .
- (3) Attention is masked between an individual text prompt and non-corresponding visual segments

or other text prompts (including crel). This ensures Pcondind and Ptgtind exclusively modulate their designated visual counterparts without cross-contamination.

- Relational Prompt Bridging:

- (1) Relational text (crel) performs self-attention. Its cross-modal interaction is with both zcond and ztgtt (and vice-versa). This broad visual scope is crucial for crel to establish inter-shot relationships.
- (2) crel is masked from attending to cindcond or cindtgt , and vice-versa, maintaining textual independence. By enforcing these pathways, HAM enables Cut2Next to disentangle and leverage multi-level

guidance. Each signal contributes precisely to synthesizing Stgt relative to Scond, without undesired textual cross-talk.

- 4 Experiments

#### 4.1 Experimental Setup

Implementation details. We adopt FLUX.1-dev as our base model, with LoRA [17] layers set to a rank of 256. The Adam [25] optimizer is employed with a learning rate of 1 × 10−4. Our model is trained on shot images with varying aspect ratios, at a resolution of approximately 1K pixels (e.g., 1024 × 576). Our two-stage training strategy begins with pre-training the model on the RawCuts Dataset for 2 epochs. This uses an accumulated batch size of 64 on 8×A100 GPUs. Subsequently, the model is fine-tuned for 2500 iterations on the CuratedCuts Dataset.

Baselines. Cut2Next is pioneering work in the domain of conditional shot generation. To date, no publicly available open-source baselines specifically designed for generating a subsequent shot conditioned on a preceding one exist for direct comparison. Therefore, we adapted the IC-LoRA [18] framework, originally designed for text-to-multiple-image generation, to serve as a strong baseline for our task. We term this adapted version IC-LoRA-Cond.

The original IC-LoRA generates multiple shots from a textual description using markers like [IMAGE1], [IMAGE2]. To adapt it for conditional shot generation, we modify the first shot’s input: we directly use the clean latent of the provided conditional shot image Scond. The subsequent target shot is then generated conditioned on the Scond latents. Loss is calculated only on the target shot’s noisy latents. Textual input uses IC-LoRA’s concatenated prompt: an overall description plus separate descriptions for the conditional ([IMAGE1]) and target ([IMAGE2]) shots.

Evaluation. We evaluate Next Shot Generation using standard automatic metrics for visual consistency and textual fidelity. Regarding visual consistency, we measure the cosine similarity between the input shot and generated shot embeddings. This uses two feature spaces: CLIP [38] feature space (CLIP-I Similarity) and DINO [33] feature space (DINO Similarity). Regarding textual fidelity, we assess alignment of the generated shot with its individual prompt (Ptgtind), we compute the cosine similarity between its CLIP image embedding and the prompt’s CLIP text embedding (CLIP-T Fidelity). To measure perceptual similarity to real cinematic choices, we evaluate against the ground-truth (GT) shots using Fréchet Inception Distance (FID). All evaluations use our new CutBench benchmark. CutBench contains hundreds of diverse movie shot images with our hierarchical prompts designed to elicit specific cinematic continuations that test both continuity and varied editing.

#### 4.2 Main results

Qualitative analyses. We compare Cut2Next with IC-LoRA-Cond [18]. Figure 6 empirically demonstrates Cut2Next’s significant advantages in next shot generation. For instance, in a diner conversation (Row 1), Cut2Next generates a shot/reverse shot maintaining character identity ("Blond Haired Man"), crucial for dialogue. IC-LoRA-Cond fails, introducing a dissimilar individual and disrupting narrative flow. Similarly, for "Mr. Weaver" (Row 2), Cut2Next logically continues focus with appropriate framing. IC-LoRA-Cond again introduces an unrelated figure. In an indoor setting (Row 3), Cut2Next successfully transitions from a wider view to a close-up of the "Woman

[Figure 31]

[Figure 32]

[Figure 33]

|[Figure 34]<br><br>❌|
|---|

|[Figure 35]<br><br>✅|
|---|

Input shot image

IC-LoRA-Cond

Cut2Next

| |
|---|

Prompt: <Dark Haired Man> and <Blond Haired Man> are sitting in a diner booth, seemingly having a conversation. The transition between shots employs a shot/reverse shot technique, cutting from <Dark Haired Man> to <Blond Haired Man> during their diner conversation, likely to show each character's reactions.; [IMAGE1] Medium shot of <Dark Haired Man> (long dark hair, dark maroon shirt) in a diner booth holding a slice of pizza.; [IMAGE2] Medium shot of <Blond Haired Man> (blond hair, grey hoodie) in a diner booth looking serious.

[Figure 36]

[Figure 37]

[Figure 38]

|[Figure 39]<br><br>✅|
|---|

|[Figure 40]<br><br>❌|
|---|

| |
|---|

Prompt: In an outdoor rooftop setting, <Mr. Weaver> (Man with short brown hair) is sitting at a table, looking pensive. The transition between shots is a cut, shifting from a close-up of <Mr. Weaver>'s face to a wider shot of him sitting at the table with a cup of coffee. [IMAGE1] Close-up shot focusing on <Mr. Weaver> (Man with short brown hair), looking pensive; [IMAGE2] Medium shot focusing on <Mr. Weaver> (Man with short brown hair) seated at a table outdoors, looking down.

[Figure 41]

[Figure 42]

[Figure 43]

|[Figure 44]<br><br>❌|
|---|

|[Figure 45]<br><br>✅|
|---|

Prompt: <Woman with Glasses> and <Man with a Towel> are in the entryway of a house, potentially discussing something. The transition between the shots appears to be a simple cut, shifting focus from a wider view of both characters to a close-up of <Woman with Glasses>. [IMAGE1] Medium shot of <Woman with Glasses> and <Man with a Towel> standing in a house entryway with a staircase in the background; [IMAGE2] Medium close-up shot of <Woman with Glasses> in a room with patterned wallpaper.

Figure 6: Visual comparison of Cut2Next and IC-LoRA-Cond [18].

with Glasses," maintaining focus and continuity. IC-LoRA-Cond’s output deviates significantly. This visual comparison affirms Cut2Next’s enhanced capacity for generating temporally coherent, character-consistent, and cinematically plausible subsequent shots.

Quantitative evaluations. As shown in Table 1, our quantitative evaluation on CutBench demonstrates Cut2Next’s superior performance over IC-LoRA-Cond across all metrics. For inter-shot visual consistency, Cut2Next achieves higher DINO Similarity (0.4952 vs. 0.4669) and CLIP-I Similarity (0.7298 vs. 0.7152), indicating greater visual continuity with the input shot. For textual fidelity, Cut2Next also shows enhanced alignment, with a CLIP-T Fidelity of 0.2979 compared to IC-LoRA-Cond’s 0.2805. Besides, our method achieves a significantly lower FID (59.37) than the baseline (80.43). These metrics collectively affirm that Cut2Next generates shots that are more visually consistent with the preceding context and more faithful to textual descriptions.

- Table 1: Performance comparison between Cut2Next and IC-LoRA-Cond on CutBench. Best scores are in bold.

DINO ↑ CLIP-I ↑ CLIP-T ↑ FID ↓

IC-LoRA-Cond 0.4669 0.7152 0.2805 80.43 Cut2Next (ours) 0.4952 0.7298 0.2979 59.37

#### 4.3 Ablation study

Effectiveness of two-stage training. To validate our two-stage strategy, we compared our full "Cut2Next (ours)" with variants fine-tuned solely on RawCuts or CuratedCuts (Table 2). The RawCutsonly model achieved DINO 0.4853, CLIP-I 0.7135, and CLIP-T 0.2951. The smaller CuratedCutsonly model performed better (DINO 0.4913, CLIP-I 0.7221, CLIP-T 0.2973), highlighting data quality’s impact, especially with a strong base like FLUX.1-dev. Our two-stage "Cut2Next (ours)" yields the best results, validating our approach. This shows a synergistic benefit: RawCuts pre-training builds a broad foundation, refined by CuratedCuts fine-tuning, balancing robustness with specialized performance.

Figure 9 visually confirms this. Our full model captures challenging attributes ("green highlights," top row) and complex scenarios ("injured man," "SECURITY GUARD," bottom row). The variant fine-tuned only on CuratedCuts (w/o pretraining) fails on these specifics. This demonstrates that while CuratedCuts refines continuity, RawCuts pre-training provides crucial robustness for complex or less common visual elements.

- Table 2: Cut2Next ablation on training datasets. DINO, CLIP-I, and CLIP-T scores. Higher is better. Best in bold.

DINO ↑ CLIP-I ↑ CLIP-T ↑

Cut2Next (RawCuts-only) 0.4853 0.7135 0.2951 Cut2Next (CuratedCuts-only) 0.4913 0.7221 0.2973 Cut2Next (ours) 0.4952 0.7298 0.2979

Effectiveness of Hierarchical Multi-Prompting strategy. To validate our Hierarchical MultiPrompting strategy, we ablated the relational prompt (Prel), comparing our full Cut2Next model against a variant using only individual prompts ("Cut2Next w/o relational prompt"), with adapted CACI and HAM. Table 3 shows that removing Prel significantly degrades inter-shot visual consistency: our full model achieved higher DINO (0.4952 vs. 0.4752) and CLIP-I Similarity (0.7298 vs. 0.7140) scores. This demonstrates Prel’s crucial role in generating visually coherent subsequent shots. Interestingly, text fidelity (CLIP-T) showed a marginal decrease for the full model (0.2979) compared to the ablated version (0.2984). This suggests Prel, while significantly enhancing visual continuity, might introduce a subtle trade-off or different emphasis in overall textual adherence.

- Table 3: Comparison of Cut2Next and Cut2Next without relational prompt. DINO, CLIP-I, and CLIP-T scores. Higher is better. Best in bold.

DINO ↑ CLIP-I ↑ CLIP-T ↑

Cut2Next (w/o relational prompt) 0.4752 0.7140 0.2984 Cut2Next (ours) 0.4952 0.7298 0.2979

Qualitative results in Figure 8 corroborate these findings. Without Prel (right column), issues like character identity loss (e.g., second row) and failure to follow complex editing instructions like "shot/reverse shot" (first row) become apparent, even with individual prompts. While this ablated version can still match scene environment and tone better than simpler baselines (e.g., IC-LoRACond), it clearly struggles with crucial aspects of cinematic continuity, underscoring the importance of Prel.

Effectiveness of Context-Aware Condition Injection. To validate our Context-Aware Condition Injection (CACI) design, we ablated its components. We compared CACI against: 1.Synchronous Conditioning (SyncCond): A baseline where all visual and textual tokens are conditioned with the current diffusion timestep t, similar to the original Flux paradigm; 2. CACI (crel with t): A CACI variant where relational textual tokens (crel) use the current timestep t instead of t = 0 (as in our main CACI model). Figure 7 shows training loss dynamics on RawCuts. SyncCond exhibited the highest initial loss and slowest convergence, suggesting inefficiency when processing noise-free conditions at non-zero timesteps. In contrast, both CACI variants showed faster initial convergence and lower overall loss. Notably, our proposed CACI (crel with t = 0) achieved a slightly lower initial loss than CACI (crel with t), though both converged similarly. This supports our t = 0 choice for crel, indicating better initial optimization and underscoring the benefit of context-specific timestep conditioning for efficient training.

#### 4.4 Perceptual User Study

To complement automatic evaluation, we conduct a human-centric study to assess whether Cut2Next produces perceptually better results than the IC-LoRA-Cond baseline. The evaluation focuses on two criteria central to our work: (1) Cinematic Continuity (consistency of characters, environment, and key visual details from the prior shot). (2) Adherence to Editing (accurate execution of the intended cut/transition type).

We recruited fifteen participants, including nine graduate students specializing in visual or multimedia fields, three professional video editors, and three lay users with no formal training. All participants reported normal or corrected-to-normal vision and voluntarily provided informed consent. We randomly select 100 samples from the CutBench test split and generated paired outputs from Cut2Next and the IC-LoRA-Cond for each prompt. Participants are presented with the complete input context, including both the text prompt and the corresponding condition shot, and asked to select the preferred result according to two criteria: cinematic quality, and adherence to editing. Each image pair is shown side-by-side in a randomized order, with no option for ties. This process is repeated 4 times to

Training Loss for Different Condition Injection

0.70

SyncCond

0.65

CACI (crel with t)

CACI (ours)

0.60

0.55

0.50

Loss

0.45

0.40

0.35

0.30

0 500 1000 1500 2000 2500

Step

Figure 7: Training loss comparison on RawCuts. CACI (ours) shows improved convergence over SyncCond and CACI (crel with t).

compute the standard deviation. For each criterion, we calculate the average preference rate of each method.

As shown in Table 4, Cut2Next is overwhelmingly preferred across all criteria, demonstrating significant perceptual advantages over IC-LoRA-Cond.

- Table 4: Human preference (%) between Cut2Next and IC-LoRA-Cond [18]. Higher is better and best is in bold.

Method Cinematic Continuity ↑ Adherence to Editing ↑ IC-LoRA-Cond [18] 6.3 ± 1.8 3.5 ± 0.8 Cut2Next (ours) 93.7 ± 1.8 96.5 ± 0.8

- 5 Conclusion and limitations

We introduced Next Shot Generation (NSG), a novel task targeting the critical need for both professional editing patterns and strict cinematic continuity in multi-shot video. Our framework, Cut2Next, leverages a Diffusion Transformer with an innovative Hierarchical Multi-Prompting strategy. Architectural enhancements (CACI and HAM) seamlessly integrate these controls. Evaluations on our new datasets (RawCuts, CuratedCuts) and benchmark (CutBench) show Cut2Next excels quantitatively. Crucially, user studies confirm a strong preference for Cut2Next’s adherence to editing patterns and overall cinematic continuity. Our work focuses on canonical, human-centric editing patterns, thus it is limited to human-centric scenarios. Besides, it may fail in producing action sequences as we filtered high-motion shots to ensure keyframe quality. Long-term coherence is still a critical challenge. A naive autoregressive approach fails for our task because cinematic cuts (e.g., shot/reverse shot) create large visual shifts, leading to character identity loss.

### References

- [1] Kling. Accessed December 9, 2024 [Online] https://klingai.kuaishou.com/, 2024.
- [2] Sora. Accessed February 15, 2024 [Online] https://sora.com/library, 2024.
- [3] Yuval Atzmon, Rinon Gal, Yoad Tewel, Yoni Kasten, and Gal Chechik. Multi-shot character consistency for text-to-video generation. arXiv preprint arXiv:2412.07750, 2024.
- [4] Jianhong Bai, Menghan Xia, Xintao Wang, Ziyang Yuan, Xiao Fu, Zuozhu Liu, Haoji Hu, Pengfei Wan, and Di Zhang. Syncammaster: Synchronizing multi-camera video generation from diverse viewpoints. arXiv preprint arXiv:2412.07760, 2024.

- [5] Hritik Bansal, Yonatan Bitton, Michal Yarom, Idan Szpektor, Aditya Grover, and Kai-Wei Chang. Talc: Time-aligned captions for multi-scene text-to-video generation. arXiv preprint arXiv:2405.04682, 2024.
- [6] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, et al. Lumiere: A space-time diffusion model for video generation. In SIGGRAPH Asia 2024 Conference Papers, pages 1–11, 2024.
- [7] Black Forest Labs. Flux: Official inference repository for flux.1 models, 2024. Accessed: 2024-11-12.
- [8] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.
- [9] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF international conference on computer vision, pages 22560–22570, 2023.
- [10] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Juncheng Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengchen Ma, et al. Skyreels-v2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025.
- [11] FFmpeg Developers. FFmpeg. https://ffmpeg.org.
- [12] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022.
- [13] Rinon Gal, Or Lichter, Elad Richardson, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Lcm-lookahead for encoder-based text-to-image personalization. In European Conference on Computer Vision, pages 322–340. Springer, 2024.
- [14] Gemini Google, Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.
- [15] Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation, 2025.
- [16] Xuanhua He, Quande Liu, Shengju Qian, Xin Wang, Tao Hu, Ke Cao, Keyu Yan, and Jie Zhang. Id-animator: Zero-shot identity-preserving human video generation. arXiv preprint arXiv:2404.15275, 2024.
- [17] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.
- [18] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775, 2024.
- [19] Qingqiu Huang, Yu Xiong, Anyi Rao, Jiaze Wang, and Dahua Lin. Movienet: A holistic dataset for movie understanding. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part IV 16, pages 709–727. Springer, 2020.
- [20] Yuzhou Huang, Ziyang Yuan, Quande Liu, Qiulin Wang, Xintao Wang, Ruimao Zhang, Pengfei Wan, Di Zhang, and Kun Gai. Conceptmaster: Multi-concept video customization on diffusion transformer models without test-time tuning. arXiv preprint arXiv:2501.04698, 2025.
- [21] Jaided AI. EasyOCR. https://github.com/JaidedAI/EasyOCR, 2024.

- [22] Yuming Jiang, Tianxing Wu, Shuai Yang, Chenyang Si, Dahua Lin, Yu Qiao, Chen Change Loy, and Ziwei Liu. Videobooth: Diffusion-based video generation with image prompts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6689–6700, 2024.
- [23] Ozgur Kara, Krishna Kumar Singh, Feng Liu, Duygu Ceylan, James M Rehg, and Tobias Hinz. Shotadapter: Text-to-multi-shot video generation with diffusion models. arXiv preprint arXiv:2505.07652, 2025.
- [24] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pages 5148–5157, 2021.
- [25] Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.
- [26] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.
- [27] LAION-AI. aesthetic-predictor. https://github.com/LAION-AI/aesthetic-predictor, 2022.
- [28] Zhong-Yu Li, Ruoyi Du, Juncheng Yan, Le Zhuo, Zhen Li, Peng Gao, Zhanyu Ma, and MingMing Cheng. Visualcloze: A universal image generation framework via visual in-context learning. arXiv preprint arXiv:2504.07960, 2025.
- [29] Hongbo Liu, Jingwen He, Yi Jin, Dian Zheng, Yuhao Dong, Fan Zhang, Ziqi Huang, Yinan He, Yangguang Li, Weichao Chen, et al. Shotbench: Expert-level cinematic understanding in vision-language models. arXiv preprint arXiv:2506.21356, 2025.
- [30] Tao Liu, Kai Wang, Senmao Li, Joost van de Weijer, Fahad Shahbaz Khan, Shiqi Yang, Yaxing Wang, Jian Yang, and Ming-Ming Cheng. One-prompt-one-story: Free-lunch consistent text-toimage generation using a single prompt. arXiv preprint arXiv:2501.13554, 2025.
- [31] Fuchen Long, Zhaofan Qiu, Ting Yao, and Tao Mei. Videostudio: Generating consistent-content and multi-scene videos. In European Conference on Computer Vision, pages 468–485. Springer, 2024.
- [32] Chaojie Mao, Jingfeng Zhang, Yulin Pan, Zeyinzi Jiang, Zhen Han, Yu Liu, and Jingren Zhou. Ace++: Instruction-based image creation and editing via context-aware content filling. arXiv preprint arXiv:2501.02487, 2025.
- [33] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
- [34] Alejandro Pardo, Fabian Caba Heilbron, Juan León Alcázar, Ali Thabet, and Bernard Ghanem. Moviecuts: A new dataset and benchmark for cut type recognition. In European Conference on Computer Vision, pages 668–685. Springer, 2022.
- [35] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.
- [36] Quynh Phung, Long Mai, Fabian David Caba Heilbron, Feng Liu, Jia-Bin Huang, and Cusuh Ham. Cineverse: Consistent keyframe synthesis for cinematic scene composition. arXiv preprint arXiv:2504.19894, 2025.
- [37] Tianhao Qi, Jianlong Yuan, Wanquan Feng, Shancheng Fang, Jiawei Liu, SiYu Zhou, Qian He, Hongtao Xie, and Yongdong Zhang. Mask2dit: Dual mask-based diffusion transformer for multi-scene long video generation, 2025.

- [38] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.
- [39] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research, 21(140):1–67, 2020.
- [40] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500–22510, 2023.
- [41] Tomáš Souˇcek and Jakub Lokoˇc. Transnet v2: An effective deep network architecture for fast shot transition detection. arXiv preprint arXiv:2008.04838, 2020.
- [42] Zhenxiong Tan, Songhua Liu, Xingyi Yang, Qiaochu Xue, and Xinchao Wang. Ominicontrol: Minimal and universal control for diffusion transformer. arXiv preprint arXiv:2411.15098, 2024.
- [43] Yoad Tewel, Omri Kaduri, Rinon Gal, Yoni Kasten, Lior Wolf, Gal Chechik, and Yuval Atzmon. Training-free consistent text-to-image generation. ACM Transactions on Graphics (TOG), 43(4):1–18, 2024.
- [44] TostAI. NSFW Image Detection (Large). https://huggingface.co/TostAI/ nsfw-image-detection-large, 2024.
- [45] Bo Wang, Haoyang Huang, Zhiyin Lu, Fengyuan Liu, Guoqing Ma, Jianlong Yuan, Yuan Zhang, and Nan Duan. Storyanchors: Generating consistent multi-scene story frames for long-form narratives. arXiv preprint arXiv:2505.08350, 2025.
- [46] Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, and Hongming Shan. Dreamvideo: Composing your dream videos with customized subject and motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6537–6549, 2024.
- [47] Shaojin Wu, Mengqi Huang, Wenxu Wu, Yufeng Cheng, Fei Ding, and Qian He. Less-tomore generalization: Unlocking more controllability by in-context generation. arXiv preprint arXiv:2504.02160, 2025.
- [48] Junfei Xiao, Feng Cheng, Lu Qi, Liangke Gui, Jiepeng Cen, Zhibei Ma, Alan Yuille, and Lu Jiang. Videoauteur: Towards long narrative video generation. arXiv preprint arXiv:2501.06173, 2025.
- [49] Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Gongye Liu, Xintao Wang, Ying Shan, and Tien-Tsin Wong. Dynamicrafter: Animating open-domain images with video diffusion priors. In European Conference on Computer Vision, pages 399–417. Springer, 2024.
- [50] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.
- [51] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.
- [52] Shilong Zhang, Lianghua Huang, Xi Chen, Yifei Zhang, Zhi-Fan Wu, Yutong Feng, Wei Wang, Yujun Shen, Yu Liu, and Ping Luo. Flashface: Human image personalization with high-fidelity identity preservation. arXiv preprint arXiv:2403.17008, 2024.
- [53] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023.

- [54] Canyu Zhao, Mingyu Liu, Wen Wang, Weihua Chen, Fan Wang, Hao Chen, Bo Zhang, and Chunhua Shen. Moviedreamer: Hierarchical generation for coherent long visual sequence. arXiv preprint arXiv:2407.16655, 2024.
- [55] Mingzhe Zheng, Yongqi Xu, Haojian Huang, Xuran Ma, Yexin Liu, Wenjie Shu, Yatian Pang, Feilong Tang, Qifeng Chen, Harry Yang, et al. Videogen-of-thought: A collaborative framework for multi-shot video generation. arXiv preprint arXiv:2412.02259, 2024.
- [56] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent self-attention for long-range image and video generation. Advances in Neural Information Processing Systems, 37:110315–110340, 2024.

[Figure 46]

[Figure 47]

[Figure 48]

Input shot image Cut2Next (w/o relational prompt) Cut2Next

Prompt: In an interior setting, <man with suit> (dark-skinned man with short hair and a light-colored suit) and <woman with beaded dress> (light-skinned woman with dark hair and a beaded dress) are engaged in an intimate moment, likely a conversation or preparation for an event. The transition between shots employs a shot/reverse shot editing technique, cutting between <man with suit> and <woman with beaded dress> during their interaction, emphasizing their emotions and creating a sense of intimacy. [IMAGE1] Medium close-up of <woman with beaded dress> looking at <man with suit> in a dimly lit room; [IMAGE2] Medium close-up of <man with suit> looking at <woman with beaded dress> in a dimly lit room.

|[Figure 49]<br><br>❌|
|---|

[Figure 50]

[Figure 51]

[Figure 52]

|[Figure 53]<br><br>✅|
|---|

Prompt: In an outdoor setting with lush greenery, <Standing Man> (Man in olive green with crossed arms) observes as <Shooter> (Man holding rifle) practices target shooting. The transition between shots provides a shift in focus from both characters engaging in target practice to a closer look at <Standing Man> (Man in olive green with crossed arms), possibly to capture his reaction or perspective. [IMAGE1] Wide shot showing <Standing Man> (Man in olive green with crossed arms) observing <Shooter> (Man holding rifle) practicing at an outdoor shooting range; [IMAGE2] Medium shot of <Standing Man> (Man in olive green with crossed arms) standing outdoors with arms crossed and a neutral expression.

[Figure 54]

[Figure 55]

[Figure 56]

|[Figure 57]<br><br>❌|
|---|

|[Figure 58]<br><br>✅|
|---|

Prompt: <Shuri> (Woman with curly hair) is depicted on a beach, possibly after a significant event, conveyed through her thoughtful and contemplative expression. The transition between shots appears to show a shift in perspective, possibly to emphasize the character’s emotional state or the surrounding environment. [IMAGE1] Medium shot of <Shuri> (Woman with curly hair) on a beach, looking over her shoulder with a fire in the foreground; [IMAGE2] Medium shot of <Shuri> (Woman with curly hair) on a beach with palm trees in the background.

##### Figure 8: Visual comparison of Cut2Next and Cut2Next (w/o relational prompt).

[Figure 59]

[Figure 60]

[Figure 61]

|[Figure 62]<br><br>✅|
|---|

Input shot image Cut2Next (w/o pretraining) Cut2Next

|[Figure 63]<br><br>❌|
|---|

Prompt: In a brightly lit bedroom, <Mom> (short blond hair) and <Daughter> (long blond hair with green highlights) are present. The scene focuses on their interaction within the room. The

transition between shots likely serves to shift focus from <Mom> to <Daughter> in the bedroom, emphasizing <Daughter>'s emotional state and presence within the setting. [IMAGE1] Medium shot of <Mom> (short blond hair) standing in a bedroom while <Daughter> (long blond hair with green highlights) sits on the bed in the background; [IMAGE2] Medium shot of <Daughter> (long blond hair with green highlights) seated on a bed, facing away from the camera.

[Figure 64]

[Figure 65]

[Figure 66]

|[Figure 67]<br><br>✅|
|---|

|[Figure 68]<br><br>❌|
|---|

|[Figure 69]<br><br>✅|
|---|

Prompt: <RYAN> (Fair-skinned man with auburn hair), injured and inside a car, is potentially being questioned or observed by <SECURITY GUARD> (Older man in dark uniform) in a dark, industrial setting. The transition between shots offers a wider view, revealing <RYAN>‘s (Fair-skinned man with auburn hair) position relative to <SECURITY GUARD> (Older man in dark uniform) and the surrounding environment, possibly establishing spatial relationships and adding context to the interaction. [IMAGE1] Close-up shot of <RYAN> (Fair-skinned man with auburn hair) injured inside a car, showing concern; [IMAGE2] Medium shot of <RYAN> (Fair-skinned man with auburn hair) in a car with <SECURITY GUARD> (Older man in dark uniform) standing next to the car in an industrial setting.

##### Figure 9: Visual comparison of Cut2Next and Cut2Next (without pretraining on RawCuts dataset).

Multi-Angle

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

Cut-In/Cut-Out

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Shot/Reverse Shot

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

Cutaway

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

- Figure 10: More visual results from Cut2Next demonstrating performance across different editing patterns: Multi-Angle, cut-in/cut-out, shot/reverse shot, and cutaway. For each pair, the input shot image is shown above the generated subsequent shot. Prompts are omitted due to space limitations.

