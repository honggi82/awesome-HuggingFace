## CubePart: An Open-Vocabulary Part-Controllable 3D Generator

YIHENG ZHU∗, Roblox, USA KANGLE DENG∗, Roblox, USA JEAN-PHILIPPE FAUCONNIER∗, Roblox, USA INAKI NAVARRO∗, Roblox, USA DAIQING LI, Roblox, USA AVA PUN, Roblox, USA and Carnegie Mellon University, USA YINAN ZHANG, Roblox, USA PEIYE ZHUANG, Roblox, USA XIAOXIA SUN, Roblox, USA MANEESH AGRAWALA, Roblox, USA and Stanford University, USA KIRAN BHAT, Roblox, USA TINGHUI ZHOU, Roblox, USA

# arXiv:2605.28763v1[cs.AI]27May2026

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Generated Single-part Mesh Existing Single-part Mesh from Artists

Input Prompt A giant sea turtle fitted with a stone castle on its shell.

###### +

Turtle Head, Turtle Shell, Flippers, Castle towers, Main Keep structure, Cannons

Input Schema

[Figure 6]

[Figure 7]

Input Mesh

[Figure 8]

[Figure 9]

Back Tanks Head

[Figure 10]

Head

Car Body Exhaust Pipe

Lid

Left Arm

Gun

[Figure 11]

Right Arm

Left Wing

[Figure 12]

Torso Body

Wheels

Right Wing

Base

Right Leg

Tail

Left Leg

[Figure 13]

Body

Headlights

Output Multi-Part Mesh

[Figure 14]

Legs

Output Multi-Part Mesh

Animated Mesh

(a) Open-Vocabulary Part-based Mesh Generation

(b) Result Gallery

Fig. 1. We propose CubePart, an open-vocabulary part-controllable 3D generator. (a) Given a text prompt and a schema defining part decomposition, CubePart synthesizes a multi-part 3D object where each component is a distinct, structurally complete mesh. (b) This controllable part-based generative framework directly facilitates the usage of the resulting assets for scripted or physically simulated behaviors (bottom row). Additionally, CubePart can accept an existing mesh as input, decomposing it into semantic multi-part meshes according to the input part schema (last column). Please refer to the video for animated visualizations.

Interactive 3D assets used in games and simulation are typically decomposed into specific semantic parts to support animation, physics, and scripted behaviors, yet most generative 3D models produce either monolithic meshes or arbitrary part decompositions that cannot be aligned with applicationspecific requirements. We present CubePart, a generative framework for open-vocabulary, part-controllable 3D mesh generation that exposes part structure as an explicit inference-time control signal. Given a global text prompt and a user-defined parts schema expressed as an open-ended list

∗The first four authors contributed equally to this research.

Authors’ Contact Information: Yiheng Zhu, yzhu@roblox.com, Roblox, USA; Kangle Deng, kdeng@roblox.com, Roblox, USA; Jean-Philippe Fauconnier, jfauconnier@roblox. com, Roblox, USA; Inaki Navarro, inavarro@roblox.com, Roblox, USA; Daiqing Li, daiqingli@roblox.com, Roblox, USA; Ava Pun, apun@andrew.cmu.edu, Roblox, USA and Carnegie Mellon University, USA; Yinan Zhang, yinan.zhang@roblox.com, Roblox, USA; Peiye Zhuang, pzhuang@roblox.com, Roblox, USA; Xiaoxia Sun, xiaoxiasun@ roblox.com, Roblox, USA; Maneesh Agrawala, maneesh@cs.stanford.edu, Roblox, USA and Stanford University, USA; Kiran Bhat, kbhat@roblox.com, Roblox, USA; Tinghui Zhou, tzhou@roblox.com, Roblox, USA.

SIGGRAPH Conference Papers ’26, Los Angeles, CA, USA © 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2554-8/2026/07 https://doi.org/10.1145/3799902.3811117

This work is licensed under a Creative Commons Attribution 4.0 International License.

of part names, our method generates a set of meshes—one per schema element—that assemble into a coherent object while respecting the specified semantic structure. To enable this capability, we introduce a scalable data pipeline to construct a large open-vocabulary, part-labeled 3D dataset, along with a two-stage generative architecture that separates global shape synthesis from part-level decoding. We demonstrate that the resulting assets can be directly integrated into game engines and driven by animation and behavior scripts without manual post-processing.

CCS Concepts: • Computing methodologies → Artificial intelligence. Additional Key Words and Phrases: 3D Shape Generation, Part-based Generation

##### ACM Reference Format:

Yiheng Zhu, Kangle Deng, Jean-Philippe Fauconnier, Inaki Navarro, Daiqing Li, Ava Pun, Yinan Zhang, Peiye Zhuang, Xiaoxia Sun, Maneesh Agrawala, Kiran Bhat, and Tinghui Zhou. 2026. CubePart: An Open-Vocabulary PartControllable 3D Generator. In Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers (SIGGRAPH Conference Papers ’26), July 19–23, 2026, Los Angeles, CA, USA. ACM, New York, NY, USA, 17 pages. https://doi.org/10.1145/3799902.3811117

1 Introduction

3D assets in modern games and interactive applications are rarely static. Vehicles require rotating wheels, characters must articulate, containers need to open and close, and many objects respond to physics or scripted events. In game engines, these behaviors are governed by simulation systems, animation rigs, and interaction scripts that operate on a pre-defined set of parts. For an asset to be functional, its mesh must be decomposed into specific semantic components that match the "schema" expected by the game’s code.

Creating meshes that conform to a target part composition remains a largely manual process. Artists must decompose geometry into parts, assign consistent labels, and ensure that the resulting meshes assemble cleanly—an effort that scales poorly with asset diversity. While recent advances in 3D generative modeling have enabled the synthesis of complex geometries from text or image prompts, these methods either produce monolithic meshes without any explicit part structure [Xiang et al. 2025a,b; Yang et al. 2025b] or an arbitrary set of parts [Lin et al. 2025; Tang et al. 2025]; the user has no control for aligning the resulting parts with the schema required by downstream game logic. For a developer with a game that specifically expects a car to be composed of four wheel parts and one chassis part, a model that generates a random set of part segments is as unhelpful as a model that generates a car as a single monolithic object.

One mightattempttoobtainpart-level control through 2D ground-

ing, for example, by using an image segmentation model [Carion et al. 2025; Kirillov et al. 2023] to generate segmentation masks for mask-conditioned 3D part generation models like OmniPart [Yang et al. 2025c]. However, a 2D mask cannot represent or control parts that are hidden from the input view. For instance, the rear tail of an animal cannot be specified or controlled from a single front-facing view. More fundamentally, 2D control signals are view-dependent and ambiguous when lifted to 3D, making them ill-suited for defining complete semantic decompositions of 3D objects.

These limitations highlight a critical need for a 3D-native, schemadriven control interface that allows users to explicitly specify the

semantic structure of an object during generation. Such control must also be flexible: different applications may require different decompositions of the same object. For example, one game may need car doors as separate parts to enable opening animations, while another may require the hood to be independently controllable to expose the engine. Fixed or closed-vocabulary part schemas cannot accommodate this diversity of downstream requirements. We argue that text, as a modality, provides a natural and universal interface for such control. Crucially, a text prompt can specify both a global description of the desired object and an explicit parts schema, an open-ended list of part names that serves as a structural blueprint for decomposing the object into semantically meaningful components.

In this paper, we present CubePart, the first generative framework for open-vocabulary, part-controllable 3D mesh generation. Our system takes as input a global text prompt describing the object (e.g., “a jellyfish-themed race car”) together with a desired parts schema (e.g., {"car body", "front left wheel", ...}). It outputs a set of distinct meshes, one per schema element, that jointly assemble into a coherent object. Because the generation is guided by the user-provided schema, the resulting assets can directly match the requirements of game engines and animation systems without manual intervention (as we demonstrate in Section 6).

To support this capability, we introduce CubePart, a framework underpinned by a high-fidelity data engine and a novel multi-stage generative architecture. Our data engine leverages vision-language models (VLMs) and a novel 3D-aware "Set-of-Mark" [Yang et al. 2023] annotation strategy to curate a semantically grounded dataset of 462K assets and about 2M parts. Compared to existing 3D part datasets, oursisbothlargerscale(over 11 times larger than PartVerseXL [Ding et al. 2025]) and produces higher quality part labels required for precise open-vocabulary control. Building on this foundation, our architecture employs a two-stage diffusion process: the first stage generates a full mesh conditioned on both the prompt describing the object and the part schema, and the second stage decomposes the full mesh into corresponding parts specified by the schema while ensuring global geometric coherence through a novel cross-part attention mechanism with zero-initialized attention blocks.

In summary, our main contributions include:

- • A scalable data engine for constructing open-vocabulary, partlabeled 3D datasets from unstructured meshes, leveraging VLMs for 3D-aware clustering and semantic captioning.
- • A schema-driven two-stage generative architecture that supports open-vocabulary, part-controllable 3D mesh generation while preserving global coherence across parts.
- • An end-to-end demonstration showing how the generated multi-part meshes can be integrated into game engines and driven by behavior scripts without manual post-processing.

2 Related Work 2.1 3D Generative Models

Recent progress in 3D generative modeling was initially driven by 2D-to-3D lifting approaches, most notably DreamFusion [Poole et al. 2022], which introduced Score Distillation Sampling (SDS) to optimize implicit 3D representations using pretrained 2D diffusion

priors. A large body of follow-up work [Gao et al. 2022; Lin et al. 2023; Liu et al. 2023b; Wang et al. 2023] adopts this paradigm, leveraging strong 2D priors to compensate for limited 3D data. Despite impressive visual quality, these methods rely on view-dependent image supervision and provide only weak constraints on 3D structure, offering no explicit control over semantic part decomposition.

With the availability of large-scale 3D datasets such as Objaverse [Deitke et al. 2023b] and Objaverse-XL [Deitke et al. 2023a], 3D-native generative modeling has become increasingly practical. 3DShape2VecSet [Zhang et al. 2023] introduces a compact latent-set representation that enables diffusion directly in a 3D-aligned latent space, and subsequent works [Lai et al. 2025; Li et al. 2025a,c,d; Team et al. 2025; Yang et al. 2025b; Zhang et al. 2024; Zhao et al. 2023] scale this paradigm to high-quality, end-to-end 3D asset generation without reliance on 2D distillation. Building on this representation, our method conditions directly on text rather than images, enabling open-vocabulary semantic control over both object appearance and part composition.

A complementary line of work represents 3D shapes using sparse voxel grids to reduce the memory cost of dense voxelization, as in XCube [Ren et al. 2024], Trellis [Xiang et al. 2025a,b], SparseFlex [He et al. 2025b], Sparc3D [Li et al. 2025b], and Direct3D-S2 [Wu et al. 2025b]. While these methods support localized geometry synthesis and high-resolution detail, they typically generate monolithic meshes and lack explicit mechanisms for semantic part-level control or decomposition.

- 2.2 Part-aware 3D Generation

The growing demand for structured and interactive 3D assets has motivated research on part-aware 3D generation. Early methods rely on category-specific, part-level supervision, learned through autoencoder-based frameworks such as SPAGHETTI [Hertz et al. 2022] and Neural Template [Hui et al. 2022], or diffusion-based approachesincludingSALAD[Kooetal.2024]andDiffFacto[Nakayama

- et al. 2023]. While these methods demonstrate the feasibility of decomposed shape generation, they are restricted to narrow object categories and fixed part taxonomies, limiting their applicability to open-world asset creation and downstream tasks requiring flexible, application-specific part definitions.

Recent methods [Chen et al. 2025a, 2024; Liu et al. 2024] adopt multi-stage pipelines that combine multi-view diffusion–based image synthesis, 2D foundation models for part segmentation, and subsequent 3D reconstruction and composition. Part123 [Liu et al.

- 2024] generates multi-view images from a single input view, applies SAM-based segmentation to extract part masks, and reconstructs parts via multi-view geometry, while PartGen [Chen et al. 2025a] improves robustness by repurposing multi-view diffusion models for multi-view segmentation and part-aware completion. Despite these advances, such approaches remain strongly dependent on 2D segmentation quality and are inherently limited by view-dependent image evidence, often leading to incomplete or inconsistent 3D parts, especially for occluded or self-hidden components.

Several contemporaneous works [Ding et al. 2025; Dong et al. 2025; Hadgi et al. 2026; He et al. 2025a] move toward 3D-native part generation. HoloPart [Yang et al. 2025a] adopts a two-stage

pipeline that segments a 3D shape and applies 3D diffusion to complete occluded regions, whereas PartCrafter [Lin et al. 2025] uses a single unified model to directly synthesize multiple 3D parts from an RGB image without pre-segmentation. PartPacker [Tang et al. 2025] addresses inter-part contact artifacts via a dual-volume packing strategy in SDF space, while AutoPartGen [Chen et al. 2025b] autoregressively generates a variable number of parts with latent diffusion, incurring high computational cost and quality degradation due to error accumulation. BANG [Zhang et al. 2025a] formulates part generation as an object explosion process and recursive refinement that supports both unconditional generation and various explicit control signals, but often fails to preserve fine-grained geometry due to the lack of explicit per-part supervision.

To improve part controllability, OmniPart [Yang et al. 2025c] proposes a two-stage pipeline consisting of a structure planning module that predicts explicit 3D bounding boxes from 2D part masks and images, followed by a 3D-native, spatially conditioned generative model based on Trellis [Xiang et al. 2025b] to synthesize 3D parts. X-Part [Yan et al. 2025] similarly adopts a two-stage design, first leveraging the 3D-native part segmenter P3-SAM [Ma et al. 2025] to produce initial segmentations, bounding boxes, and semantic features, and then performing synchronized multi-part diffusion to generate 3D parts.

Despite this progress, existing methods either assume a fixed or learned part vocabulary or infer part structure implicitly from data or 2D segmentation. In contrast, our approach allows users to directly specify an open-vocabulary list of semantic parts at inference time, and guarantees that the generated meshes align with this user-defined structure, enabling direct integration with downstream animation and interaction pipelines.

2.3 3D Part Datasets

Part-aware generative models rely on datasets in which meshes are decomposed into meaningful components. We define a part dataset as a collection of 3D meshes that are pre-segmented into distinct parts, in contrast to datasets like ShapeNet [Chang et al. 2015], ABO [Collins et al. 2022] or Objaverse/Objaverse-XL [Deitke et al. 2023a,b] that primarily contain monolithic meshes. These parts generally correspond to meaningful object components (such as the left mechanical arm of a robot), though they may also reflect the structural choices of the original artist.

We further characterize a part dataset as "open-vocabulary" if each part is paired with free-form natural language descriptions or names, rather than labels drawn from a fixed taxonomy. This contrasts to closed-vocabulary part datasets that enforce a predefined part taxonomy such as ShapeNetPart [Yi et al. 2016] and PartNet [Mo et al. 2018].

Recent efforts toward open-vocabulary part datasets include PartVerse [Dong et al. 2025] and PartVerse-XL [Ding et al. 2025], which curate subsets of Objaverse, refine their part segmentation using human experts, and generate part captions using large vision-language models (VLMs). These datasets contain approximately 12k and 40k assets, respectively. PartObjaverse-Tiny [Yang et al. 2024] provides manually curated open-vocabulary labels for a uniformly sampled subset of 200 meshes from Objaverse, but is intended primarily for

[Figure 15]

User Input

| | |
|---|---|
| | |

MM-DiTBlock

MM-DiTBlockMM-DiTBlockMM-DiTBlock

MM-DiTBlockMM-DiTBlockMM-DiTBlock

ShapeVAE

###### Text Prompt

###### Schema

Formatted Part Prompt

Decoder

"The object contains following parts: … Target to segment: cab”

"cab", "chassis",

"A tow truck characterized by

Qwen-VL

cartoonish features”

"Wheels",

"The object contains following

Cross-PartAttentionResidualBlock

Cross-PartAttentionResidualBlock

parts: … Target to segment: chassis”

[Figure 16]

"roof beacon", "tow assembly"

[Figure 17]

…

MM-DiTBlockMM-DiTBlock

ShapeVAE

Decoder

Noisy Part Shape Latent

[Figure 18]

…

Qwen-VL

[Figure 19]

…

[Figure 20]

ShapeVAE

MM-DiTBlock

MM-DiTBlock

MM-DiTBlock

Decoder

Noisy Full Shape Latent

Generated Full Shape Latent

[Figure 21]

[Figure 22]

Single-Mesh DiT Multi-Mesh DiT

Output Multi-Part Mesh

(a) Stage 1: Single-Part Mesh Generation (b) Stage 2: Multi-Part Mesh Generation

Fig. 2. Overview. We propose a two-stage framework to generate part-controllable 3D objects conditioned on a global text prompt and a part schema. (a) Single Mesh Generation synthesizes a holistic shape latent using a Multi-Modal DiT (MM-DiT) [Esser et al. 2024], conditioned on the prompt and schema encoded by Qwen-VL [Bai et al. 2023]. (b) Multi-Mesh Generation takes the full shape latent from Stage 1 and decomposes it into distinct part latents. To achieve this, we initialize with the MM-DiT weights from Stage 1 and inject Cross-Part Attention Residual Blocks to enable structural interaction among parts.

evaluation rather than training. Although these datasets represent important progress, they remain expensive to scale and limited in coverage, motivating automated pipelines that can construct largescale, open-vocabulary part datasets from unstructured 3D assets.

- 3 Open-Vocabulary Part-Controllable 3D Generator

We aim to generate part-controllable 3D objects conditioned on a global user prompt describing the overall shape, supplemented by a text-based schema that defines its composing parts, e.g., a sleek sports car with wheels, door, body, and engine. To this end, we propose CubePart, a framework comprised of two key stages: full mesh generation and multi-part mesh generation (Figure 2). In Section 3.1, we provide a brief overview of the vecset-based diffusion transformer for mesh generation introduced in Craftsman [Li et al.

- 2025a] and other follow-up works [Li et al. 2025d; Yang et al. 2025b; Zhang et al. 2024]. We then describe how we adapt this architecture to establish our single-part mesh generation pipeline, which generates a full mesh from a user-defined text prompt. Finally, we introduce the second stage, multi-part mesh generation, detailing how we decompose the single-part mesh into corresponding components defined by the text-based schema.

- 3.1 Preliminary: Vecset Diffusion for Mesh Generation Vecset diffusion models [Li et al. 2025a,d; Yang et al. 2025b; Zhang

- et al. 2024] represent a class of latent diffusion models designed to

generate sets of unordered vectors (vecsets) that implicitly encode 3D shapes. The typical pipeline begins by encoding 3D meshes into latent vector sets using a transformer-based Variational Autoencoder (VAE) using 3DShape2VecSet [Zhang et al. 2023]. The VAE decoder employs a Signed Distance Function (SDF) representation, which enables sharper geometry reconstruction. A diffusion model, often based on flow matching formulations [Esser et al. 2024], is then trained on these VAE latents to generate novel 3D shapes from noise. For image-to-3D generation tasks, these models are commonly conditioned on single-view images through visual features, e.g., DINOv2 [Oquab et al. 2024], injected via cross-attention mechanisms in transformer blocks.

3.2 Stage 1: Single-part mesh generation

While most VecSet diffusion models are image-conditioned, images are not well suited for defining complete 3D semantic structures due to part occlusion. To this end, we adapt the VecSet diffusion model for text-to-3D generation.

Pretraining. To bootstrap the model for more complex tasks, we first pre-train the model on a text-conditioned generation task. We utilize the vecset-based shape VAE [Zhang et al. 2023] and adopt the Multi-Modal Diffusion Transformer (MM-DiT) architecture [Esser et al. 2024], for text conditioned 3D shape latent generation. Additionally, we employ Qwen-VL [Bai et al. 2023] to encode the text prompt, following Qwen-Image [Wu et al. 2025a]. The pretraining

dataset consists of approximately 4.7M mesh-text pairs, combining 745K proprietary assets with about 4M synthetically generated assets for improved text diversity, following the recipe from [Team

Full Shape Latent Part Latent N

N/A Part Prompt N

Reshape

Part Latent 1 Part Latent N Full Shape Latent

… … …

Part Prompt 1 Part Latent 1

- et al. 2025]. This network subsequently serves as the single-mesh generation model to showcase an end-to-end pipeline, though our Stage 2 multi-part mesh generation model is able to take any watertight mesh as input.

###### Cross-Part Attention Residual Block (Zero Initialized)

MM-DiT Block

Reshape & Add

Full Shape Latent Part Latent N

N/A Part Prompt N

Schema-aware Finetuning. While the pre-trained single mesh diffusion model can produce high-quality 3D shapes, the resulting mesh is not guaranteed to contain all the intended parts, even when the input schema is explicitly included in the text prompt. Conversely, certain parts might be disproportionately emphasized (Figure 7). To address these limitations, we fine-tune the base model on our curated dataset, where the text prompts are structured to explicitly enumerate the constituent parts. The full prompt is: “{global caption}. This object contains the following parts: {list of part labels}.”

… …

Part Prompt 1 Part Latent 1

Fig. 3. Cross-part Attention Block. A dedicated zero-initialized Transformer block is designed for cross-part global attention. The residual block takes in all part latent vectors and the conditional full-shape latent vectors as inputs. We insert this block to facilitate efficient inter-part communication while maintaining the pre-trained single-mesh generation capabilities.

Implementation Details. To optimize training and inference efficiency, we downsize the original Qwen-Image model. The number of layers is reduced to 21, and the hidden dimension is 1536, which results in 1.9B trainable number of parameters. We adopt the flow matching training objective, following [Esser et al. 2024; Liu et al. 2023a; Ma et al. 2024]. During training, given a VAE-encoded shape latent 𝑍0 ∼ D sampled from the training dataset D and a random noise sampled from the standard multivariate normal distribution 𝑍1 ∼ N(0, I), the model input latent at timestep 𝑡 is defined as: 𝑍𝑡 = 𝑡𝑍0 + (1 − 𝑡)𝑍1 where the timestep 𝑡 is sampled from a logitnormal distribution and shifted with a factor of 4.0, following [Li et al. 2024]. The text condition latent 𝑐 is obtained from Qwen-VL. The training loss function is defined as:

where zglobal is the latent representation of the full mesh. To distinguish between different components, we employ a part-aware prompt. The text condition is structured as: “This object has the following parts: {list of all parts}. Target to segment: {target part name}.” By explicitly providing the full list of part names, we provide the model with context regarding the other components, helping it better understand the target label and determine its segmentation boundaries. This prompt guides the model to focus on generating the geometry for a specific semantic part, e.g., a “wheel” or “chair leg”, within the context of the whole object.

Cross-part Attention Block. While the straightforward baseline can generate individual components, relying solely on text prompts for global context often results in overlapping or incomplete geometry (Shown in Table 3). To provide a stronger global context, we must modify the single-mesh model to enable information exchange between parts. Prior methods like PartCrafter [Lin et al. 2025] and PartPacker [Tang et al. 2025] address this by altering the original layers of the pre-trained model to perform global attention across all parts. However, we empirically found that such extensive modification is unnecessary and can degrade the pre-trained priors. Instead, we introduce a dedicated zero-initialized Transformer block specifically for global attention (Figure 3). By inserting this block rather than modifying existing ones, we facilitate efficient inter-part communication while minimizing the disruption to the pre-trained single-mesh generation capabilities.

L = E(𝑍0,𝑐)∼D,𝑍1,𝑡 ∥𝑓𝜃 (𝑍𝑡,𝑡,𝑐) − 𝑣𝑡 ∥2, (1)

where 𝑓𝜃 denotes the diffusion network with learnable parameters 𝜃, and 𝑣𝑡 = 𝑍1 − 𝑍0. We use a batch size of 768 and a learning rate of 10−4 with a linear warm-up schedule for the first 2,000 iterations. We adopt AdamW as the optimizer, with 𝛽 values set to 0.9 and 0.99, and weight decay disabled.

- 3.3 Stage 2: Multi-part mesh generation

While Stage 1 has established a robust text-conditioned mesh generator that produces geometry that structurally aligns with the text schema, it produces a single monolithic mesh. In Stage 2, we aim to transform this single mesh into a set of distinct parts. To achieve this efficiently and consistently, we leverage pre-trained weights from Stage 1, adapting the model to output multiple part latents while maintaining the geometric priors learned in pre-training.

Implementation Details. As previously mentioned, we initialize the Stage 2 model using the pre-trained Stage 1 model weights. In total, four cross-part attention blocks are inserted at the 1st, 5th, 9th, and 17th layers. Although these additional blocks increase the model size, we benefit from using fewer computationally expensive crosspart attention blocks while effectively leveraging the pre-trained weights. Training settings generally mirror Stage 1, with the batch size lowered to 72 to handle the multiple parts per sample. All our diffusion models are trained on 24 H200 GPUs. Stage 1’s training takes about 3 days (1500 GPU-hours). Stage 2’s training time is about 18 hours (450 GPU-hours). Inference on H200 takes 2-3 seconds for Stage 1, and 3-4 seconds for Stage 2, both including VAE decoding.

We represent a multi-part object as a set of 𝑁 parts 𝑂 = {𝑝𝑖}𝑖𝑁=1, where each part 𝑝𝑖 can be encoded by a set of latent tokens z𝑖 = {𝑧𝑖𝑗}𝐾𝑗=1 ∈ R𝐾×𝐶. Here, 𝐾 denotes the number of tokens per component, and 𝐶 is the token dimension.

Part-aware Prompting. A straightforward approach to multipart generation is to learn a diffusion network 𝑓 that predicts the latent of a specific part, conditioned on the global context. In this naive baseline, the model takes the form z𝑖 = 𝑓 (zglobal, prompt𝑖),

[Figure 23]

[Figure 24]

[Figure 25]

- 6 • Zhu et al.

#### Input asset: 7 parts PartVerse: 17 parts Ours: 4 parts

A long, slender component of the tank's turret.

A red and black circular component of a futuristic tank-like vehicle.

A long, slender component extracted from the top section of a futuristic tanklike vehicle.

turret and cannon

A close-up of the tank's turret section, isolated from the full tank model.

hull

A component of the futuristic vehicle's landing gear.

side arms

A close-up view of the tank's track system.

### v

tracks

A component of the futuristic vehicle, resembling a rectangular block with a smooth surface.

A cylindrical component of the tank's turret.

Fig. 4. Part Segmentation and Naming Comparison. Same Objaverse asset [Deitke et al. 2023b]. Top left: Original artist decomposition (7 parts). Middle: PartVerse [Dong et al. 2025] (17 parts) with VLM captions that exhibit artifacts (“A close-up of ...”) and lack spatial specificity (e.g., “A red and black circular component of ...”). Right: Our automatic pipeline (4 parts) produces concise, meaningful names (e.g., hull, tracks).

[Figure 26]

[Figure 27]

Table 1. Multi-part Training Data. Composition by source. Source Subsets Content Assets Parts Sketchfab Objaverse, Texverse,

Characters, animals, architecture, etc.

270K 1.14M

PartVerse, PartVerse-XL

Commercial Licensed libraries Furniture, CAD, etc. 64K 201K Internal Game collections Avatars, vehicles, etc. 129K 679K

Training Total 462K 2.02M

Table 2. Dataset Comparison. Scale and annotations vs. prior work. Prior work Assets Parts Open-Voc Part Text

ShapeNetPart [Yi et al. 2016] 16K 93k ✗ Taxonomy labels PartNet [Mo et al. 2018] 26K 573K ✗ Taxonomy labels PartVerse [Dong et al. 2025] 12K 91K ✓ Captions PartVerse-XL [Ding et al. 2025] 40K 320K ✓ Captions

Fig. 5. Set-of-Mark Paired Rendering. Example pair: textured render (left) with part contours and numbered markers, and part-colored render (right). These paired views are input to the VLM for clustering and naming.

Ours 462K 2.02M ✓ Names

- 4 Dataset

Training open-vocabulary part-based 3D generation models requires large-scale datasets with descriptive text labels. Current publicly released part datasets fall short. We gathered a new dataset of approximately 462K assets and 2.02M parts, built using an automated data engine that combines artist-provided segmentations with VLM priors. Using multi-view renders with Set-of-Mark [Yang et al. 2023] overlays, we cluster parts and assign semantic names.

- Table 1 summarizes our training dataset by source, aggregated

from Sketchfab-sourced datasets [Deitke et al.2023b; Ding et al.2025; Dong et al. 2025; Zhang et al. 2025b], and commercial and internal sources. We deduplicate overlapping sources by asset name, restrict to permissively-licensed assets, and exclude assets in PartObjaverseTiny [Yang et al. 2024] to prevent test set contamination.

- 4.1 Multi-Part Dataset Pipeline

Our data engine processes assets through four stages: (1) preprocessing to filter degenerate geometry and retain assets with 2–32 parts; (2) VLM-based quality filtering to remove defective meshes

and scan artifacts; (3) VLM-based part clustering and naming; and (4) postprocessing for watertight mesh conversion and point sampling. Additional details are provided in the supplemental material.

The core of our pipeline is the VLM-based clustering stage. Many 3D assets contain over-segmented parts with inconsistent names. We leverage a VLM to simultaneously cluster related parts and assign semantically meaningful names—for example, merging a car’s separate rim, tire, and hub meshes into front left wheel and front right wheel. This stage operates only on existing part boundaries, without attempting to subdivide parts that are already fused in the source mesh.

Set-of-MarkRendering. InspiredbytheSet-of-Mark(SoM)prompt-

ing technique [Yang et al. 2023], which enables visual grounding in VLMs by overlaying numeric identifiers on image regions, we adapt this approach to 3D part annotation. Our adaptation differs from the original 2D method in these key ways: we apply SoM to multi-view renders of 3D assets where parts are defined by mesh structure rather than image segmentation, and we generate paired images per viewpoint to provide complementary information.

For each asset, we render 14 orbital viewpoints. Each viewpoint produces a textured render with part contours and numbered markers, and a part-colored render where each part appears in a distinct solid color. The textured view provides semantic context for accurate naming—helping distinguish a brick chimney from a stone column—while the part-colored view enables unambiguous part identification. The color of each marker matches the part’s contour and solid color across both views, helping the VLM associate parts (see Figure 5).

VLM-Based Annotation. We query a VLM (GPT-5) with all view pairs simultaneously, leveraging the model’s ability to reason across multiple images. The prompt instructs the model to group parts into semantic clusters based on function or logical relationships, and assign each cluster a concise, descriptive name. Parts may remain in singleton clusters when they already represent coherent semantic units. The VLM returns structured JSON output containing cluster names and constituent parts; we resolve edge cases such as duplicate assignments programmatically.

Postprocessing. After clustering, we convert each part to a watertight mesh using Dual Marching Cubes on a 5123 unsigned distance field, then sample surface points with normals for training. More details about the dataset pipeline are in the appendix.

- 4.2 Comparison

- Table 2 compares our dataset to existing 3D part datasets. The most closely related efforts are PartVerse [Dong et al. 2025] and PartVerseXL [Ding et al. 2025]. Our dataset differs in three main aspects:

Scale. With 462K assets and 2.02M parts, our dataset is over 11× larger than PartVerse-XL (40K assets, 320K parts), enabling broader coverage of open-vocabulary part semantics.

Automation. PartVerse relies on human annotators to correct segmentations, limiting scalability. Our pipeline is fully automatic, using a VLM with Set-of-Mark prompting to cluster parts into functionally meaningful groups (e.g., tracks, front axle). This enables scaling to hundreds of thousands of assets without manual intervention (see Figure 4).

Part naming vs. captioning. Our pipeline produces concise part names rather than descriptive captions. Concise names are more likely to match the part descriptions users provide when querying the model. Caption-based approaches like PartVerse often exhibit VLM artifacts (e.g., “A close-up of ...”) and assign identical captions to identical parts, providing no way to distinguish them. Our pipeline generates concise names and appends positional adjectives only when needed (e.g., left arm, right arm).

- 5 Evaluations

We evaluate both stages of our pipeline, assessing their ability to generate high-quality meshes aligned with the input part schema.

- 5.1 Single-part Mesh Generation

To verify the effectiveness of our proposed schema-aware finetuning, we conduct an ablation study. As illustrated in Figure 7, the pre-trained single-mesh model suffers from missing or incorrectly

Table 3. Evaluation on Part-based Multi-mesh Generation. We evaluate the multi-mesh results using CD and F-scores for both individual parts and holistic shapes (formed by concatenating parts). Our method demonstrates consistent improvements in structural completeness and part-level accuracy.

Part-Level Holistic-Level CD ↓ F-score ↑ CD ↓ F-score ↑

PartObjaverse-Tiny

PartCrafter 0.493 0.290 0.272 0.552 PartPacker 0.374 0.475 0.164 0.792

PatchAlign3D + HoloPart 0.309 0.549 0.050 0.970 SAM3 + OmniPart 0.309 0.630 0.053 0.970

Ours w/o pre-training 0.287 0.625 0.051 0.970 Ours w/o Cross-Part Attention 0.433 0.398 0.148 0.792 Ours w/ PartCrafter-style attention 0.386 0.529 0.089 0.864 Ours 0.251 0.743 0.048 0.974

emphasized parts. With our proposed schema-aware fine-tuning, our method resolves these issues, ensuring all requested parts are present and correctly generated.

Crucially, the Stage 1 single-part mesh generation also serves as pre-training for the Stage 2 multi-part generation, which leads to significant improvement of final output quality both at the part level and holistic level (as shown in Table 3). By starting from a model that already understands the global geometric prior of a "jellyfish car" or "drone," the second stage can dedicate its capacity to learning interpart boundaries and part-specific geometries rather than learning basic 3D structure from scratch.

5.2 Multi-part Mesh Generation

Baselines. We compareStage2ofour methodwith HoloPart [Yang

et al. 2025a], OmniPart [Yang et al. 2025c], PartCrafter [Lin et al. 2025], and PartPacker [Tang et al. 2025]. A critical distinction among these methods is their capacity for “Controllable Part Generation,” which refers to methods capable of explicitly specifying both the number and the semantic identities of parts. Note that unlike our method, these baselines cannot directly take a text part schema as input. PartCrafter and PartPacker fall short of this definition as they can only control the number of generated parts. HoloPart requires a segmented 3D shape, so we provide it with a 3D segmentation produced by PatchAlign3D [Hadgi et al. 2026], given the ground-truth shape and part schema as input. OmniPart takes a 2D segmentation map as input, for which we provide a text-conditioned segmentation produced by SAM3 [Carion et al. 2025], and we additionally initialize OmniPart with ground-truth voxels. However, because its resulting 3D parts are not guaranteed to align with this condition in either part quantity or semantic identity, we treat it as a non-controllable baseline, similar to PartCrafter and PartPacker.

Comparisons. We use PartObjaverse-Tiny as our evaluation dataset. We employ two geometric metrics to measure shape quality: Chamfer Distance (CD) and F-score. We report these metrics at both part level and holistic (whole mesh) level. All shapes are first normalized to [−1, 1] unit box at holistic level, and F-score is calculated with 0.1 threshold. For part-level evaluation, we compute CD and F-score for each independent part mesh and average the scores across all parts of an object. Note that baselines except HoloPart

Controllable Part Generation Non-Controllable Part Generation

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Ground-Truth Ours PatchAlign3D

SAM3 + OmniPart

Input Schema

PartCrafter PartPacker

+ HoloPart

Door Earth Roof

Roof Frames Steps Wall Window

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Body Hair Head

Leg Tail

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

Body Collar Foot

Hand Head Horn

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

Body Flowerpot Hair Head Leaf

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

Bottom Of Pot Kettle Body Kettle Handle Pot Lid

- Fig. 6. Qualitative Comparison of Multi-part Mesh Generation. We evaluate our method against the two-stage baseline “PatchAlign3D [Hadgi et al.

2026] + HoloPart [Yang et al. 2025a]” and other image-based part generation methods. Note that both our method and the “PatchAlign3D + HoloPart” pipeline condition on a mesh and part schema, whereas the other baselines condition on a single image (with OmniPart [Yang et al. 2025c] additionally initialized with ground-truth voxels and requiring a text-conditioned 2D segmentation from SAM3 [Carion et al. 2025]). Under the mesh-conditioned setting, our method outperforms HoloPart in both schema adherence and geometric fidelity. The image-conditioned baselines (OmniPart [Yang et al. 2025c], PartCrafter [Lin et al. 2025], PartPacker [Tang et al. 2025]) fail to offer user-defined part control and produce noisier segmentation boundaries than our approach.

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

A taxi cab. This object contains the following parts: Car body, Wheels, Exhaust Pipe.

Round pedestal bowl holds ripe glossy apples inside. This object contains the following parts: Apple, Apple Stem, Tray.

Off-road buggy with large wheels and futuristic frame design. This object contains the following parts: Car Body, Wheels, Steering Wheel, Bumper, Gun, Lights.

Input Prompt

Generated Single Mesh

Missing “Steering wheel”

- Fig. 7. Ablation Study of Schema-aware Fine-Tuning. In each pair, the left mesh shows generation results without schema-aware fine-tuning, and the right mesh shows our results. Without fine-tuning, the model fails to include all schema parts (e.g., missing “Steering Wheel”) or incorrectly emphasizes others (e.g., “Exhaust Pipe”). Our fine-tuning ensures all requested parts are present and correctly generated.

[Figure 64]

[Figure 65]

[Figure 66]

CubePart : An Open-Vocabulary Part-Controllable 3D Generator • 9

###### Input Prompt & Schema Output

###### Input Prompt & Schema Output Input Prompt & Schema Output

A dwarven steam-powered drilling machine with a massive rotating drill bit at the front.

A fantasy cottage hut perched on giant mechanical chicken legs.

A heavily armored futuristic tank designed to resemble a charging rhinoceros.

Armored Hull Steam Smokestacks cockpit Grate large rotating Drill Bit side Lamps treaded Tracks

Armor Plates Caterpillar Tracks Exhaust Stacks Main Turret Horn Rhino Hull Body Side Cannons

Chimney Clawed Feet Hut Body Mechanical Chicken Legs Porch Thatch Roof

- Fig. 8. Qualitative Results of Two-Stage Generation. We present examples generated by our full pipeline. Conditioned on a text prompt and part schema, our method synthesizes detailed global shapes and decomposes them into independent, structurally complete part meshes that adhere to the defined schema.

[Figure 67]

[Figure 68]

Input 2 Parts 4 Parts 8 Parts

[Figure 69]

[Figure 70]

Body, Wheels, Mirrors, Stands

Controls, Wheels, Mirrors, Stands, Seat, Engine, Frame, Headlights

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

Body, Wheels, Gun, Fenders

Chassis, Wheels, Gun, Tails, Fenders, Bumper, Windows, Suspension

Body, Wheels

Body, Wheels

- Fig. 9. Qualitative results with varying part schema. We test our model with different number of parts on the same input assets. Our method can control generation parts accurately with small components like motorcycle stands and ambiguous closely connected components like chassis and windows. With 2 parts, fenders merge into wheels; with 4 parts, explicit “fenders” resolves this ambiguity.

part-level accuracy, highlighting the critical role of inter-part communication for resolving geometric boundaries. Furthermore, attempting to achieve this communication by modifying pre-trained local attention layers ("Ours w/ PartCrafter-style attention") following PartCrafter [Lin et al. 2025]’s style disrupts the model’s learned priors, resulting in noticeably worse performance than our dedicated zero-initialized blocks. We also observe that omitting Stage 1 pre-training reduces overall structural completeness, confirming its value in establishing strong global geometric priors before part decomposition. We also show our end-to-end two-stage pipeline results in Figure 8.

6 Application: Generating 3D Objects with Behaviors

A primary motivation for schema conditioning is to enable behaviordriven generation of 3D objects, where part structures are designed to be compatible with scripted behaviors in interactive 3D environments. In our workflow, all experiments are conducted in a large gaming platform, where object behaviors are implemented as Lua scripts that directly control individual object parts. We show behaviors in Figure 10 and the demo video, and provide more details about the behavior script pipeline in the appendix.

Driving. We use the jellyfish car in Figure 10 to illustrate how part segmentation supports different vehicle behaviors. For basic driving functionality, we specified 5 parts: body, front left wheel, front right wheel, rear left wheel, rear right wheel. To add headlights and exhaust effects, we refined the schema to separate jellyfish body, headlights, exhaust pipe in addition to the four wheels. Finally, adding a shooting behavior required further segmentation of a gun part, along with scripts for bullet generation and visual effects.

generate an unordered set of parts, requiring us to match the output parts with the inputs for part-level evaluation. To do this matching, we use a greedy approach: iterate through the ground truth part prompts, matching each one with the best-scoring output part.

As shown in Table 3, our method outperforms all baselines consistently across both holistic and part-level metrics. Qualitative results in Figure 6 further illustrate our method’s advantage in decomposing monolithic meshes into parts according to the given schema. Furthermore, Figure 9 highlights that by varying the input schema for a single mesh, our method can accurately modulate both the semantic identity and granularity of the generated parts.

Characters. We evaluate our approach on a diverse set of character types with varying structural and behavioral complexity. The robot character in Figure 10, where independently labeled parts such as the head, arms, and weapons enable behaviors such as arm extension and powered takeoff. The humanoid frog character with articulated limbs and sensory parts support behaviors such as laser emission and single-leg spinning motions. Finally, for the wizard character with external props (e.g., a magic staff), separating the character body from the props allows coordinated motions and visual effects, such as staff swinging and illumination.

Finally, we validate our architectural design choices through an ablation study. Removing the cross-part attention mechanism entirely ("Ours w/o Cross-Part Attention") leads to a severe drop in

[Figure 75]

[Figure 76]

##### Schema Input Output Behaviors

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

Body Exhaust Pipe Front Left Wheel Front Right Wheel Gun Headlights Rear Left Wheel RearRightWheel Gun Shooting​

[Figure 83]

[Figure 84]

Driving & Steering Wheels

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Head Module

Torso Body Back Tanks Left Arm

Right Arm Left Leg Right Leg

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

Arm Extension Powered Takeoff

[Figure 96]

[Figure 97]

[Figure 98]

Left Eyeball Right Eyeball Torso Body Front Sensors Left Arm Right Arm Left Leg Right Leg

[Figure 99]

[Figure 100]

Single-Leg Spin 360°

Laser Emission

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Body Left Blade Right Blade Landing Legs Claws Top Battery Box Round Button

[Figure 107]

[Figure 108]

Drone Takeoff Drone Hover Right & Back

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

Head Feather Hat Torso Body Hands Cloak Orb Magic Staff

Swing Staff

Illuminate Staff

[Figure 115]

[Figure 116]

- Fig. 10. Application: Applying object behaviors to generated 3D objects. Given an input schema and single-part meshes, our Stage 2 model decomposes the input meshes into parts following the specified schema. We then apply object behaviors, including dynamic motions and visual effects.

###### Input Output

Flying. The drone example in Figure 10 illustrates part-level control to achieve complex flight behaviors. The two propellers are treated as separate parts, allowing for asymmetric actuation for takeoff, hovering, and directional motion. In addition, the drone body is segmented into functional components such as landing gear, body shell, and lights, allowing visual effects (e.g., blinking lights) to be applied to individual parts.

Cabin, Chassis, Right Track, Left Track, Bucket, Arm

Parts Overlapping

[Figure 117]

[Figure 118]

“Left” and “Right” Flip

Missing Details Head,

- 7 Limitations and Future Work

Torso, Left Arm, Right Arm, Right Leg,

Parts Overlapping

While CubePart represents a significant step towards part-based control of 3D generations to produce game-ready assets, several technical challenges remain.

Left Leg, Weapon, Wings

Deformable parts. Currently, our model focuses on rigid-body decomposition. While ideal for vehicles, robots, and environmental props, it does not yet support the "skinned" vertex weights required for organic character mesh deformation. Future work could involve predicting skeletal rig weights alongside part geometry.

Missing Lower Torso Part

“Left” and “Right” Flip

Fig. 11. Failure examples. Here we show a few typical failure cases where parts can overlap at contact points. The model sometimes can misunderstand spatial relationships as "Left" and "Right", and occasionally drop input components when the input geometry is complicated.

Geometric interpenetration. Although our cross-part attention mechanism significantly reduces overlaps, the model can still produce parts that intersect at the boundaries even if the input schema dictates disjoint parts. See Figure 11 for failure cases.

Spatial and Positional Referencing. A core difficulty in openvocabulary part generation is the consistent handling of relative spatial identifiers, such as "front-left" versus "rear-right". While our dataset introduces spatially-aware naming, it inherits the inherent ambiguities of VLM-based labeling. VLMs occasionally struggle with

"mirroring" errors, confuse the object’s local coordinate system with the camera’s view-space, or fail to label occluded parts. This noise in the training data can lead the generative model to occasionally swap symmetrical parts or misplace components along a specific axis (Figure 11). Also see the appendix for details.

Acknowledgments

We thank the leadership, Nishchaie Khanna, Karun Channa, Anupam Singh, and David Baszucki, for their support and guidance throughoutthis work. We alsothank Michael Palleschi, Maurice Chu, Keenan Crane, and Kayvon Fatahalian for helpful discussions. We are grateful to Zhenyu Zhao, Daniel Chin, Michael Spedden, Alvin Chan, and Saurav Dhakad for setting up the evaluation pipeline

- as part of the broader project. Finally, we are thankful to the MLPlatform team, Anying Li, Yiqing Wang, Steve Han, Sourashis Roy, Chengyi Nie, Wei Zeng, Sal Pathare, Mandar Deshpande, and Andy Shen, for their contributions and collaboration that helped make this project possible.

References

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond. arXiv:2308.12966 [cs.CV] https://arxiv.org/abs/2308.12966

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, Francois Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and Christoph Feichtenhofer. 2025. SAM 3: Segment Anything with Concepts. arXiv:2511.16719 [cs.CV] https://arxiv.org/abs/2511.16719

Angel X. Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, Jianxiong Xiao, Li Yi, and Fisher Yu. 2015. ShapeNet: An Information-Rich 3D Model Repository. arXiv:1512.03012 [cs.GR] https://arxiv.org/abs/1512.03012

Minghao Chen, Roman Shapovalov, Iro Laina, Tom Monnier, Jianyuan Wang, David Novotny, and Andrea Vedaldi. 2025a. PartGen: Part-level 3D Generation and Reconstruction with Multi-view Diffusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 5881–5892.

Minghao Chen, Jianyuan Wang, Roman Shapovalov, Tom Monnier, Hyunyoung Jung, Dilin Wang, Rakesh Ranjan, Iro Laina, and Andrea Vedaldi. 2025b. AutoPartGen: Autogressive 3D Part Generation and Discovery. In Proceedings of the 39th International Conference on Neural Information Processing Systems.

Yongwei Chen, Tengfei Wang, Tong Wu, Xingang Pan, Kui Jia, and Ziwei Liu. 2024. ComboVerse: Compositional 3D Assets Creation Using Spatially-Aware Diffusion Guidance. arXiv:2403.12409 [cs.CV] https://arxiv.org/abs/2403.12409

Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu, Xi Zhang, Tomas F. Yago Vicente, Thomas Dideriksen, Himanshu Arora, Matthieu Guillaumin, and Jitendra Malik. 2022. ABO: Dataset and Benchmarks for Real-World 3D Object Understanding. arXiv:2110.06199 [cs.CV] https://arxiv.org/ abs/2110.06199

Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, and Ali Farhadi. 2023a. Objaverse-XL: A Universe of 10M+ 3D Objects. arXiv:2307.05663 [cs.CV] https://arxiv.org/abs/2307.05663

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. 2023b. Objaverse: A Universe of Annotated 3D Objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 13142–13153.

Lihe Ding, Shaocong Dong, Yaokun Li, Chenjian Gao, Xiao Chen, Rui Han, Yihao Kuang, Hong Zhang, Bo Huang, Zhanpeng Huang, Zibin Wang, Dan Xu, and Tianfan Xue. 2025. FullPart: Generating each 3D Part at Full Resolution. arXiv:2510.26140 [cs.CV] https://arxiv.org/abs/2510.26140

Shaocong Dong, Lihe Ding, Xiao Chen, Yaokun Li, Yuxin Wang, Yucheng Wang, Qi Wang, Jaehyeok Kim, Chenjian Gao, Zhanpeng Huang, Zibin Wang, Tianfan Xue, and Dan Xu. 2025. From One to More: Contextual Part Latents for 3D Generation. arXiv:2507.08772 [cs.CV] https://arxiv.org/abs/2507.08772

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, and Robin Rombach. 2024. Scaling rectified flow transformers for high-resolution image synthesis. In Proceedings of the 41st International Conference on Machine Learning (Vienna, Austria) (ICML’24). JMLR.org, Article 503, 28 pages.

Jun Gao, Tianchang Shen, Zian Wang, Wenzheng Chen, Kangxue Yin, Daiqing Li, Or Litany, Zan Gojcic, and Sanja Fidler. 2022. GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images. arXiv:2209.11163 [cs.CV] https://arxiv.org/abs/2209.11163

Souhail Hadgi, Bingchen Gong, Ramana Sundararaman, Emery Pierson, Lei Li, Peter Wonka, and Maks Ovsjanikov. 2026. PatchAlign3D: Local Feature Alignment for Dense 3D Shape understanding. arXiv preprint arXiv:2601.02457 (2026).

Xufan He, Yushuang Wu, Xiaoyang Guo, Chongjie Ye, Jiaqing Zhou, Tianlei Hu, Xiaoguang Han, and Dong Du. 2025a. UniPart: Part-Level 3D Generation with Unified 3D Geom-Seg Latents. arXiv:2512.09435 [cs.CV] https://arxiv.org/abs/2512.09435

Xianglong He, Zi-Xin Zou, Chia-Hao Chen, Yuan-Chen Guo, Ding Liang, Chun Yuan, Wanli Ouyang, Yan-Pei Cao, and Yangguang Li. 2025b. SparseFlex: High-Resolution and Arbitrary-Topology 3D Shape Modeling. arXiv:2503.21732 [cs.CV] https://arxiv. org/abs/2503.21732

Amir Hertz, Or Perel, Raja Giryes, Olga Sorkine-Hornung, and Daniel Cohen-Or.

2022. SPAGHETTI: Editing Implicit Shapes Through Part Aware Generation. arXiv:2201.13168 [cs.GR] https://arxiv.org/abs/2201.13168

Ka-Hei Hui, Ruihui Li, Jingyu Hu, and Chi-Wing Fu. 2022. Neural Template: Topology-aware Reconstruction and Disentangled Generation of 3D Meshes. arXiv:2206.04942 [cs.CV] https://arxiv.org/abs/2206.04942

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollar, and Ross Girshick. 2023. Segment Anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). 4015–4026.

Juil Koo, Seungwoo Yoo, Minh Hieu Nguyen, and Minhyuk Sung. 2024. SALAD: Part-Level Latent Diffusion for 3D Shape Generation and Manipulation. arXiv:2303.12236 [cs.CV] https://arxiv.org/abs/2303.12236

Zeqiang Lai, Yunfei Zhao, Zibo Zhao, Haolin Liu, Qingxiang Lin, Jingwei Huang, Chunchao Guo, and Xiangyu Yue. 2025. LATTICE: Democratize High-Fidelity 3D Generation at Scale. arXiv:2512.03052 [cs.GR] https://arxiv.org/abs/2512.03052 Mingxiao Li, Tingyu Qu, Ruicong Yao, Wei Sun, and Marie-Francine Moens. 2024. Alleviating Exposure Bias in Diffusion Models through Sampling with Shifted Time Steps. In International Conference on Learning Representations, B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (Eds.), Vol. 2024. 16816–16838. https://proceedings.iclr.cc/paper_files/paper/2024/file/ 483f8d2018d7025c87dd07e9b02fe4bf-Paper-Conference.pdf

Weiyu Li, Jiarui Liu, Hongyu Yan, Rui Chen, Yixun Liang, Xuelin Chen, Ping Tan, and Xiaoxiao Long. 2025a. CraftsMan3D: High-fidelity Mesh Generation with 3D Native Generation and Interactive Geometry Refiner. arXiv:2405.14979 [cs.GR] https://arxiv.org/abs/2405.14979

Weiyu Li, Xuanyang Zhang, Zheng Sun, Di Qi, Hao Li, Wei Cheng, Weiwei Cai, Shihao Wu, Jiarui Liu, Zihao Wang, Xiao Chen, Feipeng Tian, Jianxiong Pan, Zeming Li, Gang Yu, Xiangyu Zhang, Daxin Jiang, and Ping Tan. 2025c. Step1X3D: Towards High-Fidelity and Controllable Generation of Textured 3D Assets. arXiv:2505.07747 [cs.CV] https://arxiv.org/abs/2505.07747

Yangguang Li, Zi-Xin Zou, Zexiang Liu, Dehu Wang, Yuan Liang, Zhipeng Yu, Xingchao Liu, Yuan-Chen Guo, Ding Liang, Wanli Ouyang, and Yan-Pei Cao. 2025d. TripoSG: High-Fidelity 3D Shape Synthesis using Large-Scale Rectified Flow Models. arXiv:2502.06608 [cs.CV] https://arxiv.org/abs/2502.06608

Zhihao Li, Yufei Wang, Heliang Zheng, Yihao Luo, and Bihan Wen. 2025b. Sparc3D: Sparse Representation and Construction for High-Resolution 3D Shapes Modeling. arXiv:2505.14521 [cs.CV] https://arxiv.org/abs/2505.14521

Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. 2023. Magic3D: HighResolution Text-to-3D Content Creation. arXiv:2211.10440 [cs.CV] https://arxiv. org/abs/2211.10440

Yuchen Lin, Chenguo Lin, Panwang Pan, Honglei Yan, Yiqiang Feng, Yadong Mu, and Katerina Fragkiadaki. 2025. PartCrafter: Structured 3D Mesh Generation via Compositional Latent Diffusion Transformers. In Proceedings of the 39th International Conference on Neural Information Processing Systems.

Anran Liu, Cheng Lin, Yuan Liu, Xiaoxiao Long, Zhiyang Dou, Hao-Xiang Guo, Ping Luo, and Wenping Wang. 2024. Part123: Part-aware 3D Reconstruction from a Single-view Image. In ACM SIGGRAPH 2024 Conference Papers (Denver, CO, USA) (SIGGRAPH ’24). Association for Computing Machinery, New York, NY, USA, Article 24, 12 pages. doi:10.1145/3641519.3657482

Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. 2023b. Zero-1-to-3: Zero-shot One Image to 3D Object. arXiv:2303.11328 [cs.CV] https://arxiv.org/abs/2303.11328

Xingchao Liu, Chengyue Gong, and qiang liu. 2023a. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. In The Eleventh International Conference on Learning Representations. https://openreview.net/forum?id=XVjTT1nw5z

Changfeng Ma, Yang Li, Xinhao Yan, Jiachen Xu, Yunhan Yang, Chunshi Wang, Zibo Zhao, Yanwen Guo, Zhuo Chen, and Chunchao Guo. 2025. P3-SAM: Native 3D Part Segmentation. arXiv:2509.06784 [cs.CV] https://arxiv.org/abs/2509.06784

Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. 2024. Sit: Exploring flow and diffusion-based generative models

with scalable interpolant transformers. In European Conference on Computer Vision. Springer, 23–40.

Kaichun Mo, Shilin Zhu, Angel X. Chang, Li Yi, Subarna Tripathi, Leonidas J. Guibas, and Hao Su. 2018. PartNet: A Large-scale Benchmark for Fine-grained and Hierarchical Part-level 3D Object Understanding. arXiv:1812.02713 [cs.CV] https://arxiv.org/ abs/1812.02713

Kiyohiro Nakayama, Mikaela Angelina Uy, Jiahui Huang, Shi-Min Hu, Ke Li, and Leonidas J Guibas. 2023. DiffFacto: Controllable Part-Based 3D Point Cloud Generation with Cross Diffusion. arXiv:2305.01921 [cs.CV] https://arxiv.org/abs/2305.01921

Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, ShangWen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. 2024. DINOv2: Learning Robust Visual Features without Supervision. Transactions on Machine Learning Research (2024). https://openreview.net/forum?id=a68SUt6zFt Featured Certification.

Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. 2022. DreamFusion: Textto-3D using 2D Diffusion. arXiv:2209.14988 [cs.CV] https://arxiv.org/abs/2209.14988

Xuanchi Ren, Jiahui Huang, Xiaohui Zeng, Ken Museth, Sanja Fidler, and Francis Williams. 2024. XCube: Large-Scale 3D Generative Modeling using Sparse Voxel Hierarchies. arXiv:2312.03806 [cs.CV] https://arxiv.org/abs/2312.03806

Jiaxiang Tang, Ruijie Lu, Zhaoshuo Li, Zekun Hao, Xuan Li, Fangyin Wei, Shuran Song, Gang Zeng, Ming-Yu Liu, and Tsung-Yi Lin. 2025. Efficient Part-level 3D Object Generation via Dual Volume Packing. In Proceedings of the 39th International Conference on Neural Information Processing Systems.

Foundation AI Team, Kiran Bhat, Nishchaie Khanna, Karun Channa, Tinghui Zhou, Yiheng Zhu, Xiaoxia Sun, Charles Shang, Anirudh Sudarshan, Maurice Chu, Daiqing Li, Kangle Deng, Jean-Philippe Fauconnier, Tijmen Verhulsdonck, Maneesh Agrawala, Kayvon Fatahalian, Alexander Weiss, Christian Reiser, Ravi Kiran Chirravuri, Ravali Kandur, Alejandro Pelaez, Akash Garg, Michael Palleschi, Jessica Wang, Skylar Litz, Leon Liu, Anying Li, David Harmon, Derek Liu, Liangjun Feng, Denis Goupil, Lukas Kuczynski, Jihyun Yoon, Naveen Marri, Peiye Zhuang, Yinan Zhang, Brian Yin, Haomiao Jiang, Marcel van Workum, Thomas Lane, Bryce Erickson, Salil Pathare, Kyle Price, Steve Han, Yiqing Wang, Anupam Singh, and David Baszucki. 2025. Cube: A Roblox View of 3D Intelligence. arXiv:2503.15475 [cs.CV] https://arxiv.org/abs/2503.15475

Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. 2023. ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation. arXiv:2305.16213 [cs.LG] https://arxiv.org/abs/2305. 16213

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. 2025a. Qwen-Image Technical Report. arXiv:2508.02324 [cs.CV] https://arxiv.org/abs/2508.02324

Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Yikang Yang, Yajie Bao, Jiachen Qian, Siyu Zhu, Xun Cao, Philip Torr, and Yao Yao. 2025b. Direct3D-S2: Gigascale 3D Generation Made Easy with Spatial Sparse Attention. arXiv:2505.17412 [cs.CV] https://arxiv.org/abs/2505.17412

Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, and Jiaolong Yang. 2025a. Native and Compact Structured Latents for 3D Generation. arXiv:2512.14692 [cs.CV] https://arxiv.org/abs/2512.14692

Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. 2025b. Structured 3D Latents for Scalable and Versatile 3D Generation. arXiv:2412.01506 [cs.CV] https://arxiv.org/abs/2412.01506

Xinhao Yan, Jiachen Xu, Yang Li, Changfeng Ma, Yunhan Yang, Chunshi Wang, Zibo Zhao, Zeqiang Lai, Yunfei Zhao, Zhuo Chen, and Chunchao Guo. 2025. X-Part: high fidelity and structure coherent shape decomposition. arXiv:2509.08643 [cs.GR] https://arxiv.org/abs/2509.08643

Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. 2023. Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V. arXiv:2310.11441 [cs.CV] https://arxiv.org/abs/2310.11441

Shuhui Yang, Mingxin Yang, Yifei Feng, Xin Huang, Sheng Zhang, Zebin He, Di Luo, Haolin Liu, Yunfei Zhao, Qingxiang Lin, Zeqiang Lai, Xianghui Yang, Huiwen Shi, Zibo Zhao, Bowen Zhang, Hongyu Yan, Lifu Wang, Sicong Liu, Jihong Zhang, Meng Chen, Liang Dong, Yiwen Jia, Yulin Cai, Jiaao Yu, Yixuan Tang, Dongyuan Guo, Junlin Yu, Hao Zhang, Zheng Ye, Peng He, Runzhou Wu, Shida Wei, Chao Zhang, Yonghao Tan, Yifu Sun, Lin Niu, Shirui Huang, Bojian Zheng, Shu Liu, Shilin Chen, Xiang Yuan, Xiaofeng Yang, Kai Liu, Jianchen Zhu, Peng Chen, Tian Liu, Di Wang, Yuhong Liu, Linus, Jie Jiang, Jingwei Huang, and Chunchao Guo. 2025b. Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material.

arXiv:2506.15442 [cs.CV] https://arxiv.org/abs/2506.15442

Yunhan Yang, Yuan-Chen Guo, Yukun Huang, Zi-Xin Zou, Zhipeng Yu, Yangguang Li, Yan-Pei Cao, and Xihui Liu. 2025a. HoloPart: Generative 3D Part Amodal Segmentation. arXiv:2504.07943 [cs.CV] https://arxiv.org/abs/2504.07943

Yunhan Yang, Yukun Huang, Yuan-Chen Guo, Liangjun Lu, Xiaoyang Wu, Edmund Y. Lam, Yan-Pei Cao, and Xihui Liu. 2024. SAMPart3D: Segment Any Part in 3D Objects. arXiv:2411.07184 [cs.CV] https://arxiv.org/abs/2411.07184

Yunhan Yang, Yufan Zhou, Yuan-Chen Guo, Zi-Xin Zou, Yukun Huang, Ying-Tian Liu, Hao Xu, Ding Liang, Yan-Pei Cao, and Xihui Liu. 2025c. OmniPart: Part-Aware 3D Generation with Semantic Decoupling and Structural Cohesion. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers (SA Conference Papers ’25). Association for Computing Machinery, New York, NY, USA, Article 59, 12 pages. doi:10.1145/ 3757377.3763872

Li Yi, Vladimir G. Kim, Duygu Ceylan, I-Chao Shen, Mengyan Yan, Hao Su, Cewu Lu, Qixing Huang, Alla Sheffer, and Leonidas Guibas. 2016. A Scalable Active Framework for Region Annotation in 3D Shape Collections. SIGGRAPH Asia (2016).

Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 2023. 3DShape2VecSet: A 3D Shape Representation for Neural Fields and Generative Diffusion Models. arXiv:2301.11445 [cs.CV] https://arxiv.org/abs/2301.11445

Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. 2024. CLAY: A Controllable Large-scale Generative Model for Creating High-quality 3D Assets. arXiv:2406.13897 [cs.CV] https://arxiv. org/abs/2406.13897

Longwen Zhang, Qixuan Zhang, Haoran Jiang, Yinuo Bai, Wei Yang, Lan Xu, and Jingyi Yu. 2025a. BANG: Dividing 3D Assets via Generative Exploded Dynamics. ACM Transactions on Graphics 44, 4 (July 2025), 1–21. doi:10.1145/3730840

Yibo Zhang, Li Zhang, Rui Ma, and Nan Cao. 2025b. TexVerse: A Universe of 3D Objects with High-Resolution Textures. arXiv:2508.10868 [cs.CV] https://arxiv.org/abs/ 2508.10868

Zibo Zhao, Wen Liu, Xin Chen, Xianfang Zeng, Rui Wang, Pei Cheng, Bin Fu, Tao Chen, Gang Yu, and Shenghua Gao. 2023. Michelangelo: Conditional 3D Shape Generation based on Shape-Image-Text Aligned Latent Representation. arXiv:2306.17115 [cs.CV] https://arxiv.org/abs/2306.17115

Appendix

In the appendix, we provide more details about our dataset pipeline, additional visual results in Figure 14, and animated results in our attached video.

- A Dataset Pipeline Details

This section provides detailed descriptions of each stage in our multipart dataset pipeline, supplementing the overview in the main paper.

- A.1 Data Sources

As described in the main paper, our training dataset aggregates assets from multiple sources. The Sketchfab sources (Objaverse, Texverse, PartVerse, PartVerse-XL) overlap significantly; we deduplicate by asset name, prioritizing human-corrected segmentations (PartVerse family) over raw artist segmentations, restrict to permissivelylicensed assets, and exclude assets in PartObjaverse-Tiny [Yang et al. 2024] to prevent test set contamination. Commercial and internal sources contribute additional diversity in furniture, architecture, CAD models, and game assets.

- A.2 Preprocessing Stage

As the first stage of our data engine, we apply preprocessing steps to prepare assets for annotation. We first remove empty or degenerate geometry parts, then retain only assets containing between 2 and 32 parts. We exclude single-part assets because clustering requires at least two parts, and exclude overly complex assets to maximize the VLM’s success rate.

- A.3 VLM-Based Filtering Stage

Before clustering, we filter assets by quality using a VLM. Each asset is rendered from multiple viewpoints (8 in our configuration) for the model to assess quality. The model identifies mesh defects (tearing, fragmentation), scan artifacts (irregular, noisy surfaces), scene-level content (room sections, object collections, cutaway views), and problematic geometry (zero-volume meshes, overly thin structures). It also outputs complexity scores for geometry and texture, along with a brief description for asset indexing. Based on the identified tags, the VLM classifies each asset into three quality tiers: poor, moderate, or excellent, prioritizing geometric complexity over texture quality and scoring conservatively when borderline. Only moderate and excellent assets proceed to clustering.

- A.4 VLM-Based Clustering and Naming Stage

This section describes the core annotation stage of our pipeline, which uses a VLM to simultaneously cluster related parts and assign semantically meaningful names.

- A.4.1 Set-of-Mark Rendering. Inspired by the Set-of-Mark (SoM) prompting technique [Yang et al. 2023], which enables visual grounding in VLMs by overlaying numeric identifiers on image regions, we adapt this approach to 3D part annotation. While the original SoM method partitions 2D images using segmentation models and overlays marks for tasks such as referring segmentation or phrase grounding, our adaptation differs in three key ways: (1) we apply SoM to multi-view renders of 3D assets, where parts are defined by

the mesh structure rather than image segmentation; (2) we generate paired images per viewpoint—one textured and one colored per part—to provide complementary information; and (3) since our part masks are derived directly from the 3D geometry and are guaranteed non-overlapping, we place each mark independently at the point with maximum distance to the part boundary, simplifying placement compared to the original approach.

For each asset, we render the scene from 14 orbital viewpoints to ensure comprehensive part visibility. For each viewpoint, we generate a pair of square-proportioned images:

- • A textured render showing the asset with its original textures, overlaid with part contours and numbered markers (numeric identifiers enclosed in a colored circle).
- • A part-colored render where each part is rendered in a distinct solid color, with the same numbered markers overlaid.

The paired representation provides complementary information: the textured view supplies semantic context essential for accurate naming—for instance, helping distinguish a brick chimney from a stone column, or a tiled roof from a shingled roof —while the partcolored view provides unambiguous part segmentation masks that facilitate precise identification and clustering. We found this pairing beneficial through experimentation: using only textured renders resulted in poor clustering accuracy, as parts were harder to isolate visually, while using only part-colored renders led to degraded naming quality due to lost semantic context. The paired approach combines the strengths of both.

Crucially, the color of each numbered marker matches both the part’s solid color in the part-colored render and its contour color in the textured render, ensuring visual coherence across both images. This consistent color coding helps the VLM associate parts across the paired views.

A.4.2 VLM Annotation. We query a VLM with all view pairs simultaneously, leveraging the model’s ability to reason across multiple images using visual tokens. The prompt instructs the model to: (1) group parts into semantic clusters based on function or logical relationships, rather than visual similarity or spatial proximity alone; and (2) assign each cluster a concise, descriptive name.

The prompt explicitly allows for identity clustering, where a part may remain in its own singleton cluster if it already represents a coherent semantic unit—in which case the VLM effectively provides a name for that individual part.

This stage operates only on the existing part boundaries; it cannot subdivide parts that are already fused in the source mesh. If an asset contains under-segmented parts (e.g., an avatar with a separate head part but a single body part that combines the limbs and torso), the VLM assigns the most descriptive name possible given the combined geometry—it cannot name the arms, legs, and torso individually.

The VLM is instructed to return a structured JSON output containing cluster names and their constituent parts, referenced by the numeric identifiers shown in the Set-of-Mark renders. We handle edge cases through post-processing: duplicate assignments are resolved by keeping each part in its first-assigned cluster, and in rare cases where the VLM groups all parts into a single clustereffectively collapsing the part structure—the asset is filtered out.

Although source assets may contain existing part names or hierarchical structure, we do not incorporate this metadata into the VLM prompt. Existing names are often noisy and could mislead the model; omitting them also keeps the mechanism general across diverse asset sources without per-dataset adaptation.

We compared GPT-4o and GPT-5 and found that GPT-5 produced more accurate clusters with finer granularity when needed, and more consistent naming conventions, particularly for complex assets with many parts.

- A.5 Postprocessing Stage

After clustering and naming, we prepare each asset for training. Artist-created meshes are often non-watertight—containing open surfaces, self-intersections, or non-manifold edges—which prevents reliable inside/outside queries needed for occupancy supervision. We convert each part to a watertight mesh by computing an unsigned distance field on a 5123 grid and extracting a level set using Dual Marching Cubes. We then sample 128K visible surface points with normals from each part and 128K visible surface points with normals from the full mesh. Finally, we generate pre-shuffled training and validation epochs, subsampling surface points using Farthest Point Sampling (FPS) to ensure uniform coverage. We apply the same postprocessing for the single-mesh training data.

- A.6 Dataset Limitations

Our approach has several limitations. Parts that are not visible in any rendered view—either because they are too small or fully occludedcannot be identified by the VLM and are assigned a special label indicating they remain unlabeled. For under-segmented parts, the assigned name may describe only the most visually prominent component rather than the whole. Positional adjectives such as “left” and “right” are occasionally confused, and some assets exhibit naming inconsistencies. These imperfections are acceptable trade-offs for a fully automatic pipeline that scales to hundreds of thousands of assets. When higher annotation quality is required, a humanin-the-loop review stage can be applied to filter or correct noisy annotations.

- B VLM Prompts

We provide the complete prompts used in our VLM-based pipeline stages, as described in the main paper and in Section A of this document.

- B.1 VLM Prompt for Quality Filtering

The filtering stage uses a VLM to assess asset quality before clustering. The prompt instructs the model to identify visual flaws from a fixed tag vocabulary, assess geometric and texture complexity, and assign an overall quality score. Following the text prompt, we append each of the 8 rendered views with its name. The complete message structure sent to the VLM is the following:

You are a visual quality inspector for 3D assets. The images are displaying distinct views of a textured mesh. Your task is to examine the 3D asset shown in the multi-view rendering and determine whether it exhibits any of the following visual flaws

or characteristics. Only include tags from the following list. Do not invent or infer new tags, even if they seem reasonable. If no tags apply, return an empty list. Do not summarize. Return only the matching tags as a JSON array. Here are the tags you may apply (you may select more than one): ### Tag Vocabulary:

- - mesh tearing — there is tearing in the mesh, missing polygons, or visible gaps in the geometry that are semantically incorrect (possibly a scanned 3d object)
- - 3d scan — mesh geometry appears to be from a 3D scan with irregular, noisy surfaces lacking clean geometric lines, precise edges, or mathematical precision typical of modeled assets
- - cutaway view — the mesh appears to be sliced open or missing walls, exposing the interior (e.g. walls of a house removed to show room interiors, section of a ship's hull cut away to display internal compartments, etc). However, natural openings like doorways, windows, or opening that are part of an object's intended design are NOT cutaway view (e.g. the visible interiors of a car from its open windows, the visible cabin of a ships from its doorway, etc.). Those are not cutaway views.
- - fragmented object — there is an object composed of two or more disconnected pieces
- - multiple objects — the image contains more than one distinct object, not a single unified mesh
- - collection of objects — the mesh represents a grouped collection or set of related items (e.g., a toolkit, dinnerware set, or cluster of similar objects)
- - mini-scene-like — the mesh is an indoor room or part of an environment, not a standalone object
- - room section — the mesh represents architectural elements or room components like wall panels, prefab sections, or modular building parts
- - overly complex plants/foliage — the mesh contains intricate plant matter, leaves, branches, or botanical elements with high geometric complexity
- - overly thin structures — the mesh contains structures that are extremely thin, wire-like, or have minimal thickness that may cause rendering or processing issues
- - no recognizable object — there is no clear or identifiable shape
- - heavily occluded views — most of the viewing angles are unable to see semantically meaningful parts of the object (e.g. a stairwell surrounded by walls on three sides)
- - zero volume mesh — The object is truly two-dimensional and has

**absolutely no measurable thickness** in 3D space. It is a flat surface or an open sheet. Examples: a single polygon card, poster, paper sheet, or an unclosed terrain slice. Exclude all solid objects like swords, knives, rulers, tablet computers, or railroad tracks, which could plausibly exist as a free-standing objects in 3D space. Thin solids (phones, tracks, tablets) are NOT zero volume meshes. Only flat, one-sided polygons qualify.

- - has baseplate — the object is situated on top of a base plate (e.g. a thin platform, rocky formation, cutout piece of ground or turf)
- - empty image — all views are completely or approximately blank. There is little or no visible content due to rendering failure, bad normalization or missing geometry

### Output format (JSON): Return a list of all applicable tags.

Return a complexity score for the mesh geometry:

- - poor - the asset has very simplistic geometry, with an over-smoothed or blocky characteristic
- - moderate - the asset has a normal level of geometric complexity
- - high - the asset's geometry is extremely detailed, with high frequency bumps and grooves

Return a complexity score for the texture:

- poor - the asset has a very simplistic texture pattern, uses a

very simplistic color pallet, or is generally low resolution

- - moderate - the asset has a normal level of textural complexity
- - high - the asset's texture is extremely detailed, with high frequency color patterns

Return an overall quality score for the mesh. Prioritize geometric complexity and interesting standalone objects over texture quality when assigning scores. Take into account the use case when assigning a quality score. If an asset's quality is borderline between two scores, be conservative and assign the lower score.

- - poor — asset is of low quality (3d scan, mesh tearing, not recognizable, fragmented) or asset that is a prefab scene assets (room section, mini-scene-like, multiple distinct objects forming a collection like a car next to another car or two axes - we care about intra part object, not scene level) that should likely be removed from dataset before training. Assets tagged as '3d scan', 'multiple objects', 'cutaway view', 'room section', 'mini-scene-like', 'overly complex plants/foliage', etc. are of poor quality and should be tagged as 'poor'.
- - moderate — asset represents an interesting standalone object, probably good enough for pretraining a large model. Simplistic geometry, simple texture, simplistic design, blocky, low-poly, low-geometric details are still going in the 'moderate', as long as the asset is recognizable and not broken/categorized as 'poor'. For instance, a low-poly humanoid character is acceptable and should be tagged as 'moderate'. As rule-of-thumb, no tags are commonly associated with assets scored as 'moderate'.
- - excellent — asset has high geometric complexity, represents a high-quality interesting standalone object, and can be used in a small golden training set.

Return a short text description of the textured mesh. This will be used for searching the assets, not for training a model. If the image is empty or ambiguous, the description can simply be "Empty" or "Unknown".

The output should be in json format, specifying the list of "tags", the "geometric complexity", the "texture complexity", the "reasoning" used to determine the overall quality of the asset, the quality "score", and a brief "description" of the asset.

Here is an example output: {

"tags": [ "mesh tearing", "heavily occluded views", "scene-like",

], "geometric complexity": "high", "texture complexity": "moderate", "reasoning": "The asset represents a rocky terrain with dense

forest. There are clear gaps in the mesh where there should be terrain, indicating the mesh is incomplete with tears. Many areas of the terrain are not visible from any of the provided views because of the density of the trees. While the asset has very detailed geometry seen in the foliage and rocky portions, and moderately detailed texturing, its scene-like nature and incomplete mesh indicate limited utility for single-object 3D training. Therefore, it deserves a "poor" quality score, and should probably not be included in the training set.

"score": "poor", "description": "An outdoor scene with trees, rocks and dirt.",

} front_tilt: [IMAGE: textured render]

front: [IMAGE: textured render]

right_tilt: [IMAGE: textured render]

... (repeated for all 8 views)

B.2 VLM Prompt for Part Clustering and Naming

The clustering stage uses a VLM to group related parts and assign semantic names. The prompt is structured into three components:

- (1) a system context that describes the paired image input format,
- (2) task instructions including internal reasoning guidelines and clustering rules, and (3) an output format specification with fewshot examples demonstrating both grouping and identity clustering scenarios.

Following the text prompt, we append each view with its name and the corresponding image pair (textured render followed by partcolored render). The complete message structure sent to the VLM is the following:

You are an expert in 3D asset analysis, specializing in part identification and semantic grouping. You will receive pairs of images for a 3D asset. The first image in each pair shows the asset with its original textures and overlays with numbers for each part and a contour. The second image shows the same view, but with numeric overlays on each part and part contours drawn in a single color. This second image is designed to help you identify and isolate specific parts.

Your task is to group all visible part IDs into high-level semantic clusters based on function, assembly, or logical relationship.

## Internal Reasoning Guidelines To create accurate clusters, you must first mentally identify each part. Follow these rules in your reasoning:

- - Identify parts with concise, singular, engineering-style names (e.g., "wheel", "upper arm", "rear bumper").
- - Take some perspective: IDs are centered and can represent a whole body / object.
- - Prefer the most specific commonly used term visible.

## Clustering Rules (Final Output)

- - Create logical, high-level groups. Good clusters represent functional systems (e.g., "propulsion", "suspension") or major assemblies (e.g., "front axle", "front lights").
- - Do not cluster parts solely based on visual similarity or proximity; focus on semantic relationships.
- - Do not cluster parts far away that have no logical connection.
- - Cluster names should be descriptive, often plural or collective.
- - Every visible part ID MUST be included in exactly one of the semantic_clusters.
- - **Identity Clustering:** If the individual parts *already* represent the most logical semantic grouping (e.g., each part is a distinct, high-level component like "engine", "gearbox", "chassis"), then returning each part as its own cluster is the correct and preferred output. Do not force illogical merges.
- - Ensure the clusters are holistic and cover all the identified parts.
- - More complex objects with many parts may require more clusters; simpler objects may need fewer.
- - When using position adjectives (e.g., "front", "rear", "left", "right"), ensure they are accurate based on the object perspective shown in the images and not from the viewer's perspective.
- - Be concise in your cluster naming; avoid unnecessary words. If a single word suffices, use it. Avoid using descriptive phrases unless absolutely necessary for clarity. Use adjectives just to distinguish between similar parts like by location (e.g., "left door" vs. "right door"), but do not add extra descriptive terms. Avoid using the whole 3D asset name in the cluster names.

## Output Format Respond with a single JSON object ONLY. The object must contain a single key: "semantic_clusters".

- ### Example 1: Grouping components {

"semantic_clusters": [ {

"cluster_name": "wheels", "part_ids": [1, 2, 3, 4]

}, {

"cluster_name": "engine", "part_ids": [5, 6]

} ]

}

- ### Example 2: Identity clustering (already well-grouped) {

"semantic_clusters": [ {

"cluster_name": "front left wheel",

- "part_ids": [1]

}, {

"cluster_name": "front right wheel",

- "part_ids": [2]

}, {

"cluster_name": "engine block",

- "part_ids": [3]

}, {

"cluster_name": "chassis",

- "part_ids": [4]

} ]

} front: [IMAGE: textured render with contours and numbered markers] [IMAGE: part-colored render with numbered markers] left: [IMAGE: textured render with contours and numbered markers] [IMAGE: part-colored render with numbered markers] left_tilt_top: [IMAGE: textured render with contours and numbered markers] [IMAGE: part-colored render with numbered markers]

... (repeated for all 14 orbital views)

- C Data Distribution

Figure 12 shows the distribution of parts per asset across our 462K training assets. Parts per asset follow a right-skewed distribution: 30% of assets have exactly 2 parts, 45% fall in the 3–5 range, 20% in 6–10, and only 4% exceed 10 parts (max 32).

- D Behavior Script pipeline

Using a generated drone as an example, we show the skeleton of a behavior script in Figure 13, following four core stages:

- (1) Welding: Assembles modular parts into a cohesive rigid body.
- (2) Rigging: Configures hinge constraints to define pivot points, e.g., propeller axes.

[Figure 119]

- Fig. 12. Distribution of parts per asset across the 462K training assets (Sketchfab, commercial, and internal sources combined). The majority of assets fall in the 2-10 parts range.

|local function makeDroneFunctional(root , parts , config<br><br>)<br><br>local body , blades , legs , button = classifyParts(<br><br>parts)<br><br>-- Stage 1: Welding -- assemble rigid body weldPartsTogether(root , body , legs , button)<br><br>-- Stage 2: Rigging -- hinge constraints for propeller axes<br><br>for _, blade in ipairs(blades) do<br><br>attachHinge(body , blade) end<br><br>-- Stage 3: Dynamic force control -- physics -based thrust<br><br>local flightData = setupPhysicsFlight(root , body ,<br><br>blades)<br><br>RunService.Heartbeat:Connect(function(dt) updateThrust(flightData , dt) updateBladeTilt(blades , flightData.<br><br>moveDirection) end)<br><br>-- Stage 4: Interaction -- bind player input bindClickDetector(parts , function() toggleFlight(<br><br><br>flightData) end)<br><br>bindMoveEvent(root , flightData) end<br><br>|
|---|

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8
- 9
- 10
- 11
- 12
- 13
- 14
- 15
- 16
- 17
- 18
- 19
- 20
- 21
- 22

- Fig. 13. Skeleton of a drone behavior script, illustrating the four-stage pipeline. The example targets the Lua scripting API of a commercial gaming platform; the four-stage structure transfers to other engines with comparable rigid-body and event APIs.

- (3) DynamicForceControl: Computes real-timephysical thrust based on angular velocity and lift formulas.
- (4) Interaction: Synchronizes user commands to translate player input into physical motion.

The controllable part structure allows developers to easily inject category-specific behavior functions.

E Additional results

Figure 14 presents a broader gallery of generated assets, showcasing the diversity across various object categories and complex part schemas. Furthermore, we include a video that highlights the practical utility of our generated multi-part meshes.

[Figure 120]

[Figure 121]

[Figure 122]

CubePart : An Open-Vocabulary Part-Controllable 3D Generator • 17

###### Input Prompt & Schema Output Input Prompt & Schema Output Input Prompt & Schema Output

A sleek, single-seat anti-gravity racing speeder with glowing engine nacelles

A space suit designed like medieval plate armor

A floating surveillance unit housed inside a human skull

Sleek fuselage body anti-grav engine pods control handlebars open cockpit stabilizer fins thrust vents

Chainmail Thermal Layer Gauntlet Controls Oxygen Tank Pauldrons Plated Visor Helmet Rocket Greaves

Anti-Grav Jaw Bone Cranium Camera Eye Sockets Sensor Halo Vertebrae Tail

[Figure 123]

[Figure 124]

[Figure 125]

A mechanical horse construct made of brass gears, copper plating, and exposed clockwork mechanisms

A bipedal industrial utility mech designed for lifting heavy cargo containers

A mystical golem creature assembled from floating rocks and glowing crystals Crystal Core Heart Floating Rock Torso Glowing Eye Gems Jagged Legs Shoulder Spikes Stone Arms

Central torso cockpit hydraulic arms with clamp hands rear power pack shoulder-mounted floodlights two heavy-duty legs

Brass plated body segments copper mane and tail exposed gear mechanisms head with lens eyes jointed legs winding key mechanism

[Figure 126]

[Figure 127]

[Figure 128]

A yellow deep-sea research submersible

A long-range weapon carved from a living elder tree

A flight pack designed to look like mechanical angel wings

Pressure hull domed viewport manipulator robot arms rear propeller skid legs top hatch

Enchanted Berry Ammo Leaf Sight Root Stock Vine Trigger Wooden Barrel

Halo Control Ring

Harness Straps Metal Feathers Thruster Tips

[Figure 129]

[Figure 130]

[Figure 131]

Wing Frame

A spacecraft designed with cathedral-like gothic architecture

A fantasy house grown inside a giant red mushroom with white spots

A robotic canine assembled from scrap metal and trash

Flying Buttress Wings Iron Hull Spire Nose Cone Stained Glass Portholes Stone Gargoyle Thrusters

Car Tire Paws Lightbulb Eyes Rusty Pipe Body Spring Tail Toaster Head

Mushroom Cap Roof Porch Steps Round Windows Stalk Walls Stone Chimney Wooden Front Door

[Figure 132]

[Figure 133]

[Figure 134]

A futuristic energy weapon with an old western revolver aesthetic

A high-tech astronaut helmet with a gold visor

A medieval crossbow upgraded with magnetic acceleration rails

Comm Unit Gold Visor Interior Padding Neck Seal Ring Oxygen Hose Port

Battery Quiver Electric Bowstring Magnetic Rails Scope Sight Wooden Stock

Brass Sights Rusty Iron Barrel Spur Hammer Tesla Coil Cylinder Wooden Grip

[Figure 135]

[Figure 136]

[Figure 137]

A high-speed futuristic train engine retrofitted with massive rocket boosters

A futuristic samurai helmet fusing feudal aesthetics with cyberpunk tech

A military combat mech designed to look like a giant scorpion

Streamlined locomotive body driver cabin windows exhaust nozzles multiple large rocket boosters attached to sides and rear reinforced wheel bogies

Data Cables Kabuto Crest Lacquered Neck Guard Neon Visor Steel Menpo Mask

Armored Carapace Body Head Turret Multi-segmented Tail Pincer Claws Stinger Cannon Walking Legs

Fig. 14. Additional Results. We present more results generated by our full pipeline.

