## MARVEL-40M+: Multi-Level Visual Elaboration for High-Fidelity Text-to-3D Content Creation

Sankalp Sinha1∗ Mohammad Sadil Khan1∗† Muhammad Usama1 Shino Sam1 Didier Stricker1 Sk Aziz Ali2 Muhammad Zeshan Afzal1 1,2DFKI 1RPTU Kaiserslautern-Landau 1MindGarage 2BITS Pilani, Hyderabad

# arXiv:2411.17945v2[cs.CV]26Mar2025

sankalp.sinha@dfki.de mohammad.khan@dfki.de

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

The 3D model shows the cholesterol molecule with four fused rings, carbon (grey), hydrogen (white), and a single oxygen (red) atom. It has a flat, twisted shape and includes a hydroxyl group.

1988 Soviet Union coin, 5 kopeck, 25 mm diameter, cupro-nickel, circular, reeded edge, star, wheat stalks, hammer and sickle, "CCCP," number "5," golden color [...]

Tyranid Genestealer, a predatory creature with an elongated skull, jointed arms, a muscular body, strong legs, a whip-like tail, and bony spines.

An old, moss-covered wishing well. Rough stones, aged wood, rusty chains, mushrooms, fallen leaves, and twigs create an enchanting, ancient, and rustic atmosphere.

Detailed, samurai armor, steel, engraved patterns, leather straps, aged look.

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

A medieval fantasy tavern with a wooden building, purple roof, black chimney, and wooden fence. Features include a pine tree, barrels, and a well. Decorated with red [...] banners [...] Set on an octagonal platform, surrounded by blue ground and green plants.

The 3D model is a Japanese house with a rectangular shape [...] pitched roof. The walls are dark brown wood, and the roof is bright red. The house has a small porch with steps, [...] Bamboo trees surround the house, and the ground has sandy... areas with green mossy rocks and grass.

Stitches is a cute, plush-like bear with a large head and small body. He has [...] black X-shaped eyes. His fur is yellow with purple patches and blue ear tips. He wears a white shirt with colorful stars. Stitches stand [...] circular base with a brown surface and a green pattern.

A red panda in a bamboo forest, showing its round, soft face framed by white markings around its eyes and snout. The reddish-brown fur on its back is dense, with a bushy tail ringed in alternating shades of red and brown. Its small, dark eyes [...]

An island with vibrant, multicolored trees, featuring pink, orange, and blue foliage. Waterfalls [...] weathered stone and moss, and hanging vines for detail. Glowing crystals in shades of teal and purple with colorful flowers [...] fantasy RPG setting.

MARVEL 40M+ Annotations MARVEL-FX3D Results

Figure 1. Left: Examples of MARVEL annotations created using our proposed pipeline, which produces high-quality, domain-specific and multi-level text descriptions for 3D assets (Sec 3.1). Right: Qualitative results from MARVEL-FX3D, our two-stage text-to-3D pipeline, which can generate textured mesh from text within 15s (Sec 3.3). Please zoom in for details.

##### Abstract

level descriptions, ranging from detailed (150-200 words) to concise semantic tags (10-20 words). This structure supports both fine-grained 3D reconstruction and rapid prototyping. Furthermore, we incorporate human metadata from source datasets into our annotation pipeline to add domainspecific information in our annotation and reduce VLM hallucinations. Additionally, we develop MARVEL-FX3D, a two-stage text-to-3D pipeline. We fine-tune Stable Diffusion with our annotations and use a pretrained image-to-3D network to generate 3D textured meshes within 15s. Extensive evaluations show that MARVEL-40M+ significantly outperforms existing datasets in annotation quality and linguistic diversity, achieving win rates of 72.41% by GPT-4 and

Generating high-fidelity 3D content from text prompts remains a significant challenge in computer vision due to the limited size, diversity, and annotation depth of the existing datasets. To address this, we introduce MARVEL40M+, an extensive dataset with 40 million text annotations for over 8.9 million 3D assets aggregated from seven major 3D datasets. Our contribution is a novel multi-stage annotation pipeline that integrates open-source pretrained multi-view VLMs and LLMs to automatically produce multi-

*Equally contributing first authors. †Corresponding Author.

73.40% by human evaluators. Project page is available at https://sankalpsinha-cmos.github.io/MARVEL/.

##### 1. Introduction

Text-to-3D (TT3D) content generation has emerged as a pivotal area in computer graphics, vision, and AI, enabling the creation of complex 3D objects from textual prompts [32, 38, 62] by understanding the shape, material properties [71, 89], and complex visual elaborations [36, 77, 90]. This technology holds significant potential for various industries, including gaming, augmented reality (AR), virtual reality (VR), and film production [32, 38]. Recent advancements in text-to-image (TTI) synthesis [3, 21, 67, 69] have achieved remarkable realism and precise control over visual effects [19, 65, 67]. However, extending these capabilities to high-fidelity TT3D generation remains a significant challenge [22, 32, 38, 90]. This is due to the intricate nature of modeling 3D shapes [35, 36, 44, 76, 89], textures [43, 44, 77], colors [71, 89] and spatial relationships [22, 90] from text descriptions, a challenge further amplified by the scarcity of high-quality 3D captions.

Current TT3D datasets like CAP3D [53], 3D-Topia [28], CLAY [89] and Kabra et al [34] attempt to bridge this gap through automated annotations but often fall short due to their reliance on single-view VLMs [11, 39, 45, 46] or GPT-

- 4 [60] for caption generation. This approach frequently results in contradictory or inconsistent captions [34, 54]. Moreover, the captions lack the necessary details for finegrained 3D reconstruction. Additionally, their dependence on proprietary models like GPT-4 [60] introduces significant scalability and cost constraints. Manual annotation is also impractical for large-scale datasets like Objaverse [18] and Objaverse-XL [17]. These datasets contain a diverse range of 3D models—from characters and biological elements to historical artifacts and complex ambigu-

- ous structures—requiring domain-specific expertise for accurate annotation (See Figure 1 - Left). Beyond being timeconsuming and expensive, CAP3D [53] has shown that human-generated captions may not necessarily surpass automated methods in quality.

To address the previously mentioned challenges, we introduce MARVEL(Multi-Level Visual ELAboRation), an automated and scalable 3D captioning pipeline. Our approach combines state-of-the-art multi-view VLM InternVL2 [13, 15] and Qwen 2.5 LLM [85] to generate high-quality captions for over 8.9 million 3D models across seven datasets [10, 16–18, 20, 73, 74, 80]. To ensure domain specific information into our captions and reduce VLM hallucinations, we integrate human metadata from source datasets into our pipeline. Following [12, 92], we identify five key aspects for fine-grained 3D reconstruction: object names and components, shape and geometry, texture and materi-

als, colors, and contextual environments. Our pipeline progressively compresses these aspects to generate five levels of annotations, ranging from comprehensive descriptions (∼200 words) for fine-grained 3D reconstruction to concise tags (∼10 words) for quick modeling, resulting in 40+ million annotations. Our pipeline addresses three fundamental challenges in 3D captioning - detail, accuracy, and scalability. Through comprehensive experimental analysis, we show that MARVEL-40M+ has superior annotation quality, information density, and linguistic diversity compared to other methods [28, 34, 53].

To showcase the application of our dataset, we introduce MARVEL-FX3D (Fast eXecution for 3D), a two-stage pipeline designed for high-fidelity TT3D generation. In the first stage, we fine-tune Stable Diffusion (SD) 3.5 [3] with our annotations to improve its capability to produce images for suitable 3D reconstruction. In the second stage, we leverage the pre-trained Stable Fast 3D (SF3D) [7] for rapid image-to-3D conversion. This enables the creation of textured meshes from texts within 15s . Our approach is inspired by multi-stage TT3D pipelines [41, 58, 71], a promising direction [32, 38, 41] that addresses the limitation of existing Score Distillation Sampling (SDS)[62]-based methods like janus problem [43, 44, 62, 77], oversaturation [43], and lengthy per-prompt optimization [43, 44, 62, 76, 77, 91]. Our experiments demonstrate that MARVEL-FX3D outperforms current state-of-the-art TT3D methods in terms of prompt fidelity and overall preference.

Our contributions can be summarized as follows:

- 1. We present MARVEL, an automated, scalable annotation pipeline for generating high-quality 3D captions. To the best of our knowledge, MARVEL-40M+ is the largest 3D caption dataset to date.
- 2. We propose a multi-level annotation structure that spans from detailed descriptions for fine-grained 3D reconstruction to concise tags for quick modeling.
- 3. We incorporate human metadata from source datasets into our pipeline to inject domain-specific information in the text descriptions and reduce VLM hallucinations.
- 4. As a downstream task, we introduce MARVEL-FX3D, a two-stage framework for high-fidelity TT3D generation.
- 5. Thorough experiments demonstrate that MARVEL40M+ achieves state-of-the-art performances in linguistic diversity, image-text alignment, caption accuracy, and high-fidelity TT3D generation.

- 2. Related Work
- 3D-Text Data: 3D datasets such as ShapeNet [10], Objaverse [17, 18], and Omniobject3D [80] have played a crucial role in advancing 3D understanding tasks such as single [29, 48, 49, 84] or multi-view [70, 86] 3D reconstruction, multi-view consistent image generation [27, 50, 87], and 3D object synthesis [31, 89]. However, they often lack

| |ShapeNet Pix3D OmniObject3D Toys4K GSO ABO Objaverse Objaverse-XL Total 3D Objects<br><br>|Total Captions|
|---|---|---|
|Cap3D [53] 3DTopia [28] Kabra [34] MARVEL|✗ ✗ ✗ ✗ ✗ 6,400 [52] 785,150 221,632 1,013,182<br><br>✗ ✗ ✗ ✗ ✗ ✗ 361,357 ✗ 361,357<br><br>✗ ✗ ✗ ✗ ✗ ✗ 763,827 ✗ 763,827<br><br><br>52,472 374 5,878 4,000 1,030 7,953 798,759 8,031,637 8,902,103|1,013,182<br><br>361,357 763,827 44,510,515<br><br>|

Table 1. Overview of datasets [10, 16–18, 20, 73, 74, 80] annotated using our MARVEL pipeline. MARVEL provides the most extensive 3D asset annotations to date, encompassing over 8.9M 3D objects and 40M captions.

meaningful language descriptions, with available metadata being either noisy or inadequate [47, 53]. This language-3D gap has been a major bottleneck in developing high-fidelity TT3D models [28, 34, 53]. Recent works like CAP3D [53] addresses this by proposing an automated pipeline. It starts with BLIP [40] for single-view captioning of 3D assets followed by refinement using CLIP [64] and caption aggregation by GPT-4 [60]. Subsequent works, Kabra et al.[34] introduced ScoreAgg and PaLI-X[11] to improve caption accuracy, while 3D-Topia [28] explored an alternative path with LLaVA [45, 46] and GPT-3.5. CLAY [89] took a more direct approach, leveraging GPT-4 [60] for multi-view caption generation. Yet, all these approaches face inherent trade-offs. Single-view VLM approaches [28, 34, 53, 54] often produce incomplete or inaccurate annotations [34, 54] for 3D models, while GPT-4-based methods [28, 53, 60, 89] struggle with scalability and cost[4]. Our work presents a novel solution to these challenges through three key innovations. First, we leverage open-source multi-view VLM InternVL2 [13, 15] and Qwen 2.5 LLM [85], achieving GPT-4 [60] comparable performance [2, 13, 15, 85] with-

- out its scalability and cost constraints. Second, instead of discarding human metadata from source datasets as done in previous works [28, 34, 53, 89], we recognize its value as domain-specific prior knowledge. We incorporate filtered versions of this metadata into our pipeline to inject relevant context and reduce VLM hallucinations. Finally, we introduce a hierarchical annotation framework with five distinct levels, ranging from detailed descriptions to abstract tags. This multi-level approach represents a significant departure from existing methods [28, 34, 53, 89], which typically provide only single-level annotations.

Text-to-3D: Current TT3D methodologies can be broadly categorized into two main approaches. One prominent direction is based on the seminal work of DreamFusion [62], which introduced Score Distillation Sampling (SDS) to learn a NeRF [57] representation by leveraging information from pretrained TTI models [5, 67, 69]. Subsequent studies have advanced this framework by improving training stability [43, 77],increasing output diversity [43, 77, 91] and geometry extraction [14, 44, 76, 88]. However, SDSbased methods face two key challenges: geometric inconsistencies known as the Janus problem [32, 37, 38] and slow optimization times. This issue has been partially addressed using amortization efforts [51, 82]. The second group of methods consists of multistage pipelines [24, 33,

41, 42, 58, 71]. The goal is to generate single or multiview images from a TTI model [3, 5, 21, 67, 69], followed by view reconstruction into various 3D representations [7, 24, 29, 41, 57, 70]. These methods often fine-tune the TTI [3, 5, 21, 67, 69] models on TT3D datasets [34, 53] to align the output image with reconstruction needs. PointE [58] fine-tunes GLIDE [59] for TTI synthesis and uses a point diffusion transformer for 3D point cloud generation. Instant3D [41] fine-tunes SD [67] to produce a 2 × 2 grid of multi-view images and uses LRM [29] for 3D Gaussian reconstruction. AssetGen [71] extends LRM towards high-quality 3D meshes with detailed textures and PBR materials. Our dataset, MARVEL-40M+, is uniquely positioned to advance this domain by providing comprehensive, high-quality, and domain-specific text annotations that bridge the gap between TTI generation and image-to-3D reconstruction. By fine-tuning on MARVEL-40M+, we develop MARVEL-FX3D, which demonstrates better performance for high-fidelity TT3D generation compared to existing state-of-the-art methods.

##### 3. Methodology

###### 3.1. Multi-Stage Annotation Pipeline

We now present our proposed MARVEL annotation pipeline, shown in Figure 2 (left). Our goal is to generate detailed and domain-specific captions, for both finegrained and abstract 3D modeling cases. Through a carefully designed five-stage process, MARVEL produces a hierarchy of information-rich and domain-specific annotations. These annotations range from detailed descriptions of object names, shapes, textures, and contextual relationships to concise summaries. Starting with multi-view rendering, our pipeline processes each asset through sequential stages of human metadata refinement, dense description generation via InternVL2 [15], multi-level elaboration using Qwen 2.5 [85], and ethical filtering. Below, we detail each component of our pipeline.

Multi-View Rendering: We first generate 4 multi-view images of resolution 512 × 512 for each 3D model using Blender [1]. We rotate the camera around the object with azimuth angle, θ = {πi2 }ii=4=1 and fixed elevation angle, ϕ = 30. Models are scaled to a unit bounding box, centered at the origin, with the camera distance set as 1.5 relative to this. The four images correspond to the front, back, left, and right sides of the 3D model. Unlike existing

###### Human Metadata Dense Description

MARVEL-FX3D Filtering Redundant Metadata

Comprehensive Description (Level 1) Functional Semantic (Level 3)

Sauron from "The Lord of the Rings" [..] humanoid figure with a muscular build and a menacing presence [..] detailed horned helmet, sharp-edged armor covering the torso, shoulders, [..] dark, flowing cape. The right hand [..] mace with multiple spikes, suggesting combat use. The base [..] jagged rocks [..] with a hexagonal platform featuring a circular runic design at the center. Textures are low-poly but solid and dark, with armor having a matte finish [..] The cape is smooth yet slightly textured for realistic fabric flow. The primary color scheme is gray and black, with gold [..]

Sauron, a tall, muscular figure with a dark, menacing look. He wears a horned helmet, sharp armor, and a flowing black cape. In his right hand, he holds a spiked mace. The base is made of jagged rocks and a hexagonal platform with a circular design. The colors are mostly gray and black, with some gold accents.

Name: Sauron Tags: [lotr, sauron, mace, lowpoly, animated, rings, evil]

Object Name: Sauron [..] Lord of the Rings.

An intricate Fabergé egg, covered in gold filigree with enameled flowers in red, green, and blue.

Components

[Figure 17]

Mistral

Description : “One of my all time favourite fantasy characters![...] I'd love to throw this guy into a game where ...”

Sauron's body, armor, mace, cape [..]

###### Shape and Geometry:

Sauron's overall shape is humanoid [..]

[Figure 18]

Head features prominent horned [..]

Name: Sauron Tags: [lotr, sauron, lowpoly]

Stable Diffusion (Fine-Tuned)

Armor covers his torso, shoulders, arms, legs, and feet [..]

10s

The mace in his right hand has multiple spikes [...] flowing cape [..]

Summary (Level 4)

[..] jagged rocks [..] around the perimeter [..] with a circular runic design [..] center.

A 3D model of Sauron, a tall, dark figure with a horned helmet, armor, and a spiked mace. He wears a flowing black cape and stands on a base with jagged rocks and a hexagonal platform. The colors are gray and black with gold accents.

[Figure 19]

Texture and Materials:

[...] relatively low-poly but are designed to convey solidity and darkness....

[Figure 20]

Moderately Descriptive (Level 2)

Armor [..] matte finish...

[..] Sauron, a humanoid figure with a muscular build and a threatening appearance. [..] horned helmet, angular armor, a spiked mace, and a flowing dark cape. The base features jagged rocks and a hexagonal platform with a circular runic design. The textures are solid and dark, with armor and mace showing minimal detail. The color palette is primarily gray and black, with gold accents on the base.

The mace looks rugged, [..] stone or iron [..]. The cape [..] mimic fabric flow [..]

Stable Fast 3D (Pretrained)

5s

Colors:

Concise Tags (Level 5)

[..] shades of gray and black [..]

Humanoid, muscular, horned helmet, angular armor, spiked mace, flowing black cape, jagged rocks, hexagonal platform, circular runic design, gray, black, gold accents.

Slight highlights in gold [..]

[Figure 21]

Contextual Environment:

[..] hexagonal platform [..] rocky outcrops.

[..] might be part of Mordor.

Multi-View Images

M

[Figure 22]

Multi-Level Visual Elaboration

Qwen2

Figure 2. Left: MARVEL annotation pipeline for 3D assets. Our pipeline starts with human metadata [17, 18] and rendered multi-view images to create detailed visual descriptions using InternVL-2 [13]. These contain object names, shapes, textures, colors, and environments. Qwen2 [85] then processes these descriptions into five hierarchical levels, progressively compressing different aspects of the 3D assets. Right: Our Text-to-3D pipeline finetunes SD 3.5 [3, 21] with this dataset and uses pretrained SF3D [7] to generate a textured mesh in 15s .

images along with our metadata-augmented prompt to generate a dense description of the 3D models. This step avoids separate processing and view aggregation [28, 53] required by pipelines using models like LLaVA [45, 46] or BLIP-2 [39, 40]. The generated description contains several key requirements for fine-grained 3D model reconstruction: (1) structural decomposition with object identification and relative positions, (2) geometric properties, analyzing shape characteristics, symmetry axes, and proportional relationships, (3) surface characteristics, addressing texture and material properties and tactile qualities such as roughness and reflectivity; (4) chromatic analysis, mapping colors across primary objects and sub-components, including patterns and transitions (5) environmental context, capturing spatial relationships and its interaction with other elements. To efficiently scale this process for large-scale annotation, we select InternVL2-40B [13, 15] for its balance of speed, accuracy, and prompt adherence. Recent studies show that InternVL2-40B [13, 15] performs comparably [2, 13, 15] to GPT-4o [60] with significantly lower annotation cost*.

3D captioning pipelines [28, 53], we focus solely on these standard viewpoints. This method aligns with recent studies [68, 79], which demonstrate that VLMs perform better on images from these viewpoints.

Human Metadata Filtering: High-quality 3D annotation requires capturing both visual characteristics (e.g. shape, color, texture) and semantic properties (e.g. domainspecific nomenclature and object identification). This dual focus ensures that descriptions are not only visually precise but also contextually relevant within specific domains. A significant challenge in this process is the tendency of pretrained VLMs [40] to hallucinate when dealing with complex datasets, such as Objaverse [17, 18], due to the inherent 2D-3D domain gap [34]. To address this, we use the user-generated metadata from source datasets, which provides valuable domain-specific names and descriptions that can guide VLMs [13, 15] toward generating more precise and informative annotations. However, this metadata often contains noise, including personalized or sensitive information [17, 18], which can compromise annotation quality. To mitigate this, we use Mistral-Nemo-Instruct-2407 [72] to filter the metadata, removing random, redundant, and sensitive content to ensure that only information relevant to 3D attributes is passed to the annotation pipeline. It is worth noting that our pipeline functions independently of human metadata, with it serving purely as an optional enhancement to add domain-specific terminology in the captions.

Multi-Level Visual Elaboration: This stage focuses on generating multi-level visual elaborations using Qwen2.572B [85] by compressing different aspects of 3D reconstructions at varying levels of detail. This hierarchical approach allows for flexible and adaptive 3D modeling outputs optimized for different use cases, such as scenarios where only key details—like object name and colors—are

Dense Description Generation: In this stage, InternVL2 [13, 15] takes as input the 4 rendered multi-view

*InternVL2-40B [13, 15] ranked third on the Huggingface Open-VLM Leaderboard [2] during our project.

specified, but texture is excluded or where simplified semantic tags is necessary for rapid prototyping. While a direct prompting method will be to specify which aspects to compress, we found that this strategy often constrains the model’s ability to create rich and meaningful captions, aligning with findings from recent studies [75, 78]. To overcome these challenges, we develop a hierarchical prompting strategy that specifies the essential content for each level of elaboration, balancing detail and brevity. Below, we describe each level:

- 1. Comprehensive Description (Level 1): A detailed description covering all aspects of the 3D model, including precise geometric specifications, materials, spatial relationships, and structural details, in 150-200 words.
- 2. Moderately Descriptive (Level 2): Description of the model’s primary structures, components, and key geometric features. This level focuses on the overall shape and main features of the model in 100-150 words.
- 3. Functional-Semantic Description (Level 3): Basic description about the model’s functional aspects, general form, and primary characteristics in 50-100 words.
- 4. Summary (Level 4): A brief description of the object, highlighting its basic form, purpose, and most notable features, similar to existing datasets in 30 words.
- 5. Concise tags (Level 5): A list of distinct concepts of the 3D model for rapid 3D modeling in 10-20 words.

An example of multi-level visual elaboration is illustrated in

- Figure 2 (left), where Qwen 2.5 [85] compresses texture information progressively from Level 1 to Level 4. By Level

- 5, the output shifts to a concise format with colored words representing key semantic tags and core attributes. To assess the effectiveness of this hierarchical compression, we conduct an ablation study in Section 4.3B, measuring the retention of semantic information across all levels.

Ethical Filtering: Given the diverse metadata sources in our annotation pipeline, there is a risk of ethically problematic content being included in the multi-level descriptions. To mitigate this, we use the Qwen 2.5-14B [85] model with a targeted prompt for ethical filtering. This prompt removes meaningless or offensive words, personal names (unless they are famous or contextually relevant), and overly specific identifiers. Importantly, it retains well-known terms, scientific and cultural references, preserving valuable context. This filtering step maintains annotation quality, prevents the leakage of sensitive or inappropriate information, and upholds the integrity of the dataset. For details, please refer to the supplementary material.

###### 3.2. Caption Generation

Datasets: We aggregate ∼ 8.9M 3D assets from seven diverse sources [10, 16–18, 20, 73, 74, 80]. For human metadata injection, we use the name, tags and description from Objaverse 1.0 [18] (Sketchfab) and metadata

from Objaverse-XL [18] (thingiverse and github). Samples from Objaverse-XL [17] containing the file extension .ply were excluded from the dataset. For the rest of the six datasets [10, 16, 20, 73, 74, 80], we use the class categories as metadata. Samples without any renderable multiview images or contains zero-length final annotations are removed from the dataset. The final details of the dataset are provided in Table 1, with additional preparation information available in the supplementary materials.

Implementation Details: Our MARVEL annotation pipeline is optimized for large-scale processing, achieving a throughput of ∼24,000 samples per day. For human metadata filtering, we run the Mistral-Nemo-Instruct-2407 [72] on a single NVIDIA RTX 4090 GPU. Both InternVL240B [13, 15] for dense description generation and the Qwen2.5-72B [85] with 8-bit quantization for multi-level visual elaboration, runs on a single NVIDIA H100 GPU. For the final ethical filtering stage, we run Qwen 2.514B [85] on a single NVIDIA RTX A6000 GPU. For complete details including hyperparameter details, please refer to the supplementary material.

###### 3.3. MARVEL-FX3D Architecture

In this section, we present MARVEL-FX3D, a two-stage pipeline that demonstrates the practicality of the MARVEL40M+ dataset for TT3D synthesis. By leveraging our dataset’s comprehensive text descriptions and diverse 3D assets [18], MARVEL-FX3D generates high-quality textured 3D meshes from text descriptions that can specify multiple objects, scenes, geometric properties, colors, and textures. The pipeline consists of (1) TTI generation using fine-tuned Stable Diffusion [21], followed by (2) singleview 3D reconstruction with a pretrained view reconstruction model [7]. This entire process generates high quality 3D assets in 15s , as illustrated in Figure 2 (right).

Fine-Tuning TTI Model: The objective of this stage is to generate high-quality, diverse images from text prompts that can be effectively converted into 3D textured meshes using pretrained image-to-3D methods [7, 83]. A primary challenge in multi-stage TT3D pipelines [41, 42, 55, 58, 71] is the inherent 2D-3D domain gap, where reconstructing accurate and geometrically consistent 3D models from 2D images is hindered by the ambiguity between background and foreground information [42, 55]. To address this, some methods have fine-tuned TTI [5, 67] models on TT3D datasets [28, 34, 53, 89]. Following this approach, we finetune Stable Diffusion 3.5 [3, 21] using the LORA [30] strategy to bridge this domain gap and generate images similar to the training distribution of the image-to-3D methods [7].

Image-to-3D Generation: In the second stage, the background is removed from the generated image using DIS [63]. The refined image is then passed to pretrained

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

Cap3D -- a man in a tuxedo from Persona 4.

[Figure 28]

Cap3D -- A yellow and

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Cap3D -- A statue of a woman with a cloak, standing on a pedestal.

Cap3D -- A zombie

black bicycle helmet with wheels, a green light, and a black

a man in a tuxedo from Persona 4

in a suit and tie, holding a knife.

handle, resembling a spaceship and a bug.

[Figure 34]

[Figure 35]

3DTopia -- A lifesized mannequin dressed in a suit and tie, with arms outstretched, standing on a white background.

3DTopia -- A welldressed man in a black suit and tie, standing confidently with crossed arms on a white background.

3DTopia -- A white marble statue of a woman, standing on a pedestal, with crossed arms and a contemplative posture.

3DTopia -- A yellow banana-shaped inflatable object with a curved body and a long tail, possibly representing a futuristic sci-fi theme.

[Figure 36]

Kabra -- a man with two faces

K

Kabra -- a silhouette of a man in a tuxedo.

###### Kabra -- a yellow helmet

Kabra --Ka statue wearing a suit.

K

K

MARVEL -- The 3D model is of "Two Face" from Batman: Arkham City. It is a humanoid figure with a damaged left side and an intact right side. The tuxedo jacket is torn on the left, revealing burn damage. The face has prominent scarring, and the hair is messy and charred on the left. The shoes are metallic with a subtle shine.

MARVEL -- Diablo is a tall, slim character from the anime "Tensei Shitara Slime Datta Ken." He has black hair with a red streak, a fair complexion, and wears a black suit with white details, a red chain, and black shoes.

MARVEL -- A futuristic spacecraft with a teardrop shape, yellow color, green lights, and black circular structures at the front. It has segmented fins or wings for maneuverability and is designed for high-speed travel in space.

MARVEL -- The Monument of Dante Alighieri shows a standing figure of Dante in flowing robes, holding a book. It stands on a decorated stone plinth and is made of smooth, polished marble. Located outdoors in Padua, Italy.

MARVEL Caption accurately describes Diablo from "Tensei Shitara Slime Datta Ken" with his tall, slim figure, black suit, red chain, and distinctive red streak in his hair.

[Figure 37]

[Figure 38]

MARVEL Caption best matches the image as it accurately describes the Two-Face character from Batman, with detailed mentions of the damaged left

MARVEL Caption best matches the image as it accurately describes the statue as Dante Alighieri, with details like the flowing robes, book, and stone pedestal.

[Figure 39]

[Figure 40]

MARVEL Caption gives accurate description of shape, color, lights, and structure.

Evaluation

Evaluation

Evaluation Evaluation side, torn tuxedo, scarring, and distinctive features.

- Figure 3. Qualitative Annotation Comparison: From top to bottom Cap3D [53], 3DTopia [28], Kabra [34], MARVEL (Level-4) annotations and GPT-4 [60] evaluation. MARVEL consistently provides the most comprehensive and precise annotations, capturing intricate details such as object names, color, structure, and specific attributes. Red is for wrong captions. Green shows important information. SF3D [7] to generate a high-quality textured mesh in 5s.
- 4. Experiment

due to the evaluation’s time demands. More details are provided in the supplementary section.

|Dataset<br><br>Average Length<br><br>MTLD [56] (@50K)<br><br>Unigram (@50K)<br><br>Bi-Gram (@50K)|GPT4 (@5K)<br><br>Human (@1K)<br><br>|
|---|---|
|Cap3D [53] 16 39.71 15,189 123,071 3D-Topia [28] 29 41.43 10,329 95,856 Kabra [34] 5 25.85 3,862 19,753 MARVEL (Level 4) 44 47.43 27,659 239,052|14.55 9.50<br><br>10.80 14.00 2.24 3.10 72.41 73.40<br><br>|

The experiment section is divided into two parts. In Sec.4.1, we evaluate the quality of our annotations in comparison to the baseline datasets [28, 34, 53]. While Sec. 4.2 presents the performance evaluation of MARVEL-FX3D against current state-of-the-art methods [33, 43, 62, 91]. Both experiments are conducted on Objaverse [18] dataset.

Table 2. Quantitative comparison of annotation quality across datasets. MARVEL surpasses existing datasets [28, 34, 53] in all metrics, showcasing superior linguistic diversity, vocabulary coverage, and significantly higher ratings from GPT-4 and humans.

###### 4.1. Annotation Evaluation

Experimental Setup and Metrics: We assess annotation quality through (1) Linguistic Assessment, (2) Image-Text Alignment, and (3) Caption Accuracy. The linguistic assessment evaluates annotation richness and diversity using the Measure of Textual Lexical Diversity (MTLD) [56] and N-gram analysis [8]. The MTLD metric calculates the average segment length at which the typetoken ratio (TTR) drops below a threshold (typically 0.72), with higher MTLD scores indicating more diverse annotations. We randomly select 50K annotations for analysis.

Linguistic Assessment: Table 2 (left) shows the MTLD [56] score and N-Gram analysis [8]. MARVEL demonstrates a notable improvement, achieving an MTLD score approximately 83% higher than Kabra [34] 19% higher than Cap3D [28] and 14% higher than 3D-Topia[53] signifying richer caption diversity. In addition, MARVEL shows a significantly higher unigram vocabulary size, surpassing Kabra [34], Cap3D [53] and 3D-Topia [28] by factors of approximately 7.1, 1.8 and 2.6 respectively. The trend extends to bigram analysis and average word length as well, showing MARVEL’s superior linguistic diversity and information density. Figure 3 also illustrates that MARVEL’s annotations contain more unique words, particularly focusing on object names, colors, textures, and attributes.

Image-text alignment is measured using both GPT-4 [60] and human evaluators who review four multi-view images of each 3D model and select the best-matching caption. 5,000 samples are evaluated using GPT-4 and 1,000 samples by five human reviewers with each assigned 200 samples. Level 4 annotations from MARVEL-40M+ are used for fair comparison due to their similar average length to baseline datasets [28, 34, 53] as shown in Table 2. Caption accuracy is separately assessed, where GPT-4 and human reviewers evaluate whether all the 3D attributes mentioned in the captions accurately correspond to the 3D models using four multi-view images. For MARVEL40M+, Level 1 annotations are used, which are detailed and form the foundation for subsequent levels. GPT-4 evaluates 1,000 samples, while human reviewers assess 250 samples

Image-Text Alignment: As shown in Table 2, MARVEL achieves notably higher ratings in image-text alignment, with win rates of 72.41% from GPT-4 [81] and 73.40% from human evaluators, outperforming prior methods. This reflects MARVEL’s superior alignment of captions with 3D models. Figure 3 highlights this through examples, showing that MARVEL’s detailed descriptions capture nuances, such as the “flowing robes and book-holding posture of a historical statue”, even at Level 4. In contrast, baselines [28, 34, 53] produce simpler, more generic descriptions. MARVEL’s integration of filtered human-

[Figure 41]

[Figure 42]

[Figure 43]

SHAP-E (5s) Dream Fusion (30m) LucidDreamer (45m) HiFA (1h)

MARVEL-FX3D (15s)

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

A Gothic church with a tall spire, flying buttresses, and lancet windows. Made of weathered stone, it has a grey and brown color.

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

The 3D model is a snowman with three stacked spheres, a carrot nose, a beanie hat, and symmetric arms. It is white with black snowflake patterns and has a matte texture.

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

A 3D model of a windmill on a small floating island, with four wooden blades, a shingled roof, and a green landscape. Decorative golden orbs and gray clouds in a black sky add to the scene's floating effect.

- Figure 4. Visual results of high fidelity TT3D generation. Left to right, the reconstructed 3D assets from Shap-E [33], DreamFusion [62], Lucid-Dreamer [43], HIFA [91] and MARVEL-FX3D.

annotated metadata further enhances the identification of complex, domain-specific entities like “Monument of Dante Alighieri”, “Two Face”, and “Diablo”.

Caption Accuracy: Table 3 shows the caption accuracy results, where MARVEL (Level 1) achieves the highest scores—84.70% in GPT-4 evaluation and 82.80% in human evaluation—demonstrating superior consistency compared to other methods. Baseline datasets with shorter captions, like Kabra [34] and others [28, 53], tend to capture objects semantically (e.g., “a statute”,“a man in a tuxedo”) but lack detailed descriptions. Although longer captions increase the risk of errors, MARVEL (Level 1) maintains high accuracy with an average length of 170 words (34× that of Kabra [34], 10× that of Cap3D [53], and 5× that of 3DTopia [28]), effectively balancing detail and correctness. As shown in Figure 3, MARVEL captures both domain-specific names and intricate features, exemplified by captions like “The Monument of Dante Alighieri... flowing robes, holding a book ... smooth, polished marble...”

Correct GPT4 Evaluation (@1k)

Method

Average Length

Human Evaluation (@250)

Cap3D [53] 16 76.00 72.80 3D-Topia [28] 29 54.60 44.80 Kabra [34] 5 83.40 78.20 MARVEL (Level 1) 170 84.70 82.80

Table 3. Comparison of caption accuracy using GPT-4V [60] and humans, highlighting MARVEL’s (Level 1) superior consistency desite significantly higher caption length.

###### 4.2. Text-to-3D Generation

Implementation Details: We fine-tune SD 3.5 [3, 21] using the Objaverse [18] dataset, which includes 798,759 3D assets, split into training, validation, and test sets in a 90 : 5 : 5 ratio. Fine-tuning is conducted in half-precision

for 5 epochs with a batch size of 8, using a single NVIDIA H100 GPU, with LoRA [30] rank and alpha set to 4.

Baselines: To assess MARVEL-FX3D’s performance in high-fidelity TT3D generation, we compare it with Shap-E [33], Dreamfusion [62], Luciddreamer [43], and HIFA [91]. Due to their slower optimization [43, 62, 91], we evaluate 50 random level 4 captions from the Objaverse [18] test set. Instant3D [41] and Assetgen [71] are excluded due to unavailable code. See supplementary for details.

User Study: We conducted a human evaluation to assess the geometric consistency, visual quality, prompt fidelity, and overall preference of reconstructed 3D assets. Geometric consistency measures realism and physical plausibility, identifying issues like the janus problem. Prompt fidelity evaluates alignment with input text, while visual quality considers aesthetic elements such as colors and textures. Five users were presented with the text prompt and videos of the rendered 3D assets generated by the baseline methods and MARVEL-FX3D. The users scored each asset separately from 1 to 10 based on these criteria, and the final scores were averaged across all users.

Geometric Visual Prompt Overall Consistency Quality Fidelity

Time↓

Shap-E [33] 5s 3.31± 0.71 2.25 ± 0.43 2.65 ± 0.51 2.41 ± 0.50 DreamFusion [62] 30m 4.88 ± 0.47 3.74 ± 0.80 4.22 ± 0.79 4.09 ± 0.81 HiFA [91] >1h 6.59 ± 0.57 6.42 ± 0.26 6.88 ± 0.46 6.44 ± 0.35 Lucid-Dreamer [43] 45m 7.25 ± 0.60 6.47 ± 1.24 6.62 ± 1.37 6.59 ± 0.86 MARVEL-FX3D 15s 7.20 ± 0.91 6.58 ± 0.86 7.71 ± 0.68 6.94 ± 0.71

Table 4. Quantitative evaluation focusing on time and human evaluation criteria: geometric consistency, visual quality, prompt fidelity, and overall preference.

Results: Table 4 presents the quantitative comparison of MARVEL-FX3D against the baselines [33, 43, 62, 91]. MARVEL-FX3D shows notable improvements, achieving the highest prompt fidelity (7.71) and overall preference (6.94), indicating strong alignment with input descriptions and balanced performance across criteria. It also tops visual quality with a score of 6.58, slightly ahead of LucidDreamer [43] (6.47), which marginally exceeds MARVELFX3D in geometric consistency (7.25 vs. 7.20) due to occasional flat outputs from SF3D [7]. Despite this, MARVELFX3D’s processing time is significantly faster, completing in just 15s compared to Lucid-Dreamer’s 45 minutes, HiFA’s over 1 hour, and DreamFusion’s 30 minutes. ShapE [33], while the quickest at 5 seconds, shows considerably lower performance across all metrics. Figure 4 includes some qualitative examples.

Cap3D vs MARVEL-40M+: To demonstrate the effectiveness of our dataset on TT3D generation, we conduct human evaluation on the performance of MARVEL-FX3D when SD 3.5 is either pretrained or finetuned on MARVEL40M+ and Cap3D [53] captions. The evaluation follows the same strategy and uses the same 50 test samples as before. Results on Table 5 confirm that MARVEL-FX3D trained

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[66] and the compression ratio (word count ratio) [6]. Results in Table 6 show strong semantic retention from Levels 1-4, demonstrating effective compression while preserving meaning. However, the shift to Level 5 results in lower similarity, reflecting the transition to a list of concepts at the expense of cohesive descriptions.

Name: Moon Hansteen & Billy craters

Name: Petroglyph C25 Khatm Al Melaha, Kalba, Sharjah

Hansteen

[Figure 60]

Rima Hansteen

[Figure 61]

###### Description:

[Figure 62]

Hansteen is a lunar impact crater ... Billy ... the southwest outer wall ... Rima Hansteen, ... rises Mons Hansteen, or Hansteen Alpha (α) ... triangular in shape ... of volcanic material.

Description: Three human feet or foot prints. Likely Neolithic or earlier. [Fossati 2019 Messages from the Past: Rock Art of the Al-Hajar Mountains (Oman)].

Mons Hansteen

[Figure 63]

[Figure 64]

Billy

Semantic Similarity

Compression Ratio

Source Level Target Level

InternVl2: A surface with multiple prominent circular craters, likely indicative of a celestial body like the Moon or a planet with a heavily cratered surface.

InternVl2: A broken stone with inscriptions or carvings, resting on a bed of dark material, possibly volcanic rock or soil.

- Level 1 Level 2 0.91 0.30
- Level 2 Level 3 0.92 0.27
- Level 3 Level 4 0.88 0.47
- Level 4 Level 5 0.72 0.22

GPT-4: An ancient, fractured stone slab with faint engravings, surrounded by dark debris, suggesting potential archaeological importance and hinting at historical inscriptions.

GPT-4: A grayscale 3D terrain image showcasing the rugged surface of the moon with prominent craters, emphasizing its detailed geological features.

MARVEL: A 3D model of three human footprints carved into a rough, rocky surface. The prints are elongated and rounded with toe impressions, varying in size and depth. The color is gray with darker shadows in the grooves, and the texture is rough and uneven. It looks like it belongs in a desert setting.

MARVEL: The 3D model shows the Moon's surface with the Hansteen and Billy craters, a triangular volcanic feature called Mons Hansteen, and a linear formation known as Rima Hansteen. The terrain is rough and cratered, ...

Table 6. MARVEL-40M+ illustrates strong semantic retention through Levels 1-4 and reduced detail at Level 5.

- Figure 5. MARVEL uses human-generated metadata from source datasets to create detailed, accurate captions (e.g., names of the lunar craters, detection of human footprints) and reduce hallucinations. Without metadata, VLMs like GPT-4 [60] and InternVL2 [15] generate vague or speculative descriptions.

##### 5. Limitation

Our analysis reveals some limitations in MARVEL annotation pipeline and MARVEL-FX3D. First, the underlying VLMs and LLMs struggle with numerical precision [9, 61] and directional understanding [26] in complex scenes with multiple objects and occlusion. Second, InternVL-2 misidentifies very thin objects, often treating side-views as separate entities. Finally, without metadata, captions become generic for complex 3D structures, especially in fragmented geometries like architectural interiors. In supplementary, some visual examples are provided. Additionally, MARVEL-FX3D sometimes generates flat 3D objects due to depth ambiguity in the input image. Despite these challenges, our pipeline remains model-agnostic and adaptable to future enhancements.

on our captions consistently outperforms pretrained and Cap3D versions across all metrics, with a notable increase in prompt fidelity.

|Dataset<br><br>|Text-to-Image [3]<br><br>|Geometric Visual Prompt Overall Consistency Quality Fidelity|
|---|---|---|
|Cap3D MARVEL<br><br>|Pretrained Finetuned Finetuned<br><br>|2.51 ± 1.84 2.54 ± 1.78 2.58 ± 1.86 2.41 ± 1.69<br><br>6.51 ± 1.48 6.53 ± 1.52 6.54 ± 1.68 6.43 ± 1.47<br>7.20 ± 0.91 6.58 ± 0.86 7.71 ± 0.68 6.94 ± 0.71<br>|

Table 5. Quantitative evaluation of MARVEL-FX3D without and with fine-tuning on Cap3D and MARVEL captions, showing improved performance with MARVEL.

###### 4.3. Ablation Study

- A. Effect of Human Metadata on Annotation Quality: Human-generated metadata is vital in the MARVEL annotation pipeline, enriching text captions with domain-specific details. While quantitative analysis requires specialized expertise, we provide its qualitative evidence. As shown in Figure 5, MARVEL accurately identifies specific details, such as “three human footprints on a rocky surface”, which both InternVL-2 [13, 15] and GPT-4 [60] miss, producing only generic descriptions. Similarly, MARVEL captures identifiers like specific lunar craters, absent in InternVL-2 and GPT-4 outputs. This highlights how integrating human metadata enhances the context in annotations. Additional examples from other domains (e.g., biology, historical sites) are in the supplementary material.
- B. Inter-Level Semantic Retention Evaluation: This ablation study measures how well semantic information is retained across MARVEL-40M+ annotation levels as they progress from detailed descriptions to concise tokens. To evaluate this, we report the semantic similarity (cosine similarity of embeddings) between levels using sentence-BERT

##### 6. Conclusion

We present MARVEL-40M+, the largest 3D captioning dataset to date, comprising over 40M+ high-quality text annotations for 8.9 million 3D assets across seven major 3D datasets. Our primary contributions include a scalable, multi-stage annotation pipeline that combines open-source pretrained multi-view VLMs and LLMs with filtered human metadata to reduce hallucinations and introduce domainspecific information. Our pipeline produces five levels of annotations for diverse 3D modeling needs, from detailed descriptions to concise tags. Additionally, we introduce MARVEL-FX3D, a two-stage architecture that leverages fine-tuned Stable Diffusion on our dataset and pretrained Stable Fast 3D to generate high-quality, textured 3D meshes in just 15s. Through extensive experimentation, we demonstrated both MARVEL-40M+’s superior annotation quality and linguistic depth, and MARVEL-FX3D’s improved performance in high fidelity TT3D generation. We believe MARVEL-40M+ will be a foundational resource for advancing TT3D content creation, inspiring further research to address current limitations and expand its applications.

##### 7. Acknowledgement

This work was co-funded by the European Union under Horizon Europe, grant number 101135724, project LUMINOUS. However, the views and opinions expressed are those of the author(s) only and do not necessarily reflect those of the European Union. Neither the European Union nor the granting authority can be held responsible. Human evaluation for this project was primarily funded by BITS Pilani Hyderabad’s NFSG Grant (Reference N4/24/1033).

##### References

- [1] Blender - a 3d modelling and rendering software. https: //www.blender.org. 3
- [2] Openvlm leaderboard - a hugging face space. https:// huggingface.co/spaces/opencompass/open_ vlm_leaderboard. 3, 4
- [3] Stable diffusion 3.5 large - huggingface. https : / / huggingface . co / stabilityai / stable diffusion-3.5-large. 2, 3, 4, 5, 7, 8, 16
- [4] Mohammed Aldeen, Joshua Luo, Ashley Lian, Venus Zheng, Allen Hong, Preethika Yetukuri, and Long Cheng. Chatgpt vs. human annotators: A comprehensive analysis of chatgpt for text annotation. In 2023 International Conference on Machine Learning and Applications (ICMLA), pages 602–609. IEEE, 2023. 3
- [5] DeepFloyd Lab at StabilityAI. DeepFloyd IF: a novel stateof-the-art open-source text-to-image model with a high degree of photorealism and language understanding. https: //www.deepfloyd.ai/deepfloyd-if, 2023. Retrieved on 2023-11-08. 3, 5
- [6] Timothy Bell, Ian H Witten, and John G Cleary. Modeling for text compression. ACM Computing Surveys (CSUR), 21

(4):557–591, 1989. 8

- [7] Mark Boss, Zixuan Huang, Aaryaman Vasishta, and Varun Jampani. Sf3d: Stable fast 3d mesh reconstruction with uvunwrapping and illumination disentanglement, 2024. 2, 3, 4, 5, 6, 7
- [8] Peter F Brown, Stephen A Della Pietra, Vincent J Della Pietra, Jennifer C Lai, and Robert L Mercer. An estimate of an upper bound for the entropy of english. Computational Linguistics, 18(1):31–40, 1992. 6
- [9] Declan Campbell, Sunayana Rane, Tyler Giallanza, Nicol`o De Sabbata, Kia Ghods, Amogh Joshi, Alexander Ku, Steven M. Frankland, Thomas L. Griffiths, Jonathan D. Cohen, and Taylor W. Webb. Understanding the limits of vision language models through the lens of the binding problem,

2024. 8

- [10] Angel X. Chang, Thomas A. Funkhouser, Leonidas J. Guibas, Pat Hanrahan, Qi-Xing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, Jianxiong Xiao, L. Yi, and Fisher Yu. Shapenet: An information-rich 3d model repository. ArXiv, abs/1512.03012, 2015. 2, 3, 5, 15, 24
- [11] Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz,

- Sebastian Goodman, Xiao Wang, Yi Tay, Siamak Shakeri, Mostafa Dehghani, Daniel Salz, Mario Lucic, Michael Tschannen, Arsha Nagrani, Hexiang Hu, Mandar Joshi, Bo Pang, Ceslee Montgomery, Paulina Pietrzyk, Marvin Ritter, AJ Piergiovanni, Matthias Minderer, Filip Pavetic, Austin Waters, Gang Li, Ibrahim Alabdulmohsin, Lucas Beyer, Julien Amelot, Kenton Lee, Andreas Peter Steiner, Yang Li, Daniel Keysers, Anurag Arnab, Yuanzhong Xu, Keran Rong, Alexander Kolesnikov, Mojtaba Seyedhosseini, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, and Radu Soricut. Palix: On scaling up a multilingual vision and language model, 2023. 2, 3
- [12] Yixin Chen, Junfeng Ni, Nan Jiang, Yaowei Zhang, Yixin Zhu, and Siyuan Huang. Single-view 3d scene reconstruction with high-fidelity shape and texture. In 2024 International Conference on 3D Vision (3DV), pages 1456–1467. IEEE, 2024. 2
- [13] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023. 2, 3, 4, 5, 8, 14, 15
- [14] Zilong Chen, Feng Wang, Yikai Wang, and Huaping Liu. Text-to-3d using gaussian splatting, 2024. 3
- [15] Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024. 2, 3, 4, 5, 8, 14
- [16] Jasmine Collins, Shubham Goel, Kenan Deng, Achleshwar Luthra, Leon Xu, Erhan Gundogdu, Xi Zhang, Tomas F Yago Vicente, Thomas Dideriksen, Himanshu Arora, Matthieu Guillaumin, and Jitendra Malik. Abo: Dataset and benchmarks for real-world 3d object understanding. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2, 3, 5, 13, 15, 26
- [17] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, and Ali Farhadi. Objaverse-XL: A universe of 10m+ 3d objects. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023. 2, 4, 5, 13
- [18] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13142–13153, 2023. 2, 3, 4, 5, 6, 7, 13, 15, 22
- [19] Ankit Dhiman, Manan Shah, Rishubh Parihar, Yash Bhalgat, Lokesh R Boregowda, and R Venkatesh Babu. Reflecting reality: Enabling diffusion models to produce faithful mirror reflections. arXiv preprint arXiv:2409.14677, 2024. 2

- [20] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B. McHugh, and Vincent Vanhoucke. Google scanned objects: A highquality dataset of 3d scanned household items, 2022. 2, 3, 5, 13, 15, 27
- [21] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis, 2024. 2, 3, 4, 5, 7, 16
- [22] Chongjian Ge, Chenfeng Xu, Yuanfeng Ji, Chensheng Peng, Masayoshi Tomizuka, Ping Luo, Mingyu Ding, Varun Jampani, and Wei Zhan. Compgs: Unleashing 2d compositionality for compositional text-to-3d via dynamically optimizing 3d gaussians, 2024. 2
- [23] Yuan-Chen Guo, Ying-Tian Liu, Ruizhi Shao, Christian Laforte, Vikram Voleti, Guan Luo, Chia-Hao Chen, ZiXin Zou, Chen Wang, Yan-Pei Cao, and Song-Hai Zhang. threestudio: A unified framework for 3d content generation. https://github.com/threestudio-project/ threestudio, 2023. 16
- [24] Junlin Han, Jianyuan Wang, Andrea Vedaldi, Philip Torr, and Filippos Kokkinos. Flex3d: Feed-forward 3d generation with flexible reconstruction model and input view curation. arXiv preprint arXiv:2410.00890, 2024. 3
- [25] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance, 2022. 16
- [26] Nils Hoehing, Ellen Rushe, and Anthony Ventresque. What’s left can’t be right – the remaining positional incompetence of contrastive vision-language models, 2023. 8
- [27] Lukas H¨ollein, Aljaˇz Boˇziˇc, Norman M¨uller, David Novotny, Hung-Yu Tseng, Christian Richardt, Michael Zollh¨ofer, and Matthias Nießner. Viewdiff: 3d-consistent image generation with text-to-image models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition,

2024. 2

- [28] Fangzhou Hong, Jiaxiang Tang, Ziang Cao, Min Shi, Tong Wu, Zhaoxi Chen, Shuai Yang, Tengfei Wang, Liang Pan, Dahua Lin, and Ziwei Liu. 3dtopia: Large text-to-3d generation model with hybrid diffusion priors, 2024. 2, 3, 4, 5, 6, 7, 14, 17, 18, 19, 20, 21
- [29] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400, 2023. 2, 3
- [30] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 5, 7
- [31] Ka-Hei Hui, Aditya Sanghi, Arianna Rampini, Kamal Rahimi Malekshan, Zhengzhe Liu, Hooman Shayani, and Chi-Wing Fu. Make-a-shape: a ten-million-scale 3D shape model. In Proceedings of the 41st International Conference on Machine Learning, pages 20660–20681. PMLR, 2024. 2
- [32] Chenhan Jiang. A survey on text-to-3d contents generation in the wild. arXiv preprint arXiv:2405.09431, 2024. 2, 3

- [33] Heewoo Jun and Alex Nichol. Shap-e: Generating conditional 3d implicit functions, 2023. 3, 6, 7, 16, 30, 31
- [34] Rishabh Kabra, Loic Matthey, Alexander Lerchner, and Niloy J. Mitra. Leveraging vlm-based pipelines to annotate 3d objects. In Proceedings of the 41st International Conference on Machine Learning. PMLR, 2024. 2, 3, 4, 5, 6, 7, 14, 17, 18, 19, 20, 21
- [35] Mohammad Sadil Khan, Elona Dupont, Sk Aziz Ali, Kseniya Cherenkova, Anis Kacem, and Djamila Aouada. Cad-signet: Cad language inference from point clouds using layer-wise sketch instance guided attention. In In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4713–4722, 2024. 2
- [36] Mohammad Sadil Khan, Sankalp Sinha, Talha Uddin Sheikh, Didier Stricker, Sk Aziz Ali, and Muhammad Zeshan Afzal. Text2cad: Generating sequential cad models from beginner-to-expert level text prompts. In Advances in Neural Information Processing Systems, 2024. 2
- [37] Min-Seop Kwak, Donghoon Ahn, Ines Hyeonsu Kim, JinHwa Kim, and Seungryong Kim. Geometry-aware score distillation via 3d consistent noising and gradient consistency modeling, 2024. 3
- [38] Chenghao Li, Chaoning Zhang, Atish Waghwase, Lik-Hang Lee, Francois Rameau, Yang Yang, Sung-Ho Bae, and Choong Seon Hong. Generative ai meets 3d: A survey on text-to-3d in aigc era. arXiv preprint arXiv:2305.06131,

2023. 2, 3

- [39] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML,

- 2022. 2, 4

[40] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML,

- 2023. 3, 4

- [41] Jiahao Li, Hao Tan, Kai Zhang, Zexiang Xu, Fujun Luan, Yinghao Xu, Yicong Hong, Kalyan Sunkavalli, Greg Shakhnarovich, and Sai Bi. Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model,

2023. 2, 3, 5, 7

- [42] Xinyang Li, Zhangyu Lai, Linning Xu, Jianfei Guo, Liujuan Cao, Shengchuan Zhang, Bo Dai, and Rongrong Ji. Dual3d: Efficient and consistent text-to-3d generation with dual-mode multi-view latent diffusion, 2024. 3, 5
- [43] Yixun Liang, Xin Yang, Jiantao Lin, Haodong Li, Xiaogang Xu, and Yingcong Chen. Luciddreamer: Towards highfidelity text-to-3d generation via interval score matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6517–6526,

2024. 2, 3, 6, 7, 16, 30, 31

- [44] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2, 3
- [45] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 2, 3, 4

- [46] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, 2024. 2, 3, 4
- [47] Minghua Liu, Ruoxi Shi, Kaiming Kuang, Yinhao Zhu, Xuanlin Li, Shizhong Han, Hong Cai, Fatih Porikli, and Hao Su. Openshape: Scaling up 3d shape representation towards open-world understanding. Advances in neural information processing systems, 36, 2024. 3
- [48] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, and Hao Su. One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. Advances in Neural Information Processing Systems, 36, 2024. 2
- [49] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object, 2023. 2
- [50] Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. Syncdreamer: Generating multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453, 2023. 2
- [51] Jonathan Lorraine, Kevin Xie, Xiaohui Zeng, Chen-Hsuan Lin, Towaki Takikawa, Nicholas Sharp, Tsung-Yi Lin, MingYu Liu, Sanja Fidler, and James Lucas. Att3d: Amortized text-to-3d object synthesis. The International Conference on Computer Vision (ICCV), 2023. 3
- [52] Tiange Luo, Honglak Lee, and Justin Johnson. Neural shape compiler: A unified framework for transforming between text, point cloud, and program. 2022. 3
- [53] Tiange Luo, Chris Rockwell, Honglak Lee, and Justin Johnson. Scalable 3d captioning with pretrained models. In Advances in Neural Information Processing Systems, pages 75307–75337. Curran Associates, Inc., 2023. 2, 3, 4, 5, 6, 7, 14, 17, 18, 19, 20, 21
- [54] Tiange Luo, Justin Johnson, and Honglak Lee. View selection for 3d captioning via diffusion ranking. arXiv preprint arXiv:2404.07984, 2024. 2, 3
- [55] Yiwei Ma, Yijun Fan, Jiayi Ji, Haowei Wang, Xiaoshuai Sun, Guannan Jiang, Annan Shu, and Rongrong Ji. X-dreamer: Creating high-quality 3d content by bridging the domain gap between text-to-2d and text-to-3d generation, 2024. 5
- [56] Philip M. McCarthy and Scott Jarvis. Mtld, vocd-d, and hdd: A validation study of sophisticated approaches to lexical diversity assessment. Behavior Research Methods, 42:381– 392, 2010. 6, 16
- [57] Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In Computer Vision – ECCV 2020, 2020. 3
- [58] Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, and Mark Chen. Point-e: A system for generating 3d point clouds from complex prompts. arXiv preprint arXiv:2212.08751, 2022. 2, 3, 5
- [59] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion mod-

- els, 2022a eprint=2112.10741, archivePrefix=arXiv, primaryClass=cs.CV, url=https://arxiv.org/abs/2112.10741,. 3
- [60] Josh OpenAI, Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 2, 3, 4, 6, 7, 8, 14, 15, 16
- [61] Letitia Parcalabescu, Albert Gatt, Anette Frank, and Iacer Calixto. Seeing past words: Testing the cross-modal capabilities of pretrained v&l models on counting tasks, 2021. 8
- [62] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv,

2022. 2, 3, 6, 7, 16, 30, 31

- [63] Xuebin Qin, Hang Dai, Xiaobin Hu, Deng-Ping Fan, Ling Shao, and Luc Van Gool. Highly accurate dichotomous image segmentation. In ECCV, 2022. 5
- [64] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 3
- [65] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In International conference on machine learning, pages 8821–8831. Pmlr, 2021. 2
- [66] N Reimers. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084,

2019. 8

- [67] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 3, 5
- [68] Shouwei Ruan, Yinpeng Dong, Hanqing Liu, Yao Huang, Hang Su, and Xingxing Wei. Omniview-tuning: Boosting viewpoint invariance of vision-language pre-training models,

2024. 4

- [69] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 2, 3
- [70] Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, and Xiao Yang. MVDream: Multi-view diffusion for 3d generation. In The Twelfth International Conference on Learning Representations, 2024. 2, 3
- [71] Yawar Siddiqui, Tom Monnier, Filippos Kokkinos, Mahendra Kariya, Yanir Kleiman, Emilien Garreau, Oran Gafni, Natalia Neverova, Andrea Vedaldi, Roman Shapovalov, and David Novotny. Meta 3d assetgen: Text-to-mesh generation with high-quality geometry, texture, and pbr materials. arXiv, 2024. 2, 3, 5, 7
- [72] Sharath Turuvekere Sreenivas, Saurav Muralidharan, Raviraj Joshi, Marcin Chochowski, Mostofa Patwary, Mohammad Shoeybi, Bryan Catanzaro, Jan Kautz, and Pavlo Molchanov. Llm pruning and distillation in practice: The minitron approach, 2024. 4, 5

- [73] Stefan Stojanov, Anh Thai, and James M. Rehg. Using shape to categorize: Low-shot learning with an explicit shape bias.

2021. 2, 3, 5, 13, 15

- [74] Xingyuan Sun, Jiajun Wu, Xiuming Zhang, Zhoutong Zhang, Chengkai Zhang, Tianfan Xue, Joshua B Tenenbaum, and William T Freeman. Pix3d: Dataset and methods for single-image 3d shape modeling. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018. 2, 3, 5, 13, 15, 28
- [75] Zhi Rui Tam, Cheng-Kuang Wu, Yi-Lin Tsai, Chieh-Yen Lin, Hung-yi Lee, and Yun-Nung Chen. Let me speak freely? a study on the impact of format restrictions on performance of large language models. arXiv preprint arXiv:2408.02442,

2024. 5

- [76] Christina Tsalicoglou, Fabian Manhardt, Alessio Tonioni, Michael Niemeyer, and Federico Tombari. Textmesh: Generation of realistic 3d meshes from text prompts. In International conference on 3D vision (3DV), 2024. 2, 3
- [77] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. In Advances in Neural Information Processing Systems (NeurIPS), 2023. 2, 3
- [78] Jules White, Quchen Fu, Sam Hays, Michael Sandborn, Carlos Olea, Henry Gilbert, Ashraf Elnashar, Jesse SpencerSmith, and Douglas C Schmidt. A prompt pattern catalog to enhance prompt engineering with chatgpt. arXiv preprint arXiv:2302.11382, 2023. 5
- [79] Sangmin Woo, Jaehyuk Jang, Donguk Kim, Yubin Choi, and Changick Kim. Ritual: Random image transformations as a universal anti-hallucination lever in lvlms, 2024. 4
- [80] Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Liang Pan Jiawei Ren, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian, Dahua Lin, and Ziwei Liu. Omniobject3d: Large-vocabulary 3d object dataset for realistic perception, reconstruction and generation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2, 3, 5, 13, 15, 23
- [81] Tong Wu, Guandao Yang, Zhibing Li, Kai Zhang, Ziwei Liu, Leonidas Guibas, Dahua Lin, and Gordon Wetzstein. Gpt4v(ision) is a human-aligned evaluator for text-to-3d generation, 2024. 6
- [82] Kevin Xie, Jonathan Lorraine, Tianshi Cao, Jun Gao, James Lucas, Antonio Torralba, Sanja Fidler, and Xiaohui Zeng. Latte3d: Large-scale amortized text-to-enhanced3d synthesis. The 18th European Conference on Computer Vision (ECCV), 2024. 3
- [83] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191,

2024. 5

- [84] Xinchen Yan, Jimei Yang, Ersin Yumer, Yijie Guo, and Honglak Lee. Perspective transformer nets: Learning singleview 3d object reconstruction without 3d supervision. Advances in neural information processing systems, 29, 2016. 2
- [85] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng

- Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report, 2024. 2, 3, 4, 5
- [86] Jiayu Yang, Ziang Cheng, Yunfei Duan, Pan Ji, and Hongdong Li. Consistnet: Enforcing 3d consistency for multiview images diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7079–7088, 2024. 2
- [87] Yunhan Yang, Yukun Huang, Xiaoyang Wu, Yuan-Chen Guo, Song-Hai Zhang, Hengshuang Zhao, Tong He, and Xihui Liu. Dreamcomposer: Controllable 3d object generation via multi-view conditions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8111–8120, 2024. 2
- [88] Taoran Yi, Jiemin Fang, Junjie Wang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, and Xinggang Wang. Gaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d diffusion models, 2024. 3
- [89] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. Clay: A controllable large-scale generative model for creating high-quality 3d assets. ACM Transactions on Graphics (TOG), 43(4):1–20, 2024. 2, 3, 5
- [90] Xiaoyu Zhou, Xingjian Ran, Yajiao Xiong, Jinlin He, Zhiwei Lin, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang. Gala3d: Towards text-to-3d complex scene generation via layout-guided generative gaussian splatting. arXiv preprint arXiv:2402.07207, 2024. 2
- [91] Junzhe Zhu and Peiye Zhuang. Hifa: High-fidelity text-to-3d generation with advanced diffusion guidance, 2023. 2, 3, 6, 7, 16, 30, 31
- [92] Peiye Zhuang, Songfang Han, Chaoyang Wang, Aliaksandr Siarohin, Jiaxu Zou, Michael Vasilkovsky, Vladislav Shakhrai, Sergey Korolev, Sergey Tulyakov, and HsinYing Lee. Gtr: Improving large 3d reconstruction models through geometry and texture refinement. arXiv preprint arXiv:2406.05649, 2024. 2

## MARVEL-40M+: Multi-Level Visual Elaboration for High-Fidelity Text-to-3D Content Creation

### Supplementary Material

Create a charming village house with walls made of timber and plaster, featuring a thatched roof with a slight sag from years of use. The front of the house has a small wooden porch with a couple of uneven steps leading to a stout wooden door reinforced with iron brackets. Include two shuttered windows on either side of the door, one slightly open, revealing a warm glow from within. Surround the house with a dirt path, a small garden filled with vegetables, and a wooden fence that's partially broken.

Create a small wooden market stall draped with tattered cloth awnings, displaying rows of colorful, ripe fruits like apples, oranges, and grapes stacked in rustic wooden crates. Include a small hand-painted sign hanging crookedly from the stall’s edge, with faded lettering barely legible. Add a wooden stool in front of the counter

[Figure 65]

An old woman

A tree

An old man

A rogue assassin wearing heavy steel armor with noticeable dents and scratches.

Generate a rugged cobblestone path with uneven stones weathered by time, surrounded by patches of moss and tiny cracks filled with dirt. Place small puddles. Add a subtle gradient of dampness near the edges of the path, blending into dark, compacted soil. Ensure the ground looks realistic, evoking a sense of a well-trodden, mysterious alleyway or forest trail.

Design a rustic stone well. The well is circular, built from uneven, weathered gray stones, with moss growing in the crevices. A sturdy wooden frame arches over the well, holding a thick rope tied to a wooden bucket. The rope is slightly frayed, and the bucket has a few cracks, showing signs of long-term use. Surround the well with a small, muddy clearing.

- Figure 6. An example use case of MARVEL-FX3D, demonstrating how multiple prompts can be combined to create a detailed and complex 3D scene, with each prompt contributing specific elements such as characters, structures, and environmental details (Zoom in for details).

This supplementary material provides additional details and results to support the main paper. Section 8 outlines the captioning process, including dataset preparation and implementation specifics. Sections 9 and 10 delve deeper into MARVEL annotations and MARVEL-FX3D results, offering more examples, discussions, and insights into their applications and limitations.

##### 8. Additional Details on Captioning Process

###### 8.1. Dataset Preparation

Objaverse: Objaverse† [18] contains 798,759 3D assets, with metadata (e.g., name, tags, description) available for ∼93% samples after filtering. From ObjaverseXL [17], we rendered 8,031,637 assets, of which ∼3.7M included metadata. After filtering, around 3M samples are retained as valid metadata.

ShapeNet: For the ShapeNet dataset, which contains 52,472 samples, we use the ShapeNet taxonomy as its meta-

†https://objaverse.allenai.org/objaverse-1.0

data (e.g., airplane, bowl, cap, clock, etc.).

Pix3D: For the Pix3D‡[74] dataset, which contains 374 samples, we use the associated category tag as its metadata (e.g., bed, table, desk, chair, etc.).

OmniObject3D: The Omni-Object-3D§[80] dataset, which contains 5,878 samples, we use the folder names (e.g., bed, table, desk, chair, etc.) as our metadata.

Toys4K: For the Toys4K¶[73] dataset, which contains 4,000 samples, we use the folder names (e.g., car, airplane, train, robot, etc.) as our metadata.

GSO: The GSO (Google Scanned Objects)||[20] dataset, which contains 1,030 samples, we use the folder names (e.g., lamp, sofa, vase, refrigerator, etc.) as our metadata.

ABO: The ABO (Amazon Berkeley Objects)**[16] dataset,

‡http://pix3d.csail.mit.edu/ §https://omniobject3d.github.io/ ¶https://github.com/rehg-lab/lowshot-shapebias/

tree/main/toys4k ||https://goo.gle/scanned-objects

**https://amazon-berkeley-objects.s3.amazonaws. com/index.html

which contains 7,953 samples, provides metadata through listings information. Since these listings are multilingual, we first use the nllb-200†† model to translate the listings to English. The translated English listings are then used as our metadata.

###### 8.2. Implementation Details

For human metadata filtering, we use the Mistral-NemoInstruct-2407 model with a temperature of 0.3 and a top-p value of 0.95. For dense description generation, we employ InternVL2-40B, configured with a temperature of 0.70, a top-p value of 0.95, and a repetition penalty of 1.10, with multinomial sampling enabled. For multi-level visual elaboration, we utilize Qwen2.5-72B with 8-bit quantization, a temperature of 0.70, a top-p value of 0.80, and a repetition penalty of 1.05. Finally, the Qwen2.5-14B model, used for the ethical filtering stage, is configured with a temperature of 0 and a top-p value of 0.90. For human evaluations in our paper, we developed a Gradio app to compare our captions with those from baseline datasets, including Cap3D, 3DTopia, and Kabra, as well as to evaluate FX3D results against text-to-3D baselines. The evaluations were conducted by a panel of 5 human experts.

###### 8.3. Compute and GPU Hours

MARVEL’s annotation pipeline utilizes one NVIDIA H100-80GB GPU, one RTX-4090 GPU, and one RTXA6000 GPU, achieving a throughput of approximately 24,000 samples per day. Annotating the entire Objaverse dataset (800,000 samples) would thus require about 33 days, incurring an estimated total computational cost of approximately $2,700–$3,000, based on publicly available GPU pricing‡‡. For comparison, sequential human annotation has a considerably lower throughput (1,400 samples/day) and higher cost ($87.18 per 1,000 annotations), resulting in approximately 572 days (about 1.57 years) and a total cost of roughly $69,744 for annotating the complete Objaverse dataset. In contrast, the automated Cap3D pipeline—leveraging BLIP2, CLIP, and GPT-4 models on cloud-hosted NVIDIA A40 GPUs—achieves significantly higher throughput (65,000 samples/day) at a lower cost ($8.35 per 1k annotations), requiring only about 13 days and totaling approximately $6,680 for the entire dataset [53]. Our pipeline annotates the Objaverse dataset at approximately half the total cost of Cap3D, although with a lower throughput (33 days vs. Cap3D’s 13 days). Both automated methods substantially outperform sequential human annotation in terms of speed and cost. Importantly, our

††https : / / huggingface . co / facebook / nllb - 200 distilled-600M

‡‡https://tinyurl.com/gpu-usage-pricing (Original)

pipeline delivers annotations of significantly higher quality compared to Cap3D, making it particularly advantageous when balancing annotation quality and cost efficiency. All comparisons assume sequential (non-parallelized) processing; parallelization would further reduce annotation time for all methods.

Total Cost (800k samples) Human 1,400 572 $87.18 $69,744 Cap3D 65,000 13 $8.35 $6,680 MARVEL 24,000 33 $3.38–$3.75 $2,700–$3,000

Throughput (samples/day)

Total Days (800k samples)

Cost per 1k annotations

Method

Table 7. Comparison of annotation pipelines based on throughput, annotation time, and cost for annotating the Objaverse dataset (800k samples). All estimates assume sequential annotation without parallelization.

##### 9. Additional details on MARVEL annotations

###### 9.1. More Results on Effects of Human Metadata

Figure 7 showcases examples where human-provided metadata from source datasets reduce VLM hallucination and enhances annotations with domain-specific information. To generate captions using InternVL2 [13, 15] and GPT-4 [60], we input the same multi-view images used for MARVEL annotations, instructing them to produce concise descriptions that include names, shapes, textures, colors, and contextual environments. Examples 1, 2, and 3 demonstrate how the inclusion of simple metadata (e.g. “La Cava Window”, “Mount St. Helens”) significantly reduces VLM hallucination, resulting in more accurate captions. Example 4 illustrates how metadata can support the generation of highly domain-specific information (e.g. “alpha-helices and beta sheets”, “N-terminus, middle, and C-terminus”).

###### 9.2. More 3D Captioning Results

We provide more qualitative comparisons of annotations, highlighting differences between the baseline [28, 34, 53] and our proposed MARVEL-40M+ dataset. For consistency, we used only Level 4 annotations, as their length closely matches that of the baselines. To improve clarity, we further categorized examples into distinct domains.

- • Figure 9 showcases 3D models of automotive designs (e.g., cars, planes) and CAD models.
- • Figure 10 features iconic characters from anime, movies, and video games.
- • Figure 11 illustrates biological elements such as animals, plants, and molecules.
- • Figure 12 includes diverse items ranging from everyday objects, essentials, food to luxury items.
- • Figures 13 and 14 depict historical artifacts (e.g., statues, memorials) and various scenes (e.g. digital elevation maps, realistic and animated scenes) respectively.

[Figure 66]

[Figure 67]

HallucinationReduction+ DomainSpecificInformation

Hallucination Reduction + Domain Specific Information

Hallucination Reduction

Domain Specific Information

[Figure 68]

[Figure 69]

[Figure 70]

Name: Mount St. Helens, Washington State, USA

Name: La Cava Window

Name: Knightmare Deathstroke

Name: Human Salivary Amylase Molecular Structure

[Figure 71]

Description: Joe Manganiello's portrayal of Deathstroke - Knightmare version.

Description: Molecular structure of human salivary alphaamylase. Salivary alpha-amylase is a major component of human saliv. It plays a role in the initial digestion of starch. The salivary amylase folds into a multidomain structure consisting of three domains, A, B and C.

Description: La Cava Window section

Description:

InternVl2: This 3D model, "Sword Warrior," is monochromatic with black, gray, and brown accents. It has a smooth, polished texture, wears detailed armor, and is equipped with a sword.

InternVl2: The 3D model is of Mount Fuji. It is a majestic, snow-capped volcano with a conical shape. The color varies from white at the summit to green and brown lower down, showcasing its natural texture and rugged terrain.

InternVl2: The 3D model depicts a detailed book with a textured cover, showcasing an open window with stained glass and a brick wall.

InternVl2: The 3D model is a complex, intertwined structure resembling a molecular or protein model, with detailed, curving pathways and loops.

GPT-4: This 3D model of Mount St. Helens showcases its iconic conical shape, snow-dusted summit, and rugged terrain. The green vegetation contrasts with eroded volcanic ridges, detailed lava flows, and a prominent crater, capturing the volcano's dramatic and dynamic landscape.

GPT-4: The model represents a futuristic warrior clad in a black and beige armored suit with metallic accents. It features angular, sharp textures, a muscular humanoid shape, a tactical sword, and a holstered weapon, evoking a sleek, battle-ready appearance.

GPT-4: A 3D model of a beige-colored protein structure, featuring intricate, smooth-textured helices and loops with a tangled, organic design.

GPT-4: 3D model shows rustic arched stained glass window, textured walls, ribbed paneling, and wooden floor featuring 'La Casa Restaurant' text.

MARVEL: A window with a dark wood frame and colorful stained-glass panels, surrounded by brick and painted plaster walls. A wooden mat with "La Cava Restaurant" text and decorative elements is in front. White wainscoting and a wooden baseboard below, a smooth ceiling above ...

MARVEL: Mount St. Helens is a conical volcano with a wide, bowl-shaped crater at the summit. The slopes are steep,with rocky and granular surfaces. The south side is steeper than the north. Vegetation covers the lower slopes, and there are smooth, reflective lakes nearby.

MARVEL: The 3D model is of Human Salivary Amylase, a protein with three main parts: N-terminus, middle, and C-terminus. It has a compact core with alpha-helices and beta-sheets. The surface is smooth, and color highlights (red, green, blue) show different residues.

MARVEL: Deathstroke in his Knightmare outfit, muscular and armored, with white hair and an eyepatch. One hand holds a sword. Dark gray, black, and brown colors with a low-poly aesthetic.

- Figure 7. Effect of including human metadata, highlighting improvements in descriptive accuracy and contextual relevance compared to outputs generated without metadata, even when using state-of-the-art models like GPT-4 [60] and InternVL2 [13]. Metadata inclusion helps reduce hallucinations and enhances domain-specific understanding.

MARVEL -- A cartoon hornbill with a large head, orange beak, light blue body, and yellow legs. The tail has two blue feathers with white tips.

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

MARVEL -- The Vulcan Statue is a 3D model of a Roman god holding a hammer and an anvil, standing on a pedestal with steps. The figure has a dynamic posture, dressed in traditional Roman attire. The statue is made from cast iron, with a smooth, rugged texture and a monochromatic finish.

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

MARVEL -- The 3D model includes two thin metal pins and a trading card in a clear, rectangular protective slab. The pins are sharp and have a broader top. The slab has a metallic label with a Boston Celtics logo and player details. The colors are green, black, silver, and white.

Counting Mistake Object Misidentification Ambiguous Views

- Figure 8. Failure cases of the MARVEL annotation pipeline. From left to right, the examples illustrate errors such as counting mistakes, object misidentification, and challenges with ambiguous views.

As illustrated in the figures, MARVEL annotations offer more precise and domain-specific descriptions, leveraging accurate nomenclature and contextual terminology, surpassing the quality of the baseline datasets.

- 9.3. More Multi-Level Examples

rical lemon head with sunglasses—by emphasizing their geometric precision and structural symmetry. Additionally, for simpler models like the low-poly tree with geometric leaves and the realistically textured orange, the pipeline adeptly captures essential details, highlighting subtle irregularities and primary shape characteristics. This flexibility ensures consistent annotation quality across diverse modeling scenarios.

We present additional qualitative results showcasing our multi-level annotations across all seven datasets [10, 16, 18, 20, 73, 74, 80], with two examples per dataset - Objaverse (Figure 15), OmniObject3D (Figure 16), ShapeNet (Figure 17), Toys4k (Figure 18), ABO (Figure 19) and GSO (Figure 20).

###### 9.5. Need for Multi-Level-Structure

Hierarchical structures are essential in AI. Examples include multi-resolution models in computer vision, such as feature pyramids, and hierarchical embeddings in NLP, such as document summarization. In 3D modeling, ShapeNet uses a hierarchical taxonomy [10] to improve adaptability across tasks. MARVEL similarly adopts a hierarchical design, ensuring task-specific granularity. By using a predefined hierarchy, MARVEL eliminates the need for repeated prompting. This reduces latency, computational costs, and inconsistencies associated with dynamically ad-

###### 9.4. More on Simple and Textureless Models

Our annotation pipeline is robust and adaptable, effectively handling both simple and texture-less models by dynamically adjusting its descriptive verbosity. As illustrated in Figure 22, the pipeline generates concise yet accurate annotations for texture-less models—such as the smooth, monochromatic humanoid figure and the symmet-

justing verbosity. Such dynamic adjustments would require multiple inference steps and additional processing, making them impractical for large-scale pipelines—even for future LLMs/VLMs. Additionally, dynamic generation introduces risks like semantic drift, information loss, and verbosity imbalance, decreasing annotation reliability. Our ablation study (Section 4.3B and Table 5) confirms that MARVEL’s structured verbosity effectively maintains essential details. It optimizes verbosity levels according to task requirements, as validated by cosine similarity and compression ratio.

###### 9.6. Failure Cases

- Figure 8 presents examples of the failure cases discussed in Section 5 of the main paper, illustrating the challenges associated with using pretrained VLMs to generate dense descriptions of 3D models.

###### 9.7. More on MTLD Scores

The MTLD (Measure of Textual Lexical Diversity) algorithm quantifies vocabulary diversity by segmenting a text whenever the Type-Token Ratio (TTR)—the ratio of unique words to total words—drops below a fixed threshold (commonly 0.72). It processes the text both forwards and in reverse to reduce positional bias, and calculates the final MTLD score as the total number of words divided by the number of segments (called factors). A low MTLD score indicates a repetitive vocabulary and low lexical diversity, while a high score reflects a rich and varied vocabulary. For instance, the repetitive string "hello hello hello hello hello hello" results in a low MTLD score of approximately 2.02, due to the lack of word variation. In contrast, the diverse sentence "the quick brown fox jumps over the lazy dog" yields a high MTLD score of around 22.68, as it contains many unique words. The pseudo-code for the algorithm is given in Algorithm 1 as seen in [56].

##### 10. Additional results of MARVEL-FX3D

###### 10.1. More Implementation Details

As discussed in the main paper, MARVEL-FX3D is a twostage pipeline. In the first stage, Stable Diffusion 3.5 [3, 21] is fine-tuned. During each epoch, one annotation is sampled from five levels and paired with a randomly selected multiview image for MSE loss calculation. During inference, CFG [25] is set to 7.5, and 30 steps are used to balance speed and output diversity.

###### 10.2. Baseline Adaptation

We use the official implementations and pretrained models for Shap-E [33] and Luciddreamer [43], training the latter for 3k steps. Dreamfusion [62] and HIFA [91] are

Algorithm 1 MTLD Score [56]

- 1: function MTLD(text, min = 10)
- 2: forward ← MTLDPROCESS(text,min)
- 3: reverse ← MTLDPROCESS(Reverse text,min)

- 4: return (forward + reverse)/2
- 5: end function
- 6: function MTLDPROCESS(text, min)
- 7: factor ← 0
- 8: factor lengths ← 0

- 9: start ← 0
- 10: for x ← 0 to length(text) − 1 do
- 11: segment ← text[start : x + 1]
- 12: if x + 1 = length(text) then
- 13: partial ← 1−TTR1−(segment0.72 )

- 14: factor ← factor + partial
- 15: factor lengths ← factor lengths + length(segment)

- 16: else if TTR(segment) < 0.72 and length(segment) ≥ min then
- 17: factor ← factor + 1
- 18: factor lengths ← factor lengths + length(segment)

- 19: start ← x + 1
- 20: end if
- 21: end for
- 22: return factorfactorlengths

- 23: end function

trained using the open-source threestudio [23] implementation, with 10,000 and 24,000 steps, respectively, under default settings.

###### 10.3. More Text-to-3D Results

Figures 23 and 24 showcase visual results of TT3D generation on unseen prompts. Using GPT-4 [60], we generated

- 10 random prompts focused on shape and scene descriptions. As demonstrated, MARVEL-FX3D produces higherfidelity 3D models from text prompts compared to the baseline methods.
- 11. Discussion on Application of MARVEL

The MARVEL-40M+ dataset, with its scale and diversity, serves as a powerful resource for text-to-3D tasks such as reconstruction, multi-view consistency, and compositional scene generation. A notable real-world use case, illustrated in Figure 6, demonstrates how MARVEL-FX3D which is trained on MARVEL dataset enables rapid prototyping of diverse 3D objects from complex, fine-grained or simple text prompts. This capability facilitates the creation of intricate scenes, making it particularly valuable for applications in gaming, AR, and VR.

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

Cap3D -- Air Force One flying in the sky.

###### Cap3D --

###### Cap3D -- RAF Spitfire III biplane in orange and white

Cap3D -- a small black car.

Yellow sports car toy model

[Figure 89]

[Figure 90]

a man in a tuxedo from Persona 4

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

3DTopia -- A sleek and modern white and blue Boeing 747 commercial jetliner with a long fuselage, two wings, and a tail, flying in the air. It has a distinctive shape, pointed nose, wide curved tail, and multiple engines.

3DTopia -A small black hatchback car with a red roof, compact and sleek design, suitable for city driving and parking in tight spaces.

3DTopia -- A small sleek and streamlined propeller plane with a red and white color scheme and intricate details of classic propellerdriven aircraft.

###### 3DTopia --

(No Captions)

[Figure 97]

[Figure 98]

Kabra -- a yellow car

Kabra -- a red and gray airplane flying in the air

Kabra -- a black car

Kabra -- an airplane flying in the air

K

K

[Figure 99]

MARVEL -- The Dragon Rapide is a biplane ... rectangular fuselage and two sets of rectangular wings connected by

MARVEL -- The 1948 Ferrari 166 MM Barchetta is a classic yellow roadster with a sleek, low profile. It has four black wheels with chrome hubcaps, two round headlights, and a prominent vertical grille. Inside, there are two brown leather seats and a red steering wheel.

MARVEL -- Air Force One is a large, white Boeing 747 with a hump at the front, four engines, and a tall tail fin. It has blue trim and the United States flag on the tail. The text “UNITED STATES OF AMERICA” is on the side. It looks official and detailed.

MARVEL -- The Peugeot 107 Buba is a small, rounded car with a modern look. It is primarily black with silver or chrome wheels. Key features include headlights, taillights, side mirrors, and a license plate.

struts ... twin-engine setup with propellers and a cockpit with windows ... gray with orange accents on the wings, nose, and tail. Roundels with red, white, and blue concentric circles ... on the wings and fuselage. The landing gear is sturdy and metallic.

[Figure 100]

[Figure 101]

[Figure 102]

Cap3D -- a white valve resembling a steering wheel.

Cap3D -- Model of a vintage jet airliner with red, white, and blue livery, featuring the logo BEA on its fuselage and Trident near the nose.

Cap3D -- a green lathe machine with a light on top.

[Figure 103]

Cap3D -- a largetired, orange tractor.

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

3DTopia -- A wellmaintained yellow and green John Deere tractor, used for agricultural tasks. It has a distinctive shape with a large round body, smaller rectangular cab, and wheels. It is sitting on a dirt field with a trailer attached to it.

3DTopia -- A green rectangular metal machine, with various buttons and dials, designed for sewing and crafting. It is a functional and practical tool, durable and sturdy, used in a workshop or industrial setting.

3DTopia -(No Captions)

[Figure 108]

[Figure 109]

[Figure 110]

###### 3DTopia --

(No Captions)

[Figure 111]

Kabra -- a green machine with a lamp on top of it

Kabra -- an airplane flying in the air

Kabra -- a tractor sitting on top of a dirt patch.

K

###### Kabra -- a steering wheel

K

MARVEL -- The lathe machine has a rectangular base with cooling vents, a spindle/chuck area for holding and rotating

MARVEL --The Keystone 6" Knife Gate Valve is a cylindrical industrial valve with a rectangular gearbox and a handwheel for manual operation. It has symmetrical bolt holes and is made of smooth, gray metal. Used for controlling fluid flow in harsh conditions.

MARVEL -- The Trident aircraft has a long, narrow body with swept-back wings and a tail fin. It has smooth surfaces and red accents on the wings and tail. A red "BEA" logo is on the side, and the tail features a blue background with a red Union Jack flag. Red and white stripes are on the rear.

workpieces, a headstock for controlling the spindle, a tailstock for holding tools, a tool holder that moves along the bed, and an adjustable lamp for lighting. It is symmetrical and has a bulky, metallic body painted green.

MARVEL -- A robust, boxy Ford tractor with large rear wheels, small front wheels, and an orange color. It has a metallic body, an operator’s seat, and a roll bar. Suitable for agricultural or maintenance tasks.

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

Cap3D -- a race track with a fence and a Red Bull racing car.

[Figure 116]

Cap3D -- a futuristic black and white Lamborghini Aventador RS sports car.

Cap3D -- Carbon fiber racing steering wheel with integrated multifunctional display, various control knobs, and buttons.

Cap3D -- a small, white, futuristic electric car with wheels.

[Figure 117]

[Figure 118]

3DTopia -- A detailed and realistic miniature race car track made of plastic, with a green color scheme featuring a red and white checkered pattern. The track has a starting line, curves, turns, straight sections, and barriers ... surrounded by a fence and ... sign ... "No Drinking and Driving".

[Figure 119]

[Figure 120]

[Figure 121]

3DTopia -- A black cylindrical steering wheel, typically covered in leather, used to control the direction and speed of a vehicle.

3DTopia -(No Caption)

3DTopia -(No Caption)

[Figure 122]

Kabra --a black and gray car

Kabra -- a race track with two cars on it

Kabra -- a gray and black car

Kabra -- a black steering wheel

MARVEL -- The 3D model shows a Formula 1 race scene with two cars, one red and one black, on a racetrack ... has red and white striped curbs, ... A catch fence with metal mesh surrounds the track, ... large green signs ... "ROLEX" and "Heineken." Smaller boards ... Pirelli logo. A digital camera tripod is behind the fence ... The Ferrari car is red with black accents, and the Red Bull car is black with yellow and red details.

MARVEL -- The Lamborghini Terzo Millennio is a futuristic sports car with a wide, aerodynamic body, sharp angles, and pronounced wheel arches. It has a sleek windshield, angular headlights, and is primarily white/gray with black and red accents. The car features large air intakes on the hood and a sloping roof, with horizontal LED taillights and an integrated spoiler at the rear.

MARVEL -- A Ferrari SF70H Steering Wheel from 2017, circular with ergonomic grips, a central display screen, and various buttons and dials. It has a main lever at the bottom and is made of carbon fiber with color-coded buttons.

MARVEL -- The Tesla Cybertruck is a futuristic truck with sharp edges and a flat, rectangular body. It has a low front and higher rear, with a large open bed. The truck is mainly silver/grey with a smooth, metallic texture and glossy finish.

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

###### Cap3D -- a white and blue speed boat.

Cap3D -- a stainless steel watch with a silver diamond bracelet.

Cap3D -- Black and orange motorcycle model

Cap3D -- A red and rusty moped with white accents.

[Figure 128]

3DTopia -- A unique and intricate silver bracelet with a curved shape, cut-out section, and a twisted or intertwined pattern.

3DTopia -- A white and blue sleek and streamlined

[Figure 129]

3DTopia -- A sleek and modern black and silver Harley Davidson motorcycle with a distinctive design, including a fuel tank, handlebars, and a single seat.

sailboat with a pointed bow, curved stern, and slender body, suitable for recreational or leisure use.

###### 3DTopia --

[Figure 130]

(No Captions)

Kabra -- a black watch without a band

Kabra -- a black motorcycle

Kabra -- a red scooter

K

Kabra -- a boat floating in the air

MARVEL -- The Harley Davidson XR1200x is a black motorcycle with a sturdy frame, V-twin engine, and dual exhaust pipes. It has a black leather seat, raised handlebars, and a round headlight. The bike is streamlined and aggressive-looking, with disc brakes and visible suspension.

MARVEL -- The 3D model is a vintage red Vespa scooter with a long, cylindrical body, a black seat, and two round rearview mirrors. It has a central front headlamp, worn side panels, and silver-rimmed wheels with tread marks. The rear fender partially covers the rear wheel, and the exhaust pipe curves up near the rear.

MARVEL -- The 3D model is a Rolex Submariner watch with a round case, fluted bezel, and Oyster bracelet. It features Mercedes hands and a black dial with white or luminous accents. The bezel has white markings for diving.

MARVEL -- A small, streamlined yacht with a smooth, glossy hull, cushioned seating, and a helm station. It has a wood-paneled cabin, stainless steel railings, and blue accents. Highly reflective exterior.

- Figure 9. Qualitative comparison of 3D annotations across baselines [28, 34, 53] and the proposed MARVEL-40M+ for automotive (cars, planes, etc) and CAD models. MARVEL-40M+ provides more accurate and domain-specific annotations, compared to the baselines. Incorrect captions are highlighted in red, while important captions are highlighted in green.

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

Cap3D -Wolverine in black shirt and pants, holding a knife.

###### Cap3D -- a samurai riding a blue dragon with a sword.

Cap3D -- Stylized character with large triangular head, red tuft of hair, blue eyes, wearing a yellow and white striped shirt, blue shorts, and white shoes.

[Figure 145]

Cap3D -- a superhero in a yellow and red costume with a white cape.

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

3DTopia -- A small figurine of a Dragon Ball Z superhero character, wearing a white and yellow costume, red cape, and standing on a white background.

###### 3DTopia --

3DTopia -(No Captions)

3DTopia -- A yellow sponge character with a red nose and a big smile, standing on a white background, likely a toy figurine from the animated television show "SpongeBob SquarePants."

[Figure 151]

(No Captions)

Kabra -- a man fighting a blue dragon with a sword

[Figure 152]

Kabra -- a man in a superhero costume with a cape.

Kabra -- a man with claws in his hands

[Figure 153]

Kabra -- (No Captions)

[Figure 154]

[Figure 155]

MARVEL -- The 3D model features Roronoa Zoro from One Piece, holding three swords, with a blue dragon coiling around him. A water splash surrounds them, adding intensity to the dynamic scene.

MARVEL -- The 3D model shows Saitama from "One Punch Man," a bald, muscular character in a yellow suit with a white cape, black belt, red gloves, and red boots. He stands in a dynamic, forward-leaning pose.

MARVEL -- Wolverine is a muscular, strong figure with short, dark hair and a beard. He wears a light grey shirt and black pants. His arms have three metal claws, and his legs are slightly bent.

MARVEL -- The 3D model is of Phineas from "Phineas and Ferb." He has a conical head, large blue eyes, a red tuft of hair, a yellow striped shirt, blue shorts, and blue shoes with white laces. The model is smooth and cartoonish, with a simple cylindrical body

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

Cap3D -- a soldier wearing a gas mask, cloak, and suit.

Cap3D -- a girl with dreadlocks in a dress.

Cap3D -- a character depicted as a red devil superhero in a suit with black and green elements.

[Figure 160]

Cap3D -- Lightning McQueen from Cars

[Figure 161]

[Figure 162]

3DTopia -- A detailed and realistic figurine of a soldier in military attire, wearing a camouflage jacket, a helmet, and holding a weapon, standing on a white background.

3DTopia -- A red and yellow toy car, resembling Lightning McQueen from the Disney Pixar movie Cars, with a sleek and aerodynamic design, long pointed nose, curved body, and racing stripe down the middle.

3DTopia -(No Captions)

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

3DTopia -(No Captions)

Kabra -- a cartoon character with dreadlocks and a dress

[Figure 169]

Kabra -(No Captions)

Kabra -- a man wearing a cape and a helmet

Kabra -- lightning mcqueen from the movie cars

[Figure 170]

MARVEL -- The 3D model shows Ghost Jawbone from Call of Duty Mobile. He wears a black helmet, a brownish-khaki cape, a dark green shirt with a tattoo, an olive green vest with a red pouch, olive green cargo pants, and dark combat boots. The character stands upright with a flowing cape.

[Figure 171]

MARVEL -- Rath is a strong, muscular alien with a tigerlike appearance. He has a large, striped body, a white underbelly, and sharp claws. His face has smooth, animalistic features and glossy eyes.

MARVEL -- Lightning McQueen is a sleek, red racing car with a long front, narrow middle, and large rear spoiler. It has yellow and black flames, rugged tires, and the number "95" displayed prominently.

MARVEL -- Tia Dalma model with dark dreadlocks, flowing dress with earthy patterns, and a silver pendant necklace.

[Figure 172]

[Figure 173]

[Figure 174]

Cap3D -- a stylized character with exaggerated head shape, grey hair, purple and black clothing, and simple facial features.

[Figure 175]

Cap3D -- a camper van with luggage on top.

Cap3D -- Brown, melting wax-like structure resembling a drooping, curved witch's hat with a textured, glossy surface.

Cap3D -- a blue and yellow fish with a large head, big yellow eyes, and a cute, colorful creaturelike appearance.

[Figure 176]

[Figure 177]

[Figure 178]

3DTopia -- A yellow and brown vintagestyle camper van with a distinctive design, including a camper section on top, parked on a white background.

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

3DTopia -- A small white boy wearing a blue shirt, depicted as a 3D rendering, standing on his tiptoes with arms stretched out.

3DTopia -- A brown cone-shaped wizard's hat with a pointed top and a unique shape, covered in chocolate coating.

3DTopia -(No Captions)

Kabra -- a cartoon rv with a raft on top

Kabra -- a blue and black cartoon fish.

Kabra -- a cartoon character with white hair

Kabra -- a brown witches hat

MARVEL -- The Rustbucket is a modified RV with a boxy shape, large cargo container on the roof, and heavy-duty wheels. It has multiple windows, doors, and utility panels, and is painted in light beige and brown with metallic green and red highlights.

MARVEL -- The Sorting Hat is a tall, pointed wizard's hat with a wide base. It has a fabric-like texture and is primarily brown. It is used in the Harry Potter series to sort students into their Hogwarts houses.

MARVEL -- Baby Dory is a 3D model with a curved, oval body, a large round head, big pink-eyed circles, a small mouth, a yellow tail fin, and small yellow pectoral fins. The body is light blue with darker blue patterns.

MARVEL -- A chibi-style 3D model of Killua from "Hunter x Hunter." Large, rounded head with white, spiky hair, light purple shirt with dark blue sleeves, grayish-purple shorts, and purple shoes with white soles.

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

Cap3D --A pink cartoon cat with a horn, long tail, cigarette, syringe, bow and arrow, and resembling a toy mouse.

Cap3D -- an Iron Man robot holding various weapons, including a knife, gun, and sword.

Cap3D -- Goofy yellow cartoon dog with a green collar

Cap3D -- an orange cartoon bear-dinosaur hybrid with claws and waving arms.

[Figure 188]

[Figure 189]

[Figure 190]

3DTopia --A silver and gray humanoid robot with a metallic body, standing on one leg, holding a weapon, and ready for action on a white background.

3DTopia -(No Caption)

[Figure 191]

###### 3DTopia --

[Figure 192]

[Figure 193]

3DTopia -(No Caption)

(No Captions)

Kabra -- a cartoon character giving the peace sign

Kabra -- a cartoon pink panther sitting down

Kabra -(No Caption)

Kabra -- a cartoon dog with a green collar

MARVEL -- War Machine has a heavily armored suit with a white faceplate helmet, a chest emblem, and jet thrusters on the backpack. It has weapons like a machine gun and rocket launcher. The armor is metallic, gray, and silver.

MARVEL -- Pluto is a blocky, rectangular 3D character with a round head, black floppy ears, and a red tongue. It has a black oval nose, short stubby legs, a long thin black tail, and a green collar. The model is smooth and low-poly,with bright orange, black, red, and green colors.

MARVEL -- The Pink Panther is a smooth, pink, cartoonlike character with a large head, expressive eyes, a small nose, a wide mouth holding a cigarette, and whiskers. It has a long neck, a compact body, four legs with blacktipped paws, and a long, thin tail.

MARVEL -- The 3D model is of Sid, a sloth from "Ice Age." He has a bulky body, short legs, long arms, and a small, wrinkled head. Sid stands upright with one arm raised, looking cheerful. The model is covered in soft, shaggy fur and is colored orange.

###### Figure 10. Qualitative comparison of 3D annotations across baselines [28, 34, 53] and the proposed MARVEL-40M+ for popular anime, movie, and cartoon characters. MARVEL-40M+ provides more accurate and domain-specific annotations, compared to the baselines.

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

###### Cap3D -- an anteater, meerkat, and zebra.

Cap3D -- Colorful parrot

Cap3D -- A small bird perched on a branch.

Cap3D -- A stylized a fish with a textured body, a prominent red cap, and delicate, translucent fins.

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

3DTopia -- A detailed and realistic representation of a small, vibrant parrot with red, blue, and yellow feathers. It is perched on a branch with spread out wings.

[Figure 206]

3DTopia -- A small, brown animal, possibly a fox or a dog, with a long, slender body, a bushy tail, and standing on its hind legs. It has a pointed snout and a playful or curious appearance.

[Figure 207]

3DTopia -- A small bird with a long beak perched on a wooden branch, displaying detailed and lifelike feather patterns and body structure.

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

3DTopia -- A white and red fish with a unique shape, long curved body, small head, and a red cap swimming in the water.

[Figure 212]

[Figure 213]

[Figure 214]

Kabra -- a colorful parrot

Kabra -(No Caption)

Kabra -- a black bird perched on a branch

Kabra -(No Caption)

MARVEL -- The 3D model of the parrot has a streamlined body, a robust head with a pronounced beak and large, black eyes, large curved wings, and long, tapered tail feathers. It is richly colored, with a red head, white beak, colorful chest, blue and yellow wings, and red and blue tail feathers. The legs are rough, and the claws are hard and pointed. The background is black, and the parrot is shown from multiple angles.

[Figure 215]

[Figure 216]

MARVEL -- The Hairy Woodpecker is a compact, elongated bird with a stout beak, broad wings, and a fan-like tail. It has earthy colors, a lighter gray chest, and a reddish patch behind the eyes. The bird is perched on a brown branch.

MARVEL -- The 3D model is a mongoose with an elongated, streamlined body, a rounded snout, and a long, slender tail. It has light brown fur with darker brown stripes along its back and short legs. The fur is natural and matte.

[Figure 217]

[Figure 218]

[Figure 219]

MARVEL -- A 3D model of a Redcap Oranda goldfish with a rounded body, bright red cap, and delicate, feathery fins. It has detailed eyes and a flowing tail fin.

[Figure 220]

Cap3D -- a black bird, possibly an eagle, perched on various surfaces including a wooden table, board, block, and cardboard box.

Cap3D -- a crab with two legs and claws.

Cap3D -- a stegosaurus dinosaur with its mouth open.

###### Cap3D -- a shark in black and white.

[Figure 221]

[Figure 222]

[Figure 223]

3DTopia -- A detailed and realistic 3D model of a gray and white shark swimming in the water. It has a long, pointed snout, sharp teeth, and a sleek and streamlined body. The shark also has visible dorsal and tail fins.

###### 3DTopia --

3DTopia -- A brown crab-like object with a round body, multiple legs, and claws on its front legs, positioned in a relaxed manner.

[Figure 224]

[Figure 225]

A detailed and realistic 3D model of a Velociraptor, a small agile predator with a slender body, long tail, and sharp claws.

3DTopia --A detailed and realistic blue parrot sculpture with spread wings, perched on a wooden surface.

[Figure 226]

[Figure 227]

[Figure 228]

Kabra -- (No Captions)

Kabra -- a statue of a bird on a table

Kabra -- a crab

[Figure 229]

Kabra -- a black shark

[Figure 230]

MARVEL -- The 3D model is a Dodo bird with a stout, rotund body, large hooked beak, and stubby wings. It has robust, yellowish legs and a plump, nonaerodynamic form. The texture is rough and fibrous, and the colors are mainly brown and grey.

MARVEL -- The Hammerhead Shark has a unique hammershaped head, a slender body, and a strong tail with a caudal fin. It has a prominent dorsal fin and smaller pectoral fins. The model is smooth and dark gray.

MARVEL -- The Spinosaurus is a large, elongated dinosaur with a long, narrow head, strong hind legs, and a prominent sail on its back. It has a rough, scaly skin and a muted gray color.

MARVEL -- A 3D model of a crab with a rounded body, broad shell, and eight legs, including two large claws. It has short antennae and a reddish-brown color with realistic shading and highlights.

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

Cap3D -- a palm tree on a sandy island.

###### Cap3D -- a white pepper.

###### Cap3D-- white lotus flowers in a vase.

Cap3D -- blue flowers in a glass vase.

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

3DTopia -- A lifelike, tall and slender palm tree with a single trunk, a fan-shaped crown of leaves, and a few branches extending from the top, placed on a sandy beach.

3DTopia -- A fresh and ripe green bell pepper with a round shape, slightly flattened top and bottom, vibrant coloring, and a small opening at the top.

3DTopia -- A transparent cylindrical vase with a flat bottom and curved top, filled with blue and pink flowers.

3DTopia -(No Captions)

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

Kabra -- a plant with yellow flowers and green leaves.

Kabra -- a vase filled with purple flowers

Kabra -- a green pepper.

Kabra -- a palm tree sitting on top of a sandy beach.

MARVEL -- The bell pepper is a smooth, teardropshaped model with three lobes, a short stem, and a color gradient from green to red/orange. It has a glossy, slightly waxy surface with minor imperfections.

MARVEL -- A 3D model of a Marsh Marigold with yellow flowers and green, heart-shaped leaves. The flowers have five petals and a center with stamens. Slender stems support the leaves and flowers. The model is bright and simple.

MARVEL -- A clear glass jar with a floral arrangement of hydrangeas, daisies, and lily of the valley in pastel colors. The jar is glossy, and the flowers are matte.

MARVEL -- A tall, slender coconut tree with curved trunk, large green leaves forming a canopy, and coconuts at the base, standing on a sandy platform.

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

Cap3D -- a purple bowl filled with colorful candy resembling an animal cell.

Cap3D -- a molecule with red, blue, and white spheres.

Cap3D -- Purple and white molecule with two attached spheres

Cap3D -- a green and purple kidney with a cane, frog, pipe, leaf, intestine, and flower.

[Figure 252]

[Figure 253]

3DTopia -- A light purple curved bowl or dish with a flat bottom and a raised rim, filled with various food items including fruits, vegetables, and other ingredients.

3DTopia -(No Caption)

3DTopia -(No Caption)

###### 3DTopia --

(No Captions)

Kabra -- a molecular structure

Kabra -- a molecular structure

Kabra -- a purple and green object

###### Kabra -- a cell

MARVEL -- The 3D model is a eukaryotic cell, shaped like an ellipsoid .... purple cell membrane, a pink/red nucleus with a yellow core ... The endoplasmic reticulum (ER) surrounds the nucleus, with rough ER (greenish-yellow with dots) and smooth ER (green). The Golgi apparatus, orange/yellow, is a stack of pouches. Mitochondria are red and sausage-shaped, and lysosomes are smaller vesicles in yellow and reddish-brown. ...

MARVEL -- A 3D model of a water molecule (H2O) with a central purple oxygen atom and two smaller white hydrogen atoms, connected by white sticks with red accents. The atoms form a trigonal planar shape with a bond angle of about 104.5 degrees.

MARVEL -- Vitamin B2 (Riboflavin) is a complex molecule with a central ring system and extending side chains. It has an elongated, irregular shape and uses color coding to differentiate atoms: carbon (grey/black), oxygen (red), nitrogen (blue), and hydrogen (white/grey).

MARVEL -- The 3D model shows the aorta and two kidneys. The aorta is a long, curved tube, and the kidneys are oval shapes on either side. The aorta is purple, the left kidney is green, and the right kidney is teal.

- Figure 11. Qualitative comparison of 3D annotations across baselines [28, 34, 53] and the proposed MARVEL-40M+ for biological objects, including animals, plants, and molecular models. MARVEL-40M+ provides more accurate and domain-specific annotations, compared to the baselines. Incorrect captions are highlighted in red, while important captions are highlighted in green.

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

###### Cap3D -- two bananas with a banana peel.

Cap3D -- sushi rolls and chicken skewers on a wooden cutting board.

Cap3D -- Indonesian instant Mi Goreng noodles with an egg on the package.

Cap3D -- a birthday cake with a candle.

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

3DTopia -- A detailed and realistic representation of a yellow, curved banana, sitting on a white background.

[Figure 263]

3DTopia -- A rectangular white package made of cardboard or similar material, with a folded appearance, likely used for packaging food items, specifically a pack of noodles.

3DTopia -- A wooden plate with a visually appealing arrangement of various sushi rolls, including avocado, placed neatly on it.

[Figure 264]

[Figure 265]

3DTopia -(No Caption)

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

Kabra -- a chocolate cake with a single candle on it

Kabra -- a bunch of bananas

Kabra -- the back of a package of instant noodles

Kabra -- three sushi rolls on a wooden tray

MARVEL -- A bunch of three ripe bananas connected by a thick, brown stem and supported by a white circular base plate. The bananas are slightly curved and have a realistic skin texture with dark spots. The stem is rough and woody, and the base plate is smooth and reflective.

MARVEL -- The Indomie Instant Noodles package is a rectangular box with rounded corners, featuring the brand logo "Indomie" an image of cooked noodles, and cooking instructions. It is smooth and glossy, mainly in white, green, yellow, red, and blue.

MARVEL -- Three pieces of salmon nigiri sushi on a wooden board, with a small bowl of soy sauce. The sushi has oval rice balls topped with pink salmon, green and red garnishes. The wooden board has rounded edges, and the soy sauce bowl is circular and matte. A small green spot is on the board.

MARVEL -- A chocolate birthday cake with a single lit candle and the text "Felicidades Dani." The cake is circular, topped with cream frosting and sprinkles, and a white candle with a small flame.

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

Cap3D -- Triangular prism-shaped chocolate bar package with the brand logo and mountain graphic design.

[Figure 274]

Cap3D -- a rabbit with a red collar, resembling a chocolate Easter bunny.

Cap3D --a blue and green

[Figure 275]

star-shaped object with a hole, featuring sword and knife elements.

[Figure 276]

###### Cap3D -- two Coca Cola cans.

[Figure 277]

[Figure 278]

3DTopia -- A visually appealing and eye-catching metallic star-shaped object, made of wood and metal, with a sharp and pointed tip, a circular base, and multiple points radiating outwards from the center.

[Figure 279]

[Figure 280]

3DTopia -(No Caption)

3DTopia -(No Caption)

[Figure 281]

3DTopia -(No Caption)

Kabra -- a chocolate bunny with a red ribbon around its...

Kabra -- a can of coke and a can of sprite

[Figure 282]

Kabra -- a five pointed star with a hole in the middle

Kabra -- a toblerone candy bar

[Figure 283]

MARVEL -- Two soda cans, one red Coca-Cola standing upright and one green-to-blue Sprite lying on its side, partially overlapping. Both have smooth, glossy finishes and metallic pull tabs.

MARVEL -- A 3D model of a Lindt Gold Chocolate Bunny, sitting with a shiny gold wrapper, red ribbon, and white waxed paper at the bottom. The colors are gold, red, and brown.

MARVEL -- A four-pointed throwing star with sharp, curved tips and a central hole. The surface is rough and dark blue, with a thin red line.

MARVEL -- A Toblerone chocolate bar box, long and narrow with a tapered end, featuring Swiss mountains and a bear. Smooth paperboard, white with red and gold accents.

[Figure 284]

[Figure 285]

Cap3D -- H&M shopping bag with the word 'hm' on it.

Cap3D -- A gray sports shoe with distinct brand logo, white laces, and padded collar.

Cap3D -- Puffy jacket with monogram pattern, featuring a zippered front and elastic cuffs and hem.

[Figure 286]

Cap3D -- Nivea tube containing

[Figure 287]

toothpaste, hand cream, facial cleanser, and SPF 50 body lotion.

[Figure 288]

[Figure 289]

[Figure 290]

3DTopia -- A white rectangular shopping bag with a flat bottom, open top, and a red H&M logo. It has a handle for easy carrying and appears to be empty.

[Figure 291]

[Figure 292]

[Figure 293]

3DTopia -- A brown teddy bear with a soft, plush texture, cuddly appearance, and round shape.

3DTopia -(No Caption)

3DTopia -- A cylindrical white and blue Nivea cream tube with a label, standing upright.

[Figure 294]

[Figure 295]

[Figure 296]

Kabra -- a grey new balance shoe

Kabra -- a bottle of nivea hand cream

Kabra -- a jacket with a pattern on it

Kabra -- a gray shopping bag with red letters on it

MARVEL -- A pair of New Balance sneakers with a low-top design, made of suede-like material with white laces and a beige "N" logo on the side. The shoe has a cushioned white tongue and a light gray foam midsole for comfort, with a rubber outsole for traction.

[Figure 297]

MARVEL -- A 3D model of a Nivea hand cream tube, cylindrical with a narrow top, smooth plastic, white with blue and red accents, and a glossy logo.

MARVEL -- The Louis Vuitton Puffer Jacket is a rectangular, quilted jacket with a high collar and two sleeves. It is light beige with dark brown logos and is made of lightweight material for outdoor use.

MARVEL -- A rectangular shopping bag with the H&M logo in bold red on one side. It has a flat base and two handles, and is primarily light gray.

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

Cap3D -- a blue and white computer, blue and yellow laser printer, Sony tape recorder, blue cassette player, and

###### Cap3D -- iPhone X, iPhone 11 Pro, and iPhone XS Max royalty-free.

Cap3D -- a Samsung coffee machine, black washing machine, water dispenser, and small printer.

Cap3D -- a yellow and black screwdriver.

[Figure 305]

###### Nintendo DSi.

[Figure 306]

3DTopia -(No Caption)

3DTopia -(No Caption)

3DTopia -- A small, rectangular blue box with a hinged lid, designed to hold and store various items.

[Figure 307]

3DTopia -(No Caption)

[Figure 308]

[Figure 309]

Kabra -- the back of an apple iphone

Kabra -- a black object

Kabra -- a black and yellow stanley knife

Kabra -- a sony walkman

MARVEL -- The Sony Walkman TPS-L2 is a small, portable cassette player with a rectangular shape and rounded edges. It has a display window, control buttons, and a cassette slot. The device is light blue with white sides and orange accents. Colorful stickers with cartoon characters add a playful look.

MARVEL -- The Siemens Coffee Machine is a sleek, rectangular device with a bean hopper on top, a control panel below, a transparent water tank on the left, and a metallic drip tray at the bottom. It is mostly black with a

MARVEL -- The iPhone 13 is a rectangular smartphone with a large, smooth screen and a matte back. It has a diagonal dual-camera setup and buttons on the sides. The phone is in a neutral metallic color.

MARVEL -- A utility knife with a curved, rubberized handle, a rectangular blade compartment, and a flat safety lock. The handle is black with yellow accents and a white "STANLEY" logo. The blade compartment has a metallic texture.

transparent tank and a silver tray.

###### Figure 12. Qualitative comparison of 3D annotations across baselines [28, 34, 53] and the proposed MARVEL-40M+ for diverse items including daily objects, essentials. MARVEL-40M+ provides more accurate and domain-specific annotations, compared to the baselines.

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

Cap3D -- a small white marble stone fountain with a statue, turrets, and a window.

Cap3D -- a monument surrounded by plants, trees, and grass in a park setting.

Cap3D -- Brown reclining figure sculpture with rough textures on a rustic wooden plank base.

Cap3D -- a ruined ancient building, a floating boat, and a military tank

[Figure 316]

[Figure 317]

[Figure 318]

3DTopia -- A tall, rectangular stone monument with a pointed top and wider base, situated on a grassy area surrounded by trees and greenery. It appears to be a memorial or commemorative structure.

3DTopia -- A white stone-like rectangular fountain or basin, with intricate carvings and decorations, used for outdoor landscaping or as a focal point in a garden or park.

3DTopia -- A lifelike wooden sculpture of a person lying in a bed, intricately carved and realistically detailed.

[Figure 319]

3DTopia -(No Captions)

[Figure 320]

[Figure 321]

[Figure 322]

Kabra -- a statue of a person laying on a bed.

Kabra -- a ruined building

K

K

Kabra -- a statue

K

Kabra -- a statue

K

MARVEL -- A 3D model of an Old Kingdom Period mummy from Abydos, lying in a fetal position inside a rectangular wooden sarcophagus. The mummy is wrapped in linen bandages, and the head is covered separately. The sarcophagus has a smooth wooden texture. The colors are earthy tones like browns and tans.

MARVEL -- The Fountain of Peirene is an ancient Roman structure with a central pool, stone steps, and uneven walls. It has an arched entrance and is made of weathered limestone with a rough texture. Located in Corinth, it reflects the historical importance of the site.

MARVEL -- The Spomenik na Grdelici is a World War II memorial with two tall stone slabs, one taller than the other. It sits on a flat stone platform surrounded by grass and small shrubs. The design is simple and somber.

MARVEL -- The "Pilar Puert del Perdón" is a decorative water fountain with a rectangular basin, intricate carvings, and three spires. It's attached to a building and made of rough stone and white marble.

[Figure 323]

Cap3D -- a woman's head with a bun, resembling an Egyptian statue.

[Figure 324]

Cap3D --a cloaked man's bust.

[Figure 325]

Cap3D -- marble bust of a Roman emperor with long hair.

Cap3D -- a man riding a white winged horse, equipped with a sword and shield.

[Figure 326]

[Figure 327]

[Figure 328]

3DTopia -- A realistic and lifelike bust of a man, likely made of stone or marble, with a serious expression, detailed and expressive face, displayed on a pedestal.

3DTopia -(No Caption)

3DTopia -- A detailed and realistic clay sculpture of a woman's head with intricate features and facial details.

[Figure 329]

[Figure 330]

3DTopia -(No Caption)

[Figure 331]

[Figure 332]

Kabra -- a person riding a horse with wings

Kabra -- a bust of a man

Kabra -- a bust of a man

Kabra -- a statue of a woman 's head

MARVEL -- The 3D model shows Perseus, a human figure in ancient Greek armor, riding a white winged horse named Pegasus. They appear to be flying, with Perseus holding a sword and shield. The horse has smooth white fur and feathered wings.

MARVEL -- The 3D model is a bust of William Shakespeare, featuring a detailed face, beard, and draped fabric with lace details. The base is cylindrical, and the skin and fabric have natural colors.

MARVEL -- A bust of Alexander the Great with detailed facial features, wavy hair, and a draped garment. Smooth, polished surface, light beige color.

MARVEL -- A 3D model of a terracotta warrior head with a realistic face, tight bun of hair, and a band around the forehead. The texture is rough and the color is dark, earthen brown. Suitable for historical displays.

- Figure 13. Qualitative comparison of 3D annotations across baselines [28, 34, 53] and the proposed MARVEL-40M+ for historical elements including statues, places, memorials, etc. Incorrect captions are highlighted in red, while important captions are highlighted in green.

MARVEL -- A small tropical island with sand, three palm trees, a red and white beach umbrella, a blue beach chair, a wooden boat, and a few other items like a table and a cooler. The setting is peaceful and perfect for relaxation.

3DTopia -(No Caption)

Kabra -- a small island with a boat and umbrella

Cap3D -- a small

island with palm trees and a boat.

[Figure 333]

[Figure 334]

MARVEL -- Mount Toratau is a 3D model of a mountain with a conical peak, snow and rock at the top, and green vegetation at the base. The slopes are rocky and steep, transitioning to flatter, grassy areas. The colors are white/gray, green, and brown.

Cap3D -- a small green mountain

3DTopia -- A small green grassy mountain or hill with a rounded shape, situated on a green field.

Kabra -- a green mountain

[Figure 335]

[Figure 336]

MARVEL -- A 3D model of a forest reserve within an urban campus, featuring a dense forest, pathways, and modern buildings. The forest is lush and green, while the buildings are in neutral tones. The terrain is gently undulating, and there are paved pathways.

Cap3D --a small island town with trees and buildings, resembling a park.

3DTopia -- A small white house or building with a rectangular shape and a flat roof, situated on a small rectangularshaped piece of land with a flat surface.

Kabra -- an island with trees and buildings on it

[Figure 337]

[Figure 338]

MARVEL -- The 3D model shows the Campo del Sur area in Cádiz, with a large cathedral, surrounding buildings, a coastal road, and a sandy or rocky shore. The colors are mostly light and neutral, with some green and blue from vegetation and the ocean.

3DTopia -- A detailed and realistic 3D model of a coastal city with a mix of modern and traditional architectural styles, including a church and a castle, situated on a rocky terrain.

Kabra -- a city

Cap3D -- a city with buildings surrounded by the ocean.

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

MARVEL -- A colorful flamingo stands in a tranquil pond, surrounded by green lily pads with white flowers and tall grasses. The pond water is glossy, and the flamingo has a rainbow gradient on its wings. The base is flat and brownish, creating a peaceful, natural scene.

3DTopia -(No Caption)

Kabra -- a statue of a bird with rainbow colored wings

Cap3D -- a unicorn and a flamingo in a pond, with a colorful rainbow bird flying above.

[Figure 347]

[Figure 348]

MARVEL -- The Château de La Varenne is a large rectangular building with two cylindrical towers, a formal garden in front, and a landscaped area around it. It has evenly spaced windows and a grey slate roof.

Cap3D -- Classical twostory chateau with slate blue mansard roofs and prominent chimneys, set on a landscaped plot with patches of greenery and a conspicuous rock formation. The building features numerous windows with white facades and architectural detailing.

3DTopia -- A large, white luxury house with a tall, slender tower and a large, round tower on top, situated on a green lawn.

Kabra -- a castle

[Figure 349]

[Figure 350]

MARVEL -- Marksburg Castle is a medieval fortress with a central tower, surrounded by smaller towers and walls. It has a spacious courtyard and is made of rough stone with slate roofs. The castle is surrounded by a dense forest and sits on a hilltop, accessible via a winding path.

Cap3D -- a small castle on an island with a hill and surrounded by trees.

3DTopia -- A small white castle with a tower on top, situated on a small island surrounded by water and trees. The castle is a detailed plastic model with intricate design, resembling a medieval fortress

Kabra -- a castle in the middle of a forest

[Figure 351]

[Figure 352]

MARVEL -- The 3D model shows the Glaciar Perito Moreno in Santa Cruz, Argentina. It features jagged mountains, a large turquoise lake, and ice formations extending into the water. The surrounding area has green grass and brown rocks. Rivers flow into the lake from the mountains. The scene is natural and isolated, highlighting the glacier's grandeur.

3DTopia -- A large, rectangular topographic map displaying a highly detailed and accurate representation of a mountainous terrain, including snow-capped peaks and a lake.

Kabra -- a landscape with mountains and a river

Cap3D -- a mountain range with a lake.

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

- Figure 14. Qualitative comparison of 3D annotations across baselines [28, 34, 53] and the proposed MARVEL-40M+ for diverse scenes including digital elevation maps, places, realistic or animated scenes. Incorrect captions are in red, while important captions are in green.

[Figure 362]

[Figure 363]

Blue hair, red eyes, anime style, blue-to-white hoodie, black shorts, knee-high socks, multicolored sneakers, neutral expression, smooth skin, matte hoodie, shiny shoes.

NEARCHAN is a 3D anime character with blue hair, red eyes, a blue-to-white hoodie, black shorts, and multicolored sneakers.

NEARCHAN is a 3D model of an anime-style character with blue, short hair and large red eyes. She wears a blue-to-white hoodie with a hood, black shorts, and knee-high socks with blue cuffs. Her hands are open, and she has multicolored sneakers.

NEARCHAN is a 3D model of an anime-style character with a large head and exaggerated proportions. She has blue, short bob cut hair and large, red eyes. Her torso is covered in a hoodie that transitions from blue at the top to white at the bottom, with a hood resting on her head. Her arms extend downward, with hands slightly away from the body and fingers spread. She wears black shorts and knee-high socks with blue cuffs. Her feet are in multicolored sneakers. The skin is smooth, the hoodie is matte, and the shoes are shiny.

NEARCHAN is a 3D model of an anime-style character with a humanoid form and exaggerated proportions. Her head is larger than her body, featuring blue, short bob cut hair with straight bangs. Her anime-styled head includes large, red eyes and a neutral expression. The neck is slim, connecting to a torso clad in a hoodie jacket that transitions from blue at the top to white at the bottom. The hood rests on her head, adding to the casual look. Her upper torso is light blue, while the lower torso is white with no distinct patterns. The arms extend downward from the shoulders, with hands positioned slightly away from the body, fingers spread out in an open gesture. She wears black shorts over knee-high socks, with blue cuffs and black fabric above. Her feet are adorned with multicolored sneakers featuring blue, pink, and light green. The skin is smooth and slightly glossy, the hoodie has a matte finish, and the shoes have a shiny appearance.

[Figure 364]

[Figure 365]

Elongated body, striped abdomen, semi-transparent wings, red eyes, thin legs, yellow stinger, metallic sheen.

A 3D wasp with a long, striped body, large transparent wings, thin dark legs, and a yellow stinger.

The 3D model is a wasp with a long, striped abdomen, small head, and thorax. It has large, see-through wings with vein patterns, thin, dark legs, and a yellow stinger at the end of the abdomen. The head features red eyes, and the body has a smooth texture with a metallic sheen on the legs.

The 3D model represents a stylized wasp with an elongated body divided into a head, thorax, and abdomen. The head and thorax are small and dark brown, with the head featuring red eyes. The abdomen is long and striped with dark brown and tan, tapering at the end. Large, semi-transparent wings with visible veins extend from the thorax, and the legs are thin and dark with a metallic sheen. A pointed yellow stinger is located at the tip of the abdomen.

The 3D model of a wasp features an elongated body divided into three main sections: head, thorax, and abdomen. The head is small with smooth, dark brown surfaces and red eyes. The thorax, similarly smooth and dark brown, supports two pairs of semi-transparent wings with intricate veining patterns. The abdomen is long and slender, tapering at the end with a slight upward curve, and is marked by alternating dark brown and tan stripes, creating a segmented appearance. The wings are large relative to the body, reflecting light differently and adding depth. The legs are thin, jointed, and have a dark metallic sheen, while the stinger at the end of the abdomen is pointed and yellow, contrasting sharply against the darker segments.

- Figure 15. Multi-level annotation examples of MARVEL for the Objaverse [18] dataset. Words corresponding to Object and Components are highlighted in violet, Shape and Geometry in green, Texture and Materials in orange, Colors in blue, and Contextual Environment in purple. From top to bottom, we go from level-5 (Concise Tags) captions to level-1 (Comprehensive Description) captions.

[Figure 366]

Tennis paddle. circular or oval blade, smooth and glossy surface, dark brown handle, matte finish, gradient brown hues, "TABLE TENNIS MATCH" text, cat's face illustration, slight edge curvature.

[Figure 367]

A table tennis paddle with a smooth, brown blade featuring a cat's face and the text "TABLE TENNIS MATCH" and a dark brown, matte handle.

The 3D model is a table tennis paddle with a circular or oval blade and a handle. The blade is smooth and glossy, with a gradient of brown hues, darker at the edges and lighter in the center. The text "TABLE TENNIS MATCH" is in white with a blue shadow, and there's a cat's face illustration in orange and white. The handle is dark brown and matte.

The 3D model is a table tennis paddle with a blade and a handle. The blade is circular or oval, slightly curved at the edges for better control. The blade has a smooth, glossy surface, contrasting with the matte handle. The blade features a gradient of brown hues, with darker edges and lighter centers. The text "TABLE TENNIS MATCH" is prominently displayed in white with a blue shadow, and there's a central illustration of a cat's face in orange and white. The handle is dark brown.

The 3D model represents a table tennis paddle, consisting of a blade and a handle. The blade is circular or oval, with a symmetrical design and a slight curvature at the edges, enhancing maneuverability. The proportions ensure a wide striking surface. The blade is smooth and glossy, likely made of high-quality wood laminate, while the handle has a matte finish, indicating a grip material over a wooden core. The blade features a gradient of brown hues, darker at the edges and lighter towards the center. The text "TABLE TENNIS MATCH" is displayed in white uppercase letters with a blue shadow, adding a 3D effect. An illustration of a cat's face, using orange and white, is centrally located. The handle is uniformly dark brown, matching the blade's darker edges.

[Figure 368]

[Figure 369]

Round shape, wide opening, tapered base, glossy finish, white base color, central strawberry design, scattered colorful dots, thin orange rim.

A round ceramic bowl with a wide opening, glossy white surface, central strawberry design, and colorful dots. The rim has a thin orange border.

The ceramic bowl is round with a wide opening that tapers down to a smaller base. It has a smooth, glossy finish and is primarily white. A central strawberry design in red and green is featured at the bottom, with scattered colorful dots. The rim is outlined with a thin orange border.

The ceramic bowl has a smooth, symmetrical shape with a wide circular opening that narrows towards the base. Both the interior and exterior follow a gentle conical curve, creating a balanced form. The bowl is made of ceramic or porcelain and has a glossy, reflective finish. The base color is white, with a central strawberry design in red and green, and scattered decorative dots in orange, green, and yellow. The rim is accented with a thin orange border.

The ceramic bowl features a smooth, symmetrical shape with a wide circular opening that tapers down to a narrower base. The interior is conically shaped, transitioning smoothly from the rim to the base. The exterior mirrors this curvature, creating a balanced and harmonious form. The bowl is made of ceramic or porcelain, with a glossy finish that reflects light, giving it a smooth, polished appearance. The base color is white, serving as a neutral backdrop for the decorative elements. A central strawberry design, rendered in red and green, is prominently featured at the bottom, surrounded by scattered dots and splashes in orange, green, and yellow, adding a playful touch. The rim is highlighted with a thin orange border, framing the bowl elegantly.

- Figure 16. Multi-level annotation examples of MARVEL for the Omni-Object [80] dataset. Words corresponding to Object and Components are highlighted in violet, Shape and Geometry in green, Texture and Materials in orange, Colors in blue, and Contextual Environment in purple. From top to bottom, we go from level-5 (Concise Tags) captions to level-1 (Comprehensive Description) captions.

[Figure 370]

[Figure 371]

Biplane, stacked wings, circular insignias, elongated fuselage, propeller, landing gear, horizontal tailplane, vertical stabilizers, green body, blue and red markings, silver engine.

A classic biplane with two sets of wings, a long cylindrical body, and a propeller at the front. It has blue and red wing insignias, two small wheels, and a silver engine.

The 3D model is a biplane with two sets of wings stacked on top of each other. The fuselage is long and cylindrical, narrowing at the back, with a propeller at the front. The wings have blue circular insignias with red dots. The landing gear consists of two small wheels under the fuselage. The control surfaces include a horizontal tailplane and vertical stabilizers at the rear. The plane is mainly green, with blue and red wing markings and a silver engine and propeller.

The 3D model is a biplane with a traditional layout. It features two sets of horizontally stacked wings, each adorned with circular blue insignias and red dots. The fuselage is elongated and cylindrical, tapering towards the rear, with a cylindrical engine and propeller at the front. The landing gear includes two small wheels at the bottom center of the fuselage. Control surfaces consist of a horizontal tailplane and vertical stabilizers at the rear. Struts and braces provide structural support between the upper and lower wings. The aircraft is primarily green, with blue and red wing insignias and a silver engine and propeller.

The 3D model represents a classic biplane with a detailed and symmetrical design. The main wings are horizontally stacked in a biplane configuration, each featuring circular blue insignias with red dots at their centers. The fuselage is elongated and cylindrical, tapering slightly towards the rear, with a cylindrical engine mounted on top and a prominent propeller at the front. The landing gear consists of two small wheels positioned near the bottom center of the fuselage. The control surfaces include a horizontal tailplane and vertical stabilizers at the rear. Struts and braces connect the upper and lower wings for structural support. The wings are rectangular with rounded tips, and the fuselage and engine housing exhibit a semi-matte finish, simulating lightweight materials like aluminum or wood. The engine and propeller blades have a metallic, slightly glossy appearance. The primary color of the aircraft is green, with blue and red accents on the wing insignias and silver for the engine and propeller.

[Figure 372]

[Figure 373]

Hammock, fabric body, wooden spreader bars, black pillow, vibrant colors, woven pattern, ropes, hooks, outdoor use, rectangular form, bilateral symmetry.

A colorful hammock with wooden spreader bars, a black pillow, and ropes for hanging. Suitable for outdoor relaxation.

The 3D model is a hammock with a colorful fabric body, two wooden spreader bars, and a black pillow. The fabric is an elongated rectangle that curves slightly and features vibrant patterns. The spreader bars keep the hammock open and rigid. The pillow, attached at one end, is soft and comfortable. Ropes and hooks are provided for hanging the hammock. The spreader bars are wood-colored, and the pillow is brownish-yellow. It is designed for outdoor use.

The 3D model is a hammock with a main fabric body, two wooden spreader bars, and a black pillow. The fabric is an elongated rectangle that curves slightly, with a woven pattern and vibrant colors including red, yellow, pink, and green. The spreader bars maintain the rectangular form and provide rigidity. The pillow, attached at one end, adds comfort. Ropes and hooks are included for secure suspension. The spreader bars are a natural wood color, and the pillow is brownish-yellow. The design is suitable for outdoor settings.

The 3D model represents a hammock with a main fabric body, two wooden spreader bars, and a black pillow. The fabric is an elongated rectangle that curves slightly, with a woven pattern suggesting a synthetic blend. The spreader bars, attached at the ends, maintain the rectangular form and provide rigidity. The hammock exhibits bilateral symmetry along its central axis, with open ends revealing the inner fabric and attachment points. The pillow, attached at one end, has a plush texture. Ropes and hooks are included for suspension, likely made of nylon or polyester. The fabric features vibrant red, yellow, pink, and green patterns, with the spreader bars in a natural wood color and the pillow in a brownish-yellow hue. The design is versatile for outdoor use, such as patios, gardens, or camping.

- Figure 17. Multi-level annotation examples of MARVEL for the ShapeNet [10] dataset. Words corresponding to Object and Components are highlighted in violet, Shape and Geometry in green, Texture and Materials in orange, Colors in blue, and Contextual Environment in purple. From top to bottom, we go from level-5 (Concise Tags) captions to level-1 (Comprehensive Description) captions.

[Figure 374]

[Figure 375]

Triceratops, large head, eye horns, nasal horn, frill with spikes, robust body, strong legs, long tapering tail, rough scaly skin, green body, lighter horns.

The 3D model is a Triceratops, a large dinosaur with a big head, two eye horns, a nasal horn, and a frill with spikes. It has a robust body, strong legs, and a long, tapering tail. The skin is rough and scaly, with a green body and lighter horns.

The 3D model is a Triceratops, known for its large head with two eye horns and a big nasal horn, and a frill with spikes. The body is robust and supported by strong legs. The tail is long and tapers. The skin is rough and scaly, with a green body and lighter horns. This model captures the essential features of a Triceratops, making it suitable for both educational and creative projects.

The 3D model is a Triceratops, featuring a large head with two eye horns and a prominent nasal horn, set against a frill with spikes. The body is robust and tapers into a shorter tail. Strong, sturdy legs support the heavy frame. The skin is rough and scaly, with raised bumps along the back. The horns and frill are smooth and slightly glossy, contrasting with the green body and lighter horns.

The 3D model represents a Triceratops, a large ornithopod dinosaur. The head features two prominent horns above the eyes and a larger nasal horn, with a frill at the back adorned with spikes. The body is bulky and robust, transitioning into powerful hindquarters and a relatively short but thick neck. The tail is long and tapers backward, providing balance. The skin texture is rough and scaly, with raised bumps along the back, suggesting osteoderms. The horns and frill have a smooth, slightly glossy surface, possibly covered with keratinous material. The main body color is predominantly green with darker patches, while the horns and frill are a lighter, almost white shade.

[Figure 376]

[Figure 377]

Pug, traffic cone hat, round body, folded ears, small round eyes, flat snout, short sturdy legs, curled tail, bright yellow, brown eyes, matte texture, yellow and white stripes.

A pug dog with a round body and a traffic cone hat. The pug has folded ears, small round eyes, and a flat snout. The hat is yellow with white stripes. The body is bright yellow with brown eyes and nose. The texture is matte.

The 3D model is a pug dog wearing a traffic cone hat. The pug has a round, short body with a broad face and folded ears. Its eyes are small and round, and the snout is flat and large. The legs are short and sturdy, and the tail is small and curled. The traffic cone hat is yellow with white stripes and fits around the pug’s head. The pug’s body is mostly bright yellow, with brown areas for the eyes and nose. The texture is matte, and the model is simple.

The 3D model depicts a pug dog wearing a traffic cone hat. The pug has a rounded, short-statured body with a broad face and compact build. Its ears are folded downward, and the eyes are small and round, positioned slightly above mid-face level. The snout is flat and large, typical of pugs. The legs are short and sturdy, and the tail is small and curled up over the back. The traffic cone hat is conical with alternating yellow and white stripes, fitting around the pug’s head. The pug’s body is primarily bright yellow, with brownish areas for the eyes and nose. The texture is matte, and the model is in a low poly style.

The 3D model represents a pug dog adorned with a traffic cone hat. The pug's body is rounded and short-statured, with a broad face and compact build. The ears are folded downward, close to the sides of the face. The eyes are small and round, positioned slightly above mid-face level, with a glossy appearance, possibly indicating a glass or polished plastic material. The snout is flat and large, characteristic of pugs. The legs are short and sturdy, supporting the round body, while the tail is small and curled up over the back. The traffic cone hat is conical, featuring alternating yellow and white stripes, with a circular base that fits snugly around the pug’s head. The pug’s body is a uniform bright yellow, with brownish areas for the eyes, nose, and some parts of the face. The texture of the pug and the cone hat is matte, with no shiny highlights or reflections. The model is designed in a low poly style, with subtle geometric facets.

- Figure 18. Multi-level annotation examples of MARVEL for the Toys4K dataset. Words corresponding to Object and Components are highlighted in violet, Shape and Geometry in green, Texture and Materials in orange, Colors in blue, and Contextual Environment in purple. From top to bottom, we go from level-5 (Concise Tags) captions to level-1 (Comprehensive Description) captions.

Blue ceramic cup, embossed flower patterns, tapered shape, smooth interior, deep blue color, lighter interior, gradient effect, symmetrical design, stable base.

[Figure 378]

[Figure 379]

A blue ceramic cup with embossed flower patterns on the outside and a smooth interior. It tapers from a wide top to a narrow base and is primarily deep blue with a lighter interior.

The 3D model is a blue ceramic cup with flower patterns. It has a standard cup shape, tapering slightly from a wide top to a narrow base. The exterior is textured with embossed flowers, while the interior and base are smooth. The cup is primarily deep blue, with a lighter shade inside. The flowers have a subtle gradient effect.

The 3D model is a blue ceramic cup with a flower pattern. It has a standard cup shape, slightly tapering from a wide opening at the top to a narrower base at the bottom. The exterior features embossed flower patterns, evenly distributed around the cylindrical body, creating a textured look. The interior and base are smooth, making it easy to clean and stable on surfaces. The primary color is deep blue, with a lighter shade inside. The flower patterns have a subtle gradient effect, with darker centers and lighter petals.

The 3D model represents a blue ceramic cup with a flower pattern. The cup has a standard shape, slightly tapering from a wider opening at the top to a narrower base at the bottom, ensuring stability. The geometry is symmetrical along its vertical axis, with equal proportions on all sides. The exterior surface features multiple embossed flower patterns, evenly distributed around the cylindrical body. Each pattern consists of concentric petals radiating outward from a central point, resembling a sunflower. The texture of the exterior is raised, providing a tactile quality. The interior and base surfaces are smooth, facilitating easy cleaning and enhancing stability. The primary color is deep blue, with a lighter shade of blue or off-white on the interior. The flower patterns have a subtle gradient effect, with darker centers and lighter petals, creating a harmonious visual contrast.

[Figure 380]

[Figure 381]

Ergonomic seat, high backrest, headrest, horizontal armrests, five-spoke base, caster wheels, smooth white upholstery, gray metal accents, minimalistic design.

Modern office chair with a curved seat, high backrest, and horizontal armrests. Five-spoke base with wheels. Smooth white upholstery, gray metal accents. Clean, minimalist design.

The office chair has a comfortable, slightly curved seat and a high, curved backrest with a headrest. Armrests are horizontal with slight upward curves. The base has five spokes with wheels. The chair is covered in smooth, white leather-like material with gray metal accents. It has a clean, modern look.

The office chair has an ergonomic design with a slightly curved, rectangular seat that tapers at the front. The high backrest curves gently and includes a headrest. Armrests are positioned mid-width and extend horizontally with slight upward curves. The base features five spokes with caster wheels. The chair is upholstered in a smooth, white leather-like material, with gray accents on the metal parts. The design is clean and minimalistic, ideal for modern offices.

The modern office chair features an ergonomic design with a slightly curved, rectangular seat that tapers at the front for leg comfort. The high backrest provides substantial lumbar support and gently curves from top to bottom, integrating a headrest. Armrests, positioned near the midpoint of the backrest width, extend horizontally with slight upward curves for optimal forearm rest and shoulder alignment. The base consists of five spokes converging into a central hub, each ending in a caster wheel for mobility. The chair is upholstered in a smooth, leather-like material, predominantly white, with subtle gray accents on the metal components, including the base and adjustment mechanisms. The design is minimalistic, with uniform colors and no patterns, making it suitable for modern office settings.

- Figure 19. Multi-level annotation examples of MARVEL for the ABO (Amazon Berkeley Objects) [16] dataset. Words corresponding to Object and Components are highlighted in violet, Shape and Geometry in green, Texture and Materials in orange, Colors in blue, and Contextual Environment in purple. From top to bottom, we go from level-5 (Concise Tags) captions to level-1 (Comprehensive Description) captions.

[Figure 382]

[Figure 383]

Cylindrical shape, beige fabric, horse print, light beige zipper, corner reinforcements, blue tag, flat base, stands upright, vibrant colors, symmetrical design.

A cylindrical pencil case with a beige fabric body and colorful horse print. Features a light beige zipper, corner reinforcements, and a blue tag. Stands upright on a flat base.

The Horse Print Pencil Case is a cylindrical pencil holder with a beige fabric body and a colorful horse print. It has a light beige zipper, corner reinforcements, and a blue tag near the zipper. The flat base allows it to stand upright. The design is clean and functional.

The Horse Print Pencil Case is a cylindrical, symmetrical object with a beige fabric body featuring a vibrant horse print. It includes a light beige zipper mechanism for opening and closing, and corner reinforcements to prevent damage. A blue rectangular tag is attached near the zipper. The flat base allows the case to stand upright.

The Horse Print Pencil Case is a cylindrical, symmetrical object designed to hold writing instruments. It features a beige fabric body with a vibrant horse print pattern, showcasing horses in various poses and colors such as black, brown, white, and blue. The case has a light beige zipper mechanism, likely made of metal with plastic components for ease of use. Corner reinforcements at both ends are made of sturdy material to prevent fraying and tearing, matching the main body's color for a seamless look. A blue rectangular tag, possibly fabric or plastic, is sewn onto one end near the zipper, providing additional branding or information. The flat base allows the case to stand upright, and the overall proportions are consistent throughout its length.

[Figure 384]

Blue, Nintendo 3DS XL, handheld gaming console, rounded rectangle, matte finish, two screens, black bezels, touchscreen, directional pad, action buttons, start/select buttons, strap holes, hinges, detachable upper cover, speaker slots, branding.

[Figure 385]

The 3D model is a blue Nintendo 3DS XL handheld gaming console with a rounded rectangular shape. It has two screens, a larger touchscreen on the bottom and a smaller screen above, both with black bezels. Controls include a directional pad, action buttons, and start/select buttons. The upper case is a detachable cover, and the backside has "Nintendo 3DS XL" branding.

The 3D model is a blue Nintendo 3DS XL handheld gaming console. It has a slightly rounded rectangular shape with a matte finish. The console features two screens: a larger touchscreen on the bottom and a smaller screen above, both with black bezels. Controls include a directional pad, action buttons ('A', 'B', 'X', 'Y'), and start/select buttons. There are strap holes on the sides and hinges connecting the upper and lower parts. The upper case is a detachable cover, and the backside has "Nintendo 3DS XL" branding. The primary material is plastic, and the color scheme is bright blue with black accents.

The 3D model is a blue Nintendo 3DS XL handheld gaming console. The main body is a slightly rounded rectangle with a matte finish. It features two screens: a larger touchscreen on the bottom and a smaller screen above, both with black bezels. Controls include a directional pad, action buttons ('A', 'B', 'X', 'Y'), and start/select buttons, with a power button near the top-left corner. Strap holes are on the upper edges, and hinges connect the upper and lower parts. The upper case is a detachable cover, and speaker slots are on the sides of the upper case. The backside has "Nintendo 3DS XL" branding and regulatory text. The primary material is plastic, and the color scheme is bright blue with black accents.

The 3D model represents a blue Nintendo 3DS XL handheld gaming console. The main body is a slightly rounded rectangular shape with a matte finish, housing all internal components. Two screens are present: a larger touchscreen on the bottom and a smaller screen above, both surrounded by black bezels. Controls include a directional pad on the left, action buttons ('A', 'B', 'X', 'Y') on the right, and start/select buttons at the center, with a power button near the top-left corner. Strap holes are located on the upper edges of both sides. The closure mechanism features visible hinges where the upper and lower parts meet. The upper case is a detachable cover that folds over the main body, and the lower case houses the screens and controls. Speaker slots are visible on either side of the upper case, just below the hinge area. The backside displays "Nintendo 3DS XL" branding, along with regulatory text and logos. The primary material is plastic with a matte texture, offering a soft tactile feel, while control areas may have a slightly glossier finish. The predominant color is bright blue, with black accents for contrast.

- Figure 20. Multi-level annotation examples of MARVEL for the GSO (Google Scanned Objects) [20] dataset. Words corresponding to Object and Components are highlighted in violet, Shape and Geometry in green, Texture and Materials in orange, Colors in blue, and Contextual Environment in purple. From top to bottom, we go from level-5 (Concise Tags) captions to level-1 (Comprehensive Description) captions.

[Figure 386]

[Figure 387]

Rectangular tabletop, smooth wood grain, four straight legs, horizontal support beam, light brown color, natural wood appearance.

A rectangular wooden table with a smooth, light brown tabletop and four straight legs. A horizontal support beam runs underneath, connecting the legs. The table has a natural wood grain appearance.

The 3D model is a rectangular wooden table. It has a flat, smooth tabletop with wood grain patterns. Four legs are positioned at each corner, and a horizontal support beam runs underneath, connecting the legs. The table is light brown with subtle wood grain, giving it a natural look.

The 3D model is a rectangular wooden table. The tabletop is flat and smooth with visible wood grain, indicating natural materials. Four legs, positioned at each corner, have a simple, straight shape and a slightly rounded texture. A horizontal support beam connects the legs underneath, running parallel to the longer edges of the table. The table has a light brown color with subtle variations in hue, creating a realistic wood grain effect.

The 3D model represents a rectangular wooden table. The tabletop is a flat, smooth surface with visible wood grain patterns, suggesting natural materials like plywood or solid wood veneer. The table measures significantly longer than it is wide, maintaining symmetry in leg placement and underframe design. Four legs, positioned at each corner, have a simple, straight shape with a slightly rounded texture, showing consistent wood grain patterns. A horizontal support beam connects two pairs of legs underneath the tabletop, running parallel to the longer edges, enhancing stability. All components share a light brown color with subtle variations in hue, creating a realistic wood grain effect.

[Figure 388]

[Figure 389]

Modern rectangular bed, white metal frame, slightly curved headboard, grayish floral pattern, plain white mattress, symmetric design, extended headboard, smooth matte finish.

A modern rectangular bed with a slightly curved headboard. The bed has a white metal frame and a grayish floral-patterned headboard. The mattress is plain white. The headboard extends slightly beyond the mattress width.

The bed with headboard is a modern, rectangular design. It has a sturdy white metal frame and a slightly curved headboard covered in grayish fabric with a floral pattern. The mattress is plain white and sits on top of the frame. The design is symmetric and well-proportioned, with the headboard extending slightly beyond the mattress width.

The bed with headboard is a modern, rectangular design. The base frame is made from a sturdy metal with a smooth, matte white finish. The headboard, which is slightly curved, is covered in a grayish fabric with a detailed floral pattern. The mattress is plain white and sits on top of the frame. The design is symmetric, with sharp, clean lines and a balanced proportion between the headboard and the bed's length. The headboard extends slightly beyond the width of the mattress. Ensure the mattress aligns perfectly with the headboard and maintain the smooth, matte finish of the frame.

The bed with headboard is a modern, rectangular design featuring a slightly curved headboard at one end. The base frame is constructed from a sturdy, sleek metal with a smooth, matte white finish. The headboard is covered in a grayish fabric with an intricate floral pattern, featuring small dark flowers and leaves. The fabric has a soft, slightly raised texture, adding depth and detail. The mattress, placed on top of the base frame, is covered in plain white fabric, suggesting a smooth, padded surface for comfort. The design is symmetric along both axes, with sharp, clean lines defining each side. The headboard extends beyond the width of the mattress, maintaining a balanced proportion with the bed's length. Ensure the mattress dimensions match those of the frame, aligning perfectly with the headboard. Pay close attention to the textures and proportions to achieve a faithful recreation.

- Figure 21. Multi-level annotation examples of MARVEL for the Pix3D [74] dataset. Words corresponding to Object and Components are highlighted in violet, Shape and Geometry in green, Texture and Materials in orange, Colors in blue, and Contextual Environment in purple. From top to bottom, we go from level-5 (Concise Tags) captions to level-1 (Comprehensive Description) captions.

[Figure 390]

[Figure 391]

The 3D model represents a humanoid figure with fantastical attributes. The head is symmetrical with a rounded top and two protruding elements resembling horns or large earlobes. Facial features include pronounced cheekbones, a well-defined nose, and a long, flowing beard extending from the chin to the chest. The torso tapers slightly at the waist and widens at the shoulders, with arms positioned naturally alongside the body. Hands are partially visible but not fully detailed. The surface texture is smooth and uniform, with no visible seams or joints, indicating a high level of detailing and polishing. The color scheme is monochromatic, using various shades of gray, which highlights the geometric forms and details.

#### TexturelessSimple

[Figure 392]

[Figure 393]

The 3D model consists of a perfectly spherical lemon head with a pair of sunglasses positioned symmetrically on its surface. The sphere is smooth and lacks any texture, with a uniform matte finish. The sunglasses feature angular, flat frames that rest precisely on the lemon's surface, creating the illusion of being worn. The entire model is uniformly gray, with no color variations, patterns, or gradients. The geometry is simple, focusing on the symmetry and clean lines of the sunglasses. To accurately recreate this model, ensure the sphere is perfectly symmetrical and the sunglasses are aligned with precision.

[Figure 394]

[Figure 395]

The 3D model represents a tree with a singular vertical trunk, multiple branching structures, and a canopy of pink, polyhedral leaves. The trunk is cylindrical, tapering slightly towards the top, and features a textured bark-like surface, providing a realistic roughness. Branches extend from the trunk in various directions, creating an asymmetrical, expansive canopy. Each branch is thinner and smoother than the trunk, contributing to a natural look. The leaves are flat, geometric polygons, primarily hexagons and pentagons, arranged to give a low-poly aesthetic. They are uniformly pink without any gradients or texture.

[Figure 396]

[Figure 397]

The 3D model represents a homegrown orange, characterized by a spherical shape with slight natural irregularities. The surface is generally smooth but textured with tiny, uneven bumps typical of citrus fruits. The orange is symmetrical but exhibits slight asymmetry from different angles, reflecting its natural growth. The peel maintains consistent thickness, with a small, slightly raised stem scar at the top, indicating the attachment point to the tree. The primary color is a bright yellow-orange, with subtle gradients and a vertical line running along one side, suggesting a natural growth line. The texture is rough and bumpy, contributing to a tactile feel.

- Figure 22. Examples illustrating MARVEL’s robustness to simple and textureless models. Our annotation pipeline dynamically adjusts verbosity, ensuring concise yet accurate descriptions even when texture details are minimal or absent. (Top) Textureless models: a smooth humanoid figure with fantastical attributes, and a symmetrical, matte-finished spherical lemon head wearing sunglasses. (Bottom) Simple yet detailed models: a low-poly tree with geometric pink leaves, and a realistically textured homegrown orange showcasing subtle natural irregularities.

###### SHAP-E (5s) DreamFusion (30m) LucidDreamer (45m) HiFA (1h) MARVEL-FX3D (15s)

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

A lively rainforest with tall trees, dense foliage, and a waterfall cascading into a pool surrounded by wildlife.

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

A gentle giant with moss-covered shoulders and vines hanging from its body, resting in a lush jungle.

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

A mischievous elf with pointy ears and a playful grin, holding a small bag of tricks in a bustling marketplace.

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

A cozy cabin in the woods with smoke coming from the chimney and snow covering the roof and trees.

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

A cheerful elf baker with flour-dusted apron and a tray of fresh cookies, working in a cozy kitchen.

- Figure 23. Qualitative Results for high fidelity TT3D generation on unseen prompts. From left to right, 3D models generated using ShapE [33], DreamFusion [62], LucidDreamer [43], HiFA [91] and MARVEL-FX3D (ours).

###### SHAP-E (5s) DreamFusion (30m) LucidDreamer (45m) HiFA (1h) MARVEL-FX3D (15s)

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

A shy fairy with transparent wings and a green dress, sitting on a lily pad in a pond.

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

A wise old wizard with an impressive white beard, reading a scroll in an ancient library.

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

A peaceful garden with a stone path, blooming roses, and a small fountain surrounded by benches.

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

A dark sorcerer with flowing black robes and glowing red eyes, holding an ancient spellbook.

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

A curious gnome with a bushy white beard and a pointy red hat, sitting on a mushroom in an enchanted forest.

- Figure 24. Qualitative Results for high fidelity TT3D generation on unseen prompts. From left to right, 3D models generated using ShapE [33], DreamFusion [62], LucidDreamer [43], HiFA [91] and MARVEL-FX3D (ours).

