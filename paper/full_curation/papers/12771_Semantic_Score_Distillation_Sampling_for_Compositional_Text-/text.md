# arXiv:2410.09009v1[cs.CV]11Oct2024

## SEMANTIC SCORE DISTILLATION SAMPLING FOR COMPOSITIONAL TEXT-TO-3D GENERATION

Ling Yang1†∗, Zixiang Zhang1∗, Junlin Han 2, Bohan Zeng1, Runjia Li2 Philip Torr2, Wentao Zhang1† 1Peking University 2University of Oxford Project: https://github.com/YangLing0818/SemanticSDS-3D

ABSTRACT

Generating high-quality 3D assets from textual descriptions remains a pivotal challenge in computer graphics and vision research. Due to the scarcity of 3D data, state-of-the-art approaches utilize pre-trained 2D diffusion priors, optimized through Score Distillation Sampling (SDS). Despite progress, crafting complex 3D scenes featuring multiple objects or intricate interactions is still difficult. To tackle this, recent methods have incorporated box or layout guidance. However, these layout-guided compositional methods often struggle to provide fine-grained control, as they are generally coarse and lack expressiveness. To overcome these challenges, we introduce a novel SDS approach, Semantic Score Distillation Sampling (SEMANTICSDS), designed to effectively improve the expressiveness and accuracy of compositional text-to-3D generation. Our approach integrates new semantic embeddings that maintain consistency across different rendering views and clearly differentiate between various objects and parts. These embeddings are transformed into a semantic map, which directs a region-specific SDS process, enabling precise optimization and compositional generation. By leveraging explicit semantic guidance, our method unlocks the compositional capabilities of existing pre-trained diffusion models, thereby achieving superior quality in 3D content generation, particularly for complex objects and scenes. Experimental results demonstrate that our SEMANTICSDS framework is highly effective for generating state-of-the-art complex 3D content.

1 INTRODUCTION

Generating high-quality 3D assets from textual descriptions is a long-standing goal in computer graphics and vision research. However, due to the scarcity of 3D data, existing text-to-3D generation models have primarily relied on leveraging powerful pre-trained 2D diffusion priors to optimize 3D representations, typically based on a score distillation sampling (SDS) loss (Poole et al., 2023). Notable examples include DreamFusion, which pioneered the use of SDS to optimize Neural Radiance Field (NeRF) representations (Mildenhall et al., 2021), and Magic3D (Lin et al., 2023a), which further advanced this approach by proposing a coarse-to-fine framework to enhance its performance.

Despite the advancements in lifting and SDS-based methods, generating complex 3D scenes with multiple objects or intricate interactions remains a significant challenge. Recent efforts have focused on incorporating additional guidance, such as box or layout information(Po & Wetzstein, 2024; Epstein et al., 2024; Zhou et al., 2024). Among them, Po & Wetzstein (2024) introduce locally conditioned diffusion for compositional scene diffusion based on input bounding boxes with one shared NeRF representation while Epstein et al. (2024) instantiate and render multiple NeRFs for a given scene using each NeRF to represent a separate 3D entity with a set of layouts. Further advancing this field, GALA3D (Zhou et al., 2024) utilizes large language models (LLMs) to generate coarse layouts to guide 3D generation for compositional scenes.

However, existing layout-guided compositional methods often fall short in achieving fine-grained control over the generated 3D scenes. The current form of box or layout guidance is relatively coarse

∗Contributed equally. †Corresponding authors: yangling0818@163.com, wentao.zhang@pku.edu.cn

GALA3D GSGEN

GraphDreamer LucidDreamer Ours

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

A rabbit sits atop a large, expensive watch with many shiny gears, made half of iron and half of gold, ea ng a birthday cake that is in front of the rabbit

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

A mannequin adorned with a dress made of feathers and moss stands at the center, ﬂanked by a vase with a single blue tulip and another with blue roses.

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

A car with the front right side made of cheese, the front le  side made of sushi, and the back made of LEGO.

- Figure 1: SEMANTICSDS achieves superior compositional text-to-3d generation results over stateof-the-art baselines, particularly in generating multiple objects with diverse attibutes.

and lacks the expressiveness required to effectively guide the SDS process in optimizing the intricate interactions or intersecting parts between multiple objects, particularly when generating objects with multiple attributes. This limitation stems from the fact that pre-trained 2D diffusion models, which are used in SDS, struggle to estimate accurate scores for complex scenarios with consistent views when explicit spatial guidance is absent (Li et al., 2023; Shi et al., 2024). As a result, the generated 3D scenes may lack the level of detail and realism desired, highlighting the need for more precise guidance mechanisms that can provide finer-grained control over the generation process.

To address these limitations, we propose Semantic Score Distillation Sampling (SEMANTICSDS), which boosts the expressiveness and precision of compositional text-to-3D generation. For more explicit 3D expression, we equip SEMANTICSDS with 3D Gaussian Splatting (3DGS) (Kerbl et al.,

- 2023) as the 3D representation. Our approach consists of three key steps: (1) Given a text prompt, we propose a program-aided approach to improve the accuracy of LLM-based layout planning for 3D scenes. (2) We introduce novel semantic embeddings that remain consistent across various rendering views and explicitly distinguish different objects and parts. (3) We then render these semantic embeddings into a semantic map, which serves as guidance for a region-wise SDS process, facilitating fine-grained optimization and compositional generation. Our approach addresses the challenge

of leveraging pre-trained diffusion models, which possess powerful compositional diffusion priors but are difficult to utilize (Wang et al., 2024a; Yang et al., 2024). By using explicit semantic map guidance, we innovatively unlock these compositional 2D diffusion priors for high-quality 3D content generation.

Our main contributions are summarized as follows:

- • We propose SEMANTICSDS, a novel semantic-guided score distillation sampling approach that effectively enhances the expressiveness and precision of compositional text-to-3D generation, as shown in Figure 1.
- • We introduce program-aided layout planning to improve positional and relational accuracy in generated 3D scenes, deriving precise 3D coordinates from ambiguous descriptions.
- • We develop expressive semantic embeddings to augment 3D Gaussian representations, and propose a region-wise SDS process with the rendered semantic map, distinguishing different objects and parts in the compositional generation process.

- 2 RELATED WORK

Text-to-3D Generation Different approaches have been developed to achieve text-to-3D content generation (Deitke et al., 2024; Zeng et al., 2023), such as employing multi-view diffusion models (Shi et al., 2024; Wu et al., 2024a; Kong et al., 2024; Blattmann et al., 2023), direct 3D diffusion models (Gupta et al., 2023; Shue et al., 2023; Wu et al., 2024b) and large reconstruction models (Hong et al., 2024). For instance, multi-view diffusion models are trained and optimized by finetuning video diffusion on 3D datasets, aiding in 3D reconstruction (Voleti et al., 2024; Chen et al., 2024d; Han et al., 2024b). You et al. (2024) propose a training-free method that employs video diffusion as a zero-shot novel view synthesizer. However, these methods require numerous 3D data for training. In contrast, Score Distillation Sampling (SDS) (Poole et al., 2023; Wang et al.,

- 2023) is 3D data-free and generally produces higher quality assets. SDS approaches harness the creative potential of 2D diffusion and have achieved significant advancements (Wang et al., 2024b; Yang et al., 2023b; Hertz et al., 2023), resulting in realistic 3D content generation and enhanced resolution of generative models (Zhu et al., 2024). In this paper, we propose a new SDS paradigm, namely SEMANTICSDS, for text-to-3D generation in complex scenarios, which first incorporates explicit semantic guidance into the SDS process.

Compositional 3D Generation Modeling compositional 3D data distribution is a fundamental and critical task for generative models. Current feed-forward methods (Shue et al., 2023; Shi et al.,

- 2024) are primarily capable of generating single objects and face challenges when creating more complex scenes containing multiple objects due to limited training data. Po & Wetzstein (2024) fix the layout in multiple 3D bounding boxes and generate compositional assets with boundingbox-specific SDS. Recently, a series of learnable-layout compositional methods have been proposed (Epstein et al., 2024; Vilesov et al., 2023; Han et al., 2024a; Chen et al., 2024b; Li et al., 2024; Yan et al., 2024; Gao et al., 2024) . These methods combine multiple object-ad-hoc radiance fields and then optimize the positions of the radiance fields from external feedback. For example, Epstein et al.

(2024) propose learning a distribution of reasonable layouts based solely on the knowledge from a large pre-trained text-to-image model. Vilesov et al. (2023) introduce an optimization method based on Monte-Carlo sampling and physical constraints. Non-learnable layout methods like (Zhou et al., 2024) and Lin et al. (2023b) further utilize LLMs or MLLMs to convert text into reasonable layouts. However, the current form of layout guidance is relatively coarse and not expressive enough for finegrained control. We address this problem by incorporating semantic embeddings that ensure view consistency and distinctly differentiate objects into SDS processes, which are flexible and expressive for optimizing 3D scenes.

- 3 PRELIMINARIES

Compositional 3D Gaussian Splatting 3D Gaussian Splatting explicitly represents a 3D scene as a collection of anisotropic 3D Gaussians, each characterized by a mean µ ∈ R3 and a covariance

matrix Σ (Kerbl et al., 2023). The Gaussian function G(x) is defined as:

G(x) = exp −

- 1

- 2

(x − µ)⊤Σ−1(x − µ) (1)

Rendering a compositional scene necessitates a transformation from object to composition coordinates, involving a rotation R ∈ R3×3, translation t ∈ R3, and scale s ∈ R (Zhou et al., 2024; Vilesov et al., 2023). This transformation is applied to the mean and variance of individual Gaussians, transitioning from the object’s local coordinates to global coordinates: µglobal = sRµlocal+t, Σglobal = s2RΣlocalR⊤.

For optimized rendering of compositional 3D Gaussians into 2D image planes, a tile-based rasterizer enhances rendering efficiency. The rendered color at pixel v is computed as follows:

- i−1
- j=1

(1 − αj), (2)

I(v) =

ciαi

i∈N

where ci represents the color of the i-th Gaussian, N denotes the set of Gaussians within the tile, and αi is the opacity.

Score Distillation Sampling Yang et al. (2023a); Wang et al. (2023) have introduced a method to leverage a pretrained diffusion model, ϵϕ(xt;y,t), to optimize the 3D representation, where xt, y, and t signify the noisy image, text embedding, and timestep, respectively.

Let g represent the differentiable rendering fcuntion, θ denote the parameters of the optimizable

- 3D representation and I = g(θ) be the resulting rendered image. The gradient for optimization is performed via Score Distillation Sampling:

∇θLSDS = Eϵ,t w(t)(ϵϕ (xt;y,t) − ϵ)

∂I ∂θ

(3)

where ϵ is Gaussian noise and w(t) is a weighting function. In compositional 3D generation, local object optimizations and global scene optimizations alternate in a compositional optimization scheme (Zhou et al., 2024). During local optimization, the parameters θ include the mean, covariance, and color of individual Gaussians. In global scene optimization, the parameters θ additionally include transformations—translation, scale, and rotation—that convert local to global coordinates.

- 4 METHOD

- 4.1 PROGRAM-AIDED LAYOUT PLANNING

A detailed characterization of multiple objects’ positions, dimensions, and orientations requires numerous parameters, especially when additionally describing distinct attributes of various object components. In scenarios involving multiple objects, utilizing Large Language Models (LLMs) to derive precise 3D coordinates from ambiguous descriptions within a scene is often challenging. This difficulty arises because purely 3D numerical data and corresponding natural language descriptions do not frequently co-occur in the training data of LLMs (Hong et al., 2023; Xu et al., 2023). Consequently, issues such as overlapping objects or excessive distances between them may occur, particularly during interactions among objects. Therefore, we propose to leverage programs as the intermediate reasoning and planning steps (Gao et al., 2023) to effectively mitigate these challenges.

Let yc represent the complex user input, which includes multiple objects with various attributes. First, We utilize Large Language Models to identify all objects {Ok}Kk=1 within yc, where K denotes the total number of objects. For each object, the corresponding prompt yk is recognized, and its dimensions are estimated. This includes considering the object’s real-world size and its relationship with other objects to determine its relative size, facilitating the placement of all objects within the same scene.

Subsequently, LLMs sequentially position each object within the scene. In designing each object’s placement, LLMs articulate the spatial relationships with relevant entities using programmable language descriptions that explicitly outline all mathematical calculations. This language is then converted into a program executed by a runtime, such as a Python interpreter, to produce the layout

|1. Spatial Layouts<br>2. Semantic Layouts<br><br><br>User prompt Program-aided Layout Planning<br><br>Initialization of Semantic 3DGS<br><br>[Figure 47]<br><br>A car with its front half made<br><br>of cheese and its rear half made of<br><br>sushi is situated to the right of a LEGO house.|
|---|

[Figure 48]

###### RGB Image & Semantic Map

[Figure 49]

[Figure 50]

Render

[Figure 51]

Noise

Back Propagation

Text-to-image

Diffusion 𝓛𝑺𝒆𝒎𝒂𝒏𝒕𝒊𝒄 𝑺𝑫𝑺

[Figure 52]

[Figure 53]

[Figure 54]

A LEGO house， side view

[Figure 55]

[Figure 56]

[Figure 57]

Regional Denoising with Semantic Map

[Figure 58]

Compose

[Figure 59]

SemanticSDS

A car made of cheese, overhead view

A car made of sushi, overhead view

- Figure 2: Overview of SEMANTICSDS, comprising of program-aided layout planning (top) and regional denoising with semantic map (bottom).

solution. These layouts, which include scale factors, Euler angles, and translation vectors, are employed to transform 3D Gaussians from local coordinates to global coordinates during rendering.

Furthermore, for each object Ok, LLMs decomposes its layout space into nk complementary regions, each with distinct attributes and different subprompts {yk,l}n gions are designed to be non-overlapping and collectively encompass the entire layout space of their respective object. To generate meaningful and accurate complementary regions, LLMs employ a structured decomposition process that segments the space of object Ok into hierarchical divisions based on depth, width, and length dimensions. This process is documented using programmable language descriptions and subsequently converted into precise bounding boxes by a program. Details on the prompts used for this program-aided layout planning are provided in Appendix A.1.

- k
- l=1. These complementary re-

- 4.2 SEMANTIC SCORE DISTILLATION SAMPLING

Prompt-Guided Semantic 3D Gaussian Representation To generate 3D scenes involving multiple objects with diverse attributes and to precisely control the attributes of distinct spatial regions within each object, it is essential to utilize features that represent the fine-grained semantics of 3D Gaussians. We design new prompt-guided semantic 3D Gaussian representations. During initialization, the subprompt yk,l corresponding to the i-th Gaussian is encoded via the CLIP text encoder Φ (Radford et al., 2021) to obtain the high-dimensional semantic embedding, hi = Φ(yk,l) ∈ Rd

h. Given the significant memory demands imposed by the large dimensions of

dh, a lightweight autoencoder is employed. This autoencoder effectively compresses the scene’s high-dimensional semantic embeddings into more manageable, low-dimensional representations,

represented as fi = E(hi) ∈ Rd

f. The loss function for the autoencoder is defined as:

dae(D(E(hi)),hi) (4)

Lae =

i∈N

where dae denotes the metric combining the L1 loss and the symmetric cross entropy loss from CLIP (Radford et al., 2021).

The i-th Gaussian is then augmented with a semantic embedding fi ∈ Rd. And semantic information is integrated into the rendered 2D image by rendering the semantic embedding at pixel v using the formula:

- i−1
- j=1

(1 − αj) (5)

F(v) =

fiαi

i∈N

The rendered semantic embedding F(v), derived from equation 5, is fed into the decoder D to reconstruct S(v) = D(F(v)) ∈ Rd

h indicating the rendered image’s semantic attributes.

h and then generates a semantic map S ∈ RH×W×d

Semantic Score Distillation Sampling To enable fine-grained controllable generation, the generated semantic map is integrated into the spatial composition of scores for distillation sampling. The subprompt yk,l is processed through the CLIP text encoder Φ to produce the subprompt embedding qk,l = Φ(yk,l) ∈ Rd

h. The probability that pixel v corresponds to subprompt yk,l is computed as:

exp(cos(qk,l,S(v))/τ)

(6)

p(k,l | v) =

nk′ l′=1 exp(cos(qk,l,S(v))/τ)

K k′=1

where τ is a temperature parameter learned by CLIP and cos(·,·) denotes cosine similarity. This facilitates the derivation of the mask Mk,l(v), which indicates whether the semantic properties of pixel v align with subprompt yk,l.

Mk,l(v) =

1 if (k,l) = arg maxk′,l′ p(k′,l′ | v) 0 otherwise

(7)

The semantic mask Mk,l ∈ {0,1}H×W is subsequently utilized to guide the score distillation sampling. To ensure that the Gaussians near the edges of objects are not overlooked, the mask Mk,l is subjected to a max pooling operation with a 5 × 5 kernel, resulting in Mˆ k,l. Although diffusion models generally lack an inherent distinction at the object and part levels in their latent spaces or attention maps for fine-grained control (Lian et al., 2024), recent advancements in compositional

- 2D image generation have implemented spatially-conditioned generation (Chen et al., 2024a; Yang et al., 2024; Xie et al., 2023). This is achieved through regional denoising or attention manipulation, allowing for fine-grained control over the semantics of the generated images. Specifically, the overall denoising score is calculated as the aggregate of the individually masked denoising scores for each visible subprompt yk,l:

### ϵˆϕ (xt;y,t) = Ek,l ϵϕ (xt;yk,l,t) ⊙ Mˆ k,l (8)

where ⊙ denotes element-wise multiplication. Instead of conditioning the diffusion models on a single text prompt, our semantic score distillation sampling employs the compositional denoising score as follows:

∂x ∂θ

(9)

∇θLSemanticSDS = Eϵ,t w(t)(ˆϵϕ (xt;y,t) − ϵ)

This methodology effectively leverages the expressive compositional generation capabilities of pretrained 2D diffusion models for text-to-3D generation. Further details on SEMANTICSDS are provided in Appendix A.2.

Object-Specific View Descriptor for Global Scene Optimization Unlike object-centric optimization, scenes do not exhibit distinct perspectives as individual objects do. Effective scene generation necessitates precise, part-level control over the optimization of distinct object views. Terms such as ”side view” or ”back view” are rarely applicable to multi-object scenes, and pretrained diffusion models often struggle to generate images accurately from such prompts (Li et al., 2023). Moreover, within a single rendered image, different objects may be visible from varying perspectives. Using a unified view descriptor for an entire scene with multiple objects exacerbates the Janus Problem (Poole et al., 2023). Although the compositional optimization scheme alternates between local object optimizations and global scene optimizations (Zhou et al., 2024), allowing for the correct optimization of different views of objects in local coordinates, it is confounded by optimizations

|[Figure 60]<br><br>[Figure 61]<br><br>Camera<br><br>[Figure 62]<br><br>[Figure 63]<br><br>BlockA<br><br>Ours BlockB<br><br>|
|---|

|[Figure 64]<br><br>[Figure 65]<br><br>Camera<br><br>[Figure 66]<br><br>[Figure 67]<br><br>BlockA<br><br>Original Scene center BlockB<br><br>[Figure 68]<br><br>[Figure 69]<br><br>[Figure 70]<br><br>[Figure 71]|
|---|

- Prompt 1: A photo of blockA, side view.
- Prompt 2: A photo of blockB, front view.

Prompt: A photo of blockA and blockB, front view.

Rendered image

|[Figure 72]<br><br>[Figure 73]|
|---|

- Figure 3: Illustration of our proposed object-specific view descriptor for global scene optimization.

under global coordinates. This limits the frequency of global scene optimizations and results in a lack of scene coherence, harmony, and lighting consistency.

To address this issue, in our SEMANTICSDS, we append an object-specific view descriptor ykview to the corresponding subprompts {yk,l}n

l=1 to optimize individual objects within the rendered image

K

(in Figure 3). The same view descriptor ykview is consistently applied across different parts of each multi-attribute object. Specifically, we determine the camera’s elevation and azimuth angles relative

to each object by computing the angle between the vector nˆ, which extends from the object to the camera, and specific reference axis vectors, such as the positive z-axis. This calculation facilitates the selection of the most appropriate object-specific view descriptor. For instance, if the angle between nˆ and the positive z-axis remains below a predefined threshold, indicative of a high azimuth angle, the descriptor ykview is assigned as an overhead view descriptor for that object.

- 5 EXPERIMENTS

Implementation Details. The guidance model is implemented using the publicly accessible diffusion model, StableDiffusion (Rombach et al., 2022), specifically utilizing the checkpoint runwayml/stable-diffusion-v1-5. Positions of the Gaussians are initialized using Shap-E (Jun & Nichol, 2023), with each object initially comprising 12288 Gaussians. For densification, Gaussians are cloned or split based on the view-space position gradient using a threshold Tpos = 2, with semantic embeddings copied. Compactness-based densification is also applied every 2000 iterations, involving each Gaussian and one of its nearest neighbors, as described in GSGEN (Chen et al., 2024c). Pruning involves removing Gaussians with opacity lower than αmin = 0.3, as well as those with excessively large radii in either world-space or view-space, every 200 iterations.

Training alternates between local and global optimization. During global optimization, the rendered objects vary by switching between the entire scene and pairs of objects. Camera sampling maintains the same focal length, elevation, and azimuth range as specified in (Chen et al., 2024c). The threshold for selecting object-specific view descriptors includes: an overhead view descriptor for elevation angles exceeding 60°, a front view descriptor for azimuth angles within ±45° of the positive x-axis, and a back view descriptor for ±45° angles on the negative x-axis.

Table 1: Quantitative Comparison

Metrics GraphDreamer GSGEN LucidDreamer GALA3D SemanticSDS (Ours) CLIP Score ↑ 0.289 0.314 0.311 0.305 0.321 Prompt Alignment ↑ 56.9 63.3 64.4 85.0 91.1 Spatial Arrangement ↑ 53.8 62.8 65 80.0 85.7 Geometric Fidelity ↑ 53.8 71.1 71.8 80.3 83.0 Scene Quality ↑ 54.9 71.2 65.9 82.3 86.9

Baseline methods. To evaluate the performance of SEMANTICSDS on the complex Text-to-3D task involving multiple objects with varied attributes, we compare it with state-of-the-art (SOTA)

GSGEN Ours

GraphDreamer LucidDreamer GALA3D

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

A corgi is posi oned to the le  of a LEGO house, while a car with its front half made of cheese and its rear half made of sushi is situated to the right of the house made of LEGO.

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

In a library's reading room, a stone block table is ﬂanked by two types of chairs: a high-

back leather chair on the le  side and a low-slung, blue chair on the right. Two lamps, one with a classic design and the other with a modern aesthe c, are posi oned above the table to provide ligh ng.

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

In a botanic garden, a greenhouse is split into two climates. The le  side is a tropical environment with lush greenery, and the right side is a icy snowy climate with cac  and succulents. Two watering cans, one large and the other small, are placed at the entrance.

- Figure 4: Qualitative comparisons of text-to-3D generation. Comparison results demonstrate that SEMANTICSDS synthesizes more precise and realistic multi-object scenes with better visual details, geometric expressiveness, and semantic consistency.

methods. These include the compositional 3D generation method GALA3D (Zhou et al., 2024) and GraphDreamer (Gao et al., 2024), noted for their ability to generate intricate scenes with multiple objects. Additionally, we consider GSGEN (Chen et al., 2024c) and LucidDreamer (Liang et al., 2024), both are capable of producing high-quality, complex objects with diverse attributes.

Metrics. CLIP Score (Radford et al., 2021) is employed as the evaluation metric to assess the quality and consistency of the generated 3D scenes with textual descriptions. However, CLIP tends to focus on the primary objects within the rendered image, and when used to evaluate complex text-to-3D tasks involving multiple objects with varied attributes, it may not adequately assess the geometry of all objects or the rationality of their spatial arrangements. This limitation results in a misalignment with human judgment regarding evaluation criteria. Therefore, following Wu et al. (2024c), GPT-4V is utilized as a human-aligned evaluator to compare 3D assets based on predefined

criteria. These criteria include: (1) Prompt Alignment: ensuring that all objects specified in the user prompts are present and correctly quantified; (2) Spatial Arrangement: evaluating the logical and thematic spatial arrangement of objects; (3) Geometric Fidelity: assessing the geometric fidelity of each object for realistic representation; and (4) Scene Quality: determining the overall scene quality in terms of coherence and visual harmony. More details on metrics are provided in the Appendix A.3.

- 5.1 MAIN RESULTS

Quantitative Analysis To evaluate the performance of SEMANTICSDS in Text-to-3D tasks involving multiple objects with varied attributes, quantitative metrics were employed. As shown in Table 1, the CLIP Score indicates that SEMANTICSDS exhibits strong alignment with the primary semantics of user prompts. Specifically, SEMANTICSDS excels in Prompt Alignment, ensuring that all objects specified in user prompts are present and correctly quantified. Additionally, it demonstrates superior performance in Spatial Arrangement, effectively designing the layout of interactive objects to support the scene’s intended theme. Furthermore, by explicitly guiding SDS with rendered semantic maps, SEMANTICSDS achieves outstanding generation of individual objects with diverse attributes across different spatial components, resulting in high scores in object-level Geometric Fidelity. Additionally, the use of compositional 3D Gaussian Splatting for scene representation helps SEMANTICSDS to effectively disentangle objects within the scene. This, combined with explicit semantic guidance to the SDS, contributes to achieving the highest score in Scene Quality.

Qualitative Analysis To intuitively demonstrate the superiority of the proposed method in generating complex 3D scenes with multiple objects possessing diverse attributes, a qualitative comparison with baseline models is conducted. As illustrated in Figure 4, GALA3D, with a compositional optimization scheme, successfully generates individual objects that align with user prompts. However, it fails to produce plausible results when objects have multiple attributes. Although GSGEN and LucidDreamer generate high-quality individual objects, the presence of multiple objects often leads to entanglement, compromising consistency with user prompts. Additionally, these models are unable to generate reasonable objects when individual objects possess numerous attributes. In contrast, SEMANTICSDS employs guided diffusion models with explicit semantics, effectively generating scenes that include multiple objects with diverse attributes. Moreover, by utilizing programaided layout planning, SEMANTICSDS produces more coherent layouts than GALA3D in scenarios involving complex spatial relationships among multiple objects. For example, in Figure 1, both table lamps are correctly placed on the table without appearing to float when using SEMANTICSDS.

SemanticSDS(Ours)

60%

Gala3d

13%

GSGEN

10%

LucidDreamer

10%

GraphDreamer

7%

Figure 5: User study results. SEMANTICSDS is preferred 60% of the time by users than baseline methods.

User Study We conducted a user study to compare our method with baseline methods across 30 scenes involving more than 100 objects. Each participant was shown a user prompt alongside 3D scenes generated by all methods simultaneously and asked to select the most realistic assets based on geometry, prompt alignment, and accurate placement. Figure 5 illustrates that SEMANTICSDS significantly outperformed previous methods in terms of human preference.

- 5.2 MODEL ANALYSIS

Effectiveness of Program-aided Layout Planning We assess the necessity of program-aided layout planning through an ablation study. The qualitative comparison of generated layouts is illustrated in Figure 6. Without program-aided planning, layout placement often lacks rationale and results in poor spatial arrangements. In contrast, the program-aided strategy positions the layouts logically and divides the layout into meaningful and precise complementary regions for objects with multiple attributes, resulting in an effective spatial arrangement.

Impact of Semantic Score Distillation Sampling Ablation experiments are performed on Semantic Score Distillation Sampling to evaluate the effects of explicitly guiding SDS with rendered semantic maps. In Figure 7, without SEMANTICSDS, while objects with single attributes are generated effectively, those with varied attributes often experience blending issues. For instance, the

##### Without program-aided With program-aided (Ours)

|Overhead View|
|---|

[Figure 117]

[Figure 118]

|OverheadOverhead ViewView|
|---|

Decomposed prompt

[Figure 119]

|Side View|
|---|

|Side View|
|---|

[Figure 120]

[Figure 121]

Text prompt: A table, half made of wood and half white, holds a roasted turkey, a salad, a glass of orange juice, and a plate with a loaf of French bread.

- Figure 6: Qualitative comparisons between without and with our program-aided layout planning.

”house” shows snow bricks mixed with LEGO bricks, failing to meet the user prompt’s spatial requirements. The snow bricks are inaccurately represented as white LEGO bricks, which do not align with the intended attributes. Additionally, one attribute may dominate, causing others to disappear, such as in the ”car” with three attributes in Figure 7. Conversely, SemanticSDS enables precise control over the attributes in distinct spatial regions of each object, producing objects with diverse attributes and smooth transitions between regions with different attributes.

[Figure 122]

[Figure 123]

Text prompt: A corgi is positioned to the left of a house that is half made of LEGO and half of snow. To the right of the house, there is a car with its front right side made of cheese, front left side made of sushi, and the back made of LEGO.

Without SemanticSDS Without object-specific view descriptor With both

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

- Figure 7: Qualitative analysis. Our SEMANTICSDS provides more precise and fine-grained control and our proposed object-specific view descriptor helps with better multi-view understanding.

Object-Specific View Descriptor To assess the effectiveness of the object-specific view descriptor, we replace it with the scene-centric view descriptor utilized by GSGEN during global optimization. This change increases the occurrence of the Janus Problem, as illustrated by the overhead view of the corgi in the middle of Figure 7. These findings highlight the crucial role of selecting an appropriate view descriptor to enhance the plausibility of generated 3D scenes.

- 6 CONCLUSION

In this paper, we introduce SEMANTICSDS, a novel SDS method that significantly enhances the expressiveness and precision of compositional text-to-3D generation. By leveraging program-aided layout planning, semantic embeddings, and explicit semantic guidance, we unlock the compositional priors of pre-trained diffusion models and achieve realistic high-quality generation in complex scenarios. Our extensive experiments demonstrate that SEMANTICSDS achieves state-of-the-art results for generating complex 3D content. As we look to the future, we envision SEMANTICSDS as a foundation for even more applications, such as automatic editing and closed-loop refinement, paving the way for unprecedented levels of creativity and innovation in 3D content generation.

REFERENCES

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Minghao Chen, Iro Laina, and Andrea Vedaldi. Training-free layout control with cross-attention guidance. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pp. 5343–5353, 2024a.

Yongwei Chen, Tengfei Wang, Tong Wu, Xingang Pan, Kui Jia, and Ziwei Liu. Comboverse: Compositional 3d assets creation using spatially-aware diffusion guidance. arXiv preprint arXiv:2403.12409, 2024b.

Zilong Chen, Feng Wang, Yikai Wang, and Huaping Liu. Text-to-3d using gaussian splatting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21401– 21412, 2024c.

Zilong Chen, Yikai Wang, Feng Wang, Zhengyi Wang, and Huaping Liu. V3d: Video diffusion models are effective 3d generators. arXiv preprint arXiv:2403.06738, 2024d.

Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, et al. Objaverse-xl: A universe of 10m+ 3d objects. Advances in Neural Information Processing Systems, 36, 2024.

Dave Epstein, Ben Poole, Ben Mildenhall, Alexei A Efros, and Aleksander Holynski. Disentangled 3d scene generation with layout learning. In International Conference on Machine Learning, 2024.

Gege Gao, Weiyang Liu, Anpei Chen, Andreas Geiger, and Bernhard Sch¨olkopf. Graphdreamer: Compositional 3d scene synthesis from scene graphs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21295–21304, 2024.

Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. In International Conference on Machine Learning, pp. 10764–10799. PMLR, 2023.

Anchit Gupta, Wenhan Xiong, Yixin Nie, Ian Jones, and Barlas O˘guz. 3dgen: Triplane latent diffusion for textured mesh generation. arXiv preprint arXiv:2303.05371, 2023.

Haonan Han, Rui Yang, Huan Liao, Jiankai Xing, Zunnan Xu, Xiaoming Yu, Junwei Zha, Xiu Li, and Wanhua Li. Reparo: Compositional 3d assets generation with differentiable 3d layout alignment. arXiv preprint arXiv:2405.18525, 2024a.

Junlin Han, Filippos Kokkinos, and Philip Torr. Vfusion3d: Learning scalable 3d generative models from video diffusion models. European Conference on Computer Vision (ECCV), 2024b.

Amir Hertz, Kfir Aberman, and Daniel Cohen-Or. Delta denoising score. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2328–2337, 2023.

Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. ICLR, 2024.

Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. Advances in Neural Information Processing Systems, 36:20482–20494, 2023.

Heewoo Jun and Alex Nichol. Shap-e: Generating conditional 3d implicit functions. arXiv preprint arXiv:2305.02463, 2023.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.

Xin Kong, Shikun Liu, Xiaoyang Lyu, Marwan Taher, Xiaojuan Qi, and Andrew J Davison. Eschernet: A generative model for scalable view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9503–9513, 2024.

Runjia Li, Junlin Han, Luke Melas-Kyriazi, Chunyi Sun, Zhaochong An, Zhongrui Gui, Shuyang Sun, Philip Torr, and Tomas Jakab. Dreambeast: Distilling 3d fantastical animals with part-aware knowledge transfer, 2024. URL https://arxiv.org/abs/2409.08271.

Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22511–22521, 2023.

Long Lian, Boyi Li, Adam Yala, and Trevor Darrell. LLM-grounded diffusion: Enhancing prompt understanding of text-to-image diffusion models with large language models. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/ forum?id=hFALpTb4fR. Featured Certification.

Yixun Liang, Xin Yang, Jiantao Lin, Haodong Li, Xiaogang Xu, and Yingcong Chen. Luciddreamer: Towards high-fidelity text-to-3d generation via interval score matching. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6517–6526, 2024.

Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 300–309, 2023a.

Yiqi Lin, Hao Wu, Ruichen Wang, Haonan Lu, Xiaodong Lin, Hui Xiong, and Lin Wang. Towards language-guided interactive 3d generation: Llms as layout interpreter with generative feedback. arXiv preprint arXiv:2305.15808, 2023b.

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021.

Ryan Po and Gordon Wetzstein. Compositional 3d scene generation using locally conditioned diffusion. In 2024 International Conference on 3D Vision (3DV), pp. 651–663. IEEE, 2024.

Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. In International Conference on Learning Representations, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. In The Twelfth International Conference on Learning Representations, 2024.

J Ryan Shue, Eric Ryan Chan, Ryan Po, Zachary Ankner, Jiajun Wu, and Gordon Wetzstein. 3d neural field generation using triplane diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20875–20886, 2023.

Alexander Vilesov, Pradyumna Chari, and Achuta Kadambi. Cg3d: Compositional generation for text-to-3d via gaussian splatting. arXiv preprint arXiv:2311.17907, 2023.

Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, David Pankratz, Dmitry Tochilkin, Christian Laforte, Robin Rombach, and Varun Jampani. Sv3d: Novel multi-view synthesis and 3d generation from a single image using latent video diffusion. arXiv preprint arXiv:2403.12008, 2024.

Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A Yeh, and Greg Shakhnarovich. Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12619–12629, 2023.

Ruichen Wang, Zekang Chen, Chen Chen, Jian Ma, Haonan Lu, and Xiaodong Lin. Compositional text-to-image synthesis with attention map control of diffusion models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 5544–5552, 2024a.

Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems, 36, 2024b.

Kailu Wu, Fangfu Liu, Zhihan Cai, Runjie Yan, Hanyang Wang, Yating Hu, Yueqi Duan, and Kaisheng Ma. Unique3d: High-quality and efficient 3d mesh generation from a single image. arXiv preprint arXiv:2405.20343, 2024a.

Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3d: Scalable image-to-3d generation via 3d latent diffusion transformer. arXiv preprint arXiv:2405.14832, 2024b.

Tong Wu, Guandao Yang, Zhibing Li, Kai Zhang, Ziwei Liu, Leonidas Guibas, Dahua Lin, and Gordon Wetzstein. Gpt-4v (ision) is a human-aligned evaluator for text-to-3d generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 22227– 22238, 2024c.

Jinheng Xie, Yuexiang Li, Yawen Huang, Haozhe Liu, Wentian Zhang, Yefeng Zheng, and Mike Zheng Shou. Boxdiff: Text-to-image synthesis with training-free box-constrained diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7452–7461, 2023.

Runsen Xu, Xiaolong Wang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. Pointllm: Empowering large language models to understand point clouds. arXiv preprint arXiv:2308.16911, 2023.

Han Yan, Yang Li, Zhennan Wu, Shenzhou Chen, Weixuan Sun, Taizhang Shang, Weizhe Liu, Tian Chen, Xiaqiang Dai, Chao Ma, et al. Frankenstein: Generating semantic-compositional 3d scenes in one tri-plane. arXiv preprint arXiv:2403.16210, 2024.

Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. Diffusion models: A comprehensive survey of methods and applications. ACM Computing Surveys, 56(4):1–39, 2023a.

Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and CUI Bin. Mastering textto-image diffusion: Recaptioning, planning, and generating with multimodal llms. In Forty-first International Conference on Machine Learning, 2024.

Xiaofeng Yang, Yiwen Chen, Cheng Chen, Chi Zhang, Yi Xu, Xulei Yang, Fayao Liu, and Guosheng Lin. Learn to optimize denoising scores for 3d generation: A unified and improved diffusion prior on nerf and 3d gaussian splatting. arXiv preprint arXiv:2312.04820, 2023b.

Meng You, Zhiyu Zhu, Hui Liu, and Junhui Hou. Nvs-solver: Video diffusion model as zero-shot novel view synthesizer. arXiv preprint arXiv:2405.15364, 2024.

Bohan Zeng, Shanglin Li, Yutang Feng, Hong Li, Sicheng Gao, Jiaming Liu, Huaxia Li, Xu Tang, Jianzhuang Liu, and Baochang Zhang. Ipdreamer: Appearance-controllable 3d object generation with image prompts. arXiv preprint arXiv:2310.05375, 2023.

Xiaoyu Zhou, Xingjian Ran, Yajiao Xiong, Jinlin He, Zhiwei Lin, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang. Gala3d: Towards text-to-3d complex scene generation via layout-guided generative gaussian splatting. In Forty-first International Conference on Machine Learning, 2024.

Junzhe Zhu, Peiye Zhuang, and Sanmi Koyejo. HIFA: High-fidelity text-to-3d generation with advanced diffusion guidance. In International Conference on Learning Representations, 2024.

- A MORE IMPLEMENTATION DETAILS

- A.1 PROMPTS FOR PROGRAM-AIDED LAYOUT PLANNING

You are a proﬁcient 3D scene designer with the ability to eﬀec vely posi on 3D models within a 3D cubic space. Using a provided scene descrip on, please carry out the following tasks:

###### 1. Iden fy 3D Models:

- Iden fy and list the 3D models men oned in the descrip on.

###### 2. Write Python Code to Es mate Dimensions and Arrange 3D models:

- - Measure the dimensions of each 3D model as if they were toys. They don't need to be realis c, but they should ﬁt together in one cubic space. Models can diﬀer in size but shouldn't be more than twice as big as the smallest one.
- - Arrange the iden ﬁed 3D models in a 3D cubic space centered at coordinates [0, 0, 0], with measurements in cen meters. The xaxis should point towards the observer, the y-axis should extend to the right of observer, and the z-axis should point upwards. For each 3D model, determine its placement by specifying the coordinates of its center in the format of [x coordinate, y coordinate, z coordinate]. Ensure that the 3D models are posi oned in a plausible manner, avoiding overlaps or extending beyond the conﬁnes of the deﬁned space. If necessary, you may make educated es ma ons to achieve a coherent arrangement. Here are some examples, follow the example to design the 3D scene: Scene descrip on: …… Let's think step by step and write the python codes.

- Task 1: Iden fy Models. Iden fy and list the 3D models men oned in the descrip on. If two models are closely associated, I will iden fy them as one model. From the scene descrip on, the tangible models men oned are: ….. models = {

"corgi": {"object descrip on": "Corgi dog"}, "beret": {"object descrip on": "Beret hat"}, "house": {"object descrip on": "The house straddles a divide that separates spring and winter horizontally."}, "car": {"object descrip on": "Car, with front layer made of wood and rear layer made of sushi and cheese. The le  half of the rear

layer is made of sushi, and the right half is made of cheese."} }

- Task 2: Es mate Dimensions. For the scene descrip on that involves a corgi, a beret, a house, and a car with layered materials, let's es mate the dimensions of each model to ensure they ﬁt within a uniﬁed cubic space. models["corgi"]["dimension"] = {"x": 30, "y": 15, "z": 20} # cm models["beret"]["dimension"] = {"x": 15, "y": 15, "z": 5} # cm, diameter ﬁ ng the corgi's head, thickness/height when laid ﬂat models["house"]["dimension"] = {"x": 45, "y": 30, "z": 35} # cm, the size of a house can vary widely. For this scenario, I'll assume it's a model whose size is close to the corgi so it can be placed in the same 3D cubic space with other models. models["car"]["dimension"] = {"x": 40, "y": 20, "z": 15} # cm, toy car size ﬁ ng the scene
- Task3: Calculate the posi ons considering viewing from the front. Arrange the iden ﬁed 3D models in a 3D cubic space centered at coordinates [0, 0, 0]. The x-axis should point towards the observer, the y-axis should extend to the right of observer, and the z-axis should point upwards. We'll start by placing the house at the center, then posi on the corgi and car rela ve to the house, and ﬁnally, place the beret on top of the corgi. ……

# Posi on the corgi to the le  of the house, on the ground models["corgi"]["posi on"] = {

# Centered on x-axis, aligned with the house

- "x": 0,
- "y": models["house"]["posi on"]["y"] - (models["house"]["dimension"]["y"] / 2 + models["corgi"]["dimension"]["y"] / 2 + 5), # Half the height of the corgi oﬀ the ground to represent the corgi si ng on the ground
- "z": models["corgi"]["dimension"]["z"] / 2

} ……

Scene descrip on: {{user_prompt}} Let's think step by step and write the python codes.

Figure 8: The prompt for scene-level decomposition in program-aided layout planning.

Large Language Models (LLMs) have the potential for spatial awareness; however, precise 3D layout generation from vague language descriptions is challenging. This difficulty arises because

- 3D digital data and corresponding natural language descriptions often do not appear simultaneously (Hong et al., 2023; Xu et al., 2023). Moreover, minor numerical changes, which might not be reflected in imprecise language, can lead to unrealistic spatial arrangements of 3D scenes. Additionally, the spatial arrangement of multi-object scenes requires numerous parameters, making a program-aided approach necessary to bridge the gap between natural language descriptions and 3D digital data.

Specifically, we decompose the process of generating multiple objects with diverse attributes into two steps: scene-level decomposition and object-level decomposition. In scene decomposition, we guide LLMs to translate user prompts into Python programs, using explicit mathematical operations

As a 3D model designer, you are tasked with designing an object described in the user prompt. This object has mul ple a ributes, with diﬀerent parts possessing diﬀerent a ributes. Your job is to divide the object as described in the user prompt into parts, each with a single a ribute, and rewrite the corresponding prompt for each part. Speciﬁcally, you need to divide the 3D bounding box encompassing the object into diﬀerent complementary smaller bounding boxes, and output in the speciﬁc format. # The speciﬁc format descrip on The output should be a JSON object that represents the 3D bounding box of the object. This object should have a key named "depth split" that contains an array of objects. Each object represents a division of the bounding box along the depth axis. The object should have two keys: "size" and "ver cal split". The "size" key represents the size of this part rela ve to other parts in the same split. The "ver cal split" key should contain an array of objects. Each object represents a division of the bounding box along the ver cal axis. The object should have two keys: "size" and "horizontal split". The "size" key represents the size of this part rela ve to other parts in the same split. The "horizontal split" key should contain an array of objects. Each object represents a division of the bounding box along the horizontal axis. The object should have two keys: "size" and "prompt". The "size" key represents the size of this part rela ve to other parts in the same split. The "prompt" key should contain the prompt for the speciﬁc part of the object. The prompt should be a string that describes the part of the object and its single a ributes.

# Examples

......

Figure 9: The prompt for decomposing each object into complementary regions.

to represent relationships between objects. For object decomposition, since complementary regions are designed to be non-overlapping and collectively encompass the entire layout space of their respective objects, we devised a scheme employing structured JavaScript Object Notation (JSON) to represent hierarchical divisions based on depth, width, and length dimensions. Figures 8 and 9 illustrate the detailed prompts for scene and object decomposition, respectively.

- A.2 SEMANTICSDS

Camera Sampling Training alternates between local and global optimization. During local optimization, objects are not transformed into global coordinates. In global optimization, the rendering of objects varies by switching between the entire scene and pairs of objects to better optimize those that interact or occlude each other. When rendering only a pair of objects, the camera’s look-at point is sampled at the midpoint between the two objects rather than the center of the entire scene. Additionally, we apply a dynamic camera distance from the object pair to ensure the objects are appropriately sized in the rendered images. Specifically, the camera distance is determined by the scale of the objects and the distance between their centers.

Pooling of Semantic Masks Given that the rendered RGB images and the semantic map have sizes of 512 × 512, whereas the latents for denoising are of size 64 × 64, we convert the semantic map S into masks to compose the denoising scores predicted by diffusion models. Subsequently, for each mask Mk,l ∈ {0,1}512×512, we apply average pooling with a stride of 8 using an 8 × 8 kernel to downsample the data. To ensure that Gaussians near the edges of objects and isolated Gaussians are not overlooked, the mask Mk,l undergoes a max pooling operation with a 5×5 kernel, resulting in Mˆ k,l.

Compositional Optimization Scheme The compositional optimization scheme encompasses both global scene and local object optimizations. Only global scene optimizations apply affine transformations to convert objects from local to global coordinates. During local optimization, θ in equation 9 includes the mean, covariance, and color of individual Gaussians. In global scene optimization, θ additionally includes the parameters of affine transformations—translation, scale, and rotation—that convert local to global coordinates.

- A.3 DETAILS OF METRICS

CLIP Score The CLIP score utilizes CLIP embeddings (Radford et al., 2021) to evaluate text-to-

- 3D alignment. Following previous methods (Zhou et al., 2024; Gao et al., 2024), we calculate the cosine similarity between the user prompt and scene images rendered from different perspectives. For each scene, we take the maximum CLIP score from all rendered images as the representative score. We then compare the average of these maximum scores across different scenes for each method.

Our task is to evaluate two complex 3D scenes that have been generated from the speciﬁc user prompt "{{user_prompt}}". I will provide you with images of these scenes, speciﬁcally image renderings, for each method used. We want to assign a score from 1 to 100 (where 1 is the lowest and 100 is the highest) according to the provided four criteria:

- 1. User Prompt & Scene Alignment: Assess whether all objects men oned in the user prompt "{{user_prompt}}" are present in the 3D scenes generated by both methods and whether the quan ty of each type of object matches the numbers speciﬁed in the prompt. Describe each scene brieﬂy and then evaluate the completeness and accuracy in replica ng the described elements for both methods.
- 2. Spa al Arrangement of Objects: Look at the RGB images to assess the arrangement and posi oning of objects within the scenes. Determine whether the spa al rela onships and layout of objects appear logical and conducive to the scene's intended func on or theme for both methods.
- 3. Geometric Fidelity: Examine each object within the scenes through the RGB images for both methods. Evaluate the overall shape and structure of each object, checking for any geometric inconsistencies or distor ons that might aﬀect the object's realis c representa on.
- 4. Overall Scene Quality: Evaluate the overall coherence and technical quality of the scenes as a composite assessment, based on the integra on of user prompt alignment, spa al arrangement, and geometric ﬁdelity. Consider factors like visual harmony and technical execu on in your overall assessment.

For each of the criteria, you will need to provide a score from 1 to 100 for each method. Addi onally, provide a short analysis for each of the aforemen oned evalua on criteria for both methods. The analysis should be very concise and accurate.

Let's step by step analyze the alignment of the scenes with the user prompt "{{user_prompt}}" and proceed to score and describe each method systema cally.

# Example output: Analysis:

- 1. User Prompt & Scene Alignment:

- - Method A: The scene includes objects such as trees, benches, and lamps; Score: 85 All described objects are present, and the quan  es are mostly accurate with minor devia ons.
- - Method B: The scene includes the same objects but with slight varia ons in quan ty; Score: 80 Most objects are present, but there are notable discrepancies in object count.

- 2. Spa al Arrangement of Objects: ……
- 3. Geometric Fidelity: ……
- 4. Overall Scene Quality: …… Final scores:

- - Method A: 85, 78, 82, 90
- - Method B: 80, 83, 88, 84

Figure 10: The prompt for guiding GPT-4 as a human-aligned evaluator

GPT-4V as A Human-Aligned Evaluator Due to the limitations of the CLIP score in capturing spatial arrangement and geometric fidelity, we follow Wu et al. (2024c) and employ GPT-4V to evaluate complex 3D scenes involving multiple objects with varied attributes. Specifically, we provide GPT-4V with rendered images of the same 3D scene generated by different methods and require it to score each scene on four aspects: Prompt Alignment, Spatial Arrangement, Geometric Fidelity, and Scene Quality, each on a scale from 1 to 100. For each scene and method pair, we perform three independent evaluations. The final score for each method is obtained by averaging the scores across different scenes and comparisons with other methods. Figure 10 presents the prompt used to guide the GPT-4V evaluator. In the prompt, ”method A” and ”method B” are used to anonymize the methods, preventing name bias in GPT-4V’s judgment.

- B MORE SYNTHESIS RESULTS

[Figure 131]

[Figure 132]

[Figure 133]

A cozy scene with a plush triceratops toy surrounded by a plate of chocolate chip cookies, a glistening cinnamon roll, and a flaky croissant.

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

A hamburger, a loaf of bread, an order of fries, and a cup of Coke.

A glass block, a wooden block, a stone block, and a glowing lamp are displayed. They are arranged sequentially from left to right: the wooden block is first, followed by the stone block, then the glass block, and the glowing lamp is placed at the back of the stone block.

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

A table with a roasted turkey, a salad, a loaf of French bread, a glass of orange juice and plate.

[Figure 146]

[Figure 147]

[Figure 148]

A puppy is lying on the iron plate at the top of the Great Pyramid, which is made of snow bricks and stone bricks.

[Figure 149]

[Figure 150]

[Figure 151]

A camping scene with a tent and two wooden stools with colorful patterns next to a campfire.

A white cat lies on a plank of wood, flanked by two sparkling balloons, one orange and one blue.

Figure 11: More synthesis results of multiple objects with our SEMANTICSDS.

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

A castle made of snow bricks and stone bricks.

[Figure 158]

[Figure 159]

[Figure 160]

A pyramid-shaped burrito artistically blended with the Great Pyramid.

[Figure 161]

[Figure 162]

[Figure 163]

A train with a front of cake and a back of steam engine.

[Figure 164]

[Figure 165]

[Figure 166]

A tray of sushi, apples and oranges.

[Figure 167]

[Figure 168]

[Figure 169]

A bust of Theodoros Kolokotronis made of bronze and marble.

A motorcycle made of amigurumi and origami.

Figure 12: More synthesis results of single object with diverse attributes with our SEMANTICSDS.

