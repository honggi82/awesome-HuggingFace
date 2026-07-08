# arXiv:2512.03540v2[cs.CV]5Dec2025

## CookAnything: A Framework for Flexible and Consistent Multi-Step Recipe Image Generation

Ruoxuan Zhang

zhangrx22@mails.jlu.edu.cn Jilin University Changchun, China

Bin Wen

wenbin2122@mails.jlu.edu.cn Jilin University Changchun, China

Hongxia Xie∗

hongxiaxie@jlu.edu.cn Jilin University Changchun, China

Yi Yao

leo81005.ee10@nycu.edu.tw National Yang Ming Chiao Tung University Hsinchu, Taiwan

Songhan Zuo

zuosh2122@mails.jlu.edu.cn Jilin University Changchun, China

Jian-Yu Jiang-Lin

jianyu@cmlab.csie.ntu.edu.tw National Taiwan University Taipei, Taiwan

Hong-Han Shuai

hhshuai@nycu.edu.tw National Yang Ming Chiao Tung University Hsinchu, Taiwan

Wen-Huang Cheng

wenhuang@csie.ntu.edu.tw National Taiwan University Taipei, Taiwan

[Figure 1]

Figure 1: Demonstration of our CookAnything model generating multi-step cooking instructions in a single pass. Each example shows the user’s prompt (left) and the corresponding series of dish images (right), from initial preparation steps through the final plated result (Details of the complete recipe text can be found in the Supplementary A.6.).

### Abstract

Cooking is a sequential and visually grounded activity, where each step such as chopping, mixing, or frying carries both procedural logic and visual semantics. While recent diffusion models have shown strong capabilities in text-to-image generation, they struggle to handle structured multi-step scenarios like recipe illustration. Additionally, current recipe illustration methods are unable to adjust to the natural variability in recipe length, generating a fixed number of images regardless of the actual instructions structure. To address these limitations, we present CookAnything, a flexible and consistent diffusion-based framework that generates coherent,

∗Corresponding Author.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

MM ’25, Dublin, Ireland © 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-2035-2/2025/10 https://doi.org/10.1145/3746027.3755174

semantically distinct image sequences from textual cooking instructions of arbitrary length. The framework introduces three key components: (1) Step-wise Regional Control (SRC), which aligns textual steps with corresponding image regions within a single denoising process; (2) Flexible RoPE, a step-aware positional encoding mechanism that enhances both temporal coherence and spatial diversity; and (3) Cross-Step Consistency Control (CSCC), which maintains fine-grained ingredient consistency across steps. Experimental results on recipe illustration benchmarks show that CookAnything performs better than existing methods in trainingbased and training-free settings. The proposed framework supports scalable, high-quality visual synthesis of complex multi-step instructions and holds significant potential for broad applications in instructional media, and procedural content creation. More details are at https://github.com/zhangdaxia22/CookAnything.

### CCS Concepts

• Computing methodologies → Computer vision tasks.

### Keywords

Recipe image generation, procedural sequence generation, food computing

ACM Reference Format:

Ruoxuan Zhang, Bin Wen, Hongxia Xie, Yi Yao, Songhan Zuo, Jian-Yu JiangLin, Hong-Han Shuai, and Wen-Huang Cheng. 2025. CookAnything: A Framework for Flexible and Consistent Multi-Step Recipe Image Generation. In Proceedings of the 33rd ACM International Conference on Multimedia (MM ’25), October 27–31, 2025, Dublin, Ireland. ACM, New York, NY, USA, 19 pages. https://doi.org/10.1145/3746027.3755174

### 1 Introduction

Cooking is a richly visual and sequential activity: from chopping onions to garnishing a dish, each step not only involves semantic transitions but also yields observable visual transformations [16, 21, 43]. Accurately illustrating these processes from textual instructions holds significant value for applications in culinary education, assistive technology, and multimodal content generation, enabling users to better understand, follow, and interact with complex procedures in an intuitive visual manner.

As textual recipes abstract the cooking process into language, recipe illustration aspires to reverse this abstraction, generating coherent image sequences that visually narrate each procedural step [8, 14, 18, 23]. Compared to single-image generation, this task introduces unique challenges: it requires maintaining temporal progression, preserving ingredient consistency, and capturing subtle visual distinctions between stages. StackedDiffusion [18] pioneered the task of illustrated recipe instructions by generating one image per recipe step. However, its design assumes a fixed number of steps, ignoring the inherent variability across recipes, resulting in both under-generation and over-generation in real-world settings.

While recent advances in text-to-image synthesis, particularly diffusion-based models such as FLUX [11] have achieved remarkable success in high-fidelity image generation, these models are predominantly designed for single-image outputs. This limits their applicability in structured, multi-step domains such as cooking recipes, where each visual output must correspond to a distinct

semantic step and together form a coherent sequence. Attempts to extend these models, such as In-Context LoRA [9], adopt a simple concatenation of step prompts to jointly synthesize multi-step outputs. However, this design leads to semantic entanglement, where visual features bleed across steps, producing indistinguishable or incoherent images that undermine the narrative flow.

Motivated by these challenges, we investigate the key question: How can we achieve flexible, coherent, and semantically disentangled multi-step recipe image generation in a unified framework?

We propose CookAnything, a diffusion-based framework for generating step-by-step illustrated recipes with variable length and high visual consistency (as shown in Fig. 1). It introduces three key components: (1) a Step-wise Regional Control (SRC) mechanism that assigns each instruction to a distinct latent region, ensuring semantic separation and global coherence; (2) a new positional encoding method, Flexible Rotary Position Embedding (RoPE), which resets coordinate indices per step to support diverse layouts; and (3) a Cross-Step Consistency Control (CSCC) module that preserves the visual continuity of fine-grained ingredients across steps. These innovations enable structured, coherent, and flexible multi-image generation for sequential visual synthesis.

Our contributions are summarized as follows:

- • We propose CookAnything, the first diffusion-based framework for illustrated recipe generation with arbitrary-length, step-wise image sequences, flexibly adapting to diverse realworld structures.
- • We introduce SRC and Flexible RoPE to address position misalignment via step-aware spatial encoding and region binding, and CSCC to ensure visual consistency of recurring ingredients across steps.
- • Experiments show that our method achieves state-of-theart results in both training-based and training-free settings, with broad potential in instructional and procedural content generation.

### 2 Related Work 2.1 Recipe Analysis Task

The study of food-centric multimedia has garnered increasing attention within the multimedia research community due to its significant relevance to human survival, nutrition, health, and sensory enjoyment [22, 36, 38, 40, 42, 47, 49, 50]. The growing availability of large-scale food datasets such as Recipe1M+ [17], Food-101 [2], VireoFood-172 [5], and Nutrition5k [37] has fueled research in a wide range of fine-grained food analysis tasks. These tasks include food and ingredient classification [10, 19, 20, 44, 45], food instance segmentation [12, 39], and nutrition or weight estimation [7], among others.

In this work, we focus on the task of step-wise recipe image generation, which aims to synthesize a sequence of visual illustrations corresponding to each step in a cooking recipe. Early efforts, such as CookGAN [8], generated dish images based on latent representations of ingredient lists, while ChefGAN [23] employed recipe instructions as input. ML-CookGAN [14] further combined both ingredients and steps to generate the final dish image. However, these methods are inherently limited to producing only a single

image corresponding to the completed dish, thus failing to capture the procedural nature of cooking. To address this limitation, StackedDiffusion [18] introduced the novel task of illustrated recipe instructions, where an image is generated for each individual step in the recipe. However, it falls short in adapting to the natural variability of recipe lengths, producing a fixed number of step images regardless of the actual recipe structure. In this work, we propose a flexible framework, CookAnything, that dynamically adapts to the varying number of steps in different recipes while ensuring accurate visual-semantic alignment, procedural coherence, and ingredient consistency across the generated sequence.

### 2.2 Procedural Sequence Generation Model

In the context of text-to-image synthesis, a Procedural Sequence Generation Model decomposes the generation pipeline into discrete stages or operations, such as layout planning, object placement, attribute assignment, and appearance refinement. Recently, diffusion models have revolutionized text-to-image generation, producing high-fidelity visuals through iterative denoising [6, 9, 11, 27]. Latent Diffusion Models (LDM) [30] and Vision Transformer-based Diffusion Transformers (DiTs) [11, 24] balance generation quality with scalability, capturing global semantics via attention mechanisms. However, most existing work focuses on single-image synthesis, overlooking structured, multi-step image generation.

Emerging efforts in story visualization [15] hint at the potential of sequential visual generation but fall short in domains like recipes, which demand procedural consistency, spatial coherence, and semantic disentanglement across steps. Crucially, ingredient continuity must be preserved, ensuring logical visual evolution throughout the cooking process.

To our knowledge, we are the first to introduce a diffusion-based model for step-wise recipe illustration. Building on the DiT backbone, our framework explicitly models semantic grounding, temporal alignment, and ingredient consistency across visual sequences, offering a structured and extensible solution for procedural image generation.

### 3 Method

In this section, we introduce CookAnything, a framework for flexible and consistent multi-step recipe image generation (see Fig. 2). To enable the generation of multiple step-specific images in a single denoising process, we propose a Step-wise Regional Control (SRC) mechanism, which aligns each step instruction with its corresponding visual region. However, standard Rotary Positional Encoding (RoPE) suffers from positional misalignment and weakened long-range dependencies when applied across multiple regions. Additionally, to tackle the Tiny Ingredient Continuity Problem where small but critical ingredients may be visually inconsistent or even missing across steps, we design Cross-Step Consistency Control (CSCC). CSCC utilizes Contextual Step Tokens to promote visual consistency of shared ingredients without compromising step-wise independence.

### 3.1 Preliminary

Flux.1-dev. Flux.1-dev is a text-to-image model that generates a single high-quality image from a text prompt [11]. It replaces the UNet [33] in Stable Diffusion (SD) [31] with a Diffusion Transformer (DiT)[24] for better representation learning. For text encoding, it combines T5 [29] and CLIP [28] to improve text-image alignment. DiT performs joint self-attention over concatenated text and latent tokens, processing noisy latent tokens z ∈ R𝑁×𝑑 and text condition tokens C𝑇 ∈ R𝑀×𝑑, where 𝑑 is the embedding dimension, and 𝑁, 𝑀 are the numbers of image and text tokens.

RoPE. In the Flux.1-dev model, Rotary Position Embedding (RoPE) is employed to encode positional information for latent tokens. Given noisy latent tokens 𝑧, the positional encoding process can be mathematically expressed as:

𝑧𝑖,𝑗 = 𝑧𝑖,𝑗 · 𝑅(𝑖, 𝑗), (1)

where 𝑅(𝑖, 𝑗) is the rotation matrix corresponding to the position (𝑖, 𝑗), effectively encoding the spatial location of each token within the image. This approach enhances the model’s ability to capture spatial relationships and dependencies inherent in visual data.

Joint Attention. The joint attention mechanism maps positionencoded tokens into Query 𝑄, Key 𝐾, and Value 𝑉. Additionally, it concatenates the text tokens for attention calculation. The attention operation can be expressed as:

𝑄𝐾⊤

Attn([𝐶𝑇; 𝑧𝑖,𝑗]) = SoftMax

𝑉, (2)

√

𝑑

where 𝑄, 𝐾, and 𝑉 are the queries, keys, and values derived from the token embeddings, and 𝑑 represents the dimensionality of the embeddings. The concatenation of the image and text tokens, denoted as [𝐶𝑇; 𝑧𝑖,𝑗], facilitates the multi-modal attention mechanism, enabling joint attention across both modalities.

### 3.2 Step-wise Regional Control

Limitations of Flux.1-dev. Flux.1-dev is designed for generating a single image from a text prompt, making it inherently unsuitable for tasks that require a coherent sequence of step-wise images—such as recipe visualization. This application demands accurate per-step generation, global consistency, and visual diversity across steps. To adapt Flux.1-dev for multi-image generation, In-Context LoRA [9] proposes a simple strategy: concatenating all step-level prompts and jointly generating the full sequence of images in a single pass. While this method allows simultaneous generation, it lacks explicit steplevel separation, resulting in severe semantic entanglement across steps. Visual features from one instruction often leak into others, producing highly similar and indistinct images across steps.

We quantify this limitation using the Cross-Step Consistency (CSC) metric. The In-Context LoRA baseline yields a CSC score of 44.12—9.03 points lower than the ground truth score of 53.15—and the lowest among all evaluated methods (more details can be found in Tab. 2). This significant drop clearly indicates that without explicit step-level control, the model fails to maintain step-wise distinctiveness and coherence.

Step-wise Regional Control (SRC). To address the entanglement and lack of step-wise control in Flux.1-dev and its simple concatenation-based variants, we propose SRC-a novel mechanism

[Figure 2]

- Figure 2: Overall structure of our CookAnything model, illustrated with a 3-step vegetable pancake recipe. The Cooking Agent reformats the raw recipe into context-tagged steps, supplementing missing ingredient details. Each step is encoded by a T5 Encoder in two ways: (1) all steps are concatenated to capture global context and produce contextual step tokens, and (2) each step is encoded independently to preserve local semantics and generate step tokens. These two types of tokens are fused via weighted averaging. Meanwhile, noisy latent tokens, processed by Flexible RoPE, are fed into DiT. A Step-wise Regional Attention Mask is applied during DiT’s self-attention to constrain attention within each step, ensuring step-wise focus and visual consistency. In the illustration, purple, green, and pink tokens represent Steps 1, 2, and 3, respectively

that enables the model to synthesize a coherent sequence of semantically distinct step images within a single denoising process. SRC introduces architectural changes that explicitly bind each textual step instruction to a designated image region while preserving global coherence across the entire image. This allows for both localized control and smooth visual transitions, bridging the gap between single-image generation and structured multi-image synthesis.

Step-wise Encoding and Integration. SRC modifies the decoding pipeline of Flux.1-dev by introducing a Step-wise Encoding Mechanism and a Step-wise Regional Attention Mask. Specifically, each recipe step is first independently encoded using a shared text encoder, and the resulting step tokens are concatenated before the latent tokens:

𝐶input = 𝐶(1);𝐶(2); . . . ;𝐶(𝑁) , (3) 𝑋input = 𝐶input;𝑧input , (4)

where 𝐶(𝑛) denotes the encoded tokens of the 𝑛-th step and 𝑧input denotes the noisy latent tokens. Step-wise Regional Attention Mask. To prevent semantic leakage between steps and ensure localized step-to-region alignment, we design a Step-wise Regional Attention Mask 𝑀 ∈ R2𝑁×2𝑁 restricting attention within each step-region pair. Formally, it is defined as:

1 if 𝑖 = 𝑗 or |𝑖 − 𝑗| = 𝑁, 0 otherwise.

(5)

𝑀𝑖,𝑗 =

Here, 𝑖 and 𝑗 represent the 𝑖-th and 𝑗-th step/image token sets. The attention operation within DiT then becomes:

Attn(𝑄,𝐾,𝑉,𝑀) = Softmax

𝑄𝐾𝑇 √𝑑𝑘 ⊙ 𝑀 𝑉, (6)

ensuring each step attends only to its paired visual region and associated instruction tokens.

The updated regional latent representation at timestep 𝑡 − 1 is computed as:

𝑧𝑡region−1 =𝜓 Attn(𝑄𝑡region−1 ,𝐾region𝑡−1 ,𝑉region𝑡−1 ,𝑀region) , (7) where 𝜓 represents the DiT block from Flux.1-dev.

Whole-Description Control for Global Coherence. To complement regional specificity with global visual consistency, we incorporate a Whole-Description Control Mechanism that processes the full recipe description in parallel:

𝑧𝑡base−1 =𝜓(Attn(𝑄𝑡base−1,𝐾base𝑡−1,𝑉base𝑡−1,𝑀base)). (8) Finally, we fuse the regional and global latent representations

through a weighted interpolation:

𝑧𝑡 = 𝛼 · 𝑧𝑡base−1 + (1 − 𝛼) · 𝑧𝑡region−1 , (9) where 𝛼 ∈ [0, 1] controls the trade-off between global context and regional detail.

By preserving global structure while controlling step-wise semantics and region-level generation, SRC overcomes the limitations of prior methods and enables high-quality procedural image synthesis.

### 3.3 Flexible RoPE

Limitation of the Origin RoPE. When generating a flexible number of step-wise images, Flux.1-dev, which adopts the original Rotary Position Embedding (RoPE), encounters two key limitations, as illustrated in Fig. 3 using a Lamb Pilaf example. The first issue is the Misaligned Positional Embedding. When generating multiimages in a single forward pass using FLUX, the use of original

[Figure 3]

- Figure 3: The example from Original RoPE. Visualization comparison between original RoPE and our proposed Flexible RoPE using the example of Lamb Pilaf. With original RoPE, repeated step images appear as early as Step 2. Steps

- 3 and 6 exhibit positional misalignment, and Step 9 suffers from noticeable blurring. In contrast, Flexible RoPE maintains clear step-wise differentiation, stable spatial alignment, and improved visual sharpness throughout the cooking process.

RoPE causes the positional encoding across steps to remain entangled in a global coordinate frame. RoPE tends to overemphasize absolute positional alignment, which is suboptimal for tasks requiring local step-wise independence. As a result, each step image in our generation process is decoded from a similar latent spatial origin, leading to visual redundancy and layout collapse. In Fig. 3, Step 2 is visually repeated, and the semantic boundary between steps becomes ambiguous. The second issue is the Attenuation of Long-range Dependencies. As step count grows, the model struggles to maintain semantic consistency, causing blurry or collapsed outputs in later steps—clearly seen at Step 9 in Fig. 3.

To address this, we propose a step-aware positional encoding that explicitly re-initializes positional indices for each step, enabling the model to better capture both step-wise independence and inter-step coherence.

Flexible RoPE. Unlike standard RoPE, which uses a globally continuous encoding across all image regions and leads to entangled positional dependencies, our proposed Flexible RoPE assigns an independent positional encoding to each image region. This disentanglement allows the model to clearly differentiate the position and semantics of each step, reducing cross-region interference and preserving generation fidelity even when scaling to many steps.

[Figure 4]

Figure 4: Examples before and after applying Cross-Step Consistency Control (CSCC). Left: Stir-Fried Carrot with Dried Tofu. Without CSCC, the carrot changes from cubes to strips in Step 4. Visualization of contextual tokens (using Flux.1dev) shows shape continuity is preserved, so CSCC helps maintain a consistent appearance. Right: Steamed Chicken Wings with Taro. In Step 5, taro should appear beneath the wings but disappears without CSCC. Since contextual tokens confirm its presence, CSCC successfully preserves it.

Specifically, for each image region 𝑛, we apply a separate RoPE encoding:

𝑧𝑖,𝑗(𝑛) = 𝑧𝑖,𝑗(𝑛) · 𝑅(𝑛)(𝑖, 𝑗), (10) where 𝑅(𝑛)(𝑖, 𝑗) is the region-specific rotation matrix for the 𝑛-th image, applied to token 𝑧𝑖,𝑗(𝑛) at position (𝑖, 𝑗). This design ensures that each region learns positionally independent patterns, effectively preventing the model from inheriting noise or structural artifacts from adjacent steps.

We then concatenate all positionally encoded tokens as input to the model:

𝑧input = 𝑧(1); 𝑧(2); . . . ; 𝑧(𝑁) , (11)

where 𝑁 denotes the number of step-wise images. This forms a joint input sequence where each region maintains its own positional integrity while enabling global attention across regions.

In Fig.3, Flexible RoPE leads to significantly improved spatial alignment and visual consistency across steps, especially in long recipes, demonstrating its effectiveness and scalability.

### 3.4 Cross-Step Consistency Control

While generating all step-wise images in a single denoising pass enhances stylistic and background consistency, it struggles with recipes involving numerous small ingredients—especially in stir-fry dishes where chopped ingredients are scattered and repeatedly appear in small visual regions. In such cases, simultaneous denoising fails to preserve crucial visual attributes such as shape, color, and texture, and may even omit ingredients. We refer to this as the Tiny Ingredient Continuity Problem (see Fig. 4).

To address this challenge, we propose Cross-Step Consistency Control (CSCC), a lightweight yet effective solution that promotes the visual continuity of fine-grained ingredients across steps, while maintaining the independence of each step’s image content. Our

approach consists of two stages: (1) Contextual Step Token Extraction.

We encode the entire recipe by concatenating all step instructions and feeding them into a T5 encoder, generating a unified representation. Since recurring ingredients appear in multiple steps, their token representations inherently share semantic similarities. We then segment the token sequence according to step lengths to extract context-aware tokens for each step—retaining both global context and local specificity. Many recipe steps contain vague descriptions, such as "Pour the batter in the pan," without specifying the exact ingredients (e.g., carrot and zucchini strips). To address this, we introduce the Cooking Agent, based on GPT-4o [1], which supplements missing ingredient details for each step. The Cooking Agent fills in any implicit ingredient information not explicitly mentioned in the recipe text, while ensuring that descriptions—such as shape and color—remain consistent across steps. This consistency is crucial for maintaining ingredient coherence across images, enabling greater consistency in Contextual Step Token Extraction. As shown in Fig.4 and Supplementary A.3, these step-specific tokens effectively capture ingredient-level consistency, allowing for coherent representation of shared ingredients across steps. (2) ContextAware Fusion via Weighted Averaging. To reinforce ingredient continuity across steps, we combine the globally informed step tokens with the tokens generated from individually encoding each step from SRC. This combination is achieved through a weighted averaging approach, which strikes a balance between maintaining the unique details of each step and ensuring consistency in ingredient appearance. As a result, the model preserves fine-grained details—such as the color and shape of ingredients—across all generated images, while keeping the independence of each step’s content intact. The process can be represented as:

𝐶(𝑛) [0 : 𝑡(𝑛)] = 𝐶(𝑛) [0 : 𝑡(𝑛)] +𝜆 ∗𝐶𝑟𝑒𝑐𝑖𝑝𝑒[𝑏(𝑛) : 𝑏(𝑛) +𝑡(𝑛)], (12)

where𝐶(𝑛) represents the tokens obtained by individually decoding the𝑛-th step and𝐶𝑟𝑒𝑐𝑖𝑝𝑒 represents the tokens obtained by decoding the entire recipe together. 𝑏(𝑛) denotes the starting position of the 𝑛-th step in the token sequence 𝐶𝑟𝑒𝑐𝑖𝑝𝑒. 𝑡(𝑛) represents the length of the 𝑛-th step. 𝜆 is a weight factor that balances the contribution of the recipe-wide context and the individual step decoding.

- 4 Experiment

- 4.1 Experiment Settings

Datasets.We conductexperimentsonRecipeGen [46]andVGSI[48]. Details on how the two datasets are used can be found in Supplementary A.1.

ImplementationDetails.We evaluateourmodelin bothtrainingfree and training-based settings. Detail is in Supplementary A.2.

Evaluationmetrics.We evaluateCookAnythingontwo datasets: RecipeGen [46] and VGSI-Recipe [41], using the following metrics in Tab.1.

- 4.2 Quantitative Evaluation

We evaluate our model against a wide range of baselines, including UNet-based methods: StoryDiffusion [51], Stable Diffusion XL (SDXL) [26], and StackedDiffusion (SKD) [18]. We also compare with DiT-based models: Flux.1-dev , In-Context LoRA (IC-LoRA)

#### Table 1: Evaluation metrics for CookAnything.

Metric Description Step Flexibility

Indicates whether the model can generate a variable number of step images in a single pass.

Indicates whether the model can generate all step images simultaneously to ensure consistency and efficiency.

Joint Generation

CLIP [28] similarity between the final image and last-step caption, measuring alignment with the overall goal.

Goal Faithfulness

Assesses each image’s alignment with its step caption using CLIP and contextual consistency with the recipe via GPT-4o [1]. Detailed prompt can be found in the Supplementary A.4.

Step Faithfulness

Based on StackedDiffusion [18] and DINOv2 [3], uses 𝑙2 distance and step count difference to assess visual and numerical consistency.

Cross-Step Consistency

Uses GPT-4o and manual inspection to verify whether the expected ingredients are visually present in each step and to check for any omissions. Detailed prompt is in Supplementary A.4.

Ingredient Accuracy

Usability Evaluates spatial alignment in jointly generated images to avoid layout collapse or misalignment, using GPT-4o and human inspection. GPT-4o scores usability across five aspects: size consistency, step clarity, content duplication, process completeness, and step count reasonableness. Full results in Tab. 2, details in Supplementary A.4.

[9], Stable Diffusion 3.5 (SD3.5) [35] and Regional Prompt Flux (RPF) [4]. Additionally, we include layout-aware methods such as GLIGEN [13] and Attention Refocusing (A-R) [25].

As shown in Tab. 2, our CookingAnything model achieves state-of-the-art results on the RecipeGen dataset, surpassing all baselines across key metrics. It excels in Goal Faithfulness (GF) and Step Faithfulness (SF), indicating precise visual-step alignment, and achieves the lowest Cross-Step Consistency (CSC), reflecting strong procedural coherence. Additionally, it leads in Ingredient Accuracy (IA) and Usability (UB) under both GPT-based (G) and Human Evaluation (H), demonstrating its ability to capture fine-grained details and deliver user-preferred outputs.

We further validate our model on on the VGSI-Recipe dataset using both Training-Free (TF) and Training-Based (TB) settings. As shown in Tab. 3, our method consistently achieves the best performance across all metrics. These results highlight the effectiveness of our approach in producing accurate, and coherent recipe visualizations across diverse scenarios.

### 4.3 Ablation Study

Our main experiments are conducted in the Training-Based Mode, while the ablation study on the Training-Free Mode is provided in Supplementary Section A.7.

Effectiveness of Cross-Step Consistent Control. We propose the Cross-Step Consistent Control (CSCC) to ensure ingredient

[Figure 5]

Figure 5: Qualitative comparisons. SKD refers to StackedDiffusion, and SD3.5 refers to Stable Diffusion 3.5. Both SD3.5 Flux.1-dev and SKD exhibit issues with ingredient accuracy, discontinuous ingredient shapes, and the generation of incorrect ingredients. In contrast, our model excels in maintaining the shape and continuity of ingredients.

consistency across steps. As shown in Tab. 4, removing CSCC leads to a notable drop in Cross-Step Consistency (CSC), demonstrating the effectiveness of our design. We also examine the effect of the hyperparameter 𝜆 in Equation 5, balancing the regional and contextenhanced prompts. Testing 𝜆 values from 0, 0.2, 0.4, 0.6, 0.8, 1 (Fig. ?? (d)-(f)) reveals 𝜆 = 0.2 as optimal, so we set 𝜆 = 0.2 in this paper.

Effectiveness of Flexible RoPE. We evaluate Flexible RoPE by replacing it with the original RoPE. As shown in Tab.4, removing Flexible RoPE causes a notable drop in Goal Faithfulness (1.93) and Step Faithfulness (0.98), and introduces visual inconsistencies across steps. Specifically, consecutive images exhibit blurred transitions, making step boundaries harder to distinguish (Fig.3).

Effectiveness of Cooking Agent. We remove the Cooking Agent and test on diverse recipe steps, including those with vague or implicit ingredient descriptions. As shown in Tab.4, CookingAnything still outperforms all other models in Tab.2, showing its ability to understand recipe text and perform well even without explicit instructions.

Evaluation on Variable-Length Recipes. We evaluate the performance of our model, as well as In-Context LoRA, Regional

Prompting FLUX, and Flux.1-dev, on recipes ranging from 3 to 10 steps. Detailed results can be found in Supplementary Section A.7.

### 4.4 Qualitative Evaluation

Fig.5 presents qualitative comparisons between our model and other DiT-based methods, including StackedDiffusion, Stable Diffusion

- 3.5 and Flux.1-dev, covering Western cuisine, and Asian dishes. Our approach consistently demonstrates superior cross-step consistency in terms of both global scene layout and fine-grained visual fidelity. For instance, it accurately maintains the shape of the soufflé and the structure of the salmon and avocado. Furthermore, our generations are more faithful to the recipe content and step-wise instructions. In contrast, other methods frequently suffer from ingredient omissions or hallucinations and show inconsistency in both background and ingredient appearance throughout the sequence.
- 4.5 User Study on Perceptual Quality

To evaluate the perceptual quality of our generated images, we conducted a user study involving 53 participants on 75 questions,

#### Table 2: Comparison with Other Models in RecipeGen. SF means Step Faithfulness, IA means Ingredient Accuracy and UB means Usability. (C) means CLIP score, (G) means GPT-Score and (H) is Human evaluation. The UB metric is applicable only to methods capable of Joint Generation. TF means Traning-Free and TB means Traning-Based.

SF↑ IA↑ UB ↑ (C) (G) (G) (H) (G) (H)

Cross-Step Consistency↓

Step Flexibility

Joint Generation

Goal Faithfulness ↑

Category Method

StoryDiffusion [51] 17.51 6.01 25.54 5.30 7.25 3.41 — SDXL [26] 27.46 2.98 29.37 6.79 7.51 3.71 — SKD [18] 26.62 0.7 28.53 4.57 6.67 2.59 6.43 3.59

UNet-based

SD3.5 [35] 27.42 2.97 28.77 6.73 7.58 3.97 — Flux.1-dev [11] 26.47 3.47 28.31 5.31 5.93 5.71 — IC-LoRA [9] 26.07 9.03 26.58 4.03 5.50 3.91 5.34 4.45 RPF [4] 27.19 8.73 25.99 4.45 7.05 3.02 3.89 4.24

DiT-based

GLIGEN [13] 26.99 2.17 26.72 5.17 6.16 5.28 — A-R [25] 26.31 2.46 27.63 4.58 5.46 5.29 — —

Layout-aware

Ours (TF) 30.12 0.17 29.80 8.52 9.12 6.92 9.89 7.66 Ours (TB) 30.59 0.19 30.45 8.69 9.27 7.15 9.70 8.48

DiT-based

#### Table 3: Comparison with Other Models in VGSI-Recipe. SF (C) and SF (F) denote CLIP-based and GPT-based Step Faithfulness, respectively.

Method GF↑ SF(C)↑ SF(G)↑ CSC↓ IA↑ UB↑

SD1.5 [32] 27.03 26.63 3.41 12.28 5.32 SD2.1 [34] 27.03 26.63 3.32 11.94 5.01 SDXL [26] 27.78 28.32 3.60 11.7 5.87 SD3.5 [35] 26.12 27.03 3.43 5.27 5.17 Flux.1-dev [11] 25.82 27.77 2.99 5.97 4.55 IC-LoRA [9] 26.18 28.25 3.77 4.88 5.74 6.06 SKD [18] 28.25 28.26 4.22 3.14 6.35 5.75 RPF [4] 28.40 26.54 3.25 7.12 6.03 3.96 GLIGEN [13] 29.70 29.48 5.01 4.08 6.93 A-R [25] 28.87 28.34 4.32 3.94 6.31 —

Ours (TF) 31.22 29.61 7.12 1.67 8.42 9.06 Ours (TB) 29.88 29.71 6.63 2.26 8.07 7.72

#### Table 4: Ablation Study. F-RoPE refers to the Flexible RoPE we proposed, CSCC stands for Cross-Step Consistency Control, and C-Agent refers to the Cooking Agent. SF (C) and SF (F) denote CLIP-based and GPT-based Step Faithfulness, respectively.

##### Method GF ↑ SF(C)↑ SF(G)↑ CSC ↓

w/o F-RoPE 28.66 29.33 7.93 0.23 w/o CSCC 30.57 30.28 8.67 0.29 w/o C-Agent 29.00 29.59 7.84 3.06

Ours 30.59 30.45 8.69 0.19

assessing five key aspects: Cross-Step Consistency (CSC), Step Faithfulness (SF), Goal Faithfulness (GF), Aesthetic Quality (AQ), and Overall Appeal (OA). As summarized in Tab. 5, our method consistently surpasses Stable Diffusion 3.5 and Flux.1-dev across all metrics, achieving a significantly higher Aesthetic Quality score of 60.38.

These results highlight the superior visual fidelity and user preference of our approach. Further details are provided in Supplementary A.9.

#### Table 5: Human Evaluation on Perceptual Quality.

Method GF ↑ SF(C)↑ CSC ↑ AQ ↑ OA↑ SD3.5 12.08 15.60 12.70 13.58 13.46 Flux.1-dev 17.11 16.23 17.61 26.34 16.86 Ours 70.82 68.18 69.69 60.38 69.68

### 5 Conclusion

In this work, we present the CookAnything, a novel framework for flexible, high-fidelity illustrated recipe generation from stepwise textual instructions. By integrating Step-wise Regional Control, Flexible RoPE, and Cross-Step Consistency Control, our approach addresses key limitations of prior methods, achieving accurate semantic alignment, step-wise visual disentanglement, and finegrained ingredient continuity within a unified generation process. Extensive evaluations demonstrate that CookAnything not only produces visually coherent and semantically diverse step images, but also scales effectively to variable-length recipes under both Training-Based and Training-Free settings.

### Acknowledgments

This work was funded by the National Natural Science Foundation of China (Grant No. 62406126) and the Scientific Research Project of the Education Department of Jilin Province (Grant No. JJKH20250119KJ).

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774

(2023).

- [2] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. 2014. Food-101–mining discriminative components with random forests. In Proceedings of the European Conference on Computer Vision. Springer, 446–461.
- [3] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. 2021. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision. 9650–9660.
- [4] Anthony Chen, Jianjin Xu, Wenzhao Zheng, Gaole Dai, Yida Wang, Renrui Zhang, Haofan Wang, and Shanghang Zhang. 2024. Training-free regional prompting for diffusion transformers. arXiv preprint arXiv:2411.02395 (2024).
- [5] Jingjing Chen and Chong-Wah Ngo. 2016. Deep-based ingredient recognition for cooking recipe retrieval. In Proceedings of the 24th ACM international conference on Multimedia. 32–41.
- [6] Mengmeng Ge, Xu Jia, Takashi Isobe, Xiaomin Li, Qinghe Wang, Jing Mu, Dong Zhou, Li Wang, Huchuan Lu, Lu Tian, et al. 2024. Customizing text-to-image generation with inverted interaction. In Proceedings of the 32nd ACM International Conference on Multimedia. 10901–10909.
- [7] Yinxuan Gui, Bin Zhu, Jingjing Chen, Chong Wah Ngo, and Yu-Gang Jiang. 2024. Navigating weight prediction with diet diary. In Proceedings of the 32nd ACM International Conference on Multimedia. 127–136.
- [8] Fangda Han, Ricardo Guerrero, and Vladimir Pavlovic. 2020. CookGAN: Meal image synthesis from ingredients. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. 1450–1458.
- [9] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. 2024. In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775 (2024).
- [10] Shuqiang Jiang, Weiqing Min, Linhu Liu, and Zhengdong Luo. 2020. Multi-Scale Multi-View Deep Feature Aggregation for Food Recognition. IEEE Transactions on Image Processing 29 (2020), 265–276. doi:10.1109/TIP.2019.2929447
- [11] Black Forest Labs. 2024. Flux.1 AI. https://flux1ai.com/. Accessed: 2025-04-04.
- [12] Xing Lan, Jiayi Lyu, Hanyu Jiang, Kun Dong, Zehai Niu, Yi Zhang, and Jian Xue.

2023. Foodsam: Any food segmentation. IEEE Transactions on Multimedia (2023).

- [13] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. 2023. Gligen: Open-set grounded text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22511–22521.
- [14] Zhiming Liu, Kai Niu, and Zhiqiang He. 2023. ML-CookGAN: Multi-label generative adversarial network for food image generation. ACM Transactions on Multimedia Computing, Communications and Applications 19, 2s (2023), 1–21.
- [15] Adyasha Maharana, Darryl Hannan, and Mohit Bansal. 2022. Storydall-e: Adapting pretrained text-to-image transformers for story continuation. In European conference on computer vision. Springer, 70–87.
- [16] Jonathan Malmaud, Earl Wagner, Nancy Chang, and Kevin Murphy. 2014. Cooking with semantics. In Proceedings of the ACL 2014 Workshop on Semantic Parsing. 33–38.
- [17] Javier Marın, Aritro Biswas, Ferda Ofli, Nicholas Hynes, Amaia Salvador, Yusuf Aytar, Ingmar Weber, and Antonio Torralba. 2021. Recipe1m+: A dataset for learning cross-modal embeddings for cooking recipes and food images. IEEE Transactions on Pattern Analysis and Machine Intelligence 43, 1 (2021), 187–203.
- [18] Sachit Menon, Ishan Misra, and Rohit Girdhar. 2024. Generating Illustrated Instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6274–6284.
- [19] Weiqing Min, Linhu Liu, Zhiling Wang, Zhengdong Luo, Xiaoming Wei, Xiaolin Wei, and Shuqiang Jiang. 2020. Isia food-500: A dataset for large-scale food recognition via stacked global-local attention network. In Proceedings of the 28th ACM International Conference on Multimedia. 393–401.
- [20] Weiqing Min, Zhiling Wang, Yuxin Liu, Mengjiang Luo, Liping Kang, Xiaoming Wei, Xiaolin Wei, and Shuqiang Jiang. 2023. Large scale visual food recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence 45, 8 (2023), 9932– 9949.
- [21] Liang-Ming Pan, Jingjing Chen, Jianlong Wu, Shaoteng Liu, Chong-Wah Ngo, Min-Yen Kan, Yugang Jiang, and Tat-Seng Chua. 2020. Multi-modal cooking workflow construction for food recipes. In Proceedings of the 28th ACM International Conference on Multimedia. 1132–1141.

- [22] Liang-Ming Pan, Jingjing Chen, Jianlong Wu, Shaoteng Liu, Chong-Wah Ngo, Min-Yen Kan, Yugang Jiang, and Tat-Seng Chua. 2020. Multi-modal cooking workflow construction for food recipes. In Proceedings of the 28th ACM International Conference on Multimedia. 1132–1141.
- [23] Siyuan Pan, Ling Dai, Xuhong Hou, Huating Li, and Bin Sheng. 2020. ChefGAN: Food image generation from recipes. In Proceedings of the 28th ACM International Conference on Multimedia. 4244–4252.
- [24] William Peebles and Saining Xie. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision. 4195–4205.
- [25] Quynh Phung, Songwei Ge, and Jia-Bin Huang. 2024. Grounded text-to-image synthesis with attention refocusing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7932–7942.
- [26] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. 2023. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952

(2023).

- [27] Leigang Qu, Shengqiong Wu, Hao Fei, Liqiang Nie, and Tat-Seng Chua. 2023. Layoutllm-t2i: Eliciting layout guidance from llm for text-to-image generation. In Proceedings of the 31st ACM International Conference on Multimedia. 643–654.
- [28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR, 8748–8763.
- [29] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research 21, 140 (2020), 1–67.
- [30] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.
- [31] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.
- [32] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.
- [33] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer, 234–241.
- [34] Stability AI. 2022. Stable Diffusion v2.1 Model Card. https://huggingface.co/ stabilityai/stable-diffusion-2-1. Accessed: 2025-04-11.
- [35] Stability AI. 2024. Stable Diffusion 3.5 Large Model Card. https://huggingface. co/stabilityai/stable-diffusion-3.5-large. Accessed: 2025-04-11.
- [36] Yu Sugiyama and Keiji Yanai. 2021. Cross-modal recipe embeddings by disentangling recipe contents and dish styles. In Proceedings of the 29th ACM International Conference on Multimedia. 2501–2509.
- [37] Quin Thames, Arjun Karpur, Wade Norris, Fangting Xia, Liviu Panait, Tobias Weyand, and Jack Sim. 2021. Nutrition5k: Towards automatic nutritional understanding of generic food. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 8903–8911.
- [38] Liangyu Wang, Yoko Yamakata, Ryoma Maeda, and Kiyoharu Aizawa. 2024. Measure and Improve Your Food: Ingredient Estimation Based Nutrition Calculator. In Proceedings of the 32nd ACM International Conference on Multimedia. 11273–11275.
- [39] Xiongwei Wu, Xin Fu, Ying Liu, Ee-Peng Lim, Steven CH Hoi, and Qianru Sun.

2021. A large-scale benchmark for food image segmentation. In Proceedings of the 29th ACM international conference on multimedia. 506–515.

- [40] Yoko Yamakata, Akihisa Ishino, Akiko Sunto, Sosuke Amano, and Kiyoharu Aizawa. 2022. Recipe-oriented food logging for nutritional management. In Proceedings of the 30th ACM International Conference on Multimedia. 6898–6904.
- [41] Yue Yang, Artemis Panagopoulou, Qing Lyu, Li Zhang, Mark Yatskar, and Chris Callison-Burch. 2021. Visual Goal-Step Inference using wikiHow. arXiv preprint arXiv:2104.05845 (2021).
- [42] Yuehao Yin, Huiyan Qi, Bin Zhu, Jingjing Chen, Yu-Gang Jiang, and Chong-Wah Ngo. 2023. Foodlmm: A versatile food assistant using large multi-modal model. arXiv preprint arXiv:2312.14991 (2023).
- [43] Takuya Yonezawa, Yuanyuan Wang, Yukiko Kawai, and Kazutoshi Sumiya. 2019. A cooking support system by extracting difficult scenes for cooking operations from recipe short videos. In Proceedings of the 27th ACM International Conference on Multimedia. 2225–2227.
- [44] Ruoxuan Zhang, Dantong Ouyang, Lili He, Lingjin Kuang, and Hongtao Bai.

2024. Recognize after early fusion: the Chinese food recognition based on the alignment of image and ingredients. Multimedia Systems 30, 2 (2024), 93.

- [45] Ruoxuan Zhang, Dantong Ouyang, Ximing Li, Hongtao Bai, Chenming Zhang, and Lili He. 2025. Learning multi-scale features automatically from food and ingredients. Multimedia Systems 31, 3 (2025), 1–11.
- [46] Ruoxuan Zhang, Hongxia Xie, Yi Yao, Jian-Yu Jiang-Lin, Bin Wen, Ling Lo, HongHan Shuai, Yung-Hui Li, and Wen-Huang Cheng. 2025. RecipeGen: A Benchmark for Real-World Recipe Image Generation. arXiv preprint arXiv:2503.05228 (2025).
- [47] Yixin Zhang, Yoko Yamakata, and Keishi Tajima. 2022. Miais: a multimedia recipe dataset with ingredient annotation at each instructional step. In Proceedings of the 1st International Workshop on Multimedia for Cooking, Eating, and related APPlications. 49–52.
- [48] Luowei Zhou, Chenliang Xu, and Jason Corso. 2018. Towards automatic learning of procedures from web instructional videos. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 32.
- [49] Pengfei Zhou, Weiqing Min, Chaoran Fu, Ying Jin, Mingyu Huang, Xiangyang Li, Shuhuan Mei, and Shuqiang Jiang. 2024. FoodSky: A Food-oriented Large Language Model that Passes the Chef and Dietetic Examination. arXiv preprint arXiv:2406.10261 (2024).
- [50] Pengfei Zhou, Weiqing Min, Yang Zhang, Jiajun Song, Ying Jin, and Shuqiang Jiang. 2023. SeeDS: Semantic separable diffusion synthesizer for zero-shot food detection. In Proceedings of the 31st ACM International Conference on Multimedia. 8157–8166.
- [51] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou.

2024. Storydiffusion: Consistent self-attention for long-range image and video generation. Advances in Neural Information Processing Systems 37 (2024), 110315– 110340.

### 6 Supplementary Material

Sec.6.1 introduces the two datasets used in our work: RecipeGen and VGSI. Sec.6.2 provides a detailed description of our experimental settings. Sec.6.3 presents experiments specifically conducted on Contextual Step Tokens, evaluated using three key metrics. Sec.6.4 elaborates on the definitions and usage of the Ingredient Accuracy and Usability metrics. Sec.6.5 describes the detailed prompts used for our Cooking Agent. Sec.6.6 lists the specific recipe steps corresponding to the images shown in Fig.1 of the main paper. Sec.6.7 presents additional ablation studies. Sec.6.8 provides more qualitative visualization results. Sec.6.9 describes the setup of our human evaluation study. Finally, Sec. 6.10 discusses potential directions for future work.

### 6.1 Datasets Details

In our experiments, we utilize two datasets: RecipeGen [46] and VGSI [48]. To the best of our knowledge, RecipeGen represents the first and currently the only large-scale dataset specifically constructed for the task of recipe image generation. It spans a wide range of cooking types and regional cuisines, including both liquidbased recipes and solid dishes. The number of steps per recipe is widely distributed (ranging from 2 to 15), which facilitates flexible multi-step image generation. The dataset consists of 21,944 recipes. For our experiments, we randomly sample 5,000 recipes from the training split for model training, and we use the official test split, comprising 4,389 recipes, for evaluation.

VGSI is a visual goal-step instruction dataset collected from WikiHow, where recipe-related samples represent only a small subset of the overall data. Compared to RecipeGen, VGSI encompasses fewer cooking types and exhibits more limited visual diversity, with a number of samples presented in a comic style. To focus on reciperelated content, we filter VGSI by the keyword “cook”, resulting in 1,157 recipes with a total of 6,417 images. We then apply an 85:15 split, randomly selecting 173 recipes as the test set.

### 6.2 Implementation Details

We evaluate our model under two distinct settings: training-free and training-based. In the training-based framework, individual step images are standardized to a resolution of 512×512 pixels. These images are then concatenated vertically to yield a complete multi-step visual sequence, ensuring spatial consistency across the recipe’s steps.

For the textual input, we incorporate the entire recipe text. To improve the model’s comprehension, we generate a concise recipe summary using GPT-4o, which is prepended to the full text. Moreover, each step description is explicitly marked with a prefix in the format “[step-i]”, denoting its corresponding step number. This results in a final input structure consisting of the GPT-4o-produced summary followed by all step-wise instructions.

Training is carried out on a single A100 GPU for 20,000 training steps, using a batch size of 2. We train on both the VGSI-recipe and RecipeGen datasets. Our implementation is built upon the Flux.1-dev variant, which consists of 19 DoubleStreamBlocks and 38 SingleStreamBlocks, aggregating to approximately 12 billion parameters. We apply LoRA with a rank of 16 to adapt the model, and optimization is performed using the Adam optimizer with

#### Table 6: Experiments in Contextual Tokens.

Goal Faithfulness ↑

Step Faithfulness↑

Cross-Step

Consistency ↓ 29.91 28.70 1.8

#### Table 7: Detailed result in Usability.

##### Method ISC↑ CSR↑ DIC↑ PCL↑ RNS↑

IC-LoRA 1.98 0.86 0.92 0.79 0.78 SKD 2.00 1.07 1.29 1.04 1.03 RPF 1.75 0.52 0.63 0.50 0.50

Ours (TF) 1.99 1.98 1.97 1.98 1.97 Ours (TB) 1.97 1.96 1.85 1.96 1.96

an initial learning rate of 0.0001. We set 𝛼 = 0.1 and 𝜆 = 0.1 in CookAnything.

### 6.3 Experiments on Contextual Step Token

In CookAnything, we concatenate all step instructions into a single sequence and pass it through the T5 encoder to obtain a unified token representation for the entire recipe. We then split the tokens for each step. In this section, we visualize the tokens representing each step and test the Goal and Step Faithfulness, as well as CrossStep Consistency. This approach, where we only visualize the step tokens after the entire decoding process, results in performance metrics that outperform all models listed in the Tab.6, except for CookAnything.

### 6.4 More details about Step Faithfulness, Ingredient Accuracy, and Usability

In addition to using CLIP to compute Step Faithfulness, we also employ GPT-4o to assess the semantic alignment between the generated image and the original recipe text. Specifically, we design a five-level scoring system to evaluate whether the ingredient shapes, containers, and states depicted in the image accurately correspond to those described in the respective recipe step. The prompt for GPT-4o is in Fig.9. . Ingredient Accuracy evaluates whether each step includes the correct and relevant ingredients. Existing models often suffer from ingredient omission, confusion, or hallucination, especially in multi-step cooking processes where ingredients appear, transform, or disappear over time. To address this, we assess both the presence and correctness of visible ingredients at each step, using a combination of GPT-based scoring and human verification. The prompt used for Ingredient Accuracy is in Fig.10.

As shown in Fig.11, we evaluate the Usability based on five key aspects. First, Image Size Consistency (ISC) ensures that the dimensions of all sub-images are uniform, with no issues such as incorrect cropping or inconsistent sizes that could hinder understanding. Second, Clarity of Step Representation (CSR) assesses whether each sub-image clearly represents a distinct cooking step and aligns with a specific step in the recipe, making it easy for users to follow the process. Third, Duplication of Image Content (DIC) checks for repetition in image content, such as identical

perspectives or similar compositions, ensuring that the images remain diverse and informative. Fourth, Process Completeness and Logic (PCL) evaluates whether the image sequence accurately shows the entire recipe process—from preparation to the finished product—and whether the sequence logically matches the recipe text. Finally, Reasonableness of Number of Steps (RNS) verifies whether the number of sub-images aligns with the number of steps in the recipe, ensuring the image sequence is neither too sparse nor overloaded. These criteria together ensure that the generated images effectively represent the cooking process, are logically structured, and provide clear guidance for the user. The detailed result is in Tab.7. CookAnything performs the best on CSR, DIC, PCL, and RNS. However, the SKD model performs well in Image Size Consistency because it is fixed to output only six images, which allows for consistent cropping and uniform image sizes. Despite this, its output does not align with the number of steps in the text.

Details about Human Evaluation in Ingredient Accuracy, and Usability. We conducted an evaluation on RecipeGen, where we selected 780 recipes and over 5000 images from 12 models for assessing ingredient accuracy. These were rated by 13 evaluators. Additionally, for usability, we chose 1300 recipes from four models to conduct a comprehensive review. The evaluation focused on various aspects of usability, such as the clarity of step representation, image consistency, and the logical flow of the recipe process. This thorough testing across both ingredient accuracy and usability helps ensure that our models not only generate correct ingredient representations but also provide a seamless user experience in terms of clarity and coherence.

### 6.5 Prompt for Cooking Agent

We employ GPT-4o as our Cooking Agent to generate step-wise visual descriptions of recipe images. To ensure consistency across steps, we design two specialized prompts that guide the language model to produce structured and detail-rich outputs. A key objective of this design is to maintain descriptive consistency: once an ingredient is described in a particular form—e.g., “thinly sliced cucumber” in Step 1—the same terminology is preserved in subsequent steps unless its physical state has been explicitly altered (e.g., stir-fried, softened, browned). This consistency is critical for our Cross-Step Consistency Control (CSCC) block, which aligns visual representations across multiple image. The prompt is in Fig.8

### 6.6 Specific Recipe for Fig.1 and 10 in Main Paper

There are prompts for examples in Fig.1 of main paper: Example 1: <Goal>: Broccoli Egg Salad. <Step-1> Blanching broccoli: Boil water in a pot, add some salt and cooking oil, and blanch the cleaned broccoli until cooked, then remove and set aside. <Step-2> Preparing ingredients: Add the cut pieces of boiled eggs. <Step-3> Final dish: Finished dish.

Example 2: <Goal>: Hamburger. <Step-1>: Knead the dough: Mix all the ingredients except the butter and sesame seeds, and slowly add warm water while stirring with chopsticks into a flocculent state, then knead by hand to form a dough. Once it reaches the initial expansion stage, add the butter and continue kneading until the dough forms a glove film. <Step-2> Treat the dough: Deflate

the fermented dough, evenly divide it into 4 portions, cover with cling film, and let rest for 10 minutes. Round the rested dough, brush with water, coat with sesame seeds, and let it ferment in the oven for another 30 minutes. <Step-3> Bake the dough: Place the fermented dough into a preheated oven at 160°C upper tube and 140°C lower tube, bake for about 16 minutes until done. The burger buns are now ready. <Step-4> Pan-fry patties and eggs: Shape the mixed chicken into patties, pan-fry until golden brown on both sides, then fry the eggs until done and set aside. <Step-5> Assemble the burger: Slice the burger bun horizontally in two, place a lettuce leaf and a chicken patty on one half, squeeze a little ketchup on the patty, then add cheese, egg, bell pepper rings, and top with the other bun half.

- Example 3: <Goal>: Mexican Chicken Wrap. <Step-1>: Prepare the tortilla: I used semi-finished tortillas, slightly fry them or microwave for one minute. <Step-2>: Prepare the vegetables: Get lettuce and carrots ready. <Step-3>: Fry chicken breast: Fry until golden brown (double fry for a crispier texture). <Step-4> Lay out tortilla: Place lettuce leaves and carrots on one side of the tortilla. <Step-5>Add chicken breast: Add the fried chicken breast to the tortilla. <Step-6> Roll the tortilla: Roll up the tortilla. Cut the tortilla in half, and the Mexican chicken wrap is ready.
- Example 4: <Goal>: Tomato and Scrambled Egg Rice. <Step-1> Prepare ingredients: Rinse the rice and cook it in a rice cooker; beat eggs with a pinch of salt. <Step-2> Cook the eggs: Heat oil in a pan, pour in the beaten eggs, quickly stir to scramble, and set aside. <Step-3> Handle tomatoes: Make cross cuts on tomatoes, blanch in hot water to remove the skins, and cut into pieces. <Step-4> Sauté garlic and scallions: Heat oil in a pan, sauté garlic slices and white part of scallions on low heat until fragrant. <Step-5> Cook tomatoes: Add tomatoes, stir-fry until juicy, then add sugar and salt, and stir evenly. <Step-6> Combine ingredients and reduce the sauce: Add the scrambled eggs and cook until the sauce thickens, adding a bit of water if necessary. <Step-7> Finish the dish: Pour the cooked tomato and eggs mixture over the steamed rice, and sprinkle with chopped scallions.
- Example 5: <Goal>: Dried Fruit Pound Cake. <Step-1> Prepare ingredients: Prepare the ingredients. <Step-2> Dried fruits: Process and cut the dried fruits into small pieces. <Step-3> Melt and mix butter: Melt butter over water bath, beat with a whisk at low speed until smooth. Add salt and powdered sugar, mix briefly, then beat at low speed until combined. <Step-4> Add egg mixture: Beat the eggs. Gradually add the beaten eggs to the butter in three portions, mixing well at low speed each time. <Step-5> Mix flour and dried fruits: Sift in the mixture of low-gluten flour, baking powder, and almond flour. Fold with a spatula until it’s smooth. Then fold in the dried fruits. <Step-6> Prepare for baking: Line the mold with parchment paper, and preheat the oven to 135°C. Pour the cake batter into the mold, smooth the surface, and tap the mold lightly to remove air bubbles. <Step-7> Bake the cake: Place the mold in the oven at 135°C on the middle-lower rack for 30 minutes. Insert a toothpick into the cake; if it comes out clean, the cake is done. <Step8> Cool and cut: Take the cake out of the oven, with a beautiful golden color. Let it cool slightly and then cut into pieces.

There are example for fig.10 of main paper: Example (a): <Goal>: How to Eat Kimchi. <Step-1> Eat kimchi

out of the jar for an effortless snack.<Step-2> Serve individual

#### Table 8: Effect of 𝜆 on Generation Performance.

Goal Faithfulness ↑

Step Faithfulness↑

Cross-Step Consistency ↓

𝜆

0 29.99 29.79 0.19 0.2 30.12 29.80 0.17 0.4 29.99 29.70 0.3 0.6 29.78 29.43 0.81

- 0.8 29.39 29.01 1.66
- 1.0 29.18 28.87 3.15

pieces of kimchi with toothpicks to easily share it. <Step-3> You can eat kimchi straight out of the fridge, or you can throw it in a small skillet and heat it up with 1 US tbsp (15 mL) of vegetable oil.

- Example (b): <Goal>: How to Use Emu Oil for Health and Skin Benefits. <Step-1> Place a small dab of the oil on the palm of your hand or on the affected area and rub it in until its clear. Within a short amount of time, you should feel relief and notice swelling go down. You may apply emu oil when your back or neck feels swollen or sore. Emu oil can be purchased online or in your local pharmacy. Use the oil once or twice a day. <Step-2> Emu oil has a slight pain-killing effect when applied to the skin, so rub it onto a scrape or bruise to reduce your pain once a day. The antioxidants found in the oil can also help prevent additional damage or further infection. Seek medical help if you have large, deep cuts. <Step-3> Gently rub the area with the emu oil once a day until its completely absorbed by your skin. The oil will reach deep into your skin and alleviate the pain quickly while the sunburn heals. Speak with your doctor or dermatologist to determine if emu oil is a good option for you. Have a friend help you get the oil on hard-to-reach areas such as your back. You can also use emu oil as a natural sunscreen. Apply the oil as you would with a regular sunscreen.
- Example (c): <Goal>:How can I repair peeling for a faux leather sofa with a vinyl repair kit? <Step-1>: Mix paint colors until they match the sofa. <Step-2>: Brush paint onto the affected area. <Step3>: Apply texture relief paper to the paint, if desired. <Step-4>: Use a heat tool with the paper for 2 minutes and finish.The sofa looks as good as new.

### 6.7 More Ablation Study

Table 8 presents the ablation results of our Training-Free module under different levels of contextual integration, controlled by the hyperparameter 𝜆. As 𝜆 increases from 0 to 1, we observe a gradual degradation in performance for all three metrics: Goal Faithfulness (GF), Step Faithfulness (SF), and Cross-Step Consistency (CSC). The best overall performance is achieved when 𝜆 = 0.2, which yields the highest GF and SF scores (30.12 and 29.80, respectively) and the lowest CSC value (0.17).

### 6.8 More Visualization Result

In this section, we visualize results across three major food categories. First, beyond the Asian and Western dishes already shown in Fig.1, 3, 4, and 6 of the main paper, we present additional examples of diverse regional cuisines in Fig.6, demonstrating the model’s adaptability to various cultural food styles. Second, we showcase our model’s performance on liquid-based dishes in Fig.7, which

often pose challenges due to their fluid textures and fine-grained visual details.

- 6.9 User Study The question for User Study is in Fig.12.
- 6.10 Future Work

Beyond cooking, our method lays a foundation for structured visual generation in broader procedural domains such as instructional manuals, scientific workflows, and educational storytelling. Future work will explore extending our framework to multimodal video generation, interactive editing, and alignment with real-world cooking data.

[Figure 6]

#### Figure 6: Visualization of dishes from different regions.

[Figure 7]

Figure 7: Visualization about liquid.

[Figure 8]

Figure 8: Prompt for refining recipe caption, with GPT-4o

[Figure 9]

#### Figure 9: Prompt to measure the Step Faithfulness of the generated image, with GPT-4o

[Figure 10]

#### Figure 10: Prompt to measure the Ingredient Accuracy of the generated image, with GPT-4o

[Figure 11]

#### Figure 11: Prompt to measure the Usability of the generated image, with GPT-4o

[Figure 12]

#### Figure 12: Template for Human Study.

