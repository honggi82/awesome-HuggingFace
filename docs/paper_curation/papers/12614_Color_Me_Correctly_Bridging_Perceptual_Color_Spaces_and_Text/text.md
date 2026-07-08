# arXiv:2509.10058v1[cs.CV]12Sep2025

## Color Me Correctly: Bridging Perceptual Color Spaces and Text Embeddings for Improved Diffusion Generation

Sung-Lin Tsai

National Yang Ming Chiao Tung University Hsinchu, Taiwan tsai412504004.ee12@nycu.edu.tw

Bo-Lun Huang

National Yang Ming Chiao Tung University Hsinchu, Taiwan kevin503.ee12@nycu.edu.tw

Yu-Ting Shen∗

National Yang Ming Chiao Tung University Hsinchu, Taiwan yuting89830.cs11@nycu.edu.tw

Cheng-Yu Yeo∗

National Yang Ming Chiao Tung University Hsinchu, Taiwan boyyeo123.ee12@nycu.edu.tw

Chiang Tseng∗

National Yang Ming Chiao Tung University Hsinchu, Taiwan chiang.ee11@nycu.edu.tw

Bo-Kai Ruan∗

National Yang Ming Chiao Tung University Hsinchu, Taiwan bkruan.ee11@nycu.edu.tw

Hong-Han Shuai†

Wen-Sheng Lien

National Yang Ming Chiao Tung University Hsinchu, Taiwan hhshuai@nycu.edu.tw

National Yang Ming Chiao Tung University Hsinchu, Taiwan vincentlien.ii13@nycu.edu.tw

Blended Color Modified Color Object Color Signature Color Abstract Color

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

| |
|---|

SynGen

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Ours

"A young orange red-haired girl is finishing her lunch."

"A child in a dark red t-shirt is playing in the jungle."

"A dog climbs down a jungle green ramp."

"The little child is wearing Android green arm floaties."

"A shocking pink-haired women is watching comedy."

Figure 1: Examples of prompt-induced ambiguity. Top: baseline T2I diffusion (SynGen) outputs misinterpret color terms. Bottom: our disambiguation-guided method resolves these issues, yielding more accurate, semantically aligned results.

### Abstract

Accurate color alignment in text-to-image (T2I) generation is critical for applications such as fashion, product visualization, and interior design, yet current diffusion models struggle with nuanced and compound color terms (e.g., Tiffany blue, baby pink), often producing images that are misaligned with human intent. Existing approaches rely on cross-attention manipulation, reference images,

∗Contribute equally to this work. †Corresponding author.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

© 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 979-8-4007-2035-2/2025/10 https://doi.org/10.1145/3746027.3755369

MM ’25, Dublin, Ireland.

or fine-tuning but fail to systematically resolve ambiguous color descriptions. To precisely render colors under prompt ambiguity, we propose a training-free framework that enhances color fidelity by leveraging a large language model (LLM) to disambiguate colorrelated prompts and guiding color blending operations directly in the text embedding space. Our method first employs a large language model (LLM) to resolve ambiguous color terms in the text prompt, and then refines the text embeddings based on the spatial relationships of the resulting color terms in the CIELab color space. Unlike prior methods, our approach improves color accuracy without requiring additional training or external reference images. Experimental results demonstrate that our framework improves color alignment without compromising image quality, bridging the gap between text semantics and visual generation. All supplementary materials are available at https://Sung-Lin.github.io/TintBench/.

### CCS Concepts

• Information systems → Multimedia content creation.

### Keywords

color disambiguation; diffusion model; training-free

ACM Reference Format:

Sung-Lin Tsai, Bo-Lun Huang, Yu-Ting Shen, Cheng-Yu Yeo, Chiang Tseng, Bo-Kai Ruan, Wen-Sheng Lien, and Hong-Han Shuai. 2025. Color Me Correctly: Bridging Perceptual Color Spaces and Text Embeddings for Improved Diffusion Generation. In Proceedings of the 33rd ACM International Conference on Multimedia (MM ’25), October 27–31, 2025, Dublin, Ireland. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3746027.3755369

### 1 Introduction

Color plays a crucial role in visual perception and aesthetics, significantly impacting domains such as fashion [10], product design [12], in-house decoration [23], and digital art [18]. Accurate color alignment is essential for applications where color fidelity determines the outcome, such as e-commerce visualization, where customers expect generated product previews to match real-world colors precisely. Similarly, in virtual interior design, users may specify complex color schemes for furniture or walls, and any misinterpretation can lead to unrealistic renderings. However, current text-to-image (T2I) diffusion models often struggle with accurately rendering hues, particularly when interpreting compound color names such as lime green, which can lead to imprecise color reproduction (e.g., generating plain green) or object-color lexical confusion (e.g., generating limes).

To better address this issue, several approaches have been proposed to address color alignment in T2I generation. Early works such as Attend-and-Excite [5] and Divide & Bind [13] explore cross-attention manipulation to enforce more faithful text-image correspondence, including color attributes. More recent methods introduce training-free solutions for color control. For example, ColorEdit [26] applies reference-based color adjustments using cross-attention feature alignment, while Color-Style Disentanglement [1] leverages CIELab space feature separation to isolate color properties from luminance and style. Fine-tuning-based methods such as ColorPeel [3] train diffusion models to internalize new color embeddings, enabling precise color reproduction for specific

shades. Meanwhile, LLM-based guidance frameworks like LLMgrounded Diffusion [14] improve compositional reasoning in T2I synthesis but do not explicitly address color fidelity. While these approaches make significant strides, they often rely on reference images [1, 26], training-intensive pipelines [3], or indirect color manipulations [5, 13, 14] that fail to resolve ambiguous or underspecified color descriptions systematically.

Specifically, a major challenge in addressing color ambiguity lies in interpreting complex color expressions and compound descriptors within natural language prompts. Many such expressions—e.g., cerulean blue, dusty rose, or warm taupe—lack precise semantic grounding and may be interpreted differently depending on context or model internalization. While large diffusion models are trained on vast web data, they often struggle to render accurate colors when prompts include fine-grained or less conventional color names. This difficulty stems not from limited data, but from the semantic variability and ambiguity inherent in human color language, which is rarely standardized. As a result, text-to-image models often produce outputs that diverge from user intent when color terms are complex or linguistically nuanced.

As illustrated in Fig. 1, text-to-image diffusion models frequently produce incorrect generations when the input prompt contains ambiguous or compound color terms. These failures typically arise from the model’s misinterpretation of modifiers or the holistic blending of complex expressions. For example, in the second column, the modifier dark leads the model to apply a globally darker tone, diverging from the intended object-specific coloration. In the third column, the term jungle green is misunderstood, prompting the model to insert jungle-like visual elements into the background instead of applying the correct hue. This motivates the need for more explicit disambiguation strategies that can bridge the gap between human color semantics and learned visual representations.

In this paper, we propose a novel approach to enhance color fidelity in T2I diffusion models via Semantic Color Disambiguation with LLMs and Retrieval-Based Embedding Refinement for Color Representation. Our method consists of two stages: First, an LLM refines ambiguous color descriptions by translating them into explicit, unambiguous terms for clearer intent interpretation. Second, we introduce a retrieval-based embedding refinement that interpolates between nearby basic color embeddings to yield a more precise target color representation. By mapping these interpolated terms to the CIELab space, we use surrogate color templates for numerical interpolation. This enables smooth, controllable color blending and embeddings that closely match the intended shade. Our approach integrates semantic understanding with visual accuracy, ensuring high-fidelity color rendering while preserving textual input. Unlike prior methods requiring reference images or tuning cross-attention, ours directly improves color awareness in a training-free manner—yielding significantly better color consistency and realism. Our main contributions are summarized as follows.

• We identify critical challenges in achieving accurate color alignment in T2I synthesis, particularly with novel color terms. A new benchmark is established to systematically assess color fidelity across various T2I models, setting a standard for future evaluations.

- • The method introduces two key modules: Semantic Color Disambiguation with LLMs to disambiguate color terms, enhancing semantic clarity; and Retrieval-Based Embedding Refinement for Color Representation, which applies semantic arithmetic of word embeddings based on precise surrogate color matching in the CIELab space. This training-free approach ensures efficient and scalable improvements in color fidelity for T2I synthesis.
- • Extensiveexperimentsdemonstrate the superior performance of the proposed method. The results showcase significant improvements in color accuracy and consistency, validating the effectiveness of the approach in practical T2I generation tasks.

- 2 Related Work

- 2.1 Alignment with Attention Control

Several techniques have manipulated cross-attention to improve text-to-image diffusion in various contexts. For instance, Promptto-Prompt [11] preserves structure by partially injecting crossattention maps, enabling fine-grained image edits. Attend-andExcite[5]incorporatetoken-specificattention to avoid “catastrophic neglect,” thereby ensuring each prompt element is faithfully rendered. Likewise, Training-Free Layout Control [6] and MasaCtrl [4] modify attention to respect layouts or maintain consistent appearances across edits. SynGen [21] aligns attention maps with grammatical structure to mitigate color-attribute swaps. On the other hand, a recent line of studies focuses on controlling color without retraining: Color-Style Disentanglement [1] transfers reference image color distributions in the CIELab space, disentangling color from luminance. ColorEdit [26] aligns cross-attention features with a given color sample, enforcing color fidelity. While enhancing color rendering, these methods often rely on reference images or extensive attention tuning. In contrast, our approach better aligns the compound color without reference images or manual attention alignment.

- 2.2 LLM-Enhanced Generation and Fine-Tuning

Recent work has also integrated large language models (LLMs) to parse and refine prompts. LLM-grounded Diffusion [14] and RPG [25] use LLMs to handle complex instructions, produce structured scene layouts, and plan multi-entity compositions. Though these frameworks enhance compositional consistency, they do not explicitly address subtle color nuances. Conversely, ColorPeel [3] proposes a fine-tuning strategy that learns a new prompt embedding for each specific color token, yielding accurate color rendering but at the cost of extra training. Our solution remains training-free yet leverages an LLM to disambiguate compound or ambiguous color terms, then directly blends their CIELab space representations within the diffusion process. By fusing LLM-based disambiguation strategies with precise color interpolation, we focus specifically on color accuracy, offering a lightweight alternative to training-heavy methods while expanding T2I models’ capability to handle intricate shades.

[Figure 11]

[Figure 12]

"a red car." "a rose red car."

[Figure 13]

[Figure 14]

"a man in a green shirt in supermarket."

"a middle-aged man in a green shirt walking..."

Figure 2: Failure cases of T2I diffusion models when processing prompts containing ambiguous color terms. Top: the term rose red is misinterpreted, causing the model to generate multiple rose flowers instead of the intended color. Bottom: when additional descriptive details are added to the prompt (e.g.middle-aged, walking), the model incorrectly renders a green shirt as yellow.

### 3 Tint Benchmark

To estimate whether different T2I generation models can accurately interpret color descriptions, especially as users naturally employ expressive, context-dependent language, it is crucial to build a dataset that reflects real-world usage. Existing prompt datasets lack coverage of complex color expressions, focusing mainly on basic color terms like red or blue, and relying on rigid syntactic templates. This limits their utility in evaluating fine-grained color understanding. For instance, while CC-500 prompt dataset [7] explicitly defines attribute binding for color, its prompts follow fixed structures such as "a {adj} {noun} and a {adj} {noun}" (e.g., “a blue backpack and a red bench.”), ailing to capture the richness of natural language. As analyzed in Fig. 2, enhancing prompts with more detailed and human-like descriptions often leads to generation failures in current T2I models. This highlights the need for a dataset that better reflects the expressive diversity of real-world color usage. To address these limitations, we introduce TintBench, a benchmark that combines a curated taxonomy of compound color names, including blended, modified, object-based, signature, and abstract types, with prompts that resemble natural image captions. This design allows us to evaluate model performance in more realistic settings, where color expressions appear within diverse and contextually

#### Table 1: Summary of the TintBench dataset. We report the number of prompts per category and provide representative examples after compound color substitution.

Category Example Total single-color

"A woman with ruby red bags rides her bike over a bridge."

500

"A man with a light blue shirt and Barbie pink shorts is walking off of a soccer field."

500

multi-color

grounded descriptions. In the following sections, we describe how the compound color taxonomy is constructed and how the prompts are generated accordingly.

### 3.1 Compound Color Names

Color perception varies significantly between individuals and is highly influenced by context. As a result, people use tens of thousands of phrases to describe colors. Among these, compound color names are especially important for expressing subtle shades and tones. A compound color typically consists of a basic color term-a foundational color category that can be modified with descriptive adjectives like light, deep, etc. To capture this diversity in color language, which is often overlooked in existing datasets, we curated an extensive collection of compound color names as a foundation for constructing TintBench. These compound color names provide the fine-grained semantics needed to rigorously test a model’s capacity for color understanding in realistic T2I scenarios.

Following prior work [2, 17], we begin with eleven core basic colors: black, blue, brown, gray, green, orange, pink, purple, red, white, and yellow. Building upon these, we searched the corresponding compound color names from web-based repositories such as Wikipedia and specialized color naming databases1, and collected a diverse set of compound colors. Each compound color is linked to a standardized color representation (i.e., RGB code), enabling precise integration into prompt augmentation and evaluation.

The constructed compound colors can be categorized into the following five types:

- • Blended Color: Created by combining two basic color terms to indicate a mixed hue, such as red purple and yellow green.
- • Modified Color: Formed by modifying a basic color term with lightness-related adjectives, such as dark brown and light blue.
- • Object Color: Constructed by prefixing a basic color term with the name of an object that represents the color, such as olive green and salmon pink.
- • Signature Color: A color associated with a specific organization, institution, or geographical region, serving as an identifying hue, such as Duke blue and Caribbean green.
- • Abstract Color: A color name where the prefixing adjective originates from abstract concepts, cultural references, or human-assigned labels, rather than physical attributes, such as Baker-Miller pink and cyber yellow.

By incorporating compound color names from all five categories, TintBench introduces challenging prompts that better reflect the

1For example, https://colornames.org/.

[Figure 15]

Figure 3: Comparison of pairwise distance matrices and Spearman correlations between text embeddings and color spaces. Overall correlation (𝜌) is computed across eleven basic color terms, with group-wise coefficients (𝜌𝑤, 𝜌𝑛, 𝜌𝑐) for warm (orange), neutral (gray), and cool (purple) colors. CIELab achieves the highest alignment overall and within all groups.

complexity of real-world user inputs. The full list of compound color names and codes is available in the Appendix.

### 3.2 TintBench Construction

Equipped with the collected compound colors, we aim to construct the prompt benchmarks in a natural way. Therefore, we selected the Flickr30k dataset [27] as our starting point due to its relatively rich and varied descriptions. We first selected the prompts with color terms, leading to 30.21% of the captions. Please note that these only contain basic color terms, and none sufficiently capture the diversity of compound color names. To introduce the compound colors, we first divided the captions into two groups: single-color and multi-color, based on the number of color terms present. We then applied k-means clustering independently to each group, forming 20 clusters per group. From each cluster, we sampled five captions, resulting in 100 prompts for the single-color group and 100 for the multi-color group, for a total of 200 prompts.

Afterward, we augmented the selected prompts by replacing basic color terms with compound color names, randomly sampled from our five predefined categories. This process generated 500 single-color and 500 multi-color prompts, resulting in a total of 1,000 augmented examples. The augmentation enhances the granularity of color representation in the text prompts and enables more robust benchmarking of T2I models in interpreting descriptive, real-world color language. Tab. 1 summarizes the composition of TintBench and provides representative examples.

### 4 Method

Current methods often fail to resolve the ambiguity of diverse color terms and struggle to ensure precise color rendering, even when the intended semantics are correctly inferred. While fine-tuning with annotated color datasets is a potential solution, it is costly and requires expert input (e.g., via Photoshop). To address this, we propose a training-free framework for color blending in diffusion models via LLM-guided disambiguation, as illustrated in Fig. 4.

Given a text prompt with fine-grained or compound color expressions, our pipeline uses a large language model to rewrite ambiguous terms into clarified descriptions grounded in basic colors (Sec. 4.1). Using CIELab codes of both the target and basic colors,

Input Prompt: "A orange red dog is chewing on a stick."

[Figure 16]

###### ⚙ Embedding Refinement

Basic Colors

Color Code Interpolation

Color Embedding Interpolation

Warm Colors

weight smoothing

[Figure 17]

[Figure 18]

Semantic Color Disambiguation

Color term : orange red, Ambiguous term : orange, Basic term : red

(obtain hue group)

Color code : RGB(255, 21, 0)

###### Text Encoder

Disambiguated Prompt : "A red dog is chewing on a stick."

replace original embedding

···

🎨 Color-Binding Step

|U-Net|
|---|

[Figure 19]

[Figure 20]

[Figure 21]

···

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

···

Figure 4: Our pipeline begins by using a large language model (GPT-4o) to resolve color ambiguity, producing a structured color analysis list. In the Embedding Refinement stage, the target color is interpolated with nearby basic colors within its Color Group to generate a precise embedding, which guides generation via cross-attention binding during denoising.

we compute a color-space offset that captures their perceptual difference. This offset guides embedding refinement via interpolation in the text embedding space (Sec. 4.2), analogous to semantic vector arithmetic (e.g., king − man + woman ≈ queen). The refined embeddings are then passed to the diffusion model, enabling more accurate and semantically faithful color generation.

### 4.1 Semantic Color Disambiguation with LLMs

Text prompts often contain diverse compound color expressions, which can lead to the generation of unintended objects or visually inconsistent content. Recent T2I methods have incorporated POSbased analyses to improve attribute binding in diffusion models [7, 21]. However, as shown in Sec. 3, these approaches remain limited in handling semantically ambiguous or fine-grained color terms, often falling short of user expectations. Motivated by the recent progress of large language models (LLMs) in semantic understanding, we propose leveraging their strong reasoning capabilities to explicitly address the ambiguity in complex color descriptions.

As illustrated in the top-left of Fig. 4, our method employs an LLM (GPT-4o) to perform semantic analysis on text prompts containing intricate color expressions. Given an input prompt 𝑝original, the LLM first analyzes each color term 𝑐. If an ambiguous term 𝑐ambiguous is identified—i.e., one likely to mislead the diffusion model—it is explicitly labeled. The LLM then selects the basic color term 𝑐basic that best represents the intended meaning and rewrites the prompt into a disambiguated version 𝑝disambiguated to resolve semantic confusion. In addition, the LLM provides a reference color code𝛾𝑐 (e.g., in RGB), informed by the scene context and intended interpretation of𝑐. This process mitigates the risk of misinterpretation by aligning textual color semantics with perceptual expectations. The resulting color code directly informs the refinement of color-related embeddings, as detailed in Sec. 4.2. Further details on prompt setup are provided in the Appendix.

### 4.2 Retrieval-Based Embedding Refinement for Color Representation

Text embeddings encode semantic relationships between words in continuous vector spaces, enabling models to capture subtle linguistic similarities. Since the introduction of vector-based representations such as Word2Vec [16] and GloVe [19], numerous studies have shown that semantically related concepts tend to occupy nearby regions in the embedding space. For example, embeddings of color terms like scarlet, crimson, and ruby—all denoting shades of red—typically cluster together more closely than with unrelated terms such as blue or green. This compositional structure of text embeddings offers a promising foundation for modeling color semantics, particularly for interpolating between new color descriptions.

Motivated by these observations, we hypothesize that the spatial arrangement of color terms in learned text embedding spaces reflects their perceptual relationships in human color space. Specifically, we expect that terms representing similar hues (e.g., various shades of red) are embedded near each other, while those corresponding to perceptually distinct hues (e.g., red vs. blue) are more widely separated. To validate this hypothesis, we examine the spatial correlation between eleven basic color terms in the human color perception space and their corresponding representations in the text embedding space.

Specifically, we first compute pairwise distances between color terms across several commonly used color spaces, including RGB, CIELab, HSV, YCbCr, YUV, and CIE1931. To account for the characteristics of non-linear color spaces, we adopt CIEDE2000 color difference formula (Δ𝐸00) [24] in the CIELab space. Unlike the simpler Euclidean formulation in CIE1931, Δ𝐸00 incorporates corrections for lightness, chroma, and hue differences and includes a rotation term to account for perceptual interactions between chroma and hue in the blue region. The formula is defined as:

#### Table 2: User study results on TintBench. We report the average win rate for each evaluation metric. Bold values indicate a win rate exceeding 50%.

Single Multiple

Method

Prompt Ambiguous Color Overall Prompt Ambiguous Color Overall

SD [22] 95.84% 60.44% 79.17% 93.75% 86.11% 66.67% 86.11% 94.44% AE [5] 91.66% 72.20% 94.47% 94.47% 87.50% 54.17% 91.67% 87.50% Conform [15] 83.35% 62.50% 75.00% 79.20% 63.89% 75.00% 80.55% 66.66% DivideBind [13] 75.00% 66.67% 69.44% 80.56% 74.90% 68.06% 68.54% 83.34% Rich [8] 77.78% 69.44% 86.11% 83.33% 75.00% 75.00% 68.75% 83.34% InitNO [9] 77.09% 79.17% 68.75% 79.16% 86.07% 74.97% 86.90% 91.53% SDG [1] 87.70% 79.17% 85.42% 89.58% 79.17% 79.17% 68.75% 83.33% SynGen [21] 81.25% 72.91% 83.34% 91.67% 80.57% 91.57% 69.32% 77.67%

SDXL [20] 69.83% 69.44% 74.99% 68.06% 68.75% 83.15% 67.66% 91.62% SDXL SynGen [21] 81.48% 74.99% 84.26% 87.96% 79.16% 79.16% 69.47% 87.50% SDXL Rich [8] 88.10% 84.50% 65.00% 89.28% 77.78% 83.33% 76.39% 88.90%

Δ𝐸00 = 𝑘 Δ𝐿′

𝐿𝑆𝐿

2

+ 𝑘 Δ𝐶′

𝐶𝑆𝐶

2

+ 𝑘 Δ𝐻′

𝐻𝑆𝐻

2

+ 𝑅𝑇 · 𝑘 Δ𝐶′

𝐶𝑆𝐶 · 𝑘 Δ𝐻′

𝐻𝑆𝐻

1/2

, (1)

where Δ𝐿′, Δ𝐶′, and Δ𝐻′ represent the differences in lightness, chroma, and hue, respectively; 𝑆𝐿, 𝑆𝐶, and 𝑆𝐻 are the weighting functions; 𝑘𝐿, 𝑘𝐶, and 𝑘𝐻 are typically set to 1; and 𝑅𝑇 is a rotation term that accounts for the interaction between chroma and hue. For HSV, we isolate and project the hue component before computing distances. For all other linear color spaces, Euclidean distance is directly applied. We then calculate Spearman’s rank correlation coefficient (𝜌) between each color space’s distance matrix and the corresponding pairwise distances in the text embedding space. As shown in Fig. 3, all color spaces exhibit only weak correlations with the text embedding space, suggesting that while linguistic and perceptual representations may share some structure, the alignment is imperfect and varies across encoding schemes.

Interestingly, we observe that the text embeddings of the color terms naturally form clusters based on hue categories. Based on this observation, we group the 11 basic color terms into three semantic categories: warm colors (including red, orange, pink, and yellow), cool colors (blue, green, and purple), and neutral colors (black, white, gray, and brown). We then compute Spearman correlations for each group individually.

Results show that among all tested color spaces, CIELab consistently exhibits the highest positive correlation with the text embedding space across all three color groups, with an average correlation coefficient of 0.924. The correlation coefficients for the warm, neutral, and cool groups are denoted as 𝜌𝑤, 𝜌𝑛, and 𝜌𝑐, respectively, and are visualized in Fig. 3. We attribute this outcome to findings in color science, which suggest that human vision is particularly sensitive to variations in lightness. As a perceptually uniform space, CIELab more closely reflects how humans perceive color differences in images. This result also aligns with findings from ColorPeel [3], which showed that projecting diffusion latents into CIELab enables more precise control over generated colors.

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

Figure 5: Interpolated text embeddings between “orange car.” and “yellow car.” As the blending ratio changes, the resulting hues transition smoothly from orange to yellow, depending on the dominant color.

Consequently, we adopt CIELab as the default color space for all subsequent operations.

To further demonstrate that the text embeddings can be manipulated to achieve more accurate representations of target colors, we perform color blending within embedding space based on the relationships among basic color terms in perceptual color space. Specifically, we apply proportional blending to the text embeddings of color terms, following the behavior of color interpolation in the CIELab space. For example, the Lab code (66, 43, 68) corresponds to a darker shade of orange, while (92, -21, 94) represents a highly saturated yellow. Averaging these two values yields a color code that lies between them, producing a yellow-orange hue that visually resembles a perceptual blend of both original colors.

Following this principle, we interpolate between the text embeddings of orange and yellow in varying proportions, controlled by a blending factor 𝛼 ∈ [0, 1]. Specifically, we compute the interpolated embedding using the formula: 𝛼 · 𝑒yellow + (1 − 𝛼) · 𝑒orange, where 𝑒yellow and 𝑒orange denote the text embeddings of the tokens orange and yellow, respectively. The resulting vector is then used to replace the color token embedding in the prompt. As shown in Fig. 5, when the blending ratio 𝛼 shifts, the car’s color also shifts toward either orange or yellow, depending on the dominant component.

To enable perceptually grounded color blending, we first classify the predicted color into a predefined hue group based on the color code generated by the LLM during the Semantic Color Disambiguation stage. As demonstrated in our earlier analysis, hue

Ours

SD1.4 Attend & Excite Conform Divide & Bind Initno SynGen RICH SDG

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

“A light blue dog runs in the surf with his mouth open.."

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

"An india green dog is chewing on a stick"

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

"A man in a St.patrick's blue and blood red life vest kayaks down a rapids."

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

"A little girl and boy dressed in sky blue eat cereal out of Duke blue bowls."

#### Figure 6: Qualitative comparison between our method and previous work using different prompts in SD v1.4. The first two rows show results for single-color prompts, while the last two rows demonstrate the multi-color prompt setting.

groups exhibit a stronger correlation between perceptual color distances and text embedding similarities. Restricting subsequent operations to colors within the same hue group ensures higher semantic relevance in the blending process. Within the identified hue group, we then retrieve the top-𝑘 nearest basic color terms by computing perceptual distances in the CIELab space using the Δ𝐸00 metric. This metric closely aligns with human visual perception and provides a more reliable measure than Euclidean distance in RGB or other non-uniform color spaces. Prior work [17] has shown that diffusion models respond more consistently to basic color terms. Motivated by this, we treat the embeddings of the selected basic color terms as directional anchors, using them to guide the offset computation for the target color term in the text embedding space. This strategy ensures that the blended embedding is grounded in semantically and perceptually salient color representations. To compute the blending weights between the selected basic color embeddings, we move beyond simple linear interpolation and instead use a Gaussian-based softmax formulation. This accounts for the non-linear structure of the CIELab space and yields smooth, perceptually meaningful blending. The target color embedding 𝑒target and weight 𝛼𝑖 for each basic color 𝑖 is defined as:

##### ∑︁𝑘

𝑑𝑖2 2𝜎2

, (2)

𝛼𝑖𝑒𝑖, with 𝛼𝑖 = softmax −

𝑒𝑡𝑎𝑟𝑔𝑒𝑡 =

𝑖

where 𝑘 is the number of color terms, 𝑒𝑖 is the embedding for the basic color 𝑖, 𝑑𝑖 is the perceptual distance (Δ𝐸00) between the target color and the𝑖-th basic color, and 𝜎 is a temperature-like scaling factor that controls the sharpness of the weight distribution. Through this retrieval mechanism, we can generate text embeddings that accurately capture the semantics of the blended color representation.

This ensures that the resulting embedding effectively conveys the intended color meaning.

To improve the binding between color terms and their corresponding visual entities, we incorporate a guidance loss during the denoising step. This component is inspired by the positive loss formulation in Cross-Attention-based Guidance used in SynGen [21]. Specifically, the Color-Binding loss Lbinding is applied after embedding refinement and serves as a soft constraint that guides the attention maps toward more semantically consistent color-object alignment:

- 1

- 2DKL(𝐴𝑖color ∥ 𝐴𝑖entity) +

- 1

- 2DKL(𝐴𝑖entity ∥ 𝐴𝑖color), (3)

Lbinding𝑖 =

where 𝐴𝑖color and 𝐴𝑖entity denote the cross-attention maps corresponding to the 𝑖thcolor term and the 𝑖th entity term, respectively. Each attention map is normalized such that its elements sum to 1. The symmetric Kullback-Leibler divergence Lbinding encourages alignment between these two distributions, guiding the model to associate the correct color with the correct visual region. After computing the binding loss Lbinding, we update the output latents 𝑥𝑡 during the denoising step 𝑡 by the Color-Binding Step:

𝑥𝑡 ← 𝑥𝑡 − 𝛼 · ∇𝑥 ∑︁ 𝑐∈𝐶

L𝑐binding, (4)

where 𝛼 is a scalar that controls the binding scale, and 𝐶 represents the set of basic color terms identified in the input prompt.

5 Experiments

- 5.1 Experimental setup

Dataset. We evaluate our method using TintBench, a benchmark we constructed as described in Section 3, which includes 500 singlecolor and 500 multi-color prompts. These prompts cover five types of compound color expressions, as previously outlined.

Implementation Details. We run our model on both Stable Diffusion (SD) 1.4 and SDXL. Image generation is performed using the DDIM scheduler with 50 denoising steps per image. Experiments with SD 1.4 are conducted on NVIDIA RTX 3090 GPUs, while those with SDXL are conducted on NVIDIA V100 GPUs.

Comparison Baselines. To evaluate the effectiveness of our approach, we compare it with several representative baseline methods that primarily rely on attribute binding and cross-attention control. Specifically, we include Attend-and-Excite [5], Conform [15], DivideBind [13], Rich [8], Initon [9], SDG [1], and SynGen [21].

### 5.2 Quantitative Result

Following prior work, we assess the quality and fidelity of generated images through a comprehensive user study, where participants compare outputs from our method against several strong baseline approaches. The results are summarized in Tab. 2. The evaluation focuses on three key aspects: (1) Prompt Alignment, measuring how well the generated image matches the full textual prompt; (2) Color Fidelity, assessing the accuracy and realism of the rendered colors; and (3) Ambiguity Resolution, evaluating the method’s ability to handle context-sensitive or vague color descriptions that often lead to misinterpretation or visual inconsistencies. As shown in the table, our method consistently achieves the highest average win rates across nearly all metrics, with bold numbers indicating win rates exceeding 50%. This demonstrates that users significantly prefer our results over those of other methods. Notably, our performance is particularly strong in the “Ambiguous” category, highlighting the effectiveness of our semantic disambiguation strategy. These results underscore our method’s advantage in understanding finegrained and nuanced color expressions, which are challenging for most existing baselines. In addition, we also demonstrate strong performance when integrating our pipeline with SDXL, further validating the robustness and flexibility of our approach in both standard and high-capacity diffusion settings. Additional analyses on experimental results are provided in the Appendix.

### 5.3 Qualitative Result

We present a qualitative comparison between our method and other approaches in Fig. 6 and Fig. 7. In Fig. 6, we show that the original Stable Diffusion 1.4 (SD1.4) model is easily misled by ambiguous color terms in the prompt. For example, in row 4, the word "DUKE" appears on the shirt of a young boy—an artifact caused by the model interpreting the compound color name "Duke blue" as referring to the university rather than the intended color. Although previous works on attribute binding help mitigate this issue to some extent, they still struggle to generate accurate compound colors as defined in our Tint Benchmark. In contrast, our method addresses both the ambiguity in prompts and successfully generates the correct color output according to the specified compound

[Figure 67]

[Figure 68]

Ours SDXL SynGen RICH

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

"A blond girl walking and talking on a cellphone while clutching a blood red wallet."

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

"A couple in ghost white is feeding each other."

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

"The doctor in the Alice blue scrubs is listening to a woman in a Barbie pink top."

[Figure 81]

[Figure 82]

"A French blue and tickleme pink dog is swimming_through some water."

Figure 7: Qualitative comparison between our method and previous work using different prompts in SDXL. The first two rows show results for single-color prompts, while the last two rows demonstrate the multi-color prompt setting.

color. Similarly, Fig. 7 demonstrates that this issue also occurs in Stable Diffusion XL (SDXL), and again, our method effectively resolves it. These results highlight the importance of semantic color disambiguation when interpreting compound color names. By integrating perceptual grounding and LLM-driven understanding, our approach ensures that visually grounded color meanings are preserved during generations.

### 6 Conclusion and Future Work

In this work, we addressed the long-overlooked challenge of finegrained color understanding in text-to-image generation. We introduced TintBench, a comprehensive benchmark constructed by augmenting real-world prompts with diverse and nuanced compound color expressions. Our dataset enables a more rigorous evaluation of how well generative models capture human-like interpretations of color semantics. To bridge the gap between textual color descriptions and perceptual color representations, we proposed a training-free pipeline that leverages LLMs for semantic color disambiguation. Additionally, we introduced a perceptually weighted interpolation mechanism grounded in the Δ𝐸00 metric, enabling smooth and semantically meaningful blending of basic color embeddings. Our user studies and visualizations demonstrate that the proposed approach offers superior color fidelity and effectively resolves color ambiguities in prompt interpretation. We hope that TintBench and our accompanying framework will facilitate future research in controllable generation and semantic grounding, particularly in applications that demand high-fidelity coloring. Looking ahead, we plan to scale our dataset to include a wider range of input modalities, such as raw color codes, image regions, and free-form text descriptions. We also aim to develop a foundation model for color grounding that can flexibly interpret and generate accurate color representations across modalities. This would pave the way toward more generalizable and interactive multimodal generation systems with fine-grained color control.

### Acknowledgments

This work is partially supported by the National Science and Technology Council, Taiwan, under Grant: NSTC-112-2221-E-A49-059MY3 and NSTC-112-2221-E-A49-094-MY3.

### References

- [1] Aishwarya Agarwal, Srikrishna Karanam, and Balaji Vasan Srinivasan. 2024. Training-free Color-Style Disentanglement for Constrained Text-to-Image Synthesis. arXiv preprint arXiv:2409.02429 (2024).
- [2] Brent Berlin and Paul Kay. 1991. Basic color terms: Their universality and evolution. Univ of California Press.
- [3] Muhammad Atif Butt, Kai Wang, Javier Vazquez-Corral, and Joost van de Weijer.

2024. ColorPeel: Color prompt learning with diffusion models via color and shape disentanglement. In European Conference on Computer Vision.

- [4] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. 2023. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF international conference on computer vision.
- [5] Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. 2023. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM transactions on Graphics (TOG) (2023).
- [6] Minghao Chen, Iro Laina, and Andrea Vedaldi. 2024. Training-free layout control with cross-attention guidance. In Proceedings of the IEEE/CVF winter conference on applications of computer vision.
- [7] Weixi Feng, Xuehai He, Tsu-Jui Fu, Varun Jampani, Arjun Reddy Akula, Pradyumna Narayana, Sugato Basu, Xin Eric Wang, and William Yang Wang. 2023. Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis. In The Eleventh International Conference on Learning Representations.
- [8] Songwei Ge, Taesung Park, Jun-Yan Zhu, and Jia-Bin Huang. 2023. Expressive textto-image generation with rich text. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7545–7556.
- [9] Xiefan Guo, Jinlin Liu, Miaomiao Cui, Jiankai Li, Hongyu Yang, and Di Huang.

2024. Initno: Boosting text-to-image diffusion models via initial noise optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9380–9389.

- [10] Yunpeng Han, Lisai Zhang, Qingcai Chen, Zhijian Chen, Zhonghua Li, Jianxin Yang, and Zhao Cao. 2023. Fashionsap: Symbols and attributes prompt for finegrained fashion vision-language pre-training. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 15028–15038.
- [11] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-or. 2023. Prompt-to-Prompt Image Editing with Cross-Attention Control. In The Eleventh International Conference on Learning Representations.
- [12] Yihan Hou, Xingchen Zeng, Yusong Wang, Manling Yang, Xiaojiao Chen, and Wei Zeng. 2025. GenColor: Generative Color-Concept Association in Visual Design. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems. 1–19.
- [13] Yumeng Li, Margret Keuper, Dan Zhang, and Anna Khoreva. 2023. Divide & Bind Your Attention for Improved Generative Semantic Nursing. In BMVC.

- [14] Long Lian, Boyi Li, Adam Yala, and Trevor Darrell. 2024. LLM-grounded Diffusion: Enhancing Prompt Understanding of Text-to-Image Diffusion Models with Large Language Models. Transactions on Machine Learning Research (2024).
- [15] Tuna Han Salih Meral, Enis Simsar, Federico Tombari, and Pinar Yanardag. 2024. Conform: Contrast is all you need for high-fidelity text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9005–9014.
- [16] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient estimation of word representations in vector space. arXiv preprint arXiv:1301.3781

(2013).

- [17] Nathan Moroney. 2024. Color Terms and Stable Diffusion. In Color and Imaging Conference.
- [18] Muragul Muratbekova and Pakizar Shamoi. 2024. Color-emotion associations in art: Fuzzy approach. IEEE Access 12 (2024), 37937–37956.
- [19] Jeffrey Pennington, Richard Socher, and Christopher D Manning. 2014. Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP).
- [20] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. 2023. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952

(2023).

- [21] Royi Rassin, Eran Hirsch, Daniel Glickman, Shauli Ravfogel, Yoav Goldberg, and Gal Chechik. 2023. Linguistic Binding in Diffusion Models: Enhancing Attribute Correspondence through Attention Map Alignment. In Thirty-seventh Conference on Neural Information Processing Systems.
- [22] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn

Ommer. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–10695.

- [23] Pakizar Shamoi, Muragul Muratbekova, Assylzhan Izbassar, Atsushi Inoue, and Hiroharu Kawanaka. 2023. Towards a universal understanding of color harmony: Fuzzy approach. In Fuzzy Systems and Data Mining IX. IOS Press, 20–28.
- [24] Gaurav Sharma, Wencheng Wu, and Edul N Dalal. 2005. The CIEDE2000 colordifference formula: Implementation notes, supplementary test data, and mathematical observations. Color Research & Application: Endorsed by Inter-Society Color Council, The Colour Group (Great Britain), Canadian Society for Color, Color Science Association of Japan, Dutch Society for the Study of Color, The Swedish Colour Centre Foundation, Colour Society of Australia, Centre Français de la Couleur

(2005).

- [25] Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and Bin Cui.

2024. Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms. In Forty-first International Conference on Machine Learning.

- [26] Xingxi Yin, Zhi Li, Jingfeng Zhang, Chenglin Li, and Yin Zhang. 2024. ColorEdit: Training-free Image-Guided Color Editing with Diffusion Model. arXiv preprint arXiv:2411.10232 (2024).
- [27] Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. 2014. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. Transactions of the association for computational linguistics (2014).

