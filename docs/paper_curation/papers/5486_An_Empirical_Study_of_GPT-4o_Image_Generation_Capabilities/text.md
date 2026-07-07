# arXiv:2504.05979v2[cs.CV]10Apr2025

## An Empirical Study of GPT-4o Image Generation Capabilities

Sixiang Chen1∗, Jinbin Bai2∗, Zhuoran Zhao1∗, Tian Ye1∗, Qingyu Shi3, Donghao Zhou4, Wenhao Chai5, Xin Lin6, Jianzong Wu3, Chao Tang3, Shilin Xu3, Tao Zhang6, Haobo Yuan6, Yikang Zhou6,

†

†

†

Wei Chow2, Linfeng Li2, Xiangtai Li3

, Lei Zhu1,7

, Lu Qi6

1The Hong Kong University of Science and Technology (GZ) 2National University of Singapore 3Peking University 4The Chinese University of Hong Kong 5University of Washington 6Wuhan University 7The Hong Kong University of Science and Technology

###### Abstract

The landscape of image generation has rapidly evolved, from early GAN-based approaches to diffusion models and, most recently, to unified generative architectures that seek to bridge understanding and generation tasks. Recent advances, especially the GPT-4o, have demonstrated the feasibility of high-fidelity multimodal generation, their architectural design remains mysterious and unpublished. This prompts the question of whether image and text generation have already been successfully integrated into a unified framework for those methods. In this work, we conduct an empirical study of GPT-4o’s image generation capabilities, benchmarking it against leading open-source and commercial models. Our evaluation covers four main categories, including text-to-image, image-to-image, image-to-3D, and image-to-X generation, with more than 20 tasks. Our analysis highlights the strengths and limitations of GPT-4o under various settings, and situates it within the broader evolution of generative modeling. Through this investigation, we identify promising directions for future unified generative models, emphasizing the role of architectural design and data scaling. For a high-definition version of the PDF, please refer to the link on GitHub: https://github.com/Ephemeral182/Empirical-Study-of-GPT-4o-Image-Gen.

###### 1 Introduction

Over the past decade, image generation has undergone a remarkable evolution—from the early successes of GANs [35] to the dominance of diffusion models [89, 82, 26], which have significantly advanced image fidelity and diversity [37, 7]. In parallel, Large Language Models (LLMs) have achieved exceptional performance across diverse natural language tasks by scaling autoregressive next-token prediction, demonstrating the power of unified modeling principles. These advances naturally raise a compelling question: can such principles be extended to image generation?

However, fundamental differences between autoregressive and diffusion-based paradigms present non-trivial challenges. Autoregressive models excel in sequential text generation, while diffusion models have become the de facto standard for high-quality image synthesis. Bridging these modalities within a unified framework remains an open challenge. Several works [96, 101, 100, 34, 24, 13] attempt to bridge this gap via multimodal connectors or instruction tuning, with LLMs serving as planning modules that produce intermediate representations for image generation. While effective to some extent, these paradigms often exhibit limited interaction between text and image modalities, and struggle with content consistency—particularly in image-to-image generation and complex instruction-based synthesis.

To address these limitations, recent research explores unified generation models that integrate understanding and generation within a single architecture, following three main technical paradigms. The first line of work represents both language and vision as discrete token sequences [67, 98, 110, 104, 19, 65, 109], leveraging VQGAN [28] or similar compressors to tokenize images for compatibility with autoregressive models. A second direction integrates

*Equal contributions. : schen691@connect.hkust-gz.edu.cn †Corresponding authors.

Preprint. Work in progress.

large language models directly into the diffusion process [128, 126, 112, 72], employing them as denoising backbones for image generation and as unified sequence models for text. While promising, these approaches typically rely on intermediate compression modules such as VAEs or VQVAEs, which may limit visual fidelity or increase architectural complexity. A third and increasingly prominent paradigm investigates discrete diffusion frameworks that natively support both image and text generation within a unified modeling space [71, 73, 93]. Building on this insight, recent works [58, 97] propose fully end-to-end diffusion architectures based on shared Transformer backbones, demonstrating competitive performance and seamless modality integration comparable to similarly sized LLMs.

Despite these promising directions, such systems still lag behind the sophistication and generalization capabilities of proprietary models like Flux [51] and Midjourney [75], which may lack reasoning capabilities.

The recent release of GPT-4o [78] marks a significant milestone in multimodal generative modeling. As a native multimodal architecture, GPT-4o demonstrates strong capabilities in generating high-fidelity, photorealistic images while seamlessly unifying vision and language generation—reportedly in an autoregressive fashion. However, its closed-source nature—particularly the lack of disclosure about its architecture, training regimen, and inference mechanisms—poses substantial challenges for scientific scrutiny. This motivates a careful empirical assessment of its capabilities relative to open-source state-of-the-art models.

Although the visual performance of GPT-4o and Gemini is widely recognized, much of their success likely stems from unprecedented scale in training data, model parameters, and compute resources. Prior studies, including diffusion models and connected-based models, suggest that scaling is a key enabler of generative quality—potentially more so than architectural novelty alone. These trends point to a promising trajectory for unified generative models: with sufficient scale, they may rival or even surpass today’s best proprietary systems.

In this study, we conduct a comprehensive evaluation of GPT-4o’s image generation performance, benchmarking its outputs against leading systems including Gemini 2.0 Flash Experimental [99] and other state-of-the-art models. Building upon our comparative evaluation across text-to-image, image-to-image, image-to-3D, and image-to-X generation tasks, GPT-4o demonstrates several distinctive strengths:

- • Exceptional Text Rendering Capability. GPT-4o demonstrates exceptional capability in rendering textual elements within images, maintaining correct spelling, alignment, and formatting even in documentstyle generation tasks. This level of text fluency is rarely seen in prior models and is crucial for practical applications such as chart generation, document layout synthesis, and instruction-rich visual storytelling.
- • Compositional Generalization and Prompt Following. GPT-4o displays impressive compositional abilities, accurately assembling complex scene elements, styles, or attributes described in prompts. This high prompt following enables it to handle fine-grained multi-attribute conditions in generation tasks with minimal loss of semantic detail.
- • Spatial Reasoning and Multi-View Consistency. In generation tasks involving spatial manipulation, such as 3D view synthesis, camera control, and depth-conditioned rendering, GPT-4o maintains geometric consistency and viewpoint realism. This indicates an inherent capacity for spatial reasoning and structural awareness, even without explicit 3D modeling modules.
- • Comprehensive Image Transformation Capability. GPT-4o shows strong generalization across a wide spectrum of image-to-image tasks, ranging from low-level image restoration to high-level perceptual understanding. Without task-specific tuning, it almost handles diverse transformations such as denoising, deblurring, relighting, segmentation, and depth estimation. This suggests the model has learned robust visual priors and spatial semantics, enabling it to perform correction and abstract structural prediction under a unified framework.

However, limitations remain in inconsistent generation, hallucination, and data bias in underrepresented cultural elements and non-Latin scripts, highlighting current trade-offs in model design and training data coverage.

While we do not analyze the internal architecture or implementation details of GPT-4o in this paper*, we believe it plays an important role toward unified multimodal generation. We also emphasize that model architecture is only one part of this progress—training data, model scale, and optimization strategies are equally important. We hope future work will provide more empirical evidence to better understand such proprietary systems and their position within this evolving research landscape.

*There is currently no definitive evidence regarding the specific implementation details or architectural design of GPT-4o’s image generation capabilities. To ensure the credibility and accuracy of our analysis, we will refrain from making speculative claims in current version.

###### 2 Evaluation

As GPT-4o’s image generation capability has only recently been released and no API is available, we conduct only qualitative comparisons between GPT-4o, Gemini 2.0 Flash [99], and other state-of-the-art models in their respective domains.

To systematically compare these models’ performance across diverse image generation tasks including text-toimage generation, image-to-image generation, text/image to 3D generation, and various image-to-X generation, we conduct a detailed case study focused on analyzing the performance of these models. This qualitative analysis provides insight into gpt 4o’s strengths and limitations in various tasks, as shown in Table 1.

Low Visual Quality : The image synthesis model fails to generate fine-grained object details or produces blurry outputs. Typical cases include distorted human bodies or unrealistic hand shapes.

Inconsistent Generation : The image synthesis model produces inconsistent output or image details with input image.

Lack of Knowledge : The image synthesis model lacks domain-specific knowledge, such as particular artistic styles, and thus generates visually plausible but incorrect results.

Failure to Follow Instructions : The image synthesis model misinterprets the input prompt and produces inconsistent results. For example, it may fail to capture specified numbers, colors, or object arrangements.

###### Table 1: GPT-4o vs. Baselines: Qualitative error analysis across image generation tasks.

Case Figure Meta-task Sub-task GPT-4o Gemini-2.0-flash Domain-SOTA

- Figure 1

Text-to-Image

Complex Text Following

Success Failure to Follow Instructions Failure to Follow Instructions

- Figure 2 Success Failure to Follow Instructions Failure to Follow Instructions

- Figure 3 Success Success Success

- Figure 4 Success Success Success

- Figure 5 Text Rendering

Success Success Success

- Figure 6 Success Low Visual Quality Low Visual Quality

- Figure 7 Success Low Visual Quality Low Visual Quality

- Figure 8 Document Generation

Success Low Visual Quality Low Visual Quality

- Figure 9 Success Low Visual Quality Low Visual Quality

- Figure 10 Success Low Visual Quality Low Visual Quality

- Figure 11 Panorama Lack of Knowledge Success Success

- Figure 12

Image-to-Image

Style Transfer

Success Lack of Knowledge Lack of Knowledge

- Figure 13 Success Lack of Knowledge Lack of Knowledge

- Figure 14

Image Editing

Low Visual Quality Success Failure to Follow Instructions

- Figure 15 Failure to Follow Instructions Failure to Follow Instructions Failure to Follow Instructions

- Figure 16 Success Failure to Follow Instructions Failure to Follow Instructions

- Figure 17 Success Failure to Follow Instructions Failure to Follow Instructions

- Figure 18 Success Failure to Follow Instructions Failure to Follow Instructions

- Figure 19 Success Inconsistent Generation Failure to Follow Instructions

- Figure 20 Single-Concept Customization Success Failure to Follow Instructions Success

- Figure 21 Multi-Concept Customization Inconsistent Generation Inconsistent Generation Success

- Figure 22 Story Image Generation

Success Failure to Follow Instructions Success

- Figure 23 Success Inconsistent Generation Success

- Figure 24 Low-Level Vision-Denoising Low Visual Quality Low Visual Quality Success

- Figure 25 Low-Level Vision-Deraining Success Inconsistent Generation Success

- Figure 26 Low-Level Vision-Dehazing Success Low Visual Quality Success

- Figure 27 Low-Level Vision-Low Light Enhancement Low Visual Quality Low Visual Quality Success

- Figure 28 Low-Level Vision-Debluring Success Low Visual Quality Success

- Figure 29 Low-Level Vision-Super Resolution Success Low Visual Quality Success

- Figure 30 Low-Level Vision-Inpainting Inconsistent Generation Inconsistent Generation Success

- Figure 31 Low-Level Vision-Outpainting Inconsistent Generation Success Success

- Figure 32 Low-Level Vision-Colorization Success Success Success

- Figure 33 Low-Level Vision-Shadow Removal Success Failure to Follow Instructions Success

- Figure 34 Low-Level Vision-Reflection Removal Inconsistent Generation Failure to Follow Instructions Success

- Figure 35 Low-Level Vision-Relighting Success Failure to Follow Instructions Success

- Figure 36 Spatial Control-Canny Inconsistent Generation Failure to Follow Instructions Success

- Figure 37 Spatial Control-Depth Success Failure to Follow Instructions Success

- Figure 38 Spatial Control-Sketch Inconsistent Generation Inconsistent Generation Success

- Figure 39 Spatial Control-Pose Success Inconsistent Generation Success

- Figure 40 Spatial Control-Mask Inconsistent Generation Failure to Follow Instructions Inconsistent Generation

- Figure 41 Camera Control

Inconsistent Generation Failure to Follow Instructions Success

- Figure 42 Failure to Follow Instructions Failure to Follow Instructions Success

- Figure 43 In-Context Visual Prompting Failure to Follow Instructions Failure to Follow Instructions N/A

- Figure 44 Image-to-3D

Image to 3D Modeling Success Failure to Follow Instructions Failure to Follow Instructions

- Figure 45 UV Map to 3D Rendering Success Inconsistent Generation Failure to Follow Instructions

- Figure 46 Novel View Synthesis Success Success Failure to Follow Instructions

- Figure 47

Image-to-X

Image Segmentation

Failure to Follow Instructions Failure to Follow Instructions Success

- Figure 48 Success Failure to Follow Instructions Success

- Figure 49 Success Failure to Follow Instructions Success

- Figure 50 Edge Detection

Success Success Success

- Figure 51 Success Failure to Follow Instructions Success

- Figure 52 Salient Object

Success Failure to Follow Instructions Success

- Figure 53 Success Failure to Follow Instructions Success

- Figure 54 Success Success Success

- Figure 55 Success Success Success

- Figure 56 Depth Estimation Success Failure to Follow Instructions Success

- Figure 57 Normal Estimation Success Failure to Follow Instructions Success

- Figure 58 Layout Detection Inconsistent Generation Inconsistent Generation Success

- Figure 59 Text Detection Failure to Follow Instructions Failure to Follow Instructions Success

- Figure 60 Object Tracking

Inconsistent Generation Inconsistent Generation Success

- Figure 61 Inconsistent Generation Inconsistent Generation Success

- Figure 62 Inconsistent Generation Inconsistent Generation Success

- Figure 63 Inconsistent Generation Inconsistent Generation Success

- 2.1 Text-to-Image Tasks

- 2.1.1 Complex Text Following Capability

Recent progress in text-to-image generation has shown impressive abilities in generating diverse and realistic images based on text prompts. However, composing multiple objects with various attributes and relationships accurately into one scene remains a significant challenge for current text-to-image generative models [92, 85, 8, 81, 6]. In this section, we assess models’ ability for compositional text-to-image generation from four perspectives following [41], which include attribute binding, numeracy, object relationship, and complex compositions. Attribute binding evaluates whether the model correctly assigns attributes, such as color, shape, and texture to the appropriate objects. Numeracy evaluates whether the number of generated objects matches the quantities specified in the prompt. Object relationships refer to both spatial (2D/3D) and non-spatial interactions among objects. Complex compositions evaluate the model’s ability to handle multiple types of constraints simultaneously, especially given long or detailed prompts.

As shown in Figure 1 row 1, GPT-4o outperforms both Gemini 2.0 Flash and Midjourney in numeracy tasks. While GPT-4o accurately represents a single plate, Gemini 2.0 and Midjourney represent two plates instead. In terms of understanding object relationships, GPT-4o is the only model that correctly infers the action “walk towards” from the ragdoll to the labrador. However, GPT-4o struggles with more complex terms like “pentagonal pyramid”, failing to interpret it correctly (see Figure 1 row 4). This suggests that GPT-4o may have difficulty accurately interpreting objects with unusual geometries. When it comes to abstract prompts, GPT-4o also appears to lack imagination (see Figure 2 row 2), whereas Midjourney v6.1 demonstrates better creativity in this case, outperforming both GPT-4o and Gemini 2.0 Flash.

For complex text-to-image generation, we evaluate GPT-4o’s performance with Gemini 2.0 Flash [99] and FLUX.1Pro [51], using the text prompts collected from [124, 106, 115]. As shown in Figure 3, both GPT-4o and FLUX excel at generating realistic and harmonious scenes align with the text prompts. However, we observe that GPT-4o shows limitations in generating culturally related elements. For example, the generated crown for the Chinese general is western-style rather than chinese-style (see Figure 4 row 2). Additionally, in large scene generation, GPT-4o struggles to maintain boundary continuity, whereas FLUX produces a more natural composition (see Figure 4 row 3).

Overall, we conclude that GPT-4o excels at text-to-image generation in terms of attribute binding, generative numeracy, object relationship, and complex compositions. However, it exhibits limitations in generating uncommon objects, culturally specific elements and in maintaining continuity when composing large scenes.

Text-to-Image Generation

[Figure 1]

[Figure 2]

###### 🌟 Evaluation: Visual content precisely following the text instruction.

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Input Text: “A yellow bowl, a blue mug and a pink plate on the table.”

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

###### Input Text: “A ragdoll walks towards a labrador.”

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Input Text:“Three differently colored apples (yellow, green, red from left to right) with a Coca-Cola bottle placed behind the middle apple.”

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

Input Text: “The oval sphere was nestled between the rectangular prism and the pentagonal pyramid.”

GPT 4o Gemini 2.0 Flash Midjourney v6.1

- Figure 1: Task: Compositional text-to-image generation. Evaluate the image-text alignment on attribute binding, numeracy, and object relationship. Setup: Each row shows a text prompt and the generated outputs from GPT-4o, Gemini 2.0 Flash [99], and Midjourney v6.1 [75]. Observation: GPT-4o outperforms Gemini 2.0 Flash and Midjourney v6.1 across all aspects. However, GPT-4o struggles with uncommon objects with a special geometry.

Text-to-Image Generation

[Figure 27]

[Figure 28]

###### ! Evaluation: Visual content precisely following the text instruction.

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

Input Text: “The round, juicy watermelon sat in the cool, refreshing bowl of ice, waiting to be sliced open and devoured.”

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

Input Text: “The bold, expressive strokes of the artist's brush brought the blank canvas to life, forming a vibrant and dynamic masterpiece.”

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

###### Input Text:“The heavy raindrops fell on the smooth glass and the textured roof.”

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Input Text: “The gentle, soothing melody of the piano filled the concert hall, as the pianist's fingers danced over the keys.”

GPT 4o Gemini 2.0 Flash Midjourney v6.1

- Figure 2: Task: Compositional text-to-image generation. Evaluate the image-text alignment on attribute binding and complex compositions. Setup: Each row shows a text prompt and the generated outputs from GPT-4o, Gemini 2.0 Flash [99], and Midjourney v6.1 [75]. Observation: GPT-4o outperforms the other two models in generating objects aligned with the text prompts accurately. But for more abstract and creative tasks, Midjourney v6.1 performs the best.

###### Text-to-Image Generation (with complex text prompt)

[Figure 53]

[Figure 54]

###### ! Evaluation: Visual content precisely following the text instruction.

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

InputText:“An icy landscape. A vast expanse of snow-covered mountain peaks stretches endlessly. Beneath them is a dense forest and a colossal frozen lake. Three people are boating in three boats separately in the lake. Not far from the lake, a volcano threatens eruption, its rumblings felt even from afar. Above, a ferocious red dragon dominates the sky and commands the heavens, fueled by the volcano's relentless energy flow.” (Prompt from GenArtist)

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

InputText:“On the rooftop of a skyscraper in a bustling cyberpunk city, a figure in a trench coat and neon-lit visor stands amidst a garden of bio-luminescent plants, overlooking the maze of flying cars and towering holograms. Robotic birds flit among the foliage, digital billboards flash advertisements in the distance.” (Prompt from IterComp)

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

InputText:“In a magical seascape, a majestic ship sails through crystal blue waters surrounded by vibrant marine life and soaring birds. Towering cliffs frame the scene, while a stunning rainbow arches across the sky, blending with ethereal clouds. This enchanting journey captures the serene beauty of nature's wonders.” (Prompt from IterComp)

GPT 4o Gemini 2.0 Flash FLUX

- Figure 3: Task: Compositional text-to-image generation. Evaluate the image-text alignment on complex compositions. Setup: Each row shows a text prompt and the generated outputs from GPT-4o, Gemini 2.0 Flash [99], and FLUX.1-Pro [51]. Observation: GPT-4o and FLUX can generate more harmonious and natural scene than Gemini 2.0 Flash.

###### Text-to-Image Generation (with complex text prompt)

[Figure 73]

[Figure 74]

#### ! Evaluation: Visual content precisely following the text instruction.

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

InputText:“Under the luminous full moon, a serene Japanese garden with traditional pagodas and a tranquil pond creates a magical night scene. The soft glow from the lantern-lit buildings reflects on the water, blending nature and architecture in harmony. The moonlight bathes the landscape, enhancing the peaceful ambiance.” (Prompt from IterComp)

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

InputText:“A Chinese general wearing a crown, with whiskers and golden Chinese style armor, standing with a majestic dragon head on his chest, symbolizing his strength, wearing black and gold boots. His appearance exudes a sense of authority, wisdom, and an unyielding spirit, embodying the ideal ancient Chinese hero.” (Prompt from RPG)

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

InputText:“A beautiful landscape with a river in the middle, the left of the river is in the evening and in the winter with a big iceberg and a small village while some people are skiing on the river and some people are skating, the right of the river is in the summer with a volcano in the morning and a small village while some people are playing.” (Prompt from RPG)

GPT 4o Gemini 2.0 Flash FLUX

- Figure 4: Task: Compositional text-to-image generation. Evaluate the image-text alignment on complex compositions. Setup: Each row shows a text prompt and the generated outputs from GPT-4o, Gemini 2.0 Flash [99], and FLUX.1-Pro [51]. Observation: GPT-4o struggles to generate culturally related elements and maintain boundary continuity (see rows 2 and 3), similar to Gemini 2.0 Flash and FLUX.

###### 2.1.2 Text Rendering

Text rendering is a task that aims at generating texts (characters, sentences, or even paragraphs) on an image. The text content is usually guided by the input prompt. Previous models [27, 2] show good capability in generating short text (within 10 words, such as signs or short phrases), but their ability to generate long texts remains limited.

As shown in Figure 5, GPT-4o demonstrates comparable abilities to existing state-of-the-art (SOTA) baselines when generating short texts. All the methods except FLUX [51] perform well at rendering short text following the prompt. In this section, we primarily focus on long text rendering to examine whether GPT-4o can surpass these baselines for extended textual content.

We choose POSTA [12], Gemini 2.0 Flash [99], Ideogram 3.0 [2], and Playground-v3 [64] as the baselines because of their established capabilities in rendering longer texts. The results are shown in Figure 6 and Figure 7.

From these examples, we make the following key observations:

- • GPT-4o’s strength in long text generation: Compared with other baselines, GPT-4o demonstrates a superior ability to generate long, coherent text. In example 1 and example 3, GPT-4o produces detailed textual information with fewer than three characters generated incorrectly across more than 100 characters of text.
- • Baseline limitations: When the input prompt becomes extremely long, models such as Gemini 2.0 Flash, Ideogram 3.0, and Playground-v3 often produce significantly more errors or produce vague text patches that are difficult to recognize.
- • POSTA’s performance: As a model specifically designed for poster-style text generation, POSTA performs closely to, or in some instances slightly more precisely than, GPT-4o. We hypothesize this is due to its multi-step pipeline tailored for long text rendering.

Overall, we conclude that GPT-4o excels at long text rendering, offering overwhelming performance compared to most existing commercial models, and delivering results on par with the latest specialized research models.

Short TextRendering

[Figure 93]

[Figure 94]

###### 🌟 Evaluation: Text Rendering Precision.

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

Input Text:“A beautiful painting of flowing colors and styles forming the words ‘The GPT4o/Ideogram/FLUX/SD3 research paper is nowhere!’. the background is speckled with drops and splashes of paint.”

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

Input Text: “Beautiful pixel art of a Wizard with hovering text ‘Achievement unlocked: Diffusion models can spell now’.”

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Input Text:“A monkey holding a sign reading ‘Scaling transformer models is awesome!’.”

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Input Text:“A surreal and humorous scene in a classroom with the words 'GPUs go brrrrr' written in white chalk on a blackboard. In front of the blackboard.”

GPT 4o FLUX SD 3

Ideogram 3.0

- Figure 5: Task: Short text rendering. Generate prompt-aligned, concise textual content (typically within 10 words) on an image. Setup: Each sample is produced based on a guiding text prompt. Comparisons are made with prior SOTA models [27, 2] and FLUX [51]. Observations: GPT-4o achieves performance on par with existing SOTA baselines in rendering short texts, consistently following the prompt with minimal errors. All evaluated methods—except FLUX [51]—deliver high-fidelity results in this setting.

Long Text Rendering

[Figure 111]

[Figure 112]

###### 🌟 Evaluation: Text Rendering Precision.

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

GPT 4o POSTA Gemini 2.0 Flash Ideogram 3.0

###### Input Text:

“Generate a movie poster with a sci-fi space theme, a solitary figure standing on an alien planet, facing a massive outpost. The poster displays the following text: Title: The Last Outpost Subtitle: When the stars fall, the truth rises Information: Produced by Jackson Ward Music by Aria Calloway Screenplay by Elena Sharpe Directed By Sylvia Hartman ”A visually stunning and narratively gripping exploration of the unknown. The Last Outpost masterfully blends elements of science fiction, mystery, and psychological thriller, creating a hauntingly atmospheric journey that will leave audiences on the edge of their seats.“ -- Global Film Review”.

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

GPT 4o POSTA Gemini 2.0 Flash Ideogram 3.0

###### Input Text:

“Create a poster with the theme of a Journey of Solitude. The background should depict a lone figure walking toward an unusable form of transportation. The scene should evoke a sense of being lost, helplessness, and desolation, capturing the emotional weight of losing oneself in a barren, unforgiving landscape. Title: Solitary Journeys Subtitle: Elara Voss Information: WANDERING THROUGH THE UNKNOWN”.

- Figure 6: Task: Long text rendering. Generate extended, coherent, and prompt-consistent textual content on an image. Setup: Evaluations are conducted against advanced baselines including POSTA [12], Gemini 2.0 Flash [99], Ideogram 3.0 [2], and Playground-v3 [64]. Observations: GPT-4o excels in long text rendering by producing coherent, detailed textual information with very few character errors. In contrast, models like Gemini 2.0 Flash, Ideogram 3.0, and Playground-v3 often exhibit increased errors or generate vague text when faced with lengthy prompts, while POSTA’s tailored multi-step pipeline sometimes yields competitive precision. Overall, GPT-4o outperforms most commercial models and rivals specialized research approaches in extended text generation.

Long Text Rendering

[Figure 121]

[Figure 122]

###### 🌟 Evaluation: Text Rendering Precision.

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

GPT 4o POSTA Gemini 2.0 Flash Playground-v3

###### Input Text:

“Please generate an artistic and stylized promotional poster. The style is an artistic painting style. The theme is about nature and city. The poster displays the following information: Title: Fragmented Harmony Subtitle Between the steel and sky, life finds its way. Information: Amid the towering structions and the quiet persistence of nature, a delicate balance emerges. The complex and often contradictory relationship between urban development and the natural world reveals itself in fleeting moments of harmony. Though fragmented, life continues, threading its way through the shadows of progress. Here, conflict and coexistence form an intricate dance--sometimes at odds, sometimes in unexpected unity”.

Figure 7: Task: Long text rendering. The Setup and Observations are the same as Figure 6.

###### 2.1.3 Document Generation

We also explore a novel task: document image generation with GPT-4o, comparing its performance with Gemini 2.0 Flash [99] and Playground-v3 [64]. As shown in Figure 8 - 10, GPT-4o produces document images with cleaner layouts and more consistent content.

Document Image Generation

[Figure 127]

[Figure 128]

###### 🌟 Evaluation: Text Rendering Precision.

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

Playground-v3

GPT 4o Gemini 2.0 Flash

###### InputText:

“Generate A realistic screenshot of the first page of the Paper from the following information: Title: Attention Is All You Need Author List: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin Abstract: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new singlemodel state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.”

Figure 8: Task: Document image generation. Setup: Each row shows a text prompt and the generated outputs from GPT-4o, Gemini 2.0 Flash [99], and Playground-v3 [64]. Observation: GPT-4o can generate more consistent and accurate font and format than the other two models.

###### Document Image Generation

[Figure 135]

[Figure 136]

###### ! Evaluation: Text Rendering Precision.

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

Playground-v3

GPT 4o Gemini 2.0 Flash

###### InputText:

Generate A realistic screenshot of the first page of the Paper from the following information: Title: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding Author List: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova Abstract: We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial taskspecific architecture modifications. BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).”

Figure 9: Task: Document image generation. The Setup and Observations are the same as Fig. 8.

###### Document Image Generation

[Figure 143]

[Figure 144]

###### ! Evaluation: Text Rendering Precision.

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

Playground-v3

GPT 4o Gemini 2.0 Flash

###### InputText:

“Generate A realistic screenshot of the first page of the Paper from the following information: Title: You Only Look Once: Unified, Real-Time Object Detection Author List: Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi Abstract: We present YOLO, a new approach to object detection. Prior work on object detection repurposes classifiers to perform detection. Instead, we frame object detection as a regression problem to spatially separated bounding boxes and associated class probabilities. A single neural network predicts bounding boxes and class probabilities directly from full images in one evaluation. Since the whole detection pipeline is a single network, it can be optimized end-to-end directly on detection performance. Our unified architecture is extremely fast. Our base YOLO model processes images in real-time at 45 frames per second. A smaller version of the network, Fast YOLO, processes an astounding 155 frames per second while still achieving double the mAP of other real-time detectors. Compared to state-of-the-art detection systems, YOLO makes more localization errors but is far less likely to predict false detections where nothing exists. Finally, YOLO learns very general representations of objects. It outperforms all other detection methods, including DPM and R-CNN, by a wide margin when generalizing from natural images to artwork on both the Picasso Dataset and the People-Art Dataset.”

Figure 10: Task: Document image generation. The Setup and Observations are the same as Fig. 8.

###### 2.1.4 Panorama Image Generation

Panorama image generation aims at creating a 360-degree view of a static scene, enabling immersive and comprehensive visual experiences. In our experiments, we select Pano-SD [119] and Gemini 2.0 Flash [99] as the baselines, with representative results illustrated in Figure 11. The comparisons reveal that while the baseline models can generate coherent panorama-like images with seamlessly connectable left and right sides, GPT-4o struggles to produce a true panorama. In most cases, GPT-4o generates images that approximate a panoramic view but still fall short in ensuring the necessary continuity across the image boundaries. We attribute this limitation to the insufficient representation of panorama images in its training data, as well as a predisposition towards generating images with a higher vertical aspect ratio rather than a wider one. Consequently, in the realm of panorama image generation, GPT-4o is inferior to the existing baseline models.

PanoramaImage Generation

[Figure 151]

[Figure 152]

###### 🌟 Evaluation: Is panorama image?

[Figure 153]

[Figure 154]

[Figure 155]

Input Text:“Please generate a panorama image: A living room with hardwook floors, a fireplace, and large windows.”

[Figure 156]

[Figure 157]

[Figure 158]

Input Text:“Please generate a panorama image: A cozy study with built-in bookshelves and a leather.”

[Figure 159]

[Figure 160]

[Figure 161]

Input Text:“Please generate a panorama image: A bedroom with a ceiling fan, gray walls, hardwood floors, a bed, and a TV on the wall.”

GPT 4o Gemini 2.0 Flash Pano-SD

- Figure 11: Task: Panorama image generation, aiming to create immersive 360-degree views of static scenes. Setup: We compare GPT-4o with established baselines such as Pano-SD [119] and Gemini 2.0 Flash [99] to evaluate the generation of coherent panoramic images. Observations: While the baseline models reliably produce panoramas with seamlessly connected left and right sides, GPT-4o tends to only approximate a panoramic view and struggles to maintain continuity across image boundaries. This shortfall is likely due to limited panorama image representation in its training data and a tendency to generate images with a higher vertical aspect ratio rather than a wider one, rendering it inferior to the baselines in this task.

- 2.2 Image-to-Image Tasks

- 2.2.1 Style Transfer

Style transfer is a classic yet evolving task in computer vision, aiming to render an image in a specific artistic style while preserving the original content. It bridges the domains of vision and art, enabling applications such as digital artwork creation, film post-production, and virtual reality environment design. Early approach [33] used convolutional neural networks to separate and recombine content and style representations from images. This seminal work enabled the artistic stylization of photographs by optimizing pixel values to match a desired style. To improve efficiency, Johnson et al. [47] proposed feed-forward networks for real-time style transfer using perceptual losses. Later methods such as AdaIN [43] and WCT [57] enabled arbitrary style transfer without retraining for each new style. Transformer-based models like StyTr2 [23] have been introduced to enhance style transfer quality and better preserve structural details. More recently, with the rapid development of image synthesis techniques, especially diffusion models, style transfer has seen further advancements in both quality and controllability. However, transferring specific artistic styles still typically requires a non-trivial amount of training data.

To comprehensively evaluate the style transfer capability of GPT-4o, we conduct comparisons against several recent competitive models, including Gemini 2.0 Flash [99] and Midjourney v6.1 [75]. Specifically, Figure 12 illustrates style transfer results for natural scenes, while Figure 13 focuses on human facial images. Across a diverse range of styles, such as Monet, Van Gogh, Pixar, Cyberpunk, Snoopy, Disney, Ghibli, and Cubism, GPT-4o demonstrates consistently superior performance in both stylistic fidelity and content preservation.

Notably, in the case of Ghibli style transfer, GPT-4o exhibits remarkable fidelity to the original artistic aesthetics, closely resembling the target style with vivid color palettes and soft contours. In contrast, both Gemini and Midjourney often produce inconsistent visual styles and textures. Furthermore, GPT-4o excels at preserving finegrained content details, such as facial structure, earrings, clothing, and hairstyles, which are often misrepresented or lost in the outputs of other models. These results suggest that GPT-4o not only captures high-level style semantics but also maintains strong spatial consistency and semantic alignment.

Prompted Stylization

[Figure 162]

[Figure 163]

###### 🌟 Evaluation: Consistency/style.

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

Input Text:“Generate the Monet style of this picture.”

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

Input Text:“Generate the Van Gogh style of this picture.”

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

Input Text:“Generate the Pixar style of this picture.”

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

Input Text:“Generate the Cyberpunk style of this picture.”

Input Image GPT 4o Gemini 2.0 Flash Midjourney v6.1

- Figure 12: Task: Style transfer, aiming to render an image in a specific artistic style while preserving the original content. Setup: We compare GPT-4o with Gemini 2.0 Flash [99] and Midjourney v6.1 [75] on natural scene style transfer across multiple artistic domains. Observations: GPT-4o exhibits significantly better content preservation compared to Midjourney v6.1, maintaining fine-grained content details and structural consistency. In terms of style, it faithfully adheres to the textual description, effectively rendering vivid color palettes and soft contours that characterize the target style. This alignment notably surpasses both Gemini 2.0 Flash and Midjourney v6.1, highlighting GPT-4o’s strong capabilities in preserving content and faithfully rendering diverse styles.

Prompted Stylization

[Figure 184]

[Figure 185]

###### 🌟Evaluation: Consistency/style.

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

Input Text: “Generate the Simpsons style of this picture.”

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

Input Text: “Generate the Snoopy style of this picture.”

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

Input Text: “Generate the Disney style of this picture.”

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

Input Text: “Generate the Ghibli style of this picture.”

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

Input Text: “Generate the Cubism style of this picture.”

Input Image GPT 4o Gemini 2.0 Flash Midjourney v6.1

- Figure 13: Task: Style transfer, aiming to render an image in a specific artistic style while preserving the original content. Setup: We compare GPT-4o with Gemini 2.0 Flash [99] and Midjourney v6.1 [75] on human face style transfer across multiple artistic domains. Observations: GPT-4o exhibits significantly better content preservation compared to Gemini 2.0 Flash and Midjourney v6.1, maintaining fine-grained content details and structural consistency. In terms of style, it faithfully adheres to the textual description, effectively rendering vivid color palettes and soft contours that characterize the target style. This alignment notably surpasses both Gemini 2.0 Flash and Midjourney v6.1 far away, highlighting GPT-4o’s strong capabilities in preserving content and faithfully rendering diverse styles.

###### 2.2.2 Image Editing

Image editing involves modifying the visual elements, composition, or data of an image to achieve a desired outcome. This process can range from minor refinements to significant alterations, while maintaining the integrity of the original image. Over time, image editing techniques have evolved from manual, labor-intensive methods to sophisticated AI-driven approaches. Prior works [10, 30, 9, 120, 5, 29, 4, 40] have demonstrated the ability to perform various editing tasks based on textual instructions, such as adding, removing, or replacing objects; altering backgrounds, colors, or styles; and adjusting the number, size, or positions of objects. However, these models still exhibit limitations in certain scenarios, particularly in preserving non-edited regions, maintaining consistent image characteristics, and ensuring seamless blending between edited and non-edited areas.

We compare GPT-4o with MGIE [30], LEDITS++ [9], MagicBrush [120], and Gemini 2.0 Flash [99], which are representative of current SOTA methods. These experiments evaluate GPT-4o’s subject preservation and instruction-following capabilities to determine its effectiveness compared with existing methods. Comparative results are shown in Figure 14 through Figure 19. We find that GPT-4o achieves performance comparable to, and in many cases surpassing, SOTA baselines in image editing tasks. From these examples, GPT-4o exhibits the fewest failure cases, demonstrating a strong generalization ability across a wide variety of editing tasks. It consistently outperforms baseline models across multiple editing scenarios. We highlight several key observations:

###### • Strengths of GPT-4o in image editing:

- – Fine-grained editing: GPT-4o shows a superior ability to handle fine-grained editing tasks. For instance, in example 2 of Figure 14 and example 1 of Figure 15, GPT-4o successfully modified small, detailed objects such as a toothpick and pink ballerina slippers, outperforming prior methods.
- – Substantial image transformations: GPT-4o excels at large-scale edits, such as background changes or object transformations, while maintaining visual coherence and realism. These complex edits require robust contextual and semantic understanding. Example 1 in Figure 16 illustrates GPT-4o’s effective handling of a major background alteration task.
- – Subject preservation: GPT-4o demonstrates strong subject-preserving capabilities, avoiding common artifacts such as facial distortions or component loss. In example 2 of Figure 14, GPT-4o retains the content of a drink that Gemini 2.0 Flash erroneously altered. Similarly, in example 5 of Figure 19, GPT-4o best preserves fuselage patterns and textual markings on an airplane.
- – Instruction and original image adherence: GPT-4o shows a notable ability to follow instructions and maintain the structure of the original image, particularly in style editing and tasks involving object quantity, size, or position. This likely stems from its advanced understanding of both the image content and the editing instructions. For example, Figure 18 demonstrates GPT-4o’s capability in style translation. Example 2 in Figure 17 shows its understanding of the term “orange” in both textual and visual contexts. A similar ability is illustrated in example 4 of Figure 19.

###### • Limitations of GPT-4o in image editing:

- – GPT-4o underperforms in scenarios where strict preservation of the original image’s lighting, shading, and color tones is required. In such cases, the edited images may exhibit noticeable shifts in visual consistency. This is evident in examples 1 and 5 of Figure 14 and example 4 of Figure 15.
- – In some cases, GPT-4o may fail to retain image details outside the intended edit region. For instance, example 4 in Figure 14 shows a degradation in image quality in non-targeted areas.

In summary, GPT-4o demonstrates substantial advancements in image editing, showing exceptional capabilities in detailed and large-scale edits, subject preservation, and adherence to instructions. While there are limitations in strictly maintaining original image characteristics such as lighting and tonal consistency, GPT-4o significantly reduces failure cases and outperforms existing baselines across a wide range of editing tasks, pushing the boundaries of current SOTA performance.

ImageEditing

[Figure 226]

[Figure 227]

###### ! Evaluation: Instruction-following / faithful.

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

GPT 4o

Input Text:“Add a notebook to the desk.”

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

Input Text:“Put a toothpick in the top of the left sandwich.”

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

Input Text:“Change the goats into moose.”

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

Input Text:“Replace potatoes with baked beans.”

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

Input Text:“Change the fire hydrant to a parking meter.”

Input Image GPT 4o Gemini 2.0 Flash MGIE

- Figure 14: Task: Image editing for modifying visual elements and composition. Setup: GPT-4o vs. Gemini 2.0 Flash [99]/MGIE [30]. Observations: GPT-4o achieves higher success rates than MGIE (examples 2/5) but occasionally alters unintended elements (bread in example 4) or lighting/shading structures (example 5). This likely stems from stronger generalization capacity and creative adaptation focus in training, though reduced fidelity suggests insufficient constraints on structural details during fine-tuning.

ImageEditing

[Figure 268]

[Figure 269]

###### ! Evaluation: Instruction-following / faithful.

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

Input Text:“Turn everyone shoes into pink ballerina slippers.”

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

InputText:“Remove the fence from in front of the horses.”

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

Input Text:“Remove the baby elephant in the picture.”

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

Input Text:“Change the yellow hat into a cowboy hat.”

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

Input Text:“Remove the people from the background”.

Input Image GPT 4o Gemini 2.0 Flash MGIE

- Figure 15: Task: Image editing for modifying visual elements and composition. Setup: GPT-4o vs. Gemini 2.0 Flash [99]/MGIE [30]. Observations: From examples 1-3, GPT-4o shows higher success in fine detail edits and large-scale edits with occlusions. This likely stems from GPT-4o’s stronger contextual understanding and ability to infer missing or obscured elements, enabling more precise localized edits and coherent large-scale modifications even with partial visibility. However, it sometimes erases non-target elements (e.g., the house in example 5) and significantly alters global lighting (example 4).

ImageEditing

[Figure 310]

[Figure 311]

###### ! Evaluation: Instruction-following / faithful.

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

Input Text:“Change the background to the set of a nickelodeon game show.”

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

Input Text:“Have the dog prick up its ears.”

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

Input Text:“Have the elephant's tail raised.”

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

Input Text:“Change the background to Vatican City.”

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

Input Text:“Change the background to Mount Rainier.”

Input Image GPT 4o Gemini 2.0 Flash MGIE

- Figure 16: Task: Image editing for modifying visual elements and composition. Setup: GPT-4o vs. Gemini 2.0 Flash [99]/MGIE [30]. Observations: From Example 1, GPT-4o demonstrates superior performance in style editing, effectively interpreting style instructions and preserving global image structure—a capability lacking in baseline models (MGIE, Gemini 2.0 Flash, and MagicBrush, as will be shown later). This likely stems from its stronger cross-modal comprehension and structural awareness during training.

Image Editing

[Figure 352]

[Figure 353]

###### ! Evaluation: Instruction-following / faithful.

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

Input Text:“Add a white hat to the woman's head.”

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

Input Text:“Delete the oranges from the shelf in the image.”

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

Input Text:“Get rid of the water the elephants are walking through.”

Input Image GPT-4o Gemini 2.0 Flash LEDITS++

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

Input Text:“Show the seal raising its head.”

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

Input Text: “Change the sky to stars at night.”

Input Image GPT-4o Gemini 2.0 Flash MagicBrush

- Figure 17: Task: Image editing for modifying visual elements and composition. Setup: GPT-4o vs. Gemini 2.0 Flash [99]/LEDITS++ [9]/MagicBrush [120]. Observations: From Examples 2 and 3, GPT-4o demonstrates stronger comprehension of instructions involving ‘the oranges on the shelf’ and ‘the water the elephants are walking through’, translating this understanding into more accurate edits. This suggests better grounding of textual prompts in visual context during generation.

ImageEditing

[Figure 394]

[Figure 395]

###### ! Evaluation: Instruction-following / faithful.

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

Input Text:“Change the image to a 1950s Flintstones cartoon art style.”

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

Input Text:“Change this into a cubist painting.”

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

Input Text:“Make the image appear as if it's a woodblock print by Hokusai.”

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

Input Text:“Change the background to Fushimi Inari Taisha.”

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

Input Text:“Make the image appear like a Rembrandt painting.”

Input Image GPT 4o Gemini 2.0 Flash MagicBrush

- Figure 18: Task: Image editing for modifying visual elements and composition. Setup: GPT-4o vs. Gemini 2.0 Flash [99]/MagicBrush [120]. Observations: This set of examples further demonstrates GPT-4o’s robust capabilities in style editing and background modification, consistent with the findings previously presented in Figure 16.

Image Editing

[Figure 436]

[Figure 437]

###### ! Evaluation: Instruction-following / faithful.

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

Input Text:“Make the image look like a cartoon.”

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

Input Text:“Change the bike frame to be shiny metal instead of red.”

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

Input Text:“Change the table color from blue to black.”

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

Input Text:“Change the woman's hair to be all blue.”

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

Input Text:“Make the color of the airplane be yellow instead.”

Input Image GPT-4o Gemini 2.0 Flash MagicBrush

- Figure 19: Task: Image editing for modifying visual elements and composition. Setup: GPT-4o vs. Gemini 2.0 Flash [99]/MagicBrush [120]. Observations: Example 4 highlights GPT-4o’s superior image understanding—accurately distinguishing between hair and a scarf (where MagicBrush fails) to execute the edit. In Example

- 5, its precise retention of the plane’s logo and text further demonstrates robust object-preservation capabilities.

###### 2.2.3 Customization

Customization, also known as subject-driven generation or personalization, aims to enable visual generative models to generate visual concepts from given reference images. Initial methods [31, 91] have achieved this by optimizing text embeddings or model weights. Subsequent approaches [50, 36, 46, 125, 94, 129] expanded on these approaches to handle multiple visual concepts. Customization plays a crucial role in making visual generative models more flexible and applicable across diverse domains. By empowering models to adapt to user-provided inputs, it ensures outputs are tailored to specific visual concepts. This is particularly significant in industries such as artistic creation and advertising, where individualization and creativity are paramount.

To evaluate the performance of GPT-4o in this challenging task, we collect reference images from previous relevant works [130, 103], and conduct qualitative comparisons as shown in Figure 20 and Figure 21. For single-concept customization, we compare GPT-4o with Gemini 2.0 Flash and DisEnvisioner [130]. The results demonstrate that GPT-4o not only faithfully reproduces the visual concept from the reference image but also accurately adheres to the given textual description. In this task, GPT-4o significantly outperforms Gemini 2.0 Flash and achieves performance on par with the SOTA customization method. However, the images generated by GPT-4o still exhibit some “copy-paste” artifacts, leaving room for further improvement in the future. For multi-concept customization, we compare GPT-4o with Gemini 2.0 Flash and MS-Diffusion [103]. In this task, GPT-4o can still achieve competitive results for customizing multiple visual concepts in different contexts. Unfortunately, it struggles with certain unique combinations (e.g., making a dog wear a human dress), which could be attributed to the lack of relevant customization training data.

Overall, GPT-4o demonstrates impressive performance in both single-concept and multi-concept customization tasks, showcasing strong concept fidelity and great text alignment. Despite some limitations, GPT-4o achieves remarkable results on par with SOTA customization methods and outperforms Gemini 2.0 Flash.

Customization (Single concept)

[Figure 478]

[Figure 479]

###### ! Evaluation: Corresponding visual concepts of given reference images.

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

InputText:“A dog on top of a purple rug in a forest, with reference to the attached image.”

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

InputText:“A cat wearing a Santa hat, with reference to the attached image.”

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

InputText:“A pair of glasses with a tree and autumn leaves in the background, with reference to the attached image.”

Input Image GPT 4o Gemini 2.0 Flash DisEnvisioner

- Figure 20: Task: Single-concept customization. The goal is to generate images that faithfully reproduce a single visual concept from reference images while aligning with a given textual description. Setup: Reference images are collected from prior works [130], and results are compared across GPT-4o, Gemini 2.0 Flash [99], and DisEnvisioner [130]. Each row includes the input reference image, text prompt, and the corresponding outputs. Observations: GPT-4o demonstrates strong performance in faithfully reproducing the single visual concept with high fidelity while adhering closely to the given textual description. It consistently outperforms Gemini 2.0 Flash and achieves results comparable to the SOTA method DisEnvisioner. However, some generated images still exhibit minor “copy-paste” artifacts, indicating room for further improvement.

Customization (Multiple concepts)

[Figure 504]

[Figure 505]

##### ! Evaluation: Corresponding visual concepts of given reference images.

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

InputText:“A dog wearing a dress in the snow, with reference to the attached images.”

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

InputText:“A flower with a barn in the background, with reference to the attached images.”

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

InputText:“A backpack and a stuffed animal in the jungle, with reference to the attached images.”

Input Image GPT 4o MS-Diffusion

Input Image

Gemini 2.0 Flash

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

InputText:“A lantern, a clock, and a backpack on a cobblestone street, with reference to the attached images.”

Input Image Input Image Input Image GPT 4o MS-Diffusion

Gemini 2.0 Flash

- Figure 21: Task: Multi-concept customization. The goal is to generate images that effectively combine multiple visual concepts from reference images while aligning with a given textual description. Setup: Reference images are collected from prior works [103], and results are compared across GPT-4o, Gemini 2.0 Flash [99], and MS-Diffusion [103]. Each row includes the input reference images, text prompt, and the corresponding outputs. Observations: GPT-4o achieves competitive results in combining multiple visual concepts, showing strong fidelity to individual concepts and alignment with text prompts. However, its performance declines with unique or complex combinations. Despite this, GPT-4o outperforms Gemini 2.0 Flash and achieves results on par with SOTA methods.

###### 2.2.4 Story Image Generation

Story image generation is a task to generate coherent stories based on input text narratives. The conditions may also include the first story frame or character images. We choose Gemini 2.0 Flash [99], StoryDiffusion [38], SEED-Story [111], and DiffSensei [108] as baselines, due to their proven ability to generate coherent and expressive story images and their public availability. The results are shown in Figure 22 and Figure 23.

In the first example, GPT-4o and StoryDiffusion successfully generate a three-panel short story about a fisherman, whereas Gemini 2.0 Flash fails by producing a single panel that appears to combine the three story narratives. In the second example, the story narrative is longer, spanning 11 panels. To evaluate this scenario with GPT-4o, we instruct the model to generate story images sequentially—using the input image and all previously generated images along with the corresponding text prompts. As shown in the figure, GPT-4o is capable of generating a long story with consistency. In the final example, we examine a Japanese black-and-white manga style with multiple input character images. GPT-4o is able to generate coherent stories, though it exhibits minor errors in character consistency (notably with the depiction of the woman) and misalignment with the input narrative (the narrative requires 7 panels, but only 6 are generated). The baseline Gemini 2.0 Flash performs worse, failing to preserve character status and the correct number of panels, as it also produces only 6 panels. Conversely, the DiffSensei model demonstrates superior performance, likely due to its specialized design and training for Japanese black-and-white manga generation.

In conclusion, while GPT-4o achieves comparable performance to current baselines in story image generation, it shows limitations in specific scenarios—such as Japanese black-and-white manga and precise character status preservation—when compared to methods specifically tailored for those tasks.

Story ImageGeneration

[Figure 548]

[Figure 549]

###### ! Evaluation: Subject Consistency.

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

GPT 4o StoryDiffusion Gemini 2.0 Flash

###### Input Text:

“Draw a story about: An old fisherman in a cable-knit sweater and boots

- 1. Laying out a picnic solo
- 2. Rowing a boat at dawn
- 3. Stargazing with a telescope”.

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

Input Image GPT 4o SEED-Story

###### Input Text:

“Draw a story about George, a monkey:

- 1. He looked around with a curious expression, wondering what adventures awaited him.
- 2. Suddenly, George heard a noise. …
- 3. To his surprise, the noise was George‘s friend, a small brown dog …
- 4. George and the dog then played a game of hide and seek. George hid behind a couch …
- 5. The next day, George and the dog decided to explore the city …
- 6. George stopped on the city sidewalk, looking up at the sky …
- 7. George then noticed a building with a reflective glass …
- 8. George and the dog stood in front of the building, looking up at the lit windows …
- 9. They were in a room with a door, waiting for their friend to join them
- 10. Suddenly, the door opened, and a man in a yellow suit walked in …
- 11. He seemed deep in thought, unaware of George and the dog watching him from below …”.

- Figure 22: Task: Story image generation. The goal is to generate coherent story sequences based on narrative text, optionally conditioned on initial story frames or character images. Setup: Each example combines an input narrative (and, when available, reference character images) with a series of generated story panels. We compare outputs from GPT-4o against Gemini 2.0 Flash [99], StoryDiffusion [38], and SEED-Story [111]. Observations: GPT-4o exhibits strong narrative coherence and panel continuity, matching or surpassing general baselines.

Story ImageGeneration

[Figure 568]

[Figure 569]

###### ! Evaluation: Subject Consistency.

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

Input Images GPT 4o DiffSensei

Gemini 2.0 Flash

###### Input Text:

“Please generate a black-and-white manga using the given characters (a young man, a child, and a woman). Each panel may appear 0-3 characters.

- 1. A man is lying on the floor surrounded by books and papers, with a radio nearby.
- 2. A woman with curly hair is smiling. She’s wearing a patterned shirt and apron. She’s holding a baby.
- 3. A man with a surprised expression, his mouth open as if he’s about to shout or scream.
- 4. A young man with a surprised expression, is holding a baby on his back.
- 5. A man is holding a baby. The man’s hair is disheveled.
- 6. A man with a surprised expression. His eyes wide and eyebrows raised.
- 7. A man carrying a child on his back walk up a staircase. The man is wearing a stripped shirt”.

- Figure 23: Task: Story image generation. The goal is to generate coherent story sequences based on narrative text, optionally conditioned on initial story frames or character images. Setup: Each example combines an input narrative (and, when available, reference character images) with a series of generated story panels. We compare outputs from GPT-4o against baselines including Gemini 2.0 Flash [99] and DiffSensei [108]. Observations: GPT-4o shows minor shortcomings in precise character consistency and panel count in specialized contexts, such as Japanese black-and-white manga, where dedicated models like DiffSensei deliver superior performance.

###### 2.2.5 Low-level Vision

Low-level vision tasks aim to enhance the basic quality or detail of visual content by improving various aspects of an image. Initial methods often focused on optimizing single tasks, such as super-resolution [88, 95], denoising [61, 63, 55], restoration [60, 20, 62, 84, 15, 16, 17], color adjustment [59], and more [22, 66, 116, 1, 122]. As the technology progressed, subsequent approaches expanded these techniques to handle multiple low-level tasks simultaneously, which is called universal image restoration. Low-level tasks play a critical role in image generation and editing, allowing visual generative models to provide higher-quality outputs in real-world applications. By enabling models to adapt to diverse inputs, they ensure that the generated images perform well across different visual tasks. This is especially important in areas such as image restoration and video enhancement, where high-precision visual content optimization is crucial, such as in film post-production and autonomous driving.

We evaluate the performance of GPT-4o in this challenging task. Firstly, for some image restoration tasks, such as super resolution, denoising, deraining, low-light enhancement, deblurring and dehazing. We collect reference images from previous relevant works Gemini 2.0 Flash and a universal image restoration model, InstrucIR [20], as shown in Figures 24, 25, 26, 27, 28, 29, 33, 34. In most scenarios, GPT-4o guarantees high-quality output images, outperforming Gemini 2.0 Flash. However, there are still some degradation issues that are difficult to remove, as seen in the second image of the image denoising task. On the other hand, for low-level image restoration tasks, maintaining pixel consistency between the output and input images is crucial. GPT-4o does not perform well in this regard, as the content of many images changes. In contrast, InstructIR, designed specifically for image restoration, performs better, effectively removing degradation while maintaining pixel consistency throughout.

For image inpainting and outpainting in Figure 30, 31. We compared Gemini 2.0 Flash with the latest inpainting and outpainting methods [66, 116, 22, 1]. Only the missing information needs to be completed, but GPT-4o still changes the undesired content of the image. Although the output image quality is higher, this is not ideal for evaluating the task itself. For human face inpainting, compared to the other two methods, the overall artistic style is more natural. For the colorization, we choose the latest colorization model CtrlColor [59]. The overall style is somewhat dark in Figure 32. Compared to Gemini 2.0 Flash, GPT-4o’s colors are more natural and consistent with the style. However, there are some inaccuracies in color control. For example, in the second image, the cat’s color is not white as specified in the text. Additionally, GPT-4o still exhibits issues with changes in image content, such as the shape of the human’s face in the fourth image.

For the image re-lighting task in Figure 35, GPT-4o performs well in applying realistic lighting and shadows, with natural color tones that match the scene. However, it occasionally struggles with maintaining light consistency, particularly in complex lighting scenarios, such as neon or vibrant lights. Compared to Gemini 2.0 Flash, GPT-4o produces more natural and consistent results, but it doesn’t always accurately replicate the lighting effects as seen in the second image, where the neon lighting could have been better captured. IC-Light [122] is effective in applying realistic lighting, but tends to lose detail in some complex objects or faces under different light conditions. Overall, GPT-4o is a strong contender for the image re-light task, providing good light consistency but leaving room for improvement in some specific scenarios.

In summary, GPT-4o demonstrates strong performance in various low-level vision tasks, often surpassing Gemini 2.0 Flash in output quality with more natural and visually appealing results. However, it struggles with maintaining pixel consistency and avoiding undesired changes to image content, which are critical for tasks like restoration and inpainting. While its adaptability and realism are impressive, there is room for improvement in precision and task-specific consistency compared to specialized models like InstructIR and IC-Light.

Denoising

[Figure 576]

[Figure 577]

###### ! Evaluation: Image Quality.

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

InputText:“Remove the noise, make the image clear.”

InstructIR

Input Image GPT 4o Gemini 2.0 Flash

- Figure 24: Task: image denoising, aiming to remove the noise information and obtain high-quality clear version. Setup: We compare GPT-4o with InstructIR [20] and Gemini 2.0 Flash [99] to evaluate the denoised images. Observations: GPT-4o can restore high-quality denoised images. Except for the second image, where the noise cannot be completely removed, the other images are free from noise. However, for low-level tasks, GPT-4o does not maintain content consistency well — the background colors and object shapes in many images have changed, such as the background color in the first image and the floor in the fourth image.

[Figure 618]

Deraining

[Figure 619]

###### ! Evaluation: Image Quality.

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

InputText:“Remove the rain, make the image clear.”

Input Image GPT 4o Gemini 2.0 Flash

InstructIR

- Figure 25: Task: image deraining, aiming to remove the rain streak and get high-quality clear version. Setup: We compare GPT-4o with established baselines such as InstructIR [20] and Gemini 2.0 Flash [99] to evaluate the derained images. Observations: The overall performance of the GPT-4o is well. However, the model struggles with maintaining content consistency in low-level visual details — for instance, the polar bear’s background in the first image becomes unnaturally pink, and the underwater scene loses depth and clarity. The flowers also appear altered in color and arrangement. In contrast, InstructIR demonstrates the most consistent performance across all examples, effectively removing rain while preserving the original scene’s structure, color, and composition. Overall, InstructIR is the most balanced and accurate model for image restoration in this comparison.

[Figure 652]

Dehazing

[Figure 653]

###### ! Evaluation: Image Quality.

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

Input Text: “I took this photo during a foggy day. Can you improve it?”

Input Image GPT 4o Gemini 2.0 Flash

InstructIR

- Figure 26: Task: image dehazing, aiming to remove the haze information and get high-quality clear version. Setup: We compare GPT-4o with established baselines such as InstructIR [20] and Gemini 2.0 Flash [99] to evaluate the dehazed images. Observations: GPT-4o performs moderately well in dehazing, managing to restore clearer structures and contrast in most scenes. However, its outputs often have a grayish or desaturated tone, especially visible in the second and third rows. Gemini 2.0 Flash produces more colorful results but tends to leave some haze behind, leading to a less crisp output. InstructIR outperforms both, offering the most visually natural and sharp dehazing across all examples while preserving original colors and details. Overall, InstructIR demonstrates the strongest capability in removing haze while maintaining realism.

[Figure 678]

Low-light Enhancement

[Figure 679]

###### ! Evaluation: Consistency.

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

Input Text:“I took My image is too dark, I cannot see anything. Can you fix it?”

Input Image GPT 4o Gemini 2.0 Flash InstructIR

- Figure 27: Task: low-light image enhancement, aiming to increase the brightness of the image to obtain a high brightness image. Setup: We compare GPT-4o with established baselines such as InstructIR [20] and Gemini 2.0 Flash [99] to evaluate the brightness images. Observations: In low-light enhancement tasks, GPT-4o can brighten images and recover basic visibility, but often introduces unnatural lighting and loses detail, especially in the second row, where the image remains overly dark. InstructIR consistently delivers the most balanced results, enhancing visibility while preserving true colors and textures, making it the best performer across all three examples.

Debluring

[Figure 704]

[Figure 705]

###### ! Evaluation: Image Quality.

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

Input Text:“I took this photo while I was running, can you stabilize the image? it is too blurry.”

Input Image GPT 4o Gemini 2.0 Flash InstructIR

- Figure 28: Task: image deblurring, aiming to remove the blur information to obtain a clear image. Setup: We compare GPT-4o with established baselines such as InstructIR [20] and Gemini 2.0 Flash [99] to evaluate the deblurred images. Observations: For motion deblurring, GPT-4o recovers some sharpness, especially in fine details like text or faces, but the content is not matched with the original image. Gemini 2.0 Flash sharpens the image slightly better in some cases but can introduce over-smoothing, making the result look artificial. InstructIR demonstrates the best deblurring performance overall — restoring clear edges, facial features, and text while maintaining natural textures. It consistently produces the most stable and visually convincing results across all examples.

[Figure 730]

Super-Resolution

[Figure 731]

###### ! Evaluation: Image Quality.

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

Input Text:“Make my photo bigger and better. Add details to this image. Increase the resolution of this photo.”

Input Image GPT 4o Gemini 2.0 Flash InstructIR

- Figure 29: Task: image super-resolution, aiming to improve the image resolution. Setup: We compare GPT-4o with established baselines such as InstructIR [20] and Gemini 2.0 Flash [99] to evaluate the deblurred images. Observations: In super-resolution, InstructIR delivers the most natural and detailed results across all examples—restoring fine edges in the card reader, realistic texture on the octopus, and sharp trees in the landscape. GPT-4o enhances clarity but misses details like the octopus surface and tree leaves. Gemini 2.0 Flash produces sharper outputs than GPT-4o but introduces unnatural textures and artifacts, especially in organic regions like the octopus and foliage.

[Figure 756]

Inpainting

[Figure 757]

###### ! Evaluation: Image Quality.

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

Input Text:”Please inpainting the image, make it looks reasonable.”

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

Input Text:”Please inpainting the image, make it looks reasonable.”

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

Input Text:“Inpaint the missing part of the face in the image, making the restored area look natural and seamless.”

Input Image GPT 4o Gemini 2.0 Flash LatentPaint

- Figure 30: Task: Image inpainting, aiming to restore missing or masked regions in an image to appear natural and consistent with the context. Setup: We compare GPT-4o with baselines such as Gemini 2.0 Flash [99] and LatentPaint [22], evaluating their ability to fill in masked regions realistically. Observations: GPT-4o produces plausible completions but often lacks fine structure and texture alignment—e.g., the bricks in the first row appear flat and misaligned. Gemini 2.0 Flash generates more visually coherent textures, especially in natural scenes like the second row, but can introduce slight over-smoothing. LatentPaint performs the best, accurately reconstructing facial details and complex textures such as hair and expression in the third row, demonstrating superior semantic understanding and visual consistency.

Outpainting

[Figure 782]

[Figure 783]

###### ! Evaluation: Image Quality.

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

Input Text:“Inpainting this image: a classic dark brown leather Chesterfield loveseat with tufted detailing and rolled arms. It sits in a cozy, traditionally styled living room with green walls, framed artwork, and warm lighting, creating an elegant and vintage atmosphere.”

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

[Figure 799]

Input Text:“Extend the image to the left and right with a realistic continuation of the street, sidewalk, and background buildings. Maintain consistent lighting, shadows, and overall style.”

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

Input Text:“Extend the image to the left and right, filling the black areas with a natural continuation of the snowy mountain landscape, ski path, trees, and sky. Keep the lighting, shadows, and textures consistent with the original image.”

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

Input Text: “Outpaint the center of this panoramic image to naturally connect the left and right desert landscape. Fill the middle area with a realistic continuation of the rocky desert terrain and blue sky with clouds, ensuring seamless blending and consistent perspective.”

Input Image GPT 4o Gemini 2.0 Flash Dream 360

- Figure 31: Task: Image outpainting, aiming to extend the visual content of an image beyond its original boundaries coherently and realistically. Setup: We compare GPT-4o with Gemini 2.0 Flash [99], and some Specialized outpainting methods (SGT+ [116], StrDiffusion [66] and Dream360 [1]), evaluating their ability to extend content while maintaining visual consistency in lighting, texture, and semantics. Observations: The Specialized outpainting methods consistently produces the most coherent extensions — for example, it accurately maintains the room’s lighting and decor in the first row, continues architectural lines and street perspective in the second, and creates seamless snowy landscapes in the third. GPT-4o offers plausible structure but often lacks fine detail and texture continuity, such as mismatched snow gradients or missing shadows. Gemini 2.0 Flash performs slightly better in semantic extension than GPT-4o but can introduce lighting inconsistencies and abrupt transitions, particularly in wide scenes like the desert in the final row.

[Figure 816]

Colorization

[Figure 817]

###### ! Evaluation: Image Quality.

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

Input Text: “Colorize it: a red car parked on a cobblestone street.”

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

Input Text: “Colorize it: a couple of white and black kittens that are sitting in the purple grass.”

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

Input Text: “Colorize it: a red sports car parked on the side of a street.”

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 847]

[Figure 848]

[Figure 849]

Input Text: “Colorize it: a woman wearing a yellow sunglasses with green lips”

Input Image GPT 4o Gemini 2.0 Flash CtrlColor

- Figure 32: Task: Image colorization, aiming to add realistic and semantically consistent color to grayscale images based on textual prompts. Setup: We compare GPT-4o with Gemini 2.0 Flash [99] and CtrlColor [59], focusing on their ability to follow instructions and produce visually natural colorized outputs. Observations: CtrlColor performs the best overall, generating vivid and accurate colors that precisely match the prompts—such as green lips and yellow sunglasses in the last row, or the purple grass and kitten hues in the second. GPT-4o provides reasonably faithful colorization but often lacks richness or misinterprets tones (e.g., slightly dull red in the third row or inconsistent purple grass). Gemini 2.0 Flash is more vivid than GPT-4o but tends to oversaturate or produce stylized effects, especially on human features.

[Figure 850]

Shadow Removal

[Figure 851]

###### ! Evaluation: Image Quality.

[Figure 852]

[Figure 853]

[Figure 854]

[Figure 855]

[Figure 856]

[Figure 857]

[Figure 858]

[Figure 859]

[Figure 860]

[Figure 861]

[Figure 862]

[Figure 863]

[Figure 864]

[Figure 865]

[Figure 866]

[Figure 867]

[Figure 868]

[Figure 869]

[Figure 870]

[Figure 871]

[Figure 872]

[Figure 873]

[Figure 874]

[Figure 875]

[Figure 876]

[Figure 877]

[Figure 878]

[Figure 879]

[Figure 880]

[Figure 881]

[Figure 882]

[Figure 883]

Input Text:“Remove all harsh shadows from the image. Make the lighting even and soft across the entire scene. Preserve all objects, colors, and details exactly as they are. Make it look like it was taken under diffuse studio lighting.”

Input Image GPT 4o Gemini 2.0 Flash ShadowRefiner

- Figure 33: Task: Shadow removal, aiming to eliminate harsh shadows while preserving the integrity of the scene, textures, and lighting balance. Setup: We compare GPT-4o with Gemini 2.0 Flash [99] and ShadowRefiner [25] to evaluate how well each method removes shadows and retains original object fidelity and lighting consistency. Observations: ShadowRefiner consistently achieves the most natural and effective shadow removal. It produces even, diffuse lighting across all scenes—e.g., softening shadows without distorting textures in complex scenes like the miniatures and dog portrait. Gemini 2.0 Flash removes shadows reasonably but occasionally leaves faint traces or flattens contrast, as seen in the second and fourth rows. GPT-4o shows stronger shadow reduction than Gemini 2.0 Flash but sometimes alters surface brightness or loses detail fidelity. ShadowRefiner best preserves the original color tones and textures while eliminating harsh shadows.

[Figure 884]

Reflection Removal

[Figure 885]

###### ! Evaluation: Image Quality.

[Figure 886]

[Figure 887]

[Figure 888]

[Figure 889]

[Figure 890]

[Figure 891]

[Figure 892]

[Figure 893]

Input Text: “Remove window reflections, preserve interior details clearly visible through the glass, maintain natural lighting and perspective, photo-realistic result.”

[Figure 894]

[Figure 895]

[Figure 896]

[Figure 897]

[Figure 898]

[Figure 899]

[Figure 900]

[Figure 901]

Input Text: “Remove the reflection of buildings on the wet ground surface, make it look like a clean and dry textured concrete floor, realistic lighting and natural color tones.”

[Figure 902]

[Figure 903]

[Figure 904]

[Figure 905]

[Figure 906]

[Figure 907]

[Figure 908]

[Figure 909]

Input Text: “Remove reflections from the glass doors, make the interior clearly visible with natural lighting and sharp details, keep the golden door frame realistic and intact.”

[Figure 910]

[Figure 911]

[Figure 912]

[Figure 913]

[Figure 914]

[Figure 915]

[Figure 916]

[Figure 917]

Input Text:“Remove reflections from the car window, make the interior of the vehicle clearly visible, preserve natural lighting and realistic textures, keep the car frame untouched.”

Input Image GPT 4o Gemini 2.0 Flash DSIT

- Figure 34: Task: Reflection removal, aiming to eliminate unwanted reflections from transparent or reflective surfaces while preserving original content and realistic lighting. Setup: We compare GPT-4o with Gemini 2.0 Flash [99] and DSIT [39], assessing their ability to remove reflections while maintaining scene realism, texture fidelity, and lighting consistency. Observations: DSIT shows the most effective and natural reflection removal across all examples. It restores interior visibility through windows (e.g., bed and car interior) while preserving lighting and geometry. Gemini 2.0 Flash removes some reflections but often leaves faded traces or dulls textures, especially on glass doors and wet pavement. GPT-4o performs better than Gemini 2.0 Flash in preserving background details but sometimes alters color tones and sharpness. Overall, DSIT provides the cleanest and most photorealistic results, especially for transparent surfaces like glass and reflective wet ground.

Image Re-lightning

[Figure 918]

[Figure 919]

###### ! Evaluation: Light consistency.

[Figure 920]

[Figure 921]

[Figure 922]

[Figure 923]

[Figure 924]

[Figure 925]

[Figure 926]

[Figure 927]

[Figure 928]

[Figure 929]

Input Text: “Given two input images:

- Image 1: A classical marble statue in neutral lighting.
- Image 2: A city street at night illuminated by neon pink and blue lights. Please generate a relit version of the statue from Image 1, as if it were lit by the lighting conditions of Image 2. The result should preserve the details and pose of the statue but apply realistic colored lighting and shadows consistent with the vibrant, mixed neon lighting of the second image.

Light Map Input Image GPT 4o Gemini Pro 2.0 IC-Light

Text-PromptImage Re-lightning

[Figure 930]

[Figure 931]

[Figure 932]

[Figure 933]

[Figure 934]

[Figure 935]

[Figure 936]

[Figure 937]

[Figure 938]

Input Text:“Sunlight through the blinds, near window blinds with a reasonable background.”

[Figure 939]

[Figure 940]

[Figure 941]

[Figure 942]

[Figure 943]

[Figure 944]

[Figure 945]

[Figure 946]

Input Text:“Sunlight from the left side, beach with a reasonable background.”

[Figure 947]

[Figure 948]

[Figure 949]

[Figure 950]

[Figure 951]

[Figure 952]

[Figure 953]

[Figure 954]

Input Text: “Sunlight from the left side, beach with a reasonable background.”

Input Image GPT 4o Gemini 2.0 Flash IC-Light

- Figure 35: Task: Image relighting, aiming to modify the lighting of a given image based on either a reference light map or a textual description, while preserving identity, texture, and spatial consistency. Setup: We compare GPT-4o with Gemini 2.0 Flash [99] and IC-Light [122] on two subtasks: reference-based and text-based relighting. Evaluations focus on lighting realism, directionality, shadow accuracy, and semantic preservation. Observations: IC-Light achieves the most realistic and consistent relighting across both tasks—accurately applying neon lighting from a reference image and generating sharp shadows and natural light from text prompts. Gemini 2.0 Flash preserves content well but produces softer, less directional lighting. GPT-4o offers more vivid lighting than Gemini 2.0 Flash but sometimes lacks shadow accuracy or background coherence.

###### 2.2.6 Spatial Control

Spatial control aims to generate visual outputs that not only reflect the content described in the prompt, but also precisely adhere to additional structural conditions (e.g., canny edge maps, depth maps, sketches, poses, and masks). This task evaluates a model’s ability to faithfully align text guidance with visual constraints—an essential capability for real-world creative applications such as illustration, animation, digital content creation, and visual storytelling.

In this section, we examine GPT-4o’s performance across five representative types of controllable conditions: canny, depth, sketch, pose, and mask. For each setting, we compare its outputs with those from Gemini 2.0 Flash [99] and a strong baseline method using ControlNet-based [121] diffusion backbones (FLUX.1-Dev [51], SDXL1.0 [82], SD3 Medium [27] or SD1.5 [90]). The results are illustrated in Figures 36, 37, 38, 39, 40.

Overall, GPT-4o achieves performance that is on par with ControlNet-based methods in many cases, especially under common or moderately complex conditions. In particular, GPT-4o is capable of handling semantically rich or contextually complex prompts, where its strong foundation model understanding can help preserve both high-level semantics and visual plausibility. This is especially evident in tasks like pose-to-image or mask-to-image, where the structural signal may be sparse or ambiguous. However, GPT-4o’s strong generative prior can sometimes lead to overly detailed or hallucinated elements, which compromises structural fidelity. For instance, in canny-to-image or depth-to-image tasks that require fine-grained geometric alignment, GPT-4o may deviate from the input layout more noticeably than traditional diffusion-based methods. In contrast, ControlNet exhibits more stable and accurate control in these low-level structure-guided scenarios, making it better suited for applications where spatial accuracy is critical. That said, ControlNet may struggle in more complex or open-ended cases, such as mask-to-image scenes involving multiple objects or interactions (e.g., aquariums with visitors and fish). In these scenarios, GPT-4o’s strong cross-modal understanding partially compensates for its weaker control, offering plausible but not fully precise outputs. By comparison, Gemini 2.0 Flash lacks robust controllable generation capabilities across all evaluated control types. Its outputs often fail to match either the control condition or the textual prompt, reflecting limited capacity in multimodal alignment and structural grounding.

In summary, GPT-4o demonstrates performance comparable to SOTA methods in most cases, excelling in tasks that require rich semantic understanding and contextual complexity while maintaining a balance between high-level semantics and visual plausibility. Although it may exhibit structural deviations in tasks requiring precise geometric alignment, its strong generative prior gives it an advantage in handling complex or open-ended scenarios.

[Figure 955]

Canny-to-Image

[Figure 956]

###### ! Evaluation: Controllability and text consistency.

[Figure 957]

[Figure 958]

[Figure 959]

[Figure 960]

[Figure 961]

[Figure 962]

[Figure 963]

[Figure 964]

InputText:“Follow the prompt and canny condition below to generate a controllable image. The prompt is: a cigarette with purple tobacco.”

[Figure 965]

[Figure 966]

[Figure 967]

[Figure 968]

[Figure 969]

[Figure 970]

[Figure 971]

[Figure 972]

InputText:“Follow the prompt and canny condition below to generate a controllable image. The prompt is: a traffic sign with red cross written on it.”

[Figure 973]

[Figure 974]

[Figure 975]

[Figure 976]

[Figure 977]

[Figure 978]

[Figure 979]

[Figure 980]

InputText:“Follow the prompt and canny condition below to generate a controllable image. The prompt is: oil painting of geese flying in a v formation over a pond at sunset.”

FLUX.1-Dev w. ControlNet

Input Image GPT 4o Gemini 2.0 Flash

- Figure 36: Task: Canny-to-Image generation. The goal is to generate prompt-aligned images guided by canny maps. Setup: Each row shows an input canny map and a text prompt, with outputs from GPT-4o, Gemini 2.0 Flash [99], and FLUX.1-Dev w. ControlNet [51]. Observations: GPT-4o performs worse than FLUX.1-Dev [51] in structural fidelity, often introducing additional visual details that deviate from the input edge map. However, it produces more semantically aligned and aesthetically pleasing results overall. Compared to Gemini 2.0 Flash, GPT-4o significantly outperforms in both structure preservation and prompt consistency.

Depth-to-Image

[Figure 981]

[Figure 982]

###### ! Evaluation: Controllability and text consistency.

[Figure 983]

[Figure 984]

[Figure 985]

[Figure 986]

[Figure 987]

[Figure 988]

[Figure 989]

[Figure 990]

InputText:“Follow the prompt and depth condition below to generate a controllable image. The prompt is: a wooden bridge that has fallen down in the grass.”

[Figure 991]

[Figure 992]

[Figure 993]

[Figure 994]

[Figure 995]

[Figure 996]

[Figure 997]

[Figure 998]

InputText:“Follow the prompt and depth condition below to generate a controllable image. The prompt is: a 3d image of a stone building with plants and rocks.”

[Figure 999]

[Figure 1000]

[Figure 1001]

[Figure 1002]

[Figure 1003]

[Figure 1004]

[Figure 1005]

[Figure 1006]

InputText:“Follow the prompt and depth condition below to generate a controllable image. The prompt is: a red pillow on a chair.”

FLUX.1-Dev w. ControlNet

Input Image GPT 4o Gemini 2.0 Flash

- Figure 37: Task: Depth-to-image generation, aiming to synthesize controllable and visually coherent images based on a text prompt and a given depth map. Setup: We compare GPT-4o with Gemini 2.0 Flash [99] and FLUX.1-Dev w. ControlNet [51], focusing on controllability, text-prompt alignment, and the visual quality of generated scenes. Observations: GPT-4o generates visually appealing and stylistically consistent images that align reasonably with text and depth cues—such as the bridge scene and stone ruins with rich lighting and artistic tone. However, its controllability is weaker than FLUX.1-Dev w. ControlNet [51], which shows more precise depth alignment and object placement, as seen in the accurate layout of the bridge and red pillow. GPT-4o leans toward stylized coherence, while FLUX emphasizes photorealism with sharper spatial fidelity. Gemini 2.0 Flash lags behind both, often showing depth misalignment, shape distortion, and weaker semantic grounding.

[Figure 1007]

Sketch-to-Image

[Figure 1008]

###### ! Evaluation: Controllability and text consistency.

[Figure 1009]

[Figure 1010]

[Figure 1011]

[Figure 1012]

[Figure 1013]

[Figure 1014]

[Figure 1015]

[Figure 1016]

InputText:“Follow the prompt and sketch condition below to generate a controllable image. The prompt is: A small giraffe eating grass.”

[Figure 1017]

[Figure 1018]

[Figure 1019]

[Figure 1020]

[Figure 1021]

[Figure 1022]

[Figure 1023]

[Figure 1024]

InputText:“Follow the prompt and sketch condition below to generate a controllable image. The prompt is: A red metal electric fan.”

[Figure 1025]

[Figure 1026]

[Figure 1027]

[Figure 1028]

[Figure 1029]

[Figure 1030]

[Figure 1031]

[Figure 1032]

InputText:“Follow the prompt and sketch condition below to generate a controllable image. The prompt is: a man holding on to the strings of a flying parachute.”

SDXL1.0 w. ControlNet

Input Image GPT 4o Gemini 2.0 Flash

- Figure 38: Task: Sketch-to-image generation, which requires translating rough line drawings into realistic and semantically accurate images guided by text prompts. Setup: We evaluate GPT-4o against Gemini 2.0 Flash [99] and SDXL1.0 w. ControlNet [82], focusing on how well each model respects the provided sketch while reflecting the described content. Observations: GPT-4o excels at generating lifelike scenes that match the prompt, often delivering visually pleasing and contextually grounded outputs—like the natural posture and setting of the giraffe or the dynamic movement in the parachute example. However, it tends to soften or reinterpret sketch lines, leading to slight mismatches in fine structure. In contrast, SDXL1.0 w. ControlNet [82] offers stronger adherence to the input sketch, capturing geometric details more accurately (e.g., fan blades and figure outlines), albeit with slightly more synthetic textures. Gemini 2.0 Flash shows limited understanding of both sketch and prompt, often producing less realistic or structurally off-target images.

[Figure 1033]

Pose-to-Image

[Figure 1034]

###### ! Evaluation: Controllability and text consistency.

[Figure 1035]

[Figure 1036]

[Figure 1037]

[Figure 1038]

[Figure 1039]

[Figure 1040]

[Figure 1041]

[Figure 1042]

InputText:“Follow the prompt and pose condition below to generate a controllable image. The prompt is: Quarterback in a blue and white jersey with number 14, preparing to throw a football during a game.”

[Figure 1043]

[Figure 1044]

[Figure 1045]

[Figure 1046]

[Figure 1047]

[Figure 1048]

[Figure 1049]

[Figure 1050]

InputText:“Follow the prompt and pose condition below to generate a controllable image. The prompt is : A young woman with long brown hair, wearing a blue strapless dress and a black necklace with a butterfly pendant, poses against a beige background.”

[Figure 1051]

[Figure 1052]

[Figure 1053]

[Figure 1054]

[Figure 1055]

[Figure 1056]

[Figure 1057]

[Figure 1058]

InputText:“Follow the prompt and pose condition below to generate a controllable image. The prompt is : A woman is performing a pull-up exercise on a gym rack.”

SD3 Medium w. ControlNet

Input Image GPT 4o Gemini 2.0 Flash

- Figure 39: Task: Pose-to-image generation, aiming to synthesize realistic images that reflect both the human pose and descriptive prompt. Setup: We benchmark GPT-4o against Gemini 2.0 Flash [99] and SD3 Medium w. ControlNet [27], evaluating their ability to follow pose conditions while generating semantically accurate and coherent images. Observations: GPT-4o performs well in complex scenes—such as the football example—where it effectively integrates pose, clothing, and background with strong realism, contextual and pose accuracy. In simpler cases like the pull-up exercise, it shows occasional pose drift, especially in limbs. SD3 Medium w. ControlNet [27] offers better pose fidelity overall, though its visual quality can be inconsistent. Gemini 2.0 Flash underperforms in both structure and coherence, often generating anatomically incorrect or visually weak results. Overall, GPT-4o balances text understanding and generation quality, especially in detailed prompts.

[Figure 1059]

Mask-to-Image

[Figure 1060]

###### ! Evaluation: Controllability and text consistency.

[Figure 1061]

[Figure 1062]

[Figure 1063]

[Figure 1064]

[Figure 1065]

[Figure 1066]

[Figure 1067]

[Figure 1068]

InputText:“Follow the prompt and pose condition below to generate a controllable image. The prompt is : A peaceful indoor church scene with a plain wall, stained glass windows, a wooden podium, and a stone altar under soft sunlight. ”

[Figure 1069]

[Figure 1070]

[Figure 1071]

[Figure 1072]

[Figure 1073]

[Figure 1074]

[Figure 1075]

[Figure 1076]

InputText:“Follow the prompt and pose condition below to generate a controllable image. The prompt is : An indoor aquarium scene with a large fish tank full of colorful tropical fish swimming. The fish tank is surrounded by walls and has a visible floor at the bottom. The environment is bright and underwater-themed. ”

[Figure 1077]

[Figure 1078]

[Figure 1079]

[Figure 1080]

[Figure 1081]

[Figure 1082]

[Figure 1083]

[Figure 1084]

InputText:“Follow the prompt and mask condition below to generate a controllable image. The prompt is: An indoor aquarium with a large fish tank and colorful tropical fish, with a few visitors in the scene.”

Input Image GPT 4o Gemini 2.0 Flash SD w. ControlNet

- Figure 40: Task: Mask-to-image generation, which requires translating semantic segmentation maps and textual prompts into coherent and realistic images. Setup: We compare GPT-4o with Gemini 2.0 Flash [99] and SD1.5 w. ControlNet [90], focusing on their ability to combine spatial layout from the mask with deeper scene understanding from the prompt. Observations: Compared to previous control tasks, this setting demands more from the model in terms of semantic reasoning and compositional understanding. GPT-4o excels in this regard, producing visually consistent scenes that align with the prompt’s intent—such as the serene church interior and the immersive aquarium setting with visitors. However, in fine-grained spatial control, especially with small or tightly shaped objects like tropical fish, SD1.5 w. ControlNet [90] performs better in preserving shape and positioning. Gemini 2.0 Flash continues to struggle in both fidelity and adherence to masks, often missing key scene elements or producing oversimplified outputs.

###### 2.2.7 Camera Control

Although recent visual generative models demonstrate remarkable capabilities in creating high-quality images, generating images with specific camera settings (e.g., bokeh blur parameters, focal length, shutter speed, color temperature) and making further adjustments remains a challenging task. We further explore GPT-4o’s performance in camera control, evaluating its ability to generate images with desired photographic parameters in text instructions. This task is particularly significant as it bridges the gap between artistic creativity and technical precision, enabling users to simulate professional photography techniques and achieve greater control over the visual output. Such advancements have broad applications in fields like photography, cinematography, and visual design.

Specifically, we collect text prompts from [118], and compare GPT-4o and Gemini 2.0 Flash [99] with Generative Photography (GP) [118]. The results are reported in Figures 41, 42. We can observe that GPT-4o achieves decent results in controlling bokeh blur parameters and color temperature, demonstrating its strong generalizability to various photographic settings. However, it still falls short in adjusting focal length and shutter speed, occasionally leading to inconsistent visual semantics or incorrect visual effects. By comparison, Gemini 2.0 Flash struggles significantly across all camera control scenarios, failing to produce coherent or accurate outputs that align with the specified photographic parameters, highlighting its limited capability in this domain.

In this task, GPT-4o shows promising potential in camera control, outperforming Gemini 2.0 Flash and achieving competitive results in certain aspects. Nonetheless, there remains room for improvement in handling more complex adjustments, which could further enhance its applicability in professional photography and creative industries.

[Figure 1085]

Camera Control

[Figure 1086]

###### ! Evaluation: Camera setting adjustment, semantic consistency.

[Figure 1087]

[Figure 1088]

[Figure 1089]

[Figure 1090]

[Figure 1091]

[Figure 1092]

[Figure 1093]

[Figure 1094]

[Figure 1095]

[Figure 1096]

GPT 4o

[Figure 1097]

[Figure 1098]

[Figure 1099]

[Figure 1100]

[Figure 1101]

[Figure 1102]

[Figure 1103]

[Figure 1104]

[Figure 1105]

[Figure 1106]

Gemini 2.0 Flash

[Figure 1107]

[Figure 1108]

[Figure 1109]

[Figure 1110]

[Figure 1111]

[Figure 1112]

[Figure 1113]

[Figure 1114]

[Figure 1115]

[Figure 1116]

GP

28.0 14.0 10.0 6.0 2.0

InputText:“A horse with a white face stands in a grassy field, looking at the camera; with bokeh blur parameter *” & “Adjust the bokeh blur parameter to *” (* indicates a specific value).

[Figure 1117]

[Figure 1118]

[Figure 1119]

[Figure 1120]

[Figure 1121]

[Figure 1122]

[Figure 1123]

[Figure 1124]

[Figure 1125]

[Figure 1126]

GPT 4o

[Figure 1127]

[Figure 1128]

[Figure 1129]

[Figure 1130]

[Figure 1131]

[Figure 1132]

[Figure 1133]

[Figure 1134]

[Figure 1135]

[Figure 1136]

Gemini 2.0 Flash

[Figure 1137]

[Figure 1138]

[Figure 1139]

[Figure 1140]

[Figure 1141]

[Figure 1142]

[Figure 1143]

[Figure 1144]

[Figure 1145]

[Figure 1146]

GP

24.9 36.9 48.9 60.9 69.9 InputText:“A beautiful garden filled with red roses and green leaves; with * mm lens” & “Adjust the lens to * mm”.

- Figure 41: Task: Camera control. The goal is to generate images aligned with specific photographic parameters, such as bokeh blur, focal length, shutter speed, and color temperature. Setup: Results are based on text prompts collected from [118], comparing outputs from GPT-4o, Gemini 2.0 Flash [99], and Generative Photography (GP) [118]. Each row includes the input text instructions and corresponding outputs. Observations: GPT-4o demonstrates strong performance in controlling bokeh blur, producing visually appealing and parameter-aligned results. However, it shows limitations in handling focal length, occasionally generating inconsistent or less accurate outputs. By contrast, Gemini 2.0 Flash struggles significantly in both aspects, often failing to produce coherent results. Overall, GPT-4o achieves better performance in this task but still requires further refinement to enhance focal length control.

[Figure 1147]

Camera Control

[Figure 1148]

###### ! Evaluation: Camera setting adjustment, semantic consistency.

[Figure 1149]

[Figure 1150]

[Figure 1151]

[Figure 1152]

[Figure 1153]

[Figure 1154]

[Figure 1155]

[Figure 1156]

[Figure 1157]

[Figure 1158]

GPT 4o

[Figure 1159]

[Figure 1160]

[Figure 1161]

[Figure 1162]

[Figure 1163]

[Figure 1164]

[Figure 1165]

[Figure 1166]

[Figure 1167]

[Figure 1168]

Gemini 2.0 Flash

[Figure 1169]

[Figure 1170]

[Figure 1171]

[Figure 1172]

[Figure 1173]

[Figure 1174]

[Figure 1175]

[Figure 1176]

[Figure 1177]

[Figure 1178]

GP

0.88 0.68 0.48 0.38 0.28

InputText:“A blue pot with a plant in it is placed on a window sill, surrounded by other potted plants; with shutter speed * second” & “Adjust the shutter speed to * second”.

[Figure 1179]

[Figure 1180]

[Figure 1181]

[Figure 1182]

[Figure 1183]

[Figure 1184]

[Figure 1185]

[Figure 1186]

[Figure 1187]

[Figure 1188]

GPT 4o

[Figure 1189]

[Figure 1190]

[Figure 1191]

[Figure 1192]

[Figure 1193]

[Figure 1194]

[Figure 1195]

[Figure 1196]

[Figure 1197]

[Figure 1198]

Gemini 2.0 Flash

[Figure 1199]

[Figure 1200]

[Figure 1201]

[Figure 1202]

[Figure 1203]

[Figure 1204]

[Figure 1205]

[Figure 1206]

[Figure 1207]

[Figure 1208]

GP

3100.0 4000.0 8000.0 7000.0 3000.0

InputText: “A collection of trash cans and a potted plant are seen in the image. The trash cans are individually in blue, black and yellow; with temperature * kelvin” & “Adjust the temperature to * kelvin”.

- Figure 42: Task: Camera control. The goal is to generate images aligned with specific photographic parameters, such as bokeh blur, focal length, shutter speed, and color temperature. Setup: Results are based on text prompts collected from [118], comparing outputs from GPT-4o, Gemini 2.0 Flash [99], and Generative Photography (GP) [118]. Each row includes the input text instructions and corresponding outputs. Observations: GPT-4o demonstrates strong performance in controlling color temperature, producing coherent and visually accurate results. However, it struggles with shutter speed, occasionally resulting in inconsistent or unrealistic motion effects. In contrast, Gemini 2.0 Flash fails to consistently handle either parameter, often producing outputs that lack alignment with the desired settings. Overall, GPT-4o outperforms Gemini 2.0 Flash in this task, but further improvements are needed for precise shutter speed control.

###### 2.2.8 In-context Visual Prompting

The in-context visual prompting tasks aim at understanding and executing specific tasks on new query images by leveraging a pair of task-specific example images and accompanying text instructions. Previous works [105, 18, 52] have explored this capability in the context of diffusion and autoregressive models, demonstrating its potential in enhancing model adaptability. The significance of in-context visual prompting lies in its ability to enable models to generalize to novel tasks. This approach mirrors human-like learning, where new tasks can be understood and performed by observing relevant examples. This capability has broad implications across various domains, and paves the way for more flexible and efficient paradigms capable of adapting to a wide range of specific tasks.

We curate four representative tasks to evaluate the performance of GPT-4o in in-context visual prompting. These tasks are designed to assess the model’s ability to understand and adapt to specific visual tasks based on provided examples and guidance, including:

- • Movie-Shot Generation: A three-shot image collected from [42] is provided as an example, and the model is instructed to follow this format to generate similar movie shots for the query image.
- • Ray-Tracing Rendering: An example gaming scene is provided with and without ray tracing, and the model is expected to render a ray-traced version of the query image.
- • Overlaid Mask Visualization: The model receives an original image accompanied by its corresponding segmented results from [49] and is tasked with outputting the segmented results in the same format for the query image.
- • Maze Solving: A maze and its corresponding solution path are provided as examples, and the model is required to draw the solution path for a new maze presented in the query image.

All the results are illustrated in Figure 43. Compared with Gemini 2.0 Flash [99], GPT-4o demonstrates promising performance in movie-shot generation and ray-tracing rendering tasks, showcasing its ability to follow example formats and generate visually coherent outputs. However, it still struggles with maintaining consistent visual semantics across the generated outputs. For the overlaid mask visualization task, GPT-4o falls short in effectively executing the instructions. The result fails to adhere to the required format, indicating that the model’s ability to process and generate complex outputs remains limited. For maze solving, a task that demands advanced visual reasoning and logical inference, GPT-4o struggles significantly. This highlights the challenges in combining higherlevel reasoning with visual generation capabilities, suggesting that more sophisticated reasoning mechanisms are needed for tasks of this nature.

In summary, GPT-4o shows considerable potential in in-context visual prompting, while it still underperforms in certain difficult tasks. These observations suggest that further advancements are necessary to enhance its generation and reasoning capabilities for more complex and diverse visual tasks.

In-ContextVisualPrompting

[Figure 1209]

[Figure 1210]

###### ! Evaluation: Understanding and executing specific tasks with example images.

[Figure 1211]

[Figure 1212]

[Figure 1213]

[Figure 1214]

[Figure 1215]

[Figure 1216]

[Figure 1217]

[Figure 1218]

InputText:“The first image contains three movie shots. Please imitate this image and create the subsequent movie shots for the second image..”

Input Image Input Image GPT 4o

Gemini 2.0 Flash

[Figure 1219]

[Figure 1220]

[Figure 1221]

[Figure 1222]

GPT 4o

Input Image

[Figure 1223]

[Figure 1224]

[Figure 1225]

[Figure 1226]

Input Image

Gemini 2.0 Flash

InputText:“The first image includes an original gaming scene, and the scene enhanced with ray tracing. Please imitate this image and create the scene enhanced with ray tracing for the second image.”

[Figure 1227]

[Figure 1228]

[Figure 1229]

[Figure 1230]

[Figure 1231]

[Figure 1232]

[Figure 1233]

[Figure 1234]

InputText:“The first image shows an original image and its segmented results. Please imitate this image and output the segmented results in the same format for the second image.”

Input Image Input Image

GPT 4o

Gemini 2.0 Flash

[Figure 1235]

[Figure 1236]

[Figure 1237]

[Figure 1238]

[Figure 1239]

[Figure 1240]

[Figure 1241]

[Figure 1242]

InputText:“The first image displays an unsolved maze and the maze with a solution path in red. Please imitate this image and identify the solution path for the second image.”

Input Image Input Image GPT 4o

Gemini 2.0 Flash

- Figure 43: Task: In-context visual prompting. The goal is to perform specific visual tasks on new query images based on task-specific example images and text instructions. Setup: Four representative tasks are evaluated: movie-shot generation, ray-tracing rendering, overlaid mask visualization, and maze solving. Each row includes example images, query images, and the corresponding outputs. Observations: GPT-4o excels in movie-shot generation and ray-tracing, producing coherent outputs but lacks consistency in visual semantics. It fails with overlaid mask visualization and maze solving, showing limits in complex task integration. While promising for in-context visual prompting, it needs refinement for more complex and reasoning-intensive tasks.

###### 2.3 Image-to-3D Tasks

We evaluate the 3D understanding capabilities from 2D images of GPT-4o across three tasks: 2D image-to-3D modeling, 2D UV map-to-3D rendering, and novel view synthesis.

###### 2.3.1 Image to 3D modeling

Generating 3D models from monocular images boosts a wide range of applications, including augmented reality, virtual reality, and the gaming industry. This capability not only facilitates the content creation process but also mitigates the reliance on specialized 3D artists for creating 3D assets, which is more time- and cost-effective. Therefore, there is a growing research interest in generating 3D models from 2D images. Early methods on image-to-3D employ the learning-based approaches for single-view reconstruction [74, 77, 102, 79]. Recent works leverage the diffusion model prior to perform image-conditioned 3D generative modeling [69, 68, 83, 113].

In this section, we investigate the potential of GPT-4o for 3D modeling from 2D images. We begin by prompting GPT-4o to generate a Cinema 4D modeling interface to test its ability to produce coherent representations of structure, material, and wireframe based on the input image. As shown in Figure 44, GPT-4o can generate high-quality 3D model renderings within the application interface. Notably, the generated models exhibit clear wireframes and textures consistent with the input images. In contrast, Gemini 2.0 Flash and Midjourney v6.1 fail to achieve comparable results under the same conditions, which produce inconsistent modelings. We then prompt the GPT-4o to generate corresponding 3D object and material files in .obj and .mtl formats to further evaluate its understanding of the underlying structure in the rendered images. However, the output 3D models are coarse and inconsistent with input images, indicating that although GPT-4o can produce visually coherent 3D renderings, its capability to transform these into accurate and usable 3D object files remains limited. Additionally, Gemini 2.0 Flash and Midjourney v6.1 do not support exporting 3D models.

###### 2.3.2 UV Map to 3D rendering

UV maps are 2D images that store texture information for 3D models. In 3D modeling, geometric data is represented in 3D space, while texture data is defined in a 2D texture space. UV mapping is the process of projecting a 2D UV map onto a 3D model, accurately aligning texture with geometry. The UV mapping process can evaluate models’ capability for 3D perception and spatial understanding. Moreover, this task has broad applications in design, helping to reduce the burden on designers to create product renderings from 2D maps manually and provide useful references.

As shown in Figure 45, GPT-4o exhibits a superior ability to generate consistent 3D renderings from 2D maps compared to Gemini 2.0 Flash and Midjourney v6.1. However, some outputs remain unsatisfactory, displaying inconsistencies in patterns and structure (see row 3 in Figure 45). Gemini 2.0 Flash struggles to correctly wrap the 3D model, though it maintains pattern consistency. Midjourney v6.1 tends to introduce additional, imagined features, which reduce controllability in this task.

###### 2.3.3 Novel View Synthesis

From a monocular view, humans can imagine an object’s 3D shape and appearance since humans have collected enough prior knowledge for different objects throughout their daily lives. This ability to infer novel views of objects is essential for a wide range of tasks, from object manipulation to artistic creation such as painting. Early works achieve image-to-3D reconstruction using category-specific priors or large-scale pre-training [45, 80, 87, 32, 131]. Recent studies have shown that large diffusion models contain rich 3D prior information of the visual world, enabling them to perform novel view synthesis [69, 68, 83, 70]. These novel views can then be used for zero-shot 3D reconstruction using different 3D representations such as NeRF [76], mesh, or SDF.

In this section, we evaluate the ability of GPT-4o for novel view synthesis on objects with artistic styles and asymmetric geometry. As shown in Figure 46, for artistically styled objects, GPT-4o and Gemini 2.0 Flash largely preserve structural consistency with the input image, although they may change some elements or fine details. For the asymmetric object, GPT-4o can preserve the object scale and size better than Gemini 2.0 Flash. However, Midjourney v6.1 fails to generate consistent novel views, instead producing visually appealing images that do not align with the given prompt of this task.

Image to 3D Model

[Figure 1243]

[Figure 1244]

###### 🌟 Evaluation: Shape/texture consistency, wireframe plausibility.

[Figure 1245]

[Figure 1246]

[Figure 1247]

[Figure 1248]

[Figure 1249]

[Figure 1250]

[Figure 1251]

[Figure 1252]

[Figure 1253]

[Figure 1254]

[Figure 1255]

[Figure 1256]

[Figure 1257]

[Figure 1258]

[Figure 1259]

[Figure 1260]

[Figure 1261]

[Figure 1262]

[Figure 1263]

[Figure 1264]

[Figure 1265]

[Figure 1266]

[Figure 1267]

[Figure 1268]

[Figure 1269]

[Figure 1270]

[Figure 1271]

[Figure 1272]

[Figure 1273]

[Figure 1274]

[Figure 1275]

[Figure 1276]

InputText:“Generate a pre-render view of a C4D model, including the UI, wireframe and material.”

Input Image GPT 4o Midjourney v6.1

Gemini 2.0 Flash

- Figure 44: Task: Image-to-3D model rendering. Evaluate the 3D modeling ability given a 2D image. Setup: Each row shows an input image and a text prompt with outputs from GPT-4o, Gemini 2.0 Flash [99], and Midjourney v6.1 [75]. Observation: GPT-4o can generate better 3D model rendering with consistent shape, texture, and plausible wireframe than Gemini 2.0 Flash and Midjourney v6.1.

2D UV map to 3D rendering

[Figure 1277]

[Figure 1278]

###### 🌟 Evaluation: Structure/pattern consistency.

[Figure 1279]

[Figure 1280]

[Figure 1281]

[Figure 1282]

[Figure 1283]

[Figure 1284]

[Figure 1285]

[Figure 1286]

[Figure 1287]

[Figure 1288]

[Figure 1289]

[Figure 1290]

[Figure 1291]

[Figure 1292]

[Figure 1293]

[Figure 1294]

[Figure 1295]

[Figure 1296]

[Figure 1297]

[Figure 1298]

[Figure 1299]

[Figure 1300]

[Figure 1301]

[Figure 1302]

[Figure 1303]

[Figure 1304]

[Figure 1305]

[Figure 1306]

[Figure 1307]

[Figure 1308]

[Figure 1309]

[Figure 1310]

InputText:“Assemble this packaging cutout into a complete product and output a 3D rendered image.”

Input Image GPT 4o Gemini 2.0 Flash Midjourney v6.1

- Figure 45: Task: 2D UV map to 3D rendering. Evaluate the 3D perception and spatial understanding ability. Setup: Each row shows an input image and a text prompt with outputs from GPT-4o, Gemini 2.0 Flash [99], and Midjourney v6.1 [75]. Observation: GPT-4o can generate better 3D renderings based on 2D maps than Gemini 2.0 Flash and Midjourney v6.1. However, structure and pattern inconsistencies still exist among these three models.

[Figure 1311]

[Figure 1312]

NovelViewSynthesis

[Figure 1313]

###### 🌟 Evaluation: Consistency.

[Figure 1314]

[Figure 1315]

[Figure 1316]

[Figure 1317]

[Figure 1318]

[Figure 1319]

[Figure 1320]

[Figure 1321]

[Figure 1322]

[Figure 1323]

[Figure 1324]

[Figure 1325]

[Figure 1326]

[Figure 1327]

[Figure 1328]

[Figure 1329]

[Figure 1330]

[Figure 1331]

[Figure 1332]

[Figure 1333]

[Figure 1334]

[Figure 1335]

[Figure 1336]

[Figure 1337]

[Figure 1338]

[Figure 1339]

[Figure 1340]

[Figure 1341]

[Figure 1342]

[Figure 1343]

InputText: “Generate three views of this picture.”

Input Image GPT 4o Gemini 2.0 Flash Midjourney v6.1

- Figure 46: Task: Novel view synthesis. Evaluate the 3D perception and spatial understanding ability. Setup: Each row shows an input image and a text prompt with outputs from GPT-4o, Gemini 2.0 Flash [99], and Midjourney v6.1 [75]. Observation: GPT-4o can generate better style and structure-consistent novel views for both artistic painting and asymmetric objects.

###### 2.4 Image-to-X Tasks

In this section, we further evaluate both GPT-4o and Gemini 2.0 Flash for several dense image understanding tasks, including segmentation-related tasks, depth estimation, normal estimation, matting, salient object detection, edge detection, layout detection, text detection, and object tracking.

###### 2.4.1 Image Segmentation

Image segmentation tasks group pixels of the given image or video into semantic regions. It is a fundamental problem in computer vision and involves numerous real-world applications, such as robotics, automated surveillance, and image/video editing. With the development of recent deep learning methods, this domain has achieved rapid progress. Early works mainly adopt CNN-based methods with large kernels or respective fields. Recently, transformer-based methods have also worked well and surpassed previous CNN-based methods on various benchmarks. In particular, we test three segmentation tasks, including referring segmentation, semantic segmentation, and panoptic segmentation.

Referring Segmentation. This task outputs the corresponding mask according to the input texts, and the goal is to test the pixel-level grounding ability of the model. In Figure 47, we compare GPT-4o, Gemin 2.0 Flash and recent state-of-the art method, Sa2VA [117] (8B model †). We show five open-world test cases. For the first two cases, GPT-4o shows the coarse localization ability on the background region. For example, it can mark the grass region despite the unfavorable boundaries. However, compared to the SOTA method, Sa2VA, GPT-4o mistakenly merges both large regions. In the third row, both GPT-4o and Gemini 2.0 Flash cannot perform grounding with complex text inputs. In the fourth row, all models perform badly. GPT-4o generates an unseen chair in the images while Gemin 2.0 Flash performs image editing functions by replacing the smallest chair with a normal chair. Sa2VA also segments the wrong object (the nearest chair). In the last example, GPT-4o also cannot segment smaller objects (“bag”). For all examples, both GPT-4o and Gemini 2.0 Flash modify the image contents. These examples indicate that GPT-4o has weak pixel grounding ability.

Semantic Segmentation. Semantic segmentation assigns each pixel a semantic label, which is one basic vision task. In Figure 48, we show several test cases on the semantic segmentation task. In particular, we adopt DeeplabV3+ [14] (ResNet101 as backbone, trained on Pascal-Context) as one expert model for reference. Surprisingly, the mask quality of GPT-4o is good on four examples, even comparable with an expert model, Deeplab-V3+. During the testing, we find the texts may be randomly appended to the masks. This is why the first row differs from the remaining examples. For the second and third examples, GPT-4o misaligns the text and mask regions. Compared to Gemin 2.0 Flash, GPT-4o has a much stronger ability in semantic segmentation, particularly for mask shape. However, there is still a lot of room for this task, including a unified semantic segmentation format, enhanced text and mask alignments, and more correct mask labels.

Panoptic Segmentation. This task assigns the foreground region a semantic label and assigns one mask label and one instance ID to each instance, which is a unified task format of semantic segmentation and instance segmentation.

- In Figure 49, we compare the panoptic segmentation ability of GPT-4o, Gemini 2.0 Flash, and one expert model, K-Net [123](trained on the COCO panoptic segmentation dataset, with ResNet50 as backbone). Overall, the mask shapes of GPT-4o are good. The model can understand the panoptic segmentation task, while the Gemini 2.0 Flash cannot do this task in the first and third cases. However, the spatial locations have been changed for all cases. The generated masks are in part-whole formats and are even finer-grained than K-Net. For example, in the first example, the jersey number (17) of the person and the hair of the people are also marked. Meanwhile, we also find a similar issue: several examples have text, while several do not have text, even though they adopt the same text prompt. In addition, GPT-4o can distinguish different instances with different colors, despite most of them not being good (see the last example).

†https://huggingface.co/ByteDance/Sa2VA-8B

[Figure 1344]

[Figure 1345]

###### 🌟 Evaluation: Referring Expression Segmentation, Grounding and Grouping.

[Figure 1346]

[Figure 1347]

[Figure 1348]

[Figure 1349]

Input Text:“Please segment the grass in the image and directly generate the output image. ”

[Figure 1350]

[Figure 1351]

[Figure 1352]

[Figure 1353]

[Figure 1354]

[Figure 1355]

[Figure 1356]

[Figure 1357]

Input Text:“Please segment the sand in the image and directly generate the output image.”

[Figure 1358]

[Figure 1359]

[Figure 1360]

[Figure 1361]

[Figure 1362]

[Figure 1363]

[Figure 1364]

[Figure 1365]

Input Text:“Please segment the table beside the black sofa in the image and directly generate the output image.”

[Figure 1366]

[Figure 1367]

[Figure 1368]

[Figure 1369]

[Figure 1370]

[Figure 1371]

[Figure 1372]

[Figure 1373]

Input Text:“Please segment the smallest chair and directly generate the output image.”

[Figure 1374]

[Figure 1375]

[Figure 1376]

[Figure 1377]

[Figure 1378]

[Figure 1379]

[Figure 1380]

[Figure 1381]

Input Text:“Please segment the bag in the image and directly generate the output image.”

Input Image GPT 4o Gemini 2.0 Flash Sa2VA

- Figure 47: Task: Image to X: Referring expression segmentation. Evaluate the grounding and grouping ability. Setup: Each row shows an input image and a text prompt with outputs from GPT-4o, Gemini 2.0 Flash [99], and Sa2VA [117]. Observation: These examples indicate that current GPT-4o has weak pixel-level grounding ability.

Image-to-X

[Figure 1382]

[Figure 1383]

###### 🌟 Evaluation: Semantic Segmentation, Shape and Grouping.

[Figure 1384]

[Figure 1385]

[Figure 1386]

[Figure 1387]

[Figure 1388]

[Figure 1389]

[Figure 1390]

[Figure 1391]

GPT 4o

Input Text: “Please generate the semantic segmentation result of the image.”

[Figure 1392]

[Figure 1393]

[Figure 1394]

[Figure 1395]

[Figure 1396]

[Figure 1397]

[Figure 1398]

[Figure 1399]

Input Text:“Please generate the semantic segmentation result of the image.”

[Figure 1400]

[Figure 1401]

[Figure 1402]

[Figure 1403]

[Figure 1404]

[Figure 1405]

[Figure 1406]

[Figure 1407]

Input Text:“Please generate the semantic segmentation result of the image.”

[Figure 1408]

[Figure 1409]

[Figure 1410]

[Figure 1411]

[Figure 1412]

[Figure 1413]

[Figure 1414]

[Figure 1415]

Input Text:“Please generate the semantic segmentation result of the image.”

Input Image GPT 4o Gemini 2.0 Flash Deeplab-V3+

- Figure 48: Task: Image to X: Semantic segmentation. Evaluate the shape and grouping ability. Setup: Each row shows an input image and a text prompt with outputs from GPT-4o, Gemini 2.0 Flash [99], and Deeplab-V3+ [14]. Observation: Compared with Gemin-2.0, the mask quality of GPT-4o is good. However, there are still huge gaps in the standard semantic segmentation format.

[Figure 1416]

[Figure 1417]

###### 🌟 Evaluation: Panoptic Segmentation, Grouping and Shape.

[Figure 1418]

[Figure 1419]

[Figure 1420]

[Figure 1421]

[Figure 1422]

[Figure 1423]

[Figure 1424]

[Figure 1425]

Input Text:“Please generate the panoptic segmentation result of the image.”

[Figure 1426]

[Figure 1427]

[Figure 1428]

[Figure 1429]

[Figure 1430]

[Figure 1431]

[Figure 1432]

[Figure 1433]

Input Text:“Please generate the panoptic segmentation result of the image.”

[Figure 1434]

[Figure 1435]

[Figure 1436]

[Figure 1437]

[Figure 1438]

[Figure 1439]

[Figure 1440]

[Figure 1441]

Input Text:“Please generate the panoptic segmentation result of the image.”

[Figure 1442]

[Figure 1443]

[Figure 1444]

[Figure 1445]

[Figure 1446]

[Figure 1447]

[Figure 1448]

[Figure 1449]

Input Text:“Please generate the panoptic segmentation result of the image.”

[Figure 1450]

[Figure 1451]

[Figure 1452]

[Figure 1453]

[Figure 1454]

[Figure 1455]

[Figure 1456]

[Figure 1457]

Input Text:“Please generate the panoptic segmentation result of the image.”

Input Image GPT 4o Gemini 2.0 Flash K-Net

- Figure 49: Task: Image to X: Panoptic segmentation. Evaluate the shape and grouping ability. Setup: Each row shows an input image and a text prompt with outputs from GPT-4o, Gemini 2.0 Flash [99], and K-Net [123]. Observation: GPT-4o can understand the panoptic segmentation task, while Gemini 2.0 Flash cannot do this task in the first and third cases.

###### 2.4.2 Edge Detection

Edge Detection. As a classic vision task, edge detection aims to identify the boundaries or edges of objects within an image. These edges represent the locations with significant changes in image intensity, color, or other visual features. Common edge detection operators include the Sobel, Prewitt, and Canny operators. Recent works adopt deep learning-based approaches.

- In Figure 50, we compare this ability with a recent SOTA deep learning based approach, EMDB [56]. For four examples, we find both GPT-4o and Gemini 2.0 Flash can detect object edges for both foreground and background objects. In addition, the details are even good using GPT-4o. We find two critical issues: 1) The spatial localization of GPT-4o is changed as observed by the segmentation tasks. 2) The content of GPT-4o is also changed. For example, in the first example, the road is generated, which does not exist in the input image.

Image Matting. Image matting is a technique in image processing that aims to separate a foreground object from its background and obtain a detailed alpha matte, which indicates the transparency or opacity of each pixel in the foreground. It goes beyond simple segmentation by providing more precise information about the boundaries and fine details of the object, especially for complex objects like hair or smoke.

- In Figure 51, we show three testing examples, with one expert model, Matting Anything [53]. Compared with Gemini, GPT-4o can handle the simple cases, as shown in the third row. Thus, it can understand the task goal. For example, it can even keep the fine-grained details of a horse hair. However, considering the strict requirements of image matting (fine-grained and aligned details), the overall quality is bad. Compared with Matting Anything, both GPT-4o and Gemini work poorly. We find nearly the same issues: 1) Wrong spatial localization, 2) Changed contents.

Image-to-X

[Figure 1458]

[Figure 1459]

###### ! Evaluation: Edge Detection, Shape Analysis.

[Figure 1460]

[Figure 1461]

[Figure 1462]

[Figure 1463]

Input Text:“Please detect the edge of object in this image and output the final image.”

[Figure 1464]

[Figure 1465]

[Figure 1466]

[Figure 1467]

Input Text:“Please detect the edge of object in this image and output the final image.”

[Figure 1468]

[Figure 1469]

[Figure 1470]

[Figure 1471]

Input Text:“Please detect the edge of object in this image and output the final image.”

Input Image GPT 4o Gemini 2.0 Flash EDMB

- Figure 50: Task: Image to X: Edge detection. Evaluate the shape analysis ability. Setup: Each row shows an input image and a text prompt with outputs from GPT-4o, Gemini 2.0 Flash [99], and EDMB [56]. Observation: We find both GPT-4o and Gemini 2.0 Flash can detect object edges for both foreground and background objects.

Image-to-X

[Figure 1472]

[Figure 1473]

###### 🌟 Evaluation: Image Matting, Grouping and Shape.

[Figure 1474]

[Figure 1475]

[Figure 1476]

[Figure 1477]

[Figure 1478]

[Figure 1479]

[Figure 1480]

[Figure 1481]

Input Text:“Please Please matting the foreground and remove the background. Please directly generate the output image.”

[Figure 1482]

[Figure 1483]

[Figure 1484]

[Figure 1485]

[Figure 1486]

[Figure 1487]

[Figure 1488]

[Figure 1489]

Input Text:“Please Please matting the foreground and remove the background. Please directly generate the output image.”

[Figure 1490]

[Figure 1491]

[Figure 1492]

[Figure 1493]

[Figure 1494]

[Figure 1495]

[Figure 1496]

[Figure 1497]

Input Text:“Please Please matting the foreground and remove the background. Please directly generate the output image.”

Matting Anything

Input Image

GPT 4o Gemini 2.0 Flash

- Figure 51: Task: Image to X: Image matting. Evaluate the grouping and shape analysis ability. Setup: Each row shows an input image and a text prompt with outputs from GPT-4o, Gemini 2.0 Flash [99], and Matting Anything [54]. Observation: Compared with Gemini, GPT-4o can handle the simple cases, as shown in the third row. However, considering the strict requirements of image matting (fine-grained and aligned details), the overall quality is bad.

###### 2.4.3 Salient Object

Salient Object Detection. Salient object detection is a crucial technique in the field of computer vision and image processing. It aims to identify and locate the most visually prominent objects within an image or a video sequence.

- In Figure 52, we adopt one expert model, BiRefNet [127], as reference. For all examples, compared with Gemini

- 2.0 Flash, GPT-4o can detect relevant salient objects with the text prompts while Gemini can not achieve this. The second example shows that the GPT-4o can generate the aligned salient masks. However, for other examples, the spatial location is not changed where the results are generated according to the input image and potential classes. In the last examples, GPT-4o cannot generate multiple salient object masks, which is also a limitation when dealing with multiple objects.

Mirror Detection. Mirror detection is a task in computer vision that focuses on identifying mirror surfaces within an image or a scene. Previous works explore this direction by adopting visual cues and geometric cues.

- In Figure 53, we also explore this ability for both GPT-4o and Gemini 2.0 Flash. As for comparison, we adopt a recent SOTA expert model, VMD [107]. For simple cases, we find that GPT-4o can carry out mirror detection, as shown in the first example. For the complex scene, it cannot work as well as the expert model, VMD. As shown in the second example, it generates a fake mirror and leads to a wrong image output with a line to mark the boundaries of the fake mirror. As shown in the last row, GPT-4o treats several rectangular objects as mirrors, leading to several false positive examples.

Shadow Detection. Shadow detection is a significant process in computer vision and image processing that aims to identify and localize shadow regions in an image or a video. This technique is crucial, as shadows can otherwise disrupt object detection, recognition, and scene analysis.

- In Figure 54, we compare and test this ability for GPT-4o. We adopt the SOTA model, SDDNet [21] for reference. For the simple examples (single objects and no objects in the image), both GPT-4o and Gemini can localize the shadow, as shown in the first two rows. For more complex examples, both models detect both objects and their shadows with one mask output, as shown in the last two rows. Thus, GPT-4o cannot handle these inputs. In addition, the spatial misalignments also happen for all the cases.

Camouflage Object Detection. Camouflage object detection is a challenging task in computer vision. It aims to identify objects that are designed to blend into their backgrounds, making them difficult to distinguish by human eyes or traditional detection methods. This has a wide application for the military, security, and wildlife conservation.

As shown in Figure 55, we also include one expert model, BiRefNet [127] for reference. For all examples, both GPT-4o and Gemini 2.0 Flash can detect and segment the camouflage animals for simple cases, as shown in the last two rows. GPT-4o can also detect the specific object, given the text prompt, as shown in the first row. However, the same misalignment issues still exist. In addition, it also mixes segmentation maps (in binary masks or color masks), as shown in the last row.

Image-to-X

[Figure 1498]

[Figure 1499]

###### ! Evaluation: Salient Object Detection, Grouping and Shape.

[Figure 1500]

[Figure 1501]

[Figure 1502]

[Figure 1503]

[Figure 1504]

[Figure 1505]

[Figure 1506]

[Figure 1507]

Input Text:“Give me the segmentation map of the most salient objects in this image. Return resulting image by using image generation. ”

[Figure 1508]

[Figure 1509]

[Figure 1510]

[Figure 1511]

[Figure 1512]

[Figure 1513]

[Figure 1514]

[Figure 1515]

Input Text:“Give me the segmentation map of the most salient objects in this image. Return resulting image by using image generation. ”

[Figure 1516]

[Figure 1517]

[Figure 1518]

[Figure 1519]

[Figure 1520]

[Figure 1521]

[Figure 1522]

[Figure 1523]

Input Text:“Give me the segmentation map of the most salient objects in this image. Return resulting image by using image generation. ”

[Figure 1524]

[Figure 1525]

[Figure 1526]

[Figure 1527]

[Figure 1528]

[Figure 1529]

[Figure 1530]

[Figure 1531]

Input Text:“Give me the segmentation map of the most salient objects in this image. Return resulting image by using image generation. ”

Input Image GPT 4o Gemini 2.0 Flash BiRefNet

- Figure 52: Task: Image to X: Salient object detection. Evaluate the grouping and shape analysis ability. Setup: Each row shows an input image and a text prompt with outputs from GPT-4o, Gemini 2.0 Flash [99], and BiRefNet [127]. Observation: For all examples, compared with Gemini, GPT-4o can detect related salient objects with the text prompts while Gemini can not achieve this function.

Image-to-X

[Figure 1532]

[Figure 1533]

###### 🌟 Evaluation: Mirror Detection, Grouping and Shape.

[Figure 1534]

[Figure 1535]

[Figure 1536]

[Figure 1537]

[Figure 1538]

[Figure 1539]

[Figure 1540]

[Figure 1541]

Input Text:“Please segment all the mirror in the image and directly generate the output image.”

[Figure 1542]

[Figure 1543]

[Figure 1544]

[Figure 1545]

[Figure 1546]

[Figure 1547]

[Figure 1548]

[Figure 1549]

Input Text:“Please segment all the mirror in the image and directly generate the output image.”

[Figure 1550]

[Figure 1551]

[Figure 1552]

[Figure 1553]

[Figure 1554]

[Figure 1555]

[Figure 1556]

[Figure 1557]

Input Text:“Please segment all the mirror in the image and directly generate the output image.”

Input Image GPT 4o Gemini 2.0 Flash VMD

- Figure 53: Task: Image to X: Mirror detection. Evaluate the grouping and shape analysis ability. Setup: Each row shows an input image and a text prompt with outputs from GPT-4o, Gemini 2.0 Flash [99], and VMD [107]. Observation: For simple cases, we find that GPT-4o can carry out mirror detection, as shown in the first example. For the complex scene, it cannot work as well as VMD.

Image-to-X

[Figure 1558]

[Figure 1559]

###### ! Evaluation: Shadow Detection, Grouping and Shape.

[Figure 1560]

[Figure 1561]

[Figure 1562]

[Figure 1563]

[Figure 1564]

[Figure 1565]

[Figure 1566]

[Figure 1567]

Input Text:“Give me with the segmentation map of the shadow in this image. Set the shadow region to white and the other regions to black. Return the resulting image using image generation. ”

[Figure 1568]

[Figure 1569]

[Figure 1570]

[Figure 1571]

[Figure 1572]

[Figure 1573]

[Figure 1574]

[Figure 1575]

Input Text:“Give me with the segmentation map of the shadow in this image. Set the shadow region to white and the other regions to black. Return the resulting image using image generation. ”

[Figure 1576]

[Figure 1577]

[Figure 1578]

[Figure 1579]

[Figure 1580]

[Figure 1581]

[Figure 1582]

[Figure 1583]

Input Text:“Give me with the segmentation map of the shadow in this image. Set the shadow region to white and the other regions to black. Return the resulting image using image generation. ”

[Figure 1584]

[Figure 1585]

[Figure 1586]

[Figure 1587]

[Figure 1588]

[Figure 1589]

[Figure 1590]

[Figure 1591]

Input Text:“Give me with the segmentation map of the shadow in this image. Set the shadow region to white and the other regions to black. Return the resulting image using image generation. ”

Input Image GPT-4o Gemini 2.0 Flash SDDNet

- Figure 54: Task: Image to X: Shadow detection. Evaluate the grouping and shape analysis ability. Setup: Each row shows an input image and a text prompt with outputs from GPT-4o, Gemini 2.0 Flash [99], and SDDNet [21]. Observation: For more complex examples, both models detect both objects and their shadows with one mask output, as shown in the last two rows, leading to false positive predictions.

Image-to-X

[Figure 1592]

[Figure 1593]

###### ! Evaluation: Camouflage Object Detection, Grouping and Shape.

[Figure 1594]

[Figure 1595]

[Figure 1596]

[Figure 1597]

[Figure 1598]

[Figure 1599]

[Figure 1600]

[Figure 1601]

Input Text:“Give me the segmentation map of the crocodile in this image. Return resulting image by using image generation. ”

[Figure 1602]

[Figure 1603]

[Figure 1604]

[Figure 1605]

[Figure 1606]

[Figure 1607]

[Figure 1608]

[Figure 1609]

Input Text:“Give me the segmentation map of the fish in this image. Return resulting image by using image generation. ”

[Figure 1610]

[Figure 1611]

[Figure 1612]

[Figure 1613]

[Figure 1614]

[Figure 1615]

[Figure 1616]

[Figure 1617]

Input Text:“Give me the segmentation map of the fish in this image. Return resulting image by using image generation. ”

[Figure 1618]

[Figure 1619]

[Figure 1620]

[Figure 1621]

[Figure 1622]

[Figure 1623]

[Figure 1624]

[Figure 1625]

Input Text:“Give me the segmentation map of the toad in this image. Return resulting image by using image generation. ”

Input Image GPT-4o Gemini 2.0 Flash BiRefNet

- Figure 55: Task: Image to X: Camouflage object detection. Evaluate the grouping and shape analysis ability. Setup: Each row shows an input image and a text prompt with outputs from GPT-4o, Gemini 2.0 Flash [99], and BiRefNet [127]. Observation: Both GPT-4o and Gemini 2.0 Flash can detect and segment the camouflage animals for simple cases. However, the spatial misalignments still exist.

###### 2.4.4 Depth Estimation

The depth estimation task involves predicting the distance from the camera to objects within a scene. In this paper, we focus on monocular depth estimation, which takes a single image as input. In Figure 56, we compare GPT-4o, Gemini 2.0 Flash, and a recent SOTA method, Depth-Anything [114]. We first notice that Gemini cannot produce reasonable depth estimations. For GPT-4o, although it can output a fancy depth map visualization, we want to point out that this output is a grayscale visualization of depth estimation and cannot be directly converted to the depth of each pixel. We show mainly five cases. In the first test case, we notice that GPT-4o is good at capturing details in images, which Depth-Anything may not be good at. Although we cannot directly determine the accuracy of the depth value, we can judge from the visualization that the depth relationship between objects is accurate. What GPT-4o cannot do well is the background. Since the background in the image is the sky, we can infer from common sense that these areas are infinitely far away from the camera. However, the depth map output of GPT-4o does not handle these areas correctly. GPT-4o performs similarly in the second, fourth, and fifth examples. Among them, we would like to emphasize the fourth test case, since for buildings farther away, GPT-4o has no way to effectively analyze the distance between each building and the camera. In the third example, although the output of GPT-4o is very confusing, it completely misunderstands the depth relationship of the entire image. Therefore, we believe that the depth estimation performance of GPT-4o is still unstable.

[Figure 1626]

###### Image-to-X

[Figure 1627]

### 🌟 Evaluation: Depth Estimation

[Figure 1628]

[Figure 1629]

[Figure 1630]

[Figure 1631]

[Figure 1632]

[Figure 1633]

[Figure 1634]

[Figure 1635]

InputText: “Please generate the depth map prediction of this image.”

[Figure 1636]

[Figure 1637]

[Figure 1638]

[Figure 1639]

[Figure 1640]

[Figure 1641]

[Figure 1642]

[Figure 1643]

InputText: “Please generate the depth map prediction of this image.”

[Figure 1644]

[Figure 1645]

[Figure 1646]

[Figure 1647]

[Figure 1648]

[Figure 1649]

[Figure 1650]

[Figure 1651]

InputText: “Please generate the depth map prediction of this image.”

[Figure 1652]

[Figure 1653]

[Figure 1654]

[Figure 1655]

[Figure 1656]

[Figure 1657]

[Figure 1658]

[Figure 1659]

InputText: “Please generate the depth map prediction of this image.”

[Figure 1660]

[Figure 1661]

[Figure 1662]

[Figure 1663]

[Figure 1664]

[Figure 1665]

[Figure 1666]

[Figure 1667]

InputText: “Please generate the depth map prediction of this image.”

Input Image GPT-4o Gemini 2.0 Flash Depth-Anything

- Figure 56: Task: Image to X: Depth estimation. Setup: Each row shows an input image and a text prompt with outputs from GPT-4o, Gemini 2.0 Flash [99], and Depth-Anything [114]. Observation: We convert the depth map generated by Depth-Anything into a visualization map similar to GPT-4o. This evaluation shows that GPT-4o has the capability of distinguishing the depth relationship of different parts in the image, but its understanding of the background is insufficient.

###### 2.4.5 Normal Estimation

The surface normal estimation task involves predicting the orientation of surfaces at each pixel in an image, typically represented as 3D vectors. In Figure 57, we compare GPT-4o, Gemini 2.0 Flash, and Marigold normals [48]. The results show that GPT-4o can generate reasonable results. However, since GPT-4o’s output is an appealing normal map visualization, we want to clarify that this output is a color-coded visualization and does not directly provide the exact normal vector for each pixel. Thus, we cannot use lighting or other methods to verify the accuracy of the normal maps, and downstream tasks cannot use the output results. However, we also find some unreasonable details. In the third test case, common sense suggests that the ground should be flat, but GPT-4o predicts normals for these textured areas that differ from the surrounding areas.

Image-to-X

[Figure 1668]

[Figure 1669]

###### 🌟 Evaluation: Consistency/accuracy.

[Figure 1670]

[Figure 1671]

[Figure 1672]

[Figure 1673]

[Figure 1674]

[Figure 1675]

[Figure 1676]

[Figure 1677]

[Figure 1678]

[Figure 1679]

[Figure 1680]

[Figure 1681]

[Figure 1682]

[Figure 1683]

[Figure 1684]

[Figure 1685]

[Figure 1686]

[Figure 1687]

[Figure 1688]

[Figure 1689]

[Figure 1690]

[Figure 1691]

[Figure 1692]

[Figure 1693]

Input Text: ”Generate the surface normal map of this picture.”

Input Image GPT 4o Gemini 2.0 Flash Marigold

- Figure 57: Task: Image to X: Normal estimation. Setup: Each row shows an input image and a text prompt with outputs from GPT-4o, Gemini 2.0 Flash [99], and Marigold [48]. Observation: This evaluation shows that GPT-4o has the capability of generating a visualization map of the surface normal, but the understanding of the details is still insufficient.

###### 2.4.6 Layout Detection

The layout detection task requires the model to identify structural components (e.g., titles, paragraphs, tables, images) in the given image. In Figure 58, we compare the performance of GPT-4o, Gemini 2.0 Flash, and LayoutLMV3 [44] on the layout detection task. In the test cases, GPT-4o hallucinates layout elements that do not exist, although the final output is another document with “layout detection” results. If we consider the use in downstream tasks, such results are meaningless. Therefore, we conclude that GPT-4o is not capable of the layout detection task.

Image-to-X

[Figure 1694]

[Figure 1695]

###### 🌟 Evaluation: Document Detection.

[Figure 1696]

[Figure 1697]

[Figure 1698]

[Figure 1699]

Input Text:“Generate a new image which contains the layout detection results of the input image.”

[Figure 1700]

[Figure 1701]

[Figure 1702]

[Figure 1703]

Input Text:“Generate a new image which contains the layout detection results of the input image.”

Input Image GPT 4o Gemini 2.0 Flash LayoutLMV3

- Figure 58: Task: Image to X: Layout detection. Setup: Each row shows an input image and a text prompt with outputs from GPT-4o, Gemini 2.0 Flash [99], and LayoutLMV3 [44]. Observation: The results show that GPT-4o and Gemini frequently generate a different document but a correct detected layout.

###### 2.4.7 Text Detection

The text detection task requires the model to detect the texts in the given image. In Figure 59, we compare the performance of GPT-4o, Gemini 2.0 Flash [99], and CRAFT [3] regarding to text detection. We observe that CRAFT exhibits better performance compared to the other models.

In the first test case, GPT-4o demonstrates comparable performance to CRAFT. However, in other cases, GPT-4o continuously generates some nonexistent texts and labels them as “text area”. This issue becomes particularly evident in cluttered scenes or images with complex backgrounds. These false positives not only reduce detection precision but also make the output less reliable for downstream tasks such as OCR or document understanding. On the other hand, Gemini does not generate nonexistent texts but tends to over-predict some areas as text areas.

Image-to-X

[Figure 1704]

[Figure 1705]

###### ! Evaluation: Text Detection.

[Figure 1706]

[Figure 1707]

[Figure 1708]

[Figure 1709]

Input Text:“Generate a new image and label each line of text in the image with a green box ”

[Figure 1710]

[Figure 1711]

[Figure 1712]

[Figure 1713]

Input Text:“Generate a new image and label each line of text in the image with a green box ”

[Figure 1714]

[Figure 1715]

[Figure 1716]

[Figure 1717]

Input Text:“Generate a new image and label each line of text in the image with a green box ”

[Figure 1718]

[Figure 1719]

[Figure 1720]

[Figure 1721]

Input Text:“Generate a new image and label each line of text in the image with a green box ”

Input Image GPT 4o Gemini 2.0 Flash

CRAFT

- Figure 59: Task: Image to X: Text detection. Setup: Each row shows an input image and a text prompt with outputs from GPT-4o, Gemini 2.0 Flash [99], and CRAFT [3]. Observation: The results show that GPT-4o frequently generates text that does not exist.

###### 2.4.8 Object Tracking

The object tracking task requires the model to continuously locate and follow the specific object across the frames in a video sequence. We test the multi-object tracking, which requires the model to track several objects concurrently. We test four cases (Figure 60, 61, 62, 63). We compare GPT-4o, Gemini 2.0 Flash, and a recent SOTA method SAM-2 [86]. Our first observation is that GPT-4o seems unable to generate images that are consistent with the original image. This may be related to the nature of its generative model. Even if we ignore this, for the tracking task, SAM-2 still performs better, while GPT-4o will have problems such as failing to maintain consistent tracking of the target, frequently drifting, or losing the object entirely. In Figure 60, the output of GPT-4o generally demonstrates the ability to track objects, but there are also some defects. For example, a new object is even created out of the existing objects in the last picture generated by GPT-4o. We speculate that this is caused by the influence of the conversation context. In Figure 61, GPT-4o outputs some content that should not be in the output, such as the “caf” tag. In Figure 62, GPT-4o can track a relatively simple object, but it fuses two separate objects. In Figure 63, GPT-4o lacks the capability of tracking in the dense scenario.

Image-to-X

[Figure 1722]

[Figure 1723]

###### ! Evaluation: Object Tracking, Matching and Video Analysis.

[Figure 1724]

[Figure 1725]

[Figure 1726]

[Figure 1727]

[Figure 1728]

[Figure 1729]

[Figure 1730]

[Figure 1731]

Input Text:“This is the first frame of a video where I've marked four targets with different colored bounding boxes. I'll subsequently provide you with other frames from this video for object tracking of these four targets. Understood?”

[Figure 1732]

[Figure 1733]

[Figure 1734]

[Figure 1735]

[Figure 1736]

[Figure 1737]

[Figure 1738]

[Figure 1739]

Input Text:“You now need to perform object tracking on the four targets in this image and draw the detected bounding boxes on them. Please provide me directly with the final output image. Return result image by using image generation.”

[Figure 1740]

[Figure 1741]

[Figure 1742]

[Figure 1743]

[Figure 1744]

[Figure 1745]

[Figure 1746]

[Figure 1747]

Input Text:“Continue tracking the four targets on this new frame and draw the detected bounding boxes on them. Please provide me directly with the final output image. Return result image by using image generation.”

[Figure 1748]

[Figure 1749]

[Figure 1750]

[Figure 1751]

[Figure 1752]

[Figure 1753]

[Figure 1754]

[Figure 1755]

Input Text:“Continue tracking the four targets on this new frame and draw the detected bounding boxes on them. Please provide me directly with the final output image. Return result image by using image generation.”

Input Image GPT 4o Gemini 2.0 Flash SAM2

- Figure 60: Task: Image to X: Object tracking, matching, and video analysis (1/4). Setup: Each row shows an input image and a text prompt with outputs from GPT-4o, Gemini 2.0 Flash [99], and SAM-2 [86]. Observation: This evaluation shows that GPT-4o has the capability of tracking objects, but it cannot generate a consistent image compared to the input image. 81

Image-to-X

[Figure 1756]

[Figure 1757]

###### ! Evaluation: Object Tracking, Matching and Video Analysis.

[Figure 1758]

[Figure 1759]

[Figure 1760]

[Figure 1761]

[Figure 1762]

[Figure 1763]

[Figure 1764]

[Figure 1765]

Input Text:“This is the first frame of a video where I've marked three targets with different colored bounding boxes. I'll subsequently provide you with other frames from this video for object tracking of these three targets. Understood?”

[Figure 1766]

[Figure 1767]

[Figure 1768]

[Figure 1769]

[Figure 1770]

[Figure 1771]

[Figure 1772]

[Figure 1773]

Input Text:“You now need to perform object tracking on the three targets in this image and draw the detected bounding boxes on them. Please provide me directly with the final output image. Return result image by using image generation.”

[Figure 1774]

[Figure 1775]

[Figure 1776]

[Figure 1777]

[Figure 1778]

[Figure 1779]

[Figure 1780]

[Figure 1781]

Input Text:“Continue tracking the three targets on this new frame and draw the detected bounding boxes on them. Please provide me directly with the final output image. Return result image by using image generation.”

[Figure 1782]

[Figure 1783]

[Figure 1784]

[Figure 1785]

[Figure 1786]

[Figure 1787]

[Figure 1788]

[Figure 1789]

Input Text:“Continue tracking the three targets on this new frame and draw the detected bounding boxes on them. Please provide me directly with the final output image. Return result image by using image generation.”

Input Image GPT 4o Gemini 2.0 Flash SAM2

- Figure 61: Task: Image to X: Object tracking, matching, and video analysis (2/4). Setup: Each row shows an input image and a text prompt with outputs from GPT-4o, Gemini 2.0 Flash [99], and SAM-2 [86]. Observation: This evaluation shows that GPT-4o has the capability of tracking objects, but it cannot generate a consistent image compared to the input image.

Image-to-X

[Figure 1790]

[Figure 1791]

###### ! Evaluation: Object Tracking, Matching and Video Analysis.

[Figure 1792]

[Figure 1793]

[Figure 1794]

[Figure 1795]

[Figure 1796]

[Figure 1797]

[Figure 1798]

[Figure 1799]

Input Text:“This is the first frame of a video where I've marked four targets with different colored bounding boxes. I'll subsequently provide you with other frames from this video for object tracking of these four targets. Understood?”

[Figure 1800]

[Figure 1801]

[Figure 1802]

[Figure 1803]

[Figure 1804]

[Figure 1805]

[Figure 1806]

[Figure 1807]

Input Text:“You now need to perform object tracking on the four targets in this image and draw the detected bounding boxes on them. Please provide me directly with the final output image. Return result image by using image generation.”

[Figure 1808]

[Figure 1809]

[Figure 1810]

[Figure 1811]

[Figure 1812]

[Figure 1813]

[Figure 1814]

[Figure 1815]

Input Text:“Continue tracking the four targets on this new frame and draw the detected bounding boxes on them. Please provide me directly with the final output image. Return result image by using image generation.”

[Figure 1816]

[Figure 1817]

[Figure 1818]

[Figure 1819]

[Figure 1820]

[Figure 1821]

[Figure 1822]

[Figure 1823]

Input Text:“Continue tracking the four targets on this new frame and draw the detected bounding boxes on them. Please provide me directly with the final output image. Return result image by using image generation.”

Input Image GPT 4o Gemini 2.0 Flash SAM2

- Figure 62: Task: Image to X: Object tracking, matching, and video analysis (3/4). Setup: Each row shows an input image and a text prompt with outputs from GPT-4o, Gemini 2.0 Flash [99], and SAM-2 [86]. Observation: This evaluation shows that GPT-4o has the capability of tracking objects, but it cannot generate a consistent image compared to the input image.

Image-to-X

[Figure 1824]

[Figure 1825]

###### ! Evaluation: Object Tracking, Matching and Video Analysis.

[Figure 1826]

[Figure 1827]

[Figure 1828]

[Figure 1829]

[Figure 1830]

[Figure 1831]

[Figure 1832]

[Figure 1833]

Input Text:“This is the first frame of a video where I've marked six targets with different colored bounding boxes. I'll subsequently provide you with other frames from this video for object tracking of these six targets. Understood?”

[Figure 1834]

[Figure 1835]

[Figure 1836]

[Figure 1837]

[Figure 1838]

[Figure 1839]

[Figure 1840]

[Figure 1841]

Input Text:“You now need to perform object tracking on the six targets in this image and draw the detected bounding boxes on them. Please provide me directly with the final output image. Return result image by using image generation.”

[Figure 1842]

[Figure 1843]

[Figure 1844]

[Figure 1845]

[Figure 1846]

[Figure 1847]

[Figure 1848]

[Figure 1849]

Input Text:“Continue tracking the six targets on this new frame and draw the detected bounding boxes on them. Please provide me directly with the final output image. Return result image by using image generation.”

[Figure 1850]

[Figure 1851]

[Figure 1852]

[Figure 1853]

[Figure 1854]

[Figure 1855]

[Figure 1856]

[Figure 1857]

Input Text:“Continue tracking the six targets on this new frame and draw the detected bounding boxes on them. Please provide me directly with the final output image. Return result image by using image generation.”

Input Image GPT 4o Gemini 2.0 Flash SAM2

- Figure 63: Task: Image to X: Object tracking, matching, and video analysis (4/4). Setup: Each row shows an input image and a text prompt with outputs from GPT-4o, Gemini 2.0 Flash [99], and SAM-2 [86]. Observation: This evaluation shows that GPT-4o has the capability of tracking objects, but it cannot generate a consistent image compared to the input image.

###### 3 Limitations

Although GPT-4o demonstrates impressive capabilities across a wide range of image generation tasks, several limitations remain. These challenges highlight key areas for future improvement in developing unified foundation models for vision-language generation.

3.1 Inconsistent Generation

While GPT-4o often produces high-quality and semantically relevant images conditioned on textual prompts, it occasionally exhibits inconsistencies. Specifically, the model may generate visually compelling outputs that deviate from precise semantic cues of the input image, such as object count, spatial layout, specific shapes, or designated colors. These inconsistencies are especially problematic in tasks requiring partial image editing or compositional accuracy. Notably, such issues are less common in diffusion-based models or discrete denoising architectures like MaskGIT [11, 6], suggesting that GPT-4o operates under a distinct generative paradigm with inherent trade-offs in fidelity and control.

3.2 Hallucination

GPT-4o is also susceptible to hallucinations—producing content that is logically implausible, semantically inconsistent, or factually incorrect. These include fabricating non-existent objects or geographical features (e.g., imaginary islands or landmarks), and misrepresenting relationships between entities. Such errors are particularly prevalent in complex or underspecified prompts, where the model appears to rely on internal priors rather than grounded world knowledge. While hallucination is a common challenge across generative models, it poses notable limitations for real-world applications demanding precision, such as education, medical illustration, or scientific visualization.

3.3 Data Bias

Despite strong alignment between text and vision modalities, GPT-4o struggles with data bias issue, which fail in generating underrepresented cultural elements and rendering non-Latin scripts such as Chinese, Japanese, and Arabic. The generated characters are often incomplete, distorted, or replaced with Latin-like approximations. These artifacts reflect underlying challenges in multilingual representation, likely due to limited exposure to diverse scripts during training and the inherent difficulty of accurate typographic rendering in pixel space. This phenomenon is emblematic of a larger issue in AI systems—data bias. The training data used to develop models like GPT-4o may disproportionately represent certain languages, cultures, and writing systems, leading to disparities in performance across different linguistic groups. These biases are not only technical limitations but also ethical concerns, as they can contribute to the exclusion of underrepresented languages and cultures from AI applications. As vision-language models are increasingly deployed globally, improving support for multilingual text remains a crucial step toward inclusive and culturally competent AI systems.

###### 4 Conclusion

In conclusion, this work presents a comprehensive study on the development of unified vision-language generative models, with a focus on evaluating GPT-4o across a wide range of image generation tasks. Our analysis shows that GPT-4o demonstrates strong capabilities in aligning vision and language, achieving competitive results across text-to-image, image-to-image, image-to-3D, and image-to-X tasks. However, limitations remain in inconsistent generation, hallucination, and data bias in underrepresented cultural elements and non-Latin scripts, highlighting current trade-offs in model design and training data coverage. We also emphasize that architecture alone does not determine success; training data, model scale, and optimization strategies are equally critical components of progress. We hope future work will provide deeper empirical insights into such proprietary systems and clarify their position within the broader landscape of unified generative modeling.

###### References

- [1] Hao Ai, Zidong Cao, Haonan Lu, Chen Chen, Jian Ma, Pengyuan Zhou, Tae-Kyun Kim, Pan Hui, and Lin Wang. Dream360: Diverse and immersive outdoor virtual scene creation via transformer-based 360 image outpainting. IEEE transactions on visualization and computer graphics, 2024. 34, 42
- [2] Ideogram AI. Ideogram. https://ideogram.ai/, 2024. 10, 11, 12

- [3] Youngmin Baek, Bado Lee, Dongyoon Han, Sangdoo Yun, and Hwalsuk Lee. Character region awareness for text detection. In CVPR, 2019. 78, 79
- [4] Jinbin Bai, Wei Chow, Ling Yang, Xiangtai Li, Juncheng Li, Hanwang Zhang, and Shuicheng Yan. Humanedit: A high-quality human-rewarded dataset for instruction-based image editing. arXiv preprint arXiv:2412.04280, 2024. 21
- [5] Jinbin Bai, Zhen Dong, Aosong Feng, Xiao Zhang, Tian Ye, Kaicheng Zhou, and Mike Zheng Shou. Integrating view conditions for image synthesis. arXiv preprint arXiv:2310.16002, 2023. 21
- [6] Jinbin Bai, Tian Ye, Wei Chow, Enxin Song, Qing-Guo Chen, Xiangtai Li, Zhen Dong, Lei Zhu, and Shuicheng Yan. Meissonic: Revitalizing masked generative transformers for efficient high-resolution text-to-image synthesis. arXiv preprint arXiv:2410.08261, 2024. 5, 85
- [7] Shane Barratt and Rishi Sharma. A note on the inception score. arXiv preprint arXiv:1801.01973, 2018. 1
- [8] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2023. 5
- [9] Manuel Brack, Felix Friedrich, Katharina Kornmeier, Linoy Tsaban, Patrick Schramowski, Kristian Kersting, and Apolinário Passos. Ledits++: Limitless image editing using text-to-image models. 2023. 21, 25
- [10] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. arXiv preprint arXiv:2211.09800, 2022. 21
- [11] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11315–11325, 2022. 85
- [12] Haoyu Chen, Xiaojie Xu, Wenbo Li, Jingjing Ren, Tian Ye, Songhua Liu, Ying-Cong Chen, Lei Zhu, and Xinchao Wang. Posta: A go-to framework for customized artistic poster generation. arXiv preprint arXiv:2503.14908, 2025. 10, 12
- [13] Liang Chen, Shuai Bai, Wenhao Chai, Weichu Xie, Haozhe Zhao, Leon Vinci, Junyang Lin, and Baobao Chang. Multimodal representation alignment for image generation: Text-image interleaved control is easier than you think. arXiv preprint arXiv:2502.20172, 2025. 1
- [14] Liang-Chieh Chen, George Papandreou, Florian Schroff, and Hartwig Adam. Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587, 2017. 62, 64
- [15] Sixiang Chen, Tian Ye, Jinbin Bai, Erkang Chen, Jun Shi, and Lei Zhu. Sparse sampling transformer with uncertaintydriven ranking for unified removal of raindrops and rain streaks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 13106–13117, 2023. 34
- [16] Sixiang Chen, Tian Ye, Yun Liu, and Erkang Chen. Snowformer: Context interaction transformer with scale-awareness for single image desnowing. arXiv preprint arXiv:2208.09703, 2022. 34
- [17] Sixiang Chen, Tian Ye, Kai Zhang, Zhaohu Xing, Yunlong Lin, and Lei Zhu. Teaching tailored to talent: Adverse weather restoration via prompt pool and depth-anything constraint. In European Conference on Computer Vision, pages 95–115. Springer, 2024. 34
- [18] Tianqi Chen, Yongfei Liu, Zhendong Wang, Jianbo Yuan, Quanzeng You, Hongxia Yang, and Mingyuan Zhou. Improving in-context learning in diffusion models with visual context-modulated prompts. arXiv preprint arXiv:2312.01408, 2023. 56
- [19] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025. 1
- [20] Marcos V. Conde, Gregor Geigle, and Radu Timofte. Instructir: High-quality image restoration following human instructions. In ECCV, 2024. 34, 35, 36, 37, 38, 39, 40
- [21] Runmin Cong, Yuchen Guan, Jinpeng Chen, Wei Zhang, Yao Zhao, and Sam Kwong. Sddnet: Style-guided dual-layer disentanglement network for shadow detection. In ACM MM, 2023. 69, 72
- [22] Ciprian Corneanu, Raghudeep Gadde, and Aleix M Martinez. Latentpaint: Image inpainting in latent space with diffusion models. In WACV, 2024. 34, 41
- [23] Yingying Deng, Fan Tang, Weiming Dong, Chongyang Ma, Xingjia Pan, Lei Wang, and Changsheng Xu. Stytr2: Image style transfer with transformers. In CVPR, 2022. 18
- [24] Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, et al. Dreamllm: Synergistic multimodal comprehension and creation. arXiv preprint arXiv:2309.11499,

2023. 1

- [25] Wei Dong, Han Zhou, Yuqiong Tian, Jingke Sun, Xiaohong Liu, Guangtao Zhai, and Jun Chen. Shadowrefiner: Towards mask-free shadow removal via fast fourier transformer. arXiv preprint arXiv:2406.02559. 44
- [26] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first International Conference on Machine Learning, 2024. 1

- [27] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024. 10, 11, 47, 51
- [28] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 12873–12883,

2021. 1

- [29] Aosong Feng, Weikang Qiu, Jinbin Bai, Kaicheng Zhou, Zhen Dong, Xiao Zhang, Rex Ying, and Leandros Tassiulas. An item is worth a prompt: Versatile image editing with disentangled control. arXiv preprint arXiv:2403.04880, 2024. 21
- [30] Tsu-Jui Fu, Wenze Hu, Xianzhi Du, William Yang Wang, Yinfei Yang, and Zhe Gan. Guiding instruction-based image editing via multimodal large language models. ICLR, 2024. 21, 22, 23, 24
- [31] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. ICLR, 2023. 28
- [32] Jun Gao, Tianchang Shen, Zian Wang, Wenzheng Chen, Kangxue Yin, Daiqing Li, Or Litany, Zan Gojcic, and Sanja Fidler. Get3d: A generative model of high quality 3d textured shapes learned from images. NeurIPS, 2022. 58
- [33] Leon A Gatys, Alexander S Ecker, and Matthias Bethge. Image style transfer using convolutional neural networks. CVPR, 2016. 18
- [34] Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396,

2024. 1

- [35] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020. 1
- [36] Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao, Rui Zhao, Shuning Chang, Weijia Wu, et al. Mix-of-show: Decentralized low-rank adaptation for multi-concept customization of diffusion models. In NeurIPS, 2024. 28
- [37] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS, 2017. 1
- [38] Qibin Hou, Yuying Ge, Jing Zhang, Yuchao Dai, and Ming-Ming Cheng. Storydiffusion: Consistent self-attention for long-range image and video generation. In Advances in Neural Information Processing Systems (NeurIPS), 2024. 31, 32
- [39] Qiming Hu, Hainuo Wang, and Xiaojie Guo. Single image reflection separation via dual-stream interactive transformers. Advances in Neural Information Processing Systems, 37:55228–55248, 2024. 45
- [40] Jiancheng Huang, Yi Huang, Jianzhuang Liu, Donghao Zhou, Yifan Liu, and Shifeng Chen. Dual-schedule inversion: Training-and tuning-free inversion for real image editing. arXiv preprint arXiv:2412.11152, 2024. 21
- [41] Kaiyi Huang, Chengqi Duan, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench++: An enhanced and comprehensive benchmark for compositional text-to-image generation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 5
- [42] Lianghua Huang, Wei Wang, Zhi-Fan Wu, Yupeng Shi, Huanzhang Dou, Chen Liang, Yutong Feng, Yu Liu, and Jingren Zhou. In-context lora for diffusion transformers. arXiv preprint arXiv:2410.23775, 2024. 56
- [43] Xun Huang and Serge Belongie. Arbitrary style transfer in real-time with adaptive instance normalization. In ICCV,

2017. 18

- [44] Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. Layoutlmv3: Pre-training for document ai with unified text and image masking. In ACM MM, 2022. 77
- [45] Zixuan Huang, Stefan Stojanov, Anh Thai, Varun Jampani, and James M Rehg. Planes vs. chairs: Category-guided 3d shape learning without any 3d cues. In ECCV, 2022. 58
- [46] Jiaxiu Jiang, Yabo Zhang, Kailai Feng, Xiaohe Wu, Wenbo Li, Renjing Pei, Fan Li, and Wangmeng Zuo. Mc2: Multi-concept guidance for customized multi-concept generation. arXiv preprint arXiv:2404.05268, 2024. 28
- [47] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In ECCV, 2016. 18
- [48] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 76
- [49] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, 2023. 56
- [50] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In CVPR, 2023. 28
- [51] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024. 2, 5, 8, 9, 10, 11, 47, 48, 49

- [52] Bolin Lai, Felix Juefei-Xu, Miao Liu, Xiaoliang Dai, Nikhil Mehta, Chenguang Zhu, Zeyi Huang, James M Rehg, Sangmin Lee, Ning Zhang, et al. Unleashing in-context learning of autoregressive models for few-shot image manipulation. arXiv preprint arXiv:2412.01027, 2024. 56
- [53] Jiachen Li, Jitesh Jain, and Humphrey Shi. Matting anything. arXiv: 2306.05399, 2023. 66
- [54] Jiachen Li, Jitesh Jain, and Humphrey Shi. Matting anything. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1775–1785, 2024. 68
- [55] Junyi Li, Zhilu Zhang, Xiaoyu Liu, Chaoyu Feng, Xiaotao Wang, Lei Lei, and Wangmeng Zuo. Spatially adaptive self-supervised learning for real-world image denoising. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2023. 34
- [56] Yachuan Li, Xavier Soria Poma, Yun Bai, Qian Xiao, Chaozhi Yang, Guanlin Li, and Zongmin Li. Edmb: Edge detector with mamba. arXiv preprint arXiv:2501.04846, 2025. 66, 67
- [57] Yijun Li, Chen Fang, Jimei Yang, Zhaowen Wang, Xin Lu, and Ming-Hsuan Yang. Universal style transfer via feature transforms. In NIPS, 2017. 18
- [58] Zijie Li, Henry Li, Yichun Shi, Amir Barati Farimani, Yuval Kluger, Linjie Yang, and Peng Wang. Dual diffusion for unified image generation and understanding. arXiv preprint arXiv:2501.00289, 2024. 2
- [59] Zhexin Liang, Zhaochen Li, Shangchen Zhou, Chongyi Li, and Chen Change Loy. Control color: Multimodal diffusionbased interactive image colorization. arXiv preprint arXiv:2402.10855, 2024. 34, 43
- [60] Xin Lin, Chao Ren, Kelvin CK Chan, Lu Qi, Jinshan Pan, and Ming-Hsuan Yang. Multi-task image restoration guided by robust dino features. arXiv preprint arXiv:2312.01677, 2023. 34
- [61] Xin Lin, Chao Ren, and Xiao Liu. Unsupervised image denoising in real-world scenarios via self-collaboration parallel generative adversarial branches. In ICCV, 2023. 34
- [62] Xin Lin, Jingtong Yue, Sixian Ding, Chao Ren, Lu Qi, and Ming-Hsuan Yang. Dual degradation representation for joint deraining and low-light enhancement in the dark. IEEE Transactions on Circuits and Systems for Video Technology,

2024. 34

- [63] Xin Lin, Yuyan Zhou, Jingtong Yue, Chao Ren, Kelvin CK Chan, Lu Qi, and Ming-Hsuan Yang. Re-boosting selfcollaboration parallel prompt gan for unsupervised image restoration. arXiv preprint arXiv:2408.09241, 2024. 34
- [64] Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Chase Lambert, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving text-to-image alignment with deep-fusion large language models. arXiv preprint arXiv:2409.10695, 2024. 10, 12, 14
- [65] Dongyang Liu, Shitian Zhao, Le Zhuo, Weifeng Lin, Yu Qiao, Hongsheng Li, and Peng Gao. Lumina-mgpt: Illuminate flexible photorealistic text-to-image generation with multimodal generative pretraining, 2024. 1
- [66] Haipeng Liu, Yang Wang, Biao Qian, Meng Wang, and Yong Rui. Structure matters: Tackling the semantic discrepancy in diffusion models for image inpainting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 34, 42
- [67] Hao Liu, Wilson Yan, Matei Zaharia, and Pieter Abbeel. World model on million-length video and language with ringattention. arXiv preprint arXiv:2402.08268, 2024. 1
- [68] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Mukund Varma T, Zexiang Xu, and Hao Su. One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. Advances in Neural Information Processing Systems,

2023. 58

- [69] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF international conference on computer vision, 2023. 58
- [70] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. Wonder3d: Single image to 3d using cross-domain diffusion. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024. 58
- [71] Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834, 2023. 2
- [72] Yiyang Ma, Xingchao Liu, Xiaokang Chen, Wen Liu, Chengyue Wu, Zhiyu Wu, Zizheng Pan, Zhenda Xie, Haowei Zhang, Liang Zhao, et al. Janusflow: Harmonizing autoregression and rectified flow for unified multimodal understanding and generation. arXiv preprint arXiv:2411.07975, 2024. 2
- [73] Chenlin Meng, Kristy Choi, Jiaming Song, and Stefano Ermon. Concrete score matching: Generalized score matching for discrete data. Advances in Neural Information Processing Systems, 35:34532–34545, 2022. 2
- [74] Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. Occupancy networks: Learning 3d reconstruction in function space. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019. 58
- [75] Midjourney. Midjourney. https://www.midjourney.com, 2024. 2, 6, 7, 18, 19, 20, 59, 60, 61

- [76] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 2021. 58
- [77] Michael Niemeyer, Lars Mescheder, Michael Oechsle, and Andreas Geiger. Differentiable volumetric rendering: Learning implicit 3d representations without 3d supervision. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020. 58
- [78] OpenAI. Addendum to gpt-4o system card: 4o image generation, 2025. Accessed: 2025-04-02. 2
- [79] Junyi Pan, Xiaoguang Han, Weikai Chen, Jiapeng Tang, and Kui Jia. Deep mesh reconstruction from single rgb images via topology modification networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision,

2019. 58

- [80] Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani, Timo Bolkart, Ahmed AA Osman, Dimitrios Tzionas, and Michael J Black. Expressive body capture: 3d hands, face, and body from a single image. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019. 58
- [81] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952,

2023. 5

- [82] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In The Twelfth International Conference on Learning Representations (ICLR), 2024. 1, 47, 50
- [83] Guocheng Qian, Jinjie Mai, Abdullah Hamdi, Jian Ren, Aliaksandr Siarohin, Bing Li, Hsin-Ying Lee, Ivan Skorokhodov, Peter Wonka, Sergey Tulyakov, et al. Magic123: One image to high-quality 3d object generation using both 2d and 3d diffusion priors. arXiv preprint arXiv:2306.17843, 2023. 58
- [84] Chu-Jie Qin, Rui-Qi Wu, Zikun Liu, Xin Lin, Chun-Le Guo, Hyun Hee Park, and Chongyi Li. Restore anything with masks: Leveraging mask image modeling for blind all-in-one image restoration. In ECCV, 2024. 34
- [85] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022. 5
- [86] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. SAM 2: Segment anything in images and videos. ICLR, 2025. 80, 81, 82, 83, 84
- [87] Jeremy Reizenstein, Roman Shapovalov, Philipp Henzler, Luca Sbordone, Patrick Labatut, and David Novotny. Common objects in 3d: Large-scale learning and evaluation of real-life 3d category reconstruction. In ICCV, 2021. 58
- [88] Bin Ren, Yawei Li, Nancy Mehta, and Radu Timofte. The ninth ntire 2024 efficient super-resolution challenge report. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2024. 34
- [89] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (CVPR), pages 10684–10695, 2022. 1
- [90] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, June 2022. 47, 52
- [91] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, 2023. 28
- [92] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 2022. 5
- [93] Subham Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems, 37:130136–130184, 2024. 2
- [94] Qingyu Shi, Lu Qi, Jianzong Wu, Jinbin Bai, Jingbo Wang, Yunhai Tong, Xiangtai Li, and Ming-Husan Yang. Relationbooth: Towards relation-aware customized object generation. arXiv preprint arXiv:2410.23280, 2024. 28
- [95] Haoze Sun, Wenbo Li, Jianzhuang Liu, Haoyu Chen, Renjing Pei, Xueyi Zou, Youliang Yan, and Yujiu Yang. Coser: Bridging image and language for cognitive super-resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 25868–25878, 2024. 34
- [96] Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative pretraining in multimodality. arXiv preprint arXiv:2307.05222, 2023. 1
- [97] Alexander Swerdlow, Mihir Prabhudesai, Siddharth Gandhi, Deepak Pathak, and Katerina Fragkiadaki. Unified multimodal discrete diffusion. arXiv preprint arXiv:2503.20853, 2025. 2
- [98] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. 1

- [99] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 2, 3, 5, 6, 7, 8, 9, 10, 12, 14, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 59, 60, 61, 63, 64, 65, 67, 68, 70, 71, 72, 73, 75, 76, 77, 78, 79, 81, 82, 83, 84
- [100] Shengbang Tong, David Fan, Jiachen Zhu, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, and Zhuang Liu. Metamorph: Multimodal understanding and generation via instruction tuning. arXiv preprint arXiv:2412.14164, 2024. 1
- [101] Chunwei Wang, Guansong Lu, Junwei Yang, Runhui Huang, Jianhua Han, Lu Hou, Wei Zhang, and Hang Xu. Illume: Illuminating your llms to see, draw, and self-enhance. arXiv preprint arXiv:2412.06673, 2024. 1
- [102] Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, and Yu-Gang Jiang. Pixel2mesh: Generating 3d mesh models from single rgb images. In Proceedings of the European conference on computer vision (ECCV), 2018. 58
- [103] Xierui Wang, Siming Fu, Qihan Huang, Wanggui He, and Hao Jiang. Ms-diffusion: Multi-subject zero-shot image personalization with layout guidance. arXiv preprint arXiv:2406.07209, 2024. 28, 30
- [104] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024. 1
- [105] Zhendong Wang, Yifan Jiang, Yadong Lu, Pengcheng He, Weizhu Chen, Zhangyang Wang, Mingyuan Zhou, et al. In-context learning unlocked for diffusion models. NeurIPS, 2023. 56
- [106] Zhenyu Wang, Aoxue Li, Zhenguo Li, and Xihui Liu. Genartist: Multimodal llm as an agent for unified image generation and editing. NeurIPS, 2024. 5
- [107] Alex Warren, Ke Xu, Jiaying Lin, Gary KL Tam, and Rynson WH Lau. Effective video mirror detection with inconsistent motion cues. In CVPR, 2024. 69, 71
- [108] Jianzong Wu, Chao Tang, Jingbo Wang, Yanhong Zeng, Xiangtai Li, and Yunhai Tong. Diffsensei: Bridging multi-modal llms and diffusion models for customized manga generation. CVPR, 2025. 31, 33
- [109] Size Wu, Wenwei Zhang, Lumin Xu, Sheng Jin, Zhonghua Wu, Qingyi Tao, Wentao Liu, Wei Li, and Chen Change Loy. Harmonizing visual representations for unified multimodal understanding and generation. arXiv preprint arXiv:2503.21979, 2025. 1
- [110] Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024. 1
- [111] Yifan Xia, Yuying Ge, Jing Zhang, Yuchao Dai, and Ming-Ming Cheng. Seed-story: Multimodal long story generation with large language model. arXiv preprint arXiv:2407.08683, 2024. 31, 32
- [112] Shitao Xiao, Yueze Wang, Junjie Zhou, Huaying Yuan, Xingrun Xing, Ruiran Yan, Shuting Wang, Tiejun Huang, and Zheng Liu. Omnigen: Unified image generation. arXiv preprint arXiv:2409.11340, 2024. 2
- [113] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. Instantmesh: Efficient 3d mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191, 2024. 58
- [114] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10371–10381, 2024. 74, 75
- [115] Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and Bin Cui. Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms. In ICML, 2024. 5
- [116] Hang Yu, Ruilin Li, Shaorong Xie, and Jiayan Qiu. Shadow-enlightened image outpainting. In CVPR, 2024. 34, 42
- [117] Haobo Yuan, Xiangtai Li, Tao Zhang, Zilong Huang, Shilin Xu, Shunping Ji, Yunhai Tong, Lu Qi, Jiashi Feng, and Ming-Hsuan Yang. Sa2va: Marrying sam2 with llava for dense grounded understanding of images and videos. arXiv,

2025. 62, 63

- [118] Yu Yuan, Xijun Wang, Yichen Sheng, Prateek Chennuri, Xingguang Zhang, and Stanley Chan. Generative photography: Scene-consistent camera control for realistic text-to-image synthesis. arXiv preprint arXiv:2412.02168, 2024. 53, 54, 55
- [119] Cheng Zhang, Qianyi Wu, Camilo Cruz Gambardella, Xiaoshui Huang, Dinh Phung, Wanli Ouyang, and Jianfei Cai. Taming stable diffusion for text to 360◦ panorama image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 17
- [120] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instructionguided image editing. In NeurIPS, 2023. 21, 25, 26, 27
- [121] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023. 47

- [122] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Scaling in-the-wild training for diffusion-based illumination harmonization and editing by imposing consistent light transport. In ICLR, 2025. 34, 46
- [123] Wenwei Zhang, Jiangmiao Pang, Kai Chen, and Chen Change Loy. K-net: Towards unified image segmentation. Advances in Neural Information Processing Systems, 34:10326–10338, 2021. 62, 65
- [124] Xinchen Zhang, Ling Yang, Guohao Li, Yaqi Cai, Jiake Xie, Yong Tang, Yujiu Yang, Mengdi Wang, and Bin Cui. Itercomp: Iterative composition-aware feedback learning from model gallery for text-to-image generation. arXiv preprint arXiv:2410.07171, 2024. 5
- [125] Yuxuan Zhang, Yiren Song, Jiaming Liu, Rui Wang, Jinpeng Yu, Hao Tang, Huaxia Li, Xu Tang, Yao Hu, Han Pan, et al. Ssr-encoder: Encoding selective subject representation for subject-driven generation. In CVPR, 2024. 28
- [126] Chuyang Zhao, Yuxing Song, Wenhao Wang, Haocheng Feng, Errui Ding, Yifan Sun, Xinyan Xiao, and Jingdong Wang. Monoformer: One transformer for both diffusion and autoregression. arXiv preprint arXiv:2409.16280, 2024. 2
- [127] Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma Laaksonen, Wanli Ouyang, and Nicu Sebe. Bilateral reference for high-resolution dichotomous image segmentation. CCAI, 2024. 69, 70, 73
- [128] Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024. 2
- [129] Donghao Zhou, Jiancheng Huang, Jinbin Bai, Jiaze Wang, Hao Chen, Guangyong Chen, Xiaowei Hu, and PhengAnn Heng. MagicTailor: Component-controllable personalization in text-to-image diffusion models. arXiv preprint arXiv:2410.13370, 2024. 28
- [130] Zhiyu Zhu, Yingcong Chen, Zhenyu Xie, and Jingyi Yu. Disenvisioner: Disentangled and enriched visual prompt for customized image generation. arXiv preprint arXiv:2410.02067, 2024. 28, 29
- [131] Silvia Zuffi, Angjoo Kanazawa, and Michael J Black. Lions and tigers and bears: Capturing non-rigid, 3d, articulated shape from images. In CVPR, 2018. 58

