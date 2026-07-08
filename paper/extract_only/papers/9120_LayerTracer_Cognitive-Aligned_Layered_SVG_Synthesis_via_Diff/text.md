# arXiv:2502.01105v3[cs.CV]13Aug2025

## LayerTracer: Cognitive-Aligned Layered SVG Synthesis via Diffusion Transformer

Yiren Song Danze Chen Mike Zheng Shou* Show Lab, National University of Singapore

[Figure 1]

an icon of a bottle with a lightning symbol

an icon of iced lemon tea

a emoji of female doctor

an icon of a logistics customer service

a castle icon with line outlines

Input image An icon of a delivery truck with a notification bell

Text to layer-wise SVG Genera on Layer-wise Vectoriza on

Figure 1. LayerTracer creates cognitively aligned layered SVGs from text prompts or by converting images into layered SVGs.

#### Abstract

#### 1. Introduction

Generating cognitive-aligned layered SVGs remains challenging due to existing methods’ tendencies toward either oversimplified single-layer outputs or optimizationinduced shape redundancies. We propose LayerTracer, a DiT based framework that bridges this gap by learning designers’ layered SVG creation processes from a novel dataset of sequential design operations. Our approach operates in two phases: First, a text-conditioned DiT generates multi-phase rasterized construction blueprints that simulate human design workflows. Second, layer-wise vectorization with path deduplication produces clean, editable SVGs. For image vectorization, we introduce a conditional diffusion mechanism that encodes reference images into latent tokens, guiding hierarchical reconstruction while preserving structural integrity. Extensive experiments show that LayerTracer surpasses optimization-based and neural baselines in generation quality and editability. Code is released at https://github.com/showlab/LayerTracer.

Scalable Vector Graphics (SVG) is widely used in modern digital design, representing visual elements such as paths, curves, and geometric shapes through mathematical equations rather than pixel grids. Unlike raster images, SVG preserves resolution-independent clarity at any scale, making it suitable for applications requiring high precision, including UI/UX design and industrial CAD systems. Layered SVGs further enhance this flexibility by allowing designers to manipulate individual layers to adjust stroke properties, spatial arrangements, and compositing effects. This structured editability supports dynamic modifications and collaborative workflows in contemporary design practices.

Nevertheless, a significant gap persists between current deep learning-based SVG generation techniques and professional requirements. Existing approaches face three systemic challenges: First, the scarcity of large-scale layered SVG datasets forces models to rely on synthetic or oversimplified training data, resulting in outputs devoid of the nuanced hierarchical structures inherent to

*Corresponding author.

human designs. Second, methodological fragmentation prevails—optimization-based methods [13, 17, 19, 23, 50, 56, 57] generate vector paths using raster priors but often produce cluttered geometries with redundant anchor points; and large language models (LLMs) [29, 44, 54, 55], constrained by token limits, remain limited to basic icons . Most critically, no existing method addresses the designer’s cognitive process—the logical sequencing, spatial reasoning, and element grouping strategies employed during layer construction—resulting in AI-generated SVGs that resemble fragmented collages rather than intentionally editable professional designs.

To address these challenges, we present LayerTracer, a Diffusion Transformer (DiT)-based framework that redefines layered SVG synthesis by modeling designers’ layerby-layer construction logic. Our approach is grounded in three key insights: (1). Cognitive alignment: DiT models pretrained on text-image corpora inherently capture contextual relationships between visual elements, which can be steered through targeted fine-tuning to mimic designer decision-making. (2). Spatiotemporal consistency: The self-attention mechanism’s bias toward local token interactions—a byproduct of training on natural image pixel correlations—can be repurposed to enforce coherence across sequential design steps. 3. Structured decomposition: Disassembling layered SVGs into channel-wise components and organizing them as grid sequences provides generation models with an interpretable blueprint of layer evolution.

In implementation, LayerTracer integrates two innovations. First, we curate a pioneering dataset of 20,000+ designer process traces, automatically converting layered SVGs into timestamped creation sequences. These sequences are rasterized and organized into training grids using a serpentin layout, ensuring temporally adjacent design steps remain spatially proximate. Second, we develop a dual-phase generation pipeline: (1) a text-conditioned DiT generates rasterized construction process sequences that simulate a designer’s workflow, followed by (2) a layerwise vectorization module that converts these sequences into clean, editable SVG layers while eliminating redundant paths.

Beyond text-to-SVG synthesis, LayerTracer tackles the inverse task: converting raster images into layered vector graphics. We reframe this as a process-conditioned generation problem, where reference images guide the model to ”reverse-engineer” plausible layer construction steps. Specifically, we build upon a pretrained DiT model and adapt it through LoRA fine-tuning to ingest image context. By encoding reference images into conditional tokens injected into the denoising process, the model autonomously deduces layer assembly sequences (e.g., ”background first, then foreground elements”), faithfully reconstructing input images while adhering to practical editing constraints.

Our main contributions are as follows:

- • Cognitive-aligned SVG synthesis: As the first framework to generate layered SVGs by learning designers’ construction logic—element ordering, layer grouping, and spatial reasoning—LayerTracer ensures outputs meet professional editing standards.
- • Unified DiT-based architecture: Our framework seamlessly integrates text-to-SVG generation and layer-wise vectorization tasks, eliminating the need for task-specific pipelines.
- • Process-centric dataset: We release a scalable pipeline for collecting designer workflow data, addressing the critical gap in layered vector graphics training resources. Extensive experiments validate LayerTracer’s state-of-theart performance and effectiveness.

#### 2. Related Works

##### 2.1. Text2image Diffusion Model

Recent studies have demonstrated that diffusion models are capable of generating high-quality synthetic images, effectively balancing diversity and fidelity. Models based on diffusion models or their variants, such as those paper in [25, 30], have successfully addressed the challenges associated with text-conditioned image synthesis. Stable Diffusion [30], a model based on the Latent Diffusion Model, incorporates text conditioning within a UNet framework to facilitate text-based image generation [38, 39, 41], establishing itself as a mainstream model in image generation. Finetuning pre-trained image generation models can enhance their adaptation to specific application scenarios, as seen in techniques like LoRA [14] and DreamBooth [26]. For theme control in text-to-image generation, several works [4, 48, 58, 61–63, 65] focus on custom generation for defined pictorial concepts, with ControlNet [59] additionally offering control over other modalities such as depth information. Some methods [9, 11, 16, 22, 31, 32, 36, 37, 51, 64] explore controllable generation using Diffusion Transformer-based approaches. AnimateDiff [12] introduces a temporal attention module, extending Stable Diffusion into a video generation model. Inspired by ProcessPainter [35], which first proposed learning an artist’s painting process through pretrained temporal models, this paper leverages the in-context capabilities of DiT to generate a layered SVG creation process.

##### 2.2. SVG Generation

Scalable Vector Graphics (SVGs) are widely utilized in design owing to their advantages like geometric manipulability, resolution independence, and compact file structure. SVG generation often involves training neural networks to produce predefined SVG commands and attributes using architectures such as RNNs [28], VAEs [2, 21, 44], and Trans-

formers [2, 49, 52]. Nonetheless, the absence of large-scale vector datasets constrains their generalization capabilities and the creation of complex graphics, with most datasets focusing on specific areas like monochromatic vector icons [52] and fonts [42, 49].

An alternative to directly training an SVG generation network is optimizing it to match a target image during the evaluation phase, employing differentiable rasterizers to bridge vector graphics and raster images [19]. This method optimizes SVG parameters based on pretrained visionlanguage models. Advances in models like CLIP [27] have facilitated effective SVG generation methods such as CLIPDraw [8], CLIPasso [46], and CLIPVG [40]. while DreamFusion [26] demonstrates the superior generative capabilities of diffusion models. VecFusion [17], DiffSketcher [56], and SVGDreamer combine differentiable rasterizers with text-to-image diffusion models to produce vector graphics, achieving notable results in iconography and sketching. However, these methods still face challenges with editability and graphical quality. Recent studies [45, 60] have blended optimization-based methods with neural networks to enhance vector representations by integrating geometric constraints.

The primary issue with methods that optimize a set of vector primitives through SDS loss is their reliance on image generation model priors, which often leads to redundant and noisy results. These outputs lack clear hierarchical structures and fail to meet design specifications. In this paper, we innovatively propose an alternative approach to utilizing image generation model priors. Specifically, we leverage the in-context learning capability of Diffusion Transformers to generate the creation process of SVG graphics, combined with vectorization to achieve cognitivealigned layered SVG generation.

##### 2.3. Vectorization

Raster image vectorization or image tracing is a well-studied problem in computer graphics[1, 2, 5, 7]. Diffvg[19] proposes a differentiable rendering method for vectorization, which found shape gradients by differentiating the formula of Reynolds transport theorem with MontaCarlo edge sampling. Meanwhile, combining differentiable rendering techniques with deep learning models are also studied for image vectorization[23, 34, 46]. Direct rasterto-vector conversion with neural networks are supported for the relatively simple images[2, 21, 28]. Stroke-based rendering can be used to fit a complex image with a sequence of vector strokes [15, 20, 33], but the performance is limited by the predefined strokes. Diffvg[19] can also be leveraged to fit an input image with a set of randomly initialized vector graphical elements. Based on Diffvg, LIVE [23] proposes a coarse-to-fine vectorization strategy, with cost tens of minute. CLIPVG[40] proposes a multi-round vectorization

strategy, providing additional graphic elements for the image manipulation task. LIVE [23] and O&R [13] achieve hierarchical vectorization through optimization-based methods, but their results show a significant gap compared to human-designed works, lacking logical coherence. In contrast to these approaches, our proposed LayerTracer leverages the prior knowledge of the Diffusion Transformer model, reformulating the hierarchical vectorization task as a problem of predicting preceding frames from a reference image.

#### 3. Method

In this section, we begin by exploring the preliminaries on diffusion transformer as detailed in section 3.1. Then introduce the overall architecture of our method in section 3.2, followed by detailed descriptions of the key modules: dataset construction methods in 3.3, Layer-wise image generation in section 3.4, Image2Layers in section 3.5, and Layer-Wise Vectorization in section 3.6.

##### 3.1. Preliminary

The Diffusion Transformer (DiT) model [25], which appears in frameworks such as FLUX.1 [18], Stable Diffusion 3 [30], and PixArt [3], employs a transformer-based denoising network to iteratively refine noisy image tokens.

DiT processes two categories of tokens: noisy image tokens X ∈ RN×d and text condition tokens CT ∈ RM×d, where d is the embedding dimension, and N and M respectively represent the numbers of image and text tokens. As these tokens move through the transformer blocks, they retain consistent dimensions.

In FLUX.1, each DiT block applies layer normalization before Multi-Modal Attention (MMA) [24], incorporating Rotary Position Embedding (RoPE) [43] to capture spatial context. For image tokens X, RoPE applies rotation matrices based on a token’s position (i,j) in the 2D grid:

Xi,j → Xi,j · R(i,j), (1)

where R(i,j) is the rotation matrix at position (i,j). Text tokens CT are similarly transformed with their positions specified as (0,0).

The multi-modal attention mechanism then projects these position-encoded tokens into query Q, key K, and value V representations, enabling attention across all tokens:

MMA([X;CT]) = softmax

QK⊤ √

d

V, (2)

where [X;CT] denotes concatenation of image and text tokens. This formulation ensures bidirectional attention among the tokens.

|[Figure 2]<br><br>[Figure 3]<br><br>1⃣ 2⃣ 3⃣<br><br>[Figure 4]<br><br>[Figure 5]<br><br>[Figure 6]<br><br>6⃣ 5⃣ 4⃣<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]<br><br>7⃣ 8⃣ 9⃣<br><br><br>[Figure 10]<br><br>[Figure 11]<br><br>|
|---|

[Figure 12]

[Figure 13]

“This image illustrates the nine-step creation process of an SVG icon illustration, arranged in a 3*3 grid. The top left tile is a line art sketch of a icon, depicting its basic shape and structure. Subsequent tiles gradually add color and detail to the illustration…”

[Figure 14]

[Figure 15]

[Figure 16]

Feed Forward

…

…

QKVProj.❄ LoRA🔥 DiTBlock

[Figure 17]

[Figure 18]

Diﬀerencing, Filtering, Vectoriza on

DiT Block

…

…

|[Figure 19]<br><br>[Figure 20]<br><br>1⃣ 2⃣<br><br>[Figure 21]<br><br>[Figure 22]<br><br>4⃣ 3⃣<br><br>[Figure 23]<br><br>|
|---|

Self-A en on

DiT Block

DiT Block

“The image is a 2x2 grid of folder icons. The top-left shows a folder with a plain document, while the topright depicts a folder containing a document with visible text lines…”

…

…

Text Tokens Latent Tokens (noised） Condi on Tokens (clean)

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

layered pixel images Layer-wise SVG

Serpen n Dataset Construc on

Layer-wise Model Training Image2Layers Model Training Layer-Wise Vectoriza on

Figure 2. The LayerTracer architecture comprises three key components: (1) Layer-wise Model: Pretrained on our proposed dataset to generate layered pixel sequences from text prompt; (2) Image2Layers Model: Merges LoRA with the Flux base DiT, enabling imageconditioned generation through VAE-encoded latent tokens; (3) Layer-wise Vectorization: Converts raster sequences to SVGs via differential analysis between adjacent layers, followed by B´ezier optimization using vtracer to eliminate redundant paths while preserving structural fidelity.

##### 3.2. Overall Architecture.

strong correlations between adjacent image pixels captured during the pre-training of diffusion models. To enhance the model’s learning of grid sequences [47], we introduce the serpentine dataset construction. As shown in Figure 2, we arrange the sequences of 9 and 4 frames in a serpentine layout within the grid, ensuring that temporally adjacent frames are also spatially adjacent (either horizontally or vertically). In our ablation experiments, we confirmed that this design is crucial for the coherence of sequence generation. To aid hierarchical vectorization, icons with black strokes isolate the line layer in the first frame during dataset creation. During the generation phase, we vectorize the black line layer and overlay it onto the subsequent results.

LayerTracer consists of the following components: Serpentine dataset construction, Layer-wise model training, Image2Layers model, and the layer-wise vectorization. Initially, we collected the processes designers use to create layered vector graphics and organized them into grid-based datasets. Following this, we utilized the LoRA method for pre-training on the proposed dataset, thereby enabling the generation of layered pixel images from textual descriptions. Subsequently, we integrated the LoRA from the previous step with the Flux base model to establish a new foundational model. The Image2Layers model introduces an image-based conditional mechanism that, through additional LoRA fine-tuning, predicts the creation process of reference images. Finally, in the layer-wise vectorization stage, the model sequentially transforms the generated pixel images into high-quality vector graphics, which are analyzed, filtered, and vectorized based on the differences between adjacent layers.

##### 3.4. Layer-wise Image Generation

DiT models, trained on massive image-text pairs, inherently possess contextual generation capabilities. By appropriately activating and enhancing this ability, they can be utilized for complex generation tasks. Since text-to-image models can interpret merged prompts, they can be reused for in-context generation without altering their architecture. This only requires changes to the input data rather than modifications to the model itself. Building on this insight, we designed a simple yet effective pipeline to learn the hierarchical logic employed by human designers in creating layered SVGs.

##### 3.3. Serpentine Dataset Construction

Our dataset construction includes 20,000 layered SVGs created by designers, encompassing black outline icons, regular icons, emojis, and illustrative graphics. As shown in Fig. 2, each sequence is composed of either 9 or 4 frames, arranged in 3x3 or 2x2 grids, resulting in resolutions of 1056x1056 and 1024x1024 respectively. To capture the process of designers creating layered SVGs, we propose a automated data generation pipeline that deconstructs the layered SVG graphics into sequences based on the grouping logic and element hierarchy within the SVG files. Additionally, the pipeline incorporates a human-in-the-loop process to filter out nonsensical sequences.

Layer-wise Model Training. Due to the size of the dataset, we adopt LoRA fine-tuning for training which can be formulated as:

W = W0 + ∆W, (3) where W0 represents the original weights of the pretrained model, and ∆W denotes the low-rank adaptation updates introduced during fine-tuning. This formulation enables efficient training by keeping W0 fixed and applying lightweight updates through ∆W, which allows the model

In the attention mechanism of DiT, tokens tend to focus on spatially adjacent tokens. This tendency stems from the

to balance generalization from the pre-trained weights with task-specific adaptation provided by the fine-tuned updates. Loss function. We employ the conditional flow matching loss function, integral to training and optimizing the generative model, is defined as follows:

t(X|ϵ), p(ϵ) vΘ(X,t) − ut(X|ϵ) 2 (4)

LCFM = Et, p

Where vΘ(X,t) represents the velocity field parameterized by neural network weights, t denotes the timestep, and ut(X|ϵ) is the conditional vector field mapping the path between noise and true data distributions.

##### 3.5. Image2Layers Model

In this section, we introduce the Image2Layers model, which builds upon the previous section by incorporating image conditioning. This approach redefines hierarchical vectorization as a ”reverse engineering” task, predicting how an SVG is created layer by layer.

The primary challenge in training the Image2Layers model lies in the limited availability of high-quality sequential data. While a small dataset may suffice for LoRA training, initializing a controllable plugin (such as ControlNet [59] or IP-Adapter [58]) from scratch with limited data is highly challenging. To address this, we design an efficient controllability framework by repurposing a pre-trained DiT model and adapting it to accept image context as a conditioning input. Specifically:

Training Phase. We concatenate procedural sequences into 2×2 or 3×3 training grids. The final frame (context image) is passed through the VAE to extract latent variables, which are directly appended to the end of the denoising latent. Through self-attention mechanisms, the context latent provides conditional information to the denoising processes of other frames, enhancing the logical consistency and coherence of the generation. The condition image is then fed into a VAE to obtain its latent representation, which is directly appended to the denoising latent at the end. Multi-modal attention mechanisms are used to provide conditional information for the denoising of other frames.

QK⊤ √

V, (5)

MMA([X;CI;CT]) = softmax

d

where [X;CI;CT] denotes the concatenation of image and text tokens. This formulation enables bidirectional attention.

Inference Phase. During inference, we use the reference image as a condition to predict the earlier layers, thereby inferring how the SVG in the reference image was constructed layer by layer.

##### 3.6. Layer-Wise Vectorization

To achieve hierarchical vectorization of input images, our process begins by segmenting grid images into individual cells. This facilitates independent processing for sub-

sequent vectorization stages. Specifically for icons, accurate extraction of black lines is crucial as they represent key structural elements of the image. To address common issues of line distortion, we employ a series of preprocessing steps: grayscale conversion highlights contrast between lines and background; Gaussian blurring smooths out noise; and adaptive thresholding via the Otsu’s thresholding method ensures robust line separation. These preprocessed lines are then integrated into a transparent PNG, focusing vectorization efforts on relevant areas.

Furthermore, to capture the layered details of images, we perform differential extraction between adjacent cells, identifying significant pixel changes to highlight areas of variation. This involves converting cell images to grayscale, computing absolute differences, applying binary thresholding, and refining the output with morphological operations to produce clean, meaningful contours of change. These contours are saved as transparent PNGs for subsequent vectorization.

The final step involves vectorizing these differential layers using tools like vtracer, optimizing parameters to balance detail retention and file size, and ultimately merging all vectorized layers into a single SVG file. This method preserves the image’s global structure while highlighting intricate changes between cells, resulting in a layered and editable SVG suitable for detailed graphical representations.

#### 4. Experiment

##### 4.1. Experiment Setting

Experiment Details. During the pretraining stage, we utilized the Flux 1.0 dev model based on the pretrained DiT architecture. Training resolutions included 1056×1056 (3×3 grids) and 1024×1024 (2×2 grids). The LoRA finetuning approach was applied with a LoRA rank of 256, a batch size of 16, a learning rate of 0.001, and 20,000 finetuning steps. For the three styles of SVGs, we trained separate LoRA models for black outline icons, illustrations, and emojis, using 3K, 2K, and 15K training samples, respectively. During the Image2Layers Model training phase, we merged the LayerTracer LoRA with the base model, using a LoRA merge weight of 1.0. Then, we fine-tuned it for 20,000 steps using LoRA-based training on the same dataset as the Layer-wise Model training. All training was conducted on a single A100 GPU with 80GB of memory.

Baseline Methods. The baseline methods in text-tosvg genration methods are SVGDreamer [57], Vecfusion [17]and DiffSketcher [56]. The baseline methods in vectorization include diffvg [19], LIVE [23], and O&R [13]. All baseline methods are evaluated with their default settings to ensure fairness.

Benchmarks. To address the lack of high-quality layered SVG datasets, this paper introduces a dataset containing

[Figure 28]

[Figure 29]

[Figure 30]

… a yellow face with a frown, a blue tear rolling down its cheek, sadness … … an emoji, a blonde woman with short hair and a black blazer, pilot's hat…

… illustration, a woman with silver hair sitting in a cozy corner of a library …

… a four-step illustration, a man standing under a road sign checking his phone …

… a nine-step SVG creation of a modern office building icon in blue shades... The icons progress from a simple outline to more detailed versions with glass windows …

… a set of nine minimalistic icons of a delivery truck with a white arrow to complex arrangements of orange boxes with directional arrows…

Figure 3. Given a text prompt, LayerTracer generates cognitive-aligned layered SVGs that mimic human design cognition.

[Figure 31]

… a set of nine minimalistic icons showing a woman with various expressions，with long dark green hair and a yellow blouse, displays cheerful gestures…

###### Input image Layer-wise vectoriza on results

Figure 4. Given a raster image of an icon as input, LayerTracer predicts how the icon was created layer by layer, achieving cognitivealigned layered vectorization.

##### 4.2. Generation Results

over 20,000 layered SVGs and their creation processes, named the LayerSVG Dataset. To ensure fairness in comparative experiments, the Noto-Emoji [10] dataset is also included in the benchmark for quantitative evaluation. For the text-to-SVG task and the layer-wise vectorization task, we select 50 prompts and 50 images, respectively, as benchmarks for testing.

Fig. 3 demonstrates LayerTracer’s capability to generate cognitively-aligned layered SVGs that adhere to text descriptions while maintaining logical layer hierarchies (e.g., background-to-foreground ordering and grouped semantic elements). The outputs preserve essential design properties including layer independence, non-overlapping paths,

[Figure 32]

ing with specific requirements for icons and emojis. Baseline methods, in contrast, result in visual clutter and irregular paths. Figure 7 demonstrates our method’s superior performance in layer-wise vectorization tasks, maintaining logical spatial hierarchies and semantic grouping, unlike baselines which yield fragmented and misaligned outputs.

- Table 1. Comparison with SOTA SVG Generation Methods. The best results are denoted as Bold.

Methods CLIPScore↑

Time Cost(s)↓

No. Paths↓

layerwise

Vecfusion [17] 31.10 4668 128.00 False SVGDreamer [57] 32.68 5715 512.00 False DiffSketcher [56] 31.47 3374 512.00 False Ours 33.76 27 35.39 True

- Table 2. Comparison with SOTA Vectorization Methods. The best results are denoted as Bold.

Figure 5. Layer-wise SVG generation with color gradients.

and topological editability. Fig. 4 further illustrates layeraware vectorization results, where input raster images are decomposed into clean vector layers with consistent spatial alignment and minimal shape redundancy. Fig. 5 shows that training with gradient-colored samples and advanced vectorization methods [6] enables LayerTracer to generate layered vector graphics with color gradients. More experimental results can be found in the supplementary material.

Methods MSE↓ Time Cost(s)↓

No. Paths↓

Layerwise

Diffvg [19] 2.02 × 10−4 393 256.00 False LIVE [23] 5.21 × 10−4 3147 46.00 True O&R [13] 2.01 × 10−4 612 64.00 True Ours 1.96 × 10−4 34 29.98 True

##### 4.3. Comparison and Evaluation

In the text-to-SVG task, we compute FID and CLIP Score. For the hierarchical vectorization task, we follow the evaluation methodology from previous works. We calculate MSE to assess the consistency between the reconstructed image and the input image. Additionally, we record the number of SVG shapes used, as fewer shapes indicate a more concise and efficient result. Table 1 and 2 show that LayerTracer achieves the best results across most metrics.

Quantitative Evaluation. Table 1 and 2 present the quantitative evaluation results. In the SVG generation task, our method achieves the highest CLIP-Score with the lowest average number of paths and shortest time cost. Notably, baseline methods fail to produce rationally layered outputs. For the layer-wise vectorization task, our approach outperforms all baselines across metrics: the lowest average path count demonstrates superior simplicity and efficiency, while faster runtime and higher reconstruction consistency further validate its effectiveness.

[Figure 33]

An icon of a pink jerry can with a small green leaf symbol.

Table 3. Quantitative Evaluation of Serpentine Layout Strategy.

An icon of sashimi sushi.

Methods MSE train↓ SSIM train↑MSE test↓ SSIM test↑

w/o Serpentine Layout 2.03 × 10−4 0.964 2.41 × 10−4 0.959 Full 1.65 × 10−4 0.971 1.96 × 10−4 0.963

Illustrator，a women stand beside a window, holding a cafe

##### 4.4. Ablation Study of Serpentine Layout Strategy.

In this section, we conduct an ablation study on the serpentine layout strategy. As shown in Fig. 8, when the serpentine layout strategy is not used to construct the training dataset, incomplete decomposition, undesirable repetitions, and abrupt changes between frames are more likely to occur. The quantitative evaluation results are presented in Table 3. For the layer-wise vectorization task, we calculate the MSE between the predicted results for 9 frames and the ground truth on both the training and test sets. When the serpentine layout strategy is not applied, the MSE is higher.

An icon of man in a suit holding a gavel.

Prompt SVGDreamer Vecfusion DiﬀSketcher Ours

- Figure 6. Compare with baseline methods in Text-to-SVG generation task.

Qualitative Evaluation. This section presents qualitative analysis results. Figure 6 shows that our method produces concise and coherent outputs for text-to-SVG tasks, align-

[Figure 34]

Input image LIVE O & R Ours

- Figure 7. Compare with baseline methods in layer-wise vectorization task. Our results are more concise and exhibit more logical layering.

[Figure 35]

logically layered sequences. As Fig. 9 shows, LayerTracer outperformed all baselines in user preference, prompt adherence, and layer rationality.

Ground Truth

[Figure 36]

Full

#### 5. Limitations and Future Work

|Unwanted repeat|
|---|

| |
|---|

[Figure 37]

w/o Serpen ne Layout Strategy

During layer-wise vectorization, LayerTracer relies on methods like Vtracer, inheriting its limitations such as need for manual adjustment of hyperparameters. Its performance also falters on out-of-distribution data and complex images. We aim to develop a smarter single-layer vectorization solution to replace Vtracer in the future.

No layering

Figure 8. Ablation study of Serpentine Layout Strategy.

This indicates that the Serpentine Layout Strategy benefits the model in learning a consistent layering process.

#### 6. Conclusion

###### (a) User Perference

100

In this work, we introduced LayerTracer, a novel framework that bridges the gap between automated SVG generation and professional design standards. Leveraging the strengths of Diffusion Transformers, LayerTracer achieves cognitive-aligned, layer-wise SVG generation and vectorization. By learning the workflows and design logic of human designers, LayerTracer effectively generates clean, editable, and semantically meaningful vector graphics from textual descriptions or raster images. To overcome the scarcity of layered SVG creation data, we established a pipeline that collects over 20,000 SVG creation sequences. We proposed Serpentin dataset construction method, enabling effective model training. Extensive experiments demonstrate that LayerTracer not only excels in SVG generation quality but also offers unparalleled flexibility and interpretability, setting a new benchmark for scalable vector graphics creation.

75

50

25

0

SVGDreamer Vecfusion Diﬀsketcher LIVE O&R Diﬀvg

###### (b) Text-Image Alignment

###### (c) Ra onality of layering

100

100

75

75

50

50

25

25

0

0

Vecfusion SVGDreamer Diﬀsketcher

LIVE O&R Diﬀvg

Other is better Comparable Ours is better

Figure 9. User Study Results. LayerTracer outperform in all three metrics.

##### 4.5. User Study

We conducted a user study with 46 design enthusiasts using a digital questionnaire, presenting results from both our method and baseline methods. Participants selected their preferred results and those best matching the prompt descriptions in the text-to-SVG task. In the layer-wise vectorization task, they chose their preferred results and the most

#### Acknowledgement

This research is supported by the National Research Foundation, Singapore under its AI Singapore Programme (AISG Award No: AISG3-RP-2022-030).

#### References

- [1] Jules Bloomenthal and Ken Shoemake. 1991. Convolution surfaces. In Proceedings of the 18th annual conference on Computer graphics and interactive techniques. 251–256. 3

- [2] Alexandre Carlier, Martin Danelljan, Alexandre Alahi, and Radu Timofte. 2020. Deepsvg: A hierarchical generative network for vector graphics animation. Advances in Neural Information Processing Systems 33 (2020), 16351–16361. 2, 3

- [3] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. 2023. Pixartα: Fast training of diffusion transformer for photorealistic text-to-image synthesis. arXiv preprint arXiv:2310.00426 (2023). 3

- [4] Xuewei Chen, Zhimin Chen, and Yiren Song. 2025. Transanimate: Taming layer diffusion to generate rgba video. arXiv preprint arXiv:2503.17934 (2025). 2

- [5] Robert L Cook. 1986. Stochastic sampling in computer graphics. ACM Transactions on Graphics (TOG) 5, 1 (1986), 51–72. 3

- [6] Zheng-Jun Du, Liang-Fu Kang, Jianchao Tan, Yotam Gingold, and Kun Xu. 2023. Image vectorization and editing via linear gradient layer decomposition. ACM Transactions on Graphics (TOG) 42, 4 (2023), 1–13. 7

- [7] Vage Egiazarian, Oleg Voynov, Alexey Artemov, Denis Volkhonskiy, Aleksandr Safin, Maria Taktasheva, Denis Zorin, and Evgeny Burnaev. 2020. Deep vectorization of technical drawings. In European conference on computer vision. Springer, 582–598. 3

- [8] Kevin Frans, LB Soros, and Olaf Witkowski.

2021. Clipdraw: Exploring text-to-drawing synthesis through language-image encoders. arXiv preprint arXiv:2106.14843 (2021). 3

- [9] Yan Gong, Yiren Song, Yicheng Li, Chenglin Li, and Yin Zhang. 2025. RelationAdapter: Learning and Transferring Visual Relation with Diffusion Transformers. arXiv preprint arXiv:2506.02528 (2025). 2

- [10] Google. 2014. Noto Emoji Fonts. https:// github.com/googlefonts/noto-emoji. 6
- [11] Hailong Guo, Bohan Zeng, Yiren Song, Wentao Zhang, Chuang Zhang, and Jiaming Liu. 2025. Any2AnyTryon: Leveraging Adaptive Position Embeddings for Versatile Virtual Clothing Tasks. arXiv preprint arXiv:2501.15891 (2025). 2

- [12] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. 2023. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725 (2023). 2

- [13] Or Hirschorn, Amir Jevnisek, and Shai Avidan. 2024. Optimize & Reduce: A Top-Down Approach for Image Vectorization. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 2148–

2156. 2, 3, 5, 7

- [14] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685 (2021). 2

- [15] Teng Hu, Ran Yi, Haokun Zhu, Liang Liu, Jinlong Peng, Yabiao Wang, Chengjie Wang, and Lizhuang Ma. 2023. Stroke-based Neural Painting and Stylization with Dynamically Predicted Painting Region. In Proceedings of the 31st ACM International Conference on Multimedia. 7470–7480. 3

- [16] Shijie Huang, Yiren Song, Yuxuan Zhang, Hailong Guo, Xueyin Wang, Mike Zheng Shou, and Jiaming Liu. 2025. Photodoodle: Learning artistic image editing from few-shot pairwise data. arXiv preprint arXiv:2502.14397 (2025). 2

- [17] Ajay Jain, Amber Xie, and Pieter Abbeel. 2023. Vectorfusion: Text-to-svg by abstracting pixelbased diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 1911–1920. 2, 3, 5, 7

- [18] Black Forest Labs. 2023. FLUX. https:// github.com/black-forest-labs/flux. 3
- [19] Tzu-Mao Li, Michal Luk´aˇc, Micha¨el Gharbi, and Jonathan Ragan-Kelley. 2020. Differentiable vector graphics rasterization for editing and learning. ACM Transactions on Graphics (TOG) 39, 6 (2020), 1–15. 2, 3, 5, 7

- [20] Songhua Liu, Tianwei Lin, Dongliang He, Fu Li, Ruifeng Deng, Xin Li, Errui Ding, and Hao Wang.

2021. Paint transformer: Feed forward neural painting with stroke prediction. In Proceedings of the IEEE/CVF international conference on computer vision. 6598–6607. 3

- [21] Raphael Gontijo Lopes, David Ha, Douglas Eck, and Jonathon Shlens. 2019. A learned representation for scalable vector graphics. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 7930–7939. 2, 3

- [22] Runnan Lu, Yuxuan Zhang, Jiaming Liu, Haofan Wang, and Yiren Song. 2025. EasyText: Controllable Diffusion Transformer for Multilingual Text Rendering. arXiv preprint arXiv:2505.24417 (2025). 2

- [23] Xu Ma, Yuqian Zhou, Xingqian Xu, Bin Sun, Valerii Filev, Nikita Orlov, Yun Fu, and Humphrey Shi. 2022. Towards layer-wise image vectorization. In Proceedings of the IEEE/CVF Conference

on Computer Vision and Pattern Recognition. 16314–

16323. 2, 3, 5, 7

- [24] Zexu Pan, Zhaojie Luo, Jichen Yang, and Haizhou Li. 2020. Multi-modal attention for speech emotion recognition. arXiv preprint arXiv:2009.04107 (2020). 3

- [25] William Peebles and Saining Xie. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 4195–4205. 2, 3

- [26] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. 2022. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988 (2022). 2, 3

- [27] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PMLR, 8748–8763. 3

- [28] Pradyumna Reddy, Michael Gharbi, Michal Lukac, and Niloy J Mitra. 2021. Im2vec: Synthesizing vector graphics without vector supervision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7342–7351. 2, 3

- [29] Juan A Rodriguez, Shubham Agarwal, Issam H Laradji, Pau Rodriguez, David Vazquez, Christopher Pal, and Marco Pedersoli. 2023. Starvector: Generating scalable vector graphics code from images. arXiv preprint arXiv:2312.11556 (2023). 2

- [30] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. 2022. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684–

10695. 2, 3

- [31] Wenda Shi, Yiren Song, Zihan Rao, Dengming Zhang, Jiaming Liu, and Xingxing Zou. 2025. WordCon: Word-level Typography Control in Scene Text Rendering. arXiv preprint arXiv:2506.21276 (2025). 2

- [32] Wenda Shi, Yiren Song, Dengming Zhang, Jiaming Liu, and Xingxing Zou. 2024. FonTS: Text Rendering with Typography and Style Controls. arXiv preprint arXiv:2412.00136 (2024). 2

- [33] Jaskirat Singh, Cameron Smith, Jose Echevarria, and Liang Zheng. 2022. Intelli-Paint: Towards developing more human-intelligible painting agents. In European Conference on Computer Vision. Springer, 685–701. 3

- [34] Yiren Song. 2022. Cliptexture: Text-driven texture synthesis. In Proceedings of the 30th ACM

- International Conference on Multimedia. 5468–5476. 3
- [35] Yiren Song, Shijie Huang, Chen Yao, Xiaojun Ye, Hai Ci, Jiaming Liu, Yuxuan Zhang, and Mike Zheng Shou. 2024. ProcessPainter: Learn Painting Process from Sequence Data. arXiv preprint arXiv:2406.06062 (2024). 2

- [36] Yiren Song, Cheng Liu, and Mike Zheng Shou. 2025. MakeAnything: Harnessing Diffusion Transformers for Multi-Domain Procedural Sequence Generation. arXiv preprint arXiv:2502.01572 (2025). 2

- [37] Yiren Song, Cheng Liu, and Mike Zheng Shou.

2025. Omniconsistency: Learning style-agnostic consistency from paired stylization data. arXiv preprint arXiv:2505.18445 (2025). 2

- [38] Yiren Song, Xiaokang Liu, and Mike Zheng Shou. 2024. DiffSim: Taming Diffusion Models for Evaluating Visual Similarity. arXiv preprint arXiv:2412.14580 (2024). 2

- [39] Yiren Song, Shengtao Lou, Xiaokang Liu, Hai Ci, Pei Yang, Jiaming Liu, and Mike Zheng Shou. 2024. Anti-Reference: Universal and Immediate Defense Against Reference-Based Generation. arXiv preprint arXiv:2412.05980 (2024). 2

- [40] Yiren Song, Xuning Shao, Kang Chen, Weidong Zhang, Zhongliang Jing, and Minzhe Li. 2023. Clipvg: Text-guided image manipulation using differentiable vector graphics. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 37. 2312–

2320. 3

- [41] Yiren Song, Pei Yang, Hai Ci, and Mike Zheng Shou.

2024. IDProtector: An Adversarial Noise Encoder to Protect Against ID-Preserving Image Generation. arXiv preprint arXiv:2412.11638 (2024). 2

- [42] Yiren Song and Yuxuan Zhang. 2022. CLIPFont: Text Guided Vector WordArt Generation.. In BMVC. 543. 3

- [43] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing 568 (2024), 127063. 3

- [44] Zecheng Tang, Chenfei Wu, Zekai Zhang, Mingheng Ni, Shengming Yin, Yu Liu, Zhengyuan Yang, Lijuan Wang, Zicheng Liu, Juntao Li, et al. 2024. StrokeNUWA: Tokenizing Strokes for Vector Graphic Synthesis. arXiv preprint arXiv:2401.17093 (2024). 2

- [45] Vikas Thamizharasan, Difan Liu, Matthew Fisher, Nanxuan Zhao, Evangelos Kalogerakis, and Michal Lukac. 2024. NIVeL: Neural Implicit Vector Layers for Text-to-Vector Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4589–4597. 3

- [46] Yael Vinker, Ehsan Pajouheshgar, Jessica Y Bo, Roman Christian Bachmann, Amit Haim Bermano, Daniel Cohen-Or, Amir Zamir, and Ariel Shamir.

2022. Clipasso: Semantically-aware object sketching. ACM Transactions on Graphics (TOG) 41, 4 (2022), 1–11. 3

- [47] Cong Wan, Xiangyang Luo, Zijian Cai, Yiren Song, Yunlong Zhao, Yifan Bai, Yuhang He, and Yihong Gong. 2024. GRID: Visual Layout Generation. arXiv preprint arXiv:2412.10718 (2024). 4

- [48] Qixun Wang, Xu Bai, Haofan Wang, Zekui Qin, and Anthony Chen. 2024. Instantid: Zero-shot identitypreserving generation in seconds. arXiv preprint arXiv:2401.07519 (2024). 2

- [49] Yizhi Wang and Zhouhui Lian. 2021. Deepvecfont: synthesizing high-quality vector fonts via dualmodality learning. ACM Transactions on Graphics (TOG) 40, 6 (2021), 1–15. 3

- [50] Zhenyu Wang, Jianxi Huang, Zhida Sun, Daniel Cohen-Or, and Min Lu. 2024. Layered Image Vectorization via Semantic Simplification. arXiv preprint arXiv:2406.05404 (2024). 2

- [51] Zitong Wang, Hang Zhao, Qianyu Zhou, Xuequan Lu, Xiangtai Li, and Yiren Song. 2025. DiffDecompose: Layer-Wise Decomposition of Alpha-Composited Images via Diffusion Transformers. arXiv preprint arXiv:2505.21541 (2025). 2

- [52] Ronghuan Wu, Wanchao Su, Kede Ma, and Jing Liao.

2023. IconShop: Text-Based Vector Icon Synthesis with Autoregressive Transformers. arXiv preprint arXiv:2304.14400 (2023). 3

- [53] Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan. 2023. Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks. 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2023), 4818–

4829. https://api.semanticscholar. org/CorpusID:265128818 1

- [54] Ximing Xing, Juncheng Hu, Guotao Liang, Jing Zhang, Dong Xu, and Qian Yu. 2024. Empowering LLMs to Understand and Generate Complex Vector Graphics. arXiv preprint arXiv:2412.11102 (2024). 2

- [55] Ximing Xing, juncheng Hu, Liang Zhang, Jing Guotao, Dong Xu, and Qian Yu. 2024. Empowering LLMs to Understand and Generate Complex Vector Graphics. arXiv preprint (2024). 2

- [56] Ximing Xing, Chuang Wang, Haitao Zhou, Jing Zhang, Qian Yu, and Dong Xu. 2023. Diffsketcher: Text guided vector sketch synthesis through latent diffusion models. Advances in Neural Information Processing Systems 36 (2023), 15869–15889. 2, 3, 5, 7

- [57] Ximing Xing, Haitao Zhou, Chuang Wang, Jing Zhang, Dong Xu, and Qian Yu. 2024. SVGDreamer: Text guided SVG generation with diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4546–

4555. 2, 5, 7

- [58] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang.

2023. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721 (2023). 2, 5

- [59] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala.

2023. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836– 3847. 2, 5

- [60] Peiying Zhang, Nanxuan Zhao, and Jing Liao. 2024. Text-to-vector generation with neural path representation. ACM Transactions on Graphics (TOG) 43, 4

(2024), 1–13. 3

- [61] Yuxuan Zhang, Yiren Song, Jiaming Liu, Rui Wang, Jinpeng Yu, Hao Tang, Huaxia Li, Xu Tang, Yao Hu, Han Pan, and Zhongliang Jing. 2024. SSREncoder: Encoding Selective Subject Representation for Subject-Driven Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 8069–8078. 2

- [62] Yuxuan Zhang, Yiren Song, Jinpeng Yu, Han Pan, and Zhongliang Jing. 2024. Fast Personalized Text to Image Synthesis with Attention Injection. In ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). 6195–6199. https://doi.org/10.1109/ ICASSP48485.2024.10447042

- [63] Yuxuan Zhang, Lifu Wei, Qing Zhang, Yiren Song, Jiaming Liu, Huaxia Li, Xu Tang, Yao Hu, and Haibo Zhao. 2024. Stable-Makeup: When Real-World Makeup Transfer Meets Diffusion Model. arXiv preprint arXiv:2403.07764 (2024). 2

- [64] Yuxuan Zhang, Yirui Yuan, Yiren Song, Haofan Wang, and Jiaming Liu. 2025. Easycontrol: Adding efficient and flexible control for diffusion transformer. arXiv preprint arXiv:2503.07027 (2025). 2

- [65] Yuxuan Zhang, Qing Zhang, Yiren Song, and Jiaming Liu. 2024. Stable-Hair: Real-World Hair Transfer via Diffusion Model. arXiv preprint arXiv:2407.14078

(2024). 2

## LayerTracer: Cognitive-Aligned Layered SVG Synthesis via Diffusion Transformer

### Supplementary Material

#### A. User Study Details

We conducted a user study in the form of an online survey, with a total of 46 participants evaluating 36 questions per questionnaire. The study was divided into two sections:

The first section evaluated text-to-vector generation results, comparing different methods. Each comparison included the corresponding text prompt, allowing participants to select the generated vector graphic that best reflected the original textual description. The evaluation consisted of three examples per method, leading to a total of nine comparisons, with two questions per comparison:

1. Which vector graphics result looks better? 2. Which result better reflects the meaning of the original text?

The second section focused on layer-wise vectorization, comparing different approaches. To ensure a fair evaluation, we provided a structured breakdown of the vectorization process, allowing participants to assess the quality and rationality of the vector graphic creation process. This section also contained three examples per method, resulting in nine comparisons, with two evaluation questions:

1. Which vectorization result looks better? 2. Which result better preserves the characteristics of the original raster image?

#### B. Dataset Construction Methods.

The SVG dataset proposed in this work is collected from multiple sources, including the internet, vendor procurement, and designer-created assets. In addition to manually crafted layered SVGs, we also engaged designers to reorganize and arrange layers of other SVGs into a logical sequence that aligns with the design creation process. The final training data was obtained through multiple rounds of human-in-the-loop filtering. The dataset consists of a total of 20K samples across three styles: black outline icons (3K), illustrations (2K), and emojis (15K). For caption annotation, we added different triggers for different datasets and used the Florence-2 model [53] to label the last frame of the SVG sequence.

#### C. Failure Cases

This section presents some failure cases of LayerTracer. As shown in the figure, when generating pixel-format sequence data, issues such as unwanted repeat, no layering, and inconsistencies between consecutive frames may occur. The problem of undesired repetition can be mitigated during the Layer-wise vectorization stage by comparing adjacent

[Figure 38]

Figure 10. Examples of questions in the User Study online questionnaire.

frames to remove duplicates. However, incorrect layering and frame inconsistencies can negatively impact the quality of Layer-wise SVG generation.

|[Figure 39]<br><br>[Figure 40]|
|---|

|[Figure 41]<br><br>[Figure 42]|
|---|

[Figure 43]

··· a small, yellow bird with a round body, beak, and wings ··· ··· an emoji, a person with orange hair in a green shirt, cheerfully ···

SVG Results SVG Results

··· an emoji, a person in a white spacesuit, wearing a friendly smile ··· ··· an emoji, a person in a blue uniform with a cap, an official appearance ···

|[Figure 44]<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]|
|---|

··· a nine-step SVG icon, a woman with green hair, yellow blouse, holding red folder and cereal bowl, with a thin black outline ···

··· a nine-step SVG icon, a blue rectangular appliance with circular power button, with a thin black outline ···

··· a nine-step SVG icon, a mason jar with bright orange liquid and cute kawaii face, green leaf on top, wide smile, with a thin black outline ···

··· a nine-step SVG icon, a cityscape line drawing with tall modern building and power plant, with a thin black outline ···

SVG Results

··· a nine-step SVG icon, a soft gold wristwatch with a thin black outline ···

Figure 11. More results of layered vector graphics generation. The red boxes highlight the SVG format results, which can be zoomed in to view the details.

[Figure 49]

###### Figure 12. Each row represents examples of SVG datasets for three categories: emojis, illustrations, and black outline icons.

| |
|---|

| |
|---|

[Figure 50]

Unwanted repetition

··· A freshly cut dragon fruit with vibrant pink skin, green leaves ···

| |
|---|

[Figure 51]

Unwanted repetition

··· A cheerful, cartoon-style toaster in orange and white, with golden toast popping out ···

[Figure 52]

| |
|---|

Incorrect layering

··· A refreshing glass of orange drink with ice cubes, an orange slice garnish ···

| |
|---|

[Figure 53]

Inconsistencies

··· An electric fan with blue blades ···

| |
|---|

[Figure 54]

[Figure 55]

| |
|---|

| |
|---|

Incorrect layering Inconsistencies

###### SVG Results SVG Results

··· A person sits on a bench, engrossed in reading a

··· A person in a green sweater is intently using a smartphone while seated inside a vehicle ···

newspaper ···

Figure 13. Some failure cases.

