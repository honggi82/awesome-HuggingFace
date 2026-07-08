## Inpaint4Drag: Repurposing Inpainting Models for Drag-Based Image Editing via Bidirectional Warping

Jingyi Lu Kai Han† Visual AI Lab, The University of Hong Kong

lujingyi@connect.hku.hk, kaihanx@hku.hk

# arXiv:2509.04582v1[cs.CV]4Sep2025

### Abstract

Drag-based image editing has emerged as a powerful paradigm for intuitive image manipulation. However, existing approaches predominantly rely on manipulating the latent space of generative models, leading to limited precision, delayed feedback, and model-specific constraints. Accordingly, we present Inpaint4Drag, a novel framework that decomposes drag-based editing into pixel-space bidirectional warping and image inpainting. Inspired by elastic object deformation in the physical world, we treat image regions as deformable materials that maintain natural shape under user manipulation. Our method achieves real-time warping previews (0.01s) and efficient inpainting (0.3s) at 512×512 resolution, significantly improving the interaction experience compared to existing methods that require minutes per edit. By transforming drag inputs directly into standard inpainting formats, our approach serves as a universal adapter for any inpainting model without architecture modification, automatically inheriting all future improvements in inpainting technology. Extensive experiments demonstrate that our method achieves superior visual quality and precise control while maintaining realtime performance. Project page: https://visualai.github.io/inpaint4drag

### 1. Introduction

Image manipulation remains a fundamental challenge in computer vision, with increasing demand for intuitive tools that enable users to naturally modify image content. Among various interaction paradigms, drag-based editing has emerged as a promising direction [31, 40], offering an intuitive way to directly manipulate image elements through simple mouse operations. Recent advances in this field have demonstrated impressive results [22, 52], allowing users to move, resize, or deform objects within images through simple drag operations.

†Corresponding author.

However, existing drag-based editing methods face several fundamental limitations. These approaches [12, 20, 26, 31] rely on manipulating the latent space of generative models like Stable Diffusion [23, 28, 35], leading to three key challenges: (1) imprecise control – latent space manipulations obscure the relationship between user inputs and resulting changes when control points are downscaled to match the lower latent resolution (e.g., from 512×512 to 32×32 in SD-UNet), (2) poor interactivity – users are forced into time-consuming trial-and-error cycles without immediate visual feedback during the generative process, and (3) limited capability (shown in Fig. 1) – these approaches often produce unrealistic results when handling large occlusions (e.g., rotating heads or opening mouths) as they rely on general-purpose text-to-image models that are not specifically trained to handle missing or occluded regions.

In this paper, we present Inpaint4Drag, a novel interactive framework that decomposes drag-based image editing into bidirectional warping and image inpainting. Inspired by elastic object deformation in the physical world [38, 39], we treat image regions as deformable materials that maintain natural shape under user manipulation. Given an input image, users specify a region mask to define the deformable area and place handle and target control points to guide the transformation. To assist with complex object manipulation, we provide an optional mask refinement module that automatically captures precise object boundaries, ensuring consistent deformation of both edges and interior regions. The core of our approach lies in the bidirectional warping algorithm that transforms the input region based on user control inputs. While forward warping alone moves source pixels to their target positions, it inevitably creates holes and gaps that lead to artifacts. We address this by combining forward warping to define initial contours and rough deformation, with backward warping to fill gaps and establish dense correspondence for the geometric transformation. The warped result is complemented by computed inpainting masks that include dilated revealed regions (areas that emerge after image deformation) and a narrow band around warped contours for boundary smoothing. This

Sparse Input DragDiff. DiffEditor FastDrag SDE-Drag Our Input Preview Inpainted

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

[Figure 16]

Run Time ~3 min 43 sec 4.2 sec ~2 min 0.01 sec 0.3 sec

- Figure 1. Comparison of drag-based image editing methods. Our approach (rightmost columns) enables both precise edits and large-scale occlusion handling with real-time preview (0.01 sec) followed by rapid inpainting (0.3 sec). Users select deformable regions and drag from handle points to target positions, with the preview column showing grid overlays in areas requiring inpainting. In contrast, existing methods (leftmost columns) require substantially longer processing times without providing interactive feedback during editing, and often struggle with precise manipulations due to latent-space operations.

regions and boundary smoothing to enable seamless integration with existing inpainting models.

complete input is then passed to modern inpainting models (e.g., LaMa [43], SD-Inpaint [35]) for final completion. Our decomposition enables a clear separation between geometric transformation and content generation while maintaining the familiar drag interaction paradigm.

### 2. Related Work

Generative Models for Image Editing. Generative models have revolutionized image editing, with generative adversarial networks (GANs) marking a pivotal breakthrough [11, 13–15]. While numerous editing techniques have emerged from the GAN framework [7, 9, 31, 33, 45–47], their practical applications remain constrained by training data diversity, model capacity, and challenges in inverting real images into GAN latent spaces [7, 18, 53]. The emergence of large-scale text-to-image diffusion models [35, 37] has expanded image manipulation capabilities through text prompts [2, 5, 6, 10, 16], enabling control over style, motion, and object categories. However, these text-based approaches lack pixel-level precision, focusing primarily on semantic alterations. To address this limitation, drag-based techniques [9, 20, 24, 26, 27, 31, 40] have emerged, allowing fine-grained manipulation of object posture and shape through interactive keypoint control, effectively combining generative modeling with the precision needed for detailed image editing.

Our physics-inspired formulation and bidirectional warping algorithm enable effective drag-based manipulation for inpainting models. It possesses three key advantages: First, our pixel-space deformation estimation enables precise geometric control while preserving colors of dragged content and maintaining image quality in unedited regions. Second, our method effectively handles large occlusions by leveraging specialized inpainting models, enabling challenging edits like opening a lion’s mouth or rotating a person’s head. Third, our approach serves as a universal adapter for the inpainting field - by transforming drag inputs directly into standard inpainting formats, we enable any inpainting model to function as a drag editing method without architecture modification, automatically inheriting all future improvements in inpainting technology. Notably, our system achieves exceptional interaction fluidity with real-time warping previews (0.01s) and efficient inpainting operations (0.3s) (measured on 512×512 images). We summarize our technical contributions as follows:

Drag-based Image Editing. Drag-based methods have transformed image editing by enabling feature manipulation through paired handle and target points. DragGAN [31] pioneered multi-point editing in GAN-generated images using iterative motion supervision and point tracking, while FreeDrag [20] enhanced precision by operating in feature space. Building on these foundations, several diffusion-based approaches have emerged. DragDiffusion [40] adapted the supervise-and-track framework to diffusion models, while DiffEditor [27] and EasyDrag [12] extended feature dragging across the denoising process. SDE-

- • A physics-inspired deformation framework that treats image regions as elastic materials, enabling natural transformations through user-specified control points and region masks, with optional mask refinement for precise object boundary handling.
- • An efficient bidirectional warping algorithm that establishes initial shape through forward warping and fills gaps via backward mapping, creating dense pixel correspondences while maintaining real-time performance.
- • A modular pipeline that clearly separates transformation from generation, computing precise masks for revealed

inpainting models for seamless completion.

Image

I Point Sampling

SAM Result pred

( pred ∩ dilated)∪ eroded New User MaskM

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

SAM

#### 3.1.RegionSpecificationandBoundaryRefinement

User Mask

User Control {ℎ, }

M

Previous drag editing methods [31, 41, 52] typically use sparse control points to guide deformation, with optional masking to restrict editable regions. However, this sparse input format introduces fundamental ambiguity in deformation interpretation – the movement of numerous pixels relies on the guidance from only a few control points. A single drag operation on a character’s arm could produce vastly different outcomes: full-body rotation, localized arm movement, or isolated point translation (see S4 in supplementary material for examples). Without clear deformation specifications, existing methods often produce unpredictable results that deviate from user intentions.

eroded dilated

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

1

1 1 1 1 1 1 1 1 1

Dilation / Erosion Kernel

- Figure 2. Overview of our optional mask refinement module. Users can achieve precise object boundaries for coherent deformation through SAM [17], constrained by eroded and dilated mask boundaries to preserve user intent.

Drag [28] introduced sparse latent copy-paste techniques, and RegionDrag [22] improved efficiency through dense region mapping. InstantDrag [42], a motion prediction approach, combines motion guidance with generative models but requires test-time optimization on video clips for each new edit. FastDrag [52], while sharing our intuition of object stretching, operates solely in latent space with sequential processing, lacking our method’s vectorized pixel-space manipulation and specialized inpainting capabilities.

Image Inpainting. Image inpainting has evolved from classical PDE-based [4] and patch-based [3, 8] methods to modern deep learning approaches. CNN-based methods [34] first learned semantic priors, followed by GANs [21, 48] and transformers [19] that improve the inpainting performance through adversarial training or enhancing longrange consistency. Recent diffusion models like Stable Diffusion [35] and LaMa [43] achieve superior quality in structural coherence and detail synthesis, widely adopted by commercial tools like Adobe Photoshop [1], DALLE3 [30], and RunwayML [36], which inspires us to consider the emerging drag-based image editing problem from the inpainting perspective.

- 3. Method

We address this ambiguity by requiring users to explicitly mask regions M intended for deformation, similar to how we naturally identify movable parts of objects. Given an input image I ∈ RH×W×3, a set of handle points {hi} and their target positions {ti}, our key insight is to treat each masked region as an elastic material, where moving a handle point creates a ripple effect throughout the connected area – much like stretching a rubber sheet. Our bidirectional warping approach (Sec. 3.2) computes dense deformation fields that ensure smooth influence propagation from control points to every pixel p ∈ M while maintaining complete pixel coverage throughout the deformed region. For effective deformation propagation, mask boundaries should align with object boundaries – pixels belonging to the same object should move coherently. To maintain deformation coherence across object boundaries while simplifying user interactions, we propose an optional mask refinement module based on the Segment Anything Model (SAM) [17]. Given a user input mask M containing points P = {pi}Ni=1, directly using all mask points as SAM input would introduce computational bottlenecks. To address this, we sample grid points Ps ⊂ P from the user-drawn mask as SAM input, which maintains interactive performance while preserving critical boundary information.

We present Inpaint4Drag, an interactive approach for dragbased image editing that decomposes the task into bidirectional warping and standard image inpainting. Given an input image, users can draw masks to select objects for deformation, with our mask refinement module to improve boundary precision (Sec. 3.1). After specifying drag point pairs, our bidirectional warping algorithm immediately deforms the selected region (Sec. 3.2), providing real-time preview of the editing result that also serves as input for the inpainting model. Users iteratively refine masks and control points to achieve desired dragging effects before executing the relatively expensive inpainting operation, which fills regions revealed by deformation (Sec. 3.3). Our approach enables real-time interaction while delivering high-quality results through specialized deformation control and integrated

We process these sampled points through SAM’s prompt encoder fSAM to obtain point embeddings, which are combined with the image embedding to generate a prediction mask Mpred = fSAM(I,Ps). Despite SAM’s powerful segmentation capabilities, we found that direct SAM predictions can generate disconnected regions or capture unintended objects far from the user’s specified area due to SAM’s tendency to segment complete objects and semantically similar instances rather than user-intended regions (see Fig. 2 or Fig. 6 for examples). We address this limitation with a two-step refinement approach that balances automatic boundary detection while preserving user intent. First, we generate dilated and eroded versions of the input

###### Extracted Contours Point Association

###### Forward-warped Contour Boundary

Backward mapping warped

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

[Figure 47]

[Figure 48]

[Figure 49]

’

[Figure 50]

[Figure 51]

’

’

1

3 3

2

2

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

’

’

- 2
- 3

’

2

1

1

2

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

’ 1

1

[Figure 60]

’

’

3

3

User Mask M

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

1

- 2
- 3

[Figure 72]

User Control {ℎ, }

Bidirectional-warped Image

Deformation

Forward-warped Image

푤푎푟   

- Figure 3. Overview of our bidirectional warping pipeline. Given a user mask and control points, we first extract region contours and establish point associations. The forward warping step then maps these contours to their target locations and builds initial correspondences. Finally, through backward mapping, we generate the warped image with complete pixel coverage and its corresponding warped mask, which will be used for subsequent inpainting. mask:

deformable regions; (2) forward warping to define target region boundaries and establish initial mapping; (3) backward mapping to ensure complete pixel coverage in target regions; and (4) leveraging the established mapping to generate warped content and identify areas requiring inpainting.

M(x − i,y − j), (1)

Mdilated(x,y) = max

(i,j)∈K1

M(x − i,y − j), (2)

Meroded(x,y) = min

(i,j)∈K1

##### 3.2.1. Contour Extraction and Control Point Association

- where K1 represents a dilation / erosion kernel with radius r1. The refined mask M is obtained through our boundaryguided formulation:

We first decompose the binary mask M into distinct contours representing separate deformable regions (left of pink block, Fig. 3). A contour here refers to a closed curve that traces the boundary of a connected area in the mask:

M = (Mpred ∩ Mdilated) ∪ Meroded, (3)

C = findContours(M). (4)

where ∪ and ∩ represent the logical OR and AND operations respectively. This approach uses the dilated and eroded masks to create natural boundary constraints, ensuring the refined result respects the user’s original intent while improving boundary precision. By exposing the kernel radius r1 as a control parameter, users can tailor the refinement strength, achieving their desired balance between automatic boundary detection and manual control.

To maintain local deformation coherence, we associate each contour point c ∈ C with control points located inside its region:

(Hc,Tc) = {(hi,ti)|hi inside C}. (5) This ensures that each region responds only to control points within its boundaries, preserving the intuitive behavior where deformation occurs only where user interaction is directly applied. For brevity, we simply use (hi,ti) in subsequent equations to refer to a control points associated with a contour point c.

#### 3.2. Bidirectional Warping for Region Deformation Given a user mask M (either original or refined) and con-

trol point pairs {(hi,ti)}Ki=1, our bidirectional warping algorithm aims to translate these inputs into a coherent deformation of the masked region and prepares standard inputs for inpainting models: a warped image Iwarped containing deformed content alongside an inpainting mask Minpaint identifying areas left vacant by relocated pixels.

##### 3.2.2. Forward Warping

For regions with a single control point pair, deformation simplifies to translation, where region pixels directly shift from handle to target position. When multiple control points are present, the deformation becomes non-rigid, requiring a interpolation-based transformation. We first perform forward warping (green block, Fig. 3) which serves

As shown in Fig. 3 and Fig. 4, our approach consists of four steps: (1) contour extraction to identify independent

User Mask M

temp temp ∪ warped

inpaint

[Figure 73]

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

M ∩ ( 1 - warped )

2

Inpainting Model

1 1 1 1 1 1 1 1 1

Dilation Kernel

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

1 - warped

warped warped

warped edit

- Figure 4. Overview of inpainting mask computation. The final mask (Minpaint) combines dilated unmapped regions (Mtemp) from areas absent after warping and dilated boundaries of the warped mask (∂Mwarped). This ensures smooth transitions between warped and generated regions in the final result.

two critical purposes: i) it transforms the contour itself to define the target region boundary C′, and ii) it establishes initial pixel-level mapping to guide subsequent processing. For each point p in the source region (including boundary points of contour C), we compute its target position pt through weighted interpolation:

where Nn represents the n nearest target pixels that were successfully matched during the forward process, ptgti represents one of these matched nearest target positions, psrci is its corresponding source position, and wi are the inverse distance weights as previously defined.

This local neighborhood approach provides computational efficiency compared to using global pixel references and helps preserve structural coherence by limiting the influence of distant forward warping results on the final mapping. We validate each computed mapping pair by ensuring all coordinates remain within image boundaries:

NC

wi(ti − hi), (6)

pt = p +

i=1

where NC denotes the number of control points associated with contour C, and weights wi are computed through inverse distance weighting:

Valid(ps,pt) = ps,pt ∈ [0,W) × [0,H), (9)

where W and H denote the image width and height, respectively. This prevents sampling from undefined regions outside the image domain.

1/(∥p − hi∥ + ϵ)

. (7)

wi =

NC j=1 1/(∥p − hj∥ + ϵ)

Overall, our backward mapping strategy completes the bidirectional framework by establishing reliable pixel-level mappings that maintain local structural relationships while addressing the limitations of forward transformation.

While this forward approach establishes initial mapping, using it alone for warping creates sampling artifacts (see Fig. 3 or Fig. 7 for examples). Specifically, when nonrigid transformations stretch image regions, the discrete nature of pixels creates gaps where target locations receive no mapped value. This occurs when the transformed source pixel grid becomes stretched and discontinuous in the target space, leaving unmapped gaps.

3.2.4. Computing Warped Result and Inpainting Mask Using the established pixel mappings, we generate the warped image by transferring pixel values:

Iwarped(pt) = I(ps) for all valid (ps,pt) pairs. (10)

##### 3.2.3. Backward Mapping

As shown in Fig. 4, we then identify regions requiring inpainting by determining pixels present in the original mask M but unmapped in the deformed result:

To ensure complete pixel coverage within the deformed region (blue block, Fig. 3), we compute source positions for all pixels in the transformed target contour C′. For each target pixel pt ∈ C′, we determine its corresponding source position ps using:

Mtemp = M \ Mwarped. (11) To ensure smooth transitions at both warped content boundaries and vacated regions, we apply dilation to both the unmapped areas Mtemp and the boundaries of the warped mask

Nn

wi(psrci − ptgti ), (8)

ps = pt +

i=1

∂Mwarped. This creates a buffer zone around the boundaries by expanding the inpainting mask, allowing the inpainting model to handle the transition areas and avoid abrupt edges between warped and newly generated content:

Minpaint = dilate(Mtemp ∪ ∂Mwarped,K2), (12)

- where K2 represents the dilation kernel with radius r2. To this end, our bidirectional warping algorithm has processed

the user mask M and control point pairs {(hi,ti)}Ki=1 to produce the warped image Iwarped and an inpainting mask Minpaint, which together form the standard input required by image inpainting models.

- 3.3. Integration with Image Inpainting

The final step in our pipeline is to apply an inpainting model to generate content for areas revealed during deformation:

Iedit = Inpaint(Iwarped,Minpaint). (13)

Before executing inpainting operations, our method provides Iwarped as a real-time preview of the final result Iedit—a capability absent in existing approaches. This preview enables users to achieve desired dragging effects by adjusting mask and control points, resulting in a more interactive editing experience that avoids unnecessary expense from repeated inpainting attempts.

For our implementation, we selected the Stable Diffusion 1.5 Inpainting Checkpoint [35], which was fine-tuned from the regular Stable Diffusion v1.2 model with additional training for inpainting tasks. The inpainting process follows a straightforward pipeline: the mask is resized and concatenated with the image VAE latent representation. During conditional diffusion denoising, we initialize masked regions with pure noise to generate entirely new content in areas revealed by deformation. Finally, the result is transformed back to pixel space through VAE decoding.

To optimize performance, we incorporated several efficiency enhancements: TinyAutoencoder SD (TAESD) [29], a distilled VAE that reduces memory requirements; LCM (Latent Consistency Model) LoRA [25] to reduce sampling steps; an empty text prompt to eliminate classifier-free guidance computation; and caching of the empty prompt embeddings to avoid repetitive calculations during editing sessions. It’s worth noting that while we report experimental results using this representative inpainting model, our framework can accommodate any inpainting model as a drop-in replacement (see S2 in supplementary material for examples).

- 4. Experiments

- 4.1. Datasets

We evaluate drag editing methods using two benchmarks: DragBench-S from SDE-Drag [28] with 100 samples, and

DragBench-D from DragDiffusion [40] containing 205 samples. Each benchmark entry includes a source image, a text prompt describing the image, a binary mask for the editable region, and point pairs showing desired movements. In our framework, masks serve a different purpose as deformable regions. While editable regions indicate where editing is allowed, deformable regions define coherent parts that should move together during transformation. We therefore re-annotated the deformable region locations and dragging points while preserving the original user editing intentions and keeping the source images unchanged.

#### 4.2. Evaluation Metrics

Following [22], we measure editing performance using LPIPS [50] and Mean Distance (MD).

LPIPS: Learned Perceptual Image Patch Similarity (LPIPS) v0.1 [50] measures identity preservation between original and edited images. Lower LPIPS indicates better identity preservation. However, a limitation of this metric is that it only measures low-level feature similarities between image patches. In drag editing, intentional and correct shape deformations often result in unavoidable LPIPS increases that should not be penalized.

Mean Distance (MD): This metric evaluates how accurately handle points are moved to target positions. DIFT [44] is employed to find matched points for the userspecified handle points in the edited image, restricting the search area to regions around user-specified handle and target points to avoid false matches. MD is calculated as the average normalized Euclidean distance between target points and DIFT-matched points.

#### 4.3. Implementation Details

Our framework is implemented in Python using PyTorch [32] on a single NVIDIA Tesla V100-SXM2 GPU. For mask refinement, we adopt EfficientVIT-SAM-L0 [51] and sample points from user masks, limiting the number of sampled points to |Ps| ≤ 128 for computational efficiency. The dilation/erosion kernel radius r1 is set to 10 pixels to provide sufficient boundary refinement while preserving user intent. In the bidirectional warping module, we set ϵ = 10−6 for numerical stability in weight computation and use Nn = 4 nearest neighbors for backward mapping. The final inpainting mask is dilated with kernel radius r2 = 5 pixels to ensure smooth transitions. For inpainting, we employ Stable Diffusion 1.5 inpainting checkpoint [35] with TAESD [29] and LCM LoRA [25], processing images resized to 512 pixels on the longer edge while maintaining aspect ratio. We use 8 sampling steps for diffusion without classifier-free guidance.

Sparse Input FastDrag DragDiff. DiffEditor SDE-Drag Our Input Preview Inpainted

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

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

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

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

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

- Figure 5. Qualitative comparison with state-of-the-art methods on challenging cases from DragBench-S [28] and DragBench-D [40]. Our method enables users to specify deformable regions and manipulate them by dragging handle points toward target destinations. The grid overlay in the preview column indicates areas to be inpainted. Our approach shows advantages over existing methods by effectively preserving local details in dragged regions through our bidirectional warping algorithm while demonstrating strong capability in generating occluded regions. The real-time preview (∼10ms) allows users to interactively adjust edits before executing the inpainting process (∼0.3s).

Method DragBench-S DragBench-D Mem(GB)↓ Time(s)↓ MD↓ LPIPS↓ MD↓ LPIPS↓

DragDiffusion [40] 11.6 177.7 7.0 18.0 6.7 10.2 DiffEditor [27] 6.6 43.1 23.6 17.6 22.1 10.9 SDE-Drag [28] 6.9 126.1 7.5 11.4 8.1 14.9 FastDrag [52] 5.0 4.2 4.1 24.1 5.1 13.5 Ours 2.7 0.3 3.6 11.4 3.9 9.1

Table 1. Comparison of different drag-based image editing methods. MD and LPIPS values are scaled by 100. Time and GPU memory are measured at 512×512 resolution.

#### 4.4. Quantitative Evaluation

As shown in Tab. 1, we compare Inpaint4Drag with stateof-the-art methods [27, 28, 40, 52]. Our bidirectional warping algorithm enables significant improvements in dragging precision and image consistency on DragBench-S [28] and DragBench-D [40], achieving the lowest MD scores (3.6/3.9) and competitive LPIPS values (11.4/9.1). This advantage stems from our dense pixel-wise deformation calculation that preserves color and geometric relationships during manipulation. Notably, Inpaint4Drag is 14× faster than FastDrag [52] and nearly 600× faster than DragDif-

Initial Mask SAM Prediction Refined Mask

[Figure 149]

[Figure 150]

[Figure 151]

Dilation Boundary Dilation Boundary

[Figure 152]

[Figure 153]

[Figure 154]

Dilation Boundary Dilation Boundary

- Figure 6. Qualitative results of our mask refinement module Sec. 3.1. Left: User-provided initial input mask. Middle: Raw segmentation predictions from Segment Anything Model (SAM). Right: Final refined results where the green dashed boundary, derived from dilating the initial input mask, effectively filters out undesired SAM predictions beyond the user’s intended scope.

fusion [40], with SAM-based refinement taking 0.02s and bidirectional warping only 0.01s. The computational peak occurs during SD inpainting for previews, requiring 0.29s while using the least memory (2.7GB) among all methods.

#### 4.5. Qualitative Results

As shown in Fig. 5, Inpaint4Drag outperforms state-ofthe-art methods on challenging cases from DragBench-S and DragBench-D (see S5 in supplementary materials for more results). By leveraging specialized inpainting models, Inpaint4Drag generates realistic content in previously occluded regions (e.g., newly exposed facial features in rows 1, and 4, and the lion’s open mouth in row 3). Our optional mask refinement module enables users to coherently deform object boundaries (rows 4, 5, and 7) or focus on precise local edits (remaining examples). Unlike latent-space methods that lose precision when downscaling control points to latent resolution, our pixel-space approach enables accurate manipulation of local details (row 2, and 5). Our bidirectional warping generates an informative preview of the dragged content, providing informative context for subsequent inpainting process.

#### 4.6. Method Analysis

Mask Refinement. As demonstrated in Fig. 6, our approach transforms rough initial masks (left) into refined results (right) by constraining SAM’s predictions (middle) within a dilated region of the user’s input. While our refinement module incorporates both inner and outer boundary constraints, for visualization clarity, we only showcase the outer boundary usage in these examples. The resulting masks effectively capture object boundaries while preserving the user’s intended editing scope.

Input Unidirectional Bidirectional

[Figure 155]

[Figure 156]

[Figure 157]

| |
|---|

[Figure 158]

[Figure 159]

[Figure 160]

| |
|---|

[Figure 161]

[Figure 162]

[Figure 163]

| |
|---|

Figure 7. Unidirectional vs. bidirectional warping (ours). Our method fixes sampling gaps using backward pixel mapping.

Unidirectional vs. Bidirectional Warping. We present qualitative comparisons between unidirectional (forwardwarping-only) and bidirectional warping (Sec. 3.2) in Fig. 7. The unidirectional approach struggles with sampling artifacts, creating noticeable gaps in stretched regions during deformation. These artifacts emerge when non-rigid transformations stretch image regions, as the discrete nature of pixels results in unmapped gaps where target locations receive no source values due to discontinuities in the transformed pixel grid. Our bidirectional method effectively addresses these challenges through a two-step process: first identifying target contours via forward warping, then employing pixel-level backward mapping to fill the gaps. This approach yields smooth transformations without discontinuities, providing reliable visual context for both user preview and image inpainting.

### 5. Conclusion

In this paper, we introduce Inpaint4Drag, a novel approach that repurposes image inpainting for drag-based editing through pixel-space bidirectional warping. Unlike existing solutions that rely on general-purpose text-to-image models unoptimized for drag operations, our specialized separation of warping and inpainting effectively maintains pixel consistency while generating high-quality content in newly revealed areas. Our bidirectional warping algorithm and SAM-based boundary refinement provide realtime feedback for intuitive interaction. Experimental results show that Inpaint4Drag delivers superior performance while reducing processing time from minutes to milliseconds. Moreover, since Inpaint4Drag is compatible with any inpainting model, it can continuously improve alongside advancements in inpainting technology.

Acknowledgments. This work is supported by the Hong Kong Research Grants Council - General Research Fund (Grant No.: 17211024).

### References

- [1] Adobe. Adobe Photoshop Content-Aware Fill. https: //www.adobe.com/products/photoshop.html,

2024. Accessed: 2024-02-14. 3

- [2] Omri Avrahami, Ohad Fried, and Dani Lischinski. Blended latent diffusion. TOG, 2023. 2
- [3] Connelly Barnes, Eli Shechtman, Adam Finkelstein, and Dan B Goldman. Patchmatch: A randomized correspondence algorithm for structural image editing. TOG, 2009. 3
- [4] Marcelo Bertalmio, Guillermo Sapiro, Vincent Caselles, and Coloma Ballester. Image inpainting. In ACM SIGGRAPH,

2000. 3

- [5] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, 2023. 2
- [6] Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. arXiv preprint arXiv:2304.08465, 2023. 2
- [7] Antonia Creswell and Anil Anthony Bharath. Inverting the generator of a generative adversarial network. IEEE TNNLS,

2018. 2

- [8] Antonio Criminisi, Patrick P’erez, and Kentaro Toyama. Region filling and object removal by exemplar-based image inpainting. IEEE TIP, 2004. 3
- [9] Yuki Endo. User-controllable latent transformer for stylegan image layout editing. In CGF, 2022. 2
- [10] Dave Epstein, Allan Jabri, Ben Poole, Alexei Efros, and Aleksander Holynski. Diffusion self-guidance for controllable image generation. In NeurIPS, 2024. 2
- [11] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In NeurIPS,

2014. 2

- [12] Xingzhong Hou, Boxiao Liu, Yi Zhang, Jihao Liu, Yu Liu, and Haihang You. Easydrag: Efficient point-based manipulation on diffusion models. In CVPR, 2024. 1, 2
- [13] Minguk Kang, Jun-Yan Zhu, Richard Zhang, Jaesik Park, Eli Shechtman, Sylvain Paris, and Taesung Park. Scaling up gans for text-to-image synthesis. In CVPR, 2023. 2
- [14] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of stylegan. In CVPR, 2020.
- [15] Tero Karras, Miika Aittala, Samuli Laine, Erik H¨ark¨onen, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Alias-free generative adversarial networks. In NeurIPS, 2021. 2
- [16] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In CVPR, 2023. 2

- [17] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, 2023. 3
- [18] Muyang Li, Ji Lin, Yaoyao Ding, Zhijian Liu, Jun-Yan Zhu, and Song Han. Gan compression: Efficient architectures for interactive conditional gans. In CVPR, 2020. 2
- [19] Wenbo Li, Zhe Lin, Kun Zhou, Lu Qi, Yi Wang, and Jiaya Jia. Mat: Mask-aware transformer for large hole image inpainting. In CVPR, 2022. 3
- [20] Pengyang Ling, Lin Chen, Pan Zhang, Huaian Chen, and Yi Jin. Freedrag: Point tracking is not you need for interactive point-based image editing. arXiv preprint arXiv:2307.04684, 2023. 1, 2
- [21] Guilin Liu, Fitsum A Reda, Kevin J Shih, Ting-Chun Wang, Andrew Tao, and Bryan Catanzaro. Image inpainting for irregular holes using partial convolutions. In ECCV, 2018. 3
- [22] Jingyi Lu, Xinghui Li, and Kai Han. Regiondrag: Fast region-based image editing with diffusion models. In ECCV,

2024. 1, 3, 6

- [23] Grace Luo, Trevor Darrell, Oliver Wang, Dan B Goldman, and Aleksander Holynski. Readout guidance: Learning control from diffusion features. In CVPR, 2024. 1
- [24] Minxing Luo, Wentao Cheng, and Jian Yang. Rotationdrag: Point-based image editing with rotated diffusion features. arXiv preprint arXiv:2401.06442, 2024. 2
- [25] Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, Patrick von Platen, Apolin´ario Passos, Longbo Huang, Jian Li, and Hang Zhao. Lcm-lora: A universal stable-diffusion acceleration module. arXiv preprint arXiv:2311.05556, 2023. 6
- [26] Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. Dragondiffusion: Enabling drag-style manipulation on diffusion models. arXiv preprint arXiv:2307.02421,

2023. 1, 2

- [27] Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. Diffeditor: Boosting accuracy and flexibility on diffusion-based image editing. arXiv preprint arXiv:2402.02583, 2024. 2, 7
- [28] Shen Nie, Hanzhong Allan Guo, Cheng Lu, Yuhao Zhou, Chenyu Zheng, and Chongxuan Li. The blessing of randomness: Sde beats ode in general diffusion-based image editing. arXiv preprint arXiv:2311.01410, 2023. 1, 3, 6, 7
- [29] Ollin. Taesd: Tiny autoencoder for stable diffusion. https: //github.com/madebyollin/taesd, 2023. 6
- [30] OpenAI. DALL-E 3. https://openai.com/dalle-3, 2024. Accessed: 2024-02-14. 3
- [31] Xingang Pan, Ayush Tewari, Thomas Leimk¨uhler, Lingjie Liu, Abhimitra Meka, and Christian Theobalt. Drag your gan: Interactive point-based manipulation on the generative image manifold. In ACM SIGGRAPH, 2023. 1, 2, 3, 14
- [32] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. In NeurIPS, 2019. 6
- [33] Or Patashnik, Zongze Wu, Eli Shechtman, Daniel Cohen-Or, and Dani Lischinski. Styleclip: Text-driven manipulation of stylegan imagery. In ICCV, 2021. 2

- [34] Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A Efros. Context encoders: Feature learning by inpainting. In CVPR, 2016. 3
- [35] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 1, 2, 3, 6, 12
- [36] Runway. RunwayML. https://runwayml.com, 2024. Accessed: 2024-02-14. 3
- [37] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022. 2
- [38] Scott Schaefer, Travis McPhail, and Joe Warren. Image deformation using moving least squares. In ACM SIGGRAPH,

2006. 1

- [39] Thomas W Sederberg and Scott R Parry. Free-form deformation of solid geometric models. In ACM SIGGRAPH, 1986. 1
- [40] Yujun Shi, Chuhui Xue, Jiachun Pan, Wenqing Zhang, Vincent YF Tan, and Song Bai. Dragdiffusion: Harnessing diffusion models for interactive point-based image editing. arXiv preprint arXiv:2306.14435, 2023. 1, 2, 6, 7, 8
- [41] Yujun Shi, Jun Hao Liew, Hanshu Yan, Vincent YF Tan, and Jiashi Feng. Instadrag: Lightning fast and accurate dragbased image editing emerging from videos. arXiv preprint arXiv:2405.13722, 2024. 3, 14
- [42] Joonghyuk Shin, Daehyeon Choi, and Jaesik Park. Instantdrag: Improving interactivity in drag-based image editing. TOG, 2024. 3
- [43] Roman Suvorov, Elizaveta Logacheva, Anton Mashikhin, Anastasia Remizova, Arsenii Ashukha, Aleksei Silvestrov, Naejin Kong, Harshith Goka, Kiwoong Park, and Victor Lempitsky. Resolution-robust large mask inpainting with fourier convolutions. In WACV, 2022. 2, 3, 12
- [44] Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence from image diffusion. In NeurIPS, 2024. 6
- [45] Tengfei Wang, Yong Zhang, Yanbo Fan, Jue Wang, and Qifeng Chen. High-fidelity gan inversion for image attribute editing. In CVPR, 2022. 2
- [46] Weihao Xia, Yulun Zhang, Yujiu Yang, Jing-Hao Xue, Bolei Zhou, and Ming-Hsuan Yang. Gan inversion: A survey. IEEE TPAMI, 2022.
- [47] Tao Yang, Peiran Ren, Xuansong Xie, and Lei Zhang. Gan prior embedded network for blind face restoration in the wild. In CVPR, 2021. 2
- [48] Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S Huang. Generative image inpainting with contextual attention. In CVPR, 2018. 3
- [49] Jiahui Yu, Zhe Lin, Jimei Yang, Xiaohui Shen, Xin Lu, and Thomas S Huang. Free-form image inpainting with gated convolution. In ICCV, 2019. 12
- [50] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 6

- [51] Zhuoyang Zhang, Han Cai, and Song Han. Efficientvit-sam: Accelerated segment anything model without performance loss. In CVPR, 2024. 6
- [52] Xuanjia Zhao, Jian Guan, Congyi Fan, Dongli Xu, Youtian Lin, Haiwei Pan, and Pengming Feng. Fastdrag: Manipulate anything in one step. In NeurIPS, 2024. 1, 3, 7, 14
- [53] Jun-Yan Zhu, Philipp Kr¨ahenb¨uhl, Eli Shechtman, and Alexei A Efros. Generative visual manipulation on the natural image manifold. In ECCV, 2016. 2

## Inpaint4Drag: Repurposing Inpainting Models for Drag-Based Image Editing via Bidirectional Warping

– Supplementary Material –

Jingyi Lu Kai Han† Visual AI Lab, The University of Hong Kong

lujingyi@connect.hku.hk, kaihanx@hku.hk

### Contents

- S1. Supplementary Videos . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- S2. Integration with More Inpainting Methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12
- S3. Multi-round Interactive Editing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 13
- S4. Discussion of Input Ambiguity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- S5. More Qualitative Results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 15
- S6. Pseudo Code for Inpaint4Drag . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- S7. Limitations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

### S1. Supplementary Videos

Two supplementary videos are available on our project page https://visual-ai.github.io/inpaint4drag: one demonstrating our bidirectional warping algorithm with visualizations, and another showcasing the real-time user interface interaction.

### S2. Integration with More Inpainting Methods

We integrate our framework with diverse image inpainting approaches, from early methods like LaMa [43] and DeepFillv2 [49] to recent generative model-based techniques [35]. While quantitative metrics in Tab. S1 show comparable drag editing performance across methods, qualitative differences emerge in Fig. S1. Early approaches offer computational efficiency, whereas generative methods sometimes produce more realistic results–a quality distinction not fully captured by existing metrics. Our framework provides users flexibility to select the inpainting method best suited to their specific requirements, balancing computational resources and visual fidelity.

Method DragBench-S DragBench-D Mem(GB)↓ Time(s)↓ MD↓ LPIPS↓ MD↓ LPIPS↓

DeepFillv2 [49] 0.8 0.05 3.2 13.7 3.7 9.0 LaMa [43] 1.1 0.07 3.4 13.6 3.7 9.0 SD-XL-Inpaint [35] 8.1 1.3 3.2 12.5 3.8 8.8 SD-1.5-Inpaint [35] 2.7 0.3 3.6 11.4 3.9 9.1

Table S1. Comparison of different inpainting methods. MD and LPIPS values are scaled by 100. Time and GPU memory are measured at 512×512 resolution.

Input Preview DeepFillv2 LaMa SD-XL-Inpaint SD-1.5-Inpaint

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

- Figure S1. Qualitative comparison of different inpainting methods. The figure illustrates how various inpainting approaches affect drag editing results, highlighting differences in visual fidelity, artifact handling, and preservation of semantic content across traditional and generative model-based techniques.

### S3. Multi-round Interactive Editing

Our system enables fluid multi-round interactions, allowing users to execute sequential edits with minimal delay. In Fig. S2, we demonstrate this capability through a chess sequence where pieces are repositioned in rapid succession to create a checkmate scenario–highlighting our system’s responsiveness to iterative user inputs.

[Figure 188]

- Figure S2. Multi-round interactive drag editing demonstrated through a three-move checkmate sequence including five consecutive edits. Users select deformable regions (chess pieces) and drag them from handle points to target positions. Grid overlays in the preview columns indicate areas requiring inpainting. Our method provides real-time preview (∼10ms) of the warping effect, followed by high-quality inpainting results (∼0.3s). Existing approaches typically require minutes for inference and fail during the initial interaction.

### S4. Discussion of Input Ambiguity

Previous drag editing methods [31, 41, 52] typically use sparse control points to guide deformation, with optional masking to restrict editable regions. However, this sparse input format (shown on the left of each row in Fig. S3) introduces fundamental ambiguity in deformation interpretation. Through our explicit region-based control (visualized in bottom-right insets), we demonstrate how a single ambiguous drag input can be precisely controlled to achieve five distinct editing intentions - from local manipulation to global translation. For instance, the same drag operation on a polar bear can be accurately interpreted as body translation, forearm bending, hand raising, upper body stretching, or scene translation. We address this ambiguity by requesting users to specify deformable regions through masking, treating each region as an elastic material where movement smoothly propagates from control points throughout the connected area.

Sparse Input Translate body Bend forearm Raise hand Stretch upper body Translate scene

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

Sparse Input Translate arm Bend front body Translate body Bend arm Translate scene

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

Sparse Input Translate head Squeeze head Rotate head Stretch right face Translate sculpture

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

- Figure S3. Precise control over ambiguous drag operations. Left: Ambiguous sparse input from previous methods can represent at least five different user intentions. Right: Through our explicit deformation-based control interface (bottom-right insets), we precisely implement each distinct user intention, effectively eliminating ambiguity while maintaining intuitive interaction.

### S5. More Qualitative Results

We present extensive qualitative results in Figs. S4 to S6. Our method allows users to specify handle points (red) and target points (blue) with arrows defining deformation regions (highlighted in red). By applying elastic material principles directly in pixel space, we achieve superior performance across diverse editing scenarios. The results demonstrate our method’s effectiveness in facial edits, large-scale deformations, and precise local manipulations while maintaining geometric stability. This advantage is particularly evident when handling significant boundary changes and occlusions, where our inpainting models realistically complete both texture and background.

Sparse Input FastDrag DragDiff. DiffEditor SDE-Drag Our Input Preview Inpainted

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

Figure S4. Qualitative comparison of Inpaint4Drag with state-of-the-art methods: wildlife, artworks, flowers, birds, and landscapes.

Sparse Input FastDrag DragDiff. DiffEditor SDE-Drag Our Input Preview Inpainted

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

- Figure S5. Qualitative comparison of Inpaint4Drag with state-of-the-art methods: portraits, interiors, statues, wildlife, still life, and sports.

- Sparse Input FastDrag DragDiff. DiffEditor SDE-Drag Our Input Preview Inpainted
- Figure S6. Qualitative comparison of Inpaint4Drag with state-of-the-art methods: urban scenes, landscapes, animals, pets, and reptiles.

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

### S6. Pseudo Code for Inpaint4Drag

To complement the detailed description of our method of the main paper, we provide a concise algorithmic representation of the data flow in Algorithm 1.

Algorithm 1: Inpaint4Drag: Drag-based Image Editing via Bidirectional Warping and Inpainting

Input: Image I, user-drawn mask M, handle points {hi} and target positions {ti} Output: Edited image Iedit Region Specification and Boundary Refinement:;

Ps ← SampleGridPoints(M) ; // Sample grid points from user mask Mpred ← fSAM(I,Ps) ; // SAM prediction Mdilated ← Dilate(M, K1); Meroded ← Erode(M, K1) ; // Create boundary constraints M ← (Mpred ∩ Mdilated) ∪ Meroded ; // Boundary-guided refinement Bidirectional Warping:;

C ← ExtractContours(M) ; // Get deformable regions foreach contour C ∈ C do

Associate control points: (hi,ti) ← {(hi,ti) | hi inside C}; Forward Warping: ; // Define target region boundary foreach point p in C do wi ← 1/(∥p−h

i∥+ϵ)

j 1/(∥p−hj∥+ϵ); pt ← p + i wi(ti − hi) ; // Weighted interpolation Store mapping pair (p,pt);

end C′ ← transformed contour from forward warping; Backward Mapping: ; // Ensure complete pixel coverage foreach pixel pt within boundary of C′ do

Find Nn nearest matched pixels {ptgti } with source positions {psrci }; ps ← pt + Ni=1n wi(psrci − ptgti ) ; // Local neighborhood interpolation if ps ∈ [0,W) × [0,H) and pt ∈ [0,W) × [0,H) then

Store valid mapping (ps,pt); end

end

end Compute Warped Image and Inpainting Mask:; foreach valid mapping pair (ps,pt) do

Iwarped(pt) ← I(ps) ; // Transfer pixel values

end Mwarped ← mask of pixels filled in warped image; Mtemp ← M \ Mwarped; ∂Mwarped ← boundary of Mwarped ; // Identify unmapped regions Minpaint ← Dilate(Mtemp ∪ ∂Mwarped,K2) ; // Create buffer zone Image Inpainting:; Iedit ← Inpaint(Iwarped,Minpaint) ; // Apply inpainting model return Iedit

### S7. Limitations

While our method achieves significant improvements in efficiency and precision, it relies on accurate user-specified masks and control points for optimal performance. Imprecise user inputs, such as masks that inadvertently include background elements or poorly positioned control points, can lead to undesired deformation artifacts. Future work could explore understandingenabled models that automatically filter irrelevant background elements or provide intelligent suggestions for mask and control point placement, reducing the burden on users to provide perfectly accurate inputs while maintaining the intuitive nature of drag-based interaction.

