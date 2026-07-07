## NANO3D: A TRAINING-FREE APPROACH FOR EFFICIENT 3D EDITING WITHOUT MASKS

#### Junliang Ye1∗ Shenghao Xie2,1∗ Ruowen Zhao1 Zhengyi Wang1 Hongyu Yan4 Wenqiang Zu5,2 Lei Ma2† Jun Zhu1,3† 1Tsinghua University 2Peking University 3ShengShu 4HKUST 5CASIA

[Figure 1]

[Figure 2]

User: replace the chicken with a dog

[Figure 3]

[Figure 4]

FlowEdit

[Figure 5]

Editing Prompt

[Figure 6]

User: Replacing the eagle head with lion head

[Figure 7]

User: Removing the chair

User: Holding a sword

# arXiv:2510.15019v1[cs.CV]16Oct2025

Voxel Merge

### …

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

SLat Merge

[Figure 12]

3D assets w/o mask

Nano3D

Task-1:Removing Task-2:Adding Task-3: Replacing

Output

Input

Figure 1: Highly-consistent 3D objects edited by Nano3D. Our framework supports a range of training-free and part-level tasks especially on shape, including removal, addition, and replacement, while only requiring users to provide source 3D objects and instructions, without any mask.

ABSTRACT

3D object editing is essential for interactive content creation in gaming, animation, and robotics, yet current approaches remain inefficient, inconsistent, and often fail to preserve unedited regions. Most methods rely on editing multi-view renderings followed by reconstruction, which introduces artifacts and limits practicality. To address these challenges, we propose Nano3D, a training-free framework for precise and coherent 3D object editing without masks. Nano3D integrates FlowEdit into TRELLIS to perform localized edits guided by front-view renderings, and further introduces region-aware merging strategies, Voxel/Slat-Merge, which adaptively preserve structural fidelity by ensuring consistency between edited and unedited areas. Experiments demonstrate that Nano3D achieves superior 3D consistency and visual quality compared with existing methods. Based on this framework, we construct the first large-scale 3D editing datasets Nano3DEdit-100k, which contains over 100,000 high-quality 3D editing pairs. This work addresses long-standing challenges in both algorithm design and data availability, significantly improving the generality and reliability of 3D editing, and laying the groundwork for the development of feed-forward 3D editing models.

Project Page:https://jamesyjl.github.io/Nano3D

1 INTRODUCTION

Generative models for 3D asset creation have made tremendous progress Lai et al. (2025); Chen et al. (2025d; 2024d); Wang et al. (2023), leading to widespread applications across entertainment, robotics, and healthcare. In particular, recent rectified flows (reflows) Liu et al. (2022), such as TRELLIS Xiang et al. (2025), achieve high-quality 3D object generation by embedding heterogeneous representations into a unified latent space while explicitly disentangling geometry and appearance. Beyond generation, editing (i.e., revising the intended region while keeping other regions

∗Equal contribution †Corresponding author.

unchanged) is also valuable as users usually need to refine existing assets rather than regenerate entirely new ones, which requires multiple unpredictable iterations to obtain a satisfactory result. For example, TRELLIS can generates diverse plausible appearances easily with style-modified text or image prompts, such as texture and material, but fail to reliably repeat identical geometries.

In image editing, an increasing number of powerful models have recently emerged, including GPT4o Hurst et al. (2024), Flux.1 Kontext Labs et al. (2025), and Nano Banana Fortin et al. (2025). A closer look at the evolution of these models reveals a clear three-stage development paradigm. Stage 1 introduced training-free image editing algorithms Hertz et al. (2022), which demonstrated the feasibility of editing without model finetuning. Stage 2 focused on the automatic construction of large-scale, high-quality paired editing datasets, providing the foundation for supervised learning Brooks et al. (2023). Stage 3 leveraged these datasets to train feedforward image editing models capable of real-time inference and high fidelity generation.

In contrast, 3D object editing still remains bottlenecked in the initial stage (i.e., algorithm). Specifically, existing methods, such as those based on Score Distillation Sampling (SDS) Sella et al. (2023) or the “multi-view editing then reconstruction” paradigm Qi et al. (2024), struggle to maintain consistency across views or attributes and usually demand time-consuming optimization. This leaves us wondering: can 3D objects be edited versatilely, efficiently and consistently in a training-free manner using only pretrained generative models, as achieved in 2D images? Resolving this problem will allow 3D object editing to enter a virtuous cycle of data expansion and training models capable of flexible asset customization, thereby accelerating the whole field toward maturity like 2D images.

We propose Nano3D, a training-free 3D editing algorithm designed for constructing paired 3D editing datasets. Drawing inspiration from the training-free 2D editing method FlowEdit Kulikov et al.

- (2024), Nano3D leverages the first stage of TRELLIS to generate an iterative trajectory from input to edited voxel representations, thereby enabling efficient training-free 3D editing.

To further enhance source consistency between the original and edited objects, we introduce a region-aware merging strategy, Voxel/Slat-Merge, applied after TRELLIS’s two-stage geometry and appearance editing. Based on simple connectivity analysis, this strategy adaptively identifies modified voxel regions in the edited 3D object and integrates them back into the original object. This effectively merges the edited content while preserving the structure of unedited regions.

Building on the Nano3D algorithm, we design an efficient pipeline for large-scale construction of 3D editing datasets and generate a high-quality dataset of 100,000 samples——Nano3D-Edit-100k. Our work addresses two long-standing gaps in the 3D editing domain—the lack of training-free editing algorithms and the absence of large-scale datasets—thereby laying a solid foundation for the third stage of 3D editing: training feedforward models under 3D editing supervision.

Overall, our contributions can be summarized as follows:

- • We make the first attempt to introduce FlowEdit to 3D object editing, demonstrating that the powerful priors of 3D object generative models can also support effective training-free editing (like 2D images)
- • We propose Voxel/Slat-Merge, a region-aware merging strategy that automatically preserves source consistency in the non-edited regions of 3D objects.
- • We develop a user-friendly 3D editing framework, Nano3D, which achieves state-of-the-art editing performance without requiring any manual masks.
- • Building upon Nano3D, we curate the first large-scale 3D editing dataset Nano3D-Edit-100k, comprising over 100,000 high-quality samples to support further research and development.

- 2 RELATED WORK

- 2.1 2D IMAGE EDITING

With the advent of large-scale 2D generative models, image editing has shifted from manual pixellevel operations to controllable semantic-level manipulation. Early approaches modify noisy latents via inversion to balance new details with original structures Meng et al. (2021); Mokady et al. (2023);

Abdal et al. (2019), while others finetune generative models on curated editing pairs to enable instruction following Brooks et al. (2023); Wei et al. (2024); Sheynin et al. (2024). Localized editing has also been explored through attention map manipulation Hertz et al. (2022); Tumanyan et al.

- (2023); Couairon et al. (2022), and adapters have been introduced to inject additional conditions for enhanced controllability Ye et al. (2023); Ju et al. (2024); Mou et al. (2024). More recently, rectified flows (reflow) Liu et al. (2022); Esser et al. (2024) have enabled high-fidelity synthesis with few sampling steps. To support reflow-based editing, RFSolver Wang et al. (2024a) approximates ODEs via higher-order Taylor expansion while preserving structures through attention replacement, whereas FlowEdit Kulikov et al. (2024) introduces an inversion-free strategy by interpolating between sampled noise and the source image.

2.2 RELATED WORK ABOUT 3D OBJECT GENERATION

Over the past years, 3D object generation has been pursued under several paradigms.Generative Adversarial Net (GAN)-based approaches are among the earliest to directly model 3D distributions Goodfellow et al. (2014); Chan et al. (2022); Gao et al. (2022); Weng et al. (2024b); Zheng et al. (2022). Diffusion models later become popular by treating 3D representations as denoising tasks, which provides improved training stability and superior generative fidelity Chen et al.

- (2023a); Wang et al. (2023); Ye et al. (2024); Liu et al. (2025b); Wang et al. (2024b); Liu et al.

(2025a). At the same time, autoregressive (AR) models Weng et al. (2024a); Zhao et al. (2025a); Chen et al. (2024d); Wei et al. (2025); Chen et al. (2025b) enable fine-grained conditional control and structured synthesis through ordered generation. More recently, reflow models formulate 3D generation as a continuous ODE-based linear transformation, enabling faster inference with competitive performance Xiang et al. (2025); Hunyuan3D et al. (2025); Li et al. (2025c); Chen et al. (2025c); Ye et al. (2025a).

2.3 3D OBJECT EDITING

Compared to 2D image editing, maintaining spatial consistency is substantially more challenging in 3D. Many approaches adopt score distillation sampling (SDS) Poole et al. (2022) to optimize 3D representations using gradients from pretrained 2D diffusion models Sella et al. (2023); Li et al.

- (2024); Chen et al. (2024c); Palandra et al. (2024); Chen et al. (2023b). Others edit multi-view images and reconstruct them with large reconstruction models (LRMs) Qi et al. (2024); Chen et al.

- (2024a); Barda et al. (2025); Huang et al. (2025); Erko¸c et al. (2025); Bar-On et al. (2025); Zheng et al. (2025); Li et al. (2025b); Gao et al. (2024), or directly manipulate triplanes as a bridge between

- 2D and 3D Kathare et al. (2025); Bilecen et al. (2025). Inspired by InstructPix2Pix Brooks et al.

(2023), several works construct paired 3D editing datasets for supervised training Ye et al. (2025b); Xu et al. (2023). To enable finer control, diverse conditions such as sketches Mikaeili et al. (2023); Liu et al. (2024); Guillard et al. (2021), part-level masks Chen et al. (2025a); Yang et al. (2025a;b); Li et al. (2025a), and point-based dragging Chen et al. (2024b); Xie et al. (2023); Lu et al. (2025) have been explored. More recently, rectified flows (reflow) Zhao et al. (2025b); Li et al. (2025c); He et al. (2025) achieved large-scale 3D generation and zero-shot appearance editing, yet still face bottlenecks in shape modification. In this work, we unlock their potential for versatile and consistent

- 3D editing in a training-free and user-friendly manner.

- 3 PRELIMINARY

- 3.1 FLOWEDIT

FlowEdit is a text-guided image editing method tailored for text-to-image flow models. It is characterized by being inversion-free, optimization-free, and model-agnostic. Rather than relying on traditional inversion-reconstruction processes that often introduce distortion, FlowEdit constructs an ordinary differential equation (ODE) trajectory in the latent space from the source prompt to the target prompt. This trajectory enables direct evolution of image representations over the velocity field. By leveraging a weighted combination of the source and target velocity fields, FlowEdit ensures a shorter editing path and stronger structural preservation throughout the editing process.

- 3.2 TRELLIS

Trellis represents a 3D asset using Structured Latents (SLAT) — a set of activated voxels, each associated with a local latent vector, denoted as (zi,pi). Here, each zi ∈ RC jointly encodes both geometric and appearance information in a compact yet expressive manner. The inference pipeline consists of two successive stages. In the Structure Prediction (ST) stage, a 643 voxel grid is employed to estimate occupancy, thereby producing a sparse structural representation of the asset. In the Sparse Latent (SLAT) stage, local latent vectors are further inferred for the voxels identified in the first stage, capturing richer semantic and geometric details. Finally, a powerful SLAT-VAE decoder reconstructs the complete 3D asset from the predicted latent representations with high fidelity.

- 4 METHOD

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Overview

Render

[Figure 17]

Stage1: Voxel-based Structural Edit Stage2: Latent-based Structure Edit

[Figure 18]

[Figure 19]

Editing Prompt: Remove Image the wings of the dragon

[Figure 20]

Flow Edit

Voxel Merge

Voxelize

Edit SLat

Sparse Flow Latent Merge

Edit Voxel

3D assets Edited 3D assets

Voxel

[Figure 21]

[Figure 22]

0 Input 0

1 1

[Figure 23]

Stage1

Stage2

Output

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

Edit Voxel

SLat Merge

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

Sparse Flow

FlowEdit

[Figure 44]

[Figure 45]

…

…

- 0 Encode

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

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

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

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

- 1

1 0

1 1

1 0

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

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

Voxel Merge

Tar-SLat

Input

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

Remove the wings of the dragon

[Figure 164]

Dino

Gemini2.5-FlashImage

Image Embeddings

Input Output Merge Src-SLat

Edit SLat

Edit Image

Image

Edit Image

- Figure 2: The Nano3D pipeline. The original 3D object is voxelized and encoded into sparse structure and structured latent respectively. Stage 1 modifies geometry via Flow Transformer with FlowEdit, guided by Nano Banana–edited images. Stage 2 generates structured latents with Sparse Flow Transformer, supporting TRELLIS-inherent appearance editing. Voxel/Slat-Merge further ensures consistency across both stages before decoding the final 3D object.

- 4.1 OVERVIEW

A common approach is to edit rendered images of a 3D object and reconstruct it with a generative model, but this often breaks geometric consistency. To address this, we introduce FlowEdit into the first-stage generation of TRELLIS (Sec. 4.2). To further ensure geometric and appearance consistency, we propose Voxel/Slat-Merge (Sec. 4.3), which detects edited regions and integrates them with unedited ones. Finally, we present a training-free, user-friendly pipeline (Sec. 4.4) that also supports large-scale dataset construction. By combining FlowEdit with Voxel/Slat-Merge, Nano3D achieves geometrically consistent and semantically faithful 3D object editing within TRELLIS.

- 4.2 FLOWEDIT

Inspired by FlowEdit’s success in 2D image editing, we extend it to 3D object editing by integrating it into TRELLIS stage 1, leveraging the pretrained generative prior to establish an editing path between source and target objects instead of starting from noise. Specifically, we use input 3D object’s front view as source condition and Nana-Banana–edited image as the target. From the voxelized input, FlowEdit’s iterative inference generate the edited voxel output.

- 4.3 VOXEL/SLAT-MERGE

Voxel-Merge We observe that voxels produced by FlowEdit do not always preserve structural consistency between the input and the edited asset. For example, when editing a dragon to remove its

wings, the resulting voxels may not only modify the wings but also inadvertently alter other unrelated regions. To effectively mitigate this issue, we introduce a region-aware merging strategy, Voxel-Merge, applied at the end of the geometry editing stage. This method explicitly defines a difference map g via a voxel-wise XOR operation between sparse structures:

g(i) = ssrc(i) ⊕ stgt(i) =

1, if ssrc(i) ̸= stgt(i), 0, if ssrc(i) = stgt(i). ∀i (1)

Under different connectivities (e.g., 6/18/26-neighborhoods), g is decomposed to several components {gj}Cj=1, where |gj| is the number of voxels. After the descending sorting, we adaptively select top-k components and construct the flip mask m = {mj}Cj=1, where mj = {1}|g

j| if j ∈ Ik and mj = {0}|g

j| otherwise. Alternatively, components larger than a threshold τ can be selected, i.e., mj = {1}|g

j| if |gj| > τ, which ignores small noisy and irrelevant regions, ensuring only significant discrepancies (i.e., edited regions) are masked. Then we perform ssrc⊕M to transfer edited regions of stgt onto the source 3D object, and inherently ensure non-edited regions unchanged.

SLat-Merge After applying Voxel-Merge, we obtain voxels that are fully consistent before and after editing. We then feed the edited image together with the merged voxels into the second stage of TRELLIS to generate SLat representations. To ensure that the generated SLat features remain consistent with those encoded from the original 3D asset, we further introduce SLat-Merge. Specifically, this module continues to utilize the previously defined mask to perform zsrc ⊕ M, thereby merging the edited and non-edited regions within the structured latent space.

- 4.4 NANO3D

As illustrated in Fig. 2, Nano3D builds upon TRELLIS to enable decoupled geometry and appearance editing of 3D objects. The input object is voxelized and, along with DINOv2 Oquab et al. (2023) features, encoded into a structured latent representation via a VAE Kingma & Welling (2013). Meanwhile, we use Nano Banana with the front view of a 3D asset and editing instructions as input to generate the edited front view. In TRELLIS-Stage 1, we replace the standard flow iteration with FlowEdit, which takes the source object’s voxel and the before/after front views as input, and outputs the edited voxel. We then apply Voxel-Merge to ensure geometric consistency. In TRELLIS-Stage 2, the edited voxel and edited front view jointly guide TRELLIS to generate the final SLat. At this stage, we further adopt Slat-Merge to guarantee both geometric and texture consistency. Finally, the edited SLat is decoded by the VAE to reconstruct the target 3D object.

Data Construction Pipeline. As illustrated in Fig. 3, we extend Nano3D by constructing a complete and streamlined 3D editing data generation pipeline. The process consists of the following stages:

Replace <the chicken> with <a dog>.

[Figure 165]

Qwen2.5-vl Yes No

Sample From Image Dataset

Edited Instruction

Data Filter

[Figure 166]

[Figure 167]

Nano3D

Trellis Source Image

[Figure 168]

[Figure 169]

Source Mesh

Edited Mesh

- Figure 3: Data Construction Pipeline. The figure shows our pipeline. We first sample images from the dataset and prompt Qwen2.5-VL to generate editing instructions by completing templates. Trellis then generates 3D meshes from the images. Finally, the image, instruction, and mesh are fed into Nano3D, and the resulting 3D assets are filtered for quality.

- 1. Image Sampling from Existing Datasets: We sample views from publicly available 3D asset datasets Xiang et al. (2025); Deitke et al. (2022). For each asset, the frontal view is selected as the editing target.
- 2. Instruction Generation via VLM: An editing instruction is automatically generated using the vision-language model Qwen-VL-2.5 Bai et al. (2025), based on three predefined prompt templates:

- • Add: add <something> to <somewhere>
- • Remove: remove <something> in <somewhere>
- • Replace: replace <something> with <something>

The model fills in these templates with visual context from the image to produce diverse and semantically grounded instructions.

- 3. 3D Asset Generation via TRELLIS: Given the selected image, we use TRELLIS to reconstruct the corresponding 3D asset. Instead of using the original mesh, we choose to regenerate the source mesh via TRELLIS for two reasons: (1) obtaining the structured latent (sLat) from the original mesh requires rendering ∼150 views, which is inefficient;

(2) the reconstructed sLat still diverges from the original mesh due to the inherent loss in TRELLIS’s VAE encoding. Using the TRELLIS-reconstructed mesh ensures consistency and reduces computational overhead.

- 4. Image Editing via Nano-Banana or Flux-Kontext: The generated instruction is input into Nano-Banana or Flux-Kontext to synthesize the edited target image.
- 5. 3D Editing via Nano3D: The original 3D asset, the source image, and the edited image are fed into Nano3D, which outputs an edited 3D asset.

- 5 EVALUATION

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

Add a shield to the penguin’s left hand.

Remove one leaf from the right side of the stem.

Remove the backpack from the character’s back.

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

Replace the shield in the character’s hand with a sword.

Remove the handle from the lamp.

Remove the chimney from the train.

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

Remove the dragon horn headpiece from the character’s head.

Add a satellite dish on top of the house.

Replace the jacket with a yellow down coat

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

- Figure 4: Qualitative results. We present three edit types—object removal, addition, and replacement. In each case, Nano3D confines changes to the target region (red dashed circles) and produces view-consistent edits, while leaving the rest of the scene unchanged. Geometry stays sharp and textures remain faithful in unedited areas, with no noticeable artifacts.

Add a basket to the front of the scooter.

Add a shield to the warrior’s hand.

Remove the bottom drawer.

[Figure 194]

[Figure 195]

6

[Figure 196]

[Figure 197]

Add trousers to the character.

Replace the front wheel with a wooden wheel.

Add a carrot to the snowman’s mouth.

Add a white flag to the mast of the boat.

Open the wooden door from the archway.

Replace the stack of green notebooks with a stack of books.

Make the backpack

Replace the headban d with a crown.

Holding a sword Remove the wings.

the boy is carrying bigger.

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

Input

[Figure 209]

[Figure 210]

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

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

Tailor3D

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

Vox-E

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

Trellis

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

Nano3D

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

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

- Figure 5: Qualitative comparison. Our method achieves the best editing performance with faithful instruction semantic alignment and perfect original structure consistency across multi-view images.

- 5.1 SETUP

Implementation Detail. Our method is implemented on TRELLIS. The sampling step is fixed at 25, and FlowEdit is configured with nmax = 15, nmin = 0, and navg = 5. The CFG guidance scales for vtθ(pt) and vtθ(qt) are set to 1.5 and 5.5, respectively, with λ set to 0.5. For both Voxel-Merge and Slat-Merge, τ is set to 100. For the construction of Nano3D-Edit-100k, we employ 32 A800 GPUs for inference, utilizing the Qwen2.5-vl-72B API to generate editing instructions and Flux-Kontext to perform image editing operations. The creation of each editing pair required approximately five minutes, and empirical observations revealed two key findings: first, the vast majority of failed cases originated from errors in the image editing stage, whereas successful adherence to instructions at this stage led to a very high success rate in the subsequent Nano3D editing process; second, the predominant computational cost arose from the Flexicube module, which consumed nearly 4.5 minutes per pair, while the preceding steps required only about 30 seconds. Based on these observations and in order to further reduce computational overhead, we adopted a storage strategy in which only the SLAT (Structured Latent) representation and the voxel sum of each asset are preserved, thereby allowing users to flexibly decide whether to directly train on SLAT or to employ Flexicube to convert SLAT into explicit GLB meshes for downstream applications. To improve dataset quality, we use Qwen2.5-VL-7B to automatically filter edited images based on instruction compliance. Non-compliant samples are returned to the pool for re-sampling.

Baseline. We select three representative state-of-the-art methods as baselines: Vox-E based on SDS, Tailor3D based on ”multi-view editing then reconstruction”, and TRELLIS, which leverages a RePaint-based method. For all baselines, we strictly follow their original implementations and use the official codebases to obtain the results reported in this paper.

Dataset. Our Nano3D-Edit-100k dataset comprises two sources of image data: images collected from the internet and rendered views from the Trellis-500K dataset. During dataset construction, we follow the methodology of 3D-Alpaca Ye et al. (2025b), employing Qwen2.5-VL to automatically annotate 3D assets and classify them accordingly. We then perform class balancing across ten distinct categories, ultimately selecting 100K image samples. We select 100 representative cases from the Nano3D-Edit-100k dataset for the experiments and demonstrations in this section.

Metric. We systematically evaluate the edited 3D objects from three perspectives: source structure preservation, target semantic alignment, and generation quality. For source structure preservation, we assess non-edited regions against the original 3D object using Chamfer Distance (CD) Fan et al. (2017). For target semantic alignment, we employ the DINO-I Caron et al. (2021) metric to quantify adherence to the target edited image. For generation quality, we use FID Heusel et al. (2017) on rendered multi-view images to measure fidelity and diversity.

- 5.2 MAIN RESULT

Qualitative Comparison. As shown in Fig. 5, Nano3D not only strictly follows editing instructions but also maintains perfect structure consistency with the source 3D object across multi-view images. In contrast, Tailor3D introduces noticeable geometry distortions and appearance artifacts. Vox-E produces results that are overly blurry, smoothed, and misaligned with the target semantic. TRELLIS, though showing relative improvements, still suffers from several issues, such as local detail corruption, shape enlargement, and incorrect orientation. These findings demonstrate that our method delivers impressive and steady visual effects beyond the reach of existing methods.

Table 1: Quantitative comparison. Our method achieves the best structure consistency, semantic alignment with the target edited image, and generation fidelity.

Method CD↓ DINO-I↑ FID↓

Tailor3D 0.037 0.759 140.93 Vox-E / 0.782 117.12 TRELLIS 0.019 0.901 49.57 Nano3D 0.013 0.950 27.85

Table 2: User study. Given that most users favored TRELLIS and Nano3D, the results for Tailor3D and Vox-E are omitted from the table for clarity. As shown in the table, our method is strongly preferred by participants, significantly outperforming TRELLIS.

Shape Preserv. TRELLIS 32% 21% 5% Nano3D 68% 79% 95%

Prompt Algn.

Visual Quality

Method

Quantitative Comparison. As shown in Tab. 1, Nano3D outperforms all baselines, achieving the lowest CD and FID and the highest DINO-I score, indicating superior structural consistency, perceptual quality, and semantic alignment, as seen in Fig. 5.

User Study. To assess editing quality and usability, we conducted a user study with 50 participants. Each round presented the original 3D object, task instructions, and results from Tailor3D, Vox-E, TRELLIS, and Nano3D. Participants selected the best method based on Prompt Alignment, Visual Quality, and Shape Preservation. As shown in Tab. 2, Nano3D received the highest preference across all criteria, demonstrating superior semantic alignment, visual quality, and shape fidelity. For clarity, Tailor3D and Vox-E results are omitted, as user choices mainly favored TRELLIS and Nano3D.

Nano3D-Edit-100k v.s. 3D-Alpaca. High-quality 3D editing requires consistency in both 2D image appearance and 3D structure—that is, the rendered images before and after editing should remain coherent, and the 3D assets themselves should preserve structural integrity throughout the editing process. The 3D-Alpaca dataset lacks both aspects, leading to significantly lower data quality compared to ours. To quantify this, we randomly sample 500 edited pairs from each dataset and evaluate text–image alignment using CLIPScore Hessel et al. (2021) and ViLT R-Precision Kim

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

###### Src Voxel Thres=100 Thres=50 Thres=30

- Figure 6: Ablation study on τ. The leftmost voxel represents the pre-editing state, with the editing instruction being to remove the wings. The three voxels on the right correspond to the masks generated during the voxel-merge stage for τ = 100, τ = 50, and τ = 30 (from left to right). As observed, when τ = 100, the detected mask most accurately aligns with the editing regions, while lower values include irrelevant non-editing areas.

et al. (2021). Specifically, we use a VLM to infer the caption of the edited asset based on the original asset’s caption and the editing instruction. As shown in Table 3, our Nano3D-Edit-100k consistently outperforms 3D-Alpaca across all metrics.

Table 3: Semantic alignment comparison between NANO3D-EDIT-100K and 3D-Alpaca.

CLIPScore ViLT R-Precision R@5 ViLT R-Precision R@10

3D-Alpaca 28.42 33.6 40.2 Nano3D-Edit-100k 39.71 45.3 52.4

- 5.3 ABLATION STUDY

+FlowEdit +VoxleMerge

+FlowEdit +VoxleMerge +SLatMerge

+FlowEdit

Source Mesh

replace the hat with a helmet

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

remove the chimney replace the chicken

with a dog

add a monkey on the chair

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

- Figure 7: Ablation study on Voxel/Slage-Merge. Our methods sequentially ensure geometry and appearance consistency, demonstrating their complementary roles.

Voxel/Slat-Merge. We sequentially validate the effectiveness of Voxel-Merge and Slat-Merge strategies. As shown in Fig. 7, relying solely on FlowEdit leads to geometry misalignments and

deformations, accompanied by missing, blurred, and distorted appearances, resulting in obvious inconsistencies with the original 3D object. Incorporating Voxel-Merge substantially improves the overall performance, restoring geometry and enhancing cross-view global consistency, but leaving appearance issues unresolved. With the additional incorporation of Slat-Merge, local visual quality is further enhanced, and appearances exhibit greater consistency before and after editing. These results indicate that our methods effectively exploit the advantage of geometry-appearance decoupling in 3D objects, ensuring more reliable consistency. Ablation on τ. We further compare different val-

[Figure 340]

- Figure 8: Comparison with Editing Methods Requiring Masked Input Comparison between Nano3D and VoxHammer under identical mask-guided editing settings. Both methods receive the same bounding boxes, editing instructions, and target outputs. Nano3D preserves high consistency in non-edited regions, while VoxHammer exhibits inconsistencies compared to the input 3D assets in these areas, as evidenced by the red-bordered highlights.

ues of τ, as shown in Fig. 6. The leftmost voxel represents the pre-editing state, with the editing instruction being to remove the wings. The three voxels on the right correspond to the masks generated during the voxel-merge stage for τ = 100, τ = 50, and τ = 30 (from left to right). As observed, when τ = 100, the detected mask most accurately aligns with the editing regions, while lower values include irrelevant non-editing areas.

- 5.4 COMPARISON WITH EDITING METHODS REQUIRING MASKED INPUT

We extend Nano3D to support user-provided masks for controlling editing regions. Experimental results demonstrate that mask-guided editing further improves Nano3D’s performance. To validate its effectiveness, we compare Nano3D with VoxHammer Li et al. (2025a), a state-of-the-art editing method that also requires mask inputs. As shown in Figure 8, both methods receive identical bounding boxes, editing instructions, and target edited images. Nano3D achieves high consistency in nonedited regions, attributed to the robust capability of FlowEdit and the effectiveness of our voxel/slat merge algorithm. In contrast, VoxHammer exhibits noticeable inconsistencies in non-edited areas

before and after editing. To emphasize this difference, we outline the inconsistent regions with red borders for direct visual comparison. The comparison demonstrates that our algorithm not only offers broader applicability—achieving high-consistency 3D editing without requiring user-provided masks—but also maintains state-of-the-art performance when mask inputs are provided.

- 6 CONCLUSION

In this work, we present Nano3D, a training-free and user-friendly framework for localized 3D object editing, supporting operations such as object removal, addition, and replacement. By integrating FlowEdit into the TRELLIS pipeline and introducing region-aware merging strategies (Voxel/SlatMerge), Nano3D achieves geometrically consistent and semantically faithful edits. Extensive experiments demonstrate its state-of-the-art performance across diverse editing tasks. Furthermore, we construct Nano3D-Edit-100k, the first large-scale dataset tailored for 3D editing, enabling future research on feedforward DiT-based editing models.

Limitation. Nano3D demonstrates strong performance in 3D editing tasks, but has the following limitations: it supports only localized edits; the VAE in TRELLIS introduces reconstruction loss; and the overall performance is constrained by TRELLIS’s generative capacity. We view these limitations as important directions for future research.

REFERENCES

Rameen Abdal, Yipeng Qin, and Peter Wonka. Image2stylegan: How to embed images into the stylegan latent space? In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4432–4441, 2019.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. URL https://arxiv.org/abs/2502.13923.

Roi Bar-On, Dana Cohen-Bar, and Daniel Cohen-Or. Editp23: 3d editing via propagation of image prompts to multi-view. arXiv preprint arXiv:2506.20652, 2025.

Amir Barda, Matheus Gadelha, Vladimir G Kim, Noam Aigerman, Amit H Bermano, and Thibault Groueix. Instant3dit: Multiview inpainting for fast editing of 3d objects. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 16273–16282, 2025.

Bahri Batuhan Bilecen, Yigit Yalin, Ning Yu, and Aysegul Dundar. Reference-based 3d-aware image editing with triplanes. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 5904–5915, 2025.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 18392–18402, 2023.

Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´e J´egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9650–9660, 2021.

Eric R. Chan, Connor Z. Lin, Matthew A. Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas Guibas, Jonathan Tremblay, Sameh Khamis, Tero Karras, and Gordon Wetzstein. Efficient geometry-aware 3d generative adversarial networks, 2022. URL https: //arxiv.org/abs/2112.07945.

Hansheng Chen, Ruoxi Shi, Yulin Liu, Bokui Shen, Jiayuan Gu, Gordon Wetzstein, Hao Su, and Leonidas Guibas. Generic 3d diffusion adapter using controlled multi-view editing. arXiv preprint arXiv:2403.12032, 2024a.

Honghua Chen, Yushi Lan, Yongwei Chen, Yifan Zhou, and Xingang Pan. Mvdrag3d: Dragbased creative 3d editing via multi-view generation-reconstruction priors. arXiv preprint arXiv:2410.16272, 2024b.

Minghao Chen, Junyu Xie, Iro Laina, and Andrea Vedaldi. Shap-editor: Instruction-guided latent 3d editing in seconds. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 26456–26466, 2024c.

Minghao Chen, Roman Shapovalov, Iro Laina, Tom Monnier, Jianyuan Wang, David Novotny, and Andrea Vedaldi. Partgen: Part-level 3d generation and reconstruction with multi-view diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 5881– 5892, 2025a.

Minghao Chen, Jianyuan Wang, Roman Shapovalov, Tom Monnier, Hyunyoung Jung, Dilin Wang, Rakesh Ranjan, Iro Laina, and Andrea Vedaldi. Autopartgen: Autogressive 3d part generation and discovery, 2025b. URL https://arxiv.org/abs/2507.13346.

Rui Chen, Yongwei Chen, Ningxin Jiao, and Kui Jia. Fantasia3d: Disentangling geometry and appearance for high-quality text-to-3d content creation. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 22246–22256, 2023a.

Yige Chen, Teng Hu, Yizhe Tang, Siyuan Chen, Ang Chen, and Ran Yi. Plasticine3d: 3d non-rigid editing with text guidance by multi-view embedding optimization. arXiv preprint arXiv:2312.10111, 2023b.

Yiwen Chen, Yikai Wang, Yihao Luo, Zhengyi Wang, Zilong Chen, Jun Zhu, Chi Zhang, and Guosheng Lin. Meshanything v2: Artist-created mesh generation with adjacent mesh tokenization. arXiv preprint arXiv:2408.02555, 2024d.

Yiwen Chen, Zhihao Li, Yikai Wang, Hu Zhang, Qin Li, Chi Zhang, and Guosheng Lin. Ultra3d: Efficient and high-fidelity 3d generation with part attention. arXiv preprint arXiv:2507.17745, 2025c.

Zhaoxi Chen, Jiaxiang Tang, Yuhao Dong, Ziang Cao, Fangzhou Hong, Yushi Lan, Tengfei Wang, Haozhe Xie, Tong Wu, Shunsuke Saito, et al. 3dtopia-xl: Scaling high-quality 3d asset generation via primitive diffusion. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 26576–26586, 2025d.

Guillaume Couairon, Jakob Verbeek, Holger Schwenk, and Matthieu Cord. Diffedit: Diffusionbased semantic image editing with mask guidance. arXiv preprint arXiv:2210.11427, 2022.

Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects, 2022. URL https://arxiv.org/abs/2212.08051.

Ziya Erkoc¸, Can G¨umeli, Chaoyang Wang, Matthias Nießner, Angela Dai, Peter Wonka, Hsin-Ying Lee, and Peiye Zhuang. Preditor3d: Fast and precise 3d shape editing. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 640–649, 2025.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Haoqiang Fan, Hao Su, and Leonidas J Guibas. A point set generation network for 3d object reconstruction from a single image. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 605–613, 2017.

Alisa Fortin, Guillaume Vernade, Kat Kampf, and Ammaar Reshi. Introducing Gemini 2.5 Flash Image, our state-of-the-art image model. Blog Post, August 2025. URL https://developers. googleblog.com/en/introducing-gemini-2-5-flash-image/. Accessed: 2025-09-12.

Jun Gao, Tianchang Shen, Zian Wang, Wenzheng Chen, Kangxue Yin, Daiqing Li, Or Litany, Zan Gojcic, and Sanja Fidler. Get3d: A generative model of high quality 3d textured shapes learned from images, 2022. URL https://arxiv.org/abs/2209.11163.

Will Gao, Dilin Wang, Yuchen Fan, Aljaz Bozic, Tuur Stuyck, Zhengqin Li, Zhao Dong, Rakesh Ranjan, and Nikolaos Sarafianos. 3d mesh editing using masked lrms. arXiv preprint arXiv:2412.08641, 2024.

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks, 2014. URL https: //arxiv.org/abs/1406.2661.

Benoit Guillard, Edoardo Remelli, Pierre Yvernay, and Pascal Fua. Sketch2mesh: Reconstructing and editing 3d shapes from sketches. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 13023–13032, 2021.

Xianglong He, Zi-Xin Zou, Chia-Hao Chen, Yuan-Chen Guo, Ding Liang, Chun Yuan, Wanli Ouyang, Yan-Pei Cao, and Yangguang Li. Sparseflex: High-resolution and arbitrary-topology 3d shape modeling. arXiv preprint arXiv:2503.21732, 2025.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718, 2021.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

Junchao Huang, Xinting Hu, Shaoshuai Shi, Zhuotao Tian, and Li Jiang. Edit360: 2d image edits to 3d assets from any angle. arXiv preprint arXiv:2506.10507, 2025.

Team Hunyuan3D, Shuhui Yang, Mingxin Yang, Yifei Feng, Xin Huang, Sheng Zhang, Zebin He, Di Luo, Haolin Liu, Yunfei Zhao, et al. Hunyuan3d 2.1: From images to high-fidelity 3d assets with production-ready pbr material. arXiv preprint arXiv:2506.15442, 2025.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu. Brushnet: A plug-andplay image inpainting model with decomposed dual-branch diffusion. In European Conference on Computer Vision, pp. 150–168. Springer, 2024.

Kunal Kathare, Ankit Dhiman, K Vikas Gowda, Siddharth Aravindan, Shubham Monga, Basavaraja Shanthappa Vandrotti, and Lokesh R Boregowda. Instructive3d: Editing large reconstruction models with text instructions. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 3246–3256. IEEE, 2025.

Wonjae Kim, Bokyung Son, and Ildoo Kim. Vilt: Vision-and-language transformer without convolution or region supervision. In International conference on machine learning, pp. 5583–5594. PMLR, 2021.

Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

Vladimir Kulikov, Matan Kleiner, Inbar Huberman-Spiegelglas, and Tomer Michaeli. Flowedit: Inversion-free text-based editing using pre-trained flow models. arXiv preprint arXiv:2412.08629, 2024.

Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, et al. Flux. 1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742, 2025.

Zeqiang Lai, Yunfei Zhao, Haolin Liu, Zibo Zhao, Qingxiang Lin, Huiwen Shi, Xianghui Yang, Mingxin Yang, Shuhui Yang, Yifei Feng, et al. Hunyuan3d 2.5: Towards high-fidelity 3d assets generation with ultimate details. arXiv preprint arXiv:2506.16504, 2025.

Lin Li, Zehuan Huang, Haoran Feng, Gengxiong Zhuang, Rui Chen, Chunchao Guo, and Lu Sheng. Voxhammer: Training-free precise and coherent 3d editing in native 3d space. arXiv preprint arXiv:2508.19247, 2025a.

Peng Li, Suizhi Ma, Jialiang Chen, Yuan Liu, Congyi Zhang, Wei Xue, Wenhan Luo, Alla Sheffer, Wenping Wang, and Yike Guo. Cmd: Controllable multiview diffusion for 3d editing and progressive generation. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pp. 1–10, 2025b.

Yangguang Li, Zi-Xin Zou, Zexiang Liu, Dehu Wang, Yuan Liang, Zhipeng Yu, Xingchao Liu, Yuan-Chen Guo, Ding Liang, Wanli Ouyang, et al. Triposg: High-fidelity 3d shape synthesis using large-scale rectified flow models. arXiv preprint arXiv:2502.06608, 2025c.

Yuhan Li, Yishun Dou, Yue Shi, Yu Lei, Xuanhong Chen, Yi Zhang, Peng Zhou, and Bingbing Ni. Focaldreamer: Text-driven 3d editing via focal-fusion assembly. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pp. 3279–3287, 2024.

Fangfu Liu, Wenqiang Sun, Hanyang Wang, Yikai Wang, Haowen Sun, Junliang Ye, Jun Zhang, and Yueqi Duan. Reconx: Reconstruct any scene from sparse views with video diffusion model, 2025a. URL https://arxiv.org/abs/2408.16767.

Fangfu Liu, Junliang Ye, Yikai Wang, Hanyang Wang, Zhengyi Wang, Jun Zhu, and Yueqi Duan. Dreamreward-x: Boosting high-quality 3d generation with human preference alignment. IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–14, 2025b. doi: 10.1109/ TPAMI.2025.3609680.

Feng-Lin Liu, Hongbo Fu, Yu-Kun Lai, and Lin Gao. Sketchdream: Sketch-based text-to-3d generation and editing. ACM Transactions on Graphics (TOG), 43(4):1–13, 2024.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.

Ruijie Lu, Yu Liu, Jiaxiang Tang, Junfeng Ni, Yuxiang Wang, Diwen Wan, Gang Zeng, Yixin Chen, and Siyuan Huang. Dreamart: Generating interactable articulated objects from a single image. arXiv preprint arXiv:2507.05763, 2025.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021.

Aryan Mikaeili, Or Perel, Mehdi Safaee, Daniel Cohen-Or, and Ali Mahdavi-Amiri. Sked: Sketchguided text-based 3d editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 14607–14619, 2023.

Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6038–6047, 2023.

Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pp. 4296– 4304, 2024.

Maxime Oquab, Timoth´ee Darcet, Th´eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

Francesco Palandra, Andrea Sanchietti, Daniele Baieri, and Emanuele Rodola. Gsedit: Efficient text-guided editing of 3d objects via gaussian splatting. arXiv preprint arXiv:2403.05154, 2024.

Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022.

Zhangyang Qi, Yunhan Yang, Mengchen Zhang, Long Xing, Xiaoyang Wu, Tong Wu, Dahua Lin, Xihui Liu, Jiaqi Wang, and Hengshuang Zhao. Tailor3d: Customized 3d assets editing and generation with dual-side images. arXiv preprint arXiv:2407.06191, 2024.

Etai Sella, Gal Fiebelman, Peter Hedman, and Hadar Averbuch-Elor. Vox-e: Text-guided voxel editing of 3d objects. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 430–440, 2023.

Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8871–8879, 2024.

Narek Tumanyan, Michal Geyer, Shai Bagon, and Tali Dekel. Plug-and-play diffusion features for text-driven image-to-image translation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1921–1930, 2023.

Jiangshan Wang, Junfu Pu, Zhongang Qi, Jiayi Guo, Yue Ma, Nisha Huang, Yuxin Chen, Xiu Li, and Ying Shan. Taming rectified flow for inversion and editing. arXiv preprint arXiv:2411.04746, 2024a.

Xinzhou Wang, Yikai Wang, Junliang Ye, Zhengyi Wang, Fuchun Sun, Pengkun Liu, Ling Wang, Kai Sun, Xintong Wang, and Bin He. Animatabledreamer: Text-guided non-rigid 3d model generation and reconstruction with canonical score distillation, 2024b. URL https://arxiv. org/abs/2312.03795.

Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in neural information processing systems, 36:8406–8441, 2023.

Cong Wei, Zheyang Xiong, Weiming Ren, Xeron Du, Ge Zhang, and Wenhu Chen. Omniedit: Building image editing generalist models through specialist supervision. In The Thirteenth International Conference on Learning Representations, 2024.

Si-Tong Wei, Rui-Huan Wang, Chuan-Zhi Zhou, Baoquan Chen, and Peng-Shuai Wang. Octgpt: Octree-based multiscale autoregressive models for 3d shape generation, 2025. URL https: //arxiv.org/abs/2504.09975.

Haohan Weng, Yikai Wang, Tong Zhang, CL Chen, and Jun Zhu. Pivotmesh: Generic 3d mesh generation via pivot vertices guidance. arXiv preprint arXiv:2405.16890, 2024a.

Haohan Weng, Zibo Zhao, Biwen Lei, Xianghui Yang, Jian Liu, Zeqiang Lai, Zhuo Chen, Yuhong Liu, Jie Jiang, Chunchao Guo, Tong Zhang, Shenghua Gao, and C. L. Philip Chen. Scaling mesh generation via compressive tokenization, 2024b. URL https://arxiv.org/abs/2411. 07025.

Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 21469–21480, 2025.

Tianhao Xie, Eugene Belilovsky, Sudhir Mudur, and Tiberiu Popa. Dragd3d: Realistic mesh editing with rigidity control driven by 2d diffusion priors. arXiv preprint arXiv:2310.04561, 2023.

Jiale Xu, Xintao Wang, Yan-Pei Cao, Weihao Cheng, Ying Shan, and Shenghua Gao. Instructp2p: Learning to edit 3d point clouds with text instructions. arXiv preprint arXiv:2306.07154, 2023.

Yunhan Yang, Yuan-Chen Guo, Yukun Huang, Zi-Xin Zou, Zhipeng Yu, Yangguang Li, YanPei Cao, and Xihui Liu. Holopart: Generative 3d part amodal segmentation. arXiv preprint arXiv:2504.07943, 2025a.

Yunhan Yang, Yufan Zhou, Yuan-Chen Guo, Zi-Xin Zou, Yukun Huang, Ying-Tian Liu, Hao Xu, Ding Liang, Yan-Pei Cao, and Xihui Liu. Omnipart: Part-aware 3d generation with semantic decoupling and structural cohesion. arXiv preprint arXiv:2507.06165, 2025b.

Chongjie Ye, Yushuang Wu, Ziteng Lu, Jiahao Chang, Xiaoyang Guo, Jiaqing Zhou, Hao Zhao, and Xiaoguang Han. Hi3dgen: High-fidelity 3d geometry generation from images via normal bridging. arXiv preprint arXiv:2503.22236, 3:2, 2025a.

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.

Junliang Ye, Fangfu Liu, Qixiu Li, Zhengyi Wang, Yikai Wang, Xinzhou Wang, Yueqi Duan, and Jun Zhu. Dreamreward: Text-to-3d generation with human preference, 2024. URL https: //arxiv.org/abs/2403.14613.

Junliang Ye, Zhengyi Wang, Ruowen Zhao, Shenghao Xie, and Jun Zhu. Shapellm-omni: A native multimodal llm for 3d generation and understanding. arXiv preprint arXiv:2506.01853, 2025b.

Biao Zhang, Jiapeng Tang, Matthias Niessner, and Peter Wonka. 3dshape2vecset: A 3d shape representation for neural fields and generative diffusion models. ACM Transactions On Graphics (TOG), 42(4):1–16, 2023.

Ruowen Zhao, Junliang Ye, Zhengyi Wang, Guangce Liu, Yiwen Chen, Yikai Wang, and Jun Zhu. Deepmesh: Auto-regressive artist-mesh creation with reinforcement learning, 2025a. URL https://arxiv.org/abs/2503.15265.

Zibo Zhao, Zeqiang Lai, Qingxiang Lin, Yunfei Zhao, Haolin Liu, Shuhui Yang, Yifei Feng, Mingxin Yang, Sheng Zhang, Xianghui Yang, et al. Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation. arXiv preprint arXiv:2501.12202, 2025b.

Xin-Yang Zheng, Yang Liu, Peng-Shuai Wang, and Xin Tong. Sdf-stylegan: Implicit sdf-based stylegan for 3d shape generation, 2022. URL https://arxiv.org/abs/2206.12055.

Yang Zheng, Mengqi Huang, Nan Chen, and Zhendong Mao. Pro3d-editor: A progressive-views perspective for consistent and precise 3d editing. arXiv preprint arXiv:2506.00512, 2025.

A APPENDIX

- A.1 MORE VISUALIZATION RESULTS

In Fig. 10, we showcase additional editing results covering three types of operations: addition, removal, and replacement. As illustrated in the figure, our method, Nano3D, effectively preserves both geometric and textural consistency of the 3D assets before and after editing.

- A.2 CHOICE OF 3D REPRESENTATION: VOXEL VS. VECSET

We attempted to integrate FlowEdit into Hunyuan2.1 Hunyuan3D et al. (2025), but the inference results were highly unstable. When using aggressive hyperparameters, the generated 3D assets collapsed into fragmented or ”mud-like” shapes. Conversely, with more conservative hyperparameters, the edited assets ignored the target condition entirely and reproduced a mesh nearly identical to the source. We believe the primary reason FlowEdit works in TRELLIS but fails in Hunyuan2.1 lies in the difference in 3D representations: TRELLIS uses a voxel-based representation, which is more local and thus compatible with localized editing methods, whereas Hunyuan2.1 adopts a vecsetbased representation Zhang et al. (2023), which is more global and less suitable for transferring such localized editing techniques.

- A.3 EFFECT OF IMAGE CONSISTENCY IN FLOWEDIT EDITING

We sample several cases from the 3D-Alpaca dataset Ye et al. (2025b). As shown in Fig. 9, the dataset exhibits poor 2D consistency: in the left example, the cabinet changes its position after editing, while in the right example, the character’s scale is altered. Such inconsistencies between pre- and post-edit renderings also lead to significant mismatches in the corresponding 3D assets. Following this observation, we further evaluate our Nano3D framework using the same data in Fig. 9. The results show that under such inconsistent 2D conditions, FlowEdit fails to achieve reliable localized editing. Specifically, when n max is set large, the output remains nearly identical to the source asset, ignoring the target condition; when n max is set small, the source condition is disregarded and the results become entirely inconsistent.

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

Source image Edited image Source image Edited image

example1 example2

Figure 9: A bad case sampled from the 3D-Alpaca Ye et al. (2025b) dataset shows that its image consistency before and after editing is poorly maintained.

- A.4 THE PROMPT USED TO GENERATE EDITING INSTRUCTION FROM THE RENDERING

As shown in the Table. 5, we present an example of constructing editing instructions with a VLM. A strict template is used to constrain the VLM and prevent it from generating instructions beyond Nano3D’s capabilities.

Table 4: Category distribution.

Rank Category Count Share (%)

- 1 Human 20,755 20.75

- 2 Weapon 11,021 11.03

- 3 Furniture 10,442 10.45

- 4 Personal Item 10,277 10.28

- 5 Animal 10,186 10.19

- 6 Vehicle 9,376 9.38

- 7 Building 9,005 8.97

- 8 Else 6,593 6.60

- 9 Electronic Device 5,283 5.29

- 10 Plant 4,441 4.45

- 11 electronics 2,622 5.29

Table 5: The prompt used to generate editing instruction from the rendering

Editing Action Prompt

Given an image, generate a short “replace” type editing instruction in the format: Replace [original object/part/pattern] with [new element]

Replace

Additional Requirements: The [original object/part/pattern] must already exist in the image. It can be an entire object, a part of an object, a geometric shape, or a pattern. The [new element] should clearly differ from the original and fit naturally into the image. It can be another object, a different part, a new shape, text, or a new pattern. Avoid replacing with intangible elements (e.g., gases, smoke, light, shadow). Do not change colors — replacements must not involve altering the color of any existing element.

General Rules: Keep the instruction short and clear. No extra explanation or description.

Given an image, generate a short ’remove; type editing instruction in the format: Remove [object/part]

Remove

Additional Requirements: The [object/part] must already exist in the image. It can be the whole object or a specific part of an object (e.g., handle of a cup, branch of a tree). The removal should be visually noticeable and affect the composition of the image. Avoid removing intangible elements (e.g., light, shadow, gases, smoke).

General Rules: Keep the instruction short and clear. No extra explanation or description.

Given an image, generate a short “add” type editing instruction in the format: Add [element] to [location]

Add

Additional Requirements: The [location] can be: an existing object in the image, a position within the image (e.g., top left, bottom center), or a specific part/position of an object (e.g., handle of a cup, roof of a house). The [element] should blend naturally into the image and not appear abrupt. It can be an object, text, pattern, or other visual addition. Avoid adding gases, smoke, or other intangible elements.

General Rules: Keep the instruction short and clear. No extra explanation or description.

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

Replace the bucket with a bulldozer blade.

Remove the veil from the character’s head.

Remove the tail feathers from the bird.

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

Replace the chimney with a bell.

Remove the hat from the character’s head.

Add a statue to the top of the pyramid.

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

Add a white cushion to the seat of the chair.

Add a padlock to he chest. Remove the red flag from the top of the chest.

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

Add a basket to the front of the scooter.

Add a shield to the warrior’s hand.

Remove the bottom drawer.

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

Replace the front wheel with a wooden wheel.

Add trousers to the character.

Replace the bottom part of the lamp with a rectangular stone base.

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

Open the wooden door from the archway.

Add a carrot to the snowman’s mouth.

Add a white flag to the mast of the boat.

- Figure 10: We present additional editing results involving addition, removal, and replacement. Edited regions are highlighted with red dashed circles. As shown, Nano3D achieves high editing consistency, preserving geometry and texture outside the edited areas.

