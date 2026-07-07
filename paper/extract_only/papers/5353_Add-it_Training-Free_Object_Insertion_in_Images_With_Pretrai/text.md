# arXiv:2411.07232v2[cs.CV]12Nov2024

## ADD-IT: TRAINING-FREE OBJECT INSERTION IN IMAGES WITH PRETRAINED DIFFUSION MODELS

Yoad Tewel NVIDIA, Tel-Aviv University

Yuval Atzmon NVIDIA

Rinon Gal NVIDIA, Tel-Aviv University

Lior Wolf Tel-Aviv University

Dvir Samuel Bar-Ilan University

Gal Chechik NVIDIA

Source Output Source Output

The Nvidia logo on the wall The tennis player wearing a headband

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

RealGeneratedStep-by-Step

A sheep wearing boots A man sitting next to a table

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

An empty living room

A painting of cat saying “Do It”

A couch in the room

A dog lying on the couch

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Figure 1: Given an input image (left in each pair), either real (top row) or generated (mid row), along with a simple textual prompt describing an object to be added Add-it seamlessly adds the object to the image in a natural way. Add-it allows the step-by-step creation of complex scenes without the need for optimization or pre-training.

ABSTRACT

Adding Object into images based on text instructions is a challenging task in semantic image editing, requiring a balance between preserving the original scene and seamlessly integrating the new object in a fitting location. Despite extensive efforts, existing models often struggle with this balance, particularly with finding a natural location for adding an object in complex scenes. We introduce Add-it, a training-free approach that extends diffusion models’ attention mechanisms to incorporate information from three key sources: the scene image, the text prompt, and the generated image itself. Our weighted extended-attention mechanism maintains structural consistency and fine details while ensuring natural object placement. Without task-specific fine-tuning, Add-it achieves state-ofthe-art results on both real and generated image insertion benchmarks, including

our newly constructed ”Additing Affordance Benchmark” for evaluating object placement plausibility, outperforming supervised methods. Human evaluations show that Add-it is preferred in over 80% of cases, and it also demonstrates improvements in various automated metrics. Our code and data will be available at: https://research.nvidia.com/labs/par/addit/

- 1 INTRODUCTION

Adding objects to images based on textual instructions is a challenging task in image editing, with numerous applications in computer graphics, content creation and synthetic data generation. A creator may want to use text-to-image models to iteratively build a complex visual scene, while autonomous driving researchers may wish to draw pedestrians in new scenarios for training their car-perception system. Despite considerable recent research efforts on text-based editing, this particular task remains a challenge . When adding objects, one needs to preserve the appearance and structure of the original scene as closely as possible, while inserting the novel objects in a way that appears natural. To do so, one must first understand affordance—the deep semantic knowledge of how people and objects interact, in order to position an object in a reasonable location. For brevity, we call this task Image Additing.

Several studies (Hertz et al., 2022; Meng et al., 2022) tried addressing this task by leveraging modern text-to-image diffusion models. This is a natural choice since these models embody substantial knowledge about arrangements of objects in scenes and support open-world conditioning on text. While these methods perform well for various editing tasks, their success rate for adding objects is disappointingly low, failing to align with both the source image and the text prompt. In response, another set of methods took a more direct learning approach (Brooks et al., 2023; Zhang et al., 2023; Canberk et al., 2024). They trained deep models on large image editing datasets, pairing images with and without an object to add. However, these often struggle with generalization beyond their training data, falling short of the general nature of the original diffusion model itself. This typically manifests

- as a failure to insert the new object, the creation of visual artifacts, or more commonly – failing to insert the object in the correct place, i.e. struggling with affordances. Indeed, we remain far from achieving open-world object insertions from text instructions.

Here we describe an open-world, training-free method that can successfully leverage the knowledge stored in text-to-image foundation models, to naturally add objects into images. As a guiding principle, we propose that addressing the affordance challenge requires methods to carefully balance between the context of the existing scene and the instructions provided in the prompt. We achieve this by: first, extending the multi-modal attention mechanism (Esser et al., 2024) of recent T2I diffusion models to also consider tokens from a source image; and second, controlling the influence of each multi-modal attention component: the source image, the target image and the text prompt. A main contribution of this paper is a mechanism to balance these three sources of attention during generation. We also apply a structure transfer step and introduce a novel subject-guided latent blending mechanism to preserve the fine details of the source image while enabling necessary adjustments, such as shadows or reflections. Our full pipeline is shown at fig. 2. We name our method Add-it.

Image Additing methods typically face three main failure modes: neglect, appearance, and affordance. While current CLIP-based evaluation protocols can partially assess neglect and appearance, there is a lack of reliable methods for evaluating affordance. To address this gap, we introduce the “Additing Affordance Benchmark,” where we manually annotate suitable areas for object insertion in images and propose a new protocol specifically designed to evaluate the plausibility of object placement. Additionally, we introduce a metric to capture object neglect. Add-it outperforms all baselines, improving affordance from 47% to 83%. We also evaluate our method on an existing benchmark (Sheynin et al., 2023) with real images, as well as our newly proposed Additing Benchmark for generated images. Add-it consistently surpasses previous methods, as reflected by CLIP-based metrics, our object inclusion metric, and human preference, where our method is favored in over 80% of cases, even against methods specifically trained for this task.

Our contributions are as follows: (i) We propose a training-free method that achieves state-of-theart results on the task of object insertion, significantly outperforming previous methods, including supervised ones trained for this task. (ii) We analyze the components of attention in a modern diffusion model and introduce a novel mechanism to control their contribution, along with novel

Subject Guided Latent Blending and a noise structure transfer. (iii) We introduce an affordance benchmark and a new evaluation protocol to assess the plausibility of object insertion, addressing a critical gap in current Image Additing evaluation methods.

- 2 RELATED WORK

Object Placement and Insertion. Inserting objects into images remains a core challenge in image editing. Traditional computer graphics methods often depend on manual object placement (C. Wang, 2014) or utilize synthetic data-driven approaches (Fisher et al., 2012). Early computer vision techniques employed contextual cues to predict possible object positions (Choi et al., 2012; Lin et al., 2013; Zhao et al., 2011). With advancements in deep learning, generative models have been trained to learn object placements. For example, Compositing GAN (Azadi et al., 2020) generates object composites by refining geometry and appearance, while RelaxedPlacement (Lee et al., 2022) optimizes object placement and sizing based on relationships depicted in scene graphs. OBJect3DIT (Michel et al., 2024) explores 3D-aware object insertion guided by language instructions, primarily using synthetic data. Despite their effectiveness, these methods often struggle with the complexities of real-world placement scenarios.

Editing with Text-to-Image Diffusion Models. The emergence of high-performing text-toimage diffusion models (Rombach et al., 2022; Saharia et al., 2022; Ramesh et al., 2022; Balaji

- et al., 2022; Esser et al., 2024) has paved the way for effective text-based image editing techniques. Methods like Prompt-to-Prompt (Hertz et al., 2022) modify attention maps by injecting the input caption’s attention into the target caption’s attention, while SDEdit (Meng et al., 2022) uses a stochastic differential equation to iteratively denoise and enhance the realism of user-provided pixel edits. For editing real images, inversion techniques (Mokady et al., 2023; Wallace et al., 2022; Pan
- et al., 2023; Samuel et al., 2023; Deutch et al., 2024; Huberman-Spiegelglas et al., 2023; Tsaban & Passos, 2023; Brack et al., 2024; Garibi et al., 2024) first invert an input image to its latent noise representation using a given caption, enabling edits via methods like SDEdit or Prompt2Prompt. Cao et al. (2023) further improves real image editing using a mutual extended self-attention mechanism, an idea later extended to an array of generation (Tewel et al., 2024) and editing tasks like styleand appearance-transfer (Alaluf et al., 2024; Hertz et al., 2024) or object-dragging (Avrahami et al., 2024). Despite their effectiveness in various tasks, these methods struggle with object addition, often failing to align new objects with both the original image and the text prompt.

To improve editing performance, several methods proposed to directly fine-tune diffusion models. Imagic (Kawar et al., 2023) fine-tunes text embeddings (Gal et al., 2022a) and the diffusion UNet (Ronneberger et al., 2015) to handle complex textual instructions, whereas Text2LIVE (Bar-Tal et al., 2022) and Blended Diffusion (Avrahami et al., 2022) blend edited regions throughout the generation. InstructPix2Pix (Brooks et al., 2023) introduced an instructable image editing model trained on a large synthetic dataset for instruction-based edits, while MagicBrush (Zhang et al.,

- 2023) enhances this approach by fine-tuning InstructPix2Pix on a manually annotated dataset collected through an online editing tool. EmuEdit (Sheynin et al., 2023) trains a diffusion model on a large synthetic dataset to perform different editing tasks given a task embedding. EraseDraw (Canberk et al., 2024) leverages inpainting models to automatically generate high-quality training data for learning object insertion. They show that one can train models to realistically insert diverse objects into images based on language instructions.

Despite advancements in instruction-based image editing, we demonstrate that current methods still face significant challenges in accurately interpreting and executing object addition within images. In this paper, we propose a novel approach addressing the challenging task of object insertion. We show that by controlling the various attention components in the diffusion model, one can add new objects to existing images without further training or fine-tuning of the diffusion model.

- 3 METHOD

Our goal is to insert an object into a real or generated image using a simple textual prompt, ensuring the result appears natural and consistent with the source image. To achieve this, we leverage a

Diffusion Step

Diffusion Step

"A man sitting

at the table" K V X

K V

[Figure 13]

###### Subject Guided Latent Blending

[Figure 14]

Q K V

Q K V

Subject Attention

Structure Transfer

[Figure 15]

SAM-2

[Figure 16]

[Figure 17]

K V

K V

Figure 2: Architecture outline: Given a tuple of source noise XsourceT , target noise XtargetT , and a text prompt Ptarget, we first apply Structure Transfer to inject the source image’s structure into the

target image. We then extend the self-attention blocks so that XtargetT pulls keys and values from both Ptarget and XsourceT , with each source weighted separately. Finally, we use Subject Guided Latent Blending to retain fine details from the source image.

pretrained diffusion model without any additional training or optimization. Our solution consists of three core components: (1) a weighted extended self-attention mechanism that balances information from the source image, text prompt, and target image, (2) a noising approach that preserves the source image’s structure, and (3) a novel Subject-Guided Latent Blending mechanism to retain fine background details. For real images, we also introduce an inversion step, detailed below.

- 3.1 PRELIMINARIES: ATTENTION IN MM-DiT BLOCKS

Modern Diffusion Transformers (DiTs) models, such as SD3 (Esser et al., 2024) and FLUX (BlackForest, 2024), process concatenated sequences of textual-prompt and image-patch tokens through unified multi-modal self-attention blocks (MM-DiT blocks). Specifically, FLUX has two types of attention blocks: Multi-stream blocks which use separate projection matrices (WK,WV ,WQ) for text and image tokens, and Single-stream blocks where the same projection matrices are used for both. Both block types compute attention on the concatenated tokens as follows:

A = softmax([Qp,Qimg][Kp,Kimg]⊤/ dk), h = A · [Vp,Vimg] (1)

where Qp, Qimg are the textual-prompt and the image-patch queries, respectively. The same applies to K and V . Notably, Flux is composed of a series of Multi-stream blocks followed by a series of Single-stream blocks.

- 3.2 WEIGHTED EXTENDED SELF-ATTENTION

Our approach builds on top of the attention mechanism in MM-DiT blocks. In this attention mechanism, tokens are drawn from two sources: the image patches Ximage and the textual prompt P. In prior attention-based diffusion architectures, it was shown that the appearance of a source image can be transferred to a target through an extended self-attention mechanism, where the new image can attend to the tokens of the source. We propose a similar extension here, by allowing the multimodal attention to include another source — the tokens of the input image we wish to edit. More formally, we define the three sources of information as: the source image Xsource, the generated image Xtarget and the textual prompt describing the edit Ptarget. To compute the source image tokens, we simply denoise it in parallel to the target image, and concatenate its keys and values to the self-attention blocks, extending eq. (1):

A = softmax([Qp,Qtarget][Ksource,Kp,Ktarget]⊤/ dk), h = A · [Vsource,Vp,Vtarget] (2)

where Ksource and Vsource are the keys and value extracted from the source image, and Kp, Vp, Ktarget, Vtarget are the keys and values from the prompt and target image respectively. When

Xsource is a generated image, denoising it in parallel is trivial - we simply need to start denoising from the same seed that created Xsource. Dealing with a real image is more complicated, and we will describe our solution in the inversion section below.

However, we notice that simply appending the keys and values of the source image to the attention blocks leads to the source image controlling the attention, which in turn leads to neglect of the edit prompt, with the final generated image being a simple copy of the source image. We explore the dynamics of this phenomenon in detail in section 5. To avoid this effect, we can re-balance the contribution of different attention components by weighting their keys. Indeed, by reducing the weight of the source image tokens, we can achieve better balance and allow for more changes. However, if this is not done carefully, then we risk upsetting the balance in the opposite fashion and seeing alignment with the source image completely ignored. Hence, we can introduce a weighting term to each source of information, giving us the following multi-modal attention equation:

A = softmax([Qp,Qtarget][γs · Ksource,γp · Kp,γt · Ktarget]⊤/ dk) h = A · [Vsource,Vp,Vtarget]

(3)

where γs, γp, γt represent the weighting terms for the source image, the prompt, and the target image, respectively. In section 5 we explore the dynamics of the attention distribution across these three sources. In practice, we find that it is necessary to balance two key terms: the first is the

attention distributed over the source image Asource = exp(QpZ·Ksource) and the second is the attention distributed over the target image, Atarget = exp(γ·QZp·Ktarget), where Z is the softmax normalization term. To determine γ we define the function f(γ)=Asource−Atarget and use a root-solver algorithm to find γ such that f(γ)=0.

- 3.3 STRUCTURE TRANSFER

The weighted extended-attention mechanism allows to balance between information from the source image and the prompt, but the added objects do not always adhere to the image context (e.g. dog is too big for the chair). We attribute this issue to different seeds dictating specific structures in the generated image, which do not always align with the source image. We show that effect in fig. 8, where images generated with the same seed produce similar objects with or without the extended attention mechanism. To address this problem, we propose to ”choose” seeds with a structural similarity to the source image. We do so by noising the source latent Xsource to a very high noise level tstruct with randomly sampled noise ϵ ∼ N(0,I) following the recitified flow denoising formula Xt = (1 − σt)x0 + σtϵ. When tstruct is high enough, starting the denoising process from Xt

struct

will result in an image with similar global structure to the source image, while still allowing for changes to image content as demonstrated in fig. 8.

- 3.4 SUBJECT GUIDED LATENT BLENDING

The combination of structure transfer and the weighted attention mechanism ensures that the target image remains consistent with the structure and appearance of the source image, though some fine details, such as textures and small background objects, may still change. Our goal is to preserve all elements of the source image not affected by the added object. To achieve this, we propose Latent Blending; A naive approach would involve identifying the pixels unaffected by the object insertion and keeping them identical to those in the source image. However, two challenges arise: First, a perfect mask is needed to separate the object from the background to avoid artifacts. Second, we aim to preserve collateral effects from the object insertion, such as shadows and reflections. To address these issues, we propose generating a rough mask of the object, which is then refined using SAM-2 (Ravi et al., 2024) to obtain a final mask M. We then blend (Avrahami et al., 2022) the source and target noisy latents at timestep Tblend based on this mask.

To extract the rough object mask, we gather the self-attention maps corresponding to the token representing the object. We achieve this by multiplying the queries from the target image patches, Qtarget, with the key associated with the added object token, kobject. These maps are then aggregated across specific timesteps and layers that we identified as generating the most accurate results (further details can be found in the appendix A.1. We then apply a dynamic threshold to the attention maps using the Otsu method (Otsu, 1979) to obtain a rough object mask, Mr. Finally, we refine

| |I-Pix2Pix Erasedraw Magicbrush SDEdit P2P Ours|
|---|---|
|Affordance<br><br>|0.276 0.341 0.418 0.397 0.474 0.828|

Table 1: Comparison of methods based on Affordance score for the Additing Affordance Benchmark.

Ours

79% 21%

I-P2PEDMB

Ours

75% 25%

Ours

84% 16%

50% Win Rate

Figure 3: User Study results evaluated on the real images from the Emu Edit Benchmark.

Ours

SDEditP2P

90% 10%

Ours

83% 17%

50% Win Rate

Figure 4: User Study results evaluated on the generated images from the Image Additing Benchmark.

this mask using the general-purpose segmentation model, SAM-2. Since SAM-2 operates on images rather than noisy latents, we first estimate an image, X0, from the model’s velocity prediction, vθ, using the formula X0 = XT

) · vθ. In addition to an input image, SAM-2 requires a localization prompt in the form of points, a bounding box, or an input mask. In our method, we provide input points, as they tend to produce the most accurate masks. To extract these localization points, we iteratively sample local maxima from the attention maps - full details of this sampling process are provided in appendix A.1. Using these input points, we generate the refined object mask, M. Finally, we apply a simple latent blending step at timestep Tblend, where we compute Ztarget = M ⊙ Ztarget + (1 − M) ⊙ Zsource. We present results with and without latent blending, along with the resulting mask M, in fig. 9.

blend+1 − σT

+ (σT

blend

blend

- 3.5 Additing REAL IMAGES AND STEP-BY-STEP GENERATION

In the previous sections, we described our method for generating an edited image by drawing information from a source image within the same batch. When editing a generated image, this process is straightforward: one can save the source noise, ϵsource, that generated the source image and create an input batch containing both ϵsource and a random noise, ϵtarget, used to generate the target image. However, when editing an existing image, xsource, we do not have access to its original noise. A common approach would be to use an inversion method to recover the original noise, ϵsource, that generated Xsource. However, in our experiments, popular inversion methods, such as DDIM inversion (Song et al., 2020), do not adequately reconstruct the image using FLUX. We propose a simple solution: instead of recovering the original noise ϵsource, we sample a random noise ϵ. At each denoising step t, we produce a noisy source latent, Xsourcet = (1 − σt)Xsource + σtϵ. We then apply our method as usual, using the input batch at timestep t, [Xsourcet ,Xtargett ], where the target image draws information from the source image. This simple technique ensures perfect reconstruction of the source image, since σ0 = 0 and therefore Xsource0 = Xsource.

Our method, applicable to both generated and real images, can be extended for step-by-step generation. Users can start with an initial image from a textual prompt and iteratively modify it with additional prompts, progressively adding elements or changes to the scene. Examples of step-bystep generation are shown in fig. 1 and fig. 11.

- 4 EXPERIMENTS

Evaluation Baselines We compare our method with two classes of baselines: (1) Training-Free methods that leverage the existing capabilities of text-to-image models: Prompt-to-Prompt (Hertz et al., 2022), a method which injects the attention map of the source image into the target image to preserve its structure, and SDEdit (Meng et al., 2022), a method that adds partial noise to an existing image and then denoises it. Both methods were re-implemented on the FLUX.1-dev model for fair comparison. (2) Pretrained Instruction following models, specifically trained to edit and add objects to existing images: InstructPix2Pix (Brooks et al., 2023) an instruction following model

#### Emu Edit Additing Benchmark

|Method|CLIPdir CLIPout CLIPim Inc.<br><br>|CLIPdir CLIPout CLIPim Inc.|
|---|---|---|
|InstructPix2Pix Erasedraw Magicbrush SDEdit Prompt2Prompt|0.074 0.312 0.929 34% 0.088 0.313 0.941 65% 0.091 0.313 0.927 66%<br><br>— — — — — — —<br><br>|0.074 0.244 0.943 55% 0.117 0.248 0.958 76% 0.114 0.250 0.925 86% 0.091 0.248 0.955 60% 0.170 0.280 0.850 97%<br><br>|
|Ours|0.101 0.322 0.929 81%<br><br>|0.200 0.261 0.968 93%<br><br>|

##### Table 2: CLIP and Inclusion metric results for EmuEdit and Additing Benchmark.

Source Ours Instruct Pix2Pix Erase Draw

Magic Brush

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

“A Coyote behind the red-headed bird”

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

“A fork on the left side of the plate”

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

“A basket on the front of the bicycles”

- Figure 5: Qualitative Results from the Emu-Edit Benchmark. Unlike other methods, which fail to place the object in a plausible location, our method successfully achieves realistic object insertion.

trained on a large scale of synthetic instruction data, Magicbrush (Zhang et al., 2023) a version of InstructPix2Pix fine-tuned on manually annotated editing dataset, and Erasedraw (Canberk et al.,

- 2024) a model trained on large dataset constructed using an inpainting model. Add-it implementation details can be found in appendix A.1.

Metrics We evaluate the results of our method and the baselines using automatic metrics and human evaluations for each source and target image-caption pair. Automatic Metrics: we start by adopting the CLIP (Radford et al., 2021) based metrics proposed in Emu-Edit (Sheynin et al., 2023): (i) CLIPdir (Gal et al., 2022b) measures the agreement between change in captions and the change in images. (ii) CLIPimg measures similarity between source and target images. (iii) CLIPout measures the target image and caption similarity. We propose two additional metrics: (iv) Inclusion measures the portions of cases the object was added to the image, evaluated automatically using the open-vocabulary detection model Grounding-DINO (Liu et al., 2023). (v) Affordance measures whether the object was added to a plausible location, utilizing Grounding-DINO and a manually annotated set of possible locations. Human Evaluations: we ask human raters to pick the best Additing output when faced with a source image, instruction and images generated by our method and a competing baseline. Further details in appendix A.5.

4.1 EVALUATION RESULTS

Emu-Edit Benchmark Following EraseDraw (Canberk et al., 2024) we evaluate our method on a subset of EmuEdit’s (Sheynin et al., 2023) validation set with the task class of ”Add”, designed for insertion instructions. The benchmark consists of sets of images and prompts before and after an edit, and the corresponding instruction. Table 2 shows our model outperforms all previous approaches in the CLIPdir, CLIPout and the Inclusion metrics. In the CLIPim metric, which indicates how close the edited image is to the source image, we are second only to Erasedraw. This result is

Source Ours Prompt-to-Prompt SDEdit

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

“A toy truck in the child's hands”

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

“The man is holding a shopping bag”

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

“A microscope on the counter next to samples”

- Figure 6: Qualitative Results from the Additing Benchmark. While Prompt-to-Prompt fails to align with the source image, and SDEdit fails to align with the prompt, our method offers Additing that adheres to both prompt and source image.

not surprising given that in 35% of the cases Erasedraw did not add an object to the image (indicated by the Inclusion metric), artificially boosting the image similarity score. Due to the limitations of automatic metrics, especially in assessing the naturalness of edits, we conducted a head-to-head evaluation with human raters against each baseline, as shown in fig. 3. Our method’s outputs were preferred in 80% of cases. Finally, we present a qualitative comparison to other methods using images from the EmuEdit benchmark in fig. 5. Previous methods often produce artifacts, unnatural object placements, or fail to modify the image. In contrast, our method generates high-quality outputs that consider the context of the source image.

Additing Benchmark To evaluate our method against both pre-trained models and zero-shot methods, which tend to perform better on generated images, we created a benchmark for the Additing task. We asked ChatGPT to generate 200 sets of source and target prompts along with Additing instructions. Using Flux, we generated images and filtered 100 sets where the instructions were viable. We report all results in Table 2. Our method outperforms all baselines on the CLIPdir and CLIPim metrics. Although Prompt-to-Prompt slightly surpasses us on CLIPout and Inclusion, it does so by heavily altering the source image, as shown by its low CLIPim score. As in the EmuEdit Benchmark, we asked human raters to compare our method against the zero-shot baseline. Our method was preferred in 90% of cases against Prompt2Prompt and 83% against SDEdit fig. 4. Finally, fig. 6 shows a comparison on the Additing Benchmark, where other methods struggle to balance object addition, background preservation, and context, while ours produces natural, appealing outputs.

Additing Affordance Benchmark Throughout our experiments we observed that the major shortcoming of existing methods is incorrect affordance, namely, objects are added at implausible locations (see the basket in fig. 5). To automatically quantify affordance, we constructed an affordance benchmark. It contains 200 images and prompts, with manually annotated bounding-boxes indicating the plausible locations to add objects in each image. Dataset construction and evaluation protocol details are available in appendix A.4. We present the results of all methods in table 1. As expected, previous methods perform poorly, with low affordance scores, particularly trained models like InstructPix2Pix, which scored as low as 0.276. In contrast, Add-it scores nearly double that of the best-performing method, demonstrating its ability to balance information from the source image and target prompt. We explore additional results of our method in appendix A.2.

|(A)<br><br>[Figure 45]|[Figure 46]<br><br>(B)|
|---|---|
|[Figure 47]<br><br>[Figure 48]<br><br>[Figure 49]<br><br>[Figure 50]<br><br>Source Scale=1.0 Auto (Ours) Scale=1.2<br><br>“A pile of laundry in the basket”<br><br>(C)| |

- Figure 7: (A) Affordance and Object Inclusion scores across weight scale values, with our automatic weight scale achieving a good balance between the two. (B) Visualization of the prompt token attention spread across different sources, model blocks, and weight scales, averaged over multiple examples from a small validation set. (C) A representative example demonstrating the effect of varying target weight scales.

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

Vanilla Generation Source Image

w/o Structure Transfer Ours

[Figure 55]

- Figure 8: Ablation over various steps for applying the Structure Transfer mechanism. Applying it too early misaligns the generated images with the source image’s structure while applying it too late causes the output image to neglect the object. Our chosen step strikes a balance between both.

- 5 ANALYSIS

In this section, we analyze the attention distribution in the MM-DiT block and the key components of our method to better justify our design choices. In appendix A.3 we analyze the role of positional encoding in the extended-attention mechanism.

MM-DiT Attention Distribution First, we analyze the different attention components in the extended MM-DiT blocks. Recall that in the extended-attention mechanism described in section 3.2 there are three token sets: the source image Xsource, the target image Xtarget and the prompt Ptarget. In our experiments, we notice that simply applying the extended attention mechanism results in the target image closely following the appearance of the source image while neglecting the prompt - meaning no object is added to the image. We attribute this problem to the way the attention is distributed across the three sets of tokens. In particular, we find empirically that the target prompt’s attention Ap ∝ exp(Qp·[Ksource,Kp,Ktarget]) serves as an effective proxy for balancing the three sources of attention. A simple way to control the attention distribution is by introducing scale factor

Affordance Map w/o blend Ours

Source

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

“The girl is wearing headphones”

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

“A water bottle on the bicycle”

- Figure 9: Images generated by Add-it with and without the latent blending step, along with the resulting affordance map. The latent blending block helps align fine details from the source image, such as removing the girl’s glasses or adjusting the shadows of the bicycles.

γp,γtarget so that Ap ∝ exp(Qp·[Ksource,γp·Kp,γtarget·Ktarget]). In practice, we find that using γ = γp = γtarget is adequate. In fig. 7 (B) we visualize the prompt attention Ap spread across the three token sets. In the standard extended-attention case (γ = 1.0), the source image tokens (purple) receive more attention than the target image tokens (orange), preventing the generated image from incorporating the added object. On the other hand when scaling up too much (γ = 1.2), the target image tokens overwhelm the source image token, causing the output image to stray away from the source image structure. Finally, when the scaling value balances the attention between Xsource and Xtarget (γ = Auto), the output image successfully incorporates the added object, while preserving the target image structure and taking into account its context when placing the object. These observations are qualitatively shown in fig. 7 (C) and are also reflected in fig. 7 (A), where the scale that balances the attention offers a good balance between affordance and Object Inclusion.

Ablation Study Next, we evaluate the impact of different components of our method. First, we demonstrate the effect of the weight scale, γ. In fig. 7 (A) we present a graph showing affordance and Object Inclusion as functions of different weight scales. As the weight scale increases, the added object tends to appear more frequently in the image. However, beyond a certain threshold, the affordance score drops. This decline occurs when the target image ignores the structure of the source image, generating objects in unnatural locations, as illustrated in fig. 7 (C). Next, we explore the effect of latent blending. In fig. 9 we show output images with and without the latent blending step, along with the affordance map automatically extracted by our method. Notice how the blending step aligns the fine details of the source image without introducing artifacts. Finally, we examine the structure transfer component. In fig. 8 we illustrate the effect of applying the structure transfer step at different stages of the denoising process. When the structure transfer is applied too early, the affordance score is low, meaning the target image does not adhere to the structure of the source image. On the other hand, applying it later in the process results in a lower object inclusion metric, indicating that the target image neglects the object. Ultimately, when the structure transfer is applied

- at t = 933, we achieve a balance between object inclusion and affordance. A qualitative example is also provided in fig. 8.

- 6 LIMITATIONS

Add-it shows strong performance across various benchmarks, but it has some limitations. Since the method relies on pretrained diffusion models, it may inherit biases from the training data, which could affect object placement in unfamiliar or highly complex scenes. Additionally, because our method uses target prompts rather than explicit instructions, users may need to construct more detailed prompts to achieve the same edit. For instance, with an image of a dog, the prompt “A dog” won’t add another dog to the scene, as seen in fig. 10. Instead, the user would need to provide alternative prompts, such as “Two dogs sitting on the grass”. Lastly, we observe that Additing on real images is still not as effective as it is on generated images. We attribute this shortcoming to the

Failure Success

|[Figure 64]<br><br>[Figure 65]<br><br>Another dog sitting on the side|
|---|

|[Figure 66]<br><br>[Figure 67]<br><br>A person standing behind the dog|
|---|

Source

[Figure 68]

- Figure 10: Add-it may fail to add a subject that already exists in the source image. When prompted to add another dog to the image, Add-it generates the same dog instead, though it successfully adds a person behind the dog.

current FLUX inversion algorithm and believe that a more advanced inversion algorithm could help bridge this gap. Additional failure cases of the model are presented in fig. 16.

- 7 CONCLUSION

We introduced Add-it, a training-free method for adding objects to images using simple text prompts. We analyzed the attention distribution in MM-DiT blocks and introduced novel mechanisms such as weighted extended-attention and Subject-Guided Latent Blending. Additionally, we addressed a critical gap in evaluation by creating the ”Additing Affordance Benchmark,” which allows for an accurate assessment of object placement plausibility in image Additing methods. Add-it consistently outperforms previous approaches, improving affordance from 47% to 83% and achieving state-ofthe-art results on both real and generated image benchmarks. Our work demonstrates that leveraging the knowledge in pretrained diffusion models is a promising direction for tackling challenging tasks like image Additing. As diffusion models continue to evolve, methods like Add-it have the potential to drive further advancements in semantic image editing and related applications.

ETHICS STATEMENT

In this work, we acknowledge the ethical considerations associated with image editing technologies. While our method enables advanced object insertion capabilities, it also has the potential for misuse, such as creating misleading or harmful visual content. We strongly encourage the responsible and ethical use of this technology, emphasizing transparency and consent in its applications. Additionally, biases present in pretrained models may affect generated outputs, and we recommend further research to mitigate such issues in future work. Human evaluations were conducted with informed consent.

ACKNOWLEDGMENTS

We thank Assaf Shocher, Lior Hirsch and Omri Kaduri for useful discussions and for providing feedback on an earlier version of this manuscript. This work was completed as part of the first author’s PhD thesis at Tel-Aviv University.

REFERENCES

Yuval Alaluf, Daniel Garibi, Or Patashnik, Hadar Averbuch-Elor, and Daniel Cohen-Or. Crossimage attention for zero-shot appearance transfer. In ACM SIGGRAPH 2024 Conference Papers, pp. 1–12, 2024.

Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In CVPR, 2022.

Omri Avrahami, Rinon Gal, Gal Chechik, Ohad Fried, Dani Lischinski, Arash Vahdat, and Weili Nie. Diffuhaul: A training-free method for object dragging in images. arXiv preprint arXiv:2406.01594, 2024.

S. Azadi, D. Pathak, S. Ebrahimi, and T. Darrell. Compositional gan: Learning image-conditional binary composition. In International Journal of Computer Vision, volume 128, pp. 2629–2642, 2020.

Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, et al. eDiff-I: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.

Omer Bar-Tal, Dolev Ofri-Amar, Rafail Fridman, Yoni Kasten, and Tali Dekel. Text2live: Textdriven layered image and video editing. In ECCV, 2022.

Black-Forest. Flux: Diffusion models for layered image generation. https://github.com/ black-forest-labs/flux, 2024. Accessed: 2024-09-24.

Manuel Brack, Felix Friedrich, Katharia Kornmeier, Linoy Tsaban, Patrick Schramowski, Kristian Kersting, and Apolin´ario Passos. Ledits++: Limitless image editing using text-to-image models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8861–8870, 2024.

Tim Brooks, Aleksander Holynski, and Alexei A. Efros. Instructpix2pix: Learning to follow image editing instructions. In CVPR, 2023.

R C. Wang, L. Yang. Scene design by integrating geometry and physics for realistic image synthesis. Computer Graphics Forum, 2014.

Alper Canberk, Maksym Bondarenko, Ege Ozguroglu, Ruoshi Liu, and Carl Vondrick. Erasedraw: Learning to draw step-by-step via erasing objects from images. 2024.

Mingdeng Cao, Xintao Wang, Zhongang Qi, Ying Shan, Xiaohu Qie, and Yinqiang Zheng. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In ICCV, 2023.

W. Choi, Y. W. Chao, C. Pantofaru, and S. Savarese. Context-driven 3d scene understanding from a single image. In ICCV, 2012.

Gilad Deutch, Rinon Gal, Daniel Garibi, Or Patashnik, and Daniel Cohen-Or. Turboedit: Text-based image editing using few-step diffusion models, 2024.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis, 2024.

M. Fisher, D. Ritchie, M. Savva, T. Funkhouser, and P. Hanrahan. Example-based synthesis of 3d object arrangements. In ACM Transactions on Graphics (TOG), 2012.

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022a.

Rinon Gal, Or Patashnik, Haggai Maron, Amit H Bermano, Gal Chechik, and Daniel CohenOr. Stylegan-nada: Clip-guided domain adaptation of image generators. ACM Transactions on Graphics (TOG), 41(4):1–13, 2022b.

Daniel Garibi, Or Patashnik, Andrey Voynov, Hadar Averbuch-Elor, and Daniel Cohen-Or. Renoise: Real image inversion through iterative noising. arXiv preprint arXiv:2403.14602, 2024.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. Style aligned image generation via shared attention. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4775–4785, 2024.

Inbar Huberman-Spiegelglas, Vladimir Kulikov, and Tomer Michaeli. An edit friendly ddpm noise space: Inversion and manipulations. arXiv:2304.06140, 2023.

Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In CVPR, 2023.

J. Y. Lee, Z. Tseng, and P. Abbeel. Relaxedplacement: Learning to synthesize compositional scene layouts with object relations. In Computer Vision and Pattern Recognition (CVPR), 2022.

D. Lin, S. Fidler, and R. Urtasun. Holistic scene understanding for 3d object detection with rgbd cameras. In ICCV, 2013.

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023.

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In ICLR, 2022.

O. Michel, A. Bhattad, E. VanderBilt, R. Krishna, A. Kembhavi, and T. Gupta. Object3dit: Language-guided 3d-aware image editing. Advances in Neural Information Processing Systems, 36, 2024.

Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. CVPR, 2023.

Nobuyuki Otsu. A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and Cybernetics, 9(1):62–66, 1979. doi: 10.1109/TSMC.1979.4310076.

Zhihong Pan, Riccardo Gherardi, Xiufeng Xie, and Stephen Huang. Effective real image editing with accelerated iterative diffusion inversion. In ICCV, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman R¨adle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models. In CVPR, 2022.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention– MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pp. 234–241. Springer, 2015.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. NeurIPS, 2022.

Dvir Samuel, Barak Meiri, Nir Darshan, Shai Avidan, Gal Chechik, and Rami Ben-Ari. Regularized newton raphson inversion for text-to-image diffusion models, 2023.

Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. 2023.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.

Yoad Tewel, Omri Kaduri, Rinon Gal, Yoni Kasten, Lior Wolf, Gal Chechik, and Yuval Atzmon. Training-free consistent text-to-image generation. ACM Transactions on Graphics (TOG), 43(4): 1–18, 2024.

Linoy Tsaban and Apolin´ario Passos. Ledits: Real image editing with ddpm inversion and semantic guidance, 2023. URL https://arxiv.org/abs/2307.00522.

Bram Wallace, Akash Gokul, and Nikhil Vijay Naik. EDICT: Exact diffusion inversion via coupled transformations. CVPR, 2022.

Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. Magicbrush: A manually annotated dataset for instruction-guided image editing. In NeurIPS, 2023.

W. H. Zhao, J. Jiang, J. Weng, J. He, E. P. Lim, H. Yan, and X. Li. Image-based contextual advertisement recommendation. Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval, 2011.

A APPENDIX

- A.1 IMPLEMENTATION DETAILS

Add-it When evaluating Add-it, we use tstruct = 933 for generated images and tstruct = 867 for real images and tblend = 500. For the scaling factor γ, we use the root-finding solver described in section 3.2 on a set of validation images and set γ to 1.05, as it is close to the average result and performs well in practice. We generate the images with 30 denoising steps, building upon the diffusers implementation of the FLUX.1-dev model. We apply the extended attention mechanism until step t = 670 in the multi-stream blocks, and step t = 340 for the single-stream blocks.

Latent Blending Localization To extract a refined object mask as part of the Subject Guided Latent Blending component, we begin by extracting subject attention maps. Empirically, we find that the best-performing layers for this task are: ["transformer blocks.13","transformer blocks.14", "transformer blocks.18", "single transformer blocks.23", "single transformer blocks.33"]. To refine the mask from these attention maps, we need to identify points to use as prompts for SAM-2. To extract points from the attention map, we first select the point with the highest attention value. Then, we exclude the area around the chosen point and select the next highest point. This process is repeated until we either identify 4 points or the current maximal point value falls below 0.35 · pmax, where pmax is the initial maximum attention value. Finally, we feed the points to the SAM-2 model to end up with a refined object mask.

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

“A horse standing in the garden”

“The horse is wearing a pink dress”

“The horse is wearing a crown”

“A knight is riding the horse”

“A young girl looking at the camera”

“The girl is wearing a raincoat”

“A heart-shaped patch on the raincoat”

“A bird is perched on the girl’s shoulder”

Figure 11: Step-by-Step Generation: Add-it can generate images incrementally, allowing it to better adapt to user preferences at each step.

- A.2 ADDITIONAL RESULTS

- In fig. 11 we present step-by-step outputs generated with Add-it. Notice that the scene remains unchanged, while each prompt adds an additional ”layer” to the final image, resulting in a more complex scene.
- In fig. 12 we show additional results from the Additing Affordance benchmark. In each case, the object must be added to a specific location in the source image. Across all examples, Add-it successfully places the object in a plausible location, preserving the natural appearance of the image.

- In fig. 13 we demonstrate that Add-it can operate on non-photorealistic source images, such as paintings and pixel art. Since our method requires no tuning, we preserve all the generation capabilities of the base model.
- In fig. 14 we show various generation results produced by our model, each originating from a different initial noise. Our method preserves the diversity of the base model, enabling users to generate multiple variations of the added object until they find the desired one.

Source Output

Source Output

Source Output

A smartphone displaying a map

A bunny sitting on the chair

The middle dog wearing a hat

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

A bouquet of flowers in the basket

A bottle in the side of the backpack

A woven basket with laundry

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

A parrot on the man’s shoulder

A person pointing at the equation e=mc^2

A flower in her hair above her ear

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

##### Figure 12: Qualitative results of our method on the Additing Affordance Benchmark show that our method successfully adds objects naturally and in plausible locations.

The character holding a diamond sword A knight riding the horse A bird perched on the piano

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

A face painted on the rocket A girl walking in the forest The cat is wearing a wizard hat

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

Figure 13: Our method can operate on non-photorealistic images.

- A.3 THE ROLE OF POSITIONAL ENCODING

Here, we examine the significance of positional encodings in the extended attention mechanism. fig. 15 demonstrates their role through a simple experiment: we applied our method to a source image where the positional encoding vectors were shifted down and to the right. This misalignment resulted in a mismatch between the positional encoding of the child’s head in the source and target images. Consequently, instead of generating headsets at the actual position of the child’s head, the model produced them in the area corresponding to the ”shifted head” position. This outcome

“A dog sitting on the bench”

Source

Variation 1 Variation 2 Variation 3

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

Variation 4 Variation 5 Variation 6 Variation 7

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

- Figure 14: Our method generates different outputs when given different starting noises. All the outputs remain plausible.

Source Image w/ PE Shift w/o PE Shift

[Figure 115]

[Figure 116]

[Figure 117]

- Figure 15: Positional Encoding Analysis: shifting the positional encoding of the source image results in a corresponding shift in the object’s location in the generated image.

demonstrates that the model heavily relies on positional information to transfer features between the source and target images. Despite the target image containing ”laptop” features instead of ”head” features at the relevant location, the model chose to place the headphones there. This decision was based on the area having the same positional encoding as the ”head area” in the source image, rather than on the actual content of the target image at that location. We believe further research on the role of positional encoding vectors is an interesting direction for future work in the context of DiT models.

- A.4 Additing AFFORDANCE BENCHMARK

Dataset Construction Here, we provide the details for constructing the Additing Affordance Benchmark dataset. First, we used ChatGPT-4 to generate a dataset of tuples, each consisting of a source prompt and a target prompt, representing an image before and after object insertion, along with an instruction for the transition and a subject token representing the object to be added. The exact prompt is shown in fig. 18. Next, we used FLUX.1-dev to generate the source images from the source prompts in each tuple. We manually filtered out images where the object had no plausible

Source Output

Source Output

Source Output

A surfing Pikachu in the water

Two sharp knives on the top of the counter

A woman with a blue sweater on

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

Sunglasses placed on one of the pillows

A bicycle leaning against the car

A tea bag placed inside on the spoon

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

- Figure 16: Failure cases: Add-it may fail generating the added object in the right location (sunglasses), it can be biased to replace existing object in the scene (Pikachu) and it can struggle with complicated scenes (woman cooking).

location or too many possible locations, resulting in a dataset of 200 images. Finally, we manually annotated bounding boxes for each image, marking the plausible locations where the object could be added, as shown in fig. 17.

Evaluation protocol: Given a set of an Additing model output images, we use Grounding-DINO to detect the area where new objects were added and set the affordance score of a single image to be the fraction of added object that at least 0.5 of their area falls inside the GT box.

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

A person pointing at the equation e=mc^2

A bottle in the side of the backpack

The teddy bear has a red bow tie

The middle dog wearing a hat

A bunny sitting on the chair

- Figure 17: Visual examples from the Additing Affordance Benchmark. Each image is annotated with bounding boxes highlighting the plausible areas where the object can be added.

- A.5 USER STUDY DETAILS

We evaluate the models through an Amazon Mechanical Turk user study using a two-alternative forced choice protocol. In the study, raters saw an instruction, a source image, and two edited images, each produced by a different approach. They chose the edit that best followed the instruction, taking into account: image quality and realism, instruction following and preservation of the source image. For the evaluation, each head-to-head example was rated by two raters. In fig. 19 we show an example of a single trial a rater has seen.

Please generate a JSON list of 300 sets. Each set consists of: an index, a source prompt, instruction, a target prompt, and a subject token. The source prompt describes a source image. The target prompt describes the source image after an object has been added to it. The instruction is a description of what needs to be changed to go from the source to the target prompt. The subject token is the noun that refers to the added object, a single word that appears in the target prompt. Here are is an example: {

"src_prompt": "A person sitting on a chair", "tgt_prompt": "A scarf wrapped around their neck", "subject_token": "scarf", "instruction": "Wrap a scarf around the person’s neck."

} Only generate examples where there is clearly only one possible place for the object to be added, so it can be tagged correctly. Write it as a JSON list yourself. Please DO NOT include negative examples in your prompts, such as "a man wearing no hat" in the source prompt. DO NOT write code; Return only the JSON list.

Figure 18: The prompt provided to ChatGPT in order to generate the Affordance Benchmark.

[Figure 135]

##### Figure 19: One trial of the Additing user study.

