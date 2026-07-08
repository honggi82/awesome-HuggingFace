# arXiv:2409.08857v2[cs.CV]30Nov2024

## InstantDrag: Improving Interactivity in Drag-based Image Editing

JOONGHYUK SHIN, Seoul National University, South Korea DAEHYEON CHOI, POSTECH, South Korea JAESIK PARK, Seoul National University, South Korea

Image + Drag Instr. Result 2nd Edit Result Time (s) Memory (GB)

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

Drag Diffusion

Dragon Diffusion

SDE Drag

Readout Guidance

Ours

Fig. 1. We introduce InstantDrag, an efficient framework that combines an optical flow generator with a motion-conditioned diffusion model. Our approach enables realistic drag edits in roughly a second, leveraging real-world video datasets. (Inputs: ○c Louvre Museum1, ○c A. Shvets and P. Danilyuk, via Pexels)

Drag-based image editing has recently gained popularity for its interactivity and precision. However, despite the ability of text-to-image models to generate samples within a second, drag editing still lags behind due to the challenge of accurately reflecting user interaction while maintaining image content. Some existing approaches rely on computationally intensive per-image optimization or intricate guidance-based methods, requiring additional inputs such as masks for movable regions and text prompts, thereby compromising the interactivity of the editing process. We introduce InstantDrag, an optimization-free pipeline that enhances interactivity and speed, requiring only an image and a drag instruction as input. InstantDrag consists of two carefully designed networks: a drag-conditioned optical flow generator (FlowGen) and an optical flow-conditioned diffusion model (FlowDiffusion). InstantDrag learns motion dynamics for drag-based image editing in real-world video datasets by decomposing the task into motion generation and motion-conditioned image generation. We demonstrate InstantDrag’s capability to perform fast, photo-realistic edits without masks or text prompts through experiments on facial video datasets and general scenes. These results highlight the efficiency of our approach in handling drag-based image editing, making it a promising solution for interactive, real-time applications.

CCS Concepts: • Computing methodologies → Computer vision; Image manipulation.

Permission to make digital or hard copies of part or all of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for third-party components of this work must be honored. For all other uses, contact the owner/author(s).

SA Conference Papers ’24, December 3–6, 2024, Tokyo, Japan © 2024 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-1131-2/24/12. https://doi.org/10.1145/3680528.3687668

Additional Key Words and Phrases: Interactive image editing, Drag-based image editing, GAN, Diffusion models, Optical flow

##### ACM Reference Format:

Joonghyuk Shin, Daehyeon Choi, and Jaesik Park. 2024. InstantDrag: Improving Interactivity in Drag-based Image Editing. In SIGGRAPH Asia 2024 Conference Papers (SA Conference Papers ’24), December 3–6, 2024, Tokyo, Japan. ACM, New York, NY, USA, 17 pages. https://doi.org/10.1145/3680528.3687668

1 INTRODUCTION

Recent advancements in diffusion models [Ho et al. 2020; Song et al.

- 2021] have empowered users to transform their creative visions into digital images using only a few text prompts. Further research has proposed faster ODE solvers [Lu et al. 2022a], consistency distillation [Song et al. 2023], straightening of ODE trajectories [Liu et al.
- 2022], and adversarial learning [Sauer et al. 2023] to improve inference speed. However, while text-to-image generation can now be achieved in a fraction of a second, editing real images with precision and interactivity remains relatively slow and less performant.

Text-guided editing techniques allow users to easily modify highfrequency features and the overall mood of an image, but accurately pinpointing the region of interest for editing remains challenging. Drag editing, introduced in the pioneering work DragGAN [Pan et al. 2023], directly operates in pixel space, granting users more precise control and the ability to edit low-frequency features, including motion-aware structural changes. This increased editing capability comes at the cost of increased complexity. Specifically, drag-based

1○c 2011 GrandPalaisRmn (musée du Louvre) / Michel Urtado. Mona Lisa. Louvre Collections: https://collections.louvre.fr/ark:/53355/cl010062370

editing requires the model to have a comprehensive understanding of potential motion information, posing different challenges compared to text-based modifications. Consequently, state-of-theart approaches usually start from LoRA training [Hu et al. 2021], DDIM inversion [Song et al. 2020] and perform computationally intensive latent optimization or intricate score guiding to capture and manipulate motion dynamics.

Despite the core motivation of drag-based image editing being the interactivity of the editing process, we observe that most existing works are based on complex optimization techniques, which can fundamentally slow down the editing speed and hinder interactivity. To address this, we propose a new approach that focuses on four key aspects: 1) speed, 2) real image editing quality, 3) removing the reliance on user input masks to define movable regions, and 4) eliminating the necessity of text prompts.

To handle these challenges, we present InstantDrag, a pipeline designed to enhance interactivity and speed without optimization, taking only an image and a drag instruction as input. We decouple the drag-editing task into motion generation and motion-conditioned image generation, assigning each task to a generative model with the appropriate capacity and design. FlowGen, a lightweight and fast GAN-based model, generates dense optical flow from sparse user inputs. FlowDiffusion, an efficiently tuned optimization-free diffusion model, performs high-quality edits conditioned on the generated motion. Our method achieves high-quality results in roughly a second by streamlining the architecture and removing redundant components such as the text encoder in the diffusion model.

One significant challenge in this setting is the lack of fine-grained paired datasets of drag instructions and edited images. The naïve use of video datasets can result in the undesired movements of nontarget objects, including the background, reducing the precision. We carefully analyze and ablate motion cues in videos and experimentally demonstrate the effectiveness of our training strategy in learning motion dynamics for drag-based image editing.

We conduct comprehensive experiments to validate the strength of each module in InstantDrag. Our experiments on facial video datasets and general scenes demonstrate that our method can deliver photo-realistic edits on real images without requiring metadata, such as user input masks or text prompts. Compared to recent works, editing real images with InstantDrag is up to ∼75× faster while consuming up to ∼5× less GPU memory. Fig. 1 shows our results and compares computational complexity with other approaches.

- 2 RELATED WORK

- 2.1 Image Editing with Generative Models

Generative adversarial networks (GANs) have achieved significant successes in image generation [Goodfellow et al. 2014], laying the foundation for several seminal works in the field of image editing. One prominent approach, image-to-image translation, focuses on mapping an image from a source domain to a target domain. Numerous methods have been proposed for both paired settings [Isola et al.

- 2017] and unpaired settings [Zhu et al. 2017]. These approaches typically employ U-shaped networks to directly transform images with the aid of a discriminator to enhance finer details. A combination of reconstruction and adversarial losses guides the transformation

process. Other approaches involve inverting an input image back to a continuous latent space of a pre-trained GAN and perform editing guided by informative features from external sources. However, editing general images with more complex instructions, such as text cues, remains challenging due to the inherent difficulties in scaling up GANs. Consequently, the generalizability of these methods is constrained by the representative power of GANs.

Rapid developments in large-scale text-to-image diffusion models [Ramesh et al. 2022; Rombach et al. 2022] have significantly improved image quality for general scenes by training on web-scale datasets [Schuhmann et al. 2022]. Building upon these diffusion models, several approaches have been proposed to enable text-guided image editing [Brooks et al. 2023; Hertz et al. 2022; Meng et al. 2021; Mokady et al. 2023; Parmar et al. 2024]. Recent advancements have extended these capabilities to 3D and movement-aware edits by leveraging 3D-lifted activations and learning from dynamic video data [Alzayer et al. 2024; Pandey et al. 2024]. Compared to GAN-based methods, diffusion-based generation and editing offers better stability and generalizability, albeit at the cost of increased computational load and memory consumption.

2.2 Drag-based Image Editing

DragGAN [Pan et al. 2023] introduced an interactive image editing technique where users perform motion-aware edits by dragging source points to target points. This method optimizes the W+ latent space of StyleGAN2 [Karras et al. 2020] using iterative motion supervision and point tracking. Given source points {𝑠𝑖 = (𝑥𝑠,𝑖,𝑦𝑠,𝑖)} and target points {𝑡𝑖 = (𝑥𝑡,𝑖,𝑦𝑡,𝑖)}, motion supervision moves a small patch around 𝑠𝑖 to 𝑡𝑖. The motion supervision loss is:

∑︁𝑛

#### ∑︁

#### ∥𝐹 (𝑞𝑖) − 𝐹 (𝑞𝑖 + 𝑑𝑖)∥1 + 𝜆 ∥(𝐹 − 𝐹0) · (1 − 𝑀)∥1 .

L =

𝑖=0

𝑞𝑖 ∈Ω1(𝑠𝑖,𝑟1)

(1)

The first term drives pixel features around 𝑠𝑖 to move towards 𝑡𝑖 along 𝑑𝑖, a normalized vector directing from source to target, and the second term maintains the unmasked features close to the initial features 𝐹0. Features 𝐹 (𝑥) are from StyleGAN2’s 6th block. After back-propagating to update the latent code 𝑤′, point tracking finds the new source point 𝑠𝑖′ via nearest neighbor search:

𝑠𝑖′ := argmin

∥𝐹 (𝑞𝑖) − 𝐹0(𝑠𝑖)∥. (2)

𝑞𝑖 ∈Ω2(𝑠𝑖,𝑟2)

Several works have extended this iterative latent optimization framework using motion supervision and point tracking. DragDiffusion [Shi et al. 2023] adapts DragGAN to text-to-image diffusion models, enabling editing of general scenes. FreeDrag [Ling et al. 2023] addresses ambiguous point tracking by incorporating adaptive template features and line search with backtracking. More recent approaches such as GoodDrag [Zhang et al. 2024], Drag Your Noise [Liu et al. 2024b], StableDrag [Cui et al. 2024], and EasyDrag [Hou et al. 2024] enhance the DragDiffusion framework by alternating drag and denoising operations, leveraging diffusion semantic propagation, introducing discriminative point tracking methods with confidence-based motion supervision, and applying more stable motion supervision. Other approaches focus on guidance from external networks [Luo et al. 2024; Mou et al. 2023], or

fGT

FlowGen

f

{I1,s(f),f}

|[Figure 14]|
|---|

[Figure 15]

{I1,s(f)}

|[Figure 16]|
|---|

[Figure 17]

[Figure 18]

G D

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

R/F

[Figure 27]

|[Figure 28]|
|---|

Ladv

~ 0.003s ~ 1s

I

Image, User Drag Instruction

I

FlowDiffusion

fpred

Fig. 2. Illustration of our inference pipeline. Given sparse user drag input, our FlowGen estimates dense optical flow, and our FlowDiffusion edits the original image with the flow guidance. Our approach does not require auxiliary input, such as texts or foreground masks. Our approach is inversion and optimization-free, providing the edited image in about a second.

Fig. 3. Illustration of our FlowGen architecture (Sec. 3.1.1). Sparse user drag input is channel-wise concatenated with the input image and fed into the generator, which predicts dense optical flow. Based on a Pix2Pix-like GAN architecture, FlowGen is trained using the adversarial loss from the discriminator and the reconstruction loss from the generator.

copy & paste style latent manipulation with SDE formulation [Nie et al. 2023].

- 3.1.1 FlowGen. Our rationale for employing a GAN in motion generation is based on conceptualizing this task as a translation problem, where the goal is to map an RGB image with drag instructions (sparse flow) to a dense optical flow. Inspired by the success of GAN-based models like Pix2Pix [Isola et al. 2017], we recognize that this translation task can be efficiently performed with a one-step generative model, which also helps reduce the editing time. Consequently, we opted to train our model from scratch in a Pix2Pix-style manner. As illustrated in Fig. 3, our generator receives a total of 5 channels of input data: 3 channels for the input image and 2 channels for the condition sparse drag instructions. It outputs a 2-channel dense optical flow. On the other hand, the discriminator processes a 7-channel input comprising 3 channels of input image, 2 channels of sparse flow, and 2 channels of dense flow.

To accelerate training and increase the model capacity for larger datasets, we employ GroupNorm instead of InstanceNorm and utilize a deeper architecture for the PatchGAN-based discriminator. To encourage the generator to learn a robust mapping from various sparse flows to dense flow, we update the generator four times using randomly sampled sparse flows for each update of the discriminator. More details can be found in the Appendix. Given the initial frame 𝑥, dense optical flow 𝑓 , and a sparse flow 𝑓𝑠 sampled using the sampler 𝑆 (i.e., 𝑓𝑠 = 𝑆(𝑓 )), FlowGen’s adversarial L𝑎𝑑𝑣(𝐺,𝐷) and reconstruction L𝑟𝑒𝑐(𝐺) losses can be written as follows:

L𝑎𝑑𝑣 = E𝑥,𝑓 [log𝐷(𝑥, 𝑓𝑠, 𝑓 )] + E𝑥,𝑓 [log (1 − 𝐷(𝑥, 𝑓𝑠,𝐺(𝑥, 𝑓𝑠)))] , L𝑟𝑒𝑐 = E𝑥,𝑓 ∥𝑓 − 𝐺(𝑥, 𝑓𝑠)∥2 . (3)

- 3.1.2 FlowDiffusion. Making changes to reflect the motion condition is known to be challenging and time-consuming. Specifically, directly adopting off-the-shelf optical flow networks to guide the denoising process can take dozens of minutes [Geng and Owens 2023]. Therefore, we decide to train a diffusion model specifically designed to accept dense optical flow as a condition. Our baseline is set by Instruct-Pix2Pix [Brooks et al. 2023], which incorporates conditioning of the input image and text prompts in fine-tuning Stable Diffusion. As detailed in Sec. 3.2.4, we discovered that simply concatenating the optical flow channels with the input image channels is highly effective, provided the optical flow is properly normalized. As shown in Fig. 4, our U-net now has a 10-channel input, consisting of 4-channel latent noise, 4-channel latent image, and 2 channels for optical flow.

While these approaches demonstrate impressive quality edits, they have limitations in terms of interactivity. A typical latent optimization-based drag editing framework, like DragDiffusion, comprises 1) LoRA training of U-Net for enhanced identity preservation, 2) DDIM inversion for real images, 3) latent optimization using motion supervision and point tracking, and 4) diffusion sampling process. These steps can take tens of seconds to minutes, depending on the drag instruction and image size. Similarly, guidance-based methods like DragonDiffusion and Readout Guidance also encounter computational bottlenecks. These methods require DDIM inversion followed by a long sampling process, guided by either an additional diffusion branch or backpropagated gradients from external networks. For some works that involve manual drawing of movable region masks and prompt engineering, the actual editing process can be even longer. Furthermore, during the DDIM inversion process, high-frequency details are not accurately inverted, resulting in notable performance degradation with real images.

To address these limitations, we aim to build a more interactive pipeline where images can be dragged without optimization and metadata by training dedicated models. Concurrent work [Li et al. 2024] adopts a similar optimization-free editing idea by training on synthetic objects, but it primarily focuses on part-level movements of relatively simple objects with white backgrounds. In contrast, our work focuses on dragging real-world images with complex backgrounds. Our motion-conditioned diffusion model shares similarities with motion guidance [Geng and Owens 2023] in its utilization of optical flow. However, their training-free method uses a differentiable off-the-shelf optical flow network to guide the sampling process. In contrast, our method primarily trains a dedicated diffusion model to accept flow conditions and perform classifier-free guidance [Ho and Salimans 2021], resulting in faster sampling.

- 3 METHOD

- 3.1 Model Architectures

We tackle drag-editing by dividing the task into two components: motion generation and motion-conditioned image generation. Each component is handled by a specialized network: a GAN-based network, FlowGen, for motion generation, and a diffusion-based network, FlowDiffusion, for motion-conditioned image generation. The overview of our inference pipeline is illustrated in Fig. 2.

Input Image Generated Optical Flows FlowDiffusion output

GT

|[Figure 29]<br><br>A|[Figure 30]<br><br>B|
|---|---|
|[Figure 31]<br><br>C|[Figure 32]<br><br>D|

|[Figure 33]<br><br>A|[Figure 34]<br><br>B|
|---|---|
|[Figure 35]<br><br>C|[Figure 36]<br><br>D|

|[Figure 37]|
|---|

|[Figure 38]|
|---|

[Figure 39]

θ

[Figure 40]

[Figure 41]

Decoder

Encoder

[Figure 42]

|[Figure 43]|
|---|

| |
|---|
| |

Trainable Frozen

Pred

Fig. 4. Illustration of our FlowDiffusion architecture (Sec. 3.1.2). The denoising U-Net of FlowDiffusion takes encoded image and downscaled optical flow as inputs. It leverages channel-wise concatenated input image and optical flow to guide the denoising process, learning to predict subsequent video frames in the latent space based on motion information.

Fig. 5. Dragging results from FlowGen trained under four settings: (A) Stochastic sampling strategy (Sec. 3.2.2), (B) 1 fixed point (nose), (C) 100 fixed grid points, (D) 900 fixed grid points. Excessive points (C, D) generate sparse motion while a single point (B) causes undesired movements. We find (A) to be the most robust, combining the advantages of the other approaches.

To demonstrate that our method is not limited to facial editing, we also pre-train our model on the widely used optical flow datasets FlyingChairs [Dosovitskiy et al. 2015] and SINTEL [Butler et al. 2012]. For general scenes, due to the computational constraints on training on full web-scale video datasets, we propose a novel setting where a short video (10∼60s) is provided for model fine-tuning at test time. The resulting dataset is composed of 𝑛 pairs, each consisting of two images, two masks, and an optical flow directing frame 1 to 2: {𝐼1,𝑖,𝐼2,𝑖,𝑀1,𝑖,𝑀2,𝑖, 𝑓𝑖}𝑛𝑖=1. For training FlowGen and FlowDiffusion, we use masked optical flow 𝑓𝑖 · 𝑀1,𝑖 as GT to capture the object’s motion exclusively.

A key difference in our approach compared to Instruct-Pix2Pix is that our edit signal is encoded in additional channels for the flow dimension, not in the text prompt. While Instruct-Pix2Pix needs to reflect signals from both text and image domains, our model needs to maintain consistency except for the dragged regions, reflecting only the dense flows without relying on textual input.

We experimented with different guidance and training mechanisms, finding out that replacing text tokens with null tokens works reasonably well. This approach also has the advantage of reducing the computational costs associated with text encoders. Given our denoising network 𝜖𝜃, a noisy latent 𝑧𝑡, encoded input image 𝑐𝐼, optical flow 𝑐𝐹, and guidance scales (𝑠𝐼,𝑠𝐹) for each conditional input, FlowDiffusion’s classifier-free guidance with input image and optical flow can be written as follows:

3.2.2 Pseudo Drag Instructions for FlowGen. Drag-based manipulation is an inherently ill-posed problem where there can be multiple valid movements for a given input. User inputs can range from a single point to an extensive set of points. We expect our model to be capable of handling this wide variation to generate plausible motion. Providing a single sparse point to encapsulate overall movements would require the generator to infer the entire motion across the image. For example, moving the nose might necessitate adjustments in other parts, like the eyes, leading to the stitching of uncorrelated movements. On the other hand, providing thousands of dense points, where each point captures local motion, would result in very localized movements. Achieving accurate results at test time with this approach would require an impractically large number of points (see Fig. 5). Additionally, we need to sample points from areas of interest to ensure meaningful motion generation. Sampling points outside the target object would be ineffective.

𝜖˜𝜃 (𝑧𝑡,𝑐𝐼,𝑐𝐹) = 𝜖𝜃 (𝑧𝑡, ∅, ∅)

#### + 𝑠𝐼 · [𝜖𝜃 (𝑧𝑡,𝑐𝐼, ∅) − 𝜖𝜃 (𝑧𝑡, ∅, ∅)]

+ 𝑠𝐹 · [𝜖𝜃 (𝑧𝑡,𝑐𝐼,𝑐𝐹) − 𝜖𝜃 (𝑧𝑡,𝑐𝐼, ∅)]. (4) Unlike the image and text prompt relationship in the InstructPix2Pix pipeline, where conditional dropout occurs for 10% in each case, with a 5% overlap for dropping both, dropping the image only while using flow as input does not intuitively make sense. Therefore, for training of FlowDiffusion, we drop the image condition 5% of the time and the flow condition 10% of the time, ensuring the model is not conditioned solely on flow without the image.

3.2 Implementation Details

To address these challenges, we conducted an extensive search on sampling strategies. We first initialize the sparse flow 𝑓𝑠 ∈ R2×ℎ×𝑤, with random values sampled from 𝑈 (0, 1). Then, we assign −∞ values to masked regions (background) while adding a value 𝑣𝑔 to grid locations, and 𝑣𝑠 to specific points such as facial keypoints. Finally, we select top-𝑘 points in the sparse flow and zero out the remaining parts, where 𝑘 is also randomly chosen in predefined range. It is important to note that the grid size plays a crucial role in preventing points from being selected only in certain areas. We use a subset of dlib keypoints as special points for facial videos and apply grid-based sampling exclusively for fine-tuning general scenes. To provide flexibility, we offer different GAN configurations depending on the user’s intentions. For example, users can choose configurations for single-point dragging, keypoint-based fine-grained motion

- 3.2.1 Preparing Datasets. The main challenge in training a drag-

dedicated model, (i.e. 𝑜𝑢𝑡𝑝𝑢𝑡 = 𝑚𝑜𝑑𝑒𝑙(𝑖𝑛𝑝𝑢𝑡,𝑐𝑑𝑟𝑎𝑔)), is the lack of a curated dataset composed of triplets: input image, output image, and drag condition. Due to the difficulty of obtaining such edit pairs in image domains, we resort to video datasets.

From each video, we extract frames and randomly apply a sliding window technique to sample pairs. Choosing an appropriate window size is crucial, as it determines the extent of motion that the model can learn; hence, it must be large enough to encapsulate realistic motions. For the large-scale facial video dataset CelebV-Text [Yu et al. 2023], we sample frames at 10fps and use a maximum interval of 8 for sampling two pairs. Optical flows between these pairs are extracted using FlowFormer [Huang et al. 2022], and segmentation models are employed to obtain masks of the objects of interest.

Image 1 (𝐼 = 𝐼 ) Image 2 (𝐼 = 𝐼 ) Composited Image (𝐼 )

[Figure 44]

[Figure 45]

[Figure 46]

Fig. 6. Visualization of the mask operation in Sec 3.2.3. 𝐼𝑛𝑒𝑤 combines the object from 𝐼1 and the background from 𝐼2. Blue contour shows 𝐼2’s mask and green contour shows its dilated mask.

editing, or extremely fine-grained editing such as hair. This plugand-play flexibility of FlowGen allows our model to handle varying levels of detail in user inputs, ensuring robust motion generation. Further details are available in the Appendix.

- 3.2.3 Enforcing Background Consistency for FlowDiffusion. Another difficulty in directly adopting natural videos is maintaining background consistency. Unlike video generation, where any plausible movements are acceptable, drag-based image editing requires a consistent background and allows movements only in the dragged object. By using the binary mask obtained in Sec.3.2.1, we can enforce background consistency with mask-based operations.

Consider two frames, 𝐼1 = 𝐼𝑓 𝑔 and 𝐼2 = 𝐼𝑏𝑔, with corresponding masks 𝑀1 = 𝑀𝑓 𝑔 and 𝑀2 = 𝑀𝑏𝑔. We aim to create a composite image combining the object from 𝐼𝑓 𝑔 with the background from 𝐼𝑏𝑔. To account for imperfect masks, we dilate 𝑀𝑏𝑔 using a 15x15 kernel to expand the coverage of 𝐼𝑓 𝑔 in boundary regions. We define

inverted masks 𝑀𝑓 𝑔inv = 1−𝑀𝑓 𝑔 and 𝑀𝑏𝑔inv = 1−𝑀𝑏𝑔dilated to hold 1 for background and 0 for the object. Our mask operation is as follows:

𝐼𝑓 𝑔new = 𝐼𝑓 𝑔 · 𝑀𝑓 𝑔 (5) 𝐼𝑏𝑔new = (𝑀𝑓 𝑔inv ∧ 𝑀𝑏𝑔inv) · 𝐼𝑏𝑔 + [1 − (𝑀𝑓 𝑔 ∨ 𝑀𝑏𝑔inv)] · 𝐼𝑓 𝑔 (6) 𝐼new = 𝐼𝑓 𝑔new + 𝐼𝑏𝑔new (7)

The visualization of our method can be found in Fig. 6. We explored three approaches: (1) compositing 𝐼2’s background with 𝐼1’s object, (2) compositing 𝐼1’s background with 𝐼2’s object, and (3) no background consistency. Ablation studies showed the first approach performed best. The intuition is as follows: when the diffusion model learns to transform 𝐼1 to 𝐼2 using flow 𝑓 , it uses 𝐼1 and 𝑓 as conditions and calculates the loss between its output and 𝐼2. Although mask operations are generally robust, modifying the ground truth 𝐼2 could lead to training with perturbed targets, potentially causing artifacts at test time. In contrast, modifying the input condition 𝐼1 allows the model to handle and ignore these perturbations during the denoising process, resulting in a proper reconstruction of 𝐼2. For this reason, we set 𝐼1 as the foreground image and 𝐼2 as the background image. Note that masks are only used during training, not inference.

- 3.2.4 Optical Flow Normalization. Conditioning on optical flow directly without a warping operation is a relatively unexplored area.

[Figure 47]

Density

Loss (L1)

Input GT

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

(A) Fixed size norm. (B) Sample-wise norm.

(a)

(b) (c)

Fig. 7. (a) Flow distribution after normalization. (b) L1 Reconstruction loss of FlowGen for two normalization methods. (c) Inference results of FlowDiffusion, showing that sample-wise normalization (B) does not consider the actual flow scale. Details are described in Sec. 3.2.4.

An optical flow 𝑓 ∈ R2×ℎ×𝑤 can range from (−ℎ,ℎ) or (−𝑤,𝑤). Directly calculating a loss or conditioning on this scale can lead to numerical errors due to the sensitive scales of model weights. Additionally, resizing the flow to the latent space causes scale inconsistencies. Therefore we consider two variants: (1) fixed size normalization where we divide 𝑓 by the spatial dimensions and (2) sample-wise normalization where we divide each channels by its absolute maximum value. An advantage of fixed size normalization is that it preserves the actual size and scale, making the numbers directly proportional to the dimensions. However, as shown in Fig. 7, this results in a very narrow distribution with all samples densely clustered around 0. In contrast, sample-wise normalization results in a relatively wide-spread distribution, although it doesn’t provide any indication of the actual flow size.

For FlowGen, sample-wise normalization is more effective as the loss is directly calculated on the flows. We empirically found that avoiding excessively small flow scales leads to more stable loss, faster convergence, and more robust outputs. For FlowDiffusion, fixed size normalization proves to be more stable as the model benefits from the knowledge of actual scale. Using per-sample normalization for FlowDiffusion sometimes result in movements that are too large or too small. During inference, we normalize the sparse flow per sample, rescale the output dense flow using the maximum absolute magnitude of the input sparse drag instruction, and apply fixed size normalization before feeding it into the diffusion model. These normalization processes enable the diffusion model to generate good results without a specialized encoder network for optical flow.

4 EXPERIMENTS 4.1 Settings

For face editing, we train our models on CelebV-Text [Yu et al. 2023], a large-scale dataset of 70,000 high-quality video clips from the web. We extract frames at 10fps and use the sliding window technique to sample frame pairs. We use FlowFormer for optical flow estimation and YOLO [Redmon et al. 2016] for fast mask generation. After filtering out noisy samples, we obtain a dataset of 8M pairs. We analyze model trends and characteristics in this dataset, including ablation experiments on the dataset size. For general scene editing, due to the aforementioned computational limitation, we opt for a two-stage strategy where the model is fine-tuned on short videos (10∼60s) containing scenes and motions relevant to the input image.

DragGAN (1st row) Instruct-Pix2Pix (rest)

Image + Drag Instr. DragDiffusion DragonDiffusion SDE-Drag Readout Guidance Ours

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

- Fig. 8. Qualitative comparison with other models. Zoom in for finer image details and drag instructions. (Second row input: ○c V. Karpovich, via Pexels)

For the user study, we sample 22 dragged results from various domains, including cartoons, drawings, and real-world images, with 12 being facial images and 10 being general scenes. Although the recently introduced DragBench [Shi et al. 2023] includes images from 10 general categories, we opt for alternative data since our approach requires short video clips for general scenes. Therefore, we perform additional qualitative evaluations on face manipulation following DragGAN and DragonDiffusion. To obtain ground truth for comparison, we use frames from the validation set of TalkingHead1KH [Wang et al. 2021] and report PSNR, SSIM, LPIPS [Zhang et al.

images generated at 512px from authors’ official implementations with default settings. More details on dataset generation and training are elaborated in the Appendix.

4.2 Qualitative Evaluation

We compare our method with other recent approaches. For DragGAN, as stated in [Zhang et al. 2024], we observe that PTI inversion [Roich et al. 2022] of real images and subsequent editing shows suboptimal performance, even with minor distribution shifts. Due to these difficulties, we include DragGAN only with the lion sample in Fig. 8 and provide the Instruct-Pix2Pix baseline, using text descriptions to move the object.

- 2018], and CLIP image similarity scores [Radford et al. 2021] using 68 dlib keypoint-based editing.

The final number of parameters for FlowGen generator is 54M and FlowDiffusion is 860M. We perform all experiments on A6000 GPUs. We train FlowGen at 256px for approximately three days using a single GPU and train FlowDiffusion at 512px for roughly five days using 8 GPUs. Fine-tuning FlowGen and FlowDiffusion on a single video takes about 20 minutes. For sampling, we use DPM++ sampler [Lu et al. 2022b] with 20 steps. For comparisons, we use

The first observation is about the preservation of high-frequency features. Most methods employ DDIM inversion to convert the real image. However, as shown in Fig. 8 and Fig. 11, this process often results in a significant loss of detailed information. Without explicitly overfitting a network using techniques like LoRA or utilizing masks to set fixed regions, naïvely guiding with text can lead to a change in identity. As can be seen in Fig. 12, this issue is particularly

Table 1. Comparison of different drag editing methods using TalkingHead-1KH data. We mark the best and second-best scores in bold and underlined numbers. O and E refer to scores calculated using the original and edited images, respectively. * denotes results taken from the original paper.

PSNR (↑) SSIM (↑) LPIPS (↓) CLIP𝑖𝑚𝑔 (↑) O E O E O E O E

Input Time (s) Mem. (GB)

DragDiffusion Image, Drag, Prompt, Mask 75.3 11.6 23.83 23.59 0.81 0.77 0.194 0.216 0.957 0.945 DragonDiffusion Image, Drag, Prompt, Mask 11.0 6.6 21.53 21.26 0.80 0.74 0.233 0.260 0.895 0.891 SDE-Drag Image, Drag, Prompt, Mask 53.0 7.4 15.71 15.40 0.68 0.61 0.347 0.389 0.666 0.665 Readout Guidance Image, Drag, Prompt 55.4∗ 19.2∗ 25.44 21.26 0.87 0.71 0.205 0.289 0.892 0.885 InstantDrag (Ours) Image, Drag 1.1 3.4 26.51 22.92 0.85 0.75 0.154 0.224 0.957 0.948

[Figure 89]

Dragon Diffusion

Readout Guidance

DragDiffusion SDE-Drag Ours

Fig. 9. Bar plots of human evaluation results.

noticeable for methods like Readout Guidance, where neither LoRA training nor masking is applied. Our method, being inversion-free, excels at preserving fine details even without the use of masks.

The second observation is the exceptional generalization ability of our model. Although trained solely on real-world facial videos, our method effectively generalizes to domains like drawings and cartoons as shown in Fig. 1, 8, and 13. Interestingly, though in preliminary stages, our model also demonstrates capability in handling non-facial general objects and scenes without fine-tuning. For a more in-depth discussion on the generalizability of our model, please refer to the Appendix.

Lastly, we observe that our model performs well in generating fine-grained movements, such as facial expressions, as shown in Fig. 1, 8 and 13. We attribute this to FlowGen’s capability to accurately generate pixel-level motion cues, which efficiently guides the denoising process of FlowDiffusion.

- 4.3 Quantitative Evaluation

- Fig. 9 presents human evaluation results based on 66 responses. Participants rated each model’s output on a scale from 1 (very poor) to 5 (very good) across three criteria: instruction-following, identity preservation, and overall preference. The overall preference rating considered both instruction-following and identity preservation comprehensively, reflecting participants’ inclination to use each model for drag-based image editing tasks.

Even without considering the advantage of our models being additionally fine-tuned on short videos for general scenes, they excel in zero-shot facial editing and overall preference. We attribute this to our model’s strong ability to preserve high-frequency features and generate fine-grained movements. By learning motion cues from real-world data, our model generates plausible images that balance instruction-following and consistency, which is particularly

important for human faces, where people are highly sensitive to subtle differences.

Table 1 presents the evaluation results on 100 frame pairs from the TalkingHead-1KH dataset. We generate drag instructions using 68 dlib keypoints between two sampled frame pairs, serving as the original and oracle edited images. For models requiring masks or text prompts, we manually annotate masks and generate captions using LLaVA [Liu et al. 2024a]. While Mean Distance (MD) has been recently used, we could not employ it due to noisy or out-ofdistribution outputs from some models, which hindered keypoint detection. Moreover, we observed cases where models generated implausible images that lacked consistency and realism, merely moving keypoints without considering the context. To address these issues, we report results calculated with respect to both the original and edited images, with the latter representing the ground truth movement. Scores calculated against original images indicate content preservation, while those calculated against edited images reflect the resemblance to actual ground truth movement. Quantitative scores and qualitative inspection suggest that DragDiffusion and InstantDrag generally outperform other models. Notably, DragDiffusion, a point tracking and optimization-based method, accurately moves points to exact locations, whereas our model prioritizes consistency and plausible movements.

We additionally perform ablation experiments to validate the effect of image and flow guidance scales as shown in Fig. 10.

4.4 Discussion and Limitation

After a thorough evaluation, we identify the following strengths of our method: 1) it excels at preserving consistency, especially high-frequency features, even without a mask, due to being an inversion-free method; 2) it generates plausible images with realistic motions; and 3) it offers lighter and more efficient model pipeline and increased interactivity due to requiring fewer user inputs. Nevertheless, we also note a few limitations. Since our model’s learning is based on the output of an optical flow estimation network, it faces difficulty handling very large motions that optical flow networks cannot capture. Additionally, while our model shows promising generalizability (as detailed in the Appendix), we occasionally observe limitations in preserving identity or creating accurate motions for non-facial scenes without fine-tuning, due to the model being trained solely on facial videos. However, given the positive signs of generalizability, we believe training on diverse motions from larger datasets can further improve performance across various domains.

[Figure 90]

[Figure 91]

Target Input Drag Instr. Output

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

0.96

0.95

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

0.94

0.93

Image Guidance Scale (sF=1.5) Flow Guidance Scale (sI=1.5)

(a) (b)

- Fig. 10. (a) Ablation studies on image and flow guidance using a subset of TalkingHead-1KH. Increasing image guidance makes generated images more like the original, while increasing flow guidance causes them to diverse from the original. Another scale is fixed at 1.5. (b) Results for face landmark manipulation.

Input + Drag Instr. (1st row)

Masked Input (2nd row) DragDiffusion DragonDiffusion SDE-Drag

[Figure 101]

[Figure 102]

[Figure 103]

w/omaskw/mask

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

Ours

[Figure 110]

- Fig. 11. Generated samples with and without masking. As an inversion-free method, InstantDrag produces consistent drag edits without using a mask. (Input: ○c Louvre Museum2)

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

Original

Image Denoised

w/oRG Denoised

w/RG

t=0 t=30 t=60 t=90 t=100

Noisy Image

- Fig. 12. DDIM inversion and subsequent denoising process (100 step) in Readout Guidance (RG). We observe that fine details are often lost during inversion. Additionally, guidance-based methods can be sensitive to prompts and may exhibit some inconsistency.

drag editing paradigm, we proposed InstantDrag, a drag-dedicated pipeline composed of carefully designed and trained FlowGen and FlowDiffusion. Our method not only achieves faster speeds and a lighter model but also enhances interactivity by reducing inputs to just an image and a drag instruction. This potentially decreases the editing time even further. Given the intuitive nature and interactivity of drag editing, we believe our work points in a promising direction for improving the accessibility and efficiency of real-time interactive editing on a wide range of devices.

ACKNOWLEDGMENTS

This work was supported by the National Research Foundation (NRF) grant (RS-2024-00405857, 95%) and the IITP grant (RS-2021II211343: AI Graduate School Program, Seoul National University, 5%) funded by the Korean government (MSIT).

REFERENCES

Hadi Alzayer, Zhihao Xia, Xuaner Zhang, Eli Shechtman, Jia-Bin Huang, and Michael Gharbi. 2024. Magic Fixup: Streamlining Photo Editing by Watching Dynamic Videos. arXiv preprint arXiv:2403.13044 (2024).

Tim Brooks, Aleksander Holynski, and Alexei A Efros. 2023. Instructpix2pix: Learning to follow image editing instructions. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Daniel J Butler, Jonas Wulff, Garrett B Stanley, and Michael J Black. 2012. A naturalistic open source movie for optical flow evaluation. In European Conference on Computer Vision (ECCV).

Yutao Cui, Xiaotong Zhao, Guozhen Zhang, Shengming Cao, Kai Ma, and Limin Wang.

2024. StableDrag: Stable Dragging for Point-based Image Editing. arXiv preprint arXiv:2403.04437 (2024).

Alexey Dosovitskiy, Philipp Fischer, Eddy Ilg, Philip Hausser, Caner Hazirbas, Vladimir Golkov, Patrick Van Der Smagt, Daniel Cremers, and Thomas Brox. 2015. Flownet: Learning optical flow with convolutional networks. In IEEE International Conference on Computer Vision (ICCV).

Daniel Geng and Andrew Owens. 2023. Motion Guidance: Diffusion-Based Image Editing with Differentiable Motion Estimators. In International Conference on Learning Representations (ICLR).

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative adversarial nets.

- 5 CONCLUSION

- 2○c 2011 GrandPalaisRmn (musée du Louvre) / Michel Urtado. Mona Lisa. Louvre Collections: https://collections.louvre.fr/ark:/53355/cl010062370
- 3○c National Museum of Korea. Treasure Lee Chae Portrait (2006). Public Nuri, KOGL Type 1. Link: https://www.museum.go.kr/site/main/relic/search/view?relicId=1138

While large-scale foundational models are being distilled to generate high-quality images from texts within a second, adding precise controls to generated images still lags behind. Inspired by the recent

Image + Drag Instr. DragDiffusion DragonDiffusion SDE-Drag Readout Guidance Ours

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

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

- Fig. 13. Additional qualitative comparison with other models. 1∼4th row inputs: ○c J. Loiterton (Pexels), ○c National Museum of Korea3, ○c mds524680 (Pixabay), and DragBench ○c [Shi et al. 2023].

Original Image Edited Image GT Flow Original Image Edited Image GT Flow FlowDiffusion Pred.

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

FlowGen - Grid Size 100 FlowGen - Grid Size 900 FlowGen - Grid Size 2500

- Fig. 14. FlowGen predictions for SINTEL dataset (left) with varying grid size. FlowDiffusion predictions for FlyingChairs dataset (right). Samples from SINTEL dataset ○c [Butler et al. 2012], and FlyingChairs dataset ○c [Dosovitskiy et al. 2015].

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-or. 2022. Prompt-to-Prompt Image Editing with Cross-Attention Control. In International Conference on Learning Representations (ICLR).

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. In Conference on Neural Information Processing Systems (NeurIPS). Jonathan Ho and Tim Salimans. 2021. Classifier-Free Diffusion Guidance. In Conference on Neural Information Processing Systems (NeurIPS) Workshop.

Xingzhong Hou, Boxiao Liu, Yi Zhang, Jihao Liu, Yu Liu, and Haihang You. 2024. EasyDrag: Efficient Point-based Manipulation on Diffusion Models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8404–8413.

Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2021. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations (ICLR).

Zhaoyang Huang, Xiaoyu Shi, Chao Zhang, Qiang Wang, Ka Chun Cheung, Hongwei Qin, Jifeng Dai, and Hongsheng Li. 2022. Flowformer: A transformer architecture for optical flow. In European Conference on Computer Vision (ECCV).

Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A. Efros. 2017. Image-To-Image Translation With Conditional Adversarial Networks. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Minguk Kang, Richard Zhang, Connelly Barnes, Sylvain Paris, Suha Kwak, Jaesik Park, Eli Shechtman, Jun-Yan Zhu, and Taesung Park. 2024. Distilling Diffusion Models into Conditional GANs. In European Conference on Computer Vision (ECCV).

Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. 2020. Analyzing and Improving the Image Quality of StyleGAN. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Diederik Kingma and Jimmy Ba. 2015. Adam: A Method for Stochastic Optimization. In International Conference on Learning Representations (ICLR).

Ruining Li, Chuanxia Zheng, Christian Rupprecht, and Andrea Vedaldi. 2024. DragAPart: Learning a Part-Level Motion Prior for Articulated Objects. arXiv preprint arXiv:2403.15382 (2024).

Pengyang Ling, Lin Chen, Pan Zhang, Huaian Chen, and Yi Jin. 2023. Freedrag: Point tracking is not you need for interactive point-based image editing. arXiv preprint arXiv:2307.04684 (2023).

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2024a. Visual instruction tuning. In Conference on Neural Information Processing Systems (NeurIPS).

Haofeng Liu, Chenshu Xu, Yifei Yang, Lihua Zeng, and Shengfeng He. 2024b. Drag Your Noise: Interactive Point-based Editing via Diffusion Semantic Propagation. arXiv preprint arXiv:2404.01050 (2024).

Xingchao Liu, Chengyue Gong, et al. 2022. Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow. In International Conference on Learning Representations (ICLR).

Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017).

- Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. 2022a. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. In Conference on Neural Information Processing Systems (NeurIPS).
- Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. 2022b. Dpm-solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095 (2022).

Grace Luo, Trevor Darrell, Oliver Wang, Dan B Goldman, and Aleksander Holynski.

2024. Readout Guidance: Learning Control from Diffusion Features. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, Patrick von Platen, Apolinário Passos, Longbo Huang, Jian Li, and Hang Zhao. 2023. Lcm-lora: A universal stable-diffusion acceleration module. arXiv preprint arXiv:2311.05556 (2023).

Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. 2021. SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations. In International Conference on Learning Representations (ICLR).

Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. 2023. Null-text inversion for editing real images using guided diffusion models. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. 2023. DragonDiffusion: Enabling Drag-style Manipulation on Diffusion Models. In International Conference on Learning Representations (ICLR).

Shen Nie, Hanzhong Allan Guo, Cheng Lu, Yuhao Zhou, Chenyu Zheng, and Chongxuan Li. 2023. The Blessing of Randomness: SDE Beats ODE in General Diffusion-based Image Editing. In International Conference on Learning Representations (ICLR).

Xingang Pan, Ayush Tewari, Thomas Leimkühler, Lingjie Liu, Abhimitra Meka, and Christian Theobalt. 2023. Drag your gan: Interactive point-based manipulation on the generative image manifold. In ACM SIGGRAPH 2023 Conference Proceedings. 1–11.

Karran Pandey, Paul Guerrero, Matheus Gadelha, Yannick Hold-Geoffroy, Karan Singh, and Niloy J Mitra. 2024. Diffusion Handles Enabling 3D Edits for Diffusion Models by Lifting Activations to 3D. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7695–7704.

Gaurav Parmar, Taesung Park, Srinivasa Narasimhan, and Jun-Yan Zhu. 2024. One-Step Image Translation with Text-to-Image Models. arXiv e-prints (2024), arXiv–2403.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML).

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. 2022. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125 1, 2 (2022), 3.

Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi. 2016. You only look once: Unified, real-time object detection. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Tianhe Ren, Shilong Liu, Ailing Zeng, Jing Lin, Kunchang Li, He Cao, Jiayu Chen, Xinyu Huang, Yukang Chen, Feng Yan, et al. 2024. Grounded sam: Assembling open-world models for diverse visual tasks. arXiv pcreprint arXiv:2401.14159 (2024).

Daniel Roich, Ron Mokady, Amit H Bermano, and Daniel Cohen-Or. 2022. Pivotal tuning for latent-based editing of real images. ACM Transactions on Graphics (TOG) 42, 1 (2022), 1–13.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent diffusion models. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. 2023. Adversarial Diffusion Distillation. arXiv e-prints (2023), arXiv–2311.

Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. 2022. Laion-5b: An open large-scale dataset for training next generation image-text models. In Conference on Neural Information Processing Systems (NeurIPS).

Yujun Shi, Chuhui Xue, Jiachun Pan, Wenqing Zhang, Vincent YF Tan, and Song Bai.

2023. DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing. arXiv preprint arXiv:2306.14435 (2023).

Jiaming Song, Chenlin Meng, and Stefano Ermon. 2020. Denoising Diffusion Implicit Models. In International Conference on Learning Representations (ICLR). Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. 2023. Consistency Models. In International Conference on Machine Learning (ICML).

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. 2021. Score-Based Generative Modeling through Stochastic Differential Equations. In International Conference on Learning Representations (ICLR). Ting-Chun Wang, Arun Mallya, and Ming-Yu Liu. 2021. One-shot free-view neural talking-head synthesis for video conferencing. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Jianhui Yu, Hao Zhu, Liming Jiang, Chen Change Loy, Weidong Cai, and Wayne Wu.

2023. Celebv-text: A large-scale facial text-video dataset. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Xiaohang Zhan, Xingang Pan, Ziwei Liu, Dahua Lin, and Chen Change Loy. 2019. Self-supervised learning via conditional motion propagation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

Zewei Zhang, Huan Liu, Jun Chen, and Xiangyu Xu. 2024. GoodDrag: Towards Good Practices for Drag Editing with Diffusion Models. arXiv preprint arXiv:2404.07206

(2024).

Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros. 2017. Unpaired ImageTo-Image Translation Using Cycle-Consistent Adversarial Networks. In IEEE International Conference on Computer Vision (ICCV).

A ON THE GENERALIZABILITY OF THE INSTANTDRAG PIPELINE

As discussed in the main paper, our base models (FlowGen and FlowDiffusion) are trained exclusively on the facial video dataset CelebV-Text [Yu et al. 2023]. To handle general scene editing, we proposed a fine-tuning approach for FlowGen and FlowDiffusion using short videos containing relevant scenes. Interestingly, we observed that our base (non-fine-tuned) model successfully generalizes to unseen (non-facial) general scenes and objects in many cases. We hypothesize that this generalization capability stems from training on relatively in-the-wild, uncurated videos, allowing our models to naturally learn some basic dynamics of real-world objects.

Fig. A4 presents editing results of non-facial images. The first five rows showcase relatively successful results, while the last three rows demonstrate less successful outcomes. We observe that many movements successfully generalize to completely unseen domains such as calligraphy, 3D objects, animals, and natural scenes. However, since our models were not explicitly trained on such diverse data, they sometimes struggle to preserve subtle details or produce accurate motions in complex images.

Typical failure cases include: 1) instances where images are not properly dragged or moved, resulting in outputs nearly identical to the input, 2) undesired movements, such as shifting the entire background or the whole object instead of just a specific part, and 3) cases where fine details are lost or undesired artifacts are produced. Our analysis reveals that issues 1) and 2) typically occur when FlowGen’s output is inaccurate. Given that FlowGen is trained solely on facial videos, it sometimes fails to distinguish between parts that should be moved and those that should remain static. This ambiguity often leads to inaccurate motions or minimal movement. Issue 3), the loss of details or production of artifacts, is primarily attributed to the capacity limitations of FlowDiffusion. As it is also trained exclusively on facial videos, it sometimes struggles to accurately preserve fine details in completely unseen domains. For extremely fine-grained images with intricate details, we suspect that this issue is also related to the capacity limitations of Stable Diffusion v1.5 [Rombach et al. 2022], which serves as the backbone for our FlowDiffusion model.

While our models show reasonable results without fine-tuning, the general scene editing results presented in the main paper are based on a fine-tuning approach using base models trained on SINTEL [Dosovitskiy et al. 2015] and FlyingChairs datasets [Butler et al. 2012]. We observe similar results when initializing the fine-tuning process with our base CelebV-trained FlowGen and FlowDiffusion models instead of those trained on SINTEL and FlyingChairs. Although both models can be fine-tuned end-to-end, we find that applying LoRA fine-tuning (rank 8) to FlowDiffusion yields comparable results with minimal storage overhead. Following the approach in [Luo et al. 2023], we add low-rank matrices to selected convolutional and attention layers. For fine-tuning scenarios, we use short videos (10-60 seconds) capturing the motion of the target object. We process these videos using the same methodology as for CelebVText, except that we employ Grounded-SAM to obtain binary masks and use grid-based sampling for the extraction of pseudo-drag instructions. The fine-tuning process for FlowGen and FlowDiffusion on a single video takes approximately 20 minutes.

### B ADDITIONAL EVALUATION SAMPLES

We additionally provide more qualitative results on our base model. Randomly selected qualitative results for TalkingHead-1KH [Wang et al. 2021] data used in Table 1 are provided in Fig.A5. As mentioned in the original paper, we guide the drag editing process using 68 dlib keypoints extracted from two video frames. Since TalkingHead1KH includes some wild samples of low resolution and unaligned faces, we observe that the editing results sometimes yield significant artifacts or complete changes of identity, occasionally resulting in images that no longer resemble faces. As we are consistently manipulating 68 points, we used the third variant of our model presented in Sec. D.3 with an image guidance scale of 1.75 and a flow guidance scale of 1.5. We also provide qualitative results on the facial images of DragBench [Shi et al. 2023] in Fig. A6, which mainly consists of editing scenarios involving fewer drag instructions on synthetic faces.

### C COMPARISONS WITH OTHER MODELS AT DIFFERENT SPEEDS

We compare our method with other models operating at various speeds. To reduce inference times for models like DragDiffusion, we can adjust the number of LoRA training steps, latent optimization steps, and DDIM inversion and sampling steps. As there can be multiple possible combinations for step reduction to match specific time constraints, we use configurations which we found to be effective based on our heuristic trials. Qualitative results are presented in Fig. A3, with corresponding settings detailed in Table A1. In Fig. A3, a check and cross indicate that further extension or reduction in time for that particular model is not necessary or feasible, respectively.

All time-dependent experiments in our paper were conducted using a single A6000 GPU. We note that employing an industrystandard A100 GPU further reduces our model’s latency by 35% and achieves an impressive editing time of 0.72 seconds.

### D ADDITIONAL DETAILS ON MODELS

This section provides additional descriptions regarding network design choices and extra details. Our trained models and codes will be open-sourced upon publication.

D.1 FlowGen

We viewed the task of generating dense optical flow from drag instructions as a translation task conditioned on an input image. To achieve this, we explored several one-step generative models. A notable consideration was the recently released Pix2Pix-Turbo [Parmar et al. 2024]. Following Pix2Pix-Turbo, we experimented with fine-tuning a pre-trained SD-Turbo [Sauer et al. 2023] by slightly modifying the input and output channels. We tested both full finetuning of SD-Turbo, including the autoencoder, and LoRA training of SD-Turbo as performed in Pix2Pix-Turbo. However, we observed that the training loss diverged under the settings we tried. This result indicates the difficulty of utilizing one-step text-to-image model priors for translation to a completely different motion domain. We believe that estimating scores for translation to a completely different domain (e.g., optical flow, normal map, etc.) in one-step generative models remains challenging and constitutes an important

Table A1. Settings used for Fig. A3. A dash (–) indicates that the specific operation is not used for that method. Times measured with an A6000 gpu.

Actual LoRA Latent (Inversion) & Time (s) steps Opt. steps Sampling steps

DragDiffusion ∼75s 75.3 80 80 50 ∼50s 52.0 40 80 20 ∼20s 23.0 20 30 10 ∼10s 10.5 0 10 4 ∼5s 6.3 0 5 2

SDE-Drag ∼50s 53.0 100 – 100 ∼20s 19.5 40 – 40 ∼10s 9.0 0 – 40 ∼5s 5.5 0 – 20

DragonDiffusion

∼10s 11.0 – – 50 ∼5s 4.67 – – 20

- ∼1s 0.98 – – 3

Ours

- ∼1s 1.10 – – 20

topic for future research. We found that training FlowGen with a Ushaped Generator and discriminator from scratch yielded acceptable results for our task.

FlowGen is expected to correctly interpret all input sparse flows and map them to dense optical flow. Since a single motion can be represented via multiple different sets of points, we sampled four different sparse flows using pseudo drag instruction as detailed in Sec. 3.2.2. We then updated the generator four times to map these different outputs to the same ground truth flow while updating the discriminator once. More details on extracting pseudo drag instructions can be found in D.3.

Since our FlowGen generator is based on a U-shaped convolutional network, it can accept images of various resolutions. For training, we primarily used 256x256 images at a batch size of 256 and the Adam optimizer [Kingma and Ba 2015] with a constant learning rate of 2e-4. Given that we predict normalized flow values, our model scales well to images of larger or smaller sizes. For the SINTEL dataset, we resized images and trained our model on 512x256 images. Since FlowGen is lightweight, with 54 million parameters for the generator and 15 million for the discriminator, we performed full precision (fp32) training for 80,000 steps using a single A6000 GPU, which took approximately three days. For training FlowGen’s generator, we employ an objective:

𝐺∗ = argmin

max

L𝑎𝑑𝑣(𝐺,𝐷) + 𝜆L𝑟𝑒𝑐(𝐺), (1)

𝐺

𝐷

where 𝜆 = 100. L𝑎𝑑𝑣(𝐺,𝐷) and L𝑟𝑒𝑐(𝐺) are as defined in the main paper (Sec. 3.1.1). The inference time of FlowGen in an actual drag editing scenario is just 0.003 seconds, which is marginal compared to FlowDiffusion.

D.2 FlowDiffusion

Given a denoising network 𝜖𝜃 (𝑧𝑡,𝑐𝐼,𝑐𝑇 ), a noisy latent 𝑧𝑡 and input conditions 𝑐𝐼,𝑐𝑇 , Instruct-Pix2pix performs classifier-free guidance on both the image and text domains using the modified score estimate as follows:

𝜖˜𝜃 (𝑧𝑡,𝑐𝐼,𝑐𝑇 ) = 𝜖𝜃 (𝑧𝑡, ∅, ∅)

#### + 𝑠𝐼 · [𝜖𝜃 (𝑧𝑡,𝑐𝐼, ∅) − 𝜖𝜃 (𝑧𝑡, ∅, ∅)]

+ 𝑠𝑇 · [𝜖𝜃 (𝑧𝑡,𝑐𝐼,𝑐𝑇 ) − 𝜖𝜃 (𝑧𝑡,𝑐𝐼, ∅)]. (2)

Inspired by the effectiveness of this approach, we adapted the pretrained Stable Diffusion v1.5’s U-Net to accept flow conditions by introducing additional input channels, resulting in a denoising network, 𝜖𝜃 (𝑧𝑡,𝑐𝐼,𝑐𝐹,𝑐𝑇 ). We began with a straightforward approach, utilizing 1.5 for 𝑠𝐼 and 𝑠𝐹 and 3.0 for 𝑠𝑇 unless otherwise noted.

𝜖˜𝜃 (𝑧𝑡,𝑐𝐼,𝑐𝐹,𝑐𝑇 ) = 𝜖𝜃 (𝑧𝑡, ∅, ∅, ∅)

#### + 𝑠𝐼 · [𝜖𝜃 (𝑧𝑡,𝑐𝐼, ∅, ∅) − 𝜖𝜃 (𝑧𝑡, ∅, ∅, ∅)]

+ 𝑠𝐹 · [𝜖𝜃 (𝑧𝑡,𝑐𝐼,𝑐𝐹, ∅) − 𝜖𝜃 (𝑧𝑡,𝑐𝐼, ∅, ∅)]

+ 𝑠𝑇 · [𝜖𝜃 (𝑧𝑡,𝑐𝐼,𝑐𝐹,𝑐𝑇 ) − 𝜖𝜃 (𝑧𝑡,𝑐𝐼,𝑐𝐹, ∅)]. (3)

Since changes in the training pairs were less correlated with the text condition, we observed that the trained score estimator’s predictions remained consistent regardless of the text input. Motivated by this observation, we tested two additional variants by explicitly disabling the text condition, consistently setting the input to a null embedding.

𝜖˜𝜃 (𝑧𝑡,𝑐𝐼,𝑐𝐹,𝑐𝑇 = ∅) = 𝜖𝜃 (𝑧𝑡,𝑐𝐼,𝑐𝐹, ∅)

+ 𝑠𝐹 · [𝜖𝜃 (𝑧𝑡,𝑐𝐼,𝑐𝐹, ∅) − 𝜖𝜃 (𝑧𝑡,𝑐𝐼, ∅, ∅)].

(4)

𝜖˜𝜃 (𝑧𝑡,𝑐𝐼,𝑐𝐹,𝑐𝑇 = ∅) = 𝜖𝜃 (𝑧𝑡, ∅, ∅, ∅)

#### + 𝑠𝐼 · [𝜖𝜃 (𝑧𝑡,𝑐𝐼, ∅, ∅) − 𝜖𝜃 (𝑧𝑡, ∅, ∅, ∅)]

+ 𝑠𝐹 · [𝜖𝜃 (𝑧𝑡,𝑐𝐼,𝑐𝐹, ∅) − 𝜖𝜃 (𝑧𝑡,𝑐𝐼, ∅, ∅)].

(5)

For Eq.4, which does not perform classifier-free guidance on the input image, we did not drop any image during training. However, the inability to enforce explicit image guidance resulted in identity and color tone changes. Finally, we arrived at Eq.5, where we perform classifier-free guidance only on the image and flow. As shown in Fig. 10, this results in a controllable setting suitable for drag-based editing, where users can adjust the image guidance scale and flow guidance scale to find a balance between the amount of motion and identity preservation.

We also attempted to integrate perceptual losses such as LPIPS into the FlowDiffusion training to preserve identity better. However, calculating LPIPS in pixel space requires forward and backward passes through the VAE’s decoder, introducing extra overhead in training time and memory usage. Although models converged slightly faster in terms of steps, we found that models without LPIPS were much more efficient in computing and time. We believe that adopting recent efficient implementations of LPIPS in latent space could be beneficial in the future [Kang et al. 2024].

We initiated FlowDiffusion training from a pre-trained Stable Diffusion 1.5 model. As mentioned in the main text, we dropped

Reconstruction Loss – L1 (FlowGen)

[Figure 171]

###### CelebV-100 CelebV-1k CelebV-10k CelebV-100k CelebV-1M CelebV-8M

Fig. A1. Reconstruction loss for FlowGen using different dataset sizes.

images 5% of the time and flow 10% of the time during training, ensuring that images were never dropped when the flow was present. For training, we used 512x512 images at a batch size of 512 and the AdamW optimizer [Loshchilov and Hutter 2017] with a constant learning rate of 1e-4. The resulting FlowDiffusion model has roughly the same number of parameters as Stable Diffusion 1.5 (860M). We performed mixed precision training for 50,000 steps using 8 A6000 GPUs, which took approximately five days. We used fp16 for inference, and it takes about 1.1 seconds to sample 20 steps with the DPM++ sampler.

D.3 Details on extracting pseudo drag instructions

Here, we provide additional details on the random sparse flow sampling algorithm previously mentioned in the main text. When a mask is not available (e.g., in general scenes such as FlyingChairs), we use grid-based sampling instead of stochastic random sampling of target points in the mask regions.

We first initialize the sparse flow 𝑓𝑠 ∈ R2×ℎ×𝑤 with random values sampled from 𝑈 (0, 1). We also explore a variant where the sparse flow is initialized with random values proportional to the absolute magnitude of movement by multiplying the actual dense optical flow. The intuition here is to sample more from areas with significant movement, though this could lead to saturation of sampled points in specific locations. This approach is sometimes useful when a mask is not provided.

After initialization, we assign −∞ values to masked regions while adding a value 𝑣𝑔 to grid locations and 𝑣𝑠 to specific points, such as facial keypoints. Finally, we select the top-𝑘 points in the sparse flow and zero out the remaining parts, where 𝑘 is randomly chosen within predefined ranges. For training, we introduce a max_points parameter that determines the range of 𝑘.

For the face model, we provide four variants:

- (1) A model trained with a grid size of 100 and max_points of 10, always including the nose coordinate as a specific point. (𝑣𝑔 = 0.4, 𝑣𝑠 = 1.0)
- (2) A model trained with a grid size of 100 and max_points of 50, providing 23 key locations on the human face. (𝑣𝑔 = 0.4, 𝑣𝑠 = 0.7)
- (3) A model trained with a grid size of 100 without any specific points. (𝑣𝑔 = 0.4, 𝑣𝑠 = 0.0)

Reconstruction Loss – L2 (FlowDiffusion)

[Figure 172]

###### CelebV-100 CelebV-1k CelebV-10k CelebV-100k CelebV-1M CelebV-8M

Fig. A2. Reconstruction loss for FlowDiffusion using different dataset sizes.

(4) A model trained with grid size 900 without any specific points. (𝑣𝑔 = 0.4, 𝑣𝑠 = 0.0)

As shown in Sec. 3.2.2 and Fig. 5, we found the second variant (the stochastic model with 23 dlib keypoints) to be the most robust and performant overall. Unless otherwise specified, this variant served as our primary and base model for the figures presented. However, each variant excels in different scenarios. The first model (with nose coordinate) demonstrates particular strength in tasks involving single-point movements. In contrast, the third and fourth models (without specific points) excel in fine-grained editing tasks that require numerous control points or extremely precise and localized movements.

As previously emphasized, the sampling of appropriate points for FlowGen is crucial to its performance. While our stochastic sampling strategy based on grid and pre-defined specific points (facial keypoints) has proven effective, more sophisticated methods may further enhance the process. For instance, we can utilize the watershed strategy from [Zhan et al. 2019] to efficiently sample points along the edges of moving objects. This approach involves detecting motion edges with a Sobel filter, generating a topological-distance watershed map, and refining the key points through non-maximum suppression. Exploring these advanced sampling techniques remains a promising direction for future work.

E DATASETS

Lastly, we provide details on curating the dataset. The most crucial part of dataset pre-processing is accurately masking and estimating the flow of the target object of interest. As long as we can accurately detect and mask the target object, new appearances of other objects are acceptable, as we enforce background consistency as mentioned in Sec. 3.2.3. However, maintaining consistent masks is challenging since we use YOLO to make human masks, which detect every part of a human. For instance, if a hand appears suddenly in the second frame, YOLO would detect and mask the hand in that frame but not in the first frame, leading to inconsistent mask data between the two frames. This issue is relatively minor in general scenes, as we are less sensitive to minor occlusions and subtle changes. However, the sudden appearance of large hands occluding or partially obscuring human faces can significantly alter the perception of these faces. To address this, we first filter out low-quality (low bit rate) videos and

Input Image Drag Instr. DragDiffusion (75s) SDE-Drag (53s) DragonDiffusion (11s) Ours (1.1s)

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

~ 75s ~50s ~20s ~10s ~5s ~1s

DragDiffusion

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

SDE-Drag

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

DragonDiffusion

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

Ours

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

Fig. A3. Comparisons with other models operating at different speeds. The specific conditions under which these images were generated are detailed in Table A1. A check and cross indicate that further extension or reduction in time is not necessary or feasible, respectively. (Inputs: ○c Louvre Museum4)

sample pairs. Among the sampled pairs, we use an additional YOLObased hand detector and disregard samples where the number of hands or people changes. Using these curation methods, we obtain a total of 8 million pairs. To reduce storage overhead, we quantize optical flow into uint8 and de-quantize them on the fly during training, as saving optical flow in full fp32 precision consumes too much storage. In the case of general scenes where YOLO could not mask the object, we used GroundedSAM [Ren et al. 2024].

subsets: CelebV-Text-100, CelebV-Text-1k, CelebV-Text-10k, CelebVText-100k, CelebV-Text-1M, and the original CelebV-Text-8M. We train FlowGen and FlowDiffusion for a short duration (30k and 15k steps, respectively) and empirically validate that using more than 100k samples for both FlowGen and FlowDiffusion is crucial to avoid overfitting in the case of human faces. Results are shown in Fig.A1 and Fig.A2. Please note that although we use L2 reconstruction loss for our final FlowGen model, the ablations above and experiment in Fig. 7 were performed using L1 reconstruction loss. While both L1 and L2 losses exhibit a similar trend, we chose L2 for our final model as it produced slightly more continuous dense flows.

We ablate the number of pairs required to train FlowGen and FlowDiffusion. Using CelebV-Text’s 8 million pairs, we create six

4○c 2011 GrandPalaisRmn (musée du Louvre) / Michel Urtado. Mona Lisa. Louvre Collections: https://collections.louvre.fr/ark:/53355/cl010062370

Input Image Drag Instr. Result Input Image Drag Instr. Result

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

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

Fig. A4. Qualitative results on editing general scenes and objects using our base model without fine-tuning.

Input Image (Initial Frame)

Input Image + Drag Instr.

GT Result Image (Subsequent Frame)

DragDiffusion

DragonDiffusion

SDE-Drag Readout Guidance Ours

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

Fig. A5. Additional qualitative results. (Inputs: TalkingHead-1KH ○c [Wang et al. 2021])

Input + Drag Instr. DragDiffusion DragonDiffusion SDE-Drag Readout Guidance Ours

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

Fig. A6. Additional qualitative results. (Inputs: facial images of DragBench ○c [Shi et al. 2023])

