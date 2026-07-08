###### TRAILBLAZER: TRAJECTORY CONTROL FOR DIFFUSION-BASED VIDEO GENERATION

A PREPRINT

##### arXiv:2401.00896v2[cs.CV]8Apr2024

Wan-Duo Kurt Ma Victoria University of Wellington mawand@ecs.vuw.ac.nz

J. P. Lewis NVIDIA Research jpl@nvidia.com

W. Bastiaan Kleijn Victoria University of Wellington bastiaan.kleijn@vuw.ac.nz

|| |
|---|
<br><br>[Figure 1]|
|---|

|[Figure 2]<br><br>[Figure 3]|
|---|

|[Figure 4]<br><br>[Figure 5]|
|---|

|[Figure 6]<br><br>[Figure 7]|
|---|

|[Figure 8]<br><br>[Figure 9]|
|---|

|[Figure 10]<br><br>|
|---|

|[Figure 11]<br><br>[Figure 12]|
|---|

|[Figure 13]<br><br>[Figure 14]|
|---|

|[Figure 15]<br><br>[Figure 16]|
|---|

|April<br><br>[Figure 17]<br><br>[Figure 18]|
|---|

10, 2024

Static bbox An astronaut walking on the moon

Dynamic bbox A tiger walking on the road

|[Figure 19]<br><br>[Figure 20]|
|---|

|[Figure 21]<br><br>[Figure 22]|
|---|

|[Figure 23]<br><br>[Figure 24]|
|---|

|[Figure 25]<br><br>[Figure 26]|
|---|

|| | | |
|---|---|---|
<br><br>[Figure 27]|
|---|

|| |
|---|
<br><br>[Figure 28]|
|---|

|[Figure 29]<br><br>[Figure 30]|
|---|

|[Figure 31]<br><br>[Figure 32]|
|---|

|[Figure 33]<br><br>[Figure 34]|
|---|

|[Figure 35]<br><br>[Figure 36]|
|---|

Key-framing A cat [ sitting→running ] on the road

Morphing A [cat→dog] walking on the grass field

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

| |
|---|
|[Figure 41]|
| |

| |
|---|
|[Figure 42]|
| |

[Figure 43]

|[Figure 44]<br><br>[Figure 45]|
|---|

|[Figure 46]<br><br>[Figure 47]|
|---|

|[Figure 48]<br><br>[Figure 49]|
|---|

|[Figure 50]<br><br>[Figure 51]|
|---|

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

{ {

Multiple A dog [ watching→playing with ]

Compositing A cat and dog playing

the red balls on the road

[on the moon | in the city | in the garden]

Figure 1: TrailBlazer extends a pre-trained video diffusion model to introduce trajectory control over one or multiple subjects. Its primary contribution lies in the ability to animate the synthesized subject using a bounding box (bbox), whether it remains static (Top-left) or dynamic in terms of location and bbox size (Top-right), morphing for subject interpolation (Middle-left), and varied movement speed (Middle-right). The moving subjects fit naturally within an environment specified by the overall prompt (Bottom-right). Additionally, the speed of the subjects can be controlled through keyframing (Bottom-left).

###### ABSTRACT

Within recent approaches to text-to-video (T2V) generation, achieving controllability in the synthesized video is often a challenge. Typically, this issue is addressed by providing low-level per-frame guidance in the form of edge maps, depth maps, or an existing video to be altered. However, the process of obtaining such guidance can be labor-intensive. This paper focuses on enhancing controllability in video synthesis by employing straightforward bounding boxes to guide the subject in various ways, all without the need for neural network training, finetuning, optimization at inference time, or the use of pre-existing videos. Our algorithm, TrailBlazer, is constructed upon a pre-trained (T2V) model, and easy to implement. 1 The subject is directed by a bounding box through the proposed spatial and temporal attention map editing. Moreover, we introduce the concept of keyframing, allowing the subject trajectory, morphing, and overall appearance to be guided by both a moving bounding box and corresponding prompts, without the need to provide a detailed mask. The method is efficient, with negligible additional computation relative to the underlying pre-trained model. Despite the simplicity of the bounding box guidance, the resulting motion is surprisingly natural, with emergent effects including perspective and movement toward the virtual camera as the box size increases.

1Our project page: https://hohonu-vicml.github.io/Trailblazer.Page/

###### 1 Introduction

Advancements in generative models for text-to-image (T2I) have been dramatic Ramesh et al. (2022); Saharia et al.

- (2022a); Rombach et al. (2022); Balaji et al. (2022). Recently, text-to-video (T2V) systems have made significant strides, enabling the automatic generation of videos based on textual prompt descriptions Ho et al. (2022a,b); Wu et al.
- (2023); Esser et al. (2023). One primary challenge in video synthesis lies in the extensive memory and required training data. Methods based on the pre-trained Stable Diffusion (SD) model have been proposed to address the efficiency issues in T2V synthesis. These approaches address the problem from several perspectives including finetuning and zero-shot learning Khachatryan et al. (2023); Qi et al. (2023).

However, text prompts do not provide good control over the spatial layout and trajectories of objects in the generated video. This control is known to be required for understandable narration of a story Arijon (1976). Existing work such

- as Hu and Xu (2023) has approached this problem by providing low-level control signals, e.g., using Canny edge maps or tracked skeletons to guide the objects in the video using ControlNet Zhang and Agrawala (2023). These methods achieve good controllability, but they can require considerable effort to produce the control signal. For example, capturing the desired motion of an animal (e.g., a tiger) or an expensive object (e.g., a jet plane) would be quite difficult, while sketching the desired movement on a frame-by-frame basis would be tedious.

To address the needs of casual users, we introduce a high-level interface for the control of object trajectories in synthesized videos summarized in Fig. 1. Users simply provide bounding boxes (bboxes) specifying the desired position of an object at several times (keyframes) in the video, together with the text prompt(s) describing the object at the corresponding times. The provided bboxes are interpolated between the keyframes, resulting in smooth motion and size changes of the object. For instance, the cat sitting in the early half of the video in the red bbox, and then moving with the cyan bbox achieved through keyframing in the middle right of Fig. 1.

If more than one different text prompt is provided, these prompts are also interpolated, resulting in a “morphing” effect such as the cat that transforms into a dog in Fig. 1. To achieve this guidance we take inspiration from the observation Liew et al. (2022) that object position is established early in the denoising diffusion process, and we leverage the clear spatial interpretation of spatial and temporal attention maps as illustrated in Fig. 2. Our resulting strategy involves editing both spatial and temporal attention maps for a specific object during the initial denoising diffusion steps to concentrate activation at the desired object location. Our inference-time editing approach achieves this without disrupting the learned text-image association in the pre-trained model, and requires minimal code modifications.

Our method, TrailBlazer, builds on previous works. We use the pre-trained ZeroScope model cerspense (2023), which is a fine-tuned version of Wang et al. (2023), as our underlying model. A body of previous and concurrent works have addressed guiding object position in image generation models, including Zhao et al. (2020); Sun and Wu (2022); Yang et al. (2022b); Balaji et al. (2022); Ma et al. (2023); Xie et al. (2023); Li et al. (2023); Bar-Tal et al. (2023). TrailBlazer most closely resembles the cross-attention injection used in Ma et al. (2023), and we adopt some notation from that paper. However, our work addresses a different problem, that of controlling position and trajectories in videos, which requires a different approach to control temporal cross-frame attention. Our work also does not require the inference-time optimization algorithm used in Ma et al. (2023).

Our contributions are three-fold:

- • Novelty. We introduce a novel approach employing high-level bounding boxes to guide the subject in diffusionbased video synthesis. This approach is suitable for casual users, as it avoids the need to record or draw a frame-by-frame positioning control signal. In contrast, the low-level guidance signals such as detailed masks, edge maps, used by some other approaches have two disadvantages: it is difficult for non-artists to draw these shapes, and processing existing videos to obtain these signals limits the available motion to copies of existing sources.
- • Position, size, and prompt trajectory control. Our approach enables users to position the subject by keyframing its bounding box. The size of the bbox can be similarly controlled, thereby producing perspective effects (Figs. 1,6). Finally, users can also keyframe the text prompt to influence the behavior and identity of the subject in the synthesized video (Figs. 1).
- • Simplicity. Our method operates by directly editing the spatial and temporal attention in the pre-trained denoising UNet. It requires no training or optimization, and the core algorithm can be implemented in less than 200 lines of code.

###### 2 Related Work

###### 2.1 Text-to-Image (T2I)

Denoising diffusion models construct a stochastic Sohl-Dickstein et al. (2015); Song and Ermon (2019); Ho et al. (2020) or deterministic Song et al. (2021) mapping between the data space and a corresponding-dimension multivariate Gaussian. Signals are synthesized by sampling from a normal distribution and performing a sequence of denoising steps. A number of works Nichol et al. (2022); Nichol and Dhariwal (2021); Ramesh et al. (2022); Saharia et al. (2022b) have performed T2I synthesis using images conditioned on the text embedding from a model such as CLIP Radford

- et al. (2021). Performance is significantly improved in the Latent Diffusion Model Rombach et al. (2022) (LDM) by performing the diffusion computation in the latent space of a carefully trained variational autoencoder. LDM was trained with a large scale dataset, resulting in the widely adopted Stable Diffusion (SD) system. We omit the basic diffusion derivation as tutorials are available, e.g., Weng (2021).

Despite the success of image generation using SD, it is widely acknowledged that SD lacks controllability in synthesis. SD faces challenges in synthesizing multiple objects, often resulting in missing objects or incorrect assignment of prompt attributes to different objects. Recently, ControlNet Zhang and Agrawala (2023) and T2I-Adapter Mou et al. (2023) introduced additional fine-tuning layers to train the model with various forms of image conditioning such as edge maps, or rigging skeletons.

The methods of Zhao et al. (2020); Sun and Wu (2022); Yang et al. (2022b); Ma et al. (2023); Xie et al. (2023); Bar-Tal et al. (2023) have addressed the layout-to-image (L2I) issue using few-shot learning. Directed Diffusion Ma et al. (2023), BoxDiff Xie et al. (2023), and MultiDiffusion Bar-Tal et al. (2023) use coarse bboxes to control subject position, achieving good results by manipulating the spatial latent and text embeddings cross attention map Hertz et al. (2022).

###### 2.2 Text-to-Video (T2V)

Text-to-video (T2V) synthesis is generally more difficult than T2I due to the difficulty of ensuring temporal consistency and the requirement for a large paired text and video dataset. Ho et al. (2022b); Harvey et al. (2022); Höppe et al.

- (2022); Voleti et al. (2022); Yang et al. (2022a); Ge et al. (2023) show methods that build on top of image diffusion models. Some works Blattmann et al. (2023); Luo et al. (2023) also introduce 3D convolutional layers in the denoising UNet to learn temporal information. Imagen Video Ho et al. (2022a) achieves higher resolution by computing temporal and spatial super-resolution on initial low resolution videos. VideoLDM Blattmann et al. (2023) and ModelScope Luo et al. (2023) insert a temporal attention layer by reshaping the latent tensor. Text2Video-Zero Khachatryan et al. (2023), denoted as T2V-Zero, and FateZero Qi et al. (2023) investigate how the temporal coherence can be improved by cross frame attention manipulation with pre-trained T2I models. Ge et al. (2023) addresses the same problem by introducing temporal correlation in the diffusion noise. However, these pioneering studies generally lack position control in the video synthesis.

Recently several works have been proposed to solve the controllability in video synthesis problem by using pre-trained models together with low-level conditioning information such as edge or depth maps. Control-A-Video Chen et al.

- (2023) and MagicProp Yan et al. (2023) use depth maps with ControlNet to train a temporal-aware network. T2V-Zero Khachatryan et al. (2023) partially achieves controllability by initializing the latent frames conditioned on the first frame with applied linear translation. However, the control is indirect and requires two steps. First, the user first needs to locate the subject’s numerical location, and then adjust a translation offset. Distinct from the methods above, we use an attention injection method to guide the denoising path rather than optimization, and in general this is robust to different random seeds. The recent project Peekaboo Jain et al. (2023) is concurrent with TrailBlazer and shares similar goals. Both Peekaboo and TrailBlazer guide subjects in video by manipulating the attention, however the formulations differ in many details. Peekaboo’s use of an infinite negative attention injection in the background regions appears to often result in backgrounds with missing detail. In Sec. 4, we will provide both quantitative and visual evidence to demonstrate the better controllability and quality of our results. Other very recent preprints that address the layout-to-video problem in differing ways include Lian et al. (2023); Wang et al. (2024); Yang et al. (2024). We do not compare against these concurrent works because their source was not available at the time of writing.

###### 3 Method

TrailBlazer is based on the open-source pre-trained model ZeroScope cerspense (2023). This is a fine-tuned version of ModelScope Luo et al. (2023), known for its ability to generate high-quality videos without significant temporal flickering. It is noteworthy that TrailBlazer preserves this desirable temporal coherence effect achieved in their work. TrailBlazer does not require any training, optimization, or low-level control signals (e.g., edge, depth maps with

ControlNet Zhang and Agrawala (2023)). On the contrary, all that is required from the user is the prompt and an approximate bounding box (bbox) of the subject. Bboxes and corresponding prompts can be specified at several points in the video, and these are treated as keyframes and interpolated to smoothly control both the motion and prompt content.

We use the following notation: Bold capital letters (e.g., M) denote a matrix or a tensor depending on the context, vectors are represented with bold lowercase letters (e.g., m), and scalars are denoted as lowercase letters (e.g., m). We use superscripts to denote an indexed tensor slice (e.g., M(i)). A synthesized video is composed of a number of images ordered in time. The individual images will be referred to as frames, and the collection of corresponding times is the timeline. Spatial or temporal attention will be informally referred to as correlation.

Similar to the work in Ma et al. (2023), our method draws significant inspiration from visual inspection of cross-attention maps. Consider the final cross-attention result depicted in Fig. 2, generated from the prompt “an astronaut walking on the moon”. The spatial cross attention, denoted as SA-Cross, associated with the prompt word “astronaut” is highlighted

- at the left of the second row, showcasing the overall position of the subject. Furthermore, we visualize the attention map from the temporal module in the pre-trained model. The right in the first row displays “self-frame” temporal attention maps, denoted as TA-Self, which consistently align with SA-Cross.

The right in the second row of Fig. 2 presents the visualization of cross-frame temporal attention maps, denoted as TA-Cross, illustrating the attention between the first frame and subsequent frames in the video. As the distance between frames increases, the attention becomes less correlated in the subject area and becomes more correlated in the background area. This observation aligns with the reconstructed video shown in the left of the first row, where the background remains nearly static while the astronaut’s position varies frame by frame. We will consider the temporal attention in detail in Sec. 3.3.

|[Figure 65]<br><br>[Figure 66]|
|---|

|[Figure 67]<br><br>[Figure 68]|
|---|

|[Figure 69]<br><br>[Figure 70]|
|---|

|[Figure 71]<br><br>[Figure 72]|
|---|

|[Figure 73]<br><br>[Figure 74]|
|---|

|[Figure 75]<br><br>[Figure 76]|
|---|

|[Figure 77]<br><br>[Figure 78]|
|---|

|[Figure 79]<br><br>[Figure 80]|
|---|

Recons

TA-Self

Frame 1 Frame 4 Frame 16 Frame 24

- Attn(1,1) Attn(4,4) Attn(16,16) Attn(24,24)
- Attn(1,2) Attn(1,4)

TA-Cross

|[Figure 81]<br><br>[Figure 82]|
|---|

|[Figure 83]<br><br>[Figure 84]|
|---|

|[Figure 85]<br><br>[Figure 86]|
|---|

|[Figure 87]<br><br>[Figure 88]|
|---|

|[Figure 89]<br><br>[Figure 90]|
|---|

|[Figure 91]<br><br>[Figure 92]|
|---|

|[Figure 93]<br><br>[Figure 94]|
|---|

|[Figure 95]<br><br>[Figure 96]|
|---|

SA-Cross

Frame 1 Frame 4 Frame 16 Frame 24

Attn(1,18) Attn(1,23)

- Figure 2: Basis of our method. We draw inspiration from inspection of the spatial (SA) and temporal (TA) attention maps viewed with self-frame attention (Self) and cross-frame attention (Cross). Thus, TA-Self and TA-Cross denote the self- and cross-frame attention map, respectively. SA-Cross is the spatial cross-attention map with the prompt word “astronaut”. The symbol “Attn(i,j)” denotes the temporal attention map between frame i and frame j. The Recons subfigure shows reconstructions sampled from frames 1, 4, 16, and 24, respectively. In the TA-Cross, the frame number were manually chosen to best illustrate the cross-frame attention between the astronaut and the background. Please refer to the main text for more details.

###### 3.1 Pipeline

As mentioned above, keyframing wiki (2023) is a technique that defines properties of images at particular frames (keys) in a timeline and then automatically interpolates these values to achieve a smooth transition between the keys. It is widely used in the movie animation and visual effects industries since it reduces the artist’s work while simultaneously producing temporally smooth motion that would be hard to achieve if the artist directly edited every image. Our system takes advantage of this principle, and asks the user to specify several keys, consisting of bboxes and the associated prompts, describing the subject location and appearance or behavior at the particular times. For instance, as shown in Fig. 1 (Middle-right), the video of the cat initially sitting on the left, then running to the right, is achieved simply by placing keys at three frames only. Specifically, the sitting cat in the first part of the video is obtained with two identically positioned bboxes on the left, with the keyframes at the beginning and middle of the timeline and the prompt word “sitting” associated with both. A third keyframe is placed at the end of the video, with the bbox positioned on the right together with the prompt changing to “running”. This results in the cat smoothly transitioning from sitting to running in the second part of the video.

#### Pipeline

TrailBlazer: Trajectory Control for Diffusion-Based Video Generation A PREPRINT

[Figure 97]

[Figure 98]

Prompt AttnMap CrossFrame AttnMap

Trailings AttnMap

Prompt Embedding

UNet Denoiser

Frame i Embedding

Frame j Embedding

Frame Embedding

Prompt Words

| |
|---|

[Figure 99]

{

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

×

[Figure 104]

[Figure 105]

×

|𝒫|

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Q(mi)

###### K(mj) Gaussian A(mi,j)

[Figure 113]

[Figure 114]

###### {

[Figure 115]

[Figure 116]

Trailings

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Qs

[Figure 121]

[Figure 122]

[Figure 123]

|𝒯|

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

A(si)

Ks

[Figure 129]

[Figure 130]

[Figure 131]

Weight Map

spatial cross attention temporal attention

- Figure 3: Pipeline Overview. Our pipeline highlights the central components of spatial cross-attention editing (left, in the blanched almond-colored section) and temporal cross-frame attention editing (right, in the blue section). This operation is exclusively applied during the denoising process in the early stage. The objective is to alter the attention

map (e.g., As,Am) using a Gaussian weighting within a user-specified bbox. This example uses one prompt word AttnMap and two trailing AttnMaps for guidance as highlighted in red.

We use the pre-trained ZeroScope model cerspense (2023) in all our experiments with no neural network training, finetuning, or optimization at inference time. Our pipeline is shown in Fig. 3. The spatial cross attention and the temporal attention is discussed in detail in the Sec. 3.2 and Sec. 3.3, respectively. All spatial and temporal editing is performed in the early steps t ∈ {T,...,T − NS}, and t ∈ {T,...,T − NM} of the backward denoising process, where T is the total number of denoising time steps, and NS, and NM are hyperparameters specifying the number of steps of spatial and temporal attention editing. The parameter settings are detailed in our supplementary material.

In the subsequent sections we describe how our algorithm is implemented by modifying the spatial and temporal attention in a pre-trained diffusion model. Please refer to Rombach et al. (2022); Song et al. (2021); Ho et al. (2020); Weng (2021) for background on overall diffusion model architectures.

Our system processes a set of keyframes, encompassing associated bbox regions Rf and prompts Pf at frame f, where f denotes the frame index within the range f ∈ {1,...,NF}. The users are required to specify a minimum of two keyframes: one at the start and one at the end of the video sequence. The information in these keyframes is linearly interpolated, such as the bbox Bf and the prompt text embedding y(Pf) through the text encoder y(·). To enhance readability, we omit the subscript f and the linearly blended video sequence between the keyframes when discussing the core method.

A region R is characterized by a set of parameters R = {B,I,T }: a set of bbox positions (e.g., B), the indices of the subject we would like to constrain (e.g., I), and the indices of the trailing maps (e.g., T ) to enhance the controllability. The subject indices I ⊂ {i|i ∈ N,1 ≤ i ≤ |P|}, are 1-indexed with the associated word in the prompt. For example, I = {1,2} is associated with “a”, “cat” in the prompt “a cat sitting on the car”.

The trailing attention maps indices T ⊂ {i|i ∈ N,|P| < i ≤ NP} is the set of indices corresponding to the crossattention maps generated without a prompt word association, where NP denotes the maximum prompt length that a tokenizer model can take, which is NP = 77 when CLIP is used Radford et al. (2021). The trailing attention maps serve as a means of controlling the spatial location of the synthesized subject and its attributes. A larger trailing indices set |T | provides greater controllability but comes with the risk of failed reconstruction Ma et al. (2023).

A bbox B = (x,y)|bleft × w ≤ x ≤ bright × w, btop × h ≤ y ≤ bbottom × h , is a set of all pixel coordinates inside the bbox of resolution w × h. In our implementation, B is produced by a tuple of the four scalars representing the boundary

of the bbox b = (bleft,btop,bright,bbottom), where bleft,btop,bright,bbottom ∈ [0,1] specify the bbox relative to the synthesis resolution. The height h and width w, are defined by the resolution of the UNet intermediate representation Rombach

- et al. (2022).

###### 3.2 Spatial Cross Attention Guidance

The spatial cross attention modules are implemented in the denoising UNet module of Rombach et al. (2022). This module finds the cross attention between the query representation Qs ∈ RN

F×dh×d obtained from the SD latent zt, and the representations Ks,Vs ∈ RN

F×|W|×d of the |W| prompt words from the text model, where d is the feature dimension of the keys and queries. Usually |W| ≡ 77 when the text embedding model is CLIP Radford et al. (2021).

√

F×dh×|W|,2 where dh ≡ w × h, defined by the spatial resolution height and width at the specific layer. For simplicity we omit the batch size and the number of attention heads Vaswani et al. (2017) in our definition. As illustrated in the blanched almond-colored section in Fig. 3, we guide the denoising path by editing the spatial cross attention (e.g., Ma et al. (2023)) for the attention maps A(si) ∈ RN

The cross attention map Hertz et al. (2022) is then defined as As = Softmax(QsKTs /

d) ∈ RN

F×dh associated with a particular prompt word and trailing indices i ∈ I ∪ T . Given B, our spatial attention editing is defined by

Ss(x,y) =

cs g(x,y), (x,y) ∈ B 0, otherwise,

Ws(x,y) =

cw, (x,y) ∈ B′ 1, otherwise,

(1)

where x,y are are the spatial location indices of the attention map and B′ is the complement of B. Ss(B) uses a function g(·,·) that “injects” attention inside B, as illustrated in the gray box in Fig. 3. The parameters cw ≤ 1, cs > 0 attenuate the attention outside of B and strengthen it inside. We define g(·,·) as a Gaussian window of size σx = bw/2,σy = bh/2, where bw = ceil((bright − bleft) × w),bh = ceil((btop − bbottom) × h) are the width and the height of B. In contrast, Ws(·) attenuates the attention outside B. The bbox B is extended across the entire video

sequence through linear interpolation of the keyframes. For example, Bf = (1 − a) × Bb + a × Be, where a = Nf

,

F

and Bb, Be denotes the bbox for the beginning and the end of keyframe. Given the set of indices of subject prompt words I and trailing maps T , each cross-activation component at location (x,y) in As is modified as follows:

A(si)(x,y) := A(si)(x,y) ⊙ Ws(x,y) + Ss(x,y), ∀i ∈ I ∪ T , (2)

where ⊙ denotes the Hadamard (element-wise) product that scales the x,y element of the cross-attention map As by the corresponding weight in Ws(·). The result is that the attention in the cross-attention map for the particular prompt word as well as the trailing maps, is stronger in the user-specified bbox region.

###### 3.3 Temporal Cross-Frame Attention Guidance

To capture the temporal correlation in the video clip during training, a prevalent approach involves reshaping the latent tensor. This involves shifting the spatial information to the first dimension, a technique employed in VideoLDM Blattmann et al. (2023). The reshaping is done before passing the hidden activation into the temporal layers, allowing the model to learn about the “correlation” of spatial components through the convolutional layers. As shown in blue

√

section in Fig. 3, the temporal attention map is obtained by Am = Softmax(QmKTm/

h×NF×NF , where dh is the spatial dimensions of this tensor, Qm ∈ Rd

d) ∈ Rd

h×NF×d, and Km ∈ Rd

h×NF×d.

What is different from the spatial counterpart is that now Am learns about the relation between the correlated components across all frames. For instance, A(mx,y,i,j) denotes the correlation at location (x,y) between frame i and frame j. We denote such tensors as A(mi,j)(x,y) to keep the notation consistent. As seen in our visual investigation (Fig. 2, Right), the background attention is higher when the cross frame attention compares the frames that are temporally far from each other, and the foreground attention is higher when the frames are temporally closer in the video sequence.

To achieve this pattern of activations under user control we design an approach similar to Eq. 2 but considering the normalized video temporal distance d = |iN−j|

,i,j ∈ {1,...,NF}, the temporal injection function is defined as,

F

(1 − d) g(x,y) − d g(x,y), (x,y) ∈ B, 0, otherwise.

Sm(x,y) =

Here the normalized video temporal distance d determines the level of the weight injection as a triangular window in time. Values d ≈ 0 increase the activation inside the bbox. In contrast, when d ≈ 1, the activation inside the box is reduced, approximating the temporal “anti-correlation” effect seen in Fig. 2. The editing by Sm(·) is performed during the initial NM steps of the denoising process.

Then, similarly to Eq. 2, the temporal cross-frame attention map editing is,

A(mi,j)(x,y) := A(mi,j)(x,y) ⊙ Wm(x,y) + Sm(x,y), (3) where Wm(·) is defined the same as Ws(·).

2Note that this is a “batch” matrix multiplication (e.g., the method torch.bmm in PyTorch Paszke et al. (2019)), that is C = AB ∈ Rb×m×n, where A ∈ Rb×m×p, and B ∈ Rb×p×n. Similarly, the transpose operation is A⊤ ∈ Rb×p×m.

z(tdog) z(tball)

[Figure 132]

[Figure 133]

|[Figure 134]<br><br>[Figure 135]|
|---|

|[Figure 136]<br><br>[Figure 137]|
|---|

|[Figure 138]<br><br>[Figure 139]|
|---|

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

Comp Prompt

A dog [ standing→ running] on the road

A red ball on the grass field A dog chasing the red ball on the grass field

SC

- Figure 4: Scene Compositing. Given the set of latents generated from our system using a single bbox denoted as z(tball)

and z(tdog) for the case of prompts related to ball and dog. Then, the scene compositor (SC) produces a synthesis of multiple subjects with the complete prompt and the single subject latents. We refer reader to our supplementary video to view the implemented speed control of the dog.

###### 3.4 Scene compositing

The problem space becomes more complicated for video synthesis with more than one moving subject. Although the parameters cs,cw in Eq. 2 are specific to a particular subject, they indirectly affect the entire scene through the global denoising. Thus, the choice of these parameters for different subjects might interact and require a parameter search in the number of subjects to find the best synthesis. If the prompt P and bbox B are in conflict then the result might be poor. For instance, a user may specify motion of B from left to right associated with the prompt word “dog”, while P is given as “a dog is sitting on the road”. In fact, the dog moves in accordance with the configured bbox, either walking or running.

Considering the reasons above, we follow work such as Ma et al. (2023); Bar-Tal et al. (2023) that combine multiple subjects, each with their own prompt, during the latent denoising. The latents z(tr) for the r-th subject are then composited into an overall image latent zt under the control of a “composed” prompt, as illustrated in Fig. 4 and formulated as,

NR

1 R

w zt(x,y) + (1−w)z(tr)(x,y), (4)

zt(x,y) :=

r=0

where ∀t ∈ {T,...,T−NC}, (x,y) ∈ Br, where w ∈ [0,1] determines the weight of linear interpolation between the specific subject latent z(tr) and the composed latent zt. It is formulated by considering the ratio of the current denoising timestep between NC and T, such that w = 1 − NC − (T − t) /NC. At the beginning of the denoising process (so at t = T), the compositing fully prioritizes the subject latent z(tr) in each local region in the associated bbox Br. As t decreases, w gradually increases, giving higher priority to composed latent zt. This process concludes when t = T − NC, resulting in w = 1, which stops using the subject latent in the remaining denoising steps.

###### 4 Experiments

Here we briefly present some experiments and quantitative evaluations of our work. Please see our supplementary materials and the project video for full experiments, including implementation details, limitations, ablations, comparisons, and finer details. The figures show an evenly spaced temporal sampling of frames from the videos.

###### 4.1 Main result

- Fig. 5 shows our main result on trajectory control of a single subject. We compare TrailBlazer to T2V-Zero Khachatryan

- et al. (2023) and Peekaboo Jain et al. (2023) using the same prompts, without conditioning guidance (e.g., edge or depth maps) to provide a fair comparison. T2V-Zero accepts motion guidance in the form of an (x,y) translation vector. We set this vector to (8,0) to produce horizontal motion. More detailed visual comparisons with Peekaboo under extreme conditions are depicted in Sec. 9 of the supplementary materials. The presented results for each method are visually selected as the best of a pool of 10 experiments conducted with different random seeds.

In Fig. 5, the results are generated from linearly interpolated bboxes starting at the left of the image and moving to the right. The results from TrailBlazer demonstrate anatomically plausible motion of the subject and a more accurate fitting

[Figure 154]

|[Figure 155]<br><br>[Figure 156]<br><br>| |
|---|
|
|---|

|[Figure 157]<br><br>[Figure 158]<br><br>| |
|---|
|
|---|

|[Figure 159]<br><br>[Figure 160]<br><br>| |
|---|
|
|---|

|[Figure 161]<br><br>[Figure 162]<br><br>| |
|---|
|
|---|

|[Figure 163]<br><br>[Figure 164]<br><br>| |
|---|
|
|---|

|[Figure 165]<br><br>[Figure 166]<br><br>| |
|---|
|
|---|

|[Figure 167]<br><br>[Figure 168]<br><br>| |
|---|
|
|---|

[Figure 169]

###### TrailBlazerPeekabooT2V-Zero

[Figure 170]

|[Figure 171]<br><br>[Figure 172]<br><br>| |
|---|
|
|---|

|[Figure 173]<br><br>[Figure 174]<br><br>| |
|---|
|
|---|

|[Figure 175]<br><br>[Figure 176]<br><br>| |
|---|
|
|---|

|[Figure 177]<br><br>[Figure 178]<br><br>| |
|---|
|
|---|

|[Figure 179]<br><br>[Figure 180]<br><br>| |
|---|
|
|---|

|[Figure 181]<br><br>[Figure 182]<br><br>| |
|---|
|
|---|

|[Figure 183]<br><br>[Figure 184]<br><br>| |
|---|
|
|---|

[Figure 185]

[Figure 186]

|[Figure 187]<br><br>[Figure 188]<br><br>| |
|---|
|
|---|

|[Figure 189]<br><br>[Figure 190]<br><br>| |
|---|
|
|---|

|[Figure 191]<br><br>[Figure 192]<br><br>| |
|---|
|
|---|

|[Figure 193]<br><br>[Figure 194]<br><br>| |
|---|
|
|---|

|[Figure 195]<br><br>[Figure 196]<br><br>| |
|---|
|
|---|

|[Figure 197]<br><br>[Figure 198]<br><br>| |
|---|
|
|---|

|[Figure 199]<br><br>[Figure 200]<br><br>| |
|---|
|
|---|

[Figure 201]

[Figure 202]

|[Figure 203]<br><br>[Figure 204]<br><br>| |
|---|
|
|---|

|[Figure 205]<br><br>[Figure 206]<br><br>| |
|---|
|
|---|

|[Figure 207]<br><br>[Figure 208]<br><br>| |
|---|
|
|---|

|[Figure 209]<br><br>[Figure 210]<br><br>| |
|---|
|
|---|

|[Figure 211]<br><br>[Figure 212]<br><br>| |
|---|
|
|---|

|[Figure 213]<br><br>[Figure 214]<br><br>| |
|---|
|
|---|

|[Figure 215]<br><br>[Figure 216]<br><br>| |
|---|
|
|---|

[Figure 217]

|[Figure 218]<br><br>[Figure 219]|
|---|

|[Figure 220]<br><br>[Figure 221]|
|---|

|[Figure 222]<br><br>[Figure 223]|
|---|

|[Figure 224]<br><br>[Figure 225]|
|---|

|[Figure 226]<br><br>[Figure 227]|
|---|

|[Figure 228]<br><br>[Figure 229]|
|---|

|[Figure 230]<br><br>[Figure 231]|
|---|

|[Figure 232]<br><br>[Figure 233]|
|---|

|[Figure 234]<br><br>[Figure 235]|
|---|

|[Figure 236]<br><br>[Figure 237]|
|---|

|[Figure 238]<br><br>[Figure 239]|
|---|

|[Figure 240]<br><br>[Figure 241]|
|---|

|[Figure 242]<br><br>[Figure 243]|
|---|

|[Figure 244]<br><br>[Figure 245]|
|---|

|[Figure 246]<br><br>[Figure 247]|
|---|

|[Figure 248]<br><br>[Figure 249]|
|---|

- Figure 5: Main result: Rigid bbox moving from left to right. TrailBlazer and Peekaboo use identical bboxes, while T2V-Zero uses the corresponding motion vector instead. The same prompt is used across each method. The four prompts used (clockwise from top left): An astronaut walking on the moon; A macro video of a bee pollinating a flower; A clown fish swimming in a coral reef; A cat walking on the grass field. The bold text represents the directed object.

of the subject within the bbox. Further, all subjects (e.g., cat, bee, astronaut, and clown fish) face in the direction that they move. However, this is not a common occurrence in T2V-Zero, as they directly apply the editing operation on the diffusion latents. This approach merely translates the subject without re-orienting it. Although the synthesized subject’s motion generally follows the bbox in Peekaboo, it does not fit the bbox well. Occasionally, artifacts may emerge, such as a rectangular object following the astronaut. Moreover, our synthesized background exhibits better visual quality. In the competing methods, the background often appears plain, blurry, or lacks detail behind the subject (e.g., the area behind the subject path in Peekaboo).

- Fig. 6 illustrates the speed control, and dynamically changing the bbox size producing an effect of the subject moving toward or away from the virtual camera. In the top two rows of comparisons, the bbox setup is between the top-left corner and the bottom-right corner. The dynamically changing bbox size is annotated with a green box as illustrated on the left. Note that the generated subjects share the desirable characteristic that the subject naturally faces toward the virtual camera when the bbox transitions from small to large as seen in the top sequence and vice versa in the second sequence from the top. The results also show a desirable perspective effect. Increasing or reducing the bbox size over time causes the synthesized object to produce the motion of “coming to” and “going away from” the camera as shown in the tiger example. We believe these effects arise naturally as a result of manipulating a model that was trained on video sequences rather than images. Peekaboo’s tiger fails to produce these perspective effects when guided with identical bboxes.

Furthermore, our method adeptly manages fast motion, outperforming Peekaboo in this regard. This is evident in the third row of both subfigures of Fig. 6, which illustrate a cat rapidly running from one side to the other multiple times in the video clip, where Nf = 24. More precisely, the bbox is initially positioned at the left (1st keyframe), after which it is moved to the right (2nd keyframe), then left (3rd keyframe), right (4th keyframe), and left (5th keyframe).

Multi-subject synthesis is generally challenging, particularly when the number of objects exceeds two. We delve into this issue in the supplementary materials. In Fig. 7, we present experiments with two subjects, a cat and a dog, guided by the green bbox in the sub-figure. The synthesis of the dog and cat in isolation is depicted in the top row on the left, serving as a sanity check with annotated image frame. We also show six results combining environment prompts (e.g., “... on the moon”) after composed prompt (e.g., “A white cat and a yellow dog running...”). Each experiment demonstrates the flexibility of TrailBlazer in synthesizing subjects under varied environmental conditions. Notably, the interactions between the background and subjects appear plausible, as seen in reflections and splashes in the swimming pool case and consistent shadows across all samples. The results also show some artifacts such as extra limbs that are inherited from the underlying model.

[Figure 250]

|[Figure 251]<br><br>[Figure 252]|
|---|

|[Figure 253]<br><br>[Figure 254]|
|---|

|[Figure 255]<br><br>[Figure 256]|
|---|

|[Figure 257]<br><br>[Figure 258]|
|---|

|[Figure 259]<br><br>[Figure 260]|
|---|

|[Figure 261]<br><br>[Figure 262]|
|---|

|[Figure 263]<br><br>[Figure 264]|
|---|

|[Figure 265]<br><br>[Figure 266]|
|---|

###### TrailBlazerPeekaboo

[Figure 267]

|[Figure 268]<br><br>[Figure 269]|
|---|

|[Figure 270]<br><br>[Figure 271]|
|---|

|[Figure 272]<br><br>[Figure 273]|
|---|

|[Figure 274]<br><br>[Figure 275]|
|---|

|[Figure 276]<br><br>[Figure 277]|
|---|

|[Figure 278]<br><br>[Figure 279]|
|---|

|[Figure 280]<br><br>[Figure 281]|
|---|

|[Figure 282]<br><br>[Figure 283]|
|---|

|[Figure 284]<br><br>|
|---|
| |

|[Figure 285]<br><br>[Figure 286]|
|---|

|[Figure 287]<br><br>[Figure 288]|
|---|

|[Figure 289]<br><br>[Figure 290]|
|---|

|[Figure 291]<br><br>[Figure 292]|
|---|

|[Figure 293]<br><br>[Figure 294]|
|---|

|[Figure 295]<br><br>[Figure 296]|
|---|

|[Figure 297]<br><br>[Figure 298]|
|---|

|[Figure 299]<br><br>[Figure 300]|
|---|

[Figure 301]

|[Figure 302]<br><br>[Figure 303]|
|---|

|[Figure 304]<br><br>[Figure 305]|
|---|

|[Figure 306]<br><br>[Figure 307]|
|---|

|[Figure 308]<br><br>[Figure 309]|
|---|

|[Figure 310]<br><br>[Figure 311]|
|---|

|[Figure 312]<br><br>[Figure 313]|
|---|

|[Figure 314]<br><br>[Figure 315]|
|---|

|[Figure 316]<br><br>[Figure 317]|
|---|

[Figure 318]

|[Figure 319]<br><br>[Figure 320]|
|---|

|[Figure 321]<br><br>[Figure 322]|
|---|

|[Figure 323]<br><br>[Figure 324]|
|---|

|[Figure 325]<br><br>[Figure 326]|
|---|

|[Figure 327]<br><br>[Figure 328]|
|---|

|[Figure 329]<br><br>[Figure 330]|
|---|

|[Figure 331]<br><br>[Figure 332]|
|---|

|[Figure 333]<br><br>[Figure 334]|
|---|

|[Figure 335]<br><br>|
|---|
| |

|[Figure 336]<br><br>[Figure 337]|
|---|

|[Figure 338]<br><br>[Figure 339]|
|---|

|[Figure 340]<br><br>[Figure 341]|
|---|

|[Figure 342]<br><br>[Figure 343]|
|---|

|[Figure 344]<br><br>[Figure 345]|
|---|

|[Figure 346]<br><br>[Figure 347]|
|---|

|[Figure 348]<br><br>[Figure 349]|
|---|

|[Figure 350]<br><br>[Figure 351]|
|---|

- Figure 6: Main result: Dynamic moving bbox. (Top/Middle row): The tiger walking on the street. (Bottom row): The cat running on the grass. The first column illustrates the bbox keyframes in the squared layout, where the green bbox is guided by the almond-colored motion vector. Note that there are five keyframes in the third row of each subfigure, with the bbox located on the left at the initial keyframe. The bbox used in top and the middle row are linearly interpolated with varied sizes. The bbox used in the last row has a static size with 5 keyframes moving back-and-forth. Please refer to the main text for more detail.

|[Figure 352]<br><br>[Figure 353]|
|---|

|[Figure 354]<br><br>[Figure 355]|
|---|

|[Figure 356]<br><br>[Figure 357]|
|---|

|[Figure 358]<br><br>[Figure 359]|
|---|

|[Figure 360]<br><br>[Figure 361]|
|---|

|[Figure 362]<br><br>[Figure 363]|
|---|

|[Figure 364]<br><br>[Figure 365]|
|---|

|[Figure 366]<br><br>[Figure 367]|
|---|

|[Figure 368]<br><br>[Figure 369]|
|---|

|[Figure 370]<br><br>[Figure 371]|
|---|

|[Figure 372]<br><br>[Figure 373]|
|---|

|[Figure 374]<br><br>[Figure 375]|
|---|

|[Figure 376]<br><br>[Figure 377]|
|---|

|[Figure 378]<br><br>[Figure 379]|
|---|

|[Figure 380]<br><br>[Figure 381]|
|---|

|[Figure 382]<br><br>[Figure 383]|
|---|

|[Figure 384]<br><br>[Figure 385]|
|---|

|[Figure 386]<br><br>[Figure 387]|
|---|

A white cat and a yellow dog running....

... on the moon ... in the park

... in the forest ... in the city

... near the swimming pool ... in the botanic garden

[Figure 388]

[Figure 389]

|[Figure 390]<br><br>[Figure 391]<br><br>| |
|---|
<br><br>|
|---|

|[Figure 392]<br><br>[Figure 393]<br><br>| |
|---|
<br><br>stop|
|---|

|[Figure 394]<br><br>[Figure 395]<br><br>| |
|---|
<br><br>stop|
|---|

|[Figure 396]<br><br>[Figure 397]<br><br>| |
|---|
<br><br>|
|---|

[Figure 398]

[Figure 399]

- Figure 7: Main result: Subjects compositing. Each set of the three sub-figures representing the first, middle, and the end frame of the synthesized video. The first row on the left with annotated frame shows the video synthesis of the two subjects: “cat” and the “dog” guided by the bbox directed by the annotated arrows, respectively. Next, each set of results show the varied post-fixed prompt.

###### 4.2 Quantitative evaluation

Following the methodology in Blattmann et al. (2023); Hu and Xu (2023); Jain et al. (2023), we report Fréchet Inception Distance Heusel et al. (2017) (FID), Fréchet Video Distance (FVD), Inception Score (IS), Kernel Inception Distance (KID), mean intersection of union (mIoU), and CLIP similarities (CLIPSim) metrics against the random selected 400 videos in AnimalKingdom dataset Ng et al. (2022) on all images of video sequences. As described in the supplementary materials, we evaluate both methods using the prompt set published in Jain et al. (2023). The mIoU evaluation utilizes the OWL-ViT-large open-vocabulary object detector Minderer et al. (2022) to obtain the bbox of the synthesized subject.

For a fair quantitative evaluation, we generated baseline results using Peekaboo Jain et al. (2023) and T2V-Zero Khachatryan et al. (2023) without additional conditioning input. Both TrailBlazer and Peekaboo share the same keyframed bbox, the motion vectors are used for T2V-Zero depending on the tasks below, and we use a 24-frame video sequence as our baseline comparison. We conducted two experiments with the associated random keyframing for our work: Static bbox, and Dynamic bbox.

The bboxes in the Static bbox experiments are constant across all keyframes, where the top left corner is randomly generated in the second quadrant, and the width and height is randomly selected between 25% to 50% of the image resolution. This experiments mainly evaluate the method without considering the bbox motion. The result is summarized

Method FID(↓) FVD(↓) IS(↑) KID(↓) mIoU(↑) CLIPSim(↑) T2V-Zero Khachatryan et al. (2023) 198.45 2897.57 2.48 ± 0.31 4.39% - 31.54

Peekaboo Jain et al. (2023) 159.55 1521.27 2.27 ± 0.49 2.75% 0.23 31.31 TrailBlazer 182.28 1220.98 2.91 ± 0.54 3.51% 0.26 30.93

Table 1: Quantitative results for static bbox.

Method FID(↓) FVD(↓) IS(↑) KID(↓) mIoU(↑) CLIPSim(↑) T2V-Zero Khachatryan et al. (2023) 153.36 2350.69 4.51 ± 1.36 3.01% - 31.32

Peekaboo Jain et al. (2023) 165.88 1571.04 2.16 ± 0.57 2.86% 0.25 31.92 TrailBlazer 148.18 1721.08 2.45 ± 0.61 3.12% 0.37 30.71

Table 2: Quantitative results for dynamic bbox.

in Table. 1. As observed, our performance is roughly equivalent across all metrics, while our FVD is significantly lower than that of T2V-Zero and Peekaboo. Motion vector x = 0,y = 0 is used in T2V-Zero.

Table. 2 presents the results of the Dynamic bbox experiments. The bboxes were generated by randomly specifying 2 to 6 keyframes evenly in the video clip where the location is at the image boundary and its opposite as shown in Fig. 6. The height and width of the particular bbox is between 10% to 50%. Thus the location and the size of all bboxes are varied in a video clip. Motion vector x = 8,y = 0 is used in T2V-Zero.

In Table. 2, the notable improvement is our mIoU score compared to Peekaboo, which can be attributed to the capabilities demonstrated in Fig. 6, showcasing TrailBlazer’s proficiency in generating a perspective view with dynamically changing bboxes. In comparison to Peekaboo, TrailBlazer’s FID is better, while they exhibit better FVD. This discrepancy may be explained by the nature of the AnimalKingdom Ng et al. (2022) dataset, where creatures typically perform actions in a stationary setting (e.g., birds singing, animals walking). Notably, the running cat motion in Fig. 6 is generally absent in their dataset, contributing to the lower FVD score in our case. Our better FID score suggests that the individual frame quality in our video clip is better.

In summary, the objective scores in Tables 1, 2 do not give a clear ordering of methods. However, recall that our goal is controlling movement. TrailBlazer achieves this, showing significantly better mIoU scores. Equally important, TrailBlazer shows improved subjective movement, with moving objects facing in plausible directions and having realistic motion. Please refer to our video.

###### 5 Conclusion

We have addressed the problem of controlling the motion of objects in a diffusion-based text-to-video model. Specifically, we introduced a combined spatial and temporal attention guidance algorithm, TrailBlazer, operating in the pre-trained ZeroScope model. The spatial location of a subject can be guided through simple bounding boxes. Bounding boxes and prompts can be animated via keyframes, enabling users to alter the trajectory and coarse behavior of the subject along the timeline. The resulting subject(s) fit seamlessly in the specified environment, providing a viable approach to video storytelling by casual users. Our approach requires no model finetuning, training, or online optimization, ensuring computational efficiency and a good user experience. Lastly, the results are natural, with desirable effects such as perspective, motion with the correct object orientation, and the interactions between object and environment arising automatically.

###### TrailBlazer: Trajectory Control for Diffusion-Based Video Generation

Supplementary Material

###### 6 Implementation

In this section we describe details of implementation in TrailBlazer, including the core library, hyperparameters, and other pertinent information. Our method is developed using PyTorch 2.01 Paszke et al. (2019), and the Diffusers library version 0.21.4 from Huggingface Huggingface (2023). We override the Diffusers pipeline TextToVideoSDPipeline to produce our implementation.

Parameters are selected as follows: We use classifier-free guidance with a strength of 9, conduct 40 denoising steps, and maintain a video resolution of 512x512 for the conventional stable diffusion backward denoising process. For all the comparisons with Peekaboo Jain et al. (2023) we use their official repository3 at the commit 6564274 (12 Feb 2024). In our comparisions we utilize a resolution of 576x320 as employed in the Peekaboo code to ensure fair assessment.

Regarding the parameters specific to our proposed method, the majority of our results are generated using the default values outlined as follows: We execute 5 editing steps for both spatial and temporal attention, denoted as NS ≡ NM ≡ 5. The editing coefficients cw ≡ 0.001 and cs ≡ 0.1 are used in both spatial and temporal attention in most cases. The number of trailing attention maps |T | is the only parameter that needs to be tuned. Generally, 10 ≤ |T | ≤ 20 yields satisfactory results in practice and we set |T | ≡ 15 for our paper results.

As highlighted in Sec. 1. in the main text, we adapt the pre-trained ZeroScope4 cerspense (2023) T2V model. This model is fine-tuned from the initial weights of ModelScope Luo et al. (2023)5 utilizing nearly ten thousand clips, each comprising 24 frames as training data. Consequently, we adhere to the recommended practice of setting the length of the synthesized sequence to 24 frames, drawing insights from user experiences shared in relevant blogs. 6

Spatial attention editing is performed at several resolutions with a module with the following architecture:

transformer_in.transformer_blocks.0.attn2

- down_blocks.0.attentions.0.transformer_blocks.0.attn2
- down_blocks.0.attentions.1.transformer_blocks.0.attn2

- down_blocks.1.attentions.0.transformer_blocks.0.attn2
- down_blocks.1.attentions.1.transformer_blocks.0.attn2
- down_blocks.2.attentions.0.transformer_blocks.0.attn2

- down_blocks.2.attentions.1.transformer_blocks.0.attn2

- up_blocks.1.attentions.0.transformer_blocks.0.attn2
- up_blocks.1.attentions.1.transformer_blocks.0.attn2
- up_blocks.1.attentions.2.transformer_blocks.0.attn2

- up_blocks.2.attentions.0.transformer_blocks.0.attn2
- up_blocks.2.attentions.1.transformer_blocks.0.attn2
- up_blocks.2.attentions.2.transformer_blocks.0.attn2
- up_blocks.3.attentions.0.transformer_blocks.0.attn2

- up_blocks.3.attentions.1.transformer_blocks.0.attn2 up_blocks.3.attentions.2.transformer_blocks.0.attn2

For temporal attention editing, we found that a multiple-resolution approach was not necessary and produced unpredictable results. Instead, temporal attention editing uses a single layer:

mid_block.attentions.0.transformer_blocks.0.attn2

Following Jain et al. (2023), the following prompt set is used for the experiments in our quantitative comparison in Sec. 4.2 in the main text. We include it here only for completeness. The prompt word(s) in bold case is the subject for positioning:

- • A woodpecker climbing up a tree trunk.
- • A squirrel descending a tree after gathering nuts.

3https://github.com/microsoft/Peekaboo

- 4Huggingface (2023):cerspense/zeroscope_v2_576w
- 5Huggingface (2023):damo-vilab/modelscope-damo-text-to-video-synthesis 6https://zeroscope.replicate.dev/

- • A bird diving towards the water to catch fish.
- • A frog leaping up to catch a fly.
- • A parrot flying upwards towards the treetops.
- • A squirrel jumping from one tree to another.
- • A rabbit burrowing downwards into its warren.
- • A satellite orbiting Earth in outer space.
- • A skateboarder performing tricks at a skate park.
- • A leaf falling gently from a tree.
- • A paper plane gliding in the air.
- • A bear climbing down a tree after spotting a threat.
- • A duck diving underwater in search of food.
- • A kangaroo hopping down a gentle slope.
- • An owl swooping down on its prey during the night.
- • A hot air balloon drifting across a clear sky.
- • A red double-decker bus moving through London streets.
- • A jet plane flying high in the sky.
- • A helicopter hovering above a cityscape.
- • A roller coaster looping in an amusement park.
- • A streetcar trundling down tracks in a historic district.
- • A rocket launching into space from a launchpad.
- • A deer standing in a snowy field.
- • A horse grazing in a meadow.
- • A fox sitting in a forest clearing.
- • A swan floating gracefully on a lake.
- • A panda munching bamboo in a bamboo forest.
- • A penguin standing on an iceberg.
- • A lion lying in the savanna grass.
- • An owl perched silently in a tree at night.
- • A dolphin just breaking the ocean surface.
- • A camel resting in a desert landscape.
- • A kangaroo standing in the Australian outback.
- • A colorful hot air balloon tethered to the ground.

###### 7 Ablations

We conduct ablation experiments on the number of trailing attention maps and the number of temporal steps.

Trailing attention maps. Fig. 8 shows an ablation varying the number of trailing attention maps used in our spatial cross attention process, where the top row shows our method without trailing attention maps (e.g, |T | = 0) to the bottom row (e.g., |T | = 30). The guided bbox is annotated with green bbox moving from left to right. It is observed that the astronaut remains static at the image center without the trailing attention maps. In contrast, the synthesis with a large number of trailing attentions can lead to failed results such as a flag rather than the intended astronaut. A good number of edited trailing attention maps is between |T | = 10 and |T | = 20.

Temporal attention editing. We further show an ablation test in Fig. 9 with varied number of temporal attention editing steps. We take the case of the astronaut from Fig. 8 with |T | = 10, and set NM = 0 (no editing steps), and NM = 10. The result with NM = 0 shows a red blob moving from left to right. The value NM = 10 gives satisfactory result on the astronaut, but the background along the bbox path is missing. From these results we see that a reasonable balance between spatial and the temporal attention editing must be maintained, while extreme values of either produce poor results. An intermediate value such as NM = 5 used in most of our experiments produces the desired result of an astronaut moving over a moon background.

### Ablation - #trailings

TrailBlazer: Trajectory Control for Diffusion-Based Video Generation A PREPRINT

[Figure 400]

|[Figure 401]<br><br>[Figure 402]<br><br>| |
|---|
|
|---|

|[Figure 403]<br><br>[Figure 404]<br><br>| |
|---|
|
|---|

|[Figure 405]<br><br>[Figure 406]<br><br>| |
|---|
|
|---|

[Figure 407]

|[Figure 408]<br><br>[Figure 409]<br><br>| |
|---|
|
|---|

|[Figure 410]<br><br>[Figure 411]<br><br>| |
|---|
|
|---|

0trailings10trailings20trailings30trailings

|[Figure 412]<br><br>[Figure 413]<br><br>| |
|---|
|
|---|

|[Figure 414]<br><br>[Figure 415]<br><br>| |
|---|
|
|---|

|[Figure 416]<br><br>[Figure 417]<br><br>| |
|---|
|
|---|

|[Figure 418]<br><br>[Figure 419]<br><br>| |
|---|
|
|---|

|[Figure 420]<br><br>[Figure 421]<br><br>| |
|---|
|
|---|

|[Figure 422]<br><br>[Figure 423]<br><br>| |
|---|
|
|---|

|[Figure 424]<br><br>[Figure 425]<br><br>| |
|---|
|
|---|

|[Figure 426]<br><br>[Figure 427]<br><br>| |
|---|
|
|---|

|[Figure 428]<br><br>[Figure 429]<br><br>| |
|---|
|
|---|

|[Figure 430]<br><br>[Figure 431]<br><br>| |
|---|
|
|---|

|[Figure 432]<br><br>[Figure 433]<br><br>| |
|---|
|
|---|

|[Figure 434]<br><br>[Figure 435]<br><br>| |
|---|
|
|---|

|[Figure 436]<br><br>[Figure 437]<br><br>| |
|---|
|
|---|

|[Figure 438]<br><br>[Figure 439]<br><br>| |
|---|
|
|---|

|[Figure 440]<br><br>[Figure 441]<br><br>| |
|---|
|
|---|

|[Figure 442]<br><br>[Figure 443]<br><br>| |
|---|
|
|---|

|[Figure 444]<br><br>[Figure 445]<br><br>| |
|---|
|
|---|

|[Figure 446]<br><br>[Figure 447]<br><br>| |
|---|
|
|---|

## Ablation - temproal steps

Figure 8: Ablation: Trailing maps. The rows from top to bottom show the video synthesis with 0 (no trailing maps), 10, 20, and 30 trailing maps. Prompt: “The astronaut walking on the moon”, where “astronaut” is the directed subject. The number of temporal edit steps is five in all cases.

|[Figure 448]<br><br>[Figure 449]<br><br>| |
|---|
|
|---|

|[Figure 450]<br><br>[Figure 451]<br><br>| |
|---|
|
|---|

|[Figure 452]<br><br>[Figure 453]<br><br>| |
|---|
|
|---|

|[Figure 454]<br><br>[Figure 455]<br><br>| |
|---|
|
|---|

|[Figure 456]<br><br>[Figure 457]<br><br>| |
|---|
|
|---|

|[Figure 458]<br><br>[Figure 459]<br><br>| |
|---|
|
|---|

0steps10steps

|[Figure 460]<br><br>[Figure 461]<br><br>| |
|---|
|
|---|

|[Figure 462]<br><br>[Figure 463]<br><br>| |
|---|
|
|---|

|[Figure 464]<br><br>[Figure 465]<br><br>| |
|---|
|
|---|

|[Figure 466]<br><br>[Figure 467]<br><br>| |
|---|
|
|---|

|[Figure 468]<br><br>[Figure 469]<br><br>| |
|---|
|
|---|

|[Figure 470]<br><br>[Figure 471]<br><br>| |
|---|
|
|---|

Figure 9: Ablation: Temporal edits. Following up the experiments in Fig. 8, the ablation test on the temporal attention editing with varied number of steps of the first and last frame of video reconstruction, shown at the left/right of each set of experiments. (Left/Right): No temporal attention editing, and 10 steps editing, respectively. The number of trailing is 10 for the two cases.

# Limitation

TrailBlazer: Trajectory Control for Diffusion-Based Video Generation A PREPRINT

|[Figure 472]<br><br>[Figure 473]<br><br>| |
|---|
|
|---|

|[Figure 474]<br><br>[Figure 475]<br><br>| |
|---|
|
|---|

|[Figure 476]<br><br>[Figure 477]<br><br>| |
|---|
|
|---|

|[Figure 478]<br><br>[Figure 479]<br><br>| |
|---|
|
|---|

|[Figure 480]<br><br>[Figure 481]<br><br>| |
|---|
|
|---|

|[Figure 482]<br><br>[Figure 483]<br><br>| |
|---|
|
|---|

[Figure 484]

[Figure 485]

|[Figure 486]<br><br>[Figure 487]<br><br>| |
|---|
|
|---|

|[Figure 488]<br><br>[Figure 489]<br><br>| |
|---|
|
|---|

|[Figure 490]<br><br>[Figure 491]<br><br>| |
|---|
|
|---|

|[Figure 492]<br><br>[Figure 493]<br><br>| |
|---|
|
|---|

[Figure 494]

[Figure 495]

- Figure 10: Failure cases. Prompts used in subfigures: “A red jeep driving on the road”, “A red car driving on the highway”, “a panda eating bamboo”, and “Darth Vader surfing in waves”, where the bold prompt word is the directed subject.

###### 8 Limitations

Our method shares and inherits common failure cases of the underlying diffusion model. Notably, at the time of writing, models based on CLIP and Stable Diffusion sometimes generate deformed objects and struggle to generate multiple objects and correctly assign attributes (e.g. color) to objects. We show some failures in Fig. 10. For instance, we requested a red jeep driving on the road but the synthesis shows it sinking into a mud road. The panda example shows the camera moving instead of the panda itself. The red car has implausible deformation, and Darth Vader’s light saber turns into a surf board. The length of the resulting video clips is restricted to that produced by the pre-trained model, for instance, the 24 images in the case of ZeroScope. This is not a crucial limitation, as movies are commonly (with some exceptions!) composed of short “shots” of several seconds each. The bbox guides object placement without precisely constraining it. This is an advantage as well, however, since otherwise the user would have to specify the correct x-y aspect ratio for objects, a complicated task for non-artists.

|[Figure 496]<br><br>|
|---|
|[Figure 497]<br><br>[Figure 498]<br><br>[Figure 499]<br><br>[Figure 500]<br><br>[Figure 501]<br><br>|
|[Figure 502]<br><br>[Figure 503]<br><br>[Figure 504]<br><br>[Figure 505]<br><br>[Figure 506]<br><br>|
|[Figure 507]<br><br>[Figure 508]<br><br>[Figure 509]<br><br>[Figure 510]<br><br>[Figure 511]<br><br>|
|[Figure 512]|

|[Figure 513]<br><br>[Figure 514]|
|---|

|[Figure 515]<br><br>[Figure 516]|
|---|

|[Figure 517]<br><br>[Figure 518]|
|---|

|[Figure 519]<br><br>[Figure 520]|
|---|

|[Figure 521]<br><br>[Figure 522]|
|---|

|[Figure 523]<br><br>[Figure 524]|
|---|

|[Figure 525]<br><br>[Figure 526]|
|---|

|[Figure 527]<br><br>[Figure 528]|
|---|

|[Figure 529]<br><br>[Figure 530]|
|---|

|[Figure 531]<br><br>[Figure 532]|
|---|

|[Figure 533]<br><br>[Figure 534]|
|---|

|[Figure 535]<br><br>[Figure 536]|
|---|

|[Figure 537]<br><br>[Figure 538]|
|---|

|[Figure 539]<br><br>[Figure 540]|
|---|

|[Figure 541]<br><br>[Figure 542]|
|---|

|[Figure 543]<br><br>[Figure 544]|
|---|

|[Figure 545]<br><br>[Figure 546]|
|---|

|[Figure 547]<br><br>[Figure 548]|
|---|

|[Figure 549]<br><br>[Figure 550]|
|---|

|[Figure 551]<br><br>[Figure 552]|
|---|

|[Figure 553]<br><br>[Figure 554]|
|---|

|[Figure 555]<br><br>[Figure 556]|
|---|

|[Figure 557]<br><br>[Figure 558]|
|---|

|[Figure 559]<br><br>[Figure 560]|
|---|

|[Figure 561]<br><br>[Figure 562]|
|---|

|[Figure 563]<br><br>[Figure 564]|
|---|

|[Figure 565]<br><br>[Figure 566]|
|---|

|[Figure 567]<br><br>[Figure 568]|
|---|

|[Figure 569]<br><br>[Figure 570]|
|---|

|[Figure 571]<br><br>[Figure 572]|
|---|

|[Figure 573]<br><br>[Figure 574]|
|---|

|[Figure 575]<br><br>[Figure 576]|
|---|

|[Figure 577]<br><br>[Figure 578]|
|---|

|[Figure 579]<br><br>[Figure 580]|
|---|

|[Figure 581]<br><br>[Figure 582]|
|---|

|[Figure 583]<br><br>[Figure 584]|
|---|

|[Figure 585]<br><br>[Figure 586]|
|---|

|[Figure 587]<br><br>[Figure 588]|
|---|

|[Figure 589]<br><br>[Figure 590]|
|---|

|[Figure 591]<br><br>[Figure 592]|
|---|

- Figure 11: Extreme comparison: Various conditions. TrailBlazer (left) and Peekaboo (right) use identical bboxes, while T2V-Zero uses the corresponding motion vector instead. The same prompt is used across each method. The five prompts used from top: An elephant walking on the moon; a photorealistic whale jumping out of water while smoking a cigar; A horse galloping fast on a street; A dog is running on the grass; A clownfish swimming in a coral reef. The first column at left displays the bbox, and its trajectory. For the sequences with complex motion (2nd, 3rd, 4th row), the frames shown in the figure are denoted by the red dots along the trajectory in the first column. The orange bbox in the first row represents the starting motion of the elephant running to the right near the end of video clip. For additional details, please refer to the accompanying text.

Method FID(↓) FVD(↓) IS(↑) KID(↓) mIoU(↑) CLIPSim(↑) Peekaboo Jain et al. (2023) 258.34 2863.77 1.25 ± 0.24 6.71% 0.19 30.89

TrailBlazer 215.37 3416.41 1.47 ± 0.58 5.08% 0.39 29.41 Table 3: Quantitative results for static bbox.

###### 9 Further comparison with Peekaboo

This section provides a comprehensive comparison between TrailBlazer and Peekaboo. In Fig. 11, we explore additional experiments encompassing various scenarios related to bbox size, location, and their combinations under extreme conditions. From top row to bottom: 1) Extremely fast motion by the timing of the second keyframe; 2) Rapid size changes along the bbox trajectory; 3) Zigzag trajectory; 4) Extremely fast motion through numerous keyframes; and 5) Extremely small bbox. Please refer to our the supplementary video to examine the motion for each of the following figures.

As depicted in Fig. 11, TrailBlazer excels in most extreme scenarios for the synthesized subject’s location, motion speed, and identity. For example, our representation of an elephant maintains a stationary position for the initial 75% of the video before initiating movement then runs to the right. The whale gracefully descends into the ocean during the latter part of its jumping motion. The horse accurately follows a zigzag path, simulating a galloping motion. Remarkably, the dog seamlessly follows a large number of keyframes (8 keyframes) within a 24-frame video clip, covering the distance from one boundary to the opposite in approximately 2 frames. The clownfish fits into a tiny bbox. These successes are generally not evident in the Peekaboo Jain et al. (2023) method.

The metrics presented in Table. 3 are derived from the analysis of experiments shown in Fig. 11 using the AnimalKingdom dataset Ng et al. (2022), as described in our main text. The mIoU and FID in TrailBlazer surpasses Peekaboo, indicating that our method excels at effectively generating the subject in extreme conditions. Notably, as shown in Table. 3, our mIoU is approximately twice the value of Peekaboo. As mentioned in the main text, we believe the FVD in TrailBlazer is worse than Peekaboo in this section because the AnimalKingdom dataset does not contain the varied and extreme motion that we used in our experiments.

###### 10 Subject Morphing

Subject morphing involves blending semantics for generating images and videos. To our best knowledge, TrailBlazer is first in demonstrating subject morphing by prompt manipulation in the video diffusion domain. Related concepts have earlier been shown for image generation in MagicMix Liew et al. (2022) with, for example, the "corgi coffee machine".

While subject morphing through prompt embedding interpolation may seem less intuitive for real-world applications, it is widely used in the entertainment industry for example for superheroes (e.g., the She-Hulk can transform from a human to a monstrous character). For general usage, it could potentially serve as an entry point for generating new content that is more efficient than using a single prompt, particularly due to the limitations of CLIP Radford et al. (2021). For example, it might be challenging to generate a “fish-like” cat using the prompt "A fish-like cat walking on the grass" with a diffusion model. Instead, it would be easier to accomplish this goal by combining the prompt embeddings from “A fish swimming on the grass” and “A cat walking on the grass.”

- Fig. 12 illustrates the morphing outcomes generated by TrailBlazer. All results are generated using default hyperparameter settings and involve linear interpolation of the prompt embeddings across video frames. The animated bounding boxes shift from right to left in the top two rows, and from left to right in the bottom two rows.

The outcome depicted in Fig. 12 demonstrates that morphing in a video clip can transition smoothly from one identity to another without significant artifacts. Notably, it avoids unrealistic deformations such as generating new joints in unexpected body parts (e.g., a tail on the head) or transforming one animal feature into another (e.g., an eye into an ear). Additionally, the subjects follow exactly the same motion in the synthesis (e.g., walking) across video frames.

|[Figure 593]<br><br>[Figure 594]|
|---|

|[Figure 595]<br><br>[Figure 596]|
|---|

|[Figure 597]<br><br>[Figure 598]|
|---|

|[Figure 599]<br><br>[Figure 600]|
|---|

|[Figure 601]<br><br>[Figure 602]|
|---|

|[Figure 603]<br><br>[Figure 604]|
|---|

|[Figure 605]<br><br>[Figure 606]|
|---|

|[Figure 607]<br><br>[Figure 608]|
|---|

|[Figure 609]<br><br>[Figure 610]|
|---|

|[Figure 611]<br><br>[Figure 612]|
|---|

|[Figure 613]<br><br>[Figure 614]|
|---|

|[Figure 615]<br><br>[Figure 616]|
|---|

|[Figure 617]<br><br>[Figure 618]|
|---|

|[Figure 619]<br><br>[Figure 620]|
|---|

|[Figure 621]<br><br>[Figure 622]|
|---|

|[Figure 623]<br><br>[Figure 624]|
|---|

|[Figure 625]<br><br>[Figure 626]|
|---|

|[Figure 627]<br><br>[Figure 628]|
|---|

|[Figure 629]<br><br>[Figure 630]|
|---|

|[Figure 631]<br><br>[Figure 632]|
|---|

|[Figure 633]<br><br>[Figure 634]|
|---|

|[Figure 635]<br><br>[Figure 636]|
|---|

|[Figure 637]<br><br>[Figure 638]|
|---|

|[Figure 639]<br><br>[Figure 640]|
|---|

|[Figure 641]<br><br>[Figure 642]|
|---|

|[Figure 643]<br><br>[Figure 644]|
|---|

|[Figure 645]<br><br>[Figure 646]|
|---|

|[Figure 647]<br><br>[Figure 648]|
|---|

|[Figure 649]<br><br>[Figure 650]|
|---|

|[Figure 651]<br><br>[Figure 652]|
|---|

|[Figure 653]<br><br>[Figure 654]|
|---|

|[Figure 655]<br><br>[Figure 656]|
|---|

- Figure 12: Subject Morphing. The prompts used starting from the first row: “A [cat → dog] walking on the grass”, “A [cat walking → fish swimming] on the grass”, “A [parrot → king penguin] walking on the beach”, and “A [tiger → elephant] walking in the wild park.”. Please refer to the text for more detail.

###### 11 Comprehensive ablations

Given the limited space in the primary text, here we offer more supplementary ablation tests to substantiate our proposed approach. Broadly, we illustrate the impact of the spatial and temporal placement of guidance bounding boxes (bboxes) on the overall result quality, exploring the effect of various bbox speed and size choices directed by user keyframing. To see details, please zoom in to the experiment images, and especially refer to our supplementary video.

- Fig. 13 illustrates video synthesis using the pre-trained ZeroScope model without applying our approach. Broadly, all the synthesized results exhibit fine details with plausible temporal coherence as would be seen in a real video featuring relatively slow motion. However, several side effects may be introduced alongside this realism. For example, the synthesized subject is often positioned in the same general area near the center of the images regardless of portrayed motion, and subjects like a galloping horse do not conveying the notion of speed. Additionally, artifacts such as extra or missing limbs (e.g., the cat in the second row) or other implausible results occasionally occur.

###### Zeroscope, no DD

|[Figure 657]<br><br>[Figure 658]|
|---|

|[Figure 659]<br><br>[Figure 660]|
|---|

|[Figure 661]<br><br>[Figure 662]|
|---|

|[Figure 663]<br><br>[Figure 664]|
|---|

|[Figure 665]<br><br>[Figure 666]|
|---|

|[Figure 667]<br><br>[Figure 668]|
|---|

|[Figure 669]<br><br>[Figure 670]|
|---|

|[Figure 671]<br><br>[Figure 672]|
|---|

|[Figure 673]<br><br>[Figure 674]|
|---|

|[Figure 675]<br><br>[Figure 676]|
|---|

|[Figure 677]<br><br>[Figure 678]|
|---|

|[Figure 679]<br><br>[Figure 680]|
|---|

|[Figure 681]<br><br>[Figure 682]|
|---|

|[Figure 683]<br><br>[Figure 684]|
|---|

|[Figure 685]<br><br>[Figure 686]|
|---|

|[Figure 687]<br><br>[Figure 688]|
|---|

|[Figure 689]<br><br>[Figure 690]|
|---|

|[Figure 691]<br><br>[Figure 692]|
|---|

|[Figure 693]<br><br>[Figure 694]|
|---|

|[Figure 695]<br><br>[Figure 696]|
|---|

|[Figure 697]<br><br>[Figure 698]|
|---|

|[Figure 699]<br><br>[Figure 700]|
|---|

|[Figure 701]<br><br>[Figure 702]|
|---|

|[Figure 703]<br><br>[Figure 704]|
|---|

|[Figure 705]<br><br>[Figure 706]|
|---|

|[Figure 707]<br><br>[Figure 708]|
|---|

|[Figure 709]<br><br>[Figure 710]|
|---|

|[Figure 711]<br><br>[Figure 712]|
|---|

|[Figure 713]<br><br>[Figure 714]|
|---|

|[Figure 715]<br><br>[Figure 716]|
|---|

|[Figure 717]<br><br>[Figure 718]|
|---|

|[Figure 719]<br><br>[Figure 720]|
|---|

|[Figure 721]<br><br>[Figure 722]|
|---|

|[Figure 723]<br><br>[Figure 724]|
|---|

|[Figure 725]<br><br>[Figure 726]|
|---|

|[Figure 727]<br><br>[Figure 728]|
|---|

|[Figure 729]<br><br>[Figure 730]|
|---|

|[Figure 731]<br><br>[Figure 732]|
|---|

|[Figure 733]<br><br>[Figure 734]|
|---|

|[Figure 735]<br><br>[Figure 736]|
|---|

Figure 13: Baseline results. Each row shows equally-spaced frames sampled from a video generated using ZeroScope without applying our trajectory control approach. The prompts used starting from the first row: “A fish swimming in the sea”, “The cat running on the grass field”, “The horse galloping on the road”, and “An astronaut walking on the moon”. These prompts are reused in subsequent examples in these supplementary results.

###### 11.1 Exploration and Ablation: Varied static bbox sizes

Fig. 14 shows the effect of the size of the bbox without considering motion. The results indicate that the bbox size significantly influences the outcome. In extreme cases, the top row illustrates that a smaller bbox may yield unexpected entities in the area (e.g., white smoke next to the horse) or information leakage to the neighboring area (e.g., the blue attribute affecting the road). In contrast, the bottom row demonstrates that a overly large bbox can lead to broken results in general (e.g., the fish disappearing into the coral reef, and the strange blue pattern in place of the expected blue car). We expect this may be in large part due to the centered-object bias Szabó and Horváth (2021) in the pre-trained model’s training data.

###### Statc Bbox: Prompt and size

Our recommended bbox size falls within the range of 30% to 60% for optimal reconstruction quality. Note that very small- or large-sized bboxes can still be employed in our approach, but they are best specified for a particular frame rather than the entire sequence. This is demonstrated, for example, in Fig. 15 guiding the swimming fish.

|[Figure 737]<br><br>|
|---|

|[Figure 738]<br><br>[Figure 739]|
|---|

|[Figure 740]<br><br>[Figure 741]|
|---|

|[Figure 742]<br><br>[Figure 743]|
|---|

|[Figure 744]<br><br>[Figure 745]|
|---|

|[Figure 746]<br><br>[Figure 747]|
|---|

|[Figure 748]<br><br>[Figure 749]|
|---|

|[Figure 750]<br><br>[Figure 751]|
|---|

|[Figure 752]<br><br>[Figure 753]|
|---|

|[Figure 754]<br><br>[Figure 755]|
|---|

|[Figure 756]<br><br>|
|---|

|[Figure 757]<br><br>[Figure 758]|
|---|

|[Figure 759]<br><br>[Figure 760]|
|---|

|[Figure 761]<br><br>[Figure 762]|
|---|

|[Figure 763]<br><br>[Figure 764]|
|---|

|[Figure 765]<br><br>[Figure 766]|
|---|

|[Figure 767]<br><br>[Figure 768]|
|---|

|[Figure 769]<br><br>[Figure 770]|
|---|

|[Figure 771]<br><br>[Figure 772]|
|---|

|[Figure 773]<br><br>[Figure 774]|
|---|

|[Figure 775]|
|---|

|[Figure 776]<br><br>[Figure 777]|
|---|

|[Figure 778]<br><br>[Figure 779]|
|---|

|[Figure 780]<br><br>[Figure 781]|
|---|

|[Figure 782]<br><br>[Figure 783]|
|---|

|[Figure 784]<br><br>[Figure 785]|
|---|

|[Figure 786]<br><br>[Figure 787]|
|---|

|[Figure 788]<br><br>[Figure 789]|
|---|

|[Figure 790]<br><br>[Figure 791]|
|---|

|[Figure 792]<br><br>[Figure 793]|
|---|

- Figure 14: Static bbox sizes. Each row shows the result of a static square bbox positioned at the center, where the width and height are 25%, 50%, and 90% of the original image size (represented by the the green square on the left). The prompts used in the three sets of the experiments are: “The white horse standing on the street”, “The fish swimming in the sea”, and “The blue car running on the road”.

###### 11.2 Exploration and Ablation: Varied dynamic bbox sizes

- Fig. 15 demonstrates video synthesis with a dynamically changing bbox size. In the top-left example, the bbox grows larger and then shrinks, resulting in a perspective effect where the fish swims towards the camera and then away from it. The frame highlighted in red indicates the middle keyframe with a large bbox. This aligns with our main text results in Fig. 6, showcasing that the animated tiger and car respect the bbox size. The top-right example is a comparison to the top-left, portraying the fish only swimming toward the camera.

The second and the third rows show a comparison of the same bbox condition with the prompt words “fish” (second row), and “sardine” (third row), respectively. This experiment aims to assess how well our method adapts to large bbox size variations, represented by the short/wide target bbox on the left and tall/thin target bbox on the right. The result on the left indicates that the output from the “fish” prompt does not adequately conform to the short-wide aspect ratio of the bounding box, whereas the result from the “sardine” prompt can more closely adjust to the desired bbox thanks to the elongated shape of the sardine. Conversely, in the experiment on the right, both “fish” and “sardine” perform well with the tall/thin bounding box, since the tall aspect ratio can be satisfied by a fish facing directly toward or away from the camera. In general we expect that the obtained results will mimic the situations found in ZeroScope’s training data, while views that are outside the typical data (such as a fish swimming vertically, or a horse at the top of the image) will be difficult to synthesize.

As with all our results, we see that the guided subject approximately follows the specified bounding box, but does not exactly lie within the bbox. While this is a disadvantage for some purposes, we argue that it is also an advantage for casual users – if the subject exactly fit the bounding box it would require the user to imagine the correct aspect ratio of the subject under perspective (a difficult task for a non-artists) as well as do per-frame animation of the bbox to produce the oscillating motion of the swimming fish seen here.

|[Figure 794]<br><br>|
|---|

|[Figure 795]<br><br>|
|---|

|[Figure 796]<br><br>[Figure 797]|
|---|

|[Figure 798]<br><br>[Figure 799]|
|---|

|[Figure 800]<br><br>[Figure 801]|
|---|

|[Figure 802]<br><br>[Figure 803]|
|---|

|[Figure 804]<br><br>[Figure 805]|
|---|

|[Figure 806]<br><br>[Figure 807]|
|---|

|[Figure 808]<br><br>[Figure 809]|
|---|

|[Figure 810]<br><br>[Figure 811]|
|---|

|[Figure 812]<br><br>[Figure 813]|
|---|

|[Figure 814]<br><br>[Figure 815]|
|---|

|[Figure 816]<br><br>[Figure 817]|
|---|

|[Figure 818]<br><br>[Figure 819]|
|---|

|[Figure 820]<br><br>[Figure 821]|
|---|

|[Figure 822]<br><br>[Figure 823]|
|---|

|[Figure 824]<br><br>[Figure 825]|
|---|

|[Figure 826]<br><br>[Figure 827]|
|---|

|[Figure 828]<br><br>[Figure 829]|
|---|

|[Figure 830]<br><br>[Figure 831]|
|---|

|[Figure 832]|
|---|

|[Figure 833]<br><br>|
|---|

|[Figure 834]<br><br>[Figure 835]|
|---|

|[Figure 836]<br><br>[Figure 837]|
|---|

|[Figure 838]<br><br>[Figure 839]|
|---|

|[Figure 840]<br><br>[Figure 841]|
|---|

|[Figure 842]<br><br>[Figure 843]|
|---|

|[Figure 844]<br><br>[Figure 845]|
|---|

|[Figure 846]<br><br>[Figure 847]|
|---|

|[Figure 848]<br><br>[Figure 849]|
|---|

|[Figure 850]<br><br>[Figure 851]|
|---|

|[Figure 852]<br><br>[Figure 853]|
|---|

|[Figure 854]<br><br>[Figure 855]|
|---|

|[Figure 856]<br><br>[Figure 857]|
|---|

|[Figure 858]<br><br>[Figure 859]|
|---|

|[Figure 860]<br><br>[Figure 861]|
|---|

|[Figure 862]<br><br>[Figure 863]|
|---|

|[Figure 864]<br><br>[Figure 865]|
|---|

|[Figure 866]<br><br>[Figure 867]|
|---|

|[Figure 868]<br><br>[Figure 869]|
|---|

|[Figure 870]<br><br>|
|---|

|[Figure 871]|
|---|

|[Figure 872]<br><br>[Figure 873]|
|---|

|[Figure 874]<br><br>[Figure 875]|
|---|

|[Figure 876]<br><br>[Figure 877]|
|---|

|[Figure 878]<br><br>[Figure 879]|
|---|

|[Figure 880]<br><br>[Figure 881]|
|---|

|[Figure 882]<br><br>[Figure 883]|
|---|

|[Figure 884]<br><br>[Figure 885]|
|---|

|[Figure 886]<br><br>[Figure 887]|
|---|

|[Figure 888]<br><br>[Figure 889]|
|---|

|[Figure 890]<br><br>[Figure 891]|
|---|

|[Figure 892]<br><br>[Figure 893]|
|---|

|[Figure 894]<br><br>[Figure 895]|
|---|

|[Figure 896]<br><br>[Figure 897]|
|---|

|[Figure 898]<br><br>[Figure 899]|
|---|

|[Figure 900]<br><br>[Figure 901]|
|---|

|[Figure 902]<br><br>[Figure 903]|
|---|

|[Figure 904]<br><br>[Figure 905]|
|---|

|[Figure 906]<br><br>[Figure 907]|
|---|

- Figure 15: Dynamic bbox sizes. The result showcases six synthesized video sequences with the subject directed by the yellow arrow starting at the position indicated by green bbox. The number of the bboxes corresponding to the number of keyframes used in the experiment is, clockwise from top-left, |K| = 3, 2, 2, 2, 2, and 2, respectively. The prompt used in each result: “The [X] swimming in the sea”, where “[X]” denotes the “fish” for the first and second rows, and “sardine” for the third row.

###### 11.3 Exploration and Ablation: Speed control with multiple keys

- Fig. 16 demonstrates controlling the subject’s speed through varying the number of keyframes in the video synthesis.

Given the recommended sequence length Nf = 24 for ZeroScope, we show the result of adding different keyframes in between the start and end keyframes at the left/right image boundary, simulating the cat running back and forth on the grass field. It is clear that the cat moves relatively naturally according to the motion flow indicated by the yellow arrows. For instance, the cat looks back first before turning around, rather than showing an unnatural motion where the position of the head and tail is instantaneously swapped. As the cat moves faster, motion blur also introduced in the result annotated with red arrows. We found that this motion blur is hard to eliminate using negative prompts.

###### Varied speed: multiple keys

|[Figure 908]|
|---|

|[Figure 909]<br><br>|
|---|

|[Figure 910]<br><br>[Figure 911]|
|---|

|[Figure 912]<br><br>[Figure 913]|
|---|

|[Figure 914]<br><br>[Figure 915]|
|---|

|[Figure 916]<br><br>[Figure 917]|
|---|

|[Figure 918]<br><br>[Figure 919]|
|---|

|[Figure 920]<br><br>[Figure 921]|
|---|

|[Figure 922]<br><br>[Figure 923]|
|---|

|[Figure 924]<br><br>[Figure 925]|
|---|

|[Figure 926]<br><br>[Figure 927]|
|---|

|[Figure 928]<br><br>[Figure 929]|
|---|

|[Figure 930]<br><br>[Figure 931]|
|---|

|[Figure 932]<br><br>[Figure 933]|
|---|

|[Figure 934]<br><br>[Figure 935]|
|---|

|[Figure 936]<br><br>[Figure 937]|
|---|

|[Figure 938]<br><br>[Figure 939]|
|---|

|[Figure 940]<br><br>[Figure 941]|
|---|

|[Figure 942]<br><br>[Figure 943]|
|---|

|[Figure 944]<br><br>[Figure 945]|
|---|

|[Figure 946]<br><br>|
|---|

|[Figure 947]<br><br>|
|---|

|[Figure 948]<br><br>[Figure 949]|
|---|

|[Figure 950]<br><br>[Figure 951]|
|---|

|[Figure 952]<br><br>[Figure 953]|
|---|

|[Figure 954]<br><br>[Figure 955]|
|---|

|[Figure 956]<br><br>[Figure 957]|
|---|

|[Figure 958]<br><br>[Figure 959]|
|---|

|[Figure 960]<br><br>[Figure 961]|
|---|

|[Figure 962]<br><br>[Figure 963]<br><br>|
|---|

|[Figure 964]<br><br>[Figure 965]|
|---|

|[Figure 966]<br><br>[Figure 967]|
|---|

|[Figure 968]<br><br>[Figure 969]|
|---|

|[Figure 970]<br><br>[Figure 971]|
|---|

|[Figure 972]<br><br>[Figure 973]|
|---|

|[Figure 974]<br><br>[Figure 975]<br><br>|
|---|

|[Figure 976]<br><br>[Figure 977]|
|---|

|[Figure 978]<br><br>[Figure 979]|
|---|

|[Figure 980]<br><br>[Figure 981]<br><br>|
|---|

|[Figure 982]<br><br>[Figure 983]|
|---|

Figure 16: Speed Test: number of keyframes. This result shows four synthesized video sequences with the cat’s motion directed according to the yellow arrows starting from the position indicated by green bbox. The number of the arrows denotes the number of keyframes (excluding the start/end keyframes) used in each experiment. Specifically, starting from the top-left and proceeding in left/right top/down (English reading) order, there are |K| = 2, 3, 4, and 5, keyframes, respectively. The frames highlighted with red correspond to the user-specified keyframes, excluding the start and end keyframes. The prompt used for all experiments is “A cat running on the grass field”. The red arrows in the bottom-right example shows the introduced motion blur representing fast-moving speed.

###### 11.4 Exploration and Ablation: Controlling speed with different placement of a single keyframe

Fig. 17 shows the results of moving the subject with increasing speeds. The first row shows the astronaut moving with constant speed obtained by the linearly interpolating bboxes at the left and right of the image. Starting from the second row, the astronaut holds the position of the first bbox on the left side of the image for some period of time, then moves more rapidly to the right side of the image, as illustrated in the second column of the figure. This is obtained by changing the timing of a single “middle” keyframe Kf1

###### Varied speed: key timing

, where the first keyframe and the middle keyframe have the same bbox location (e.g., Bf0 ≡ Bf1

). Similar to the results in Fig. 16, the synthesis may generate motion blur and artifacts when the speed is high (e.g., last row).

[Figure 984]

|[Figure 985]|
|---|

|[Figure 986]<br><br>[Figure 987]|
|---|

|[Figure 988]<br><br>[Figure 989]|
|---|

|[Figure 990]<br><br>[Figure 991]|
|---|

|[Figure 992]<br><br>[Figure 993]|
|---|

|[Figure 994]<br><br>[Figure 995]|
|---|

|[Figure 996]<br><br>[Figure 997]|
|---|

|[Figure 998]<br><br>[Figure 999]|
|---|

|[Figure 1000]<br><br>[Figure 1001]|
|---|

[Figure 1002]

R

[Figure 1003]

L

[Figure 1004]

[Figure 1005]

|[Figure 1006]|
|---|

|[Figure 1007]<br><br>[Figure 1008]|
|---|

|[Figure 1009]<br><br>[Figure 1010]|
|---|

|[Figure 1011]<br><br>[Figure 1012]|
|---|

|[Figure 1013]<br><br>[Figure 1014]|
|---|

|[Figure 1015]<br><br>[Figure 1016]|
|---|

|[Figure 1017]<br><br>[Figure 1018]|
|---|

|[Figure 1019]<br><br>[Figure 1020]|
|---|

|[Figure 1021]<br><br>[Figure 1022]|
|---|

[Figure 1023]

R

[Figure 1024]

L

[Figure 1025]

[Figure 1026]

|[Figure 1027]|
|---|

|[Figure 1028]<br><br>[Figure 1029]|
|---|

|[Figure 1030]<br><br>[Figure 1031]|
|---|

|[Figure 1032]<br><br>[Figure 1033]|
|---|

|[Figure 1034]<br><br>[Figure 1035]|
|---|

|[Figure 1036]<br><br>[Figure 1037]|
|---|

|[Figure 1038]<br><br>[Figure 1039]|
|---|

|[Figure 1040]<br><br>[Figure 1041]|
|---|

|[Figure 1042]<br><br>[Figure 1043]|
|---|

[Figure 1044]

R

[Figure 1045]

L

[Figure 1046]

[Figure 1047]

|[Figure 1048]|
|---|

|[Figure 1049]<br><br>[Figure 1050]|
|---|

|[Figure 1051]<br><br>[Figure 1052]|
|---|

|[Figure 1053]<br><br>[Figure 1054]|
|---|

|[Figure 1055]<br><br>[Figure 1056]|
|---|

|[Figure 1057]<br><br>[Figure 1058]|
|---|

|[Figure 1059]<br><br>[Figure 1060]|
|---|

|[Figure 1061]<br><br>[Figure 1062]|
|---|

|[Figure 1063]<br><br>[Figure 1064]|
|---|

[Figure 1065]

R

[Figure 1066]

L

[Figure 1067]

- Figure 17: Speed Test: the timing of a keyframe. The result shows four synthesized video sequences with the subject directed according to the yellow arrow starting at the position indicated by green bbox, as illustrated in the first column.

All experiments except the first use three keyframes (|K| = 3), where the timing of the internal keyframe (e.g., Kf1

) controls the duration of a stationary phase and the speed of the subsequent motion, as illustrated in the second column. The horizontal and vertical axis in the second column represent the left/right position and timing, respectively. The frame outlined in red indicates the frame controlled by Kf1

, corresponding to the time when the astronaut starts to move. The prompt used for all experiments: “The astronaut walking on the moon”.

###### 11.5 Exploration and Ablation: Irregular trajectory

We illustrate irregular trajectories determined by varied keyframes in Fig. 18. The four experiments involve a zigzag trajectory (top-left), a triangle trajectory (top-right), a discontinuous trajectory (bottom-left), and a down-pointing triangle trajectory (bottom-right). In every result the horse shows high-speed running with motion blur. However, the results with turning points show limitations in depicting the horse quickly turning around and may show artifacts. For example, in the third frame of the down-pointing triangle case, the horse appears to swap its head and tail. Difficulty portraying this turn is somewhat expected, as horses cannot naturally execute tight high-speed turns, unlike cats or dogs. On the other hand, the down-pointing triangle video naturally introduces a perspective-like size change as the horse moves higher in the image, similar to the previous results in Fig. 15, and also the tiger example Fig. 6 in our main text. In summary, maintaining consistency between the prompt and the timing and location of the keyframed bounding boxes is crucial for producing realistic results.

|[Figure 1068]<br><br>[Figure 1069]|
|---|

|[Figure 1070]<br><br>|
|---|

|[Figure 1071]<br><br>|
|---|

|[Figure 1072]<br><br>[Figure 1073]|
|---|

|[Figure 1074]<br><br>[Figure 1075]|
|---|

|[Figure 1076]<br><br>[Figure 1077]|
|---|

|[Figure 1078]<br><br>[Figure 1079]|
|---|

|[Figure 1080]<br><br>[Figure 1081]|
|---|

|[Figure 1082]<br><br>[Figure 1083]|
|---|

|[Figure 1084]<br><br>[Figure 1085]|
|---|

|[Figure 1086]<br><br>[Figure 1087]|
|---|

|[Figure 1088]<br><br>[Figure 1089]|
|---|

|[Figure 1090]<br><br>[Figure 1091]|
|---|

|[Figure 1092]<br><br>[Figure 1093]|
|---|

|[Figure 1094]<br><br>[Figure 1095]|
|---|

|[Figure 1096]<br><br>[Figure 1097]|
|---|

|[Figure 1098]<br><br>[Figure 1099]|
|---|

|[Figure 1100]<br><br>[Figure 1101]|
|---|

|[Figure 1102]<br><br>[Figure 1103]|
|---|

|[Figure 1104]<br><br>[Figure 1105]|
|---|

|[Figure 1106]<br><br>|
|---|

|[Figure 1107]<br><br>|
|---|

|[Figure 1108]<br><br>[Figure 1109]|
|---|

|[Figure 1110]<br><br>[Figure 1111]|
|---|

|[Figure 1112]<br><br>[Figure 1113]|
|---|

|[Figure 1114]<br><br>[Figure 1115]|
|---|

|[Figure 1116]<br><br>[Figure 1117]|
|---|

|[Figure 1118]<br><br>[Figure 1119]|
|---|

|[Figure 1120]<br><br>[Figure 1121]|
|---|

|[Figure 1122]<br><br>[Figure 1123]|
|---|

|[Figure 1124]<br><br>[Figure 1125]|
|---|

|[Figure 1126]<br><br>[Figure 1127]|
|---|

|[Figure 1128]<br><br>[Figure 1129]|
|---|

|[Figure 1130]<br><br>[Figure 1131]|
|---|

|[Figure 1132]<br><br>[Figure 1133]|
|---|

|[Figure 1134]<br><br>[Figure 1135]|
|---|

|[Figure 1136]<br><br>[Figure 1137]|
|---|

|[Figure 1138]<br><br>[Figure 1139]|
|---|

|[Figure 1140]<br><br>[Figure 1141]|
|---|

|[Figure 1142]<br><br>[Figure 1143]|
|---|

- Figure 18: Irregular trajectory. The figure shows four synthesized video sequences with the horse subject directed according to the yellow arrows starting from the position indicated by green bbox. The frames highlighted in red correspond to keyframes. The start and end keyframes are not indicated. The prompt used for all examples: “A horse galloping on the road”.

###### References

Daniel Arijon. Grammar of the Film Language. Focal Press, 1976. Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine,

Bryan Catanzaro, Tero Karras, and Ming-Yu Liu. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. CoRR, abs/2211.01324, 2022.

Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. Multidiffusion: Fusing diffusion paths for controlled image generation. CoRR, abs/2302.08113, 2023.

Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

cerspense. zeroscope-v2-576w, 2023. Accessed: 2023-10-01. Weifeng Chen, Jie Wu, Pan Xie, Hefeng Wu, Jiashi Li, Xin Xia, Xuefeng Xiao, and Liang Lin. Control-a-video: Controllable

text-to-video generation with diffusion models, 2023. Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. ArXiv, abs/2302.03011, 2023.

Songwei Ge, Seungjun Nah, Guilin Liu, Tyler Poon, Andrew Tao, Bryan Catanzaro, David Jacobs, Jia-Bin Huang, Ming-Yu Liu, and Yogesh Balaji. Preserve your own correlation: A noise prior for video diffusion models. Proceedings of the IEEE/CVF International Conference on Computer Vision 2023, 2023.

William Harvey, Saeid Naderiparizi, Vaden Masrani, Christian Weilbach, and Frank Wood. Flexible diffusion modeling of long videos, 2022.

Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2017.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33, 2020.

Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey A. Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, and Tim Salimans. Imagen video: High definition video generation with diffusion models. ArXiv, abs/2210.02303, 2022a.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video diffusion models, 2022b.

Zhihao Hu and Dong Xu. Videocontrolnet: A motion-guided video-to-video translation framework by using diffusion model with

controlnet, 2023. Huggingface. Stable diffusion 1 demo, 2023. Accessed: 2023-01-01. Tobias Höppe, Arash Mehrjou, Stefan Bauer, Didrik Nielsen, and Andrea Dittadi. Diffusion models for video prediction and infilling,

- 2022.

Yash Jain, Anshul Nasery, Vibhav Vineet, and Harkirat Behl. Peekaboo: Interactive video generation via masked-diffusion, 2023. Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and

Humphrey Shi. Text2video-zero: Text-to-image diffusion models are zero-shot video generators. arXiv preprint arXiv:2303.13439,

- 2023.

Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. GLIGEN:

open-set grounded text-to-image generation. CoRR, abs/2301.07093, 2023. Long Lian, Baifeng Shi, Adam Yala, Trevor Darrell, and Boyi Li. Llm-grounded video diffusion models, 2023. Jun Hao Liew, Hanshu Yan, Daquan Zhou, and Jiashi Feng. Magicmix: Semantic mixing with diffusion models. CoRR,

abs/2210.16056, 2022. Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and Tieniu Tan. Videofusion: Decomposed diffusion models for high-quality video generation, 2023. Wan-Duo Kurt Ma, J. P. Lewis, Avisek Lahiri, Thomas Leung, and W. Bastiaan Kleijn. Directed diffusion: Direct control of object placement through attention guidance, 2023.

Matthias Minderer, Alexey Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf, and Neil Houlsby. Simple openvocabulary object detection. In Computer Vision – ECCV 2022, pages 728–755, Cham, 2022. Springer Nature Switzerland.

Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, Ying Shan, and Xiaohu Qie. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. arXiv preprint arXiv:2302.08453, 2023.

Xun Long Ng, Kian Eng Ong, Qichen Zheng, Yun Ni, Si Yong Yeo, and Jun Liu. Animal kingdom: A large and diverse dataset for animal behavior understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19023–19034, 2022.

Alex Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models, 2021. Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. GLIDE: towards photorealistic image generation and editing with text-guided diffusion models. In ICML, 2022.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library, 2019.

Chenyang Qi, Xiaodong Cun, Yong Zhang, Chenyang Lei, Xintao Wang, Ying Shan, and Qifeng Chen. Fatezero: Fusing attentions for zero-shot text-based video editing. arXiv:2303.09535, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proc. ICML, 2021.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with CLIP latents. CoRR, abs/2204.06125, 2022.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10684–10695, 2022.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. CoRR, abs/2205.11487, 2022a.

Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L. Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, Seyedeh Sara Mahdavi, Raphael Gontijo Lopes, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. ArXiv, abs/2205.11487, 2022b.

Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, 2015.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In 9th International Conference on Learning

Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021, 2021. Yang Song and Stefano Ermon. Generative modeling by estimating gradients of the data distribution. In NeurIPS, 2019. Wei Sun and Tianfu Wu. Learning layout and style reconfigurable gans for controllable image synthesis. TPAMI, 44:5070–5087,

2022. Gergely Szabó and András Horváth. Mitigating the bias of centered objects in common datasets. CoRR, abs/2112.09195, 2021. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin.

Attention is all you need. In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2017. Vikram Voleti, Alexia Jolicoeur-Martineau, and Christopher Pal. Mcvd: Masked conditional video diffusion for prediction, generation, and interpolation. In (NeurIPS) Advances in Neural Information Processing Systems, 2022. Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report, 2023. Jiawei Wang, Yuchen Zhang, Jiaxin Zou, Yan Zeng, Guoqiang Wei, Liping Yuan, and Hang Li. Boximator: Generating rich and

controllable motions for video synthesis, 2024. Lilian Weng. What are diffusion models?, 2021. wiki. keyframe, 2023. Accessed: 2023-10-01. Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and

Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023.

Jinheng Xie, Yuexiang Li, Yawen Huang, Haozhe Liu, Wentian Zhang, Yefeng Zheng, and Mike Zheng Shou. Boxdiff: Text-to-image synthesis with training-free box-constrained diffusion. CoRR, abs/2307.10816, 2023.

Hanshu Yan, Jun Hao Liew, Long Mai, Shanchuan Lin, and Jiashi Feng. Magicprop: Diffusion-based video editing via motion-aware

appearance propagation, 2023. Ruihan Yang, Prakhar Srivastava, and Stephan Mandt. Diffusion probabilistic modeling for video generation, 2022a. Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma, Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao. Direct-a-video:

Customized video generation with user-directed camera movement and object motion, 2024. Zuopeng Yang, Daqing Liu, Chaoyue Wang, J. Yang, and Dacheng Tao. Modeling image composition for complex scene generation.

CVPR, pages 7754–7763, 2022b. Lvmin Zhang and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023. Bo Zhao, Weidong Yin, Lili Meng, and Leonid Sigal. Layout2image: Image generation from layout. Int. J. Comput. Vis., 128(10):

2418–2435, 2020.

