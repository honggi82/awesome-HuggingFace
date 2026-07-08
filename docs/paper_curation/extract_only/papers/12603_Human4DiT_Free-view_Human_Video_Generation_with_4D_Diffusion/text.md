# arXiv:2405.17405v2[cs.CV]23Sep2024

## Human4DiT: 360-Degree Human Video Generation with 4D Diffusion Transformer

RUIZHI SHAO*, Tsinghua University, China YOUXIN PANG*, Tsinghua University, China ZERONG ZHENG, Tsinghua University, China JINGXIANG SUN, Tsinghua University, China YEBIN LIU, Tsinghua University, China

[Figure 1]

Fig. 1. We propose Human4DiT, a novel approach to generate 360-degree high-quality, spatio-temporally coherent human videos given a reference image. With the proposed 4D diffusion transformer, our method is capable of generating monocular video, multi-view video, 3D static video, and 360-degree rotating video.

We present a novel approach for generating 360-degree high-quality, spatiotemporally coherent human videos from a single image. Our framework combines the strengths of diffusion transformers for capturing global correlations across viewpoints and time, and CNNs for accurate condition injection. The core is a hierarchical 4D transformer architecture that factorizes selfattention across views, time steps, and spatial dimensions, enabling efficient modeling of the 4D space. Precise conditioning is achieved by injecting human identity, camera parameters, and temporal signals into the respective transformers. To train this model, we collect a multi-dimensional dataset spanning images, videos, multi-view data, and limited 4D footage, along with a tailored multi-dimensional training strategy. Our approach overcomes the limitations of previous methods based on generative adversarial networks or vanilla diffusion models, which struggle with complex motions, viewpoint changes, and generalization. Through extensive experiments, we demonstrate our method’s ability to synthesize 360-degree realistic, coherent

Authors’ addresses: Ruizhi Shao*, Tsinghua University, Department of Automation, Beijing, China; Youxin Pang*, Tsinghua University, Department of Automation, Beijing, China; Zerong Zheng, Tsinghua University, Department of Automation, Beijing, China; Jingxiang Sun, Tsinghua University, Department of Automation, Beijing, China; Yebin Liu, Tsinghua University, Department of Automation, Beijing, China.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

© 2024 Association for Computing Machinery. 0730-0301/2024/12-ART $15.00 https://doi.org/10.1145/3687980

human motion videos, paving the way for advanced multimedia applications in areas such as virtual reality and animation.

##### ACM Reference Format:

Ruizhi Shao*, Youxin Pang*, Zerong Zheng, Jingxiang Sun, and Yebin Liu. 2024. Human4DiT: 360-Degree Human Video Generation with 4D Diffusion Transformer. ACM Trans. Graph. 43, 6 (December 2024), 13 pages. https: //doi.org/10.1145/3687980

1 INTRODUCTION

Human video generation is an active research topic in the field of video generation. It has widespread applications in areas such as virtual reality, animation, gaming, and movie production. Moreover, generating realistic human videos holds great significance for advancing multimedia technologies and enabling new forms of human-computer interaction.

Recently, with the rapid development of diffusion models, especially latent diffusion models [Rombach et al. 2022], leveraging diffusion models for human video generation has become the mainstream approach [Hu et al. 2023; Wang et al. 2023; Xu et al. 2023; Zhu et al. 2024]. To incorporate human priors as control conditions into the diffusion model, some approaches [Lee et al. 2023] have adopted skeleton-based schemes, using the skeletal connectivity graph as a control condition either through ControlNet [Zhang

* equal contribution.

et al. 2023b] or by concatenating it with the input. Other methods are based on the SMPL body model template [Loper et al. 2023], injecting SMPL-derived representations such as UV maps, depth, normals [Zhu et al. 2024], or direct dense pose embeddings [Xu et al. 2023] into the diffusion model. Current human video diffusion models based on the CNNs architecture [Ronneberger et al. 2015] such as UNet from Stable Diffusion could inject control conditions into the network in a pixel-aligned manner. However, the UNet’s reliance on local convolutional operations makes it more focus on local generation, resulting in relatively poorer performance for the global aspects, especially when generating long and complex human motions. Moreover, these methods only consider the human body itself, neglecting viewpoint information from the camera perspective, especially for scenarios involving significant viewpoint changes such as 360-degree human video generation. Incorporating viewpoint control signals into the network while simultaneously maintaining coherence across different viewpoints and time poses a significant challenge.

To overcome the challenges of generating complex human motions across views and time, we propose a novel video generation network architecture that combines the strengths of CNNs and diffusion transformers. First, we propose to leverage 3D SMPL models instead of 2D skeleton maps to efficiently incorporate view information and correspondences across multiple viewpoints. Our method employs the rendering of normal human maps as robust view-dependent guidance. These rendered normal maps are subsequently processed through a CNN-based encoder, which encourages the network to capture view-dependent information and precisely inject pixel-aligned conditions.

However, it remains a challenge to ensure temporal consistency when using CNNs-based architectures to generate long videos. Recently, OpenAI’s recent work on long video generation, SORA [OpenAI 2024], adopted a diffusion transformer [Peebles and Xie 2023] architecture and demonstrated substantially better realism and spatiotemporal coherence than CNNs-based models. Inspired by SORA, we introduce a 4D diffusion transformer for human video generation, which not only exhibits greater potential for scalability, but also demonstrates the capacity to learn complex 360-degree human video generation. By employing the unified attention mechanism across various dimensions, our 4D DiT could efficiently build spatialtemporal correspondences across different views and time, preserving spatial-temporal consistency in the generated human videos. However, directly applying a diffusion transformer to learn correlations over the views and time would be computationally prohibitive. Therefore, we propose an efficient novel 4D transformer architecture. Its core principle is to cascadingly learn the correlations across the 4D space (view, time, height, width) via self-attention. Specifically, we factorize the 4D diffusion transformer into three transformer blocks: 2D image transformer blocks, temporal transformer blocks, and view transformer blocks, each attending to different dimensions of the 4D space. These three types of blocks are interconnected to form a 4D transformer block. Multiple such 4D transformer blocks are then cascaded to construct the final 4D transformer. This efficiently captures the interrelations between body parts (height, width) across viewpoints (views) and time steps (times).

To enhance the controllability of our proposed 4D diffusion transformer beyond SMPL motion, we integrate additional control signals into the respective network modules, including human identity, temporal information, and camera parameters. Human identity embeddings and latent tokens, extracted via the CLIP and a CNNs-based encoder, are incorporated into image transformers. Camera embeddings, derived from camera parameters, are integrated into the view transformer, while temporal embeddings are incorporated into the temporal transformer. Through these modules, we effectively inject various control conditions into the network, facilitating viewpoint manipulation and the generation of high-fidelity, consistent human videos.

To train our proposed 4D diffusion transformer model, we also collected a large multi-dimensional dataset and devised a multidimensional training strategy that fully leverages all available data modalities. Our multi-dimensional dataset comprises images, videos, multi-view videos, 3D scans, as well as a limited amount of 4D scans spanning different viewpoints and time steps. During the inference stage, we propose a spatio-temporal consistent diffusion sampling strategy that enables the generation of long 360-degree videos despite limited spatio-temporal window constraints. The strategy is carried out in two stages. The first stage treats the 360degree video as a monocular long video sequence, maximizing the temporal window to ensure long-term temporal consistency. The second stage regards the 360-degree video as a collection of multiview video clips, using a larger viewpoint window with a smaller temporal window, encouraging consistency across viewpoints.

To summarize, our main contributions are:

- • We are the first to introduce diffusion transformers to human video generation. Combined with methods that simply use CNNs-based encoder and several control modules, our method achieves high-quality 360-degree spatio-temporally consistent generation of long human videos.
- • We propose an efficient 4D diffusion transformer architecture composed of three transformers attending to 2D images, time, and viewpoints respectively, significantly reducing computational requirements while effectively capturing correlations between body parts across space and time.
- • We collect a large multi-dimensional 4D human dataset and introduce a multi-dimensional training strategy that fully leverages data from all modalities.
- • During inference, we also propose a spatio-temporal consistent diffusion sampling strategy to generate coherent 360degree long human videos.

2 RELATED WORK 2.1 Human Video Generation

Human video generation is the task of creating realistic and temporally coherent videos of humans from input data such as text descriptions, images, or motion sequences. The current paradiam of human video generation lies in two main categories: GAN-based and diffusion-based approaches.

GAN-based methods [Siarohin et al. 2019a,b, 2018; Tian et al. 2021; Wang et al. 2021, 2020] leverage the inherent generative capabilities of adversarial networks [Goodfellow et al. 2014; Mirza and Osindero

[Figure 2]

- Fig. 2. Pipeline of Human4DiT: our framework is based on 4D diffusion transformer, which adopts a cascaded structure consisting of the 2D image transformer, the view transformer, and the temporal transformer. The input contains a reference image y𝑟, dynamic SMPL sequences P, and camera parameters c. Starting from a generated noisy latent representation z𝑡, we denoise them conditioned on y𝑟, P, and c to recover the original latent frames. First, the 2D image transformer block is designed to capture spatial self-attention within latent frame tokens and pose frame tokens extracted by latent encoder Ez and pose encoder E𝑝, respectively. In addition, ID tokens y𝑖𝑑 and image embedding y𝑒 extracted from y𝑟 by ID encoder E𝑖𝑑 and CLIP are also injected to ensure identity consistency. Secondly, we use the view transformer block to learn correspondences across different viewpoints conditioned on camera embedding. Finally, we adopt a temporal transformer to capture temporal correlations with time embedding. The time embedding and camera embedding are obtained by positional encoding time T and camera c, respectively.5

2014] to spatially transform reference images according to input motion. These approaches commonly employ warping functions to generate sequential video frames, aiming to fill in missing regions and improve visually implausible areas within the generated content. While GAN-based methods have shown promising results in dynamic visual content generation, they often struggle with effectively transferring motion, especially when there are significant variations in human identity and scene dynamics between the reference image and the source video motion. This can lead to unrealistic visual artifacts and temporal inconsistencies in the synthesized videos.

On the other hand, diffusion models, known for the superior generation quality and stable controllability, have been successfully integrated into human image animation. These models [Bhunia et al. 2023; Hu et al. 2023; Karras et al. 2023; Wang et al. 2023; Xu et al. 2023; Zhu et al. 2024] employ various strategies, such as texture diffusion blocks, optical flow synthesis in latent space, and motion representation using flow maps, to enhance the visual fidelity of the generated videos. Animate Anyone [Hu et al. 2023] employs a UNet-based ReferenceNet to extract features from reference images and incorporates pose information through an efficient pose guider. However, recent diffusion-based methods are still mainly relying on image diffusion model and face challenges in maintaining texture consistency and temporal stability across frames. Furthermore, these methods don’t explore view controllability.

2.2 Diffusion Model with Camera Control

Camera information is conditioned in diffusion models for view control. One line of works inject camera parameters into the textto-image diffusion models for consistent view synthesis from a single input image, which has inspired researchers to explore their potential for generating consistent multi-view images [Liu et al.

2023b; Shi et al. 2023b; ?; ?]. Zero-1-to-3 [Liu et al. 2023b] finetunes Stable diffusion by conditioning camera poses for zero-shot novel view synthesis. Syncdreamer [Liu et al. 2023a] employs 3D volumes and depth-wise attention to maintain consistency across views. MVDream [Shi et al. 2023b] and other methods like Wonder3D [Long et al. 2023] and Zero123++ [Shi et al. 2023a] leverage 3D self-attention to extend multi-view image generation to more general and efficient. There are other works [He et al. 2024; Yang et al. 2024] inject camera information into text-to-video (T2V) models. For example, Direct-a-Video [Yang et al. 2024] injects quantitative camera movements into temporal cross-attention blocks for view control. CameraCtrl [He et al. 2024] proposes to use an efficient representation, Plücker ray embedding, for camera conditions.

2.3 Diffusion Transformer

The transformer architecture [Vaswani et al. 2017] has revolutionized the field of natural language processing, with models like GPT [Radford et al. 2018, 2019] achieving remarkable success. Recent research has demonstrated the potential of transformers in various computer vision tasks, including image classification [Touvron et al. 2021; Yuan et al. 2021], semantic segmentation [Strudel et al. 2021; Xie et al. 2021; Zheng et al. 2021].

Building upon this progress, the Diffusion Transformer (DiT) [Peebles and Xie 2023] and its variants [Bao et al. 2023; Zheng et al. 2023] have taken a step further by substituting the conventional convolutional-based U-Net backbone [Ronneberger et al. 2015] with transformers in diffusion models. This architectural shift offers enhanced scalability compared to U-Net-based models, enabling the seamless expansion of model parameters. Recently, the transformer architecture is also integrated into text-to-video models [Lu et al. 2023; Ma et al. 2024; OpenAI 2024], improving generation performance.

[Figure 3]

- Fig. 3. Human4DiT dataset: In addition to the open-source dataset, we collect a multi-dimension dataset including 10k monocular videos from Internet, 5k high-quality 3D human scans and 100 animatable human models for dynamic free-view rendering.

- 3 OVERVIEW Given a reference image of a person y𝑟, a sequence of dynamic

SMPL models {P𝑇𝑖=1 = (𝜃, 𝛽)𝑇𝑖=1}, and camera parameters {c𝑉𝑖=1}, the goal of our method is to generate a video of that person performing

the corresponding motion from the specified viewpoint. The overall pipeline of our approach is illustrated in Figure 2. Since our framework is based on a latent diffusion transformer, we first generate a

noisy latent representation z𝑡. We then inject y𝑟, {P𝑇𝑖=1}, and {c𝑉𝑖=1} as control signals into the diffusion transformer, which iteratively

performs denoising to ultimately generate the latent video z0 and decoded 360-degree human video x0(Sec. 4).

To train our proposed model, we collected a multi-dimensional human dataset comprising images, videos, multi-view data, as well as 3D and 4D human data. We further introduce a multi-dimensional mixed training strategy that fully leverages all available data modalities for effective network training (Sec. 5).

During the inference stage, to enable the generation of 360-degree long-duration human motion videos under limited temporal and viewpoint window constraints, we propose an efficient diffusion sampling strategy. This strategy achieves improved spatial-temporal coherence by two-staged viewpoint and temporal window planning in the diffusion sampling process (Sec. 6).

- 4 NETWORK STRUCTURE

In this section, we introduce the core components of our network structure: the 4D diffusion transformer and the control condition injection modules. To efficiently establish spatial-temporal relationships, our 4D diffusion transformer adopts a cascaded structure consisting of the 2D image, the temporal, and the view transformer blocks. Additionally, our control condition injection modules comprise the reference image injection module, the SMPL injection module, the time injection module, and the viewpoint injection module. Notably, the reference image injection module and the SMPL injection module utilize CNNs-based architectures to ensure pixel-aligned injection of the control conditions.

4.1 4D Diffusion Transformer

Our 4D diffusion transformer performs denoising on a randomly initialized noisy multi-view latent human video z𝑡. After multiple denoising steps, it produces the denoised output z0, which is eventually decoded by a VAE to generate the video. We first utilize a CNNs-based encoder Ez to extract latent tokens. Then the input could be regarded as a set of tokens in 5D (z𝑡 ∈ R𝑉×𝑇×𝐻×𝑊 ×𝐶), where 𝐶 represents the feature dimension of each token, 𝑉 is the viewpoint dimension, 𝑇 is the time dimension, and 𝐻 and 𝑊 are the height and width, respectively. To establish correlations across viewpoints, time, and 2D spatial dimensions, we propose a novel cascaded diffusion transformer architecture. It could decompose the complex 4D attention into three types of attention including image, temporal and view attention, which efficiently learns the 4D correspondences and decreases the memory usage.

First, we feed 𝑧𝑡 into a 2D image transformer block to capture spatial self-attention within each frame:

z𝑠𝑡 = rearrange(z𝑡, (𝑉 ×𝑇,𝐻 ×𝑊,𝐶)), Q𝑠, K𝑠, V𝑠 = 𝑓𝑠Q(z𝑠𝑡), 𝑓𝑠K(z𝑠𝑡), 𝑓𝑠V(z𝑠𝑡),

(1)

Q𝑠K𝑇𝑠 √︁𝑑𝑘

zˆ𝑠𝑡 = softmax(

)V𝑠,

where 𝑓𝑠Q, 𝑓𝑠K, 𝑓𝑠V are the linear layers in the transformer block. The rearrange operation reshapes the tensor z𝑡 into a 3D tensor z𝑠𝑡 ∈ 𝑅(𝑉×𝑇)×(𝐻×𝑊 )×𝐶, where the first dimension 𝑉 ×𝑇 is treated as the batch size. The actual attention operation is then performed on the last two dimensions𝐻×𝑊 and𝐶, which correspond to the 2D spatial dimensions of the images. After the 2D image transformer, we further model correspondences across different viewpoints through the view transformer block:

z𝑡𝑣 = rearrange(zˆ𝑠𝑡, (𝑇,𝑉 × 𝐻 ×𝑊,𝐶)), Q𝑣, K𝑣, V𝑣 = 𝑓𝑣Q(z𝑡𝑣), 𝑓𝑣K(z𝑡𝑣), 𝑓𝑣V(z𝑡𝑣),

(2)

Q𝑣K𝑇𝑣 √︁𝑑𝑘

zˆ𝑡𝑣 = softmax(

)V𝑣,

In the view transformer, due to the substantial variations across different viewpoints, especially between frontal, side, and back views, we perform attention jointly across the viewpoint and the 2D spatial dimensions. This attention allows for better modeling of the global correlations across the entire frame. It also makes the view transformer the most computationally and memory-intensive component, limiting the maximum viewpoint window size. We will address this constraint during the inference process, as discussed later. After the view transformer, we finally employ a temporal transformer to capture temporal correlations across time steps.

z𝑚𝑡 = rearrange(zˆ𝑡𝑣, (𝑉 × 𝐻 ×𝑊,𝑇,𝐶)), Q𝑚, K𝑚, V𝑚 = 𝑓𝑚Q(x𝑚𝑡 ), 𝑓𝑚K(z𝑚𝑡 ), 𝑓𝑚V(z𝑚𝑡 ),

Q𝑚K𝑇𝑚 √︁𝑑𝑘

zˆ𝑚𝑡 = softmax(

)V𝑚,

(3)

|Dataset|Clips Frames Views Resolution 3D Dynamic SMPL Body<br><br>|
|---|---|
|HumanArt [Ju et al. 2023] TikTok [Jafarian and Park 2021] TalkShow [Yi et al. 2023] AIST [Tsuchida et al. 2019] Motion-X [Lin et al. 2024] DNA-rendering [Cheng et al. 2023] Twindom [Twindom 2022] THuman2.0 [Yu et al. 2021] THuman-CloSET [Zhang et al. 2023a] Bedlam [Black et al. 2023]<br><br>|- 50k 1 5122-10242 ✗ ✗ Ground Truth Half-body + Full-body 2k 600k 1 480P-720P ✗ ✓ Fitting Half-body + Full-body<br><br>1k 300k 1 480P-720P ✗ ✓ Fitting Half-body 10k 2000k 6 1080P ✗ ✓ Ground Truth Full-body 10k 1000k 1 720P-1080P ✗ ✓ Ground Truth Full-body<br>2k 400k 16 4K ✗ ✓ Fitting Full-body 2k 360k 180 10242 ✗ ✓ Fitting Full-body<br><br><br>500 90k 180 10242 ✓ ✗ Ground Truth Full-body 500 90k 180 10242 ✓ ✗ Ground Truth Full-body 10k 1500k 1 720P-1080P ✗ ✓ Ground Truth Full-body|
|Human4DiT-3D<br><br>Human4DiT-Video<br><br>Human4DiT-4D<br>|5k 900k 180 10242 ✓ ✗ Fitting Full-body<br><br>10k 2000k 1 720P-1080P ✗ ✓ Fitting Half-body + Full-body<br><br>100 168k 180 10242 ✓ ✓ Fitting Full-body|

Table 1. The collected multi-dimensional training dataset.

The three transformer blocks (2D image, view, and temporal) are interconnected to form a single 4D transformer block. Our complete 4D diffusion transformer architecture is composed of 10 cascaded 4D transformer blocks. Through this cascaded multi-level attention scheme, our approach significantly reduces the computational overhead while improving the training efficiency and effectively ensuring spatio-temporal consistency in the generation process.

- 4.2 Control Modules

- 4.2.1 Camera Control Module. To incorporate camera viewpoint

control into the network, we assume the first camera c1 as the world coordinate system, extract its rotation matrix (i.e., the identity matrix) as a 9D tensor r1, and apply positional encoding:

y𝑐 = (sin(20𝜋r1), cos(20𝜋r1), ..., (sin(2𝐿−1𝜋r1), cos(2𝐿−1𝜋r1))

(4)

The parameters of other cameras {c𝑉𝑖=2} are computed as rotation matrices relative to the first camera, and their encodings are ob-

tained through the same positional encoding process. We then map these encodings to the same dimension as the CLIP image embeddings using an MLP 𝑓𝑐 and inject them into the view transformer module via addition to influence the intermediate features after self-attention:

z𝑡𝑣′ = z𝑡𝑣 + 𝑓𝑐(y𝑐) (5) This relative encoding formulation effectively represents the correlations between different viewpoints, enabling better generation with varying multi-view setups.

- 4.2.2 Temporal Embedding Module. To incorporate temporal control into the network, we apply positional encoding directly to the time𝑇𝑚 (frame number):

y𝑚 = (sin(20𝜋𝑇𝑚), cos(20𝜋𝑇𝑚), ..., (sin(2𝐿−1𝜋𝑇𝑚), cos(2𝐿−1𝜋𝑇𝑚)

(6) We then map the temporal encoding to the latent space using an MLP 𝑓𝑚 and add it to the temporal transformer’s features after self-attention:

z𝑚𝑡 ′ = z𝑚𝑡 + 𝑓𝑚(y𝑚) (7)

- 4.2.3 SMPL Control Module. To inject the SMPL parameters into the network, we first obtain the SMPL mesh vertex positions from the given shape and pose parameters p𝑖 = 𝐿𝐵𝑆(𝜃𝑖, 𝛽𝑖) of each frame 𝑖. We then render these vertices p𝑖 into normal maps M𝑛(𝑖,𝑣) using

camera parameters c𝑣, as normals provide an effective representation of the 3D human. Since the camera parameters c𝑣 have already been injected into the network, we decouple the SMPL information from the rendered normal maps by multiplying it with the inverse of the camera rotation matrix r𝑣−1. We then use a CNNs-based Encoder E𝑝 to extract features y𝑛 from these SMPL rendered normal maps M𝑛(𝑖,𝑣). These features y𝑛 are eventually added to the input x𝑡 of the 4D Diffusion Transformer before being fed into the network. Through this approach, the control condition is injected into the network in a pixel-aligned manner, ensuring consistent and accurate human motion video generation.

4.2.4 Human Identity Reference Module. During human video generation, we extract the human identity from a reference image and inject it into the network. We employ two ways to maintain identity consistency. First, similar to the SMPL injection, we use a CNNsbased encoder 𝐸𝑖𝑑 to extract human features y𝑖𝑑 from the reference image. These features are then added to the input of the 4D diffusion transformer x𝑡, allowing the network to better capture the detailed identity characteristics from the reference image. Additionally, we extract an image embedding y𝑒 using CLIP and inject it into each transformer block via cross-attention after the self-attention mechanism:

Q, K, V = 𝑓 Q(zˆ𝑡), 𝑓 K(y𝑒), 𝑓 V(y𝑒), z˜𝑡 = softmax(

(8)

QK𝑇 √︁𝑑𝑘

)V

This ensures that the generated output maintains global consistency with the reference human identity in addition to preserving local details.

5 DATASET AND TRAINING STRATEGY

To train this 4D transformer model, we collected a multi-dimensional dataset and devised a multi-dimensional training strategy that fully leverages all available data modalities. As shown in Tab. 1, in addition to utilizing publicly available human datasets, we collect a new multi-dimensional dataset comprises human images, videos, multiview videos, 3D data, as well as a small amount of 4D data spanning different viewpoints and time steps. It contains 5k 3D human scans capture by camera rigs, 10k videos from YouTube/Bilibili, and 100 animatable human models from artists. Some samples are presented

- in Fig. 3. We adopt two methods to estimate SMPL for our dataset: For multi-view and 3D/4D data, we detect 2D key points and optimize SMPLs to align with the corresponding images [Zhang et al. 2021]. For monocular videos, we directly use Humans-in-4D [Goel et al. 2023] to estimate image-aligned SMPLs and cameras.

We employ different training strategies for each data modality. For the 2D images dataset, we only train the 2D transformer, using the CLIP image embedding as the human identity condition. For the single-view video dataset, we train the 2D and temporal transformers, randomly selecting one frame from the video as the reference image. For multi-view videos, we train all transformers simultaneously, randomly choosing one frame from the multi-view data as the reference image. For 3D dataset, we train our transformer with two strategies. multi-view images and train only the 2D and view transformers, randomly selecting one rendered view as the reference. For 4D dataset, we render it into dense-view videos or videos with continuous viewpoint movements and train all transformers concurrently.

This multi-dimensional training approach allows us to effectively leverage diverse data sources, with each modality contributing to different components of the model. The unified 4D transformer architecture seamlessly integrates information from all modalities during training, enabling coherent 360-degree human video generation.

Algorithm 1: Efficient Spatial-Temporal Sampling Input: 𝑇𝑖 (number of inference timesteps); y (Control signals); 𝑇𝐿

(number of frames); 𝑀𝑇1,𝑀𝑇2,𝑀𝑉2 (window size of view and time); 𝜖𝜃 (4D diffusion transformer)

Output: 𝑥0 (sampled latent video)

- 1 x𝑡 ∼ N(0,𝐼), 𝜖 ∼ N(0,𝐼) ;
- 2 𝑁𝑇1 = 𝑀𝑇𝐿1 𝑇

, 𝑁𝑇2 = 𝑀𝑇𝐿2 𝑇

, 𝑁𝑉2 = 𝑀𝑇𝐿2 𝑉

;

- 3 for 𝑡 ← 𝑇𝑖 to 1 do

- 4 for 𝑚 ← 1 to 𝑁𝑇1 do

- 5 𝑠𝑙𝑖𝑐𝑒1 = 𝑚𝑀𝑇1 : (𝑚 + 1)𝑀𝑇1 ;
- 6 𝜖1 𝑠𝑙𝑖𝑐𝑒1 = 𝜖𝜃 (x𝑡 𝑠𝑙𝑖𝑐𝑒1 ,𝑡, y 𝑠𝑙𝑖𝑐𝑒1 );

- 7 for 𝑚 ← 1 to 𝑁𝑇2, 𝑣 ← 1 to 𝑁𝑉2 do

- 8 𝑠𝑙𝑖𝑐𝑒2 = 𝑣 : 𝑇𝐿 : 𝑁𝑉2 ,𝑚𝑀𝑇2 : (𝑚 + 1)𝑀𝑇2 ;
- 9 𝜖2 𝑠𝑙𝑖𝑐𝑒2 = 𝜖𝜃 (x𝑡 𝑠𝑙𝑖𝑐𝑒2 ,𝑡, y 𝑠𝑙𝑖𝑐𝑒2 );

- 10 𝜇𝜃 ← √𝛼1𝑡 𝑥𝑡 − √1𝛽−𝑡𝛼¯𝑡 (𝜆1𝜖1 + 𝜆2𝜖2) ;

- 11 𝑥𝑡−1 ← 𝜇𝜃 + 𝜎𝑡𝜖

- 12 return 𝑥0;

6 EFFICIENT SPATIAL-TEMPORAL SAMPLING

Our 4D diffusion transformer is capable of generating multi-view human motion videos. However, it cannot directly generate 360-degree videos where both viewpoint and time vary simultaneously. This limitation arises from computational memory constraints, where the total number of frames 𝑀 in the network input is bounded by the product of the temporal window size 𝑀𝑇 and the view window size 𝑀𝑉 . To enable generation of long 360-degree videos, we propose an efficient and spatio-temporally consistent sampling method. During the denoising process, this method employs two strategies.

|Method|PSNR ↑ SSIM ↑ LPIPS ↓ FVD ↓<br><br>|
|---|---|
|Disco MagicAnimate AnimateAnyone Champ|20.07 0.661 0.285 585.3<br><br>21.08 0.717 0.256 550.7<br><br>22.18 0.789 0.195 479.5<br><br><br>22.88 0.824 0.171 359.3<br><br>|
|AnimateAnyone* Champ*<br><br>|24.27 0.862 0.148 341.0 24.93 0.870 0.139 307.6|
|Ours (small) Ours (image) Ours|25.45 0.877 0.121 276.1<br><br>24.15 0.858 0.147 491.2<br><br>26.12 0.888 0.116 237.4<br>|

Table 2. Quantitative comparison on monocular video. * indicates that we fine-tuned the model on our dataset. Ours (small) is the small version of our 4D transformer model with fewer layers. Ours (image) indicates that our model only has image transformer blocks.

The first strategy treats the 360-degree video as a monocular long video sequence, maximizing the temporal window 𝑀𝑇1 to ensure long-term temporal consistency. The second strategy regards the 360-degree video as a collection of multi-view short video clips, using a larger viewpoint window 𝑀𝑉2 with a smaller temporal window 𝑀𝑇2, focusing on maintaining consistency across viewpoints. During denoising, the noise predictions from these two strategies are combined using respective weightings 𝜆1,𝜆2. The specific sampling algorithm is outlined in Alg. 1.

This sampling approach combines the advantages of both strategies. Globally, it leverages the view transformer to maintain consistency across large viewpoint separations. For smaller inter-frame motions between adjacent viewpoints, the temporal transformer ensures coherence. Ultimately, this method achieves spatio-temporally consistent generation of long 360-degree human motion videos under limited input window constraints.

7 EXPERIMENT

In this section, to validate the capabilities of our 4D diffusion transformer, we conducted comprehensive comparisons against current state-of-the-art human video generation methods including Disco [Wang et al. 2023], MagicAnimate [Xu et al. 2023], AnimateAnyone [Hu et al. 2023], and Champ [Zhu et al. 2024] on monocular video, multi-view video, 3D static video, and 360-degree video generation.

7.1 Implementation Details

7.1.1 Network Architecture. Our method is based on the latent diffusion model that utilizes the VAE from Stable Diffusion XL. Our 4D diffusion transformer contains 10 4D transformer blocks, totaling 30 transformer layers. Our CNNs-based encoder including E𝑝, E𝑖𝑑 compresses the input by a factor of 8, with each token having 1280 channels, and the text embedding and image identity embedding are also mapped to 1280 channels. The latent encoder Ez further compresses the input by a factor of 2. During training, the video data is resized to 768x768 resolution, and for each GPU, the input consists of 24 frames. When training on image data, we use a batch size of 24, while for monocular video training, we use a batch size of 1 with a video length of 24. For multi-view video, 3D video, and 360-degree video training, we use a batch size of 1 with a video length of 6 and 4 views.

[Figure 4]

Fig. 4. Qualitative comparison on monocular video.

|Method|PSNR ↑ MV 3D 4D<br><br>|SSIM ↑ MV 3D 4D<br><br>|LPIPS ↓ MV 3D 4D|FVD ↓ MV 3D 4D<br><br>|
|---|---|---|---|---|
|Disco MagicAnimate AnimateAnyone Champ<br><br>|18.86 17.13 19.98<br>19.30 19.28 21.74<br><br><br>19.87 20.53 22.01<br>20.15 21.11 23.35<br>|0.796 0.882 0.872 0.845 0.906 0.920 0.858 0.922 0.912 0.886 0.927 0.922<br><br>|0.293 0.209 0.169 0.232 0.159 0.135 0.216 0.111 0.131 0.203 0.106 0.110|646.2 451.6 559.7 517.3 356.4 418.4 472.8 285.4 410.3 442.4 204.3 347.6<br><br>|
|AnimateAnyone* Champ*|20.78 21.92 24.06 20.90 22.18 24.34<br><br>|0.904 0.937 0.931<br><br>0.905 0.940 0.935<br><br><br>|0.193 0.068 0.089 0.185 0.060 0.083<br><br>|413.0 194.9 301.2 397.3 186.3 286.1|
|Ours (small) Ours (image) Ours (image+temporal) Ours<br><br>|22.02 22.94 24.56<br><br>20.66 21.41 24.15<br>21.16 22.58 24.74<br>22.40 23.37 25.02<br>|0.911 0.952 0.935 0.899 0.931 0.928 0.907 0.941 0.936 0.920 0.962 0.947<br><br>|0.166 0.053 0.069 0.197 0.072 0.088 0.190 0.058 0.085 0.159 0.045 0.062<br><br>|344.3 129.7 256.5 563.1 319.0 435.4 374.2 165.2 274.1<br><br>296.53 110.0 234.8|

Table 3. Quantitative comparison on multi-view (MV), 3D static (3D), and 360-degree (4D) videos. * indicates that we fine-tuned the model on our dataset. Ours (small) is the small version of our 4D transformer model with fewer layers. Ours (image) indicates that our model only has image transformer blocks. Ours (image+temporal) indicates our model without the view transformer blocks.

- 7.1.2 Training Details. We employed 24 A100 GPUs, and the total training time was 14 days. We train our model with a learning rate 1e-5, first on the image dataset for 50k iterations, then on the full dataset for 150k iterations. We use ColossalAI’s HybridAdam optimizer with Zero2 strategy and zero weight decay to optimize parameters. We also clip the gradient in the range of [-1, 1]. 7.2 Comparisons
- 7.2.1 Comparisons on Monocular Video. For monocular video, we randomly select 200 videos from the Human4DiT-Video dataset

as our test set for comparison, with the first frame of each video serving as the reference image. During inference, our method employs a temporal window of 24 frames, with an overlap of 6 frames between consecutive windows. For Disco [Wang et al. 2023], MagicAnimate [Xu et al. 2023], and Champ [Zhu et al. 2024], we use their official open-source code. As for AnimateAnyone [Hu et al. 2023], we employ the opensource implementation from MooreThreads 1 . To ensure a more fair comparison, we fine-tuned AnimateAnyone

1https://github.com/MooreThreads/Moore-AnimateAnyone

[Figure 5]

Fig. 5. Ablation study of spatial-temporal sampling.

and Champ on our Human4DiT dataset for 10k iterations. Results from these fine-tuned models are denoted with an asterisk (*) in our evaluations. Quantitative comparisons are presented in Tab. 2, where our method demonstrates a clear numerical advantage, significantly outperforming other approaches. This highlights the superiority of our 4D diffusion transformer over CNNs-based methods. We have also conducted qualitative comparisons, with results shown

- in Fig. 4. Our method generates more natural dynamic effects with fewer deformation and jitter artifacts compared to other methods, indicating the 4D transformer’s stronger capability in establishing spatial-temporal consistency than U-Net-based approaches. Please refer to our submitted project webpage for dynamic video effects.

- 7.2.2 Comparisons on Multi-view Video. For multi-view settings, we select 25 multi-view groups from the AIST and DNA-rendering datasets respectively as our test set for comparison, with each group containing videos captured from 4 different viewpoints. We use the first frame of the frontal view as the reference image. During inference, our method employs the spatial-temporal sampling strategy, with a spatial window size of 4 views and a temporal window of 6 frames for spatial sampling. For temporal sampling, the window size

is 24 frames. The blending weights for spatial and temporal 𝜆1,𝜆2 are set to 0.5 and 0.5, respectively. For Disco, MagicAnimate, Champ, and AnimateAnyone, we treat each view’s video as a monocular video and perform inference separately. Quantitative comparisons are presented in Tab. 3, where we masked out the backgrounds for metric computation since inferring other views’ backgrounds from a single view is an ill-posed problem. Our method outperforms others, demonstrating the 4D diffusion transformer’s superior ability to establish stronger cross-view correlations compared to U-Net-based approaches. We have also conducted qualitative comparisons, with results shown in Fig. 8. Our method generates spatio-temporally consistent multi-view videos without exhibiting multi-face artifacts. Please refer to the project webpage for dynamic video effects.

|Method|Parameters T-size Resolution FPS<br><br>|
|---|---|
|AnimateAnyone Champ Ours (small) Ours<br><br>|2.07B 24 512x512 0.769<br><br>2.04B 24 768x768 0.657 1.83B 24 768x768 0.605<br>3.10B 24 768x768 0.443<br>|

Table 4. Performance comparisons of different video models. T-size is the temporal window size. FPS is the average number of generated frames every second with 25 inference steps.

- 7.2.3 Comparisons on 3D Static Video. For 3D setting, we select 100

- 3D scans from THUman2.0 and render them into 3D static videos as our test set for comparison. We use the frontal view as the reference image. During inference, our method employs the same spatialtemporal sampling strategy as the multi-view setting. Quantitative comparisons are presented in Tab. 3. We mask out the backgrounds for a fair comparison since other methods tend to generate noisy backgrounds. Our method outperforms others, demonstrating the
- 4D diffusion transformer’s ability to learn physical 3D viewpoint changes. We have also conducted qualitative comparisons, with results shown in Fig. 9. Our method generates spatially consistent 360-degree videos compared to other approaches. Please refer to our submitted project webpage for 3D static video effects.

- 7.2.4 Comparisons on 360-degree Video. For 360-degree video evaluation, we select 10 4D scans from the Human4DiT dataset, each performing 5 different motions, and render them with different camera trajectories to create 50 test videos. We use the frontal view as the reference image. During inference, our method employs the same spatial-temporal sampling strategy as the multi-view setting. Quantitative comparisons are presented in Tab. 3, where we mask out the backgrounds for the evaluation. The results demonstrate our 4D diffusion transformer’s ability to handle both viewpoint and human motion changes in a 4D dynamic scenario. We have also conducted qualitative comparisons, with results shown in Fig. 10. Our method generates spatio-temporally consistent 360-degree dynamic videos compared to other approaches. Please refer to our submitted project webpage for 360-degree video effects.
- 7.2.5 Comparisons on model size. In addition to comparing video generation quality, we conducted a comprehensive analysis of model sizes across various approaches, as illustrated in Tab 4. We also train a smaller version of Human4DiT "Ours (small)" with 6 4D transformer blocks. Our Human4DiT model in its small setting exhibits comparable parameter counts to AnimateAnyone and Champ. However, due to the higher computational complexity of attention operations in the DiT architecture compared to convolutional operations, our model does not demonstrate an advantage in terms of inference speed. Nonetheless, as presented by our previous experimental comparisons, our model achieves superior performance within fewer parameters, thus validating the efficiency of our proposed Human4DiT architecture.

7.3 Ablation Study

7.3.1 Temporal Transformer. First, we validate our temporal transformer blocks and conduct an ablation study by removing the temporal transformers. The quantitative and qualitative results "Ours (image)" are presented in Tab. 3 and Fig. 6. Observations indicate

[Figure 6]

- Fig. 6. Ablation study of temporal transformer. When using only an image transformer without a temporal transformer, temporal consistency could not be guaranteed, resulting in artifacts such as discontinuous arm generation (see top images). After introducing the temporal transformer, our method can produce continuous and natural human motion (see bottom images).

[Figure 7]

- Fig. 7. Failure cases. Our method produces distortions and artifacts in cases where the facial region is small (see left images). Additionally, our approach struggles to generate natural finger structures (see right images).

that the incorporation of temporal transformers significantly enhances generation performance. Notably, even in the absence of these components, our method demonstrates comparable performance to CNN-based approaches.

- 7.3.2 View Transformer. To validate the effectiveness of introducing the fourth view dimension and the view transformer in our 4D transformer, we conduct ablation studies for multi-view video, 3D static video, and 360-degree video generation tasks. Specifically, we disable the view transformer and rely solely on the temporal transformer to generate the corresponding videos. Quantitative results of "Ours (image+temporal)" are presented in Tab. 3.It can be observed that the introduction of the view transformer leads to improvements in view-related generation tasks, demonstrating the efficacy of the view transformer.

- 7.3.3 Efficient Spatial-temporal Sampling. To validate the efficacy of our spatial-temporal sampling, we conducted a qualitative comparison, as illustrated in Fig 5. When only temporal window sampling is employed, the generated 360-degree video exhibits noticeable misalignments (The tie on the person’s chest has become a flat, texture surface). In contrast, our spatial-temporal sampling method mitigates these inconsistencies, achieving global coherence throughout the generated sequence. The results demonstrate that our method effectively addresses the challenges of long-range dependencies in 360-degree human video generation, contributing to enhanced visual quality and coherence.

7.4 Failure Cases

As illustrated in Fig. 7, our work primarily addresses full-body, 360degree human video generation. However, the face and hands of human, typically occupying less than 100x100 pixels within our 768x768 generated video frames, are leading to artifacts due to their relatively small scale. Furthermore, inaccuracies in SMPL estimation also affect the overall generation quality. We believe that future research incorporating specialized high-resolution modules for facial and hand generation could potentially address these issues.

Additionally, our current approach demonstrates limitations in effectively modeling background dynamics. This problem arises from the nature of our dataset, which predominantly features static backgrounds. Moreover, our 3D/4D datasets are rendered against a uniform white background. Consequently, the model has not learned to accurately represent background transformations across varying viewpoints.

### 8 DISCUSSION

Conclusion. We have presented a novel method for human video generation, which takes as input only a single image and produces spatio-temporally coherent video of dynamic human motions under 360-degree viewpoints. Our approach employs an efficient 4D transformer architecture to model the correlations across multiple domains, including view, time and poses. Combining with UNets for accurate condition injection, our model can be trained on a multidimensional dataset spanning images, videos, multi-view data, and 4D scans. After training, our method can synthesize 360-degree realistic, coherent human motion videos, and we believe our contributions will inspire future work towards 4D content generation.

Limitations and Future Work. Instead of generating an explicit 4D models, our method directly synthesizes 2D videos from the given viewpoints, and the 4D scene structure are implicitly encoded through the attention mechanism. The absence of an explicit 4D representation results in some artifacts when rendering 360degree videos. Furthermore, our current implementation cannot generate tiny structures coherently as mentioned in Sec. 7.4. Future works may incorporate a body part-aware generator to resolve this limitation. Exploring other condition injection techniques is also significant to enhance controllability. Potential directions include improved camera representations, Plücker embeddings, and direct utilization of 3D SMPL as tokens. We also believe collecting a more diverse dataset is significant to fully leverage the capabilities of DiT, and address video generation challenges in out-of-distribution (OOD) scenarios, such as anime or stylized content.

### 9 ETHICS STATEMENT

While our method for human video generation could benefit a lot of applications for entertainment, education, and accessibility, we recognize the risks of misuse for creating deepfakes, privacy violations, and perpetuating societal biases. We are committed to responsible development and promoting transparency in our methods. By discussing these ethical considerations, we aim to foster the positive applications of this technology while minimizing potential harm.

Acknowledgement The work is supported by the National Science Foundation of China under Grant Number 62125107.

REFERENCES

Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. 2023. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 22669–22679.

Ankan Kumar Bhunia, Salman Khan, Hisham Cholakkal, Rao Muhammad Anwer, Jorma Laaksonen, Mubarak Shah, and Fahad Shahbaz Khan. 2023. Person image synthesis via denoising diffusion model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5968–5976.

Michael J Black, Priyanka Patel, Joachim Tesch, and Jinlong Yang. 2023. Bedlam: A synthetic dataset of bodies exhibiting detailed lifelike animated motion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8726– 8737.

Wei Cheng, Ruixiang Chen, Siming Fan, Wanqi Yin, Keyu Chen, Zhongang Cai, Jingbo Wang, Yang Gao, Zhengming Yu, Zhengyu Lin, et al. 2023. Dna-rendering: A diverse neural actor repository for high-fidelity human-centric rendering. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 19982–19993.

Shubham Goel, Georgios Pavlakos, Jathushan Rajasegaran, Angjoo Kanazawa*, and Jitendra Malik*. 2023. Humans in 4D: Reconstructing and Tracking Humans with Transformers. In International Conference on Computer Vision (ICCV).

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative adversarial nets. Advances in neural information processing systems 27 (2014).

Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. 2024. CameraCtrl: Enabling Camera Control for Text-to-Video Generation. arXiv:2404.02101 [cs.CV]

Li Hu, Xin Gao, Peng Zhang, Ke Sun, Bang Zhang, and Liefeng Bo. 2023. Animate anyone: Consistent and controllable image-to-video synthesis for character animation. arXiv preprint arXiv:2311.17117 (2023).

Yasamin Jafarian and Hyun Soo Park. 2021. Learning high fidelity depths of dressed humans by watching social media dance videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 12753–12762.

Xuan Ju, Ailing Zeng, Jianan Wang, Qiang Xu, and Lei Zhang. 2023. Human-art: A versatile human-centric dataset bridging natural and artificial scenes. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 618– 629.

Johanna Karras, Aleksander Holynski, Ting-Chun Wang, and Ira KemelmacherShlizerman. 2023. Dreampose: Fashion video synthesis with stable diffusion. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 22680– 22690.

Sunmin Lee, Taeho Kang, Jungnam Park, Jehee Lee, and Jungdam Won. 2023. SAME: Skeleton-Agnostic Motion Embedding for Character Animation. In SIGGRAPH Asia 2023 Conference Papers. 1–11.

Jing Lin, Ailing Zeng, Shunlin Lu, Yuanhao Cai, Ruimao Zhang, Haoqian Wang, and Lei Zhang. 2024. Motion-x: A large-scale 3d expressive whole-body human motion dataset. Advances in Neural Information Processing Systems 36 (2024).

Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. 2023b. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 9298–9309.

Yuan Liu, Cheng Lin, Zijiao Zeng, Xiaoxiao Long, Lingjie Liu, Taku Komura, and Wenping Wang. 2023a. Syncdreamer: Generating multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453 (2023).

Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, et al. 2023. Wonder3d: Single image to 3d using cross-domain diffusion. arXiv preprint arXiv:2310.15008 (2023).

Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J Black. 2023. SMPL: A skinned multi-person linear model. In Seminal Graphics Papers: Pushing the Boundaries, Volume 2. 851–866.

Haoyu Lu, Guoxing Yang, Nanyi Fei, Yuqi Huo, Zhiwu Lu, Ping Luo, and Mingyu Ding.

2023. Vdt: General-purpose video diffusion transformers via mask modeling. In The Twelfth International Conference on Learning Representations.

Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. 2024. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048 (2024).

Mehdi Mirza and Simon Osindero. 2014. Conditional generative adversarial nets. arXiv preprint arXiv:1411.1784 (2014). OpenAI. 2024. Video generation models as world simulators. https://openai.com/index/ video-generation-models-as-world-simulators/. Accessed: 2024-05-19.

William Peebles and Saining Xie. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 4195– 4205.

Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. 2018. Improving language understanding by generative pre-training. (2018).

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog 1, 8

(2019), 9.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10684– 10695.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18. Springer, 234–241.

Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. 2023a. Zero123++: a single image to consistent multi-view diffusion base model. arXiv preprint arXiv:2310.15110 (2023).

Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. 2023b. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512

(2023).

Aliaksandr Siarohin, Stéphane Lathuilière, Enver Sangineto, and Nicu Sebe. 2019a. Appearance and pose-conditioned human image generation using deformable gans. IEEE transactions on pattern analysis and machine intelligence 43, 4 (2019), 1156– 1171.

Aliaksandr Siarohin, Stéphane Lathuilière, Sergey Tulyakov, Elisa Ricci, and Nicu Sebe. 2019b. First order motion model for image animation. Advances in neural information processing systems 32 (2019).

Aliaksandr Siarohin, Enver Sangineto, Stéphane Lathuiliere, and Nicu Sebe. 2018. Deformable gans for pose-based human image generation. In Proceedings of the IEEE conference on computer vision and pattern recognition. 3408–3416.

Robin Strudel, Ricardo Garcia, Ivan Laptev, and Cordelia Schmid. 2021. Segmenter: Transformer for semantic segmentation. In Proceedings of the IEEE/CVF international conference on computer vision. 7262–7272.

Yu Tian, Jian Ren, Menglei Chai, Kyle Olszewski, Xi Peng, Dimitris N Metaxas, and Sergey Tulyakov. 2021. A good image generator is what you need for high-resolution video synthesis. arXiv preprint arXiv:2104.15069 (2021).

Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. 2021. Training data-efficient image transformers & distillation through attention. In International conference on machine learning. PMLR, 10347–10357.

Shuhei Tsuchida, Satoru Fukayama, Masahiro Hamasaki, and Masataka Goto. 2019. AIST Dance Video Database: Multi-genre, Multi-dancer, and Multi-camera Database for Dance Information Processing. In Proceedings of the 20th International Society for Music Information Retrieval Conference, ISMIR 2019. Delft, Netherlands.

Twindom. 2022. Twindom 3D Avatar Dataset. https://web.twindom.com/ Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N

Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems 30 (2017).

- Tan Wang, Linjie Li, Kevin Lin, Chung-Ching Lin, Zhengyuan Yang, Hanwang Zhang, Zicheng Liu, and Lijuan Wang. 2023. Disco: Disentangled control for referring human dance generation in real world. arXiv e-prints (2023), arXiv–2307.

Ting-Chun Wang, Arun Mallya, and Ming-Yu Liu. 2021. One-shot free-view neural talking-head synthesis for video conferencing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10039–10049.

Yaohui Wang, Piotr Bilinski, Francois Bremond, and Antitza Dantcheva. 2020. G3AN: Disentangling appearance and motion for video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5264–5273.

Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M Alvarez, and Ping Luo. 2021. SegFormer: Simple and efficient design for semantic segmentation with transformers. Advances in neural information processing systems 34 (2021), 12077– 12090.

Zhongcong Xu, Jianfeng Zhang, Jun Hao Liew, Hanshu Yan, Jia-Wei Liu, Chenxu Zhang, Jiashi Feng, and Mike Zheng Shou. 2023. Magicanimate: Temporally consistent human image animation using diffusion model. arXiv preprint arXiv:2311.16498 (2023).

Shiyuan Yang, Liang Hou, Haibin Huang, Chongyang Ma, Pengfei Wan, Di Zhang, Xiaodong Chen, and Jing Liao. 2024. Direct-a-Video: Customized Video Generation with User-Directed Camera Movement and Object Motion. arXiv preprint arXiv:2402.03162 (2024).

Hongwei Yi, Hualin Liang, Yifei Liu, Qiong Cao, Yandong Wen, Timo Bolkart, Dacheng Tao, and Michael J Black. 2023. Generating Holistic 3D Human Motion from Speech. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 469–480.

- Tao Yu, Zerong Zheng, Kaiwen Guo, Pengpeng Liu, Qionghai Dai, and Yebin Liu. 2021. Function4D: Real-time Human Volumetric Capture from Very Sparse Consumer RGBD Sensors. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR2021).

Li Yuan, Yunpeng Chen, Tao Wang, Weihao Yu, Yujun Shi, Zi-Hang Jiang, Francis EH Tay, Jiashi Feng, and Shuicheng Yan. 2021. Tokens-to-token vit: Training vision transformers from scratch on imagenet. In Proceedings of the IEEE/CVF international conference on computer vision. 558–567.

Hongwen Zhang, Siyou Lin, Ruizhi Shao, Yuxiang Zhang, Zerong Zheng, Han Huang, Yandong Guo, and Yebin Liu. 2023a. Closet: Modeling clothed humans on continuous surface with explicit template decomposition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 501–511.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023b. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 3836–3847.

Yuxiang Zhang, Zhe Li, Liang An, Mengcheng Li, Tao Yu, and Yebin Liu. 2021. Lightweight Multi-person Total Capture Using Sparse Multi-view Cameras. In IEEE International Conference on Computer Vision.

Hongkai Zheng, Weili Nie, Arash Vahdat, and Anima Anandkumar. 2023. Fast training of diffusion models with masked transformers. arXiv preprint arXiv:2306.09305

(2023).

Sixiao Zheng, Jiachen Lu, Hengshuang Zhao, Xiatian Zhu, Zekun Luo, Yabiao Wang, Yanwei Fu, Jianfeng Feng, Tao Xiang, Philip HS Torr, et al. 2021. Rethinking semantic segmentation from a sequence-to-sequence perspective with transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 6881–6890.

Shenhao Zhu, Junming Leo Chen, Zuozhuo Dai, Yinghui Xu, Xun Cao, Yao Yao, Hao Zhu, and Siyu Zhu. 2024. Champ: Controllable and Consistent Human Image Animation with 3D Parametric Guidance. arXiv preprint arXiv:2403.14781 (2024).

[Figure 8]

###### Fig. 8. Qualitative comparison multi-view video.

[Figure 9]

###### Fig. 9. Qualitative comparison 3D static video.

[Figure 10]

###### Fig. 10. Qualitative comparison free-view video.

