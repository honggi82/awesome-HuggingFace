## LN3DIFF++: Scalable Latent Neural Fields Diffusion for Speedy 3D Generation

Yushi Lan, Student Member, IEEE, Fangzhou Hong, Shangchen Zhou, Shuai Yang, Member, IEEE, Xuyi Meng, Yongwei Chen, Zhaoyang Lyu, Bo Dai, Xingang Pan, Chen Change Loy , Senior Member, IEEE,

### arXiv:2403.12019v3[cs.CV]19Dec2025

Abstract—The field of neural rendering has seen remarkable progress, driven by advancements in generative models and differentiable rendering techniques. While 2D diffusion has achieved notable success, the development of a unified 3D diffusion pipeline remains an open challenge. This paper presents a novel framework, LN3DIFF++, designed to bridge this gap and facilitate fast, high-quality, and versatile conditional 3D generation. Our method leverages a 3D-aware architecture and a variational autoencoder (VAE) to encode input image(s) into a structured, compact 3D latent space. The latent representation is then decoded by a transformer-based decoder into a high-capacity 3D neural field. By training a diffusion model on this 3D-aware latent space, our method achieves superior performance for category-specific 3D generation on ShapeNet and FFHQ, as well as category-free image/text-conditioned 3D generation over Objaverse. Moreover, it surpasses existing 3D diffusion methods in inference speed, requiring no per-instance optimization. Video demos can be found on our project webpage: https://nirvanalan.github.io/projects/ln3diff.

Index Terms—Generative Model, 3D Reconstruction, Latent Diffusion Model

I. INTRODUCTION

# T

HE advancement of generative models [34], [44] and differentiable rendering [121] has paved the way for a

new research direction called neural rendering [121]. This field is continuously pushing the limits of view synthesis [82], editing [64], and particularly 3D object synthesis [7]. While

- 2D diffusion models [44], [114] have outperformed generative adversarial networks (GANs)-based methods in image synthesis [23] in quality [102], controllability [152], and scalability [106], a unified 3D diffusion pipeline has yet to be established.
- 3D object generation methods using diffusion models can

be categorized into 2D-lifting and feed-forward 3D diffusion models. In 2D-lifting methods, score distillation sampling (SDS) [97], [130] and Zero-123 [73], [109] achieve 3D generation by leveraging pre-trained 2D diffusion models. However, SDS-based methods require costly per-instance optimization and are prone to the multi-face Janus problem [97], where models struggle to maintain consistency and fidelity across different views of the same face. Meanwhile, Zero-123 fails to enforce strict view consistency. On the other hand, feedforward 3D diffusion models [11], [51], [66], [84], [129], [150] enable fast 3D synthesis without per-instance optimization. However, these methods typically involve a two-stage preprocessing approach. First, during the data preparation stage, a shared decoder is learned over a large number of instances to ensure a shared latent space. This is followed by per-

instance optimization to convert each 3D asset in the datasets into neural fields [138]. After this, the feed-forward diffusion model is trained on the prepared neural fields.

While the pipeline above is straightforward, it poses extra challenges to achieve high-quality 3D diffusion: 1) Scalability. In the data preparation stage, existing methods face scalability issues due to the use of a shared, low-capacity MLP decoder for per-instance optimization. This approach is data inefficient, requiring over 50 views per instance [11], [84] during training. Consequently, the computation cost scales linearly with the size of the dataset, hindering scalability for large, diverse 3D datasets. 2) Efficiency. Employing 3D-specific architectures [17], [98], [124] is computationally intensive and necessitates representation-specific designs [155]. Consequently, existing methods compress each 3D asset into neural fields [138] before training. However, this compression introduces highdimensional 3D latent, increasing computational demands and training challenges. Limiting the neural field size [11] might mitigate these issues but at the cost of reconstruction quality. In addition, the auto-decoding paradigm can result in an unclean latent space [35], [51], [111], unsuitable for 3D diffusion training [102]. 3) Generalizability. Existing 3D diffusion models primarily focus on unconditional generation over single classes, neglecting high-quality conditional 3D generation (e.g, text-to-3D) across generic category-free 3D datasets. Furthermore, projecting monocular input images into the diffusion latent space is crucial for conditional generation and editing [65], [152], but this is challenging with the shared decoder designed for multi-view inputs.

In this study, we propose a novel framework called Latent Neural fields 3D Diffusion (LN3DIFF++) to address these challenges and enable fast, high-quality and generic conditional 3D generation. Our method involves training a variational autoencoder [60] (VAE) to compress input images into a lower-dimensional 3D-aware latent space, which is more expressive and flexible compared to pixel-space diffusion [23], [44], [48], [114]. From this space, a 3D-aware transformerbased decoder gradually decodes the latent into a high-capacity 3D neural field. This autoencoding stage is trained amortized with differentiable rendering [121], incorporating novel view supervision for multi-view datasets [9], [22] and adversarial supervision for monocular datasets [53]. Thanks to the highcapacity model design, our method is more view efficient, requiring only two views per instance during training. After training, we leverage the learned 3D latent space for conditional 3D diffusion learning, ensuring effective utilization of the trained model for high-quality 3D generation. The pre-

[Figure 1]

[Figure 2]

JOURNAL OF LATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 2

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

LN3Diff++LN3Diff

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

A Space War Tie Fighter.

A Formula 1 racing car.

[Figure 21]

[Figure 22]

[Figure 23]

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

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

3D VAE Reconstruction

Conditional 3D Generation

- Fig. 1. We present LN3DIFF++, which performs efficient 3D diffusion learning over a compact latent space. Compared to LN3Diff which adopts NeRF rendering and supports text-conditioned 3D generation, LN3DIFF++ further enables SDF-based 3D representation and image-conditioned 3D generation. The resulting model enables both high-quality monocular 3D reconstruction and text-to-3D synthesis.

trained encoder can amortize the data encoding over incoming data, thus streamlining operations and facilitating efficient 3D diffusion learning while remaining compatible with advances in 3D representations.

To enhance efficient information flow in the 3D space and promote coherent geometry reconstruction, we introduce a novel 3D-aware architecture tailored for fast and high-quality 3D reconstruction while maintaining a structured latent space. Specifically, we employ a convolutional tokenizer to encode the input image(s) into a KL-regularized 3D latent space, leveraging its superior perceptual compression ability [30]. We employ transformers [24], [94] to enable flexible 3D-aware attention across 3D tokens in the latent space. Finally, we upsample the 3D latent and apply differentiable rendering for image-space supervision, making our method a self-supervised 3D learner [113].

An earlier version of this work appeared in Lan et al [63], which focuses on unconditional generation over ShapeNet and preliminary study over category-free text-conditioned 3D generation. Besides, U-Net and DDPM framework is adopted following the conventions [23]. To further improve the flexibility of 3D generation, this journal extension further explores the architecture of image-conditioned 3D generation based on the modern diffusion transformer (DiT) [94] architecture. Besides, the well-established flow matching (FM) [1], [71], [74] framework is incorporated for scalable diffusion training. Motivated by recent evidence that pairing FM with Diffusion Transformers (DiT) advances video generation [127], we re-

place the DDPM+U-Net paradigm with FM+DiT. This change improves both generation quality and training efficiency in our method and situates our 3D generative model within a unified image–video framework, potentially enabling joint reasoning and generation as demonstrated in state-of-the-art image diffusion models [12], [92]. Additionally, to enable high-quality mesh extraction, we fine-tune the NeRF-based 3D VAE with an SDF-based representation [108]. Considering that SDFs can be readily converged into meshes, this design choice broadens the utility of our model for more downstream tasks. To demonstrate the effectiveness of our extension, we conduct comprehensive experiments to evaluate the newly extended LN3DIFF++ framework, including qualitative and quantitative assessments of image-conditioned 3D generation, along with comparisons to recent competitive baselines.

In summary, we contribute a 3D-representation-agnostic pipeline for building generic, high-quality 3D generative models. This pipeline provides opportunities to resolve a series of downstream 3D vision and graphics tasks. Specifically, we propose a novel 3D-aware reconstruction model that achieves high-quality 3D data encoding in an amortized manner. Learning in the compact latent space, our model demonstrates state-of-the-art 3D generation performance on the ShapeNet benchmark [9], surpassing both GAN-based and 3D diffusionbased approaches. Our method shows superior performance in monocular 3D reconstruction and conditional 3D generation on ShapeNet, FFHQ, and Objaverse datasets, with a fast inference speed, i.e, 3× faster against existing latent-free 3D

diffusion methods [2].

II. RELATED WORK

3D-aware GANs. GANs [34] have shown promising results in generating photorealistic images [4], [55], [56], inspiring researchers to explore 3D-aware generation [41], [85], [91]. Motivated by the recent success of neural rendering [80], [82], [93], researchers have introduced 3D inductive bias into the generation task [6], [107], demonstrating impressive 3D-aware synthesis through hybrid designs [7], [36], [45], [87], [90]. This has made 3D-aware generation applicable to a series of downstream applications [64], [115], [116], [151]. However, GAN-based methods suffer from mode collapse [123] and struggle to model datasets with larger scale and diversity [23]. Besides, 3D reconstruction and editing with GANs require elaborately designed inversion algorithms [65].

3D Generation via 2D Diffusion Models. The remarkable success of 2D diffusion models [44], [114] has motivated their adaptation for 3D content generation. Early approaches, such as score distillation sampling [97], [130], attempt to distill 3D structures from 2D diffusion models. However, these methods face significant challenges, including computationally expensive optimization, mode collapse, and the Janus problem. Some methods propose learning the 3D prior in a

- 2D manner [8], [73], [75], [122]. While these can produce photorealistic results, they lack view consistency and fail to fully capture the 3D structure.

Recent advancements have shifted toward a two-stage pipeline: generating multi-view images [75], [109], [110], followed by feed-forward 3D reconstruction [47], [118], [141]. While these approaches have demonstrated promising results, their performance is inherently limited by the quality of multi-view image generation. Issues such as inconsistent viewpoints [73] and difficulties in scaling to higher resolutions [109] often hinder the overall effectiveness. Furthermore, the reliance on a two-stage pipeline restricts 3D editing capabilities due to the absence of a unified, 3D-aware latent space, posing challenges for flexible and intuitive manipulation of 3D content. To exploit rich pretrained 2D priors while ensuring geometric consistency, [70], [79] synthesize splatter-based

- 3D representations directly with 2D diffusion models, thereby achieving 3D-consistent generation in a single 2D diffusion stage. Native 3D Diffusion Models. Recently, native 3D diffusion models [14], [16], [63], [66], [68], [69], [135], [139], [147], [148], [150], [153] have been proposed to enable high-quality, efficient, and scalable 3D generation. These models follow a two-stage training process: first, encoding 3D objects into a latent space, and second, applying a latent diffusion model to the encoded latent codes.

While this pipeline is conceptually straightforward, existing methods vary in their choice of VAE input formats, latent space structures, and output 3D representations. Previous 3D diffusion pipeline adopts multi-view supervised auto-decoder as the stage-1 encoding tool [26], [51], [66], [84], [111], [129], [148]. Then, the encoded 3D latent codes serve as the training corpus for diffusion. However, the auto-decoding stage

leads to an unclean latent space and limited scalability [142]. Moreover, large latent codes, e.g, 256×256×96 [129], hinder efficient diffusion learning [48].

Another line of work incorporates rendering operation into 3D diffusion training. RenderDiffusion [2] and DMV3D [142] propose latent-free 3D diffusion by integrating rendering into diffusion sampling. However, this approach involves timeconsuming volume rendering at each denoising step, significantly slowing down sampling. SSDNeRF [11] suggests a joint 3D reconstruction and diffusion approach, but requires a complex training schedule and shows performance only in single-category unconditional generation. GaussianCube [148] proposes an optimal transport-driven 3DGS [59] encoding pipeline and leverages 3D-UNet for scalable 3D generation. However, it still leverages costly per-instance fitting before 3D diffusion training. 3DShape2Vec [150] proposed the first vecset-based latent 3D generative model, and has been demonstrated scalable by recent industry efforts [135]. However, these methods focus on 3D shape modeling and cannot generate colorful 3D mesh in a single stage. OctFusion [139] presents a native 3D diffusion framework leveraging the compact octree latent representation, achieving both highquality and efficient 3D generation. Nonetheless, its experiments are restricted to toy ShapeNet datasets, in contrast to our method, which enables category-free object generation. Likewise, prior SDF-based 3D generative models [16], [68] suffer from the same limitation, with no evidence of general 3D object generation. In contrast, our proposed LN3DIFF++ trains 3D diffusion in a compressed VAE [60], [61] latent space without rendering operations. As shown in Section IV, our method outperforms others in 3D generation and monocular 3D reconstruction, achieving three times faster speed. Additionally, we demonstrate conditional 3D generation over diverse datasets, whereas RenderDiffusion and SSDNeRF focus on simpler classes. Other approaches, like 3DGen [38] and VolumeDiffusion [119], perform diffusion in the 3D latent space but heavily rely on 3D data (e.g, point clouds and voxels) and do not support monocular datasets like FFHQ [53]. Moreover, their methods are designed for U-Net, whereas our DiT-based architecture offers greater scalability.

Generalized 3D Reconstruction and View Synthesis. To bypass the per-scene optimization of NeRF, researchers have proposed learning a prior model through image-based rendering [15], [47], [104], [128], [145]. However, these methods are primarily designed for view synthesis and lack explicit 3D representations. LoLNeRF [101] learns a prior through autodecoding but is limited to simple, category-specific settings. Moreover, these methods are intended for view synthesis and cannot generate new 3D objects. VQ3D [105] adapts the generalized reconstruction pipeline to 3D generative models. However, it uses a 2D architecture with autoregressive modeling over a 1D latent space, ignoring much of the inherent 3D structure. NeRF-VAE [61] directly models 3D likelihood with a VAE posterior but is constrained to simple 3D scenes due to the limited capacity of VAE. Concurrently, LRM-line of work [47], [118], [131] have proposed a feedforward framework for generalized monocular reconstruction. However, they are still regression-based models and lack the

Prompt: “A teddy bear”

ℒ + ℒ + ℒ + ℒ

|Reshape|
|---|

|z|
|---|
|z|
|z|

|Reshape|
|---|

LearnablePE

[Figure 39]

[Figure 40]

AdaLN

AdaLN

DiTBlock

Reshape

DiTBlock

Reshape

MHA

FFN

⊙

⊕

⊕ ⊙

z

| | |
|---|---|
| | |
| | |
| | |
|×T| |

[Figure 41]

Q KV

𝛾 ,𝛽

𝛾 ,𝛽

[Figure 42]

[Figure 43]

[Figure 44]

U-Net/DiT-3D

[Figure 45]

z , , 

𝛼

𝛼

z

MLP

Q KV

Self-plane Attention Cross-plane Attention

DiT Block

Source & Novel View

×N

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Rendering

Q KV

|Reshape|
|---|

z , , 

Q KV

[Figure 50]

Encoder

Transformer

Upsampler

Reshape

z

ℰ

𝒟 z

𝒟

3D Latent

[Figure 51]

𝒟 + 𝒟

Tri-plane

[Figure 52]

[Figure 53]

[Figure 54]

Stage I: 3D Latent Space Learning Stage II: Latent Diffusion

- Fig. 2. Pipeline of LN3DIFF++. In the 3D latent space learning stage, a convolutional encoder Eϕ encodes a set of images I into the KL-regularized latent space. The encoded 3D latent is further decoded by a 3D-aware DiT transformer DT , in which we perform self-plane attention and cross-plane attention. The transformer-decoded latent is up-sampled by a convolutional upsampler DU towards a high-res tri-plane for rendering supervisions. In the next stage, we perform conditional diffusion learning over the compact latent space using either U-Net or DiT. The detailed architecture of DiT is shown in Fig. 3.

Inspired by previous work [30], [102], we propose to take multi-view image(s) as a proxy of the underlying 3D scene and compress the input image(s) into a compact 3D latent space. Though this paradigm is well-adopted in the image domain [30], [102] with similar trials in specific 3D tasks [5], [65], [81], [105], we, for the first time, demonstrate that a high-quality compression model is feasible, whose latent space serves as a compact proxy for efficient diffusion learning.

latent space designed for generative modeling and 3D editing. Besides, they are limited to 3D reconstruction only and fail to support other modalities.

III. SCALABLE LATENT NEURAL FIELDS DIFFUSION

This section introduces our latent 3D diffusion model, which learns efficient diffusion prior over the compressed latent space by a dedicated variational autoencoder. Specifically, the training goal is to learn a variational encoder Eϕ that maps a set of posed 2D image(s) I = {Ii,...,IV }, of a 3D object to a latent code z, a denoiser ϵθ(zt,t) to denoise the noisy latent code zt given diffusion time step t, and a decoder Dψ (including a Transformer DT and an Upsampler DU) to map z0 to the 3D tri-plane X corresponding to the input object.

Encoder. Given a set of image(s) I of an 3D object where each image within the set I ∈ RH×W×3 is an observation of an underlying 3D object from viewpoints C = {c1,...,cV }, LN3DIFF++ adopts a convolutional encoder Eϕ to encode the image set I into a latent representation z ∼ Eϕ(I). To inject camera condition, we concatenate Plucker coordinates ri = (di,pi × di) ∈ R6 channel-wise as part of the input [112], where di is the normalized ray direction, pi is the camera origin corresponding to the camera ci, and × denotes the cross product. For challenging datasets like Objaverse [22], we also concatenate the rendered depth map, making our input a dense 3D colored point cloud [133].

Such a design offers several advantages: 1) By explicitly separating the 3D data compression and diffusion stage, we avoid the representation-specific 3D diffusion design [2], [51], [84], [111], [155] and achieve a 3D representation/renderingagnostic diffusion, which can be applied to any neural rendering techniques. 2) By leaving the high-dimensional 3D space, we reuse the well-studied Latent Diffusion Model (LDM) architecture [94], [102], [126] for computationally efficient learning and achieve better sampling performance with faster speed. 3) The trained 3D compression model in the first stage serves as an efficient and general-purpose 3D tokenizer, whose latent space can be easily reused over downstream applications or extended to new datasets [21], [146].

Unlike existing works [30], [105] that operate on 1D order latent and ignore the internal structure, we choose to output 3D latent z ∈ Rh×w×d×c to facilitate 3D-aware operations, where h = H/f,w = W/f are the spatial resolution with downsample factor f, and d denotes the 3D dimension. Here we set f = 8 and d = 3 to make z ∈ Rh×w×3×c a tri-latent, which is similar to tri-plane [7], [95] but in the compact 3D latent space. We further impose KL-reg [60] to encourage a wellstructured latent space to facilitate diffusion training [102], [126].

In the following subsections, we first discuss the compressive stage with a detailed framework design in Sec. III-A. Based on that, we introduce the 3D diffusion generation stage in Sec. III-B and present the condition injection in Sec. III-C. The method overview is shown in Fig. 2.

Decoder Transformer. The decoder aims to decode the compact 3D codes z for high-quality 3D reconstruction. Existing image-to-3D methods [5], [7], [35], [65] adopt convolution as the building block, which lacks 3D-aware operations and impedes information flow in the 3D space. Here, we adopt ViT [24], [94] as the decoder backbone due to its flexibility and effectiveness. Inspired by Rodin [129], we made the

A. Perceptual 3D Latent Compression

As analyzed in Sec. I, directly leveraging neural fields for diffusion training hinders model scalability and performance.

following reformulation to the raw ViT decoder to encourage 3D inductive bias and avoid the mixing of uncorrelated 3D features: 1) Self-plane Attention Block. Given the input z ∈ Rl×3×c where l = h × w is the sequence length, we treat each of the three latent planes as a data point and conduct self-attention within itself. This operation is efficient and encourages local feature aggregation. 2) Cross-plane Attention Block. To further encourage 3D inductive bias, we roll out z

- as a long sequence l × 3 × c → 3l × c to conduct attention across planes, so that all tokens in the latent space could attend to each other. In this way, we encourage global information flow for more coherent 3D reconstruction and generation. Compared to Rodin, our design is fully attention-based and naturally supports parallel computing without the expensive axis pooling aggregation.

Empirically, we observe that using DiT [94] block and injecting the latent z as conditions yields better performance compared to the ViT [24], [89] block, which takes the latent z0 as the regular input. Specifically, the adaptive layer norm (adaLN) layer [94] fuses the input latent z with the learnable positional encoding for attention operations. Moreover, we interleave the two types of attention layers to make sure the overall parameters count consistent with the pre-defined DiT length, ensuring efficient training and inference. As all operations are defined in the token space, the decoder achieves efficient computation against Rodin [129] while promoting 3D priors.

Decoder Upsampler. After all the attention operations, we obtain the tokens from the last transformer layer z as the output. The context-rich tokens are reshaped back into spatial domain [39] and up-sampled by a convolutional decoder to the final tri-plane representation with shape Hˆ × Wˆ × 3C. Here, we adopt a lighter version of the convolutional decoder for efficient upsampling, where the three spatial latent of z are processed in parallel.

Learning a Perceptually Rich and Intact 3D Latent Space. Adversarial learning [34] has been widely applied in learning a compact and perceptually rich latent space [30], [102]. In the 3D domain, the adversarial loss can also encourage correct 3D geometry when novel-view reconstruction supervisions are inapplicable [6], [57], [105], e.g, the monocular dataset such as FFHQ. Inspired by previous research [57], [105], we leverage adversarial loss to bypass this issue. Specifically, we impose an input-view discriminator to maintain perceptuallyreasonable input view reconstruction, and an auxiliary novelview discriminator to distinguish the rendered images between the input and novel views. We observe that if asking the novelview discriminator to differentiate novel-view renderings and real images instead, the reconstruction model will suffer from posterior collapse [76], which outputs input-irrelevant but high-fidelity results to fool the novel-view discriminator. This phenomenon has also been observed by Kato et al [57].

Training. After the decoder Dψ decodes a high-resolution neural field zˆ0 from the latent, we have Iˆ = R( X) = R(Dψ(z)) = R(Dψ(Eϕ(I))), where R stands for differentiable rendering [121] and we take Dψ(z) = R(Dψ(z)) for brevity. Here, we choose X as tri-plane [7], [95] and R as volume rendering [82] for experiments. Note that our com-

pression model is 3D representations/rendering agnostic and new neural rendering techniques [59] can be easily integrated by alternating the decoder architecture [117]. The final training objective reads as

L(ϕ,ψ) = Lrender + λgeoLgeo + λklLKL + λGANLGAN, (1) where Lrender is a mixture of L1 and perceptual loss [154],

Lreg encourages smooth geometry [132], LKL is the loss KLreg to regularize a structured latent space [102], and LGAN improves perceptual quality and enforces correct geometry for monocular datasets. Note that Lrender is applied to both inputview and randomly sampled novel-view images.

To facilitate 3D mesh extraction, we further finetune the model to the hybrid representation Flexicubes [108], [141] with the extra Lflex loss:

Lflex = λnormalLnormal + λregLreg, (2)

where Lnormal is MAE loss between rendered normal and ground truth, Lreg is regularization term for Flexicubes parameters [108]. The corresponding loss weights are represented by λ*. Similar to LATTE3D [137], we only fine-tune the decoder of the 3D VAE at this stage for stabilized training.

For category-specific datasets such as ShapeNet [9], we only supervise one novel view, which already yields good enough performance. For category-free datasets with diverse shape variations, e.g, Objaverse [22], we supervise four novel views. Our method is more data-efficient against the existing stateof-the-art 3D diffusion method [11], [84], which requires 50 views to converge. The implementation details are included in the supplementary material.

B. Latent Diffusion and Denoising

Latent Diffusion Models. LDM [102], [126] is designed to acquire a prior distribution pθ(z0) within the perceptual latent space, whose training data is the latent obtained online from the trained Eϕ. Here, we use the score-based latent diffusion model [126], which is the continuous derivation of DDPM variational objective [44]. Specifically, the denoiser ϵθ parameterizes the score function score [114] as ∇zt

log p(zt) := −ϵθ(zt,t)/σt, with continuous time sequence t. By training to predict a denoised variant of the noisy input zt, ϵθ gradually learns to denoise from a standard Normal prior N(0,I) by solving a reverse SDE [44].

Following LSGM [126], we formulate the learned prior at time t geometric mixing ϵθ(zt,t) := σt(1 − α) ⊙ zt + α ⊙ ϵ′θ(zt,t), where ϵ′θ(zt,t) is the denoiser output and α ∈ [0,1] is a learnable scalar coefficient. Intuitively, this formulation can bring the denoiser input closer to a standard Normal distribution, on which the reverse SDE can be solved faster. Similarly, Stable Diffusion [96], [102] also scales the input latent by a factor to maintain a unit variance, which is pre-calculated on the billion-level dataset [106]. The training objective reads as

wt 2 ∥ϵ − ϵθ(zt,t,c)∥22 , (3)

Ldiff = EE

ϕ(I),ϵ∼N(0,1),t

where t ∼ U[0,1] and wt is an empirical time-dependent weighting function, c is the corresponding condition.

denotes its derivative. By setting wt = 1−t t with zt = (1−t)x0 +tϵ, flow matching defines the forward process as a straight path between the data distribution and the Normal distribution. The network ϵθ directly predicts the velocity vΘ, and please check the Sec.2 of SD-3 [29] for theoretical derivation.

#### ×N

⊕

𝛼

#### Text

Scale

FFN

𝛾 , 𝛽 Scale, Shift

After training, the denoised samples will be decoded to the 3D neural field (i.e, tri-plane here) with a single forward pass through Dψ, on which neural rendering can be applied.

CLIP RMSNorm

⊕

Image/Text Encoder

Multi-Head Cross Attention

C. Conditioning Mechanisms

(w/ QK-Norm)

Compared to LRM [47] line of work, which intrinsically relies on image(s) as the input, our native diffusion-based method enables more flexible 3D generation from diverse conditions. As shown in Fig. 2 and Fig. 3, we propose to inject CLIP embeddings [100] and DINO embeddings [89] into the latent 3D diffusion model to support image/text-conditioned 3D generation. Given the input condition y, the diffusion model formulates the conditional distribution p(z|y) with ϵθ(zt,t,y). The inputs y can be text captions for datasets like Objaverse, or images for general 3D datasets like ShapeNet, FFHQ, and Objaverse.

⊕

𝛼

Scale

Image Encoder

Multi-Head Self Attention

(w/ QK-Norm)

𝛾 , 𝛽

Scale, Shift

DINO

RMS Norm

MLP

(shared)

#### Image

Time t

Text Conditioning. For datasets with text captions, we follow Stable Diffusion [102] to directly leverage the CLIP text encoder CLIPT to encode the text caption as conditions. All output tokens 77 × 768 are used and injected into the diffusion denoiser with cross attention blocks. For both U-Net architecture and DiT architecture, the CLIP text conditions are injected via cross attention.

DiT Block

- Fig. 3. Diffusion training of LN3DIFF++. We adopt DiT architecture with AdaLN-single [13] and QK-Norm [20], [30]. For both conditioning modalities, we incorporate the conditional features using attention mechanisms. Specifically, for CLIP-based conditioning, we employ cross-attention blocks to inject the condition, following the approach used in PixelArt [13]. For image-conditioned 3D generation, we additionally concatenate DINO patch features into the self-attention block.

Image Conditioning. To support more flexible 3D content creation, LN3DIFF++ further supports image conditions.. Specifically, for category-specific 3D dataset, we first encode the input I corresponding to the latent code z0 using the CLIP image encoder CLIPI and adopt the output embedding as the condition. To support both image and text conditions, we re-scale the image latent code with a pre-calculated factor to match the scale of the text latent. Cross attention is also leveraged to inject CLIP image features.

The denoiser ϵθ is realized by a time-dependent UNet [103], as visualized in Fig. 2. During training, we obtain z0 online from the fixed Eϕ, roll-out the tri-latent h × w × 3 × c → h × (3w) × c, and add time-dependent noise to get zt. Here, we choose the importance sampling schedule [126] with velocity [78] parameterization, which yields more stable behavior against ϵ parameterization for diffusion learning.

For model trained on category-free 3D dataset like Objaverse [22], we further incorporate DINO [89] features to improve the reconstruction fidelity. Rather than introducing another cross-attention module, we incorporate the DINO features by pre-pending the patch tokens in the self-attention layers, similar to SD-3 [29]. Compared to leveraging CLIP features for image conditions, introducing low-level DINO features improves the 3D generation faithfulness and fidelity in image-conditioned setting.

3D Flow Matching. Beyond the vanilla LSGM framework, we also explore the flow matching [1], [71], [74]-based diffusion framework. Specifically, flow matching involves training a neural network ϵθ to predict the velocity v of the noisy input zt with the straight-line trajectory. After training, ϵθ can sample from a standard Normal prior N(0,I) by solving the reverse ODE/SDE [54]. In our case, the training data point is the compact tri-plane latent code. Note that compared to U-Net architecture that adopts roll-out tri-latent, in flow matching training, we opt for DiT [94] with full attention [29] over L = h × w × 3, as detailed in Fig. 3. The training objective now reads as

Classifier-free Guidance. We adopt classifier-free guidance [43] for latent conditioning to support conditional and unconditional generation. During diffusion model training, we randomly zero out the corresponding conditioning latent embeddings with 15% probability to jointly train unconditional and conditional settings. During sampling, we perform a linear combination of the conditional and unconditional score estimates:

- 1

- 2

ϕ(I),ϵ∼N(0,I),t wtFMλ′t∥ϵ − ϵθ(zt,t,c)∥22 ,

EE

LFM = −

(4) where λt := log a

##### ϵˆθ(zt,τθ(y)) = sϵθ(zt,τθ(y)) + (1 − s)ϵθ(zt), (5)

2 t

b2t denotes signal-to-noise ratio, and λ′t

ShapeNet Car ShapeNet Chair ShapeNet Plane

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

(Uncond) 3DS2V

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

SSDNeRFGET3DEG3DDiffRFRenderDiff Ours

- Fig. 4. ShapeNet Unconditional Generation. We show four samples for each method. Zoom in for the best view.

where s is the guidance scale to control the mixing strength to balance sampling diversity and quality.

Stable and Efficient Training. For efficient and scalable training, we leverage BFloat16 [52] with FlashAttention [18], [19] enabled. All conditioning transformer blocks follow a pre-norm design [140] with QK-Norm [20] enabled. RMSNorm [149] is leveraged for efficient AdaLN operation defined in diffusion transformer, following SD-3 [29].

IV. EXPERIMENTS

Datasets. Following most previous work, we use ShapeNet [9] to benchmark 3D generation performance. We use the Car, Chair, and Plane categories with 3514, 6700, and 4045 instances, respectively. Each instance is randomly rendered from 50 views following a spherical uniform distribution. Moreover, to evaluate the performance over diverse, high-quality 3D datasets, we also include experiments on Objaverse [22], which is the largest 3D dataset with challenging categories and complicated geometry. We use the renderings provided by G-

ShapeNet Car ShapeNet Chair ShapeNet Plane

[Figure 80]

Ours(TextCond)Ours(ImgCond)

A green/grey Porsche 911.

A brown wooden chair.

A sofa chair with soft pad.

A star war Tie Fighter.

A SUV trunk.

A Boeing 747.

[Figure 81]

Cond Image

Cond Image

Cond Image

- Fig. 5. ShapeNet Conditional Generation. We show conditional generation with both texts and image as inputs. Zoom in for the best view.

state-of-the-art GAN-based methods: EG3D [7], GET3D [33] as well as recent diffusion-based 3D generative methods: DiffRF [84], RenderDiffusion [2] and SSDNeRF [11] as baselines. We also include a canonical 3D shape generation method 3DShape2Vec [150] for geometry qualitative comparison. Since LN3DIFF++ only leverages V = 2 for ShapeNet experiments, for SSDNeRF, we include both the official 50views trained SSDNeRFV=50 version as well as the reproduced SSDNeRFV=3 for fair comparison. We find SSDNeRF fails to converge with V = 2. We set the guidance scale in Eq. (5) to s = 0 for unconditional generation, and s = 6.5 for all conditional generation sampling.

Objaverse [99] and choose a high-quality subset with around 176K 3D instances, where each consists of 40 random views. Text prompts from Cap3D [77] are used for text-conditioned 3D generation training.

Training Details. For VAE training with ShapeNet and FFHQ, we adopt a monocular input setting with V = 1 and target rendering size 128 × 128. For Objaverse, we adopt V = 6 posed RGB-D renderings with 256 × 256 resolution as inputs to guarantee a thorough coverage of the 3D object. The target rendering size is set to 192 × 192. The encoder Eϕ has down-sample factor f = 8 and the decoder upsampler DU outputs tri-plane with size Hˆ = Wˆ = 128 and C = 32. To trade off rendering resolution and training batch size, we impose supervision over 80 × 80 randomly cropped patches. For adversarial loss, we use DINOv2 [89] in vision-aided GAN [62] with non-saturating GAN loss [37] for discriminator training. For Flexicube fine-tuning, we directly render the whole image with resolution 256 × 256. We follow the same initialization strategy in InstantMesh [141], where the weights of σ prediction MLP are flipped to accommodate for SDF prediction.

Regarding the metrics, following prior work [11], we adopt both 2D and 3D metrics to benchmark the generation performance: Fr´echet Inception Distance (FID@50K) [42] and Kernel Inception Distance (KID@50K) [3] to evaluate 2D renderings, as well as Coverage Score (COV) and Minimum Matching Distance (MMD) to benchmark 3D geometry. We compute all metrics at 128 × 128 resolution to ensure fair comparisons across all baselines.

Evaluating Text-to-3D Generation. Regarding textconditioned 3D generation methods, we compare against Point-E [86], Shape-E [51], and 3DTopia [46]. Moreover, we also include the comparison with the latest 3DGS-based text-to-3D generative model, GaussianCube V1.1 [148] for reference. Note that GaussianCube V1.1 is trained on auxiliary data beyond Objaverse and adopts 3D captions from proprietary language model [88], rather than Cap3D. The result is included here for reference, and we emphasize that the comparison is indeed unfair. CLIP score [100] is reported following the previous works [51], with aesthetic scores MUSIQ-AVA [58] and Q-Align [134] also included.

On the generation side, for conditional diffusion training, we use CLIP image embeddings for ShapeNet and FFHQ, and CLIP text embeddings from the official text caption for Objaverse. For image-conditioned training, we randomly select an image from the dataset corresponding to each 3D instance as the conditioning input. The DiT-based 3D generative model uses the DiT-L architecture [94], which consists of 24 transformer layers with 16 attention heads and 1024 latent dimensions. Both the autoencoding model and the diffusion model are trained for 800K iterations, which take around 7 days with 8 A100 GPUs in total.

Evaluating Image-to-3D Generation. Regarding imageconditioned 3D generation methods, we compare the proposed method with three lines of methods: single-image to 3D methods: OpenLRM [40], [47], Splatter Image [117], multi-

- A. Metrics and Baselines

Evaluating Unconditional Generation. For unconditional 3D generation, we adopt category-specific ShapeNet [9] and adopt

view images to 3D methods: One-2-3-45 [72], CRM [131], Lara [10], LGM [118], and native 3D diffusion models: ShapeE [51].

TABLE I QUANTITATIVE METRICS ON TEXT-TO-3D. THE PROPOSED METHOD OUTPERFORMS POINT-E AND SHAPE-E ON CLIP SCORES OVER TWO DIFFERENT BACKBONES.

Method ViT-B/32↑ ViT-L/14↑ MUSIQ-AVA ↑ Q-Align ↑

Point-E 26.35 21.40 4.08 1.21 Shape-E 27.84 25.84 3.69 1.56 3DTopia 28.10 26.31 3.31 1.42 GaussianCube-V1.1 29.84 27.36 4.89 2.86

Ours 29.12 27.80 4.16 2.22

Quantitatively, we benchmark rendering metrics with CLIPI [100], FID [42], KID [3], and MUSIQ-koniq [58]. For 3D quality metrics, we adopt Point cloud FID (P-FID), Point cloud KID (P-KID), Coverage Score (COV), and Minimum Matching Distance (MMD) as the metrics. Following previous works [86], [144], [150], we adopt the pre-trained PointNet++ provided by Point-E [86] to calculate P-FID and K-FID. Qualitatively, GSO [25] dataset is used for visually inspecting image-conditioned generation.

- B. Evaluation

Unconditional Generation on ShapeNet. To evaluate our methods against existing 3D generation methods, we include the quantitative and qualitative results for unconditional singlecategory 3D generation on ShapeNet in Tab. II and Fig. 4. We evaluate all baseline 3D diffusion methods with 250 DDIM steps and GAN-based baselines with psi = 0.8 to guarantee each sample is intact for COV/MMD evaluation. For FID/KID evaluation, we re-train the baselines and calculate the metrics using a fixed upper-sphere ellipsoid camera trajectory [113] across all datasets. For COV/MMD evaluation, we randomly sample 4096 points around the extracted sampled mesh and ground truth mesh surface and adopt Chamfer Distance for evaluation.

As shown in Tab. II, LN3DIFF++ achieves quantitatively better performance against all GAN-based baselines regarding rendering quality and 3D coverage. Fig. 4 further demonstrates that GAN-based methods suffer greatly from mode collapse: in the ShapeNet Plane category, both EG3D and GET3D are limited to the white civil airplanes, which is fairly common in the dataset. Our methods can sample more diverse results with high-fidelity texture.

Compared to diffusion-based baselines, LN3DIFF++ delivers higher visual quality and stronger quantitative performance.. SSDNeRFV=50 shows a better coverage score, which benefits from leveraging more views during training. However, our method with V = 2 shows comparative performance against SSDNeRFV=3 on the ShapeNet Chair and even better performance on the remaining datasets.

Text-to-3D Generation. Conditional 3D generation has the potential to streamline the 3D modeling process in both the gaming and film industries. As visualized in Fig. 5, we present our conditional generation performance on the ShapeNet

dataset, where either text or image serves as the input prompt. Visually inspected, our method demonstrates promising performance in conditional generation, closely aligning the generated outputs with the input conditions. For image-conditioned generation, our method yields semantically similar samples while maintaining diversity.

For category-free text-conditioned 3D generation, we include its qualitative evaluation against state-of-the-art generic 3D generative models in Fig. 6, along with the quantitative benchmark in Tab. I. As shown, thanks to the compact 3D VAE latent space and DiT-based scalable diffusion model, our method yields better quality compared to previous diffusionbased baselines (Point-E, Shape-E, and 3DTopia) even trained with less computation resources. GaussianCube V1.1 yields sharper texture and higher aesthetic score due to the the efficient and high-resolution 3DGS rendering. Please note that GaussianCube V1.1 is trained on proprietary data and is included here for reference. More visual results of our textconditioned LN3DIFF++ is included in the Supp.

We also compare our method against RenderDiffusion, the only 3D diffusion method that supports 3D generation over FFHQ. As shown in the lower part of Fig. 7, beyond view-consistent 3D generation, our method further supports conditional 3D generation at 128 × 128 resolution, while RenderDiffusion is limited to 64 × 64 resolution due to the expensive volume rendering integrated into diffusion training. Quantitatively, our method achieves an FID score of 36.6, compared to 59.3 by RenderDiffusion.

Image-to-3D Generation. On the category-specific setting, beyond the samples shown in Fig. 1, we include monocular reconstruction results over FFHQ datasets in the upper half of Fig. 7 and compare against RenderDiffusion. As can be observed, our method demonstrates high fidelity and preserves semantic details even in self-occluded images. The novel view generated by RenderDiffusion appears blurry and misses semantic components that are not visible in the input view, such as the leg of the eyeglass.

On the category-free setting, our proposed framework enables 3D generation given single-view image conditions, leveraging the DiT architecture detailed in Fig. 3 (b). Following Tang et al [118], we qualitatively benchmark our method in Fig. 8 over the single-view 3D reconstruction task on unseen images from the GSO dataset. Our proposed framework is robust to inputs with complicated structures (rows 1, 3, 5) and self-occlusion (row 2), yielding consistently intact 3D reconstruction. Besides, our generative-based method shows a more natural back-view reconstruction, as opposed to regressionbased methods that are commonly blurry in uncertain areas.

Quantitatively, we show the evaluation in Tab. III. As can be seen, our proposed method achieves state-of-the-art performance over CLIP-I and all 3D metrics, with competitive results over conventional 2D rendering metrics FID/KID. Note that LGM leverages pre-trained MVDream [110] as the first-stage generation and then maps the generated four views to pixelaligned 3D Gaussians. This cascaded pipeline achieves better visual quality but is prone to yield distorted 3D geometry, as visualized in Fig. 8.

TABLE II QUANTITATIVE COMPARISON OF UNCONDITIONAL GENERATION ON SHAPENET. THE PROPOSED LN3DIFF++ SHOWS SATISFACTORY PERFORMANCE ON SINGLE-CATEGORY GENERATION.

Category Method FID@50K↓ KID@50K(%)↓ COV(%)↑ MMD(‰)↓

EG3D [7] 33.33 1.4 35.32 3.95 GET3D [33] 41.41 1.8 37.78 3.87 DiffRF [84] 75.09 5.1 29.01 4.52

Car

RenderDiffusion [2] 46.5 4.1 - -

SSDNeRFV=3 [11] 47.72 2.8 37.84 3.46 SSDNeRF∗

V=50 [11] 45.37 2.1 67.82 2.50 LN3DIFF++ (Ours) 17.6 0.49 43.12 2.32

EG3D [7] 14.47 0.54 18.12 4.50 GET3D [33] 26.80 1.7 21.30 4.06 DiffRF [84] 101.79 6.5 37.57 3.99

Plane

RenderDiffusion [2] 43.5 5.9 - -

SSDNeRFV=3 [11] 21.01 1.0 42.50 2.94 LN3DIFF++ (Ours) 8.84 0.36 43.40 2.71

EG3D [7] 26.09 1.1 19.17 10.31 GET3D [33] 35.33 1.5 28.07 9.10 DiffRF [84] 99.37 4.9 17.05 14.97

Chair

RenderDiffusion [2] 53.3 6.4 - -

SSDNeRFV=3 [11] 65.04 3.0 47.54 6.71 LN3DIFF++ (Ours) 16.9 0.47 47.1 5.28

mance. Though RenderDiffusion follows a latent-free design, its intermediate 3D neural field has a shape of 2562 × 96 and hinders efficient diffusion training.

- C. Ablation Study and Analysis

Reconstruction Arch Design. In Tab. IV, we benchmark each component of our auto-encoding architecture over a subset of Objaverse with 7K instances and record the PSNR

DINO Features for Image-conditioned 3D Generation. As shown in Fig. 10, we ablate the DINO features in imageconditioned 3D generation. As can be seen, using CLIP features only as the condition leads to unfaithful 3D generation, where the generated asset does not correctly reflect the input condition. Introducing low-level DINO features effectively resolves this issue. Note that we adopt the prepend-based image conditioning mechanism in Fig.3, motivated by its proven effectiveness in SD-3 [29]. While our follow-up work GaussianAnything [67] shows that cross-attention–based conditioning offers comparable performance with lower VRAM cost, we prioritize the prepend-based design here for its

- at 100K iterations. Each component introduces consistent improvements with negligible parameter increases. Novel View Discriminator for Monocular Dataset. Novel view discriminator is crucial for monocular datasets like FFHQ. As shown in Fig. 9, without it, the VAE model fails to yield a plausible novel view. Diffusion Sampling Speed and Latent Size. We report the sampling speed and latent space size comparison in Tab. V. By performing on the compact latent space, our method achieves the fastest sampling while keeping the best generation perfor-

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

A voxelized dog.

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

|N/A|
|---|

|N/A|
|---|

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

An 18th century cannon.

[Figure 114]

|N/A|
|---|

|N/A|
|---|

[Figure 115]

[Figure 116]

[Figure 117]

Ours Point-E Shap-E

3DTopia

GaussianCube V1.1

- Fig. 6. Qualitative Comparison of Text-to-3D We showcase uncurated samples generated by LN3DIFF++ on ShapeNet three categories. We visualize two views for each sample. Better zoom in.

[Figure 118]

JOURNAL OF LATEX CLASS FILES, VOL. 18, NO. 9, SEPTEMBER 2020 11

RenderDiffusion Ours

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

NovelViewGenerationInput

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

An old man with wrinkles on his face, white short hair.

A young boy with bangs and smiling face.

A middle-aged woman with brown hair, wearing glasses.

Unconditional Generation (RenderDiffusion)

- Fig. 7. FFHQ Monocular Reconstruction (upper half) and 3D Generation (lower half). For monocular reconstruction, we test our method with hold-out test set and visualize the input-view and novel-view. Compared to baseline, our method shows consistently better performance on both reconstruction and generation.

simplicity and demonstrated reliability.

V. CONCLUSION AND DISCUSSIONS

In this work, we introduce a new paradigm for 3D generative modeling by training a diffusion model over a compact, 3D-aware latent space.. A dedicated variational autoencoder encodes (multi-view) image(s) into a low-dimensional structured latent space, where conditional diffusion learning can be efficiently performed. We achieve state-of-the-art performance over ShapeNet and demonstrate our method over generic category-free Objaverse 3D datasets. Our work can facilitate numerous downstream applications in 3D vision and graphics tasks.

Limitations and Future Work. Our method still has several unresolved limitations. From the VAE perspective, we observe that volume rendering remains memory-intensive. In addition, the visual quality of the fine-tuned Flexicube variant is inferior to that of the NeRF-based version. Extending our decoder to more efficient 3D representations, such as

- 2DGS [49], may enable faster rendering while preserving high-quality 3D surfaces. Moreover, our triplane-structured 3D VAE latent space may not be the optimal choice for learning
- 3D diffusion. More explicit 3D representations, such as point clouds or sparse voxels [17], could also serve as effective latent proxies for 3D VAE training. Besides, directly incorporating 3D supervision, such as SDF or occupancy signals, may further improve geometric reconstruction performance.

TABLE III QUANTITATIVE EVALUATION OF IMAGE-CONDITIONED 3D GENERATION. WE EVALUATE THE QUALITY OF BOTH 2D RENDERINGS AND 3D SHAPES. THE PROPOSED METHOD DEMONSTRATES STRONG PERFORMANCE ACROSS ALL METRICS. ALTHOUGH MULTI-VIEW IMAGES-TO-3D APPROACHES LIKE LGM ACHIEVE BETTER RESULTS ON FID/KID METRICS, THEY FALL SHORT ON MORE ADVANCED IMAGE QUALITY ASSESSMENT METRICS SUCH AS CLIP-I AND PERFORM SIGNIFICANTLY WORSE IN 3D SHAPE QUALITY. FOR MULTI-VIEW TO 3D METHODS, WE ALSO INCLUDE THE NUMBER OF INPUT VIEWS (V=#). NOTE THAT SHAPE-E∗ IS TRAINED ON PROPRIETARY INTERNAL 3D DATA. IT IS INCLUDED HERE FOR REFERENCE.

Method CLIP-I↑ FID↓ KID(%)↓ COV(%)↑ MMD(‰)↓

OpenLRM 86.37 38.41 1.87 39.33 29.08 Splatter-Image 84.10 48.80 3.65 37.66 30.69

One-2-3-45 (V=12) 80.72 88.39 6.34 35.09 CRM (V=6) 85.76 45.53 1.93 38.83 28.91 Lara (V=4) 84.64 43.74 1.95 39.33 28.84 LGM (V=4) 87.99 19.93 0.55 50.83 22.06

Shape-E∗ 77.05 138.53 11.95 61.33 19.17 LN3DIFF++ 88.29 23.01 0.75 55.17 19.94 LN3DIFF++-Flexi 85.46 36.28 1.12 53.17 22.02

Single-view Image	to 3D Multi-view Image	to 3D Native	3D Diffusion	Model

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

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

Input Open-LRM Splatter Image One-2-3-45 CRM Lara LGM Shape-E LN3Diff

LN3Diff-Flexicube

- Fig. 8. Qualitative Comparison of Image-to-3D. We showcase the novel view 3D reconstruction of all methods given a single image from the unseen GSO dataset. Our proposed method achieves consistently stable performance across all cases. Note that though feed-forward 3D reconstruction methods achieve sharper texture reconstruction, these methods fail to yield intact 3D predictions under challenging cases (e.g, the house in row 1). In contrast, our proposed native 3D diffusion model achieves consistently better performance. The LN3DIFF++-Flexicube yields high-quality normal rendering, which cannot be easily rendered from the NeRF version. Zoom in for the best view.

TABLE IV ABLATION OF RECONSTRUCTION ARCH DESIGN. WE ABLATE THE DESIGN OF OUR AUTO-ENCODING ARCHITECTURE. EACH COMPONENT CONTRIBUTES TO A CONSISTENT GAIN IN THE RECONSTRUCTION PERFORMANCE, INDICATING AN IMPROVEMENT IN THE MODELING CAPACITY.

Design PSNR@100K 2D Conv Baseline 17.46 + ViT Block 18.92 ViT Block → DiT Block 20.61 + Plucker Embedding 21.29 + Cross-Plane Attention 21.70 + Self-Plane Attention 21.95

TABLE V DIFFUSION SAMPLING SPEED AND LATENT SIZE. WE PROVIDE THE SAMPLING TIME PER INSTANCE EVALUATED ON 1 V100, ALONG WITH THE LATENT SIZE. OUR METHOD ACHIEVES FASTER SAMPLING SPEED WHILE MAINTAINING SUPERIOR GENERATION PERFORMANCE.

Method V100-sec Latent Size Get3D/EG3D <0.5 256 SSDNeRF 8.1 1282 × 18 RenderDiffusion 15.8 DiffRF 18.7 323 × 4

LN3DIFF++uncond 5.7 322 × 12 LN3DIFF++cfg 7.5 322 × 12

[Figure 162]

[Figure 163]

Input Image W/O Novel View Discriminator W Novel View Discriminator

[Figure 164]

- Fig. 9. Ablation of novel view discriminator. Adding novel view discriminator leads to more 3D consistent predictions.

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

Input w/o DINO	features w/	DINO	features

In addition, training our proposed model on single-view datasets (e.g, FFHQ) leads to unnatural background patterns, as illustrated in Fig. 7. We hypothesize that these artifacts arise from the novel view discriminator: while it enforces plausible novel-view rendering, the background region often develops artifacts that function as adversarial shortcuts to increase

Fig. 10. Ablation of DINO features in image-conditioned 3D generation. Using CLIP features only leads to unfaithful 3D generation, as highlighted in the red box. Introducing auxiliary DINO features improves both fidelity and faithfulness.

the discriminator score. A potential solution to mitigate this artifact is to reduce the GAN loss weights of the novel view discrimination training, but at the cost of 3D plausibility 9. Therefore, developing more effective ways for training 3D diffusion models over a monocular view dataset is a valuable avenue for future research. On the diffusion formulation, LN3DIFF++ learns the joint distribution of geometry and texture. This is likely to yield suboptimal performance. A geometry-texture disentangled framework [153] enables more flexible generation and better texture quality by leveraging the powerful priors of 2D diffusion model [102]. Regarding data, LN3DIFF++ is trained on the artist-created data only. Adding more real-world data such as MVImageNet [146] shall further improve the generality. Besides, Objaverse-XL [21] and 3D-Future [32] may enhance model diversity. Application side, Current 3D generation research remains largely focused on base model design. Future work could explore incorporating 3D-aware control signals [152], toonification [143], [151] and training-free editing [83]. Overall, our method represents a step toward a native 3D diffusion model and offers inspiration for future research in this direction.

Acknowledgement. This study is supported under the RIE2020 Industry Alignment Fund Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and inkind contributions from the industry partner(s). It is also supported by Singapore MOE AcRF Tier 2 (MOE-T2EP202210011).

APPENDIX

In this supplementary material, we provide additional details regarding the implementations and additional results. We also discuss the limitations of our model.

Broader Social Impact. In this paper, we introduce a new latent 3D diffusion model designed to produce high-quality textures and geometry using a single model. As a result, our approach has the potential to be applied to generating DeepFakes or deceptive 3D assets, facilitating the creation of falsified images or videos. This raises concerns as individuals could exploit such technology with malicious intent, aiming to spread misinformation or tarnish reputations.

A. Training details

Diffusion. We mainly adopt the diffusion training pipeline implementation from ADM [23], continuous noise schedule from LSGM [126] with the spatial transformer attention implementation from LDM [102]. For ShapeNet and FFHQ dataset, we adopt U-Net [103] architecture and list the hyperparameters in Tab. VI. For Objaverse dataset, we adopt DiT-L [94] architecture with cross attention design, as proposed in PixArt [13]. The diffusion transformer is built with 24 layers with 16 heads and 1024 hidden dimension, which result in 458M parameters. VAE Architecture. For the convolutional encoder Eϕ, we adopt a lighter version of LDM [102] encoder with channel 64 and 1 residual blocks for efficiency. When training on Objaverse with V = 6, we incorporate 3D-aware attention [110] in the middle layer of the convolutional encoder. For

TABLE VI HYPERPARAMETERS AND ARCHITECTURE OF DIFFUSION MODEL ϵθ.

Diffusion Model Details Learning Rate 2e − 5 Batch Size 96 Optimizer AdamW Iterations 500K U-Net base channels 320 U-Net channel multiplier 1, 1, 2, 2, 4, 4 U-Net res block 2 U-Net attention resolutions 4,2,1 U-Net Use Spatial Transformer True U-Net Learn Sigma False U-Net Spatial Context Dim 768 U-Net attention head channels 64 U-Net pred type v U-Net norm layer type GroupNorm Noise Schedules Linear CFG Dropout prob 15% CLIP Latent Scaling Factor 18.4

convolutional upsampler DU, we further half the channel to 32. All other hyper-parameters remain at their default settings. Regarding the transformer decoder DT, we employ the DiTL/2 architecture, and overall saved VAE model takes around 1.5 GiB storage. The input dimension of z to the MLP in each DiT block is h × w × c for self-plane attention, and h×w ×3×c in cross-plane attention. When ablating the 3Daware attention in Tab.3, we adopt channel-wise concatenated latent h × w × (3c) for model input, as in SSDNeRF. Note that we trade off a smaller model with faster training speed due to the overall compute limit, and a heavier model would certainly empower better performance [94], [142]. We ignore the plucker camera condition for the ShapeNet and FFHQ dataset, over which we find raw RGB input already yields good enough performance.

B. Data and Baseline Comparison

Training data. For ShapeNet, following GET3D [33], we use the blender to render the multi-view images from 50 viewpoints for all ShapeNet datasets with foreground mask. Those camera points sample from the upper sphere of a ball with a 1.2 radius. For Objaverse, we use a high-quality subset from the pre-processed rendering from G-buffer Objaverse [99] for experiments. Since G-buffer Objaverse splits the subset into 10 general categories, we use all the 3D instances except from “Poor-quality”: Human-Shape, Animals, DailyUsed, Furniture, Buildings&Outdoor, Transportations, Plants, Food and Electronics. The ground truth camera pose, rendered multi-view images and depth maps are used for stage-1 VAE training.

Evaluation. The 2D metrics are calculated between 50k generated images and all available real images. Furthermore, for comparison of the geometrical quality, we sample 4096 points from the surface of 5000 objects and apply the Coverage Score (COV) and Minimum Matching Distance (MMD) using

[Figure 175]

[Figure 176]

[Figure 177]

The Eiffel Tower.

A plate of Sushi.

[Figure 178]

[Figure 179]

a stone water well with a wooden shed.

A wooden worktable.

[Figure 180]

[Figure 181]

A wooden chest with golden trim.

A blue plastic chair.

- Fig. 11. Objaverse Conditional Generation Given Text Prompt. We show two samples for each prompt. All results are sampled from DiT architectures. Zoom in for the best view.

Chamfer Distance (CD) as follows:

||x − y||22 +

||x − y||22,

##### CD(X,Y ) =

min

min

y∈Y

x∈X

x∈X

y∈Y

COV (Sg,Sr) = |{arg minY ∈S

CD(X,Y )|X ∈ Sg}|

r

,

|Sr| MMD(Sg,Sr) =

1 |Sr| Y ∈S

CD(X,Y )

min

X∈Sg

r

(6) where X ∈ Sg and Y ∈ Sr represent the generated shape and reference shape.

Note that we use 5k generated objects Sg and all training shapes Sr to calculate COV and MMD. For fairness, we normalize all point clouds by centering in the original and recalling the extent to [-1,1]. Coverage Score aims to evaluate the diversity of the generated samples, and MMD is used for measuring the quality of the generated samples. 2D metrics are evaluated at a resolution of 128 × 128. Since the GT data contains intern structures, we only sample the points from the outer surface of the object for results of all methods and ground truth.

For FID/KID evaluation, since different methods have their unique evaluation settings, we standardize this process by re-rendering each baseline’s samples using a fixed uppersphere ellipsoid camera pose trajectory of size 20. With 2.5K sampled 3D instances for each method, we recalculate FID@50K/KID@50K, ensuring a fair comparison across all methods. For text-conditioned 3D generation on category-free datasets, we leverage the text prompt provided in GPTEval3D [136] for evaluation. The CLIP similarity score is calculated following previous work [46]. Note that GaussianCube

V1.1 is trained on proprietary data, while our method is trained on public dataset. We failed to run GaussianCube V1.0 since the publicly released checkpoint is broken.

Details about Baselines. We reproduce EG3D, GET3D, and SSDNeRF on our ShapeNet rendering using their officially released codebases. In the case of RenderDiffusion, we use the code and pre-trained model shared by the author for ShapeNet experiments. Regarding FFHQ dataset, due to the unavailability of the corresponding inference configuration and checkpoint from the authors, we incorporate their unconditional generation and monocular reconstruction results as reported in their paper. For DiffRF, given the absence of the public code, we reproduce their method with Plenoxel [31] and ADM [23].

C. More Qualitative 3D Generation Results

We include more uncurated samples generated by our method on ShapeNet in Fig. 12, on FFHQ in Fig. 13, and category-free text-conditioned 3D generation results in Fig. 11

D. More Monocular 3D Reconstruction Results

We further benchmark the generalization ability of our stage-1 monocular 3D reconstruction VAE. For ShapeNet, we include the quantitative evaluation in Tab. VII. Our method achieves a comparable performance with monocular 3D reconstruction baselines. Note that strictly saying, our stage1 VAE shares a similar setting with Pix2NeRF [5], whose encoder also has a latent space for generative modeling. Other reconstruction-specific methods like PixelNeRF [145] do not have these requirements and can leverage some designs like

[Figure 182]

ShapeNetCarShapeNetChairShapeNetPlane

[Figure 183]

[Figure 184]

- Fig. 12. Unconditional 3D Generation by LN3DIFF++ (Uncurated). We showcase uncurated samples generated by LN3DIFF++ on ShapeNet three categories. We visualize two views for each sample. Better zoom in.

pixel-aligned features and long-skip connections to further boost the reconstruction performance. We include their performance mainly for reference and leave training the stage-1 VAE model with performance comparable with those state-ofthe-art 3D reconstruction models for future work.

Besides, we visualize LN3DIFF++’s stage-1 monocular VAE reconstruction performance over our Objaverse split in Fig. 15. As can be seen, though only one view is provided as the input, our monocular VAE reconstruction can yield high-quality and view-consistent 3D reconstruction with a detailed depth map. Quantitatively, the novel-view reconstruction performance over our whole Objaverse dataset achieves an

average PSNR of 26.14. This demonstrates that our latent space can be treated as a compact proxy for efficient 3D diffusion training.

We have included a brief discussion of limitations in the main submission. Here we include more details along with the visual failure cases for a more in-depth analysis of LN3DIFF++’s limitations and future improvement directions.

E. VAE Limitations

We have demonstrated that using a monocular image as encoder input can achieve high-quality 3D reconstruction. However, we noticed that for some challenging cases with

[Figure 185]

FFHQ Unconditional Generation

- Fig. 13. Unconditional 3D Generation by LN3DIFF++ (Uncurated). We showcase uncurated samples generated by LN3DIFF++ on FFHQ. We visualize two views for each sample along with the extracted depth. Better zoom in.

TABLE VII QUANTITATIVE RESULTS ON SHAPENET-SRN [9], [113] CHAIRS EVALUATE ON 128 × 128. LEGEND: * – REQUIRES TEST TIME OPTIMIZATION. NOTE THAT OUR STAGE-1 VAE SHARES THE SAME SETTING ONLY WITH PIX2NERF [145], WHICH ALSO HAS AN EXPLICIT LATENT SPACE FOR GENERATIVE LEARNING. OTHER BASELINES ARE INCLUDED FOR REFERENCE.

Method PSNR ↑ SSIM ↑

GRF [125] 21.25 0.86 TCO [120] 21.27 0.88 dGQN [28] 21.59 0.87 ENR [27] 22.83 SRN* [113] 22.89 0.89 CodeNeRF* [50] 22.39 0.87 PixelNeRF [145] 23.72 0.91

Pix2NeRF [5] conditional 18.14 0.84 Ours 20.91 0.89

diverse color and geometry details, the monocular encoder leads to blurry artifacts. As labeled in Fig. 15, our method with monocular input may yield floating artifacts over unseen viewpoints. We hypothesize that these artifacts are largely due to the ambiguity of monocular input and the use of regression loss (L2/LPIPS) during training. These observations demon-

[Figure 186]

[Figure 187]

A yellow plastic chair with armrests.

[Figure 188]

[Figure 189]

Two yellow plastic chair with armrests.

Fig. 14. Limitation analysis. We showcase the deficiency to generate composed 3D scenes by LN3DIFF++. As shown here, the prompt Two chair yields similar results with A chair.

strate that switching to a multi-view encoder is necessary for better performance.

Besides, since our VAE requires plucker camera condition as input, the pre-trained VAE method cannot be directly applied to the unposed dataset. However, we believe this is not a research issue at the current time, considering the current methods still perform lower than expected on existing highquality posed 3D datasets like Objaverse.

- F. 3D Diffusion Limitations

As one of the earliest 3D diffusion models that works on Objaverse, our method still suffers from several limitations that require investigation in the future. (1) The support of image-to-3D on Objaverse. Currently, we leverage CLIPtext encoder with the 77 tokens as the conditional input. However, unlike 2D AIGC with T2I models [102], 3D content creation can be greatly simplified by providing easy-to-get 2D images. An intuitive implementation is by using our ShapeNet 3D diffusion setting, which provides the final normalized CLIP text embeddings as the diffusion condition. However, as shown in the lower half of Fig. 4 in the main submission, the CLIP encoder is better at extracting high-level semantics rather than low-level visual details. Therefore, incorporating more accurate image-conditioned 3D diffusion design like ControlNet [152] to enable monocular 3D reconstruction and control is worth exploring in the future. (2) Compositionality. Currently, our method is trained on object-centric dataset with simple captions, so the current model does not support composed 3D generation. For example, the prompt ”Two yellow plastic chair with armchests” will still yield one chair, as visualized in Fig. 14. (3) UV map. To better integrate the learning-based method into the gaming and movie industry, a high-quality UV texture map is required. A potential solution is to disentangle the learned geometry and texture space and build the connection with UV space through dense correspondences [64].

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

###### Fig. 15. Monocular 3D Reconstruction by LN3DIFF++ stage-1 VAE on Objaverse (Uncurated). We showcase uncurated samples monocular-reconstructed by LN3DIFF++ on Objaverse. From left to right, we visualize the input image, four reconstructed novel views with the corresponding depth maps. Artifacts are labeled in Red. Better zoom in.

- G. Biographies

Yushi Lan is a postdoctoral research fellow at the Visual Geometry Group, University of Oxford. He received his Ph.D. degree from the College of Computing and Data Science, Nanyang Technological University, in 2025, under the supervision of Prof. Chen Change Loy. He received his B.Eng. degree in Software Engineering from Beijing University of Posts and Telecommunications, China, in 2020. His research focuses on the intersection of computer vision, graphics, and machine learning, with particular interests in 3D generative models and 3D

[Figure 194]

representation learning. He has served as a reviewer for top-tier conferences and journals, including CVPR, ICCV, ECCV, ICLR, NeurIPS, and TPAMI.

Fangzhou Hong is a Research Fellow in the College of Computing and Data Science at Nanyang Technological University, advised by Prof. Ziwei Liu. He received Ph.D. degree in the College of Computing and Data Science at Nanyang Technological University in 2025. He received the B.Eng. degree in Software Engineering from Tsinghua University, China, in 2020. His research interests lie on computer vision and deep learning. Particularly, he is interested in 3D representation learning. He has published several papers and served as a reviewer for top conferences

[Figure 195]

and journals, e.g, CVPR, ICCV, ECCV, ICLR, NeurIPS, TPAMI.

Shangchen Zhou is currently a Research Assistant Professor at MMLab@NTU, Nanyang Technological University, Singapore. He received his Ph.D. (2024) in Computer Science from the same institution. He was selected as an outstanding reviewer in NeurIPS 2020. He won first place in three image restoration and enhancement challenges in NTIRE 2021. His works received notable recognition including the WAIC Youth Outstanding Paper Award Honorable Mention in 2023, the Snap Fellowship Honorable Mention in 2022, and the Best Paper

[Figure 196]

Award at ICIMCS, ACM in 2016. Additionally, He co-organized the “MIPI workshop” series in conjunction with ECCV 2022, CVPR 2023, and CVPR 2024. His research interests include image/video restoration and enhancement, generation and editing, etc.

Shuai Yang (S’19-M’20) received the B.S. and Ph.D. degrees (Hons.) in computer science from Peking University, Beijing, China, in 2015 and 2020, respectively. He is currently an assistant professor with the Wangxuan Institute of Computer Technology, Peking University. His current research interests include image stylization, image translation and image editing. He was a Research Assistant Professor with the S-Lab, Nanyang Technological University, Singapore, from Mar. 2023 to Feb. 2024. He was a postdoctoral research fellow at Nanyang

[Figure 197]

Technological University, from Oct. 2020 to Feb. 2023. He was a Visiting Scholar with the Texas A&M University, from Sep. 2018 to Sep. 2019. He was a Visiting Student with the National Institute of Informatics, Japan, from Mar. 2017 to Aug. 2017. He received the IEEE ICME 2020 Best Paper Awards and IEEE MMSP 2015 Top10% Paper Awards. He serves/served as an Area Chair of NeurIPS, ACM MM, BMVC, CVPR and ICLR.

[Figure 198]

Xuyi Meng is a Ph.D. student in Computer Science at Cornell University, advised by Prof. Wei-Chiu Ma. She received the M.S. degree in Computer and Information Science from the University of Pennsylvania in 2025 and the B.E. degree in Computer Science from Nanyang Technological University (NTU), Singapore, in 2023. Her research interests include 3D and 4D computer vision, generative modeling, and neural representations, with a focus on efficient and high-quality modeling of real-world scenes and dynamics.

[Figure 199]

Yongwei Chen is a Ph.D. student at the College of Computing and Data Science, Nanyang Technological University, Singapore. His research focuses on 3D vision, with an emphasis on generative 3D modeling, neural rendering, and 3D understanding. His work has been published in top-tier conferences such as CVPR, ICCV, ECCV, and NeurIPS, contributing to advancements in 3D computer vision and its applications.

[Figure 200]

Zhaoyang Lyu is a research scientist with the Shanghai AI Laboratory, working in a research group on Embodied AI. He received his Ph.D. (2018-2022) from Multimedia Laboratory (MMLab) at CUHK, advised by Prof. Dahua Lin. He obtained his Bachelor’s Degree (2014-2018) at Xi’an Jiaotong University. His current research focuses on generative world modeling for Embodied AI, including 3D interactive object and scene generation.

Bo Dai is an assistant professor in the Musketeers Foundation Institute of Data Science, The University of Hong Kong. He obtained his PhD degree from The Chinese University of Hong Kong. He was a research scientist with the Shanghai Artificial Intelligence Laboratory, and was a research assistant professor with S-Lab for advanced intelligence, Nanyang Technological University. He has authored or coauthored more than 80 papers in top-tier conferences and journals, with over 14000 google scholar citations. His research interests include Generative

[Figure 201]

AI and its interdisciplinary applications in areas covering Embodied AI, Scientific Discovery, Metaverse and Creativity. He is an area chair of ICLR, CVPR, NeurIPS and AAAI.

[Figure 202]

Xingang Pan Xingang Pan received the PhD degree in information engineering from The Chinese University of Hong Kong in 2021. He is currently a Nanyang assistant professor at College of Computing and Data Science, Nanyang Technological University. Previously, he was a postdoctoral researcher at Max Planck Institute for Informatics from 2021 to 2023. His research interests include generative models, visual generation, and 3D vision. He serves as an Area Chair for CVPR and 3DV. He is awarded the Singapore NRF Fellowship in 2024.

Chen Change Loy (Senior Member, IEEE) is a President’s Chair Professor at the School of Computer Science and Engineering, Nanyang Technological University, Singapore. Before joining NTU, he served as a Research Assistant Professor at the MMLab of The Chinese University of Hong Kong, from 2013 to 2018. He received his Ph.D. (2010) in Computer Science from the Queen Mary University of London. He was a postdoctoral researcher at Queen Mary University of London and Vision Semantics Limited, from 2010 to 2013. He serves/served as an

[Figure 203]

Associate Editor of the IEEE Transactions on Pattern Analysis and Machine Intelligence, International Journal of Computer Vision and Computer Vision and Image Understanding. He also serves/served as an Area Chair of major conferences such as ICCV, CVPR, ECCV, NeurIPS, and ICLR. He serves as the Program Co-Chair of CVPR 2026 and General Co-Chair of ACCV 2028. His research interests include image/video restoration and enhancement, generative tasks, and representation learning.

REFERENCES

- [1] Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797, 2023.
- [2] Titas Anciukeviˇcius, Zexiang Xu, Matthew Fisher, Paul Henderson, Hakan Bilen, Niloy J Mitra, and Paul Guerrero. RenderDiffusion: Image diffusion for 3D reconstruction, inpainting and generation. In CVPR, 2023.
- [3] Mikołaj Bi´nkowski, Dougal J. Sutherland, Michael Arbel, and Arthur Gretton. Demystifying MMD GANs. In ICLR, 2018.
- [4] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale GAN training for high fidelity natural image synthesis. In ICLR, 2019.
- [5] Shengqu Cai, Anton Obukhov, Dengxin Dai, and Luc Van Gool. Pix2NeRF: Unsupervised Conditional p-GAN for Single Image to Neural Radiance Fields Translation. In CVPR, 2022.
- [6] Eric Chan, Marco Monteiro, Petr Kellnhofer, Jiajun Wu, and G. Wetzstein. Pi-GAN: Periodic implicit generative adversarial networks for 3D-aware image synthesis. In CVPR, 2021.
- [7] Eric R. Chan, Connor Z. Lin, Matthew A. Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas Guibas, Jonathan Tremblay, Sameh Khamis, Tero Karras, and Gordon Wetzstein. Efficient geometry-aware 3D generative adversarial networks. In CVPR, 2022.
- [8] Eric R. Chan, Koki Nagano, Matthew A. Chan, Alexander W. Bergman, Jeong Joon Park, Axel Levy, Miika Aittala, Shalini De Mello, Tero Karras, and Gordon Wetzstein. GeNVS: Generative novel view synthesis with 3D-aware diffusion models. In arXiv, 2023.
- [9] Angel X. Chang, Thomas Funkhouser, Leonidas Guibas, Pat Hanrahan, Qixing Huang, Zimo Li, Silvio Savarese, Manolis Savva, Shuran Song, Hao Su, Jianxiong Xiao, Li Yi, and Fisher Yu. ShapeNet: An Information-Rich 3D Model Repository. arXiv preprint arXiv:1512.03012, 2015.
- [10] Anpei Chen, Haofei Xu, Stefano Esposito, Siyu Tang, and Andreas Geiger. Lara: Efficient large-baseline radiance fields. In ECCV, 2024.
- [11] Hansheng Chen, Jiatao Gu, Anpei Chen, Wei Tian, Zhuowen Tu, Lingjie Liu, and Hao Su. Single-stage diffusion NeRF: A unified approach to 3D generation and reconstruction. In ICCV, 2023.
- [12] Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, Le Xue, Caiming Xiong, and Ran Xu. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset, 2025.
- [13] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li. PixArt-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis, 2023.
- [14] Yongwei Chen, Yushi Lan, Shangchen Zhou, Tengfei Wang, and Xingang Pan. SAR3D: Autoregressive 3D object generation and understanding via multi-scale 3D VQVAE. In CVPR, pages 28371– 28382, 2025.
- [15] Yongwei Chen, Tengfei Wang, Tong Wu, Xingang Pan, Kui Jia, and Ziwei Liu. ComboVerse: Compositional 3D assets creation using spatially-aware diffusion guidance. arXiv preprint arXiv:2403.12409, 2024.
- [16] Gene Chou, Yuval Bahat, and Felix Heide. Diffusion-SDF: Conditional generative modeling of signed distance functions. In ICCV, 2023.
- [17] SpConv Contributors. SpConv: Spatially sparse convolution library. https://github.com/traveller59/spconv, 2022.
- [18] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In ICLR, 2024.
- [19] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In NeurIPS, 2022.
- [20] Mostafa Dehghani, Josip Djolonga, Basil Mustafa, Piotr Padlewski, Jonathan Heek, Justin Gilmer, Andreas Peter Steiner, Mathilde Caron, Robert Geirhos, Ibrahim Alabdulmohsin, et al. Scaling vision transformers to 22 billion parameters. In ICML, 2023.
- [21] Matt Deitke, Ruoshi Liu, Matthew Wallingford, Huong Ngo, Oscar Michel, Aditya Kusupati, Alan Fan, Christian Laforte, Vikram Voleti, Samir Yitzhak Gadre, Eli VanderBilt, Aniruddha Kembhavi, Carl Vondrick, Georgia Gkioxari, Kiana Ehsani, Ludwig Schmidt, and Ali Farhadi. Objaverse-XL: A universe of 10m+ 3d objects. arXiv preprint arXiv:2307.05663, 2023.
- [22] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3D objects. arXiv preprint arXiv:2212.08051, 2022.
- [23] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. NeurIPS, 2021.
- [24] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani,

- Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.
- [25] Laura Downs, Anthony Francis, Nate Koenig, Brandon Kinman, Ryan Hickman, Krista Reymann, Thomas B McHugh, and Vincent Vanhoucke. Google scanned objects: A high-quality dataset of 3d scanned household items. In ICRA, 2022.
- [26] Emilien Dupont, Hyunjik Kim, S. M. Ali Eslami, Danilo Jimenez Rezende, and Dan Rosenbaum. From data to functa: Your data point is a function and you can treat it like one. In ICML, 2022.
- [27] Emilien Dupont, Miguel Bautista Martin, Alex Colburn, Aditya Sankar, Josh Susskind, and Qi Shan. Equivariant neural rendering. In International Conference on Machine Learning, pages 2761–2770. PMLR, 2020.
- [28] S. M. Ali Eslami, Danilo Jimenez Rezende, Frederic Besse, Fabio Viola, Ari S. Morcos, Marta Garnelo, Avraham Ruderman, Andrei A. Rusu, Ivo Danihelka, Karol Gregor, David P. Reichert, Lars Buesing, Th´eophane Weber, Oriol Vinyals, Dan Rosenbaum, Neil C. Rabinowitz, Helen King, Chloe Hillier, Matthew M. Botvinick, Daan Wierstra, Koray Kavukcuoglu, and Demis Hassabis. Neural scene representation and rendering. Science, 360:1204 – 1210, 2018.
- [29] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024.
- [30] Patrick Esser, Robin Rombach, and Bj¨orn Ommer. Taming transformers for high-resolution image synthesis. In CVPR, 2021.
- [31] Fridovich-Keil and Yu, Matthew Tancik, Qinhong Chen, Benjamin Recht, and Angjoo Kanazawa. Plenoxels: Radiance fields without neural networks. In CVPR, 2022.
- [32] Huan Fu, Rongfei Jia, Lin Gao, Mingming Gong, Binqiang Zhao, Steve Maybank, and Dacheng Tao. 3d-future: 3d furniture shape with texture. International Journal of Computer Vision, pages 1–25, 2021.
- [33] Jun Gao, Tianchang Shen, Zian Wang, Wenzheng Chen, Kangxue Yin, Daiqing Li, Or Litany, Zan Gojcic, and Sanja Fidler. Get3D: A generative model of high quality 3D textured shapes learned from images. In NeurIPS, 2022.
- [34] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron C. Courville, and Yoshua Bengio. Generative adversarial nets. In NeurIPS, 2014.
- [35] Jiatao Gu, Qingzhe Gao, Shuangfei Zhai, Baoquan Chen, Lingjie Liu, and Josh Susskind. Learning controllable 3D diffusion models from single-view images. arXiv preprint arXiv:2304.06700, 2023.
- [36] Jiatao Gu, Lingjie Liu, Peng Wang, and Christian Theobalt. StyleNeRF: A style-based 3D-aware generator for high-resolution image synthesis. In ICLR, 2021.
- [37] Ishaan Gulrajani, Faruk Ahmed, Mart´ın Arjovsky, Vincent Dumoulin, and Aaron C. Courville. Improved training of wasserstein GANs. In NeurIPS, 2017.
- [38] Anchit Gupta, Wenhan Xiong, Yixin Nie, Ian Jones, and Barlas O˘guz. 3dgen: Triplane latent diffusion for textured mesh generation, 2023.
- [39] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll’ar, and Ross B. Girshick. Masked autoencoders are scalable vision learners. In CVPR, 2022.
- [40] Zexin He and Tengfei Wang. OpenLRM: Open-source large reconstruction models. https://github.com/3DTopia/OpenLRM, 2023.
- [41] Philipp Henzler, Niloy J Mitra, and Tobias Ritschel. Escaping plato’s cave: 3D shape from adversarial rendering. In ICCV, 2019.
- [42] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local nash equilibrium. NeurIPS, 2017.
- [43] Jonathan Ho. Classifier-free diffusion guidance. In NeurIPS, 2021.
- [44] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020.
- [45] Fangzhou Hong, Zhaoxi Chen, Yushi Lan, Liang Pan, and Ziwei Liu. EVA3D: Compositional 3D human generation from 2d image collections. In ICLR, 2022.
- [46] Fangzhou Hong, Jiaxiang Tang, Ziang Cao, Min Shi, Tong Wu, Zhaoxi Chen, Tengfei Wang, Liang Pan, Dahua Lin, and Ziwei Liu. 3DTopia: Large text-to-3D generation model with hybrid diffusion priors. arXiv preprint arXiv:2403.02234, 2024.
- [47] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. In ICLR, 2024.
- [48] Emiel Hoogeboom, Jonathan Heek, and Tim Salimans. simple diffusion: End-to-end diffusion for high resolution images. In ICML, 2023.
- [49] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d Gaussian splatting for geometrically accurate radiance fields.

- In SIGGRAPH 2024 Conference Papers. Association for Computing Machinery, 2024.
- [50] Wonbong Jang and Lourdes Agapito. Codenerf: Disentangled neural radiance fields for object categories. In ICCV, pages 12949–12958, 2021.
- [51] Heewoo Jun and Alex Nichol. Shap-E: Generating conditional 3D implicit functions. arXiv preprint arXiv:2305.02463, 2023.
- [52] Dhiraj Kalamkar, Dheevatsa Mudigere, Naveen Mellempudi, Dipankar Das, Kunal Banerjee, Sasikanth Avancha, Dharma Teja Vooturi, Nataraj Jammalamadaka, Jianyu Huang, Hector Yuen, Jiyan Yang, Jongsoo Park, Alexander Heinecke, Evangelos Georganas, Sudarshan Srinivasan, Abhisek Kundu, Misha Smelyanskiy, Bharat Kaul, and Pradeep Dubey. A study of bfloat16 for deep learning training, 2019.
- [53] Tero Karras, Timo Aila, Samuli Laine, and Jaakko Lehtinen. Progressive growing of GANs for improved quality, stability, and variation. In ICLR, 2018.
- [54] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In Neurips, 2022.
- [55] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In CVPR, 2019.
- [56] Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, and Timo Aila. Analyzing and improving the image quality of StyleGAN. In CVPR, 2020.
- [57] Hiroharu Kato and Tatsuya Harada. Learning view priors for singleview 3D reconstruction. In CVPR, 2019.
- [58] Junjie Ke, Qifei Wang, Yilin Wang, Peyman Milanfar, and Feng Yang. Musiq: Multi-scale image quality transformer. In ICCV, pages 5148– 5157, 2021.
- [59] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3D gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42(4):1–14, 2023.
- [60] Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. arXiv, 2013.
- [61] Adam R. Kosiorek, Heiko Strathmann, Daniel Zoran, Pol Moreno, Rosalia Schneider, Sovna Mokr’a, and Danilo Jimenez Rezende. NeRFVAE: A geometry aware 3D scene generative model. ICML, 2021.
- [62] Nupur Kumari, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Ensembling off-the-shelf models for gan training. In CVPR, 2022.
- [63] Yushi Lan, Fangzhou Hong, Shuai Yang, Shangchen Zhou, Xuyi Meng, Bo Dai, Xingang Pan, and Chen Change Loy. Ln3diff: Scalable latent neural fields diffusion for speedy 3d generation. In ECCV, 2024.
- [64] Yushi Lan, Chen Change Loy, and Bo Dai. DDF: Correspondence distillation from nerf-based gan. IJCV, 2022.
- [65] Yushi Lan, Xuyi Meng, Shuai Yang, Chen Change Loy, and Bo Dai. E3DGE: Self-supervised geometry-aware encoder for style-based 3D gan inversion. In CVPR, 2023.
- [66] Yushi Lan, Feitong Tan, Di Qiu, Qiangeng Xu, Kyle Genova, Zeng Huang, Sean Fanello, Rohit Pandey, Thomas Funkhouser, Chen Change Loy, and Yinda Zhang. Gaussian3diff: 3d gaussian diffusion for 3d full head synthesis and editing. In ECCV, 2024.
- [67] Yushi Lan, Shangchen Zhou, Zhaoyang Lyu, Fangzhou Hong, Shuai Yang, Bo Dai, Xingang Pan, and Chen Change Loy. GaussianAnything: Interactive point cloud latent diffusion for 3D generation. In ICLR, 2025.
- [68] Muheng Li, Yueqi Duan, Jie Zhou, and Jiwen Lu. Diffusion-SDF: Text-to-shape via voxelized diffusion. In CVPR, 2023.
- [69] Weiyu Li, Jiarui Liu, Hongyu Yan, Rui Chen, Yixun Liang, Xuelin Chen, Ping Tan, and Xiaoxiao Long. Craftsman3d: High-fidelity mesh generation with 3d native diffusion and interactive geometry refiner. In CVPR, pages 5307–5317, June 2025.
- [70] Chenguo Lin, Panwang Pan, Bangbang Yang, Zeming Li, and Yadong Mu. Diffsplat: Repurposing image diffusion models for scalable 3d gaussian splat generation. In ICLR, 2025.
- [71] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In ICLR, 2023.
- [72] Minghua Liu, Chao Xu, Haian Jin, Linghao Chen, Zexiang Xu, Hao Su, et al. One-2-3-45: Any single image to 3D mesh in 45 seconds without per-shape optimization. arXiv preprint arXiv:2306.16928, 2023.
- [73] Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3d object. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 9298–9309, October 2023.
- [74] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2023.
- [75] Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian

- Theobalt, et al. Wonder3D: Single image to 3D using cross-domain diffusion. In CVPR, 2024.
- [76] James Lucas, G. Tucker, Roger Baker Grosse, and Mohammad Norouzi. Understanding posterior collapse in generative latent variable models. In ICLR, 2019.
- [77] Tiange Luo, Chris Rockwell, Honglak Lee, and Justin Johnson. Scalable 3d captioning with pretrained models. Neurips, 36:75307–75337, 2023.
- [78] Chenlin Meng, Ruiqi Gao, Diederik P. Kingma, Stefano Ermon, Jonathan Ho, and Tim Salimans. On distillation of guided diffusion models. In CVPR, pages 14297–14306, 2022.
- [79] Xuyi Meng, Chen Wang, Jiahui Lei, Kostas Daniilidis, Jiatao Gu, and Lingjie Liu. Zero-1-to-g: Taming pretrained 2d diffusion model for direct 3d generation. TMLR, 2025.
- [80] Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. Occupancy Networks: Learning 3D reconstruction in function space. In CVPR, 2019.
- [81] Lu Mi, Abhijit Kundu, David Ross, Frank Dellaert, Noah Snavely, and Alireza Fathi. im2nerf: Image to neural radiance field in the wild. In arXiv, 2022.
- [82] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020.
- [83] Chong Mou, Xintao Wang, Jiechong Song, Ying Shan, and Jian Zhang. Diffeditor: Boosting accuracy and flexibility on diffusion-based image editing. arXiv preprint arXiv:2402.02583, 2023.
- [84] Norman M¨uller, Yawar Siddiqui, Lorenzo Porzi, Samuel Rota Bulo, Peter Kontschieder, and Matthias Nießner. DiffRF: Rendering-guided 3D radiance field diffusion. In CVPR, 2023.
- [85] Thu Nguyen-Phuoc, Chuan Li, Lucas Theis, Christian Richardt, and Yongliang Yang. HoloGAN: Unsupervised Learning of 3D Representations From Natural Images. In ICCV, 2019.
- [86] Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, and Mark Chen. Point-E: A system for generating 3D point clouds from complex prompts, 2022.
- [87] Michael Niemeyer and Andreas Geiger. GIRAFFE: Representing scenes as compositional generative neural feature fields. In CVPR, 2021.
- [88] OpenAI. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.
- [89] Maxime Oquab, Timoth´ee Darcet, Theo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Russell Howes, Po-Yao Huang, Hu Xu, Vasu Sharma, Shang-Wen Li, Wojciech Galuba, Mike Rabbat, Mido Assran, Nicolas Ballas, Gabriel Synnaeve, Ishan Misra, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision, 2023.
- [90] Roy Or-El, Xuan Luo, Mengyi Shan, Eli Shechtman, Jeong Joon Park, and Ira Kemelmacher-Shlizerman. StyleSDF: High-resolution 3D-consistent image and geometry generation. In CVPR, 2021.
- [91] Xingang Pan, Bo Dai, Ziwei Liu, Chen Change Loy, and Ping Luo. Do 2D GANs know 3D shape? Unsupervised 3D Shape Reconstruction from 2D Image GANs. In ICLR, 2021.
- [92] Xichen Pan, Satya Narayan Shukla, Aashu Singh, Zhuokai Zhao, Shlok Kumar Mishra, Jialiang Wang, Zhiyang Xu, Jiuhai Chen, Kunpeng Li, Felix Juefei-Xu, Ji Hou, and Saining Xie. Transfer between modalities with metaqueries. arXiv preprint arXiv:2504.06256, 2025.
- [93] Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. Deepsdf: Learning continuous signed distance functions for shape representation. In CVPR, pages 165–174, 2019.
- [94] William Peebles and Saining Xie. Scalable diffusion models with transformers. In ICCV, 2023.
- [95] Songyou Peng, Michael Niemeyer, Lars Mescheder, Marc Pollefeys, and Andreas Geiger. Convolutional occupancy networks. In ECCV, 2020.
- [96] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In arXiv, 2023.
- [97] Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. DreamFusion: Text-to-3D using 2D diffusion. ICLR, 2022.
- [98] Charles Qi, Hao Su, Kaichun Mo, and Leonidas Guibas. PointNet: Deep learning on point sets for 3D classification and segmentation. arXiv, 2016.
- [99] Lingteng Qiu, Guanying Chen, Xiaodong Gu, Qi Zuo, Mutian Xu, Yushuang Wu, Weihao Yuan, Zilong Dong, Liefeng Bo, and Xiaoguang Han. Richdreamer: A generalizable normal-depth diffusion model for detail richness in text-to-3d. In CVPR, pages 9914–9925, 2024.
- [100] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh,

- Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021.
- [101] Daniel Rebain, Mark Matthews, Kwang Moo Yi, Dmitry Lagun, and Andrea Tagliasacchi. LOLNeRF: Learn from one look. In CVPR, 2022.
- [102] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.
- [103] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional networks for biomedical image segmentation. In MICCAI, 2015.
- [104] Mehdi S. M. Sajjadi, Henning Meyer, Etienne Pot, Urs Bergmann, Klaus Greff, Noha Radwan, Suhani Vora, Mario Lucic, Daniel Duckworth, Alexey Dosovitskiy, Jakob Uszkoreit, Thomas Funkhouser, and Andrea Tagliasacchi. Scene Representation Transformer: Geometryfree novel view synthesis through set-latent scene representations. CVPR, 2022.
- [105] Kyle Sargent, Jing Yu Koh, Han Zhang, Huiwen Chang, Charles Herrmann, Pratul P. Srinivasan, Jiajun Wu, and Deqing Sun. VQ3D: Learning a 3D-aware generative model on imagenet. ICCV, 2023.
- [106] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, Patrick Schramowski, Srivatsa Kundurthy, Katherine Crowson, Ludwig Schmidt, Robert Kaczmarczyk, and Jenia Jitsev. LAION-5B: An open large-scale dataset for training next generation image-text models. In arXiv, 2022.
- [107] Katja Schwarz, Yiyi Liao, Michael Niemeyer, and Andreas Geiger. GRAF: Generative radiance fields for 3D-aware image synthesis. In NeurIPS, 2020.
- [108] Tianchang Shen, Jacob Munkberg, Jon Hasselgren, Kangxue Yin, Zian Wang, Wenzheng Chen, Zan Gojcic, Sanja Fidler, Nicholas Sharp, and Jun Gao. Flexible isosurface extraction for gradient-based mesh optimization. ACM Trans. Graph., 42(4), jul 2023.
- [109] Ruoxi Shi, Hansheng Chen, Zhuoyang Zhang, Minghua Liu, Chao Xu, Xinyue Wei, Linghao Chen, Chong Zeng, and Hao Su. Zero123++: a single image to consistent multi-view diffusion base model. In arXiv, 2023.
- [110] Yichun Shi, Peng Wang, Jianglong Ye, Long Mai, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3D generation. arXiv:2308.16512, 2023.
- [111] Jessica Shue, Eric Chan, Ryan Po, Zachary Ankner, Jiajun Wu, and Gordon Wetzstein. 3d neural field generation using triplane diffusion. In CVPR, 2022.
- [112] Vincent Sitzmann, Semon Rezchikov, William T. Freeman, Joshua B. Tenenbaum, and Fredo Durand. Light field networks: Neural scene representations with single-evaluation rendering. In NeurIPS, 2021.
- [113] Vincent Sitzmann, Michael Zollh¨ofer, and Gordon Wetzstein. Scene Representation Networks: Continuous 3D-structure-aware neural scene representations. In NeurIPS, 2019.
- [114] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In ICLR, 2021.
- [115] Jingxiang Sun, Xuan Wang, Yichun Shi, Lizhen Wang, Jue Wang, and Yebin Liu. Ide-3d: Interactive disentangled editing for high-resolution 3D-aware portrait synthesis. ACM Transactions on Graphics (TOG), 41(6):1–10, 2022.
- [116] Jingxiang Sun, Xuan Wang, Yong Zhang, Xiaoyu Li, Qi Zhang, Yebin Liu, and Jue Wang. FENeRF: Face editing in neural radiance fields. In arXiv, 2021.
- [117] Stanislaw Szymanowicz, Christian Rupprecht, and Andrea Vedaldi. Splatter image: Ultra-fast single-view 3d reconstruction. In The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [118] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view Gaussian model for high-resolution 3d content creation. In ECCV, 2024.
- [119] Zhicong Tang, Shuyang Gu, Chunyu Wang, Ting Zhang, Jianmin Bao, Dong Chen, and Baining Guo. Volumediffusion: Flexible text-to-3D generation with efficient volumetric encoder, 2023.
- [120] Maxim Tatarchenko, Alexey Dosovitskiy, and Thomas Brox. Multiview 3d models from single images with a convolutional network, 2016.
- [121] Anju Tewari, Otto Fried, Justus Thies, Vincent Sitzmann, S. Lombardi, Z Xu, Tanaba Simon, Matthias Nießner, Edgar Tretschk, L. Liu, Ben Mildenhall, Pranatharthi Srinivasan, R. Pandey, Sergio OrtsEscolano, S. Fanello, M. Guang Guo, Gordon Wetzstein, J y Zhu, Christian Theobalt, Manju Agrawala, Donald B. Goldman, and Michael Zollh¨ofer. Advances in neural rendering. Computer Graphics Forum, 41, 2021.

- [122] Ayush Tewari, Tianwei Yin, George Cazenavette, Semon Rezchikov, Joshua B. Tenenbaum, Fr´edo Durand, William T. Freeman, and Vincent Sitzmann. Diffusion with forward models: Solving stochastic inverse problems without direct supervision. In NeurIPS, 2023.
- [123] Hoang Thanh-Tung and T. Tran. Catastrophic forgetting and mode collapse in gans. IJCNN, pages 1–10, 2020.
- [124] Hugues Thomas, Charles R Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, Fran¸cois Goulette, and Leonidas J Guibas. KPConv: Flexible and deformable convolution for point clouds. In ICCV, 2019.
- [125] Alex Trevithick and Bo Yang. GRF: Learning a general radiance field for 3D scene representation and rendering. In ICCV, 2021.
- [126] Arash Vahdat, Karsten Kreis, and Jan Kautz. Score-based generative modeling in latent space. In NeurIPS, 2021.
- [127] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.
- [128] Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P. Srinivasan, Howard Zhou, Jonathan T. Barron, Ricardo Martin-Brualla, Noah Snavely, and Thomas A. Funkhouser. IBRNet: Learning Multi-View Image-Based Rendering. In CVPR, 2021.
- [129] Tengfei Wang, Bo Zhang, Ting Zhang, Shuyang Gu, Jianmin Bao, Tadas Baltrusaitis, Jingjing Shen, Dong Chen, Fang Wen, Qifeng Chen, et al. RODIN: A generative model for sculpting 3D digital avatars using diffusion. In CVPR, 2023.
- [130] Zhengyi Wang, Cheng Lu, Yikai Wang, Fan Bao, Chongxuan Li, Hang Su, and Jun Zhu. Prolificdreamer: High-fidelity and diverse text-to-3D generation with variational score distillation. In NeurIPS, 2023.
- [131] Zhengyi Wang, Yikai Wang, Yifei Chen, Chendong Xiang, Shuo Chen, Dajiang Yu, Chongxuan Li, Hang Su, and Jun Zhu. CRM: Single image to 3D textured mesh with convolutional reconstruction model. In ECCV, 2024.
- [132] Chung-Yi Weng, Pratul P. Srinivasan, Brian Curless, and Ira Kemelmacher-Shlizerman. PersonNeRF: Personalized reconstruction from photo collections. In CVPR, pages 524–533, June 2023.
- [133] Chao-Yuan Wu, Justin Johnson, Jitendra Malik, Christoph Feichtenhofer, and Georgia Gkioxari. Multiview compressive coding for 3D reconstruction. arXiv preprint arXiv:2301.08247, 2023.
- [134] Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Chunyi Li, Liang Liao, Annan Wang, Erli Zhang, Wenxiu Sun, Qiong Yan, Xiongkuo Min, Guangtai Zhai, and Weisi Lin. Q-align: Teaching lmms for visual scoring via discrete text-defined levels. arXiv preprint arXiv:2312.17090, 2023. Equal Contribution by Wu, Haoning and Zhang, Zicheng. Project Lead by Wu, Haoning. Corresponding Authors: Zhai, Guangtai and Lin, Weisi.
- [135] Shuang Wu, Youtian Lin, Feihu Zhang, Yifei Zeng, Jingxi Xu, Philip Torr, Xun Cao, and Yao Yao. Direct3D: Scalable image-to-3D generation via 3D latent diffusion transformer. In NeurIPS, 2024.
- [136] Tong Wu, Guandao Yang, Zhibing Li, Kai Zhang, Ziwei Liu, Leonidas Guibas, Dahua Lin, and Gordon Wetzstein. GPT-4v(ision) is a humanaligned evaluator for text-to-3d generation. In CVPR, 2024.
- [137] Kevin Xie, Jonathan Lorraine, Tianshi Cao, Jun Gao, James Lucas, Antonio Torralba, Sanja Fidler, and Xiaohui Zeng. LATTE3D: Largescale amortized text-to-enhanced 3D synthesis. ECCV, 2024.
- [138] Yiheng Xie, Towaki Takikawa, Shunsuke Saito, Or Litany, Shiqin Yan, Numair Khan, Federico Tombari, James Tompkin, Vincent Sitzmann, and Srinath Sridhar. Neural fields in visual computing and beyond. Computer Graphics Forum, 41, 2021.
- [139] Bojun Xiong, Si-Tong Wei, Xin-Yang Zheng, Yan-Pei Cao, Zhouhui Lian, and Peng-Shuai Wang. OctFusion: Octree-based Diffusion Models for 3D Shape Generation. Computer Graphics Forum, 2025.
- [140] Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tie-Yan Liu. On layer normalization in the transformer architecture, 2020.
- [141] Jiale Xu, Weihao Cheng, Yiming Gao, Xintao Wang, Shenghua Gao, and Ying Shan. Instantmesh: Efficient 3D mesh generation from a single image with sparse-view large reconstruction models. arXiv preprint arXiv:2404.07191, 2024.
- [142] Yinghao Xu, Hao Tan, Fujun Luan, Sai Bi, Peng Wang, Jiahao Li, Zifan

- Shi, Kalyan Sunkavalli, Gordon Wetzstein, Zexiang Xu, and Kai Zhang. DMV3D: Denoising multi-view diffusion using 3D large reconstruction model. In ICLR, 2024.
- [143] Shuai Yang, Liming Jiang, Ziwei Liu, and Chen Change Loy. VToonify: Controllable high-resolution portrait video style transfer. ACM Transactions on Graphics (TOG), 41(6):1–15, 2022.
- [144] Lior Yariv, Omri Puny, Natalia Neverova, Oran Gafni, and Yaron Lipman. Mosaic-SDF for 3d generative models. In CVPR, 2024.
- [145] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. PixelNeRF: Neural radiance fields from one or few images. In CVPR, 2021.
- [146] Xianggang Yu, Mutian Xu, Yidan Zhang, Haolin Liu, Chongjie Ye, Yushuang Wu, Zizheng Yan, Tianyou Liang, Guanying Chen, Shuguang Cui, and Xiaoguang Han. MVImgNet: A large-scale dataset of multiview images. In CVPR, 2023.
- [147] Xiaohui Zeng, Arash Vahdat, Francis Williams, Zan Gojcic, Or Litany, Sanja Fidler, and Karsten Kreis. Lion: Latent point diffusion models for 3D shape generation. In NeurIPS, 2022.
- [148] Bowen Zhang, Yiji Cheng, Jiaolong Yang, Chunyu Wang, Feng Zhao, Yansong Tang, Dong Chen, and Baining Guo. GaussianCube: Structuring gaussian splatting using optimal transport for 3d generative modeling. In NeurIPS, 2022.
- [149] Biao Zhang and Rico Sennrich. Root Mean Square Layer Normalization. In Neurips, Vancouver, Canada, 2019.
- [150] Biao Zhang, Jiapeng Tang, Matthias Nießner, and Peter Wonka. 3DShape2VecSet: A 3D shape representation for neural fields and generative diffusion models. ACM Trans. Graph., 42(4), jul 2023.
- [151] Junzhe Zhang, Yushi Lan, Shuai Yang, Fangzhou Hong, Quan Wang, Chai Kiat Yeo, Ziwei Liu, and Chen Change Loy. Deformtoon3d: Deformable 3D toonification from neural radiance fields. In ICCV, 2023.
- [152] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023.
- [153] Longwen Zhang, Ziyu Wang, Qixuan Zhang, Qiwei Qiu, Anqi Pang, Haoran Jiang, Wei Yang, Lan Xu, and Jingyi Yu. CLAY: A controllable large-scale generative model for creating high-quality 3D assets. ACM Transactions on Graphics, 2024.
- [154] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018.
- [155] Linqi Zhou, Yilun Du, and Jiajun Wu. 3D shape generation and completion through point-voxel diffusion. In ICCV, 2021.

