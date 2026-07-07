|Noname manuscript No. (will be inserted by the editor)|
|---|

## Instant3D: Instant Text-to-3D Generation

### Ming Li · Pan Zhou · Jia-Wei Liu · Jussi Keppo · Min Lin · Shuicheng Yan · Xiangyu Xu

# arXiv:2311.08403v2[cs.CV]29Apr2024

Received: date / Accepted: date

Abstract Text-to-3D generation has attracted much attention from the computer vision community. Existing methods mainly optimize a neural field from scratch for each text prompt, relying on heavy and repetitive training cost which impedes their practical deployment. In this paper, we propose a novel framework for fast text-to-3D generation, dubbed Instant3D. Once trained, Instant3D is able to create a 3D object for an unseen text prompt in less than one second with a single run of a feedforward network. We achieve this remarkable speed by devising a new network that directly constructs a 3D triplane from a text prompt. The core innovation of our Instant3D lies in our exploration of strategies to effectively inject text conditions into the

Ming Li Institute of Data Science, National University of Singapore and Sea AI Lab E-mail: ming.li@u.nus.edu

Pan Zhou School of Computing and Information Systems, Singapore Management University E-mail: panzhou@smu.edu.sg

Jia-Wei Liu Show Lab, National University of Singapore E-mail: jiawei.liu@u.nus.edu

Jussi Keppo Business School, National University of Singapore E-mail: keppo@nus.edu.sg

Min Lin Sea AI Lab, Singapore E-mail: linmin@sea.com

Shuicheng Yan Skywork AI, Singapore E-mail: shuicheng.yan@gmail.com

Xiangyu Xu (Project Lead, Corresponding Author) Xi’an Jiaotong University, China E-mail: xuxiangyu2014@gmail.com

network. In particular, we propose to combine three key mechanisms: cross-attention, style injection, and tokento-plane transformation, which collectively ensure precise alignment of the output with the input text. Furthermore, we propose a simple yet effective activation function, the scaled-sigmoid, to replace the original sigmoid function, which speeds up the training convergence by more than ten times. Finally, to address the Janus (multi-head) problem in 3D generation, we propose an adaptive Perp-Neg algorithm that can dynamically adjust its concept negation scales according to the severity of the Janus problem during training, effectively reducing the multi-head effect. Extensive experiments on a wide variety of benchmark datasets demonstrate that the proposed algorithm performs favorably against the state-of-the-art methods both qualitatively and quantitatively, while achieving significantly better efficiency. The code, data, and models are available at https://github.com/ming1993li/Instant3DCodes.

Keywords Text-to-3D Generation · Large-Scale Generative Models · Neural Radiance Fields

### 1 Introduction

Text-guided 3D content generation has immense potential across diverse applications, such as film production, animation, and virtual reality. Recently, this field has witnessed rapid and notable progress, as exemplified by the pioneering works including DreamFusion (Poole et al., 2022), Latent-NeRF (Metzer et al., 2023), SJC (Jain et al., 2022), and ProlificDreamer (Wang et al., 2023b). These works typically represent a target object with a randomly initialized Neural Radiance Field (NeRF) (Mildenhall et al., 2020) and then optimize the NeRF under the guidance of 2D

In this paper, we propose a novel framework, named Instant3D, for fast text-to-3D generation. It is able to generate a realistic 3D object that faithfully aligns with a given text description, all within a single forward pass of a neural network. Remarkably, this generation process takes less than one second (bottom of Figure 1). The proposed Instant3D is a feedforward neural network conditioned on the text prompt and produces a triplane representation (Chan et al., 2022) encapsulating the desired 3D object. Similar to DreamFusion (Poole et al., 2022), we use the Score Distillation Sampling (SDS) loss to train our network, eliminating the need for costly 3D training data.

NeRF

“a deer”

[Figure 1]

[Figure 2]

Rendering

Objective

...

[Figure 3]

Update NeRF

SOTA (> 10000 Iterations)

“a dog wearing a tie and wearing a beanie”

Conditional NeRF

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

While conceptually simple, this network is substantially challenging to design in practice. The main reason is that the SDS loss provides relatively weak supervision signals, making it difficult to learn an accurate connection between the text condition and the 3D output for a common condition model. This stands in contrast to training models that connect 2D images with texts, such as Stable Diffusion (Rombach et al., 2022) and CLIP (Wang et al., 2022b), where numerous image-text pairs can provide strong supervision signals.

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

Rendering

...

[Figure 15]

Our Approach (One Forward Pass)

- Fig. 1: Comparison of existing SOTA methods with our proposal. Existing methods optimize a randomly initialized NeRF from scratch for each text prompt, which usually requires more than 10,000 iterations, taking hours to converge. In contrast, our approach is designed to learn a text-conditioned NeRF, which takes much less training computational cost and has strong generalization ability on new prompts. It is able to generate a conditional 3D representation (triplane) of a 3D object for an unseen text prompt in one single pass of a feedforward network, taking about 25ms.

The challenge of weak supervision motivates us to devise more effective condition mechanisms that can better inject text information into the network, alleviating the difficulty of learning the relationship between the two drastically different modalities: text and 3D. As the core innovation of Instant3D, our proposed solution to this challenge involves the fusion of three modules: cross-attention, style injection, and token-toplane transformation. The integration of these components collectively ensures the precise alignment of the generated 3D output with the condition text prompt.

diffusion priors leveraging a pre-trained large-scale textto-image diffusion model (Rombach et al., 2022).

Moreover, we also propose a simple yet effective activation function, called scaled-sigmoid, to replace the original sigmoid function in NeRF, which speeds up the network convergence by more than ten times.

While achieving impressive results, these works face a critical challenge: they rely on an optimization-based learning paradigm, requiring thousands of iterations to optimize a NeRF for each new input (top of Figure 1). This incurs substantial computational costs and leads to notably slow response speed for practical text-to3D systems, often taking hours to process a single input (Poole et al., 2022; Wang et al., 2023b). Moreover, since the NeRFs are separately learned for different text inputs, they are not able to leverage general 3D priors shared across objects.

Meanwhile, text-to-3D methods often suffer from severe Janus (multi-head) problems. While the PerpNeg algorithm (Armandpour et al., 2023) can effectively remedy this issue for existing text-to-3D methods, it does not perform well in our context. We find that this is mainly due to the different degrees of multi-head effect exhibited by different 3D objects. As previous textto-3D approaches only train a single object at a time, one can easily tune the concept negation scale in the Perp-Neg algorithm to adapt to the specific multi-head level. However, in our framework, it is challenging to find a universal, optimum concept negation scale for all various training samples. To tackle this issue, we present an adaptive variant of the Perp-Neg algorithm which dynamically adjusts its concept negation scales

Note that Point-E (Nichol et al., 2022) abandons this optimization paradigm by directly training a diffusion model for 3D point cloud generation. However, these diffusion models require a large number of iterative diffusion steps, which inevitably leads to low efficiency. Besides, it is trained on millions of private 3D assets, which is not feasible for most researchers.

according to the severity of the multi-head effect during training, thereby significantly improving the performance.

Our approach demonstrates a powerful capacity to generate a high-quality 3D object consistent with a novel text prompt in only one second. We conduct extensive evaluations of the proposed algorithm on a wide variety of benchmark datasets, showing that the Instant3D performs favorably against the state-of-the-art approaches both qualitatively and quantitatively with much improved efficiency.

In summary, our contributions are as follows:

- – We make early explorations of fast text-to-3D generation with a single run of a feedforward neural network.
- – We propose a novel condition model that can effectively absorb text information, facilitating the establishment of the connection between text and 3D under weak supervision.
- – We present a new activation function, called scaledsigmoid, which significantly accelerates the training convergence.
- – We present an adaptive Perp-Neg algorithm to better tackle the Janus problem.

### 2 Related Works

Text-to-Image. Previously, the research in this field mainly concentrates on generating images belonging to a specific domain or distribution (Reed et al., 2016; Zhang et al., 2017; Xu et al., 2018; Qiao et al., 2019; Tan et al., 2019; Ruan et al., 2021). The literature is dominated by Generative Neural Networks (GANs) (Goodfellow et al., 2020) and various invariants of GANs are proposed. Recently, inspired by large-scale image-text pairs (Schuhmann et al., 2021, 2022) and generative models, text-to-image generation has shown unprecedented imaginations, i.e., synthesizing all sorts of imaginative images corresponding to text prompts. DALLE (Ramesh et al., 2021) and CogView (Ding et al., 2021) are developed based on an autoregressive architecture (Esser et al., 2021; Ramesh et al., 2021), while GLIDE (Nichol et al., 2021) employs a diffusion model (Song et al., 2020; Ho et al., 2020) conditioned on CLIP guidance (Radford et al., 2021). DALL-E-2 (Ramesh et al., 2022) also leverages diffusion priors to translate CLIP text embeddings to CLIP image embeddings, followed by synthesizing images from them. Imagen (Saharia et al., 2022) employs a large pre-trained language model to guide the reverse process of a diffusion model in pixel space. It is similar to Stable Diffusion (Rombach et al., 2022). Differently, the latter is

based on a large-scale UNet architecture and works in the vector-quantized discrete latent space, enabling an efficient sampling process.

Text-to-3D. Recent methods of text-to-3D generation without 3D supervision target to generate 3D objects corresponding to input prompts only with the guidance from CLIP or text-to-image models (Sanghi et al., 2022; Liu et al., 2022; Jain et al., 2022; Khalid et al., 2022; Poole et al., 2022; Lee and Chang, 2022; Yi et al., 2023). By leveraging CLIP embeddings to make the generated object closer to the text prompt, DreamFields (Jain et al., 2022) and CLIP-Mesh (Khalid et al., 2022) trigger the research in this field. They represent 3D objects by NeRFs and spherical meshes, respectively. PureCLIPNeRF (Lee and Chang, 2022) follows their paradigm except for replacing NeRFs or spherical meshes with grid-based representation (Sun et al.,

- 2022). Inspired by pre-trained large-scale text-to-image diffusion models, DreamFusion (Poole et al., 2022) proposes a score distillation sampling loss to distill 2D image priors into 3D generation process and achieves much better results than previous works. A concurrent work SJC (Wang et al., 2023a) presents a similar approach. Following this line, Latent-NeRF (Metzer et al.,
- 2023) proposes to learn 3D representations in the latent space instead of pixel space and incorporate more guidance like sketch shapes into the object generation. To solve the problem of low-diversity introduced by SDS, ProlificDreamer (Wang et al., 2023b) presents variational score distillation to construct multiple particles for a single prompt input, sampling the optimal 3D representation from the corresponding probabilistic distribution. These works mainly focus on optimizing an implicit 3D representation for a prompt. Given a new prompt, they need to repeat the optimization process, taking more than one hour. In contrast, our Instant3D is able to generate a high-quality 3D object for a new prompt in less than one second after training.

A concurrent work ATT3D (Lorraine et al., 2023) follows a similar paradigm to ours, learning a neural network for fast text-to-3D generation. Our Instant3D differs from ATT3D in that ATT3D employs a straightforward MLP to learn a hash grid (Mu¨ller et al., 2022), while we devise a novel decoder architecture with enhanced condition mechanisms and a scaled-sigmoid function to generate triplanes, which significantly improves the results. In addition, we present an adaptive Perp-Neg algorithm that effectively tackles the Janus problem. The proposed method achieves higher generation quality and more accurate text-3D alignment than ATT3D, as will be seen later in the experiment section. 3D Priors in Generation. Recently, there has been another line of research that introduces 3D pri-

ors into 3D generation. MVDiffusion (Tang et al., 2023) pioneers at this direction by proposing the correspondence-aware attention to maintain cross-view consistency primarily for text-to-panorama generation. MVDream (Shi et al., 2023) designs a multi-view diffusion model with 3D self-attentions. After training on a 3D dataset, the model can provide strong 3D diffusion priors via multi-view SDS. SweetDreamer (Li

- et al., 2023b) tunes a 2D diffusion model to produce canonical coordinate maps for a prompt, conferring

3D geometric consistency in generation. Additionally, there are other works investigating multi-view consistency for image-to-3D generation. SyncDreamer (Liu

- et al., 2023c) proposes a synchronized multi-view diffusion model with a 3D-aware interaction mechanism to correlate the features across different views. Wonder3D (Long et al., 2023) presents a multi-view crossdomain attention mechanism to improve the consistency and develops a geometry-aware normal fusion algorithm to extract high-quality surfaces. One-2-345 (Liu et al., 2023a) reconstructs 3D object meshes in a feedforward manner in 45 seconds based on SDFbased neural surface reconstruction and Zero123 (Liu et al., 2023b).

Specially, a concurrent work (Li et al., 2023a) proposes a two-stage pipeline of applying 3D priors from Objaverse (Deitke et al., 2023) for ensuring multiview consistency. They first train a multi-view diffusion model to generate a sparse set of view images for a given text and then optimize a large reconstruction model to directly regress the NeRF from the generated images. Its inference process takes 20 seconds for generating one object, much longer than ours (20ms). Moverover, our Instant3D does not rely on any 3D dataset and only applies 2D diffusion priors from a text-to-image model for 3D object generation.

### 3 Methodology

- 3.1 Preliminaries

- 3.1.1 Neural Radiance Field

NeRF (Mildenhall et al., 2020) is an implicit 3D representation widely used in neural inverse rendering. Given a NeRF, rendering an image for a camera view c involves casting a ray for each pixel of the image; these rays originate from the camera’s center of projection and extend through the pixel’s location on the image plane, reaching out into the 3D world. Sampled 3D points p along each ray are then passed through a MultiLayer Perceptron (MLP), which produces 4 scalar val-

ues as output: τ = fsoftplus (fMLP(p;θ)[1]), ρ = fsigmoid (fMLP(p;θ)[2 : 4]),

(1)

where τ ∈ R is volumetric density, indicating the opacity of the scene geometry at the 3D coordinate. ρ ∈ R3 is the albedo, capturing the intrinsic colors that define the scene. fMLP represents an MLP with parameters θ. [i : j] denotes extracting channels from i to j. fsoftplus is the Softplus activation function, ensuring positive volumetric density. fsigmoid is the sigmoid function, which confines the albedo to the range of [0,1]. These density and albedo values on a ray are then composited with volume rendering, producing the RGB value for the pixel. We use a shading formulation similar to DreamFusion (Poole et al., 2022) in rendering. Carrying out this process for all pixels on the image plane results in the final rendered RGB image denoted as gθ(c), which is parameterized by θ and conditioned on the camera view c.

3.1.2 Score Distillation Sampling (SDS)

Most existing text-to-3D methods are based on the SDS loss (Poole et al., 2022) powered by text-to-image diffusion models (Rombach et al., 2022). It facilitates the generation of a NeRF from a text prompt by enforcing that the rendered image from the NeRF at any viewpoint maintains semantic consistency with the given text prompt.

In detail, given a rendered image gθ(c), the SDS loss first introduces random noise ϵ ∼ N(0,I):

xt = √αtgθ(c) + √1 − αtϵ, (2) where α1:T is a predefined decreasing sequence, and the time step t is selected randomly. Subsequently, a pretrained diffusion UNet fdiffusion is employed to predict the noise in Equation 2 from xt, which is proportional to the negative score function (NSF) of the image distribution (Song et al., 2021):

ϵunc = fdiffusion(xt;t), ϵpos = fdiffusion(xt;y,t) − ϵunc, (3) ϵpredict = ϵunc + wguidanceϵpos,

where ϵunc corresponds to the NSF of the unconditional data distribution, and ϵpos is proportional to the residual NSF conditioned on the text embedding y. ϵpredict is based on a classifier-free guidance formulation (Ho and Salimans, 2022), indicating the direction for updating the image to match the text description y, and wguidance denotes the guidance weight.

arabbitwearingabackpack

Token Embeddings

[Figure 16]

Token-to-Plane

Decoder

푁  

[Figure 17]

  8 8

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Projection

Reshape

[Figure 22]

KV Q

[Figure 23]

[Figure 24]

Transformer Layer

CLIP Text Encoder

AdaIN

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

Class Embedding

[Figure 29]

[Figure 30]

Projection

Reshape

Transformer Layer

[Figure 31]

[Figure 32]

Concat

Triplane

Gaussian Noise

Style Injection

[Figure 33]

[Figure 34]

ScaledSigmoid

[Figure 35]

Albedo Volume Rendering

VAE Encoder

MLP

UNet

SDS

Softplus

Density

Stable Diffusion

Rendered Output CLIP Image Encoder

KV Q

Adaptive Instance Normalization

Cross-Attention

AdaIN

Cosine Distance

CLIP

Class Embedding

[Figure 36]

- Fig. 2: Overview of the proposed Instant3D, which applies a conditional decoder network to map a text prompt to a corresponding triplane. Three condition mechanisms, i.e., cross-attention, style injection, and token-to-plane transformation, are seamlessly combined to bridge text and 3D, tackling the issue of weak supervision from SDS (Poole et al., 2022). Given a random camera pose, a 2D image is rendered from the conditioned triplane through coordinate-based feature sampling, point density and albedo prediction, and differentiable volume rendering. For albedo activation, we propose a scaled-sigmoid function, effectively accelerating the training convergence. During training, the view image is first diffused by adding random noise and then fed into a pretrained UNet conditioned on the text prompt for denoising, which provides the gradient of the SDS loss. We also present an adaptive Perp-Neg algorithm to better solve the Janus problem in our framework. During inference, our Instant3D can infer a faithful 3D object from an unseen text prompt in less than one second.

Finally, the gradient of the SDS loss with respect to the NeRF parameters θ can be written as:

fast text-to-3D generation. An overview of Instant3D is shown in Figure 2.

In particular, we design a conditional network fInstant(p,y;θ), which is conditioned on the text feature y. We obtain the text feature with the pretrained text encoder of CLIP (Wang et al., 2022b). The conditional network can be formulated as:

∂gθ(c) ∂θ

∇θLSDS(θ) = Et,ϵ w(t)(ϵpredict − ϵ)

, (4)

where w(t) is a weight function depending on αt.

- 3.2 Conditional NeRF Generation

By combining NeRF and SDS, existing methods (Poole et al., 2022; Metzer et al., 2023; Wang et al., 2022a) have demonstrated impressive results for text-to-3D generation. However, these methods suffer from an important drawback that they need to train a new NeRF for each new text prompt, leading to low efficiency. To address this issue, we propose to learn a single feedforward network Instant3D for multiple text prompts, enabling

fInstant(p,y;θ) = fˆMLP (fsam (p,fdec(y;θ1));θ2). (5) Inspired by StyleGAN (Karras et al., 2019), we adopt a decoder architecture fdec to transform the text condition y into a triplane (Chan et al., 2022). Then we sample a feature vector for each 3D point p from the triplane using fsam, which projects p onto each plane, retrieves the corresponding feature vector via bilinear interpolation, and aggregates the three feature vectors via concatenation (Chan et al., 2022). This point feature is subsequently sent to an MLP fˆMLP to produce

[Figure 37]

[Figure 38]

[Figure 39]

cross-attention module can be formulated as:

QKT √

Attention(Q,K,V ) = Softmax

V, (6)

d

where d is the channel dimension of K.

Cross-Attention + Style Injection + Token-to-Plane Transformation

- Fig. 3: Integrating three condition mechanisms produces high-quality results faithful to the text prompt. The prompt here is “a teddy bear sitting in a basket and wearing a scarf and wearing a baseball cap”.

the density and albedo values. The parameters of the conditional model θ include both the decoder parameters θ1 and MLP parameters θ2. Replacing fMLP with fInstant in Equation 1 enables conditional NeRF prediction, where different y yields different NeRF outputs, and the learned model parameters θ are shared among diverse text prompts.

The key component in Equation 5 is the decoder network which plays the role of transferring text information into 3D. Developing a decoder within our framework, despite seeming straightforward, presents significant obstacles. A key hurdle is the weak supervision signals from the SDS loss, as outlined in Equation 4, making it difficult to establish a reliable link between the textual conditions and the 3D output. This situation starkly differs from that of text-to-image models like Stable Diffusion (Rombach et al., 2022) and CLIP (Wang et al., 2022b), which benefit from the strong guidance provided by numerous image-text pairs.

The weak supervision issue demands a more effective conditional model that can better absorb text information, alleviating the difficulty of bridging text and 3D. To this end, we introduce an integrated solution combining three condition mechanisms: cross-attention, style injection, and token-to-plane transformation.

- 3.2.1 Cross-Attention

Cross-attention is one of the most important strategies for information interaction across various modalities. Following the prominent success of text-to-image models, we apply cross-attention to inject text descriptions into the 3D representation. Similar to Stable Diffusion (Rombach et al., 2022), we employ the multihead cross-attention module (MHCA) to fuse text embeddings with feature maps of the decoder. Specifically, the query (Q) is projected from the feature maps, while the key (K) and value (V ) are projected from text embeddings. In principle, the interacted output of the

3.2.2 Style Injection

Surprisingly, the model with only the cross-attention mechanism completely fails, yielding a meaningless output as shown in Figure 3. Instant3D aims to generate 3D objects whose 2D renderings closely align with the synthesized images of Stable Diffusion for the corresponding prompts. Inspired by the GAN literature (Arjovsky and Bottou, 2016), we hypothesize that this optimization failure is mainly due to that there is a significant discrepancy between the 2D renderings from Instant3D and the images generated by Stable Diffusion during the initial stage of training. Arjovsky and Bottou (2016) highlight that the optimization gradient is unreliable when the generated and target distributions are disjoint. By integrating Gaussian noise into the network, the overlap between the rendered and targeted distributions can be enhanced, thereby leading to more meaningful gradients during training. To this end, we introduce the Adaptive Instance Normalization (AdaIN) to inject random noise into the decoder.

In addition to noise, our style injection module also encodes text features into the decoder to improve the controllability of the 3D generation process. As shown in Figure 2, it starts with a linear projection layer to more compactly represent the text embeddings. Subsequently, we apply a self-attention based Transformer layer to adapt the feature to the style space. The output text features are flattened to a vector and concatenated together with random Gaussian noise to generate the style vector. Finally, the output vector of the style injection module is embedded into the feature maps of our decoder network with AdaIN in Figure 2.

Specifically, one linear layer is applied to project the output vector of the style injection module into styles l = (ls,lb) that control AdaIN operations after each convolution block of the decoder network. The AdaIN operation can be formulated as

mi − µ(mi) σ (mi)

AdaIN(mi,l) = ls,i

+ lb,i, (7)

where each input feature channel mi is normalized independently, and then scaled and biased by the corresponding scalar components from the style l. Thus the dimensionality of l is twice the number of the input feature channels.

2.0

3D objects to reflect all mentioned entities and maintain their inherent relationships, matching the tokenlevel granularity of the text prompt. Thus, we use the token embeddings as the text feature to guide our 3D generation, which leads to superior generation results on novel prompts.

1.0

= 0.5 = 0.7

= 0.5 = 0.7

- = 0.9

- = 1.0

- = 0.9

- = 1.0

1.5

Response

Response

1.0

0.5

0.5

0.0

0.0

5 3 1 1 3 5 Input

5 3 1 1 3 5 Input

- 3.3 Scaled-Sigmoid

As shown in Equation 1, the sigmoid function is usually used to ensure the albedo prediction falls within the range of [0,1]. However, we observe that applying the conventional sigmoid to albedo causes difficulties in training convergence. A deeper analysis reveals that the MLP output may exceed the high-gradient region around zero, leading to gradient vanishing. This phenomenon significantly impedes the learning of the 3D representation, resulting in extended training durations and occasional convergence failures.

To address this issue, we stretch the high-gradient region by multiplying the input with a fractional coefficient α:

fˆsigmoid(x) = fsigmoid(αx), α ∈ (0,1]. (8) The behavior of fˆsigmoid with different α is shown in Figure 4 (left). Note that the stretching operation reduces the gradient around x = 0 by a factor of α. To counter this effect, we further apply a scaling factor of 1

α to the stretched sigmoid:

f˜sigmoid(x) =

1 α

fsigmoid(αx), α ∈ (0,1]. (9)

We refer to f˜sigmoid as the scaled-sigmoid, which is visualized in Figure 4 (right). Since f˜sigmoid can surpass the [0,1] bounds, we employ an annealing strategy: initializing α with a small value (0.5 in our experiment) to accelerate training; over time, we gradually increase α to 1, ensuring the output albedo lies in [0,1].

- 3.4 Adaptive Perp-Neg Algorithm

- Fig. 4: Comparison of the original sigmoid function with its stretched variant (left) and the proposed scaledsigmoid function (right). With α < 1.0, the scaledsigmoid function possesses a broader high-gradient region, which accelerates training.

- 3.2.3 Token-to-Plane Transformation

Existing decoder architectures typically start with a learnable planar base tensor (Karras et al., 2019, 2020, 2021), which is kept constant after training. In contrast, we reformulate this paradigm by predicting the base tensor from the text embedding with a token-to-plane transformation module as shown in Figure 2. Consequently, the base tensor evolves from a static entity to a dynamic one, significantly deviating from the common practices of StyleGAN, which can better convey condition information to the decoder.

As illustrated in Figure 2, we employ a self-attention based Transformer layer to transform sequential token embeddings into a 2D feature map. Specifically, we first reduce the dimension of tokens with linear projection for more efficient computations. Then a multi-head selfattention (MHSA) layer (Vaswani et al., 2017) is performed to enable global information interaction among tokens. We further apply an MLP to refine the token embeddings and then reshape them to a feature map with the shape of 8 × 8. The token-to-plane transformation not only transforms the sequential tokens into a desired planar shape, but also shifts the CLIP embeddings to a feature space suitable for text-to-3D generation.

As shown in Figure 3, the integration of all three condition mechanisms collectively ensures the precise alignment of the generated 3D output with the condition text prompt.

Previous Perp-Neg Algorithm. Existing text-to-3D algorithms often face a challenging Janus (multi-head) problem (Metzer et al., 2023). Rather than generating a coherent 3D output, the learned 3D object tends to exhibit repeated front views at different angles, as the front view has been more prominent in the training data of text-to-image models. For instance, when generating a 3D animal, the resulting output often has multiple faces without capturing the natural side and back views of the animal.

Text Feature. The text encoder of CLIP (Radford et al., 2021) generates both class and token embeddings for a text prompt. The class embedding usually contains sentence-level semantic information. In contrast, the token embeddings contain rich word-level semantic details of a prompt, which can better represent items in a scene and are commonly used in text-to-image generation (Rombach et al., 2022). We aim for the generated

To address this issue, Armandpour et al. (2023) propose a Perp-Neg algorithm: ϵpredict = ϵunc + wguidance ϵpos − wnegϵneg⊥ , ϵneg = fdiffusion(xt;yneg,t) − ϵunc,

(10)

where ϵpredict is used in Equation 4 to calculate the gradient of the SDS loss. yneg denotes the token embeddings of negative prompts. For example, when rendering images at the back view, the negative prompts refer to side and front views. ϵneg is the residual NSF of negative text prompts, penalizing the generation of wrong views at inappropriate angles. ⊥ indicates projection onto the direction perpendicular to ϵpos. wneg represents the concept negation scale, controlling the degree to which Equation 10 penalizes the Janus problem.

Adaptive Adjustment. Selecting an appropriate value for wneg is critical for the success of Perp-Neg. If it is too small, the Janus effect will not be sufficiently mitigated. If it is too large, it may cause the flat-head issue, where the entire head is squeezed into a flat plane. This is because the yneg in Equation 10 punishes front views when rendering in side angles, and a flat head ensures no face can be seen from a side perspective. For existing optimization-based methods, it is feasible to manually adjust wneg for each new text prompt. However, as our model is trained on an extensive set of text prompts, it is challenging to find a universally optimum concept negation scale for all training samples. To overcome this challenge, we propose an adaptive Perp-Neg algorithm to dynamically adjust the value of wneg according to the severity of the multi-head problem for different samples.

Specifically, we design the negation scale in our adaptive Perp-Neg as

wneg = wmin + C · ∆w, (11) where wmin is the minimum scale, and ∆w describes the variation range of wneg. The adaptive parameter C ∈ [0,1] measures the severity of the multi-head problem. When the multi-head effect is less pronounced (C = 0), we apply a small punishment in Equation 10 with wneg = wmin. Conversely, when the multi-head effect is highly severe (C = 1), the negation scale increases to wmin + ∆w.

To assess the severity of the multi-head problem for a given rendered image Iv at the current viewpoint v, we define C as:

n

1 4n

[1 − cos(v − vi)]·[1 + ⟨ϕ(Iv),ϕ(Iv

)⟩], (12)

C =

i

i=1

where we randomly select n additional views {vi}ni=1 and measure the similarity between Iv and each Iv

in

i

feature space. The feature extractor ϕ is the image encoder of CLIP (Wang et al., 2022b), and ⟨·,·⟩ represents the cosine similarity. The rationale for this design is that a higher similarity between different views indicates more severe multi-head effect, and conversely, a lower similarity value among different views suggests a less significant Janus problem. Besides, we weight the contribution of different view angles in Equation 12 with 1 − cos(v − vi). This weighting mechanism takes into account that high similarity between views that are farther apart indicates a more severe multi-head effect.

While Equation 12 offers an effective approach for adjusting the negation scale adaptively, it comes with a notable computational overhead. This is primarily due to the extra cost involved in rendering {Iv

i}ni=1. To address this issue, we adopt a simplified approach by setting n = 1 and utilizing the rendered image from the previous iteration as the additional view. To facilitate this process, we maintain a cache that stores image features ϕ(Iv) for all training samples. This cache enables us to reuse the feature representations extracted during the preceding training steps. The features in the cache are dynamically updated, where the current feature is added while the old feature is removed. This approach allows us to implement adaptive Perp-Neg without introducing additional computational costs.

- 3.5 Training Objective

Except for the SDS loss in Section 3.1.2, the proposed Instant3D also incorporates a CLIP loss LCLIP. As shown in Figure 2, we employ the CLIP image encoder to extract image features from the rendered image and use the class embeddings from the CLIP text encoder to represent the text prompt. We then define the CLIP loss as the cosine distance between the image and text features to align the rendered image more closely with the input prompt. Remarkably, these two training objectives allow learning a text-to-3D model without requiring any 3D data.

4 Experiments

- 4.1 Datasets

As this work is a pioneering effort for fast text-to-3D generation, we devise three benchmark prompt sets to comprehensively evaluate the proposed framework: Animals, Portraits, and Daily Life.

Animals. This prompt set is constructed by combining several keywords, where each keyword can be randomly chosen from a predefined set of candidates:

- – species: wolf, dog, panda, fox, civet, cat, red panda, teddy bear, rabbit, and koala.
- – item: in a bathtub, on a stone, on books, on a table, on the lawn, in a basket, and null.
- – gadget: a tie, a cape, sunglasses, a scarf and null.
- – hat: beret, beanie, cowboy hat, straw hat, baseball cap, tophat, party hat, sombrero, and null.

A sample prompt is structured as “a species sitting item and wearing gadget and wearing a hat”. The null means the keyword is absent from the prompt. In total, the Animals set consists of 3,150 prompts. We randomly allocate 60% of them for training and the remaining 40% for testing.

Portraits. We construct this prompt set by describing portraits of commonly seen figures with various hats and facial expressions. The keywords and their available choices include:

- – figure: a white man, a white woman, a boy, a girl, an elderly man, an elderly woman, a black woman, a black man, Obama, Kobe.
- – hat: Santa hat, peaked cap, steampunk hat, crown.
- – expressing: laughing, crying, grinning, singing, shouting, looking ahead with a very serious expression, opening mouth wide in shock, angry, talking, feeling sad.

These keywords are combined in the manner of “figure wearing a hat is expressing”. The Portraits set consists of 400 prompts in total, with 60% for training and 40% for testing.

Daily Life. The above two prompt sets are constructed by combining phrases from a limited range of candidates in a structured manner. To demonstrate that our approach can work well in a more general setting where more complex and varied sentences are included, we contribute another dataset named Daily Life by collaborating with ChatGPT (Brown et al., 2020). ChatGPT is a large language model designed to converse with humans and can produce high-quality responses when given appropriate context. We craft our question as “generate a text describing a man or woman’s appearance and daily activities. The examples are as follows: a tall man wearing sunglasses is eating a hamburger, a slim woman wearing a baseball cap is reading a book, a beautiful woman wearing a tie is watching a TV ”. By repeatedly submitting this question to ChatGPT, we collect more than 17,000 answers as our prompt set. These prompts exhibit a wide variety of structures and contain 3,135 unique words in total, demonstrating enhanced complexity and diversity. We train our framework on the whole set and perform evaluation by generating additional new prompts via ChatGPT.

- 4.2 Implementation Details

Camera Setting. For NeRF rendering, we randomly sample a camera pose in a spherical coordinate system with the polar angle γ ∈ [25◦,110◦], azimuth angle ϕ ∈ [0◦,360◦], and radius from the origin r ∈ [3.0,3.6]. The camera FOV is randomly selected between 70◦ and 80◦. Neural Rendering. For volumetric rendering, 128 points are sampled along each ray. Among them, 64 points are uniformly sampled, and the remaining 64 are sampled based on the importance distribution along the ray. The resolution of rendered images is 64×64 during training and increased to 256 × 256 for testing.

Architecture. Our decoder network is composed of attention blocks and convolution blocks similar to Stable Diffusion (Rombach et al., 2022). More details can be found in the appendix.

Optimization. We use the Adam optimizer (Kingma and Ba, 2014) with a learning rate of 1e-4, β1 = 0.9, β2 = 0.99 and a zero weight decay. All experiments are conducted on NVIDIA A100 GPUs. Training on Animals and Daily Life prompt sets is performed on 8 GPUs with a batch size of 96, which takes about 8 hours and 60 hours, respectively. Training on Portraits uses 4 GPUs with a batch size of 48, taking about 3.5 hours. For inference, given a new text prompt, the generation of the 3D triplane takes about 25ms, and the rendering of one view image at a resolution of 256 × 256 takes about 0.5s.

- 4.3 Comparison with the State-of-the-Art Methods

Qualitative Comparison. We compare with the state-of-the-art text-to-3D methods including TextMesh (Tsalicoglou et al., 2023), SJC (Wang et al.,

- 2023a), DreamFusion (Poole et al., 2022), Latent-NeRF (Metzer et al., 2023) and ProlificDreamer (Wang et al.,
- 2023b). Note that these methods typically take more than an hour to optimize a NeRF for an input prompt, while our framework can generate a 3D object in under a second.

Meanwhile, since our model is trained on a large set of text inputs, it is able to leverage general 3D priors shared across objects. Therefore, the generated objects by Instant3D have much higher quality than the baselines as illustrated in Figure 5 and 6. Besides, our method captures all elements specified in the prompts as well as the relationships between them. In contrast, objects produced by the baseline approaches frequently miss important items, and their inter-item relationships can appear disorganized.

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

Instant3DDreamFusionSJCLatent-NeRFPoint-ETextMeshProlificDreamer

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

“a cat sitting on books and wearing sunglasses and wearing a cowboy hat”

“a fox sitting in a bathtub and wearing a straw hat”

“a wolf wearing a scarf and wearing a party hat”

“a civet sitting in a bathtub and wearing a beanie”

“a koala sitting on a table and wearing a cape and wearing a baseball cap”

“a teddy bear sitting on books and wearing a tie and wearing a tophat”

“a red panda sitting in a bathtub and wearing a cowboy hat”

- Fig. 5: Qualitative comparison with the state-of-the-art methods including TextMesh (Tsalicoglou et al., 2023), SJC (Wang et al., 2023a), DreamFusion (Poole et al., 2022), Latent-NeRF (Metzer et al., 2023), ProlificDreamer (Wang et al., 2023b), and Point-E (Nichol et al., 2022) on the Animals dataset. The proposed Instant3D achieves higher-quality results while being much more efficient than the baselines.

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

TextMesh SJC DreamFusion Latent-NeRF ProlificDreamer Instant3D

- Fig. 6: Qualitative comparison against the baseline approaches on the Portraits (top) and Daily Life (bottom) datasets. The prompts for the top and bottom are “a white woman wearing a Santa hat is grinning” and “a handsome man wearing a leather jacket is riding a motorcycle”, respectively.

[Figure 101]

(a) “a teddy bear sitting on a stone and wearing a scarf and wearing a baseball cap”

[Figure 102]

(b) “a panda sitting in a basket and wearing a straw hat”

- Fig. 7: Visual results on the Animals set, which are inferred by our Instant3D for novel prompts. The results demonstrate accurate text-3D alignment and satisfying multi-view consistency.

[Figure 103]

(a) “a white man wearing a steampunk hat is singing”

[Figure 104]

(b) “Kobe wearing a crown is crying”

- Fig. 8: Visual results on the Portraits set. Our Instant3D accurately generates the figures, items, and expressions described in the text prompts, while showing favorable multi-view consistency.

In addition, we provide a comparison with PointE (Nichol et al., 2022), which directly trains a 3D diffusion model. This method is based on a substantially different paradigm and requires millions of 3D-text pairs

that are not available to the public. In contrast, our algorithm does not require any 3D data. Moreover, due to the limitation of the 3D datasets, especially in terms of scale and diversity, Point-E can only generate simple

[Figure 105]

[Figure 106]

(a) “a muscular man is exercising in a fitness room”

[Figure 107]

[Figure 108]

(b) “a slim woman is trying on a dress”

[Figure 109]

[Figure 110]

(c) “a bearded man wearing a backpack is climbing a mountain”

[Figure 111]

[Figure 112]

(d) “a man wearing a hat is mowing the lawn”

- Fig. 9: Visual results on the Daily Life set. For each prompt, four views and their normal images are shown, demonstrating that our Instant3D is able to generate complex and diverse objects with accurate geometric details for novel prompts.

objects and cannot perform well on more intricate text prompts as shown in Figure 5.

Finally, we present more visual examples from the Animals, Portraits, and Daily Life datasets in Figure 79. The proposed Instant3D consistently produces highquality text-to-3D results with favorable multi-view consistency. Notably, its performance on the challenging Daily Life dataset (Figure 9) underscores its capability to handle intricate real-world text prompts.

Computation Costs. To evaluate the training costs of different methods, we report the number of rendered view images per prompt on average (Views-PP) vs. CLIP retrieval probability (CLIP-RP) on the Animals set. Views-PP is computed as:

Iterations × Batch Size Number of Prompts

Views-PP =

, (13)

which essentially measures the computation cost as the number of images rendered for each prompt during training. CLIP-RP is defined as the average probability of assigning the correct prompt to a rendered image among a set of distraction texts (called the query set). The similarity between a text and an image is measured by CLIP text and image encoders. For each prompt, the CLIP-RP is averaged over four view images rendered

| || |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>ATT3D<br><br>Instant3D| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| |TextMesh<br><br>ProlicDreamer<br><br>SJC<br><br>DreamFusion Latent_-NeRF Instant3D<br><br>| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.20

0.25

0.15

0.20

CLIP-RP

CLIP-RP

0.10

0.15

0.05

0.10

0.00

500 5500 10500 15500 Views-PP

500 5500 10500 15500 Views-PP

SOTA ATT3D

Fig. 10: Computation cost comparison against the baseline methods. Left: comparison with the state-of-theart methods on the Animals dataset. Right: comparison with a concurrent approach ATT3D on their dataset. Views-PP is the number of rendered images per prompt in training, and CLIP-RP is the retrieval probability computed by CLIP encoders.

from distinct camera poses as in DreamFusion (Poole et al., 2022). All prompts of the Animals set are used as the query set. As shown in Figure 10, the generation quality generally gets better as the training iteration increases, while the improvement speed of the baseline methods is much slower than that of our Instant3D. Notably, the baselines can only achieve an ac-

Table 1: User study for comparing our method with the baseline approaches in terms of realism and text3D consistency.

Table 2: User preference study for comparing our results against those of ATT3D in terms of image realism and text-3D consistency.

Method TextMesh SJC DreamFusion Latent-NeRF ProlificDreamer Instant3D Realism 4.22% 4.95% 6.32% 7.49% 1.74% 75.28%

Consistency 2.35% 3.91% 3.69% 5.10% 1.33% 83.62%

Method ATT3D Instant3D Realism 8.68% 91.32%

Consistency 13.23% 86.77%

ceptable CLIP-RP after 10,000 iterations per prompt. In contrast, our method obtains a much higher CLIPRP score in only 2,000 iterations per prompt.

User Preference Study. We conduct a user study to evaluate different methods based on subjective perceptual quality assessment. We show users videos rendered from multiple views of objects, generated by different methods using the same text prompt. Users are asked to select the result that most closely matches the description in terms of realism and consistency. Each prompt is evaluated by at least 3 different users. As shown in Table 1, most users prefer our results to those generated by baselines, highlighting the effectiveness of Instant3D in achieving high fidelity and accurate text-3D alignment.

- 4.4 Comparison with ATT3D

bers from the original paper (Lorraine et al., 2023). Notably, our Instant3D requires fewer renderings per prompt during optimization, yet achieves superior text3D alignment, underscoring its efficiency and effectiveness in high-quality text-to-3D generation.

User Preference Study. We also conduct user studies to compare ATT3D and our approach based on user preferences. We show users videos rendered from multiple views of objects generated by two methods for the same text prompt. We ask them to select the result that has better quality in realism and text-3D consistency. To enhance the reliability of our study and mitigate variance, each prompt is evaluated by at least 3 different users. As shown in Table 2, the results of our Instant3D are clearly preferred over those of the ATT3D in terms of both standards, highlighting the effectiveness of the proposed algorithm.

In this section, we compare our method with a concurrent work ATT3D (Lorraine et al., 2023). Considering that the code of ATT3D is not publicly available, we retrain and evaluate our model on the dataset of ATT3D composed of 2,400 texts for a better comparison. Although ATT3D also aims for fast text-to-3D generation similar to this work, it requires per-prompt finetuning after each network inference, leading to lower efficiency. Moreover, with the techniques proposed in Section 3, our network is more effective than the model of ATT3D, which significantly improves the generation quality.

Qualitative Comparison. We compare our generated objects with those of ATT3D in Figure 11. The animals generated by ATT3D are blurry and lack essential details, and some described items in the prompts are not distinguishable. Moreover, the interactions between animals and their holding items could be disordered, for example, the erroneous orientations of the guitar and katana. In contrast, our Instant3D consistently produces sharp renderings and ensures plausible interactions between entities.

Computation Costs. We compare our computation cost with ATT3D by reporting Views-PP vs. CLIP-RP in Figure 10. As the code of ATT3D is not accessible to the public, we directly extract the reported num-

4.5 Ablation Study

We conduct extensive ablation studies on the Animals set to investigate the effectiveness of the key components in our network, including the style injection module, token-to-plane transformation, scaled-sigmoid function, and the adaptive Perp-Neg algorithm.

Effectiveness of Style Injection. To evaluate the importance of our style injection module, we evaluate two ablated versions: one without Gaussian noise and the other excluding token embeddings, whose results are shown in Figure 12.

Comparing the top and bottom rows, it is evident that adding Gaussian noise is instrumental in circumventing the training failure issue, which prevents generating null or meaningless outputs. The middle and bottom rows underscore the significance of token embeddings in the style injection module, which better inject textual conditions into the triplane generation process, leading to improved text-3D consistency under weak supervision signals.

Effectiveness of Token-to-Plane Transformation. To understand the effect of our token-to-plane transformation, we replace it with a learnable constant to feed into the decoder. As illustrated in Figure 13 (top),

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

ATT3DInstant3D

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

“a lion playing the guitar wearing a suit wearing a baseball cap”

“a bear holding a book wearing a sweater wearing a baseball cap”

“a lion holding a shovel wearing a cape wearing a tophat”

“a bear riding a motorcycle wearing a sweater wearing a baseball cap”

“a lion wielding a katana wearing a cape wearing a baseball cap”

“a raccoon playing the guitar wearing a suit wearing a baseball cap”

- Fig. 11: Comparison with the concurrent work ATT3D. The text prompts are given in the last row. Our generated 3D objects contain more details and are more distinguishable with better interactions between animals and items.

w/oNoiseCompletew/oTokenEmbeddings

[Figure 125]

[Figure 126]

[Figure 127]

- Fig. 12: Effectiveness of the proposed style injection module. Top: without Gaussian noise; middle: without token embeddings; bottom: full model. The prompt is “a wolf sitting on books and wearing a tie and wearing a party hat”. The comparison shows that the incorporation of Gaussian noise prevents training failure, and the token embeddings contribute to better text-3D consistency.

objects generated without token-to-plane transformation cannot be well controlled by the condition text prompts. For example, the expected hat is absent in the first column, and the bathtub is missing in the third column. In contrast, our full approach (bottom row) ensures all described items in the prompts are accurately generated and interact in a plausible manner.

Effectiveness of Scaled-Sigmoid. In Section 3.3, we introduce a scaled-sigmoid function for NeRF albedo prediction to accelerate the training process. To verify this design, we provide a comparison between the scaled-sigmoid and the conventional sigmoid function. Their corresponding generation processes for the same

prompt are shown in Figure 14. The numbers below images are the training iterations in terms of Views-PP. It is evident that the 3D generation driven by the standard sigmoid activation evolves much slower than that of the scaled-sigmoid. Notably, even with ten times training iterations of the scaled-sigmoid, the network with the conventional sigmoid function still does not achieve satisfactory convergence. The evolution process of the generated 3D object also shows that the scaled-sigmoid enables a more significant capability of our model in adjusting the output to match the text description.

Effectiveness of Adaptive Perp-Neg. As detailed in Section 3.4, we propose the adaptive Perp-Neg al-

[Figure 128]

[Figure 129]

[Figure 130]

w/owith

[Figure 131]

[Figure 132]

[Figure 133]

“a civet sitting in a basket and wearing a straw hat”

“a cat wearing a tie and wearing a sombrero”

“a koala sitting in a bathtub and wearing a sombrero”

- Fig. 13: Effectiveness of the proposed token-to-plane transformation. Top: without token-to-plane transformation; bottom: full model.

gorithm to address the Janus problem commonly encountered in tex-to-3D. To understand the benefits of this algorithm, we provide a comparison against the original Perp-Neg algorithm where wneg remains fixed throughout training. As shown in Figure 15 (top), it is not feasible to find a universally optimum value for all objects: the Janus problem is not fully resolved in the first column (three feet for the teddy bear), which indicates the wneg is too small; but for the other two columns, the Janus problem has been overly punished, evidenced by flat faces, which indicates the wneg is too large. In Figure 15 (bottom), as our adaptive Perp-Neg can dynamically adjust wneg for each object according to the severity of the Janus problem, it effectively overcomes the above challenge and produces higher-quality results.

### 5 Discussions

Relationship with Baseline Methods. Similar to existing SDS-based text-to-3D models (Poole et al., 2022; Wang et al., 2023b), we aim to generate one specific object for one given text prompt. The difference is that we amortize optimization over text prompts by training on many prompts simultaneously with a unified model, rather than optimizing separately for each prompt. This allows us to share computation across a prompt set, resulting in faster training compared to perprompt optimization as demonstrated in Figure 10.

Diversity. Our model is designed to produce a single, consistent output for a given text prompt, which is a different objective from general 2D generative models that aim to model the entire data distribution and generate different outputs for the same text prompt. This task setting is similar to that described in (Lorraine et al., 2023).

While our main goal is not to generate multiple outputs for a single prompt, it is worth noting that it is straightforward to combine the proposed Instant3D with the post-processing step to achieve more diverse results. As shown in Figure 16, we can easily generate text-to-3D results with different styles for the same prompt by including more detailed text descriptions in the refinement stage. Similar to ProlificDreamer (Wang et al., 2023b), we use DMTet (Shen et al., 2021) for refinement.

Fine-Grained Consistency. Fine-grained consistency between text and 3D is often of interest to the research community and can be useful for applications requiring precise control over the generated 3D models. As demonstrated in Figure 17, our model can effectively change the animal species in the text prompt while maintaining other items in the scene. This capability reflects our model’s strong understanding of the individual components described in the text, allowing it to accurately modify specific aspects of the 3D object based on the given prompt.

Limitations. As an early exploration of fast text-to-3D generation, we verify the proposed Instant3D on three compositional datasets and one open-world prompt set, i.e., Daily Life. However, compared with the dataset scale of text-to-image synthesis, these prompt sets are still relatively small. This can be attributed to two factors. On one hand, it is difficult and time-consuming to collect a large dataset with coupled prompts. On the other hand, the computation burden of training a textto-3D network is thought to be much heavier than that of a text-to-image model due to the higher dimension of 3D objects. These can be solved by more efficient 3D representation methods and powerful computation devices in the future.

Negative Societal Impacts. The fast text-to-3D generation approach proposed in this paper aims to infer a faithful 3D object for a testing prompt in 20ms. We note that it could be potentially applied to unsafe scenarios such as generating violent or sexual content through third-party fine-tuning like other existing methods. However, it actually inherits the data bias from the used text-to-image model, i.e., Stable Diffusion (Rombach et al., 2022), whose training text-image pairs should be carefully filtered. Besides, there is also growing interest in focusing on such potential safety problems in the research community.

### 6 Conclusion

In this paper, we demonstrate the feasibility of fast text-to-3D generation with a one-pass feedforward net-

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

Sigmoid

[Figure 141]

[Figure 142]

[Figure 143]

###### 0 3500 7000 10500 14000 17500 21000

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

Scaled-Sigmoid

0 350 700 1050 1400 1750 2100

- Fig. 14: Evolution process of the generated objects with the conventional sigmoid activation (top) and the scaledsigmoid activation (bottom) for albedo prediction. The number below each image is the corresponding training iteration in terms of Views-PP. The prompt is “a dog sitting on a table and wearing a tie and wearing a beanie”. The scaled-sigmoid function significantly accelerates the training convergence and enables better capability in text-3D alignment.

OriginAdaptive

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

- Fig. 15: Comparison of the generated objects by using the original Perp-Neg algorithm (top) and our adaptive variant (bottom). In the top row, the teddy bear has three feet, while the dog and koala already display severe flat faces. This suggests that the original algorithm can not simultaneously address Janus problem for multiple objects. In contrast, our adaptive algorithm adjusts the concept negation scales of various objects during the generation process and significantly overcomes the above challenge.

ate faithful 3D objects for novel prompts in less than one second. Moreover, our approach shows much better optimization efficiency. We expect this work to inspire further explorations in fast text-to-3D generation.

### Acknowledgement

Ming Li is funded by the ISEP-IDS PhD scholarship in NUS. Pan Zhou is supported by the Singapore Ministry of Education (MOE) Academic Research Fund (AcRF) Tier 1 grant. Xiangyu Xu is supported by NSFC (62302385) and the computational resources provided by the HPC platform of Xi’an Jiaotong University.

### References

Arjovsky M, Bottou L (2016) Towards principled methods for training generative adversarial networks. In: International Conference on Learning Representations

work. To overcome the learning difficulties brought by weak supervision from SDS loss, we integrate three condition mechanisms, i.e., cross-attention, style injection, and token-to-plane transformation, to better embed text prompts into a decoder network. We also present a scaled-sigmoid function for albedo activation, which accelerates the optimization convergence by more than ten times. In addition, we propose an adaptive Perp-Neg algorithm, which effectively tackles the Janus problem in our new paradigm without introducing extra computation costs. Extensive experiments on four prompt sets show that our Instant3D is able to gener-

Armandpour M, Zheng H, Sadeghian A, Sadeghian A, Zhou M (2023) Re-imagine the negative prompt algorithm: Transform 2d diffusion into 3d, alleviate janus problem and beyond. arXiv preprint arXiv:230404968

Brown T, Mann B, Ryder N, Subbiah M, Kaplan JD, Dhariwal P, Neelakantan A, Shyam P, Sastry G, Askell A, et al. (2020) Language models are few-shot learners. Advances in neural information processing systems 33:1877–1901

Chan ER, Lin CZ, Chan MA, Nagano K, Pan B, De Mello S, Gallo O, Guibas LJ, Tremblay J, Khamis

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

Origin “cartoon style” “Monet style” “Van Gogh style” “Michelangelo style statue of”

- Fig. 16: It is straightforward to diversify our generated objects by adding more detailed descriptions in an additional post-processing step, such as DMTet (Shen et al., 2021). The main prompt is “a boy wearing a peaked cap is shouting”.

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

“cat” “civet” “dog” “fox” “koala” “rabbit”

- Fig. 17: Fine-grained consistency for changing the animal species. The corresponding text is “a species sitting in a basket and wearing a sombrero”. The remaining items other than species are well maintained.

S, et al. (2022) Efficient geometry-aware 3d generative adversarial networks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 16123–16133

Deitke M, Schwenk D, Salvador J, Weihs L, Michel O, VanderBilt E, Schmidt L, Ehsani K, Kembhavi A, Farhadi A (2023) Objaverse: A universe of annotated 3d objects. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 13142–13153

Ding M, Yang Z, Hong W, Zheng W, Zhou C, Yin D, Lin J, Zou X, Shao Z, Yang H, et al. (2021) Cogview: Mastering text-to-image generation via transformers. Advances in Neural Information Processing Systems 34:19822–19835

Esser P, Rombach R, Ommer B (2021) Taming transformers for high-resolution image synthesis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp 12873– 12883

Goodfellow I, Pouget-Abadie J, Mirza M, Xu B, WardeFarley D, Ozair S, Courville A, Bengio Y (2020) Generative adversarial networks. Communications of the ACM 63(11):139–144

Ho J, Salimans T (2022) Classifier-free diffusion guidance. arXiv preprint arXiv:220712598

Ho J, Jain A, Abbeel P (2020) Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems 33:6840–6851

Jain A, Mildenhall B, Barron JT, Abbeel P, Poole B (2022) Zero-shot text-guided object generation with

dream fields. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp 867–876

Karras T, Laine S, Aila T (2019) A style-based generator architecture for generative adversarial networks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp 4401–4410

Karras T, Laine S, Aittala M, Hellsten J, Lehtinen J, Aila T (2020) Analyzing and improving the image quality of StyleGAN. In: CVPR

Karras T, Aittala M, Laine S, H¨rk¨nen E, Hellsten J, Lehtinen J, Aila T (2021) Alias-free generative adversarial networks. In: NeurIPS

Khalid NM, Xie T, Belilovsky E, Tiberiu P (2022) Clipmesh: Generating textured meshes from text using pretrained image-text models. SIGGRAPH Asia 2022 Conference Papers

Kingma DP, Ba J (2014) Adam: A method for stochastic optimization. arXiv preprint arXiv:14126980

Lee HH, Chang AX (2022) Understanding pure clip guidance for voxel grid nerf models. arXiv preprint arXiv:220915172

Li J, Tan H, Zhang K, Xu Z, Luan F, Xu Y, Hong Y, Sunkavalli K, Shakhnarovich G, Bi S (2023a) Instant3d: Fast text-to-3d with sparse-view generation and large reconstruction model. arXiv preprint arXiv:231106214

Li W, Chen R, Chen X, Tan P (2023b) Sweetdreamer: Aligning geometric priors in 2d diffusion for consistent text-to-3d. arXiv preprint arXiv:231002596

Liu M, Xu C, Jin H, Chen L, Xu Z, Su H, et al. (2023a) One-2-3-45: Any single image to 3d mesh in 45 seconds without per-shape optimization. arXiv preprint arXiv:230616928

Liu R, Wu R, Van Hoorick B, Tokmakov P, Zakharov S, Vondrick C (2023b) Zero-1-to-3: Zero-shot one image to 3d object. In: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp 9298– 9309

- Liu Y, Lin C, Zeng Z, Long X, Liu L, Komura T, Wang W (2023c) Syncdreamer: Generating multiviewconsistent images from a single-view image. arXiv preprint arXiv:230903453
- Liu Z, Dai P, Li R, Qi X, Fu CW (2022) Iss: Image as stetting stone for text-guided 3d shape generation. arXiv preprint arXiv:220904145

Long X, Guo YC, Lin C, Liu Y, Dou Z, Liu L, Ma Y, Zhang SH, Habermann M, Theobalt C, et al. (2023) Wonder3d: Single image to 3d using cross-domain diffusion. arXiv preprint arXiv:231015008

Lorraine J, Xie K, Zeng X, Lin CH, Takikawa T, Sharp N, Lin TY, Liu MY, Fidler S, Lucas J (2023) Att3d: Amortized text-to-3d object synthesis. arXiv preprint arXiv:230607349

Metzer G, Richardson E, Patashnik O, Giryes R, Cohen-Or D (2023) Latent-nerf for shape-guided generation of 3d shapes and textures. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition

Mildenhall B, Srinivasan PP, Tancik M, Barron JT, Ramamoorthi R, Ng R (2020) Nerf: Representing scenes as neural radiance fields for view synthesis. In: ECCV

Mu¨ller T, Evans A, Schied C, Keller A (2022) Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (ToG) 41(4):1–15

Nichol A, Dhariwal P, Ramesh A, Shyam P, Mishkin P, McGrew B, Sutskever I, Chen M (2021) Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:211210741

Nichol A, Jun H, Dhariwal P, Mishkin P, Chen M (2022) Point-e: A system for generating 3d point clouds from complex prompts. arXiv preprint arXiv:221208751 Poole B, Jain A, Barron JT, Mildenhall B (2022) Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:220914988

Qiao T, Zhang J, Xu D, Tao D (2019) Mirrorgan: Learning text-to-image generation by redescription. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)

Radford A, Kim JW, Hallacy C, Ramesh A, Goh G, Agarwal S, Sastry G, Askell A, Mishkin P, Clark J,

et al. (2021) Learning transferable visual models from natural language supervision. In: International Conference on Machine Learning, PMLR, pp 8748–8763

Ramesh A, Pavlov M, Goh G, Gray S, Voss C, Radford A, Chen M, Sutskever I (2021) Zero-shot textto-image generation. In: International Conference on Machine Learning, PMLR, pp 8821–8831

Ramesh A, Dhariwal P, Nichol A, Chu C, Chen M (2022) Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:220406125

Reed S, Akata Z, Yan X, Logeswaran L, Schiele B, Lee H (2016) Generative adversarial text to image synthesis. In: International conference on machine learning, PMLR, pp 1060–1069

Rombach R, Blattmann A, Lorenz D, Esser P, Ommer B (2022) High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 10684–10695

Ruan S, Zhang Y, Zhang K, Fan Y, Tang F, Liu Q, Chen E (2021) Dae-gan: Dynamic aspect-aware gan for text-to-image synthesis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp 13960–13969

Saharia C, Chan W, Saxena S, Li L, Whang J, Denton E, Ghasemipour SKS, Ayan BK, Mahdavi SS, Lopes RG, et al. (2022) Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:220511487

Sanghi A, Chu H, Lambourne JG, Wang Y, Cheng CY, Fumero M, Malekshan KR (2022) Clip-forge: Towards zero-shot text-to-shape generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp 18603– 18613

Schuhmann C, Vencu R, Beaumont R, Kaczmarczyk R, Mullis C, Katta A, Coombes T, Jitsev J, Komatsuzaki A (2021) Laion-400m: Open dataset of clipfiltered 400 million image-text pairs. arXiv preprint arXiv:211102114

Schuhmann C, Beaumont R, Vencu R, Gordon C, Wightman R, Cherti M, Coombes T, Katta A, Mullis C, Wortsman M, et al. (2022) Laion-5b: An open large-scale dataset for training next generation image-text models. arXiv preprint arXiv:221008402

Shen T, Gao J, Yin K, Liu MY, Fidler S (2021) Deep marching tetrahedra: a hybrid representation for high-resolution 3d shape synthesis. Advances in Neural Information Processing Systems 34:6087–6101 Shi Y, Wang P, Ye J, Long M, Li K, Yang X (2023) Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:230816512

Song J, Meng C, Ermon S (2020) Denoising diffusion implicit models. arXiv preprint arXiv:201002502

Song Y, Sohl-Dickstein J, Kingma DP, Kumar A, Ermon S, Poole B (2021) Score-based generative modeling through stochastic differential equations. 2011. 13456

Sun C, Sun M, Chen HT (2022) Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp 5459–5469

Tan H, Liu X, Li X, Zhang Y, Yin B (2019) Semanticsenhanced adversarial nets for text-to-image synthesis. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)

Tang S, Zhang F, Chen J, Wang P, Furukawa Y (2023) Mvdiffusion: Enabling holistic multi-view image generation with correspondence-aware diffusion. CoRR abs/2307.01097

Tsalicoglou C, Manhardt F, Tonioni A, Niemeyer M, Tombari F (2023) Textmesh: Generation of realistic 3d meshes from text prompts. arXiv preprint arXiv:230412439

Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser  L, Polosukhin I (2017) Attention is all you need. Advances in neural information processing systems 30

Wang H, Du X, Li J, Yeh RA, Shakhnarovich G (2022a) Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. arXiv preprint arXiv:221200774

Wang H, Du X, Li J, Yeh RA, Shakhnarovich G (2023a) Score jacobian chaining: Lifting pretrained 2d diffusion models for 3d generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp 12619–12629

Wang Z, Liu W, He Q, Wu X, Yi Z (2022b) Clip-gen: Language-free training of a text-to-image generator with clip. arXiv preprint arXiv:220300386

Wang Z, Lu C, Wang Y, Bao F, Li C, Su H, Zhu J (2023b) Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. arXiv preprint arXiv:230516213

Xu T, Zhang P, Huang Q, Zhang H, Gan Z, Huang X, He X (2018) Attngan: Fine-grained text to image generation with attentional generative adversarial networks. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)

Yi H, Zheng Z, Xu X, Chua Ts (2023) Progressive textto-3d generation for automatic 3d prototyping. arXiv preprint arXiv:230914600

Zhang H, Xu T, Li H, Zhang S, Wang X, Huang X, Metaxas DN (2017) Stackgan: Text to photo-realistic image synthesis with stacked generative adversarial networks. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV)

### A Network Architecture

Our decoder network is inspired by the UNet architecture of Stable Diffusion (Rombach et al., 2022) whose encoder network is removed herein. Overall, we set the base channel as 80 instead of the original 320 and increase the layers per block to 10, resulting in a deeper and more powerful decoder network. It takes the output of the token-to-plane transformation, with a resolution of 8×8, as input. Its five stages gradually increase the feature resolution to 16×16, 32×32, 64×64, 128×128 and 256 × 256, respectively. The corresponding channel multipliers are 4, 4, 2, 1 and 1. Each of the first three stages consists of ten alternate attention and convolution blocks, while the last two stages are only composed of convolution blocks. One ×2-upsampling layer is inserted after each stage to increase the feature resolution, finally yielding an output triplane with the shape of 3 × 32 × 256 × 256.

Attention Block. The attention block is shown in Figure 18, which enables inner information interaction by multi-head self-attention and text-3D information interchange through multi-head cross-attention.

Convolution Block. The convolution block is illustrated in Figure 19, which uses Adaptive Instance Normalization (AdaIN) to absorb the style information from the style injection module. Convolution layers are used in this block to introduce spatial inductive bias.

### B More Visualization Results

We show more objects generated by our Instant3D trained on the Animals set (Figure 20), Portraits set (Figure 21), and Daily Life set (Figure 22). The results demonstrate that our Instant3D successfully generates faithful 3D objects for new prompts.

Upsample x2

Conv 3x3 Conv 1x1

2퐻 2 

[Figure 168]

[Figure 169]

[Figure 170]

Upsample x2

Conv 3x3

AdaIN Conv 3x3

AdaIN

퐻  

2퐻 2 

2퐻 2 

FC

FC

Style

Style

Stylized Convolution Blocks

##### 20 Ming Li et al.

Multi-head Cross-attention

Multi-head Self-attention

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

Q Projection

QKV Projection

Reshape

Reshape

Norm

Norm

Norm

MLP

퐶 퐻  

퐶 퐻  

퐻  퐶

K V Projection

Token Embeddings

#### Fig. 18: Illustration of the attention block, which mainly contains a self-attention mechanism for inner-feature map information interaction and a cross-attention mechanism for text-3D information interchange.

[Figure 178]

Attentional Blocks

Conv 1x1

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

AdaIN Conv 3x3

AdaIN Conv 3x3

퐻  

퐻  

FC

FC

Style

Style

#### Fig. 19: Illustration of the convolution block, which employs AdaIN to incorporate the output of the style injection module into the triplane generation.

Stylized Convolution Blocks

[Figure 184]

- (a) “a dog sitting on the lawn and wearing a tie and wearing a sombrero”

[Figure 185]

- (b) “a koala sitting on books and wearing a cape and wearing a baseball cap”

[Figure 186]

(c) “a rabbit sitting on a stone and wearing a beret”

[Figure 187]

- (d) “a panda sitting on the lawn and wearing a beanie”

[Figure 188]

- (e) “a rabbit sitting in a basket and wearing a tie”

[Figure 189]

(f) “a dog sitting on a stone and wearing a tie and wearing a beanie”

- Fig. 20: More results on the Animals set, which are inferred directly by our Instant3D for novel prompts. We visualize continuous view images rendered around the front direction of each object to show the cross-view consistency.

22 Ming Li et al.

[Figure 190]

(a) “an elderly woman wearing a Santa hat is looking ahead with a very serious expression”

[Figure 191]

- (b) “a girl wearing a Santa hat is angry”

[Figure 192]

- (c) “Obama wearing a peaked cap is talking”

[Figure 193]

(d) “a black woman wearing a steampunk hat is feeling sad”

[Figure 194]

(e) “a boy wearing a peaked cap is shouting”

[Figure 195]

(f) “an elderly man wearing a crown is opening mouth wide in shock”

[Figure 196]

(g) “a black man wearing a peaked cap is laughing”

- Fig. 21: More results on the Portraits set. The prompts describe various figures wearing different hats and expressing different emotions. We visualize continuous view images rendered from each object.

[Figure 197]

[Figure 198]

- (a) “a woman with long hair wearing jeans is shopping”

[Figure 199]

[Figure 200]

- (b) “a strong and tall man with a beard is jogging”

[Figure 201]

[Figure 202]

(c) “a trendy woman is dancing to music”

[Figure 203]

[Figure 204]

(d) “a man wearing a baseball cap is playing video games”

[Figure 205]

[Figure 206]

(e) “a woman with glasses is working on a laptop”

[Figure 207]

[Figure 208]

(f) “a stocky man is barbecuing”

- Fig. 22: More generated objects for new testing prompts by our Instant3D trained on the Daily Life set. For each prompt, four view images and their normal images are shown. Our framework is able to generate complex and diverse objects for new prompts with high-quality geometric details.

