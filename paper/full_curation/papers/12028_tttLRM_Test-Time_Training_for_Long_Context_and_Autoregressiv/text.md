## tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction

# arXiv:2602.20160v2[cs.CV]2Mar2026

Chen Wang1∗ Hao Tan2 Wang Yifan2 Zhiqin Chen2 Yuheng Liu3∗ Kalyan Sunkavalli2 Sai Bi2 Lingjie Liu1† Yiwei Hu2†

1University of Pennsylvania 2Adobe Research 3UCI https://cwchenwang.github.io/tttLRM

[Figure 1]

Input View Gaussians Novel Views Feedforward GS Reconstruction Novel Views

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Autoregressive GS Reconstruction

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

4 Views 8 Views 32 Views

Figure 1. We propose tttLRM, a Large Reconstruction Model based on Test-Time Training, enabling high-resolution, long-context, autoregressive 3D reconstruction. Our model achieves 1) high-resolution (1024px) single-image-to-3D reconstruction via a multi-view generator 2) long-context (64 input views) and feedforward 3DGS reconstruction, and supports 3) autoregressive‡ streaming reconstruction.

#### Abstract

We propose tttLRM, a novel large 3D reconstruction model that leverages a Test-Time Training (TTT) layer to enable long-context, autoregressive 3D reconstruction with linear computational complexity, further scaling the model’s capability. Our framework efficiently compresses multiple image observations into the fast weights of the TTT layer, forming an implicit 3D representation in the latent space that can be decoded into various explicit formats, such as Gaussian Splats (GS) for downstream applications. The online learning variant of our model supports progressive 3D reconstruction and refinement from streaming observations. We demonstrate that pretraining on novel view synthesis tasks effectively transfers to explicit 3D modeling, resulting in improved reconstruction quality and faster convergence. Extensive experiments show that our method

*Work done as interns at Adobe Research. †Equal advising. ‡To clarify, we use autoregressive to denote online, causal version.

achieves superior performance in feedforward 3D Gaussian reconstruction compared to state-of-the-art approaches on both objects and scenes.

#### 1. Introduction

Reconstructing explicit 3D representations for photorealistic rendering from streaming visual input is a central goal of 3D reconstruction. This process is similar to how humans perceive the physical world: we observe a continuous visual stream, build an abstract internal representation of the world, and decode this abstraction into explicit 3D only when needed for fine-grained tasks or to recall detailed 3D structure. In light of this human-like process, we aim to enable long-context, autoregressive reconstruction of explicit 3D from streaming visual input.

However, existing 3D reconstruction methods are not designed for long-context scenarios with a memory mechanism. Traditional approaches to generating 3D representations and synthesizing novel views, including Neural Ra-

diance Fields (NeRF) [35, 64] and 3D Gaussian Splatting (3DGS) [21] have achieved substantial progress for high-quality rendering, but they either require slow scenespecific optimization or rely on feedforward reconstruction models with limited input-view scalability.

For example, Large Reconstruction Models (LRMs) have been proposed to rapidly reconstruct various 3D representations such as NeRFs [16], meshes [55], and 3DGS [68] from input images. However, these models are typically restricted to only a few input views (e.g., four), which limits their ability to reconstruct large-scale scenes. While Long-LRM [71] extends the number of input views to 32, its use of bidirectional attention layers hinders further scalability and prevents efficient processing of inputs with longer and streamed context, limiting its applicability in real-world scenarios. On the other hand, recent research [12, 18, 38, 70] on implicit latent-space 3D representations has demonstrated superior novel view synthesis quality using purely neural networks. However, despite being feedforward models for reconstruction, their rendering speed is significantly slower than that of explicit representations such as 3DGS due to repetitive network inference, and they lack controllability and interpretability, making them less suitable for many downstream applications.

In this paper, we propose tttLRM, a novel reconstruction model that leverages neural architectures and the knowledge distilled from pretrained implicit latent-space 3D models, decoding them into explicit 3D representations. This design ensures high-quality novel view synthesis with long-context and autoregressive modeling, while maintaining real-time rendering capability via explicit 3D outputs.

Our model builds upon Test-Time Training (TTT) [47, 70] and introduces an architecture composed of LaCT [70] blocks that has only linear computational complexity. We interpret the fast weights of TTT models, which are updated according to inputs during inference, as implicit latentspace 3D representations that can be decoded into various explicit formats such as 3DGS or NeRFs. We demonstrate that, with minimal architectural modification, our framework effectively leverages the pretrained knowledge of large novel view synthesis models [18, 70] for explicit 3D reconstruction. Specifically, our model is trained to query the fast weights for different 3D representations, such as a set of virtual view planes for 3DGS, or a triplane feature grid for NeRF-based reconstruction. This design unlocks greater flexibility in the final 3D representation. Also, by redesigning the fast-weight update and query mechanism, tttLRM enables autoregressive 3D reconstruction and refinement with streaming inputs. We further introduce sequence parallelism to enhance scalability.

We validate our model on both object- and scene-level datasets. Across both datasets, our model achieves superior reconstruction quality compared to baseline methods, while

also being highly efficient. We also show that our model supports autoregressive reconstruction, enabling practical real-world applications. The contributions of our paper can be summarized as following:

- • We propose tttLRM, the first large reconstruction model that leverages TTT for both feedforward long-context and autoregressive 3D modeling with linear complexity.
- • We design a scalable, unified 3D modeling framework that interprets TTT fast weights into observable and controllable explicit 3D representations.
- • We achieve state-of-the-art results on both object- and scene-level datasets, delivering superior quality and efficiency in 3D reconstruction and novel view synthesis.

#### 2. Related Work

Multi-view 3D Reconstruction 3D reconstruction from images has been extensively studied in computer vision. Traditional methods such as structure-from-motion [40] or multi-view stereo (MVS) [14] focus on recovering 3D geometry. Deep learning has enabled feed-forward 3D reconstruction [60, 61], which builds cost volumes using plane sweep for per-view depth estimation. Recently, learningbased MVS approaches [7, 24, 26, 52, 54, 58] directly estimate point clouds from input images and have been applied to camera pose estimation. Test3R [66] optimizes the network at test time in a self-supervised manner to improve 3D reconstruction. Concurrent work TTT3R [7] defines a gradient to update states for point cloud reconstruction. However, none of these methods can produce photo-realistic novel view synthesis.

Neural representations then have emerged as a promising way for both geometry reconstruction and NVS. NeRF [35] represents the scene as a continuous field and leverages a coordinate-based MLP to predict per-point color and density, enabling differential volumetric rendering with rendering-based supervision. Original NeRF takes hours to optimize a single scene and following works improved its training and rendering efficiency using advanced representations, including voxels [28, 45], points [57], hash grids [36], and triplanes [3, 5, 13]. Recently, 3D Gaussian Splatting [17, 21] has become the state-of-the-art neural scene representation. It uses volume rendering and rendering loss for optimization similar to NeRF but represents the scene with simple Gaussian primitives which enables real-time rendering and large-scale scene reconstruction [22, 29]. However, 3DGS still requires optimizing 3D Gaussians from scratch, taking several minutes per scene, whereas our model performs 3D reconstruction within seconds in a feed-forward way.

Learning-based Feedforward 3D Reconstruction The development on learning-based methods enables 3D reconstruction and novel view synthesis by training neural networks on large-scale datasets to directly infer 3D structures

without per-scene optimization. Early work utilizes Convolutional Neural Networks (CNN) to predict multi-plane images [11, 34], points [1, 63] or voxels [44]. Large Reconstruction Models (LRM) [16] propose a transformerbased architecture without 3D inductive bias for 3D object reconstruction from multi-view images, with triplane as the 3D representation. GS-LRM [68] further extends LRM to predict pixel-aligned 3DGS, but the model can only take very few images as input due to the quadratic complexity of attention layers. Similarly, the subsequence approach [4, 8, 48, 56] also apply a feedforward framework with different neural architectures and 3D inductive bias for Gaussian prediction. Mamba-based models [41, 62] has attempted to reduce the complexity of attention layers, but are still limited to very few input views. Long-LRM [71] represents the state of the art in long-sequence Gaussian reconstruction, but it remains limited to 32 input views and relies on additional attention layers. By leveraging TTT, our model achieves longer-context and autoregressive reconstruction with improved NVS quality.

Linear Attention and State Space Models To circumvent the quadratic complexity of attention [49], recent research has explored efficient alternatives that retain contextual expressivity while reducing computational cost. Linear attention models [20, 39, 42] approximate the softmax kernel with linearized feature maps to achieve linear complexity, but uniform compression of past key–value pairs often degrades the upper bound of long sequence modeling.

State Space Models (SSMs) introduce a state variable to represent historical information, similar to classical Recurrent Neural Networks (RNNs). Recent works [9, 15, 30, 46] incorporate attenuation factors into the state updates, allowing the model to retain more recent information while gradually forgetting the distant past. Among them, Mamba [9, 15, 30] proposes “date-dependent decay” to model sequences as continuous-time dynamical systems governed by state transition, but it still cannot compete with transformers in long-context reasoning [51]. Jamba [25] implements a hybrid mamba attention model to improve the performance. Test Time Training (TTT), on the other hand, [2, 47, 70] transforms the problem into an online learning problem and applies modern optimizers to learn the states. DeltaNet [39, 59] and MesaNet [50] share the same idea but use different update rules when updating. Inspired by its success, we introduce Test-Time Training into 3D reconstruction tasks for high-quality long-context novel view synthesis, but with only linear complexity.

#### 3. Method

##### 3.1. Preliminary: TTT and LaCT Layer

We first briefly introduce the fundamentals of TTT and Large Chunk Test-Time Training (LaCT) layer, which form the core building blocks of our model. In sequence mod-

eling, the input is typically represented as a sequence of tokens of length L, denoted by [x1,x2,...,xL], where each token has dimension d: xi ∈ Rd. In standard attention, each input token will be projected into query, key and value vectors, denoted as qi, ki, vi. Each token attends to all others via a dot-product operation, leading to quadratic complexity in sequence length.

TTT [47] learns a set of fast weights W that are updated at inference time according to the input to capture the relationship between input tokens. Specifically, it treats the key-value pairs (ki, vi) of input tokens as training data to update fast weights using mean-square error: W ← W − η∇LMSE(fW(k),v), which can be further applied to queries to obtain the final input o = fW(q). In this way, the fast weights effectively encode the key–value (KV) cache of the input sequence into a fixed-size neural memory.

Originally, the TTT model [47] updates the fast weights using only a small minibatch (e.g. 16 tokens), which results in very low GPU FLOP utilization and difficulty in handling long sequences. Large Chunk Test-Time Training (LaCT) [70] instead updates fast weights with large chunk size (up to 1M tokens). Its chunk-wise update computes the gradient of the summed loss over all keys and values within the chunk. More details can be found in [70].

##### 3.2. Model Architecture

We illustrate our model architecture in Figure 2, using 3DGS reconstruction as an example, though the same framework can be applied to other 3D representations as well. Given a set of posed images, denoted as {Ii ∈ RH×W×3|i = 1,2,..,N}, we concatenate them channelwise with their ray embeddings {Ri ∈ RH×W×9|i = 1,2,..,N} as the positional embedding. After dividing each image into non-overlapping patches of size p × p, we tokenize these image patches using a lightweight linear layer into a sequence of tokens T:

HW/p2 j=1 = Tokenize Patchify([{Ii}Ni=1,{R}Ni=1]) ,

{Ti,j}Ni=1

These visual tokens then iteratively update the fast weights W of a set of LaCT blocks using Muon [19] optimizer:

Ti = Ti + WinAttn(Ti), (1) W = Update({Ti}Ni=1), (2) Ti = Apply(W,Ti) (3)

Each LaCT layer includes a window attention module that captures local relationships within each view. We omit the feedforward layers in the block in the equation for simplicity. The update and apply operations are in linear complexity with respect to the sequence length.

To retrieve information from the fast weights, we introduce a set of virtual tokens that serve as queries to our model. In 3DGS reconstruction, these virtual tokens are

Fast Weights

Input Tokens

|[Figure 13]<br><br>[Figure 14]|
|---|

𝑊

K

[Figure 15]

Update

InputViewsVirtualViews

Patchify & Linear

Window Attention

+

+

𝑊 MLP

V

+

+

| | | | |
|---|---|---|---|
| | | | |
| | | | |
| | | | |
| | | | |

|[Figure 16]<br><br>[Figure 17]|
|---|

Apply

Q

x L

Query Tokens

[Figure 18]

|[Figure 19]<br><br>[Figure 20]|
|---|

[Figure 21]

[Figure 22]

Unpatchify

Linear&

Patchify & Linear

[Figure 23]

[Figure 24]

|[Figure 25]<br><br>[Figure 26]|
|---|

Scene-Level GS

Novel View Renderings

- Figure 2. Given a set of posed input images, tttLRM encodes them into tokens (green boxes) after patchifying. The input tokens are fed into the LaCT block (shown in the blue frame) where fast weights are updated accordingly. Another set of virtual tokens (blue boxes) are used to query the updated fast weights, and decoded into 3D representations like 3DGS for high-quality novel view synthesis.

virtual views {Ivi ∈ RH×W×3|i = 1,2,..,M} for GS prediction, which will also be patchified and tokenized to

to update the fast weights before decoding, the streaming variant performs incremental updates in a causal manner. As illustrated in Algorithm 1, providing our model F, for each incoming mini-batch of views I(b) (e.g., four images at a time), the model updates the fast weights and immediately predicts the corresponding 3D Gaussian parameters for the new query views I(vb), returning the current reconstructed Gaussian splat results G(b). This design effectively transforms the model into an RNN-like inference process, where the internal state (fast weights) evolves as new observations arrive, enabling online 3D Gaussian reconstruction. The fast weight update can also consider historical gradients and fast weights to mitigate drifting (See Supplemental).

HW/p2 j=1 . In other 3D representations, such as triplane NeRFs, these virtual tokens are learnable triplane features. The virtual tokens are only used in the apply operation without updating the fast weights:

{Tvi,j}Ni=1

Tvi = Apply(W,Tvi) (4)

Given the updated query tokens Tvi, a linear token decoder transforms them into explicit 3D representations, such as per-patch Gaussian parameters in 3DGS reconstruction. The RGB color, scale, rotation, and opacity of each Gaussian are predicted directly. For Gaussian positions, we first decode the depth of each pixel and use a range function (object-centric for object data and linear for scene data) to convert it to real depth. After that, we convert depth to a Gaussian position with known ray locations and directions.

##### 3.4. Distributed Feedforward Reconstruction

A large number of input views and high-resolution images introduce a substantial number of tokens, leading to a significant increase in both computation and memory cost. A key limitation of prior works lies in their inability to handle long input sequences efficiently, largely due to the lack of parallelism at the sequence level and most methods process all input views within a single device.

##### 3.3. Autoregressive Reconstruction

Algorithm 1 Autoregressive 3DGS Reconstruction Input: Reconstructor F with initial fast weights W0; in-

put/query view batches {(I(b),I(vb))}Bb=1 Output: Reconstructed GS G

To address this limitation, we introduce sequence parallelism for training feedforward reconstruction models, exemplified by 3DGS reconstruction as shown in Figure 3. Specifically, we partition the tokenized input views along the sequence dimension and assign each shard to a separate device. During training:

- 1: W ← W0
- 2: for b = 1 to B do
- 3: ,W ← F(W,I(b))

- 4: G(b), ← F(W,I(vb))

- 5: end for
- 6: return G(B)

- • Since Gaussians can be predicted independently for each virtual view once the fast weights are synchronized, each GPU predicts pixel-aligned Gaussian primitives for its assigned views (first row).
- • The predicted Gaussians from all devices are gathered to form the complete scene representation (second row).
- • Each GPU subsequently renders its own set of novel views and computes photometric reconstruction losses

An important feature of our architecture is its support for autoregressive modeling with streamed input images. To enable this, we modify the update and apply steps to incorporate causal dependencies among tokens. Unlike the standard setting where all input views are jointly processed

GPU 1 GPU 2

by aligning the Gaussian position along the depth direction (z axis) with ground truth depth for that Gaussian. We opt for using the monocular depth estimator [53] for pseudo ground truth since we found that feedforward MVS methods like VGGT [52] provide less detailed depth prediction, albeit being multi-view consistent. Similar to LongLRM [71], we also use opacity regularization to reduce the number of Gaussians. Our final loss function can be written as follows:

|[Figure 27]<br><br>[Figure 28]|
|---|

|[Figure 29]<br><br>[Figure 30]|
|---|

[Figure 31]

[Figure 32]

Fast Weight

[Figure 33]

L = LRGB + λdepthLdepth + λopacityLopacity (6)

[Figure 34]

#### 4. Experiments

[Figure 35]

##### 4.1. Model and Training

Model Details Our model consists of 24 LaCT blocks with the hidden dimension of 768. The window attention layers have 64-dimension for each head with QK-normalization for stability. For the feedforward layer, we use a two-layer MLP with 4 intermediate expansion ratios for the intermediate dimension. We use a patch size of 8 × 8 for the image tokenizer. Our architecture shares the same parameterization as TTT-LVSM [70] except for the decoding module, allowing us to effectively leverage its pretrained weights as a strong initialization for our model.

∇ + ∇ 2

∇ + ∇ 2

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

Novel View Grad ∇ Novel View Grad ∇

- Figure 3. Illustration of distributed feedforward reconstruction training. First, image tokens are sharded across GPUs, and each GPU predicts Gaussians for its assigned virtual views after the fast weights are synchronized. The predicted Gaussians are then gathered to construct the full scene, after which each GPU renders a subset of novel views and computes its respective losses. Gradients are finally all reduced and backpropagated across all devices.

##### 4.2. Datasets

Object-level Dataset We train our object-level reconstruction model on the Objaverse dataset [10]. Following prior works [16, 68], each 3D object is centered and normalized to fit within a bounding box of [−1,1]. We render 32 views per object, where cameras are randomly distributed around the object at distances uniformly sampled from [1.5,2.8]. All images are rendered at a resolution of 512 × 512 under uniform lighting conditions. In total, we use 730K objects for training. We evaluate our model on 100 objects sampled from the Google Scanned Objects (GSO) dataset. For evaluation, we select a few views as input and the same random 8 views for testing.

against the ground truth, and gradients are all reduced to enable sequence-level backpropagation (third row).

Thanks to the linearity of our LaCT fast-weight updates, gradients of the fast weights across devices can be easily synchronized through PyTorch Distributed Data Parallel (DDP), ensuring consistent global optimization. During inference, the distributed reconstruction also allows us to accelerate the reconstruction with more GPUs.

Scene-level Dataset We train our model on the challenging DL3DV-10K [27] dataset, which consists of 10,510 highresolution videos, each containing up to 500 keyframes with camera pose annotation obtained from COLMAP [37]. The testing set of DL3DV-140 contains 140 test scenes. We use the same input and target split from that provided by LongLRM [71]: the testing views are evenly selected from every 8 views (around 40 images each scene) and input views are selected based on K-means clustering based on camera positions and view directions. We also tested our model on Tanks&Temples [23] dataset.

##### 3.5. Training Objective

Our training does not require explicit 3D supervision. We render the reconstructed GS on the target views for supervised training, and minimize the rendering loss that is a combination of Mean Squared Error (MSE) and perceptual loss based on VGG-19 features [43]:

LRGB = MSE(Ipred,Igt) + λ Perceptual(Ipred,Igt) (5)

For non-autoregressive training, we randomly sample unordered input–target image pairs from the dataset. For the autoregressive model, we instead sample ordered input sequences to better simulate streaming use case.

##### 4.3. Baselines and Metrics

Object-level We compare our method with GS-LRM [68], an attention-based method. We train the model under 8 input views setting with the same iterations of our method.

Apart from rendering loss, for scene-level data, we use depth regularization with the scale-invariant depth loss [71]

3D Gaussian Scene (Ours) GT Ours LongLRM 3DGS Mipsplatting Scaffold-GS

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

PSNR: 16.15

PSNR: 16.27

- PSNR: 19.18

[Figure 90]

- PSNR: 20.43

PSNR: 17.43

PSNR: 17.88

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

PSNR: 19.76

PSNR: 19.51

PSNR: 18.14

PSNR: 18.54

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

PSNR: 26.00

- PSNR: 24.48

[Figure 114]

[Figure 115]

- PSNR: 25.21 PSNR: 24.56

PSNR: 23.54 PSNR: 25.65

PSNR: 24.93

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

PSNR: 26.97

PSNR: 21.29

PSNR: 26.62

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

PSNR: 25.60

- PSNR: 23.50

[Figure 137]

[Figure 138]

- PSNR: 24.58

PSNR: 28.56

PSNR: 23.53

- PSNR: 24.61

[Figure 139]

[Figure 140]

- PSNR: 25.57

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

PSNR: 26.56

PSNR: 23.56

PSNR: 25.04

- Figure 4. Qualitative comparison between our method and baseline approaches. Our model reconstructs the 3DGS scene with higher fidelity than both optimization-based and feedforward baselines, as also reflected in the PSNR metrics. Please zoom in for a better comparison.

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

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

Input

Input Input

Novel View GS Rendering Input

Novel View GS Rendering

Novel View GS Rendering

Novel View GS Rendering

- Figure 5. We demonstrate that our high-resolution 1024 × 1024 3DGS tttLRM can be effectively used for image-to-3D generation when combined with a multi-view generator. Our model enables the reconstruction of fine-grained, photorealistic details e.g., hair, fur, and text, from the input images. Video results are provided in the supplemental material.

Scene-level Previous feedforward reconstruction methods like GS-LRM [68] cannot be directly extended to long sequence due to the high complexity of attention. Long-LRM [71] is the only available feedforward method that can handle more than 16 input views. We also include three optimization-based methods: 3DGS [21], MipSplatting [65] and Scaffold-GS [31].

Thanks to the linear complexity of our architecture with respect to sequence length, at a resolution of 512 × 512, our model runs twice as fast as attention-based models while achieving over a 1 dB PSNR improvement. Our model also demonstrates strong generalization ability—when trained with 8 input views, it can be directly applied to 16 or 24 views (last two rows in the table). With longer sequences, inference becomes substantially faster, and rendering quality further improves through test-time training.

Metrics For all baselines, in addition to visual comparisons, we report three metrics to evaluate novel view synthesis quality: PSNR, SSIM, and LPIPS [69].

Moreover, our model scales seamlessly to 1024 × 1024 resolution, whereas GS-LRM encounters out-of-memory issues under high-resolution training. Results in Figure 5 show that our model can achieve high-quality 3D reconstruction of humans, animals, and texts from a single image when combined with a multi-view diffusion model.

##### 4.4. Results

Object-level We present quantitative comparison results under varying resolutions and numbers of input views in Table 1. Across all settings, our method consistently outperforms the baselines. At lower resolutions and shorter sequences, our inference speed is comparable to full attention models, as it is primarily determined by MLP operations.

Scene-level We further evaluate our model on scene reconstruction, as shown in Table 2. Compared with

[Figure 167]

[Figure 168]

Novel View Rendering

Novel View Rendering

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

Depth Triplane

Depth Triplane

- Figure 6. We show that tttLRM, as a general framework, can also interpret the latent 3D memory into formats besides 3DGS. In this experiment, we use a set of triplane tokens to query the fast weights and then fine-tune the model for triplane-based NeRF reconstruction. We visualize the resulting triplanes and present the corresponding renderings and depth maps for 4 views at a resolution of 512 × 512.

- Table 1. Comparison between our method and GS-LRM [68] on the GSO dataset under different resolutions and numbers of input views. Our method consistently outperforms GS-LRM in both inference speed and reconstruction quality, and also shows strong generalization ability. V. denotes the number of virtual views used to query the fast weight, which equals input views unless noted.

Method Resolution Views Time (s) ↓ PSNR↑ SSIM↑ LPIPS↓ GS-LRM [68]

256 × 256

8 0.1 31.55 0.964 0.028 Ours 8 0.1 33.14 0.972 0.024

GS-LRM [68]

512 × 512

8 0.7 32.83 0.969 0.029 Ours 8 0.3 34.02 0.974 0.025

GS-LRM [68] 16 2.5 33.55 0.976 0.023 Ours 16 (10 V.) 0.8 34.67 0.978 0.022

GS-LRM [68] 24 5.5 33.26 0.976 0.022 Ours 24 (10 V.) 1.1 34.80 0.979 0.022

- Table 2. Quantitative comparison on both DL3DV-140 and Tanks&Temples datasets under different numbers of input views. Our method surpasses previous feedforward methods and is comparable with optimization-based methods. Note that Long-LRM trains a separate model for each input view, while we are a single model across all input views. Our model can be linearly accelerated with multiple GPUs, here we report time on 1 A100.

datasets like Tanks & Temples.

Compared to the feedforward baseline LongLRM [71], tttLRM achieves substantially better performance—approximately 1 dB PSNR improvement—across different numbers of input views. On the other hand, we show our model constantly outperforms Long-LRM even when it’s combined with additional post-optimization. Furthermore, our method can be linearly accelerated by distributing the input across multiple GPUs, as described in Section 3.4.

Figure 4 shows visual comparisons between our method and baselines. tttLRM achieves better visual quality with fewer artifacts than optimization-based methods, thanks to the learned priors across diverse scenes. Our model also outperforms Long-LRM by reconstructing sharper and more detailed geometry (as shown in the red boxes).

Autoregressive Reconstruction We demonstrate the autoregressive reconstruction capability of our model in the second row of Figure 1. With only 4 input views, the model already produces reasonable 3D Gaussian reconstructions; as additional views arrive (8 and 32 views), both the rendering quality and scene coverage progressively improve. Additional examples of autoregressive reconstruction are provided on our project page. Table 2 also shows the quantitatively results of our autoregressive model, which constantly outperforms Long-LRM and remains competitive with, or superior to, optimization-based baselines.

DL3DV-140 Tanks&Temples

Views Method Time↓

PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓

3D GS30k 13m 21.20 0.708 0.264 16.76 0.598 0.334 Mip-Splatting30k 13m 20.88 0.712 0.274 16.82 0.616 0.332

16

Scaffold-GS30k 16m 22.13 0.738 0.250 17.02 0.634 0.321 Long-LRM (16v model) 0.4s 22.66 0.740 0.292 17.51 0.555 0.408

###### Ours (single model) 3.6s 23.60 0.784 0.255 18.15 0.613 0.360

3D GS30k 13m 23.60 0.779 0.213 18.10 0.688 0.269 Mip-Splatting30k 13m 23.32 0.784 0.217 18.39 0.700 0.262

Decoding into Other 3D Formats Beyond using 3DGS as the output representation, our architecture can also decode the latent 3D representation into other formats, such as triplane-based NeRFs. As described in Section 3.2, replacing the virtual tokens with triplane tokens enables the fast weights to be queried as a triplane representation for NeRF reconstruction. We finetune the model with a rendering loss to enable this capability. We show the NeRF renderings and the corresponding queried triplanes in Figure 6. This demonstrates that our architecture is flexible and can generalize to different 3D output formats.

Scaffold-GS30k 16m 24.77 0.805 0.205 18.41 0.691 0.290 Long-LRM (32v model) 1s 24.10 0.783 0.254 18.38 0.601 0.363

32

Long-LRM (32v model w/ optim) 12s 24.99 0.809 0.243 18.69 0.623 0.360 Ours (single model, AR) 7.5s 24.31 0.803 0.237 18.96 0.653 0.322 Ours (single model) 7.2s 25.07 0.822 0.215 19.22 0.662 0.305

3D GS30k 13m 26.55 0.852 0.164 20.78 0.778 0.205 Mip-Splatting30k 13m 26.29 0.850 0.166 20.08 0.759 0.220

64

Scaffold-GS30k 16m 27.07 0.857 0.173 20.96 0.768 0.240 Long-LRM (64v model) 3.7s 24.63 0.799 0.243 19.11 0.627 0.346 Ours (single model, AR) 15.2s 24.81 0.814 0.225 19.80 0.675 0.308

Ours (single model) 14.8s 25.95 0.844 0.195 20.31 0.700 0.274

optimization-based methods that tend to overfit to input views, our method achieves better results on 16 and 32 input views. With more input views, it remains competitive in reconstruction quality, while being hundreds times faster. Moreover, one single tttLRM model can be applied to different sequence lengths and effectively generalizes to new

##### 4.5. Ablation Study

We conduct ablation studies to analyze our design choices in LVSM pretraining and the autoregressive reconstruction

strategy.

Pretraining from TTT-LVSM We investigate the effectiveness of leveraging pretrained knowledge for both Gaussian Splatting and triplane training at a resolution of 256 × 256. The GS reconstruction has 8 input views with patch size 8 × 8, while the triplane version has 4 input views with patch size 16 × 16. As shown in Figure 7, Using GS model as an example, initialization with pretrained checkpoints substantially accelerates convergence, especially in the early training stage, where models quickly reach a high PSNR compared to the one trained from scratch.

Moreover, as reported in Table 3, pretrained initialization not only improves convergence speed but also leads to higher final quality after full training. The gains persist even when trying to adapt the pretrained weights to different 3D representations. The results suggest that pretrained knowledge of novel view synthesis serves as an effective inductive bias for 3D reconstruction, improving both training efficiency and final rendering fidelity.

- Table 3. Leveraging pretrained knowledge from novel view synthesis tasks improves the final 3D reconstruction quality across different 3D representations.

3D Rep. Type PSNR↑ SSIM↑ LPIPS↓ GS

w/o Pretrain 32.77 0.026 0.969

w Pretrain 33.14 0.024 0.972 Triplane

w/o Pretrain 26.40 0.903 0.093 w Pretrain 27.87 0.925 0.075

20

18

16

PSNR

14

12

w/ Pretrain

10

w/o Pretrain

0 2000 4000 6000 8000 10000 Step

- Figure 7. Our 3DGS reconstruction model leverages pretraining with LVSM on novel view synthesis tasks, which significantly accelerates learning and leads to better performance, compared to training from scratch. Autoregressive strategy In Section 3.3, we introduce our autoregressive reconstruction strategy. Here we consider a more straightforward way called “Predict & Merge”: in-

stead of generating a new 3DGS G(b) ← F(W,Iv(b)) for each step, we reuse the previously predicted Gaussians

G(b−1) and merge them with the newly predicted subset G(b′) ← F(W,I(vb′)), forming G(b) = G(b−1) ∪ G(b′). Here, I(vb′) is a subset of I(vb) containing only new virtual views not covered in I(vb−1). However, we found that though this approach is computationally more efficient, it cannot correct the accumulated errors in G(b−1), leading to

worse results than our proposed full reconstruction method, as shown in Table 4.

- Table 4. Although progressive GS prediction with merging provides more efficient computation, the reconstruction quality is degraded due to accumulated errors (compared on 32 views under 1K iterations finetuning).

PSNR↑ SSIM↑ LPIPS↓ Predict & Merge 21.50 0.891 0.318

Ours 23.63 0.904 0.259

Optimizer and Losses We use Muon optimizer for its stabilty and robustness. Table 5 shows that use Muon as opitmizer can bring better results even on low resolution setting. It will bring better results on longer sequence (e.g. high resolution and more input views). Also, using depth and opacity as regularization can help reduce opaque Gaussians.

- Table 5. Ablation on 32 view 256 × 144 input with the same iterations across settings.

Muon Opacity+Depth PSNR↑ SSIM↑ LPIPS↓ Opacity> 0.001

✗ ✗ 20.44 0.649 0.295 96% ✓ ✗ 20.68 0.661 0.290 97% ✓ ✓ 20.76 0.666 0.285 47%

##### 4.6. Discussions and Limitations

Our fast-weight memory has a fixed size, which may limit its ability to handle highly complex scenarios with extremely large numbers of input views. More discussions can be found in the supplemental. Also, we observe that, compared with the pretrained LVSM model from which we finetune, our quality slightly degraded but we have much faster rendering speed and explicit 3D representations for flexible downstream tasks. This might reflect the inherent trade-off between implicit and explicit representations. Future works might design a better memory mechanism, further improve the quality, and speed up the inference to enable real-time high-quality reconstruction for streaming inputs.

#### 5. Conclusion

In this paper, we present tttLRM, a large reconstruction model that supports both feedforward long-context and autoregressive 3D modeling. Under the Test-Time Training framework, it produces implicit fast-weight representations and converts them into explicit 3D representations such as Gaussian splats and triplanes for efficient, high-quality novel view synthesis. Experiments on object- and scenelevel datasets show that tttLRM outperforms prior feedforward methods in quality and scalability while approaching the speed of explicit representations. Our framework helps close the gap between neural network rendering and realtime explicit 3D systems.

#### Acknowledgment

The authors would like to thank Ziwen Chen for the evaluation of the baselines and Tianyuan Zhang for helpful discussions on LaCT.

#### References

- [1] Kara-Ali Aliev, Artem Sevastopolsky, Maria Kolos, Dmitry Ulyanov, and Victor Lempitsky. Neural point-based graphics. In European conference on computer vision, pages 696–

712. Springer, 2020. 3

- [2] Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time. arXiv preprint arXiv:2501.00663, 2024. 3
- [3] Eric R Chan, Connor Z Lin, Matthew A Chan, Koki Nagano, Boxiao Pan, Shalini De Mello, Orazio Gallo, Leonidas J Guibas, Jonathan Tremblay, Sameh Khamis, et al. Efficient geometry-aware 3d generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16123–16133, 2022. 2
- [4] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19457–19467, 2024. 3
- [5] Anpei Chen, Zexiang Xu, Andreas Geiger, Jingyi Yu, and Hao Su. Tensorf: Tensorial radiance fields. In European conference on computer vision, pages 333–350. Springer, 2022. 2
- [6] Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. Training deep nets with sublinear memory cost. arXiv preprint arXiv:1604.06174, 2016. 2
- [7] Xingyu Chen, Yue Chen, Yuliang Xiu, Andreas Geiger, and Anpei Chen. Ttt3r: 3d reconstruction as test-time training. arXiv preprint arXiv:2509.26645, 2025. 2
- [8] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In European Conference on Computer Vision, pages 370–386. Springer, 2024. 3
- [9] Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. arXiv preprint arXiv:2405.21060, 2024. 3
- [10] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objaverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13142–13153, 2023. 5
- [11] John Flynn, Michael Broxton, Paul Debevec, Matthew DuVall, Graham Fyffe, Ryan Overbeck, Noah Snavely, and Richard Tucker. Deepview: View synthesis with learned gradient descent. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2367– 2376, 2019. 3
- [12] John Flynn, Michael Broxton, Lukas Murmann, Lucy Chai, Matthew DuVall, Cl´ement Godard, Kathryn Heal, Srinivas

- Kaza, Stephen Lombardi, Xuan Luo, et al. Quark: Real-time, high-resolution, and general neural view synthesis. ACM Transactions on Graphics (TOG), 43(6):1–20, 2024. 2
- [13] Quankai Gao, Qiangeng Xu, Hao Su, Ulrich Neumann, and Zexiang Xu. Strivec: Sparse tri-vector radiance fields. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17569–17579, 2023. 2
- [14] Michael Goesele, Brian Curless, and Steven M Seitz. Multiview stereo revisited. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’06), pages 2402–2409. IEEE, 2006. 2
- [15] Albert Gu, Karan Goel, and Christopher R´e. Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396, 2021. 3
- [16] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. Lrm: Large reconstruction model for single image to 3d. arXiv preprint arXiv:2311.04400, 2023. 2, 3, 5
- [17] Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, and Shenghua Gao. 2d gaussian splatting for geometrically accurate radiance fields. In ACM SIGGRAPH 2024 conference papers, pages 1–11, 2024. 2
- [18] Haian Jin, Hanwen Jiang, Hao Tan, Kai Zhang, Sai Bi, Tianyuan Zhang, Fujun Luan, Noah Snavely, and Zexiang Xu. Lvsm: A large view synthesis model with minimal 3d inductive bias. arXiv preprint arXiv:2410.17242, 2024. 2
- [19] Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cecista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024. URL https://kellerjordan.github.io/posts/muon, 6. 3
- [20] Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Fran¸cois Fleuret. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pages 5156–5165. PMLR, 2020. 3
- [21] Bernhard Kerbl, Georgios Kopanas, Thomas Leimk¨uhler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1,

2023. 2, 6

- [22] Bernhard Kerbl, Andreas Meuleman, Georgios Kopanas, Michael Wimmer, Alexandre Lanvin, and George Drettakis. A hierarchical 3d gaussian representation for real-time rendering of very large datasets. ACM Transactions on Graphics (TOG), 43(4):1–15, 2024. 2
- [23] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics (ToG), 36

(4):1–13, 2017. 5

- [24] Yushi Lan, Yihang Luo, Fangzhou Hong, Shangchen Zhou, Honghua Chen, Zhaoyang Lyu, Shuai Yang, Bo Dai, Chen Change Loy, and Xingang Pan. Stream3r: Scalable sequential 3d reconstruction with causal transformer. arXiv preprint arXiv:2508.10893, 2025. 2
- [25] Barak Lenz, Opher Lieber, Alan Arazi, Amir Bergman, Avshalom Manevich, Barak Peleg, Ben Aviram, Chen Almagor, Clara Fridman, Dan Padnos, et al. Jamba: Hybrid transformer-mamba language models. In The thirteenth international conference on learning representations, 2025. 3

- [26] Vincent Leroy, Yohann Cabon, and J´erˆome Revaud. Grounding image matching in 3d with mast3r. In European Conference on Computer Vision, pages 71–91. Springer, 2024. 2
- [27] Lu Ling, Yichen Sheng, Zhi Tu, Wentian Zhao, Cheng Xin, Kun Wan, Lantao Yu, Qianyu Guo, Zixun Yu, Yawen Lu, et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22160–22169, 2024. 5
- [28] Lingjie Liu, Jiatao Gu, Kyaw Zaw Lin, Tat-Seng Chua, and Christian Theobalt. Neural sparse voxel fields. Advances in Neural Information Processing Systems, 33:15651–15663,

2020. 2

- [29] Yang Liu, Chuanchen Luo, Lue Fan, Naiyan Wang, Junran Peng, and Zhaoxiang Zhang. Citygaussian: Real-time high-quality large-scale scene rendering with gaussians. In European Conference on Computer Vision, pages 265–282. Springer, 2024. 2
- [30] Yue Liu, Yunjie Tian, Yuzhong Zhao, Hongtian Yu, Lingxi Xie, Yaowei Wang, Qixiang Ye, Jianbin Jiao, and Yunfan Liu. Vmamba: Visual state space model. Advances in neural information processing systems, 37:103031–103063, 2024. 3
- [31] Tao Lu, Mulin Yu, Linning Xu, Yuanbo Xiangli, Limin Wang, Dahua Lin, and Bo Dai. Scaffold-gs: Structured 3d gaussians for view-adaptive rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20654–20664, 2024. 6
- [32] Ziqiao Ma, Xueyang Yu, Haoyu Zhen, Yuncong Yang, Joyce Chai, and Chuang Gan. Fast spatial memory with scalable elastic test-time training. https://mars-tin. github.io/blogs/posts/elastic_ttt.html,

2025. Blog post. 1

- [33] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, et al. Mixed precision training. arXiv preprint arXiv:1710.03740, 2017. 2
- [34] Ben Mildenhall, Pratul P Srinivasan, Rodrigo Ortiz-Cayon, Nima Khademi Kalantari, Ravi Ramamoorthi, Ren Ng, and Abhishek Kar. Local light field fusion: Practical view synthesis with prescriptive sampling guidelines. ACM Transactions on Graphics (ToG), 38(4):1–14, 2019. 3
- [35] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. Communications of the ACM, 65(1):99–106, 2021. 2
- [36] Thomas M¨uller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM transactions on graphics (TOG), 41(4):1–15, 2022. 2
- [37] Linfei Pan, D´aniel Bar´ath, Marc Pollefeys, and Johannes L Sch¨onberger. Global structure-from-motion revisited. In European Conference on Computer Vision, pages 58–77. Springer, 2024. 5
- [38] Mehdi SM Sajjadi, Henning Meyer, Etienne Pot, Urs Bergmann, Klaus Greff, Noha Radwan, Suhani Vora, Mario

- Luˇci´c, Daniel Duckworth, Alexey Dosovitskiy, et al. Scene representation transformer: Geometry-free novel view synthesis through set-latent scene representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6229–6238, 2022. 2
- [39] Imanol Schlag, Kazuki Irie, and J¨urgen Schmidhuber. Linear transformers are secretly fast weight programmers. In International conference on machine learning, pages 9355–9366. PMLR, 2021. 3
- [40] Johannes L Schonberger and Jan-Michael Frahm. Structurefrom-motion revisited. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4104–4113, 2016. 2
- [41] Qiuhong Shen, Zike Wu, Xuanyu Yi, Pan Zhou, Hanwang Zhang, Shuicheng Yan, and Xinchao Wang. Gamba: Marry gaussian splatting with mamba for single-view 3d reconstruction. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 3
- [42] Zhuoran Shen, Mingyuan Zhang, Haiyu Zhao, Shuai Yi, and Hongsheng Li. Efficient attention: Attention with linear complexities. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 3531– 3539, 2021. 3
- [43] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556, 2014. 5
- [44] Vincent Sitzmann, Justus Thies, Felix Heide, Matthias Nießner, Gordon Wetzstein, and Michael Zollhofer. Deepvoxels: Learning persistent 3d feature embeddings. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2437–2446, 2019. 3
- [45] Cheng Sun, Min Sun, and Hwann-Tzong Chen. Direct voxel grid optimization: Super-fast convergence for radiance fields reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5459– 5469, 2022. 2
- [46] Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive network: A successor to transformer for large language models. arXiv preprint arXiv:2307.08621, 2023. 3
- [47] Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, et al. Learning to (learn at test time): Rnns with expressive hidden states. arXiv preprint arXiv:2407.04620, 2024. 2, 3
- [48] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In European Conference on Computer Vision, pages 1–18. Springer, 2024. 3
- [49] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 3
- [50] Johannes von Oswald, Nino Scherrer, Seijin Kobayashi, Luca Versari, Songlin Yang, Maximilian Schlegel, Kaitlin Maile, Yanick Schimpf, Oliver Sieberling, Alexander Meule-

- mans, et al. Mesanet: Sequence modeling by locally optimal test-time training. arXiv preprint arXiv:2506.05233, 2025. 3
- [51] Roger Waleffe, Wonmin Byeon, Duncan Riach, Brandon Norick, Vijay Korthikanti, Tri Dao, Albert Gu, Ali Hatamizadeh, Sudhakar Singh, Deepak Narayanan, et al. An empirical study of mamba-based language models. arXiv preprint arXiv:2406.07887, 2024. 3
- [52] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Vggt: Visual geometry grounded transformer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5294–5306, 2025. 2, 5
- [53] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5261–5271, 2025. 5
- [54] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20697– 20709, 2024. 2
- [55] Xinyue Wei, Kai Zhang, Sai Bi, Hao Tan, Fujun Luan, Valentin Deschaintre, Kalyan Sunkavalli, Hao Su, and Zexiang Xu. Meshlrm: Large reconstruction model for highquality meshes. arXiv preprint arXiv:2404.12385, 2024. 2
- [56] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. Depthsplat: Connecting gaussian splatting and depth. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 16453–16463, 2025. 3
- [57] Qiangeng Xu, Zexiang Xu, Julien Philip, Sai Bi, Zhixin Shu, Kalyan Sunkavalli, and Ulrich Neumann. Pointnerf: Point-based neural radiance fields. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5438–5448, 2022. 2
- [58] Jianing Yang, Alexander Sax, Kevin J Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3r: Towards 3d reconstruction of 1000+ images in one forward pass. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 21924–21935,

2025. 2

- [59] Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, and Yoon Kim. Parallelizing linear transformers with the delta rule over sequence length. Advances in neural information processing systems, 37:115491–115522, 2024. 3
- [60] Yao Yao, Zixin Luo, Shiwei Li, Tian Fang, and Long Quan. Mvsnet: Depth inference for unstructured multi-view stereo. In Proceedings of the European conference on computer vision (ECCV), pages 767–783, 2018. 2
- [61] Yao Yao, Zixin Luo, Shiwei Li, Tianwei Shen, Tian Fang, and Long Quan. Recurrent mvsnet for high-resolution multi-view stereo depth inference. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5525–5534, 2019. 2
- [62] Xuanyu Yi, Zike Wu, Qiuhong Shen, Qingshan Xu, Pan Zhou, Joo-Hwee Lim, Shuicheng Yan, Xinchao Wang, and

- Hanwang Zhang. Mvgamba: Unify 3d content generation as state space sequence modeling. Advances in Neural Information Processing Systems, 37:7580–7607, 2024. 3
- [63] Wang Yifan, Felice Serena, Shihao Wu, Cengiz Oztireli,¨ and Olga Sorkine-Hornung. Differentiable surface splatting for point-based geometry processing. ACM Transactions On Graphics (TOG), 38(6):1–14, 2019. 3
- [64] Alex Yu, Vickie Ye, Matthew Tancik, and Angjoo Kanazawa. pixelnerf: Neural radiance fields from one or few images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4578–4587, 2021. 2
- [65] Zehao Yu, Anpei Chen, Binbin Huang, Torsten Sattler, and Andreas Geiger. Mip-splatting: Alias-free 3d gaussian splatting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19447–19456,

2024. 6

- [66] Yuheng Yuan, Qiuhong Shen, Shizun Wang, Xingyi Yang, and Xinchao Wang. Test3r: Learning to reconstruct 3d at test time. arXiv preprint arXiv:2506.13750, 5, 2025. 2
- [67] Kai Zhang, Nick Kolkin, Sai Bi, Fujun Luan, Zexiang Xu, Eli Shechtman, and Noah Snavely. Arf: Artistic radiance fields. In European Conference on Computer Vision, pages 717–733. Springer, 2022. 2
- [68] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. Gs-lrm: Large reconstruction model for 3d gaussian splatting. In European Conference on Computer Vision, pages 1–19. Springer, 2024. 2, 3, 5, 6, 7
- [69] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In CVPR, 2018. 6
- [70] Tianyuan Zhang, Sai Bi, Yicong Hong, Kai Zhang, Fujun Luan, Songlin Yang, Kalyan Sunkavalli, William T Freeman, and Hao Tan. Test-time training done right. arXiv preprint arXiv:2505.23884, 2025. 2, 3, 5, 1
- [71] Chen Ziwen, Hao Tan, Kai Zhang, Sai Bi, Fujun Luan, Yicong Hong, Li Fuxin, and Zexiang Xu. Long-lrm: Longsequence large reconstruction model for wide-coverage gaussian splats. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4349–4359,

2025. 2, 3, 5, 6, 7

## tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction

### Supplementary Material

#### Contents

- A. Further Discussions 1
- B. Experiment Details 1

- B.1. Scene-level Training . . . . . . . . . . . . . 1
- B.2. Object-level training . . . . . . . . . . . . . 2

- C. More results and Comparison 2

#### A. Further Discussions

Effect of Scene Complexity on Fast Weights The memory of fast-weights has a fixed capacity and is bounded, especially in the autoregressive setting. Our empirical analysis on DL3DV scene labels indicates that higher scene complexity leads to degraded performance, as observed in outdoor vs. indoor scenes (PSNR: 24.45 vs. 24.96) and high- vs. low-frequency scenes (PSNR: 24.20 vs. 25.97). The memory capacity is also influenced by sequence length, where earlier inputs may be gradually forgotten as more tokens are processed.

Selective Update of Fast Weights in AR Setting Instead of updating the fast weights only according to current inputs, we can further use history states for selective update to mitigate drifiting. Inspired by [32], we explore a mechanism to prevent weight drift. Specifically, we approximate the diagonal of the Fisher information using an exponential moving average of squared gradients, as an estimate of parameter importance. Meanwhile, we maintain a sliding anchor via EMA to track the historical trajectory of the fast weights. After each gradient update, we apply elastic regularization based on parameter importance. Specifically, we leverage Fisher information for selective update, where parameters with high Fisher values that are important for the current input, are left parameters with high Fisher values unconstrained, while parameters with low Fisher values are pulled back toward the anchor. This encourages adaptation to the current input and suppresses drift in unimportant parameters. This training-free strategy can further improve our autoregressive model, and we envision it to be more effective by incorporating it into training for future work.

Table 6. Training-free selective update considering history fast weights can further enhance our AR model.

PSNR↑ SSIM↑ LPIPS↓ w/o selective 24.81 0.814 0.225

w selective 24.95 0.818 0.223

Scaling to More Input Views With distributed training, tttLRM can be further scaled to hundreds of views given enough compute. For example, by finetuning our full model with more iterations on 128 input views (more than 1M tokens), it can achieve 26.80 PSNR.

Possible Usage of Attention Layers We deliberately avoid attention blocks in our model since it has quadratic complexity O(N2d) compared to our linear FLOPS O(Nd2) of LaCT blocks (N is the number of tokens and d is hidden dimension). Therefore, attention will bottleneck the computation with growing number of tokens and be very slow in our million-level token setting. As shown in Figure 8, even a 3-layer attention only will be slower than our 24layer LaCT blocks from 2M tokens (256 views). With more compute, our model can easily scale to longer sequence and remain linear complexity.

3-Layer Attention (With Token Merging)

500

24-Layer LaCT (Ours)

400

Time(s)

300

200

100

0

0.5 1 2 4 8 Number of Tokens (M)

Figure 8. Time comparison of 3 Attention layers vs 24 layers of LaCT blocks under different numbers of tokens.

#### B. Experiment Details

##### B.1. Scene-level Training

We adopt a curriculum training strategy that progresses from low to high resolution, motivated by two main reasons. First, fast low-res pretraining enables the model to train with a large batch size with faster iteration time. Second, we found that even with pretrained TTT-LVSM [70] checkpoints at high-resolution (i.e. 960 × 540), the model cannot predict reasonable Gaussians at the beginning iterations, leading to excessive GPU memory usage due to the rendering of a large number of Gaussians.

For scene-level training, We train our model on three stages with 144×256, 288×512 and 540×960 resolution. For each stage, we resize the images to the target resolution, which all have the same aspect ratio as the original dataset. For all stages, we first determine a continuous range based on the start and end frames from the entire video sequence

from the dataset. The range is randomly sampled from 128 to 512 for each sample to ensure enough coverage of the scene. Then, we randomly sample 124 frames from this range, from which both input and target views will be further sampled. They are ensured to have overlap frames for stable training. We train the model across 16 to 64 input views. For training, we use the input views as the virtual views and found that provides the best results.

For the first stage, we train the model with a peak learn-

- ing rate of 3e−4 with 2K warmup steps and cosine decay. We use AdamW optimizer with betas (0.9,0.95) and weight decay 0.05. We train the model using a batch size of 128 for 80K steps, which is around 0.3T tokens. For the second stage, we finetune the model at the resolution of 288 × 512 with a peak learning rate of 5e−5. We use a batch size of 64 to train 6K steps. For the final stage, we enable depth loss and opacity loss, training the model with 32 input views with 5K steps with peak learning rate 1e−5 and batch size of 64. Finally, we train the model with 16 to 64 input views for another 1K steps. We prune 70% Gaussians with the smallest opacity for 64 views and 60% otherwise.

For autoregressive model training, we finetune our model on the final stage checkpoints for around another 3K iterations with peak learning rate 1e−4 and batch size of 64. We train the model on input views from 8 to 64. Our models are trained on 64 Nvidia A100 80GB GPUs.

Besides, we use gsplat Python library for efficient Gaussian training. We enable torch.compile to accelerate computation, achieving roughly a 30% per-iteration speedup. To further optimize memory and stability, we implement gradient checkpointing [6] and mixed-precision training [33] with the BFloat16 format. For Gaussian rendering, we utilize deferred backpropagation [67] to reduce GPU memory consumption. In addition, iterations with a gradient norm that exceeds 5.0 are skipped to improve training stability.

B.2. Object-level training

For the GS-based model, we use 8 views as input and another 8 views as supervision and use a patch size of 16×16. We firstly sample a set of 15 images (from 32 renderings) as a data point, from which we randomly select 8 input views and 8 supervision views independently. This sampling strategy encourages more overlap between input views and rendering views than directly sampling from 32 rendering views. We train on the resolution of 256 × 256 with a batch size of 512 for 80K iterations with a peak learn-

- ing rate of 4e−4. We then finetune on 512×512 with a batch size of 128 for another 10K iterations with a peak learning rate of 5e−5. We further finetune on 1024 × 1024 with a batch size of 64 for another 4K iterations with a peak learn-
- ing rate of 5e−5. For the triplane-based model, we use 4 views as input

and another 4 views as supervision and use a patch size of 16×16. We train on the resolution of 256×256 with a batch size of 256 for 60K iterations and finetune on 512×512 with a batch size of 64 for another 20K iterations.

#### C. More results and Comparison

In Table 7, we show results where we combine our method with a few additional optimization steps. It demonstrates that the reconstructed model can be further improved with minimal optimization cost, surpassing both purely optimization-based methods and the previous state-of-theart feed-forward method, Long-LRM, under the same postoptimization setup.

Notably, the quality of Long-LRM with 3-step postoptimization is still lower than our model without postoptimization, even though it requires more time to perform the optimization than our feedforward inference.

Table 7. More quantitative comparison on both DL3DV-140 and Tanks&Temples datasets under 32/64 input views. Our method surpasses previous feedforward methods and can further surpass optimization-based methods with a few steps post-optimization. Note that Long-LRM trains a separate model for each input view, while we are a single model across all input views. Our model can be linearly accelerated with multiple GPUs, here we report time on a single Nvidia A100 80GB GPU.

DL3DV-140 Tanks&Temples

Views Method Time↓

PSNR↑ SSIM↑ LPIPS↓ PSNR↑ SSIM↑ LPIPS↓

3D GS30k 13m 23.60 0.779 0.213 18.10 0.688 0.269 Mip-Splatting30k 13m 23.32 0.784 0.217 18.39 0.700 0.262

Scaffold-GS30k 16m 24.77 0.805 0.205 18.41 0.691 0.290 Long-LRM (32v model) 1s 24.10 0.783 0.254 18.38 0.601 0.363

Long-LRM (w/ 3-step optim) 12s 24.99 0.809 0.243 18.69 0.623 0.360 Long-LRM (w/ 10-step optim) 37s 25.60 0.826 0.233 18.90 0.642 0.350

32

Ours 7.2s 25.07 0.822 0.215 19.22 0.662 0.305 Ours (w/ 3-step optim) 18s 25.86 0.842 0.208 19.57 0.687 0.300 Ours (w/ 10-step optim) 42s 26.37 0.854 0.201 19.78 0.704 0.291

3D GS30k 13m 26.55 0.852 0.164 20.78 0.778 0.205 Mip-Splatting30k 13m 26.29 0.850 0.166 20.08 0.759 0.220 Scaffold-GS30k 16m 27.07 0.857 0.173 20.96 0.768 0.240 Long-LRM (64v model) 3.7s 24.63 0.799 0.243 19.11 0.627 0.346

Long-LRM (w/ 3-step optim) 38.9s 25.74 0.833 0.225 19.69 0.659 0.333 Long-LRM (w/ 10-step optim) 114s 26.72 0.852 0.212 20.03 0.681 0.320

64

Ours 14.8s 25.95 0.844 0.195 20.31 0.700 0.274 Ours (w/ 3-step optim) 47s 26.97 0.866 0.185 20.76 0.724 0.269 Ours (w/ 10-step optim) 124s 27.65 0.880 0.177 21.07 0.743 0.260

