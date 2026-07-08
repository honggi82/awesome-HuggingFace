arXiv:2403.09055v4[cs.CV]1Jun2025

# SemanticDraw: Towards Real-Time Interactive Content Creation from Image Diffusion Models

Jaerin Lee1 Daniel Sungho Jung2,3∗ Kanggeon Lee1 Kyoung Mu Lee1,2,3 1ASRI, Department of ECE, 2Interdisciplinary Program in Artificial Intelligence, 3SNU-LG AI Research Center, Seoul National University, Korea

{ironjr,dqj5182,dlrkdrjs97,kyoungmu}@snu.ac.kr

[Figure 1]

Figure 1. Overview. Our SEMANTICDRAW is a sub-second (0.64 seconds) solution for region-based text-to-image generation. This streaming architecture enables an interactive application framework, dubbed semantic palette, where image is generated in near instant interactivity based on online user commands of hand-drawn semantic masks.

### Abstract

We introduce SemanticDraw, a new paradigm of interactive content creation where high-quality images are generated in near real-time from given multiple hand-drawn regions, each encoding prescribed semantic meaning. In order to maximize the productivity of content creators and to fully realize their artistic imagination, it requires both quick interactive interfaces and fine-grained regional controls in their tools. Despite astonishing generation quality from recent diffusion models, we find that existing approaches for regional controllability are very slow (52 seconds for 512 × 512 image) while not compatible with acceleration methods such as LCM, blocking their huge potential in interactive

content creation. From this observation, we build our solution for interactive content creation in two steps: (1) we establish compatibility between region-based controls and acceleration techniques for diffusion models, maintaining high fidelity of multi-prompt image generation with ×10 reduced number of inference steps, (2) we increase the generation throughput with our new multi-prompt stream batch pipeline, enabling low-latency generation from multiple, region-based text prompts on a single RTX 2080 Ti GPU. Our proposed framework is generalizable to any existing diffusion models and acceleration schedulers, allowing subsecond (0.64 seconds) image content creation application upon well-established image diffusion models. The code is https://github.com/ironjr/semantic-draw.

### 1. Introduction

Recent massive advancements and widespread adoptions of generative AI [1, 42, 43, 45, 47, 59] are fundamentally transforming the landscape of content creation, demonstrating huge potential for improving efficiency of production processes and expanding the boundaries of creativity. Especially, diffusion models [45] are gaining significant attention in generative AI for image content creation because of their ability to produce realistic, high-resolution images. Nevertheless, in the perspective of content creators, a pure generative quality is not the only point of consideration [36]. Diffusion models for content creators should require efficient, interactive tools that can swiftly translate their artistic imaginations into refined outputs, supporting a more responsive and iterative creative process with finegrained controllability under straightforward control panels as illustrated in Figure 1 and 7. These goals should all be satisfied simultaneously.

The academic community had several attempts to address these criteria in isolated areas, but has yet to tackle them comprehensively. On one hand, there is a line of works dealing with acceleration of the inference speed [8, 26, 33, 34, 44, 52, 53] of diffusion models. Acceleration schedulers including DDIM [52], latent consistency models (LCM) [33, 34, 53], SDXL-Lightning [26], HyperSD [44], and Flash Diffusion [8] reduced the number of required inference steps from several thousand to a few tens and then down to 4. Focusing on the throughput directly, StreamDiffusion [21] reformed diffusion models into a pipelined architecture, enabling streamed generation and real-time video styling. On the other hand, methods to enhance the controllability [4, 5, 58, 59] of the generative framework were also heavily sought. ControlNet [59] and IP-Adapter [58] enabled image-based conditioning of the pre-trained diffusion models. SpaText [4] and MultiDiffusion [5] achieved image generation from multiple regionbased texts, allowing more fine-grained controls over the generation process from localized text prompts. Those two areas of research have developed largely independently. This suggests a straightforward approach for fast yet controllable generation: simply combining achievements from both, e.g., acceleration technique such as LCM [34] can serve a pair of a noise schedule sequence and fine-tuned model weights.

However, directly combining multiple works together does not work as intended. Figure 2 illustrates an example where diffusion models fail when extended to complex realworld scenarios. Here, inspired from the famous yet complex artwork of Korean royal folding screen, Irworobongdo (“Painting of the Sun, Moon, and the Five Peaks”)1, we generate an image of size 768 × 1920 from nine region-

1https://g.co/arts/9DESwLeAtdtaHkGv9

PromptandMaskMD[5]+LCM[40]()MultiDiffusion[5]()4m47s51m39sOurs(59s)

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

Text prompt: Background: “Clear deep blue sky”, Green: “Summer mountains”, Red: “The Sun”, Pale Blue: “The Moon”, Light Orange: “A giant waterfall”, Purple: “A giant waterfall”, Blue: “Clean deep blue lake”, Orange: “A large tree”, Light Green: “A large tree”

Figure 2. Example of large-size region-based text-to-image synthesis inspired by Korean traditional art, Irworobongdo. Our SEMANTICDRAW can synthesize high-resolution images from multiple, locally assigned text prompts with ×52.5 faster speed of convergence. The size of the image is 768 × 1920 and we use 9 text prompt-mask pairs including the background. The time is measured with a RTX 2080 Ti GPU. Note that time takes longer than regular sized images (e.g., 512 ×512) due to panoramic shape.

ally assigned text prompts as defined by a user under Figure 2. At this scale, previous state-of-the-art (SOTA) region-based controlling pipeline [5] fails to match the designated mask regions and text prompts despite its extremely slow and, hence, cautious reverse diffusion process. Applying a famous acceleration method LCM [34] on the diffusion model [5] does not solve high-latency problem, producing noisy output in the second row in Figure 2. This proves that the problem of controllability and acceleration cannot be scaled to real-world scenarios when simply com-

bining the existing diffusion models and acceleration methods, due to their poor compatibility.

Our goal is to build a real-time pipeline for image content creation, ready for interactive user applications. The system should be operated at least in near real-time, while maintaining stability of fine-grained regional controls. In the end, we propose SEMANTICDRAW which solves the problems from existing methods as shown in Figure 3. Elaborated in Section 3.2, we establish a stable pipeline for accelerated image synthesis with fine-grained controls, given through multiple, locally assigned text prompts. Building upon the rapid development from both acceleration schedulers [8, 26, 33, 34, 44] and network architectures [40, 45, 49] for diffusion models, we propose the first method to allow the acceleration schedulers to be compatible with region-based controllable diffusion models. We achieve up to ×50 speed-up of the multi-prompt generation while maintaining or even surpassing the image fidelity of the original algorithm [5].

Even after resolving the compatibility problem between the acceleration and controllability modules, generation throughput remains to be a main obstacle to interactive application. To this end, as illustrated in Section 3.3, we restructure our multi-prompt reverse diffusion process into a pipelined architecture [21], which we call the multi-prompt stream batch architecture. By bundling multi-prompt latents at different timesteps as a batched sequence of requests for image generation, we can perform the multi-prompt text-to-image synthesis endlessly by repeating a single, batched reverse diffusion. The result is a sub-second interactive image generation framework, achieving 1.57 FPS in a single 2080 Ti GPU. This high, stable throughput from SEMANTICDRAW allows a novel type of application for image content creation, named semantic palette, in which we can draw semantic masks in real-time to create an endless stream of images as in Figure 1 and 7. Our model-agnostic and acceleration-agnostic design allows the framework to be suitable for any existing diffusion pipelines [40, 45, 49]. We highly recommend readers to try our technical demo application of semantic palette in our Supplementary Material for better understanding.

### 2. Related Work

Accelerating Inference from Diffusion Models. Diffusion model [15, 45, 52] is a branch of generative models that sample target data distributions, e.g., images, videos, sounds, etc., by iteratively reducing randomness from pure noise. Its earliest form [15, 51, 52] traded off inference efficiency against sample diversity and quality. These have required thousands of iterations to generate a single image, raising a need for acceleration of inference to gain practicality. Majority of works [30, 31, 52] achieved speed through reformulating the reverse diffusion process.

DDIM [52] utilized a non-Markovian graphical model, and DPM-Solvers [30, 31] interpreted the generation process as Euler’s method for solving ordinary differential equations, cutting the required number of inference steps from thousands down to 20. Later, Consistency Models [53] exploited identity map boundary condition, and Flow Matching [28] adopted optimal transport for efficient sampling. These became the foundations of the most recent accelerated schedulers, including latent consistency model (LCM) [33, 34], SDXL-Lightning [26], Hyer-SD [44] and Flash Diffusion [8], which are utilized by large-scale latent diffusion models [10, 40, 45] in form of low-rank adaptations (LoRA) [16], a weight modifier upon baseline diffusion models. Alternatively, StreamDiffusion [21] introduced a novel pipelined architecture for video-to-video transfer, video stylization, and streamed image generation from a latent consistency model [33]. Our multi-prompt stream batch pipeline for interactive semantic drawing extends this philosophy with multi-prompt-based generation.

Controlling Generation from Diffusion Models. Enhanced controllability of diffusion models is another intensely investigated field of study. There are five major subgroups: (1) modifying from intermediate latent vectors, (2) modifying from inpainting masks, (3) attaching separate conditional branches, (4) connecting a subset of prompt tokens to positions in an image, and (5) enabling finer-grained generation from multiple, region-based prompts. The first group including ILVR [9], RePaint [32], and SDEdit [35] attempt to hijack the intermediate latent variables in the reverse process. SSI [24] accelerates this group of methods by utilizing locality of edit command. LazyDiffusion [39] take advantage of the transformer architecture to progressively edit and generate images within few seconds of latency, whereas our method builds upon arbitrary architecture and achieves sub-second generation time. The second major group utilizes the in-painting functionality [38] of diffusion models for editting [3, 17, 32, 38, 56]. After diffusion models have become massively publicized as image generation [2, 45] and editing [12, 19, 20, 29, 35, 37, 54, 57] tools, the demand for easier, modularized controls on behalf of professional creators has increased. In the third group, ControlNet [59] and IP-Adapter [58] introduce simple yet effective way to append image conditioning feature to existing pre-trained diffusion models. Our method applies orthogonally with the control methods in this group. Various text-conditioning [12, 19, 20, 29, 37] and imageconditioning [11, 54, 57] methods can also be placed in this group. The fourth group, including GLIGEN [25] and InstanceDiffusion [55] attach add-on modules to the diffusion model that focus on increasing the positional accuracy of a single prompt. Alternatively, we are mainly interested in a scenario where image diffusion models continuously cre-

Background: “A photo of a Greek temple”, Yellow: “A photo of God Zeus with arms open”, Red: “A tiny sitting eagle”

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

(a) Prompt (b) MD 50steps 52s (c) MD+LCM 5steps 5s (d) +PreAvg 5steps 5s (e) +Bstrap 5steps 5s (f) +QMask(Ours)5steps 5s

- Figure 3. Our SEMANTICDRAW enables fast region-based text-to-image generation by stable acceleration of MultiDiffusion [5]. PreAvg, Bstrap, and QMask stand for the latent pre-averaging, mask-centering bootstrapping, and quantized masks, our first three proposed strategies. Each method used in (d), (e), (f) contains the method used in the previous image. The images are single tiles of size 768 × 512 .

ate new images from multiple, dynamically moving, regionally assigned text prompts. This is most related to the final group [4–6, 41] which focus on controlling the semantic composition of the generated images.

Content Creation from Regional Text Prompts. The last group mentioned above provides a way to flexibly integrate multiple regionally assigned text prompts into a single image. SpaText [4] achieves generation from multiple spatially localized text prompts by utilizing CLIP-based spatio-temporal representation. Differential Diffusion [22] and Mixture of Diffusers [6] similarly operate on maskbased generation but differs in their approaches to overlapping regions and noise addition. MultiDiffusion [5] and more recent LRDiff [41] present simple yet effective way to generate from multiple different semantic masks: to iteratively decompose and recompose the latent images according to different regional prompts during reverse diffusion process. This simple formulation works not only with irregular-shaped regions, but also with irregular-sized canvases. However, as mentioned in Section 1 and depicted in Figure 2, this breakthrough has not been developed in aware of modern acceleration methods, reducing their practical attraction in this era of rapid diffusion models. Starting from the following section, we will establish the compatibility between these type of pipeline architecture with accelerated samplers. This opens a new type of semantic drawing application, SEMANTICDRAW, where users draw images interactively with brush-type tools that paints semantic meanings as shown in Section 5.

### 3. Method

#### 3.1. Preliminary

A latent diffusion model (LDM) [45] ϵθ is an additive Gaussian noise estimator defined over a latent space. The model

ϵθ receives a combination of a noisy latent x, a text prompt embedding y, and a timestep t ∈ [0,T]. It outputs an estimation of the noise ϵ that was mixed with the true latent x0 . At inference, the diffusion model ϵθ is consulted multiple times to estimate a latent xˆ0 ≈ x0, which correlates to the information described in the conditional input y , starting from a pure noise xT ∼ N(0,1)HWD . Each of the recursive calls to the reverse diffusion process can be expressed as a summation of a denoising term and a noise-adding term to the intermediate latent:

= STEP(xt

,y,i,ϵ;ϵθ,α,t), (1)

##### xt

i−1

i

where, we denote i as the index of the current time step ti . Note that the newly added noise ϵ depends on the type of scheduler.

Although this abstract form embraces almost every generation algorithm of diffusion models [15, 30, 52], it does not consider practical scenarios of our interest: (1) when the desired shape (H′ × W′) of the latent xˆ′0 is different from that of the training set (H × W) or (2) multiple text prompts y1,...,yp correlate to different regions of the generated images. MultiDiffusion [5] is one of the pioneers to deal with this problem. Their main idea is to aggregate (AGGRSTEP) multiple overlapping tiles of intermediate latents with simple averaging. That is, for every sampling step ti , perform:

##### x′t

= AGGRSTEP(x′t

,y,i,W; STEP) (2)

i−1

i

STEP(crop(w ⊙ x′t

),yw,i,ϵ) w∈W w

= w∈W

, (3)

i

where ⊙ is an element-wise multiplication, w ∈ W ⊂ {0,1}H

′W′ is a binary mask for each latent tile, yw is a conditional embedding corresponding to the tile w , and crop is a cropping operation to chop down large x′t

into tiles of same size as training image latents.

i

[Figure 12]

(a) Bootstrapping strategy overview.

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

(b) Multi-prompt stream batch architecture.

- Figure 4. SEMANTICDRAW pipeline. Our acceleration technique for region-based multi-prompt generation consists of three strategies. Figure 4a summarizes the first two of three: (1) latent pre-averaging and (2) mask-centering bootstrapping. In Figure 4b, we devise multiprompt stream batch pipeline that aggregates foreground and background latents from different time steps to maximize the throughput of generation, enabling near real-time content creation. Further, text embeddings are cached for interactive brush-like interface, elaborated in Section 5. Our method can be applied to arbitrary diffusion pipelines. We also provide the full algorithm in the Supplementary Material.

#### 3.2. Acceleration-Compatible Regional Controls

Our objective is to build an accelerated solution to image generation from multiple regionally assigned text prompts. Unfortunately, simply replacing the Stable Diffusion (SD) model [10, 40, 45] with an acceleration module, such as Latent Consistency Model (LCM) [33] or SDXLLightning [26], etc., and updating the default DDIM sampler [52] with the corresponding accelerated sampler [18, 33] does not work in general. This incompatibility greatly limits potential applications of both acceleration [8, 26, 33, 44] and region-based control techniques [4, 5]. We discuss each of the causes and seek for faster and stronger alternatives. In summary, our stabilization trick consists of three strategies: (1) latent pre-averaging, (2) maskcentering bootstrapping, and (3) quantized masks.

Step 1: Achieving Compatibility through Latent PreAveraging. The primary reason for the blurry image of the second row of Figure 2 is that the previous algorithm [5] is not aware of different types of underlying reverse diffusion step functions STEP. While the reverse diffusion algorithms can be categorized into two types: (1) additional noise at each step [30, 33], (2) no additional noise at each step [5, 52], the previous SOTA region-based controllable method [5] falls into the latter. Hence, applying the averaging aggregation of the method [5] cancels the prompt-wise added noises in STEP, which leads to overly smooth latents. We can avoid this problem with a simple workaround. First, we split the STEP function into a deterministic denoising part (DENOISE) and an optional noise addition:

ϵ (4)

= x˜t

##### xt

+ ηt

i−1

i−1

i−1

= DENOISE(xt

ϵ, (5)

,y,i;ϵθ,α,t) + ηt

i−1

i

where ηt is an algorithm-dependent parameter. The averaging of equation (3) is then applied to the output of the denoising part x˜t

, instead of the output of the full step xt

i−1

. Note that the noise is added after aggregation step.

i−1

##### x′t

= AGGRSTEP(x′t

,y,i,W; DENOISE) + ηt

##### ϵ.

i−1

i−1

i

(6) As it can be seen in Figure 3d, this change alleviates the compatibility issue with acceleration methods like LCM.

Step 2: Mask-Centering Bootstrapping for Few-Step Generation. The second cause of the incompatibility lies in the bootstrapping stage of the previous method [5]. MultiDiffusion [5] introduced bootstrapping stages that replace the background latents with random colors in the first 40% of total steps. This is performed to cut out the generated regions outside of object masks, which claims to enhance mask-fidelity. In original form, the perturbation introduced by the bootstrapping cancels out during long inference steps. However, as we decrease the number of timesteps in ten-fold from n = 50 steps to n = 4 or 5 steps, the number of bootstrapping stage is reduced down to n = 2. Regrettably, this magnifies the effect of perturbation introduced by the random color latents in the bootstrapping phase, and results in leakage of mixed colors onto the final image as shown in Figure 3. Instead, we propose to use a mixture of white background and aggregation of contents co-generated from other regional prompts (blue in Figure 4a), which alleviates the problem and allows compatibility with the accelerated generation as seen in Figure 3e.

Furthermore, we empirically found that first two steps of reverse diffusion process determine the overall structure of generated images when sampling with accelerated schedulers. Even after the first step, the network formulates the

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

| | |
|---|---|
| | |

- Figure 5. Quantized mask. As the last of three stabilizing techniques, binary masks are blurred and quantized by scheduler noise level to trade off between mask-fidelity and overall harmonization. See Supplementary Material for utilizing this trade-off.

rough structure of the objects being created. The problem is that diffusion models exhibit a strong tendency to generate screen-centered objects than off-centered ones, following image datasets [50] they are trained for. After the first step, the object for every mask is generated at the center of the screen, not at the center of the mask. Off-centered objects are often masked out by the pre-averaging step (yellow in Figure 4a). The final results often neglect small, offcentered regional prompts, and the large objects are often unnaturally cut, lacking harmonization within the image. To prevent this, we propose mask centering strategy (pink in Figure 4a) to exploit the center-bias of the diffusion model. Especially, for the first two steps of generation, we shift the intermediate latents from each prompt to the center of the frame before being handled by the noise estimator ϵθ . The result of Step 2 can be seen in Figure 3e.

Step 3: Quantized Mask for Seamless Generation. Another problem from the reduced number of inference steps is that harmonization of the generated content becomes more difficult. As Figure 3e shows, all the objects appear to be salient and their abrupt boundaries are visible between regions. This is because the number of later sampling steps that contribute to the harmonization is now insufficient. In contrast, the baseline with long reverse diffusion steps 3b effectively smooth out the mask boundaries by consecutively adding noises and blurring them. To mitigate this issue, we develop an alternative way to seamlessly amalgamate generated regions: quantized masks, shown in Figure 5. Given a binary mask, we obtain a smoothened mask by applying Gaussian blur. Then, we quantize the realnumbered mask by the noise levels of the diffusion sampler. As Figure 4a illustrates, for each denoising step, we use a mask with corresponding noise level. Since the noise levels monotonically decrease throughout iterations, the coverage of a mask gradually increases along with each sampling step, gradually mixing the boundary regions. The final result can be seen from Figure 3f. This relaxation of semantic masks also provides intuitive interpretation of brushes, one of the most widely used tool in professional graphics editing

software. We will revisit this interpretation in Section 5.

#### 3.3. Optimization for Throughput

As mentioned in Section 1, achieving real-time response is important for practical end-user application. Inspired by StreamDiffusion [21], we reconstruct our region-based textto-image synthesis framework into a pipelined architecture to maximize the throughput of image generation.

Multi-Prompt Stream Batch Architecture. Figure 4b illustrates the architecture and the interfaces of our pipeline. Instead of the typical mini-batched use of diffusion model with synchronized timesteps, the noise estimator is fed with a new input image every timestep along with the last processed batch of images. In other words, each image in a mini-batch has different timestep. This architecture hides the latency caused by multi-step algorithm of reverse diffusion. Restructuring our stabilized framework in 4a takes several steps. The quantized masks, the background images, the noises, and the prompt embeddings differ along each timesteps and should be saved separately. Instead of a single image, we change the architecture to process a minibatch of images of different prompts and masks to the UNet at every timestep, as depicted in Figure 4b. We call this the multi-prompt stream batch architecture. To further reduce the latency, we added asynchronous pre-calculation step applied only when a user command changes the configuration of the text prompts and masks. This allows interactive brush-like interfaces elaborated in Section 5.

Optimizing Throughput. Additional increase of throughput can be achieved by using a compressed autoencoder such as Tiny AutoEncoder [7]. Detailed analysis on the effect of throughput optimization is in Table 6.

### 4. Experiment

We provide comprehensive evaluation of our SEMANTICDRAW using various types of acceleration modules and samplers. We compare our experiments based on the public checkpoints of Stable Diffusion 1.5 [45], SDXL [40], and SD3 [49]. However, we note that our method can be applied to any community-trained models using DreamBooth [46]. More results can be found in Section S2 of our Supplementary Materials.

#### 4.1. Quality of Generation

Generation from Multiple Region-Based Prompts. We first demonstrate the stability and speed of our algorithm for image generation from multiple regionally assigned text prompts. The evaluation is based on COCO validation dataset [27], where we generate images from the image captions as background prompts and object masks with categories as foreground prompts. The public latent diffusion

###### Table 1. Comparison of generation from region-based prompts between DDIM [52] (default) and LCM [33] sampler.

Method Sampler FID ↓ IS ↑ CLIPfg ↑ CLIPbg ↑ Time (s) ↓ SD1.5 (512 × 512) MultiDiffusion (Ref.) DDIM [52] 70.93 16.24 24.09 27.55 14.1

MultiDiffusion (MD) LCM [33] 270.55 2.653 22.53 19.63 1.7 SemanticDraw (Ours) LCM [33] 93.93 14.12 24.14 24.00 1.3

###### Table 2. Comparison of generation from region-based prompts between DDIM [52] (default) and Hyper-SD [44] sampler.

Method Sampler FID ↓ IS ↑ CLIPfg ↑ CLIPbg ↑ Time (s) ↓ SD1.5 (512 × 512) MultiDiffusion (Ref.) DDIM [52] 70.93 16.24 24.09 27.55 14.1

MultiDiffusion (MD) Hyper-SD [44] 168.34 10.12 20.08 15.90 1.7 SemanticDraw (Ours) Hyper-SD [44] 98.60 14.90 24.48 23.31 1.3

###### Table 3. Comparison of generation from region-based prompts between DDIM [52] (default) and Euler Discrete [18] sampler.

Method Sampler FID ↓ IS ↑ CLIPfg ↑ CLIPbg ↑ Time (s) ↓ SDXL (1024 × 1024) MultiDiffusion (Ref.) DDIM [52] 73.77 16.31 24.16 28.11 50.6

MultiDiffusion (MD) EulerDiscrete [18] 572.95 1.328 21.02 17.36 4.3 SemanticDraw (Ours) EulerDiscrete [18] 84.27 15.04 24.19 24.22 3.6

###### Table 4. Comparison of generation from region-based prompts between Flow Match Euler Discrete [10] (default) and Flash Flow Match Euler Discrete [8] sampler.

Method Sampler FID ↓ IS ↑ CLIPfg ↑ CLIPbg ↑ Time (s) ↓ SD3 (1024 × 1024) MultiDiffusion (Ref.) FlowMatch [10] 166.42 8.517 20.66 16.39 46.3

MultiDiffusion (MD) FlashFlowMatch [8] 209.36 5.347 19.83 14.48 4.0 SemanticDraw (Ours) FlashFlowMatch [8] 79.2 17.41 23.59 27.83 3.2

models [40, 45, 49] are trained for specific range of image sizes, and reportedly fail when given image sizes are small. Since COCO datasets consists of relatively small images compared to the default size the models were trained for, we rescale the object masks with nearest neighbor interpolation to the default size of each model. This is 512×512 for SD1.5 [45] and 1024 × 1024 for SDXL [40] and SD3 [49]. To compare the image fidelity, we use Fr´echet Inception Distance (FID) [14] and Inception Score (IS) [48]. We also use CLIP scores [13] to compare the text prompt fidelity. We separate the foreground score (CLIPfg), which is obtained by taking the average CLIP score between each generated image and corresponding set of foreground object categories, from the background score (CLIPbg), which is a measured between images and their corresponding COCO captions. Tables 1 through 4 summarizes the results.

We implement MultiDiffusion [5] for SDXL [40] and SD3 [49] simply by changing the pipelines, accelerator LoRAs [16], and schedulers, from the official implementation. Even though schedulers with higher numbers of iterations generally produce better quality images [52], the tables show that our accelerated pipeline achieves comparable quality with more than ×10 reduction of time. These results demonstrate that our method provides universal acceleration under different types of diffusion pipelines (SD1.5 [34]

Background: “Plain wall”, Yellow: “A desk”, Red: “A flower vase”, Blue: “A window”

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Background: “A photo of backyard”, Yellow: “A yellow bird”, Red: “A red bird”

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Background: “A floor”, Yellow: “A box”, Red: “A tiny head of a cat”

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Background: “A photo”, Yellow: “A smiling girl”, Red: “A cool beret hat”, Blue: “Sky at noon”

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

(d) Ours, 5 steps

(a) Prompt (b) MD, 50 steps (c) MD+LCM, 5 steps

Figure 6. Region-based text-to-image synthesis results. Our stabilization methods accelerate MultiDiffusion [5] up to ×10 while preserving quality.

, SDXL [40], SD3 [10]), noise schedulers (DDIM [52], LCM [33], Euler Discrete [18], Flow Match Euler Discrete [10]), and acceleration methods (LCM [33], Lightning [26], Hyper-SD [44], Flash Diffusion [8]), without compromising the visual quality. Figure 6 shows a random subset of generation from the experiments in Table 1. Comparable visual quality from our method is consistent to the quantitative comparisons.

Stabilized Acceleration of Region-Based Generation. Next, we evaluate the effectiveness of each stabilization step introduced in Section 3.2. Figure 3 and Table 5 summarize the result on region-based text-to-image generation from the same setup as Table 1. Applying each strategy consistently boosts both perceptual quality, measured by FID score [14], and text prompt-fidelity, measured by the two CLIP scores [13]. This reveals that our techniques help alleviating the incompatibility as intended.

Throughput Maximization. Table 6 compares the effect of throughput optimization. We have already achieved ×9.7 speed-up by establishing the compatibility with acceleration modules. This is further enhanced though our multi-

- Table 5. Ablation on the effectiveness of our stabilization techniques on the fidelity of region-based generation. Method FID ↓ CLIPfg ↑ CLIPbg ↑

No stabilization 270.55 22.53 19.63 + Latent pre-averaging 80.64 22.80 26.95 + Mask-centering bootstrapping 79.54 23.06 26.72 + Quantized masks (σ = 4) 78.21 23.08 26.72

- Table 6. Ablations on throughput optimization techniques, measured with a single RTX 2080 Ti. Images of 512 × 512 are generated from three prompt-mask pairs.

Method Throughput (FPS) Relative Speedup

Baseline [5] 0.0189 ×1.0 + Stable Acceleration 0.183 ×9.7

+ Multi-Prompt Stream Batch 1.38 ×73.0 + Tiny AutoEncoder [7] 1.57 ×83.1

Table 7. User preference regarding quality of generation.

Method Ours MD [5] MD [5]+LCM [33] Preference 90.9% 9.1% 0.0%

prompt stream batch architecture. With low-memory autoencoder [7] to trade quality off for speed, we could finally achieve 1.57 FPS (0.64 seconds per frame). This near real-time, sub-second generation speed is a necessary step towards practical applications of generative models.

User Study. Finally, we conduct a user study on the methods of Table 1. Its result, summarized in Table 7, shows that our method greatly increases generation quality with multiple region-based text prompts.

### 5. Semantic Draw

Our real-time interface of SEMANTICDRAW opens up a new paradigm of user-interactive application for image generation. We discuss the key features of the application.

Concept. Responsive region-based text-to-image synthesis enabled by our streaming pipeline allows users to edit their prompt masks similarly to drawing. Since the standard text encoding by large text encoders (e.g., CLIP) accounts for approximately 40% of our sub-second runtime (1.57 FPS), caching and reusing these encodings when only mask modifications occur hides this latency and provides even faster feedback to users. This allows them to iteratively refine their commands according to the generated image. In scenarios where users change text prompts, standard text processing occurs while still maintaining the original sub-second runtime. This enables users to paint with text prompts just like they can paint a drawing with colored brushes, hence the name: SEMANTICDRAW.

[Figure 29]

[Figure 30]

(a) Semantic Draw. (b) Streaming Semantic Draw.

Figure 7. Screenshot of the sample applications of SEMANTICDRAW. After registering prompts and optional background image, the users can create images in real-time by drawing with text prompts. We invite the readers to play with the application.

Sample Application Design. We briefly describe our minimal demo application in Figure 7. The application consists of a front-end user interface and a back-end server that runs SEMANTICDRAW. Each user input is either a modification of the background image, the text prompts, the masks, and the tweakable options for the text prompts and the masks such as mix ratios and blur strengths. When commanding major changes requiring preprocessing stages, such as a change of prompts or the background, the backend pipeline is flushed and reinitialized with the newly given context. Otherwise, the pipeline is repeatedly called to obtain a stream of generated images. The user first selects the background image and creates a palette of semantic masks by entering a pair of positive and negative text prompts. The user can then draw masks corresponding to the created palette with a familiar brush tool, a shape tool, or a paint tool. The application automatically generates a stream of synthesized images according to user inputs. We gently invite the readers to play with our technical demo provided with the official code2.

### 6. Conclusion

We proposed SEMANTICDRAW, a new type image content creation where users interactively draw with a brush tool that paints semantic masks to endlessly and continuously create images. Enabling this application required high generation throughput and well-established compatibility between regional control pipelines and acceleration schedulers. We devised multi-prompt regional control pipeline that is both scheduler-agnostic and model-agnostic in order to maximize the compatibility. We further proposed multiprompt stream batch architecture to build a near real-time, highly interactive image content creation system for professional usage. Our SEMANTICDRAW achieves up to ×50 faster generation of large scale images than the baseline, bringing the latency of multi-prompt irregular-sized generation down to a practically meaningful bounds.

2https://github.com/ironjr/semantic-draw

### Acknowledgment

This work was supported in part by the IITP grants [No.2021-0-01343, Artificial Intelligence Graduate School Program (Seoul National University), No. 2021-0-02068, and No.2023-0-00156], and the NOTIE grant (No. RS2024-00432410) by the Korean government.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 2

- [2] AUTOMATIC1111. Stable diffusion WebUI. https://github.com/AUTOMATIC1111/stablediffusion-webui, 2022. 3
- [3] Omri Avrahami, Dani Lischinski, and Ohad Fried. Blended diffusion for text-driven editing of natural images. In CVPR,

2022. 3

- [4] Omri Avrahami, Thomas Hayes, Oran Gafni, Sonal Gupta, Yaniv Taigman, Devi Parikh, Dani Lischinski, Ohad Fried, and Xi Yin. SpaText: Spatio-textual representation for controllable image generation. In CVPR, 2023. 2, 4, 5
- [5] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. MultiDiffusion: Fusing diffusion paths for controlled image generation. In ICML, 2023. 2, 3, 4, 5, 7, 8, 12, 13, 15, 16, 17, 19
- [6] Alvaro´ Barbero Jim´enez. Mixture of Diffusers for scene composition and high resolution image generation, 2023. 4
- [7] Ollin Boer Bohan. Tiny autoencoder for stable diffusion. https://github.com/madebyollin/taesd,

2023. 6, 8

- [8] Clement Chadebec, Onur Tasar, Eyal Benaroche, and Benjamin Aubin. Flash Diffusion: Accelerating any conditional diffusion model for few steps image generation. In AAAI,

2025. 2, 3, 5, 7, 12, 15, 16, 19, 21, 22

- [9] Jooyoung Choi, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. ILVR: Conditioning method for denoising diffusion probabilistic models. In ICCV, 2021. 3
- [10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In ICML, 2024. 3, 5, 7
- [11] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. AnimateDiff: Animate your personalized text-toimage diffusion models without specific tuning. In ICLR,

2023. 3

- [12] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-or. Prompt-to-prompt image editing with cross-attention control. In ICLR, 2022. 3
- [13] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. CLIPScore: A reference-free evaluation metric for image captioning. In EMNLP, 2021. 7

- [14] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs trained by a two time-scale update rule converge to a local nash equilibrium. In NeurIPS, 2017. 7
- [15] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020. 3, 4
- [16] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. LoRA: Lowrank adaptation of large language models. In ICLR, 2021. 3, 7
- [17] Xuan Ju, Xian Liu, Xintao Wang, Yuxuan Bian, Ying Shan, and Qiang Xu. BrushNet: A plug-and-play image inpainting model with decomposed dual-branch diffusion. In ECCV,

2024. 3

- [18] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. In NeurIPS, 2022. 5, 7
- [19] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In CVPR, 2023. 3
- [20] Gwanghyun Kim, Taesung Kwon, and Jong Chul Ye. DiffusionCLIP: Text-guided diffusion models for robust image manipulation. In CVPR, 2022. 3
- [21] Akio Kodaira, Chenfeng Xu, Toshiki Hazama, Takanori Yoshimoto, Kohei Ohno, Shogo Mitsuhori, Soichi Sugano, Hanying Cho, Zhijian Liu, and Kurt Keutzer. StreamDiffusion: A pipeline-level solution for real-time interactive generation. arXiv preprint arXiv:2312.12491, 2023. 2, 3, 6
- [22] Eran Levin and Ohad Fried. Differential diffusion: Giving each pixel its strength, 2023. 4
- [23] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In ICML,

2023. 22

- [24] Muyang Li, Ji Lin, Chenlin Meng, Stefano Ermon, Song Han, and Jun-Yan Zhu. Efficient Spatially Sparse Inference for Conditional GANs and Diffusion Models. In NeurIPS,

2022. 3

- [25] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. GLIGEN: Open-Set Grounded Text-to-Image Generation. In CVPR, 2023. 3
- [26] Shanchuan Lin, Anran Wang, and Xiao Yang. SDXLLightning: Progressive adversarial diffusion distillation. arXiv preprint arXiv:2402.13929, 2024. 2, 3, 5, 7, 12, 15, 16, 19, 21, 22
- [27] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft COCO: Common objects in context. In ECCV, 2014. 6
- [28] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow Matching for Generative Modeling. In ICLR, 2023. 3
- [29] Xihui Liu, Dong Huk Park, Samaneh Azadi, Gong Zhang, Arman Chopikyan, Yuxiao Hu, Humphrey Shi, Anna

- Rohrbach, and Trevor Darrell. More control for free! Image synthesis with semantic diffusion guidance. In WACV, 2023. 3
- [30] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-Solver: A fast ODE solver for diffusion probabilistic model sampling in around 10 steps. In NeurIPS, 2022. 3, 4, 5
- [31] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-Solver++: Fast solver for guided sampling of diffusion probabilistic models. arXiv preprint arXiv:2211.01095, 2022. 3
- [32] Andreas Lugmayr, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. RePaint: Inpainting using denoising diffusion probabilistic models. In CVPR,

2022. 3

- [33] Simian Luo, Yiqin Tan, Longbo Huang, Jian Li, and Hang Zhao. Latent consistency models: Synthesizing highresolution images with few-step inference. arXiv preprint arXiv:2310.04378, 2023. 2, 3, 5, 7, 8, 12, 15, 16, 19, 21, 22
- [34] Simian Luo, Yiqin Tan, Suraj Patil, Daniel Gu, Patrick von Platen, Apolin´ario Passos, Longbo Huang, Jian Li, and Hang Zhao. LCM-LoRA: A universal stable-diffusion acceleration module. arXiv preprint arXiv:2311.05556, 2023. 2, 3, 7, 12, 15, 16, 19, 21, 22
- [35] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. SDEdit: Guided image synthesis and editing with stochastic differential equations. In ICLR, 2022. 3
- [36] Niloy J. Mitra, Duygu Ceylan, Or Patashnik, Danny CohenOr, Paul Guerrero, Chun-Hao Huang, and Minhyuk Sung. Diffusion models for visual content creation. In SIGGRAPH Tutorial, 2024. 2
- [37] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In CVPR, 2023. 3
- [38] Alexander Quinn Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob Mcgrew, Ilya Sutskever, and Mark Chen. GLIDE: Towards photorealistic image generation and editing with text-guided diffusion models. In ICML, 2022. 3
- [39] Yotam Nitzan, Zongze Wu, Richard Zhang, Eli Shechtman, Daniel Cohen-Or, Taesung Park, and Micha¨el Gharbi. Lazy Diffusion Transformer for Interactive Image Editing. In ECCV, 2024. 3
- [40] Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas M¨uller, Joe Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In ICLR, 2024. 2, 3, 5, 6, 7, 22
- [41] Zipeng Qi, Guoxi Huang, Chenyang Liu, and Fei Ye. Layered Rendering Diffusion Model for Controllable Zero-Shot Image Synthesis. In ECCV, 2024. 4, 16
- [42] Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, and Ilya Sutskever. Zero-shot text-to-image generation. In ICML, 2021. 2
- [43] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image gen-

eration with clip latents. arXiv preprint arXiv:2204.06125,

- 2022. 2

- [44] Yuxi Ren, Xin Xia, Yanzuo Lu, Jiacheng Zhang, Jie Wu, Pan Xie, XING WANG, and Xuefeng Xiao. Hyper-SD: Trajectory segmented consistency model for efficient image synthesis. In NeurIPS, 2024. 2, 3, 5, 7, 12, 15, 16, 19, 21, 22
- [45] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022. 2, 3, 4, 5, 6, 7, 16, 22
- [46] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, 2023. 6
- [47] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022. 2
- [48] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved Techniques for Training GANs. In NeurIPS, 2016. 7
- [49] Axel Sauer, Frederic Boesel, Tim Dockhorn, Andreas Blattmann, Patrick Esser, and Robin Rombach. Fast highresolution image synthesis with latent adversarial diffusion distillation. In SIGGRAPH Asia, 2024. 3, 6, 7, 22
- [50] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. LAION400M: Open dataset of CLIP-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 6
- [51] Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep Unsupervised Learning using Nonequilibrium Thermodynamics. In ICML,

2015. 3

- [52] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In ICLR, 2020. 2, 3, 4, 5, 7, 16
- [53] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In ICML, 2023. 2, 3, 12, 21
- [54] Xuan Su, Jiaming Song, Chenlin Meng, and Stefano Ermon. Dual diffusion implicit bridges for image-to-image translation. In ICLR, 2022. 3
- [55] Xudong Wang, Trevor Darrell, Sai Saketh Rambhatla, Rohit Girdhar, and Ishan Misra. InstanceDiffusion: Instance-level Control for Image Generation. In CVPR, 2024. 3
- [56] Shaoan Xie, Zhifei Zhang, Zhe Lin, Tobias Hinz, and Kun Zhang. SmartBrush: Text and shape guided object inpainting with diffusion model. In CVPR, 2023. 3
- [57] Binxin Yang, Shuyang Gu, Bo Zhang, Ting Zhang, Xuejin Chen, Xiaoyan Sun, Dong Chen, and Fang Wen. Paint by example: Exemplar-based image editing with diffusion models. In CVPR, 2023. 3
- [58] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. IPAdapter: Text compatible image prompt adapter for text-toimage diffusion models. arXiv preprint arXiv:2308.06721,

- 2023. 2, 3

###### [59] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023. 2, 3

# SemanticDraw: Towards Real-Time Interactive Content Creation from Image Diffusion Models

## Supplementary Material

### Abstract

Section S1 shows implementation details of our acceleration methods. In Section S2, additional visual results are shown. Finally, we provide our demo application as we have promised in our main manuscript. Our formulation introduces new controllable hyperparameters that users may interact in order to create images that respect their intentions. Section S3 demonstrates how our new tool can be used in image content creation.

### S1. Implementation Details

We begin by providing additional implementation details.

#### S1.1. Acceleration-Compatible Regional Controls

Algorithm S1 compares between the the baseline MultiDiffusion [5] and our stabilized sampling from multiple regionally assigned text prompts introduced in Section 3.2 of the main manuscript. As we have discussed in Section 3 of the main manuscript, improper placing of the aggregation step and strong interference of its bootstrapping strategy limit the ability to generate visually pleasing images under modern fast inference algorithms [8, 26, 33, 34, 44, 53]. Therefore, we instead focus on changing the bootstrapping stage of line 9-13 and the diffusion update stage of line 14-15 of Algorithm S1 in order to establish compatibility to accelerating diffusion samplers.

The resulting Algorithm S2 developed in Section 3.2 of the main manuscript achieves this. The differences between our approach from the baseline inference algorithm are marked with blue. First, in line 10, we change the bootstrapping background color to white. Having extremely low number of sampling steps (4-5), this bootstrapping background is easily leaked through the final image as seen in Figure 3 of the main manuscript. We notice that white backgrounds are common in public image datasets on which the diffusion models are trained. Therefore, changing random background images into white backgrounds alleviate this leakage problem.

Diffusion models have a strong tendency to generate objects at the center of the frame. This positional bias makes generation from small, off-centered masks difficult especially in the accelerated sampling, where the final structure of generated images are determined at the first two inference steps. By masking with off-centered masks, the objects under generation are unnaturally cut, leading to

User Cmd t t + 1 t + 2 t + 3 t + 4 t + 5

| | |
|---|---|
|1: initialize<br>2: no-op<br>3: draw mask<br>4: edit prompt<br>5: edit mask<br>6: no-op<br>||E|
|---|
<br><br>U0 U1 U2 U3 U4 D U0 U1 U2 U3 U4 D<br><br>U0 U1 U2 U3 U0 U1 U2 U0 U1 U0<br><br>|E|
|---|
<br><br>|

Figure S1. Example execution process of Multi-Prompt Stream Batch pipeline of SEMANTICDRAW. By aggregating latents at different timesteps a single batch, we can maximize throughput by hiding the latency.

defective generations. Lines 13-14 of Algorithm S2 are our mask-centering stage for bootstrapping to alleviate this problem. In the first few steps of generation, for each mask-designated object, intermediate latents are masked then shifted to the center of the object bounding box. This operation enforces the denoising network to focus on each foreground object located at the center of the screen. Lines 17-19 of Algorithm S2 undo this centering operation done in lines 13-14. The separately estimated foreground objects are aggregated into the single scene by shifting them back to their original absolute positions.

Finally, a single reverse diffusion step in line 14 of Al-

- gorithm S1 is split into the denoising part in line 16 of Al-
- gorithm S2 and the noise addition part in line 24 of Algorithm S2. As we have discussed with visual example in Figure 3c in the main manuscript, this simple augmentation of the original MultiDiffusion [5] stabilizes the algorithm to work with fast inference techniques such as LCMLoRA [33, 34], SDXL-Lightning [26], Hyper-SD [44], and Flash Diffusion [8]. Also refer to panorama generation in Figure S7 where this wrongly placed aggregation after STEP operation causing extremely blurry generation under accelerating schedulers [33, 34]. The readers may also consult our submitted code for the implementation of Algorithm S2.

#### S1.2. Streaming Pipeline Execution

Extending Figure 4b of the main manuscript, Figure S1 elaborates on the pipelined execution from our multi-prompt stream batch architecture for near real-time generation from multiple regionally assigned text prompts. We have empirically found that the text and image encoders for popular diffusion models take significantly longer latency than the de-

###### Algorithm S1: Baseline [5].

Input: a diffusion model ϵθ , a latent autoencoder (enc, dec) , prompt embeddings y1:p , masks w1:p , timesteps t = t1:n , the output size (H′, W′) , the tile size (H, W) , an inference algorithm STEP , a noise schedule α , the number of bootstrapping steps nbstrap . Output: An image I of designated size (8H′, 8W′) generated from multiple text-mask pairs.

- 1 x′tn ∼ N(0, 1)H′×W′×D // sample the initial latent
- 2 {T1, . . . , Tm} ⊂ {(ht, hb, wl, wr) : 0 ≤ ht < hb ≤ H′, 0 ≤ wl < wr ≤ W′} // get a set of overlapping tiles
- 3 for i ← n to 1 do

- 4 x˜ ← 0 ∈ RH′×W′×D // placeholder for the next step latent
- 5 w˜ ← 0 ∈ RH′×W′ // placeholder for the next step mask weights
- 6 for j ← 1 to m do

- 7 x¯1:p ← repeat(crop(xti, Tj), p) // get a cropped intermediate latent tile
- 8 w¯ 1:p ← crop(w1:p, Tj) // get cropped mask tiles
- 9 if i ≤ nbstrap then

- 10 xbg ← enc(c1) , where c ∼ U(0, 1)3 // get a uniform color background
- 11 xbg ← α(ti)xbg 1 − α(ti)ϵ , where ϵ ∼ N(0, 1)H×W×D // add noise to the background for mixing

- 12 x¯1:p ← w¯ 1:p ⊙ x¯1:p + (1 − w¯ 1:p) ⊙ xbg // bootstrap by treating as multiple single-instance images
- 13 end
- 14 x¯1:p ← STEP(x¯1:p, y1:p, i; ϵθ, α, t) // prompt-wise batched diffusion update
- 15 x˜[Tj] ← x˜[Tj] + pk=1 w¯ k ⊙ x¯k // aggregation by averaging
- 16 w˜ [Tj] ← w˜ [Tj] + pk=1 w¯ k // total weights for normalization
- 17 end
- 18 xti−1 ← x˜ ⊙ w˜ −1 // reverse diffusion step
- 19 end
- 20 I ← dec(xt1) // decode latents to get an image

Algorithm S2: SEMANTICDRAW pipeline of Section 3.2.

Input: a diffusion model ϵθ , a latent autoencoder (enc, dec) , prompt embeddings y1:p , quantized masks w(1:t1:p n) , timesteps t = t1:n , the output size (H′, W′) , a noise schedule α and η , the tile size (H, W) , an inference algorithm STEPEXCEPTNOISE , the number of bootstrapping steps nbstrap .

Output: An image I of designated size (8H′, 8W′) generated from multiple text-mask pairs.

- 1 x′tn ∼ N(0, 1)H′×W′×D
- 2 {T1, . . . , Tm} ⊂ {(ht, hb, wl, wr) : 0 ≤ ht < hb ≤ H′, 0 ≤ wl < wr ≤ W′}
- 3 for i ← n to 1 do

- 4 x˜ ← 0 ∈ RH′×W′×D
- 5 w˜ ← 0 ∈ RH′×W′
- 6 for j ← 1 to m do

- 7 x¯1:p ← repeat(crop(xti, Tj), p)
- 8 w¯ (1:tip) ← crop(w(1:tip), Tj) // use different quantized masks for each timestep
- 9 if i ≤ nbstrap then

- 10 xbg ← enc(1) // get a white color background
- 11 xbg ← α(ti)xbg 1 − α(ti)ϵ , where ϵ ∼ N(0, 1)H×W×D

- 12 x¯1:p ← w¯ 1:p ⊙ x¯1:p + (1 − w¯ 1:p) ⊙ xbg
- 13 u1:p ← get bounding box centers(w¯ 1:p) ∈ Rp×2 // get the bounding box center of each mask

- 14 x¯1:p ← roll by coordinates(x¯1:p, u1:p) // center foregrounds to their mask centers

- 15 end
- 16 x¯1:p ← STEPEXCEPTNOISE(x¯1:p, y1:p, i; ϵθ, α, t) // pre-averaging
- 17 if i ≤ nbstrap then

- 18 x¯1:p ← roll by coordinates(x¯1:p, −u1:p) // restore from centering

- 19 end
- 20 x˜[Tj] ← x˜[Tj] + pk=1 w¯ k ⊙ x¯k
- 21 w˜ [Tj] ← w˜ [Tj] + pk=1 w¯ k
- 22 end
- 23 xti−1 ← x˜ ⊙ w˜ −1
- 24 xti−1 ← xti−1 + ηti−1ϵ , where ϵ ∼ N(0, 1)H×W×D // post-addition of noise
- 25 end
- 26 I ← dec(xt1)

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

(a) Masks (b) 1 step (c) 2 steps (d) 3 steps (e) 4 steps

- Figure S2. The number of centering steps effectively trades off centered-bias against overall harmony. Composition, harmony, and maskobedience are achieved in the sweet spot of 2-3 steps.

Background: “A brick wall”, Red: “A moss”

[Figure 36]

[Figure 37]

(a) Prompt mask. (b) σ = 0 , i.e., no QMask.

[Figure 38]

[Figure 39]

(c) σ = 16 . (d) σ = 32 .

- Figure S3. Effect of the standard deviation in mask smoothing.

#### S1.3. Controlling Fidelity-Harmony Trade-off

As we have mentioned in Section 3.2, accelerated samplers involving 5 or few steps like in our case rely heavily on the first few steps of inference in determining the structure of the image. Many diffusion models are trained using natural images that place their objects of interest at the center of the canvas. This makes these diffusion models generate all of their prompt-guided objects at the center of the canvas. Cropping by masks occasionally leads to destruction of such objects. Mask-centering bootstrapping is devised in order to alleviate this problem. However, applying bootstrapping from beginning to the end causes another problem of disharmony in the overall image with multiple regionbased prompts. This can be seen in Figure S2e, where the girls’ upper part of head is unnaturally cut. This problem also caused by the acceleration. Unlike gradual generation over tens of inference steps, in our accelerated scenario, the later inference steps are responsible for both high quality texture generation and boundary creation. Those quickly generated model-generated boundaries do not align well with the user-given mask inputs, creating unnatural cuts after merging with other prompt-guided subsections of the creation. We, therefore, provide a simple control handle that trades off mask-fidelity against overall harmony: the number of mask-centering steps in the bootstrapping stage. The effect of this control handle can be seen in Figure S2. We have empirically found that 1-3 steps work best, and we have used 2 steps throughout this work.

noising network. Assuming that users change text prompts and background images less frequently than they change the areas occupied by each semantic masks, such latency can be hidden under the high-throughput streaming generation of images. Moreover, mask processing takes almost negligible latency compared to image generation or text encoding. In other words, drawing with semantic masks of preencoded text prompts do not affect the generation speed, allowing users to almost seamlessly interact with the generation pipeline by friendly drawing interface. This user interface of our drawing-based interactive content creation is the same as commercial drawing software with brush tools. The only difference is that our brush tools apply semantic masks instead of colors or patterns. This similarity opens up a novel application for diffusion models, i.e., SEMANTICDRAW.

#### S1.4. Mask Quantization

To increase harmonization within a created image, we have introduced mask quantization as our final piece of the puzzle in Section 3.2 of the main manuscript. Mask quantization allows smooth masks with controllable smoothness that resemble soft brush tools in common drawing software. Therefore, this stage not only increases image fidelity but also enhances user experience in our SEMANTICDRAW application. This section explains additional technical details of mask quantization.

As Figure 5 of the main manuscript shows, the mask

[Figure 40]

Ours,MaskOverlay

Image prompt (row, column): Background (1, 1): “Clear deep blue sky”, Green (1, 2): “Summer mountains”, Red (1, 3): “The Sun”, Pale Blue (2, 1): “The Moon”, Light Orange (2, 2): “A giant waterfall”, Purple (2, 3): “A giant waterfall”, Blue (3, 1): “Clean deep blue lake”, Orange (3, 2): “A large tree”, Light Green (3, 3): “A large tree”

- Figure S4. Mask overlay images of the generation result in Figure 2 of the main manuscript. Generation by our SEMANTICDRAW not only achieves high speed of convergence, but also high mask fidelity in the large-size region-based text-to-image synthesis, compared to the baseline MultiDiffusion [5]. Each cell shows how each mask (including the background one) maps to each generated region of the image, as described in the label below. Note that we have not provided any additional color or structural control other than our semantic palette, which is simply pairs of text prompts and binary masks.

smoothing is an optional preprocessing procedure before generation. Once users provide a set of masks corresponding to a set of text prompts they want to draw, the binary masks are smoothened with a low-pass filter such as Gaussian blurs. In order to perform masking with these continuous masks for discrete denoising steps of the accelerated schedulers [8, 26, 33, 34, 44], we create a set of binary masks from each of the continuous masks by thresholding with the noise levels predefined by the diffusion scheduler. For example, Figure 5 of the main manuscript shows five noise levels actually used in generating the results in the main manuscript and throughout this Supplementary Material. The resulting set of binary masks have monotonically increasing sizes as the corresponding noise levels become lower. Note that we can interpret a noise level of each generating step as a magnitude of uncertainty during the reverse diffusion process. Since the boundary of an object is fuzzier than the center of the object of prescribed masked region, the more uncertain boundary regions can be sampled only during the few latest steps where detailed textures dominant over structural development. Therefore, a natural way of applying these binary masks is in the order of increasing size. By applying each generated binary mask at the timestep with corresponding noise level, we effectively enlarge the size of the mask of a foreground text prompt as we proceed on the generative denoising steps.

The blurring and quantization of the binary masks have a nice interpretation of a rough sketch. In many cases where

users prescribe masks to query for multi-object generation, the exact boundary locations for the best visual construction of an image are not known a priori. In other words, human creation of arts almost always starts with rough sketches. We can increase or decrease the standard deviation of the blur to control the roughness of the sketch, i.e., the certainty of our designation on the boundary. This additional control knob is effective in creating AI-driven arts which inherently exploits high randomness in practice. For reference, Figure S3 shows the effect of increasing the blurriness at the mask proprocessing step. As the standard deviation of the mask blur increases from 0 to 32, the moss, the content of the mask, gradually shrinks and semantically blurred with the brick wall, the background content. As our supplementary code show, this semantic mixing effect of mask blurring and quantization is helpful to harmonize contents in generative editing tasks, i.e., inpainting, where background images are predefined and not fully masked out during generation.

### S2. More Results

In this section, we provide additional visual comparison results between baseline MultiDiffusion [5], a simple application of acceleration modules [33, 34] to the baseline, and our stabilized Algorithm S2. We show that our algorithm is capable of generating large-scale images from multiple regional prompts with a single commercial off-theshelf graphics card, e.g., an RTX 2080 Ti GPU.

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

(a) Masks (b) LRDiff (45s) (c) (b)+LCM (1s) (d) Ours (1s)

- Figure S5. Qualitative comparison between LRDiff+LCM and ours. Background prompt: “Iron Man and Hulk stand amidst the ruins, engaged in a fierce battle with each other.” Left box prompt: “Iron-man” Right box prompt: “Hulk”

#### S2.1. Region-Based Text-to-Image Generation

We show additional region-based text-to-image generation results in Figure S6. In addition to Figure 6 of the main manuscript, the generated samples show that our method is able to accelerate region-based text-to-image generation consistently by ×10 without compromising the generation quality. Moreover, Figure 2 of the main manuscript has shown that the benefits from our acceleration method for arbitrary-sized generation and region-based controls are indeed simultaneously enjoyable. Our acceleration method enables publicly available Stable Diffusion v1.5 [45] to generate a 1920 × 768 scene from eight hand-drawn masks in 59 seconds, which is ×52.5 faster than the baseline [5] taking more than 51 minutes to converge into a low-fidelity image. Figure S4 shows mask fidelity of this generation. We can visualize that even if the generated image has larger dimension than the dimensions the model has been trained for, i.e., 768 × 768, the mask fidelity is achieved under this accelerated generation. Locations and sizes of the Sun and the Moon match to the provided masks in near perfection; whereas mountains and waterfalls are harmonized within the overall image, without violating region boundaries. This shows that the flexibility and the speed of our generation paradigm, SEMANTICDRAW, is also capable of professional usage.

Furthermore, we have found that more recent methods such as LRDiff [41] also suffers the same instability problem when accelerated. Figure S5 shows one example. In this qualitative results, our method not only achieves faster generation speed (×45), but also enjoys better mask fidelity and perceptual quality. This further validates the significance of our strategy in professional interactive content creation.

Regarding that professional image creation process using diffusion models typically involves a multitude of resampling trials with different seeds, the original baseline model’s convergence speed of one image per hour severely limits the applicability of the algorithm. In contrast, our ac-

celeration method enables the same large-size region-based text-to-image synthesis to be done under a minute, making this technology practical to industrial usage.

#### S2.2. Panorama Generation

We can also visually compare arbitrary-sized image creation with panorama image generation task. As briefly mentioned in Section S1, comparing with this task reveals the problem of incompatibility between accelerating schedulers and current region-based multiple text-to-image synthesis pipelines. Figure S7 shows the results of large-scale panorama image generation using our method, where we generate 512 × 4608 images from a single text prompt. Na¨ıvely applying acceleration to existing solution leads to blurry unrealistic generation, enforcing users to resort to more conventional diffusion schedulers that take long time to generate [52]. Instead, our method is compatible to accelerated samplers [8, 26, 33, 34, 44], showing ×13 faster generation of images with sizes much larger than the resolutions of 512 × 512 or 768 × 768, for which the diffusion model [45] is trained. Combining results from Section S2.1 and S2.2 our Algorithm S2 significantly broadens the usability of diffusion models for professional content creators. This leads to the last section of this Supplementary Material, the description of our submitted demo application.

### S3. Sample Application

This last section elaborates on the design and the example usage of our demo application of SEMANTICDRAW, introduced in Section 5 of the main manuscript. Starting from the basic description of user interface in Section S3.1, we discuss the expected usage of the app in Section S3.2. Our discussion mainly focuses on how real-time interactive content creation is achieved from accelerated region-based textto-image generation algorithm we have provided.

#### S3.1. User Interface

As illustrated in Figure S8b, user interactions are classified into two groups, i.e., the slow processes and the fast pro-

Background: “A cinematic photo of a sunset”, Yellow: “An abandoned castle wall”, Red: “A photo of Alps”, Blue: “A daisy field”

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Background: “A photo of outside”, Yellow: “A river”, Red: “A photo of a boy”, Blue: “A purple balloon”

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

Background: “A grassland”, Yellow: “A tree blossom”, Red: “A photo of small polar bear”

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Background: “A photo of mountains with lion on the cliff”, Yellow: “A rocky cliff”, Red: “A dense forest”, Blue: “A walking lion”

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Background: “A photo of the starry sky”, Yellow: “The Earth seen from ISS”, Red: “A photo of a falling asteroid”

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

(a) Prompt (b) MD, 50 steps (c) MD+LCM, 5 steps (d) Ours, 5 steps

- Figure S6. Additional region-based text-to-image synthesis results. Our method accelerates MultiDiffusion [5] up to ×10 while preserving or even boosting mask fidelity. 17

“A photo of Alps”

[Figure 65]

MD()MD+LCM(10s)154sOurs(12s)

[Figure 66]

[Figure 67]

“The battle of Cannae drawn by Hieronymus Bosch”

[Figure 68]

301sOurs(21s)MD+LCM(17s)MD()

[Figure 69]

[Figure 70]

“A photo of a medieval castle in the distance over rocky mountains in winter”

[Figure 71]

MD()MD+LCM(19s)296sOurs(23s)

[Figure 72]

[Figure 73]

“A photo under the deep sea with many sea animals”

[Figure 74]

MD()MD+LCM(18s)290sOurs(23s)

[Figure 75]

[Figure 76]

- Figure S7. Additional panorama generation results. The images of size 512 × 4608 are sampled with 50 steps for MD and 4 steps for MD+LCM and Ours. Our SEMANTICDRAW can synthesize high-resolution images in seconds. We achieve ×13 improvement in inference latency.

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

(a) Screenshot of the application. (b) Application design schematics.

- Figure S8. Sample application demonstrating semantic palette enabled by our SEMANTICDRAW algorithm. After registering prompts and optional background image, the users can create images in real-time by drawing with text prompts.

cesses, based on the latency of response from the model. Due to the high overhead of text encoder and image encoder, the processes involving these modules are classified as slow processes. However, operations such as preprocessing or saving of mask tensors and sampling of the U-Net for a single step take less than a second. These processes are, therefore, classified as fast ones. SEMANTICDRAW, our suggested paradigm of image generation, comes from the observation that, if a user first registers text prompts, image generation from user-drawn regions can be done in real-time.

The user interface of Figure S9 is designed based on the philosophy to maximize user interactions of the fast type and to hide the latency of the slow type. Figure S9 and Table S1 summarize the components of our user interface. The interface is divided into four compartments: the (a) semantic palette, which is a palette of registered text prompts (no. 1-2), the (b) drawing screen (no. 3-5), the (c) streaming display and controls (no. 6-7), and the (d) control panel for the additional controls (no 8-13). The (a) semantic palette manages the number of semantic brushes to be used in the generation, which will be further explained below. Users are expected to interact with the application mainly through (b) drawing screen, where users can upload backgrounds and draw on the screen with selected semantic brush. Then, by turning (c) streaming interface on, the users can receive generated images based on the drawn regional text prompts in real-time. The attributes of semantic brushes are modified through (d) control panel.

Types of the transaction data between application and user are in twofold: a (1) background and a (2) list of text prompt-mask pairs, named semantic brushes. The user can register these two types of data to control the generation stream. Each semantic brush consists of two part: (1) text prompt, which can be edited in the (d) control panel after clicking on the brush in (a) semantic palette, a set of avail-

able text prompts to draw with, and (2) mask, which can be edited by selecting the corresponding color brush at drawing tools (no. 5), and drawing on the drawing pad (no. 3) with a brush of any color. Note that in the released version of our code, the color of semantic brush does not affect generation results. Its color only separates a semantic brush from another for the users to discern.

As the interface of the (d) control panel implies, our reformulation of MultiDiffusion [5] provides additional hyperparameters that can be utilized for professional creators to control their creation processes. The mask alpha (no.

- 11) and the mask blur std (no. 12) determine preprocessing attributes of selected semantic brush. Before the mask is quantized into predefined noise levels of scheduler, as elaborated in Section S1.4, mask is first multiplied by mask alpha and goes through an isotropic Gaussian blur with a specified standard deviation. That is, given a mask w , a mask alpha a, and the noise level scheduling function β(t) = 1 − α(t), the resulting quantized mask w(1:tip) is:

w(1:tip) = [aw > β(ti)], (S7) where [·] is an indicator function taking the inequality as a binary operator to make a boolean mask tensor w(1:tip) at time ti . The default noise levels β(t) of the acceleration modules [8, 26, 33, 34, 44] are close to one, as shown in Figure 5 of the main manuscript. This makes mask alpha extremely sensitive. By changing its value only slightly, e.g., down to 0.98, the corresponding prompt already skips first two sampling steps. This quickly degenerates the content of the prompt, and therefore, the mask alpha (no. 11) should be used in care. The effect of mask blur std (no.

- 12) is shown in Figure S3, and will not be further elaborated in this section. The seed of the system can be tuned by seed control (no. 13). Nonetheless, controlling pseudorandom generator will rarely be needed since the applica-

[Figure 83]

Figure S9. Screenshot of our supplementary demo application. Details of the numbered components are elaborated in Table S1.

Table S1. Description of each numbered component in the SEMANTICDRAW demo application of Figure S9.

No. Component Name Description

- 1 Semantic palette Create and manage text prompt-mask pairs.
- 2 Import/export semantic palette Easy management of text prompt sets to draw.
- 3 Main drawing pad User draws at each semantic layers with brush tool.
- 4 Background image upload User uploads background image to start drawing.
- 5 Drawing tools Using brush and erasers to interactively edit the prompt masks.
- 6 Display Generated images are streamed through this component.
- 7 History Generated images are logged for later reuse.
- 8 Prompt edit User can interactively change the positive/negative prompts at need.
- 9 Prompt strength control Prompt embedding mix ratio between the current & the background. Helps content blending.
- 10 Brush name edit Adds convenience by changing the name of the brush. Does not affect the generation.
- 11 Mask alpha control Changes the mask alpha value before quantization. Recommended: > 0.95.
- 12 Mask blur std. dev. control Changes the standard deviation of the quantized mask of the current semantic brush.
- 13 Seed control Changes the seed of the application.

[Figure 84]

[Figure 85]

(a) Upload a background image. (b) Register semantic palette.

[Figure 86]

[Figure 87]

(c) Draw with semantic brushes. (d) Play the stream and interact.

Figure S10. Illustrated usage guide of our demo application of SEMANTICDRAW.

tion generates images in an infinite stream. The prompt edit (no. 8) is the main control of semantic brush. The users can change text prompt even when generation is on stream. It takes exactly the total number of inference steps, i.e., 5 steps, for a change in prompts to take effect. Further, we provide prompt strength (no. 9) as an alternative to highly sensitive mask alpha (no. 11) to control the saliency of the target prompt. Although modifying the alpha channel provides good intuition for graphics designer being already familiar to alpha blending, the noise levels of consistency model [8, 26, 33, 34, 44, 53] make the mask alpha value not aligned well with our intuition in alpha blending. Prompt strength is a mix ratio between the embeddings of the foreground text prompt of given semantic brush and background text prompt. We empirically find that changing the prompt strengths gives smoother control to the

foreground-background blending strength than mask alpha. However, whereas the mask alpha can be applied locally, the prompt strength only globally takes effect. Therefore, we believe that the two controls are complementary to one another.

Finally, we provide seed-fixing option that enables incremental generation for drawing-like experience. The difference between simple streaming generation and streaming generation with our seed-fixing option is elaborated in Figure S11. By not only caching the prompt embeddings during streaming but also sharing noise tensors within a stream of generation, we can simply switch into incremental editing in our application. Therefore, with the seed-fixing option, we can maintain strong consistency across entire stream of generation, which we may call a session of content creation. This enables content creators to switch from

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

Figure S11. Sequential generation of frames from real-time drawing of masks. Top row: Original without seed-fixing. Bottom row: Increased determinism with seed-fixing option. A row of images comes sequentially from a single stream of generation given the same sequence of interactive controls (from left to right).

random ideation to detailed editing and vice versa, greatly increasing the practicality of the application. Both options are available in our official code.

#### S3.2. Basic Usage

We provide the simplest procedure of creating images from SEMANTICDRAW pipeline. Screenshots in Figure S10 illustrate the four-step process.

1. Start the Application. After installing the required packages, the user can open the application with the following command prompt:

1 python app.py --model "KBlueLeaf/kohaku-v2.1" --height 512

--width 512

The application front-end is web-based and can be opened with any web browser through localhost:8000. We currently support various baseline architecture including Stable Diffusion 1.5 [45], Stable Diffusion XL [40], and Stable Diffusion 3 [49] checkpoints for --model option. For SD1.5, we support latent consistency model (LCM) [33, 34] and Hyper-SD [44], for SDXL, we support SDXL-Lightning [26], and for SD3, we support Flash Diffusion [8] for acceleration of the generation process. The height and the width of canvas should be predefined at the startup of the application.

as elaborated in Section S3.1. The interpolation takes place when a foreground text prompt embedding is assigned with a prompt strength less than one. User is always able to change the background prompt like other prompts in the semantic palette.

- 3. Type in Text Prompts. See Figure S10b. The next step is to create and manage semantic brushes by interacting with the semantic palette (no. 1). A minimal required modification is text prompt assignment through the prompt edit (no. 8) panel. The user can additionally modify other options in the control panel marked as yellow in Figure S10b.
- 4. Draw. See Figure S10c. A user may start drawing by selecting a brush in drawing tools (no. 5) toolbar that matches the user-specified text prompt in the previous step. Grab a brush, draw, and submit the drawn masks. After initiating the content creation, the images are streamed through the display (no. 6) in real-time from dynamically changing user inputs. The past generations are saved in history (no. 7).

2. Upload Background Image. See Figure S10a. The first interaction with the application is to upload any image as background by clicking the background image upload (no. 4) panel. The uploaded background image will be resized to match the canvas. After uploading the image, the background prompt of the uploaded image is automatically generated for the user by pre-trained BLIP-2 model [23]. The background prompt is used to blend between foreground and background in prompt-level globally,

