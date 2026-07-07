## 4DSloMo: 4D Reconstruction for High Speed Scene with Asynchronous Capture

YUTIAN CHEN, Shanghai AI Laboratory, China and The Chinese University of Hong Kong, China SHI GUO, Shanghai AI Laboratory, China TIANSHUO YANG, The University of Hong Kong, China LIHE DING, The Chinese University of Hong Kong, China XIUYUAN YU, The Chinese University of Hong Kong, China JINWEI GU, NVIDIA, USA TIANFAN XUE, The Chinese University of Hong Kong, China and Shanghai AI Laboratory, China

[Figure 1]

# arXiv:2507.05163v2[cs.CV]16Jun2026

[Figure 2]

[Figure 3]

[Figure 4]

#### (a) Synchronous Capture

#### (c) Results

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

+Artifact-fix Video Model

+Async. Capture

[Figure 9]

- Cam1
- Cam2

GS4D Ours

25fps

0.04s

[Figure 10]

[Figure 11]

time

[Figure 12]

[Figure 13]

#### (b) Asynchronous Capture

[Figure 14]

[Figure 15]

[Figure 16]

Cam1 Cam2

- Cam1
- Cam2

0.02s

time

[Figure 17]

[Figure 18]

Fig. 1. Our 4D Reconstruction Results of the real-captured scene. We propose an asynchronous capture scheme, which increases the effective capture frame rate by staggering the start times of cameras without any additional cost. We further leverage video diffusion priors to enhance the reconstruction results. The results show that our method can reconstruct high speed and complex motion with high quality.

Reconstructing fast-dynamic scenes from multi-view videos is crucial for high-speed motion analysis and realistic 4D reconstruction. However, the majority of 4D capture systems are limited to frame rates below 30 FPS (frames per second), and a direct 4D reconstruction of high-speed motion from low FPS input may lead to undesirable results. In this work, we propose a high-speed 4D capturing system only using low FPS cameras, through novel capturing and processing modules. On the capturing side, we propose an asynchronous capture scheme that increases the effective frame rate by

staggering the start times of cameras. By grouping cameras and leveraging a base frame rate of 25 FPS, our method achieves an equivalent frame rate of 100–200 FPS without requiring specialized high-speed cameras. On processing side, we also propose a novel generative model to fix artifacts caused by 4D sparse-view reconstruction, as asynchrony reduces the number of viewpoints at each timestamp. Specifically, we propose to train a video-diffusion-based artifact-fix model for sparse 4D reconstruction, which refines missing details, maintains temporal consistency, and improves overall reconstruction quality. Experimental results demonstrate that our method significantly enhances high-speed 4D reconstruction compared to synchronous capture. Project page: https://openimaginglab.github.io/4DSloMo/

Authors’ Contact Information: Yutian Chen, Shanghai AI Laboratory, Shanghai, China and The Chinese University of Hong Kong, Hong Kong, China, chenyt0205@gmail.com; Shi Guo, Shanghai AI Laboratory, Shanghai, China, guoshi@pjlab.org.cn; Tianshuo Yang, The University of Hong Kong, Hong Kong, China, yangtianshuo@connect.hku.hk; Lihe Ding, The Chinese University of Hong Kong, Hong Kong, China, dean.dinglihe@ outlook.com; Xiuyuan Yu, The Chinese University of Hong Kong, Hong Kong, China, 1155211255@link.cuhk.edu.hk; Jinwei Gu, NVIDIA, San Jose, USA, gujinwei@gmail. com; Tianfan Xue, The Chinese University of Hong Kong, Hong Kong, China and Shanghai AI Laboratory, Shanghai, China, tfxue@ie.cuhk.edu.hk.

CCS Concepts: • Computing methodologies → 3D imaging. Additional Key Words and Phrases: 4D Reconstruction, High Speed Scene, Asynchronous Capture, Artifact-fix Video Diffusion Mode

1 Introduction

Fast-dynamic scenes reconstruction from multi-view videos is a fundamental challenge in 3D vision with broad applications. In

This work is licensed under a Creative Commons Attribution 4.0 International License.

sports and biomechanics, high-speed 4D reconstruction enables precise motion capture for athlete performance evaluation and injury prevention. In autonomous driving and robotics, accurately reconstructing 3D models of fast-moving objects, such as pedestrians and vehicles, is essential for perception and decision-making. Additionally, in VR/AR content production, reconstructing high-fidelity human performances, such as dance and martial arts, is crucial for creating realistic digital avatars.

Still, 4D reconstruction of fast-moving objects remains challenging, as the majority of 4D-capturing camera arrays operate at no more than 30 FPS (frames per second). For instance, DNA-Rendering [Cheng et al. 2023b] operates at 15 FPS, while ENeRF-Outdoor [Lin et al. 2022] and Neural3DV [Li et al. 2022] achieve 30 FPS. In contrast, many activities, such as cloth movement shown in Fig. 1, require a higher frame rate; for example, standard high-speed 2D photography typically operates at 120 FPS or higher to capture

- them. Extending frame rate of existing camera arrays is hard, as that requires more expensive and dedicated hardware, and greatly increases data transmission bandwidth requirements.

Another way to capture high-speed 4D motion without hardware changes is to increase the frame rate at the reconstruction stage. Recently, gaussian-splatting-based 4D reconstruction methods [Li et al. 2022; Lin et al. 2024; Wang et al. 2024, 2025; Wu et al. 2024b; Xu et al. 2024a] have significantly improved novel view synthesis for dynamic scenes. For simple motion, it can reconstruct continuous frames from sparse temporal inputs, effectively increasing the frame rate. However, it still fails to handle complex nonlinear motion, such as cloth movement, resulting in obvious artifacts, as shown in the top right of Fig. 1. Thus, this raises an interesting question: is it possible to recover intermediate frames of high-speed motion using regular video cameras at 30 FPS?

To achieve that, we propose a novel asynchronous capture scheme to increase frame rate. We deliberately add delay to the starting time between different cameras, so different cameras can capture different timestamps, effectively boosting the frame rate. An example is shown in Fig. 1. Two synchronized cameras can only achieve 25 FPS (top row) capturing. However, in our setup, by differing the capture time of Cam2, the capture interval reduces to 0.02s, achieving 50FPS capturing. In practice, we use eight 25FPS cameras, and divide them into 4 or 8 groups to effectively increase the perceived temporal resolution to 100 or 200 FPS, respectively. By capturing temporally denser frames, we can more accurately model the intermediate motion information, particularly for complex motion.

A key challenge introduced by such asynchronous capturing is the limited number of available views at each timestamp, which increases viewpoint sparsity and results in visible artifacts in the reconstruction, as illustrated in Fig. 1. Recent 3D sparse view reconstruction methods [Gao et al. 2024; Wu et al. 2024a; Wynn and Turmukhambetov 2023; Yu et al. 2024] utilized the pre-trained image diffusion model to prove the reconstruction results. However, we experimentally find that utilizing image diffusion model to perform artifacts-fix cause temporal inconsistency for sparse 4D reconstruction problem, due to frame-wise independent refine. Thus we propose to train the video-diffusion based artifact-fix model for 4D reconstruction. To train this model, we construct a dataset by temporally sub-sampling 4D sequences to simulate large motion and our

proposed asynchronous capturing pattern. The sparsely sampled sequences are used to train a 4D Gaussian Splatting model, whose rendered outputs naturally exhibit reconstruction artifacts. These degraded outputs are then paired with the original ground-truth sequences to form training data. Despite the limited size of available 4D datasets, our artifact-fix video diffusion model demonstrates strong generalization across multiple scenes, benefiting from the powerful spatiotemporal priors inherent in video diffusion model. Notably, with only 750 training pairs, our method effectively removes reconstruction artifacts and significantly improves visual quality, as shown in Fig. 1.

We compare our method with several state-of-the-art approaches [Sara Fridovich-Keil and Giacomo Meanti et al. 2023; Wu et al. 2024c; Yang et al. 2024c] using the DNA-Rendering dataset [Cheng et al. 2023a] and the Neural4DV dataset [Li et al. 2022]. To simulate fast motion and asynchronous capture, we apply temporal subsampling. To further validate our approach on real-world asynchronous multiview video data, we capture 12 sequences of fast and complex dynamic scenes using an asynchronous capture setup. Experimental results demonstrate that our method improves high-speed 4D reconstruction compared to synchronized capture.

Our key contributions are summarized as follows:

- • Hardware solution: We design and implement a low-cost asynchronous capture system that increases the effective frame rate by staggering the start times of commodity cameras.
- • Software solution: We train an artifact-fix video diffusion model to refine the 4D reconstruction results, significantly improving the visual quality of the rendered images.
- • Dataset: Wedevelop thefirstdatasetcontaining 12 sequences of asynchronous multi-view videos to validate our approach, demonstrating the practical feasibility and effectiveness of our method.

2 Related Work 2.1 Dynamic 3D Reconstruction

Reconstruction and novel view synthesis of dynamic 3D scenes remain challenging. Many existing methods require multiple synchronized videos captured from different viewpoints as input [Lin et al. 2022, 2024; Sara Fridovich-Keil and Giacomo Meanti et al. 2023; Wu et al. 2024c; Xu et al. 2024b; Yang et al. 2024c]. These works tend to use radiance field models such as NeRF [Mildenhall et al. 2021] or

- 3DGS [Kerbl et al. 2023] as the underlying static 3D representations, and model motion patterns using various types of the deformation field, for example, MLPs [Lu et al. 2024; Pumarola et al. 2021; Yang et al. 2024a], spatial-temporal planes [Sara Fridovich-Keil and Giacomo Meanti et al. 2023; Wu et al. 2024c], polynomial functions [Li et al. 2024] and Fourier series [Katsumata et al. 2024]. These methods learn motion patterns from independent time input in a vanilla way and not adept at frames interpolation in the temporal dimension.

Existing camera arrays for 4D scene capture are often limited to low frame rates (e.g., 15-30 FPS [cheng2023dna, lin2022efficient, li2022neural]), posing challenges for reconstructing high-speed dynamics. While specialized systems like multi-view photometric

- 4DSloMo: 4D Reconstruction for High Speed Scene with Asynchronous Capture • 3

Asynchronous Capture 4D Gaussian Splatting

Artifact-fix Model

[Figure 19]

[Figure 20]

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

Render

- Cam1
- Cam2
- Cam3

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

Noisy Video

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

| |
|---|

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

Video Repair Model

Cam3

Cam1

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Cam2

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

Supervised

Time

Enhanced Video

[Figure 60]

[Figure 61]

Captured Uncaptured

Fig. 2. The overall pipeline of our model. Given several asynchronous multi-view videos, we first initialize a 4D Gaussian model for a specific iteration. We

- then employ an artifact-fix video diffusion model to refine the input videos. The refined videos are subsequently used to update the 4D Gaussian model.

stereo [Vlasic et al. 2009] offer higher quality, they typically require complex lighting and are costly.

estimated from pre-trained models to provide additional constraints during optimization. Monosdf [Yu et al. 2022] integrates both depth and normal maps as supplementary supervision signals. This combination helps refine the surface details of the reconstructed scene, enhancing its geometric accuracy. Därf [Song et al. 2023] further explored the use of depth maps as an auxiliary signal in their method. Collectively, these works demonstrate the potential of combining various forms of supervision and regularization to enhance sparse view reconstruction.

To improve temporal resolution without specialized hardware, one direction is to manipulate the camera exposure process. Wu et al. [Wu et al. 2012] pioneered staggering the exposure of multiple cameras to increase the effective frame rate, which then requires model-based deblurring in post-processing. A similar principle was applied to a single camera via coded exposure to embed high-speed information into one frame [liu2013efficient]. Other works like TimeFormer [jiang2024timeformer] enhance temporal modeling purely through algorithms, but they lack the new physical information needed to reconstruct truly rapid motion.

Our work adopts asynchronous capture but replaces model-based deblurring with a video-diffusion model. This generative approach corrects sparse-view artifacts to refine detail and ensure temporal consistency, removing the need for a pre-defined geometric proxy.

2.3 3D/4D Reconstruction with Diffusion Model

The recent progress in diffusion models has driven significant advancements in 3D and 4D applications. Pioneer work ReconFusion [Wu et al. 2024a] trains a novel view synthesis diffusion model conditioning on sparse input views and then adopts a SDS (Score Distillation Sampling) [Poole et al. 2022] style optimization strategy for novel view supervision. Unlike the SDS loss, ReconFusion directly predicts the pseudo ground truth of novel views by sampling from the trained diffusion model at each optimization step, which is then used to compute the reconstruction loss. Follow-up works [Gao et al. 2024; Wynn and Turmukhambetov 2023; Yu et al. 2024] also integrate diffusion models with NeRF or 3DGS. To further improve the optimization efficiency, Deceptive-NeRF [Liu et al. 2023] and 3D-GS Enhancer first render pseudo images from the sparse-view-reconstructed 3D representation and use a diffusion model to enhance these pseudo views to obtain high-quality novel view supervision without querying the diffusion model at every optimization step. Inspired by these few-shot 3D reconstruction methods, we utilize a diffusion model to remove the artifacts from the 4D model rendered images.

- 2.2 Sparse View Scene Reconstruction

As the asynchronous capture method reduces the number of available views at each timestamp, our work is closely related to the field of sparse view scene reconstruction. Sparse view reconstruction is a challenging problem due to the limited availability of input views, which can lead to incomplete or ambiguous scene representations. To address this challenge, several recent works have proposed innovative solutions. For example, Freenerf [Yang et al. 2023] introduced a method that incorporates depth regularization to improve the reconstruction quality from sparse views. Similarly, Nerf in 3d vision [Gao et al. 2022] and Regnerf [Niemeyer et al. 2022] explored frequency regularization techniques to enhance the robustness and accuracy of scene reconstruction when only a few input views are available.

In addition to regularization-based methods, several recent works have leveraged extra supervision signals to guide the reconstruction process. For instance, SPARF [Truong et al. 2023] utilized optical flow

[Figure 62]

3 Method

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Trainable Frozen

[Figure 71]

Video Latent

To achieve high-quality 4D reconstruction for scenes with large and complex motion, we first revisit the data acquisition process and propose a novel asynchronous capture scheme (Sec. 3.2) that surpasses the conventional 30 FPS (frame-per-second) limitation. Although the captured data can be leveraged by state-of-the-art method, i.e., GS4D [Yang et al. 2024c] (revisited in Sec. 3.1), to achieve temporally dense 4D reconstruction, the asynchronous capture scheme inherently introduces challenges of sparse viewpoints and temporal inconsistency. To address the artifacts caused by sparse views, a artifact-fix video diffusion model is introduced in Sec. 3.3, followed by the 4D reconstruction process with diffusion priors in Sec. 3.4. The overall framework of our approach is illustrated in Fig. 2.

###### Diffusion Transformer

…

WanEncoder

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

WanDecoder

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

###### DiTBlock

[Figure 88]

DiTBlock

###### DiTBlock

[Figure 89]

[Figure 90]

Conditional Video Latent

…

…

…

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

LoRA

Conditional Image Feature

LoRA

LoRA

CLIP

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

Fig. 3. Illustration of artifact-fix video diffusion model setup. Our model freezes all parameters in the network, except for the LoRA weights, to fine-tune a video diffusion model. Precisely, we integrate Lora parameters into the DiT model. With a LoRA rank designated as 16, this integration takes place in each transformer block.

- 3.1 Preliminary: 4D Gaussian Splatting

We choose 4D Gaussian Splatting (GS4D) [Yang et al. 2024c] as the 4D representation for reconstruction. In GS4D, dynamic 3D scenes are represented by introducing an additional time dimension 𝑡 into anisotropic 3D Gaussians [Kerbl et al. 2023]. Formally, a 4D Gaussian is defined by its mean vector 𝜇 ∈ R4 and covariance matrix Σ ∈ R4×4:

starting at different times for the capture: the 𝑖-th camera starts capturing an image at the timestamp 𝑡𝑖 = 𝑖 · (𝜏/𝐾) + 𝑗 · 𝜏, where 𝑖 ∈ {1, 2, . . .,𝐾} denotes the index of the camera within the group, and 𝑗 represents the frame index in the video sequence recorded at a frame rate of 25 FPS. This design temporally staggers the capture timing of different cameras within the 1/25 second exposure intervals, effectively increases the frame rate of the camera system by a factor of 𝐾 without introducing additional hardware costs. In the experiment, we choose 𝐾 = 4, which effectively pushes the frame rate to 100 FPS.

- 1

- 2 (x − 𝜇)𝑇 Σ−1(x − 𝜇) , (1)

𝑝(x|𝜇, Σ) = exp −

where x = (𝑥,𝑦,𝑧,𝑡), with (𝑥,𝑦,𝑧) represents the spatial coordinates, and 𝑡 denotes the temporal coordinate. The covariance matrix Σ is factorized into a scaling matrix 𝑆 and a rotation matrix 𝑅, similar

- to 3D Gaussian splatting: Σ = 𝑅𝑆𝑆𝑇𝑅𝑇, where 𝑆 = diag(𝑠𝑥,𝑠𝑦,𝑠𝑧,𝑠𝑡) defines the anisotropic scaling of the Gaussian, and𝑅 is a 4D rotation matrix parameterized by two quaternions 𝑞𝑙,𝑞𝑟, ensuring a valid rotation in the 4D space: 𝑅 = 𝐿(𝑞𝑙)𝑅(𝑞𝑟).

3.3 Artifact-fix Video Diffusion Model

Although the asynchronous capture introduced in the previous section increase the effective frame rate by 𝐾 times to better capture highspeed movement, it also introduce additional artifacts. Particularly, at each time stamp, asynchronous capture reduces the number of viewpoints by 𝐾 times, resulting in “floater" artifacts in the 4D reconstruction as shown in Fig. 1 (c) due to the sparse view reconstruction challenges. Although there are some recent sparse view

The rendering process follows the standard differentiable rasterization used in Gaussian splatting. Given a pixel (𝑢,𝑣) at time 𝑡, the final rendered image 𝑥𝑟 is computed by blending the visible 3D Gaussians:

∑︁𝑁

𝑖−1

(1 − 𝑝𝑗 (𝑢,𝑣,𝑡)𝛼𝑗), (2)

𝑥𝑟 (𝑢,𝑣,𝑡) =

𝑝𝑖(𝑢,𝑣,𝑡)𝛼𝑖𝑐𝑖(𝑑,𝑡)

𝑖=1

𝑗=1

- 3D reconstruction methods [Gao et al. 2024; Wu et al. 2025, 2024a; Wynn and Turmukhambetov 2023; Yu et al. 2024] can leverage image diffusion priors to enhance reconstruction quality, directly applying them to 4D scenes still causes temporal inconsistency due to frame-wise independent processing.

Therefore, we propose a novel artifact-fix video diffusion model to solve this challenge in 4D sparse reconstruction. Particularly, we take a pretrained video diffusion model and fine-tune on 4D reconstruction data to tailor for our tasks. The model takes the rendered video 𝑉𝑟𝑒𝑛𝑑𝑒𝑟 ∈ R𝐶×𝑇×𝐻×𝑊 containing floater artifacts as input and generates temporally coherent, sharp, and outputs a clean video 𝑉ˆ ∈ R𝐶×𝑇×𝐻×𝑊 . This clean video will later be used to refined

- 4D Gaussian model, which we will discuss in Sec. 3.4. Below we introduce details of our model.

where 𝑝𝑖(𝑢,𝑣,𝑡) is the probability density of the 𝑖-th Gaussian at pixel (𝑢,𝑣,𝑡),𝛼𝑖 representsitsopacity,and𝑐𝑖(𝑑,𝑡) is thetime-dependent color modeled by 4D spherical harmonics (4DSH) [Yang et al. 2024c].

3.2 Asynchronous Capture Scheme

To capture 4D scenes, given a set of cameras at different viewpoints {𝑃𝑖}𝑖𝑁=1, the previous methods [Cheng et al. 2023b; Li et al. 2022] would synchronize the capture timing of 𝑁 cameras to capture images simultaneously, as shown in Fig. 1 (a). However, for highspeed motion, none of this 𝑁 cameras capture the information between two neighboring consecutive frames. The normal frame rate of cameras is less than or equal to 30FPS and 4D reconstruction may fail for large and complex motion that requires high frame rate [Cheng et al. 2023a; Li et al. 2022; Lin et al. 2022].

Data curation. First, to train the model, we constructed a dataset consisting of pairs of videos with artifacts and corresponding clean version as ground truth. Specifically, we asynchronously sub-sample the multi-view video along the temporal dimension, as shown in Fig. 2(a), to train a 4D Gaussian Splatting (GS4D) model. The trained

To capture a temporally denser information, we propose a novel asynchronous capture scheme that enables cameras to start capturing at different time instants, as shown in Fig. 1 (b). Specifically, we put 𝐾 cameras into a group (for example, 𝐾 = 2 in Fig. 1 (b)),

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

4DSloMo: 4D Reconstruction for High Speed Scene with Asynchronous Capture • 5

Ground Truth 4DGS GS4D Ours

K-Planes

| |
|---|

| |
|---|

| |
|---|

| |
|---|

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

| |
|---|

| |
|---|

[Figure 121]

| |
|---|

| |
|---|

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

Fig. 4. Qualitative result on the Neural3DV dataset [Li et al. 2022].

Finally, the refined video 𝑉ˆ is used to supervise the rendering process through the following loss function:

GS4D is then used to render videos at the original frame rate, producing noisy outputs containing reconstruction artifacts. These rendered videos are paired with the corresponding clean videos to form the training data for diffusion model fine-tuning.

𝐿𝑑𝑖𝑓 𝑓 = ||𝑉𝑟𝑒𝑛𝑑𝑒𝑟 −𝑉ˆ ||1 + 𝐿𝑝(𝑉𝑟𝑒𝑛𝑑𝑒𝑟,𝑉ˆ) (4) where 𝐿𝑝 is the perceptual distance LPIPS [Zhang et al. 2018].

Fine-tuning. We build our artifact-fix model on top of the video diffusion model Wan2.1 [Wan et al. 2025] as shown in Fig. 3. To be specific, we use Wan-VAE E to compress rendered video𝑉𝑟𝑒𝑛𝑑𝑒𝑟 and target video𝑉𝑡𝑎𝑟𝑔𝑒𝑡 into the latent space 𝑧𝑟𝑒𝑛𝑑𝑒𝑟,𝑧𝑡𝑎𝑟𝑔𝑒𝑡 ∈ R𝑐×𝑡×ℎ×𝑤. The noise latent𝑧𝑡𝑡𝑎𝑟𝑔𝑒𝑡 and condition latent𝑧𝑟𝑒𝑛𝑑𝑒𝑟 are concatenated along the channel axis and then passed through the DiT model of Wan2.1. To repurpose the original image-to-video generation setting in Wan2.1 for video enhancement, we use all frames of 𝑧render as conditioning to guide artifact repair across the entire sequence, as described in [Wan et al. 2025].

The effectiveness of this approach is illustrated in Fig. 1(c). The proposed asynchronous capture can recover fast-dynamic motion compared to conventional synchronous capture. However, due to the inherently sparse views at each time step, the learned Gaussian representation introduces noticeable artifacts. By applying our per-scene artifact-fix diffusion model to refine the Gaussian representation, these artifacts can be effectively suppressed, leading to improved final results.

4 Experiments

During fine-tuning, we freeze the encoder and decoder, and apply LoRA-based fine-tuning only to the DiT component. The training loss is defined as:

- 4.1 Implementation Details

Our framework is illustrated in Fig. 2. We adopt 4D Gaussian Splatting (GS4D) [Yang et al. 2024c] as the underlying 4D representation. To enhance visual quality, we build an artifact-correction video diffusion model based on Wan-I2V-14B [Wan et al. 2025], fine-tuning its DiT backbone with injected LoRA layers. The model is trained on 750 noisy-clean video pairs from the DNA-Rendering dataset [Cheng et al. 2023a], using a learning rate of 10−4 and a LoRA rank of 16.

Next, we train the 4D Gaussian representation. The 4DGS model undergoes an initial optimization phase for 7k iterations. However, the output of this stage still contains noticeable artifacts. To mitigate these artifacts, we apply the trained artifact-fix diffusion model. In the subsequent 7k iterations, the optimization incorporates only with the refined videos produced by the diffusion model.

- 4.2 Datasets

L𝑡𝑢𝑛𝑒 = E𝑧𝑡𝑎𝑟𝑔𝑒𝑡,𝑡,𝜖,𝑧𝑟𝑒𝑛𝑑𝑒𝑟 [||(𝜖𝜃 (𝑧𝑡𝑡𝑎𝑟𝑔𝑒𝑡,𝑡,𝑧𝑟𝑒𝑛𝑑𝑒𝑟,𝑐𝑡𝑒𝑥) − 𝜖)||22] (3) where 𝑐𝑡𝑒𝑥 denotes an object-specific language prompt.

- 3.4 4D Reconstruction with Diffusion Prior

With the artifact-fix video diffusion model M, we can improve the quality of 4D Gaussian model. As shown in Fig. 2, we first reconstruct an initial 4D Gaussian models G from all the captured images. Then, for each training view, we render a high-frame-rate video 𝑉𝑟𝑒𝑛𝑑𝑒𝑟, covering all timestamps observed by any of the cameras. Because the initial 4D Gaussian is reconstructed from sparse views, these videos 𝑉𝑟𝑒𝑛𝑑𝑒𝑟 contain floater artifacts, but can provide the diffusion model M with essential spatial viewpoint information and temporal object motion information.

We evaluateourmethodonmultiple widely used multi-view datasets, including DNA-Rendering [Cheng et al. 2023b] and Neural3DV [Li et al. 2022]. DNA-Rendering captures 10-second clips of dynamic human subjects and objects at 15 FPS using a combination of 4K and 2K cameras. To induce large motions, we temporally subsample the dataset to one-fourth of its original frame rate. The test set

To remove those artifact, we then obtain the latent representation 𝑧𝑟𝑒𝑛𝑑𝑒𝑟 ∈ R𝑐×𝑡×ℎ×𝑤 by applying Wan-VAE to 𝑉𝑟𝑒𝑛𝑑𝑒𝑟, and concatenate it with pure noise of the same shape along the channel dimension. This combined representation is then used to generate the refined video 𝑉ˆ, which is clearer and sharper.

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

6 • Yutian Chen, Shi Guo, Tianshuo Yang, Lihe Ding, Xiuyuan Yu, Jinwei Gu, and Tianfan Xue

Ground Truth K-Planes 4DGS GS4D Ours

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

Fig. 5. Qualitative result on the DNA-Rendering dataset [Cheng et al. 2023a].

the same scene. This setup enables direct comparison between the two acquisition strategies under large-motion conditions.

[Figure 136]

It is worth noting that the artifact-fix video diffusion model is trained on the DNA-Rendering dataset. We construct a total of 750 noisy-clean video pairs for training, each with a resolution of 1024 × 1024 and a length of 25 frames.

3m

22.5°

Camera

2m

Capturing setup: Since this is the first work to employ an asynchronous capture strategy for 4D Gaussian Splatting, no existing real-world 4D dataset has been recorded using such a setup. To evaluate our method on real asynchronously captured scenes, we build a custom multi-view camera array for 4D data acquisition. As illustrated in Fig. 6, the setup consists of 12 cameras operating at 25 FPS, all capable of hardware-synchronized triggering. The cameras are arranged at three different vertical levels, with four cameras evenly spaced at each level, approximately 22.5 degrees apart. Although all 12 cameras support synchronous triggering, we manually introduce varying trigger delays across cameras to enable asynchronous capture. By dividing the camera array into four or eight temporal groups, we effectively achieve a capture rate of 100 FPS or 200 FPS, respectively, as discussed in Sec. 3.2.

1m

(a) Bird-view of the capture system (b) The arrangement of cameras

Fig. 6. Illustration of the capture system.

consists of all frames from a held-out view. Neural3DV captures multi-view videos at 30 FPS, each lasting ten seconds. We subsample the videos to one-twelfth of their original frame rate. For each scene, one view is held out for testing, while the remaining views are used for training. By applying temporal downsampling to existing 4D datasets, we effectively simulate large inter-frame motion and generate both synchronous and asynchronous capture settings within

[Figure 137]

[Figure 138]

4DSloMo: 4D Reconstruction for High Speed Scene with Asynchronous Capture • 7

[Figure 139]

[Figure 140]

[Figure 141]

4DGS, sync. 4DGS, async. GS4D, sync. GS4D, async. Ours

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

Fig. 7. Quality result on our real-capture dataset.

[Figure 150]

[Figure 151]

(a) Ground Truth (b) Sync., w/o AF (c) Sync., with AF (d) Async., w/o AF (e) Async, with AF (ours)

- Fig. 8. Ablation study of asynchronous capture and diffusion model on the DNA-Rendering dataset [Cheng et al. 2023a](AF represents artifact-fix video diffusion model).

- Table 1. Quantitative comparison on the DNA-Rendering dataset [Cheng et al. 2023b]. Higher values indicate better performance for metrics marked with (↑), while lower values are preferable for metrics marked with (↓).

Method PSNR ↑ SSIM ↑ LPIPS ↓

K-Planes 22.74 0.750 0.443 4DGS [Wu et al. 2024c] 23.09 0.772 0.351 GS4D [Yang et al. 2024c] 24.75 0.797 0.337 Ours 26.76 0.845 0.293

- Table 2. Quantitative comparison on the Neural3DV dataset [Li et al. 2022]. Higher values indicate better performance for metrics marked with (↑), while lower values are preferable for metrics marked with (↓).

### Method PSNR ↑ SSIM ↑ LPIPS ↓

K-Planes 28.31 0.875 0.211 4DGS [Wu et al. 2024c] 29.99 0.928 0.252 GS4D [Yang et al. 2024c] 30.54 0.917 0.178 Ours 33.48 0.951 0.134

Benchmark for Asynchronous 4D Reconstruction Using the above camera array setup, we captured a variety of high-speed motion scenes, including dancing, sports activities, and rapid object interactions such as waving a chess piece. In total, we construct a dataset consisting of 12 sequences of asynchronously recorded multiview videos, focusing on non-linear and large-motion scenarios. Each video has a resolution of 2048 × 2248 pixels.

- 4.3 Comparison Experiments

For assessment, we use three metrics, encompassing peak-signal-tonoise ratio (PSNR), structural similarity index (SSIM) and perceptual quality measure (LPIPS). For baselines, we choose three state-of-theart 4D reconstruction methods, including K-Planes [Sara FridovichKeil and Giacomo Meanti et al. 2023], 4DGS [Wu et al. 2024c] and GS4D [Yang et al. 2024c].

We first conduct a quantitative evaluate on two existing datasets, DNA-Rendering and Neural3DV, as shown in Table 1 and Table 2. Our method achieves the best results across fidelity metrics (PSNR and SSIM) as well as perceptual metrics (LPIPS). This demonstrates the effectiveness of asynchronous capture and artifact-fix model for high-frame-rate 4D reconstruction in large-motion scenarios.

We also presenta quantitative comparisons on Neural3DV dataset [Li

et al. 2022] and DNA-Rendering dataset [Cheng et al. 2023b] in Fig. 4 and Fig. 5 respectively. The results show that our methods can produce higher fidelity renderings using the same size of input data. It becomes apparent that, due to insufficient input information in the temporal dimension, all comparison methods have generated severe artifacts and motion distortions, especially in regions of rapid motion. Specifically, methods like 4DGS and K-Planes, which estimate a deformation field on top of a static canonical representation, tend to become stuck at certain poses when lacking temporal supervision. In contrast, methods like GS4D, which treat temporal and spatial dimensions as an integrated whole, are prone to generating numerous artifacts in the absence of temporal supervision. Supporting this, the experimental results also prove that existing 4D representations

Table 3. Ablation study on DNA-Rendering dataset [Cheng et al. 2023b]. Higher values indicate better performance for metrics marked with (↑), while lower values are preferable for metrics marked with (↓).

### Capture Artifact-fix PSNR ↑ SSIM ↑ LPIPS ↓

Sync. ✗ 24.75 0.797 0.337 Sync. ✓ 24.15 0.800 0.331 Async. ✗ 26.23 0.831 0.315 Async. ✓ 26.76 0.845 0.293

are unable to automatically interpolate the temporal dimension to recover plausible motion information. Benefiting from the proposed asynchronous capture, our system hardware-wise obtains sufficient temporal supervision, enabling the reconstruction of plausible motion even in large-motion scenarios, as shown in Fig. 4 and Fig. 5. Furthermore, leveraging the intrinsic modeling capabilities of the

- 4D representation in the spatial domain and the diffusion prior, our method effectively mitigates artifacts caused by sparse views, achieving improved temporal-spatial consistency.

The qualitative results on our real-captured dataset is shown in Fig. 7. Since we are unable to capture both synchronous and asynchronous videos from the same perspective at the same moment, we captured two separate sequences while maintain the same motion patterns to ensure a fair comparison. The results show that, whether using 4DGS or GS4D, the asynchronous capture method helps the 4D reconstruction model recover more accurate geometry compared to the synchronous method. With the addition of the artifact-fix model, our approach is further capable of reconstructing fast-moving regions (e.g., arms), non-linear motions, and complex texture areas (e.g., skirts) with high quality.

- 5 Discussion

- 5.1 Effect of Asynchronous Capture

To evaluate the effectiveness of asynchronous capture, we compare it with synchronous capture on the DNA-Rendering dataset and LongVolCap datasets [Xu et al. 2024c]. Quantitative results are shown in Table 3, and qualitative comparisons are provided in Fig. 8 and Fig. 9. Under the original synchronous capture setup, the camera system’s frame rate is too low to handle fast human motion, resulting in large displacements between adjacent frames. Moreover, the 4D representation is unable to interpolate along the temporal dimension to recover plausible motion. As a result, the images rendered by the 4D Gaussian model exhibit poor visual quality (Fig.8(b)), and even the artifact-fix video diffusion model fails to generate satisfactory outputs (Fig.8(c)). In contrast, with asynchronous capture (Fig. 8(d)), although some artifacts remain due to sparse views, the reconstructed motion is more accurate. The results are clearly superior to those from synchronous capture, further demonstrating the effectiveness of asynchronous acquisition for high-frame-rate 4D reconstruction in large-motion scenarios.

- 5.2 Artifact-fix Video Diffusion Model

Effectiveness of the artifact-fix video diffusion model. To demonstrate the effectiveness of our artifact-fix model, we compare rendering results with and without it. The comparison is presented

[Figure 152]

4DSloMo: 4D Reconstruction for High Speed Scene with Asynchronous Capture • 9

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

(a) Ground Truth (b) Sync., w/o AF (c) Sync., with AF (d) Async., w/o AF (e) Async, with AF (ours)

[Figure 159]

| |
|---|

[Figure 160]

- Fig. 9. Ablation study of asynchronous capture and diffusion model on LongVolCap datasets [Xu et al. 2024c](AF represents artifact-fix video diffusion model).

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

- (a) Video Diffusion (Ours)
- (b) Image Diffusion

Frame 1 Frame 2

Frame 1 Frame 2

[Figure 165]

| |
|---|

- Fig. 10. Ablation study on video diffusion model and image diffusion model.

(a) Ground Truth (b) Input

[Figure 166]

- (c) w/o per-scene
- (d) with per-scene

[Figure 167]

[Figure 168]

| |
|---|

Fig. 11. Ablation study on per-scene finetuning.

5.3 Per-scene Finetuning Artifact-Fix Model

in Table 3 and Fig.8(d)(e). Leveraging the strong spatiotemporal priors of the video diffusion model, our approach effectively removes “floater” artifacts caused by sparse views in asynchronous capture, restores fine textures, and eliminates temporal discontinuities in the rendered video. By optimizing with the diffusion loss (Eqn.4), we distill the diffusion prior into the 4D reconstruction pipeline, ultimately producing visually compelling results, as shown in Fig. 8(e).

To further improve visual quality, we also explored the idea of using per-scene optimization strategy for our artifact-fix video diffusion model. When the input to the artifact-fix video model is complex—such as scenes in Fig. 11(b) where some regions lack texture while others contain dense artifacts—the model may struggle to distinguish between noise that should be removed and texture details that should be preserved. Consequently, its performance may become suboptimal. To better incorporate scene-specific information, we explore the idea of per-scene finetune the artifact-fix model. Specifically, we adopt a leave-one-out strategy (similar to [Yang et al. 2024b]) to construct noisy-clean video pairs from the input video, and then finetune the artifact-fix model on these pairs for a small number of iterations. The results in Fig. 11(c)(d) demonstrates that per-scene fine-tuning enables the model to recover finer dress details, as the diffusion model leverages multi-view and temporal cues to learn 4D scene distributions. While this strategy improves performance in complex scenes, the leave-one-out construction introduces significant time overhead. Therefore, we do not adopt per-scene finetuning as the default setting in this work. Nevertheless, we recognize it as a promising direction for high-quality

Comparing video diffusion with image diffusion. To verify that using a video diffusion model as the backbone for the 4D artifact-fix task outperforms image-based diffusion, we train a Stable Diffusion 1.5 model with the ControlNet Tile module [Zhang et al. 2023] on the same dataset, following [Yang et al. 2024b; Yu et al. 2024]. The outputs of the two diffusion models are shown in Fig. 10. We observe that, beyond differences in facial generation quality due to the base model’s capabilities, the video diffusion model produces temporally consistent textures around the collarbone across adjacent frames. In contrast, the image diffusion model introduces noticeable texture variation in this region due to its frame-wise randomness. This temporal inconsistency poses a major challenge to maintaining smooth motion in 4D reconstruction.

Table 4. Quantitative comparison with frame interpolation methods. Higher values indicate better performance for metrics marked with (↑), while lower values are preferable for metrics marked with (↓).

### Method PSNR ↑ SSIM ↑ LPIPS ↓

RIFE [Huang et al. 2022] 25.06 0.824 0.316 MoMo [Lew et al. 2025] 24.85 0.811 0.324 Ours 26.76 0.845 0.293

reconstruction and include it in our discussion to inspire future research.

- 5.4 Compare with frame interpolation methods

To compare with interpolation methods, we first use state-of-the-art models to generate intermediate frames, and then train GS4D on the high-frame-rate videos. We benchmarked two baselines—CNNbased model RIFE [Huang et al. 2022] and diffusion-based model MoMo [Lew et al. 2025]—on the DNA-Rendering Dataset, with the results shown in Table 4. For our target scenes with large motion, CNN-based models cause severe artifacts and deformation, while diffusion-based models generate incorrect motion. Therefore, neither performs well in these challenging scenarios.

5.5 Limitation

Due to the inherent fidelity limitations of pre-trained diffusion models, some fine textures, despite the per-scene optimization, may still exhibit reduced fidelity after artifact removal. Future advancements in foundational models could further enhance the reconstruction accuracy of asynchronous capture. In addition, our current artifactfixing model is trained on the DNA-Rendering dataset, which primarily features human performances. Consequently, while our method demonstrates strong results on such scenes, its generalization to more diverse, in-the-wild scenarios with non-human subjects or different types of motion has not yet been verified. A promising direction for future work is to train our video diffusion model on a broader range of dynamic 4D datasets as they become available.

6 Conclusion

In conclusion, we have presented a novel asynchronous capture scheme and reconstruction pipeline to address the challenge of fast-dynamic scene reconstruction. By introducing deliberate delays between cameras, our method effectively increases the frame rate, enabling more accurate modeling of intermediate motion information in fast-moving scenes. To address the viewpoint sparsity and reconstruction artifacts introduced by asynchronous capturing, we train a video-diffusion-based artifact-fix model that significantly reduces these artifacts and enhances visual quality. Compared to previous sparse 3D reconstruction methods that employ image diffusion for refinement, our video diffusion–based approach effectively resolves the temporal inconsistency issues that arise in 4D reconstruction by leveraging spatiotemporal priors from video diffusion. Our method achieves superior performance in both real and synthetic scenarios, demonstrating its robustness across diverse conditions. Furthermore, we provide the first real-world asynchronous multi-view video dataset to advance research in high-speed 4D

reconstruction. Our approach provides a promising joint hardwaresoftware solution to improve the accuracy of 4D reconstruction for fast-motion scenes, with potential applications in sports, autonomous driving, robotics, and VR/AR content creation.

7 Acknowledgements

This work was supported by the RGC Early Career Scheme (ECS) No. 24209224. This study was also supported in part by the Centre for Perceptual and Interactive Intelligence (CPII) Ltd., a CUHKled InnoCentre under the InnoHK initiative of the Innovation and Technology Commission of the Hong Kong Special Administrative Region Government.

References

Wei Cheng, Ruixiang Chen, Siming Fan, Wanqi Yin, Keyu Chen, Zhongang Cai, Jingbo Wang, Yang Gao, Zhengming Yu, Zhengyu Lin, et al. 2023a. Dna-rendering: A diverse neural actor repository for high-fidelity human-centric rendering. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 19982–19993.

Wei Cheng, Ruixiang Chen, Wanqi Yin, Siming Fan, Keyu Chen, Honglin He, Huiwen Luo, Zhongang Cai, Jingbo Wang, Yang Gao, Zhengming Yu, Zhengyu Lin, Daxuan Ren, Lei Yang, Ziwei Liu, Chen Change Loy, Chen Qian, Wayne Wu, Dahua Lin, Bo Dai, and Kwan-Yee Lin. 2023b. DNA-Rendering: A Diverse Neural Actor Repository for High-Fidelity Human-centric Rendering. arXiv preprint arXiv:2307.10173 (2023).

Kyle Gao, Yina Gao, Hongjie He, Dening Lu, Linlin Xu, and Jonathan Li. 2022. Nerf: Neural radiance field in 3d vision, a comprehensive review. arXiv preprint arXiv:2210.00379 (2022).

Ruiqi Gao, Aleksander Holynski, Philipp Henzler, Arthur Brussee, Ricardo MartinBrualla, Pratul Srinivasan, Jonathan T Barron, and Ben Poole. 2024. Cat3d: Create anything in 3d with multi-view diffusion models. arXiv preprint arXiv:2405.10314 (2024).

Zhewei Huang, Tianyuan Zhang, Wen Heng, Boxin Shi, and Shuchang Zhou. 2022. Real-time intermediate flow estimation for video frame interpolation. In European Conference on Computer Vision. Springer, 624–642.

Kai Katsumata, Duc Minh Vo, and Hideki Nakayama. 2024. A compact dynamic 3d gaussian representation for real-time dynamic view synthesis. In European Conference on Computer Vision. Springer, 394–412.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 2023.

- 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph. 42,
- 4 (2023), 139–1.

Jaihyun Lew, Jooyoung Choi, Chaehun Shin, Dahuin Jung, and Sungroh Yoon. 2025. Disentangled motion modeling for video frame interpolation. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39. 4607–4615.

Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, Richard Newcombe, et al. 2022. Neural 3d video synthesis from multi-view video. InProceedings of the IEEE/CVF conference on computer vision and pattern recognition. 5521–5531.

Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. 2024. Spacetime gaussian feature splatting for real-time dynamic view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8508–8520.

Haotong Lin, Sida Peng, Zhen Xu, Yunzhi Yan, Qing Shuai, Hujun Bao, and Xiaowei Zhou. 2022. Efficient neural radiance fields for interactive free-viewpoint video. In SIGGRAPH Asia 2022 Conference Papers. 1–9.

Youtian Lin, Zuozhuo Dai, Siyu Zhu, and Yao Yao. 2024. Gaussian-flow: 4d reconstruction with dynamic 3d gaussian particle. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21136–21145.

Xinhang Liu, Jiaben Chen, Shiu-hong Kao, Yu-Wing Tai, and Chi-Keung Tang. 2023. Deceptive-nerf: Enhancing nerf reconstruction using pseudo-observations from diffusion models. (2023).

Zhicheng Lu, Xiang Guo, Le Hui, Tianrui Chen, Min Yang, Xiao Tang, Feng Zhu, and Yuchao Dai. 2024. 3d geometry-aware deformable gaussian splatting for dynamic view synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 8900–8910.

Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. 2021. Nerf: Representing scenes as neural radiance fields for view synthesis. Commun. ACM 65, 1 (2021), 99–106.

Michael Niemeyer, Jonathan T Barron, Ben Mildenhall, Mehdi SM Sajjadi, Andreas Geiger, and Noha Radwan. 2022. Regnerf: Regularizing neural radiance fields for view synthesis from sparse inputs. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 5480–5490.

Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. 2022. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988 (2022).

4DSloMo: 4D Reconstruction for High Speed Scene with Asynchronous Capture • 11

Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. 2021. D-nerf: Neural radiance fields for dynamic scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10318–10327.

Sara Fridovich-Keil and Giacomo Meanti, Frederik Rahbæk Warburg, Benjamin Recht, and Angjoo Kanazawa. 2023. K-Planes: Explicit Radiance Fields in Space, Time, and Appearance. In CVPR.

Jiuhn Song, Seonghoon Park, Honggyu An, Seokju Cho, Min-Seop Kwak, Sungjin Cho, and Seungryong Kim. 2023. Därf: Boosting radiance fields from sparse input views with monocular depth adaptation. Advances in Neural Information Processing Systems 36 (2023), 68458–68470.

Prune Truong, Marie-Julie Rakotosaona, Fabian Manhardt, and Federico Tombari. 2023. SPARF: Neural Radiance Fields from Sparse and Noisy Poses. IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR.

Daniel Vlasic, Pieter Peers, Ilya Baran, Paul Debevec, Jovan Popović, Szymon Rusinkiewicz, and Wojciech Matusik. 2009. Dynamic Shape Capture using MultiView Photometric Stereo. ACM Transactions on Graphics (TOG) 28, 5 (2009), 174.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. 2025. Wan: Open and Advanced Large-Scale Video Generative Models. arXiv preprint arXiv:2503.20314 (2025).

Qianqian Wang, Vickie Ye, Hang Gao, Jake Austin, Zhengqi Li, and Angjoo Kanazawa.

2024. Shape of motion: 4d reconstruction from a single video. arXiv preprint arXiv:2407.13764 (2024).

Yikai Wang, Xinzhou Wang, Zilong Chen, Zhengyi Wang, Fuchun Sun, and Jun Zhu. 2025. Vidu4d: Single generated video to high-fidelity 4d reconstruction with dynamic gaussian surfels. Advances in Neural Information Processing Systems 37 (2025), 131316–131343.

Di Wu, Yebin Liu, Ivo Ihrke, Christian Theobalt, and Qionghai Dai. 2012. Performance Capture of High-Speed Motion Using Staggered Multi-View Recording. Computer Graphics Forum 31, 7 (2012), 2019–2028.

Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 2024b. 4d gaussian splatting for real-time dynamic scene rendering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 20310–20320.

Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 2024c. 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 20310–20320.

Jay Zhangjie Wu, Yuxuan Zhang, Haithem Turki, Xuanchi Ren, Jun Gao, Mike Zheng Shou, Sanja Fidler, Zan Gojcic, and Huan Ling. 2025. Difix3D+: Improving 3D Reconstructions with Single-Step Diffusion Models. arXiv preprint arXiv: 2503.01774 (2025).

Rundi Wu, Ben Mildenhall, Philipp Henzler, Keunhong Park, Ruiqi Gao, Daniel Watson, Pratul P Srinivasan, Dor Verbin, Jonathan T Barron, Ben Poole, et al. 2024a. Reconfusion: 3d reconstruction with diffusion priors. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 21551–21561.

Jamie Wynn and Daniyar Turmukhambetov. 2023. Diffusionerf: Regularizing neural radiance fields with denoising diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 4180–4189.

Zhen Xu, Sida Peng, Haotong Lin, Guangzhao He, Jiaming Sun, Yujun Shen, Hujun Bao, and Xiaowei Zhou. 2024a. 4k4d: Real-time 4d view synthesis at 4k resolution. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 20029–20040.

Zhen Xu, Sida Peng, Haotong Lin, Guangzhao He, Jiaming Sun, Yujun Shen, Hujun Bao, and Xiaowei Zhou. 2024b. 4K4D: Real-Time 4D View Synthesis at 4K Resolution. In CVPR.

Zhen Xu, Yinghao Xu, Zhiyuan Yu, Sida Peng, Jiaming Sun, Hujun Bao, and Xiaowei Zhou. 2024c. Representing Long Volumetric Video with Temporal Gaussian Hierarchy. ACM Transactions on Graphics 43, 6 (November 2024). https: //zju3dv.github.io/longvolcap

Chen Yang, Sikuang Li, Jiemin Fang, Ruofan Liang, Lingxi Xie, Xiaopeng Zhang, Wei Shen, and Qi Tian. 2024b. Gaussianobject: Just taking four images to get a highquality 3d object with gaussian splatting. arXiv e-prints (2024), arXiv–2402.

Jiawei Yang, Marco Pavone, and Yue Wang. 2023. Freenerf: Improving few-shot neural rendering with free frequency regularization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 8254–8263.

Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. 2024a. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition.

20331–20341.

Zeyu Yang, Hongye Yang, Zijie Pan, and Li Zhang. 2024c. Real-time Photorealistic Dynamic Scene Representation and Rendering with 4D Gaussian Splatting. In International Conference on Learning Representations (ICLR).

Hanyang Yu, Xiaoxiao Long, and Ping Tan. 2024. Lm-gaussian: Boost sparse-view 3d gaussian splatting with large model priors. arXiv preprint arXiv:2409.03456 (2024).

Zehao Yu, Songyou Peng, Michael Niemeyer, Torsten Sattler, and Andreas Geiger. 2022. Monosdf: Exploring monocular geometric cues for neural implicit surface reconstruction. Advances in neural information processing systems 35 (2022), 25018– 25032.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision. 3836–3847.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition. 586–595.

