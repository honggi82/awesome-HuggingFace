# arXiv:2408.13423v4[cs.CV]25Dec2024

## Decoupled Video Generation with Chain of Training-free Diffusion Model Experts

Wenhao Li1*, Yichao Cao2*, Xiu Su3†, Xi Lin4, Shan You5, Mingkai Zheng1, Yi Chen6, Chang Xu1

1University of Sydney, 2Southeast University, 3Central South University, 4Shanghai Jiaotong University, 5Sensetime Research, 6Hong Kong University of Science and Technology

#### Abstract

[Figure 1]

[Figure 2]

Video generation models hold substantial potential in areas such as filmmaking. However, current video diffusion models need high computational costs and produce suboptimal results due to extreme complexity of video generation task. In this paper, we propose ConFiner, an efficient video generation framework that decouples video generation into easier subtasks: structure control and spatialtemporal refinement. It can generate high-quality videos with chain of off-the-shelf diffusion model experts, each expert responsible for a decoupled subtask. During the refinement, we introduce coordinated denoising, which can merge multiple diffusion experts’ capabilities into a single sampling. Furthermore, we design ConFiner-Long framework, which can generate long coherent video with three constraint strategies on ConFiner. Experimental results indicate that with only 10% of the inference cost, our ConFiner surpasses representative models like Lavie and Modelscope across all objective and subjective metrics. And ConFinerLong can generate high-quality and coherent videos with up to 600 frames. All the code will be available at project website: https://confiner2025.github.io.

- (a)
- (b)

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

Figure 1. (a) Conventional video generation process. (b) Motivation of the proposed ConFiner.

first type uses T2I (Text to Image) models to generate videos directly without further training [20, 38, 41, 42]. The second type incorporates a temporal module into T2I models and trains on video datasets [2, 25, 37]. The third type is trained from scratch [1, 15, 26, 47]. Regardless of which type, these methods use a single model to undertake the entire task of video generation, like Fig. 1(a). However, video generation is extremely intricate [53]. After our in-depth analysis, we believe that this complex task consists of three subtasks: modeling the video structure, which includes designing the overall visual structure and plot; generating spatial details, ensuring each frame with sufficient clarity and high aesthetic score; and producing temporal details, maintaining consistency and coherence between frames to ensure natural and logical transitions. Therefore, relying on a single model to handle such a complex and multidimensional task is challenging.

#### 1. Introduction

Generative AI [3, 40, 50] has recently emerged as a hotspot in research, influencing various aspects of our daily life. For visual AIGC, numerous image generation models, such as Stable Diffusion [33] and Imagen [34], have achieved significant success. These models can create high-resolution images that are rich in creativity and imagination, rivaling those created by human artists. Compared to image generation, video generation models [7, 13, 14, 43] hold higher practical value with the potential to reduce expenses in the fields of filmmaking and animation.

Overall, there are three main challenges in the field of video generation [5, 22, 23, 54]: i) The quality of the generated videos is low, hard to achieve high-quality temporal and spatial modeling simultaneously [53]. ii) The generation process is time-consuming, often requiring hundreds of inference steps [44]. Utilizing a single model to handle complex video generation task is one of the key reasons for these two issues. iii) The length of the generated videos are typically short [49]. Due to limitations in VRAM, the length of videos generated in a single attempt generally ranges between only 2-3 seconds.

However, current video generation models are still in their early stages of development. Existing video diffusion models can primarily be categorized into three types. The

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

StreamingT2V

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

ConFiner-Long

"Clownfish swim among the colorful corals."

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

StreamingT2V

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

ConFiner-Lon

g

"One mech on the Pacific Ocean."

Frame 0 Frame 100 Frame 200 Frame 300 Frame 400 Frame 500 Frame 600

Figure 2. Comparison between Our ConFiner-Long and StreamingT2V [10]. We exhibit better consistency and imaging quality.

In order to enhance generation quality, some methods employ multiple models on different resolutions or in different spaces to perform progressive generation. Some methods [12, 26, 39, 52] train several diffusion models on gradually increasing resolutions to first generate low-resolution videos, and then progressively scale up. Show-1 [51] trains a model in pixel space to generate low-quality videos, followed by a latent space model to enhance quality. Compared to methods using a single model, these approaches achieve higher performance. However, each model still needs to handle both spatial and temporal modeling. This leaves each model still heavily burdened.

To improve quality of videos while reducing inference time, we rethink the demands of video generation tasks, which include modeling video structure, generating spatial details, and producing temporal details. We find out that a more rational approach is utilizing three specialized models, each handling one demand. By doing so, these models can collaboratively accomplish this comprehensive task. To this end, we propose a framework named ConFiner, which decouples the video generation process into three parts: structure control, temporal refinement, and spatial refinement. During generation, we employ chain of three ready-made diffusion experts, each specializing in respective tasks, like Fig. 1(b). In the control stage, a highly controllable T2V

(Text to Video) model is employed as control expert, tasked with structure control. During the refinement stage, a T2I model and a T2V model skilled at generating details are employed as spatial and temporal experts to refine details. This framework can reduce the burden on individual models, enhancing both the quality and speed of generation. Moreover,

- as it utilizes ready-made diffusion experts, this framework does not incur additional training costs.

Furthermore, based on ConFiner, we propose ConFinerLong framework, which can generate long videos by ensuring the coherence and consistency between video segments. As the initial noise significantly impacts the final videos, we first introduce a segments consistency initialization strategy to ensure the consistency of the initial noise between segments by sharing a base noise. Additionally, in order to enhance the coherence of the motion between segments, we propose a coherence guidance strategy that uses the gradient of noise differences between two segments to guide the denoising direction. Also, to address the flickering problem

- at the junctions of segments, we design a staggered refinement strategy that staggers the control stage and the refinement stage. It places the tail of one video structure and the head of the next into the same refinement process to achieve more natural transitions between segments.

Experimental results have shown that ConFiner requires

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

FreeNoise

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

ConFiner-Long

"A spectacular fireworks display over Sydney Harbour, 4K, high resolution"

Figure 3. Comparison of Our ConFiner-Long with FreeNoise [32]. We achieve much better imaging clarity and quality.

only 9 sampling steps (less than 5 seconds) to surpass the performance of models like AnimateDiff-Lightning [25], LaVie [39], and ModelScope T2V [37] with 100-step sampling (more than 1 minute). Furthermore, ConFiner-Long can generate high-quality coherent videos up to 600 frames long. To sum up, our contributions are as follows:

eration. And some other methods [1, 15, 26, 47] are trained from scratch. However, the generation quality and speed of these methods are unsatisfactory, which we attribute to the overwhelming burden placed on a single model to handle the complexity of video generation tasks.

#### 3. Method

- 1. We introduced ConFiner, which decouples the video generation task into three sub-tasks. It utilizes three ready-made diffusion experts, each handling its specialized task. This approach reduces the model’s burden, enhancing the quality and speed of generation.
- 2. We designed coordinated denoising strategy, allowing two experts on different noise schedulers to collaborate timestep-wise in video generation process.
- 3. We proposed ConFiner-Long framework, which harmonizes the initial states, generation directions, and transitions between segments to achieve high-quality, coherent long video production.

##### 3.1. Overview

Our ConFiner consists of two stages: the control stage and the refinement stage. In the control stage, it generates a video structure containing coarse-grained spatio-temporal information, which determines the overall structure and plot of the final video. During the refinement stage, it refines spatial and temporal details based on video structure. In this stage, we propose coordinated denoising to enable cooperation of spatial expert and temporal expert. Based on ConFiner, we introduce ConFiner-Long framework for producing coherent and consistent long videos.

#### 2. Related Work

##### 3.2. Revisiting Diffusion Models

The workflow of diffusion models consists of two processes: the forward process and the reverse denoising process. The forward process from timestep 0 to timestep t can be expressed as follows:

Diffusion models (DMs). DMs have achieved remarkable successes in the generation of images [4, 6, 21, 29, 45, 46], music [8, 16, 18, 27, 30], and 3D models [19, 24, 28, 31, 35, 48]. These models typically involve thousands of timesteps, with a scheduler that manages the noise level. Diffusion models consist of two processes [11]. In the forward process, noise is progressively added to the original data until it is completely transformed into noise. During the reverse denoising process, the model starts with random noise and gradually eliminates the noise using a denoising model, ultimately transforming it into a target sample.

xt = √αtx0 + √1 − αtϵ (1)

where αt = 1 − βt, αt = ti=1 αi, t is the diffusion step, ϵ is a random noise sampled from Standard Gaussian Distri-

bution N(0,1) and βt is a small positive constant between 0 and 1, representing the noise level of each timestep.

During the reverse denoising process, starting from a random noise at timestep T, the denoising model progressively predicts xt−1 from xt, ultimately getting the target data x0. Taking DDIM [36] as an example, the denoising model initially uses xt to predict the noise. Then, xt and the predicted noise are utilized together to predict x0 via the following expression:

Video Diffusion Models (VDMs). Compared to the success of diffusion models in areas like image generation, VDMs are still at a very early stage. Some methods [20, 38, 41, 42] use stable diffusion without additional training for direct video generation. These methods suffer from poor coherence and evident visual tearing. Some Models [2, 25, 37] convert the U-Net of stable diffusion [39] into a 3D U-Net through the addition of temporal convolution or attention, and train it on video datasets to achieve video gen-

√1 − αtϵ(θt)(xt) √αt

xt −

(2)

xˆ0 =

[Figure 53]

Consistency Initialization

i Text P j Text P k Text P

Base Noise

### … …

Coherence Guidance

Structurei Structurej Structurek Refinei Refinej Refinek

ConFiner-Long

[Figure 54]

Text P

[Figure 55]

[Figure 56]

###### Coordinated Denoising

×T

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

Spatial Expert

Spatial Expert Convert

Control Expert

Convert

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

Temporal Expert

Video Structure

Structure Generation Spatio-temporal Refinement

- Figure 4. Pipeline of Our ConFiner and ConFiner-Long. ConFiner decouples the video generation process. Firstly, control expert generates a video structure. Subsequently, temporal and spatial experts perform the refinement of spatio-temporal details. Spatial and temporal experts work together with our coordinated denoising. By adding consistency initialization, coherence guidance and staggered refinement to ConFiner, ConFiner-Long can generate coherent long videos.

where Vˆ0(tk(i1)) represents the predicted V0 at timestep tk(i1), Vtk(i1) denotes V at timestep tk(i1), θcon represents control expert and Scon is the scheduler of control expert.

where ϵ(θt)(xt), xˆ0 represents the predicted noise and x0.

Then, based on the predicted noise and xˆ0 , a prediction for xt−1 is derived as:

While we completed the entire sampling to obtain the first version of video V0, the quality and coherence of the video are compromised due to our choice of a small Ti

xt−1 = αt−1 · xˆ0 + 1 − αt−1·ϵ(θt)(xt) (3)

.

1

Therefore, we introduce Te steps of noise to V0. This operation is intended to create refinement opportunities for spatial and temporal experts. In this noise addition process, we utilize the Scheduler Ss from the spatial expert used in refinement stage, resulting in the noisy video VT′e at timestep Te. Transformed from Eq. (1), this noise addition process can be expressed as:

By combining Eq. (2) and Eq. (3), single-step denoising can be expressed as:

xˆ0,xt−1 = Denoising(θ,xt,t,S) (4)

where S denotes the noise scheduler and θ represents the corresponding denoising model.

##### 3.3. Video Structure Generation

In the control stage, we select a video diffusion model skilled at handling video structure and employ it as control expert. The scheduler used in this expert can be denoted as Scon. During inference, to reduce computational overhead, we opt for a DDIM scheduler with a total inference step of Ti

. When conducting inference, the list of timesteps utilized is: [t1(i1),t2(i1),...,tT

1

(i1)]. The selection of timesteps is made at uniform intervals.

i1

After obtaining the timesteps list, we start with a random noise VtTi1

and progressively denoise over these

(i1)

timesteps, getting the first version of the video V0. Singlestep sampling from Eq. (4) can be rewritten as follows.

Vˆ0(tk(i1)),Vtk−1(i1) = Denoising(θcon,Vtk(i1),tk(i1),Scon)

(5)

VT′e = αT

(Ss) · ϵ (6) where αT

(Ss) · V0 + 1 − αT

e

e

(Ss) is the αt in scheduler Ss at timestep Te.

e

##### 3.4. Spatial and Temporal Details Refinement

During the refinement stage, we add spatial and temporal details with spatial expert and temporal expert in

the process of transforming VT′e to V0′. Similar to the control stage, we select Ti

steps for sampling between timestep Te and timestep 0. The list of timesteps used is: [t1(i2),t2(i2),...,tT

2

(i2)].

i2

Given that two experts respectively excel in spatial and temporal modeling, we aim to synergistically utilize both experts in the process of denoising VT′e to V0′, thus enhancing the spatio-temporal detail. A straightforward approach is alternating between the two experts at each timestep,

leveraging the strengths of both models concurrently. In this case, Eq. (4) can be rewritten as follows:

Vˆ0(tk(i2)),Vt′k−1(i2) = Denoising(θX,Vt′k(i2),tk(i2),Ss)

- θS if k ≡ 2 (mod 0)
- θT if k ≡ 2 (mod 1)

where θX =

(7) where θS, θT represent sptial expert and temporal expert, and Ss denotes spatial expert’s scheduler.

However, this method is ineffective because spatial expert and temporal expert are often on different noise scheduler. The data distributions for the spatial and temporal experts at the same timestep are inconsistent. The original data is on the scheduler of spatial expert, and directly switching to the scheduler of temporal expert at a certain timestep leads to conflicts and inconsistencies. To trans-

form Vt′k(i2) to Vt′k−1(i2), we provide two options.

###### Option 1 (Standard Denoising): Since the original data

VT′e is on the scheduler of spatial expert, we can directly employ the spatial expert for denoising at time step tk(i2):

Vˆ0(tk(i2)),Vt′k−1(i2) = Denoising(θS,Vt′k(i2),tk(i2),Ss)

(8)

###### Option 2 (Coordinated Denoising): Although two experts’ schedulers differ, both schedulers share the same distribution at timestep 0. Hence, we can utilize timestep 0 to establish a connection between the two schedulers, facilitating the concurrent use of two experts within the same timestep. The specific details of this process are as follows.

First, at timestep tk(i2), given Vt′k(i2), we employ the spatial expert for a one-step inference as Eq. (8). After obtaining the predicted Vˆ0(tk(i2)), it can be converted to Vt′′k(i2) on the scheduler of temporal expert.

k(i2)(St)·Vˆ0(tk(i2))+ 1 − αt

Vt′′k(i2) = αt

k(i2)(St)·ϵ

(9) where St represents the noise scheduler of temporal expert.

Then we can employ the temporal expert for denoising: Vˆ0(tk(i2)),Vt′′k−1(i2) = Denoising(θT,Vt′′k(i2),tk(i2),St)

(10)

This version of Vˆ0(tk(i2)) predicted by temporal expert contains richer temporal information and demonstrates enhanced inter-frame coherence. Subsequently, we transform Vˆ0(tk(i2)) using the scheduler of spatial expert into a Vt′k(i2) with more extensive temporal information.

k(i2)(Ss)·Vˆ0(tk(i2))+ 1 − αt

Vt′k(i2) = αt

k(i2)(Ss)·ϵ (11)

Finally, the spatial expert is used again to predict Vt′k−1(i2) including more spatio-temporal details as Eq. (8).

Algorithm 1 ConFiner (Control + Refinement)

- 1: Input: Prompt P, Control Expert Con, Spatial Expert S, Temporal Expert T, Noisy timestep Te
- 2: Output: Generated video V
- 3: V0 ← Generate(P,Con) ▷ Generate coarse video.
- 4: V ideo Structure ← Add noise(V0,Te,Con)

- 5: VT′e ← V ideo Structure ▷ Extract video structure.

- 6: for each refinement step Tk do
- 7: if Standard Denoising then
- 8: VT′k−1 ← Denoise(VT′k,Tk,S)
- 9: else if Coordinated Denoising then
- 10: V0′(Tk) ← Denoise(VT′k,Tk,S,P)
- 11: VT′′k ← Add noise(V0′(Tk),Tk,T)
- 12: V0′′(Tk) ← Denoise(VT′′k,Tk,T,P)
- 13: VT′k ← Add noise(V0′′(Tk),Tk,S)
- 14: VT′k−1 ← Denoise(VT′k,Tk,S,P)
- 15: Return V = V0′, with refined spatial-temporal details.

##### 3.5. ConFiner-Long Framework

We also leverage ConFiner to design a pipeline for long video generation. This pipeline generate multiple short video segments and introduces three strategies to ensure consistency and coherence between these segments.

First, we design consistency initialization strategy to promote consistency between segments. The initial noise affects the content of video significantly. To improve the consistency between segments, we first sample a Noise base ∈ RH×W×C×F, which is then subjected to frame-wise shuffling to obtain the initial noise for each segment. Sharing base noise enhances the content consistency between segments while shuffling maintains a little randomness.

Although consistency initialization have ensured content consistency between segments, if the motion modes of video structures are not coherent, it will be impossible to combine them into a reasonable long video. Thus, we propose a coherent guidance to promote the motion mode of new segment to follow the preceding segment. In video generation, predicted noises affect the direction of generation and determine the motion mode. So we generate each structure one by one, using noises of the previous segments to guide the subsequent structure. Specifically, during the sampling process, we use the gradient of the L2 loss to guide the sampling direction. The L2 loss is calculated between the predicted noise of the current segment and the noise in the previous segment. The guided noise is calculated as follows:

ϵS

t = ϵˆS

∥ϵˆS

t − ϵS

t ∥2 (12) where ϵˆS

t − γ∇ϵˆS2

2

2

2

1

t

t represents the noise of current segment predicted by denoising model at timestep t, ϵS

2

t is the noise of former segment at timestep t and γ is a constant.

1

Subject Consistency↑

Imaging Quality↑

Inference Steps

Motion Smoothness↑

Aesthetic Quality ↑

Method

Lavie [39] 10 0.940 ± 0.001 0.967 ± 0.001 0.570 ± 0.003 0.658 ± 0.008 Lavie [39] 20 0.954 ± 0.002 0.966 ± 0.002 0.587 ± 0.001 0.683 ± 0.001 Lavie [39] 50 0.958 ± 0.004 0.965 ± 0.006 0.597 ± 0.005 0.696 ± 0.005 Lavie [39] 100 0.957 ± 0.003 0.965 ± 0.001 0.596 ± 0.006 0.695 ± 0.007

AnimateDiff-Lightning [25] 10 0.983 ± 0.002 0.983 ± 0.001 0.635 ± 0.002 0.689 ± 0.001 AnimateDiff-Lightning [25] 20 0.984 ± 0.004 0.980 ± 0.002 0.636 ± 0.006 0.697 ± 0.002 AnimateDiff-Lightning [25] 50 0.981 ± 0.004 0.971 ± 0.003 0.638 ± 0.002 0.705 ± 0.003 AnimateDiff-Lightning [25] 100 0.977 ± 0.004 0.964 ± 0.003 0.623 ± 0.006 0.699 ± 0.005

Modelscope T2V [37] 10 0.983 ± 0.002 0.980 ± 0.001 0.570 ± 0.002 0.670 ± 0.004 Modelscope T2V [37] 20 0.985 ± 0.004 0.980 ± 0.003 0.575 ± 0.003 0.702 ± 0.004 Modelscope T2V [37] 50 0.988 ± 0.002 0.990 ± 0.001 0.592 ± 0.002 0.716 ± 0.002 Modelscope T2V [37] 100 0.987 ± 0.002 0.990 ± 0.000 0.594 ± 0.001 0.715 ± 0.004

ConFiner w/ Lavie 9 0.993 ± 0.000 0.991 ± 0.000 0.699 ± 0.005 0.734 ± 0.005 ConFiner w/ Lavie 18 0.993 ± 0.002 0.990 ± 0.001 0.703 ± 0.009 0.739 ± 0.004

ConFiner w/ Modelscope 9 0.994 ± 0.000 0.991 ± 0.000 0.698 ± 0.006 0.731 ± 0.003 ConFiner w/ Modelscope 18 0.994 ± 0.002 0.991 ± 0.002 0.707 ± 0.004 0.739 ± 0.004

- Table 1. Objective Evaluation Results. In this experiment, ConFiner utilized AnimateDiff-Lightning as the control expert and selected stable diffusion 1.5 for spatial expert. Lavie and Modelscope T2V are chosen as temporal expert.

Coherence Text-Match Visual Quality

Method Bad↓ Normal∼ Good↑ Bad↓ Normal∼ Good↑ Bad↓ Normal∼ Good↑ AnimateDiff-Lightning 0.37 0.42 0.21 0.06 0.51 0.43 0.29 0.51 0.20

Modelscope T2V 0.14 0.48 0.38 0.21 0.53 0.26 0.34 0.45 0.21 Lavie 0.11 0.46 0.43 0.24 0.46 0.30 0.32 0.49 0.19

ConFiner w/ Lavie 0.08 0.43 0.49 0.08 0.48 0.44 0.13 0.36 0.51 ConFiner w/ Modelscope 0.07 0.42 0.51 0.08 0.50 0.42 0.09 0.41 0.50

- Table 2. Subjective Evaluation Results. Each model generates videos using the top 100 prompts from Vbench [17]. The videos were evaluated by 30 users, with each video being rated as good, normal, or bad on three dimensions.

Additionally, we introduce a staggered refinement mechanism to further improve the overall coherence of the video. In our segmented generation approach, the transition points between segments tend to exhibit the highest inconsistency. Therefore, in long video generation, we perform the Control Stage and Refinement Stage in a staggered manner. Specifically, the latter half of the preceding structure and the former half of the succeeding structure are used as inputs for a same refinement pass. The refinement stage can seamlessly stitch the two structures together, which ensures a more natural and smoother transition between segments.

Segment = Refine(SpL/2:L + Sn0:L/2) (13)

Where Sp represents the previous structure, Sn represents the next structure, L represents structures’ frames number.

In this way, coherent guidance can make the noise of the two segments similar, which allows the motion mode of the latter segment to inherit that of the previous segment. Additionally, coherence guidance also reduces the pixel distance between noises of two segments, which can help maintain content consistency between segments.

#### 4. Experiments

In the experiment, we selected AnimateDiff-Lightning [25] as control expert, and Stable Diffusion 1.5 [33] as the spatial expert. For the temporal expert, we opted for two opensource models, lavie [39] and modelscope [37].

##### 4.1. Objective Evaluation

For objective evaluation experiments, we utilized the cutting-edge benchmark, Vbench [17]. Vbench provides 800 prompts that test various capabilities of video generation models. In our experiments, each model generated 800 videos using these prompts, and the resulting videos were assessed using four metrics to evaluate their Temporal Quality and Frame-wise Quality.

For Temporal Quality Metrics, we use Subject Consistency and Motion Smoothness. For Frame-wise Quality Metrics, we use Aesthetic Quality and Imaging Quality.

In this experiment, we employed AnimateDiffLightning, Lavie, and mocelscope T2V to generate over total timesteps of 10, 20, 50, and 100. We then utilize our

Subject Consistency↑

Imaging Quality↑

Control Stage Steps Te

Motion Smoothness↑

Aesthetic Quality ↑

Method

ConFiner w/ Lavie 4 50 0.993 0.991 0.703 0.733 ConFiner w/ Lavie 4 100 0.993 0.990 0.702 0.737 ConFiner w/ Lavie 4 200 0.992 0.989 0.710 0.744 ConFiner w/ Lavie 4 300 0.978 0.986 0.383 0.303 ConFiner w/ Lavie 4 500 0.967 0.983 0.338 0.265

ConFiner w/ Modelscope 4 50 0.995 0.991 0.701 0.733 ConFiner w/ Modelscope 4 100 0.994 0.991 0.698 0.733 ConFiner w/ Modelscope 4 200 0.994 0.990 0.712 0.736 ConFiner w/ Modelscope 4 300 0.990 0.987 0.560 0.429 ConFiner w/ Modelscope 4 500 0.993 0.992 0.513 0.370

ConFiner w/ Lavie 8 50 0.994 0.991 0.708 0.741 ConFiner w/ Lavie 8 100 0.993 0.990 0.706 0.739 ConFiner w/ Lavie 8 200 0.991 0.989 0.716 0.742 ConFiner w/ Lavie 8 300 0.983 0.985 0.718 0.744 ConFiner w/ Lavie 8 500 0.978 0.980 0.721 0.751

ConFiner w/ Modelscope 8 50 0.994 0.991 0.708 0.740 ConFiner w/ Modelscope 8 100 0.994 0.991 0.707 0.739 ConFiner w/ Modelscope 8 200 0.993 0.990 0.716 0.742 ConFiner w/ Modelscope 8 300 0.992 0.989 0.720 0.747 ConFiner w/ Modelscope 8 500 0.991 0.987 0.727 0.752

- Table 3. Ablation Study of Te. In most cases, as Te increases, the temporal metric decreases and the imaging quality improves. However, when the control stage involves only 4 steps, too high values of Te (such as 300 or 500) can lead to imaging collapse.

Time Cost ConFiner Lavie[39] Animate Diffusion[9] Modelscope[37]

Training 0 > 100× A100 day > 100× A100 day > 100× A100 day Inference ≈5S >1min >1min >1min

- Table 4. Comparison of Training and Inference Time. We don’t need training and only require less than 10% inference overhead.

ConFiner to conduct generation with 9(4+5) and 18(8+10) timesteps, where Te is set to 100. All evaluation results are presented in Tab. 1. Each individual experiment can be completed in 3-5 hours on a single RTX 4090. In each experiment, we repeated for five times with different random seeds.

##### 4.2. Subjective Evaluation

In our subjective evaluation, we employed our ConFiner with 18 inference steps to generate videos using the top 100 prompts from Vbench. These videos were evaluated alongside those generated by AnimateDiff-Lightning, Modelscope T2V, and Lavie with 50-step inference, by 30 users. Users rated each video across three dimensions: coherence, text-match, and visual quality, each dimension being categorized into three levels: good, normal, and bad. The scoring results are shown in Tab. 2.

##### 4.3. Comparison of Computation Efficiency

In this section, we compare the training and inference cost of our ConFiner with other video diffusion models. The results are shown in Tab. 4.

##### 4.4. Ablation Study on Control and Refinement Level

As Eq. (6), we apply noise for Te steps to the videos generated during the control stage to create optimization space for the refinement stage. A larger Te value increases the impact of the refinement stage. For the four settings same as objective experiment, we set Te to 50, 100, 200, 300, and 500, with other experimental settings consistent. The performance comparison is shown in Tab. 3.

##### 4.5. Ablation Study on Coordinated Denoising

To verify the effectiveness of coordinated denoising, we conducted ablation experiments on the denoising type during the refinement stage. Specifically, in this experiment, we used Lavie and ModelScope as the temporal experts, setting the total inference steps to 9 and 18, respectively, thus constructing four experimental settings. For each setting, we refined using three different denoising types during the refinement stage: using coordinated denoising; using only the temporal expert; and using only the spatial expert. The performance of the three denoising types is shown in Tab. 5.

Subject Consistency↑

Imaging Quality↑

Denoising Type

Inference Steps

Motion Smoothness↑

Aesthetic Quality ↑

Method

ConFiner w/ Lavie 9 Coordinated Denoising 0.993 0.991 0.699 0.734 ConFiner w/ Lavie 9 Only Temporal Expert 0.994 0.993 0.552 0.618 ConFiner w/ Lavie 9 Only Spatial Expert 0.883 0.907 0.749 0.766 ConFiner w/ Lavie 18 Coordinated Denoising 0.993 0.990 0.703 0.739 ConFiner w/ Lavie 18 Only Temporal Expert 0.993 0.991 0.583 0.632 ConFiner w/ Lavie 18 Only Spatial Expert 0.859 0.880 0.754 0.758

ConFiner w/ Modelscope 9 Coordinated Denoising 0.994 0.991 0.698 0.731 ConFiner w/ Modelscope 9 Only Temporal Expert 0.995 0.993 0.518 0.599 ConFiner w/ Modelscope 9 Only Spatial Expert 0.912 0.922 0.732 0.758 ConFiner w/ Modelscope 18 Coordinated Denoising 0.994 0.991 0.707 0.739 ConFiner w/ Modelscope 18 Only Temporal Expert 0.993 0.992 0.577 0.641 ConFiner w/ Modelscope 18 Only Spatial Expert 0.861 0.893 0.765 0.772

- Table 5. Ablation Study of Denoising Type. Coordinated denoising achieves a balance between spatial quality and temporal quality.

"A clownfish in the seabed, with background of sand."

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

All Strategies

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

W/O Staggered Refinement

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

W/O Coherence Guidance

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

W/O consistency initialization

Frame 0 Frame 2 Frame 4 Frame 6 Frame 8 Frame 10 Frame 12 Frame 14

From Structure i From Structure i+1

- Figure 5. Ablation Study on Three Strategies of ConFiner-Long. Three strategies work together to achieve coherence between segments.

##### 4.6. Ablation Study on Strategies of ConFiner-Long

In this section, we conducted ablation experiments on three strategies of ConFiner-Long framework. Using the same preceding video segments, we generated subsequent video segments with either all strategies or only two. The visual comparison of the four video segments against the preceding one is shown in Fig. 5. The overall visual comparison between ConFiner-Long and the existing training-free long video generation method Freenoise[32] is shown in Fig. 3.

#### 5. Conclusion

In this paper, we introduce ConFiner, a training-free framework that can generate high-quality videos with chain of

diffusion model experts. It decouples video generation into three subtasks: structure control, spatial refinement and temporal refinement. Each subtask is handled by a off-theshelf expert skilled at this task. Additionally, we propose coordinated denoising to enable two expert cooperate at the timestep level when denoising. Based on ConFiner, we also design ConFiner-Long framework to generate long coherent videos by harmonizing various segments. Experimental results confirm that our ConFiner enhances the aesthetics and coherence of generated videos while reducing sampling time significantly. And our ConFiner-Long can generate consistent and coherent videos with up to 600 frames. Our approach paves the way for cost-effective new possibilities in filmmaking, animation production, and video editing.

#### References

- [1] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, et al. Lumiere: A spacetime diffusion model for video generation. arXiv preprint arXiv:2401.12945, 2024. 1, 3
- [2] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 1, 3
- [3] Yihan Cao, Siyu Li, Yixin Liu, Zhiling Yan, Yutong Dai, Philip S Yu, and Lichao Sun. A comprehensive survey of ai-generated content (aigc): A history of generative ai from gan to chatgpt. arXiv preprint arXiv:2303.04226, 2023. 1
- [4] Wenhu Chen, Hexiang Hu, Yandong Li, Nataniel Ruiz, Xuhui Jia, Ming-Wei Chang, and William W Cohen. Subject-driven text-to-image generation via apprenticeship learning. Advances in Neural Information Processing Systems, 36, 2024. 3
- [5] Joseph Cho, Fachrina Dewi Puspitasari, Sheng Zheng, Jingyao Zheng, Lik-Hang Lee, Tae-Ho Kim, Choong Seon Hong, and Chaoning Zhang. Sora as an agi world model? a complete survey on text-to-video generation. arXiv preprint arXiv:2403.05131, 2024. 1
- [6] Dave Epstein, Allan Jabri, Ben Poole, Alexei Efros, and Aleksander Holynski. Diffusion self-guidance for controllable image generation. Advances in Neural Information Processing Systems, 36:16222–16239, 2023. 3
- [7] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7346–7356, 2023. 1
- [8] Zach Evans, Julian D Parker, CJ Carr, Zack Zukowski, Josiah Taylor, and Jordi Pons. Long-form music generation with latent diffusion. arXiv preprint arXiv:2404.10301, 2024. 3
- [9] Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 7
- [10] Roberto Henschel, Levon Khachatryan, Daniil Hayrapetyan, Hayk Poghosyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. arXiv preprint arXiv:2403.14773, 2024. 2
- [11] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 3
- [12] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 2
- [13] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen

- video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 1
- [14] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. Advances in Neural Information Processing Systems, 35:8633–8646, 2022. 1
- [15] Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022. 1, 3
- [16] Qingqing Huang, Daniel S Park, Tao Wang, Timo I Denk, Andy Ly, Nanxin Chen, Zhengdong Zhang, Zhishuai Zhang, Jiahui Yu, Christian Frank, et al. Noise2music: Textconditioned music generation with diffusion models. arXiv preprint arXiv:2302.03917, 2023. 3
- [17] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. arXiv preprint arXiv:2311.17982, 2023. 6
- [18] Shulei Ji, Xinyu Yang, and Jing Luo. A survey on deep learning for symbolic music generation: Representations, algorithms, evaluations, and challenges. ACM Computing Surveys, 56(1):1–39, 2023. 3
- [19] Animesh Karnewar, Andrea Vedaldi, David Novotny, and Niloy J Mitra. Holodiffusion: Training a 3d diffusion model using 2d images. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18423–18433, 2023. 3
- [20] Levon Khachatryan, Andranik Movsisyan, Vahram Tadevosyan, Roberto Henschel, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Text2video-zero: Textto-image diffusion models are zero-shot video generators. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15954–15964, 2023. 1, 3
- [21] Hyung-Kwon Ko, Gwanmo Park, Hyeon Jeon, Jaemin Jo, Juho Kim, and Jinwook Seo. Large-scale text-to-image generation models for visual artists’ creative works. In Proceedings of the 28th international conference on intelligent user interfaces, pages 919–933, 2023. 3
- [22] Wentao Lei, Jinting Wang, Fengji Ma, Guanjie Huang, and Li Liu. A comprehensive survey on human video generation: Challenges, methods, and insights. arXiv preprint arXiv:2407.08428, 2024. 1
- [23] Chengxuan Li, Di Huang, Zeyu Lu, Yang Xiao, Qingqi Pei, and Lei Bai. A survey on long video generation: Challenges, methods, and prospects. arXiv preprint arXiv:2403.16407,

2024. 1

- [24] Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3d: High-resolution text-to-3d content creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 300–309, 2023. 3
- [25] Shanchuan Lin and Xiao Yang. Animatediff-lightning: Cross-model diffusion distillation. arXiv preprint arXiv:2403.12706, 2024. 1, 3, 6

- [26] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024. 1, 2, 3
- [27] Gautam Mittal, Jesse Engel, Curtis Hawthorne, and Ian Simon. Symbolic music generation with diffusion models. arXiv preprint arXiv:2103.16091, 2021. 3
- [28] Shentong Mo, Enze Xie, Ruihang Chu, Lanqing Hong, Matthias Niessner, and Zhenguo Li. Dit-3d: Exploring plain diffusion transformers for 3d shape generation. Advances in neural information processing systems, 36:67960–67971,

2023. 3

- [29] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International conference on machine learning, pages 8162–8171. PMLR,

2021. 3

- [30] Zachary Novack, Julian McAuley, Taylor Berg-Kirkpatrick, and Nicholas J Bryan. Ditto: Diffusion inferencetime t-optimization for music generation. arXiv preprint arXiv:2401.12179, 2024. 3
- [31] Ben Poole, Ajay Jain, Jonathan T Barron, and Ben Mildenhall. Dreamfusion: Text-to-3d using 2d diffusion. arXiv preprint arXiv:2209.14988, 2022. 3
- [32] Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, and Ziwei Liu. Freenoise: Tuning-free longer video diffusion via noise rescheduling. arXiv preprint arXiv:2310.15169, 2023. 3, 8
- [33] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 6
- [34] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. Advances in neural information processing systems, 35:36479–36494, 2022. 1
- [35] Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. Mvdream: Multi-view diffusion for 3d generation. arXiv preprint arXiv:2308.16512, 2023. 3
- [36] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020. 3
- [37] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023. 1, 3, 6, 7
- [38] Wen Wang, Yan Jiang, Kangyang Xie, Zide Liu, Hao Chen, Yue Cao, Xinlong Wang, and Chunhua Shen. Zero-shot video editing using off-the-shelf image diffusion models. arXiv preprint arXiv:2303.17599, 2023. 1, 3
- [39] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. arXiv preprint arXiv:2309.15103, 2023. 2, 3, 6, 7

- [40] Jiayang Wu, Wensheng Gan, Zefeng Chen, Shicheng Wan, and Hong Lin. Ai-generated content (aigc): A survey. arXiv preprint arXiv:2304.06632, 2023. 1
- [41] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023. 1, 3
- [42] Ruiqi Wu, Liangyu Chen, Tong Yang, Chunle Guo, Chongyi Li, and Xiangyu Zhang. Lamp: Learn a motion pattern for few-shot-based video generation. arXiv preprint arXiv:2310.10769, 2023. 1, 3
- [43] Zhen Xing, Qijun Feng, Haoran Chen, Qi Dai, Han Hu, Hang Xu, Zuxuan Wu, and Yu-Gang Jiang. A survey on video diffusion models. arXiv preprint arXiv:2310.10647, 2023. 1
- [44] Zhen Xing, Qi Dai, Han Hu, Zuxuan Wu, and Yu-Gang Jiang. Simda: Simple diffusion adapter for efficient video generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7827– 7839, 2024. 1
- [45] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for textto-image generation. Advances in Neural Information Processing Systems, 36, 2024. 3
- [46] Zeyue Xue, Guanglu Song, Qiushan Guo, Boxiao Liu, Zhuofan Zong, Yu Liu, and Ping Luo. Raphael: Text-to-image generation via large mixture of diffusion paths. Advances in Neural Information Processing Systems, 36, 2024. 3
- [47] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 1, 3
- [48] Taoran Yi, Jiemin Fang, Junjie Wang, Guanjun Wu, Lingxi Xie, Xiaopeng Zhang, Wenyu Liu, Qi Tian, and Xinggang Wang. Gaussiandreamer: Fast generation from text to 3d gaussians by bridging 2d and 3d diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6796–6807, 2024. 3
- [49] Shengming Yin, Chenfei Wu, Huan Yang, Jianfeng Wang, Xiaodong Wang, Minheng Ni, Zhengyuan Yang, Linjie Li, Shuguang Liu, Fan Yang, et al. Nuwa-xl: Diffusion over diffusion for extremely long video generation. arXiv preprint arXiv:2303.12346, 2023. 1
- [50] Chaoning Zhang, Chenshuang Zhang, Sheng Zheng, Yu Qiao, Chenghao Li, Mengchun Zhang, Sumit Kumar Dam, Chu Myaet Thwal, Ye Lin Tun, Le Luang Huy, et al. A complete survey on generative ai (aigc): Is chatgpt from gpt-4 to gpt-5 all you need? arXiv preprint arXiv:2303.11717, 2023. 1
- [51] David Junhao Zhang, Jay Zhangjie Wu, Jia-Wei Liu, Rui Zhao, Lingmin Ran, Yuchao Gu, Difei Gao, and Mike Zheng Shou. Show-1: Marrying pixel and latent diffusion models for text-to-video generation. arXiv preprint arXiv:2309.15818, 2023. 2

- [52] Shiwei Zhang, Jiayu Wang, Yingya Zhang, Kang Zhao, Hangjie Yuan, Zhiwu Qin, Xiang Wang, Deli Zhao, and Jingren Zhou. I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145, 2023. 2
- [53] Yabo Zhang, Yuxiang Wei, Xianhui Lin, Zheng Hui, Peiran Ren, Xuansong Xie, Xiangyang Ji, and Wangmeng Zuo. Videoelevator: Elevating video generation quality with versatile text-to-image diffusion models. arXiv preprint arXiv:2403.05438, 2024. 1
- [54] Pengyuan Zhou, Lin Wang, Zhi Liu, Yanbin Hao, Pan Hui, Sasu Tarkoma, and Jussi Kangasharju. A survey on generative ai and llm for video generation, understanding, and streaming. arXiv preprint arXiv:2404.16038, 2024. 1

