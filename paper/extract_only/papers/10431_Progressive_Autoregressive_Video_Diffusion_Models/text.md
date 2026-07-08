## Progressive Autoregressive Video Diffusion Models

Desai Xie†1 Zhan Xu2 Yicong Hong2 Hao Tan2 Difan Liu2 Feng Liu2 Arie Kaufman1 Yang Zhou2 1Stony Brook University 2Adobe Research

# arXiv:2410.08151v2[cs.CV]18May2025

### Abstract

t = 1

sample new noise t = 1

sample new noise

t = 1

0

… …

…

t = 0

Current frontier video diffusion models have demonstrated remarkable results at generating high-quality videos. However, they can only generate short video clips, normally around 10 seconds or 240 frames, due to computation limitations during training. Existing methods naively achieve autoregressive long video generation by directly placing the ending of the previous clip at the front of the attention window as conditioning, which leads to abrupt scene changes, unnatural motion, and error accumulation. In this work, we introduce a more natural formulation of autoregressive long video generation by revisiting the noise level assumption in video diffusion models. Our key idea is to 1. assign the frames with per-frame, progressively increasing noise levels rather than a single noise level and 2. denoise and shift the frames in small intervals rather than all at once. This allows for smoother attention correspondence among frames with adjacent noise levels, larger overlaps between the attention windows, and better propagation of information from the earlier to the later frames. Video diffusion models equipped with our progressive noise schedule can autoregressively generate long videos with much improved fidelity compared to the baselines and minimal quality degradation over time. We present the first results on text-conditioned 60-second (1440 frames) long video generation at a quality close to frontier models. Code and video results are available at https://desaixie.github.io/pa-vdm/.

…

replace replace

Video Diffusion … Model

Video Diffusion Model

30x

30x

t = 0

…

…

…

…

save

save

…

…

…

…

[Figure 1]

[Figure 2]

[Figure 3]

long video:

[Figure 4]

[Figure 5]

(a) the replacement method

t = 1

sample new noise

initialization

t = 1

t = [1/30, 2/30, …, 29/30, 1]

append

0

… …

###### …

###### …

shift forward

shift forward

Progressive Autoregressive Video Diffusion Model

Progressive Autoregressive Video Diffusion Model

…

1x

1x

t = [0, 1/30, …, 28/30, 29/30]

…

…

…

…

save save

|[Figure 6]|
|---|

|[Figure 7]|
|---|

…

[Figure 8]

long video:

[Figure 9]

[Figure 10]

(b) our PA-VDM

Figure 1. Comparison of autoregressive long video generation methods. Top: the replacement method, which replaces the front of noisy latent frames with the ending of previous clip as condition and denoising all the frames at once. Bottom: our PA-VDM, which applies progressive noise levels and denoises and shifts the frames in small intervals. The final long video consists of autoregressively generated clean frames. ⊕ denotes concatenation. The noise level t for each frame is illustrated by the solid color of the frame, where darker colors are closer to 1 and lighter colors are closer to 0.

### 1. Introduction

Frontier video diffusion models [3, 9, 21, 22, 24, 29, 31, 34, 39, 51, 55] have recently demonstrated remarkable success in generating high-quality video contents by scaling up transformer-based [32, 48] architectures. However, they can only generate videos of relatively short duration, typically up to about 10 seconds or 240 frames, due to the demanding computation cost of long-sequence training. This temporal

restriction leads to challenges for broader applications that require longer, more continuous video outputs.

†This work is done while Desai is an intern at Adobe Research.

Several approaches [1, 7, 12, 15, 58] have been proposed to autoregressively apply video diffusion models for long video generation; they generate short video clips in a windowed fashion, where each subsequent clip conditions on the final frames of the previous one. One solution [7, 58] directly places the conditioning frames into the input frames, replacing the noisy frames. Another solution [15, 43] additionally adds the same level of noise to the conditioning frames as the noisy frames. This naive way of conditioning suffers from various flaws, including temporal inconsistency, abrupt scene changes, unnatural motion dynamics, and accumulated errors that lead to divergence.

In this work, we propose Progressive Autoregressive Video Diffusion Models (PA-VDM) for high-quality long video generation. The core innovation of our method lies in the denoising process: instead of applying a single noise level across all frames used in traditional video diffusion models [2, 15], we apply progressively increasing noise levels across the frames; correspondingly, we denoise and shift the frames in small intervals, instead of denoising and shift them all at once. We illustrate our method in Fig. 1. Such progressive noise levels and autoregressive video denoising benefit from larger overlaps between subsequent attention windows, smoother attention correspondence among frames with adjacent noise levels, and better propagation of information from the earlier to the later frames. When applying our variable length progressive noise schedule, our models can start or end the autoregressive generation at arbitrary video lengths. Our chunked frames and overlapped conditioning techniques prevent divergent results and chunk-to-chunk discontinuity. Together, our method can autoregressively generate long videos while maintaining the initial quality over time.

PA-VDM provides a range of benefits for the video generation community. It can be easily implemented by changing the noise scheduling and finetuning pre-trained video diffusion models without changing the original model architecture; this allows our method to be easily reproduced and combined with orthogonal methods, such as external memory modules [12] and multiple text prompts [11, 59]. While we choose to demonstrate PA-VDM on Diffusion Transformer (DiT)-based [3, 30, 32] models, PA-VDM is model agnostic and can be extended to UNet-based [15, 37] models. As shown in Sec. 4.2, our method can work training-free, if the model has been trained on varied noise levels [58]. Moreover, the additional inference computational cost of PA-VDM is minimal without sacrificing any generation quality, as opposed to previous works [12, 36, 49] that need to trade off quality for efficiency, making this approach more efficient for practical use in long video generation.

We compare our method to the baselines on a textconditioned 60-second (1440 frames) long video generation benchmark consisting of 40 real videos and their captions.

Our quantitative results demonstrate that our results have overall the best quality across various dimensions and are the best at maintaining these metrics over the entire 60-second duration. Qualitatively, our method substantially outperforms the baselines in terms of temporal consistency, motion dynamics, and maintaining quality over time. In human evaluation, our models are also favored over various baseline models. Our ablation studies demonstrate the effectiveness of our chunked frames and overlapped conditioning techniques at preventing cumulative error and temporal jittering, respectively. By applying our method to two base models and outperforming their respective baselines, we confirm its universal applicability to existing video diffusion models. We encourage readers to check out our project webpage for video results qualitatively comparing ours and the baselines. To facilitate future research, we also release our code based on Open-Sora [58]. We summarize our contribution as follows:

- 1. We propose a progressive noise level schedule, an autoregressive video denoising algorithm, and the chunked frames and overlapped conditioning techniques. Together, these enable high-quality long video generation building upon pre-trained video diffusion models.
- 2. We are the first to achieve 60-second long video generation with quality that are close to frontier models, when compared at the same resolution. On our 60-second long video generation benchmark, we achieve superior VBench and FVD scores, majority preference in human evaluations, and strong qualitative results. This marks a significant step forward in generating longer videos, a dimension that has not been explored by recent frontier video diffusion models [9, 22, 29, 31, 34, 55].
- 3. Our method benefits the video generation research community in many ways, including easy implementation and reproduction, training-free application, minimal additional inference cost, and universal applicability on video diffusion models.

### 2. Background

#### 2.1. Video Diffusion Models

Diffusion models [13, 40] are generative models that learn to generate samples from a data distribution q(x0) through an iterative denoising process. During training, data samples are first corrupted using the forward diffusion process q(xt|x0)

√

αtx0,(1 − αt)I) (1) xt =

q xt x0 = N(xt;

√

√

αtx0 +

1 − αtϵ (2)

where t ∈ [0,T) is the noise level or diffusion timestep, ϵ ∼ N(0,I) is the noise, and α1:T is the variance schedule. With those noisy data samples xt, diffusion models are trained to fit to the data distribution q(x0) by maximizing

the variational lower bound [20] of the log likelihood of x0, which can be simplified into a mean squared error loss [13]

L(θ) = ϵ − ϵθ(xt,t) 2 (3)

where t is uniform between 0 and T, ϵ ∼ N(0,I) and ϵθ is the noise predicted by the model with parameters θ.

At sampling time, we consider the sampling noise level schedule τ = {τ0,τ1,...,τS} , which is a monotonically increasing subset of t ∈ [0,T) of length S +1 [42]. Starting from xτ

∼ N(0,I),τS = T, the reverse denoising process is iteratively applied as

S

xτ,fθ(xt,t) (4)

|xτ

) = qσ xτ

pθ (xτ

i−1

i−1

i

where xˆ0 = fθ(xt,t) is the x0 predicted by the model and fθ(xt,t) is the DDIM [42] reverse process equation, which we omit for simplicity. This gives us a sequence of samples xT,xτ

,x0, and the last sample x0 is the clean output result.

,...,xτ

S−1

1

Latent video diffusion models [2, 15] are diffusion models that models latent representations of video data, consisting of F latent frames x0:F−1 = {x0,x1,...,xF−1}. The video latent frames are usually spatially and temporally [57] compressed through a VAE [20]. For simplicity, we refer to latent video diffusion models as video diffusion models and latent frames as frames. The same forward process, reverse process, and loss (Eqs. (1) to (4)) can be applied to model these video data by treating all the frames as one entity, ignoring the correlation among the frames. Recent video diffusion models [34, 58] have employed various diffusion model variants [25–27] to improve training and inference efficiency as well as output quality. Nevertheless, our method is compatible with any diffusion model variant as long as the model corrupts the data xt at the same noise levels t.

#### 2.2. Autoregressive Long Video Generation via Replacement

Video diffusion models can only generate short video clips, because they are only trained on videos with a limited length F due to GPU memory limit. When adapted to generating L > F latent frames at sampling time, their generation quality substantially degrades [36]. The straightforward solution is to autoregressively apply video diffusion models, generating each video clip while conditioning on the previous clip. In this paper, we refer to the F frames that the video diffusion model processes as the attention window.

Given E < F clean frames x00:E as condition, there are two methods for autoregressively applying video diffusion models. [1, 7, 58] place the clean condition frames x¯00:E−1 directly at the front of the attention window, directly replacing the sampled frames xτ

0:E−1 at each denoising step pθ x ¯00:E−1,xEτi:−F1−1|x¯00:E−1,xτ

i

E:F−1 (5)

i

[Figure 11]

[Figure 12]

Figure 2. Comparison of noise levels of a sequence of video frames when using the replacement without noise method (left) and ours (right).

We will refer to this method as the replacement-withoutnoise method.

[15, 43] additionally add noise to the condition frames

pθ x ¯τ0:iE−1−1,xτEi:−F1−1 x ¯τ

0:E−1,xτ

E:F−1 (6) where x¯τ

i

i

0:E−1 are the condition frames x¯00:E−1 noised via the forward process (Eqs. (1) and (2)). This maintains the same noise level distribution and training objective as regular video diffusion models. We will refer to this method as the replacement-with-noise method. Note that [15] proposes reconstruction guidance for the replacement-withnoise method but is not widely adopted.

i

Both the replacement-with-noise method and the replacement-without-noise method allow a video diffusion model to autoregressively generate video frames by conditioning on previous frames. We consider them as baselines in our experiments in Sec. 4.2.

See Appendix B for a detailed discussion of two parallel works [19, 38] that share a high-level idea similar to our work. Please refer to Appendix C for related works.

### 3. Progressive Autoregressive Video Diffusion Models

We consider long video generation with video diffusion models. As discussed in Sec. 2.2, existing video diffusion models can only generate short video clips up to a limited length F, and the replacement methods [7, 15, 58] suffer from various flaws. We describe a more natural formulation of autoregressive long video generation, which we call Progressive Autoregressive Video Diffusion Models (PA-VDM). We propose a per-frame progressively increasing noise schedule, which is inspired by [4]. During training, we finetune pre-trained video diffusion models to adapt to our noise schedule; during sampling, our models adopt such noise schedule and autoregressively generate video frames.

#### 3.1. Progressive Noise Levels and Autoregressive Generation

Conventional video diffusion methods assign a single noise level t to all the latent frames. Inspired by [4], we

Algorithm 1 Inference procedure of progressive autoregressive video diffusion models Require: Initial video latent frames x00:F−1 = {x00,x01,...,x0F−1}, maximum noise level T, number of inference

steps S, and attention window size F = S

- 1: τ0:S = {τ0,τ1,...,τS} = 0,

T S

,...,T ▷ Eq. (7), linear sampling noise level schedule

- 2: ϵ ∼ N(0,I)
- 3: xτ

1:S

0:F−1 =

√

ατ1:Sx00:F−1 + √1 − ατ1:Sϵ ▷ Eq. (2), add noise and set to progressive noise levels

- 4: for each autoregressive generation step i = 1,2,...,N do
- 5: x0:τ0:FS−−11 = x00,xτ

1

1 ,...,xτS−1

F−1 ∼ pθ xτ0:0:FS−−11 xτ

1:S

0:F−1 ▷ Eq. (8), one sampling step

- 6: xTF−1 ∼ N(0,I) ▷ Sample a new noisy frame
- 7: Append x00 to the list of clean frames
- 8: xτ

0:S

0:F−1 = xτ

1

1 ,...,xτS−1

F−2,xTF−1 ▷ Remove x00, shift frames forward, and append xTF−1

- 9: end for
- 10: return List of clean frames

adopt per-frame noise levels t0:F−1 = {t0,t1,...,tF−1} to the F latent frames in the attention window. In particular, we consider monotonically increasing noise levels for each frame, where earlier frames are less noisy and later frames are more noisy. In this work, we consider the linear sampling noise schedule with S sampling steps

T S

τ0:S = 0,

,

(S − 1)T S

2T S

,T (7)

,...,

which is monotonically increasing. Given a sampling noise schedule, instead of all the frames sharing a noise level and jointly going through the schedule as in conventional video diffusion models, each frame now goes through the schedule independently; at each step, the per-frame noise levels τ still maintain the progressively increasing pattern.

Since both the sampling noise schedule and our target per-frame noise levels are monotonically increasing, we can now set per-frame noise levels t0:F−1 to be an interpolation of the sampling noise schedule τ. Let us first consider the simple case of F = S, when our per-frame progressive noise levels can equal to either t = τ0:S−1 or t = τ1:S. At each sampling step, the video diffusion model takes τ0:S−1 as input and predicts τ1:S

1 ,...,xτS−2

F−2,xτS−1

1 ,...,xτS−1

F−2,xτ

##### pθ xτ

##### 0 ,xτ

F−1 xτ

##### 0 ,xτ

0

1

1

2

S

F−1

(8) We illustrate progressive noise levels when F = S in Fig. 2.

Now we construct our autoregressive generation algorithm for video latent frames with progressive noise levels. Notice that the input and output noise levels in Eq. (8), τ0:S−1 and τ1:S, only differ by τ0 = 0 and τS = T. We can simply transition the output frames back into the correct input noise levels by removing the clean frame x00 at the front, shifting the frame sequence forward by one frame, and appending a new noisy frame xTF−1 ∼ N(0,I) at back, as illustrated in Fig. 1. We describe the autoregressive generation

algorithm when F = S in Alg. 1. The algorithm requires a clean short video x00:F−1 as initialization and extends from it. We describe how to avoid this requirement in Sec. 3.2.

More generally, when F is a multiple of S, e.g. F = 90,S = 30, every set of F/S = 3 frames would always share the same noise level during denoising and be removed from the attention window together as they reach t = 0; when S is a multiple of F, e.g. F = 10,S = 30, the save, shift, and append operations for the sequence of frames (line 6, 7, 8 in Alg. 1) would happen once every S/F = 3 steps.

Note that, regardless of what the noise level a frame initially has, it always goes through the same noise level schedule τS:0 as in conventional diffusion models. Thus, for each individual frame, it is still modeled under the valid assumptions in diffusion model training [13, 25–27] and sampling [42]. We only diverge from the noise level assumption in conventional video diffusion models [15]: now, each frame is modeled independently instead of jointly with the whole sequence of frames, and the progressive autoregressive video diffusion model attends to frames with different noise levels t0:F−1 instead of the same noise level t. Thus, we can obtain our progressive autoregressive video diffusion models from pre-trained video diffusion models by adapting the model to the new noise level distribution through finetuning. This saves us from the highly demanding computation cost of video diffusion model pre-training [34, 55].

Intuitively, the benefit of our progressive video denoising process is that it gradually establishes correlation among consecutive latent frames. Given some existing video frames as conditioning, it is challenging for video diffusion models to produce temporally consistent extension frames from newly sampled noisy frames [36]. In contrast to the replacementwith-noise method [1, 15] where the frames are denoised together at the same noise level, our progressive video denoising encourages the later frames with higher uncertainty to follow the patterns of the earlier and more certain frames, fa-

cilitating modeling a smoother temporal transition and better preserving motion velocity. Compared to the replacementwithout-noise method where there is a large noise level gap between the clean condition frames x¯00:E−1 and the noisy frames xτ

E:F−1, our method provides smoother attention correspondence, where the difference between neighboring noise levels is only TS , as illustrated in Eq. (7) and Fig. 2.

i

#### 3.2. Variable Length

The above design only allows for autoregressive video extension given an initial video of length F. In addition, the noisy frames remaining in the attention window xτ

0:F−1 (line 8 of Alg. 1) are discarded after the end of the autoregressive inference, which can cause wasted computing resources and inaccurate handling of the ending of text prompt. To enable text-to-long-video generation without any starting condition frames and properly ending the generation without wasting computation, we extend the base design in Eq. (8) and Alg. 1 to add an initialization stage and a termination stage, where the model operates on variable attention window lengths from 1 to F − 1. During initialization, we simply disable the “removing x00” operation in line 8 of Alg. 1: starting from a noisy frame {xT0 }, we denoise and append to obtain {xτS−1

1:S

0 ,xT1 }; we repeat this by F − 1 times to obtain xτ

1 ,...,xτS−1

0:F−1 = xτ

F−2,xTF−1 , i.e. the input to line 5

1:S

1

of Alg. 1. During termination, we disable the “append xTF−1” operation in line 6 and 7 of Alg. 1: starting with F frames xτ

0 ,...,xτS−1

0:F−1 = xτ

F−2,xTF−1 , we denoise, save and remove to obtain x0:τ1:FS−−21 = xτ

1:S

1

0 ,...,xτS−1

F−2 ; we repeat this by F times to save and remove all the remaining frames in the attention window. We train the model accordingly on video latent frames with variable lengths ranging from 1 to F − 1, following the noise levels described above.

1

#### 3.3. Chunked Frames

3D VAEs [20, 34, 57, 58] usually encode and decode video latent frames chunk-by-chunk. In our early experiments, we find that naively implementing our method on latent video diffusion models, i.e. when all latent frames are given different noise levels and the attention window is shifted by one frame at a time, leads to serious cumulative error and the videos diverge quickly after a few seconds, as shown in Ablation 2 in Fig. 6. We resolve the problem by treating a chunk of latent frames as a whole: they are assigned with the same noise level, and are added and removed from the attention window together. In other words, for a 3D VAE chunk size of C latent frames, e.g. C = 5 as mentioned in Sec. 4, we shift the attention window by C frames every C sampling steps. Effectively, the C frames that belong to the same chunk always have the same noise level t and are added to or removed from the attention window together. Our ablation experiments shows that, for models using a 3D VAE, treating a chunk of frames as a whole effectively

prevents accumulated errors that would lead to divergence.

#### 3.4. Overlapped Conditioning

In our early experiments, naively implementing our method on video diffusion models results in temporal jittering. We hypothesize that this is because the clean frames x00:C−1 are immediately removed from the attention window; as the later frames cannot attend to the previous clean frames, it is hard for the model to denoise the later frames to be perfectly temporally consistent with the previous clean frames. In practice, we always keep a chunk of C clean frames by prepending it to the attention window. Our ablation study shows that overlapped conditioning helps resolving the frame-to-frame discontinuity issue.

Overlapped conditioning requires an additional inference cost at C/F (5/50 in our implementation) of the original cost. When using the same number of conditioning frames E and F, the replacement methods [7, 15, 58] and ours have the same inference efficiency. The key advantage of our method is that the large overlap of noisy frames enables the model to preserve the high-level information—such as motion—from prior frames. Thus, we only need a single chunk of C clean condition frames to propagate high-frequency details and prevent per-chunk temporal jittering. In contrast, the replacement methods need to balance the tradeoff between more overlap between video clips or better inference efficiency. In practice, their implementation [58] often use one chunk of frames as condition to save inference computation, but the limited overlap causes unnatural motion transition and abrupt scene changes across clips, as discussed in Sec. 4.2.

#### 3.5. Training

As described in Sec. 3.1, PA-VDM requires change in the noise level distribution. We finetune pre-trained video diffusion model to adapt to our progressive noise level distribution. Conventional diffusion model training [13, 26, 27] involves uniformly sampling a noise level t ∈ [0,T), adding noise to the samples x00:F−1 via the forward diffusion process (Eqs. (1) and (2)), and computing the loss (Eq. (3)). During the finetuning process for PA-VDM, we simply continue with the conventional video diffusion model training but with our per-frame progressive training noise levels t0:F−1. In our experiment, we observed that, similar to the sampling noise levels τ0:S in Eq. (7), training on a simple linear noise schedule yielded satisfactory results for all reported experiments. During training, the noise levels t is perturbated by a random shift δ to fully cover of the diffusion timestep range [0,T) [41]. δ = 0.4ϵ(ti − ti+1),ϵ ∼ N(0,I) is randomly sampled for each training iteration and remains constant for all t0:F−1 within that iteration.

Table 1. Quantitative comparison of our progressive autoregressive video generation (PA) and two baseline methods replacement-with-noise (RW) and replacement-without-noise (RN) on two base models (M and O), and other baselines StreamingT2V [12], Stable Video Diffusion (SVD) [1], and FIFO-Diffusion [19].

Subject Background Motion Dynamic Aesthetic Imaging Num Consistency ↑ Consistency ↑ Smoothness ↑ Degree ↑ Quality ↑ Quality ↑ Scenes ↓ FVD ↓

PA-M (ours) 0.7923 0.8964 0.9896 0.8000 0.4726 0.5927 1.75 358.020 RW-M 0.8001 0.8851 0.9836 0.3958 0.4123 0.5961 1.10 669.747

PA-O-base (ours) 0.7656 0.8880 0.9859 0.5625 0.4582 0.5033 2.04 548.117 RN-O-base 0.7406 0.8820 0.9873 0.5750 0.4034 0.4464 5.19 600.690

StreamingSVD 0.8172 0.8916 0.9929 0.65 0.4264 0.5566 1.08 440.272 SVD-XT 0.6102 0.8136 0.9724 0.9875 0.3019 0.4814 2.10 702.343 FIFO-OSP 0.7577 0.8990 0.9731 0.75 N/A 0.5675 18.32 975.459

### 4. Experiments

#### 4.1. Implementation

Our models and baseline models We implement our progressive autoregressive video diffusion models by finetuning from pre-trained models. Specifically, we use two video diffusion models based on the diffusion transformer architecture [3, 32]: Open-Sora v1.2 [58] (denoted as O) and a modified variant of Open-Sora (denoted as M in later experiments). Both models are latent video diffusion models [2], each utilizing a corresponding 3D VAE that encodes 17 (O) or 16 (M) raw video frames into 5 latent frames. O generates videos at 240×424 resolution 24 FPS with 30 sampling steps. M produces results at 176×320 resolution 24 FPS with 50 sampling steps. Based on O and M, we also implement two baseline autoregressive video generation methods, replacement-with-noise (denoted as RW) and replacementwithout-noise (denoted as RN) (Sec. 2.2), to compare with our proposed progressive autoregressive (denoted as PA) video generation method (Sec. 3).

We train M on our progressive noise levels, as discussed in Sec. 3.5. The resulting model can perform progressive autoregressive video generation, which we denote as PA-M. We also train M with the replacement-with-noise method, which we will denote as RW-M. Starting from the same pretrained weight of the base model, RW-M is trained for 3 times more training steps compared to PA-M.

O undergoes masked pre-training [58], where the masked latent frames x00:E−1 are clean without any added noise [58]. This allows the O base model to perform autoregressive video generation with the replacement-without-noise method. We denote this model as RN-O-base. Such training also allows O to learn that the noise levels t0:F−1 can be independent with respect to the latent frames and thus enables our progressive autoregressive video denoising sampling procedure (Alg. 1) to work training-free. We denote this model as

PA-O-base. Please refer to Appendix E for training details.

#### 4.2. Long video generation

The baseline methods are described in Appendix F.1. Metrics We consider 6 metrics in VBench [17]: subject consistency, background consistency, motion smoothness, dynamic degree, aesthetic quality, and imaging quality. We compute average metrics using VBench-long, where each metric is computed on 30 2-second clips for each 60-second video; for subject and background consistency, a clip-to-clip metric is considered in addition to the average metric over the clips. We also show how the metrics vary over time by plotting the metrics over the 30 2-second clips averaged over the 80 60-second videos.

Similar to [12], we also use the Adaptive Detector algorithm from PySceneDetect [35] to count the number of detected scene changes, where Num Scenes = 1 means that there is no scene change detected.

We also compute Fr´echet Video Distance (FVD) [47] to measure the overall quality of the generated videos compared to real videos. We adopt the improved implementation of FVD proposed in [8] using the VideoMAE-v2 [50] model. The FVD metric usually requires a large number of video samples in order to produce a reliable value. Since our testing set includes only 40 real videos and each model only generate 80 videos, naively computing FVD on them results in erroneous values such as -3.62e+64. Instead, we compute FVD on the 2-second clips of the long videos, so that we have 1495 real videos and 2400 generated videos.

Quantitative Results We present the average metrics for each model in Tab. 1. The metrics are averaged over all the videos that each model generates from our testing set described above. Our PA-M has the best results overall. Notably, it surpasses other methods in FVD by a substantial margin, illustrating that its results are the most realistic. It also achieves either the best or close-to-best in other met-

rics. Its replacement-with-noise counterpart, RW-M, suffers from poor Dynamic Degree and FVD, because its videos are mostly static. Our RW-O-base surpasses its replacementwithout-noise counterpart RN-O-base in all metrics except for being close at Dynamic Degree, while using the exact same model parameters without any finetuning. RN-O-base mainly suffers from a high number of scene changes.

In Fig. 3, we also illustrate the trend of metrics over the 1-minute duration of videos for each model. Our models M-PA and O-PA can best maintain the level of all metrics, while their replacement-method counterparts, M-RW and O-RN, both exhibit distinct reduction in dynamic degree, aesthetic quality, and imaging quality.

[Figure 13]

Figure 3. VBench [17] scores over the 60-second duration, which are computed on 30 2-second clips.

Qualitative Results We also show strength of our method with qualitative comparison results in Fig. 5. Both of our models demonstrate strong performance in terms of frame fidelity and motion realism (e.g. camera motion, wave motion, and running gestures) and outperforms other baselines. For more qualitative results, please refer to our project webpage.

[Figure 14]

Figure 4. Human evaluation results comparing long video methods on long-shot (L), motion (M), temporal consistency (C), and overall (O).

User study We conduct a human evaluation with 12 users to compare the generated videos from each method. As shown in Fig. 4, our PA-M is favored in each duel by a large margin.

#### 4.3. Ablation Study

We conduct ablation studies on the PA-M model to evaluate the impact of chunked frames (Sec. 3.3), and overlapped conditioning (Sec. 3.4). Qualitative comparison is shown in Fig. 6 and in the project webpage. In Ablation 1, we observe that the absence of clean frames in the input sequence prevents noisy frames from attending to previous clean frames, resulting in poor performance over a long duration. This also causes frame-to-frame discontinuity, which is more noticeable in the project webpage. In Ablation 2, not decoding the video chunk-by-chunk leads to severe cumulative errors, causing the video to diverge after only a few seconds.

See Appendix H for additional ablation study on variable length and the number of sampling steps S.

### 5. Conclusion

In this work, we target long video generation, a fundamental challenge of current video diffusion models. We show that they can be naturally adapted to become progressive autoregressive video diffusion models without changing the architectures. With our progressive noise levels and the autoregressive video denoising process (Sec. 3.1), we achieve state-of-the-art results on 60-second long video generation. Since our method does not require model architecture changes, it can be seamlessly combined with orthogonal works, paving the way for generating longer videos at higher quality, long-term dependency, and controllability.

### 6. Acknowledgments

This research was supported in part by NSF award IIS2107224 and ONR award N000142312124.

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

OMPA--bPA-RW-MS-T2VSVDRN-O-bFIFO

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

- Figure 5. Qualitative comparison of PA-M (ours), RW-M, PA-O-base (ours), RN-O-base, StreamingSVD from StreamingT2V [12], SVD-XT from Stable Video Diffusion [1], and FIFO-Diffusion [19]. Frames are evenly sampled from 1 minute long generated video, i.e. at 10, 20, 30, 40, 50, and 60 seconds. Our models can autoregressively generate 60-second, 1440-frame videos without quality degradation.

FullAblation1Ablation2

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

- Figure 6. Qualitative comparison for ablation study. Full represents for our full solution based on PA-M, Ablation 1 is with chunked frames but without overlapped conditioning. Ablation 2 is without both techniques. The frames are evenly sampled from 16-second generated videos.

### References

- [1] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2, 3, 4, 6, 8, 12, 13, 14, 15
- [2] Andreas Blattmann, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook Kim, Sanja Fidler, and Karsten Kreis. Align your latents: High-resolution video synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22563–22575, 2023. 2, 3, 6
- [3] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024. 1, 2, 6
- [4] Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion,

2024. 3, 12

- [5] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Seine: Short-to-long video diffusion model for generative transition and prediction. In The Twelfth International Conference on Learning Representations, 2023. 12
- [6] Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. Reinforcement learning for fine-tuning text-to-image diffusion models. Advances in Neural Information Processing Systems, 36, 2024. 12, 13
- [7] Kaifeng Gao, Jiaxin Shi, Hanwang Zhang, Chunping Wang, and Jun Xiao. ViD-GPT: introducing GPT-style autoregressive generation in video diffusion models, 2024. 2, 3, 5, 12
- [8] Songwei Ge, Aniruddha Mahapatra, Gaurav Parmar, Jun-Yan Zhu, and Jia-Bin Huang. On the content bias in fr´echet video distance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7277–7288,

2024. 6

- [9] Genmo. Mochi 1 preview. https://www.genmo.ai/ blog, 2024. Accessed: 2024-11-13. 1, 2
- [10] Yuwei Guo, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, Maneesh Agrawala, Dahua Lin, and Bo Dai. Animatediff: Animate your personalized text-to-image diffusion models without specific tuning. arXiv preprint arXiv:2307.04725, 2023. 12
- [11] Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation, 2025. 2
- [12] Roberto Henschel, Levon Khachatryan, Daniil Hayrapetyan, Hayk Poghosyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. StreamingT2V: Consistent, dynamic, and extendable long video generation from text, 2024. 2, 6, 8, 12, 13, 14, 15

- [13] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020. 2, 3, 4, 5, 13
- [14] Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P Kingma, Ben Poole, Mohammad Norouzi, David J Fleet, et al. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303, 2022. 12
- [15] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J. Fleet. Video diffusion models, 2022. 2, 3, 4, 5, 12, 13
- [16] Li Hu. Animate anyone: Consistent and controllable imageto-video synthesis for character animation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8153–8163, 2024. 12
- [17] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 6, 7
- [18] Xuan Ju, Yiming Gao, Zhaoyang Zhang, Ziyang Yuan, Xintao Wang, Ailing Zeng, Yu Xiong, Qiang Xu, and Ying Shan. Miradata: A large-scale video dataset with long durations and structured captions, 2024. 14
- [19] Jihwan Kim, Junoh Kang, Jinyoung Choi, and Bohyung Han. Fifo-diffusion: Generating infinite videos from text without training. arXiv preprint arXiv:2405.11473, 2024. 3, 6, 8, 12, 13, 14, 15
- [20] Diederik P Kingma and Max Welling. Auto-encoding variational bayes, 2013. 3, 5
- [21] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, Kathrina Wu, Qin Lin, Junkun Yuan, Yanxin Long, Aladdin Wang, Andong Wang, Changlin Li, Duojun Huang, Fang Yang, Hao Tan, Hongmei Wang, Jacob Song, Jiawang Bai, Jianbing Wu, Jinbao Xue, Joey Wang, Kai Wang, Mengyang Liu, Pengyu Li, Shuai Li, Weiyan Wang, Wenqing Yu, Xinchi Deng, Yang Li, Yi Chen, Yutao Cui, Yuanbo Peng, Zhentao Yu, Zhiyu He, Zhiyong Xu, Zixiang Zhou, Zunnan Xu, Yangyu Tao, Qinglin Lu, Songtao Liu, Dax Zhou, Hongfa Wang, Yong Yang, Di Wang, Yuhong Liu, Jie Jiang, and Caesar Zhong. Hunyuanvideo: A systematic framework for large video generative models, 2025. 1
- [22] Kuaishou. Kling. https://www.klingai.com/, 2024. Accessed: 2024-11-13. 1, 2
- [23] PKU-Yuan Lab and Tuzhan AI etc. Open-sora-plan, 2024. 13, 14
- [24] Zongyu Lin, Wei Liu, Chen Chen, Jiasen Lu, Wenze Hu, Tsu-Jui Fu, Jesse Allardice, Zhengfeng Lai, Liangchen Song, Bowen Zhang, Cha Chen, Yiran Fei, Yifan Jiang, Lezhi Li, Yizhou Sun, Kai-Wei Chang, and Yinfei Yang. Stiv: Scalable text and image conditioned video generation, 2024. 1
- [25] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling, 2023. 3, 4

- [26] Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow, 2022. 5
- [27] Xingchao Liu, Xiwen Zhang, Jianzhu Ma, Jian Peng, et al. Instaflow: One step is enough for high-quality diffusion-based text-to-image generation. In The Twelfth International Conference on Learning Representations, 2023. 3, 4, 5
- [28] Yu Lu, Yuanzhi Liang, Linchao Zhu, and Yi Yang. Freelong: Training-free long video generation with spectralblend temporal attention. arXiv preprint arXiv:2407.19918, 2024. 12
- [29] Luma. Dream machine. https://lumalabs.ai/ dream-machine, 2024. Accessed: 2024-11-13. 1, 2
- [30] Xin Ma, Yaohui Wang, Gengyun Jia, Xinyuan Chen, Ziwei Liu, Yuan-Fang Li, Cunjian Chen, and Yu Qiao. Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048, 2024. 2
- [31] Runway ML. Gen-3 alpha. https://runwayml.com/ research/introducing-gen-3-alpha, 2024. Accessed: 2024-11-13. 1, 2
- [32] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205,

2023. 1, 2, 6, 13, 14

- [33] Ethan Perez, Florian Strub, Harm De Vries, Vincent Dumoulin, and Aaron Courville. Film: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI conference on artificial intelligence, 2018. 13
- [34] Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, ChihYao Ma, Ching-Yao Chuang, David Yan, Dhruv Choudhary, Dingkang Wang, Geet Sethi, Guan Pang, Haoyu Ma, Ishan Misra, Ji Hou, Jialiang Wang, Kiran Jagadeesh, Kunpeng Li, Luxin Zhang, Mannat Singh, Mary Williamson, Matt Le, Matthew Yu, Mitesh Kumar Singh, Peizhao Zhang, Peter Vajda, Quentin Duval, Rohit Girdhar, Roshan Sumbaly, Sai Saketh Rambhatla, Sam Tsai, Samaneh Azadi, Samyak Datta, Sanyuan Chen, Sean Bell, Sharadh Ramaswamy, Shelly Sheynin, Siddharth Bhattacharya, Simran Motwani, Tao Xu, Tianhe Li, Tingbo Hou, Wei-Ning Hsu, Xi Yin, Xiaoliang Dai, Yaniv Taigman, Yaqiao Luo, Yen-Cheng Liu, Yi-Chiao Wu, Yue Zhao, Yuval Kirstain, Zecheng He, Zijian He, Albert Pumarola, Ali Thabet, Artsiom Sanakoyeu, Arun Mallya, Baishan Guo, Boris Araya, Breena Kerr, Carleigh Wood, Ce Liu, Cen Peng, Dimitry Vengertsev, Edgar Schonfeld, Elliot Blanchard, Felix Juefei-Xu, Fraylie Nord, Jeff Liang, John Hoffman, Jonas Kohler, Kaolin Fire, Karthik Sivakumar, Lawrence Chen, Licheng Yu, Luya Gao, Markos Georgopoulos, Rashel Moritz, Sara K. Sampson, Shikai Li, Simone Parmeggiani, Steve Fine, Tara Fowler, Vladan Petrovic, and Yuming Du. Movie gen: A cast of media foundation models,

2024. 1, 2, 3, 4, 5

- [35] PySceneDetect. PySceneDetect. https : / / www . scenedetect.com/. Accessed: 2024-10-10. 6
- [36] Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, and Ziwei Liu. FreeNoise: tuning-free longer video diffusion via noise rescheduling, 2024. 2, 3, 4,

12

- [37] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234–241. Springer, 2015. 2
- [38] David Ruhe, Jonathan Heek, Tim Salimans, and Emiel Hoogeboom. Rolling diffusion models. arXiv preprint arXiv:2402.09470, 2024. 3, 12
- [39] Team Seawead, Ceyuan Yang, Zhijie Lin, Yang Zhao, Shanchuan Lin, Zhibei Ma, Haoyuan Guo, Hao Chen, Lu Qi, Sen Wang, Feng Cheng, Feilong Zuo Xuejiao Zeng, Ziyan Yang, Fangyuan Kong, Zhiwu Qing, Fei Xiao, Meng Wei, Tuyen Hoang, Siyu Zhang, Peihao Zhu, Qi Zhao, Jiangqiao Yan, Liangke Gui, Sheng Bi, Jiashi Li, Yuxi Ren, Rui Wang, Huixia Li, Xuefeng Xiao, Shu Liu, Feng Ling, Heng Zhang, Houmin Wei, Huafeng Kuang, Jerry Duncan, Junda Zhang, Junru Zheng, Li Sun, Manlin Zhang, Renfei Sun, Xiaobin Zhuang, Xiaojie Li, Xin Xia, Xuyan Chi, Yanghua Peng, Yuping Wang, Yuxuan Wang, Zhongkai Zhao, Zhuo Chen, Zuquan Song, Zhenheng Yang, Jiashi Feng, Jianchao Yang, and Lu Jiang. Seaweed-7b: Cost-effective training of video generation foundation model, 2025. 1
- [40] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015. 2
- [41] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502,

2020. 5

- [42] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021. 3, 4
- [43] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020. 2, 3
- [44] K Soomro. UCF101: a dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402,

2012. 14

- [45] Linrui Tian, Qi Wang, Bang Zhang, and Liefeng Bo. Emo: Emote portrait alive-generating expressive portrait videos with audio2video diffusion model under weak conditions. arXiv preprint arXiv:2402.17485, 2024. 12
- [46] Ye Tian, Ling Yang, Haotian Yang, Yuan Gao, Yufan Deng, Jingmin Chen, Xintao Wang, Zhaochen Yu, Xin Tao, Pengfei Wan, et al. Videotetris: Towards compositional text-to-video generation. arXiv preprint arXiv:2406.04277, 2024. 12
- [47] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018. 6
- [48] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems. Curran Associates, Inc.,

2017. 1, 13

- [49] Fu-Yun Wang, Wenshuo Chen, Guanglu Song, Han-Jia Ye, Yu Liu, and Hongsheng Li. Gen-L-Video: multi-text to long video generation via temporal co-denoising, 2023. 2, 12
- [50] Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. Videomae v2: Scaling video masked autoencoders with dual masking. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14549–14560, 2023. 6
- [51] WanTeam, :, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced largescale video generative models, 2025. 1
- [52] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023. 14
- [53] Jay Zhangjie Wu, Xiuyu Li, Difei Gao, Zhen Dong, Jinbin Bai, Aishani Singh, Xiaoyu Xiang, Youzeng Li, Zuwei Huang, Yuanxi Sun, Rui He, Feng Hu, Junhua Hu, Hai Huang, Hanyu Zhu, Xu Cheng, Jie Tang, Mike Zheng Shou, Kurt Keutzer, and Forrest Iandola. Cvpr 2023 text guided video editing competition, 2023. 14
- [54] Desai Xie, Jiahao Li, Hao Tan, Xin Sun, Zhixin Shu, Yi Zhou, Sai Bi, S¨oren Pirk, and Arie E Kaufman. Carve3d: Improving multi-view reconstruction consistency for diffusion models with rl finetuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6369–6379, 2024. 13
- [55] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024. 1, 2, 4
- [56] Shengming Yin, Chenfei Wu, Huan Yang, Jianfeng Wang, Xiaodong Wang, Minheng Ni, Zhengyuan Yang, Linjie Li, Shuguang Liu, Fan Yang, et al. Nuwa-xl: Diffusion over diffusion for extremely long video generation. arXiv preprint arXiv:2303.12346, 2023. 12
- [57] Lijun Yu, Jos´e Lezama, Nitesh B. Gundavarapu, Luca Versari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, Alexander G. Hauptmann, Boqing Gong, Ming-Hsuan Yang, Irfan Essa, David A. Ross, and Lu Jiang. Language model beats diffusion – tokenizer is key to visual generation, 2024. 3, 5

- [58] Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-Sora: democratizing efficient video production for all, 2024. 2, 3, 5, 6, 13, 14
- [59] Yupeng Zhou, Daquan Zhou, Ming-Ming Cheng, Jiashi Feng, and Qibin Hou. Storydiffusion: Consistent self-attention for long-range image and video generation. In Advances in Neural Information Processing Systems, pages 110315–

110340. Curran Associates, Inc., 2024. 2

- [60] Shenhao Zhu, Junming Leo Chen, Zuozhuo Dai, Yinghui Xu, Xun Cao, Yao Yao, Hao Zhu, and Siyu Zhu. Champ: Controllable and consistent human image animation with 3d parametric guidance. arXiv preprint arXiv:2403.14781, 2024. 12
- [61] Shaobin Zhuang, Kunchang Li, Xinyuan Chen, Yaohui Wang, Ziwei Liu, Yu Qiao, and Yali Wang. Vlogger: Make your dream a vlog. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8806–8817,

2024. 12

## Progressive Autoregressive Video Diffusion Models Supplementary Material

### A. Summary

In this Appendix, we cover parallel works in Appendix B, related works in Appendix C, limitations and discussions in Appendix D, training details in Appendix E, evaluation details in Appendix F, additional qualitative results in Appendix G, and additional ablation study in Appendix H.

### B. Parallel Works

The core idea of our PA-VDM is to 1. assign progressively increasing noise levels to the F frames in the attention window and 2. autoregressively apply the video diffusion model on progressively noised frames to generate long videos. The first part is inspired by Diffusion Forcing [4], which proposes to assign independent per-frame noise levels to some frames rather than a single noise level. We began developing our work right after July 1st, 2024, when Diffusion Forcing [4] was released on arXiv. The first version of our preprint was submitted to arXiv on October 10th, 2024. During this period, our work was developed independently, without the knowledge of two papers, Rolling Diffusion [38] and FIFO-Diffusion [19]. While Rolling Diffusion, FIFODiffusion, and PA-VDM arrive at a similar high-level idea in parallel, the three methods have different focuses, naming and framing of the idea, implementation details, experimental setups, and final result quality.

Compared to [6, 38], PA-VDM:

- 1. shows that it is possible to adapt a pre-trained video diffusion model to the progressive noise level schedule through finetuning, thus avoiding the otherwise immensely expensive computation cost of pre-training video diffusion models. [38] is trained from scratch and [19] is trainingfree.
- 2. achieves state-of-the-art 60-second long video generation at a quality comparable to frontier video diffusion models, demonstrating much longer video length and better quality than [19, 38].

We provide comparisons between our models (PA-M and PAO) and [19] on our 60-second long video generation benchmark (Appendix F.2) in Sec. 4.2 and Tab. 1. Our method achieves substantially better qualitative and quantitative results than [19]. [19] also requires doubled inference cost, while our method only requires additional inference cost that is a fraction of the original cost (10% for PA-M and 16.66% for PA-O). We do not compare PA-VDM with [38] as there is no released code and it does not support text-conditioned open-domain generation.

### C. Related Works

The field of long video generation has faced significant challenges due to the computational complexity and resource constraints associated with training models on longer videos. As a result, most existing text-to-video diffusion models [1, 10, 14, 15] have been limited to generating fixed-size video clips, which leads to noticeable degradation in quality when attempting to generate longer videos. Recent works are proposed to address these challenges through innovative approaches that either extend existing models or introduce novel architectures and fusion methods.

Freenoise [36] utilizes sliding window temporal attention to ensure smooth transitions between video clips but falls short in maintaining global consistency across long video sequences. Gen-L-video [49], on the other hand, decomposes long videos into multiple short segments, decodes them in parallel using short video generation models, and later applies an optimization step to align the overlapping regions for continuity. FreeLong [28] introduces a sophisticated approach which balances the frequency distribution of long video features in different frequency during the denoising process. Vid-GPT [7] introduces GPT-style autoregressive causal generation for long videos.

More recently, Short-to-Long (S2L) approaches are proposed, where correlated short videos are firstly generated and then smoothly transit in-between to form coherent long videos. StreamingT2V [12] adopts this strategy by introducing the conditional attention and appearance preservation modules to capture content information from previous frames, ensuring consistency with the starting frames. It further enhances the visual coherence by blending shared noisy frames in overlapping regions, similar to the approach used by SEINE [5]. NUWA-XL [56] leverages a hierarchical diffusion model to generate long videos using a coarseto-fine approach, progressing from sparse key frames to denser intermediate frames. However, it has only been evaluated on a cartoon video dataset rather than natural videos. VideoTetris [46] introduces decomposing prompts temporally and leveraging a spatio-temporal composing module for compositional video generation.

Another line of research focuses on controllable video generation [16, 45, 60, 61] and has proposed solutions for long video generation using overlapped window frames. These approaches condition diffusion models using both frames from previous windows and signals from the current window. While these methods demonstrate promising results in maintaining consistent appearances and motions, they are limited to their specific application domains which relies

heavily on strong conditional inputs.

### D. Limitations and discussions

A limitation of our method is the demand of a well-trained base video diffusion model. Similar to the replacement methods [15, 58] and other approaches like StreamingT2V [12], our method autoregressively applies a video diffusion model to generate long videos. Such autoregressive video generation poses huge challenge on the base video diffusion model. Some slight errors remaining in the “clean” frames x0 may not be noticeable in a single video clip; however, in the autoregressive scenario, these error can be carried onto later frames, resulting in quality degradation. Further more, as the video diffusion model is only trained on denoising latent frames of real video data, it may poorly handle such distribution shift towards the generated erroneous frames [6, 54], resulting in more severe quality drop. This means that even after finetuning on our progressive noise levels, our method could still generate videos with some degree of quality degradation close to the ending, if the base video diffusion model is not well trained. Among the qualitative videos generated by our PA-M, in some cases, the video quality slightly degrades in the last 10 seconds.

Another limitation of our method is the subtle temporal flickering happening about every second in our PA-M results. It is caused by a flaw in the backbone video diffusion model M’s 3D VAE, as evident by the presence of such flickering in both PA-M and RW-M results while no such flickering is present in the PA-O results.

There are many promising future directions to extend this work. We only train on progressively increasing noise levels to reduce the space of noise levels for easier convergence. If sufficient computing resources are available, training on fully random, per-frame independent noise levels would enable a single model for various tasks with arbitrary lengths, including video extension, connection, temporal super-resolution. Another promising future application of the long video generation ability of our models is to use them as world simulators, useful for tasks in robotics and 3D vision. Being able to generate long videos without quality degradation is an substantial step towards this direction.

### E. Training details

M is pre-trained on captioned image and video datasets, containing 1 million videos and 2.3 billion images. These data are licensed and have been filtered to remove low-quality content. We train PA-M on video clips of 16,32,...,176 raw frames that correspond to F = 5,10,...,55 latent frames. The F = 55 attention window length is derived by setting F = S + 5, where S = 50 is the number of sampling steps in M (S = 30 in O) and 5 is the length of an additional chunk of latent frames, as de-

scribed in Secs. 3.3 and 3.4. The shorter latent frame lengths F = 5,10,...,50 are used for the variable length training,

- as discussed in Sec. 3.2. RW-M is trained on videos of 64 frames that corresponds to F = 20 frames.

- E.1. Modification to the base model

To implement progressive autoregressive video diffusion models on top of their pre-trained foundation video diffusion models, we do not need to modify the base model architectures. Instead, we only need to modify the model’s forward, training, and inference procedures. In the training and inference procedures, we replace the single noise level t ∈ [0,T) from regular diffusion model training [13, 15] with our per-frame noise level t0:F−1 and τ0:′ S−1 (Secs. 3.1 and 3.5). To accommodate this change, we only need to make a single modification to the the noise level embedding computation in the model’s forward procedure. While the regular timestep only has the batch size dimension B, our progressive timesteps has two dimensions B,F. We first flatten them into the batch dimension of size B × F, pass it to the timesteps embedding module, unflatten the two dimensions, and finally broadcast the timestep embedding to the same shape of the frames so they can be combined through either addition, concatenation, modulation, or crossattention [32, 33, 48].

- F. Evaluation details F.1. Baselines

As discussed in Sec. 4, using our base models, we implement two baseline autoregressive video generation methods on three models, which are denoted as RW-M, RN-Obase, and RN-O. We also compare to Stable Video Diffusion (SVD) [1] and StreamingT2V [12] model families. Specifically, we consider the SVD-XT model from SVD, a imageto-video model that generates a short video clip of 25 frames

- at 576x1024 resolution given an conditioning image. We apply it autoregressively, using the last image of the previous clip as the condition for generating a new clip. This is equivalent to the replacement-without-noise method except that it only conditions on a single frame rather than a chunk of 17 frames as RN-O. We also consider the StreamingSVD model from StreamingT2V, a image-to-long-video generation model that uses SVD as the base model [12]; its autoregressive video generation is enabled by training additional modules that connect to the base model via crossattention. Similar to our progressive autoregressive video diffusion models, StreamingSVD can autoregressively generate long videos at 720x1280 resolution with arbitrary lengths, which we set to 1440 frames. We also compare to a concurrent work FIFO-Diffusion [19] implemented on OpenSora-Plan v1.0.0 [23], denoted as FIFO-OSP. It generates at 256x256 resolution with a context window of 65 latent

frames. See Appendix B for a discussion on [19] and other concurrent works. See Appendix F for details on our testing set, quantitative metrics, and traditional video quality evaluation.

FIFO-OSP FIFO-Diffusion [19] is a parallel work that adopts a similar high-level idea as our method on pre-trained video diffusion models without any fine-tuning (see more discussion in Appendix B). It provides training-free implementations on VideoCrafter2 and Open-Sora-Plan v1.1.0 [23]. We choose its Open-Sora-Plan implementation since our method is also implemented on DiT-base [32] models, M and Open-Sora (O) [58]. Open-Sora-Plan v1.1.0 generate videos at 512x512 resolution. Since there is no distributed inference support in the released code of FIFO-Diffusion, we adopt Open-Sora-Plan v1.0.0 in our reproduced FIFODiffusion results in order to saving computation costs by inferencing at the 256x256 resolution instead of the original 512x512 resolution.

- F.2. Testing set

Text prompts and real videos Our testing set consists of 40 text prompts and the corresponding real videos, sampled from Sora [58] demo videos, MiraData [18], UCF-101 [44], and LOVEU [52, 53]. For each text prompt, we generate two videos with 1440 frames, 60 seconds long at 24 FPS, resulting in a total of 80 videos. We use these 80 videos from each model for both quantitative and qualitative results, unless specified otherwise. Due to computation resource limitations of sampling 1-minute long videos, we only obtained partial results from M-PA, StreamingSVD and FIFO-OSP, including 48, 40, 40 videos from 24, 40, 40 text prompts respectively. This testing set measures the zero-shot long video generation ability of the models, since none of them are specifically trained on any of the above datasets.

Real video initialization Since our focus is on long video generation, we focus on the video extension capability of the models rather than the text-to-short-video generation capability. Thus, we use the initial frames of the videos as the condition for all models, similar to the setting in [12]. M, O [58], StreamingSVD [12], SVD-XT [1], and FIFOOSP [19, 23] use 16, 17, 1, 1, and 65 frames from the real video as the initial condition. Note that our PA-M and PA-O only require one chunk of frames (16 and 17 for M and O respectively), which is substantially less than the full context window of 65 frames required by FIFO-Diffusion [19]. This advantage is obtained from our variable-length autoregressive generation design as described in Sec. 3.2.

- G. Additional Qualitative Results We provide additional quailtative results in Fig. 7.

S FVD↓

50 358.20 100 339.59 150 399.91

Table 2. Ablation on the number of sampling steps S of the PA-M model.

### H. Additional Ablation Study

In our project webpage, we show an ablation study on our Variable Length design (Sec. 3.2). We compare Variable Length inference results of PA-M models trained with and without Variable Length. Without Variable Length training, the second video shows temporal jittering and abrupt scene change at the 1st and 59th seconds. This is because the model is not trained to generate the first/last chunk of latent frames to be consistent with the prior chunks. With Variable Length training, the first video avoids the jittering and abrupt scene change at the 1st and 59th seconds, and the video is temporally smooth. Furthermore, Variable Length inference enables the model to generate precisely 1440 frames, whereas without this technique the model would need to discard the noisy chunks remaining in the context window, which correspond to the 1441-1584th frames, when it reaches the 1440th frame. Being able to stop the autoregressive video denoising at a precise ending frame allows our model to generate a proper ending to the video, e.g. the woman exits the camera view in the first video, which is not possible without the Variable Length technique.

Additionally, we ablate the number of sampling steps S of the PA-M. Note that our progressive video denoising can work with arbitrary S; when the chunked frames technique is used, S only needs to be divisible by C. We compute FVD scores in the same way as described in Sec. 4.2. As shown in Tab. 2, further increasing S from 50 to 100 provides marginal benefits despite doubling the inference compute cost, while increasing S to 150 leads to slightly worse results.

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

OMPA--bPA-RN-O-bS-T2VSVD

[Figure 81]

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

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

Figure 7. Qualitative comparison of PA-M (ours), RW-M, PA-O-base (ours), RN-O-base, StreamingSVD from StreamingT2V [12], SVD-XT from Stable Video Diffusion [1], and FIFO-Diffusion [19]. Frames are evenly sampled from 1 minute long generated video, i.e. at 10, 20, 30, 40, 50, and 60 seconds. Our models can autoregressively generate 60-second, 1440-frame videos without quality degradation.

