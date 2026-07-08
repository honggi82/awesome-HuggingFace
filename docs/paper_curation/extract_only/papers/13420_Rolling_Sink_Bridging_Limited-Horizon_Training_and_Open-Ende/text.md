## Rolling Sink: Bridging Limited-Horizon Training and Open-Ended Testing in Autoregressive Video Diffusion

##### Haodong Li1 Shaoteng Liu2 Zhe Lin2 Manmohan Chandraker1§

###### 1UC San Diego 2Adobe Research {hal211,mkchandraker}@ucsd.edu {shaotengl,zlin}@adobe.com

# arXiv:2602.07775v6[cs.CV]3May2026

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

𝑡 = 1s 𝑡 = 5s 𝑡 = 15s 𝑡 = 30s

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

𝑡 = 90s 𝑡 = 150s 𝑡 = 225s 𝑡 = 300s

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

𝑡 = 1s 𝑡 = 5s 𝑡 = 15s 𝑡 = 30s

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

𝑡 = 90s 𝑡 = 150s 𝑡 = 225s 𝑡 = 300s

Fig. 1: Rolling Sink unlocks open-ended AR video generation. Despite a 5s training duration, Rolling Sink effectively scales the AR video synthesis to minutes long during testing, e.g., 5-minute and 30-minute (please see Fig. S28, S29 in our Supp1).

Abstract. Recently, autoregressive (AR) video diffusion models have achieved remarkable performance. However, due to their limited training durations, a train-test gap emerges when testing at longer horizons, leading to rapid visual degradations. Following Self Forcing, which studies the train-test gap within the training duration, this work studies the train-test gap beyond the training duration, i.e., the gap between the limited horizons during training and open-ended horizons during testing. Since open-ended testing can extend beyond any finite training window, and long-video training is computationally expensive, we pursue a training-free solution to bridge this gap. To explore a training-free

𝑡 = 1s 𝑡 = 5s 𝑡 = 15s 𝑡 = 30s

𝑡 = 90s 𝑡 = 150s 𝑡 = 225s 𝑡 = 300s

§ Corresponding author. 1 Supp: Supplementary Material.

𝑡 = 1s 𝑡 = 5s 𝑡 = 15s 𝑡 = 30s

𝑡 = 90s 𝑡 = 150s 𝑡 = 225s 𝑡 = 300s

solution, we conduct a systematic analysis of AR cache maintenance. These insights lead to Rolling Sink. Built on Self Forcing (trained on only 5s clips), Rolling Sink effectively scales the AR video synthesis to ultra-long durations (e.g., 5-30 minutes at 16 FPS) at test time, with consistent subjects, stable colors, coherent structures, and smooth motions. As demonstrated by extensive experiments, Rolling Sink achieves superior long-horizon visual fidelity and temporal consistency compared to SOTA baselines. Project page: https://rolling-sink.github.io/.

Keywords: Autoregressive Video Diffusion · Open-Ended Video Generation · Autoregressive Cache Maintenance

### 1 Introduction

Generating a long video (e.g., a movie) typically requires a “multi-shot” input, i.e., a sequence of prompts. Each shot typically corresponds to a single prompt, and can vary from few seconds to minutes, even hours long. For instance, Steve McQueen’s Hunger [68] begins with a classic 16.5 minutes dialogue shot between Bobby Sands and the priest2. Stanley Kubrick’s The Shining [50] also features a minutelong tracking shot following Danny’s tricycle ride through the Overlook Hotel corridors, which builds tension in the audience. This motivates an “open-ended” video generation setting, where the video length is not fixed in advance and the model is expected to continue generating for arbitrary horizons when deployed (i.e., at test time).

[Figure 17]

[Figure 18]

𝑡 = 1s

𝑡 = 1s

[Figure 19]

[Figure 20]

Accurate Latent

Drifted Latent

w/ Rolling Sink w/o Rolling Sink (Self Forcing)

𝑡 = 5s

𝑡 = 5s

[Figure 21]

[Figure 22]

𝑡 = 15s

𝑡 = 15s

[Figure 23]

[Figure 24]

𝑡 = 30s

𝑡 = 30s

[Figure 25]

[Figure 26]

AR Drift

𝑡 = 60s 𝑡 = 60s

Self Forcing

Video Manifold

Fig. 2: Bridging the gap between limitedhorizon training and open-ended testing. Self Forcing [39] studies the train-test gap when testing within the training window (i.e., 5s at 16 FPS), while we extend the focus to the train-test gap that emerges when testing beyond the training window.

Though large video diffusion models [21,47,70,79,91] have achieved remarkable performance, they usually rely on bidirectional attentions in DiTs [71] and denoise all frames simultaneously, making them incompatible with such “openended” setting. In contrast, autoregressive (AR) video diffusion models archi-

2 https://www.youtube.com/watch?v=aycGYu_8Hhw. From 1:00 to 17:30.

tecturally enables open-ended video generation by continuously predicting the next-frame3 conditioned on previous ones.

However, AR models are typically trained on limited and fixed durations, e.g., 5s at 16 FPS in Self Forcing [39], which can hardly cover the wide range of video lengths (e.g., from seconds to minutes or hours) during testing. When extrapolating to long horizons, especially beyond the training duration, these models often suffer from rapid visual degradation, exhibiting inconsistent subjects, oversaturated colors, vanished dynamics, and collapsed structures, as illustrated in the first two rows of each case in Fig. 7 and Supp’s Fig. S9-S18.

Such AR drift is commonly attributed to error accumulation. In this work, we further interpret it through the lens of “exposure bias” [6,54,57,69,76,80,112], i.e., a mismatch between limited training horizons and open-ended generation at test time. During training, AR video diffusion models are supervised on videos with fixed, limited durations. When testing within the training window, the predictions can be considered accurate. However, when testing on durations longer than the training window, where the model hasn’t been sufficiently regularized, the predictions may gradually drift as the horizon grows. As illustrated in Fig. 2, following Self Forcing [39], which studies the train-test gap within its training duration (Sec. 3.1), this work studies the train-test gap that emerges when testing beyond the training duration. In other words: bridging the gap between limitedhorizon training and open-ended testing.

Indeed, training on longer videos can mitigate this mismatch. But fundamentally speaking, as long as the training is conducted on finite-length clips, the open-ended testing can always exceed the training window. As the rollout length grows beyond this window, long-horizon drift can still occur. Moreover, scaling the training horizon to very long durations is computationally expensive. Also, in practice, most AR video diffusion models proposed after Self Forcing [39] are trained on not only limited but short clips, e.g., 5s at 16 FPS [63,66,72], 10s [111],

- 1 minute [60,99], and 100s [16]. These considerations motivate a training-free approach to bridge the limited-horizon training and open-ended testing.

The goal is to constantly reproduce the impressive video synthesis quality, exhibited when testing within the training duration, over ultra-long horizons. Since the prompt embedding stays fixed throughout AR video synthesis, and the initial noise for each block is always drawn from the same Gaussian distribution, the context (i.e., cache) is the major factor of long-horizon AR drift. Thus, for maintaining “drift-free” during open-ended testing, the AR cache should stay consistent with its within-duration behavior/characteristic. Derived from a systematic analysis (Sec. 3.2) of how the AR cache is maintained when testing in long horizons, we propose Rolling Sink, a training-free approach for bridging the gap between limited-horizon training and open-ended testing. Built on Self Forcing [39], which is trained on only 5s videos, Rolling Sink is able to synthesize ultra-long videos (e.g., 5-30 minutes) with consistent ID and structures, stable colors, and smooth dynamics. Rolling Sink also preserves the streaming efficiency

3 In this work, each AR step generates a “block” of frames following [39,105], please allow us to use “frame” here and also in Sec. 3.1 for better readability.

of Self Forcing, since it uses the same, strictly bounded total cache size and the same few-step denoising per AR generation step.

Extensive experiments are conducted to comprehensively assess Rolling Sink’s performance: ① qualitative comparisons (Fig. 7 and Supp’s Fig. S9-S18), and ② quantitative evaluations using VBench-Long [40, 41, 113] on both 1-minute (Tab. 1) and 5-minute (Tab. 2) AR video synthesis across multiple dimensions. As illustrated in Fig. 7 and Supp’s Fig. S9-S18, when synthesizing long videos, prior SOTA methods [39, 99] often suffer from over-saturated colors, distorted subjects, and inconsistent surroundings. In contrast, our method excels in producing much more stable and consistent videos, with superior visual fidelity over long horizons. Moreover, as shown in Tab. 1 and Tab. 2, Rolling Sink attains the highest (best) scores on most evaluation dimensions defined in VBench-Long, and consequently achieves the lowest (best) averaged rank over all dimensions. In summary, our key contributions are:

- – We characterize the long-horizon drift in AR video diffusion as the exposure bias from a train-test horizon mismatch, and provide a systematic analysis of cache mechanisms towards a training-free solution.
- – We introduce Rolling Sink, which effectively scales the AR video synthesis to ultra-long durations at test time without additional training and under a strictly bounded cache, despite a 5s training duration.
- – Rolling Sink achieves SOTA performance in long-horizon (e.g., 1-minute, 5-minute) AR video synthesis, as demonstrated by extensive experiments.

- 2 Related Works Please see Sec. B in the Supp.
- 3 Methodology

#### 3.1 Preliminaries: Autoregressive Video Diffusion Models

Autoregressive (AR) video diffusion models continuously generate the nextframe3 conditioned on prior ones, while each AR generation step is modeled

- as a denoising diffusion process. Specifically, the joint distribution of a video

contains N frames y[0,N) = y0,y1,··· ,yN−1 can be factorized into a cumulative product of N conditional distributions:

pθ y[0,N) | c =

N−1

pθ(yi | y[0,i),c), (1)

i=0

where c is the user’s prompt.

Following Eq. 1, each AR generation step (i.e., conditional distribution) is modeled by a denoising diffusion model Gθ [61,64]. We term the set of denoising timesteps as {t0,t1,...,tT}, where t0 = 0 and tT = 1000. At each denoising

timestep tj, the diffusion model Gθ first denoises the noisy frame ytij to a clean one yˆt

i (i.e., yˆ0i) [83]. Note that here we use the hat sign (ˆ·) for distinguishing the intermediate prediction of the clean sample yˆt

0

i produced at timestep tj (j > 1) from the final prediction of the clean sample yt

0

i (produced at timestep t1).

0

After that, ytij−1 is obtained by applying the forward noising process Ψ(·) to the intermediate clean sample yˆt

at a lower noise level corresponding to timestep tj−1. Thus, the conditional distribution of each AR generation step can be formulated as:

i , injecting Gaussian noise ϵt

0

j−1

(yt

i ), (2) where:

pθ(yi | y[0,i),c) = fθ,t

1 ◦ fθ,t

2 ◦ ... ◦ fθ,t

T

T

(ytij) = ytij−1 = Ψ y ˆt

fθ,t

i ,ϵt

,tj−1

0

j−1

j

= Ψ Gθ(ytij,tj,y[0,i)),ϵt

(3)

,tj−1 , ϵt

j−1

#### ,yt

i ∼ N(0,I).

T

j−1

During training, major techniques for building the AR cache are teacher forcing (TF) [18,38,45,111], diffusion forcing (DF) [12,13,25,72,84,86,105], and self forcing (SF) [16,37,39,66,99,102–104]. In TF, the conditional distribution is: pθ(yi | ygt[0,i,t)=0,c), all cached preceding context are clean ground-truth (GT) frames. In DF: pθ(yi | ygt[0,i,t)⩾0,c), where the preceding context are noised GT frames with randomly sampled noise levels. No matter in TF or DF, the cache is drawn from GT distribution during training but from self-generated distribution

- at test time, leading to a train–test gap. In contrast, SF draws the cache from the model’s own generated frames

during training4: pθ(yi | yt[0=0,i),c). Thus, the cache distribution at training and testing are better matched. While achieving remarkable performance when test-

ing within the training duration, SOTA SF-styled methods [39,99] still fall short when synthesizing long videos, especially beyond their training durations.

#### 3.2 Systematic Analysis & Rolling Sink

Self Forcing (Fig. 3, a). Self Forcing [39] is the pioneering method in SFstyled AR video synthesis. Specifically, it’s trained on sequences of 21 latent frames (corresponding to 81 frames after VAE [91] decoding). At each AR step, it generates a block of 3 latent frames, xi[k] = z3i+k, k ∈ {0,1,2}, where z denotes a latent frame and x denotes a block (21 latent frames correspond to 7 blocks). During training, all prior self-generated blocks are cached as context:

pθ xi | x[0,i) , i ∈ {0,1,··· ,K}, (4)

where K = 6. Also, for clarity, here we omit the superscript t=0 denoting timestep t = 0 (i.e., the clean sample) and the condition c denoting the user’s prompt. During testing, Self Forcing can endlessly generate the next-block:

pθ xi | x[i−K,i) , i ∈ {0,1,··· ,∞}. (5)

- 4 By default, all yi are “predicted”, unless marked as ygti .

|[Figure 27]|
|---|

Recent Block

Sink in forward order Sink in reversed order

𝑥

𝑥

𝑥

𝑥

Cache Slot (Temporal Indices)

|[Figure 28]|
|---|

Current Block

𝑥

𝑥 𝑥

𝑥 𝑥

𝑥 𝑥

𝑥

|[Figure 29]|
|---|

| |
|---|

Evicted Block

[Figure 30]

[Figure 31]

[Figure 32]

𝑥 𝑥

𝑥 𝑥 𝑥

𝑥

𝑥 𝑥

𝑥

𝑥 𝑥

𝑥

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥

𝑥 𝑥 𝑥 𝑥

𝑥

𝑥

𝑥 𝑥

𝑥

𝑥

𝑥 𝑥

𝑥

Temporal 𝑥 𝑥 Reindex

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

[Figure 44]

𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥

𝑥

𝑥 𝑥

𝑥

𝑥

𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥 𝑥

𝑥

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

𝑥 𝑥 𝑥

𝑥 𝑥 𝑥 𝑥 𝑥 𝑥

𝑥

𝑥

𝑥

AR Generation Step

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

𝑥 𝑥

𝑥

𝑥

𝑥

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

𝑥 𝑥 𝑥 𝑥 𝑥 𝑥

𝑥

𝑥

𝑥

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

𝑥 𝑥 𝑥 𝑥

𝑥 𝑥 𝑥

𝑥

𝑥

𝑥 𝑥 𝑥 𝑥

Self Forcing (a)

w/ Attention Sink (b)

w/ Sliding Indices (c)

w/ Sliding Semantic (Rolling Sink) (d)

- Fig. 3: Overview of our analysis and the proposed Rolling Sink. ○a The caching mechanism of Self Forcing [39], the total cache capacity K is strictly bounded for streaming efficiency. ○b We first apply Attention Sink (i.e., pinning the first S blocks as sink blocks where both the time indices and semantics are static), and analyze the effect

of different sink ratios (KS ). ○c Sliding Indices: Treating the time indices as a global axis i ∈ [0, ∞), at each AR step i, we shift sink blocks’ time indices as a fixed-length

(i.e., S) sliding window on this axis. ○d Sliding Semantics: Ideally, the sink blocks’ semantic content should also slide along a drift-free, global video manifold that lasts endlessly. Since finite-length training cannot naturally realize this, we approximate the true semantic sliding by rolling the sink content (i.e., at each AR step, we update the sink blocks’ semantic content with a rolling segment from the within-duration history). Finally, we propose ○d and name it Rolling Sink. For clarity, here we set K = 3 and S = 2. Please see Sec. 3.2 for more technical details.

Key Issue: Cache Maintenance. Following Sec. 1, our goal is to reproduce the high video quality observed when testing within the training duration over ultra-long horizons. Since the global prompt embedding (encoded by umT5 [15]) is constant during AR video synthesis and the initial noise of each block is also sampled from the same distribution ϵi ∼ N(0,I), the remaining key factor that causes the AR drift is the conditioning context (i.e., cache). Therefore, the key issue of bridging the limited-horizon training and open-ended testing is: how to maintain the AR cache consistent with its within-duration behavior?

Concretely, building on Self Forcing [39], we aim to keep the AR cache consistent with its within-duration behavior under a strictly bounded capacity K. This within-duration behavior includes these characteristics: ① Minimally drifted: all cached blocks should be drift-free (i.e., no over-saturated colors, no collapsed structures, etc.). ② Sliding in both indices5 and semantics: the cached blocks’ time indices should be assigned from a fixed-length sliding window on a global axis i ∈ [0,∞) right before the current block (i.e., sliding indices); similarly, the cached blocks’ semantic content should also be updated as a moving slice from a global video manifold that lasts endlessly (i.e., sliding semantics).

When testing within 5s (i.e., the training duration), the conditioned AR cache naturally meets the above requirements. But when synthesizing longer videos, the latents written into the cache are potentially corrupted, which will bias subsequent predictions and may further amplify the AR drift. Below we conduct thorough analysis over the above within-duration characteristics of the

5 The time indices are embedded using rotary positional embeddings (RoPE) [85].

Video Quality vs. Sink Size (on 1min Videos)

| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | |w/|Att|enti|on|ink| | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | |w/|Slid|ing|Ind|ices| | | | | | | |
| | | | | | | | | | |w/|Slid|ing|Sem|an|tic|(i.e|., R|olli|ngS|ink|)|
| | | | | | | | | | | | | | | | | | | | | | |

0.70

0.68

VideoQuality

VideoQuality

0.66

0.64

0.62

0.60

0% 17% 33% 50% 67% 83%

(Sink Size) / (Total Cache Size)

Video Quality vs. Sink Size (on 5min Videos)

| | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | |w/ w/|Atte Slid|nti ing|on Ind|ink ices| | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | |w/|Slid|ing|Sem|an|tic|(i.e|., R|olli|ngS|ink|)|
| | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | |

0.70

0.65

0.60

0.55

0.50

0.45

0% 17% 33% 50% 67% 83%

(Sink Size) / (Total Cache Size)

- Fig. 4: Evaluation results during the systematic analysis on both 1-minute (left) and 5-minute (right) AR video synthesis. The video quality metric is the averaged score across all dimensions evaluated by VBench-Long [40,41,113]. As illustrated, the video quality is consistently improved during our systematic analysis and the derived Rolling

Sink yields the best performance (particularly when KS = 83%). Please see Supp’s Sec. C for the specific numerical results of all dimensions.

AR cache. Among those requirements, keeping the AR cache minimally drifted is the basis for reproducing the within-duration video quality over ultra-long horizons, because the sliding of semantics hardly makes sense when the cached latents themselves no longer preserve valid and faithful content. Thus, keeping the cache minimally drifted is studied first. After that, we analyze the effect of sliding indices and sliding semantics.

Attention Sink (Fig. 3, b). The latents synthesized within the training duration are “the least drifted”. Thus, analogous to the idea of Attention Sink [96], which has been widely adopted in both large language models (LLMs) [22, 24,43,44,87] and AR video synthesis [42,65,81,99,102], we start by pinning a static prefix of early self-generated latents inside the AR cache:

pθ xi | Cat x[0,S),x[i−(K−S),i) , (6)

where S ∈ [0,K] denotes sink size and i ∈ {0,1,··· ,∞}.

Fig. 5 shows visual comparisons across different sink sizes6. Consistent with prior works that study attention sinks in AR video diffusion [39,65,81,99,102], we also find that enlarging the sink reliably stabilizes color. However, artifacts (i.e., AR drift) still remain, most notably intermittent frame flickering (typically every several seconds; please see the second row of each example in Supp’s Fig. S26). Notably, in 1-minute rollouts, two flickers (usually take place at ≈ 33s and ≈ 50s) are particularly prominent, after which the generation tends to collapse into repetition (please see the middle parts of Supp’s Fig. S21–S24).

6 Note that we do not test the degenerate case when S = K (i.e., the sink occupies the entire cache), because at least one recent block is needed to maintain a basic local smoothness; otherwise, the generation will remain in a persistently flickering state.

[Figure 69]

[Figure 70]

0% 17% 33% 50% 67% 83% (Sink Size) / (Total Cache Size)

###### Fig. 5: Visual comparisons across various sink sizes. Larger sink sizes stabilize colors. But noticeable AR drift still persists, e.g., frame flickers. Here we set t = 60s.

Following Sec. 1, we continue to interpret these artifacts (shown on the right of Fig. 5) as a weaker form of AR drift, compared to the more severe artifacts shown on the left of Fig. 5. And this weaker form of AR drift is still caused by the insufficient match of the AR cache characteristics between testing within the training duration and beyond. Such drift suggests that keeping AR cache minimally drifted is only part of the solution and further requirements should be considered, e.g., sliding in time indices and semantics.

[Figure 71]

[Figure 72]

[Figure 73]

⬅ Ini al frame ➡

[Figure 74]

[Figure 75]

[Figure 76]

⬆ Sliding Indices & Sliding Seman c (i.e., Rolling Sink)

[Figure 77]

[Figure 78]

[Figure 79]

⬆ Sliding Indices & Sta c Seman c

[Figure 80]

[Figure 81]

[Figure 82]

⬆ Sta c Indices & Sta c Seman c

[Figure 83]

[Figure 84]

[Figure 85]

#### Sliding Indices (Fig. 3, c).

Next, we analyze the effect of sliding indices. In Fig. 3 (b), the time indices of sink blocks are fixed. Considering the time indices of the synthesized (latent) video frames as a linearly growing global axis i ∈ [0,∞), we here shift the time indices of sink blocks as a sliding window on this global time axis right before the indices of recent and current blocks. Specifically, we use xji to denote the block xi embedded with time index j:

[Figure 86]

Fig. 6: Visual comparisons of sliding indices and sliding semantics (when KS = 83%). Incorporating sliding indices and then sliding semantics consistently mitigates the artifacts (or AR drift). Following Fig. 5, the left, middle, and right frames are sampled at 59.8s, 60.0s, and 60.2s.

[Figure 87]

[Figure 88]

xji = RoPE(xi,j), xji[k] = RoPE(z3i+k, 3j + k), k ∈ {0,1,2}.

(7)

Note that if not explicitly marked, j ≡ i. Following Eq. 6, with sliding indices introduced, the conditional distribution is:

pθ xi | Cat x[[0i−,SK,i) −(K−S)),x[i−(K−S),i) , x[[0i−,SK,i) −(K−S))[l] = xil−K+l, l ∈ {0,1,··· ,S − 1}.

(8)

As shown in Fig. 6 (3rd row vs. 4th row or 6th row vs. 7th row), introducing sliding indices further reduces AR drift, most noticeably by mitigating flicker. However, noticeable AR drift still persists, manifesting as inconsistencies.

Sliding Semantics (Fig. 3, d). We here further analyze the effect of sliding semantics. As discussed in Fig. 3 and Sec. 3.2 (2nd part), not only the sink blocks’ time indices, their semantic content should also correspond to a moving slice of a minimally drifted, global video manifold that lasts endlessly. Since finitelength training cannot naturally realize this, we approximate this characteristic by periodically rolling the semantic content of sink blocks (synthesized within the training duration) alternatively between forward and reversed orders. That is, at each AR step, we update the sink blocks’ semantic content as a rolling segment drawn from the within-duration history. Following Eq. 6 and Eq. 8, with sliding semantics introduced, the conditional distribution is:

pθ xi | Cat Roll(x[0,K))[i−K,i−(K−S)),x[i−(K−S),i) , (9) where

Roll x[0,K) [i−K,i−(K−S)) = {Roll x[0,K) [i − K],··· , Roll x[0,K) [i − (K − S) − 1]},

(10)

and Roll(·) denotes the rolling operation. Specifically:

 

l K

xll mod K, when

mod 2 = 0

, (11)

Roll x[0,K) [l] =

l K



x˜l(K−1)−(l mod K), when

mod 2 = 1

where l ∈ {0,1,··· ,∞}, and x˜ji denotes the reversed form of block xji:

x˜ji[k] = RoPE(z3i+(2−k), 3j + k), k ∈ {0,1,2}; (12)

Different from Eq. 6 and Eq. 8, the rolling operation is applied over the whole set of (minimally drifted) within-duration blocks x[0,K), rather than fixing the sink to only the first S blocks x[0,S). At each AR step, Eq. 9 conditions on a rolling segment of S blocks over K within-duration blocks. The derived method is therefore named: Rolling Sink. As illustrated in Fig. 6 (2nd row vs. 3rd row or 5th row vs. 6th row), such rolling operation (i.e., sliding semantics) further mitigates the AR drift, noticeably illustrated as improved consistencies. And empirically, the enhancement on subject consistencies is more pronounced.

#### Quantitative Results during Analysis. During our analysis towards a training-

free solution, we also quantitatively conduct corresponding evaluations using VBench-Long [40,41,113], to assess the performance gains across each analysis step over different sink sizes6. As reported in Fig. 4, the evaluation results on both 1-minute and 5-minute AR video synthesis demonstrate that the synthesized videos gradually yield higher quality scores (over various sink sizes) during our systematic analysis. Though we can never close the gap between limitedhorizon training and open-ended testing when training on finite-length clips, the results in Fig. 4 support that our analysis effectively bridges this gap to a much closer state. Eventually, we set KS = 83% in the derived Rolling Sink.

More Discussions about Our Analysis. Following Sec. 1, our goal is to bridge the gap between limited-horizon training and open-ended testing. As discussed in Sec. 3.2 (2nd part), this gap primarily manifests as a mismatch in the behavior of AR cache when testing within and beyond the training duration. Accordingly, we study how to keep the AR cache consistent with its within-duration behavior when extrapolating over long horizons (Sec. 3.2, 3rd-5th parts).

We here emphasize that the specific designs in each step are intentionally simple and standard, the goal is simply to meet or approximate the properties discussed in Sec. 3.2 (2nd part). Moreover, these properties should not be viewed as an exhaustive characterization of the actual within-duration behavior of the AR cache. Due to limited-horizon training, we can never fully close this mismatch (i.e., a residual of it can still remain), and additional cache maintenance requirements or more advanced methods may further improve the open-ended synthesis at test time. We therefore view Rolling Sink as a simple baseline that satisfies several necessary cache properties for mitigating the long-horizon AR drift, and we hope it motivates future works toward more complete solutions.

### 4 Experiments

#### 4.1 Experimental Settings

Implementation Details. Rolling Sink is implemented on top of Self Forcing [39], which builds upon CausVid [103–105] and Wan [91]. Please see Sec. F in our Supp for the discussions of why Rolling Sink is developed on Self Forcing rather than other works like LongLive [99]. The cache (i.e., clean visual tokens of prior self-generated blocks) is conditioned by concatenating with the tokens of the current block to form the keys and values in the self-attentions of DiTs [71]. Whereas the queries come solely from the tokens of the current block. As discussed in Sec. 1, to preserve the same streaming efficiency as in Self Forcing, the total cache capacity is strictly bounded (i.e., K = 6) and each AR step is modeled by a 4-step video diffusion sampler following Self Forcing and CausVid. Moreover, as discussed in Sec. 3.2 (6th part) and illustrated in Fig. 4, we set S = 5 (i.e., KS = 83%) in the following comparisons of the proposed Rolling Sink with SOTA AR video synthesis baselines [39,99].

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

Self Forcing

= 1s = 5s = 15s = 30s

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

LongLive

= 1s = 5s = 15s = 30s

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

###### Ours

= 1s = 5s = 15s = 30s

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

= 90s = 150s = 225s = 300s

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

Self Forcing

= 1s = 5s = 15s = 30s

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

LongLive

= 1s = 5s = 15s = 30s

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

###### Ours

= 1s = 5s = 15s = 30s

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

= 90s = 150s = 225s = 300s

- Fig. 7: Qualitative comparisons of Rolling Sink with SOTA AR video synthesis baselines. When extrapolating beyond the training horizon, SOTA baselines often exhibit rapid AR drift, leading to noticeable visual degradation (e.g., over-saturated colors, collapsed structures, etc.). In contrast, Rolling Sink substantially reduces the AR drift, preserving stable identities and scene structure while maintaining coherent motions over long horizons. More qualitative results are provided in Supp’s Fig. S9-S18.

[Figure 121]

- Table 1: Quantitative comparison of Rolling Sink with SOTA baselines on 1-minute AR video synthesis using VBench-Long. The best results are bolded and the second-best are underlined. Rolling Sink achieves the best performance on most dimensions and thus attains the lowest (best) average rank. Dimension names are abbreviated to save space. Please see Tab. S11 in our Supp for the legend.

Dimension Self Forcing LongLive Rolling Sink (Ours)

sub_con ↑ 0.9679 0.9668 0.9858 bg_con ↑ 0.9653 0.9588 0.9694 aes_qual ↑ 0.5916 0.5850 0.6308 img_qual ↑ 0.6980 0.6519 0.6968 obj_cls ↑ 0.8680 0.9780 1.0000 multi_obj ↑ 0.3639 0.5802 0.6998 col ↑ 0.6433 0.7712 0.8023 spa_rel ↑ 0.7121 0.9683 1.0000 scn ↑ 0.1079 0.2540 0.2159 temp_sty ↑ 0.2220 0.2398 0.2503 ovrl_con ↑ 0.1991 0.2160 0.2316 hum_act ↑ 0.6886 0.8857 0.7800 temp_flick ↑ 0.9763 0.9643 0.9816 mot_smooth ↑ 0.9814 0.9730 0.9865 dyn_deg ↑ 0.4857 0.7592 0.7469 app_sty ↑ 0.2099 0.2018 0.1891

Avg. Rank ↓ 2.4375 2.1875 1.3750

- Table 2: Quantitative comparison on 5-minute AR video synthesis using VBench-Long. Consistent with Tab. 1, Rolling Sink continuously achieves the strongest overall performance, obtaining the best scores on most dimensions. Notably, the superiority over prior methods becomes more pronounced when testing at 5 minutes, highlighting Rolling Sink’s long-horizon video synthesis ability.

Dimension Self Forcing LongLive Rolling Sink (Ours)

sub_con ↑ 0.9424 0.9393 0.9804 bg_con ↑ 0.9610 0.9427 0.9629 aes_qual ↑ 0.4289 0.5718 0.6296 img_qual ↑ 0.5701 0.6431 0.6987 obj_cls ↑ 0.3339 0.9665 1.0000 multi_obj ↑ 0.1427 0.6998 0.7284 col ↑ 0.6105 0.7302 0.7883 spa_rel ↑ 0.3570 0.9697 1.0000 scn ↑ 0.0430 0.2079 0.2616 temp_sty ↑ 0.1132 0.2435 0.2533 ovrl_con ↑ 0.1139 0.2150 0.2310 hum_act ↑ 0.2710 0.9548 0.8710 temp_flick ↑ 0.9820 0.9687 0.9832 mot_smooth ↑ 0.9865 0.9766 0.9859 dyn_deg ↑ 0.2419 0.7379 0.6411 app_sty ↑ 0.2053 0.2086 0.1891

Avg. Rank ↓ 2.6250 2.0625 1.3125

Evaluation Benchmark & Metrics. In this work, we adopt VBench-Long [40, 41, 113] as the primary quantitative benchmark for evaluating Rolling Sink’s performance and the performance gains of different steps during the systematic analysis. VBench-Long is a long-video evaluation benchmark released as part of VBench++ [41], extending the original VBench [40] on long-horizon video generations while maintaining the same fine-grained evaluation philosophy (i.e., decomposing the “video quality” into multiple diagnostic dimensions, each measured by one or multiple expert models that are massively pretrained).

Prior SOTA Baselines. We compare Rolling Sink against two well-recognized, open-sourced, and SOTA AR video diffusion baselines: Self Forcing [39] and LongLive [99]. In our main experiments (Fig. 7, Fig. S9-S18 in our Supp, and Tab. 1, 2), to ensure all methods share the same training duration (i.e., 5s at 16 FPS) for a fair comparison, LongLive’s LoRA weights (further trained on 1 minute videos) are not loaded (i.e., w/o LoRA). The qualitative and quantitative comparisons between LongLive (w/ LoRA) and our method are reported in our Supp’s Fig. S19, S20 and Tab. S3, S4 (please also see Supp’s Sec. E).

#### 4.2 Qualitative Comparisons

The qualitative comparisons between Rolling Sink and SOTA AR video synthesis baselines are reported in Fig. 7. Please also check Fig. S9-S18 in our Supp for additional qualitative comparisons. When extrapolating beyond the training horizon, baseline methods typically accumulate AR drift quickly, which manifests as noticeable visual degradation like over-saturated colors and collapsed structures. In contrast, the proposed Rolling Sink substantially suppresses such AR drift over long horizons, preserving both subject identity and scene geometry while maintaining coherent motions.

#### 4.3 Quantitative Comparisons

The quantitative comparisons between Rolling Sink and SOTA baselines are reported in Tab. 1 (1-minute) and Tab. 2 (5-minute). The corresponding radar charts are shown in Fig. 8 for more intuitive presentations. On both settings, the proposed Rolling Sink achieves the best average rank and obtains the top scores on most dimensions, reflecting the reduced drift, improved visual quality, and more stable AR rollouts when testing beyond the training window.

While testing on VBench-Long, the clip_length is set to 2.0 for 1-minute setting and 10.0 for 5-minute. We randomly sample 10 prompts per dimension (the prompt lists are provided in Supp’s Sec. H) to form the evaluation suite. Originally, each dimension contains 70-100 prompts. This prompt suite is only sampled once, then fixed and reused across all quantitative evaluations (Tab. 1, 2 and Supp’s Tab. S3-S10). All these quantitative experiments (including both inference and evaluation) take about 8 weeks on 16 NVIDIA A40 GPUs.

Quantitative Comparison (1min)

Self Forcing LongLive s

Subject Consistency Background

Human

|Dynamic Degree<br><br>Action|Consistency<br><br>Aesthetic Quality<br><br>0.1<br><br>Ours|
|---|---|
|Temporal<br><br>Overall Consistency<br><br>0.4<br><br>0.8<br><br>1.0|Color<br><br>Spatial|

Motion Smoothness

Imaging Quality

Temporal Flickering

Object Class

Appearance Style

Multiple Objects

Style

Relationship Scene

Quantitative Comparison (5min)

Self Forcing LongLive s

Subject Consistency Background

Human

|Dynamic Degree<br><br>Action|Consistency<br><br>Aesthetic Quality<br><br>0.1<br><br>Ours|
|---|---|
|Temporal<br><br>Overall Consistency<br><br>0.4<br><br>0.8<br><br>1.0|Color<br><br>Spatial<br><br>|

Motion Smoothness

Imaging Quality

Temporal Flickering

Object Class

Appearance Style

Multiple Objects

Style

Relationship Scene

- Fig. 8: Radar charts of quantitative comparisons on 1-minute and 5-minute AR video synthesis. Rolling Sink achieves the highest scores on most VBench-Long dimensions. Notably, though Rolling Sink is built on top of Self Forcing [39] and requires no additional training, it yields substantial performance gains.

### 5 Summary

In this paper, we study the long-horizon drift of AR video diffusion and attribute it to an exposure bias between the limited-horizon training and open-ended testing. Building on a systematic analysis of AR cache maintenance, we propose Rolling Sink, a training-free method that aims to keep the AR cache consistent with its within-duration behavior. As a result, Rolling Sink effectively scales the AR video synthesis to ultra-long durations (e.g., 5-30 minutes, despite the limited

- 5s training duration) while maintaining stable identities/colors/structures and smooth dynamics, without sacrificing the efficiency. Extensive experiments validate that our method achieves superior long-horizon visual fidelity and temporal consistency compared to SOTA baselines. Limitations. Rolling Sink primarily targets single-shot, long video synthesis under a fixed prompt. However, in more general long video generation scenarios (e.g., movies), multiple shots are needed to continuously introduce new semantics (based on new prompts) over time rather than faithfully maintaining and extrapolating existing content. For instance, the buildings around the walking woman (Supp’s Fig. S12, bottom) are better continuously updated with new semantics, rather than staying consistent with earlier synthesized content. Future Works. The gap between limited-horizon training and open-ended testing also exists in multi-shot AR video synthesis. A natural future direction is extending our drift-mitigation principle into multi-shot settings, to enable coherent and smooth transitions, while continuously preserving the long-horizon stability and high visual fidelity beyond the limited training durations.

## Supplementary Material of Rolling Sink: Bridging Limited-Horizon Training and Open-Ended Testing in Autoregressive Video Diffusion

- Table S3: Quantitative comparison with LongLive (w/ LoRA) on 1-minute AR video synthesis. The evaluation results of Self Forcing and LongLive (w/o

- LoRA) are also borrowed from Tab. 1. Despite the much shorter training duration, Rolling Sink still achieves lower (better) average rank than LongLive (w/ LoRA). The best results are bolded and the second-best are underlined. LL: LongLive.

Dimension Self Forcing LL(w/o LoRA) LL(w/ LoRA) Ours Training Duration 5s 5s 1min 5s

sub_con ↑ 0.9679 0.9668 0.9840 0.9858 bg_con ↑ 0.9653 0.9588 0.9650 0.9694 aes_qual ↑ 0.5916 0.5850 0.6256 0.6308 img_qual ↑ 0.6980 0.6519 0.6947 0.6968 obj_cls ↑ 0.8680 0.9780 1.0000 1.0000 multi_obj ↑ 0.3639 0.5802 0.8864 0.6998 col ↑ 0.6433 0.7712 0.9866 0.8023 spa_rel ↑ 0.7121 0.9683 1.0000 1.0000 scn ↑ 0.1079 0.2540 0.1587 0.2159 temp_sty ↑ 0.2220 0.2398 0.2329 0.2503 ovrl_con ↑ 0.1991 0.2160 0.2321 0.2316 hum_act ↑ 0.6886 0.8857 0.8057 0.7800 temp_flick ↑ 0.9763 0.9643 0.9818 0.9816 mot_smooth ↑ 0.9814 0.9730 0.9848 0.9865 dyn_deg ↑ 0.4857 0.7592 0.5500 0.7469 app_sty ↑ 0.2099 0.2018 0.1877 0.1891

Average Rank ↓ 3.2500 2.8750 2.0625 1.6875

- Table S4: Quantitative comparison with LongLive (w/ LoRA) on 5-minute AR video synthesis. The evaluation results of Self Forcing and LongLive (w/o

- LoRA) are also borrowed from Tab. 2. Despite the much shorter training duration, Rolling Sink still achieves lower (better) average rank than LongLive (w/ LoRA).

Dimension Self Forcing LL(w/o LoRA) LL(w/ LoRA) Ours Training Duration 5s 5s 1min 5s

sub_con ↑ 0.9424 0.9393 0.9691 0.9804 bg_con ↑ 0.9610 0.9427 0.9601 0.9629 aes_qual ↑ 0.4289 0.5718 0.6370 0.6296 img_qual ↑ 0.5701 0.6431 0.6978 0.6987 obj_cls ↑ 0.3339 0.9665 1.0000 1.0000 multi_obj ↑ 0.1427 0.6998 0.7690 0.7284 col ↑ 0.6105 0.7302 0.8280 0.7883 spa_rel ↑ 0.3570 0.9697 1.0000 1.0000 scn ↑ 0.0430 0.2079 0.2007 0.2616 temp_sty ↑ 0.1132 0.2435 0.2457 0.2533 ovrl_con ↑ 0.1139 0.2150 0.2266 0.2310 hum_act ↑ 0.2710 0.9548 0.8613 0.8710 temp_flick ↑ 0.9820 0.9687 0.9835 0.9832 mot_smooth ↑ 0.9865 0.9766 0.9846 0.9859 dyn_deg ↑ 0.2419 0.7379 0.5968 0.6411 app_sty ↑ 0.2053 0.2086 0.1854 0.1891

Average Rank ↓ 3.4375 2.8125 2.0625 1.5625

###### Table S5: Quantitative results on 1-minute AR video synthesis during our analysis (w/ Attention Sink) using VBench-Long, across various sink sizes. The best results are bolded and the second-best are underlined.

Dimension 0% 17% 33% 50% 67% 83%

sub_con ↑ 0.9679 0.9870 0.9876 0.9903 0.9905 0.9898 bg_con ↑ 0.9653 0.9661 0.9691 0.9693 0.9692 0.9762 aes_qual ↑ 0.5916 0.6121 0.6187 0.6209 0.6385 0.6246 img_qual ↑ 0.6980 0.7004 0.7012 0.6931 0.6869 0.6913 obj_cls ↑ 0.8680 0.9291 0.9850 1.0000 1.0000 1.0000 multi_obj ↑ 0.3639 0.5884 0.6991 0.8032 0.7821 0.7000 col ↑ 0.6433 0.6836 0.7909 0.7671 0.8193 0.8732 spa_rel ↑ 0.7121 0.9178 0.9564 0.9831 0.9988 1.0000 scn ↑ 0.1079 0.1587 0.1302 0.1810 0.1841 0.2381 temp_sty ↑ 0.2220 0.2191 0.2270 0.2294 0.2372 0.2511 ovrl_con ↑ 0.1991 0.2136 0.2153 0.2270 0.2328 0.2360 hum_act ↑ 0.6886 0.7200 0.8371 0.8314 0.7943 0.7371 temp_flick ↑ 0.9763 0.9820 0.9801 0.9822 0.9839 0.9757 mot_smooth ↑ 0.9814 0.9903 0.9912 0.9914 0.9916 0.9836 dyn_deg ↑ 0.4857 0.1679 0.1714 0.1643 0.2429 0.5357 app_sty ↑ 0.2099 0.2039 0.1995 0.1936 0.1890 0.1869

Average Score ↑ 0.6051 0.6275 0.6537 0.6642 0.6713 0.6875

###### Table S6: Quantitative results on 1-minute AR video synthesis during our analysis (w/ Sliding Indices) using VBench-Long, across various sink sizes.

Dimension 0% 17% 33% 50% 67% 83%

sub_con ↑ 0.9679 0.9762 0.9807 0.9783 0.9799 0.9825 bg_con ↑ 0.9653 0.9667 0.9662 0.9649 0.9645 0.9701 aes_qual ↑ 0.5916 0.6080 0.6050 0.6094 0.6055 0.6251 img_qual ↑ 0.6980 0.7058 0.7073 0.6995 0.7074 0.7039 obj_cls ↑ 0.8680 0.9988 0.9989 1.0000 0.9993 0.9996 multi_obj ↑ 0.3639 0.5457 0.6336 0.6693 0.7066 0.6996 col ↑ 0.6433 0.7589 0.7820 0.8573 0.8675 0.7936 spa_rel ↑ 0.7121 0.9711 0.9673 0.9629 0.9965 1.0000 scn ↑ 0.1079 0.1714 0.2286 0.2159 0.1746 0.2603 temp_sty ↑ 0.2220 0.2228 0.2247 0.2210 0.2296 0.2433 ovrl_con ↑ 0.1991 0.2245 0.2201 0.2222 0.2228 0.2309 hum_act ↑ 0.6886 0.6800 0.6914 0.7229 0.7686 0.8000 temp_flick ↑ 0.9763 0.9787 0.9780 0.9755 0.9762 0.9766 mot_smooth ↑ 0.9814 0.9760 0.9806 0.9762 0.9822 0.9826 dyn_deg ↑ 0.4857 0.6750 0.6464 0.6857 0.6179 0.6607 app_sty ↑ 0.2099 0.2045 0.1998 0.1950 0.1921 0.1887

Average Score ↑ 0.6051 0.6665 0.6757 0.6847 0.6870 0.6949

###### Table S7: Quantitative results on 1-minute AR video synthesis during our analysis (w/ Sliding Semantics) using VBench-Long, across various sink sizes.

Dimension 0% 17% 33% 50% 67% 83%

sub_con ↑ 0.9679 0.9766 0.9816 0.9827 0.9824 0.9858 bg_con ↑ 0.9653 0.9678 0.9686 0.9686 0.9678 0.9694 aes_qual ↑ 0.5916 0.6152 0.6236 0.6268 0.6279 0.6308 img_qual ↑ 0.6980 0.6886 0.6901 0.6962 0.6941 0.6968 obj_cls ↑ 0.8680 1.0000 1.0000 1.0000 1.0000 1.0000 multi_obj ↑ 0.3639 0.6388 0.6937 0.6993 0.6984 0.6998 col ↑ 0.6433 0.7526 0.7976 0.8052 0.8058 0.8023 spa_rel ↑ 0.7121 0.9887 0.9945 0.9996 0.9999 1.0000 scn ↑ 0.1079 0.2349 0.2190 0.1778 0.1905 0.2159 temp_sty ↑ 0.2220 0.2446 0.2477 0.2480 0.2482 0.2503 ovrl_con ↑ 0.1991 0.2308 0.2325 0.2304 0.2328 0.2316 hum_act ↑ 0.6886 0.7714 0.7886 0.8371 0.8286 0.7800 temp_flick ↑ 0.9763 0.9818 0.9811 0.9817 0.9808 0.9816 mot_smooth ↑ 0.9814 0.9764 0.9831 0.9848 0.9844 0.9865 dyn_deg ↑ 0.4857 0.6893 0.7036 0.6643 0.7071 0.7469 app_sty ↑ 0.2099 0.2037 0.1969 0.1953 0.1920 0.1891

Average Score ↑ 0.6051 0.6851 0.6939 0.6936 0.6963 0.6979

###### Table S8: Quantitative results on 5-minute AR video synthesis during our analysis (w/ Attention Sink) using VBench-Long, across various sink sizes.

Dimension 0% 17% 33% 50% 67% 83%

sub_con ↑ 0.9424 0.9586 0.9665 0.9784 0.9847 0.9896 bg_con ↑ 0.9610 0.9571 0.9591 0.9592 0.9602 0.9729 aes_qual ↑ 0.4289 0.5037 0.5491 0.5805 0.6210 0.6270 img_qual ↑ 0.5701 0.6193 0.6928 0.6986 0.6951 0.6918 obj_cls ↑ 0.3339 0.5254 0.8167 0.9585 1.0000 1.0000 multi_obj ↑ 0.1427 0.2353 0.4363 0.7611 0.7968 0.7000 col ↑ 0.6105 0.5278 0.7287 0.7007 0.7637 0.8819 spa_rel ↑ 0.3570 0.4578 0.6850 0.9603 0.9980 1.0000 scn ↑ 0.0430 0.0573 0.0896 0.1470 0.2043 0.2294 temp_sty ↑ 0.1132 0.1425 0.1565 0.1959 0.2258 0.2512 ovrl_con ↑ 0.1139 0.1636 0.1759 0.2011 0.2219 0.2356 hum_act ↑ 0.2710 0.4452 0.6548 0.7710 0.7774 0.7323 temp_flick ↑ 0.9820 0.9694 0.9819 0.9853 0.9857 0.9758 mot_smooth ↑ 0.9865 0.9888 0.9905 0.9910 0.9917 0.9836 dyn_deg ↑ 0.2419 0.1694 0.0605 0.1411 0.2742 0.6290 app_sty ↑ 0.2053 0.2048 0.2068 0.2018 0.1906 0.1864

Average Score ↑ 0.4565 0.4954 0.5719 0.6395 0.6682 0.6929

###### Table S9: Quantitative results on 5-minute AR video synthesis during our analysis (w/ Sliding Indices) using VBench-Long, across various sink sizes.

Dimension 0% 17% 33% 50% 67% 83%

sub_con ↑ 0.9424 0.9514 0.9587 0.9610 0.9646 0.9727 bg_con ↑ 0.9610 0.9571 0.9530 0.9513 0.9513 0.9603 aes_qual ↑ 0.4289 0.4961 0.5193 0.5657 0.5901 0.6198 img_qual ↑ 0.5701 0.6136 0.6281 0.6879 0.6981 0.7021 obj_cls ↑ 0.3339 0.6147 0.8107 0.9371 0.8859 0.9917 multi_obj ↑ 0.1427 0.3187 0.3958 0.5175 0.6591 0.7040 col ↑ 0.6105 0.6349 0.6826 0.7667 0.8076 0.8292 spa_rel ↑ 0.3570 0.6828 0.8062 0.8970 0.9479 0.9972 scn ↑ 0.0430 0.0789 0.0717 0.1147 0.2043 0.2616 temp_sty ↑ 0.1132 0.1390 0.1620 0.1943 0.2212 0.2445 ovrl_con ↑ 0.1139 0.1692 0.1793 0.1999 0.2178 0.2290 hum_act ↑ 0.2710 0.4839 0.6548 0.6548 0.7871 0.8097 temp_flick ↑ 0.9820 0.9822 0.9790 0.9762 0.9767 0.9765 mot_smooth ↑ 0.9865 0.9731 0.9758 0.9703 0.9755 0.9791 dyn_deg ↑ 0.2419 0.7097 0.6452 0.7177 0.6774 0.6250 app_sty ↑ 0.2053 0.2050 0.2055 0.2028 0.1956 0.1895

Average Score ↑ 0.4565 0.5631 0.6017 0.6447 0.6725 0.6932

###### Table S10: Quantitative results on 5-minute AR video synthesis during our analysis (w/ Sliding Semantics) using VBench-Long, across various sink sizes.

Dimension 0% 17% 33% 50% 67% 83%

sub_con ↑ 0.9424 0.9535 0.9645 0.9679 0.9701 0.9804 bg_con ↑ 0.9610 0.9523 0.9566 0.9597 0.9607 0.9629 aes_qual ↑ 0.4289 0.5810 0.6065 0.6206 0.6227 0.6296 img_qual ↑ 0.5701 0.6450 0.6538 0.6705 0.6880 0.6987 obj_cls ↑ 0.3339 0.8935 0.9841 0.9980 1.0000 1.0000 multi_obj ↑ 0.1427 0.5050 0.6379 0.6887 0.6968 0.7284 col ↑ 0.6105 0.6945 0.7324 0.8092 0.8251 0.7883 spa_rel ↑ 0.3570 0.8940 0.9646 0.9987 0.9982 1.0000 scn ↑ 0.0430 0.2079 0.2330 0.1613 0.1720 0.2616 temp_sty ↑ 0.1132 0.2308 0.2424 0.2468 0.2484 0.2533 ovrl_con ↑ 0.1139 0.2185 0.2200 0.2237 0.2246 0.2310 hum_act ↑ 0.2710 0.7323 0.7677 0.8194 0.8581 0.8710 temp_flick ↑ 0.9820 0.9841 0.9827 0.9828 0.9825 0.9832 mot_smooth ↑ 0.9865 0.9805 0.9835 0.9836 0.9834 0.9859 dyn_deg ↑ 0.2419 0.5484 0.6815 0.6734 0.7218 0.6411 app_sty ↑ 0.2053 0.2120 0.2058 0.2002 0.1973 0.1891

Average Score ↑ 0.4565 0.6396 0.6761 0.6878 0.6969 0.7003

[Figure 123]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 129]

Self Forcing

= 1s = 5s = 15s = 30s

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

LongLive

= 1s = 5s = 15s = 30s

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

Ours

= 1s = 5s = 15s = 30s

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

= 90s = 150s = 225s = 300s

[Figure 155]

[Figure 157]

[Figure 158]

[Figure 160]

[Figure 161]

Self Forcing

= 1s = 5s = 15s = 30s

[Figure 164]

[Figure 165]

[Figure 167]

[Figure 168]

[Figure 169]

LongLive

[Figure 170]

= 1s = 5s = 15s = 30s

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

Ours

= 1s = 5s = 15s = 30s

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

= 90s = 150s = 225s = 300s

[Figure 189]

###### Fig. S9: More qualitative comparisons of Rolling Sink with SOTA baselines.

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

Self Forcing

= 1s = 5s = 15s = 30s

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

LongLive

= 1s = 5s = 15s = 30s

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

Ours

= 1s = 5s = 15s = 30s

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

= 90s = 150s = 225s = 300s

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

Self Forcing

= 1s = 5s = 15s = 30s

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

LongLive

= 1s = 5s = 15s = 30s

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

###### Ours

= 1s = 5s = 15s = 30s

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

= 90s = 150s = 225s = 300s

###### Fig. S10: More qualitative comparisons of Rolling Sink with SOTA baselines.

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

Self Forcing

= 1s = 5s = 15s = 30s

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

LongLive

= 1s = 5s = 15s = 30s

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

Ours

= 1s = 5s = 15s = 30s

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

= 90s = 150s = 225s = 300s

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

Self Forcing

= 1s = 5s = 15s = 30s

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

LongLive

= 1s = 5s = 15s = 30s

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

###### Ours

= 1s = 5s = 15s = 30s

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

= 90s = 150s = 225s = 300s

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

Self Forcing

= 1s = 5s = 15s = 30s

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

LongLive

= 1s = 5s = 15s = 30s

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

Ours

= 1s = 5s = 15s = 30s

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

= 90s = 150s = 225s = 300s

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

Self Forcing

= 1s = 5s = 15s = 30s

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

LongLive

= 1s = 5s = 15s = 30s

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

Ours

= 1s = 5s = 15s = 30s

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

= 90s = 150s = 225s = 300s

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

Self Forcing

= 1s = 5s = 15s = 30s

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

LongLive

= 1s = 5s = 15s = 30s

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

###### Ours

= 1s = 5s = 15s = 30s

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

= 90s = 150s = 225s = 300s

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

Self Forcing

= 1s = 5s = 15s = 30s

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

LongLive

= 1s = 5s = 15s = 30s

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

###### Ours

= 1s = 5s = 15s = 30s

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

= 90s = 150s = 225s = 300s

###### Fig. S13: More qualitative comparisons of Rolling Sink with SOTA baselines.

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

Self Forcing

= 1s = 5s = 15s = 30s

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

LongLive

= 1s = 5s = 15s = 30s

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

###### Ours

= 1s = 5s = 15s = 30s

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

= 90s = 150s = 225s = 300s

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

Self Forcing

= 1s = 5s = 15s = 30s

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

LongLive

= 1s = 5s = 15s = 30s

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

###### Ours

= 1s = 5s = 15s = 30s

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

Self Forcing

= 1s = 5s = 15s = 30s

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

LongLive

= 1s = 5s = 15s = 30s

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

###### Ours

= 1s = 5s = 15s = 30s

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

= 90s = 150s = 225s = 300s

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

Self Forcing

= 1s = 5s = 15s = 30s

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

LongLive

= 1s = 5s = 15s = 30s

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

###### Ours

= 1s = 5s = 15s = 30s

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

Self Forcing

= 1s = 5s = 15s = 30s

[Figure 389]

[Figure 390]

[Figure 391]

[Figure 392]

LongLive

= 1s = 5s = 15s = 30s

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

###### Ours

= 1s = 5s = 15s = 30s

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

= 90s = 150s = 225s = 300s

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

Self Forcing

= 1s = 5s = 15s = 30s

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

LongLive

= 1s = 5s = 15s = 30s

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

###### Ours

= 1s = 5s = 15s = 30s

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

Self Forcing

= 1s = 5s = 15s = 30s

[Figure 421]

[Figure 422]

[Figure 423]

[Figure 424]

LongLive

= 1s = 5s = 15s = 30s

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

###### Ours

= 1s = 5s = 15s = 30s

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

= 90s = 150s = 225s = 300s

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

Self Forcing

= 1s = 5s = 15s = 30s

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

LongLive

= 1s = 5s = 15s = 30s

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

###### Ours

= 1s = 5s = 15s = 30s

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

Self Forcing

= 1s = 5s = 15s = 30s

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

LongLive

= 1s = 5s = 15s = 30s

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

###### Ours

= 1s = 5s = 15s = 30s

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

= 90s = 150s = 225s = 300s

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

Self Forcing

= 1s = 5s = 15s = 30s

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

LongLive

= 1s = 5s = 15s = 30s

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

###### Ours

= 1s = 5s = 15s = 30s

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

= 90s = 150s = 225s = 300s

### A Text Prompts in Fig. 1

Upper figure: A dynamic snowboarding scene in the style of a high-energy action shot, featuring a young snowboarder accelerating down a powdery slope. The snowboarder, with a determined expression, weaves expertly between tall pine trees, their trunks partially obscured by the swirling snow. The snow is pristine and fluffy, with the sun casting soft shadows and highlighting the snowboarder’s movements. The background showcases a breathtaking mountain vista, with peaks shrouded in mist and a few distant ski lifts visible. The camera angle captures the snowboarder from a slightly behind-the-action perspective, emphasizing their speed and agility.

Bottom figure: A dynamic action shot of a surfer accelerating on a powerful wave, carving through the water with grace and agility. The surfer, with a tanned complexion and muscular build, rides the wave with one hand gripping the board while the other extends outwards for balance. The water splashes behind, creating a foamy trail, and the sun casts a golden glow over the scene. The background features a clear blue ocean and distant white-capped waves, with a few seagulls flying overhead. The surfer’s expression is one of exhilaration and focus. A mid-shot from a low-angle perspective capturing the surfer’s motion and the wave’s power.

### B Related Works

Video Diffusion Models. Video generation is of great benefit in neural simulators [2,3,9] and world models [4,5,23,37,46]. Synthesizing photorealistic videos using video diffusion models [7,8,14,20,28,29,33,34,36,49,67,73,82,90,100,108, 110] has become the community standard, following the substantial success of image diffusion models [30–32,35,51–53,56,58,59,61,64,78,83,92]. Thanks to the strong scaling abilities of video diffusion models and the internet-scale data, the industries have presented many powerful video generators [21,47,70,79,91]. Autoregressive Video Diffusion Models. Video diffusion models typically adopt bidirectional attentions [71] and denoise all frames simultaneously. Therefore, though impressive, the generated videos are generally limited to short clips. In contrast, AR models [1,10,75,87–89] can in principle, infinitely predict nextstate conditioned on prior ones. To marry the best of both paradigms, a rapidly growing number of AR video diffusion models [11,12,16,18,25–27,37,38,42,48,55, 62,63,66,72,74,77,84,93–95,97,98,101,102,106,107,109,111,114] have emerged. Earlier methods, e.g., NOVA [17], SkyReels-V2 [13], and MAGI-1 [86] still rely on inefficient multi-step denoising in each AR generation step. Recently, Pyramid Flow [45] and CausVid [103–105] adopt few-step generation, making AR video generation temporally efficient. However, as the cached history grows longer, the demand of computational resources grows dramatically, which significantly constricts their generation length. More recent SOTA methods like Self Forcing [39] and LongLive [99] cache only a bounded context window, making AR video generation further spatially efficient and thus (architecturally) enabling open-ended generation. However, these models still fall short when synthesizing long videos, especially beyond their training video durations.

### C Quantitative Results during Our Analysis

During our systematic analysis towards a training-free solution, we also conduct quantitative evaluations using VBench-Long [40,41,113] on both 1-minute and 5-minute AR video synthesis, for assessing the performance gains across different analysis steps over different sink sizes6. The experimental setting strictly follows Sec. 4.3. The specific numbers across all dimensions are reported in Tab. S5-S7 (1-minute) and Tab. S8-S10 (5-minute).

Consistent with Fig. 4 and Sec. 3.2, Tab. S5-S10 show a clear upward trend in VBench-Long scores throughout our analysis. In particular, increasing the sink ratio generally yields higher average scores on both 1-minute and 5-minute synthesis. Finally the derived Rolling Sink (after applying w/ Sliding Semantics) achieves the best overall performance at a sink ratio of 83% (i.e., S = 5,K = 6).

### D Additional Qualitative Comparisons

Additional qualitative comparisons between Rolling Sink and prior SOTA AR video diffusion baselines [39,99] are provided in Fig. S9-S18, covering a diverse set of prompts and visual styles. Analogous to Fig. 7 and Sec. 4.2, baseline methods tend to exhibit rapid visual degradations when extrapolating beyond their training horizons, e.g., over-saturated colors and distorted scene structures. In contrast, Rolling Sink consistently suppresses such long-horizon degradation, better preserving subject identity and scene geometry while maintaining coherent motions over long horizons.

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

LongLive w/ LoRA (trained on 1min)

= 1s = 5s = 15s = 30s

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

= 90s = 150s = 225s = 300s

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

Ours (trained on 5s)

= 1s = 5s = 15s = 30s

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

= 90s = 150s = 225s = 300s

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

LongLive w/ LoRA (trained on 1min)

= 1s = 5s = 15s = 30s

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

= 90s = 150s = 225s = 300s

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

Ours (trained on 5s)

= 1s = 5s = 15s = 30s

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

= 90s = 150s = 225s = 300s

###### Fig. S19: Qualitative comparisons between Rolling Sink and LongLive (w/ LoRA). Despite being trained on longer video clips, LongLive (w/ LoRA) still exhibits noticeable AR drift, including identity inconsistency and structural instability over time. In contrast, Rolling Sink better preserves stable appearance and coherent structure, yielding more consistent long-horizon generations.

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

LongLive w/ LoRA (trained on 1min)

= 1s = 5s = 15s = 30s

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

= 90s = 150s = 225s = 300s

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

Ours (trained on 5s)

= 1s = 5s = 15s = 30s

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

= 90s = 150s = 225s = 300s

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

LongLive w/ LoRA (trained on 1min)

= 1s = 5s = 15s = 30s

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

= 90s = 150s = 225s = 300s

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

Ours (trained on 5s)

= 1s = 5s = 15s = 30s

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

= 90s = 150s = 225s = 300s

###### Fig. S20: Additional qualitative comparisons against LongLive (w/ LoRA).

[Figure 547]

[Figure 548]

[Figure 549]

Attention Sink Training (LongLive)

= 1.0  = 1.2s = 1.4s

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 558]

[Figure 559]

[Figure 560]

Attention Sink Training-Free

= 1.0  = 1.2s = 1.4s

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 569]

[Figure 570]

[Figure 571]

Rolling Sink Training-Free

= 1.0  = 1.2s = 1.4s

[Figure 572]

[Figure 573]

[Figure 574]

[Figure 575]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

= = + 0.2  = + 0.4  = + 0.6 

- Fig. S21: Empirical study of severe flickers and video repetition collapse on 1-minute AR video synthesis. We observe two prominent flickers that appear at T1 ≈ 33s and T2 ≈ 50s. Both training (e.g., LongLive [99]) and training-free methods exhibit abrupt appearance/structure changes at these time indices and may further collapse into repetitive frames, whereas Rolling Sink largely suppresses such flickers and maintains coherent motions without video repetition. Here we set KS = 83%.

[Figure 580]

[Figure 581]

[Figure 582]

Attention Sink Training (LongLive)

= 1.0  = 1.2s = 1.4s

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 591]

[Figure 592]

[Figure 593]

Attention Sink Training-Free

= 1.0  = 1.2s = 1.4s

[Figure 594]

[Figure 595]

[Figure 596]

[Figure 597]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 602]

[Figure 603]

[Figure 604]

Rolling Sink Training-Free

= 1.0  = 1.2s = 1.4s

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 613]

[Figure 614]

[Figure 615]

Attention Sink Training (LongLive)

= 1.0  = 1.2s = 1.4s

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 624]

[Figure 625]

[Figure 626]

Attention Sink Training-Free

= 1.0  = 1.2s = 1.4s

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 635]

[Figure 636]

[Figure 637]

Rolling Sink Training-Free

= 1.0  = 1.2s = 1.4s

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 646]

[Figure 647]

[Figure 648]

Attention Sink Training (LongLive)

= 1.0  = 1.2s = 1.4s

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 657]

[Figure 658]

[Figure 659]

Attention Sink Training-Free

= 1.0  = 1.2s = 1.4s

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 668]

[Figure 669]

[Figure 670]

Rolling Sink Training-Free

= 1.0  = 1.2s = 1.4s

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 679]

[Figure 680]

[Figure 681]

Attention Sink Training on 1min (LongLive w/ LoRA)

= 1.0  = 1.2s = 1.4s

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 686]

[Figure 687]

[Figure 688]

Attention Sink Training on 1min

(LongLive w/ LoRA) = 1.0  = 1.2s = 1.4s

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 693]

[Figure 694]

[Figure 695]

Attention Sink Training on 1min

(LongLive w/ LoRA) = 1.0  = 1.2s = 1.4s

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

= = + 0.2  = + 0.4  = + 0.6 

[Figure 700]

[Figure 701]

[Figure 702]

Attention Sink Training on 1min

(LongLive w/ LoRA) = 1.0  = 1.2s = 1.4s

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

= = + 0.2  = + 0.4  = + 0.6 

###### Fig. S25: Empirical study of severe flickers and video repetition collapse of LongLive (w/ LoRA) on 1-minute AR video synthesis. Though LongLive’s LoRA is further trained on 1-minute videos, it still exhibits similar issues. Here T1 ≈ 50.

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

Attention Sink Training (LongLive)

[Figure 711]

= 33s = 50s = 58s = 60s

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

Attention Sink Training-Free

= 33s = 49s = 58s = 60s

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

Attention Sink Training (LongLive)

= 34s = 51s = 56s = 60s

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

Attention Sink Training-Free

= 34s = 51s = 56s = 60s

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

Attention Sink Training (LongLive)

= 34s = 37s = 47s = 51s

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

Attention Sink Training-Free

= 34s = 37s = 49s = 51s

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

Attention Sink Training (LongLive)

= 34s = 37s = 49s = 51s

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

Attention Sink Training-Free

= 34s = 37s = 49s = 51s

###### Fig. S26: Empirical study of intermittent frame flickers in AR video synthesis during 30-60s. Frame flickers (usually every several seconds) persist in both training (e.g., LongLive [99]) and training-free methods, suggesting that applying attention sinks is only part of the solution in eliminating the long-horizon AR drift.

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

= 34s = 50s = 33s = 50s

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

= 33s = 50s = 33s = 49s

###### Fig. S27: Empirical study of intermittent frame flickers of LongLive (w/ LoRA) during 30-60s. Consistent with Fig. S26, training on longer videos (i.e., LongLive w/ LoRA) still exhibits frame flickers but in a lower frequency.

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

= 1s = 5s = 15s = 30s

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

= 45s = 60s = 90s = 120s

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

= 3min = 4min = 5min = 6min

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

= 7min = 8min = 9min = 10min

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

= 11min = 12min = 13min = 14min

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

= 15min = 16min = 17min = 18min

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

= 19min = 20min = 21min = 22min

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

= 23min = 24min

= 29min = 30min

[Figure 789]

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

= 1s = 5s = 15s = 30s

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

= 45s = 60s = 90s = 120s

[Figure 798]

[Figure 799]

[Figure 800]

[Figure 801]

= 3min = 4min = 5min = 6min

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

= 7min = 8min = 9min = 10min

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

= 11min = 12min = 13min = 14min

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

= 15min = 16min = 17min = 18min

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

= 19min = 20min = 21min = 22min

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

= 23min = 24min

= 29min = 30min

[Figure 825]

### E Rolling Sink vs. LongLive (w/ LoRA)

As discussed in Sec. 4.1, we do not load LongLive’s train-long-test-long LoRA (further trained on 1-minute videos) in our main comparisons (Fig. 7, S9-S18 and Tab. 1, 2), to ensure all compared methods share the same 5s training duration.

To further evaluate Rolling Sink under more challenging situations, we additionally compare against LongLive [99] with its train-long-test-long LoRA. We report qualitative results in Fig. S19 and S20, and quantitative results in Tab. S3 (1-minute) and S4 (5-minute), where we also include the results of Self Forcing [39] and LongLive (w/o LoRA) from Tab. 1 and 2 for reference.

Indeed, LongLive’s train-long-test-long LoRA effectively improves the longhorizon AR video synthesis quality compared with its w/o LoRA variant. However, as illustrated in Fig. S19 and S20, LongLive (w/ LoRA) still exhibits noticeable AR drift, especially the inconsistent subject identities. In contrast, Rolling Sink (built on Self Forcing [39] and trained on only 5s clips) produces more stable and temporally consistent rollouts over long horizons. The superiority of Rolling Sink is also reflected quantitatively in Tab. S3 and S4, where Rolling Sink continuously achieves the lowest (best) average rank. These results highlight that “drift-free” long-horizon AR video synthesis is not solely determined by the training horizon. Stabilizing the AR cache is a key factor for bridging the mismatch between limited-horizon training and open-ended testing, and can be more effective than expensively extending the training horizon.

### F Why Not Building on LongLive?

Self Forcing is a more standardized base, which is more suitable for systematic analysis. LongLive [99] revisits not only recent blocks but also initially self-generated blocks (i.e., attention sink) during AR video synthesis. While Self Forcing [39] only revisits recent blocks. Moreover, different from Self Forcing, LongLive further incorporates a train-long-test-long LoRA (trained on 1-minute videos), which extends the feasible generation horizon at test time (Tab. S3 and S4). Therefore, LongLive (and many other methods after Self Forcing) can be viewed as an extension of Self Forcing.

Self Forcing is a more standardized AR video synthesis method categorized as “Self Forcing”, which is itself initially proposed by Self Forcing [39] (please see Sec. 3.1 for more details). Analyzing and exploring based on Self Forcing provides this community with more systematic and in-depth insights, which will help us better understand the role of AR cache maintenance in AR video synthesis. Moreover, starting from Self Forcing also makes our performance gains more transparent: we can show clear improvements step-by-step (following Fig. 3). Also, as reported in Fig. 7, Fig. S9-S18, and Tab. 1, 2, building on Self Forcing can more significantly demonstrate the effect of Rolling Sink, which yields superior performance over Self Forcing with a large margin.

Noticeable AR drift (e.g., flickers and repetition collapse) persist in LongLive, analogous to our analysis. Though LongLive [99] exhibits more stable colors than Self Forcing [39] by training with attention sinks, it still exhibits noticeable AR drift (e.g., flickers in every several seconds) especially at long horizons, as illustrated in Fig. S26 (the first row of each case). Empirically, when synthesizing 1-minute videos, as illustrated in Fig. S21-S24, two particularly prominent flickers usually appear at around 33s (T1) and 50s (T2), followed by repetition collapse. Moreover, as illustrated in Fig. S27 and S25, training on longer durations (i.e., LongLive w/ LoRA) can only partially reduce frame flickering. Empirically, when testing on 1-minute AR video synthesis, we still observe a prominent flicker that appears at around 50s followed by repetition collapse.

It’s worthwhile to note that these empirical studies are analogous to our analysis in Sec. 3.2, pinning a static prefix of initially self-generated latents effectively stabilizes colors, but noticeable AR drifts persist, e.g., flickers in every several seconds. Our analysis (Sec. 3.2) suggests that this behavior is closely tied to the gap caused by static time indices and semantic content of the sink blocks. During LongLive’s training, the gap is small within the limited 5s (16 FPS) training window. But during testing, this gap increasingly amplifies as the testing duration grows, which destabilizes the AR video synthesis and flickers emerge. Some prominent flickers will also cause repetition collapse (Fig. S21-S25). In addition, the long-horizon instability of RoPE [85] may also be part of the reason: as discussed in DroPE [19], high-frequency dimensions in RoPE quickly saturate due to the rapid rotation angles, making the positional encoding back to the beginning; meanwhile, low-frequency dimensions change their rotation angles too slowly, likewise failing to provide positional information. Moreover, as revealed in MotionStream [81], during AR video synthesis, the current latent block tends to attend predominantly to the earliest self-generated blocks (e.g., attention sink), which may further contribute to this instability.

As illustrated in Fig. S21-S24, both training (i.e., LongLive) and training-free methods exhibit similar temporal instabilities. This supports that keeping the AR cache minimally drifted is only part of the solution, long-horizon stability requires further characteristics, e.g., the sliding in time indices and semantics.

SOTA performance based on Self Forcing. Even based on Self Forcing [39], the proposed Rolling Sink still achieves superior performance compared with prior representative baselines (i.e., Self Forcing and LongLive [99]), as demonstrated by extensive experiments (Fig. 7, Fig. S9-S18, and Tab. 1, 2). Our method even performs better compared with LongLive (w/ LoRA), which is further trained on 1 minute videos (Fig. S19, S20 and Tab. S3, S4).

###### Table S11: Abbreviation legend for VBench-Long dimensions.

Abbreviation Full Dimension Name

sub_con subject_consistency bg_con background_consistency aes_qual aesthetic_quality img_qual imaging_quality obj_cls object_class multi_obj multiple_objects col color spa_rel spatial_relationship scn scene temp_sty temporal_style ovrl_con overall_consistency hum_act human_action temp_flick temporal_flickering mot_smooth motion_smoothness dyn_deg dynamic_degree app_sty appearance_style

### G 30-Minute AR Video Synthesis

We further evaluate Rolling Sink in ultra-long, open-ended settings by extending AR video synthesis to 30 minutes at test time. As shown in Fig. S28 and S29, Rolling Sink maintains strong long-horizon stability across both realistic and animated content. Over the entire 30-minute rollout, Rolling Sink preserves consistent subject identity (e.g., appearance, shape, etc.), maintains coherent structures and colors, and produces smooth dynamics, without long-horizon AR drift such as over-saturation, texture degradation, or structural collapse. These illustrations also highlight the potential of cache maintenance as a principled and practical path to bridge the limited-horizon training and open-ended testing.

### H Abbreviation Legend & Prompt Lists

Please see Tab. S11 for the legend of the abbreviations. Prompt list for aesthetic_quality, imaging_quality, and overall_consistency:

- – A corgi is playing drum kit.
- – A jellyfish floating through the ocean, with bioluminescent tentacles
- – golden fish swimming in the ocean.
- – Hyper-realistic spaceship landing on Mars
- – Yoda playing guitar on the stage
- – A future where humans have achieved teleportation technology
- – Turtle swimming in ocean.
- – Origami dancers in white paper, 3D render, on white background, studio shot, dancing modern dance.

- – A robot DJ is playing the turntable, in heavy raining futuristic tokyo rooftop cyberpunk night, sci-fi, fantasy
- – An astronaut is riding a horse in the space in a photorealistic style. Prompt list for appearance_style:
- – A cute happy Corgi playing in park, sunset, animated style
- – A couple in formal evening wear going home get caught in a heavy downpour with umbrellas, in cyberpunk style
- – a shark is swimming in the ocean, animated style
- – A boat sailing leisurely along the Seine River with the Eiffel Tower in background, watercolor painting
- – A cute happy Corgi playing in park, sunset, watercolor painting
- – A boat sailing leisurely along the Seine River with the Eiffel Tower in background by Hokusai, in the style of Ukiyo
- – a shark is swimming in the ocean, surrealism style
- – A couple in formal evening wear going home get caught in a heavy downpour with umbrellas, Van Gogh style
- – A couple in formal evening wear going home get caught in a heavy downpour with umbrellas, watercolor painting
- – A cute happy Corgi playing in park, sunset by Hokusai, in the style of Ukiyo Prompt list for background_consistency and scene:
- – outdoor track
- – train station platform
- – indoor swimming pool
- – windmill
- – phone booth
- – train railway
- – indoor movie theater
- – underwater coral reef
- – river
- – supermarket Prompt list for color:
- – a black bird
- – a black car
- – a pink bird
- – a red car
- – a red bird
- – an orange bird
- – a green car
- – a red bicycle
- – a blue car
- – a yellow bicycle Prompt list for dynamic_degree, motion_smoothness, and subject_consistency:
- – a bear catching a salmon in its powerful jaws
- – a dog playing in park
- – a person washing the dishes

- – a bear hunting for prey
- – a zebra bending down to drink water from a river
- – a cat playing in park
- – an elephant running to join a herd of its kind
- – a sheep running to join a herd of its kind
- – a cat running happily
- – a giraffe running to join a herd of its kind Prompt list for human_action:
- – A person is climbing a rope
- – A person is robot dancing
- – A person is ice skating
- – A person is doing aerobics
- – A person is air drumming
- – A person is smoking
- – A person is taking a shower
- – A person is riding or walking with horse
- – A person is hula hooping
- – A person is riding a bike Prompt list for multiple_objects:
- – a person and a toilet
- – a zebra and a giraffe
- – a dog and a horse
- – a person and a hair drier
- – a cow and an elephant
- – a bear and a zebra
- – a person and a sink
- – a giraffe and a bird
- – an elephant and a bear
- – a bird and a cat Prompt list for object_class:
- – a car
- – a motorcycle
- – a bear
- – a sheep
- – a giraffe
- – a bird
- – an elephant
- – a zebra
- – an airplane
- – a dog Prompt list for spatial_relationship:
- – a bird on the left of a cat, front view
- – a cat on the right of a dog, front view
- – a dog on the left of a horse, front view
- – a horse on the right of a sheep, front view

- – a sheep on the left of a cow, front view
- – a cow on the right of an elephant, front view
- – an elephant on the left of a bear, front view
- – a bear on the right of a zebra, front view
- – a zebra on the left of a giraffe, front view
- – a giraffe on the right of a bird, front view Prompt list for temporal_flickering:
- – A tranquil tableau of in the desolate beauty of the American Southwest, Chaco Canyon’s ancient ruins whispered tales of an enigmatic civilization that once thrived amidst the arid landscapes
- – A tranquil tableau of the lampposts were adorned with Art Deco motifs, their geometric shapes and frosted glass creating a sense of vintage glamour
- – A tranquil tableau of an exquisite mahogany dining table
- – In a still frame, amidst the cobblestone streets, an Art Nouveau lamppost stood tall
- – A tranquil tableau of a vintage rocking chair was placed on the porch
- – A tranquil tableau of beneath the shade of a solitary oak tree, an old wooden park bench sat patiently
- – In a still frame, phone booth
- – A tranquil tableau of barn
- – In a still frame, a vintage gas lantern, adorned with intricate details, gracing a historic cobblestone square
- – A tranquil tableau of a beautiful wrought-iron bench surrounded by blooming flowers

##### Prompt list for temporal_style:

- – A cute happy Corgi playing in park, sunset, pan right
- – A boat sailing leisurely along the Seine River with the Eiffel Tower in background, zoom out
- – a shark is swimming in the ocean, tilt down
- – A boat sailing leisurely along the Seine River with the Eiffel Tower in background, tilt down
- – a shark is swimming in the ocean, with an intense shaking effect
- – A panda drinking coffee in a cafe in Paris, in super slow motion
- – An astronaut flying in space, in super slow motion
- – A panda drinking coffee in a cafe in Paris, tilt down
- – An astronaut flying in space, tilt down
- – A boat sailing leisurely along the Seine River with the Eiffel Tower in background, featuring a steady and smooth perspective

### References

- 1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al.: Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023) 30
- 2. Agarwal, N., Ali, A., Bala, M., Balaji, Y., Barker, E., Cai, T., Chattopadhyay, P., Chen, Y., Cui, Y., Ding, Y., et al.: Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575 (2025) 30
- 3. Ali, A., Bai, J., Bala, M., Balaji, Y., Blakeman, A., Cai, T., Cao, J., Cao, T., Cha, E., Chao, Y.W., et al.: World simulation with video foundation models for physical ai. arXiv preprint arXiv:2511.00062 (2025) 30
- 4. Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Komeili, M., Muckley, M., Rizvi, A., Roberts, C., Sinha, K., Zholus, A., Arnaud, S., Gejji, A., Martin, A., Robert Hogan, F., Dugas, D., Bojanowski, P., Khalidov, V., Labatut, P., Massa, F., Szafraniec, M., Krishnakumar, K., Li, Y., Ma, X., Chandar, S., Meier, F., LeCun, Y., Rabbat, M., Ballas, N.: V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985 (2025) 30
- 5. Ball, P.J., Bauer, J., Belletti, F., Brownfield, B., Ephrat, A., Fruchter, S., Gupta, A., Holsheimer, K., Holynski, A., Hron, J., Kaplanis, C., Limont, M., McGill, M., Oliveira, Y., Parker-Holder, J., Perbet, F., Scully, G., Shar, J., Spencer, S., Tov, O., Villegas, R., Wang, E., Yung, J., Baetu, C., Berbel, J., Bridson, D., Bruce, J., Buttimore, G., Chakera, S., Chandra, B., Collins, P., Cullum, A., Damoc, B., Dasagi, V., Gazeau, M., Gbadamosi, C., Han, W., Hirst, E., Kachra, A., Kerley, L., Kjems, K., Knoepfel, E., Koriakin, V., Lo, J., Lu, C., Mehring, Z., Moufarek, A., Nandwani, H., Oliveira, V., Pardo, F., Park, J., Pierson, A., Poole, B., Ran, H., Salimans, T., Sanchez, M., Saprykin, I., Shen, A., Sidhwani, S., Smith, D., Stanton, J., Tomlinson, H., Vijaykumar, D., Wang, L., Wingfield, P., Wong, N., Xu, K., Yew, C., Young, N., Zubov, V., Eck, D., Erhan, D., Kavukcuoglu, K., Hassabis, D., Gharamani, Z., Hadsell, R., van den Oord, A., Mosseri, I., Bolton, A., Singh, S., Rocktäschel, T.: Genie 3: A new frontier for world models (2025), https: //deepmind.google/blog/genie-3-a-new-frontier-for-world-models/ 30
- 6. Bengio, S., Vinyals, O., Jaitly, N., Shazeer, N.: Scheduled sampling for sequence prediction with recurrent neural networks. Advances in neural information processing systems 28 (2015) 3
- 7. Blattmann, A., Dockhorn, T., Kulal, S., Mendelevitch, D., Kilian, M., Lorenz, D., Levi, Y., English, Z., Voleti, V., Letts, A., et al.: Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127

(2023) 30

- 8. Blattmann, A., Rombach, R., Ling, H., Dockhorn, T., Kim, S.W., Fidler, S., Kreis, K.: Align your latents: High-resolution video synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 22563–22575 (2023) 30
- 9. Brooks, T., Peebles, B., Holmes, C., DePue, W., Guo, Y., Jing, L., Schnurr, D., Taylor, J., Luhman, T., Luhman, E., et al.: Video generation models as world simulators. OpenAI Blog 1(8), 1 (2024) 30
- 10. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are fewshot learners. Advances in neural information processing systems 33, 1877–1901

(2020) 30

- 11. Bruce, J., Dennis, M.D., Edwards, A., Parker-Holder, J., Shi, Y., Hughes, E., Lai, M., Mavalankar, A., Steigerwald, R., Apps, C., et al.: Genie: Generative interactive environments. In: Forty-first International Conference on Machine Learning

(2024) 30

- 12. Chen, B., Martí Monsó, D., Du, Y., Simchowitz, M., Tedrake, R., Sitzmann, V.: Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems 37, 24081–24125 (2024) 5, 30
- 13. Chen, G., Lin, D., Yang, J., Lin, C., Zhu, J., Fan, M., Zhang, H., Chen, S., Chen, Z., Ma, C., et al.: Skyreels-v2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074 (2025) 5, 30
- 14. Chen, H., Xia, M., He, Y., Zhang, Y., Cun, X., Yang, S., Xing, J., Liu, Y., Chen, Q., Wang, X., et al.: Videocrafter1: Open diffusion models for high-quality video generation. arXiv preprint arXiv:2310.19512 (2023) 30
- 15. Chung, H.W., Constant, N., Garcia, X., Roberts, A., Tay, Y., Narang, S., Firat, O.: Unimax: Fairer and more effective language sampling for large-scale multilingual pretraining. arXiv preprint arXiv:2304.09151 (2023) 6
- 16. Cui, J., Wu, J., Li, M., Yang, T., Li, X., Wang, R., Bai, A., Ban, Y., Hsieh, C.J.: Self-forcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283 (2025) 3, 5, 30
- 17. Deng, H., Pan, T., Diao, H., Luo, Z., Cui, Y., Lu, H., Shan, S., Qi, Y., Wang, X.: Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169 (2024) 30
- 18. Gao, K., Shi, J., Zhang, H., Wang, C., Xiao, J., Chen, L.: Ca2-vdm: Efficient autoregressive video diffusion model with causal generation and cache sharing. arXiv preprint arXiv:2411.16375 (2024) 5, 30
- 19. Gelberg, Y., Eguchi, K., Akiba, T., Cetin, E.: Extending the context of pretrained llms by dropping their positional embeddings. arXiv preprint arXiv:2512.12167

(2025) 44

- 20. Girdhar, R., Singh, M., Brown, A., Duval, Q., Azadi, S., Rambhatla, S.S., Shah, A., Yin, X., Parikh, D., Misra, I.: Emu video: Factorizing text-to-video generation by explicit image conditioning. arXiv preprint arXiv:2311.10709 (2023) 30
- 21. Google: Introducing veo 3, our video generation model with expanded creative controls – including native audio and extended videos. https://deepmind. google/models/veo/ (2025) 2, 30
- 22. Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al.: The llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024) 7
- 23. Gu, S., Yin, W., Jin, B., Guo, X., Wang, J., Li, H., Zhang, Q., Long, X.: Dome: Taming diffusion model into high-fidelity controllable occupancy world model. arXiv preprint arXiv:2410.10429 (2024) 30
- 24. Gu, X., Pang, T., Du, C., Liu, Q., Zhang, F., Du, C., Wang, Y., Lin, M.: When attention sink emerges in language models: An empirical view. arXiv preprint arXiv:2410.10781 (2024) 7
- 25. Gu, Y., Mao, W., Shou, M.Z.: Long-context autoregressive video modeling with next-frame prediction. arXiv preprint arXiv:2503.19325 (2025) 5, 30
- 26. Guo, Y., Yang, C., He, H., Zhao, Y., Wei, M., Yang, Z., Huang, W., Lin, D.: End-to-end training for autoregressive video diffusion via self-resampling. arXiv preprint arXiv:2512.15702 (2025) 30
- 27. Guo, Y., Yang, C., Yang, Z., Ma, Z., Lin, Z., Yang, Z., Lin, D., Jiang, L.: Long context tuning for video generation. arXiv preprint arXiv:2503.10589 (2025) 30

- 28. Gupta, A., Yu, L., Sohn, K., Gu, X., Hahn, M., Li, F.F., Essa, I., Jiang, L., Lezama, J.: Photorealistic video generation with diffusion models. In: European Conference on Computer Vision. pp. 393–411. Springer (2024) 30
- 29. HaCohen, Y., Chiprut, N., Brazowski, B., Shalem, D., Moshe, D., Richardson, E., Levin, E., Shiran, G., Zabari, N., Gordon, O., et al.: Ltx-video: Realtime video latent diffusion. arXiv preprint arXiv:2501.00103 (2024) 30
- 30. He, J., Li, H., Hu, Y., Shen, G., Cai, Y., Qiu, W., Chen, Y.C.: Disenvisioner: Disentangled and enriched visual prompt for customized image generation. arXiv preprint arXiv:2410.02067 (2024) 30
- 31. He, J., Li, H., Sheng, M., Chen, Y.C.: Lotus-2: Advancing geometric dense prediction with powerful image generative model. arXiv preprint arXiv:2512.01030

(2025) 30

- 32. He, J., Li, H., Yin, W., Liang, Y., Li, L., Zhou, K., Zhang, H., Liu, B., Chen, Y.C.: Lotus: Diffusion-based visual foundation model for high-quality dense prediction. arXiv preprint arXiv:2409.18124 (2024) 30
- 33. Henschel, R., Khachatryan, L., Hayrapetyan, D., Poghosyan, H., Tadevosyan, V., Wang, Z., Navasardyan, S., Shi, H.: Streamingt2v: Consistent, dynamic, and extendable long video generation from text. arXiv preprint arXiv:2403.14773 (2024) 30
- 34. Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko, A., Kingma, D.P., Poole, B., Norouzi, M., Fleet, D.J., et al.: Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv:2210.02303 (2022) 30
- 35. Ho, J., Jain, A., Abbeel, P.: Denoising diffusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020) 30
- 36. Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., Fleet, D.J.: Video diffusion models. Advances in neural information processing systems 35, 8633– 8646 (2022) 30
- 37. Hong, Y., Mei, Y., Ge, C., Xu, Y., Zhou, Y., Bi, S., Hold-Geoffroy, Y., Roberts, M., Fisher, M., Shechtman, E., et al.: Relic: Interactive video world model with long-horizon memory. arXiv preprint arXiv:2512.04040 (2025) 5, 30
- 38. Hu, J., Hu, S., Song, Y., Huang, Y., Wang, M., Zhou, H., Liu, Z., Ma, W.Y., Sun, M.: Acdit: Interpolating autoregressive conditional modeling and diffusion transformer. arXiv preprint arXiv:2412.07720 (2024) 5, 30
- 39. Huang, X., Li, Z., He, G., Zhou, M., Shechtman, E.: Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009

(2025) 2, 3, 4, 5, 6, 7, 10, 13, 14, 30, 31, 43, 44

- 40. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., Wang, Y., Chen, X., Wang, L., Lin, D., Qiao, Y., Liu, Z.: VBench: Comprehensive benchmark suite for video generative models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2024) 4, 7, 10, 13, 31
- 41. Huang, Z., Zhang, F., Xu, X., He, Y., Yu, J., Dong, Z., Ma, Q., Chanpaisit, N., Si, C., Jiang, Y., Wang, Y., Chen, X., Chen, Y.C., Wang, L., Lin, D., Qiao, Y., Liu, Z.: VBench++: Comprehensive and versatile benchmark suite for video generative models. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025). https://doi.org/10.1109/TPAMI.2025.3633890 4, 7, 10, 13, 31
- 42. Ji, S., Chen, X., Yang, S., Tao, X., Wan, P., Zhao, H.: Memflow: Flowing adaptive memory for consistent and efficient long video narratives. arXiv preprint arXiv:2512.14699 (2025) 7, 30

- 43. Jiang, A.Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D.S., de las Casas, D., Bressand, F., Lengyel, G., Lample, G., Saulnier, L., Lavaud, L.R., Lachaux, M.A., Stock, P., Scao, T.L., Lavril, T., Wang, T., Lacroix, T., Sayed, W.E.: Mistral 7b (2025) 7
- 44. Jiang, A.Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D.S., Casas, D.d.l., Hanna, E.B., Bressand, F., et al.: Mixtral of experts. arXiv preprint arXiv:2401.04088 (2024) 7
- 45. Jin, Y., Sun, Z., Li, N., Xu, K., Jiang, H., Zhuang, N., Huang, Q., Song, Y., Mu, Y., Lin, Z.: Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954 (2024) 5, 30
- 46. Kanervisto, A., Bignell, D., Wen, L.Y., Grayson, M., Georgescu, R., Valcarcel Macua, S., Tan, S.Z., Rashid, T., Pearce, T., Cao, Y., et al.: World and human action models towards gameplay ideation. Nature 638(8051), 656–663 (2025) 30
- 47. Kling: Kling video 2.6 – kling’s first “native audio” model official launched! https: //app.klingai.com/global/release-notes/c605hp1tzd (2025) 2, 30
- 48. Kondratyuk, D., Yu, L., Gu, X., Lezama, J., Huang, J., Schindler, G., Hornung, R., Birodkar, V., Yan, J., Chiu, M.C., et al.: Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125 (2023) 30
- 49. Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., et al.: Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603 (2024) 30
- 50. Kubrick, S.: The shining. https://en.wikipedia.org/wiki/The_Shining_ (film) (1980) 2
- 51. Labs, B.F.: Flux. https://github.com/black-forest-labs/flux (2024) 30
- 52. Labs, B.F.: Flux.2: Frontier visual intelligence. https://bfl.ai/blog/flux-2

(2025) 30

- 53. Labs, B.F., Batifol, S., Blattmann, A., Boesel, F., Consul, S., Diagne, C., Dockhorn, T., English, J., English, Z., Esser, P., Kulal, S., Lacey, K., Levi, Y., Li, C., Lorenz, D., Müller, J., Podell, D., Rombach, R., Saini, H., Sauer, A., Smith, L.: Flux.1 kontext: Flow matching for in-context image generation and editing in latent space. arXiv preprint arXiv:2506.15742 (2025) 30
- 54. Lamb, A.M., ALIAS PARTH GOYAL, A.G., Zhang, Y., Zhang, S., Courville, A.C., Bengio, Y.: Professor forcing: A new algorithm for training recurrent networks. Advances in neural information processing systems 29 (2016) 3
- 55. Li, C., Wang, R., Zhou, L., Feng, J., Luo, H., Zhang, H., Wu, Y., He, X.: Joyavatar: Real-time and infinite audio-driven avatar generation with autoregressive diffusion. arXiv preprint arXiv:2512.11423 (2025) 30
- 56. Li, H., Zheng, W., He, J., Liu, Y., Lin, X., Yang, X., Chen, Y.C., Guo, C.: Da 2: Depth anything in any direction. arXiv preprint arXiv:2509.26618 (2025) 30
- 57. Li, M., Qu, T., Yao, R., Sun, W., Moens, M.F.: Alleviating exposure bias in diffusion models through sampling with shifted time steps. arXiv preprint arXiv:2305.15583 (2023) 3
- 58. Li, X.L., Li, H., Chen, H.X., Mu, T.J., Hu, S.M.: Discene: Object decoupling and interaction modeling for complex scene generation. In: SIGGRAPH Asia 2024 Conference Papers. pp. 1–12 (2024) 30
- 59. Liang, Y., Yang, X., Lin, J., Li, H., Xu, X., Chen, Y.: Luciddreamer: Towards high-fidelity text-to-3d generation via interval score matching. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6517–6526 (2024) 30

- 60. Lin, S., Yang, C., He, H., Jiang, J., Ren, Y., Xia, X., Zhao, Y., Xiao, X., Jiang, L.: Autoregressive adversarial post-training for real-time interactive video generation. arXiv preprint arXiv:2506.09350 (2025) 3
- 61. Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 (2022) 4, 30
- 62. Liu, H., Liu, S., Zhou, Z., Xu, M., Xie, Y., Han, X., Pérez, J.C., Liu, D., Kahatapitiya, K., Jia, M., et al.: Mardini: Masked autoregressive diffusion for video generation at scale. arXiv preprint arXiv:2410.20280 (2024) 30
- 63. Liu, K., Hu, W., Xu, J., Shan, Y., Lu, S.: Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161 (2025) 3, 30
- 64. Liu, X., Gong, C., Liu, Q.: Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003 (2022) 4, 30
- 65. Low, C., Wang, W.: Talkingmachines: Real-time audio-driven facetime-style video via autoregressive diffusion models. arXiv preprint arXiv:2506.03099 (2025) 7
- 66. Lu, Y., Zeng, Y., Li, H., Ouyang, H., Wang, Q., Cheng, K.L., Zhu, J., Cao, H., Zhang, Z., Zhu, X., et al.: Reward forcing: Efficient streaming video generation with rewarded distribution matching distillation. arXiv preprint arXiv:2512.04678

(2025) 3, 5, 30

- 67. Ma, X., Wang, Y., Chen, X., Jia, G., Liu, Z., Li, Y.F., Chen, C., Qiao, Y.: Latte: Latent diffusion transformer for video generation. arXiv preprint arXiv:2401.03048 (2024) 30
- 68. McQueen, S.: Hunger. https://en.wikipedia.org/wiki/Hunger_(2008_film)

(2008) 2

- 69. Ning, M., Li, M., Su, J., Salah, A.A., Ertugrul, I.O.: Elucidating the exposure bias in diffusion models. arXiv preprint arXiv:2308.15321 (2023) 3
- 70. OpenAI: Sora 2 is here. https://openai.com/index/sora-2/ (2025) 2, 30
- 71. Peebles, W., Xie, S.: Scalable diffusion models with transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4195–4205

(2023) 2, 10, 30

- 72. Po, R., Chan, E.R., Chen, C., Wetzstein, G.: Bagger: Backwards aggregation for mitigating drift in autoregressive video diffusion models. arXiv preprint arXiv:2512.12080 (2025) 3, 5, 30
- 73. Polyak, A., Zohar, A., Brown, A., Tjandra, A., Sinha, A., Lee, A., Vyas, A., Shi, B., Ma, C.Y., Chuang, C.Y., et al.: Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720 (2024) 30
- 74. Qiu, H., Liu, S., Zhou, Z., An, Z., Ren, W., Liu, Z., Schult, J., He, S., Chen, S., Cong, Y., et al.: Histream: Efficient high-resolution video generation via redundancy-eliminated streaming. arXiv preprint arXiv:2512.21338 (2025) 30
- 75. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al.: Language models are unsupervised multitask learners. OpenAI blog 1(8), 9 (2019) 30
- 76. Ranzato, M., Chopra, S., Auli, M., Zaremba, W.: Sequence level training with recurrent neural networks. arXiv preprint arXiv:1511.06732 (2015) 3
- 77. Ren, S., Ma, S., Sun, X., Wei, F.: Next block prediction: Video generation via semi-autoregressive modeling. arXiv preprint arXiv:2502.07737 (2025) 30
- 78. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent diffusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022) 30
- 79. Runway: Introducing runway gen-4.5: A new frontier for video generation. https: //runwayml.com/research/introducing-runway-gen-4.5 (2025) 2, 30

- 80. Schmidt, F.: Generalization in generation: A closer look at exposure bias. arXiv preprint arXiv:1910.00292 (2019) 3
- 81. Shin, J., Li, Z., Zhang, R., Zhu, J.Y., Park, J., Shechtman, E., Huang, X.: Motionstream: Real-time video generation with interactive motion controls. arXiv preprint arXiv:2511.01266 (2025) 7, 44
- 82. Singer, U., Polyak, A., Hayes, T., Yin, X., An, J., Zhang, S., Hu, Q., Yang, H., Ashual, O., Gafni, O., et al.: Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792 (2022) 30
- 83. Song, J., Meng, C., Ermon, S.: Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502 (2020) 5, 30
- 84. Song, K., Chen, B., Simchowitz, M., Du, Y., Tedrake, R., Sitzmann, V.: Historyguided video diffusion. arXiv preprint arXiv:2502.06764 (2025) 5, 30
- 85. Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., Liu, Y.: Roformer: Enhanced transformer with rotary position embedding. Neurocomputing 568, 127063 (2024) 6, 44
- 86. Teng, H., Jia, H., Sun, L., Li, L., Li, M., Tang, M., Han, S., Zhang, T., Zhang, W., Luo, W., et al.: Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211 (2025) 5, 30
- 87. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al.: Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971 (2023) 7, 30
- 88. Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al.: Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023) 30
- 89. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017) 30
- 90. Villegas, R., Babaeizadeh, M., Kindermans, P.J., Moraldo, H., Zhang, H., Saffar, M.T., Castro, S., Kunze, J., Erhan, D.: Phenaki: Variable length video generation from open domain textual description. arXiv preprint arXiv:2210.02399 (2022) 30
- 91. Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.W., Chen, D., Yu, F., Zhao, H., Yang, J., Zeng, J., Wang, J., Zhang, J., Zhou, J., Wang, J., Chen, J., Zhu, K., Zhao, K., Yan, K., Huang, L., Feng, M., Zhang, N., Li, P., Wu, P., Chu, R., Feng, R., Zhang, S., Sun, S., Fang, T., Wang, T., Gui, T., Weng, T., Shen, T., Lin, W., Wang, W., Wang, W., Zhou, W., Wang, W., Shen, W., Yu, W., Shi, X., Huang,

- X., Xu, X., Kou, Y., Lv, Y., Li, Y., Liu, Y., Wang, Y., Zhang, Y., Huang, Y., Li,
- Y., Wu, Y., Liu, Y., Pan, Y., Zheng, Y., Hong, Y., Shi, Y., Feng, Y., Jiang, Z., Han, Z., Wu, Z.F., Liu, Z.: Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025) 2, 5, 10, 30

- 92. Wang, J., Lin, C., Guan, C., Nie, L., He, J., Li, H., Liao, K., Zhao, Y.: Jasmine: Harnessing diffusion prior for self-supervised depth estimation. arXiv preprint arXiv:2503.15905 (2025) 30
- 93. Wang, Y., Xiong, T., Zhou, D., Lin, Z., Zhao, Y., Kang, B., Feng, J., Liu, X.: Loong: Generating minute-level long videos with autoregressive language models. arXiv preprint arXiv:2410.02757 (2024) 30
- 94. Weissenborn, D., Täckström, O., Uszkoreit, J.: Scaling autoregressive video models. arXiv preprint arXiv:1906.02634 (2020) 30
- 95. Weng, W., Feng, R., Wang, Y., Dai, Q., Wang, C., Yin, D., Zhao, Z., Qiu, K., Bao, J., Yuan, Y., et al.: Art-v: Auto-regressive text-to-video generation with diffusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7395–7405 (2024) 30

- 96. Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M.: Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453 (2023) 7
- 97. Xiao, S., Zhang, X., Meng, D., Wang, Q., Zhang, P., Zhang, B.: Knot forcing: Taming autoregressive video diffusion models for real-time infinite interactive portrait animation. arXiv preprint arXiv:2512.21734 (2025) 30
- 98. Yan, W., Zhang, Y., Abbeel, P., Srinivas, A.: Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157 (2021) 30
- 99. Yang, S., Huang, W., Chu, R., Xiao, Y., Zhao, Y., Wang, X., Li, M., Xie, E., Chen, Y., Lu, Y., et al.: Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622 (2025) 3, 4, 5, 7, 10, 13, 30, 31, 34, 39, 43, 44
- 100. Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al.: Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072 (2024) 30
- 101. Yesiltepe, H., Meral, T.H.S., Akan, A.K., Oktay, K., Yanardag, P.: Infinity-rope: Action-controllable infinite video generation emerges from autoregressive selfrollout. arXiv preprint arXiv:2511.20649 (2025) 30
- 102. Yi, J., Jang, W., Cho, P.H., Nam, J., Yoon, H., Kim, S.: Deep forcing: Trainingfree long video generation with deep sink and participative compression. arXiv preprint arXiv:2512.05081 (2025) 5, 7, 30
- 103. Yin, T., Gharbi, M., Park, T., Zhang, R., Shechtman, E., Durand, F., Freeman, B.: Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems 37, 47455–47487 (2024) 5, 10, 30
- 104. Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W.T., Park, T.: One-step diffusion with distribution matching distillation. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6613– 6623 (2024) 5, 10, 30
- 105. Yin, T., Zhang, Q., Zhang, R., Freeman, W.T., Durand, F., Shechtman, E., Huang, X.: From slow bidirectional to fast autoregressive video diffusion models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 22963–22974 (2025) 3, 5, 10, 30
- 106. Yu, Y., Wu, X., Hu, X., Hu, T., Sun, Y., Lyu, X., Wang, B., Ma, L., Ma, Y., Wang, Z., et al.: Videossm: Autoregressive long video generation with hybrid state-space memory. arXiv preprint arXiv:2512.04519 (2025) 30
- 107. Yuan, H., Chen, W., Cen, J., Yu, H., Liang, J., Chang, S., Lin, Z., Feng, T., Liu, P., Xing, J., et al.: Lumos-1: On autoregressive video generation from a unified model perspective. arXiv preprint arXiv:2507.08801 (2025) 30
- 108. Zhang, D.J., Wu, J.Z., Liu, J.W., Zhao, R., Ran, L., Gu, Y., Gao, D., Shou, M.Z.: Show-1: Marrying pixel and latent diffusion models for text-to-video generation. International Journal of Computer Vision 133(4), 1879–1893 (2025) 30
- 109. Zhang, L., Cai, S., Li, M., Wetzstein, G., Agrawala, M.: Frame context packing and drift prevention in next-frame-prediction video diffusion models. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems

(2025) 30

- 110. Zhang, S., Wang, J., Zhang, Y., Zhao, K., Yuan, H., Qin, Z., Wang, X., Zhao, D., Zhou, J.: I2vgen-xl: High-quality image-to-video synthesis via cascaded diffusion models. arXiv preprint arXiv:2311.04145 (2023) 30
- 111. Zhang, T., Bi, S., Hong, Y., Zhang, K., Luan, F., Yang, S., Sunkavalli, K., Freeman, W.T., Tan, H.: Test-time training done right. arXiv preprint arXiv:2505.23884 (2025) 3, 5, 30

- 112. Zhang, W., Feng, Y., Meng, F., You, D., Liu, Q.: Bridging the gap between training and inference for neural machine translation. arXiv preprint arXiv:1906.02448

(2019) 3

- 113. Zheng, D., Huang, Z., Liu, H., Zou, K., He, Y., Zhang, F., Zhang, Y., He, J., Zheng, W.S., Qiao, Y., Liu, Z.: VBench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755 (2025) 4, 7, 10, 13, 31
- 114. Zhou, J., Du, Y., Xu, X., Wang, L., Zhuang, Z., Zhang, Y., Li, S., Hu, X., Su, B., Chen, Y.c.: Videomemory: Toward consistent video generation via memory integration. arXiv preprint arXiv:2601.03655 (2026) 30

