# arXiv:2511.20123v2[cs.CV]1Mar2026

## ULTRAVICO: BREAKING EXTRAPOLATION LIMITS IN VIDEO DIFFUSION TRANSFORMERS

#### Min Zhao1,2 ∗, Hongzhou Zhu1,2 ∗, Yingze Wang1 ∗, Bokai Yan3, Jintao Zhang1,2, Guande He4, Ling Yang5, Chongxuan Li3, Jun Zhu1,2

‡

1Dept. of Comp. Sci. & Tech., BNRist Center, THU-Bosch ML Center, Tsinghua University. 2ShengShu. 3Gaoling School of Artificial Intelligence, Renmin University of China. 4The University of Texas at Austin. 5 Princeton University. gracezhao1997@gmail.com, zhuhz22@mails.tsinghua.edu.cn

ABSTRACT

Despite advances, video diffusion transformers still struggle to generalize beyond their training length, a challenge we term video length extrapolation. We identify two failure modes: model-specific periodic content repetition and a universal quality degradation. Prior works attempt to solve repetition via positional encodings, overlooking quality degradation and achieving only limited extrapolation. In this paper, we revisit this challenge from a more fundamental view—attention maps, which directly govern how context influences outputs. We identify that both failure modes arise from a unified cause: attention dispersion, where tokens beyond the training window dilute learned attention patterns. This leads to quality degradation and repetition emerges as a special case when this dispersion becomes structured into periodic attention patterns, induced by harmonic properties of positional encodings. Building on this insight, we propose UltraViCo, a training-free, plug-and-play method that suppresses attention for tokens beyond the training window via a constant decay factor. By jointly addressing both failure modes, we outperform a broad set of baselines largely across models and extrapolation ratios, pushing the extrapolation limit from 2× to 4×. Remarkably, it improves Dynamic Degree and Imaging Quality by 233% and 40.5% over the previous best method at 4× extrapolation. Furthermore, our method generalizes seamlessly to downstream tasks such as controllable video synthesis and editing. Project page is available at https://thu-ml.github.io/UltraViCo.github.io/.

1 INTRODUCTION

Building upon the expressive power of diffusion transformers (DiTs) (Bao et al., 2023; Peebles & Xie, 2023), recent advances in text-to-video (T2V) generation (Bao et al., 2024; Zheng et al., 2024b; Brooks et al., 2024; Wan et al., 2025; Kong et al., 2024; Hong et al., 2022) have enabled models to synthesize high-fidelity videos. However, these models are typically trained on a fixed maximum sequence length (e.g., 5 seconds) (Wan et al., 2025; Kong et al., 2024; Hong et al., 2022) and struggle to generate videos beyond their training length, a task we term video length extrapolation, which is critical for practical applications.

To investigate the core challenges of this task, we conduct experiments on a range of models and identify two failure modes: (i) a model-specific periodic content repetition, where short clips loop indefinitely in certain models; and (ii) a universal quality degradation, manifested as blurred spatial details and frozen temporal dynamics across all models. Both failures become increasingly severe as the extrapolation length grows. Prior work, such as RIFLEx (Zhao et al., 2025), tackles repetition from the perspective of positional encodings, while overlooking quality degradation and therefore achieving limited extrapolation. We contend, however, that positional encodings play only an indirect role by perturbing queries and keys to influence attention. In contrast, attention itself—directly aggregating contextual information to generate outputs—offers a more fundamental view.

∗⋆Equal contribution. ‡The corresponding author (dcszj@tsinghua.edu.cn).

- 3× extrapolation: Ours (top) vs. RIFLEx (bottom)
- 4× extrapolation: Ours (top) vs. RIFLEx (bottom)

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

(a) Extending T2V models up to 4×, where existing method yields nearly static, low-quality videos.

Controllable video generation

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

Masked video-to-video editing

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

(b) Generalization to downstream tasks at 3×. See more tasks in Appendix C.4.

- Figure 1: Visual results. UltraViCo achieves significant extrapolation improvement on (a) T2V models and (b) downstream tasks. See prompts and videos in supplementary materials.

Therefore, we revisit extrapolation failures through the lens of attention maps. Our systematic analysis of attention maps shows that both failure modes arise from a unified mechanism: attention dispersion. This occurs when new tokens beyond the training length dilute the learned attention patterns. This leads to quality degradation and repetition arises as a special case when dispersion becomes organized into periodic attention patterns. Specifically, this happens when positional encoding frequencies form harmonics, enabling the largest-amplitude frequency and its harmonics to accumulate amplitude and contribute substantially to the overall amplitude.

Building on this unified view, we propose Ultra-extrapolated Video via Attention Concentration (UltraViCo), a plug-and-play method that suppresses attention for tokens beyond the training window with a constant decay factor. This adjustment reallocates attention to reliable in-window context while naturally breaking periodic patterns, thus simultaneously addressing both failure modes. Notably, standard attention implementations encounter out-of-memory errors when modifying logits for long video sequences. We therefore develop a memory-efficient CUDA kernel that enables scalable applications on large video models.

To validate our approach, we conduct comprehensive evaluations on various T2V models (Kong et al., 2024; Yang et al., 2024; Wan et al., 2025) and extrapolation ratios, against a large family of baselines (Chen et al., 2023b; bloc97, 2023; Zhuo et al., 2024; Peng et al., 2023; Zhao et al., 2025). Experiments demonstrate that our method consistently surpasses all baselines in all settings by simultaneously addressing both failure modes. Notably, while prior methods collapse beyond 3× extrapolation and yield static videos, ours maintains fluid motion, effectively extending the practical limit from 2× to 4×. Remarkably, it improves Dynamic Degree and Imaging Quality by 233% and

40.5% over the previous best method at 4× extrapolation. Beyond this, our method also generalizes seamlessly to downstream tasks such as various controllable video synthesis and editing.

HunyuanVideo Wan

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

… …

Normal

length

Video of 129 frames Video of 81 frames

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

… … … …

3 ×

extra.

(a) Periodic content repetition and quality degradation. (b) Quality degradation.

80

80

- 0
- 1
- 2
- 3

| |HunyuanVideo<br><br>Wan2.1<br><br>[Figure 77]| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| |HunyuanVideo<br><br>Wan2.1<br><br>[Figure 78]| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

| |HunyuanVideo<br><br>Wan2.1| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

[Figure 79]

[Figure 80]

[Figure 81]

RepetitionCount

DynamicDegree

ImagingQuality

70

60

60

40

Variable

50

20

###### extra.

40

0

1× 2× 3× 4× 5×

1× 2× 3× 4× 5×

1× 2× 3× 4× 5×

(c) Both quality and repetition worsen as the extrapolation grows from 1× to 5×.

- Figure 2: Failure modes of video length extrapolation. Some models exhibit periodic content repetition, while quality degradation occurs universally. Both failure modes intensify with longer extrapolations. “extra.” denotes extrapolation. See Appendix C.1 for additional models.

- 2 PRELIMINARY

Attention mechanism with rotary position embedding. Modern video diffusion models are largely built on DiTs whose core is the attention mechanism (Vaswani et al., 2017; Li et al., 2025a). The input video is patched into L tokens, each projected into queries, keys, and values. To encode the position information, DiTs mainly adopt Rotary Position Embedding (RoPE) (Su et al., 2024), which injects position into queries and keys through complex rotations. Concretely, for each query or key vector x ∈ RD at position t, RoPE maps it to RD as

cos(ϕit) −sin(ϕit) sin(ϕit) cos(ϕit)

x2i x2i+1

fRoPE(x,t)i = Ri(t)

, i ∈ {0,...,D/2 − 1}. (1)

, Ri(t) =

Here, each frequency ϕi depends exponentially on i and is used to encode the (2i,2i+1) components of x. After RoPE, the queries and keys form matrices Q ∈ RL×D and K ∈ RL×D. Their interaction yields the attention logits S ∈ RL×L, which are normalized by the softmax function to obtain the attention scores P ∈ RL×L. These scores are then applied to the value matrix V ∈ RL×D

′

to produce the output O ∈ RL×D

′

:

S √

S = QK⊤, P = softmax(

#### ), O = PV . (2)

D

For videos with temporal and spatial axes, Multimodal RoPE (M-RoPE) (Wang et al., 2024a) partitions the dimension D = dT + dH + dW and encodes each subspace separately. Since we focus on temporal extrapolation, we consider only the temporal axis and denote dT as d for simplicity (see details in Appendix B.2).

Problem setting: video length extrapolation. Despite advances, DiT-based video generation models struggle to produce videos longer than their training duration. This task, known as video length extrapolation (Zhao et al., 2025), aims to adapt a pre-trained model to generate high-quality videos of a sequence length L′ that exceeds its training length L, with the extrapolation ratio defined as s = L′/L > 1. Notably, video length extrapolation targets the model’s intrinsic ability to generate longer sequences in a single forward generation, which is orthogonal to prior methods (Qiu et al., 2023; Wang et al., 2023; Kim et al., 2024; Wang et al., 2024c; Lu et al., 2024) that rely on inference-time modifications. See Appendix A for more related work.

- 3 METHOD

- 3.1 FAILURE MODES OF VIDEO LENGTH EXTRAPOLATION

In this section, we investigate the core challenges of video length extrapolation on a range of SOTA video diffusion transformers, including Wan (Wan et al., 2025), HunyuanVideo (Kong et al., 2024), and CogVideoX (Yang et al., 2024).

Qualitative results in Fig.2a and Fig.2b reveal two distinct failure modes. The first is a periodic content repetition, which occurs in certain models such as HunyuanVideo and CogVideoX. The second is a universal quality degradation, characterized by compromised spatial fidelity and temporal dynamics across all models. To further investigate their trends across extrapolation lengths, we perform a quantitative analysis on 10 prompts using metrics including Imaging Quality (Huang et al., 2024), Dynamic Degree (Huang et al., 2024), and Repetition Count. Fig. 2c confirms that both failures become more severe as the extrapolation factor increases.

These findings raise three critical questions: First, why does periodic content repetition only manifest in specific models? Second, what is the underlying cause of the universal quality degradation? Most importantly, is there a unified cause behind these two seemingly independent failure modes?

Existing work such as RIFLEx addresses only content repetition, neglecting quality degradation, which limits both model generalization and extrapolation capacity. While RIFLEx attributes repetition to positional encoding periodicity, we argue that positional encodings play only an indirect role by modulating queries and keys. Instead, as Eq. (2) shows, the attention map itself is fundamental, since it directly determines how context is aggregated. This motivates us to revisit extrapolation failures through attention analysis.

- 3.2 ATTENTION ANALYSIS OF THE CAUSE

In this section, we first focus on the specific issue of periodic content repetition (Sec. 3.2.1). Through an in-depth attention analysis of its underlying mechanism, we find, surprisingly, that the solution designed to resolve repetition also improves video quality. This key finding then allows us to understand the cause of the more universal problem of quality degradation (Sec. 3.2.2), and ultimately reveals the intrinsic connection between the two failure modes.

- 3.2.1 THE CAUSE OF CONTENT REPETITION: PERIODIC ATTENTION PATTERNS

Periodic attention induces output repetition. We analyze the cause of content repetition by inspecting the attention map P ∈ RL

′×L′ during 4× extrapolation, where L′ is the extrapolated sequence length (i.e., video features flattened into a 1D sequence). The entry at row i, column j of P, denoted Pij, is the attention score from query i to key j. As shown in Fig. 3a, the attention map of HunyuanVideo reveals two properties that jointly induce periodic outputs.

First, the map exhibits a distinct row-wise periodicity. Specifically, for any query at position i, its attention scores to key positions j and j+T are nearly identical: Pi,j ≈ Pi,j+T, where T corresponds to the observed repetition period in Sec. 3.1. As indicated in Fig. 3a, the blue and purple circles highlight nearly equal scores. Second, the map shows relative positional invariance: query–key pairs with the same relative displacement p yield approximately equal scores, Pi,j ≈ Pi+p,j+p. This RoPE-induced property appears as uniform values along diagonals and subdiagonals; for example, when p = T, the scores marked by the blue and green circles are nearly identical.

Combining these properties, we can derive that entire query rows also repeat periodically: Pi+T,j ≈ Pi,j, as shown by the green and purple circles. Thus, rows i and i + T retrieve nearly the same weighted information from the value V , leading to periodic outputs (see Appendix B.1 for details):

Oi+T =

L′−1

Pi+T,jVj ≈

j=0

L′−1

Pi,jVj = Oi. (3)

j=0

This periodicity is directly reflected in repeated content in pixel space. Larger extrapolation ratios traverse more periods, thus increasing repetition counts, which is consistent with our observations in

Model Attention maps Statistical row-wise attention analysis

Individual frequencies

Final composite attention

5.0

15

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

[Figure 82]

acos(ϕΔt+b)iii

10

2.5

S(Δt)±std

5

0.0

Hun.

0

−2.5

−5

ϕ0 ϕ1 ϕ2 ϕ3 ϕ4 ϕ5 ϕ6 ϕ7

−5.0

0 25 50 75 100 125

0 25 50 75 100 125

Temporal distance Δt

Temporal distance Δt

(a) Periodic attention: (b) Harmonic RoPE frequencies (ϕi/ϕN−1 ∈ N+) amplify the largest-amplitude Pi,j ≈ Pi,j+T frequency and its harmonics (dashed line), inducing periodic composite attention.

Individual frequencies

Final composite attention

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

10

[Figure 83]

acos(ϕΔt+b)iii

20

5

S(Δt)±std

0

###### Wan

0

−5

ϕ0 ϕ1 ϕ2 ϕ3 ϕ4 ϕ5 ϕ6…21

−10

0 20 40 60 80

0 20 40 60 80

Temporal distance Δt

Temporal distance Δt

(c) Non-periodic attention: (d) Inharmonic RoPE frequencies (ϕi/ϕN−1 ∈/ N+) disperse spectrum (dashed Pi,j ̸= Pi,j+T line), yielding non-periodicity in the final composite attention.

- Figure 3: Periodic attention patterns as cause of content repetition. Left: unlike Wan, HunyuanVideo exhibits row-wise periodic attention during 4× extrapolation, causing repeated outputs. Right: statistical row-wise attention can be expressed as a linear combination of trigonometric functions of RoPE frequencies, whose properties govern this periodicity. Hun. denotes HunyuanVideo.

Sec. 3.1. By contrast, the attention map of Wan (Fig. 3c) does not display such row-wise periodicity, and accordingly its outputs remain free of repetition.

Origin of periodic attention patterns. Next, we show that such model-specific row-wise periodicity originates from the RoPE frequencies. To reveal the core row-wise attention structure from noise, we construct a statistical row attention pattern S¯(∆t), which captures the relation between a query and keys at the same spatial location but ∆t latent frames apart. This is achieved by taking the expectation of the pre-softmax attention logits across all layers, heads, and query positions. As derived in Appendix B.3 (based on Eq. (2)), this quantity admits the following trigonometric decomposition:

S¯(∆t) =

d/2−1

ai cos(ϕi∆t + bi) + C, (4)

i=0

where {ϕi}d/i=02−1 are the RoPE frequencies defined in Sec. 2, and {ai}d/i=02−1,{bi}d/i=02−1,C are constants determined by the statistics of queries and keys from models, with bi typically close to zero. Visualizations of these frequency components for HunyuanVideo and Wan highlight a crucial difference (Fig. 3b,d, left). The periodicity of such a superposition is decided by the frequency relationships, as formalized in Proposition 1.

Proposition 1 (Period and Amplitude of Harmonics). For a function f(∆t) = Ni=0−1 ai cos(ϕi∆t), where ai > 0,ϕi > 0 and mini ϕi = ϕN−1, if and only if ∀i, ϕi/ϕN−1 ∈ N+ (i.e., they form

a set of harmonics), f(∆t) is periodic with period TN−1 = ϕ2π

. In this case, max∆t f(∆t) =

N−1

N−1 i=0 ai, whenever ∆t = mTN−1, m ∈ Z (i.e., whenever ∆t is at harmonic alignment positions).

We find that HunyuanVideo’s frequencies satisfy this harmonic condition in Proposition 1, allowing amplitude accumulation of the largest-amplitude frequency ϕ3 and its harmonics (i < 3) at harmonic alignment positions mT (dashed line in Fig. 3b), where m ∈ Z. This yields a dominant component that contributes 79.6% of the total amplitude, producing a strongly periodic composite attention pattern (Fig. 3b, right). A similar harmonic alignment is also observed in CogVideoX (Appendix B.6). In contrast, Wan’s frequencies are not harmonically aligned, resulting in a dispersed spectrum where no frequency dominates (largest 31.6%), and thus no clear periodicity emerges (Fig. 3d). Notably,

while the strict periodicity of HunyuanVideo is determined by the lowest frequency, its small amplitude and long period make it negligible; the observed periodicity T is effectively governed by the dominant frequency (see Appendix B.6).

In summary, our analysis establishes the causal chain: RoPE-induced frequency harmonics lead to periodic attention patterns, which in turn produce periodic output features and ultimately manifest as content repetition. To validate this, we mask tokens at harmonic alignment positions mT. Breaking these constructive interference points disrupts periodic attention and, as shown in Fig. 4a, effectively mitigates repetition.

Model Generated videos: baseline vs. intervention Attention maps: baseline vs. intervention

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

… …

Hun.

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

(a) Non-repetition and improved video quality after intervention (b) Attention focused centrally after intervention

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

… …

Wan

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

(c) Improved video quality after intervention (d) Attention focused centrally after intervention

- Figure 4: Fixing repetition reveals attention dispersion as the fundamental cause. Left: our intervention, initially targeting repetition, surprisingly enhances video quality in both models. Right: the shared mechanism is revealed, where the intervention refocuses diffuse baseline attention toward the central training window. This suggests attention dispersion as the unified cause.

- 3.2.2 THE CAUSE OF QUALITY DEGRADATION: ATTENTION DISPERSION

Surprisingly, we find the above repetition-resolving intervention also improves video quality across both models (Fig. 4a, c). This finding suggests a more profound hypothesis: content repetition and quality degradation may arise from a shared, fundamental underlying mechanism.

A comparison of attention maps shows our intervention consistently concentrates the initially diffuse attention (Fig. 4b, d). This occurs because masking the harmonic peaks forces a softmax renormalization, which sharpens the attention distribution by proportionally increasing the remaining scores. To further identify where this sharpened focus is most beneficial, we systematically masked different attention regions and found that concentrating attention within the original central training window yielded the strongest improvements (see details in Appendix B.7). This leads us to hypothesize that attention dispersion is the underlying issue. New tokens during extrapolation dilute the learned attention patterns within the original training window. This dispersion has two detrimental effects. Spatially, the model needs to consider far-away extrapolated frames, which makes it difficult to focus on fine details and results in visual blurriness. Temporally, taking these distant frames into account mixes local motion with unrelated movements, causing the video to appear static and unnatural. These effects are consistent with the quality degradation observed in Sec. 3.1.

To validate this hypothesis, we conduct a controlled experiment where we progressively mask attention scores for tokens outside the training window, thereby forcing the attention to concentrate centrally. The results, presented in Fig. 5, demonstrate a clear positive correlation: more concentrated attention (i.e., by increasing the proportion of masked out-of-window scores) consistently improves both the visual quality and motion dynamics of the generated video. This provides strong evidence that attention dispersion is the cause of quality degradation. Consequently, as the extrapolation ratio increases, attention becomes more dispersed, leading to worse quality, consistent with the observations in Sec. 3.1.

A unified view: periodic attention as a case of attention dispersion. Building upon the above analysis, we can unify both failure modes under a single perspective: attention dispersion is the fundamental cause of extrapolation failure, with periodic attention patterns representing a special

[Figure 104]

70

DynamicDegreeImagingQuality

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

65

60

55

50

0 20% 60% 100%

Focus degree (masking ratio)

[Figure 105]

60

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |

40

20

0

0 20% 60% 100%

Focus degree (masking ratio)

(a) Quantitative results.

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

highly focused (100% masking ratio)

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

moderately focused (60% masking ratio)

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

weakly focused (20% masking ratio)

(b) Qualitative results.

- Figure 5: Validation of attention dispersion as the cause of quality degradation. Both (a) quantitative and (b) qualitative results show that video quality improves monotonically as the degree of attention central focusing (i.e., the masking ratio of out-of-window scores) increases.

case. Specifically, when a RoPE frequency contributes substantially to the overall amplitude (e.g., due to harmonic alignment), it induces a strongly periodic attention pattern; otherwise, the model exhibits generic, non-periodic dispersion.

- 3.3 ULTRAVICO

Building on the above unified view, we propose Ultra-extrapolated Video via Attention Concentration (UltraViCo), a simple yet effective method that suppresses attention for tokens beyond the training window via a decay factor, thereby restoring the model’s focusing ability. To achieve this, we introduce a position-dependent decay factor λij applied to the original attention logits Sij, yielding the corrected attention Sij′ :

Sij′ = λij · Sij, where λij =

1, if |i − j| ≤ L/2 or Sij < 0, α, otherwise,

(5)

where α < 1 is a constant decay hyperparameter and L is the training length. Here, λij is set to be 1 for all pairs within the training window, preserving the model’s core learned dynamics. For out-of-

window tokens, only positive logits (Sij ≥ 0) are down-scaled because multiplying negative logits Sij < 0 by α < 1 can undesirably increase its value, while multiplying α > 1 or 1 for negative logits has a negligible effect. We also experimented with various decay strategies, such as linear decay, but found the constant form is sufficient, indicating that the key is distinguishing in-window from out-of-window tokens rather than the decay shape itself (see Sec. 4.2 for details).

However, in models showing periodic repetition (Sec. 3.2.1), harmonic alignment positions mT attract disproportionately high attention. Applying a uniform small decay α would overly suppress all out-of-window context, harming temporal consistency. To address this, we apply a stronger decay β < α specifically to these risky positions mT, while keeping α for other out-of-window tokens:

 

1, if |i − j| ≤ L/2 or Sij < 0, β, else if(i,j) ∈ Prisk, α, otherwise,

(6)

λij =



where Prisk = {(i,j)| mT − γ ≤ i − j ≤ mT + γ, m ∈ Z,γ ∈ N+ } denotes the set of positions within γ frames around the harmonic alignment positions mT and β < α < 1. This targeted adjustment reallocates attention to reliable in-window context while eliminating spurious periodic patterns, allowing UltraViCo to mitigate both failure modes simultaneously.

Efficient CUDA implementation. UltraViCo requires modifying attention logits, but standard PyTorch attention is infeasible for long sequences. At a 3× extrapolation (∼200K tokens for Hunyuan-

Table 1: Quantitative illustrative results on VBench for HunyuanVideo and Wan. For Wan, which does not exhibit content repetition, we omit the NoRepeat Score. Additional results for more extrapolation ratios and models are provided in Appendix C.3. Consist., Dyn., Qual., Over. and NoRe. denote Consistency, Dynamics, Quality, Overall and NoRepeat Score respectively. Normal. indicates the training length for reference.

Wan2.1-1.3B HunyuanVideo

Method

Consist.↑ Dyn.↑ Qual.↑ Over.↑ User↓ Consist.↑ NoRe.↑ Dyn.↑ Qual.↑ Over.↑ User↓ Normal. 0.9554 51 70.34 24.25 – 0.9786 – 71 69.31 26.81 –

- 3× extrapolation

PE 0.9419 6 56.28 18.53 3.82 0.9795 53.17 16 51.85 21.62 3.96 PI 0.9667 7 52.16 17.48 4.69 0.9787 90.23 1 46.30 21.29 4.91 NTK 0.9437 3 57.73 18.50 4.40 0.9802 84.80 24 53.11 22.14 3.74 YaRN 0.9676 5 53.46 17.53 4.71 0.9790 88.74 0 47.05 21.42 5.05 TASR 0.9434 6 57.41 18.48 4.47 0.9807 80.74 22 51.95 22.02 4.65 RIFLEx 0.9431 5 53.79 17.54 4.90 0.9823 73.97 17 50.57 21.22 4.67 Ours 0.944 46 62.43 23.21 1.01 0.9465 100.0 62 65.00 26.45 1.02

- 4× extrapolation

PE 0.9415 11 55.25 16.65 3.75 0.9891 31.41 14 47.12 17.61 3.70 PI 0.9711 12 50.44 16.34 4.87 0.9885 70.93 0 42.19 17.83 4.82 NTK 0.9477 11 55.37 16.09 4.24 0.9915 72.39 10 50.01 18.92 4.23 YaRN 0.9729 7 51.16 16.69 4.57 0.9877 62.87 1 41.37 18.53 5.03 TASR 0.9495 9 55.18 16.16 4.72 0.9911 51.28 14 46.81 18.47 4.51 RIFLEx 0.9453 10 51.05 15.83 4.84 0.9906 52.84 11 41.02 16.47 4.69 Ours 0.9484 47 59.36 21.61 1.01 0.9468 99.87 42 66.54 24.52 1.02

Video), for instance, materializing a 200K × 200K attention mask consumes over 80GB of memory in bf16, causing an immediate out-of-memory error. To address this, we integrate UltraViCo into Triton-based FlashAttention (Dao et al., 2022) and SageAttention (Zhang et al., 2024b), where the online-softmax formulation avoids explicit mask construction. This yields scalable, memoryefficient computation, enabling UltraViCo on large video models.

- 4 EXPERIMENTS

- 4.1 SETUP

Evaluation. We evaluate methods on three video diffusion models, including HunyuanVideo, Wan2.1-1.3B and CogVideoX-5B. Following RIFLEx, we use 100 prompts sampled from VBench (Huang et al., 2024). For quantitative evaluation, following RIFLEx, we adopt Imaging Quality (Quality), Dynamic Degree (Dynamics), and Overall Consistency (Overall) from VBench, along with the NoRepeat Score for models prone to content repetition. Notably, our NoRepeat Score is a variant of that in RIFLEx, tailored for multiple-repetition (see Appendix C.2 for details). Finally, we conduct a user study with 10 participants on 10 prompts, where users rank (User) the overall quality of videos across all methods. More details are provided in Appendix C.2.

Implementation Details. The decay factor α is set to 0.9 for Wan and HunyuanVideo at 3× and 4× extrapolation. For HunyuanVideo, we set γ = 4 for all ratios, and β = 0.6 at 3× and 0.8 at 4×. Our baseline configurations follow RIFLEx. Further details are provided in Appendix C.2.

- 4.2 RESULTS

Performance comparison. We compare a wide range of length extrapolation baselines on three SOTA models (Kong et al., 2024; Yang et al., 2024; Wan et al., 2025) across various extrapolation ratios, including PE (Zhao et al., 2025), PI (Chen et al., 2023b), NTK (bloc97, 2023), TASR (Zhuo et al., 2024), YaRN (Peng et al., 2023), and RIFLEx. Tab. 1 reports 3× and 4× results on HunyuanVideo and Wan, while Fig. 6 shows qualitative samples on HunyuanVideo. Results for additional ratios and models are provided in the Appendix C.3.

As shown in Tab. 1, our method consistently outperforms all baselines across models and extrapolation ratios, simultaneously improving video quality and eliminating content repetition. Specifi-

cally, PE suffers from severe repetition, reflected in low NoRepeat Scores. In contrast, our method achieves substantially higher scores, effectively removing repetition. Beyond repetition, unlike RIFLEx which targets only this issue, our method delivers broader gains in both visual quality and motion quality. For instance, it improves Dynamic Degree and Imaging Quality on HunyuanVideo by 233% and 40.5% over the previous best method at 4× extrapolation, respectively. Notably, on Wan beyond 3× extrapolation, while prior methods collapse and yield static videos (Dynamic Degree ≤ 12), our method restores fluid motion. By addressing both core failure modes, our method extends the extrapolation limit from 2× to 4×. These improvements are further corroborated by user rankings (Tab. 1) and qualitative visualizations (Fig. 6), which consistently confirm the superior quality of our generated videos over baselines.

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

PE ··· ···

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

PI ··· ···

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

NTK ··· ···

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

YaRN ··· ···

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

TASR ··· ···

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

RIFLEx ··· ···

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

###### Ours ··· ···

(a) 3× extrapolation (b) 4× extrapolation

- Figure 6: Qualitative results on HunyuanVideo. The baselines produce nearly static videos with poor visual quality, whereas our method achieves significantly better quality by addressing extrapolation failure modes. Additional qualitative results for other models are in Appendix C.4.

parabolic linear constant

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

- Figure 7: Ablation studies. Top row: different decay strategies have minor impact, suggesting simple constant decay suffices. Bottom row: small α harms consistency while large α offers limited gains. An intermediate value (α = 0.9) enhances quality while preserving consistency.

Ablation studies. We ablate the decay strategy and the decay factor α on Wan at 3× extrapolation. As shown in Fig. 7 (top), different decay strategies yield minor differences, indicating that simple constant decay suffices. As shown in Fig. 7 (bottom), strong decay harms consistency (i.e., the spare tire of the car disappears) while weak decay offers limited gains. An intermediate value (α = 0.9) enhances quality while preserving consistency. Further details are provided in Appendix C.2. A sensitivity analysis for α and β (Fig. 8) shows a stable trend: α ≥ 0.9 and β ≥ 0.6 improve visual quality and motion dynamics while keeping temporal consistency near baseline. We adopt α = 0.9 and β = 0.6 as robust defaults, with small adjustments possible (e.g., β = 0.8 for stronger consistency, α = 0.85 for better quality). Although larger α and β may introduce a mild reduction

in consistency, values above 0.94 remain visually stable, aligning with common long-video settings (e.g., Wan’s training-horizon consistency ≈ 0.95). See more metrics of α,β in Tab. 4, 5, 6, and Fig. 17.

Connection with other long-video generation methods. UltraViCo aims to extend the effective training window of video diffusion transformers and is therefore orthogonal to existing long-video generation techniques such as FreeNoise (Qiu et al., 2023), FIFO-Diffusion (Kim et al., 2024), and sliding-window. As demonstrated in Table 2, enlarging the context window via UltraViCo consistently improves the long-term temporal consistency of these methods, without negatively affecting other performance. In Table 2, all methods follow the same evaluation setup (6× extrapolation for 30-second videos on Wan), where UltraViCo extends the base model’s training window by 3×.

Generalization to downstream tasks. Our method enhances the model’s inherent ability to handle longer sequences, making it naturally applicable to downstream tasks. As shown in Fig. 1, based on VACE (Jiang et al., 2025b), UltraViCo enables 3× extrapolation in controllable generation and video editing. See Appendix C.4 for additional results.

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

(a) Illustration of the α sensitivity curve. (b) Illustration of the β sensitivity curve.

- Figure 8: Illustration of the hyperparameter sensitivity curve. (a) When α ≥ 0.9, motion dynamics improve while consistency stays stable; below 0.9, consistency drops sharply. (b) When β ≥ 0.6, dynamics remain high with comparable consistency; below 0.6, consistency degrades significantly.

Table 2: Application of UltraViCo on existing long-video methods. Method Consistency↑ Dynamics↑ Quality↑ Overall↑

Sliding Window 0.8478 56 62.94 23.57 + UltraViCo 0.9183 54 62.85 23.95

FreeNoise 0.9243 38 63.09 23.75 + UltraViCo 0.9431 41 62.12 23.92

FIFO-Diffusion 0.9131 53 61.31 23.81 + UltraViCo 0.9319 51 63.09 24.24

- 5 CONCLUSION

In this paper, we identify attention dispersion as the unified cause behind video length extrapolation failures. Based on this insight, we propose a training-free method that suppresses attention scores for tokens beyond training length. Experiments show that it significantly improves video quality, extending the practical extrapolation limit from 2× to 4×.

### ETHICS STATEMENT

This paper advances the field of video generation, while emphasizing the importance of responsible use to avoid potential negative societal impacts, such as the creation of misleading or harmful content.

ACKNOWLEDGEMENTS

This work was supported by Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (No. JYB2025XDXM101); National Natural Science Foundation of China (Nos. 62522609, 92470118); the Beijing Natural Science Foundation (No. L247030); and the fund for building world-class universities (disciplines) of Renmin University of China.

REFERENCES

Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 22669–22679, 2023.

Fan Bao, Chendong Xiang, Gang Yue, Guande He, Hongzhou Zhu, Kaiwen Zheng, Min Zhao, Shilong Liu, Yaole Wang, and Jun Zhu. Vidu: a highly consistent, dynamic and skilled text-tovideo generator with diffusion models. arXiv preprint arXiv:2405.04233, 2024.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets. NONE, 2023.

bloc97. NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation., 2023. URL https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_ scaled_rope_allows_llama_models_to_have/.

Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators. 2024.

Minghong Cai, Xiaodong Cun, Xiaoyu Li, Wenze Liu, Zhaoyang Zhang, Yong Zhang, Ying Shan, and Xiangyu Yue. Ditctrl: Exploring attention control in multi-modal diffusion transformer for tuning-free multi-prompt longer video generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 7763–7772, 2025.

Boyuan Chen, Diego Mart´ı Mons´o, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2024a.

Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao Weng, and Ying Shan. Videocrafter1: Open diffusion models for high-quality video generation, 2023a.

Haoxin Chen, Yong Zhang, Xiaodong Cun, Menghan Xia, Xintao Wang, Chao Weng, and Ying

Shan. Videocrafter2: Overcoming data limitations for high-quality video diffusion models, 2024b. Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window

of large language models via positional interpolation. arXiv preprint arXiv:2306.15595, 2023b.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. Flashattention: Fast and memoryefficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022.

Jianxiong Gao, Zhaoxi Chen, Xian Liu, Jianfeng Feng, Chenyang Si, Yanwei Fu, Yu Qiao, and Ziwei Liu. Longvie: Multimodal-guided controllable ultra-long video generation. arXiv preprint arXiv:2508.03694, 2025.

Yuchao Gu, Weijia Mao, and Mike Zheng Shou. Long-context autoregressive video modeling with next-frame prediction. arXiv preprint arXiv:2503.19325, 2025.

Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation. 2022.

Roberto Henschel, Levon Khachatryan, Hayk Poghosyan, Daniil Hayrapetyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 2568–2577, 2025.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Jonathan Ho, William Chan, Chitwan Saharia, Jay Whang, Ruiqi Gao, Alexey Gritsenko, Diederik P. Kingma, Ben Poole, Mohammad Norouzi, David J. Fleet, and Tim Salimans. Imagen video: High definition video generation with diffusion models. arXiv preprint arXiv: 2210.02303, 2022.

Wenyi Hong, Ming Ding, Wendi Zheng, Xinghan Liu, and Jie Tang. Cogvideo: Large-scale pretraining for text-to-video generation via transformers. arXiv preprint arXiv:2205.15868, 2022.

Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025.

Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21807–21818, 2024.

Jiaxiu Jiang, Wenbo Li, Jingjing Ren, Yuping Qiu, Yong Guo, Xiaogang Xu, Han Wu, and Wangmeng Zuo. Lovic: Efficient long video generation with context compression. arXiv preprint arXiv:2507.12952, 2025a.

Zeyinzi Jiang, Zhen Han, Chaojie Mao, Jingfeng Zhang, Yulin Pan, and Yu Liu. Vace: All-in-one video creation and editing. arXiv preprint arXiv:2503.07598, 2025b.

Jihwan Kim, Junoh Kang, Jinyoung Choi, and Bohyung Han. Fifo-diffusion: Generating infinite videos from text without training. Advances in Neural Information Processing Systems, 37: 89834–89868, 2024.

Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jos´e Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125, 2023.

Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024.

Xingyang Li, Muyang Li, Tianle Cai, Haocheng Xi, Shuo Yang, Yujun Lin, Lvmin Zhang, Songlin Yang, Jinbo Hu, Kelly Peng, et al. Radial attention: O(nlog n) sparse attention with energy decay for long video generation. arXiv preprint arXiv:2506.19852, 2025a.

Zhuoling Li, Hossein Rahmani, Qiuhong Ke, and Jun Liu. Longdiff: Training-free long video generation in one go. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 17789–17798, 2025b.

Yu Lu, Yuanzhi Liang, Linchao Zhu, and Yi Yang. Freelong: Training-free long video generation with spectralblend temporal attention. Advances in Neural Information Processing Systems, 37: 131434–131455, 2024.

Yue Ma, Yingqing He, Xiaodong Cun, Xintao Wang, Siran Chen, Xiu Li, and Qifeng Chen. Follow your pose: Pose-guided text-to-video generation using pose-free videos. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 4117–4125, 2024a.

Yue Ma, Hongyu Liu, Hongfa Wang, Heng Pan, Yingqing He, Junkun Yuan, Ailing Zeng, Chengfei Cai, Heung-Yeung Shum, Wei Liu, et al. Follow-your-emoji: Fine-controllable and expressive freestyle portrait animation. In SIGGRAPH Asia 2024 Conference Papers, pp. 1–12, 2024b.

Yue Ma, Kunyu Feng, Zhongyuan Hu, Xinyu Wang, Yucheng Wang, Mingzhe Zheng, Xuanhua He, Chenyang Zhu, Hongyu Liu, Yingqing He, et al. Controllable video generation: A survey. arXiv preprint arXiv:2507.16869, 2025.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205, 2023.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. Yarn: Efficient context window extension of large language models. International Conference on Learning Representations., 2023.

Adam Polyak, Amit Zohar, Andrew Brown, Andros Tjandra, Animesh Sinha, Ann Lee, Apoorv Vyas, Bowen Shi, Chih-Yao Ma, Ching-Yao Chuang, David Yan, Dhruv Choudhary, Dingkang Wang, Geet Sethi, Guan Pang, Haoyu Ma, Ishan Misra, Ji Hou, Jialiang Wang, Kiran Jagadeesh, Kunpeng Li, Luxin Zhang, Mannat Singh, Mary Williamson, Matt Le, Matthew Yu, Mitesh Kumar Singh, Peizhao Zhang, Peter Vajda, Quentin Duval, Rohit Girdhar, Roshan Sumbaly, Sai Saketh Rambhatla, Sam Tsai, Samaneh Azadi, Samyak Datta, Sanyuan Chen, Sean Bell, Sharadh Ramaswamy, Shelly Sheynin, Siddharth Bhattacharya, Simran Motwani, Tao Xu, Tianhe Li, Tingbo Hou, Wei-Ning Hsu, Xi Yin, Xiaoliang Dai, Yaniv Taigman, Yaqiao Luo, YenCheng Liu, Yi-Chiao Wu, Yue Zhao, Yuval Kirstain, Zecheng He, Zijian He, Albert Pumarola, Ali Thabet, Artsiom Sanakoyeu, Arun Mallya, Baishan Guo, Boris Araya, Breena Kerr, Carleigh Wood, Ce Liu, Cen Peng, Dimitry Vengertsev, Edgar Schonfeld, Elliot Blanchard, Felix Juefei-Xu, Fraylie Nord, Jeff Liang, John Hoffman, Jonas Kohler, Kaolin Fire, Karthik Sivakumar, Lawrence Chen, Licheng Yu, Luya Gao, Markos Georgopoulos, Rashel Moritz, Sara K. Sampson, Shikai Li, Simone Parmeggiani, Steve Fine, Tara Fowler, Vladan Petrovic, and Yuming Du. Movie gen: A cast of media foundation models. arXiv preprint arXiv: 2410.13720, 2024.

Ofir Press, Noah A Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. arXiv preprint arXiv:2108.12409, 2021.

Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, and Ziwei Liu. Freenoise: Tuning-free longer video diffusion via noise rescheduling. arXiv preprint arXiv:2310.15169, 2023.

Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. arXiv preprint arXiv:2011.13456, 2020.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative multimodal models are in-context learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14398–14409, 2024.

Jiangtong Tan, Hu Yu, Jie Huang, Jie Xiao, and Feng Zhao. Freepca: Integrating consistency information across long-short frames in training-free long video generation via principal component analysis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 27979– 27988, 2025.

Zhenxiong Tan, Xingyi Yang, Songhua Liu, and Xinchao Wang. Video-infinity: Distributed long

video generation. arXiv preprint arXiv:2406.16260, 2024. Genmo Team. Mochi 1. https://github.com/genmoai/models, 2024a. The FastVideo Team. Fastvideo: A unified framework for accelerated video generation, April 2024b.

URL https://github.com/hao-ai-lab/FastVideo.

Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Fu-Yun Wang, Wenshuo Chen, Guanglu Song, Han-Jia Ye, Yu Liu, and Hongsheng Li. Gen-l-video: Multi-text to long video generation via temporal co-denoising. arXiv preprint arXiv:2305.18264, 2023.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024a.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need.

- arXiv preprint arXiv:2409.18869, 2024b.

Yuqing Wang, Tianwei Xiong, Daquan Zhou, Zhijie Lin, Yang Zhao, Bingyi Kang, Jiashi Feng, and Xihui Liu. Loong: Generating minute-level long videos with autoregressive language models.

- arXiv preprint arXiv:2410.02757, 2024c.

Chenfei Wu, Lun Huang, Qianxi Zhang, Binyang Li, Lei Ji, Fan Yang, Guillermo Sapiro, and Nan Duan. Godiva: Generating open-domain videos from natural descriptions. arXiv preprint arXiv:2104.14806, 2021.

Chenfei Wu, Jian Liang, Lei Ji, Fan Yang, Yuejian Fang, Daxin Jiang, and Nan Duan. N¨uwa: Visual synthesis pre-training for neural visual world creation. In European conference on computer vision, pp. 720–736. Springer, 2022.

Yecheng Wu, Zhuoyang Zhang, Junyu Chen, Haotian Tang, Dacheng Li, Yunhao Fang, Ligeng Zhu, Enze Xie, Hongxu Yin, Li Yi, et al. Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429, 2024.

Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. Dynamicrafter: Animating open-domain images with video diffusion priors. 2023.

Feihong Yan, Peiru Wang, Yao Zhu, Kaiyu Pang, Qingyan Wei, Huiqi Li, and Linfeng Zhang. Generation then reconstruction: Accelerating masked autoregressive models via two-stage sampling. arXiv preprint arXiv:2510.17171, 2025a.

Feihong Yan, Qingyan Wei, Jiayi Tang, Jiajun Li, Yulin Wang, Xuming Hu, Huiqi Li, and Linfeng Zhang. Lazymar: Accelerating masked autoregressive models via feature caching. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 15552–15561, October 2025b.

Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157, 2021.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. arXiv preprint arXiv:2408.06072, 2024.

Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 22963–22974, 2025.

Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, and Jianfei Chen. Sageattention2: Efficient attention with thorough outlier smoothing and per-thread int4 quantization. arXiv preprint arXiv:2411.10958, 2024a.

Jintao Zhang, Jia Wei, Haofeng Huang, Pengle Zhang, Jun Zhu, and Jianfei Chen. Sageattention: Accurate 8-bit attention for plug-and-play inference acceleration. arXiv preprint arXiv:2410.02367, 2024b.

Min Zhao, Fan Bao, Chongxuan Li, and Jun Zhu. Egsde: Unpaired image-to-image translation via energy-guided stochastic differential equations. Advances in Neural Information Processing Systems, 35:3609–3623, 2022.

Min Zhao, Rongzhen Wang, Fan Bao, Chongxuan Li, and Jun Zhu. Controlvideo: Adding conditional control for one shot text-to-video editing. arXiv preprint arXiv:2305.17098, 2(3), 2023.

Min Zhao, Hongzhou Zhu, Chendong Xiang, Kaiwen Zheng, Chongxuan Li, and Jun Zhu. Identifying and solving conditional image leakage in image-to-video diffusion model. Advances in Neural Information Processing Systems, 37:30300–30326, 2024.

Min Zhao, Guande He, Yixiao Chen, Hongzhou Zhu, Chongxuan Li, and Jun Zhu. Riflex: A free lunch for length extrapolation in video diffusion transformers. arXiv preprint arXiv:2502.15894, 2025.

Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv:2412.20404, 2024a.

Zangwei Zheng, Xiangyu Peng, Tianji Yang, Chenhui Shen, Shenggui Li, Hongxin Liu, Yukun Zhou, Tianyi Li, and Yang You. Open-sora: Democratizing efficient video production for all. arXiv preprint arXiv: 2412.20404, 2024b.

Yuan Zhou, Qiuyue Wang, Yuxuan Cai, and Huan Yang. Allegro: Open the black box of commercial-level video generation model. arXiv preprint arXiv: 2410.15458, 2024.

Le Zhuo, Ruoyi Du, Han Xiao, Yangguang Li, Dongyang Liu, Rongjie Huang, Wenze Liu, Lirui Zhao, Fu-Yun Wang, Zhanyu Ma, et al. Lumina-next: Making lumina-t2x stronger and faster with next-dit. Advances in Neural Information Processing Systems., 2024.

USE OF LARGE LANGUAGE MODELS

We used a large language model solely to assist in polishing English writing and improving clarity. All research ideas, experiments, results, and interpretations are entirely our own.

- A RELATED WORK

Text-to-video Diffusion Transformers. Beyond autoregressive approaches Yan et al. (2025a;b), the recent advances in text-to-video generation have been primarily driven by diffusion models (Ho et al., 2020; Song et al., 2020; Ho et al., 2022; He et al., 2022; Zhao et al., 2022; 2023; Blattmann

- et al., 2023; Xing et al., 2023; Chen et al., 2023a; Zhao et al., 2024; Polyak et al., 2024; Zhou et al.,

- 2024; Team, 2024a; Chen et al., 2024b; Ma et al., 2024a;b; 2025). With the development of diffusion transformers (DiTs) (Bao et al., 2023; Peebles & Xie, 2023), DiT-based text-to-video diffusion models have achieved remarkable performance, such as Sora (Brooks et al., 2024), Vidu (Bao et al., 2024), CogVideoX (Yang et al., 2024) and Open-Sora (Zheng et al., 2024a). Although achieving high quality, leading models are trained only on a fixed maximum sequence length, limiting longterm capacity. During video length extrapolation, they suffer from repetition or quality degradation, underscoring the need for length extrapolation.

Length Extrapolation in Transformers. The goal of length extrapolation is to enable transformers to generate sequences longer than those seen during training in a single forward (Press et al., 2021). This is typically achieved by modifying positional encodings. For example, position interpolation (PI) (Chen et al., 2023b) improves performance by interpolating the frequencies in RoPE so that they remain within the training range even under extrapolation. NTK (bloc97, 2023),

YaRN (Peng et al., 2023), and Time-aware Scaled RoPE (TASR) (Zhuo et al., 2024) combine interpolation with direct extrapolation, incorporating adjustments along the token dimension, denoising timesteps, and other factors to achieve better results. However, these methods perform poorly on image and video DiTs, often leading to content collapse or repetition. RIFLEx (Zhao et al., 2025) mitigates repetition by identifying and attenuating the intrinsic RoPE frequency, yet it still suffers from degraded visual quality. In contrast, our method effectively addresses both content repetition and quality degradation.

Long Video Generation. There also exist many approaches to long video generation (Qiu et al., 2023; Wang et al., 2023; Henschel et al., 2025; Kim et al., 2024; Tan et al., 2024; Yin et al., 2025; Wang et al., 2024c; Cai et al., 2025; Li et al., 2025b; Lu et al., 2024; Tan et al., 2025; Jiang et al.,

- 2025a; Gao et al., 2025; Gu et al., 2025), most of which intervene in the diffusion inference process. For instance, FreeNoise (Qiu et al., 2023) enhances temporal consistency via noise initialization, FIFO-Diffusion (Kim et al., 2024) feeds frames sequentially into a denoising window of training length, and Video-Infinity (Tan et al., 2024) exploits distributed computation to scale up video length. While effective for generating long videos, these methods are orthogonal to our length extrapolation strategy, which extends the intrinsic capacity of DiTs to longer sequences and can be readily integrated with them.

In addition to diffusion-based approaches to long video generation, alternative modeling paradigms such as autoregressive methods (Wu et al., 2021; Yan et al., 2021; Hong et al., 2022; Wu et al., 2022; Kondratyuk et al., 2023; Wu et al., 2024; Sun et al., 2024; Wang et al., 2024b) and diffusion forcing (Chen et al., 2024a; Huang et al., 2025; Teng et al., 2025) are also capable of generating long videos. Although our method is designed for diffusion models, it may also offer insights into length extrapolation for these alternative paradigms.

- B MORE DETAILS OF OUR METHOD

- B.1 DERIVATION OF THE PERIODIC OUTPUTS

In this section, we present a formal derivation of Eq. (3). Specifically, the attention score matrix P ∈ RL

′×L′ satisfies the following properties up to negligible error:

- Prop.1 (Row-wise periodicity): Pi,j = Pi,j+T,∀i ∈ {0,...,L′ − 1},j ∈ {0,...,L′ − T − 1}, where T ∈ N+ corresponds to the observed repetition period in Sec. 3.1.
- Prop.2 (Relative positional invariance): Pi,j = Pi+p,j+p,∀i ∈ {0,...,L′−p−1},j ∈ {0,...,L′− p−1}, where p ∈ N+ is the relative displacement. In the ffollowing derivation we instantiate p = T.

On basis of the above properties, we derive the periodicity of the attention scores and outputs as follows. ∀i ∈ {0,...,L′ − T − 1},

L′−1 j=0

Pi+T,jVj (7)

Oi+T =

L′−T−1 j=0

L′−1 j=L′−T

##### Pi+T,jVj (8)

=

Pi+T,jVj +

- Prop.1=

L′−T−1 j=0

Pi+T,j+TVj +

L′−1 j=L′−T

Pi+T,jVj (9)

- Prop.2=

L′−T−1 j=0

L′−1 j=L′−T

Pi,j−TVj (10) Prop.1=

Pi,jVj +

L′−1 j=L′−T

L′−T−1 j=0

Pi,jVj (11)

Pi,jVj +

L′−1 j=0

Pi,jVj (12) = Oi. (13)

=

- B.2 DETAILS OF THE MULTIMODAL ROTARY POSITION EMBEDDING

In this section, we provide the details of the Multimodal RoPE (M-RoPE) (Wang et al., 2024a) introduced in Sec. 2. Specifically, for a token at position (t,h,w), the input vector x ∈ RD is divided into three subspaces of dimensions dT ,dH,dW, respectively assigned to temporal, height, and width encodings. Each subspace is modulated by its own frequency series {ϕTi }d

T −1

i=0 ,{ϕHi }d

T +dH−1

i=dT ,{ϕWi }Di=−d1T +dH. Concretely, we define fRoPE(x,t,h,w)i = Riα(pα)

x2i x2i+1

, Riα(pα) =

cos(ϕαi pα) −sin(ϕαi pα) sin(ϕαi pα) cos(ϕαi pα)

, (14)

where α ∈ {T ,H,W} indexes the temporal, height, and width dimensions with corresponding positions pα ∈ {t,h,w} and frequency components {ϕαi }. The index ranges are

i ∈

 



{0,...,dT /2 − 1}, α = T , {dt/2,...,dT /2 + dH/2 − 1}, α = H, {dT /2 + dH/2,...,D/2 − 1}, α = W.

(15)

After M-RoPE encoding, the queries and keys form Q ∈ RL

′×D and K ∈ RL

′×D. As in Eq. (2), they produce the attention logits matrix S ∈ RL

′×L′, where the attention logit between the query at

(t,h,w), denoted q(t,h,w), and the key at (t + ∆t,h + ∆h,w + ∆w), denoted k(t+∆t,h+∆h,w+∆w), expands explicitly as:

S(t,h,w),(t+∆t,h+∆h,w+∆w) =

dT /2−1

i=0

q((2t,h,wi:2i+1)) ⊤RiT (∆t)k((2t+∆i:2it,h+1)+∆h,w+∆w)+

dT /2+dH/2−1

i=dT /2

q((2t,h,wi:2i+1)) ⊤RiH(∆h)k((2t+∆i:2it,h+1)+∆h,w+∆w)+

D/2−1

i=dT /2+dH/2

q((2t,h,wi:2i+1)) ⊤RiW(∆w)k((2t+∆i:2it,h+1)+∆h,w+∆w) (16)

=

dT /2−1

i=0

λ(1i) cos(ϕTi ∆t) + λ(2i) sin(ϕTi ∆t) +

dT /2+dH/2−1

i=dT /2

λ(1i) cos(ϕHi ∆h) + λ(2i) sin(ϕHi ∆h) +

D/2−1

i=dT /2+dH/2

λ(1i) cos(ϕWi ∆w) + λ(2i) sin(ϕWi ∆w) , (17)

where

- λ(1i) = q((2t,h,wi) )k((2t+∆i) t,h+∆h,w+∆w) + q((2t,h,wi+1))k((2t+∆i+1)t,h+∆h,w+∆w), (18)
- λ(2i) = q((2t,h,wi+1))k((2t+∆i) t,h+∆h,w+∆w) − q((2t,h,wi) )k((2t+∆i+1)t,h+∆h,w+∆w). (19)

- B.3 DERIVATION OF THE STATISTICAL ATTENTION PATTERN S¯(∆t)

In this section, we present the derivation of Eq. (4) in Sec. 3.2.1. We investigate the row-wise pattern of attention logits by examining the expectation of the attention logits between queries and keys at relative temporal distance ∆t (i.e., E S(t,h,w),(t+∆t,h,w) )1. This expectation is taken across attention layers, heads, and query positions. In Appendix B.4, we further show that when the true variance is taken into account, the actual attention logits still follow the same patterns as indicated by this expectation.

1Strictly speaking, the analysis should target S(t,h,w),(t+∆t,h+∆h,w+∆w) for all ∆h, ∆w, but as the phenomena are similar across ∆h, ∆w, we focus on S(t,h,w),(t+∆t,h,w) for simplicity.

Specifically, on basis of the formula of M-RoPE (i.e., Eq. (16)), the target expectation is given by2

dT /2−1

q((2t,h,wi:2i+1)) ⊤RiT (∆t)k((2t+∆i:2it,h,w+1) )+

Et,h,w S(t,h,w),(t+∆t,h,w) = Et,h,w

i=0

dT /2+dH/2−1

D/2−1

q((2t,h,wi:2i+1)) ⊤RiH(0)k((2t+∆i:2it,h,w+1) ) +

q((2t,h,wi:2i+1)) ⊤RiW(0)k((2t+∆i:2it,h,w+1) ) (20)

i=dT /2

i=dT /2+dH/2

D/2−1

dT /2−1

E1(i), (21)

E1(i) cos ϕTi ∆t + E2(i) sin ϕTi ∆t +

=

i=0

i=dT /2

where

- E1(i) = Et,h,w q((2t,h,wi) )k((2t+∆i) t,h,w) + q((2t,h,wi+1))k((2t+∆i+1)t,h,w) , (22)
- E2(i) = Et,h,w q((2t,h,wi+1))k((2t+∆i) t,h,w) − q((2t,h,wi) )k((2t+∆i+1)t,h,w) . (23)

In practice, though the integrands of these expectations are actually functions of ∆t, the empirical statistics in Fig. 9 (col. 1) indicate that their variances with respect to ∆t are negligible. Hence, we

approximate E1(i) and E2(i) as constants up to negligible error, which is defined by

- E1(i) ≈ Et,h,w,∆t q((2t,h,wi) )k((2t+∆i) t,h,w) + q((2t,h,wi+1))k((2t+∆i+1)t,h,w) =: Eˆ1(i), (24)
- E2(i) ≈ Et,h,w,∆t q((2t,h,wi+1))k((2t+∆i) t,h,w) − q((2t,h,wi) )k((2t+∆i+1)t,h,w) =: Eˆ2(i). (25)

By substituting these two expressions into Eq. (22) and Eq. (23), the expected attention logits can be well approximated as S¯(∆t), where

dT /2−1

D/2−1

E ˆ1(i) cos ϕTi ∆t + Eˆ2(i) sin ϕTi ∆t +

Eˆ1(i). (26)

S¯(∆t) =

i=0

i=dT /2

To simplify the expression, we employ the auxiliary angle formula to rewrite the two trigonometric functions as one, i.e.,

dT /2−1

S¯(∆t) =

i=0

ai cos(ϕi∆t + bi) + C, (27)

2

2

,bi = atan2(−Eˆ2(i),Eˆ1(i)). Interestingly, as shown in Fig. 9

+ E ˆ2(i)

where ai = E ˆ1(i)

(col. 2), Eˆ2(i) remains consistently close to zero, which in turn makes bi nearly vanish (for example, b0 is 0.039 for HunyuanVideo). This observation allows us to apply Proposition 1 in Sec. 3.2.1 up

to an error of negligible magnitude. Detailed statistical data for Eˆ1(i),Eˆ2(i),ai,bi are shown in Fig. 9 (col. 2, 3, 4).

- B.4 CONSISTENCY OF ACTUAL ATTENTION PATTERN WITH S¯(∆t)

In this section, we investigate the actual attention scores under the true variance, demonstrating that they preserve the same characteristics as the averaged values described in Sec. 3.2.1. As shown in Fig. 10, when the standard deviation over attention layers, heads, and query positions is incorporated into the mean, the attention logits of HunyuanVideo still exhibit clear periodicity at their peaks, whereas those of Wan2.1 remain non-periodic. Therefore, the conclusions drawn in Sec. 3.2.1 from the mean-based analysis hold with strong generality in practice.

2For brevity, we omit layer and head indices in the expectation notation.

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

(a) Statistics of HunyuanVideo.

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

(b) Statistics of Wan.

- Figure 9: Statistics of attention logits in HunyuanVideo and Wan. The variances of E1(i),E2(i) with respect to ∆t (col. 1) are negligible compared to their expectations (col. 2), making the approximation in Eq. (24), Eq. (25) accurate. The bias angles bi (col. 4) are close to zero, except for b9 and b15 in Wan whose impact is negligible since the corresponding a9,a15 are near zero (col. 3).

[Figure 200]

[Figure 201]

- Figure 10: Attention logits under actual variance. Even with standard deviation across layers, heads, and query positions, HunyuanVideo retains clear periodic peaks while Wan 2.1 remains nonperiodic, confirming the general validity of the mean-based analysis in Sec. 3.2.1.

- B.5 PROOF OF PROPOSITION 1

Proposition 1 is well-known in harmonic analysis and signal processing, and we provide the proof here only for completeness.

Proof. Sufficiency. If ϕi/ϕN−1 ∈ N+ for all i, write ϕi = kiϕN−1 with ki ∈ N+. Let TN−1 = 2π/ϕN−1. Then for each i,

cos ϕi(∆t + TN−1) = cos kiϕN−1∆t + 2πki = cos(ϕi∆t), ∀∆t ∈ R, (28) so f(∆t + TN−1) = f(∆t), ∀∆t ∈ R. Hence TN−1 is a period of f. Necessity. Suppose TN−1 = 2π/ϕN−1 is a period of f. Then for all ∆t,

N−1

ai cos(ϕi∆t + ϕiTN−1) − cos(ϕi∆t) . (29)

0 = f(∆t + TN−1) − f(∆t) =

i=0

Using cos(x + y) − cosx = (cosy − 1)cosx − siny sinx,

N−1

ai (cos(ϕiTN−1) − 1)cos(ϕi∆t) − sin(ϕiTN−1)sin(ϕi∆t) , ∀∆t ∈ R. (30)

0 =

i=0

Model Attention maps Statistical row attention analysis

Individual frequencies

Final composite attention

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

[Figure 202]

10

acos(ϕΔt+b)iii

2

S(Δt)

0

5

Hun.

−2

0

ϕ0 ϕ1 ϕ2 ϕ3 ϕ4 ϕ5 ϕ6 ϕ7

0 10 20 30 40 50

0 10 20 30 40 50

Temporal distance Δt

Temporal distance Δt

(a) Periodic attention: (b) Approximately harmonic RoPE frequencies (ϕ0/ϕ1 ≈ N+) amplify the largest Pi,j ≈ Pi,j+T amplitude ϕ1 (dashed line), inducing approximately periodic composite attention.

- Figure 11: Periodic attention patterns of CogVideoX. The RoPE frequencies of CogVideoX approximately satisfy the harmonic condition, which amplifies the largest-amplitude component and thereby induces periodic attention patterns.

The family {cos(ϕi·),sin(ϕi·)}i with distinct positive ϕi is linearly independent over R (e.g., via independence of e±iϕ

it). Hence for each i,

cos(ϕiTN−1) − 1 = 0, sin(ϕiTN−1) = 0, (31) so ϕiTN−1 ∈ 2πZ. Substituting TN−1 = 2π/ϕN−1 yields

ϕi ϕN−1 ∈ N+, (32)

- as all ϕi > 0.

| |
|---|

- B.6 REMARKS ON PROPOSITION 1

Relaxed conditions under which the proposition holds approximately. Although the strict condition for forming harmonics in Proposition 1 is ϕi/ϕN−1 ∈ N+, in this section we highlight approximate conditions that can likewise induce a dominant frequency leading to content repetition in videos. Specifically, if ϕi/ϕN−1 is sufficiently close to an integer, constructive amplification can still occur for small |t| (e.g., |t| ≤ 2TN−1). For example, for CogVideoX, the ratio of the first two frequencies is ϕ0/ϕ1 = 3.16, which is close to the integer 3, thereby producing a dominant component that accounts for 50.80% of the total amplitude. This gives rise to an approximately periodic composite attention pattern (Fig. 11), which in turn leads to content repetition (Fig. 13, right).

Remarks on the strict period of HunyuanVideo. We herein examine the strict periodicity of HunyuanVideo. Strictly speaking, its fundamental frequency is ϕ7, with ratios ϕi/ϕ7 = 27−i,i ∈ {0,...,7}. According to Proposition 1, the theoretical period of S¯(∆t) is T7 = 2ϕπ

7

. However,

- as shown in Fig. 9a (col. 3), the amplification contributed by ϕ7 is very small, accounting for only 6.677%, which makes its impact negligible. Moreover, its period of 804 is far larger than the extrapolation length (e.g., 132 at 4× extrapolation), rendering the variation of the corresponding component almost imperceptible within this range. The same reasoning applies to ϕi for i ∈ {4,5,6}. Consequently, our analysis focuses on ϕi with i ∈ {0,1,2,3}, whose single-frequency contributions are both large enough in amplitude and sufficiently oscillatory to shape S¯(∆t).

- B.7 NECESSITY OF CONCENTRATING ON THE TRAINING WINDOW

In this section, we provide detailed experimental evidence supporting the discussion in Sec. 3.2.2 on where sharpened attention focus is most beneficial. Specifically, on Wan with extrapolation ratio

s = 3, we test four strategies for sharpening attention: concentrating on the leading 1s of each row, the trailing 1s, the training window, and the top–1s tokens according to the original attention scores. As shown in Fig. 12, concentrating on the leading or trailing 1s of each row causes the video to collapse, while top–1s yields poor visual quality with little dynamics. In contrast, restricting attention to the training window leads to the most significant improvement in video quality.

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

concentrating on the leading segment concentrating on the trailing segment

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

concentrating on top - tokens concentrating on the training window

- Figure 12: Comparison of attention concentration strategies on Wan at s = 3. Concentrating on the leading or trailing 1s of each row collapses the video, and top–1s yields poor quality with little dynamics. Restricting attention to the training window proves most effective.

- C MORE DETAILS OF EXPERIMENTS

- C.1 FAILURE MODES OF COGVIDEOX

In this section, we present the manifestation of the failure modes of video length extrapolation as discussed in Sec. 3.1 on an additional model, CogVideoX. As shown in Fig. 13, when extrapolated to three times the normal training length, the generated videos exhibit a sharp decline in both dynamic degree and visual quality, along with noticeable content repetition.

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

… …

normal length 3× extrapolation

Figure 13: Failure modes of CogVideoX under 3× extrapolation. The generated videos show degraded visual quality, reduced dynamics, and clear content repetition, consistent with the failure modes discussed in Sec. 3.1.

- C.2 MORE IMPLEMENTATION DETAILS In this section, we provide further details of Sec. 4.2.

The implementation of NoRepeat Score. The NoRepeat Score implemented in RIFLEx (Zhao et al., 2025) is only applicable when the content repeats once, which makes it unsuitable for longer extrapolation tasks. We therefore modify it accordingly. Specifically, the computation of the NoRepeat Score consists of two steps: static-video filtering and repeated-frame ratio calculation. In the first step, we uniformly sample 8 frames across the video; if the mean pairwise L2 distance among

- them falls below a threshold, the video is considered static and discarded. This prevents completely static videos from interfering with subsequent repetition detection. In the second step, we measure the ratio of repeated frames to the total frame count, which defines the NoRepeat Score. Following

RIFLEx, we first search around the dominant-frequency period for the frame with the minimal L2 distance to the first frame. This frame is then taken as the start of a candidate repeated sequence. We

- then compare each frame in this candidate sequence with the corresponding frame at the beginning

of the video; frames whose L2 distance is below the threshold are counted as repetitions. Empirically, a threshold of 55 was found to align better with human perception and was consequently applied to both steps. Finally, we report the mean NoRepeat Score across all videos as the final result. The detailed implementation code is included in the supplementary material.

The implementation of RIFLEx and UltraViCo on Wan. Since Wan does not exhibit content repetition, it is not applicable to determine the dominant frequency from the repetition period as per-

formed in Zhao et al. (2025). Instead, following Sec. 3.2.1, we take the largest-amplitude frequency ϕ0 as the dominant frequency.

For UltraViCo, the first frame’s decay factor is set negative to fix its blurring. We hypothesize that this is caused by the causal design of the video VAE, where the first frame is encoded independently and without temporal compression. As a result, it exhibits different statistical properties from subsequent frames and becomes more sensitive to perturbations.

Details of the ablation study. Herein, we detail the setup of the ablation study in Sec. 4.2. Specifically, as shown in Fig. 7 (top), we compare three decay strategies—parabolic, linear, and constant. The parabolic strategy takes the following form:

1, if |i − j| ≤ L/2 or Sij < 0, α1(|i − j|/L′)2 + α2(1 − (|i − j|/L′)2), otherwise,

Sij′ = λij·Sij, where λij =

whereas the linear strategy takes the following form:

Sij′ = λij · Sij, where λij =

and the constant strategy is

1, if |i − j| ≤ L/2 or Sij < 0, α1|i − j|/L′ + α2(1 − |i − j|/L′), otherwise,

(33)

(34)

Sij′ = λij · Sij, where λij =

1, if |i − j| ≤ L/2 or Sij < 0, α, otherwise.

(35)

We set α = 0.9 for the constant strategy, and α1 = 0.85,α2 = 0.95 for the parabolic and the linear strategies. As shown in Fig. 7 (top), parabolic, linear, and constant decay yield only minor differences, indicating that the key is distinguishing in-window from out-of-window tokens rather than the decay shape.

- C.3 ADDITIONAL EXPERIMENTS OF DIFFERENT EXTRAPOLATION RATIOS AND MODELS

Settings. In this section, we provide some additional extrapolation ratios from s = 2 to 5 and models based on 25 prompts from VBench (Huang et al., 2024). To evaluate the generality of UltraViCo, we test 2× extrapolation on HunyuanVideo, Wan, and CogVideoX, as well as 3× and

- 4× extrapolation on CogVideoX. In addition, we assess 5× extrapolation on HunyuanVideo. For Wan, we set α = 0.9. For HunyuanVideo, we use γ = 4 across all ratios, with α = 0.95,β = 0.6

- at 2× and α = 0.9,β = 0.8 at 5×. For CogVideoX, we use γ = 1 and β = 0.6 for all ratios, with α = 0.9 at 2× and 3×, and α = 0.85 at 4×. The configurations of other baselines follow Sec. 4.1.

Results. We compare UltraViCo with the baselines in Sec. 4.2. As shown in Tab. 3, UltraViCo achieves the best performance across all models and extrapolation ratios, not only avoiding content repetition but also substantially improving video quality. For example, CogVideoX exhibits nearly static videos at 4× extrapolation (Dynamic Degree ≤ 16) with poor visual quality (Imaging Quality ≤ 56), whereas our method significantly enhances both temporal dynamics and visual quality, with Dynamic Degree and Imaging Quality improving by 200% and 13.48%, respectively. Furthermore,

- at 5× extrapolation, UltraViCo also demonstrates strong performance, surpassing the best baseline scores by 350% in Dynamic Degree and 47.59% in Imaging Quality, indicating the potential of our method to extend to larger extrapolation ratios.

- C.4 MORE QUALITATIVE RESULTS OF OUR METHOD

In this section, we provide additional qualitive results for the experiments in Sec. 4.2. As shown in Fig. 14 and Fig. 15, whether under 3× or 4× extrapolation ratios, and across Wan and CogVideoX, our method consistently achieves substantially superior visual quality and temporal dynamics compared to the baselines. For example, as shown in Fig. 14, the videos generated by various baselines for 3× and 4× extrapolation on Wan are nearly completely static, whereas our method produces highly fluid and natural large-scale motion. Similarly, as shown in Fig. 15, the videos from the baselines are very blurry with dull colors, while our method generates realistic, natural results with rich details.

- Table 3: Quantitative results on VBench for more models and extrapolation. Note that NoRepeat Score is essentially a binary indicator: red entries indicate visually obvious repetitions, while others show no noticeable repetition.

Wan with 2× extrapolation CogVideoX with 3× extrapolation NoRepeat↑ Dynamic↑ Quality↑ Overall↑ NoRepeat↑ Dynamic↑ Quality↑ Overall↑

Method

PE N/A 32 58.13 23.22 82.52 16 57.91 19.59 PI N/A 32 54.23 21.52 99.07 4 54.27 18.17 NTK N/A 44 59.59 23.52 86.07 4 55.24 19.33 YaRN N/A 24 55.14 21.57 97.47 0 53.96 18.05 TASR N/A 36 59.97 23.70 97.93 8 55.75 19.24 RIFLEx N/A 16 48.15 20.34 97.86 8 55.31 19.03 Ours N/A 68 66.88 25.28 99.38 32 60.09 24.77

HunyuanVideo with 2× extrapolation CogVideoX with 4× extrapolation NoRepeat↑ Dynamic↑ Quality↑ Overall↑ NoRepeat↑ Dynamic↑ Quality↑ Overall↑

Method

PE 80.43 40 62.67 24.36 76.57 16 55.25 17.27 PI 98.87 4 52.35 23.55 88.53 4 46.82 16.63 NTK 94.97 32 65.47 24.62 78.89 2 52.74 18.14

- YaRN 97.99 4 52.87 23.26 94.75 4 47.36 16.90

- TASR 94.85 36 64.55 24.59 99.13 16 46.75 17.28 RIFLEx 97.27 36 65.19 24.52 97.00 12 50.59 16.66

- Ours 97.53 44 66.50 24.82 96.79 48 62.70 25.39

Method

CogVideoX with 2× extrapolation HunyuanVideo with 5× extrapolation NoRepeat↑ Dynamic↑ Quality↑ Overall↑ NoRepeat↑ Dynamic↑ Quality↑ Overall↑

PE 92.31 28 64.28 22.83 30.78 4 39.04 15.64 PI 98.85 8 57.11 21.88 81.58 0 36.63 16.76 NTK 94.66 16 63.04 23.55 71.54 8 43.43 17.78 YaRN 98.81 8 58.83 21.81 77.70 0 37.88 17.85 TASR 95.91 16 62.17 23.44 35.31 8 42.88 17.88 RIFLEx 99.42 16 60.30 23.28 53.65 4 40.55 15.71

- Ours 98.92 32 64.39 25.36 99.44 36 64.10 24.16

Moreover, we present another downstream task in Fig. 16, where generation is performed based on a given pose. Our method achieves high quality and dynamic results while closely following the given conditions.

- C.5 ACCELERATION OF ULTRAVICO VIA SPARSE ATTENTION AND DISTILLATION

Building upon recent advances in sparse-attention-based video acceleration and distillation (Team, 2024b), UltraViCo achieves about 16× speed-up without compromising performance (see Table 7).

- C.6 RUNTIME AND MEMORY COST

As shown in Table 8, built on top of FlashAttention (Dao et al., 2022) and SageAttention (Zhang

- et al., 2024b;a), UltraViCo incurs almost no additional overhead in either latency or memory usage.

- D FURTHER DETAILS OF ULTRAVICO

- D.1 ULTRAVICO WITH EFFIEIENT ONLINE ATTENTION

UltraViCo does not require materializing the full attention matrix and can be seamlessly integrated into efficient online attention kernels. Herein, we present its implementation based on FlashAttention, as illustrated by Algorithm 1.

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

PE ··· ···

[Figure 229]

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

PI ··· ···

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

NTK ··· ···

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

YaRN ··· ···

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

TASR ··· ···

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

RIFLEx ··· ···

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

###### Ours ··· ···

(a) 3× extrapolation (b) 4× extrapolation

- Figure 14: Qualitative results on Wan. The baselines produce nearly static videos with poor visual quality, whereas our method achieves significantly better quality and much more motion.

(a) 3× extrapolation (b) 4× extrapolation

PE ··· ···

PI ··· ···

NTK ··· ···

YaRN ··· ···

TASR ··· ···

RIFLEx ··· ···

Ours ··· ···

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

- Figure 15: Qualitative results on CogVideoX. The baselines produce nearly static videos with poor visual quality, whereas our method generates realistic results with rich details and fluid motion.

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

- Figure 16: Our method for pose-guided video generation. Our method closely aligns with the given pose conditions, while ensuring high dynamic range and excellent visual quality.

Algorithm 1 UltraViCo FlashAttention Kernel Require: Matrices Q, K, V ∈ RN×d, block size bq, bkv.

- 1: Divide Q into Tm = N/bq blocks {Qm}, and divide K, V into Tn = N/bkv blocks {Kn} and {Vn};
- 2: for m in [1, Tm] do
- 3: for n in [1, Tn] do
- 4: ⃗i = m × bq + range(0, bq), ⃗j = n × bkv + range(0, bkv), ⃗i ∈ R1×bq,⃗j ∈ R1×bkv ;
- 5: Initialize λ ∈ Rbq×bkv to 0 ;
- 6: λ = Eq. 6(⃗i,⃗j) ;
- 7: Smn = λQmKnT ;
- 8: pnm = max(pnm−1, rowmax(Smn )) ;
- 9: Pmn = exp(Smn − pnm) ;
- 10: lmn = epnm−1−pnm lmn−1 + rowsum( Pij) ;
- 11: Omn = diag(epnm−1−pnm)Omn−1 + PmnVn ;
- 12: end for
- 13: Om = diag(lmTn)−1OmTn ;
- 14: end for
- 15: return O = {Om};

- D.2 ABLATION ON HYPERPARAMETERS

In this section, we present more detailed illustrative ablation results for the hyperparameters α and β. The detailed sensitivity curve is shown in Fig. 17, while the illustrative ablations on the independent effects of α and β in the main experiments are reported in Tab. 6.

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

- (a) Schematic diagram of the α sensitivity curve.

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

- (b) Schematic diagram of the β sensitivity curve.

#### Figure 17: Illustration of the hyperparameter sensitivity curve.

#### Table 4: Illustrative sensitivity analysis of α on Hunyuan at 3× extrapolation. We set β equal

to α, i.e., a single decay factor is shared globally.

α Consistency↑ Dynamics↑ Quality↑ Overall↑ NoRepeat↑

1.0 0.9795 16 51.85 21.62 53.17 0.95 0.9663 25 54.92 24.07 100 0.9 0.9647 32 57.53 26.25 93.34 0.85 0.9298 68 69.93 26.89 99.53

- 0.8 0.9231 73 70.35 26.96 100

Table 5: Illustrative sensitivity analysis of β on Hunyuan at 3× extrapolation. We set α = 0.9 across all settings.

β Consistency↑ Dynamics↑ Quality↑ Overall↑ NoRepeat↑

- 1.0 0.9716 28 55.23 24.52 57.42 0.9 0.9647 32 57.53 26.25 93.34 0.8 0.9510 45 59.35 26.42 97.25 0.75 0.9496 51 62.11 26.98 95.77 0.6 0.9465 62 65.00 26.45 100 0.45 0.9349 65 68.34 26.99 100 0.3 0.9318 66 70.45 26.98 100

#### Table 6: Illustrative ablation experiments that independently examine the individual effects of α and β.

Method Consistency↑ Dynamics↑ Quality↑ Overall↑ NoRepeat↑ HunyuanVideo with 3× extrapolation

α = 1,β = 1 0.9795 16 51.85 21.62 53.17 α = 0.9,β = 1 0.9716 28 55.23 24.52 57.42 α = 1,β = 0.6 0.9784 25 55.13 23.13 93.52 α = 0.9,β = 0.6 0.9465 62 65.00 26.45 100

Wan2.1-1.3B with 3× extrapolation

α = 1 0.9419 6 56.28 18.53 – α = 0.9 0.9444 46 62.43 23.21 –

#### Table 7: Illustrative performance when combined with recent video-acceleration methods on HunyuanVideo.

Setting Time Cost↓ Consistency↑ Dynamics↑ Quality↑ Overall↑ NoRepeat↑

- 3× 5 GPU·hours 0.9465 62 65.00 26.45 100

- 3× with FastVideo 0.3 GPU·hours 0.9432 64 63.89 25.98 100
- 4× 8 GPU·hours 0.9491 42 66.54 24.52 99.87

- 4× with FastVideo 0.5 GPU·hours 0.9399 40 62.24 24.83 96.32

- Table 8: Illustrative runtime and memory comparison. Note that SageAttention is optimized for 4090-like architectures; on A800, its runtime is comparable to FlashAttention.

Model / Method Time (s / iter) Memory (per GPU) HunyuanVideo (3× extrapolation)

SageAttention 341.2 73188M SageAttention + Ours 349.6 72346M FlashAttention 349.3 76030M FlashAttention + Ours 355.3 75932M

Wan (3× extrapolation)

SageAttention 32.13 24342M SageAttention + Ours 34.12 24342M FlashAttention 32.64 24349M FlashAttention + Ours 33.74 24346M

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

(a) Performance of the video-continuation baseline alone.

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

(b) Illustration of combining UltraViCo with the video-continuation method.

Figure 18: Application of UltraViCo to segment-wise long-video generation. (a) Wan2.2-TI2V uses only a few ending frames, causing identity drift; (b) UltraViCo alleviates this issue.

