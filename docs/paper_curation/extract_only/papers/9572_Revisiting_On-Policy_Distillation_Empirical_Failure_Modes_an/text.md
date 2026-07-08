# arXiv:2603.25562v2[cs.LG]27Apr2026

### Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes

###### Yuqian Fu1,2∗ Haohuan Huang1,2∗ Kaiwen Jiang1,2 Jiacai Liu3 Zhuo Jiang4 Yuanheng Zhu1,2† Dongbin Zhao1,2

- 1 State Key Laboratory of Multimodal Artificial Intelligence Systems, CASIA
- 2 School of Artificial Intelligence, UCAS 3 Fudan University 4 Independent Researcher {fuyuqian2022, yuanheng.zhu}@ia.ac.cn

##### Abstract

On-policy distillation (OPD) is increasingly used in LLM post-training because it can leverage a teacher model to provide dense supervision on student rollouts. The standard implementation, however, usually reduces distribution matching to a sampled-token log-ratio, which can make the learning signal fragile on long rollouts whose prefixes drift away from the teacher’s typical support. We revisit this formulation from both theoretical and implementation perspectives. Theoretically, token-level OPD is biased relative to sequence-level reverse-KL minimization, but admits a substantially tighter worst-case variance bound; a controlled synthetic study further shows that stronger future-reward coupling increases gradient variance and destabilizes training. Empirically, we identify three failure modes of sampled-token OPD: imbalanced token-level supervision, unreliable teacher guidance on student-generated prefixes, and tokenizer or special-token mismatch. These findings motivate teacher top-K local support matching, a truncated reverse-KL objective that compares teacher and student distributions over a teacher-supported token set at each prefix, together with top-p rollout sampling and special-token masking. Across single-task reasoning and multi-task benchmarks spanning agentic and reasoning settings, this objective improves optimization stability and yields a +19.8% performance gain over standard sampled-token OPD baselines, providing a practical recipe for more stable on-policy distillation.

Code Blog

[Figure 1]

##### 1 Introduction

On-policy distillation (OPD) is becoming an increasingly common component of LLM posttraining, especially for reasoning and agentic models. Recent public reports from Thinking Machines Lab (Lu & Lab, 2025), Qwen3 (Yang et al., 2025), MiMo-V2-Flash (Xiao et al., 2026), and GLM-5 (Zeng et al., 2026) suggest a broader shift toward supervision on modelgenerated trajectories, or closely related on-policy variants, alongside off-policy distillation and reinforcement learning. By training on student rollouts while evaluating them with a stronger teacher, OPD combines on-policy data collection with dense token-level feedback at relatively low cost (Agarwal et al., 2024; Gu et al., 2024). This profile is attractive in practical post-training pipelines, where training efficiency matters and models often need to combine or recover capabilities across domains and training stages (Xiao et al., 2026; Zeng et al., 2026; Wang et al., 2026a; DeepSeek-AI, 2026).

Most OPD implementations in current LLM pipelines use a token-level estimator, even though it is biased relative to sequence-level reverse-KL (Lu & Lab, 2025). A basic reason is that sequence-level objectives couple each token update to many future rewards, which can make optimization substantially noisier in long-horizon settings. We make this trade-off

∗Equal contribution. †Corresponding authors. ‡Work in progress.

explicit: token-level OPD removes future-reward coupling and is therefore biased, yet it admits more favorable worst-case variance scaling. Our toy experiment shows the same pattern empirically, with stronger future coupling leading to higher gradient variance and less stable optimization. This suggests a practical design principle for long-horizon training: keep supervision token-level to control variance.

In current LLM pipelines, this token-level objective is usually instantiated by sampledtoken comparison: at each training step, the update is driven by the teacher–student log-probability difference on the sampled token (Lu & Lab, 2025; Xiao et al., 2026). This implementation is simple and efficient, but its learning signal can become brittle once rollouts grow long. Gu et al. (2024) report degraded outputs such as repetition, which is consistent with our observations. Recent work also reports entropy collapse under sampledtoken OPD in certain cases (Ko et al., 2026; Jin et al., 2026b). More recently, full-vocabulary distillation has been reported to outperform sampled-token variants in some settings (Zhao et al., 2026; DeepSeek-AI, 2026), suggesting that the one-token formulation can leave useful teacher information unused.

Our empirical analysis shows that this brittleness stems from several recurring failure modes. In particular, we identify two objective-level issues: the one-token signal is often highly imbalanced, and teacher guidance can become unreliable on student-generated prefixes. We also observe an additional implementation-level issue from tokenizer or special-token mismatch, which can further distort one-token comparisons. Taken together, these results point to a practical question: how can we retain the variance advantage of token-level OPD while making its supervision signal less brittle in practice?

The analysis above suggests a targeted modification of the standard sampled-token objective. We replace one-token supervision with teacher top-K local support matching, where teacher and student are compared over a teacher-supported token subset at each prefix rather than only on the sampled token. We implement this objective as truncated reverse-KL, together with top-p rollout sampling and special-token masking. The resulting update remains local and inexpensive, while providing a less brittle training signal.

###### Overall, our main contributions are as follows:

- • We clarify the theoretical trade-off in OPD: token-level OPD is biased relative to sequence-level OPD, but has substantially better worst-case variance scaling with sequence length, making it attractive for long-horizon LLM post-training.
- • We present an empirical analysis of why sampled-token OPD can be unstable in practice, highlighting two recurring objective-level issues—imbalanced one-token supervision and unreliable teacher guidance on student-generated prefixes—along with an additional implementation issue from tokenizer or special-token mismatch.
- • We propose teacher top-K local support matching as an analysis-driven revision of sampled-token OPD, implemented with truncated reverse-KL, top-p rollouts, and special-token masking, and show that it yields more stable optimization and stronger empirical performance than sampled-token OPD in single-task math reasoning and multi-task agentic-plus-reasoning training.

##### 2 Understanding Sampled-Token OPD: Trade-offs and Failure Modes

###### 2.1 From reverse-KL to token-level OPD

On-policy distillation aims to transfer the capabilities of a stronger teacher model into a student model by minimizing the reverse-KL from the student to the teacher. For a prompt x, the OPD objective is

JOPD(θ) = Ex∼D [DKL (πθ(· | x) ∥ q(· | x))] ,

where πθ and q denote the student and teacher models, respectively. Its gradient can be written as

∇θJOPD(θ) = Ex,y∼πθ(·|x) log πθ(y | x) − log q(y | x) ∇θ log πθ(y | x) .

Left Task (odd iterations)

Right Task (even iterations)

=0 (token-level)

=0 (token-level)

- 101

- 102

- 103

- 104

- 105

- 106

- 107

- 108

- 101

- 102

- 103

- 104

- 105

- 106

- 107

=0.25

=0.25

=0.5

=0.5

- =0.75

- =1.0

- =0.75

- =1.0

GradientVariance

GradientVariance

0 250 500 750 1000 1250 1500 1750 2000

0 250 500 750 1000 1250 1500 1750 2000

Iteration

Iteration

- (a) Gradient variance in the toy experiment. Larger γ generally yields higher variance in both tasks.

|[Figure 2]| | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0

time step

8

6

4

2

0

2

4

6

8

positions

tok-level Left Task (init=2)

|[Figure 3]| | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0

time step

seq-level(gamma=0.5) Left Task (init=2)

|[Figure 4]| | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0

time step

seq-level(gamma=1.0) Left Task (init=2)

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
|[Figure 5]| | | | | | | |
| | | | | | | | |

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0

time step

8

6

4

2

0

2

4

6

8

positions

tok-level Right Task (init=-2)

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
|[Figure 6]| | | | | | | |
| | | | | | | | |

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0

time step

seq-level(gamma=0.5) Right Task (init=-2)

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
|[Figure 7]| | | | | | | |
| | | | | | | | |

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0

time step

seq-level(gamma=1.0) Right Task (init=-2)

|[Figure 8]| |
|---|---|
| | |
| | |
| | |
| | |

0.0

0.2

0.4

0.6

0.8

1.0

normalizedvisitation(sqrtscale)

- (b) State visitation under γ ∈ {0.0,0.5,1.0} in the toy environment. For γ = 1.0, the policy model fails to consistently move toward the target, and instead exhibits drifting behavior.

- Figure 1: Effect of increasing γ in the toy experiment. Larger γ yields a higher and more persistent variance regime and, in the sequence-level limit, drifting policies in state space.

For each decoding step t, let ct = (x, y<t) denote the prefix context, and define st = ∇θ log πθ(yt | ct), rt = log

πθ(yt | ct) q(yt | ct)

. Using the autoregressive factorization

log πθ(y | x) − log q(y | x) =

we obtain the sequence-level estimator

T

T

#### ∑

#### ∑

rt′, ∇θ log πθ(y | x) =

st,

t=1

t′=1

T

T

#### ∑

#### ∑

gˆseq =

rt′ st. (1)

t=1

t′=1

For t′ < t, we have E[rt′st] = 0, because rt′ depends only on the prefix before step t, while

E[st | x,y<t] = ∑

πθ(yt | ct) ∇θ log πθ(yt | ct) = 0.

yt

The same gradient can therefore be expressed in causal reward-to-go form

T

T

#### ∑

#### ∑

E[gˆseq] = E

rt′ st , where each token update is coupled to all future rewards along the trajectory.

t=1

t′=t

Another approximation in LLM training retains only the immediate term at each position

T

#### ∑

gˆtok =

rtst. (2)

t=1

We refer to Eq. (2) as the token-level estimator. This approximation removes future-reward coupling, so the update for token yt depends only on its immediate reward. Consequently, it is biased relative to the sequence-level reverse-KL estimator, but exhibits lower variance in long-horizon settings. Under bounded rewards and bounded gradients, the worst-case variance upper bound of token-level OPD scales as O(T2), whereas the sequence-level estimator scales as O(T4). We provide a detailed derivation in Appendix D. To interpolate between these extremes, we consider the discounted return-to-go estimator

T

T

γt′−trt′ st. γ ∈ [0,1] (3)

#### ∑

#### ∑

gˆγ =

t=1

t′=t

The case γ = 0 recovers token-level OPD, while γ = 1 recovers the causal sequence-level estimator. We further validate this trade-off in a two-task toy experiment (Figure 1): stronger future coupling leads to substantially higher gradient variance and less stable optimization. This motivates our focus on token-level supervision in the remainder of the paper, where the main question becomes how to improve its local training signal in practical LLM settings. Additional experimental details are provided in Appendix E.

###### 2.2 Why sampled-token OPD is brittle in practice

Although token-level OPD is appealing from a bias–variance perspective, the standard sampled-token formulation can be brittle in practice. We isolate three failure modes: (1) a highly imbalanced token-level distillation signal, (2) unreliable teacher guidance on studentgenerated prefixes, and (3) distortions introduced by tokenizer or special-token mismatch. These observations come from sampled-token OPD experiments on math reasoning, using Qwen2.5-7B-Instruct (Qwen et al., 2024) as the student and OpenThinker3-7B (Guha et al., 2025), an SFT model built on Qwen2.5-7B-Instruct, as the teacher.

A highly imbalanced sampled-token signal. In sampled-token OPD, the update at step t is driven by the log-ratio on a single sampled token:

log q(yt | ct) − log πθ(yt | ct).

Negative rewards arise whenever the student assigns higher probability to a sampled token than the teacher. As shown in Figure 2, most sampled tokens receive negative reward. This leaves optimization dominated by a small subset of locally positive tokens, so training becomes sensitive to high-frequency fillers and short continuations that can receive favorable local scores while contributing little to trajectory-level quality.

The teacher signal can become unreliable on student-generated prefixes. Sampled-token OPD assumes that teacher probability on a student-generated token is a useful proxy for trajectory quality. This proxy becomes unreliable on prefixes that are common under the student but uncommon for the teacher. In such regions, tokens with high teacher probability can remain rewarded even after the trajectory has drifted into repetition, self-resetting reasoning, or other meaningless continuations (Figure 3; Appendix H). This creates an objective mismatch between token-level teacher agreement and trajectory-level quality.

We hypothesize that two factors amplify this issue: sharp teacher distributions, where modest teacher–student mismatch can produce large log-ratio values, and growing teacher– student divergence along long rollouts. Consistent with this view, Figure 4 shows that the distribution of teacher–student gaps becomes wider later in the sequence.

Tokenizer and special-token mismatch. Sampled-token OPD compares the exact token generated by the student using the teacher distribution. When the two models use different

1.0

| |y=x| | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

0.8

Teacherprobability

0.6

0.4

0.2

0.0

0.0 0.2 0.4 0.6 0.8 1.0

Student probability

- Figure 2: Scatter of token probabilities (teacher vs. student) at the sampled-token OPD first iteration. The sampled-token signal is heavily skewed toward penalizing the current token.

... ### FinalAnswer :\n \ boxed {- 1 4 }\n\n ### FinalAnswer :\n \ boxed {- 1 4 }\n\n ...

0.6

0.7

0.8

0.9

1.0

Probability

repetition

Teacher prob Student prob

Question

Find the product of all real values of r for which 21x = r−7x has exactly one real solution.

- Figure 3: Example of unreliable teacher guidance. The student falls into a repetition loop, yet the teacher remains locally aligned with the student on repeated tokens, indicating that sampled-token OPD may fail to penalize this behavior.

tokenizations, the same raw text can be segmented differently, so a student-generated token may not correspond to a natural token under the teacher (Boizard et al., 2025; Patiño et al., 2025; Minixhofer et al., 2025). For example, the student may generate ‘<think>’ as ‘<’,

‘think’, ‘>’, while the teacher expects ‘<th’, ‘ink’, ‘>’. Then token ‘<’ receives low probability from the teacher, even though both models produce the same semantic content. Similar mismatches arise for special tokens such as end-of-sequence markers. In this setting, a one-token comparison confuses semantic disagreement with tokenizer mismatch. Since supervision is applied on a single token, such mismatch can distort the reward signal.

These observations motivate moving beyond one-token supervision: instead of comparing only the sampled token, we compare teacher and student over a set of teacher-supported next-token continuations at each prefix, while retaining token-level updates for stability.

##### 3 Method

Our method is designed as a direct response to the failure modes above: it retains tokenlevel OPD, but replaces one-token supervision with a distribution-level comparison over a teacher-selected support set at each prefix. This yields a truncated reverse-KL objective that preserves the efficiency of local updates while reducing dependence on any single sampled token. Section 3.1 introduces the objective, and Section 3.2 describes the practical choices that ensure stable training.

###### 3.1 Teacher top-K local support matching

Instead of comparing teacher and student on a single sampled token, we compare their next-token distributions over a teacher-defined local support set. A natural starting point is

0.2

teacherlogprob-studentlogprob

0.0

0.2

0.4

0.6

0.8

1.0

1.2

0 1k 2k 3k 4k 5k 6k 7k 8k 9k 10k 11k 12k 13k 14k 15k 16k

Token Position

- Figure 4: Distribution of teacher–student log-probability gaps across token positions. Several later position buckets exhibit wider lower tails and more extreme values, indicating a noisier teacher signal on long-horizon student-generated rollouts.

|<|
|---|

, think, > …

|<|im_end|>|
|---|

Raw Text Student tokenization

<think> Okay, let’s… Final answer is \boxed{7}\n <EOS>

- -0.07
- -19.16

Student log prob

Teacher log prob

[Figure 9]

- -0.00
- -58.71

- Figure 5: Token-level comparison can penalize semantically correct outputs due to tokenizer mismatch between the teacher and student.

the full-vocabulary reverse-KL at prefix ct:

πθ(v | ct) q(v | ct)

Lfull(ct) = ∑

πθ(v | ct) log

. (4)

v∈V

Sampled-token OPD can be viewed as a Monte Carlo approximation to this quantity:

πθ(yt | ct) q(yt | ct)

Lsample(ct, yt) = log

, yt ∼ πθ(· | ct). (5)

This approximation is computationally attractive, but it concentrates the entire update on a single sampled token. We instead compare teacher and student over a teacher-supported candidate set at each prefix.

For each prompt x, we sample a group of outputs {oi}iG=1 using the student inference policy. Let ci,t = (x, yi,<t) be the prefix at position t of output oi, and define the teacher support set

S(ci,t) = TopKq(ci,t), (6) which contains the K highest-probability tokens under the teacher at that prefix. Within this support, we renormalize both teacher and student distributions:

q(v | ci,t) ∑u∈S(ci,t) q(u | ci,t)

πθ(v | ci,t) ∑u∈S(ci,t) πθ(u | ci,t)

πˆθ(v | ci,t) =

, qˆ(v | ci,t) =

. (7)

Our local support matching (LSM) objective averages the truncated reverse-KL over all rollout positions:

######   1

  . (8)

|oi|

G

πˆθ(v | ci,t) qˆ(v | ci,t)

#### ∑

#### ∑

#### ∑

πˆθ(v | ci,t) log

LLSM = Ex,{oi}∼πθ,infer

∑iG=1 |oi|

t=1

i=1

v∈S(ci,t)

Relative to sampled-token OPD, this replaces a one-token point estimate with a distributionlevel comparison over teacher-supported candidates at the same prefix. The update is therefore no longer determined by the sign and magnitude of a single sampled-token log-ratio, while remaining much cheaper than full-vocabulary KL.

###### 3.2 Practical stabilization choices

Support-set renormalization. Renormalization is necessary because the objective is evaluated on a truncated support rather than the full vocabulary. Specifically, we apply a separate softmax over the logits inside the support set, so gradients do not directly propagate to tokens outside this set. Without this step, optimization can become unstable because the teacher and student probability masses inside the support are not directly comparable.

Top-p rollout sampling. We generate rollouts with top-p sampling. Unconstrained sampling occasionally produces very low-probability tokens, creating prefixes on which the teacher signal becomes less informative and optimization less stable. Top-p sampling keeps trajectories closer to typical continuations and makes the teacher signal more reliable.

Special-token masking. We mask problematic special tokens to reduce false negatives caused by incompatible tokenization conventions. This is an orthogonal practical fix: it materially helps sampled-token OPD in our experiments, while our local support objective is much less sensitive to it. In principle, one could also merge multi-token marker variants or average over equivalent tokenizations, but we do not pursue those tokenizer-specific remedies here because masking is the simplest model-agnostic correction.

##### 4 Experiments

We evaluate local support matching in three settings: single-task math reasoning (Section 4.2), alternating multi-task training over math and agentic tasks (Section 4.3), and an additional single-task agentic setting on a smaller student model (Appendix G.1). We also present ablations in Section 4.4 and provide training-dynamics analysis in Appendix G.2.

###### 4.1 Setup

We implement local support matching on top of the verl-agent framework (Feng et al., 2025), using Qwen2.5-Instruct models as students. We consider two main settings. The first is single-task math reasoning, where OpenThinker3-7B (Guha et al., 2025) serves as the teacher and training uses the English portion of DAPO-Math-17K (Yu et al., 2025) with a maximum context length of 16K. The second is a multi-task setting that alternates batches between math reasoning and a multi-turn agentic task based on ALFWorld (Shridhar et al., 2021). In this setting, math uses OpenThinker3-7B (Guha et al., 2025) as the teacher, while the agentic side uses the released GiGPO-Qwen2.5-7B-Instruct-ALFWorld checkpoint (Feng et al., 2025).

For math reasoning, we report pass@1 on five benchmarks: Math500 (Hendrycks et al., 2021), AIME24 (Zhang & Math-AI, 2024), AIME25 (Zhang & Math-AI, 2025), Minerva (Lewkowycz et al., 2022), and OlympiadBench (He et al., 2024). For ALFWorld (Shridhar et al., 2021), we report success rate by default. In a small number of cases, we additionally report avg@32 on the math benchmarks. More details on experimental setups can be found in Appendix F.

###### 4.2 Single-task math reasoning

- Table 1 shows that local support matching improves over sampled-token OPD in single-task math reasoning. Sampled-token OPD already improves the student from 28.2 to 36.4 average score, but still remains substantially below the teacher. Applying special-token masking to sampled-token OPD further improves the average to 40.7, indicating that tokenizer-related mismatch is a meaningful part of the failure.

Both variants of our method outperform sampled-token OPD and its masked variant in average score. This shows that the gain is not solely attributable to mismatch handling, and instead supports the role of a stronger distribution-level distillation signal. In addition, masking changes our method only modestly (41.7 vs. 41.5), consistent with the view that local support matching is less sensitive to tokenizer mismatch than one-token supervision. Additional evidence on WebShop (Yao et al., 2022) is shown in Appendix G.1.

Method Math500 AIME24 AIME25 Minerva OlympiadBench Avg. Qwen2.5-7B-It 68.2 13.3 0.0 26.5 32.9 28.2 OpenThinker3-7B 92.2 53.3 40.0 39.0 55.6 56.0 Sampled-token OPD 80.0 10.0 16.7 32.4 43.1 36.4 Sampled-token OPD w/ mask 81.4 26.7 16.7 34.2 44.7 40.7 Ours w/o mask 80.4 23.3 26.7 34.2 43.9 41.7 Ours w/ mask 82.0 23.3 23.3 34.9 43.9 41.5

Table 1: Results on single-task math reasoning.

###### 4.3 Multi-task agentic-plus-reasoning training

- Table 2 shows that the effect of local support matching differs across the two task families in alternating multi-task training. The unmasked version raises the average math score from 34.8 to 41.7 (+19.8%), while maintaining competitive ALFWorld performance. The masked version achieves the best ALFWorld score at 97.7, but gives up part of the math improvement. This pattern suggests that local support matching is especially helpful for the reasoning side of the mixture, where sampled-token signals are more exposed to prefix drift; masking, in contrast, mainly shifts the trade-off toward the agentic task in this run. Beyond evaluation performance, our objective also yields consistently better optimization dynamics. We defer the full learning curves and diagnostic plots to Appendix G.2.

Agentic Reasoning ALFWorld MATH500 AIME24 AIME25 Minerva OlympiadBench Avg.

Method

Qwen2.5-7B-It 21.9 68.2 13.3 0.0 26.5 32.9 28.2 GiGPO-Qwen2.5-7B-It-Alfworld 95.3 – – – – – – OpenThinker3-7B – 92.2 53.3 40.0 39.0 55.6 56.0 Sampled-token OPD 90.6 74.8 13.3 13.3 32.1 40.5 34.8 Sampled-token OPD w/ mask 93.8 76.0 20.0 13.3 33.5 40.4 36.6 Ours w/o mask 95.3 82.0 33.3 16.7 32.7 44.0 41.7 Ours w/ mask 97.7 79.0 20.0 16.7 34.6 42.5 38.6

- Table 2: Results on batch-alternating multi-task training over ALFWorld and math reasoning.

4.4 Ablations

- Table 3 and Figure 6 suggest that the gains arise from several design choices rather than any single modification. Teacher top-K comparison alone is not sufficient: the rollout policy must also remain in a stable region, and adding top-p sampling turns an initially weaker top-K variant into a stronger configuration. Under the same top-p rollout conditions, teacher top-K local support matching still improves AIME24 avg@32 from 21.6 to 23.6. Renormalization inside the truncated support is essential, as removing it leads to rapid collapse. Performance is not especially sensitive to the exact support size once K is large enough, but training becomes unstable when the support is too small or rollouts are fully unconstrained.

###### Method AIME24 avg@32

Qwen2.5-7B-Instruct 10.0 OpenThinker3-7B 63.3 Sampled-token OPD 20.4 Sampled-token OPD + top-p 21.6 Teacher top-K 17.7 Teacher top-K + top-p 23.6

Table 3: Component ablation under single-task math training.

Top-K support variants. Our main experiments define the truncated expectation on the teacher’s top-K support. Alternative objectives can compute the expectation over teacher top-K (used in the main results), student top-K, or teacher top-K augmented with the student-sampled token. We provide a preliminary comparison in Table 4 under both the single-task and multi-task settings.

- 4

6

8

10

PolicyEntropy

w/o normalization

w/ normalization

0 100 200 300 400

Step

0.0

0.1

0.2

0.3

0.4

0.5

TrainScore(DAPO-Math-17k)

w/o normalization

w/ normalization

- (a) Support renormalization.

0 100 200 300 400

Step

0.0

0.5

1.0

1.5

2.0

2.5

3.0

3.5

4.0

PolicyEntropy

k=16 k=32 k=48

100 150 200 250 300 350 400

Step

0.14

0.16

0.18

0.20

0.22

0.24

TestScore(AIME24_avg@32)

k=16 k=32 k=48

(b) Support size K.

0 100 200 300 400

Step

- 0

- 1

- 2

- 3

- 4

- 5

- 6

PolicyEntropy

top_p=1.0 top_p=0.9 top_p=0.8

100 150 200 250 300 350 400

Step

0.14

0.16

0.18

0.20

0.22

0.24

TestScore(AIME24_avg@32)

top_p=1.0 top_p=0.9 top_p=0.8

(c) Rollout top-p.

Figure 6: Ablations of the main design choices. Renormalization is required for stability, very small support sets hurt learning, and unconstrained rollout sampling degrades optimization.

Method

avg@32 pass@1 AIME24 MATH500 AIME24 AIME25 Minerva OlympiadBench Avg.

teacher top-K 23.6 80.4 23.3 26.7 34.2 43.9 41.7 student top-K w/ sampled 22.3 82.4 30.0 16.7 35.7 44.9 41.9 teacher top-K w/ sampled 22.4 81.6 26.7 23.3 36.4 46.7 42.9

(a) Alternative KL expectation supports in the single-task setting.

Method

Agentic Reasoning ALFWorld MATH500 AIME24 AIME25 Minerva OlympiadBench Avg.

teacher top-K 95.3 82.0 33.3 16.7 32.7 44.0 41.7 student top-K w/ sampled 95.3 65.6 10.0 10.0 25.0 31.6 28.4 teacher top-K w/ sampled 94.5 63.2 10.0 10.0 21.0 30.1 26.9

- (b) Alternative KL expectation supports in the multi-task setting. We report pass@1 for all math reasoning benchmarks. The final column averages pass@1 metrics only.

Table 4: Alternative top-K support variants under single-task and multi-task settings.

In the single-task setting, the three variants achieve broadly comparable results. The teacher top-K + sampled-token variant attains the highest average pass@1 score, while student top-K performs competitively on several individual benchmarks, and the original teacher top-K gives the best AIME24 avg@32 among the three. The multi-task results are less uniform. In that setting, the original teacher top-K variant performs substantially better than the other two alternatives, while both student top-K and teacher top-K + sampled-token variants degrade considerably on the math benchmarks. We therefore treat these results as evidence that support construction matters, but do not over-interpret the ranking among variants.

Overall, these results suggest that the choice of where the KL expectation is computed can matter in nontrivial ways, especially in the multi-task setting. We treat this as a partial ablation and a preliminary exploration, and discuss possible causes, including support-set construction and remaining off-policy effects, in Appendix A.

- 5 Conclusion

2

0

0 100 200 300 400

Step

This work revisits on-policy distillation (OPD) for LLM post-training from both theoretical and implementation perspectives. Our theoretical analysis shows why token-level OPD is an attractive approximation for long-horizon training: it is biased relative to sequence-level reverse-KL, but avoids future-reward coupling and has substantially better worst-case variance scaling. At the same time, our empirical study shows that the standard sampledtoken implementation can provide brittle supervision because its signal is imbalanced, can remain misleading on student-drifted prefixes, and is sensitive to tokenizer or special-token mismatch. Teacher top-K local support matching addresses these issues by preserving local token-level updates while replacing one-token supervision with a truncated distributionlevel comparison. Across single-task math reasoning and alternating agentic-plus-reasoning training, this simple modification improves optimization stability and downstream performance over sampled-token OPD, while also clarifying where teacher matching remains an imperfect proxy for task success.

##### References

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=3zKtaqxLhW.

Nicolas Boizard, Kevin El Haddad, CELINE HUDELOT, and Pierre Colombo. Towards cross-tokenizer distillation: the universal logit distillation loss for LLMs. Transactions on Machine Learning Research, 2025. ISSN 2835-8856. URL https://openreview.net/forum? id=bwRxXiGO9A.

Howard Chen, Noam Razin, Karthik Narasimhan, and Danqi Chen. Retaining by doing: The role of on-policy data in mitigating forgetting. arXiv preprint arXiv:2510.18874, 2025.

DeepSeek-AI. Deepseek-v4: Towards highly efficient million-token context intelligence, 2026.

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-group policy optimization for LLM agent training. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=QXEhBMNrCW.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: Knowledge distillation of large language models. In The Twelfth International Conference on Learning Representations,

2024. URL https://openreview.net/forum?id=5h0qf7IBZZ.

Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, et al. OpenThoughts: Data recipes for reasoning models. arXiv preprint arXiv:2506.04178, 2025.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3828–3850, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. URL https://openreview.net/forum?id=7Bywt2mQsCe.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.

Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, et al. Reinforcement learning via self-distillation. arXiv preprint arXiv:2601.20802, 2026.

Woogyeol Jin, Taywon Min, Yongjin Yang, Swanand Ravindra Kadhe, Yi Zhou, Dennis Wei, Nathalie Baracaldo, and Kimin Lee. Entropy-aware on-policy distillation of language models. In The 1st Workshop on Scaling Post-training for LLMs, 2026a. URL https:// openreview.net/forum?id=WSRQ37tzk1.

Woogyeol Jin, Taywon Min, Yongjin Yang, Swanand Ravindra Kadhe, Yi Zhou, Dennis Wei, Nathalie Baracaldo, and Kimin Lee. Entropy-aware on-policy distillation of language models. arXiv preprint arXiv:2603.07079, 2026b.

Yoon Kim and Alexander M Rush. Sequence-level knowledge distillation. In Proceedings of the 2016 conference on empirical methods in natural language processing, pp. 1317–1327, 2016.

Jongwoo Ko, Sara Abdali, Young Jin Kim, Tianyi Chen, and Pashmina Cameron. Scaling reasoning efficiently via relaxed on-policy distillation. arXiv preprint arXiv:2603.11137, 2026.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in neural information processing systems, 35:3843–3857, 2022.

Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huan-ang Gao, Wenkai Yang, Zhiyuan Liu, et al. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. arXiv preprint arXiv:2604.13016, 2026.

Jiacai Liu, Yingru Li, Yuqian Fu, Jiawei Wang, Qian Liu, and Zhuo Jiang. When speed kills stability: Demystifying RL collapse from the training-inference mismatch, September

2025. URL https://richardli.xyz/rl-collapse.

Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20251026. https://thinkingmachines.ai/blog/onpolicy-distillation.

Benjamin Minixhofer, Ivan Vulic´, and Edoardo Ponti. Universal cross-tokenizer distillation via approximate likelihood matching. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=DxKP2E0xK2.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Carlos Miguel Patiño, Kashif Rasul, Quentin Gallouédec, Ben Burtenshaw, Sergio Paniego, Vaibhav Srivastav, Thibaud Frere, Ed Beeching, Lewis Tunstall, Leandro von Werra, and Thomas Wolf. Unlocking on-policy distillation for any model family, 2025.

Qwen, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. DistilBERT, a distilled

version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108, 2019. John Schulman. Approximating KL divergence. John Schulman’s Homepage, 5, 2020.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. HybridFlow: A flexible and efficient RLHF framework. In Proceedings of the Twentieth European Conference on Computer Systems, EuroSys ’25, pp. 1279–1297. ACM, March 2025. doi: 10.1145/3689031.3696075. URL http://dx.doi.org/ 10.1145/3689031.3696075.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cote, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. ALFWorld: Aligning text and embodied environments for interactive learning. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=0IOX0YcCdTn.

Haoqing Wang, Xiang Long, Ziheng Li, Yilong Xu, Tingguang Li, and Yehui Tang. To mix or to merge: Toward multi-domain reinforcement learning for large language models. arXiv preprint arXiv:2602.12566, 2026a.

Yinjie Wang, Xuyang Chen, Xiaolong Jin, Mengdi Wang, and Ling Yang. Openclaw-RL: Train any agent simply by talking. arXiv preprint arXiv:2603.10165, 2026b.

Bangjun Xiao, Bingquan Xia, Bo Yang, Bofei Gao, Bowen Shen, Chen Zhang, Chenhong He, Chiheng Lou, Fuli Luo, Gang Wang, et al. MiMo-V2-Flash technical report. arXiv preprint arXiv:2601.02780, 2026.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Chenxu Yang, Chuanyu Qin, Qingyi Si, Minghui Chen, Naibin Gu, Dingyu Yao, Zheng Lin, Weiping Wang, Jiaqi Wang, and Nan Duan. Self-distilled RLVR. arXiv preprint arXiv:2604.03128, 2026a.

Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, and Yankai Lin. Learning beyond teacher: Generalized on-policy distillation with reward extrapolation. arXiv preprint arXiv:2602.12125, 2026b.

Feng Yao, Liyuan Liu, Dinghuai Zhang, Chengyu Dong, Jingbo Shang, and Jianfeng Gao. Your efficient RL framework secretly brings you off-policy RL training, August 2025. URL https://fengyao.notion.site/off-policy-rl.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. WebShop: Towards scalable real-world web interaction with grounded language agents. Advances in Neural Information Processing Systems, 35:20744–20757, 2022.

Chenlu Ye, Xuanchang Zhang, Yifan Hao, Zhou Yu, Ziji Zhang, Abhinav Gullapalli, Hao Chen, Jing Huang, and Tong Zhang. Adaptive layerwise perturbation: Unifying off-policy corrections for LLM RL. arXiv preprint arXiv:2603.19470, 2026a.

Tianzhu Ye, Li Dong, Zewen Chi, Xun Wu, Shaohan Huang, and Furu Wei. Black-box on-policy distillation of large language models. arXiv preprint arXiv:2511.10643, 2025.

Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, and Furu Wei. On-policy context distillation for language models. arXiv preprint arXiv:2602.12275, 2026b.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Juncai Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Ru Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Yonghui Wu, and Mingxuan Wang. DAPO: An open-source LLM reinforcement learning system at scale. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=2a36EMSSTp.

Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chengxing Xie, Cunxiang Wang, et al. GLM-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026.

Juzheng Zhang, Abhimanyu Hans, John Kirchenbauer, Micah Goldblum, Ashwinee Panda, and Tom Goldstein. Learning from mixed rollouts: Logit fusion as a bridge between imitation and exploration. Notion Blog, 2026. URL https://juzhengz.notion.site/ logit-fusion.

Lunjun Zhang and Jimmy Ba. EMA policy gradient: Taming reinforcement learning for LLMs with EMA anchor and Top-k KL. arXiv preprint arXiv:2602.04417, 2026.

Yifan Zhang and Team Math-AI. American invitational mathematics examination (AIME)

- 2024. Wei Zhao, Zhe Li, Yige Li, Ye Zhang, and Junfeng Sun, 2024.

Yifan Zhang and Team Math-AI. American invitational mathematics examination (AIME)

- 2025. Wei Zhao, Zhe Li, Yige Li, Ye Zhang, and Junfeng Sun, 2025.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734, 2026.

## Appendix

##### A Discussion and Limitations

Sampled-token augmentation of the teacher top-K support. In our current implementation, the support is S(ct) = TopKq(ct): teacher and student distributions are renormalized on this teacher-selected set before computing the KL term. However, this is not the only possible design. One can instead use an augmented support S+(ct) = S(ct) ∪ {yt}, or include the sampled token through an importance-weighted correction that more explicitly preserves unbiasedness with respect to the full-vocabulary reverse-KL (Zhang & Ba, 2026). We provide formal definitions of several support-augmentation variants, including alternatives based on teacher- or student-centered supports and EMA-PG-style corrections, and evaluate them in Appendix G.3. Overall, our default renormalized formulation remains competitive across both single-task and multi-task settings, which is why we adopt it in the main experiments.

Support-set KL is a truncated objective. Our formulation computes the KL term only over a restricted token subset, i.e., the teacher top-K support, rather than over the full vocabulary. This means that the expected gradient is formed within that subset: tokens outside the support do not receive gradient contributions from this objective. Relative to full-vocabulary reverse-KL, this introduces bias; more generally, it changes where the expectation is taken and which parts of the vocabulary participate in the update. We describe this here as a property of the estimator, rather than as a settled benefit or drawback.

Training–inference mismatch. Prefixes are generated under a rollout policy, e.g., topp sampling in the vLLM engine, while the training engine updates the model without correcting for this sampling process. This may introduce a training–inference mismatch (Liu et al., 2025; Yao et al., 2025), which we leave to ongoing work.

Teacher matching remains an imperfect proxy for task success. Even when OPD is well defined as a teacher-matching objective, the resulting reward can still diverge from the underlying notion of successful behavior. Our reward-hacking cases make this gap concrete: locally teacher-preferred continuations can remain rewardable even when the overall trajectory is already unhelpful or harmful. A noticeable gap to the teacher also remains in our experiments, which suggests that better local supervision is only one part of the distillation problem, especially when teacher and student differ substantially. Closing that gap may require stronger rollout control, better handling of distribution shift, better use of teacher uncertainty, and combinations with outcome-verifiable rewards.

##### B Future Directions

Top-p truncation as an adaptive support. Another route toward a more compute-efficient reverse-KL objective is top-p truncation instead of top-K truncation, where the KL divergence is computed on the subset of tokens whose cumulative probability mass reaches a prescribed top-p threshold.

OPD versus RL in multi-task settings. Our multi-task results motivate a more direct comparison between OPD and RL as transfer mechanisms. In RL, positive or negative transfer can be read directly from environment reward across tasks. In OPD, the optimization target remains teacher-derived, so transfer is filtered through what the teacher regards as locally preferable behavior. This distinction may help explain why our multi-task gains are strongest on the math side and why nearby support-set definitions become less uniform in that setting. A matched-task, matched-compute comparison between OPD and RL would help clarify when teacher-guided transfer tracks environment-level generalization and when the teacher–reward gap becomes the bottleneck.

Continual learning as a testbed. Continual learning is another natural setting for OPD. A teacher-guided on-policy objective could act as a retention mechanism while the student adapts to new tasks, but that regime would also stress exactly the issues surfaced in this paper: distribution shift, teacher staleness, and the accumulation of approximation error over long adaptation horizons. Testing OPD there would therefore probe not only whether local support matching mitigates forgetting, but also whether teacher-based objectives remain useful once the student keeps moving away from the teacher’s original domain.

Relation to other stabilization directions. This work is complementary to directions such as reward-hacking mitigation, EMA-anchor stabilization with top-K KL (Zhang & Ba, 2026), perturbation-based off-policy correction (Ye et al., 2026a), and logit-level fusion between teacher and student rollouts (Zhang et al., 2026). These methods address different parts of the same broader problem: how to keep teacher-derived signals useful once teacher and student policies begin to diverge (Li et al., 2026). We view our method as one component in that larger toolbox, rather than as a replacement for those stabilization strategies.

##### C Related Work

On-policy distillation. Many widely used post-training methods for language models rely on off-policy supervision from fixed datasets, including supervised fine-tuning on demonstration data (Ouyang et al., 2022; Guha et al., 2025) and conventional knowledge distillation on teacher-provided targets (Hinton et al., 2015; Sanh et al., 2019). In autoregressive generation, however, such fixed-prefix training creates a mismatch between the contexts seen during training and the prefixes induced by the student’s own test-time rollouts, motivating subsequent work on on-policy distillation (OPD) (Agarwal et al., 2024), where supervision is computed on student-generated trajectories. Agarwal et al. (2024) formulate generalized knowledge distillation for autoregressive models, while Gu et al. (2024) derive an effective on-policy optimization procedure. More recently, OPD and closely related variants have appeared in large-scale post-training pipelines, including Qwen3 (Yang et al.,

- 2025), Thinking Machines Lab (Lu & Lab, 2025), MiMo-V2-Flash (Xiao et al., 2026), and GLM-

- 5 (Zeng et al., 2026), underscoring the practical value of on-policy supervision for reasoning transfer, continual adaptation, and capability recovery across training stages. Subsequent work broadens this family in several directions: Ye et al. (2025), Yang et al. (2026b), and Ko et al. (2026) extend OPD to black-box teachers, more flexible reward formulations, and relaxed imitation objectives, respectively. Ye et al. (2026b) further connect OPD with context distillation, showing that on-policy supervision can also be used to internalize context- or prompt-conditioned behaviors. Related self- or hindsight-distillation approaches replace the external teacher with guidance derived from privileged information or feedback (Zhao et al., 2026; Hübotter et al., 2026; Wang et al., 2026b; Yang et al., 2026a). In contrast, our work does not introduce a new supervision regime; instead, we revisit the standard sampled-token estimator itself and improve its token-level learning signal through teacher top-K local support matching.

KL computation for LLM post-training. Sampled KL estimators in LLM post-training are commonly based on forms such as K1, K2, and K3 (Schulman, 2020). These estimators are lightweight, but they retain only the sampled token and therefore discard the richer nexttoken distribution available from teacher logits in white-box settings. When teacher logits are available, one can in principle match the full next-token distribution, but materializing full-vocabulary logits is often prohibitively memory-intensive for long responses and large models (Hinton et al., 2015; Kim & Rush, 2016; Agarwal et al., 2024). At one extreme, Zhao et al. (2026) use full-vocabulary logit distillation in a self-distillation setting. To reduce the cost of full-distribution matching, Zhang & Ba (2026) derive an unbiased top-K KL estimator that combines exact computation on a top-K set with a sampled correction for the tail. Related self-distillation methods such as SDPO and OpenClaw-RL also adopt top-K approximations for efficiency, but without explicitly incorporating the sampled token into the truncated support (Hübotter et al., 2026; Wang et al., 2026b). Our method instead computes a truncated reverse-KL objective on the teacher’s local top-K support, with both teacher and student distributions renormalized within the selected set. This yields a

simple distribution-level alternative to sampled-token OPD without incurring the cost of full-vocabulary matching.

KL divergence in RL for LLMs. The choice of KL divergence can substantially affect distribution matching in on-policy distillation and related post-training methods for language models. Gu et al. (2024) argue that reverse KL is often better suited than the forward-KL objective commonly used in conventional distillation because it discourages the student from overestimating the teacher’s low-probability regions and tends to yield more precise generation. Agarwal et al. (2024) further show that different divergences induce a quality–diversity trade-off: moving from forward KL toward reverse KL through generalized Jensen–Shannon divergence (JSD) yields increasingly mode-seeking behavior and lower diversity. A related perspective is provided by Chen et al. (2025), who connect the reverse-KL-like, mode-seeking character of on-policy RL to reduced forgetting during posttraining. More recently, Jin et al. (2026a) argue that pure reverse-KL-style OPD can become brittle when the teacher distribution has high entropy, and propose augmenting reverse KL with forward KL on high-entropy tokens to better preserve diversity. At the same time, the preferred divergence can depend on the supervision setting. In self-distillation with privileged information, Zhao et al. (2026) report that forward KL outperforms reverse KL and JSD, whereas SDPO adopts JSD as a stability-oriented design choice (Hübotter et al.,

- 2026). In this work, we adopt a reverse-KL formulation, but focus on a different question: how to obtain a stable local reverse-KL-style training signal under sampled-token OPD.

##### D Bias and variance analysis of token-level versus sequence-level OPD

- D.1 Bias of the token-level estimator Recall the sequence-level estimator in causal return-to-go form

gˆseq =

T

∑

t=1

T

∑

t′=t

rt′ st.

Expanding the inner sum gives

gˆseq =

T

∑

t=1

rtst +

T

∑

t=1

T

∑

t′=t+1

rt′st.

Since the token-level estimator keeps only the first term,

gˆtok =

T

∑

t=1

rtst,

their expectation gap is

E[gˆseq] − E[gˆtok] = E

T

∑

t=1

T

∑

t′=t+1

rt′st .

This makes explicit that token-level OPD removes the future-reward coupling terms and is therefore generally biased with respect to the sequence-level objective.

- D.2 Worst-case variance upper bounds

Assume there exist constants Br, Bs > 0 such that

|rt| ≤ Br, ∥st∥ ≤ Bs for all t. For the token-level estimator,

∥gˆtok∥ ≤

T

#### ∑

|rt| ∥st∥ ≤ TBrBs,

t=1

which implies

E∥gˆtok∥2 ≤ T2Br2Bs2. Using Var(X) ≤ E∥X∥2, we obtain

Var(gˆtok) = O(T2). For the sequence-level estimator, define

T

#### ∑

rt, S =

R =

t=1

T

#### ∑

st, gˆseq = RS.

t=1

Then

|R| ≤ TBr, ∥S∥ ≤ TBs, so

∥gˆseq∥ ≤ T2BrBs, E∥gˆseq∥2 ≤ T4Br2Bs2. Therefore,

Var(gˆseq) = O(T4).

###### D.3 Discussion

The sequence-level estimator is closer to the exact trajectory-level objective, but it couples each score term with many future rewards. In worst-case scaling, this changes variance growth from quadratic to quartic in sequence length. The argument is conservative, but it captures why stronger reward coupling can become problematic in long-horizon training.

##### E Toy experiment details

###### E.1 Environment

We use a two-task one-dimensional continuous-control environment to visualize how stronger return coupling changes OPD optimization. The student policy is a three-layer MLP with roughly 4K parameters. Its input is a three-dimensional vector containing task identity, current position, and normalized time step. The policy outputs the mean and standard deviation of a Gaussian action distribution, and the state transition is

st+1 = st + δ, δ ∼ N (µ, σ).

The two tasks are mirror images of each other: the left task starts from +2 and targets −3, while the right task starts from −2 and targets +3. We first train separate teachers with REINFORCE and then distill them into a shared student with alternating-task OPD.

###### E.2 Gradient variance estimation

At each training step, we split a batch of B = 64 trajectories into M = 8 micro-batches. For each micro-batch m, we compute a loss Lm and the corresponding gradient vector gm on the output layer parameters. We then estimate gradient variance by

1 M

Var(g) =

M

∥gm − g¯∥2 , g¯ =

#### ∑

m=1

M

1 M

#### ∑

gm.

m=1

We use this quantity only as a qualitative proxy, but it is sufficient for comparing relative variance across different γ settings.

###### E.3 Additional Results of Toy Experiments

Figures A1, A2, and A3 report gradient-variance curves and corresponding state-visitation heatmaps for different OPD estimators (γ ∈ {0.0,0.25,0.5,0.75,1.0}) across three random seeds. Although the exact magnitudes vary by seed, the qualitative pattern is consistent. All settings exhibit large variance spikes during early optimization, and larger γ typically remains at a higher variance level later in training. In several runs, the variance under γ = 0.75 or γ = 1.0 stays one to several orders of magnitude above that of smaller γ values. Across runs, token-level OPD (γ = 0) consistently learns trajectories that move toward the target states for both tasks. Intermediate values of γ remain qualitatively similar but become more diffuse. When γ approaches the sequence-level case (γ = 1.0), the learned trajectories often deviate from the desired direction and stabilize around sub-optimal regions.

Left Task (odd iterations)

Right Task (even iterations)

=0 (token-level)

=0 (token-level)

- 101

- 102

- 103

- 104

- 105

- 106

- 107

- 108

- 101

- 102

- 103

- 104

- 105

- 106

- 107

=0.25

=0.25

=0.5

=0.5

- =0.75

- =1.0

- =0.75

- =1.0

GradientVariance

GradientVariance

0 250 500 750 1000 1250 1500 1750 2000

0 250 500 750 1000 1250 1500 1750 2000

Iteration

Iteration

seq-level(gamma=0.5) Left Task (init=2)

seq-level(gamma=1.0) Left Task (init=2)

tok-level Left Task (init=2)

8

1.0

|[Figure 10]| | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

|[Figure 11]| | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

|[Figure 12]| | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

|[Figure 13]| |
|---|---|
| | |
| | |
| | |
| | |

6

4

positions

2

0.8

0

2

4

normalizedvisitation(sqrtscale)

6

0.6

8

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0

time step

time step

time step

seq-level(gamma=0.5) Right Task (init=-2)

seq-level(gamma=1.0) Right Task (init=-2)

tok-level Right Task (init=-2)

8

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
|[Figure 14]| | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
|[Figure 15]| | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
|[Figure 16]| | | | | | | |
| | | | | | | | |

0.4

6

4

positions

2

0

0.2

2

4

6

8

0.0

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0

time step

time step

time step

- Figure A1: Toy experiment with random seed 42: gradient variance and state visitation.

###### Left Task (odd iterations)

###### Right Task (even iterations)

1024

=0 (token-level)

=0 (token-level)

1021

=0.25

=0.25

1020

=0.5

=0.5

1018

- =0.75

- =1.0

- =0.75

- =1.0

1016

1015

GradientVariance

GradientVariance

1012

1012

109

108

106

104

103

100

100

0 250 500 750 1000 1250 1500 1750 2000

0 250 500 750 1000 1250 1500 1750 2000

Iteration

Iteration

###### seq-level(gamma0.5) Left Task (init=2)

###### seq-level(gamma1.0) Left Task (init=2)

###### tok-level Left Task (init=2)

8

1.0

|[Figure 17]| | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

|[Figure 18]| | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

|[Figure 19]| | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

|[Figure 20]| |
|---|---|
| | |
| | |
| | |
| | |

6

4

2

positions

0.8

0

2

4

normalizedvisitation(sqrtscale)

6

0.6

8

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 time step

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 time step

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 time step

###### seq-level(gamma0.5) Right Task (init=-2)

###### seq-level(gamma1.0) Right Task (init=-2)

###### tok-level Right Task (init=-2)

8

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
|[Figure 21]| | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
|[Figure 22]| | | | | | | |
| | | | | | | | |

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
|[Figure 23]| | | | | | | |
| | | | | | | | |

0.4

6

4

2

positions

0

0.2

2

4

6

8

0.0

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 time step

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 time step

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 time step

- Figure A2: Toy experiment with random seed 43: gradient variance and state visitation.

0 250 500 750 1000 1250 1500 1750 2000

Iteration

10 2

10 1

- 100

- 101

- 102

- 103

GradientVariance

Left Task (odd iterations)

=0 (token-level)

=0.25

=0.5

- =0.75

- =1.0

0 250 500 750 1000 1250 1500 1750 2000

Iteration

10 2

10 1

- 100

- 101

- 102

- 103

- 104

GradientVariance

Right Task (even iterations)

=0 (token-level)

=0.25

=0.5

- =0.75

- =1.0

|[Figure 24]| | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 time step

8

6

4

2

0

2

4

6

8

positions

tok-level Left Task (init=2)

|[Figure 25]| | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 time step

seq-level(gamma0.5) Left Task (init=2)

|[Figure 26]| | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 time step

seq-level(gamma1.0) Left Task (init=2)

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
|[Figure 27]| | | | | | | |
| | | | | | | | |

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 time step

8

6

4

2

0

2

4

6

8

positions

tok-level Right Task (init=-2)

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
|[Figure 28]| | | | | | | |
| | | | | | | | |

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 time step

seq-level(gamma0.5) Right Task (init=-2)

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
|[Figure 29]| | | | | | | |
| | | | | | | | |

0.0 2.5 5.0 7.5 10.0 12.5 15.0 17.5 20.0 time step

seq-level(gamma1.0) Right Task (init=-2)

|[Figure 30]| |
|---|---|
| | |
| | |
| | |
| | |

0.0

0.2

0.4

0.6

0.8

1.0

normalizedvisitation(sqrtscale)

- Figure A3: Toy experiment with random seed 2026: gradient variance and state visitation.

##### F Experiment Setups

We build our training pipeline on verl1 (Sheng et al., 2025) and the verl-agent framework2 (Feng et al., 2025), and conduct all experiments on a node with 8 NVIDIA H100 GPUs. We report the settings and hyperparameters used in our experiments in Table A1.

Category Item Math (§ 4.2,§ 4.3) ALFWorld (§ 4.3) WebShop (§ G.1)

Student Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct Qwen2.5-1.5B-Instruct Teacher OpenThinker3-7B GiGPO-Qwen2.5-7B-It-ALFWorld GiGPO-Qwen2.5-1.5B-It-WebShop

Model

Max prompt length 2048 2048 2048 Max response length 16384 512 512 Max turns – 30 15

Environment

Group size 8 8 8 Top-p* 0.9 0.9 0.9 Temperature 1 1 1

Rollout

Top-p 0.9 0.9 0.9 Temperature 1 1 1

Evaluation

Teacher top-K 32 32 32 Optimizer AdamW AdamW AdamW Total training steps† 400 400 60 Learning rate 2 × 10−6 2 × 10−6 2 × 10−6 Warmup steps 0 0 0 Batch size 128 128 128 Mini batch size 64 64 64

Training

- Table A1: Training and evaluation settings. * Rollout top-p is used in local support matching, but not in the sampled-token OPD baseline. † Training steps denote the total number of training updates; in the alternating multi-task setting, 400 total steps correspond to 200 math updates and 200 ALFWorld updates.

G Additional Results

- G.1 Additional Evidence on Single-task WebShop

To test whether the same trend transfers beyond the main math and ALFWorld settings, we additionally evaluate single-task WebShop (Yao et al., 2022) using Qwen2.5-1.5B-Instruct as the student. As shown in Table A2, local support matching improves over sampled-token OPD on both task score and success rate, increasing success rate from 50.0 to 57.8. Although the gap to the teacher remains substantial, the result suggests that the method transfers beyond the main math/agentic settings. We do not include a masking variant in this setting because the teacher is obtained by RL training from the same base model, so tokenizer or special-token mismatch is less likely to be a dominant factor here.

Method Task score Success rate

Qwen2.5-1.5B-It 12.7 2.3 GiGPO-Qwen2.5-1.5B-Webshop (Feng et al., 2025) 81.9 66.4 Sampled-token OPD 73.0 50.0 Ours 75.1 57.8

- Table A2: Results on single-task WebShop with Qwen2.5-1.5B-Instruct as the student. Local support matching improves over sampled-token OPD on both task score and success rate, reducing the gap to the teacher. We do not include a masking variant here because the teacher is obtained by RL training from the same base model, making tokenizer-mismatch effects less central in this setting.

- 1https://github.com/verl-project/verl
- 2https://github.com/langfengQ/verl-agent

- G.2 Training Dynamics and Alignment Figures A4, A5, and A6 provide a more detailed view of the optimization dynamics.

Better learning curves. On math reasoning, our method improves both training reward and evaluation performance across most of training rather than only at the final checkpoint. This pattern holds in both the single-task setting and the alternating multi-task setting.

More stable optimization. Our method yields smaller gradient norms and lower clippingboundary fractions while maintaining sufficient policy entropy, indicating more stable optimization. We also observe that special-token masking substantially reduces the clippingboundary fraction of sampled-token OPD during early and middle training, while having only minor effects on our method.

Improved teacher–student alignment. The teacher–student log-probability gap on sampled tokens also moves closer to zero, suggesting that the truncated local support objective improves alignment even under the sampled-token diagnostic used by the baseline.

0 100 200 300 400

Step

0.15

0.20

0.25

0.30

0.35

0.40

0.45

0.50

TrainScore(DAPO-Math-17k)

ours w/o mask

ours w/ mask

opd w/o mask

opd w/ mask

100 150 200 250 300 350 400

Step

0.16

0.18

0.20

0.22

0.24

TestScore(AIME24_avg@32)

ours w/o mask

ours w/ mask

opd w/o mask

opd w/ mask

Performance on Math Reasoning Tasks

Figure A4: Single-task training curves for math reasoning. Local support matching improves both training reward and evaluation performance over the course of training.

0 50 100 150 200 250 300 350 400

Step

0.2

0.4

0.6

0.8

TrainSuccessRate

(a) Agentic Train

100 150 200 250 300 350 400

Step

0.88

0.90

0.92

0.94

0.96

0.98

TestSuccessRate

(b) Agentic Test

0 50 100 150 200 250 300 350 400

Step

0.10

0.15

0.20

0.25

0.30

0.35

TrainScore(DAPO-Math-17k)

(c) Math Train

100 150 200 250 300 350 400

Step

0.08

0.10

0.12

0.14

0.16

0.18

0.20

TestScore(AIME24_avg@32)

(d) Math Test

ours w/o mask ours w/ mask opd w/o mask opd w/ mask

Figure A5: Multi-task learning curves for ALFWorld and math reasoning. The main gains appear on the math side while agentic performance remains strong.

- G.3 Additional Results of Top-K Variants

We primarily evaluate five top-K variants in our experiments. Below, we present their formulations under unified notation and evaluate them in both single-task math and multitask settings. All experimental setups are kept the same as in the main experiments.

###### G.3.1 Formulations of Five Top-K Variants

Shared notation. For each prompt x, we sample a group of outputs {oi}iG=1 from the student inference policy πinfer. Let yi,t denote the token at position t in output oi, and let

ci,t = (x, yi,<t)

0.9

ours w/o mask

- 1

- 2

- 3

- 4

- 5

ours w/ mask

0.8

opd w/o mask

0.7

opd w/ mask

GradientNorm

###### PolicyEntropy

0.6

0.5

0.4

0.3

ours w/o mask

ours w/ mask

0.2

opd w/o mask

0.1

opd w/ mask

0 50 100 150 200 250 300 350 400

0 50 100 150 200 250 300 350 400

Step

Step

(a) Single-task optimization statistics.

ours w/o mask

ours w/o mask

5.0

3.5

ours w/ mask

ours w/ mask

opd w/o mask

opd w/o mask

4.5

GradientNorm(alfworld)

3.0

opd w/ mask

opd w/ mask

GradientNorm(math)

4.0

2.5

3.5

2.0

3.0

2.5

1.5

2.0

1.0

1.5

0 50 100 150 200 250 300 350 400

0 50 100 150 200 250 300 350 400

Step

Step

(b) Multi-task gradient norms.

###### teacher-studentmeanlog-prob(alfworld)

###### teacher-studentmeanlog-prob(math)

1.0

ours w/o mask

16000

0.20

ours w/ mask

14000

opd w/o mask

0.1

###### ResponseLength(mean)

0.8

0.25

opd w/ mask

ResponseClipRatio

12000

0.30

0.6

0.2

10000

0.35

8000

0.4

0.3

0.40

6000

ours w/o mask

ours w/o mask

ours w/o mask

0.45

4000

0.2

ours w/ mask

ours w/ mask

ours w/ mask

0.4

opd w/o mask

opd w/o mask

opd w/o mask

0.50

2000

opd w/ mask

opd w/ mask

opd w/ mask

0.0

0 50 100 150 200 250 300 350 400

0 50 100 150 200 250 300 350 400

0 50 100 150 200 250 300 350 400

0 50 100 150 200 250 300 350 400

Step

Step

Step

Step

(c) Response length statistics.

(d) Teacher–student log-probability gaps.

- Figure A6: Optimization and alignment diagnostics. Relative to sampled-token OPD, local support matching yields smaller gradient norms, fewer clipping-boundary hits, shorter responses, and teacher–student log-probability gaps closer to zero.

be the corresponding prefix. We write

TKπ(ci,t) = TopKπ(ci,t), TKq(ci,t) = TopKq(ci,t),

for the student and teacher top-K token sets at prefix ci,t. All variants below share the same rollout-level aggregation:

|oi|

G

1 ∑iG=1 |oi|

#### ∑

#### ∑

ℓ(ci,t, yi,t) .

L[ℓ] = Ex,{oi}∼πinfer

t=1

i=1

They differ only in how the local loss ℓ(ci,t, yi,t) is defined.

Renormalized local reverse-KL. For any support set S(ci,t) ⊆ V, define the renormalized student and teacher distributions on S(ci,t) by

πθ(v | ci,t) ∑u∈S(ci,t) πθ(u | ci,t)

q(v | ci,t) ∑u∈S(ci,t) q(u | ci,t)

π¯θ(v | ci,t; S) =

, q¯(v | ci,t; S) =

###### , v ∈ S(ci,t).

The corresponding local reverse-KL is

π¯θ(v | ci,t; S) q¯(v | ci,t; S)

ℓrenorm(ci,t;S) = ∑

π¯θ(v | ci,t; S) log

.

v∈S(ci,t)

- Variant 1: teacher top-K renormalization (our method). Our default local support matching (LSM) method uses the teacher top-K support

SLSM(ci,t) = TKq(ci,t), and defines

ℓLSM(ci,t, yi,t) = ℓrenorm(ci,t; SLSM). The overall objective is

LLSM = L[ℓLSM].

- Variant 2: student top-K + sampled token + renormalization. For this variant, the support is obtained by augmenting the student top-K set with the sampled token when needed:

Sπ+y(ci,t) = TKπ(ci,t), yi,t ∈ TKπ(ci,t),

TKπ(ci,t) ∪ {yi,t}, yi,t ∈/ TKπ(ci,t). We then define

ℓπ+y(ci,t, yi,t) = ℓrenorm(ci,t; Sπ+y), and

Lπ+y = L[ℓπ+y].

- Variant 3: teacher top-K + sampled token + renormalization. Similarly, this variant augments the teacher top-K set with the sampled token:

Sq+y(ci,t) = TKq(ci,t), yi,t ∈ TKq(ci,t),

TKq(ci,t) ∪ {yi,t}, yi,t ∈/ TKq(ci,t). Its local and rollout-level losses are

ℓq+y(ci,t, yi,t) = ℓrenorm(ci,t; Sq+y), Lq+y = L[ℓq+y].

EMA-PG-style local loss. The remaining two variants follow the EMA-PG decomposition (Zhang & Ba, 2026), where a top-K truncated head term is combined with an importance-weighted sampled-token tail correction to obtain an unbiased estimator of the full-vocabulary reverse-KL. For any support set S(ci,t) ⊆ V, define the truncated head term

ℓhead(ci,t;S) = ∑

v∈S(ci,t)

πθ(v | ci,t) log

πθ(v | ci,t) q(v | ci,t)

.

Let sg(·) denote stop-gradient, and define

ri,t =

πθ(yi,t | ci,t) sg(πθ(yi,t | ci,t))

, si,t = sg

πθ(yi,t | ci,t) πinfer(yi,t | ci,t)

,

where si,t is the sampling-policy correction (optionally clipped, as in EMA-PG). The sampledtoken tail correction is

ℓtail(ci,t, yi,t; S) = si,t 1[yi,t ∈/ S(ci,t)] ri,t sg log

πθ(yi,t | ci,t) q(yi,t | ci,t)

.

The corresponding local EMA-PG loss is

ℓEMA(ci,t, yi,t; S) = ℓhead(ci,t; S) + ℓtail(ci,t, yi,t; S).

- Variant 4: student top-K + EMA-PG correction. This variant uses the student top-K support

Sπ-EMA(ci,t) = TKπ(ci,t), and defines

ℓπ-EMA(ci,t, yi,t) = ℓEMA(ci,t, yi,t; Sπ-EMA), Lπ-EMA = L[ℓπ-EMA].

- Variant 5: teacher top-K + EMA-PG correction. This variant instead uses the teacher top-K support

Sq-EMA(ci,t) = TKq(ci,t), and defines

ℓq-EMA(ci,t, yi,t) = ℓEMA(ci,t, yi,t; Sq-EMA), Lq-EMA = L[ℓq-EMA].

Summary. The five formulations differ only in how the local support is constructed and whether the sampled token is incorporated through support augmentation or through an EMA-PG-style tail correction. Variants 1–3 compute a renormalized reverse-KL on the resulting local support, while Variants 4–5 replace support renormalization by a truncated head term plus a sampled-token correction (Zhang & Ba, 2026).

###### G.3.2 Results under Single-task and Multi-task Settings

avg@32 pass@1 AIME24 MATH500 AIME24 AIME25 Minerva OlympiadBench Avg.

Method

teacher top-K 23.6 80.4 23.3 26.7 34.2 43.9 41.7 student top-K w/ sampled token (renorm) 22.3 82.4 30.0 16.7 35.7 44.9 41.9 teacher top-K w/ sampled token (renorm) 22.4 81.6 26.7 23.3 36.4 46.7 42.9 student top-K w/ sampled token (EMA-PG) 21.1 82.0 26.7 23.3 32.4 41.8 41.2 teacher top-K w/ sampled token (EMA-PG) 20.7 81.4 16.7 16.7 32.7 43.0 38.1

(a) Five top-K variants in the single-task setting.

Agentic Reasoning ALFWorld MATH500 AIME24 AIME25 Minerva OlympiadBench Avg.

Method

teacher top-K 95.3 82.0 33.3 16.7 32.7 44.0 41.7 student top-K w/ sampled (renorm) 95.3 65.6 10.0 10.0 25.0 31.6 28.4 teacher top-K w/ sampled (renorm) 94.5 63.2 10.0 10.0 21.0 30.1 26.9 student top-K w/ sampled token (EMA-PG) 93.8 76.0 20.0 16.7 29.4 39.1 36.2 teacher top-K w/ sampled token (EMA-PG) 96.1 72.8 16.7 13.3 26.8 38.7 33.7

(b) Five top-K variants in the multi-task setting. We report pass@1 for all math reasoning benchmarks. The final column averages pass@1 metrics only.

Table A3: Ablation of five top-K variants under single-task and multi-task settings.

As shown in Table A3, teacher top-K performs reasonably well in the single-task setting and remains strong in the multi-task setting. By contrast, the EMA-PG variants, despite their unbiasedness motivation, do not translate into better empirical performance here. We therefore treat the variant comparison as a diagnostic result rather than a definitive ranking of support-set objectives.

##### H Qualitative OPD reward-hacking case study

To complement the representative failures in the main text, we summarize a longer trajectory from multi-task training under sampled-token OPD. Read chronologically, the case exhibits the same pattern in several forms: the model continues reasoning after a valid answer has already been reached, falls into repetition loops such as wait, drifts into malformed continuations, and still receives high local teacher probability on those tokens. In these visualizations, uncolored tokens indicate positions where the displayed teacher and student probabilities are approximately matched.

The failure first appears as over-continuation. Even after the answer is effectively available, the local signal continues to place substantial mass on generic reasoning fillers and connective tokens, encouraging the model to keep going instead of stopping cleanly. The same pattern later appears on prefixes such as confirm, where the local signal still favors additional verification rather than termination. Some of this behavior may also reflect the teacher’s own output habits. Figure A7 illustrates several representative cases.

RESEARCHVISUALIZATION· TOKENPROBABILITYTRACE

TeachervsStudentDistribution

MEANSIGNEDDIFF(P-R)

This layout focuses on the readable question/response view from the polished reference example while keping the original statistics and debug metadata in a colapsible panel.

+The po0licy is. m0ore confi1dent2 than the reference on average. The largest token-level absolute gap reaches 0. 91.

RESEARCHVISUALIZATION· TOKENPROBABILITYTRACE

Step5 Sample0 Taskmath Generated2026-02-24 0:27:58

TeachervsStudentDistribution

MEANSIGNEDDIFF(P-R)

###### Sumary Statistics and Debug Metadata

This layout focuses on the readable question/response view from the polished reference example while keping the original statistics and debug metadata in a colapsible panel.

+The po0licy is. m0ore c4onfide7nt th2an the reference on average. The largest token-level absolute gap reaches 0. 98.

###### QUESTION

Step9 Sample0 Taskmath Generated2026-02-24 0:51:51

Let $p(x)$ be a polynomial of degree 100 such that \begin{align*}

- p(1) &= 1, \\
- p(2) &= 2, \\
- p(3) &= 3, \\ &\dots, \\

###### Sumary Statistics and Debug Metadata

- p(99) &= 99, \\
- p(100) &= 100, \\
- p(101) &= 102. \end{align*}Find $p(102).$

###### QUESTION

Find the least positive integer $k$ so that $k + 25973$ is a palindrome (a number which reads the same forward and backwards).

###### Teacher stronger Student stronger

###### RESPONSE

###### Teacher stronger Student stronger

\ ] The closest and smallest \( k \ ), when added to 2 5 9 7 3 , will result in the palindrome 2 6 0 0 0 . Therefore , \( \ boxed { 2 7 } \ ) is the least positive integer \( k \ ). </ think > To find the least positive integer \( k \ ) such that \( k + 2 5 9 7 3 \ ) is a palindrome , we need to identify the nearest palindrome to 2 5 9 7 3 first . A palindrome is a number that reads the same forward as backward . The closest larger palindrome is 2 6 0 0 0 , as 2 6 0 0 0 is the smallest number greater than 2 5 9 7 3 that is a palindrome . Let 's verify that 2 6 0 0 0 is a palindrome :

###### RESPONSE

\[ c \ cdot 1 0 0 ! = 1 \ impl ies c = \ frac { 1 }{ 1 0 0 ! } \ ] So , the polynomial \( q (x ) \ ) is : \[ q (x ) = \ frac { 1 }{ 1 0 0 ! } ( x - 1 )( x - 2 )\ cd ots (x - 1 0 0 ) \ ] Now , recall that \( p (x ) = q (x ) + x \ ). Therefore , to find \( p ( 1 0 2 ) \ ), we substitute \( x = 1 0 2 \ ) into the equation :

- \[ p ( 1 0 2 ) = q ( 1 0 2 ) + 1 0 2 \ ] We need to calculate \( q ( 1 0 2 ) \ ). Using the expression for \( q (x ) \ ):

- \[ q ( 1 0 2 ) = \ frac { 1 }{ 1 0 0 ! } ( 1 0 2 - 1 )( 1 0 2 - 2 )\ cd ots ( 1 0 2 - 1 0 0 ) \ ] This product is : \[ q ( 1 0 2 ) = \ frac { 1 }{ 1 0 0 ! } \ cdot 1 0 1 \ cdot 1 0 0 \ cdot 9 9 \ cdot \ ld ots \ cdot 2 \ ] Notice that the numerator is the product of the numbers from 2 to 1 0 1 , which is \ (\ frac { 1 0 1 ! }{ 1 }\ ) or 1 0 1 !. The denominator is

- Reading 2 6 0 0 0 backwards gives 0 0 0 6 2 , which is 2 6 0 0 0 ( ign oring the leading zeros ). Now , we need to find the difference between 2 6 0 0 0 and 2 5 9 7 3 to determine \( k \ ): \ [ k = 2 6 0 0 0 - 2 5 9 7 3 = 2 7 \ ] Next , let 's check if there is a smaller \( k \ ) that also results in a palindrome . We can check the closest smaller candidate 2 6 0 0

- 1 0 0 !. Therefore , the fraction \ (\ frac { 1 0 1 ! }{ 1 0 0 ! }\ ) simpl ifies to 1 0 1 . Thus : \[ q ( 1 0 2 ) = \ frac { 1 0 1 ! }{ 1 0 0 ! } \ cdot \ frac { 1 }{ 1 0 0 ! } = \ frac { 1 0 1 ! }{ ( 1 0 0 !) ^ 2 } = \ frac { 1 0 1 \ cdot 1 0 0 ! }{ 1 0 0 ! \ cdot 1 0 0 ! } = \ frac { 1 0 1 \ cdot 1 0 0 ! }{ 1 0 0 ! ^ 2 } = \ frac { 1 0 1 }{ 1 0 0 ! } \ cdot \ frac { 1 0 0 ! }{ 1 0 0 ! } = \ frac { 1 0 1 }{ 1 0 0 } =

- 1 0 1 \ ] Therefore : \[ p ( 1 0 2 ) = q ( 1 0 2 ) + 1 0 2 = 1 + 1 0 2 = 1 0 3 \ ] So , the value of \( p ( 1 0 2 ) \ ) is 1 0 3 . </ think >

- 0 - 1 = 2 5 9 9 9 for example , but since 2 5 9 9 9 is not a palindrome , we can be confident that 2 7 is the smallest \( k \ ). To confirm , let 's add 2 7 to 2 5 9 7 3 : \ [

- 2 5 9 7 3 + 2 7 = 2 6 0 0 0 \ ]

- 2 6 0 0 0 is indeed a palindrome , as shown before . Therefore , the least positive integer \( k \ ) is 2 7 . H ence , the least positive integer \( k \ ) is :

\ [ \ boxed { 2 7 } \ ]

\ boxed { 2 7 } is the final answer . </ think > To find the least positive integer \( k \ ) such that \( k + 2 5 9 7 3 \ ) is a palindrome , we first need to find the nearest palindrome to 2 5 9 7 3 .

- 1 . ** Ident ify the nearest palindrome ** :

The value of \( p ( 1 0 2 ) \ ) is \ boxed { 1 0 3 }. </ think > </ solution > </ think > </ solution > </ think > </ solution > </ think ></ think ></ think ></ think ></ think ></ think ></ think > </ think ></ think ></ think ></ think ></ think ></ think > </ think ></ think ></ think ></ think ></ think ></ think ></ think ></ think ></ think ></ think ></ think ></ think ></ think ></ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </ think > </

(a) High teacher probability on generic reasoning fillers (implies) at step 5.

(b) Teacher and student remain well aligned even after the answer is effectively available.

- The nearest palindrome greater than 2 5 9 7 3 is 2 6 0 0 0 , as verified by the property of pal ind rom es ( it reads the same

- Figure A7: Even after the student has effectively reached an answer, the teacher can still assign high conditional probability to low-information continuations.

The trajectory then develops into hesitation loops and low-information continuations. Repeated wait tokens, punctuation-heavy continuations, and other semantically weak fillers can remain locally rewardable even after the overall trajectory has become unproductive. This is consistent with the repetition-loop discussion in Section 2.2. We provide two similar cases in Figure A8.

Finally, once the student drifts further off-distribution, the local signal can remain misleadingly positive rather than self-correcting. In the case study, this appears as degenerate continuations and malformed non-English text, yet many tokens still receive high teacher probability. An example is shown in Figure A9.

RESEARCHVISUALIZATION· TOKENPROBABILITYTRACE

TeachervsStudentDistribution

Preprint. Under review.

###### MEANSIGNEDDIFF(P-R)

This layout focuses on the readable question/response view from the polished reference example while keping the original statistics and debug metadata in a colapsible panel.

+The po0licy is.1 more con4fiden7t than the reference on average. The largest token-level absolute gap reaches 0. 95.

Step31 Sample0 Taskmath Generated2026-02-24 02:51:46

###### Sumary Statistics and Debug Metadata

###### QUESTION

Square \(ABCD\) has side length \(2\). A semicircle with diameter \(\overline{AB}\) is constructed inside the square, and the tangent to the semicircle from \(C\) intersects side \(\overline{AD}\) at \ (E\). The length of \(\overline{CE}\) can be expressed in the form \(\frac{k}{m}\), where \(\frac{k}{m}\) is a simplified fraction. Please find the value of \(k + m\).

###### Teacher stronger Student stronger

###### RESPONSE

Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait . Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait , Wait ,

- (a) The teacher may fail to penalize, and sometimes even reinforce, repetitive generation.

RESEARCHVISUALIZATION· TOKENPROBABILITYTRACE

TeachervsStudentDistribution

This layout focuses on the readable question/response view from the polished reference example while keping the original statistics and debug metadata in a colapsible panel.

Step39 Sample0 Taskmath Generated2026-02-24 03:34: 5

MEANSIGNEDDIFF(P-R)

+The po0licy is. m0ore c5onfid1ent t7han the reference on average. The largest token-level absolute gap reaches 0. 97.

Sumary Statistics and Debug Metadata

QUESTION

One test is a multiple choice test with $5$ questions, each having $4$ options. There are $2000$ candidates, each choosing only one answer for each question. Find the smallest possible integer $n$ such that a student's answer sheet, labeled $n$, has the following property: The answer sheet contains four sheets, and any two of these four sheets have exactly the same three answers.

Teacher stronger Student stronger

RESPONSE

Hmm , I need four answer sheets such that any two have exactly three entries the same , so another way to think about this is that for any pair of sheets , 3 out of 5 entries are the same , and 2 are different . This seems to be a very restrictive condition .

Wait , maybe I can think of it in terms of combinations . When you have two sheets which have three answers the same , how many possible combinations are there for that pair ? If we have three questions matching and two different , then there are 5 choose 3 ( the number of questions where the answers match ), which is 1 0 . But then once you pick three positions , the answers on those positions can be anything , but the other positions need to be different .

However , I think that to pair any two sheets , for them to share three answers , they need to also have their unique answers on the other two question positions . So that might require some careful consideration .

There is a problem because I have 4 sheets , so there are 4 choose 2 = 6 pairs of sheets , and each of these pairs must have three answers the same .

Wait , maybe I can first think of a smaller example , and then maybe generalize . Wait , let me see . Let me try to think of a smaller example especially . Suppose I have two answer sheets . The requirement would be that they share exactly 3 correct answers . That would mean , if the first sheet is [ AB CD ]. Then the second sheet has to have 3 of the same answers as the first one .

But here , I can take two different sheets , for instance , [ 0 0 0 0 0 ] and [ 0 0 0 0 1 ], which have 4 answers the same . So in that case , it does not satisfy the condition . So maybe my first example fails . Wait , so if I have two sheets , they cannot have all four choices the same . They all must share exactly three same answers . Let me think of an example .

Let me see . For two sheets to have three answers the same , and two different , if I have A =[ 0 0 0 0 0 ] and B has three 0 's and two different . Suppose B =[ 0 0 0 1 1 ]. That would work .

Wait , but is this general izable ? Let me see . Let me try to express two answer sheets more formally . Let sheet A be ( a 1 , a 2 , a 3 , a 4 , a 5 ) and sheet B is ( b 1 , b 2 , b 3 , b 4 , b 5 ). To have three answers the same :

The intersection between the two sheets ( in terms of questions they answered the same ) would be three elements . So , for example , if A is [ 0 , 0 , 0 , 0 , 0 ] and B is [ 0 , 0 , 1 , 1 , 1 ], then they have answers the same for the first three questions , which is 3 out of 5 .

Wait , so if I construct two answer sheets like that , then to get a third one that also has three of the same , I could do the same for th

- (b) Training can also produce overlong chain-of-thought traces with substantial low-quality content, a pattern that may partly reflect the teacher’s output style.

- Figure A8: Loops, overlong chain-of-thought traces, and punctuation-heavy continuations can remain locally rewardable.

RESEARCHVISUALIZATION· TOKENPROBABILITYTRACE

TeachervsStudentDistribution

###### MEANSIGNEDDIFF(P-R)

This layout focuses on the readable question/response view from the polished reference example while keping the original statistics and debug metadata in a colapsible panel.

+The po0licy is. m0ore c5onfid1ent t7han the reference on average. The largest token-level absolute gap reaches 0. 97.

Preprint. Under review.

Step39 Sample0 Taskmath Generated2026-02-24 03:34: 5

###### Sumary Statistics and Debug Metadata

###### QUESTION

One test is a multiple choice test with $5$ questions, each having $4$ options. There are $2000$ candidates, each choosing only one answer for each question. Find the smallest possible integer $n$ such that a student's answer sheet, labeled $n$, has the following property: The answer sheet contains four sheets, and any two of these four sheets have exactly the same three answers.

###### Teacher stronger Student stronger

###### RESPONSE

- A [ 0 , 0 , 0 , 0 , 0 ] （ 全 零 ）

- B = [ 0 , 0 , 1 , 1 , 0 ] （ 第⼀ 、 ⼆ 、 四 、 五 位置 相同 ， 第三 位置 不同 ）

- C = [ 0 , 0 , 0 , 1 , 1 ] （ 第⼀ 、 ⼆ 、 三 、 四 位置 相同 ， 第五 位置 不同 ）

- D = [ 0 , 0 , 0 , 0 , 1 ] （ 第⼀ 、 ⼆ 、 三 、 五 位置 相同 ， 第四 位置 不同 ） 但是 ， 这 四个 向 量 并不 满⾜ 条件 ， 因为 B , C , D 之间的 配 对 并⽆ 三个 相同的 元素 。 需要 调整 。 需要 更 精细 定义 。 实际上 ， 我们 可以通过 构建 不同 类型的 向 量 ， 通过 坐标 交错 变化 来 实现 。 以下是 构建 这 四个 向 量 的⼀个 巨⼤ 进展 ， 尝试 写下 可能 的 满⾜ 条件 的 四 组 向 量 之⼀ ： 假设 有很多 选项 ， ⽐如 ：

- A 1 = [ 0 , 0 , 0 , 0 , 0 ]

- A 2 = [ 0 , 0 , 0 , 0 , 1 ] （ 第⼀ 列 的 第五 个 位置 不同 ）

- A 3 = [ 0 , 0 , 0 , 1 , 0 ] （ 第⼆ 列 的 第四 位置 不同 ）

- A 4 = [ 0 , 0 , 0 , 1 , 1 ] （ 第⼆ 列 的 第四 位置 和 第五 列 的不同 ） 让 A 1 -A 4 确保 每个 两 两 之间 相差 2 个 位 。 上述 四个 向 量 ⽆法 满⾜ ， 让我们 换 ⼀种 ⽅法 。 举例 说 ， 如果 ：

- A 1 = [ 0 , 0 , 0 , 0 , 0 ]

- A 2 = [ 1 , 1 , 1 , 1 , 1 ] A 1 和 A 2 位置 完全不同 ， 有 五 处 不同 。

- A 3 = [ 0 , 0 , 0 , 0 , 1 ] 与 A 1 同 ， 只是 第五 项 不同 ， 与 A 2 同 前 四 ， 第五 项 不同 。

- A 4 = [ 0 , 0 , 0 , 1 , 0 ] A 3 的 前 四 相同 ， 第五 项 也 不同 。 这⾥的 A 1 和 A 2 是 ⾮常 不同的 ， 有 5 个 不同的 位 ; A 1 和 A 2 是 不 满⾜ 条件 的 ， 因此 我 似乎 需要 重新 组织 。 似乎 我 需要 重新 思考 。 将 这个问题 作为 集合 论 问题 ， 看 是否 可以⽤ 鸽 笼 原理 等 其他 策略 。 既然 问题 是要 找 四 张 卡⽚ ， 每 张 卡⽚ 有 5 个 答案 ， 每 两个 卡⽚ 共享 3 个 答案 。

这 让我 想到 ， 可能 可以 构造 ⼀个 乘 积 空间 。 ⽐如 ， 先 考虑 三 元 情况 ， 即 每个 问题 只有 三个 选项 。 设 选项 为 0 , 1 , 2 ， 那么 有 3 ^ 5 = 2 4 3 种 可能 的 试卷 ， 每 两张 试卷 之间 最多 可以 有 5 - 2 = 3 个 不同的 位置 。 但 这⾥是 4 选项 的情况 ， 即 2 ^ 2 = 4 , 2 ^ 5 = 3 2 个 答案 组合 ， 我 需要 在 尽可能 ⼩ 的 组合 下 满⾜ 题 意 。 这 需要 每个 答案 的位置 最多 有 2 个 不同 。

但 这⾥ 每个 卡⽚ 是 5 位 ， 每个 位 上 可能 有 4 种 选择 ， 4 ^ 5 = 1 0 2 4 可能 的 组合 。 我要 找出 4 个 组合 ， 使得 每 两个 的 三个 位 相同 。

那么 它们 必须 是 分类 化的 ， 即 我能 设置 ⼀些 模板 ， 使得 每个 模板 重 叠 三个 位 ， 但 显 性 不同 两 位 。 我 需要 更好的 构造 。 假设 四 张 试卷

- Figure A9: The teacher still assigns high probability to several tokens after the student drifts into malformed non-English output.

