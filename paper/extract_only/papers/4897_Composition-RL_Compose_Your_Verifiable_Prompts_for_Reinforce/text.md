[Figure 1]

April 23, 2026

### Composition-RL: Compose Your Verifiable Prompts for Reinforcement Learning of Large Language Models

Xin Xu12 Clive Bai1 Kai Yang1 Tianhao Chen2 Yangkun Chen1 Weijie Liu1 Hao Chen2 Yang Wang3 Saiyong Yang1 Can Yang2

## arXiv:2602.12036v2[cs.CL]22Apr2026

#### Abstract

Large-scale verifiable prompts underpin the success of Reinforcement Learning with Verifiable Rewards (RLVR), but they contain many uninformative examples and are costly to expand further. Recent studies focus on better exploiting limited training data by prioritizing hard prompts whose rollout pass rate is 0. However, easy prompts with a pass rate of 1 also become increasingly prevalent as training progresses, thereby reducing the effective data size. To mitigate this, we propose Composition-RL, a simple yet useful approach for better utilizing limited verifiable prompts targeting pass-rate-1 prompts. More specifically, Composition-RL automatically composes multiple problems into a new verifiable question and uses these compositional prompts for RL training. Extensive experiments across model sizes from 4B to 30B show that CompositionRL consistently improves reasoning capability over RL trained on the original dataset. Performance can be further boosted with a curriculum variant of Composition-RL that gradually increases compositional depth over training. Additionally, Composition-RL enables more effective cross-domain RL by composing prompts drawn from different domains. Codes, datasets, and models are available at https://github.com/XinXUUSTC/Composition-RL.

ing both text-only reasoning (Luo et al., 2025; Yang et al.,

- 2025a; Liu et al., 2025b; Cai et al., 2025) and multimodal question answering (Meng et al., 2025; Xiao et al., 2025). Rapid progress in RLVR, including improved optimization algorithms (Nan et al., 2025; Yu et al., 2025; Chen

- et al., 2025a; Liu et al., 2025b), more efficient training frameworks (Sheng et al., 2024; Fu et al., 2025; Zhu et al., 2025b), and techniques to mitigate training–inference mismatch (Yao et al., 2025; Qi et al., 2025), has contributed to the strong slow-thinking ability of large reasoning models (LRMs), often manifested as longer chain of thought (CoT) (Wei et al., 2022). At its core, RLVR relies on large collections of training prompts paired with ground-truth answers to enable verifiable reward computation during training (Hu et al., 2025; He et al., 2025b; Luo et al., 2025).

Prompts with 0/1 rollout accuracy yield zero gradient signals in RLVR algorithms (Yu et al., 2025), substantially reducing the number of available informative prompts during training. However, collecting and cleaning additional high-quality training prompts is often expensive (He et al., 2025b; Zeng et al., 2025). To mitigate this, prior work has primarily focused on better leveraging hard prompts with zero success rate, via advantage shaping (Le et al., 2025; Nan et al., 2025), allocating more rollouts (Yang et al., 2025c; Li et al., 2025c), and hint-based augmentation (Chen

- et al., 2025b; Li et al., 2025a). Nevertheless, while all-zero prompts constitute some fraction of the training set, as training progresses, an increasing proportion of prompts attain rollout accuracy of 1. This motivates the need for methods that can better exploit these “easy” prompts.

#### 1. Introduction

After the advent of OpenAI-o1 (Jaech et al., 2024) and DeepSeek-R1 (Guo et al., 2025), Reinforcement Learning with Verifiable Rewards (RLVR) has reshaped the training lifecycle of large language models (LLMs), improv-

1HY, Tencent 2The Hong Kong University of Science and Technology 3The University of Hong Kong. Correspondence to: Can Yang <macyang@ust.hk>, Saiyong Yang <stevesyang@tencent.com>.

Preprint. April 23, 2026.

In this work, we propose Composition-RL, a simple yet effective approach for better utilizing limited verifiable training prompts by transforming simple prompts into more challenging ones. We first introduce a procedure for composing K existing prompts into new prompts (Section 3.1) and empirically show that prompt composition can, to some extent, mitigate the growing number of “too-easy” prompts (Section 3.2). We then formalize Composition-RL as RL training on compositional prompts (Section 3.3); an overview is provided in Figure 1. As shown in Figure 1, CompositionRL outperforms RL training on the original prompts, with

|𝒒𝟏: What is the sum of … 𝒅𝟏: Define X is the sum of<br><br>the value(s) of 𝑛 that satisfy the equation 2𝑛 − 7 = 3.<br><br>𝒒𝟐 : Simplify 3( 𝟓𝒑 + 𝒀 − 2𝑝 ∗ 4) + (4 − 1/3)(6𝑝 −<br><br>9) to the form 𝑎𝑝 − 𝑏, where a and b are positive.<br><br>|𝒒𝟏:𝟐:What is the sum of the value(s) of 𝑛 for which 2𝑛 − 7 = 3?<br><br>Define X is the sum of the value(s) of 𝑛 that satisfy the equation 2𝑛 − 7 = 3. Y is -6 more than X. Simplify 3( 𝟓𝒑 + 𝒀 − 2𝑝 ∗ 4) + (4 − 1/3)(6𝑝 − 9) to the form 𝑎𝑝 − 𝑏, where a and b are positive. 𝑔𝑡 :  = 𝑔𝑡 = 13𝑝 − 30<br><br>91:2:What is the sum of the value(s) of n<br><br>for which |2n-7|=3?<br><br>Define Xis the sum of the value(s) ofn<br><br>that satisfy the equation |2n-7l=3.<br><br>Y is -6 more than X.<br><br>Simplify 3((5p+Y)-2p*4)+(41/3)(6p-9)to the form ap-b, where<br><br>a and b are positive.<br><br>gt1:2= gt2=13p-30|
|---|
<br><br>(1) Modify 𝑞 with 𝑔𝑡<br><br>(2) Modify 𝑞<br><br>(3) Connect 𝑞 and 𝑞<br><br>Constructed Constructed Compositional Prompt<br><br>(1)Modify q1with gt<br><br>q1: What is the sum of... (3)Connect q1 and q<br><br>(2)Modify q<br><br>q2:Simplify 3((5p+Y)a and b are positive.<br><br><br>ConstructedCompositionalPrompt<br><br>with gt<br><br>sum of...<br><br>fy q<br><br>(5p+Y)-<br><br>tive.<br><br>d :Define X is the sum of<br><br>2p*4)+(4-1/3)(6p一<br><br>the sum of<br><br>1/3)(6p一<br><br>the value(s) of n that satisfy<br><br>9) to the form ap-b, where<br><br>n that satisfy<br><br>p-b, where<br><br>the equationn-7|=3.|2n-7|=3.|
|---|

###### Original Prompts

Original Prompts

# →

91:What is the sum of the

Composition-RL

𝒒𝟏: What is the sum of the value(s) of 𝑛 for which

Composition-RL

value(s) of n for which

|2n-7|=3?

2𝑛 − 7 = 3? 𝒈𝒕𝟏: 7

RL Training

[Figure 2]

gt17

RL Training

q2:Simplify 3((5p+1)-

𝒒𝟐: Simplify 3( 5𝑝 + 1 − 2𝑝 ∗ 4) + (4 − 1/3)(6𝑝 − 9) to the form 𝑎𝑝 − 𝑏,

2p*4)+(4-1/3)(6p-

- 9) to the form ap-b, where a and b are positive.

40 Original Composition-RL-4BL-4B 37.937.9

Composition

35- Curriculum ■34.6

Beyond-80/20-8BBeyond-80/20-8B

30

Alpha-RL-8B:28.3▲B:28.3▲

25 RL-ZVP-8B:24.6●B:24.6●

20-

15-

- 10 Depth1→Depth2Depth2→Depth3

where a and b are positive.

engineering

54

pass@1(%)in AIME 24

52

law 50 chemistry

30.129.4 287 49 69 717 73 一 Math-Only

64 66 — Math-Composition

27.9 27.2

— Physics-Math-Composition

72 65 一 Math-then-Physics

75 67 — MIX-Training Z2 68 79 70

math physics

71

0 100 200 300 400 500 600

- Figure 1. Overview of Composition-RL. Top: an example of composing two math problems, illustrating the high-level idea of CompositionRL. Bottom left: pass@1 (%) on AIME24 versus training steps for different methods, summarizing key findings in Sections 4.2 and 4.3. Bottom right: cross-topic results on MMLU-Pro subjects with the top-5 largest sample sizes, highlighting the main finding in Section 4.4.

RLVR. RLVR optimizes the expected verifiable reward: maxθ Eq∼D JRLVR(θ,q) (= Eq∼D, r∼π

increasing performance gains when combined with a curriculum over compositional depth K. Moreover, composing prompts from different domains shows strong potential for cross-domain RL training. Our contributions can be summarized as follows: ❶ We propose Composition-RL, an approach that performs RL on composed prompts that are automatically transformed from existing ones. ❷ Extensive experiments on 4B-30B LLMs demonstrate the effectiveness of Composition-RL and the curriculum variant of Composition-RL. ❸ We show that RL on composed prompts spanning physics and math is more effective than simply mixing training problems, regardless of whether sequential or joint training. ❹ We analyze the reasons behind the success of Composition-RL through the lenses of compositional generalization and implicit process supervision.

θ(·|q) v(q,r) ). A standard policy gradient estimator (Sutton et al., 1999) is:

gθ(q,r) = A(q,r) · ∇θ log πθ(r|q), (1)

where A(q,r) = v(q,r) − b(q) is called “advantage” and b(q) is a baseline function that depends only on the query q. Group Relative Policy Optimization (GRPO) (Shao et al., 2024) approximates the advantage by sampling a group of G responses {r1,...,rG} from the old policy πθ

(· | q):

old

v(q,ri) − mean {v(q,rj)}Gj=1 std {v(q,rj)}Gj=1

Aˆi =

. (2)

Then the objective of GRPO becomes JGRPO(θ) = Eq∼D JGRPO(θ,q) and JGRPO(θ,q) is defined as follows1:

#### 2. Preliminary

|ri|

G

1

min ii,t(θ) Aˆi, clip ii,t(θ), 1 − ϵ, 1 + ϵ A ˆi ,

Notation. We denote an LLM parameterized by θ as a policy πθ. Let q be an input query (i.e., a prompt) and D be the set of all queries. Given a response r = (r1,...,r|r|) to q, the policy likelihood can be written as πθ(r | q) =

G i=1 |ri|

t=1

i=1

(3)

and the token-level importance ratio is given by

|r|

t=1 πθ(rt | q,r<t), where r<t = (r1,...,rt−1) and |r| is the number of tokens in r. Each (q,r) can be evaluated by a verifier v(q,r) ∈ {0,1}, which indicates whether r matches the ground-truth answer of q (denoted as gt).

πθ(ri,t | q,ri,<t) πθ

. (4)

ii,t(θ) =

(ri,t | q,ri,<t)

old

1We adopt a more-commonly used version with token-level normalization suggested by (Yu et al., 2025).

0.8

Original

Composition

0.7

0.6

| |
|---|

solve_allratio

0.5

Avg@8

0.4

0.3

0.2

0.1

| |
|---|

0.0

0 50 100 150 200 250

Training Steps

| |Original Composition<br><br>| | |
|---|---|---|---|
| |92.3 94.6<br><br>79.2| | |
| |72.6| | |
| | | | |
| | | | |
| | | | |
| | | | |

100

80

60

40

20

0

OpenMath-Reasoning-1.5B JustRL-1.5B

Model

- Figure 2. Visualization of meta-experiments. Left: solve all ratio curve for RL of Qwen3-4B-Base with original prompts (MATH12K) versus compositional prompts. Right: avg@8 accuracy on a subset of MATH500 and its corresponding compositional test prompts.

Dynamic Sampling. In practice, GRPO objective can be approximated by JˆGRPO(θ) = Eq∼B JGRPO(θ,q) , where B ⊂ D denotes a sampled mini-batch of prompts at a given training step. When a prompt has an empirical success rate of 0 or 1 (i.e., all sampled responses are incorrect or all are correct), its advantage is set to zero; consequently, by Equation (1), policy-gradient updates vanish. To mitigate this, dynamic sampling (Yu et al., 2025) first over-samples a larger candidate set Bˆ and then constructs the training batch by filtering out uninformative prompts:

B = q ∈ Bˆ : 0 < mean {v(q,rj)}Gj=1 < 1 . (5) Hereafter, we call a prompt solve all if its sampled responses {rj}Gj=1 are all correct, and solve none if they are all incorrect. Following (Qu et al., 2025; Le et al., 2025),we use “uninformative,” “zero-variance,” and “zero-advantage” prompts as synonyms for solve all and solve none prompts.

- 3. Methodology & Meta-Experiments

The operator Compose consists of three steps (see Figure 1(a) for one concrete example):

❶ Modify q1 with gt1. Extract a numeric value from gt1, denoted by v1. We then introduce a natural-language definition d1 that names this value in terms of (q1,gt1), and form q¯1 = q1 ⊕ d1. For instance, if q1 is “What is the sum of the value(s) of n for which |2n − 7| = 3?” and gt1 = 7, we set v1 = 7 and add a definition such as: “Let X be the sum of the value(s) of n satisfying |2n − 7| = 3.”

❷ Modify q2. Extract a numeric value from q2 and replace it with a new variable v2, yielding q¯2 = q2(v2). For example, if q2 is “Simplify 2((5p+1)−2p·4)+(4−1÷3)(6p−9) to the form ap−b, where a and b are positive,” we may choose the constant 1 as v2 and replace it with variable name Y , obtaining “Simplify 2((5p+Y )−2p·4)+(4−1÷3)(6p−9) to the form ap − b, where a and b are positive.”

❸ Connect q1 and q2. Compute v1 − v2 and express the resulting relation between the two variables as a naturallanguage statement r. Continuing the example above, with v1 = 7 and v2 = 1, we have v1 − v2 = 6, so we can add a constraint r such as: “Y is 6 less than X.” The composed prompt is then q1:2 = q¯1 ⊕ r ⊕ q¯2. By construction, the ground-truth answer of the composed prompt is gt1:2 = gt2. This composition is asymmetric to the order of q1 and q2, and solving q1:2 requires solving q1 first and then q2.

###### 3.1. SPC: Sequential Prompt Composition

Yuan et al. (2025) studies the role of composition in RL using a synthetic string-transformation setting, and Xiao & Zhao (2025) evaluates LLM performance under the composition of two math problems. We extend this line of work by investigating how composing training prompts affects RL training. In this section, we describe Sequential Prompt Composition (SPC): we first define how to compose two prompts, and then generalize to composing K prompts. The whole composition process is illustrated in Figure 1.

Composing K Prompts. More generally, we can compose K prompts into one prompt. Given q1,...,qK with groundtruth answers gt1,...,gtK, Sequential Prompt Composition (SPC) applies Compose recursively for K − 1 steps:

SPC(q1, . . . , qK; gt1, . . . , gtK) = Compose(q1, q2:K; gt1, gt2:K), where (q2:K, gt2:K) = SPC(q2, . . . , qK; gt2, . . . , gtK).

Composing Two Prompts. Given two prompts q1 and q2 with ground-truth answers gt1 and gt2, we define a composition operator Compose that maps (q1,q2;gt1,gt2) to a composed prompt q1:2 with ground-truth answer gt1:2:

Finally, we will get the composed prompt q1:K and its answer gt1:K. We term K as the Compositional Depth. Intuitively, solving q1:K requires the model having the ability to

q1:2, gt1:2 = Compose(q1, q2; gt1, gt2). (6)

solve all {qk}Kk=1. The process of composing 2 prompts can be viewed as a special case of SPC with K = 2. Therefore,

we do not distinguish between these two hereinafter.

###### 3.2. Meta-Experiments & Observation

As collecting new high-quality, verifiable training prompts can be costly (He et al., 2025b; Zeng et al., 2025), a growing body of work has focused on better leveraging uninformative prompts in RLVR (Li et al., 2025a;c). However, existing methods primarily target solve none prompts. In this section, we conduct some meta-experiments and have the following key observations: ❶ Beyond solve none, the increasing prevalence of solve all prompts is another major impediment to effective RL training. ❷ SPC can make easy prompts harder and reduce the ratio of solve all prompts.

Dilemma of Effective Training Prompts. As the policy model becomes stronger during RLVR, the proportion of solve all prompts observed during rollouts increases.

- Figure 2 (Left) plots the solve all rate across RL training steps for Qwen3-4B-Base on the MATH training set (Hendrycks et al., 2021). The solve all ratio rises rapidly from near zero to over 50% within the first 50 steps and then stabilizes around 75%. Although dynamic sampling is enabled to remove zero-variance prompts, the actual effective size of the whole training set at later stages is reduced to roughly 3K prompts (12,000 × (1 − 0.75)). In contrast, the solve none ratio remains low (about 5%) at 250 steps. These results motivate methods that can deal with solve all prompts, in addition to solve none prompts.

SPC can nudge the last bits out of existing prompts. Intuitively, SPC makes easy prompts harder. We empirically validate this on a subset of the MATH500 test set using OpenMath-Reasoning-1.5B (Moshkov et al., 2025) and JustRL-1.5B (He et al., 2025a); additional details are provided in Section A.2. As shown in Figure 2 (Right), switching to compositional prompts reduces avg@8 by 19.7% for OpenMath-Reasoning-1.5B and by 15.4% for JustRL-1.5B. The solve all rate also drops substantially: from 81.5% to 41.4% for OpenMath-Reasoning-1.5B, and from 88.5% to 60.0% for JustRL-1.5B. These results suggest that SPC can effectively reduce solve all prompts, potentially turn part of the original uninformative prompts useful again. Additionally, even SPC with K = 2 can almost double the training set size in principle (from |D| to |D| · (|D| − 1)).

Note that JustRL-1.5B is obtained by RL training OpenMath-Reasoning-1.5B. Another interesting observation is that JustRL-1.5B improves performance both on the MATH500 subset (by 2.3%) and the compositional test set (by 6.6%). This suggests that RL training

on normal prompts2 can also improve performance on compositional prompts. This raises a natural question: Does RL training on compositional prompts benefit performance on normal reasoning tasks?

###### 3.3. Composition-RL: RL with Compositional Data

This section introduces Composition-RL, a simple yet effective framework that leverages compositional data for RLVR

training. Given the original training set D = {(qi,gti)}|D|i=1, we can construct a level-K compositional prompt set via

the LLM-driven SPC procedure:

DCK = (q, gt) : q, gt = SPC(q1, . . . , qK; gt1, . . . , gtK), (qk, gtk) ∈ D, k = 1, . . . , K, qi ̸= qj ∀i ̸= j .

Since the size of DCK

can be extremely large, we instead use a smaller surrogate set:

DˆCK = (q, gt) : q, gt = SPC(q1, . . . , qK; gt1, . . . , gtK), (qk, gtk) ∈ Dk, k = 1, . . . , K, qi ̸= qj ∀i ̸= j , (7)

where each Dk is a small random subset of D and serve as the candidate set for qk, i.e., qk ∼ Dk. In practice, we set |Dk| = 20 for k = 1,...,K − 1, and DK = D. Whenever we draw a new qK ∈ D, we independently resample D1,...,DK−1; these candidate pools are not fixed.

Composition-RL then optimizes the RLVR objective over compositional prompts: maxθ Eq∼Dˆ

JRLVR(θ) . We use the same GRPO objective JGRPO(θ), advantage estimator, and importance ratio as in Equations (2) to (4), except that prompts q are sampled from the compositional dataset. Unless otherwise specified, we use K = 2 in all remaining experiments, and abbreviate DC2

CK

as DC.

#### 4. Experiments 4.1. Experimental Setting

In this section, we briefly summarize our experimental setup, including training procedures, baselines, and evaluation. Additional details are provided in Section B.

Training Details We conduct RL training using the VeRL codebase (Sheng et al., 2024). Unless otherwise specified, we use a unified set of hyperparameters (batch size 256, learning rate 1 × 10−6, and no warm-up) and fixed rollout settings (temperature 1, top p 1, top k −1, 8 rollouts per problem, and a maximum output length of 16K tokens). We train Qwen3-4B-Base, Qwen3-8B-Base, Qwen3-14B-Base, and Qwen3-30B-A3B-Base on the MATH training set (Hendrycks et al., 2021). For the cross-topic experiments in Section 4.4, we use the physics

2As compositional prompts differ substantially in structure from the original ones, we refer to prompts from all existing datasets as normal prompts for simplicity.

- Table 1. Results of Composition-RL across different benchmarks. “Avg@k” denotes the average accuracy (%) over k random generations (i.e., pass@1). The rows “Depth 1 + 2” and “+ Depth 3” are the results of curriculum Composition-RL in Section 4.3. ∗ indicates that the corresponding results are obtained using the same training set size as MATH12K. Specifically, “✔” denotes Composition-RL on MATH-Composition-199K, while “✔∗” denotes Composition-RL trained on a 12K subset of MATH-Composition-199K.

Mathematics (In-Domain) Multi-Task (Out-Of-Domain) Overall AIME 24 AIME 25 Beyond AIME IMOBench Overall GPQA MMLU-Pro Overall Overall

Composition

Avg@32 Avg@32 Avg@8 Avg@4 Avg. Avg@8 Avg@1 Avg. Avg.

###### Qwen3-4B-Base

✘ 23.3 19.5 9.0 14.4 16.6 43.7 58.6 51.2 28.1 MetaMath∗ 12.7 9.7 3.9 8.1 8.6 43.9 58.5 51.2 22.8

SAND-Math∗ 25.6 20.5 8.9 15.4 17.6 45.4 57.9 51.7 29.0

- ✔∗ 27.3 ↑4.0 23.1 ↑3.6 9.0 ↑0.0 15.9 ↑1.5 18.8 ↑2.0 46.2 ↑2.5 60.1 ↑1.5 53.2 ↑2.0 30.3 ↑2.2

- ✔ 30.5 ↑7.2 23.3 ↑3.8 12.6 ↑3.6 14.3 ↓0.1 20.2 ↑3.6 46.3 ↑2.6 61.4 ↑2.8 53.9 ↑2.7 31.4 ↑3.3

Depth 1 + 2 33.0 ↑9.7 27.8 ↑8.3 13.1 ↑4.1 20.1 ↑5.7 23.5 ↑6.9 48.3 ↑4.6 63.8 ↑5.2 56.1 ↑4.9 34.4 ↑6.3 + Depth 3 37.9 ↑14.6 29.7 ↑10.2 14.6 ↑5.6 22.9 ↑8.5 26.3 ↑9.7 48.5 ↑4.8 64.5 ↑5.9 56.5 ↑5.3 36.4 ↑8.3

###### Qwen3-8B-Base

✘ 26.1 20.4 13.7 16.2 19.1 48.2 62.6 55.4 31.2

- ✔∗ 29.2 ↑3.1 25.2 ↑4.8 13.3 ↓0.4 18.8 ↑2.6 21.6 ↑2.5 49.1 ↑0.9 65.9 ↑3.3 57.5 ↑2.1 33.6 ↑2.4

- ✔ 36.9 ↑10.8 26.5 ↑6.1 13.9 ↑0.2 18.4 ↑2.2 23.9 ↑4.8 48.9 ↑0.7 64.5 ↑1.9 56.7 ↑1.3 34.9 ↑3.7 Qwen3-14B-Base

✘ 34.4 30.2 17.0 21.3 25.7 55.0 67.2 61.1 37.5

- ✔∗ 43.3 ↑8.9 36.0 ↑5.8 18.6 ↑1.6 23.5 ↑2.2 30.4 ↑4.7 54.7 ↓0.3 69.3 ↑2.1 62.0 ↑0.9 40.9 ↑3.4

- ✔ 44.5 ↑10.1 36.9 ↑6.7 19.7 ↑2.7 25.9 ↑4.6 31.8 ↑6.1 54.2 ↓0.8 69.3 ↑2.1 61.8 ↑0.7 41.8 ↑4.3 Qwen3-30B-A3B-Base

✘ 25.2 16.2 7.5 13.2 15.5 50.7 62.6 56.7 29.2

- ✔∗ 47.7 ↑22.5 29.8 ↑13.6 20.1 ↑12.6 22.9 ↑9.7 30.1 ↑14.6 58.2 ↑7.5 65.6 ↑3.0 61.9 ↑5.2 40.7 ↑11.5

- ✔ 46.4 ↑21.4 30.3 ↑14.1 19.5 ↑12.0 22.8 ↑9.6 29.8 ↑14.3 54.6 ↑3.9 64.6 ↑2.0 59.6 ↑2.9 39.7 ↑10.5

subset of MegaScience (Fan et al., 2025). For the verifier, we choose Math-Verify, a rule-based verifier. For a fair comparison, we enable dynamic sampling to filter uninformative prompts, ensuring that the effective batch size at each step remains constant across experiments.

Baselines. In Section 4.2 (see Table 1), we compare Composition-RL with standard RLVR on MATH12K under the same number of gradient updates. For Composition-RL, we construct approximately 199K compositional prompts, which we denote as MATH-Composition-199K. In Section 4.3, we additionally report several RL-zero methods as reference points for our curriculum-based Composition-RL, including Beyond-80/20 (Wang et al., 2025), AlphaRL (Cai et al., 2025), and RL-ZVP (Le et al., 2025). For the 4B setting, we also adapt SFT data augmentation methods as baselines, including MetaMath (Yu et al., 2023b; Lu et al., 2024) and SAND-Math (Manem et al., 2025). For fairness, we additionally report results of Composition-RL under a controlled training set size (denoted by “*” in Table 1). For the cross-domain experiments in Section 4.4, we compare Composition-RL with two baselines: Mix Training (RL on a mixed dataset comprising MATH12K and the MegaScience Physics subset) and Math-then-Physics (continued RL on Physics starting from a MATH12K-trained checkpoint). Additional details are provided in Section B.2.

Evaluation Details. Our evaluation benchmarks include both in-domain (ID) math reasoning tasks, AIME24/25, BeyondAIME (ByteDance-Seed, 2025), and IMOBench (Luong et al., 2025), and out-of-domain (OOD) multi-task reasoning benchmarks, GPQA-Diamond (Rein et al., 2024) and MMLU-Pro (Wang et al., 2024). Following Guo et al. (2025), we sample multiple responses per problem (from 1 to 32, depending on the benchmark size) and report pass@1 accuracy. All evaluation scripts are adapted from the DeepscaleR codebase (Luo et al., 2025). Following Yang et al. (2025a); Xu et al. (2025a), we set the temperature to 0.6, top p to 0.95, top k to 20, and the maximum output length to 32K tokens. See also Section B.3 for details.

###### 4.2. Compositional Prompts Are Beneficial to RLVR

To evaluate Composition-RL, we report main results in Table 1 with additional metrics provided in Section E. From Table 1, we have the following observations:

❶ RL on compositional prompts consistently outperforms RL on the original prompts on both in-domain math and out-of-domain (OOD) general benchmarks. Across all model sizes, Composition-RL improves the overall mathematics performance by +3.6%, +4.8%, +6.1%, and +14.3% for Qwen3-4B/8B/14B/30B-A3B, respectively. Notably, gains are observed on challenging math

benchmarks, including AIME24 (up to +21.4%), AIME25 (up to +14.1%), Beyond AIME (up to +12.0%), and IMOBench (up to +9.6%). Moreover, Composition-RL also improves OOD performance, increasing the multi-task overall by +2.7%, +1.3%, +0.7%, and +2.9%, leading to overall average gains of +3.3%, +3.7%, +4.3%, and +10.5% across the four base models. These significant gains demonstrate the effectiveness of Composition-RL and highlight the value of MATH-Composition-199K.

❷ The benefits of Composition-RL scale with model size, with larger models exhibiting substantially larger improvements, especially in mathematics. Overall gains increase from +3.3% (4B) and +3.7% (8B) to +4.3% (14B), and peak at +10.5% for Qwen3-30B-A3B. The scaling effect is most pronounced on in-domain mathematics: improvements rise from +3.6%/+4.8%/+6.1% to +14.3% as model size increases from 4B to 30B, whereas OOD multi-task gains are smaller but remain consistently positive. Notably, the MoE 30B-A3B model underperforms the 14B dense model, consistent with the fact that MoE activates only a subset of experts per token and can be more sensitive to routing and optimization under a fixed training budget; nevertheless, Composition-RL still yields large gains on this model. Overall, these results highlight the strong potential of Composition-RL, particularly for larger models.

❸ Composition-RL still outperforms the baselines under the same training set size. Compared with RL training on the original dataset, Composition-RL with 12K compositional prompts yields gains ranging from 2.2% to 11.5% for models from 4B to 30B, although the margin is slightly smaller than that on MATH-Composition-199K. In addition, Composition-RL outperforms MetaMath by 7.5% and SAND-MATH by 1.3%. This indicates that Composition-RL is substantially more effective than SFTbased data augmentation methods, even though its seed questions are drawn solely from the MATH training set. Overall, Composition-RL achieves the best performance under a controlled dataset size, and its ability to generate substantially more training prompts from existing humanannotated data is another key advantage.

###### 4.3. Curriculum RL to Higher Compositional Depth

We have shown that directly training on MATH-Composition-199K outperforms training on the original MATH12K. As discussed in Section 3.2, during RL on MATH12K, the solve all ratio gradually rises to a high level and performance begins to saturate; SPC can alleviate this issue. A natural extension is to adopt a curriculum that progressively increases the composition depth and continues RL training. Concretely, we first train on MATH12K; once performance saturates, we switch to Composition-RL with Depth 2. This transition causes

the solve all ratio to drop sharply and enables further performance gains. We experiment with this curriculum version of Composition-RL from Depth 1 to Depth 3. Additional details are provided in Section B.2. As shown in Table 1 and Figure 1, we have the following observations:

❶ Curriculum Composition-RL can make full use of the original prompts, producing progressively stronger LRMs as the composition depth increases. Continuing RL with Depth 2 data after Depth 1 (i.e., the original MATH12K) yields substantial gains over the Depth 1 checkpoint, improving by +9.7% on AIME24 and +5.9% on MMLU-Pro. Moreover, the Depth 1→Depth 2 curriculum even outperforms training directly on MATH-Composition-199K, delivering a further +3.0% improvement on the overall average. Adding an additional Depth 3 stage continues to improve both indomain tasks and OOD question answering, with a further +2.0% overall gain. Figure 1 presents the validation performance curves throughout the curriculum training process. In summary, these results imply that Composition-RL effectively converts limited prompts (with high solve all rates) into more useful samples.

❷ Curriculum Composition-RL on a 4B model surpasses several 8B baselines, even under unfavorable settings. As shown in Figure 1, our final Composition-RL-4B model achieves 37.9% on AIME24, outperforming Beyond80/20-8B (Wang et al., 2025) (34.6%), Alpha-RL-8B (Cai et al., 2025) (28.3%), and RL-ZVP-8B (Le et al., 2025) (24.6%). Notably, Composition-RL uses only MATH12K and Qwen3-4B-Base, whereas these baselines train on DAPO-MATH-17K and Qwen3-8B-Base. Additional details are provided in Section B.2. Even in this unfavorable setting, Composition-RL achieves stronger performance, underscoring the importance of fully leveraging existing training prompts via composition.

###### 4.4. Potential For General Domains

Previously, Composition-RL considered only problems in the mathematical domain. In this section, we explore whether it can compose problems across domains. Specifically, we sample q1 from the physics subset and q2 from MATH12K, yielding a cross-domain compositional dataset, Physics-MATH-Composition-141K. We compare against the Mix Training and Physics-then-Math baselines described in Section 4.1, with details in Section B.2. As shown in Table 2, we make the following observations:

❶ Adding physics prompts for RL training improves multi-task reasoning performance. Both the Mix Training and Math-then-Physics baselines improve GPQA and MMLU-Pro performance relative to training on MATH12K alone. On average, Mix Training increases the multi-task average by 0.8%, and Math-then-Physics yields a larger gain

- Table 2. Results of cross-topic experiments across multiple benchmarks. “Avg@k” denotes the average accuracy (%) over k random generations (i.e., pass@1). “MATH12K + Physics” corresponds to the Mix Training baseline, and “Physics after MATH12K” corresponds to the Math-then-Physics baseline. Best results in each column are in bold.

Dataset

Mathematics Multi-Task Overall AIME 24 AIME 25 Beyond AIME IMOBench Overall GPQA MMLU-Pro Overall Overall

Avg@32 Avg@32 Avg@8 Avg@4 Avg. Avg@8 Avg@1 Avg. Avg. MATH12K 23.3 19.5 9.0 14.4 16.6 43.7 58.6 51.2 28.1

MATH12K + Physics 19.7 16.5 8.3 12.0 14.1 44.4 59.6 52.0 26.8 Physics after MATH12K 25.3 22.3 8.6 14.4 17.7 45.2 61.4 53.3 29.5

Physics-MATH-Composition-141K 32.4 25.5 10.6 17.8 21.6 46.6 62.7 54.7 32.6

- Table 3. Results of the ablation study. “Avg@k” denotes the average accuracy (%) over k random generations (i.e., pass@1). D1 specifies the strategy for constructing the candidate set for q1: “RAND” randomly samples 20 prompts, whereas “FULL” selects from the entire original prompt set D. “Baseline” denotes RL training on the original prompts D. “Direct Concat” denotes RL training on prompts that are directly concatenated from two prompts without composition. For results on the fully solvable subset, n is the number of rollouts.

Mathematics (In-Domain) Multi-Task (Out-Of-Domain) Overall AIME 24 AIME 25 Beyond AIME IMOBench Overall GPQA MMLU-Pro Overall Overall

D1 D2

Avg@32 Avg@32 Avg@8 Avg@4 Avg. Avg@8 Avg@1 Avg. Avg.

Baseline 23.3 19.5 9.0 14.4 16.6 43.7 58.6 51.2 28.1 RAND RAND 22.6 19.6 8.2 13.8 16.1 43.6 60.0 51.8 28.0 FULL RAND 24.5 23.4 8.7 14.1 17.7 44.8 59.9 52.4 29.2 RAND FULL 30.5 23.3 12.6 14.3 20.2 46.3 61.4 53.9 31.4

Direct Concat 12K 18.4 14.4 6.4 10.4 12.4 43.1 55.9 49.5 24.8 Composition-RL 12K 27.3 23.1 9.0 15.9 18.8 46.2 60.1 53.2 30.3

###### Results on Fully Solvable Subset of MATH12K

initial model 13.1 10.1 5.2 8.7 9.3 42 53.9 48.0 22.2 Baseline (n=32) 13.2 11.6 5.1 9.9 10.0 42.1 53.7 47.9 22.6

Composition-RL (n=8) 17.1 15.6 6.1 10.5 12.3 42.5 57.6 50.1 24.9

of 2.1%. Moreover, Math-then-Physics can further increase performance on math reasoning tasks, whereas Mix Training slightly degrades the math reasoning ability. These results suggest that, while incorporating physics data benefits multi-task performance, sequential training (math followed by physics) is more effective than mixed training across topics. As shown in Figure 1, adding physics prompts (via Mix Training or Math-then-Physics) consistently improves generalization to law, engineering, and chemistry compared to training solely on math (Math-Only). Interestingly, training on MATH-Composition-199K (Math-Composition) also yields generalization beyond the math domain.

❷ Composing physics and math problems is more effective than naively combining physics and math prompts. RL training on Physics-MATH-Composition-141K outperforms all baselines by a large margin. Specifically, our method achieves a +1.3% gain over Math-then-Physics and a +4.3% gain over training solely on MATH12K on MMLU-Pro. On AIME24, it improves by +7.1% over Math-then-Physics and by +9.1% over training solely on MATH12K. As shown in Figure 1, RL training on Physics-MATH-Composition-141K (Physics-MathComposition) consistently delivers the best results on both in-domain subjects (math and physics) and OOD subjects (law, engineering, and chemistry). These results highlight

the great potential of Composition-RL for RL of multiple topics: training on composed prompts that require multidomain knowledge will definitely induce broad improvements across the corresponding topics, and Composition-RL can generate such prompts using existing ones.

#### 5. Analysis

###### 5.1. Ablation Study of Candidate Sets Dk

As described in Section 3.3, each candidate set (except DK) is constructed by sampling from the full prompt pool D; specifically, qk ∈ Dk for k = 1,...,K−1. For K = 2, q1 is drawn from a 20-prompt subset, whereas q2 is drawn from the full set D (D1 is different for different q2). We further evaluate the following variants for constructing the surrogate compositional set: A) Both D1 and D2 are small randomly sampled subsets (|D1| = |D2| = 500). B) D1 is the full set D, while D2 is a small randomly sampled subset (|D2| = 12,000, |D2| = 20) (D2 is different for different q1). To ensure a fair comparison of these variants, we keep the total amount of compositional data approximately constant and train for the same number of gradient updates under the unified training configuration in Section 4.1. Additional construction details are provided in Section B.4.

Results are reported in Table 3. Our Composition-RL config-

uration (sampling D1 as a random subset and using the full set for D2) achieves the best performance, improving overall accuracy by +3.4% over variant A and by +2.2% over variant B. variant A performs comparably to the baseline (RL on the original D), while both underperform relative to variant B and our Composition-RL setting. This is not surprising because |D1|+|D2| = 1,000 is substantially smaller than |D| = 12,000, implying reduced diversity in the seed prompts used to construct the compositional set. Notably, despite using only 1K seed prompts, variant A matches the baseline trained on 12K prompts, highlighting the potential of Composition-RL in limited-data regimes.

Importantly, Composition-RL also outperforms variant B by a clear margin; for instance, on AIME24, CompositionRL achieves a +6.0% accuracy gain. This suggests that increasing the diversity of D2 is beneficial. We hypothesize that this effect arises because the composed prompt q1:2 shares the same ground-truth answer gt1:2 as q2 and the current training paradigm verifies only the final answer of model responses. Under variant B, the model is repeatedly trained and verified on only answers from a proper subset (D2), potentially limiting the coverage of training signals. In contrast, our Composition-RL configuration exposes the model to verification over the full set D2 = D, yielding a substantially more diverse set of answers to be verified.

###### 5.2. The Necessity of Composition

To validate the necessity of conditioning q2 on gt1, we conduct an ablation study in which q1 and q2 are directly concatenated using the same question pairs as CompositionRL. As reported in Table 3, this direct-concatenation variant performs even worse than RL training on the original dataset. Moreover, Composition-RL outperforms the directconcatenation variant by 5.5% in overall accuracy. We believe this degradation stems from a distribution shift, as direct concatenation essentially combines two problems into a single prompt. In contrast, our SPC guarantees that the compositional prompt functions as “one prompt”. These results highlight the necessity of composition.

###### 5.3. Effectiveness on Fully Solvable Prompts

Previously, we showed that Composition-RL performs better on the full MATH12K dataset. To proceed further, we start from an intermediate checkpoint of the 4B model and retain only the solve-all prompts from MATH12K, resulting in 7.2K prompts. We then compare Composition-RL on this fully solvable subset against a DAPO-style baseline, in which additional trajectories are adaptively sampled for easier problems until an incorrect answer is obtained, with a maximum of 32 rollouts. As shown in Table 3, this DAPOstyle baseline improves the initial model by only a small margin (+0.4% overall). In comparison, Composition-RL

yields a +2.7% improvement even on this fully solvable subset, even with 8 rollouts per prompt. These results suggest that Composition-RL can indeed serve as a “from-scratch” approach, even when all prompts are relatively simple.

###### 5.4. Why Composition-RL Works

In this section, we further investigate why Composition-RL works. We analyze it from two perspectives:

Compositional Generalization. Compositional data may incentivize the acquisition of new skills. Yuan et al. (2025) show that, in a controlled synthetic setting, training on compositional data can elicit new reasoning skills. Analogously, if we view an original problem as requiring a stack of skills, composing prompts can create training instances that demand skill recombination. As shown in Figure 3 (left), Composition-RL substantially improves performance on Depth-2 compositional test prompts relative to training on Depth-1 data, even though Depth-2 is more challenging. This result supports compositional generalization: models trained with composed prompts transfer better to deeper compositions and, consequently, also improve on the standard test set, likely because the acquired skills are useful for solving more complex problems.

Implicit Process Supervision. The final-outcome reward for composed prompts also provides implicit signals for the solution process. As shown in Figure 1, to solve the composed prompt q1:2, LLMs must first obtain v1 and then use v1 to solve q¯2. We posit that this structured dependency nudges the model toward a correct intermediate step, at least “halfway” through the reasoning. As illustrated in Figure 3 (right), the steady improvement in recovering v1 provides evidence that composed prompts can serve as implicit process supervision, even when training relies only on the verification of the final answers.

For more analysis, please refer to Section D.

#### 6. Related Work

Longer Training with Finite Prompts. Amid the surge of interest in RLVR (Jaech et al., 2024; Guo et al., 2025), many studies investigate how to enable longer and more stable training under a fixed prompt set (Liu et al., 2025b; He et al., 2025a). One line of work improves training stability from an algorithmic perspective (Chen et al., 2025a; Yang et al., 2025b; Liu et al., 2025a). Another line aims to exploit limited training data better, including filtering uninformative prompts (Yu et al., 2025; Zheng et al., 2025; Qu et al., 2025), shaping advantages for zero-advantage prompts (Zhu et al., 2025a; Nan et al., 2025; Le et al., 2025), and allocating more samples to harder prompts (Yang et al., 2025c; Li et al., 2025c). Among these, hint-based problem augmentation (Chen et al., 2025b; Li et al., 2025a) is most

| |Original Composition<br><br>| | | | |
|---|---|---|---|---|---|
| |+0.5<br><br>+0.3 +0.7| | | | |
| |+0.5<br><br>+8.1<br><br>+5.6<br><br>+10.9| | | | |
| |+8.6| | | | |
| | | | | | |
| | | | | | |
| | | | | | |

100

80

90

60

Accuracy(%)

80

Avg@8

40

70

20

60

Acc of q1:2

Acc of v1

0

50

0 50 100 150 200 250

4b 8b 14b 30b

Training Steps

Model Size

- Figure 3. Left: avg@8 accuracy on a subset of MATH500 and the corresponding compositional test prompts across different model sizes. The darker color and the numbers denote the improvement of our Composition-RL over the RL training on the MATH12K baseline. Right: The fraction of prompts for which q1:2 is solved correctly, and the accuracy of recovering v1 at each training step.

closely related to our work: they use hints to transform originally hard prompts into easier ones. In contrast, we make easy prompts harder via compositional prompt generation.

Qwen-2.5-3B-Instruct; in contrast, we focus on mitigating the solve all bottleneck in RLVR via automated sequential prompt composition, including cross-domain composition and analyses of implicit process supervision.

Enlarging RLVR Training Prompts. The fuel of RLVR is its training prompts. A substantial body of work is devoted to collecting and curating high-quality data from diverse sources (Albalak et al., 2025; He et al., 2025b; Hu et al., 2025). Synthesizing data from existing datasets has also been extensively studied, both for evaluation (Shi et al., 2023; Xu et al., 2024; Xiao & Zhao, 2025) and for SFT (Yang et al., 2024; Yu et al., 2023b; Tong et al., 2024). More recently, several efforts have begun to synthesize prompts specifically for RL training (Xie et al., 2025; Li et al., 2025b; Stojanovski et al., 2025; Zeng et al., 2025). In contrast to synthetic logic-only problems or game-like environments, we target general reasoning tasks, achieving strong performance on mathematical reasoning and highlighting the potential for cross-domain integration.

#### 7. Conclusion & Discussion

In this paper, we study how to maximize the utility of existing prompts for RL training. Comprehensive experiments across various model sizes show that Composition-RL consistently outperforms RL on the original prompts. We also demonstrate the potential of composing prompts from different topics. Our analysis suggests that compositional prompts can provide implicit process supervision by encouraging correct intermediate steps. We will release our codes, compositional datasets, and trained models to support future RL research. Promising future directions include: ❶ Extending beyond MATH12K by composing more challenging math training set like Polaris-53K. ❷ Expanding composition to cover more domains. ❸ Adapting Composition-RL to on-policy distillation (Lu & Lab, 2025).

Compositional Generalization. Compositional generalization refers to a model’s ability to recombine learned skills to solve novel tasks. It has been a longstanding focus in natural language processing (Keysers et al., 2019; Hupkes et al., 2020; Lake & Baroni, 2018). Prior work often studies compositionality using controlled testbeds, such as Skill-Mix (Yu et al., 2023a) for language tasks, compositional math benchmarks (Sun et al., 2025), or algorithmic tasks (Dziri et al., 2023). Zhao et al. (2024) show that composing textual skills can benefit SFT. Yuan et al. (2025) suggest that compositionality is important for RL to acquire new skills. However, their results are restricted to synthesized string-manipulation tasks. In comparison, we extend composition to broader reasoning settings and demonstrate the effectiveness of composing RL training prompts. Notably, Motwani et al. (2025) study RL on GSM8K problems constructed via fixed-template transformations primarily using

#### Impact Statement

This paper presents Composition-RL, which aims to advance research on RLVR. We plan to release two compositional datasets, MATH-Composition-199K and Physics-MATH-Composition-141K, which we expect to be useful resources for future work on RL for LLMs. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

#### Acknowledgements

We thank Xisheng Xiao and Hanlin Zhao for sharing wellimplemented code for compositional data construction. We

also thank Mengyu Zhang for helpful discussions and for feedback that improved the wording in Section 3.3.

#### References

Albalak, A., Phung, D., Lile, N., Rafailov, R., Gandhi, K., Castricato, L., Singh, A., Blagden, C., Xiang, V., Mahan,

- D., et al. Big-math: A large-scale, high-quality math dataset for reinforcement learning in language models. arXiv preprint arXiv:2502.17387, 2025.

ByteDance-Seed. Beyondaime: Advancing math reasoning evaluation beyond high school olympiads. https://huggingface.co/datasets/ ByteDance-Seed/BeyondAIME, 2025. Hugging Face repository.

Cai, Y., Cao, D., Xu, X., Yao, Z., Huang, Y., Tan, Z., Zhang, B., Liu, G., and Fang, J. On predictability of reinforcement learning dynamics for large language models. arXiv preprint arXiv:2510.00553, 2025.

Chen, A., Li, A., Gong, B., Jiang, B., Fei, B., Yang, B., Shan, B., Yu, C., Wang, C., Zhu, C., et al. Minimaxm1: Scaling test-time compute efficiently with lightning attention. arXiv preprint arXiv:2506.13585, 2025a.

Chen, J. C.-Y., Peng, B. X., Choubey, P. K., Huang, K.-H., Zhang, J., Bansal, M., and Wu, C.-S. Nudging the boundaries of llm reasoning. arXiv preprint arXiv:2509.25666, 2025b.

Dziri, N., Lu, X., Sclar, M., Li, X. L., Jiang, L., Lin, B. Y., Welleck, S., West, P., Bhagavatula, C., Le Bras, R., et al. Faith and fate: Limits of transformers on compositionality. Advances in Neural Information Processing Systems, 36: 70293–70332, 2023.

Fan, R.-Z., Wang, Z., and Liu, P. Megascience: Pushing the frontiers of post-training datasets for science reasoning. arXiv preprint arXiv:2507.16812, 2025. URL https: //arxiv.org/abs/2507.16812.

Fu, W., Gao, J., Shen, X., Zhu, C., Mei, Z., He, C., Xu, S., Wei, G., Mei, J., Wang, J., Yang, T., Yuan, B., and Wu, Y. Areal: A large-scale asynchronous reinforcement learning system for language reasoning. ArXiv, 2025.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

He, B., Qu, Z., Liu, Z., Chen, Y., Zuo, Y., Qian, C., Zhang, K., Chen, W., Xiao, C., Cui, G., et al. Justrl: Scaling a 1.5 b llm with a simple rl recipe. arXiv preprint arXiv:2512.16649, 2025a.

He, Z., Liang, T., Xu, J., Liu, Q., Chen, X., Wang, Y., Song, L., Yu, D., Liang, Z., Wang, W., et al. Deepmath-103k: A large-scale, challenging, decontaminated, and verifiable mathematical dataset for advancing reasoning. arXiv preprint arXiv:2504.11456, 2025b.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Hu, J., Zhang, Y., Han, Q., Jiang, D., Zhang, X., and Shum, H.-Y. Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model. arXiv preprint arXiv:2503.24290, 2025.

Hupkes, D., Dankers, V., Mul, M., and Bruni, E. Compositionality decomposed: How do neural networks generalise? Journal of Artificial Intelligence Research, 67: 757–795, 2020.

Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky, A., Low, A., Helyar, A., Madry, A., Beutel, A., Carney, A., et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Keysers, D., Sch¨arli, N., Scales, N., Buisman, H., Furrer, D., Kashubin, S., Momchev, N., Sinopalnikov, D., Stafiniak, L., Tihon, T., et al. Measuring compositional generalization: A comprehensive method on realistic data. arXiv preprint arXiv:1912.09713, 2019.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Lake, B. and Baroni, M. Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. In International conference on machine learning, pp. 2873–2882. PMLR, 2018.

Le, T.-L. V., Jeon, M., Vu, K., Lai, V., and Yang, E. No prompt left behind: Exploiting zero-variance prompts in llm reinforcement learning via entropy-guided advantage shaping. arXiv preprint arXiv:2509.21880, 2025.

Li, J., Lin, H., Lu, H., Wen, K., Yang, Z., Gao, J., Wu, Y., and Zhang, J. Questa: Expanding reasoning capacity in llms via question augmentation. arXiv preprint arXiv:2507.13266, 2025a.

Li, P., Ye, J., Chen, Y., Ma, Y., Yu, Z., Chen, K., Cui, G., Li, H., Chen, J., Lyu, C., et al. Internbootcamp technical report: Boosting llm reasoning with verifiable task scaling. arXiv preprint arXiv:2508.08636, 2025b.

Li, Z., Chen, C., Yang, T., Ding, T., Sun, R., Zhang, G., Huang, W., and Luo, Z.-Q. Knapsack rl: Unlocking exploration of llms via optimizing budget allocation. arXiv preprint arXiv:2509.25849, 2025c.

Liu, A., Mei, A., Lin, B., Xue, B., Wang, B., Xu, B., Wu, B., Zhang, B., Lin, C., Dong, C., et al. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025a.

Liu, M., Diao, S., Lu, X., Hu, J., Dong, X., Choi, Y., Kautz, J., and Dong, Y. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models. arXiv preprint arXiv:2505.24864, 2025b.

Lu, K. and Lab, T. M. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml. 20251026. https://thinkingmachines.ai/blog/on-policydistillation.

Lu, Z., Zhou, A., Ren, H., Wang, K., Shi, W., Pan, J., Zhan, M., and Li, H. Mathgenie: Generating synthetic data with question back-translation for enhancing mathematical reasoning of llms. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2732–2747, 2024.

Luo, M., Tan, S., Wong, J., Shi, X., Tang, W. Y., Roongta, M., Cai, C., Luo, J., Li, L. E., Popa, R. A., and Stoica, I. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl, 2025. Notion Blog.

Luong, M.-T., Hwang, D., Nguyen, H. H., Ghiasi, G., Chervonyi, Y., Seo, I., Kim, J., Bingham, G., Lee, J., Mishra, S., et al. Towards robust mathematical reasoning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 35406–35430, 2025.

Manem, C., Brahma, P. P., Mishra, P., Liu, Z., and Barsoum,

- E. Sand-math: Using llms to generate novel, difficult and useful mathematics questions and answers. arXiv preprint arXiv:2507.20527, 2025.

Meng, F., Du, L., Liu, Z., Zhou, Z., Lu, Q., Fu, D., Han, T., Shi, B., Wang, W., He, J., et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025.

Moshkov, I., Hanley, D., Sorokin, I., Toshniwal, S., Henkel, C., Schifferer, B., Du, W., and Gitman, I. Aimo-2 winning solution: Building state-of-the-art mathematical reasoning models with openmathreasoning dataset. arXiv preprint arXiv:2504.16891, 2025.

Motwani, S. R., Ivanova, A., Cai, Z., Torr, P., Islam, R., Shah, S., de Witt, C. S., and London, C. h1: Bootstrapping llms to reason over longer horizons via reinforcement learning. ArXiv, abs/2510.07312, 2025.

Nan, G., Chen, S., Huang, J., Lu, M., Wang, D., Xie, C., Xiong, W., Zeng, X., Zhou, Q., Li, Y., et al. Ngrpo: Negative-enhanced group relative policy optimization. arXiv preprint arXiv:2509.18851, 2025.

Qi, P., Liu, Z., Zhou, X., Pang, T., Du, C., Lee, W. S., and Lin, M. Defeating the training-inference mismatch via fp16. arXiv preprint arXiv:2510.26788, 2025.

Qu, Y., Wang, Q., Mao, Y., Hu, V. T., Ommer, B., and Ji, X. Can prompt difficulty be online predicted for accelerating rl finetuning of reasoning models? arXiv preprint arXiv:2507.04632, 2025.

Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., and Bowman, S. R. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.

Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R., Peng, Y., Lin, H., and Wu, C. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.

Shi, F., Chen, X., Misra, K., Scales, N., Dohan, D., Chi, E. H., Sch¨arli, N., and Zhou, D. Large language models can be easily distracted by irrelevant context. In International Conference on Machine Learning, pp. 31210– 31227. PMLR, 2023.

Stojanovski, Z., Stanley, O., Sharratt, J., Jones, R., Adefioye, A., Kaddour, J., and K¨opf, A. Reasoning gym: Reasoning environments for reinforcement learning with verifiable rewards. arXiv preprint arXiv:2505.24760, 2025.

Sun, Y., Hu, S., Zhou, G., Zheng, K., Hajishirzi, H., Dziri, N., and Song, D. Omega: Can llms reason outside the box in math? evaluating exploratory, compositional, and transformative generalization. arXiv preprint arXiv:2506.18880, 2025.

Sutton, R. S., McAllester, D., Singh, S., and Mansour, Y. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12, 1999.

Team, Q. Qwen2.5: A party of foundation models, September 2024. URL https://qwenlm.github.io/ blog/qwen2.5/.

Tong, Y., Zhang, X., Wang, R., Wu, R., and He, J. Dartmath: Difficulty-aware rejection tuning for mathematical problem-solving. ArXiv preprint, abs/2407.13690, 2024. URL https://arxiv.org/abs/2407.13690.

Wang, S., Yu, L., Gao, C., Zheng, C., Liu, S., Lu, R., Dang, K., Chen, X., Yang, J., Zhang, Z., et al. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.01939, 2025.

Xu, X., Xu, Q., Xiao, T., Chen, T., Yan, Y., Zhang, J., Diao, S., Yang, C., and Wang, Y. Ugphysics: A comprehensive benchmark for undergraduate physics reasoning with large language models. arXiv preprint arXiv:2502.00334, 2025b.

Yang, A., Zhang, B., Hui, B., Gao, B., Yu, B., Li, C., Liu, D., Tu, J., Zhou, J., Lin, J., et al. Qwen2. 5-math technical report: Toward mathematical expert model via selfimprovement. arXiv preprint arXiv:2409.12122, 2024.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.

Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290, 2024.

Yang, K., Xu, X., Chen, Y., Liu, W., Lyu, J., Lin, Z., Ye, D., and Yang, S. Entropic: Towards stable long-term training of llms via entropy stabilization with proportionalintegral control. arXiv preprint arXiv:2511.15248, 2025b.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E. H., Le, Q. V., and Zhou, D. Chain-ofthought prompting elicits reasoning in large language models. In Koyejo, S., Mohamed, S., Agarwal, A., Belgrave, D., Cho, K., and Oh, A. (eds.), Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.

Yang, Z., Guo, Z., Huang, Y., Wang, Y., Xie, D., Wang, Y., Liang, X., and Tang, J. Depth-breadth synergy in rlvr: Unlocking llm reasoning gains with adaptive exploration. arXiv preprint arXiv:2508.13755, 2025c.

Yao, F., Liu, L., Zhang, D., Dong, C., Shang, J., and Gao, J. Your efficient rl framework secretly brings you off-policy rl training, August 2025. URL https://fengyao. notion.site/off-policy-rl.

nips.cc/paper_files/paper/2022/hash/ 9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference. html.

Yu, D., Kaur, S., Gupta, A., Brown-Cohen, J., Goyal, A., and Arora, S. Skill-mix: A flexible and expandable family of evaluations for ai models. arXiv preprint arXiv:2310.17567, 2023a.

Xiao, T., Xu, X., Huang, Z., Gao, H., Liu, Q., Liu, Q., and Chen, E. Advancing multimodal reasoning capabilities of multimodal large language models via visual perception reward. arXiv preprint arXiv:2506.07218, 2025.

Yu, L., Jiang, W., Shi, H., Yu, J., Liu, Z., Zhang, Y., Kwok, J. T., Li, Z., Weller, A., and Liu, W. Metamath: Bootstrap your own mathematical questions for large language models. ArXiv preprint, abs/2309.12284, 2023b. URL https://arxiv.org/abs/2309.12284.

Xiao, X. and Zhao, H. From a and b to a+ b: Can large language models solve compositional math problems? In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 13068– 13089, 2025.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Xie, T., Gao, Z., Ren, Q., Luo, H., Hong, Y., Dai, B., Zhou, J., Qiu, K., Wu, Z., and Luo, C. Logic-rl: Unleashing llm reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2502.14768, 2025.

Yuan, L., Chen, W., Zhang, Y., Cui, G., Wang, H., You, Z., Ding, N., Liu, Z., Sun, M., and Peng, H. From f(x) and g(x) to f(g(x)): Llms learn new skills in rl by composing old ones. arXiv preprint arXiv:2509.25123, 2025.

Xu, X., Xiao, T., Chao, Z., Huang, Z., Yang, C., and Wang, Y. Can llms solve longer math word problems better? ArXiv preprint, abs/2405.14804, 2024. URL https: //arxiv.org/abs/2405.14804.

Zeng, Z., Ivison, H., Wang, Y., Yuan, L., Li, S. S., Ye, Z., Li, S., He, J., Zhou, R., Chen, T., et al. Rlve: Scaling up reinforcement learning for language models with adaptive verifiable environments. arXiv preprint arXiv:2511.07317, 2025.

Xu, X., AI, C., Yang, K., Chen, T., Wang, Y., Yang, S., and Yang, C. Thinking-free policy initialization makes distilled reasoning models more effective and efficient reasoners. arXiv preprint arXiv:2509.26226, 2025a.

Zhao, H., Kaur, S., Yu, D., Goyal, A., and Arora, S. Can models learn skill composition from examples? Advances in Neural Information Processing Systems, 37:102393– 102427, 2024.

Zheng, H., Zhou, Y., Bartoldson, B. R., Kailkhura, B., Lai, F., Zhao, J., and Chen, B. Act only when it pays: Efficient reinforcement learning for llm reasoning via selective rollouts. arXiv preprint arXiv:2506.02177, 2025.

Zhu, X., Xia, M., Wei, Z., Chen, W.-L., Chen, D., and Meng, Y. The surprising effectiveness of negative reinforcement in llm reasoning. arXiv preprint arXiv:2506.01347, 2025a.

Zhu, Z., Xie, C., Lv, X., and slime Contributors. slime: An llm post-training framework for rl scaling. https:// github.com/THUDM/slime, 2025b. GitHub repository. Corresponding author: Xin Lv.

#### Limitation

Our method currently requires the first question’s answer to contain at least one numeric value; extending Composition-RL to broader domains (e.g., pure textural reasoning or other unverifiable domains) is a valuable direction for future work. Additionally, although our verification procedure filters out many low-quality compositions, some generated prompts may still be invalid. Finally, there might be potential misuse of our released data or models. We have discussed dataset construction and intended use to mitigate such risks.

#### A. Details of Meta-Experiments

- A.1. The solve all Ratio The training setup in Figure 2 (Left) follows the main experimental protocol (see Section B.1). We define the solve all ratio as the fraction of solve all prompts among all over-sampled prompts collected during dynamic-sampling rollouts at a given training step. For compositional data construction, please refer to Section F. For RL training details with compositional data, please refer to Section B.1 and Section B.2.

- A.2. Details of Initial Evaluation For SPC

For the evaluation in Figure 2 (Right), we randomly sample 200 questions from MATH500 as seed prompts and use SPC to construct level-2 compositional prompts. Specifically, we form pairs (q1,q2) by sampling 5 seed questions as candidates for q1 and pairing each with the 200 seed questions as q2. After filtering, this procedure yields approximately 400 compositional test prompts. We use the same decoding settings as in Section B.3.

#### B. Experimental Details

###### B.1. Training Details

In Section 4.1, we briefly describe the training setup. This appendix provides additional details on our RL training configuration. Two settings are particularly important: we enable dynamic sampling (Yu et al., 2025) to filter uninformative prompts, and we use rollout correction to mitigate training–inference mismatch (Yao et al., 2025).

Hyperparameters and Rollout Settings. Unless otherwise specified, we use a unified set of hyperparameters: batch size 256 (rollout batch size is identical to the mini-batch size), learning rate 1 × 10−6, and no warm-up. We also adopt a unified rollout configuration: temperature 1, top p 1, top k −1, 8 rollouts per problem, and a maximum output length of 16K tokens.

Datasets and Verifiers. For the main experiments in Section 4.2 and Section 4.3, we train on the MATH training set (Hendrycks et al., 2021). Following the standard protocol, we exclude the MATH500 test set, leaving roughly 12K training prompts spanning five difficulty levels. For the cross-topic experiments in Section 4.4, we utilize the physics subset of MegaScience (Fan et al., 2025), which comprises approximately 23K prompts.

For training efficiency, we use Math-Verify as the verifier. Considering that rule-based verifiers do not reliably evaluate model outputs on physics problems (Xu et al., 2025b), we filter the MegaScience physics subset by removing examples for which all eight responses from Qwen3-4B-Thinking-2507 are judged incorrect by Math-Verify. This yields approximately 8.2K prompts on which rule-based verification is reliable.

###### B.2. Baselines In this appendix, we provide additional baseline details for the experiments in Sections 4.2 to 4.4.

- For the experiments in Section 4.2, the baseline corresponds to RL training on the original MATH12K training set, which contains 12K training prompts. Our Composition-RL trains instead on compositional prompts constructed from MATH12K. As described in Section 3.3, for each q2 drawn from the full dataset, we independently resample a uniform 20-prompt subset from which q1 is drawn, yielding 20 × 12K = 240K compositional prompts in principle. As discussed in Section F.1, we apply a verification-and-filtering procedure to improve data quality. After verification of Step 1, approximately 231K prompts remain; after verifying Step 2, this is reduced to roughly 200K; and after the final check, we obtain about 199K

- compositional prompts. We refer to this composition set as MATH-Composition-199K.
- For the experiments in Section 4.3, we include Beyond-80/20 (Wang et al., 2025), AlphaRL (Cai et al., 2025), and RLZVP (Le et al., 2025) as additional reference baselines. We use the models that are initialized from Qwen3-8B-Base and trained on DAPO-MATH-17K prompts from these works. We report their results as RL-zero baselines to our curriculumbased Composition-RL trained from Qwen3-4B-Base. This comparison is unfavorable to Composition-RL due to differences in both model scale and training data. For curriculum Composition-RL, we first train Qwen3-4B-Base on the original MATH12K set (Depth 1). After performance saturates, we switch to the Depth 2 training set (MATH-Composition-199K), and then to the Depth 3 training set once Depth 2 saturates. The construction of the Depth 3 compositional set follows the procedure used for Depth 2. Since Beyond-80/20 (Wang et al., 2025), AlphaRL (Cai et al., 2025), and RL-ZVP (Le et al., 2025) have not released their models at the time we were writing our paper, we report their results as quoted directly from the corresponding papers.
- For the experiments in Section 4.4, we consider two natural RL baselines. The first is RL training on a mixture of MATH12K and the MegaScience Physics subset, which we denote as Mix Training. The second baseline continues RL training on Physics data, starting from a checkpoint trained on MATH12K, which we denote as Math-then-Physics. For Math-thenPhysics, we train until performance saturates. For a fair comparison, we train Mix Training for approximately the same number of total gradient updates as the combined MATH12K stage plus the physics training stage. or Composition-RL, we

consider sampling q1 from Physics and q2 from MATH12K. After filtering, the resulting compositional dataset contains approximately 141K prompts, which we denote as Physics-MATH-Composition-141K

###### B.3. Evaluation Details

To comprehensively evaluate model capabilities, we use a diverse suite of benchmarks spanning mathematical reasoning and multi-task reasoning:

- 1. Mathematical reasoning: We evaluate on AIME24, AIME25, BeyondAIME (ByteDance-Seed, 2025), and IMOBench (Luong et al., 2025). Since AIME24 and AIME25 each contain 30 problems, we report pass@1 using 32 samples per problem (avg@32). BeyondAIME contains 100 problems; we report avg@8. For IMO-Bench, we use the AnswerBench subset to enable rule-based verification; it contains 400 problems, and we report avg@4.
- 2. Multi-task reasoning: We evaluate on GPQA-Diamond (Rein et al., 2024) (approximately 200 problems) and report pass@1 using 8 samples per problem. We also evaluate on MMLU-Pro (Wang et al., 2024); since it contains over 5K problems, we report results from a single run.

All evaluation codes are adapted from the DeepscaleR (Luo et al., 2025) codebase, and we use vLLM (Kwon et al., 2023) to accelerate inference and Math-Verify to evaluate the LLMs’ answers. For decoding, we follow Xu et al. (2025a) and set the temperature to 0.6, top p to 0.95, top k to 20, and the maximum output length to 32K tokens.

###### B.4. Details of Ablation for Candidate Sets Dk As noted in Section 5.1, the default configuration of Composition-RL sets D2 = D (the full prompt pool) and samples D1

- as a small random subset ( |D2| = 12,000, |D1| = 20 ). We also consider two variants: A) Both D1 and D2 are small randomly sampled subsets ( |D1| = |D2| = 500 ). B) D1 is the full set D, while D2 is a small randomly sampled subset ( |D1| = 12,000, |D2| = 20 ). These settings are designed to yield roughly the same theoretical compositional dataset size. After applying the filtering procedure in Section F.1, the resulting actual dataset sizes are:

- • Composition-RL: 199K (see Section B.2).
- • Variant A: 240K after step 1, 202K after step 2, and 200K after step 3.
- • Variant B: 231K after step 1, 201K after step 2, and 200K after step 3.

Thus, the final dataset sizes are approximately matched across configurations.

- C. Analysis Details In this appendix, we provide additional details for Section 5.4.

To evaluate compositional generalization, we use the same setting as in Section A.1. To determine whether the first variable v1 is solved correctly, we prompt Qwen2.5-32B-Instruct using the default generation configuration and the prompt shown in Figure 4.

Prompt for Verifying the Correctness of Finding v1 in LLMs’ Response

You are a math solution verifier. Your task is to check if a given response correctly solves for a specific intermediate variable in a composite math problem.

**Problem:** {problem}

**Target Variable:** ${symbol}$

**Variable Definition:** {definition}

**Correct Answer for ${symbol}$:** {correct answer}

**Model’s Response:** {response}

---

**Your Task:** 1. Identify what value the model computed for ${symbol}$ in its response.

- 2. Compare it with the correct answer: {correct answer}.

- 3. Determine if the model correctly solved for ${symbol}$.

**Important Notes:** - Focus ONLY on whether ${symbol}$ was correctly computed; ignore the final answer of the composite problem.

- - The value might be stated explicitly (e.g., ‘‘${symbol}$ = 7’’) or implicitly derived.
- - Accept equivalent forms (e.g., ‘‘7’’, ‘‘7.0’’, ‘‘seven’’ are all correct if the answer is 7).

**Output in JSON format:** { "extracted value": "<the value the model gave for {symbol}, or ’NOT FOUND’ if not mentioned>", "is equivalent": <true if extracted value equals {correct answer}, false otherwise>, "reasoning": "<brief explanation>", "verdict": "<CORRECT or INCORRECT>" }

Figure 4. The Prompt for Verifying the Correctness of Finding v1 in LLMs’ Response.

- D. More Analysis

Further evidence of implicit process supervision We provide more evidence of our “implicit process supervision” claim based on AIME24 using the 4B model. We consider three metrics: (i) First-Attempt Correctness, defined as the proportion of responses that produce the correct answer with a single \boxed{}; (ii) Reflection Effectiveness, defined as the accuracy conditioned on the presence of self-correction keywords; and (iii) Self-Correction Success Rate, defined as the correctness when the model revises its \boxed{} answer.

As shown in Table 4, our method consistently outperforms the baseline on all three metrics. Specifically, first-attempt correctness improves from 42.8 to 51.7, indicating that our method leads to stronger initial reasoning before any explicit revision. Moreover, the accuracy conditioned on reflection increases from 11.0 to 19.8, suggesting that reflective behavior is more effective under our method (implicit process supervision). Finally, the self-correction success rate rises substantially from 9.1 to 26.1, showing that when the model revises its answer, the revision is much more likely to be correct. Taken together, these results provide further evidence that our method induces implicit process supervision, improving not only final accuracy but also the quality of intermediate reasoning and self-correction behavior.

Error analysis. We further conduct an error analysis on cases where the model can solve both constituent questions in isolation but fails on the corresponding compositional question. We manually inspect 40 such cases and categorize the

###### Metric (%) Baseline Ours

First-attempt correctness 42.8 51.7 P(correct | reflection) 11.0 19.8 Self-correction success 9.1 26.1

Table 4. Additional evidence of implicit process supervision on AIME24 with the 4B model.

Error Type Original Composition-RL Linkage error 22 7

- q1 error 7 1
- q2 error 11 5

- Table 5. Error analysis on 40 cases where the model solves both constituent questions in isolation but fails on the compositional question.
- Table 6. Average response length (in K tokens) across different benchmarks. “ID Avg” denotes the average response length over in-domain mathematics benchmarks, “OOD Avg” denotes the average response length over out-of-domain multi-task benchmarks, and “Avg” denotes the overall average.

Composition

Mathematics (In-Domain) Multi-Task (Out-Of-Domain) Overall AIME 24 AIME 25 Beyond AIME IMO-Answer ID Avg GPQA MMLU-Pro OOD Avg Avg

Qwen3-4B-Base

- ✘ 16.0 13.4 13.3 12.8 13.9 4.2 2.3 3.3 10.3

✔ 14.7 13.2 12.3 12.9 13.3 4.0 2.0 3.0 9.9

Qwen3-8B-Base

- ✘ 17.5 15.7 16.9 15.8 16.5 5.1 2.5 3.8 12.3

- ✔ 14.2 14.9 15.9 14.9 15.0 5.5 2.8 4.2 11.4 Qwen3-14B-Base

✘ 13.6 13.4 12.9 12.1 13.0 3.7 1.9 2.8 9.6

- ✔ 15.3 14.3 15.9 15.9 15.4 4.9 2.2 3.6 11.4 Qwen3-30B-A3B-Base

✘ 13.0 16.1 15.3 16.2 15.2 4.4 2.3 3.4 11.2 ✔ 13.1 15.6 14.5 15.3 14.6 4.3 2.1 3.2 10.8

errors into three types: (1) Linkage error, where the model fails at the “linkage condition” for any reasons; (2) q1 error, where the model makes an error on q1 when solving it in the compositional setting, despite being able to solve it correctly in isolation; and (3) q2 error, where the model solves q1 and derives the linkage correctly but still fails on q2. As shown in Table 5, linkage error is the dominant failure mode for the original model, accounting for 22 out of 40 cases, compared with

- 7 q1 errors and 11 q2 errors. After applying Composition-RL, all three error types are reduced, with the largest reduction observed in linkage errors (from 22 to 7). Qualitatively, we find that linkage errors often arise from incorrect intermediate computation, unproductive self-verification loops, or selecting an incorrect linkage value despite otherwise correct reasoning. Interestingly, these examples also shed light on common failure modes of current reasoning models: (1) intermediate computation errors during the thinking process, (2) repeated self-verification loops that lead to truncation without progress, and (3) loss of direction mid-reasoning even when correct intermediate results are available. Composition-RL reduces linkage errors across all three failure modes, confirming that it teaches models to maintain correct intermediate reasoning throughout multi-step problems.

#### E. More Results

- E.1. About Average Response Length In Section 4, we primarily discuss results based on pass@1; here, we also report the average response length in Table 6. As

- shown in Table 6, Composition-RL does not significantly increase response length across all model scales.

Table 7. Pass@k (%) across different benchmarks. The value of k for each benchmark is consistent with that used in the corresponding Avg@k setting in the main results table. “ID Avg” denotes the average pass@k over in-domain mathematics benchmarks, “OOD Avg” denotes the average pass@k over out-of-domain multi-task benchmarks, and “Avg” denotes the overall average.

Mathematics (In-Domain) Multi-Task (Out-Of-Domain) Overall AIME 24 AIME 25 Beyond AIME IMO-Answer ID Avg GPQA MMLU-Pro OOD Avg Avg

Composition

Pass@32 Pass@32 Pass@8 Pass@4 Avg. Pass@8 Pass@1 Avg. Avg.

###### Qwen3-4B-Base

✘ 56.7 46.7 23.0 25.5 38.0 79.3 58.6 69.0 48.3 ✔ 66.7 56.7 31.0 26.0 45.1 77.8 61.4 69.6 53.3

###### Qwen3-8B-Base

✘ 66.7 46.7 29.0 30.7 43.3 81.3 62.6 72.0 52.8 ✔ 73.3 53.3 35.0 32.0 48.4 81.3 64.5 72.9 56.6

###### Qwen3-14B-Base

✘ 70.0 70.0 39.0 34.5 53.4 81.8 67.2 74.5 60.4 ✔ 76.7 76.7 39.0 41.5 58.5 81.8 69.3 75.6 64.2

###### Qwen3-30B-A3B-Base

✘ 73.3 56.7 39.0 40.0 52.3 83.8 64.5 74.2 59.6 ✔ 76.7 63.3 40.0 39.0 54.8 84.8 65.6 75.2 61.6

###### E.2. About pass@k

In Section 4, we primarily discuss results based on pass@1; here, we additionally report pass@k results in Table 7. As shown in Table 7, Composition-RL consistently improves pass@k across model scales, indicating that it broadens the set of solvable problems.

###### E.3. Ratio of solve all and solve none

We also report the proportions of solve all and solve none examples across training steps in Figure 5. As shown in Figure 5,

- at later stages of training, Composition-RL exhibits a lower solve all ratio and a comparable solve none ratio. Therefore, Composition-RL yields a larger amount of effective training data overall.

#### F. Details of SPC

###### F.1. Reliability of SPC

The full SPC pipeline can be automated with an LLM assistant; implementation details and the corresponding prompts are provided in Section F.2 and Section F.3, respectively. To make this automated process more reliable, we have some additional verification steps to filter potential mistakes during composing. Following (Xiao & Zhao, 2025), we use LLM-based self-verification at each composition step. Concretely, we prompt the same LLM to perform the following checks:

- • ❶ Verification of “Modify q1 with gt1.” In this step, the LLM extracts a variable v1 from gt1 and its definition d1. We then ask the LLM to compute the value of v1 given q1 and d1, and compare the computed value against the extracted v1. If they do not match, we discard the prompt. This verification improves the reliability of the modification of q1.
- • ❷ Verification of “Modify q2.” Analogously, we prompt the LLM to verify whether the extracted variable v2 (and its definition) is consistent with q2. Prompts that fail this check are filtered out.
- • ❸ Verification of “Connect q1 and q2.” This step primarily involves concatenation. To ensure quality, we prompt the LLM to check for inconsistencies (e.g., conflicting variable names) and filter out any inconsistent prompts.

This verification procedure removes many low-quality compositions, leaving a substantially more reliable set of composed prompts. As reported in (Xiao & Zhao, 2025), the rate of erroneous prompts after filtering is below 2%. We believe this error rate is acceptable for training.

0.8

Composition

0.5

Original

0.7

0.6

0.4

solve_noneratio

solve_allratio

0.5

0.3

0.4

0.3

0.2

0.2

0.1

0.1

Composition

Original

0.0

0 50 100 150 200 250

0 50 100 150 200 250

Training Steps

Training Steps

0.35

0.8

Composition

Original

0.7

0.30

0.6

solve_noneratio

0.25

solve_allratio

0.5

0.20

0.4

0.15

0.3

0.2

0.10

0.1

Composition

0.05

Original

0.0

0 50 100 150 200 250

0 50 100 150 200 250

Training Steps

Training Steps

Composition

0.8

Original

0.25

solve_noneratio

0.6

0.20

solve_allratio

0.15

0.4

0.10

0.2

0.05

Composition

Original

0.0

0 50 100 150 200 250

0 50 100 150 200 250

Training Steps

Training Steps

Composition

0.8

0.35

Original

0.30

0.6

solve_noneratio

solve_allratio

0.25

0.20

0.4

0.15

0.2

0.10

Composition

0.05

Original

0.0

0 50 100 150 200 250

0 50 100 150 200 250

Training Steps

Training Steps

Figure 5. solve all and solve none ratio across model sizes.

###### F.2. Implementation Details

We use Qwen2.5-32B-Instruct (Team, 2024) with step-specific prompts to implement each stage of Section 3.1 as well as the verification procedure in Section F.1. Unless otherwise specified, we set the temperature to 0.1, top p to 0.7, and the maximum output length to 4096 tokens. The prompts are provided in Section F.3.

###### F.3. Prompts of SPC

Following (Xiao & Zhao, 2025), we provide the prompt used to modify q1 in 6 and the self-verification prompt used to check the modification in 7. We use similar prompts for the other steps of SPC, and we will release the complete set of prompts in our codes.

###### Prompt for Modifying q1 with gt1

Given a math problem and the final answer, your task is to find out one number from the answer and provide the corresponding definition. Follow the steps below:

- Step 1: Identify a specific integer, float, or fraction within final answer and name it as new variable1; There are several situations:

- 1. If the final answer contains unknown variables:

- (a) If the final answer is an expression, choose one coefficient as new variable1, for example, 2x + 3, you can choose the coefficient of x as new variable1, which is 2, and in the case of sin(x), there is a hidden coefficient 1 and a hidden amplitude 1, you can choose either one as new variable1;

- (b) If the final answer is an equation, you can choose one solution as new variable1, for example, y = 2x + 1, you can define the value of y as new variable1 when given x = 1, which is 3;

- (c) If the final answer is a symbol of an option or a word, such as ‘A’, ‘B’, ‘CAT’, etc., use their first letter’s order in the alphabet as a variable, such as ‘A’ = 1, ‘B’ = 2, ‘CAT’ = 3, etc.;

- (d) If the final answer contains 2 or more items, e.g., multiple choice questions, choose the smallest or the largest one, and then apply the corresponding situation.

- 2. If the final answer has no unknown variables, there are several situations:

- (a) If the final answer itself is a numerical value, like ‘four’, ‘4’, ‘2 + √2’, ‘3π’, and ‘34’, use it directly as new variable1;

- (b) If the final answer contains 2 or more numerical values, use the largest or the smallest one as new variable1;

- (c) If the final answer is an interval or ratio, choose one boundary and ∞ is not allowed, for example, [2,∞), you can define the lower bound as new variable1, which is 2;

- (d) If the final answer is a ratio, choose one part of the ratio, for example, 3 : 4; you can define the first part of the simplified ratio as new variable1, which is 3;

- (e) If the final answer is a non-base 10 number, for example, 10012, you can define ‘the number of digits in the base 2 representation’ as new variable1, which is 4;

- (f) If the final answer is an angle or degree, choose the corresponding radian value, for example, 30◦ or 30◦, define the corresponding radian value of final answer as new variable1, which is π/6.

All in all, find a way to identify a specific numerical value as new variable1 without unknown, and make sure the reader can get the value of new variable1 from the final answer through your definition.

- Step 2: Output the value of new variable1, keep the exact value or math symbol, and simplify the fraction if necessary, for example, keep π as π, keep √2 as √2, and simplify 68 as 34, without rounding to a decimal point.

- Step 3: Output the definition of new variable1 without mentioning the real value.

—

Output Format: ... (omit for simplicity) Examples: ... (omit for simplicity)

Figure 6. The Prompt for Generating Variable v1 and Definition d1 for q1.

###### Prompt for Verifying the Modification of q1

- 1. Check the extraction of variable v1: {Problem 1} Assume that the final answer of the problem is {FINAL_ANSWER}. {DEFINITION_OF_NEW_VARIABLE1} Then what is the value of new variable1? Please output the value of new variable1 directly, wrapping it in \boxed{}, for example, \boxed{3}.

- 2. Check the value of v1 using Python:

**Task Description:** Write a Python program to compare two given values and determine if they are equal. Follow these guidelines:

- 1. Use the sympy library to handle symbolic comparisons, ensuring that equivalent expressions (e.g., 24 and 12) are recognized as equal.

- 2. For values involving irrational constants (e.g., π, e), perform comparisons up to two decimal places for practical equivalence.
- 3. Include clear intermediate steps in the program, such as evaluating or simplifying the values where appropriate.
- 4. Wrap the final comparison outcome in a \boxed{} command for clarity.
- 5. Provide both the Python code and the results of running the code.

**Output Format:**

‘‘‘python {The Python code that compares the two given values, including print statements for intermediate steps and the \boxed{final comparison outcome}.} ‘‘‘

‘‘‘output {The output of the Python program.} ‘‘‘ --[Examples Here]

Figure 7. The Prompt for Verifying the Modification of q1.

