How Far Ahead Do LLMs Plan?
Uncovering the Latent Horizon in Chain-of-Thought Reasoning
Liyan Xu 1 Mo Yu 1 Fandong Meng 1 Jie Zhou 1
Abstract
Chain-of-thought (CoT) reasoning has become
a central mechanism for eliciting multi-step rea-
soning in Large Language Models (LLMs). Yet
recent evidence presents a tension: hidden states
appear to already encode future reasoning before
CoT fully unfolds, while explicit steps still remain
crucial for tasks requiring compositional com-
putation. To deepen the understanding between
LLM’s internal states and its verbalized reason-
ing trajectories, we investigate the latent planning
strength of LLMs, through our probing method,
Tele-Lens, applying to hidden states across di-
verse task domains. Our empirical results indicate
that LLMs exhibit a myopic horizon, primarily
conducting incremental transitions without pre-
cise global planning. Leveraging this characteris-
tic, we propose a hypothesis on enhancing uncer-
tainty estimation of CoT, which we validate that a
sparse set of pivot positions can effectively repre-
sent the uncertainty of the entire path. We further
underscore the significance of exploiting CoT dy-
namics, and demonstrate that automatic recogni-
tion of CoT bypass can be achieved without per-
formance degradation. Our code, data and models
are released at https://github.com/lxucs/tele-lens.
1. Introduction
Chain-of-Thought (CoT) (Nye et al., 2021; Wei et al., 2022)
has fundamentally reshaped problem-solving in natural lan-
guage processing, marking a shift from traditional pattern-
matching approaches, e.g. encoder-based classification (De-
vlin et al., 2019; Liu et al., 2019), toward prompt-based
reasoning articulated explicitly in natural language (Zhou
et al., 2023; Dong et al., 2024; Sahoo et al., 2025). The
1WeChat
AI,
Tencent
Inc.
Correspondence
to:
Liyan
Xu
<liyanlxu@tencent.com>,
Jie
Zhou
<with-
tomzhou@tencent.com>.
Proceedings of the 43 rd International Conference on Machine
Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026
by the author(s).
capacity of CoT is further amplified through extensive think-
ing emanated from reinforcement learning, characterized by
recent models such as DeepSeek-R1 (DeepSeek-AI, 2025).
While CoT is widely perceived as the de facto reasoning
paradigm, however, recent studies on Large Language Mod-
els (LLMs) have revealed complementary, and at times
seemingly conflicting perspectives. On the one hand, LLMs
have been shown to exhibit internal planning on the rea-
soning trace prior to the explicit emergence of CoT. (Pal
et al., 2023) reveals that early hidden states already encode
information about subsequent generations; (Sheng et al.,
2025) finds that reasoning capability is largely pre-planned
before generation begins; (Dong et al., 2025) observes that
early hidden states can reliably predict reasoning steps and
key attributes with high correlation. Similarly, numerous
studies have shown that the initial stages of CoT can effec-
tively anticipate the final answers (Azaria & Mitchell, 2023;
Gottesman & Geva, 2024; Afzal et al., 2025).
The internal planning of LLMs appear to diminish the sig-
nificance of CoT generation, raising the question of whether
the thinking process is just echoing pre-determined paths
already encoded in prior internal states. On the other hand,
theoretical analyses advocate that CoT is indispensable due
to the limited expressivity of Transformers bounded by
its architectures (Bhattamishra et al., 2023; Merrill & Sab-
harwal, 2023; Li et al., 2024), and only intermediate steps
of CoT can derive length generalization (Anil et al., 2022;
Xiao & Liu, 2025) and compositional reasoning (Wies et al.,
2023; Abbe et al., 2024; Zubic et al., 2025). Therefore, the
manifestation of pre-calculated trajectories appear unlikely
via internal planning before the onset of CoT.
Nonetheless, the relationship between the model’s internal
representations and its verbalized reasoning tokens largely
remains opaque. In this work, we investigate the internal
dynamics of CoT, and target the following questions con-
cerning the latent planning horizon:
Q1. To what extent do hidden states encode a global
plan for the reasoning roadmap, as opposed to sup-
porting rather local, incremental state transitions?
Q2. How does the planning horizon further influence
other characteristics of CoT reasoning?
1
arXiv:2602.02103v2  [cs.LG]  28 May 2026

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
We believe answering these questions is important not only
for deepening our understanding of CoT dynamics, but also
for providing critical insights into broader model-thinking
design scenarios. For example, recent works leverage LLM
internal states to enable early exiting from CoT reasoning,
thereby mitigating the overthinking problem (Yong et al.,
2025; Yang et al., 2026). Recently, both GPT-51 and Claude
Code2 introduce forms of adaptive thinking that route user
requests to models of different sizes or allocate varying
levels of reasoning effort. Such mechanisms fundamentally
rely on how strong the model can “sense” or “see through”
the input complexity through internal planning.
Towards this objective, we derive empirical insights by ex-
amining the synergy between explicit CoT steps and its
latent planning horizon. Building on the observations, we
then highlight the significance of leveraging CoT dynamics
on estimating CoT’s uncertainty and necessity.
To answer the first question, Section 2 presents a series of
probing experiments designed for dissecting LLM hidden
states, aiming to evaluate the internal planning strengths
with respect to future reasoning trajectories. We first in-
troduce our probing method, termed Tele-Lens, which em-
ploys a trained low-rank adapter (Houlsby et al., 2019) that
transforms each hidden state within CoT steps to predict
Teleological information along multiple dimensions, includ-
ing subsequent tokens, final answers, reasoning lengths, etc.
Importantly, unlike prior works that primarily address single-
domain tasks, we conduct probing experiments across 12 di-
verse datasets spanning different classes and domains, rang-
ing from straightforward knowledge question answering to
classic hard problems, e.g. Parity (counting the number of
digits as even or odd), a canonical challenge for Transform-
ers (Chiang & Cholak, 2022; Hahn & Rofin, 2024).
By empirical results, we observe sharply contrasting be-
haviors across probing dimensions and task domains, as
detailed in Section 2.5. For instance, while early hidden
states may encode predictive signals of the final answer for
relatively simple tasks, they can behave nearly randomly on
compositional tasks, where they only begin to reliably cap-
ture the precise answer within the final one or two reasoning
steps before CoT completion.
Overall, our probing results bring a unified view of the
prior complementary beliefs from previous works: LLMs
exhibit a myopic planning horizon, in which hidden states
primarily support immediate, local transitions rather than
long-range, global trajectories. However, for simpler tasks
that fall within the LLM’s single-step pattern-matching ca-
pacity, early hidden states may capture a coarse gist of the
1https://openai.com/index/gpt-5-system-card
2https://platform.claude.com/docs/en/build-with-
claude/adaptive-thinking
final answer—albeit in a more heuristic manner, rather than
through a precise, pre-planned reasoning process.
Addressing Q2, leveraging latent CoT signals, we first focus
on uncertainty calibration over CoT, where an effective
confidence metric, e.g. the rollout perplexity or entropy,
should assign high scores to correct reasoning paths and
low scores to uncertain ones (Huang et al., 2024; Chen
et al., 2024; Bakman et al., 2025). We propose a hypothesis
followed by empirical validation: the uncertainty of CoT
follows a Wooden Barrel principle. Just as the capacity of
a barrel is determined not by its average stave height but by
its shortest stave, the reliability of a reasoning chain is gov-
erned by a small number of pivot positions. Intuitively, as
the model’s latent planning is myopic, most CoT tokens are
high-confident local transitions that may dilute the underly-
ing uncertainty of the entire trace. We therefore speculate
that focusing on a small set of pivot positions instead of
global aggregates could enable more precise uncertainty
estimation. Empirical results in Section 3.1 find that even a
simple strategy of top-k selection can effectively enhance
the accuracy of estimation across all three general uncer-
tainty metrics, yielding up to 6% absolute improvement.
Beyond uncertainty estimation, we also present a proof-of-
concept that the CoT planning patterns can be leveraged to
recognize whether CoT is necessary to derive the final an-
swer, achieving automatic CoT bypass that directly outputs
the answer with minimal performance degradation. Experi-
ments in Section 3.2 demonstrate that our proposed strategy
using Qwen3-32B can realize up to 16.2% CoT bypass with
only a negligible 0.03 overall accuracy drop.
Back to Q2, our proposed strategies on CoT uncertainty and
necessity estimation further underscore the significance of
analyzing CoT dynamics, which encode hidden yet valuable
information. We hope that this work on uncovering the
latent planning horizon could advance the understanding of
CoT synergy, and spur more identification of hidden signals
to be exploited in broader model-thinking scenarios.
2. CoT Planning Horizon
This section delineates the detailed experimental setup and
findings on diving into the latent planning capacity. Sec-
tion 2.1 introduces our probing method, Tele-Lens, followed
by a description of the data setup and model configurations.
Empirical results are reported in Section 2.5.
2.1. Tele-Lens
To enable probing across multiple dimensions, including
subsequent token prediction, our method is designed to sup-
port prediction over the full LLM vocabulary. To this end,
we adopt a transformation-based approach to probe various
Teleological information upon the CoT trace, dubbed Tele-
2

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
Lens. It follows the paradigm of Logit Lens (nostalgebraist,
2021; Belrose et al., 2023), originally for examining layer-
wise interpretability in Transformers, which bridges the
hidden states from intermediate Transformers layers to the
final LM head directly, thereby enabling whole-vocabulary
prediction. For Tele-Lens, to mitigate overfitting and com-
putational overhead, we adopt a bottleneck low-rank adapter
(Houlsby et al., 2019) with added nonlinearity for hidden
state transformation, more formally described as follows.
Concretely, for an LLM rollout, we denote the response
tokens in its thinking process up to the final answer as
T = {t1, t2, .., tn}, representing a reasoning trajectory of
length n (throughout this paper, we use the terms “thinking”
and “CoT” interchangeably). The hidden state correspond-
ing to token ti at the k-th Transformers layer is then denoted
as Hk
i ∈Rd, with d being the LLM hidden size. The cor-
responding transformed hidden state eHk
i ∈Rd by applying
the bottleneck adapter, and its predicted probability distribu-
tion Pk
i over the LLM vocabulary V, are defined as:
eHk
i = GeLU
 Hk
i + Embk(δ)

Ak
Bk
(1)
Pk
i (V | ti, Ak, Bk, Embk, δ) = Softmax
  eHk
i L

(2)
where Ak ∈Rd×r, Bk ∈Rr×d and Embk ∈Rm×d are
the learnable parameters of the adapter for the k-th Trans-
formers layer, typically with a low rank r < d. Particularly,
Embk is an optional embedding matrix, taking an offset
δ = 1, 2, .., m to inject the target predicting position up to
m offset. L ∈Rd×|V| is the LM head matrix that will keep
frozen during adapter training.
For each token ti in the reasoning path, we take its hid-
den state of each Transformers layer and probe along three
teleological dimensions:
• Subsequent tokens: we use solely Hk
i to predict its m
following tokens {ti+δ | δ = 1, 2, .., m}. Each offset
δ is injected respectively as in Equation (1).
• Reasoning length: we use Hk
i to predict the total
length of the thinking. Instead of applying LM head
by Equation (2), we take eHk
i followed by a single
regression layer to yield a number prediction.
• Final answer: we use Hk
i to predict the final answer
directly, with Embk removed in Equation (1). Each
answer should be uniquely identifiable by a token in V,
thus this suits only for tasks with a fixed answer space.
If LLMs maintain a precise global plan early in the rea-
soning process, then the strength of that plan should, in
principle, be measurable by these probing targets, before
explicit CoT fully unfolds. For final-answer probing partic-
ularly, as answer tokens constitute the natural continuation
of the reasoning process, they can reflect how early, and
to what extent, future outcome information is internally
represented throughout CoT generation.
2.2. Tasks and Datasets
As previous works on CoT analysis mainly focus on specific
domains of interest, the findings can be complementary that
reflects different perspectives and angles, as discussed in
Section 1. Towards more comprehensive empirical insights,
we broaden the scope of domains and include 12 diverse
tasks, which we categorize into three types as below. Con-
crete examples of these tasks are provided in Appendix A.1.
Explicit Compositional Tasks
These tasks require ex-
plicit multi-step procedures to resolve, involving a high
degree of structural modularity. Notably, as suggested by
both prior empirical studies and theoretical analyses, Trans-
formers often struggles to efficiently perform function com-
position within a single forward pass (Dziri et al., 2023;
Merrill & Sabharwal, 2023; Zubic et al., 2025). Conse-
quently, such tasks usually require intermediate CoT steps
to derive the final answer. We include the following three
tasks, for which data generation is fully controllable.
• Parity: a classic task often seen in Transformers’ expres-
sivity and learnability analysis (Chiang & Cholak, 2022;
Bhattamishra et al., 2023; Hahn & Rofin, 2024). Given a
sequence of digits, the task essentially asks whether the
total count of a target digit is even or odd.
• Cycle: we adopt a task introduced by Abbe et al. (2024),
in which the input consists of a list of directed edges,
forming either a single full-sized cycle or two half-sized
cycles. The task requires determining whether there exists
a path between two specified vertices, or equivalently,
whether they fall into the same cycle.
• Subsum: an algorithmic task, Max Subsequence Sum,
adopted in prior Transformers studies (Dziri et al., 2023).
For a list of n numbers, the task computes the maximum
sum of its subsequences, which admits an O(n) dynamic
programming solution. We query the least significant
digit of the maximum sum for a fixed answer space.
Implicit Compositional Tasks
These tasks typically re-
quire multiple reasoning steps as well, but in a more nuanced
and implicit manner embedded in the problem semantics,
such as mathematical or logical reasoning. For math-related
tasks, we adopt three following datasets: GSM8K (Cobbe
et al., 2021), MATH (Hendrycks et al., 2021b), AIME
(AIME, 2025). To enable a fixed answer space tailored
for final-answer probing, we adapt each problem into a
multi-choice format, by prompting GPT-4.1 to generate
plausible yet misleading distractor options. Details on the
multi-choice conversion are provided in Appendix A.3.
For logical reasoning, we include the following two datasets
that evaluate soft reasoning framed in natural language:
MuSR (Sprague et al., 2024), Zebra (Lin et al., 2025).
Knowledge and Semantic Tasks
These tasks primarily
focus on knowledge-intensive queries grounded in the pro-
3

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
vided semantic context, without a particular focus on in-
tense reasoning. This category comprises four datasets:
CSQA (CommonsenseQA, Talmor et al. (2019)), MMLU
(Hendrycks et al., 2021a), QuALITY (Pang et al., 2022),
GPQA (Rein et al., 2024). Brief descriptions of all existing
datasets are further provided in Appendix A.2.
As all 12 tasks have a fixed answer space, with most of
them being multi-choice questions, each answer is uniquely
identifiable by a token in the vocabulary. Accordingly, the
label set for the final-answer probing is constituted by 20
tokens in total, as detailed in Appendix A.3.
2.3. LLM Backbones
To obtain response rollouts and corresponding hidden states,
we consider two types of LLM backbones described below.
Off-the-Shelf LLM
As our probing experiments require
access to both model weights and reliable CoT outputs, we
employ the open-source Qwen3 series with native support
of both thinking and non-thinking modes. We use Qwen3-
32B as the primary backbone to ensure robust performance
while maintaining manageable computational cost.
In-Domain LLM
In addition to open-source LLMs with
readily available thinking modes, we also employ an in-
domain LLM trained with task-specific supervision, for two
key reasons. First, a task-aware model exhibits more stable
and decisive reasoning, thereby serving as an “upper bound”
on internal planning capacity. Second, this setup helps
reduce potential confounding factors inherent to general-
purpose LLMs tied to specific model families.
Our In-Domain LLM learns task-aware CoT via reinforce-
ment learning with GRPO (Shao et al., 2024). We intention-
ally train from Qwen2.5-7B-Instruct, which does not have
thinking mode natively, allowing for a cleaner bootstrap of
CoT behavior on these tasks. We introduce our detailed
GRPO training settings in Appendix B.
2.4. Experimental Settings
Dataset Construction
We construct our probing datasets
with train/dev/test splits across the 12 tasks, which contain
up to 4000 / 100 / 500 problems per task, respectively. For
the three tasks—Parity, Cycle, and Subsum—the problems
are obtained via data generation. For other tasks, problems
are sampled from their original datasets. Details of our data
generation and sampling, as well as further statistics are
provided in Appendix A.4 and A.5.
Training and Hyperparameters
For each probing dimen-
sion, we train a dedicated Tele-Lens adapter for each Trans-
formers layer of a LLM backbone, using a rank of r = 256.
Each training run is conducted for approximately 5K steps,
with early stopping on the dev set. More hyperparameters
for adapter training are provided in Appendix A.6.
2.5. Empirical Results
We first report the performance of LLM backbones to char-
acterize the 12 tasks, evaluating off-the-shelf Qwen3 and our
trained In-Domain LLM. Full results are provided in Table 5
(Appendix B.2), from which we draw the observations:
• For those compositional tasks requiring explicit multi-
step reasoning, direct answering without CoT can only
achieves near-random performance (e.g. Parity, Cycle),
corroborating prior findings on the expressivity limits of
Transformers (Chiang & Cholak, 2022; Merrill & Sab-
harwal, 2023; Zubic et al., 2025). For other tasks, CoT
generally yields substantial improvement as well.
• Owing to differences in model generation and scale, our
in-domain LLM underperforms the naive Qwen3 mod-
els on certain datasets. Despite this, it achieves the best
performance on three compositional tasks and attains
overall performance comparable to Qwen3, while produc-
ing substantially shorter CoT trajectories (approximately
1K+ characters per CoT, compared to 10K+ for Qwen3).
These results validate the training effectiveness in in-
ducing more stable and decisive reasoning paths. A
qualitative CoT comparison is provided in Appendix B.3.
The latent planning horizon of In-Domain LLM is thus
viewed as an “upper bound” for these tasks.
With Tele-Lens adapters trained and evaluated on the col-
lected CoT trajectories, we present the empirical observa-
tions for each probing dimension as follows.
2.5.1. PLANNING FOR FINAL ANSWERS
Figure 1 first presents the average probing accuracy with
In-Domain LLM along the initial CoT positions (full results
across all tasks in Figure 13). The overall trend with Off-the-
Shelf Qwen3 is also similar as in Figure 14. At first glance,
it is clear that different Transformers layers exhibit varying
predictive capacities. Notably, the highest performance does
not occur at the final layer, but rather at layers between the
middle and the last, which is consistent with findings of
prior works that intermediate layers encode richer semantic
information (Reif et al., 2019; Gar´ı Soler & Apidianaki,
2021; Skean et al., 2025). For analyses in this section, we
focus on results by layer 48 (64 total) for Off-the-Shelf
Qwen3 and layer 21 (28 total) for In-Domain LLM.
It is worth noting that although Figure 1 plots only the initial
positions, the probing is performed over the complete CoT
trajectories. We further present the accuracy evolution along
the full individual reasoning traces in Figure 2, Figure 15
and Figure 16. Overall, we draw two key findings as below.
4

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
0
1
2
3
4
CoT Position
18
19
20
21
22
23
24
25
26
27
28
Layer
0.51
0.51
0.47
0.47
0.48
0.52
0.49
0.48
0.52
0.48
0.51
0.51
0.51
0.49
0.51
0.51
0.5
0.47
0.51
0.5
0.51
0.49
0.47
0.49
0.49
0.49
0.49
0.49
0.47
0.5
0.47
0.51
0.51
0.51
0.5
0.51
0.5
0.49
0.48
0.49
0.51
0.51
0.51
0.51
0.5
0.51
0.49
0.49
0.5
0.49
0.49
0.47
0.49
0.49
0.49
Parity (Random: 0.5)
0
1
2
3
4
CoT Position
18
19
20
21
22
23
24
25
26
27
28
Layer
0.23
0.23
0.25
0.23
0.24
0.24
0.24
0.27
0.28
0.29
0.24
0.25
0.27
0.29
0.26
0.25
0.25
0.26
0.26
0.25
0.26
0.27
0.26
0.26
0.27
0.23
0.28
0.24
0.23
0.24
0.3
0.24
0.25
0.28
0.28
0.29
0.28
0.28
0.26
0.25
0.23
0.28
0.25
0.27
0.25
0.24
0.24
0.23
0.23
0.26
0.26
0.27
0.24
0.25
0.26
MATH (Random: 0.2)
0
1
2
3
4
CoT Position
18
19
20
21
22
23
24
25
26
27
28
Layer
0.27
0.26
0.35
0.32
0.41
0.3
0.28
0.4
0.43
0.6
0.4
0.33
0.63
0.55
0.76
0.42
0.45
0.51
0.53
0.71
0.28
0.31
0.39
0.43
0.68
0.27
0.35
0.42
0.35
0.55
0.33
0.33
0.46
0.43
0.56
0.3
0.34
0.32
0.31
0.47
0.27
0.34
0.39
0.3
0.65
0.28
0.3
0.3
0.3
0.49
0.29
0.3
0.3
0.3
0.48
Zebra (Random: 0.17)
0
1
2
3
4
CoT Position
18
19
20
21
22
23
24
25
26
27
28
Layer
0.23
0.24
0.42
0.41
0.35
0.3
0.51
0.56
0.51
0.51
0.5
0.64
0.65
0.52
0.51
0.54
0.68
0.6
0.52
0.55
0.57
0.66
0.61
0.5
0.46
0.48
0.66
0.61
0.39
0.41
0.51
0.63
0.58
0.4
0.4
0.25
0.53
0.54
0.38
0.48
0.29
0.6
0.57
0.34
0.47
0.3
0.51
0.54
0.34
0.43
0.27
0.48
0.51
0.33
0.43
MMLU (Random: 0.25)
0
1
2
3
4
CoT Position
18
19
20
21
22
23
24
25
26
27
28
Layer
0.33
0.33
0.32
0.3
0.3
0.28
0.33
0.31
0.25
0.23
0.3
0.32
0.37
0.33
0.3
0.39
0.38
0.32
0.31
0.28
0.44
0.39
0.34
0.26
0.28
0.41
0.42
0.34
0.31
0.3
0.36
0.42
0.37
0.22
0.24
0.36
0.44
0.38
0.26
0.27
0.37
0.37
0.35
0.22
0.25
0.28
0.38
0.35
0.27
0.28
0.27
0.28
0.33
0.23
0.27
GPQA (Random: 0.2)
0.45
0.50
0.55
0.60
0.65
0.70
0.2
0.3
0.4
0.5
0.6
0.7
0.2
0.3
0.4
0.5
0.6
0.7
0.3
0.4
0.5
0.6
0.7
0.2
0.3
0.4
0.5
0.6
0.7
Figure 1. Results for the final-answer probing: average accuracy of In-Domain LLM for the first five tokens within CoT trajectories,
measured across selected Transformers layers and tasks. The full figure across all tasks is presented in Figure 13 (see Appendix C).
0
200
400
600
800
1000
CoT Position
0
20
40
60
80
100
Probability (%)
LEFT: '<think> To determine if the count of the digit \'8\' in the sequence "88679...24831" is even or
odd, I will go digit by di ......  - \'8\': count = 20 - \'3\': count = 20 - \'1\': count = 20 At the end
of the sequence, the total count of the digit \'8\' is 2'
RIGHT: '0. Since the count is 20, which is an even number, the number of times the digit \'8\' appears in
the sequence "88679...24831" is even. Thus, the answer is: \\boxed{even} </think>'
(a) Parity example with In-Domain LLM.
0
100
200
300
400
CoT Position
0
20
40
60
80
100
Probability (%)
LEFT: '<think> To determine if there is a path from vertex v_344 to vertex v_460, I will trace the
directed edges starting from ...... -> v_161 19. v_161 -> v_399 20. v_399 -> v_210 21. v_210 -> v_249 22.
v_249 -> v_918 23. v_918 -> v_941 24. v_941 -> v_3'
RIGHT: '44 At this point, we loop back to v_344, so the path does not reach v_460. Since the path from
v_344 leads to a loop and does not reach v_460, the answer is "NO". </think>'
(b) Cycle example with In-Domain LLM.
Figure 2. Examples of final-answer probing accuracy along full CoT trajectories with In-Domain LLM (random guessing is at 50%). The
vertical dashed line indicates the position at which accuracy first spikes. “LEFT” and “RIGHT” at the bottom illustrate the reasoning
details right before and after the accuracy spike, respectively. Similar examples with Off-the-Shelf LLM are provided in Figure 15.
▶For precise future planning, LLMs exhibit a myopic
horizon rather than long-term foresight.
To illustrate this, we focus on explicit compositional tasks,
where their initial final-answer planning is near random,
shown by Parity, Cycle and Subsum in Figure 13&14. Anal-
ysis of the full planning dynamics, depicted by the two ex-
amples in Figure 2, reveals that the precise final-answer only
emerges one step before the reasoning completion, such that
the probability of final answers remain flat before the final
spike in the end: the final answer for Parity is planned only
after the counting of all digits, and for Cycle, it is planned
only after observing a complete path or cycle. As answer
tokens are natural continuations of the CoT reasoning pro-
cess, they act as anchors that reflect short-sighted internal
planning, which fails to capture multi-step foresight.
To demonstrate quantitatively, we parse the CoT trajecto-
ries and obtain the final-answer probability at each critical
intermediate steps. For Parity, we report the probabilities
of CoT positions right after counting each digit. As shown
in Table 1, the probing only converges at the final counting
position, exceeding 90%, while for preceding positions, the
accuracy hovers around random guessing as 50%.
For simpler tasks, the probing also only stabilizes near the
reasoning end. More illustrations are provided in Figure 16.
▶LLMs can exhibit coarse signals for final answers
in early stages of CoT, but reflecting only a pattern-
matching gist, rather than precise reasoning plans.
Table 1. Probing probabilities for Parity at CoT positions immedi-
ately following the counting of each of the last four digits in the
sequence (random as 0.5). Position 0 denotes the highest probing
probability after all digits have been counted (the upper bound).
-4
-3
-2
-1
0
In-Domain LLM
0.49
0.51
0.51
0.97
0.99
Off-the-Shelf LLM
0.50
0.52
0.51
0.94
0.97
As shown in Figure 1, LLMs can sometimes “sense” the
gist of the answer early on, particularly for those emphasiz-
ing semantic understanding rather than explicit multi-step
reasoning. To illustrate with more clarity, Figure 3 depicts
the probing dynamics on CSQA and MMLU, which focus
more on semantics and knowledge, in which an early spike
in probing accuracy is notably evident, in contrast to tasks
like Parity. By the full evaluation results presented in Fig-
ure 17 and Figure 18, it appears that early hidden states do
possess certain information predictive of the final answers,
just as observed in prior works (Azaria & Mitchell, 2023;
Gottesman & Geva, 2024; Afzal et al., 2025).
However, our in-depth analysis suggests that these coarse
predictive signals primarily reflect a vague perceptual cue,
but not resulting from exercising a pre-planned reasoning
path. We proceed to compare the performance of this early
coarse signal, with that of true reasoning via CoT, as well
as direct answering without CoT; the results are presented
in Figure 4 (full results in Figure 19). Across almost all
tasks, early final-answer planning yields lower task accuracy
5

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
<think>
(100%)
(100%)
Okay
(100%)
,
(100%)
let
(97.6%)
's
(84.7%)
see
(86.5%)
.
(86.5%)
I
(86.5%)
need
(86.5%)
to
(86.5%)
figure
(76.8%)
out
(76.8%)
if
(77.4%)
the
(77.8%)
number
(77.4%)
of
(77.4%)
.
(29.4%)
Therefore
(28.4%)
,
(29%)
the
(42.7%)
answer
(42.3%)
should
(42.7%)
be
(42.7%)
"
(45%)
even
(35.7%)
.
(54.2%)
</think>
(100%)
(100%)
0.6
0.8
1.0
Accuracy
Parity
CoT Beginning
CoT End
<think>
(100%)
(100%)
Okay
(100%)
,
(100%)
let
(97.8%)
's
(97.8%)
see
(90.8%)
.
(90.8%)
The
(90%)
question
(90%)
is
(90%)
asking
(59.8%)
where
(26%)
a
(14.2%)
would
(5%)
a
(6%)
a
(9.2%)
.
(12.6%)
So
(11.2%)
.
(20.4%)
So
(17.2%)
the
(13.2%)
the
(18%)
answer
(23.8%)
is
(25.4%)
C
(10.8%)
.
(100%)
</think>
(100%)
(100%)
0.25
0.50
0.75
Accuracy
CSQA
CoT Beginning
CoT End
<think>
(100%)
(100%)
Okay
(100%)
,
(100%)
let
(93%)
's
(92.2%)
see
(74.6%)
.
(74.6%)
The
(69.8%)
question
(67.6%)
is
(68.8%)
about
(49.2%)
the
(17.8%)
the
(12.4%)
of
(7%)
of
(8.2%)
the
(5.8%)
the
(8.6%)
.
(15.8%)
.
(23.4%)
So
(18.2%)
the
(23.4%)
answer
(20.8%)
answer
(25.6%)
is
(30.2%)
C
(15.2%)
.
(99.6%)
</think>
(100%)
(100%)
0.25
0.50
0.75
Accuracy
MMLU
CoT Beginning
CoT End
Figure 3. Average final-answer probing accuracy on three tasks with Off-the-Shelf LLM (Qwen3-32B) along CoT positions. The most
frequent token at each position is annotated with its occurrence frequency. The notably earlier accuracy spikes are especially pronounced
for Knowledge and Semantic tasks, but largely remain flat for Compositional tasks. The full results across all tasks are shown in Figure 17
for Off-the-Shelf LLM and Figure 18 for In-Domain LLM (Appendix C) .
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
MATH
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
Zebra
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
CSQA
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
MMLU
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
QuALITY
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
GPQA
Figure 4. Task accuracy comparison for Off-the-Shelf LLM (Qwen3-32B) under four settings: using thinking mode (w/ CoT); using
non-thinking mode (w/o CoT); the best probing accuracy among initial CoT positions (Probing); the random-guess baseline (Random).
The coarse signals of early final-answer planning are shown inferior to the direct prediction counterpart without CoT involved. Full results
across all tasks are provided in Figure 19. Similar comparisons for In-Domain LLM is provided in Figure 20.
than both standard reasoning with CoT and direct answering
without CoT. Therefore, even with comparable reasoning
budgets, early planning remains less effective than direct an-
swering. The performance gap further widens substantially
when CoT is applied, strongly indicating that such coarse
signals do not arise from precise plans in latent space.
2.5.2. PLANNING FOR REASONING PATH
Empirical results on probing subsequent tokens further ad-
vocate a myopic planning horizon of LLMs detailed below.
▶LLM hidden states generally encode limited fore-
sight over subsequent reasoning paths.
For each hidden state, we assess subsequent token predic-
tion performance up to its 8-th following token along the
CoT trajectory. As LLM generation is a sampling process
over a latent distribution, we measure by Top-5 Accuracy,
deeming a prediction correct if the true subsequent token
appears within the top-5 predictions. Figure 5 presents the
evaluation results with In-Domain LLM, which show a clear
1
2
3
4
5
6
7
8
Subsequent Position
0.2
0.4
0.6
0.8
1.0
Accuracy@Top5
Parity
Cycle
GSM8K
MATH
Zebra
CSQA
MMLU
GPQA
Figure 5. Top-5 accuracy for subsequent token prediction, using
the last Transformers layer of In-Domain LLM. Full results across
layers and tasks are presented in Figure 21 and Figure 22.
overall decline in accuracy as the subsequent token position
advances, specially for tasks dominated by semantic under-
standing and factual knowledge (e.g., MMLU and GPQA).
Figure 5 also suggests that LLM does plan the subsequent
path to a certain extent, with Top-5 accuracy exceeding
50% for the next two steps. However, a more long-term
planning is only limited to tasks with structural modularity,
such as Parity or Cycle, whose reasoning trajectories exhibit
discernible patterns (see the CoT example in Figure 10). In
6

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
300
600
900
1200
1500
300
600
900
1200
1500
Parity
300
600
900
1200
1500
300
600
900
1200
1500
Cycle
50
100
150
200
250
50
100
150
200
250
CSQA
50
100
150
200
250
50
100
150
200
250
QuALITY
0
1
2
3
4
5
6
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
0
1
2
3
4
5
6
Figure 6. Heatmap of the predicted reasoning length (y-axis) using initial CoT hidden states against the actual reasoning length (x-axis).
The rather unreliable predictions on those tasks without structural solutions (e.g. CSQA) suggest that a precise global plan does not
emerge early in CoT, even for the task-aware In-Domain LLM. Full results across tasks are provided in Figure 23 and 24.
300
600
900
1200 1500
300
600
900
1200
1500
0
1
2
3
4
5
6
(a) Reasoning length predic-
tions for Parity.
20
40
60
80
100
20
40
60
80
100
0
2
4
6
8
(b) Input sequence length pre-
dictions for Parity.
Figure 7. Task-specific factors can confound reasoning length
predictions. For Parity, the total length is typically proportional to
the input sequence length, which can be perceivable by LLMs.
general, hidden states lack a clear vision over subsequent
reasoning. Beyond In-Domain LLM as an “upper bound”, a
similar trend is also observed for Off-the-Shelf LLM, albeit
with much lower accuracy across all tasks, especially with
significant drop on structural tasks, as illustrated by the
comparison between Figure 21 and Figure 22.
2.5.3. PLANNING FOR GLOBAL STEPS
Reasoning length probing again indicates a lack of global
planning prior to the emergence of CoT, as discussed below.
▶LLMs do not grasp the reasoning length precisely,
though task-specific heuristics may offer shortcuts.
In general, if LLMs possessed a global reasoning view in
sight, early hidden states would be predictive of the total
length across input domains. However, our empirical re-
sults suggest that the initial CoT hidden states hardly have
reliable internal clock for global reasoning length, for both
In-Domain and Off-the-Shelf LLMs, as illustrated by the un-
stable and often low correlations across most tasks, shown
by the heatmaps in Figure 6 (full plots in Figure 23 and 24).
On closer inspection, two tasks appear to be exceptions, Par-
ity and Subsum, which exhibit high correlation with the true
reasoning lengths, as in Figure 23. However, interpreting
this as evidence of robust CoT planning on these tasks can
be misleading. We highlight the attribution of task-specific
confounding factors, illustrated in Figure 7: for both tasks,
reasoning paths are typically in proportional to the input
sequence length, which is readily observable by LLMs and
thus could serve as a shortcut signal in probing. In contrast,
for Cycle as in Figure 6, such shortcut does not apply, as its
reasoning length scales with the path between two vertices
rather than the input length (example in Figure 11), which
LLMs have difficulty estimating directly. The discrepancy
between Parity/Subsum and Cycle further underscores the
limited presence of actual global planning in LLMs.
3. Leveraging CoT Dynamics
Given the myopic planning horizon observed in our prob-
ing experiments, we highlight the significance of exploiting
such CoT dynamics, and demonstrate how these planning
characteristics can be leveraged to estimate both the uncer-
tainty and the necessity of CoT.
3.1. CoT Uncertainty Estimation
For language models, general metrics such as perplexity or
entropy are standard to estimate the inference confidence.
A well-calibrated uncertainty metric should ideally assign
high scores to correct outputs and lower scores to uncertain
ones. In our studies, we target metrics that utilize inter-
nal signals within CoT trajectories, focusing on the three
general uncertainty metrics described below.
(i) Perplexity, equivalent to the average Negative Log-
Likelihood of the sequence (NLL) for uncertainty. For a
sequence X with N tokens {x1, x2, .., xN}, NLL is:
NLL(X) = −1
N
N
X
i=1
log P(xi | x<i)
(3)
(ii) Average entropy H across tokens in the sequence, with
V being the model vocabulary:
H(X) = 1
N
N
X
i=1
 
−
X
w∈V
P(w|x<i) log P(w|x<i)
!
(4)
(iii) Self-Certainty (SC) (Kang et al., 2025), defined on the
7

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
vocabulary distribution as below:
SC(X) =
−1
N|V|
N
X
i=1
X
w∈V
log (|V| · P (w | x<i))
(5)
Note that the three metrics all aggregate over entire CoT
positions in the sequence. Intuitively, however, tokens that
steer the reasoning process within CoT are often sparse;
the majority of tokens function as “syntactic fillers” neces-
sary for linguistic coherence. These filler tokens are usually
high-confidence local transitions, as evidenced by the den-
sity distributions presented in Figure 12, aligned with prior
findings that most tokens in LLM are of low entropy (Wang
et al., 2025b; Li et al., 2026b).
Wooden Barrel Principle
Building on the local planning
cues in Section 2, we speculate that the internal signals
localized around a few critical tokens are more informa-
tive than trajectory-wide aggregates, where a conventional
global averaging across all generated tokens may dilute the
sensitivity of the uncertainty estimation. We thus posit such
principle: just like a barrel’s capacity is determined by its
shortest stave, analogously, we hypothesize that the uncer-
tainty of a reasoning chain is governed a subset of critical
logical leaps, which we term reasoning pivots.
We conduct empirical validation and demonstrate that fo-
cusing on these pivot positions, even through a simple top-k
selection strategy, could yield cleaner signals for uncertainty
calibration. Our validation utilizes two orthogonal sources
of latent signals, as described below.
Latent Signals by Tele-Lens
Before we proceed with
general metrics of uncertainty estimation, we first demon-
strate that latent signals from a sparse subset of tokens are
indeed effective to characterize the uncertainty of the whole
reasoning trajectory. Motivated by Figure 2, where spe-
cific positions exhibit significant accuracy spikes during
final-answer probing, we propose utilizing the entropy from
Tele-Lens to identify pivot positions: along a CoT path, we
select top-k positions with the lowest final-answer entropy,
as a proxy to indicate the confidence level of the entire path.
Accordingly, we conduct preliminary experiments with In-
Domain LLM using the last Transformers layer: after the
top-k positions are selected, we obtain their average of final-
answer entropy as a new uncertainty metric. The results,
measured by standard AUROC, are presented in Table 2.
Comparing against conventional baselines over the full path,
our top-k selection strategy upon Tele-Lens signals achieves
up to 9% absolute improvement upon the best baseline.
Notably, the best estimation is obtained with k = 5 pivot
tokens, demonstrating that latent signals from only a few
positions can be a strong indicator of the whole path.
Table 2. Uncertainty estimation results (AUROC) with In-Domain
LLM, using latent signals from final-answer probing via Tele-
Lens (Section 3.1); values closer to 1 indicate better calibration.
Using a subset of 5 positions along the CoT can better capture the
uncertainty of the full path, with 9% substantial improvement over
the best baseline. Full results across all tasks are shown in Table 6.
GSM8K
Zebra
MMLU
GPQA
Avg.
Perplexity
0.70
0.58
0.53
0.50
0.57
Entropy
0.72
0.60
0.52
0.50
0.58
Self-Certainty
0.76
0.67
0.53
0.51
0.60
Tele-Lens (Top-5)
0.87
0.77
0.73
0.56
0.69
Tele-Lens (Top-10)
0.81
0.75
0.72
0.56
0.68
Tele-Lens (Top-20)
0.82
0.67
0.65
0.51
0.63
Tele-Lens (Top-50)
0.78
0.69
0.56
0.47
0.64
Latent Signals by General Metrics
We next extend our
validation to general scenarios without involving signals
from a dedicated prober, lifting the constraint of a fixed
answer space. We consider the three general metrics derived
solely from predicted next-token logits over the model’s
vocabulary. For the generalizability of our findings, we
conduct experiments with Off-the-Shelf LLMs, using both
Qwen3-8B and Qwen3-32B. Specifically, we select the top-
k positions along a thinking path with the highest entropy /
self-certainty, or with the lowest log-likelihood, respectively,
representing the most uncertain local steps (the shortest
staves). We use the average among these positions of each
corresponding metric as the final estimation.
As shown in Table 3, for each LLM, applying the top-k
selection brings no negative impact using the selected k
values. The improvement is especially pronounced with
Qwen3-32B: k = 100 consistently drives 3+% absolute
improvement across all metrics, reaching up to 6%, thereby
supporting the efficacy of our hypothesis.
Furthermore, we highlight the potential of exploiting richer
latent signals and strategies for more effective calibration.
On the one hand, the signals by Tele-Lens are highly infor-
mative, suggesting that additional training can be beneficial,
consistent with prior findings that raw confidence metrics
from LLMs alone may be sub-optimal (Kapoor et al., 2024).
On the other hand, the spatial distributions of the selected
pivots differ substantially between signals derived from Tele-
Lens and those based on general metrics, as illustrated in
Figure 8. These divergent distributions indicate that latent
signals from multiple sources may provide complementary
information, and their integration can be promising to fur-
ther improve the identification of critical positions, leading
to more robust uncertainty calibration.
3.2. CoT Necessity Estimation
We next conduct a study to estimate the necessity of CoT
reasoning. In the broader context of improving model think-
ing efficiency, prior work has proposed various approaches
8

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
Table 3. Uncertainty estimation results (AUROC) with Qwen3-32B using the last Transformers layer, applying our top-k strategy upon
each general metric. Note that the average CoT length across inputs exceeds 7K tokens, while our simple strategy that selects top-100
positions is able to yield steady improvement. Full results across tasks with both 8B and 32B models are provided in Table 7.
GSM8K
MATH
MuSR
Zebra
CSQA
MMLU
QuALITY
GPQA
Avg.
Perplexity
0.71
0.93
0.48
0.74
0.68
0.76
0.78
0.69
0.72
w/ 100 Pivots
0.81
0.92
0.50
0.90
0.74
0.81
0.82
0.73
0.78
Entropy
0.71
0.92
0.47
0.77
0.68
0.77
0.77
0.68
0.72
w/ 100 Pivots
0.81
0.70
0.49
0.90
0.74
0.83
0.82
0.74
0.75
Self-Certainty
0.45
0.82
0.47
0.92
0.51
0.67
0.64
0.68
0.65
w/ 100 Pivots
0.55
0.90
0.47
0.93
0.59
0.74
0.70
0.70
0.70
0.0
0.2
0.4
0.6
0.8
1.0
Percentile
GSM8K
MATH
Zebra
CSQA
MMLU
LM Entropy
Tele-Lens
Figure 8. Spatial density distribution of selected pivot positions
with In-Domain LLM along CoT paths. Using Tele-Lens, the se-
lected positions tend to concentrate near CoT completion, whereas
positions selected by general LM entropy typically are distributed
across the entire CoT trajectory. Integrating multiple sources of
latent signals may spur further improvement.
for reducing CoT overhead, such as early exiting (Yong
et al., 2025; Yang et al., 2026) and CoT length steering (Li
et al., 2026a). In this study, we investigate from a different
perspective: assessing the input complexity to determine
whether CoT reasoning is necessary in the first place.
Motivated by the observations in Figure 3, we leverage the
early answer gist revealed by final-answer probing, where
the probing accuracy may spike at the initial CoT positions
for easier tasks. Our experiments show that these early
signals can be effective recognizing whether a full CoT
generation is required to accurately derive the final answer.
By selectively bypassing CoT generation in non-essential
cases, we can achieve a reduction in computational load
with negligible performance degradation.
For each rollout specifically, we first generate its initial five
CoT tokens and assess the normalized final-answer entropy
¯H over logit distribution p across C = 20 probing classes:
¯H(p) = (−
C
X
i=1
pi log pi)/ log C
(6)
As ¯H lies in the range [0, 1], we adopt a threshold-based
strategy: for initial positions, if any of their normalized
entropy falls below a predefined threshold, representing
a confident answer gist, we halt the corresponding CoT
Table 4. Evaluation results for CoT bypass, varying thresholds
of normalized entropy from final-answer probing. The bypass
ratio for each task is reported. Avg.: average bypass ratio; Perf.:
average accuracy change. Full results are provided in Table 8.
Parity
CSQA
MMLU
GPQA
Avg.
Perf.
In-Domain LLM
Th=0.1
0%
40.2%
30.4%
7%
13.3%
-0.47
Th=0.2
0%
65%
45%
12%
21.6%
-1.42
Off-the-Shelf LLM (Qwen3-32B)
Th=0.1
0%
16.2%
12.4%
1.2%
2.8%
-0.03
Th=0.2
0%
28.8%
20.2%
3.2%
6.2%
-0.37
generation and directly output the answer by disabling the
LLM’s thinking mode, bypassing a full generation. The
evaluation results are reported in Table 4.
With the threshold set to 0.1, our objective is robustly accom-
plished for both In-Domain and Off-the-Shelf LLMs: the
aforementioned heuristic automatically recognizes inputs
for which CoT is necessary to derive the final answer, such
as Parity, while bypassing CoT generation on easier tasks,
such as CSQA. For instance, Qwen3-32B attains 16.2% /
12.4% thinking reduction for CSQA / MMLU almost “for
free”, with only 0.03% overall accuracy degradation.
As our necessity estimation relies on a fixed answer space
to obtain the metric, we present it primarily as a proof-of-
concept. Nevertheless, we underscore the significance of ex-
ploiting such useful latent signals, which can benefit various
scenarios, such as to facilitate model training (Huang et al.,
2025) and CoT compression (Singh & Hakkani-T¨ur, 2026;
Li et al., 2026b) and steering (Li et al., 2026a). We provide
further discussions and Related Works in Appendix E.
4. Conclusion
In this work, we investigate the internal planning capacity
of LLMs and uncover a myopic planning horizon during
CoT generation. For analysis, we design a series of probing
experiments using our proposed method, Tele-Lens, and we
highlight the exploitation of such latent signals, demonstrat-
ing on both CoT uncertainty and necessity estimation.
9

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
Impact Statement
This paper presents work whose goal is to advance the under-
standing of internal dynamics of Large Language Models,
in particular the latent planning horizon and the according
utilization. There may be potential societal consequences
of this work, none which we feel must be specifically high-
lighted here.
References
Abbe, E., Bengio, S., Lotfi, A., Sandon, C., and Saremi, O.
How far can transformers reason? the globality barrier
and inductive scratchpad. In The Thirty-eighth Annual
Conference on Neural Information Processing Systems,
2024. URL https://openreview.net/forum?
id=FoGwiFXzuN.
Afzal, A., Matthes, F., Chechik, G., and Ziser, Y. Knowing
before saying: LLM representations encode information
about chain-of-thought success before completion. In
Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T.
(eds.), Findings of the Association for Computational
Linguistics: ACL 2025, pp. 12791–12806, Vienna, Aus-
tria, July 2025. Association for Computational Linguis-
tics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.
findings-acl.662.
URL https://aclanthology.
org/2025.findings-acl.662/.
AIME.
AIME problems and solutions, 2025.
URL
https://artofproblemsolving.com/wiki/
index.php/AIME_Problems_and_Solutions.
Anil, C., Wu, Y., Andreassen, A. J., Lewkowycz, A., Misra,
V., Ramasesh, V. V., Slone, A., Gur-Ari, G., Dyer, E.,
and Neyshabur, B. Exploring length generalization in
large language models.
In Oh, A. H., Agarwal, A.,
Belgrave, D., and Cho, K. (eds.), Advances in Neural
Information Processing Systems, 2022. URL https:
//openreview.net/forum?id=zSkYVeX7bC4.
Azaria, A. and Mitchell, T. The internal state of an LLM
knows when it’s lying. In Bouamor, H., Pino, J., and
Bali, K. (eds.), Findings of the Association for Compu-
tational Linguistics: EMNLP 2023, pp. 967–976, Singa-
pore, December 2023. Association for Computational
Linguistics.
doi:
10.18653/v1/2023.findings-emnlp.
68.
URL https://aclanthology.org/2023.
findings-emnlp.68/.
Bakman, Y. F., Yaldiz, D. N., Kang, S., Zhang, T.,
Buyukates, B., Avestimehr, S., and Karimireddy, S. P.
Reconsidering LLM uncertainty estimation methods in
the wild. In Che, W., Nabende, J., Shutova, E., and
Pilehvar, M. T. (eds.), Proceedings of the 63rd Annual
Meeting of the Association for Computational Linguis-
tics (Volume 1: Long Papers), pp. 29531–29556, Vienna,
Austria, July 2025. Association for Computational Lin-
guistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/
2025.acl-long.1429. URL https://aclanthology.
org/2025.acl-long.1429/.
Belrose, N., Ostrovsky, I., McKinney, L., Furman, Z., Smith,
L., Halawi, D., Biderman, S., and Steinhardt, J. Elicit-
ing latent predictions from transformers with the tuned
lens, 2023. URL https://arxiv.org/abs/2303.
08112.
Bhattamishra, S., Patel, A., Kanade, V., and Blunsom,
P. Simplicity bias in transformers and their ability to
learn sparse Boolean functions. In Rogers, A., Boyd-
Graber, J., and Okazaki, N. (eds.), Proceedings of the
61st Annual Meeting of the Association for Computa-
tional Linguistics (Volume 1: Long Papers), pp. 5767–
5791, Toronto, Canada, July 2023. Association for Com-
putational Linguistics. doi: 10.18653/v1/2023.acl-long.
317. URL https://aclanthology.org/2023.
acl-long.317/.
Bigelow, E. J., Holtzman, A., Tanaka, H., and Ullman, T.
Forking paths in neural text generation. In The Thirteenth
International Conference on Learning Representations,
2025. URL https://openreview.net/forum?
id=8RCmNLeeXx.
Chen, C., Liu, K., Chen, Z., Gu, Y., Wu, Y., Tao, M., Fu,
Z., and Ye, J.
INSIDE: LLMs’ internal states retain
the power of hallucination detection.
In The Twelfth
International Conference on Learning Representations,
2024. URL https://openreview.net/forum?
id=Zj12nzlQbz.
Chen, Z., Hu, W., and Hong, R. Deep hidden cognition facil-
itates reliable chain-of-thought reasoning, January 2026.
URL https://arxiv.org/abs/2507.10007.
Chiang, D. and Cholak, P. Overcoming a theoretical lim-
itation of self-attention. In Muresan, S., Nakov, P., and
Villavicencio, A. (eds.), Proceedings of the 60th Annual
Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers), pp. 7654–7664, Dublin, Ire-
land, May 2022. Association for Computational Linguis-
tics. doi: 10.18653/v1/2022.acl-long.527. URL https:
//aclanthology.org/2022.acl-long.527/.
Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H.,
Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano,
R., Hesse, C., and Schulman, J. Training verifiers to solve
math word problems, 2021. URL https://arxiv.
org/abs/2110.14168.
DeepSeek-AI. Deepseek-r1 incentivizes reasoning in llms
through reinforcement learning.
Nature, 645(8081):
633–638, September 2025.
ISSN 1476-4687.
doi:
10

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
10.1038/s41586-025-09422-z. URL http://dx.doi.
org/10.1038/s41586-025-09422-z.
Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. BERT:
Pre-training of deep bidirectional transformers for lan-
guage understanding. In Burstein, J., Doran, C., and
Solorio, T. (eds.), Proceedings of the 2019 Conference of
the North American Chapter of the Association for Com-
putational Linguistics: Human Language Technologies,
Volume 1 (Long and Short Papers), pp. 4171–4186, Min-
neapolis, Minnesota, June 2019. Association for Compu-
tational Linguistics. doi: 10.18653/v1/N19-1423. URL
https://aclanthology.org/N19-1423/.
Dong, Q., Li, L., Dai, D., Zheng, C., Ma, J., Li, R., Xia, H.,
Xu, J., Wu, Z., Chang, B., Sun, X., Li, L., and Sui, Z. A
survey on in-context learning. In Al-Onaizan, Y., Bansal,
M., and Chen, Y.-N. (eds.), Proceedings of the 2024 Con-
ference on Empirical Methods in Natural Language Pro-
cessing, pp. 1107–1128, Miami, Florida, USA, November
2024. Association for Computational Linguistics. doi:
10.18653/v1/2024.emnlp-main.64.
URL https://
aclanthology.org/2024.emnlp-main.64/.
Dong, Z., Zhou, Z., Liu, Z., Yang, C., and Lu, C.
Emergent response planning in LLMs.
In Forty-
second International Conference on Machine Learning,
2025. URL https://openreview.net/forum?
id=Ce79P8ULPY.
Dziri, N., Lu, X., Sclar, M., Li, X. L., Jiang, L., Lin,
B. Y., Welleck, S., West, P., Bhagavatula, C., Bras, R. L.,
Hwang, J. D., Sanyal, S., Ren, X., Ettinger, A., Harchaoui,
Z., and Choi, Y. Faith and fate: Limits of transformers on
compositionality. In Thirty-seventh Conference on Neu-
ral Information Processing Systems, 2023. URL https:
//openreview.net/forum?id=Fkckkr3ya8.
Gar´ı Soler, A. and Apidianaki, M. Let’s play mono-poly:
BERT can reveal words’ polysemy level and partition-
ability into senses. Transactions of the Association for
Computational Linguistics, 9:825–844, 2021. doi: 10.
1162/tacl a 00400. URL https://aclanthology.
org/2021.tacl-1.50/.
Ghandeharioun, A., Caciularu, A., Pearce, A., Dixon, L.,
and Geva, M.
Patchscopes: A unifying framework
for inspecting hidden representations of language mod-
els. In Forty-first International Conference on Machine
Learning, 2024. URL https://openreview.net/
forum?id=5uwBzcn885.
Gottesman, D. and Geva, M. Estimating knowledge in large
language models without generating a single token. In
Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), Pro-
ceedings of the 2024 Conference on Empirical Methods
in Natural Language Processing, pp. 3994–4019, Miami,
Florida, USA, November 2024. Association for Compu-
tational Linguistics. doi: 10.18653/v1/2024.emnlp-main.
232. URL https://aclanthology.org/2024.
emnlp-main.232/.
Hahn, M. and Rofin, M. Why are sensitive functions hard for
transformers? In Ku, L.-W., Martins, A., and Srikumar,
V. (eds.), Proceedings of the 62nd Annual Meeting of
the Association for Computational Linguistics (Volume
1: Long Papers), pp. 14973–15008, Bangkok, Thailand,
August 2024. Association for Computational Linguistics.
doi: 10.18653/v1/2024.acl-long.800. URL https://
aclanthology.org/2024.acl-long.800/.
Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M.,
Song, D., and Steinhardt, J. Measuring massive multitask
language understanding. In International Conference
on Learning Representations, 2021a. URL https://
openreview.net/forum?id=d7KBjmI3GmQ.
Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart,
S., Tang, E., Song, D., and Steinhardt, J. Measuring
mathematical problem solving with the MATH dataset.
In Thirty-fifth Conference on Neural Information Process-
ing Systems Datasets and Benchmarks Track (Round 2),
2021b. URL https://openreview.net/forum?
id=7Bywt2mQsCe.
Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B.,
De Laroussilhe, Q., Gesmundo, A., Attariyan, M., and
Gelly, S. Parameter-efficient transfer learning for NLP.
In Chaudhuri, K. and Salakhutdinov, R. (eds.), Proceed-
ings of the 36th International Conference on Machine
Learning, volume 97 of Proceedings of Machine Learn-
ing Research, pp. 2790–2799. PMLR, 09–15 Jun 2019.
URL https://proceedings.mlr.press/v97/
houlsby19a.html.
Huang, C., Yan, S., Xie, L., Lin, B., Fan, S., Xin, Y., Cai,
D., Shen, C., and Ye, J. Enhancing chain-of-thought
reasoning with critical representation fine-tuning.
In
Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T.
(eds.), Proceedings of the 63rd Annual Meeting of the
Association for Computational Linguistics (Volume 1:
Long Papers), pp. 23173–23195, Vienna, Austria, July
2025. Association for Computational Linguistics. ISBN
979-8-89176-251-0.
doi: 10.18653/v1/2025.acl-long.
1129. URL https://aclanthology.org/2025.
acl-long.1129/.
Huang, H.-Y., Yang, Y., Zhang, Z., Lee, S., and Wu, Y. A
survey of uncertainty estimation in llms: Theory meets
practice, 2024. URL https://arxiv.org/abs/
2410.15326.
Kang, Z., Zhao, X., and Song, D.
Scalable best-of-n
selection for large language models via self-certainty.
11

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
In The Thirty-ninth Annual Conference on Neural In-
formation Processing Systems, 2025.
URL https:
//openreview.net/forum?id=29FRqmVQK8.
Kapoor, S., Gruver, N., Roberts, M., Collins, K. M., Pal,
A., Bhatt, U., Weller, A., Dooley, S., Goldblum, M., and
Wilson, A. G. Large language models must be taught to
know what they don’t know. In The Thirty-eighth Annual
Conference on Neural Information Processing Systems,
2024. URL https://openreview.net/forum?
id=QzvWyggrYB.
Li, K., Hopkins, A. K., Bau, D., Vi´egas, F., Pfister, H.,
and Wattenberg, M. Emergent world representations:
Exploring a sequence model trained on a synthetic task.
In The Eleventh International Conference on Learning
Representations, 2023. URL https://openreview.
net/forum?id=DeG07_TcZvT.
Li, Y., Tu, T., Ding, L., Wang, J., Zhen, H.-L., Chen, Y.,
Li, Y., and Tian, Z. Efficient reasoning with balanced
thinking. In The Fourteenth International Conference
on Learning Representations, 2026a. URL https://
openreview.net/forum?id=cJseWJJ5IM.
Li, Z., Liu, H., Zhou, D., and Ma, T. Chain of thought em-
powers transformers to solve inherently serial problems.
In The Twelfth International Conference on Learning
Representations, 2024. URL https://openreview.
net/forum?id=3EWTEy9MTM.
Li, Z., Zhong, J., Zheng, Z., Wen, X., Xu, Z., Cheng, Y.,
Zhang, F., and Xu, Q. Making slow thinking faster: Com-
pressing LLM chain-of-thought via step entropy. In The
Fourteenth International Conference on Learning Rep-
resentations, 2026b. URL https://openreview.
net/forum?id=cGLqQfS5wH.
Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker,
B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and
Cobbe, K. Let’s verify step by step. In The Twelfth
International Conference on Learning Representations,
2024. URL https://openreview.net/forum?
id=v8L0pN6EOi.
Lin, B. Y., Bras, R. L., Richardson, K., Sabharwal, A.,
Poovendran, R., Clark, P., and Choi, Y. Zebralogic: On
the scaling limits of LLMs for logical reasoning. In Forty-
second International Conference on Machine Learning,
2025. URL https://openreview.net/forum?
id=sTAJ9QyA6l.
Liu, R., Geng, J., Wu, A. J., Sucholutsky, I., Lombrozo, T.,
and Griffiths, T. L. Mind your step (by step): Chain-of-
thought can reduce performance on tasks where think-
ing makes humans worse. In Forty-second International
Conference on Machine Learning, 2025. URL https:
//openreview.net/forum?id=J3gzdbYZxS.
Liu, X., Fatahi Bayat, F., and Wang, L. Enhancing lan-
guage model factuality via activation-based confidence
calibration and guided decoding.
In Al-Onaizan, Y.,
Bansal, M., and Chen, Y.-N. (eds.), Proceedings of
the 2024 Conference on Empirical Methods in Natu-
ral Language Processing, pp. 10436–10448, Miami,
Florida, USA, November 2024. Association for Compu-
tational Linguistics. doi: 10.18653/v1/2024.emnlp-main.
583. URL https://aclanthology.org/2024.
emnlp-main.583/.
Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy,
O., Lewis, M., Zettlemoyer, L., and Stoyanov, V. Roberta:
A robustly optimized bert pretraining approach, 2019.
URL https://arxiv.org/abs/1907.11692.
Merrill, W. and Sabharwal, A. The parallelism tradeoff:
Limitations of log-precision transformers. Transactions
of the Association for Computational Linguistics, 11:531–
545, 2023. doi: 10.1162/tacl a 00562. URL https:
//aclanthology.org/2023.tacl-1.31/.
nostalgebraist.
logit
lens
on
non-gpt2
mod-
els
+
extensions,
2021.
URL
https:
//colab.research.google.com/drive/
1MjdfK2srcerLrAJDRaJQKO0sUiZ-hQtA.
Nye, M., Andreassen, A. J., Gur-Ari, G., Michalewski, H.,
Austin, J., Bieber, D., Dohan, D., Lewkowycz, A., Bosma,
M., Luan, D., Sutton, C., and Odena, A. Show your
work: Scratchpads for intermediate computation with
language models, 2021. URL https://arxiv.org/
abs/2112.00114.
Pal, K., Sun, J., Yuan, A., Wallace, B., and Bau, D. Fu-
ture lens: Anticipating subsequent tokens from a sin-
gle hidden state. In Jiang, J., Reitter, D., and Deng, S.
(eds.), Proceedings of the 27th Conference on Compu-
tational Natural Language Learning (CoNLL), pp. 548–
560, Singapore, December 2023. Association for Com-
putational Linguistics. doi: 10.18653/v1/2023.conll-1.
37.
URL https://aclanthology.org/2023.
conll-1.37/.
Pang, R. Y., Parrish, A., Joshi, N., Nangia, N., Phang, J.,
Chen, A., Padmakumar, V., Ma, J., Thompson, J., He,
H., and Bowman, S. QuALITY: Question answering
with long input texts, yes!
In Carpuat, M., de Marn-
effe, M.-C., and Meza Ruiz, I. V. (eds.), Proceedings
of the 2022 Conference of the North American Chap-
ter of the Association for Computational Linguistics:
Human Language Technologies, pp. 5336–5358, Seat-
tle, United States, July 2022. Association for Computa-
tional Linguistics. doi: 10.18653/v1/2022.naacl-main.
391. URL https://aclanthology.org/2022.
naacl-main.391/.
12

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
Patel, R. and Pavlick, E. Mapping language models to
grounded conceptual spaces. In International Conference
on Learning Representations, 2022. URL https://
openreview.net/forum?id=gJcEM8sxHK.
Reif, E., Yuan, A., Wattenberg, M., Viegas, F. B., Coenen,
A., Pearce, A., and Kim, B. Visualizing and measuring
the geometry of bert.
In Wallach, H., Larochelle,
H., Beygelzimer, A., d'Alch´e-Buc, F., Fox, E., and
Garnett, R. (eds.), Advances in Neural Information
Processing Systems, volume 32. Curran Associates, Inc.,
2019.
URL https://proceedings.neurips.
cc/paper_files/paper/2019/file/
159c1ffe5b61b41b3c4d8f4c2150f6c4-Paper.
pdf.
Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y.,
Dirani, J., Michael, J., and Bowman, S. R. GPQA: A
graduate-level google-proof q&a benchmark. In First
Conference on Language Modeling, 2024. URL https:
//openreview.net/forum?id=Ti67584b98.
Sahoo, P., Singh, A. K., Saha, S., Jain, V., Mondal, S.,
and Chadha, A. A systematic survey of prompt engi-
neering in large language models: Techniques and ap-
plications, 2025. URL https://arxiv.org/abs/
2402.07927.
Shao, C., Li, D., Meng, F., and Zhou, J.
Continuous
autoregressive language models, 2025. URL https:
//arxiv.org/abs/2510.27688.
Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X.,
Zhang, H., Zhang, M., Li, Y. K., Wu, Y., and Guo,
D.
Deepseekmath: Pushing the limits of mathemat-
ical reasoning in open language models, 2024. URL
https://arxiv.org/abs/2402.03300.
Sheng, L., Zhang, A., Wu, Z., Zhao, W., Shen, C., Zhang,
Y., Wang, X., and Chua, T.-S. On reasoning strength
planning in large reasoning models. In The Thirty-ninth
Annual Conference on Neural Information Processing
Systems, 2025. URL https://openreview.net/
forum?id=H26A7cl91u.
Singh, J. and Hakkani-T¨ur, D. Do llms encode functional
importance of reasoning tokens?, 2026. URL https:
//arxiv.org/abs/2601.03066.
Skean, O., Arefin, M. R., Zhao, D., Patel, N. N., Naghiyev,
J., LeCun, Y., and Shwartz-Ziv, R. Layer by layer: Uncov-
ering hidden representations in language models. In Forty-
second International Conference on Machine Learning,
2025. URL https://openreview.net/forum?
id=WGXb7UdvTX.
Sprague, Z. R., Ye, X., Bostrom, K., Chaudhuri, S., and
Durrett, G.
MuSR: Testing the limits of chain-of-
thought with multistep soft reasoning. In The Twelfth
International Conference on Learning Representations,
2024. URL https://openreview.net/forum?
id=jenyYQzue1.
Sprague, Z. R., Yin, F., Rodriguez, J. D., Jiang, D., Wadhwa,
M., Singhal, P., Zhao, X., Ye, X., Mahowald, K., and
Durrett, G. To cot or not to cot? chain-of-thought helps
mainly on math and symbolic reasoning. In The Thir-
teenth International Conference on Learning Represen-
tations, 2025. URL https://openreview.net/
forum?id=w6nlcS8Kkn.
Talmor, A., Herzig, J., Lourie, N., and Berant, J. Com-
monsenseQA: A question answering challenge targeting
commonsense knowledge. In Burstein, J., Doran, C., and
Solorio, T. (eds.), Proceedings of the 2019 Conference of
the North American Chapter of the Association for Com-
putational Linguistics: Human Language Technologies,
Volume 1 (Long and Short Papers), pp. 4149–4158. Asso-
ciation for Computational Linguistics, June 2019. URL
https://aclanthology.org/N19-1421/.
Ton, J.-F., Taufiq, M. F., and Liu, Y. Understanding chain-
of-thought in LLMs through information theory. In Forty-
second International Conference on Machine Learning,
2025. URL https://openreview.net/forum?
id=IjOWms0hrf.
Wang, E. Z., Cassano, F., Wu, C., Bai, Y., Song, W., Nath,
V., Han, Z., Hendryx, S. M., Yue, S., and Zhang, H.
Planning in natural language improves LLM search for
code generation. In The Thirteenth International Confer-
ence on Learning Representations, 2025a. URL https:
//openreview.net/forum?id=48WAZhwHHw.
Wang, J., Peng, H., and Liu, C. Latent chain-of-thought as
planning: Decoupling reasoning from verbalization, 2026.
URL https://arxiv.org/abs/2601.21358.
Wang, L., Xu, W., Lan, Y., Hu, Z., Lan, Y., Lee, R. K.-
W., and Lim, E.-P. Plan-and-solve prompting: Improv-
ing zero-shot chain-of-thought reasoning by large lan-
guage models.
In Rogers, A., Boyd-Graber, J., and
Okazaki, N. (eds.), Proceedings of the 61st Annual Meet-
ing of the Association for Computational Linguistics (Vol-
ume 1: Long Papers), pp. 2609–2634, Toronto, Canada,
July 2023. Association for Computational Linguistics.
doi: 10.18653/v1/2023.acl-long.147.
URL https:
//aclanthology.org/2023.acl-long.147/.
Wang, S., Yu, L., Gao, C., Zheng, C., Liu, S., Lu, R., Dang,
K., Chen, X.-H., Yang, J., Zhang, Z., Liu, Y., Yang, A.,
Zhao, A., Yue, Y., Song, S., Yu, B., Huang, G., and Lin,
J. Beyond the 80/20 rule: High-entropy minority tokens
13

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
drive effective reinforcement learning for LLM reason-
ing. In The Thirty-ninth Annual Conference on Neural
Information Processing Systems, 2025b. URL https:
//openreview.net/forum?id=yfcpdY4gMP.
Wei, J., Wang, X., Schuurmans, D., Bosma, M., brian ichter,
Xia, F., Chi, E. H., Le, Q. V., and Zhou, D. Chain of
thought prompting elicits reasoning in large language
models. In Oh, A. H., Agarwal, A., Belgrave, D., and Cho,
K. (eds.), Advances in Neural Information Processing
Systems, 2022. URL https://openreview.net/
forum?id=_VjQlMeSB_J.
Wies, N., Levine, Y., and Shashua, A. Sub-task decom-
position enables learning in sequence to sequence tasks.
In The Eleventh International Conference on Learning
Representations, 2023. URL https://openreview.
net/forum?id=BrJATVZDWEH.
Xiao, C. and Liu, B. Generalizing reasoning problems to
longer lengths. In The Thirteenth International Confer-
ence on Learning Representations, 2025. URL https:
//openreview.net/forum?id=zpENPcQSj1.
Yang, C., Si, Q., Duan, Y., Zhu, Z., Zhu, C., Li, Q., Chen, M.,
Lin, Z., and Wang, W. Dynamic early exit in reasoning
models.
In The Fourteenth International Conference
on Learning Representations, 2026. URL https://
openreview.net/forum?id=NpU7ZXafRi.
Yong, X., Zhou, X., Zhang, Y., Li, J., Zheng, Y., and
Wu, X.
Think or not? exploring thinking efficiency
in large reasoning models via an information-theoretic
lens. In The Thirty-ninth Annual Conference on Neural
Information Processing Systems, 2025. URL https:
//openreview.net/forum?id=DpOSndSOZz.
Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y.,
Dai, W., Fan, T., Liu, G., Liu, L., Liu, X., Lin, H., Lin,
Z., Ma, B., Sheng, G., Tong, Y., Zhang, C., Zhang, M.,
Zhang, W., Zhu, H., Zhu, J., Chen, J., Chen, J., Wang,
C., Yu, H., Song, Y., Wei, X., Zhou, H., Liu, J., Ma, W.-
Y., Zhang, Y.-Q., Yan, L., Qiao, M., Wu, Y., and Wang,
M. Dapo: An open-source llm reinforcement learning
system at scale, 2025. URL https://arxiv.org/
abs/2503.14476.
Zhang, J., Sun, Y., Leng, T., Shen, J., Ziyin, L., Liang, P. P.,
and Zhang, H. When reasoning meets its laws, 2025.
URL https://arxiv.org/abs/2512.17901.
Zhou, Y., Muresanu, A. I., Han, Z., Paster, K., Pitis,
S., Chan, H., and Ba, J. Large language models are
human-level prompt engineers.
In The Eleventh In-
ternational Conference on Learning Representations,
2023. URL https://openreview.net/forum?
id=92gvk82DE-.
Zubic, N., Sold`a, F., Sulser, A., and Scaramuzza, D.
Limits of deep learning: Sequence modeling through
the lens of complexity theory.
In The Thirteenth In-
ternational Conference on Learning Representations,
2025. URL https://openreview.net/forum?
id=DhdqML3FdM.
14

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
A. Tasks and Datasets
As described in Section 2.2, our probing experiments span
12 diverse tasks of different types for a comprehensive view
of empirical insights. This section further provides concrete
examples, data processing details and statistics.
A.1. Task Examples
Task Example for: Parity
[Settings]:
• Sequence length:
41
• Target digit to count:
2
• Answer:
even
Determine whether the number of “2” in the following
digit sequence is even or odd; please output only your
decision by either “even” or “odd”.
91223822122515222430601862928242722242251
Task Example for: Cycle
[Settings]:
• Number of edges:
16
• Answer:
NO
Task
Given the following directed graph represented as a list of
edges (from vertex →to vertex), along with two target
vertices, you need to determine whether there exists a path
from the first target vertex to the second.
Edges
v453 →v561
v666 →v34
v34 →v791
v791 →v17
v416 →v0
v658 →v666
v0 →v74
v254 →v427
v427 →v520
v561 →v254
v74 →v453
v520 →v416
v664 →v464
v17 →v664
v640 →v658
v464 →v640
Target
v 34, v 561
Output
Please output only “YES” if a path exists, or “NO” if it
does not.
Task Example for: Subsum
[Settings]:
• Sequence length:
29
• Max subsequence sum:
84
• Answer:
4
Given the following sequence of numbers, determine the
least significant digit of the maximum sum of its subse-
quences, such that no two numbers in the subsequence
are adjacent in the original sequence. Please output only
the according least significant digit directly.
[2, 4, 6, 6, 1, 8, 5, 5, 4, 6, 6, 6, 6, 8, 1, 8, 9, 1, 9, 9, 4, 1, 9,
5, 4, 2, 4, 3, 2]
Task Example for: GSM8K (Multi-Choice)
[Settings]:
• Answer:
D
Task
Given the following problem along with its options, de-
termine the best option as the answer. Please only output
your selected answer option by the letter (e.g., A, B, C).
Problem
Rob, Royce, and Pedro are contractors getting ready to
put a new roof on three homes. If the three homes will
need 250 cases of shingles, with the first house needing
1/2 of the second, and the third needing double the first.
How many cases of shingles will the third house need?
Options:
A. 125
B. 200
C. 50
D. 100
E. 83
Task Example for: MATH (Multi-Choice)
[Settings]:
• Answer:
B
Task
Given the following problem along with its options, de-
termine the best option as the answer. Please only output
your selected answer option by the letter (e.g., A, B, C).
Problem
If each point of the circle x2 + y2 = 25 is reflected in the
point (4, 1), the set of image points satisfies the equation
x2 + ay2 + bx + cy + d = 0.
Compute the ordered quadruple (a, b, c, d) of real num-
bers.
Options:
A. (1,16,4,43)
15

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
B. (1,-16,-4,43)
C. (1,-8,-2,17)
D. (1,-16,4,43)
E. (1,-16,-4,-43)
Task Example for: AIME (Multi-Choice)
[Settings]:
• Answer:
B
Task
Given the following problem along with its options, de-
termine the best option as the answer. Please only output
your selected answer option by the letter (e.g., A, B, C).
Problem
The set of points in 3-dimensional coordinate space that
lie in the plane x + y + z = 75 whose coordinates satisfy
the inequalities x −yz < y −zx < z −xy forms three
disjoint convex regions. Exactly one of those regions has
finite area. The area of this finite region can be expressed
in the form a
√
b, where a and b are positive integers and
b is not divisible by the square of any prime. Find a + b.
Options:
A. 524
B. 510
C. 498
D. 504
E. 496
Task Example for: MuSR
[Settings]:
• Answer:
A
Task
Given the following article, along with a related question
and its answer options, please determine the best answer
option for this question.
Article
In my latest tenure at a bustling educational institution,
three staff members, Emily, Robert, and Alice, consis-
tently caught my attention amidst the sea of educators
and support personnel. As the school manager, my role
was to distribute tasks, specifically Teaching and Admin
work, in a way that capitalized on each individual’s unique
strengths, thereby streamlining the school’s operations.
These assignments, as crucial as they were intricate, were
akin to the individual notes in a symphony, each playing
a vital role in the harmony of the institution.
Alice was a unique blend of complexities, as I carefully
observed her interactions with the staff. Her proclivity for
administrative tasks was evident, a much-needed quality
in the heaving sea of paperwork the school generated. Al-
ice often took the responsibility of capturing the minutes
during our staff meetings and backed up Robert’s teach-
ings with her painstaking administrative work...
...
Question
Given the story, how would you uniquely allocate each
person to make sure both tasks are accomplished effi-
ciently?
A. Teaching: Emily, Admin work: Alice and Robert
B. Teaching: Alice, Admin work: Emily and Robert
C. Teaching: Robert, Admin work: Alice and Emily
Output
Please only output your selected answer option by
“A/B/C/...”.
Task Example for: Zebra
[Settings]:
• Answer:
A
Task
Given the following problem along with its options, de-
termine the best option as the answer. Please only output
your selected answer option by the letter (e.g., A, B, C).
Problem
There are 5 houses, numbered 1 to 5 from left to right, as
seen from across the street. Each house is occupied by
a different person. Each house has a unique attribute for
each of the following characteristics:
- Each person has a unique name: Peter, Alice, Arnold,
Bob, Eric
- Each person has a unique hobby: photography, cooking,
knitting, gardening, painting
- Each person has a unique favorite drink: root beer, milk,
water, coffee, tea
Rules:
1. Eric is the coffee drinker.
2. The tea drinker is the person who paints as a hobby.
3. The person who enjoys knitting is not in the fourth
house.
4. Peter is not in the fourth house.
5. Eric is somewhere to the right of the root beer lover.
6. Arnold is the person who loves cooking.
7. The one who only drinks water is somewhere to the
right of the person who enjoys gardening.
8. There is one house between Bob and the person who
paints as a hobby.
9. The person who enjoys gardening is directly left of the
root beer lover.
10. The photography enthusiast is the one who only drinks
water.
Question:
What is Drink of the person who lives in House 1?
A. milk
B. tea
C. root beer
D. coffee
E. water
16

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
Task Example for: CSQA
[Settings]:
• Answer:
B
Task
Given the following commonsense question, please deter-
mine its best answer option.
Question
The drought was dangerous for the trees, they were more
likely to what?
A. fire
B. burn
C. covered in snow
D. wall in
E. grow tall
Output
Please only output your selected answer option by
“A/B/C/...”.
Task Example for: MMLU
[Settings]:
• Answer:
D
Task
Given the following question and its options, determine
the best option as the answer. Please only output your
selected answer option by “A/B/C/D”.
Question
A manager’s competitor sent a defamatory letter to the
manager accusing him of professional incompetence and
calling him one of the worst businessmen in town. It was
addressed to the manager. He read it, put it in a private
drawer, and did not read it again. Later, he tried to sue the
competitor for defamation as a result of the letter. Will
the court likely grant the defendant’s motion to dismiss,
and on what grounds? Base your answer on the common
law definition of defamation.
A. No, it will not dismiss because a plaintiff in a defam-
atory action has an absolute right to a jury trial to prove
defamation.
B. Yes, it will dismiss on the basis that the language is not
damaging to the manager’s reputation.
C. No, it will not dismiss because the circumstances show
that all of the elements of defamation are all present.
D. Yes, it will dismiss on the basis that the publication is
made to the manager alone.
Task Example for: QuALITY
[Settings]:
• Answer:
B
Task
Given the following snippets from an article or a story,
along with a related question and its answer options, you
need to determine the best option based on the information
provided by these snippets.
These snippets may not necessarily always contain the
supported information to answer the question; in that case
try to give a best guess.
Snippets
For his earlier errors, Coleman had first received a sus-
pended sentence, then two terminal sentences to be fixed
by the warden. My predecessors had given him first a
few weeks, then a few months of sleep in Dreamland.
Coleman’s eyes didn’t frighten me; I focused right on the
pupils. “That was a pretty foul trick, Councilman. Did
you hope to somehow frighten me out of executing this
sentence by what you told me this morning?”
I couldn’t follow his reasoning. Just how making me think
my life was only a Dream such as I imposed on my own
prisoners could help him, I couldn’t see...
...
Question
What were Coleman’s motivations in visiting the warden?
A. Providing the warden with his annual raise announce-
ment
B. Scaring him into believing his life was a dream
C. Gathering information to bring down the warden’s
compound
D. Persuading the warden to step down from his position
Output
Please only output your selected answer option by
“A/B/C/D”.
Task Example for: GPQA
[Settings]:
• Answer:
B
Task
Given the following exam question, please determine its
best answer option.
Question
Observations of structures located at a distance of about
2.1 gigaparsecs (2.1 Gpc) are being carried out. The
detected absorption line energy equivalent is about 3.9
micro electron volts (3.9 ∗10−6 eV).
What is most likely to be observed with this absorption
line in the Milky Way?
A. Warm atomic interstellar medium.
B. Cold atomic interstellar medium.
C. Warm molecular interstellar medium.
D. Cold molecular interstellar medium.
Output
Please only output your selected answer option by
“A/B/C/D”.
17

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
A.2. Dataset Descriptions
A brief description of each existing dataset adopted in our
experiments is provided below.
Implicit Compositional Tasks
Three mathematical tasks
and two logical reasoning tasks are included:
• GSM8K: a dataset focusing on middle school-level prob-
lems of various difficulties (Cobbe et al., 2021).
• MATH: a dataset introduced by Hendrycks et al. (2021b)
with high school competition-level math problems. We
follow the MATH-500 test split by Lightman et al. (2024).
• AIME: 30 math competition problems from AIME’25
(2025 American Invitational Mathematics Examination)
(AIME, 2025).
• MuSR: a Multistep Soft Reasoning dataset to evaluate
logical deduction with natural language rules over long,
text-based narratives (Sprague et al., 2024).
• Zebra: the benchmark ZebraLogic (Lin et al., 2025)
designed to evaluate symbolic reasoning and constraint
satisfaction abilities within a natural language context.
Knowledge and Semantic Tasks
Four knowledge-
intensive benchmarks focusing on semantic understanding
rather than explicit reasoning are included:
• CSQA: CommonsenseQA (Talmor et al., 2019), targeting
commonsense reasoning based on world knowledge.
• MMLU: a broad-spectrum dataset to evaluate knowledge
from 57 subjects, covering various STEM and social sci-
ence domains (Hendrycks et al., 2021a).
• QuALITY: a narrative question answering dataset (Pang
et al., 2022). To reduce computational overhead, we
frame the context in the form of relevant snippets re-
trieved via dense retrieval (a RAG setting with a max 2K
context length), rather than using full documents.
• GPQA: a challenging dataset designed to test expert-level
knowledge of multiple domains, such as biology, physics,
chemistry, etc. (Rein et al., 2024).
A.3. Data Preparation For Existing Datasets
Among the 12 tasks used in our probing experiments, three
mathematics tasks (MATH, GSM8K, and AIME) are orig-
inally evaluated via free-form generation, without a fixed
answer space. To enable final-answer probing, we use the
following prompt to convert each problem into a multiple-
choice format using GPT-4.1.
Prompt for Multi-Choice Conversion
Task
Given the following problem and its correct answer so-
lution, you need to generate four plausible but incorrect
answer options, which will serve as misleading distractors
to construct multiple-choice questions.
Problem
{problem}
Solution [Optional]
{solution}
Correct Answer
{answer}
Output
Please first think about four misleading wrong answer
options, starting with “### Think”.
Then, starting with “### Options”, provide each option
per line, where each line is directly the according answer
option in a similar format of the correct answer, without
adding any prefix or explanation.
For other existing datasets originally in a multiple-choice
format, we also shuffle the order of options for each ques-
tion, intended to mitigate potential memorization effects
and positional bias by LLMs.
With all 12 tasks having a fixed answer space, the label set
for the final-answer probing consists of 20 tokens in total:
A, B, C, D, E, F, YES, NO, even, odd,
0, 1, 2, 3, 4, 5, 6, 7, 8, 9

A.4. Data Generation and Sampling
The data generation process for the three explicit composi-
tional tasks is fully controllable. For probing, we generate
problems and their corresponding labels for each task using
the following procedure.
Parity
We generate random digit sequences from length
5 to 100. For each sequence, the target digit to count is
randomly selected from {1, 2, 7, 8}. The label is then deter-
mined by the parity of its count.
Cycle
For each input, we first set the number of edges,
an even number randomly determined from 4 to 100. Two
instances are yielded at each time, one forming a full cy-
cle using all edges, and another forming two equal-sized
cycles each using half of the edges. We then randomly as-
sign vertex names from a pool of 1000 candidates to each
cycle, ensuring diversity in both vertex identities and edge
orderings. For the target vertices to determine whether there
exists a path in between, we randomly select two vertices
in the first case (there is always a path due to cycling), and
randomly select one vertex from each cycle for the second
case (no path exists).
Subsum
We generate random lists of integers (each in the
range from 1 to 9), with lengths ranging from 2 to 50. The
18

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
labels are obtained by applying the dynamic programming
solution to each list.
With our data generation process, the resulting labels for
each task are evenly distributed. Our code for data genera-
tion and the resulting datasets will be publicly released.
Data Sampling
For other tasks, we sample from their
original test sets, such as our test split always consists of
problems drawn from the original test sets. For our train/dev
splits, we keep sampling remaining problems from the test
sets. If a dataset does not contain a sufficient number of
test instances, we additionally sample from their original
training and dev sets, if they are available.
Note that AIME’25 contains only 30 problems in total. Fol-
lowing the above procedure, all 30 problems are included
in our test split, with no problems added to our train or dev
split for AIME.
A.5. Dataset Construction and Statistics
With our train/dev/test splits in place, we perform infer-
ence by each of the two LLM backbones, collecting their
corresponding rollouts and hidden states. We retain hidden
states of all tokens along CoT trajectories within a maximum
length of 16,384 for the test split. For the train/dev splits, to
maintain a manageable storage cost, we keep hidden states
of sampled 5% / 10% CoT tokens for Off-the-Shelf LLM
and In-Domain LLM respectively.
The resulting train/dev/test dataset has 2.4M / 81K / 11M
hidden states for Off-the-Shelf LLM, and 2.5M / 57K / 2.7M
hidden states for In-Domain LLM, respectively (for each
Transformers layer).
Each hidden state is labeled according to the corresponding
rollout outcomes for each teleological dimension, e.g. the
tokens ID for the i-th subsequent token, the token ID of
the final predicted answer, the total CoT length, etc. Given
the ample number of instances in our dataset, our Tele-
Lens adapters can automatically learn latent features that
discriminate among different labels.
A.6. Hyperparameters
The hidden size is d = 5120 for Off-the-Shelf LLM
(Qwen3-32B), and d = 3584 for In-Domain LLM that
is trained upon Qwen2.5-7B-Instruct. We use r = 256 for
both models, which is shown sufficient by prior works on
probing (Dong et al., 2025). For training, we adopt learning
rate 1 × 10−3, batch size 8000, a linear decay learning rate
schedule, and early stopping on dev set, with approximately
5000 max training steps; we do not enable weight decay or
warmup period. The training is conducted on Nvidia V100
GPUs, and only the parameters of Tele-Lens are updated;
the LM head is kept frozen during training.
B. In-Domain LLM
B.1. Training Details
As described in Section 2.3, we conduct reinforcement
learning with GRPO (Shao et al., 2024) on Qwen2.5-7B-
Instruct to obtain an In-domain LLM on the 12 tasks. Dur-
ing training, we add the format instruction in the LLM
system prompt, and use a reward signal based solely on
format validation and answer correctness (each with score
1). The training set comprises 48K problems, sampled from
original training sets of those existing datasets, as well as
auto-generated problems for explicit compositional tasks.
System Prompt for: In-Domain LLM
You are a helpful assistant.
Now the user asks
you to solve a reasoning problem.
You need to
first think about the solving process in the
mind and then provide the user with the answer.
The thinking process is enclosed within <think>
</think> tags, i.e., <think> thinking process
here </think> final answer.
For training, we use a rollout size 16, batch size 320 with
mini batch size 80, capping the max response length as
4096. We adopt a cosine learning rate schedule with initial
learning rate 1 × 10−6 and 10 warmup steps. Following
DAPO (Yu et al., 2025), we adopt the clip-higher strategy to
encourage exploration, setting the upper clip ratio as 0.3 and
the lower clip ratio as 0.2. The training converges within
800 steps, among which Parity is the slowest to converge.
B.2. Evaluation
Evaluation of our adopted LLM backbones across 12 tasks,
including our In-Domain LLM trained by reinforcement
learning, are provided in Table 5.
B.3. In-Domain CoT Examples
As shown in Table 5, our In-Domain LLM produces much
shorter CoT trajectories, indicating more stable and decisive
reasoning behavior. For qualitative comparison, we provide
examples for Parity using both Off-the-Shelf LLM (Qwen3-
32B) and In-Domain LLM, shown in Figure 9 and Figure 10.
Another example for Cycle is shown in Figure 11.
C. Probing Results
C.1. Collection of Full Results
Section 2.5 reports the empirical results of our probing
settings. Due to the limited space in the main pages, we
present figures and tables for full results across layers and
tasks, listed as below.
19

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
Table 5. Accuracy of LLM backbones on the 12 tasks spanning three categories described in Section 2.2, along with CoT length (measured
in number of characters), averaged from 5 repeated runs. For the off-the-shelf Qwen3 LLMs, we evaluate two settings, with thinking
mode disabled (w/o CoT) or enabled (w/ CoT), respectively. Our In-Domain LLM is trained by GRPO upon Qwen2.5-7B-Instruct (details
in Section 2.3), which is one generation behind the Qwen3 series, thus its performance may lag behind on certain datasets. Despite this,
it achieves the best performance on three compositional tasks while producing substantially shorter CoT trajectories. Note that in
our evaluation, the maximum CoT length is capped at 32,768 tokens; any response exceeding this limit is considered incorrect. Result
discussions are addressed in Section 2.5.
Explicit Comp.
Implicit Comp.
Knowledge & Semantics
Parity
Cycle
Subsum
GSM8K
MATH
AIME
MuSR
Zebra
CSQA
MMLU
QuALITY
GPQA
Avg.
Random
50
50
10
20
20
20
20
16.7
20
25
25
20
24.7
Qwen3-8B (w/o CoT)
49.6
45.4
0.6
28.0
26.9
3.3
58.4
43.8
76.8
70.6
87.0
35.0
43.8
Qwen3-8B (w/ CoT)
76.5
92.8
89.1
97.0
98.3
76.8
65.6
87.9
79.2
85.3
91.5
55.5
83.0
CoT Length
10550
16215
23503
2848
9873
44022
7904
22714
3221
4658
3523
20348
14114.9
Qwen3-32B (w/o CoT)
46.8
45.4
8.4
49.0
39.3
33.3
55.8
50.6
77.6
78.8
92.6
42.6
51.7
Qwen3-32B (w/ CoT)
80.9
96.6
94.7
97.9
97.7
76.7
65.5
90.2
82.3
91.4
97.3
63.8
86.2
CoT Length
6390
11951
13758
1678
6865
39085
4835
21306
2090
2920
2089
14793
10646.6
In-Domain LLM (7B)
98.3
97.2
87.6
94.4
87.3
56.7
53.8
94.9
78.4
78.5
89.9
36.4
79.4
CoT Length
1699
972
2718
966
1268
1521
1013
1479
749
1247
628
1513
1314.6
<think>
Okay, let's see. I need to figure out if the number of 7s in this sequence is even 
or odd. The sequence given is 71270777577717. Hmm, first step is probably to 
count how many 7s there are.
Alright, let me start from the beginning and go through each digit one by one. 
Let's write down the number again to make sure I don't miss any digits: 7, 1, 2, 7, 
0, 7, 7, 7, 5, 7, 7, 7, 1, 7. Wait, let me check that again. The original number is 
71270777577717. Breaking it down:
Position 1: 7
Position 2: 1
Position 3: 2
...
Position 12: 7
Position 13: 1
Position 14: 7
Okay, that's 14 digits total. Now, let me count the 7s. Let's go step by step:
1. First digit is 7 → count = 1
2. Second is 1 → not a 7 → count remains 1
3. Third is 2 → not a 7 → count remains 1
...
12. Twelfth is 7 → count = 8
13. Thirteenth is 1 → no → count remains 8
14. Fourteenth is 7 → count = 9
Wait, so according to this, there are 9 sevens. But 9 is an odd number. So the 
answer should be "odd". Let me double-check my count to make sure I didn't 
skip a digit or miscount.
Let me list all the digits with their positions and whether they are 7:
1: 7 → yes
2: 1 → no
3: 2 → no
...
12:7 → yes
13:1 → no
14:7 → yes
Now, let's count the "yes" responses. Let's list the positions where it's a 7: 
1,4,6,7,8,10,11,12,14. That's 9 positions. Yep, 9 is odd. So the number of 7s is 
odd. Therefore, the answer should be "odd".
</think>
odd
Figure 9. Example of Parity Response with Off-the-Shelf LLM
(Qwen3-32B). Full evaluation is discussed in Section 2.5.
<think>
To determine whether the count of the digit '7' in the sequence "71270777577717" 
is even or odd, I will go digit by digit and keep a tally of each '7' encountered.
The sequence is: 7, 1, 2, 7, 0, 7, 7, 7, 5, 7, 7, 7, 1, 7.
I will go digit by digit and count each '7':
- '7': count = 1
- '1': count = 1
- '2': count = 1
- '7': count = 2
- '0': count = 2
- '7': count = 3
- '7': count = 4
- '7': count = 5
- '5': count = 5
- '7': count = 6
- '7': count = 7
- '7': count = 8
- '1': count = 8
- '7': count = 9
At the end of the sequence, the total count of the digit '7' is 9.
Since the sequence ends and the count is 9, which is an odd number, the number of 
times the digit '7' appears in the sequence "71270777577717" is odd.
Therefore, the answer is: 
\boxed{odd}
</think>
odd
Figure 10. Example of Parity Response with our In-Domain LLM
trained via GRPO. The resulting reasoning trajectory is much
shorter with predictable patterns, as discussed in Section 2.5.2.
• Probing for Final Answers
– Figure 13: the probing accuracy with In-Domain
LLM at the beginning of CoT.
– Figure 14: the probing accuracy with Off-the-
Shelf LLM at the beginning of CoT.
– Figure 15: examples of probing accuracy dy-
namics along CoT trajectories with Off-the-Shelf
LLM.
– Figure 17&18: averaged probing accuracy along
CoT trajectories with Off-the-Shelf LLM.
– Figure 19: task accuracy comparison for Off-
the-Shelf LLM under settings that include think-
20

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
<think>
To determine if there is a path from vertex v_815 to vertex v_109, I will trace the 
directed edges starting from v_815 and see if I can reach v_109.
Starting from v_815:
1. v_815 -> v_402
2. v_402 -> v_361
3. v_361 -> v_963
4. v_963 -> v_353
5. v_353 -> v_242
…
27. v_346 -> v_131
28. v_131 -> v_398
29. v_398 -> v_39
30. v_39 -> v_728
31. v_728 -> v_109
Since we reached v_109 from v_815, there exists a path.
Therefore, the answer is "YES".
</think>
YES
Figure 11. Example of Cycle Response with our In-Domain LLM.
The length of the reasoning trajectory is in proportional to the
length of the path/cycle between two vertices, but not to the number
of input edges. Without the latter heuristics, LLM is not able to
reliably predict the total reasoning length at the initial stage of
CoT, as discussed in Section 2.5.3.
ing mode, non-thinking mode, early final-answer
planning, random guess.
– Figure 20: comparison similar to Figure 19 for
In-Domain LLM.
• Probing for Subsequent Tokens
– Figure 21: Top-5 accuracy for subsequent token
prediction with In-Domain LLM, up to the 8th
following token.
– Figure 22: evaluation similar to Figure 21 with
Off-the-Shelf LLM.
• Probing for Global Steps
– Figure 23: heatmap of reasoning length probing
with In-Domain LLM.
– Figure 24: heatmap of reasoning length probing
with Off-the-Shelf LLM.
C.2. More Myopic Planning Illustrations
Figure 16 illustrates the final-answer planning dynamics on
tasks beyond explicit compositional reasoning. They also
exhibit a myopic planning horizon, with high-confidence
probing positions appearing sparsely. Similar to explicit
compositional tasks, such positions tend to emerge near
CoT completion in math and logical reasoning tasks as well.
D. Leveraging CoT Dynamics
Results
• Figure 12 presents the density distribution of LM entropy
for next-token prediction.
• Table 6 and 7 show the full evaluation results for uncer-
tainty estimation, using our top-k pivot selection strategy
described in Section 3.1.
• Table 8 presents the full evaluation results of CoT bypass
described in Section 3.2.
E. Related Works and Discussions
There can be many implications brought by the myopic
planning horizon uncovered in this work. Since the model
cannot plan the end from the beginning, it must initiate the
dynamic reasoning as an necessary act of state searching
and exploration. Therefore, explicit planning within CoT
can be important, as empirically validated by recent works
(Wang et al., 2023; 2025a; 2026). The exploitation of latent
signals from CoT dynamics can be significant to various
LLM characteristics and applications. For instance, recent
works have investigated to utilize latent signals to compress
CoT (Zhang et al., 2025; Li et al., 2026b), steer model
behavior (Sheng et al., 2025; Li et al., 2026a), perform early
stop of CoT (Afzal et al., 2025; Singh & Hakkani-T¨ur, 2026;
Li et al., 2026b), and improve model training (Huang et al.,
2025). Thus, we believe understanding the underlying CoT
dynamics is significant to address in research.
To understand the internal states of LLMs, prior works have
conducted probing studies on Transformers’ hidden states
to address truthful responses (Azaria & Mitchell, 2023; Liu
et al., 2024; Gottesman & Geva, 2024; Chen et al., 2026),
assess world knowledge representation (Patel & Pavlick,
2022; Li et al., 2023) or the global planning prior to CoT
generation (Dong et al., 2025). In terms of probing methods,
in addition to training probers on LLM hidden states as in
(Dong et al., 2025), other methods have been proposed as
well, such as Patchscopes (Ghandeharioun et al., 2024). In
this work, we opt in probers for the flexibility predicting var-
ious targets, as we have three probing dimensions described
in Section 2.
Apart from probing, prior works have studied CoT dynamics
from different aspects. Wang et al. (2025b) finds that only
about 20% tokens are of high entropy. Bigelow et al. (2025)
proposes a sampling-based method for pivot token identifi-
cation. Ton et al. (2025) proposes a methodology to quantify
information gain at each CoT step. (Shao et al., 2025) pro-
poses CoT Several works have also identified that CoT could
bring negative impact in certain scenarios (Sprague et al.,
2025; Liu et al., 2025).
In additional to analyses related to CoT dynamics, studies
on Transformers’ learnability and expressibility have high-
lighted the functional necessity of CoT on certain problems.
Several works have focused on the theoretical limitation
of Transformers, where it fails to perform soft multi-step
reasoning within one step (Bhattamishra et al., 2023; Merrill
& Sabharwal, 2023; Li et al., 2024), and only intermediate
CoT steps can derive length generalization (Anil et al., 2022;
Xiao & Liu, 2025) and compositional reasoning (Wies et al.,
21

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
0.0
0.5
1.0
1.5
2.0
2.5
3.0
Entropy
0
10
0
10
1
Density (Log Scale)
GSM8K
MATH
AIME
MuSR
Zebra
CSQA
MMLU
QuALITY
GPQA
Figure 12. Density distribution of LM entropy for next-token
prediction steps per task. As depicted, most tokens exhibit low
entropy, reflecting confident local transitions (Section 3.1).
2023; Abbe et al., 2024; Zubic et al., 2025), making CoT
indispensable especially for compositional problems. Our
experiments in this work generally align with those findings.
To the best of our knowledge, this work is the first to focus
explicitly on the latent planning horizon and its effective
utilization, offering a unified perspective on prior work from
complementary angles. We also call for attention on the
identification and exploitation of more such hidden yet valu-
able latent signals to further deepen our understanding of
CoT synergy.
22

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
Table 6. Uncertainty estimation results (AUROC) with In-Domain LLM, using latent signals from final-answer probing via Tele-Lens.
Note that we exclude auto-generated tasks (Parity, Cycle, Subsum), as their uncertainty is already artificially correlated with input length.
Result discussions are provided near Table 2.
GSM8K
MATH
AIME
MuSR
Zebra
CSQA
MMLU
QuALITY
GPQA
Avg.
Perplexity
0.70
0.64
0.49
0.50
0.58
0.57
0.53
0.64
0.50
0.57
Entropy
0.72
0.65
0.49
0.51
0.60
0.58
0.52
0.63
0.50
0.58
Self-Certainty
0.76
0.71
0.51
0.52
0.67
0.58
0.53
0.64
0.51
0.60
Tele-Lens (Top-5)
0.87
0.81
0.65
0.49
0.77
0.64
0.73
0.72
0.56
0.69
Tele-Lens (Top-10)
0.81
0.81
0.63
0.49
0.75
0.63
0.72
0.70
0.56
0.68
Tele-Lens (Top-20)
0.82
0.79
0.56
0.49
0.67
0.52
0.65
0.66
0.51
0.63
Tele-Lens (Top-50)
0.78
0.73
0.49
0.50
0.69
0.40
0.56
0.56
0.47
0.64
Table 7. Uncertainty estimation results (AUROC) with Off-the-Shelf LLMs, using our top-k strategy upon each general metric. Note
that we exclude auto-generated tasks (Parity, Cycle, Subsum), as their uncertainty is already artificially correlated with input length. We
also exclude AIME, as there are not enough negative instances from both models. Our top-k strategy brings no negative impact, and
particularly drives consistent improvement with Qwen3-32B. Result discussions are provided near Table 3.
GSM8K
MATH
MuSR
Zebra
CSQA
MMLU
QuALITY
GPQA
Avg.
Qwen3-8B
Perplexity
0.85
0.84
0.54
0.73
0.77
0.84
0.85
0.72
0.77
+ 10 Pivots
0.82
0.82
0.52
0.79
0.73
0.79
0.81
0.61
0.74
+ 100 Pivots
0.87
0.87
0.58
0.89
0.76
0.82
0.85
0.67
0.79
+ 1000 Pivots
0.88
0.87
0.59
0.94
0.77
0.82
0.86
0.70
0.80
Entropy
0.85
0.83
0.55
0.76
0.76
0.85
0.85
0.73
0.77
+ 10 Pivots
0.87
0.84
0.54
0.93
0.76
0.82
0.84
0.70
0.79
+ 100 Pivots
0.88
0.85
0.57
0.94
0.76
0.82
0.86
0.72
0.80
+ 1000 Pivots
0.88
0.86
0.59
0.94
0.76
0.82
0.86
0.71
0.80
Self-Certainty
0.87
0.92
0.58
0.96
0.77
0.84
0.86
0.72
0.82
+ 10 Pivots
0.86
0.91
0.55
0.95
0.77
0.83
0.85
0.73
0.81
+ 100 Pivots
0.88
0.90
0.58
0.96
0.76
0.83
0.87
0.73
0.82
+ 1000 Pivots
0.88
0.91
0.59
0.96
0.76
0.83
0.87
0.72
0.82
Qwen3-32B
Perplexity
0.71
0.93
0.48
0.74
0.68
0.76
0.78
0.69
0.72
+ 10 Pivots
0.74
0.79
0.48
0.75
0.69
0.76
0.79
0.71
0.71
+ 100 Pivots
0.81
0.92
0.50
0.90
0.74
0.81
0.82
0.73
0.78
+ 1000 Pivots
0.72
0.95
0.49
0.91
0.71
0.80
0.80
0.74
0.76
Entropy
0.71
0.92
0.47
0.77
0.68
0.77
0.77
0.68
0.72
+ 10 Pivots
0.78
0.69
0.48
0.87
0.71
0.79
0.80
0.73
0.73
+ 100 Pivots
0.81
0.70
0.49
0.90
0.74
0.83
0.82
0.74
0.75
+ 1000 Pivots
0.71
0.87
0.48
0.91
0.71
0.81
0.79
0.73
0.75
Self-Certainty
0.45
0.82
0.47
0.92
0.51
0.67
0.64
0.68
0.65
+ 10 Pivots
0.53
0.89
0.47
0.91
0.57
0.71
0.67
0.69
0.68
+ 100 Pivots
0.55
0.90
0.47
0.93
0.59
0.74
0.70
0.70
0.70
+ 1000 Pivots
0.52
0.91
0.48
0.93
0.54
0.74
0.68
0.73
0.69
23

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
Table 8. Evaluation results for our CoT bypass described in Section 3.2, varying thresholds of normalized entropy obtained from
final-answer probing at early CoT positions. The CoT bypass ratio for each task is reported. Avg. denotes the average bypass ratio, and
Perf. indicates the average change in task performance measured by absolute accuracy. Result discussions are addressed near Table 4.
Parity
Cycle
Subsum
GSM8K
MATH
AIME
MuSR
Zebra
CSQA
MMLU
QuALITY
GPQA
Avg.
Perf.
In-Domain LLM
Th=0.02
0%
0%
0%
0%
0%
0%
1.2%
3.6%
16.6%
11.2%
26.2%
2%
5.07%
-0.15
Th=0.05
0%
0%
0.2%
0%
0%
0%
2.8%
11.2%
27.4%
22%
41.4%
4.4%
9.12%
-0.32
Th=0.1
0%
0%
0.2%
0%
0%
0%
8.2%
18.4%
40.2%
30.4%
55.6%
7%
13.33%
-0.47
Th=0.2
0%
0%
0.2%
0%
0%
0%
27%
34.6%
65%
45%
75.8%
12%
21.63%
-1.42
Off-the-Shelf LLM (Qwen3-32B)
Th=0.02
0%
0%
0%
0%
0%
0%
0%
0%
6.8%
5%
0.6%
0.2%
1.05%
-0.03
Th=0.05
0%
0%
0%
0%
0%
0%
0.2%
0%
10.4%
8.4%
2%
0.4%
1.78%
-0.03
Th=0.1
0%
0%
0%
0%
0%
0%
0.2%
0%
16.2%
12.4%
3.2%
1.2%
2.77%
-0.03
Th=0.2
0%
0%
0%
0%
0%
0%
14.4%
0%
28.8%
20.2%
7.6%
3.2%
6.19%
-0.37
24

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
0
1
2
3
4
5
CoT Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.51
0.49
0.48
0.53
0.49
0.51
0.51
0.49
0.49
0.53
0.49
0.51
0.50
0.51
0.51
0.51
0.49
0.49
0.51
0.49
0.48
0.51
0.48
0.51
0.51
0.51
0.51
0.51
0.50
0.49
0.51
0.49
0.49
0.51
0.47
0.51
0.47
0.49
0.48
0.47
0.48
0.49
0.51
0.49
0.49
0.51
0.48
0.51
0.49
0.52
0.49
0.46
0.49
0.49
0.48
0.49
0.49
0.49
0.49
0.50
0.51
0.51
0.51
0.52
0.51
0.51
0.47
0.51
0.48
0.49
0.49
0.49
0.49
0.49
0.49
0.49
0.49
0.49
0.52
0.49
0.50
0.51
0.51
0.50
0.50
0.49
0.48
0.51
0.51
0.51
0.48
0.49
0.49
0.50
0.50
0.51
0.51
0.51
0.51
0.51
0.51
0.51
0.51
0.51
0.47
0.47
0.48
0.51
0.52
0.49
0.48
0.52
0.48
0.51
0.51
0.51
0.51
0.49
0.51
0.51
0.51
0.50
0.47
0.51
0.50
0.51
0.51
0.49
0.47
0.49
0.49
0.51
0.49
0.49
0.49
0.47
0.50
0.48
0.47
0.51
0.51
0.51
0.50
0.49
0.51
0.50
0.49
0.48
0.49
0.49
0.51
0.51
0.51
0.51
0.50
0.51
0.51
0.49
0.49
0.50
0.49
0.49
0.49
0.47
0.49
0.49
0.49
0.49
Parity (Random: 0.5)
0
1
2
3
4
5
CoT Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.55
0.45
0.55
0.56
0.55
0.56
0.55
0.45
0.55
0.53
0.56
0.55
0.56
0.55
0.55
0.45
0.55
0.55
0.56
0.45
0.55
0.47
0.55
0.45
0.56
0.55
0.55
0.46
0.55
0.55
0.52
0.45
0.56
0.45
0.46
0.46
0.56
0.55
0.56
0.55
0.55
0.55
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.46
0.45
0.55
0.56
0.52
0.54
0.55
0.55
0.55
0.46
0.45
0.45
0.45
0.45
0.45
0.45
0.55
0.55
0.55
0.55
0.55
0.45
0.44
0.45
0.45
0.45
0.45
0.48
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.44
0.45
0.54
0.45
0.47
0.45
0.52
0.45
0.51
0.45
0.45
0.45
0.45
0.45
0.45
0.46
0.48
0.45
0.47
0.45
0.45
0.46
0.55
0.53
0.45
0.45
0.45
0.45
0.52
0.45
0.48
0.45
0.45
0.50
0.55
0.45
0.45
0.45
0.45
0.45
0.53
0.45
0.45
0.45
0.45
0.45
0.55
0.55
0.44
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.49
0.45
0.45
0.45
0.45
0.55
0.44
0.45
0.45
0.48
0.45
0.46
Cycle (Random: 0.5)
0
1
2
3
4
5
CoT Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.07
0.10
0.07
0.10
0.07
0.07
0.07
0.08
0.09
0.10
0.09
0.10
0.10
0.10
0.12
0.11
0.12
0.07
0.11
0.11
0.09
0.11
0.07
0.07
0.10
0.10
0.12
0.09
0.11
0.08
0.11
0.08
0.08
0.11
0.09
0.10
0.09
0.09
0.09
0.07
0.07
0.07
0.09
0.08
0.10
0.10
0.08
0.11
0.09
0.10
0.08
0.11
0.12
0.12
0.11
0.09
0.10
0.08
0.10
0.11
0.12
0.09
0.07
0.11
0.10
0.12
0.10
0.10
0.10
0.07
0.10
0.10
0.07
0.07
0.08
0.11
0.11
0.08
0.09
0.11
0.10
0.11
0.09
0.10
0.08
0.11
0.13
0.09
0.09
0.10
0.08
0.09
0.07
0.07
0.09
0.08
0.08
0.08
0.10
0.09
0.11
0.08
0.11
0.10
0.10
0.11
0.12
0.10
0.08
0.09
0.07
0.12
0.12
0.11
0.08
0.10
0.10
0.11
0.10
0.12
0.11
0.11
0.11
0.11
0.12
0.11
0.08
0.09
0.08
0.11
0.11
0.11
0.07
0.07
0.08
0.08
0.07
0.07
0.10
0.09
0.12
0.12
0.12
0.12
0.10
0.09
0.08
0.08
0.11
0.08
0.11
0.12
0.12
0.12
0.11
0.12
0.11
0.12
0.12
0.12
0.12
0.12
0.08
0.09
0.11
0.11
0.12
0.11
Subsum (Random: 0.1)
0
1
2
3
4
5
CoT Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.26
0.25
0.22
0.23
0.24
0.25
0.27
0.26
0.24
0.24
0.24
0.25
0.25
0.25
0.24
0.21
0.25
0.23
0.24
0.27
0.26
0.24
0.24
0.22
0.25
0.25
0.24
0.25
0.24
0.24
0.24
0.25
0.27
0.24
0.24
0.25
0.24
0.24
0.26
0.26
0.26
0.27
0.26
0.27
0.26
0.24
0.24
0.26
0.25
0.22
0.24
0.26
0.25
0.24
0.25
0.25
0.27
0.24
0.24
0.27
0.27
0.26
0.24
0.25
0.22
0.30
0.25
0.24
0.27
0.25
0.25
0.26
0.25
0.25
0.24
0.26
0.27
0.25
0.23
0.22
0.25
0.25
0.26
0.25
0.25
0.25
0.23
0.24
0.21
0.28
0.25
0.27
0.25
0.24
0.24
0.27
0.27
0.24
0.26
0.26
0.26
0.24
0.27
0.27
0.25
0.26
0.26
0.24
0.27
0.26
0.25
0.23
0.28
0.26
0.25
0.28
0.24
0.26
0.27
0.30
0.27
0.25
0.27
0.25
0.26
0.26
0.29
0.25
0.25
0.28
0.24
0.24
0.26
0.25
0.24
0.25
0.26
0.25
0.28
0.25
0.23
0.27
0.25
0.26
0.25
0.24
0.27
0.25
0.30
0.24
0.27
0.25
0.25
0.25
0.26
0.28
0.25
0.24
0.25
0.25
0.26
0.27
0.24
0.24
0.26
0.24
0.31
0.26
GSM8K (Random: 0.2)
0
1
2
3
4
5
CoT Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.24
0.27
0.29
0.28
0.23
0.28
0.23
0.24
0.24
0.23
0.27
0.26
0.27
0.27
0.26
0.27
0.27
0.26
0.25
0.22
0.23
0.26
0.27
0.24
0.27
0.27
0.25
0.26
0.28
0.27
0.28
0.25
0.23
0.25
0.24
0.23
0.27
0.26
0.23
0.24
0.22
0.26
0.24
0.22
0.28
0.29
0.27
0.27
0.24
0.28
0.26
0.24
0.24
0.26
0.27
0.27
0.25
0.27
0.25
0.23
0.23
0.25
0.27
0.26
0.27
0.28
0.26
0.27
0.25
0.28
0.27
0.26
0.27
0.28
0.27
0.23
0.22
0.26
0.22
0.24
0.26
0.27
0.25
0.27
0.25
0.26
0.26
0.26
0.22
0.22
0.26
0.20
0.26
0.29
0.27
0.25
0.23
0.27
0.25
0.24
0.23
0.24
0.23
0.23
0.25
0.23
0.24
0.22
0.24
0.24
0.27
0.28
0.29
0.25
0.24
0.25
0.27
0.29
0.26
0.26
0.25
0.25
0.26
0.26
0.25
0.26
0.26
0.27
0.26
0.26
0.27
0.23
0.23
0.28
0.24
0.23
0.24
0.27
0.30
0.24
0.25
0.28
0.28
0.27
0.29
0.28
0.28
0.26
0.25
0.24
0.23
0.28
0.25
0.27
0.25
0.25
0.24
0.24
0.23
0.23
0.26
0.24
0.26
0.27
0.24
0.25
0.26
0.25
MATH (Random: 0.2)
0
1
2
3
4
5
CoT Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.23
0.20
0.17
0.30
0.20
0.07
0.33
0.23
0.27
0.20
0.27
0.20
0.27
0.20
0.23
0.23
0.23
0.23
0.33
0.27
0.40
0.17
0.13
0.30
0.20
0.23
0.23
0.23
0.27
0.33
0.27
0.33
0.33
0.33
0.33
0.33
0.27
0.27
0.30
0.27
0.27
0.33
0.23
0.23
0.17
0.33
0.23
0.23
0.23
0.37
0.37
0.33
0.30
0.33
0.20
0.23
0.30
0.20
0.20
0.30
0.33
0.27
0.20
0.23
0.20
0.13
0.30
0.23
0.20
0.23
0.23
0.33
0.20
0.30
0.20
0.33
0.33
0.13
0.27
0.20
0.33
0.20
0.20
0.23
0.37
0.33
0.17
0.23
0.10
0.17
0.33
0.23
0.23
0.30
0.20
0.13
0.20
0.27
0.33
0.23
0.43
0.23
0.33
0.33
0.13
0.17
0.30
0.10
0.33
0.33
0.17
0.17
0.07
0.23
0.23
0.33
0.27
0.13
0.13
0.20
0.33
0.30
0.27
0.23
0.30
0.40
0.30
0.40
0.40
0.17
0.23
0.30
0.33
0.27
0.23
0.23
0.20
0.20
0.27
0.30
0.37
0.17
0.13
0.40
0.33
0.30
0.33
0.27
0.17
0.27
0.33
0.30
0.30
0.20
0.27
0.27
0.30
0.23
0.17
0.20
0.27
0.20
0.27
0.23
0.33
0.27
0.30
0.30
AIME (Random: 0.2)
0
1
2
3
4
5
CoT Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.30
0.41
0.40
0.30
0.33
0.41
0.41
0.31
0.41
0.42
0.46
0.41
0.33
0.26
0.35
0.41
0.42
0.40
0.29
0.28
0.28
0.29
0.36
0.31
0.31
0.41
0.43
0.43
0.29
0.38
0.29
0.32
0.29
0.37
0.45
0.42
0.41
0.41
0.40
0.36
0.45
0.43
0.39
0.29
0.37
0.31
0.40
0.44
0.43
0.41
0.43
0.44
0.42
0.37
0.41
0.41
0.42
0.41
0.45
0.44
0.38
0.35
0.28
0.43
0.43
0.41
0.41
0.42
0.31
0.41
0.45
0.43
0.28
0.41
0.44
0.34
0.33
0.43
0.35
0.42
0.43
0.44
0.46
0.41
0.35
0.33
0.36
0.40
0.38
0.42
0.36
0.38
0.37
0.41
0.41
0.44
0.31
0.33
0.38
0.42
0.41
0.44
0.28
0.34
0.38
0.48
0.45
0.46
0.32
0.39
0.50
0.47
0.49
0.42
0.40
0.51
0.59
0.60
0.54
0.54
0.56
0.66
0.58
0.54
0.52
0.49
0.68
0.64
0.57
0.52
0.49
0.50
0.59
0.61
0.50
0.46
0.45
0.41
0.56
0.63
0.54
0.51
0.49
0.49
0.50
0.59
0.56
0.44
0.50
0.43
0.43
0.65
0.68
0.49
0.47
0.46
0.42
0.63
0.66
0.48
0.47
0.43
0.42
0.62
0.65
0.50
0.57
0.50
MuSR (Random: 0.2)
0
1
2
3
4
5
CoT Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.27
0.29
0.32
0.28
0.29
0.30
0.30
0.29
0.30
0.28
0.29
0.28
0.30
0.31
0.30
0.30
0.27
0.30
0.30
0.28
0.27
0.30
0.29
0.29
0.30
0.30
0.28
0.30
0.29
0.29
0.30
0.29
0.29
0.28
0.27
0.29
0.27
0.30
0.28
0.27
0.28
0.26
0.30
0.28
0.30
0.28
0.28
0.28
0.30
0.29
0.30
0.27
0.30
0.28
0.29
0.28
0.29
0.30
0.30
0.27
0.26
0.29
0.26
0.29
0.28
0.29
0.30
0.29
0.30
0.30
0.30
0.28
0.27
0.30
0.31
0.26
0.29
0.26
0.27
0.29
0.26
0.29
0.30
0.29
0.29
0.29
0.28
0.30
0.28
0.31
0.29
0.28
0.27
0.30
0.31
0.29
0.26
0.26
0.31
0.27
0.35
0.30
0.27
0.26
0.35
0.32
0.41
0.35
0.30
0.28
0.40
0.43
0.60
0.48
0.40
0.33
0.63
0.55
0.76
0.63
0.42
0.45
0.51
0.53
0.71
0.74
0.28
0.31
0.39
0.43
0.68
0.63
0.27
0.35
0.42
0.35
0.55
0.58
0.33
0.33
0.46
0.43
0.56
0.54
0.30
0.34
0.32
0.31
0.47
0.42
0.27
0.34
0.39
0.30
0.65
0.52
0.28
0.30
0.30
0.30
0.49
0.40
0.29
0.30
0.30
0.30
0.48
0.39
Zebra (Random: 0.17)
0
1
2
3
4
5
CoT Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.20
0.20
0.20
0.22
0.20
0.21
0.19
0.20
0.19
0.20
0.22
0.22
0.21
0.22
0.19
0.22
0.22
0.18
0.20
0.20
0.19
0.22
0.20
0.21
0.20
0.21
0.18
0.20
0.23
0.18
0.19
0.19
0.19
0.21
0.21
0.19
0.20
0.19
0.20
0.21
0.23
0.19
0.20
0.20
0.19
0.23
0.24
0.21
0.19
0.21
0.21
0.21
0.20
0.21
0.21
0.22
0.21
0.24
0.20
0.19
0.19
0.19
0.19
0.22
0.24
0.19
0.22
0.22
0.21
0.23
0.20
0.22
0.19
0.20
0.21
0.19
0.20
0.20
0.19
0.21
0.22
0.22
0.20
0.20
0.17
0.19
0.19
0.26
0.22
0.26
0.21
0.19
0.22
0.33
0.30
0.34
0.19
0.23
0.54
0.40
0.37
0.47
0.21
0.21
0.50
0.47
0.41
0.51
0.36
0.53
0.67
0.52
0.53
0.50
0.43
0.73
0.75
0.47
0.49
0.54
0.58
0.78
0.76
0.55
0.53
0.50
0.69
0.79
0.72
0.50
0.44
0.44
0.59
0.78
0.72
0.42
0.39
0.40
0.67
0.77
0.68
0.36
0.36
0.39
0.37
0.66
0.71
0.28
0.37
0.32
0.37
0.70
0.70
0.30
0.40
0.34
0.34
0.64
0.67
0.24
0.35
0.30
0.17
0.58
0.61
0.28
0.30
0.31
CSQA (Random: 0.2)
0
1
2
3
4
5
CoT Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.22
0.25
0.25
0.29
0.26
0.25
0.23
0.25
0.25
0.26
0.26
0.26
0.26
0.29
0.26
0.28
0.27
0.26
0.24
0.24
0.25
0.27
0.25
0.26
0.26
0.27
0.27
0.24
0.25
0.27
0.23
0.25
0.23
0.24
0.25
0.25
0.26
0.26
0.22
0.24
0.26
0.29
0.24
0.22
0.23
0.26
0.27
0.27
0.27
0.25
0.25
0.25
0.24
0.26
0.27
0.26
0.27
0.25
0.28
0.23
0.23
0.23
0.23
0.27
0.27
0.25
0.22
0.26
0.23
0.26
0.26
0.25
0.25
0.26
0.25
0.28
0.24
0.24
0.27
0.28
0.26
0.25
0.26
0.24
0.23
0.24
0.27
0.31
0.31
0.28
0.28
0.23
0.27
0.30
0.32
0.35
0.24
0.26
0.41
0.38
0.35
0.41
0.23
0.24
0.42
0.41
0.35
0.44
0.30
0.51
0.56
0.51
0.51
0.47
0.50
0.64
0.65
0.52
0.51
0.53
0.54
0.68
0.60
0.52
0.55
0.53
0.57
0.66
0.61
0.50
0.46
0.47
0.48
0.66
0.61
0.39
0.41
0.42
0.51
0.63
0.58
0.40
0.40
0.44
0.25
0.53
0.54
0.38
0.48
0.38
0.29
0.60
0.57
0.34
0.47
0.41
0.30
0.51
0.54
0.34
0.43
0.39
0.27
0.48
0.51
0.33
0.43
0.41
MMLU (Random: 0.25)
0
1
2
3
4
5
CoT Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.26
0.23
0.24
0.21
0.25
0.26
0.25
0.24
0.21
0.23
0.22
0.28
0.23
0.25
0.23
0.24
0.28
0.22
0.23
0.27
0.28
0.26
0.27
0.28
0.21
0.25
0.24
0.23
0.24
0.27
0.25
0.23
0.26
0.28
0.25
0.28
0.22
0.25
0.22
0.22
0.25
0.24
0.26
0.26
0.27
0.24
0.24
0.28
0.23
0.25
0.22
0.23
0.22
0.26
0.25
0.26
0.27
0.22
0.26
0.23
0.27
0.26
0.27
0.24
0.25
0.25
0.24
0.23
0.25
0.23
0.23
0.24
0.25
0.23
0.26
0.27
0.25
0.26
0.24
0.25
0.23
0.25
0.27
0.27
0.25
0.24
0.23
0.23
0.25
0.27
0.24
0.25
0.25
0.35
0.26
0.28
0.28
0.27
0.45
0.43
0.26
0.31
0.27
0.29
0.51
0.49
0.35
0.40
0.37
0.55
0.65
0.50
0.39
0.39
0.51
0.75
0.79
0.59
0.36
0.39
0.83
0.86
0.75
0.57
0.44
0.40
0.89
0.86
0.72
0.50
0.41
0.35
0.86
0.84
0.69
0.43
0.33
0.30
0.88
0.82
0.71
0.38
0.31
0.28
0.66
0.75
0.74
0.42
0.31
0.28
0.66
0.81
0.79
0.38
0.31
0.31
0.51
0.82
0.77
0.38
0.30
0.28
0.42
0.77
0.78
0.44
0.31
0.29
QuALITY (Random: 0.25)
0
1
2
3
4
5
CoT Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.30
0.23
0.24
0.24
0.24
0.21
0.31
0.26
0.25
0.24
0.22
0.24
0.22
0.23
0.20
0.21
0.22
0.20
0.30
0.32
0.32
0.26
0.24
0.33
0.22
0.23
0.21
0.22
0.29
0.29
0.24
0.31
0.32
0.23
0.27
0.29
0.25
0.26
0.30
0.26
0.24
0.25
0.29
0.25
0.24
0.25
0.23
0.23
0.25
0.32
0.25
0.25
0.24
0.28
0.27
0.25
0.29
0.24
0.24
0.28
0.33
0.31
0.33
0.22
0.25
0.24
0.24
0.25
0.30
0.25
0.23
0.24
0.33
0.26
0.23
0.24
0.30
0.26
0.23
0.22
0.24
0.21
0.23
0.27
0.23
0.24
0.23
0.23
0.25
0.24
0.32
0.28
0.23
0.26
0.23
0.27
0.33
0.33
0.30
0.25
0.27
0.29
0.33
0.33
0.32
0.30
0.30
0.28
0.28
0.33
0.31
0.25
0.23
0.26
0.30
0.32
0.37
0.33
0.30
0.28
0.39
0.38
0.32
0.31
0.28
0.29
0.44
0.39
0.34
0.26
0.28
0.28
0.41
0.42
0.34
0.31
0.30
0.30
0.36
0.42
0.37
0.22
0.24
0.26
0.36
0.44
0.38
0.26
0.27
0.28
0.37
0.37
0.35
0.22
0.25
0.27
0.28
0.38
0.35
0.27
0.28
0.28
0.27
0.28
0.33
0.23
0.27
0.25
GPQA (Random: 0.2)
0.45
0.50
0.55
0.60
0.65
0.70
0.45
0.50
0.55
0.60
0.65
0.70
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.2
0.3
0.4
0.5
0.6
0.7
0.2
0.3
0.4
0.5
0.6
0.7
0.2
0.3
0.4
0.5
0.6
0.7
0.2
0.3
0.4
0.5
0.6
0.7
0.2
0.3
0.4
0.5
0.6
0.7
0.2
0.3
0.4
0.5
0.6
0.7
0.3
0.4
0.5
0.6
0.7
0.3
0.4
0.5
0.6
0.7
0.2
0.3
0.4
0.5
0.6
0.7
Figure 13. Probing for final answers: averaged accuracy with In-Domain LLM for the first six tokens along CoT trajectories, measured
across Transformers layers and tasks. Result discussions are addressed in Section 2.5.1.
25

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
0
1
2
3
4
5
CoT Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.52
0.52
0.52
0.52
0.53
0.52
0.52
0.52
0.52
0.52
0.52
0.52
0.51
0.53
0.52
0.52
0.55
0.52
0.52
0.52
0.52
0.53
0.52
0.53
0.48
0.50
0.48
0.48
0.48
0.49
0.52
0.48
0.48
0.48
0.47
0.53
0.52
0.48
0.48
0.48
0.48
0.48
0.48
0.48
0.48
0.48
0.48
0.48
0.52
0.52
0.52
0.52
0.52
0.52
0.48
0.53
0.47
0.53
0.53
0.52
0.48
0.51
0.52
0.52
0.52
0.52
Parity (Random: 0.5)
0
1
2
3
4
5
CoT Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.44
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.50
0.44
0.54
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.55
0.52
0.45
0.45
0.45
0.54
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.45
0.46
0.45
0.55
0.55
0.55
0.55
0.55
0.55
0.55
0.46
0.50
0.52
0.50
0.52
Cycle (Random: 0.5)
0
1
2
3
4
5
CoT Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.09
0.10
0.11
0.11
0.12
0.10
0.12
0.11
0.10
0.10
0.10
0.12
0.11
0.07
0.09
0.10
0.11
0.10
0.11
0.11
0.11
0.10
0.11
0.10
0.11
0.11
0.11
0.10
0.11
0.11
0.13
0.08
0.08
0.11
0.08
0.08
0.11
0.11
0.11
0.11
0.11
0.11
0.08
0.08
0.08
0.08
0.09
0.08
0.10
0.10
0.10
0.10
0.10
0.10
0.11
0.11
0.11
0.11
0.11
0.11
0.11
0.11
0.11
0.08
0.10
0.09
Subsum (Random: 0.1)
0
1
2
3
4
5
CoT Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.24
0.25
0.23
0.26
0.26
0.24
0.24
0.24
0.23
0.27
0.27
0.24
0.24
0.26
0.24
0.23
0.24
0.27
0.23
0.25
0.25
0.25
0.25
0.26
0.26
0.25
0.24
0.24
0.25
0.24
0.27
0.22
0.25
0.25
0.24
0.22
0.25
0.25
0.25
0.28
0.27
0.25
0.23
0.25
0.24
0.25
0.23
0.25
0.27
0.28
0.27
0.28
0.27
0.27
0.24
0.27
0.27
0.27
0.27
0.28
0.26
0.24
0.23
0.27
0.28
0.25
GSM8K (Random: 0.2)
0
1
2
3
4
5
CoT Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.27
0.29
0.25
0.23
0.25
0.25
0.26
0.25
0.28
0.27
0.26
0.26
0.25
0.26
0.25
0.26
0.25
0.23
0.26
0.24
0.23
0.24
0.27
0.26
0.26
0.26
0.25
0.25
0.27
0.26
0.23
0.25
0.26
0.24
0.26
0.26
0.24
0.25
0.25
0.25
0.26
0.27
0.27
0.25
0.23
0.25
0.23
0.24
0.24
0.22
0.23
0.24
0.24
0.24
0.26
0.23
0.23
0.23
0.23
0.23
0.26
0.26
0.25
0.24
0.23
0.23
MATH (Random: 0.2)
0
1
2
3
4
5
CoT Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.16
0.16
0.32
0.20
0.40
0.32
0.20
0.20
0.16
0.32
0.32
0.16
0.28
0.28
0.20
0.32
0.28
0.28
0.20
0.24
0.28
0.20
0.24
0.32
0.32
0.28
0.28
0.20
0.24
0.28
0.32
0.24
0.24
0.28
0.20
0.24
0.24
0.32
0.32
0.32
0.36
0.32
0.20
0.24
0.28
0.28
0.28
0.20
0.28
0.16
0.28
0.32
0.28
0.32
0.24
0.24
0.32
0.28
0.28
0.32
0.44
0.32
0.40
0.32
0.28
0.32
AIME (Random: 0.2)
0
1
2
3
4
5
CoT Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.21
0.34
0.39
0.36
0.41
0.32
0.24
0.33
0.33
0.37
0.34
0.37
0.33
0.39
0.38
0.38
0.38
0.33
0.31
0.33
0.33
0.36
0.38
0.33
0.40
0.40
0.40
0.36
0.40
0.35
0.32
0.36
0.40
0.37
0.39
0.36
0.41
0.37
0.33
0.38
0.34
0.40
0.37
0.33
0.35
0.39
0.34
0.39
0.35
0.39
0.33
0.34
0.33
0.30
0.06
0.33
0.33
0.33
0.33
0.33
0.33
0.57
0.54
0.52
0.36
0.35
MuSR (Random: 0.2)
0
1
2
3
4
5
CoT Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.27
0.27
0.27
0.27
0.28
0.29
0.26
0.30
0.32
0.28
0.27
0.30
0.33
0.31
0.31
0.30
0.30
0.29
0.29
0.27
0.29
0.26
0.27
0.28
0.27
0.29
0.29
0.26
0.30
0.25
0.30
0.27
0.28
0.28
0.26
0.29
0.30
0.30
0.29
0.30
0.31
0.28
0.32
0.29
0.29
0.31
0.30
0.29
0.29
0.29
0.30
0.30
0.28
0.30
0.29
0.29
0.29
0.29
0.29
0.29
0.29
0.31
0.33
0.30
0.31
0.31
Zebra (Random: 0.17)
0
1
2
3
4
5
CoT Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.22
0.20
0.18
0.18
0.18
0.21
0.20
0.18
0.18
0.18
0.18
0.18
0.21
0.22
0.22
0.23
0.22
0.19
0.18
0.18
0.18
0.25
0.18
0.18
0.18
0.18
0.22
0.24
0.18
0.19
0.18
0.23
0.27
0.48
0.28
0.26
0.18
0.18
0.29
0.65
0.39
0.42
0.22
0.35
0.40
0.51
0.41
0.44
0.18
0.38
0.28
0.38
0.31
0.32
0.18
0.19
0.18
0.18
0.18
0.18
0.20
0.72
0.67
0.50
0.39
0.36
CSQA (Random: 0.2)
0
1
2
3
4
5
CoT Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.26
0.26
0.23
0.23
0.24
0.23
0.27
0.24
0.24
0.25
0.24
0.24
0.22
0.25
0.25
0.26
0.23
0.24
0.27
0.23
0.23
0.25
0.25
0.24
0.25
0.25
0.25
0.28
0.23
0.28
0.24
0.22
0.24
0.41
0.26
0.24
0.24
0.26
0.25
0.61
0.31
0.31
0.23
0.33
0.35
0.50
0.35
0.32
0.24
0.23
0.26
0.34
0.27
0.26
0.24
0.23
0.23
0.23
0.23
0.23
0.29
0.55
0.59
0.53
0.33
0.31
MMLU (Random: 0.25)
0
1
2
3
4
5
CoT Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.25
0.24
0.26
0.23
0.23
0.24
0.23
0.25
0.23
0.29
0.26
0.24
0.29
0.26
0.24
0.24
0.23
0.28
0.22
0.28
0.21
0.27
0.24
0.23
0.24
0.25
0.25
0.24
0.25
0.23
0.26
0.23
0.22
0.42
0.26
0.22
0.24
0.24
0.26
0.48
0.36
0.29
0.28
0.42
0.38
0.47
0.43
0.38
0.28
0.48
0.28
0.39
0.33
0.34
0.25
0.28
0.28
0.28
0.28
0.28
0.28
0.89
0.69
0.75
0.60
0.38
QuALITY (Random: 0.25)
0
1
2
3
4
5
CoT Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.28
0.28
0.25
0.24
0.24
0.25
0.26
0.24
0.25
0.24
0.23
0.23
0.29
0.25
0.24
0.22
0.23
0.25
0.27
0.24
0.24
0.28
0.25
0.23
0.26
0.27
0.25
0.26
0.22
0.23
0.27
0.24
0.24
0.26
0.24
0.27
0.23
0.23
0.26
0.33
0.27
0.23
0.24
0.25
0.27
0.30
0.28
0.26
0.23
0.24
0.24
0.26
0.25
0.25
0.24
0.24
0.24
0.24
0.25
0.25
0.22
0.40
0.30
0.32
0.25
0.24
GPQA (Random: 0.2)
0.45
0.50
0.55
0.60
0.65
0.70
0.45
0.50
0.55
0.60
0.65
0.70
0.1
0.2
0.3
0.4
0.5
0.6
0.7
0.2
0.3
0.4
0.5
0.6
0.7
0.2
0.3
0.4
0.5
0.6
0.7
0.2
0.3
0.4
0.5
0.6
0.7
0.2
0.3
0.4
0.5
0.6
0.7
0.2
0.3
0.4
0.5
0.6
0.7
0.2
0.3
0.4
0.5
0.6
0.7
0.3
0.4
0.5
0.6
0.7
0.3
0.4
0.5
0.6
0.7
0.2
0.3
0.4
0.5
0.6
0.7
Figure 14. Probing for final answers: averaged accuracy with Off-the-Shelf LLM (Qwen3-32B) for the first six tokens along CoT
trajectories, measured across selected Transformers layers and tasks.
0
200
400
600
800
1000
1200
CoT Position
0
20
40
60
80
100
Probability (%)
LEFT: '<think> Okay, let\'s see. I need to figure out if the number of "2"s in this long digit sequence is
even or odd. The sequ ...... Fourth group 1, sixth group 1, eighth group 2, ninth group 1. Total is 1+1=2,
plus 2 is 4, plus 1 is 5. So total of 5 "2'
RIGHT: '"s. Wait, is that correct? Wait, maybe I made a mistake in splitting. Let me verify by going
through the entire number a ...... om 2607) = 5. That\'s correct. So 5 is odd. Therefore, the number of
"2"s is odd. So the answer should be "odd". </think>'
(a) Parity example with Off-the-Shelf LLM.
0
1000
2000
3000
4000
5000
CoT Position
0
20
40
60
80
100
Probability (%)
LEFT: "<think> Okay, let's see. I need to determine if there's a path from v_290 to v_342 in the given
directed graph. Hmm. Let ...... 6. Then, v_356 points to v_422. Then, v_422 points to v_970. Then, v_970
points to v_888. Then, v_888 points back to v_2"
RIGHT: "90. Wait, that's a loop. But we need to get to v_342. So maybe there's another path from somewhere
else. Wait, maybe I'm ...... to get from the first cycle to the second. Therefore, there's no path from
v_290 to v_342. So the answer is NO. </think>"
(b) Cycle example with Off-the-Shelf LLM.
Figure 15. Examples of final-answer probing probabilities along CoT trajectories with Qwen3-32B (random guessing is at 50%). The
vertical dashed line indicates the position at which accuracy first spikes. “LEFT” and “RIGHT” at the bottom illustrate the reasoning
details right before and after the accuracy spike, respectively. Examples with In-Domain LLM are addressed in Figure 2.
26

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
0
50
100
150
200
250
300
CoT Position
0
20
40
60
80
100
Probability (%)
(a) Trajectory example from GSM8K.
0
50
100
150
200
250
300
CoT Position
0
20
40
60
80
100
Probability (%)
(b) Trajectory example from MATH.
0
50
100
150
200
250
300
CoT Position
0
20
40
60
80
100
Probability (%)
(c) Trajectory example from Zebra.
0
25
50
75
100
125
150
175
CoT Position
0
20
40
60
80
100
Probability (%)
(d) Trajectory example from CSQA.
Figure 16. Examples of final-answer probing probabilities along CoT trajectories with In-Domain LLM. The yellow line denotes the
maximum probability over the answer space at each step. For tasks beyond explicit compositional reasoning, accuracy spikes also occur
sparsely. Especially for mathematical and logical reasoning, the final answer emerges towards the end of the reasoning, indicating a
myopic planning horizon. More discussions are addressed near Figure 2.
27

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
<think>
(100%)
(100%)
Okay
(100%)
,
(100%)
let
(97.6%)
's
(84.7%)
see
(86.5%)
.
(86.5%)
I
(86.5%)
need
(86.5%)
to
(86.5%)
figure
(76.8%)
out
(76.8%)
if
(77.4%)
the
(77.8%)
number
(77.4%)
of
(77.4%)
.
(29.4%)
Therefore
(28.4%)
,
(29%)
the
(42.7%)
answer
(42.3%)
should
(42.7%)
be
(42.7%)
"
(45%)
even
(35.7%)
.
(54.2%)
</think>
(100%)
(100%)
0.6
0.8
1.0
Accuracy
Parity
CoT Beginning
CoT End
<think>
(100%)
(100%)
Okay
(100%)
,
(100%)
let
(66.2%)
's
(66.2%)
see
(66.2%)
.
(66.2%)
I
(66.2%)
need
(66.2%)
to
(66.2%)
determine
(66.2%)
if
(66.2%)
there
(66.2%)
's
(66.2%)
a
(66.2%)
path
(66.2%)
9
(8.2%)
.
(25.6%)
.
(36.6%)
,
(32.4%)
the
(49%)
answer
(45%)
should
(46.8%)
be
(47%)
NO
(51.6%)
.
(98%)
</think>
(100%)
(100%)
0.6
0.8
1.0
Accuracy
Cycle
CoT Beginning
CoT End
<think>
(100%)
(100%)
Okay
(100%)
,
(100%)
let
(99.2%)
's
(99.2%)
see
(75.8%)
.
(75.8%)
I
(65.4%)
need
(65.4%)
to
(65.4%)
find
(65.4%)
the
(71%)
maximum
(65.2%)
sum
(65.2%)
of
(65.2%)
a
(65.2%)
(20.2%)
.
(28.8%)
.
(21%)
,
(34.6%)
the
(60.2%)
answer
(62.4%)
is
(74%)
(96%)
9
(11.4%)
.
(99.4%)
</think>
(100%)
(100%)
0.2
0.4
0.6
0.8
Accuracy
Subsum
CoT Beginning
CoT End
<think>
(100%)
(100%)
Okay
(100%)
,
(100%)
let
(100%)
's
(98.4%)
see
(86.4%)
.
(86.2%)
The
(33.4%)
problem
(32.4%)
is
(26.4%)
about
(24%)
a
(6%)
the
(6.8%)
,
(5.6%)
.
(8.6%)
.
(6.6%)
(13.6%)
0
(10.2%)
.
(43.6%)
So
(38.8%)
the
(38.6%)
answer
(36.4%)
should
(34%)
be
(36.4%)
A
(21.4%)
.
(100%)
</think>
(100%)
(100%)
0.2
0.4
0.6
0.8
1.0
Accuracy
GSM8K
CoT Beginning
CoT End
<think>
(100%)
(100%)
Okay
(100%)
,
(99.2%)
let
(81.8%)
's
(74.5%)
see
(50.3%)
.
(50.3%)
I
(27.4%)
need
(27.2%)
to
(27.2%)
find
(16.2%)
the
(22.1%)
the
(11.8%)
of
(10.1%)
the
(8.3%)
have
(5.5%)
.
(18.7%)
**
(18.7%)
.
(23.7%)
Answer
(18.7%)
the
(21.7%)
the
(20.7%)
answer
(31.8%)
is
(37.2%)
D
(14.6%)
.
(68.6%)
</think>
(100%)
(100%)
0.2
0.4
0.6
0.8
1.0
Accuracy
MATH
CoT Beginning
CoT End
<think>
(100%)
(100%)
Okay
(100%)
,
(92%)
let
(72%)
's
(64%)
try
(66.7%)
to
(87.5%)
tackle
(33.3%)
this
(62.5%)
problem
(62.5%)
step
(37.5%)
by
(37.5%)
step
(37.5%)
.
(37.5%)
First
(16.7%)
,
(25%)
.
(70.8%)
**
(70.8%)
Final
(70.8%)
Answer
(70.8%)
**
(70.8%)
\
(70.8%)
boxed
(68%)
{
(68%)
B
(24%)
}
(68%)
</think>
(100%)
(100%)
0.2
0.4
0.6
0.8
1.0
Accuracy
AIME
CoT Beginning
CoT End
Figure 17. Probing for final answers: averaged accuracy with Qwen3-32B along CoT positions. The most frequent token at each position
is annotated with its occurrence frequency (the remaining 6 tasks are shown in Figure 18).
28

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
<think>
(100%)
(100%)
Okay
(100%)
,
(100%)
let
(100%)
's
(97.4%)
try
(67.6%)
to
(67.6%)
figure
(66.8%)
out
(56.2%)
who
(31%)
the
(31.2%)
murderer
(28.2%)
is
(28.6%)
here
(26.4%)
.
(26.4%)
So
(14.2%)
.
(11.8%)
the
(9.8%)
.
(17.2%)
.
(12.6%)
is
(12.2%)
the
(16.4%)
answer
(21.4%)
is
(24.8%)
A
(15.2%)
.
(99.2%)
</think>
(100%)
(100%)
0.2
0.4
0.6
0.8
Accuracy
MuSR
CoT Beginning
CoT End
<think>
(100%)
(100%)
Okay
(100%)
,
(100%)
let
(100%)
's
(97.4%)
try
(91.6%)
to
(91.6%)
figure
(78.3%)
out
(45.7%)
out
(32.6%)
step
(30.9%)
by
(25.5%)
step
(30%)
.
(25.5%)
So
(17%)
,
(16.5%)
is
(11.6%)
is
(21%)
.
(16.3%)
.
(29.4%)
So
(26.4%)
the
(41.8%)
answer
(48.7%)
is
(49.4%)
B
(26.2%)
.
(96.8%)
</think>
(100%)
(100%)
0.2
0.4
0.6
0.8
Accuracy
Zebra
CoT Beginning
CoT End
<think>
(100%)
(100%)
Okay
(100%)
,
(100%)
let
(97.8%)
's
(97.8%)
see
(90.8%)
.
(90.8%)
The
(90%)
question
(90%)
is
(90%)
asking
(59.8%)
where
(26%)
a
(14.2%)
would
(5%)
a
(6%)
a
(9.2%)
.
(12.6%)
So
(11.2%)
.
(20.4%)
So
(17.2%)
the
(13.2%)
the
(18%)
answer
(23.8%)
is
(25.4%)
C
(10.8%)
.
(100%)
</think>
(100%)
(100%)
0.2
0.4
0.6
0.8
Accuracy
CSQA
CoT Beginning
CoT End
<think>
(100%)
(100%)
Okay
(100%)
,
(100%)
let
(93%)
's
(92.2%)
see
(74.6%)
.
(74.6%)
The
(69.8%)
question
(67.6%)
is
(68.8%)
about
(49.2%)
the
(17.8%)
the
(12.4%)
of
(7%)
of
(8.2%)
the
(5.8%)
the
(8.6%)
.
(15.8%)
.
(23.4%)
So
(18.2%)
the
(23.4%)
answer
(20.8%)
answer
(25.6%)
is
(30.2%)
C
(15.2%)
.
(99.6%)
</think>
(100%)
(100%)
0.4
0.6
0.8
Accuracy
MMLU
CoT Beginning
CoT End
<think>
(100%)
(100%)
Okay
(100%)
,
(100%)
let
(100%)
's
(96.6%)
see
(45%)
.
(45%)
The
(44.6%)
question
(41%)
is
(43%)
asking
(36.6%)
why
(13.2%)
the
(16%)
is
(8%)
the
(8.6%)
the
(9.8%)
the
(6.4%)
.
(11%)
.
(28.6%)
So
(24.4%)
the
(24.6%)
answer
(24%)
answer
(25%)
is
(27.4%)
A
(16.4%)
.
(99.8%)
</think>
(100%)
(100%)
0.4
0.6
0.8
Accuracy
QuALITY
CoT Beginning
CoT End
<think>
(100%)
(100%)
Okay
(100%)
,
(98.8%)
let
(97.6%)
's
(96%)
try
(64.2%)
to
(65.2%)
figure
(58.9%)
out
(41.8%)
this
(37.3%)
problem
(23.6%)
.
(29.3%)
So
(18.3%)
,
(18.1%)
is
(11.8%)
question
(10.4%)
.
(11.4%)
.
(15.5%)
.
(20.8%)
,
(22.4%)
the
(31.2%)
the
(23.6%)
answer
(46.2%)
is
(48.4%)
C
(20.4%)
.
(98.2%)
</think>
(100%)
(100%)
0.2
0.4
0.6
0.8
Accuracy
GPQA
CoT Beginning
CoT End
Figure 18. Probing for final answers: averaged accuracy by Qwen3-32B along CoT positions. The most frequent token at each position is
annotated with its occurrence frequency. Result discussions are addressed near Figure 3.
29

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
Parity
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
Cycle
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
Subsum
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
GSM8K
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
MATH
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
AIME
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
MuSR
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
Zebra
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
CSQA
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
MMLU
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
QuALITY
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
GPQA
Figure 19. Task accuracy comparison for Off-the-Shelf LLM (Qwen3-32B) under four settings: using thinking mode (w/ CoT); using
non-thinking mode (w/o CoT); the best probing accuracy among initial CoT positions (Probing); the random-guess baseline (Random).
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
Parity
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
Cycle
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
Subsum
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
GSM8K
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
MATH
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
AIME
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
MuSR
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
Zebra
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
CSQA
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
MMLU
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
QuALITY
w/
CoT
w/o
CoT
Probing
Random
0.0
0.2
0.4
0.6
0.8
1.0
Accuracy
GPQA
Figure 20. Task accuracy comparison for In-Domain LLM under four settings: standard inference with learned CoT (w/ CoT); direct
prediction by a separately trained model via naive supervised finetuning, without CoT learned (w/o CoT); the best probing accuracy
among initial CoT positions (Probing); the random-guess baseline (Random). Result discussions are addressed near Figure 4.
30

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
1
2
3
4
5
6
7
8
Subsequent Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.88
0.86
0.84
0.83
0.83
0.82
0.8
0.8
0.9
0.88
0.88
0.86
0.85
0.85
0.84
0.83
0.91
0.89
0.88
0.87
0.86
0.86
0.85
0.84
0.91
0.9
0.89
0.88
0.88
0.86
0.86
0.85
0.91
0.9
0.89
0.88
0.87
0.86
0.86
0.85
0.94
0.91
0.9
0.89
0.89
0.88
0.87
0.87
0.95
0.91
0.9
0.89
0.89
0.88
0.87
0.87
0.94
0.91
0.9
0.89
0.89
0.88
0.88
0.87
0.94
0.91
0.9
0.89
0.89
0.88
0.88
0.88
0.94
0.91
0.9
0.89
0.89
0.89
0.88
0.88
0.94
0.91
0.9
0.89
0.89
0.88
0.88
0.88
0.94
0.91
0.9
0.9
0.89
0.89
0.88
0.88
0.94
0.91
0.9
0.89
0.89
0.89
0.88
0.88
0.94
0.91
0.9
0.9
0.89
0.89
0.88
0.88
0.98
0.93
0.92
0.9
0.89
0.89
0.89
0.88
0.98
0.93
0.92
0.9
0.9
0.89
0.89
0.89
0.98
0.94
0.93
0.91
0.9
0.9
0.89
0.89
0.99
0.97
0.95
0.92
0.91
0.91
0.9
0.89
0.99
0.96
0.94
0.92
0.91
0.91
0.9
0.89
0.99
0.97
0.95
0.93
0.91
0.91
0.9
0.89
0.99
0.97
0.95
0.93
0.91
0.91
0.9
0.89
1
0.97
0.95
0.94
0.92
0.91
0.9
0.9
1
0.99
0.97
0.96
0.93
0.91
0.9
0.89
1
1
0.97
0.96
0.92
0.91
0.89
0.89
1
1
0.97
0.96
0.93
0.91
0.89
0.89
1
1
0.97
0.96
0.92
0.9
0.89
0.89
1
0.99
0.97
0.95
0.91
0.89
0.88
0.87
1
1
0.97
0.96
0.92
0.9
0.89
0.89
Parity
1
2
3
4
5
6
7
8
Subsequent Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.74
0.7
0.67
0.65
0.64
0.65
0.63
0.67
0.77
0.73
0.71
0.68
0.69
0.7
0.71
0.7
0.79
0.75
0.73
0.71
0.7
0.72
0.72
0.72
0.79
0.76
0.74
0.73
0.73
0.75
0.76
0.75
0.79
0.76
0.74
0.73
0.73
0.74
0.76
0.75
0.89
0.78
0.76
0.75
0.74
0.75
0.77
0.76
0.88
0.78
0.76
0.75
0.75
0.76
0.77
0.78
0.88
0.78
0.76
0.76
0.75
0.77
0.78
0.78
0.87
0.78
0.77
0.76
0.75
0.77
0.78
0.78
0.87
0.79
0.77
0.76
0.76
0.76
0.78
0.78
0.87
0.79
0.77
0.76
0.76
0.76
0.78
0.78
0.88
0.79
0.77
0.76
0.76
0.77
0.78
0.77
0.87
0.79
0.77
0.76
0.76
0.77
0.78
0.79
0.87
0.82
0.82
0.81
0.8
0.8
0.81
0.8
0.95
0.82
0.81
0.79
0.78
0.78
0.8
0.79
0.95
0.83
0.81
0.79
0.78
0.79
0.8
0.8
0.95
0.83
0.83
0.81
0.79
0.79
0.8
0.79
0.95
0.85
0.85
0.83
0.81
0.81
0.81
0.8
0.96
0.86
0.85
0.84
0.82
0.81
0.81
0.8
0.96
0.89
0.88
0.86
0.84
0.82
0.81
0.81
0.97
0.89
0.88
0.86
0.84
0.82
0.81
0.81
0.98
0.92
0.9
0.87
0.84
0.82
0.81
0.81
1
0.98
0.95
0.92
0.85
0.82
0.81
0.81
1
0.99
0.94
0.9
0.82
0.79
0.79
0.8
1
0.99
0.94
0.9
0.82
0.8
0.8
0.8
1
0.99
0.93
0.88
0.8
0.78
0.78
0.78
1
0.99
0.91
0.86
0.78
0.76
0.76
0.74
1
0.99
0.93
0.88
0.81
0.78
0.79
0.78
Cycle
1
2
3
4
5
6
7
8
Subsequent Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.87
0.84
0.8
0.8
0.78
0.75
0.72
0.73
0.89
0.87
0.86
0.85
0.84
0.81
0.8
0.78
0.91
0.88
0.87
0.86
0.84
0.84
0.82
0.82
0.92
0.89
0.89
0.88
0.87
0.86
0.84
0.84
0.92
0.89
0.89
0.88
0.86
0.86
0.84
0.83
0.95
0.9
0.9
0.89
0.89
0.88
0.87
0.86
0.95
0.91
0.9
0.9
0.89
0.89
0.88
0.87
0.95
0.91
0.91
0.9
0.9
0.89
0.88
0.88
0.95
0.91
0.91
0.9
0.9
0.89
0.89
0.88
0.95
0.91
0.91
0.9
0.9
0.89
0.88
0.88
0.95
0.91
0.91
0.9
0.91
0.89
0.89
0.88
0.95
0.9
0.91
0.9
0.9
0.89
0.88
0.89
0.95
0.92
0.91
0.91
0.91
0.89
0.89
0.88
0.95
0.91
0.92
0.92
0.91
0.9
0.9
0.89
0.97
0.93
0.92
0.92
0.92
0.9
0.9
0.89
0.98
0.94
0.92
0.92
0.91
0.91
0.9
0.9
0.98
0.94
0.93
0.92
0.93
0.92
0.9
0.9
0.98
0.95
0.94
0.93
0.93
0.92
0.91
0.9
0.98
0.96
0.94
0.93
0.93
0.92
0.91
0.9
0.99
0.97
0.95
0.93
0.93
0.92
0.91
0.91
0.99
0.97
0.96
0.94
0.93
0.92
0.92
0.91
0.99
0.98
0.97
0.94
0.93
0.92
0.92
0.91
1
0.99
0.99
0.94
0.94
0.93
0.92
0.92
1
1
0.99
0.94
0.94
0.93
0.92
0.91
1
1
0.99
0.94
0.94
0.92
0.92
0.91
1
1
0.99
0.94
0.93
0.92
0.91
0.9
1
1
0.98
0.93
0.91
0.9
0.89
0.88
1
1
0.99
0.94
0.93
0.92
0.92
0.91
Subsum
1
2
3
4
5
6
7
8
Subsequent Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.57
0.4
0.3
0.25
0.21
0.2
0.19
0.19
0.62
0.46
0.36
0.31
0.28
0.25
0.24
0.22
0.66
0.5
0.39
0.33
0.29
0.27
0.25
0.24
0.68
0.53
0.42
0.36
0.31
0.29
0.27
0.26
0.68
0.53
0.43
0.36
0.33
0.3
0.27
0.25
0.72
0.58
0.47
0.39
0.35
0.32
0.29
0.28
0.73
0.59
0.47
0.41
0.36
0.33
0.31
0.28
0.73
0.59
0.48
0.41
0.37
0.34
0.31
0.29
0.74
0.6
0.48
0.42
0.38
0.34
0.32
0.29
0.74
0.6
0.49
0.41
0.37
0.34
0.31
0.3
0.75
0.61
0.49
0.42
0.38
0.34
0.32
0.3
0.75
0.61
0.5
0.42
0.38
0.35
0.32
0.29
0.75
0.62
0.5
0.43
0.38
0.35
0.32
0.3
0.75
0.62
0.51
0.43
0.39
0.35
0.33
0.31
0.77
0.63
0.52
0.44
0.4
0.36
0.33
0.3
0.78
0.64
0.52
0.45
0.4
0.37
0.34
0.31
0.78
0.64
0.53
0.45
0.4
0.37
0.34
0.31
0.79
0.65
0.53
0.46
0.41
0.37
0.34
0.32
0.8
0.67
0.55
0.47
0.42
0.38
0.35
0.32
0.81
0.69
0.57
0.48
0.43
0.39
0.36
0.33
0.82
0.7
0.58
0.49
0.43
0.39
0.36
0.32
0.85
0.72
0.6
0.5
0.44
0.39
0.35
0.32
0.91
0.76
0.62
0.51
0.43
0.39
0.35
0.32
0.94
0.77
0.62
0.5
0.42
0.38
0.34
0.3
0.95
0.78
0.62
0.5
0.42
0.38
0.34
0.3
0.95
0.76
0.59
0.47
0.4
0.35
0.32
0.29
0.94
0.73
0.55
0.43
0.37
0.33
0.3
0.28
0.97
0.77
0.59
0.47
0.4
0.36
0.33
0.3
GSM8K
1
2
3
4
5
6
7
8
Subsequent Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.59
0.41
0.31
0.26
0.22
0.21
0.2
0.2
0.63
0.44
0.34
0.29
0.25
0.24
0.22
0.21
0.68
0.49
0.36
0.31
0.26
0.25
0.23
0.22
0.7
0.51
0.39
0.32
0.28
0.26
0.24
0.23
0.7
0.51
0.39
0.33
0.29
0.27
0.24
0.23
0.75
0.57
0.43
0.35
0.3
0.28
0.25
0.25
0.76
0.57
0.43
0.37
0.31
0.28
0.27
0.25
0.76
0.57
0.44
0.36
0.32
0.29
0.27
0.25
0.76
0.58
0.43
0.37
0.32
0.29
0.27
0.26
0.77
0.58
0.45
0.37
0.32
0.3
0.28
0.27
0.78
0.6
0.45
0.38
0.33
0.3
0.28
0.27
0.77
0.58
0.45
0.38
0.33
0.3
0.28
0.26
0.78
0.59
0.45
0.38
0.33
0.31
0.28
0.26
0.77
0.6
0.46
0.38
0.34
0.31
0.29
0.27
0.8
0.61
0.47
0.39
0.34
0.31
0.29
0.27
0.8
0.61
0.47
0.39
0.34
0.31
0.3
0.28
0.8
0.61
0.48
0.4
0.35
0.32
0.3
0.28
0.81
0.64
0.49
0.41
0.36
0.32
0.3
0.28
0.82
0.65
0.5
0.42
0.36
0.32
0.3
0.28
0.84
0.68
0.53
0.44
0.38
0.34
0.31
0.29
0.85
0.7
0.54
0.45
0.38
0.35
0.31
0.29
0.87
0.72
0.57
0.46
0.39
0.35
0.32
0.29
0.93
0.78
0.61
0.48
0.4
0.35
0.32
0.29
0.94
0.8
0.62
0.47
0.39
0.34
0.32
0.28
0.95
0.81
0.62
0.47
0.39
0.34
0.31
0.28
0.96
0.79
0.6
0.45
0.37
0.32
0.3
0.27
0.96
0.77
0.55
0.42
0.35
0.32
0.29
0.27
0.97
0.8
0.6
0.45
0.37
0.33
0.31
0.28
MATH
1
2
3
4
5
6
7
8
Subsequent Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.52
0.36
0.26
0.22
0.19
0.19
0.18
0.18
0.56
0.39
0.29
0.24
0.22
0.21
0.2
0.19
0.6
0.42
0.32
0.26
0.22
0.23
0.21
0.2
0.62
0.43
0.33
0.27
0.23
0.23
0.21
0.21
0.62
0.43
0.33
0.27
0.24
0.23
0.21
0.2
0.68
0.49
0.36
0.3
0.25
0.24
0.22
0.22
0.68
0.49
0.36
0.3
0.26
0.24
0.23
0.22
0.68
0.49
0.36
0.3
0.26
0.24
0.23
0.22
0.69
0.48
0.36
0.29
0.26
0.24
0.23
0.22
0.69
0.5
0.37
0.3
0.27
0.25
0.24
0.23
0.7
0.51
0.38
0.31
0.27
0.25
0.24
0.23
0.69
0.49
0.37
0.31
0.27
0.26
0.24
0.22
0.7
0.49
0.37
0.31
0.27
0.26
0.24
0.22
0.69
0.5
0.38
0.31
0.28
0.26
0.25
0.23
0.71
0.52
0.39
0.31
0.28
0.26
0.24
0.23
0.72
0.52
0.39
0.32
0.28
0.26
0.25
0.23
0.72
0.52
0.38
0.32
0.29
0.27
0.25
0.23
0.73
0.55
0.41
0.33
0.3
0.26
0.25
0.23
0.75
0.56
0.42
0.34
0.3
0.27
0.25
0.25
0.76
0.58
0.43
0.34
0.31
0.29
0.26
0.24
0.77
0.59
0.45
0.37
0.32
0.28
0.26
0.25
0.8
0.62
0.47
0.36
0.32
0.28
0.25
0.24
0.87
0.68
0.5
0.38
0.33
0.29
0.26
0.24
0.89
0.69
0.51
0.39
0.32
0.29
0.27
0.24
0.91
0.71
0.51
0.38
0.32
0.28
0.26
0.24
0.91
0.68
0.49
0.36
0.3
0.27
0.25
0.23
0.9
0.66
0.45
0.34
0.29
0.26
0.25
0.23
0.92
0.7
0.49
0.36
0.31
0.28
0.26
0.24
AIME
1
2
3
4
5
6
7
8
Subsequent Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.36
0.26
0.19
0.16
0.14
0.12
0.11
0.13
0.39
0.29
0.23
0.19
0.16
0.14
0.13
0.14
0.41
0.3
0.24
0.2
0.16
0.15
0.15
0.14
0.43
0.31
0.24
0.21
0.17
0.16
0.15
0.15
0.43
0.32
0.26
0.22
0.19
0.18
0.15
0.15
0.46
0.34
0.27
0.23
0.19
0.18
0.17
0.16
0.47
0.35
0.27
0.23
0.2
0.18
0.17
0.16
0.48
0.35
0.27
0.23
0.2
0.18
0.18
0.16
0.49
0.36
0.28
0.23
0.2
0.19
0.17
0.16
0.49
0.36
0.27
0.24
0.21
0.19
0.18
0.17
0.5
0.37
0.29
0.24
0.22
0.2
0.18
0.17
0.5
0.37
0.29
0.25
0.21
0.2
0.19
0.17
0.51
0.37
0.29
0.25
0.22
0.2
0.19
0.18
0.52
0.38
0.29
0.25
0.22
0.21
0.19
0.17
0.53
0.38
0.29
0.25
0.23
0.2
0.19
0.17
0.54
0.39
0.3
0.26
0.23
0.21
0.2
0.19
0.53
0.39
0.3
0.26
0.24
0.21
0.2
0.19
0.55
0.4
0.31
0.26
0.24
0.21
0.2
0.19
0.57
0.42
0.32
0.28
0.25
0.23
0.21
0.2
0.58
0.43
0.32
0.27
0.25
0.22
0.21
0.19
0.59
0.44
0.33
0.28
0.25
0.23
0.21
0.19
0.62
0.45
0.33
0.28
0.25
0.22
0.21
0.19
0.69
0.46
0.34
0.28
0.24
0.22
0.2
0.19
0.72
0.45
0.33
0.27
0.23
0.21
0.2
0.18
0.74
0.45
0.31
0.26
0.23
0.21
0.19
0.18
0.74
0.43
0.3
0.25
0.22
0.2
0.19
0.17
0.7
0.39
0.29
0.24
0.21
0.19
0.18
0.16
0.75
0.43
0.3
0.24
0.22
0.2
0.18
0.17
MuSR
1
2
3
4
5
6
7
8
Subsequent Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.59
0.42
0.31
0.25
0.23
0.19
0.19
0.2
0.66
0.49
0.37
0.31
0.29
0.26
0.23
0.22
0.69
0.51
0.41
0.33
0.3
0.27
0.26
0.25
0.72
0.55
0.44
0.36
0.32
0.29
0.27
0.26
0.72
0.56
0.44
0.37
0.33
0.3
0.27
0.26
0.76
0.6
0.49
0.4
0.36
0.32
0.3
0.27
0.77
0.61
0.5
0.42
0.37
0.34
0.31
0.28
0.77
0.6
0.5
0.42
0.38
0.33
0.31
0.28
0.77
0.61
0.5
0.42
0.37
0.34
0.32
0.29
0.78
0.62
0.5
0.42
0.38
0.34
0.32
0.29
0.78
0.62
0.51
0.43
0.38
0.35
0.32
0.3
0.78
0.61
0.5
0.42
0.38
0.35
0.32
0.29
0.79
0.62
0.51
0.43
0.38
0.35
0.32
0.3
0.78
0.63
0.53
0.44
0.39
0.35
0.33
0.3
0.8
0.63
0.53
0.45
0.39
0.36
0.33
0.3
0.81
0.64
0.53
0.45
0.39
0.37
0.34
0.31
0.81
0.65
0.54
0.46
0.4
0.37
0.34
0.31
0.82
0.67
0.54
0.47
0.41
0.38
0.35
0.32
0.83
0.68
0.56
0.48
0.42
0.39
0.35
0.32
0.85
0.71
0.58
0.49
0.43
0.38
0.36
0.33
0.87
0.72
0.6
0.49
0.43
0.39
0.36
0.33
0.89
0.74
0.6
0.51
0.43
0.39
0.36
0.33
0.94
0.78
0.63
0.52
0.44
0.39
0.36
0.33
0.95
0.78
0.62
0.5
0.42
0.37
0.34
0.31
0.96
0.79
0.62
0.5
0.42
0.38
0.35
0.32
0.96
0.77
0.59
0.47
0.39
0.35
0.32
0.3
0.95
0.74
0.56
0.45
0.37
0.34
0.31
0.29
0.96
0.79
0.61
0.48
0.41
0.36
0.34
0.31
Zebra
1
2
3
4
5
6
7
8
Subsequent Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.43
0.32
0.25
0.2
0.18
0.17
0.16
0.16
0.49
0.37
0.29
0.25
0.22
0.2
0.19
0.19
0.51
0.38
0.3
0.26
0.24
0.22
0.21
0.19
0.53
0.4
0.32
0.28
0.26
0.23
0.22
0.21
0.54
0.41
0.32
0.28
0.25
0.24
0.22
0.21
0.57
0.43
0.35
0.3
0.27
0.25
0.23
0.22
0.58
0.44
0.36
0.31
0.28
0.27
0.24
0.23
0.58
0.44
0.35
0.3
0.28
0.26
0.24
0.23
0.58
0.45
0.36
0.31
0.28
0.26
0.25
0.23
0.59
0.45
0.36
0.32
0.29
0.26
0.25
0.23
0.6
0.46
0.36
0.32
0.3
0.27
0.25
0.23
0.6
0.46
0.37
0.32
0.3
0.27
0.25
0.23
0.6
0.46
0.37
0.33
0.29
0.26
0.25
0.24
0.6
0.46
0.38
0.33
0.3
0.28
0.26
0.24
0.61
0.47
0.38
0.33
0.3
0.28
0.26
0.24
0.62
0.48
0.39
0.34
0.31
0.29
0.27
0.25
0.62
0.48
0.39
0.34
0.32
0.29
0.27
0.25
0.64
0.49
0.4
0.35
0.32
0.3
0.27
0.25
0.66
0.52
0.41
0.36
0.33
0.3
0.28
0.26
0.67
0.53
0.42
0.36
0.32
0.3
0.28
0.26
0.68
0.54
0.42
0.36
0.33
0.3
0.28
0.26
0.71
0.55
0.42
0.35
0.32
0.29
0.27
0.24
0.78
0.56
0.42
0.35
0.31
0.29
0.26
0.24
0.8
0.56
0.41
0.34
0.3
0.28
0.25
0.23
0.81
0.56
0.4
0.33
0.3
0.28
0.25
0.23
0.82
0.54
0.38
0.32
0.29
0.27
0.25
0.23
0.8
0.51
0.36
0.3
0.27
0.25
0.23
0.21
0.85
0.54
0.39
0.32
0.28
0.27
0.25
0.23
CSQA
1
2
3
4
5
6
7
8
Subsequent Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.4
0.28
0.22
0.18
0.16
0.14
0.14
0.15
0.44
0.31
0.25
0.21
0.19
0.17
0.17
0.17
0.47
0.33
0.26
0.22
0.2
0.18
0.18
0.17
0.48
0.35
0.27
0.24
0.21
0.2
0.18
0.18
0.48
0.35
0.28
0.24
0.21
0.2
0.19
0.18
0.52
0.38
0.3
0.25
0.23
0.21
0.2
0.19
0.53
0.39
0.3
0.26
0.23
0.22
0.21
0.2
0.53
0.39
0.31
0.26
0.23
0.22
0.21
0.19
0.54
0.39
0.3
0.26
0.24
0.22
0.21
0.2
0.54
0.39
0.31
0.27
0.24
0.22
0.21
0.2
0.54
0.4
0.31
0.27
0.24
0.22
0.21
0.2
0.54
0.4
0.31
0.27
0.24
0.23
0.22
0.2
0.55
0.4
0.31
0.27
0.24
0.22
0.21
0.2
0.55
0.41
0.32
0.28
0.25
0.23
0.22
0.21
0.56
0.41
0.33
0.28
0.25
0.24
0.22
0.2
0.57
0.42
0.33
0.28
0.26
0.24
0.23
0.21
0.57
0.42
0.33
0.29
0.26
0.24
0.23
0.21
0.58
0.43
0.34
0.29
0.27
0.25
0.23
0.22
0.6
0.45
0.35
0.3
0.27
0.25
0.23
0.22
0.61
0.46
0.36
0.31
0.27
0.25
0.24
0.22
0.62
0.47
0.37
0.31
0.27
0.25
0.23
0.22
0.65
0.48
0.37
0.3
0.27
0.25
0.23
0.21
0.73
0.5
0.37
0.3
0.26
0.24
0.22
0.21
0.75
0.51
0.37
0.29
0.26
0.24
0.22
0.2
0.77
0.51
0.36
0.29
0.26
0.24
0.22
0.2
0.78
0.48
0.34
0.28
0.25
0.23
0.21
0.2
0.75
0.46
0.33
0.27
0.24
0.22
0.21
0.2
0.8
0.49
0.35
0.28
0.25
0.23
0.21
0.2
MMLU
1
2
3
4
5
6
7
8
Subsequent Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.35
0.26
0.21
0.17
0.14
0.12
0.12
0.12
0.38
0.28
0.23
0.19
0.15
0.14
0.13
0.13
0.4
0.29
0.24
0.2
0.16
0.15
0.14
0.13
0.42
0.3
0.24
0.2
0.17
0.16
0.15
0.15
0.41
0.3
0.25
0.21
0.18
0.17
0.14
0.15
0.45
0.32
0.26
0.22
0.18
0.17
0.15
0.15
0.46
0.33
0.27
0.23
0.19
0.19
0.17
0.16
0.47
0.33
0.26
0.22
0.19
0.18
0.17
0.16
0.47
0.33
0.26
0.22
0.19
0.18
0.16
0.16
0.48
0.33
0.27
0.23
0.2
0.18
0.17
0.16
0.48
0.34
0.27
0.23
0.2
0.19
0.17
0.17
0.49
0.34
0.27
0.24
0.2
0.19
0.18
0.17
0.49
0.34
0.28
0.24
0.2
0.18
0.18
0.16
0.49
0.34
0.28
0.23
0.21
0.19
0.18
0.17
0.5
0.35
0.28
0.24
0.21
0.19
0.18
0.16
0.51
0.36
0.29
0.25
0.22
0.2
0.19
0.18
0.51
0.36
0.29
0.25
0.23
0.21
0.19
0.18
0.52
0.37
0.3
0.25
0.23
0.2
0.19
0.18
0.54
0.38
0.31
0.26
0.22
0.21
0.2
0.18
0.56
0.39
0.31
0.26
0.23
0.21
0.19
0.18
0.57
0.4
0.31
0.26
0.23
0.21
0.19
0.18
0.6
0.41
0.31
0.26
0.23
0.21
0.19
0.18
0.7
0.42
0.31
0.25
0.22
0.2
0.18
0.17
0.73
0.42
0.3
0.25
0.22
0.2
0.18
0.17
0.75
0.41
0.29
0.24
0.21
0.19
0.18
0.17
0.75
0.39
0.28
0.24
0.2
0.19
0.18
0.16
0.72
0.37
0.27
0.23
0.2
0.19
0.17
0.16
0.77
0.4
0.28
0.23
0.21
0.19
0.17
0.16
QuALITY
1
2
3
4
5
6
7
8
Subsequent Position
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
Layer
0.39
0.26
0.2
0.16
0.15
0.13
0.14
0.14
0.42
0.28
0.23
0.19
0.17
0.16
0.15
0.15
0.46
0.31
0.24
0.2
0.18
0.17
0.16
0.15
0.47
0.33
0.25
0.21
0.19
0.18
0.16
0.16
0.47
0.33
0.25
0.21
0.19
0.18
0.17
0.16
0.51
0.36
0.27
0.23
0.2
0.19
0.17
0.17
0.52
0.36
0.27
0.23
0.2
0.19
0.18
0.17
0.52
0.36
0.28
0.24
0.21
0.19
0.19
0.17
0.52
0.36
0.28
0.24
0.21
0.19
0.19
0.18
0.52
0.36
0.28
0.24
0.21
0.19
0.18
0.18
0.52
0.37
0.29
0.24
0.22
0.2
0.19
0.18
0.52
0.37
0.28
0.24
0.22
0.2
0.19
0.18
0.52
0.37
0.28
0.24
0.21
0.2
0.19
0.18
0.52
0.37
0.29
0.25
0.22
0.2
0.19
0.18
0.55
0.38
0.29
0.25
0.22
0.21
0.19
0.18
0.55
0.38
0.29
0.25
0.23
0.21
0.2
0.19
0.55
0.38
0.3
0.26
0.23
0.21
0.2
0.19
0.56
0.4
0.3
0.26
0.23
0.21
0.2
0.19
0.58
0.41
0.32
0.27
0.24
0.22
0.2
0.2
0.58
0.42
0.32
0.27
0.24
0.23
0.21
0.2
0.59
0.43
0.33
0.28
0.25
0.23
0.21
0.19
0.62
0.45
0.34
0.27
0.24
0.22
0.2
0.19
0.72
0.48
0.35
0.28
0.24
0.22
0.2
0.19
0.74
0.49
0.35
0.28
0.24
0.21
0.2
0.18
0.76
0.49
0.34
0.26
0.23
0.21
0.19
0.18
0.76
0.47
0.33
0.26
0.22
0.21
0.19
0.18
0.74
0.44
0.31
0.25
0.22
0.2
0.19
0.18
0.79
0.47
0.32
0.25
0.22
0.2
0.19
0.18
GPQA
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
Figure 21. Averaged Top-5 Accuracy of subsequent token prediction with In-Domain LLM, across Transformers layers and subsequent
positions (up to the 8th following position). Result discussions are addressed near Figure 5.
31

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
1
2
3
4
5
6
7
8
Subsequent Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.76
0.62
0.52
0.47
0.47
0.45
0.45
0.44
0.75
0.61
0.53
0.47
0.46
0.46
0.44
0.43
0.75
0.61
0.49
0.46
0.42
0.43
0.41
0.41
0.75
0.6
0.51
0.45
0.43
0.43
0.42
0.42
0.78
0.62
0.52
0.47
0.45
0.44
0.43
0.42
0.77
0.6
0.49
0.44
0.43
0.43
0.41
0.4
0.8
0.62
0.51
0.45
0.43
0.44
0.42
0.41
0.84
0.68
0.52
0.47
0.42
0.42
0.42
0.41
0.9
0.71
0.53
0.44
0.39
0.41
0.4
0.38
0.9
0.69
0.48
0.4
0.39
0.39
0.39
0.39
0.97
0.84
0.67
0.51
0.45
0.45
0.44
0.43
Parity
1
2
3
4
5
6
7
8
Subsequent Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.76
0.66
0.6
0.56
0.53
0.51
0.48
0.44
0.76
0.67
0.6
0.56
0.54
0.52
0.48
0.46
0.75
0.66
0.6
0.55
0.52
0.5
0.46
0.43
0.76
0.67
0.6
0.55
0.53
0.5
0.47
0.45
0.82
0.68
0.62
0.57
0.54
0.51
0.48
0.45
0.8
0.69
0.61
0.56
0.53
0.52
0.48
0.45
0.84
0.71
0.62
0.56
0.53
0.49
0.46
0.43
0.85
0.75
0.65
0.59
0.55
0.52
0.49
0.43
0.94
0.81
0.69
0.57
0.5
0.47
0.44
0.38
0.93
0.79
0.65
0.52
0.45
0.39
0.38
0.33
0.98
0.9
0.77
0.65
0.58
0.54
0.5
0.46
Cycle
1
2
3
4
5
6
7
8
Subsequent Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.79
0.64
0.54
0.48
0.44
0.4
0.38
0.36
0.8
0.66
0.56
0.48
0.45
0.42
0.39
0.37
0.79
0.65
0.54
0.47
0.41
0.4
0.36
0.34
0.79
0.65
0.55
0.47
0.42
0.38
0.36
0.35
0.82
0.67
0.56
0.49
0.44
0.41
0.38
0.36
0.82
0.66
0.56
0.48
0.44
0.4
0.37
0.35
0.85
0.68
0.56
0.47
0.43
0.41
0.36
0.34
0.88
0.74
0.59
0.5
0.45
0.4
0.37
0.34
0.93
0.78
0.58
0.45
0.39
0.36
0.33
0.3
0.92
0.74
0.52
0.39
0.33
0.33
0.3
0.29
0.98
0.87
0.7
0.55
0.47
0.43
0.4
0.36
Subsum
1
2
3
4
5
6
7
8
Subsequent Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.66
0.51
0.4
0.35
0.32
0.3
0.29
0.29
0.66
0.5
0.41
0.36
0.33
0.31
0.3
0.29
0.68
0.51
0.4
0.35
0.33
0.3
0.3
0.29
0.68
0.51
0.41
0.35
0.32
0.31
0.3
0.28
0.7
0.53
0.41
0.36
0.33
0.31
0.3
0.29
0.69
0.52
0.41
0.36
0.33
0.32
0.3
0.29
0.71
0.54
0.43
0.37
0.34
0.32
0.31
0.29
0.74
0.58
0.45
0.38
0.35
0.33
0.31
0.29
0.8
0.6
0.44
0.37
0.33
0.31
0.3
0.28
0.79
0.57
0.41
0.34
0.31
0.3
0.29
0.27
0.92
0.71
0.53
0.42
0.36
0.33
0.31
0.3
GSM8K
1
2
3
4
5
6
7
8
Subsequent Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.64
0.45
0.35
0.3
0.28
0.27
0.26
0.26
0.64
0.45
0.35
0.3
0.28
0.27
0.26
0.26
0.66
0.46
0.34
0.3
0.28
0.27
0.26
0.25
0.66
0.46
0.35
0.3
0.28
0.27
0.27
0.25
0.68
0.47
0.36
0.31
0.29
0.27
0.27
0.26
0.67
0.47
0.35
0.31
0.29
0.27
0.26
0.26
0.69
0.49
0.37
0.32
0.29
0.28
0.27
0.26
0.73
0.54
0.39
0.33
0.3
0.29
0.27
0.26
0.8
0.57
0.4
0.32
0.29
0.28
0.27
0.26
0.8
0.53
0.36
0.3
0.28
0.27
0.26
0.25
0.93
0.69
0.49
0.36
0.31
0.29
0.28
0.27
MATH
1
2
3
4
5
6
7
8
Subsequent Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.61
0.4
0.32
0.27
0.26
0.23
0.23
0.23
0.6
0.41
0.32
0.28
0.26
0.23
0.23
0.23
0.62
0.43
0.32
0.27
0.26
0.23
0.23
0.22
0.62
0.43
0.33
0.27
0.24
0.23
0.23
0.22
0.64
0.44
0.33
0.29
0.26
0.24
0.25
0.23
0.64
0.44
0.33
0.28
0.26
0.23
0.23
0.22
0.67
0.46
0.33
0.28
0.27
0.24
0.23
0.23
0.7
0.49
0.37
0.3
0.27
0.25
0.24
0.23
0.79
0.53
0.37
0.32
0.26
0.24
0.23
0.22
0.79
0.5
0.33
0.26
0.25
0.23
0.23
0.22
0.91
0.66
0.46
0.33
0.28
0.26
0.24
0.23
AIME
1
2
3
4
5
6
7
8
Subsequent Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.44
0.3
0.24
0.21
0.19
0.18
0.18
0.18
0.44
0.3
0.24
0.21
0.2
0.19
0.19
0.19
0.46
0.31
0.23
0.21
0.19
0.18
0.18
0.18
0.46
0.31
0.24
0.22
0.2
0.19
0.19
0.18
0.48
0.32
0.25
0.22
0.21
0.2
0.19
0.19
0.48
0.33
0.25
0.22
0.21
0.2
0.19
0.19
0.49
0.34
0.26
0.23
0.22
0.21
0.2
0.19
0.52
0.35
0.26
0.23
0.22
0.21
0.2
0.2
0.55
0.34
0.24
0.22
0.2
0.19
0.19
0.18
0.55
0.3
0.22
0.2
0.19
0.18
0.17
0.17
0.75
0.41
0.27
0.23
0.2
0.19
0.19
0.18
MuSR
1
2
3
4
5
6
7
8
Subsequent Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.75
0.56
0.42
0.35
0.3
0.28
0.27
0.26
0.75
0.57
0.43
0.35
0.31
0.28
0.27
0.26
0.75
0.57
0.43
0.34
0.3
0.28
0.26
0.25
0.75
0.57
0.43
0.35
0.3
0.28
0.26
0.25
0.77
0.59
0.45
0.36
0.31
0.29
0.27
0.26
0.77
0.59
0.44
0.36
0.31
0.29
0.27
0.26
0.8
0.6
0.46
0.37
0.32
0.3
0.28
0.26
0.86
0.67
0.51
0.4
0.34
0.3
0.27
0.26
0.91
0.67
0.48
0.37
0.31
0.27
0.26
0.24
0.89
0.6
0.41
0.3
0.27
0.25
0.24
0.24
0.97
0.8
0.59
0.44
0.35
0.3
0.28
0.26
Zebra
1
2
3
4
5
6
7
8
Subsequent Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.53
0.36
0.29
0.25
0.22
0.21
0.2
0.2
0.53
0.37
0.3
0.26
0.24
0.23
0.22
0.21
0.54
0.37
0.29
0.26
0.23
0.22
0.21
0.2
0.54
0.37
0.3
0.26
0.23
0.22
0.21
0.2
0.56
0.39
0.3
0.26
0.24
0.23
0.22
0.21
0.56
0.39
0.3
0.27
0.25
0.23
0.21
0.21
0.58
0.41
0.32
0.28
0.25
0.24
0.23
0.22
0.61
0.43
0.33
0.28
0.25
0.24
0.22
0.22
0.63
0.42
0.3
0.26
0.24
0.23
0.22
0.21
0.63
0.38
0.27
0.25
0.23
0.22
0.2
0.2
0.81
0.5
0.34
0.28
0.24
0.23
0.22
0.21
CSQA
1
2
3
4
5
6
7
8
Subsequent Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.48
0.33
0.26
0.22
0.2
0.19
0.19
0.19
0.47
0.33
0.27
0.23
0.21
0.2
0.2
0.2
0.49
0.33
0.25
0.23
0.21
0.19
0.19
0.19
0.49
0.33
0.26
0.22
0.2
0.19
0.19
0.18
0.5
0.34
0.26
0.23
0.21
0.2
0.2
0.19
0.51
0.35
0.27
0.23
0.22
0.21
0.2
0.19
0.53
0.37
0.28
0.24
0.23
0.22
0.21
0.2
0.55
0.39
0.29
0.25
0.23
0.22
0.21
0.2
0.58
0.38
0.27
0.24
0.22
0.21
0.2
0.19
0.59
0.35
0.25
0.23
0.21
0.2
0.19
0.18
0.75
0.46
0.31
0.25
0.22
0.21
0.2
0.2
MMLU
1
2
3
4
5
6
7
8
Subsequent Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.46
0.31
0.24
0.2
0.18
0.17
0.17
0.17
0.46
0.31
0.24
0.21
0.19
0.18
0.17
0.17
0.48
0.32
0.23
0.21
0.19
0.17
0.17
0.16
0.48
0.32
0.24
0.21
0.19
0.18
0.17
0.17
0.5
0.33
0.25
0.22
0.2
0.19
0.18
0.18
0.5
0.33
0.26
0.22
0.21
0.19
0.18
0.18
0.51
0.35
0.26
0.23
0.21
0.19
0.18
0.18
0.54
0.37
0.27
0.23
0.21
0.19
0.18
0.18
0.58
0.35
0.24
0.22
0.19
0.18
0.17
0.17
0.58
0.32
0.22
0.2
0.18
0.17
0.16
0.15
0.75
0.43
0.28
0.23
0.2
0.18
0.18
0.17
QuALITY
1
2
3
4
5
6
7
8
Subsequent Position
16
24
32
36
40
44
48
52
56
60
64
Layer
0.45
0.3
0.23
0.21
0.19
0.18
0.18
0.18
0.44
0.3
0.23
0.21
0.19
0.19
0.19
0.19
0.47
0.31
0.23
0.21
0.2
0.19
0.18
0.18
0.46
0.31
0.24
0.21
0.19
0.19
0.19
0.18
0.48
0.32
0.24
0.21
0.2
0.19
0.19
0.19
0.48
0.32
0.24
0.21
0.2
0.19
0.19
0.19
0.51
0.34
0.25
0.22
0.21
0.2
0.19
0.19
0.53
0.37
0.27
0.23
0.21
0.2
0.2
0.19
0.57
0.37
0.26
0.22
0.21
0.2
0.19
0.19
0.58
0.34
0.23
0.21
0.2
0.19
0.19
0.18
0.74
0.45
0.28
0.23
0.2
0.19
0.19
0.18
GPQA
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
0.2
0.3
0.4
0.5
0.6
0.7
0.8
0.9
1.0
Accuracy@Top5
Figure 22. Averaged Top-5 accuracy for subsequent token prediction with Off-the-Shelf LLM, across selected Transformers layers and
subsequent positions (up to the 8th following position).
32

How Far Ahead Do LLMs Plan? Uncovering the Latent Horizon in Chain-of-Thought Reasoning
300
600
900
1200
1500
300
600
900
1200
1500
Parity
300
600
900
1200 1500
300
600
900
1200
1500
Cycle
0.5k
1k
1.5k
2k
2.5k
3k
0.5k
1k
1.5k
2k
2.5k
3k
Subsum
200
300
400
500
200
300
400
500
GSM8K
200
400
600
800
200
400
600
800
MATH
300 400 500 600 700 800 900
300
400
500
600
700
800
900
AIME
50
100 150 200 250 300 350
50
100
150
200
250
300
350
MuSR
200
300
400
500
200
300
400
500
Zebra
50
100
150
200
250
50
100
150
200
250
CSQA
100
200
300
400
500
100
200
300
400
500
MMLU
50
100
150
200
250
50
100
150
200
250
QuALITY
200
400
600
800
1000
200
400
600
800
1000
GPQA
0
1
2
3
4
5
6
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
7
0
1
2
3
4
5
6
0.00
0.25
0.50
0.75
1.00
1.25
1.50
1.75
2.00
0
1
2
3
4
5
0
1
2
3
4
5
6
7
8
0
1
2
3
4
5
6
0
1
2
3
4
5
6
0
1
2
3
4
5
6
0
1
2
3
4
5
6
7
8
Figure 23. Probing for reasoning length: heatmap of the predicted length (y-axis) against the actual length (x-axis) for In-Domain LLM.
1k
2k
3k
4k
5k
6k
1k
2k
3k
4k
5k
6k
Parity
3k
6k
9k
12k
3k
6k
9k
12k
Cycle
2k
4k
6k
8k
2k
4k
6k
8k
Subsum
200
400
600
800 1000 1200
200
400
600
800
1000
1200
GSM8K
0k
1k
2k
3k
4k
5k
0k
1k
2k
3k
4k
5k
MATH
0k
4k
8k
12k
0k
4k
8k
12k
AIME
400
800
1200 1600 2000
400
800
1200
1600
2000
MuSR
1k
2k
3k
4k
5k
6k
1k
2k
3k
4k
5k
6k
Zebra
200
400
600
800
200
400
600
800
CSQA
300
600
900
1200 1500
300
600
900
1200
1500
MMLU
150
300
450
600
750
150
300
450
600
750
QuALITY
0k
1k
2k
3k
4k
5k
6k
0k
1k
2k
3k
4k
5k
6k
GPQA
0
1
2
3
4
5
6
7
8
9
0
2
4
6
8
10
12
14
16
18
0
2
4
6
8
10
0
1
2
3
4
5
6
0
2
4
6
8
10
0.00
0.25
0.50
0.75
1.00
1.25
1.50
1.75
2.00
0
1
2
3
4
5
6
7
8
9
0
2
4
6
8
10
0
1
2
3
4
5
6
7
0
1
2
3
4
5
6
0
1
2
3
4
5
6
0
1
2
3
4
5
6
Figure 24. Probing for reasoning length: heatmap of the predicted length (y-axis) against the actual length (x-axis) for Off-the-Shelf LLM.
Result discussions are addressed near Figure 6.
33
