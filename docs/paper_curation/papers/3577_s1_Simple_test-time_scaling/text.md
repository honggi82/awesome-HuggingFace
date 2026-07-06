# arXiv:2501.19393v3[cs.CL]1Mar2025

## s1: Simple test-time scaling

Niklas Muennighoff* 1 3 4 Zitong Yang* 1 Weijia Shi* 2 3 Xiang Lisa Li* 1 Li Fei-Fei1 Hannaneh Hajishirzi2 3 Luke Zettlemoyer2 Percy Liang1 Emmanuel Candès1 Tatsunori Hashimoto1

### Abstract

Test-time scaling is a promising new approach to language modeling that uses extra test-time compute to improve performance. Recently, OpenAI’s

- o1 model showed this capability but did not publicly share its methodology, leading to many replication efforts. We seek the simplest approach to achieve test-time scaling and strong reasoning performance. First, we curate a small dataset s1K
- of 1,000 questions paired with reasoning traces relying on three criteria we validate through ablations: difficulty, diversity, and quality. Second, we develop budget forcing to control test-time compute by forcefully terminating the model’s thinking process or lengthening it by appending “Wait” multiple times to the model’s generation when it tries to end. This can lead the model to doublecheck its answer, often fixing incorrect reasoning steps. After supervised finetuning the Qwen2.532B-Instruct language model on s1K and equipping it with budget forcing, our model s1-32B exceeds o1-preview on competition math questions by up to 27% (MATH and AIME24). Further, scaling s1-32B with budget forcing allows extrapolating beyond its performance without test-time intervention: from 50% to 57% on AIME24. Our model, data, and code are open-source at https: //github.com/simplescaling/s1.

### 1. Introduction

Performance improvements of language models (LMs) over the past years have largely relied on scaling up train-time compute using large-scale self-supervised pretraining (Kaplan et al., 2020; Hoffmann et al., 2022). The creation of these powerful models has set the stage for a new scaling paradigm built on top of them: test-time scaling. The aim

*Equal contribution. ZY and NM started the project. WS, NM and ZY collected the prompts, XL, ZY and NM, built the data pipeline, LZ and WS proposed using a 1K subset and ZY and NM built budget forcing. 1 Stanford University. 2 University of Washington, Seattle. 3 Allen Institute for AI. 4 Contextual AI.

Mathematical Problem Solving (MATH500)

Competition Math (AIME24)

PhD-Level Science Questions (GPQA Diamond)

95

60

60

Accuracy(%)

85

40

50

75

20

40

65

0

512 2048

512 2048 8192

1024 4096

Average thinking time (tokens)

Figure 1. Test-time scaling with s1-32B. We benchmark s1-32B on reasoning-intensive tasks and vary test-time compute.

of this approach is to increase the compute at test time to get better results. There has been much work exploring this idea (Snell et al., 2024; Welleck et al., 2024), and the viability of this paradigm was recently validated by OpenAI o1 (OpenAI, 2024). o1 has demonstrated strong reasoning performance with consistent gains from scaling test-time compute. OpenAI describes their approach as using largescale reinforcement learning (RL) implying the use of sizable amounts of data (OpenAI, 2024). This has led to various attempts to replicate their models relying on techniques like Monte Carlo Tree Search (Gao et al., 2024b; Zhang et al., 2024a), multi-agent approaches (Qin et al., 2024), and others (Wang et al., 2024a; Huang et al., 2024b; 2025). Among these approaches, DeepSeek R1 (DeepSeek-AI et al., 2025) has successfully replicated o1-level performance, also employing reinforcement learning via millions of samples and multiple training stages. However, despite the large number of o1 replication attempts, none have openly replicated a clear test-time scaling behavior. Thus, we ask: what is the simplest approach to achieve both test-time scaling and strong reasoning performance?

We show that training on only 1,000 samples with next-token prediction and controlling thinking duration via a simple test-time technique we refer to as budget forcing leads to a strong reasoning model that scales in performance with more test-time compute. Specifically, we construct s1K, which consists of 1,000 carefully curated questions paired with reasoning traces and answers distilled from Gemini Thinking Experimental (Google, 2024). We perform supervised fine-tuning (SFT) of an off-the-shelf pretrained model

on our small dataset requiring just 26 minutes of training on 16 H100 GPUs. After training, we control the amount of test-time compute our model spends using budget forcing: (I) If the model generates more thinking tokens than a desired limit, we forcefully end the thinking process by appending an end-of-thinking token delimiter. Ending the thinking this way makes the model transition to generating its answer. (II) If we want the model to spend more test-time compute on a problem, we suppress the generation of the end-of-thinking token delimiter and instead append “Wait” to the model’s current reasoning trace to encourage more exploration. Equipped with this simple recipe – SFT on 1,000 samples and test-time budget forcing – our model s132B exhibits test-time scaling (Figure 1). Further, s1-32B is the most sample-efficient reasoning model and outperforms closed-source models like OpenAI’s o1-preview (Figure 2).

sity: Datasets should stem from various fields to cover different reasoning tasks. We collect datasets of two categories:

Curation of existing datasets Our largest source is NuminaMATH (LI et al., 2024) with 30,660 mathematical problems from online websites. We also include historical AIME problems (1983-2021). To enhance diversity, we add OlympicArena (Huang et al., 2024a) with 4,250 questions spanning Astronomy, Biology, Chemistry, Computer Science, Geography, Mathematics, and Physics from various Olympiads. OmniMath (Gao et al., 2024a) adds 4,238 competition-level mathematics problems. We also include 2,385 problems from AGIEval (Zhong et al., 2023), which features questions from standardized tests like SAT and LSAT, covering English, Law, and Logic. We refer to Table 7 in §C for our other sources.

We conduct extensive ablation experiments targeting (a) our selection of 1,000 (1K) reasoning samples and (b) our testtime scaling. For (a), we find that jointly incorporating difficulty, diversity, and quality measures into our selection algorithm is important. Random selection, selecting samples with the longest reasoning traces, or only selecting maximally diverse samples all lead to significantly worse performance (around −30% on AIME24 on average). Training on our full data pool of 59K examples, a superset of s1K, does not offer substantial gains over our 1K selection. This highlights the importance of careful data selection and echoes prior findings for instruction tuning (Zhou et al.,

New datasets in quantitative reasoning To complement these existing datasets, we create two original datasets. s1-prob consists of 182 questions from the probability section of Stanford University’s Statistics Department’s PhD Qualifying Exams (https://statistics.

stanford.edu), accompanied by handwritten solutions that cover difficult proofs. The probability qualifying exam is held yearly and requires professional-level mathematical problem-solving. s1-teasers comprises 23 challenging brain-teasers commonly used in interview questions for quantitative trading positions. Each sample consists of a problem and solution taken from PuzzledQuant (https:

- 2023). For (b), we define desiderata for test-time scaling methods to compare different approaches. Budget forcing leads to the best scaling as it has perfect controllability with a clear positive slope leading to strong performance.

//www.puzzledquant.com/). We only take examples with the highest difficulty level ("Hard").

For each question, we generate a reasoning trace and solution using the Google Gemini Flash Thinking API (Google, 2024) extracting its reasoning trace and response. This yields 59K triplets of a question, generated reasoning trace, and generated solution. Examples from our dataset are in §D.2. We decontaminate all samples against our evaluation questions (MATH500, GPQA Diamond, AIME24; §C.5) using 8-grams and deduplicate the data.

In summary, our contributions are: We develop simple methods for creating a sample-efficient reasoning dataset (§2) and test-time scaling (§3); Based on these we build s1-32B which is competitive with o1-preview (§4); We ablate subtleties of data (§5.1) and test-time scaling (§5.2). We end with a discussion to motivate future work on simple reasoning (§6). Our code, model, and data are open-source at https://github.com/simplescaling/s1.

#### 2.2. Final selection of 1K samples

We could directly train on our pool of 59K questions, however, our goal is to find the simplest approach with minimal resources. Thus, we go through three stages of filtering to arrive at a minimal set of 1,000 samples relying on our three guiding data principles: Quality, Difficulty, and Diversity.

### 2. Reasoning data curation to create s1K

In this section, we describe our process for creating a large dataset first in §2.1 and then filtering it down to s1K in §2.2.

#### 2.1. Initial collection of 59K samples

Quality We first remove any questions where we ran into any API errors reducing our dataset to 54,116 samples. Next, we filter out low-quality examples by checking if they contain any string patterns with formatting issues, such as ASCII art diagrams, non-existent image references, or inconsistent question numbering reducing our dataset to 51,581 examples.

We collect an initial 59,029 questions from 16 sources following three guiding principles. Quality: Datasets should be high-quality; we always inspect samples and ignore datasets with, e.g., poor formatting; Difficulty: Datasets should be challenging and require significant reasoning effort; Diver-

General topology

Statistics

Geo− physics

Approxi− mation theory

General relativity

Convex geometry

100

Special functions

|s1<br><br>r1-distill<br><br>Sky-T1<br><br>QwQ<br><br>Bespoke-Stratos<br><br>Most sample-efficient| | | | |
|---|---|---|---|---|
| | | | | |

Operations research

Probability theory

Integral transform

Linear algebra

MATH500Accuracy(%)

Computer science

Comm− utative algebra

Math− ematical logic

Group theory

Statistical mechanics

95

Real functions

Integral equations

Biology

Numerical analysis

Electro− dynamics

Differential geometry

Dynamical systems

Q

Complex functions

Measure theory

Geometry Number theory

90

Math− ematics education

Difference equations

Partial differential equations

Calculus

Quantum theory

Differential equations

o1-preview

Combin− atorics

Calculus of variations

Potential theory

85

Information theory

Algebraic structures

Functional analysis

Field theory

Thermo− dynamics

Mechanics

Harmonic analysis

Algebraic systems

Solid mech −anics

Astronomy

Associative rings

Fluid mechanics

80

Control theory

1000 17000 800000 N/A Number of Examples

Economics

Algebraic geometry

- Figure 2. s1K and s1-32B. (left) s1K is a dataset of 1,000 high-quality, diverse, and difficult questions with reasoning traces. (right) s1-32B, a 32B parameter model finetuned on s1K is on the sample-efficiency frontier. See Table 1 for details on other models.

From this pool, we identify 384 samples for our final 1,000 samples from datasets that we perceive as high-quality and not in need of further filtering (see §C.4 for details).

nomics. To select our final examples from the pool of 24,496 questions, we first choose one domain uniformly at random. Then, we sample one problem from this domain according to a distribution that favors longer reasoning traces (see §C.4 for details) as motivated in Difficulty. We repeat this process until we have 1,000 total samples spanning 50 domains.

Difficulty For difficulty, we use two indicators: model performance and reasoning trace length. We evaluate two models on each question: Qwen2.5-7B-Instruct and Qwen2.532B-Instruct (Qwen et al., 2024), with correctness assessed by Claude 3.5 Sonnet comparing each attempt against the reference solution (see §C.3 for the grading protocol). We measure the token length of each reasoning trace to indicate problem difficulty using the Qwen2.5 tokenizer. This relies on the assumption that more difficult problems require more thinking tokens. Based on the grading, we remove questions that either Qwen2.5-7B-Instruct or Qwen2.5-32B-Instruct can solve correctly and thus may be too easy. By using two models we reduce the likelihood of an easy sample slipping through our filtering due to a rare mistake on an easy question of one of the models. This brings our total samples down to 24,496, setting the stage for the next round of subsampling based on diversity. While filtering with these two models may be optimized for our setup as we will also use Qwen2.5-32B-Instruct as our model to finetune, the idea of model-based filtering generalizes to other setups.

In §5.1, we will show that using our three criteria in combination is important, as only relying on quality, diversity, or difficulty in isolation leads to worse datasets. Some distilled generations are incorrect, which we allow in our data as we focus on capturing the reasoning process rather than entirely correct solutions. Our grader (§C.3) deems 53.6% correct in s1K and 63.0% in our follow-up s1K-1.1 (see §A).

### 3. Test-time scaling

#### 3.1. Method

We classify test-time scaling methods into 1) Sequential, where later computations depend on earlier ones (e.g., a long reasoning trace), and 2) Parallel, where computations run independently (e.g., majority voting) (Snell et al., 2024; Brown et al., 2024). We focus on sequential scaling as intuitively we believe it should scale better, since later computations can build on intermediate results, allowing for deeper reasoning and iterative refinement. We propose new sequential scaling methods and ways to benchmark them.

Diversity To quantify diversity, we classify questions into domains using Claude 3.5 Sonnet based on the Mathematics Subject Classification (MSC) system (e.g., geometry, combinatorics, etc.) from the American Mathematical Society.1 The taxonomy focuses on topics in mathematics but also includes other sciences such as biology, physics, and eco-

Budget forcing We propose a simple decoding-time intervention by forcing a maximum and/or minimum number of thinking tokens. Specifically, we enforce a maximum token count by simply appending the end-of-thinking token delimiter and optionally “Final Answer:” to early exit

1https://mathscinet.ams.org/mathscinet/ msc/msc2020.html

three metrics: Control =

𝕀(𝑎min ≤ 𝑎 ≤ 𝑎max) (1)

∑

1 ||

𝑎∈

where 𝑎min,𝑎max refer to a pre-specified minimum and maximum amount of test-time compute; in our case thinking

tokens. We usually only constrain 𝑎max. As tokens generated correspond to the amount of test-time compute spent, this metric measures the extent to which a method allows controllability over the use of that test-time compute. We report it as a percentage with 100% being perfect control.

∑

𝑓(𝑏) − 𝑓(𝑎) 𝑏 − 𝑎

1 (|2|)

Scaling =

(2)

| |
|---|

𝑎,𝑏∈ 𝑏>𝑎

- Figure 3. Budget forcing with s1-32B. The model tries to stop after “...is 2.”, but we suppress the end-of-thinking token delimiter instead appending “Wait” leading s1-32B to self-correct its answer.

Scaling is the average slope of the piece-wise linear function. It must be positive for useful methods and larger is better.

Performance = max

𝑓(𝑎) (3)

𝑎∈

the thinking stage and make the model provide its current best answer. To enforce a minimum, we suppress the generation of the end-of-thinking token delimiter and optionally append the string “Wait” to the model’s current reasoning trace to encourage the model to reflect on its current generation. Figure 3 contains an example of how this simple approach can lead the model to arrive at a better answer.

Performance is simply the maximum performance the method achieves on the benchmark. A method with monotonically increasing scaling achieves 100% performance on any benchmark in the limit. However, the methods we investigate eventually flatten out or further scaling fails due to control or context window limitations.

Baselines We benchmark budget forcing with: (I) Conditional length-control methods, which rely on telling the model in the prompt how long it should generate for. We group them by granularity into (a) Token-conditional control: We specify an upper bound of thinking tokens in the prompt; (b) Step-conditional control: We specify an upper bound of thinking steps, where each step is around 100 tokens; (c) Class-conditional control: We write two generic prompts that tell the model to either think for a short or long amount of time (see §E.1 for details). (II) Rejection sampling, which samples until a generation fits a predetermined compute budget. This oracle captures the posterior over responses conditioned on its length.

#### 3.2. Metrics

We establish a set of desiderata as evaluation metrics to measure test-time scaling across methods. Importantly, we do not only care about the accuracy a method can achieve but also its controllability and test-time scaling slope. For each method we consider, we run a set of evaluations 𝑎 ∈  varying test-time compute on a fixed benchmark, e.g.

AIME24. This produces a piece-wise linear function 𝑓 with compute as the x-axis measured in thinking tokens and accuracy as the y-axis (see Figure 1, where the rightmost dot for AIME24 corresponds to 𝑓(7320) = 57%). We measure

### 4. Results

#### 4.1. Setup

Training We perform supervised finetuning on Qwen2.532B-Instruct using s1K to obtain our model s1-32B using basic hyperparameters outlined in §D. Finetuning took 26 minutes on 16 NVIDIA H100 GPUs with PyTorch FSDP.

Evaluation We select three representative reasoning benchmarks widely used in the field: AIME24 (of America, 2024) has 30 problems that were used in the 2024 American Invitational Mathematics Examination (AIME) held from January 31 – February 1, 2024. AIME tests mathematical problem-solving with arithmetic, algebra, counting, geometry, number theory, probability, and other secondary school math topics. High-scoring high school students in the test are invited to participate in the United States of America Mathematics Olympiad (USAMO). All AIME answers are integers ranging from 000 to 999, inclusive. Some AIME problems rely on figures that we provide to our model using the vector graphics language Asymptote as it cannot take image inputs. MATH500 (Hendrycks et al., 2021) is a benchmark of competition math problems of varying difficulty. We evaluate on the same 500 samples selected by OpenAI in prior work (Lightman et al., 2023). GPQA Diamond (Rein et al., 2023) consists of 198 PhD-level science questions

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

(a) Sequential scaling via budget forcing (b) Parallel scaling via majority voting

- Figure 4. Sequential and parallel test-time scaling. (a): Budget forcing shows clear scaling trends and extrapolates to some extent. For the three rightmost dots, we prevent the model from stopping its thinking 2/4/6 times, each time appending “Wait” to its current reasoning trace. (b): For Qwen2.5-32B-Instruct we perform 64 evaluations for each sample with a temperature of 1 and visualize the performance when majority voting across 2, 4, 8, 16, 32, and 64 of these.

from Biology, Chemistry and Physics. Experts with PhDs in the corresponding domains only achieved 69.7% on GPQA Diamond (OpenAI, 2024). When we write “GPQA” in the context of evaluation in this work, we always refer to the Diamond subset. We build on the “lm-evaluation-harness” framework (Gao et al., 2021; Biderman et al., 2024). Unless otherwise specified, we evaluate with a temperature of 0 (greedy) and measure accuracy (equivalent to pass@1).

#### Other models We benchmark s1-32B against: OpenAI

- o1 series (OpenAI, 2024), closed-source models that popularized test-time scaling; DeepSeek r1 series (DeepSeekAI et al., 2025), open-weight reasoning models with up to
- o1-level performance; Qwen’s QwQ-32B-preview (Team,

- 2024), a 32B open-weight reasoning model without disclosed methodology; Sky-T1-32B-Preview (Team, 2025) and Bespoke-32B (Labs, 2025), open models with open reasoning data distilled from QwQ-32B-preview and r1; Google Gemini 2.0 Flash Thinking Experimental (Google, 2024), the API that we distill from. As it has no official evaluation scores, we use the Gemini API to benchmark it ourselves. However, the “recitation error” of the Gemini API makes evaluation challenging.2 We circumvent this, by manually inserting all 30 AIME24 questions in its web interface where the error does not appear. However, we leave out MATH500 (500 questions) and GPQA Diamond (198 questions), thus they are N.A. in Table 1. Our model, s1-32B, is fully open including weights, reasoning data, and code.

2https://github.com/google/ generative-ai-docs/issues/257

Table 1. s1-32B is a strong open reasoning model. We evaluate s1-32B, Qwen, and Gemini (some entries are unknown (N.A.), see §4). Other results are from the respective reports (Qwen et al.,

- 2024; Team, 2024; OpenAI, 2024; DeepSeek-AI et al., 2025; Labs,
- 2025; Team, 2025). # ex. = number examples used for reasoning finetuning; BF = budget forcing. See §A for our better s1.1 model.

GPQA Diamond API only

AIME 2024

MATH 500

Model # ex.

- o1-preview N.A. 44.6 85.5 73.3
- o1-mini N.A. 70.0 90.0 60.0
- o1 N.A. 74.4 94.8 77.3

Gemini 2.0

N.A. 60.0 N.A. N.A. Flash Think.

Open Weights Qwen2.5-

N.A. 26.7 84.0 49.0

32B-Instruct QwQ-32B N.A. 50.0 90.6 54.5

#### r1 ≫800K 79.8 97.3 71.5

- r1-distill 800K 72.6 94.3 62.1 Open Weights and Open Data

Sky-T1 17K 43.3 82.4 56.8 Bespoke-32B 17K 63.3 93.0 58.1

- s1 w/o BF 1K 50.0 92.6 56.6

#### s1-32B 1K 56.7 93.0 59.6

- 4.2. Performance

Test-time scaling Figure 1 shows the performance of s132B with budget forcing scales with more test-time compute. In Figure 4 (left), we expand the plot from Figure 1 (middle) showing that while we can improve AIME24 performance using our budget forcing technique (§3) and more test-time compute it does eventually flatten out at six times. Suppressing the end-of-thinking token delimiter too often can lead the model into repetitive loops instead of continued reasoning. In Figure 4 (right), we show that after training Qwen2.532B-Instruct on our 1,000 samples to produce s1-32B and equipping it with the simple budget forcing technique, it operates in a different scaling paradigm. Scaling test-time compute on the base model via majority voting cannot catch up with the performance of s1-32B which validates our intuition from §3 that sequential scaling is more effective than parallel. We provide example generations of s1-32B in

- Figure 5.

Sample-efficiency In Figure 2 (right) and Table 1 we compare s1-32B with other models. We find that s132B is the most sample-efficient open data reasoning model. It performs significantly better than our base model (Qwen2.5-32B-Instruct) despite just training it on an additional 1,000 samples. The concurrently released r1-32B shows stronger performance than s1-32B while also only using SFT (DeepSeek-AI et al., 2025). However, it is trained on 800 × more reasoning samples. It is an open question whether one can achieve their performance with just 1,000 samples. Finally, our model nearly matches Gemini 2.0 Thinking on AIME24. As the data for s1-32B is distilled from Gemini 2.0, this shows our distillation procedure was likely effective.

- 5. Ablations

- 5.1. Data Quantity, Diversity, and Difficulty

In §2 we outlined our three guiding principles in curating s1K: Quality, Difficulty, and Diversity. Here we test the importance of combining them and the overall efficacy of our selection. Only Quality (1K-random): After obtaining our high-quality reasoning chains from Gemini, we select 1,000 samples at random; not relying on our difficulty and diversity filtering at all. Table 2 shows this approach performs much worse than s1K across all benchmarks. Only Diversity (1K-diverse): For this dataset, we sample uniformly across domains to maximize diversity disregarding any notion of difficulty. This approach also leads to poor performance similar to 1K-random. Only Difficulty (1K-longest): Here we rely on one of our difficulty indicators introduced in §2 by selecting the 1,000 samples with the longest reasoning traces. This approach significantly boosts GPQA performance but overall still falls short of using s1K. Maximize Quantity:

- Table 2. s1K data ablations. We budget force (BF) a maximum of around 30,000 thinking tokens for all scores in this table. This performs slightly better than the scores without BF (Table 1) as it allows the model to finish with a best guess when stuck in an infinite loop. We report 95% paired bootstrap confidence intervals for differences relative to the s1K model using 10,000 bootstrap samples. E.g., the interval [-13%, 20%] means that, with 95% confidence, the true difference between 59K-full and s1K is between

-13% and +20%. If the entire interval is negative, e.g. [-27%, -3%], we can confidently say that the performance is worse than s1K.

Model

AIME 2024

MATH 500

GPQA Diamond 1K-random

36.7 90.6 52.0

[-26.7%, -3.3%] [-4.8%, 0.0%] [-12.6%, 2.5%]

1K-diverse

26.7 91.2 54.6

[-40.0%, -10.0%] [-4.0%, 0.2%] [-10.1%, 5.1%]

1K-longest

33.3 90.4 59.6

[-36.7%, 0.0%] [-5.0%, -0.2%] [-5.1%, 10.1%]

59K-full

53.3 92.8 58.1

[-13.3%, 20.0%] [-2.6%, 2.2%] [-6.6%, 8.6%]

s1K 50.0 93.0 57.6

Finally, we compare with just training on all of our 59K samples, a superset of all the 1K-sample versions. This leads to a strong model but uses much more resources. To finetune on 59K samples, we use 394 H100 GPU hours while s1-32B only required 7 H100 GPU hours. Moreover, relying only on s1K is extremely competitive as shown in §2. Overall, combining all three criteria – Quality, Difficulty, Diversity – via our methodology in §2 is key for sampleefficient reasoning training.

5.2. Test-time scaling methods

- Table 3. Ablations on methods to scale test-time compute on AIME24. || refers to the number of evaluation runs used to estimate the properties; thus a higher value indicates more robustness. Bold indicates our chosen method and the best values. BF

= budget forcing, TCC/SCC/CCC = token/step/class-conditional control, RS = rejection sampling.

MethodBF Control100% Scaling15 Performance56.7 |5 |

TCC 40% -24 40.0 5 TCC + BF 100% 13 40.0 5 SCC 60% 3 36.7 5 SCC + BF 100% 6 36.7 5 CCC 50% 25 36.7 2

RS 100% -35 40.0 5

[AIME24]

Alice and Bob play the following game. A stack of 𝑛 tokens lies before them. The players take turns with Alice going first. On each turn, the player removes either 1 token or 4 tokens from the stack. Whoever removes the last token wins. Find the number of positive integers 𝑛 less than or equal to 2024 for which there exists a strategy for Bob that guarantees that Bob will win the game regardless of Alice’s play.

The problem asks for the number of positive integers 𝑛 ≤ 2024 such that Bob has a winning strategy. This is a typical impartial game that can be analyzed using the Sprague-Grundy theorem or by directly finding the winning and losing positions. Let 𝑊 be the set of winning positions (number of tokens 𝑛 such that the player whose turn it is can force a win) and 𝐿 be the set of losing positions. [...]

Let 𝑃(𝑛) be the statement "the player whose turn it is when there are 𝑛 tokens has a winning strategy. 𝑃(0) is false (the player whose turn it is and faces 0 tokens has already lost). Consider the small values of 𝑛:

- 𝑛 = 1: The player whose turn it is can take 1 token and win. So 𝑃(1) is true. 1 ∈ 𝑊 .
- 𝑛 = 2: The player whose turn it is can take 1 token,

- leaving 1 token for the opponent. Since 1 ∈ 𝑊 , the opponent wins. So 𝑃(2) is false. 2 ∈ 𝐿.

𝑛 = 3: The player whose turn it is can take 1 token,

- leaving 2 tokens for the opponent. Since 2 ∈ 𝐿, the player whose turn it is wins. So 𝑃(3) is true. 3 ∈ 𝑊 .[...]

We are looking for the number of positive integers 𝑛 ≤ 2024 for which Bob has a winning strategy. This means that if the game starts with 𝑛 tokens, and Alice goes first, Bob can win. This happens if and only if 𝑛 is a losing position (because if 𝑛 ∈ 𝐿, then Alice’s first move must lead to a winning position for Bob). [...]

Let’s try to find a pattern in the losing positions 𝐿 = {0,2,5,7,10,12,15,17,20,22,...}. Differences between consecutive terms: 2,3,2,3,2,3,2,3,2,... The pattern of differences is alternating 2,3. [...] The losing positions are of the form 5𝑚 or 5𝑚 + 2, where 𝑚 ≥ 0. We need to find the number of positive integers 𝑛 ≤ 2024 that are in the set of losing positions 𝐿. [...]

- Case 1: 𝑛 = 5𝑚, where 𝑚 ≥ 1. 1 ≤ 5𝑚 ≤ 2024 1∕5 ≤ 𝑚 ≤ 2024∕5 = 404.8 Since 𝑚 is an integer, 1 ≤ 𝑚 ≤ 404. The number of values of 𝑚 is 404.
- Case 2: 𝑛 = 5𝑚 + 2, where 𝑚 ≥ 0. 5𝑚 + 2 ≤ 2024 5𝑚 ≤ 2022 𝑚 ≤ 2022∕5 = 404.4 Since 𝑚 is an integer, 0 ≤ 𝑚 ≤ 404. The number of values of 𝑚 is 404 − 0 + 1 = 405.

The total number of positive integers 𝑛 ≤ 2024 that are losing positions is 404 + 405 =

.

|809|
|---|

[MATH500]

[GPQA]

Find the projection of 𝑎 onto 𝑏 = (

A quantum mechanical particle of mass 𝑚 moves in two dimensions in the following potential, as a function of the polar coordinates (𝑟,𝜃):

) if 𝑎 ⋅ 𝑏 = 8.

- 2 6
- 3

The projection of vector 𝐚 onto vector 𝐛 is given by the formula:

- 1

- 2

3 2

𝑘𝑟2 +

𝑘𝑟2 cos2(𝜃)

𝑉 (𝑟,𝜃) =

𝐚 ⋅ 𝐛 ‖𝐛‖2

proj𝐛𝐚 =

Find the energy spectrum. Hint: Write the potential in Cartesian coordinates.

###### 𝐛

We are given 𝐛 = (

) and 𝐚⋅𝐛 = 8.

- 2 6
- 3

The problem asks for the energy spectrum of a particle in a given 2D potential. The potential is given in polar coordinates (𝑟,𝜃):

First, we need to find the magnitude squared of 𝐛, which is ‖𝐛‖2.

- 1

- 2

3 2

𝑘𝑟2 +

𝑘𝑟2 cos2(𝜃).

𝑉 (𝑟,𝜃) =

‖𝐛‖2 = 𝐛⋅𝐛 = 22+62+32 = 4+36+9 = 49.

We can rewrite the potential as:

- 1

- 2

Now we can plug the values into the projection formula:

𝑘𝑟2(1 + 3cos2(𝜃)).

𝑉 (𝑟,𝜃) =

The Hamiltonian for a particle in 2D is given by:

8 49

proj𝐛𝐚 =

###### 𝐛

( 𝜕2 𝜕𝑥2

) + 𝑉 (𝑥,𝑦).

Substitute the vector 𝐛:

+ 𝜕2 𝜕𝑦2

𝐻 = − ℏ2 2𝑚

(

)

- 2 6
- 3

8 49

proj𝐛𝐚 =

In polar coordinates (𝑟,𝜃), the Laplacian operator is:

(𝑟

) +

𝜕2 𝜕𝜃2

1 𝑟

1 𝑟2

𝜕 𝜕𝑟

𝜕 𝜕𝑟

∇2 =

.

Multiply the scalar by each component of the vector:

So the Hamiltonian in polar coordinates is:

8

16 49

⎞

proj𝐛𝐚 = ⎛ ⎜ ⎝

⎞

= ⎛

- 49 × 2 8

49 × 6 8

- 49 × 3

[

]+𝑉 (𝑟,𝜃).

(𝑟

) +

- 48

- 49 24 49

𝜕2 𝜕𝜃2

𝐻 = − ℏ2 2𝑚

1 𝑟2

1 𝑟

𝜕 𝜕𝑟

𝜕 𝜕𝑟

⎟ ⎠

⎟ ⎠

⎜ ⎝

The time-independent Schrödinger equation is 𝐻𝜓 = 𝐸𝜓. We can try to separate variables by assuming a solution of the form 𝜓(𝑟,𝜃) = 𝑅(𝑟)Θ(𝜃). [...]

Final Answer: The final answer is

|⎛16 49<br><br>48<br><br>49 24 49<br><br><br>⎞|
|---|

This confirms the energy spectrum is:

⎜ ⎝

⎟ ⎠

𝐸 = ℏ√ 𝑘 𝑚

(2𝑛𝑥 + 𝑛𝑦 +

3 2).

Final Answer: 𝐸 = ℏ√

(2𝑛𝑥 + 𝑛𝑦 + 23).

𝑘 𝑚

- Figure 5. Example model outputs. We pick a question from AIME24 (left), MATH500 (middle), and GPQA (right), where our model generates the correct answer. The black text is the prompt, the light blue text is the reasoning trace, and the blue text is the answer of s1-32B. The gray ellipsis [...] indicates that the text was trimmed to fit this page, but the generated text is actually longer.

Budget forcing In Table 3 we compare the test-time scaling methods we have introduced in §3. Overall, we find that budget forcing provides perfect control, good scaling, and leads to our best AIME24 score. Thus, this is the method we use for s1-32B in Figure 1 and in §4. In Table 4, we compare different strings for extrapolating performance. We find that “Wait” generally gives the best performance.

spect a question, which was answered correctly by the model when rejection sampling for ≤ 4000, but not for the ≤ 8000 token setting. In the ≤ 4000 setting the model directly jumps to the correct approach, while for the ≤ 8000 setting it backtracks a lot. We hypothesize that there is a correlation such that shorter generations tend to be the ones where the model was on the right track from the start, whereas longer ones tend to be ones where the model made mistakes and thus backtracks or questions itself. This leads to longer samples often being wrong when rejection sampling and thus the inverse scaling trend.

Class-conditional control We provide benchmark scores for this method in §E.1 and summarize three findings here: (1) Token-conditional control fails without budget forcing, as our model cannot reliably count tokens - even when trained to do so. (2) Under step-conditional control, the model generates a similar total number of tokens when given different step targets, as the model goes from few steps with many tokens per step, to many steps with few tokens in each step. Thus, the model learns to hack its way around the compute constraint making the controllability of this method mediocre. (3) Class-conditional control can work - telling a model to simply think longer can increase its test-time compute and performance, which leads good scaling in Table 3.

### 6. Discussion and related work

#### 6.1. Sample-efficient reasoning

Models There are a number of concurrent efforts to build models that replicate the performance of o1 (OpenAI, 2024). For example, DeepSeek-r1 and k1.5 (DeepSeek-AI et al., 2025; Team et al., 2025) are built with reinforcement learning methods, while others rely on SFT using tens of thousands of distilled examples (Team, 2025; Xu et al., 2025; Labs, 2025). We show that SFT on only 1,000 examples suffices to build a competitive reasoning model matching

Table 4. Budget forcing extrapolation ablations. We compare ignoring the end-of-thinking delimiter twice and appending none or various strings.

- o1-preview and produces a model that lies on the pareto frontier (Figure 2). Further, we introduce budget forcing which combined with our reasoning model leads to the first reproduction of OpenAI’s test-time scaling curves (OpenAI, 2024). Why does supervised finetuning on just 1,000 samples lead to such performance gains? We hypothesize that the model is already exposed to large amounts of reasoning data during pretraining which spans trillions of tokens. Thus, the ability to perform reasoning is already present in
- our model. Our sample-efficient finetuning stage just activates it and we scale it further at test time with budget forcing. This is similar to the "Superficial Alignment Hypothesis" presented in LIMA (Zhou et al., 2023), where the authors find that 1,000 examples can be sufficient to align a model to adhere to user preferences.

AIME 2024

MATH 500

GPQA Diamond

Model

No extrapolation 50.0 93.0 57.6 2x without string 50.0 90.2 55.1 2x “Alternatively” 50.0 92.2 59.6 2x “Hmm” 50.0 93.0 59.6 2x “Wait” 53.3 93.0 59.6

Rejection sampling Surprisingly, we find that simply sampling until the generation fits a specific length leads to an inverse scaling trend as depicted in Figure 6. In §E.2 we in-

42

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

Accuracy(%)

37

32

27

22

3072 4096 5120 Average thinking time (tokens)

- Figure 6. Rejection sampling on AIME24 with s1-32B. We sample with a temperature of 1 until all generations have less than (from left to right) 3500, 4000, 5000, 8000, and 16000 thinking tokens requiring an average of 655, 97, 8, 3, 2, and 1 tries per sample.

Benchmarks and methods To evaluate and push the limits of these models, increasingly challenging benchmarks have been introduced, such as Olympiad-level science competitions (He et al., 2024; Jain et al., 2024; Zhong et al., 2023) and others (Srivastava et al., 2023; Glazer et al., 2024; Su et al., 2024; Kim et al., 2024; Phan et al., 2025). To enhance models’ performance on reasoning-related tasks, researchers have pursued several strategies: Prior works have explored continuing training language models on specialized corpora related to mathematics and science (Azerbayev et al., 2023; Yang et al., 2024), sometimes even synthetically generated data (Yu et al., 2024). Others have developed training methodologies specifically aimed at reasoning performance (Zelikman et al., 2022; 2024; Luo et al., 2025;

Yuan et al., 2025; Wu et al., 2024a). Another significant line of work focuses on prompting-based methods to elicit and improve reasoning abilities, including methods like Chainof-Thought prompting (Wei et al., 2023; Yao et al., 2023a;b; Bi et al., 2023; Fu et al., 2023; Zhang et al., 2024b; Xiang et al., 2025; Hu et al., 2024; Diao et al., 2024). These combined efforts aim to advance the reasoning ability of language models, enabling them to handle more complex and abstract tasks effectively.

60

| |REBASE Majority Sequential scaling<br><br>| | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |

Accuracy(%)

50

40

30

2048 8192 32768 131072 Average thinking time (tokens)

#### 6.2. Test-time scaling

Methods As we introduce in §3, we differentiate two methods to scale test-time compute: parallel and sequential. The former relies on multiple solution attempts generated in parallel and selecting the best outcome via specific criteria. These criteria include choosing the most frequent response for majority voting or the best response based on an external reward for Best-of-N (Brown et al., 2024; Irvine et al., 2023; Levi, 2024). Unlike repeated sampling, previous sequential scaling methods let the model generate solution attempts sequentially based on previous attempts, allowing it to refine each attempt based on previous outcomes (Snell et al., 2024; Hou et al., 2025; Lee et al., 2025). Tree-based search methods (Gandhi et al., 2024; Wu et al., 2024b) offer a hybrid approach between sequential and parallel scaling, such as Monte-Carlo Tree Search (MCTS) (Liu et al., 2024; Zhang et al., 2023; Zhou et al., 2024; Choi et al., 2023) and guided beam search (Xie et al., 2023). REBASE (Wu et al., 2024b) employs a process reward model to balance exploitation and pruning during tree search. Empirically, REBASE has been shown to outperform sampling-based methods and MCTS (Wu et al., 2024b). Reward models (Lightman et al., 2023; Wang et al., 2024b;c) play a key role in these methods. They come in two variants: outcome reward models and process reward models. Outcome reward models (Xin et al., 2024; Ankner et al., 2024) assign a score to complete solutions and are particularly useful in Best-of-N selection, while process reward models (Lightman et al., 2023; Wang et al., 2024b; Wu et al., 2024b) assess individual reasoning steps and are effective in guiding tree-based search methods.

Limits to further test-time scaling We have shown that budget forcing allows extrapolating test-time compute in §4, e.g., improving AIME24 performance from 50% to 57%. However, it has two key limitations when scaling further: it eventually flattens out (Figure 4), and the context window of the underlying language model constrains it. Despite these constraints, our work shows test-time scaling across a wide range of accuracies (Figure 1), partly because scaling down test-time compute behaves predictably and does not suffer from these constraints.

Continuing test-time scaling will require approaches that can further extrapolate test-time compute. How can we get

Figure 7. Scaling further with parallel scaling methods. All metrics averaged over the 30 questions in AIME24. Average thinking tokens forREBASE do not account for the additional compute from the reward model. For sequential scaling, we prompt the model to use up to (from left to right) 32, 64, 256, and 512 steps. For REBASE and majority voting we generate 16 parallel trajectories to aggregate across.

such extrapolation? There may be improvements to budget forcing such as rotating through different strings, not only “Wait”, or combining it with frequency penalties or higher temperature to avoid repetitive loops. An exciting direction for future work is also researching whether applying budget forcing to a reasoning model trained with reinforcement learning yields better extrapolation; or if RL allows for new ways of test-time scaling beyond budget forcing. Our work defines the right metrics (§3.2) – Control, Scaling, and Performance – to enable future research and progress on extrapolating test-time compute.

Parallel scaling as a solution Parallel scaling offers one solution to the limits of sequential scaling, thus we augment our sequentially scaled model with two methods: (I) Majority voting: After generating 𝑘 solutions, the final solution is the most frequent one across generations; (II) Tree search via REBASE: We use the REBASE process reward model, which is initialized from LLaMA-34B and further finetuned on a synthetic process reward modeling dataset (Wu et al., 2024b). We then aggregate the solutions generated by REBASE via majority voting. As shown in Figure 7, augmenting our model with REBASE scales better than majority voting, and even sequential scaling in this scenario. However, REBASE requires an additional forward pass at each step for the reward model adding some computation overhead. For sequential scaling, when prompted to use up to 512 steps, for 12 out of the 30 evaluation questions the model generates a response that exceeds the context window leading to a large performance drop. Overall, we find that these parallel scaling methods complement sequential scaling thus they offer an avenue for scaling test-time compute even further; beyond fixed context windows.

inference compute with repeated sampling, 2024. URL https://arxiv.org/abs/2407.21787.

### Impact Statement

Language models with strong reasoning capabilities have the potential to greatly enhance human productivity, from assisting in complex decision-making to driving scientific breakthroughs. However, recent advances in reasoning, such as OpenAI’s o1 and DeepSeek’s r1, lack transparency, limiting broader research progress. Our work aims to push the frontier of reasoning in a fully open manner, fostering innovation and collaboration to accelerate advancements that ultimately benefit society.

Cesista, F. L. Multimodal structured generation: Cvpr’s 2nd mmfm challenge technical report, 2024. URL https: //arxiv.org/abs/2406.11403.

Chen, W., Yin, M., Ku, M., Lu, P., Wan, Y., Ma, X., Xu, J., Wang, X., and Xia, T. Theoremqa: A theoremdriven question answering dataset, 2023. URL https: //arxiv.org/abs/2305.12524.

Choi, S., Fang, T., Wang, Z., and Song, Y. Kcts: Knowledgeconstrained tree search decoding with token-level hallucination detection, 2023. URL https://arxiv.org/ abs/2310.09044.

### Acknowledgements

We thank Ryan Marten for generating traces from DeepSeek r1 for s1.1 using Bespoke Curator (Marten et al., 2025). This work was partly conducted using the Stanford Marlowe GPU cluster (Kapfer et al., 2025), made possible by financial support from Stanford University. We thank Alexander M. Rush, Andrew Ilyas, Banghua Zhu, Chenglei Si, Chunting Zhou, John Yang, Ludwig Schmidt, Samy Jelassi, Suhas Kotha, Tengyu Ma, Xuechen Li, Yu Sun, and Yue Zhang for very constructive discussions.

DeepSeek-AI, Guo, D., Yang, D., Zhang, H., Song, J., Zhang,

- R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., Zhang, X., Yu, X., Wu, Y., Wu, Z. F., Gou, Z., Shao, Z., Li, Z., Gao, Z., Liu, A., Xue, B., Wang, B., Wu, B., Feng, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., Dai, D., Chen, D., Ji, D., Li, E., Lin, F., Dai, F., Luo, F., Hao,

- G., Chen, G., Li, G., Zhang, H., Bao, H., Xu, H., Wang,
- H., Ding, H., Xin, H., Gao, H., Qu, H., Li, H., Guo, J., Li, J., Wang, J., Chen, J., Yuan, J., Qiu, J., Li, J., Cai,

- J. L., Ni, J., Liang, J., Chen, J., Dong, K., Hu, K., Gao,
- K., Guan, K., Huang, K., Yu, K., Wang, L., Zhang, L., Zhao, L., Wang, L., Zhang, L., Xu, L., Xia, L., Zhang,

- M., Zhang, M., Tang, M., Li, M., Wang, M., Li, M., Tian,
- N., Huang, P., Zhang, P., Wang, Q., Chen, Q., Du, Q., Ge, R., Zhang, R., Pan, R., Wang, R., Chen, R. J., Jin,

- R. L., Chen, R., Lu, S., Zhou, S., Chen, S., Ye, S., Wang,
- S., Yu, S., Zhou, S., Pan, S., Li, S. S., Zhou, S., Wu,

- S., Ye, S., Yun, T., Pei, T., Sun, T., Wang, T., Zeng, W., Zhao, W., Liu, W., Liang, W., Gao, W., Yu, W., Zhang,

### References

Ankner, Z., Paul, M., Cui, B., Chang, J. D., and Ammanabrolu, P. Critique-out-loud reward models, 2024. URL https://arxiv.org/abs/2408.11791.

Arora, D., Singh, H. G., and Mausam. Have llms advanced enough? a challenging problem solving benchmark for large language models, 2023. URL https://arxiv. org/abs/2305.15074.

Azerbayev, Z., Schoelkopf, H., Paster, K., Santos, M. D., McAleer, S., Jiang, A. Q., Deng, J., Biderman, S., and Welleck, S. Llemma: An open language model for mathematics, 2023.

- W., Xiao, W. L., An, W., Liu, X., Wang, X., Chen, X., Nie, X., Cheng, X., Liu, X., Xie, X., Liu, X., Yang, X., Li, X., Su, X., Lin, X., Li, X. Q., Jin, X., Shen, X., Chen,
- X., Sun, X., Wang, X., Song, X., Zhou, X., Wang, X., Shan, X., Li, Y. K., Wang, Y. Q., Wei, Y. X., Zhang, Y., Xu, Y., Li, Y., Zhao, Y., Sun, Y., Wang, Y., Yu, Y., Zhang,
- Y., Shi, Y., Xiong, Y., He, Y., Piao, Y., Wang, Y., Tan,

Bi, Z., Zhang, N., Jiang, Y., Deng, S., Zheng, G., and Chen, H. When do program-of-thoughts work for reasoning?, 2023. URL https://arxiv.org/abs/2308. 15452.

- Y., Ma, Y., Liu, Y., Guo, Y., Ou, Y., Wang, Y., Gong,

- Y., Zou, Y., He, Y., Xiong, Y., Luo, Y., You, Y., Liu, Y., Zhou, Y., Zhu, Y. X., Xu, Y., Huang, Y., Li, Y., Zheng,

- Y., Zhu, Y., Ma, Y., Tang, Y., Zha, Y., Yan, Y., Ren, Z. Z., Ren, Z., Sha, Z., Fu, Z., Xu, Z., Xie, Z., Zhang, Z., Hao,
- Z., Ma, Z., Yan, Z., Wu, Z., Gu, Z., Zhu, Z., Liu, Z., Li,

- Z., Xie, Z., Song, Z., Pan, Z., Huang, Z., Xu, Z., Zhang,

- Z., and Zhang, Z. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.

Biderman, S., Schoelkopf, H., Sutawika, L., Gao, L., Tow, J., Abbasi, B., Aji, A. F., Ammanamanchi, P. S., Black, S., Clive, J., DiPofi, A., Etxaniz, J., Fattori, B., Forde, J. Z., Foster, C., Hsu, J., Jaiswal, M., Lee, W. Y., Li, H., Lovering, C., Muennighoff, N., Pavlick, E., Phang, J., Skowron, A., Tan, S., Tang, X., Wang, K. A., Winata, G. I., Yvon, F., and Zou, A. Lessons from the trenches on reproducible evaluation of language models, 2024.

Brown, B., Juravsky, J., Ehrlich, R., Clark, R., Le, Q. V., Ré, C., and Mirhoseini, A. Large language monkeys: Scaling

Diao, S., Wang, P., Lin, Y., Pan, R., Liu, X., and Zhang, T. Active prompting with chain-of-thought for large lan-

guage models, 2024. URL https://arxiv.org/ abs/2302.12246.

Wang, Y., Arora, S., Atkinson, D., Authur, R., Chandu, K. R., Cohan, A., Dumas, J., Elazar, Y., Gu, Y., Hessel, J., Khot, T., Merrill, W., Morrison, J., Muennighoff, N., Naik, A., Nam, C., Peters, M. E., Pyatkin, V., Ravichander, A., Schwenk, D., Shah, S., Smith, W., Strubell, E., Subramani, N., Wortsman, M., Dasigi, P., Lambert, N., Richardson, K., Zettlemoyer, L., Dodge, J., Lo, K., Soldaini, L., Smith, N. A., and Hajishirzi, H. Olmo: Accelerating the science of language models, 2024.

Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., Goyal, A., Hartshorn, A., Yang, A., Mitra, A., Sravankumar, A., Korenev, A., Hinsvark, A., Rao, A., Zhang, A., Rodriguez, A., Gregerson, A., et al. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/ 2407.21783.

He, C., Luo, R., Bai, Y., Hu, S., Thai, Z. L., Shen, J., Hu, J., Han, X., Huang, Y., Zhang, Y., Liu, J., Qi, L., Liu, Z., and Sun, M. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems, 2024. URL https://arxiv.org/abs/2402.14008.

Fu, Y., Peng, H., Sabharwal, A., Clark, P., and Khot, T. Complexity-based prompting for multi-step reasoning, 2023. URL https://arxiv.org/abs/2210. 00720.

Gandhi, K., Lee, D., Grand, G., Liu, M., Cheng, W., Sharma, A., and Goodman, N. D. Stream of search (sos): Learning to search in language, 2024. URL https://arxiv. org/abs/2404.03683.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. Measuring mathematical problem solving with the math dataset, 2021. URL https://arxiv.org/abs/2103.03874.

Gao, B., Song, F., Yang, Z., Cai, Z., Miao, Y., Dong, Q., Li, L., Ma, C., Chen, L., Xu, R., Tang, Z., Wang, B., Zan, D., Quan, S., Zhang, G., Sha, L., Zhang, Y., Ren, X., Liu, T., and Chang, B. Omni-math: A universal olympiad level mathematic benchmark for large language models, 2024a. URL https://arxiv.org/abs/2410.07985.

Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Rae, J. W., Vinyals, O., and Sifre, L. Training compute-optimal large language models, 2022. URL https://arxiv.org/abs/2203.

Gao, L., Tow, J., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., McDonell, K., Muennighoff, N., Phang, J., Reynolds, L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A. A framework for few-shot language model evaluation, September 2021. URL https:// doi.org/10.5281/zenodo.5371628.

15556.

Hou, Z., Lv, X., Lu, R., Zhang, J., Li, Y., Yao, Z., Li, J., Tang, J., and Dong, Y. Advancing language model reasoning through reinforcement learning and inference scaling, 2025. URL https://arxiv.org/abs/2501.

Gao, Z., Niu, B., He, X., Xu, H., Liu, H., Liu, A., Hu, X., and Wen, L. Interpretable contrastive monte carlo tree search reasoning, 2024b. URL https://arxiv.org/abs/ 2410.01707.

11651.

Hu, Y., Shi, W., Fu, X., Roth, D., Ostendorf, M., Zettlemoyer, L., Smith, N. A., and Krishna, R. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models, 2024. URL https://arxiv.org/ abs/2406.09403.

Glazer, E., Erdil, E., Besiroglu, T., Chicharro, D., Chen, E., Gunning, A., Olsson, C. F., Denain, J.-S., Ho, A., de Oliveira Santos, E., Järviniemi, O., Barnett, M., Sandler, R., Vrzala, M., Sevilla, J., Ren, Q., Pratt, E., Levine,

Huang, Z., Wang, Z., Xia, S., Li, X., Zou, H., Xu, R., Fan, R.-Z., Ye, L., Chern, E., Ye, Y., Zhang, Y., Yang, Y., Wu, T., Wang, B., Sun, S., Xiao, Y., Li, Y., Zhou, F., Chern, S., Qin, Y., Ma, Y., Su, J., Liu, Y., Zheng, Y., Zhang, S., Lin, D., Qiao, Y., and Liu, P. Olympicarena: Benchmarking multi-discipline cognitive reasoning for superintelligent ai, 2024a. URL https://arxiv.org/abs/2406.

- L., Barkley, G., Stewart, N., Grechuk, B., Grechuk, T., Enugandla, S. V., and Wildon, M. Frontiermath: A benchmark for evaluating advanced mathematical reasoning in ai, 2024. URL https://arxiv.org/abs/2411. 04872.

Google. Gemini 2.0 flash thinking mode (gemini-2.0flash-thinking-exp-1219), December 2024. URL https://cloud.google.com/vertex-ai/ generative-ai/docs/thinking-mode.

12753.

Huang, Z., Zou, H., Li, X., Liu, Y., Zheng, Y., Chern, E., Xia, S., Qin, Y., Yuan, W., and Liu, P. O1 replication journey – part 2: Surpassing o1-preview through simple distillation, big progress or bitter lesson?, 2024b. URL https://arxiv.org/abs/2411.16489.

Groeneveld, D., Beltagy, I., Walsh, P., Bhagia, A., Kinney, R., Tafjord, O., Jha, A. H., Ivison, H., Magnusson, I.,

Huang, Z., Geng, G., Hua, S., Huang, Z., Zou, H., Zhang, S., Liu, P., and Zhang, X. O1 replication journey – part 3: Inference-time scaling for medical reasoning, 2025. URL https://arxiv.org/abs/2501.06458.

aimo-progress-prize/blob/main/report/ numina_dataset.pdf.

Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step, 2023. URL https: //arxiv.org/abs/2305.20050.

Irvine, R., Boubert, D., Raina, V., Liusie, A., Zhu, Z., Mudupalli, V., Korshuk, A., Liu, Z., Cremer, F., Assassi, V., Beauchamp, C.-C., Lu, X., Rialan, T., and Beauchamp, W. Rewarding chatbots for real-world engagement with millions of users, 2023. URL https: //arxiv.org/abs/2303.06135.

Ling, W., Yogatama, D., Dyer, C., and Blunsom, P. Program induction by rationale generation : Learning to solve and explain algebraic word problems, 2017. URL https: //arxiv.org/abs/1705.04146.

Jain, N., Han, K., Gu, A., Li, W.-D., Yan, F., Zhang, T., Wang, S., Solar-Lezama, A., Sen, K., and Stoica, I. Livecodebench: Holistic and contamination free evaluation of large language models for code, 2024. URL https://arxiv.org/abs/2403.07974.

Liu, J., Cui, L., Liu, H., Huang, D., Wang, Y., and Zhang, Y. Logiqa: A challenge dataset for machine reading comprehension with logical reasoning, 2020. URL https: //arxiv.org/abs/2007.08124.

Kapfer, C., Stine, K., Narasimhan, B., Mentzel, C., and Candes, E. Marlowe: Stanford’s gpu-based computational instrument, January 2025. URL https://doi.org/ 10.5281/zenodo.14751899.

Liu, J., Cohen, A., Pasunuru, R., Choi, Y., Hajishirzi, H., and Celikyilmaz, A. Don’t throw away your value model! generating more preferable text with value-guided monte-carlo tree search decoding, 2024. URL https:

//arxiv.org/abs/2309.15028. Loshchilov, I. and Hutter, F. Decoupled weight decay regularization, 2019.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J., and Amodei, D. Scaling laws for neural language models, 2020. URL https://arxiv.org/abs/2001.

08361.

Luo, H., Sun, Q., Xu, C., Zhao, P., Lou, J., Tao, C., Geng, X., Lin, Q., Chen, S., Tang, Y., and Zhang, D. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct, 2025. URL https:

Kim, E., Suk, J., Kim, S., Muennighoff, N., Kim, D., and Oh, A. Llm-as-an-interviewer: Beyond static testing through dynamic llm evaluation, 2024. URL https://arxiv. org/abs/2412.10424.

//arxiv.org/abs/2308.09583.

Marten, R., Vu, T., Ji, C. C.-J., Sharma, K., Pimpalgaonkar, S., Dimakis, A., and Sathiamoorthy, M. Curator: A tool for synthetic data creation. https://github.com/ bespokelabsai/curator, January 2025.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention, 2023. URL https://arxiv.

org/abs/2309.06180.

Muennighoff, N., Soldaini, L., Groeneveld, D., Lo, K., Morrison, J., Min, S., Shi, W., Walsh, P., Tafjord, O., Lambert, N., Gu, Y., Arora, S., Bhagia, A., Schwenk, D., Wadden, D., Wettig, A., Hui, B., Dettmers, T., Kiela, D., Farhadi, A., Smith, N. A., Koh, P. W., Singh, A., and Hajishirzi, H. Olmoe: Open mixture-of-experts language models, 2024. URL https://arxiv.org/abs/2409.02060.

Labs, B. Bespoke-stratos: The unreasonable effectiveness of reasoning distillation, 2025. URL https://hf. co/bespokelabs/Bespoke-Stratos-32B. Accessed: 2025-01-22.

Lee, K.-H., Fischer, I., Wu, Y.-H., Marwood, D., Baluja, S., Schuurmans, D., and Chen, X. Evolving deeper llm thinking, 2025. URL https://arxiv.org/abs/ 2501.09891.

of America, M. A. Aime, February 2024. URL https: //artofproblemsolving.com/wiki/index. php/AIME_Problems_and_Solutions/.

Levi, N. A simple model of inference scaling laws, 2024. URL https://arxiv.org/abs/2410.16377.

OpenAI. Learning to reason with llms, September 2024. URL https://openai.com/index/ learning-to-reason-with-llms/.

- LI, J., Beeching, E., Tunstall, L., Lipkin, B., Soletskyi, R., Huang, S. C., Rasul, K., Yu, L., Jiang, A., Shen, Z., Qin, Z., Dong, B., Zhou, L., Fleureau, Y., Lample, G., and Polu, S. Numinamath, 2024. URL https://github.com/project-numina/

OpenAI. Openai o3-mini, 2025. URL https://openai. com/index/openai-o3-mini/. Accessed: 202502-24.

Phan, L., Gatti, A., Han, Z., Li, N., Hu, J., Zhang, H., Shi, S., Choi, M., Agrawal, A., Chopra, A., et al. Humanity’s last exam, 2025. URL https://arxiv.org/abs/ 2501.14249.

H., Yuan, H., Chen, J., Guo, J., Su, J., Wang, J., Zhao, J., Zhang, J., Liu, J., Yan, J., Wu, J., Shi, L., Ye, L., Yu, L., Dong, M., Zhang, N., Ma, N., Pan, Q., Gong, Q., Liu, S., Ma, S., Wei, S., Cao, S., Huang, S., Jiang, T., Gao, W., Xiong, W., He, W., Huang, W., Wu, W., He, W., Wei, X., Jia, X., Wu, X., Xu, X., Zu, X., Zhou, X., Pan, X., Charles, Y., Li, Y., Hu, Y., Liu, Y., Chen, Y., Wang, Y., Liu, Y., Qin, Y., Liu, Y., Yang, Y., Bao, Y., Du, Y., Wu, Y., Wang, Y., Zhou, Z., Wang, Z., Li, Z., Zhu, Z., Zhang, Z., Wang, Z., Yang, Z., Huang, Z., Huang, Z., Xu, Z., and Yang, Z. Kimi k1.5: Scaling reinforcement learning with llms, 2025. URL https://arxiv.org/abs/2501.12599.

Qin, Y., Li, X., Zou, H., Liu, Y., Xia, S., Huang, Z., Ye, Y., Yuan, W., Liu, H., Li, Y., and Liu, P. O1 replication journey: A strategic progress report – part 1, 2024. URL https://arxiv.org/abs/2410.18982.

Qwen, :, Yang, A., Yang, B., Zhang, B., Hui, B., Zheng,

- B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L., Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R., Li, T., Xia, T., Ren, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Wan, Y., Liu, Y., Cui, Z., Zhang, Z., and Qiu, Z. Qwen2.5 technical report, 2024. URL https://arxiv.org/ abs/2412.15115.

Team, N. Sky-t1: Fully open-source reasoning model with o1-preview performance in $450 budget, 2025. URL https://novasky-ai.github.io/ posts/sky-t1. Accessed: 2025-01-09.

Team, Q. Qwq: Reflect deeply on the boundaries of the unknown, November 2024. URL https://qwenlm. github.io/blog/qwq-32b-preview/.

Rein, D., Hou, B. L., Stickland, A. C., Petty, J., Pang, R. Y., Dirani, J., Michael, J., and Bowman, S. R. Gpqa: A graduate-level google-proof q&a benchmark, 2023. URL https://arxiv.org/abs/2311.12022.

Wang, J., Meng, F., Liang, Y., and Zhou, J. Drt-o1: Optimized deep reasoning translation via long chain-ofthought, 2024a. URL https://arxiv.org/abs/ 2412.17498.

Shi, Q., Tang, M., Narasimhan, K., and Yao, S. Can language models solve olympiad programming?, 2024. URL https://arxiv.org/abs/2404.10952.

Wang, P., Li, L., Shao, Z., Xu, R. X., Dai, D., Li, Y., Chen, D., Wu, Y., and Sui, Z. Math-shepherd: Verify and reinforce llms step-by-step without human annotations, 2024b. URL https://arxiv.org/abs/2312.08935.

Snell, C., Lee, J., Xu, K., and Kumar, A. Scaling llm testtime compute optimally can be more effective than scaling model parameters, 2024. URL https://arxiv. org/abs/2408.03314.

Wang, S., Liu, Z., Zhong, W., Zhou, M., Wei, Z., Chen, Z., and Duan, N. From lsat: The progress and challenges of complex reasoning, 2021. URL https://arxiv. org/abs/2108.00648.

Srivastava, A., Rastogi, A., Rao, A., Shoeb, A. A. M., Abid, A., Fisch, A., Brown, A. R., Santoro, A., Gupta, A., Garriga-Alonso, A., et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models, 2023.

Wang, Z., Dong, Y., Delalleau, O., Zeng, J., Shen, G., Egert, D., Zhang, J. J., Sreedhar, M. N., and Kuchaiev, O. Helpsteer2: Open-source dataset for training top-performing reward models, 2024c. URL https://arxiv.org/ abs/2406.08673.

Su, H., Yen, H., Xia, M., Shi, W., Muennighoff, N., yu Wang, H., Liu, H., Shi, Q., Siegel, Z. S., Tang, M., Sun, R., Yoon, J., Arik, S. O., Chen, D., and Yu, T. Bright: A realistic and challenging benchmark for reasoning-intensive retrieval, 2024. URL https://arxiv.org/abs/ 2407.12883.

Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B., Xia, F., Chi, E., Le, Q., and Zhou, D. Chain-ofthought prompting elicits reasoning in large language models, 2023. URL https://arxiv.org/abs/ 2201.11903.

Sun, L., Han, Y., Zhao, Z., Ma, D., Shen, Z., Chen, B., Chen, L., and Yu, K. Scieval: A multi-level large language model evaluation benchmark for scientific research, 2024. URL https://arxiv.org/abs/2308.13149.

Team, K., Du, A., Gao, B., Xing, B., Jiang, C., Chen, C., Li, C., Xiao, C., Du, C., Liao, C., Tang, C., Wang, C., Zhang, D., Yuan, E., Lu, E., Tang, F., Sung, F., Wei, G., Lai, G., Guo, H., Zhu, H., Ding, H., Hu, H., Yang, H., Zhang, H., Yao, H., Zhao, H., Lu, H., Li, H., Yu, H., Gao, H., Zheng,

Welleck, S., Bertsch, A., Finlayson, M., Schoelkopf, H., Xie, A., Neubig, G., Kulikov, I., and Harchaoui, Z. From decoding to meta-generation: Inference-time algorithms for large language models, 2024. URL https://arxiv.

org/abs/2406.16838.

Wu, T., Lan, J., Yuan, W., Jiao, J., Weston, J., and Sukhbaatar, S. Thinking llms: General instruction following with thought generation, 2024a. URL https: //arxiv.org/abs/2410.10630.

Yu, L., Jiang, W., Shi, H., Yu, J., Liu, Z., Zhang, Y., Kwok, J. T., Li, Z., Weller, A., and Liu, W. Metamath: Bootstrap your own mathematical questions for large language models, 2024. URL https://arxiv.org/abs/2309.

12284.

Wu, Y., Sun, Z., Li, S., Welleck, S., and Yang, Y. Inference scaling laws: An empirical analysis of computeoptimal inference for problem-solving with language models, 2024b. URL https://arxiv.org/abs/2408.

Yuan, S., Chen, Z., Xi, Z., Ye, J., Du, Z., and Chen, J. Agent-r: Training language model agents to reflect via iterative self-training, 2025. URL https://arxiv. org/abs/2501.11425.

00724.

Xiang, V., Snell, C., Gandhi, K., Albalak, A., Singh, A., Blagden, C., Phung, D., Rafailov, R., Lile, N., Mahan, D., Castricato, L., Franken, J.-P., Haber, N., and Finn, C. Towards system 2 reasoning in llms: Learning how to think with meta chain-of-thought, 2025. URL https:

Zelikman, E., Wu, Y., Mu, J., and Goodman, N. D. Star: Bootstrapping reasoning with reasoning, 2022. URL https://arxiv.org/abs/2203.14465.

Zelikman, E., Harik, G., Shao, Y., Jayasiri, V., Haber, N., and Goodman, N. D. Quiet-star: Language models can teach themselves to think before speaking, 2024. URL https://arxiv.org/abs/2403.09629.

//arxiv.org/abs/2501.04682.

Xie, Y., Kawaguchi, K., Zhao, Y., Zhao, X., Kan, M.-Y., He, J., and Xie, Q. Self-evaluation guided beam search for reasoning, 2023. URL https://arxiv.org/abs/ 2305.00633.

Zhang, H. and Chen, C. Test-time compute scaling laws,

2024. URL https://github.com/hughbzhang/ o1_inference_scaling_laws.

Xin, H., Guo, D., Shao, Z., Ren, Z., Zhu, Q., Liu, B., Ruan, C., Li, W., and Liang, X. Deepseek-prover: Advancing theorem proving in llms through large-scale synthetic data, 2024. URL https://arxiv.org/abs/2405.

Zhang, S., Chen, Z., Shen, Y., Ding, M., Tenenbaum, J. B., and Gan, C. Planning with large language models for code generation, 2023. URL https://arxiv.org/ abs/2303.05510.

14333.

Zhang, Y., Wu, S., Yang, Y., Shu, J., Xiao, J., Kong, C., and Sang, J. o1-coder: an o1 replication for coding, 2024a. URL https://arxiv.org/abs/2412.00154.

Xu, H., Wu, X., Wang, W., Li, Z., Zheng, D., Chen, B., Hu, Y., Kang, S., Ji, J., Zhang, Y., Guo, Z., Yang, Y., Zhang, M., and Zhang, D. Redstar: Does scaling long-cot data unlock better slow-reasoning systems?, 2025. URL https://arxiv.org/abs/2501.11284.

Zhang, Y., Yang, J., Yuan, Y., and Yao, A. C.-C. Cumulative reasoning with large language models, 2024b. URL https://arxiv.org/abs/2308.04371.

Yang, Z., Band, N., Li, S., Candès, E., and Hashimoto, T. Synthetic continued pretraining, 2024. URL https: //arxiv.org/abs/2409.07431.

Zhong, H., Xiao, C., Tu, C., Zhang, T., Liu, Z., and Sun, M. Jec-qa: A legal-domain question answering dataset, 2019. URL https://arxiv.org/abs/1911.12011.

Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T. L., Cao, Y., and Narasimhan, K. Tree of thoughts: Deliberate problem solving with large language models, 2023a. URL https://arxiv.org/abs/2305.10601.

Zhong, W., Cui, R., Guo, Y., Liang, Y., Lu, S., Wang, Y., Saied, A., Chen, W., and Duan, N. Agieval: A human-centric benchmark for evaluating foundation models, 2023. URL https://arxiv.org/abs/2304.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., and Cao, Y. React: Synergizing reasoning and acting in language models, 2023b. URL https://arxiv. org/abs/2210.03629.

06364.

Zhou, A., Yan, K., Shlapentokh-Rothman, M., Wang, H., and Wang, Y.-X. Language agent tree search unifies reasoning acting and planning in language models, 2024. URL https://arxiv.org/abs/2310.04406.

Ye, Y., Huang, Z., Xiao, Y., Chern, E., Xia, S., and Liu, P. Limo: Less is more for reasoning, 2025a. URL https: //arxiv.org/abs/2502.03387.

Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X., Efrat, A., Yu, P., Yu, L., Zhang, S., Ghosh, G., Lewis, M., Zettlemoyer, L., and Levy, O. Lima: Less is more for alignment, 2023. URL https://arxiv.org/abs/ 2305.11206.

Ye, Y., Xiao, Y., Mi, T., and Liu, P. Aime-preview: A rigorous and immediate evaluation framework for advanced mathematical reasoning. https://github. com/GAIR-NLP/AIME-Preview, 2025b. GitHub repository.

### Contents

- 1 Introduction 1
- 2 Reasoning data curation to create s1K 2

- 2.1 Initial collection of 59K samples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
- 2.2 Final selection of 1K samples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2

- 3 Test-time scaling 3

- 3.1 Method . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
- 3.2 Metrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4

- 4 Results 4

- 4.1 Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
- 4.2 Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 5 Ablations 6

- 5.1 Data Quantity, Diversity, and Difficulty . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6
- 5.2 Test-time scaling methods . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 6 Discussion and related work 8

- 6.1 Sample-efficient reasoning . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 6.2 Test-time scaling . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- A s1.1 16
- B Evaluation determinism 16
- C s1K details 17

- C.1 s1K summary . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- C.2 Dataset composition for full 59K questions . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- C.3 s1K grading prompt . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- C.4 s1K diversity selection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19
- C.5 Decontamination . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- D Training details 21

- D.1 Training Ablations: Sequence length . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- D.2 Training Samples . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- E Test-time scaling details 35

- E.1 Sequential scaling ablations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 35
- E.2 Examples for rejection sampling ablation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 37

- F Version Control 46

### A. s1.1

Seven days after our release of s1, we released s1.1. We regenerated traces for our 1,000 samples in s1K using DeepSeek r1 (DeepSeek-AI et al., 2025) to create s1K-1.1. We use the same training procedure to train our model s1.1. Other updates since our launch include the release of o3 (OpenAI, 2025), LIMO (Ye et al., 2025a), and AIME 2025. We consider all these new developments in Table 5. We find that s1.1 performs significantly better than s1. We also tried distilling from Claude 3.7, which led to worse performance than from r1 (not reported).

- Table 5. s1-32B is an open and sample-efficient reasoning model. We evaluate s1-32B, Qwen, and Gemini (some entries are unknown (N.A.), see §4). Other results are from the respective reports (Qwen et al., 2024; Team, 2024; OpenAI, 2024; 2025; DeepSeek-AI et al., 2025; Labs, 2025; Team, 2025) except for AIME 2025 (Ye et al., 2025b). # ex. = number examples used for reasoning finetuning; BF = budget forcing.

Model # Examples MATH500 GPQA AIME 2024 AIME 2025 API only

- o3-mini-low N/A 95.8 70.6 56.3 42.1
- o3-mini-medium N/A 97.3 76.8 75.8 70.4
- o3-mini-high N/A 97.9 79.7 83.8 80.9 Open Weights

QwQ-32B N.A. 90.6 54.5 46.7 32.7

- r1 ≫800K 07.3 71.5 79.8 70.0

- r1-distill-Llama-70B 800K 94.5 65.2 57.1 56.3

- r1-distill-Qwen-14B 800K 93.9 59.1 61.7 48.0

- r1-distill-Qwen-32B 800K 94.3 62.1 58.3 49.6 Open Weights and Open Data

LIMO 817 94.8 66.7 56.3 44.6

- s1 w/o BF 1K 92.6 56.6 50.0 26.7

- s1 with Budget Forcing “Wait” 1x 1K 92.8 59.6 53.3 30.0

- s1 with Budget Forcing “Wait” 2x 1K 93.0 59.6 53.3 33.3

- s1 with Budget Forcing “Wait” 4x 1K 92.2 58.6 56.7 36.7 s1.1 w/o BF 1K 94.4 60.6 56.7 50.0

- s1.1 with Budget Forcing “Wait” 1x 1K 95.4 62.6 56.7 50.0
- s1.1 with Budget Forcing “Wait” 2x 1K 95.4 63.6 56.7 50.0

### B. Evaluation determinism

We run our evaluations using vLLM (Kwon et al., 2023) as it is faster than the alternatives we tried. However, we find that even when using the same random seeds and greedy sampling, evaluation scores can change significantly across runs:

- • Different batch sizes causing different results see https://github.com/vllm-project/vllm/issues/ 5898
- • Continuing generations causing different results see https://github.com/vllm-project/vllm/issues/ 11783
- • Changes in tensor parallelism causing different results

As our model generates long reasoning traces prior to its answer, small numeric changes can snowball into large differences. We encounter many generations that are exactly the same for thousands of tokens and then suddenly differ in one token eventually ending up with an entirely different answer. To partly counter this issue we generally run our final evaluations using full precision unless otherwise indicated.

### C. s1K details

#### C.1. s1K summary

- Table 6. Summary of our dataset s1K. Token count measured by the Qwen-2.5 tokenizer. We prompt Claude to produce keywords given several questions from the domain.

Domain #questions Total token count Keywords Geometry 109 560.2K Area, Triangle, Distance Number theory 98 522.5K Sequences, Divisibility Combinatorics 75 384.7K Permutations, Counting Real functions 43 234.8K Trigonometry, Calculus Biology 41 120.9K Organic reactions Complex functions 32 170.2K Complex roots Quantum theory 32 127.9K Particles, Wave functions Field theory 28 150.1K Polynomials, Roots Calculus of variations 28 155.5K Optimization, Control Difference equations 24 132.5K Recurrence, Recursion Electromagnetic theory 23 95.8K Optics, Waves, Diffraction Group theory 22 100.0K Groups, Automorphisms Linear algebra 22 128.3K Matrices, Determinants Probability theory 20 114.6K Random walk, Expectation Algebraic systems 19 109.9K Functional equations Mechanics 19 103.6K Forces, Motion, Energy Thermodynamics 19 74.2K Heat engines, Entropy Differential equations 18 89.6K Substitution, Existence Computer science 18 34.2K Complexity theory, Algorithms Numerical analysis 18 76.5K Error analysis, Stability Calculus 17 96.3K Convergence, Summation Algebraic structures 17 90.4K Inequalities, Sets Astronomy 16 37.7K Stellar populations, Orbits Remaining 27 domains 242 982.2K Domains with ≤ 16 questions All domains (51) 1000 4.7M s1K

#### C.2. Dataset composition for full 59K questions

- Table 7. Composition of full 59K questions. Thinking and response lengths are measured in tokens using the Qwen2.5-32B-Instruct tokenizer (Qwen et al., 2024). In addition to excluding our evaluation benchmark, AIME24, we also exclude AIME questions from 2022-2023 as we use these 90 questions during our development stage of s1-32B.

Source Description #Samples Avg. thinking length

NuminaMATH (LI et al., 2024) Math problems from online websites 30660 4.1K MATH (Hendrycks et al., 2021) Math problems from competitions 11999 2.9K OlympicArena (Huang et al.,

Astronomy, Biology, Chemistry, Computer Science, Geography, Math, and Physics olympiad questions

4250 3.2K

2024a)

OmniMath (Gao et al., 2024a) Math problems from competitions 4238 4.4K AGIEval (Zhong et al., 2023; Ling et al., 2017; Hendrycks et al., 2021; Liu et al., 2020; Zhong et al., 2019; Wang et al., 2021)

English, Law, Logic and Math problems from the SAT, LSAT and other exams

2385 1.2K

xword Crossword puzzles 999 0.7K OlympiadBench (He et al., 2024) Math and Physics olympiad questions 896 3.9K AIME (1983-2021) American Invitational Mathematics Ex-

890 4.7K

amination

TheoremQA (Chen et al., 2023) Computer Science, Finance, Math, and Physics university-level questions relating to theorems

747 2.1K

USACO (Shi et al., 2024) Code problems from the USA Computing Olympiad

519 3.6K

JEEBench (Arora et al., 2023) Chemistry, Math, and Physics problems used in the university entrance examination of the Indian Institute of Technology

515 2.9K

GPQA (Rein et al., 2023) PhD-Level Science Questions 348 2.9K SciEval (Sun et al., 2024) Biology, Chemistry, and Physics prob-

227 0.7K

lems from various sources

s1-prob Stanford statistics qualifying exams 182 4.0K LiveCodeBench (Jain et al., 2024) Code problems from coding websites

151 3.5K s1-teasers Math brain-teasers crawled from the In-

(LeetCode, AtCoder, and CodeForces)

23 4.1K

ternet

All 59K questions Composite of the above datasets with

59029 3.6K

reasoning traces and solutions

#### C.3. s1K grading prompt

To grade whether an example is correct for our dataset selection in §2, we use the prompt in Figure 8. We grade using Claude 3.5 except for the correctness among the final 1,000 samples, which we graded with Claude 3.7.

You are an AI assistant for grading a science problem. The user will provide you with the question itself, an attempt made by a student and the correct answer to the problem. Your job is to judge whether the attempt is correct by comparing it with the correct answer. If the expected solution concludes with a number or choice, there should be no ambiguity. If the expected solution involves going through the entire reasoning process, you should judge the attempt based on whether the reasoning process is correct with correct answer if helpful.

The user will provide the attempt and the correct answer in the following format:

# Problem {problem}

## Attempt {attempt}

## Correct answer {solution}

Explain your reasoning, and end your response on a new line with only "Yes" or "No" (without quotes).

Figure 8. Grading prompt.

#### C.4. s1K diversity selection

Algorithm 1 provides our algorithm for selecting data in our diversity selection stage. As mentioned in §2, we also include samples from some specific benchmarks we perceive as high-quality. None of the samples overlap with our final evaluation.

#### C.5. Decontamination

We filter all samples by checking for an 8-gram overlap between the selected examples and the evaluation benchmarks: MATH500, GPTQA Diamond, and AIME24. We exclude questions with more than an 8-gram overlap.

Algorithm 1 Two-stage sampling for s1K

- 1: Input:  := Set of 24,496 questions with features
- 2: Output:  := Set of 1,000 selected questions
- 3:  ← ∅ Initialize the output set (only tracks unique elements)
- 4: for 𝑞 ∈  do
- 5: if IsGeminiCorrect(𝑞) and (IsAIME(𝑞) or IsGPQA(𝑞)) then
- 6:  ←  ∪ {𝑞}
- 7: Select all correct AIME/GPQA solutions
- 8: else if IsGeminiCorrect(𝑞) and IsMATH(𝑞) and ThinkingLength(𝑞) > 5600 then
- 9:  ←  ∪ {𝑞}
- 10: Select correct MATH500 solutions with long chains
- 11: end if
- 12: end for
- 13:  ← All available domains
- 14: Initialize domain pool
- 15: while || < 1000 do
- 16: 𝑑 ← RandomChoice()
- 17: Randomly select a domain
- 18: 𝑄𝑑 ← Questions in domain 𝑑
- 19: Get questions from this domain
- 20: ranks ← RankByThinkingLength(𝑄𝑑)
- 21: Rank by thinking length
- 22: weights ← 2−ranks
- 23: Apply power-law weighting
- 24: 𝑞 ← WeightedSample(𝑄𝑑, weights)
- 25: Sample favoring longer chains
- 26:  ←  ∪ {𝑞}
- 27: Add selected question
- 28: 𝑄𝑑 ← 𝑄𝑑 ⧵ {𝑞}
- 29: if 𝑄𝑑 = ∅ then
- 30:  ←  ⧵ {𝑑}
- 31: Remove exhausted domains
- 32: end if
- 33: end while

### D. Training details

We take a model that has already been pretrained and instruction tuned and further finetune it for reasoning. Specifically, we use Qwen2.5-32B-Instruct (Qwen et al., 2024), which on math tasks generally matches or outperforms the larger Qwen2.572B-Instruct (Qwen et al., 2024) or other open models (Dubey et al., 2024; Groeneveld et al., 2024; Muennighoff et al., 2024). We use token delimiters to separate the thinking stage from the answering stage. We enclose the thinking stage with <|im_start|>think and <|im_start|>answer; both preceded and followed by a newline. Samples from our dataset are in §D.2. We use basic fine-tuning hyperparameters: we train for 5 epochs with a batch size of 16 for a total of 315 gradient steps. We train in bfloat16 precision with a learning rate of 1𝑒 − 5 warmed up linearly for 5% (16 steps) and then decayed to 0 over the rest of training (299 steps) following a cosine schedule. We use the AdamW optimizer (Loshchilov & Hutter, 2019) with 𝛽1 = 0.9,𝛽2 = 0.95 and weight decay of 1𝑒 − 4. We do not compute loss on questions, only on reasoning traces and solutions. We ensure the sequence length is large enough to avoid cutting off any samples; a setting we ablate in §D.1. The training takes just 26 minutes on 16 NVIDIA H100 GPUs.

Learning Rate Schedule

Training Loss

Gradient Norm

1e 5

1.0

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

GradientNorm

LearningRate

0.4

- 1

- 2

- 3

Loss

0.5

0.2

0.0

0.0

0 100 200 300

0 100 200 300

0 100 200 300

Training Steps

Training Steps

Training Steps

Figure 9. Training dynamics of s1-32B on s1K.

#### D.1. Training Ablations: Sequence length

- Table 8. Training sequence length ablation. We report “accuracy / average thinking tokens per sample”; the higher the accuracy and the fewer the thinking tokens (inference cost) the better.

Model A Model B

Training sequence length 4096 32768 % training samples cutoff 74% 0%

AIME24 30.0% / 20721 50.0% / 6984 MATH500 90.0% / 5324 91.0% / 3268 GPQA 52.5% / 6841 53.0% / 3568

Besides our scaling ablations in §5.2, the main training hyperparameter we ablate is the sequence length used during training. We find that a shorter training sequence length leads to longer reasoning traces at test time. This is because when training with a shorter sequence length the answer section of the training sample is more commonly cut off. Inversely, when the training sequence length is longer, more samples appear in their entirety with the section where the model answers. Thus the model receives more gradient updates where it learns to generate an answer following its chain. This in turn leads to a higher log probability of the answer section at any point during the generation and thus shorter reasoning traces at test time. Performance-wise, we also find that the model trained with a longer sequence length performs better. Thus we opt for the longest training sequence length as it leads to better performance and makes inference more efficient by leading to shorter reasoning traces.

D.2. Training Samples Table 9, Table 10, Table 11 contain training samples from s1K.

- Table 9. s1K sample stemming from s1-prob. The black text is the question and the yellow text is the solution. Based solely on the question we generate the light blue reasoning trace and blue response with Gemini, which we use for training.

You are given integrable random variables 𝑋,𝑌0 and 𝑍0 on the same probability space (Ω,,ℙ), and two 𝜎-algebras  ⊂ , ⊂ . For 𝑘 = 1,2,…,, let

𝑌𝑘 ∶= 𝔼[𝑋|𝜎(,𝑍0,…,𝑍𝑘−1)], 𝑍𝑘 ∶= 𝔼[𝑋|𝜎(,𝑌0,…,𝑌𝑘−1)].

(a) Show that there exist integrable random variables 𝑌∞ and 𝑍∞ such that as 𝑛 → ∞ both 𝔼[|𝑌𝑛 − 𝑌∞|] → 0 and 𝔼[|𝑍𝑛 − 𝑍∞|] → 0. (b) Prove that almost surely 𝑌∞ = 𝑍∞.

- 1. Introduce the notations, 𝑛 ∶= 𝜎(,𝑍0,…,𝑍𝑛−1) and 𝑛 ∶= 𝜎(,𝑌0,…,𝑌𝑛−1), for all 𝑛 ≥ 1 and 0 ∶= , 0 ∶= . Clearly, both are filtrations and by definition, 𝑌𝑛 = 𝔼[𝑋|𝑛

]

and 𝑍𝑛 = 𝔼[𝑋|𝑛

]

, for all 𝑛 ≥ 0. Thus

{𝑌𝑛}𝑛≥0 (or {𝑍𝑛}𝑛≥0 ) is a Doob’s Martingale with respect to filtration

{

𝑛

}𝑛≥0 (or

{

𝑛

}𝑛≥0) and hence is U.I., 𝑋 being integrable. Now apply the fact that a Martingale is U.I. if and only if it converges in 𝐿1 and conclude.

- 2. We provide two different proofs.

- 1. Observe the following identity which follows from Tower Property.

𝔼[𝑌𝑛|𝑛

]

= 𝔼[𝔼[𝑋|𝑛

]

|𝑛

]

= 𝔼[𝔼[𝑋|𝑛

]

|𝑛

]

= 𝔼[𝑍𝑛|𝑛

]. (4)

Note that 𝔼|𝔼[𝑌𝑛|𝑛

]

− 𝔼[𝑌∞|𝑛

]

| ≤ 𝔼|𝑌𝑛 − 𝑌∞| = 𝑜(1) and by Levy’s Upward Theorem, 𝔼[𝑌∞|𝑛

] ⟶𝐿1 𝔼[𝑌∞|∞

]

, where ∞ = 𝜎 (

∪𝑛≥0𝑛

)

= 𝜎(,𝑌0,𝑌1 …). It is evident that 𝑌∞ ∈ 𝑚∞ (since it is almost sure limit of 𝑌𝑛 ∈ 𝑚𝑛) and hence 𝔼[𝑌∞|∞

]

= 𝑌∞ almost surely. Combining the above observations we conclude that 𝔼[𝑌𝑛|𝑛

] ⟶𝐿1 𝑌∞. Similarly, 𝔼[𝑍𝑛|𝑛

] ⟶𝐿1 𝑍∞. Now invoke Equation (4), to conclude that 𝑌∞ = 𝑍∞ almost surely.

- 2. Since 𝑋 is integrable, we have by Levy’s Upward Theorem, 𝑌∞ = lim𝑛→∞ 𝑌𝑛 = lim𝑛→∞ 𝔼[𝑋|𝑛

= 𝔼[𝑋|∞

]. Similarly, 𝑍∞ = 𝔼[𝑋|∞

]

]. On the otherhand, in first proof we have observed that 𝑌∞ ∈ 𝑚∞ and hence 𝑌∞ ∈ 𝑚∞ ∩ 𝑚∞ = 𝑚(

∞ ∩ ∞

)

. This yields the following.

= 𝔼[𝑋|∞ ∩ ∞

|∞ ∩ ∞

= 𝔼[𝔼[𝑋|∞

𝑌∞ = 𝔼[𝑌∞|∞ ∩ ∞

].

]

]

]

Similarly, 𝑍∞ = 𝔼[𝑋|∞ ∩ ∞

]. This proves that 𝑌∞ = 𝑍∞ almost surely.

Let’s analyze the problem. We are given two sequences of random variables 𝑌𝑘 and 𝑍𝑘 defined by conditional expectations. We need to show that these sequences converge in 𝐿1 and that their limits are equal almost surely.

- Part (a): Convergence in 𝐿1. The sigma-algebras involved are increasing:

𝑘 = 𝜎(,𝑍0,…,𝑍𝑘−1) 𝑘 = 𝜎(,𝑌0,…,𝑌𝑘−1) For the 𝑌𝑘 sequence:

- 𝑌1 = 𝔼[𝑋 𝜎( )]
- 𝑌2 = 𝔼[𝑋 𝜎( ,𝑍0)]
- 𝑌3 = 𝔼[𝑋|𝜎(,𝑍0,𝑍1)]

...

The sequence of sigma-algebras 𝜎(,𝑍0,…,𝑍𝑘−1) is increasing. By the Martingale Convergence Theorem (for conditional expectations), since 𝑋 is integrable, the sequence 𝑌𝑘 converges almost surely and in 𝐿1 to 𝔼[𝑋|𝜎(,𝑍0,𝑍1,…)]. Let 𝑌∞ = 𝔼[𝑋|𝜎(,𝑍0,𝑍1,…)].

Similarly, for the 𝑍𝑘 sequence:

- 𝑍1 = 𝔼[𝑋 𝜎( )]
- 𝑍2 = 𝔼[𝑋 𝜎( ,𝑌0)]
- 𝑍3 = 𝔼[𝑋|𝜎(,𝑌0,𝑌1)]

... The sequence of sigma-algebras 𝜎(,𝑌0,…,𝑌𝑘−1) is increasing. By the Martingale Convergence Theorem, the sequence 𝑍𝑘 converges almost surely and in 𝐿1 to 𝔼[𝑋|𝜎(,𝑌0,𝑌1,…)]. Let 𝑍∞ = 𝔼[𝑋|𝜎(,𝑌0,𝑌1,…)].

- Part (b): Prove that almost surely 𝑌∞ = 𝑍∞. We have 𝑌∞ = 𝔼[𝑋|𝜎(,𝑍0,𝑍1,…)] and 𝑍∞ = 𝔼[𝑋|𝜎(,𝑌0,𝑌1,…)]. Let ∞ = 𝜎(,𝑍0,𝑍1,…) and ∞ = 𝜎(,𝑌0,𝑌1,…). Then 𝑌∞ = 𝔼[𝑋|∞] and 𝑍∞ = 𝔼[𝑋|∞]. Consider 𝑌1 = 𝔼[𝑋|]. 𝑍1 = 𝔼[𝑋|]. 𝑌2 = 𝔼[𝑋|𝜎(,𝑍0)]. Since 𝑍0 = 𝔼[𝑋|], 𝜎(,𝑍0) = 𝜎(,𝔼[𝑋|]). 𝑍2 = 𝔼[𝑋|𝜎(,𝑌0)]. Since 𝑌0 is not defined, let’s assume 𝑌0 and 𝑍0 are given. Let’s consider the sigma-algebra generated by all the 𝑌𝑖 and 𝑍𝑖. Let  = 𝜎(𝑌0,𝑍0,𝑌1,𝑍1,…). Both 𝑌𝑘 and 𝑍𝑘 are measurable with respect to . Thus, 𝑌∞ and 𝑍∞ are measurable with respect to . Consider 𝔼[𝑌𝑛+1|,𝑍0,…,𝑍𝑛−1] = 𝔼[𝔼[𝑋|𝜎(,𝑍0,…,𝑍𝑛)]|,𝑍0,…,𝑍𝑛−1] = 𝑌𝑛. Consider 𝔼[𝑍𝑛+1|,𝑌0,…,𝑌𝑛−1] = 𝔼[𝔼[𝑋|𝜎(,𝑌0,…,𝑌𝑛)]|,𝑌0,…,𝑌𝑛−1] = 𝑍𝑛. Let  = 𝜎( ∪  ∪ 𝜎(𝑌0,𝑍0,𝑌1,𝑍1,…)). Consider 𝔼[𝑋 ,𝑍0,…,𝑍𝑘−1] = 𝑌𝑘. Consider 𝔼[𝑋|,𝑌0,…,𝑌𝑘−1] = 𝑍𝑘. Let’s consider the limit sigma-algebras ∞ = 𝜎(,𝑍0,𝑍1,…) and ∞ = 𝜎(,𝑌0,𝑌1,…).

We know that 𝑌∞ is ∞-measurable and 𝑍∞ is ∞-measurable. Let  = 𝜎( ∪ ).

- 𝑌1 = 𝔼[𝑋|]. 𝑍1 = 𝔼[𝑋|].
- 𝑌2 = 𝔼[𝑋|𝜎(,𝑍0)] = 𝔼[𝑋|𝜎(,𝔼[𝑋|])]. 𝑍2 = 𝔼[𝑋|𝜎(,𝑌0)] = 𝔼[𝑋|𝜎( ,𝔼[𝑋|])]. Consider the case when  and  are independent. Then 𝑌1 = 𝔼[𝑋|] and 𝑍1 = 𝔼[𝑋|]. 𝑌2 = 𝔼[𝑋|𝜎(,𝔼[𝑋|])]. Let’s consider the sigma-algebra generated by all the information. Let ∞ = 𝜎(,,𝑌0,𝑍0,𝑌1,𝑍1,…). Then both 𝑌∞ and 𝑍∞ are ∞-measurable. We have 𝑌∞ = lim𝑛→∞ 𝑌𝑛 and 𝑍∞ = lim𝑛→∞ 𝑍𝑛 a.s. and in 𝐿1. Consider 𝔼[𝑋|,𝑍0,𝑍1,…] = 𝑌∞ and 𝔼[𝑋|,𝑌0,𝑌1,…] = 𝑍∞. Let ∞ = 𝜎( ∪ ). Consider 𝔼[𝑋|∞]. Let’s try to show that both 𝑌∞ and 𝑍∞ are equal to 𝔼[𝑋|𝜎( ∨ )]. This seems too simple. Let’s consider the definition of conditional expectation. For any set 𝐴 ∈ ∞, 𝔼[𝑌∞1𝐴] = 𝔼[𝑋1𝐴]. For any set 𝐵 ∈ ∞, 𝔼[𝑍∞1𝐵] = 𝔼[𝑋1𝐵]. Consider 𝔼[𝑌𝑘+1|𝑘] = 𝑌𝑘 where 𝑘 = 𝜎(,𝑍0,…,𝑍𝑘−1).

Consider 𝔼[𝑍𝑘+1|𝑘] = 𝑍𝑘 where 𝑘 = 𝜎(,𝑌0,…,𝑌𝑘−1). Consider the sigma-algebra 𝜎( ∪ ). Let 𝑊 = 𝔼[𝑋|𝜎( ∪ )]. We have 𝑌∞ = 𝔼[𝑋|𝜎(,𝑍0,𝑍1,…)]. Since 𝑍𝑘 depends on 𝑌𝑘−1 which depends on 𝑍𝑘−2 etc., it is complex. Let’s consider the case where 𝑌0 = 𝔼[𝑋|] and 𝑍0 = 𝔼[𝑋|]. Then 𝑌1 = 𝔼[𝑋|] = 𝑌0. 𝑍1 = 𝔼[𝑋|] = 𝑍0. If this were the case, then 𝑌𝑘 = 𝑌0 and 𝑍𝑘 = 𝑍0, so 𝑌∞ = 𝑌0 and 𝑍∞ = 𝑍0. We need 𝑌0 = 𝑍0 a.s., i.e., 𝔼[𝑋|] = 𝔼[𝑋|] a.s. This is not generally true. Consider the sigma-algebra generated by all 𝑌𝑖 and 𝑍𝑖, denoted by ∞ = 𝜎(𝑌0,𝑍0,𝑌1,𝑍1,…). Then ∞ = 𝜎(,𝑍0,𝑍1,…) ⊆ 𝜎(,∞). ∞ = 𝜎(,𝑌0,𝑌1,…) ⊆ 𝜎(,∞).

Consider 𝔼[𝑋 ,𝑍0,𝑍1,…] = 𝑌∞. This means for any 𝐴 ∈ 𝜎(,𝑍0,𝑍1,…), 𝔼[𝑋1𝐴] = 𝔼[𝑌∞1𝐴]. Consider 𝔼[𝑋|,𝑌0,𝑌1,…] = 𝑍∞. This means for any 𝐵 ∈ 𝜎(,𝑌0,𝑌1,…), 𝔼[𝑋1𝐵] = 𝔼[𝑍∞1𝐵]. Let  = 𝜎(,). Consider 𝔼[𝑋|]. Consider 𝔼[𝑌∞|,𝑌0,𝑌1,…] = 𝔼[𝔼[𝑋|,𝑍0,𝑍1,…]|,𝑌0,𝑌1,…]. Let’s use the fact that 𝑌∞ is 𝜎(,𝑍0,𝑍1,…)-measurable and 𝑍∞ is 𝜎(,𝑌0,𝑌1,…)-measurable. Consider the smallest sigma-algebra containing both 𝜎(,𝑍0,𝑍1,…) and 𝜎(,𝑌0,𝑌1,…). This is 𝜎( ∪  ∪ {𝑌𝑖}𝑖≥0 ∪ {𝑍𝑖}𝑖≥0). Let’s try to show that 𝔼[(𝑌∞ − 𝑍∞)2] = 0. 𝔼[(𝑌∞ − 𝑍∞)2] = 𝔼[𝑌∞2] + 𝔼[𝑍∞2 ] − 2𝔼[𝑌∞𝑍∞]. Consider the definition of 𝑌𝑘 and 𝑍𝑘. 𝔼[𝑌𝑘+1|,𝑍0,…,𝑍𝑘−1] = 𝑌𝑘. 𝔼[𝑍𝑘+1|,𝑌0,…,𝑌𝑘−1] = 𝑍𝑘. Consider 𝔼[𝑌∞|,𝑌0,𝑌1,…]. Since 𝑌∞ is measurable with respect to 𝜎(,𝑍0,𝑍1,…), this does not simplify easily. Let’s assume 𝑌∞ ≠ 𝑍∞ on a set of positive measure. Consider the case where  and  are trivial. Then 𝑌𝑘 = 𝔼[𝑋|𝜎(𝑍0,…,𝑍𝑘−1)] and 𝑍𝑘 = 𝔼[𝑋|𝜎(𝑌0,…,𝑌𝑘−1)]. Let ∞ = 𝜎( ∪  ∪ {𝑌𝑖,𝑍𝑖}𝑖≥0). Consider 𝔼[𝑋|∞]. We have 𝑌∞ = 𝔼[𝑋|𝜎(,𝑍0,𝑍1,…)]. By the properties of conditional expectation, 𝔼[𝑌∞|] = 𝔼[𝔼[𝑋|,𝑍0,𝑍1,…]|]. Consider the sigma-algebra  generated by all the random variables involved.  = 𝜎( ∪  ∪ {𝑌𝑖}𝑖≥0 ∪ {𝑍𝑖}𝑖≥0). Let 𝑀∞ = 𝔼[𝑋| ∨ ]. Consider the limits. As 𝑛 → ∞, 𝜎(,𝑍0,…,𝑍𝑛−1) ↑ 𝜎(,𝑍0,𝑍1,…). As 𝑛 → ∞, 𝜎(,𝑌0,…,𝑌𝑛−1) ↑ 𝜎(,𝑌0,𝑌1,…). Let ∞ = 𝜎(,𝑍0,𝑍1,…) and ∞ = 𝜎(,𝑌0,𝑌1,…). We have 𝑌∞ = 𝔼[𝑋|∞] and 𝑍∞ = 𝔼[𝑋|∞]. Consider 𝔼[𝑌∞ − 𝑍∞]2 = 𝔼[(𝑌∞ − 𝑋) + (𝑋 − 𝑍∞)]2

= 𝔼[(𝑌∞ − 𝑋)2] + 𝔼[(𝑋 − 𝑍∞)2] + 2𝔼[(𝑌∞ − 𝑋)(𝑋 − 𝑍∞)]

= 𝔼[(𝑋 − 𝔼[𝑋|∞])2] + 𝔼[(𝑋 − 𝔼[𝑋|∞])2] + 2𝔼[𝑌∞𝑋 − 𝑌∞𝑍∞ − 𝑋2 + 𝑋𝑍∞]

= 𝔼[𝑋2] − 𝔼[𝑌∞2] + 𝔼[𝑋2] − 𝔼[𝑍∞2 ] + 2𝔼[𝑌∞𝑋] − 2𝔼[𝑌∞𝑍∞] − 2𝔼[𝑋2] + 2𝔼[𝑋𝑍∞]

= −𝔼[𝑌∞2] − 𝔼[𝑍∞2 ] + 2𝔼[𝑌∞2] − 2𝔼[𝑌∞𝑍∞] + 2𝔼[𝑍∞2 ]

= 𝔼[𝑌∞2] + 𝔼[𝑍∞2 ] − 2𝔼[𝑌∞𝑍∞] Consider the sigma-algebra  = 𝜎( ∪ ). Let 𝑊 = 𝔼[𝑋|]. Consider the limit sigma-algebra generated by all the information. Let ∞ = 𝜎(,,𝑌0,𝑍0,𝑌1,𝑍1,…). Then ∞ ⊆ ∞ and ∞ ⊆ ∞. We have 𝔼[𝑋|∞] = 𝑌∞ and 𝔼[𝑋|∞] = 𝑍∞. Consider 𝔼[𝑌∞|∞] and 𝔼[𝑍∞|∞]. Let  = 𝜎( ∪ ). Consider 𝔼[𝑋|]. Consider the sequence of pairs (𝑌𝑘,𝑍𝑘). If the limits are the same, then for any bounded measurable function 𝑓, 𝔼[𝑓(𝑌∞)] = 𝔼[𝑓(𝑍∞)]. Consider the sigma-algebra  = 𝜎( ∪  ∪ 𝜎(𝑌0,𝑍0,…)). Then both 𝑌∞ and 𝑍∞ are measurable with respect to . Consider 𝔼[(𝑌𝑛 − 𝑍𝑛)2]. Let  = 𝜎( ∪ ). Consider 𝑊 = 𝔼[𝑋|]. Then 𝔼[𝑊 |] = 𝔼[𝑋|] and 𝔼[𝑊 |] = 𝔼[𝑋|].

Consider the limit sigma-algebra  = 𝜎( ∪  ∪ 𝜎(𝑌0,𝑍0,…)). We have 𝑌∞ = 𝔼[𝑋|𝜎(,𝑍0,𝑍1,…)] and 𝑍∞ = 𝔼[𝑋|𝜎(,𝑌0,𝑌1,…)]. Let  = 𝜎(,𝑍0,𝑍1,…) ∨ 𝜎(,𝑌0,𝑌1,…). Consider 𝔼[𝑋|]. Consider the random variable 𝑋. Then 𝑌∞ = lim𝑛 𝔼[𝑋|𝑛] and 𝑍∞ = lim𝑛 𝔼[𝑋|𝑛]. Let ∞ = 𝜎( ∪  ∪ 𝜎(𝑌0,𝑍0,…)). Consider 𝐸 = {𝜔 ∶ 𝑌∞(𝜔) ≠ 𝑍∞(𝜔)}. Suppose 𝑃(𝐸) > 0. Consider the conditional expectation with respect to the join of the sigma-algebras. Let ∞ = 𝜎(,,𝑌0,𝑍0,…). Consider 𝔼[𝑋|∞]. Then 𝔼[𝔼[𝑋|∞]|,𝑍0,…] = 𝔼[𝑋|,𝑍0,…] = 𝑌𝑛+1. And 𝔼[𝔼[𝑋|∞]|,𝑌0,…] = 𝔼[𝑋|,𝑌0,…] = 𝑍𝑛+1. So 𝑌∞ = 𝔼[𝑋|𝜎(,𝑍0,𝑍1,…)] and 𝑍∞ = 𝔼[𝑋|𝜎(,𝑌0,𝑌1,…)]. Let ∞ = 𝜎(,𝑍0,𝑍1,…) and ∞ = 𝜎(,𝑌0,𝑌1,…). Consider 𝔼[𝑋|∞ ∨ ∞]. Then 𝔼[𝔼[𝑋|∞ ∨ ∞]|∞] = 𝔼[𝑋|∞] = 𝑌∞. And 𝔼[𝔼[𝑋|∞ ∨ ∞]|∞] = 𝔼[𝑋|∞] = 𝑍∞. This means 𝑌∞ = 𝔼[𝑋|∞ ∨ ∞] a.s. and 𝑍∞ = 𝔼[𝑋|∞ ∨ ∞] a.s. Therefore, 𝑌∞ = 𝑍∞ a.s. Solution:

- (a) For the sequence 𝑌𝑘, the 𝜎-algebras 𝑘 = 𝜎(,𝑍0,…,𝑍𝑘−1) are increasing as 𝑘 increases. Since 𝑋 is integrable, by the Martingale Convergence Theorem for conditional expectations, the sequence 𝑌𝑘 = 𝔼[𝑋|𝑘] converges almost surely and in 𝐿1 to 𝑌∞ = 𝔼[𝑋|𝜎(,𝑍0,𝑍1,…)]. Similarly, for the sequence 𝑍𝑘, the 𝜎-algebras 𝑘 = 𝜎(,𝑌0,…,𝑌𝑘−1) are increasing. Since 𝑋 is integrable, by the Martingale Convergence Theorem, the sequence 𝑍𝑘 = 𝔼[𝑋|𝑘] converges almost surely and in 𝐿1 to 𝑍∞ = 𝔼[𝑋|𝜎(,𝑌0,𝑌1,…)].
- (b) Let ∞ = 𝜎(,𝑍0,𝑍1,…) and ∞ = 𝜎(,𝑌0,𝑌1,…). We have 𝑌∞ = 𝔼[𝑋|∞] and 𝑍∞ = 𝔼[𝑋|∞]. Consider the 𝜎-algebra  = ∞ ∨ ∞ = 𝜎(,,𝑌0,𝑍0,𝑌1,𝑍1,…). Consider 𝑊 = 𝔼[𝑋|]. We have ∞ ⊆ , so 𝔼[𝑊 |∞] = 𝔼[𝔼[𝑋|]|∞] = 𝔼[𝑋|∞] = 𝑌∞. Since 𝑊 is -measurable, 𝔼[𝑊 |∞] = 𝑌∞ implies that 𝑌∞ is the conditional expectation of 𝑊 given ∞. But since 𝑌∞ is already ∞-measurable, we have 𝔼[𝑊 |∞] = 𝑊 if 𝑊 is ∞-measurable, which is not necessarily true. However, we know that 𝔼[𝑌∞|∞] = 𝑌∞ and 𝔼[𝑍∞|∞] = 𝑍∞. Consider 𝔼[𝑋|∞ ∨ ∞]. Then 𝔼[𝔼[𝑋|∞ ∨ ∞]|∞] = 𝔼[𝑋|∞] = 𝑌∞. This implies that 𝔼[𝑋|∞ ∨ ∞] = 𝑌∞ almost surely. Similarly, 𝔼[𝔼[𝑋|∞ ∨ ∞]|∞] = 𝔼[𝑋|∞] = 𝑍∞. This implies that 𝔼[𝑋|∞ ∨ ∞] = 𝑍∞ almost surely. Therefore, 𝑌∞ = 𝑍∞ almost surely. Final Answer: The final answer is

|𝑌∞ = 𝑍∞|
|---|

- Table 10. s1K sample stemming from s1-teasers. The black text is the question and the yellow text is the solution. Based solely on the question we generate the light blue reasoning trace and blue response with Gemini, which we use for training.

Alice and Bob each have a coin and flip it until they get a heads. If Bob flipped his coin more times than Alice, what is the expected number of times Alice flipped her coin?

4/3

The problem asks for the expected number of times Alice flipped her coin, given that Bob flipped his coin more times than Alice. Let 𝑋 be the number of flips Alice makes until she gets a head. Let 𝑌 be the number of flips Bob makes until he gets a head. 𝑋 and 𝑌 are independent random variables following a geometric distribution with parameter 𝑝 = 0.5 (probability of getting a head). The probability mass function for a geometric distribution is 𝑃(𝐾 = 𝑘) = (1 − 𝑝)𝑘−1𝑝, for 𝑘 = 1,2,….

In this case, 𝑝 = 0.5, so 𝑃(𝐾 = 𝑘) = (0.5)𝑘−1(0.5) = (0.5)𝑘 = 21𝑘. We are given the condition that Bob flipped his coin more times than Alice, which means 𝑌 > 𝑋.

We want to find the expected value of 𝑋 given that 𝑌 > 𝑋, i.e., 𝐸[𝑋|𝑌 > 𝑋]. The formula for conditional expectation is 𝐸[𝑋|𝐴] =

∑

𝑥 𝑥𝑃(𝑋 = 𝑥|𝐴).

Also, 𝑃(𝑋 = 𝑥|𝐴) = 𝑃(𝑋𝑃=(𝐴𝑥)∩𝐴). In our case, 𝐴 is the event 𝑌 > 𝑋.

We need to find 𝑃(𝑌 > 𝑋). 𝑃(𝑌 > 𝑋) =

∑∞𝑥=1

∑∞𝑦=𝑥+1 𝑃(𝑋 = 𝑥,𝑌 = 𝑦)

Since 𝑋 and 𝑌 are independent, 𝑃(𝑋 = 𝑥,𝑌 = 𝑦) = 𝑃(𝑋 = 𝑥)𝑃(𝑌 = 𝑦) = 21𝑥 21𝑦 = 2𝑥1+𝑦. 𝑃(𝑌 > 𝑋) =

- 1

- 2𝑥 ∑∞𝑦=𝑥+1

∑∞𝑥=1 The inner sum is a geometric series:

∑∞𝑥=1

∑∞𝑦=𝑥+1

- 1

- 2𝑦

- 1

- 2𝑥+𝑦 =

∑∞𝑦=𝑥+1(12)𝑦 = (12)𝑥+1 + (12)𝑥+2 + … This is a geometric series with first term 𝑎 = (12)𝑥+1 and common ratio 𝑟 = 12. The sum is 1−𝑎𝑟 = (

- 1

- 2)𝑥+1

- 1

- 2)𝑥+1

= (

= (12)𝑥. So, 𝑃(𝑌 > 𝑋) =

1−21

- 1

- 2

∑∞𝑥=1

∑∞𝑥=1(14)𝑥. This is another geometric series with first term 𝑎 = 14 and common ratio 𝑟 = 14. The sum is 1−𝑎𝑟 =

1 2𝑥(12)𝑥 =

1 4

1 4

= 13.

=

1−14

- 3

- 4

Now, we need to find 𝑃(𝑋 = 𝑥 ∩ 𝑌 > 𝑋) = 𝑃(𝑋 = 𝑥,𝑌 > 𝑥). 𝑃(𝑋 = 𝑥,𝑌 > 𝑥) = 𝑃(𝑋 = 𝑥)𝑃(𝑌 > 𝑥), due to independence. 𝑃(𝑌 > 𝑥) =

∑∞𝑦=𝑥+1 So, 𝑃(𝑋 = 𝑥,𝑌 > 𝑥) = 21𝑥 × (12)𝑥 = (14)𝑥. Now, we can find 𝑃(𝑋 = 𝑥|𝑌 > 𝑋) = 𝑃(𝑋𝑃=(𝑌 >𝑋𝑥∩𝑌 >𝑋) ) = (

∑∞𝑦=𝑥+1 𝑃(𝑌 = 𝑦) =

- 1

- 2𝑦 = (12)𝑥+1 + (12)𝑥+2 + ⋯ = (12)𝑥.

1 4)𝑥

= 3(41)𝑥.

1 3

Finally, we can calculate the conditional expectation: 𝐸[𝑋|𝑌 > 𝑋] =

∑∞𝑥=1 𝑥𝑃(𝑋 = 𝑥|𝑌 > 𝑋) =

∑∞𝑥=1 𝑥 ⋅ 3(14)𝑥 = 3

∑∞𝑥=1 𝑥(14)𝑥. Consider the series 𝑆 =

∑∞𝑥=1 𝑥𝑟𝑥, where |𝑟| < 1. 𝑆 = 𝑟 + 2𝑟2 + 3𝑟3 + 4𝑟4 + … 𝑆 𝑟 = 1 + 2𝑟 + 3𝑟2 + 4𝑟3 + …

Consider the geometric series 𝐺 = 1 + 𝑟 + 𝑟2 + 𝑟3 + ⋯ = 1−1𝑟. Differentiating with respect to 𝑟:

𝑑𝐺 𝑑𝑟 = 1 + 2𝑟 + 3𝑟2 + 4𝑟3 + ⋯ = 𝑑𝑟𝑑 (1−1𝑟) = (1−1𝑟)2 .

So, 𝑆𝑟 = (1−1𝑟)2, which means 𝑆 = (1−𝑟𝑟)2. In our case, 𝑟 = 14.

1 4

1 4

1 4 9 16

∑∞𝑥=1 𝑥(14)𝑥 =

= 14 × 169 = 94.

=

=

(1−14)2

(34)2

Therefore, 𝐸[𝑋|𝑌 > 𝑋] = 3 × 49 = 129 = 43. Alternative approach: Symmetry.

Consider the events 𝑌 > 𝑋, 𝑋 > 𝑌 , and 𝑋 = 𝑌 . 𝑃(𝑌 > 𝑋) = 𝑃(𝑋 > 𝑌 ) due to symmetry. 𝑃(𝑋 = 𝑌 ) =

1 4

1 4

∑∞𝑘=1(41)𝑘 =

∑∞𝑘=1 𝑃(𝑋 = 𝑘,𝑌 = 𝑘) =

∑∞𝑘=1 𝑃(𝑋 = 𝑘)𝑃(𝑌 = 𝑘) =

∑∞𝑘=1(21𝑘 )2 =

= 13.

=

1−14

- 3

- 4

We have 𝑃(𝑌 > 𝑋) + 𝑃(𝑋 > 𝑌 ) + 𝑃(𝑋 = 𝑌 ) = 1. 𝑃(𝑌 > 𝑋) + 𝑃(𝑌 > 𝑋) + 13 = 1.

- 2𝑃(𝑌 > 𝑋) = 1 − 13 = 32.

𝑃(𝑌 > 𝑋) = 31. This confirms our previous calculation. Let 𝐸[𝑋|𝑌 > 𝑋] = 𝜇. We are looking for the expected number of flips Alice made, given Bob flipped more times. Consider the case where the first flip for both is tails. Then the process restarts, and the condition remains. Let 𝐴1 be Alice’s first flip, 𝐵1 be Bob’s first flip.

𝑃(𝐴1 = 𝐻) = 1∕2,𝑃(𝐴1 = 𝑇) = 1∕2. 𝑃(𝐵1 = 𝐻) = 1∕2,𝑃(𝐵1 = 𝑇) = 1∕2.

- Case 1: 𝐴1 = 𝐻,𝐵1 = 𝐻. 𝑋 = 1,𝑌 = 1, so 𝑌 > 𝑋 is false.
- Case 2: 𝐴1 = 𝐻,𝐵1 = 𝑇. Alice stops at 1, Bob continues. 𝑌 > 1.
- Case 3: 𝐴1 = 𝑇,𝐵1 = 𝐻. Bob stops at 1, Alice continues.
- Case 4: 𝐴1 = 𝑇,𝐵1 = 𝑇. Both continue. Consider the conditional probabilities given 𝑌 > 𝑋. Let 𝐸 = 𝐸[𝑋|𝑌 > 𝑋]. If the first flips are HH, the condition 𝑌 > 𝑋 is not met. If the first flips are HT, Alice’s number of flips is 1. If the first flips are TH, Bob’s number of flips is 1, Alice’s is at least 2, so 𝑌 < 𝑋. If the first flips are TT, the problem restarts, and the expected number of additional flips for Alice is still 𝐸.

Let 𝑁𝐴 be the number of flips Alice makes, 𝑁𝐵 be the number of flips Bob makes. We are given 𝑁𝐵 > 𝑁𝐴. Consider the first flips: HH (prob 1/4): Condition not met. HT (prob 1/4): Alice flips 1 time. TH (prob 1/4): Bob flips 1 time, Alice > 1, so 𝑁𝐴 > 𝑁𝐵, condition not met. TT (prob 1/4): Both continue. Let 𝐸 be the expected number of Alice’s flips given Bob flipped more. If the first flips are HT, Alice flipped 1 time, and the condition is met. If the first flips are TT, the situation is the same as the beginning, so the expected number of additional flips for Alice is 𝐸. Consider the event 𝑌 > 𝑋. Conditioning on the first flips: 𝐸[𝑋|𝑌 > 𝑋] = 𝐸[𝑋|𝑌 > 𝑋,𝐻𝐻]𝑃(𝐻𝐻|𝑌 > 𝑋)+𝐸[𝑋|𝑌 > 𝑋,𝐻𝑇]𝑃(𝐻𝑇|𝑌 > 𝑋)+𝐸[𝑋|𝑌 > 𝑋,𝑇𝐻]𝑃(𝑇𝐻|𝑌 > 𝑋) + 𝐸[𝑋|𝑌 > 𝑋,𝑇𝑇]𝑃(𝑇𝑇|𝑌 > 𝑋) 𝑃(𝑌 > 𝑋 𝐻𝐻) = 0. 𝑃(𝑌 > 𝑋 𝐻𝑇) = 1. 𝑃(𝑌 > 𝑋 𝑇𝐻) = 0. 𝑃(𝑌 > 𝑋|𝑇𝑇) depends on the future flips. Let’s use the definition of conditional expectation directly. 𝐸[𝑋|𝑌 > 𝑋] =

∑∞𝑥=1 𝑥𝑃(𝑃𝑋(=𝑌 >𝑋𝑥,𝑌 >𝑥) ) Consider the probabilities of the first flips given 𝑌 > 𝑋. 𝑃(𝐻𝐻|𝑌 > 𝑋) = 𝑃(𝑌 >𝑋𝑃|(𝐻𝐻𝑌 >𝑋)𝑃)(𝐻𝐻) = 01∕3⋅1∕4 = 0. 𝑃(𝐻𝑇|𝑌 > 𝑋) = 𝑃(𝑌 >𝑋𝑃(|𝑌 >𝑋𝐻𝑇)𝑃)(𝐻𝑇) = 11∕3⋅1∕4 = 34.

∑∞𝑥=1 𝑥𝑃(𝑋 = 𝑥|𝑌 > 𝑋) =

𝑃(𝑇𝐻|𝑌 > 𝑋) = 𝑃(𝑌 >𝑋𝑃(|𝑌 >𝑋𝑇𝐻)𝑃)(𝑇𝐻) = 01∕3⋅1∕4 = 0. 𝑃(𝑇𝑇|𝑌 > 𝑋) = 𝑃(𝑌 >𝑋𝑃(|𝑌 >𝑋𝑇𝑇)𝑃)(𝑇𝑇) If the first flips are TT, the future behavior is the same, and the condition 𝑌 > 𝑋 holds if Bob takes more flips than Alice in the subsequent trials. 𝑃(𝑌 > 𝑋|𝑇𝑇) = 𝑃(Bob needs more flips than Alice starting from scratch) = 𝑃(𝑌 > 𝑋) = 1∕3. 𝑃(𝑇𝑇|𝑌 > 𝑋) = (1∕3)1∕3⋅(1∕4) = 14. Sum of conditional probabilities: 0 + 3∕4 + 0 + 1∕4 = 1. Now consider the conditional expectations of 𝑋: 𝐸[𝑋|𝑌 > 𝑋,𝐻𝐻]: Not relevant since 𝑃(𝐻𝐻|𝑌 > 𝑋) = 0. 𝐸[𝑋 𝑌 > 𝑋,𝐻𝑇]: Given HT and 𝑌 > 𝑋, Alice flipped 1 time. 𝐸 = 1. 𝐸[𝑋 𝑌 > 𝑋,𝑇𝐻]: Not relevant. 𝐸[𝑋|𝑌 > 𝑋,𝑇𝑇]: Given TT and 𝑌 > 𝑋, Alice has already flipped once, and the expected number of additional flips is 𝐸. So, the total number of flips is 1 + 𝐸. 𝐸 = 0 ⋅ 0 + 1 ⋅ (3∕4) + undefined ⋅ 0 + (1 + 𝐸) ⋅ (1∕4)

- 𝐸 = 0 + 3∕4 + 0 + 1∕4 + 𝐸∕4
- 𝐸 = 1 + 𝐸∕4 𝐸 − 𝐸∕4 = 1

- 3𝐸∕4 = 1 𝐸 = 4∕3. This confirms the previous result. Let’s think about the intuition. If we have the condition 𝑌 > 𝑋, it means Alice was "lucky" and got a head early, or Bob was "unlucky". Consider the random variable 𝑍 = 𝑌 − 𝑋. We are given 𝑍 > 0. We want 𝐸[𝑋|𝑍 > 0]. Let 𝑝 = 𝑃(head) = 0.5. The number of tails before the first head follows a geometric distribution with parameter 𝑝. 𝑋 − 1 ∼ 𝐺𝑒𝑜𝑚(𝑝), 𝑌 − 1 ∼ 𝐺𝑒𝑜𝑚(𝑝). 𝑃(𝑋 − 1 = 𝑘) = (1 − 𝑝)𝑘𝑝. 𝑃(𝑋 = 𝑥) = (1 − 𝑝)𝑥−1𝑝. Consider the joint distribution of 𝑋 and 𝑌 . 𝑃(𝑋 = 𝑥,𝑌 = 𝑦) = 𝑝2(1 − 𝑝)𝑥−1(1 − 𝑝)𝑦−1 Consider the event 𝑌 = 𝑋 + 𝑘, where 𝑘 ≥ 1.

∑∞𝑥=1 𝑝2(1 − 𝑝)𝑥−1(1 − 𝑝)𝑥+𝑘−1

∑∞𝑥=1 𝑃(𝑋 = 𝑥,𝑌 = 𝑥 + 𝑘) =

𝑃(𝑌 = 𝑋 + 𝑘) =

∑∞𝑥=1((1 − 𝑝)2)𝑥−1 = 𝑝2(1 − 𝑝)𝑘−1 1−(1−1 𝑝)2

= 𝑝2(1 − 𝑝)𝑘−1

= 𝑝2(1 − 𝑝)𝑘−1 1−(1−21𝑝+𝑝2) = 𝑝2(1 − 𝑝)𝑘−1 2𝑝−1𝑝2 = 𝑝2(1 − 𝑝)𝑘−1 𝑝(2−1 𝑝)

= 𝑝(1−2−𝑝𝑝)𝑘−1. If 𝑝 = 0.5, then 1 − 𝑝 = 0.5, 2 − 𝑝 = 1.5.

𝑃(𝑌 = 𝑋 + 𝑘) = 0.5(01..5)5 𝑘−1 = (01.5).5𝑘 = (1∕2)3∕2𝑘 = 21𝑘 ⋅ 23 = 3⋅21𝑘−1. Check if

∑∞𝑘=1 𝑃(𝑌 = 𝑋 + 𝑘) = 𝑃(𝑌 > 𝑋) = 1∕3. ∑∞𝑘=1

∑∞𝑘=1(12)𝑘−1 = 31(1 + 1∕2 + 1∕4 + …) = 13 1−1∕21 = 13 ⋅ 2 = 23. Something is wrong here. Let’s recalculate 𝑃(𝑌 > 𝑋). 𝑃(𝑌 > 𝑋) =

3⋅2𝑘−1 = 13

1

∑∞𝑥=1

∑∞𝑦=𝑥+1(1 − 𝑝)𝑥−1𝑝(1 − 𝑝)𝑦−1𝑝

∑∞𝑦=𝑥+1(1 − 𝑝)𝑦−1 Inner sum: (1 − 𝑝)𝑥 + (1 − 𝑝)𝑥+1 + ⋯ = 1−(1−(1−𝑝)𝑥𝑝) = (1−𝑝𝑝)𝑥. 𝑃(𝑌 > 𝑋) = 𝑝2 ∑∞𝑥=1(1 − 𝑝)𝑥−1 (1−𝑝𝑝)𝑥 = 𝑝(1 − 𝑝)

= 𝑝2 ∑∞𝑥=1(1 − 𝑝)𝑥−1

∑∞𝑥=1(1 − 𝑝)2(𝑥−1)

∑∞𝑥=1((1 − 𝑝)2)𝑥−1 = 𝑝(1 − 𝑝)1−(1−1 𝑝)2

= 𝑝(1 − 𝑝)

= 𝑝(1 − 𝑝)1−(1−21𝑝+𝑝2) = 𝑝(1 − 𝑝)2𝑝−1𝑝2 = 𝑝(1 − 𝑝)𝑝(2−1 𝑝) = 2−1−𝑝𝑝. If 𝑝 = 0.5, 𝑃(𝑌 > 𝑋) = 01..55 = 13. Correct.

Now, let’s find 𝐸[𝑋|𝑌 > 𝑋]. 𝐸[𝑋|𝑌 > 𝑋] =

∑∞𝑥=1 𝑥𝑃(𝑃𝑋(=𝑌 >𝑋𝑥,𝑌 >𝑥) ). 𝑃(𝑋 = 𝑥,𝑌 > 𝑥) = 𝑃(𝑋 = 𝑥)𝑃(𝑌 > 𝑥) (independence) 𝑃(𝑋 = 𝑥) = (1 − 𝑝)𝑥−1𝑝. 𝑃(𝑌 > 𝑥) =

∑∞𝑥=1 𝑥𝑃(𝑋 = 𝑥|𝑌 > 𝑋) =

∑∞𝑦=𝑥+1(1 − 𝑝)𝑦−1𝑝 = 𝑝[(1 − 𝑝)𝑥 + (1 − 𝑝)𝑥+1 + …] = 𝑝1−(1−(1−𝑝)𝑥𝑝) = (1 − 𝑝)𝑥. 𝑃(𝑋 = 𝑥,𝑌 > 𝑥) = (1 − 𝑝)𝑥−1𝑝(1 − 𝑝)𝑥 = 𝑝(1 − 𝑝)2𝑥−1. 𝐸[𝑋|𝑌 > 𝑋] = 𝑃(𝑌 >𝑋1 )

∑∞𝑥=1 𝑥𝑝(1 − 𝑝)2𝑥−1

∑∞𝑥=1 𝑥((1 − 𝑝)2)𝑥 (error in power) 𝑃(𝑌 > 𝑥) = 1−(1−(1−𝑝)𝑥𝑝)𝑝 = (1 − 𝑝)𝑥. 𝐸[𝑋|𝑌 > 𝑋] = 𝑃(𝑌 >𝑋1 )

= 1−2−𝑝𝑝𝑝(1 − 𝑝)−1

∑∞𝑥=1 𝑥𝑃(𝑋 = 𝑥)𝑃(𝑌 > 𝑥)

= 𝑃(𝑌 >𝑋1 ) ∞𝑥=1 𝑥(1 − 𝑝)𝑥−1𝑝(1 − 𝑝)𝑥

∑∞𝑥=1 𝑥(1 − 𝑝)2𝑥−1

= 𝑃(𝑌 >𝑋𝑝 )

∑∞𝑥=1 𝑥((1 − 𝑝)2)𝑥

= (1−𝑝)∕(2−𝑝 𝑝)(1 − 𝑝)−1

= 𝑝(1−2−𝑝𝑝)2 ∑∞𝑥=1 𝑥((1 − 𝑝)2)𝑥. Let 𝑟 = (1 − 𝑝)2.

∑∞𝑥=1 𝑥𝑟𝑥 = (1−𝑟𝑟)2. 1 − 𝑟 = 1 − (1 − 𝑝)2 = 1 − (1 − 2𝑝 + 𝑝2) = 2𝑝 − 𝑝2 = 𝑝(2 − 𝑝). (1 − 𝑟)2 = 𝑝2(2 − 𝑝)2.

∑∞𝑥=1 𝑥((1 − 𝑝)2)𝑥 = 𝑝(1−2(2−𝑝𝑝)2)2. 𝐸[𝑋|𝑌 > 𝑋] = 𝑝(1−2−𝑝𝑝)2 𝑝(1−2(2−𝑝𝑝)2)2 = 𝑝(2−1 𝑝). If 𝑝 = 0.5, 𝐸[𝑋|𝑌 > 𝑋] = 0.5×11 .5 = 0.175 = 3∕41 = 4∕3.

Solution: Let 𝑋 be the number of flips Alice makes until she gets a head, and 𝑌 be the number of flips Bob makes until he gets a head. 𝑋 and 𝑌 are independent geometric random variables with parameter 𝑝 = 0.5. The probability mass function is 𝑃(𝐾 = 𝑘) = (1 − 𝑝)𝑘−1𝑝. We are interested in the expected number of times Alice flipped her coin, given that Bob flipped his coin more times than Alice, i.e., 𝐸[𝑋|𝑌 > 𝑋]. Using the definition of conditional expectation, 𝐸[𝑋|𝑌 > 𝑋] =

∑∞𝑥=1 𝑥𝑃(𝑋 = 𝑥|𝑌 > 𝑋).

𝑃(𝑋 = 𝑥|𝑌 > 𝑋) = 𝑃(𝑃𝑋(=𝑌 >𝑋𝑥,𝑌 >𝑥) ). First, calculate 𝑃(𝑌 > 𝑋). 𝑃(𝑌 > 𝑋) =

∑∞𝑥=1

∑∞𝑦=𝑥+1 𝑃(𝑋 = 𝑥,𝑌 = 𝑦) =

∑∞𝑥=1 𝑃(𝑋 = 𝑥)𝑃(𝑌 > 𝑥).

𝑃(𝑋 = 𝑥) = (0.5)𝑥−1(0.5) = (0.5)𝑥 = 21𝑥. 𝑃(𝑌 > 𝑥) =

∑∞𝑦=𝑥+1(0.5)𝑦−1(0.5) = (0.5)𝑥. 𝑃(𝑌 > 𝑋) =

∑∞𝑥=1(41)𝑥 = 1−1∕41∕4 = 1∕43∕4 = 13. Next, calculate 𝑃(𝑋 = 𝑥,𝑌 > 𝑥) = 𝑃(𝑋 = 𝑥)𝑃(𝑌 > 𝑥) = 21𝑥 (12)𝑥 = (14)𝑥. Now, find 𝑃(𝑋 = 𝑥|𝑌 > 𝑋) = 𝑃(𝑃𝑋(=𝑌 >𝑋𝑥,𝑌 >𝑥) ) = (1∕4)1∕3𝑥 = 3(14)𝑥. Finally, calculate the conditional expectation: 𝐸[𝑋|𝑌 > 𝑋] =

∑∞𝑥=1

- 1

- 2𝑥 (12)𝑥 =

∑∞𝑥=1 𝑥(14)𝑥. Using the formula

∑∞𝑥=1 𝑥𝑃(𝑋 = 𝑥|𝑌 > 𝑋) =

∑∞𝑥=1 𝑥 ⋅ 3(14)𝑥 = 3

∑∞𝑥=1 𝑥𝑟𝑥 = (1−𝑟𝑟)2 with 𝑟 = 1∕4: ∑∞𝑥=1 𝑥(14)𝑥 = (1−1∕4)1∕4 2 = (3∕4)1∕42 = 9∕161∕4 = 14 × 169 = 49. 𝐸[𝑋|𝑌 > 𝑋] = 3 × 94 = 129 = 34. Final Answer: The final answer is

|4∕3|
|---|

- Table 11. s1K sample stemming from the astronomy subset of OlympicArena. The black text is the question and the blue text is the solution. Based solely on the question we generate the light blue reasoning trace and blue response with Gemini, which we use for training.

An often-repeated fun fact is that humans produce more power per unit volume than stars. If the sun were the same size, but it produced the same amount of power per unit volume as a human, what would its surface temperature be? Assume the "average human" produces 100 watts of power and has a volume of 66400 cubic centimeters.

- A: 3500 K
- B: 10000 K
- C: 25000 K
- D: 40000 K
- E: 50000 K Using the numbers from the problem, the average human produces

100 66400 × 10−6

= 1506 W∕m3

𝑢 =

The volume of the sun is 43𝜋𝑅3⊙, so its new power output would be 𝑃 = 43𝜋𝑅3⊙𝑢. To find the equilibrium temperature, we use

𝑃 = 𝜎𝐴𝑇4 = 4𝜋𝜎𝑅2⊙𝑇4 Solving for 𝑇, we get

𝑇 = √4 𝑅⊙𝑢 3𝜎

)

(

6.96 × 108

(1506) 3

= √4

) = 49823 ≈ 50000 K.

(

5.67 × 10−8

The problem asks for the surface temperature of the sun if it had the same size but produced power per unit volume as a human. First, let’s find the power density of a human. Power produced by a human, 𝑃ℎ𝑢𝑚𝑎𝑛 = 100 W Volume of a human, 𝑉ℎ𝑢𝑚𝑎𝑛 = 66400 cm3 Convert the volume to cubic meters: 1 m = 100 cm 1 m3 = (100 cm)3 = 106 cm3 𝑉ℎ𝑢𝑚𝑎𝑛 = 66400 cm3 = 66400 × 10−6 m3 = 0.0664 m3 Power density of a human, 𝜌𝑝𝑜𝑤𝑒𝑟,ℎ𝑢𝑚𝑎𝑛 = 𝑃𝑉ℎ𝑢𝑚𝑎𝑛

= 0.0664100 Wm3

ℎ𝑢𝑚𝑎𝑛

𝜌𝑝𝑜𝑤𝑒𝑟,ℎ𝑢𝑚𝑎𝑛 = 1506.024 W∕m3 Now, consider the sun with the same size as it currently is, but with a power density equal to that of a human. We need the radius of the sun to calculate its volume. The average radius of the sun is 𝑅𝑠𝑢𝑛 = 6.95 × 108 m. Volume of the sun, 𝑉𝑠𝑢𝑛 = 43𝜋𝑅3𝑠𝑢𝑛 = 34𝜋(6.95 × 108 m)3

𝑉𝑠𝑢𝑛 = 34𝜋(337.159375 × 1024) m3 𝑉𝑠𝑢𝑛 = 1410.24 × 1024 × 𝜋 m3 𝑉𝑠𝑢𝑛 = 4.4307 × 1027 m3 If the sun had the same power density as a human, the total power produced by this modified sun would be:

𝑃𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑_𝑠𝑢𝑛 = 𝜌𝑝𝑜𝑤𝑒𝑟,ℎ𝑢𝑚𝑎𝑛 × 𝑉𝑠𝑢𝑛 𝑃𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑_𝑠𝑢𝑛 = 1506.024 W∕m3 × 4.4307 × 1027 m3 𝑃𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑_𝑠𝑢𝑛 = 6.6726 × 1030 W

The surface temperature of a star is related to its luminosity (total power output) through the Stefan-Boltzmann law: 𝐿 = 4𝜋𝑅2𝜎𝑇4 where 𝐿 is the luminosity (total power output), 𝑅 is the radius of the star, 𝜎 is the Stefan-Boltzmann constant, and 𝑇 is the surface temperature. In this case, the luminosity of the modified sun is 𝑃𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑_𝑠𝑢𝑛. The radius of the modified sun is the same as the actual sun, 𝑅𝑠𝑢𝑛. The Stefan-Boltzmann constant, 𝜎 = 5.67 × 10−8 Wm−2K−4. 𝑃𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑_𝑠𝑢𝑛 = 4𝜋𝑅2𝑠𝑢𝑛𝜎𝑇4 6.6726 × 1030 W = 4𝜋(6.95 × 108 m)2 × 5.67 × 10−8 Wm−2K−4 × 𝑇4 Calculate the term 4𝜋𝑅2𝑠𝑢𝑛𝜎: 𝑅2𝑠𝑢𝑛 = (6.95 × 108)2 = 48.3025 × 1016 m2

- 4𝜋𝑅2𝑠𝑢𝑛 = 4𝜋 × 48.3025 × 1016 = 607.08 × 1016 m2 4𝜋𝑅2𝑠𝑢𝑛𝜎 = 607.08 × 1016 m2 × 5.67 × 10−8 Wm−2K−4

4𝜋𝑅2𝑠𝑢𝑛𝜎 = 3441.93 × 108 WK−4 = 3.44193 × 1011 WK−4 Now, substitute this back into the equation:

6.6726 × 1030 = 3.44193 × 1011 × 𝑇4 Solve for 𝑇4:

𝑇4 = 36.44193×10.6726×103011 = 1.9386 × 1019 Solve for 𝑇:

𝑇 = (1.9386 × 1019)1∕4 = (193.86 × 1017)1∕4 𝑇 = (1.9386)1∕4 × (1019)1∕4 = (1.9386)0.25 × 104.75 Calculate (1.9386)0.25: Let 𝑦 = 1.93860.25, then ln𝑦 = 0.25ln(1.9386) ln(1.9386) ≈ 0.6619 ln𝑦 = 0.25 × 0.6619 = 0.165475

- 𝑦 = 𝑒0.165475 ≈ 1.1799 So, 𝑇 = 1.1799 × 104.75 = 1.1799 × 104 × 100.75 100.75 = 103∕4 = (103)1∕4 = (1000)1∕4 We know 54 = 625 and 64 = 1296, so (1000)1∕4 is between 5 and 6. Alternatively, 100.75 = 100.5 × 100.25 = √10 × 100.25 ≈ 3.16 × 1.778 = 5.62 𝑇 = 1.1799 × 104 × 5.62 = 6.63 × 104 K = 66300 K Let’s double-check the calculations. Power density of human: 𝜌𝑝𝑜𝑤𝑒𝑟,ℎ𝑢𝑚𝑎𝑛 = 1506.02 W∕m3 Volume of sun: 𝑉𝑠𝑢𝑛 = 4.43 × 1027 m3 Power of modified sun: 𝑃𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑_𝑠𝑢𝑛 = 1506.02 × 4.43 × 1027 = 6671.67 × 1027 = 6.67 × 1030 W Stefan-Boltzmann law: 𝑃 = 4𝜋𝑅2𝜎𝑇4 𝑇4 = 4𝜋𝑅𝑃2𝜎 4𝜋𝑅2𝑠𝑢𝑛𝜎 = 3.44 × 1011 𝑇4 = 36..44×1067×101130 = 1.939 × 1019 𝑇 = (1.939 × 1019)1∕4 = (193.9 × 1017)1∕4 𝑇 = (193.9)0.25 × (1017)0.25 = (193.9)0.25 × 104.25 Calculate (193.9)0.25: Let 𝑦 = 193.90.25, then ln𝑦 = 0.25ln(193.9) ln(193.9) ≈ 5.267 ln𝑦 = 0.25 × 5.267 = 1.31675

- 𝑦 = 𝑒1.31675 ≈ 3.73 𝑇 = 3.73 × 104.25 = 3.73 × 104 × 100.25 = 37300 × 1.778 = 66329 K The result is around 66000 K. This is not among the options. Let’s review the assumptions and calculations. Assumption: The sun radiates as a black body, which is reasonable for estimating surface temperature.

Consider potential errors in calculations. Let’s try a simpler approach by setting up ratios. The power per unit volume of the modified sun is equal to that of a human.

𝑃𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑_𝑠𝑢𝑛

𝑉𝑠𝑢𝑛 = 𝑃𝑉ℎ𝑢𝑚𝑎𝑛

ℎ𝑢𝑚𝑎𝑛 We also know 𝑃𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑_𝑠𝑢𝑛 = 4𝜋𝑅2𝑠𝑢𝑛𝜎𝑇𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑4 _𝑠𝑢𝑛. 4𝜋𝑅2𝑠𝑢𝑛𝜎𝑇𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑4 _𝑠𝑢𝑛

= 𝑃𝑉ℎ𝑢𝑚𝑎𝑛

4 3𝜋𝑅3𝑠𝑢𝑛

ℎ𝑢𝑚𝑎𝑛 3𝜎𝑇𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑4 _𝑠𝑢𝑛

𝑅𝑠𝑢𝑛 = 𝜌𝑝𝑜𝑤𝑒𝑟,ℎ𝑢𝑚𝑎𝑛

𝑇𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑4 _𝑠𝑢𝑛 = 𝜌𝑝𝑜𝑤𝑒𝑟,ℎ𝑢𝑚𝑎𝑛3𝜎 𝑅𝑠𝑢𝑛 Substitute the values:

𝜌𝑝𝑜𝑤𝑒𝑟,ℎ𝑢𝑚𝑎𝑛 = 1506.02 W∕m3 𝑅𝑠𝑢𝑛 = 6.95 × 108 m 𝜎 = 5.67 × 10−8 Wm−2K−4 𝑇𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑4 _𝑠𝑢𝑛 = 15063×5.02×6.67×10.95×10−8 8 𝑇𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑4 _𝑠𝑢𝑛 = 1046617.01×10.839×10−8 8 = 11..0466×10701×10−712 = 0.6153 × 1019 = 6.153 × 1018 𝑇𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑_𝑠𝑢𝑛 = (6.153 × 1018)1∕4 = (6153 × 1015)1∕4 𝑇𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑_𝑠𝑢𝑛 = (6153)0.25 × (1015)0.25 = (6153)0.25 × 103.75 Calculate (6153)0.25: Let 𝑦 = 61530.25, then ln𝑦 = 0.25ln(6153) ln(6153) ≈ 8.724 ln𝑦 = 0.25 × 8.724 = 2.181

- 𝑦 = 𝑒2.181 ≈ 8.85

𝑇𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑_𝑠𝑢𝑛 = 8.85 × 103.75 = 8.85 × 103 × 100.75 = 8850 × 5.62 = 49737 K This is close to option E: 50000 K. Let’s re-check the units and calculations carefully.

Power density of human: 0.0664100 Wm3 = 1506.02 W∕m3 The formula derived was 𝑇4 = 𝜌𝑝𝑜𝑤𝑒𝑟,ℎ𝑢𝑚𝑎𝑛3𝜎 𝑅𝑠𝑢𝑛. Units check:

[𝜌𝑝𝑜𝑤𝑒𝑟,ℎ𝑢𝑚𝑎𝑛] = W∕m3 [𝑅𝑠𝑢𝑛] = m [𝜎] = Wm−2K−4 (W∕m3)×m Wm−2K−4 = WmW∕m−2K2−4 = mW2 × mW2K4 = K4

The units match for 𝑇4. Numerical calculation check: Numerator: 1506.02 × 6.95 × 108 = 10466.839 × 108 = 1.0466 × 1012 Denominator: 3 × 5.67 × 10−8 = 17.01 × 10−8

Ratio: 117.0466×10.01×10−812 = 0.061528 × 1020 = 6.1528 × 1018 𝑇4 = 6.1528 × 1018 𝑇 = (6.1528 × 1018)1∕4 = (6152.8 × 1015)1∕4 𝑇 = (6152.8)0.25 × 103.75 Let’s use logarithms for precision.

log10(6152.8) ≈ 3.789 0.25 × 3.789 = 0.94725 100.94725 ≈ 8.856

- 𝑇 = 8.856 × 103.75 = 8.856 × 103 × 100.75 = 8856 × 5.623 = 49806 K This is very close to 50000 K. Possible sources of minor discrepancies: rounding errors, precision of constants. Let’s verify the formula derivation. Power per unit volume = TotalVolumePower

𝜌𝑝𝑜𝑤𝑒𝑟 = 𝑉𝐿 𝐿 = 4𝜋𝑅2𝜎𝑇4

𝑉 = 43𝜋𝑅3 𝜌𝑝𝑜𝑤𝑒𝑟 = 4𝜋𝑅4 2𝜎𝑇4

= 3𝜎𝑇𝑅 4 If the modified sun has the power density of a human: 𝜌𝑝𝑜𝑤𝑒𝑟,ℎ𝑢𝑚𝑎𝑛 =

3𝜋𝑅3

3𝜎𝑇𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑4 _𝑠𝑢𝑛 𝑅𝑠𝑢𝑛

𝑇𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑4 _𝑠𝑢𝑛 = 𝜌𝑝𝑜𝑤𝑒𝑟,ℎ𝑢𝑚𝑎𝑛3𝜎 𝑅𝑠𝑢𝑛 The formula derivation seems correct. Let’s use approximate values to check the order of magnitude.

𝜌𝑝𝑜𝑤𝑒𝑟,ℎ𝑢𝑚𝑎𝑛 ≈ 103 W∕m3 𝑅𝑠𝑢𝑛 ≈ 7 × 108 m 𝜎 ≈ 6 × 10−8 Wm−2K−4 𝑇4 ≈ 103×6×103×7×10−88 = 18×107×1011−8 = 0.38 × 1019 = 3.8 × 1018 𝑇 ≈ (3.8 × 1018)1∕4 = (3800 × 1015)1∕4 𝑇 ≈ (3800)0.25 × 103.75

(3800)0.25: 74 = 2401, 84 = 4096. So, between 7 and 8. ≈ 7.8 𝑇 ≈ 7.8 × 103.75 = 7800 × 5.6 = 43680 K The order of magnitude seems consistent with the detailed calculation. The problem statement is straightforward and doesn’t seem to have any hidden complexities. The assumptions made are standard for such problems. Consider the precision of the given values. Power of human: 100 W (exact) Volume of human: 66400 cm3 (exact) The result obtained is very close to option E. Given the nature of multiple-choice questions, the closest option is likely the intended answer. If there were any misinterpretations of the question, it would likely be in the setup. However, the phrase "If the sun were the same size, but it produced the same amount of power per unit volume as a human" seems clear. The surface temperature is calculated using the Stefan-Boltzmann law, assuming black-body radiation. Double-check the provided options:

- A: 3500 K
- B: 10000 K
- C: 25000 K
- D: 40000 K
- E: 50000 K Our calculated value is approximately 49800 K, which is closest to 50000 K. If we used more precise values for the constants, the result might be slightly different, but likely not enough to shift to another option. The fun fact mentioned in the question implies that the power density of a human is higher than that of the sun. Let’s verify this. Power output of the sun: 3.846 × 1026 W Volume of the sun: 4.43 × 1027 m3

Power density of the sun: 34..846×1043×102726 = 0.868 W∕m3 Power density of human: 1506 W∕m3 Indeed, the power density of a human is significantly higher than that of the sun. Final check of the calculation steps and logic. The approach of equating the power density of the modified sun to that of a human and then using the Stefan-Boltzmann law seems correct.

Solution: The power produced by an average human is 𝑃ℎ𝑢𝑚𝑎𝑛 = 100 W, and the volume of an average human is 𝑉ℎ𝑢𝑚𝑎𝑛 = 66400 cm3 = 0.0664 m3. The power density of a human is 𝜌𝑝𝑜𝑤𝑒𝑟,ℎ𝑢𝑚𝑎𝑛 = 𝑃𝑉ℎ𝑢𝑚𝑎𝑛

= 0.0664100 Wm3 ≈ 1506.02 W∕m3. If the sun were the same size but produced the same amount of power per unit volume as a human, its power density would be 𝜌𝑝𝑜𝑤𝑒𝑟,𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑_𝑠𝑢𝑛 = 𝜌𝑝𝑜𝑤𝑒𝑟,ℎ𝑢𝑚𝑎𝑛. The radius of the sun is 𝑅𝑠𝑢𝑛 = 6.95 × 108 m. The volume of the sun is 𝑉𝑠𝑢𝑛 = 34𝜋𝑅3𝑠𝑢𝑛. The power density of the modified sun is given by 𝜌𝑝𝑜𝑤𝑒𝑟,𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑_𝑠𝑢𝑛 = 𝑃𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑𝑉 _𝑠𝑢𝑛

ℎ𝑢𝑚𝑎𝑛

.

𝑠𝑢𝑛

We also know that 𝑃𝑚𝑜𝑑𝑖𝑓𝑖𝑒𝑑_𝑠𝑢𝑛 = 4𝜋𝑅2𝑠𝑢𝑛𝜎𝑇4, where 𝑇 is the surface temperature of the modified sun. Equating the power density:

2 𝑠𝑢𝑛𝜎𝑇4

𝜌𝑝𝑜𝑤𝑒𝑟,ℎ𝑢𝑚𝑎𝑛 = 4𝜋𝑅

= 3𝑅𝜎𝑇4

4 3𝜋𝑅3𝑠𝑢𝑛

𝑠𝑢𝑛

Solving for 𝑇4: 𝑇4 = 𝜌𝑝𝑜𝑤𝑒𝑟,ℎ𝑢𝑚𝑎𝑛3𝜎 𝑅𝑠𝑢𝑛 Substitute the values: 𝑇4 = 15063×5.02.67×10W∕m−83×6Wm.95×10−2K−48 m 𝑇4 = 11..0466×10701×10−712 = 6.1528 × 1018 K4 Solving for 𝑇: 𝑇 = (6.1528 × 1018)1∕4 = (6152.8)0.25 × 1018∕4 = (6152.8)0.25 × 104.5 Using logarithms:

log10(6152.8) ≈ 3.789 0.25 × 3.789 = 0.94725 (6152.8)0.25 = 100.94725 ≈ 8.856

- 𝑇 = 8.856 × 104.5 = 8.856 × 104 × 100.5 = 88560 × √10 ≈ 88560 × 3.16 = 280000 K Let’s redo the exponent calculation: 18∕4 = 4.5. 𝑇 = 8.856 × 104.5 = 8.856 × 104 × 100.5 - error in previous step. 𝑇 = (6.1528 × 1018)0.25 = 49806 K The surface temperature would be approximately 49806 K, which is closest to 50000 K. Final Answer: The final answer is

|𝐸|
|---|

### E. Test-time scaling details

#### E.1. Sequential scaling ablations

<|im_start|>user What is the answer to Life, the Universe and Everything? Think for up to 2048 tokens. <|im_start|>assistant

<|im_start|>user What is the answer to Life, the Universe and Everything? Think for up to 64 steps. <|im_start|>assistant

<|im_start|>64 steps left Let me break down this question into the three parts it is asking for:

<|im_start|>think Let me break down this question into the three parts it is asking for: 1) Life 2) Universe 3) Everything Let me start with life...

1) Life 2) Universe 3) Everything <|im_start|>63 steps left Let me start with life...

<|im_start|>answer The answer is...

<|im_start|>answer The answer is...

Figure 10. Token and step instruction data formats for controlling test-time compute. We only train our model on the reasoning trace and the answer.

Table 12. Scaling thinking time via tokens-conditional control. All metrics are averaged over the 30 questions in AIME24.

Tokens instructed (→) 1024 2048 4096 8192 16384 No intervention at test-time

Thinking tokens 7939 7158 8263 7108 7500 Answer tokens 689 669 659 722 724 AIME24 26.7 30.0 33.3 33.3 40.0

Forcing end of thinking when token budget is reached

Thinking tokens 1024 2048 4031 5664 6330 Answer tokens 15 15 142 722 691 AIME24 3.3 30.0 33.3 33.3 40.0

Token-conditional control One general approach is to simply tell a model in the prompt precisely how many tokens it should generate. Ideally, the model can keep track of its token count and adjust its generation to finish within the desired limits. We experiment with this approach by training a model with token instructions using the format in Figure 10 (left). We bucket the lengths of the reasoning traces from our 1,000 training examples into powers of two (rounded upwards) and add a corresponding instruction to the user prompt. For example, if the instruction says “Think for up to 2048 tokens”, then the reasoning trace has anywhere between 1024 and 2048 tokens. In Table 12, we show that after training the model hardly follows the token instruction. It does sometimes generate more tokens when given a higher limit but often overshoots the limit. This may not be unique to our model as prior work suggests that OpenAI o1-mini can also not follow token instructions (Zhang & Chen, 2024). To prevent exceeding the limit, we test budget forcing the thinking to end once the limit is reached. This leads to perfect control (Table 12 (lower)). With budget forcing, the scaling trend is also clearer as the model can no longer overshoot the limit when given a small thinking budget. This leads to better test-time scaling values for Token Prompting + budget forcing in Table 3. To compute Control reported in Table 3 for token-conditional control variants we divide the number of times the thinking tokens in Table 12 are less than the upper limit by the total evaluations (2/5 for without intervention; 5/5 for with intervention).

Step-conditional control Token instructions fail as current models cannot count tokens. To accommodate this lack of capability, we experiment with making the counting more coarse-grained. We partition the reasoning traces into steps and ask the model to think for a specific number of steps rather than tokens. We split our reasoning traces on double newlines into steps, which we find act as intuitive separators based on manual inspection of samples. We bucket our training samples into powers of 2 depending on their number of steps and add a corresponding step instruction following the format in Figure 10 (right). This format is based on early experiments, where we found the model to be more likely to adhere to the step limit

- Table 13. Scaling thinking time via step-conditional control. All metrics are averaged over the 30 samples in AIME24. Token counts ignore the thinking and step delimiters.

Steps instructed (→) 16 32 64 128 256 No intervention at test-time

Steps used 123 90 80 82 136 Tokens per step 60 70 69 66 56 Thinking tokens 7252 6277 5396 5552 7551 Answer tokens 665 653 735 777 754 AIME24 33.3 23.3 33.3 36.7 33.3

Forcing end of thinking when 0 steps are reached

Steps used 16 32 59 78 136 Tokens per step 96 94 80 70 56 Thinking tokens 1517 2963 4636 5409 7551 Answer tokens 1111 788 799 794 754 AIME24 23.3 23.3 33.3 36.7 33.3

- Table 14. Scaling thinking time via class-conditional control. We report “accuracy / average thinking tokens per sample”; the higher the accuracy and the fewer the thinking tokens (inference cost) the better.

Prompt appended to the question after two newlines AIME24 MATH500 GPQA

30.0% / 8033

90.4% / 2537

56.6% / 4177

Answer after a short amount of thinking. Do not spend excessive time double-checking your work.

36.7% / 9651

91.4% / 3875

51.0% / 4827

Answer after a long amount of thinking. If you feel like you are finished early, spend the extra time trying to doublecheck your work until you are absolutely sure that you have the correct answer.

Without generic prompt appending 50.0% / 6109

93.0% / 3298

57.6% / 3510

when counting down (“3 steps left...2 steps left”) rather than counting up (“Step2...Step3...”). This is likely because if counting down, the final step is always 1, which will act as a strong prior to the model to finish its generation. If counting up, the final step before the answer varies, thus if the model does not remember the original step instruction, it may fail to stop. We conclude the following from our results in Table 13: (1) The model still struggles to adhere to the step limit. The model sometimes simply continues counting into negative steps, e.g. “-1 steps left”. To solve this issue, we automatically stop the thinking process once 0 steps are reached and then force the model to transition to answering mode by appending the answer token delimiter (§3). This leads to perfect step adherence (lower half of Table 13), yet problems remain. (2) The model compensates for fewer steps by making each step longer. For example, when forced to use up to 16 steps vs 256 steps, the model generates an average of 96 tokens per step vs 56. Despite this issue, more steps still clearly correlate with more total thinking tokens in Table 13 and better performance leading to a positive slope (3) Step instructions are more costly than other methods. The step delimiters require around 6 tokens each which for e.g. 64 steps adds up to a total of around 380 tokens. When ignoring the step delimiters in token counts as in Table 13, the model still requires 7551 thinking tokens on average to achieve only 33.3% on AIME24. To compute Control reported in Table 3 for step-conditional control variants, we first decide that 100 tokens are an upper limit per step and then multiply this number by the steps instructed to arrive at a proxy total token limit, e.g. 1600 for 16 steps instructed. We then check whether the thinking tokens in Table 13 fit within the respective limit for each evaluation run (3/5 for without intervention; 5/5 for with intervention). For the model in Figure 7, we use a model with step-conditional control trained on an earlier version of our data and using an earlier version of our evaluation codebase.

Class-conditional control OpenAI exposes test-time compute control to users via a “reasoning_effort” API parameter with three possible settings: low, medium, and high.3 The OpenAI documentation also states that “Reducing reasoning effort can result in faster responses and fewer tokens used on reasoning in a response." suggesting that they are unable to control test-time compute with guarantees. Thus, maybe OpenAI simply adjusts the prompt or system instruction depending on the reasoning effort desired. In Table 14, we show that separate prompts for short and long thinking allow us to control thinking time to some extent: Prompting the model to think for longer leads to longer thinking. However, it does not reliably improve performance and control is not precise. The current adherence to control may suffice when we only have three classes, but it might not scale to finer-grained classes. To compute Control reported in Table 3 for this method, we assume that prompting the model to think for a short time in Table 14 should produce fewer tokens than the default for AIME24, while the long prompt should produce more. As 8033 > 6109 and 9651 > 6109, one out of two follows our expected control thus Control is 50%.

#### E.2. Examples for rejection sampling ablation

3https://github.com/openai/openai-python/blob/44d6210f101abedeb2dd68507fcffcb329df70ea/ src/openai/types/chat/completion_create_params.py#L172

- Table 15. Answers generated by an earlier version of s1-32B when rejection sampling on AIME24. The black text is the question and the yellow text is the solution. Based solely on the question we generate the light blue reasoning trace and blue response with s1-32B using a temperature of 1. We generate until the reasoning trace has less than 4000 tokens for the first and less than 8000 tokens for the second example. It required on average 199 (1) tries per example until thinking was ≤ 4000 (≤ 8000 tokens).

2 ) be points in the coordinate plane. Let  be the family of segments 𝑃𝑄 of unit length lying in the first quadrant with 𝑃 on the 𝑥-axis and 𝑄 on the 𝑦-axis. There is a unique point 𝐶 on 𝐴𝐵, distinct from 𝐴 and 𝐵, that does not belong to any segment from  other than 𝐴𝐵. Then 𝑂𝐶2 = 𝑞𝑝, where 𝑝 and 𝑞 are relatively prime positive integers. Find 𝑝 + 𝑞.

√3

Let 𝑂(0,0),𝐴(12,0), and 𝐵(0,

23 [Correct generation with 3981 thinking tokens and 1055 answer tokens] The problem asks for a point 𝐶 on the segment 𝐴𝐵 that is not contained in any other segment from the family , except for 𝐴𝐵 itself. The points 𝑂,𝐴,𝐵 form a triangle in the first quadrant.

𝑂 = (0,0) 𝐴 = (12,0) 𝐵 = (0,

√3

2 ) The segment 𝐴𝐵 connects 𝐴(21,0) and 𝐵(0,

√3

2 ). The equation of the line containing 𝐴𝐵 can be found using the two points: Slope 𝑚 =

√3 2 −0 0−12

= −√3

Using the point-slope form 𝑦 − 𝑦1 = 𝑚(𝑥 − 𝑥1) with point 𝐴(12,0): 𝑦 − 0 = −√3(𝑥 − 12) 𝑦 = −√3𝑥 + √3

2

√3

2 ). The family  consists of segments 𝑃𝑄 of unit length, where 𝑃 is on the 𝑥-axis and 𝑄 is on the 𝑦-axis. Let 𝑃 = (𝑎,0) and 𝑄 = (0,𝑏). The length of 𝑃𝑄 is

The segment 𝐴𝐵 lies in the first quadrant and connects (21,0) to (0,

√(𝑎 − 0)2 + (0 − 𝑏)2 = √𝑎2 + 𝑏2 = 1.

So 𝑎2 + 𝑏2 = 1. Since 𝑃 is on the positive 𝑥-axis and 𝑄 is on the positive 𝑦-axis, we have 𝑎 > 0 and 𝑏 > 0.

The segment 𝑃𝑄 connects (𝑎,0) and (0,𝑏). The equation of the line containing 𝑃𝑄 is 𝑥𝑎 + 𝑦𝑏 = 1. The region covered by the family  is the area bounded by the lines containing these segments. The endpoints (𝑎,0) and

(0,𝑏) vary such that 𝑎2 + 𝑏2 = 1. Consider the parametrization 𝑎 = cos𝜃, 𝑏 = sin𝜃, where 0 < 𝜃 < 𝜋2. The segment 𝑃𝑄 connects (cos𝜃,0) and (0,sin𝜃). The equation of the line is cos𝑥𝜃 + sin𝑦𝜃 = 1. We are looking for a unique point 𝐶 on 𝐴𝐵, distinct from 𝐴 and 𝐵, that does not belong to any segment from  other than 𝐴𝐵. Let 𝐶 be a point on 𝐴𝐵. We can parametrize the points on 𝐴𝐵 as (1 − 𝑡)𝐴 + 𝑡𝐵 for 0 ≤ 𝑡 ≤ 1. 𝐶 = (1 − 𝑡)(12,0) + 𝑡(0,

√3

√3

2 𝑡). Since 𝐶 is distinct from 𝐴 and 𝐵, we have 0 < 𝑡 < 1. Suppose the point 𝐶 lies on some segment 𝑃𝑄 ∈ , where 𝑃 = (cos𝜃,0) and 𝑄 = (0,sin𝜃).

2 ) = (12 − 21𝑡,

The point 𝐶 satisfies the equation cos𝑥𝜃 + sin𝑦𝜃 = 1.

Plugging in the coordinates of 𝐶:

√3

- 1

- 2−21𝑡 cos𝜃 +

2 𝑡

sin𝜃 = 1 1−𝑡

2cos𝜃 + √3𝑡

2sin𝜃 = 1 Multiply by 2:

- 1−𝑡 cos𝜃 + √3𝑡

sin𝜃 = 2 (1 − 𝑡)sin𝜃 + √3𝑡cos𝜃 = 2sin𝜃 cos𝜃 sin𝜃 − 𝑡sin𝜃 + √3𝑡cos𝜃 = sin(2𝜃)∕2 We are looking for a point 𝐶 on 𝐴𝐵 such that the only segment from  containing 𝐶 is 𝐴𝐵 itself. The segment 𝐴𝐵 is part of the family .

For 𝐴𝐵, 𝑃 = 𝐴 = (12,0) and 𝑄 = 𝐵 = (0,

√3

2 ). Here 𝑎 = 21, 𝑏 = √3

2 . 𝑎2 + 𝑏2 = (21)2 + (√3 2 )2 = 14 + 43 = 1. The line segment 𝐴𝐵 connects (12,0) and (0,

√3

2 ). The equation of the line is 1∕2𝑥 + √3∕2𝑦

= 1, which simplifies to 2𝑥 + √23

𝑦 = 1, or

√3𝑥 + 𝑦 = √3

2 . This matches the equation 𝑦 = −√3𝑥 + √3

2 .

Consider the envelope of the family of lines cos𝑥𝜃 + sin𝑦𝜃 = 1. Differentiating with respect to 𝜃:

−𝑥cossin2 𝜃𝜃 + 𝑦sincos2 𝜃𝜃 = 0 𝑦cos𝜃

sin2 𝜃 = 𝑥cossin2 𝜃𝜃 𝑦cos3 𝜃 = 𝑥sin3 𝜃 𝑦 𝑥 = tan3 𝜃

tan𝜃 = (𝑥𝑦)1∕3 From the line equation: 𝑦sin𝜃 = sin𝜃 cos𝜃 − 𝑥sin𝜃

𝑦 = cos𝜃 − 𝑥 Substitute cos𝜃 = √1+tan1 2 𝜃

= √1+(𝑦1∕𝑥)2∕3

cos𝜃 = (𝑥2∕3+𝑥1∕3𝑦2∕3)1∕2 sin𝜃 = tan𝜃 cos𝜃 = (𝑥𝑦)1∕3 (𝑥2∕3+𝑥1∕3𝑦2∕3)1∕2 = (𝑥2∕3+𝑦1∕3𝑦2∕3)1∕2 Substitute back into the line equation:

𝑥 cos𝜃 + sin𝑦𝜃 = 1 𝑥(𝑥2∕3 + 𝑦2∕3)1∕2∕𝑥1∕3 + 𝑦(𝑥2∕3 + 𝑦2∕3)1∕2∕𝑦1∕3 = 1 𝑥2∕3(𝑥2∕3 + 𝑦2∕3)1∕2 + 𝑦2∕3(𝑥2∕3 + 𝑦2∕3)1∕2 = 1 (𝑥2∕3 + 𝑦2∕3)3∕2 = 1 𝑥2∕3 + 𝑦2∕3 = 1 The envelope of the family  is the astroid 𝑥2∕3 + 𝑦2∕3 = 1. The segments from  are tangent to this astroid. The point 𝐶 on 𝐴𝐵 that we are looking for is the point where 𝐴𝐵 is tangent to the astroid. The equation of 𝐴𝐵 is 𝑦 = −√3𝑥 + √3

2 .

Substitute into the astroid equation: 𝑥2∕3 + (−√3𝑥 + √3

2 )2∕3 = 1 Let 𝑢 = 𝑥1∕3 and 𝑣 = 𝑦1∕3. The astroid is 𝑢2 + 𝑣2 = 1. The point 𝐶 lies on both the line 𝐴𝐵 and the astroid. Differentiating the astroid equation implicitly:

- 2

- 3𝑥−1∕3 + 23𝑦−1∕3𝑦′ = 0 𝑦′ = −𝑥𝑦1∕31∕3 = −(𝑥𝑦)1∕3

The slope of the tangent to the astroid at a point (𝑥,𝑦) is −√3

𝑦 𝑥.

The slope of the line 𝐴𝐵 is −√3. So, at the point of tangency, −√3

𝑦 𝑥 = −√3.

√

𝑦 𝑥 = √3

- 3

𝑦 𝑥 = (√3)3 = 3√3

The point 𝐶 lies on the line 𝑦 = −√3𝑥 + √3 2 and satisfies 𝑥𝑦 = 3√3.

So 𝑦 = 3√3𝑥. Substitute into the line equation:

3√3𝑥 = −√3𝑥 + √3

2

- 3𝑥 = −𝑥 + 12

- 4𝑥 = 12 𝑥 = 18

Then 𝑦 = 3√3(18) = 3√3

8 . The point 𝐶 is (18, 3

√3 8 ).

We need to check if this point lies on the segment 𝐴𝐵. The point 𝐶 is of the form (12 − 21𝑡,

√3

2 𝑡).

- 1

- 2 − 12𝑡 = 18 ⟹ 4 − 4𝑡 = 1 ⟹ 3 = 4𝑡 ⟹ 𝑡 = 34. √3

2 𝑡 = √3

2 ⋅ 34 = 3√3

8 .

So 𝐶 lies on 𝐴𝐵. We need to make sure 0 < 𝑡 < 1, which is true for 𝑡 = 34. Now, we need to check if this point 𝐶 does not belong to any other segment from . The point 𝐶 lies on the astroid. The segment 𝐴𝐵 is tangent to the astroid at 𝐶. Any other segment from  that contains 𝐶 would have 𝐶 as an interior point, not a tangent point. Consider the parametrization of the family . A point (𝑥,𝑦) belongs to some segment in  iff there exists 𝜃 such that cos𝑥𝜃 + sin𝑦𝜃 = 1. For the point 𝐶 = (18, 3

√3 8 ), we have:

√3∕8 sin𝜃 = 1

1∕8 cos𝜃 + 3

cos𝜃 + 3√3

- 1

sin𝜃 = 8 sin𝜃 + 3 3cos𝜃 = 8sin𝜃 cos𝜃 sin𝜃 + 3√3cos𝜃 = 4sin(2𝜃) We know that 𝐶 lies on 𝐴𝐵, which corresponds to some specific segment in . The problem statement assumes the existence and uniqueness of such a point 𝐶. The unique point 𝐶 on 𝐴𝐵, distinct from 𝐴 and 𝐵, that does not belong to any segment from  other than 𝐴𝐵. This means that if 𝐶 lies on another segment 𝑃′𝑄′ ∈ , then 𝑃′𝑄′ = 𝐴𝐵. The point 𝐶 is the point of tangency of 𝐴𝐵 with the astroid. Now we need to find 𝑂𝐶2.

𝑂𝐶2 = (81)2 + (3√3

8 )2 = 641 + 2764 = 6428 = 167 . Here 𝑝 = 7 and 𝑞 = 16. 𝑝 and 𝑞 are relatively prime. 𝑝 + 𝑞 = 7 + 16 = 23. Final check of the reasoning: The family  consists of unit-length segments with endpoints on the axes. The envelope of this family is the astroid 𝑥2∕3 + 𝑦2∕3 = 1. The segments from  are tangent to the astroid. The segment 𝐴𝐵 is also a member of the family . We are looking for a point 𝐶 on 𝐴𝐵 that lies on the astroid. This point 𝐶 is the point of tangency of 𝐴𝐵 with the astroid.

40

If a point on 𝐴𝐵 lies on the astroid, it belongs to the tangent segment at that point. Since 𝐴𝐵 is the tangent segment, this point does not belong to any other segment from . Consider the case when the segment 𝐴𝐵 intersects the astroid at two points. If the segment 𝐴𝐵 is not tangent to the astroid, then there are two points of intersection. Let the points of intersection be 𝐶1 and 𝐶2.

- If 𝐶1 lies on 𝐴𝐵, then the tangent to the astroid at 𝐶1 is the segment 𝐴𝐵.

- If 𝐶2 lies on 𝐴𝐵, then the tangent to the astroid at 𝐶2 is the segment 𝐴𝐵. The slope of 𝐴𝐵 is −√3. We found the point on the astroid with tangent slope −√3. The problem statement guarantees the uniqueness of such a point 𝐶. The point 𝐶 must be interior to 𝐴𝐵, so 0 < 𝑡 < 1.

The parameter 𝑡 = 34 is in this range. The point 𝐶 is distinct from 𝐴 and 𝐵.

If 𝐶 = 𝐴, then 𝑡 = 0, 𝐶 = (21,0). If 𝐶 = 𝐵, then 𝑡 = 1, 𝐶 = (0,

√3

2 ). The point of tangency 𝐶 is the unique point on 𝐴𝐵 (excluding endpoints) that lies on the astroid. Solution: Let the family of segments  be defined by segments 𝑃𝑄 of unit length with 𝑃 on the positive 𝑥-axis and 𝑄 on the positive 𝑦-axis. Let 𝑃 = (𝑎,0) and 𝑄 = (0,𝑏). The length condition gives 𝑎2 + 𝑏2 = 1. The segment 𝑃𝑄 lies on the line 𝑥𝑎 + 𝑦𝑏 = 1. Using the parametrization 𝑎 = cos𝜃, 𝑏 = sin𝜃 with 0 < 𝜃 < 𝜋2, the equation of the line is cos𝑥𝜃 + sin𝑦𝜃 = 1. The envelope of this family of lines is the astroid 𝑥2∕3 + 𝑦2∕3 = 1. The segments from  are tangent to this astroid. The segment 𝐴𝐵 connects 𝐴(12,0) and 𝐵(0,

√3

2 ). The equation of the line containing 𝐴𝐵 is 𝑦 = −√3𝑥 + √3

2 . This segment

is part of the family . We are looking for a unique point 𝐶 on 𝐴𝐵, distinct from 𝐴 and 𝐵, that does not belong to any segment from  other than 𝐴𝐵. This means 𝐶 is the point where 𝐴𝐵 is tangent to the astroid. The slope of the tangent to the astroid at a point (𝑥,𝑦) is −√3

𝑦 𝑥. The slope of 𝐴𝐵 is −√3.

Setting the slopes equal: −√3

𝑦 𝑥 = −√3 ⟹ 𝑥𝑦 = 3√3.

The point 𝐶 lies on the line 𝑦 = −√3𝑥 + √3

2 and satisfies 𝑦 = 3√3𝑥. Substituting, 3√3𝑥 = −√3𝑥 + √3

2 ⟹ 4√3𝑥 = √3

2 ⟹ 𝑥 = 18. Then 𝑦 = 3√3(18) = 3√3

8 . The point 𝐶 is (18, 3

√3 8 ).

2 𝑡) for 0 ≤ 𝑡 ≤ 1.

√3

We need to check if 𝐶 lies on 𝐴𝐵. The points on 𝐴𝐵 are (12 − 12𝑡,

For 𝑥 = 18, 12 − 12𝑡 = 18 ⟹ 𝑡 = 43. For 𝑦 = 3√3

8 , √3

2 𝑡 = 3√3

8 ⟹ 𝑡 = 34. Since 0 < 34 < 1, 𝐶 is an interior point of 𝐴𝐵. The point 𝐶 is the unique point on 𝐴𝐵 (excluding endpoints) that lies on the astroid. The tangent to the astroid at 𝐶 is the segment 𝐴𝐵. Thus, 𝐶 does not belong to any other segment from .

Finally, we calculate 𝑂𝐶2 = (18)2 + (3√3

8 )2 = 641 + 2764 = 2864 = 167 . We have 𝑝 = 7 and 𝑞 = 16. These are relatively prime. 𝑝 + 𝑞 = 7 + 16 = 23. Final Answer: The final answer is

|23|
|---|

[Incorrect generation with 5923 thinking tokens and 754 answer tokens] The problem asks for a point 𝐶 on the segment 𝐴𝐵 such that 𝐶 is not on any other segment from the family , except 𝐴𝐵 itself. The family  consists of segments 𝑃𝑄 of unit length, with 𝑃 on the 𝑥-axis and 𝑄 on the 𝑦-axis in the first quadrant. Let 𝑃 = (𝑥,0) and 𝑄 = (0,𝑦), where 𝑥 > 0 and 𝑦 > 0. The length of the segment 𝑃𝑄 is 1, so the distance between 𝑃 and 𝑄 is 1.

√(𝑥𝑥2 −+0)𝑦22=+1(0 − 𝑦)2 = 1 𝑥2 + 𝑦2 = 1 The segment 𝑃𝑄 lies in the first quadrant, so 𝑥 > 0 and 𝑦 > 0. The equation 𝑥2 + 𝑦2 = 1 represents the quarter circle in the first quadrant, with endpoints at (1,0) and (0,1). The points on the segment 𝑃𝑄 can be parameterized as (1 − 𝑡)𝑥 + 𝑡(0),(1 − 𝑡)0 + 𝑡𝑦) = (𝑥(1 − 𝑡),𝑦𝑡), where 0 ≤ 𝑡 ≤ 1.

√3

2 ). The equation of the line containing 𝐴𝐵 can be found using the two points: Slope 𝑚 =

The segment 𝐴𝐵 connects 𝐴(21,0) and 𝐵(0,

√3 2 −0 0−12

√3 2

= −√3.

=

−12

Using point-slope form with point 𝐴(12,0): 𝑦 − 0 = −√3(𝑥 − 12) 𝑦 = −√3𝑥 + √3

2 The segment 𝐴𝐵 lies in the first quadrant, between 𝑥 = 0 and 𝑥 = 12. Alternatively, we can parameterize the points on 𝐴𝐵. Let 𝐶 be a point on 𝐴𝐵. Then 𝐶 = (1 − 𝜆)𝐴 + 𝜆𝐵 for some 0 ≤ 𝜆 ≤ 1.

√3

√3

2 𝜆). We are looking for a unique point 𝐶 on 𝐴𝐵, distinct from 𝐴 and 𝐵, that does not belong to any segment from  other than 𝐴𝐵. Consider a segment from , with endpoints 𝑃 = (𝑥,0) and 𝑄 = (0,𝑦), where 𝑥2 + 𝑦2 = 1. A point on this segment is of the form (𝑥(1 − 𝑡),𝑦𝑡), where 0 ≤ 𝑡 ≤ 1. We want to find a point 𝐶 on 𝐴𝐵, 𝐶 ≠ 𝐴,𝐵, such that if 𝐶 lies on a segment 𝑃𝑄 ∈ , then 𝑃𝑄 = 𝐴𝐵.

𝐶 = (1 − 𝜆)(12,0) + 𝜆(0,

2 ) = (12 − 12𝜆,

√3

Let 𝐶 = (12 − 12𝜆,

2 𝜆), where 0 < 𝜆 < 1. If 𝐶 lies on a segment 𝑃𝑄 from , then there exist 𝑥,𝑦 > 0 with 𝑥2 + 𝑦2 = 1, and 𝑡 ∈ [0,1] such that: 1 2 − 12𝜆 = 𝑥(1 − 𝑡) √3

2 𝜆 = 𝑦𝑡 From the first equation, 1 − 𝑡 = From the second equation, 𝑡 =

- 1

- 2−12𝜆 𝑥 = 1−2𝑥𝜆.

√3 2 𝜆 𝑦 = √3𝜆

2𝑦 .

Since 𝑡 ∈ [0,1], we have 1 − 𝑡 ∈ [0,1]. Also, 𝑥 > 0, so 1−2𝑥𝜆 > 0, which implies 1 − 𝜆 > 0, or 𝜆 < 1. This is consistent with the condition for point on 𝐴𝐵. We have 1 − 𝑡 = 1−2𝑥𝜆 and 𝑡 = √3𝜆

2𝑦 . Adding these equations: 1 = 1−2𝑥𝜆 + √3𝜆

2𝑦

- 2𝑥𝑦 = (1 − 𝜆)𝑦 + √3𝜆𝑥

We know 𝑥2 + 𝑦2 = 1. If the point 𝐶 belongs to the segment 𝐴𝐵 itself, then the endpoints of the segment are 𝐴 and 𝐵.

√3

So, the segment is from (12,0) to (0,

2 ). Here, 𝑥 = 12 and 𝑦 = √3

2 . Check if 𝑥2 + 𝑦2 = 1: (21)2 + (√3 2 )2 = 41 + 34 = 1. So 𝐴𝐵 ∈ . If 𝐶 belongs to another segment 𝑃𝑄 from , with endpoints 𝑃 = (𝑥,0) and 𝑄 = (0,𝑦), where (𝑥,𝑦) ≠ (12,

√3

2 ). Consider the geometric interpretation. The family  consists of segments of unit length with endpoints on the axes. The midpoints of these segments lie on the circle (𝑥 − 12)2 + (𝑦 − 12)2 = 12. Let the point 𝐶 correspond to some 𝜆. If 𝐶 lies on 𝑃𝑄, then 𝐶 = (𝑥(1 − 𝑡),𝑦𝑡). So, 21 − 12𝜆 = 𝑥(1 − 𝑡) and √3

2 𝜆 = 𝑦𝑡. We need to find a 𝜆 ∈ (0,1) such that the only solution to these equations, with 𝑥2 + 𝑦2 = 1, 𝑥,𝑦 > 0, is 𝑥 = 12 and 𝑦 = √3

2 . From the equations, 1 − 𝑡 = 1−2𝑥𝜆 and 𝑡 = √3𝜆

2𝑦 .

Substitute these into 𝑥2 + 𝑦2 = 1. Try a different approach. Consider the tangents to the family of segments. The segments are formed by points (𝑥,0) and (0,𝑦) with 𝑥2 + 𝑦2 = 1.

The equation of the line containing the segment 𝑃𝑄 is 𝑥𝑥 + 𝑦𝑦 = 1, or 𝑥𝑥 + 𝑦𝑦 = 1. The equation of the line is 𝑥𝑥 + 𝑦𝑦 = 1. Consider the equation of the segment in parametric form: (𝑥(1 − 𝑡),𝑦𝑡). If a point 𝐶 lies on this segment, then 𝐶 = (𝑥𝑐,𝑦𝑐), where 𝑥𝑐 = 𝑥(1 − 𝑡) and 𝑦𝑐 = 𝑦𝑡. So, 𝑥𝑐

𝑥 = 1 − 𝑡 and 𝑦𝑐

𝑦 = 𝑡.

𝑥𝑐 𝑥 + 𝑦𝑦𝑐 = 1 − 𝑡 + 𝑡 = 1.

The segments from  are parts of the lines 𝑥𝑥

= 1, where 𝑥20 + 𝑦20 = 1, 𝑥0,𝑦0 > 0. Consider the point 𝐶 = (12 − 21𝜆,

+ 𝑦𝑦

0

0

√3

2 𝜆). If 𝐶 lies on a segment from  with endpoints (𝑥,0) and (0,𝑦), then:

√3 2 𝜆 𝑦 = 1

- 1

- 2−21𝜆 𝑥 +

- 1−𝜆

- 2𝑥 + √3𝜆

2𝑦 = 1 We need to find a unique 𝜆 ∈ (0,1) such that the only solution (𝑥,𝑦) to this equation, with 𝑥2 + 𝑦2 = 1, 𝑥,𝑦 > 0, is (12,

√3

2 ). Substitute 𝑥 = 12 and 𝑦 = √3

2 into the equation:

+ √3𝜆 2(√3

1−𝜆 2(21)

= 1

2 )

1 − 𝜆 + 𝜆 = 1 1 = 1 This shows that any point on 𝐴𝐵 lies on the segment 𝐴𝐵 itself. Consider the line containing 𝐴𝐵: 𝑦 = −√3𝑥 + √3

2 . If a point 𝐶 on 𝐴𝐵 also lies on another segment from  with endpoints (𝑥,0) and (0,𝑦), then:

𝑥 + 𝑦𝑦𝐶 = 1. Let the point 𝐶 divide 𝐴𝐵 in the ratio 𝜇 ∶ (1 − 𝜇), where 0 < 𝜇 < 1. 𝐶 = (1 − 𝜇)𝐴 + 𝜇𝐵 = (1 − 𝜇)(12,0) + 𝜇(0,

𝑥𝐶

√3

√3

2 𝜇). If 𝐶 lies on a segment with endpoints (𝑥,0) and (0,𝑦), then:

2 ) = (21 − 12𝜇,

√3 2 𝜇 𝑦 = 1.

- 1

- 2−21𝜇 𝑥 +

We are looking for a point 𝐶 such that the line passing through 𝐶 with the form 𝑥𝑥

= 1, where 𝑥20 + 𝑦20 = 1, is unique.

+ 𝑦𝑦

0

0

Consider the tangents to the unit circle centered at the origin. The lines of the form 𝑥𝑥

= 1 are tangents to the circle

+ 𝑦𝑦

0

0

𝑥2 + 𝑦2 = 𝑟2. Consider the dual problem. The family of segments corresponds to points on the quarter circle 𝑥2 + 𝑦2 = 1 in the first quadrant.

= 1. The envelope of these lines is the curve whose tangents are these lines. The equation 𝑥𝑥

The lines containing the segments are 𝑥𝑥

+ 𝑦𝑦

0

0

= 1, with the constraint 𝑥20 + 𝑦20 = 1.

+ 𝑦𝑦

0

0

Let 𝑥0 = cos𝜃, 𝑦0 = sin𝜃, where 0 < 𝜃 < 𝜋2. The equations of the lines are 𝑥cos𝜃 + 𝑦sin𝜃 = 1.

The envelope of the family of lines 𝑥cos𝜃 + 𝑦sin𝜃 = 1 is the circle 𝑥2 + 𝑦2 = 1. The segments from  are parts of the tangents to the unit circle in the first quadrant. The point 𝐶 on 𝐴𝐵 that does not belong to any other segment from  means that the tangent to the unit circle passing through 𝐶 is unique.

√3

2 𝜆). The tangents to the unit circle passing through 𝐶 have the form 𝑥𝑥𝐶 + 𝑦𝑦𝐶 = 1. If the tangent is unique, then 𝐶 lies on the unit circle.

Consider the point 𝐶 = (12 − 21𝜆,

(12 − 12𝜆)2 + (√3

2 𝜆)2 = 1 (41 − 12𝜆 + 14𝜆2) + 34𝜆2 = 1

- 4 − 12𝜆 + 𝜆2 = 1

- 1

- 4𝜆2 − 2𝜆 + 1 = 4 4𝜆2 − 2𝜆 − 3 = 0 Solve for 𝜆:

𝜆 = −(−2)±√(−2)2−4(4)(−3) 2(4) = 2±√4+48 8 = 2±√52 8 = 2±2√13 8 = 1±√13

4 . We√13need≈ 3.06.< 𝜆 < 1.

4 = 44.6 > 1. 1−3.6

1+3.6

4 = −24.6 < 0. Something is wrong. The point 𝐶 is inside the unit circle. Let’s rephrase the condition. There is a unique point 𝐶 on 𝐴𝐵 such that if 𝐶 lies on a segment 𝑃𝑄 ∈ , then 𝑃𝑄 = 𝐴𝐵.

√3

2 𝜆). Consider a segment from  with endpoints (𝑥,0) and (0,𝑦), 𝑥2 + 𝑦2 = 1. The parametric form of the segment is (𝑥(1 − 𝑡),𝑦𝑡). If 𝐶 is on this segment, then for some 𝑡 ∈ [0,1]: 1 2 − 12𝜆 = 𝑥(1 − 𝑡) √3

Let the point 𝐶 on 𝐴𝐵 be (12 − 12𝜆,

- 2 𝜆 = 𝑦𝑡

If the segment is unique, then the values of 𝑥 and 𝑦 are unique. The only solution to 1−2𝑥𝜆 + √3𝜆

2𝑦 = 1, with 𝑥2 + 𝑦2 = 1, 𝑥,𝑦 > 0, should be 𝑥 = 12,𝑦 = √3

2 . Consider the intersection of the line containing 𝐴𝐵 with the boundary of the region formed by the segments in . The boundaries are the 𝑥-axis, 𝑦-axis, and the quarter circle 𝑥2 + 𝑦2 = 1.

= 1, where 𝑥20 + 𝑦20 = 1, 𝑥0,𝑦0 > 0.

Consider the dual problem again. The family of lines 𝑥𝑥

+ 𝑦𝑦

0

0

The envelope of these lines is the circle 𝑥2 + 𝑦2 = 1. A point belongs to a unique segment from  if it lies on the circle 𝑥2 + 𝑦2 = 1. The point 𝐶 on 𝐴𝐵 satisfies 𝐶 = 𝑡𝐴 + (1 − 𝑡)𝐵.

Let’s consider the case where the segment is 𝐴𝐵. Here 𝑥 = 12,𝑦 = √3

2 . The point 𝐶 can be represented as (21(1 − 𝑠),

2 𝑠) for 0 ≤ 𝑠 ≤ 1. We are looking for a point 𝐶 on 𝐴𝐵 that does not lie on any other segment from .

√3

Consider the geometry. The family  forms a region bounded by the axes and the quarter circle. The segments are chords of circles centered on the line 𝑦 = 𝑥, tangent to the axes. Let the endpoints of a segment be (cos𝜃,0) and (0,sin𝜃). A point on the segment is (cos𝜃(1 − 𝑡),sin𝜃𝑡).

√3

2 𝜆). If 𝐶 lies on such a segment, then

The point 𝐶 = (21 − 12𝜆,

√3

- 1

- 2−12𝜆 cos𝜃 = 1 − 𝑡 and

sin𝜃 = 𝑡. 1−𝜆

2 𝜆

- 2cos𝜃 + √3𝜆

2sin𝜃 = 1. We need a unique 𝜆 ∈ (0,1) such that the only solution (cos𝜃,sin𝜃) is (12,

√3

2 ), which corresponds to 𝜃 = 𝜋3. Consider the function 𝑓(𝜃) = 2cos1 𝜃 − 2cos𝜆 𝜃 + √3𝜆

2sin𝜃.

We need 𝑓(𝜃) = 1. If 𝜆 = 12, then 𝐶 = (14,

√3

4 ).

√3(1∕2)

1−1∕2 2cos𝜃 +

2sin𝜃 = 1 1

4cos𝜃 + √3

4sin𝜃 = 1 1

cos𝜃 + √3

sin𝜃 = 4 sin𝜃 + 3cos𝜃 = 4sin𝜃 cos𝜃 sin𝜃 + √3cos𝜃 = 2sin(2𝜃)

Divide by 2: 12 sin𝜃 + √3

2 cos𝜃 = sin(2𝜃)

sin(𝜃 + 𝜋3) = sin(2𝜃). 𝜃 + 𝜋3 = 2𝜃 + 2𝜋𝑘, or 𝜃 + 𝜋3 = 𝜋 − 2𝜃 + 2𝜋𝑘. 𝜃 = 𝜋3 − 2𝜋𝑘, so 𝜃 = 𝜋3.

- 3𝜃 = 𝜋 − 𝜋3 + 2𝜋𝑘 = 23𝜋 + 2𝜋𝑘. 𝜃 = 29𝜋 + 2𝜋𝑘3 .

- If 𝑘 = 0, 𝜃 = 29𝜋.

- If 𝑘 = 1, 𝜃 = 29𝜋 + 23𝜋 = 2𝜋+69 𝜋 = 89𝜋.

√3

If 𝜆 = 21, the point is (41,

4 ).

This point lies on segments corresponding to 𝜃 = 𝜋3 and 𝜃 = 29𝜋. The unique point 𝐶 means that for other values of 𝜆, there is more than one solution for (𝑥,𝑦).

Consider the tangent to the unit circle passing through 𝐶. If the tangent is unique, 𝐶 is on the unit circle.

This leads to 𝜆 = 1±√13

4 , neither of which is in (0,1). Consider the geometric interpretation of the family . The segments connect points on the axes such that the distance is 1. The locus of the centers of these segments is the circle (𝑥 − 21)2 + (𝑦 − 12)2 = 12, in the first quadrant. If a point 𝐶 lies on a segment from , then 𝐶 is on a line segment connecting (𝑥,0) and (0,𝑦) with 𝑥2 + 𝑦2 = 1. The line containing the segment is 𝑥𝑥

= 1. If 𝐶 belongs to a unique segment, then there is a unique line of this form passing through 𝐶. The point 𝐶 on 𝐴𝐵 is (12 − 12𝜆,

+ 𝑦𝑦

0

0

√3

2 𝜆). We need a unique (𝑥0,𝑦0) with 𝑥20 + 𝑦20 = 1, 𝑥0,𝑦0 > 0, such that

√3 2 𝜆 𝑦0 = 1.

1 2−21𝜆

𝑥0 +

Let the point 𝐶 be such that the line passing through 𝐶 with the form 𝑥𝑥

= 1 is unique, subject to 𝑥20 + 𝑦20 = 1.

+ 𝑦𝑦

This line is tangent to the unit circle. The point 𝐶 lies on the intersection of the line containing 𝐴𝐵 and the unit circle.

0

0

(12 − 12𝜆)2 + (√3

2 𝜆)2 = 1. We found no solution for 𝜆 ∈ (0,1).

The unique point 𝐶 means that if any other segment passes through 𝐶, it must be 𝐴𝐵. Consider the slope of the line containing the segment connecting (𝑥,0) and (0,𝑦). Slope is −𝑥𝑦. The equation is 𝑦 = −𝑥𝑦𝑥 + 𝑦. If 𝐶 lies on the segment, then 1−2𝑥𝜆 + √3𝜆

2𝑦 = 1.

Consider the geometric mean. Let the point 𝐶 have coordinates (𝑎,𝑏). The segment connecting (𝑥,0) and (0,𝑦) passes through (𝑎,𝑏). 𝑎 𝑥 + 𝑦𝑏 = 1. If 𝐶 is on 𝐴𝐵, then 𝑏 = −√3𝑎 + √3

2 . 𝑎 = 12 − 12𝜆, 𝑏 = √3

2 𝜆. The point 𝐶 is tangent to the envelope of the family of segments.

Solution: Let the family of segments be . The segments have endpoints 𝑃 = (𝑥,0) and 𝑄 = (0,𝑦) with 𝑥2 + 𝑦2 = 1 and 𝑥,𝑦 > 0.

The points on the segment 𝑃𝑄 satisfy 𝑥𝑥 + 𝑦𝑦 = 1. The segments lie on the lines 𝑥𝑥

= 1, where 𝑥20 + 𝑦20 = 1, 𝑥0,𝑦0 > 0. The segment 𝐴𝐵 has endpoints 𝐴(21,0) and 𝐵(0,

+ 𝑦𝑦

0

0

2 ). The equation of the line containing 𝐴𝐵 is 𝑦 = −√3𝑥 + √3

√3

2 . A point 𝐶 on 𝐴𝐵 can be parameterized as 𝐶 = (12 − 12𝜆,

2 𝜆), where 0 ≤ 𝜆 ≤ 1. If 𝐶 belongs to a segment from  with endpoints (𝑥,0) and (0,𝑦), then

√3

√3 2 𝜆 𝑦 = 1.

- 1

- 2−21𝜆 𝑥 +

We are looking for a unique point 𝐶 on 𝐴𝐵, distinct from 𝐴 and 𝐵, that does not belong to any segment from  other than 𝐴𝐵. This means that the only solution (𝑥,𝑦) to the equation with 𝑥2 + 𝑦2 = 1, 𝑥,𝑦 > 0, is (12,

√3

2 ). The segments from  are parts of the tangents to the unit circle. The point 𝐶 must be tangent to the envelope of the family of segments. The envelope is the unit circle itself. The unique point 𝐶 on 𝐴𝐵, distinct from 𝐴 and 𝐵, that lies on the unit circle. We need to solve:

(12 − 12𝜆)2 + (√3

2 𝜆)2 = 1 𝜆 = 12. The point 𝐶 = (14,

√3

4 ). 𝑂𝐶2 = (41)2 + (√3

4 )2 = 161 + 163 = 164 = 14.

𝑂𝐶2 = 41 = 𝑞𝑝, so 𝑝 = 1, 𝑞 = 4. 𝑝 and 𝑞 are relatively prime.

𝑝 + 𝑞 = 1 + 4 = 5. Final Answer: The final answer is

|5|
|---|

F. Version Control

V2 → V3 (2025-03):

- • Added §A
- • Added number of correct samples in §2

#### V1 → V2 (2025-02):

• Added citations and other small writing changes

