# arXiv:2602.16699v3[cs.CL]15May2026

## Calibrate-Then-Act: Cost-Aware Exploration in LLM Agents

Wenxuan Ding Nicholas Tomlin Greg Durrett New York University wd2403@nyu.edu

### Abstract

LLM agents are deployed in environments where they must interact to acquire information. In these scenarios, the agent must reason about inherent cost-uncertainty tradeoffs in how to act, such as when to stop exploring and commit to an answer. For instance, on a programming task, an agent might run the code it generates, or it might generate tests for that code snippet; the cost of writing and running a test is nonzero, but typically lower than the cost of running buggy code. In this work, we show that we can induce LLM agents to explicitly reason about balancing these cost-uncertainty tradeoffs, then act more optimally in their environments. We formalize multiple tasks, including retrieval-augmented QA and a file reading coding task, as sequential decision-making problems under uncertainty. Each problem has latent environment state that impacts the agent’s performance. We introduce a framework called Calibrate-Then-Act (CTA), where we pass the agent an inferred prior about this environment state to enable it to act more optimally. This information qualitatively changes agent behavior, and adds environment sensitivity to the agent which is not learned via standard RL training. Our results on a synthetic task, QA, and file reading show that making cost-benefit tradeoffs explicit with CTA helps agents discover more optimal decision-making strategies.

### 1 Introduction

[Figure 1]

[Figure 2]

Find the item id with max price in sales.csv

Find the item id with max price in sales_fr.tsv

Large language model (LLM) agents are increasingly tasked with operating in environments where information is incomplete. Behaving rationally requires gaining information by exploring the environment. However, exploration comes with a cost: every additional step increases API costs, interaction latency, and user burden.

[Figure 3]

[Figure 4]

This might not be standard; I’m only 50% confident I know what to do.

Okay, I’m 99% sure I should just read the csv and extract “price”. If I’m wrong, I’ll have to fix my mistake.

Deliberate Testing ↵

Guess-and-Go↵

column names?

[Figure 5]

[Figure 6]

INFO: …

> head -n 50

[Figure 7]

[Figure 8]

INFO: …

df = pd.read_csv("sales.csv")

df = pd.read_csv("sales_fr.tsv",

df["price"].max()

sep="\t") df["prix"].max()

This exploration and its cost come in many forms. In software development and debugging, agents must decide whether to run targeted checks or perform full execution before committing to a solution [68]. In machine learning experimentation, practitioners balance inexpensive proxy evaluations against costly full training runs under limited compute budgets [28, 24, 60]. In diagnosis and scientific discovery, additional tests or experiments reduce uncertainty but incur monetary, temporal, or safety costs [29, 33, 21]. Online decision-making settings such as shopping [61, 55],

[Figure 9]

[Figure 10]

PASS

PASS

The optimal strategy is a function of model’s success probability and environment costs of diﬀerent actions. Can LLMs behave optimally?

Figure 1: Given the same task, a coding agent may either verify assumptions via intermediate checks carefully (right) or attempt a direct solution as soon as possible (left). The optimal choice depends on uncertainty and specific cost constraints. Calibrate-ThenAct (CTA) materializes this information for better decision-making.

Preprint. Code: https://github.com/NSF-Simons-CosmicAI-Institute/CalibrateThenAct.

recommendation [25], and tool-augmented question answering [63] exhibit the same structure, as agents weigh further information gathering against acting with partial information.

Agent policies for this exploration depend in a complex way on their prompt, their inputs, and their training data. However, these policies are frequently static. For instance, ChatGPT Deep Research always asks a single round of clarifying questions before searching [12], and coding agents like SWE-agent [62] start by reading through an existing codebase. This contrasts with settings like Figure 1, where we see that a model can act more efficiently if it has confidence that it understands the problem setup and appropriately trades off exploration against action costs.

In this work, we ask: how can we develop LLMs that explore in an optimal way under varying cost and uncertainty profiles? Leveraging the strong reasoning capabilities of modern LLMs, we propose a framework called Calibrate-Then-Act (CTA), which decouples the calibration of uncertainty from the reasoning of action selection. The key insight is that by presenting priors explicitly to the model, we induce the model to reason about an underlying sequential decision-making problem abstractly and discover the optimal action.

We study two settings inspired by realistic problems: (1) knowledge-based QA with optional retrieval, where uncertainty is inferred from the model’s own confidence; and (2) coding tasks where priors regarding environmental structure (e.g., file schemas) are derived from cues learned through past experience. We show that even in these simple settings, LLM agents prompted or trained with RL fail to behave optimally. However, this is not due to an inability to perform the correct reasoning: we show with a synthetic “Pandora’s Box” problem [56], and show that even a small thinking model (Qwen3-8B) can follow the optimal policy. The Calibrate-Then-Act method builds on this result, feeding the agent with relevant information about the environment, determined by either estimating model confidence or a trained predictor of environment state. A zero-shot prompted implementation of this method, CTA-PROMPTED, already enables dynamic decision-making. We show that training with RL further improves performance, and notably outperforms baseline RL training of the agent.

Our contributions are: (1) Framing environment exploration as a sequential decision-making problem, unified across three settings; (2) The Calibrate-Then-Act method, which induces LLMs to reason about the optimality of their actions and achieve better cost-performance tradeoffs than baselines.

### 2 Background: Agentic Exploration with Costs

We formalize cost-aware environment exploration task as sequential decision-making problem. Our agent, which for the purposes of this work is an LLM, is given a query x and operates in some environment, which can be defined as a partially-observable Markov Decision Process W = (S,A,O,O,T,R,Dθ), a tuple of states S, actions A, observations O, observation function O, transition function T, reward function R, and parameterized discount function Dθ, which integrates the cost.

In the settings we consider, A and O are both string-valued spaces; LLMs produce string actions (code, API calls, etc.) and receive string-valued responses from the environment. The observation function O produces string realizations of the underlying environment; e.g., in Figure 1, the results of executing commands return string output in the terminal reflecting the underlying state of the environment.

The environment contains problem-critical unobserved features that will determine the agent’s performance, e.g., details about the formatting of the unobserved file in Figure 1. These can be thought of as a subset of the information in S. We represent these as a random variable Z taking values z ∈ Z.

The agent interacts with the environment over multiple timesteps before terminating. At each timestep t, the agent selects an action at ∈ A and receives an observation ot ∈ O; for simplicity, we assume that ot encodes at. Based on this information, we can form an idealized posterior distribution bt(Z) = p(Z | x,o0:t) which reflects remaining uncertainty over the latent variables. The action space generally consists of multiple exploration actions and a commit action, which terminates the episode by producing a final result. In Figure 1, exploration actions include querying aspects of the input data file. Other actions (not shown) include running the code or writing and running unit tests.

- Table 1: Unified formalization of cost-aware decision problems and their instantiations across tasks. We characterize each by latent variables z∗, a prior belief p over z, an action space A consisting of exploration and commit actions, observations O revealed through exploration, costs θ associated with the exploration actions, and a final reward R that discounts task success by incurred costs.

Setting z∗ ∈ Z p A O θ R UNIFIED (§2) Latent env. state

Prior over relevant latent variables

Actions Observations Cost Reward

z∗ ∈ {z1, . . . , zK} Prize-containing box {pk = π(z∗ = zk)}

PANDORA (§3.1)

Verify(k), Commit(k)

Whether zk has prize γ γTI(zk = z∗)

(pda, pret)

Answer probabilities with and without retrieval

pda = δ(z = kda(x)), pret = δ(z = kret)

Retrieve, Answer(a)

Retrieved context γ γTI(a = a∗)

QA (§3.2)

UNIT_TEST(f), CODE(d, q, s), ANSWER(a)

CODE (§3.3) z∗ = (zd∗, zq∗, zs∗) ∈ Zd × Zq × Zs

Format value, stdout + stderr (du, dc) dUu dCc I(a = a∗)

p(z | n)

Each action incurs different costs depending on the setting, while the commit action corresponds to returning a final solution. These costs are reflected by a discount factor Dθ(a1:T) ∈ [0,1], which discounts the value of successful task completion based on the exploration actions taken prior to commitment.

Finally, the agent receives reward upon committing:

R = I[task completed at at] · Dθ(a1:T),

Overall, the agent’s objective is to maximize the expected discounted reward R by carefully selecting actions that adaptively balance exploration and commitment in response to uncertainty and cost constraints of the environment.

### 3 Tasks

We now connect our general task setting to three concrete problems we address in this work. The first, Pandora’s Box [56], serves as a proof-of-concept instance of cost-aware exploration, where we demonstrate that even small LLMs have the capability to reason about uncertainty-cost tradeoffs in environments. We then describe two realistic problems where LLMs struggle to reason about costs out of the box, motivating the method in this paper. Table 1 summarizes how these problems map onto our general framework.

#### 3.1 Pandora’s Box: A Minimal Setting for Cost-Aware Decisions

Formalization We consider a variant of the classic Pandora’s Box Problem with discounted reward over time [56]. The task involves a finite set of K boxes {z1,z2,...,zK}, among which exactly one box z∗ contains a prize of value 1. This box is unknown to the agent and is drawn from a prior distribution

K

p(zk = z∗) = pk,

pk = 1. At each timestep t, the agent can either verify a box zk

k=1

of its choice to check if it contains the prize and discount their final reward by γ ∈ [0,1], or commit to a box given its current information and receive a reward R = γt · I(zk

t

= z∗).

t

This task can be viewed as a toy version of real-world use cases for LLM agents. For instance, if trying to identify a bug in a piece of code, we can view committing as directly taking action to fix a bug and verifying as writing a unit test to check the correctness of the code.

Knowledge of the prior probabilities is necessary to behave optimally. The optimal policy is to commit if its posterior probability of a box is greater than γ, which means we have better expected value for guessing than for verifying. Otherwise, the optimal decision is to verify the highest-probability remaining box. The full algorithm is shown in Algorithm 1 and proof is provided in Appendix H.

#### 3.2 Task QA: Knowledge QA with Optional Retrieval

We study a knowledge question answering setting in which an LLM can optionally acquire external information at a cost [15]. Given a factual query, the model must decide whether to rely on its parametric knowledge or defer commitment and retrieve additional evidence, trading off potential accuracy gains against latency and API costs.

Formalization Given a question x, a discount factor γ ∈ [0,1], and access to a retriever, the agent chooses between actions A = {retrieve,answer}. A retrieve action queries a retrieval system with the input x, while an answer action invokes the LLM to answer the question given the context so far, producing an answer a. The model receives reward R = γt · I(a = a∗), where a∗ is the ground-truth answer to x. This task only features up to one round of retrieval, so t is either 0 or 1.

Latent structure and prior There are two relevant latent variables for this problem. First, we define pda = p(a = a∗ | x) that the model will correctly answer the question if asked directly, without retrieval. We model pda with a distribution p(pda | x) = δ(z = k(x)) for an estimate kda(x) that the LLM returns the correct answer, where δ is the Dirac delta function. This distribution represents that we have a point estimate of the model’s probability to answer correctly.

Second, we define the probability of answering post-retrieval to be pret = p(a′ = a∗ | x,c), where c is the retrieved context. We similarly represent this as a delta function p(pret | x) = δ(z = kret)

Under this abstraction, the oracle policy retrieves whenever the expected discounted accuracy after retrieval exceeds that of direct answering: pret · γ ≥ pda.

#### 3.3 Task FILEREADING: Coding with Selective Testing

Following Figure 1, we design a coding setting to encapsulate the idea of a model choosing to either run code directly or test it. Concretely, the model needs to write code to load a data file with correct formatting options and return the answer to a task query (e.g., identifying the user_id associated with the maximum score). The true file schema is not specified in the input; instead, the model may infer likely formats from filename cues and past experience. Unlike the QA setting, these priors are not known to the model a priori, but must be learned from training on this task. Despite the simplicity of this setting, we show that LLM agents fail to learn the optimal behavior via RL.

Formalization Given a query x that contains a task specification and a CSV filename n, the agent needs to write code to load the file correctly and compute an answer a. Specifically, each file is associated with latent formatting attributes z = (zd,zq,zs) ∈ Z, where zd ∈ {,,;,\t} denotes the delimiter, zq ∈ {’,"} the quote character, and zs ∈ {0,1} the number of skipped header rows. Without making a correct inference about z, the task is not solvable.

At each timestep t, the agent selects one action at ∈ A from three types: UNIT_TEST(f), CODE(d,q,s), or ANSWER. A UNIT_TEST probes a chosen formatting attribute f and reveals its true value. A CODE action executes code written under the agent’s current belief (d,q,s) and returns feedback via stdout and stderr, which may contain the answer or signals useful for debugging and refinement. An ANSWER action commits to a final answer a′ and terminates the episode. The agent may interleave UNIT_TEST and CODE actions in any order, and may perform multiple CODE actions to refine its solution based on previous execution feedback.

Each UNIT_TEST and CODE action incurs multiplicative discounts du and dc, respectively. Upon committing at time T by ANSWER, the agent receives reward R = dUu · dCc · I(a′ = a∗), where a∗ denotes the ground-truth answer.

Prior The prior distribution p(z | n) over formatting attributes may be inferred from conventions or past experience, or provided explicitly by a format predictor. Prompt templates for these tasks are provided in Appendix J.

### 4 Method: Calibrating Agent Environment Exploration

To behave Pareto-optimally in a sequential decision-making problem, an agent must jointly compare the cost of additional exploration against the expected value of additional information to decide

- 000
- 001
- 002
- 003
- 004
- 005
- 006
- 007
- 008
- 009
- 010
- 011
- 012
- 013
- 014
- 015
- 016
- 017
- 018
- 019
- 020
- 021
- 022
- 023
- 024
- 025
- 026
- 027
- 028
- 029
- 030
- 031
- 032
- 033
- 034
- 035
- 036
- 037
- 038
- 039
- 040
- 041
- 042
- 043
- 044
- 045
- 046
- 047
- 048
- 049
- 050
- 051
- 052
- 053
- 054

###### Submission and Formatting Instructions for International Conference on Machine Learning (ICML 2026)

###### Anonymous Authors1

Require: query x, actions A, discount factors Dω(·), and estimated prior pˆ(Z | x) t → 0 while not done do

- 000
- 001
- 002
- 003
- 004
- 005
- 006
- 007
- 008
- 009
- 010
- 011
- 012
- 013
- 014
- 015
- 016
- 017
- 018
- 019
- 020
- 021
- 022
- 023
- 024
- 025
- 026
- 027
- 028
- 029
- 030
- 031
- 032
- 033
- 034
- 035
- 036
- 037
- 038
- 039
- 040
- 041
- 042
- 043
- 044
- 045
- 046
- 047
- 048
- 049
- 050
- 051
- 052
- 053
- 054

at+1 ↑ ω(· | x,A,Dω(·),o0:t,pˆ(Z | x)) if at+1.type = COMMIT then

###### Submission and Formatting Instructions for International Conference on Machine Learning (ICML 2026)

###### Return at+1

else Execute action at+1 in the environment Receive observation ot+1

###### Baseline: Prompted/RLed agents Calibrate-Then-Act First, estimate priors

p(̂ Z ∣ x)

[Figure 11]

end if end while

[Figure 12]

[Figure 13]

###### Anonymous Authors1

[Figure 14]

Require: query x, actions A, discount factors Dω(·) t → 0 while not done do

Require: query x, actions A, discount factors Dω(·), and estimated prior pˆ(Z | x) t → 0 while not done do

Prompted

estimated priors p(̂ Z|x)

[Figure 15]

CTA-Prompted

at+1 ↑ ω(· | x,A,Dω(·),o0:t) if at+1.type = COMMIT then

RL

[Figure 16]

p(̂ Z|x)

RL

[Figure 17]

###### Return at+1

[Figure 18]

at+1 ↑ ω(· | x,A,Dω(·),o0:t,pˆ(Z | x)) if at+1.type = COMMIT then

else Execute action at+1 in the environment Receive observation ot+1

[Figure 19]

[Figure 20]

###### Return at+1

else Execute action at+1 in the environment Receive observation ot+1

end if end while

RL

CTA-RL

end if end while

Learns action selection implicitly from internal conﬀidence (Prompted) or training statistics (RL)

CTA: Enables explicitly reasoning over prior probabilities in the decision making process

Require: query x, actions A, discount factors Dω(·) t → 0 while not done do

- Figure 2: Standard agentic decision loop (left) and proposed method CTA (right). In CTA, we learn a prior estimator from training data and condition the agent on estimated pˆat inference and/or training time, inducing more optimal decision making through explicit reasoning over prior probabilities.

at+1 ↑ ω(· | x,A,Dω(·),o0:t) if at+1.type = COMMIT then

###### Return at+1

else Execute action at+1 in the environment Receive observation ot+1

whether to continue exploring or to commit based on its current partial information. The value of additional information depends on reasoning over current beliefs about the underlying world state via the prior p(z | x) and posterior bt.

end if end while

We define our LLM agent as π(at | x,A,Dθ(·),o0:t), placing a distribution over the next action in a given state. Figure 2 (left side) shows the basic form of this agent. π can be implemented either via a prompted LLM or through a model trained with reinforcement learning. However, in practice, it is difficult for π to learn to do the right reasoning in the environments we evaluate on.

1

Our key methodological contribution is to explicitly provide estimates of the prior, denoted as pˆ(Z | x). Figure 2 (right side) illustrates the role of a prior estimate in the agentic decision loop. We instantiate two versions of this approach.

- • CTA-Prompted: In this setting, we use the same agent π as the baseline, but with a modified prompt to include the prior information pˆ(Z | x) together with x. Examples of the prompt are given in Appendix J.
- • CTA-RL: In this setting, we use the prompt of CTA-Prompted and perform additional training via RL. We fine-tune the model end-to-end using GRPO [47] with the discounted reward objective from the task at hand. We evaluate it on new examples with the same prompt used at training time.

1

Both settings rely on the ability of a thinking LLM to determine how to use this information. In Section 6.1, we show that the Pandora’s Box problem is solvable by thinking models given the appropriate priors. In Section 6.2 and Figure 3, we then show that thinking models also perform substantially better on our tasks of interest.

Having accurate prior estimates is important for CTA to do well. We now describe how to estimate the priors described in Table 1. For Pandora’s Box, the priors are available explicitly, but for our other two problems, estimation is needed. Our two problems require two different ways of estimating these priors. However, a commonality is that both estimation techniques draw inferences about the problem that are not easily learned from end-to-end RL, thereby motivating the need for CTA.

#### 4.1 Estimating Priors from Model Confidence (QA)

In QA, the true kda(x), or the probability that the model can answer x correctly without retrieval, is not directly observable. This problem is one of confidence estimation. There are several ways to obtain estimates of model confidence, including inspecting logits, probe-based methods, and verbalized confidence. In this work, we use verbalized confidence [40], which is a general method that can apply to both blackbox and glassbox model usage. Given a question x, we prompt the model to produce a discrete verbalized confidence value pv(x), and apply an isotonic regression model [64] ISO trained on the validation set to obtain a calibrated estimate

kˆda(x) = ISO pv(x) .

The expected calibration error (ECE) reduces from 0.618 to 0.029 on the PopQA dataset [38] (described in Section 5) with this regression. This drop is reflective of initial poor calibration

[20, 59, 48, 53] and demonstrates that rescaling can help [13]. This kind of recalibration is very hard for the agent to learn with approaches like end-to-end RL: if RLed for a long time on a QA dataset, a model is more likely to memorize specific facts than it is to systematically recalibrate its beliefs.

#### 4.2 Estimating Priors from Training Data (FILEREADING)

Coding agents bring to bear many kinds of priors when completing tasks. Our FILEREADING task studies inferences about file formats as a controlled representative problem. In our setting, we assume that the agent is trained on our task, and therefore both baseline RL and the CTA methods are exposed to data showing the mapping of filenames to format information.

For prior estimation, we train a filename-to-format predictor, denoted as MBERT, to estimate the distribution p(z | n) from the filename. The predictor is based on a lightweight BERT-tiny encoder (4.4M parameters) [5, 51]. Given a filename n, the model encodes the tokenized string and uses the [CLS] representation to produce three independent categorical distributions via linear heads: delimiter, quote character, and skiprows. The model is trained with a summed cross-entropy objective across the three heads for one epoch on the training split. On the validation split, MBERT achieves an average classification accuracy of 67% across the three formatting attributes. This accuracy is expected: the ground-truth format values are sampled from a soft filename-dependent prior whose most likely option has probability about 67% on average. Thus, the task is intentionally stochastic rather than deterministically predictable from the filename. Appendix C shows that CTA performs worse as this predictor gets worse, but in a smooth fashion, with noisy predictors still giving benefit.

After training, MBERT outputs marginal probabilities {p(zd | n),p(zq | n),p(zs | n)}, which are provided to the agent associated with each task during RL training or test time, thereby decoupling uncertainty estimation from action selection.

### 5 Experimental Setup

Datasets For QA, we evaluate on PopQA [38], a QA benchmark that covers long-tail factual knowledge and benefits from retrieval. We sample 1,000 questions for evaluation and build the retriever based on Contriever [26, 3]. For each question, we sample a discount factor γ ∼ U[0.1,0.65] to study model behavior across varying retrieval costs.

Our coding problem involves a CSV-based question-answering dataset called FILEREADING, where filename cues provide informative signals about file formats and correct parsing requires executing code with appropriate format values. At test time, the true file format is hidden and only the filename is provided as part of the task query x. FILEREADING contains 2,000 tasks, split into 1,400 training, 300 validation, and 300 test examples. For each task, we randomly sample a unit-test discount du from [0.5,1] and duplicate the instance across four code discount settings dc = dρu, with ρ ∈ {0.5,1.0,2.0,4.0}, varying the relative cost of code execution while holding the task fixed. Details of the dataset construction are provided in Appendix G.

Metrics We evaluate the model performance across three sets of metrics. (1) Exploration statistics: for QA, we report retrieval rate Retrieve% (the fraction of questions for which retrieval is invoked), and for FILEREADING, we report the number of unit tests U and code attempts C. (2) Task accuracy, which measures whether the final model output matches the ground-truth answer for a task query. (3) Reward, which discounts correctness with exploration costs.

Models and Baselines We use Qwen3-8B [43] as the base model for both the QA and FILEREADING tasks. We compare the following baselines against our methods CTA-PROMPTED and CTA-RL.

- • PROMPTED: We prompt the base model directly with task description and query x.
- • PROMPTED-NONTHINK : Similar to PROMPTED, but with thinking mode disabled by prepending <think></think> tags without thinking content. Unless specified with -NONTHINK, we enable thinking mode by default for all other settings.
- • RL: We fine-tune the model end-to-end using GRPO [47] with discounted reward objective, and evaluate it by prompting with the task description and query x.

Oracle-Match Rate

Retrieve

Direct Answer

| |
|---|

| |
|---|

Prompted-NonThink

Prompted

CTA-Prompted (Ours)

100

| | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | |41%| | |70%| | |78%| | |76%| | |77%| |
| | | | | | | | | | | | | | | | | | | |
| | |27%| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | |70%| | |72%| | |66%| | |72%| |
| | |56%| | |64%| | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |

| | |89%| | |91%| | |92%| | |98%| | |99%| | |97%| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |

Percentage(%)

75

50

25

0

0.10 0.20 0.30 0.40 0.50 0.60 0.70

0.10 0.20 0.30 0.40 0.50 0.60 0.70

0.10 0.20 0.30 0.40 0.50 0.60 0.70

Discount Factor ( ) bin

Discount Factor ( ) bin

Discount Factor ( ) bin

- Figure 3: Model retrieval decision across retrieval discount factors γ on the QA task. Red bars show the percentage of examples where the model retrieves before answering; green bars show direct answers. The black line reports the oracle policy match rate. CTA-PROMPTED makes adaptive cost-sensitive decisions and achieves highest oracle match rates across settings.

To expose both RL settings to diverse cost trade-offs in FILEREADING, we duplicate each training instance across multiple relative cost values ρ ∈ {0.5,1.0,2.0,4.0}, yielding a 4× larger RL training set than that used for training MBERT. More training details are in Appendix I.

### 6 Results

#### 6.1 LLMs can solve Pandora’s Box problems

We first evaluate whether LLM agents can solve the proof-of-concept Pandora’s Box task introduced in Section 3.1. We instantiate the agent using the framework in Figure 2, with prompts in Figure 9, and use Qwen38B as the base model. We evaluate on 100 examples with K = 3, discount factors sampled from [0,0.1,...,1.0], and priors sampled from a symmetric Dirichlet distribution with concentration parameter α = 0.5.

Table 2 reports the performance with two metrics: average reward and optimal policy match rate, measuring whether the model’s interaction trace aligns with the oracle strategy described in Appendix Algorithm 1. Without explicit priors or when the thinking mode is disabled, the agent exhibits low optimal match rates, indicating failure to recover the optimal decision rule. In contrast, CTAPROMPTED achieves 94.0% optimal match rate and significantly higher reward, indicating that the model is capable of reasoning about optimal exploration-commitment tradeoffs and adapting to different discount factors and prior distributions when the environment constraints are made explicit. Appendix E compares interaction traces across settings, showing how explicit reasoning in thinking traces is used. Crucially, the presence of explicit prior information triggers the model to reason about cost-accuracy tradeoffs, from which it can make optimal decisions.

#### 6.2 Calibrate-Then-Act enables models to make calibrated decisions

Table 2: Performance comparison on the Pandora’s Box task. A thinking model given access to priors can solve the problem near-optimally.

Method Optimal Match Rate (%) Avg. Reward Oracle policy 100.0 0.649 PROMPTED-NONTHINK 11.0 0.441 PROMPTED 23.0 0.476 CTA-PROMPTED-NONTHINK 20.0 0.436 CTA-PROMPTED 94.0 0.625

Prompted

CTA-Prompted

R=0.305

R=0.258

R=0.213

R=0.138

| |R=0.329 R=0.282 R=0.211 R=0.137| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | |37.3%| | |34.0%| | |33.0%| | |24.3%| | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |

100%

75%

50%

25%

16.3%

12.7%

11.3%

9.0%

0%

0.5 1.0 2.0 4.0

0.5 1.0 2.0 4.0

RL

CTA-RL

| |R=0.359 R=0.303 R=0.227 R=0.148| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | | | | | | | | | | | | | |
| | |0.0%| | |0.0%| | |0.0%| | |0.0%| | |
| | | | | | | | | | | | | | |

| |R=0.382 R=0.315 R=0.230 R=0.147| | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | |16.0%| |12.0%| | | |8.0%| |
| | | | | | | | | | |
| | | | | | |5.7%| | | |

100%

75%

50%

25%

0%

0.5 1.0 2.0 4.0

0.5 1.0 2.0 4.0

CODE UNIT CODE ANS CODE ANS CODE UNIT CODE UNIT CODE ANS

UNIT CODE ANS UNIT CODE UNIT CODE ANS Other

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Figure 4: Action distribution for agents on FILEREADING across relative cost parameters ρ. Each stacked bar shows the proportion of decision traces corresponding to different action patterns, with the reward R labeled above. Annotated percentages indicate the fraction of tasks where the agent attempts code execution before any unit tests.

Baseline LLM agents struggle to make calibrated decisions, even after RL. Across both QA and FILEREADING, we find that without explicit priors, LLMs do not naturally form calibrated decision rules from task descriptions alone.

- Table 3: Performance on QA. Our principal metric is reward. One-turn baselines use fixed strategies, while multi-turn agents adaptively decide when to retrieve. CTA-PROMPTED achieves the highest reward.

Method Retrieve % Acc. Reward Single-turn baselines

NEVER RETRIEVE 0.0 0.226 0.226 ALWAYS RETRIEVE 100.0 0.578 0.213

Multi-turn agents

PROMPTED-NONTHINK 97.7 0.619 0.244 PROMPTED 61.4 0.501 0.283 CTA-PROMPTED (Ours) 65.3 0.512 0.293

Table 4: Performance on FILEREADING, averaged across relative unit-test and code-execution cost ratios ρ ∈ {0.5,1.0,2.0,4.0}. We report the average number of turns, unit-test calls (U), code executions (C), accuracy, and discounted reward.

Method # Turns U C Acc. Reward Without Training

PROMPTED 3.62 2.67 1.42 0.958 0.229 CTA-PROMPTED (Ours) 3.47 2.51 1.41 0.945 0.240

With RL Training

RL 3.51 2.13 1.39 0.997 0.259 CTA-RL (Ours) 3.46 1.98 1.46 0.991 0.268

For QA, Figure 3 visualizes the retrieval behavior across retrieval discount factors γ. Each stacked bar shows the model’s decisions for problems within a discount factor bin: red denotes the percentage of examples where the model retrieves before answering, while green denotes direct answers. The black line reports oracle match rate, i.e., the fraction of decisions that match the optimal retrieve-or-not rule defined by the threshold kda = γ · kret. Specifically, PROMPTED-NONTHINK (Figure 3, left) retrieves almost always in all cost regimes, reflecting a conservative and cost-insensitive strategy.

Enabling thinking reduces unnecessary retrieval, but PROMPTED (Figure 3, mid) adapts only weakly to retrieval costs and achieves modest oracle-match rates. Although PROMPTED sometimes reasons qualitatively about its parametric knowledge, it rarely quantifies these as certainty values for an expected-value comparison. Similarly, it sometimes mentions costs, but does not consistently tie the final decision to the costs. As a result, its decisions often fall back to a recall-based heuristic: answer if a plausible candidate comes to mind, retrieve otherwise, rather than systematically adapting to γ and confidence. This suggests that, without explicit prior estimates, the models struggle to assess their own parametric knowledge and make consistent retrieval decisions.

On FILEREADING, Figure 4 shows a similar analysis. The x-axis ρ = log dc/log du reflects relative cost: as ρ increases, code attempts become relatively more costly, and a rational agent will tend to test more before coding. Each stacked bar represents the collection of tasks with a specific ρ, and each color represents the proportion of action trace pattern in the collection. The reward is labeled above each bar, and the percentage of “guess-and-go” x% (attempting code without any preceding unit tests) are also labeled.

Both PROMPTED and RL collapse to a static test-first policy rather than adapting across cost regimes, with 0% of “code-first” traces (attempts to execute code before any unit tests). RL without prior cannot internalize the structure of the training data and instead defaults to a suboptimal exploration policy. Together, these results show that LLMs struggle to make calibrated, cost-sensitive decisions when prior estimates are implicit.

CTA pushes the model towards adaptive, optimal decision making in both task settings. Our main results are given in Tables 3 and 4. Table 3 compares QA performance across fixed retrieval strategies and various multi-turn settings. CTA-PROMPTED achieves the highest reward, outperforming both PROMPTED-NONTHINK, which retrieves too often (97.7%), and PROMPTED, which reduces retrieval but does not align as well with the oracle policy. Figure 3 shows that CTA-Prompted is much more adaptive in its behavior than baselines: it retrieves much more frequently at high discount factors and less at low discount factors. This is also reflected in it obtaining the highest oracle match rate across cost constraints.

- Table 4 shows the model performance on FILEREADING aggregated across different cost settings. Note that both RL and CTA-RL have access to the same set of training data which encodes correlations between filenames and file formats implicitly; CTA-PROMPTED uses more information than PROMPTED by virtue of the trained model. The agent achieves a discounted reward of 0.259 by training end-to-end (RL) with the discounted reward as objective, while conditioning the training on explicit estimated priors (CTA-RL) further improves by 3.5% with an overall reward of 0.268. This shows that incorporating estimated priors helps the model generalize better on unseen test data than end-to-end RL.

- Figure 4 shows that while end-to-end RL improves reward over its non-training counterpart PROMPTED, its decision-making behavior collapses to a static policy which always triggers unit tests before the first code attempt (0% “guess-and-go”). In contrast, CTA-PROMPTED, which conditions the agentic decision-making on estimated priors, already exhibits adaptive behavior in response to costs by acting more conservatively with higher ρ. After training, this adaptivity remains pronounced in CTA-RL.

0.5 1.0 2.0 4.0

= logdc /logdu

0.02

0.00

0.02

0.04

0.06

0.08

Rewardover

Baseline(3*TestCode)

Baseline: Test Test Test Code

Baseline (w/ Bert Prior): Code first

RL

CTA-RL

Figure 5: Average reward under varying costs for our approaches. Static strategies (test-first or code-first) achieve high reward only in limited regimes, whereas CTA-RL with estimated priors is consistently on the Pareto frontier

- Figure 5 visualizes the ∆ Reward against the naive baseline “3*Tests→Code” for each method. While RL only shows advantage with large ρ, and static policy such as Code first only performs well with small ρ, our method CTA-RL stays at the Pareto frontier across ρ values (significance tests in Appendix D). This suggests that conditioning training with explicit priors effectively reinforces the adaptive decision reasoning.

Even with imperfect prior estimates, CTA-RL consistently performs better than RL baseline across cost regimes. Appendix C further analyzes sensitivity to prior estimator quality. Appendix K provides representative model traces comparing RL and CTA-RL, illustrating how CTA induces more cost-aware decisions.

### 7 Related Work

Decision making under incomplete information LLMs are increasingly being applied to tasks with incomplete information, arising from underspecified user queries [10, 66, 65, 32, 46], ambiguity [39, 9, 12], and partially observed environments [57, 34, 14, 7, 17]. To resolve uncertainty, models often need to ask clarifying questions [44, 23, 31], query the environments [6, 42, 41], or engage in collaboration [58, 8]. While prior work has typically focused on training or prompting strategies, we show that models can abstractly reason about the optimal solution when provided with explicit priors, which we use to induce such reasoning.

Agents in cost-aware deployment LLM-based agents are increasingly deployed in real-world settings that require multi-step reasoning and tool use, including interactive coding [50, 68, 54, 27], planning [69, 36], question answering [63, 15], and scientific research [45, 22, 30, 1, 2]. While tool use expands agents’ capabilities and reliability, interacting with external environments often incurs latency [18], resources [11, 37], and overhead that can negatively affect user experience [16, 25]. In response, several lines of work have emerged to study agent behavior under explicit cost constraints. Liu et al. [36] introduce a cost-centric benchmark for evaluating agents’ tool-planning abilities. Wang et al. [52], Gul et al. [19], Wang et al. [53], Lin et al. [35] study how models can reduce unnecessary retrieval or tool use while maintaining answer quality, for example through abstention, selective search, or efficiency-oriented action policies. Berant et al. [4] train steerable clarification policies that adapt to cost coefficients. An underexplored aspect of efficient exploration is the joint treatment of uncertainty priors and cost constraints, which together determine Pareto-optimal decisions. We propose a unified framework for interactive agentic tasks and show that calibrated priors are key to inducing appropriate decision-making in LLM agents.

### 8 Conclusion

This paper presents a method for having LLMs balance uncertainty-cost tradeoffs in their environment interaction. By presenting an LLM with priors over unobserved features of the environment, the LLM can successfully reason about Pareto-optimal behavior and navigate action costs effectively. This work illustrates new ways of inducing agents to think optimally, and suggests that meta-level information (priors about capabilities) may have a role to play in shaping agent policies.

### Acknowledgments and Disclosure of Funding

Thanks to Xi Ye for comments on a draft of this work. This work was supported by the NSF under Cooperative Agreement 2421782 and the Simons Foundation grant MPS-AI-00010515 awarded to the NSF-Simons AI Institute for Cosmic Origins — CosmicAI, https://www.cosmicai.org/. This was was also partially supported by NSF CAREER Award IIS-2145280, NSF grant IIS-2433071, by the Sloan Foundation, and by grants from Amazon and Open Philanthropy. This research has been supported by computing support on the Vista GPU Cluster through the Center for Generative AI (CGAI) and the Texas Advanced Computing Center (TACC) at the University of Texas at Austin, through the Torch cluster at NYU, and through a compute grant from NVIDIA.

### References

- [1] Amirhossein Abaskohi, Tianyi Chen, Miguel Muñoz-Mármol, Curtis Fox, Amrutha Varshini Ramesh, Étienne Marcotte, Xing Han Lù, Nicolas Chapados, Spandana Gella, Christopher Pal, Alexandre Drouin, and Issam H. Laradji. DRBench: A realistic benchmark for enterprise deep research. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=IGYQ4c92e2.
- [2] Dhruv Agarwal, Bodhisattwa Prasad Majumder, Reece Adamson, Megha Chakravorty, Satvika Reddy Gavireddy, Aditya Parashar, Harshit Surana, Bhavana Dalvi Mishra, Andrew McCallum, Ashish Sabharwal, et al. Open-ended Scientific Discovery via Bayesian Surprise. arXiv preprint arXiv:2507.00310, 2025.
- [3] Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, et al. MS MARCO: A human generated machine reading comprehension dataset. arXiv preprint arXiv:1611.09268, 2016.
- [4] Jonathan Berant, Maximillian Chen, Adam Fisch, Reza Aghajani, Fantine Huot, Mirella Lapata, and Jacob Eisenstein. Learning steerable clarification policies with collaborative self-play. arXiv preprint arXiv:2512.04068, 2025.
- [5] Prajjwal Bhargava, Aleksandr Drozd, and Anna Rogers. Generalization in NLI: Ways (Not) To Go Beyond Simple Heuristics, 2021.
- [6] Moses Charikar, Ronald Fagin, Venkatesan Guruswami, Jon Kleinberg, Prabhakar Raghavan, and Amit Sahai. Query strategies for priced information. Journal of Computer and System Sciences, 64(4):785–819, 2002. ISSN 0022-0000. doi: https://doi.org/10.1006/jcss.2002.1828. URL https://www.sciencedirect.com/science/article/pii/S0022000002918283.
- [7] Sanxing Chen, Xiaoyin Chen, Yukun Huang, Roy Xie, and Bhuwan Dhingra. When greedy wins: Emergent exploitation bias in meta-bandit llm training. ArXiv, abs/2509.24923, 2025. URL https://api.semanticscholar.org/CorpusID:281674231.
- [8] Weize Chen, Jiarui Yuan, Chen Qian, Cheng Yang, Zhiyuan Liu, and Maosong Sun. Optima: Optimizing effectiveness and efficiency for LLM-based multi-agent system. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Findings of the Association for Computational Linguistics: ACL 2025, pages 11534–11557, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/ v1/2025.findings-acl.601. URL https://aclanthology.org/2025.findings-acl.601/.
- [9] Jio Choi, Mohit Bansal, and Elias Stengel-Eskin. Language models identify ambiguities and exploit loopholes. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 32991–33006, 2025.
- [10] Jeremy R. Cole, Michael JQ Zhang, Daniel Gillick, Julian Martin Eisenschlos, Bhuwan Dhingra, and Jacob Eisenstein. Selectively answering ambiguous questions. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023. URL https://openreview. net/forum?id=x2W2dKdNI8.
- [11] Mehul Damani, Idan Shenfeld, Andi Peng, Andreea Bobu, and Jacob Andreas. Learning how hard to think: Input-adaptive allocation of lm computation. ArXiv, abs/2410.04707, 2024. URL https://api.semanticscholar.org/CorpusID:273186996.

- [12] Mingyi Deng, Lijun Huang, Yani Fan, Jiayi Zhang, Fashen Ren, Jinyi Bai, Fuzhen Yang, Dayi Miao, Zhaoyang Yu, Yifan Wu, Yanfei Zhang, Fengwei Teng, Yingjia Wan, Song Hu, Yude Li, Xin Jin, Conghao Hu, Haoyu Li, Qirui Fu, Tai Zhong, Xinyu Wang, Xiangru Tang, Nan Tang, Chenglin Wu, and Yuyu Luo. InteractComp: Evaluating Search Agents With Ambiguous Queries. ArXiv, abs/2510.24668, 2025. URL https://api.semanticscholar. org/CorpusID:282401680.
- [13] Shrey Desai and Greg Durrett. Calibration of pre-trained transformers. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu, editors, Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 295–302, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.21. URL https://aclanthology.org/2020.emnlp-main.21/.
- [14] Vikranth Dwaracherla, Seyed Mohammad Asghari, Botao Hao, and Benjamin Van Roy. Efficient exploration for LLMs. arXiv preprint arXiv:2402.00396, 2024.
- [15] Jacob Eisenstein, Reza Aghajani, Adam Fisch, Dheeru Dua, Fantine Huot, Mirella Lapata, Vicky Zayats, and Jonathan Berant. Don’t lie to your friends: Learning what you know from collaborative self-play. In Second Conference on Language Modeling, 2025. URL https://openreview.net/forum?id=2vDJiGUfhV.
- [16] Morad Elfleet and Mathieu Chollet. Investigating the Impact of Multimodal Feedback on UserPerceived Latency and Immersion with LLM-Powered Embodied Conversational Agents in Virtual Reality. In IVA, pages 12:1–12:9, 2024. URL https://doi.org/10.1145/3652988. 3673965.
- [17] Gabriel Grand, Valerio Pepe, Jacob Andreas, and Joshua B Tenenbaum. Shoot First, Ask Questions Later? Building Rational Agents that Explore and Act Like People. arXiv preprint arXiv:2510.20886, 2025.
- [18] Yilin Guan, Qingfeng Lan, Sun Fei, Dujian Ding, Devang Acharya, Chi Wang, William Yang Wang, and Wenyue Hua. Dynamic speculative agent planning. arXiv preprint arXiv:2509.01920, 2025.
- [19] Mustafa Omer Gul, Claire Cardie, and Tanya Goyal. Pay-per-search models are abstention models. arXiv preprint arXiv:2510.01152, 2025.
- [20] Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q Weinberger. On calibration of modern neural networks. In International conference on machine learning, pages 1321–1330. PMLR, 2017.
- [21] Rushil Gupta, Jason Hartford, and Bang Liu. LLMs for experiment design in scientific domains: Are we there yet? In ICML 2025 Generative AI and Biology (GenBio) Workshop, 2025. URL https://openreview.net/forum?id=dIEeOwrmOe.
- [22] Anthony GX-Chen, Dongyan Lin, Mandana Samiei, Doina Precup, Blake A. Richards, Rob Fergus, and Kenneth Marino. Language agents mirror human causal reasoning biases. how can we help them think like scientists? ArXiv, abs/2505.09614, 2025. URL https://api. semanticscholar.org/CorpusID:278602122.
- [23] Kunal Handa, Yarin Gal, Ellie Pavlick, Noah Goodman, Jacob Andreas, Alex Tamkin, and Belinda Z Li. Bayesian preference elicitation with language models. arXiv preprint

- arXiv:2403.05534, 2024.

[24] Leona Hennig, Tanja Tornede, and Marius Lindauer. Towards leveraging AutoML for sustainable deep learning: A multi-objective HPO approach on deep shift neural networks. arXiv preprint

- arXiv:2404.01965, 2024.

- [25] Christine Herlihy, Jennifer Neville, Tobias Schnabel, and Adith Swaminathan. On Overcoming Miscalibrated Conversational Priors in LLM-based ChatBots. In Uncertainty in Artificial Intelligence, pages 1599–1620. PMLR, 2024.

- [26] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning. Transactions on Machine Learning Research, 2022. ISSN 2835-8856. URL https://openreview.net/forum?id=jKN1pXi7b0.
- [27] Arnav Kumar Jain, Gonzalo Gonzalez-Pumariega, Wayne Chen, Alexander M Rush, Wenting Zhao, and Sanjiban Choudhury. Multi-turn code generation through single-step rewards. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=aJeLhLcsh0.
- [28] Shihao Ji and Lawrence Carin. Cost-sensitive feature acquisition and classification. Pattern Recognition, 40(5):1474–1485, 2007.
- [29] Kimmo Kärkkäinen, Mohammad Kachuee, Orpaz Goldstein, and Majid Sarrafzadeh. Costsensitive feature-value acquisition using feature relevance. arXiv preprint arXiv:1912.08281, 2019.
- [30] Zaid Khan, Archiki Prasad, Elias Stengel-Eskin, Jaemin Cho, and Mohit Bansal. One life to learn: Inferring symbolic world models for stochastic environments from unguided exploration. ArXiv, abs/2510.12088, 2025. URL https://api.semanticscholar.org/CorpusID: 282064346.
- [31] Harsh Nishant Lalai, Raj Sanjay Shah, Jiaxin Pei, Sashank Varma, Yi-Chia Wang, and Ali Emami. The world according to LLMs: How geographic origin influences LLMs’ entity deduction capabilities. In Second Conference on Language Modeling, 2025. URL https: //openreview.net/forum?id=hJtvCfDfs1.
- [32] Belinda Z. Li, Been Kim, and Zi Wang. Questbench: Can LLMs ask the right question to acquire information in reasoning tasks? In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025. URL https://openreview.net/ forum?id=gpwA9aZLTZ.
- [33] Yang Li and Junier Oliva. Towards cost sensitive decision making. In Yingzhen Li, Stephan Mandt, Shipra Agrawal, and Emtiyaz Khan, editors, Proceedings of The 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pages 3601–3609. PMLR, 03–05 May 2025. URL https://proceedings. mlr.press/v258/li25h.html.
- [34] Jessy Lin, Nicholas Tomlin, Jacob Andreas, and Jason Eisner. Decision-oriented dialogue for human-AI collaboration. Transactions of the Association for Computational Linguistics, 12:892–911, 2024. doi: 10.1162/tacl_a_00679. URL https://aclanthology.org/2024. tacl-1.50/.
- [35] Tzu-Han Lin, Wei-Lin Chen, Chen-An Li, Hung yi Lee, Yun-Nung Chen, and Yu Meng. AdaSearch: Balancing Parametric Knowledge and Search in Large Language Models via Reinforcement Learning. ArXiv, abs/2512.16883, 2025. URL https://api.semanticscholar. org/CorpusID:283933928.
- [36] Jiayu Liu, Cheng Qian, Zhaochen Su, Qing Zong, Shijue Huang, Bingxiang He, and Yi R Fung. Costbench: Evaluating multi-turn cost-optimal planning and adaptation in dynamic environments for llm tool-use agents. arXiv preprint arXiv:2511.02734, 2025.
- [37] Tengxiao Liu, Zifeng Wang, Jin Miao, I Hsu, Jun Yan, Jiefeng Chen, Rujun Han, Fangyuan Xu, Yanfei Chen, Ke Jiang, et al. Budget-aware tool-use enables effective agent scaling. arXiv preprint arXiv:2511.17006, 2025.
- [38] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9802–9822, Toronto, Canada, July

2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.546. URL https://aclanthology.org/2023.acl-long.546/.

- [39] Sewon Min, Julian Michael, Hannaneh Hajishirzi, and Luke Zettlemoyer. AmbigQA: Answering ambiguous open-domain questions. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu, editors, Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 5783–5797, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main.466. URL https://aclanthology.org/2020.emnlp-main.466/.
- [40] Christopher Mohri and Tatsunori Hashimoto. Language models with conformal factuality guarantees. In Proceedings of the 41st International Conference on Machine Learning, pages 36029–36047, 2024.
- [41] Giovanni Monea, Antoine Bosselut, Kianté Brantley, and Yoav Artzi. LLMs Are In-Context Bandit Reinforcement Learners. arXiv preprint arXiv:2410.05362, 2024.
- [42] Shivam Nadimpalli, Mingda Qiao, and Ronitt Rubinfeld. No price tags? no problem: Query strategies for unpriced information. ArXiv, abs/2511.06170, 2025. URL https: //api.semanticscholar.org/CorpusID:282911938.
- [43] Qwen Team. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.
- [44] Sudha Rao and Hal Daumé III. Learning to ask good questions: Ranking clarification questions using neural expected value of perfect information. In Iryna Gurevych and Yusuke Miyao, editors, Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2737–2746, Melbourne, Australia, July 2018. Association for Computational Linguistics. doi: 10.18653/v1/P18-1255. URL https://aclanthology.org/P18-1255/.
- [45] Sarah Schwettmann, Tamar Shaham, Joanna Materzynska, Neil Chowdhury, Shuang Li, Jacob Andreas, David Bau, and Antonio Torralba. FIND: A Function Description Benchmark for Evaluating Interpretability Methods. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 75688–75715. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper_files/paper/2023/file/ ef0164c1112f56246224af540857348f-Paper-Datasets_and_Benchmarks.pdf.
- [46] Omar Shaikh, Hussein Mozannar, Gagan Bansal, Adam Fourney, and Eric Horvitz. Navigating rifts in human-LLM grounding: Study and benchmark. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 20832–20847, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.1016. URL https://aclanthology. org/2025.acl-long.1016/.
- [47] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [48] Yuanhao Shen, Xiaodan Zhu, and Lei Chen. SMARTCAL: An approach to self-aware tool-use evaluation and calibration. arXiv preprint arXiv:2412.12151, 2024.
- [49] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.
- [50] Hao Tang, Keya Hu, Jin Peng Zhou, Sicheng Zhong, Wei-Long Zheng, Xujie Si, and Kevin Ellis. Code Repair with LLMs gives an Exploration-Exploitation Tradeoff. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 117954–117996. Curran Associates, Inc., 2024. doi: 10.52202/ 079017-3746. URL https://proceedings.neurips.cc/paper_files/paper/2024/ file/d5c56ec4f69c9a473089b16000d3f8cd-Paper-Conference.pdf.

- [51] Iulia Turc, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Well-read students learn better: The impact of student initialization on knowledge distillation. CoRR, abs/1908.08962,

2019. URL http://arxiv.org/abs/1908.08962.

- [52] Hongru Wang, Cheng Qian, Wanjun Zhong, Xiusi Chen, Jiahao Qiu, Shijue Huang, Bowen Jin, Mengdi Wang, Kam-Fai Wong, and Heng Ji. Acting Less is Reasoning More! Teaching Model to Act Efficiently. arXiv preprint arXiv:2504.14870, 2025.
- [53] Hongru Wang, Boyang Xue, Baohang Zhou, Tianhua Zhang, Cunxiang Wang, Huimin Wang, Guanhua Chen, and Kam-Fai Wong. Self-DC: When to Reason and When to Act? Self Divideand-Conquer for Compositional Unknown Questions. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 6510–6525, 2025.
- [54] Yunkun Wang, Yue Zhang, Zhen Qin, Chen Zhi, Binhua Li, Fei Huang, Yongbin Li, and Shuiguang Deng. ExploraCoder: Advancing code generation for multiple unseen APIs via planning and chained exploration. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 18124–18145, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10. 18653/v1/2025.acl-long.887. URL https://aclanthology.org/2025.acl-long.887/.
- [55] Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, Eli Gottlieb, Monica Lam, Yiping Lu, Kyunghyun Cho, Jiajun Wu, Fei-Fei Li, Lijuan Wang, Yejin Choi, and Manling Li. RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning. ArXiv, abs/2504.20073,

2025. URL https://api.semanticscholar.org/CorpusID:278170861.

- [56] Martin L. Weitzman. Optimal search for the best alternative. Econometrica, 47(3):641–654,

1979. ISSN 00129682, 14680262. URL http://www.jstor.org/stable/1910412.

- [57] Li Siang Wong, Gabriel Grand, Alexander K. Lew, Noah D. Goodman, Vikash K. Mansinghka, Jacob Andreas, and Joshua B. Tenenbaum. From word models to world models: Translating from natural language to the probabilistic language of thought. ArXiv, abs/2306.12672, 2023. URL https://api.semanticscholar.org/CorpusID:259224900.
- [58] Shirley Wu, Michel Galley, Baolin Peng, Hao Cheng, Gavin Li, Yao Dou, Weixin Cai, James Zou, Jure Leskovec, and Jianfeng Gao. CollabLLM: From passive responders to active collaborators. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=DmH4HHVb3y.
- [59] Miao Xiong, Zhiyuan Hu, Xinyang Lu, YIFEI LI, Jie Fu, Junxian He, and Bryan Hooi. Can LLMs express their uncertainty? an empirical evaluation of confidence elicitation in LLMs. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=gjeQKFxFpZ.
- [60] Yuebin Xu, Zhiyi Chen, and Zeyi Wen. EcoTune: Token-efficient multi-fidelity hyperparameter optimization for large language model inference. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 7735–7745, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10. 18653/v1/2025.emnlp-main.394. URL https://aclanthology.org/2025.emnlp-main. 394/.
- [61] Hui Yang, Sifu Yue, and Yunzhong He. Auto-GPT for online decision making: Benchmarks and additional opinions. arXiv preprint arXiv:2306.02224, 2023.
- [62] John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang, editors, Advances in Neural Information Processing Systems, volume 37, pages 50528–50652. Curran Associates, Inc., 2024. doi: 10.52202/ 079017-1601. URL https://proceedings.neurips.cc/paper_files/paper/2024/ file/5a7c947568c1b1328ccc5230172e1e7c-Paper-Conference.pdf.

- [63] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. ReAct: Synergizing Reasoning and Acting in Language Models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview. net/forum?id=WE_vluYUL-X.
- [64] Bianca Zadrozny and Charles Elkan. Transforming classifier scores into accurate multiclass probability estimates. In Proceedings of the eighth ACM SIGKDD international conference on Knowledge discovery and data mining, pages 694–699, 2002.
- [65] Michael JQ Zhang and Eunsol Choi. Clarify when necessary: Resolving ambiguity through interaction with LMs. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors, Findings of the Association for Computational Linguistics: NAACL 2025, pages 5526–5543, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979-8-89176195-7. doi: 10.18653/v1/2025.findings-naacl.306. URL https://aclanthology.org/2025. findings-naacl.306/.
- [66] Michael JQ Zhang, W. Bradley Knox, and Eunsol Choi. Modeling future conversation turns to teach LLMs to ask clarifying questions. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=cwuSAR7EKd.
- [67] Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, and Yuwen Xiong. Easyr1: An efficient, scalable, multi-modality rl training framework, 2025. https: //github.com/hiyouga/EasyR1.
- [68] Lingfeng Zhou, Junhao Shi, Jin Gao, and Dequan Wang. Credit-budgeted ICPC-style coding: When LLM agents must pay for every decision. In NeurIPS 2025 Workshop on Bridging Language, Agent, and World Models for Reasoning and Planning, 2025. URL https:// openreview.net/forum?id=JEdeMSKvbT.
- [69] Yifei Zhou, Andrea Zanette, Jiayi Pan, Sergey Levine, and Aviral Kumar. ArCHer: Training language model agents via hierarchical multi-turn RL. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id=b6rA0kAHT1.

- Table 5: Sensitivity to prior quality in the FILEREADING task. Performance degrades smoothly as priors become less accurate, without a sharp failure transition.

Prior Prior Acc Avg Reward Task Acc #Code Rρ=0.5 Rρ=1.0 Rρ=2.0 Rρ=4.0

BERT 0.67 0.269 99.1% 1.46 0.382 0.315 0.230 0.147 Noisy 0.52 0.251 96.3% 1.71 0.365 0.295 0.206 0.136 Uniform 0.46 0.251 96.8% 1.53 0.349 0.300 0.219 0.134 Adversarial 0.33 0.203 94.4% 2.26 0.299 0.240 0.173 0.099

### A Limitations

Our experiments focus on controlled QA and FILEREADING settings that isolate cost-aware exploration, but they are simpler than open-ended agent deployments with richer action spaces, longer horizons, and noisier observations. We use these settings to enable reliable evaluation of decision quality, while extending CTA to more complex environments remains a promising direction for future work.

CTA also relies on the availability of useful prior estimates. Our sensitivity analysis shows that CTA is robust to moderate estimation error, but performance can degrade when priors are highly miscalibrated or confidently wrong. We demonstrate several ways to obtain priors without groundtruth access, including from model confidence and historical task data, but designing reliable prior estimators for broader domains remains future work.

Finally, CTA does not guarantee optimal behavior in arbitrary environments. The model must still correctly interpret the provided priors and reason about their implications for action selection. In more complex scenarios, optimal action selection may require stronger reasoning capabilities of the base model to account for posterior updates, multi-step value of information, and dependencies among multiple latent variables.

### B Impact Statement

This paper presents a method for broadly improving the cost-benefit tradeoffs of LLM agents. Although this is not yet integrated into production agent systems, we envision that this approach or one derived from it could be, which would ideally lead to cost savings and increased efficiency. We do not foresee specific drawbacks of this approach relative to other advancements in LLMs, agents, and machine learning more broadly. Potential broad drawbacks are inherited from the drawbacks of capability advancements for LLMs and agents more broadly, such as enabling the further propagation of AI technology in society.

### C Sensitivity to Prior Quality

We evaluate how performance varies with prior quality in the FILEREADING task by systematically degrading the prior estimator used by CTA-RL. In addition to the learned BERT-based prior, we construct three alternative priors with progressively worse calibration:

- • Noisy: Derived from the BERT prior by randomly selecting 40% of predictions and swapping the probabilities assigned to the correct and incorrect options.
- • Uniform: A uniform distribution over all candidate options, representing the absence of informative prior knowledge.
- • Adversarial: Constructed by swapping the probabilities assigned to correct and incorrect options and inflating the incorrect option to 95% confidence.

- Table 5 reports performance across different prior qualities. As prior quality decreases, both reward and task accuracy degrade gradually. Importantly, the degradation is smooth rather than exhibiting a sharp failure transition, indicating robustness to imperfect priors. This suggests that approximate or imperfectly estimated priors are sufficient to yield meaningful gains (even the BERT prior achieves only 67% accuracy), indicating that our approach is inherently error-tolerant and applicable in

- Table 6: Paired bootstrap tests for CTA-RL on FILEREADING. ∆ denotes the mean per-task reward gain of CTA-RL over each baseline. Tests use n = 300 paired tasks per ρ and B = 10,000 bootstrap resamples.

ρ vs. Test first (B1) vs. Code first (B2) vs. RL

- 0.5 +0.0845∗∗ +0.0149= +0.0224∗∗
- 1.0 +0.0508∗∗ +0.0059= +0.0117∗
- 2.0 +0.0169∗∗ +0.0153= +0.0030= 4.0 −0.0021= +0.0287∗∗ −0.0010=

∗p < 0.01, ∗∗p < 0.001, and = denotes no significant difference.

complex real-world settings where accurate priors are difficult to obtain due to noisy or highdimensional features. At the same time, the adversarial setting highlights that being confidently wrong is substantially more harmful than being uncertain, leading to the largest drop in performance.

### D Significance Testing for Pareto-Optimality in FILEREADING

We evaluate whether the gains of CTA-RL are statistically reliable using paired bootstrap tests on per-task rewards. For each cost setting ρ ∈ {0.5,1.0,2.0,4.0}, we pair the test examples across methods by task ID. We compute per-task reward differences between CTA-RL and each baseline, and use 10,000 bootstrap resamples to test whether the mean difference is significantly different from zero. Table 6 reports the observed mean reward gain ∆.

Overall, CTA-RL achieves significant gains over different baselines in several cost regimes and is never statistically worse than any baseline. These results support the conclusion that CTA-RL remains on or near the Pareto frontier across cost regimes.

CTA-RL significantly outperforms the no-prior test-first baseline at ρ ∈ {0.5,1.0,2.0} and is statistically indistinguishable from it at ρ = 4.0. Against the prior-aware code-first baseline, CTA-RL is statistically indistinguishable at lower ρ, where code-first already benefits from prior information, but significantly outperforms it at ρ = 4.0, where code execution becomes relatively expensive. Compared to the no-prior RL agent, CTA-RL achieves significant gains at ρ = 0.5 and ρ = 1.0 and is statistically indistinguishable at higher ρ. Overall, these results support the conclusion that CTA-RL remains on or near the Pareto frontier across cost regimes.

### E Qualitative Trace Analysis of Pandora’s Box Problem

We present representative interaction traces from three settings: CTA-PROMPTED-NONTHINK, PROMPTED, and CTA-PROMPTED. In CTA-PROMPTED-NONTHINK (with thinking mode disabled; Figure 6), the model does not appear to compare the expected value of additional information against the exploration cost. As a result, it tends to verify all options before committing, regardless of the prior distribution, leading to unnecessary exploration. In PROMPTED (Figure 7), the model does not have access to the prior probabilities. Lacking calibrated uncertainty information, it effectively operates under an implicit uniform prior and consequently follows a suboptimal verification strategy.

In contrast, in CTA-PROMPTED (Figure 8), the model is provided with prior probabilities and has thinking mode enabled. In this setting, it explicitly reasons about the trade-off between expected reward and exploration cost by comparing the value of immediate commitment with the discounted value of further verification. The resulting behavior aligns with the oracle policy.

These qualitative examples illustrate how explicit prior information induces the correct reasoning of the model over value of additional information against action cost and making the optimal decision accordingly.

### F Experiment Details for QA

For QA, we evaluate on the POPQA dataset. We build the retriever based on Contriever [26, 3]. The retriever quality pret, defined as the probability that the model can answer a question correctly

### Case Study: Pandora’s Box Problem, CTA-PROMPTED-NONTHINK

###### Problem Parameters

- - Bag Labels: A, B, C
- - Prior Probabilities: A: 0.04, B: 0.68, C: 0.28
- - Discount Factor r: 0.2 Interaction Trace

- Turn 0 User: {Instruction instantiated with the problem parameters}

- Assistant: VERIFY A

Turn 1

- User: The verification result is: NO, A is incorrect. Given this, please provide your next action.

Assistant: VERIFY B Turn 2

- User: The verification result is: NO, B is incorrect. Given this, please provide your next action. Assistant: GUESS C

- Figure 6: Example interaction trace on a 3-bag Pandora’s Box instance with priors (0.04, 0.68, 0.28) and discount factor γ = 0.2 with thinking mode disabled. In this setting, the model explores all bags before committing and follows a suboptimal verification order, rather than prioritizing the highest-probability option.

when conditioned on the document retrieved from the retriever, is estimated on a validation set and provided to the agent as a part of the input x at inference time. Note that pret depends on the retriever and the agent being used, but not on individual questions.

### G Dataset Construction Details for Task: Coding with Selective Testing

This appendix describes the oracle filename-to-format model and the procedure used to generate the coding task dataset FILEREADING.

Each task consists of a structured filename, a CSV file generated under a latent formatting configuration, and an associated query. Filenames combine a small set of indicative tokens (e.g., _eu, _tab, _sas, _cn) with additional irrelevant strings, inducing a prior over possible parsing configurations.

To capture scenarios in which filename cues provide informative signals about file formats, we construct a synthetic dataset of CSV-based question-answering tasks with structured filenames. Each filename is generated by combining a small set of indicative tokens (e.g., _eu, _tab, _sas, _cn) with additional irrelevant strings.

We define an oracle filename-to-format model

Moracle : n  → p(z | n),

which maps a filename n to a prior distribution over formatting configurations z = (zd,zq,zs). For example, filename tokens like _tsv substantially increase the probability of a tab delimiter relative to a default comma delimiter.

We then sample the true formatting configuration z∗ ∼ p(z | n) and generate the corresponding CSV content and task query.

The dataset is constructed so that the correct answer is obtainable only when the file is parsed with the correct configuration; incorrect formatting assumptions lead to parsing failures or misaligned columns. Filenames are represented by four binary features, yielding 24 = 16 distinct filename feature configurations and corresponding prior distributions over formatting attributes. We generate 2,000 task instances, each consisting of a filename, a CSV file generated under a sampled formatting configuration, and an associated query. The dataset is split into 1,400 training examples, 300 validation examples, and 300 test examples. Details of the oracle filename-to-format model, feature templates.

### Case Study: Pandora’s Box Problem, PROMPTED

###### Problem Parameters

- - Bag Labels: A, B, C
- - Prior Probabilities: A: 0.04, B: 0.68, C: 0.28
- - Discount Factor r: 0.2 Interaction Trace Turn 0 User: {Instruction instantiated with the problem parameters, but the prior probabilities are not provided to the model} Assistant: <think> Okay, let’s see. I need to figure out the best action here. So, there are three bags: A, B, and C. Only one of them has the prize worth 1. The discount factor is 0.2. At each step, I can either verify a bag to get information (which takes a timestep) or guess a bag to end the game and get the reward, which is discounted by rt where t is the timestep. Starting at t=0. If I guess now, the reward would be 1 ∗ 0.20 = 1. But if I guess wrong, I get nothing. Since there’s a 1/3 chance of being correct, the expected reward for guessing at t=0 is (1/3) ∗ 1 + (2/3) ∗ 0 = 1/3 ≈ 0.333. Alternatively, if I verify a bag. Let’s say I verify bag A. It takes one timestep (t=0 to t=1). If the verification says "YES", then I know A has the prize. Then, I can guess A at t=1, getting 1 ∗ 0.21 = 0.2. If the verification says "NO", then the prize is in B or C. Then, I have two bags left. But I need to think about the next steps. Wait, but verifying takes a timestep. So, if I verify at t=0, then the next step is t=1. Let’s think about the expected value of verifying. Suppose I verify A at t=0. Then, the result is YES with 1/3, NO with 2/3. If YES: Then at t=1, I can guess A and get 0.2. If NO: Then the prize is in B or C. Now, at t=1, I have two bags left. If I guess at t=1, the expected reward would be (1/2) ∗ 0.21 + (1/2) ∗ 0 = 0.1. Alternatively, I can verify another bag. Let’s say I verify B at t=1. Then, if it’s YES, I can guess B at t=2 (reward 0.22 = 0.04), else verify C at t=2, etc. But this might take longer. Alternatively, maybe it’s better to guess at t=1 if I have two bags left. The expected value would be 0.1 as above. But if I verify, maybe I can get more information. Let me calculate the expected value of verifying A at t=0. So, if I verify A at t=0, then at t=1, if it’s YES (prob 1/3), I get 0.2. If NO (prob 2/3), then I have two bags left. Then, the expected value from that point is the maximum between guessing or verifying again. Let’s assume that the optimal strategy is to guess when there are two bags left. Then, the expected value would be 0.1 as before. So the total expected value of verifying A at t=0 is (1/3) ∗ 0.2 + (2/3) ∗ 0.1 = 0.2/3 + 0.2/3 = 0.4/3 ≈ 0.133. Which is less than the 0.333 from guessing at t=0. So, it seems better to guess at t=0. Wait, but maybe I’m missing something. Because if I verify A at t=0, and it’s NO, then at t=1, I can verify another bag. Let’s say I verify B at t=1. Then, if it’s YES (prob 1/2), then I can guess B at t=2 (reward 0.22 = 0.04). If NO (prob

- 1/2), then the prize is in C, so I can guess C at t=2, getting 0.04. So the expected value here would be 0.04. So, the total expected value if I verify A at t=0, then B at t=1, would be (1/3)∗0.2+(2/3)∗[(1/2)∗0.04+(1/2)∗0.04] = (1/3) ∗ 0.2 + (2/3) ∗ 0.04 = 0.0667 + 0.0267 ≈ 0.0933. Which is worse than the 0.333 from guessing at t=0. So, in that case, guessing at t=0 is better. Alternatively, maybe there’s a better strategy. Let me think again. Suppose I verify one bag, and if it’s no, then verify another. But with discounting, the later rewards are smaller. So, perhaps the optimal strategy is to guess immediately. Because the expected reward from guessing at t=0 is 1/3, which is higher than the expected value of verifying and then proceeding. So, maybe the best action is to GUESS at t=0. But wait, what if I verify one bag and then guess? Let’s say I verify A at t=0. If it’s YES, then I can guess at t=1 (0.2). If it’s NO, then I have two bags left. Then, if I guess at t=1, the expected reward is 0.1. So the total expected value is (1/3) ∗ 0.2 + (2/3) ∗ 0.1 = 0.2/3 + 0.2/3 = 0.4/3 ≈ 0.133, which is less than 0.333. So, indeed, guessing at t=0 is better. Alternatively, if I verify two bags, but that would take more steps. For example, verify A at t=0, then B at t=1. If both are NO, then the prize is in C. Then at t=2, guess C and get 0.22 = 0.04. So the expected value would be (probability that A is NO and B is NO) * 0.04. The probability that A is NO is 2/3, then B is NO is 1/2, so
- 2/3 ∗ 1/2 = 1/3. So expected value is 1/3 ∗ 0.04 = 0.0133. Which is worse than guessing at t=0. So, that’s even worse. So, the conclusion is that guessing at t=0 gives a higher expected value than any strategy involving verification. Therefore, the optimal action is to GUESS one of the bags. But since all bags are symmetric, it doesn’t matter which one. So, the answer should be GUESS A, B, or C. But the user probably expects me to choose one. Let me check if there’s any other consideration. For example, maybe the discount factor is 0.2, which is quite low. So, waiting to get more information might not be worth it. So, the best action is to GUESS now. </think> GUESS A

- Figure 7: Example interaction trace on a 3-bag Pandora’s Box instance with priors (0.04, 0.68, 0.28) and discount factor γ = 0.2, where the model is not given access to the prior probabilities. In this setting, the model implicitly treats the bags as equally likely and follows a suboptimal strategy that deviates from the optimal policy.

### Case Study: Pandora’s Box Problem, CTA-PROMPTED

###### Problem Parameters

- - Bag Labels: A, B, C
- - Prior Probabilities: A: 0.04, B: 0.68, C: 0.28
- - Discount Factor r: 0.2 Interaction Trace Turn 0 User: {Instruction instantiated with the problem parameters} Assistant: <think> Okay, let’s see. I need to figure out whether to verify or guess in this scenario. The problem has three bags: A, B, and C with prior probabilities 0.04, 0.68, and 0.28 respectively. The discount factor is 0.2. First, I should remember that each VERIFY action costs a timestep, and GUESS ends the game immediately. The reward for a correct guess is discounted by rt, where t is the timestep. So, if I guess at t = 0, the discount is 1 (since

- 0.20 = 1). But if I verify, I spend a timestep and then maybe have more information to make a better decision. The key here is to balance between getting more information (which might lead to a higher reward) and the risk of waiting too long, which could reduce the discounted reward. Let me think about the expected value of guessing immediately versus verifying. If I guess at t = 0, the expected reward would be the sum of the probabilities of each bag multiplied by the discount factor. Wait, no. Wait, the reward is 1 if the guessed bag has the prize, 0 otherwise. So the expected reward for guessing at t=0 would be the probability of the chosen bag having the prize. But since I can choose any bag to guess, I should pick the one with the highest prior probability. Looking at the priors: B is 0.68, C is 0.28, A is 0.04. So if I guess B, the expected reward is 0.68 * 1 (since if it’s correct, I get 1, else 0). But since the discount is 0.20 = 1, the expected reward is 0.68. Alternatively, if I verify a bag, I can get more information. Let’s say I verify B first. If I verify B, then I spend a timestep (t=1). Then, depending on the result, I can decide whether to guess or verify again.

... But the problem is that the discount factor is 0.2. So, each time you wait, the reward is discounted. So, the longer you wait, the less valuable the reward becomes. So, the question is whether the information gained from verifying is worth the cost of waiting. Let me think about the expected value of verifying versus guessing. If I guess at t=0, the expected reward is the maximum prior probability, which is 0.68. If I verify a bag, say B, then I spend one time step (t=1). Let’s calculate the expected reward if I verify B. After verifying B, there are two possibilities:

- 1. The result is YES (probability 0.68). Then, I know B has the prize. So, I can guess B at t=1, which would give me a reward of 1 ∗ 0.21 = 0.2.
- 2. The result is NO (probability 1 − 0.68 = 0.32). Then, B does not have the prize. Now, I have to decide whether to guess or verify another bag. But if I get NO on B, then the remaining bags are A and C. Their probabilities would be updated based on the fact that B is not the correct one. Wait, but the problem says that the bags are independent. Wait, actually, the problem states that the prior probabilities are for each bag. So, if B is not the correct one, then the prize is in A or C. However, the original priors are A:0.04, B:0.68, C:0.28. But if we know that B is not the correct one, then the total probability of the prize being in A or C is 0.04 + 0.28 = 0.32. So, the posterior probabilities would be A: 0.04/0.32 = 0.125, and C: 0.28/0.32 = 0.875. So, in this case, if we get a NO on B, then the optimal choice would be to guess C, which has a 0.875 chance of being correct. Then, the reward would be 0.875 ∗ 0.21 (since t=1). But wait, after verifying B at t=0, we spend a timestep (t=1), and then we can guess. So, if we get NO on B, then we can guess C at t=1, which gives us 0.875 ∗ 0.2 = 0.175. Alternatively, if we get YES on B, then we can guess B at t=1, which gives 1 ∗ 0.2 = 0.2. So, the expected reward of verifying B is: 0.68 ∗ 0.2 + 0.32 ∗ 0.175 = 0.136 + 0.056 = 0.192. Compare that to the expected reward of guessing immediately (0.68). So, 0.68 is higher than 0.192. Therefore, verifying B is worse than guessing immediately. </think> Action: GUESS B

- Figure 8: Example model reasoning trace on a 3-bag Pandora’s Box instance with priors (0.04, 0.68, 0.28) and discount factor γ = 0.2. The model explicitly compares the expected value of immediate guessing versus verification and then chooses to guess B immediately, which is the optimal strategy in this case. Key reasoning steps, including the explicit comparison between action value and exploration cost, are highlighted in yellow. This example illustrates that when priors are provided explicitly, the model can reason about uncertainty and exploration cost to select an optimal strategy.

Latent Formatting Variables Each task instance is associated with a latent formatting configuration z = (zd,zq,zs),

where zd ∈ {,,;,\t} denotes the delimiter, zq ∈ {",’} the quote character, and zs ∈ {0,1} the number of header rows to skip. The correct answer can be obtained if and only if the file is parsed using the fully correct configuration z∗.

Filename Features We extract four binary features from each filename n, each indicating the presence or absence of a specific substring: has_eu, has_tsv, has_sas, and has_cn. Each feature is either on or off, resulting in 24 = 16 possible filename feature configurations. Each configuration corresponds to a distinct prior distribution over formatting attributes induced by the oracle model.

Oracle Filename-to-Format Model We define an oracle filename-to-format model

Moracle : n  → p(z | n),

which maps a filename n to a prior distribution over formatting configurations z = (zd,zq,zs). For each of the 16 possible filename feature configurations, the model induces a corresponding prior over formatting attributes. The prior factorizes as

p(z | n) = [p(zd | n),p(zq | n),p(zs | n)],

where each factor is parameterized as a log-linear model over the filename features.

Sampling and File Generation For each filename n, we sample a formatting configuration z ∼ p(z | n) from the oracle model Moracle. We then generate a CSV file whose content conforms to z. The data are constructed such that incorrect parsing—due to an incorrect delimiter, quote character, or number of skipped rows—either produces malformed outputs or prevents access to the correct answer.

Task Instances and Splits Each task instance consists of a filename n, a generated CSV file, and a query requiring the agent to compute an answer from the file. We generate 2,000 task instances in total, split into 1,400 training examples, 300 validation examples, and 300 test examples.

### H Oracle Strategy for Pandora’s Box Problem

Algorithm 1 presents the optimal policy for the Pandora’s Box problem. In this section, we prove its optimality.

We begin by characterizing the structure of an optimal policy. At any state, let S denote the remaining set of boxes and let qi(S) denote the posterior probability that box i ∈ S contains the prize. An optimal policy only needs to consider the box with maximum posterior probability at each step.

First, if the agent commits, the expected reward equals the posterior success probability of the chosen box. Hence, committing to any box other than a maximum-posterior box is suboptimal. Second, verification is beneficial only insofar as it increases the probability of early termination before further discounting. Verifying a higher-posterior box increases the chance of immediate success and therefore weakly dominates verifying a lower-posterior box. Consequently, it suffices to consider the box i⋆ ∈ arg maxi∈S qi(S) at each decision step.

It remains to determine whether the agent should commit to i⋆ immediately or verify it first. Let q = qi⋆(S) = maxi∈S qi(S) be the posterior probability of the most likely box i⋆. If the agent commits immediately, the expected value is

##### Vguess(S) = q.

If it verifies i⋆, then with probability q verification succeeds and yields reward 1, and with probability 1 − q it fails and the problem reduces to the smaller set S \ {i⋆}. Since verification incurs one multiplicative discount factor γ, the expected value of verifying is

Vverify(S) = γ q + (1 − q)V (S \ {i⋆}) .

Algorithm 1 Oracle policy for Pandora’s box with n boxes function SOLVE(S) if |S| = 1 then

return (1, COMMIT(only element of S)) ▷ Base case: Final box selection end if

W ← j∈S pj i⋆ ← arg maxi∈S pi ▷ Select candidate with highest success posterior q ← pi⋆/W

Vguess ← q ▷ Expected value if committing to box i∗ now (Vfail,_) ← SOLVE(S \ {i⋆}) ▷ Recurse to find value if box i∗ is empty Vverify ← γ q + (1 − q) · Vfail ▷ The value of verifying i∗ first

if Vguess ≥ Vverify then

return (Vguess, COMMIT(i⋆)) else

#### return (Vverify, VERIFY(i⋆)) end if end function

Therefore, optimality implies the Bellman recursion

##### V (S) = max Vguess(S), Vverify(S) ,

with base case V ({i}) = 1. This recursion is exactly implemented by Algorithm 1.

### I GRPO Training Details

We use the GRPO implementation of the veRL library [49, 67] for the RL training. The model is trained with a group size of n = 8 and a KL coefficient β = 0.01. During rollout, tokens are sampled with a temperature of 1.0. We use a learning rate of 1 × 10−6 and train with a global batch size of 16. We use a micro-batch size of 2 for model updates and a micro-batch size of 4 for experience. We conduct GRPO training on Qwen3-8B using a single node with 8 NVIDIA A100 GPUs.

### J Prompt Templates

- J.1 Prompts for PANDORA Prompt templates used for PANDORA are shown in Figure 9.
- J.2 Prompts for QA Prompts used in QA are provided in Figure 10.
- J.3 Prompts for FILEREADING Prompts used in the FILEREADING setting are provided in Figures 11, 12, 13, and 14.

### K Case Study: Cost-Aware Decision Traces in FILEREADING with CTA-RL and RL

Figures 15 and 16 compare representative traces under a high relative code cost setting (ρ = 4.0). The RL model (trained without conditioning on explicit priors) tends to default to running unit tests before attempting any code, and does not explicitly reason about the relative costs of UNIT_TESTS versus FILEREADING when choosing its next action. In contrast, the CTA-RL model exhibits the

### Prompts: Pandora’s Box

SYSTEM You are a rational agent tasked with solving sequential decision-making problems under uncertainty. You are given a set of options (bags) with prior probabilities of containing a prize with value 1. You can either VERIFY an option to get information (YES/NO) or GUESS an option to end the game and collect the reward.

- - Each VERIFY action consumes one timestep.
- - The reward for a correct GUESS is discounted by a factor rt, where t is the timestep when you GUESS.
- - You must balance information gathering (VERIFY) with timely exploitation (GUESS) to maximize expected discounted reward. Always respond with exactly one action token per step, using the format: VERIFY <Option> or GUESS <Option>. Instruction
- --- NEW GAME --TIMESTEP: t=0 PROBLEM PARAMETERS:
- - Bag Labels: {labels_str}
- - Prior Probabilities: {priors_str} //only provided to the model in CTA- settings
- - Discount Factor r: {r} Choose your action.

Figure 9: Prompt template for Pandora’s Box setting.

intended cost-aware behavior: it reasons about both (i) uncertainty over the CSV format and (ii) the relative cost of unit tests and code execution, and uses these factors to decide whether verification is worth performing before committing to a code attempt.

### Prompts: QA with Optional Retriever

SYSTEM You are a rational agent tasked with answering factual questions under uncertainty. At each step, you can either directly answer the question or retrieve additional context before answering. Available actions:

- - RETRIEVE: request a related context to consult before deciding your final answer. This consumes one timestep.
- - ANSWER: <your short factual answer>, provide your final answer and end the interaction. Each question comes with your estimated probabilities of answering correctly:
- - p_no_context: probability you can answer correctly without retrieving.
- - p_with_context: probability you can answer correctly after retrieval.

Your goal is to maximize expected discounted reward: Reward = rt∗ correctness, where t is the timestep when you issue ANSWER and correctness ∈ {0,1}.

Retrieval information: If you choose RETRIEVE, your expected answer accuracy after retrieval is p_with_context. If you choose not to retrieve, you must rely only on your current knowledge. Be deliberate — retrieving may improve accuracy but reduces reward due to time discounting. Balance speed and correctness carefully. Always respond with exactly one action token per step, using the format: RETRIEVE or ANSWER: <short factual answer>.

- Instruction - turn 1

–- NEW QUESTION –-

- TIMESTEP: t=0 Question: question Parameters:

- - Discount factor (r): {r}
- - Success probability with retrieval (p_with_context): {p_with_context}

Choose your action: RETRIEVE or ANSWER: <short factual answer>.

Instruction - turn 2

- TIMESTEP: t=1 You have retrieved the following context: {context}

Question: {question} Now decide whether to answer: Respond with: ANSWER: <short factual answer>

Figure 10: Prompt templates for QA.

### System Prompt: FILEREADING

You are an agent tasked with writing code to fulfill an instruction about a CSV file (e.g., answering a question using its contents). Your goal is to produce a correct answer while efficiently using available resources, as measured by discounted reward.

The exact CSV formatting may not be fully known. In practice, you can either proceed using reasonable default assumptions about the format, or run unit tests to verify specific formatting details you are unsure about before committing to a final answer.

###### Allowed actions (choose exactly ONE per turn):

- 1) UNIT_TESTS Run unit tests to debug CSV formatting assumptions. Unit test outputs are perfectly reliable. Available unit tests:

- • test_delimiter(path) → {‘,’, ‘;’, ‘\t’}
- • test_quotechar(path) → {‘"’, “ ’ ”}
- • test_skiprows(path) → {0, 1}

Format (NO code fences): UNIT_TESTS: test_delimiter("file.csv"), test_quotechar("file.csv")

You may include multiple unit tests in a single UNIT_TESTS action. Each individual unit test counts toward the total number of unit tests used.

- 2) CODE Write Python code toward solving the task using your current assumptions about the CSV format.

- • Enclose code in “‘python ... “‘
- • You may import pandas as pd and read the file with: pd.read_csv(filepath, delimiter=..., quotechar=..., skiprows=...)
- • Do NOT print the entire CSV.
- • If your code computes the final result, print it to stdout so it can be read from the output.

After submission, the code will be executed and its stdout and stderr will be returned. You may use this feedback to extract the answer, debug, run additional unit tests, refine, or write additional CODE.

- 3) ANSWER Provide the final answer to the task and end the conversation. Format exactly: ANSWER: <your_answer> The conversation ends immediately after you provide ANSWER. Reward:

- • Let U be the total number of unit tests used.
- • Let C be the total number of CODE actions taken.
- • Final reward = correctness ×(dunit)U × (dcode)C.
- • Discount factors represent cost multiplicatively.
- • A smaller discount factor means a MORE expensive action.
- • If dcode = dkunit, one CODE attempt costs about as much as k UNIT_TESTS.

Figure 11: System prompt for FILEREADING.

### System Prompt: FILEREADING (continuation)

###### General guidance:

- • Start from reasonable default beliefs about the CSV format based on common conventions or provided likelihoods.
- • Both UNIT_TESTS and CODE are costly actions; neither should be treated as free.
- • Use UNIT_TESTS to reduce uncertainty when the expected benefit outweighs their cost.
- • Use CODE to make progress toward solving the task, but recognize that failed or repeated CODE attempts are also costly.
- • Decide when it is better to verify assumptions with UNIT_TESTS versus attempting CODE earlier, taking into account your confidence and the relative cost of these actions.
- • Decide rationally how much debugging and iteration is worthwhile before committing to a final ANSWER.

Figure 12: Continuation of the system prompt for FILEREADING.

### Instruction Prompt Template for FILEREADING (without estimated prior)

You are given a CSV file {csv_name}. Your task: {task_description} Additional context:

- • No format likelihoods are provided.
- • Make reasonable default assumptions about the CSV format based on common conventions, unless you choose to verify them with unit tests.

###### Reward parameters:

- • Unit test discount dunit: {d_unit}
- • Code iteration discount dcode: {d_code}

###### Constraints:

- • You should never print all rows of the CSV or you will get zero reward.
- • You may use UNIT_TESTS, CODE, or ANSWER as described in the system instructions in any order; only the final ANSWER ends the conversation.
- • Incorrect intermediate CODE does not end the episode; only the final ANSWER determines correctness.

- Figure 13: Instruction prompt template specifying the CSV task, reward parameters, and constraints provided to the agent.

### Instruction Prompt Template for FILEREADING (with CTA)

You are given a CSV file {csv_name}. Your task: {task_description} Additional context:

- • Estimated format likelihoods are provided below.
- • These likelihoods reflect how likely each formatting option is in practice and can be used as default assumptions.

Format likelihoods: {prior}

Reward parameters:

- • Unit test discount dunit: {d_unit}
- • Code iteration discount dcode: {d_code}

###### Constraints:

- • You should never print all rows of the CSV or you will get zero reward.
- • You may use UNIT_TESTS, CODE, or ANSWER as described in the system instructions.

- Figure 14: Instruction prompt template with estimated CSV format likelihoods, enabling the agent to use probabilistic defaults when trading off unit tests, code execution, and early commitment.

Case study: Example thinking trace of RL model (without estimated priors) (ρ = 4.0)

Content: <think>

Okay, let’s tackle this problem. The user has a CSV file named ‘race_tsv_sas.tsv‘ and they want the minimum salary value, excluding any None entries. The file’s name ends with .tsv, which usually stands for Tab-Separated Values. So, the delimiter is likely a tab character. But I should confirm that. First, I need to figure out the CSV format. Since it’s a TSV, the delimiter is probably a tab. But maybe the file uses a different delimiter. To be safe, I should run a unit test to check the delimiter. The test_delimiter function can help here. If the delimiter is a tab, then the test should return ‘\t‘. Next, the quote character. In TSVs, sometimes fields are enclosed in quotes, but it’s not always the case. The default quotechar for pandas is double quotes, but maybe this file uses single quotes. Testing with test_quotechar would help. Also, sometimes CSV files have header rows that need to be skipped. The test_skiprows function can check if there’s a need to skip rows. But since the file is a TSV, maybe the first row is the header. However, without knowing the exact structure, it’s safer to assume that the first row is the header and set skiprows=0. But maybe the file has some leading rows to skip. But since the user mentioned it’s a TSV, I can start by assuming the delimiter is a tab. Let me run the test_delimiter first. If the delimiter is a tab, then I can proceed. If not, I’ll adjust accordingly. Once the delimiter is confirmed, I can use pandas to read the CSV. Then, check the ‘salary‘ column, filter out None values, and find the minimum. But to avoid errors, maybe I should first run the unit tests to confirm the delimiter and quotechar. Let’s start with that. </think>

Action: UNIT_TESTS: test_delimiter("race_tsv_sas.tsv"), test_quotechar("race_tsv_sas.tsv")

- Figure 15: Example reasoning trace of an RL-trained model without explicit prior conditioning in the CSV exploration task. Despite operating under the same high relative code cost setting (ρ = 4.0), the model defaults to verification-first behavior based on surface cues (e.g., file extension) and does not explicitly reason about uncertainty or cost trade-offs, illustrating a lack of adaptive decision-making compared to the CTA-RL model.

### Case study: Example thinking trace of CTA-RL model (ρ = 4.0)

Content: <think>

Okay, let’s tackle this problem. The task is to compute the average of the ‘salary‘ column from the CSV file ‘must_eu.csv‘, excluding any None values. The challenge is figuring out the correct CSV format parameters to read the file properly. First, I need to determine the delimiter, quotechar, and skiprows parameters. The format likelihoods are given, so I should start with the most probable options. The delimiter is most likely to be ‘;‘ with a probability of ∼0.85. The quotechar is probably ‘"‘ with 0.589, and skiprows is 0 with 0.607. But I’m not 100% sure, so maybe I should run some unit tests to confirm. But wait, the reward parameters are important. Each unit test costs 0.77 per use, and code attempts cost 0.3515 per use. Since dcode = d4unit, one code attempt is equivalent to 4 unit tests. So, if I’m uncertain, it’s cheaper to run unit tests first. However, if I can make an educated guess, maybe I can proceed with code. Let me think: the delimiter is most likely ‘;‘, so maybe I can try reading the file with that. But if the actual delimiter is different, the code will fail. Alternatively, running a unit test for the delimiter would be better. Let’s check the delimiter first. So, I’ll run UNIT_TESTS to check the delimiter. The test_delimiter function will tell me if it’s comma, semicolon, or tab. Since the likelihood is high for semicolon, but I need to confirm. Let’s do that. Once I know the delimiter, I can check the quotechar. The quotechar is either ‘"‘ or ‘’‘, with higher likelihood for ‘"‘ (0.589). But again, maybe a unit test is better. However, if I proceed with code, maybe I can handle the possible quotechar. But if the code fails, I’ll have to adjust. Alternatively, maybe I can proceed with the most likely parameters and see. Let’s try that. Let’s assume delimiter is ‘;‘, quotechar is ‘"‘, and skiprows 0. Then, read the CSV with these parameters, compute the average of salary, skipping None values. But if the actual delimiter is different, the code will fail. So, perhaps it’s better to run unit tests first. Let’s start with the delimiter. So, the first action is to run UNIT_TESTS to determine the delimiter and quotechar. Let’s do that. </think> Action: UNIT_TESTS: test_delimiter("must_eu.csv"), test_quotechar("must_eu.csv")

- Figure 16: Example reasoning trace of the CTA-RL model on the FILEREADING task (ρ = 4.0), illustrating cost-aware trade-offs between unit tests and code execution under a high relative code cost setting, while jointly reasoning about format uncertainty.

### NeurIPS Paper Checklist

- 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and introduction describe the paper’s scope as cost-aware exploration for LLM agents. We claim that we can induce LLMs to explicitly reason about balancing the cost and uncertainty tradeoffs and act more optimally in the environment by conditioning the LLMs on inferred priors. The claims are supported by the formalization, method, and experimental results in the rest of the paper.

Guidelines:

- • The answer [N/A] means that the abstract and introduction do not include the claims made in the paper.
- • The abstract and/or introduction should clearly state the claims made, including the contributions made in the paper and important assumptions and limitations. A [No] or [N/A] answer to this question will not be perceived well by the reviewers.
- • The claims made should match theoretical and experimental results, and reflect how much the results can be expected to generalize to other settings.
- • It is fine to include aspirational goals as motivation as long as it is clear that these goals are not attained by the paper.

- 2. Limitations Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: We discuss the limitations in Appendix A. Guidelines:

- • The answer [N/A] means that the paper has no limitation while the answer [No] means that the paper has limitations, but those are not discussed in the paper.
- • The authors are encouraged to create a separate “Limitations” section in their paper.
- • The paper should point out any strong assumptions and how robust the results are to violations of these assumptions (e.g., independence assumptions, noiseless settings, model well-specification, asymptotic approximations only holding locally). The authors should reflect on how these assumptions might be violated in practice and what the implications would be.
- • The authors should reflect on the scope of the claims made, e.g., if the approach was only tested on a few datasets or with a few runs. In general, empirical results often depend on implicit assumptions, which should be articulated.
- • The authors should reflect on the factors that influence the performance of the approach. For example, a facial recognition algorithm may perform poorly when image resolution is low or images are taken in low lighting. Or a speech-to-text system might not be used reliably to provide closed captions for online lectures because it fails to handle technical jargon.
- • The authors should discuss the computational efficiency of the proposed algorithms and how they scale with dataset size.
- • If applicable, the authors should discuss possible limitations of their approach to address problems of privacy and fairness.
- • While the authors might fear that complete honesty about limitations might be used by reviewers as grounds for rejection, a worse outcome might be that reviewers discover limitations that aren’t acknowledged in the paper. The authors should use their best judgment and recognize that individual actions in favor of transparency play an important role in developing norms that preserve the integrity of the community. Reviewers will be specifically instructed to not penalize honesty concerning limitations.

- 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [Yes]

Justification: The paper derives the optimal policy for the discounted Pandora’s Box setting and provides the full algorithm and proof in Appendix H. The assumptions of the problem are stated in Section 3.1.

Guidelines:

- • The answer [N/A] means that the paper does not include theoretical results.
- • All the theorems, formulas, and proofs in the paper should be numbered and crossreferenced.
- • All assumptions should be clearly stated or referenced in the statement of any theorems.
- • The proofs can either appear in the main paper or the supplemental material, but if they appear in the supplemental material, the authors are encouraged to provide a short proof sketch to provide intuition.
- • Inversely, any informal proof provided in the core of the paper should be complemented by formal proofs provided in appendix or supplemental material.
- • Theorems and Lemmas that the proof relies upon should be properly referenced.

#### 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The paper describes the task definitions, datasets, prior estimation procedures, baselines, metrics, and prompt templates (Sec J) needed to reproduce the main experiments.

Guidelines:

- • The answer [N/A] means that the paper does not include experiments.
- • If the paper includes experiments, a [No] answer to this question will not be perceived well by the reviewers: Making the paper reproducible is important, regardless of whether the code and data are provided or not.
- • If the contribution is a dataset and/or model, the authors should describe the steps taken to make their results reproducible or verifiable.
- • Depending on the contribution, reproducibility can be accomplished in various ways. For example, if the contribution is a novel architecture, describing the architecture fully might suffice, or if the contribution is a specific model and empirical evaluation, it may be necessary to either make it possible for others to replicate the model with the same dataset, or provide access to the model. In general. releasing code and data is often one good way to accomplish this, but reproducibility can also be provided via detailed instructions for how to replicate the results, access to a hosted model (e.g., in the case of a large language model), releasing of a model checkpoint, or other means that are appropriate to the research performed.
- • While NeurIPS does not require releasing code, the conference does require all submissions to provide some reasonable avenue for reproducibility, which may depend on the nature of the contribution. For example

- (a) If the contribution is primarily a new algorithm, the paper should make it clear how to reproduce that algorithm.
- (b) If the contribution is primarily a new model architecture, the paper should describe the architecture clearly and fully.
- (c) If the contribution is a new model (e.g., a large language model), then there should either be a way to access this model for reproducing the results or a way to reproduce the model (e.g., with an open-source dataset or instructions for how to construct the dataset).
- (d) We recognize that reproducibility may be tricky in some cases, in which case authors are welcome to describe the particular way they provide for reproducibility. In the case of closed-source models, it may be that access to the model is limited in

some way (e.g., to registered users), but it should be possible for other researchers to have some path to reproducing or verifying the results.

#### 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes] Justification: We provide the data used via supplementary materials in OpenReview. Guidelines:

- • The answer [N/A] means that paper does not include experiments requiring code.
- • Please see the NeurIPS code and data submission guidelines (https://neurips.cc/ public/guides/CodeSubmissionPolicy) for more details.
- • While we encourage the release of code and data, we understand that this might not be possible, so [No] is an acceptable answer. Papers cannot be rejected simply for not including code, unless this is central to the contribution (e.g., for a new open-source benchmark).
- • The instructions should contain the exact command and environment needed to run to reproduce the results. See the NeurIPS code and data submission guidelines (https: //neurips.cc/public/guides/CodeSubmissionPolicy) for more details.
- • The authors should provide instructions on data access and preparation, including how to access the raw data, preprocessed data, intermediate data, and generated data, etc.
- • The authors should provide scripts to reproduce all experimental results for the new proposed method and baselines. If only a subset of experiments are reproducible, they should state which ones are omitted from the script and why.
- • At submission time, to preserve anonymity, the authors should release anonymized versions (if applicable).
- • Providing as much information as possible in supplemental material (appended to the paper) is recommended, but including URLs to data and code is permitted.

#### 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes] Justification: We specify the details in Sec 5, Sec G, Sec I. Guidelines:

- • The answer [N/A] means that the paper does not include experiments.
- • The experimental setting should be presented in the core of the paper to a level of detail that is necessary to appreciate the results and make sense of them.
- • The full details can be provided either with the code, in appendix, or as supplemental material.

#### 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes] Justification: We provide significance tests in Sec D. Guidelines:

- • The answer [N/A] means that the paper does not include experiments.
- • The authors should answer [Yes] if the results are accompanied by error bars, confidence intervals, or statistical significance tests, at least for the experiments that support the main claims of the paper.
- • The factors of variability that the error bars are capturing should be clearly stated (for example, train/test split, initialization, random drawing of some parameter, or overall run with given experimental conditions).

- • The method for calculating the error bars should be explained (closed form formula, call to a library function, bootstrap, etc.)
- • The assumptions made should be given (e.g., Normally distributed errors).
- • It should be clear whether the error bar is the standard deviation or the standard error of the mean.
- • It is OK to report 1-sigma error bars, but one should state it. The authors should preferably report a 2-sigma error bar than state that they have a 96% CI, if the hypothesis of Normality of errors is not verified.
- • For asymmetric distributions, the authors should be careful not to show in tables or figures symmetric error bars that would yield results that are out of range (e.g., negative error rates).
- • If error bars are reported in tables or plots, the authors should explain in the text how they were calculated and reference the corresponding figures or tables in the text.

#### 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [Yes] Justification: We provide details in Sec I. Guidelines:

- • The answer [N/A] means that the paper does not include experiments.
- • The paper should indicate the type of compute workers CPU or GPU, internal cluster, or cloud provider, including relevant memory and storage.
- • The paper should provide the amount of compute required for each of the individual experimental runs as well as estimate the total compute.
- • The paper should disclose whether the full research project required more compute than the experiments reported in the paper (e.g., preliminary or failed experiments that didn’t make it into the paper).

#### 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research uses public datasets, synthetic data, and open model/retriever components for evaluating cost-aware LLM agent behavior. We are not aware of any violations of the NeurIPS Code of Ethics in our work.

Guidelines:

- • The answer [N/A] means that the authors have not reviewed the NeurIPS Code of Ethics.
- • If the authors answer [No], they should explain the special circumstances that require a deviation from the Code of Ethics.
- • The authors should make sure to preserve anonymity (e.g., if there is a special consideration due to laws or regulations in their jurisdiction).

#### 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes] Justification: We discuss the impacts in Conclusion section and Sec B. Guidelines:

- • The answer [N/A] means that there is no societal impact of the work performed.
- • If the authors answer [N/A] or [No], they should explain why their work has no societal impact or why the paper does not address societal impact.

- • Examples of negative societal impacts include potential malicious or unintended uses (e.g., disinformation, generating fake profiles, surveillance), fairness considerations (e.g., deployment of technologies that could make decisions that unfairly impact specific groups), privacy considerations, and security considerations.
- • The conference expects that many papers will be foundational research and not tied to particular applications, let alone deployments. However, if there is a direct path to any negative applications, the authors should point it out. For example, it is legitimate to point out that an improvement in the quality of generative models could be used to generate Deepfakes for disinformation. On the other hand, it is not needed to point out that a generic algorithm for optimizing neural networks could enable people to train models that generate Deepfakes faster.
- • The authors should consider possible harms that could arise when the technology is being used as intended and functioning correctly, harms that could arise when the technology is being used as intended but gives incorrect results, and harms following from (intentional or unintentional) misuse of the technology.
- • If there are negative societal impacts, the authors could also discuss possible mitigation strategies (e.g., gated release of models, providing defenses in addition to attacks, mechanisms for monitoring misuse, mechanisms to monitor how a system learns from feedback over time, improving the efficiency and accessibility of ML).

#### 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: The paper operates on small LLMs and does not directly release a new highrisk model or dataset. We do not perceive the method itself as having a high potential for misuse.

Guidelines:

- • The answer [N/A] means that the paper poses no such risks.
- • Released models that have a high risk for misuse or dual-use should be released with necessary safeguards to allow for controlled use of the model, for example by requiring that users adhere to usage guidelines or restrictions to access the model or implementing safety filters.
- • Datasets that have been scraped from the Internet could pose safety risks. The authors should describe how they avoided releasing unsafe images.
- • We recognize that providing effective safeguards is challenging, and many papers do not require this, but we encourage authors to take this into account and make a best faith effort.

#### 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes] Justification: We have cited relevant papers in the paper. Guidelines:

- • The answer [N/A] means that the paper does not use existing assets.
- • The authors should cite the original paper that produced the code package or dataset.
- • The authors should state which version of the asset is used and, if possible, include a URL.
- • The name of the license (e.g., CC-BY 4.0) should be included for each asset.
- • For scraped data from a particular source (e.g., website), the copyright and terms of service of that source should be provided.

- • If assets are released, the license, copyright information, and terms of use in the package should be provided. For popular datasets, paperswithcode.com/datasets has curated licenses for some datasets. Their licensing guide can help determine the license of a dataset.
- • For existing datasets that are re-packaged, both the original license and the license of the derived asset (if it has changed) should be provided.
- • If this information is not available online, the authors are encouraged to reach out to the asset’s creators.

#### 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: We have documented the construction procedure of the dataset FILEREADING, design of framework, and eval protocol in the paper. We will also release the code and data.

Guidelines:

- • The answer [N/A] means that the paper does not release new assets.
- • Researchers should communicate the details of the dataset/code/model as part of their submissions via structured templates. This includes details about training, license, limitations, etc.
- • The paper should discuss whether and how consent was obtained from people whose asset is used.
- • At submission time, remember to anonymize your assets (if applicable). You can either create an anonymized URL or include an anonymized zip file.

#### 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A] Justification: This work does not involve human subjects. Guidelines:

- • The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.
- • Including this information in the supplemental material is fine, but if the main contribution of the paper involves human subjects, then as much detail as possible should be included in the main paper.
- • According to the NeurIPS Code of Ethics, workers involved in data collection, curation, or other labor should be paid at least the minimum wage in the country of the data collector.

#### 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A] Justification: This work does not involve human subjects. Guidelines:

- • The answer [N/A] means that the paper does not involve crowdsourcing nor research with human subjects.
- • Depending on the country in which research is conducted, IRB approval (or equivalent) may be required for any human subjects research. If you obtained IRB approval, you should clearly state this in the paper.

- • We recognize that the procedures for this may vary significantly between institutions and locations, and we expect authors to adhere to the NeurIPS Code of Ethics and the guidelines for their institution.
- • For initial submissions, do not include any information that would break anonymity (if applicable), such as the institution conducting the review.

#### 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [Yes]

Justification: LLMs are central to the paper’s methodology and evaluation. The paper specifies the base model, prompting variants, evaluation settings, and RL-trained variants in main body and the appendix.

Guidelines:

- • The answer [N/A] means that the core method development in this research does not involve LLMs as any important, original, or non-standard components.
- • Please refer to our LLM policy in the NeurIPS handbook for what should or should not be described.

