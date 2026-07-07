## Self-Execution Simulation Improves Coding Models

# arXiv:2604.03253v1[cs.CL]11Mar2026

Gallil Maimon12 Ori Yoran1 Felix Kreuk1 Michael Hassid12 Gal Cohen1 Pierre Chambon13 Yossi Adi12

### Abstract

A promising research direction in enabling LLMs to generate consistently correct code involves addressing their inability to properly estimate program execution, particularly for code they generate. In this work, we demonstrate that code LLMs can be trained to simulate program execution in a step-by-step manner and that this capability can be leveraged to improve competitive programming performance. Our approach combines supervised fine-tuning on natural language execution traces, textual explanations grounded in true execution, with reinforcement learning using verifiable rewards. We introduce two complementary objectives: output prediction given code and inputs, and solving competitive programming tasks with either ground-truth or self-predicted execution feedback. These objectives enable models to perform self-verification over multiple candidate solutions, and iterative self-fixing by simulating test execution. Across multiple competitive programming benchmarks, our method yields consistent improvements over standard reasoning approaches. We further present ablations and analysis to elucidate the role of execution simulation and its limitations.

### 1. Introduction

Going beyond treating code as a static text block holds great promise in advancing code LLMs. This involves jointly modelling program syntax and execution dynamics, similar to how developers reason during debugging and development (Armengol-Estap´e et al., 2025; Li et al., 2025; Thimmaiah et al., 2025; Copet et al., 2025; Beck et al., 2026). Despite its promise, translating execution prediction capabilities into consistent gains on practical programming tasks remains an open challenge. Moreover, Gu et al. (2024a); Olausson et al. (2024); Kamoi et al. (2024) indicate that current models often fail to faithfully simulate runtime be-

1FAIR team, Meta 2Hebrew University of Jerusalem 3Inria. Correspondence to: Gallil Maimon <gallil.maimon@mail.huji.ac.il>. Preprint. April 7, 2026.

haviour or to consistently identify and explain errors in code they generate.

Code execution is widely used in various parts of training and inference of code LLMs. This includes feedback from code execution (Gehring et al., 2025; Peng et al., 2025) or rich tool-based signals in agentic settings (Xia et al., 2025). However, executing code at scale for training or inference poses practical challenges, such as environment setup (Bogin et al., 2024), managing code dependencies (Jimenez

- et al., 2023), handling partial or non-executable code, and sandboxing. In broader settings, program execution can also be computationally expensive and time-consuming; for example, runs of MLE-Bench can take up to 9 hours (Chan et al., 2024; Zheng et al., 2026). Predicting execution outcomes could mitigate these challenges by enabling large rollouts and policy optimisation without code execution (MiniMax, 2026; Kimi et al., 2025). More broadly, using execution prediction to support reasoning and planning in coding tasks can be viewed as a form of world modelling in the code domain (Ha & Schmidhuber, 2018; Ding et al., 2025).

In this work, we take a step in this direction. We show LLMs can learn to simulate program execution step-by-step, including code they generated, and use this capability to improve competitive programming performance. We start by training models on natural-language execution traces – text explanations grounded in real program executions – and then further refining them using single-turn reinforcement learning for code output prediction. Equipped with this capability, we empirically demonstrate how models can perform self-verification over parallel solutions based on simulated execution (best@k). Inspired by Gehring et al. (2025), we also design a multi-turn reinforcement learning pipeline that enables iterative self-fixing through code proposal, execution simulation, and refinement. Figure 1 provides a conceptual overview of the proposed methods.

Results suggest the proposed training recipe leads to significant improvements in output prediction on CruxEval (Gu

- et al., 2024b) (up to 43%) and competitive programming solutions (Li et al., 2022; Jain et al., 2024) (up to 39%) relative to the evaluated baseline. This applies to both external and self-generated code solutions. Under the best@k setting, using the model’s output prediction to verify its own candidate solutions improves code correctness by up to 5.5%

[Figure 1]

###### Solve

Improve Coding

Simulate execution

[Figure 2]

[Figure 3]

solution.py

Write a python code solution that given a string prints the string repeated based on the total appearances of “a” and “b” in the string.

`bab`

stdout

Rank Solutions by Predicted Pass

[Figure 4]

<think>

>>> babbab

x = input()

[Figure 5]

- n_a = x.count('a')
- n_b = x.count('b') n = max(n_a, n_b) print(x * n)

#### OR

[Figure 6]

[Figure 7]

Simulate execution

Input n

[Figure 8]

Fix Code Based on Feedback

- Figure 1. A conceptual outline of how one can use self-execution simulation of a generated code solution (or solutions) on public or generated test cases to improve coding performance. The simulation can be used as feedback to select the best solution from a few candidates (best@k) or to iteratively fix the code as needed (self-RLEF). See Section 3 for details.

absolute points on competitive programming tasks. In the multi-turn setting, we observe consistent gains across all evaluated configurations. Compared to ground-truth execution, both best@k and multi-turn variants show a relatively small degradation. Finally, we conduct analysis to highlight the strengths and limitations of the proposed approach.

∼30M functions from basic code sources and 35k from competitive programming problems. For all of the above, we use Llama3-70B-Instruct (Grattafiori et al., 2024).

While CWM (Copet et al., 2025) focused primarily on a structured, JSON-like format to describe the step-by-step execution, we wish to focus on natural language description of this data. Relative to the structured format, a free-form variant holds several benefits. First, as based on natural language, it closely matches the reasoning-style data already used by LLMs. It also enables adding semantic context to operations, e.g., explaining an update to an array in the scope of a dynamic programming code. Finally, it naturally abstracts away unnecessary details, such as summarising a long loop that reverses strings character by character.

Our Contributions: We provide a training recipe, showing that code LLMs are capable of simulating the program execution for both external and self-generated code. With that in mind, we introduce a practical framework for leveraging this behaviour by filtering code solutions based on predicted output (i.e., self-verification). Lastly, we present a multi-turn training and inference process to perform iterative self-fixing of the model’s generated code.

To this end, we prompt Qwen3-32B-FP8 (without thinking) (Yang et al., 2025) to “translate” execution traces from raw structured format to a natural language explanation. See Appendix A.4 for the exact prompt. We discard instances where the model’s predicted output does not match the ground truth, resulting in ∼80 M execution descriptions for general Python functions and 115 k for competitive programming solutions (notice, each traced function may contain several io-pairs). The resulting data is formatted as instruction-following examples and used for model finetuning during the SFT phase. In which, the user requests a step-by-step explanation of a program’s execution for a given input, and the assistant provides the translated explanation. Sample instances are provided in Appendix A.5.

### 2. Boosting Execution Simulation

Following Copet et al. (2025), we collect executable Python programs with input–output pairs and record their line-byline execution. Next, we convert these execution traces into natural-language explanations and use the resulting data for supervised fine-tuning. We then further train the model using verifiable rewards on an output prediction task. The next sections describe these post-training steps in detail.

##### 2.1. Natural Language Execution Tracing (NLEX)

We collect Python functions from public repositories and automatically synthesise suitable inputs using a combination of LLM prompting and lightweight fuzzing techniques. In addition, we collect LLM-generated solutions to competitive programming problems from CodeContests (Li et al., 2022), and keep their provided tests as inputs. Although this portion of the data is smaller in scale, it involves substantially more complex programs. We then record execution traces for each program–input pair, capturing intermediate variable states throughout execution. Following Copet et al. (2025), we exclude traces exceeding 10k events or requiring more than 1MB of storage. The resulting dataset comprises

##### 2.2. Output Prediction Environment

Following standard practice in reasoning models, we posttrain our model using Reinforcement Learning with Verifiable Rewards (RLVR). We define an output prediction environment, based on coding tasks, where the model reasons over a given (code, stdin) pair to predict the resulting stdout. We employ a terminal binary reward, scoring +1 if the prediction matches the true stdout, and −1 other-

[Figure 9]

code.py Simulate

Inputs

stdout

[Figure 10]

| | |
|---|---|
| | |

>>> stdout

[Figure 11]

Code Execution

- Tracer

STDIN code.py

(a) Output Prediction

(b) Competitive programming with (self) execution feedback

Question Solve

[Figure 12]

[Figure 13]

code.py

Test in Simulate

[Figure 14]

[Figure 15]

Execute

2) Multi-Task RLVR

[Figure 16]

Natural Language Execution

- Traces

###### Fix

[Figure 17]

###### Execution Feedback

Explain

[Figure 18]

[Figure 19]

Fix/ Submit!

| | |
|---|---|
| | |

[Figure 20]

Final Code

1) Supervised Fine-Tuning Data Creation

- Figure 2. The two parts of our training pipeline. 1) Supervised fine tuning on natural language execution traces (NLEX), 2) Multi-task reinforcement learning on output prediction and competitive programming (optionally with multi-turn feedback and fixing).

wise, allowing 1e − 5 tolerance in float comparisons.

The intended downstream use of the output prediction ability is simulating the execution of model generated solutions to competitive programming questions. To that end, we construct the output prediction environment on precisely such data. We collect solutions from strong LLMs to competitive programming questions and use the stdin of the matching public tests. Moreover, the higher difficulty of competitive programming problems makes them particularly well suited for post-training. Figure 2 depicts the optimisation pipeline.

- 3. Self-Execution For Verification

Formally, given a set of solutions S, with public input-output pairs (int,outt) ∈ T , we use a model to simulate execution, predict the output Msim(s,int), and select:

Best(S) := arg max

1[Msim(s,int) = outt].

s∈S (int,outt)∈T

We use rank_score_at_k (Hassid et al., 2024) to provide an unbiased accuracy estimate for generating k solutions and selecting the one with the highest score under the proposed heuristic. Specifically, we use 20 generated solutions per task and 5 output-prediction attempts per test.

Given models with increased ability to simulate code execution, we ask “How can this ability be used to boost programming abilities?” Arguably, the simplest and most straightforward approach to leverage such capability is through post-hoc solution filtering. In this approach, candidate solutions are simulated on public or generated tests and retained only if their predicted outputs align with the expected ones.

Recall, the primary focus of this work is self-simulation. In which, the same LLM is used to both generate candidate solutions and simulate their execution. That said, the same method can also be applied to solutions produced by other models. In Section 6, we present empirical evidence demonstrating the efficacy of this approach in both setups.

### 4. Self-Execution For Fixing

For that, we adopt a best-of-k (best@k) evaluation setup, where the model independently samples k candidate solutions and selects the final one based on predefined criteria. In our setup, selection is based on the model’s execution prediction. In other words, for each candidate, the model simulates its execution on public test cases and checks whether the predicted outputs match the expected ones. The candidate predicted to pass the greatest number of public tests is selected for submission. In case of a tie we randomly select a solution among the ones that passed the maximum number of tests. We denote this approach best@k simulate. Notice, during inference we do not access any private tests nor ground-truth verification.

Another approach to leveraging execution feedback is through multi-turn interaction with a computational environment to perform code fixing. Gehring et al. (2025) demonstrated that exposing LLMs to environmental feedback can enhance programming performance by allowing models to iteratively refine solutions based on information from failed test cases. However, as mentioned above, this may introduce practical challenges such as environment configuration, code dependencies, and non-executable code.

Motivated by this paradigm, we introduce an approach that uses predicted execution outputs as feedback instead of

Algorithm 1 Multi-Turn Self-RLEF Rollout Require: model M, question q, public tests Tpub =

output, predicted output), decide whether the code is correct or not. If correct, submit the code solution, otherwise, fix the code to provide a new solution.

{(int,outt)}

- 1: s ← M(q) {Generate an initial code solution}
- 2: k ← 1
- 3: while k ≤ Kmax do
- 4: for (int,outt) ∈ Tpub do
- 5: outˆ t ← M(s,int) {self execution simulation}
- 6: end for
- 7: sub fix ← M(q,s,{(int,outt,outˆ t)}) {Submit current code as correct or provide new solution}

- 8: if sub fix = submit then

- 9: Return s
- 10: else
- 11: s ← sub fix

- 12: end if
- 13: k ← k + 1
- 14: end while

• Optional - Repeat turns 2 and 3 until a code solution is submitted or the maximum turns are reached.

Since the model’s ability to accurately predict execution outcomes may be weak at the start of RL training, relying solely on self-predicted feedback could lead the model to disregard this noisy signal. To mitigate this, we initially provide ground-truth execution feedback during training. As training progresses, one might switch from true execution signals to model-predicted execution outputs (Bengio et al., 2015). Alternatively, transition can also be deferred entirely to inference time. Our preliminary results showed no noticeable gap between the approaches, so we use the latter for simplicity. We denote the following approach Self-RLEF.

### 5. Experimental Setup

actual program execution. Note, unlike the method presented in Section 3, that verifies multiple solutions via selfexecution, the multi-turn setup refines solutions sequentially based on predicted feedback. Ideally, this approach can leverage richer signals, such as past solutions and execution details. While similar world-modelling approaches have been explored in vision, recent work shows limited gains from such signals (Qian et al., 2026). In contrast, we show that using execution simulation can improve performance.

Specifically, we propose a multi-turn environment with explicit context switching, i.e. where each interaction step is represented as an independent single-turn prompt containing only the relevant information (see details in the bullets below). This design enables fine-grained control over information flow. For instance, ensuring that execution simulation is isolated from solution reasoning and from access to the correct outputs. Moreover, it mitigates long-context challenges commonly associated with multi-turn reasoning (Yao

- et al., 2022). Finally, this context switching also naturally allows one to extend the number of fix turns at inference

- as each fix turn is independent. A formal description of the rollout procedure is provided in Algorithm 1, and an example inference of our model in Appendix A.2. In words, the multi-turn setup is designed as follows:

- • Turn 1 - Solve - Given a question, provide a code solution to solve the provided question.
- • Turn 2 - Simulate - Given a code snippet and a test input, simulate the execution and predict the output that will be printed to the standard output. This step is performed independently for each public test.
- • Turn 3 - Submit or Fix - Given a question, a candidate solution and feedback about each test (input, expected

##### 5.1. Datasets

We describe all datasets and configurations used to train and evaluate our models and baselines below. Note that each problem in competitive programming usually includes between one and four public test cases, typically provided in the problem description. These serve as basic checks for correctness and output formatting. In addition, a larger set of private tests, unavailable to the model, is used to better assess solution correctness, including coverage of edge cases and compliance with runtime constraints.

- 5.1.1. TRAINING DATASETS

The NLEX dataset, as presented in Section 2.1, was used for supervised fine-tuning, together with OpenMathReasoning (Moshkov et al., 2025) and OpenCodeReasoning (Ahmad et al., 2025) datasets, to bootstrap reasoning abilities.

During RL, models were optimised for both solving and predicting the output of competitive programming solutions. For that, we use the training split derived from CodeContests (Li et al., 2022), after heuristically filtering malformed instances, which results in ∼12.2k problems. For output prediction, we sample 10 candidate solutions and retain up to four correct ones per model, since all correct submissions yield identical outputs. We then pair each retained solution with all of its public test cases, treating each as an output prediction instance, resulting in a total of ∼143k code–input–output examples. All solutions were generated using CWM and Qwen2.5-7B (Qwen et al., 2025).

- 5.1.2. EVALUATION DATASETS

We present evaluation datasets for competitive programming questions (first two), and output prediction (last two).

80

DeepSeek-V3

Qwen3-32B Qwen3-14B

CruxEval-OutputAccuracy(%)

70

Qwen2.5-72B

Qwen3-8B

Llama-4-Scout

Gemma-3-27B

60

Qwen3-4B

+21.9%

+43.0%

Gemma-3-12B

50

Qwen2.5-3B

40

W.O NLEX

W. NLEX

Qwen2.5-7B

###### 2B 10B 100B

Active Parameters (log scale)

- Figure 3. CruxEval-O performance compared to model active parameters. Arrows demonstrate the benefit from training on NLEX data. We also compare to open models.

LCB-IO. We curate a subset from LiveCodeBench-v6 (Jain et al., 2024) containing only problems evaluated via stdio tests, which we refer to as LCB-IO. This restriction simplifies output prediction, as the task reduces to determining the content written to stdout given a specific stdin. The resulting set includes 287 problems.

DMC. We follow Gehring et al. (2025) and use the validation and test splits of CodeContests (Li et al., 2022), yielding an additional evaluation set with a different distribution, denoted DMC, and consisting of 282 problems.

CruxEval-O (Gu et al., 2024b) is a widely adopted benchmark consisting of short Python functions paired with input–output examples. The task requires the model to infer the function’s return value given the code and its input.

Output prediction for competitive programming. We generate 20 solutions per-question from both DMC and LCB-IO using the same LLMs as mentioned above, without filtering or de-duplicating solutions, to perfectly match the real distribution of generated solutions. Such data is also used for best@k type metric calculations.

##### 5.2. Trained Models

We post-train Qwen2.5-Base models of sizes 3B and 7B, together with CWM-base using the datasets described in Section 5.1.1. For RL we use an asynchronous RL infrastructure, adopting the same RL algorithm as in CWM, with different hyperparameters. When performing multi-task training we employ sample-level weighting. Furthermore, we apply reward scaling, following Ruan et al. (2025), assigning a weight of 0.8 to the output prediction objective. For all multi-turn repair environments, including self-RLEF, we allow a maximum of one repair attempt during training (two solution turns in total, including the initial attempt), and

Table 1. Output prediction performance of Qwen models trained with RLVR for output prediction, with and without NLEX data.

BASE MODEL LCB-IO-OUT DMC-TEST-OUT

PASS@1 @5 @10 @1 @5 @10 QWEN-7B (OURS) 79.7 89.4 91.1 77.3 89.3 91.6

W.O NLEX 75.7 87.8 89.7 71.8 87.0 89.9 QWEN-3B (OURS) 66.4 80.6 83.9 59.4 78.2 82.8

W.O NLEX 57.1 74.2 78.3 45.9 66.2 72.4

9 at inference (overall 10 turns). Full training configurations, including hyperparameters, are provided in Appendix A.3.

### 6. Results

##### 6.1. Output Prediction

CruxEval-O. We start by evaluating performance considering CruxEval-O. Results are presented in Figure 3. We evaluate both Qwen2.5-3B and Qwen2.5-7B (after SFT only), trained with and without the NLEX data. For reference we provide pass@1 scores of common open-weights LLMs. Results show a clear superiority of the NLEX data as part of the training mix, achieving comparable performance to significantly larger models, with Qwen2.5-3B increasing from 37.5 to 68.0 and Qwen2.5-7B improving 48.5 to 75.5 pass@1 scores. We provide results for standard coding metrics in Appendix A.1, showing no regression in performance considering other benchmarks and tasks.

Competitive programming. Next, we evaluate output prediction performance on LLM solutions to competitive programming questions from LCB-IO and DMC (test split). Compared to CruxEval-O, these functions are often more complex and challenging. For that, we consider posttrained Qwen2.5 models (3B and 7B) on the task of outputprediction. Similarly to before, we consider models trained with and without NLEX data as part of the mix. Results presented in Table 1 suggest that including NLEX data as part of the mix boosts output-prediction capabilities also after RL. While RL on output prediction with a standard reasoning SFT data (i.e., OMR and OCR) shows impressive performance, mixing them with NLEX provides superior results across both model sizes. To understand the effect of the RL phase, we additionally evaluate CWM on output prediction with and without RL. As expected, the RL phase significantly improves results on output prediction, see Appendix A.1.2 and Table 8 for more details.

Self-execution prediction. So far we evaluated outputprediction capabilities on code generated by external models. We now turn to evaluate self-execution, i.e. models perform output-prediction on their own generated solutions. For that we post-train CWM and Qwen2.5-7B on both output prediction and competitive problems solving. We report results for questions derived from both LCB-IO and DMC in

###### Qwen-7B on DMCValTest

###### Qwen-7B on LCB-IO

###### CWM on DMCValTest

CWM on LCB-IO

0.60

0.76

- 0.600

pass@k

0.68

best@k exec (oracle)

0.74

0.575

0.58

best@k simulate (ours)

0.67

short1@k

0.72

0.550

Simulation Gap

0.56

0.66

0.70

0.525

0.54

0.65

0.68

pass

0.500

0.64

0.66

0.52

0.475

0.64

0.63

0.50

0.450

0.62

0.62

0.48

0.425

0.60

0.61

1 2 3 4 5 6 7 8 9 10

1 2 3 4 5 6 7 8 9 10

1 2 3 4 5 6 7 8 9 10

1 2 3 4 5 6 7 8 9 10

k

k

k

k

- Figure 4. Best@k performance of self-verification with self-simulation. Solutions and output predictions are produced by the same model based on Qwen2.5-7B or CWM, trained for both solving and output prediction. Even though the tests used for filtering are in the solve prompt, there is still room for notable gains from simulating them.

Table 2. Output prediction performance for models trained on standard code solving, jointly with output prediction (Joint), on their own solutions. We compare this to a model trained for output prediction only, models from Tab. 1, (Out Pred), and official CWM.

MODEL RL OBJ. DMC-OUT LCB-IO-OUT @1 @5 @1 @5

CWM OFFICIAL 57.7 80.4 68.6 87.9 JOINT 80.2 87.2 86.5 91.0 OUT PRED 85.0 89.8 88.6 92.7

QWEN-7B JOINT 68.4 83.1 76.5 87.1 OUT PRED 74.6 86.8 80.1 89.2

- Table 2 and compare performance to models trained to perform output-prediction only (as a topline) and to the official CWM model (as a baseline). As expected, results suggest that jointly training both models for solving competitive programming questions and output prediction perform worse than output prediction only. For instance, CWM reaches 80.2 and 86.5 pass@1 scores in joint training compared to 85.0 and 88.6 scores when trained for output prediction only. However, both are significantly superior to the official CWM model, that reaches 57.7 pass@1. Interestingly, these results suggest that unlike previous findings (Gu et al.,

- 2024a) models can reliably perform self-execution.

##### 6.2. Self-Execution for Competitive Programming

Self-verification. Given a model’s prediction of the execution output of its own code on public tests, one can use this to self-verify the solutions. Specifically, following the best@k simulate approach described in Section 3, we select and submit the solution predicted to pass most tests. To better estimate the effectiveness of the proposed method, we compare it to short1@k (Hassid et al., 2025), which selects the shortest response among the k solutions, and pass@k

(for reference). To directly assess the quality of execution simulation, we also compare against an oracle that executes the public tests, following the same filtering procedure (denoted best@k exec). This comparison will provide us with the simulation gap, i.e., the performance gap between fully executing the code vs. simulating it with the model. Our results provided in Fig. 4 show that self-verification provides a large boost in performance under the best@k setup, (2 − 8 points compared to standard solving), despite the tests used for filtering being provided when generating the solution. This also outperforms short1@k. Notably, for Qwen2.5-7B the simulation gap is larger than CWM perhaps implying the need for larger or stronger models to learn to both solve and simulate execution effectively. Further results in Fig. 9 show that using Qwen2.5-7B trained for output prediction only to filter the same solutions leads to a smaller gap.

Self-RLEF. We train our model using the procedure described in Section 4. We report pass@k scores for k ∈ {1,5,10} on both LCB-IO and DMC, along with publictest pass rates. We evaluate three variants: the official CWM model, CWM post-trained specifically for competitive programming (CWM-RL), and CWM jointly optimised for output prediction and competitive programming with execution feedback. The latter is evaluated under the proposed selfRLEF framework, using either simulated execution or real execution as an oracle. In both settings, the model is allowed up to 10 coding turns (initial solution + 9 fix), although in practice, it uses 3.33 turns on average (Appendix A.1.4 provides additional results with less turns). Results in Table 3 show that self-RLEF consistently outperforms both the official CWM and CWM-RL across all settings, improving pass rates on both public and private tests. Compared to the oracle, a performance gap remains, particularly for pass@1, indicating room for improvement in execution prediction. Interestingly, pass@1 scores (with 10 turns) are lower than the corresponding best@10 results shown in Figure 4. We

- Table 3. Solve rates for training and evaluating with a standard reasoning approach vs using real or simulated execution feedback.

DMC LCB-IO

PASS@1 PASS@5 PASS@10 PUBLIC PASS@1 PASS@5 PASS@10 PUBLIC

EXECUTION RLEF (ORACLE) 65.3 77.6 80.6 86.1 63.8 70.9 72.8 88.5 CWM (OFFICIAL) 49.0 63.7 67.9 60.8 57.4 67.3 70.1 71.4 CWM RL 60.8 72.8 76.0 74.7 61.0 67.6 69.2 82.9 SELF-RLEF (OURS) 63.2 76.8 80.2 82.5 62.3 70.0 71.9 87.1

- Table 4. Comparing performance of using the self-RLEF scaffold

Table 5. Pass rates of the initial generated solution (Init), compared to the final submitted solution (Sub) in Self-RLEF inference on DMC considering both public and private tests.

- at inference only with open source reasoning models.

DMC LCB-IO

(a) Public Tests Init\Sub Fail Pass

(b) Private Tests Init\Sub Fail Pass

MODEL \ PASS @1 @5 @10 @1 @5 @10 QWEN3-32B 44.7 61.4 66.0 58.6 68.9 72.2

+ SRLEF (∆) -10.6 -2.2 -1.4 -20.1 -1.0 -0.1 CWM 49.0 63.7 67.9 57.4 67.3 70.1 + SRLEF (∆) -4.8 +0.5 +0.9 -7.4 +0.1 +0.2

Fail 16.3% 17.0% Pass 1.2% 65.5%

Fail 31.8% 10.4% Pass 5.0% 52.8%

hypothesise that this gap arises from limited exploration, as the model tends to iteratively fix a solution rather than explore alternative ones. In addition, the model frequently does not use all turns as seen by the average number of turns. For certain settings where an ”early exit” is preferred this approach can provide a better tradeoff considering compute.

##### 6.3. Ablations

Self-RLEF scaffold. One may wonder to what extent the self-refinement pipeline itself leads to the inference performance gain irrespective of model training. Hence, we investigate inference using the Self-RLEF approach with public open-weights models, specifically Qwen3-32B and CWM. We compare these results to using these models in a standard single turn inference procedure. Results provided in Table 4 show no noticeable improvement from using the proposed self-RLEF approach, and even a decrease in performance across both models over all metrics and datasets. By manual analysis, we observe the model struggles to correctly predict the output, and ignores the feedback.

Knowing when to submit or fix. To better identify the source of performance gains from self-RLEF, we analyse model behaviour during inference time. We measure how often the model successfully fixes a solution, i.e., cases where the initial solution fails the tests but the final solution passes, as well as how often it breaks a solution, i.e., where an initially passing solution fails after revision. For completeness, we also report cases where both the initial and final solutions either pass or fail the tests. Table 5 reports results for CWM on DMC using both public and private tests. Results suggest the model rarely degrades correct solutions, breaking only

- 1.2% of cases on public tests and 5.0% on private tests. In

contrast, when the initial solution fails, the model frequently produces effective fixes, succeeding in 17.0% of public-tests and 10.4% of private-tests. These improvements increase the pass rate from 57.8% for the initial solutions to 63.2% for the final submissions.

##### 6.4. Beyond Self-Verification

While the focus of this work is self-execution, one could imagine use cases for building a model as a code simulator only. As an initial analysis, we measure public test pass rates for multiple models on LCB-IO and DMC to assess the potential of the best@k approach. Table 6 reports public pass@1 and pass@10 results for Qwen and CWM. While models can generate solutions which pass the public tests (as evidenced in pass@10), they often submit solutions which do not pass them even though they are provided in the question. This suggests that standard reasoning approaches under-utilise such test information. This observation motivates the use of solution verification.

We next assess the ability of our trained CWM model to predict execution outputs for solutions generated by Qwen332B. CWM achieves pass@1 and pass@5 scores of 86.1 and 91.4, respectively, on public test output prediction. Based on this, we apply the best@k evaluation on both LCB-IO and DMC, as shown in Figure 5. The results indicate that using CWM with this filtering strategy is very effective and can correctly filter solutions generated by external models. Furthermore, compared to real execution, we observe only a minor simulation gap. This again highlights the efficacy of the output prediction method to alleviate the need for execution, and further shows generalisation to other models’ solutions. Results for this setup using Qwen3-4B and CWMRL are provided in Appendix A.1, showing similar trends.

###### DMCValTest

###### LCB-IO

0.72

pass@k (topline)

pass@k (topline)

0.65

best@k exec (oracle)

best@k exec (oracle)

0.70

best@k simulate (ours)

best@k simulate (ours)

0.60

short1@k

short1@k

0.68

Simulation Gap

Simulation Gap

0.66

pass

pass

0.55

0.64

0.62

0.50

0.60

0.45

0.58

1 2 3 4 5 6 7 8 9 10

1 2 3 4 5 6 7 8 9 10

k

k

- Figure 5. Comparing best@k when ranking Qwen3-32B solutions, using CWM post-trained only for output prediction as a verifier.

- Table 6. Public pass@1 and 10 of different models. The large gap of standard reasoning models can suggest that they under-utilise provided test information.

DMC LCB-IO

MODEL \ PUBLIC PASS @1 @10 @1 @10 QWEN3-4B 42.5 65.4 64.4 80.9 QWEN3-32B 56.3 79.0 72.3 88.8 QWEN2.5-7B RL 55.1 76.5 68.0 84.7 CWM RL 73.4 87.6 81.7 90.6

### 7. Related Work

Code Simulation & Verification. Several works ask how well LLMs can simulate or predict the output of a given code snippet (Hora, 2024; Li et al., 2025; Gu et al., 2024b; Xu et al., 2025; Copet et al., 2025; Armengol-Estap´e et al.,

- 2025). Others suggest that models struggle to simulate their own code, as they are blind to its flaws (Gu et al., 2024a). Some works use models to simulate tool execution as part of a synthetic data generation (Kimi et al., 2025). Furthermore, some studies explicitly train a model as a verifier of solutions (Le et al., 2022). Many report challenges in verification performance (Ruan et al., 2025; Wang et al., 2025).

Learning from Feedback. Gehring et al. (2025) showed that models can learn to utilise feedback about the execution of their generated code. Providing models with access to interpreters is a popular approach that has been used to improve performance in maths (Chen et al., 2023b; Gao

- et al., 2023), code generation (Liu et al., 2023b; Shinn et al., 2023), competitive programming (Zheng et al., 2025), and agentic coding (Yang et al., 2024; Xia et al., 2025). Several prompting approaches were suggested for non-reasoning models to elicit self-improvement, often joint with external verification signals (Chen et al., 2023c; Renze & Guven, 2024; Madaan et al., 2023; Kumar et al., 2024). Chen et al.

(2023a) further showed that training with human written feedback on code can improve performance.

### 8. Discussion

Limitations. The main limitation of simulating program execution is estimating complex computational operations (e.g., multiplying large numbers, compute logarithms, etc.). Yet, while execution simulation is imperfect and can introduce noise, our findings suggest that it provides a useful inductive bias for reasoning about program behaviour, particularly in settings where direct execution is expensive or infeasible. Furthermore, while our approach showed promising results, it is currently limited to single file competitive programming questions. Generalising this to full repository SWE tasks poses an interesting future research direction.

Future Work. We believe our work opens several directions for future research. The most interesting direction in our opinion is using the full rich execution simulation, and not only the final output as feedback for iterative code fixing. Such feedback can convey richer information than the output alone (even beyond real execution), capturing not just what output is produced, but why it arises. Such explanations can reveal cases where a test appears to pass for incidental reasons, as well as provide insight into the underlying causes of failures. In preliminary results we observe that training with rich textual feedback presents challenges to training stability. We hypothesise this is due to both inability to train with teacher forcing and unclear definition of the verifiable reward of the simulation. We leave such exploration for future work.

### 9. Conclusion

In this work we investigated if LLMs can be trained to simulate code execution and whether this capability can be used

to improve code generation. By combining SFT on NLEX with RLVR, we showed that models can acquire the ability to predict execution outcomes for general programs as well as code they generate. Leveraging this ability, we introduced self-verification and iterative self-fix strategies using predicted execution signals to select or refine candidate solutions without relying on external execution. Our empirical results on competitive programming tasks demonstrate consistent improvements over standard baselines considering both output prediction and question solving. Compared with real execution we show a relatively small simulation gap, demonstrating the usability of the proposed approach compared to the top-line of code execution. More broadly, our results suggest that enabling models to reason about the outcomes of the code they generate may be a key for building more reliable programming agents.

### Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

### References

Ahmad, W. U., Narenthiran, S., Majumdar, S., Ficek, A., Jain, S., Huang, J., Noroozi, V., and Ginsburg, B. Opencodereasoning: Advancing data distillation for competitive coding. arXiv preprint arXiv:2504.01943, 2025.

Armengol-Estap´e, J., Carbonneaux, Q., Zhang, T., Markosyan, A. H., Seeker, V., Cummins, C., Kambadur, M., O’Boyle, M. F., Wang, S., Synnaeve, G., et al. What i cannot execute, i do not understand: Training and evaluating llms on program execution traces. arXiv preprint arXiv:2503.05703, 2025.

Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., et al. Program synthesis with large language models, 2021. URL https://arxiv.org/abs/2108.

07732. arXiv:2108.07732.

Beck, M., Gehring, J., Kossen, J., and Synnaeve, G. Towards a neural debugger for python, 2026. URL https:// arxiv.org/abs/2603.09951.

Bengio, S., Vinyals, O., Jaitly, N., and Shazeer, N. Scheduled sampling for sequence prediction with recurrent neural networks. Advances in neural information processing systems, 28, 2015.

Bogin, B., Yang, K., Gupta, S., Richardson, K., Bransom, E., Clark, P., Sabharwal, A., and Khot, T. Super: Evaluating

agents on setting up and executing tasks from research repositories. arXiv preprint arXiv:2409.07440, 2024.

Chan, J. S., Chowdhury, N., Jaffe, O., Aung, J., Sherburn, D., Mays, E., Starace, G., Liu, K., Maksin, L., Patwardhan, T., et al. Mle-bench: Evaluating machine learning agents on machine learning engineering. arXiv preprint arXiv:2410.07095, 2024.

Chen, A., Scheurer, J., Korbak, T., Campos, J. A., Chan, J. S., Bowman, S. R., Cho, K., and Perez, E. Improving code generation by training with natural language feedback. arXiv preprint arXiv:2303.16749, 2023a.

- Chen, W., Ma, X., Wang, X., and Cohen, W. W. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. Transactions on Machine Learning Research, 2023b. ISSN 2835-

8856. URL https://openreview.net/forum? id=YfZ4ZPt8zd.

- Chen, X., Lin, M., Sch¨arli, N., and Zhou, D. Teaching large language models to self-debug. arXiv preprint arXiv:2304.05128, 2023c.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Copet, J., Carbonneaux, Q., Cohen, G., Gehring, J., Kahn, J., Kossen, J., Kreuk, F., McMilin, E., Meyer, M., Wei, Y., et al. Cwm: An open-weights llm for research on code generation with world models. arXiv preprint arXiv:2510.02387, 2025.

Ding, J., Zhang, Y., Shang, Y., Zhang, Y., Zong, Z., Feng, J., Yuan, Y., Su, H., Li, N., Sukiennik, N., et al. Understanding world or predicting future? a comprehensive survey of world models. ACM Computing Surveys, 58(3):1–38, 2025.

Gao, L., Madaan, A., Zhou, S., Alon, U., Liu, P., Yang, Y., Callan, J., and Neubig, G. PAL: Program-aided language models. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 10764–10799. PMLR, 23–29 Jul 2023. URL https://proceedings.mlr.press/ v202/gao23f.html.

Gehring, J., Zheng, K., Copet, J., Mella, V., Cohen, T., and Synnaeve, G. RLEF: Grounding code LLMs in execution feedback with reinforcement learning. In Fortysecond International Conference on Machine Learning,

2025. URL https://openreview.net/forum? id=PzSG5nKe1q.

Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Gu, A., Li, W.-D., Jain, N., Olausson, T., Lee, C., Sen, K., and Solar-Lezama, A. The counterfeit conundrum: Can code language models grasp the nuances of their incorrect generations? In Findings of the Association for Computational Linguistics ACL 2024, pp. 74–117, 2024a.

Gu, A., Roziere, B., Leather, H. J., Solar-Lezama, A., Synnaeve, G., and Wang, S. CRUXEval: A benchmark for code reasoning, understanding and execution. In Salakhutdinov, R., Kolter, Z., Heller, K., Weller, A., Oliver, N., Scarlett, J., and Berkenkamp, F. (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 16568–16621. PMLR, 21– 27 Jul 2024b. URL https://proceedings.mlr.

press/v235/gu24c.html. Ha, D. and Schmidhuber, J. World models. arXiv preprint arXiv:1803.10122, 2(3), 2018.

Hassid, M., Remez, T., Gehring, J., Schwartz, R., and Adi, Y. The larger the better? improved llm code-generation via budget reallocation. arXiv preprint arXiv:2404.00725,

- 2024.

Hassid, M., Synnaeve, G., Adi, Y., and Schwartz, R. Don’t overthink it. preferring shorter thinking chains for improved llm reasoning. arXiv preprint arXiv:2505.17813,

- 2025.

Hora, A. Predicting test results without execution. In Companion Proceedings of the 32nd ACM International Conference on the Foundations of Software Engineering, pp. 542–546, 2024.

Jain, N., Han, K., Gu, A., Li, W.-D., Yan, F., Zhang, T., Wang, S., Solar-Lezama, A., Sen, K., and Stoica, I. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024.

Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., and Narasimhan, K. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023.

Kamoi, R., Zhang, Y., Zhang, N., Han, J., and Zhang, R. When can llms actually correct their own mistakes? a critical survey of self-correction of llms. Transactions of the Association for Computational Linguistics, 12:1417– 1440, 2024.

Kimi, T., Bai, Y., Bao, Y., Chen, G., Chen, J., Chen, N., Chen, R., Chen, Y., Chen, Y., Chen, Y., et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.

Kumar, A., Zhuang, V., Agarwal, R., Su, Y., Co-Reyes, J. D., Singh, A., Baumli, K., Iqbal, S., Bishop, C., Roelofs, R., et al. Training language models to self-correct via reinforcement learning. arXiv preprint arXiv:2409.12917, 2024.

Le, H., Wang, Y., Gotmare, A. D., Savarese, S., and Hoi, S. C. H. Coderl: Mastering code generation through pretrained models and deep reinforcement learning. Advances in Neural Information Processing Systems, 35: 21314–21328, 2022.

Li, J., Guo, D., Yang, D., Xu, R., Wu, Y., and He, J. CodeIO: Condensing reasoning patterns via code inputoutput prediction. In Forty-second International Conference on Machine Learning, 2025. URL https: //openreview.net/forum?id=feIaF6vYFl.

Li, Y., Choi, D., Chung, J., Kushman, N., Schrittwieser, J., Leblond, R., Eccles, T., Keeling, J., Gimeno, F., Dal Lago, A., et al. Competition-level code generation with alphacode. Science, 378(6624):1092–1097, 2022.

Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.

Liu, J., Xia, C. S., Wang, Y., and Zhang, L. Is your code generated by chatGPT really correct? rigorous evaluation of large language models for code generation. In Thirtyseventh Conference on Neural Information Processing Systems, 2023a. URL https://openreview.net/ forum?id=1qvx610Cu7.

Liu, Z., Zhang, Y., Li, P., Liu, Y., and Yang, D. Dynamic llmagent network: An llm-agent collaboration framework with agent team optimization, 2023b.

Madaan, A., Tandon, N., Gupta, P., Hallinan, S., Gao, L., Wiegreffe, S., Alon, U., Dziri, N., Prabhumoye, S., Yang, Y., et al. Self-refine: Iterative refinement with selffeedback. Advances in Neural Information Processing Systems, 36:46534–46594, 2023.

MiniMax. M2.1: Multilingual and multi-task coding with strong generalization. https://x.com/MiniMax_ _AI/status/2007843119832695114, January 4 2026.

Moshkov, I., Hanley, D., Sorokin, I., Toshniwal, S., Henkel, C., Schifferer, B., Du, W., and Gitman, I. Aimo-2 winning solution: Building state-of-the-art mathematical reasoning models with openmathreasoning dataset. arXiv preprint arXiv:2504.16891, 2025.

Olausson, T. X., Inala, J. P., Wang, C., Gao, J., and Solar-Lezama, A. Is self-repair a silver bullet for code generation? In The Twelfth International Conference on Learning Representations, 2024. URL https:// openreview.net/forum?id=y0GJXRungR.

Peng, Y., Gotmare, A. D., Lyu, M. R., Xiong, C., Savarese, S., and Sahoo, D. Perfcodegen: Improving performance of llm generated code with execution feedback. In 2025 IEEE/ACM Second International Conference on AI Foundation Models and Software Engineering (Forge), pp. 1–13, 2025. doi: 10.1109/Forge66646.2025.00008.

Qian, C., Acikgoz, E. C., Li, B., Chen, X., Zhang, Y., He, B., Luo, Q., Hakkani-T¨ur, D., Tur, G., Li, Y., et al. Current agents fail to leverage world model as tool for foresight. arXiv preprint arXiv:2601.03905, 2026.

Qwen, :, Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L., Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R., Li, T., Tang, T., Xia, T., Ren, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Wan, Y., Liu, Y., Cui, Z., Zhang, Z., and Qiu, Z. Qwen2.5 technical report, 2025. URL https:

//arxiv.org/abs/2412.15115.

Renze, M. and Guven, E. Self-reflection in llm agents: Effects on problem-solving performance. arXiv preprint arXiv:2405.06682, 2024.

Ruan, C., Jiang, D., Wang, Y., and Chen, W. Critiquecoder: Enhancing coder models by critique reinforcement learning. arXiv preprint arXiv:2509.22824, 2025.

Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K. R., and Yao, S. Reflexion: language agents with verbal reinforcement learning. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https: //openreview.net/forum?id=vAElhFcKW6.

Wang, Y., Yue, X., and Chen, W. Critique fine-tuning: Learning to critique is more effective than learning to imitate. arXiv preprint arXiv:2501.17703, 2025.

Xia, C. S., Wang, Z., Yang, Y., Wei, Y., and Zhang, L. Liveswe-agent: Can software engineering agents self-evolve on the fly? arXiv preprint arXiv:2511.13646, 2025.

Xu, R., Cao, J., Lu, Y., Wen, M., Lin, H., Han, X., He, B., Cheung, S.-C., and Sun, L. CRUXEVALX: A benchmark for multilingual code reasoning, understanding and execution. In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 23762–23779, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.1158. URL https:// aclanthology.org/2025.acl-long.1158/.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Yang, J., Jimenez, C. E., Wettig, A., Lieret, K., Yao, S., Narasimhan, K., and Press, O. Swe-agent: Agentcomputer interfaces enable automated software engineering. Advances in Neural Information Processing Systems, 37:50528–50652, 2024.

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., and Cao, Y. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022.

- Zheng, J., Zhang, J., Luo, Y., Mao, Y., Gao, Y., Du, L., Chen, H., and Zhang, N. Can we predict before executing machine learning agents? arXiv preprint arXiv:2601.05930, 2026.
- Zheng, K., Decugis, J., Gehring, J., Cohen, T., benjamin negrevergne, and Synnaeve, G. What makes large language models reason in (multi-turn) code generation? In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview. net/forum?id=Zk9guOl9NS.

Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Thimmaiah, A., Zhang, J., Srinivasa, J., Li, J. J., and Gligoric, M. Plsemanticsbench: Large language models

- as programming language interpreters. arXiv preprint arXiv:2510.03415, 2025.

### A. Appendix.

##### A.1. Additional Results

- A.1.1. SUPERVISED FINE-TUNING.

To confirm that our NLEX data mix does not negatively impact the performance of the model on general tasks at the expense of boosting output prediction, we look at several standard coding and maths benchmarks. Specifically we consider CruxEval-Input (Gu et al., 2024b), MBPP (Austin et al., 2021), HumanEval Plus (Liu et al., 2023a), LiveCodeBench v5 (Jain et al., 2024), GSM8k (Cobbe et al., 2021), and Math 500 (Lightman et al., 2023). As reported in Table 7, using the NLEX data mix does not notably harm any metric, and even improves output prediction abilities as noted by CruxEval-Input.

Table 7. Investigating the impact of the NLEX data mix compared to a standard reasoning and instruction following mix on various standard coding and maths benchmarks. All models are trained with supervised fine tuning for the same budget only changing the data.

Model CruxEval-In MBPP HumanEval+ LCBv5 GSM8k Math 500 Qwen2.5-7B regular mix 0.469 0.634 0.652 0.414 0.842 0.518

+ NLEX 0.505 0.632 0.659 0.413 0.826 0.528 Qwen2.5-3B regular mix 0.361 0.522 0.543 0.195 0.748 0.398

+ NLEX 0.445 0.524 0.537 0.203 0.729 0.406

- A.1.2. THE EFFECT OF RL ON OUTPUT PREDICTION

To better evaluate the effect of additional RL phase on top of the SFT, we evaluate CWM model, trained with and without RL considering output prediction only. For a reference, we additionally include results for the official post-trained CWM model. Results are reported in Table 8. As expected, results suggest that the RL phase significantly improve results on output prediction over competitive programming questions. We omit Qwen results as without RL, their performance was significantly lower.

Table 8. Output prediction performance of the CWM with and without RL. For reference we additionally report results for the official CWM model.

BASE MODEL LCB-IO-OUT DMC-TEST-OUT

PASS@1 @5 @10 @1 @5 @10 CWM (OFFICIAL) 60.4 79.3 82.0 68.9 85.8 87.8 CWM (WO. RL) 30.3 55.6 62.1 38.4 67.5 73.5 CWM (W. RL) 89.6 93.4 94.2 89.2 93.3 94.0

- A.1.3. SELF-VERIFICATION

To further analyse the impact if self-verification using self-execution simulation, we wanted to study a setup where the tests used for verification were not present at the time of generating the solutions. To that end, we generate solutions using a model trained jointly for output prediction and competitive programming solving, but without tests in the question description for training and inference. This represents a case where the tests for verification contain completely new information unseen when generating the solution. Results are provided in Figure 6. These suggest that while removing the tests from the description has a negative notable impact on performance, much of the performance can be gained by filtering solutions using these tests. It can also suggest that tests not used for generating the solution could have a higher positive impact for verification, motivating future investigation of test generation.

- A.1.4. SELF-RLEF

By default we allow a maximum of 10 turns for execution RLEF, and Self-RLEF. However in practice the model often submits its solution prior leading to an average of 3.33 turns. We wish to consider the performance of Self-RLEF when limiting the number of solve turns to a maximum of 3. We provide results in Table 9. In practice the model uses an average of 2.38 turns.

###### DMCValTest

###### LCB-IO

0.68

pass@k

pass@k

0.700

best@k exec (oracle)

best@k exec (oracle)

0.675

best@k simulate (ours)

best@k simulate (ours)

0.66

short1@k

short1@k

0.650

Simulation Gap

Simulation Gap

0.64

0.625

pass

pass

0.600

0.62

0.575

0.550

0.60

0.525

1 2 3 4 5 6 7 8 9 10

1 2 3 4 5 6 7 8 9 10

k

k

- Figure 6. Comparing best@k when ranking solutions generated by CWM post-trained jointly for solving and output prediction, using the same model as a verifier. The model here was trained and evaluated without the public tests as part of the description.

- Table 9. Solve rates when using real or simulated execution feedback, but limiting to 3 turns. This extends Table 3 under a more compute constraint setup.

DMC LCB-IO

PASS@1 PASS@5 PASS@10 PUBLIC PASS@1 PASS@5 PASS@10 PUBLIC

SELF-RLEF (OURS) 61.5 75.6 79.6 79.0 61.5 69.2 71.1 84.2 EXECUTION RLEF (ORACLE) 62.7 75.8 78.8 81.5 63.3 70.3 72.2 86.3

- A.1.5. BEYOND Self -VERIFICATION

We provide results for using a dedicated output prediction model as a tool for verifying solutions of other models in a best@k setup. Results provided in Figures 7 and 8 show consistent improvements from this approach, for both Qwen3-4B and CWM Solve-RL, with only a slight degradation compared to ground truth execution of these tests. Like the results for Qwen3-32B in the main paper this further demonstrates the efficacy of this approach.

###### DMCValTest

###### LCB-IO

0.62

pass@k

pass@k

0.500

best@k exec (oracle)

best@k exec (oracle)

0.60

0.475

best@k simulate (ours)

best@k simulate (ours)

short1@k

short1@k

0.450

0.58

Simulation Gap

Simulation Gap

0.425

pass

pass

0.56

0.400

0.54

0.375

0.350

0.52

0.325

1 2 3 4 5 6 7 8 9 10

1 2 3 4 5 6 7 8 9 10

k

k

- Figure 7. Comparing best@k when ranking Qwen3-4B solutions, using CWM post-trained only for output prediction as a verifier.

We also provide results of using a dedicated verifier based on a smaller model (Qwen2.5-7B), on solutions generated by a model starting from the same base model. Results provided in Figure 9 show that this method is also effective with models

- at this scale. This outperforms the performance in Figure 4 which suggests that the constraint of having the same model for solving and verification does impose challenges especially with models with limited capacity.

###### DMCValTest

###### LCB-IO

0.69

pass@k

pass@k (topline)

0.74

0.68

best@k exec (oracle)

best@k exec (oracle)

0.72

best@k simulate (ours)

best@k simulate (ours)

0.67

short1@k

short1@k

0.70

0.66

Simulation Gap

Simulation Gap

0.68

0.65

pass

pass

0.64

0.66

0.63

0.64

0.62

0.62

0.61

0.60

1 2 3 4 5 6 7 8 9 10

1 2 3 4 5 6 7 8 9 10

k

k

- Figure 8. Comparing best@k when ranking solutions by CWM post-trained only for competitive programming solving (denoted SOLVERL in Table 3), using CWM post-trained only for output prediction as a verifier.

1 2 3 4 5 6 7 8 9 10

k

0.425

0.450

0.475

0.500

0.525

0.550

0.575

0.600

pass

DMCValTest

pass@k (topline)

best@k exec (oracle)

best@k simulate (ours)

short1@k

Simulation Gap

1 2 3 4 5 6 7 8 9 10

k

0.48

0.50

0.52

0.54

0.56

0.58

0.60

pass

LCB-IO

pass@k (topline)

best@k exec (oracle)

best@k simulate (ours)

short1@k

Simulation Gap

- Figure 9. Comparing best@k when ranking solutions by Qwen-7B post-trained for competitive programming solving, using Qwen-7B post-trained only for output prediction as a verifier. This mirrors the results for Qwen in Figure 4, but when each model has a dedicated role.

##### A.2. Self-RLEF Example Inference

To demonstrate how the iterative self-fixing and self verification looks in practice in Self-RLEF we provide the (abbreviated) multi-turn inference for a successful LCB-IO solution using our model.

##### Turn 1 - (Abbreviated) Solve Prompt

Provide a Python solution for [...]: You are given two strings s and t, both consisting of lowercase English letters. You are going to type the string s character by character, from the first character to the last one.

When typing a character, instead of pressing the button corresponding to it, you can press the "Backspace" button. It deletes the last character you have typed among those that aren’t deleted yet (or does nothing if there are no characters in the current string). For example, if s is "abcbd" and you press Backspace instead of typing the first and the fourth characters, you will get the string "bd" (the first press of Backspace deletes no character, and the second press deletes the character ’c’). Another example, if s is "abcaa" and you press Backspace instead of the last two letters, then the resulting text is "a". Your task is to determine whether you can obtain the string t, if you type the string s and press "Backspace" instead of typing several (maybe zero) characters of s.[...]

Output For each test case, print "YES" if you can obtain the string t by typing the string s

and replacing some characters with presses of "Backspace" button, or "NO" if not.

--------------- Example Tests------------

----- Test 0 ----## Input ‘4\nababa\nba\nababa\nbb\naaa\naaaa\naababa\nababa\n‘

## Expected Output ‘YES\nNO\nNO\nYES\n‘.

##### Turn 1 - (Abbreviated) Solve Response

[...Reasoning...] import sys def main():

q = int(sys.stdin.readline()) for _ in range(q):

- s = sys.stdin.readline().strip()
- t = sys.stdin.readline().strip() if len(t) > len(s):

print("NO") continue

current_states = {’’} for c in s:

new_states = set() for state in current_states:

- # Option 1: type the character new_state = state + c new_states.add(new_state)
- # Option 2: press backspace if len(state) > 0:

new_state_back = state[:-1] new_states.add(new_state_back)

current_states = new_states

print("YES" if t in current_states else "NO") if __name__ == "__main__":

main()

##### Turn 2 - (Abbreviated) Simulation Response

[...Simulation Reasoning...] For each state in current_states (which is [’’]), we have two options:

- - Type ’a’: new state is ’’ + ’a’ = ’a’
- - Press backspace: since state is empty, can’t do this. [...] State ’’:
- - Type ’a’: ’’ + ’a’ = ’a’
- - Press backspace: can’t, since state is empty. [...] Therefore, the final answer is:</think> <output>NO\nNO\nNO\nNO\n</output>

##### Turn 3 - (Abbreviated) Fix or Submit Prompt

[...] Review the provided code solution given the execution feedback and judge if it is

correct or buggy. If the code is correct, end your response with a markdown comment EXACTLY ‘‘‘#SUBMIT‘‘‘. DO NOT repeat the given code. If the code is incorrect, provide a correct solution. [...]

Problem: [...] Attempted Solution: [...]

----- Test 0 -----

## Input ‘4\nababa\nba\nababa\nbb\naaa\naaaa\naababa\nababa\n‘

## Expected Output ‘YES\nNO\nNO\nYES\n‘

## Output for the attempt ‘NO\nNO\nNO\nNO\n‘

##### Turn 3 - (Abbreviated) Fix or Submit Prompt

[...Reasoning...] But according to the attempted solution, the output for this test case is NO, but the

expected output is YES. So the attempted solution is not working correctly. [...] Wait, here’s a problem! Ah! Here’s the issue. In the attempted solution, when

considering pressing Backspace, the code checks if len(state) > 0. If so, it adds state[:-1]. But when the state is empty, pressing Backspace does nothing, so the new

state should still be empty. However, the code does not consider this case. [...] ‘‘‘import sys def main():

q = int(sys.stdin.readline()) for _ in range(q):

- s = sys.stdin.readline().strip()
- t = sys.stdin.readline().strip() if len(t) > len(s):

print("NO") continue

current_states = {’’} for c in s:

new_states = set() for state in current_states:

- # Option 1: type the character new_state = state + c new_states.add(new_state)
- # Option 2: press backspace new_state_back = state[:-1] new_states.add(new_state_back)

current_states = new_states

print("YES" if t in current_states else "NO") if __name__ == "__main__":

main()‘‘‘

##### Turn 4 - (Abbreviated) Simulation Response

[...Simulation Reasoning...] Therefore, the output is as I determined. <output>YES\nNO\nNO\nYES\n</output>

##### Turn 5 - (Abbreviated) Fix or Submit Response

[...Reasoning...] The given code seems to work for the provided test cases, as the output matches the

expected output. [...] Given that the code passes the provided test cases and the logic seems correct, I would

judge the code as correct.</think> ‘‘‘#SUBMIT‘‘‘

##### A.3. Hyper-Parameters

Supervised Fine-Tuning. All Qwen supervised fine-tuning use a sequence length of 65,536 by applying scaled RoPE (Su et al., 2024) with a factor of two relative to the base models to support longer context. CWM uses a context length of 131,072 like in the original paper. Models were trained for 15.5k steps, with a batch size of 4M tokens per-update step, for a total of 65B tokens. They were trained using a peak learning rate of 8e − 6 after a warmup of 1k steps. The estimated compute per-training run is 7.9e21 FLOPs for 7B, and 5.0e21 for 3B. Both models were trained for ∼20 hours on 128 and 64 NVIDIA H100 GPUs respectively.

Reinforcement Learning. We train the models on NVIDIA H100 GPUs, with a standard configuration of 192 GPUs for a single training run of CWM, and 86 for Qwen 7B and 3B. Typically 1/3 of the GPUs are used as trainers and the rest for rollouts. By default, we employ the maximum context of the model from SFT for generation, packing training sequences by maximum of 131,072 tokens, use a global batch size of 1M tokens, a group size of 8, discarding rollouts with a staleness of more than 8 off-policy steps. We train the CWM models for 10k update steps, and the Qwen models for 4k, as we noted loops and collapses with longer training. This corresponds to roughly 9B and 3.2B tokens respectively. We use the last checkpoint for CWM, as training was stable, and the best checkpoint based on pass@1 by DMCValidation for Qwen (at 200 step increments) as the training was more prone to degradation in the end of training. We use 400 steps of linear learning rate warmup to a peak 1.4e − 7, with gradient clipping at 0.1. For single turn solving jointly with output prediction we sample output prediction at 15% of the time while the rest is for solving. For Self-RLEF we increase this ratio to 25%.

For sampling in evaluation we compare temperature 0.6 and 1.0, with top-p 0.95 as these were common values for Qwen and CWM. We select the best per-model based on DMC pass@1 rates. For CWM results didn’t change notably for all training setups, and yet for Qwen with temperature of 0.6 there were many loops leading to not finishing rollouts, this could be to the smaller model size. Thus for all Qwen models we use temperature 1.0, as well as for all CWM models except of the results with two fixing turns which performed slightly better with 0.6.

##### A.4. Prompts

As mentioned in Section 2.1 the data is created by converting raw traces to natural language by prompting an LLM, followed by a verification procedure. Below we provide the prompt used for the conversion.

##### Trace to Natural Language Prompt

You are an expert programmer tasked with explaining the step-by-step execution of a Python code snippet based on a provided execution trace. Focus and explain the specific values of variables at each step, not just vaguely say

what the code does. Be specific about what the values of variables are. Note that the code could have bugs making it NOT do what the names suggest! The trace shows <local> and <global> variables at each step, only where the values

change, and the current <code> line. Explain the provided execution. NEVER refer to or mention the trace itself. The output should ONLY be the execution explanation. DO NOT use <> tags. You should adhere strictly to the trace for the execution, even if

there is a mistake. Do NOT suggest code fixes, even if the function is incorrect. Finally, in addition to the explanation, provide the correct function output formatted

as a valid Python literal, so that it can be easily verified using an assert statement (e.g. assert foo(inputs) == output).

For example, if the output is a string, it should be enclosed in quotes; if it’s a list

it should be enclosed in square brackets and if a tuple in parentheses. We have the following code: ‘‘‘ {source_code} ‘‘‘ And this line by line trace execution of running {func_name}({input_str}): {stack_trace_string}

Explain the line by line execution of {func_name}({input_str}), followed by the correct output of the function.

The output format should be: [EXPLANATION]EXECUTION_EXPLANATION[/EXPLANATION] [OUTPUT] EXECUTION_OUTPUT[/OUTPUT], where EXECUTION_OUTPUT is the result of running { func_name}({input_str}).

- A.5. Data Samples We provide examples from our NLEX data below.

##### Ex. 1 - Code + Input

[PYTHON] def maxSubArrayDP(arr):

dp = [0] * len(arr) dp[0] = arr[0] result = arr[0] for i in range(1, len(arr)):

dp[i] = max(arr[i], dp[i - 1] + arr[i]) result = max(result, dp[i])

return result assert maxSubArrayDP([1, 0, 0, 0, 0, 0]) == ?? [/PYTHON]

##### Ex. 1 - Natural Language Execution Trace

[THOUGHT] The function is executed as follows:

- 1. The input array is ‘[1, 0, 0, 0, 0, 0]‘. A variable ‘dp‘ is initialized with the same length as the array, filled with 0s.
- 2. The first element of ‘dp‘ is set to the first element of the input array, which is ‘1‘.
- 3. A variable ‘result‘ is also initialized with the first element of the array, which is ‘1‘, representing the current maximum subsequence sum.
- 4. The code enters a loop to process each element in the array starting from the second position.
- 5. At the first iteration (‘i = 1‘), the value of ‘dp[i]‘ is computed as the maximum of the current element ‘0‘ and the sum of the previous ‘dp‘ value ‘1‘ and the current

element. So, ‘dp[1] = 1‘. The ‘result‘ is updated to the maximum of itself and ‘dp [1]‘, remaining ‘1‘.

- 6. At the second iteration (‘i = 2‘), the value of ‘dp[2]‘ is computed similarly as the maximum of ‘0‘ and ‘1 (dp[i-1]) + 0‘. So, ‘dp[2] = 1‘. The ‘result‘ still stays at

‘1‘.

- 7. This process is repeated for the remaining indices (3, 4, and 5).

- - For each, the value of ‘dp[i]‘ at ‘0‘ will be compared with the sum of the previous ‘dp[i-1]‘ value (which was updated in the prior step, incrementally adding

1) and ‘0‘. Thus, ‘dp[i]‘ will become ‘1‘ for each index ‘i >= 2‘.

- - The ‘result‘ continues to remain at ‘1‘ since all updated ‘dp[i]‘ values are equal to the current maximum.

- 8. After all iterations, the loop finishes, and the final value of ‘result‘ is returned , which is the maximum sum of any contiguous subarray of the input array.

[/THOUGHT] [ANSWER] assert maxSubArrayDP([1, 0, 0, 0, 0, 0]) == 1 [/ANSWER]

##### Ex. 2 - Code + Input

[PYTHON] def translate(value, to_min, to_max, from_min, from_max):

’’’

Translate a value from one range to another

’’’ # Figure out how ’wide’ each range is to_span = to_max - to_min from_span = from_max - from_min

spans_decimal = to_span / from_span # Convert the left range into a 0-1 range (float) scaled_value = int(round(float(value - to_min) * spans_decimal, 1) + 1) return scaled_value

assert translate(11, 0, 10, 0, 20) == ?? [/PYTHON]

Ex. 2 - Natural Language Execution Trace

[THOUGHT] The function ‘translate‘ is executed with the arguments 11, 0, 10, 0, 20.

- 1. At the start of the function, the local variables ‘value‘, ‘to_min‘, ‘to_max‘, ‘ from_min‘, and ‘from_max‘ are assigned the values 11, 0, 10, 0, and 20 respectively.
- 2. On the sixth line of the function, the variable ‘to_span‘ is calculated by subtracting ‘to_min‘ from ‘to_max‘, resulting in ‘10 - 0 = 10‘.
- 3. On the seventh line, the variable ‘from_span‘ is calculated by subtracting ‘from_min ‘ from ‘from_max‘, resulting in ‘20 - 0 = 20‘.
- 4. On the ninth line, the variable ‘spans_decimal‘ is calculated by dividing ‘to_span‘ by ‘from_span‘, resulting in ‘10 / 20 = 0.5‘.
- 5. On the twelfth line, a new variable ‘scaled_value‘ is calculated in several steps:

- - First, the difference between ‘value‘ and ‘to_min‘ is calculated, resulting in ‘11
- - 0 = 11‘.
- - Then, this result is multiplied by ‘spans_decimal‘, resulting in ‘11 * 0.5 = 5.5‘.
- - The result is rounded to one decimal place, which may not be consistent with the code: ‘round(5.5, 1)‘ results in 5.5 again.
- - One is then added to this result, giving ‘5.5 + 1 = 6.5‘.
- - Finally, this is cast to an integer, ‘int(6.5) = 6‘, since any decimal part is truncated.

- 6. On the fourteenth line, the ‘scaled_value‘ (6) is returned. [/THOUGHT] [ANSWER] assert translate(11, 0, 10, 0, 20) == 6 [/ANSWER]

Ex. 3 - Code + Input

[PYTHON] def additionLossFunc(x, inc):

y = [] for i in x:

y.append(inc*i*100)

return y assert additionLossFunc([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1.25) == ?? [/PYTHON]

##### Ex. 3 - Natural Language Execution Trace

[THOUGHT] The function starts by initializing an empty list ‘y‘.

It then begins a loop using each element in the list ‘x‘ as the loop variable ‘i‘. In the first iteration, the value of ‘i‘ is 1. The value 1.25 (which is ‘inc‘) is

multiplied by ‘i‘ and then by 100. The result is 125.0, which is appended to the list ‘y‘.

In the second iteration, the value of ‘i‘ is 2. The calculation becomes 1.25 * 2 * 100, which gives 250.0. This is appended to the list ‘y‘.

In the third iteration, ‘i‘ is 3. The value is calculated as 1.25 * 3 * 100, resulting in 375.0, and appended to ‘y‘.

Continuing in the same way for the rest of the elements in the list, the following values are computed and appended to ‘y‘:

- - For ‘i‘ = 4, result is 500.0
- - For ‘i‘ = 5, result is 625.0
- - For ‘i‘ = 6, result is 750.0
- - For ‘i‘ = 7, result is 875.0
- - For ‘i‘ = 8, result is 1000.0
- - For ‘i‘ = 9, result is 1125.0
- - For ‘i‘ = 10, result is 1250.0

After all the iterations, the final list ‘y‘ contains all the computed results from

each iteration. The function returns this list as the output. [/THOUGHT] [ANSWER] assert additionLossFunc([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1.25) == [125.0, 250.0, 375.0,

500.0, 625.0, 750.0, 875.0, 1000.0, 1125.0, 1250.0] [/ANSWER]

