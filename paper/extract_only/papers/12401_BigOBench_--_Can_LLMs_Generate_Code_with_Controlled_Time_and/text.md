# arXiv:2503.15242v2[cs.CL]20Mar2025

## BigO(Bench) - Can LLMs Generate Code with Controlled Time and Space Complexity?

Pierre Chambon1,2, Baptiste Roziere∗, Benoit Sagot2, Gabriel Synnaeve1 1FAIR at Meta, 2Inria

∗Work done at Meta, now working at Mistral AI

We introduce BigO(Bench), a novel coding benchmark designed to evaluate the capabilities of generative language models in understanding and generating code with specified time and space complexities. This benchmark addresses the gap in current evaluations that often overlook the ability of models to comprehend and produce code constrained by computational complexity. BigO(Bench) includes tooling to infer the algorithmic complexity of any Python function from profiling measurements, including human- or LLM-generated solutions. BigO(Bench) also includes of set of 3,105 coding problems and 1,190,250 solutions from Code Contests annotated with inferred (synthetic) time and space complexity labels from the complexity framework, as well as corresponding runtime and memory footprint values for a large set of input sizes. We present results from evaluating multiple state-of-the-art language models on this benchmark, highlighting their strengths and weaknesses in handling complexity requirements. In particular, token-space reasoning models are unrivaled in code generation but not in complexity understanding, hinting that they may not generalize well to tasks for which no reward was given at training time.

Date: March 21, 2025 Correspondence: Pierre Chambon at pchambon@meta.com Code: https://github.com/facebookresearch/bigobench Data: https://huggingface.co/datasets/facebook/BigOBench Homepage: https://facebookresearch.github.io/BigOBench Leaderboard: https://facebookresearch.github.io/BigOBench/leaderboard.html

1 Introduction

A junior developer writes an elegant solution to a coding challenge that passes all test cases, yet fails catastrophically in production. The issue isn’t a bug – it’s a O(n2) algorithm processing millions of records, when a O(n.log(n)) solution could have handled the load effortlessly. As large language models (LLMs) increasingly assist in code generation, their ability to understand and control computational complexity becomes critical. While modern LLMs can generate syntactically correct and functional code with impressive accuracy, our new benchmark BigO(Bench) show that they often struggle with the higher-level reasoning required to optimize time and space complexity – a skill that separates novice programmers from experienced engineers.

Our comprehensive evaluation of state-of-the-art code generation models reveals a concerning gap: while reasoning models like DeepSeek R1 achieve above 70% accuracy (pass@1) on programming contests

(CodeForces), they show significantly weaker performance (4.8%) when tasked with generating solutions under specific complexity constraints (a detailed failure example is in appendix B). They fare barely better than non-reasoning models at analyzing a function for its complexity, e.g. 3.1% better in complexity prediction vs. Llama 3.1 405B. This limitation becomes particularly acute in real-world applications, where scalability and controllable, understandable, performance constraints are often as critical as functional correctness.

Our main contributions are threefold:

• Firstly, to address the challenge mentioned above, we introduce BigO(Bench), a novel benchmark for code generation that evaluates a model’s understanding of time and space complexities, including runtime and memory profiling measurements for a set of 3,105 coding problems and 1,190,250 solutions from Code Contests. As shown in Figure 1, for a given coding challenge and human solution, the

Complexity Framework

###### Problem Description Human Solutions

n=int(input()) a=list(map(int, input().split())) a.sort() b,t,c=[],1,0 for i in range(n):

Nicholas, a painter is going to paint several new canvases. Nicholas is sure that the canvases will turn out so great that each one will need framing and being hung on the wall. Frames are what Nicholas decided to begin with.

n=int(input()) a=list(map(int, input().split())) a.sort() b,t,c=[],1,0 for i in range(n):

n=int(input()) a=list(map(int, input().split())) a.sort() b,t,c=[],1,0 for i in range(n):

if i==n-1 or a[i]!=a[i+1]: c+=t//4 if t%4>=2:b.append(t%2) t=1

if i==n-1 or a[i]!=a[i+1]: c+=t//4 if t%4>=2:b.append(t%2) t=1

else:t+=1 print(len(b)//2+c)

if i==n-1 or a[i]!=a[i+1]: c+=t//4 if t%4>=2:b.append(t%2) t=1

else:t+=1 print(len(b)//2+c)

Nicholas has n sticks whose lengths equal a1, a2, ... an. Nicholas does...

else:t+=1 print(len(b)//2+c)

|Time O(nlogn)|
|---|

|Time O(n2)|
|---|

Time O(n)

“Can you generate an optimized solution in O(n) time?”

from collections import Counter

[Figure 1] -- Can LLMs Generate Code with Controlled Time and Space Complexity_images/imageFile1.png>)

[Figure 2] -- Can LLMs Generate Code with Controlled Time and Space Complexity_images/imageFile2.png>)

[Figure 3] -- Can LLMs Generate Code with Controlled Time and Space Complexity_images/imageFile3.png>)

n = int(input()) arr = list(map(int, input().split())) freq = Counter(arr) ans = 0

“What is the time complexity of this solution?”

Complexity Coeff Distribution

for val in freq.values():

ans += val//2*2 print(ans//4)

###### ✅ ❌

better coeff worse coeff

3.Complexity Coeff Ranking

✅ ❌

“It’s O(n) !” “It’s O(n2) !”

n = int(input()) a = map(int, input().split

✅ ❌

### LLM

|())<br><br>| |
|---|---|
|, 0) + 1 val in<br><br>| |

O(n) not O(n)

cnt = {} for num in a:

1.Complexity Prediction

cnt[num] = cnt.get(num

2.Complexity Generation

pairs = sum(val // 2 for cnt.values()) print(pairs // 2)

- Figure 1 BigO(Bench) framework overview: Given a coding problem and human solutions, the framework evaluates language models on three key tasks: (1) predicting time-space complexities of existing solutions, (2) generating new code that meets specified complexity requirements, and (3) ranking solutions against human-written code with similar complexity profiles. The complexity framework automatically validates model outputs by computing runtime distributions and curve coefficients.

model can be queried to a. predict time-space complexities, b. generate code that solves the challenge while adhering to a specified (known to be feasible) complexity, and c. on top of it ranks better than human solutions of the same challenge and complexity.

top scores 41.4% and 4.8% on time complexity prediction and generation, 3.4% on space complexity generation and is outperformed on space prediction by Llama 3.1 405B with 10.3%.

2 Related Work

- • Secondly, we release the code for our complexity inference framework, that takes a Python function and returns time and space complexities. It’s a rule-based algorithm based on fuzzing, profiling, and regressing of major complexity classes (including multi-dimensional). This is what we used to produce ground truth labels for BigO(Bench), which are statistically significant ground truth performance profiles and not theoretical complexities. This complexity evaluation framework achieves 92% and 84% match (with human annotated theoretical complexity) respectively on the time and space complexity test sets.
- • Thirdly, we evaluate 12 popular models on our benchmark along fined-tuned ones and compare in details their performance: using our All@1 metric, DeepSeek-R1 Llama 70B achieves

2.1 Benchmarks for Code Generation

As the coding skills of LLMs were still limited, benchmarks for code generation originally focused on simple functions and coding challenges, as illustrated by HumanEval (Chen et al., 2021) and MBPP (Austin et al., 2021) , probably the most famous coding benchmarks. Today, these benchmarks are considered saturated, as top reported pass@1 scores lean towards 90-95% success rate. A first area of work has focused on extending, improving quality and correcting these benchmarks, be it with HumanEval+ (Liu et al. (2023) added more tests to pass) or HumanEvalPack (Muennighoff et al. (2024) added more tasks and programming languages).

A different area of research pursues scale and reasoning around code as a way to formulate benchmarks

that can challenge LLMs over a longer term. SWEBench (Jimenez et al., 2024) and AssistantBench (Yoran et al., 2024) leverage Github as a source of large software development tasks, that do not resemble nor include obvious patterns of reproduction, therefore not solvable with simple fine-tuning on similar data. Nevertheless, the evaluation cost in time and compute is a non-negligible limitation that restrains certain teams from using these benchmarks.

BigO(Bench) is an tentative benchmark to integrate challenging notions of reasoning around code into a simple formulation, providing a practical evaluation metric for code LLMs that can easily be used to discriminate and iterate improvements of their coding capabilities.

- 2.2 Time-Space Complexity Task

Coding interviews for software engineers are centered around small coding challenges to be solved and explained, typically by discussing the time-space complexity of the proposed solutions. Only a few previous works attempted to frame the task of time-space complexity explanation for LLMs. Nevertheless, they all fall short of providing sufficient elements to build a solid benchmark: CoRCoD (Sikka et al., 2019) was a benchmark made of 932 pieces of code in JAVA programming language, labeled for five classes of time complexity only, attempting to solve the task without using any large language model; TASTY (Moudgalya et al., 2023) studies the task based on 3000 C++ and Python problems from five complexity classes, restricted to a classification objective and only benchmarks small BERT models (Devlin et al., 2019); CodeComplex (Baik et al., 2024) is made out of 10k Python and Java programs, annotated only for five classes of time complexity, and formulated as a classification task only.

In order to improve on the previous attempts, BigO(Bench) explores not only time but also space complexity, out of an unconstrained set of classes to capture more various solutions and problems. The benchmark not only studies the classification task but also the more challenging open-framed generation task, so to mimic the real-world thought process of designing a solution for a target complexity. Altogether, this turns out to create a challenging task that wide-used LLMs are benchmarked upon, and hopefully it provides a new perspective on the limitations of current models and their reasoning capabilities around code.

#### 3 Dynamic Complexity Inference Framework

In all this study, complexity refers to worst-case complexity of a coding snippet, that is to say finding the dependency between how a chosen set of inputs can grow in size and have the worst effect on the runtime and memory footprint of the program. Python is considered as the only language studied, and therefore complexity can account for python-specific optimizations (for instance through CPython or the compiler) that get reflected in the empirical time and space measures. In the quest of finding the worst case scenario of a snippet of code, natural language constraints on the inputs as detailed in the problem description can be ignored, as long as the program runs and does not fail. Any basic operator like the addition of numbers or the initialization of an empty list are considered as constant time and space.

3.1 Implementation

The time-space complexity framework is a rule-based algorithm that can process any Python function in order to infer its time and space complexities dynamically. As inputs, it takes a Python function along its function inputs and their corresponding dataclass (see Section 4.2), which are then processed and modified before being run while runtime and memory footprints are measured. From a high-level perspective, the framework increases the size of inputs following various strategies, in order to assess the impact of their size on execution metrics (e.g. execution time, memory used). When the function has several arguments, they can be expanded independently or together to determine the overall complexity of the function, taking into account potential interdependencies. The prepared code, along with the various sets of expanded inputs are queued up and run in independent sandboxes, using the Bubblewrap library (bub, 2024), to avoid any harmful side effects of the code being run. While running, Cprofiler is used for time execution measures and tracemalloc for memory footprint. Using non-negative least squares curve fitting on each set of measures, the coefficients and residuals of each complexity class are computed. The gold complexity class output for a given set of measures is chosen as the minimizer of the residuals, taking into account a simplicity bias (the more simple the complexity class is, the smaller the simplicity bias). This curve fitting is applied on each set of measures, each corresponding to a different subset of arguments being expanded with a different expansion method. Using ensemble methods, the global

Execution Queue

Code Snippet

Code Processing

Cleaning, solving dependencies, ...

from collections import Counter

code

- arr random;size 1 n copy;size 1

- 1

code arr random;size 2 n copy;size 1

- 2

code arr random;size 3 n copy;size 1

- 3

n = int(input()) arr = list(map(int, input().split())) arr.sort() freq = Counter(arr) ans = 0

Input Generation

arr: List[int]

|[2, 3, 5,|6, 1]|[2, 3, 5, 4, 5, 6, 7,|6, 1, 2]|[2, 3, 5, 4, 5, 6, 7, 10, 5, 9, 12]|6, 1,<br><br>2, 13,|…|
|---|---|---|---|---|---|---|
|[1, 3, 2, 4, 5]| |[1, 3, 2, 4, 5, 1, 3, 2, 4, 5]| |[1, 3, 2, 4, 5, 1, 3, 2, 4, 5, 1, 3, 2, 4, 5]| |…|
|[1, 1, 1, 1, 1]| |[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]| |[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]| |…|

Random

for val in freq.values():

ans += val//2*2 print(ans//4)

Copy

Identity

##### …

n:int

Input Example

|7|14|21|…|
|---|---|---|---|

Copy

n

code

arr: [2, 3, 5, 6, 1] n: 7

- inp1 method;size m
- inp2 method;size k

1 2 3 Input size

Complexity Execution Measures Output

###### Complexity Fitting

Process 1

Process 2

[Figure 4] -- Can LLMs Generate Code with Controlled Time and Space Complexity_images/imageFile4.png>)

{arr: List[int]} X {n: int}

[Figure 5] -- Can LLMs Generate Code with Controlled Time and Space Complexity_images/imageFile5.png>)

— Case 1: O(nlogn)

Process 3

Process 4

Time: O(nlogn)

arr: List[int] n: int

- — Case 1: O(nlogn)
- — Case 2: O(nlogn)
- — Case 3: O(nlogn)
- — Case 4: O(nlogn)
- — Case 5: O(n)
- — Case 6: O(n)
- — Case 7: O(nlogn)
- — Case 8: O(nlogn)

- — Case 1: O(1)
- — Case 2: O(n)

Execution Sandboxes

Space: O(n)

Runtime Curves Memory Footprint Curves

Process n

Process m

- Figure 2 Outline of the dynamic complexity inference framework. The framework takes a code snippet and a single example of inputs to this code snippet. Then, it processes the code snippet and proceeds with extensive inputs generation, based on the provided example of inputs: inputs are independently or interdependently increased in size, using several expansion methods that can be the identity or random, among else. This forms a queue of synthetic inputs on which to execute the provided code snippet. These executions happen independently in sandboxes, where runtime and memory footprint measures are taken. Once all the measures are collected, the framework can model the code snippet time and space dependencies to the different inputs. Using curve fitting, the time and space complexity of the code is computed on each input separately and then altogether. The global time and space complexity over all inputs is what is being returned.

complexity of the Python function is computed by aggregating the individual complexity outputs along the different set of measures. Finally, the complexity framework also returns the coefficients of the curve of each elected complexity. These coefficients can be leveraged to rank and classify the optimisations of different Python solutions within the same complexity class. More details and set up instructions are shared on Github.

consists in leveraging various measuring tools (different clocks, deterministic or statistical measures, ...) for various ranges of input size values with more or less granularity. Third, Complexity Fitting, relies on the measures obtained to apply various curve fitting methods, aggregation methods and ensemble methods to form the global complexity formula across all inputs. Parameters from each of these three groups were optimized towards three metrics of interest: pure accuracy of detecting the correct complexity class, coverage by handling as many code snippets and problems as possible, and self-consistency of outputting stable results over multiple runs and across different compute instances.

- 3.2 Parametrization

The framework involves three main parametrized steps. The first step, Process Allocation, handles the multiple (Code,Expanded inputs) pairs to be run and measured for time and space, trying to maximize the execution throughput while minimizing its variability and instability. Second, Execution Measures

#### 4 Benchmark Data Release

- 4.1 Composition

[Figure 6] -- Can LLMs Generate Code with Controlled Time and Space Complexity_images/imageFile6.png>)

- Figure 3 Distribution of time-space complexity classes across BigO(Bench) dataset of 3,105 coding problems. Each problem is included when at least one solution exists with that specific time-space complexity pair. Linear time O(n) represents 38% of solutions, while constant space O(1) accounts for 25%. The chart orders classes by computational efficiency, with less common classes grouped under “other”. Problems for which the framework can not infer a time complexity and/or a space complexity are not counted. Codeforces is an online competitive coding platform that gathers challenging problems to be solved in various programming languages. Humans can submit candidate solutions that are rewarded more the faster and more memory-efficient they are. Using coding problems and solutions from Codeforces mostly (and in minority from a few other coding platforms), Code Contests is a dataset that provides the problem descriptions along with correct and incorrect human solutions.

We annotated data from Code Contests for time and space complexity to create BigO(Bench) using the complexity framework described in Section 3. Code Contests data was limited to correct solutions (according to public and private tests) written in Python code only, which sum up to 8,139 problems and 1,485,888 solutions in total. In addition, problems that have no working dataclass (see Section 4.2), too few solutions (fewer than 50) or unusual data types are also filtered out. This leads to our general dataset, annotated and released as part of BigO(Bench), consisting of 3,105 coding problems and 1,190,250 solutions. Problems are characterized by their difficulty level (A: 942; B: 682; C: 427; D+: 321; Unknown: 733), their algorithmic notions (37 different notions in total) and their inputs (1 to 11 distinct arguments across 32 different data types).

Each solution is annotated by the complexity framework and associated with a time complexity class, a space complexity, the corresponding coefficients of the complexity curves and the runtime/memory-footprint measures that were used to infer these attributes. Therefore, each problem gets associated with one or several time-space complexity classes consisting of solutions that have various complexity coefficients, corresponding to different optimization tricks within the same class of complexity. Dataclasses generated for each problems are also released (see Section 4.2).

The time and space complexity test sets are selected among this global pool of problems and solutions by executing a range of post-processing and filtering steps, so to provide meaningful metrics by enforcing diversity of classes as well as performance and stability of the framework on the problems. For instance, only problems with several complexity classes are kept; absolute and relative thresholds (to the most popular class of the problem) filter out outliers; complexities with an unlikely number of variables or with a rate of failure too high are removed; finally solutions with an unlikely abstract tree given its inferred complexity or unstable complexity framework predictions are also withdrawn. The resulting time and space complexity test sets have distinct supports of problems (though there is an overlap of 63 problems), given that empirically only few problems have diversity of classes both in time and in space. Moreover they do not respect the official Code Contests splits, which are too imbalanced in terms of complexity classes and with a test split not large enough.

The time complexity test set is made out of 311 problems and 640 corresponding solutions covering 11 different classes (the most represented ones being O(n), O(n.log(n)), O(n2), O(1), O(n × m) and the least represented O((n + m)log(n + m))). The space complexity test set consists in 308 problems and 636 solutions covering 5 different classes (by order of popularity O(n), O(1), O(n2), O(n + m), O(n × m)). A training split for fine-tuning purposes is also released. Figure 3 shares more details about the distribution of complexity classes in the data being released. It is imbalanced and heavily tailed: linear time complexity represents 38% of all solutions, constant time complexity 20%; for space complexity, distribution is even more skewed with respectively 47% and 25%.

4.2 Dataclass Generation

In order to infer labels with the complexity framework on Code Contests code snippets, code inputs need to be parsable into a dataclass of the format Algo-

###### Algorithm 1 Dataclass Template

@dataclass c l a s s Input :

- var_1 : type_1
- var_2 : type_2

. . . @classmethod def from_str ( cls , input_ : str ) :

. . . return c l s (var_1 , var_2 , . . . )

def __repr__( s e l f ) :

. . . return input_

- Table 1 Comparison of models for generating problemspecific dataclasses that can parse the incoming input streams into each problem’s variables, on Code Contests. All models but CodeLlama 70B Instruct (16k only) use a context window of 32k tokens.

Model Corr@10 BckTr@10

Codestral 22B 63.6 54.0 CodeLlama 34B Instruct 22.1 17.8 CodeLlama 70B Instruct 10.3 7.9 Llama 3.1 8B Instruct 31.9 21.4 Llama 3.1 405B Instruct 70.2 58.1

rithm 1. We define the task of dataclass generation as querying a LLM for such dataclass given the problem description and an example solution. Performance is measured by Corr, that accounts for the correction (executability) of the methods Input.from_str and Input.__repr__, and BckTr, which measures accuracy of the backtranslation:

input_ == Input.from_str(input_).__repr__()

Corr@k and BckTr@k are unbiased estimators of performance among k samples, following the definition of Chen et al. (2021). Table 1 sums up the benchmark results of the dataclass generation task. Llama 3.1 405B reaches best performance, capable of 58.1% correct backtranslation for one dataclass out of ten attempts per problem. To further boost performance, several passes corresponding to different solutions per problem are performed, thus generating a correct dataclass for 82% of Code Contests problems.

- 4.3 Complexity Framework Performance

Using the general dataset and the test splits of BigO(Bench), the complexity framework is evaluated for its accuracy, coverage and self-consistency.

Accuracy A human review measured the accuracy

of the labels as output by the complexity framework compared to the labels assigned by a human. On the time complexity test set (after post-processing and filtering), the framework reaches 92% accuracy on a 50 sample split. On the space complexity test set (after post-processing and filtering), the framework reaches 84% on a 50 sample split.

Coverage Figure 4 measures the ratio of solutions per problem for which the framework fails to predict a label. Whatever the type of complexity, approximately 84% of problems have a fail rate below 30%, and only 4.5% of problems have a fail rate above 0.9, for reasons ranging from incorrect generated dataclass to an edge case not covered by the range of tests the framework performs. Most input types are correctly covered by the framework. Exceptions such as tuple and triple-nested lists are infrequent.

Self-consistency As it relies on empirical measures of runtimes and memory footprints, the framework is exposed to stochastic noise that can alter the reliability of its complexity predictions. Running the framework 20 times on 10 solutions of every problem and complexity class of the candidate test set, before any filtering based precisely on stability, 91.9% (resp. 89.1%) self-consistency is achieved for time (resp. space) complexity, for a total of 10130 (resp. 10520) different code solutions.

#### 5 Evaluation

We use BigO(Bench) to evaluate several LLMs commonly used for coding and reasoning tasks: Llama 3.1 8B, 70B, 405B, 3.3 70B (Dubey et al., 2024), Codestral 22B (MistralAI, 2024), GPT4o (OpenAI et al., 2024b), o1-mini (OpenAI et al.,

- 2024a), Qwen 2.5-Coder 32B (Hui et al., 2024), DeepSeek-Coder-V2 236B (DeepSeek-AI et al.,
- 2024b), DeepSeek-V3 671B (DeepSeek-AI et al., 2024a) and DeepSeek-R1 Qwen/Llama distilled (DeepSeek-AI et al., 2025). All models are evaluated using their instruct variant, if available, in a zeroshot fashion (unless otherwise stated). GPT4-o and o1-mini do not share any estimate on inference compute. Also, o1-mini returned many empty answers, potentially due to reasoning collapse: we discarded these answers and used only non-empty answers to compute metrics. As a result, its performance can be regarded as an upper-bound optimistic estimate. DeepSeek-R1 distilled models used substantially more compute than Llama 3.1 405B (×2 compute nodes, ×5 compute time, ×16 generation tokens).1

1DeepSeek-R1 is not reported as an initial assessment led to over-budget compute usage; this partial run gave similar

- Table 2 BigO(Bench) benchmark results for popular LLMs. Program Synthesis checks correctness of model-generated solutions to given programming problems.Complexity Prediction measures whether a model can find the time-space complexity of a code snippet. Complexity Generation evaluates whether a model outputs a working code snippet to a given problem, that meets a time-space complexity requirement. Pass@k treats all complexity classes of all problems independently, Best@k only evaluates the most optimized complexity class of each problem, All@k measures whether all complexity classes per problem are correct at once.

Prog. Synthesis Complexity Pred. Complexity Gen.

###### Pass Pass Pass Best All Pass Pass Best All @1 @10 @1 @1 @1 @1 @10 @1 @1

Model

Time

Baselines 30.3 55.4 39.5 68.5 0.0 12.1 29.7 19.0 0.9 Llama 3.1 8B 12.0 28.0 46.5 58.3 21.8 5.2 16.5 7.7 0.6 Llama 3.1 70B 29.6 54.0 57.2 68.9 33.8 14.2 34.8 20.3 3.1 Llama 3.3 70B 43.4 66.0 58.2 72.6 33.7 17.7 40.0 25.7 3.3 Llama 3.1 405B 40.4 65.8 60.9 72.8 38.3 19.5 43.6 26.4 4.2 Codestral 22B 23.7 47.5 56.0 67.8 33.5 10.6 26.6 14.9 1.3 GPT-4o 51.0 78.3 57.7 69.7 33.1 20.6 44.7 30.2 4.3 O1-mini 62.5 76.8 58.3 65.2 35.6 19.8 65.2 27.6 4.5 Qwen2.5-Coder 32B 30.5 50.8 58.5 68.2 34.9 12.2 26.5 15.2 3.1 DeepSeekCoderV2 236B 44.1 65.5 54.9 68.9 29.6 19.5 38.0 27.6 3.3 DeepSeekV3 671B 41.4 63.6 54.4 72.4 27.1 17.7 37.7 23.0 3.4 DeepSeekR1 Qwen 32B 70.1 83.7 62.2 72.7 41.1 29.0 49.9 46.1 4.8 DeepSeekR1 Llama 70B 70.1 83.8 64.2 75.4 41.4 29.2 51.6 46.5 4.8

Space

Baselines 30.1 52.6 45.4 50.3 0.0 12.2 32.4 17.8 1.3 Llama 3.1 8B 14.1 29.7 36.6 40.3 7.9 5.4 16.1 7.9 0.5 Llama 3.1 70B 29.7 52.3 42.5 56.1 11.9 11.7 33.0 17.2 1.8 Llama 3.3 70B 42.6 62.5 41.1 55.2 10.9 15.0 37.7 21.9 1.8 Llama 3.1 405B 39.5 63.2 44.8 58.9 14.0 16.1 42.0 22.6 2.7 Codestral 22B 25.7 47.6 44.3 62.5 10.6 11.0 29.4 16.7 1.3 GPT-4o 51.6 74.4 43.4 61.4 11.0 18.1 39.9 28.0 1.4 O1-mini 58.0 72.9 42.7 45.6 8.1 16.6 61.3 25.7 2.5 Qwen2.5-Coder 32B 31.1 49.2 45.6 63.4 12.6 10.1 23.3 15.3 1.2 DeepSeekCoderV2 236B 43.1 63.8 44.1 59.6 8.2 16.7 34.5 25.6 1.0 DeepSeekV3 671B 41.8 62.5 43.5 62.6 11.2 15.0 35.4 22.6 1.6 DeepSeekR1 Qwen 32B 68.0 80.6 43.2 55.0 8.1 24.8 48.6 38.6 3.1 DeepSeekR1 Llama 70B 68.8 81.2 44.4 56.1 10.4 25.6 50.0 38.7 3.4

Pure program synthesis performance is also displayed on the same test splits as the rest of the metrics. It is evaluated for pass@k using all public, private and generated tests. For each metric, best values or any values not significantly lower than the best are displayed in boldface. Metrics are macro-averaged first by complexity classes within each problem and then across problems. 2

- 5.1 Time-Space Complexity Prediction

The first evaluation task of BigO(Bench), Complexity Prediction, consists in predicting the time and

results on complexity tasks as DeepSeek-R1 Llama 70B.

2One-tailed paired t-tests on 1000 bootstraps samples of the model results evaluate the significance of the superiority of the best model. Any @k metric uses an unbiased estimator based on 20 samples.

space complexity given a problem description and a human solution. Our baseline for this task is the naive model that always returns O(n), the most frequent class. Pass@k measures the accuracy of finding the correct complexity; Best@k measures accuracy only across the most optimized complexity class of each problem; All@k requires correct complexity output across all complexity classes at once per problem. Results are displayed in Table 2. A query example, along with an output example of DeepSeek-R1 Llama 70B, is provided in Appendix A.

5.2 Time-Space Complexity Code Generation

The second task Complexity Generation requires the LLM to generate a correct solution to a given problem description that has to respect a feasible time or

[Figure 7] -- Can LLMs Generate Code with Controlled Time and Space Complexity_images/imageFile7.png>)

- Figure 4 Failure rate analysis of the complexity inference framework. The top plot shows the overall distribution of framework failures across all problems. The bottom heatmap breaks down failure rates by input type and number of distinct inputs. Approximately 84% of problems have failure rates below 30%, demonstrating robust performance across most input configurations.

space complexity requirement. Our baseline for this task is a Llama 3.1 70B model that is queried for the same prompts without the complexity requirement. Pass@k measures the accuracy of finding a correct solution, according to public, private and generated tests, that has the correct complexity, as measured by the complexity framework; Best@k and All@k are similarly defined as their counterparts of Section 5.1. Results are displayed in Table 2. An example with DeepSeek-R1 Llama 70B is provided in Appendix B.

- 5.3 Time-Space Complexity Coefficient Percentile Ranking

The third task, ComplexityCoefficientPercentileRanking, measures how a generated solution to a given problem, respecting a complexity requirement, ranks among human solutions of the same complexity class and problem. The ranking is performed based on the coefficient of the complexity curve, as measured by the framework: the lower the coefficient, the more flat the complexity curve and the more optimized the solution. Ranking results are given in percentile of the distribution, where a solution of the nth percentile

Table 3 Using the complexity framework, the best measured coefficient of the complexity curve, out of 20 attempts, is used to rank LLM-generated code among human solutions from the same problem and time-space complexity class. Ranking is percentile based, n% ranking score amounts for n% human solutions having worse complexity coefficient. If no LLM solution passes correctness tests, ranking score is set to 0. Intersec is the subset where all starred (*) models have at least one successful solution.

Coefficient

###### Ranking All Full Intersec @1

Model

Time

Llama 3.1 8B 13.9 43.2 0.5 Llama 3.1 70B 28.3 74.1 2.5 Llama 3.3 70B* 33.8 74.3 2.8 Llama 3.1 405B 33.9 78.4 4.3 Codestral 22B 21.6 58.6 1.5 GPT-4o* 36.6 71.8 4.2 O1-mini* 26.3 79.3 3.1 Qwen2.5-Coder 32B 19.7 51.4 2.2 DeepSeekCoderV2 236B 27.7 59.2 2.8 DeepSeekV3 671B 28.7 63.1 3.4 DeepSeekR1 Qwen 32B* 38.6 79.6 4.2 DeepSeekR1 Llama 70B* 38.3 79.2 4.0

Space

Llama 3.1 8B 14.7 49.3 0.8 Llama 3.1 70B* 26.9 86.1 1.1 Llama 3.3 70B 32.8 79.1 1.6 Llama 3.1 405B 35.5 85.4 2.2 Codestral 22B 25.2 71.2 1.2 GPT-4o* 31.6 84.1 1.3 O1-mini* 21.1 81.6 1.5 Qwen2.5-Coder 32B 20.5 74.8 0.6 DeepSeekCoderV2 236B 26.8 69.7 1.2 DeepSeekV3 671B 27.2 75.2 1.3 DeepSeekR1 Qwen 32B* 40.1 84.6 3.0 DeepSeekR1 Llama 70B* 41.6 86.9 3.3

is more optimized than n% of human solutions. The querying is similar to Section 5.2 with the addition of the requirement "Try to optimize the runtime of your code as much as you can, while respecting the time complexity requirement". Results are displayed in Table 3.

5.4 Complexity Prediction and Generation Fine-tuning

Using training sets of 2000 problems and 20k code solutions, Llama 3.1 70B is being fine-tuned for the first and second task. Time and space generation training sets sums up to 22M tokens, prediction training sets to 18-19M. Each complexity class of each problem includes 10 human examples, filtered following the same steps as the test set creation (see Section 4.1). Models are fine-tuned for 10 epochs in

###### Table 4 BigO(Bench) benchmark results for fine-tuned Llama 3.1 70B on the time-space prediction and generation tasks. Same metrics are reported as in Table 2.

Prog. Prediction Generation Synth. Time Space Time Space

Method

Pass@1 All@1 All@1 All@1 All@1

Zero-shot 29.6 33.8 11.9 3.1 1.8 Few-shot 28.9 33.6 12.1 2.4 1.4

Prediction Fine-tuning

###### Time 27.4 36.5 6.6 2.9 1.3 Space 26.6 9.0 14.0 2.4 1.4

Generation Fine-tuning

Time 23.2 34.7 12.7 1.2 1.3 Space 23.4 34.6 13.0 1.5 1.4

instruct format. Results are displayed in Table 4.

6 Quantitative Analysis

Understanding time-space complexity Looking at Table 2, all LLMs undergo a noticeable drop of performance on the combined task ComplexityGeneration compared to the individual tasks Program Synthesis and Complexity Prediction. Across all tasks, the top performing model remains DeepSeek-R1 Llama 70B with 64.2 and 29.2 Pass@1 on respectively time prediction and generation, except on space prediction where models tend to overthink and misunderstand the notion of extra space complexity, though explicitly described in the test prompts. Models tend to be even more misled when asked to "Optimize the solution while respecting the complexity requirement", which leads to average 12% loss of performance for time generation All@1 in Table 3, up to 30% for GPT-4o and o1-mini.

At a more granular level, models tend to underperform on non-optimal complexity classes, compared to the most optimized class of every problem, as underlined in Figure 5. This seems counterintuitive for any human programmer, usually accustomed to easily find non-optimized solutions, but struggling at the best ones, especially on such a difficult set of competition programming questions. In addition, LLMs do not understand that adding dummy pieces of code, like a list sorting, could easily transform a working linear solution into a less optimized linearithmic one. In the end, true reasoning is better captured by All@1 metrics that uniformly evaluate LLMs across optimized and non-optimized solutions, verifying that LLMs do not just stumble upon a solution of the right complexity because they learned by heart the widespread optimized code snippets. Top performing

[Figure 8] -- Can LLMs Generate Code with Controlled Time and Space Complexity_images/imageFile8.png>)

Figure 5 LLM results aggregated by time complexity class and by algorithmic notions for all models part of BigO(Bench).

model is DeepSeek-R1 Llama 70B with scores as low as 4.8 and 3.4 All@1 on time and space generation. These reasoning metrics can also be used as a proxy of the understanding of other notions, such as probabilities, where models explicitly optimized for math reach noticeably higher scores, as displayed on Figure 5. When analyzing these results, one has to keep in mind that the splits were done on the training data of Code Contests that includes all solutions of the different complexity classes to all problems, therefore already seen by the models.

Token-space reasoning models Though they largely outperform other LLMs on pure program synthesis, token-space reasoning models such as o1-mini, DeepSeek-R1 Qwen 32B and DeepSeek-R1 Llama 70B are much closer in terms of performance on complexity-related tasks, and even outperformed by Llama 3.1 405B on space complexity prediction specifically: while the latter obtains 14.0 All@1, the former respectively only reach 8.1 and 10.4. For this specific case, reasoning models seem to exhibit patterns of overthinking, misunderstanding the notion of extra space complexity, even if it is explicitly described in the test prompt. When explicitly prompted in order to understand their failure modes, it turns out that such models are able to recognize all classes of complexity of a problem from a prediction point of view, but when asked to generate them, they fail to return the less optimized classes of complexity, favoring the optimal solution. Not to mention that these models could just easily ‘cheat’ by tweaking the

optimal solution with a dummy sort or nested forloop, and therefore transform it, as any programmer could figure it out, as a less-than-optimal solution.

In general, these models struggle with the ambiguity of higher-level reasoning tasks, especially when there is no explicit verifier that they may have been confronted with during their reinforcement, such as programming tests to pass. This triggers the question of whether they really understand how to ‘think’ about notions they ‘know’, or if they only learn by heart patterns of ‘thoughts’ exhibited by human annotation efforts and training rewards. As all these complexity solutions were in fact in their training data, it demonstrates that these highly efficient search models can still fail to recover the correct data points when they were not reinforced for the specific search criteria.

Developing challenging reasoning benchmarks In the context of newly released benchmarks getting quickly saturated, BigO(Bench) aims at evaluating highlevel reasoning skills that stay out-of-scope of current LLMs, bringing their performance down as displayed by Figure 6. Table 4 measures that the benchmark remains robust to fine-tuning, and problems from Code Contests training data ensures that we are indeed evaluating the difficulty of the task itself, not that coding problems are just out of distribution. With reasoning models topping benchmarks, new challenging benchmarks are perhaps more about finding out-of-distribution patterns of thinking rather than new data points of existing reasoning scenarios. It was especially difficult to design BigO(Bench) without any human labels available, as the qualification level required from annotators was not reachable. But this is also probably why the models had been underexposed to this reasoning task, therefore making it challenging for them.

Limitations The complexity framework itself is prone to errors, as for specific problems it can potentially fall upon worst-complexity edge cases. In addition, the measures on which the complexity prediction is based remain noisy, still relying on real CPU runtimes and using statistical measuring tools; they could potentially become more reliable with the help of virtual CPU cores.

Although we fine-tune Llama models, we did not used advanced multiturn prompting nor further reinforcement. A human annotation effort could also help models reason better for these tasks. Finally, the coding problems, and the framework itself, remain limited to Python. Mixing other languages such as C++ and Java could lead to interesting studies about cross-languages optimization strategies.

[Figure 9] -- Can LLMs Generate Code with Controlled Time and Space Complexity_images/imageFile9.png>)

###### BigO(Bench)

Figure 6 Model performance per coding benchmarks: HumanEval, MBPP and BigCodeBench main metrics are all Pass@1; for BigO(Bench), we display All@1 results.

7 Conclusion

In this work, we introduced BigO(Bench), a novel benchmark of LLMs on code generation, focusing on their understanding of time and space complexities when producing code. It consists in three tasks: first, given a coding challenge, predicting the time-space complexity of a given corresponding solution; second, for a given challenge and time or space complexity, generating a solution that solves the challenge while fulfilling the complexity requirements; third, optimizing the coefficient of the complexity cure compared to the human distribution. This benchmark is supported by the release of time-space complexity labels corresponding to 3,105 coding problems and 1,190,250 corresponding solutions from Code Contests. In addition, we developed and release the code of a complexity framework, capable of dynamically inferring the time-space complexity of a given snippet of code, used to automatically evaluate any synthetic snippet of code and therefore the generation of LLMs conditioned on a given complexity trade-off. Finally, we benchmark 12 LLMs considered as the top coding assistants and analyse their performance.

BigO(Bench) is a challenging benchmark with current top scores belonging to the DeepSeek-R1 Llama 70B, achieving 4.8% and 3.4% All@1 on the time-space complexity generation tasks. Even when fine-tuning a Llama 3.1 70B model, performance increases only marginally, only on prediction tasks. We hope this benchmark can keep challenging upcoming models and help stir the releases of new models towards better understanding of coding abstract notions beyond pure code generation.

References

Bubblewrap library. https://github.com/containers/ bubblewrap, 2024.

Jacob Austin, Augustus Odena, Maxwell I. Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie J. Cai, Michael Terry, Quoc V. Le, and Charles Sutton. Program synthesis with large language models. CoRR, abs/2108.07732, 2021. https://arxiv.org/abs/ 2108.07732.

Seung-Yeop Baik, Mingi Jeon, Joonghyuk Hahn, Jungin Kim, Yo-Sub Han, and Sang-Ki Ko. Codecomplex: A time-complexity dataset for bilingual source codes, 2024. https://arxiv.org/abs/2401.08719.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Pondé de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. CoRR, abs/2107.03374, 2021. https://arxiv.org/abs/2107.03374.

DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, Junxiao Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Litong Wang, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qiancheng Wang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu, Ruoyu Zhang, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Shuting

Pan, T. Wang, Tao Yun, Tian Pei, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wanjia Zhao, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaokang Zhang, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xinnan Song, Xinxia Shan, Xinyi Zhou, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, Y. K. Li, Y. Q. Wang, Y. X. Wei, Y. X. Zhu, Yang Zhang, Yanhong Xu, Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Yu, Yi Zheng, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Ying Tang, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yu Wu, Yuan Ou, Yuchen Zhu, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yukun Zha, Yunfan Xiong, Yunxian Ma, Yuting Yan, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Z. F. Wu, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhen Huang, Zhen Zhang, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhipeng Xu, Zhiyu Wu, Zhongyu Zhang, Zhuoshu Li, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Ziyi Gao, and Zizheng Pan. Deepseek-v3 technical report, 2024a. https://arxiv.org/abs/2412.19437.

DeepSeek-AI, Qihao Zhu, Daya Guo, Zhihong Shao, Dejian Yang, Peiyi Wang, Runxin Xu, Y. Wu, Yukun Li, Huazuo Gao, Shirong Ma, Wangding Zeng, Xiao Bi, Zihui Gu, Hanwei Xu, Damai Dai, Kai Dong, Liyue Zhang, Yishi Piao, Zhibin Gou, Zhenda Xie, Zhewen Hao, Bingxuan Wang, Junxiao Song, Deli Chen, Xin Xie, Kang Guan, Yuxiang You, Aixin Liu, Qiushi Du, Wenjun Gao, Xuan Lu, Qinyu Chen, Yaohui Wang, Chengqi Deng, Jiashi Li, Chenggang Zhao, Chong Ruan, Fuli Luo, and Wenfeng Liang. Deepseek-coderv2: Breaking the barrier of closed-source models in code intelligence, 2024b. https://arxiv.org/abs/2406.

11931.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui

Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. https://arxiv.org/abs/2501.12948.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding, 2019. https: //arxiv.org/abs/1810.04805.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta

Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaoqing Ellen Tan, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aaron Grattafiori, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alex Vaughan, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Franco, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian

Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, Danny Wyatt, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Firat Ozgenel, Francesco Caggioni, Francisco Guzmán, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Govind Thattai, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Japhet Asher, JeanBaptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Karthik Prasad, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kun Huang, Kunal Chawla, Kushal Lakhotia, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Maria Tsimpoukelli, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikolay Pavlovich Laptev, Ning Dong, Ning Zhang, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Rohan Maheswari, Russ Howes, Ruty Rinott, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha

Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Kohler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vítor Albiero, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaofang Wang, Xiaojian Wu, Xiaolan Wang, Xide Xia, Xilun Wu, Xinbo Gao, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yuchen Hao, Yundi Qian, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, and Zhiwei Zhao. The llama 3 herd of models, 2024. https://arxiv.org/abs/2407.21783.

Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, Kai Dang, Yang Fan, Yichang Zhang, An Yang, Rui Men, Fei Huang, Bo Zheng, Yibo Miao, Shanghaoran Quan, Yunlong Feng, Xingzhang Ren, Xuancheng Ren, Jingren Zhou, and Junyang Lin. Qwen2.5-coder technical report, 2024. https: //arxiv.org/abs/2409.12186.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues?, 2024. https://arxiv.org/abs/2310.

06770.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation, 2023. https://arxiv.org/abs/2305.

01210. MistralAI. Codestral, 2024. https://mistral.ai/news/ codestral/. Accessed: 2024-05-29.

Kaushik Moudgalya, Ankit Ramakrishnan, Vamsikrishna Chemudupati, and Xing Han Lu. Tasty: A transformer based approach to space and time complexity, 2023. https://arxiv.org/abs/2305.05379.

Niklas Muennighoff, Qian Liu, Armel Zebaze, Qinkai Zheng, Binyuan Hui, Terry Yue Zhuo, Swayam Singh, Xiangru Tang, Leandro von Werra, and Shayne Longpre. Octopack: Instruction tuning code large language models, 2024. https://arxiv.org/abs/2308.07124.

OpenAI, :, Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, Alex Iftimie, Alex Karpenko, Alex Tachard Passos, Alexander Neitz, Alexander Prokofiev, Alexander Wei, Allison Tam, Ally Bennett, Ananya Kumar, Andre Saraiva, Andrea Vallone, Andrew Duberstein, Andrew

Kondrich, Andrey Mishchenko, Andy Applebaum, Angela Jiang, Ashvin Nair, Barret Zoph, Behrooz Ghorbani, Ben Rossen, Benjamin Sokolowsky, Boaz Barak, Bob McGrew, Borys Minaiev, Botao Hao, Bowen Baker, Brandon Houghton, Brandon McKinzie, Brydon Eastman, Camillo Lugaresi, Cary Bassin, Cary Hudson, Chak Ming Li, Charles de Bourcy, Chelsea Voss, Chen Shen, Chong Zhang, Chris Koch, Chris Orsinger, Christopher Hesse, Claudia Fischer, Clive Chan, Dan Roberts, Daniel Kappler, Daniel Levy, Daniel Selsam, David Dohan, David Farhi, David Mely, David Robinson, Dimitris Tsipras, Doug Li, Dragos Oprica, Eben Freeman, Eddie Zhang, Edmund Wong, Elizabeth Proehl, Enoch Cheung, Eric Mitchell, Eric Wallace, Erik Ritter, Evan Mays, Fan Wang, Felipe Petroski Such, Filippo Raso, Florencia Leoni, Foivos Tsimpourlas, Francis Song, Fred von Lohmann, Freddie Sulit, Geoff Salmon, Giambattista Parascandolo, Gildas Chabot, Grace Zhao, Greg Brockman, Guillaume Leclerc, Hadi Salman, Haiming Bao, Hao Sheng, Hart Andrin, Hessam Bagherinezhad, Hongyu Ren, Hunter Lightman, Hyung Won Chung, Ian Kivlichan, Ian O’Connell, Ian Osband, Ignasi Clavera Gilaberte, Ilge Akkaya, Ilya Kostrikov, Ilya Sutskever, Irina Kofman, Jakub Pachocki, James Lennon, Jason Wei, Jean Harb, Jerry Twore, Jiacheng Feng, Jiahui Yu, Jiayi Weng, Jie Tang, Jieqi Yu, Joaquin Quiñonero Candela, Joe Palermo, Joel Parish, Johannes Heidecke, John Hallman, John Rizzo, Jonathan Gordon, Jonathan Uesato, Jonathan Ward, Joost Huizinga, Julie Wang, Kai Chen, Kai Xiao, Karan Singhal, Karina Nguyen, Karl Cobbe, Katy Shi, Kayla Wood, Kendra Rimbach, Keren Gu-Lemberg, Kevin Liu, Kevin Lu, Kevin Stone, Kevin Yu, Lama Ahmad, Lauren Yang, Leo Liu, Leon Maksin, Leyton Ho, Liam Fedus, Lilian Weng, Linden Li, Lindsay McCallum, Lindsey Held, Lorenz Kuhn, Lukas Kondraciuk, Lukasz Kaiser, Luke Metz, Madelaine Boyd, Maja Trebacz, Manas Joglekar, Mark Chen, Marko Tintor, Mason Meyer, Matt Jones, Matt Kaufer, Max Schwarzer, Meghan Shah, Mehmet Yatbaz, Melody Y. Guan, Mengyuan Xu, Mengyuan Yan, Mia Glaese, Mianna Chen, Michael Lampe, Michael Malek, Michele Wang, Michelle Fradin, Mike McClay, Mikhail Pavlov, Miles Wang, Mingxuan Wang, Mira Murati, Mo Bavarian, Mostafa Rohaninejad, Nat McAleese, Neil Chowdhury, Neil Chowdhury, Nick Ryder, Nikolas Tezak, Noam Brown, Ofir Nachum, Oleg Boiko, Oleg Murk, Olivia Watkins, Patrick Chao, Paul Ashbourne, Pavel Izmailov, Peter Zhokhov, Rachel Dias, Rahul Arora, Randall Lin, Rapha Gontijo Lopes, Raz Gaon, Reah Miyara, Reimar Leike, Renny Hwang, Rhythm Garg, Robin Brown, Roshan James, Rui Shu, Ryan Cheu, Ryan Greene, Saachi Jain, Sam Altman, Sam Toizer, Sam Toyer, Samuel Miserendino, Sandhini Agarwal, Santiago Hernandez, Sasha Baker, Scott McKinney, Scottie Yan, Shengjia Zhao, Shengli Hu, Shibani Santurkar, Shraman Ray Chaudhuri, Shuyuan Zhang, Siyuan Fu, Spencer Papay, Steph Lin, Suchir Balaji, Suvansh Sanjeev, Szymon Sidor, Tal Broda, Aidan Clark, Tao Wang, Taylor Gordon, Ted Sanders,

Tejal Patwardhan, Thibault Sottiaux, Thomas Degry, Thomas Dimson, Tianhao Zheng, Timur Garipov, Tom Stasi, Trapit Bansal, Trevor Creech, Troy Peterson, Tyna Eloundou, Valerie Qi, Vineet Kosaraju, Vinnie Monaco, Vitchyr Pong, Vlad Fomenko, Weiyi Zheng, Wenda Zhou, Wes McCabe, Wojciech Zaremba, Yann Dubois, Yinghai Lu, Yining Chen, Young Cha, Yu Bai, Yuchen He, Yuchen Zhang, Yunyun Wang, Zheng Shao, and Zhuohan Li. Openai o1 system card, 2024a. https://arxiv.org/abs/2412.16720.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro

Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang, Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. Gpt-4 technical report, 2024b. https://arxiv.org/abs/2303.08774.

Jagriti Sikka, Kushal Satya, Yaman Kumar, Shagun Uppal, Rajiv Ratn Shah, and Roger Zimmermann. Learning based methods for code runtime complexity prediction, 2019. https://arxiv.org/abs/1911.01155.

Ori Yoran, Samuel Joseph Amouyal, Chaitanya Malaviya, Ben Bogin, Ofir Press, and Jonathan Berant. Assistantbench: Can web agents solve realistic and timeconsuming tasks?, 2024. https://arxiv.org/abs/2407. 15711.

## Appendix

#### A Complexity Prediction Example

- A.1 Example of query

Provide the time complexity for the following competitive programming question and corresponding solution. When analyzing the complexity of an algorithm, consider the worst-case scenario where all possible input combinations are tried, given the following conditions: 1. the inputs must adhere to the specified data types of the problem; 2. the inputs should not cause the code to crash or exit on an exception; 3. the inputs do not necessarily need to satisfy additional constraints that are potentially mentioned in the problem statement; 4. calling input() does not consume runtime nor memory, but of course any operations on top of it or afterwards will be counted towards runtime and memory footprint; 5. Anything printed gets added to the memory. You can take advantage of Python-specific optimizations provided by the underlying CPython interpreter or compiler to achieve the desired complexity, and you must account for them when analyzing the complexity. Here is the programming question:

You are given n segments on a number line; each endpoint of every segment has integer coordinates. Some segments can degenerate to points. Segments can intersect with each other, be nested in each other or even coincide.

The intersection of a sequence of segments is such a maximal set of points (not necesserily having integer coordinates) that each point lies within every segment from the sequence. If the resulting set isn’t empty, then it always forms some continuous segment. The length of the intersection is the length of the resulting segment or 0 in case the intersection is an empty set.

For example, the intersection of segments [1;5] and [3;10] is [3;5] (length 2), the intersection of segments [1;5] and [5;7] is [5;5] (length 0) and the intersection of segments [1;5] and [6;6] is an empty set (length

- 0).

Your task is to remove exactly one segment from the given sequence in such a way that the intersection of the remaining (n - 1) segments has the maximal possible length.

Input The first line contains a single integer n (2 ≤ n ≤ 3 × 105) the number of segments in the sequence. Each of the next n lines contains two integers li and ri (0 ≤ li ≤ ri ≤ 109) the description of the ith segment. Output Print a single integer - the maximal possible length of the intersection of (n - 1) remaining segments after you remove exactly one segment from the sequence. Examples Input

- 4 1 3 2 6 0 4 3 3 Output

1 Input

- 5 2 6 1 3 0 4 1 20 0 4 Output

- 2

- Input
- 3 4 5 1 2 9 20 Output

- 0 Input 2 3 10 1 5 Output 7 Note

In the first example you should remove the segment [3;3], the intersection will become [2;3] (length 1). Removing any other segment will result in the intersection [3;3] (length 0).

In the second example you should remove the segment [1;3] or segment [2;6], the intersection will become [2;4] (length 2) or [1;3] (length 2), respectively. Removing any other segment will result in the intersection [2;3] (length 1).

In the third example the intersection will become an empty set no matter the segment you remove. In the fourth example you will get the intersection [3;10] (length 7) if you remove the segment [1;5] or the intersection [1;5] (length 4) if you remove the segment [3;10]. Here is the corresponding Python solution:

n = int ( input ()) L = [ ]

- R = [ ]
- S = [ ] for _ in range (n ) :

a , b = [ int (x) for x in input ( ) . s p l i t ( ) ] L. append (a )

- R. append (b)
- S . append (( a , b ))

L. sort ( reverse = True ) R. sort ()

i f (L [ 0 ] ,R[ 0 ] ) in S :

print (max(R[ 1 ] -L [ 1 ] , 0 ) ) e l s e :

print (max(R[ 0 ] -L [ 1 ] ,R[ 1 ] -L [ 0 ] , 0 ) ) .

Please ignore any constraints on the input sizes that may have been previously mentioned in the problem description. Compute the big-O complexity as if inputs can be as large as possible. Output the big-O time complexity only, no explanation needed, no other words needed.

- A.2 Example of response from DeepSeek-R1 Llama 70B <think>

Okay, I need to figure out the time complexity of the given Python solution for the problem. Let’s go through the code step by step.

First, the code reads n, which is the number of segments. Then, for each of the next n lines, it reads two integers l_i and r_i, appending them to lists L and R, and storing the tuples in S.

So the initial part is O(n) time because each of the n lines is processed once. Next, the code sorts L in descending order. Sorting a list of size n in Python using the built-in sort function is O(n log n). Similarly, R is sorted in ascending order, which is also O(n log n). After sorting, there’s a condition checking whether the tuple (L[0], R[0]) is in S. Checking membership in a list in Python is O(n) because it has to potentially scan all elements. So this check is O(n). Then, depending on the condition, it either prints max(R[1] - L[1], 0) or max(R[0] - L[1], R[1] - L[0], 0). Both of these operations are O(1) since they involve a couple of subtractions and a max function. So putting it all together:

- Reading the inputs: O(n) - Sorting L and R: O(n log n) each, so total O(2n log n) which simplifies to O(n log n) - Checking the tuple in S: O(n) - The rest is O(1)

The dominant term here is O(n log n) from the sorting steps. The other steps are either O(n) or O(1), which are lower order terms.

Therefore, the overall time complexity is O(n log n). <\think> The time complexity is O(n log n). O(n log n)

#### B Complexity Generation Example

- B.1 Example of query

Provide a Python solution for the following competitive programming question: You are given n segments on a number line; each endpoint of every segment has integer coordinates. Some segments can degenerate to points. Segments can intersect with each other, be nested in each other or even coincide.

The intersection of a sequence of segments is such a maximal set of points (not necesserily having integer coordinates) that each point lies within every segment from the sequence. If the resulting set isn’t empty, then it always forms some continuous segment. The length of the intersection is the length of the resulting segment or 0 in case the intersection is an empty set.

For example, the intersection of segments [1;5] and [3;10] is [3;5] (length 2), the intersection of segments [1;5] and [5;7] is [5;5] (length 0) and the intersection of segments [1;5] and [6;6] is an empty set (length

- 0).

Your task is to remove exactly one segment from the given sequence in such a way that the intersection of the remaining (n - 1) segments has the maximal possible length.

Input The first line contains a single integer n (2 ≤ n ≤ 3 × 105) the number of segments in the sequence. Each of the next n lines contains two integers li and ri (0 ≤ li ≤ ri ≤ 109) the description of the ith segment. Output Print a single integer - the maximal possible length of the intersection of (n - 1) remaining segments after you remove exactly one segment from the sequence. Examples Input

- 4 1 3 2 6 0 4 3 3

Output

- 1 Input

5 2 6 1 3 0 4 1 20 0 4 Output

- 2 Input
- 3 4 5 1 2 9 20 Output

- 0 Input

- 2 3 10 1 5 Output 7 Note

In the first example you should remove the segment [3;3], the intersection will become [2;3] (length 1). Removing any other segment will result in the intersection [3;3] (length 0).

In the second example you should remove the segment [1;3] or segment [2;6], the intersection will become [2;4] (length 2) or [1;3] (length 2), respectively. Removing any other segment will result in the intersection [2;3] (length 1).

In the third example the intersection will become an empty set no matter the segment you remove. In the fourth example you will get the intersection [3;10] (length 7) if you remove the segment [1;5] or the intersection [1;5] (length 4) if you remove the segment [3;10]. Output the code only. Generate code that has an algorithmic time complexity of O(n.log(n)).

When analyzing the complexity of an algorithm, consider the worst-case scenario where all possible input combinations are tried, given the following conditions: 1. the inputs must adhere to the specified data types of the problem; 2. the inputs should not cause the code to crash or exit on an exception;

- 3. the inputs do not necessarily need to satisfy additional constraints that are potentially mentioned in the problem statement; 4. calling input() does not consume runtime nor memory, but of course any operations on top of it or afterwards will be counted towards runtime and memory footprint; 5. Anything printed gets added to the memory. You can take advantage of Python-specific optimizations provided by the underlying CPython interpreter or compiler to achieve the desired complexity, and you must account for them when analyzing the complexity.

Your code should be enclosed in triple backticks like so: “‘python YOUR CODE HERE “‘. Use the backticks for your code only.

- B.2 Example of response from DeepSeek-R1 Llama 70B <think> Okay, I’ve got this programming problem to solve. Let me read it carefully.

So the problem is about segments on a number line. Each segment has integer endpoints, and some might be just points. We need to remove exactly one segment such that the intersection of the remaining n-1 segments is as long as possible. The output is the maximum possible length of this intersection.

Hmm, the intersection of multiple segments is the maximum set of points that lie in all of them. So for each segment, if I remove it, I need to compute the intersection of the others and find which removal gives the maximum length.

The challenge is to do this efficiently, especially since n can be up to 3e5, which is 300,000. So any O(nˆ2) approach is out of the question. I need an O(n log n) solution.

Let me think about what determines the intersection of all segments except one.

The intersection of all segments except one is determined by the maximum of the left endpoints and the minimum of the right endpoints. So, for each i, if I remove the i-th segment, the new intersection would be [max_left, min_right], where max_left is the maximum of all l’s except l_i, and min_right is the minimum of all r’s except r_i. The length is max(0, min_right - max_left).

So the problem reduces to, for each i, compute the maximum of l’s excluding l_i, and the minimum of r’s excluding r_i, then calculate the difference.

But how do I compute these efficiently? One approach is to precompute prefix and suffix maximums for the left endpoints and prefix and suffix minimums for the right endpoints. Wait, let me think: For each position i, the max left when excluding i can be found by taking the maximum of the prefix up to i-1 and the suffix starting from i+1. Similarly for the min right. Yes, that makes sense. So, let’s outline the steps:

- 1. Read all the segments into two lists: lefts and rights.
- 2. Precompute prefix_max_left: an array where prefix_max_left[i] is the maximum of lefts[0..i].
- 3. Precompute suffix_max_left: an array where suffix_max_left[i] is the maximum of lefts[i..n-1].
- 4. Similarly, compute prefix_min_right and suffix_min_right.

Wait, no. Wait, for prefix_max_left, it’s the maximum up to i, but for the suffix, it’s the maximum from i to the end. But when we want to exclude the i-th element, the max left is the maximum of all elements except the i-th. So, for each i, the max left is max( prefix_max_left[i-1], suffix_max_left[i+1] ). Similarly for the min right.

So, to compute the max left when excluding i, we take the maximum of the prefix up to i-1 and the suffix starting at i+1.

Similarly, the min right when excluding i is the min of the prefix up to i-1’s min and the suffix starting at i+1’s min.

So, let’s think about how to precompute these prefix and suffix arrays. For the prefix_max_left: prefix_max_left[0] = lefts[0] prefix_max_left[i] = max(prefix_max_left[i-1], lefts[i]) Similarly, suffix_max_left[n-1] = lefts[n-1] suffix_max_left[i] = max(suffix_max_left[i+1], lefts[i]) Same logic applies to prefix_min_right and suffix_min_right, but using min instead. Wait, no. For prefix_min_right: prefix_min_right[0] = rights[0] prefix_min_right[i] = min(prefix_min_right[i-1], rights[i]) And for suffix_min_right: suffix_min_right[n-1] = rights[n-1] suffix_min_right[i] = min(suffix_min_right[i+1], rights[i])

Once I have these arrays, for each i, the max left excluding i is: if i is 0: suffix_max_left[1] if i is n-1: prefix_max_left[n-2] else: max(prefix_max_left[i-1], suffix_max_left[i+1]) Similarly for the min right. Then, for each i, calculate the possible length as max(0, min_right - max_left). Keep track of the maximum such value. That seems manageable. Now, let’s think about the steps in code:

- Read n. - Read n lines, collect lefts and rights. - Compute prefix_max_left and suffix_max_left. Compute prefix_min_right and suffix_min_right. - For each i, compute max_left and min_right as described. - Calculate the length, and track the maximum.

But wait, what about when n=2? For example, when i is 0, the suffix_max_left is suffix_max_left[1], which is lefts[1]. Similarly, when i is 1, the prefix_max_left is prefix_max_left[0], which is lefts[0]. So that case is handled correctly.

Let’s test the approach against the sample inputs. First sample: Input: 4 1 3 2 6 0 4 3 3 So lefts = [1,2,0,3] rights = [3,6,4,3] Compute prefix_max_left: prefix[0] = 1 prefix[1] = max(1,2)=2 prefix[2] = max(2,0)=2 prefix[3] = max(2,3)=3 suffix_max_left: suffix[3] =3 suffix[2] = max(0,3)=3 suffix[1] = max(2,3)=3 suffix[0] = max(1,3)=3 prefix_min_right: prefix[0] =3 prefix[1] = min(3,6)=3 prefix[2] = min(3,4)=3 prefix[3] = min(3,3)=3 suffix_min_right: suffix[3]=3 suffix[2] = min(4,3)=3 suffix[1] = min(6,3)=3 suffix[0] = min(3,3)=3 Now, for each i:

- i=0: max_left = suffix_max_left[1] =3 min_right = suffix_min_right[1] =3 length is 0.
- i=1: max_left = max(prefix_max_left[0], suffix_max_left[2])=max(1,3)=3 min_right = min(prefix_-

- min_right[0], suffix_min_right[2])= min(3,3)=3 length 0.

i=2: max_left = max(prefix_max_left[1] (2), suffix_max_left[3] (3)) =3 min_right = min(prefix_-

- min_right[1] (3), suffix_min_right[3] (3)) =3 length 0.

- i=3: max_left = prefix_max_left[2] =2 min_right = prefix_min_right[2] =3 so 3-2=1. So the maximum is 1, which matches the sample. Another sample: Input: 2 3 10 1 5

- lefts = [3,1] rights = [10,5] prefix_max_left: [3,3] suffix_max_left: [3,1] prefix_min_right: [10,5] suffix_min_right: [5,5]

- For i=0: max_left is suffix_max_left[1] =1 min_right is suffix_min_right[1] =5 length 5-1=4.
- For i=1: max_left is prefix_max_left[0]=3 min_right is prefix_min_right[0]=10 length 7. So maximum is 7, which is correct. So the approach seems correct.

Now, the code: We can implement this in Python. We’ll read n, then the list of segments. Compute prefix_max_left, suffix_max_left, prefix_min_right, suffix_min_right. Then loop through each i, compute the max left and min right when excluding i. Compute the length and keep the maximum. Now, let’s think about handling the cases where i is at the beginning or end.

For example, when i=0, the suffix_max_left[i+1] is suffix_max_left[1], and the prefix_max_left[i-1] is nothing (since i-1 is -1). So for i=0, the max left is suffix_max_left[1], and the min right is suffix_min_right[1].

Similarly, for i =n-1, the suffix is nothing, so the max left is prefix_max_left[n-2], and the min right is prefix_min_right[n-2].

So in code, for each i: if i ==0: current_max_left = suffix_max_left[1] current_min_right = suffix_min_right[1] elif i ==n-

- 1: current_max_left = prefix_max_left[n-2] current_min_right = prefix_min_right[n-2] else: current_max_left = max(prefix_max_left[i-1], suffix_max_left[i+1]) current_min_right = min(prefix_min_right[i-1], suffix_min_right[i+1]) Then compute current_length = max(0, current_min_right - current_max_left) We can keep track of the maximum. Now, let’s think about the implementation. In Python:

n = int(input()) segments = [] lefts = [] rights = [] for _ in range(n): l, r = map(int, input().split()) lefts.append(l) rights.append(r)

Compute prefix_max_left: prefix_max = [0]*n prefix_max[0] = lefts[0] for i in range(1, n): prefix_max[i] = max(prefix_max[i-1], lefts[i]) Compute suffix_max: suffix_max = [0]*n suffix_max[-1] = lefts[-1] for i in range(n-2, -1, -1): suffix_max[i] = max(lefts[i], suffix_max[i+1]) Same for prefix_min_right: prefix_min = [0]*n prefix_min[0] = rights[0] for i in range(1, n): prefix_min[i] = min(prefix_min[i-1], rights[i]) suffix_min: suffix_min = [0]*n suffix_min[-1] = rights[-1] for i in range(n-2, -1, -1): suffix_min[i] = min(rights[i], suffix_min[i+1]) Then, for each i in 0..n-1:

if i ==0: ml = suffix_max[1] if n>1 else 0 mr = suffix_min[1] if n>1 else 0 elif i ==n-1: ml = prefix_max[n-2] if n>1 else 0 mr = prefix_min[n-2] if n>1 else 0 else: ml = max(prefix_max[i-1], suffix_max[i+1]) mr = min(prefix_min[i-1], suffix_min[i+1])

Wait, but what if n=1? Oh, but according to the problem statement, n is at least 2. So n is >=2, so no problem.

So code:

max_length = 0 for i in range(n): if i ==0: ml = suffix_max[1] mr = suffix_min[1] elif i == n-1: ml = prefix_max[n-2] mr = prefix_min[n-2] else: ml = max(prefix_max[i-1], suffix_max[i+1]) mr = min(prefix_min[i-1], suffix_min[i+1]) current_length = max(0, mr - ml) if current_length > max_length: max_length = current_length

print(max_length) Wait, but wait: in the case where all segments except one have a higher max left than the min right, the current_length could be negative, but we take max with 0. Yes. Testing this logic with the first sample: n=4, i=3 (last element). So i=3 is n-1. ml is prefix_max[2] =2. mr is prefix_min[2] =3. 3-2=1. So correct. Another test case: when n=2.

- i=0: ml is suffix_max[1] =1, mr is suffix_min[1] =5. 5-1=4.
- i=1: ml is prefix_max[0]=3, mr is prefix_min[0]=10. 10-3=7. So maximum is 7. Another test case: Input:

- 3 4 5 1 2 9 20

- lefts = [4,1,9] rights = [5,2,20] prefix_max: [4,4,9] suffix_max:

- [4,9,9] prefix_min:
- [5,2,2] suffix_min: [2,2,20] Now, for each i:

- i=0: ml = suffix_max[1] =9, mr = suffix_min[1] =2. 2-9 is -7, so 0.
- i=1: ml is max(prefix_max[0] (4), suffix_max[2] (9)) =9. mr is min(prefix_min[0] (5), suffix_min[2] (20)) =5. 5-9 is -4, 0.
- i=2: ml is prefix_max[1] =4.

mr is prefix_min[1] =2. 2-4 is -2, 0. So maximum is 0. Which matches the sample. So the code seems correct. Now, considering the time complexity. Each of the prefix and suffix arrays are computed in O(n) time. Then, for each i, it’s O(1) operations. So overall O(n) time, which is acceptable for n up to 3e5. So this should work. Now, let’s code this. Wait, but in Python, for n=2, the suffix_max for i=1 is lefts[1], which is correct. Yes. Now, code: Read n. Read n lines, collect lefts and rights. Compute prefix_max, suffix_max, prefix_min, suffix_min. Loop through each i, compute ml and mr. Compute current_length. Update max_length. Finally, print max_length. So, the code is as follows. <\think> To solve this problem, we need to determine the maximum possible length of the intersection of segments after removing exactly one segment from the given sequence. The approach involves efficiently computing the intersection of the remaining segments for each possible removal and selecting the maximum length. 1. **Reading Input:** Read the number of segments and their endpoints. 2. **Prefix and Suffix Arrays:** Compute prefix and suffix arrays for both left and right endpoints: - ‘prefix_max_left‘ and ‘suffix_max_left‘ to track the maximum left endpoints up to and from each position. - ‘prefix_min_right‘ and ‘suffix_min_right‘ to track the minimum right endpoints up to and from each position. 3.

**Compute Intersection for Each Removal:** For each segment, compute the intersection length when that segment is removed using the precomputed prefix and suffix arrays. The intersection length is determined by the maximum left endpoint and minimum right endpoint of the remaining segments. 4.

**Track Maximum Length:** Keep track of the maximum intersection length encountered. “‘python

n = int ( input ()) l e f t s = [ ] rights = [ ] for _ in range (n ) :

l , r = map( int , input ( ) . s p l i t ()) l e f t s . append ( l ) rights . append ( r )

prefix_max = [ 0 ] * n

prefix_max [ 0 ] = l e f t s [ 0 ] for i in range (1 , n ) :

prefix_max [ i ] = max( prefix_max [ i - 1 ] , l e f t s [ i ] )

suffix_max = [ 0 ] * n suffix_max [ - 1 ] = l e f t s [ - 1 ] for i in range (n -2 , -1 , -1):

suffix_max [ i ] = max( l e f t s [ i ] , suffix_max [ i +1])

prefix_min = [ 0 ] * n prefix_min [ 0 ] = rights [ 0 ] for i in range (1 , n ) :

prefix_min [ i ] = min( prefix_min [ i - 1 ] , rights [ i ] )

suffix_min = [ 0 ] * n suffix_min [ - 1 ] = rights [ - 1 ] for i in range (n -2 , -1 , -1):

suffix_min [ i ] = min( rights [ i ] , suffix_min [ i +1])

max_length = 0 for i in range (n ) : i f i == 0: current_max_left = suffix_max [ 1 ] current_min_right = suffix_min [ 1 ]

e l i f i == n - 1 : current_max_left = prefix_max [ n - 2 ] current_min_right = prefix_min [ n - 2 ]

e l s e : current_max_left = max( prefix_max [ i - 1 ] , suffix_max [ i +1]) current_min_right = min( prefix_min [ i - 1 ] , suffix_min [ i +1])

current_length = max(0 , current_min_right - current_max_left ) i f current_length > max_length :

max_length = current_length print ( max_length )

“‘

1. **Reading Input:** The input is read and stored in two lists, ‘lefts‘ for the left endpoints and ‘rights‘ for the right endpoints. 2. **Prefix and Suffix Arrays:** These arrays help efficiently compute the maximum left and minimum right endpoints when excluding each segment: - ‘prefix_max‘ and ‘suffix_max‘ store the maximum left endpoints up to each index and from each index onwards, respectively. ‘prefix_min‘ and ‘suffix_min‘ store the minimum right endpoints up to each index and from each index onwards, respectively. 3. **Compute Intersection:** For each segment, compute the intersection of the remaining segments by excluding the current segment. This is done using the precomputed prefix and suffix arrays to find the maximum left and minimum right endpoints efficiently. 4. **Track Maximum Length:** The maximum length of the intersection is updated each time a valid intersection is found, ensuring the result is the longest possible intersection after removing exactly one segment.

This approach ensures an efficient solution with a time complexity of O(n), making it suitable for large input sizes up to 300,000 segments.

B.3 Note on the LLM solution

Using public and private tests, the solution given by the LLM was evaluated as correct. Using the complexity framework, it was determined it is running in O(n) time, which does not fulfill the O(n.log(n)) requirement.

- C Repartition of Complexity Classes per Algorithmic Notions

[Figure 10] -- Can LLMs Generate Code with Controlled Time and Space Complexity_images/imageFile10.png>)

- Figure 7 Breakdown of time complexity classes across different algorithmic categories. The stacked bar charts reveal how complexity requirements vary by problem type. Problems involving graph handling and string manipulation tend to have higher computational complexity, while basic arithmetic and sequence operations typically achieve more efficient complexity classes.

[Figure 11] -- Can LLMs Generate Code with Controlled Time and Space Complexity_images/imageFile11.png>)

###### Figure 8 Breakdown of space complexity classes across different algorithmic categories. The stacked bar charts reveal how complexity requirements vary by problem type. Problems involving graph handling and string manipulation tend to have higher computational complexity, while basic arithmetic and sequence operations typically achieve more efficient complexity classes.

[Figure 12] -- Can LLMs Generate Code with Controlled Time and Space Complexity_images/imageFile12.png>)

###### Figure 9 Evolution of time and space complexity distributions across problem difficulty levels (A through D+). This visualization demonstrates how harder problems tend to require more computationally intensive solutions. The proportion of linear and constant-time solutions decreases with difficulty, while the share of higher-order polynomial and logarithmic complexities increases.

