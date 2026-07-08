## arXiv:2410.04717v3[cs.CL]18Oct2024

# Only-IF: Revealing the Decisive Effect of Instruction Diversity on Generalization

###### Dylan Zhang1, Justin Wang2 and Francois Charton3

1University of Illinois Urbana-Champaign, 2University Of Chicago, 3Meta

Understanding and accurately following instructions is critical for large language models (LLMs) to be effective across diverse tasks. In this work, we rigorously examine the key factors that enable models to generalize to unseen instructions, providing insights to guide the collection of data for instruction-tuning. Through controlled experiments, inspired by the Turing-complete Markov algorithm, we demonstrate that such generalization only emerges when training data is diversified enough across semantic domains. Our findings also reveal that merely diversifying within limited domains fails to ensure robust generalization. In contrast, cross-domain data diversification, even under constrained data budgets, significantly enhances a model’s adaptability. We further extend our analysis to real-world scenarios, including fine-tuning of specialist and generalist models. In both cases, we demonstrate that 1) better performance can be achieved by increasing the diversity of an established dataset while keeping the data size constant, and 2) when scaling up the data, diversifying the semantics of instructions is more effective than simply increasing the quantity of similar data. Our research provides important insights for dataset collation, particularly when optimizing model performance by expanding training data for both specialist and generalist scenarios. We show that careful consideration of data diversification is key: training specialist models with data extending beyond their core domain leads to significant performance improvements, while generalist models benefit from diverse data mixtures that enhance their overall instruction-following capabilities across a wide range of applications. Our results highlight the critical role of strategic diversification and offer clear guidelines for improving data quality.

#### 1. Introduction

The rapid advancements in large language models (LLMs) have revolutionized a wide range of tasks, including language comprehension [41], generation [3], knowledge-based question answering [15], and solving complex reasoning problems in fields like mathematics [11, 16] and programming [6, 1, 7, 25]. These successes hinge on the foundational capabilities of LLMs, such as knowledge retrieval, reasoning, planning, and notably, instruction following—enabled through instruction-tuning [34, 40, 45, 37]. Instruction-tuning trains LLMs on a wide variety of instruction-output pairs, allowing them to handle diverse prompts and better generalize to new tasks.

While knowledge retrieval focuses on accessing stored information and reasoning involves multi-step problem-solving, instruction following concerns the accurate interpretation and execution of diverse natural language prompts [55]. This capability is vital for user interaction, as it involves understanding the intent behind instructions and performing tasks without relying on complex logic [29]. Despite its importance, the mechanisms underlying instruction following remain less explored compared to other capabilities like reasoning.

The current research landscape on instruction tuning has produced varied and sometimes contradictory findings, ranging from the impact of dataset selection [54] to the effects of scaling up data [51, 52]. These

Corresponding author(s): Dylan Zhang; shizhuo2@illinois.edu, Francois Charton; fcharton@meta.com

disparate findings suggest that instruction following in LLMs is influenced by the scale and composition of fine-tuning data in complex ways [13, 53]. However, a systematic investigation into the effect of data on each core capability of LLMs —specifically isolating instruction following from reasoning and knowledge retrieval—has been limited. As a result, there is a lack in principled and practical guidelines on how to compose data to improve instruction-following capabilities.

Our work addresses this gap by focusing explicitly ONLY on the Instruction-Following capabilities of LLMs. We first introduce a systematic analysis of instruction diversity through a controlled symbolic task—string rewrites—inspired by the Turing-complete Markov algorithm [33]. By isolating the effects of instruction diversity, we focus on the model’s ability to follow instructions without conflating this with reasoning capabilities. Through controlled experiments, we examine the impact of instruction diversification across various semantic domains on the model’s ability to adapt to unseen instruction semantics. We find that generalization to unseen task semantics emerges ONLY IF the instructions are sufficiently diversified. Our findings reveal that diversification confined to limited domains does not guarantee robust generalization. In contrast, cross-domain diversification significantly enhances the model’s adaptability to new instructions, highlighting the importance of a more diverse training strategy.

10

5

0

HumanEval

Alpaca

MBPP

MathInstruct

OSS-Instruct Evol-Instruct

CruxEval

−5

LIMA

COT-Collection

StackOverﬂow

−10

ORCA-Math

0 5 10 15 20

(a) Semantic clustering of relevant datasets.

12

HumanEval

10

MBPP

OSS

Alpaca

8

6

4

2

0

2 4 6 8 10 12 14

(b) OSS-Alpaca mixture and test instructions.

- Figure 1: Visualization of embedded instructions.

14

OSS-Instruct

COT-Collection

12

Alpaca

HumanEval

MBPP

10

8

6

4

2

−2.5 0.0 2.5 5.0 7.5 10.0 12.5

(c) OSS-COT-Alpaca mixture and test instructions.

Additionally, our research offers and empirically verifies key insights into dataset collation strategies for both specialist and generalist LLMs. We show that when training specialist models (e.g., code language models), extending data diversification beyond their core domain yields substantial performance improvements in instruction following. For generalist models, using diverse data mixtures enhances their instruction-following capabilities across a broad range of applications, even for domains they were not explicitly trained on.

For both specialist and generalist models, we showed that our findings can imply 1) better performance than training on established datasets can be achieved by increasing the dataset’s diversity while maintaining the same data size, and 2) when scaling up the data, diversifying instruction semantics is more effective than simply increasing the amount of data from the same distribution.

These findings emphasize the importance of diversification in optimizing LLM performance and adaptability. Our study provides insights for dataset collation, especially when dataset expansion is needed, showing how proper data diversification boosts performance in both specialist and generalist LLM scenarios, and is

more effective than even multiplying dataset sizes. The rest of the paper is organised as follows:

We introduce the Markov-algorithm style rewrite experiments and our findings in Section 2. In Section 3 we generalize the rewriting by introducing abstraction, to simulate the situation of specialist and generalist training. In Section 4 we present the real-world application of our findings in specialist training, and in Section 5 we present the case for generalist.

#### 2. Experiments with String Replacements

##### 2.1. String Replacement Task

To start a systematic investigation of instruction-following, we model instruction-following tasks as stringreplacement operations, which are fundamental in theoretical computer science. String replacement forms the basis of Markov Algorithms [33], a Turing-complete model where sequences are iteratively transformed using ordered rewrite rules. These algorithms represent a structured, rule-driven process in which the first applicable rule is used to replace the leftmost matching substring, continuing until no further matches exist. This structured transformation mirrors the sequential, instruction-driven behavior we aim to explore. Appendix A provides illustrative examples of Markov algorithms in action.

In our study, we focus on a simplified form of this process. The replacement rule R—a pair like aa → bac—is applied to an input string ξ (e.g., caaba), yielding an output string τ (e.g., cbacba). The rule is applied to the leftmost occurrence of a match, and if no match exists, the original input remains unchanged. For instance, applying ι : iss → art to ξ = mississippi produces τ = martissippi, while applying ι to ξ = canada leaves the string unaltered.

This task, though simple, serves as a powerful proxy for instruction tuning by teaching models to handle structured, rule-based transformations. Training on these string rewrites allows models to internalize core aspects of instruction-following tasks, equipping them with the capability to generalize across a wide range of rule-based tasks without the overhead of more complex operations.

##### 2.2. Experiment Set-Up

We designed two string replacement tasks to evaluate the model’s ability to follow simple transformation rules: Basic Replacement: Apply the rule ι : x → y to any input string containing x. The model replaces the first occurrence of x with y and returns the modified string; and Conditional Replacement: Apply the rule ι : x → y if x is present in the input string; otherwise, return the input unchanged.

The model takes in ξ and ι, outputs either the transformed string or the original input if no match is found.

Task 1 focuses on fundamental rewrite operations, while Task 2 introduces a conditional decision-making aspect. This experimental setup allows us to assess the model’s capacity for both direct rule application and handling cases where no action is required.

In later sections, we also generalize the task from regular to context-sensitive, where x and y represent abstract structures e.g. x = a2 will represent all squared terms.

We train GPT-2 models (256 dimensions, 6 layers, 4 attention heads) on synthetic instruction/outcome pairs. The dataset contains S × I sequences, with S input strings and I replacement rules. Models are tested on 105 unseen examples to assess generalization across rules. Full training details are in Appendix B.

- 0.8

- 1.0

AverageAccuracy

101

103

105

Num.ExamplesPerInstruction

Average Accuracy

Num.Examples Per Instruction

(a) Re-writing accuracy against the number of instructions with a fixed-size training set.

100 200 300 400 500 1000 10000 100000

Number of Instructions

0.0

0.2

0.4

0.6

0.8

1.0

Accuracy

No-Op

Has-Op

- 1:9

- 2:8

- 3:7

- 4:6

- 5:5

Average

(b) Rewriting with no-op situation included.

Figure 2: Generalization versus the number of instructions during training.

- 2.3. Results

0.6

0.4

0.2

0.0

1020501002003004005006001k2k2.5k5k10k100k1000k

Num.Instructions

##### 2.3.1. Instruction Diversity Is Decisive To Generalization

Figure 2a presents the generalization accuracy for models trained on a fixed budget of I × S = 106 examples, as a function of the number I of different instructions in the training set. The number of examples per instruction (S) decreases as I increases. In these experiments, all input sequences feature at least one instance of the replacement rule.

In Figure 2a,a sharp step-shaped transition is observed: models trained on fewer than Imin (where Imin = 300) unique instructions consistently fail to generalize, regardless of example count per instruction. Conversely, models exposed to over Imax (where Imax = 1,000) distinct instructions generalize effectively to unseen instructions, even with minimal examples per instruction. This phase transition underscores our hypothesis that the sheer number of distinct instructions (I) is a key driver for generalization, affirming the necessity of cross-domain diversification for robust instruction-following for unseen instructions.

- 2.3.2. Diversification Allows Generalization In Case-Based Reasoning Set-Up

In earlier experiments, the model replaced a sub-string always present in the input. Now, we introduce a more complex task

1.0

0.9

0.8

0.7

0.6

Accuracy

0.5

0.4

Num. Instructions

0.3

1000

0.2

10000

0.1

100000

0.0

Uniform 2 1 0.5 0.2 0.1 0.07 0.03

Shape Parameter

Figure 3: Effect of long-tail task distributions on model’s generalization ability.

where some rules may not apply, requiring the model to decide whether a rule is applicable and either perform the operation or leave the input unchanged. This two-step process challenges the model to determine rule applicability and execute the correct action simultaneously.

To explore this, we introduce a third parameter in the training set: the frequency of “No-Ops” (instructions that cannot be satisfied), which we vary between 10% and 50%. The size of the training and test sets remains the same as in previous experiments,keeping the data size constant.

Figure 2b presents the generalization accuracies of trained models, as a function of the number of instructions and the frequency of No-Ops. Interestingly, despite No-Ops dominating the dataset1, the model generalizes well to unseen instructions after training on around 400 distinct cases. The proportion of No-Ops does not significantly affect generalization beyond that point, demonstrating that training on diverse instructions effectively teach the model to assess rule relevance and apply them accurately.

Overall, our conclusions remain consistent with previous experiments, albeit with a slightly lower number of instructions needed for generalization (400 vs. 500). This demonstrates the effectiveness of diversification in more complex scenarios involving case-based reasoning and rule relevance assessment.

##### 2.3.3. Imbalanced Distribution Is Still Effective In Driving Generalization

In previous experiments, instructions were evenly distributed between examples in the training set: in a training set of 1 million examples, with 500 different instructions, every instruction would be featured 2000 times.

Such a situation is unlikely to happen in real-world settings. In real-world training sets, some tasks will be much more common than others (due to the availability of fine-tuning data and the nature of the tasks).

To investigate the impact of the distribution of instructions on generalization to unseen tasks, we generate datasets of 1,000, 10,000 and 100,000 different instructions, and distribute the number of examples per instruction according to a power law distribution with PMF f(x) = αxα−1 where α is the shape parameter. By varying the shape parameter of the power law, we can generate a distribution of examples that range from close to uniform, to extremely peaked as shown in Fig. 7.

Figure 3 shows model generalization as a function of the power law’s shape parameter for training sets of N = 1 million examples with I1 = 1,000, I2 = 10,000, and I3 = 100,000 instructions. Models trained on I2 or more instructions are robust to the distribution of examples per instruction. For models trained on I1, generalization accuracy drops sharply when the shape parameter exceeds α = 0.2. In such cases, instructions with a probability lower than plow = 0.1% are barely represented, and the model effectively trains on fewer than Imin = 500 instructions, the minimum for generalizing to unseen instructions.

This observation suggests that generalization is achievable as long as there is sufficient semantic coverage, even if the data distribution is imbalanced. In practice, achieving uniform semantic coverage across the data may not be necessary to enable generalization. Instead, focusing on ensuring a broad enough range of semantics can still support effective model generalization, despite potential data imbalances.

1Consider a dataset containing 100,000 data points, 10% No-Ops, and 100 rules. No-Ops takes up 10,000 in total, ∼11× of any has-Ops.

##### 2.3.4. Semantic Diversification Boosts Task Performance

In real-world instruction-tuning, it is impractical to sample instructions uniformly from all semantic spaces due to their vastness. Instead, focusing on constrained sub-domains is crucial. We emulate this scenario by training models on instructions with semantic constraints, such as repeated characters (aaabbbccc for k = 3 - each character repeated 3 times), periodic patterns (abcabc for k = 2 a sub-string repeated 2 times), and mirrored structures (abccbaabc for k = 3 mirroring the substring for 3 times), and measure their generalization across different levels of k, where k is a parameter controlling the constrained-ness of rule semantics.

Our findings show that models trained on a single constrained sub-domain with high constraints (large k) fail to generalize to less constrained tasks (low k), indicating overfitting to specific patterns. Training on mixtures of constrained sets improves generalization, but only when the subspace is sufficiently rich. Larger instruction sets (e.g., 5000 examples) boost generalization, while highly restricted semantic domains (larger k) make it harder.

1.0

Range of k 3-4 3-6 3-12 3-20 3-30 3-60

0.9

| |
|---|

0.8

| |
|---|

0.7

| |
|---|

Accuracy

0.6

| |
|---|

0.5

| |
|---|

0.4

| |
|---|

0.3

0.2

0.1

0.0

500 1000 5000 10000 Num. Instructions

These results underscore that instruction diversity must span a range of semantically rich sub-spaces, not just rely on larger datasets within highly restricted domains, to foster robust gener-

- Figure 4: Model’s performance on k < 3 when trained on the three classes of restricted semantics as in 2.3.4. Models trained on 500 or less instructions never generalize to smaller k.

alization across tasks.

#### 3. Rewriting with Abstraction

In real-world applications, instruction-following often extends beyond simple "search-and-replace" tasks, requiring a model to abstract high-level concepts and ground them in specific contexts. To simulate more complex real-world instruction-following tasks, we extend the string-rewriting experiment by introducing two layers: abstract rules and their specific applications (groundings). Abstract rules represent high-level, task-independent patterns, while groundings refer to how these rules are instantiated in concrete input expressions. For example, in translation tasks, abstract rules might capture syntactic transformations, while their groundings apply these rules to specific phrases based on context.

Concretely, here we introduce a mathematical deduction task, which is a context-sensitive counterpart of the Markov string replacement task introduced in Section 2. Specifically, the model is tasked with simplifying algebraic expressions by applying specified deduction rules. Here, handling different abstract rules in mathematics is analogous to following instructions across distinct semantic domains. We ensure the task is inherently non-trivial for pre-trained language models (see ZS and FS performances on Figure 5a) to rule out confounding factors.

In this experiment, we present the model with a randomly generated mathematical expression I and an

abstract mathematical deduction rule ιabs : (X = Y) where X and Y are math expressions (e.g. an example abstract rule would be a2 − b2 = (a + b) × (a − b), to assess the model’s ability to identify the relevant

sub-expression in ξ (e.g. in (2x + 5)2 − (3y − 6)2, here a = 2x + 5,b = 3y − 6 ) and correctly apply the transformation (get (2x + 5 + 3y − 6) × (2x + 5 − (3y − 6))). We observe how well the model generalizes when trained on datasets of varying rule diversity.

A full example will be:

Rule: a2 − b2 = (a + b) × (a - b) Input: ((2x + 5)2 − (3y − 6)2)3 + (log(5t) − cos4k) Output: ( (2x + 5 + 3y - 6) × (2x + 5-(3y - 6)) )3 + (log(5t) − cos4k)

- 3.1. Data Generation We collate a set of distinct equational algebraic deduction rules of the form LHS = RHS.

Random Tree Generation We randomly construct mathematical expression trees of a specified depth d. Non-leaf nodes were systematically assigned operators (e.g., +, −, ∗, /), while leaf nodes were populated with variables, constants, or unary operations.

| | |0.80| |
|---|---|---|---|
| |D3 D4<br><br>| | |
| |D3 FS D4 FS<br><br>| | |
| |0.49<br><br>D30.52ZS<br><br>|0.58| |
| |D4 ZS| | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |
| | | | |

0.8

0.7

0.6

0.5

Accuracy

0.44

0.4

0.33

0.3

0.2

0.1

0.0

10 20 40

Rules

(a) Accuracy On Unseen Deduction Rules vs. Abstract Rule Diversity. D3:pattern with tree depth 3. ZS:Zero shot. FS: Few shot.

0.36 In-Domain Rules

0.35

0.33

20 40

0.30

0.25

###### Accuracy

0.20

0.18

0.15

0.10

0.07

0.06

0.05

5 20 40

Num. Diver Rules

(b) Accuracy on unseen depths against number of diversification rules with the same in-domain / outof-domain mixture.

- Figure 5: Rewriting With Abstraction.

| |0.40| | |
|---|---|---|---|
| | | | |
| | | | |
| |0.27<br><br>0.26| | |
| |0.22| | |
| |Diver-20 Diver-40<br><br>| | |
| | | | |

0.40

0.35

###### Accuracy

0.30

0.25

0.20

0.19

0.16

0.15

1:49 5:45 10:40

In-Domain vs Out-of-Domain

(c) Accuracy on unseen depths vs. In-domain/out-of-domain combination. Diver- means number of diversification rules. X axis ticks are |Rdiver|: |Rspec|.

Producing Pattern-Carrying Sub-Expressions To generate expressions, a pattern-carrying sub-tree was generated with a depth of dp, denoting the depth of expression tree for each entry in the rule. e.g. we may replace a with (y + 2x + 5) with dp = 2. We then randomly choose a leaf node and swap it with the concrete sub-expression.

##### 3.2. Simulating Specialist and Generalist Training

We examine two common settings for instruction-tuning: generalist, and specialist training via the set of simulated experiments. We fine-tune all models from a pre-trained Mistral-7B-v0.3 [19] checkpoint.

- 3.2.1. Diversifying Instruction Semantics Empowers Better Generalist LMs

We control the number of training instances (50K), and vary the number of abstract deduction rules & the instantiated rules per abstract rule. We test the models on unseen sequences and unseen abstract rules. We train the model on triples of (ξ, ιabs, τ) pairs where τ is the result of applying ιabs to ξ. The result is shown in Figure 5a.Our findings demonstrate a clear advantage of increased rule diversity, consistent with previous string rewriting experiments. Holding the number of training instances constant, we observed that expanding the diversity of rules (ι) in the training data significantly enhances the model’s ability to generalize to unseen abstract rules during testing. This improvement is achieved despite the model encountering fewer groundings per rule and expression. These results not only validate the hypothesis from our string rewriting experiments but also underscore that increased diversity in training data significantly enhances the model’s ability to concretize abstract instructions, improving generalization even with limited exposure to specific instances. This underscores the importance of diversity in developing models that can unseen tasks.

- 3.2.2. Sweet Point Between Specialization and Diversification For Optimal Specialist Performance

In this experiment, we simulate a scenario where a model is trained as a specialist and tested on out-ofdistribution queries that still belong to the same overall instruction type. To achieve this, we divide the set of rules R into two categories: specialized rules Rspec and diversification rules Rdiver = R \ Rspec. The training data is constructed as a mixture of instances generated from these two sets of rules. Specifically, the instances based on the specialized rules use patterns with depth dtrainp < dtestp to simulate out-of-distribution test instances, while the instances from the diversification rules add variety to the training. This setup allows us to examine the trade-off between specialization and diversification for better instruction following in this specialized task.

Figure 5c, the results exhibit a clearly peaked structure as we incorporate more out-of-domain data for diversification. We demonstrated within a fixed budget, such trade-off indeed exists between specialization and enhanced instruction-following via training on a more diverse set of instructions. Figure 5b shows the trend when we diversify across an increasingly rich semantics. We notice the benefit of a more diverse Rdiver, which suggests that even when diversifying for specialists, one should be mindful to curate a dataset that spans over wider domains.

#### 4. Fine-Tuning A Sepcialist Instruction-Follower: Case of Code Generation

Building on our foundational insights from controlled string-rewriting tasks, we extend our analysis to training a real-world specialist instruction-follower.

Here, we investigate how cross-domain instruction diversity impacts the model’s ability to handle realworld tasks that require a nuanced balance between instruction-following and domain-specific expertise by the example of code generation.

Code generation is primarily an instruction-following task, where models translate explicit prompts into executable code. Pre-trained code LLMs, having encountered common coding patterns during pre-training on

|Base Budget (OSS)<br><br>+OSS +Alpaca +CoT<br><br>|HE HE+ MBPP MBPP+<br><br>|Avg (Base)<br><br>Avg (+)<br><br>Avg<br><br>Rel. Gain|
|---|---|---|
|15K 5K 0 0 15 4 1 0 15 3 2 0 15 2 3 0 15 1 4 0 15 0 5 0<br><br>|62.2 56.7 75.7 62.4<br><br>67.1 59.8 75.7 62.2<br><br>68.3 61.6 75.4 63.2<br><br><br>64.6 60.4 76.4 63.4<br><br>65.2 58.5 76.7 63.7<br><br><br>64.6 57.9 73.9 61.4<br><br>|68.9 59.5 64.2 71.4 61 66.2 3.1 71.9 62.4 67.1 4.6 70.5 61.9 66.2 3.1<br><br>71 61.1 66 2.8<br><br>69.3 59.7 64.5 0.5<br>|
|60 15 0 0 60 0 15 0 60 0 7.5 7.5 60 7.5 7.5 0 60 7.5 3.25 3.25 60 12.5 2.5 0 60 12.5 1.25 1.25 60 14 1 0 60 14 0.5 0.5<br><br>|66.5 61.6 75.4 61.9<br><br>67.1 59.8 77 64.8<br><br>64.6 59.1 76.2 63.8<br><br>68.9 61.6 76.2 63.8<br><br><br>66.5 62.2 76.2 64.3 68.3 61 76.2 64.3<br><br>68.3 62.2 77.8 65.1<br><br>69.5 62.2 76.2 64.3<br><br><br>66.5 61 76.5 63.2<br><br>|71 61.8 66.4 72.1 62.3 67.2 1.2<br><br>70.4 61.5 65.9 -0.6 72.6 62.7 67.6 1.9<br>71.4 63.3 67.3 1.4<br>72.3 62.7 67.5 1.6<br><br>73.1 63.7 68.4 3 72.9 63.3 68.1 2.5 71.5 62.1 66.8 0.7<br><br><br>|

- Table 1: Results on DeepSeek-Coder-6.7B and comparison With MagiCoder-DS-6.7B [46]. Plum-colored row surpasses the performance of full-data training. Best configurations corresponding to each setting are highlighted. We demonstrated that one could achieve higher performances by means of diversification.

large code corpora, are fine-tuned to utilize these structures effectively. Unlike reasoning tasks that demand creative problem-solving, code generation focuses on adhering to established patterns and passing test cases, emphasizing precise procedural execution over complex reasoning.

In this experiment, we aim to demonstrate that a diverse set of instructions can significantly enhance a model’s adaptability to new coding instructions. This analysis provides crucial evidence for optimizing instruction-tuning datasets in real-world scenarios where both precision and generalization are paramount.

##### 4.1. Experiments

To rigorously evaluate the impact of cross-domain instruction diversity, we employ two widely-used code generation benchmarks: HumanEval [6] and MBPP [1], alongside the augmented EvalPlus [30]. These benchmarks present a diverse array of coding challenges that test the model’s ability to interpret and execute novel instructions. As our training dataset, we use 20,000 instances sampled from the OSS-Instruct [46], a synthetic coding dataset, which has been sanitized to avoid data contamination. We also incorporate general-domain instruction data from Alpaca [40], a well-known dataset that covers a wide semantic domain. By gradually replacing code-specific instruction data with general-domain instructions, we assess how this cross-domain diversity influences the model’s performance in code generation. We utilize two state-of-the-art pre-trained code language models as base models: DeepSeek-Coder-6.7B-Base [14] and CodeQwen-7B-Base [2].

##### 4.2. Striking the Right Balance Between Coding and Diverse Data

The Role of Semantic Diversity in Generalization Our results in Tables 2 and 1 highlight a crucial insight: while increasing the size of coding datasets may seem like the obvious solution for improving performance, this strategy is not always optimal. In fact, diversifying instruction domains leads to greater improvements. For example, adding data from general-purpose QA (Alpaca) and reasoning tasks (CoT) significantly outperforms incorporating an equivalent amount of coding data. Notably, the Plum-colored configuration in Table 1, which uses only 20,000 data points, surpasses the performance of models trained on 75,000 coding data points.

This pattern is consistent with our synthetic experiments in Section 3.2.2, where diverse instructions enabled the model to generalize better to out-of-distribution task instances for seen abstract instructions. Even in the basic string-replacement experiments (Section 2.3.4), we observed that introducing varied instructions—such as those in the Alpaca dataset—expanded the semantic range and boosted code generation performance, even when the quantity of each instruction is low compared to the main task (the long-tail distribution scenario in Section 2.3.3).

The Power of Cross-Domain Diversification Figure 1a demonstrates how instruction-tuning datasets cluster within specific semantic sub-spaces. By incorporating datasets like Alpaca, which is designed for human language interaction, and the CoT-Collection [21], which challenges the model with complex reasoning tasks, we extend the model’s exposure to diverse semantic spaces. This further improves the model’s generalization capabilities across domains.

Results from CodeQwen, presented in Table 2 support this conclusion. Models trained on a balanced mixture of coding, general QA, and reasoning data outperform those trained solely on a mix of coding data and Alpaca, reinforcing the importance of cross-domain diversification for handling complex and varied instructions.

Avg. (Base) Avg. (EvalPlus) Avg.

DeepSeek

CodeQwen

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

0.72

0.70

0.68

Pass@1

0.66

Balancing Generalization and Specialization While diversifying instruction data offers substantial benefits, there are limits. As shown in Figure 6, incorporating general-domain instructions initially boosts the model’s ability to follow natural language specifications, leading to improved Pass@1 scores for code generation. However, as more non-coding data is added, the model’s capacity to handle the nuanced requirements of coding tasks diminishes.

0.64

0.62

0.60

20K:019K:1K18K:2K17K:3K16K:4K15K:5K

20K:019K:1K18K:2K17K:3K16K:4K15K:5K

Figure 6: Sweet spots of Pass@1 with data mixture. Baseline is marked with dotted lines.

This trade-off echoes our findings from the synthetic experiments in Section 3.2.2: achieving optimal performance requires balancing general instruction-following capabilities with specialized domain knowledge [28]. Figure 6 illustrates the plateau and eventual decline in performance, underscoring the importance of calibrating the mixture of coding and non-coding data to achieve the best overall results.

The real-world experiment reinforces the insights from our synthetic experiments, emphasizing the importance of instruction diversity.

#### 5. Fine-tuning Generalist LLMs

In this section, we evaluate the impact of cross-domain instruction diversification on general reasoning tasks and investigate the optimal high-level strategy to improve the quality of a dataset.

|Base Budget (OSS)<br><br>+ OSS<br><br>+ Alpaca<br><br>+ CoT|HE HE+ MBPP MBPP+<br><br>Avg (Base)<br><br>Avg (+)<br><br>Avg<br><br>Rel.Gain Intra -Budget<br><br>Rel.. Gain wrt. No Diver.|
|---|---|
|19K 1K 0 0 19 0 1 0 18 0 2 0 18 0 1 1 16 0 4 0 16 0 2 2<br><br>|65.9 59.8 73.7 62.2 69.8 61.0 65.4 - -<br><br>67.7 61.6 75.9 63.4 71.8 62.5 67.2 2.8 2.8 69.5 63.4 76.4 63.2 73.0 63.3 68.1 - 4.1<br><br>68.9 63.4 77.4 64.2 73.2 63.8 68.5 0.6 4.7 65.2 58.5 76.7 63.7 71.0 61.1 66.0 - 0.9 68.3 61.6 77.2 63.7 72.8 62.7 67.7 2.6 3.5<br><br><br>|

- Table 2: Pass@1 with CodeQwen-7B. We highlighted the accuracies surpassing baseline result with green and those falling below with red.

Base Data Budget

+UI +OO +AL Total

Overall Avg.

IF Avg.

Overall wo IF

Overall. wo Math

Rel. Gain + 20K Data Budget

Rel. Gain Diver. w./In Budget 10K 0 0 39.07 25.32 47.27 44.94 - -

10K 0 10 0 44.93 43.85 49.03 48.77 - 15.00

0 0 10

20K

44.78 36.43 51.51 50.05 - 14.64

20 0 0 41.01 23.99 50.27 48.97 4.97 0 20 0 46.87 46.25 50.96 49.38 19.99 14.31 0 0 20 44.41 37.13 50.86 49.21 13.68 8.31 10 10 0 47.59 46.58 51.64 50.44 21.81 16.05

20K

10 0 10

40K

43.47 30.84 51.50 49.51 11.27 6.01

20 0 0 40.39 22.01 50.43 46.34 -1.50 0 20 0 45.64 46.69 49.75 52.45 11.31 13.00 0 0 20 40.64 31.93 47.86 48.62 -0.90 0.61

- 10 10 0 45.79 40.10 51.89 51.08 11.66 13.37

40K

10 0 10

60K

42.73 31.30 50.63 48.45 4.21 5.79 20 0 0 41.71 24.67 51.62 47.97 3.27 -

0 20 0 45.86 42.34 51.19 50.50 13.54 9.95 0 0 20 43.27 33.02 50.83 48.39 7.14 3.75

- 10 10 0 46.07 40.82 51.93 51.17 14.06 10.45

60K

10 0 10

80K

42.66 30.58 50.49 47.36 5.63 2.29

- Table 3: The table shows the performance of generalist models trained with different data mixtures. UI refers to UltraInteract, OO refers to OpenOrca, and AL refers to Alpaca. The column labeled Rel. Gain + 20K Data indicates the relative performance gain of the model in the current row compared to the UI-only baseline that uses 20K fewer data points. For example, the performance of a model trained on 40K data will be compared to the baseline model trained on 20K UI data, as indicated by the blue row above the current data quantity.Rel. Gain Diver. denotes the gain of diversification compared to the baseline with the same data budget.

##### 5.1. Experimental Setup

In this study, we evaluate the impact of cross-domain instruction diversification on large language models (LLMs) by comparing our approach with a baseline model trained exclusively on UltraInteract-SFT dataset [50]. UltraInteract-SFT is a collection of complex, multi-step reasoning problems emphasizing on math problem-solving, code generation, and logical reasoning, promoting robust reasoning and planning capabilities in LLMs.

While UltraInteract-SFT primarily focuses on math and coding problems and contains a rich collection of those problems, its scope is limited to these domains. OpenOrca [26] and Alpaca, though sparse and varied, introduce broader instruction-following tasks.

To assess the effectiveness of cross-domain instruction diversity, we constructed a training set that includes a mixture of UltraInteract-SFT, OpenOrca, and Alpaca datasets. While UltraInteract-SFT is rich in math, coding and complex QA problems, it remains limited to primarily these domains despite its challenging and

diverse nature within them. OpenOrca and Alpaca, on the other hand, introduce instruction-following tasks across a broader range of domains, enriching the training data with varied instruction types. We gauged the model’s overall capabilities using the same set of benchmarks consisting of coding [1, 6], math [16, 11, 8], knowledge (MMLU [15]), instruction following [55] and chain-of-thought reasoning [39] as [50] and computed average performance.

To reflect on its precision in instruction following, we adopted IF-Eval [55] benchmark, comprising over 500 prompts for rigorous instruction-following tests. We followed a core-set selection approach when curating datasets of various budgets. We finetune from a pre-trained Mistral-7B-v0.3 checkpoint for all results in table 3.

In the experiments, we study the effect of adding controlled quantities of different types of data to various pre-defined base budgets, and compare the performances across budgets to exhibit the advantage of dataset expansion along the dimension of improving diversity.

##### 5.2. Data Diversity Matters More Than Quantity For Generalists

Our findings emphasize that expanding the model’s exposure to varied domains leads to superior overall performance, underscoring the importance of data variety over sheer volume in enhancing model robustness and adaptability. Table 3 demonstrates the clear advantage of training models with diversified data across different domains, a phenomenon observed across various data budgets. By simulating the process of composing data up to a predefined budget, we show that the model benefits from composition that includes data diversifying beyond UI’s domain. Additionally, when increasing the overall data quantity, comparisons across different Total budgets reveal that incorporating diverse data improves performance more than simply increasing the size of the UI dataset.

The key takeaway is that models exposed to a broad range of domains consistently achieve better performance than those trained solely on domain-specific reasoning data. Apart from the benefits from significantly enhanced instruction-following, this holds true even when tested on tasks that demand strong reasoning skills, which reinforces the results discussed in Section 3.2.1.

The results suggest that when aiming to improve a model’s overall capability through dataset expansion, it’s more effective to prioritize diverse datasets rather than simply increasing the volume of data from a specific domain. Our findings highlight that exposing models to varied domains enhances their overall performance, emphasizing the importance of diversity in training data for building robust and adaptable models, compared to focusing on dataset size alone.

#### 6. Related works

Datasets for instruction-tuning. Many datasets for instruction-tuning have been proposed. The best quality is achieved for sets collated by human annotators [20, 49, 38, 42, 32, 12, 23], but their size is constrained by the cost of annotation. Alternative methods, which use large language models to generate instruction sets, have been proposed [44, 17, 40, 35, 9, 47, 22, 21]. They provide larger instruction sets, at the cost of reduced annotation quality.

Data curation for instruction-tuning. It is widely recognized that the quality of instruction-tuning datasets has a massive impact on the performance of fine-tuned models. Previous works acknowledged the contributions of several key factors. Most research on the subject insist on the importance of the size and quality of the instruction sets [10, 18, 43]. Liang et al. [27] point out the importance of consistent formats.

Several recent works [54, 5] suggest that models fine-tuned on carefully selected examples can achieve high performance with small datasets. Various strategies for data curation have been proposed, focusing on instruction diversity, and the quality of answers [54, 5, 48, 24, 31]. Several authors discuss the benefit of mixing tasks from different categories [32, 18, 4]. Closest to our work, Dong et al. [13] discuss the impact of mixing general and domain-specific instructions, in order to achieve the best results with the smallest dataset.

#### 7. Conclusion

In this work, we systematically studied instruction-following of language models via a suite of carefully designed controlled experiments. By introducing a symbolic string rewrite framework, we provide a focused model to isolate and study models’ instruction-following abilities, offering a new perspective on this fundamental capability. Our experiments further demonstrate that instruction diversity, even within a fixed data budget, plays a critical role in improving model generalization. This finding underscores the value of diverse instruction semantics over large dataset size, in enhancing performance across both specialized and generalized LLM applications. Finally, we offer practical insights into dataset collation strategies, highlighting that proper diversification can significantly outperform dataset expansion.

#### References

- [1] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program synthesis with large language models, 2021.
- [2] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report, 2023.
- [3] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020.
- [4] Alexander Bukharin and Tuo Zhao. Data diversity matters for robust instruction tuning, 2024.
- [5] Yihan Cao, Yanbin Kang, Chi Wang, and Lichao Sun. Instruction mining: When data mining meets large language model finetuning, 2023.
- [6] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet,

- Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code, 2021.
- [7] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code, 2021.
- [8] Wenhu Chen, Ming Yin, Max Ku, Pan Lu, Yixin Wan, Xueguang Ma, Jianyu Xu, Xinyi Wang, and Tony Xia. Theoremqa: A theorem-driven question answering dataset, 2023. URL https://arxiv.org/ abs/2305.12524.
- [9] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, March 2023. URL https://lmsys.org/blog/ 2023-03-30-vicuna/.
- [10] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Alex Castro-Ros, Marie Pellat, Kevin Robinson, Dasha Valter, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Zhao, Yanping Huang, Andrew Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. Scaling instruction-finetuned language models, 2022.
- [11] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021.
- [12] Mike Conover, Matt Hayes, Ankit Mathur, Jianwei Xie, Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell, Matei Zaharia, and Reynold Xin. Free dolly: Introducing the world’s first truly open instruction-tuned llm, 2023. URL https://www.databricks.com/blog/2023/04/12/ dolly-first-open-commercially-viable-instruction-tuned-llm.
- [13] Guanting Dong, Hongyi Yuan, Keming Lu, Chengpeng Li, Mingfeng Xue, Dayiheng Liu, Wei Wang, Zheng Yuan, Chang Zhou, and Jingren Zhou. How abilities in large language models are affected by supervised fine-tuning data composition, 2024. URL https://openreview.net/forum?id=6M5G5hNiAU.
- [14] Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y. Wu, Y. K. Li, Fuli Luo, Yingfei Xiong, and Wenfeng Liang. Deepseek-coder: When the large language model meets programming – the rise of code intelligence, 2024.

- [15] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding, 2021. URL https://arxiv.org/ abs/2009.03300.
- [16] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset, 2021. URL https://arxiv.org/abs/2103.03874.
- [17] Or Honovich, Thomas Scialom, Omer Levy, and Timo Schick. Unnatural instructions: Tuning language models with (almost) no human labor, 2022.
- [18] Srinivasan Iyer, Xi Victoria Lin, Ramakanth Pasunuru, Todor Mihaylov, Daniel Simig, Ping Yu, Kurt Shuster, Tianlu Wang, Qing Liu, Punit Singh Koura, et al. Opt-iml: Scaling language model instruction meta learning through the lens of generalization. arXiv preprint arXiv:2212.12017, 2022.
- [19] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b, 2023. URL https://arxiv.org/abs/2310.06825.
- [20] Daniel Khashabi, Sewon Min, Tushar Khot, Ashish Sabharwal, Oyvind Tafjord, Peter Clark, and Hannaneh Hajishirzi. UNIFIEDQA: Crossing format boundaries with a single QA system. In Trevor Cohn, Yulan He, and Yang Liu, editors, Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1896–1907, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/ v1/2020.findings-emnlp.171. URL https://aclanthology.org/2020.findings-emnlp.171.
- [21] Seungone Kim, Se June Joo, Doyoung Kim, Joel Jang, Seonghyeon Ye, Jamin Shin, and Minjoon Seo. The cot collection: Improving zero-shot and few-shot learning of language models via chain-of-thought fine-tuning, 2023.
- [22] Abdullatif Köksal, Timo Schick, Anna Korhonen, and Hinrich Schütze. Longform: Optimizing instruction tuning for long text generation with corpus extraction, 2023.
- [23] Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi-Rui Tam, Keith Stevens, Abdullah Barhoum, Nguyen Minh Duc, Oliver Stanley, Richárd Nagyfi, Shahul ES, Sameer Suri, David Glushkov, Arnav Dantuluri, Andrew Maguire, Christoph Schuhmann, Huu Nguyen, and Alexander Mattick. Openassistant conversations – democratizing large language model alignment, 2023.
- [24] Ming Li, Lichang Chen, Jiuhai Chen, Shwai He, Jiuxiang Gu, and Tianyi Zhou. Selective reflectiontuning: Student-selected data recycling for llm instruction-tuning, 2024.
- [25] Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien de Masson d’Autume, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov, James Molloy, Daniel J. Mankowitz, Esme Sutherland Robson, Pushmeet Kohli, Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals. Competition-level code generation with alphacode. Science, 378(6624):1092–1097, December 2022. ISSN 1095-9203. doi: 10.1126/science.abq1158. URL http://dx.doi.org/10.1126/science.abq1158.
- [26] Wing Lian, Bleys Goodson, Eugene Pentland, Austin Cook, Chanvichet Vong, and "Teknium". Openorca: An open dataset of gpt augmented flan reasoning traces. https://https://huggingface.co/ Open-Orca/OpenOrca, 2023.

- [27] Shihao Liang, Runchu Tian, Kunlun Zhu, Yujia Qin, Huadong Wang, Xin Cong, Zhiyuan Liu, Xiaojiang Liu, and Maosong Sun. Exploring format consistency for instruction tuning, 2024.
- [28] Chen Ling, Xujiang Zhao, Jiaying Lu, Chengyuan Deng, Can Zheng, Junxiang Wang, Tanmoy Chowdhury, Yun Li, Hejie Cui, Xuchao Zhang, Tianjiao Zhao, Amit Panalkar, Dhagash Mehta, Stefano Pasquali, Wei Cheng, Haoyu Wang, Yanchi Liu, Zhengzhang Chen, Haifeng Chen, Chris White, Quanquan Gu, Jian Pei, Carl Yang, and Liang Zhao. Domain specialization as the key to make large language models disruptive: A comprehensive survey, 2024.
- [29] Emmy Liu, Graham Neubig, and Jacob Andreas. An incomplete loop: Instruction inference, instruction following, and in-context learning in language models, 2024. URL https://arxiv.org/abs/2404. 03028.
- [30] Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation, 2023.
- [31] Liangxin Liu, Xuebo Liu, Derek F. Wong, Dongfang Li, Ziyi Wang, Baotian Hu, and Min Zhang. Selectit: Selective instruction tuning for large language models via uncertainty-aware self-reflection, 2024.
- [32] Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V. Le, Barret Zoph, Jason Wei, and Adam Roberts. The flan collection: Designing data and methods for effective instruction tuning, 2023.
- [33] A. A. Markov. The theory of algorithms, volume 42. Acad. Sci. USSR, 1954.
- [34] Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022. URL https: //arxiv.org/abs/2203.02155.
- [35] Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruction tuning with gpt-4, 2023.
- [36] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners, 2019. URL https://api.semanticscholar.org/CorpusID: 160025533.
- [37] Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Fevry, Jason Alan Fries, Ryan Teehan, Tali Bers, Stella Biderman, Leo Gao, Thomas Wolf, and Alexander M. Rush. Multitask prompted training enables zero-shot task generalization, 2022. URL https://arxiv.org/abs/2110.08207.
- [38] Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen,

- Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Fevry, Jason Alan Fries, Ryan Teehan, Tali Bers, Stella Biderman, Leo Gao, Thomas Wolf, and Alexander M. Rush. Multitask prompted training enables zero-shot task generalization, 2022.
- [39] Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, and Jason Wei. Challenging big-bench tasks and whether chain-of-thought can solve them, 2022. URL https://arxiv.org/abs/2210.09261.
- [40] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https:// github.com/tatsu-lab/stanford_alpaca, 2023.
- [41] Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems, 2020.
- [42] Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Gary Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Maitreya Patel, Kuntal Kumar Pal, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Shailaja Keyur Sampat, Savan Doshi, Siddhartha Mishra, Sujan Reddy, Sumanta Patro, Tanay Dixit, Xudong Shen, Chitta Baral, Yejin Choi, Noah A. Smith, Hannaneh Hajishirzi, and Daniel Khashabi. Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks, 2022.
- [43] Yizhong Wang, Hamish Ivison, Pradeep Dasigi, Jack Hessel, Tushar Khot, Khyathi Raghavi Chandu, David Wadden, Kelsey MacMillan, Noah A. Smith, Iz Beltagy, and Hannaneh Hajishirzi. How far can camels go? exploring the state of instruction tuning on open resources, 2023.
- [44] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions, 2023.
- [45] Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners, 2022. URL https://arxiv.org/abs/2109.01652.
- [46] Yuxiang Wei, Zhe Wang, Jiawei Liu, Yifeng Ding, and Lingming Zhang. Magicoder: Source code is all you need, 2023.
- [47] Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions, 2023.
- [48] Yang Xu, Yongqiang Yao, Yufan Huang, Mengnan Qi, Maoquan Wang, Bin Gu, and Neel Sundaresan. Rethinking the instruction quality: Lift is what you need, 2023.
- [49] Qinyuan Ye, Bill Yuchen Lin, and Xiang Ren. Crossfit: A few-shot learning challenge for cross-task generalization in nlp, 2021.
- [50] Lifan Yuan, Ganqu Cui, Hanbin Wang, Ning Ding, Xingyao Wang, Jia Deng, Boji Shan, Huimin Chen, Ruobing Xie, Yankai Lin, Zhenghao Liu, Bowen Zhou, Hao Peng, Zhiyuan Liu, and Maosong Sun.

- Advancing llm reasoning generalists with preference trees, 2024. URL https://arxiv.org/abs/ 2404.02078.
- [51] Liang Zeng, Liangjun Zhong, Liang Zhao, Tianwen Wei, Liu Yang, Jujie He, Cheng Cheng, Rui Hu, Yang Liu, Shuicheng Yan, Han Fang, and Yahui Zhou. Skywork-math: Data scaling laws for mathematical reasoning in large language models – the story goes on, 2024. URL https://arxiv.org/abs/2407. 08348.
- [52] Biao Zhang, Zhongtao Liu, Colin Cherry, and Orhan Firat. When scaling meets LLM finetuning: The effect of data, model and finetuning method. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=5HCnKDeTws.
- [53] Xinlu Zhang, Zhiyu Zoey Chen, Xi Ye, Xianjun Yang, Lichang Chen, William Yang Wang, and Linda Ruth Petzold. Unveiling the impact of coding data instruction fine-tuning on large language models reasoning,

2024. URL https://arxiv.org/abs/2405.20535.

- [54] Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. Lima: Less is more for alignment, 2023.
- [55] Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models, 2023. URL https://arxiv. org/abs/2311.07911.

#### A. Complement on Markov algorithms

Markov algorithms [33] are ordered sets of rewrite rules, operating on sequences of symbols in a fixed alphabet U. A sequence S is processed by applying the first rewrite applicable to S, at the leftmost position if several exist: i.e. the rewrite rule ss → tr transforms the sequence S = mississipi into S′ = mitrissipi. The algorithm is then applied to S′, and the process is repeated until either no rules apply, and the algorithm is said to be blocked, or a special rule, called a stop rule is invoked, and the algorithm terminates and returns the final rewritten sequence.

Specifically, the algorithm uses and alphabet A, which includes the alphabet U used buy the sequences to be processed (henceforth, small case latin letters), a set of additional symbols (henceforth, the small case greek letters {α, β . . . }, and a special symbol · indicating a stop rule.

For instance, we could define the following algorithm, with U = {a, b}, and A = {a, b, α, β, ·}, and the rules

αx → xαβx (1) βxy → yβx (2) αβx → xα (3)

α → · (4) → α (5)

where x and y stand for any letter a or b. This will transform any sequence of a and b into a concatenation of the sequence and its reverse. Applied on abb, the algorithm will perform the following rewrites:

abb → αabb (by 5)

αabb → aαβabb (by 1) aαβabb → aαbβab (by 2) aαbβab → abαβbβab (by 1)

abαbβbβab → abαβbbβa (by 2) abαβbbβa → abαbβbβa (by 2) abαbβbβa → abbαβbβbβa (by 1)

abbαβbβbβa → abbbαβbβa (by 3) abbbαβbβa → abbbbαβa (by 3) abbbbαβa → abbbbaα (by 3) abbbbaα → abbbba (by 4)

Since rule 4 is a stop rule, the algorithm terminates and returns abbbba. Judicious introduction of additional (greek) letters allows one to compose Markov algorithms, effectively

writing complex programs. Any effective process (i.e. finite computation) can be represented as a Markov algorithm (this is Markov’s thesis).

#### B. Experimental set-up

- B.1. Model and Training

In rewrite experiments, we train GPT-2 models [36], a decoder-only transformer-based architecture, with 6 layers, 256 dimensions and 4 attention heads from scratch, on a generated instruction-tuning dataset using standard supervised fine-tuning approach. We use the AdamW optimizer, a learning rate of 10−3, and linear scheduling. All models are trained for 50 epochs. For the encrypted-rewriting task, we LoRA fine-tuned Llama-2 models with a learning rate of 1e-5, batch size 64, gradient accumulation step 1, and 8-bit quantization. The model takes about 2000 steps to converge. For coding experiments, we trained the model with a learning rate of 1e-5, batch size 4, and gradient accumulation step 1, 8-bit quantization for 3 epochs with a maximum length of 768. The models are trained and inferenced on 1 Nvidia A40 GPU. We used greedy decoding for all experiments.

- B.2. Data Generation

Synthetic Experiment Except for the diversity of semantics experiment, the results we reported in the main paper are obtained from an input length of 50 and a pattern length of 20. To validate the generality of our findings, we conducted experiments on various input sizes {50, 100, 200} and, correspondingly, pattern lengths {20,40,50}.

In the diversity of semantics experiment, we used an input length of 500 and a pattern length of 60. We strictly restricted the sub-strings to look for and to replace them with both to be unseen during testing.

Real World Data We downloaded the datasets (OSS-Instruct, Alpaca, CoT, UltraInteract, OpenOrca) from the official Huggingface Datasets Repos.

Demonstration of dataset sizes for long-tail rule distribution experiments. We included the plot of percentage against rank index with different α’s.

#### C. More Details On Evaluation

For coding task, we evaluated the performance following the standard settings in EvalPlus [30].

In Section 5, we evaluated a on variety of tasks. We used the evaluation suite (prompt, score computation script) provided by Yuan et al. [50].

| | |
|---|---|
| | |

### Percentage

### Rank Index

Figure 7: The sorted percentage of each instruction following power-law distribution with different shape parameters. The y-axis is the percentage of the rules in the training mixture. The x-axis is the ranked index (by proportion of examples) of instructions.

