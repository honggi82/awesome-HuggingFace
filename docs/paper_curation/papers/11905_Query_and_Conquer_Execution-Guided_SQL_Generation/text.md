# arXiv:2503.24364v1[cs.CL]31Mar2025

## Query and Conquer: Execution-Guided SQL Generation

Łukasz Borchmann and Marek Wydmuch Snowflake AI Research lukasz.borchmann@snowflake.com

### Abstract

55%

|o<br><br>Best accuracy at lowest cost|3-mini 4o|o1|
|---|---|---|
|x<br><br>2<br><br>0<br><br>x<br><br>10<br><br>| | |
|Qwen 7B| | |

We propose a novel approach for generating complex outputs that significantly improves accuracy in text-to-SQL tasks. Our method leverages execution results to select the most semantically consistent query from multiple candidates, enabling smaller, cost-effective models to surpass computationally intensive reasoning methods such as o1, o3-mini, and DeepSeek R1 while reducing inference cost by as much as 30 times. It integrates effortlessly with existing models, offering a practical and scalable pathway to state-of-the-art SQL generation.

50%

Accuracy

45%

1 10 100 1000

Total inference cost (USD)

### 1 Introduction

Figure 1: Cost-accuracy analysis for Qwen 2.5 Coder 7B, with or without self-consistency (10-20 samples), compared alongside OpenAI models.

Large language models frequently produce correct outputs when multiple samples are considered (pass@k), yet their one-shot accuracy (pass@1) remains significantly lower. This gap motivates exploring methods that leverage multiple outputs to reliably identify the most promising one.

SELECT department_id FROM (

SELECT department_id , ROW_NUMBER()

OVER (PARTITION BY department_id ORDER BY department_id) AS rn

A prime example is the self-consistency method, which generates a diverse set of reasoning paths from the model and then employs majority voting on the final answers (e.g. numerical values for mathematical problems) to select the most likely correct output (Wang et al., 2023). While practical for short, well-defined answers, this strategy quickly breaks down when outputs have multiple correct yet structurally distinct representations, such as in code generation tasks.

FROM employees

) AS subquery WHERE rn = 1;

Each of these variations produces the same result but differs in structure, rendering majority voting strategies ineffective. We argue that overcoming this challenge demands methods that measure equivalence at the execution level rather than depending on structural comparison and propose such a strategy substantially narrowing the gap between pass@1 and pass@k accuracy.

Consider a simple SQL query retrieving all unique department IDs from an ‘employees’ table:

SELECT DISTINCT department_id FROM employees;

Specifically, we propose a novel self-consistency approach tailored to SQL generation, leveraging exact and approximate execution-based similarity metrics to assess semantic equivalence directly from query outputs (Figure 2). We further frame the problem of self-consistency within the Minimum Bayes Risk (MBR) decoding framework, providing a theoretical justification for our method

This query can be restructured in a variety of different yet equivalent forms, including:

SELECT department_id FROM employees GROUP BY department_id;

Alternatively, it can be expressed in an entirely different way, such as:

1

1

| |
|---|

SELECT…

###### SELECT…

0.0

0.5

SELECT…

(3) The query is judged the best if it is the most average, that is, has the highest accumulated similarity from the queryto-query comparisons.

| |
|---|

| |
|---|

SELECT…

0.5

(1)

We start by sampling several SQLs.

(2) Every SQL is executed, and each pair of resulting

data frames is compared against each other.

Figure 2: Execution-Guided SQL Generation.

and extending self-consistency to output spaces defined by execution behavior rather than superficial syntactic forms. Finally, we exploit the prefix executability property inherent in specific SQL dialects to incrementally apply execution-based selfconsistency during intermediate query generation stages, enabling more robust refinement of complex queries.

ture alone (Figure 2).

Significantly, the proposed method distinguishes itself from recent sequential, multi-step errorcorrection methods (Lee et al., 2024; Cen et al., 2025; Wang et al., 2024). These involve iterative refinement, inherently increasing complexity and latency. Conversely, our execution-guided similarity selection enables parallelizable generation processes and relies on efficient comparisons that primarily leverage CPU memory, typically underutilized in standard large language model deployment infrastructures (Oliaro et al., 2024).

These methodological advancements yield substantial empirical improvements. Notably, we demonstrate that by applying execution-based selfconsistency, smaller, less expensive models can match the performance of much larger models, highlighting a significant improvement in costefficiency. In particular, the 7B Qwen 2.5 Coder employing our method improves the accuracy by nearly 10%, reaching the level of O1 despite yielding a 30 times lower inference cost (Figure 1). These findings underscore our method’s efficiency, and scalability, positioning it as a strong candidate for real-world SQL generation tasks.

### 3 Execution-Guided SQL Generation

The justification for execution-based similarity stems from the concept of Minimum Bayes Risk decoding (Nadas, 1983, 1985; Kumar and Byrne, 2004; Eikema and Aziz, 2020).

#### 3.1 MBR Decoding and Code Utility

In MBR decoding, the objective is to find a hypothesis h⋆ that minimizes the expected loss (maximizes the expected utility) relative to the set of possible outputs H. If we assume the posterior distribution is derived from the empirical distribution obtained through sampling, it can be described as:

### 2 Related Works

Previous authors addressed SQL generation consistency through heuristics primarily focused on textual and structural similarities among queries (Hong et al., 2025; Li et al., 2024; Chen et al., 2021; Li et al., 2024). Such inherently struggle when queries differ structurally yet produce identical outputs, and thus, are semantically equivalent.

c(hˆ) · U(h,hˆ)

h⋆ = arg max

h∈H h ˆ∈H

where c(hˆ) denote the count of a particular hypothesis hˆ ∈ H in these samples, and U assesses the utility of choosing h when the true hypothesis is hˆ.

To address this limitation, we propose evaluating equivalence directly at the execution level. Precisely, we execute candidate queries (or approximate their execution) and compare resulting outputs, thus capturing semantic correctness based on actual behavior rather than superficial query struc-

In other words, we select the generation closest on average to all the likely generations, where closeness is measured under some function appropriate for the considered domain.

Concerning majority voting, such utility function is simply:

U(h,hˆ) =

1, if h = h,ˆ 0, otherwise.

and the objective simplifies to:

h⋆ = arg max

c(h)

h∈H

.

The zero-one utility is justified for short and unambiguous hypotheses, such as numerical answers for mathematical problems faced by LLM, because these are commonly evaluated with exact match accuracy. However, what does it mean for the two distinct SQL statements to be, to some extent, equivalent?

Evaluation practices in the text-to-SQL domain answer this question by resorting to executionbased similarities (Zhong et al., 2020). Thus, the similarity used in self-consistency should be based on the execution equivalence rather than the shallow form of the query or semantic interpretation lacking the information from the execution plan.

#### 3.2 Execution Similarity

We follow the abovementioned observation and the notion that comparing two codes’ behavior is the natural way to measure their similarity. Such similarity results from yielding the same output under the same circumstances and naturally captures semantic equivalence, overcoming structural differences in queries (Cheers et al., 2021; Simpson and Voorneveld, 2018).

Exact Execution Similarity. Upon execution, SQL queries we consider return values (cells) structured in horizontal rows and vertical columns identifiable by name.

For the purposes of experiments, we define execution similarity for such SELECT statements based on the resulting tables A and B as a form of recall of cells with respect to the larger column. Take an example of the following two tables considered column-by-column:

X

X Y

- 1
- 2
- 3

- 1 ♡
- 2 ♢

We retrieved two out of three values in column X (1 and 2), and missed two out of two values in

column Y (♡, ♢), yielding similarity of 2/5 = 0.4 (refer to Appendix B.1 for formal definition).

It is one of the many suitable functions that grow monotonically as we approach the desired outcome. Because it relies on the execution result, it fulfills our assumption of measuring behavior.

Approximate Execution Similarity. Though in most cases, computing execution similarity is cheap, we consider a variant without the actual execution. Here, logical execution plans for considered statements are compared, that is, the operations (e.g. table scans and joins) that the database engine would perform to execute the query.

As the execution plan, such as returned by the EXPLAIN query, can be represented as a table, we use the same metric for table-to-table comparison

- as in the proposed execution similarity. However, this time, it compares the execution plans rather than the actual results.

3.3 Decoding of Partially-Executable SQL

Parts of standard SQL code, such as subqueries or Common Table Expressions, are executable independently, without the context of the entire query. This property could further improve accuracy without increasing the inference cost, as one could refine the decoding trajectories in the middle of the generation process based on the self-consistency of the considered SQL part.

Due to the property of prefix executability, PipeSQL (Shute et al., 2024) appears to be even better aligned with this objective. In this dialect, each query prefix (up to the pipe sequence |>) is also a valid query. Consider the example of

FROM users |> WHERE views > 10 |> AGGREGATE COUNT(id);

The first pipe would produce the complete user’s table upon execution, the first two pipes—users filtered by the number of views, whereas the complete SQL returns the count of filtered users.

Consequently, one can sample n continuations

- at each step until the |> sequence and apply execution-based self-consistency on partial completion. Figure 3 presents an example of such an approach. In the middle of generation, we select a consensus continuation between two WHERE clauses and SELECT based on the similarity matrix:

SELECT name;

WHERE country != “France" |> SELECT name, surname;

WHERE country <> ‘France' |>

SELECT surname;

FROM singer |>

###### Sample

SELECT name;

Sample

- Figure 3: PipeSQL dialect has a property that each query prefix (up to the pipe sequence |>) is also a valid query, making it possible to apply execution-based self-consistency in the middle of the generation process. Instead of sampling n complete SQL sequences, we sample n p e generation process. Th most consistent pipe and continue the generation sampling the next pipe.

| | |1|0| |1|
|---|---|---|---|---|---|
|ipe|1s|nd|0st|op|the1|
|n|va0|ria0|nts|of|0|
| | | | | | |

| | |1|0| |1|
|---|---|---|---|---|---|
|en|,1w|e p|0ic|k th|1e|
| |0|0| | |0|
| | | | | | |

| | |1|0| |1|
|---|---|---|---|---|---|
| |1| |0| |1|
| |0|0| | |0|
| | | | | | |

pairs are not. To accommodate this property, we introduce the patience parameter, which keeps the generation unless it was selected for rejection at least n times in pipe-to-pipe comparisons.

.5 0 .5 .5 .5 1 0 .5 .5

### 4 Text-to-SQL Experiments

There is a tie because WHERE clauses are equivalent, so we pick the first and proceed. In the next step, there are three SELECTs considered:

We perform primary experiments on the BIRDSQL (Li et al., 2023) dataset. In all cases, the input is the same fixed prompt with the serialized database schema, and the model is requested to generate SQL preceded by a CoT (Appendix C.1).

| | |1|0| |1|
|---|---|---|---|---|---|
| |1| |0| |1|
| |0|0| | |0|
| | | | | | |

.5 0 .5 .5 .5 1 0 .5 .5

A wide range of families and parameter counts is considered, including general-purpose models, such as Llama 3 (Grattafiori et al., 2024), Gemma 3 (Gemma Team, 2025), Mistral Large (Mistral AI, 2024b), GPT-4o (OpenAI, 2024), and Gemini 2.0 (Gemini Team, 2024), as well as code-specialized Qwen 2.5 Coder (Hui et al., 2024), DeepSeek Coder (Guo et al., 2024), and Codestral (Mistral AI, 2024a).

The second one is picked—the most average since it produces a union of the remaining onesand the generation finishes.

Following this idea, we consider text-to-SQL tasks using a partially executable PipeSQL dialect in addition to self-consistency based on complete SQL generation. Such partial-execution-guided decoding can potentially reduce errors early in query generation, improving efficiency and accuracy.

Baselines. Compute-matched baselines are provided for each model considered in addition to the SQL obtained under greedy decoding. These include the majority vote with SQL normalization1 and beam search (in fact, beam search with n beams yields even higher cost than sampling n completions, but we assume that they are comparable for simplicity).

Patience. In practice, information carried by a single pipe can be too fine-grained for efficient execution-based consistency. Take an example of join statements that can begin with any of two tables leading to equivalent code when complete SQL sequences are considered:

Heavy Reasoners. Additionally, we provide results of computationally intensive reasoning approaches such as o1 and o3, DeepSeek R1, and Gemini 2.0 Flash Thinking, which require significantly more compute (OpenAI, 2025; DeepSeek AI, 2025; Gemini Team, 2024).

FROM emp |> JOIN dept ON emp.did = dept.did |> SELECT emp.name , dept.name;

FROM dept |> JOIN emp ON emp.did = dept.did |> SELECT emp.name , dept.name;

The first pipes from each query are dissimilar, even though the sequences of the first and second

1Using SQLGlot to reformat code: https://github. com/tobymao/sqlglot.git

Bound Baseline Scores Exec@10 Exec@20 Exec@30

Model

Pass@10 Greedy Maj@10 Beam@10 Approx. Exact Approx. Exact Approx. Exact

- Llama 3.2 3B 43.5 18.6 20.2 20.0 29.1 25.6 32.2 31.8 34.5 34.8 Qwen 2.5 Coder 3B 57.7 30.2 32.5 34.9 42.6 40.2 45.6 45.4 46.1 47.6 Qwen 2.5 Coder 7B 67.9 44.1 45.4 45.4 51.3 51.7 52.6 53.8 53.1 54.8 Llama 3.1 8B 62.1 32.9 34.3 34.9 43.1 43.6 44.6 47.5 45.0 48.8 Gemma 3 12B 64.9 49.7 52.2 49.2 53.2 53.0 54.7 53.9 54.8 54.6 Qwen 2.5 Coder 14B 71.5 54.7 54.9 52.9 56.8 57.2 57.4 58.3 57.8 58.3 Codestral 22B v0.1 63.5 45.6 48.6 46.2 50.6 51.3 51.0 52.3 51.6 52.8 Gemma 3 27B 66.3 53.1 55.5 54.6 56.0 55.6 56.6 56.3 56.6 56.7 Qwen 2.5 Coder 32B 71.1 55.0 55.2 54.9 56.3 57.1 56.9 57.6 57.8 57.6 DeepSeek Coder 33B 63.4 40.1 43.7 41.8 46.6 47.7 48.5 49.7 49.9 50.5

- Llama 3.3 70B 67.9 53.7 55.8 54.7 56.1 56.6 56.9 57.4 56.7 57.2 Mistral Large 2411 66.1 53.1 53.2 52.5 53.8 54.2 53.7 54.6 54.0 54.7 Llama 3.1 405 B 68.2 54.2 54.8 × 55.6 56.5 56.6 57.3 56.7 57.2 GPT-4o 2024-11-20 62.2 51.6 51.6 × 51.6 52.4 52.4 52.6 52.6 52.9 GPT-4o mini 2024-11-20 62.1 46.9 49.3 × 50.5 50.5 50.9 51.3 51.2 51.6 Gemini 2.0 Flash 001 70.9 60.6 61.8 × 61.7 61.9 62.2 62.1 62.0 62.1 Gemini 2.0 Flash-Lite 02-05 69.4 56.5 57.9 × 57.6 57.5 57.8 58.7 57.9 59.2

Gemini 2.0 Thinking 01-26 × 59.1 × × × × × × × × DeepSeek R1 × 52.5 × × × × × × × ×

- o1 2024-12-17 × 53.9 × × × × × × × ×

- o3-mini 2025-01-31 × 52.1 × × × × × × × ×

Table 1: BIRD-SQL Accuracy (Text-to-SQLite). The proposed MBR decoding with execution similarity (exec@n), compared to baselines: greedy decoding, majority voting with normalization (maj@10), beam search (beam@10), theoretical maximum (pass@10), and heavy reasoning LLMs. Samplings with temp = 0.7, validation subset.

Upper bound. Pass@10 score is provided for reference as a theoretical upper bound for all methods based on sampling complete SQL sequences. It is a score with an oracle judge always selecting the best available SQL.

#### 4.1 Overall Accuracy Improvements

Results in Table 1 show that execution-guided generation consistently outperforms greedy decoding and compute-matched baselines, holding across a diverse set of language models—from smaller general-purpose LLMs to code-specific and proprietary state-of-the-art systems.

Smaller open-source models (3B–7B parameters) can see substantial boosts, often improving accuracy by 10 points or more with around 30 samples. These improvements highlight that even relatively modest model scales can achieve competitive accuracy when combined with our self-consistency strategy. Larger open-source models, such as Qwen 2.5 Coder 34B, Mistral, or Llama 70B, typically gain 1.5–3.5 points, while huge ones (Llama 405B) still slightly benefit. Closed-source systems also show considerable improvements—GPT-4o mini gains around 5 points, whereas more powerful proprietary models improve by 1–2 points. While im-

provements for larger models are modest, these results serve primarily as confirmation that applying self-consistency does not negatively impact their already strong performance.

Notably, a 7B Qwen Coder with 30 samples surpasses most heavier reasoning-intensive models, with only Gemini 2.0 Thinking maintaining an edge. Since details on these proprietary systems remain limited, Figure 1 treats cost as a proxy for inference complexity. It compares Qwen 2.5 Coder 7B to two families of OpenAI models in terms of cost and accuracy, evaluated with and without self-consistency. The results demonstrate that selfconsistency significantly increases accuracy at a moderate computational cost, offering an effective balance between quality and resource expenditure.

Although exact variants typically provide superior accuracy, approximate methods remain highly attractive in latency-sensitive or resourceconstrained production environments.

Overall, these findings underscore the robust effectiveness of execution-guided generation. Even at more minor model scales, self-consistency offers notable gains in the quality-cost tradeoff, matching or exceeding the performance of more complex proprietary solutions.

Qwen 7B, diﬀerent temperatures

Model Greedy Exec + Part. + Pat.

60

| |
|---|
|0.4|
|1.0<br><br>0.7|
|Greedy|
| |
|1.3|

Qwen Coder 7B 27.1 41.6 42.8 44.3 Llama 8B 11.6 14.8 22.8 24.7 Gemma 12B 21.6 42.0 42.0 45.3 Qwen Coder 14B 38.9 51.2 49.6 53.0 Gemma 27B 31.2 47.3 46.8 49.1 Qwen Coder 32B 40.3 53.8 53.2 55.2 Codestral 33.6 46.8 47.4 53.0 Llama 70B 31.3 51.2 48.7 52.0 Mistral Large 44.3 50.4 50.8 53.0 LLama 405B 37.4 54.0 53.4 56.7

50

40

1 10 100

Temperature of 0.7, diﬀerent models

Table 2: BIRD-SQL Accuracy (Text-to-PipeSQL). Greedy decoding results compared to ten samples budged with standard execution-based self-consistency, partial self-consistency, or its variant with patience.

70

| |
|---|
|Gemini|
|Llama|
| |
|Codestral|
|Qwen 7B|

60

50

#### 4.3 Leveraging Partial Executability

Since we rely on the PipeSQL dialect for partial executability (see Section3.3), these experiments required transpiling ground-truth BIRD-SQL queries, converting the underlying databases, and establishing separate baseline scores (Appendix B.3).

1 10 100

- Figure 4: Self-consistency gains for various sample sizes, temperatures, and models (Gemini 2.0 Flash, Llama 3.3 70B, Codestral, Qwen 2.5 Coder 7B).

Moreover, unlike the widely used SQLite dialect in previous experiments, PipeSQL is relatively novel and not recognized by default, necessitating thorough in-prompt documentation with examples. Consequently, our inputs in this setup reached approximately 15k tokens, so certain opensource models with limited context windows could not be evaluated. Commercial API-based models were similarly excluded, as their interfaces typically do not allow mid-generation refinement.

#### 4.2 Number of Samples and Temperature

- Figure 4 examines how quickly improvements from execution-based self-consistency begin to plateau given different models and temperatures.

Notably, gains become visible with as few as three samples, and using 15 samples proves to be a strong balance between accuracy and computational cost. Although improvements tend to flatten around 50 samples, steady growth continues beyond that point—reflecting how each additional sample helps approximate the distribution of SQL queries generated by the model more accurately.

3

Despite few-shot prompting and including detailed documentation on the new dialect, models typically perform substantially worse with PipeSQL compared to standard SQL. Switching from SQLite yields accuracy drops of 5–20 points depending on model capabilities, underscoring limited generalizability to novel query dialects and problems with instruction-following.

Sampling temperature is key to balancing immediate accuracy with the potential gains from larger sampling budgets. While increased noise from higher temperatures undermines small-sample performance, it also fosters greater diversity in generated SQLs—ultimately boosting accuracy once the sampling budget is large enough to absorb occasional failures.

Table 2 summarizes the results in multiple settings: (1) Greedy decoding, (2) Standard selfconsistency without leveraging partial executability, (3) Self-consistency enhanced by partial, pipe-bypipe executability, and (4) Partial executability with a patience parameter (n = 3), accommodating temporary divergences in intermediate SQL steps.

In broad terms, weaker models benefit more from the proposed method. Interestingly, a high pass@1 score does not always translate into superior self-consistency accuracy. E.g., while Qwen Coder 7B outperforms Codestral in a high-sample regime, it underperforms until around 10 samples.

Standard self-consistency notably improves accuracy over greedy decoding (typically by around 15 points), because non-executable or failing SQL generations, more common with unfamiliar di-

Improves valid Improves invalid

Improves valid Improves invalid

Logical Form

Logical Form

Makes incorrect Keeps correct

Makes incorrect Keeps correct

180

180

Keeps incorrect

Keeps incorrect

120

120

9%

9%

- 3%

- 4%

- 3%

- 4%

Table Joins

Schema Linking

Table Joins

Schema Linking

60

60

0

0

45%

45%

DeepSeek Coder 33B

DeepSeek Coder 33B

DeepSeek Coder

DeepSeek Coder

39%

39%

Aggregation Projection

Aggregation Projection

Logical Form

Logical Form

6%

6%

180

180

2%

2%

120

120

3%

3%

Table Joins

Schema Linking

Table Joins

Schema Linking

60

60

0

44%

0

44%

GPT-4o mini

GPT-4o mini

45%

45%

GPT-4o mini

GPT-4o mini

Aggregation Projection

Aggregation Projection

Figure 6: Top problems explaining why BIRD-SQL generations of DeepSeek Coder and GPT-4o mini were incorrect. Greedy decoding compared to selfconsistency outputs.

- Figure 5: Effect of replacing outputs produced under greedy decoding by self-consistency outputs. Valid and invalid refer to executability, whereas correct and incorrect—conforming to the gold standard.

alects, naturally receive low similarity scores and are thus filtered out.

Partial pipe-by-pipe executability alone provides mixed results. Depending on the model, accuracy may slightly increase or decrease (1–2 points) compared to standard self-consistency. Finally, introducing the patience parameter generally yields further accuracy gains (1–10 points over standard self-consistency), demonstrating the benefit of tolerating intermediate divergences.

We hypothesize considerable potential in the partial executability approach but fully realizing it requires models capable of generating high-quality PipeSQL. Currently, limitations include insufficient training data on such dialects and the lack of robust transpilers for existing SQL datasets.

### 5 Qualitative Analysis

To gain deeper insights into how execution-based self-consistency enhances text-to-SQL generation, we analyze the specific error types it effectively addresses and present key statistical comparisons between greedy-decoded outputs and those refined through self-consistency.

The starting point for this part is the previously established taxonomy of SQL generation errors (Shen et al., 2025) and linguistic ambiguities in text-to-SQL tasks (Huang et al., 2023). Specifically, we consider common categories of mistakes, including dialect mismatches, schema linking failures, data type mismatches, incorrect aggregation, logical form inaccuracies, improper table joins, and projection errors (see Appendix B.4 for details).

Figure 5 examines the per-example impact of

adopting self-consistency compared to greedy decoding. While self-consistency occasionally yields inferior SQL queries, its overall effect is beneficial. Most improvements occur because the method successfully replaces executable but incorrect queries with queries that correctly address the user’s intent.

A closer inspection of the error categories reveals that most text-to-SQL failures stem from flawed logical forms, schema linking mistakes, and incorrect projections. These arise when queries are not logically coherent when the model confuses or hallucinates database elements (e.g., referencing nonexistent columns), and when it selects columns that do not match the intent of the user query.

Execution-based self-consistency mitigates these issues by filtering out candidates that fail under execution or yield outlier data frames, thus promoting solutions that produce similar outputs across multiple independent samples. For example, applying self-consistency in DeepSeek Coder reduces schema linking errors by 40%, projection, and table join mistakes by 20%, and logical form errors by 11% (see Figure 6). Models with fewer initial errors tend to benefit less; for instance, GPT-4o mini still sees a notable 30% drop in schema linking mistakes. However, the low incidence of such errors in its greedy baseline limits the overall improvement.

Hence, this error-reduction pattern also aligns with the findings described in the earlier sections, where models with lower baseline performance benefited disproportionately more from executionbased self-consistency.

### 6 Limitations

While our self-consistency provides substantial accuracy gains, it requires generating multiple candidate queries and assessing their behavior. In some cases, actual query execution can be expensive or time-consuming. Still, the proposed approximate approach significantly reduces this concern since EXPLAIN queries typically incur negligible costs.

Moreover, generating multiple solutions introduces additional compute time compared to a single pass, yet the approach remains easily parallelizable. By contrast, step-by-step agentic solutions can inflate latency significantly because their sequential reasoning is far more challenging to distribute across multiple processes.

Next, even though our preliminary evaluation with the partially executable SQL dialect suggests promising results, a direct comparison to standard

SQLite tasks remains an area for further studies.

Finally, models with fewer initial errors or a high baseline accuracy may exhibit diminishing returns, limiting the offered improvement.

### 7 Summary

A family of self-consistency techniques relying on sampling multiple SQL queries and comparing their execution results has been introduced. Additionally, a partial-execution variant allowing stepby-step refinement of intermediate query fragments was proposed, enabling further precision improvements when partial queries are reliably generated.

The proposed methods robustly identify semantically equivalent queries even when there are structural variations, allowing smaller and inexpensive models to achieve accuracy typically reserved for larger and costlier LLMs—all while avoiding substantial processing time overhead typically associated with iterative refinement strategies.

The presented analysis reveals that offered improvements can be attributed to effectively addressing common SQL generation errors, yielding 20– 40% reductions in schema linking errors, projection and table join mistakes, and logical form errors.

While the presented experiments focused on SQL generation, the underlying principle of leveraging execution-based similarity naturally extends to other programming languages and codegeneration tasks. Supplementary results on text-tocode benchmarks provided in Appendix A.1 suggest the broad applicability of execution-guided self-consistency and highlight its potential in program synthesis across diverse domains.

We believe that further exploration of executionguided generation will open promising avenues toward robust, efficient, and universally applicable code-generation models.

### Acknowledgments

We sincerely thank the colleagues whose insightful discussions and feedback shaped this paper. Special thanks to Anupam Datta, Michał Zaja˛c, Filip Grali´nski, Zhewei Yao, Andrzej Szwabe, Michał Pietruszka, Jakub Swia˛tkowski,´ Wojciech Ja´skowski, and Tomasz Dwojak for generously sharing their perspectives, inspiring research questions, and impacting the evolution of our ideas.

### References

Jipeng Cen, Jiaxin Liu, Zhixu Li, and Jingjing Wang. 2025. SQLFixAgent: Towards SemanticAccurate Text-to-SQL Parsing via ConsistencyEnhanced Multi-Agent Collaboration. Preprint, arXiv:2406.13408.

Hayden Cheers, Yuqing Lin, and Shamus P. Smith. 2021. Academic Source Code Plagiarism Detection by Measuring Program Behavioral Similarity. IEEE Access, 9:50391–50412.

Mark Chen et al. 2021. Evaluating Large Language Models Trained on Code. Preprint, arXiv:2107.03374.

DeepSeek AI. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. Preprint, arXiv:2501.12948.

Bryan Eikema and Wilker Aziz. 2020. Is MAP decoding all you need? the inadequacy of the mode in neural machine translation. In Proceedings of the 28th International Conference on Computational Linguistics, pages 4506–4520, Barcelona, Spain (Online). International Committee on Computational Linguistics.

Gemini Team. 2024. Gemini: A Family of Highly Capable Multimodal Models. Preprint, arXiv:2312.11805.

Gemma Team. 2025. Gemma 3 Technical Report. Aaron Grattafiori et al. 2024. The Llama 3 Herd of

Models. Preprint, arXiv:2407.21783.

Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y. Wu, Y. K. Li, Fuli Luo, Yingfei Xiong, and Wenfeng Liang. 2024. DeepSeek-Coder: When the Large Language Model Meets Programming – The Rise of Code Intelligence. Preprint, arXiv:2401.14196.

Zijin Hong, Zheng Yuan, Qinggang Zhang, Hao Chen, Junnan Dong, Feiran Huang, and Xiao Huang. 2025. Next-Generation Database Interfaces: A Survey of LLM-based Text-to-SQL. Preprint, arXiv:2406.08426.

Zezhou Huang, Pavan Kalyan Damalapati, and Eugene Wu. 2023. Data Ambiguity Strikes Back: How Documentation Improves GPT’s Text-to-SQL. Preprint, arXiv:2310.18742.

Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, Kai Dang, Yang Fan, Yichang Zhang, An Yang, Rui Men, Fei Huang, Bo Zheng, Yibo Miao, Shanghaoran Quan, Yunlong Feng, Xingzhang Ren, Xuancheng Ren, Jingren Zhou, and Junyang Lin. 2024. Qwen2.5-Coder Technical Report. Preprint, arXiv:2409.12186.

Shankar Kumar and William Byrne. 2004. Minimum Bayes-Risk Decoding for Statistical Machine Translation. In Proceedings of the Human Language Technology Conference of the North American Chapter of the Association for Computational Linguistics: HLT-NAACL 2004, pages 169–176, Boston, Massachusetts, USA. Association for Computational Linguistics.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles.

Dongjun Lee, Choongwon Park, Jaehyuk Kim, and Heesoo Park. 2024. MCS-SQL: Leveraging Multiple Prompts and Multiple-Choice Selection For Text-toSQL Generation. Preprint, arXiv:2405.07467.

Boyan Li, Yuyu Luo, Chengliang Chai, Guoliang Li, and Nan Tang. 2024. The Dawn of Natural Language to SQL: Are We Fully Ready? Proceedings of the VLDB Endowment, 17(11):3318–3331.

Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang, Bowen Qin, Rongyu Cao, Ruiying Geng, Nan Huo, Xuanhe Zhou, Chenhao Ma, Guoliang Li, Kevin C. C. Chang, Fei Huang, Reynold Cheng, and Yongbin Li. 2023. Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs. Preprint, arXiv:2305.03111.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. 2023. Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation. In Thirtyseventh Conference on Neural Information Processing Systems.

- Mistral AI. 2024a. Codestral: Hello, World!
- Mistral AI. 2024b. Large Enough.

A. Nadas. 1983. A decision theorectic formulation of a training problem in speech recognition and a comparison of training by unconditional versus conditional maximum likelihood. IEEE Transactions on Acoustics, Speech, and Signal Processing, 31(4):814–817.

A. Nadas. 1985. Optimal solution of a training problem in speech recognition. IEEE Transactions on Acoustics, Speech, and Signal Processing, 33(1):326–329.

Gabriele Oliaro, Zhihao Jia, Daniel Campos, and Aurick Qiao. 2024. SuffixDecoding: A Model-Free Approach to Speeding Up Large Language Model Inference. Preprint, arXiv:2411.04975.

- OpenAI. 2024. GPT-4 Technical Report. Preprint, arXiv:2303.08774.
- OpenAI. 2025. Competitive Programming with Large Reasoning Models. Preprint, arXiv:2502.06807.

Jiawei Shen, Chengcheng Wan, Ruoyi Qiao, Jiazhen Zou, Hang Xu, Yuchen Shao, Yueling Zhang, Weikai Miao, and Geguang Pu. 2025. A Study of In-Context-Learning-Based Text-to-SQL Errors. Preprint, arXiv:2501.09310.

Jeff Shute, Shannon Bales, Matthew Brown, JeanDaniel Browne, Brandon Dolphin, Romit Kudtarkar, Andrey Litvinov, Jingchi Ma, John Morcos, Michael Shen, David Wilhite, Xi Wu, and Lulan Yu. 2024. SQL Has Problems. We Can Fix Them: Pipe Syntax In SQL. pages 4051–4063.

Alex Simpson and Niels Voorneveld. 2018. Behavioural Equivalence via Modalities for Algebraic Effects. In Programming Languages and Systems, pages 300– 326, Cham. Springer International Publishing.

Transaction Processing Performance Council. 2014. TPC Benchmark™ H (Decision Support) Standard Specification Revision 2.17.1.

Bing Wang, Changyu Ren, Jian Yang, Xinnian Liang, Jiaqi Bai, Linzheng Chai, Zhao Yan, Qian-Wen Zhang, Di Yin, Xing Sun, and Zhoujun Li. 2024. MACSQL: A Multi-Agent Collaborative Framework for Text-to-SQL. Preprint, arXiv:2312.11242.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-Consistency Improves Chain of Thought Reasoning in Language Models. Preprint, arXiv:2203.11171.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-Art Natural Language Processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Ruiqi Zhong, Tao Yu, and Dan Klein. 2020. Semantic Evaluation for Text-to-SQL with Distilled Test Suites. Preprint, arXiv:2010.02840.

### A Supplementary Experiments

#### A.1 Beyond SQL Generation

The execution-based self-consistency we proposed has applications beyond SQL generation, e.g., for ordinary Python code.

We consider two functions similar if they produce the same outputs for the same inputs. The more inputs we test and outputs we compare, the more we know about their similarity. For example, we would expect a similarity of 0.5 with Python functions:

Model Greedy Exec@10

Mistral Large 2407 84.8 85.2 Codestral 22B 73.8 78.6 Llama 3.1 8B 61.6 66.1 Llama 3.1 405B 79.9 83.8

- (a) HumanEval+

Model Greedy Exec@10

Mistral Large 2407 65.9 75.3 Codestral 22B 62.4 72.4 Llama 3.1 8B 55.3 68.0 Llama 3.1 405B 70.4 77.5

- (b) MBPP+

Table 3: Python generation accuracy. Greedy decoding compared to execution-based self-consistency.

- def A(x: int) -> int: return x
- def B(x: int) -> int: return x if x % 2 == 0 else 0

as they yield the same output for half of the inputs. Specifically, given functions A,B : X  → Y we sample arguments {x1,...,xn} and measure the expected fraction of agreement between A(x) and B(x) over the domain X. Similarity is defined as:

1 n

S(A,B) =

n

1[A(xi) = B(xi)],

i=1

where 1[A(xi) = B(xi)] is the indicator function that checks if A(xi) equals B(xi).

Experiments. In practice, using random input values is sample-inefficient, and having a source of somewhat reasonable inputs concerning the intended behavior is better.

For simplicity, in this section, we assume such a source is given and rely on real test cases from HumanEval+ and MBPP+ benchmarks (Liu et al., 2023). Note that we do not leverage function definitions or expected outputs but merely the tested function arguments.

We observe consistent accuracy gains under execution-based self-consistency in both HumanEval+ and MBPP+ (Table 3). On HumanEval+, the improvements tend to be modest—e.g., Mistral Large rises from 84.8% to 85.2%, while Llama 3.1 8B increases by nearly five points (61.6% to 66.1%). In contrast, MBPP+ shows more pronounced gains: Mistral Large jumps by almost ten points (65.9% to 75.3%), and Llama 3.1 8B improves by over twelve (55.3% to 68.0%).

Bound Sample Budget Pass@10 @10 @20 @30

Model

DeepSeek Coder 33B 63.4 47.7 49.7 50.5 Qwen Coder 7B 67.9 51.7 53.8 54.8 Codestral 22B v0.1 63.5 51.3 52.3 52.8

+ DeepSeek Coder 33B 66.5 51.4 53.3 54.2 + Qwen Coder 7B 70.1 52.7 54.9 55.9

Llama 70B 67.9 56.6 57.4 57.2 Gemini Flash 001 70.9 61.9 62.1 62.1 Qwen Coder 32B 71.1 57.1 57.6 57.6

+ Llama 70B 72.2 57.4 58.1 58.4 + Gemini Flash 74.6 61.3 62.1 62.6

Table 4: Impact of cross-model consistency on BIRDSQL Accuracy (Text-to-SQLite).

These results indicate that execution-based comparisons become increasingly valuable for more challenging code generation benchmarks, where model outputs show more significant variability, and there is more room for error correction by filtering out inconsistent or failing candidates.

#### A.2 Cross-Model SQL Consistency

Another way to broaden the range of generated solutions is to sample from multiple LLMs rather than relying on a single one.

Results presented in Table 4 indicate that combining samples from multiple LLMs, even without weighting them by expected accuracy, yields modest but consistent accuracy improvementstypically on the order of 0.5 to 1 point. In the larger model group, for instance, pairing a stronger model (Qwen Coder 32B) with a weaker one (Llama 70B) provides a slight boost over using only the stronger model for the same total number of samples. Adding the more capable Gemini model to this mix further improves results.

Similar trends emerge among smaller and midsized models (e.g. Qwen 2.5 Coder 7B, Deepseek Coder, and Codestral), where combining their outputs slightly enhances performance without increasing the total sample count in self-consistency.

Despite these incremental gains, the approach is appealing for its simplicity and low overhead, underscoring the potential of straightforward crossmodel ensembling.

### B Details of SQL Experiments

All open-source language models were evaluated using vLLM (Kwon et al., 2023), except for the

beam-search baselines, which relied on the Transformers library (Wolf et al., 2020). In both setups, inference runs were performed on Nvidia DGX nodes with 8×H100 GPUs. Nearly all inferences relied on bf16 precision, except Llama 405B, which was evaluated in fp8 due to hardware constraints.

OpenAI models were accessed through their public API, Gemini models through the proprietary Google interface, and DeepSeek R1 via Snowflake Cortex. When reasoning-intensive models were examined, up to 32k tokens were allowed for their chain-of-thought processes.

- B.1 Data Frame Similarity Specifically, we define similarity as:

S(A,B) =

R max(|A|,|B|)

R =

c∈C v∈Vc

min fAc(v),fBc(v) ,

where |A| and |B| denote the total number of cells in tables, C is the set of all column names, Vc is the set of unique values in column c, whereas fAc(v) and fBc(v) denote the frequency of a value v in column c of A and B, respectively.

- B.2 Inference Cost

All cost estimates reflect actual billing under nonbatch inference. Specifically, OpenAI’s usagebased pricing was applied to the OpenAI models, while Together.ai’s pricing was used to obtain Qwen 2.5 Coder costs. Because the Qwen 2.5 Coder 7B was unavailable on Together.ai, its cost was approximated based on the non-coder Qwen 2.5 7B Instruct Turbo pricing tier.

- B.3 BIRD for PipeSQL

For simplicity, gold standard queries were converted to the regular BigQuery format rather than an explicit pipe-based syntax. Since the same execution engine powers both dialects and produces identical results for semantically equivalent queries, the choice does not affect correctness.

Each original query was first transpiled from SQLite to BigQuery using the SQLGlot library. Both forms were executed—one in SQLite, the other in BigQuery—and their resulting data frames were compared for consistency. If the SQLGlottranspiled query failed or did not match the original query’s result, the query was processed by an LLM

(o3-mini) for manual repair. The corrected query was again tested for matching data frames.

For fewer than 100 instances, neither automatic nor LLM-based transpilation yielded matching results. These queries were removed from the final PipeSQL dataset. All other queries were successfully converted, ensuring that the PipeSQL variant of BIRD-SQL closely mirrors the original in both content and accuracy.

#### B.4 SQL Error Taxonomy

For the purposes of qualitative analysis, we simplify the previously established taxonomy of SQL generation errors (Shen et al., 2025) and linguistic ambiguities in text-to-SQL tasks (Huang et al., 2023) distinguishing the following categories.

Dialect. Errors arising from dialect differences between SQL variants. A generated query might be semantically and syntactically correct in general but fail when executed due to differences specific to the target database.

Schema Linking. Errors due to incorrect or failed mappings between natural language references and corresponding database schema elements. This includes hallucinating non-existent tables or columns that appear relevant.

Data Type. Errors occurring when queries fail or produce incorrect results due to mismatches or unexpected data types within the database.

Aggregation. Errors involving incorrect use or omission of aggregation functions (e.g. COUNT, SUM), leading to inaccurate summary calculations or improperly grouped results.

Logical Form and Condition. Errors resulting from incorrect logical structures or incomplete query specifications. This includes missing or incorrect conditions, inappropriate filters, erroneous ordering logic, or improperly scoped constraints.

Table Joins. Errors involving incorrect table relationships, such as joining irrelevant tables, omitting necessary joins, or misidentifying join conditions.

Projection. The model may choose the wrong columns or expressions to return. Sometimes, it might select a computed value when the question asks for an entity or vice versa.

### C Prompts

#### C.1 SQLite Generation

For BIRD-SQL, we provide a straightforward instruction, serialised database, and questions concatenated with evidence.

You are an AI assistant helping a data analyst

→ write SQL queries to answer questions.

→ Below I will provide a DB schema with → example values and a question that can → be answered by querying the provided DB

→ . You will then write out your thought → process in detail followed by a single → SQL query enclosed in ‘‘‘sql ...‘‘‘

→ that answers the question.

SQLite database schema: Table: alignment : id : integer, primary key, example values: ( 1

→ , 2 , 3 )

alignment : text, example values: ( ’Good’ , ’

→ Bad’ , ’Neutral’ )

Table: attribute : id : integer, primary key, example values: ( 1

→ , 2 , 3 )

attribute_name : text, example values: ( ’

→ Intelligence’ , ’Strength’ , ’Speed’ )

Table: colour : id : integer, primary key, example values: ( 1

→ , 2 , 3 )

colour : text, example values: ( ’No Colour’ ,

→ ’Amber’ , ’Auburn’ )

Table: gender : id : integer, primary key, example values: ( 1

→ , 2 , 3 )

gender : text, example values: ( ’Male’ , ’

→ Female’ , ’N/A’ )

Table: publisher : id : integer, primary key, example values: ( 1

→ , 2 , 3 )

publisher_name : text, example values: ( ’’ , ’

→ ABC Studios’ , ’Dark Horse Comics’ )

Table: race : id : integer, primary key, example values: ( 1

→ , 2 , 3 )

race : text, example values: ( ’-’ , ’Alien’ ,

→ ’Alpha’ )

Table: superhero : id : integer, primary key, example values: ( 1

→ , 2 , 3 )

superhero_name : text, example values: ( ’3-D → Man’ , ’A-Bomb’ , ’Abe Sapien’ ) full_name : text, example values: ( ’Charles

→ Chandler’ , ’Richard Milhouse Jones’ ,

→ ’Abraham Sapien’ )

gender_id : integer, foreign key, references

→ gender, example values: ( 1 , 2 , 3 ) eye_colour_id : integer, foreign key,

→ references colour, example values: ( 9

→ , 33 , 7 )

hair_colour_id : integer, foreign key,

→ references colour, example values: ( 13

→ , 1 , 4 )

skin_colour_id : integer, foreign key,

→ references colour, example values: ( 1

→ , 7 , 23 )

race_id : integer, foreign key, references

→ race, example values: ( 1 , 24 , 33 ) publisher_id : integer, foreign key,

→ references publisher, example values: (

→ 13 , 3 , 4 )

alignment_id : integer, foreign key,

→ references alignment, example values: (

→ 1 , 2 )

height_cm : integer, example values: ( 188 ,

→ 203 , 191 )

weight_kg : integer, example values: ( 90 ,

→ 441 , 65 )

Table: hero_attribute : hero_id : integer, foreign key, references

→ superhero, example values: ( 1 , 2 , 3

###### → )

attribute_id : integer, foreign key,

→ references attribute, example values: (

→ 1 , 2 , 3 )

attribute_value : integer, example values: (

→ 80 , 75 , 95 )

Table: superpower : id : integer, primary key, example values: ( 1

→ , 2 , 3 )

power_name : text, example values: ( ’Agility’

→ , ’Accelerated Healing’ , ’Lantern

→ Power Ring’ )

Table: hero_power : hero_id : integer, foreign key, references

→ superhero, example values: ( 1 , 2 , 3

###### → )

power_id : integer, foreign key, references

→ superpower, example values: ( 1 , 18 ,

→ 26 )

The question: Who is the publisher of Sauron?

→ (the publisher refers to publisher_name

→ ; Sauron refers to superhero_name = ’

→ Sauron’)"

#### C.2 PipeSQL Generation

For PipeSQL generation, we provide quite elaborate prompt. It consists of simplified dialect documentation2 with complex examples of transpiled TPC-H queries3 (Transaction Processing Performance Council, 2014).

You will be asked to generate SQL using

→ BigQuery’s pipe query syntax.

# Pipe Query Syntax Pipe syntax has the following key

→ characteristics

- 2https://github.com/google/zetasql/blob/

master/docs/pipe-syntax.md

- 3https://github.com/google/zetasql/tree/

master/zetasql/examples/tpch/pipe_queries

- - Each pipe operator in pipe syntax consists

→ of the pipe symbol, |>, an operator

→ name, and any arguments.

- - Pipe syntax works anywhere standard syntax

→ is supported: in queries, views, table-

→ valued functions (TVFs), and other

→ contexts.

- - Pipe syntax can be mixed with standard

→ syntax in the same query. E.g.,

→ subqueries can use different syntax

→ from the parent query.

- - A query can start with a FROM clause, and

→ pipe operators can optionally be added

→ after the FROM clause.

Pipe operators have the following semantic

→ behavior

- - Each pipe operator performs a self-contained

→ operation.

- - A pipe operator consumes the input table

→ passed to it through the pipe symbol,

→ |>, and produces a new table as output.

- - A pipe operator can reference only columns

→ from its immediate input table. Columns

→ from earlier in the same query aren’t → visible. Inside subqueries, correlated → references to outer columns are still

→ allowed. Operator list

- - SELECT: Produces a new table with the listed

→ columns.

- - EXTEND: Propagates the existing table and

→ adds computed columns.

- - SET: Replaces the values of columns in the

→ current table.

- - DROP: Removes listed columns from the

→ current table.

- - RENAME: Renames specified columns.

- - AS: Introduces a table alias for the input

→ table.

- - WHERE: Filters the results of the input

→ table.

- - LIMIT: Limits the number of rows to return

→ in a query, with an optional OFFSET

→ clause to skip over rows.

- - AGGREGATE: Performs aggregation on data

→ across groups of rows or the full input

→ table.

- - DISTINCT: Returns distinct rows from the

→ input table, while preserving table

→ aliases.

- - ORDER BY: Sorts results by a list of

→ expressions.

- - UNION: Combines the results of the input

→ queries to the left and right of the

→ pipe operator by pairing columns from

→ the results of each query and

→ vertically concatenating them.

- - INTERSECT: Returns rows that are found in

→ the results of both the input query to

→ the left of the pipe operator and all

→ input queries to the right of the pipe

→ operator.

- - EXCEPT: Returns rows from the input query to

→ the left of the pipe operator that

→ aren’t present in any input queries to

→ the right of the pipe operator.

- - JOIN: Joins rows from the input table with

→ rows from a second table provided as an

→ argument.

- - CALL: Calls a table-valued function (TVF),

→ passing the pipe input table as a table

→ argument.

- - WINDOW: Adds columns with the result of

→ computing the function over some window

→ of existing rows

- - TABLESAMPLE: Selects a random sample of rows

→ from the input table.

- - PIVOT: Rotates rows into columns.

- - UNPIVOT: Rotates columns into rows.

- - ASSERT: Evaluates that an expression is true

→ for all input rows, raising an error

→ if not. ## Examples

- ### 1. Pricing Summary Report Query How can I generate a pricing summary report

→ that shows, for each combination of

→ return flag and line status, the total

→ quantity shipped, total base price,

→ total discounted price, total charge (

→ discounted price plus tax), average

→ quantity, average extended price,

→ average discount, and the count of

→ orders? The report should only include

→ line items shipped on or before the

→ date obtained by subtracting 74 days

→ from December 1, 1998, and it should be → ordered by return flag and line status → in ascending order.

‘‘‘sql SELECT

l_returnflag, l_linestatus, sum(l_quantity) AS sum_qty, sum(l_extendedprice) AS sum_base_price, sum(l_extendedprice * (1 - l_discount)) AS

→ sum_disc_price,

sum(l_extendedprice * (1 - l_discount) * (1 +

→ l_tax)) AS sum_charge, avg(l_quantity) AS avg_qty, avg(l_extendedprice) AS avg_price, avg(l_discount) AS avg_disc, COUNT(*) AS count_order

FROM

lineitem WHERE

l_shipdate <= date_sub(date ’1998-12-01’,

→ INTERVAL 74 day)

GROUP BY l_returnflag, l_linestatus

ORDER BY

l_returnflag, l_linestatus;

‘‘‘

- ### 2. Minimum Cost Supplier Query Which supplier in the Middle East should I

→ select to order parts of size 19 that

→ are of a type containing ’COPPER’,

→ based on the lowest available supply

→ cost? If multiple suppliers offer the → part at the same minimum cost, I want

→ to consider only the top 100 suppliers → with the highest account balances. For → each supplier, please provide their

→ account balance, name, nation, the part

→ number, manufacturer, address, phone

→ number, and any additional comments,

→ and sort the results by account balance

→ (highest first), then by nation,

→ supplier name, and part number.

‘‘‘sql FROM

part, supplier, partsupp, nation, region

|> WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey AND p_size = 19 AND p_type LIKE ’%COPPER’ AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey AND r_name = ’MIDDLE EAST’ AND ps_supplycost = (

FROM partsupp, supplier, nation, region

|> WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey AND r_name = ’MIDDLE EAST’

|> AGGREGATE min(ps_supplycost))

|> SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment

|> ORDER BY s_acctbal DESC, n_name, s_name, p_partkey

|> LIMIT 100; ‘‘‘

### 3. Order Priority Checking Query Can you help me determine how many orders,

→ placed in the quarter starting June 1,

→ 1997, had at least one lineitem

→ delivered after its committed date? I

→ need the results grouped by order

→ priority, with the count of such orders

→ sorted in ascending order by order

→ priority.

‘‘‘sql FROM

orders

|> WHERE o_orderdate >= date ’1997-06-01’ AND o_orderdate < date_add(date

→ ’1997-06-01’, INTERVAL 3 month) AND EXISTS(

FROM lineitem |> WHERE

l_orderkey = o_orderkey AND l_commitdate < l_receiptdate)

|> AGGREGATE COUNT(*) AS order_count

GROUP AND ORDER BY o_orderpriority; ‘‘‘

- ### 4. Potential Part Promotion Query Which suppliers in Peru supply parts whose

→ names begin with ’tan’ and have an

→ excess inventory of these parts-where → excess is defined as having available → quantity greater than 50% of the total

→ quantity shipped in 1996? Please return

→ the supplier’s name and address,

→ sorted in alphabetical order by name.

‘‘‘sql FROM

supplier, nation

|> WHERE s_suppkey IN (

FROM partsupp, part

|> WHERE p_name LIKE ’tan%’ |> WHERE

ps_partkey = p_partkey AND ps_availqty > (

FROM lineitem |> WHERE

l_partkey = ps_partkey AND l_suppkey = ps_suppkey AND l_shipdate >= date

→ ’1996-01-01’

AND l_shipdate < date_add( → date ’1996-01-01’,

→ INTERVAL 1 year)

|> AGGREGATE 0.5 * sum(l_quantity)

→ )

|> SELECT ps_suppkey) AND s_nationkey = n_nationkey AND n_name = ’PERU’

|> SELECT s_name, s_address |> ORDER BY s_name; ‘‘‘

- ### 5. Customer Distribution Query Can you generate a report that shows the

→ distribution of customers based on the

→ number of orders they have placed?

→ Include customers with zero orders, and

→ make sure to exclude any orders where → the comment contains the text ’unusual → packages’. The output should list, for

→ each order count, how many customers → have that many orders, sorted by the → number of customers (in descending

→ order) and then by the order count (

→ also in descending order).

‘‘‘sql FROM customer |> LEFT OUTER JOIN orders ON c_custkey =

→ o_custkey AND o_comment NOT LIKE ’%

→ unusual%packages%’

|> AGGREGATE COUNT(o_orderkey) c_count GROUP BY c_custkey

|> AGGREGATE COUNT(*) AS custdist GROUP BY c_count

|> ORDER BY custdist DESC, c_count DESC;

‘‘‘ ### 6. Discounted Revenue Query Can you calculate the gross discounted revenue

→ for orders where the parts meet any of

→ the following criteria?

- - Parts of brand ’Brand#53’ contained in ’SM

→ CASE’, ’SM BOX’, ’SM PACK’, or ’SM PKG

→ ’, with a quantity between 5 and 15 and

→ a size between 1 and 5.

- - Parts of brand ’Brand#41’ contained in ’MED

→ BAG’, ’MED BOX’, ’MED PKG’, or ’MED

→ PACK’, with a quantity between 15 and

→ 25 and a size between 1 and 10.

- - Parts of brand ’Brand#21’ contained in ’LG

→ CASE’, ’LG BOX’, ’LG PACK’, or ’LG PKG

→ ’, with a quantity between 29 and 39

→ and a size between 1 and 15.

Additionally, only consider orders that were

→ shipped by air (i.e., with a shipping

→ mode of ’AIR’ or ’AIR REG’) and were

→ delivered in person. The revenue should

→ be computed as the sum of

→ l_extendedprice * (1 - l_discount) for

→ all orders that qualify.

‘‘‘sql FROM

lineitem, part |> WHERE

# Added this because optimizer is needed → to pull this out of the OR. p_partkey = l_partkey AND ( (

p_partkey = l_partkey AND p_brand = ’Brand#53’ and p_container in (’SM CASE’, ’SM BOX

→ ’, ’SM PACK’, ’SM PKG’) AND l_quantity >= 5 AND l_quantity <= 5 + 10 AND p_size BETWEEN 1 AND 5

and l_shipmode in (’AIR’, ’AIR REG’) AND l_shipinstruct = ’DELIVER IN

→ PERSON’)

OR ( p_partkey = l_partkey AND p_brand = ’Brand#41’ and p_container in (’MED BAG’, ’MED

→ BOX’, ’MED PKG’, ’MED PACK’) AND l_quantity >= 15 AND l_quantity <= 15 + 10 AND p_size BETWEEN 1 AND 10 and l_shipmode in (’AIR’, ’AIR REG’) AND l_shipinstruct = ’DELIVER IN

→ PERSON’)

OR ( p_partkey = l_partkey AND p_brand = ’Brand#21’ and p_container in (’LG CASE’, ’LG BOX

→ ’, ’LG PACK’, ’LG PKG’) AND l_quantity >= 29 AND l_quantity <= 29 + 10 AND p_size BETWEEN 1 AND 15

and l_shipmode in (’AIR’, ’AIR REG’) AND l_shipinstruct = ’DELIVER IN

→ PERSON’)) |> AGGREGATE

sum(l_extendedprice * (1 - l_discount)) AS

→ revenue;

‘‘‘ ### 7. Global Sales Opportunity Query Can you identify the potential global sales

→ opportunities by finding customers from

→ specific regions-where the country

→ code is defined as the first two digits

→ of their phone number (i.e., one of

→ ’10’, ’19’, ’14’, ’22’, ’23’, ’31’,

→ ’13’) - who have not placed any orders

→ and whose account balance is greater

→ than the average positive account

→ balance for these regions? For each

→ country code, please return the number

→ of such customers and the total account

→ balance, sorting the results by the

→ country code.

‘‘‘sql FROM customer |> WHERE

substr(c_phone, 1, 2) IN (’10’, ’19’, → ’14’, ’22’, ’23’, ’31’, ’13’)

AND c_acctbal > ( SELECT avg(c_acctbal) FROM customer WHERE

c_acctbal > 0.00 AND substr(c_phone, 1, 2) IN (’10’,

→ ’19’, ’14’, ’22’, ’23’, ’31’,

→ ’13’)

) AND NOT EXISTS( FROM orders |> WHERE o_custkey = c_custkey

)

|> AGGREGATE COUNT(*) AS numcust, sum(c_acctbal) AS totacctbal

GROUP AND ORDER BY substr(c_phone, 1, 2) AS

→ cntrycode;

‘‘‘ ### 8. Suppliers Who Kept Orders Waiting Query Which suppliers in Peru were solely

→ responsible for delaying shipments in

→ multi-supplier orders with a final

→ status of ’F’? For each supplier, count → the number of orders where they failed → to meet the committed delivery date-

→ while every other supplier on the same

→ order delivered on time. Please list

→ the supplier names along with the count

→ of such delayed orders, ordered from

→ the highest number of delays to the

→ lowest, and show only the top 100

→ suppliers.

‘‘‘sql FROM

supplier, lineitem l1, orders, nation

|> WHERE s_suppkey = l1.l_suppkey AND o_orderkey = l1.l_orderkey AND o_orderstatus = ’F’ AND l1.l_receiptdate > l1.l_commitdate AND EXISTS(

FROM lineitem l2 |> WHERE

l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey)

AND NOT EXISTS( FROM lineitem l3 |> WHERE

l3.l_orderkey = l1.l_orderkey AND l3.l_suppkey <> l1.l_suppkey AND l3.l_receiptdate > l3. → l_commitdate) AND s_nationkey = n_nationkey AND n_name = ’PERU’

|> AGGREGATE COUNT(*) AS numwait GROUP BY s_name

|> ORDER BY numwait DESC, s_name

|> LIMIT 100; ‘‘‘

# Task Your task is to generate SQL in BigQuery’s

→ pipe query syntax described above based

→ on questions in natural language and

→ the presented database structure.

You will then write out your thought process

→ in detail followed by a single SQL

→ query enclosed in ‘‘‘sql ...‘‘‘ that

→ answers the question.

BigQuery database schema: | song : singer_id [number] (1, 2, 4), title [

→ text] (’Do They Know It’s Christmas’, ’

→ F**k It (I Don’t Want You Back)’, ’Cha

→ Cha Slide’), song_id [number] (1, 2, 3)

→ , sales [float] (1094000.0, 552407.0,

→ 300000.0), highest_position [float]

→ (1.0, 3.0) | singer : birth_year [float → ] (1944.0, 1948.0, 1949.0), citizenship → [text] (’France’, ’United States’, ’

→ Chile’), name [text] (’Liliane

→ Bettencourt’, ’Christy Walton’, ’Alice

→ Walton’), singer_id [number] (1, 2, 3),

→ net_worth_millions [float] (30.0,

→ 28.8, 26.3) |

The question: List the name of singers in → ascending order of net worth.

#### C.3 Error Classification

The prompt used for Section 5 simply outlines the taxonomy described in Appendix B.4.

You are provided with the following

→ information:

- **Natural Language Question:** A question

→ posed by a user.

- - **Predicted SQL Query:** SQL query generated

→ by a text-to-SQL model.

- - **Execution Results of Predicted Query:**

→ The results of executing the predicted

→ query (or an error message).

- - **Gold Standard SQL Query:** The correct SQL

→ query.

- - **Gold Standard Query Execution Results:**

→ Execution results of the correct (gold

→ standard) SQL query.

Analyze the predicted SQL query and determine

→ the type of error based on provided

→ execution results and query structures.

→ Classify the error into one of the

→ following categories:

- - **Dialect:** Issues due to differences

→ between SQL dialects.

- - **Schema Linking:** Incorrect matching of

→ natural language terms to schema

→ elements (e.g., hallucinated or

→ incorrect columns or tables).

- - **Data Type:** Issues arising from

→ mismatches or unexpected data types

→ within the database.

- - **Aggregation:** Incorrect use or omission

→ of aggregation functions (e.g., COUNT,

→ SUM), leading to inaccurate

→ summarization.

- - **Logical Form and Condition:** Incorrect

→ logical query structures, missing

→ conditions, inappropriate filters, or

→ incorrect ordering logic.

- - **Table Joins:** Incorrect or missing table

→ joins, irrelevant tables, or

→ misidentified join conditions.

- - **Projection:** Overall correct queries with

→ incorrect columns selected.

- - **Other:** Does not fit into any of the

→ categories above. Provide your analysis in the following format: ‘‘‘json {

"error_category": "Projection" | "Dialect" |

→ "Schema Linking" | "Logical Form and

→ Condition" | "Data Type" | "

→ Aggregation" | "Table Joins" | "Other

###### → ",

"reasoning": "Brief explanation supporting

→ your classification."

} ‘‘‘

