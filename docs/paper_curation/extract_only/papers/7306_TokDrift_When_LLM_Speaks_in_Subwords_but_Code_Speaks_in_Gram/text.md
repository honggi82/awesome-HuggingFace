## TOKDRIFT: When LLM Speaks in Subwords but Code Speaks in Grammar

Yinxi Li, Yuntian Deng, Pengyu Nie University of Waterloo {yinxi.li, yuntian, pynie}@uwaterloo.ca

# arXiv:2510.14972v1[cs.CL]16Oct2025

### Abstract

Large language models (LLMs) for code rely on subword tokenizers, such as byte-pair encoding (BPE), learned from mixed natural language text and programming language code but driven by statistics rather than grammar. As a result, semantically identical code snippets can be tokenized differently depending on superficial factors such as whitespace or identifier naming. To measure the impact of this misalignment, we introduce TOKDRIFT, a framework that applies semantic-preserving rewrite rules to create code variants differing only in tokenization. Across nine code LLMs, including large ones with over 30B parameters, even minor formatting changes can cause substantial shifts in model behavior. Layer-wise analysis shows that the issue originates in early embeddings, where subword segmentation fails to capture grammar token boundaries. Our findings identify misaligned tokenization as a hidden obstacle to reliable code understanding and generation, highlighting the need for grammar-aware tokenization for future code LLMs.

### 1 Introduction

Large language models (LLMs) have become powerful tools for programming tasks (Chen et al., 2021; Nye et al., 2021; Yang et al., 2024; Guo et al., 2024; Meta FAIR CodeGen Team, 2025). Before any modeling occurs, code is first tokenized into discrete units using a pretrained subword tokenizer such as byte-pair encoding (BPE) (Sennrich et al., 2016). However, the tokens that LLMs see, which are based on subword frequencies, are often very different from the tokens defined by programming language (PL) grammar. Whereas PLs have clear syntactic boundaries (e.g., keywords, identifiers, operators), subword tokenizers merge character sequences statistically, sometimes splitting identifiers at arbitrary points or combining unrelated symbols into a single token. This misalignment between

[Figure 1]

- (a) Workflow of TOKDRIFT, our framework for quantifying LLM sensitivity to semantic-preserving code rewrite rules.

[Figure 2]

- (b) Example of tokenization misalignment. Adding a space between dot (“.”) and “factorial” causes a significant change in token sequences, from [“.factor”, “ial”] to [“.”, “␣factorial”]. Consequently, the LLM’s code translation prediction shifts from incorrect (naming the factorial function as “comb” and later referring to it as “combin”) to correct.

Figure 1: TOKDRIFT workflow and example.

subwords and syntax means that LLMs do not always process code in the units that programmers or compilers would expect.

As an example, the presence of a space before an identifier can lead to completely different token sequences, and thus different predictions, despite identical program semantics (Figure 1). While such differences may appear superficial, they raise a deeper concern about how robustly code LLMs represent grammar and meaning. If tokenization determines how code is segmented and embedded, even small discrepancies could propagate through the model and alter its predictions. This motivates the central question of our study:

Does the misalignment between subword tokenization and PL grammar limit LLMs’ ability to understand and generate code?

To study this question, we introduce TOKDRIFT, a framework that applies semantic-preserving rewrite rules, such as changing whitespace or identifier casing style, to create pairs of programs that are semantically equivalent but tokenized differently. We evaluate nine code LLMs across three representative programming tasks—bug fixing, code summarization, and code translationand measure whether model outputs remain functionally equivalent when tokenization changes.

Our experiments show that even minor tokenization variations can substantially impact model behavior. For example, the most performant LLM in our experiment, Qwen2.5-Coder-32B-Instruct, changes its prediction 6.09% of the times when the input tokenization changes (and up to 60% under a single rewrite rule). Layer-wise analysis further indicates that the effect originates in early layers, where subword segmentation fails to align with grammatical token boundaries. Together, these findings suggest that tokenizer design remains a critical yet under-explored factor in developing robust and grammar-aware code LLMs. The main contributions of this work include:

- • We identify and formalize the misaligned tokenization problem in code LLMs.
- • We introduce TOKDRIFT, a framework for quantifying model sensitivity to semantic-preserving code rewrites that alter tokenization.
- • We conduct a large-scale empirical study showing that misaligned tokenization affects all evaluated models and persists with scaling.
- • We open-source our framework and data to facilitate future research on grammar-aware and domain-adaptive tokenization.

Our code and data are available at: https://github.com/uw-swag/tokdrift

### 2 Background

#### 2.1 LLM Tokenization

Tokenization is the first step in processing input for LLMs, converting raw text into a sequence of discrete tokens. Each token corresponds to a model time step and has a dedicated embedding. Modern LLMs use learned tokenization strategies that eliminate the out-of-vocabulary problem by starting from minimal units, such as characters or bytes, and learning how to merge them into longer fragments based on frequency in a large corpus. Popular approaches like BPE (Sennrich et al., 2016) and

[Figure 3]

Figure 2: Heatmap of (code) LLMs’ vocabulary distances (Amba Hombaiah et al., 2021).

WordPiece (Schuster and Nakajima, 2012; Devlin et al., 2019) follow this general principle, differing mainly in their merge heuristics. Often, pretokenization steps like splitting at whitespace are applied before learning to prevent tokens from spanning across word boundaries.

The tokenizers used by different LLMs can vary significantly due to differences in pre-tokenization rules, token learning algorithms, and pretraining corpora. As shown in Figure 2, even models from the same family often share less than half of their vocabulary, such as Llama 3 vs. Llama 4. The main exception occurs when model developers intentionally reuse the same tokenizer across variants, such as Qwen2.5 and Qwen2.5-Coder, which share an identical vocabulary and tokenizer configuration.

#### 2.2 PL Tokenization

Tokenization in PLs, often called lexing, is the first step of code parsing: it transforms a stream of characters into a sequence of tokens according to a PL’s grammar. These tokens are then passed to a parser, which constructs an abstract syntax tree (AST) to represent the program’s structure.

While exact rules vary by language, most PLs share a common set of token types, including: identifiers (e.g., variable or function names), operators (e.g., +, *), keywords (e.g., if, return), literals (e.g., numeric or string constants), and whitespace, which is typically used to separate tokens but is otherwise ignored.

Unlike LLM tokenization, PL tokenization in compilers and interpreters is deterministic. For example, the snippet x+1 is always tokenized into three tokens: an identifier (x), an operator (+), and a literal (1). Formatting changes, such as adding spaces, do not affect the token sequence as long as the code remains syntactically valid.

Table 1: Benchmarks in our experiments. We manually examine the benchmarks to follow the naming conventions, and to fix/exclude invalid tests and samples, see details in Section C.1.

Benchmark Source Task Input PL Output PL # Samples

|HumanEval-Fix-py HumanEval-Fix-java|HumanEvalPack (Muennighoff et al., 2023)<br><br>|bug fixing<br><br>|Python Python 164 Java Java 164|
|---|---|---|---|
|HumanEval-Explain-py HumanEval-Explain-java| |code summarization<br><br>|Python Python 164 Java Java 164|
|Avatar-py2java Avatar-java2py<br><br>|Avatar (Ahmad et al., 2023; Pan et al., 2024)<br><br>|code translation<br><br>|Python Java 244 Java Python 246|
|CodeNet-py2java CodeNet-java2py<br><br>|CodeNet (Puri et al., 2021; Pan et al., 2024)| |Python Java 200 Java Python 200|

This behavior misaligns with LLM tokenizers: while PL tokenizers produce stable, grammaraware units, LLM tokenizers frequently break code structure, resulting in inconsistent or fragmented representations of semantically identical programs. In this work, we refer to grammar-aware tokens as PL tokens, and contrast them with the LLM tokens produced by learned subword tokenizers.

### 3 TOKDRIFT Framework

Figure 1a illustrates the overall workflow of TOKDRIFT, our framework for quantifying model sensitivity to semantic-preserving code rewrites that alter tokenization. In a nutshell, TOKDRIFT systematically compares the LLM outputs given the baseline input tokens and variant input tokens (after applying rewrite rules) through a large set of experiments. Each experiment is performed on a specific benchmark, and tests the sensitivity of a given LLM against a specific rewrite rule.

#### 3.1 Benchmarks

We searched for recent popular coding LLM benchmarks where: (1) the input includes a code snippet, since rewrite rules cannot be applied on natural language; (2) the output is evaluated with an automated functional correctness metric.We focused on two popular PLs, Java and Python.

Based on these criteria, we selected eight benchmarks covering three tasks, listed in Table 1. Bug fixing (Tufano et al., 2019) transforms a buggy code snippet into a correct one. Code summarization (Hu et al., 2018; Panthaplackel et al., 2020) aims at summarizing a code snippet into natural language description; following HumanEvalPack’s setup (Muennighoff et al., 2023), the description is fed back to LLM to generate code for measuring correctness. Code translation (Ahmad et al., 2023; Puri et al., 2021) is the task of translating a code snippet from one PL to another. All benchmarks use tests to evaluate the correctness of outputs.

Table 2: Models used in our experiments.

Series S M L

Llama-3 3B 8B 70B Qwen2.5-Coder 1.5B 7B 32B DeepSeek-Coder 1.3B 6.7B 33B

#### 3.2 Models

- Table 2 lists the models used in TOKDRIFT. We selected three series of popular open-source LLMs (using the coding-specific variants if available), namely Llama-3, Qwen2.5-Coder, and DeepSeekCoder. To cover the model size spectrum, we used small (∼1B parameters), medium (∼7B), and large (>30B) variants in each series. All models are instruction-tuned. We perform greedy decoding to generate deterministic outputs (see experimental environment details in Section C.4).

3.3 Rewrite Rules

- Table 3 lists the rewrite rules used in TOKDRIFT. Each rewrite rule converts all occurrences of the left-hand side substring to the right-hand side substring. According to the grammars of the two PLs we experiment on (and generally for most modern PLs), these rewrite rules are semanticallypreserving by design. We apply one rewrite rule at a time to investigate their impact in isolation.

The six rewrite rules starting with “N” are inspired by naming conventions. Identifiers usually follow one of the four casing styles: camelCase (for variables/functions in Java), PascalCase (for classes in Java/Python), snake_case (for variables/functions in Python), and SCREAMING_CASE (for constants in Java/Python). Since variables/functions are most common among identifiers, we design rewrite rules to alter their casing style. Specifically, N1, N2, N3 convert camelCase identifiers in Java to the other three casing styles, while N4, N5, N6 convert snake_case identifiers in Python. These rewrite rules challenge LLMs’ robustness to different naming styles.

Table 3: Rewrite rules supported by TOKDRIFT, inspired by naming conventions (starting with N) and spacing conventions (starting with S). Each rewrite rule may apply to Java (marked by J), Python (marked by P), or both.

No. PL Rewrite Rule Description Example

- N1 J camelCase →snake_case

Convert identifiers from the most common casing style in the input PL to alternative ones

␣sorted L st → ␣sorted _lst

- N2 J camelCase →PascalCase ␣cloestPair → ␣Close st Pair

- N3 J camelCase →SCREAMING_CASE ␣possible S olutions → ␣POSS IBLE _S OLUTION S

- N4 P snake_case →camelCase ␣input _clip board → ␣input Clipboard

- N5 P snake_case →PascalCase ␣string _xor → ␣String X or

- N6 P snake_case →SCREAMING_CASE ␣triangle _area → ␣TRI ANGLE _AREA

- S1 P OP-→OP␣- Add space between operator and minus sign [::- 1 ] → [ :: ␣- 1 ]

- S2 P OP[→OP␣[ Add space between operator and left square bracket )) )[ 2 :]\n → ))) ␣[ 2 :]\n

- S3 J ).→)␣. Add space between right parentheses and period ␣'. '). replace → ␣'.') ␣. replace

- S4 P ])→]␣) Add space between right square bracket and right parentheses : ]):\n → :] ␣):\n

- S5 P OP]→OP␣] Add space between operator and right square bracket = ␣[[] → = ␣[[ ␣]

- S6 J OP(→OP ␣( Add space between operator and left parentheses (( ! is True → ( ␣(! is True

- S7 P [ID→[␣ID Add space between left square bracket and identifier ([ v ow els → ([ ␣vowels

- S8 J ++)→++␣) Add space between increment operator and right parentheses ␣i ++) → ␣i ++ ␣)

- S9 J .*→.␣* Add space between period and asterisk .*;\n → . ␣* ;\n

- S10 P ):→)␣: Add space between right parentheses and colon ␣main ():\n → ␣main () ␣:\n

- S11 J );→)␣; Add space between right parentheses and semicolon <>();\n → < >() ␣;\n

- S12 J OP;→OP␣; Add space between operator and semicolon Ac ++; → Ac ++ ␣;

- S13 J P ))→)␣) Add space between two right parentheses .toCharArray ()) → .toCharArray () ␣)

- S14 J P ()→(␣) Add space between two left parentheses alpha () → alpha ( ␣)

- S15 J P .ID→.␣ID Add space between period and identifier .factor ial → . ␣factorial

- S16 J P (ID→(␣ID Add space between left parentheses and identifier (String → ( ␣String

- S17 J P OPID→OP␣ID Add space between operator and identifier :i +len (sub string → : ␣i + ␣len ( ␣substring

- S18 J P OPALL→OP␣ALL Add space between operator and identifier/operator (l : ␣list ):\n → ( ␣l : ␣list ) ␣:\n

The eighteen rewrite rules starting with “S” are inspired by spacing conventions. Whitespace around most operators usually carries no semantic meaning and is optional. Thus, the spacingrelated rewrite rules identifies two consecutive tokens (one of them is an operator) and inserts a space in between. Specifically, we look for combinations where one of them is a specific operator or any kind of operator (represented by OP), and the other one is another specific operator or an identifier (represented by ID). Exploring all combinations would be infeasible, thus we select the top-10 frequently appearing combinations in the benchmarks for each PL. In addition, we add S17 and S18 as “wildcard” rules to cover all cases where an OP is followed by an ID or ID/OP for both PLs. These rewrite rules challenge LLM and its tokenizer’s robustness to different formatting styles. Notably, in most LLMs with a pre-tokenization step of splitting before whitespace, these rewrite rules will lead to more LLM tokens.

#### 3.4 Metrics

Recall that each experiment on a given {benchmark, model, rewrite rule} triplet compares the baseline outputs (given the original inputs) and the variant outputs (given the inputs after applying rewrite rule). The benchmark provides a set of tests to

evaluate whether each output is correct or incorrect. We define accuracy as the percentage of correct outputs, and ∆accuracy as the variant’s accuracy minus the baseline’s accuracy.

The ∆accuracy metric, although intuitive, has two limitations: (1) accuracy improvements and degradations on individual samples cancel out; (2) some samples may not be affected by a rewrite rule if the left-hand side substring does not appear in the input; the outputs of those samples will never change. To address these, we introduce an unbiased metric called sensitivity, defined as the percentage of the samples whose output correctness flips (from correct to incorrect or vice versa) out of the samples whose input is changed by the rewrite rule. A lower sensitivity indicates that the model is more robust against the token changes introduced by a rewrite rule; when averaged across all rewrite rules, it reflects how sensitive the model is to the LLM-PL tokenization misalignment.

### 4 Evaluation

#### 4.1 Results

Table 4 shows the accuracy and ∆accuracy of each model on each rewrite rule. We can observe that most rewrite rules cause measurable changes in model accuracy, ranging from -2.90 to +0.32 abso-

Table 4: Accuracy and ∆accuracy (in parenthesis) of each model on each rewrite rule.

Variant Llama-3B Llama-8B Llama-70B Qwen-1.5B Qwen-7B Qwen-32B DS-1.3B DS-6.7B DS-33B Average Input PL = Java baseline 32.04 43.15 57.24 33.59 57.36 70.41 38.50 58.01 57.36 49.74

- N1 32.69 (+0.65) 43.54 (+0.39) 57.49 (+0.25) 35.27 (+1.68) 57.62 (+0.26) 70.28 (-0.13) 37.98 (-0.52) 57.36 (-0.65) 57.11 (-0.25) 49.93 (+0.19)

- N2 32.17 (+0.13) 43.54 (+0.39) 56.85 (-0.39) 35.27 (+1.68) 57.75 (+0.39) 70.41 (+0.00) 39.02 (+0.52) 58.14 (+0.13) 57.36 (+0.00) 50.06 (+0.32)

- N3 32.56 (+0.52) 44.19 (+1.04) 56.20 (-1.04) 35.53 (+1.94) 58.01 (+0.65) 69.12 (-1.29) 38.37 (-0.13) 56.33 (-1.68) 56.46 (-0.90) 49.64 (-0.10)

- S3 31.65 (-0.39) 43.02 (-0.13) 56.20 (-1.04) 34.37 (+0.78) 56.72 (-0.64) 70.41 (+0.00) 37.34 (-1.16) 58.66 (+0.65) 57.88 (+0.52) 49.58 (-0.16)

- S6 31.52 (-0.52) 43.02 (-0.13) 57.62 (+0.38) 33.20 (-0.39) 57.49 (+0.13) 70.28 (-0.13) 37.98 (-0.52) 58.53 (+0.52) 57.49 (+0.13) 49.68 (-0.06)

- S8 31.91 (-0.13) 43.28 (+0.13) 57.24 (+0.00) 34.11 (+0.52) 56.72 (-0.64) 71.45 (+1.04) 38.63 (+0.13) 57.49 (-0.52) 58.27 (+0.91) 49.90 (+0.16)

- S9 32.30 (+0.26) 40.96 (-2.19) 58.66 (+1.42) 33.46 (-0.13) 58.14 (+0.78) 69.51 (-0.90) 36.95 (-1.55) 56.59 (-1.42) 57.75 (+0.39) 49.37 (-0.37)

- S11 32.69 (+0.65) 44.57 (+1.42) 55.17 (-2.07) 35.14 (+1.55) 56.33 (-1.03) 71.58 (+1.17) 37.34 (-1.16) 57.11 (-0.90) 57.11 (-0.25) 49.67 (-0.07)

- S12 30.49 (-1.55) 43.02 (-0.13) 56.07 (-1.17) 34.75 (+1.16) 55.81 (-1.55) 67.05 (-3.36) 38.63 (+0.13) 55.94 (-2.07) 58.53 (+1.17) 48.92 (-0.82)

- S13 32.43 (+0.39) 42.64 (-0.51) 56.59 (-0.65) 33.46 (-0.13) 57.36 (+0.00) 69.77 (-0.64) 37.47 (-1.03) 58.27 (+0.26) 56.98 (-0.38) 49.44 (-0.30)

- S14 29.84 (-2.20) 41.09 (-2.06) 54.13 (-3.11) 32.17 (-1.42) 56.85 (-0.51) 71.19 (+0.78) 37.86 (-0.64) 57.11 (-0.90) 57.62 (+0.26) 48.65 (-1.09)

- S15 30.62 (-1.42) 36.82 (-6.33) 57.24 (+0.00) 33.46 (-0.13) 56.72 (-0.64) 70.28 (-0.13) 37.34 (-1.16) 55.43 (-2.58) 59.43 (+2.07) 48.59 (-1.15)

- S16 30.88 (-1.16) 40.83 (-2.32) 55.94 (-1.30) 34.88 (+1.29) 57.36 (+0.00) 71.96 (+1.55) 36.43 (-2.07) 57.49 (-0.52) 58.66 (+1.30) 49.38 (-0.36)

- S17 28.68 (-3.36) 37.34 (-5.81) 56.07 (-1.17) 35.66 (+2.07) 55.43 (-1.93) 70.03 (-0.38) 35.40 (-3.10) 55.04 (-2.97) 58.91 (+1.55) 48.06 (-1.68)

- S18 25.97 (-6.07) 34.88 (-8.27) 56.85 (-0.39) 34.11 (+0.52) 56.07 (-1.29) 70.28 (-0.13) 33.98 (-4.52) 53.10 (-4.91) 56.33 (-1.03) 46.84 (-2.90) Input PL = Python

baseline 39.12 49.87 69.04 40.67 64.51 76.17 44.82 61.92 68.13 57.14 N4 40.03 (+0.91) 51.04 (+1.17) 68.91 (-0.13) 39.77 (-0.90) 65.03 (+0.52) 77.85 (+1.68) 44.30 (-0.52) 61.53 (-0.39) 68.39 (+0.26) 57.43 (+0.29) N5 37.56 (-1.56) 50.91 (+1.04) 68.65 (-0.39) 39.25 (-1.42) 64.77 (+0.26) 77.72 (+1.55) 42.88 (-1.94) 61.53 (-0.39) 68.39 (+0.26) 56.85 (-0.29) N6 38.08 (-1.04) 50.65 (+0.78) 66.19 (-2.85) 39.38 (-1.29) 64.51 (+0.00) 76.81 (+0.64) 42.23 (-2.59) 61.14 (-0.78) 67.62 (-0.51) 56.29 (-0.85)

- S1 39.38 (+0.26) 50.39 (+0.52) 68.65 (-0.39) 40.54 (-0.13) 64.51 (+0.00) 76.68 (+0.51) 44.69 (-0.13) 62.56 (+0.64) 67.62 (-0.51) 57.22 (+0.08)

- S2 39.64 (+0.52) 50.65 (+0.78) 68.78 (-0.26) 40.41 (-0.26) 64.77 (+0.26) 75.91 (-0.26) 43.65 (-1.17) 62.44 (+0.52) 67.75 (-0.38) 57.11 (-0.03)

S4 39.77 (+0.65) 50.65 (+0.78) 69.30 (+0.26) 40.54 (-0.13) 64.51 (+0.00) 73.19 (-2.98) 44.82 (+0.00) 61.92 (+0.00) 67.36 (-0.77) 56.90 (-0.24) S5 38.60 (-0.52) 50.78 (+0.91) 68.91 (-0.13) 40.80 (+0.13) 64.12 (-0.39) 76.94 (+0.77) 44.43 (-0.39) 62.69 (+0.77) 66.71 (-1.42) 57.11 (-0.03) S7 40.03 (+0.91) 49.35 (-0.52) 68.26 (-0.78) 40.67 (+0.00) 63.34 (-1.17) 76.42 (+0.25) 44.30 (-0.52) 62.69 (+0.77) 67.23 (-0.90) 56.92 (-0.22)

- S10 38.47 (-0.65) 50.65 (+0.78) 69.17 (+0.13) 40.67 (+0.00) 63.99 (-0.52) 77.46 (+1.29) 44.56 (-0.26) 62.05 (+0.13) 67.10 (-1.03) 57.12 (-0.02)

- S13 37.95 (-1.17) 50.13 (+0.26) 69.30 (+0.26) 40.54 (-0.13) 64.90 (+0.39) 76.55 (+0.38) 44.30 (-0.52) 62.05 (+0.13) 67.10 (-1.03) 56.98 (-0.16)

- S14 38.73 (-0.39) 49.22 (-0.65) 68.39 (-0.65) 39.38 (-1.29) 63.73 (-0.78) 74.09 (-2.08) 45.08 (+0.26) 61.66 (-0.26) 67.49 (-0.64) 56.42 (-0.72)

- S15 39.12 (+0.00) 50.26 (+0.39) 67.49 (-1.55) 39.77 (-0.90) 62.69 (-1.82) 76.30 (+0.13) 44.17 (-0.65) 61.66 (-0.26) 67.23 (-0.90) 56.52 (-0.62)

- S16 40.16 (+1.04) 49.87 (+0.00) 69.04 (+0.00) 39.64 (-1.03) 63.08 (-1.43) 76.68 (+0.51) 43.65 (-1.17) 61.27 (-0.65) 67.23 (-0.90) 56.74 (-0.40)

- S17 40.41 (+1.29) 50.39 (+0.52) 67.62 (-1.42) 39.38 (-1.29) 61.92 (-2.59) 76.55 (+0.38) 42.62 (-2.20) 60.49 (-1.43) 66.32 (-1.81) 56.19 (-0.95)

- S18 37.44 (-1.68) 49.87 (+0.00) 67.62 (-1.42) 38.34 (-2.33) 63.08 (-1.43) 75.13 (-1.04) 42.49 (-2.33) 62.05 (+0.13) 67.36 (-0.77) 55.93 (-1.21)

Background color: baseline in grey, variants better than baseline in green, and variants worse than baseline in red. The best variant is highlighted in bold and the worst variant is underlined.

lute percentage points if averaging across all models. The largest ∆accuracy of -8.27% happens on Llama-8B for Java benchmarks, whose accuracy drops from 43.15% to 34.88% when applying rewrite rule S18 (adding space after each operator). Considering advances in LLM performance are sometimes claimed with around 1 percentage point margin, these accuracy deltas caused by simple rewrite rules are non-negligible.

The impact of misaligned tokenization is more apparent in the sensitivity metric, as shown in the distribution plots in Figure 3. The average sensitivity is 9.26% for naming rewrites and 8.29% for spacing rewrites. Among the naming rewrites (Figure 3a), LLMs are relatively less sensitive to transductions between camelCase and snake_case (N1 and N4), likely because camelCase and SCREAMING_CASE are less frequent. This finding implies that the casing styles of identifiers, while technically convey no semantic meaning in PLs, are an important factor in LLMs’ understanding of code. In Figure 3b, we can see that LLMs’ aver-

age sensitivity is over 10% for the two “wildcard” spacing rewrite rules (S17 and S18). Other spacing rewrite rules result in varying levels of sensitivity, among which the most impactful ones are S15 (adding space between period and identifier), S14 (adding space between a pair of parentheses), and S12 (adding space between operator and semicolon). In terms of the average sensitivity of models (Figure 3c), we observe that Llama-3 models are more sensitive than the other two series, but all models persist a non-negligible sensitivity of at least 5.71% (Qwen-32B on spacing rewrite rules).

#### 4.2 Impact of Model Size

We investigate whether larger models are less sensitive to tokenization changes, with the general assumption of larger models being more robust. Table 5 shows the the average sensitivity of models at different sizes, where the small, medium, and large models in each series are compared on a row. While the small and medium models are at around the same level of sensitivity, the large models are

[Figure 4]

- (a) grouped by naming rewrite rule

[Figure 5]

- (b) grouped by spacing rewrite rule

[Figure 6]

(c) grouped by model

Figure 3: Violin plots of sensitivity distributions.

usually less sensitive (i.e., more robust) than their smaller counterparts, with only one exception of Qwen-32B on naming rewrite rules.

We also perform statistically significant tests via Wilcoxon signed-rank test (Conover, 1999). The results show that the differences are not significant for naming rules, but significant for spacing rules (except between the small and medium models for Qwen2.5-Coder and DeepSeek-Coder series).

#### 4.3 Impact of Identifier Fragment Changes

We noticed that identifiers are frequently tokenized into different subwords before and after applying rewrite rules. For example, Llama-3 tokenizes ‘␣sortedLst’ into three tokens [‘␣sorted’, ‘L’,

‘st’], and applying N1 changes it into two tokens [‘␣sorted’, ‘_lst’]. We define this case as identifier fragment change: the list of fragments (tokens but ignoring spaces and underscores) changes be-

Table 5: Impact of model size on sensitivity.

Rewrite Rule Model Series S M L

Llama-3 11.48 10.68 9.43 Qwen2.5-Coder 7.73 7.95 8.27 DeepSeek-Coder 9.88 8.95 8.95

Naming

Llama-3 10.22 10.99 8.51 Qwen2.5-Coder 7.07 8.87 5.71 DeepSeek-Coder 8.36 8.71 6.26

Spacing

Table 6: Impact of identifier fragment changes on sensitivity. “Unchanged” samples do not have any identifier fragment change, and “Changed” samples have at least one identifier fragment change.

Rewrite Rule Model Unchanged Changed

Llama-70B 8.13 11.21 Qwen-32B 6.58 10.57 DS-33B 6.61 10.82

Naming

Llama-70B 7.24 11.89 Qwen-32B 5.09 7.37 DS-33B 5.80 7.12

Spacing

fore and after applying rewrite rules. Using this concept, we can categorize the samples into two groups, one without any identifier fragment change (i.e., “Unchanged”), and the other with at least one identifier fragment change (i.e., “Changed”).

Table 6 shows the average sensitivity of models on the two groups of samples; note that we focus on the large model in each series in this analysis. The identifier fragment changed group shows consistently higher sensitivity than the unchanged group, with the largest difference on for naming rewrite rules (10.82% vs. 6.61%). This finding suggests that how identifiers are tokenized into subwords play an important role in LLMs’ understanding of code. Arguably, identifiers are frequently not tokenized into semantically meaningful subwords (such as the ‘␣sortedLst’ example), which may fundamentally limit the model’s code comprehension and generation capabilities.

### 5 Root Cause Analyses

In addition to quantifying its impact, we also study why LLMs are sensitive to tokenization changes, along two aspects: (1) word frequency in the pretraining corpus (Section 5.1); (2) LLM’s hidden states before and after the rewrite rule (Section 5.2).

#### 5.1 Word Frequency Analysis

Our hypothesis is that there is a correlation between sensitivity and the word frequencies of the rewrite rule’s left-hand side and right-hand side. If the ratio of right-hand side to left-hand side word fre-

Table 7: Word frequency of rewrite rules’ left-hand side (LHS) and right-hand side (RHS) on GitHub. Ratio is the percentage of RHS to LHS word frequency.

Rewrite Rule LHS RHS Ratio [%] Java

- S3: ).→)␣. 78.9M 45.7K 0.06

- S8: ++)→++␣) 22.9M 664K 2.90
- S9: .*→.␣* 34.2M 7.3M 21.35 S11: );→)␣; 161M 924K 0.57

- S13: ))→)␣) 102M 3.4M 3.33
- S14: ()→(␣) 144M 195K 0.14
- S15: .ID→.␣ID 175M 45.9M 16.22
- S16: (ID→(␣ID 172M 6.6M 3.84 Python

S4: ])→]␣) 44.6M 1.7M 3.81 S7: [ID→[␣ID 61.1M 1.1M 1.83

- S10: ):→)␣: 76M 1.4M 1.84

- S13: ))→)␣) 59M 2.4M 4.07
- S14: ()→(␣) 78.1M 71.7K 0.09
- S15: .ID→.␣ID 107M 40.6M 37.94
- S16: (ID→(␣ID 105M 2.9M 2.76

quency is small (meaning right-hand side is rare in the corpus), LLMs will likely perform worse after applying the rewrite rule. We measure the word frequencies on GitHub, a primary source of code data in LLMs’ pretraining corpora.1

Table 7 shows the word frequencies of the rewrite rules, and the ratio (in percentages) of the right-hand side to the left-hand side word frequency. The ratio is always less than 100%, which explains why LLMs exhibit non-negligible sensitivity to all rewrite rules. Some rewrite rules with low ratio, e.g., S14, also exhibit high sensitivity in Figure 3b.

#### 5.2 Hidden State Analysis

LLMs’ hidden states represent their internal comprehension and reasoning processes, which may help explain their sensitive to tokenization changes. We compare the hidden states before and after applying the rewrite rules. For each tokens sequence changed, we extract the hidden states of the last token in the sequence, which summarizes the information of the entire sequence. We focus this analysis on the best-performing LLM, Qwen-32B.

We first measure the cosine similarity between the hidden states before and after applying the rewrite rules. Figure 4 shows correlation between the layer from which the hidden states are extracted and the similarity. For both naming and spacing rewrite rules, the similarity starts from almost 0 in the first (input) layer, increases (and stabilizes in

1We use GitHub’s search feature to measure word frequencies; due to the limitation in regular expressions and characters that can be used in the search string, we can only conduct this analysis on a subset of the spacing rewrite rules.

[Figure 7]

- (a) naming rewrite rules

[Figure 8]

- (b) spacing rewrite rules

Figure 4: The similarity of each layer’s hidden states before and after applying rewrite rules.

most cases) in middle layers, and drops again at the last (output) layer. This observation is consistent with the information bottleneck theory (Saxe et al., 2019), which states that the middle layers capture the compressed semantic information. Interestingly, in Figure 4b, we observe that for some spacing rewrite rules (S14 and S3), the similarity in middle layers is also low, implying that the model sees the before and after versions as semantically different. These rewrite rules match the ones that LLMs are most sensitive to in Figure 3b.

Then, we compute the hidden state diffs as the hidden states after applying rewrite rules minus those before applying, on the medium layer of the model which should best capture semantic information. Figure 5 shows the visualizations of the hidden state diffs using t-SNE (Maaten and Hinton, 2008). We observe that the diffs of naming and spacing rewrite rules are clearly distinguishable (Figure 5a), so are the diffs of naming (Figure 5b) and spacing rewrite rules (Figure 5c, note that S17 and S18 are excluded since they are supersets of other rewrite rules). This confirms that the hidden states, especially from the middle layers, are good representations of semantic information and may be utilized to mitigate the tokenization changes.

[Figure 9]

[Figure 10]

[Figure 11]

(a) naming vs. spacing

(b) naming rewrite rules (c) spacing rewrite rules

Figure 5: Visualizations of the hidden state diffs using t-SNE (Maaten and Hinton, 2008).

### 6 Related Work

Tokenization Most modern LLMs use subword tokenizers such as BPE (Sennrich et al., 2016), which create vocabularies based on how often character sequences occur together. The resulting token types do not always correspond to meaningful words or code elements, and can vary depending on how the tokenizer was trained. For example, Liu et al. (2025) shows that allowing token merges across whitespace boundaries produces more meaningful units, compared to tokenizers that always split at spaces. Chirkova and Troshin (2023) introduces a tokenizer designed to better align with PL syntax, achieving lower token counts while preserving model performance. These studies show that tokenization can influence how well a model understands and generates code, and our work builds on this line of inquiry by quantifying the effects of semantic-preserving tokenization changes.

Robustness to Representation Variations Another important question is how robust LLMs are to variations in tokenization and representation at inference time. Zheng et al. (2025) show that instruction-tuned models can often retain high performance even when inputs are tokenized in unconventional or character-level formats, suggesting that such models may learn generalizable internal representations. However, their study also shows a measurable performance drop compared to standard tokenizations, and other work highlights further limitations. Wang et al. (2025) find that adversarial changes to token boundaries can significantly degrade model predictions, especially in models that have not undergone instruction tuning. In structured domains like chemistry, Yan et al. (2025) demonstrate that LLMs produce inconsistent outputs across semantically equivalent molecular representations. These findings suggest that LLMs remain sensitive to surface-level varia-

tions. Our work contributes to this line by focusing specifically on PLs.

Syntax-Aware Code Modeling To address the mismatch between subword tokenization and PL grammar, several approaches incorporate grammar constraints into the LLM decoding process. Synchromesh (Poesia et al., 2022) and PICARD (Scholak et al., 2021) enforce syntactic validity at generation time by using runtime parsing to filter out invalid token continuations. SynCode (Ugare et al., 2024) improves the efficiency of such methods by constructing a DFA-based mask that precomputes token legality while explicitly handling partial tokens. Boundless BPE (Schmidt et al., 2025) removes fixed pretokenizers and enables dynamic boundary selection, allowing the model to learn tokens that correspond to syntactic or semantic units. Together, these efforts aim to align LLM outputs more closely with formal code structure, a disconnect that our work quantifies by measuring how semantics-preserving tokenization variations affect model behavior.

### 7 Conclusions

This work studies the tokenization misalignment between subword-based LLMs and PL grammar. While subword tokenizers like BPE are widely used in code LLMs, they segment inputs based on frequency statistics, not grammar, leading to token boundaries that may not align with syntactic units in code. Through a suite of semantic-preserving rewrite rules, our framework TOKDRIFT shows that even minor formatting changes, such as whitespace edits or identifier renamings, can cause substantial shifts in model outputs. These effects hold across nine coding LLMs and three tasks (fixing, summarization, and translation). These findings motivate future research for grammar-aware or domain-adaptive tokenizers that more faithfully reflect PL structure.

### Limitations

While our study shows limitations of current tokenizer designs in code LLMs, our analysis focuses on a targeted set of semantic-preserving rewrites based on common formatting and naming conventions; these do not encompass all potential sources of tokenization drift. Second, although we evaluate nine widely used code LLMs, our findings may not generalize to models with fundamentally different architectures (e.g., state space models (Gu et al.,

- 2022)) or tokenization strategies (e.g., characterlevel or grammar-driven tokenizers (Kim et al., 2016)). Third, our work centers on measurement and diagnosis, and we do not explore mitigation strategies. Future work could investigate tokenizer retraining, ensemble decoding over multiple tokenizations, or architectural modifications to improve the alignment between token boundaries and programming language syntax. Acknowledgments

We thank Yu Liu for valuable comments and feedback. This work was supported in part by Compute Ontario (computeontario.ca) and the Digital Research Alliance of Canada (alliancecan.ca). It was also partially supported by a Natural Sciences and Engineering Research Council of Canada (NSERC) Discovery Grant (RGPIN-2024-04909) and a startup grant from the University of Waterloo. Yuntian Deng is additionally supported by an NSERC Discovery Grant (RGPIN-2024-05178) and a start-up grant from the University of Waterloo.

### References

Wasi Ahmad, Md Golam Rahman Tushar, Saikat Chakraborty, and Kai-Wei Chang. 2023. AVATAR: A parallel corpus for Java-Python program translation. In Findings of the Association for Computational Linguistics: ACL, pages 2268–2281.

Spurthi Amba Hombaiah, Tao Chen, Mingyang Zhang, Michael Bendersky, and Marc Najork. 2021. Dynamic language models for continuously evolving content. In International Conference on Knowledge Discovery and Data Mining, pages 2514–2524.

Loubna Ben Allal, Niklas Muennighoff, Logesh Kumar Umapathi, Ben Lipkin, and Leandro von Werra. 2022. A framework for the evaluation of code generation models. https://github.com/bigcode-project/ bigcode-evaluation-harness.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan,

Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, and 1 others. 2021. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Nadezhda Chirkova and Sergey Troshin. 2023. Codebpe: Investigating subtokenization options for large language model pretraining on source code. Preprint, arXiv:2308.00683.

William Jay Conover. 1999. Practical nonparametric statistics. john wiley & sons.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Albert Gu, Karan Goel, and Christopher Ré. 2022. Efficiently modeling long sequences with structured state spaces. In The International Conference on Learning Representations (ICLR).

Batu Guan, Xiao Wu, Yuanyuan Yuan, and Shaohua Li. 2025. Is your benchmark (still) useful? dynamic benchmarking for code language models. arXiv preprint arXiv:2503.06643.

Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y. Wu, Y. K. Li, Fuli Luo, Yingfei Xiong, and Wenfeng Liang. 2024. Deepseek-coder: When the large language model meets programming – the rise of code intelligence. Preprint, arXiv:2401.14196.

Xing Hu, Ge Li, Xin Xia, David Lo, and Zhi Jin. 2018. Deep code comment generation. In International Conference on Program Comprehension, pages 200– 210.

Yoon Kim, Yacine Jernite, David Sontag, and Alexander Rush. 2016. Character-aware neural language models. Proceedings of the AAAI Conference on Artificial Intelligence, 30(1).

Alisa Liu, Jonathan Hayase, Valentin Hofmann, Sewoong Oh, Noah A. Smith, and Yejin Choi. 2025. Superbpe: Space travel for language models. Preprint, arXiv:2503.13423.

Laurens van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-SNE. Journal of Machine Learning Research, 9(Nov):2579–2605.

Meta FAIR CodeGen Team. 2025. Cwm: An openweights llm for research on code generation with world models. Technical report, Meta. 32Bparameter open-weights model; inference code and weights released.

Niklas Muennighoff, Qian Liu, Armel Zebaze, Qinkai Zheng, Binyuan Hui, Terry Yue Zhuo, Swayam Singh, Xiangru Tang, Leandro von Werra, and Shayne Longpre. 2023. OctoPack: Instruction tuning code large language models. arXiv preprint arXiv:2308.07124.

Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, and Augustus Odena. 2021. Show your work: Scratchpads for intermediate computation with language models. Preprint, arXiv:2112.00114.

Rangeet Pan, Ali Reza Ibrahimzada, Rahul Krishna, Divya Sankar, Lambert Pouguem Wassi, Michele Merler, Boris Sobolev, Raju Pavuluri, Saurabh Sinha, and Reyhaneh Jabbarvand. 2024. Lost in translation: A study of bugs introduced by large language models while translating code. In Proceedings of the IEEE/ACM 46th International Conference on Software Engineering, pages 1–13.

Sheena Panthaplackel, Pengyu Nie, Milos Gligoric, Junyi Jessy Li, and Raymond Mooney. 2020. Learning to update natural language comments based on code changes. In Annual Meeting of the Association for Computational Linguistics, pages 1853–1868.

F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel,

- B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. 2011. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12:2825–2830.

Gabriel Poesia, Oleksandr Polozov, Vu Le, Ashish Tiwari, Gustavo Soares, Christopher Meek, and Sumit Gulwani. 2022. Synchromesh: Reliable code generation from pre-trained language models. Preprint, arXiv:2201.11227.

Ruchir Puri, David S Kung, Geert Janssen, Wei Zhang, Giacomo Domeniconi, Vladimir Zolotov, Julian Dolby, Jie Chen, Mihir Choudhury, Lindsey Decker, Veronika Thost, Luca Buratti, Saurabh Pujar, Shyam Ramji, Ulrich Finkler, Susan Malaika, and Frederick Reiss. 2021. CodeNet: A large-scale AI for code dataset for learning a diversity of coding tasks. In Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Andrew M Saxe, Yamini Bansal, Joel Dapello, Madhu Advani, Artemy Kolchinsky, Brendan D Tracey, and David D Cox. 2019. On the information bottleneck theory of deep learning. Journal of Statistical Mechanics: Theory and Experiment, 2019(12):124020.

Craig W. Schmidt, Varshini Reddy, Chris Tanner, and Yuval Pinter. 2025. Boundless byte pair encoding: Breaking the pre-tokenization barrier. Preprint, arXiv:2504.00178.

Torsten Scholak, Nathan Schucher, and Dzmitry Bahdanau. 2021. PICARD: Parsing incrementally for constrained auto-regressive decoding from language models. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 9895–9901, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Mike Schuster and Kaisuke Nakajima. 2012. Japanese and korean voice search. In International Conference on Acoustics, Speech and Signal Processing, pages 5149–5152.

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural machine translation of rare words with subword units. In Annual Meeting of the Association for Computational Linguistics, pages 1715–1725.

Michele Tufano, Jevgenija Pantiuchina, Cody Watson, Gabriele Bavota, and Denys Poshyvanyk. 2019. On learning meaningful code changes via neural machine translation. In International Conference on Software Engineering, pages 25–36.

Shubham Ugare, Tarun Suresh, Hangoo Kang, Sasa Misailovic, and Gagandeep Singh. 2024. Syncode: Llm generation with grammar augmentation. Preprint, arXiv:2403.01632.

Dixuan Wang, Yanda Li, Junyuan Jiang, Zepeng Ding, Ziqin Luo, Guochao Jiang, Jiaqing Liang, and Deqing Yang. 2025. Tokenization matters! degrading large language models through challenging their tokenization. Preprint, arXiv:2405.17067.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, and 3 others. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Bing Yan, Angelica Chen, and Kyunghyun Cho. 2025. Inconsistency of llms in molecular representations. Digital Discovery.

John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik R Narasimhan, and Ofir Press. 2024. SWE-agent: Agent-computer interfaces enable automated software engineering. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Brian Siyuan Zheng, Alisa Liu, Orevaoghene Ahia, Jonathan Hayase, Yejin Choi, and Noah A. Smith. 2025. Broken tokens? your language model can secretly handle non-canonical tokenizations. Preprint, arXiv:2506.19004.

### A Use of LLMs

We used an LLM-based writing assistant to polish grammar. All ideas, analyses, experiments, and scientific claims are our own, and we take full responsibility for the content of this work.

### B Additional Background: Tokenizer Differences Between LLMs

Figure 6 shows the heatmap of vocabulary distances between tokenizers, which includes 19 popular open-source (coding) LLMs from 8 model families. Notably, most LLMs adopt a pre-tokenization strategy that splits text into linguistically and layout-meaningful chunks before a byte-level BPE. While details vary by family, common choices include isolating short digit runs (often 1–3; Qwen and some DeepSeek variants prefer per-digit), treating contiguous letters with combining marks as words, splitting punctuation and symbol runs (sometimes with an optional leading space), and separating newline blocks and longer space runs. Non-Latin scripts such as Han/Hiragana/Katakana (and in some cases Hangul) are taken as contiguous spans. Family differences that matter for our study include LLaMA-3 explicitly detaching English clitics, CodeQwen-1.5 disabling pre-tokenization (leaving underscores and long ASCII spans intact), DeepSeek-Coder using code-oriented splits (letters, punctuation, newlines, CJK, digits), and DeepSeekV3/LLaMA-4/GPT-OSS converging on a similar unified scheme. In practice, more aggressive presegmentation tends to make models tolerant to superficial spacing around symbols but sensitive to numeric chunk boundaries, whereas byte-only or lightly pre-segmented designs make underscore and identifier edits more likely to introduce new token boundaries.

### C Additional Experimental Methodology

#### C.1 Benchmarks Normalization

To ensure that our semantic-preserving naming/spacing rewrite rules (Section 3.3) do not spuriously break compilation or tests, we perform a lightweight normalization pass before evaluation.

For the bug fixing and code summarization tasks from HumanEvalPack (Muennighoff et al.,

- 2023), we first canonicalize Java identifier style to camelCase from snake_case2, then propagate

2HumanEvalPack (Muennighoff et al., 2023) translates the HumanEval (Chen et al., 2021) benchmark from Python

any renamings consistently to tests, entry points, and declarations to preserve their functionalities.

For the code translation tasks, we start from the Avatar and CodeNet benchmarks prepared by Pan et al. (2024), following their task definitions and tests. We fixed some samples with harness-compatibility issues that would otherwise cause false negatives and prune a small number of unsalvageable or pathological samples (e.g., extremely long inputs or cases that time out), without changing the underlying problem semantics. And finally, we dropped 6 python2java tasks and 4 java2python tasks in Avatar that we could not fix.

The most common adjustments fall into a few categories: (i) IO/formatting normalization. For example, we replace non-portable characters such as U+FFFD or segmentation markers like U+2581 with ASCII equivalents; ensure consistent tokenization by splitting on spaces instead of empty strings; remove trailing spaces/newlines; standardize numeric output with Java DecimalFormat or Python f-strings to fixed precision; (ii) test correctness fixes where expected outputs were inconsistent with the reference implementation or ordering; and (iii) minimal code-context edits that preserve semantics but align with tests (e.g., renaming helper methods where tokenizer-specific splits would otherwise occur, adding @Override annotations, or make Scanner/FastScanner usage consistent). All edits are specified once, applied uniformly to baseline and variant inputs, and never conditioned on model outputs.

#### C.2 Rewrite Algorithms

To mutatively rewrite a code context on naming, we first parse it to obtain a code token index and two identifier sets: (i) immutable identifiers derived from configured immutable types (e.g., Java: importDeclaration, methodCall; Python: import_as_name, trailer); (ii) declaration identifiers that are safe to rename (excluding Java methods annotated with @Override). We restrict candidates by casing using regexes, specifically, snake case matches [a-z0-9]+(?:_[A-Za-z0-9]+)+ and camel case identifier matches the regex [a-z]+(?:[A-Z]+[A-Za-z0-9]+[A-Za-z0-9]*)+.

For each eligible identifier, we segment its lexeme by a well designed regex, convert from the source to the target case, and record the absolute character positions in the original string where underscores

to other PLs (including Java), but all the identifiers were remained in snake_case regardless of the target PL.

[Figure 12]

Figure 6: Heatmap of vocabulary distances between tokenizers (Full ver.).

would be inserted or removed (edit events). For HumanEval tasks, we additionally propagate the same renamings to tests, entry points, and declarations to keep the harness consistent, and these are treated as optional ancillary patches and do not alter the core algorithm. The immutable/declaration settings aim to maximize safe coverage while preserving compilation and test pass behavior.

Spacing rewrite follows the same structure but, instead of changing identifier lexemes, we insert exactly one space between adjacent tokens when their kinds match a configured token-type bigram (former, latter) from Table 3. For each match, we insert whitespace at the boundary between the two tokens, record an insertion event at that position, and update offsets.

Conceptually, although rewrites are defined over PL tokens, the notion of a fragment uses LLM tokens. For each rewrite site, we consider the minimal contiguous list of LLM tokens that covers

the affected PL tokens (identifiers for naming and the two code tokens of each combination for spacing) as the fragment. Our fragment-change classification is based on an analysis of all fragments’ transformation in a code context. Specifically, a merge occurs when at least one old LLM token boundary inside those spans disappears, and a split occurs when at least one new boundary appears after rewriting. To detect and analyze all LLM token boundary transformations, we compute LLM token start positions before and after rewriting with the same LLM tokenizer. We ignore boundaries created exactly at an edit site between two code tokens or those created right next to the edit site within one code token. Meaning that for insertions, we disregard any boundary introduced by the inserted whitespace between two code tokens that were fully or partially combined into one LLM token, as well as those caused by standalone underscores immediately to its right (a behavior commonly observed in

the DeepSeek-Coder or CodeQwen-1.5 tokenizer, where underscores are usually treated as a single token), as encoded by the various edit masks classified by edit types in Algorithm 3. The loop in Algorithm 3 shifts the original boundary set by the cumulative δ pre edit to align coordinate systems, builds the masks for shift edits and edit-adjacent positions for insert operation, and then compares adjusted old versus filtered new starts. Specifically, let Sold and Snew be the sets of LLM token starts before and after rewriting, and after masking specific edit sites, we compute A=Sold \ Snew (old boundaries lost) and B=Snew \ Sold (new boundaries gained). The label is unchanged if A=∅ and B=∅, merged if A̸=∅ and B=∅, split if A=∅ and B̸=∅, and mixed otherwise.

#### C.3 Metrics Computation Algorithm

We evaluate on two input programming language subsets for accuracy and ∆accuracy, Xp (Python inputs) and Xj (Java inputs), where their union X =Xj ∪Xp with |X|=1546. For a fixed rewrite rule wi and model m, let Ti be the deterministic transformation that applies wi to an input x ∈ X, and we define T0 as no rule would be applied on the input. And we let W={i|i = 0,1,··· ,24} denote the assignment set for all rules where i=0 is the baseline, i>0 means the variant of applying rule wi. Running the model yields code fm(Ti(x)), which the harness evaluates on the test set T (x). We define the test-level pass fraction

rm,i(x) ≜

1 |T (x)|

[[fm(Ti(x))]]t,

t∈T (x)

where [[fm(Ti(x))]]t ∈ {0,1} denotes the execution result of program fm(Ti(x)) from test t (Guan et al., 2025). Follow that we define the task-level correctness indicator Ym,i(x) ≜ I{rm,i(x) = 1} ∈ {0,1}. So the accuracy of a rule assignment i ∈ W on a set S ∈ {Xp,Xj} is

1 |S| x∈S

Accuracyi(m;S) =

Ym,i(x).

We report ∆accuracy as ∆accuracyi(m;S) ≜ Accuracyi(m;S) − Accuracy0(m;S), where i ∈ W and i ̸= 0. Not all inputs would be modified by a given rule, we therefore define the actually-affected subset

Xi′ ≜ {x ∈ X : Ti(x) ̸= x},

whose summed sizes for all rules classified by model series are shown in Table 8. Then our proposed sensitivity measures how often correctness flips among affected inputs only by

1 |X′

Sensitivityi(m) ≜

i|

x∈Xi′

Ym,i(x)−Ym,0(x) .

Intuitively, ∆accuracy captures net gains/losses which may cancel when aggregating, whereas sensitivity isolates the flip rate on inputs whose tokens were actually changed by wi.

#### C.4 Experimental Environment

We conduct all experiments on an NVIDIA H100 GPU cluster, consuming approximately 1840 GPUhours in total across runs. All model checkpoints are obtained from the Hugging Face Hub and loaded with the Hugging Face Transformers library (v4.53.2) (Wolf et al., 2020). Unless otherwise stated, models are executed in fp32, the only exceptions are Llama-3.3-70B-Instruct, Qwen2.5-Coder-32B-Instruct, and deepseek-coder33b-instruct, which we run in fp16. All evaluations use the bigcode-evaluation-harness framework (Ben Allal et al., 2022) with its standard protocols. We use deterministic decoding without sampling and a batch size of 1 throughout. All tests are executed with Java 21.0.1 and Python 3.8. The maximum generation length is set to 1,024 tokens for HumanEvalPack and Avatar tasks, and 2,048 tokens for CodeNet tasks. For t-SNE visualizations, we use scikit-learn v1.7.1 (sklearn.manifold.TSNE) with perplexity to 70 and use the Barnes–Hut method with 1000 iterations, PCA initialization, learning_rate=’auto’, and n_jobs=16 (Pedregosa et al., 2011).

### D Additional Results and Analysis

In Figures 7 and 8, the line plots summarize sensitivity for each rewrite rule. In Figure 9, comparing samples with and without identifier fragment change shows the overall trend of sensitivity on different model size. The per-series breakdowns in Figures 10 to 12 echo this pattern across Llama-3, Qwen2.5-Coder, and DeepSeek-Coder, while Llama tends to be more sensitive overall, all families exhibit a variation in sensitivity between "changed" and "unchanged" groups. Figure 13 shows the distribution of ∆accuracy per rewrite rule. Compared to sensitivity, ∆accuracy

- Algorithm 1 Naming Rewrite

Input: C: code context, P: code parser, Itypes: immutable identifier types, ρsrc: source case regex, tgt:

target case, (optional) ExtraPatches: extra patches. Output: C′, E, (optional) ExtraPatches′.

// TokIdx: list of (x,τ,[i,j)) where x=code token, τ=token kind, [i,j)=char span

- 1: (TokIdx,Sim,Sdec) ← INDEX(C,P,Itypes)
- 2: E ← [], R ← ∅, C′ ← C, O ← 0 // E: list of edit underscore events (pos,δ); O: total offset
- 3:
- 4: for (x,τ,[i,j)) ∈ TokIdx in ascending i do

- 5: if τ = id ∧ (x ∈/ Sim ∨ x ∈ Sdec) ∧ REGEXCHECK(x,ρsrc) then

- 6: y ← CASECONV(x,tgt) // rewrite the identifier to the target case
- 7: ∆list ← DIFFUNDERLINEPOS(x,y,i) // return a list of add/del underscore events (pos,δ)
- 8: E ← APPEND E, ∆list)
- 9: R[x] ← y
- 10: C′ ← CONCAT(C′[0:(i+O)], y, C′[(i+O+|x|):|C′|]) // string concatenation
- 11: O ← O + (|y| − |x|)
- 12:
- 13: ExtraPatches′ ← APPLYREWRITES(ExtraPatches,R)
- 14: return (C′, E, ExtraPatches′)

may not be ideal for quantifying robustness because gains and losses cancel and many samples are unaffected. Table 8 reports, for each rewrite rule, the number of benchmark samples actually modified, stratified by model series. Table 9 provides the full breakdown of sensitivity by fragment-change category. Together, these tables clarify both the scope of input perturbations and the source of robustness differences observed in the main results.

- Algorithm 2 Spacing Rewrite

Input: C: code context, P: code parser, (Kf,Kℓ): token-type bigram. Output: C′, E.

// TokIdx: list of (x,τ,[i,j)) where x=code token, τ=token kind, [i,j)=char span

- 1: TokIdx ← INDEX(C,P)
- 2: E ← [], C′ ← C, O ← 0 // E: list of insert events (pos,+1); O: total offset
- 3:
- 4: for k ← 0 to |TokIdx| − 2 do

- 5: (xf,τf,[if,jf)) ← TokIdx[k]; (xℓ,τℓ,[iℓ,jℓ)) ← TokIdx[k+1]
- 6: if MATCH(τf,Kf) ∧ MATCH(τℓ,Kℓ) then

- 7: E ← APPEND(E, (iℓ,+1))
- 8: C′ ← CONCAT C′[0:(jf+O)], " ", C′[(iℓ+O):|C′|] // insert one space

- 9: O ← O + 1
- 10:
- 11: return (C′, E)

Table 8: Samples that been modified by rewrite rule, broken down by model series.

Changed

Rewrite Rule Model Series Total Unchanged

All Merged Split Mixed

Llama-3 2238 1292 946 105 767 74 Qwen2.5-Coder 2238 1292 946 105 767 74 deepseek-coder 2247 999 1248 123 996 129 CodeQwen1.5 2247 954 1293 136 1043 114

Naming

Llama-3 12804 9315 3489 660 2394 435 Qwen2.5-Coder 12804 9315 3489 660 2391 438 deepseek-coder 12804 8381 4423 608 3091 724 CodeQwen1.5 12804 8720 4084 725 2504 855

Spacing

[Figure 13]

Figure 7: Percentage difference for naming rewrite transformations.

- Algorithm 3 Fragment-Change Classification (CLASSIFY)

Input: C: original code context, C′: new code context, E: list of edit events (pos,δ) with δ ∈ {±1},

EditType: edit type, T: LLM tokenizer.

Output: type ∈ {unchanged, merged, split, mixed}

- 1: Lold ← POSLLMTOKENS(C,T) // cumulative first character positions of LLM tokens in C
- 2: Lnew ← POSLLMTOKENS(C′,T)
- 3: Sold ← SET(Lold), Snew ← SET(Lnew), Sed ← {pos | (pos,δ) ∈ E }, Sed+ ← ∅
- 4: O ← 0 // cumulative offset from prior edits
- 5:
- 6: for each (pos,δ) in E do

- 7: a ← pos + O // adjusted position of this edit
- 8: Sold ← {p+δ if p > a else p | p ∈ Sold }
- 9: Sed ← {e+δ if e > a else e | e ∈ Sed }
- 10: Sed+ ← Sed+ ∪ {a + max(δ,0)}
- 11: O ← O + δ
- 12: if EditType = underscore then

- 13: Snew ← Snew \ (Sed+ \ Sed) // ignore starts next-to inserted standalone underscore edit boundaries
- 14: else if EditType = whitespace then

- 15: Snew ← Snew \ (Sed \ Sold) // ignore new starts created at whitespace edit boundaries
- 16:
- 17: A ← Sold \ Snew // A: lost tokens after rewrite (some tokens merged)
- 18: B ← Snew \ Sold // B: gained tokens after rewrite (some tokens split)
- 19:
- 20: if A ̸= ∅ and B = ∅ then

- 21: return merged
- 22: else if A = ∅ and B ̸= ∅ then

- 23: return split
- 24: else if A ̸= ∅ and B ̸= ∅ then

- 25: return mixed
- 26: else

- 27: return unchanged

[Figure 14]

Figure 8: Percentage difference for spacing rewrite transformations.

Table 9: Impact of different types of fragment change on sensitivity (Full ver.).

Changed (subcategories)

Rewrite Rule Model Total Unchanged Changed (all)

Merged Split Mixed

|Llama-S Llama-M Llama-L<br><br>|11.48 10.68<br><br>9.43|10.68 12.58 9.44 12.37 8.13 11.21<br><br>|10.48 13.17 9.46<br><br>8.57 13.30 8.11<br>9.52 11.73 8.11<br>|
|---|---|---|---|
|Qwen-S Qwen-M Qwen-L<br><br>|7.73 7.95 8.27|6.97 8.77<br><br>7.35 8.77<br><br><br>6.58 10.57<br><br>|8.57 9.00 6.76 5.71 9.00 10.81<br><br>11.43 10.82 6.76|
|DS-S DS-M DS-L|9.88 8.95 8.95<br><br>|8.31 11.14 7.91 9.78 6.61 10.82|4.88 11.95 10.85 7.32 10.54 6.20<br><br>10.57 10.64 12.40|

Naming

|Llama-S Llama-M Llama-L|10.22 10.99<br><br>8.51|9.32 12.61 9.69 14.45 7.24 11.89<br><br>|11.06 13.37 10.80 13.03 14.83 14.48 10.00 11.53 16.78<br><br>|
|---|---|---|---|
|Qwen-S Qwen-M Qwen-L|7.07 8.87 5.71<br><br>|6.04 9.80<br><br>7.53 12.47<br><br><br>5.09 7.37<br><br>|8.33 9.62 13.01 12.42 10.71 22.15 7.42 6.48 12.10|
|DS-S DS-M DS-L<br><br>|8.36 8.71 6.26|7.25 10.47 7.58 10.85 5.80 7.12<br><br>|8.72 10.45 12.02 10.36 10.19 14.09 6.41 6.44 10.64|

Spacing

[Figure 15]

- Figure 9: Naming rewrite rules percentage difference (with or without fragment change).

[Figure 16]

- Figure 10: (Llama series) Spacing rewrite rules percentage difference (with or without fragment change).

[Figure 17]

- Figure 11: (Qwen series) Spacing rewrite rules percentage difference (with or without fragment change).

[Figure 18]

- Figure 12: (Deepseek series) Spacing rewrite rules percentage difference (with or without fragment change).

[Figure 19]

- Figure 13: Distribution of ∆accuracy per rewrite rule across models and benchmarks.

