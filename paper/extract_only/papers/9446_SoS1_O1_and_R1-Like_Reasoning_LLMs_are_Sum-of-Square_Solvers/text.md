# arXiv:2502.20545v1[cs.LG]27Feb2025

## SoS1: O1 and R1-Like Reasoning LLMs are Sum-of-Square Solvers

Kechen Li1 Wenqi Zhu3 Coralia Cartis3 Tianbo Ji2 Shiwei Liu3 1Nanjing University of Aeronautics and Astronautics 2School of Transportation and Civil Engineering, Nantong University 3Mathematical Institute, University of Oxford kechenli@nuaa.edu.cn, jitianbo@ntu.edu.cn {wenqi.zhu, cartis, shiwei.liu}@maths.ox.ac.uk

### Abstract

Large Language Models (LLMs) have achieved human-level proficiency across diverse tasks, but their ability to perform rigorous mathematical problem solving remains an open challenge. In this work, we investigate a fundamental yet computationally intractable problem: determining whether a given multivariate polynomial is nonnegative. This problem, closely related to Hilbert’s Seventeenth Problem, plays a crucial role in global polynomial optimization and has applications in various fields. First, we introduce SoS-1K, a meticulously curated dataset of approximately 1,000 polynomials, along with expert-designed reasoning instructions based on five progressively challenging criteria. Evaluating multiple state-of-the-art LLMs, we find that without structured guidance, all models perform only slightly above the random guess baseline (50%). However, high-quality reasoning instructions significantly improve accuracy—boosting performance up to 81%. Furthermore, our 7B model, SoS-7B, fine-tuned on SoS-1K for just 4 hours, outperforms the 671B DeepSeek-V3 and GPT-4o-mini in accuracy while only requiring 1.8% and 5% of the computation time needed for letters, respectively. Our findings highlight the potential of LLMs to push the boundaries of mathematical reasoning and tackle NP-hard problems. Code is available at https://github.com/Joe-2002/SoS1.

### 1 Introduction

With Large Language Models (LLMs) reaching human-level proficiency across a diverse range of tasks (Brown et al., 2020; Singhal et al., 2023; Cai

et al., 2023; Yoshikawa et al., 2023), their ability to reason has emerged as a central topic of interest (Wei et al., 2022; Huang and Chang, 2022). Among these, mathematical reasoning stands out as one of the most rigorous and demanding (Kant, 1908; Hendrycks et al., 2021; Ahn et al., 2024; Liu et al.,

- 2024). As a result, the ability of LLMs to solve research-level mathematical problems is not only a critical benchmark for evaluating their reasoning capabilities but also has the potential to transform mathematical research and practice.

Demonstrated by the success of OpenAI o1 (OpenAI, 2024) and DeepSeek-R1 (Guo et al.,

- 2025), test-time scaling has emerged as a promising technique for enhancing LLMs’ performance in mathematical reasoning (Snell et al., 2024a; Welleck et al., 2024). This approach involves prompting LLMs to generate more reasoning steps, either sequentially (Snell et al., 2024b; Hou et al., 2025; Lee et al., 2025) or in parallel (Brown et al., 2024; Xin et al., 2024), to increase the accuracy of their final answers. However, the community’s primary focus has been limited to relatively simple levels of mathematics, ranging from high school and Olympiad-level problems to early undergraduate topics (of America, 2024; Hendrycks et al., 2021). Whether the promise of test-time scaling extends to research-level mathematics remains an open question.

In this paper, we investigate a fundamental yet formally well-posed problem in mathematics: determining whether a given multivariate polynomial is nonnegative. This question is closely related to

Hilbert’s Seventeenth Problem, which was posed by David Hilbert in 1900 as part of his famous 23 problems presented at the International Congress of Mathematicians (ICM) (Hilbert, 1893), and it remains central to global polynomial optimization. Many key challenges in applied and computational mathematics can be reframed as deciding the nonnegativity of certain polynomials including control theory (Parrilo, 2000), quantum computation (Doherty et al., 2002), polynomial games (Gvozdenovi´c and Laurent, 2007), tensor methods (Zhu and Cartis, 2024; Ahmadi et al., 2023a) and combinatorial optimization (Gvozdenovi´c and Laurent, 2007).

Testing whether a general polynomial is nonnegative is provably NP-hard, even for polynomials of relatively low degrees or with a small number of variables. People usually seek special cases of polynomials where the challenging nonnegativity constraints can be replaced with more manageable conditions. For instance, the sum of squares (SoS) condition, a mathematical technique in polynomial optimization where a polynomial is expressed as a sum of squared polynomials, provides a sufficient criterion for polynomial nonnegativity. Classical solvers, such as SoSTOOLS, YALMIP, and Gloptipoly have been developed to verify these SoS conditions (Prajna et al., 2002). However, a significant limitation of these approaches lies in the typically large size of the resulting semidefinite programming (SDP) problem. Specifically, for a polynomial with n variables and a degree of 2d, the SDP’s dimension is given by N = n+22dd , making it challenging to scale this approach to larger problems.

To evaluate if state-of-the-art (SOTA) reasoning LLMs like Openai o1 and DeepSeek-R1 can solve large-scale SoS programming problems, we introduce SoS-1K, a meticulously curated dataset of approximately 1,000 polynomials—along with expert-designed, SoS-specialized, reasoningguiding instructions based on five progressively

challenging criteria: polynomial degree, nonnegativity of the leading search direction, identification of special structures, assessment of square-form expressions, and matrix decomposition into the quadratic form of monomials. Our comprehensive evaluation of multiple SOTA LLMs, including DeepSeek-R1, DeepSeek-V3, GPT-4o, OpenAI o1-mini, Qwen2.5 series, and QwQ-32B-Preview, demonstrate the following interesting findings:

- • When presented with a plain question, all SOTA LLMs bluntly fail to solve SoS with most models achieving only around 60% accuracy, just slightly above the random guess baseline of 50%.1
- • When prompted with high-quality reasoning traces, we consistently observe a significant accuracy boost across all models, up to 21%. And models perform better with higher-quality reasoning traces.
- • Reasoning-focused LLMs generally outperform general-purpose LLMs, regardless of prompt quality.
- • Higher-capacity models require fewer thinking tokens to make correct predictions, while lowercapacity models need more reasoning steps to achieve their optimal performance.

We further demonstrate that supervised finetuning (SFT) of a pre-trained 7B model on SoS1K for just 4 hours using 2 A100 GPUs significantly improves accuracy from 54% to 70% with significantly faster response times. Specifically, SoS-7B requires only 1.8% and 5% of the computation time needed for DeepSeek-V3 and GPT4o-mini, respectively. The resulting SoS-7B surpasses much larger models, including the 671B DeepSeek-V3 and GPT-4o-mini. More interestingly, when prompted with high-quality reasoning, the models demonstrate an understanding of

1Since SoS is a binary classification problem, random guessing yields 50% accuracy.

###### Question: Can you determine whether the following polynomial is SoS?

59.1 + 30.3𝑥 + 4.8𝑥 –6.7𝑥 + ⋯ + 356.2𝑥 𝑥 − 548.4𝑥 𝑥 + 993.7𝑥 − 835.9𝑥 𝑥 + 1286.7𝑥

SoS Simple SoS Reasoning

SoS Plain

Step-by-step

Thinking Process

Thinking Process

Iterative Argument

Conclusion: Not SoS

[Figure 1]

Conclusion: Not SoS

- Step 1: Degree Analysis: Highest Degree = 2.
- Step 2: Coefficients of Highest-Degree Terms: some negative
- Step 3: Special Properties: Quadratic Polynomials and Mixed Signs
- Step 4: Check Negative Values: negative linear coefficients implies that the polynomial can take negative values
- Step 5: Direct SOS Representation or Matrix Methods: highly non-trivial and unlikely.

- Step 1: Degree Analysis: Highest Degree = 2.
- Step 2: Coefficients of Highest-Degree Terms: All univariate quadratic terms have positive coefficients.
- Step 3: Check Negative Values: No evidence of p(x) < 0.
- Step 4: Special Properties: ü Applicable Case: Any nonnegative quadratic

|Solution Incorrect × No thinking process × Lacks a systematic<br><br>method for checking SoS|
|---|

[Figure 2]

[Figure 3]

polynomial is SOS. ü Verification: Step 2 and Step 3 indicate the polynomial is nonnegative.

[Figure 4]

[Figure 5]

[Figure 6]

Conclusion: Not SoS Conclusion: SoS

|Solution Correct ü Step-by-step category-based checking method ü Expert-annotated methodology in Steps 2 and 3 ü Proceeds from easy to difficult steps ü Stops immediately at Step 4 when answer found|
|---|

|Solution Incorrect ü Category-based checking process × Simple methodology in Steps 2 and 4 give wrong<br><br>answer × No distinction between easy and hard steps × Executes all steps with no order|
|---|

Figure 1: Demonstration of SoS Plain (left), SoS Simple (mid), and SoS Reasoning (right).

research-level questions. For instance, Qwen2.514B-Instruct-1M leverages the Motzkin polynomial to generate new, previously unseen counterexamples to Hilbert’s 17th problem (Motzkin, 1967). Such examples are nontrivial, as the first counterexample to Hilbert’s 17th problem was discovered 27 years after Hilbert originally posed it (Hilbert, 1893). These findings suggest that LLMs exhibit reasoning patterns, expanding the boundaries of solving NP-hard problems.

2001). The development of numerical solvers began in 2009, with tools like GloptiPoly and SoSTOOLS introduced in (Henrion et al., 2009; Papachristodoulou et al., 2013). More details on the literature review for SoS can be found in Appendix A.

The first attempt to tackle SoS-related mathematical difficulty via AI was presented in (Alfarano et al., 2024). The authors trained Transformers to address a long-standing open problem in mathematics: discovering a Lyapunov function that guarantees the global stability of a dynamical system. The global stability of polynomial Lyapunov systems is closely tied to SoS framework. Their approach was tested on relatively small systems (with at most five equations for polynomial systems) and demonstrated promising results compared to state-of-theart conventional solvers.

Our work serves as a pilot study on leveraging reasoning LLMs for solving SoS problems, paving the way for tackling large-scale research-level questions in mathematics using AI.

#### 1.1 Related Work

The study of SoS and nonnegative polynomials has a rich history spanning over 120 years, with numerous scholars contributing to this field. Classical methods for characterizing an SoS polynomial involve expressing the polynomial as a quadratic form of monomials and then reformulating the problem as a semidefinite program (SDP) (Lasserre, 2000, 2001). Classic techniques, such as the Lasserre hierarchy, date back to 2001 (Lasserre,

### 2 SoS-1K Dataset

In this section, we first provide the definition of SoS polynomials. We describe the expert-annotated reasoning traces based on five criteria of increasing difficulty in Section 2.1, which lead to the generation of the SoS-1K dataset, detailed in Section 2.2.

Definition 2.1 SoS polynomial (Ahmadi et al., 2023a; Ahmadi and Parrilo, 2013; Kojima, 2003) A 2d-degree multivariate polynomial q(x) : Rn → R where x = [x1,...,xn] ∈ Rn is a sum of squares (SoS) if there exist polynomials q˜1,...,q˜r : Rn → R, for some r ∈ N, such that q(x) =

r j=1 q˜j(x)2 for all x ∈ Rn (Ahmadi et al., 2023a,

Def. 1). Further details on the theory for SoS polynomials can be found in Appendix B.

- 2.1 Expert-Designed Reasoning Instructions for SoS

Instructions play a crucial role in guiding LLMs toward better reasoning and problem-solving (Zhang et al., 2023). The way an LLM processes a problem can be significantly improved with carefully crafted instructions that provide structure, constraints, and logical flow.

To evaluate the capacity of SOTA LLMs on SoS reasoning, we create three sets of reasoningguiding instructions with increasing quality that can be applied across multiple SoS problem types: (1) plain question (SoS Plain); (2) simple SoS instruction (SoS Simple); (3) reasoning-guiding SoS instruction (SoS Reasoning).

SoS Plain simply asks LLMs: “Please analyze if this polynomial can be expressed as a Sum of Squares (SOS)”.

SoS Simple classifies SoS polynomials into five distinct groups, each defined by a concise, one-line criterion. The full instruction set for SoS Simple contains 78 words and 647 characters, with complete details provided in Appendix C.

SoS Reasoning is a structured five-step framework designed to identify SoS polynomials. Unlike SoS Simple, which provides only basic classification criteria, SoS Reasoning encourages the model follow a step by step mathematical verification process. The framework introduces progressively more detailed reasoning steps to guide the model in verifying whether a polynomial is SoS.

Specifically, we provide a logical reasoning trace based on proofs and theorems, offering necessary and sufficient conditions for identifying SoS polynomials. A large number of positive and negative examples accompany each set of theorems, helping the model recognize special structures, symmetries, and mathematical forms inherent to SoS polynomials. Additionally, SoS Reasoning introduces intermediate steps and incorporates key and challenging reasoning processes, such as the Q matrix and the squared form ps, enabling deeper reasoning and iterative refinement.

Below is an illustration of SoS Reasoning. The full version is provided in Appendix D.

#### Step 1. Check the Degree: An SoS polynomial

must have an even highest degree.

#### Step 2. Check for Non-negativity: SoS poly-

nomials are nonnegative for all real inputs. We verify this by examining the constant term, the coefficients of the leading term, and performing a grid-based numerical check.

Step 3. Check for Well-known Special Cases: Any nonnegative quadratic polynomial and any nonnegative quartic polynomial in one or two variables is SoS.

#### Step 4. Check for Square Form: By Defini-

tion 2.1, an SoS polynomial can be expressed as: ps(x) = i qi(x)2, where each qi(x) is a polynomial.

#### Step 5. Check for Matrix Decomposition:

Based on Theorem B.1, we express the polynomial as p(x) = y∗⊤Qy∗, where Q is a symmetric matrix2. We then check whether Q is positive semidefinite.

#### 2.2 Construction of SoS-1K Test Set

Building on the above expert design, we construct SoS-1K, a dataset comprising five subsets of polynomials, each corresponding to polynomials filtered out at steps 1–5. Appendix E provides a comprehensive summary of the SoS-1K test set and its

2Q and y∗ are provided in Appendix D

subsets. Since most LLMs struggle with very long polynomials, we have ensured that the majority remain within a length of 4,000. Approximately half of the polynomials are SoS, while the other half are not. The number of variables and the polynomial degree both range from 2 to 10. For all test subsets (except Set 1), we provide two corresponding sets: one containing SoS polynomials and another containing non-SoS polynomials. Each polynomial is labeled as either SoS or non-SoS, with accompanying justifications and difficulty levels. For certain polynomial classes, we provide theoretical proofs confirming their SoS status, while the remaining polynomials are labeled based on results from standard solvers.

### 3 Evaluation of LLMs on SoS

In this section, we compare the performance of SOTA LLMs using SoS Plain, SoS Simple, and SoS Reasoning on a subset of SoS-1K of approximately 340 randomly chosen from all sub-classes of test problems (see Table 3). These test samples are drawn such that the number of samples in each subtest is approximately equal. We test across subclasses and report results per test subclasses. The models evaluated include reasoning-purpose models like DeepSeek-R1, OpenAI o1-mini, and QwQ32B-Preview, as well as general-purpose models such as DeepSeek-V3, GPT-4o, and Qwen2.5 series.

The full results for each model on each test set are summarized in Table 1. We summarize our main observations below:

- OB1. All LLMs fail when presented with a plain question. When given SoS-plain, all LLMs exhibit poor accuracy, ranging from 50% to 60%, with QwQ-32B-Preview being the sole exception, achieving 64% valid accuracy. This result suggests that, despite being trained on vast amounts of mathematical data, SOTA LLMs struggle to solve SoS problems without explicit prompting.
- OB2. LLMs have a significant perfor-

mance boost when prompted with high-quality reasoning-guiding instructions. We consistently observe a substantial accuracy improvement across all models. With SoS Simple, QwQ-32B-Preview achieves 71% accuracy, while DeepSeek-R1 with the highest-level SoS Reasoning reaches the highest accuracy (81%). It suggests that while LLMs may possess the underlying knowledge to solve SoS problems, they require clear and structured instructions to effectively retrieve and apply it.

Furthermore, it is worth mentioning that SoS Reasoning plays a crucial role in achieving these improvements. The improvement from SoS Simple over the baseline (SoS Plain) is relatively small, averaging 5% increase in the accuracy of valid samples and 6% increase in the accuracy of valid samples, whereas SoS Reasoning improves performance by 17% and 11%, respectively over the same baseline.

- OB3. Reasoning-purpose LLMs benefit more from high-quality instructions than generalpurpose LLMs. Overall, reasoning-focused LLMs such as DeepSeek-R1, OpenAI o1-mini, and QwQ-32B-Preview achieve a higher average accuracy (79.0%) compared to general-purpose LLMs (72.9%). This result suggests that stronger reasoning capabilities contribute to improved performance in solving SoS problems.
- OB4. Many LLMs struggle to consistently provide valid answers to SoS questions. Most LLMs fail to consistently provide valid answers, often encountering timeout issues. Nevertheless, GPT-4o, DeepSeek-V3, and o1-mini demonstrate robustness in this regard, consistently producing effective and reliable answers. 4 Further Analysis

In this section, we outline several research questions that the authors find particularly intriguing. Q1: Does the model follow a truly mathematical step by step verification process?

We find that LLMs are able to generate answers

Model Accuracy on Valid Samples Accuracy on Total Samples Response Time (s) Instruction Type SoS Plain SoS Simple SoS Reasoning SoS Plain SoS Simple SoS Reasoning SoS Plain SoS Simple SoS Reasoning General-purpose LLMs

Qwen2.5-7B-Instruct 55% 61% 76% 52% 59% 62% 22.4 31.2 68.5 Qwen2.5-7B-Instruct-1M 54% 64% 75% 54% 64% 63% 5.6 8.4 35.2 Qwen2.5-14B-Instruct 55% 66% 74% 52% 66% 69% 12.9 23.1 48.3 Qwen2.5-14B-Instruct-1M 56% 60% 74% 56% 59% 67% 12.7 20.7 52.7 Qwen2.5-32B-Instruct 56% 58% 67% 55% 58% 62% 13.0 18.0 37.4 DeepSeek-V3 54% 60% 70% 54% 60% 69% 29.6 39.8 95.0 GPT-4o-mini 59% 67% 72% 59% 67% 69% 10.8 15.4 53.1 GPT-4o 60% 61% 75% 59% 61% 75% 14.6 16.2 27.8

Reasoning-purpose LLMs QwQ-32B-Preview 64% 71% 79% 44% 54% 52% 105.7 101.8 100.0 DeepSeek-R1 62% 62% 81% 55% 55% 56% 514.5 565.6 492.5 OpenAI o1-mini 58% 61% 77% 57% 61% 76% 8.3 18.1 34.9 Average 57% 63% 75% 54% 60% 65% 68.2 78.0 95.0

Table 1: Accuracy comparison across various SOTA LLMs on a subset of SoS-1K with 340 samples. Results are divided into “Valid Samples” and “Total Samples”, as we found that LLMs sometimes suffer from timeout issues.

that are both logically and mathematically correct, step-by-step, following our SoS Reasoning instruction. For instance, we demonstrate o1-mini’s response in Appendix F where we can observe that the responses are logically and mathematically correct, and the model stops naturally once it derives an answer, rather than blindly going through all possible steps.

- Q2: Can LLMs effectively retrieve critical information from long-context polynomials?

Unlike standard text input, polynomials are complex algebraic expressions consisting of variables, coefficients, exponents, and terms. Thus, it is crucial for LLMs to effectively interpret and extract critical information from such structured formats. Our analysis reveals that while QwQ-32B-Preview struggles with questions exceeding 4K tokens in length, most SOTA LLMs can successfully extract the necessary coefficients from 4K-length polynomials for evaluation, producing correct answers.

- Q3: At which of the Steps 1 through 5 in SoS Reasoning does the accuracy improve?

In Figure 2, we illustrate the accuracy improvement across different test sets for the o1-mini model

under SoS Plain, SoS Simple, and SoS Reasoning.3 We observe that the simplest test set, Test Set 1 (corresponding to Step 1), unsurprisingly achieves 100% accuracy across all prompting methods. For the more challenging test sets, Test Sets 2a, 3.1a, 5.1a–5.4a, we observe a continuous improvement from SoS Plain to SoS Simple and further to SoS Reasoning. This improvement is attributed to Steps 2 and 5 in SoS Reasoning, where a series of mathematical verification methods for non-negativity are introduced, including constant coefficient check, grid evaluation, leading order and dominant terms comparison, finding minima, matrix decomposition, and finding symmetry and translation.

Q4. Will LLMs become lazy (take shortcuts) during reasoning?

Yes, another interesting phenomenon observed under the SoS Reasoning prompt is that the model tends to be lazy in Step 5. Specifically, instead of fully executing Step 5, it often avoids matrix decomposition or semidefinite programming (SDP) due to complexity and instead guesses an answer based on prior steps. This behavior is particularly

3Similar patterns are observed for other models.

TS 3.1b TS 3.2a

TS 3.1a

Prompt Types

TS 3.2b

SoS Simple

SoS Plain

TS 2b

SoS Reasoning

TS 4a

TS 2a

- TS 4b

- TS 5.1a

TS 1

TS 5.4b

TS 5.1b

TS 5.4a

TS 5.2a

TS 5.3b

TS 5.2b

TS 5.3a

- Figure 2: Accuracy of different test sets using o1-mini.

prevalent for long inputs and complex polynomials, such as those in Test Set 5.4a. For simpler problems, reasoning models such as o1-mini (which had the shortest runtime of 17s) and larger models like QwQ-32B-Preview tend to take shortcuts, skipping Step 5 and inferring the answer from earlier, simpler steps. In contrast, DeepSeek-V3 does not take shortcuts and instead spends significantly more time solving all steps properly (40s).

- Q5: How does reasoning length affect accuracy?

Figure 3 shows that higher-capacity models generally require fewer thinking tokens to make correct predictions, whereas lower-capacity models need more reasoning steps to reach optimal performance. For instance, DeepSeek-R1 and o1-mini achieve the highest number of correct predictions with a 1K2K response length, whereas the Qwen2.5 series require 3K–4K tokens to produce correct answers.

- Q6: Do SOTA LLMs have any limitations? Though we demonstrate that SoS Reasoning ef-

Figure 3: Number of correct samples with various response lengths.

Table 2: Accuracy Comparison on SoS Reasoning Benchmark, where "—" denotes the undisclosed model size. Accuracy is measured on full evaluation samples.

Model Size Acc. (%) Closed Source

GPT-4o — 75 o1-mini — 76

Open Source

Qwen2.5-7B-Instruct-1M 7B 63 Qwen2.5-32B-Instruct 32B 62 QwQ-32B-Preview 32B 52 DeepSeek-V3 671B 69 DeepSeek-R1 671B 56 SoS-7B (Ours) 7B 70

in cases where the quadratic form of the polynomial involves a low-rank matrix decomposition.

fectively improves accuracy, it is subject to the following limitations. Firstly, for long input cases, invalid samples occur. For example, in DeepSeekR1, only 234 out of 340 samples were valid. Secondly, when handling complex problems, "taking shortcuts" may save time; however, stopping prematurely at difficult steps and guessing an answer can negatively impact prediction accuracy. Thirdly, while these LLMs excel on small-sized polynomials (achieving accuracy close to 90%), they struggle

### 5 Performing SFT on SoS-1K

We further conduct supervised fine-tuning (SFT) with Qwen2.5-7B-Instruct-1M on SoS-1K using LLaMA-Factory (Zheng et al., 2024). The training process was performed on 2× NVIDIA A100 GPUs for 4 hours. The resulting model, SoS-7B, establishes a SOTA total accuracy of 70%, outperforming 671B DeepSeek-V3 (69%), while requiring only 1.8 seconds response time compared to DeepSeek-V3’s 100 seconds. While o1-mini achieves a higher accuracy (75%), it is acceptable as our model is only trained with 1K dataset and enjoys a much faster response time, i.e., 1.8s vs 34.9s.

### 6 Model’s Understanding of SoS and Nonnegativity

One might ask: Did the model merely learn to classify, or has it truly developed the ability to “think” and “construct” new proofs and examples? When faced with research questions in SoS or polynomial optimization, can the model generate mathematically meaningful insights?

To explore this, we designed a series of researchdriven questions (Appendix H) to test the model’s ability to understand the mathematical concepts behind Sum of Squares (SoS) and nonnegativity properties. Finding nonnegative polynomials that are not Sum of Squares (NNSoS) is a fundamental and ongoing research problem in real algebraic geometry and polynomial optimization (Ahmadi et al., 2023b, 2024; Ahmadi and Zhang, 2022). This problem is closely connected to Hilbert’s 17th problem, semidefinite programming (SDP), and positivity certificates in polynomial optimization.

We test the model’s ability to create and analyze unseen mathematical examples. We asked the following question to Qwen-7B-1M and Qwen14B-1M: Can you provide a new NNSoS that was never found in the literature?

Interestingly, when prompted with SoS plain, Qwen-14B-1M can only give examples that are well-known in the literature and Qwen-7B-1M returned an incorrect example:

pa(x) = x41 + x42 + 1 − x21 − x22 − x21x22.

Although this example is incorrect, it is nontrivial, as classic solvers such as YALMIP also fail to extract global optimality4. The reason this example is challenging is that it has four global minima at (1,1), (1,−1), (−1,1), and (−1,−1), with a global minimum value of 0. The number of global minima exceeds the rank of the moment matrix, making it difficult to extract global optimality certificates.

4YALMIP returns status 0, indicating that although the SDP is solvable, rank conditions cannot ensure global optimality.

However, when the same research question was posed to the model with SoS reasoning, the model correctly identified that pa is not a valid solution to our question. This improvement can be attributed to Step 4 of SoS Reasoning (Appendix D), where the trained model recognized that pa(x) is a nonnegative quartic polynomial in two variables and, therefore, cannot be NNSoS.

Moreover, using SoS reasoning, Qwen-14B-1M derived a new valid example for NNSoS,

##### qa(x) = x41x22x23 + x21x42x23 + x43 + 1 − 3x21x22x23.

We cross-checked this polynomial using the classic solver YALMIP and confirmed that qa is indeed a NNSoS. The trained model’s approach to constructing this example is particularly interesting. It began with the well-known example of NNSoS such as pm(x) = x41x22+x21x42+1−3x21x22, Then, the model then introduced a new variable and slightly modified the coefficients to generate qa. This demonstrates that the trained model not only recognizes existing patterns in polynomial optimization but also generalizes and constructs novel cases, providing valuable mathematical insights.

### 7 Conclusion and Discussion

This paper investigates the capacity of LLMs on a research-level mathematical problem: determining whether a given multivariate polynomial is SoS. This problem, closely related to Hilbert’s Seventeenth Problem, plays a crucial role in various fields. We first introduce SoS-1K, a dataset of approximately 1,000 polynomials, along with a set of SoSspecific reasoning-guiding instructions. Our results show that with our expert-designed, current SOTA LLMs is able to achieve up to 81% accuracy. Furthermore, an in-depth analysis of model responses reveals several intriguing insights. This study highlights the potential of AI in tackling large-scale open problems in mathematics, paving the way for future advancements.

### 8 Limitation and Potential Risks

One limitation of this study is that we constrain the context length of SoS question within 4K, as some LLMs tend to fail with longer sequence length. Due to this reason, most SoS polynomials are within the capacity of traditional solvers. In the future, we will extend our dataset, targeting more challenging questions with a scale that is beyond traditional SoS solvers. One potential risk of this study is that LLMs have the chance to make wrong decisions, which might be misleading, and therefore we need to use traditional solvers to verify.

### Acknowledgments

This work was supported by Hong Kong Innovation and Technology Commission (InnoHK Project CIMDA). S. Liu is supported by the Royal Society with the Newton International Fellowship.

### References

Amir Ali Ahmadi, Grigoriy Blekherman, and Pablo A Parrilo. 2024. Convex ternary quartics are sosconvex. arXiv preprint arXiv:2404.14440.

Amir Ali Ahmadi, Abraar Chaudhry, and Jeffrey Zhang. 2023a. Higher-order newton methods with polynomial work per iteration. arXiv preprint arXiv:2311.06374.

Amir Ali Ahmadi, Cemil Dibek, and Georgina Hall. 2023b. Sums of separable and quadratic polynomials. Mathematics of Operations Research, 48(3):1316– 1343.

Amir Ali Ahmadi and Anirudha Majumdar. 2019. Dsos and sdsos optimization: more tractable alternatives to sum of squares and semidefinite optimization. SIAM Journal on Applied Algebra and Geometry, 3(2):193– 230.

Amir Ali Ahmadi, Alex Olshevsky, Pablo A Parrilo, and John N Tsitsiklis. 2013. Np-hardness of deciding convexity of quartic polynomials and related problems. Mathematical programming, 137:453–476.

Amir Ali Ahmadi and Pablo A Parrilo. 2013. A complete characterization of the gap between convexity and sos-convexity. SIAM Journal on Optimization, 23(2):811–833.

Amir Ali Ahmadi and Jeffrey Zhang. 2022. On the complexity of finding a local minimizer of a quadratic function over a polytope. Mathematical Programming, 195(1-2):783–792.

Janice Ahn, Rishu Verma, Renze Lou, Di Liu, Rui Zhang, and Wenpeng Yin. 2024. Large language models for mathematical reasoning: Progresses and challenges. arXiv preprint arXiv:2402.00157.

Alberto Alfarano, François Charton, and Amaury Hayat. 2024. Global lyapunov functions: a long-standing open problem in mathematics, with symbolic transformers. arXiv preprint arXiv:2410.08304.

Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, Christopher Ré, and Azalia Mirhoseini. 2024. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Tianle Cai, Xuezhi Wang, Tengyu Ma, Xinyun Chen, and Denny Zhou. 2023. Large language models as tool makers. arXiv preprint arXiv:2305.17126.

Octavia Camps and Mario Sznaier. 2017. The interplay between big-data and sparsity in systems identification. In Jean-Paul Laumond, Nicolas Mansard, and Jean-Bernard Lasserre, editors, Geometric and Numerical Foundations of Movements, number 117 in Springer Tracts in Advanced Robotics, pages 133– 159. Springer, New York.

Andrew C Doherty, Pablo A Parrilo, and Federico M Spedalieri. 2002. Distinguishing separable and entangled states. Physical Review Letters, 88(18):187904.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Nebojša Gvozdenovi´c and Monique Laurent. 2007. Semidefinite bounds for the stability number of a graph via sums of squares of polynomials. Mathematical Programming, 110:145–173.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.

Didier Henrion, Jean-Bernard Lasserre, and Johan Löfberg. 2009. Gloptipoly 3: moments, optimization and semidefinite programming. Optimization Methods & Software, 24(4-5):761–779.

David Hilbert. 1893. Über ternäre definite formen. Journal.

Zhenyu Hou, Xin Lv, Rui Lu, Jiajie Zhang, Yujiang Li, Zijun Yao, Juanzi Li, Jie Tang, and Yuxiao Dong. 2025. Advancing language model reasoning through reinforcement learning and inference scaling. Preprint, arXiv:2501.11651.

Jie Huang and Kevin Chen-Chuan Chang. 2022. Towards reasoning in large language models: A survey. arXiv preprint arXiv:2212.10403.

Immanuel Kant. 1908. Critique of pure reason. 1781. Modern Classical Philosophers, Cambridge, MA: Houghton Mifflin, pages 370–456.

Masakazu Kojima. 2003. Sums of squares relaxations of polynomial semidefinite programs. Inst. of Technology.

Jean B Lasserre. 2001. Global optimization with polynomials and the problem of moments. SIAM Journal on optimization, 11(3):796–817.

Jean Bernard Lasserre. 2000. Optimisation globale et théorie des moments. Comptes Rendus de l’Académie des Sciences-Series I-Mathematics, 331(11):929–934.

Jean-Bernard Lasserre. 2006. Convergent sdprelaxations in polynomial optimization with sparsity. SIAM Journal on Optimization, 17(3):822–843.

JeanB Lasserre, Kim-Chuan Toh, and Shouguang Yang. 2017. A bounded degree sos hierarchy for polynomial optimization. EURO Journal on Computational Optimization, 5(1-2):87–117.

Kuang-Huei Lee, Ian Fischer, Yueh-Hua Wu, Dave Marwood, Shumeet Baluja, Dale Schuurmans, and Xinyun Chen. 2025. Evolving deeper llm thinking. Preprint, arXiv:2501.09891.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437.

Zhi-Quan Luo and Shuzhong Zhang. 2010. A semidefinite relaxation scheme for multivariate quartic polynomial optimization with quadratic constraints. SIAM Journal on Optimization, 20(4):1716–1736.

Daniel K Molzahn and Ian K Hiskens. 2015. Sparsityexploiting moment-based relaxations of the optimal power flow problem. IEEE Transactions on Power Systems, 30:3168–3180.

Theodore Samuel Motzkin. 1967. The arithmeticgeometric inequality. Inequalities (Proc. Sympos. Wright-Patterson Air Force Base, Ohio, 1965), 205:54.

Katta G Murty and Santosh N Kabadi. 1987. Some npcomplete problems in quadratic and nonlinear programming. Mathematical Programming: Series A and B, 39(2):117–129.

Mathematical Association of America. 2024. Aime. OpenAI. 2024. Learning to reason with llms.

Antonis Papachristodoulou, James Anderson, Giorgio Valmorbida, Stephen Prajna, Pete Seiler, Pablo Parrilo, Matthew M Peet, and Declan Jagt. 2013. Sostools version 4.00 sum of squares optimization toolbox for matlab. arXiv preprint arXiv:1310.4716.

Pablo A Parrilo. 2000. Structured semidefinite programs and semialgebraic geometry methods in robustness and optimization. Ph.D. thesis, California Institute of Technology.

Pablo A Parrilo and Bernd Sturmfels. 2001. Minimizing polynomial functions. arXiv preprint math/0103170.

Stephen Prajna, Antonis Papachristodoulou, and Pablo A Parrilo. 2002. Introducing sostools: A general purpose sum of squares programming solver. In Proceedings of the 41st IEEE Conference on Decision and Control, 2002., volume 1, pages 741–746. IEEE.

Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. 2023. Large language models encode clinical knowledge. Nature, 620(7972):172–180.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2024a. Scaling llm test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2024b. Scaling llm test-time compute optimally can be more effective than scaling model parameters. Preprint, arXiv:2408.03314.

Hayato Waki, Sunyoung Kim, Masakazu Kojima, and Masakazu Muramatsu. 2006. Sums of squares and semidefinite program relaxations for polynomial optimization problems with structured sparsity. SIAM Journal on Optimization, 17(1):218–242.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837.

Tillmann Weisser, Jean B Lasserre, and Kim-Chuan Toh. 2018. Sparse-bsos: a bounded degree sos hierarchy for large scale polynomial optimization with sparsity. Mathematical Programming Computation, 10:1–32.

Sean Welleck, Amanda Bertsch, Matthew Finlayson, Hailey Schoelkopf, Alex Xie, Graham Neubig, Ilia Kulikov, and Zaid Harchaoui. 2024. From decoding to meta-generation: Inference-time algorithms for large language models. arXiv preprint arXiv:2406.16838.

Huajian Xin, Daya Guo, Zhihong Shao, Zhizhou Ren, Qihao Zhu, Bo Liu, Chong Ruan, Wenda Li, and Xiaodan Liang. 2024. Deepseek-prover: Advancing theorem proving in llms through large-scale synthetic data. arXiv preprint arXiv:2405.14333.

Naruki Yoshikawa, Marta Skreta, Kourosh Darvish, Sebastian Arellano-Rubach, Zhi Ji, Lasse Bjørn Kristensen, Andrew Zou Li, Yuchi Zhao, Haoping Xu, Artur Kuramshin, et al. 2023. Large language models for chemistry robotics. Autonomous Robots, 47(8):1057–1086.

Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang, Xiaofei Sun, Shuhe Wang, Jiwei Li, Runyi Hu, Tianwei Zhang, Fei Wu, et al. 2023. Instruction tuning for large language models: A survey. arXiv preprint arXiv:2308.10792.

Yang Zheng, Giovanni Fantuzzi, and Antonis Papachristodoulou. 2019. Sparse sum-of-squares (sos) optimization: A bridge between dsos/sdsos and sos optimization for sparse polynomials. In 2019 American Control Conference (ACC), pages 5513–5518. IEEE.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. 2024. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand. Association for Computational Linguistics.

Wenqi Zhu and Coralia Cartis. 2024. Global convergence of high-order regularization methods with sums-of-squares taylor models. arXiv preprint arXiv:2404.03035.

### A Literature Review for SoS

The problem of determining whether a multivariate polynomial is nonnegative is inherently linked to the task of finding its global minimum—a fundamental challenge in the optimization community (Ahmadi et al., 2023a; Zhu and Cartis, 2024; Lasserre, 2000; Parrilo and Sturmfels, 2001). Testing whether a general polynomial is nonnegative is provably NP-hard, even for polynomials of relatively low degrees or with a small number of variables. For instance, it has been shown that finding the global minimum of general even-degree polynomials of degree at least four is NP-hard (Ahmadi et al., 2023a, 2013; Ahmadi and Zhang, 2022; Murty and Kabadi, 1987).

Due to the computational intractability of the general problem, we seek special cases of polynomials where the challenging nonnegativity constraints can be replaced with more manageable conditions. The sum of squares (SoS) condition, a mathematical technique in polynomial optimization where a polynomial is expressed as a sum of squared polynomials, provides a sufficient criterion for polynomial nonnegativity. The SoS property is particularly useful because it allows the nonnegativity problem to be reformulated as a semidefinite programming (SDP) problem, for which efficient algorithms, such as interior-point methods, exist. In certain special cases, nonnegativity and SoS are equivalent; for example, any nonnegative quadratic polynomial or any nonnegative even-degree univariate polynomial can always be expressed as a sum of squares (Hilbert, 1893; Ahmadi and Parrilo, 2013; Ahmadi et al., 2023b). For more complex polynomials, the Lasserre hierarchy provides a systematic way to approximate nonnegativity using a sequence of SoS relaxations (Lasserre, 2001). This method constructs a sequence of SDP problems that yield increasingly tighter approximations to nonnegativity.

Many large-scale problems exhibit structured sparsity patterns, enabling the application of a sparsity-adapted hierarchy of SDP relaxations (Camps and Sznaier, 2017; Lasserre, 2006; Molzahn and Hiskens, 2015; Waki et al., 2006). Additional techniques for addressing large-scale problems include Structured DSoS and SDSoS programming, as well as Bounded Degree SoS (BSoS) (Ahmadi and Majumdar, 2019; Lasserre et al., 2017; Waki et al., 2006; Weisser et al., 2018; Zheng et al., 2019). These approaches take advantage of the structure of the problem (sparsity) to generate smaller SDPs. There are also methods to reformulate the original optimization problem to reduce the size of the optimization. For instance, the optimization of a multivariate fourth-order (quar-

tic) homogeneous polynomial under quadratic constraints can be relaxed into a quadratic SDP (Luo and Zhang, 2010). In contrast to the SoS approach, which gives a matrix variable of size at least N×N, the quadratic SDP system has a size of n × n only. The resulting quadratic SDP can be well approximated in polynomial time in some cases, but it remains NP-hard. Yet, these methods primarily depend on the specific structure of the problem, and generally, the scalability of characterizing polynomial nonnegativity remains a significant challenge in the literature.

### B Mathematical Background for SoS

Definition 2.1 implies that the degree of an SoS polynomial q has to be even and that the maximum degree of each q˜j is deg[2 q]. Therefore, we denote the degree of the SoS polynomial as p′ = 2d where d is a positive integer.

Definition B.1 (Total degree polynomial space)

Let n > 0 be the dimension and x = [x1,...,xn]T be the variables. We denote Pp′[x] as the general representation of the polynomial spaces, where p′ represents the highest degree each entry can take. The associated multi-index set as α = [α1,α2,...,αn] ∈ [Z[0,p′]]n where each αr is an integer between 0 to p′ inclusively and the indices satisfy nr=1 αr ≤ p′. The number of monomial bases are N := n+p′p′ .

Theorem B.1 (From (Parrilo, 2000)) For a variable x = [x1,...,xn]T and an even integer p′ = 2d, let ϕd(x) be the vector of all monomials of degree at most d in xj for 1 ≤ j ≤ n. A polynomial pˆ : Rn → R of degree p′ is SoS if and only if there exists a symmetric matrix Q such that (i)

- pˆ(x) = ϕd(x)TQϕd(x) for all x ∈ Rn, (ii) Q ⪰ 0.

### C SoS Simple

- Step 1: Examine if the highest degree is odd or even.
- Step 2: For the even highest degree d, examine the coefficients of highest-degree terms. Check for any

negative values.

- Step 3: Consider these special properties:

- • Properties of quadratic polynomials
- • Properties of quartic polynomials in 1-2 variables.
- • Properties of quartic homogeneous polynomials in 1-3 variables.
- • Properties of even-degree univariate polynomials.

- Step 4: Try direct sum of squares representation.
- Step 5: Consider matrix methods if needed. D SoS Reasoning

Step 1. Degree: An SoS polynomial must have an even degree (i.e., its highest-degree term must have an even exponent). Any odd-degree polynomial cannot be expressed as a sum of squared polynomials. This is the simplest criterion and should always be checked first.

If the highest-degree univariate term (i.e., xd1,...,xdn) has a negative coefficient, then the polynomial is not SoS. Otherwise, we cannot determine whether it is SoS and proceed to the next step.

- Example 1: p(x) = x41 − x42 + x43 + x21x22. Since

the highest-degree univariate term has a negative coefficient (namely, −x42), by letting x2 → ∞, it is clear that p(x) becomes negative. Therefore, it is not SoS.

- Example 2: p(x) = x41+x42+x43−2x21x22+x1x2.

All the highest-degree univariate terms have nonnegative coefficients (i.e., x41,x42,x43). Thus, we cannot determine whether it is SoS, and we move to the next step.

Example 3: p(x) = x41 + x42 − 2x21x22. This polynomial is SoS because it can be rewritten as: p(x) = (x21 −x22)2. Note that a negative coefficient in the highest-degree cross term is allowed. For instance, in this case, we have the negative coefficient

cross term −2x21x22. However, the highest-degree univariate terms are positive (i.e., x41,x42).

Test Set Construction: Test Set 1 is constructed such that the highest-degree term is odd, thereby ensuring that the polynomials are not SoS.

- Step 2. Non-negativity: SoS polynomials are nonnegative for all real inputs. For example, if a polynomial p(x) has a negative constant term, then p(0) < 0, proving it is not SoS. Similarly, if a horizontally translated and scaled polynomial

q(x) = cp(x + d) (for any c ∈ R and d ∈ Rn ) satisfies q(0) < 0, then p(x) cannot be SoS.

To determine whether a polynomial is nonnegative, please use the following approaches:

Constant coefficient check: If the constant coefficient is negative, then p(0) < 0. For instance, p(x) = x4 +x3 −1, p(x) = x21 +x21x22 +x42 −0.1. are no SoS polynomials.

Grid evaluation: Try finding the minimum value of the polynomial over a selected evaluation grid. It is crucial to perform this step. Substitute multiple values of x, such as (1,0,0,...), (0,1,0,...), (0,0,1,...), etc., to check whether the polynomial evaluates to a negative value.

Leading order and dominant terms: Analyze the highest-degree terms and explore symmetries among cross terms. Evaluate the magnitude of negative coefficients relative to positive coefficients.

Finding minima: Attempt to find the local or global minimum of the polynomial to determine if it is negative.

Finding Symmetry and Translation: Example 1: Consider a horizontally translated and scaled polynomial: p(x) = 1.8x21 + 10.8x1 + 1.2x22 + 4.8x2 + 20.82. Rewriting, p(x) = 1.8(x1 + 3)2 + 1.2(x2 + 2)2 − 0.18. Since p(−3,−2) < 0, the polynomial is still not SoS.

- Step 3. Square Form: An SoS polynomial p(x)

can be written as p(x) = i qi(x)2, where each qi(x) is a polynomial. Examples and counterexamples are provided in Test Set 2, as this is the most

common method for checking SoS.

Example: Consider p(x) = (x1 − x1x2)2 + (x22 − x41)2, which is an SoS polynomial. However, polynomials are sometimes given in their expanded form. For instance, the same polynomial can be written as: p(x) = −2x21x2 + x21 + x81 − 2x41x22 + x21x22 + x42. On the other hand, consider p(x) = (x1 − x1x2)2 + (x22 − x41)2 − 20. This polynomial is not SoS. To determine whether an expanded polynomial can be expressed in SoS form with a negative constant, one should analyze the symmetries of the terms and the structure of the cross terms.

Test Set Construction: Examples and counterexamples are provided in Test Set 2. Step 4. Special Structures and Cases:

- a) Any nonnegative quadratic polynomial is a sum of squares (SoS).

Examples: p(x) = x21 + x22 − 2x1x2, p(x) = x21 + x22 + 4x23 − 3x2x3. These are SoS. Counterexamples: p(x) = x21+x22−2x1x2−1, p(x) = x21 + x22 + 4x23 − 5x2x3.

- b) Any nonnegative quartic polynomial in one or two variables is SoS.

Example: p(x) = x41 + 2x21x2 − 2x21 + x22 − 2x2 + 1 = (x21 + x2 − 1)2. Counterexample: p(x) = x41 +2x21x2 −2x21 + x22 − 2x2 = (x21 + x2 − 1)2 − 1.

- c) Any nonnegative quartic homogeneous polynomial in one, two, or three variables is SoS.
- d) Any nonnegative even-degree univariate polynomial is SoS.

Example: p(x) = x6 + 3x4 + 2x2, which is nonnegative and SoS. Counterexample: p(x) = x6 + 3x4 + 2x, which takes negative values and is not SoS.

- e) Any nonnegative polynomial with a quadratic term and quartic regularization is SoS.

Therefore, if a polynomial meets one of the above criteria and is nonnegative, it is an SoS polynomial. Nonnegativity can be verified by determining the global minimum, checking the descent direction, or performing a grid search.

Test Set Construction: Examples are provided in Test Set 3.1, 3.2, and 4, while counterexamples are constructed as polynomials that take negative values.

Step 5. Matrix Decomposition and Check for Symmetric Positive Definite Q: If the above checks fail, we can use the following theoretical reasoning:

- a) For an even degree 2d polynomial in

[x1,...,xn], construct a monomial basis using canonical ordering:

y∗ := (x∗1,...,x∗n,(x∗1)2,...,x∗1x∗n, x∗2x∗3,...,(x∗1)2d,...,(x∗n)2d).

This vector y∗ has length n+22dd .

- b) Express the polynomial as p(x) = y∗⊤Qy∗,

where Q is a symmetric matrix of size

n+2d

2d × n+22dd . Note that this representation is not unique; there are multiple valid forms of Q.

- c) Check whether Q is positive definite. This can be done by finding its smallest eigenvalue. If such a Q exists, then p(x) is a sum of squares (SoS). Otherwise, p(x) is very likely not SoS.

If all the above tests fail, we can try Semidefinite Programming (SDP), which is the test used by existing solvers (e.g., YALMIP) to verify whether a polynomial is SoS. For example, to determine if

p(x) = x41−4x31x2+7x21x22−4x1x32−4x1x2+x42

is SoS. We convert the problem to the following. We solve the SDP

γ∗ = minp = y40−4y31+7y22−4y13−4y11+y04

subject to the constraint





|1|y10 y01<br><br>|y20 y11 y02|
|---|---|---|
|y10 y01|y20 y11 y11 y02<br><br>|y30 y21 y12 y21 y12 y03<br><br>|
|y20 y11 y02<br><br>|y30 y21 y21 y12 y12 y03<br><br>|y40 y31 y22 y31 y22 y13 y22 y13 y04|

⪰ 0.

 

 

If γ∗ ≥ 0, then p is SoS; otherwise, it is not.

Test Set Construction: Examples and counterexamples are provided in Test Set 5.1-5.4. The examples are constructed using different positive semidefinite (PSD) matrices Q, such as sparse PSD Q, low-rank PSD Q, and ill-conditioned PSD Q. The counterexamples are generated using an indefinite Q and the corresponding polynomials are likely to not be SoS. However, to conclusively prove that a polynomial is not SoS, all possible Q must be examined. In our test set, we crosschecked the results with classic solvers to confirm the "Not SoS" classification for Test Sets 5.1–5.4.

- E Details for Test Subsets We provide the details of test subsets in Table 3.
- F Sample Model’s Response under SoS Reasoning Prompt

F.1 o1-mini response for a simple polynomial under SoS Reasoning Prompt

To determine whether the given polynomial is a sum of squares (SoS), we will follow the provided steps:

#### Step 1: Check the Highest Degree

The highest degree of a polynomial is the largest total degree among its terms. When expanding these squared terms, the highest degree terms are of the form x3i (e.g., x31, x32, etc.), which are degree 3 terms.

Since 3 is an odd degree, according to Step 1: If the highest degree of a polynomial is odd, it is not SoS. Therefore, the polynomial cannot be expressed as a sum of squares.

###### Polynomial Type length<4000 length>4000 Total Is it SoS? Difficulty

- Test Set 1: Odd Degree Polynomial 150 50 200 NO Easy

- Test Set 2a: SoS (Expanded Form) 69 51 120 YES Hard Test Set 2b: Negative (Expanded Form) 23 40 63 NO Hard

- Test Set 2.1a: SoS (Squared Form) 105 15 120 YES Easy
- Test Set 2.1b: Negative (Squared Form) 38 25 63 NO Easy

- Test Set 3.1a: Nonnegative Quadratic Quartic 100 0 100 YES Medium
- Test Set 3.1b: Negative Quadratic Quartic 100 0 100 NO Medium

- Test Set 3.2a: Nonnegative Quartic with 2 variables 100 0 100 YES Medium
- Test Set 3.2b: Negative Quartic 100 0 100 NO Medium

- Test Set 4a: Nonnegative Quadratic Quartic 100 0 100 YES Medium
- Test Set 4b: Negative Quartic 100 0 100 NO Medium

- Test Set 5.1a: PSD Q 80 16 96 YES Hard
- Test Set 5.1b: Non-PD Q 80 16 96 NO Hard

- Test Set 5.2a: PSD Spare Q (Sparsity 0.1) 56 16 72 YES Hard
- Test Set 5.2b: Non-PD Spare Q (Sparsity 0.1) 56 16 72 NO Hard

- Test Set 5.3a: PSD Low Rank Q (rank 3) 42 18 60 YES Hard
- Test Set 5.3b: Non-PD Low Rank Q (rank 3) 28 12 40 NO Hard

- Test Set 5.4a: PSD Ill-Conditioned Q (λ = 1 − 1012) 20 15 35 YES Hard
- Test Set 5.4b: Non-PD Ill-Conditioned Q 40 30 70 NO Hard

Table 3: Summary of SoS-1K Test Sets.

F.2 o1-mini response for a difficult polynomial under SoS Reasoning Prompt

To determine whether the given polynomial is a sum of squares (SoS), we follow the outlined steps meticulously.

Step 1: Check the Highest Degree The highest total degree among the terms is 4,

which is even. Since the highest degree is even, we proceed to the next step.

Step 2: Check Highest Degree Univariate Terms for Negative Coefficients

Univariate Terms of Degree 4: 7.9x41 (Coefficient: 7.9) 4.1x42 (Coefficient: 4.1) 0.91x43 (Coefficient: 0.91) 2.0x44 (Coefficient: 2.0) 0.92x45 (Coefficient: 0.92)

Conclusion: All highest degree univariate terms have positive coefficients. We cannot determine

the SoS property yet and move to the next step.

#### Step 3: Check for Negative Values

Testing Specific Points: Point (1, 1, 1, 0, 0), Evaluated p(1,1,1,0,0) = 66.89 (Positive). Point (0, 0, 0, 1, 1), Evaluated p(0,0,0,1,1) = 6.52 (Positive). Point (1, 1, 1, 1, 0), Evaluated p(1,1,1,1,0) = 2.037 (Positive).

Conclusion: At tested points, the polynomial remains nonnegative. While this doesn’t guarantee it’s SoS, it suggests the possibility.

- Step 4: Special Cases Analysis: The polynomial doesn’t clearly fit into any of the special cases (a to f) outlined for immediate determination.
- Step 5: Advanced Methods - Matrix Decomposition and SDP: Given the complexity of the polynomial, performing matrix decomposition or semidefinite programming (SDP) manually is im-

practical. However, based on the evaluations in

- Step 3 and the positive coefficients in Step 2, there’s a strong indication that the polynomial does not take negative values.

Final Conclusion: Based on the evaluations and the absence of negative terms in critical components, the polynomial is a sum of squares.

### G Further Results for Performance of SOTA LLMs

Model # Total Samples # Valid Samples Instruction Type SoS Plain SoS Simple SoS Reasoning

DeepSeek-R1 340 300 302 234 QWQ-32b 340 233 259 225 o1-mini 340 338 340 337 Qwen2.5-7b 340 323 332 277 GPT-4o 340 336 338 337 Qwen2.5-7b-1m 340 340 339 286 Qwen2.5-14b 340 325 340 316 Qwen2.5-14b-1m 340 340 336 309 GPT-4o-mini 340 339 339 327 DeepSeek-V1 340 340 340 332 Qwen2.5-32b 340 334 339 315 Average 340 323 328 300

Table 4: # Total Samples and Valid Samples of Different Models.

Note that the first three polynomials are wellknown examples of nonnegative but not sumof-squares (SoS) polynomials. The first is the Motzkin polynomial, a classic example of a nonnegative polynomial that is not SoS. The second is the Robinson polynomial, a counterexample to Hilbert’s 17th problem. The last polynomial is a recent result from Ahmadi’s work (Ahmadi et al., 2023b, Thm 3.3).

Finding nonnegative polynomials that are not sum of squares (SoS) is a significant ongoing research problem in real algebraic geometry and polynomial optimization (Ahmadi et al., 2023b, 2024; Ahmadi and Zhang, 2022). It connects to Hilbert’s 17th problem, semidefinite programming (SDP), and positivity certificates in polynomial optimization.

### H Research Level SoS and Nonnegativity Questions

- 1. Can you comment on the sum of squares (SoS) and nonnegativity properties of the following polynomials?

- • p(x) = x41x22 + x21x42 + 1 − 3x21x22,
- • p(x) = x61 + x62 + x63 − x41x22 − x41x23 − x42x21 −x42x23 −x43x21 −x43x22 +3x21x22x23,
- • p(x) = (x41 + x42 + x43) + 2(x21 + x22 + x23) + 8(x1x2 + x1x3 + x2x3) + 49.

- 2. Can you construct a general formula for nonnegative polynomials that are not SoS?
- 3. Can you provide a new nonnegative polynomial that is not SoS and has never been found in the literature?

