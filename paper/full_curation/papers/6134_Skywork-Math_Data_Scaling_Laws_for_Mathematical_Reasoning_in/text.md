[Figure 1]

Technical Report

# arXiv:2407.08348v2[cs.AI]17Jul2024

## Skywork-Math: Data Scaling Laws for Mathematical Reasoning in Large Language Models — The Story Goes On

###### Liang Zeng, Liangjun Zhong, Liang Zhao, Tianwen Wei,

Liu Yang, Jujie He, Cheng Cheng, Rui Hu, Yang Liu, Shuicheng Yan, Han Fang, Yahui Zhou {forename}.{surname}@kunlun-inc.com Skywork AI, Kunlun Inc.

### Abstract

In this paper, we investigate the underlying factors that potentially enhance the mathematical reasoning capabilities of large language models (LLMs). We argue that the data scaling law for math reasoning capabilities in modern LLMs is far from being saturated, highlighting how the model’s quality improves with increases in data quantity. To support this claim, we introduce the Skywork-Math model series, supervised fine-tuned (SFT) on common 7B LLMs using our proposed 2.5M-instance Skywork-MathQA dataset. Skywork-Math 7B has achieved impressive accuracies of 51.2% on the competition-level MATH benchmark and 83.9% on the GSM8K benchmark using only SFT data, outperforming an early version of GPT-4 on MATH. The superior performance of Skywork-Math models contributes to our novel two-stage data synthesis and model SFT pipelines, which include three different augmentation methods and a diverse seed problem set, ensuring both the quantity and quality of Skywork-MathQA dataset across varying difficulty levels. Most importantly, we provide several practical takeaways to enhance math reasoning abilities in LLMs for both research and industry applications.

Top@1 Accuracy (%)

| | | |Sk|ywork-Math-Mistral-7B| | | |
|---|---|---|---|---|---|---|---|
| | |Skywor|k-Math-DeepSeekMat|h-7B|GP|T-4-03|14|
| | |Skywor|DeepSe<br><br>Xw<br><br>k-Math-LLaMA2-7B|ekMath-Instruct-7B<br><br>Xwin-Math-1 in-Math-Mistral-7B|GPT<br><br>3B|-4| |
| | | |MAmmoTH-70B| | | | |
| |MAmmoTH-7B<br><br>MAm|moTH-13B|InternLM2-Math-7<br><br>InternLM<br><br>ChatGLM3-Ma|B<br><br>GPT-3.5-Turbo<br><br>2-Math-20B<br><br>th-SFT-32B/Xwin-Mat|h-7B| | |
| | |Me|taMath-13B Wiza<br><br>MetaMath-Mistral-|rdMath-70B<br><br>7B MetaMath-70B<br><br>LLaMA3-8B<br><br>LEMA-LLaMA2|-70B| | |
| |WizardMath-7B|MetaMath-7B<br><br>LEMA-LLaMA2-13B<br><br>WizardMath-13B| | | | | |
| |LEMA-LLaMA2-7B| | | | | | |
| | | | | | | | |

50

40

MATH

30

20

10

50 60 70 80 90

GSM8K

Figure 1 | Top1 accuracy on GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021) using only SFT techniques, without using external toolkits and voting techniques. Following MetaMath (Yu et al., 2024), we employ a zero-shot chain-of-thought evaluation framework. Skywork-Math models achieve state-of-the-art accuracy among models smaller than 10B parameters using only synthetic SFT data and surpass an early version of GPT-4 on MATH.

1

#### 1. Introduction

More is different.

—-Philip W. Anderson, 1972

Reasoning ability is a hallmark of human intelligence (Gendron et al., 2024; Huang and Chang, 2022; Wei et al., 2022b). Although Large Language Models (LLMs) have recently demonstrated significant capabilities in various tasks such as conversation (Achiam et al., 2023; Anthropic, 2024; Peng et al., 2023) and summarization (Almazrouei et al., 2023; Scao et al., 2022; Wei et al., 2023b; Yang et al., 2023), they often struggle with complex reasoning tasks (Gendron et al., 2024; Lu et al., 2023; Wu et al., 2023). One particularly challenging area is mathematical reasoning (Arora et al., 2023; Cobbe et al., 2021; He et al., 2024; Hendrycks et al., 2021; Zhong et al., 2023), which requires the ability to solve mathematical problems and derive logical conclusions in a step by step manner (Saxton et al., 2019; Shao et al., 2024; Toshniwal et al., 2024; Wei et al., 2022b; Yu et al., 2024).

Two prevailing beliefs guide researchers and practitioners in enhancing mathematical reasoning abilities of LLMs. The first belief posits that complex reasoning abilities, especially mathematical reasoning, are emergent abilities that exist in large language models but not in small models (Wei et al., 2022a,b). Typically, models with more than 30 billion parameters exhibit the strong mathematical reasoning ability (Brown et al., 2020). The second belief is the seminal "superficial alignment" hypothesis (Zhou et al., 2023), which asserts that "A model’s knowledge and capabilities are learnt almost entirely during pre-training, while alignment teaches it which sub-distribution of formats should be used when interacting with users.". According to this hypothesis, the alignment process, primarily through supervised fine-tuning (SFT), does not inject new knowledge or improve inherent abilities but rather adjusts the output response format. This implies that the strong mathematical reasoning ability may not be significantly improved by a large amount of synthetic SFT data.

In this paper, we re-examine these two common beliefs mentioned above regarding mathematical reasoning abilities of LLMs. For the first belief, we introduce the Skywork-Math model series, which are supervised fine-tuned (SFT) on common 7B pre-trained LLM models without employing other complex alignment techniques such as RLHF (Bai et al., 2022; Casper et al.,

- 2023) and DPO (Rafailov et al., 2024). Skywork-Math 7B models have achieved impressive accuracies of 51.2% on the competition-level MATH (Hendrycks et al., 2021) benchmark and 83.9% on the GSM8K (Cobbe et al., 2021) benchmark, notably outperforming an early version of GPT-4 on MATH. Our empirical findings, consistent with the conclusions in Li et al. (2024), suggest that strong mathematical reasoning ability can indeed exist in common 7B language models. Moreover, scaling up synthetic SFT data can further enhance the mathematical reasoning ability of Skywork-Math 7B models.

For the second belief, we propose Skywork-MathQA high-quality SFT dataset containing

###### 2.5 million instances, which is much larger than open-sourced dataset of its kind to date, such as MetaMathQA (Yu et al., 2024) containing 395K samples. We empirically observe that the scaling law curve on the SFT alignment for mathematical reasoning in modern LLMs is far from being saturated (ref. Figure 5). We have carefully scaled the Skywork-MathQA SFT dataset with diverse and high-quality samples specifically within the mathematical domain to enhance the model’s capability in understanding and solving mathematical problems.

Due to the scarcity of high-quality and challenging mathematical data, various pipelines

and prompts have been employed to generate synthetic mathematical data (Li et al., 2024; Shao et al., 2024; Toshniwal et al., 2024; Wang et al., 2022; Wei et al., 2022b; Yu et al., 2024). To address this deficiency, we employ GPT-4 to generate a substantial amount of synthetic data through a novel two-stage data synthesis pipeline, in conjunction with the corresponding model SFT process. In stage 1, our objective is to obtain normal synthetic problems to enhance the models’ general comprehension of mathematical problems. To maintain the diversity in data selection process, we utilize the core-set approach (Sener and Savarese, 2017) on enlarged seed problems. However, as the data volume increases, we empirically observe that the relationship between performance and data quantity begins to plateau. Accordingly, in stage 2, we diversify the dataset further by introducing a proportion of augmented hard problems (ref. Figure 3 for illustrative examples), thereby exposing the model to more challenging mathematical questions. Without continual pre-training on a large-scale math corpus (Azerbayev et al., 2023; Shao et al.,

- 2024), Skywork-Math models achieve impressive performance with just supervised fine-tuning on common pre-trained LLMs containing only 7B parameters.

Most importantly, we provide valuable insights and practical takeaways to enhance the mathematical reasoning ability in LLMs, benefiting both research and industry communities:

Highlighted Takeaways

- • The potential for math reasoning capabilities in modern LLMs is far from exhausted. The quality of LLMs can significantly improve with increases in data quantity (ref. Figure 5). Skywork-Math 7B models already demonstrate strong mathematical reasoning abilities by SFTing on common 7B pre-trained LLM models.
- • The learning process for accessing the math reasoning ability involves multiple stages. Training LLM models in a meaningful order, from the easy problems to the hard ones, can provide performance improvements.
- • When scaling the synthetic SFT dataset, increasing the diversity of seed problems and augmentation methods can improve the math reasoning performance of LLMs.
- • Selecting influential data with high-quality from a large dataset is non-trivial (Engstrom et al., 2024). Our empirical results indicate that some straightforward methods to select the so-called "high-quality" data may not increase (and can even hurt) LLMs’ performance compared to randomly selecting data. The selection process involves multiple constraints, and the "high-quality" data could significantly decrease the difficulty level of problems, thus negatively impacting the performance of LLMs.
- • The LLM models have strong knowledge transfer capabilities for mathematical reasoning across bilingual benchmarks (i.e., English and Chinese). We hypothesize that this can be attributed to the inherent nature of symbols and numbers in math problems, which retain their intrinsic meaning regardless of the language used.
- • Although Skywork-Math 7B models have achieved considerable improvement in robustness tests compared to other open-source LLM models, they remain sensitive to the distractors in math word problems compared with proprietary GPT-4 models.
- • Sparse MOE models cannot clearly exceed the performance upper bound of their dense counterparts through SFT alignment in the context of math reasoning.
- • Two subtle but crucial practical implementation techniques—preventing data leakage and considering the influence of model maximum length—significantly impact the final performance of LLM models.

#### 2. Related Work

Alignment in LLMs. Large Language Models (LLMs) have recently transformed Natural Language Processing (NLP) (Achiam et al., 2023; Anil et al., 2023; Anthropic, 2024; Touvron

- et al., 2023), excelling in tasks such as automated summarization (Scao et al., 2022) and machine translation (Almazrouei et al., 2023). Alignment in LLMs refers to the process of ensuring that the model’s outputs adhere to user preferences (Shen et al., 2023). Various techniques contribute to achieving alignment, including supervised fine-tuning (SFT) (Taori et al., 2023), reinforcement learning from human feedback (RLHF) (Bai et al., 2022), and direct policy optimization (DPO) (Rafailov et al., 2024). Among these techniques, SFT is typically an indispensable method for aligning LLMs and has achieved highly competitive performance across various tasks (Chiang et al., 2023), particularly in mathematical reasoning (Li et al., 2024). SFT involves fine-tuning a pre-trained large model using annotated data, making the model’s performance more accurate for downstream tasks. Our work aims to deeply explore the performance boundaries of common 7B pre-trained LLMs using only the SFT alignment technique.

Quantity and Quality of SFT Data. Data is the fuel that powers the performance of LLMs. This ongoing discussion about whether the quantity or quality of SFT data is more important highlights their significance in enhancing the SFT performance of LLMs. (1) Quantity. Many recent research demonstrates the scaling properties in LLM fine-tuning (Kaplan et al., 2020; Li

- et al., 2024). The size of the fine-tuning dataset is a crucial factor affecting the LLMs’ performance. However, the optimal fine-tuning data size is highly task-dependent (Zhang et al., 2024). (2) Quality. Several studies (Cao et al., 2023; Gunasekar et al., 2023; Li et al., 2023; Zhou et al.,

- 2023) argue that the quality of fine-tuning data is equally critical. The renowned "less is more" work (Zhou et al., 2023) suggests that substantial knowledge acquisition occurs during the pretraining stage, minimizing the need for extensive fine-tuning data. Additionally, the InstructionFollowing Difficulty (IFD) metric introduced by (Li et al., 2023) and the QaDS strategy proposed in (Ni et al., 2024) aim to select diverse and high-quality instruction-following data to enhance LLM fine-tuning efficiency. Collecting a huge number of high-quality mathematical reasoning data is often time-consuming and labor-intensive. In this work, we generate a substantial amount of SFT synthetic data to investigate how the quantity of data impacts the performance of LLM models in mathematical reasoning.

Mathematical Reasoning in LLMs. LLMs have recently achieved significant progress in the area of mathematical reasoning (Shao et al., 2024). Initial benchmarks, such as simple math problems (Lan et al., 2022; Saxton et al., 2019), were readily solved by recent LLM models. This success prompts the introduction of more challenging benchmarks, such as GSM8K (Cobbe

- et al., 2021) and MATH (Hendrycks et al., 2021). Many recent works have proposed continual pre-training on massive math corpora to improve their math reasoning capabilities (Azerbayev et al., 2023; Jiang et al., 2023; Paster et al., 2024; Shao et al., 2024). Furthermore, significant progress has been made in alignment for solving mathematical problems (Li et al., 2024; Luo et al., 2023; Ni et al., 2024; Shao et al., 2024; Xu et al., 2024; Yu et al., 2024; Yue et al., 2023). These studies focus on generating high-quality synthetic data or collecting human-labeled data for model fine-tuning and alignment in the domain of math problem-solving. Additionally, reasoning frameworks aim at improving math capacity, such as the chain-of-thought (COT) prompting technique (Wang et al., 2022; Wei et al., 2022b), which enable LLMs to break down the reasoning process into manageable steps, resulting in more accurate outputs. Moreover, some complex math problems need the ability to conduct accurate arithmetic operations, a

|Stage 1|[Figure 2]|
|---|---|
|Base Model<br><br>Intermediate Model<br><br>[Figure 3]<br><br>[Figure 4]<br><br>Normal Synthetic Problems<br><br>Seed Problems<br><br>Data Synthesis<br><br>[Figure 5]<br><br>Seed Synthetic Problems<br><br>Data Synthesis<br><br>[Figure 6]<br><br>Diversity Selection<br><br>[Figure 7]<br><br>[Figure 8]<br><br>[Figure 9]| |

|Stage 2|[Figure 10]|
|---|---|
|Intermediate Model<br><br>Skywork-Math Model<br><br>[Figure 11]<br><br>LLM<br><br>[Figure 12]<br><br>Hard Synthetic Problems<br><br>Hard Seed Problems<br><br>Data Synthesis<br><br>[Figure 13]<br><br>[Figure 14]| |

(a) Data Synthesis Pipeline (b) Model SFT Pipeline

- Figure 2 | Overview of our proposed two-stage method. (a) The data synthesis pipeline of the Skywork-MathQA dataset. (b) The model SFT pipeline of the Skywork-Math model series.

capability that LLMs often lack (Yuan et al., 2023). For tool-integrated math problem-solving, program-of-thoughts (Chen et al., 2022; Shao et al., 2024; Toshniwal et al., 2024) prompts LLMs to produce answers in the code format, which are then executed by a code interpreter. Preliminary work indicates that SFT can improve the performance of open-source LLMs on mathematical reasoning tasks by fine-tuning them on synthetic data (Li et al., 2024; Yu et al., 2024). Building on this foundation, our work aims to thoroughly investigate the performance limits of common 7B pre-trained LLMs using only SFT synthetic data. We seek to determine the extent to which data quantity impacts LLM quality and to understand the mechanisms behind this influence.

#### 3. Method

In this section, we present the detailed methodology of Skywork-Math 7B models, as illustrated in Figure 2. Skywork-Math models aim to enhance math reasoning abilities during the model alignment process, particularly in the SFT stage, using common and publicly available 7B pretrained models. We employ a two-stage SFT approach, in conjunction with two data synthesis pipelines to produce high-quality data. In stage 1, we feed base pre-trained models with our generated normal synthetic problems to produce an intermediate model. In stage 2, to mitigate the diminishing returns in LLMs’ performance as the quantity of data increases, we generate hard synthetic problems and develop our Skywork-Math models. To ensure the quality of data, we primarily utilize GPT-4 1 (Achiam et al., 2023) to generate 2.5M-instance synthetic Skywork-MathQA dataset.

Supervised Fine-Tuning (SFT). SFT is an important and widely-used alignment technique in LLMs to enhance pre-trained models for excelling at specific tasks (Shen et al., 2023). We denote the token space of an input query and output response as X and Y, respectively. Typically,

1Without further clarification, the version of GPT-4 used in this paper is GPT-4-1106-preview.

LLMs generate an output response sequence y = (𝑦1, 𝑦2, . . . , 𝑦𝑇) in response to a given prompt query x = (𝑥1, 𝑥2, . . . , 𝑥𝑛). LLMs are the auto-regressive models characterized by a conditional probability distribution parameterized by 𝜃 as

P𝜃(y | x) =

𝑇

P𝜃(𝑦𝑡 | x, 𝑦1:𝑡−1). (1)

𝑡=1

Let a mathematical reasoning SFT training dataset be D = {(x𝑖,y𝑖)}𝑖𝑁=1, where x𝑖 and y𝑖 represent the 𝑖-th query and response, respectively 2. Here, 𝑁 is the total quantity of the SFT training dataset. Given such a dataset D, SFT can be performed using the following cross-entropy loss:

###### ∑︁𝑁

###### ∑︁𝑇

1

logP𝜃(𝑦𝑡𝑖 | x𝑖,y1:𝑖 𝑡−1). (2)

L(𝜃) = −

𝑁

𝑖=1

𝑡=1

Seed Problems. We adopt publicly available high-quality mathematical datasets to generate our Skywork-MathQA dataset. To prevent data leakage in the testing phase, we only use the training sets from data sources. The data sources are as follows:

- • MATH (Hendrycks et al., 2021) contains high school-level mathematical problems, some of which are from competitions such as the AIME and AMC. This dataset consists of 7,500 training data entries. Solving these problems requires advanced reasoning abilities and a comprehensive mathematical knowledge base. This dataset categorizes problems into five levels of difficulty and seven subdomains of high school mathematics.
- • We also use other data sources as seed problems. These included non-proving problems from OlympiadBench (He et al., 2024), mathematical problems from AGIEval (Zhong et al.,

2023) benchmark, and various problems in calculus, differential, statistics domains from SciBench (Wang et al., 2024) and JEEBench (Arora et al., 2023).

Here we do not use the training set of GSM8K as the seed problems because: (1) Math word problems represent a narrow category compared to general math problems 3, and an excessive focus on math word problems may reduce the diversity of the synthetic SFT data. (2) We empirically find that the requirements of math reasoning ability to solve the easy problem in the MATH benchmark are relatively equivalent to those needed for GSM8K.

Synthesis Process. We aim to answer the following question: as we gradually increase the quantity 𝑁 of the Skywork-MathQA dataset, does the models’ math reasoning ability improve correspondingly? For a given query/problem x𝑖, particularly the challenging competition-level math problems, manually annotating the response/answer y𝑖 is time-consuming and often infeasible for non-experts due to the required specific domain knowledge. Therefore, we utilize the top-performing GPT-4 models to synthesize diverse, high-quality SFT data (Li et al., 2024). The data synthesis process in the Skywork-MathQA dataset consists of two stages. In stage 1, we generate 2.1 million normal synthetic problems. In stage 2, we further generate 0.4 million hard synthetic problems, increasing the Skywork-MathQA dataset to a total of 2.5 million instances. Note that all data samples in the Skywork-MathQA dataset strictly adhere to the

2In what follows, we use the terms query-response and question-answer pairs interchangeably. 3Typically, math word problems involve a mathematical exercise presented in a narrative form, requiring the

extraction of numbers from the text and performing a sequence of elementary calculations using basic arithmetic operations (+ − ×÷) to reach the final answer.

Normal Problem Query: What is the domain of the function 𝑓 (𝑥) = 3𝑥𝑥++81? Enter your answer using interval notation.

Response: The function 𝑓 (𝑥) = 3𝑥𝑥++81 is defined for all real numbers except where the denominator is zero. Since the denominator 𝑥 + 8 is zero when 𝑥 = −8, we exclude this

value from the domain. Thus, the domain of 𝑓 (𝑥) in interval notation is (−∞,−8) ∪ (−8,∞). The answer is (−∞,−8) ∪ (−8,∞).

Hard Problem Query: Let 𝑓 (𝑥) : R → R be a function such that

𝑓 (𝑥) 𝑓 (𝑦) − 𝑓 (𝑥𝑦) 3

###### = 𝑥 + 𝑦 + 2

for all 𝑥, 𝑦 ∈ R. Find 𝑓 (𝑥).

Response: Setting 𝑦 = 1 gives us

𝑓 (𝑥) 𝑓 (1) − 𝑓 (𝑥) 3

###### = 𝑥 + 1 + 2,

which simplifies to

###### 𝑓 (𝑥)( 𝑓 (1) − 1) = 3𝑥 + 9.

Since 𝑓 (𝑥) is a function that maps R to R, and since this holds for all 𝑥 ∈ R, 𝑓 (1) − 1 cannot be zero because otherwise the right-hand side of the equation would always be 0, which is not true. Thus, we can divide both sides of the equation by 𝑓 (1) − 1 to get

3𝑥 + 9 𝑓 (1) − 1

.

𝑓 (𝑥) =

3(1) + 9 𝑓 (1) − 1

Setting 𝑥 = 1 in this equation gives us 𝑓 (1) =

, which is equivalent to 𝑓 (1)( 𝑓 (1) −

1) = 12. Let’s denote 𝑓 (1) as 𝑎, so we have 𝑎(𝑎 − 1) = 12. This is a quadratic equation in 𝑎: 𝑎2 − 𝑎 − 12 = 0. The solutions to this quadratic equation are 𝑎 = −3 or 𝑎 = 4. Since 𝑎 represents 𝑓 (1) and we have established that 𝑓 (1) − 1 cannot be zero, we discard the solution 𝑎 = 1 and accept 𝑎 = 4. Therefore, 𝑓 (1) = 4 and the function 𝑓 (𝑥) is

3𝑥 + 9 4 − 1

3𝑥 + 9 3

3𝑥 + 9 𝑓 (1) − 1

= 𝑥 + 3.

=

=

𝑓 (𝑥) =

The answer is x + 3.

- Figure 3 | Two examples of query-response pairs in the Skywork-MathQA dataset. The top figure illustrates a normal problem, and the bottom figure depicts a hard problem.

same data format. We instruct the Skywork-Math models to use the prefix "\nThe answer is " before generating answers in their responses. Figure 3 presents two examples from our Skywork-MathQA dataset: one is a normal problem, and the other is a hard problem. In the following sections, we will introduce the two-stage data synthesis pipeline along with its model SFT process.

###### 3.1. Stage 1: Normal Synthetic Problems

In this stage, we examine how the quality of Skywork-Math models improves as the quantity of SFT data increases. We generate 2.1 million high-quality and diverse SFT data within math reasoning domains by GPT-4. Our primary goal is to equip the model with a comprehensive understanding of mathematical reasoning problems by exposing it to a diverse range of math questions. Our empirical findings indicate that diversity is crucial for generating and scaling SFT data (ref. Section 4.3.2). We investigate this issue from two perspectives: data augmentation methods and diversity selection of seed problems.

Data Augmentation Methods. To ensure diversity in our synthetic data, we employ three distinct methods to augment our Skywork-MathQA dataset. We notice that the differences among these augmentation methods are subtle, however, combining these methods to improve diversity indeed influences the model’s performance. Three data augmentation methods have distinct approaches. By combining them, we can leverage the advantages of all three unique approaches in our data synthesis pipeline. Figure 4 demonstrates three prompt snippets used in our paper to highlight the characteristics of these distinct approaches. Detailed examples of the same query with different responses using these three methods can be found in Appendix A.

The first data augmentation method we adopt is MetaMathQA (Yu et al., 2024), which comprises four specific approaches: three for query bootstrapping and one for response augmentation. For query augmentation, we leave the corresponding query unchanged and employ GPT-4 to refine its response. For query bootstrapping, the rephrasing method utilizes pre-defined prompts to generate more questions, followed by the few-shot Chain-of-Thought (COT) (Wei

- et al., 2022b) prompting to generate answers. Additionally, the FOBAR (Jiang et al., 2024b) and self-verification (Weng et al., 2022) methods deterministically convert the problem into a backward format to mimic backward reasoning, i.e., given the result and think backward to determine the unknown variable in the question. After transforming the questions, we then generate corresponding answers with COT techniques using GPT-4. We also strive to balance the quantity of SFT data produced by these four augmentation approaches.

The second data augmentation method is the Evol-Instruct approach, as implemented in WizardLM (Xu et al., 2023). Starting from the initial set of mathematical problems, Evol-Instruct iteratively rewrites them step by step into more complex queries. We set the maximum length of the evolutionary trajectory to five steps and employ the following five augmentation strategies:

- • Rewrite the original problem to create a completely new problem of similar length and difficulty.
- • Add constraints and requirements to the original problem.
- • Increase the complexity of the original problem in both depth and breadth.
- • Replace general concepts with more specific ones.
- • Explicitly request additional steps in the reasoning process of the original question.

###### Prompts for MetaMath with Backward Reasoning

You are an AI assistant to help me solve math problem. I first give you an original question with an unknown variable 𝑋, then I give you the answer to this original question. You help me determine the value of 𝑋. Follow the given examples and answer the question.

Query: On a map, a 12-centimeter length represents 𝑋 kilometers. How many kilometers does a 17-centimeter length represent? ### If we know the answer to the above question is 102, what is the value of unknown variable 𝑋?

Response: We know that on the map, a 12-centimeter length represents 𝑋 kilometers. To find out how many kilometers a 17-centimeter length represents, we can set up a proportion using the given information: 12 centimeters / 𝑋 kilometers = 17 centimeters / 102 kilometers. We can cross-multiply and solve for 𝑋: 12 * 102 = 17 * 𝑋, 1224 = 17 * 𝑋. Dividing both sides by 17, we get: 𝑋 = 72 The value of X is 72.

..... [Few-shot examples]

###### Prompts for Evol with Evol-Instruct

I want you act as a Prompt Rewriter. Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., ChatGPT and GPT4) a bit harder to handle. But the rewritten prompt must be reasonable and must be understood and responded by humans. Please add one more constraints/requirements into #Given Prompt#.

...... [Omit some specific rules] #Given Prompt#: <Here is instruction.> #Rewritten Prompt#:

###### Prompts for Xwin with Self-Correction

Please act as a professional math teacher. Your goal is to create high quality math word problems to help students learn math. You will be given a math question. Please create a new question based on the Given Question and following instructions. To achieve the goal, you have three jobs. ...... [Omit some specific rules]

VERIFICATION AND MODIFICATION: <solve the question step-by-step and modify it to follow all principles> FINAL CREATED QUESTION: <your final created question>

- Figure 4 | Prompt snippets for MetaMath Yu et al. (2024), Evol Luo et al. (2023), and Xwin Li et al. (2024) are showcased, with their distinct approaches highlighted in red. The prompts are mainly derived from the original papers with minor modifications. For the sake of brevity, some specific few-shot examples and rules have been omitted.

The third data augmentation method is question generation with self-correction, as practiced in Xwin (Li et al., 2024). Specifically, we instruct GPT-4 to refine the input question and then verify it step-by-step to assess its logical and mathematical consistency. If the question is found to be imperfect, we instruct the GPT-4 to modify it based on the verification results.

Diversity Selection of Seed Problems. Initially, we simply use the training dataset of MATH along with additional mathematical data from other sources as the seed problem to generate queries and responses. To improve the diversity of seed problems, we employ the core-set approach (Sener and Savarese, 2017), which selects a representative subset of data that maximizes diversity while maintaining coverage of the original dataset’s key features. As shown in

Figure 2, we first perform data synthesis on the initial seed problems and then apply the core-set approach (Sener and Savarese, 2017) to obtain seed synthetic problems. We further perform data synthesis on these seed synthesis problems to get the normal synthetic problems with 2.1 million instances. We select common 7B pre-trained LLMs as base models and fine-tune these models on normal synthetic problems to produce the intermediate models with a general understanding of various mathematical problems and concepts.

- 3.2. Stage 2: Hard Synthetic Problems

As the quantity of data increased, we empirically observe that the relationship between performance and data quantity begins to plateau (ref. Section 4.3.1). Motivated by the concept of curriculum learning (Bengio et al., 2009; Soviany et al., 2022), we recognize that models can learn much better when data are organized in a meaningful order rather than presented randomly, introducing more complex concepts and problems gradually. In the domain of math problemsolving, it is natural to first learn the basic math operations and then progressively tackle more difficult problems. Therefore, we employ this strategy to guide the SFT data synthetic process. The stage 2 in the data synthesis pipeline is specifically designed for models to focus on mastering the more challenging problems. In this stage, we utilize the challenging problems, i.e., those categorized as Level 4 or Level 5 in the MATH dataset (Hendrycks et al., 2021) to generate additional 0.4 million query-response pairs. Finally, combined with 2.1M normal synthetic problems in stage 1, we obtain the 2.5M-instance Skywork-MathQA dataset. The rationale behind using these two stages and the experimental analysis of their impacts are discussed in Section 4.3.1. We further fine-tune the intermediate models on these hard synthetic problems to obtain the Skywork-Math model series, which exhibit strong mathematical reasoning abilities.

Remark It is worth noting that the accuracy of our utilized GPT-4 version on the MATH benchmark is approximately 50%, indicating that about half of our synthetic data in SkyworkMathQA dataset may contain minor errors in their results and intermediate reasoning process. However, scaling these SFT synthetic data reveals a clear positive trend in the performance of LLMs (ref. Figure 5). An interesting experimental phenomenon is that before reaching the upper bound performance of Skywork-Math 7B model series, data quantity seems to play a more important role than data quality.

- 4. Experiment

- 4.1. Experimental Setup

- 4.1.1. Evaluation Datasets

We primarily conduct our experiments on two benchmarks widely recognized for assessing mathematical reasoning capabilities. (1) GSM8K (Cobbe et al., 2021) comprises a collection of high-quality math word problems at the grade school level. It contains 1,319 test questions. Typically, the reasoning steps in GSM8K vary between two and eight, ultimately yielding an integer as the answer. (2) MATH (Hendrycks et al., 2021) contains 5,000 test questions, featuring math competition-level problems. The answers in GSM8K are integer, making it relatively easy for the regular expression matching program in evaluation frameworks to extract and verify answers. However the answers in MATH may contain complex mathematical formulas (e.g.,

√2,√3)). We have explored several evaluation benchmarks to assess the results on MATH (e.g., (GPT-4o, 2024; He et al., 2024; Shao et al., 2024; Yu et al., 2024)). Different evaluation

√2

2+

- 4 , (

benchmarks implement different regular expression rules to extract mathematical formulas, leading to significant performance variations among them (in some cases, there are up to 5% accuracy variations on MATH). In this paper, we adopt the same evaluation framework as in MetaMath (Yu et al., 2024) because it is widely used and provides strict and robust evaluation results using zero-shot and COT techniques.

###### 4.1.2. Pre-Trained Models

We utilize three publicly available top-performing 7B pre-trained LLMs in the Skywork-MathQA models to push the limit of mathematical reasoning abilities in small-scale LLMs. Our empirical results indicate that Skywork-MathQA 7B models even outperform the recently released 70B LLaMA-3 Instruct Model (AI@Meta, 2024) on the MATH benchmark.

- • LLaMA2-7B (Touvron et al., 2023) is a general-purpose LLM model that has demonstrated significant performance across various benchmarks. However, it exhibits limited mathematical reasoning abilities.
- • Mistral-7B (Jiang et al., 2023) is another general-purpose LLM model that exhibits strong reasoning abilities in math problem-solving and code generation.
- • DeepSeekMath-Base-7B (Shao et al., 2024) is a specialized LLM model tailored for mathematics reasoning. It stems from DeepSeek-Coder-Base-v1.5-7B (Guo et al., 2024) and has been further pre-trained on a mathematical corpus with 120 billion tokens. Due to this extended pre-training on massive math corpus, we observe a notable performance divergence between the specialized model and general-purpose LLM model (ref. Section 4.2.2).

###### 4.1.3. Implementation Details

We utilize the GPT-4 API with a temperature of 0.7 to generate query-response pairs in SkyworkMathQA dataset. To prevent data leakage, we evaluate the Skywork-Math models on the test examples of GSM8K and MATH with a 30-gram hit, as suggested by (Azerbayev et al., 2023). For all experiments, including ablations, Skywork-MathQA models are trained for 3 epochs. A global batch size of 32 is used along with the AdamW optimizer without weight decay. Following the original configurations of 7B pre-trained models, the learning rate is set to 2𝑒 − 5 for LLaMA2-7B and 2𝑒 − 6 for both Mistral-7B and DeekSeekMath-Base-7B. The learning rate warm-up ratio is 0.03. All experiments are conducted on 8 Nvidia A800 GPUs with 80G memory. For evaluation, we use the vLLM (Kwon et al., 2023) library to generate inference responses, using the same prompt as in the SFT stage described in Section 3. Unless otherwise noted, we set the maximum length of models to 2048 in both the model SFT stage and the evaluation stage. We employ a stringent criterion similar to that used in Metamath (Yu et al., 2024), achieving nearly 100% precision but at the cost of a relatively low recall rate. This approach results in several instances where correct responses from the model are mistakenly labeled as incorrect according to our criteria. Specific examples can be found in Appendix B.

- 4.2. Main Results

- 4.2.1. Comprehensive Performance Comparison with State-of-the-art Models

Table 1 presents the comparison of Skywork-Math model series with the state-of-the-art closedand open-source models on the test set of GSM8K and MATH benchmark to evaluate their math reasoning abilities. Because GPT-4-Turbo is a commercially closed-source model and cannot be fine-tuned to adhere to specific output formats, its responses are evaluated using a

Model #Params GSM8K(%) MATH(%) Closed-source models

GPT-3.5-Turbo (Peng et al., 2023) N/A 80.8 34.1 GPT-4-Turbo (Achiam et al., 2023) N/A 90.51 57.0 GPT-4 (Achiam et al., 2023) N/A 92.0 42.5 PaLM2 (Anil et al., 2023) 540B 80.7 34.3 Flan-PaLM2 (Anil et al., 2023) 540B 84.7 33.2 Minerva (Lewkowycz et al., 2022) 8B 16.2 18.1 Minerva (Lewkowycz et al., 2022) 62B 52.4 27.6 Minerva (Lewkowycz et al., 2022) 540B 58.8 33.6 ChatGLM3-32B-SFT-2312 (Xu et al., 2024) 32B 75.8 29.0

+RFT, DPO (Xu et al., 2024) 32B 82.6 40.6 Claude-3-Oppus (Anthropic, 2024) N/A 95.0 60.1

Open-source models (1-10B)

Baichuan-2 (Yang et al., 2023) 7B 24.5 5.6 LEMA-LLaMA2 (An et al., 2023) 7B 54.1 9.4 MetaMath (Yu et al., 2024) 7B 66.5 19.8 WizardMath-V1.1 (Luo et al., 2023) 7B 83.2 33.0 Xwin-Math-LLaMA (Li et al., 2024) 7B 82.6 40.6 Xwin-Math-Mistral (Li et al., 2024) 7B 89.2 43.7 Xwin-Math-Llemma (Li et al., 2024) 7B 84.2 47.2 MAmmoTH (Yue et al., 2023) 7B 53.6 31.5 InternLM2-Math (Ying et al., 2024) 7B 78.1 34.6 DeepSeekMath-Instruct (Shao et al., 2024) 7B 82.9 46.8 Skywork-Math-LLaMA2 (ours) 7B 72.9 47.7 Skywork-Math-Mistral (ours) 7B 83.9 51.2 Skywork-Math-DeepSeekMath (ours) 7B 81.5 49.9 LLaMA3-Instruct (AI@Meta, 2024) 8B 79.6 30.0

Open-source models (10-50B)

LLaMA2 Touvron et al. (2023) 13B 28.70 3.90 Baichuan-2 (Yang et al., 2023) 13B 52.8 10.1 MetaMath (Yu et al., 2024) 13B 72.3 22.4 Wizard-Math (Luo et al., 2023) 13B 63.9 14.0 MAmmoTH (Yue et al., 2023) 13B 62.0 34.2 LEMA-LLaMA2 (An et al., 2023) 13B 65.7 12.6 Xwin-Math (Li et al., 2024) 13B 88.1 44.9 InternLM2-Math (Ying et al., 2024) 20B 82.6 37.7 LLaMA2 Touvron et al. (2023) 34B 42.20 6.20 LLema (Azerbayev et al., 2023) 34B 51.5 25.0

Open-source models (50-70B)

WizardMath (Luo et al., 2023) 70B 81.6 22.7 MetaMath (Yu et al., 2024) 70B 82.3 22.6 LLaMA2 (Touvron et al., 2023) 70B 56.8 13.5 LEMA-LLaMA2 (An et al., 2023) 70B 83.5 25.0 MAmmoTH (Yue et al., 2023) 70B 76.9 41.8 LLaMA3-Instruct (AI@Meta, 2024) 70B 90.0 50.4 Xwin-Math (Li et al., 2024) 70B 90.6 52.8

- Table 1 | Summary of math reasoning performance of closed- and open-source LLM models in terms of accuracy (%). All results for open-source models are reported as top1 accuracy using only SFT techniques. Skywork-Math models employ zero-shot chain-of-thought (COT) evaluation framework as implemented in MetaMath (Yu et al., 2024). The best result in each block are highlighted in bold. GPT-4-Turbo is evaluated using the grading criteria with 4-shot COT prompting as implemented in (Zheng et al., 2023). Skywork-Math 7B models, using only synthetic SFT data, have achieved SOTA performance on MATH among models small than 10B parameters, even outperforming 70B LLM models and an early version of GPT-4.

grading criterion with 4-shot COT prompting as used in (Zheng et al., 2023). (1) For the MATH benchmark, our Skywork-Math model series have achieved the state-of-the-art performance among LLM models smaller than 10B parameters with only the SFT technique, even surpassing the an early version of GPT-4. These results indicate that strong math reasoning abilities can be injected during the SFT stage through the extensive and high-quality Skywork-MathQA dataset. Moreover, Skywork-Math 7B models achieve competitive accuracy with 70B LLM models, which suggests 7B common LLM models can possess the strong math reasoning abilities with sufficient SFT process. These results demonstrate the significant effectiveness of our proposed two-stage data synthesis and model SFT pipeline. (2) For the GSM8K benchmark, the Skywork-Math model series also achieve comparable performance with several state-of-the-art models. It is noteworthy that our Skywork-MathQA dataset contains no data referencing GSM8K. The characteristics of math word problem (GSK8K) and math competition-level problems (MATH) differ in their problem-answer formats and difficulty. We posit that the success can be attributed to the difficulty of the relatively easy problems in MATH (Level 1&2) being similar to those in GSM8K, and the knowledge learned from solving competition-level mathematical problems can be effectively transferred to math word problems.

###### 4.2.2. Scaling Laws in SFT on Mathematical Reasoning

In Figure 5, we illustrate the relationship between synthetic SFT dataset size and model performance on GSM8K and MATH. The curve clearly exhibits a scaling law relationship between the size of SFT data and model’s performance. Here are some in-depth observations:

Quantity Breeds Quality. To enhance the mathematical reasoning abilities in LLMs, increasing the quantity of synthetic data can significantly improve the quality of model performance. This scaling trend implies that, while SFT with a small amount of data could achieve decent results Zhou et al. (2023), utilizing a larger scale of synthetic SFT data can further improve math reasoning performance.

###### Diminishing Returns from Continual Pre-Training. The DeepSeekMath-Base (Shao et al.,

- 2024) 7B model, which has been continually pre-trained with 120B math-related tokens sourced from the web, initially demonstrates superior performance. However, as we increase the synthetic dataset size in the Skywork-MathQA dataset, this advantage diminishes and is eventually surpassed by the Mistral (Jiang et al., 2023) 7B base model. As the amount of SFT data increases, Skywork-Math-Mistral-7B and Skywork-Math-LLaMA2-7B catch up in performance to the Skywork-Math-DeepSeekMath-7B. This suggests that while specialized pre-training provides a strong initial boost, its benefits are not consistently scalable and can be matched by increasing the quantity of synthetic SFT data.

Effect of Problem Difficulty. The accuracy performance for Skywork-Math 7B model series significantly increases as the synthetic data size expands from 2.1M to 2.5M, corresponding to the stage 2 in our data synthesis pipeline. This performance improvement in the final stage of data scaling indicates that incorporating more complex problems— ranging from Level 3 to Level 5 in the MATH dataset—has a substantial positive impact on model performance. This finding underscores the importance of not only generating a large quantity of data but also including more challenging problems to push the limits of math reasoning abilities of LLM models. We will discuss this in more detail in Section 4.3.1.

- Figure 5 | The zero-shot top1 performance of Skywork-Math 7B model series improves significantly as we scale up the size of synthetic SFT data in the Skywork-MathQA dataset. There is a clear trend indicating that the model’s math reasoning quality increases substantially with increases in data quantity.

- 4.3. Experimental Analysis

- 4.3.1. Fine-Grained Analysis across Different Difficulty Levels

We explore model’s performance across various difficulty levels to analyze the internal relationship between data difficulty and LLM model’s capability. The difficulty level distribution

Difficulty Levels in MATH(%) Level-1 Level-2 Level-3 Level-4 Level-5 LLaMA2-7B 7.5K 17.85 8.39 4.77 3.05 0.91 Mistral-7B 7.5K 37.99 25.17 15.12 8.48 2.49 DeepSeekMath-7B 7.5K 64.07 46.76 37.84 24.63 10.73 LLaMA2-7B 2.1M 78.03 60.29 48.19 35.09 19.56 Mistral-7B 2.1M 80.78 66.33 55.53 41.52 21.45 DeepSeekMath-7B 2.1M 80.78 65.21 58.00 41.60 21.83 LLaMA2-7B 7.5k + 0.4M (hard) 63.16 43.96 34.39 24.46 10.20 Mistral-7B 7.5k + 0.4M (hard) 71.62 57.27 48.72 34.60 16.99 DeepSeekMath-7B 7.5k + 0.4M (hard) 81.01 61.97 51.90 37.07 18.05 LLaMA2-7B 2.1M + 0.4M (hard) 78.03 62.42 52.87 37.48 18.73 Mistral-7B 2.1M + 0.4M (hard) 83.52 67.56 60.65 44.89 25.08 DeepSeekMath-7B 2.1M + 0.4M (hard) 82.84 67.23 58.71 42.01 21.30 GPT-4-Turbo - 82.84 73.38 65.34 52.88 34.06

Base Model Dataset Size

- Table 2 | Accuracies (%) across difficulty levels (from Level-1 to Level-5) with three base models in Skywork-Math 7B model series before and after fine-tuning on stage 2 in the MATH benchmark. 7.5K data samples are randomly sampled from the Skywork-MathQA dataset. GPT-4-Turbo is evaluated using our designed grading criteria with 4-shot COT prompting. In stage 1, SkyworkMath 7B models significantly improve the performance on easy problems in MATH (Level 1&2) using 2.1M synthetic SFT data. In stage 2, Skywork-Math 7B models show significant improvements on hard problems in MATH (Level 3-5) using 2.5M synthetic SFT data.

of the training and test set in MATH is illustrated in Figure 6. We can find that the number of hard problems (Level 3-5) is much larger than that of easy problem (Level 1&2) in both training and test sets. This highlights the value of hard problems to improve the overall math reasoning performance.

| |2304<br><br>Train<br><br>Test| | | | | |
|---|---|---|---|---|---|---|
| |1592<br><br>1690| | | | | |
| |1348<br><br>1131<br><br>1214<br><br>1324| | | | | |
| |566<br><br>894| | | | | |
| |437| | | | | |
| | | | | | | |

2000

1500

#Instances

1000

500

In Table 2, we conduct comprehensive experiments with three pre-trained base LLM models in Skywork-Math 7B model series. We observed a significant increase in accuracy for easy problems (Level 1&2) when scaling the dataset size from 7.5K to 2.1M, even reaching accuracies comparable to GPT4-Turbo. However, the increase in accuracy for hard problems (Level 3-5) was relatively modest compared to GPT-4-Turbo. This could be due to the lack of high-quality responses in hard problems, motivating us to perform the stage 2 in our data synthesis pipeline to generate hard synthetic problems. After fine-tuning our Skywork-Math 7B model with additional 0.4M hard synthetic problems, we observe a further increase in model performance, particularly at Level-3 and Level-4 on MATH. For comparison, we conduct an experiment to fine-tune three base models in Skywork-Math 7B models using 0.4M hard synthetic problems along with the randomly sampled 7.5k problems. We notice that for hard problems (Level 3-5), base models fine-tuned on

0

1 2 3 4 5

Difficulty Level

Figure 6 | Difficulty level distribution of the training and test set in the MATH benchmark.

- Figure 7 | Performance of different base models in Skywork-Math 7B models with various data augmentation methods on GSM8K and MATH. "Mix" represents a combination of data generated by three augmentation methods detailed in Section 3.1. For this ablation study, we utilize 60K synthetic SFT data in the Skywork-MathQA dataset.

the "2.1M + 0.4M (hard)" data perform significantly better than those fine-tuned on the "7.5k + 0.4M (hard)" data. This supports the rationale that LLM models should acquire mathematical reasoning abilities progressively from easy to hard problems. More detailed experiments can be found in Appendix C. In addition to testing on different levels, we also conducted experiments on various math subjects, as detailed in Appendix D.

###### 4.3.2. Effect of Data Diversity

Diversity on Data Augmentation Methods. One dimension of diversity is the data augmentation methods. We select 60K synthetic data in the Skywork-Math dataset to study this problem. As shown in Figure 7, the "Mix" approach, a combination of synthetic data generated by three augmentation methods, achieves the highest performance. Therefore, we utilize the "mix" method to generate our Skywork-MathQA dataset. Moreover, the Xwin-style (Li et al., 2024) approach and the MetaMathQA-style (Yu et al., 2024) approach require extensive time for answer verification and two steps for data generation, respectively. For the consideration of efficiency, we utilize the Evol-style (Luo et al., 2023) approach as a major component of the synthetic data due to requiring fewer input and output tokens within LLM models. We also observe that the impact of the mix rate of augmentation methods is not significant on the GSM8K and MATH benchmarks. However, combining these data augmentation methods is crucial for enhancing the data diversity of the Sykwork-MathQA dataset. Detailed exploration of data mixtures with different data augmentation methods is left for future work.

Diversity of Seed Problems. Another dimension of diversity is the selection of seed problems. We construct two SFT datasets, each comprising 360K entries. The first dataset uses only the training set of MATH as the seed problems. The second dataset employs the diversity selection method introduced in Section 3.1, which includes a wide range of non-proving problems from multiple academic data sources and uses the diversity selection method to further ensure the diversity. As illustrated in Table 3, the improved diversity of seed problems in SFT data substantially enhances the math reasoning abilities in Skywork-Math models across three 7B base LLM models.

###### Base Model Diversity Selection MATH(%) GSM8K(%)

LLaMA2-7B ✗ 29.48 50.57 Mistral-7B ✗ 38.50 72.71 DeepSeekMath-7B ✗ 43.96 74.30

LLaMA2-7B ✓ 29.36 52.08 Mistral-7B ✓ 39.68 73.92 DeepSeekMath-7B ✓ 43.68 75.97

- Table 3 | Ablation studies with the diversity selection method on 360K data samples applied in stage 1 of the data synthesis pipeline. ✓ (✗) means that we evaluate w (w/o) the diversity selection method.

Base Model Dataset (Size) GSM8K(%) MATH(%)

LLaMA2-7B Random selection (1M) 60.35 37.76 LLaMA2-7B Random selection (1.5M) 66.87 40.52 LLaMA2-7B Selection with a verifier (1M) 62.77 36.40

Mistral-7B Random selection (1M) 77.79 44.56 Mistral-7B Random selection (1.5M) 80.36 45.86 Mistral-7B Selection with a verifier (1M) 77.26 43.04

- Table 4 | Comparisons of the model performance on GSM8K and MATH in terms of accuracy using random selection and selection with a verifier. All data samples are selected from the Skywork-MathQA dataset. Random selection on the math reasoning dataset is a simple but hardto-beat strategy. Without a carefully designed filtering strategy, it is non-trivial to outperform random selection.

###### 4.3.3. Data Selection with a Verifier

Since the accuracy of GPT-4 on MATH is around 50%, we can infer that approximately half of the data samples in the Skywork-MathQA dataset may not have the right solving processes and answers. To ensure the collection of high-quality data, it is a natural way to perform data selection with a verifier to filter out wrong responses. We first eliminate synthetic data entries that fail to align with the ground truth final answers. However, most data samples either lack the ground truth final answers or contain errors in intermediate reasoning steps. Therefore, we should design a more precise approach to ensure the entire solution is consistent with the ground truth. We fine-tune a Mistral-7B (Jiang et al., 2023) base model with few-shot prompting to verify if the reasoning paths and final answers are correct. Finally, we obtain approximately 1 million samples deemed correct by this fine-tuned verifier. With human verification of the results judged by the trained Mistral-7B verifier, it achieves an accuracy of approximately 80%. After implementing our filtering process, the fraction of correct data (80%) increases significantly compared to its original fraction (50%). As shown in Table 4, we present the results selected using the trained verifier in contrast to a random selection in the Skywork-Math dataset. We initially anticipated that, after filtering for correctness to obtain the 1M filtered dataset, the accuracies on GSM8K and MATH would range between 1M to 1.5M samples with random selection due to their quantitative relationship. However, the actual performance on the LLaMA2-7B and Mistral-7B models showed that the 1M filtered dataset performed even worse than the 1M dataset with random selection.

###### Base Model Dataset (Size) GSM8k(%) MATH(%)

LLaMA2-7B Random selection (360K) 52.08 29.36 LLaMA2-7B Selection with hard problems (360K) 54.36 36.68

Mistral-7B Random selection (360K) 73.92 39.68 Mistral-7B Selection with hard problems (360K) 76.42 40.20

DeepSeekMath-7B Random selection (360K) 75.97 43.68 DeepSeekMath-7B Selection with hard problems (360K) 75.74 44.48

- Table 5 | Comparisons of the model performance on GSM8K and MATH in terms of accuracy using random selection and our designed selection strategy with filtering for more hard problems. All data samples are selected from the Skywork-MathQA dataset. Our strategy consistently outperform random selection.

Model

GSM8K(%) MATH(%) English Chinese English Chinese

LLaMA3-8B + Skywork-MathQA 75.97 58.83 50.30 44.10 Mixtral-8x7B + Skywork-MathQA 83.93 72.71 51.40 48.02 Llemma-7B + Skywork-MathQA 66.03 50.72 40.08 37.42 Skywork-Math-LLaMA2-7B 72.86 50.34 47.66 38.38 Skywork-Math-Mistral-7B 83.93 69.75 51.22 48.34 Skywork-Math-DeepSeekMath-7B 81.50 73.69 49.88 48.22

- Table 6 | Results of bilingual language testing on GSM8K and MATH. Note that all models are fine-tuned on English data. The Chinese version of GSM8K and MATH are translated from their English counterparts using GPT-4. LLaMA3-8B, Mixtral-8x7B, Llemma-7B are fine-tuned on our Skywork-MathQA datasets. Our empirical results indicates that the strong math reasoning capabilities can be maintained between English and Chinese.

The experimental results align with the conclusion in DsDm (Engstrom et al., 2024). The data selection process on math reasoning is non-trivial and there exist multiple objectives to affect this data selection process. Our observation suggests that although the accuracy reaches as high as 80%, the difficulty level of the selected problems significantly decreases. The selection process improves the data quality but significantly decreases the difficulty level of problems, thereby negatively impacting the performance of LLMs. In order to filter out correct problems, the verifier model predominantly selects those problems with lower difficulty. To address the scarcity of hard problems in the filtered dataset, we further utilize GPT-4 with the COT prompt to pick out around 360K hard problems. Table 5 demonstrates that data selection with hard problems is effective, as all base models in the Skywork-Math models show improved performance on both the MATH and GSM8K benchmarks compared to their random selection counterparts.

###### 4.3.4. Can Math Reasoning Abilities Transfer Between Bilingual Language?

The common view holds that mathematical problems mainly consist of symbols and expressions, and the textual language used to state them is not crucial for understanding. To explore whether math reasoning abilities can transfer between bilingual languages, we translate the GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021) benchmarks from English to

Original Question: There were a total of 15 fish in the plate. After the kitten ate some, there were 10 fish left. How many fish did the kitten eat?

Question with Distractors: There are 3 kittens in the house. There were a total of 15 fish in the plate, including 10 carp and 5 belt fish. After the kittens ate some fish, there were still 10 fish left. How many fish did the kittens eat?

###### Distractors:

- 1. "There are 3 kittens in the house."
- 2. "Including 10 carp and 5 belt fish."

- Figure 8 | An example of an original question from GSM8K (Cobbe et al., 2021) and the same question with two distrators as implemented in CMATH (Wei et al., 2023a).

Chinese for bilingual language testing. It is important to note that all models are fine-tuned only on English data. As shown in Table 6, the overall math reasoning abilities are maintained between English and Chinese. There is a relatively small-scale performance degradation on MATH between English and Chinese, especially in Skywork-Math-Mistral-7B and SkyworkMath-DeepSeekMath-7B. However, there is a significant performance drop on GSM8K between English and Chinese, with up to a 20-point drop in Skywork-Math-LLaMA2-7B. Since GSM8K is grouped in the math word problem category, which requires more linguistic understanding, the degradation in accuracy is greater than that for MATH. Notably, Skywork-Math-DeepSeekMath-

- 7B performs well in both English and Chinese. We hypothesize the reason for this is that the 120B continual pre-training corpus in the DeepSeekMath-Base-7B model includes many Chinese sources, which improves its Chinese language understanding. These results highlight the challenges associated with language dependence in understanding and performing mathematical reasoning tasks.

###### 4.3.5. Can Math Reasoning Abilities Be Maintained in Robustness Tests?

As suggested in CMATH (Wei et al., 2023a), several open-sourced LLM models, except GPT-4Turbo, are vulnerable to robustness tests of math reasoning abilities influenced by distractors. To ascertain if models effectively comprehend the fundamental elements of mathematical word problems and their solutions, we inject each problem in GSM8K with 1-5 distractors as implemented in CMATH (Wei et al., 2023a). An example of two distractors is shown in Figure 8. As listed in Table 7, open-sourced fine-tuned LLM models are sensitive to the distractors injected into math word problems. Compared to the MetaMathQA SFT dataset, our proposed Skywork-MathQA dataset significantly improves robustness performance in GSM8K based on common pre-trained models, such as Mistral-7B and DeepSeekMath-7B. We hypothesize that the reason lies in the significantly larger size of the Skywork-MathQA dataset compared to the MetaMathQA dataset. The improved diversity of the Skywork-MathQA dataset can help the LLM models STF on it to better withstand robustness tests. However, GPT-4-Turbo consistently excludes interference information and focuses on the relevant information, thereby producing correct responses with even 5 distractors in GSM8K. These results suggest that most of open-source SFT models cannot truly understand the semantic information of math world problems but rather mechanically extract numbers from the sentence and calculate them. Effectively improving math reasoning abilities while maintaining robustness like GPT-4-Turbo is an important area for future exploration.

#Distractors in GSM8K 1 2 3 4 5

Model SFT Dataset (Size) GSM8K(%)

GPT-4-Turbo - 90.51 95.30 91.44 88.98 88.02 85.37 DeepSeekMath-7B-Instruct - 82.90 73.77 62.97 51.44 48.22 43.88

Mistral-7B MetaMathQA (395K) 79.08 70.10 56.80 48.95 46.01 38.51 DeepSeekMath-7B MetaMathQA (395K) 82.49 73.20 60.33 50.26 42.31 39.40

- LLaMA2-13B MetaMathQA (395K) 70.96 65.86 50.25 41.21 33.73 31.64 Llemma-7B Skywork-MathQA (2.5M) 66.03 61.40 52.90 46.06 40.38 38.21

- LLaMA3-8B Skywork-MathQA (2.5M) 75.97 75.14 70.91 65.35 62.43 55.82 Mixtral-8x7B Skywork-MathQA (2.5M) 83.93 84.19 78.21 73.36 68.93 66.57 Skywork-Math-LLaMA2-7B Skywork-MathQA (2.5M) 72.86 64.72 58.56 54.20 49.41 44.63 Skywork-Math-Mistral-7B Skywork-MathQA (2.5M) 83.93 83.16 75.19 72.57 66.42 67.01 Skywork-Math-DeepSeekMath-7B Skywork-MathQA (2.5M) 81.50 78.35 72.54 64.70 59.17 57.31

- Table 7 | Performance against the number of the distractors added to the original GSM8K dataset. GPT-4 demonstrate remarkable robustness, while other models fail.

Model Data Synthesis Pipeline (Size) GSM8K(%) MATH(%) Mistral-7B - 50.00 12.70

- Skywork-Math-Mistral-7B Stage 1 (2.1M) 83.25 49.10
- Skywork-Math-Mistral-7B Stage 2 (2.5M) 83.93 51.22 Mixtral-8×7B - 74.40 28.40

- Mixtral-8×7B + Skywork-MathQA Stage 1 (2.1M) 85.06 50.02
- Mixtral-8×7B + Skywork-MathQA Stage 2 (2.5M) 83.93 51.40

- Table 8 | Performance comparison between the dense (Skywork-Math-Mistral-7B) and sparse MOE (Mixtral-8×7B) LLM model. We fine-tune the corresponding base models using the Skywork-MathQA dataset in both stage 1 and stage 2 of the data synthesis pipeline.

###### 4.3.6. Ablation Studies Between Sparse MOE and Dense Models

Recent advancements have witnessed the rapid development of sparse MOE models (DeepSeekAI, 2024). To evaluate the generalization capability of our Skywork-MathQA dataset across both sparse MOE and dense models, we select commonly used dense (Skywork-Math-Mistral7B (Jiang et al., 2023)) and sparse MOE (Mixtral-8×7B (Jiang et al., 2024a)) models as the pre-trained LLM base models. We conduct experiments using the Skywork-MathQA dataset in both stage 1 and stage 2. As shown in Table 8, the results confirm strong generalization across different types of LLM models. However, the Mixtral-8×7B fine-tuned on the SkyworkMathQA dataset does not show superior performance compared with its dense counterpart. The Mixtral-8×7B and Skywork-Math-Mistral-7B almost exhibit almost identical performance on GSM8K and MATH. We posit the reason is that the sparse MoE model, due to its mixtureof-expert architecture, may not significantly improve the performance on the specific task (i.e., the math reasoning task), but can better handle task-specific knowledge without compromising performance on other tasks (Wei et al., 2024; Xue et al., 2024).

###### 4.3.7. Effect of Data Leakage

Though we never use the test data from MATH (Hendrycks et al., 2021) or GSM8K (Cobbe et al., 2021) for fine-tuning LLM models, we utilize GPT-4 (Achiam et al., 2023) to synthesize

Question in Skywork-MathQA: Let 𝑥 and 𝑦 be nonzero real numbers such that

(3 − 4𝑖)(𝑥 + 𝑦𝑖) is pure imaginary. Find 𝑥/𝑦.

Question in the MATH test set: Let 𝑥 and 𝑦 be nonzero real numbers such that

𝑥𝑦(𝑥2 − 𝑦2) = 𝑥2 + 𝑦2. Find the minimum value of 𝑥2 + 𝑦2.

- Figure 9 | An example of the math questions that are completely different but get filtered by a 10-gram filter due to a common condition.

data, which may inadvertently contaminate our synthetic dataset with elements from the test data in the evaluation benchmarks. Therefore, we follow a standard 30-gram filtering process (Azerbayev et al., 2023) on test data of the corresponding benchmark to circumvent the data leakage of the Skywork-MathQA dataset. We filter out approximately 6K samples for the test set of MATH and none for GSM8K.

To assess the impact of the n-gram filter, we tested a stricter 10-gram filter, which is much more stringent than the 30-gram filter. We observe that the 10-gram filter removes a lot of data that has little relation to the data in the test set of MATH. As illustrated in Figure 9, there are two entirely unrelated examples in our synthetic Skywork-MathQA dataset and the test set of MATH. It is evident that "Let 𝑥 and 𝑦 be nonzero real numbers such that" is a very common condition in math problems. The 10-gram filter results in the removal of many completely unrelated problems in the synthetic data. Consequently, we use the 30-gram filter instead of the 10-gram filter to produce the Skywork-MathQA dataset.

We further conduct experiments to quantitatively analyze the difference between the 30gram and 10-gram filter using our Skywork-MathQA dataset in stage 1. Our Skywork-MathQA dataset, which has already been filtered using the 30-gram filter, consists of 2.16M instances. After applying 10-gram filtering, we have 2.10M instances. The filtered-out data, meaning the data samples present in the 2.16 million instances but not in the 2.10 million instances, consists of 60K samples. For a fair comparison, we also randomly select 60K data samples from the Skywork-MathQA dataset. The results of accuracies on the MATH benchmark are reported in Table 9. The observations are as follows: (1) The 10-gram filter is too strict, leading to the removal of some specific types of problems in the math benchmark (ref. Figure 9), which results in performance degradation. (2) The 60K randomly sampled data is much more useful than the 60K filtered-out data for Skywork-Math-LLaMA2-7B and Skywork-Math-Mistral-7B. The experimental results are reasonable, as the diversity in the randomly selected 60K data is much greater than that in the filtered 60K data. (3) The performance of DeepSeekMath-7B after SFT with the 2.10M dataset is significantly worse than with the 2.16M dataset. The filtered 60K dataset performs even better than the randomly selected 60K dataset. We believe this is because Skywork-Math-DeepSeekMath-7B may focus on the types of problems present in the filtered 60K data. Its base model, DeepSeekMath-Base-7B (Shao et al., 2024), is a specialized math LLM model continually pre-trained on a large collection of math data that matches some of the types in these filtered 60K problems.

###### Model Filter Method (size) MATH(%)

Skywork-Math-LLaMA2-7B 30-gram (2.16M) 45.56 Skywork-Math-LLaMA2-7B 10-gram (2.10M) 37.54 Skywork-Math-LLaMA2-7B Filter-out (60K) 10.76 Skywork-Math-LLaMA2-7B Random selection (60K) 15.16

Skywork-Math-Mistral-7B 30-gram (2.16M) 49.10 Skywork-Math-Mistral-7B 10-gram (2.10M) 40.78 Skywork-Math-Mistral-7B Filter-out (60K) 22.32 Skywork-Math-Mistral-7B Random selection (60K) 27.84

Skywork-Math-DeepSeekMath-7B 30-gram (2.16M) 48.64 Skywork-Math-DeepSeekMath-7B 10-gram (2.10M) 36.68 Skywork-Math-DeepSeekMath-7B Filter-out (60K) 40.64 Skywork-Math-DeepSeekMath-7B Random selection (60K) 39.86

- Table 9 | Accuracies (%) on MATH for the Skywork-Math models using the 30-gram and 10-gram filter methods. "Filter-out" indicates samples present in the 30-gram filter method but not in the 10-gram filter method. For a fair comparison, we also randomly sampled 60K data points from our Skywork-MathQA dataset.

Model Model Maximum Length MATH(%) GSM8k(%)

Skywork-Math-LLaMA2-7B 512 44.06 67.85 Skywork-Math-LLaMA2-7B 2048 47.66 72.86

Skywork-Math-Mistral-7B 512 50.56 82.41 Skywork-Math-Mistral-7B 2048 51.22 83.93

Skywork-Math-DeepSeekMath-7B 512 48.28 80.52 Skywork-Math-DeepSeekMath-7B 2048 49.88 81.50

- Table 10 | Comparison of performance in Skywork-Math models using the 2.5M-instacne Skywork-MathQA dataset with different maximum model lengths.

###### 4.3.8. Effect of Model Maximum Length

As the difficulty level of problems increases, the length of reasoning steps typically becomes longer, especially with those generated by LLMs. If the model’s maximum length is too small, the response may be truncated. In our synthetic Skywork-MathQA SFT dataset, around 130K problems exceed 512 tokens. Therefore, we set the maximum length of models to 2048 tokens in both the SFT stage and the evaluation stage. As shown in Table 10, increasing the model’s maximum length leads to improved performance, indicating that 7B models can comprehend and execute long reasoning processes.

#### 5. Closing Remarks and Future Directions

We study how to empower mathematical reasoning abilities for common 7B pre-trained LLM models. We propose the Skywork-MathQA dataset, consisting of 2.5 million diverse and highquality SFT instances, implemented through our novel two-stage data synthesis pipeline. We introduce Skywork-Math model series, demonstrating that common small-scale 7B language models can stimulate strong mathematical reasoning ability using only synthetic SFT data. Skywork-Math models achieve state-of-the-art accuracy among models smaller than 10B parameters using only synthetic SFT data, surpassing 70B LLM models and an early version of GPT-4 on MATH. These results suggest that the data scaling law for mathematical reasoning in LLM models remains significant and promising. Notably, this research provides several valuable insights and practical takeaways to advance our understanding of the capabilities and limitations of LLMs in mathematical reasoning.

Finally, we present two promising future directions for this work:

Code-Integrated Math Reasoning. Complex scientific calculations are essential for tackling difficult mathematical problems. By embedding executable code, LLMs can dynamically generate and execute code to solve intricate mathematical problems, ensuring higher accuracy and robustness. Some recent works have already been proposed to translate mathematical problems into executable code (Gou et al., 2023; Toshniwal et al., 2024). However, code cannot always be generated correctly on the first attempt. Therefore, iteratively utilizing code to solve challenging math problems is a promising direction for future research.

More General Reasoning Tasks. Reasoning is a crucial ability for complex problem-solving. Beyond mathematical reasoning, there are many other important reasoning tasks, such as logical reasoning, causal reasoning, and commonsense reasoning (Sun et al., 2023). It is intriguing to explore how our proposed method can be applied to these more general reasoning tasks.

#### 6. Acknowledgements

We would like to thank Longhui Yu (the author of MetaMath) and Chen Li (the author of Xwin-Math) for their valuable discussions. Our deepest gratitude goes to our boss, Yahui Zhou, whose financial assistance in scaling supervised fine-tuning data and providing access to GPU computational resources was indispensable for the successful completion of this study.

#### References

- J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt,

- S. Altman, S. Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

AI@Meta. Llama 3 model card. 2024. URL https://github.com/meta-llama/llama3/bl ob/main/MODEL_CARD.md.

- E. Almazrouei, H. Alobeidli, A. Alshamsi, A. Cappelli, R. Cojocaru, M. Debbah, É. Goffinet, D. Hesslow, J. Launay, Q. Malartic, et al. The falcon series of open language models. arXiv preprint arXiv:2311.16867, 2023.

- S. An, Z. Ma, Z. Lin, N. Zheng, J. Lou, and W. Chen. Learning from mistakes makes LLM better reasoner. CoRR, abs/2310.20689, 2023. doi: 10.48550/ARXIV.2310.20689. URL https://doi.org/10.48550/arXiv.2310.20689.

- R. Anil, A. M. Dai, O. Firat, M. Johnson, D. Lepikhin, A. Passos, S. Shakeri, E. Taropa, P. Bailey, Z. Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.

Anthropic. The claude 3 model family: Opus, sonnet, haiku. 2024. URL https://www-cdn.a nthropic.com/de8ba9b01c9ab7cbabf5c33b80b7bbc618857627/Model_Card_Claud e_3.pdf.

D. Arora, H. G. Singh, et al. Have llms advanced enough? a challenging problem solving benchmark for large language models. arXiv preprint arXiv:2305.15074, 2023.

Z. Azerbayev, H. Schoelkopf, K. Paster, M. D. Santos, S. McAleer, A. Jiang, J. Deng, S. Biderman, and S. Welleck. Llemma: An open language model for mathematics. In The 3rd Workshop on Mathematical Reasoning and AI at NeurIPS’23, 2023. URL https://openreview.net /forum?id=0QHZrCWCH0.

Y. Bai, A. Jones, K. Ndousse, A. Askell, A. Chen, N. DasSarma, D. Drain, S. Fort, D. Ganguli, T. Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Y. Bengio, J. Louradour, R. Collobert, and J. Weston. Curriculum learning. In Proceedings of the 26th annual international conference on machine learning, pages 41–48, 2009.

T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam,

- G. Sastry, A. Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

- Y. Cao, Y. Kang, C. Wang, and L. Sun. Instruction mining: When data mining meets large language model finetuning. arXiv preprint arXiv, 2307, 2023.

- S. Casper, X. Davies, C. Shi, T. K. Gilbert, J. Scheurer, J. Rando, R. Freedman, T. Korbak, D. Lindner, P. Freire, et al. Open problems and fundamental limitations of reinforcement learning from human feedback. arXiv preprint arXiv:2307.15217, 2023.

- W. Chen, X. Ma, X. Wang, and W. W. Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. CoRR, abs/2211.12588, 2022. URL https://doi.org/10.48550/arXiv.2211.12588.

- W.-L. Chiang, Z. Li, Z. Lin, Y. Sheng, Z. Wu, H. Zhang, L. Zheng, S. Zhuang, Y. Zhuang, J. E. Gonzalez, et al. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, march 2023. URL https://lmsys. org/blog/2023-03-30-vicuna, 3(5), 2023.

- K. Cobbe, V. Kosaraju, M. Bavarian, J. Hilton, R. Nakano, C. Hesse, and J. Schulman. Training verifiers to solve math word problems. CoRR, abs/2110.14168, 2021. URL https://arxiv. org/abs/2110.14168.

DeepSeek-AI. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model, 2024.

- L. Engstrom, A. Feldmann, and A. Madry. Dsdm: Model-aware dataset selection with datamodels. arXiv preprint arXiv:2401.12926, 2024.

- G. Gendron, Q. Bao, M. Witbrock, and G. Dobbie. Large language models are not strong abstract reasoners yet. In ICLR 2024 Workshop: How Far Are We From AGI, 2024. URL https://openreview.net/forum?id=Pc0fPGip78.

- Z. Gou, Z. Shao, Y. Gong, Y. Yang, M. Huang, N. Duan, W. Chen, et al. Tora: A tool-integrated reasoning agent for mathematical problem solving. arXiv preprint arXiv:2309.17452, 2023.

##### GPT-4o. Gpt-4o simple evals, 2024. URL https://github.com/openai/simple-evals.

- S. Gunasekar, Y. Zhang, J. Aneja, C. C. T. Mendes, A. Del Giorno, S. Gopi, M. Javaheripi, P. Kauffmann, G. de Rosa, O. Saarikivi, et al. Textbooks are all you need. arXiv preprint arXiv:2306.11644, 2023.

D. Guo, Q. Zhu, D. Yang, Z. Xie, K. Dong, W. Zhang, G. Chen, X. Bi, Y. Wu, Y. Li, et al. Deepseekcoder: When the large language model meets programming–the rise of code intelligence. arXiv preprint arXiv:2401.14196, 2024.

- C. He, R. Luo, Y. Bai, S. Hu, Z. L. Thai, J. Shen, J. Hu, X. Han, Y. Huang, Y. Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024.

- D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. URL https://openreview.net/forum?id=7Bywt2mQsCe.

J. Huang and K. C.-C. Chang. Towards reasoning in large language models: A survey. arXiv preprint arXiv:2212.10403, 2022.

A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. l. Casas, F. Bressand,

- G. Lengyel, G. Lample, L. Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. d. l. Casas, E. B. Hanna, F. Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024a.

- W. Jiang, H. Shi, L. Yu, Z. Liu, Y. Zhang, Z. Li, and J. Kwok. Forward-backward reasoning in large language models for mathematical verification, 2024b. URL https://openreview.n et/forum?id=GhYXocT75t.

- J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

- W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626, 2023.

- Y. Lan, L. Wang, Q. Zhang, Y. Lan, B. T. Dai, Y. Wang, D. Zhang, and E.-P. Lim. Mwptoolkit: An open-source framework for deep learning-based math word problem solvers. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 13188–13190, 2022.

A. Lewkowycz, A. J. Andreassen, D. Dohan, E. Dyer, H. Michalewski, V. V. Ramasesh, A. Slone, C. Anil, I. Schlag, T. Gutman-Solo, Y. Wu, B. Neyshabur, G. Gur-Ari, and V. Misra. Solving quantitative reasoning problems with language models. In A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, editors, Advances in Neural Information Processing Systems, 2022. URL https: //openreview.net/forum?id=IFXTZERXdM7.

C. Li, W. Wang, J. Hu, Y. Wei, N. Zheng, H. Hu, Z. Zhang, and H. Peng. Common 7b language models already possess strong math capabilities. arXiv preprint arXiv:2403.04706, 2024.

M. Li, Y. Zhang, Z. Li, J. Chen, L. Chen, N. Cheng, J. Wang, T. Zhou, and J. Xiao. From quantity to quality: Boosting llm performance with self-guided data selection for instruction tuning. CoRR, abs/2308.12032, 2023. URL https://doi.org/10.48550/arXiv.2308.12032.

P. Lu, B. Peng, H. Cheng, M. Galley, K.-W. Chang, Y. N. Wu, S.-C. Zhu, and J. Gao. Chameleon: Plug-and-play compositional reasoning with large language models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openrevi ew.net/forum?id=HtqnVSCj3q.

H. Luo, Q. Sun, C. Xu, P. Zhao, J. Lou, C. Tao, X. Geng, Q. Lin, S. Chen, and D. Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. CoRR, abs/2308.09583, 2023. URL https://doi.org/10.48550/arXiv.2 308.09583.

- X. Ni, Y. Gong, Z. Gou, Y. Shen, Y. Yang, N. Duan, and W. Chen. Exploring the mystery of influential data for mathematical reasoning. arXiv preprint arXiv:2404.01067, 2024.

- K. Paster, M. D. Santos, Z. Azerbayev, and J. Ba. Openwebmath: An open dataset of high-quality mathematical web text. In The Twelfth International Conference on Learning Representations,

##### 2024. URL https://openreview.net/forum?id=jKHmjlpViu.

A. Peng, M. Wu, J. Allard, L. Kilpatrick, and S. Heidel. Gpt-3.5 turbo fine-tuning and api updates. 2023.

R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.

D. Saxton, E. Grefenstette, F. Hill, and P. Kohli. Analysing mathematical reasoning abilities of neural models. arXiv preprint arXiv:1904.01557, 2019.

- T. L. Scao, A. Fan, C. Akiki, E. Pavlick, S. Ilic, D. Hesslow, R. Castagné, A. S. Luccioni, F. Yvon,

- M. Gallé, J. Tow, A. M. Rush, S. Biderman, A. Webson, P. S. Ammanamanchi, T. Wang, B. Sagot,

- N. Muennighoff, A. V. del Moral, O. Ruwase, R. Bawden, S. Bekman, A. McMillan-Major,

- I. Beltagy, H. Nguyen, L. Saulnier, S. Tan, P. O. Suarez, V. Sanh, H. Laurençon, Y. Jernite,
- J. Launay, M. Mitchell, C. Raffel, A. Gokaslan, A. Simhi, A. Soroa, A. F. Aji, A. Alfassy, A. Rogers, A. K. Nitzav, C. Xu, C. Mou, C. Emezue, C. Klamm, C. Leong, D. van Strien, D. I. Adelani, and et al. Bloom: A 176b-parameter open-access multilingual language model. CoRR, abs/2211.05100, 2022. URL https://doi.org/10.48550/arXiv.2211.05100.

- O. Sener and S. Savarese. Active learning for convolutional neural networks: A core-set approach. arXiv preprint arXiv:1708.00489, 2017.

Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, M. Zhang, Y. Li, Y. Wu, and D. Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

T. Shen, R. Jin, Y. Huang, C. Liu, W. Dong, Z. Guo, X. Wu, Y. Liu, and D. Xiong. Large language model alignment: A survey. arXiv preprint arXiv:2309.15025, 2023.

- P. Soviany, R. T. Ionescu, P. Rota, and N. Sebe. Curriculum learning: A survey. International Journal of Computer Vision, 130(6):1526–1565, 2022.

J. Sun, C. Zheng, E. Xie, Z. Liu, R. Chu, J. Qiu, J. Xu, M. Ding, H. Li, M. Geng, et al. A survey of reasoning with foundation models. arXiv preprint arXiv:2312.11562, 2023.

- R. Taori, I. Gulrajani, T. Zhang, Y. Dubois, X. Li, C. Guestrin, P. Liang, and T. B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab /stanford_alpaca, 2023.
- S. Toshniwal, I. Moshkov, S. Narenthiran, D. Gitman, F. Jia, and I. Gitman. Openmathinstruct-1: A 1.8 million math instruction tuning dataset. arXiv preprint arXiv:2402.10176, 2024.

H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. Canton-Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M.-A. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan,

- I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and

T. Scialom. Llama 2: Open foundation and fine-tuned chat models. CoRR, abs/2307.09288,

2023. URL https://doi.org/10.48550/arXiv.2307.09288.

X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, S. Narang, A. Chowdhery, and D. Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.

X. Wang, Z. Hu, P. Lu, Y. Zhu, J. Zhang, S. Subramaniam, A. R. Loomba, S. Zhang, Y. Sun, and W. Wang. Scibench: Evaluating college-level scientific problem-solving abilities of large language models, 2024. URL https://openreview.net/forum?id=u6jbcaCHqO.

- J. Wei, Y. Tay, R. Bommasani, C. Raffel, B. Zoph, S. Borgeaud, D. Yogatama, M. Bosma, D. Zhou, D. Metzler, et al. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682, 2022a.

- J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. Chainof-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022b.

- T. Wei, J. Luan, W. Liu, S. Dong, and B. Wang. Cmath: can your language model pass chinese elementary school math test? arXiv preprint arXiv:2306.16636, 2023a.

T. Wei, L. Zhao, L. Zhang, B. Zhu, L. Wang, H. Yang, B. Li, C. Cheng, W. Lü, R. Hu, et al. Skywork: A more open bilingual foundation model. arXiv preprint arXiv:2310.19341, 2023b.

T. Wei, B. Zhu, L. Zhao, C. Cheng, B. Li, W. Lü, P. Cheng, J. Zhang, X. Zhang, L. Zeng, et al. Skywork-moe: A deep dive into training techniques for mixture-of-experts language models. arXiv preprint arXiv:2406.06563, 2024.

- Y. Weng, M. Zhu, F. Xia, B. Li, S. He, S. Liu, B. Sun, K. Liu, and J. Zhao. Large language models are better reasoners with self-verification. arXiv preprint arXiv:2212.09561, 2022.

- Z. Wu, L. Qiu, A. Ross, E. Akyürek, B. Chen, B. Wang, N. Kim, J. Andreas, and Y. Kim. Reasoning or reciting? exploring the capabilities and limitations of language models through counterfactual tasks. CoRR, abs/2307.02477, 2023. URL https://doi.org/10.48550/arX iv.2307.02477.

C. Xu, Q. Sun, K. Zheng, X. Geng, P. Zhao, J. Feng, C. Tao, and D. Jiang. Wizardlm: Empowering large language models to follow complex instructions. CoRR, abs/2304.12244, 2023. URL https://doi.org/10.48550/arXiv.2304.12244.

- Y. Xu, X. Liu, X. Liu, Z. Hou, Y. Li, X. Zhang, Z. Wang, A. Zeng, Z. Du, W. Zhao, et al. Chatglmmath: Improving math problem-solving in large language models with a self-critique pipeline. arXiv preprint arXiv:2404.02893, 2024.

F. Xue, Z. Zheng, Y. Fu, J. Ni, Z. Zheng, W. Zhou, and Y. You. Openmoe: An early effort on open mixture-of-experts language models. arXiv preprint arXiv:2402.01739, 2024.

- A. Yang, B. Xiao, B. Wang, B. Zhang, C. Bian, C. Yin, C. Lv, D. Pan, D. Wang, D. Yan, et al. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305, 2023.

H. Ying, S. Zhang, L. Li, Z. Zhou, Y. Shao, Z. Fei, Y. Ma, J. Hong, K. Liu, Z. Wang, et al. Internlm-math: Open math large language models toward verifiable reasoning. arXiv preprint arXiv:2402.06332, 2024.

- L. Yu, W. Jiang, H. Shi, J. YU, Z. Liu, Y. Zhang, J. Kwok, Z. Li, A. Weller, and W. Liu. Metamath: Bootstrap your own mathematical questions for large language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview

.net/forum?id=N8N0hgNDRt.

- Z. Yuan, H. Yuan, C. Tan, W. Wang, and S. Huang. How well do large language models perform in arithmetic tasks? CoRR, abs/2304.02015, 2023. URL https://doi.org/10.48550/arX iv.2304.02015.

- X. Yue, X. Qu, G. Zhang, Y. Fu, W. Huang, H. Sun, Y. Su, and W. Chen. Mammoth: Building math generalist models through hybrid instruction tuning. CoRR, abs/2309.05653, 2023. URL https://doi.org/10.48550/arXiv.2309.05653.

- B. Zhang, Z. Liu, C. Cherry, and O. Firat. When scaling meets LLM finetuning: The effect of data, model and finetuning method. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=5HCnKDeTws.

- C. Zheng, Z. Liu, E. Xie, Z. Li, and Y. Li. Progressive-hint prompting improves reasoning in large language models. arXiv preprint arXiv:2304.09797, 2023.

W. Zhong, R. Cui, Y. Guo, Y. Liang, S. Lu, Y. Wang, A. Saied, W. Chen, and N. Duan. Agieval: A human-centric benchmark for evaluating foundation models. arXiv preprint arXiv:2304.06364, 2023.

C. Zhou, P. Liu, P. Xu, S. Iyer, J. Sun, Y. Mao, X. Ma, A. Efrat, P. Yu, L. Yu, S. Zhang, G. Ghosh, M. Lewis, L. Zettlemoyer, and O. Levy. Lima: Less is more for alignment. CoRR, abs/2305.11206, 2023. URL https://doi.org/10.48550/arXiv.2305.11206.

#### A. Illustrations of Three Different Data Augmentation Methods

We present three specific examples using the corresponding augmentation styles introduced in Section 3.1. We use the same query to investigate the differences in the response. Overall, the differences among these three methods are nuanced, but combining them is crucial to enhance the diversity of the Skywork-MathQA dataset (ref. Section 4.3.2). In Figure 10, the MetaMathQAstyle data is answer-focused and maintains a coherent solving process. Figure 11 illustrates the Evol-style data, which provides a more detailed solution and includes extensive text to describe the problem-solving process. Figure 12 presents the Xwin-style response with a more detailed calculation process.

Example A.1: MetaMathQA-Style Query-Response Pair Query: If 𝑥 is a positive number such that

√

√

√

√

15𝑥 = 15, find all possible values for 𝑥.

8𝑥 ·

10𝑥 ·

3𝑥 ·

Response: Let’s think step by step. We start with the equation given: √8𝑥 ·

√10𝑥 ·

√15𝑥 = 15

√3𝑥 ·

√10𝑥 ·

First, simplify the left-hand side using the property of square roots: √8𝑥 ·

√3𝑥 ·

√15𝑥 = √︁(8𝑥) · (10𝑥) · (3𝑥) · (15𝑥) Next, multiply the expressions under the square root: (8𝑥) · (10𝑥) · (3𝑥) · (15𝑥) = 8 · 10 · 3 · 15 · 𝑥4 We calculate the numerical product:

8 · 10 · 3 · 15 = 8 · 10 · 45 = 8 · 450 = 3600 Thus, the expression simplifies to:

√

3600𝑥4 = 15 Now, rewrite the expression inside the square root:

√

√

3600𝑥4 = √3600 ·

𝑥4

√

Calculate the square roots: √3600 = 60 and

𝑥4 = 𝑥2

Thus, the equation becomes: 60𝑥2 = 15 Solve for 𝑥2: 𝑥2 = 1560 = 14 Take the positive square root of both sides (since 𝑥 is positive): 𝑥 = √︃

1 4 = 12

The answer is 12.

- Figure 10 | An example of data formatted in the MetaMathQA-style.

Example A.2: Evol-Style Query-Response Pair Query: Suppose 𝑥 represents a positive quantity, and it satisfies the following mathematical relationship: √

###### √

###### √

###### √

15𝑥 = 15. Determine every viable value of 𝑥 that conforms to this equation.

8𝑥 ×

10𝑥 ×

3𝑥 ×

Response: We have the equation

###### √

###### √

###### √

###### √

8𝑥 ×

10𝑥 ×

3𝑥 ×

15𝑥 = 15.

We can simplify the left-hand side by multiplying the square roots together. Since the product of square roots is equal to the square root of the product of the values under the square roots (as long as the quantities under the square roots are positive, which they are since 𝑥 is positive), we have:√8𝑥 · 10𝑥 · 3𝑥 · 15𝑥 = 15. We can then multiply the numbers under the square root together:

###### √

8 · 10 · 3 · 15 · 𝑥4 = 15. Next, we simplify the multiplication under the square root:

###### √

###### √

3600 · 𝑥4. Notice that 3600 = 602, so we can simplify the square root as follows:

8 · 10 · 3 · 15 · 𝑥4 =

√︁3600 · 𝑥4 = √︁602 · 𝑥4. Using the property of the square root that

###### √

𝑎2 = 𝑎 for any positive 𝑎, we get:

√︁602 · 𝑥4 = 60𝑥2.

Now we can equate the simplified square root expression to the right-hand side of the original equation:60𝑥2 = 15. To solve for 𝑥2, we divide both sides of the equation by 60: 𝑥2 = 6015. Simplify the fraction: 𝑥2 = 14. To solve for 𝑥, we take the square root of both sides, remembering that we’re looking for the positive value of 𝑥 (since 𝑥 represents a positive quantity):𝑥 = √︃

4. The square root of 14 is 12, so we have:𝑥 = 12. The answer is 12.

1

- Figure 11 | An example of data formatted in the Evol-style.

Example A.3: Xwin-Style Query-Response Pair Query: If 𝑥 is a positive number such that

√

√

√

√

15𝑥 = 15, find all possible values for 𝑥.

8𝑥 ·

10𝑥 ·

3𝑥 ·

√10𝑥 ·

√15𝑥 = 15

Response: Given the equation: √8𝑥 ·

√3𝑥 ·

√𝑏 = √𝑎𝑏, so we can

First, simplify the expression on the left side. We know that √𝑎 ·

combine the square roots: √8𝑥 · 10𝑥 · 3𝑥 · 15𝑥 Next, calculate the product inside the square root: 8𝑥 · 10𝑥 · 3𝑥 · 15𝑥 = 8 · 10 · 3 · 15 · 𝑥4 Simplify the constants: 8 · 10 = 80, 80 · 3 = 240, 240 · 15 = 3600. So we have: 8𝑥 · 10𝑥 · 3𝑥 · 15𝑥 = 3600𝑥4 Therefore, the equation becomes:

√

√

√

√

3600𝑥4 = √3600·

𝑥4 = 𝑥2,

3600𝑥4 = 24 Since

𝑥4, and

we get: √3600 · 𝑥2 = 15 Isolate 𝑥2 by dividing both sides by √3600: 𝑥2 = √360015 Simplify √3600: √3600 = 60 So: 𝑥2 = 6015 = 14 Finally, solve for 𝑥: 𝑥 = √︃

1 4

Upon checking if further simplification is possible, we find: 𝑥 = 12. The answer is 12.

- Figure 12 | An example of data formatted in the Xwin-style.

#### B. Case Studies with Correct Answers Presented in Incorrect Formats

- • Different formats of the final answer but with the same value.

Ground Truth: 0.24 Response: ...The answer is 24%

Ground Truth: √2,√3 Response: ...The answer is √3,√2

√2 4

Ground Truth: 2+

√2 4

Response: ...The answer is 21 +

Ground Truth: \\text{odd} Response: ...The answer is \"odd\".

- • Unexpected format for presenting the final answer, such as rephrasing the prefix "\nThe answer is " or including extra words before, in, or after "\nThe answer is ".

- Ground Truth: 1, 2 Response: ...The correct answer is 1, 2

Ground Truth: 19 Response: ...The correct answer is 19, but this is based on an assumption that ...

- Ground Truth: 2 Response: ...The value of x is 2

Ground Truth: 24.01 Response: ...The answer is 𝑥 = 2401100 = 24.01

#### C. Performance Analysis in Stage 2 of the Data Synthesis pipeline

- Table 11 illustrates the relationship between data size in stage 2 of the data synthesis pipeline and the model performance. As we generate more hard synthetic problems in stage 2 of our data synthesis pipeline, the fine-tuned LLM models show gradual improvement in handling hard problems (Level 3-5) on the MATH benchmark.

Difficulty Levels in MATH(%) Level-1 Level-2 Level-3 Level-4 Level-5

Base Model Dataset Size

LLaMA2-7B 7.5K 17.85 8.39 4.77 3.05 0.91 Mistral-7B 7.5K 37.99 25.17 15.12 8.48 2.49 DeepSeekMath-7B 7.5K 64.07 46.76 37.84 24.63 10.73

- LLaMA2-7B 1.0M 75.29 55.03 44.56 31.22 13.75

- Mistral-7B 1.0M 80.55 63.31 53.05 38.47 19.18

- DeepSeekMath-7B 1.0M 79.18 62.30 54.82 40.44 19.71

LLaMA2-7B 2.1M 78.03 60.29 48.19 35.09 19.56 Mistral-7B 2.1M 80.78 66.33 55.53 41.52 21.45

- DeepSeekMath-7B 2.1M 80.78 65.21 58.00 41.60 21.83

- LLaMA2-7B 2.1M + 0.1M (hard) 78.03 62.19 48.89 36.66 17.98

- Mistral-7B 2.1M + 0.1M (hard) 81.01 67.45 58.44 45.22 21.53

- DeepSeekMath-7B 2.1M + 0.1M (hard) 84.90 67.45 57.91 44.07 21.22

LLaMA2-7B 2.1M + 0.2M (hard) 78.95 61.41 51.11 39.29 18.66 Mistral-7B 2.1M + 0.2M (hard) 83.52 68.90 59.50 46.05 22.21

- DeepSeekMath-7B 2.1M + 0.2M (hard) 82.84 68.46 57.91 42.50 23.41

LLaMA2-7B 2.1M + 0.4M (hard) 78.03 62.42 52.87 37.48 18.73 Mistral-7B 2.1M + 0.4M (hard) 83.52 67.56 60.65 44.89 25.08 DeepSeekMath-7B 2.1M + 0.4M (hard) 82.84 67.23 58.71 42.01 21.30

LLaMA2-7B 7.5k + 0.4M (hard) 63.16 43.96 34.39 24.46 10.20 Mistral-7B 7.5k + 0.4M (hard) 71.62 57.27 48.72 34.60 16.99 DeepSeekMath-7B 7.5k + 0.4M (hard) 81.01 61.97 51.90 37.07 18.05

GPT-4-Turbo - 82.84 73.38 65.34 52.88 34.06

- Table 11 | Difficulty level-wise performance of different base LLMs in Skywork-Math models and various sizes of SFT data on MATH. GPT-4-Turbo is evaluated using our designed grading criteria with 4-shot COT prompting.

D. Performance Analysis on MATH across Subjects

- Table 12 presents the accuracy results on the MATH benchmark across various math subjects. The Skywork-Math models excel in the "Algebra" category as we scale up the synthetic SFT data. However, it struggles in some other math subjects, such as "Geometry", where the understanding of geometric concepts may be challenging for language LLM models.

Counting & Probability

Intermediate Algebra

Number Theory

Prealgebra Precalculus

Geometry

Base Model Dataset Size Algebra

| | |
|---|---|
|LLaMA2-7B 7.5K Mistral-7B 7.5K DeepSeekMath-7B 7.5K|6.66 4.01 3.34 1.33 3.89 11.71 1.10 21.65 9.07 9.19 3.77 8.89 28.01 3.85 52.15 21.10 19.42 11.07 26.67 50.63 9.71<br><br>|
|LLaMA2-7B 1.0M Mistral-7B 1.0M DeepSeekMath-7B 1.0M<br><br>|55.69 32.28 30.69 16.06 34.26 55.80 18.50 65.37 34.60 35.49 20.49 44.44 64.87 23.81 68.16 35.02 35.91 22.81 41.30 62.34 26.74|
|LLaMA2-7B 2.1M Mistral-7B 2.1M DeepSeekMath-7B 2.1M|62.09 36.50 33.82 18.38 41.85 59.93 21.79 66.72 40.93 38.00 23.48 43.70 68.08 26.19 69.92 41.14 36.33 25.03 45.19 65.56 29.30<br><br>|
|LLaMA2-7B 2.5M Mistral-7B 2.5M DeepSeekMath-7B 2.5M|64.62 37.13 35.49 21.26 40.56 63.72 25.64 70.85 43.25 41.75 24.58 49.44 70.72 30.77 69.25 38.40 38.00 24.70 43.52 68.77 30.22|

- Table 12 | MATH accuracies across subjects with different SFT data sizes.

- E. Effect of Model Maximum Length in Two Stages of the Data Synthesis Pipeline

- Table 13 presents the performance with three 7B base models in Skywork-Math model series with maximum lengths set of 512 and 2048 in the stage 1 & 2 of the data synthesis pipeline.

Base Model Data Synthesis Pipeline (Size) Model Max Length MATH(%) GSM8k(%)

- LLaMA2-7B Stage 1 (2.1M) 512 42.36 70.81

- LLaMA2-7B Stage 1 (2.1M) 2048 45.56 73.62

- Mistral-7B Stage 1 (2.1M) 512 47.14 81.05

- Mistral-7B Stage 1 (2.1M) 2048 49.1 83.25

- DeepSeekMath-7B Stage 1 (2.1M) 512 48.24 79.61

- DeepSeekMath-7B Stage 1 (2.1M) 2048 48.64 79.30

LLaMA2-7B Stage 2 (2.5M) 512 44.06 67.85 LLaMA2-7B Stage 2 (2.5M) 2048 47.66 72.86 Mistral-7B Stage 2 (2.5M) 512 50.56 82.41 Mistral-7B Stage 2 (2.5M) 2048 51.22 83.93

- DeepSeekMath-7B Stage 2 (2.5M) 512 48.28 80.52

- DeepSeekMath-7B Stage 2 (2.5M) 2048 49.88 81.50

- Table 13 | Model performance with different model maximum lengths.

#### F. More Experiments with Base LLM models after SFTing on the SkyworkMath Dataset

As shown in Table 14, we conduct experiments with two additional pre-trained base LLM model. The results indicate that after SFTing on the Skywork-Math Dataset, both base models exhibit consistent performance improvement.

###### Base Model Data Synthesis Pipeline (Size) GSM8K(%) MATH(%)

LLaMA3-8B - 79.60 30.00 LLaMA3-8B Stage 1 (2.1M) 80.82 50.34 LLaMA3-8B Stage 2 (2.5M) 75.97 50.30

Llemma-7B - 36.40 18.00

- Llemma-7B Stage 1 (2.1M) 65.43 40.34
- Llemma-7B Stage 2 (2.5M) 66.03 40.08

- Table 14 | Performance on LLaMA3-8B AI@Meta (2024) and Llemma-7B Azerbayev et al. (2023) base LLM models. We fine-tune the corresponding base LLM models using the SkyworkMathQA dataset in stage 1 and stage 2 of the data synthesis pipeline.

