# arXiv:2310.20689v4[cs.CL]29Mar2024

## Learning From Mistakes Makes LLM Better Reasoner

Shengnan An∗♢,♣, Zexiong Ma∗♡,♣, Zeqi Lin†♣, Nanning Zheng†♢, Jian-Guang Lou♣, Weizhu Chen♣

♢IAIR, Xi’an Jiaotong University, ♣Microsoft Corporation, ♡Peking University

♢{an1006634493@stu, nnzheng@mail}.xjtu.edu.cn,

♡mazexiong@stu.pku.edu.cn, ♣{Zeqi.Lin, jlou, wzchen}@microsoft.com

#### Abstract

Large language models (LLMs) recently exhibited remarkable reasoning capabilities on solving math problems. To further improve their reasoning capabilities, this work explores whether LLMs can LEarn from MistAkes (LEMA), akin to the human learning process. Consider a human student who failed to solve a math problem, he will learn from what mistake he has made and how to correct it. Mimicking this error-driven learning process, LEMA incorporates mistake-correction data pairs during fine-tuning LLMs. Specifically, we first collect inaccurate reasoning paths from various LLMs, and then employ GPT-4 as a “corrector” to identify the mistake step, explain the reason for the mistake, correct the mistake and generate the final answer. In addition, we apply a correction-centric evolution strategy that effectively expands the question set for generating correction data. Experiments across various LLMs and reasoning tasks show that LEMA effectively improves CoT-alone fine-tuning. Our further ablations shed light on the non-homogeneous effectiveness between CoT data and correction data. These results suggest a significant potential for LLMs to improve through learning from their mistakes. Our code, models and prompts are publicly available at Github Link.

CoT-Alone Fine-Tuning LEMA

Question: Tina makes $18.00 an hour. If she works more than 8 hours per shift, she is

eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage. If she works

|85.4<br><br>84.2 83.5<br><br>77.9<br><br>60<br><br>70<br><br>80<br><br>90<br><br>WizardMath 70B<br><br>LLaMA-2 70B<br><br>LLaMA 65B<br><br>MetaMath 70B|
|---|

10 hours every day for 5 days, how much money does she make?

PerformanceonGSM8K(%)

LLM

Sampling Rationales

###### Rationales with Mistakes:

- Step 1: Tina makes $18.00 an hour for 8 hours, which is 8 * $18.00 = $144.00.
- Step 2: She makes $27.00 an hour for the 2 hours of overtime, which is 2 * $27.00 = $54.00.
- Step 3: So for 5 days, she makes $144.00 + $54.00 = $198.00.
- Step 4: The answer is 198.

###### GPT-4 As Corrector

Identifying Mistake and Making Correction

|26.9 27.1<br><br>25.0<br><br>20.8<br><br>15<br><br>20<br><br>25<br><br>30<br><br>WizardMath 70B<br><br>LLaMA-2 70B<br><br>LLaMA 65B<br><br>MetaMath 70B|
|---|

###### Correction:

PerformanceonMATH(%)

Incorrect Step: Step 3.

Explanation: Step 3 only calculates the earnings for one day, but not for the entire five days. Correct Solution:

...

- Step 3: For one day, she makes $144.00 + $54.00 = $198.00.
- Step 4: For 5 days, she makes $198.00 * 5 = $990.00.
- Step 5: The answer is 990.

Fine-Tuning on Mistake-Correction Data

LLM

Figure 1: Left: Process of LEarning from MistAkes (LEMA). Right: Performance of LEMA on GSM8K and MATH.

∗ Work done during the internship at Microsoft. † Corresponding authors.

#### 1 Introduction

|Mistakes are the portals of discovery.<br><br>—James Joyce|
|---|

With exponential growth in data size and model scale, contemporary large language models (Brown et al., 2020; Zhang et al., 2022; Hoffmann et al., 2022; Smith et al., 2022; OpenAI, 2023b; Anil et al., 2023) have demonstrated significant advancements on various NLP tasks, particularly in mathematical problem solving that necessitates complex chain-of-thought (CoT) reasoning (Wei et al., 2022; Wang et al., 2022; Li et al., 2023b; Shi et al., 2023; Qin et al., 2023; Lightman et al., 2023). In terms of performance on challenging mathematical tasks like GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021), proprietary large language models, including GPT-4 (OpenAI, 2023b) and PaLM-2 (Anil et al., 2023), have attained notable results. However, open-source LLMs such as LLaMA-2 (Touvron et al., 2023b) still have much room for improvement.

To further improve the CoT reasoning capabilities of open-source LLMs for tackling mathematical tasks, a common approach is to fine-tune these models using annotated/generated question-rationale data pairs (referred to as CoT data), which directly teach the model how to perform CoT reasoning on these tasks (Magister et al., 2022; Huang et al., 2022; Ho

- et al., 2022; Li et al., 2022; Yuan et al., 2023; Luo et al., 2023; Yu et al., 2023; Li et al., 2023a; Liang et al., 2023; Ranaldi & Freitas, 2024). While this straightforward learning process has exhibited its effectiveness, this study investigates whether the reasoning capabilities of LLMs can be further improved through a backward learning process, i.e., learning from the mistakes that LLMs have made. The insight of learning from mistakes comes from the learning process of human students. Consider a student who is just beginning to learn math. Beyond learning from golden knowledge and examples in books, he also does exercises. After failing to solve a problem, he will learn what mistakes he made and how to correct

- them. By learning from the mistakes he has made, his reasoning capability will be further improved. Inspired by this error-driven learning process, this work explores whether the reasoning capabilities of LLMs can also benefit from understanding and correcting mistakes.

To this end, we first generate mistake-correction data pairs (referred to as correction data) and then inject these correction data into the CoT fine-tuning process (Figure 1). For generating correction data, we employ multiple LLMs, including the LLaMA and GPT series models, to collect inaccurate reasoning paths (i.e., with incorrect final answers). We

- then use GPT-4 as the “corrector” to generate corrections for these inaccurate reasoning paths. The generated corrections contain three pieces of information: (1) the incorrect step in the original solution, (2) an explanation of why this step is incorrect, and (3) how to correct the original solution to arrive at the correct final answer. After filtering out corrections with incorrect final answers, our human evaluation reveals that our correction data exhibits adequate quality for the subsequent fine-tuning stage. In addition to using the original training questions to generate correction data, we also consider extending the question sets to scale up our correction data. Inspired by the evolution techniques for CoT data (Xu

- et al., 2023; Yu et al., 2023; Li et al., 2023a), we apply a correction-centric evolution strategy: compared to randomly selecting seed questions for evolution, our correction-centered evolution focuses more on moderately difficult questions for expanding the correction data. We blend the generated correction data with the CoT data and then fine-tune LLMs to perform LEarning from MistAkes (LEMA).

Our experiments on five open-source LLMs and five challenging reasoning tasks demonstrate the effectiveness of LEMA. Compared to fine-tuning on CoT data alone, LEMA consistently improves the performance across various LLMs and tasks. For instance, LEMA with LLaMA-2-70B (Touvron et al., 2023b) achieves 83.5% on GSM8K and 25.0% on MATH, while fine-tuning on CoT data alone yields 81.4% and 23.6%, respectively. By incorporating our correction-centric evolution strategy on MATH, LEMA with LLaMA-2-70B can be further improved from 25.0% to 29.3%. Moreover, LEMA can also enhance specialized LLMs such as WizardMath (Luo et al., 2023) and MetaMath(Yu et al., 2023). In addition to math tasks, LEMA also benefits commonsense reasoning, improving the performance of LLaMA-2-70B on CSQA (Talmor et al., 2019) from 84.2% to 85.3%.

|CoT Prompt|
|---|

|Correction Prompt|
|---|

|Question-Answer Pairs|
|---|

|Inaccurate Reasoning Paths<br><br>| |
|---|---|
| | |

Reasoning Model

Corrector Model

Corrections

(e.g., LLaMA2)

(i.e., GPT-4)

Expanding

###### Mistake-Correction Pairs

|Evolution Prompt| |
|---|---|
| | |

Sampling Seed Questions

Correction-Centric Evolution Model

(i.e., GPT-4)

Figure 2: Process of generating and expanding correction data.

Beyond these impressive results, our ablation studies on correction data shed further light. In controlling the training data sizes and training tokens to be the same, our experimental results reveal that mixing CoT and correction data outperforms a single data source. These results indicate the non-homogeneous effectiveness of CoT data and correction data. Moreover, compared with randomly selecting seed questions, our correction-centric evolution better improves the performance of LEMA. It demonstrates that moderately difficult questions are more suitable for expanding the correction data.

#### 2 Methodology

LEMA consists of three primary stages: generating correction data, correction-centric evolution, and fine-tuning.

- 2.1 Correction Data Generation

- Figure 2 briefly illustrates the process of generating correction data. Given a question-answer example (qi, ai) ∈ Q, a corrector model Mc, and a reasoning model Mr, we generate the mistake-correction data pair (qi ⊕ ri, ci) ∈ C, where ri represents an inaccurate reasoning path to the question qi, and ci denotes the correction for ri.

Collecting Inaccurate Reasoning Paths. We first sample multiple reasoning paths for each question qi using the reasoning model Mr and retain paths not achieving the correct final answer ai,

ri ∼ Mr(Pr ⊕ qi), Ans( ri) ̸= ai, (1)

where Pr is the few-shot prompt instructing the model to perform CoT reasoning, and Ans(·) extracts the final answer from the reasoning path.

Generating Corrections for Mistakes. For question qi and the inaccurate reasoning path ri, we employ the corrector model Mc to generate a correction and check the final answer in the correction,

ci ∼ Mc(Pc ⊕ qi ⊕ ri), Ans(ci) = ai, (2)

where Pc contains 4 annotated mistake-correction examples to guide the corrector model what kind of information should be contained in the generated corrections. Figure 3 briefly

illustrates Pc. Specifically, the annotated corrections comprises three pieces of information:

- • Incorrect Step: which step in the original reasoning path has made a mistake.
- • Explanation: explain what kind of mistake has been made in this step.
- • Correct Solution: how to revise the original reasoning path to achieve the correct answer.

Human Evaluation for Generated Corrections. Before generating data on a large scale, we first manually assess the quality of the generated corrections. We take LLaMA-2-70B as Mr, utilize GPT-4 as Mc, and generate 50 mistake-correction data pairs based on the GSM8K training set. We classify the corrections into three quality levels.

###### Prompt For Generating Corrections

For the following math word problems, the original solutions may contain errors. Please identify the incorrect step in each solution, explain why it is incorrect, and provide the correct solution starting from that step.

Question: James creates a media empire. He creates a movie for $2000. Each DVD cost $6 to make. He sells it for 2.5 times that much. He sells 500 movies a

day for 5 days a week. How much profit does he make in 20 weeks? Original Solution: Step 1, 500 movies a day, 5 days a week, for 20 weeks, he sells 500 * 5 * 20 = 50000 movies. Step 2, Each movie he sells for $6 * 2.5 = $15. Step 3, Thus, he makes a profit of $15 - $6 = $9 per movie. Step 4, Therefore, he makes a profit of $9 * 50000 = $450000. Step 5, The answer is 450000. Incorrect Step: Step 4. Explanation: The error in Step 4 is that it does not take into account the initial cost of creating the movie ($2000). The correct solution should subtract this amount from the total profit calculated in Step 4.

Correct Solution: … Step 4, Therefore, he makes a profit of $9 * 50000 - $2000 = $448000. Step 5, The answer is 448000.

… (Another 3 annotated examples)

Question: 𝒒𝒊, Original Solution: 𝒓෥𝒊, Incorrect Step: {Completion}

- Figure 3: A brief illustration of our prompt for generating corrections, containing the

|corrected step|
|---|

|incorrect step|
|---|

in the original solution, the reason of mistake, and the

.

- • Excellent: the corrector successfully identifies the incorrect step in ri, provides a reasonable explanation, and the corrected reasoning path exhibits high continuity with the pre-steps in the original reasoning path1.
- • Good: the corrector successfully identifies the incorrect step in ri, provides a reasonable explanation, while the corrected reasoning path has minor issues in continuity.
- • Poor: the corrector fails to identify the incorrect step in ri or provides unreasonable explanations.

- Appendix B.1 lists several examples under each quality level. Our evaluation finds that 35 out of 50 generated corrections are of excellent quality, 11 are good, and 4 are poor. Based on this human evaluation, we suppose the overall quality of corrections generated with GPT-4 is sufficient for the further fine-tuning stage. We generate corrections on a large scale and take all corrections that have correct final answers for fine-tuning LLMs. We provide further analysis on the choice and behavior of corrector model in Section D.6.

- 2.2 Correction-Centric Evolution

After building up the data generation pipeline, we explore how to scale up our correction data. We consider that expanding the question-answer set Q is a promising direction, as it primarily determines the correction data diversity.

Inspired by the recent success of evolution techniques on CoT augmentation (Xu et al., 2023; Yu et al., 2023; Li et al., 2023a), we explore how to effectively apply the evolution method to expand our correction data. The “evolution” means to generate a set of new question-answer pairs from the given seed questions by prompting powerful LLMs.

The general evolution method for CoT augmentation randomly selects seed questions to evolve. However, this strategy does not well suit the nature of our correction data, as too simple or too challenging questions are less valuable for evolving and collecting correction information.

- • For too simple questions, the reasoning models such as LLaMA can already solve them. Evolving these questions may not be effective for collecting mistakes.
- • For too challenging questions, the most powerful LLMs still cannot handle them. Evolving these questions may lead to much inaccurate information in corrections.

Therefore, we apply a correction-centric evolution strategy which more focuses on moderately difficult questions: we only sample seed questions that occur in our correction data C, rather than randomly sampling from the entire set Q,

##### qˆi ∼ Me(Pe ⊕ qi), qi ∈ C, (3)

1The high continuity means that the corrected reasoning steps follow the pre-steps generated before the identified mistake step.

- Table 1: Our main experimental results (%) on four mathematical reasoning tasks (GSM8K, MATH, SVAMP and ASDiv) and one commonsense reasoning task (CSQA). Appendix D.1 and D.2 illustrate the performance variances during training.

Tasks GSM8K MATH SVAMP ASDiv CSQA LLaMA-2-70B (Touvron et al., 2023b)

Model Training

CoT Fine-Tuning 81.4 23.6 80.3 80.7 84.2

+ Learning From Mistakes 83.5 (+2.1) 25.0 (+1.4) 81.6 (+1.3) 82.2 (+1.5) 85.3 (+1.1) LLaMA-65B (Touvron et al., 2023a)

CoT Fine-Tuning 76.2 19.7 71.9 77.4 83.1

+ Learning From Mistakes 77.9 (+1.7) 20.8 (+1.1) 72.8 (+0.9) 77.7 (+0.3) 84.0 (+0.9) CodeLLaMA-34B (Rozi`ere et al., 2023)

CoT Fine-Tuning 68.8 19.1 67.4 73.9 78.1

+ Learning From Mistakes 71.7 (+2.9) 20.4 (+1.3) 72.0 (+4.6) 74.4 (+0.5) 80.8 (+2.7) LLaMA-2-13B (Touvron et al., 2023b)

CoT Fine-Tuning 62.9 12.2 58.0 67.8 80.4

+ Learning From Mistakes 65.7 (+2.8) 12.6 (+0.4) 62.0 (+4.0) 71.1 (+3.3) 81.9 (+1.5) LLaMA-2-7B (Touvron et al., 2023b)

CoT Fine-Tuning 52.6 8.7 53.0 63.8 76.9

+ Learning From Mistakes 54.1 (+1.5) 9.4 (+0.7) 54.1 (+1.1) 65.5 (+1.7) 78.8 (+1.9)

where qi is the seed question, and Me and Pe are the LLM and prompt for evolving questions, respectively. Appendix B.3 illustrates our Pe.

The underlying principle of this strategy is straightforward. If one question frequently appears in correction data, it means that this question is not well solved by many reasoning models, but its inaccurate reasoning paths can be well handled by the corrector model.

- 2.3 Fine-Tuning LLMs

After generating the correction data, we fine-tune LLMs to examine whether these correction data can facilitate CoT reasoning. We compare the results under two settings:

- • Fine-Tuning on CoT Data Alone. In addition to the annotated data in each task, we additionally take CoT data augmentation following existing methods (Yuan et al., 2023; Li et al., 2023a; Yu et al., 2023). We generate more reasoning paths for each question in the training sets with GPT-4 and filter out paths with wrong final answers. We apply this CoT data augmentation to set up strong fine-tuning baselines that only utilize CoT data.
- • Fine-Tuning on CoT Data + Correction Data. We fine-tune LLMs on both CoT data and generated mistake-correction data. This setting is referred to as LEMA.

- Appendix B.2 shows the input-output formats of CoT data and correction data used for fine-tuning and evaluation.
- 3 Experimental Setup

- 3.1 Tasks

We undertake experiments on five challenging reasoning tasks, including four mathematical reasoning tasks (GSM8K, MATH, SVAMP and ASDiv) and one commonsense reasoning task (CSQA)2. For GSM8K, MATH and CSQA, we generate correction data based on their training sets. For SVAMP and ASDiv, we take the same training data for GSM8K.

GSM8K (Cobbe et al., 2021) contains high quality linguistically diverse grade school math word problems. It has 7,473 training examples with CoT and 1,319 test cases.

MATH (Hendrycks et al., 2021) examines math reasoning on solving challenging competition mathematics problems. It contains 7,500 training CoT data and 5,000 test cases.

SVAMP (Patel et al., 2021) consists of questions with short NL narratives as state descriptions. For evaluation on SVAMP, we use the same training data as for GSM8K and take all 1,000 examples in SVAMP as test cases.

2Appendix C.1 contains basic statics about the tasks and data.

CoT Fine-Tuning

###### LEMA

onGSM8K(%)Performance

75

85

80

70

60

83.5

77.9

82.3 82.5

77.3

+1.0

77.0

71.7

+0.9

81.4

76.2

+0.9

+1.1

70.8 70.7

65.7

+1.0

+2.0

+1.9

70

80

75

65

55

64.1

54.1

68.8

63.8

53.6 52.7

62.9

+1.2

+0.5

52.6

+0.1

65

75

70

60

50

32K 45K 32K 45K 32K 45K 32K 45K 32K 45K

CodeLLaMA-34B

LLaMA-2-70B

LLaMA-65B

LLaMA-2-13B

LLaMA-2-7B

- Figure 4: Performances of LEMA and CoT-alone fine-tuning with controlled data sizes (32K and 45K) on GSM8K. See Table 2 for results with controlled number of training tokens.

ASDiv (Miao et al., 2020) is a diverse math dataset in terms of both language patterns and problem types for evaluating. For evaluation on ASDiv, we use the same training data as for GSM8K and test on 2,084 examples in ASDiv3.

CSQA (Talmor et al., 2019) is a question answering dataset for commonsense reasoning. It has 9,741 examples in the training set and 1,221 examples in the dev set. As it does not contain any CoT annotation, we first annotate 4 CoT examples (detailed in Appendix C.3), then take its training set to augment CoT data and generate correction data.

- 3.2 Data Construction

CoT Data. For GSM8K (also SVAMP and ASDiv), the CoT data contains all training examples of GSM8K and 24,948 augmented reasoning paths. We first generate 30,000 reasoning paths with GPT-4 and filter out 5,052 paths with wrong final answers or unexpected format4. For MATH, the CoT data contains all training examples and 12,509 augmented reasoning paths. We sample 30,000 reasoning paths with GPT-4 and filter out 17,491 paths. For CSQA, we generate 15,000 reasoning paths with GPT-4 and then filter out 4,464 paths.

Correction Data. We utilize multiple LLMs to collect inaccurate reasoning paths, including LLaMA-2 (Touvron et al., 2023b), WizardLM (Xu et al., 2023), WizardMath (Luo et al., 2023), Text-Davinci-003 (OpenAI, 2023c), GPT-3.5-Turbo (OpenAI, 2023a) and GPT-4 (OpenAI, 2023b). We take GPT-4 as the corrector model. Finally, we collect 12,523, 6,306, 7,241 mistakecorrection pairs based on the training sets of GSM8K, MATH and CSQA, respectively.

Correction-Centric Evolution. We take 10K bootstrap samples from the questions in our correction data. We utilize GPT-4 to evolve the questions. To generate “ground-truth” answers for the evolved questions, we utilize GPT-4 to sample three answers for each question and conduct a majority voting. The question that leads to three different answers will be filtered. Note that the evolved data will only be used in Section 4.2.

- 3.3 Fine-Tuning and Evaluation

We fine-tune multiple open-source LLMs in the LLaMA (Touvron et al., 2023a), LLaMA2 (Touvron et al., 2023b), CodeLLaMA (Rozi`ere et al., 2023), WizardMath (Luo et al., 2023) and MetaMath (Yu et al., 2023) families. We utilize QLoRA5 (Hu et al., 2022; Dettmers et al., 2023) by default to conduct parameter-efficient fine-tuning (PEFT) for these models. We set low-rank dimension as 64 and dropout rate as 0.05. We set learning rate as 0.0001 for LLMs

- 3The original ASDiv contains 2,305 examples and we filter out non-numerical examples, detailed

in Appendix C.2.

- 4The unexpected format means that the final answer is failed to be extracted from the path with

the regular expression.

- 5https://github.com/artidoro/qlora.

Model Data Acc (%) LLaMA-2-70B

Model Acc (%)

WizardMath-70B (Luo et al., 2023) 81.6 WizardMath-70B + LEMA 84.2 (+2.6)

CoT-5.8M 82.1

LEMA-5.8M 83.5 (+1.4) LLaMA-2-13B

CoT-5.8M 64.2 LEMA-5.8M 65.7 (+1.5)

MetaMath-70B (Yu et al., 2023) 82.3 MetaMath-70B + LEMA 85.4 (+3.1) Table 3: Performances of LEMA with specialized LLMs on GSM8K.

Table 2: Performances with the same size of training tokens (5.8M) on GSM8K.

larger than (or equal to) 34B and 0.0002 for LLMs smaller than 34B. We set batch size as 96, train for 2,000 steps, and save checkpoints for every 100 training steps.

For evaluation, we evaluate the performance of all saved checkpoints based on vLLM library6 (Kwon et al., 2023) and report the accuracy of the best checkpoint. During inference, we set temperature as 0 (i.e., greedy decoding) and max sample length as 2,048. To clarify the influence from random disturbances during training, we provide the performances of the best three checkpoints in Appendix D.1 and the performance curves during the whole training processes in Appendix D.2. We do not add demonstration examples into the prompt for both fine-tuning and evaluation by default. All evaluations are conducted under the same CoT instruction. For models trained with LEMA, we do not generate corrections during evaluations. All our experiments can be conducted on 4 x A100 GPU stations.

#### 4 Results and Analysis

We focus on two main research questions in this section. More results and analysis are contained in Appendix D.

- 4.1 Can LLMs Learn From Mistakes?

LEMA effectively improves CoT-alone fine-tuning. Table 1 shows the main experimental results on five challenging reasoning tasks. Compared to fine-tuning on CoT data alone, incorporating correction data during fine-tuning brings improvements across all five backbone LLMs and five tasks. It demonstrates that LEMA can effectively facilicate CoT fine-tuning. Note that SVAMP and ASDiv can be regarded as two out-of-distribution tasks as the training data is constructed based on GSM8K. The gains on these two tasks reflect that LEMA has a certain extent of generalizability in the out-of-distribution scenarios.

The effectiveness of CoT data and correction data are non-homogeneous. If the effectiveness of the two data sources are homogeneous, the gains in Table 1 will be diminished if the data sizes of two fine-tuning settings are controlled as the same. To further validate the effectiveness of correction data, we conduct two ablation studies with controlled data sizes. In default settings, we have about 32K examples for CoT-alone fine-tuning and 45K examples for LEMA. Here are another two controlled settings:

- • LEMA-32K. We keep the 13K correction data and randomly remove 13K CoT data.
- • CoT-45K. To expand CoT data, we extract the corrected CoT from each correction example.

- Figure 4 shows that LEMA can still bring gains for four out of five backbone LLMs under the same data size. It means that these LLMs do learn extra information from our correction data that is not provided by the CoT data. The only exception is for LLaMA-2-7B. It indicates that a stronger backbone model can more effectively learn from mistakes.

Despite controlling the training data sizes to be the same, we also investigate the trainingtoken efficiency of LEMA compared with CoT-alone fine-tuning. Notice that the target-side length of correction data is generally longer than CoT data, so LEMA will have slightly more training tokens than CoT-alone fine-tuning under the same data size. Specifically,

6https://github.com/vllm-project/vllm.

LEMA LEMA + General Evolution LEMA + Correction-Centric Evolution

|15<br><br>25<br><br>35<br><br>25.3<br><br>27.0<br><br>28.9<br><br>31.5<br><br>33.8<br><br>34.9<br><br>+1.7<br><br>+3.6<br><br>+2.3<br><br>+3.4|
|---|

Accuracy(%)

LLaMA-2-70B Llemma-34B

(a)

LEMA + Full Fine-Tuning LEMA + QLoRA Fine-Tuning Trend Lines (Logarithmic)

|25.3<br><br>28.9<br><br>29.3<br><br>25.0<br><br>26.5 26.6<br><br>24<br><br>26<br><br>28<br><br>30<br><br>32<br><br>0 10000 20000 30000 40000|
|---|

|31.5<br><br>34.9<br><br>35.8<br><br>30.7<br><br>32.5<br><br>32.7<br><br>30<br><br>32<br><br>34<br><br>36<br><br>38<br><br>0 10000 20000 30000 40000|
|---|

Accuracy(%)

LLaMA-2-70B

Llemma-34B

(b)

- Figure 5: Performance of LEMA on MATH with evolution strategies. (a) Compare general and correction-centric evolution strategies (full fine-tuning). (b) The performance trend of LEMA with QLoRA or full fine-tuning. X-axis is the number of sampled questions.

CoT-45K has 5.4M training tokens and LEMA-45K has 5.8M (a ∼7% relative increment). To conduct the comparison under the same size of training tokens, we construct CoT-5.8M by sampling more reasoning paths (following Section 2.3) to add into CoT-45K.

- Table 2 shows that LEMA still outperforms CoT-alone fine-tuning with the same number of training tokens. Note that this comparison is under an unfavorable setup for LEMA as it increases the training samples for CoT-alone fine-tuning. The improvements in Table 2 further support the non-homogeneous effectiveness of CoT data and correction data. Moreover, we notice that augmenting more reasoning paths for LLaMA-2-70B does not continuously boost the model performance on GSM8K. To validate this, we further expand CoT-5.8M to CoT-6.8M and have a 82.2% accuracy. Such an observation is in line with the Yu et al. (2023). We suppose that this is because sampling too many reasoning paths for the same question will only bring redundant information to the training.

A stronger backbone model can be more effective at learning from mistakes. As evidenced in Table 1, LLaMA-2-70B has the highest baseline performances in CoT alone fine-tuning, while maintaining significant improvements in all five tasks (an accuracy gain of over 1%) with the help of LEMA. In contrast, for other four less powerful models in Table 1, the improvements from LEMA are occasionally less significant. This comparison, along with the performance of LLaMA-2-7B in Figure 4, suggests that the inherent strength of backbone LLMs can influence how well the models can learn from mistakes.

LEMA can also facilitate specialized LLMs. To adapt generally pre-trained LLMs into the math domain, there have been several specialized LLMs such as WizardMath (Luo et al., 2023) and MetaMath (Yu et al., 2023). We also apply LEMA on these specialized LLMs to further examine its effectiveness. As these models have been already trained on a large amount of CoT data designed for math tasks, we directly compare LEMA with the results reported in the original papers for these specialized models. Table 3 shows that LEMA can further improve these specialized LLMs. Appendix D.3 contains detailed comparisons.

- 4.2 How Beneficial Is Correction-Centric Evolution?

- Figure 5a and Figure 5b demonstrate further improvements on the performance of LEMA with incorporating the correction-centric evolution strategy to expand the correction data.

Correction-centric evolution can more effectively improve LEMA. Figure 5a shows the performance of LEMA with incorporating different evolution strategies. Besides the correction-centric evolution introduced in Section 2.2, we also compare with the general evolution strategy applied in previous work (Xu et al., 2023; Yu et al., 2023; Li et al., 2023a). For a fair comparison, the number of seed questions is kept the same for both evolution strategies (i.e., 10K). We also tried the Llemma (Azerbayev et al., 2023) model which has

been pre-trained on a math-related corpus (such as arXiv papers). We fully fine-tune LLMs as the correction data scale has been much increased7.

There are two primary conclusions. First, LEMA can effectively benefit from evolution techniques. It indicates that the performance of LEMA can be further improved by incorporating existing data augmentation techniques. Second, the correction-centric evolution outperforms the general evolution. It demonstrates that moderately difficult questions are more suitable for expanding the correction data.

Evolution techniques can better facilitate LEMA under full fine-tuning. To explore the scaling trend of LEMA, we apply the correction-centric evolution on another 10K sampled seed questions (detailed in Appendix C.5). Figure 5b shows the performance trends of LEMA as the question set expands. It shows that if only the original question-answer pairs in MATH are used (i.e., the initial points in each line), there is no significant difference in the performances of LEMA between QLoRA and full fine-tuning. However, as the question set expands, the performance with full fine-tuning improves significantly, while QLoRA fine-tuning increases only slightly. It indicates that the parameter-efficient fine-tuning can only “digest” a limited scale of correction data. Appendix D.5 provides further analysis.

#### 5 Related Work

LLMs with CoT reasoning. Wei et al. (2022) uncovered the emergence of CoT reasoning capability for extremely large language models, and this reasoning capability was then examined in various reasoning-related domains including logical reasoning (Creswell et al.,

- 2022; Pan et al., 2023; Lei et al., 2023), commonsense reasoning (Talmor et al., 2019; Geva et al., 2021; Ahn et al., 2022), and math reasoning (Miao et al., 2020; Koncel-Kedziorski et al., 2016; Patel et al., 2021; Cobbe et al., 2021; Hendrycks et al., 2021). The impressive performance of LLMs in these domains has spurred the research community to further investigate methods for effectively harnessing and enhancing CoT reasoning for LLMs (Wang et al., 2022; Zhou et al., 2022; Creswell & Shanahan, 2022; Li et al., 2023b; Lightman et al., 2023).

Enhancing CoT reasoning for solving mathematical problems. There has been much work dedicated to enhancing the performance of LLMs in solving mathematical problems from various perspectives. Some studies explored the voting or verification methods based on sampling multiple reasoning paths (Wang et al., 2022; Li et al., 2023b; Lightman et al., 2023). Some methods considered to generate executable programs to obtain the final answer or to integrate plug-in tools that facilitate the execution of external APIs during intermediate steps (Jie & Lu, 2023; Wang et al., 2023a; Yue et al., 2023; Azerbayev et al.,

- 2023; Gou et al., 2023). Some work collected math-related corpus such as arXiv papers for pre-training better base models for math (Azerbayev et al., 2023; Wang et al., 2023d). Some work focused on augmenting existing datasets, which expanded training sets or provided external annotations (Magister et al., 2022; Huang et al., 2022; Ho et al., 2022; Li et al., 2022; Luo et al., 2023; Yu et al., 2023; Li et al., 2023a; Liang et al., 2023; Liu et al., 2023a). From the perspective of the techniques used, this work follows the data augmentation approach.

Data augmentation for mathematical tasks. With the help of advanced LLMs (e.g., GPT-4 and GPT-3.5-Turbo), various methods have been proposed to generate more CoT data for mathematical tasks: Yuan et al. (2023) proposed rejection sampling for augmenting CoT data; Xu et al. (2023) evolved the math questions in the training sets; Li et al. (2023a) applied both query augmentation and response augmentation; Yu et al. (2023) used self-verification and FOBAR to generate CoT with high diversity. While the effectiveness of CoT data has been well studied, how to improve math reasoning with other auxiliary data is still underexplored. To this end, there are some preliminary explorations: Azerbayev et al. (2023) and Yue et al. (2023) found that code data can facilitate math reasoning; Liu et al. (2023b) and Wang et al. (2023e) constructed re-ranking data or verification data to make the model judge the quality of reasoning paths. This work takes a further step toward leveraging auxiliary

7Appendix C.4 contains the settings for full fine-tuning.

data: we propose and examine the effectiveness of mistake-correction data, which informs the model what kind of mistakes could be made in CoT reasoning and how to correct them.

#### 6 Conclusion

This work explores whether the reasoning capabilities of LLMs can be further improved by learning from mistakes. Experimental results and in-depth analysis demonstrate the effectiveness and potential of learning from mistakes.

#### Ethics Statement

Due to the utilization of pre-trained language models, this work could be exposed to some potential risks of ethical issues on general deep learning models (such as social bias and privacy breaches). We hope that the idea of learning from mistakes would facilitate the development of responsible AI models, for instance, on training LLMs to recognize and modify risky generated contents.

#### Reproducibility Statement

We open source our training code, evaluation scripts and fine-tuned checkpoints to facilitate further explorations on learning from mistakes. For generating the training data, we provide all our prompts used for data generation.

#### Acknowledgments

Shengnan An and Nanning Zheng were supported in part by NSFC under grant No. 62088102. Thank Chen Li at IAIR, Xi’an Jiaotong University for his valuable comments on this work.

#### References

Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Daniel Ho, Jasmine Hsu, Julian Ibarz, Brian Ichter, Alex Irpan, Eric Jang, Rosario Jauregui Ruano, Kyle Jeffrey, Sally Jesmonth, Nikhil J Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Kuang-Huei Lee, Sergey Levine, Yao Lu, Linda Luu, Carolina Parada, Peter Pastor, Jornell Quiambao, Kanishka Rao, Jarek Rettinghouse, Diego Reyes, Pierre Sermanet, Nicolas Sievers, Clayton Tan, Alexander Toshev, Vincent Vanhoucke, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Mengyuan Yan, and Andy Zeng. Do as i can, not as i say: Grounding language in robotic affordances, 2022.

Alibaba. Alibaba open sources qwen, a 7b parameter ai model, 2023. URL https://www.maginative.com/article/alibaba-open-sources-qwen-a-7b-\ parameter-ai-model/.

Shengnan An, Yifei Li, Zeqi Lin, Qian Liu, Bei Chen, Qiang Fu, Weizhu Chen, Nanning Zheng, and Jian-Guang Lou. Input-tuning: Adapting unfamiliar inputs to frozen pretrained models, 2022.

Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernandez Abrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan Botha, James Bradbury, Siddhartha Brahma, Kevin Brooks, Michele Catasta, Yong Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha

Chowdhery, Cl´ement Crepy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark D´ıaz, Nan Du, Ethan Dyer, Vlad Feinberg, Fangxiaoyu Feng, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, Guy Gur-Ari, Steven Hand, Hadi Hashemi, Le Hou, Joshua Howland, Andrea Hu, Jeffrey Hui, Jeremy Hurwitz, Michael Isard, Abe Ittycheriah, Matthew Jagielski, Wenhao Jia, Kathleen Kenealy, Maxim Krikun, Sneha Kudugunta, Chang Lan, Katherine Lee, Benjamin Lee, Eric Li, Music Li, Wei Li, YaGuang Li, Jian Li, Hyeontaek Lim, Hanzhao Lin, Zhongtao Liu, Frederick Liu, Marcello Maggioni, Aroma Mahendru, Joshua Maynez, Vedant Misra, Maysam Moussalem, Zachary Nado, John Nham, Eric Ni, Andrew Nystrom, Alicia Parrish, Marie Pellat, Martin Polacek, Alex Polozov, Reiner Pope, Siyuan Qiao, Emily Reif, Bryan Richter, Parker Riley, Alex Castro Ros, Aurko Roy, Brennan Saeta, Rajkumar Samuel, Renee Shelby, Ambrose Slone, Daniel Smilkov, David R. So, Daniel Sohn, Simon Tokumine, Dasha Valter, Vijay Vasudevan, Kiran Vodrahalli, Xuezhi Wang, Pidong Wang, Zirui Wang, Tao Wang, John Wieting, Yuhuai Wu, Kelvin Xu, Yunhan Xu, Linting Xue, Pengcheng Yin, Jiahui Yu, Qiao Zhang, Steven Zheng, Ce Zheng, Weikang Zhou, Denny Zhou, Slav Petrov, and Yonghui Wu. Palm 2 technical report, 2023.

Anthropic. Model card and evaluations for claude models, 2023. URL https://www-files.

###### anthropic.com/production/images/Model-Card-Claude-2.pdf.

Kourosh Hakhamaneshi Artur Niederfahrenhorst and Rehaan Ahmad. Finetuning llms: Lora or full-parameter? an in-depth analysis with llama 2, 2023. URL https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-\ parameter-an-in-depth-analysis-\with-llama-2.

Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q. Jiang, Jia Deng, Stella Biderman, and Sean Welleck. Llemma: An open language model for mathematics, 2023.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Antonia Creswell and Murray Shanahan. Faithful reasoning using large language models. arXiv preprint arXiv:2208.14271, 2022.

Antonia Creswell, Murray Shanahan, and Irina Higgins. Selection-inference: Exploiting large language models for interpretable logical reasoning. In The Eleventh International Conference on Learning Representations, 2022.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. arXiv preprint arXiv:2305.14314, 2023.

Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies. Transactions of the Association for Computational Linguistics, 9:346–361, 04 2021. ISSN 2307-387X. doi: 10.1162/tacl a 00370. URL https://doi.org/10.1162/tacl_ a_00370.

Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu Chen. Tora: A tool-integrated reasoning agent for mathematical problem solving, 2023.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.

Namgyu Ho, Laura Schmid, and Se-Young Yun. Large language models are reasoning teachers. arXiv preprint arXiv:2212.10071, 2022.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview. net/forum?id=nZeVKeeFYf9.

Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. Large language models can self-improve. arXiv preprint arXiv:2210.11610, 2022.

Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. Large language models cannot self-correct reasoning yet, 2023.

Zhanming Jie and Wei Lu. Leveraging training data in few-shot prompting for numerical reasoning. arXiv preprint arXiv:2305.18170, 2023.

Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. MAWPS: A math word problem repository. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 1152–1157, San Diego, California, June 2016. Association for Computational Linguistics. doi: 10.18653/v1/N16-1136. URL https://aclanthology.

###### org/N16-1136.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.

Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, and Sushant Prakash. Rlaif: Scaling reinforcement learning from human feedback with ai feedback, 2023.

Bin Lei, Chunhua Liao, Caiwen Ding, et al. Boosting logical reasoning in large language models through a new framework: The graph of thought. arXiv preprint arXiv:2308.08614, 2023.

Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 3045–3059, 2021.

Chengpeng Li, Zheng Yuan, Guanting Dong, Keming Lu, Jiancan Wu, Chuanqi Tan, Xiang Wang, and Chang Zhou. Query and response augmentation cannot help out-of-domain math reasoning generalization. arXiv preprint arXiv:2310.05506, 2023a.

Shiyang Li, Jianshu Chen, Yelong Shen, Zhiyu Chen, Xinlu Zhang, Zekun Li, Hong Wang, Jing Qian, Baolin Peng, Yi Mao, et al. Explanations from large language models make small reasoners better. arXiv preprint arXiv:2210.06726, 2022.

Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. Making language models better reasoners with step-aware verifier. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5315–5333, 2023b.

Zhenwen Liang, Wenhao Yu, Tanmay Rajpurohit, Peter Clark, Xiangliang Zhang, and Ashwin Kaylan. Let gpt be a math tutor: Teaching math word problem solvers with customized exercise generation. arXiv preprint arXiv:2305.14386, 2023.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee,

Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step, 2023. Bingbin Liu, Sebastien Bubeck, Ronen Eldan, Janardhan Kulkarni, Yuanzhi Li, Anh Nguyen,

Rachel Ward, and Yi Zhang. Tinygsm: achieving ¿80 Yixin Liu, Avi Singh, C. Daniel Freeman, John D. Co-Reyes, and Peter J. Liu. Improving large language model fine-tuning for solving math problems, 2023b.

Ximing Lu, Sean Welleck, Jack Hessel, Liwei Jiang, Lianhui Qin, Peter West, Prithviraj Ammanabrolu, and Yejin Choi. Quark: Controllable text generation with reinforced unlearning. In Advances in Neural Information Processing Systems, 2022.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. arXiv preprint arXiv:2308.09583, 2023.

Lucie Charlotte Magister, Jonathan Mallinson, Jakub Adamek, Eric Malmi, and Aliaksei Severyn. Teaching small language models to reason. arXiv preprint arXiv:2212.08410,

- 2022.

Shen-yun Miao, Chao-Chun Liang, and Keh-Yih Su. A diverse corpus for evaluating and developing English math word problem solvers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 975–984, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.92. URL https://aclanthology.org/2020.acl-main.92.

OpenAI. Gpt-3.5 turbo fine-tuning and api updates, 2023a. URL https://openai.com/

blog/gpt-3-5-turbo-fine-tuning-and-\api-updates. OpenAI. Gpt-4 technical report, 2023b. OpenAI. Openai documentation: Models, 2023c. URL https://platform.openai.com/

docs/models/gpt-3-5.

Liangming Pan, Alon Albalak, Xinyi Wang, and William Yang Wang. Logic-lm: Empowering large language models with symbolic solvers for faithful logical reasoning. arXiv preprint arXiv:2305.12295, 2023.

Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are NLP models really able to solve simple math word problems? In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 2080–2094, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.168. URL https://aclanthology.org/2021.

###### naacl-main.168.

Chengwei Qin, Aston Zhang, Zhuosheng Zhang, Jiaao Chen, Michihiro Yasunaga, and Diyi Yang. Is chatgpt a general-purpose natural language processing task solver? arXiv preprint arXiv:2302.06476, 2023.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model, 2023.

Leonardo Ranaldi and Andre Freitas. Aligning large and small language models via chain-ofthought reasoning. In Yvette Graham and Matthew Purver (eds.), Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1812–1827, St. Julian’s, Malta, March 2024. Association for Computational Linguistics. URL https://aclanthology.org/2024.eacl-long.109.

Baptiste Rozi`ere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, J´er´emy Rapin, Artyom Kozhevnikov, Ivan Evtimov, Joanna Bitton, Manish Bhatt, Cristian Canton Ferrer, Aaron Grattafiori, Wenhan Xiong, Alexandre D´efossez, Jade Copet, Faisal Azhar, Hugo Touvron, Louis Martin, Nicolas Usunier, Thomas Scialom, and Gabriel Synnaeve. Code llama: Open foundation models for code, 2023.

Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, and Jason Wei. Language models are multilingual chain-of-thought reasoners. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.

###### net/forum?id=fR3wGCk-IXp.

Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, Jared Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Korthikanti, et al. Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model. arXiv preprint arXiv:2201.11990, 2022.

Kaya Stechly, Matthew Marquez, and Subbarao Kambhampati. Gpt-4 doesn’t know it’s wrong: An analysis of iterative prompting for reasoning problems, 2023.

Yusheng Su, Chi-Min Chan, Jiali Cheng, Yujia Qin, Yankai Lin, Shengding Hu, Zonghan Yang, Ning Ding, Xingzhi Sun, Guotong Xie, Zhiyuan Liu, and Maosong Sun. Exploring the impact of model scaling on parameter-efficient tuning, 2023.

Xianghui Sun, Yunjie Ji, Baochang Ma, and Xiangang Li. A comparative study between full-parameter and lora-based fine-tuning on chinese instruction data for instruction following large language model, 2023.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4149–4158, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1421. URL https://aclanthology.org/N19-1421.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.

Karthik Valmeekam, Matthew Marquez, and Subbarao Kambhampati. Can large language models really improve by self-critiquing their own plans?, 2023.

Ke Wang, Houxing Ren, Aojun Zhou, Zimu Lu, Sichun Luo, Weikang Shi, Renrui Zhang, Linqi Song, Mingjie Zhan, and Hongsheng Li. Mathcoder: Seamless code integration in llms for enhanced mathematical reasoning, 2023a.

Ruida Wang, Wangchunshu Zhou, and Mrinmaya Sachan. Let’s synthesize step by step: Iterative dataset synthesis with large language models by extrapolating errors from small models, 2023b.

Xinyi Wang, Lucas Caccia, Oleksiy Ostapenko, Xingdi Yuan, and Alessandro Sordoni. Guiding language model reasoning with planning tokens. arXiv preprint arXiv:2310.05707, 2023c.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2022.

Zengzhi Wang, Rui Xia, and Pengfei Liu. Generative ai for math: Part i – mathpile: A billion-token-scale pretraining corpus for math, 2023d.

Zhaoyang Wang, Shaohan Huang, Yuxuan Liu, Jiahai Wang, Minghui Song, Zihan Zhang, Haizhen Huang, Furu Wei, Weiwei Deng, Feng Sun, and Qi Zhang. Democratizing reasoning ability: Tailored learning from large language model, 2023e.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022.

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions, 2023.

Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, Fan Yang, et al. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305, 2023.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284, 2023.

Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Chuanqi Tan, and Chang Zhou. Scaling relationship on learning mathematical reasoning with large language models. arXiv preprint arXiv:2308.01825, 2023.

Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning, 2023.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.

Denny Zhou, Nathanael Sch¨arli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V Le, et al. Least-to-most prompting enables complex reasoning in large language models. In The Eleventh International Conference on Learning Representations, 2022.

This is the Appendix of the paper: Learning From Mistakes Makes LLM Better Reasoner.

#### A Discussion

Here, we discuss further about the insights from our exploration on learning from mistakes.

- A.1 LLMs for Self-Correction

Recently, much work has investigated the behavior of advanced LLMs (e.g., GPT-4) on correcting mistakes generated by themselves (Valmeekam et al., 2023; Stechly et al., 2023; Huang et al., 2023). We also conduct further analysis on self-correction performance based on our correction data (detailed in Appendix D.6). These work and our analysis drew the same conclusion: the most powerful LLMs by now still struggle to perform self-correction. To achieve more reliable utilization of self-correction, we think that there are mainly three directions. (1) Inject external supervision to verify the correcting process, such as using the labeled final answers (which is applied in our work) or incorporating human feedback. (2) Train a process-based verifier to judge the quality of self-correction process. Lightman et al. (2023) has demonstrated the great potential of verifier-based method. (3) Develop trust-worth LLMs that can at least honestly tell us what it can solve and what does not.

- A.2 Training with Feedback

To align the behavior of LLMs with human expectations, existing work has tried to collect feedback for the model-generated contents and inject these feedback back into the model through various techniques, such as PPO (Lu et al., 2022), RLHF (OpenAI, 2023b) and DPO (Rafailov et al., 2023). To reduce human efforts on annotation, some recent work tried to use LLMs to generate feedback, such as RLAIF (Lee et al., 2023). From this view, LEMA can also be regarded as injecting the feedback from more powerful LLMs (i.e., GPT-4) into smaller models (e.g., LLaMA). We highlight one difference here: the injection process of LEMA is just implemented with instruction-based fine-tuning rather than RL-based methods. It sheds light that for large pre-trained models, it can directly and effectively learn from the comparison between unexpected and expected contents through the inputoutput fine-tuning process. This can much save the researchers effort to specially design the learning algorithms.

- A.3 Learning From The World Model

Recent advancements in LLMs have enabled them to perform a step-by-step approach in problem-solving. However, this multi-step generation process does not inherently imply that LLMs possess strong reasoning capabilities, as they may merely emulate the superficial behavior of human reasoning without genuinely comprehending the underlying logic and rules necessary for precise reasoning. This incomprehension results in mistakes during the reasoning process and necessitates the assistance of a “world model” that possesses a consciousness prior about the logic and rules governing the real world. From this perspective, our LEMA framework employs GPT-4 as a “world model” to teach smaller models in adhering to these logic and rules, rather than merely mimicking the step-by-step behavior.

#### B Additional Examples

- B.1 Examples in Human Evaluation Figure 13 illustrates the quality levels of three example corrections.

|Input: Please solve the following math problem. Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and<br><br>May?<br><br>Answer: Let's think step by step.<br><br>Output: Natalia sold 48 / 2 = 24 clips in May. Natalia sold 48 + 24 = 72 clips altogether in April and May. The answer is 72.<br><br>Input-Output Format of CoT Data|
|---|

|Input: For the following math problem, the original solution is incorrect. Please identify the incorrect step, explain why it is incorrect, and correct the original solution starting from the incorrect step. Question: Courtney liked to collect marbles. She kept them in mason jars. One jar had 80 marbles. Her second jar had twice that amount. She just started her third jar which currently has 1/4 the amount of her first jar. How many marbles does she have in total? Original Solution: {<br><br>Step 1: 80 marbles in the first jar, 2 x 80 = 160 marbles in the second jar.<br>Step 2: The third jar has 1/4 of the first jar, which is 80 / 4 = 20 marbles.<br>Step 3: So, Courtney has 160 + 20 = 180 marbles in total.<br>Step 4: The answer is 180. } Incorrect Step:<br><br><br>Output: Step 3.<br><br>Explanation: The calculation in Step 3 is incorrect because it only adds the marbles from the second and third jars. The marbles from the first jar should also be<br><br>included in the total. Correct Solution: { …<br><br>Step 3: So, Courtney has 80 (first jar) + 160 (second jar) + 20 (third jar) = 260 marbles in total.<br>Step 4: The answer is 260. }<br><br><br>Note: The ellipsis here are tokens that were actually used in the fine-tuning process, as we suppose that simply copying pre-steps is not much informative for learning.<br><br>Input-Output Format of Correction Data|
|---|

- Figure 6: The input-output formats for our CoT data and correction data, respectively. The input part serves as a prompt and only the loss in the output part participates in the back-propagation.

|Prompt For Question Evolution| |
|---|---|
|I want you to act as a math teacher. You need to create relevant math questions for the given question. The created math question must be reasonable, without<br><br>unknown variables, and must be understood to answer in pure number by humans. Ensure that the created questions are **diverse** from the original one.<br><br>Ensure that the created questions use **different natural language expressions**.<br><br>#Given Question#: 𝒒𝒊 #Created Question#:| |
| | |

Figure 7: The prompt for evolving questions.

- B.2 Input-Output Formats for Fine-Tuning

Figure 6 illustrate the input-output formats of CoT data and correction data, respectively. Note that during the fine-tuning process, the input part serves as a prompt and only the loss in the output part participates in the back-propagation.

- B.3 Evolution Prompt

- Figure 7 illustrates our prompt used for evolving new questions from the given seed question.

- Table 4: Statistics of data sizes for the five tasks in our experiments (without question evolution).

Task CoT Data Correction Data Test Data GSM8K (Cobbe et al., 2021) 32,421 12,523 1,319

MATH (Hendrycks et al., 2021) 20,009 6,306 5,000 SVAMP (Patel et al., 2021) - - 1,000

ASDiv (Miao et al., 2020) - - 2,084 CSQA (Talmor et al., 2019) 10,536 7,241 1,221

|Please answer the following commonsense question. Please first explain each candidate answer, then select only one answer that is most relevant to the<br><br>question and provide reasons.<br><br>Question: The fox walked from the city into the forest, what was it looking for? Candidate Answers: pretty flowers, hen house, natural habitat, storybook, dense forest. Explanations: Pretty flowers - This answer suggests that the fox might be looking for beautiful flowers. However, it is not common for a fox to seek out flowers, as they are carnivorous animals and do not have a particular interest in plants. Hen house - This answer implies that the fox might be searching for a hen house to find food, such as chickens. Foxes are known to prey on small animals, and a hen house would be a potential source of food for them.<br><br>Natural habitat - This answer suggests that the fox is looking for its natural habitat, which is typically a forest or wooded area. Foxes are wild animals and are<br><br>more comfortable in their natural environment than in a city. Storybook - This answer implies that the fox is searching for a storybook, which is an inanimate object and not something a fox would be interested in or capable of finding.<br><br>Dense forest - This answer is similar to the natural habitat option, suggesting that the fox is looking for a dense forest, which could be a part of its natural<br><br>habitat. Reasons and Final Answer:<br><br>The most relevant answer to the question is that the fox was looking for its natural habitat. This is because foxes are wild animals and are more comfortable in their natural environment, such as a forest, than in a city. The other candidate answers either do not align with the natural behavior of a fox (pretty flowers,<br><br>storybook) or are too specific (hen house, dense forest) without enough context to support them as the most likely answer.<br><br>The answer is natural habitat.<br><br>Annotated CoT Example For CSQA|
|---|

Figure 8: One annotated CoT example for CSQA.

#### C More Details For Experimental Setup

- C.1 Data Statistics

- Table 4 illustrates basic statics about the tasks and data (without question evolution).

- C.2 Evaluation on ASDiv

As mentioned in our setup, the original version of ASDiv contains 2,305 questions and part of them lead to non-numerical answers. For instance, for the question “Mrs. Hilt has two pennies, two dimes, and two nickels. Jacob has four pennies, one nickel, and one dime. Who has more money?”, the answer is the string value “Mrs. Hilt”; for the question “Tessa has 4 apples. Anita gave her 5 more. She needs 10 apples to make a pie. Does she have enough to make a pie?”, the answer is a Boolean value “False”. As our models are trained on data derived from GSM8K where questions are all leading to numerical answers, it is reasonable that these models can not generate non-numerical answers. Therefore, for evaluation on ASDiv, we filter out questions with non-numerical answers and finally leave 2,084 questions. Specifically, for the question-answer pair in ASDiv, it will be filtered out if the answer can not be successfully recognized by the Python function float(·).

- C.3 Data Construction For CSQA

The original training examples in CSQA only contain the labeled final answers without rationales. Therefore, we need to generate CoT for the training examples. We first annotate rationales for four training examples. Figure 8 shows one annotated example. Specifically, the CoT contain three parts: the explanation to each candidate answers, the predicted final answer, and the reason to choose this answer. Then, we utilize GPT-4 to generate rationales for other training examples and filter out rationales that do not contain the correct final

- Table 5: Performances of the best three checkpoints saved during the fine-tuning process and the average of three results.

GSM8K MATH 1st / 2nd / 3rd Avg. 1st / 2nd / 3rd Avg. LLaMA-2-70B (Touvron et al., 2023b)

Model Training

CoT Fine-Tuning 81.4 / 81.3 / 81.1 81.3 23.6 / 23.2 / 23.2 23.2

+ Learning From Mistakes 83.5 / 83.4 / 83.2 83.4 (+2.1) 25.0 / 25.0 / 24.6 24.9 (+1.7) LLaMA-65B (Touvron et al., 2023a)

CoT Fine-Tuning 76.2 / 76.2 / 75.7 76.0 19.7 / 19.7 / 19.2 19.5

+ Learning From Mistakes 77.9 / 77.3 / 77.2 77.5 (+1.5) 20.8 / 20.3 / 20.2 20.4 (+0.9) CodeLLaMA-34B (Rozi`ere et al., 2023)

CoT Fine-Tuning 68.8 / 68.5 / 68.2 68.5 19.1 / 19.0 / 18.9 19.0

+ Learning From Mistakes 71.7 / 71.0 / 70.9 71.2 (+2.7) 20.4 / 20.2 / 20.0 20.2 (+1.2) LLaMA-2-13B (Touvron et al., 2023b)

CoT Fine-Tuning 62.9 / 62.7 / 62.7 62.8 12.2 / 11.9 / 11.8 12.0

+ Learning From Mistakes 65.7 / 65.2 / 65.0 65.3 (+2.5) 12.6 / 12.6 / 12.4 12.5 (+0.5) LLaMA-2-7B (Touvron et al., 2023b)

CoT Fine-Tuning 52.6 / 52.5 / 52.5 52.5 8.7 / 8.5 / 8.5 8.6

+ Learning From Mistakes 54.1 / 53.7 / 53.6 53.8 (+1.3) 9.4 / 8.9 / 8.8 9.0 (+0.4)

[Figure 1]

Figure 9: The performance curves of LLaMA-2-70B during 2,000 fine-tuning steps.

answers. For generating correction data, we do not require GPT-4 to explicitly identify the position of mistake. It is because the CoT for commonsense questions does not exhibit a clear step-wise manner, and our ablation study on math tasks have showed that this information is less influential to the final performance.

- C.4 Full Fine-Tuning Setting

For fully fine-tuning LLaMA-2-70B and Llemma-34B, the learning rate is 1e-5 and the batch size is 128. We fine-tune LLaMA-2-70B for 3 epochs and Llemma-34B for 2 epochs. The evaluation results are reported on the final checkpoints. Other setting are kept the same in Section 3.3.

- C.5 Another Round of Correction-Centric Evolution

To explore the scaling trend of LEMA, we take another round of correction-centric evolution to expand correction data. The second round takes the same 10K seed questions as the first round. The only difference is that we replace the vanilla model as the fine-tuned models from the first round to collect inaccurate reasoning paths.

#### D More Results and Analysis

- D.1 Performances of Best Three Checkpoints

- Table 5 shows the performances of the best three checkpoints saved during the fine-tuning process along with the average of three results. It demonstrates that our main results are not caused by soem random disturbances during training.

Table 6: Math reasoning performances of various LLMs.

Model GSM8K MATH

closed-source models GPT-4 (OpenAI, 2023b) 92.0 42.5 Claude-2 (Anthropic, 2023) 88.0 Flan-PaLM-2 (Anil et al., 2023) 84.7 33.2 GPT-3.5-Turbo (OpenAI, 2023a) 80.8 34.1 PaLM-2 (Anil et al., 2023) 80.7 34.3

open-source models LLaMA-2-7B (Touvron et al., 2023b) 14.6 2.5 Baichuan-2-7B (Yang et al., 2023) 24.5 5.6 SQ-VAE-7B (Wang et al., 2023c) 40.0 7.0 RFT-7B (Yuan et al., 2023) 50.3 Qwen-7B (Alibaba, 2023) 51.6 LLaMA-2-7B + LEMA (ours) 54.1 9.4 WizardMath-7B (Luo et al., 2023) 54.9 10.7 WizardMath-7B + LEMA (ours) 55.9 11.9 LLaMA-2-13B (Touvron et al., 2023b) 28.7 3.9 SQ-VAE-13B (Wang et al., 2023c) 50.6 8.5 Baichuan-2-13B (Yang et al., 2023) 52.8 10.1 RFT-13B (Yuan et al., 2023) 54.8 WizardMath-13B (Luo et al., 2023) 63.9 14.0 LLaMA-2-13B + LEMA (ours) 65.7 12.6 MetaMath-13B (Yu et al., 2023) 72.3 22.4 MetaMath-13B + LEMA (ours) 73.2 22.7 LLaMA-2-70B (Touvron et al., 2023b) 56.8 13.5 RFT-70B (Yuan et al., 2023) 64.8 WizardMath-70B (Luo et al., 2023) 81.6 22.7 MuggleMath-70B (Li et al., 2023a) 82.3 MetaMath-70B (Yu et al., 2023) 82.3 26.6 LLaMA-2-70B + LEMA (ours) 83.5 25.0 WizardMath-70B + LEMA (ours) 84.2 27.1 MetaMath-70B + LEMA (ours) 85.4 26.9

- D.2 Training Curves

- Figure 9 shows the performance curves of LLaMA-2-70B during 2,000 fine-tuning steps. It shows that adding correction data leads to clear improvements during training. These consistent improvements demonstrate that the effectiveness of our correction data is robust to the random disturbances during training.

- D.3 Comparison with SOTA Models

Table 6 contains the comparison with more SOTA models. Another interesting finding in Table 6 is that the performance of LLaMA-2-70B + LEMA can be comparable with MuggleMath-70B (Li et al., 2023a) and MetaMath-70B (Yu et al., 2023). Note that these two specialized LLMs also take the LLaMA-2-70B as the backbone model while their training data sizes are much larger than LEMA: MuggleMath has ∼220K CoT data and MetaMath has ∼400K CoT data, while LEMA only has ∼70K CoT + correction data for math problems. This comparison further supports the non-homogeneous effectiveness between CoT data and correction data.

- D.4 Ablations of Correction Information

The explanations and corrected reasoning paths play important roles in LEMA. As introduced in Section 2.1, our correction data mainly contains three pieces of information: the mistake step (M.S.), the corrected solution (C.S.), and the explanation to the mistake (Exp.). To evaluate their individual contribution to the LEMA performance, we separately omit each information in our correction data. Figure 12 shows the results: the performance of LEMA drops significantly without the corrected solution or the explanation, while omitting

Inaccurate Reasoning Paths Corrections with Correct Final Answers

Success Rate

|0<br><br>10<br><br>20<br><br>30<br><br>40<br><br>50<br><br>Level 1 Level 2 Level 3 Level 4 Level 5|
|---|

|0<br><br>2,000<br><br>4,000<br><br>6,000<br><br>8,000<br><br>Level 1 Level 2 Level 3 Level 4 Level 5|
|---|

NumberofExamples

SuccessRate(%)

- Figure 10: Statistics of generated correction data according to different difficulty levels in MATH. Left: The number of collected inaccurate reasoning paths and generated corrections with correct final answers under different difficulty levels. Right: The success rate for correcting inaccurate reasoning paths under different difficulty levels.

0.10

0.15

0.20

0.25

0.30

0.35

|∆PPL|
|---|

∆PPL

GSM8K MATH

LEMA

LLaMA-2-70B LLaMA-65B

0.00

0.05

0.10

0.15

0.20

0.25

LLaMA-2-70B LLaMA-65B

CoT Fine-Tuning

- Figure 11: The differences between the PPLs (∆PPL) on mistaken CoT and correct CoT. A higher difference indicate that the model can better avoid the mistakes.

the mistake step shows less influence to the performance. We suppose it is because the corrected solution and the explanation have implicitly informed which step is incorrect. Therefore, it could be less influential to make the model explicitly identify the position of mistake.

- D.5 Additional Analysis to LEMA

LEMA can still bring improvements to CoT fine-tuning if the distributions of questions are controlled the same. In our default setting, correction data contains more challenging questions that can not be easily solved by various LLMs. This leads to a distribution shift on the difficulty of questions in training data. As Wang et al. (2023b) indicated that this distribution shift can also benefit fine-tuning LLMs, we also mitigate the influence from question distribution shift to further clarify the effectiveness of LEMA. Our ablation setting CoT-45K can be used to clarify this point: its additional CoT data are just converted from correction data, thus the question distributions of CoT-45K and our default LEMA-45K are exactly the same. Therefore, the results in Figure 4 under 45K data size demonstrate that LEMA still outperforms CoT-alone fine-tuning when the influence from question distribution shift is kept the same.

QLoRA fine-tuning cannot fully “digest” a large amount of correction data. As shown in Figure 5b, as the correction data expands, the gap between full-fine-tuning and QLoRA fine-tuning increases. Such an observation is not well aligned with the conclusions of some existing work. Some work indicated that if the model size is large enough, parameterefficient fine-tuning (PEFT) can achieve comparable performance with fine-tuning (Lester et al., 2021; An et al., 2022; Sun et al., 2023; Su et al., 2023; Artur Niederfahrenhorst & Ahmad,

- 2023). We suppose the property of correction data causes the inconsistency in observations. Specifically, correction data is just auxiliary data that do not directly contribute to the in-task training. We suppose that models with PEFT can “eat” a large amount of correction data but cannot fully “digest” them. As a results, the training on correction data with PEFT might not effectively contribute to the forward reasoning process.

LEMA

LEMA w/o M.S. LEMA w/o C.S. LEMA w/o Exp. CoT Alone

- 80
- 81
- 82
- 83
- 84

- 22
- 23
- 24
- 25
- 26

83.5

Performance(%)

83.2

25.0 24.9

-0.3

82.5

24.4

-0.1

82.3

24.1

-1.0

-0.6

-1.2

23.6

-0.9

81.4

GSM8K

MATH

- Figure 12: Performance of LEMA with ablations on correction information. The backbone LLM is LLaMA-2-70B. For each ablation setting, we mark the influence on performance compared to the default setting of LEMA.

The comparison learned in the correction data also influences the CoT generation. During training on the correction data, LLMs could be aware of the comparison between the correct and incorrect CoT. We suppose such kind of comparison can take effect during CoT generation. Based on this intuition, we evaluate the differences between PPLs defined as follows,

1

### ||C|| ∑

∆PPL(C; θ) =

[PPL( ri|qi; θ) − PPL(ri|qi; θ)],

(qi, ri,ci)∈C

where C is a set of correction data, θ represents the model parameters after fine-tuning, PPL(y|x; θ) returns the perplexity on y with x as the context, ri is one mistaken CoT for the question qi, and ri is the correct CoT extracted from the correction ci. We calculate ∆PPL for fine-tuned LLaMA-2-70B and LLaMA-65B, based on the correction data for GSM8K and MATH. Figure 11 shows ∆PPL for different fine-tuned models. It shows that LEMA consistently leads to a higher ∆PPL than CoT-alone fine-tuning.

- D.6 Further Analysis on Corrector

In our default setting, we take GPT-4 as the corrector model and our human evaluation in Section 2.1 supports this choice. In the following, we provide further analysis on the choice and behavior of the corrector model. Specifically, we want to answer the following research questions: RQ1: Can we use a less powerful model as the corrector model? RQ2: How well does GPT-4 perform in self-correction? RQ3: How well does GPT-4 correct inaccurate reasoning paths for challenging questions?

Less powerful models are not suitable for generating corrections. Despite GPT-4, we have also tried leveraging GPT-3.5-Turbo as the corrector model and assess the quality of generated corrections. We take another round of human evaluation on 20 corrections generated by GPT-3.5-Turbo and find that nearly half are of poor quality. Therefore, we just call GPT-4 for correction generation although it is much more expensive than GPT-3.5Turbo. We believe it is a valuable research direction to explore how to generate high-quality corrections without GPT-4.

GPT-4 can correct its own mistakes but with a low success rate. Specifically, for 2,696 inaccurate reasoning paths generated by GPT-4 on MATH training set, we finally get 217 corrections with correct final answers. It means that GPT-4 only achieves 8.0% success rate for self-correction. Compared with this low success rate for self-correction, GPT-4 can more effectively correct mistakes from less powerful models, such as LLaMA-2-70B (37.5% success rate on MATH) and GPT-3.5-Turbo (26.9% success rate on MATH). One possible reason for the low success rate of self-correction is that the mistakes generated by GPT-4 are from more challenging questions, thus these mistakes are naturally harder for correcting.

GPT-4 still struggles to correct inaccurate reasoning paths for challenging questions. The math problems in MATH can be categorized into five levels of difficulty: Level 1 for the easiest problems and Level 5 for the most challenging ones. Figure 10 shows statistics of our correction data on MATH according to different difficulty levels. As the difficulty increased from Level 1 to Level 5, the number of collected inaccurate reasoning paths increased, while

|Question: Courtney liked to collect marbles. She kept them in mason jars. One jar had 80 marbles. Her second jar had twice that amount. She just started her third jar which currently has 1/4 the amount of her first jar. How many marbles does she have in total? Original Solution:<br><br>Step 1: 80 marbles in the first jar, 2 x 80 = 160 marbles in the second jar.<br>Step 2: The third jar has 1/4 of the first jar, which is 80 / 4 = 20 marbles.<br>Step 3: So, Courtney has 160 + 20 = 180 marbles in total.<br>Step 4: The answer is 180. Incorrect Step: Step 3.<br><br><br>Explanation: The calculation in Step 3 is incorrect because it only adds the marbles from the second and third jars. The marbles from the first jar should also be included in the total. Correct Solution: …<br><br>Step 3: So, Courtney has 80 (first jar) + 160 (second jar) + 20 (third jar) = 260 marbles in total.<br>Step 4: The answer is 260.<br><br><br>Generated Correction (Excellent)|
|---|

|Question: Johnny is a dog walker. He can walk 3 dogs at once. He gets paid $15 for a 30-minute walk and $20 for a 60-minute walk. Johnny works for 4 hours per day. If he always walks the maximum number of dogs possible and 6 dogs have 60-minute walks per day, how much money does he make in a week where he works 5 days?<br><br>Original Solution:<br><br>Step 1: 4 hours per day * 5 days = 20 hours per week.<br>Step 2: Johnny can walk 3 dogs at once, so he can walk 20 / 3 = 6 three-dog walks per day. … Step 6: The answer is $360. Incorrect Step: Step 2.<br><br><br>Explanation: The incorrect step is Step 2. The calculation used in this step is not correct for finding the number of three-dog walks per day. Instead, we should calculate the number of 30-minute and 60-minute walks Johnny does. Correct Solution: … Step 2: Johnny works for 4 hours per day, which is 240 minutes. He walks 6 dogs for 60 minutes each, which takes 120 minutes. This leaves 120 minutes for 30-minute walks. … Step 6: The answer is 1500.<br><br>Note: The explanation is reasonable and the final answer is correct. But the new solution has issues on continuity: the new Step 2 does not exactly follow the original Step 1.<br><br>Generated Correction (Good)|
|---|

|Question: Tom, Tim, and Paul are collecting photos of cars. Paul has 10 photos more than Tim. Tim has one hundred photos less than the total amount of photos which is 152. How many photos does Tom have? Original Solution:<br><br>Step 1: 152 - 100 = 52.<br>Step 2: Tom has 52 photos.<br>Step 3: The answer is 52. Incorrect Step: Step 1.<br><br><br>Explanation: The step is incorrect because it subtracts 100 from the total number of photos (152) to find the number of photos Tim has. However, the problem states that Tim has 100 photos less than the total amount of photos, not the total amount of photos minus 100. Correct Solution: … Step 5: The answer is 38.<br><br>Note: Step 1 is correct and the mistake step is Step 2, because Tim, not Tom, has 52 photos.<br><br>Generated Correction (Poor)|
|---|

- Figure 13: Some examples of generated corrections and their quality levels under our human evaluation.

the number of correct corrections (i.e., corrections for which the final answer is correct) first increases and then decreases. We also calculate the success rate for correcting mistakes under each difficulty level, dividing the number of correct corrections by the total number of collected reasoning paths. Figure 10 shows that the success rate significantly drops with increasing the difficulty. These statistics reveals that there is still huge room for improving contemporary LLMs on correcting mistakes.

