[Figure 1]

## Mol-R1: Towards Explicit Long-CoT Reasoning in Molecule Discovery

### Jiatong Li1*, Weida Wang2,4*, Qinggang Zhang1*, Junxian Li3, Di Zhang4 Changmeng Zheng1, Shufei Zhang2†, Xiaoyong Wei1†, Qing Li1

1Hong Kong Polytechnic University 2Shanghai AI Lab 3Shanghai Jiao Tong University 4Fudan University

# arXiv:2508.08401v1[cs.CL]11Aug2025

##### Abstract

Large language models (LLMs), especially Explicit Long Chain-of-Thought (CoT) reasoning models like DeepSeekR1 and QWQ, have demonstrated powerful reasoning capabilities, achieving impressive performance in commonsense reasoning and mathematical inference. Despite their effectiveness, Long-CoT reasoning models are often criticized for their limited ability and low efficiency in knowledgeintensive domains such as molecule discovery. Success in this field requires a precise understanding of domain knowledge, including molecular structures and chemical principles, which is challenging due to the inherent complexity of molecular data and the scarcity of high-quality expert annotations. To bridge this gap, we introduce Mol-R1, a novel framework designed to improve explainability and reasoning performance of R1-like Explicit Long-CoT reasoning LLMs in textbased molecule generation. Our approach begins with a highquality reasoning dataset curated through Prior Regulation via In-context Distillation (PRID), a dedicated distillation strategy to effectively generate paired reasoning traces guided by prior regulations. Building upon this, we introduce MoIA, Molecular Iterative Adaptation, a sophisticated training strategy that iteratively combines Supervised Fine-tuning (SFT) with Reinforced Policy Optimization (RPO), tailored to boost the reasoning performance of R1-like reasoning models for molecule discovery. Finally, we examine the performance of Mol-R1 in the text-based molecule reasoning generation task, showing superior performance against existing baselines.

### Introduction

Molecule discovery focuses on finding new compounds with desired properties, essential for advances in drug development (Jayatunga et al. 2022), materials science (Higuchi et al. 2023), and sustainable chemistry (Garti et al. 2003). However, the vastness and complexity of chemical space make this process challenging. Traditional methods rely on fixed rules or templates (Walters and Barzilay 2020), making them costly, time-consuming, and limited in creativity.

Large language models (LLMs) offer a promising alternative, allowing chemists to explore chemical space using natural language and symbolic notations (Edwards, Zhai, and

*These authors contributed equally. †Corresponding authors. Emails: zhangshufei@pjlab.org.cn

and x1wei@polyu.edu.hk

###### Molecule Caption

###### Target Molecule

|[Figure 2]|
|---|

The molecule is a benzoate ester resulting from the formal condensation of the carboxy group of 2,5-dihydroxybenzoic acid with the hydroxy group of phenylmethanol. It has a role as a plant metabolite. It is a benzoate ester, a member of phenols and a benzyl ester. It derives from a 2,5 dihydroxybenzoic acid.

C1=CC=C(C=C1)COC(=O)C2=C(C=CC(=C2)O)O

|[Figure 3]|
|---|

Okay, ... Esterification involves the carboxylic acid group (COOH) of one molecule reacting with an alcohol's hydroxyl group (OH) ... Wait, maybe better to write it as O=C(O)C1=CC(O)=CC(O)=C1?

|[Figure 4]|
|---|

QWQ32B

... Wait, the benzyl group in the ester is CH2-Ph

... Thus, the final SMILES is: ...

O=C(Oc1c(O)ccc(O)c1)OC(Cc2ccccc2)

###### 5,337 words

|[Figure 5]|
|---|

We are given a description of the molecule ... So the core structure is: HO-C6H3(OH)-COO-CH2C6H5 ... Therefore, the structure becomes: O=C(OCc1ccccc1)c1cc(O)c(O)cc1 ... Our representation: O=C(OCc1ccccc1)c1c(O)ccc(O)c1 is acceptable ...

|[Figure 6]|
|---|

DeepSeekR1

O=C(OCc1ccccc1)c1cc(O)c(O)cc1

4,518 words

😊Hmm, I am now going to generate a SMILES representation ... The basic structure of a benzoate ester is a benzene ring (C1=CC=CC=C1) connected to a carboxylic acid group (C(=O)O)

|[Figure 7]|
|---|

|[Figure 8]|
|---|

... 2,5-dihydroxybenzoic acid is: C1=CC(=C(C=C1O)C(=O)O)O ... phenylmethanol is: CC1=CC=CC=C1O ...

Mol-R1 (Ours)

C1=CC=C(C=C1)COC(=O)C2=C(C=CC(=C2)O)O

428 words

Figure 1: Derivations of a SMILES representation based on its caption. We compare the reasoning traces and final predictions of QWQ-32B (Team-Qwen 2025), DeepSeek-R1 (Guo et al. 2025), and Our Mol-R1. The results show that QWQ-32B and DeepSeek-R1 tend to exhibit excessively long reasoning traces, yet still fail to yield correct results.

Ji 2021). Thanks to the large training corpora, LLMs can reason, follow instructions, and learn in context, enabling more flexible and efficient text-guided molecular discovery. For example, MolT5 (Edwards et al. 2022) introduced a molecule-caption translation approach using the ChEBI20 dataset, aligning SMILES strings (Weininger 1988) with natural language descriptions. This allows chemists to design molecules simply by describing their requirements, which LLMs can then translate into specific molecular structures (Li et al. 2024a).

Although LLM-based approaches are effective, most of them follow a translation paradigm that focuses directly generating the final results, lacking intermediate reasoning steps and interpretability. However, in real-world applications like drug development, understanding the design process is crucial. Without this transparency, critical properties, especially

those related to safety, may be overlooked.

Recently, DeepSeek-R1 (Guo et al. 2025) has emerged as a breakthrough model, excelling in mathematical inference and commonsense reasoning. This raises the question of whether its reasoning abilities can be applied to molecule discovery. However, directly applying R1-like reasoning in molecule discovery is challenging due to (i) the cold-start issue: R1-like reasoning initially depends on a small yet high-quality reasoning dataset for cold-start training, one that seamlessly integrates reasoning traces with the corresponding captions and molecules. However, existing datasets like ChEBI-20 lack such traces, and manual annotation is prohibitively expensive due to the need for expert chemical knowledge. Moreover, automated generation via rejection sampling is inefficient due to a lack of molecular knowledge for text-based molecule generation. (ii) Lack of clear guidance during policy training: R1’s optimization highly relies on outcome-based rewards through Group Relative Policy Optimization (GRPO), which only assesses the quality of reasoning traces after reasoning completes. This post-regulation mechanism causes LLMs to produce numerous erroneous and unstable reasoning processes during training. Erroneous intermediate logic propagates freely because reward signals evaluate only final outputs, not reasoning pathway validity. This critical disconnect would cause several cascading failures, including chemically invalid assumptions becoming reinforced patterns rather than corrected deviations, and models increasingly rely on superficial correlations instead of causal principles. As a result, the model rapidly plateaus at superficial pattern recognition, unable to evolve beyond its initial knowledge base or discover novel, rule-compliant molecular structures.

To address the above challenges, we introduce Mol-R1, a novel R1-like Explicit Long-CoT reasoning framework tailored for molecule discovery, enabling LLMs to mimic the deliberation of chemists to generate complex molecule structures. Specifically, Mol-R1 aims to address two fundamental research questions: (i) How to efficiently generate high-quality, expert-aligned reasoning traces for molecule discovery using minimal expert annotations? and (ii) How could we effectively leverage these reasoning traces to train LLMs, enhancing molecular reasoning capability while ensuring training stability? Our main contributions are summarized as follows:

- • We introduce Mol-R1, a novel framework to improve explainability and reasoning performance of R1-like reasoning models in text-based molecule generation.
- • Mol-R1 began with a novel distillation strategy, PRID, to curate a unique reasoning dataset with a human-labelled in-context example and logic regulations.
- • Leveraging the cold-start dataset, we introduced MoIA, which iteratively combines the Supervised Fine-tuning and Reinforced Policy Optimization, to improve the performance of Mol-R1 for molecule reasoning.
- • Comprehensive experiments validate the effectiveness of Mol-R1, demonstrating its significant potential to enable more explainable and chemist-like reasoning in molecule discovery.

### Related Work

#### LLMs for Molecule Discovery

LLMs have demonstrated great potential in molecule discovery, encompassing areas such as molecular understanding, optimization, and generation. To bridge the gap between molecular representations and natural language texts, MolT5 (Edwards et al. 2022) first proposed the molecule-caption translation task, introducing the ChEBI-20 dataset, which contains pairs of molecule SMILES representations and their textual descriptions detailing structural patterns and chemical properties. Building upon this, works like MolReGPT (Li et al. 2024b), ICMA (Li et al. 2025), ReactGPT (Chen et al. 2025a), and MolReFlect (Li et al. 2024c) took advantage of the in-context learning capability of LLMs to enhance the performance of LLMs in molecule-related tasks, while MoMu (Su et al. 2022), MolCA (Liu et al. 2023), and 3D-MoLM (Li et al. 2024d) were trying to incorporate multi-modal information to help LLMs better understand molecules.

#### Long-CoT Reasoning with LLMs

With the scale of parameters, LLMs have shown powerful reasoning capability. The introduction of chain-of-thought (CoT) (Wei et al. 2022) has brought a revolutionary change to the utilization of LLMs’ reasoning capabilities. CoT enables models to conduct more in-depth and systematic reasoning, thereby obtaining more reliable and accurate answers, which has laid a foundation for a series of subsequent methods that guide LLMs to reason. What’s more, methods such as Tree of Thought (ToT) (Yao et al. 2023) and Graph of Thought (GoT) (Besta et al. 2024) have emerged. These methods construct tree or graph structures to explore problem spaces, enabling models to trace different reasoning paths and thereby find better solutions to complex problems. Furthermore, OpenAI’s o1 (Jaech et al. 2024) and Deepseek’s R1 (Guo et al. 2025), among others, made further innovations in this regard. They entrusted the exploration and decision-making of the Long-CoT reasoning process to the LLMs themselves through reinforcement learning (RL), which enhanced the reasoning capabilities of models, enabling them to make more flexible and intelligent judgments when facing complex problems.

### Problem Formulation

#### Notation

Considering the molecule caption as the query x, the corresponding molecule SMILES string as the output y, and the LLM parameters as θ, we could define a conditional distribution over the output tokens: πθ(yt|x,y<t).

Given a training set D = {(xi,yi)}Ni=1 with N molecule SMILES strings and description pairs, the loss function for learning the policy πθ via vanilla supervised fine-tuning can be denoted as:

 −

 , (1)

|yi|

N

L(θ) =

log πθ(yi;t|xi,yi;<t)

t=1

i=1

where |yi| is the length of the i-th molecule SMILES string.

[Figure 9]

Caption:The molecule is a cyclodiene ...

Please predict the SMILES representation of the molecule. You need to reason step by step to reach the final answer.

###### Raw Training

###### LLM

###### Expert Annotated Context

Dataset Caption: The molecule is a cyclodiene ... Reasoning Trace:😊 Okay, I need to first ... 🤔Second... Molecule: C1C2C(C(C1Cl)Cl)C3(C(=C(C2(C3(Cl)Cl)Cl)Cl)Cl)Cl

[Figure 10]

Concat

Requery

[Figure 11]

LLM

###### ......

Compare the candidates with the Ground Truth molecule

###### Reasoning Trace

Raw Training Dataset

Caption: The molecule is an aromatic ether ...

Reasoning Trace 1

Reasoning Trace 2 Reasoning Trace 3 Reasoning Trace 4

Molecule: C=CCOC1=C(C=C(C=C1)CC(=O)O)Cl Please imitate the example and learn the reasoning regulations to give the reasoning

Post Validation

[Figure 12]

[Figure 13]

| |
|---|

|[Figure 14]|
|---|

[Figure 15]

[Figure 16]

trace, rather than assuming you have already known the answer from the beginning.

Fail

Pass

Ground Truth

(a) Rejection Sampling (RS)

###### (b) Prior Regulation via In-context Distillation (PRID)

###### Raw Training Dataset

Ground Truth

, save

Raw Training Pred Trace Dataset Pred Trace

Compare

[Figure 17]

Caption Molecule

Pred Molecule Pred Molecule Pred Molecule

| | |
|---|---|
| | |
| | |

[Figure 18]

###### Reward Model

Caption

Select

Examples

Pred Trace

......

Reasoning Training Set

Pred Trace Pred Molecule

Reward Ground Truth

Mol-R1

[Figure 19]

④ Rejection Re-Sampling

① PRID

[Figure 20]

Caption Trace Molecule

Caption Trace Molecule

🔥Parameter Update

Reasoning Traces

Mol-R1

[Figure 21]

🔥Parameter Update

Raw Training Dataset

Caption

Trace

Auto-regressive Loss

③ Reinforced Policy Optimization

Mol-R1

Iteratively Repeat ② ~ ④ Unitil:

Cold-start Data Preparation

② Supervised Fine-tuning

1) converges Or 2)

(c) Molecular Iterative Adaptation (MoIA)

Figure 2: The overall framework of Mol-R1. (a) Applying Rejection Sampling to distill reasoning traces. (b) Using Prior Regulation via In-context Distillation to yield cold-start reasoning traces. (c) The pipeline of Molecular Iterative Adaptation (MoIA).

#### R1-like Explicit Long-CoT Reasoning Models

Explicit Long-CoT reasoning models involve complex reasoning traces before giving the final answer, enabling them to explore and handle complex problems (Chen et al. 2025b). Normally, Explicit Long-CoT reasoning models follow a strict output format, enclosing their reasoning traces between two special tokens <think> and </think>, while the final answer will be placed between <answer> and </answer>. Compared to the previous CoT method (Wei et al. 2022) that utilizes prompts like Let’s think step by step, R1-like Explicit Long-CoT reasoning models are specifically trained and optimized for reasoning tasks, which encourages them to have a deeper, broader, and more reflective reasoning process before yielding the final answer.

#### Text-based Molecule Reasoning Generation

In this work, we formalize the text-based molecule generation as a reasoning task. Different from directly learning the policy πθ, molecule reasoning generation requires LLMs to first generate their reasoning traces and then infer the final molecule SMILES string based on both the input query and their reasoning process. Therefore, given a molecule description x, the reasoning process z, and the corresponding molecule SMILES string y, the molecule reasoning involves the following two-stage decoding process with the same LLM parameterized by θˆ:

Reasoning Process Generation. First, we define a con-

ditional distribution over the reasoning process tokens: pθˆ(zt|x,z<t). In this stage, the LLM tries to utilize its chemical knowledge to interpret the descriptions into viable chemical structures.

Final Answer Generation. Next, the LLM generates the final SMILES string based on both the input description x and its reasoning process z, with the conditional distribution: qθˆ(yt|x,z,y<t).

In this case, molecule reasoning generation requires a special reasoning training set R = {(xi,zi,yi)} (where (xi,yi) ∈ D) to initialize the policies q and p via supervised fine-tuning (i.e., cold-start training, CS). Similarly, the objective for supervised fine-tuning can be denoted as:

 

|R|

|zi|

L(θ) = −

log pθ(zi,t|xi,zi,<t)

t=1

i=1

  (2)

|yi|

log qθ(yi,t|xi,zi,yi,<t)

+

t=1

where |zi| and |yi| are the length of the reasoning process and molecule SMILES string, respectively, while |R| denotes the number of examples in R.

Text-based molecule reasoning generation simulates the problem-solving process of human experts by progressively analyzing the captions and inferring potential molecule structures, which could potentially enhance the explainability of the final predictions.

### The Framework of Mol-R1

In this section, we introduce the framework of Mol-R1. As

- shown in Figure 2, Mol-R1 consists of two important components, Prior Regulation via In-context Distillation (PRID) and Molecular Iterative Adaptation (MoIA). We first detail PRID, which targets the post-hoc regulation of the cold-start data. Subsequently, we present the design of MoIA, which iteratively involves dynamic interactions between Supervised Fine-tuning (SFT) and Reinforced Policy Optimization (RPO).

#### Prior Regulation via In-context Distillation

Facilitating Explicit Long-CoT reasoning for text-based molecule generation initially depends on a small yet highquality reasoning dataset for cold-start training, one that seamlessly integrates reasoning traces with the corresponding captions and molecules. However, the curation of the cold-start reasoning training set of Mol-R1 mainly faces the following challenges:

- 1. Molecule-caption datasets, such as ChEBI-20 (Edwards et al. 2022), lack the crucial reasoning traces that link captions with molecules. Annotating such reasoning traces is often costly and time-consuming due to the inherent complexity of molecular structure and the demand for highly specialized knowledge from domain experts.
- 2. Existing LLMs often fail to infer correct molecules through rejection sampling (Muennighoff et al. 2025), due to a lack of molecular knowledge, consequently lowering the sampling efficiency.
- 3. The reasoning traces distilled from existing Explicit Long-CoT reasoning models via rejection sampling tend to broadly explore the reasoning space. However, molecular reasoning demands precise knowledge and rigorous logic, necessitating a careful balance between the breadth of exploration and the precision of knowledge.

To overcome these challenges, we propose Prior Regulation via In-context Distillation for cold-start reasoning trace distillation. Guided by the high-quality reasoning example in PRID, LLM can learn from the inherent logic and regulations within the reasoning process. The detailed pipeline is demonstrated in Figure 2 (b):

Firstly, a human expert writes only one high-quality reasoning trace example c with prior regulations for text-based molecule generation. This example focuses on interpreting potential molecular structures from the descriptions and guiding the reasoning process of prior regulations.

Secondly, we perform in-context distillation based on the expert example. Here, expert annotation could serve as a guideline to prompt LLMs to encapsulate the underlying logical patterns and reasoning principles, providing soft boundaries for the reasoning process. Our approach strategically leverages a rather small subset of m(m ≪ N) examples from the raw training set D, while preserving free exploration of the remaining space.

Thirdly, we introduce a post-validation stage to examine the quality of reasoning traces. If the reasoning trace fails to pass the evaluation of the post-validator, we will proceed with a

re-query. Specifically, we implement a format validator and a score validator. The former is to ensure that the reasoning traces are enclosed between <think> and </think>, while the latter will be performed by adopting the LLM as a judge to score the generated reasoning trace based on the precision of the knowledge and logical coherence.

#### MoIA: Molecular Iterative Adaptation

Training Explicit Long-CoT reasoning models typically begins with an initial cold-start supervised fine-tuning stage, which introduces foundational, deterministic knowledge and establishes a consistent format for reasoning traces. Following the cold-start SFT, the model undergoes Reinforced Policy Optimization on a larger amount of data without reasoning traces, which enables LLMs to learn the molecule SMILES syntax via policies.

However, we observe that in the previous training pipeline, the reasoning model performance will quickly converge during the RPO stage, indicating a knowledge bottleneck. Therefore, to achieve better results, we need to incorporate more deterministic or established knowledge supervision into the training process.

Building on this insight, we propose MoIA, an iterative training framework that iteratively involves dynamic interactions between SFT and RPO. Specifically, we define the iteration of MoIA as T. We will iteratively update the reasoning training set RT, the reasoning traces ziT and optimize the model parameters θT. Figure 2 (c) demonstrates the detailed pipeline of MoIA:

- ① As discussed previously, we conduct the PRID process for cold-start data preparation, based on the prior guidance from expert annotated example c:

zi0 = PRID(xi,yi|c), (3)

where the superscript in zi0 refers to the initial iteration of MoIA (T = 0). Then, the distilled reasoning traces will form

the reasoning training set RT at the initial iteration:

RT = {(xi,ziT,yi)}, (4)

- ② We then introduce deterministic knowledge and perform supervised fine-tuning on the reasoning training set RT by minimizing the autoregressive loss as mentioned in Equation

(2), thereby updating the model parameters θT → θˆT: θˆT = argmin

θ

L(θ), (5)

- ③ We conduct RPO on the raw training dataset D. For each instance i, we provide only the input caption xi to the LLM,

allowing it to explore its reasoning trace zˆiT and final prediction yˆi. Subsequently, the complete generated output as well as the ground truth yi is fed into a reward model RM to acquire a feedback reward ri:

ri = RM(ˆziT,yˆi,yi), (6) This acquired reward ri provides crucial feedback on the

molecule generation policy and is then used to update the

LLM’s parameters from θˆT to θT+1 by maximizing the expected reward:

θT+1 = argmax

i,yˆi)∼Pθ(·|xi)[ri], (7) where E(ˆz

E(ˆz

θ

i,yˆi)∼Pθ(·|xi)[ri] denotes the expected reward for the generated reasoning trace zˆiT and final prediction yˆi sampled from the LLM’s current policy P, given input xi.

④ We apply rejection re-sampling to update and improve the quality of the reasoning training set for the next iteration. Using the newly optimized model parameters θT+1, we generate multiple candidate reasoning traces z˜i,kT and predictions y˜i,k for each instance i in the raw training dataset D, where k denotes the k-th rejection sampling attempt. Crucially, only those generated results where the final prediction y˜i,k is completely identical to the ground truth yi are accepted. Therefore, we have:

(xi,z˜i,kT ,yi) |

###### RT+1 =

i∈D

∃k : (˜zi,kT ,y˜i,k) ∼ PθT+1(·|xi) ∧ y˜i,k = yi , (8) where yi represents the ground truth molecule for input caption xi, and only generated outputs (˜zi,kT ,y˜i,k) where the prediction y˜i,k perfectly matches this ground truth are included. The existential quantifier ∃k : signifies that if, for a given xi, at least one of the k rejection sampling attempts yields a correct prediction, that specific successful triplet (xi,z˜i,kT ,yi), where ziT+1 = z˜i,kT , is included in RT+1. This process effectively curates and amplifies correct and deterministic knowledge within the accepted reasoning traces, thereby significantly enhancing the molecular understanding and overall reasoning capability of the LLM.

Finally, we let T = T + 1, and iteratively repeat the step

② - ④ until:

- 1) θT converges, i.e., the performance of Mol-R1 stabilizes;
- 2) |RT| == |D|, i.e., the entire raw training dataset is fully annotated with reasoning traces.

### Experiments

#### Implementation Details

To distillate reasoning traces for cold-start training, we mainly select GPT-4o (Hurst et al. 2024). Additionally, gemma-3-27b-it (abbreviated gemma3-27B) and gemma3-12b-it (abbreviated gemma3-12B) (Team-Gemma et al. 2025) are adopted via rejection sampling for comparison. We utilize the vllm1 and OpenAI2 frameworks for querying the LLMs. In all, we sampled 1,053 reasoning traces for the cold-start training of Mol-R1. Meanwhile, we merge the rejection sampling results of gemma3-27B and gemma3-12B, obtaining 2,943 valid reasoning traces.

Meanwhile, we adopt Llama-3.1-8B-Instruct (abbreviated llama3.1-8B) (Dubey et al. 2024) as the model designated for MoIA training. Here, the llama-factory3 is adopted for supervised fine-tuning, and the open-r14 framework and

- 1https://blog.vllm.ai
- 2https://openai.com/api/
- 3https://llamafactory.readthedocs.io
- 4https://open-r1.com/

Group Relative Policy Optimization (GRPO) (Guo et al. 2025) are employed for the implementation of reinforced policy optimization.

#### Evaluation Metrics

We evaluate the performance of Mol-R1 from two distinct perspectives:

Accuracy of Generated Molecules. We adopt BLEU, Exact Match (EM), Levenshtein distance, Validity, and Molecule Fingerprints scores (specifically, MACCS FTS, RDK FTS, and Morgan FTS) to evaluate the similarity between the predictions and the ground truth.

Quality of Reasoning Traces. Effective reasoning traces inherently enhance the interpretability and trustworthiness of the generated molecule. Conversely, errors in the reasoning traces allow chemists to identify underlying flaws and preemptively assess the reliability of the final molecular answer. To evaluate the quality of reasoning traces, we use Gemma327B to mimic a chemist’s judgment of the reasoning traces. The judge will predict whether the reasoning trace itself could lead to a correct conclusion solely based on its content and without access to the final answer. Then, we compare the judge’s prediction against the actual correctness of the final answer. If the judge’s prediction is the same to the actual correctness of the generated molecule, we define the reasoning trace as consistent. Finally, we report the F1 score of the consistency (i.e., Consistent-F1) to indicate the quality of generated reasoning traces.

| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |
| | | | | | | | | | |

Figure 3: Comparison of reasoning trace quality across different models, where higher values of the Consistent-F1 score indicate more reliable and robust reasoning.

#### Result & Discussion

In this work, we specifically include three powerful Explicit Long-CoT reasoning LLMs—Llama3-8B-R1-distilled (Guo et al. 2025), QWQ-32B (Team-Qwen 2025), and DeepSeek-R1 (Guo et al. 2025)—as baselines. Concurrently, we demonstrate the performance of the base model using the same molecule-caption pairs with the cold-start

Method BLEU↑ EM↑ Levenshtein↓ MACCS FTS↑ RDK FTS↑ Morgan FTS↑ Validity↑

Baselines (w/ Reasoning) Llama3-8B-R1-distilled 0.097 0.011 64.62 0.442 0.270 0.213 0.228

QWQ-32B 0.181 0.032 62.89 0.584 0.379 0.331 0.518 DeepSeek-R1 0.031 0.156 68.47 0.732 0.600 0.580 0.522

Base Model SFT (w/o Reasoning) Llama3.1-8B (n=1053) 0.375 0.021 95.34 0.676 0.502 0.416 0.612

Mol-R1 Cold-start Configurations (w/ Reasoning) Mol-R1-RS-G (n=2943) 0.179 0.047 56.42 0.623 0.385 0.329 0.701 Mol-R1-PRID-G (n=1053) 0.605 0.015 40.08 0.718 0.503 0.417 0.675 Mol-R1-PRID-4o (n=1053) 0.636 0.038 39.67 0.761 0.549 0.475 0.750

###### Mol-R1 MoIA Iterations(w/ Reasoning)

- Mol-R1 (T=0) 0.650 0.144 34.33 0.804 0.639 0.558 0.819

- Mol-R1 (T=1) 0.655 0.216 32.87 0.821 0.673 0.596 0.863

- Mol-R1 (T=2) 0.641 0.234 32.94 0.823 0.684 0.612 0.847

Table 1: Overall performance comparison. The best results are bold and the second-best results are underlined.

reasoning data to ablate the effectiveness of explicit reasoning traces. Furthermore, we report the cold-start performance of Mol-R1 under various cold-start configurations, as well as the performance across various MoIA iterations. Notably, Table 1 presents a comparison of molecule generation accuracy across the above models, while Figure 3 illustrates the Consistent-F1 scores of their respective reasoning traces.

Generally, Mol-R1 (T=2) demonstrates superior performance in molecule generation accuracy, achieving an EM score of 0.234 and the highest Molecule FTS metrics across all these models. When directly compared to even the most advanced Explicit Long-CoT reasoning baselines, such as QWQ-32B and DeepSeek-R1, Mol-R1 (T=2) significantly outperforms QWQ-32B with a BLEU score that is a staggering 354% higher, indicating a much more similar prediction to reference molecules. Furthermore, Mol-R1 (T=2) also obtains an EM score that is 1.5 times better than DeepSeekR1, showcasing its enhanced ability to generate accurate molecules based on the text descriptions.

Beyond generation accuracy, Mol-R1 (T=2) also achieves the highest Consistent-F1 score for its reasoning trace quality. This indicates that in most cases, the accuracy of the predictions can be effectively pre-assessed by analyzing its reasoning traces. This highlights the exceptional explainability and robustness of Mol-R1.

Next, we focus on ablating the key components of MolR1 with extensive experiments: (i) Assessing the effectiveness of PRID in efficiently generating high-quality, expertaligned reasoning traces; and (ii) the necessity of applying MoIA compared to exclusive policy optimization.

Assessment of Cold-start Configurations In this part, we prioritize the performance of Prior Regulation via In-context Distillation (PRID) by examining the impacts of different models and algorithms for distilling the cold-start reasoning data for Mol-R1 (i.e., cold-start configurations).

First of all, we test the supervised fine-tuning performance of the base model, Llama3.1-8B, with and without reasoning traces at the same data scale (i.e., n=1053). The results shown in Table 1 demonstrate that with the same 1,053

molecule-caption pairs, Mol-R1-PRID-4o, trained with reasoning traces distilled from GPT-4o via PRID, achieves a significant improvement over the Llama3.1-8B (the base model w/o reasoning). Notably, Mol-R1-PRID-4o achieves a BLEU score of 0.636 compared to 0.375 for Llama3.18B, and an exact match score of 0.038 versus 0.021, indicating substantial gains in molecule generation accuracy attributable to the inclusion of reasoning traces.

Then, we investigated the impacts of using a different LLM for PRID, introducing gemma3-12B for comparison. We observed that Mol-R1-PRID-G (using gemma3-12B) and Mol-R1-PRID-4o (using GPT-4o), both utilizing PRID for reasoning trace distillation, achieved comparable results in terms of molecule generation accuracy with close BLEU scores, and other metrics like Molecule FTS scores and Validity, also indicated in Table 1. This suggests that PRID is generally effective and robust across different LLMs.

However, when we consider the quality of reasoning traces, a notable divergence appears. Mol-R1-PRID-G achieves a significantly lower Consistent-F1 score compared to Mol-R1-PRID-4o. This indicates that while both models can generate molecules with similar accuracy, the reasoning traces produced by gemma3-12B are considerably less consistent or reliable for pre-assessing prediction accuracy. In essence, although gemma3-12B can contribute to effective molecule generation, the robustness and explainability of its underlying reasoning traces are not on par with GPT4o. This highlights that while PRID is effective, its full potential is realized when paired with a more powerful LLM, underscoring the importance of model choice.

Finally, we compared PRID with Rejection Sampling (RS). Specifically, we evaluated Mol-R1-RS-G, which was trained on 2,943 reasoning traces distilled using gemma312B and gemma3-27B via Rejection Sampling. The results are also shown in Table 1. It can be observed that, despite being trained on a considerably larger dataset for cold-start (2,943 data points for RS vs. 1,053 for PRID), Mol-R1-RSG still achieves inferior performance. For instance, Mol-R1RS-G obtained a BLEU score of 0.179 and a Morgan FTS score of 0.329, which are even lower than those obtained by

Figure 4: Training and inference dynamics between exclusive GRPO (above) and MoIA iterations (below).

Mol-R1-PRID-G.

These findings suggest that for highly specialized domain tasks like text-based molecule reasoning generation, the distillation of reasoning traces requires specific regulations. Rejection Sampling, which allows LLMs to explore more freely, appears to introduce excessive noise and unnecessary reasoning paths into the reasoning process, which ultimately compromises training performance. In contrast, PRID, by incorporating the expert-annotated example and regulations, provides LLMs with robust guidance on how to reason effectively to produce the target molecule, which is crucial for improving the performance on text-based molecule reasoning generation.

Ablation of Iterative Training Figure 4 illustrates the distinct training dynamics between exclusive GRPO and MoIA. For comparison, we initiated exclusive GRPO training immediately following the supervised fine-tuning at the first iteration (T=0) of MoIA.

A clear distinction emerges: the exact match (EM) reward for exclusive GRPO converges rapidly around 0.14 after approximately 800 steps. Critically, the training remains unstable with significant fluctuations throughout the remaining steps, indicating a premature plateau and lack of sustained improvement.

In contrast, within the MoIA framework, the exact match reward continues to improve and does not fully converge until roughly 2,000 steps of policy optimization. Meanwhile, as

- shown in Figure 3, the Consistent-F1 score continuously improves, reflecting a consistent enhancement in the quality of reasoning traces as MoIA iterations advance. This prolonged and stable improvement demonstrates how MoIA’s iterative training strategy effectively solidifies deterministic knowledge, enabling the model to continuously self-evolve, which is crucial to improve both the reasoning capability and the

accuracy of molecule generation.

###### Molecule Caption

###### Target Molecule

[Figure 22]

The molecule is a dicarboxylic acid monoanion that is the conjugate base of O-methylmalonylcarnitine; major species at pH 7.3. It is a conjugate base of an Omethylmalonylcarnitine.

CC(C(=O)[O-])C(=O)OC(CC(=O)[O-])C[N+](C)(C)C

😊Hmm, I need to generate ... The structure of carnitine is C[N+](C)(C)C(CO)C(=O)O, and the structure of malonic acid is C(C(=O)O)C(=O)O ... The structure of O-methylmalonylcarnitine can also be written as C(C(=O)O)C(C(=O)O)OC(CC(=O)O)C[N+](C) (C)C ... Finally, I need to check if this molecule meets our expectations ...

| |
|---|

[Figure 23]

|[Figure 24]|
|---|

Mol-R1

- (T=0)

Mol-R1

- (T=1)

Mol-R1

- (T=2)

C[N+](C)(C)CC(CC(=O)O)OC(CC(=O)[O-])C(=O)O

😊 Hmm, I am now going to ... The SMILES representation of carnitine is: C[N+](C) (C)CC(CO)C(=O)O. 🤔 ... The SMILES for methylmalonyl group is: CC(C(=O)[O-])C(=O)O ...

|[Figure 25]|
|---|

|[Figure 26]|
|---|

replace the hydroxyl group with an O-methylmalonyl group: C[N+](C)(C)CC(COC(C(=O)C)C(=O)[O-])C(=O)O ... Finally, we need to check ...

CC(=O)C(OCC(C[N+](C)(C)C)C(=O)O)C(=O)[O-]

| |
|---|

[Figure 27]

😊 Hmm, I am now🤔... The SMILES for carnitine can be written as: C[N+](C)(C)CC(CO)C(=O)O ... The SMILES for methylmalonyl group is: CC(C(=O)O)C(=O)O ... the carboxyl group of Omethylmalonylcarnitine has lost a proton, forming a carboxylate ion ...

|[Figure 28]|
|---|

CC(C(=O)[O-])C(=O)OC(CC(=O)[O-])C[N+](C)(C)C

Figure 5: Case study of the reasoning traces and final predictions across three MoIA iterations (T=0, 1, 2) of Mol-R1.

#### Case Study

Here, we present a detailed example, shown in Figure 5, to demonstrate the evolution of Mol-R1 across MoIA iterations. In this example, models need to identify the SMILES representation of ”O-methylmalonylcarnitine” and then determine which carboxyl group (-COOH) loses a proton to form the monoanion, or which additional carboxyl proton is lost if the molecule has multiple acidic sites.

At the initial iteration (T=0), guided by the PRID mechanism, the model naturally began to reflect on its previous reasoning. However, the generated molecule still contained errors; while the overall structure was similar, the two hydroxyl groups were not fully ionized as required.

Moving to the next iteration (T=1), the model recognized the need to modify or replace specific groups (e.g., handling the deprotonation of hydroxyl groups). Yet, this process remained incomplete, resulting in one of the hydroxyl groups not being fully ionized.

Finally, at MoIA (T=2), through iterative optimization and verification of its knowledge, the model accurately generated the SMILES representation that perfectly matched the target molecule. This progression clearly shows the model’s ability to overcome complex molecular structure generation hurdles through MoIA.

### Conclusion

In this work, we introduce Mol-R1, an innovative framework specifically designed to enhance explainability and reasoning performance of R1-like Explicit Long-CoT reasoning LLMs in text-based molecule generation. Mol-R1 began with Prior Regulation via In-context Distillation (PRID), which effectively curated a high-quality reasoning dataset via in-context learning guided by an expert-annotated example, addressing the inherent complexities and knowledge scarcity of reasoning traces in molecule discovery, Building

upon this, we then introduced Molecular Iterative Adaptation (MoIA), which iteratively combines Supervised Finetuning (SFT) with Reinforced Policy Optimization (RPO), enabling Mol-R1 to achieve superior performance in textbased molecule reasoning generation.

### References

Besta, M.; Blach, N.; Kubicek, A.; Gerstenberger, R.; Podstawski, M.; Gianinazzi, L.; Gajda, J.; Lehmann, T.; Niewiadomski, H.; Nyczyk, P.; et al. 2024. Graph of thoughts: Solving elaborate problems with large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 17682–17690.

Chen, Z.; Fang, Z.; Tian, W.; Long, Z.; Sun, C.; Chen, Y.; Yuan, H.; Li, H.; and Lan, M. 2025a. ReactGPT: Understanding of Chemical Reactions via In-Context Tuning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, 84–92.

Chen, Z.; Min, Y.; Zhang, B.; Chen, J.; Jiang, J.; Cheng, D.; Zhao, W. X.; Liu, Z.; Miao, X.; Lu, Y.; et al. 2025b. An empirical study on eliciting and improving r1-like reasoning models. arXiv preprint arXiv:2503.04548.

Chiang, C.-H.; and Lee, H.-Y. 2024. Over-Reasoning and Redundant Calculation of Large Language Models. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 2: Short Papers), 161–169.

Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle, A.; Letman, A.; Mathur, A.; Schelten, A.; Yang, A.; Fan, A.; et al. 2024. The llama 3 herd of models. arXiv e-prints, arXiv–2407.

Edwards, C.; Lai, T.; Ros, K.; Honke, G.; Cho, K.; and Ji, H. 2022. Translation between Molecules and Natural Language. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 375–413.

Edwards, C.; Zhai, C.; and Ji, H. 2021. Text2mol: Crossmodal molecule retrieval with natural language queries. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 595–607.

Garti, N.; Yaghmur, A.; Aserin, A.; Spernath, A.; Elfakess,

- R.; and Ezrahi, S. 2003. Solubilization of active molecules in microemulsions for improved environmental protection. Colloids and Surfaces A: Physicochemical and Engineering Aspects, 230(1-3): 183–190.

Guo, D.; Yang, D.; Zhang, H.; Song, J.; Zhang, R.; Xu, R.; Zhu, Q.; Ma, S.; Wang, P.; Bi, X.; et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Higuchi, A.; Sung, T.-C.; Wang, T.; Ling, Q.-D.; Kumar,

- S. S.; Hsu, S.-T.; and Umezawa, A. 2023. Material Design for Next-Generation mRNA Vaccines Using Lipid Nanoparticles. Polymer Reviews, 63(2): 394–436.

Hurst, A.; Lerer, A.; Goucher, A. P.; Perelman, A.; Ramesh, A.; Clark, A.; Ostrow, A.; Welihinda, A.; Hayes, A.; Radford, A.; et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276.

Jaech, A.; Kalai, A.; Lerer, A.; Richardson, A.; El-Kishky, A.; Low, A.; Helyar, A.; Madry, A.; Beutel, A.; Carney, A.; et al. 2024. Openai o1 system card. arXiv preprint arXiv:2412.16720.

Jayatunga, M. K.; Xie, W.; Ruder, L.; Schulze, U.; and Meier, C. 2022. AI in small-molecule drug discovery: a coming wave. Nat. Rev. Drug Discov, 21(3): 175–176.

Li, J.; Li, J.; Liu, Y.; Zhou, D.; and Li, Q. 2024a. TOMGBench: Evaluating LLMs on text-based open molecule generation. arXiv preprint arXiv:2412.14642.

Li, J.; Liu, W.; Ding, Z.; Fan, W.; Li, Y.; and Li, Q. 2025. Large language models are in-context molecule learners. IEEE Transactions on Knowledge and Data Engineering.

Li, J.; Liu, Y.; Fan, W.; Wei, X.-Y.; Liu, H.; Tang, J.; and Li, Q. 2024b. Empowering molecule discovery for moleculecaption translation with large language models: A chatgpt perspective. IEEE transactions on knowledge and data engineering.

Li, J.; Liu, Y.; Liu, W.; Le, J.; Zhang, D.; Fan, W.; Zhou, D.; Li, Y.; and Li, Q. 2024c. MolReFlect: Towards InContext Fine-grained Alignments between Molecules and Texts. arXiv preprint arXiv:2411.14721.

Li, S.; Liu, Z.; Luo, Y.; Wang, X.; He, X.; Kawaguchi, K.; Chua, T.-S.; and Tian, Q. 2024d. Towards 3D Molecule-Text Interpretation in Language Models. In The Twelfth International Conference on Learning Representations.

Liu, Z.; Li, S.; Luo, Y.; Fei, H.; Cao, Y.; Kawaguchi, K.; Wang, X.; and Chua, T.-S. 2023. MolCA: Molecular GraphLanguage Modeling with Cross-Modal Projector and UniModal Adapter. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 15623–15638.

Muennighoff, N.; Yang, Z.; Shi, W.; Li, X. L.; Fei-Fei, L.; Hajishirzi, H.; Zettlemoyer, L.; Liang, P.; Cand`es, E.; and Hashimoto, T. 2025. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393.

Schulman, J.; Wolski, F.; Dhariwal, P.; Radford, A.; and Klimov, O. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Su, B.; Du, D.; Yang, Z.; Zhou, Y.; Li, J.; Rao, A.; Sun, H.; Lu, Z.; and Wen, J.-R. 2022. A molecular multimodal foundation model associating molecule graphs with natural language. arXiv preprint arXiv:2209.05481.

Team-Gemma; Kamath, A.; Ferret, J.; Pathak, S.; Vieillard, N.; Merhej, R.; Perrin, S.; Matejovicova, T.; Ram´e, A.; Rivi`ere, M.; et al. 2025. Gemma 3 technical report. arXiv preprint arXiv:2503.19786.

Team-Qwen. 2025. QwQ-32B: Embracing the Power of Reinforcement Learning.

Walters, W. P.; and Barzilay, R. 2020. Applications of deep learning in molecule generation and molecular property prediction. Accounts of chemical research, 54(2): 263–270.

Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Xia, F.; Chi, E.; Le, Q. V.; Zhou, D.; et al. 2022. Chain-ofthought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35: 24824–24837.

Weininger, D. 1988. SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules. Journal of chemical information and computer sciences, 28(1): 31–36.

Yao, S.; Yu, D.; Zhao, J.; Shafran, I.; Griffiths, T.; Cao, Y.; and Narasimhan, K. 2023. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36: 11809–11822.

### SUPPLEMENTARY MATERIALS Implementation Details

In this section, we list extra implementation details that help readers fully understand our approach, including hyper-parameters, statistics of the cold-start data, and the algorithm selected for Reinforced Policy Optimization.

#### Hyper-parameters

Item Value Supervised Fine-tuning (SFT)

gpu number (A800) 2 per device train batch size 1

gradient accumulation steps 4

learning rate 1.0e-5 num train epochs 5 lr scheduler type cosine warmup ratio 0.1 Reinforced Policy Optimization (RPO)

gpu number (A800) 8

learning rate 1.0e-6 weight decay 1.0e-2

kl coef 1.0e-2 n 5

rollout.temperature 1.0 global batch size 128 rollout batch size 512

micro batch size per device for update 4 Inference

temperature 0.6 top p 0.9

max tokens 10000

Table 2: Hyper-parameters

Here, we present the specific hyper-parameters that govern Mol-R1’s training and inference processes, as summarized in

- Table 2. All computational experiments were performed on Nvidia A800 GPUs. For Supervised Fine-tuning (SFT), we employed two A800 GPUs. The training configuration of SFT included a

per device train batch size of 1 with gradient accumulation steps set to 4. We used a learning rate of 1.0e-5, a cosine learning rate scheduler with a warmup ratio of 0.1, and trained for 5 epochs.

During Reinforced Policy Optimization (RPO), eight A800 GPUs were utilized. Key parameters included a learning rate of 1.0e-6, weight decay of 1.0e-2, and a kl coef of 1.0e-2. The number for sampling n was set to 5, rollout.temperature to 1.0, global batch size to 128, rollout batch size to 512, and micro batch size per device for update to 4.

Finally, for Inference, the decoding strategy used a temperature of 0.6, top p of 0.9, and a max tokens limit of 10000 to control generation diversity and length.

Data Statistics

This section details the statistics of the datasets used for cold-start supervised fine-tuning (i.e., MoIA T=0 SFT), encompassing our raw training data, cold-start data with various configurations, and the reasoning training sets at different MoIA iterations.

- Table 3 presents the number of entries and average length for each dataset.

Raw Training Dataset We adopt ChEBI-20 as our foundational raw training dataset, comprising 26,407 entries with an average length of 58.73. This dataset contains molecule-caption pairs without reasoning traces.

Mol-R1 Cold-start Data with Various Configurations These datasets are generated with different cold-start configurations utilized in the initial SFT phase of Mol-R1.

- • RS-G: Generated by Gemma3-12B and Gemma3-27B via Rejection Sampling, containing 2,943 entries with an average length of 258.03.

Dataset # Entries # Average Length Raw Training Dataset

ChEBI-20 26,407 58.73 Mol-R1 Cold-start Configurations

RS-G 2,943 258.03

PRID-G 1,053 870.67 PRID-4o (R0) 1,053 893.88 Mol-R1 Reasoning Training Set RT

- R1 7,285 763.24

- R2 8,700 794.43

Table 3: Statistics of the supervised fine-tuning data.

- • PRID-G: Features 1,053 entries, but with a significantly longer average length of 870.67, indicating these entries incorporate more extended reasoning sequences. The ”G” here denotes that the traces were generated using the Gemma3-12B model via PRID.
- • PRID-4o: Also known as R0. It shares the same number of entries as PRID-G (1,053 entries) but has a slightly longer average length of 893.88. The ”4o” signifies the use of the GPT-4o model for reasoning trace generation. A comparison between PRID-G and PRID-4o reveals minor differences in the level of detail within the reasoning traces generated by PRID with different models.

Mol-R1 Reasoning Training Sets These datasets were iteratively updated throughout the MoIA via Rejection Sampling, aimed at progressively refining the model’s reasoning capabilities.

- • R1: The iteration T=1 of the reasoning training set includes 7,285 entries with an average length of 763.24.
- • R2: The iteration T=2 expands to 8,700 entries, with an average length of 794.43. This growth in both scale and average length across iterations reflects the increasing complexity and richness of the accumulated reasoning traces via MoIA.

#### Group Relative Policy Optmization

Group Relative Policy Optimization (GRPO) is a reinforcement learning algorithm that aims to enhance the reasoning capabilities of LLMs and decision quality by revisiting and re-evaluating past reasoning steps (Guo et al. 2025). The core idea behind GRPO is to optimize the policy model using within-group relative rewards, rather than relying on a traditional critic model (Schulman et al. 2017). Specifically, GRPO samples a set of actions at each state (i.e., responses of LLMs with the current model parameters) and then updates the policy model based on the reward differences among these actions, which avoids the estimation of value functions, simplifying the algorithm while maintaining strong performance.

Mathematically, given a molecule description x, GRPO first samples a group of G predicted molecules {m1,m2,...,mG} from the old policy model πθ

and optimize the policy model πθ via the following loss function:

old

G

πθ(mk|x) πθ

1 G

L(θ) =

{min[(

)Ak,

(mk|x)

old

k=1

πθ(mk|x) πθ

,1 − ϵ,1 + ϵ)Ak] − βDKL(πθ||πref)}, (9)

clip(

(mk|x)

old

πref(mk|x) πθ(mk|x) − log

πref(mk|x) πθ(mk|x) − 1, (10)

DKL(πθ||πref) =

where DKL denotes the KL divergence between the policy model and the reference model, which enhances the policy stability and convergence, while β and ϵ are hyper-parameters, and Ak is the group advantage that calculated by a group of rewards {r1,r2,...rG}:

rk − mean({r1,r2,...,rG}) std({r1,r2,...,rG})

(11) The design of the reward function r is also critical in GRPO, as it directly influences the quality and performance of the

Ak =

learned policy. In this work, we mainly adopt the Exact Match (EM) rewards for our policy training. The exact match score

assesses whether the predicted molecule is identical to the ground truth. Specifically, we convert the generated SMILES string into standard form using RDKit5, and then compare it with the ground truth.

four different reward functions:

- • Exact Match Rewards: The exact match score assesses whether the predicted molecule is identical to the ground truth. Specifically, we convert the generated SMILES string into the standard form via RDKit, and then compare it with the ground truth.
- • Format Rewards: The format reward requires LLMs to generate the reasoning process and the final answer following the pre-defined pattern. The reasoning process should be placed between the tags <think> and </think>, and the final answer should be placed between <answer> and </answer>.
- • Length Rewards: The length rewards encourage LLMs to engage in deeper reasoning, exploring more potential paths and thinking more carefully. However, if the reasoning trace is too long, it could harm the output efficiency (Chiang and Lee 2024). In this work, we set a manual threshold for these rewards to control the “thinking budget”—beyond a certain token length, the length reward no longer increases.
- • Similarity Rewards: Unlike exact match rewards, similarity rewards are designed to be smoother. While accuracy rewards strictly require the generation of exactly matched molecules, similarity rewards can provide positive feedback for predictions that are merely similar.

In practice, these reward functions are typically combined rather than used independently. We explore the combinations of the reward functions through detailed experiments to optimize the reasoning performance of Mol-R1 in text-based molecule reasoning generation.

### Information-Theoretic Proof of Explicit Reasoning Effectiveness

- 1. Notations and Definitions Let’s define the random variables and information-theoretic concepts used:

- • Q: Random variable representing the problem.
- • A: Random variable representing the true answer. We assume a true conditional probability distribution P(A|Q).
- • R: Random variable representing the reasoning path.
- • Aout: Random variable representing the model’s output answer. We utilize core information-theoretic definitions:
- • Entropy H(X): Measures the uncertainty of a random variable X. H(X) = − x P(x)log P(x).
- • Conditional Entropy H(X|Y ): Measures the uncertainty of X given Y . H(X|Y ) = − x,y P(x,y)log P(x|y).
- • Mutual Information I(X;Y ): Quantifies the amount of information shared between X and Y . I(X;Y ) = H(X) − H(X|Y ) = H(Y ) − H(Y |X) = H(X) + H(Y ) − H(X,Y ).

- 2. Model Architectures

- • Direct Output Model (MD): Directly maps problem Q to output answer Aout. This can be viewed as a conditional probability distribution P(Aout|Q).
- • Explicit Reasoning Model (ME): Comprises two stages:

- 1. Reasoning Path Generator (Ppath): Generates a reasoning path R from Q. Represented by P(R|Q).
- 2. Result Generator (Gres): Generates the final answer Aout from the reasoning path R. Represented by P(Aout|R). The overall process for ME is given by P(Aout|Q) = R P(Aout|R)P(R|Q).

#### 4. Proof Derivation

- 4.1. Ideal Explicit Reasoning Path: Entropy Reduction and Information Gain Premise: Assume an ideal and true reasoning path (Rtrue) exists. This path perfectly captures all relevant information about the answer A from the problem Q without introducing noise. In this ideal scenario, Q, Rtrue, and A form a Markov chain: Q → Rtrue → A. This means that given Rtrue, all information about A is contained within Rtrue, and Q provides no additional information gain regarding A. Formally, P(A|Q,Rtrue) = P(A|Rtrue). Theorem (Data Processing Inequality): For a Markov chain X → Y → Z, we have I(X;Z) ≤ I(Y ;Z). Proof:

5https://www.rdkit.org/

- 1. Information Transfer: Applying the Data Processing Inequality to our Markov chain Q → Rtrue → A: I(Q;A) ≤ I(Rtrue;A)

This means the amount of information transferred to the answer A via the perfect reasoning path Rtrue is not less than the information transferred directly from the problem Q. In an ideal scenario, I(Q;A) = I(Rtrue;A), as Rtrue acts as a sufficient statistic for A with respect to Q.

- 2. Conditional Entropy Reduction (Entropy Decrease): Using the definition of mutual information I(X;Y ) = H(Y ) − H(Y |X), we can write:

I(Rtrue;A) = H(A) − H(A|Rtrue) I(Q;A) = H(A) − H(A|Q) Substituting these into the inequality I(Q;A) ≤ I(Rtrue;A):

H(A) − H(A|Q) ≤ H(A) − H(A|Rtrue) =⇒ −H(A|Q) ≤ −H(A|Rtrue) =⇒ H(A|Rtrue) ≤ H(A|Q)

Interpretation: This inequality demonstrates the ”entropy decrease.” Given an ideal and true reasoning path Rtrue, the uncertainty about the answer H(A|Rtrue) is not higher than the uncertainty given only the problem H(A|Q). This signifies that each correct step in the reasoning process provides valuable information, reducing the overall uncertainty about the final answer.

#### 4.2. Imperfect Explicit Reasoning Path: Condition for Net Gain from Imperfect Reasoning

Premise: Here, we aim to derive the formal condition under which an imperfect, noisy reasoning path Rgen is still more beneficial for determining the true answer A than relying solely on the problem Q. In other words, we seek the boundary

condition for which the explicit reasoning model ME outperforms the direct model MD. This translates to finding when the uncertainty about the answer, given the generated reasoning, is less than the uncertainty given only the problem:

H(A|Rgen) < H(A|Q)

This inequality signifies a net reduction in entropy, meaning the reasoning process provides an overall informational gain despite its imperfections. Proof: Our goal is to transform the inequality H(A|Rgen) < H(A|Q) into a more intuitive condition based on mutual information.

- 1. Expressing Entropy via Mutual Information: We use the fundamental relationship H(X|Y ) = H(X) − I(X;Y ). Applying this to our target inequality, we get:

H(A) − I(A;Rgen) < H(A) − I(A;Q) This simplifies to the core requirement for net gain:

I(A;Rgen) > I(A;Q)

This means the mutual information between the generated reasoning and the true answer must be greater than the mutual information between the problem and the true answer.

- 2. Decomposing the Information in the Reasoning Path: To understand how I(A;Rgen) relates to I(A;Q), we use the chain rule for mutual information:

I(A;Q,Rgen) = I(A;Q) + I(A;Rgen|Q)

This states that the total information about A contained in the pair (Q,Rgen) is the information from Q plus the additional information provided by Rgen when Q is already known.

- 3. Applying the Markov Chain Assumption: As established in the problem description, the structure of the explicit reasoning

model implies a Markov chain Q → Rgen → A. This assumption is key, as it posits that once the reasoning path Rgen is generated (which typically includes a restatement of Q), the original problem Q provides no further information about the answer A.

The formal definition of this Markov chain is P(A|Q,Rgen) = P(A|Rgen), which is equivalent to the conditional mutual information being zero:

I(A;Q|Rgen) = 0

- 4. Connecting the Pieces: We can also express the joint information I(A;Q,Rgen) using the chain rule in a different order: I(A;Q,Rgen) = I(A;Rgen) + I(A;Q|Rgen)

Since we established from the Markov assumption that I(A;Q|Rgen) = 0, this simplifies to:

I(A;Q,Rgen) = I(A;Rgen) By equating this with our first chain rule expression from step 2, we get:

I(A;Rgen) = I(A;Q) + I(A;Rgen|Q) This identity is central: it beautifully decomposes the information content of the generated reasoning path. It states that the total information in Rgen about A is the sum of the information that was already available in the question (I(A;Q)) and the new, value-added information generated by the reasoning process itself (I(A;Rgen|Q)).

- 5. Establishing the Final Condition: Now, we substitute this decomposition back into our inequality from step 1:

I(A;Q) + I(A;Rgen|Q) > I(A;Q) Subtracting I(A;Q) from both sides gives us the final condition for net gain:

I(A;Rgen|Q) > 0 Conclusion: The proof establishes a clear and intuitive boundary for the effectiveness of explicit reasoning, even when the reasoning path Rgen is imperfect.

- • Condition for Benefit: An imperfect reasoning path Rgen provides a net gain over direct prediction from Q if and only if the reasoning process itself generates new information about the answer A that was not available from the problem Q alone. I(A;Rgen|Q) quantifies this “value-added” information.
- • The Role of Noise/Error: The errors and noise introduced by the reasoning generator Ppath directly degrade the quality of the reasoning path. This causes the generated path Rgen to be a less faithful representation of the ideal path Rtrue. Consequently, the amount of new information generated is reduced:

I(A;Rgen|Q) ≤ I(A;Rtrue|Q) However, as long as this degraded, noisy reasoning process can still uncover any amount of new, relevant information about the answer, the value of I(A;Rgen|Q) will be greater than zero, and the overall model will see a benefit (H(A|Rgen) < H(A|Q)).

- • The Boundary Case: The boundary where the explicit reasoning model provides no benefit (H(A|Rgen) = H(A|Q)) occurs precisely when I(A;Rgen|Q) = 0. This describes a scenario where the entire reasoning process, however lengthy or complex, fails to produce any new insight relevant to finding the answer beyond what was already contained in the question itself.

- 5. Conclusion Based on the detailed mathematical derivation, we conclude:

- 1. Ideal Explicit Reasoning Benefit: In an ideal scenario, if the reasoning path generator perfectly produces the true, infor-

mative reasoning path Rtrue, the process is entropy-decreasing, i.e., H(A|Rtrue) ≤ H(A|Q). A perfect reasoning path effectively reduces the uncertainty about the final answer, theoretically enhancing reliability.

- 2. Imperfect Reasoning Net Gain: In practice, a generated reasoning path (Rgen) is imperfect and contains noise. However, this flawed path still provides a net benefit as long as the reasoning process itself uncovers new, value-added information about the answer (A) that was not available from the problem (Q) alone. The formal condition for this benefit is I(A;Rgen|Q) > 0. This means even a noisy reasoning process is effective if it successfully reduces the overall uncertainty (H(A|Rgen) < H(A|Q)), outperforming a direct prediction. The benefit disappears only when the reasoning provides no new information (I(A;Rgen|Q) = 0).

### Extensive Experiments & Analysis

#### Impact of Reasoning with Data Quantity on Cold-start Performance

Figure 6 investigates the impact of varying data quantities on model performance in cold-start scenarios, specifically comparing models with and without reasoning capabilities.

Low Data Scenario In the initial cold-start SFT step, where data is extremely limited, models incorporating reasoning capabilities demonstrate a decisive advantage. Both in terms of Levenshtein Distance (lower values indicate better performance) and BLEU Score (higher values indicate better quality), reasoning models significantly outperform their counterparts without reasoning. This suggests that in an information-poor environment, incorporating reasoning traces enables models to extract deeper patterns and relationships from limited data, leading to more accurate and expected outputs.

Figure 6: Impact of reasoning on the cold-start supervised fine-tuning performance of LLMs in Text-based Molecule Generation with different data quantities. Liechtenstein Distance (Left); BLEU Score (Right).

Figure 7: The performance change across different MoIA steps and iterations.

Performance with Increasing Data Quantity As the available data quantity gradually increases, both model types exhibit positive performance improvements. Initially, models without reasoning traces rapidly improve, and with sufficient data, they can even surpass reasoning models in certain metrics. This phenomenon can be attributed to the nature of reasoning traces distilled from LLMs. As previously discussed, these traces represent an imperfect explicit reasoning path. When the inherent noise within these distilled traces outweighs the information gains they provide, their presence can actually hinder performance compared to directly generating final outputs.

#### Model Performance across MoIA Steps and Iterations

Table 4 and Figure 7 present a comprehensive comparison of model performance across different MoIA iterations (T=0, T=1, T=2) and middle steps in MoIA: Supervised Fine-Tuning (SFT) and Reasoning Policy Optimization (RPO).

Overall, the results consistently highlight the effectiveness of the RPO within each MoIA iteration. RPO generally enables models to outperform their SFT counterparts at each MoIA iteration, indicated by higher exact match (EM) and Molecule FTS scores.

Meanwhile, we observe a consistent increase in exact match (EM) scores during the three iterations of MoIA training, as shown in Figure 7. The initial EM is quite low at T=0’s SFT step, only 0.038. However, by implementing RPO at T=0, the EM score significantly jumps to 0.144, demonstrating the immediate positive impact of policy optimization. As the training progresses to T=1, both SFT and RPO further improve, with RPO again showing a stronger result of 0.216 compared to SFT’s 0.166. This upward trend continues into T=2, where the EM scores reach their highest points: SFT achieves 0.217, and RPO

Method BLEU↑ EM↑ Levenshtein↓ MACCS FTS↑ RDK FTS↑ Morgan FTS↑ Validity↑

MoIA T=0

- T=0 (SFT) 0.636 0.038 39.67 0.761 0.549 0.475 0.750

- T=0 (RPO) 0.650 0.144 34.33 0.804 0.639 0.558 0.819 MoIA T=1

- T=1 (SFT) 0.621 0.166 34.22 0.802 0.640 0.561 0.848

T=1 (RPO) 0.655 0.216 32.87 0.821 0.673 0.596 0.863

MoIA T=2

T=2 (SFT) 0.618 0.217 34.06 0.804 0.658 0.583 0.879

- T=2 (RPO) 0.641 0.234 32.94 0.823 0.684 0.612 0.847

Table 4: Comparative performance of models at different MoIA iterations and middle steps (SFT, RPO). The best and secondbest results are highlighted in red and blue , respectively.

culminates with the best overall EM score of 0.234. This steady rise in EM across iterations underscores MoIA’s effectiveness in refining the model’s ability to generate precisely correct outputs, with RPO consistently leading in this crucial metric.

However, we also observe an interesting trend: the BLEU score exhibits a fluctuating pattern, showing no significant monotonic improvement across the MoIA iterations. This behavior, particularly as the EM score converges, seems to indicate an upper bound on the model’s capability. This suggests a performance ceiling reachable through self-learned policies under the constraints of the limited and potentially imperfect reasoning traces used for annotation. Ultimately, we hypothesize that this ceiling is determined by an interplay of the model’s inherent capacity and the quality of the initial cold-start reasoning data.

Figure 8: The performance change with different weights of similarity and exact match in the reward model.

#### Impact of Reward Modelling

- Figure 8 illustrates the model’s performance across various reward function combinations. Here, we specifically examine the interplay between Exact Match and Validity metrics. Notably, the weights for the similarity and Exact Match rewards sum to 0.8; the remaining 0.2 is allocated to the length reward.

We observe a clear inverse relationship between Exact Match and Validity as the weighting shifts. As the Exact Match weight increases (moving from left to right on the x-axis), the Exact Match score also consistently rises. This indicates that placing more emphasis on exact matches in the reward function effectively drives the model to produce more precise outputs.

Conversely, as the Exact Match weight increases, the Validity score consistently declines. This suggests a trade-off: optimizing too aggressively for exact matches leads to a degradation in the structural integrity or chemical correctness of the generated outputs. The initial configurations, where the Similarity weight is higher, maintain very high Validity, highlighting its importance in preserving output correctness.

In essence, the trend reveals that while boosting Exact Match score is achievable by adjusting the reward function, it often comes at the cost of reduced Validity. However, in this work, we prioritize the Exact Match score more heavily because achiev-

ing a precise and chemically correct match to the target is paramount for the specific downstream applications this model is designed for.

### Examples of Reasoning Traces Generated by GPT-4o via PRID

Here, we demonstrate two examples of the reasoning traces that are distilled from GPT-4o via PRID, as demonstrated in Figures 9 and 10. The reasoning traces show clearly how to infer detailed molecular structures based on the textual representations. Basically, the reasoning traces identify keywords in the natural language description, use them to form a molecular structure hypothesis, and then refine the hypothesis by considering additional details in the description, such as functional groups, their positions, and the overall molecular class.

Caption: The molecule is a 2-hydroxydicarboxylic acid that is succinic acid in which one of the hydrogens attached to a carbon is replaced by a hydroxy group. It has a role as a food acidity regulator and a fundamental metabolite. It is a 2-hydroxydicarboxylic acid and a C4-dicarboxylic acid. It derives from a succinic acid. It is a conjugate acid of a malate(2-) and a malate.

Molecule: C(C(C(=O)O)O)C(=O)O

Reasoning Process from Caption to Molecule: <think> 😊Well, I now need to generate a SMILES representation for a molecule based on the description provided by the user. First, I have to carefully analyze each sentence to understand the structural features they correspond to.

First sentence: “The molecule is a 2-hydroxydicarboxylic acid that is succinic acid in which one of the hydrogens attached to a carbon is replaced by a hydroxy group.” There are a few keywords here: 2-hydroxydicarboxylic acid, succinic acid, and hydroxy group. 🤔Let me think about what the SMILES for succinic acid looks like. 🤔Succinic acid is a four-carbon dicarboxylic acid, and its SMILES representation is C(CC(=O)O)C(=O)O. According to the description, we need to replace one of the hydrogen atoms on a carbon with a hydroxy group. Based on the “2-hydroxy” description, the hydroxy group should be on the second carbon. Therefore, the SMILES representation for this molecule should be C(C(C(=O)O)O)C(=O)O. Second sentence: “It has a role as a food acidity regulator and a fundamental metabolite.” This part mainly describes the function and properties of the molecule. 🤔Let me think, food acidity regulators and fundamental metabolites, these functions are closely related to the molecule’s structure. Food acidity regulators are usually organic acids, while fundamental metabolites are often important compounds involved in cellular metabolic processes. This molecule is a dicarboxylic acid with an added hydroxy group, which fits these properties. Third sentence: “It is a 2-hydroxydicarboxylic acid and a C4-dicarboxylic acid.” This confirms the structural features of the molecule again: a 2hydroxydicarboxylic acid and a C4-dicarboxylic acid. C4-dicarboxylic acid means the molecule has four carbon atoms and two carboxyl groups. Fourth sentence: “It derives from a succinic acid.” This mentions that the molecule is derived from succinic acid, confirming our previous inference again. Fifth sentence: “It is a conjugate acid of a malate(2-) and a malate.” This mentions that the molecule is the conjugate acid of malate. Malate is a C4-dicarboxylic acid, a dicarboxylic acid plus a hydroxy group, which matches our previous deduction. Now, based on my current line of thought, I need to write out the SMILES representation for the molecule. The SMILES representation for this molecule should be: C(C(C(=O)O)O)C(=O)O.

ThisNext,SMILESI need torepresentationoutput this SMILESmeets ouraccordingexpectations;to the formatit is derivedspecifiedfrombysuccinicthe user.acid and also fits the characteristics of a 2-hydroxydicarboxylic acid. 🎉 </think>

- Figure 9: The reasoning process generated by GPT-4o via PRID. Here, this example demonstrates how to obtain the target molecule C(C(C(=O)O)O)C(=O)O based on the content of the molecule caption.

### Prompt Design

In this section, we demonstrate the prompts we apply in this work. Figure 11 illustrates the details of the expert-annotated context example for PRID. This example provides a clear and detailed reasoning trace that guides the LLM through the process of generating a molecular structure from a textual description. The reasoning trace serves as a form of “scaffolding” that regulates the LLM’s exploration behavior by breaking down a complex problem into a series of manageable steps.

#### Expert-annotated Context Example for PRID

The process unfolds as follows:

- • Decomposition of the Problem: The LLM is first prompted to analyze the user’s caption sentence by sentence, which encourages it to identify keywords and extract specific structural information from each piece of text. For instance, it identifies “cyclodiene,” “organochlorine,” and “indene derivative.” This step prevents the LLM from attempting to solve the entire problem at once and instead focuses its attention on individual, verifiable facts.
- • Iterative Hypothesis Formulation and Refinement: The LLM is guided to build a hypothesis incrementally. It starts with the initial, vague keyword “cyclodiene” and considers several possibilities (cyclopentadiene, cyclohexadiene). When this is not enough, it is instructed to look for clues in subsequent sentences. The mention of “indene derivative” is a crucial turning point, allowing the LLM to narrow down its initial hypothesis to a structure derived from indene. This demonstrates how the reasoning trace directs the LLM to seek out and integrate new information to refine its understanding, moving from a general concept to a more specific structural scaffold.

Caption: The molecule is the C-nitro compound that is chlorobenzene with three nitro substituents in the 2-, 4- and 6-positions. It has a role as an epitope, an explosive, a hapten and an allergen. It is a C-nitro compound and a member of monochlorobenzenes.

Molecule: C1=C(C=C(C(=C1[N+](=O)[O-])Cl)[N+](=O)[O-])[N+](=O)[O-]

Reasoning Process from Caption to Molecule: <think> 😊Well, I now need to generate a SMILES representation for a molecule based on the description provided by the user. First, I have to carefully analyze each

sentence to understand the structural features they correspond to. First sentence: "The molecule is the C-nitro compound that is chlorobenzene with three nitro substituents in the 2-, 4- and 6-positions." There are several keywords here: C-nitro compound, chlorobenzene, three nitro substituents in the 2-, 4- and 6-positions. 🤔Let me think about what the SMILES for chlorobenzene is. The SMILES for chlorobenzene is: ClC1=CC=CC=C1. This is a benzene ring where one hydrogen atom is replaced by a chlorine atom. Next, according to the description, this chlorobenzene has three nitro substituents at positions 2, 4, and 6. We need to add nitro groups (-NO2) at these appropriate positions. 🤔The SMILES for a nitro group is [N+](=O)[O-]. So we need to add nitro groups to chlorobenzene at positions 2, 4, and 6 respectively.

Second sentence: "It has a role as an epitope, an explosive, a hapten and an allergen." This part mainly describes the molecule's functions and properties. 🤔Let me think about this: an epitope is a specific region on an antigen molecule that can be recognized by antibodies. An explosive indicates this molecule has high energy and instability. A hapten is a small molecule that cannot elicit an immune response alone but can do so when conjugated with a carrier protein. An allergen is a substance capable of causing allergic reactions. These functions and properties suggest the molecule has significant chemical reactivity and instability, likely related to its multiple nitro groups (-NO2), as nitro is a strong electron-withdrawing group that increases molecular reactivity and instability. Third sentence: "It is a C-nitro compound and a member of monochlorobenzenes." Here it's mentioned that the molecule is a C-nitro compound and belongs to the monochlorobenzene family. Being a C-nitro compound means it contains nitro groups (-NO2), and being a monochlorobenzene member indicates it contains one chlorine atom. Based on these descriptions, we can determine the molecule's basic structure is chlorobenzene (ClC1=CC=CC=C1) with nitro groups (-NO2) at positions 2, 4, and 6. We can generate the final SMILES representation by adding nitro groups at these appropriate positions. Now, we need to write the SMILES representation based on our current reasoning. First, we write the chlorobenzene structure: ClC1=CC=CC=C1. Then, we add nitro groups (-NO2) at positions 2, 4, and 6 respectively to obtain the final SMILES: C1=C(C=C(C(=C1[N+](=O)[O-])Cl)[N+](=O)[O-])[N+](=O)[O-]. This SMILES representation meets our expectations; it's a chlorobenzene with nitro groups (-NO2) at positions 2, 4, and 6, and also aligns with the

characteristicsNext, we need toofoutputa C-nitrothiscompoundSMILES andaccordingmonochlorobenzene.to the format 🎉specified by the user. </think>

- Figure 10: The reasoning process generated by GPT-4o via PRID. Here, this example demonstrates how to obtain the target molecule C1=C(C=C(C(=C1[N+](=O)[O-])Cl)[N+](=O)[O-])[N+](=O)[O-] based on the content of the molecule caption.

- • Connecting Function to Structure: The context example prompts the LLM to consider functional descriptions like “GABA-gated chloride channel antagonist” and “persistent organic pollutant.” While these properties don’t directly define the structure, they provide contextual clues that help the LLM make informed decisions. For example, recognizing that the molecule is a persistent organochlorine insecticide with multiple chlorine atoms strengthens the hypothesis of a complex, chlorinated hydrocarbon structure, moving it beyond simple chlorinated indene. This step guides the LLM to validate its structural assumptions against the molecule’s known properties.
- • Structured Generation and Verification: Finally, the LLM is guided to synthesize all the information to generate the final SMILES representation. It is instructed to first think about the core structure (the indene derivative), then consider the modifications (chlorination and the elimination of double bonds), and finally, to verify that the generated SMILES aligns with all parts of the initial description. This structured approach ensures that the final output is not a guess but a well-reasoned conclusion derived from a systematic process of information synthesis and validation.

In essence, the expert-annotated context example in PRID serves as a detailed blueprint for problem-solving. It demonstrates a logical and verifiable path from textual description to molecular structure, thereby providing a clear and effective form of guidance that regulates the LLM’s reasoning process and improves the accuracy of its generated outputs.

#### Prompt for Evaluating Consistent-F1

Finally, Figure 12 shows the prompt for evaluating the Consistent-F1 score, which aims to reveal the reasoning traces quality for explicit Long-CoT reasoning models.

Caption: The molecule is a cyclodiene organochlorine insecticide. It has a role as a GABA-gated chloride channel antagonist and a persistent organic pollutant. It derives from a hydride of an indene.

Molecule: C1C2C(C(C1Cl)Cl)C3(C(=C(C2(C3(Cl)Cl)Cl)Cl)Cl)Cl

Reasoning Process from Caption to Molecule: <think> 😊Okay, I'm now tasked with generating the SMILES representation of a molecule based on the user's description. First, I need to carefully analyze each sentence and understand the corresponding structural features. The first sentence: "The molecule is a cyclodiene organochlorine insecticide." There are several keywords here: cyclodiene, organochlorine, and insecticide. 🤔Let me think, what does a cyclodiene SMILES look like? 🤔There seem to be several types of cyclodienes, differentiated by the number of carbon atoms, such as cyclopentadiene (C1=CC=CC1) and cyclohexadiene (C1=CCC=CC1, C1C=CC=CC1). However, the description only mentions cyclodiene, and I'm currently unsure which one it is. Well, I can look for information elsewhere. According to the second keyword, this molecule is an organochlorine compound, right? So, there must be chlorine atoms in the molecule's SMILES! Well, a common organochlorine compound is chloroform. Its SMILES is ClC(Cl)Cl. There are several chlorine atoms, which suggests that this molecule might also have several chlorine atoms. But this is still not enough to deduce the entire structure of the molecule. Well, what does the third keyword tell me? Oh, this molecule is an insecticide. Let me think, an insecticide containing chlorine, right? Oh, dichlorodiphenyltrichloroethane (DDT) is a common chlorine-containing insecticide, and its SMILES is ClC(Cl)(Cl)C1=CC=C(Cl)C(Cl)=C1. 🤔It looks very complicated, and I still don't have any clues. Maybe I should look at the second sentence. The second sentence: "It has a role as a GABA-gated chloride channel antagonist and a persistent organic pollutant." This part mainly describes the function and properties of the molecule. 🤔Let me think,, first of all, a GABA-gated chloride channel antagonist, indicating that this is a type of drug that affects the central nervous system. GABA is the main inhibitory neurotransmitter in the central nervous system, and the GABA-gated chloride channel is part of the GABA receptor. When GABA binds to the receptor, the chloride channel opens, chloride ions flow into the neuron, causing hyperpolarization of the neuron, thereby inhibiting the excitability of the neuron. So, such a molecule needs to have certain properties, which may be reflected in the structural features of the molecule. For example: having specific binding sites that can bind to specific regions of the GABA receptor, thereby preventing the action of GABA. 🤔This property is a bit difficult to analyze. I only know that halogens may affect its pharmacological activity, but I'm not sure which points on the GABA-specific region can bind to this molecule. However, based on other properties, this molecule also needs to have a certain lipophilicity and electroneutrality, indicating that this molecule is usually hydrophobic and needs to exclude some hydrophilic groups. In addition, this molecule should also have no ions. Furthermore, this molecule is a persistent organic pollutant. Wait, this is very similar to DDT, right! 🤔But organochlorine compounds seem to have this characteristic. The third sentence: "It derives from a hydride of an indene." Here, indene is mentioned, and the molecule I need to generate is its derivative. Indene is a bicyclic structure consisting of a benzene ring and a cyclopentene. As a hydride, it means that a proton has been removed from indene, or perhaps a certain group has been added at a certain position. 🤔So I need to determine what the SMILES of indene is. Well, as far as I know, this is an aromatic hydrocarbon with the molecular formula C6H4C3H4, which can be regarded as a benzene ring fused with a cyclopentadiene, and the SMILES can be written as C1C=Cc2ccccc21, or C1C=CC2=CC=CC=C21. 🤔I understand, the cyclodiene in the first sentence is actually cyclopentadiene! 🤔But I need to consider whether this molecule is a simple chlorinated indene. If so, the SMILES of this molecule would be very simple. However, the description in the first sentence emphasizes cyclodiene, and the second sentence emphasizes its antagonist role, so the molecule may not be so simple. However, in any case, the final molecular structure should be derivable from indene. Wait, I remember that there is a cyclopentadiene insecticide that usually has a structure, which is formed by two cyclopentadiene rings sharing three carbon atoms. In this way, if the central carbon atom is removed, the remaining structure will be a six-membered ring! As long as an additional cyclopentadiene is added next to it, it can be ensured that the molecule is derived from indene! Now, I need to write the SMILES representation of the molecule based on the current ideas. First, I need to write the structure before chlorination, then eliminate some double bonds and add chlorine atoms. Finally, I need to check whether the molecule meets our expectations. 🤔The structure before chlorination consists of three cyclopentadiene rings, two of which share three carbon atoms due to the three carbon atoms, making six carbon atoms aromatic, and a single carbon atom detached from the benzene plane. The SMILES representation of this structure can be written as C1C=CC2=C3C=CC(C3)=C21, right! Next, I will eliminate double bonds and add chlorine atoms. However, depending on the number of chlorine atoms, I can get multiple molecules that meet the description. Here, I need to combine my speculation about the other properties of this molecule, such as my understanding of chlorinated insecticides, to determine the final SMILES representation. Based on the structure of common chlorinated insecticides, I find that these compounds have multiple chlorine atoms, and their chemical properties require the presence of double bonds. Therefore, I believe that the final SMILES representation of this molecule is: C1C(Cl)C(Cl)C2C3(Cl)C(Cl)=C(Cl)C(Cl)(C3(Cl)Cl)C21, well 🤔, let me summarize, this molecule can also be written as: C1C2C(C(C1Cl)Cl)C3(C(=C(C2(C3(Cl)Cl)Cl)Cl)Cl)Cl.

ThisNext,SMILESI need torepresentationoutput this SMILESmeets ouraccordingexpectations,to the formatit is derivedspecifiedfrombyindene,the user.and also meets the characteristics of organochlorine insecticides. 🎉 </think> <answer> C1C2C(C(C1Cl)Cl)C3(C(=C(C2(C3(Cl)Cl)Cl)Cl)Cl)Cl </answer>

###### Figure 11: The expert-annotated context example for Prior Regulation via In-context Distillation.

###### You are an excellent chemist and an AI assistant for evaluating reasoning traces.

Here is a text-based molecule generation problem, which aims to predict the SMILES representation of the molecule based on the molecular description.

Your assistant has provided a detailed reasoning trace based on the molecular description. Your task is to examine the reasoning trace and determine whether it could correctly predict the SMILES representation of the molecule. Your judgment should be based on the factual correctness and logic coherence of the reasoning trace.

Here is the original molecular description: {description} Here is the reasoning trace from your assistant: {reasoning_trace}

Your response should be 'yes' if the reasoning trace is consistent with the molecular description and could lead to a correct SMILES representation, or 'no' if it is not consistent.

Your response should be in the following format: { "response": "yes" or "no" }

###### YOUR OUTPUT SHOULD BE A VALID JSON OBJECT. YOU MUST NOT INCLUDE ANY ADDITIONAL TEXT OR COMMENTS.

Figure 12: The prompt for evaluating the quality of reasoning traces (i.e., Consistent-F1).

