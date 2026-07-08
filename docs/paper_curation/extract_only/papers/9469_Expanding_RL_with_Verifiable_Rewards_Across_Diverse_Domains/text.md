# arXiv:2503.23829v2[cs.CL]1Apr2025

## Crossing the Reward Bridge: Expanding RL with Verifiable Rewards Across Diverse Domains

#### Yi Su∗,1,2 , Dian Yu1 , Linfeng Song1 , Juntao Li2 , Haitao Mi1 ,

Zhaopeng Tu1 , Min Zhang2 , and Dong Yu1 1Tencent AI Lab 2Soochow University

reasoning data I (x, a)

teacher grader

STEP 1 base model RLVR

exploration data ((x, a, y), c)

STEP 2

base model SFT reward model

reasoning data II (x, a)

reward model

STEP 3

base model RLVR final policy

Figure 1: Overview paradigm of RLVR with our cross-domain verifier.

### Abstract

Reinforcement learning with verifiable rewards (RLVR) has demonstrated significant success in enhancing mathematical reasoning and coding performance of large language models (LLMs), especially when structured reference answers are accessible for verification. However, its extension to broader, less structured domains remains unexplored. In this work, we investigate the effectiveness and scalability of RLVR across diverse realworld domains including medicine, chemistry, psychology, economics, and education, where structured reference answers are typically unavailable. We reveal that binary verification judgments on broad-domain tasks exhibit high consistency across various LLMs provided expert-written reference answers exist. Motivated by this finding, we utilize a generative scoring technique that yields soft, model-based reward signals to overcome limitations posed by binary verifications, especially in free-form, unstructured answer scenarios. We further demonstrate the feasibility of training cross-domain generative reward models using relatively small (7B) LLMs without the need for extensive domain-specific annotation. Through comprehensive experiments, our RLVR framework establishes clear performance gains, significantly outperforming state-of-the-art open-source aligned models such as Qwen2.5-72B and DeepSeek-R1-Distill-Qwen-32B across domains in free-form settings. Our approach notably enhances the robustness, flexibility, and scalability of RLVR, representing a substantial step towards practical reinforcement learning applications in complex, noisy-label scenarios.

∗The work was done during Yi’s internship at Tencent AI Lab.

### 1 Instruction

Reinforcement learning with verifiable rewards (RLVR) has recently emerged as an effective paradigm for improving the reasoning capabilities of large language models (LLMs) (Luong et al., 2024; Lambert et al., 2024), even in scenarios without supervised fine-tuning (Guo et al., 2025). RLVR typically leverages reference-based signals, assuming the availability of objective ground-truth answers to determine whether model responses align with reference outcomes. In prior studies, RLVR has mainly demonstrated success on tasks with precisely structured solutions, such as mathematical reasoning or code generation, where binary verification signals (correct or incorrect) can be reliably computed with simple rule-based verifiers (Team et al., 2025; Gandhi et al., 2024; Zhang et al., 2024b). Nonetheless, the extension of RLVR to broader, more nuanced domains remains largely unexplored, due primarily to the challenges associated with verifying complex, frequently unstructured reference answers.

In this paper, we aim to extend the applicability of RLVR to domains beyond structured mathematics and coding, by investigating its performance in a diverse set of complex reasoning-intensive areas such as medicine, chemistry, psychology, economics, and education. Central to this exploration is the observation that binary correctness judgments, even on broad-domain tasks, tend to exhibit remarkable agreement across varied large language models (LLMs), including both closed-source models (e.g., GPT-4o) and recently released powerful open-source solutions (e.g., Qwen2.5-72B-Instruct) when provided high-quality objective references authored by domain experts. This finding indicates that reference-based evaluation of diverse domain answers is typically easier than reference-free verification, which is inherently as difficult as identifying the first mistake in a response (Lightman et al., 2023). Consequently, this insight undermines the presumed necessity for extensive domainspecific annotation and motivates rethinking traditional practices in reward-model training for multi-domain scenarios.

While binary rewards have been the prevalent standard across RLVR applications (Gandhi et al., 2024; Lambert et al., 2024; Guo et al., 2025; Ma et al., 2025a), they pose clear limitations—especially for unstructured tasks. Notably, our data analysis on real-world exam questions reveals that only

- 60.3% of mathematical problems possess single-term numerical answers verifiable by rule-based methods, with the ratio dropping further to 45.4% for complex multi-domain queries. This presents inherent challenges for binary reward schemes and demonstrates the need for richer and more granular verification mechanisms. To address these limitations, we propose incorporating soft scores obtained from generative, model-based verifiers directly into RLVR. Specifically, we compute a soft reward from the probability of a single indicative token produced by a generative verifier summarizing its assessment. Crucially, we demonstrate that it is feasible to distill effective multidomain generative verifier models based on relatively compact models (sizes as small as 7B) without conducting extensive domain-specific annotation. Instead, we employ data composed of response samples and their corresponding judgments collected during RL exploration under the supervision of a larger cross-domain generative teacher model. These noisy yet more realistic datasets promote robustness of the subsequently distilled model-based rewards.

Our empirical results strongly validate the effectiveness of our extended RLVR framework across various domains. By fine-tuning modest-sized (7B) base models using various RL algorithms and our soft reward verifier, we obtain improved reasoning policies superior to state-of-the-art open-source alignment models such as Qwen2.5-72B-Instruct and DeepSeek-R1-Distill-Qwen-32B, achieving performance boosts of up to 8.0% accuracy in diverse, free-form reasoning tasks. We particularly observe that our model-based soft rewards consistently scale better and produce more robust policies compared to conventional rule-based binary rewards, especially on unstructured answer scenarios and larger training data regimes.

Contributions. Our key contributions can be summarized as follows:

- • We extend reinforcement learning with verifiable rewards (RLVR) to diverse domains, establishing its effectiveness beyond traditional structured answer scenarios.

- • We introduce and validate a novel framework incorporating generative model-based soft rewards within RLVR, demonstrating substantial improvements in generalization, robustness, and scalability relative to traditional binary rule-based rewards.
- • We empirically demonstrate the feasibility and efficacy of training compact (7B-scale) crossdomain generative reward verifiers without extensive domain-specific annotation, challenging traditional assumptions about annotation scale.
- • We release a dataset containing 570k examples of multi-domain free-form data and the corresponding trained reward model, available at https://huggingface.co/collections/virtuoussy/ rlvr-67ea349b086e3511f86d1c1f, to facilitate future research in this promising direction.

### 2 Related Work

#### 2.1 Reward Estimation in Reinforcement Learning with Verifiable Rewards

For reasoning tasks such as mathematical reasoning, whether in constructing training data or at test time, a solution is typically considered correct if it arrives at correct final answer (Cobbe et al., 2021a). This is because reliably assessing the correctness of individual steps remains an open challenge, particularly when these steps may lack ground-truth labels in real-world scenarios. Similarly, the correctness of solutions to coding problems is typically accessed based on whether all test cases pass (Austin et al., 2021; Hendrycks et al., 2021a; Gehring et al., 2024). Consequently, previous reference-based RL studies have primarily focused on mathematical reasoning and coding tasks.

In most previous studies (Zelikman et al., 2022; Gandhi et al., 2024; Zhang et al., 2024b; Lambert et al., 2024; Guo et al., 2025; Ma et al., 2025a; Yu et al., 2025), given access to the reference answer a, the correctness label z for a response y to a prompt x is typically a binary value. z can also take on a value in the range [0,1] to reflect varying degrees of correctness (Luong et al., 2024; Li et al., 2024; Ma et al., 2025b; Xie et al., 2025). Labels are assigned by a deterministic function z = f(x, y, a), which operates based on predefined rules (e.g., exact match). These rules can also be combined with tools, such as a Python library, for verification (Xiong et al., 2025; Luo et al., 2025). This method is particularly effective when the answer type is fixed and easily matchable, such as a numerical value or a multiple-choice option. Each response is rated individually, without considering any preference information.

Besides using closed-source LLMs such as GPT-4o as verifiers (Chen et al., 2024), recent studies have also explored training reference-based reward models for mathematical reasoning (Team et al., 2025). However, these models are confined to a single domain and still require large-scale training data (e.g., 800k instances for math) even within that domain.

#### 2.2 Generative Reward Modeling

Using next-token prediction for reward modeling has attracted great interest in recent years (Lightman et al., 2023; Zheng et al., 2023; Tian et al., 2024; Zhang et al., 2024a), as it enables LLMs to fully leverage their generative capabilities, not only to produce accurate rewards but also to provide rationales that justify their judgments. In this work, we explore applying generative, reference-based verifiers to reinforcement learning and investigate their effectiveness across a variety of domains, an area that remains largely underexplored.

Furthermore, we explore training generative reward models without the need for annotated or synthetic step-by-step rationales (Team et al., 2025; Zhang et al., 2024a) to justify the final assessment. Specifically, we leverage the confidence of generative verifiers to provide stable and informative reward signals, enhancing the robustness of RL training in the presence of noise and ambiguity.

#### 2.3 Verifiable Reasoning Data

Previous and on-going RLVR studies primarily focus on narrow tasks (Liu & Zhang, 2025; Xie et al., 2025) such as math word problem solving, code generation, and logic puzzles, where well-structured reference answers allow for straightforward rule-based verification. For example, SimpleRL (Zeng et al., 2025) and Tulu (Lambert et al., 2024) use math datasets GSM8K (Cobbe et al., 2021b) and MATH (Hendrycks et al., 2021b), in which each reference answer typically consists of fewer than two words. However, this reliance on well-structured data constrains the scale and diversity of resources that can be used for RLVR across broader domains.

In this work, we explore RLVR using reasoning data spanning diverse domains, where reference answers are free-form, either written by domain experts for unbiased evaluation (Yu et al., 2021), extracted from pre-training corpora (Yue et al., 2024), or generated by LLMs (Yuan et al., 2025).

### 3 Method

We focus on a setting where each prompt x is accompanied by an expert-written reference answer a. Reference answers have been shown to play a crucial role in providing accurate rewards for reinforcement learning in reasoning-intensive tasks such as coding and mathematics (Shao et al.,

- 2024). Ideally, in these domains, a response y can be objectively verified against the given reference answer a. However, in practice, this verification process may be influenced by factors such as imperfect answer extraction and matching when pattern-based verifiers are used, as well as noise introduced by automated evaluation systems, such as a reward model rϕ(x, a, y).

Nevertheless, we can still use this verifiable reward in a policy gradient algorithm, with REINFORCE (Williams, 1992) as an example, as follows:

J(θ) = E(x,a)∼D Eyi∼πθ(·|x) rϕ(x, a, yi) . (1)

When the generation of an entire response is modeled as a single action (Ahmadian et al., 2024), the gradient becomes (see Section A.3 for details):

∇θJ(θ) = E(x,a)∼D Eyi∼πθ(·|x) rϕ(x, a, yi)∇θ log πθ(yi | x) . (2)

#### 3.1 Reward Estimation

To ensure a binary reward signal, we instruct a generative LLM πϕ to output only 0 or 1 (see system prompt in Table 4). For notational simplicity, we assume that each response consists of exactly T

steps, where each step corresponds to a non-empty line. Let yiT denote the final step of response yi. The binary model-based reward function is then defined as:

rϕ(x, a, yi) = ci = 1 , (3)

where ci is sampled from πϕ(· | x, a, yiT), representing πϕ’s judgment on the correctness of yi. Using πϕ as a verifier, we can also define a soft reward function using the probability of the judgment tokens (i.e., 0 or 1):

 

πϕ(1 | x, a, yiT) if ci = 1, 1 − πϕ(0 | x, a, yiT) if ci = 0, 0 otherwise.

(4)

rϕ(x, a, yi) =



As shown in Equations 3 and 4, rϕ(x, a, yi) is bounded within [0,1], ensuring consistency with the widely adopted binary reward scale.

- 3.2 Reward Normalization

To ensure stable gradients and encourage improvement across all samples in a batch that perform above average, we apply z-score normalization to rewards, inspired by prior studies such as GRPO (Shao et al., 2024) and REINFORCE++ (Hu, 2025).

r˜(x, a, yi) =

r(x, a, yi) − µr σr

, (5)

where µr and σr denote the mean and standard deviation of the rewards within the batch containing yi, respectively. In the special case where σr = 0, we set all normalized rewards to zero, as these samples are either too difficult or too easy for the current policy.

- 3.3 Reward Model Training

When considering generative verifiers, a natural choice is to use an off-the-shelf aligned LLM as the reward model πϕ, inspired by prior work that employs LLMs as judges (Zheng et al., 2023). However, we observe a noticeable performance gap on downstream tasks when using LLMs of different sizes. For example, the 72B reward model achieves 62.7% while the 7B model gets 58.8% on math data (see training details in Section 4). To address this, we explore training a moderately sized reward model (e.g., 7B) for general domains, aiming to balance performance and efficiency.

Since there are no ground-truth reward labels, for each (x, a, y) triple, we prompt a fixed LLM to obtain the binary judgments c ∈ {0, 1}, indicating whether y matches the reference answer a. During the RL phase, we collect the data {(x, a, y, c)} from the exploration stages and use it to fine-tune our reward models with supervised learning on c. Unlike relying on a fixed LLM to generate y, the improving actor policy produces responses with varying performance and potential formatting noise, which may enhance the robustness of the trained reward models.

- 4 Experiments

- 4.1 Data

Mathematics Data To ensure high-quality reference answers, we use a large-scale dataset of 773k Chinese Question Answering (QA) pairs, collected under authorized licenses from educational websites. This dataset covers three educational levels: elementary, middle, and high school. Unlike well-structured yet small-scale benchmarks such as MATH (Hendrycks et al., 2021b) and GSM8K (Cobbe et al., 2021b), our reference answers are inherently free-form, often interwoven with rationales or involving several sub-questions yet lacking clear structural patterns. As a result, rule-based reward functions that rely on clean, well-structured answers for verification struggle to process these unstructured reference answers effectively.

We use GPT-4o-mini to translate questions and their corresponding responses into English. We randomly sample 3,000 QA pairs from each level and reserve them for testing. The average length of reference answers in the test set is 33.7, 36.3, and 53.9 words for elementary, middle, and high school levels, respectively. These are much longer than those in the GSM8K (1 word) and MATH (1.3 words) test sets.

Multi-Subject Data Since no large-scale, free-form dataset with objective reference answers exists for general domains, we use a multi-subject multiple-choice QA dataset ExamQA (Yu et al., 2021). Originally written in Chinese, ExamQA covers at least 48 first-level subjects. We remove the distractors and convert each instance into a free-form QA pair. This dataset consists of 638k college-level instances, with both questions and objective answers written by domain experts for examination purposes. We also use GPT-4o-mini to translate questions and options into English.

For evaluation, we randomly sample 6,000 questions from ExamQA as the test set, while the remaining questions are used as the training pool. Since subject labels are not provided for each QA pair, we use GPT-4o-mini to classify them into one of 48 subjects or mark them as unclassified if uncertain. The detailed classification prompt is provided in Table 5. Excluding unclassified instances (15.8% of the test data), the most frequent subjects include basic medicine, law, economics, management, civil engineering, mathematics, computer science and technology, psychology, and chemistry, as shown in Figure 2. For ease of analysis, we further categorize these subjects into four broad fields (STEM, social sciences, humanities, and applied sciences) as detailed in Table 6. See examples in Table 10.

9.9 9.2

Basic Medicine

Law Economics

6.7 5.9

Management Civil Engineering Clinical Medicine

5.5 4.9

4.6 4.5

Mathematics

Computer Science and Technology Chinese Medicine and Chinese Materia Medica

4.0 3.8

Psychology Chemistry

3.4 3.3

Biology Pharmacy Education

2.7 2.6

2.1

Information Science and System Science History Political Science Environmental/Resource Science and Technology Mechanical Engineering

2.0 1.9

1.7 1.7

1.6 1.4

Safety Science and Technology Power and Electrical Engineering

1.2 1.1 1.1 1.1

Marxism

Statistics Materials Science

Literature Agronomy Physics

1.0

0.9 0.9

Earth Science Water Conservancy Engineering

0.9 0.8 0.8 0.8 0.8

Mechanics

Sociology Philosophy

Transportation Engineering Surveying and Mapping Science and Technology

0.7

0.6 0.6 0.5

Art Electronics and Communications Technology

Preventive Medicine and Public Health

0.5 0.4

Food Science and Technology Ethnology and Cultural Studies

0.3 0.3 0.3 0.3 0.3

Chemical Engineering Journalism and Communication

Sports Science Information and System Science Related Engineering and Technology

Animal Husbandry and Veterinary Science

0.2 0.2 0.1

Linguistics Library, Information, and Documentation

Religious Studies Natural Science Related Engineering and Technology

0.0

0 2 4 6 8 10 Percentage (%)

- Figure 2: Distribution of subject occurrences in the test set of ExamQA (excluding unclassified).

Data for Training the Reward Model We construct the data for training the reward model by extracting 20k samples from each training set of the two datasets, totaling 40k samples. Using the methodology in Section 3.3, we employ Qwen2.5-7B (Team, 2024) to conduct RL training. We use the RLOO (Kool et al., 2019; Ahmadian et al., 2024) algorithm and generate four online samples for each prompt. We use Qwen2.5-72B-Instruct as the reward model for hard label determination. By preserving all input-output pairs, this process yields 160k distilled training samples from Qwen2.572B-Instruct for reward model training.

To verify the training approach’s validity, we exclude these 40k original samples from the final training dataset. This strict separation ensures that the reward model never encounters any data used in previous training stages, thereby guaranteeing evaluation objectivity.

#### 4.2 Baselines and Notations

Base Directly use the base model to generate the response of the question. SFT Directly use the label (without CoT) to fine-tune the base model. Rule-based reward RL with the reward determined by predefined rules.

Qwen2.5-72B-Instruct RL with the reward determined by the judgment of Qwen2.5-72BInstruct (Team, 2024).

RM-7B (ours) RL with the reward determined by the judgment of the reward model trained on our 160k distilled data based on Qwen2.5-7B-Instruct (Team, 2024).

Binary When using rule-based rewards, we directly judge if the label is in the answer. When using model-based rewards, we use the output of the model. The value of binary reward should be in {0,1}.

Soft When using rule-based rewards, we use Jaccard similarity (Jaccard, 1912) as the reward. When using model-based rewards, we use the probability of the first output token. The value of soft reward should be in [0,1].

#### 4.3 Evaluation

We begin by investigating majority voting using a strong open-source LLM, Qwen2.5-72BInstruct (Team, 2024), as the reward model πϕ. The evaluation process follows the prompting template provided in Table 4. Given a prompt x and a reference answer a, we generate m evaluation samples and determine the correctness of a response y via majority voting. A response is considered

correct if at least half of the evaluations classify it as such, i.e., ∑mj=1 πϕ(j)(x, yT, a) = 1 ≥ m2 .

We measure the agreement between the Qwen-based evaluation method (majority voting over m samples) and GPT-4o (a single evaluation per response) using Cohen’s Kappa (κ). As shown in Figure 3, the two evaluation methods demonstrate almost perfect agreement (0.81 ≤ κ ≤ 1.00), with κ exceeding 0.86 for mathematics and 0.88 for multi-subject college-level problems. This high level of agreement remains consistent across varying values of m, indicating that the results are not highly sensitive to the number of evaluation samples. Based on this observation, we adopt m = 1 in all subsequent evaluations to improve efficiency without compromising evaluation quality.

#### 4.4 Implementation Details

After obtaining the 160k distilled data from Qwen2.5-72B-Instruct, we perform supervised finetuning on Qwen2.5-7B-Instruct using this data, resulting in our reward model. We use different RL algorithms to validate the effectiveness of our method, including REINFORCE (Williams, 1992; Ahmadian et al., 2024), RLOO (Kool et al., 2019; Ahmadian et al., 2024), and REINFORCE++ (Hu,

- 2025). Following Stiennon et al. (2020); Ouyang et al. (2022); Hu (2025), we introduce a KullbackLeibler (KL) divergence penalty between the RL model and the reference policy (i.e., base model) distributions to mitigate bias in the reward model. We update r˜(x, a, yi) as follows:

r˜(x, a, yi) ← r˜(x, a, yi) − β log

πθ(yi | x) πref(yi | x)

, (6)

where β ≥ 0 controls the effect of the KL penalty, and πref represents the reference policy distribution. We set β = 0.01 for all experiments.

For all algorithms, we apply reward normalization as introduced in Section 3.2. We use Qwen2.57B (Team, 2024) as the base model for our experiments. Despite not undergoing post-training, it demonstrates reasonable instruction-following capabilities, as shown by its zero-shot performance in Table 1. We also include the results of Qwen2.5-72B-Instruct and DeepSeek-R1-Distill-Qwen-32B to illustrate the difficulty level of our datasets. For both datasets, we select 30k samples as the training data. The training hyper-parameters of RL distilled data collection, reward model training, and the main experiments can be found in Table 9 in the Appendix.

Math Multi-Subject E M H Avg STEM Social Humanities Applied Others Avg

Method Reward Score Type

Qwen2.5-72B-Instruct – – 44.2 57.7 40.3 47.4 25.2 20.1 28.7 20.5 21.0 22.6 DeepSeek-R1-Distill-Qwen-32B – – 27.6 34.8 17.4 26.6 23.2 21.8 26.7 20.5 18.5 21.7

Base – – 43.1 53.9 33.2 43.4 16.3 14.9 15.2 13.3 14.8 15.0 SFT – – 53.6 50.5 32.9 45.7 24.6 22.8 25.7 20.9 22.6 23.1

binary 58.5 66.5 46.7 57.2 25.3 26.6 27.7 21.1 20.7 24.2 soft 46.0 47.7 31.5 41.7 22.0 20.3 23.1 16.9 20.5 20.3

rule based

binary 64.4 72.1 51.6 62.7 27.9 27.9 30.7 24.4 23.2 26.6 soft 62.5 71.2 53.1 62.3 32.2 32.8 36.0 24.9 27.9 30.3

REINFORCE

Qwen2.5-72BInstruct

binary 63.8 71.7 51.9 62.5 29.0 29.1 28.4 23.8 24.8 27.3 soft 62.9 70.7 53.0 62.2 32.7 32.8 35.6 28.6 27.4 31.2

RM-7B (ours)

binary 56.4 65.5 47.6 56.5 26.1 26.1 26.4 21.8 24.7 25.0 soft 49.4 52.9 36.2 46.2 22.5 22.0 25.7 18.6 20.2 21.4

rule based

binary 63.0 71.3 50.4 61.6 30.7 32.8 34.3 27.5 27.8 30.3 soft 62.7 70.4 50.5 61.2 30.8 30.1 33.7 25.6 25.4 28.8

REINFORCE++

Qwen2.5-72BInstruct

binary 63.1 71.3 51.5 62.0 30.2 30.8 31.0 26.6 26.3 29.1 soft 62.7 70.3 50.8 61.3 29.5 31.7 33.7 25.8 26.2 29.0

RM-7B (ours)

binary 58.2 67.0 50.2 58.5 28.2 27.9 27.4 22.4 24.5 26.3 soft 49.6 50.3 33.9 44.6 16.7 17.3 18.8 14.5 16.9 16.6

rule based

binary 63.0 70.8 51.1 61.6 29.4 30.5 33.7 24.6 26.1 28.4 soft 63.8 71.0 52.4 62.4 32.9 31.4 34.7 27.7 26.8 30.6

RLOO

Qwen2.5-72BInstruct

binary 63.4 71.8 53.8 63.0 29.3 29.0 33.3 25.8 25.6 28.1 soft 63.3 71.7 53.6 62.9 31.0 32.0 35.6 27.0 27.0 30.0

RM-7B (ours)

Table 1: Performance Comparison of Different Methods. Base model: Qwen2.5-7B. E: elementary. M: middle. H: high.

#### 4.5 Main Results

- Table 1 shows the results on mathematics and multi-subject tasks. We have the following observations:

Evaluation on Base Models Both our math and multi-subject data have demonstrated notable difficulty, with even strong open-source models like Qwen2.5-72B-Instruct (Team, 2024) and DeepSeekR1-Distill-Qwen-32B (Guo et al., 2025) performing unsatisfactorily, particularly on multi-subject tasks (21.7% for DeepSeek-R1-Distill-Qwen-32B and 22.6% for Qwen2.5-72B-Instruct). We believe that more challenging datasets will better facilitate exploration across the industry.

SFT vs. RL SFT significantly underperforms RL on both math and multi-subject tasks. Notably on math, SFT merely improves the model performance from 43.4% to 45.7%, falling far short of rule-based reward RL (RLOO, 58.8%) and lagging even further behind model-based reward RL (RM-7B, 63.0%). These findings demonstrate RL’s distinct advantages and potential in reasoning tasks when there is no high-quality Chain-of-Thoughts for training.

Model-based Reward vs. Rule-based Reward From the table, we can conclude that modelbased reward consistently outperforms rule-based reward in free-form reference-based scenarios. For instance, RM-7B (ours) and Qwen-2.5-72b-Instruct with binary reward achieves 63.0% and

- 61.6% respectively on average with RLOO, while rule-based reward only gets 58.5%. Notably, our distilled 7B reward model exhibits competitive performance against its much larger predecessor, Qwen2.5-72B-Instruct. In multi-subject evaluations using REINFORCE, the model trained from RM-7B achieves 31.2% accuracy compared to the 72B model’s 30.3% – a significant improvement given the substantial parameter disparity. This enhanced capability likely emerges from stabilized response patterns developed during training, which better align with the generative reward model’s objectives compared to the base model’s more variable outputs.

Binary Reward vs. Soft Reward For rule-based reward, soft reward consistently underperforms binary reward. This discrepancy may stem from redundant tokens between the model’s generated answers and reference labels, which can lower the reward scores for correct answers. A potential improvement could involve adopting metrics like cosine similarity of sentence embeddings as soft rewards, as these may better capture semantic alignment. In contrast, for model-based reward, binary and soft rewards yield comparable results on math tasks. This suggests that the model likely produces judgments with extremely high confidence, as determining answer-label matches in

mathematical problems is relatively easy. However, in multi-subject tasks, where reference labels exhibit greater diversity and consequently higher judgment complexity, soft rewards demonstrate more conservative scoring behavior. This conservatism in ambiguous cases enables soft rewards to outperform binary rewards in certain scenarios (31.2% vs. 27.3%, REINFORCE, RM-7B), as their soft scoring better accommodates the inherent uncertainty of open-domain evaluation.

Summary Our method establishes new state-of-the-art performance in RLVR through three key innovations: (1) Our proposed model-based reward is much stronger than rule-based baseline, allowing various RL methods to obtain very accurate rewards in general domain scenarios. (2) Building upon the data distilled from Qwen2.5-72B-Instruct, we develop a computationally efficient 7B model that can achieve comparable or even better performance. (3) We extend binary reward to soft reward, which can get more conservative scores for ambiguous cases, which can help get better performance when the reference answers exhibit greater diversity and consequently higher judgment complexity.

#### 4.6 Scaling Experiments

Math Multi-Subject E M H Avg STEM Social Humanities Applied Others Avg

Method Scale

20k 58.9 68.1 47.6 58.2 27.3 28.0 31.4 23.5 23.0 26.2 40k 61.5 69.4 55.4 62.1 25.1 24.8 27.4 21.0 23.0 24.0 60k 62.6 69.8 56.8 63.1 20.0 21.9 26.4 16.6 19.9 20.1 80k 62.4 68.2 53.6 61.4 19.2 18.3 26.7 15.1 16.4 18.0 100k 52.6 57.2 45.2 51.7 17.8 18.2 20.5 13.4 16.4 16.9

Rule based

20k 64.9 71.8 53.4 63.4 30.8 34.6 31.7 28.0 27.7 30.8 40k 65.6 72.4 54.4 64.1 34.3 33.7 36.3 29.5 28.6 32.4 60k 66.0 71.6 53.2 63.6 33.3 36.6 37.3 31.5 28.9 33.3 80k 66.6 72.3 55.6 64.8 34.5 38.6 38.3 31.6 31.0 34.6 100k 67.1 72.3 55.6 65.0 35.1 38.5 39.3 32.7 30.7 35.0

RM-7B (ours)

- Table 2: The results of the scaling experiments. We use RLOO as the RL algorithm and binary reward as the score type.

Scalability has emerged as a critical property in the RL-based training era. A key question worthy of investigation is whether model performance can continue to improve as RL training progresses and data volume increases. To examine this, we conduct experiments using our trained reward model against rule-based reward while progressively scaling the dataset. We randomly sampled 100k samples from our training corpus as the baseline set, conducting evaluations on both mathematical reasoning and multi-subject tasks. Table 2 shows the experimental results.

The results reveal significant differences in scaling capabilities. The rule-based reward demonstrates unstable scalability across both mathematical and multi-subject tasks, exhibiting substantial performance fluctuations and eventual degradation as RL training continues. In contrast, our learned reward model shows consistent improvement trends throughout the training process. This empirical evidence highlights the inherent scalability advantages of model-based rewards compared to rule-based rewards.

#### 4.7 Out-of-Distribution Evaluation

To further validate the effectiveness of our reward model, we conduct additional evaluations on two benchmarks: NaturalReasoning (Yuan et al., 2025) and WebInstruct (Yue et al., 2024). We compare the performance of the rule-based reward with our RM-7B using RLOO with binary reward. The base model is Qwen2.5-7B. For both datasets, we randomly select 30K examples for training and

- 5K sample for evaluation. The results are shown in Table 3. As can be seen from the table, the performance of RM-7B remains significantly superior to the rule-based reward on datasets from

Method Natural Reasoning WebInstruct Rule based 29.4 33.9 RM-7B (ours) 39.8 44.0

Table 3: The results of the Out-of-Distribution evaluation

other domains. This demonstrates that our general-purpose reward model can extend to other domains while maintaining strong performance.

### 5 Discussions and Conclusions

In this work, we simplify the verification task by instructing a generative reward model to output either 1 or 0, without requiring chain-of-thought (CoT) reasoning (Nye et al., 2021; Wei et al., 2022). While CoT has proven useful in both reference-based (Team et al., 2025) and referencefree (Zhang et al., 2024a) settings, it remains an open question how necessary in-depth rationales are for assessing semantic equivalence between reference answers and model responses in the same language, particularly when focusing on the conclusive part of each response. This also raises a related question for process reward modeling (Lightman et al., 2023) in RLVR: how should rewards be assigned when there is no direct supervision for intermediate steps, regardless of the step segmentation method?

In addition, we do not consider format-based rewards (Guo et al., 2025; Xie et al., 2025) in this work. We revisit the role of format-related constraints and rewards in this context. In prior work, pattern-based functions are often used for scoring, making it critical to guide LLMs to enclose their final answers in an easily parsed format. These extracted answers are then compared with the reference answers for verification and evaluation. In contrast, by reintroducing a reward model in RLVR without imposing any format constraints on reference answers or model responses, we reduce the need for extensive human effort in data standardization and pattern design.

### References

Arash Ahmadian, Chris Cremer, Matthias Gall´e, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Ust¨ un,¨ and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arXiv preprint arXiv:2402.14740, 2024.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Junying Chen, Zhenyang Cai, Ke Ji, Xidong Wang, Wanlong Liu, Rongsheng Wang, Jianye Hou, and Benyou Wang. Huatuogpt-o1, towards medical complex reasoning with llms. arXiv preprint arXiv:2412.18925, 2024.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve

- math word problems. arXiv preprint arXiv:2110.14168, 2021a.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve

- math word problems. arXiv preprint arXiv:2110.14168, 2021b.

Kanishk Gandhi, Denise HJ Lee, Gabriel Grand, Muxin Liu, Winson Cheng, Archit Sharma, and Noah Goodman. Stream of search (sos): Learning to search in language. In First Conference on Language Modeling, 2024.

Jonas Gehring, Kunhao Zheng, Jade Copet, Vegard Mella, Taco Cohen, and Gabriel Synnaeve. Rlef: Grounding code llms in execution feedback with reinforcement learning. arXiv preprint arXiv:2410.02089, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, et al. Measuring coding challenge competence with apps. arXiv preprint arXiv:2105.09938, 2021a.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset, 2021b.

Jian Hu. Reinforce++: A simple and efficient approach for aligning large language models. arXiv

preprint arXiv:2501.03262, 2025. Paul Jaccard. The distribution of the flora in the alpine zone. New phytologist, 1912. Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 reinforce samples, get a baseline for free!

2019.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. T\” ulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

Long Li, Xuzheng He, Haozhe Wang, Linlin Wang, and Liang He. How do humans write code? large models do it the same way too. arXiv preprint arXiv:2402.15729, 2024.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.

Jiawei Liu and Lingming Zhang. Code-r1: Reproducing r1 for code with reliable rewards. 2025. Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai,

Jeffrey Luo, Tianjun Zhang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl, 2025. Notion Blog.

Trung Quoc Luong, Xinbo Zhang, Zhanming Jie, Peng Sun, Xiaoran Jin, and Hang Li. Reft: Reasoning with reinforced fine-tuning. arXiv preprint arXiv:2401.08967, 2024.

Ruotian Ma, Peisong Wang, Cheng Liu, Xingyan Liu, Jiaqi Chen, Bang Zhang, Xin Zhou, Nan Du, and Jia Li. S2r: Teaching llms to self-verify and self-correct via reinforcement learning. arXiv preprint arXiv:2502.12853, 2025a.

Zexiong Ma, Chao Peng, Pengfei Gao, Xiangxin Meng, Yanzhen Zou, and Bing Xie. Sorft: Issue resolving with subtask-oriented reinforced fine-tuning. arXiv preprint arXiv:2502.20127, 2025b.

Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. Show your work: Scratchpads for intermediate computation with language models. 2021.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730– 27744, 2022.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33:3008–3021, 2020.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.

Qwen Team. Qwen2.5: A party of foundation models, September 2024. URL https://qwenlm.

##### github.io/blog/qwen2.5/.

Ye Tian, Baolin Peng, Linfeng Song, Lifeng Jin, Dian Yu, Lei Han, Haitao Mi, and Dong Yu. Toward self-improvement of llms via imagination, searching, and criticizing. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 52723–52748. Curran Associates, Inc., 2024. URL https://proceedings.neurips.cc/paper_files/paper/2024/file/ 5e5853f35164e434015716a8c2a66543-Paper-Conference.pdf.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Ronald J Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8:229–256, 1992.

Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu, and Chong Luo. Logic-rl: Unleashing llm reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2502.14768, 2025.

Wei Xiong, Hanning Zhang, Chenlu Ye, Lichang Chen, Nan Jiang, and Tong Zhang. Self-rewarding correction for mathematical reasoning, 2025.

Dian Yu, Kai Sun, Dong Yu, and Claire Cardie. Self-teaching machines to read and comprehend with large-scale multi-subject question-answering data. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih (eds.), Findings of the Association for Computational Linguistics: EMNLP 2021, pp. 56–68, Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.findings-emnlp.6. URL https://aclanthology.org/2021.findings-emnlp.6/.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Weizhe Yuan, Jane Yu, Song Jiang, Karthik Padthe, Yang Li, Dong Wang, Ilia Kulikov, Kyunghyun Cho, Yuandong Tian, Jason E Weston, et al. Naturalreasoning: Reasoning in the wild with 2.8 m challenging questions. arXiv preprint arXiv:2502.13124, 2025.

Xiang Yue, Tuney Zheng, Ge Zhang, and Wenhu Chen. Mammoth2: Scaling instructions from the web. arXiv preprint arXiv:2405.03548, 2024.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. Star: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems, 35:15476–15488, 2022.

Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerlzoo: Investigating and taming zero reinforcement learning for open base models in the wild, 2025. URL https://arxiv.org/abs/2503.18892.

Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, and Rishabh Agarwal. Generative verifiers: Reward modeling as next-token prediction. arXiv preprint arXiv:2408.15240, 2024a.

Yuxiang Zhang, Yuqi Yang, Jiangming Shu, Yuhang Wang, Jinlin Xiao, and Jitao Sang. Openrft: Adapting reasoning foundation model for domain-specific tasks with reinforcement fine-tuning. arXiv preprint arXiv:2412.16849, 2024b.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36:46595–46623, 2023.

### A Appendix

- A.1 Template

- Table 4 shows the template for the grading task. Table 5 shows the template for the classification task. Table 6 shows the classification of subjects into STEM (Science, Technology, Engineering, and Mathematics), Social Sciences, Humanities, and Applied Sciences.

Given a problem , determine whether the final answer in the provided ( incomplete) solution process matches the reference answer.

The reference answer may be one single option character (e.g., A, B, C, D), a numerical value , an expression , or a list of answers if multiple questions are involved.

**The reference answer may be in Chinese or another language , but your

evaluation should be language -agnostic .** Your task:

- - Compare the final output of the solution process with the reference answer

.

- - If they **match exactly**, output **YES**.
- - If they **do not match**, output **NO**.
- - If the solution process is unclear , incomplete , or ambiguous , assume it is incorrect and output **NO**.

Your output must be strictly **’YES ’** or **’NO ’**, with no additional words , punctuation , or explanation.

---

**Question :** {question}

**Solution Process (Final Step Only):** {response}

**Reference Answer :** {reference} **Output :**

Table 4: Template for the grading task.

- A.2 Agreement

Note that for each instance, we have only a single decision from GPT-4o. While it may align more closely with an individual sampled decision from the reward model than with the majority vote (when m > 1), the latter provides a more stable and deterministic outcome by reducing randomness during grading.

- A.3 REINFORCE

∇θEyi∼πθ(·|x) r(x,a,yi) = ∑

yi

∇θ πθ(y|x) r(x, a, yi)

= ∑

yi

πθ(y|x)∇θ log πθ(y|x) r(x, a, yi)

= Eyi∼πθ(·|x) ∇θ log πθ(yi|x)r(x, a, yi) .

(7)

- A.4 Hyper parameters Table 9 shows the hyper parameters of our experiments.

Based on the content of ’Question ’ and ’Answer ’ classify the subject into one of the following categories.

Return only the corresponding subject ID. If classification is uncertain , return 999.

**Question :** {question}

**Answer :** {answer}

110 Mathematics 120 Information Science and System Science 130 Mechanics 140 Physics 150 Chemistry 170 Earth Science 180 Biology 190 Psychology 210 Agronomy 230 Animal Husbandry and Veterinary Science 310 Basic Medicine 320 Clinical Medicine 330 Preventive Medicine and Public Health 350 Pharmacy 360 Chinese Medicine and Chinese Materia Medica 413 Information and System Science Related Engineering and Technology 416 Natural Science Related Engineering and Technology 420 Surveying and Mapping Science and Technology 430 Materials Science 460 Mechanical Engineering 470 Power and Electrical Engineering 510 Electronics and Communications Technology 520 Computer Science and Technology 530 Chemical Engineering 550 Food Science and Technology 560 Civil Engineering 570 Water Conservancy Engineering 580 Transportation Engineering 610 Environmental/Resource Science and Technology 620 Safety Science and Technology 630 Management 710 Marxism 720 Philosophy 730 Religious Studies 740 Linguistics 750 Literature 760 Art 770 History 790 Economics 810 Political Science 820 Law 840 Sociology 850 Ethnology and Cultural Studies 860 Journalism and Communication 870 Library , Information , and Documentation 880 Education 890 Sports Science 910 Statistics 999 Unclassified

- Table 5: Template for the classification task, with subject names and IDs referenced from (Yu et al., 2021).

###### Category Subject IDs STEM

110 (Mathematics), 120 (Information Science and System Science), 130 (Mechanics), 140 (Physics), 150 (Chemistry), 170 (Earth Science), 180 (Biology), 430 (Materials Science), 460 ( Mechanical Engineering), 470 (Power and Electrical Engineering), 510 (

Electronics and Communications Technology), 520 (Computer Science and Technology), 530 (

Chemical Engineering), 560 (Civil Engineering), 570 (Water Conservancy Engineering),

580 (Transportation Engineering), 610 ( Environmental/Resource Science and Technology) ,

620 (Safety Science and Technology), 910 ( Statistics)

###### Social Sciences

190 (Psychology), 790 (Economics), 810 (Political Science), 820 (Law), 840 (Sociology), 850 (Ethnology and Cultural Studies), 860 (Journalism and Communication), 870 (Library , Information , and Documentation), 880 (Education), 890 (Sports Science), 630 ( Management)

###### Humanities

710 (Marxism), 720 (Philosophy), 730 (Religious Studies), 740 (Linguistics), 750 (Literature), 760 (Art), 770 (History)

###### Applied Sciences

210 (Agronomy), 230 (Animal Husbandry and

Veterinary Science), 310 (Basic Medicine), 320 (Clinical Medicine), 330 (Preventive Medicine and Public Health), 350 (

Pharmacy), 360 (Chinese Medicine and Chinese Materia Medica), 413 (Information and System Science Related

Engineering and Technology), 416 (Natural Science Related Engineering and Technology), 420 (Surveying and Mapping Science and Technology), 550 (Food Science and Technology)

- Table 6: Classification of subjects into STEM (Science, Technology, Engineering, and Mathematics), Social Sciences, Humanities, and Applied Sciences.

Level Agreement (κ ↑) m = 1 m = 10

elementary 0.844 0.838 middle 0.885 0.883 high 0.849 0.846 average 0.864 0.861

- Table 7: Cohen’s Kappa agreement (κ) between GPT-4o and majority voting (m: the number of votes) using Qwen2.5-72B-Instruct as evaluator across different education levels of math problems.

Cohen's Kappa: GPT-4o vs. Majority Vote with m Graders

0.88

0.87

0.86

Cohen'sKappa

Multi-Subject

0.85

Math

0.84

0.83

0.82

Almost Perfect Agreement (0.81)

0.81

1 2 3 4 5 6 7 8 9 10

Majority vote over m graders (YES count m/2 )

- Figure 3: Agreement between GPT-4o and Majority Vote with m Graders, measured by Cohen’s Kappa.

Level Agreement (κ ↑)

m = 1 m = 10 college-level 0.881 0.883

- Table 8: Cohen’s Kappa agreement (κ) between GPT-4o and majority voting (m: the number of votes) using Qwen2.5-72B-Instruct as evaluator across college-level multi-subject problems.

Hyperparameter

Reward Training Main Experiments RL SFT RL SFT

micro train batch size 8 4 8 4 train batch size 128 128 128 128 micro rollout batch size 16 – 16 – rollout batch size 128 – 128 – n samples per prompt 4 – 4 – max samples 40000 1600000 30000 30000 max epochs 1 1 1 1 prompt max len 1024 – 1024 – generate max len 1024 – 1024 – max len – 4096 – 4096 actor learning rate 5e-7 – 5e-7 – init kl coef 0.01 – 0.01 –

- Table 9: Training hyper parameters. Other hyper parameters are the default configuration in OpenRLHF.

coarse fine question answer Social Sciences Psychology Setting up an activity for students to

Self-awareness guidance

’bomb’ each other with compliments belongs to ( ).

STEM Civil Engineering

A gravity retaining wall meets the Rankine earth pressure conditions, H = 3m, top width 2m, bottom width 3m, fill c = 0, ϕ = 30◦, γ = 18.0kN/m3, the base friction coefficient is 0.4, the anti-sliding stability safety factor Ks and the antitilting stability safety factor Kt are respectively ()

2.67; 1.73

Humanities Philosophy Laozi pointed out in the ’Tao Te Ching’, ’Without leaving the door, one knows the world; without peering through the window, one knows the way of heaven. The farther one goes, the less one knows. Therefore, the sage knows without traveling, sees without looking, and achieves without doing.’ Laozi’s view here

denies the decisive role of practice in understanding

Applied Sciences Agronomy Under light, the physiological processes that can occur in the mesophyll cells and vascular bundle sheath cells of wheat (C3) are

Production of ATP and [H]

Table 10: Example question and reference answer pairs in ExamQA.

