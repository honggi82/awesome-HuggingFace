[Figure 1]

# arXiv:2601.21051v1[cs.AI]28Jan2026

## Llama-3.1-FoundationAI-SecurityLLM-Reasoning-8B Technical Report

Zhuoran Yang1,2,∗, Ed Li1,2,∗, Jianliang He1,2,∗, Aman Priyanshu1, Baturay Saglam1,2,∗, Paul Kassianik1, Sajana Weerawardhena1, Anu Vellore1, Blaine Nelson1, Neusha Javidnia1,3,∗, Arthur Goldblatt1, Fraser Burch1, Avi Zohary1, Assaf Eisenman1, Mahdi Sabbaghi1,4,∗, Supriti Vijay1,5,∗, Rahim Dharssi1, Dhruv Kedia1, Kojin Oshiba1, Yaron Singer1, Amin Karbasi1

1Foundation AI–Cisco Systems Inc. 2Yale University 3University of California, San Diego 4University of Pennsylvania

5Carnegie Mellon University

∗Work done while at Foundation AI

### Abstract

We present Foundation-Sec-8B-Reasoning, the first open-source native reasoning model for cybersecurity. Built upon our previously released Foundation-Sec-8B base model (derived from Llama-3.1-8B-Base), the model is trained through a two-stage process combining supervised fine-tuning (SFT) and reinforcement learning from verifiable rewards (RLVR). Our training leverages proprietary reasoning data spanning cybersecurity analysis, instructionfollowing, and mathematical reasoning. Evaluation across 10 cybersecurity benchmarks and 10 general-purpose benchmarks demonstrates performance competitive with significantly larger models on cybersecurity tasks while maintaining strong general capabilities. The model shows effective generalization on multi-hop reasoning tasks and strong safety performance when deployed with appropriate system prompts and guardrails. This work demonstrates that domain-specialized reasoning models can achieve strong performance on specialized tasks while maintaining broad general capabilities. We release the model publicly at huggingface.co/fdtn-ai/Foundation-Sec-8B-Reasoning.

#### 1. Introduction

The frontier of large language model (LLM) development has recently been advanced by the advent of native reasoning models, pioneered by OpenAI o1 [Jaech et al., 2024] and DeepSeek-R1 [Guo et al., 2025]. A defining characteristic of these models is their ability to generate explicit, step-by-step reasoning traces, often encapsulated in “<think>” tags, which reveal the model’s internal thought process before a final answer is synthesized. This paradigm of transparent reasoning has catalyzed significant performance improvements across a range of complex benchmarks, most notably in domains requiring sophisticated

|100<br><br>Llama-3.3-70B-Instruct<br><br>Foundation-Sec-8B-Instruct<br><br>Foundation-Sec-8B-Reasoning|
|---|

CTIBench-MCQA

|90<br><br>Llama-3.1-8B-Instruct<br><br>Foundation-Sec-8B-Instruct<br><br>Foundation-Sec-8B-Reasoning|
|---|

AlpacaEval 2

1

90

80

70

80

SecEval

CTIBench-RCM

60

MATH

BBH

70

50

60

40

50

30

| |
|---|

40

20

CTI-Reasoning

SecBench

HumanEval

IFEval

GSM8K

CWE-Prediction

(a) Cybersecurity Benchmarks (b) General-Purpose Benchmarks

- Figure 1 | Performance comparison of Foundation-Sec-8B-Reasoning against baseline models. (a) On cybersecurity benchmarks (CTIBench-MCQA, CTIBench-RCM, CTI-Reasoning, CWE-Prediction, SecBench, SecEval), our model performs on par with the 70B model (Llama-3.3-70B-Instruct) while significantly outperforming our previous instruction-tuned model (Foundation-Sec-8B-Instruct). (b) On general-purpose benchmarks (AlpacaEval 2, BBH, IFEval, GSM8K, HumanEval, MATH), our model achieves comparable performance to Llama-3.1-8B-Instruct on most tasks, with significantly better performance on AlpacaEval 2.

mathematical and computational logic. See, e.g., Kumar et al. [2025], Li et al. [2025] for recent surveys on reasoning large language models.

Despite these advancements, the application of native reasoning methodologies to the cybersecurity domain remains conspicuously nascent. While prevailing instruction-following models (see Section 2 for an overview) can address direct queries, they often struggle with the intricate, multi-step analytical processes critical to cybersecurity functions. These functions include threat intelligence analysis, vulnerability assessment, and incident response. These tasks demand more than just a correct answer; they require a verifiable and transparent line of reasoning. In cybersecurity, the “how” and “why” of a conclusion are often as important as the conclusion itself. For example, a model might correctly link an indicator of compromise to a MITRE ATT&CK technique [Strom et al.,2018], but if the reasoning is flawed, it could cause an analyst to misinterpret the adversary’s tactics and deploy ineffective countermeasures. A transparent reasoning process allows security professionals to audit the model’s logic, build trust in its outputs, and collaborate with the model to refine its analysis. This verifiability is crucial for high-stakes decisions, where a “black box” recommendation is often insufficient.

To address this critical gap, we introduce Foundation-Sec-8B-Reasoning, an 8B-parameter, native reasoning model engineered specifically for the complexities of the cybersecurity landscape. Our model is based on Foundation-Sec-8B [Kassianik et al., 2025] via post-training, a cybersecurity-specialized language model derived from Llama-3.1-8B-Base [Grattafiori et al., 2024]. Foundation-Sec-8B was developed by continued pre-training on 8 billion tokens of proprietary cybersecurity-focused data, starting from Llama-3.1-8B-Base. Building on this foundation, Foundation-Sec-8B-Reasoning is designed from its inception to “think” before it “speaks”, thereby providing the transparent and auditable reasoning capabilities required by security practitioners.

Our methodology is distinguished by its emphasis on cultivating native reasoning capabilities directly

from the base model, rather than fine-tuning an existing instruction-tuned model. Specifically, while our Foundation-Sec-8B-Instruct [Weerawardhena et al., 2025] was post-trained from the base model to directly generate answers, Foundation-Sec-8B-Reasoning undergoes a distinct post-training regimen. This regimen trains it to always generate an explicit reasoning trace before producing a final output, instilling a “think before you speak” approach from the very beginning of its development. This regimen consists of a two-stage pipeline: Supervised Fine-Tuning (SFT) followed by Reinforcement Learning with Verifiable Rewards (RLVR). The first stage involves large-scale supervised fine-tuning on a synthetic dataset of over two million exemplars, meticulously generated by a bespoke LLM agent leveraging Gemini-2.5-Flash [Comanici et al., 2025]. This foundational stage establishes robust, cross-domain reasoning behavior by training the model on a diverse corpus spanning mathematics, code generation, instruction following, and cybersecurity. The second stage further refines these capabilities through reinforcement learning, where the model’s policy is optimized against a suite of verifiable rewards across various reasoning tasks. This phase was critical for addressing and solving technical challenges such as length-dependent gradient bias, ensuring stable and balanced learning.

Comprehensive evaluations validate the efficacy of our methodology. Our results indicate that Foundation-Sec-8B-Reasoning exhibits substantial performance gains over our prior instruction-tuned model, Foundation-Sec-8B-Instruct. This is particularly evident on challenging cybersecurity benchmarks, such as the CTIBench benchmark suite [Alam et al., 2024], where the model’s performance is competitive with that of much larger models, including Llama-3.3-70B-Instruct. Furthermore, on general reasoning tasks, its performance is on par with or exceeds its instruction-following predecessor. To our knowledge, Foundation-Sec-8B-Reasoning represents the first open-source, native reasoning model explicitly architected for the cybersecurity domain. We believe this work constitutes a significant step toward integrating more powerful and transparent AI into security-critical workflows.

#### 2. Related Work

We examine prior work on instruction-tuned, cybersecurity-specialized models and relevant posttraining approaches. While much of the literature has focused on secure code generation or vulnerability detection Kassianik et al. [2025], our goal is to align a general-purpose LLM with core cybersecurity tasks. We aim to enable broader applicability of the model across diverse use cases through instructionfollowing training. For a comprehensive overview of cybersecurity-focused language models, we refer readers to existing surveys [Zhang et al., 2024].

Open-Source Reasoning Models. Recent advancements in LLMs have yielded powerful generalpurpose reasoning language models. Notably, DeepSeek-R1 [Guo et al., 2025] emerged as a powerful reasoning model demonstrating effective test-time scaling, significantly boosting performance in mathematics and coding by pioneering advanced Reinforcement Learning (RL) methodologies like Group Relative Policy Optimization (GRPO) [Shao et al., 2024] to enhance complex problem-solving capabilities. Later, the Qwen series [Yang et al., 2025] employs a multi-stage post-training recipe: it utilizes SFT for an initial cold start, followed by the application of GRPO for general-purpose Reinforcement Learning on its large models, and subsequently uses strong-to-weak distillation to imbue smaller models with robust reasoning capabilities. Furthermore, the Phi-Reasoning models [Abdin et al., 2025] have established that high-quality, “textbook-style” synthetic data utilized during SFT can endow smaller-parameter models with significant reasoning abilities, with later iterations applying targeted RL to boost performance specifically in mathematical reasoning.

The Nemotron family [Bercovich et al., 2025], built on top of the Llama series [Grattafiori et al., 2024], exemplifies a hybrid approach and notably provided one of the first comprehensive open-source recipes for reasoning model training, integrating SFT on extensive synthetic and distilled datasets

with a subsequent large-scale RL phase to augment complex reasoning. Additionally, the GPT-OSS models [OpenAI, 2025] represent open-source reasoning models leveraging a Mixture of Experts (MoE) architecture to efficiently scale reasoning capabilities while maintaining computational efficiency. The prevailing training paradigm for these leading models consists of a multi-stage pipeline. Our work adopts this established SFT + RL methodology, specializing and adapting this pipeline for the distinct and high-stakes challenges inherent in the cybersecurity domain.

Cybersecurity LLMs. Instruction-tuned large language models for cybersecurity have been developed to assist security practitioners in all types of security related tasks and workflows. Lily-Cybersecurity-

- 7B-v0.2 [segolilylabs, 2025] is a cybersecurity-specialized model based on Mistral-7B [Jiang et al.,

- 2023], finetuned with 22,000 hand-crafted question-answer pairs covering topics from advanced persistent threats to penetration testing. DeepHat-V1-7B [DeepHat, 2025] (formerly WhiteRabbitNeo) is a cybersecurity-specialized model trained from Qwen2.5-Coder-7B [Hui et al., 2024]. In addition, Yu et al. [2025b] introduced Primus, a collection of datasets for cybersecurity LLM training and an instructiontuned model with improved cybersecurity performance trained using distilled data from larger models. Furthermore, Weerawardhena et al. [2025] introduced Foundation-Sec-8B-Instruct, which applies supervised fine-tuning and direct preference optimization after continued pre-training. Our model is trained from the same base model, and using a similar two-stage post-training pipeline. In summary, these instruction-tuned models handle security-related instructions effectively, but complex cybersecurity challenges often require multi-step reasoning to trace attack chains, evaluate cascading vulnerabilities, or synthesize threat intelligence. For these reasons, we develop Foundation-Sec-8B-Reasoning, the first reasoning model specialized in cybersecurity.

#### 3. Methodology

Our development of Foundation-Sec-8B-Reasoning is predicated on a two-stage training pipeline designed to cultivate native reasoning capabilities. This process begins with Foundation-Sec-8B [Kassianik et al., 2025], our open-source base model that was continuously pre-trained on 8 billion tokens of cybersecurity-focused data starting from Llama-3.1-8B-Base.

##### 3.1. Stage 1: Supervised Fine-Tuning for Native Reasoning

The first stage instills a native reasoning behavior by training the model to generate explicit reasoning traces encapsulated within “<think>...</think>” tags. This is accomplished via Supervised Fine-Tuning (SFT) on a diverse, large-scale synthetic dataset with these reasoning traces and tags.

##### 3.1.1. SFT Dataset and Training

The SFT dataset comprises about two million exemplars, curated to build a strong foundation in general reasoning and instruction following. As illustrated in Figure 2(a), the data is diverse. Cybersecurityrelated data, including question-answering and multiple-choice questions on topics like CVEs [The MITRE Corporation, 2025], MITRE ATT&CK [Strom et al., 2018], and CWEs [Christey et al., 2013], makes up over a quarter of the dataset. Mathematical reasoning and coding problems together account for approximately one third of the data. The remainder of the dataset is composed of instruction following, chat, science, and safety data, which collectively help to improve the model’s ability to follow instructions and interact in a conversational manner.

The SFT process ran for three epochs with a cosine learning rate scheduler and a learning rate of 2 × 10−5. This stage is crucial for teaching the model the fundamental structure of reasoning-based

Safety

Math

8.9%

Cybersecurity

Science

19.2%

26.8%

10.9%

Instruction Following

41.1%

13.5%

Chat

4.1%

20.9%

39.7%

Instruction Following

14.9%

Math

Cybersecurity

Coding

(a) SFT Data Composition (b) RL Data Composition

- Figure 2 | Data composition for the SFT and RL training stages. The SFT stage uses a diverse mix of data to instill broad reasoning abilities, while the RL stage emphasizes instruction following, cybersecurity, and mathematical reasoning to further refine the model’s reasoning abilities.

responses and expanding its instruction-following capabilities within the security domain.

##### 3.2. Stage 2: Reinforcement Learning with Verifiable Rewards

To sharpen the model’s reasoning accuracy, we employ a second stage of training using RL with verifiable rewards. The RL training data comprises datasets spanning instruction following, cybersecurity tasks, and mathematical reasoning, as illustrated in Figure 2(b).

We employ the GRPO algorithm [Shao et al., 2024] for RL training. For each prompt, we generate 𝑛= 5 responses and evaluate them with a task-specific verifier to obtain the binary reward signal. The policy is trained for two epochs using a cosine learning rate scheduler with a learning rate of 10−6 and a warmup phase. To maintain stability, policy updates are regularized with a KL-divergence penalty (coefficient of 0.02) against the original SFT model checkpoint. Through our RL training, we identified and tackled the following two critical challenges: (1) heterogeneity in RL data and (2) reward hacking and format degradation.

Data Heterogeneity. Training with heterogeneous RL data introduces concrete challenges: different tasks have widely varying output lengths and solving accuracies. We observe that the model tends to output nonsensical text on tasks where it is weak, and these excessively long samples often exhibit low-quality patterns such as gibberish and repetitive words. This heterogeneity creates difficulties for loss aggregation: tokens within longer responses contribute disproportionately to the total loss, potentially allowing low-quality failure modes to dominate policy updates.

We experimented with multiple loss aggregation strategies. First, a simple token-mean approach (averaging losses across all tokens in the batch without per-sample normalization), as proposed in DAPO [Yu et al., 2025a], cannot resolve these issues and frequently leads to degenerate failure modes in our setting. We hypothesize that DAPO’s additional techniques, such as “clip higher” (clipping advantages at higher values to limit the impact of outliers) and “soft overlong punishment” (penalizing sequences that exceed reasonable length), are necessary when using token-mean aggregation to mitigate these pathologies. In contrast, we found that two approaches provide stable training: (1) the original sample-level loss calculation from GRPO [Shao et al., 2024], which first averages losses by token within each sample and then aggregates across samples, ensuring each sample contributes equally regardless of length; and (2) the loss aggregation approach proposed in Dr.GRPO [Liu et al., 2025]. Both methods

effectively prevent long, low-quality sequences from biasing the optimization.

Reward Hacking and Format Degradation. A critical failure mode in RL with outcome-based verifiers is reward hacking, where the model learns to achieve high rewards without performing the desired behavior. We observed that a naive GRPO implementation led the model to produce correct final answers while generating empty or nonsensical reasoning traces, sometimes omitting the required “<think>...</think>” tags entirely. For example, the model might output “<think> No</think>” for all prompts. The main reason is that the verifier rewards only check the validity of the final answer, not the reasoning process. Note that the GRPO loss decreases as the output length increases. All else equal, the model would prefer short answers over long ones to minimize the loss. Thus, without additional regularization, the RL objective inadvertently incentivizes degenerate reasoning traces, leading to reward hacking. To counteract this, we engineered a format penalty into our reward function that programmatically validates the response format. This penalty ensures that the required reasoning tags are present and that the enclosed reasoning is non-trivial (i.e., not overly short or repetitive), thereby enforcing the generation of well-formed and meaningful reasoning traces.

- 4. Evaluation Results We conduct a comprehensive evaluation of Foundation-Sec-8B-Reasoning across four dimensions:

- (1) cybersecurity benchmarks to assess domain-specific capabilities (Section 4.1), (2) general-purpose benchmarks to verify that specialization does not compromise general intelligence (Section 4.2), (3) safety evaluation using HarmBench to ensure responsible deployment (Section 4.3), and (4) an ablation study comparing our SFT checkpoint with the final RL-trained model to understand the contributions of each training stage (Section 4.4).

4.1. Cybersecurity Benchmarks

Benchmarks. We evaluate on 10 cybersecurity benchmarks spanning diverse aspects of the domain. Table 1 provides an overview. These include: CTIBench tasks (MCQA, RCM, VSP, ATE) for cyber threat intelligence [Alam et al., 2024]; two proprietary benchmarks (CTI-Reasoning for multi-hop reasoning and CWE-Prediction for vulnerability classification); and established benchmarks (MMLUSecurity [Hendrycks et al., 2021a], CyberMetric-2000 [Tihanyi et al., 2024], SecBench [Jing et al., 2024], SecEval [Li et al., 2023]) for general security knowledge. Notably, CWE-Prediction is constructed in the same way as CTIBench-RCM (CVE to CWE mapping) but uses recent entries from 2025 for CVE [The MITRE Corporation, 2025] and 2024-2025 for GitHub Security Advisories (GHSA) [GitHub, 2024] to test the same vulnerability classification ability on previously unseen data, evaluating model generalization beyond the training distribution. The benchmarks vary in size from 60 samples (CTIBench-ATE) to 3,000 samples (CWE-Prediction), with task formats including multiple-choice questions, vulnerability mapping, severity prediction, and attack technique extraction. Together, these provide comprehensive coverage of the knowledge and reasoning capabilities required for effective cybersecurity AI systems.

Models. We compare Foundation-Sec-8B-Reasoning against 18 baseline models organized into five groups to enable fair comparisons across different scales and training paradigms: (1) Smaller specialized models [Abdin et al., 2024, DeepHat, 2025, Yang et al., 2025]: DeepHat-v1-7B, Qwen-3-8B/14B, Phi-4;

- (2) Llama-family and cybersecurity-specialized 8B models [Grattafiori et al., 2024, Weerawardhena et al.,

- 2025, Yu et al., 2025b]: Llama-3.1-8B-Instruct, Llama-3.3-70B-Instruct, Foundation-Sec-8B-Instruct , Llama-Primus variants; (3) GPT-OSS models [OpenAI, 2025]: 20B and 120B; (4) Frontier commercial

###### Benchmark Samples Task Type Abilities Tested

CTIBench-MCQA 2500 Multiple choice CTI knowledge and understanding CTIBench-RCM 1000 CVE to CWE mapping Vulnerability analysis and reasoning CTIBench-VSP 1000 Severity prediction Technical context understanding CTIBench-ATE 60 Technique extraction Threat behavior reasoning CTI-Reasoning 200 Multiple choice Multi-hop security reasoning CWE-Prediction 3000 CVE to CWE mapping Same as RCM but with recent unseen data MMLU-Security 100 Multiple choice Broad security domain knowledge CyberMetric-2000 2000 Multiple choice Security standards and publications SecBench 595 Multiple choice Broad security domain knowledge SecEval 1225 Multiple choice Broad security domain knowledge

- Table 1 | Overview of cybersecurity benchmarks. Benchmarks cover cyber threat intelligence (CTI), vulnerability analysis through Common Vulnerabilities and Exposures (CVE) to Common Weakness Enumeration (CWE) mapping, and broad security domain knowledge. See Appendix A for detailed descriptions.

models: GPT-4.1, O3-Mini, GPT-5 family; and (5) Our reasoning model. Bold scores in Tables 2 and 3 indicate the best performance within each model group.

Evaluation Protocol. We use the Cisco Foundation-AI’s Testing Hub (FAITH)1 evaluation framework with the following settings:

- • Sampling: We run 5 independent trials with different random seeds to ensure robustness. For instruct models, we use temperature = 0.3 and top-𝑝 = 1.0. For reasoning models (FoundationSec-8B-Reasoning and Qwen-3 series), we use temperature = 0.6 and top-𝑝 = 0.95 to allow more diverse reasoning paths. For GPT-OSS models, we use temperature = 0.7 and top-𝑝 = 0.95.
- • Answer Extraction: Our system prompt requires models to place their answer in the last line of the response. We extract the last line using regular expressions and compute accuracy based on the extracted answer. For CTIBench-VSP, we compute CVSS scores as detailed in Appendix A. For CTIBench-ATE, we report micro-F1 scores. Prompts for these benchmarks, as well as the details of the two proprietary benchmarks (CTI-Reasoning and CWE-Prediction), are provided in Appendix A.

##### 4.1.1. Results and Analysis

Our Foundation-Sec-8B-Reasoning demonstrates strong and consistent performance across the benchmark suite. The model achieves particularly notable results on CTIBench-RCM (75.3%), outperforming all other models evaluated. On CTIBench-MCQA, the model achieves 69.1%, representing a +4.1 pp improvement over Foundation-Sec-8B-Instruct (65.0%) and demonstrating performance comparable to the much larger Llama-3.3-70B-Instruct (69.2%, 9× more parameters). On CWE-Prediction, FoundationSec-8B-Reasoning achieves 70.4% accuracy, demonstrating robust vulnerability classification capabilities. The model maintains competitive performance across other benchmarks including CTI-Reasoning (41.1%), MMLU-Security (78.2%), and SecEval (84.8%). Notably, within the 8B parameter group, Foundation-Sec-8B-Reasoning dominates all Llama-3.1-8B derived models on 8 out of 10 benchmarks, demonstrating the effectiveness of reasoning-focused training for cybersecurity tasks.

1https://github.com/cisco-foundation-ai/faith.

###### Model CTIBench-MCQA CTIBench-RCM CTIBench-VSP CTIBench-ATE CTI-Reasoning

DeepHat-V1-7B 0.493±0.004 0.434±0.013 0.045±0.004 0.004±0.003 0.323±0.024 Qwen-3-8B 0.649±0.003 0.542±0.008 0.863±0.001 0.408±0.020 0.395±0.025 Qwen-3-14B 0.664±0.006 0.612±0.005 0.869±0.001 0.502±0.011 0.441±0.035 Phi-4 0.658±0.005 0.629±0.003 0.647±0.013 0.435±0.014 0.444±0.017 Llama-3.1-8B-Instruct 0.607±0.004 0.531±0.003 0.811±0.005 0.132±0.017 0.335±0.027 Llama-3.3-70B-Instruct 0.692±0.002 0.684±0.004 0.841±0.001 0.519±0.016 0.507±0.016 Foundation-Sec-8B-Instruct 0.650±0.003 0.704±0.003 0.840±0.004 0.358±0.013 0.364±0.013 Llama-Primus-Merged 0.604±0.011 0.665±0.006 0.788±0.004 0.058±0.010 0.348±0.024 Llama-Primus-Nemotron-70B-Instruct 0.705±0.003 0.664±0.009 0.239±0.015 0.268±0.035 0.485±0.028 GPT-OSS-20B 0.655±0.004 0.610±0.010 0.864±0.003 0.478±0.024 0.460±0.022 GPT-OSS-120B 0.714±0.006 0.712±0.005 0.883±0.003 0.282±0.025 0.496±0.019 GPT-4.1 0.760±0.004 0.730±0.006 0.848±0.004 0.696±0.007 0.596±0.015 o3-Mini 0.716±0.002 0.708±0.002 0.843±0.008 0.599±0.013 0.479±0.011 GPT-5-Nano 0.688±0.003 0.672±0.007 0.822±0.006 0.453±0.029 0.431±0.015 GPT-5-Mini 0.753±0.003 0.723±0.005 0.892±0.001 0.681±0.003 0.578±0.014 GPT-5 0.819±0.002 0.728±0.002 0.903±0.003 0.578±0.028 0.643±0.017 Foundation-Sec-8B-Reasoning 0.691±0.006 0.753±0.008 0.856±0.004 0.491±0.017 0.411±0.017

- Table 2 | Performance on cybersecurity benchmarks (Part 1). Models are organized into five groups: (1) smaller specialized models (DeepHat-V1-7B, Qwen-3-8B/14B, Phi-4), (2) Llama-family and cybersecurityspecialized 8B models (Llama-3.1/3.3, Foundation-Sec-8B-Instruct, Llama-Primus variants), (3) GPTOSS models (20B, 120B), (4) frontier OpenAI API models (GPT-4.1, O3-Mini, GPT-5 family), and (5) our reasoning model. All scores are generated by averaging over 5 trials ± standard deviation. For CTIBench-VSP, we report average CVSS scores; for CTIBench-ATE, we report the micro-F1 scores. Bold scores indicate the best performance within each model group.

###### Model CWE-Prediction MMLU-Security Cybermetric-2000 SecBench SecEval

DeepHat-V1-7B 0.360±0.003 0.666±0.022 0.730±0.005 0.615±0.007 0.790±0.005 Qwen-3-8B 0.464±0.002 0.846±0.014 0.903±0.002 0.835±0.007 0.886±0.003 Qwen-3-14B 0.534±0.005 0.870±0.009 0.910±0.002 0.849±0.004 0.892±0.003 Phi-4 0.554±0.004 0.844±0.010 0.912±0.004 0.813±0.004 0.898±0.004 Llama-3.1-8B-Instruct 0.473±0.003 0.768±0.010 0.851±0.004 0.749±0.007 0.832±0.003 Llama-3.3-70B-Instruct 0.607±0.003 0.864±0.017 0.930±0.003 0.842±0.005 0.906±0.004 Foundation-Sec-8B-Instruct 0.616±0.003 0.770±0.026 0.847±0.005 0.744±0.014 0.829±0.003 Llama-Primus-Merged 0.550±0.002 0.762±0.029 0.862±0.002 0.749±0.004 0.811±0.009 Llama-Primus-Nemotron-70B-Instruct 0.613±0.003 0.834±0.027 0.906±0.003 0.830±0.006 0.882±0.005 GPT-OSS-20B 0.519±0.006 0.870±0.011 0.893±0.005 0.804±0.005 0.870±0.006 GPT-OSS-120B 0.661±0.002 0.880±0.019 0.926±0.002 0.853±0.004 0.904±0.005 GPT-4.1 0.682±0.004 0.872±0.007 0.937±0.003 0.872±0.002 0.919±0.001 O3-Mini 0.632±0.001 0.858±0.017 0.930±0.003 0.869±0.005 0.908±0.002 GPT-5-Nano 0.612±0.003 0.836±0.016 0.918±0.003 0.838±0.007 0.884±0.005 GPT-5-Mini 0.666±0.001 0.884±0.014 0.932±0.002 0.877±0.003 0.911±0.003 GPT-5 0.701±0.002 0.932±0.007 0.941±0.002 0.888±0.004 0.923±0.002 Foundation-Sec-8B-Reasoning 0.704±0.004 0.782±0.023 0.843±0.003 0.725±0.015 0.848±0.005

- Table 3 | Performance on cybersecurity benchmarks (Part 2). Models are organized into five groups as described in Table 2: (1) smaller specialized models, (2) Llama-family and cybersecurity-specialized 8B models, (3) GPT-OSS models, (4) frontier OpenAI API models, and (5) our reasoning model. We report average accuracy over 5 trials ± standard deviation for all benchmarks (CWE-Prediction, MMLUSecurity, Cybermetric-2000, SecBench, and SecEval). Bold scores indicate the best performance within each model group.

Figure 3 visualizes performance across selected models and benchmarks, highlighting the relative strengths of different approaches. Foundation-Sec-8B-Reasoning consistently ranks among the top performers, particularly on reasoning-intensive tasks. The visualization also reveals interesting patterns: larger models like Llama-3.3-70B-Instruct and GPT-OSS-120B excel on knowledge-based benchmarks, while our cybersecurity-specialized models maintain strong performance despite having significantly fewer parameters. See Tables 2 and 3 for detailed results2.

Key Finding: Cybersecurity Performance

Foundation-Sec-8B-Reasoning demonstrates exceptional performance on cybersecurity benchmarks:

- • Competitive with significantly larger models: Achieves comparable performance to Llama-3.3-70B-Instruct (9× more parameters) on CTIBench-MCQA (69.1% vs 69.2%) while outperforming it on CTIBench-RCM (75.3% vs 68.4%). Outperforms GPT-OSS-120B (15× more parameters) on CTIBench-RCM (75.3% vs 71.2%), with only a minor gap on CTIBench-MCQA (69.1% vs 71.4%).
- • Substantial improvement over instruction-tuned baseline: Outperforms FoundationSec-8B-Instruct on 8 out of 10 cybersecurity benchmarks, with notable gains on CTIBenchATE (+13.3 pp), CWE-Prediction (+8.7 pp), and CTIBench-RCM (+4.8 pp), where pp denotes percentage points.
- • Strong gains over Llama-3.1-8B-Instruct: Outperforms on 8 out of 10 benchmarks, with particularly significant improvements on CTIBench-ATE (+35.9 pp), CWEPrediction (+23.1 pp), CTIBench-RCM (+22.2 pp), CTIBench-MCQA (+8.3 pp), and CTI-Reasoning (+7.6 pp).

These results demonstrate that Foundation-Sec-8B-Reasoning achieves state-of-the-art cybersecurity performance among 8B parameter models while remaining competitive with significantly larger models on critical reasoning tasks.

##### 4.2. General-Purpose Benchmarks

Benchmarks. To verify that our cybersecurity specialization does not compromise general intelligence, we evaluate on 10 general-purpose benchmarks spanning diverse capabilities. Table 4 summarizes these benchmarks, which include: AlpacaEval 2.0 for human preference alignment [Dubois et al., 2024]; Big-Bench Hard (BBH) for complex reasoning [Suzgun et al., 2022]; GPQA for graduate-level knowledge [Rein et al., 2024]; GSM8K and MATH for mathematical reasoning [Cobbe et al., 2021, Hendrycks et al., 2021b]; HumanEval for code synthesis [Chen et al., 2021]; IFEval for instruction following [Zhou et al., 2023]; 2WikiMultihopQA and HotpotQA for multi-hop question answering [Ho et al., 2020, Yang et al., 2018]; and MMLU for broad multitask language understanding [Hendrycks et al., 2021a]. These benchmarks provide comprehensive coverage of capabilities essential for general-purpose AI systems.

2The numbers for DeepHat-V1-7B differ from those reported inthe Foundation-Sec-8B-Instruct technical report [Weerawardhena et al., 2025] because this model does not rigorously follow the instruction to output answers in the last line, resulting in many invalid answers. We emphasize that this instruction format is not construed in favor of any specific model—all other evaluated models follow this instruction successfully. Because LLM evaluation is complex, certain formatting requirements are necessary for computing accuracy using regular expression-based answer extractors.

CTIBench-MCQA

CTIBench-RCM

CTI-Reasoning

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

100

100

100

80

80

80

Score(%)

60

60

60

40

40

40

20

20

20

0

0

0

CWE-Prediction

SecBench

SecEval

100

100

100

80

80

80

Score(%)

60

60

60

40

40

40

20

20

20

0

0

0

|Llama-3.1-8B-Instruct<br><br>Llama-3.3-70B-Instruct<br><br>GPT-OSS-120B<br><br>Foundation-Sec-8B-Instruct<br><br>| |
|---|
<br><br>Foundation-Sec-8B-Reasoning|
|---|

- Figure 3 | Comparison of selected models across 6 key cybersecurity benchmarks. The benchmarks are organized in two rows: (top row) CTIBench-MCQA, CTIBench-RCM, and CTI-Reasoning; (bottom row) CWE-Prediction, SecBench-Reasoning, and SecEval. Foundation-Sec-8B-Reasoning demonstrates consistently strong performance across diverse tasks, particularly excelling on CTIBench-RCM (75.3%) and CWE-Prediction (70.4%). The visualization highlights the model’s competitive performance with significantly larger models while maintaining robust capabilities across both knowledgebased and reasoning-intensive benchmarks.

Models. To demonstrate that cybersecurity specialization does not compromise general capabilities, we compare models at similar scales. Our comparison set includes: (1) general-purpose baselines: Llama-

- 3.1-8B-Instruct and Llama-3.3-70B-Instruct; (2) cybersecurity-specialized models based on Llama-3.1-8B: Foundation-Sec-8B-Instruct and Llama-Primus variants; and (3) Phi-4 as an additional comparison point.

Evaluation Protocol. We use the lm-evaluation-harness3 framework [Gao et al., 2024] with fewshot settings as specified in Table 4. For MATH, we compute accuracy using the math-verify 4 package to ensure correct mathematical equivalence checking. All models are evaluated with temperature = 0.3 and top-𝑝 = 1.0, except for Foundation-Sec-8B-Reasoning, which uses temperature = 0.7 and top-𝑝 = 0.95 to allow more diverse reasoning paths.

- 4.2.1. Results and Analysis

Tables 5 and 6 present comprehensive results across all 10 general-purpose benchmarks. Our analysis focuses primarily on comparing Foundation-Sec-8B-Reasoning with Llama-3.1-8B-Instruct (the generalpurpose baseline) and Foundation-Sec-8B-Instruct (our cybersecurity-specialized instruction model) to demonstrate that domain specialization and reasoning training do not compromise general capabilities.

- 3https://github.com/EleutherAI/lm-evaluation-harness.
- 4https://github.com/huggingface/Math-Verify.

###### Benchmark Evaluation Area Few-Shot Metric

AlpacaEval Human preference alignment 0 Length-Controlled Win Rate BBH Complex reasoning 3 Exact Match GPQA Graduate-level knowledge 0 Exact Match GSM8K Grade-school math 8 Exact Match HumanEval Code synthesis 0 Pass@10 Success Rate IFEval Instruction following 0 Instruction-Level Loose Accuracy 2WikiMultihopQA Multi-hop QA (long context) 0 Question Answering F1 Score HotpotQA Multi-hop QA (long context) 0 Question Answering F1 Score MATH Competition mathematics 4 Math-Verify Accuracy MMLU Broad multitask understanding 5 Exact Match

- Table 4 | Overview of general-purpose benchmarks. All metrics range from 0 to 1.

Key Finding: General-Purpose Performance

Foundation-Sec-8B-Reasoning maintains strong general capabilities, demonstrating that cybersecurity specialization does not compromise performance on general-purpose tasks:

- • On par with general-purpose baseline: Achieves comparable or superior performance to Llama-3.1-8B-Instruct across most benchmarks, with notable advantages on AlpacaEval 2.0 (62.6% vs 25.4%, +146% relative improvement) and 2WikiMultihopQA (60.5% vs 49.6%,

+22% relative). Performance is on par for GSM8K (82.3% vs 82.2%) and HumanEval (79.9% vs 82.3%, -2.9%), demonstrating that domain specialization preserves general intelligence.

- • Competitive with cybersecurity-specialized baseline: Matches or exceeds FoundationSec-8B-Instruct on 8 out of 10 benchmarks, with substantial improvements on AlpacaEval 2.0 (62.6% vs 33.1%, +89% relative) and 2WikiMultihopQA (60.5% vs 45.4%, +33% relative). Performance is comparable on MATH (43.3% vs 43.6%) and GSM8K (82.3% vs 84.8%), showing that RL-based reasoning training enhances capabilities without sacrificing foundational skills.

These results confirm that Foundation-Sec-8B-Reasoning successfully combines cybersecurity expertise with strong general-purpose capabilities, making it suitable for diverse real-world applications beyond the security domain.

Instruction Following and Human Alignment. Foundation-Sec-8B-Reasoning demonstrates exceptional performance on instruction-following and human preference alignment benchmarks. On AlpacaEval 2.0, which measures the win rate of model responses against a reference model in head-to-head comparisons, Foundation-Sec-8B-Reasoning achieves 62.6%, substantially outperforming FoundationSec-8B-Instruct (33.1%) and Llama-3.1-8B-Instruct (25.4%). This represents a 146% relative improvement over Llama-3.1-8B-Instruct and an 89% improvement over Foundation-Sec-8B-Instruct, indicating that the extended reasoning training significantly enhances the model’s ability to generate responses preferred by human evaluators.

On IFEval, which assesses strict instruction-following through verifiable format constraints, Foundation-Sec-8B-Reasoning achieves 83.7%, approaching Foundation-Sec-8B-Instruct’s performance (86.1%) and closely matching Llama-3.1-8B-Instruct (86.2%). This demonstrates that the reasoning model maintains robust instruction-following capabilities despite the shift in training objectives toward extended reasoning. The modest 2.4 pp decrease relative to Foundation-Sec-8B-Instruct is more than

compensated by the substantial gains in reasoning and other capabilities.

Reasoning and Knowledge. The model exhibits competitive performance on reasoning and knowledge benchmarks. On BBH (Big-Bench Hard), which evaluates complex reasoning across diverse tasks, Foundation-Sec-8B-Reasoning achieves 69.9%, exceeding both Foundation-Sec-8B-Instruct (66.7%) and Llama-3.1-8B-Instruct (67.4%). This 4.8% relative improvement over Foundation-Sec-8B-Instruct provides empirical evidence that the extended reasoning training enhances general reasoning capabilities beyond the cybersecurity domain, demonstrating that the reinforcement learning phase with reasoning-focused objectives is critical for developing robust reasoning skills.

On GPQA (Graduate-Level Physics, Chemistry, and Biology Questions), all 8B-parameter models show similar performance, with Foundation-Sec-8B-Reasoning achieving 31.7%, comparable to Foundation-Sec-8B-Instruct (31.9%) and exceeding Llama-3.1-8B-Instruct (25.7%). Similarly, on MMLU (Massive Multitask Language Understanding), which evaluates broad knowledge across 57 subjects, Foundation-Sec-8B-Reasoning achieves 68.3%, outperforming Foundation-Sec-8B-Instruct (66.0%) while remaining competitive with Llama-3.1-8B-Instruct (69.8%). These results demonstrate that cybersecurity specialization does not compromise broad knowledge capabilities, with Foundation-Sec-8B-Reasoning maintaining competitive performance on challenging knowledge benchmarks.

Mathematical Reasoning. Foundation-Sec-8B-Reasoning demonstrates strong mathematical capabilities across both grade-school and advanced mathematics benchmarks. On GSM8K (Grade School Math), the model achieves 82.3%, performing on par with Llama-3.1-8B-Instruct (82.2%) and closely matching Foundation-Sec-8B-Instruct (84.8%). On MATH, which contains challenging competition-level problems, Foundation-Sec-8B-Reasoning achieves 43.3%, comparable to Foundation-Sec-8B-Instruct (43.6%) and Llama-3.1-8B-Instruct (47.9%). These results indicate that the reasoning-focused training maintains strong mathematical problem-solving abilities across difficulty levels.

Coding. On HumanEval, which evaluates program synthesis capabilities, Foundation-Sec-8BReasoning achieves 79.9%, a modest 2.9% decrease from Foundation-Sec-8B-Instruct and Llama-3.1-8BInstruct (both at 82.3%). This minor trade-off in pure code generation capabilities is within practical deployment thresholds for cybersecurity applications, where code understanding and reasoning are often more critical than pure code synthesis. The performance remains strong, and substantially exceeds models like Llama-Primus-Merged (81.7%).

Long-Form Question Answering. A particularly noteworthy result emerges on long-form question answering benchmarks. On 2WikiMultihopQA, Foundation-Sec-8B-Reasoning achieves 60.5%, substantially outperforming Llama-3.1-8B-Instruct (49.6%, +22% relative improvement), Foundation-Sec-

- 8B-Instruct (45.4%, +33% relative improvement), and even Phi-4 (28.4%). On HotpotQA, Foundation-Sec8B-Reasoning achieves 54.8%, performing comparably to Llama-3.1-8B-Instruct (54.1%) and approaching Foundation-Sec-8B-Instruct (58.4%). These benchmarks require multi-hop reasoning over retrieved documents, a capability highly relevant to cybersecurity analysis where practitioners must synthesize information from multiple sources. The substantial improvement on 2WikiMultihopQA in particular suggests that the extended reasoning training develops capabilities well-suited to complex, multi-step analysis tasks.

Comparative Analysis. Figure 4 provides a visual comparison across key benchmarks. The results reveal several important patterns. First, Foundation-Sec-8B-Reasoning achieves the highest performance

among all 8B-parameter models on AlpacaEval 2 and 2WikiMultihopQA, demonstrating clear advantages in human preference alignment and multi-hop reasoning. Second, the model maintains competitive or superior performance to Foundation-Sec-8B-Instruct across most benchmarks, indicating successful capability retention despite the shift toward extended reasoning. The comparison across models reveals that the reasoning-focused reinforcement learning training is critical for developing strong reasoning and alignment capabilities, as evidenced by substantial improvements over the base instruction model on AlpacaEval 2 (33.1% → 62.6%, +89% relative) and BBH (66.7% → 69.9%, +4.8 pp). Overall, Foundation-Sec-8B-Reasoning achieves a balanced performance profile suitable for cybersecurity applications requiring both technical skills and extended analytical reasoning.

Implications for Cybersecurity Applications. These results have important implications for deploying Foundation-Sec-8B-Reasoning in cybersecurity contexts. The model’s exceptional performance on instruction-following and long-form reasoning benchmarks suggests strong suitability for tasks requiring extended analysis, such as threat intelligence report generation, incident investigation, and vulnerability analysis. The maintained performance on mathematical reasoning and modest decrease in pure coding suggests that the model can effectively support a wide range of cybersecurity workflows while excelling at the analytical reasoning tasks most critical to senior practitioners. The substantially higher AlpacaEval 2 scores indicate that the model’s outputs are likely to be perceived as more helpful and higher quality by human evaluators, an important consideration for practical deployment.

Model AlpacaEval 2 BBH GPQA GSM8K MMLU

Llama-3.1-8B-Instruct 0.254±0.008 0.674±0.005 0.257±0.021 0.822±0.011 0.698±0.000 Llama-3.3-70B-Instruct 0.395±0.009 0.907±0.003 0.565±0.023 0.929±0.007 0.857±0.001 Llama-Primus-Merged 0.174±0.007 0.710±0.005 0.292±0.022 0.807±0.011 0.682±0.002 Phi-4 0.481±0.008 0.876±0.004 0.509±0.024 0.894±0.008 0.849±0.001 Foundation-Sec-8B-Instruct 0.331±0.008 0.667±0.005 0.319±0.022 0.848±0.010 0.660±0.003 Foundation-Sec-8B-Reasoning 0.626±0.005 0.699±0.005 0.317±0.022 0.823±0.011 0.683±0.000

- Table 5 | Performance on general-purpose benchmarks (Part 1): Instruction following, reasoning, and knowledge. Results show mean ± standard deviation. AlpacaEval 2 scores represent length-controlled win rates against reference models. Bold values indicate the best performance for each benchmark.

Model HumanEval IFEval 2WikiMulihopQA HotpotQA MATH

Llama-3.1-8B-Instruct 0.823±0.030 0.862±0.018 0.496±0.032 0.541±0.031 0.479±0.007 Llama-3.3-70B-Instruct 0.933±0.020 0.940±0.013 0.654±0.031 0.596±0.031 0.739±0.006 Llama-Primus-Merged 0.817±0.030 0.805±0.020 0.439±0.032 0.483±0.031 0.398±0.007 Phi-4 0.945±0.018 0.778±0.021 0.284±0.021 0.347±0.026 0.801±0.005 Foundation-Sec-8B-Instruct 0.823±0.030 0.861±0.018 0.454±0.032 0.584±0.031 0.436±0.007 Foundation-Sec-8B-Reasoning 0.799±0.031 0.837±0.020 0.605±0.032 0.548±0.032 0.433±0.007

- Table 6 | Performance on general-purpose benchmarks (Part 2): Coding, instruction following, and mathematical reasoning. 2WikiMultihopQA and HotpotQA are long-context question answering benchmarks. Results show mean ± standard deviation. Bold values indicate the best performance for each benchmark.

AlpacaEval 2

Big-Bench Hard

IFEval

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

100

100

100

80

80

80

Score(%)

60

60

60

40

40

40

20

20

20

0

0

0

GSM8K

HumanEval

MATH

| | | |
|---|---|---|
| | | |
| | | |
| | | |
| | | |
| | | |

100

100

100

80

80

80

Score(%)

60

60

60

40

40

40

20

20

20

0

0

0

|Llama-3.1-8B-Instruct Llama-3.3-70B-Instruct Foundation-Sec-8B-Instruct<br><br>| |
|---|
<br><br>Foundation-Sec-8B-Reasoning|
|---|

- Figure 4 | Comparison of model performance across 6 key general-purpose benchmarks. The benchmarks are organized in two rows: (top row) AlpacaEval 2 (human preference), BBH (reasoning), and IFEval (instruction following); (bottom row) GSM8K (grade school math), HumanEval (coding), and MATH (competition mathematics). Error bars represent one standard deviation. Foundation-Sec-8B-Reasoning (highlighted with hatched pattern) demonstrates strong performance across diverse capabilities, with exceptional results on AlpacaEval 2 (62.6%) and competitive performance on reasoning and coding tasks.

##### 4.3. HarmBench Evaluation

Both Foundation-Sec-8B-Instruct and Foundation-Sec-8B-Reasoning have not undergone dedicated safety alignment procedures beyond basic instruction-tuning. However, we followed standard practices to provide a baseline level of alignment. To better understand the models’ risk profiles, we conducted a comprehensive evaluation using HarmBench [Mazeika et al., 2024], a standardized framework for automated red teaming of large language models. The benchmark consists of 400 representative adversarial prompts spanning multiple risk categories including hate speech, harassment, illegal activities, malware generation, physical harm, fraud, sexual content, privacy violations, and self-harm.

Models Evaluated. We evaluate the following model configurations: (1) Llama-3.1-8B-Instruct as a general-purpose baseline, (2) Foundation-Sec-8B-Instruct with its default system prompt, (3) FoundationSec-8B-Reasoning without system prompt, (4) Foundation-Sec-8B-Reasoning with system prompt (adapted from Foundation-Sec-8B-Instruct), and (5) Foundation-Sec-8B-Reasoning with system prompt protected by Llama-Guard-3-8B [Inan et al., 2023]. The system prompt for Foundation-Sec-8B-Reasoning is modified from that of Foundation-Sec-8B-Instruct to better accommodate the reasoning model’s extended analytical capabilities; see Appendix A.3 for details. Responses are classified as either refusing the harmful request (passing) or complying with it (failing), with the pass rate representing the percentage of harmful prompts successfully refused or safely handled.

Results and Analysis. Figure 5 presents the evaluation results. Foundation-Sec-8B-Instruct demonstrates strong baseline safety performance, achieving 95.00% pass rate, substantially outperforming Llama-3.1-8B-Instruct (62.75%). For Foundation-Sec-8B-Reasoning, we report results both with and without system prompts to understand the impact of safety guidance. Without a system prompt, the model achieves 54.25% pass rate. However, when equipped with an appropriate system prompt adapted from Foundation-Sec-8B-Instruct, Foundation-Sec-8B-Reasoning achieves 93.00% pass rate, approaching the safety performance of Foundation-Sec-8B-Instruct. This demonstrates that with proper system-level safety instructions, the reasoning model maintains strong safety performance while preserving its enhanced analytical capabilities.

We strongly recommend applying additional safety layers, such as automated content filtering or LLM-based moderation systems, when deploying these models. The system prompt of FoundationSec-8B-Reasoning is derived from Foundation-Sec-8B-Instruct and can be adapted to specific use cases. See Appendix A.3 for details.

Enhanced Protection with Llama-Guard-3-8B. To explore additional safety measures for production deployments, we evaluated Foundation-Sec-8B-Reasoning (with system prompt) combined with Llama-Guard-3-8B, a taxonomy-driven input-output filtering system that analyzes both user inputs and model outputs, flagging and blocking content that violates defined safety policies. The combined system achieves 98.25% pass rate on HarmBench, representing near-complete protection against adversarial prompts. This result demonstrates the effectiveness of a defense-in-depth approach where model-level safety is complemented by external guardrails. In summary, Foundation-Sec-8B-Reasoning with proper system prompts achieves strong safety performance (93.00%), and when further protected by Llama-Guard-3-8B, delivers exceptional safety (98.25%).

##### 4.4. Discussion

To better understand the individual contributions of supervised fine-tuning (SFT) and reinforcement learning (RL) to our model’s capabilities, we conduct an ablation study comparing the SFT checkpoint (Foundation-Sec-8B-Reasoning-SFT-Checkpoint) with the final RL-trained model (Foundation-Sec-8BReasoning). This analysis reveals the specific benefits of each training stage and provides insights into how extended reasoning capabilities emerge through the RL phase.

##### 4.4.1. Role of Supervised Fine-Tuning

The SFT checkpoint represents the model state after supervised fine-tuning on a diverse mixture of cybersecurity and general-purpose data, but before any reinforcement learning with reasoning-focused objectives. Evaluating this checkpoint allows us to isolate the contributions of the SFT phase and understand what capabilities are already present before RL training begins.

As shown in Table 7, Foundation-Sec-8B-Reasoning-SFT-Checkpoint already demonstrates strong baseline performance across many benchmarks. On cybersecurity tasks, the SFT checkpoint achieves 68.4% on CTIBench-MCQA, 69.5% on CTIBench-RCM, and 85.3% on CTIBench-VSP, indicating that the supervised fine-tuning phase successfully imparts core cybersecurity knowledge and instructionfollowing capabilities. On general-purpose benchmarks, the SFT checkpoint shows competitive performance on tasks like HumanEval (82.3%) and GSM8K (75.5%), demonstrating that the SFT phase maintains broad capabilities while incorporating cybersecurity expertise.

However, the SFT checkpoint also reveals notable limitations on certain benchmarks. On multi-hop

98.25%

95.00%

100

93.00%

80

62.75%

Accuracy(%)

60

54.25%

40

20

0

|Llama-3.1-8B-Instruct<br><br>Foundation-Sec-8B-Instruct Foundation-Sec-8B-Reasoning (no system prompt)<br><br>| |
|---|
<br><br>| |
|---|
<br><br>Foundation-Sec-8B-Reasoning (with system prompt) LlamaGuard + Foundation-Sec-8B-Reasoning (with system prompt)<br><br>| |
|---|
|
|---|

- Figure 5 | HarmBench safety evaluation results showing pass rates (percentage of harmful prompts appropriately refused) across different model configurations. Foundation-Sec-8B-Instruct achieves 95.00% pass rate, while Foundation-Sec-8B-Reasoning achieves 93.00% with proper system prompts. When further protected by Llama-Guard-3-8B, Foundation-Sec-8B-Reasoning achieves 98.25% pass rate, demonstrating that our reasoning model with appropriate safety measures delivers strong protection against adversarial prompts.

question answering benchmarks, the model achieves only 24.4% on 2WikiMultihopQA and 9.6% on HotpotQA. Similarly, on BBH, which requires complex reasoning across diverse tasks, the SFT checkpoint achieves 56.3%. These low scores appear to stem primarily from insufficient instructionfollowing ability rather than a lack of knowledge or reasoning capacity. As we will demonstrate in the subsequent RL analysis, instruction-following improvements lead to substantial gains on these tasks, even though our RL training data does not directly target these specific benchmarks.

These results demonstrate that while SFT successfully establishes domain knowledge, it requires further training to develop robust instruction-following and reasoning capabilities for complex analytical tasks. This motivates the subsequent RL training phase with reasoning-focused objectives.

##### 4.4.2. Benefits of Reinforcement Learning

- Table 7 presents a comprehensive comparison between the SFT checkpoint and the final RL-trained model across both cybersecurity and general-purpose benchmarks. The results reveal several key patterns that illuminate the specific contributions of RL training.

Substantial Improvements on Reasoning-Intensive Cybersecurity Tasks. RL training demonstrates its most significant impact on cybersecurity tasks that require extended analytical reasoning. On CTIBench-RCM, the RL model achieves 75.3%, a substantial +5.8 pp improvement over the SFT checkpoint. On CTIBench-ATE, the improvement is even more pronounced at +9.7 pp (39.4% → 49.1%). On CWE-Prediction, the RL model achieves 70.4%, representing a +6.0 pp gain. These improvements on reasoning-intensive cybersecurity tasks demonstrate that the RL training phase successfully develops capabilities for complex analytical tasks that go beyond pattern matching and require deeper understanding.

Direct and Indirect Benefits from RL Training. Our RL training data comprises cybersecurity, instruction-following, and mathematical reasoning examples. We observe direct improvements on benchmarks related to these data types: cybersecurity tasks show gains of +5.8 pp to +9.7 pp on CTIBench-RCM and CTIBench-ATE; instruction-following improves by +12.7 pp on IFEval (83.7%); and mathematical reasoning advances by +10.2 pp on MATH (33.1% → 43.3%).

Beyond these direct improvements, we observe substantial indirect benefits on benchmarks not explicitly covered in our RL training data. Multi-hop question answering tasks show dramatic gains: +36.1 pp on 2WikiMultihopQA (60.5%) and +45.1 pp on HotpotQA (9.6% → 54.8%). Similarly, BBH improves by +13.7 pp (69.9%) and AlpacaEval 2 by +6.3 pp (56.3% → 62.6%). These indirect improvements demonstrate that enhanced instruction-following and reasoning capabilities from RL training generalize to diverse tasks requiring multi-step analytical thinking. Critically, other benchmarks show minimal degradation due to KL divergence regularization and the relatively small scale of RL training data, which prevents the model from deviating excessively from its SFT initialization.

Maintained Performance with Minor Trade-offs. While RL training produces substantial improvements on reasoning-intensive tasks, it maintains competitive performance on most other benchmarks with only minor trade-offs. On knowledge-based benchmarks like MMLU-Security, the difference is small (-2.4 pp), and on SecEval, the RL model actually improves slightly (+0.5 pp).

The most notable trade-off appears on HumanEval, where the RL model achieves 79.9% compared to the SFT checkpoint’s 82.3% (-2.4 pp). However, this modest decrease in pure code synthesis capability is more than compensated by the substantial gains in reasoning and analytical capabilities. For cybersecurity applications, where code understanding and reasoning about vulnerabilities are often more critical than pure code generation, this trade-off is highly favorable.

Implications for Model Development. These results have important implications for developing reasoning-capable models for specialized domains. First, they demonstrate that SFT alone, while effective for establishing domain knowledge and basic instruction-following, is insufficient for developing robust extended reasoning capabilities. Second, they show that RL training with reasoning-focused objectives can successfully develop strong analytical reasoning skills that transfer well to complex domain-specific tasks. Third, they reveal that this reasoning capability development comes with only minor trade-offs in other capabilities, making it a highly effective approach for enhancing model performance on analytical tasks.

The substantial improvements on multi-hop reasoning benchmarks (+36.1 pp on 2WikiMultihopQA, +45.1 pp on HotpotQA) are particularly noteworthy, as they suggest that RL training develops general reasoning capabilities that extend far beyond the specific training tasks. This generalization is crucial for real-world cybersecurity applications, where practitioners encounter novel and complex scenarios that require flexible analytical reasoning rather than pattern matching on familiar examples.

Benchmark SFT Final Model (RL) Difference Cybersecurity Benchmarks

CTIBench-MCQA 68.4% 69.1% +0.7 pp CTIBench-RCM 69.5% 75.3% +5.8 pp CTIBench-VSP 85.3% 85.6% +0.3 pp CTIBench-ATE 39.4% 49.1% +9.7 pp CTI-Reasoning 44.0% 41.1% -2.9 pp CWE-Prediction 64.4% 70.4% +6.0 pp MMLU-Security 80.6% 78.2% -2.4 pp CyberMetric-2000 84.7% 84.3% -0.4 pp SecBench 72.6% 72.5% -0.0 pp SecEval 84.3% 84.8% +0.5 pp

###### General-Purpose Benchmarks

AlpacaEval 2.0 56.3% 62.6% +6.3 pp BBH 56.3% 69.9% +13.7 pp GPQA 28.8% 31.7% +2.9 pp GSM8K 75.5% 82.3% +6.8 pp HumanEval 82.3% 79.9% -2.4 pp IFEval 71.0% 83.7% +12.7 pp 2WikiMultihopQA 24.4% 60.5% +36.1 pp HotpotQA 9.6% 54.8% +45.1 pp MATH 33.1% 43.3% +10.2 pp

Table 7 | Comparison between SFT checkpoint (Foundation-Sec-8B-Reasoning-SFT-Checkpoint) and final RL-trained model (Foundation-Sec-8B-Reasoning). The table shows performance improvements (green) and decreases (red) in percentage points (pp). Reinforcement learning demonstrates substantial improvements on reasoning-intensive tasks including CTIBench-RCM (+5.8 pp), CTIBench-ATE (+9.7 pp), 2WikiMultihopQA (+36.1 pp), and HotpotQA (+45.1 pp), while maintaining competitive performance on most other benchmarks.

#### 5. Conclusion

We present Foundation-Sec-8B-Reasoning, the first open-source native reasoning model specifically designed for cybersecurity applications. Built upon our previously released Foundation-Sec-8B base model (derived from Llama-3.1-8B-Base), this work demonstrates that domain-specialized reasoning models can achieve performance competitive with significantly larger general-purpose models while maintaining strong general capabilities.

Our two-stage training methodology combines supervised fine-tuning with reinforcement learning from verifiable rewards (RLVR) using proprietary reasoning data spanning cybersecurity analysis, instruction-following, and mathematical reasoning. Critically, we employ KL divergence regularization and maintain relatively small-scale RL training to prevent excessive deviation from the SFT initialization, ensuring stability across diverse benchmarks.

Evaluation demonstrates strong performance on cybersecurity tasks (e.g., 75.3% on CTIBenchRCM, outperforming 15× larger GPT-OSS-120B) while maintaining general capabilities (e.g., 62.6% on AlpacaEval 2.0). Our ablation study reveals that RL training yields direct improvements on data-related benchmarks and substantial indirect benefits on multi-hop reasoning (+36.1 pp on 2WikiMultihopQA, +45.1 pp on HotpotQA), demonstrating effective capability generalization. Safety evaluation shows 93.00% pass rate on HarmBench with system prompts, and 98.25% with Llama-Guard-3-8B protection.

To our knowledge, Foundation-Sec-8B-Reasoning represents the first open-source native reasoning model for cybersecurity—trained directly as a reasoning model rather than adapted from instructionfollowing models. This work demonstrates the viability of developing domain-specialized reasoning models that deliver strong performance on specialized tasks while maintaining broad general capabilities, opening new directions for AI systems tailored to complex analytical domains beyond cybersecurity.

#### Acknowledgements

We thank Karen Kui, Hadas Birin, Ron Kupfer, Howard Lin, Kimia Majd, Mayank Rajoria, Nathan Chang, Roee Landesman, Takahiro Matsumoto, Yasukazu Hirata, Amos Yoffe, Hyrum Anderson, and Konstantin Goldin for their invaluable support.

We thank David Bianco and the SURGe team for their continuous feedback on the model’s performance throughout the development cycle.

We also thank Omar Santos and the Cisco CSIRT team for being invaluable partners in the development and adoption of the model in real-world use cases.

#### References

Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauffmann, et al. Phi-4 technical report. arXiv preprint arXiv:2412.08905, 2024.

Marah Abdin, Sahaj Agarwal, Ahmed Awadallah, Vidhisha Balachandran, Harkirat Behl, Lingjiao Chen, Gustavo de Rosa, Suriya Gunasekar, Mojan Javaheripi, Neel Joshi, et al. Phi-4-reasoning technical report. arXiv preprint arXiv:2504.21318, 2025.

Md Tanvirul Alam, Dipkamal Bhusal, Le Nguyen, and Nidhi Rastogi. CTIBench: A benchmark for evaluating LLMs in cyber threat intelligence. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://openreview.net/forum ?id=iJAOpsXo2I.

Akhiad Bercovich, Itay Levy, Izik Golan, Mohammad Dabbah, Ran El-Yaniv, Omri Puny, Ido Galil, Zach Moshe, Tomer Ronen, Najeeb Nabwani, et al. Llama-nemotron: Efficient reasoning models. arXiv preprint arXiv:2505.00949, 2025.

CAPEC. Common attack pattern enumerations and classifications (capec). https://capec.mitre.org/,

2024. Available at https://capec.mitre.org/.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Steve Christey, J. Kenderdine, J. Mazella, and B. Miles. Common weakness enumeration. Technical report, The MITRE Corporation, 2013. URL https://cwe.mitre.org/.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

DeepHat. AI for DevSecOps, DeepHat. https://www.deephat.ai/, 2025. Accessed: 2025-7-29. Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B Hashimoto. Length-controlled alpacaeval:

A simple way to debias automatic evaluators. arXiv preprint arXiv:2404.04475, 2024.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 07 2024. URL https://zenodo.org/records/12608602.

GitHub. Github security advisories. https://github.com/advisories, 2024. Available at https: //github.com/advisories.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The Llama-3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-R1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021a. URL https://openreview.net/forum?id=d7KBjmI3GmQ.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021b.

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop QA dataset for comprehensive evaluation of reasoning steps. arXiv preprint arXiv:2011.01060, 2020.

Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Kai Dang, et al. Qwen2. 5-Coder technical report. arXiv preprint arXiv:2409.12186, 2024.

Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, et al. Llama guard: Llm-based input-output safeguard for human-ai conversations. arXiv preprint arXiv:2312.06674, 2023.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7B, 2023. URL https://arxiv.org/abs/2310.06825.

Pengfei Jing, Mengyun Tang, Xiaorong Shi, Xing Zheng, Sen Nie, Shi Wu, Yong Yang, and Xiapu Luo. Secbench: A comprehensive multi-dimensional benchmarking dataset for llms in cybersecurity. arXiv preprint arXiv:2412.20787, 2024.

Paul Kassianik, Baturay Saglam, Alexander Chen, Blaine Nelson, Anu Vellore, Massimo Aufiero, Fraser Burch, Dhruv Kedia, Avi Zohary, Sajana Weerawardhena, et al. Llama-3.1-Foundationai-SecurityLLMBase-8B technical report. arXiv preprint arXiv:2504.21039, 2025.

Komal Kumar, Tajamul Ashraf, Omkar Thawakar, Rao Muhammad Anwer, Hisham Cholakkal, Mubarak Shah, Ming-Hsuan Yang, Phillip HS Torr, Fahad Shahbaz Khan, and Salman Khan. Llm post-training: A deep dive into reasoning large language models. arXiv preprint arXiv:2502.21321, 2025.

Guancheng Li, Yifeng Li, Wang Guannan, Haoyu Yang, and Yang Yu. Seceval: A comprehensive benchmark for evaluating cybersecurity knowledge of foundation models. https://github.com/XuanwuAI/SecEval, 2023.

Zhong-Zhi Li, Duzhen Zhang, Ming-Liang Zhang, Jiaxin Zhang, Zengyan Liu, Yuxuan Yao, Haotian Xu, Junhao Zheng, Pei-Jie Wang, Xiuyi Chen, et al. From system 1 to system 2: A survey of reasoning large language models. arXiv preprint arXiv:2502.17419, 2025.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding R1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, et al. Harmbench: A standardized evaluation framework for automated red teaming and robust refusal. arXiv preprint arXiv:2402.04249, 2024.

OpenAI. gpt-oss-120b & gpt-oss-20b model card, 2025. URL https://arxiv.org/abs/2508.10925. David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani,

Julian Michael, and Samuel R Bowman. GPQA: A graduate-level google-proof Q&A benchmark. In First Conference on Language Modeling, 2024.

segolilylabs. Lily-cybersecurity-7b-v0.2: A cybersecurity assistant. Hugging Face model repository, 2025. URL https://huggingface.co/segolilylabs/Lily-Cybersecurity-7B-v0.2. Fine-tuned on Mistral-7B-Instruct-v0.2 with 22,000 hand-crafted cybersecurity/hacking pairs. 7.24B parameters, trained over 5 epochs on an A100 GPU. Apache 2.0 license.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Blake E. Strom, Andy Applebaum, Doug P. Miller, Kathryn C. Nickels, Adam G. Pennington, and Cody B. Thomas. MITRE ATT&CK: Design and Philosophy. Technical report, The MITRE Corporation, 2018. URL https://attack.mitre.org/resources/enterprise-introduction/.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261, 2022.

The MITRE Corporation. Cve® – common vulnerabilities and exposures program. https://cve.mitr e.org/, 2025.

Norbert Tihanyi, Mohamed Amine Ferrag, Ridhi Jain, Tamas Bisztray, and Merouane Debbah. Cybermetric: A benchmark dataset based on retrieval-augmented generation for evaluating llms in cybersecurity knowledge. In 2024 IEEE International Conference on Cyber Security and Resilience (CSR), pages 296–302, 2024. doi: 10.1109/CSR61664.2024.10679494.

Sajana Weerawardhena, Paul Kassianik, Blaine Nelson, Baturay Saglam, Anu Vellore, Aman Priyanshu, Supriti Vijay, Massimo Aufiero, Arthur Goldblatt, Fraser Burch, et al. Llama-3.1-FoundationaiSecurityLLM-B-Instruct technical report. arXiv preprint arXiv:2508.01059, 2025.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. arXiv preprint arXiv:1809.09600, 2018.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. DAPO: An open-source LLM reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025a.

Yao-Ching Yu, Tsun-Han Chiang, Cheng-Wei Tsai, Chien-Ming Huang, and Wen-Kwang Tsao. Primus: A pioneering collection of open-source datasets for cybersecurity llm training. arXiv preprint arXiv:2502.11191, 2025b.

Jie Zhang, Haoyu Bu, Hui Wen, Yongji Liu, Haiqiang Fei, Rongrong Xi, Lun Li, Yun Yang, Hongsong Zhu, and Dan Meng. When llms meet cybersecurity: A systematic literature review, 2024. URL https://arxiv.org/abs/2405.03644.

Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.

#### A. Appendix

This appendix provides comprehensive details on all benchmarks used in our evaluation suite, including our two proprietary benchmarks and the complete prompting strategies for all evaluation tasks. Furthemore, in Appendix A.3, we provide the system prompt used for the HarmBench safety evaluation of Foundation-Sec-8B-Reasoning.

##### A.1. Proprietary Benchmarks: CTI-Reasoning and CWE Prediction

CTI-Reasoning. The CTI-Reasoning benchmark is a proprietary evaluation dataset designed to assess deep cybersecurity reasoning capabilities beyond simple knowledge recall. Unlike standard multiplechoice questions that test factual knowledge, CTI-Reasoning evaluates a model’s ability to perform multi-hop logical analysis, understand complex technical documentation, and reason about relationships within cybersecurity taxonomies. The benchmark contains 200 test samples of expert-curated multiplechoice questions derived from CWE [Christey et al., 2013] and Common Attack Pattern Enumeration and Classification (CAPEC) [CAPEC, 2024] documentation. Notably, 96% of questions are classified as reasoning-intensive, requiring analytical thinking rather than memorization, with 77.5% demanding analysis-based cognitive processing and 22.5% requiring comprehension-based reasoning.

CTI-Reasoning Example

Question: Consider the following scenario for CAPEC-625. A company employs strong physical security for their mobile devices containing cryptographic secrets. Despite this, a successful MDFI attack occurs. What could be a contributing factor?

##### Options:

- A. Failures in electromagnetic protection allow signal-based attacks without leaving physical traces.
- B. Physical security can be bypassed, implying core MDFI techniques don’t require device access.
- C. Attack depends entirely on security failures unrelated to physical device access.
- D. Strong physical security inherently allows MDFI attacks without further protection.

##### Correct Answer: A

CWEPrediction. TheCWEPredictionbenchmarkisconstructedin exactly the same way as CTIBenchRCM (vulnerability description to CWE mapping), but uses recent entries from 2025 for CVE [The MITRE Corporation, 2025] and 2024-2025 for GitHub Security Advisories (GHSA) [GitHub, 2024] to ensure the data is new to the model. This design allows us to test the same vulnerability classification ability as CTIBench-RCM while evaluating generalization to previously unseen vulnerability descriptions. The benchmark contains 3,000 test samples covering 263 distinct CWE types from these recent and diverse sources.

CWE Prediction Example

Question: What Common Weakness Enumeration (CWE) identifier is associated with CVE-2025-26949, which describes an ’Improper Neutralization of Input During Web Page Generation (Cross-site Scripting)’ vulnerability in bPlugins Team Section Block that allows Stored XSS?

Final Answer: CWE-79

- A.2. Prompts for Cybersecurity Benchmark Evaluation

In all cybersecurity benchmark evaluations, we require models to output their final answer in the last line of their response following a specified format. We extract this last line using regular expressions to compute accuracy automatically. This standardized format ensures consistent evaluation across all models and benchmarks.

For CTIBench-VSP (CVSS Vector String Prediction), we compute the evaluation score based on the CVSS (Common Vulnerability Scoring System) score difference. The CVSS score is a numerical value ranging from 0 to 10 that represents the severity of a vulnerability, derived from the CVSS vector string. Our evaluation metric is defined as: Score = 1 −|CVSSpred − CVSStrue|/10, where CVSSpred is the score computed from the model’s predicted vector string and CVSStrue is the ground truth score.

CTIBench-MCQA Task Description:

Multiple-choice questions covering cybersecurity threat intelligence concepts, including vulnerability analysis, attack patterns, and security mechanisms sourced from the CTIBench dataset.

##### Prompt:

Given the following question and four candidate answers (A, B, C, and D), choose the best answer. The last line of your response should be in the following format: 'Answer: $LETTER' (without quotes) where $LETTER is one of A, B, C, or D.

CTIBench-RCM (Root Cause Mapping) Task Description:

Root cause mapping task that requires mapping Common Vulnerabilities and Exposures (CVE) descriptions to their corresponding Common Weakness Enumeration (CWE) identifiers, identifying the underlying weakness type of each vulnerability.

##### Prompt:

Analyze the following CVE description and map it to the most appropriate CWE. Provide a brief justification for your choice. Ensure the last line of your response contains only the CWE ID which should be of format `CWE ID: CWE-$id`.

CTIBench-VSP (CVSS Vector String Prediction) Task Description:

Vulnerability severity assessment task that requires predicting the complete Common Vulnerability Scoring System (CVSS) v3.1 vector string for a given CVE description, including all eight base metrics (Attack Vector, Attack Complexity, Privileges Required, User Interaction, Scope, Confidentiality, Integrity, and Availability).

##### Prompt:

From the following CVE description, determine the CVSS v3.1 vector string for each CVSS base metric: AV, AC, PR, UI, S, C, I, and A.

Valid options for each metric are as follows:

- - Attack Vector (AV): Network (N), Adjacent (A), Local (L), Physical (P)
- - Attack Complexity (AC): Low (L), High (H)
- - Privileges Required (PR): None (N), Low (L), High (H)
- - User Interaction (UI): None (N), Required (R)
- - Scope (S): Unchanged (U), Changed (C)
- - Confidentiality (C): None (N), Low (L), High (H)
- - Integrity (I): None (N), Low (L), High (H)
- - Availability (A): None (N), Low (L), High (H)

Provide your answer as a CVSS v3.1 vector string in format: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H

CTIBench-ATE (Attack Technique Extraction) Task Description:

Threat intelligence extraction task that requires identifying and mapping adversary tactics, techniques, and procedures (TTPs) from threat reports to MITRE ATT&CK (Adversarial Tactics, Techniques, and Common Knowledge) framework technique identifiers.

##### Prompt:

Extract all MITRE {{ platform }} attack patterns from the following text and map them to their corresponding MITRE technique IDs. Provide reasoning for each identification.

Important: Your response MUST end with a line in this exact format: Answer: T1234, T5678, T9012

where you list only the main technique IDs (excluding subtechniques), separated by commas. This final line is mandatory.

[Full list of MITRE technique IDs provided as reference]

CTI-Reasoning (Proprietary) Task Description:

Deep cybersecurity reasoning benchmark requiring multi-hop logical analysis and comprehension of complex relationships within MITRE CWE and CAPEC documentation, with 96% of questions designed to test analytical reasoning rather than factual recall.

##### Prompt:

Given the following question and four candidate answers (A, B, C, and D), choose the best answer. The last line of your response should be in the following format: 'Answer: $LETTER' (without quotes) where $LETTER is one of A, B, C, or D.

CWE Prediction (Proprietary) Task Description:

Vulnerability classification task requiring mapping of real-world vulnerability descriptions from CVEs and GitHub Security Advisories to their corresponding CWE identifiers among 263 possible weakness types.

##### Prompt:

Analyze the following vulnerability description and map it to the most appropriate CWE. Provide a brief justification for your choice. Ensure the last line of your response contains only the CWE ID which should be of format `CWE ID: CWE-$id`.

MMLU-Security Task Description:

Computer security knowledge assessment using the security subset of the Massive Multitask Language Understanding (MMLU) benchmark, covering topics such as cryptography, network security, software security, and security principles.

##### Prompt:

Given the following question and four candidate answers (A, B, C, and D), choose the best answer. The last line of your response should be in the following format: 'Answer: $LETTER' (without quotes) where $LETTER is one of A, B, C, or D.

CyberMetric-2000 Task Description:

Comprehensive cybersecurity knowledge benchmark containing 2,000 multiple-choice questions spanning various security domains including network security, cryptography, web security, system security, and security operations.

##### Prompt:

Given the following question and four candidate answers (A, B, C, and D), choose the best answer. The last line of your response should be in the following format: 'Answer: $LETTER' (without quotes) where $LETTER is one of A, B, C, or D.

SecBench Task Description:

Security reasoning benchmark focusing on questions that require understanding of security concepts, threat modeling, risk analysis, and security decision-making across various cybersecurity contexts.

##### Prompt:

Given the following question and four candidate answers (A, B, C, and D), choose the best answer. The last line of your response should be in the following format: 'Answer: $LETTER' (without quotes) where $LETTER is one of A, B, C, or D.

SecEval Task Description:

Comprehensive security evaluation benchmark covering a broad range of cybersecurity topics including offensive security, defensive security, cryptography, and security engineering principles. Dataset filtered to single-answer questions only.

##### Prompt:

Given the following question and four candidate answers (A, B, C, and D), choose the best answer. The last line of your response should be in the following format: 'Answer: $LETTER' (without quotes) where $LETTER is one of A, B, C, or D.

##### A.3. System Prompts

The system prompt plays a crucial role in guiding model behavior and establishing the context for interactions. For Foundation-Sec-8B-Reasoning, we developed a specialized system prompt that emphasizes the model’s cybersecurity expertise, reasoning capabilities, and responsible use guidelines. This prompt is integrated into the model’s chat template and provides users with clear expectations about the model’s capabilities and limitations.

The system prompt is particularly important for reasoning models, as it helps maintain focus on

analytical reasoning while ensuring adherence to safety guidelines. During our HarmBench evaluation (Section 4.3), we observed that the presence of an appropriate system prompt substantially improves safety performance, increasing the pass rate from 54.25% (no system prompt) to 93.00% (with system prompt).

##### Foundation-Sec-8B-Reasoning System Prompt

You are Metis, a cybersecurity reasoning model from the Minerva family developed by Foundation AI at Cisco. You specialize in security analysis, threat intelligence, and strategic reasoning in cybersecurity contexts. You were released in November 2025.

The user is a cybersecurity professional trying to accomplish some cybersecurity task. You must help them accomplish their tasks in the most efficient and safe manner possible. You must respond in a fashion that is direct, accurate, relevant, and helpful.

You have professional knowledge and experience of a senior-level cybersecurity specialist. For tasks relating to cyber threat intelligence (CTI), make sure that the identifiers are absolutely correct. The validity of the identifiers for common vulnerability enumerations (CVEs), common weakness enumerations (CWEs), other techniques, tactics, and procedures identifiers (TTPs), and advanced persistent threat classifications (APT) is of paramount importance.

For tasks relating to cloud security, it’s important to be precise in the response as well. These questions will often ask you to consider, verify, or produce cloud configuration settings in various formats (such as JSON, Terraform, XML, etc.). Make sure these are absolutely correct before providing them to the user. Cite sources, especially from relevant cloud providers’ documentation, and explain your logic thoroughly.

In the rare case when the user asks a harmful or unsafe question, especially pertaining to generating malware or ransomware, make sure to politely but firmly refuse. If the user asks questions not directly related to cybersecurity, you must also politely refuse the query and explain that you are only knowledgeable in cybersecurity.

Key Design Principles. The system prompt is designed around several key principles:

- • Clear Identity: Establishes the model as “Metis,” a specialized cybersecurity reasoning model, setting appropriate user expectations.
- • Domain Focus: Emphasizes cybersecurity expertise and the importance of accuracy in securitycritical information such as CVE, CWE, TTP, and APT identifiers.
- • Professional Context: Positions the model as a tool for cybersecurity professionals, encouraging direct and efficient interactions.
- • Precision Requirements: Highlights the need for accuracy in cloud security configurations and other technical details, with explicit instructions to cite sources.
- • Safety Guidelines: Clearly defines boundaries for harmful requests and off-topic queries, establishing responsible use patterns.

Customization for Deployment. The system prompt can be modified or overridden to suit specific deployment contexts. Organizations may wish to customize the prompt to:

- • Reflect internal security policies and procedures
- • Emphasize specific areas of cybersecurity relevant to their operations

- • Integrate with existing security workflows and tooling
- • Add organization-specific safety guidelines or usage restrictions

We recommend maintaining the core safety guidelines and domain focus when customizing the prompt, as these elements contribute significantly to the model’s effective and responsible operation.

