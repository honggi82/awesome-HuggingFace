### arXiv:2510.24654v2[cs.CL]10Feb2026

###### Evolving Interactive Diagnostic Agents in a Virtual Clinical Environment

Pengcheng Qiu1,2,∗, Chaoyi Wu1,2,∗, Junwei Liu3,4,∗, Qiaoyu Zheng1,2, Yusheng Liao1,2, Haowen Wang3, Yun Yue3, Qianrui Fan3, Shuai Zhen3, Jian Wang3, Jinjie Gu3, Yanfeng Wang1,2, Ya Zhang1,2,† and Weidi Xie1,2,†

1Shanghai Jiao Tong University, Shanghai, China 2Shanghai Artificial Intelligence Laboratory, Shanghai, China 3Intelligence Healthcare Department, AntGroup, Hangzhou, China 4Intelligence Computing and Sensing Laboratory, Peking University, Beijing, China ∗Equal contributions †Corresponding author Ya Zhang: ya_zhang@sjtu.edu.cn; Weidi Xie: weidi@sjtu.edu.cn

In this paper, we present a framework for training large language models (LLMs) as diagnostic agents with reinforcement learning, enabling them to manage multi-turn interactive diagnostic processes, adaptively select examinations, and commit to final diagnoses. Unlike instruction-tuned models trained on static case summaries, our method acquires diagnostic strategies through dynamic exploration and outcome-based feedback, mapping evolving patient states to the next optimal examination and subsequent diagnosis. Our contributions include: (i) we present a diagnostics world model trained with electronic health records (EHRs), termed as DiagGym, which enables to emit examination outcomes conditioned on patient history and recommended examination, serving as a virtual clinical environment to support closed-loop in-silico training and evaluation for interactive diagnosis; (ii) we train an interactive diagnostic agent, DiagAgent, via end-to-end, multi-turn reinforcement learning within the environment, to learn dynamic diagnostic policies that optimizes both interactive effectiveness and final accuracy; (iii) We introduce a comprehensive multi-center diagnostic benchmark, DiagBench, designed to evaluate multiturn diagnostic interaction trajectories. The benchmark comprises 2.2K physician-validated cases sourced from four distinct distributions, alongside 3.3K physician-written rubrics for granular process-oriented evaluation. (iv) Extensive evaluations demonstrate DiagAgent’s superior performance across both in-domain and out-of-domain (OOD) settings. DiagAgent significantly outperforms 11 state-ofthe-art LLMs (including DeepSeek-v3 and Claude-4-Sonnet) and 2 prompt-engineered agents. In the end-to-end setting, it delivers a 11.20% increase in diagnostic accuracy and a 17.58% boost in examination recommendation F1 score, while consistently maintaining state-of-the-art performance across all three external OOD centers. Furthermore, in rubric-based evaluations, it surpasses the next-best model by 7.1% in weighted rubric score. These findings indicate that learning policies in interactive clinical environments confers dynamic and clinically meaningful long-term diagnostic management abilities that are unattainable through passive training alone.

###### 1 Introduction

Large language models (LLMs) have made tremendous progress in advancing medical AI, achieving strong performance on rigorous benchmarks, including USMLE-style examinations, and across diverse clinical tasks [1, 2, 3, 4, 5]. Recent advances in reasoning [6, 7] and post-training methods such as supervised fine-tuning (SFT) [8, 9, 10] have spurred interest in diagnostic applications, where the challenge is not simply answering a question but managing a complex, evolving patient case.

Clinical diagnosis, however, is not a static prediction problem. It is inherently an interactive long-term decision-making process under uncertainty: clinicians must synthesize partial information, decide which examination to recommend, or whether to commit to a diagnosis, that balance informativeness, timeliness, cost, and safety. Yet, existing LLMs are predominantly trained on passively collected, instruction-style corpora that assume a complete, fixed patient records [1, 4, 8, 11, 12, 13] without dynamic evolving. This static

- a. Method Overview
- b. DiagGym Pipeline c. DiagAgent Training

###### DiagGym

###### DiagAgent

Virtual Clinical Environment

Diagnostic Trajectory Manager

|…<br><br>Examination Query<br><br>Examination Results<br><br>Initial Inquiry<br><br>Examination Query<br><br>Examination Results<br><br>Final Diagnosis|
|---|

[Figure 1]

Chief Complaint: abdominal pain. History of Present Illness: It started around my belly button and then moved …

[Figure 2]

Current Diagnosis: The differential diagnosis includes appendicitis, gastroenteritis … Examination Query: CBC

Here are my examination results: Absolute Lymphocyte Count: 0.67 K/μL Basophils: 0.3% …

Lab Tests

… …

[Figure 3]

Current Diagnosis: Infectious process support the possibility of appendicitis … Examination Query: CT ABD

Here are my examination results: GASTROINTESTINAL: The stomach is unremarkable. Small bowel loops …

Radiology Examinations

Final Diagnosis: Appendicitis Reason: The patient's clinical presentation of right lower quadrant abdominal pain…

|Patient Profile<br><br>Chief complaint Present medical history Past medical history Social history Allergies Family history Final diagnosis<br><br>Past Examinations<br><br>CBC: WBC: 6.2 x10^9/L … Urinalysis: Nitrite: Negative<br><br>CT ABD AND PELVIS<br><br>[Figure 4]<br><br>Examination Query<br><br>Restructured EHRs|
|---|

Policy End-to-end Multi-turn RL Evolving

Different Diagnostic Trajectory Rollouts

DiagAgent

Self-explore

…

…

…

…

…

###### DiagGym

Diagnostics World Model

[Figure 5]

Examination Results

Stomach is unremarkable. Small bowel loops demonstrate normal caliber, …

- Figure 1 | Overview of our method. a illustrates the overview of our method, we establish a virtual clinical environment, DiagGym, that can simulate examination results in real time. Then, within it, we train a diagnostic agent capable of managing multi-turn diagnostic trajectories in a long-term manner, recommendations, recommending diverse examinations until sufficient evidence is gathered for a final diagnosis. b presents the diagnostics world model we constructed based on EHRs, representing a virtual clinical environment. It receives the patient’s basic profile and performed past examinations, and the next examination query as condition input, then simulates the related results as feedback. c depicts the DiagAgent end-to-end multi-turn Reinforcement Learning(RL) training, where the agent interacts with the virtual environment, rolls out different possible diagnostic trajectories, self-explores suitable examination recommendation chains, and iteratively evolves its decision-making policy through end-to-end reinforcement rewards.

paradigm collapses the multi-turn nature of real diagnosis into a single shot, eliminating the interaction with an external environment or revise hypotheses as dynamic evidence accumulates and updates. As a result, state-of-the-art models often fail to plan full diagnostic trajectories [10, 14, 15, 16, 17], including which tests to recommend, when to stop, and when to final diagnose.

In this paper, we present an reinforcement learning(RL) based framework for establishing LLMs as interactive diagnostic agents capable of dynamically suggesting examinations and rendering final diagnoses (Figure 1a). This advances clinical LLMs from isolated, point-in-time consultations to active patient management within a continually evolving diagnostic window.

Central to our agent-training framework is a diagnostics world model, DiagGym, built on the EHRs with generative model (Figure 1b). DiagGym is able to emit examination results conditioned on the former

patient state. This enables the safe, closed-loop simulation of real patient reaction, allowing the diagnostic agents to order tests and immediately observe their consequences. Within this environment, we train an interactive diagnostic agent, DiagAgent (Figure 1c) through end-to-end, multi-turn RL. The agent’s policy network maps the current patient state to the next optimal action: either recommending the most informative examination or finalizing the diagnosis. Reward signals are derived from the informativeness of queried examinations, diagnostic accuracy, and efficiency. This approach enables the agent to explore diverse patient trajectories, refine interactive diagnostic strategies through RL without relying on risky, time-intensive real-world implementation.

To comprehensively assess the final interactive diagnostic agent across diverse clinical settings, we constructed DiagBench, a multi-center benchmark comprising 2.2K physician-validated cases. DiagBench integrates data from four distinct sources: MIMIC-IV [18], PMC-OA case reports [19], MTSamples [20], and DDXPlus [21]. Each case is linked with a physician-verified reference trajectory. Furthermore, to enable fine-grained qualitative evaluation, we engaged senior physicians to author 3.3K weighted evaluation rubrics for a subset of 399 cases. These rubrics specify critical interactive diagnostic steps and rules, each assigned a weighted score, providing a rigorous standard for measuring the validity of the agent’s interactive decision-making process.

In experiments, we first evaluate DiagGym as a diagnostics world model for examination result generation. DiagGym significantly outperforms the DeepSeek-v3-based simulator, achieving higher instance-wise consistency (96.90% vs. 88.81%) and a lower examination-wise Wasserstein distance (0.128 vs. 1.336), indicating its fidelity. Additionally, its normalized variance aligns with the ground truth (4.65 vs. 5.31), highlighting robust simulation diversity with reduced mode collapse [22].

We further evaluate DiagAgent against 11 state-of-the-art LLMs (e.g., DeepSeek-v3, Claude-4-Sonnet) and two recent agentic systems under two complementary evaluation settings: single-turn evaluations on real cases and end-to-end evaluations on simulated cases. In the former, the model executes a single interactive diagnostic decision based on real-world observations. In the latter, the agent engages in multi-step interactions with DiagGym to complete full interactive diagnostic workflows. On the in-domain MIMIC-IV test set of DiagBench, DiagAgent shows superior performance, outperforming the nearest competitor by 8.94% in single-turn diagnostic accuracy and 43.99% in examination recommendation hit rate. In end-to-end evaluations, it leads by 11.20% in accuracy and 23.09 points 17.58% in F1 score, surpassing the runner-up (Claude-4-Sonnet) by 7.1% in weighted rubric scores. DiagAgent also exhibits robust generalization on the left multi-center out-of-distribution test set. In the single-turn setting, it achieves a 65.30% hit rate and 92.57% accuracy, significantly outpacing GPT-4o (52.09% and 87.06%). Furthermore, in end-to-end scenarios, DiagAgent attains 63.84% accuracy compared to Claude-4-Sonnet’s 60.12%, while achieving a significant higher weighted rubric score of 50.27 against 39.07. These gains highlight our method can effectively equip LLMs with dynamic, clinically meaningful interactive diagnostic capabilities.

###### 2 Problem Formulation

We first formalize the functionality of DiagGym and its role in training DiagAgent.

DiagGym. As illustrated in Figure 1b, we define a diagnostics world model as a conditional textual EHR generator, Φenv, that generates synthetic examination results conditioned on a dynamically evolving patient state. At step t, the patient EHR state is (B,Et), where B is the background patient profile—including chief complaint, present medical history, and the final diagnosis. The set Et = {(a1,e1),(a2,e2),··· ,(at,et)} represents the past examination records, where each ai denotes a specific examination item, e.g., “complete blood count” or “CT abdomen examination,” and ei is the result.

DiagGym is designed to generate the potential examination result for the patient based on a specific examination query at+1, as follows:

et+1 = Φenv(at+1 | Et,B), (1) where initially E0 = ∅, and et+1 denotes the synthetic examination result.

We frame the training process as a conditional generation task, that minimizes the negative log-likelihood of the ground-truth examination results. These results are treated as free text, regardless of whether they are numerical (with numbers directly embedded as text) or textual. The training objective is formalized as:

T−1

Lsim = −

log Φenv(ˆet+1 | at+1,Et,B), (2)

t=0

where eˆt represents the ground truth examination result at step t and T denotes the total examination length recorded in a certain EHR. More details can be found in Method Section 5.1.

Once trained, Φenv can generate plausible results for any examination and patient state, capturing conditional dependencies across diseases, histories, and prior tests. This capability enables safe and repeatable reinforcement learning (RL) training of diagnostic agents without direct access to real patient records; in other words, it serves as a virtual clinical environment for RL.

DiagAgent. As illustrated in Figure 1c, within DiagGym we train an interactive diagnostic agent, DiagAgent, using reinforcement learning. Formally, at time step t, the agent’s state is defined as st = (I,Et), where I is the patient’s initial inquiry—including chief complaint, history of present illness, and other relevant presentation details—but, unlike B in the environment model, contains no information about the final diagnosis. The set Et = {(a1,e1),...,(at,et)} records the examinations performed so far and their observed results.

For the agent, its action space is defined as A = {a1,a2,··· ,aN}, representing all available clinical examination items and the final diagnosis action. In response to the current state st, the agent selects an action to recommend the next examination for the patient, based on its policy:

at+1 ∼ πθ(a | st), (3) where πθ is the learnable policy function parameterized by a large language model Φdiag:

πθ = Φdiag(st). (4)

DiagGym then returns reasonable examination results, serving as the external environment feedback:

et+1 = Φenv(at+1 | Et,B), (5) st+1 = st ∪ (at+1,et+1), (6)

where st+1 is the next state.

The diagnostic trajectory proceeds until the agent selects a final diagnosis action, after which DiagAgent outputs the predicted diagnosis D. The ultimate training objective is to optimize the interactive policy function Φdiag to maximize the expected cumulative reward:

max

Φdiag

###### EΦ

diag

T

γtR(st,at) , (7)

t=1

where γ ∈ [0,1] is the discount factor, T is the trajectory length, and R is the reward function, defined as the sum of three sub-rewards:

R = λ1rdiag + λ2rexam + λ3rturn, (8)

with λ1,λ2,λ3 as hyperparameters. Here, rdiag encourages accurate diagnoses, rexam promotes relevant examination recommendations, and rturn rewards fewer used turns. The detailed design of the reward function is provided in the Method Section 5.2.

The final trained DiagAgent can actively manage multi-turn diagnostic trajectories by iteratively interacting with patients, selecting relevant examinations, and ultimately arriving at an accurate final diagnosis.

###### 3 Results

In this section, we first evaluate the two key components: DiagGym, the high-fidelity diagnostics world model, and DiagAgent, the reinforcement-trained diagnostic agent. Then, we carry out ablation studies to

investigate the effectiveness of our approach design. Lastly, we present detailed case studies. Notably, for simplicity, throughout all the following (including the Method section), we denote all prompts as prompt x. Further details regarding them are provided in Supplementary Section A.7.

###### 3.1 Evaluation for DiagGym

We first assess the fidelity and reliability of DiagGym against other strong open-source LLMs with the same simulation prompt (Section 5.4). The evaluation aims to verify whether the world model can generate clinically consistent, context-appropriate examination results that faithfully reflect real-world patterns in EHRs.

###### 3.1.1 Evaluation Settings

We construct an evaluation set of 863 patient cases from MIMIC-IV using the process in Section 5.1. These cases span 863 distinct diseases, categorized based on the original ICD codes in MIMIC-IV corresponding to patient admissions, representing a combined 35,548 examination records: on average, 8.77 physical exams, 28.37 laboratory events, 2.04 microbiology events, and 2.01 radiology events per case.

Each case comprises two components: (i) patient profile: baseline information including chief complaint, present and past medical history, social history, allergies, family history, and the final diagnosis; (ii) examination chain: the chronological sequence of actual examination results, serving as ground truth. The simulator’s task is to reconstruct each examination result in sequence, conditioned on the patient profile, all prior examination data, and the current examination query. Thus, during RL training, the simulator can dynamically respond to arbitrary examination queries.

We quantify the generation quality using instance-wise and examination-wise metrics (Figure 2), the detailed metric calculations are provided in Supplementary Section A.1.1. These metrics jointly assess whether simulated results match real diagnostic examination results at both the instance-wise and the examination-wise.

Instance-wise metrics. These metrics assess the quality of generated examination sequences at the level of individual patient cases. This evaluation utilizes two methods: heuristic metrics (BLEU [23] & normalised MAE and LLM-as-a-judge (GPT-4o, version gpt-4o-2024-08-06). The LLM-as-a-judge metrics are defined as below:

- • Step-level similarity measures how closely the simulator’s output for each examination step matches the corresponding real-world record, given the patient profile and all prior ground-truth results. Similarity is scored on a 0-5 scale, where 5 indicates perfect medical equivalence or high similarity to the reference. Both the automated evaluator and physician raters assign an independent score from 0 (no similarity) to 5 (perfect equivalence).
- • Full-chain consistency evaluates the coherence of an entire generated sequence where each step depends on the simulator’s previously generated outputs. This setting mirrors the use of reinforcement learning, prioritising internal clinical consistency over word-for-word agreement with ground truth. Consistency is judged using a binary score (1 = consistent, 0 = inconsistent). The automated evaluator provides a binary judgment using prompt 15. In the physician rating, raters must make a judgment ensuring the sequence maintains adherence to medical common sense, features appropriate calibration of severity, and is free of internal contradictions or conflicts across all reported findings.

Examination-wise metrics. To ensure DiagGym captures the full spectrum of real-world distributions rather than just specific instances, we adopt distribution-level metrics for different examination items, which are widely adopted in EHR generation [24, 25, 26]. These metrics assess global fidelity and diversity to guard against mode collapse [22], which might otherwise circumvent instance-level evaluations.

- • Numerical fidelity & diversity compares generated numerical values (e.g., red blood cell counts) to real distributions. Fidelity is quantified via the 1-Wasserstein distance (lower is better), while diversity is measured as the normalized variance of the generated distribution (higher reflects broader coverage and less mode collapse). Metrics are averaged across all selected numerical tests (Supplementary Section A.4).
- • Free-text fidelity & diversity: compares the generated narrative reports (e.g., CT abdomen findings) in embedding space using BioLORD [27]. Fidelity is measured by the Fréchet Inception

###### a. Instance-wise Metrics

|Patient Profile<br><br>The patient is …<br><br>[Figure 6]<br><br>Examinations：<br><br>CBC: WBC: 6.2 x10^9/L … Urinalysis: Nitrite: Negative CT ABD AND PELVIS:<br><br>——————————|
|---|

|Patient Profile<br><br>The patient …<br><br>[Figure 7]<br><br>Examinations：<br><br>CBC:<br><br>—————————Urinalysis:<br><br>—————————CT ABD AND PELVIS：<br><br>——————————|
|---|

Examination Results (Prediction)

Examination Results (Prediction)

CBC: WBC: 6.4 x10^9/L … Urinalysis: Nitrite: Negative CT ABD AND PELVIS: Stomach is unremarkable …

Stomach is unremarkable …

 Similarity:

Compare AI prediction with ground-truth and score similarity from 0-5

[Figure 8]

Examination Results (Ground Truth)

 Consistency:

[Figure 9]

- 1. No internal contradictions
- 2. Alignment with patient profile

The stomach appears normal…

###### b. Examination-wise Metrics

Examination Results (prediction)

 Diversity:  Fidelity

|Patient Profile<br><br>The patient …<br><br>[Figure 10]<br><br>Examinations：<br><br>CBC (or CT ABD) :|
|---|

Variance of evaluation results: Distributional distance to ground truth:

|Free-text<br><br>Appendix normal … Appendiceal wall …<br><br>Liver lesion with …<br><br>…|
|---|

|WBC 7.2 × 10⁹/L …<br><br>…<br><br>Numerical<br><br>WBC 15.8 × 10⁹/L …<br><br>WBC 6.1 × 10⁹/L …<br><br>|
|---|

- • Normalized Variance
- • LPIPS

- • Wasserstein Distance
- • FID

Generative Distribution Ground-truth distribution

[Figure 11]

or

[Figure 12]

——————————

- Figure 2 | Overview of simulator evaluation settings. a Instance-wise metrics: GPT-4o assesses the quality of generated examination results on an individual patient case level. b Examination-wise metrics: fidelity and diversity are evaluated by comparing the statistical distributions of generated examination results against those from real cases.

Distance [28] (lower is better) and diversity by Intra-LPIPS [29] (higher indicates more inter-case variation). Metrics are averaged over the selected free-text examinations (Supplementary Section A.4).

During the calculation of examination-wise metrics, we fit the generative distribution by sampling from the first examination step, i.e., directly conditioned on the ‘patient profile’ without giving extra past examination results. This aims to preserve enough sampling randomness, thus better reflect the underlying distribution. Instead, providing too much context, e.g., detailed past examinations, could overly constrain the generation and make it excessively deterministic. For each examination item, we only use test cases where that specific examination was actually performed, ensuring that both the generative and real distributions are calculated using the same patient set.

Computational metrics. In addition to the generation quality, we assess computational efficiency, as the simulator must respond rapidly to support interactive training. Two metrics are reported: minimal GPU, the lowest number of GPU cards required for deployment. Time (GPU·s), the average wall-clock time to generate a single examination result, multiplied by the minimal GPU count. All measurements were obtained on NVIDIA A100 80GB GPUs.

Physician Evaluation. Beyond former automated metrics, we also involve three independent physicians, each with over 10 years of experience, to further rate the generated cases. Due to cost constraints, the physician evaluation utilized a random sample of 100 instances. The rating rules are the same as the instance-wise LLM-as-a-judge metrics, i.e., step-wise similarity and full-chain consistency, straightforwardly replacing LLMs with human experts.

###### 3.1.2 Results Analysis

In this part, we analyze the main quantitative results under automated evaluation, computational efficiency and physician evaluation.

###### Automated Evaluation

The automated evaluation results are provided in Table 1. Across nearly all metrics, DiagGym delivers state-of-the-art performance, combining high-fidelity with substantially greater computational efficiency.

For instance-wise quality, DiagGym demonstrates superior performance across both heuristic and LLM-as-a-

- Table 1 | Quantitative comparison of different models as diagnostics world model for generating synthetic patient examination results. We report the mean value with the 95% Confidence Interval in brackets or range where applicable. Metrics include Similarity, Consistency, BLEU, normalised MAE(NMAE), Fidelity (Wasserstein Distance, FID), and Diversity (Normalized Variance, LPIPS).

|Computational Metrics|Instance-wise Metrics<br><br>|Examination-wise Metrics| |
|---|---|---|---|
| | |Numerical<br><br>|Free-text|
|Minimal GPUs↓<br><br>Time (GPU·s)↓<br><br>|Similarity↑ Consistency(%)↑ BLEU↑ NMAE↓<br><br>|Normalized Variance↑<br><br>Wasserstein Distance↓<br><br>|LPIPS↑ FID↓|

Model

0.427

GT - - - - - - 5.31 -

-

[0.31-0.51]

###### 24.56

2.576

88.81

4.37

0.691

1.336

0.237

4.158

DeepSeek-v3-671B 16 62.72

[2.56-2.59]

[86.6-90.8]

[4.21-4.54]

[0.643-0.740]

[1.37-70.0]

[0.39-3.02]

[0.20-0.27]

[3.68-4.63]

2.181

81.64

2.54

0.831

20.18

9.680

0.256

4.800

Qwen2.5-7B 1 0.54

[2.16-2.20]

[79.0-84.4]

[2.43-2.65]

[0.776-0.889]

[6.25-42.9]

[2.37-19.4]

[0.22-0.28]

[4.27-5.42]

2.495

92.40

4.43

0.962

1.21

1.839

0.183

4.905

Qwen2.5-72B 4 18.68

[2.48-2.51]

[90.5-94.1]

[4.28-4.59]

[0.901-1.026]

[0.45-2.13]

[0.38-4.28]

[0.14-0.21]

[4.26-5.53]

2.438

89.87

3.78

0.767

18.70

16.936

0.341

4.158

MedGemma-27B 2 9.1

[2.42-2.46]

[87.8-91.8]

[3.66-3.93]

[0.714-0.824]

[2.66-39.7]

[0.86-38.8]

[0.30-0.37]

[3.80-4.47]

###### 3.565

###### 96.90

###### 45.97

###### 0.442

###### 0.128

###### 0.747

4.65

0.379

DiagGym 1 0.52

[3.55-3.58]

[95.6-98.0]

[45.36-46.54]

[0.413-0.474]

[1.39-8.80]

[0.11-0.15]

[0.24-0.47]

[0.61-0.91]

judge metrics. It significantly outperforms competitive baselines, for example, achieving a similarity score of 3.565 compared to 2.576 for DeepSeek-v3 and 2.495 for Qwen2.5-72B. While large-scale models maintain reasonable consistency, they struggle to align closely with the ground truth. Conversely, smaller models (e.g., Qwen2.5-7B) show marked degradation in both consistency and alignment.

Regarding examination-wise metrics, DiagGym effectively balances fidelity and diversity, achieving distributions closest to real-world data, e.g., lowest Wasserstein Distance (0.128) and FID (0.747). In contrast, baselines exhibit polarized behaviors: models like DeepSeek-v3-671B and MedGemma-27B show high diversity but suffer from large distribution gaps (poor fidelity), whereas Qwen2.5-72B produces overly deterministic outputs with minimal diversity.

###### Computational Efficiency

Existing large-scale baselines trade performance for heavy resource demands. For instance, a single simulation with DeepSeek-v3-671B necessitates a multi-GPU setup (16×A100) and exceeds one minute of computation time. In sharp contrast, DiagGym operates on a single A100 GPU with sub-second latency (≈0.5s). This order-of-magnitude acceleration, achieved without compromising generative quality, is critical for the rapid interactions required in diagnostic agent reinforcement learning.

###### Physician Evaluation

As shown in Table 2, DiagGym consistently outperforms baselines, achieving the highest average similarity score (4.49) and a dominant majority-vote consistency of 95%, whereas baselines fluctuate between 44% and 74%. Qualitatively, physicians noted that DiagGym effectively avoids common baseline failure modes, such as over-extrapolation and logical inconsistencies, maintaining balanced coverage and strict alignment with the case context.

Collectively, these results demonstrate that DiagGym is a high-fidelity, diverse, and computationally efficient world model. It is reliable and well-suited to serve as a virtual clinical environment for dynamic diagnostic agent training with Reinforcement Learning, substantially outperforming current open-source baselines. More detailed qualitative case study are shown in Supplementary Section A.6.1.

###### 3.2 Evaluation for DiagAgent

We next evaluate our diagnostic model, DiagAgent, against leading LLMs and agentic systems (baselines described in Section 5.4), focusing on its ability to manage complete multi-turn diagnostic trajectories.

###### 3.2.1 Evaluation Settings

We evaluate DiagAgent on DiagBench, a multi-center benchmark designed to rigorously assess multi-turn diagnostic capabilities across diverse clinical settings. As shown in Table 3, the benchmark comprises a total

- Table 2 | Human ratings comparing DiagGym with baseline models. Similarity (0-5) is reported for each of the three physicians and as the mean across physicians. Consistency is a binary judgment, we report the per-physician percentage of cases judged clinically coherent and the majority-vote consistency rate (percentage of cases deemed coherent by at least two of three physicians).

Model

|Similarity<br><br>| |Consistency(%)| |
|---|---|---|---|
|Physician 1 Physician 2 Physician 3<br><br>|Avg. Score<br><br>|Physician 1 Physician 2 Physician 3|Majority Vote|

DeepSeek-v3-671B 4.66 4.49 3.11 4.09 54.00 58.00 42.00 54.00 Qwen2.5-72B 4.50 4.37 3.04 3.97 46.00 44.00 32.00 44.00 MedGemma-27B 4.56 4.28 2.82 3.89 73.00 75.00 56.00 74.00

DiagGym 4.71 4.70 4.05 4.49 96.00 94.00 92.00 95.00

- Table 3 | Data Statistics of DiagBench. The benchmark spans one in-domain center and three out-of-domain (OOD) centers. The OOD subsets focus on diverse clinical scenarios beyond the ICU setting. All cases underwent rigorous human validation and rubric construction.

Domain Source Dataset Access Clinical Setting # Cases # Rubrics In-Domain MIMIC-IV [18] [Link] Critical Care and Emergency Medicine 750 973

PMC-OA [19] [Link] Open Access Biomedical Case Reports 631 1075 MTSamples [20] [Link] Transcribed Outpatient Records 379 620 DDXPlus [21] [Link] Synthesized Differential Diagnosis 497 650

Out-of-Domain

OOD Total - - 1,507 2,345 Total All 4 Centers - Comprehensive Dynamic Benchmark 2,257 3,318

of 2,257 physician-validated patient cases. Considering the training distribution of DiagAgent, we stratify the benchmark into: (1) a MIMIC-IV In-Domain (ID) test set (750 cases), representing critical care and emergency medicine; and (2) a multi-center Out-of-Domain (OOD) test set (1,507 cases) aggregated from PMC-OA, MTSamples, and DDXPlus, covering multi-source patient distributions, including global case reports, outpatient records, and synthesized differential diagnosis cases based on statistic prior. To enable fine-grained evaluation, the benchmark is further annotated with 3,318 physician-authored rubrics (973 for MIMIC-IV and 2,345 for the OOD subsets). These rubrics delineate critical diagnostic interaction logic and necessary examinations within the diagnostic process. The detailed DiagBench construction pipeline can be found in Section 5.3.

All cases are standardized into a unified format containing three key element: (i) initial inquiry: patient initialized presentation details (chief complaint, current and past medical history, and other relevant information), forming the starting point for the diagnostic process; (ii) referenced multi-turn diagnostic trajectory: a physician-curated sequence extracted from real EHR records, serving as the ground-truth reference; (iii) final diagnosis: the final confirmed clinical diagnosis outcome. Notably, cases are structured following the same simulation case-construction pipeline used earlier, ensuring each includes a patient profile compatible with the simulator.

We consider two complementary evaluation scenarios with corresponding metrics, namely, single-turn and end-to-end evaluation, as detailed below.

###### Single-turn Evaluation

In this setting, we evaluate the DiagAgent in the single-turn setting. As shown in Figure 3a,b, here, we leverage both the ground truth ‘initial inquiry’ and partial ‘referenced multi-turn diagnostic trajectory’ as input. The DiagAgent is directly forced (prompt details can be found in Section 5.4) to provide an examination recommendation or make a final diagnosis based on the preceding oracle diagnostic trajectory, extending the process by one additional turn, without self-deciding which action to perform next. Such single-turn evaluations are conducted at each agent response turn recorded in the referenced multi-turn trajectory. For example, for the 750 cases sourced from MIMIC-IV, this expanding into 4,485 turns for evaluation, consisting 3,735 intermediate turns for examination recommendation and 750 final turns for diagnosis.

For examination recommendation turns, we calculate the hit ratio based on whether the suggested examination is in the key examination list. Specifically, for MIMIC-IV, the relevant list is derived directly from the historical records of the current admission; for other datasets, key examinations are extracted from the raw text. To assess final diagnosis, we employ Accuracy. To avoid issues with synonyms and the inclusion relationships between examinations, we utilize GPT-4o to judge whether an examination recommendation appears within the remaining part in the ‘referenced multi-turn diagnostic trajectory’ with prompt 6.

###### End-to-End Evaluation

In this setting, we adopt an end-to-end evaluation approach. The diagnostic trajectory is initialized with the ‘initial inquiry’, after which diagnostic agents continually interact with the environment and sequentially propose examination queries until they determine that a final diagnostic decision can be made. Throughout the trajectory, all returned examination results are simulated by the DiagGym, conditioned on the background ‘patient profile,’ ensuring that all queried information is available.

This evaluation setting more closely reflects real-world clinical practice, highlighting the model’s ability to dynamically construct a complete diagnostic trajectory, autonomously determining both the timing and type of actions based on the patient’s evolving condition. While the assessment necessarily relies on the external diagnostics world model to simulate examination results, this stems from a fundamental limitation of real-world EHRs: they only contain examinations that were actually performed. Consequently, when the diagnostic agent suggests an examination that was not carried out for a given patient, there is no corresponding result in the EHR, making it impossible to evaluate that decision or its downstream effects. These inherent gaps prevent the use of real-world EHRs for fully interactive, end-to-end evaluation, as the diagnostic trajectory would be repeatedly interrupted by missing information.

After obtaining the complete predicted diagnostic trajectory, we evaluate performance from two perspectives. First, as shown in Figure 4a, for the all cases in DiagBench we employ automatic metrics to assess the efficacy of examination recommendations and the accuracy of the final diagnosis. (i) Examination recommendation compares the examination items proposed by the model with those in the reference multi-turn diagnostic trajectory, and computing precision, recall, and F1-score to measure recommendation quality; (ii) Final diagnosis assesses the accuracy of the model’s ultimate diagnosis after completing the multi-turn interaction, by directly comparing it to the ground-truth diagnosis.

Second, as illustrated in Figure 5a, for cases annotated with physician-authored rubrics, we derived a weighted rubric score to more accurately reflect expert satisfaction with the generated diagnostic interactions. Following the methodology of HealthBench [30], we adopt an LLM-as-a-judge approach, prompting GPT-4o to verify whether the diagnostic process adheres to specific criteria and computing a final score based on the assigned clinical weights. This metric serves as a necessary complement to standard automated metrics. Detailed rubrics can capture the temporal logic (which examinations should be recommended first) and critical rule-out steps essential to rigorous clinical reasoning (which examinations must be done). More implementation details and human validation of these metrics are provided in Supplementary Section A.1.2 and A.1.3 respectively.

###### 3.2.2 Single-turn Evaluation Analysis

In this section, we analyze the performance of DiagAgent under single-turn evaluation. The main results are shown in Figure 3. In-Domain Analysis

- Figure 3c summarize the performance of all models under single-turn evaluation on the MIMIC-IV test set. Models are grouped into three categories: basic LLMs, agentic systems, and our DiagAgent variants.

DiagAgent variants deliver substantial gains across all metrics. The best-performing model, DiagAgent-14B, achieves a Hit Ratio of 68.49% and diagnosis accuracy of 87.87%. This performance establishes a massive margin over baselines: it improves examination recommendation by over 40% compared to the strongest medical LLM (MedGemma) and exceeds general-purpose models like DeepSeek-v3 by nearly 48%. These results confirm that post-training within DiagGym fundamentally boosts clinical reasoning capabilities beyond current state-of-the-art (SOTA) levels.

Among Basic LLMs, close-sourced models (e.g., Claude-4-sonnet, GPT-4o) generally outperform open-source

a. Single-turn Evaluation Setting (Examination Recommendation)

b. Single-turn Evaluation Setting (Final Diagnosis)

[Figure 13]

[Figure 14]

[Figure 15]

Initial Inquiry: Patient Information: Female Chief Complaints: ….

[Figure 16]

Model Prediction: Complete Blood

Initial Inquiry: Patient Information: Female Chief Complaints: ….

Model Prediction: Appendicitis

ü Hit Ratio:

Input

Past Examinations： Liver Function Test: INR(PT):1.1 … …

Input

[Figure 17]

Past Examinations： Liver Function Test: INR(PT):1.1 … …

ü Accuracy:

Model recommended examination occur in real list?

[Figure 18]

Agreement between model and reference diagnosis?

Referenced Examinations:

…

…

- 1. General (PE)
- 2. Liver Function Test
- 3. Complete Blood …

Output

Output

Instruction: Recommend examination directly.

Instruction: Make final diagnosis directly.

?

?

Final Diagnosis: Appendicitis

c. Single-turn Evaluation Results (In-domain)

Examination Recommendation Final Diagnostic Decision-making

[Figure 19]

[Figure 20]

Basic LLM Agentic System

Ours

Accuracy(%)

HitRatio(%)

Closedsource

Open-source Medical

###### d. Single-turn Evaluation Results (Out-of-domain)

Examination Recommendation Final Diagnostic Decision-making

[Figure 21]

[Figure 22]

Accuracy(%)

HitRatio(%)

[Figure 23]

- Figure 3 | Overview of single-turn evaluation settings and results. a shows the single-turn evaluation setting for examination recommendation measured with the hit ratio. b illustrates the single-turn evaluation setting for final diagnosis measured with the accuracy. c benchmarks our DiagAgent variants against 11 leading LLMs and 2 agentic systems on the MIMIC-IV dataset. The detailed numbers can be found in Supplementary Table 3. d evaluates the model’s performance on out-of-domain datasets sourced from PMC-OA, MTsamples, and DDXPlus. The detailed number can be found in Supplementary Table 3 and 11. Error bars show 95% confidence intervals. Significance levels relative to the DiagAgent-14B are marked as: ∗p < 0.05, ∗∗p < 0.01, and ∗∗∗p < 0.001.

counterparts but still struggle with examination recommendation, remaining below 30%. Notably, scaling parameters, as seen in DeepSeek-v3 (671B), does not yield proportional gains in medical recommendation without targeted adaptation. While medical-specialized models, like MedGemma, Baichuan-M1, Baichuan-M3 show slight advantages over general LLMs, they still fall significantly short of the DiagAgent series.

Regarding Agentic Systems, frameworks like MedAgents and MDAgents (based on DeepSeek-V3) fail to deliver meaningful improvements over their base models. This suggests that without a well-aligned foundation, multi-agent coordination alone cannot overcome the complexity of interactive diagnostic reasoning and may even exacerbate hallucination-driven errors.

###### Out-of-Domain Analysis

As shown in Figure 3d, DiagAgent-14B maintains robust leadership in out-of-domain settings, achieving a Hit Ratio of 65.30% and Diagnosis Accuracy of 92.57%. This significantly outperforms both the strongest open-source baseline (Llama3.3-70B) and proprietary SOTA models (GPT-4o). Crucially, while some baselines (e.g., MedGemma, Baichuan-M1) demonstrate reasonable adaptability in diagnosis accuracy, DiagAgent is the only model to simultaneously achieve high precision in both examination retrieval and diagnostic decision-making. This confirms that the policy learned from DiagGym captures generalized medical logic rather than merely overfitting to MIMIC-IV distributions.

###### 3.2.3 End-to-end Evaluation Analysis on Automatic Metrics

In this section, we analyze the performance of DiagAgent under end-to-end evaluation on automatic metrics. The main results are shown in Figure 4.

In-Domain Analysis

As shown in Figure 4b, DiagAgent models achieve the highest scores across all evaluation metrics. Crucially, additionally reported in Supplementary Table 4, DiagAgent-14B engages in substantially longer diagnostic dialogues (average 6.66 turns) compared to standard LLMs like DeepSeek-v3 (≈2.5 turns). This sustained interaction facilitates comprehensive evidence gathering, yielding a recall of 52.14%, which is more than four times that of the strongest baselines, without compromising precision. Consequently, DiagAgent-14B achieves a dominant diagnostic accuracy of 62.63%, outperforming the nearest competitor by over 10%. These findings confirm that the active exploration capability acquired through DiagGym directly translates into more informed and accurate decision-making.

Among Basic LLMs, advanced general models (e.g., Claude-4-sonnet, DeepSeek-V3) tend to outperform medical LLMs, like MedGemma and Baichuan series. Despite their domain knowledge, medical LLMs struggle in this dynamic setting, indicating that static medical knowledge does not automatically confer the ability to actively query and integrate evidence across multi-turn interactions.

Regarding Agentic Systems, frameworks like MedAgents and MDAgents (based on DeepSeek-V3) fail to deliver meaningful improvements over their base models. Comparison of turn lengths reveals that these multi-agent systems suffer from premature closure, averaging fewer than 2.5 turns and terminating inquiry before gathering sufficient information. This results in low recall and stagnation in diagnostic accuracy, further validating that “expert discussion” prompts cannot substitute for a well-aligned underlying policy.

Out-of-Domain Analysis

We further validate robustness on the out-of-domain data in DiagBench (Figure 4c). DiagAgent-14B retains its leadership, achieving the highest F1-score (33.63%) and Diagnosis Accuracy (63.84%) among all 13 evaluated systems, significantly surpassing both proprietary SOTA models (Claude-4-Sonnet) and open-source giants (Qwen3-235B). Consistent with in-domain behavior, DiagAgent maintains a longer average turn duration (6.88 turns) compared to baselines, as detailed in Supplementary Table 12. This confirms that the model’s active inquiry strategy is a fundamental, generalized behavioral trait rather than an artifact overfitting the MIMIC-IV dataset.

In summary, these results highlight the importance of post-training LLMs within clinically realistic, interactive environments. By endowing models with the capacity to determine the timing and content of their multi-turn actions, DiagGym substantially improves both the quality of examination recommendations and the accuracy of final diagnoses, narrowing the gap towards deployable, decision-capable clinical AI systems.

###### 3.2.4 End-to-end Evaluation Analysis on Rubric-based Metrics

In this section, we analyze the performance of DiagAgent under end-to-end evaluation on rubric-based metrics. The main results are shown in Figure 5.

###### In-Domain Analysis

Figure 5b indicates that our dynamic training paradigm significantly enhances alignment with high-importance clinical protocols. DiagAgent-14B achieves a weighted rubric score of 32.86%, outperforming the strongest basic LLM (Qwen3-235B) by over 8% and the best agentic baseline (MDAgent) by over 10%. These results suggest that DiagAgent does not merely guess the final diagnosis but demonstrates superior procedural interactive reasoning, satisfying rigorously weighted criteria for history taking and examination selection.

Among baselines, general LLMs (e.g., Claude-4-sonnet, Qwen3-235B) consistently outscore smaller medicalspecialized models. While medical models may arrive at correct diagnoses, they frequently fail to satisfy process-focused rubrics due to weaker reasoning capabilities, suggesting that prior clinical LLM adaptation methods have neglected the enhancement of multi-turn interactive capabilities. Agentic systems like MDAgent provide only marginal gains (≈2.5%) over their base models. This indicates that inference-time discussion alone is insufficient to ensure procedural quality without fundamental policy alignment.

- a. End-to-end Evaluation Setting with Automatic Metrics
- b. End-to-end Results with Automatic Metrics (In-domain)

[Figure 24]

[Figure 25]

Closedsource

Open-source Medical

Initial Inquiry: Patient Information: Female Chief Complaints: ….

Instruction: Recommend further examinations as needed until sufficient information is obtained, and then make a final diagnosis.

Output

…

?

Input

? ?

Predicted Diagnostic Trajectory

Examination Recommendation Final Diagnostic Decision-making

F1-score

Accuracy(%)

- c. End-to-end Results with Automatic Metrics (Out-of-domain)

# …

[Figure 26]

ü Accuracy

General (PE) CT ABD AND PELVIS

###### Model Prediction:

Final Diagnosis: Appendicitis

[Figure 27]

Appendicitis

[Figure 28]

Referenced Exanimations List

###### Model Prediction List

[Figure 29]

- 1. General (PE)
- 2. Complete Blood …

ü Precision ü Recall ü F1

- 1. General (PE)
- 2. CT ABD AND PELVIS …

Examination Recommendation Final Diagnostic Decision-making

[Figure 30]

[Figure 31]

Basic LLM Agentic System

Ours

Accuracy(%)

F1-score

[Figure 32]

[Figure 33]

[Figure 34]

- Figure 4 | Overview of end-to-end evaluation. In this setting, diagnostic agents are evaluated through end-to-end finishing the entire diagnostic trajectory by interaction with the external diagnostics world model. a illustrates the end-to-end evaluation pipeline with automatic metrics to assess examination recommendation efficacy and diagnostic accuracy. b benchmarks our DiagAgent with 11 LLMs and 2 more agentic systems under end-to-end evaluation settings with automatic metrics on the MIMIC-IV dataset. c evaluates the model’s e performance on out-of-domain datasets sourced from PMC-OA, MTsamples, and DDXPlus. The detailed number can be found in Supplementary Table 4, 12. Error bars show 95% confidence intervals. Significance levels relative to the DiagAgent-14B are marked as: ∗p < 0.05, ∗∗p < 0.01, and ∗∗∗p < 0.001.

###### Out-of-Domain Analysis

We further assess alignment with clinical standards on the out-of-domain data in the benchmark (Figure 5c). DiagAgent-14B establishes a decisive lead with an aggregate weighted score of 50.27%, creating a substantial gap of over 10% compared to both the strongest proprietary (Claude-4-sonnet) and open-source (Qwen3-235B) baselines. This superiority is consistent across all individual data sources (PMC-OA, MTSamples, DDXPlus). DiagAgent outperforms second competitors by wide margins, i.e., surpassing Claude-4-Sonnet by 10.65% on PMC-OA, exceeding GPT-4o by 18.62% on MTSamples, and outperforming DeepSeek-V3 by 19.98% on DDXPlus. Statistical hypothesis testing confirms that DiagAgent’s improvements are significant (p < 0.05) against all baselines across these diverse clinical scenarios.

Overall, the results on DiagBench reinforce that dynamic training focused on intermediate reasoning processes enables significantly improved satisfaction of high-value clinical procedures. DiagAgent consistently surpasses both large generic LLMs and current agentic frameworks, underscoring the essential role of fine-grained, process-aware training for safe and effective clinical decision support.

###### a. End-to-end Evaluation with Rubrics b. In-domain Performance

## …

Predicted Diagnostic Trajectory MIMIC-IV

[Figure 35]

[Figure 36]

Basic LLM Agentic System

WeightedRubricScores(%）

Ours

Input

[Figure 37]

[Figure 38]

Closedsource

Open-source Medical

###### ? Rubric Points Present

Prioritize malignant partial small bowel obstruction …

?

+10 Yes 10

Output

###### …

Recommend appropriate abdominal imaging studies …

+9 Yes 9

Assess the patient's medication use to identify drug interactions…

+5 No 0

?

0

19 24 max

###### c. Out-of-domain Generalization Performance

PMC-OA MTSamples

[Figure 39]

[Figure 40]

WeightedRubricScores(%WeightedRubricScores(%））

WeightedRubricScores(%WeightedRubricScores(%））

DDXPlus Combined OOD Datasets

[Figure 41]

[Figure 42]

[Figure 43]

- Figure 5 | Overview of end-to-end evaluation settings and results. In this setting, diagnostic agents are evaluated through end-to-end finishing the entire diagnostic trajectory by interaction with the external diagnostics world model. a illustrates our end-to-end evaluation pipeline with rubric-based metrics. A judge model evaluates the full diagnostic trajectory of a diagnostic model against physician-curated rubrics, which specify criteria, clinical importance weights (Points), and whether the criterion was satisfied (Present). b compares the aggregate weighted proportion of satisfied rubrics(%) across different LLMs in different model sizes on the MIMIC-IV dataset. c evaluates the model’s e performance on three out-of-domain (OOD) sources separately: PMC-OA, MTsamples, and DDXPlus. Error bars show 95% confidence intervals. Significance levels relative to the DiagAgent-14B are marked as: ∗p < 0.05, ∗∗p < 0.01, and ∗∗∗p < 0.001.

###### 3.3 Ablation Study

We conducted ablation experiments under the end-to-end evaluation setting to assess three aspects of the proposed framework: (i) whether the reinforcement learning in virtual environment outperforms supervised fine-tuning (SFT) at the same model scale; (ii) the impact of reward design, comparing diagnosis-only rewards with dual rewards incorporating both diagnosis accuracy and examination-recommendation quality; and (iii) the generality of DiagAgent across different model sizes and families.

###### Experimental Design

As outlined in Table 4, for each base model, we first establish a zero-shot baseline in which the LLM answers without fine-tuning. We then apply full supervised finetuning, in which all cases are converted into multi-turn diagnostic dialogues for supervised training (Supplementary Section A.2), bypassing the simulator and reinforcement learning pipeline. Finally, we test three DiagAgent configurations: (i) coldstart only: supervised fine-tuning on a small subset to learn output format; (ii) cold-start + RL with diagnosis reward: RL optimising diagnosis accuracy only; (iii) full DiagAgent: diagnose reward plus examination-recommendation reward.

###### Supervised Finetuning (SFT) vs. Reinforcement Learning (RL)

As shown in Table 4, zero-shot baselines perform poorly (diagnosis accuracy: 16.93% for Qwen2.5-7B; 34.93% for Qwen2.5-14B), highlighting the difficulty of managing interactive trajectories without domain adaptation. Full SFT improves diagnosis accuracy (45.33%, 47.07%, and 47.60% for Qwen2.5-7B, Llama3.1-8B, and Qwen2.5-14B, respectively) and examination recommendation quality, but the gains are limited by static trajectory data extracted from MIMIC-IV discharge notes, which do not reflect the dynamic branching in interactive consultations. In contrast, our RL-trained DiagAgent achieves a universal performance gain across all three base models, outperforming the SFT-only counterparts. Specifically, accuracy surges to 61.47% (Qwen2.5-7B), 53.33% (Llama3.1-8B), and 62.67% (Qwen2.5-14B). Furthermore, DiagAgent reduces both the average dialogue turns and token length, proving that the transition from static imitation (SFT) to dynamic feedback learning (RL) fosters more efficient and decisive diagnostic reasoning.

To ensure the SFT baseline was not limited by insufficient compute, we analyzed the impact of training duration in Supplementary Section A.3. We found that extending SFT to 10 epochs resulted in a performance plateau after epoch 3. This confirms that our gains are attributable to the superior RL learning paradigm, not merely additional training steps.

###### Effect of Reward Design

To demonstrate the effectiveness of reward design, we first conduct a cold start phase, where each model is supervised-finetuned on a subset of the training set to learn the output format. Adding RL with only the Diagnose Reward yields large gains over SFT, for example, Qwen2.5-7B improves from 44.40% (Full SFT) to 59.41%, Qwen2.5-14B from 45.98% to 59.73%, and Llama3.1-8B from 45.35% to 54.12%. However, but F1 scores for examination recommendation remain low (≤ 36%). Introducing the Examination Recommendation Reward markedly improves F1 across all models (Qwen2.5-7B: 32.76% → 46.86%; Qwen2.5-14B: 35.95% → 47.89%), with slight additional gains in accuracy. This confirms the importance of dual-reward shaping for balancing precision in diagnosis and quality in examination planning.

###### Model Size and Family

Reinforcement learning with virtual environment benefits all LLMs training, for example, in diagnosis accuracy, from 16.38% to 60.78% for Qwen2.5-7B, 25.16% to 53.85% for Llama3.1-8B, and 33.83% to 61.63% for Qwen2.514B, proving the effectiveness of our method. Larger or intrinsically stronger base models achieve higher post-training ceilings: Qwen2.5-14B delivers the best overall performance (61.63% accuracy, 47.89% F1), followed by Qwen2.5-7B (60.78%, 46.86%), while Llama3.1-8B lags (53.85%, 43.02%). This suggests that while DiagAgent’s exploration-driven optimisation is broadly applicable, the quality of the base model constrains the attainable performance upper bound.

###### 3.4 Case Study

The following case study illustrates the qualitative performance of DiagAgent, offering insights into its ability to navigate reasoning and maintain high fidelity in complex clinical scenarios.

###### A Case of Interactive Diagnostic Trajectory

- As shown in Supplementary Figure 3, we present a case study illustrating DiagAgent-14B’s dynamic interactive diagnostic trajectory within the DiagGym environment, modeling a typical appendicitis work-up. Each case record includes the initial inquiry, final diagnosis, interactive exchanges between diagnostic agent and simulator, and a reference ground-truth diagnostic timeline drawn from clinical records.

First, the model’s decisions follow standard reasoning. Upon receiving initial symptoms—abdominal pain migrating to the right lower quadrant, nausea, diarrhoea, and anorexia—the diagnostic agent prioritises appendicitis in the differential and orders a complete blood count (CBC). When the CBC reveals an abnormally high neutrophil count, further pointing toward infection or inflammation, the diagnostic agent appropriately requests a CT scan of the abdomen and pelvis with contrast, which display a dilated, fluid-filled appendix with periappendiceal fat stranding, confirming acute appendicitis. Throughout the process, each diagnostic step and rationale aligns closely with the reference timeline, demonstrating reliable differential diagnosis.

Second, the dynamic environment’s responses remain consistent with the patient’s case summary and expected clinical progression. For instance, when the CBC is requested, it provides results consistent with an acute

- Table 4 | Results of ablation study. We report the Avg. Turns, precision, recall and F1-score for examination recommendations and accuracy for diagnosis. All metrics are reported with 95% Confidence Intervals

|Instruction Tuning<br><br>|Reinforcement Learning|
|---|---|
|Full SFT Cold Start<br><br>Recommend Reward<br><br>Diagnosis Reward| |

Examination Recommendation Diagnosis

Method

Avg. Turns

Precision Recall F1 Accuracy(%)

Qwen2.5-7B

17.76

8.62

10.09

16.93

Baseline ✗ ✗ ✗ ✗ 1.97

[15.54-20.37]

[7.37-9.94]

[8.76-11.58]

[14.40-19.60]

37.51

50.99

39.70

45.33

Full SFT ✔ ✗ ✗ ✗ 7.98

[35.59-39.50]

[49.02-52.86]

[38.03-41.29]

[41.87-49.07]

- ✗ ✔ ✗ ✗ 8.36

33.03

[31.26-34.85]

47.16

[45.18-49.25]

35.35

[33.73-36.89]

36.13

[32.66-39.73]

- ✗ ✔ ✗ ✔ 4.46

37.07

[35.11-39.03]

33.64

[31.78-35.28]

32.75

[31.00-34.37]

60.40

[56.93-64.00]

- ✗ ✔ ✔ ✔ 5.45

DiagGym

###### 46.02

###### 43.90

###### 61.47

47.33

[44.04-47.82]

[45.45-49.27]

[42.29-45.53]

[58.27-64.67]

Llama3.1-8B

23.49

19.15

18.73

25.60

Baseline ✗ ✗ ✗ ✗ 4.36

[21.59-25.31]

[17.52-20.79]

[17.17-20.25]

[22.67-28.67]

###### 49.66

###### 38.63

36.90

47.07

Full SFT ✔ ✗ ✗ ✗ 7.89

[35.06-38.89]

[47.61-51.68]

[36.99-40.28]

[43.33-50.67]

- ✗ ✔ ✗ ✗ 9.03

31.80

[30.00-33.73]

48.06

[45.93-49.96]

34.37

[32.86-36.02]

32.00

[28.67-35.33]

- ✗ ✔ ✗ ✔ 5.15

36.91

[35.13-38.86]

37.47

[35.50-39.39]

34.82

[33.07-36.52]

52.53

[48.93-56.13]

- ✗ ✔ ✔ ✔ 5.73

DiagGym

###### 39.57

###### 53.33

43.13

38.56

[37.74-41.41]

[41.29-45.12]

[36.89-40.18]

[49.73-56.80]

Qwen2.5-14B

33.98

17.14

19.70

34.93

Baseline ✗ ✗ ✗ ✗ 3.61

[31.02-36.96]

[15.61-18.61]

[18.06-21.24]

[31.47-38.53]

38.01

50.52

39.74

47.60

Full SFT ✔ ✗ ✗ ✗ 7.63

[36.03-39.84]

[48.52-52.55]

[38.15-41.38]

[43.87-51.20]

- ✗ ✔ ✗ ✗ 8.56

32.99

[31.21-34.83]

48.49

[46.36-50.40]

35.62

[34.01-37.25]

35.47

[32.27-38.67]

- ✗ ✔ ✗ ✔ 5.51

37.90

[35.97-39.78]

37.72

[35.92-39.68]

35.43

[33.71-37.12]

58.27

[54.67-61.73]

- ✗ ✔ ✔ ✔ 6.66

DiagGym

###### 42.04

###### 52.14

###### 43.72

###### 62.67

[40.13-43.89]

[50.20-54.09]

[42.09-45.56]

[59.20-66.00]

inflammatory, including elevated white blood cell and neutrophil counts; when the CT is ordered, it returns hallmark imaging features of appendicitis. This realistic feedback ensures that the agent’s decision-making unfolds in a manner faithful to real clinical workflows.

###### Cases of Rubric-based Evaluation

To further scrutinize the procedural quality of DiagAgent’s intermediate diagnostic steps, we employed an evaluation based on physician-curated rubrics that assess the integrity of multi-turn clinical interactions.

A typical successful case (left lower extremity infection) is presented in Supplementary Figure 4, illustrating the agent’s robust procedural performance. The agent exhibits high coherence and dynamic strategy adjustment: it orders a CBC and, upon receiving a result showing an elevated but non-critical neutrophil count, appropriately requests a wound culture to identify the causative organism. Following the positive culture for Staphylococcus aureus, the agent orders a blood culture to rule out bacteremia and then efficiently terminates further investigation after receiving a negative result, avoiding over-testing. The high scores on the procedural rubrics confirm that the agent’s decision-making process is both efficient and clinically sound, successfully meeting criteria for prioritizing tests, interpreting results, and achieving evidence-driven closure.

We also provide a illustrative failure case (ruptured ectopic pregnancy with hemodynamic instability) in Supplementary Figure 5 to showcase model’s fail mode. While the agent’s diagnostic reasoning is highly

effective—it correctly orders hCG and subsequent pelvic ultrasound based on the patient’s unstable presentation, rapidly confirming the diagnosis—the evaluation reveals critical omissions in immediate emergency care. Specifically, the agent fails to satisfy the highest-weighted rubrics concerning emergency resuscitation and surgical team notification. It is important to emphasize that DiagAgent is primarily designed as a diagnostic reasoning model. Its core capability of accurate differential diagnosis and sequential information gathering remains intact, validating its utility for diagnostic quality enhancement despite the observed gap in acute therapeutic and stabilization management, which falls outside its initial scope of contribution.

###### 4 Discussion

Large language models (LLMs) have achieved notable success across a range of clinical tasks [31, 8, 9, 1, 10, 3, 32], yet they remain fundamentally limited in dynamic, multi-turn decision-making. Even state-of-the-art systems often struggle with real-time interactive diagnostic reasoning, deciding which examinations to order, when to order them, and how to coordinate an efficient, end-to-end diagnostic process [10, 14, 16]. Unlike human physicians, who adaptively update decisions as new information emerges, current LLMs frequently fail to manage trajectories effectively under uncertainty.

To address this gap, recent works enhance instruction-tuning data by utilizing synthetic dialogues or extracting data from patient records [4, 33]. Frameworks like DoctorAgent-RL [34] and MedAgentSim [35] further employ RL and role-play strategy to optimize interactive diagnosis. However, they primarily target online consultation scenarios where patient information remains static. In real-world practice, the physician’s role extends to recommending examinations, facing a long-term dynamic environment with evolving patient conditions. We bridge this by training agents within a virtual environment, fostering dynamic ability to order test.

Main Contribution A diagnostics world model serving as a virtual clinical environment for end-to-end agentic RL.

- At the core of our method is the world model, DiagGym, a fine-tuned LLM that goes beyond replaying historical records, and generates dynamic results for any requested examination. Unlike prior role-playing simulators such as AgentClinic [17], AgentHospital [36], and SDBench [37], which are constrained by static, pre-collected data, DiagGym can generate novel, clinically plausible trajectories beyond the original patient records, serving as a dynamic virtual clinical environment for clinical agent evolving.

First end-to-end RL platform for interactive diagnostic agents. This paper provides an end-to-end, multi-turn RL framework in which agents interact iteratively with the clinical environment until a final diagnosis is reached. Agents actively explore diverse diagnostic trajectories, optimizing for both diagnostic accuracy and efficiency in examination recommendation. This allows them to handle complex, uncommon, or evolving patient scenarios that static supervised approaches cannot cover. We believe our approach provides the community with a robust resource for developing, testing, and comparing interactive diagnostic agents in a dynamic long-term patient management manner, moving beyond the more commonly considered yet less clinically challenging static consultation scenarios for clinical LLMs.

Interactive and exploratory dynamic diagnostic reasoning. Our final agent, DiagAgent, achieves substantial improvements over all evaluated LLMs and agentic systems. Reinforcement learning consistently improves LLMs’ competence in planning and managing interactive diagnostic trajectories, addressing the shortcomings of static fine-tuning. This dynamic, end-to-end training paradigm cultivates robust interactive competencies that are essential for real-world clinical deployment.

DiagBench, a comprehensive multi-center benchmark focusing on interactive trajectories with fine-grained physician-written rubrics. Breaking away from single-center limitations, DiagBench integrates data from four distinct clinical sources: MIMIC-IV, PMC-OA case reports, MTSamples, and DDXPlus. The benchmark comprises a total of 2.2K physician-validated cases. Furthermore, to enable granular, process-oriented assessment, 399 sampled cases are annotated with 3.3K physician-written rubrics that assign weighted points to critical steps in the reasoning process, offering deeper insights into how a diagnosis is progressively reached beyond final diagnosis accuracy.

Key Findings Better alignment with dynamic decision-making. Reinforcement learning consistently improves LLMs’

competence in planning and managing interactive diagnostic trajectories, addressing shortcomings of static fine-tuning. This dynamic, end-to-end training paradigm cultivates robust interactive competencies that are essential for real-world clinical deployment.

Superior to supervised fine-tuning (SFT). Across scales (7B-14B) and families (Qwen2.5, Llama3.1), diagnostic agent trained with reinforcement learning outperforms SFT by significant margins. This superiority is evident in interactive diagnosis tasks, where DiagAgent’s self-exploration fosters adaptability to evolving patient scenarios, leading to higher accuracy, efficiency, and robustness compared to SFT’s limitations in handling incomplete or atypical information.

Dependence on base model quality. The intrinsic capability of the foundation model strongly shapes DiagGym’s upper performance bound. While DiagGym delivers robust gains even for moderate-scale (7B–14B) models, continuing to scale up to larger foundation models could unlock even greater advancements, suggesting a promising path for enhancing diagnostic agents.

###### Limitations and Future Work

First, we acknowledge that DiagGym holds potential for further optimization to achieve higher fidelity. Currently, the system relies primarily on MIMIC-IV due to the scarcity of high-quality longitudinal records, which introduces potential biases specific to ED and ICU settings. Additionally, DiagGym operates as a text-based probabilistic model rather than a clearly verified disease mechanism simulator. While our extensive out-of-domain evaluation demonstrates that DiagGym, as a simulated RL environment, already enables the trained DiagAgent to achieve generalized, state-of-the-art diagnostic capabilities, future iterations could be significantly enhanced by incorporating cross-center data and and clinical knowledge priors, thereby facilitating the training of more robust interactive diagnostic agents.

Secondly, the models evaluated in this work are relatively modest in scale (up to 14B parameters), which may constrain the framework’s full potential. Larger foundation models, such as DeepSeek-v3, GPT-OSS-120B, could yield qualitative leaps in performance by enhancing inherent reasoning capabilities and exploratory depth. With sufficient model capacity, both DiagGym and DiagAgent could be further enhanced, culminating in higher-fidelity clinical environment simulations and superior interactive diagnostic capabilities.

Thirdly, we note that DiagAgent’s absolute scores on the rubric-based benchmark is modest. This is because the physician-authored rubrics reflect real-world clinical practice by awarding points for immediate treatment and patient management actions (e.g., “prepare for emergency transfer”). As our framework deliberately focuses on the diagnostic task, our agent was not trained to perform these out-of-scope therapeutic interventions. Extending the agent’s capabilities to integrate long-term management with timely treatment planning, interleaved with the diagnostic procedure, represents a clear direction for aligning clinical LLMs with practical usage demands in future work.

Finally, a key advantage of RL is its ability to learn without direct process supervision, enabling the emergence of strategies that may transcend current human capabilities [38, 39]. This is particularly critical in healthcare, where interactive diagnosis remains a complex, subjective challenge. By employing a diagnostics world model to facilitate RL training, our framework allows agents to explore optimal diagnostic pathways from extensive data, thereby offering the potential to refine physician decision-making trajectories. While constraints on training scale and human evaluation resources prevented a demonstration of this capability in the current study, we highlight this as a promising direction for future work to further validate.

###### 5 Methods

In this section, we provide additional details on the training process of DiagGym and DiagAgent, as well as the baseline methods used for comparison.

###### 5.1 DiagGym Training

In this section, we present the detailed training of DiagGym. Data Construction As shown in Figure 6a, to train DiagGym, we constructed a dataset of patient EHRs derived from MIMIC-IV.

- a. DiagGym Data Construction
- b. DiagGym Training

|MIMIC-IV microbiologyevents<br><br>N = 3,228,714| |
|---|---|
| | |

|MIMIC-IV labevents<br><br>N = 118,171,368| |
|---|---|
| | |

|MIMIC-IV radiology<br><br>N = 66,542,177| |
|---|---|
| | |

|MIMIC-IV discharge note<br><br>N = 331,794| |
|---|---|
| | |

|Mapping examinations to each admission (Select examinations conducted 24 hours before admission)| | |
|---|---|---|
| | | |

Physical examination performed?

|97.42 per case| |
|---|---|
| | |

|5.36 per case| |
|---|---|
| | |

|1.67 per case| |
|---|---|
| | |

N=319,244 Discharge Notes

|Remove duplicate examination for each admission (Only keep the first occurrence of each examination for each note)| | | |
|---|---|---|---|
| | | | |

Diagnosis not in current/past medical history?

|26.48 per case| |
|---|---|
| | |

|2.12 per case| |
|---|---|
| | |

|1.09 per case| |
|---|---|
| | |

N=114,239 Discharge Notes

|Reformat physical examination into Json format|
|---|

|Merge sub-events for each admission| | | |
|---|---|---|---|
| | | | |

Merge according to admission ID

|114,239 verified discharge notes, with 8.82 physical examinations, 24.82 lab events, 1.97 microbiology events, 0.99 radiology per note.|
|---|

Simulated Results

###### Ground Truth

PE: General

MRSA SCSREEN Lactate Dehydrogenase

#### …

DiagGym

Alert, oriented, no acute distress

No MRSA SCREEN

Value: 133 IU/L;

Value: 131 IU/L;

|Examination Query Examination Result|
|---|

Patient Profile Past Examinations Examination Query

- Figure 6 | Overview of DiagGym data construction and training pipeline. a shows the process for constructing the DiagGym training dataset. b illustrates the pipeline for DiagGym training.

Each patient’s EHR was reorganized into two components: (i) patient profile, (ii) time-ordered examination set. The pipeline was based on MIMIC-CDM [14], but extended to cover a broader range of diseases.

We first process the MIMIC-IV discharge notes. Leveraging their structured format, we applied heuristic string matching to extract the patient profile. Specifically, a patient profile was composed of content under the headings ‘physical examination’, ‘chief complaint’, ‘current medical history’, ‘past medical history’, ‘social history’, ‘family history’, and, most critically, the final diagnosis listed under the ‘discharge diagnosis’ heading.

Next, we apply a two-step filtering process: (i) cases without physical examination records are excluded; (ii)

DeepSeek-V3, instructed with prompt 1, is used to remove cases where the discharge diagnosis appeared in either the past medical history or the current medical history. Such cases often involve transfers with established diagnoses and typically lack diagnostically relevant examinations.

We then construct a time-ordered examination set for each patient, with each examination in the set comprising the queried examination item and its corresponding results. First, the previously extracted physical examination text from discharge notes is reformatted into a structured tabular format using DeepSeek-v3 with prompt 2, ensuring consistency with other MIMIC-IV examination records. Following MIMIC-CDM, this physical examination is designated as the initial test in the examination set. Next, we append laboratory results, microbiological examinations, and radiological records conducted within one day prior to admission. The oneday time interval is selected because examinations performed earlier generally have limited diagnostic relevance. Laboratory data is obtained from labevents.csv, microbiology data from microbiologyevents.csv, and radiology from radiology.csv.

For laboratory and microbiology data, we use the original structured records but standardized examination item names using the MIMIC-CDM mapping table to group the linked items (e.g., “red blood cell count” under the broader “complete blood count”). Radiology entries are supplemented with missing names extracted by string-matching from the ‘EXAMINATION’ section of reports. All examination entries in MIMIC-IV contain timestamps, enabling accurate chronological ordering. For repeated pre-admission examinations, we follow [14] and prioritize the earliest record to preserve the original clinical state at the time of first assessment, thereby avoiding the influence of subsequent interventions.

Finally, we split the restructured EHR dataset into training and testing sets. The resulting dataset consists of 118,478 patient EHRs, with each case containing the patient profile and a time-ordered set of examinations. Of these, 114,239 EHRs are used for the world model training, where the model is tasked with autoregressively reconstructing the examination results recorded in the examination set. These training cases span 4,897 distinct diseases. On average, each training patient underwent 29 examinations, including 26 laboratory tests, 2 microbiological tests, and 1 radiological test. The remaining 4,239 cases are reserved for evaluation, covering 863 distinct diseases. However, given the high cost of evaluation — due to the use of various commercial models – we adopted a disease-wise sampling strategy. Specifically, we selected one representative case for each disease, resulting in a balanced test set of 863 cases, with each case corresponding to a unique disease.

Training Details Leveraging the constructed data, we train a diagnostics world model, DiagGym, with text generation loss,

- as introduced in the former Section 2. We frame its training as an auto-regressive text generation task, straightforwardly viewing all examination results as free text, regardless of whether they are numerical or textual. The loss function, a standard token-wise auto-regressive objective inspired by GPT-series models [40], minimizes the negative log-likelihood of the ground-truth examination result eˆt tokens:

T

Lrecon = −

t=1 i

log Φenv(ˆeit | at+1,Et,B), (9)

where eˆit denotes the i-th token of the examination result et and Et denotes the former examination records in the examination set.

Implementation Details. For our experiments, we initialized Φenv from Qwen2.5-Instruct-7B. Training was performed on eight NVIDIA A100 GPUs using the Transformers1 library with DeepSpeed ZeRO Stage 2 for efficient distributed optimization. Models were trained for 15 epochs, with convergence achieved within this period. The learning rate was 4 × 10−5, and the maximum input length was 8,192 tokens.

###### 5.2 DiagAgent Training

In this section, we provide a detailed procedure for DiagAgent training. Data Construction

1https://github.com/huggingface/transformers

###### a. DiagAgent Data Construction

|Current diagnosis: RLQ abdominal pain Examination Query: CBC<br><br>Reason: Assess for infection<br><br>Examination results: neutrophil count (83.6%)<br><br>Initial inquiry:<br><br>Patient Information: Female Chief Complaint: Right lower abdominal pain …<br><br>Multi-turn diagnostic trajectories:<br><br>[Figure 44]<br><br>Final diagnosis:<br><br>Right hemorrhagic cyst<br><br>[Figure 45]<br><br>Final Case Example|
|---|

|DiagGym Trainset N = 110, 000 cases| |
|---|---|
| |Random Sample<br><br>|

Diagnosis occur in case summary?

|41,245 Cases| |
|---|---|
| | |

[Figure 46]

No

17,180 cases

|Structure into<br><br>diagnostic trajectories| |
|---|---|
| | |

Yes

Adhere to data format?

41,245 generated cases

16,270 cases, Train/Test: 15,324 / 946

b. DiagAgent Training

Different Diagnostic Trajectories

###### Reward 0.7

|…<br><br>| |
|---|---|
| | |

[Figure 47]

[Figure 48]

Initial Inquiry

- Rollout 1
- Rollout 2

Reward Calculation

✓ Examination Recommendation Reward: Recommended examination adequate?

|…<br><br>| |
|---|---|
| | |

[Figure 49]

0.3

✓ Final Diagnosis Reward: Final diagnosis correct?

##### … …

✓ Interaction Turns Reward: Interaction termination before max turns?

Rollout N

|…<br><br>| |
|---|---|
| | |

[Figure 50]

DiagAgent

0.1

| | | |
|---|---|---|
|DiagGym<br><br>| | |

|Examination Query<br><br>Examination Results<br><br>Final Diagnosis<br><br>Initial Inquiry|
|---|

Patient Profile

Policy Evolving

（Optimization）

- Figure 7 | Overview of DiaAgent data construction and training pipeline. a shows the data construction process for the DiagAgent. b illustrates the training pipeline for the diagnostic agent, where the agent interacts with the virtual clinical environment, DiagGym, explores different diagnostic trajectories, and evolves its policy based on reward scores.

The diagnostic agent Φdiag is trained using a reformatted version of the DiagGym training dataset, organized into a multi-turn diagnostic trajectory format, as shown in Figure 7a.

We use DeepSeek-v3, guided by prompt 3, to generate three tightly connected elements for each case based on the existing restructured EHR dataset and the original discharge notes:

- • Initial inquiry: A refined, structured summary of the patient’s medical history before admission, covering the chief complaint, history of present illness, past medical history, family history, and other relevant information. While this is similar to the previously constructed patient profile, critically, it does not include the final diagnosis information. The inquiry is further refined by LLMs to align with real inquiry formats. This serves as the starting point for the dialogue.
- • Referenced multi-turn diagnostic trajectory: A step-by-step diagnostic trajectory is reformed from the time-ordered examination chain, with DeepSeek-v3 prioritizing the most informative tests related to the final diagnosis and omitting non-essential routine ones. Each step in the trajectory consists of: (1) Current Preliminary Diagnosis: An initial hypothesis based on prior data. (2)Next Recommended Examination with Rationale: A suggested test accompanied by a detailed explanation for its necessity. (3) Corresponding Test Results: The outcome of the recommended examination. The first two components are considered the agent’s response, while the third represents the clinical environment’s

- feedback — effectively, the user’s input in the multi-turn dialogue. The trajectory concludes with the final diagnostic decision at the end-turn. Although the preliminary diagnosis and the rationale for recommending examinations are generated with the assistance of LLMs, the order of examination items and their corresponding results are directly extracted from real EHRs. Therefore, this trajectory is viewed as a referenced diagnostic pathway, grounded in the recommendations of real physicians.
- • Final diagnosis: The final diagnostic decision for the case. Since the original final diagnosis recorded in the patient profile may sometimes include multiple conditions, the LLM is prompted to construct a self-contained process focused on a single primary condition. This process also includes selecting relevant examinations from the entire chain of examinations in the previous multi-turn diagnostic trajectory construction.

All three components are generated in a single pass to maintain contextual consistency.

To ensure data quality and prevent leakage of the final diagnosis into the initial inquiry, we applied a two-stage filtering process using prompt 4. This step removes instances where the model may have inadvertently introduced the final diagnosis into earlier parts of the text due to hallucinations.

Finally, we converted the multi-turn diagnostic trajectories into a structured dialogue format following former LLM multi-turn datasets [41]. In subsequent turns, each step in the trajectory is structured as an assistant message that includes the preliminary diagnosis and the next recommended examination with its rationale, along with a user message that provides the results of the recommended examinations. The last assistant turn contains only the final diagnosis and its rationale. The resulting dataset comprised 15,324 interactive diagnostic trajectories used for training.

###### Training Details

As introduced in Section 2, we train DiagAgent Φdiag with end-to-end multi-turn RL, with a classical two-stage paradigm [42]: an initial cold-start phase and a main RL phase.

Cold start. This phase mirrors standard instruction tuning [43]. The model is optimized with an autoregressive text generation loss computed only over tokens labeled as assistant response in the dialogues:

Lcold = −

log Φdiag(yi | y≤i−1), (10)

yi∈assitant

where yi is a token, y≤i−1 is the preceding context, and the loss is restricted to assistant response tokens. The goal of cold start is to initialize the LLM to produce well-formatted, contextually appropriate responses before interacting with the environment with RL. For this stage, we use 1,000 manually selected high-quality cases from the training set, free from formatting issues or diagnostic reasoning errors.

Reinforcement learning. After cold start, we optimize Φdiag with the GRPO algorithm [42] over the full training set. At rollout start, the agent receives the initial inquiry as the initial state, s0 = I and iteratively interacts with the virtual environment Φenv until it decides sufficient information has been gathered for a final diagnosis (Section 2 ). The policy is trained to maximize the following reward:

R = λ1rdiag + λ2rexam + λ3rturn, (11)

with λ1,λ2,λ3 as hyper-parameter weights. The final diagnosis reward evaluates the accuracy of the predicted diagnosis results, formulated as:

1, if dˆ= d 0, otherwise

rdiag =

, (12)

where dˆ is the predicted disease. We adopt Qwen2.5-72B with prompt 12 to measure semantic equivalence of dˆand d. The examination recommendation reward measures the alignment between the agent’s recommended examinations Eˆ and the reference set E from the curated multi-turn trajectory based on real EHRs. We

adopt the F1 score to measure their similarity, formulated as:

|Eˆ ∩ E| |Eˆ| + |B|

rexam = F1(E,Eˆ ) = 2 ·

. (13)

We calculate the union set of the two by instructing Qwen2.5-72B with prompt 10 and prompt 11 to search through the Eˆ and E respectively. The last interaction turn penalty reward penalizes excessive rounds of dialogue without termination. To prevent unnecessarily long dialogues, we impose a maximum number of iterative turns, Tmax. If the model cannot finish the diagnosis within this limitation, it will achieve a lower reward as:

0.1, if T ≤ Tmax 0, otherwise

rturn =

, (14)

where T denotes the total number of turns used in a certain rollout. Through iteratively optimizing this reward, Φdiag learns to manage accurate, efficient diagnostic trajectory reasoning while minimizing unnecessary examination steps.

Implementation details. In our experiments, we selected Qwen2.5-Instruct-7B, Qwen2.5-Instruct-14B, and Llama3.1-Instruct-8B as initialization models for further training. The same training strategy was applied to all models. For cold start settings, we utilize the Transformers2 framework and employed DeepSpeed ZeRO Stage 2 for efficient multi-GPU training. All models share the same hyperparameters: a maximum sequence length of 8192 tokens, a learning rate of 1 × 10−5, and training on eight NVIDIA A100 GPUs. Training is conducted for three epochs, at which point the training loss plateaued, confirming convergence for each model.

For the reinforcement learning setting, we modify Verl3 to enable interactive training. During training, the DiagGym was deployed on two nodes using vLLM4. The Qwen2.5-Instruct-72B model serves as the judge and is deployed on a separate node. The weighted hyperparameters λ1,λ2,λ3 are set as 1, 0.5, 1, respectively. These values were selected based on the magnitude of reward components and preliminary experiments to ensure stability, as exhaustive grid search was computationally infeasible. The maximum interactive turns are set to 12, a threshold empirically derived from the MIMIC-IV diagnostic trajectory distribution to cover realistic workflows while preventing redundant loops. RL training is performed across four nodes in total, with each node equipped with eight NVIDIA A100 GPUs. We set the training batch size to 512, the maximum response length to 8192 tokens, the learning rate to 1 × 10−6, and the rollout number to 5. Each model is trained for 200 steps, where convergence was confirmed by the stabilization of the average cumulative reward per rollout.

###### 5.3 DiagBench Construction

In this section, we describe the construction of DiagBench, a comprehensive multi-center benchmark designed to evaluate multi-turn diagnostic interaction trajectories across diverse clinical settings. As illustrated in

- Figure 8, DiagBench integrates data from four distinct sources to ensure robust assessment of generalization:

- • MIMIC-IV: Derived from the same source as the training set, representing critical care and emergency medicine scenarios.
- • PMC-OA: Curated from open-access biomedical case reports, representing complex and rare disease presentations found in medical literature. To rigorously prevent data leakage and assess generalization on unseen data, we strictly selected case reports published after January 2025.
- • MTSamples: Sourced from transcribed medical reports, covering a wide range of outpatient and specialty clinic encounters.
- • DDXPlus: A dataset focused on differential diagnosis logic, providing structured cases for reasoning evaluation.

- 2https://github.com/huggingface/transformers
- 3https://github.com/volcengine/verl
- 4https://github.com/vllm-project/vllm

The process involved two main stages: initial dataset curation and the development of a rubric-based evaluation framework.

Initial Dataset Curation: The construction pipeline is illustrated in Figure 8. For MIMIC-IV, the test cases are generated using the same pipeline as our training data. We applied a standard train/test split to obtain them, forming an in-domain evaluation for our DiagAgent. For the other out-of-domain datasets, the construction pipeline aligns generally with MIMIC-IV, but required specific adaptations to accommodate their heterogeneity:

- • Patient Profile Generation: Unlike MIMIC-IV, other datasets often lack structured patient profiles. We utilize DeepSeek-v3 with prompt 16 to extract and synthesize a comprehensive profile from raw text to initialize the simulator.
- • Diagnostic Trajectory Generation: We generate multi-turn diagnostic trajectories utilizing the same prompt structure as the MIMIC-IV pipeline (prompt 3). However, these datasets frequently contain rare specialized examinations. To address this, we append a strict constraint: “You can only state Lab Events, Microbiology Events, or Radiology.” This limits recommendations to conventional diagnostic modalities, thereby preventing clinically outlier cases.
- • Key Examinations Extraction for Hit Ratio: To evaluate examination recommendation performance under single-turn evaluation setting, we extract a list of all relevant examinations mentioned in the raw text using prompt 17.
- • Logical Consistency Check: We implement an additional filtering step with prompt 18 to discard cases where the synthesized patient profile conflicted with the diagnostic trajectory, ensuring the logical flow of the case.

To ensure clinical validity, each case undergoes a rigorous review by a physician. The reviewer is provided with the patient’s initial inquiry and the complete reference diagnostic trajectory. They are tasked with evaluating: (1) the clinical appropriateness of each analysis and recommended examination at every step, and (2) the overall plausibility of the case. For the out-of-domain datasets, they are additionally required to review the patient profile and the extracted key examinations to confirm relevance to the patient’s situation. This curation process results in a final, validated set of cases for automated evaluation.

Figure 8 illustrates the exclusion rates at each step during data construction, highlighting the distinct characteristics of each source. PMC-OA case reports are clinically complex and often detail severe comorbidities. While we sample 3,000 recent reports, this complexity makes structured extraction challenging. Consequently, a significant portion is filtered during the profile and trajectory extraction phases or rejected during strict physician review, resulting in 631 final cases. MTSamples transcripts are often concise or contain incomplete information. This leads to a high filtration rate at the initial “Patient Profile Extraction” stage, where nearly 50% of the raw inputs are discarded due to insufficient details for simulation initialization. Ultimately, 379 high-quality cases are retained. DDXPlus, designed for differential diagnosis, features structured and clear logic. Consequently, the construction is highly stable, with the “Reference Trajectory Extraction” step retaining almost all candidates. 497 cases are selected for the final benchmark. Combined with the 750 cases from MIMIC-IV, DiagBench provides a diverse evaluation suite.

Rubric-based Evaluation Framework In addition to automated metrics, and inspired by HealthBench [30], we develop a rubric-based framework to comprehensively assess the quality of the diagnostic inference process.

First, to ensure balanced representation, we perform stratified random sampling across all four data sources in DiagBench, selecting 100 clinical cases from each center. Two physicians are independently provided with each case’s initial patient inquiry and complete diagnostic trajectory. Each physician then independently authors a set of process-oriented rubrics they consider critical for evaluating the quality of the diagnostic journey. The rubric design explicitly prioritizes the reasoning process, including history taking, hypothesis generation, and test ordering, rather than focusing solely on the accuracy of the final diagnosis.

Subsequently, a third physician conducts a secondary review of the rubrics. This reviewer’s role is to ensure the framework holistically covers the end-to-end diagnostic pipeline and that individual rubrics are coherent and well-defined. Following this screening, cases with insufficient rubric coverage are excluded, yielding a final

###### In-domain Out-of-domain

|MIMIC-IV N = 331,794| |
|---|---|
| |DiagAgent data construction pipeline|

|PMC-OA N= 324,501| |
|---|---|
| | |

|MTSamples N=4,999| |
|---|---|
| | |

|DDXPlus N=134,529| |
|---|---|
| |Random|

Publication After 2025.1

|18,608 case reports| |
|---|---|
| |Random|

Sample Sample

|3,000 case reports| |
|---|---|
| | |

|4,999 transcripts| |
|---|---|
| | |

|2,000 cases| |
|---|---|
| | |

|Extract Patient Profile for Simulator Initialization<br><br>(Exclude if doesn’t have enough information for patient profile)|
|---|

|1,789 Cases| |
|---|---|
| | |

|2,391 Cases| |
|---|---|
| | |

|1,726 Cases| |
|---|---|
| | |

|Structure into diagnostic trajectories (Exclude if doesn’t have enough information for diagnostic trajectories)| | | |
|---|---|---|---|
| | | | |

|1,595 Cases| |
|---|---|
| | |

|1,573 Cases| |
|---|---|
| | |

|1,725 Cases| |
|---|---|
| | |

|Remove cases where diagnosis occur in case summary| | | |
|---|---|---|---|
| | | | |

|1,379 Cases| |
|---|---|
| | |

|912 Cases| |
|---|---|
| | |

|1,209 Cases| |
|---|---|
| | |

|Discard cases with misalignment between profile and trajectory| | | |
|---|---|---|---|
| | | | |

|1,229 Cases| |
|---|---|
| | |

|742 Cases| |
|---|---|
| | |

|1200 Cases| |
|---|---|
| | |

tion

|Extract key Examination List for Ground Truth (Retain cases with 3-30 interactive diagnostic turns)| | | |
|---|---|---|---|
| | | | |

|Train/Test Split|
|---|

|946 Cases| |
|---|---|
| | |

|825 Cases| |
|---|---|
| | |

|390 Cases| |
|---|---|
| | |

|508 Cases| |
|---|---|
| | |

|Validated by Senior Physicians| | | | |
|---|---|---|---|---|
| | | | | |

|750 Cases|
|---|

|631 Cases|
|---|

|379 Cases|
|---|

|497 Cases|
|---|

Figure 8 | Overview of the DiagBench construction pipeline. To ensure robust evaluation across diverse clinical settings, the benchmark integrates data from four distinct sources: MIMIC-IV, PMC-OA, MTSamples, and DDXPlus. As our training data originates from MIMIC-IV, the benchmark can be further divided into in-domain and out-of-domain subsets for the final DiagAgent evaluation. The pipeline consists of three key phases: (1) Structured Extraction, where LLMs synthesize patient profiles and reference trajectories from raw text; (2) Automated Filtration, which rigorously removes cases with data leakage or logical inconsistencies; and (3) Human Validation, where senior physicians verify the clinical plausibility of the final dataset. The counts in the boxes indicate the number of cases retained at each stage.

consolidated set of 399 cases (99 from MIMIC-IV and 100 from each of the three OOD centers) annotated with a total of 3,318 physician-authored rubrics.

Finally, a fourth physician is tasked with assigning an importance weight to each rubric on a scale from 0 to 10 (where 10 signifies an essential, non-negotiable criterion and 0 indicates a non-informative one). As a result, rubrics targeting high-impact clinical steps, such as appropriate test ordering and effective diagnostic narrowing, receive higher weights. Conversely, ancillary actions, like scheduling a follow-up without a clear clinical rationale, are assigned lower weights. Illustrative examples of this rubric-based evaluation are provided in Supplementary Figure 4 and Supplementary Figure 5.

- 5.4 Baselines Here, we introduce the baseline LLMs involved in our experiments:

- • Qwen2.5 [44] and Qwen3 [45] are series of high-performance open-source language models developed by the Qwen team, available in variants ranging from 0.5 to 72 billion parameters. In this paper, we use Qwen2.5-7B-Instruct and Qwen2.5-14B-Instruct for training, and deploy Qwen2.5-72B-Instruct and Qwen3-235B-A22B locally for inference.
- • Llama3.1 and Llama3.3 [46] are series of language models developed by Meta AI and are among the most popular open-source large language models. In this paper, we utilize Llama-3.1-8B-Instruct for training and deploy Llama-3.3-70B-Instruct locally for inference.
- • OpenBioLLM [47] is an advanced open-source language model specifically designed for the biomedical domain, developed based on Llama3. In this paper, we utilize Llama3-OpenBioLLM-70B and deploy it locally.
- • Baichuan-M1 [48] is an advanced open-source medical language model developed by Baichuan Intelligence. It is the first language model in the industry designed and developed from scratch specifically for the medical field, demonstrating strong performance in medical applications. We utilize Baichuan-M1-14B-Instruct and deploy it locally.
- • Baichuan-M3 [49] is a next-generation medical-enhanced large language model developed by Baichuan Intelligence. Unlike traditional models focusing on static Q&A, Baichuan-M3 achieves native medical enhancement primarily through advanced reinforcement learning frameworks. These strategies enable the model to autonomously collect key information, build rigorous clinical reasoning paths, and dynamically suppress hallucinations in real-world scenarios. We utilize Baichuan-M3-235B and deploy it locally.
- • DeepSeek-V3 [39] is one of the most powerful open-source language models, developed by DeepSeek, with 671 billion parameters. In this paper, we use DeepSeek-V3-0324 and deploy it locally.
- • MedGemma [50] is a recent medical LLM developed by Google as a variant of the Gemma3 collection. Based on Gemma3, this model is specially optimized for the medical field and possesses multi-modal capabilities. In this paper, we utilize medgemma-27b-text-it and deploy it locally.
- • GPT-OSS [51] is an open-source large language model released by OpenAI, with strong reasoning ability. In this paper, we use the gpt-oss-120b and deploy it locally.
- • GPT-4o [52] is one of the most commonly used close-sourced language model developed by OpenAI. It is good at handling most daily tasks. In this paper, we utilize gpt-4o-2024-08-06 via API.
- • Claude-4 [53] is a high-performance large language model developed by Anthropic, optimized for coding and reasoning. In this paper, we use the claude-sonnet-4 via API.

Next, we will introduce the agentic systems involved in our experiments. Note that, in our experiment, all the agentic systems use DeepSeek-V3 as its base model.

- • MedAgents[54] is a multidisciplinary collaboration framework that requires large language models to assume the roles of medical experts from various specialties. The framework aggregates the experts’ opinions and summarizes them into a final report. This approach reveals the model’s knowledge across different domains and broadens its reasoning capabilities. In our implementation, we use DeepSeek-V3 as the base model.

- • MDAgents[55] is a framework designed to automatically assign collaboration structures. Different from MedAgents, MDAgents effectively assesses medical complexity and adapt to varing tasks accordingly. In our implementation, we use DeepSeek-V3 as the base model.

In the DiagGym evaluation, we consider prompting the open-source models as EHR wold model baselines to simulate the examination results, including DeepSeek-V3-671B, MedGemma-27B, Qwen2.5-7B, and Qwen2.572B. We exclude closed-source API-based LLMs due to their high latency and cost, which make them impractical for scalable RL training. For instance-wise metrics, these models adopt prompt 7 to generate examination results for comparison. For examination-wise metrics, which measures the distribution of given examinations, the models use prompt 8 for numerical examinations and prompt 9 for free-text examinations. We implement zero-shot inference for most settings, with the sole exception of prompt 8. Preliminary experiments indicated that under a zero-shot setting, baseline models frequently generated numerical results with inconsistent formatting or heterogeneous units, which severely impeded automated parsing. Consequently, we adopted a one-shot strategy specifically for this prompt to enforce syntactic constraints and ensure reliable quantitative evaluation.

In the DiagAgent evaluation, we include all the aforementioned basic LLMs and more complex agentic systems for comparison. We consistently apply a zero-shot prompting strategy across all models. Each system is instructed using prompt 5, which serves as a standardized test interface to unify the task definition and output format, ensuring fair comparison. For end-to-end evaluation on simulated cases, the models are required to decide whether to request an additional examination or to make a final diagnosis. For single-turn evaluation on real cases, the models are directly instructed to either request an examination or make a final diagnosis. In this setting, we modified prompt 5 by appending an explicit prompt to the end of the input in each turn, like “Next step you should query examination” or “Next step you should make final diagnosis”.

###### 6 Data Availability

The data source for this work is MIMIC-IV. Due to licensing restrictions, we are unable to directly open-source the dataset. However, we are actively communicating with the relevant parties regarding the possibility of making the dataset publicly available on https://physionet.org/.

###### 7 Code Availability All source codes of this paper have been released in https://github.com/MAGIC-AI4Med/DiagGym.

###### References

- [1] Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Mohamed Amin, Le Hou, Kevin Clark, Stephen R Pfohl, Heather Cole-Lewis, et al. Toward expert-level medical question answering with large language models. Nature Medicine, pages 1–8, 2025.
- [2] Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. Large language models encode clinical knowledge. Nature, 620(7972):172–180, 2023.
- [3] Sarah Sandmann, Stefan Hegselmann, Michael Fujarski, Lucas Bickmann, Benjamin Wild, Roland Eils, and Julian Varghese. Benchmark evaluation of deepseek large language models in clinical decision-making. Nature Medicine, pages 1–1, 2025.
- [4] Daniel McDuff, Mike Schaekermann, Tao Tu, Anil Palepu, Amy Wang, Jake Garrison, Karan Singhal, Yash Sharma, Shekoofeh Azizi, Kavita Kulkarni, et al. Towards accurate differential diagnosis with large language models. Nature, pages 1–7, 2025.
- [5] Farieda Gaber, Maqsood Shaik, Fabio Allega, Agnes Julia Bilecz, Felix Busch, Kelsey Goon, Vedran Franke, and Altuna Akalin. Evaluating large language model workflows in clinical decision support for triage and referral and diagnosis. npj Digital Medicine, 8(1):263, 2025.
- [6] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.
- [7] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [8] Chaoyi Wu, Weixiong Lin, Xiaoman Zhang, Ya Zhang, Weidi Xie, and Yanfeng Wang. Pmc-llama: toward building open-source language models for medicine. Journal of the American Medical Informatics Association, 31(9):1833–1843, 2024.
- [9] Pengcheng Qiu, Chaoyi Wu, Xiaoman Zhang, Weixiong Lin, Haicheng Wang, Ya Zhang, Yanfeng Wang, and Weidi Xie. Towards building multilingual language model for medicine. Nature Communications, 15(1):8384, 2024.
- [10] Pengcheng Qiu, Chaoyi Wu, Shuyu Liu, Weike Zhao, Zhuoxia Chen, Hongfei Gu, Chuanjin Peng, Ya Zhang, Yanfeng Wang, and Weidi Xie. Quantifying the reasoning abilities of llms on real-world clinical cases. arXiv preprint arXiv:2503.04691, 2025.
- [11] Zeming Chen, Alejandro Hernández Cano, Angelika Romanou, Antoine Bonnet, Kyle Matoba, Francesco Salvi, Matteo Pagliardini, Simin Fan, Andreas Köpf, Amirkeivan Mohtashami, et al. Meditron-70b: Scaling medical pretraining for large language models. arXiv preprint arXiv:2311.16079, 2023.
- [12] Bingning Wang, Haizhou Zhao, Huozhi Zhou, Liang Song, Mingyu Xu, Wei Cheng, Xiangrong Zeng, Yupeng Zhang, Yuqi Huo, Zecheng Wang, et al. Baichuan-m1: Pushing the medical capability of large language models. arXiv preprint arXiv:2502.12671, 2025.
- [13] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv e-prints, pages arXiv–2407, 2024.
- [14] Paul Hager, Friederike Jungmann, Robbie Holland, Kunal Bhagat, Inga Hubrecht, Manuel Knauer, Jakob Vielhauer, Marcus Makowski, Rickmer Braren, Georgios Kaissis, et al. Evaluation and mitigation of the limitations of large language models in clinical decision-making. Nature medicine, 30(9):2613–2622, 2024.

- [15] Yusheng Liao, Yutong Meng, Yuhao Wang, Hongcheng Liu, Yanfeng Wang, and Yu Wang. Automatic interactive evaluation for large language models with state aware patient simulator. arXiv preprint arXiv:2403.08495, 2024.
- [16] Shreya Johri, Jaehwan Jeong, Benjamin A Tran, Daniel I Schlessinger, Shannon Wongvibulsin, Leandra A Barnes, Hong-Yu Zhou, Zhuo Ran Cai, Eliezer M Van Allen, David Kim, et al. An evaluation framework for clinical use of large language models in patient interaction tasks. Nature medicine, 31(1):77–86, 2025.
- [17] Samuel Schmidgall, Rojin Ziaei, Carl Harris, Eduardo Reis, Jeffrey Jopling, and Michael Moor. Agentclinic: a multimodal agent benchmark to evaluate ai in simulated clinical environments. arXiv preprint arXiv:2405.07960, 2024.
- [18] Alistair EW Johnson, Lucas Bulgarelli, Lu Shen, Alvin Gayles, Ayad Shammout, Steven Horng, Tom J Pollard, Sicheng Hao, Benjamin Moody, Brian Gow, et al. Mimic-iv, a freely accessible electronic health record dataset. Scientific data, 10(1):1, 2023.
- [19] National Library of Medicine. Pmc open access subset [internet]. https://pmc.ncbi.nlm.nih.gov/tools/openftlist/,

2003. Bethesda (MD).

- [20] MTSamples. Transcribed medical transcription sample reports and examples. https://www.mtsamples.com/,

2023. Accessed: 2025-12-20.

- [21] Arsene Fansi Tchango, Rishab Goel, Zhi Wen, Julien Martel, and Joumana Ghosn. Ddxplus: A new dataset for automatic medical diagnosis. Advances in neural information processing systems, 35:31306–31318, 2022.
- [22] Mario Lucic, Karol Kurach, Marcin Michalski, Sylvain Gelly, and Olivier Bousquet. Are gans created equal? a large-scale study. Advances in neural information processing systems, 31, 2018.
- [23] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318, 2002.
- [24] Jinsung Yoon, Michel Mizrahi, Nahid Farhady Ghalaty, Thomas Jarvinen, Ashwin S Ravi, Peter Brune, Fanyu Kong, Dave Anderson, George Lee, Arie Meir, et al. Ehr-safe: generating high-fidelity and privacy-preserving synthetic electronic health records. NPJ digital medicine, 6(1):141, 2023.
- [25] Jin Li, Benjamin J Cairns, Jingsong Li, and Tingting Zhu. Generating synthetic mixed-type longitudinal electronic health records for artificial intelligent applications. NPJ digital medicine, 6(1):98, 2023.
- [26] Chao Yan, Yao Yan, Zhiyu Wan, Ziqi Zhang, Larsson Omberg, Justin Guinney, Sean D Mooney, and Bradley A Malin. A multifaceted benchmarking of synthetic electronic health record generation models. Nature communications, 13(1):7609, 2022.
- [27] François Remy, Kris Demuynck, and Thomas Demeester. Biolord-2023: semantic textual representations fusing large language models and clinical knowledge graph insights. Journal of the American Medical Informatics Association, 31(9):1844–1855, 2024.
- [28] Yu Yu, Weibin Zhang, and Yun Deng. Frechet inception distance (fid) for evaluating gans. China University of Mining Technology Beijing Graduate School, 3(11), 2021.
- [29] Utkarsh Ojha, Yijun Li, Jingwan Lu, Alexei A Efros, Yong Jae Lee, Eli Shechtman, and Richard Zhang. Few-shot image generation via cross-domain correspondence. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10743–10752, 2021.
- [30] Rahul K Arora, Jason Wei, Rebecca Soskin Hicks, Preston Bowman, Joaquin Quiñonero-Candela, Foivos Tsimpourlas, Michael Sharman, Meghan Shah, Andrea Vallone, Alex Beutel, et al. Healthbench: Evaluating large language models towards improved human health. arXiv preprint arXiv:2505.08775, 2025.

- [31] Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Le Hou, Kevin Clark, Stephen Pfohl, Heather Cole-Lewis, Darlene Neal, et al. Towards expert-level medical question answering with large language models. arXiv preprint arXiv:2305.09617, 2023.
- [32] Pawel Renc, Michal K Grzeszczyk, Nassim Oufattole, Deirdre Goode, Yugang Jia, Szymon Bieganski, Matthew McDermott, Jaroslaw Was, Anthony E Samir, Jonathan W Cunningham, et al. Foundation model of electronic medical records for adaptive risk estimation. arXiv preprint arXiv:2502.06124, 2025.
- [33] Tao Tu, Mike Schaekermann, Anil Palepu, Khaled Saab, Jan Freyberg, Ryutaro Tanno, Amy Wang, Brenna Li, Mohamed Amin, Yong Cheng, et al. Towards conversational diagnostic artificial intelligence. Nature, pages 1–9, 2025.
- [34] Yichun Feng, Jiawei Wang, Lu Zhou, Zhen Lei, and Yixue Li. Doctoragent-rl: A multi-agent collaborative reinforcement learning system for multi-turn clinical dialogue. arXiv preprint arXiv:2505.19630, 2025.
- [35] Mohammad Almansoori, Komal Kumar, and Hisham Cholakkal. Self-evolving multi-agent simulations for realistic clinical interactions. arXiv preprint arXiv:2503.22678, 2025.
- [36] Junkai Li, Yunghwei Lai, Weitao Li, Jingyi Ren, Meng Zhang, Xinhui Kang, Siyu Wang, Peng Li, Ya-Qin Zhang, Weizhi Ma, et al. Agent hospital: A simulacrum of hospital with evolvable medical agents. arXiv preprint arXiv:2405.02957, 2024.
- [37] Harsha Nori, Mayank Daswani, Christopher Kelly, Scott Lundberg, Marco Tulio Ribeiro, Marc Wilson, Xiaoxuan Liu, Viknesh Sounderajah, Jonathan Carlson, Matthew P Lungren, et al. Sequential diagnosis with language models. arXiv preprint arXiv:2506.22405, 2025.
- [38] David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. nature, 529(7587):484–489, 2016.
- [39] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.
- [40] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.
- [41] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [42] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.
- [43] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.
- [44] Qwen Team. Qwen2.5: A party of foundation models, September 2024.
- [45] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [46] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.
- [47] Malaikannan Sankarasubbu Ankit Pal. Openbiollms: Advancing open-source large language models for healthcare and life sciences. https://huggingface.co/aaditya/OpenBioLLM-Llama3-70B, 2024.

- [48] Huozhi Zhou Bingning Wang, Haizhou Zhao et al. Baichuan-m1: Pushing the medical capability of large language models. arXiv preprint arXiv:2502.12671, 2025.
- [49] Baichuan M3 Team. Baichuan-m3: Modeling clinical inquiry for reliable medical decision-making, 2025.
- [50] Andrew Sellergren, Sahar Kazemzadeh, Tiam Jaroensri, Atilla Kiraly, Madeleine Traverse, Timo Kohlberger, Shawn Xu, Fayaz Jamil, Cían Hughes, Charles Lau, et al. Medgemma technical report. arXiv preprint arXiv:2507.05201, 2025.
- [51] OpenAI. Introducing gpt-oss. https://openai.com/index/introducing-gpt-oss/. Accessed: 2025-08-08.
- [52] OpenAI. Hello gpt-4o, 2025. Accessed: 2025-02-27.
- [53] Anthropic. Introducing claude 4. https://www.anthropic.com/news/claude-4. Accessed: 2025-08-08.
- [54] Xiangru Tang, Anni Zou, Zhuosheng Zhang, Ziming Li, Yilun Zhao, Xingyao Zhang, Arman Cohan, and Mark Gerstein. Medagents: Large language models as collaborators for zero-shot medical reasoning. arXiv preprint arXiv:2311.10537, 2023.
- [55] Yubin Kim, Chanwoo Park, Hyewon Jeong, Yik Siu Chan, Xuhai Xu, Daniel McDuff, Hyeonhoon Lee, Marzyeh Ghassemi, Cynthia Breazeal, and Hae Won Park. Mdagents: An adaptive collaboration of llms for medical decision-making, 2024.

###### 8 Acknowledgments

This work is supported by the National Key R&D Program of China (No. 2022ZD0160702), and the Scientific Research Innovation Capability Support Project for Young Faculty (ZY-GXQNJSKYCXNLZCXM-I22).

###### 9 Author Contributions

All listed authors clearly meet the ICMJE 4 criteria. P.Q., C.W., and J.L. contribute equally to this work. Y.Z. and W.X. are the corresponding authors. Specifically, P.Q., C.W., J.L., Q.Z., Y.L., H.W., Y.Y., Q.F., S.Z., J.W., J.G., Y.W., Y.Z., and W.X. all make contributions to the conception or design of the work, and P.Q., C.W., and J.L. further perform acquisition, analysis, or interpretation of data for the work. In writing, P.Q., C.W., and J.L. draft the work. Q.Z., Y.L., H.W., Y.Y., Q.F., S.Z., J.W., J.G., Y.W., Y.Z., and W.X. review it critically for important intellectual content. All authors approve of the version to be published and agree to be accountable for all aspects of the work to ensure that questions related to the accuracy or integrity of any part of the work are appropriately investigated and resolved.

###### A Supplementary

- A.1 Evaluation Metrics In this section, we provide a detailed calculation format for our adopted evaluation metrics.

###### A.1.1 DiagGym Evaluation

We evaluate the DiagGym (Φenv) from two perspectives, i.e., instance-wise and examination-wise.

Instance-wise Metrics. We employ a hybrid evaluation strategy combining 2 traditional quantitative metrics with 2 LLM-based semantic assessments to evaluate the quality of generated examination sequences

- at the instance level:

- • Normalized Mean Absolute Error (NMAE). Designed specifically for numeric laboratory results, Designed specifically for numeric laboratory results, this metric quantifies the deviation between the simulator’s predicted values and the ground truth. Since different laboratory tests operate on vastly different scales (e.g., pH ≈ 7.4 vs. Platelets ≈ 300), we normalize the absolute error relative to the global data range of each specific test type. Formally, for a specific examination type k (e.g., Glucose), the NMAE is calculated as:

NMAEk =

1 Nk

Nk

i=1

|yi,k − yˆi,k| Vmax,k − Vmin,k

, (15)

where yi,k and yˆi,k are the ground truth and predicted values for the i-th instance, and Vmax,k and Vmin,k represent the maximum and minimum values of this examination type across the entire dataset. The final reported metric is the macro-average of NMAEk across all examination types.

- • BLEU Score. For free-text radiology reports, we utilize the standard BLEU metric to measure the n-gram overlap between the generated narratives and the reference reports. This provides a fundamental assessment of textual similarity and lexical coverage.
- • Step-wise Similarity. This metric evaluates the similarity between the model-predicted examination results and the ground truth at each step. During this evaluation, the simulator generates the current examination results conditioned on the queried examination names, the patient case summary, and all historical ground truth examination names and results. At each step, we apply Prompt 14 to compute similarity scores (ranging from 0 to 5) between the model-generated and ground truth examination results.
- • Full-chain Consistency. This metric assesses whether the generated examination chain is internally consistent and aligns with the patient prfile. Given the generated chain, we use Prompt 15 to assess whether the sequence of generated examination results maintains logical and clinical consistency. The evaluation final score is binary 1/0, indicating yes/no.

Examination-wise Metrics For the examination-wise metrics, we primarily assess the statistical distribution quality for both generative numerical and free-text results, covering fidelity and diversity.

- • Numerical Fidelity & Diversity. For numerical examination results, we utilize the 1-Wasserstein distance to measure the generative distribution fidelity, where shorter distances indicate closer alignment. Formally, considering a certain numerical examination, its real distribution is characterized by the mean µ and standard deviation σ. Similarly, the generative distribution is characterized by µˆ and σˆ. Denoting a generative examination value on a certain test case as xi, it can be normalized as zi = (xi − µ)/σ and similarly, for the ground-truth results, we have zˆi = (ˆxi − µˆ)/σˆ. The 1-Wasserstein distance is formulated as:

W1(Z,Zˆ) = inf

∥zi − zˆi∥dγ(zi,zˆi), (16)

γ∈Γ(Z,Zˆ)

where Z and Zˆ denote the generative and real distributions, and Γ(Z,Zˆ) is the joint distribution.

Then for diversity, we adopt normalized variance to evaluate its distribution diversity, with higher variance reflecting greater diversity. Formally, following the former notation, the normalized variance is defined as:

Var(X) µ2

σ2(X) =

, (17)

where Var(·) is the distribution variance. Notably, for simplicity, all the above numerical metric formulations assume that the examination item contains only a single value item. In practice, some examinations may consist of multiple value items, such as a “Complete Blood Count”, which may include multiple numerical values, and in these cases, the metrics are calculated by averaging the scores across all value items. Furthermore, to ensure consistency, all computations for the same value items are standardized to unified value units.

- • Free-text Fidelity & Diversity. For free-text results, we first encode the text into feature embeddings using BioLORD [27], a biomedical text encoding model. Inspired by metrics commonly used in image generation, we then calculate the Fréchet Inception Distance (FID)[28] in the embedding space to assess the fidelity. Lower FID values indicate better alignment with the ground truth. Specifically, considering a certain free-text-related examination, such as a chest CT examination, let the set of genrative free-text embeddings be F = {f1,f2,··· ,fN}, where N denotes the total number of test cases. The corresponding ground truth text embeddings are denoted as Fˆ = {fˆ1,fˆ2,··· ,fˆN}. The FID score can then be calculated as:

- 1

- 2

FID(F,Fˆ) = ∥µF − µFˆ∥22 + Tr ΣF + ΣFˆ − 2 ΣFΣFˆ

, (18)

where µF and ΣF represent the mean and covariance of the generative embeddings F, and µFˆ and ΣFˆ are for the groud turth embeddings Fˆ. The term ∥µF − µFˆ∥22 quantifies the difference between the means, while the trace term measures the distance between the covariance matrices. To evaluate diversity, inspired by the Intra-LPIPS metric [29] used in image generation, we propose using inter-case cosine similarity on the entire set of generated text embeddings. This metric reflects how well the embeddings distinguish from one another, defined as:

2 N(N − 1) i<j

cos(fi,fj), (19)

Intra-LPIPS(F) = 1 −

where cos(·,·) represents the cosine similarity function. A higher Intra-LPIPS score indicates that the generated free texts are more diverse in comparison to one another.

###### A.1.2 DiagAgent Evaluation

We evaluate DiagAgent’s performance across three key perspectives: final diagnosis accuracy, examination recommendation efficacy, and rubric-based diagnostic trajectory assessment.

Final Diagnosis Accuracy. For measure the final diagnosis, we adopt the straightforward Accuracy metric here. Similarly, considering diseases may have synonyms, we instruct GPT-4o utilizing Prompt 12 to compare the model’s diagnostic output with the reference standard.

Examination Recommendation Metrics In the end-to-end evaluation setting, we compare the predicted examination list against the referenced list and adopt Precision, Recall, and F1 scores. Considering that the same examination may be expressed in synonyms, we instruct GPT-4o employing Prompt 10 to count the number of examination names generated by the model that are covered by the true examination list, and Prompt 11 to count the number of ground truth examination names that are present in the model’s output. Precision and recall are then computed accordingly, followed by the calculation of the F1 score. In the single-turn evaluation setting, we adopt the hit ratio metric, counting whether the recommended examination item appears in the referenced following referrenced list. We adopt GPT-4o with Prompt 6 to determine whether this query appears among the examinations actually undergone by the patient.

Weighted Rubric Score To evaluate the clinical integrity of the multi-turn diagnostic interaction, we use

the Weighted Rubric Score. This metric goes beyond the final outcome by qualitatively assessing the entire diagnostic trajectory against physician-authored, process-oriented rubrics. The score is calculated as the weighted proportion of satisfied rubrics, with weights reflecting the clinical significance of each step. The final score s¯ is the average across all cases, calculated as:

1 |C| c∈C

r∈Ac wr r∈Rc wr

s¯ =

.

The GPT-4o judge model determines rubric satisfaction for each case (Prompt 13).

###### A.1.3 Human Validation of Automated Evaluation Metrics

To ensure the reliability of our LLM-based automated evaluation framework, we conduct a rigorous human validation study.

We organized a large-scale expert panel comprising 12 senior physicians, each with over 10 years of clinical experience. They performed the same evaluations as the automated LLM judge for the following metrics:

- 1. Diagnosis Accuracy: A binary task determining whether the model’s predicted diagnosis matches the ground truth label.
- 2. Rubric Hit Ratio: A binary task assessing whether a model’s diagnostic trajectory met specific physician-authored rubric criteria, which is the prerequisite step for calculating weighted rubric score.
- 3. Single-turn Hit Ratio: A binary task verifying if a recommended examination appeared in the reference list.
- 4. Examination Recommendation Count: A numerical task counting the number of correct matches (True Positives) between the predicted examination list and the reference list, which is the prerequisite step for calculating Precision, Recall, and F1 scores.

Due to the cost limitations of physician annotation, we randomly sample 100 instances for each metric. To ensure assessment reliability, each instance is independently annotated by three different physicians from the panel. We further aggregate the annotations from the assigned physician triplet. For binary classification tasks, we define the GT using the majority vote (consensus of at least 2 out of 3 physicians). Consistency is measured using the percentage of agreement between the LLM and the GT. For the numerical task, we define the GT using the median of the three physicians’ counts to mitigate outlier effects. Consistency is measured using the Mean Absolute Error (MAE) between the LLM’s count and the human median.

As detailed in Supplementary Table 1, the LLM judge demonstrated high reliability across all metrics. For binary tasks, the LLM achieved near-perfect alignment with the human consensus (e.g., 100.00% for Diagnosis Accuracy and 96.00% for Rubric Hit Ratio). Notably, the LLM’s consistency with the consensus often exceeded the average consistency of individual physicians (e.g., 100.00% vs. 92.67% for Diagnosis Accuracy), suggesting that the LLM judge functions as a standardized evaluator, mitigating the subjective variance and fatigue often observed in human annotation. For the numerical examination counting task, the LLM yielded a low MAE of 0.25. This error margin is comparable to the internal disagreement among physicians themselves (MAE 0.15), confirming that the LLM judge can effectively handle the semantic ambiguity of medical terms during counting tasks. These results substantiate the validity of our automated evaluation framework.

###### A.2 Supervised Finetuning

In this section, we introduce the multi-turn supervised fine-tuning (SFT) paradigm. In multi-turn SFT, each training sample consists of a dialogue history and the next expected response. Formally, let Dmulti = {(h(tj),yt(j))}Mj=1, where h(tj) = (u(1j),r1(j),...,u(t−j)1,rt(−j)1,u(tj)) denotes the dialogue history up to turn t (with

- u as user inputs and r as model responses), and yt(j) is the supervised response. The loss becomes:

M

Lmulti-turn SFT = −

j=1

log Pθ(yt(j) | h(tj)) (20)

- Supplementary Table 1 | Consistency analysis between Human Physicians and LLM-as-judge across 100 sampled cases. The Ground Truth (GT) is derived from the consensus of three independent senior physicians: using majority vote for Diagnosis Accuracy, Rubric Hit Ratio, Single-turn Hit Ratio; and the median value for Exam Rec. Count. Results indicate high alignment between the LLM-as-judge and the physician-derived GT. ‘Phy’ denotes physician and ‘Rec.’ denotes recommendation.

Phy. vs. AI GT vs. Phy.

Metric

GT vs. AI Phy. 1 Phy. 2 Phy. 3 Avg. of Three

Diagnosis Accuracy 82.00% 100.00% 96.00% 92.67% 100.00% Rubric Hit Ratio 62.00% 91.00% 96.00% 85.00% 96.00% Single-turn Hit Ratio 94.00% 94.00% 98.00% 96.67% 96.67%

Mean Absolute Error (MAE) ↓ Exam Rec. Count 0.12 0.33 0.35 0.15 0.25

###### A.3 Ablation on Training Compute (FLOPs)

To investigate whether the performance limitations of Supervised Fine-tuning (SFT) baselines stem from insufficient training steps or computational budget, we conducted an ablation study focusing on training duration. Using Qwen2.5-7B as the representative base model, we extended the fine-tuning process to 10 epochs — significantly beyond the standard convergence point —and monitored the performance on the end-to-end diagnostic task.

As illustrated in Supplementary Figure 1, the results reveal a clear performance ceiling inherent to the SFT paradigm.

- • Rapid Convergence: The model learns the static data distribution rapidly. Both the Final Diagnostic Accuracy (Supplementary Figure 1a) and the Examination Recommendation F1-score (Supplementary Figure 1b) reach their peak performance approximately at the 3rd epoch.
- • Performance Plateau: Extending training beyond this point (from epoch 3 to 10) yields negligible gains, with the performance curves flattening out. This plateau confirms that simply increasing training FLOPs or iterations does not enable the model to discover better diagnostic strategies.

This finding reinforces our core motivation: the performance gap between baselines and DiagAgent is not due to under-fitting, but rather the limitation of supervised imitation. SFT is bounded by the amount of the static training data, whereas our RL-based approach enables the agent to break through this ceiling by actively exploring and optimizing for long-term diagnostic utility.

[Figure 51]

a. Final Diagnostic Decision-making b. Examination Recommendation

Accuracy(%)

F1-score

Epoch Epoch

Supplementary Figure 1 | Ablation study on the impact of training epochs (FLOPs) for Supervised Fine-tuning (SFT). We conduct the experiment using Qwen2.5-7B under the end-to-end evaluation setting. a illustrates the Final Diagnostic Decision-making accuracy across 10 epochs. b presents the Examination Recommendation F1-score. The shaded regions indicate the 95% confidence interval (CI). The results demonstrate that the model converges rapidly, with performance hitting a bottleneck and plateauing after the 3rd epoch, indicating that simply increasing training steps in SFT does not lead to continuous improvement.

- Supplementary Table 2 | Selected Laboratory Events and Their Corresponding Sub-events Event Sub-events

Complete Blood Count MCH; White Blood Cells; Absolute Basophil Count; Absolute Monocyte Count; Absolute Eosinophil Count; Monocytes; Platelet Count; Hemoglobin; Hematocrit; Neutrophils; Absolute Neutrophil Count; RDW; Basophils; Eosinophils; Red Blood Cells; RDW-SD; Lymphocytes

Mean Corpuscular Volume MCV Liver Function Test PT; Alanine Aminotransferase (ALT); Asparate Aminotransferase (AST);

Albumin; Alkaline Phosphatase Anion Gap Anion Gap Comprehensive Metabolic Panel

Bicarbonate

Total Bilirubin Bilirubin, Total Total Calcium Calcium, Total Kidney Function Tests Urea Nitrogen; Glucose; Phosphate; Creatinine; Sodium; Potassium; Chlo-

ride Lactate Dehydrogenase Lactate Dehydrogenase (LD) Magnesium Magnesium Coagulation Profile PTT Lactate Lactate Urine Analysis RBC; pH; WBC Total Calculated CO2 Calculated Total CO2 pCO2 pCO2 pH pH pO2 pO2 Lipase Lipase Creatine Kinase Creatine Kinase (CK) Creatine Kinase, MB Isoenzyme

Creatine Kinase, MB Isoenzyme

- H H
- I I L L Thyroid Function Test Thyroid Stimulating Hormone

###### A.4 Selected Examinations Collection

To ensure the reliability of the distributional analysis, we considered only examinations with relatively high occurrence frequencies. For numerical-type examinations, only those with more than 500 occurrences were retained, resulting in 11 examinations comprising a total of 24 subevents. The distributional metrics were computed based on these 24 subevents, as detailed in Supplementary Table 2. For textual-type examinations, only radiology examinations with more than 100 occurrences were considered, yielding five types: “LIVER OR GALLBLADDER US (SINGLE ORGAN)”, “CT HEAD W/O CONTRAST”, “CT HEAD W/O CONTRAST Q111 CT HEAD”, “CHEST (PA AND LAT)”, and “CHEST (PORTABLE AP)”.

###### A.5 Detailed Results on DiagBench

- A.5.1 Results on MIMIC-IV

- Supplementary Table 3 | Single-turn evaluation on the MIMIC-IV dataset. We report the Hit Ratio for examination recommendation and Accuracy for final diagnosis, with 95% Confidence Intervals in brackets. Agentic systems use DeepSeek-

- v3 as their base model. Model Size Year Hit Ratio(%) Diagnosis Accuracy(%)

Basic LLM

20.21

72.13

GPT-4o - 2024.8

[18.98, 21.39]

[69.07, 75.33]

26.40

76.80

Claude-4-sonnet - 2024.6

[24.95, 27.66]

[73.60, 79.73]

19.20

74.80

Qwen2.5 72B 2024.9

[17.94, 20.43]

[71.87, 77.73]

19.97

66.27

Llama3.3 70B 2024.12

[18.77, 21.20]

[63.07, 69.47]

20.08

72.27

DeepSeek-v3 671B 2024.12

[18.77, 21.37]

[69.07, 75.60]

21.39

72.40

Qwen2.5 235B -

[20.19, 22.70]

[69.46, 75.47]

15.37

68.27

GPT-OSS 120B -

[14.27, 16.60]

[65.06, 71.73]

23.53

66.27

OpenBioLLM 70B 2024.4

[22.22, 24.79]

[62.93, 69.60]

19.60

78.93

Baichuan-M1 14B 2025.2

[18.23, 20.88]

[76.00, 81.60]

28.57

70.53

MedGemma 27B 2025.7

[27.23, 30.07]

[67.20, 73.87]

22.36

67.73

Baichuan-M3 235B 2026.1

[21.15, 23.67]

[64.27, 70.80]

Agentic System

14.08

70.27

MedAgents - 2024.1

[13.01, 15.21]

[67.07, 73.47]

12.64

73.47

MDAgents - 2024.10

[11.57, 13.63]

[70.26, 76.40]

Our Method (DiagAgent)

- 7B -

72.56

[71.19, 74.00]

85.60

[82.93, 88.14]

- 8B -

DiagAgent

56.57

82.27

[55.02, 58.13]

[79.60, 84.80]

###### 87.87

68.49

14B -

[67.07, 69.91]

[85.47, 90.13]

- Supplementary Table 4 | End-to-end evaluation on the MIMIC-IV dataset. Metrics include average conversation turns, precision, recall, F1-score for examination recommendation, and final diagnostic accuracy. All metrics are reported with 95% Confidence Intervals

Model Size Year Avg. Turns Precision Recall F1 Accuracy(%) Basic LLM

31.08

14.76

16.96

43.20

GPT-4o - 2024.8 3.30

[28.22, 33.76]

[13.48, 16.13]

[15.37, 18.44]

[39.73, 46.67]

37.10

25.02

26.32

51.47

Claude-4-Sonnet - 2024.6 3.91

[34.42, 39.67]

[23.25, 26.78]

[24.62, 28.03]

[47.86, 55.07]

35.19

12.13

16.23

44.27

Qwen2.5 72B 2024.9 2.47

[31.98, 38.59]

[10.93, 13.39]

[14.66, 17.78]

[40.67, 47.73]

28.99

22.33

23.01

38.53

Llama3.3 70B 2024.12 4.25

[27.08, 31.07]

[20.66, 24.11]

[21.52, 24.66]

[34.93, 42.00]

35.09

13.09

16.78

47.07

DeepSeek-v3 671B 2024.12 2.49

[32.03, 37.94]

[11.74, 14.50]

[15.19, 18.48]

[43.60, 50.53]

30.14

18.60

19.74

45.33

Qwen2.5 235B - 3.34

[27.63, 32.63]

[17.08, 20.20]

[18.23, 21.30]

[42.00, 49.34]

27.82

16.54

16.72

46.53

GPT-OSS 120B - 4.08

[25.11, 30.48]

[15.00, 18.02]

[15.22, 18.26]

[43.20, 49.87]

32.80

14.02

17.73

34.27

OpenBioLLM 70B 2024.4 2.59

[29.92, 35.64]

[12.63, 15.45]

[16.07, 19.49]

[30.93, 37.87]

26.05

9.66

12.55

33.33

Baichuan-M1 14B 2025.2 2.30

[23.13, 28.94]

[8.50, 10.90]

[11.16, 13.97]

[30.13, 37.07]

35.20

21.17

22.44

44.27

MedGemma 27B 2024.7 4.10

[32.73, 37.79]

[19.44, 22.90]

[20.76, 24.09]

[40.53, 48.00]

29.15

21.88

19.83

40.40

Baichuan-M3 235B 2026.1 4.83

[26.52, 31.83]

[20.03, 23.70]

[18.35, 21.48]

[36.93, 43.73]

Agentic System

30.59

11.37

14.76

45.60

MedAgents - 2024.1 2.31

[27.37, 33.76]

[10.13, 12.68]

[13.09, 16.39]

[42.00, 49.20]

32.18

11.90

15.37

45.07

MDAgents - 2024.10 2.40

[29.23, 35.40]

[10.82, 13.13]

[13.78, 17.02]

[41.47, 48.53]

Our Method (DiagAgent)

- 7B - 5.45

46.02

[44.04, 47.97]

47.33

[45.40, 49.21]

43.90

[42.21, 45.58]

61.47

[58.00, 65.20]

- 8B - 5.73

DiagAgent

39.57

43.13

38.56

53.33

[37.66, 41.35]

[41.10, 44.95]

[36.83, 40.34]

[49.60, 56.80]

###### 52.14

###### 62.67

42.04

43.72

14B - 6.66

[40.19, 43.69]

[50.14, 54.05]

[42.07, 45.41]

[59.07, 66.13]

###### A.5.2 Results on PMC-OA

- Supplementary Table 5 | Single-turn evaluation on the PMC-OA dataset. We report Hit Ratio for examination recommendation and Accuracy for final diagnosis, with 95% Confidence Intervals.

Model Size Year Hit Ratio(%) Diagnosis Accuracy(%) Basic LLM GPT-4o - 2024.8

51.03

84.15

[46.91, 54.68]

[81.14, 86.85]

46.12

84.63

Claude-4-sonnet - 2024.6

[42.15, 49.77]

[81.93, 87.48]

50.08

82.88

Qwen2.5 72B 2024.9

[46.11, 54.05]

[80.03, 85.90]

51.82

77.18

Llama3.3 70B 2024.12

[48.02, 55.63]

[73.69, 80.51]

47.86

83.84

DeepSeek-v3 671B 2024.12

[44.06, 51.66]

[80.98, 86.69]

48.65

83.68

Qwen2.5 235B -

[44.69, 52.61]

[80.51, 86.37]

43.42

79.08

GPT-OSS 120B -

[39.62, 47.39]

[76.22, 82.09]

48.18

72.42

OpenBioLLM 70B 2024.4

[44.22, 51.98]

[68.94, 75.59]

43.26

88.59

Baichuan-M1 14B 2025.2

[39.30, 47.23]

[86.05, 90.97]

48.81

83.20

MedGemma 27B 2024.7

[44.85, 52.93]

[80.35, 86.05]

48.02

81.14

Baichuan-M3 235B 2026.1

[44.06, 51.66]

[78.13, 84.31]

Agentic System

43.26

81.77

MedAgents - 2024.1

[39.30, 47.39]

[78.92, 84.63]

48.18

84.15

MDAgents - 2024.10

[44.22, 52.14]

[81.30, 87.00]

Our Method (DiagAgent)

###### 51.98

###### 88.91

DiagAgent 14B -

[47.86, 55.94]

[86.37, 91.28]

- Supplementary Table 6 | End-to-end evaluation on the PMCOA dataset. We report Precision, Recall, F1-score for examination recommendation, and final diagnostic accuracy with 95% Confidence Intervals.

Model Size Year Avg. Turns Precision Recall F1 Accuracy(%) Basic LLM

37.32

22.16

24.20

48.81

GPT-4o - 2024.8 3.04

[33.9-40.7]

[19.9-24.3]

[21.8-26.4]

[45.0-52.8]

34.70

30.03

28.05

49.13

Claude-4-sonnet - 2024.6 4.17

[32.0-37.5]

[27.7-32.5]

[26.0-30.2]

[45.0-52.9]

32.44

16.97

20.19

39.14

Qwen2.5 72B 2024.9 2.59

[29.3-36.0]

[15.1-18.9]

[18.2-22.2]

[35.3-42.8]

31.10

31.46

28.22

45.48

Llama3.3 70B 2024.12 4.09

[28.8-33.3]

[28.9-33.9]

[26.1-30.4]

[41.7-49.1]

37.63

19.61

23.46

43.11

DeepSeek-v3 671B 2024.12 2.52

[34.3-41.0]

[17.6-21.8]

[21.3-25.9]

[39.5-46.9]

34.96

28.56

27.43

49.45

Qwen2.5 235B - 3.45

[32.2-37.9]

[26.2-31.0]

[25.2-29.7]

[45.6-53.3]

28.19

25.21

22.20

45.96

GPT-OSS 120B - 4.30

[25.4-31.0]

[22.8-27.6]

[20.1-24.4]

[42.2-49.9]

31.03

16.38

19.23

35.18

OpenBioLLM 70B 2024.4 2.49

[27.9-34.6]

[14.7-18.2]

[17.2-21.4]

[31.7-38.7]

20.59

11.89

13.82

39.46

Baichuan-M1 14B 2025.2 2.62

[17.7-23.3]

[10.1-13.6]

[11.9-15.9]

[35.8-43.3]

32.64

26.79

25.47

44.37

MedGemma 27B 2024.7 4.04

[29.7-35.5]

[24.4-29.2]

[23.4-27.7]

[40.7-48.3]

29.42

27.22

23.38

42.31

Baichuan-M3 235B 2026.1 4.40

[26.5-32.4]

[24.9-29.6]

[21.4-25.7]

[38.5-46.1]

Agentic System

33.26

16.99

20.31

45.80

MedAgents - 2024.1 2.26

[29.9-36.6]

[15.2-19.0]

[18.1-22.5]

[41.8-49.8]

###### 38.11

18.75

22.52

45.48

MDAgents - 2024.10 2.40

[34.6-41.5]

[16.7-20.7]

[20.3-24.8]

[41.7-49.6]

Our Method

###### 46.21

###### 29.72

###### 53.88

24.33

DiagAgent 14B - 6.88

[22.9-25.9]

[43.7-49.0]

[27.9-31.5]

[50.2-58.0]

###### A.5.3 Results on MTSamples

- Supplementary Table 7 | Single-turn evaluation on the MTSamples dataset. We report Hit Ratio for examination recommendation and Accuracy for final diagnosis, with 95% Confidence Intervals.

Model Size Year Hit Ratio(%) Diagnosis Accuracy(%) Basic LLM GPT-4o - 2024.8

49.08

86.02

[44.33, 54.35]

[82.32, 89.19]

43.54

84.43

Claude-4-sonnet - 2024.6

[38.52, 48.55]

[81.00, 87.86]

53.03

86.81

Qwen2.5 72B 2024.9

[48.02, 58.05]

[83.38, 89.97]

49.60

75.99

Llama3.3 70B 2024.12

[44.59, 54.62]

[71.50, 80.21]

44.33

83.64

DeepSeek-v3 671B 2024.12

[39.05, 49.60]

[79.42, 87.34]

47.76

83.38

Qwen2.5 235B -

[42.48, 52.77]

[79.42, 86.81]

44.85

75.73

GPT-OSS 120B -

[39.84, 49.60]

[70.98, 79.95]

46.97

82.06

OpenBioLLM 70B 2024.4

[41.69, 51.72]

[78.10, 86.02]

44.33

90.24

Baichuan-M1 14B 2025.2

[39.05, 49.08]

[87.07, 92.88]

50.40

83.91

MedGemma 27B 2024.7

[45.38, 55.67]

[79.95, 87.34]

47.23

79.16

Baichuan-M3 235B 2026.1

[42.22, 52.51]

[74.93, 83.11]

Agentic System

41.42

79.16

MedAgents - 2024.1

[36.41, 45.91]

[74.67, 83.11]

51.45

81.00

MDAgents - 2024.10

[46.70, 56.20]

[77.31, 84.70]

Our Method (DiagAgent)

###### 60.16

###### 93.67

DiagAgent 14B -

[55.15, 65.17]

[91.29, 95.78]

- Supplementary Table 8 | End-to-end evaluation on the MTSamples dataset. We report Precision, Recall, F1-score for examination recommendation, and final diagnostic accuracy with 95% Confidence Intervals.

Model Size Year Avg. Turns Precision Recall F1 Accuracy(%) Basic LLM

41.60

27.55

30.34

47.23

GPT-4o - 2024.8 3.13

[37.13-46.00]

[24.85-30.50]

[27.64-33.47]

[42.48-52.24]

43.71

37.14

36.59

57.52

Claude-4-sonnet - 2024.6 3.54

[39.92-47.56]

[34.05-40.23]

[33.63-39.57]

[52.51-62.53]

37.21

22.82

25.85

46.17

Qwen2.5 72B 2024.9 2.58

[33.44-41.39]

[20.19-25.67]

[22.89-28.91]

[41.16-51.19]

36.77

39.25

34.92

50.40

Llama3.3 70B 2024.12 3.95

[33.98-39.79]

[36.08-42.69]

[31.99-37.95]

[45.12-55.15]

###### 46.09

24.38

29.05

52.24

DeepSeek-v3 671B 2024.12 2.42

[41.28-50.73]

[21.78-27.18]

[26.23-32.13]

[47.76-56.99]

41.17

33.31

33.50

50.40

Qwen2.5 235B - 3.25

[37.38-44.95]

[30.30-36.28]

[30.56-36.62]

[45.12-55.15]

38.68

29.74

28.57

52.51

GPT-OSS 120B - 3.60

[34.57-42.76]

[26.91-33.06]

[25.65-31.57]

[47.23-57.26]

41.72

24.23

28.75

41.16

OpenBioLLM 70B 2024.4 2.44

[37.28-46.20]

[21.53-26.98]

[25.63-31.78]

[36.15-46.70]

32.92

18.58

22.08

48.55

Baichuan-M1 14B 2025.2 2.50

[28.53-37.58]

[16.27-21.33]

[19.26-25.03]

[43.80-53.83]

41.23

32.99

32.42

48.55

MedGemma 27B 2024.7 3.84

[37.62-45.29]

[29.84-35.88]

[29.57-35.57]

[43.27-53.56]

34.55

32.59

28.02

46.44

Baichuan-M3 235B 2026.1 4.28

[30.59-38.44]

[29.67-35.72]

[25.26-30.54]

[41.69-51.19]

Agentic System

37.14

19.56

23.10

48.28

MedAgents - 2024.1 2.23

[32.56-41.79]

[16.91-22.31]

[20.20-26.22]

[43.01-53.04]

42.13

21.55

26.01

50.13

MDAgents - 2024.10 2.23

[37.20-46.93]

[19.10-24.24]

[23.05-28.86]

[45.12-54.88]

Our Method

###### 63.24

###### 39.17

###### 64.91

30.64

DiagAgent 14B - 6.64

[28.90-32.53]

[59.71-66.67]

[37.02-41.37]

[60.16-69.39]

###### A.5.4 Results on DDXPlus

- Supplementary Table 9 | Single-turn evaluation on the DDXPlus dataset. We report Hit Ratio for examination recommendation and Accuracy for final diagnosis, with 95% Confidence Intervals.

Model Size Year Hit Ratio(%) Diagnosis Accuracy(%) Basic LLM GPT-4o - 2024.8

55.73

91.55

[51.71, 60.16]

[89.13, 93.96]

46.68

94.57

Claude-4-sonnet - 2024.6

[42.25, 51.11]

[92.35, 96.38]

61.97

93.56

Qwen2.5 72B 2024.9

[58.14, 66.20]

[91.35, 95.57]

70.22

87.53

Llama3.3 70B 2024.12

[66.20, 74.04]

[84.51, 90.54]

47.69

91.15

DeepSeek-v3 671B 2024.12

[43.26, 52.52]

[88.53, 93.56]

48.89

93.36

Qwen2.5 235B -

[44.66, 53.12]

[91.15, 95.57]

47.28

80.68

GPT-OSS 120B -

[42.86, 51.71]

[77.26, 84.31]

53.52

92.76

OpenBioLLM 70B 2024.4

[49.30, 57.95]

[90.54, 94.97]

42.45

93.96

Baichuan-M1 14B 2025.2

[37.83, 46.68]

[91.55, 95.98]

50.10

92.96

MedGemma 27B 2024.7

[45.67, 54.53]

[90.74, 94.97]

51.11

89.94

Baichuan-M3 235B 2026.1

[46.68, 55.14]

[87.32, 92.56]

Agentic System

44.47

87.73

MedAgents - 2024.1

[40.44, 48.70]

[84.71, 90.34]

49.30

91.35

MDAgents - 2024.10

[45.07, 53.92]

[88.73, 93.57]

Our Method (DiagAgent)

###### 86.12

###### 96.38

DiagAgent 14B -

[83.10, 88.93]

[94.77, 97.79]

- Supplementary Table 10 | End-to-end evaluation on the DDXPlus dataset. We report Precision, Recall, F1-score for examination recommendation, and final diagnostic accuracy with 95% Confidence Intervals.

Model Size Year Avg. Turns Precision Recall F1 Accuracy(%) Basic LLM

36.95

33.68

31.29

66.80

GPT-4o - 2024.8 3.23

[33.38-40.31]

[30.41-36.77]

[28.55-34.20]

[62.98-71.03]

36.00

33.30

31.78

76.06

Claude-4-sonnet - 2024.6 3.28

[32.82-39.45]

[30.21-36.62]

[28.93-34.68]

[72.43-79.68]

37.32

27.04

29.04

66.60

Qwen2.5 72B 2024.9 2.58

[33.60-41.20]

[24.14-29.96]

[26.04-32.03]

[62.58-70.62]

###### 38.67

36.64

48.48

63.78

Llama3.3 70B 2024.12 4.00

[34.05-39.10]

[45.28-51.42]

[36.37-41.13]

[59.76-68.01]

37.32

25.18

27.49

67.61

DeepSeek-v3 671B 2024.12 2.45

[33.64-41.38]

[22.23-28.07]

[24.65-30.38]

[63.78-71.83]

35.30

37.16

33.49

68.41

Qwen2.5 235B - 3.31

[32.07-38.58]

[33.73-40.31]

[30.65-36.17]

[64.38-72.03]

27.19

27.75

23.10

67.00

GPT-OSS 120B - 3.62

[23.75-30.43]

[24.83-30.51]

[20.71-25.78]

[62.78-71.23]

30.72

22.11

23.46

53.72

OpenBioLLM 70B 2024.4 2.50

[27.35-34.14]

[19.49-24.52]

[20.82-25.96]

[49.30-57.95]

18.13

11.37

12.68

63.98

Baichuan-M1 14B 2025.2 2.43

[14.85-21.78]

[9.36-13.58]

[10.51-15.03]

[59.36-68.21]

34.69

39.31

32.81

66.20

MedGemma 27B 2024.7 3.69

[31.49-38.03]

[35.93-42.78]

[29.77-35.79]

[61.97-70.23]

30.78

32.53

26.99

67.61

Baichuan-M3 235B 2026.1 3.86

[27.66-34.17]

[29.43-35.54]

[24.39-29.64]

[63.38-71.84]

Agentic System

35.00

21.80

24.44

73.24

MedAgents - 2024.1 2.24

[31.26-39.00]

[19.11-24.38]

[21.47-27.26]

[69.41-77.26]

###### 39.01

27.41

29.77

70.42

MDAgents - 2024.10 2.53

[35.24-43.00]

[24.58-30.22]

[26.72-32.80]

[66.40-74.25]

Our Method

###### 66.74

###### 76.42

24.74

34.36

DiagAgent 14B - 7.07

[23.1-26.2]

[63.6-70.1]

[32.5-36.5]

[72.56-80.08]

###### A.5.5 Results on combined out-of-domain datasets

- Supplementary Table 11 | Single-turn evaluation of out-of-domain performance across combined data sources (PMC-OA, MTSamples, and DDXPlus; N = 1, 507). We report Hit Ratio for examination recommendation and Accuracy for final diagnosis, with 95% Confidence Intervals.

Model Size Year Hit Ratio(%) Diagnosis Accuracy(%) Basic LLM GPT-4o - 2024.8

52.09

87.06

[49.63, 54.48]

[85.40, 88.72]

45.65

87.86

Claude-4-sonnet - 2024.6

[43.00, 48.04]

[86.13, 89.52]

54.74

87.39

Qwen2.5 72B 2024.9

[52.15, 57.13]

[85.73, 89.25]

57.33

80.29

Llama3.3 70B 2024.12

[54.88, 59.99]

[78.23, 82.22]

46.91

86.20

DeepSeek-v3 671B 2024.12

[44.39, 49.37]

[84.41, 87.92]

48.51

86.79

Qwen2.5 235B -

[46.05, 51.16]

[85.00, 88.45]

45.06

78.77

GPT-OSS 120B -

[42.60, 47.64]

[76.71, 80.69]

49.64

81.55

OpenBioLLM 70B 2024.4

[47.11, 52.09]

[79.50, 83.61]

43.26

90.78

Baichuan-M1 14B 2025.2

[40.74, 45.79]

[89.25, 92.30]

49.64

86.60

MedGemma 27B 2024.7

[47.24, 52.09]

[84.94, 88.19]

48.84

83.54

Baichuan-M3 235B 2026.1

[46.25, 51.43]

[81.75, 85.40]

Agentic System

43.20

83.08

MedAgents - 2024.1

[40.68, 45.65]

[81.09, 85.14]

49.37

85.73

MDAgents - 2024.10

[47.05, 51.96]

[83.94, 87.53]

Our Method (DiagAgent)

###### 65.30

###### 92.57

DiagAgent 14B -

[62.91, 67.69]

[91.24, 93.83]

- Supplementary Table 12 | End-to-end evaluation of out-of-domain performance across combined data sources (PMC-OA, MTSamples, and DDXPlus; N = 1, 507). We report Precision, Recall, F1-score for examination recommendation, and final diagnostic accuracy with 95% Confidence Intervals.

Model Size Year Avg. Turns Precision Recall F1 Accuracy(%) Basic LLM

38.28

27.32

28.08

54.35

GPT-4o - 2024.8 3.12

[36.25-40.38]

[25.81-28.83]

[26.66-29.54]

[52.09-56.80]

37.39

32.89

31.43

60.12

Claude-4-sonnet - 2024.6 3.72

[35.47-39.42]

[31.18-34.48]

[29.87-33.04]

[57.66-62.51]

35.25

21.76

24.53

49.97

Qwen2.5 72B 2024.9 2.58

[33.15-37.51]

[20.40-23.18]

[23.04-26.15]

[47.31-52.29]

34.35

39.03

33.35

52.75

Llama3.3 70B 2024.12 4.03

[32.83-35.88]

[37.40-40.70]

[31.87-34.79]

[50.17-55.21]

###### 39.66

22.65

26.19

53.48

DeepSeek-v3 671B 2024.12 2.47

[37.48-41.57]

[21.17-24.13]

[24.64-27.83]

[50.96-55.74]

36.63

32.59

30.96

55.94

Qwen2.5 235B - 3.35

[34.72-38.56]

[30.92-34.15]

[29.29-32.50]

[53.55-58.59]

30.50

27.19

24.10

54.55

GPT-OSS 120B - 3.90

[28.69-32.47]

[25.68-28.84]

[22.70-25.63]

[52.02-57.13]

33.62

20.24

23.02

42.80

OpenBioLLM 70B 2024.4 2.48

[31.42-35.91]

[18.86-21.49]

[21.64-24.43]

[40.48-45.12]

22.88

13.40

15.52

49.83

Baichuan-M1 14B 2025.2 2.53

[20.81-24.90]

[12.11-14.64]

[14.13-16.91]

[47.31-52.29]

35.48

32.48

29.64

52.62

MedGemma 27B 2024.7 3.88

[33.57-37.48]

[30.85-34.10]

[28.17-31.12]

[50.23-55.28]

31.15

30.33

25.74

51.69

Baichuan-M3 235B 2026.1 4.19

[29.41-32.93]

[28.85-31.92]

[24.36-27.22]

[49.17-54.15]

Agentic System

34.81

19.22

22.38

55.47

MedAgents - 2024.1 2.24

[32.54-37.18]

[17.92-20.60]

[20.83-23.90]

[53.02-57.80]

39.42

22.31

25.79

54.88

MDAgents - 2024.10 2.40

[37.13-41.85]

[21.01-23.71]

[24.44-27.31]

[52.49-57.33]

Our Method

###### 57.26

###### 33.63

###### 63.84

26.05

DiagAgent 14B - 6.88

[25.10-27.07]

[55.43-59.11]

[32.48-34.81]

[61.44-66.29]

- A.6 Case Study

- A.6.1 Case study for DiagGym

During physician evaluation, they also provide qualitative insights on each methods’ simulation failure modes. DeepSeek-v3-671B produces detailed, information-rich reports with broad coverage, but it frequently overextrapolate from the diagnosis and clinical history, yielding overly severe positive findings. Also, narrative organization is sometimes disjointed with topic switching and repetition. Qwen2.5-72B offers comprehensive coverage but sometimes introduces unsupported neutral or false-positive findings with tenuous links to the core diagnosis. MedGemma-27B, as a medical-domain LLM, is more concise and structurally clear with fewer off-topic assertions, though it intermittently exhibited logical inconsistencies. In contrast, DiagGym maintains balanced converge and close alignment with case context, largely avoids unwarranted positives and material contradictions.

Supplementary Figure 2 compares outputs from the DiagGym with ground-truth examination results for a representative patient. The case involves a woman presenting with painless jaundice, with a history of breast cancer and non-ischemic cardiomyopathy. Upon admission, laboratory tests including liver function and bilirubin levels both supported a diagnosis of biliary obstruction; the final confirmed diagnosis was ‘CBD obstruction from common hepatic duct mass’.

The world model’s simulated predictions closely match the true clinical findings. A key indicator, Total Bilirubin, was predicted at 6.7 mg/dL versus the recorded value of 4.3 mg/dL (both well above the reference range of 0–1.5 mg/dL). This pronounced hyperbilirubinemia is consistent with the patient’s presentation (jaundice, scleral icterus) and strongly supports the diagnosis of “CBD obstruction from common hepatic duct mass”. Minor numerical variations occur across other laboratory results, but none alter the clinical interpretation. Such variability illustrates the model’s ability to generate plausible but non-identical results, preserving both realism and diversity.

|Note ID 16988189-DS-3| | | |
|---|---|---|---|
|Patient Profile|Sex: F Allergies: Ciprofloxacin / Erythromycin Base / Sulfa (Sulfonamide Antibiotics) / amoxicillin Chief Complaint S/p ERCP with brushing, biliary stent for CHD stricture History of Present Illness Ms. ___ is a ___ year old woman with history of breast cancer (s/p bilateral mastectomy, Adriamycin and taxol and radiation in ___), hypertension and non-ischemic cardiomyopathy presenting for monitoring post-ERCP in the setting of painless jaundice. Per Atrius records and my discussion with patient, Ms. ___ ___ presented to her PCP ___ ___ with 1 week of nausea and<br><br>constipation with rare hard stools that were light in color. She was also noting a flare of her hemorrhoids as well as itching all<br><br>over without new medications. She had labs sent at that time which showed elevated LFTs (AST 148, ALT 312, Alk phos 421, Tbili 7.6). She underwent a CT abdomen/pelvis which reportedly showed a duct tumor involving the common hepatic duct bifurcation concerning for Klatskin tumor (or hilar cholangiocarcinoma). Today, Ms. ___ presented for a planned procedure for evaluation of her mass and obstruction. She underwent ERCP with brushings and biliary stent placement for CHD stricture which she tolerated well. Upon arrival to the floor, she reports that she feels well without any symptoms. She did not take her blood pressure medications this morning. Her mouth feels dry but she denies light headedness, dizziness. No recent weight loss, fevers or night sweats. Her nausea has resolved since last week. She is having some chalky stools and dark urine and has noted yellowing of her skin. Past Medical History Past medical history includes hypertension, non-ischemic cardiomyopathy (diagnosed ___, viral vs. chemotherapy induced), left heart catheterization with normal coronaries, transthoracic echocardiogram ___: EF normal, and breast cancer (status post bilateral mastectomy, Adriamycin/Taxol, radiation therapy ___). Social History None<br><br>Family History<br><br>Father passed away ___ bone cancer. Otherwise non-contributory Final diagnosis CBD obstruction from common hepatic duct mass| | |
|Examination Chain| | | |
|Examination Name|Prediction|Ground truth|Difference & Evaluation|
|General<br><br>(Physical Exam)|Well appearing older woman, sitting<br><br>upright in bed|Well appearing thin woman sitting up in<br><br>bed in NAD, pleasant|No significant difference; both<br><br>describe a well-appearing patient.|
|Eyes (Physical Exam)|EOMI, PERRL, anicteric|Icteric sclera, PERRL, EOMI|Prediction missed icterus, but GT confirms jaundice; this supports diagnosis (CBD obstruction/mass).|
|Gastrointestinal (Physical Exam)|non-distended, soft, non-tender|hypoactive bowel sounds, soft, nondistended, mild ttp in epigastrium without rebound|Mild tenderness in GT, but both exams are overall benign with no peritonitis; does not alter main diagnosis.|
|Skin (Physical Exam)|Yellow hue, multiple nevi, no rash, warm|No rashes noted, icteric skin|Both note jaundice; difference in description detail is not significant.|
|Liver Function Test|INR(PT): 1.1 (0.9-1.1); PT: 11.5 (9.4-<br><br>12.5); ALT: 188 (0-40); Alk phos: 388<br><br>(35-105); AST: 118 (0-40)|INR(PT): 1.2 (0.9-1.1); PT: 12.5 (9.4-<br><br>12.5); ALT: 342 (0-40); Alk phos: 300 (35-<br><br>105); AST: 195 (0-40)|All abnormal, showing cholestasis/hepatocellular damage;<br><br>GT values somewhat higher, but<br><br>both confirm the same severe pattern.|
|Comprehensive Metabolic Panel|Bicarbonate: 23 (22-32)|Bicarbonate: 20 (22-32) abnormal|Mildly low in GT; not clinically important for diagnosis.|
|Total Bilirubin|6.7 (0-1.5) abnormal|4.3 (0-1.5) abnormal|Both markedly elevated; both support obstructive jaundice, difference not clinically significant.|
|Total Calcium|9.6 (8.4-10.3)|9.5 (8.4-10.3)|No significant difference.|
|Complete Blood Count|Hct: 35.8 (34–45); Hgb: 12.3 (11.2– 15.7); MCH: 31.9 (26–32); MCHC: 34.4 (32–37); Plt: 235 (150–400); RDW: 14.9 (10.5–15.5); RBC: 3.85 (3.9–5.2) abnormal; WBC: 7.4 (4–10); RDW-SD: 51.1 (35.1–46.3) abnormal|Hct: 33.2 (34–45) abnormal; Hgb: 11.0 (11.2–15.7) abnormal; MCH: 33.0 (26–32) abnormal; MCHC: 33.1 (32–37); Plt: 271 (150–400); RDW: 13.2 (10.5–15.5); RBC: 3.33 (3.9–5.2) abnormal; WBC: 12.6 (4–<br><br>10) abnormal; RDW-SD: 48.7 (35.1–46.3) abnormal|Both show mild anemia and some abnormal indices; GT WBC higher but not critical. Findings do not change main diagnosis.|
|Mean Corpuscular Volume|100 (82–98) abnormal|100 (82–98) abnormal|Both the same, macrocytic; not significant to main diagnosis.|
|Coagulation Profile|PTT: 31.1 (25–36.5)|PTT: 31.2 (25–36.5)|No significant difference.|

- Supplementary Figure 2 | Example Case Study from DiagGym: Comparison of Predicted and Ground Truth Examinations. This case presents a single patient profile and final diagnosis, illustrating the step-wise evaluation setting. The core table compares the predicted examination results generated by DiagGym with the ground truth results in sequential order. The rightmost column analyzes key differences and discusses their clinical relevance in the diagnostic process.

###### A.6.2 Case study for DiagAgent

Here we provide 2 cases for DiagAgent. Specifically, Supplementary Figure 3 demonstrates the agent’s ability to mirror real-world clinical timelines through logical, evidence-based information gathering, while Supplementary Figure 4 and Supplementary Figure 5 showcases how its diagnostic quality is validated against rigorous, physician-curated rubrics.

|Note ID 13544756-DS-14| | |
|---|---|---|
|Initial Inquiry| |Sex: Male Chief Complaint: Abdominal pain.<br><br>History of Present Illness: The patient reports several hours of abdominal pain that started periumbilical and migrated to the right<br><br>lower quadrant. Associated symptoms include chills, nausea, diarrhea, and anorexia. The pain worsened progressively. Earlier in the day, he experienced diarrhea and had only soup and bread due to nausea. Past Medical History: Sleep apnea. Personal History: None. Family History: Hypertension, diabetes mellitus. Allergy History: Penicillins.|
|Final Diagnosis| |Appendicitis|
|DiagAgent Interaction Messages| | |
|Turn|Role|Content|
|1|DiagAgent|Current diagnosis: The patient presents with abdominal pain localized to the right lower quadrant, accompanied by nausea, diarrhea,<br><br>and anorexia. The differential diagnosis includes appendicitis, gastroenteritis, or other abdominal pathologies.<br><br>Based on the patient's initial presentation, the following investigation(s) should be performed: Complete Blood Count. Reason: To evaluate for signs of infection or inflammation, such as leukocytosis, which is commonly seen in appendicitis.|
| |DiagGym|Absolute Lymphocyte Count: Value: 0.67; Numeric Value: 0.67; Units: K/uL; Reference Range: 1.2 - 3.7; Flag: abnormal Basophils: Value: 0.3; Numeric Value: 0.3; Units: %; Reference Range: 0.0 - 1.0 Eosinophils: Value: 0.0; Numeric Value: 0.0; Units: %; Reference Range: 1.0 - 7.0; Flag: abnormal Hematocrit: Value: 40.9; Numeric Value: 40.9; Units: %; Reference Range: 40.0 - 51.0 Hemoglobin: Value: 13.8; Numeric Value: 13.8; Units: g/dL; Reference Range: 13.7 - 17.5 Lymphocytes: Value: 4.8; Numeric Value: 4.8; Units: %; Reference Range: 19.0 - 53.0; Flag: abnormal MCH: Value: 30.0; Numeric Value: 30.0; Units: pg; Reference Range: 26.0 - 32.0 MCHC: Value: 33.7; Numeric Value: 33.7; Units: g/dL; Reference Range: 32.0 - 37.0 Monocytes: Value: 10.9; Numeric Value: 10.9; Units: %; Reference Range: 5.0 - 13.0 Neutrophils: Value: 83.6; Numeric Value: 83.6; Units: %; Reference Range: 34.0 - 71.0; Flag: abnormal Platelet Count: Value: 251; Numeric Value: 251.0; Units: K/uL; Reference Range: 150.0 - 400.0<br><br>RDW: Value: 12.9; Numeric Value: 12.9; Units: %; Reference Range: 10.5 - 15.5|
|2|DiagAgent|Current diagnosis: The elevated neutrophil count (83.6%) suggests an inflammatory or infectious process, supporting the possibility of appendicitis. Based on the current findings, the following additional investigation(s) are needed: CT ABD AND PELVIS WITH CONTRAST. Reason: To visualize the appendix and surrounding structures to confirm the diagnosis of appendicitis or to identify other potential causes of abdominal pain.|
| |DiagGym|… GASTROINTESTINAL: The stomach is unremarkable. Small bowel loops demonstrate normal caliber, wall thickness, and enhancement throughout. The colon and rectum are within normal limits. The appendix is fluid-filled and abnormally dilated, measuring up to 1.2 cm (series 2, image 62). There is associated fat stranding and fluid in the right lower quadrant. Findings are compatible with acute appendicitis. No drainable fluid collection or extraluminal gas is seen. …|
|3|DiagAgent|The available information is sufficient to make a diagnosis. Diagnosis: Appendicitis. Reason: The patient's clinical presentation of right lower quadrant abdominal pain, nausea, and diarrhea, combined with the CT<br><br>findings of a fluid-filled and dilated appendix with associated fat stranding and fluid in the right lower quadrant, confirm the diagnosis<br><br>of appendicitis.|
|Referenced Multi-turn Diagnostic Trajectory| | |
|Step|Content| |
|1|The patient arrives with several hours of abdominal pain, which started around the umbilicus and later migrated to the right lower quadrant. He also reports chills, nausea, diarrhea, and loss of appetite. Given this combination of symptoms, appendicitis, gastroenteritis, and other gastrointestinal causes are considered. A Complete Blood Count (CBC) is ordered to look for signs of infection or inflammation.| |
|2|CBC results show an elevated white blood cell count and significant neutrophilia, indicating an infectious or inflammatory process in the abdomen. Given these findings, the suspicion for appendicitis is heightened. To further clarify the diagnosis, a CT scan of the abdomen and pelvis with contrast is requested, aiming to directly visualize the appendix and surrounding structures.| |
|3|The CT scan reveals a dilated, fluid-filled appendix with hyperenhancing mucosa and mild fat stranding around the appendix—findings consistent<br><br>with acute appendicitis. There is no sign of perforation or abscess. Based on the patient’s clinical presentation, laboratory results, and imaging findings,<br><br>a final diagnosis of acute appendicitis is made.| |

- Supplementary Figure 3 | Interactive Diagnostic Case Study with DiagAgent: Model Trajectory and Reference Timeline. This figure illustrates a multi-turn interaction between the DiagAgent model and a simulator. The table details the agent’s step-wise reasoning, differential diagnosis, and subsequent actions (e.g., ordering lab tests). The bottom Referenced Multi-Turn Trajectory provides a ground-truth clinical timeline for comparison, demonstrating the established diagnostic process leading to the final diagnosis.

|Note ID 13309322-DS-4| | | | |
|---|---|---|---|---|
|Initial Inquiry| |Sex: Male. Chief Complaint: Left lower extremity infection. History of Present Illness: Patient sustained a left pilon fracture from a mechanical fall, treated with external fixation followed by ORIF. He was started on Keflex for concern of left lower extremity (LLE) cellulitis. Presented with progressive pain, swelling, and erythema in the LLE. Reports one area of serosanguineous drainage over the wound. Denies fevers, chills, or malaise. Able to bear full weight on LLE. Denies motor or sensory deficits distally. Past Medical History: Hypertension, hyperlipidemia. Personal History: Not provided. Family History: Non-contributory. Allergy History: No known allergies or adverse drug reactions.| | |
|Final Diagnosis| |Left lower extremity cellulitis caused by Staphylococcus aureus.| | |
|DiagAgent Interaction Messages| | | | |
|Turn|Role|Content| | |
|1|DiagAgent|Current diagnosis: Left lower extremity infection, likely cellulitis or osteomyelitis, secondary to pilon fracture and recent surgery. Based on the patient's initial presentation, the following investigation(s) should be performed: Complete Blood Count. Reason: To evaluate for signs of infection, such as leukocytosis or bandemia, which could support the diagnosis of cellulitis or osteomyelitis.| | |
| |DiagGym|Hemoglobin: 13.6 g/dL (Reference Range: 13.7 - 17.5; Flag: abnormal) Neutrophils: 63.3% (Reference Range: 34.0 - 71.0) Platelet Count: 301 K/uL (Reference Range: 150.0 - 400.0) ...| | |
|2|DiagAgent|Current diagnosis: Left lower extremity infection, likely cellulitis or osteomyelitis, supported by elevated neutrophils and normal CBC. Based on the current findings, the following additional investigation(s) are needed: WOUND CULTURE. Reason: To identify the causative organism and guide antibiotic therapy.| | |
| |DiagGym|Organism: STAPH AUREUS COAG + … Antibiotic: ERYTHROMYCIN ... Interpretation: R (Resistant)| | |
|3|DiagAgent|Current diagnosis: Left lower extremity infection, likely cellulitis or osteomyelitis, caused by Staphylococcus aureus. Based on the current findings, the following additional investigation(s) are needed: Blood Culture, Routine. Reason: To rule out bacteremia or systemic infection, which could complicate the localized infection.| | |
| |DiagGym|Comments: NO GROWTH.| | |
|4|DiagAgent|The available information is sufficient to make a diagnosis. Diagnosis: Left lower extremity infection (cellulitis or osteomyelitis) caused by Staphylococcus aureus. Reason: The patient's clinical presentation of progressive pain, swelling, and erythema, combined with the identification of Staphylococcus aureus in the wound culture, supports the diagnosis of a left lower extremity infection.| | |
|Rubrics| | | | |
|index|Rubric| |Points|Present|
|1|Accurately identifying cellulitis of the left lower extremity as the current diagnosis, based on clinical signs such as erythema, edema, warmth, and serous discharge.| |10|Yes 10|
|2|Prioritizing a Complete Blood Count (CBC) in the initial assessment to evaluate signs of infection such as leukocytosis or bandemia.| |9|Yes 9|
|3|Timely recommending a wound culture upon discovering elevated neutrophils to identify the causative microorganism and guide antibiotic therapy.| |8|Yes 8|
|4|Reasonably suggesting blood cultures to rule out bacteremia or systemic infection when Staphylococcus aureus is isolated from the wound culture.| |7|Yes 7|
|5|Timely confirming the final diagnosis after negative blood cultures, avoiding unnecessary further investigations.| |6|Yes 6|
|6|Consistently considering the possibility of osteomyelitis throughout the differential diagnosis process, but without overrelying on it and neglecting the more probable cellulitis.| |5|Yes 5|
|7|Correctly interpreting laboratory results, including recognizing elevated neutrophil percentage without misinterpreting it as systemic infection.| |4|Yes 4|
|8|Demonstrating awareness of the typical resistance patterns of *Staphylococcus aureus* in the recommendation for antibiotic selection.| |3|No 0|
|9|Appropriately considering and ruling out vascular complications, such as deep vein thrombosis, during the diagnostic process.| |2|No 0|

- Supplementary Figure 4 | Illustrative Success Case of DiagAgent Evaluated by Physician-Curated Rubrics. This figure presents a case showcasing DiagAgent’s high-quality multi-turn interaction for a lower extremity infection. The top section details the agent’s step-wise dialogue. The bottom table of Procedural Evaluation Rubrics confirms that the agent successfully satisfied the majority of high-weighted, process-oriented criteria, demonstrating procedural integrity beyond merely achieving the correct final diagnosis.

|Note ID 10172388-DS-17| | | | |
|---|---|---|---|---|
|Initial Inquiry| |Sex: Female Chief Complaint: Sudden-onset abdominal pain, hemodynamically unstable History of Present Illness: The patient experienced sudden-onset abdominal pain while driving, requiring her to pull over. She presented to the hospital with a positive FAST scan for intra-abdominal fluid and was hemodynamically unstable (SBP 80, tachycardia). She was unaware of her pregnancy prior to this event. Past Medical History: Obstetric: Spontaneous abortion x1, spontaneous vaginal delivery x1 Gynecologic: Regular menses, denies abnormal Pap smears or STIs Surgical: Laparoscopic appendectomy, breast reduction Personal History: Not specified Family History: Non-contributory Allergy History: No Known Allergies / Adverse Drug Reactions| | |
|Final Diagnosis| |Ruptured ectopic pregnancy| | |
|DiagAgent Interaction Messages| | | | |
|Turn|Role|Content| | |
|1|DiagAgent|Current diagnosis: The patient presents with sudden-onset abdominal pain and hemodynamic instability, with a positive FAST scan indicating intra-abdominal fluid. The differential includes ruptured ectopic pregnancy, ruptured ovarian cyst, or other intra-abdominal pathology. Based on the patient's initial presentation, the following investigation(s) should be performed: Human Chorionic Gonadotropin. Reason: Given the patient's history of missed menses and the possibility of pregnancy, hCG testing is essential to confirm or rule out pregnancyrelated pathology, such as ectopic pregnancy or molar pregnancy.| | |
| |DiagGym|hCG: 1876.0 mIU/mL (Reference: >25 is POSITIVE)| | |
|2|DiagAgent|Current diagnosis: The positive hCG test confirms pregnancy, making ruptured ectopic pregnancy a likely scenario. Based on the current findings, the following additional investigation(s) are needed: Pelvic Ultrasound. Reason: Pelvic ultrasound is crucial to visualize the uterus and ovaries, assess for ectopic pregnancy, and evaluate the presence of intra-abdominal fluid or masses.| | |
| |DiagGym|… In the left adnexa, there is a large amount of complex free fluid surrounding a heterogeneous rounded mass (4.2 x 3.8 x 4.7 cm). Findings are concerning for hemoperitoneum (abdominal bleeding) …| | |
|3|DiagAgent|Current diagnosis: The pelvic ultrasound findings of a heterogeneous mass in the left adnexa with surrounding free fluid strongly suggest ruptured ectopic pregnancy or hemoperitoneum. Based on the current findings, the following additional investigation(s) are needed: Complete Blood Count. Reason: A CBC is necessary to assess for anemia and leukocytosis, which are common in ectopic pregnancy or intra-abdominal pathology.| | |
| |DiagGym|… Hematocrit: 16.9% (Reference: 36.0 - 48.0; Flag: abnormal) White Blood Cells: 13.9 K/uL (Elevated) Neutrophils: 84.7% (Reference: 50.0 70.0; Flag: abnormal) …| | |
|4|DiagAgent|Current diagnosis: The CBC findings of anemia (Hgb 5.4 g/dL) and leukocytosis (WBC 13.9 K/uL) support the diagnosis of ectopic pregnancy or intra-abdominal pathology. Based on the current findings, the following additional investigation(s) are needed: Coagulation Profile. Reason: A coagulation profile is necessary to assess for coagulopathy, which can occur in ectopic pregnancy or hemoperitoneum.| | |
| |DiagGym|PTT: 31.3 sec (Reference Range: 25.0 - 36.5)| | |
|5|DiagAgent|The available information is sufficient to make a diagnosis. Diagnosis: Ruptured ectopic pregnancy or hemoperitoneum. Reason: The patient's presentation of sudden-onset abdominal pain, hemodynamic instability, positive FAST scan, positive hCG, and imaging findings of a heterogeneous mass in the left adnexa with surrounding free fluid strongly support the diagnosis of ruptured ectopic pregnancy or hemoperitoneum.| | |
|Rubrics| | | | |
|index|Rubric| |Points|Present|
|1|Identify 'Ruptured ectopic pregnancy' as the most probable diagnosis, clearly linking it to the patient's hemodynamic instability, positive pregnancy test, abdominal pain, and free fluid on ultrasound.| |10|Yes 10|
|2|Appropriately manage imaging: Recognize that further imaging (e.g., TVUS or CT) should not delay surgical intervention in an unstable patient with a positive FAST.| |8|Yes 8|
|3|Provide a concise differential diagnosis for hemoperitoneum (e.g., ruptured ovarian cyst, other non-gynecologic sources), correctly prioritizing ruptured ectopic pregnancy.| |7|Yes 7|
|4|Explain the necessity of each recommended test or step and how it will alter immediate management decisions.| |6|Yes 6|
|5|Propose a prioritized resuscitation plan: Simultaneous fluid resuscitation, immediate involvement of the OB/Gyn (surgical) team, and emergent transfer to the OR without waiting for additional tests.| |10|No 0|
|6|Recommend appropriate hemorrhagic shock resuscitation: Establish two large-bore IVs, rapid infusion of isotonic crystalloids as a bridge, initiation of massive transfusion protocol (including PRBCs, plasma, and platelets), and continuous monitoring.| |9|No 0|
|7|Order necessary labs that impact immediate management: Type and screen/crossmatch, coagulation studies (PT/INR, aPTT), Basic Metabolic Panel (BMP), Lactate, and Rh(D) typing (in addition to the provided CBC).| |8|No 0|
|8|State that Rh-D immunoglobulin should be administered if the patient is Rh-negative.| |7|No 0|
|9|Describe appropriate surgical management options (e.g., Salpingectomy vs. Salpingostomy) and note that the choice depends on intraoperative findings, patient stability, and fertility goals.| |5|No 0|

- Supplementary Figure 5 | Illustrative Fail Case of DiagAgent: Diagnostic Strength in Contrast to Management Deficit. This case study (ruptured ectopic pregnancy) highlights a current limitation of DiagAgent. While the agent’s Interactive Messages show robust step-wise diagnostic reasoning, successfully leading to the correct final diagnosis, the table of Procedural Evaluation Rubrics reveals critical omissions. Specifically, the agent fails to satisfy high-weighted criteria related to immediate emergency management (e.g., fluid resuscitation and surgical transfer). However, as a primary diagnostic model, the agent’s core capability of accurate differential diagnosis and sequential information gathering remains intact, which is its main contribution.

###### A.7 Prompt Collection

To facilitate reproducibility and provide transparency regarding our experimental setup, we detail our prompt engineering strategies and provide a full catalog of prompts used in this work.

###### A.7.1 Prompt Design Principles

Our prompt design is strictly task-oriented, categorized into three distinct phases based on their specific utility in the pipeline: Data Construction, Model Inference, and Evaluation. The prompting strategies (Zero-shot vs. One-shot) were selected to best serve the objective of each phase:

- • Data Construction & Filtering: The objective of this phase is to convert raw EHRs into high-quality, structured datasets without hallucinations or parsing errors. Consequently, we employed a hybrid strategy. We utilized One-shot prompting for tasks requiring strict adherence to complex output schemas (to minimize formatting errors) and Zero-shot prompting for semantic extraction tasks. The validity of these prompts is guaranteed by our rigorous post-generation human verification process.
- • Model Inference: For benchmarking baseline models, our primary goal is to assess their intrinsic capabilities fairly. We generally adopted a Zero-shot strategy to simulate realistic user behavior, where users typically interact with diagnostic agents without providing specific examples. Exception: The only exception is the simulation of numerical laboratory events (Prompt 8). Preliminary experiments showed that without a demonstration, baseline models generated heterogeneous units and formats, which unfairly penalized them during automated parsing. Thus, a One-shot example was introduced solely to constrain the output format for valid metric calculation.
- • Evaluation: For the “LLM-as-a-Judge” metrics, we employed Zero-shot prompts. The tasks (e.g., binary classification of correctness, counting items) are straightforward and do not require few-shot learning. The reliability of these prompts was validated through a consistency study against human experts.

- Supplementary Table 13 provides a comprehensive catalog of all prompts, their specific utility, and the corresponding prompting strategy.

Supplementary Table 13 | Catalog of all prompts used in DiagGym and DiagAgent, categorized by their specific utility in the pipeline.

Usage Category Design Purpose / Utility Prompting Strategy Ref. Data Construction & Filtering MIMIC-IV Processing

Check for data leakage (diagnosis appearing in history) Zero-shot Prompt 1 Reformat physical examination text into structured JSON Zero-shot Prompt 2

DiagAgent Training Data

Generate initial inquiry, trajectory, and diagnosis from EHRs Zero-shot Prompt 3 Quality check to ensure no diagnosis leakage in generated data Zero-shot Prompt 4

Extract structured patient profile from raw medical text One-shot Prompt 16 Extract all relevant examination names from text One-shot Prompt 17 Consistency check between patient profile and diagnostic info One-shot Prompt 18

DiagBench (OOD) Curation

###### Model Inference (System Instructions)

Simulate general examination results Zero-shot Prompt 7 Simulate numerical laboratory event results One-shot Prompt 8 Simulate free-text radiology reports Zero-shot Prompt 9

DiagGym (Simulator)

DiagAgent (Agent) Instruct LLMs to perform interactive diagnosis Zero-shot Prompt 5 Evaluation (LLM-as-a-Judge) Simulator Eval

Evaluate step-wise similarity against ground truth Zero-shot Prompt 14 Evaluate full-chain internal clinical consistency Zero-shot Prompt 15

Agent Eval (Auto Metrics)

Judge correctness of Final Diagnosis (Accuracy) Zero-shot Prompt 12 Count matching exams for Precision calculation Zero-shot Prompt 10 Count matching exams for Recall calculation Zero-shot Prompt 11 Check if recommended exam is in valid list (Hit Ratio) Zero-shot Prompt 6

Agent Eval (Rubrics) Assess if a specific rubric criterion is satisfied Zero-shot Prompt 13

- A.7.2 Full Prompt Content Prompt

- Prompt 1. Prompt to check whether the final diagnosis appears in the patient’s past medical history. You are a highly knowledgeable and detail-oriented medical expert. Your task is to analyze and compare the provided discharge diagnoses with the patient’s past medical history to determine whether any of the diagnoses in the discharge diagnoses are explicitly mentioned in the past medical history. Just output Yes or No without any other word.

INPUT DATA: Discharge Diagnoses: {discharge_diagnosis} Past Medical History: {past_medical_history}

OUTPUT FORMAT: Yes/No

Prompt

- Prompt 2. Prompt to structure physical examination results as JSON. You are a highly skilled and detail-oriented medical expert. Your task is to analyze a given physical exam report written in free text and convert it into a structured JSON format. Each JSON entry should represent a single examination that is typically completed in one step (e.g., height and weight measured together, blood pressure as a single reading). Do not split examinations into smaller components unless they are explicitly presented as separate tests in the input. Only include pre-admission physical examinations that are relevant to the initial diagnosis. Do not include any post-admission or discharge-related physical examination information. Just output the JSON without any additional text or explanation.

INPUT DATA: Physical Exam: {physical_exam}

OUTPUT FORMAT: [

{

"exam_name": "Name of the exam", "exam_results": "A string containing the results of the exam"

} ]

Prompt

- Prompt 3. Prompt to generate differential diagnosis data based on discharge notes. As an experienced physician, you will receive a patient Electronic Health Record focused on diagnosis. Your task is to:

- - Summarize only the key clinical information available prior to hospital presentation (i.e., the patient’s state before arrival at the hospital, including symptoms, history, and relevant background). Do not include any information from the hospital course, ancillary tests, laboratory or imaging results, treatments, procedures, or discharge summaries.
- - Present a Key Pertinent Results section, listing essential diagnostic tests and their results using the provided Pertinent results dict.

- - Present a Stepwise Diagnostic Reasoning Timeline, outlining the chronological diagnostic process, including the rationale for each investigation, the corresponding results, and how each step informed subsequent decisions.
- - State the final diagnosis and briefly explain the supporting evidence.

Ensure all summaries are concise, accurate, and based solely on the information provided. Do not reference images, tables, or other visual data, as these are not accessible. If the EHR is incomplete or does not meet the criteria for summarization, simply output: "I can’t."

Format to follow: ### Case Summary Provide a detailed medical history of the patient prior to hospital arrival, including chief complaint, history of present illness, past medical history, family history, and any other relevant information for initial diagnosis. Do not include any findings, investigations, or events that occurred after hospital arrival. Do not include any ancillary tests or pertinent results here. Do not mention or imply any diagnostic conclusions. Do not include any language that attributes symptoms to a specific diagnosis, even if this is present in the EHR.

- - Patient Information:
- - Chief Complaint: If none, write "None."
- - History of Present Illness: If none, write "None." This needs to be done very carefully and should only include information from before the visit.
- - Past Medical History: If none, write "None."
- - Personal History: If none, write "None."
- - Family History: If none, write "None."
- - Allergy History: If none, write "None."

...(other necessary information before hospital arrival) ### Key Pertinent Results Please output the key diagnostic tests and their results in the following json format. Copy all test names and results exactly as they appear in the given Pertinent results dict. Do not change any word: {

- "Test Name 1": "Result 1",
- "Test Name 2": "Result 2",

... }

### Stepwise Diagnostic Reasoning Timeline Present a time-ordered, step-by-step diagnostic reasoning process, as follows:

- 1. Based on the initial patient presentation, provide a preliminary diagnostic impression. Current diagnosis: [Analyses and preliminary diagnostic impression based on initial presentation]
- 2. State which diagnostic tests should be ordered next and give the detailed specific reason for each. You cannot select an examination that does not appear in pertinent results. Do not change the name from the pertinent results. Test to order: [Test Name] Reason: [Long, comprehensive and detailed specific reason for ordering the test]
- 3. For each test, copy the result exactly from the Pertinent results value, do not change any word. Test result: [Test result]
- 4. After each set of results, update the diagnostic assessment based on information gathered so far, then explain what additional tests are required. Repeat steps 2-4 as needed.
- 5. Conclude with the final diagnostic decision and the reasoning based on the available data. Diagnosis: [Final diagnosis]

- Reason: [Explanation for the final diagnosis based on all available information]
- 6. A maximum of 12 turns is allowed. The following is an example of the format to follow:

- Step 1: Current diagnosis: [Analyses and preliminary diagnostic impression based on initial presentation]

Based on the patient’s initial presentation, the following investigation(s) should be performed: [Test Name]. Reason: [Long, comprehensive and detailed reason for ordering the test] Test result: [Result]

- Step 2: Current diagnosis: [Updated analyses and diagnostic impression based on available information]

Based on the current findings, the following additional investigation(s) are needed: [Test Name]. Reason: [Long, comprehensive and detailed reason for ordering the test] Test result: [Result]

... Step n: Current diagnosis: [Final diagnostic impression] The available information is sufficient to make a diagnosis. Diagnosis: [Final diagnosis] Reason: [Explanation and justification for the final diagnosis based on the findings above] ### Final Diagnosis Integrate the patient’s clinical presentation, test results, and differential diagnosis process to summarize the final diagnosis. Briefly explain the basis for the diagnosis and highlight the key factors supporting this conclusion. ### Diagnosis results Just Output the diagnostic result without any other explanation. The following is the Electronic Health Record (EHR) of the patient: {ehr_text} The following is all the test results of the patient according to time and date. Please copy all test names and results exactly as they appear in the following dictionary: {events} Prompt

- Prompt 4. Prompt to filter high-quality data for differential diagnoses, ensuring no data leakage. Please evaluate whether the case summary provided below is qualified diagnostic task data according to the following standards. Please judge each criterion individually.

### Evaluation Criteria

- 1. Information Leakage Does the "Case Summary" section directly contain the name or diagnostic results of the "Diagnose Results"? If diagnostic results appear directly (That is, without any reasoning), it is judged as "Unqualified".
- 2. Chief Complaint Reasonableness Is the chief complaint in the "Case Summary" the patient’s subjective discomfort or disease manifestation?

If the chief complaint is a surgical procedure name or non-subjective discomfort, it is judged as "Unqualified".

Output Format:

Information Leakage: xxx Chief Complaint Reasonableness: xxx

The following is specific information: Case Summary: {case_summary}

Key Examination Results: {pertinent_results}

Diagnose Results: {diagnosis_results}

###### Prompt

- Prompt 5. Prompt to instruct LLMs to make dynamic diagnosis You are a medical AI assistant. Help the doctor with diagnosis by analyzing patient information, suggesting relevant tests, and providing a final diagnosis when sufficient information is available. RESPONSE FORMAT:

If more information is needed:

Current diagnosis: [your diagnosis according to the information provided] Based on the patient’s initial presentation, the following investigation should be performed: [one additional test] Reason: [reason for the test]

If sufficient information exists for diagnosis: The available information is sufficient to make a diagnosis. Diagnosis: [Diagnosis result] Reason: [Diagnosis reason]

- Prompt 6. Prompt to check if the examination name occurs in the key examination list Please determine if the predicted examination name matches any of the valid examination names for this case.

Predicted examination: {pred_exam} Valid examinations for this case: {gt_exam}

Please respond with "SAME" if the predicted examination matches any of the valid examinations (they refer to the same medical test or examination), or "DIFFERENT" if the predicted examination does not match any of the valid examinations.

Consider examinations as the same if they:

- 1. Are exactly the same name
- 2. Are different names for the same medical test/procedure
- 3. Are abbreviations or full forms of the same examination
- 4. Are sub-items or components of any valid examination in gt_exam
- 5. Are encompassed by any valid examination in gt_exam Please output SAME or DIFFERENT directly.

Prompt

- Prompt 7. Prompt to instruct LLMs to simulate examination results You are an expert medical AI assistant specialized in predicting medical examination results based on patient case summaries and past events. Your task is to analyze the provided patient information and predict the most likely results for a specific medical examination. Instructions:

- 1. Carefully analyze the patient case summary, including diagnosis, symptoms, and clinical presentation
- 2. Consider all past examination results and their implications
- 3. Based on the medical context, predict realistic and clinically appropriate results for the requested examination
- 4. Provide only the examination results without additional explanation or reasoning
- 5. Format your response as concise, medically accurate examination findings
- 6. If multiple measurements or findings are typical for the exam, include all relevant components
- 7. Ensure your predictions are consistent with the patient’s overall clinical picture

Patient Case Summary: {context} {past_events_text}

Current Examination to Predict: Exam name: {exam_name}

Please predict the most likely results for this examination based on the patient’s clinical information and past results. Provide only the examination results.

- Prompt 8. Prompt to instruct LLMs to simulate labevent results You are an expert medical AI assistant specialized in predicting medical examination results. Your task is to analyze patient information and predict numerical values for specific laboratory tests. CRITICAL FORMATTING REQUIREMENTS:

- - You must provide numerical values for each requested sub-test
- - Format each result as: "Sub-test Name: Numeric Value: [number] Units: [unit]"
- - Use the exact units specified for each sub-test
- - If multiple sub-tests are requested, provide each on a separate line
- - Use realistic medical values appropriate for the patient’s condition
- - Be precise with numbers (use decimals when appropriate)

For the examination “{exam_name}”, you need to provide values for these specific measurements: {subevents_text} Example format: Hemoglobin: Numeric Value: 12.5 Units: g/dL White Blood Cell Count: Numeric Value: 7200 Units: cells/µL Platelet Count: Numeric Value: 250000 Units: cells/µL Prompt

- Prompt 9. Prompt to instruct LLMs to simulate radiology results You are an expert radiologist AI assistant specialized in generating realistic radiology examination results. Your task is to analyze patient information and generate comprehensive radiology findings. CRITICAL FORMATTING REQUIREMENTS:

- - Generate a detailed and realistic radiology findings section for the specified examination
- - Include relevant anatomical findings.
- - Use appropriate medical terminology and standard radiological language
- - Provide specific details that would be clinically relevant
- - Ensure the findings are consistent with the patient’s clinical presentation For the examination “{exam_name}”, generate a comprehensive radiology report that includes:

- 1. Detailed findings of anatomical structures.
- 2. Any abnormalities or normal variations observed. The report should be professional, detailed, and clinically appropriate.

Prompt

- Prompt 10. Prompt to instruct LLMs to count the number of the predicted examinations appearing in the key examinations list Please determine how many of the recommended exams appear in the key exam list. Note that even if the expressions are different, if they refer to the same examination, they should be considered as matches.

Key exam list: {key_exam_names} Recommended exam list: {recommended_exam_names}

Please analyze each item in the recommended exam list and determine if it has a corresponding item in the key exam list (even with different expressions).

Please only output the number of matches as an integer. For example: 3

###### Prompt

- Prompt 11. Prompt to instruct LLMs to count the number of key examinations appearing the predicted examinations Please determine how many of the key exams appear in the recommended exam list. Note that even if the expressions are different, if they refer to the same examination, they should be considered as matches.

Key exam list: {key_exam_names} Recommended exam list: {recommended_exam_names}

Please analyze each item in the key exam list and determine if it has a corresponding item in the recommended exam list (even with different expressions).

Please only output the number of matches as an integer. For example: 2

Prompt

- Prompt 12. Prompt to instruct LLMs to assess the accuracy of the predicted final diagnosis. # Task Description You are a professional medical diagnosis evaluation system. Now, you will receive two diagnosis results: one is the diagnosis predicted by the model ([pred_diag]), and the other is the verified correct diagnosis ([gt_diag]). Your task is to judge whether the model-predicted diagnosis([pred_diag]) is correct. When evaluating, please consider the following factors:

- 1.The same disease may have multiple aliases, for example, “Heart disease” may also be called “Cardiac disease”.
- 2.There may be diversity in language expression, for example, “heart attack” and “myocardial infarction” may refer to the same disease.
- 3.Only judge whether the diagnosis result is correct, information such as the cause of the disease, symptoms, and treatment recommendations are not included in the evaluation scope.
- 4.If the correct diagnosis[gt_diag] is included in the predicted diagnosis but some additional complications are mentioned, it is also considered correct

# Output Requirements Only output your judgment result on the model-predicted [pred_diag] as “Correct|Wrong”, do not output any other content.

# Format to Follow: [Correct|Wrong]

Below is the diagnosis result predicted by the model and the correct diagnosis: {pred_diag} {gt_diag}

###### Prompt 13. Prompt to Assess Agent’s Adherence to Diagnostic Rubric Given a list of messages representing a medical conversation, evaluate whether the agent’s diagnostic trajectory, represented by its recommendations throughout the messages, meets the provided criterion.

- - Output "Yes" if the agent’s recommendations throughout the messages have satisfied the criterion.
- - Output "No" if the agent’s recommendations throughout the messages have not satisfied the criterion.

Conversation: {messages} Criterion: {criterion} Please only respond with "Yes" or "No".

Prompt

###### Prompt 14. Prompt to evaluate step-wise similarity between generated examination result and ground truth examination result You are a medical expert evaluating similarity between AI predictions and ground truth results. Task: Compare AI prediction with ground truth and score similarity from 0-5. Scoring (0-5):

- - 0: Completely different, opposite findings
- - 1: Major differences, different clinical implications
- - 2: Significant differences, different interpretations
- - 3: Moderate similarity, similar clinical direction
- - 4: High similarity, minor differences
- - 5: Excellent similarity, essentially equivalent Input:
- - Exam: {exam_name}
- - Ground Truth: {ground_truth}
- - AI Prediction: {prediction} Focus: Direct comparison of values/findings and clinical equivalence.

Output Format: {

"score": [0-5], "explanation": "Brief explanation of similarity assessment and score reasoning"

}

Prompt

###### Prompt 15. Prompt to evaluate full-chain consistency You are a medical expert. Task: Determine if the complete AI prediction chain is internally consistent and aligns with the patient case (0=Fail, 1=Pass). You may use the ground truth chain (GT) as a reference. The evaluation criteria should be lenient, allowing for flexibility and minor discrepancies. If the AI prediction chain is similar to the GT chain, it should be considered correct. If the AI prediction chain is not similar to the

GT chain but contains no significant factual errors, aligns with the patient case, and maintains clinical coherence, it should still be considered correct. Only assign a fail (0) if there are clear, significant factual errors or contradictions that severely undermine the clinical coherence or logic of the AI prediction chain. Minor differences or deviations are acceptable and should not lead to a fail.

Input: Case Summary: {case_summary} AI Prediction Chain: {predicted_chain} Ground Truth Chain: {ground_truth_chain} Evaluation: Check for:

- - Internal contradictions between different results in the AI prediction chain
- - Alignment with the patient’s clinical condition
- - Comparison with the ground truth chain (GT) for reference
- - Overall clinical coherence and logic Scoring:
- - 0 (Fail): Only assign a fail if there are clear and significant internal contradictions, conflicts with the patient case or GT, or clinically incoherent reasoning with factual errors that make the prediction chain unreliable.
- - 1 (Pass): Assign a pass if the AI prediction chain is internally consistent, aligns with the patient case, and forms a coherent clinical picture. This includes cases where the AI prediction chain differs from the GT chain, as long as it contains no significant factual errors and is clinically coherent. Even if there are notable differences from the GT, it should still pass unless there are explicit, critical contradictions or errors.

Output Format: {

"score": [0 or 1], "explanation": "Brief analysis of chain coherence, comparison with ground truth, and reasoning for pass/fail decision"

}

###### Prompt

- Prompt 16. Prompt to extract patient profile from raw medical text. You are an expert clinical documentation specialist. Your task is to extract a Patient Admission Profile and a Final Diagnosis from the provided [Medical Text], mimicking the strict style and format of a MIMIC-IV Discharge Note. Instructions:

- 1. Sufficiency Check: - Read the Case Report. If the information is too sparse (e.g., extremely short abstract, non-clinical text), output only: "Insufficient".

- If sufficient, proceed to generate the note.

- 2. Extraction Scope (Strict Truncation):

- - Extract sections in this exact order: Header -> History (HPI/PMH/SH/FH) -> Physical Exam -> Final Diagnosis.
- - CRITICAL: Stop extraction immediately after the "Physical Exam". Do NOT include "Pertinent Results" (Labs/Imaging data tables) or the "Brief Hospital Course" (daily treatment progress).

- - Jump directly to the "Final Diagnosis" section at the end.

- 3. Format & Style:

- - Use the exact headers from the Example below.
- - Use ‘___‘ for any missing privacy info (Name, Unit No, Dates, Attending). - Physical Exam: Group findings by system (General, HEENT, CV, Lungs, Abdomen, Ext, Neuro). Distinguish "Admission Exam" if specified.

- 4. Diagnosis Constraint:

- Final Diagnosis: Identify the ONE single primary diagnosis responsible for the admission or the main condition confirmed by the end of the case. Do not list multiple secondary diagnoses here unless they are inseparable from the main condition.

### Meidcal Text medical_text

Please output the patient profile directly.

Prompt

- Prompt 17. Prompt to extract relevant examination names from unstructured clinical text. As an experienced physician, you will receive a Transcribed Medical Note. Your task is to:

- - Identify and extract the diagnostic examinations mentioned throughout the entire Transcribed Medical Note. You must strictly limit your extraction to physical exams, Laboratory Events, Microbiology Events, or Radiology Events. Ignore procedures (like surgeries), or unrelated tests. No other types of examinations are permitted. Related events can be found in the Related Events Section.
- - Extract every occurrence of these events, regardless of whether the result was normal, abnormal, or incidental. The goal is a complete inventory of tests performed.
- - A maximum of 30 most related events. Similar events can be combined to avoid duplication.

Output format: A list of all related examination names. ["Complete Blood Count", "MR BRAIN", ...]

The following is the Transcribed Medical Note of the patient: {case_report_text} Please output the list directly without any other explanation. Prompt

- Prompt 18. Prompt to assess logical consistency between patient profile and diagnostic trajectory. You are a medical expert quality controller. Task: Determine if the "Profile Diagnosis" and the "Diagnostic Info" refer to the same or synonymous medical condition. Profile Diagnosis:

{profile_diagnosis} Diagnostic Info / Trajectory: {diagnose_info_text} Question: Is the final diagnosis in the [Diagnostic Info] consistent with the [Profile Diagnosis]? Focus on clinical equivalence (e.g., "CML" == "Chronic Myelogenous Leukemia"). Answer only "Yes" or "No".

