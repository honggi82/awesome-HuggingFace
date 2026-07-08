## Aryabhata: An exam-focused language model for JEE Math

Ritvik Rastogi PhysicsWallah ritvik.rastogi@pw.live

Sachin Dharashivkar AthenaAgent sachin@athenaagent.com

Sandeep Varma PhysicsWallah sandeep.varma@pw.live

# arXiv:2508.08665v2[cs.AI]13Aug2025

### Abstract

We present Aryabhata 1.0, a compact 7B parameter math reasoning model optimized for the Indian academic exam, the Joint Entrance Examination (JEE). Despite rapid progress in large language models (LLMs), current models often remain unsuitable for educational use. Aryabhata 1.0 is built by merging strong open-weight reasoning models, followed by supervised fine-tuning (SFT) with curriculum learning on verified chain-of-thought (CoT) traces curated through best-of-n rejection sampling. To further boost performance, we apply reinforcement learning with verifiable rewards (RLVR) using A2C objective with grouprelative advantage estimation alongwith novel exploration strategies such as Adaptive Group Resizing and Temperature Scaling. Evaluated on both in-distribution (JEE Main 2025) and out-of-distribution (MATH, GSM8K) benchmarks, Aryabhata outperforms existing models in accuracy and efficiency, while offering pedagogically useful step-by-step reasoning. We release Aryabhata as a foundation model to advance exam-centric, open-source small language models. This marks our first open release for community feedback (Aryabhata 1.0 on Hugging Face); PW is actively training future models to further improve learning outcomes for students.

### 1 Introduction

Large language models (LLMs) have shown remarkable progress in mathematical reasoning, but most existing systems fall short in supporting student learning in academic settings like India’s Joint Entrance Examination (JEE). These exams require not only accurate solutions but also transparent and precise reasoning that aids student understanding and long-term learning.

We observe three broad classes of models in this space:

Correspondence to ritvik.rastogi@pw.live

Non-reasoning models Instruction-tuned models (e.g., GPT-4o) were largely inaccurate on rigorous math exams like JEE. These models failed to perform multi-step reasoning, often guessing answers or relying on shallow pattern matching.

Early reasoning models introduced long chainof-thought (CoT) reasoning to improve accuracy, with examples including OpenAI o1 (OpenAI,

- 2024) and DeepSeek R1 (DeepSeek-AI et al.,
- 2025). While these models were more accurate than non-reasoning baselines, they remained impractical in real-world educational settings. For instance, o1 (OpenAI, 2024) did not expose its reasoning trace and provided just a summary of them, while DeepSeek R1 (DeepSeek-AI et al., 2025) produced long, nonlinear traces that made it difficult for students to follow the logic. Moreover, both models were relatively slow, generating lengthy explanations that consumed a significant amount of tokens and latency.

Modern reasoning models such as OpenAI o4mini (OpenAI, 2025), Gemini 2.5 (Comanici et al., 2025), and the updated version of DeepSeek R1 (DeepSeek-AI et al., 2025) have improved further in raw accuracy and generation speed. However, pedagogical usability remains limited. For example, o4-mini (OpenAI, 2025) provides just a summary of its reasoning traces, while Gemini (Comanici et al., 2025) and DeepSeek R1 (DeepSeek-AI et al., 2025) still produce nonlinear, self-correcting reasoning paths that confuse learners rather than clarify concepts. (Samples are provided in Appendix D.)

In this work, we present Aryabhata 1.0, a compact and open 7B parameter model tailored for math reasoning in Indian competitive exams. Built via model merging and fine-tuned with domainaligned data, Aryabhata combines accuracy, transparency, and efficiency, making it a viable foundation for educational AI applications.

### 2 Related Work

Current math LLMs built on open-weight backbones have primarily leveraged Imitation Learning, Supervised Fine Tuning, and Reinforcement Learning to enhance chain-of-thought mathematical reasoning.

For instance DeepSeekMath (Shao et al., 2024), introduced in early 2024, advanced the capabilities of open weight models by pre-training on hundreds of billions of math-focused tokens and pioneering Group Relative Policy Optimization (GRPO).

Qwen-2.5-Math-7B (Yang et al., 2024) is a math-specialized 7B instruction-tuned model that supports chain-of-thought (CoT) and tool-integrated reasoning (TIR) across both English and Chinese problem sets.

NVIDIA’s AceMath-7B-Instruct (Liu et al., 2025a), derived from Qwen, advances its performance further through a multi-stage SFT training pipeline designed to improve both mathematical and reasoning accuracy on multiple benchmarks and edging close to much larger models at 72B scale.

Meanwhile, DeepSeek-R1 (DeepSeek-AI et al., 2025) introduced a pure RL-based reasoning model trained with GRPO-style verifiable rewards, achieving impressive results. Its distilled variants (DeepSeek-R1-Distill-Qwen-7B (DeepSeekAI et al., 2025)) inherit reasoning performance via long CoT.

The AceReason-Nemotron-7B (Liu et al.,

- 2025a) demonstrates that large-scale reinforcement learning can significantly enhance the reasoning capabilities of strong small- and mid-sized models by first training on math-only prompts, then on code-only prompts.

The AceReason-Nemotron-1.1-7B (Liu et al.,

- 2025b) synergizes SFT and RL fine-tuning by employing a stage-wise RL approach on math-only and code-only prompts.

Our approach builds on these lines by merging models for hybrid capabilities (symbolic fluency + coherent CoT), followed by rejection-sampled SFT and RL with verifiable rewards, preserving both performance and efficiency in a compact model.

### 3 Methodology

The overall process can be categorized in the following four stages:

#### 3.1 Model Merging

The development of LLMs has seen a transition from System 1 (quick thinking) to System 2 (deliberate, methodical) reasoning, each with distinct advantages (Wu et al., 2025). While System 1 models excel at producing fluent answers with low latency, they often lack the depth required for complex reasoning. In contrast, System 2 models are capable of iterative self-correction and structured reasoning, but suffer from inefficiencies due to verbose or redundant CoT traces.

To address this challenge, Kimi k1.5 (Team et al., 2025) introduced the concept of merging reasoning and non-reasoning models, which was further explored by Wu et al. (2025). Building on this intuition, we carefully selected three distinct LLMs, each sharing the same base architecture.

- • Qwen2.5-Math-7B-Instruct (Yang et al., 2024), a strong open source mathematical LLM providing solid baseline capabilities and fundamental math fluency.
- • AceMath-7B-Instruct (Liu et al., 2025a) a version of Qwen 2.5 Math that was further finetuned by NVIDIA, significantly enhancing its accuracy on mathematical benchmarks.
- • DeepSeek-R1-Distill-Qwen-7B (DeepSeekAI et al., 2025), a long-form reasoning model derived by fine-tuning Qwen 2.5 Math on reasoning traces distilled from DeepSeek R1 (DeepSeek-AI et al., 2025).

We apply linear merging (Wortsman et al., 2022) to combine the models using the MergeKit (Goddard et al., 2024) framework.

Let θ1, θ2, θ3 be the parameters of Qwen, Ace, and DeepSeek, respectively. We compute:

θmerged = αθ1+βθ2+γθ3, where α+β+γ = 1

We select weights α,β,γ empirically based on the held-out math reasoning tasks. Final weights favor quickly addressing simpler problems while also performing methodical, multi-step analysis for more complex mathematical challenges.

#### 3.2 Data Curation

High-quality, domain-aligned data is essential for training effective mathematical reasoning models. To this end, we relied on a proprietary corpus curated by the subject matter experts and educators at

Topic %age Application of Derivatives 4.50% Application of Integrals 2.27% Binomial Theorem 2.37% Circles 2.85% Complex Numbers &

Quadratic Equations 6.00% Conic Section 7.55% Continuity and Differentiability 2.71% Definite Integration 2.45% Determinants 3.04% Differential Equations 3.77% Indefinite Integration 3.26% Inverse Trigonometric Functions 5.31% Limits and Derivatives 3.88% Matrices 2.46% Permutations and Combinations 4.23% Probability 5.69% Quadratic Equations 4.45% Relations and Functions 2.24% Sequence and Series 2.75% Sets 1.04% Statistics 1.89% Straight Lines 2.31% Three Dimensional Geometry 3.92% Trigonometric Functions 4.51% Vector Algebra 2.89% Miscellaneous 11.65%

Table 1: Topic-wise Question Distribution

PhysicsWallah, ensuring close alignment with the Indian Joint Entrance Examination (JEE) standards. This dataset represents years of academic effort and is considered the core intellectual property of PhysicsWallah. As such, we do not publicly release the training data.

We parsed approximately 250,000 raw questions from internal databases. To ensure syntactic coherence and semantic relevance, we applied the following filtering steps:

- • Removed diagram-based questions, which require multimodal reasoning not supported by current text-only models.
- • Filtered out non-English or poorly formatted questions.
- • Stripped all answer options from the remaining questions to frame the task as open-ended generation rather than classification. This design choice was also explored by Chandak

et al. (2025)

• Since we stripped options from the questions, we removed the questions which relied on options to be answered such as "which of the following is true"

To standardize and clean raw question-answer pairs, we designed a structured prompt (see Appendix A) that extracts the core question, normalizes the answer format, identifies dependencies and detects the question language, using OpenAI o4mini.

This process resulted in a clean dataset of around 130,000 questions suitable for the generation of further chain of thought. The topic-wise distribution of questions is outlined in Table 1.

3.3 Supervised Fine-Tuning with Rejection Sampling

To generate high-quality chain-of-thought (CoT) supervision, we employed best-of-4 rejection sampling using the merged model. For each curated question x, we sampled four CoT responses {y1,y2,y3,y4}, and selected only those completions whose final answer matched the known correct answer i.e. GT(x), using Algorithm 1. This filtering process ensures logical correctness and minimizes noisy supervision signals.

We then grouped the questions based on how many of the four generations lead to the correct answers and selected samples for curriculum-style supervised fine-tuning (Bengio et al., 2009), i.e., beginning the training with easier samples (e.g., 4/4 correct) and gradually introducing harder examples (e.g., 3/4, 2/4, 1/4 correct). This curriculum-based training stabilizes early learning and improves generalization on harder problems.

Let Dsft = {(x(i),y(i))}Ni=1 denote the dataset of input questions and their corresponding verified CoT completions. The supervised fine-tuning objective minimizes the standard next-token prediction loss:

T

log(pθ(yt | x,y<t)) (1)

##### LSFT = −

t=1

(x,y)∈Dsft

where yt is the tth token of the target CoT sequence, and pθ is the model’s probability distribution parameterized by θ.

In total, we obtained approximately 350,000 verified CoTs across around 100,000 questions, which

#### Correct CoTs # Questions Total CoTs Usage

- 0 31,470 0 Used in RLVR only
- 1 9,647 9,647 SFT
- 2 9,066 18,132 SFT
- 3 12,643 37,929 SFT
- 4 67,247 268,988 10% sampled for SFT

Table 2: Chain-of-Thought generation outcomes from best-of-4 sampling.

were sampled to serve as the core training corpus for SFT, as detailed in Table 2. The 0/4 cases were retained for downstream reinforcement learning with verifiable rewards (RLVR) to further improve coverage and robustness in challenging problem spaces.

We used Parameter Efficient Finetuning, particulary Low-Rank Adaptation (Hu et al., 2021) during SFT using peft (Mangrulkar et al., 2022) library, the training parameters are mentioned in Appendix C.

3.4 Reinforcement Learning with Verifiable Rewards

We extend Reinforcement Learning with Verifiable Rewards (RLVR) (Lambert et al., 2025) by incorporating group-based advantage estimation (Shao et al., 2024) within an Advantage ActorCritic (A2C) framework (Mnih et al., 2016) .

3.4.1 Group-Relative Policy Optimization Our approach optimizes the following A2C objective with group-relative advantage estimation:

JA2C(θ) =

E(αi) ∼ πθ

G

1 G

1 |αi|

log πθ(αi) · A˜i

i=1

We optimize the A2C objective over G sampled response sequences αi, applying length-normalized gradients weighted by sequence-level advantages A˜i computed through group-relative advantage estimation.

Binary Reward Structure: We employ a simple binary reward that provides unambiguous feedback for mathematical reasoning:

Ri =

1 if the final answer is correct 0 if the final answer is incorrect

Group Advantage Estimation The advantage function is computed using group-relative normalization:

Aˆi,t = Riσ−R¯group

group

where R¯group is the mean reward across all solutions in the group and σgroup is the standard deviation.

Key Benefits: This group-relative baseline offers several advantages:

- • Reduced variance: Group comparison stabilizes gradient estimates
- • Simplified training: Eliminates need for KL divergence constraints or probability ratio clipping
- • Natural compatibility: Works seamlessly with binary rewards, common in mathematical reasoning tasks

#### 3.4.2 Exploration Strategies

Adaptive Group Sizing: Unlike fixed group sizes in standard GRPO implementations (von Werra et al. (2020), Sheng et al. (2024), Daniel Han and team (2023)), we dynamically adjust group size based on problem difficulty. Starting with a group size of 8 for simpler problems, we scale up to a group size of 64 for harder ones.

The dynamic group size follows:

Gd = 8 × 2k

where k ∈ {0,1,2,3} is determined by the group average reward R¯group. When performance drops below preset thresholds, we increase k, scaling groups as: 8 → 16 → 32 → 64.

This adaptive scaling improves sampling diversity and advantage estimation stability for challenging problems while efficiently allocating computational resources.

Progressive Temperature Scaling: We continuously increase the sampling temperature from 0.6

to 1.0 throughout training, this was explored in contemporary works like POLARIS (An et al., 2025). This progressive scaling balances exploitation and exploration:

- • Initial phase: Low temperature (0.6) promotes training stability through conservative sampling
- • Progressive increase: Temperature gradually rises, encouraging more diverse solution exploration
- • Final phase: Temperature reaches 1.0, enabling much more exploration of the action space compared to lower temperatures.

Curriculum-Based Sampling: We filter training samples to focus on an optimal difficulty range, removing both trivial and intractable problems:

- • Too easy: Provide minimal learning signal due to high success rates
- • Too hard: Introduce noise through consistently low performance

Our filtering uses a difficulty assessment function fdifficulty(x) based on model success rates:

Dtfiltered = {x ∈ Dt : αmin ≤ fdifficulty(x) ≤ αmax}

This curriculum approach concentrates computational resources on problems that maximize learning progress.

- 3.4.3 Training Configuration and Hyperparameters

Our reinforcement learning implementation employs carefully tuned hyperparameters optimized for mathematical reasoning tasks while maintaining computational efficiency within hardware constraints. The training configuration incorporates modern optimization techniques and memoryefficient strategies to enable stable convergence at scale.

Optimization Configuration: We utilize the Adam optimizer (Kingma and Ba, 2017) with a conservative learning rate of 1 × 10−6 to ensure stable policy gradient updates throughout the training process.

Memory and Precision Management: Training is conducted using bfloat16 (BF16) mixed precision arithmetic, which provides substantial memory

savings while maintaining numerical stability for gradient computations. Gradient checkpointing is employed to further reduce memory consumption during backpropagation, enabling training of larger models within available GPU memory constraints.

Sequence and Batch Configuration: The model operates within a maximum context length of 4,096 tokens, providing sufficient capacity for complex multi-step mathematical reasoning while maintaining computational tractability.

### 4 Evaluation

We evaluated Aryabhata 1.0 across both indistribution and out-of-distribution math benchmarks to assess its accuracy and efficiency in solving problems at scale.

We evaluate model-generated solutions using the pass@1 accuracy. The solutions are generated using greedy decoding (temperature = 0). To determine whether a predicted answer matches the ground-truth answer for a question, we follow the pipeline described in the Algorithm 1.

Algorithm 1 Answer Matching Procedure

- 1: Input: Predicted answer ap, Ground-truth answer ag, Options (if any)
- 2: Output: Match status (True / False)
- 3: if ap = ag or sympy_latex_match(ap, ag) then
- 4: return True
- 5: end if
- 6: if option/identifier from ap == option/identifier from ag then
- 7: return True
- 8: end if
- 9: Query LLM judge with ap, ag, and options (if any)
- 10: if LLM returns YES then
- 11: return True
- 12: else
- 13: return False
- 14: end if

Depending on whether the question is Multiple Choice Question or a Numerical Answer Type, we use different prompts to query the judge model (GPT-4o-mini). The prompts are provided in Table 6.

[Figure 1]

Figure 1: Scatter plots showing Accuracy vs. Tokens for JEE Jan and JEE Apr.

#### Model MATH 500 GSM8K

Aryabhatta 1.0 83.6 94.8 Qwen/Qwen2.5-Math-7B-Instruct 66.0 94.7 nvidia/AceMath-7B-Instruct 80.6 93.4 GPT-4o 69.2 94.6 deepseek-ai/DeepSeek-R1-Distill-Qwen-7B 85.2 69.7 nvidia/AceReason-Nemotron-7B 84.2 76.5 nvidia/AceReason-Nemotron-1.1-7B 85.4 93.1 GPT-4.1 86.6 94.0 o4-mini 94.8 90.1 Gemini 2.5 Flash 93.6 85.1

Table 3: Performance comparison on MATH 500 and GSM8K benchmarks

#### 4.1 In-Distribution Evaluation: JEE Main 2025

To measure performance in familiar distribution settings, we evaluate Aryabhata on the JEE Main 2025 exam. The January session contains 250 questions (10 papers with 25 questions each), while the April session comprises 225 questions (9 papers with 25 questions each), all sourced from official exam papers.

Figure 1 shows that Aryabhata 1.0 achieves an accuracy of 86.0% on the January session and 90.2% on the April session, while maintaining token efficiency with an average of approximately ~2K tokens per response.

Compared to both open-weight and proprietary models, Aryabhata outperforms all baselines in accuracy while remaining competitive in inference cost.

#### 4.2 Out-of-Distribution Evaluation

To evaluate generalization beyond the fine-tuning distribution, we benchmark Aryabhata 1.0 on the following datasets:

- • MATH 500: A curated benchmark of 500 competition-style problems drawn from the larger MATH dataset originally introduced by Hendrycks et al. (2021).
- • GSM8K (Cobbe et al., 2021): A widely used benchmark of grade school math word problems.

Table 3 shows that Aryabhata demonstrates competitive generalization to unseen tasks of comparable difficulty, outperforming its base models on both MATH and GSM8K.

### Conclusion and Future Work

In this work, we introduced Aryabhata 1.0, a compact open source model with 7B parameters for

mathematical reasoning, specifically designed for the Indian competitive exam ecosystem. By merging diverse mathematical LLMs and fine-tuning on carefully curated and verified domain-specific data, Aryabhata achieves state-of-the-art performance on in-distribution benchmarks such as JEE Main, while demonstrating competitive generalization to out-of-distribution tasks like MATH and GSM8K.

Looking ahead, we plan to: Expand coverage to Physics and Chemistry, building similar reasoning capabilities in other STEM domains. Scale to the full syllabus across Foundation, JEE (Main & Advanced), and NEET, enabling end-to-end subjectlevel assistance. Develop a family of exam-centric, open source small language models (SLMs) that are compact, efficient, and aligned to Indian education standards.

We believe that this direction will empower millions of students with accessible and curriculumaligned AI tools that complement classroom learning and personalized preparation.

### Acknowledgments

We thank Tejas Chaudhari and Vishal Singh for their efforts in creating evaluation datasets and writing model evaluation scripts. Shubham Choudhari contributed to building the training pipelines and running reinforcement learning experiments. Archit Singhai and Chinmay Karkar were instrumental in exploring and benchmarking existing RL libraries.

### References

Chenxin An, Zhihui Xie, Xiaonan Li, Lei Li, Jun Zhang, Shansan Gong, Ming Zhong, Jingjing Xu, Xipeng Qiu, Mingxuan Wang, and Lingpeng Kong. 2025. Polaris: A post-training recipe for scaling reinforcement learning on advanced reasoning models.

Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. 2009. Curriculum learning. In Proceedings of the 26th International Conference on Machine Learning (ICML), volume 382 of ACM International Conference Proceeding Series, pages 41–48. ACM.

Nikhil Chandak, Shashwat Goel, Ameya Prabhu, Moritz Hardt, and Jonas Geiping. 2025. Answer matching outperforms multiple choice for language model evaluation. Preprint, arXiv:2507.02856.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman.

2021. Training verifiers to solve math word problems. Preprint, arXiv:2110.14168.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, Luke Marris, Sam Petulla, Colin Gaffney, Asaf Aharoni, Nathan Lintz, Tiago Cardal Pais, Henrik Jacobsson, Idan Szpektor, Nan-Jiang Jiang, and 3290 others. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. Preprint, arXiv:2507.06261.

Michael Han Daniel Han and Unsloth team. 2023. Unsloth.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, and 181 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. Preprint, arXiv:2501.12948.

Charles Goddard, Shamane Siriwardhana, Malikeh Ehghaghi, Luke Meyers, Vladimir Karpukhin, Brian Benedict, Mark McQuade, and Jacob Solawetz. 2024. Arcee’s MergeKit: A toolkit for merging large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 477–485, Miami, Florida, US. Association for Computational Linguistics.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. Preprint, arXiv:2103.03874.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. Preprint, arXiv:2106.09685.

Diederik P. Kingma and Jimmy Ba. 2017. Adam: A method for stochastic optimization. Preprint, arXiv:1412.6980.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, and 4 others. 2025. Tulu 3: Pushing frontiers in open language model post-training. Preprint, arXiv:2411.15124.

Zihan Liu, Yang Chen, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. 2025a. Acemath: Advancing frontier math reasoning with post-training and reward modeling. Preprint, arXiv:2412.15084.

Zihan Liu, Zhuolin Yang, Yang Chen, Chankyu Lee, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. 2025b. Acereason-nemotron 1.1: Advancing math and code reasoning through sft and rl synergy. Preprint, arXiv:2506.13284.

Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, and Benjamin Bossan. 2022. PEFT: State-of-the-art parameterefficient fine-tuning methods. https://github. com/huggingface/peft.

Volodymyr Mnih, Adrià Puigdomènech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. 2016. Asynchronous methods for deep reinforcement learning. Preprint, arXiv:1602.01783.

- OpenAI. 2024. Openai o1 model. Accessed: 2025-0805.
- OpenAI. 2025. Introducing openai o3 and o4-mini. Accessed: 2025-08-05.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2024. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, and 77 others. 2025. Kimi k1.5: Scaling reinforcement learning with llms. Preprint, arXiv:2501.12599.

Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan Lambert, Shengyi Huang, Kashif Rasul, and Quentin Gallouédec. 2020. Trl: Transformer reinforcement learning. https://github.com/huggingface/trl.

Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. 2022. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. Preprint, arXiv:2203.05482.

Han Wu, Yuxuan Yao, Shuqi Liu, Zehua Liu, Xiaojin Fu, Xiongwei Han, Xing Li, Hui-Ling Zhen, Tao Zhong, and Mingxuan Yuan. 2025. Unlocking efficient longto-short llm reasoning with model merging. Preprint, arXiv:2503.20641.

An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, Keming Lu, Mingfeng Xue, Runji Lin, Tianyu Liu, Xingzhang Ren, and Zhenru Zhang. 2024. Qwen2.5-math technical report: Toward mathematical expert model via self-improvement. Preprint, arXiv:2409.12122.

- A Prompt for Question Cleaning The prompt for question cleaning is provided in Table 7
- B Prompts for Answer Matching The prompts for answer matching are provided in Table 6
- C Hyper-parameters for Supervised Fine Tuning

The hyper-parameters for LoRA are provided in the Table 4 and the hyper-parameters for SFT are provided in the Table 5.

Parameter Value

Rank 128 LoRA Alpha 128 LoRA Dropout 0.1 Bias none Target Modules {q_proj, k_proj, v_proj, o_proj,

gate_proj, up_proj, down_proj, embeddings}

Table 4: PEFT configuration using LoRA.

Parameter Value

Precision bfloat16 Max Sequence Length 16,384 Batch Size (per device) 1 Gradient Accumulation Steps 16 Effective Batch Size 16 Number of Epochs 3 Initial Learning Rate 2 × 10−5 Final Learning Rate 2 × 10−7 Learning Rate Scheduler Linear Optimizer AdamW (8-bit) Warmup Steps 5 Packing False Logging Steps 1 WandB Reporting Enabled

Table 5: Training configuration used for supervised fine-tuning.

- D Example Model Responses

The sample question along with its correct answer is presented in Figure 2. The response generated by GPT-4o is shown in Figure 3. The response produced by DeepSeek R1 Distill Qwen 7B is illustrated across Figures 4, 5, and 6. The response from Aryabhata 1.0 is depicted in Figure 7.

[Figure 2]

- Figure 2: Sample question with the correct answer

|MCQ<br><br>|Numerical|
|---|---|
|System Prompt:<br><br>You are checking an MCQ. Given the list of options, determine if answer 1 and answer 2 are the same. Answer 1 is the same as answer 2 only if all the options match. Reason step-by-step and put the final answer YES or NO in \boxed{}.<br><br>|System Prompt:<br><br>You are checking an exam. For a given question, determine if answer 1 and answer 2 are the same. Since the answers are for the same question, you can assume similar context for both answers and make appropriate assumptions when checking if they are the same. Reason step-by-step and put the final answerYES or NO in \boxed{}.|
|User Prompt:<br><br>Options:<br><br>A: <Option 1><br>B: <Option 2><br>C: <Option 3><br>D: <Option 4><br><br><br>answer 1: <Correct Answer><br>answer 2: <Predicted Answer><br>|User Prompt:<br><br>answer 1: <Correct Answer><br>answer 2: <Predicted Answer><br>|

Table 6: Prompts used for Answer Matching

[Figure 3]

###### Figure 3: Response from GPT-4o (Part 1 of 1)

[Figure 4]

###### Figure 4: Response from DeepSeek R1 Distill Qwen 7B (Part 1 of 3)

[Figure 5]

###### Figure 5: Response from DeepSeek R1 Distill Qwen 7B (Part 2 of 3)

[Figure 6]

###### Figure 6: Response from DeepSeek R1 Distill Qwen 7B (Part 3 of 3)

[Figure 7]

###### Figure 7: Response from Aryabhata 1.0 (Part 1 of 1)

|Clean and standardize math questions by removing multiple-choice options, normalizing the answer format, identifying dependencies, and determining the language. For any answers expressed in MathML, convert them to LaTeX. Conversion of MathML in the<br><br>**question** is *not required* (but preserve LaTeX if already present). Additionally, provide a clear **step-by-step reasoning** explaining how each part of the output was derived.<br><br>### Instructions:<br><br>1. Identify and extract the core question text:<br><br>* Remove all multiple-choice options (e.g., A–D or 1–4), ensuring the main question remains grammatically and semantically intact.<br>* Preserve existing LaTeX in the question.<br>* Do **not** convert MathML in the question. It may be retained as-is.<br><br><br>2. Normalize the answer:<br><br>* If the answer is given as an option label (e.g., "Answer: B"), replace it with the corresponding value from the provided options.<br>* If the answer is already a value, retain it.<br>* If the answer is in MathML, convert it to LaTeX.<br><br><br>3. Determine dependency flags:<br><br>* **Option-dependent:** Is the question understandable and solvable without access to the answer options? Mark `True` if the question lacks key information without them; otherwise, `False`.<br>* **Diagram-dependent:** Does the question reference or rely on a diagram, figure, or visual element? Mark `True` or `False`.<br><br><br>4. Identify the language:<br><br>* Detect and report the language of the question text (e.g., `English`, `Hindi`, `Tamil`, etc.).<br><br>5. Provide reasoning:<br><br><br>* For each output field (question, answer, flags, language), include a clear explanation of how the output was determined.<br>* The reasoning should follow a logical step-by-step format, but does **not** need to be wrapped in any special `<reason>` block.<br><br><br># Output Format<br><br><question> cleaned question </question> <answer> cleaned answer </answer> <option_dependent> True/False </option_dependent> <diagram_dependent> True/False </diagram_dependent> <language> detected language </language><br><br>* All math in the **answer** must be in LaTeX.<br>* There should be **no references** to original option labels (e.g., "A", "1", or "Option B").<br>* Ensure the cleaned question is coherent, self-contained, and grammatically correct.<br>* The reasoning can be in free-text form and must explain how each part of the output was derived.<br>|
|---|

###### Table 7: Prompt used for Question Cleaning (Part 1 of 2)

|### Example 1 Input: What is the derivative of \(x^2 + 3x + 5\)?<br><br>A) \(2x + 3\)<br>B) \(x + 3\)<br>C) \(x^2 + 3\)<br>D) \(2x + 5\) Answer: A<br><br><br>Output: <question> What is the derivative of \(x^2 + 3x + 5\)? </question> <answer> \(2x + 3\) </answer> <option_dependent> False </option_dependent> <diagram_dependent> False </diagram_dependent> <language> English </language><br><br>### Example 2 Input: <p>Simplify the following expression:</p> <math xmlns="http://www.w3.org/1998/Math/MathML"><br><br><br><mfrac> <msqrt><br><br><msup><mi>a</mi><mn>2</mn></msup> </msqrt> <mi>a</mi><br><br></mfrac> </math><br><br><p>Options:</p><br><br>1) <math xmlns="http://www.w3.org/1998/Math/MathML"><msqrt><mi>a</mi></msqrt></math><br>2) <math xmlns="http://www.w3.org/1998/Math/MathML"><mi>a</mi></math><br>3) <math xmlns="http://www.w3.org/1998/Math/MathML"><mfrac><mn>1</mn><mi>a</mi> </mfrac></math><br>4) <math xmlns="http://www.w3.org/1998/Math/MathML"><mn>1</mn></math> Answer: 1<br><br><br>Output: <question> Simplify the following expression: <math xmlns="http://www.w3.org/1998/Math/MathML"><br><br><mfrac> <msqrt><br><br><msup><mi>a</mi><mn>2</mn></msup> </msqrt> <mi>a</mi><br><br></mfrac> </math> </question> <answer> \sqrt{a} </answer> <option_dependent> False </option_dependent> <diagram_dependent> False </diagram_dependent> <language> English </language>|
|---|

###### Table 8: Prompt used for Question Cleaning (Part 2 of 2)

