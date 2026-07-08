# arXiv:2503.04808v1[cs.CL]4Mar2025

## Learning from Failures in Multi-Attempt Reinforcement Learning

#### Stephen Chung∗

DualityRL

Wenyu Du∗ DualityRL

Jie Fu Shanghai AI Lab

### Abstract

Recent advancements in reinforcement learning (RL) for large language models (LLMs), exemplified by DeepSeek R1, have shown that even a simple questionanswering task can substantially improve an LLM’s reasoning capabilities. In this work, we extend this approach by modifying the task into a multi-attempt setting. Instead of generating a single response per question, the model is given multiple attempts, with feedback provided after incorrect responses. The multiattempt task encourages the model to refine its previous attempts and improve search efficiency. Experimental results show that even a small LLM trained on a multi-attempt task achieves significantly higher accuracy when evaluated with more attempts, improving from 45.6% with 1 attempt to 52.5% with 2 attempts on the math benchmark. In contrast, the same LLM trained on a standard single-turn task exhibits only a marginal improvement, increasing from 42.3% to 43.2% when given more attempts during evaluation. The results indicate that, compared to the standard single-turn task, an LLM trained on a multi-attempt task achieves slightly better performance on math benchmarks while also learning to refine its responses more effectively based on user feedback.

### 1 Introduction

Recent advancements in large-scale post-training of reinforcement learning (RL) for large language models (LLMs) have demonstrated a promising approach to enhancing their reasoning capabilities. These improvements have led to emergent abilities such as self-correction and self-refinement [1, 2]. Most existing methods rely on single-turn tasks, where the model receives a reward based on the correctness of its single response to a question. However, single-turn tasks may be inefficient due to sparse rewards, and they do not require the LLM to learn how to respond to user feedback. In this work, we propose a simple yet effective multi-turn task that enables LLMs to learn reasoning through RL.1.

Instead of requiring the LLM to provide a single response to a given question, we propose a multiattempt task that allows the LLM to generate multiple responses based on feedback. Specifically, we first randomly sample N as the number of remaining attempts for each question. The model initially generates a response to a given question as usual. If the response is correct or there are no remaining attempts (i.e., N ≤ 1), the dialogue ends. However, if the response is incorrect and there are remaining attempts (i.e., N > 1), we provide feedback indicating that the answer is incorrect and prompt the LLM to try again, while decrementing the remaining attempts N by 1. An illustration is shown in Figure 2, and an example dialogue from a trained LLM is shown in Figure 3.

Our training pipeline is simple and applies standard RL to the multi-attempt task on a math problem dataset, largely similar to how DeepSeek R1 Zero [2] is trained. In the multi-attempt task, a reward of +1 is given if the answer is correct in any attempt, -0.5 if the answer is incorrect but in the correct

∗Equal contribution. Correspondence to: stephen.chung@dualityrl.com. 1Full code is available at https://github.com/DualityRL/multi-attempt

54

52

EvaluationAccuracy(%)

50

Multi-attempt

48

Baseline

46

44

42

1 2 3 4 5 6 7

Number of Attempts

- Figure 1: Evaluation accuracy as a function of the number of allowed attempts during evaluation, averaged across five benchmarks: AIME 2024, MATH 500, AMC 2023, Minerva Math, and OlympiadBench. Both LLMs are based on Qwen 2.5 Math 1.5B and fine-tuned via RL on a small math dataset in either multi-attempt tasks or single-turn tasks (baseline).

format, and -1 otherwise. We use standard Proximal Policy Optimization (PPO) [3] as the training algorithm.

In addition, the multi-attempt task enables the LLM to learn how to respond to user feedback through RL, rather than merely answering the initial question. In most current LLMs, the ability to respond to user feedback is generally trained via RLHF [4] or supervised fine-tuning (SFT). Compared to RLHF or SFT, learning to respond to user feedback through pure RL may lead to more interesting emergent capabilities, such as significantly improved solution refinement, as demonstrated by our experimental results.

An interesting phenomenon observed in DeepSeek R1 Zero is the emergence of the Aha Moment, which refers to the model’s ability to recognize its mistakes (self-verification) and correct its answers (self-refinement). In DeepSeek R1 Zero, both self-verification and self-refinement capabilities emerge spontaneously within the model. Our multi-attempt approach facilitates the emergence of selfrefinement, as the model has a strong incentive to refine its previous failed attempts (e.g., it does not receive any reward for repeating the same failed attempt). As such, multi-attempt tasks may enable the Aha Moment to emerge more easily, as the model is explicitly trained for self-refinement.

Our experiments show that even a small LLM, such as a 1.5B model, can effectively learn selfrefinement capabilities. As illustrated in Figure 1, the evaluation accuracy of an LLM trained on a multi-attempt task improved from 45.6% to 52.5% on math benchmark when increasing the number of attempts from 1 to 2. In contrast, the same model trained on a single-turn task showed only a marginal gain, from 42.3% to 43.2%. We also observe that even under the standard 1-attempt evaluation, the multi-attempt LLM slightly outperforms its single-turn counterpart, highlighting the benefits of multi-attempt training.

As RL gains popularity in LLMs for enhancing reasoning capabilities, we believe that a multi-turn setting, in general, provides a richer learning signal and better enhances capabilities compared to a single-turn setting. We foresee that more sophisticated capabilities could emerge in LLMs trained in a multi-turn setting via RL.

[Figure 1]

##### Figure 2: Illustration of the multi-attempt question-answer task. We extend the single-turn questionanswer task from DeepSeek R1 to a multi-attempt setting, enabling iterative refinement.

|Example Dialogue:<br><br>System: A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and the answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.<br><br>User: You have 2 attempts to answer the question. Your final answer will be extracted automatically by the boxed tag. This is the problem:<br><br>What is the value of c if x · (3x + 1) < c if and only if x ∈ −73,2 ?<br><br>Assistant: <think> To solve this problem, we need to determine the value of c such that the inequality x·(3x+1) < c holds true if and only if x ∈ −37,2 . Let’s break it down step-by-step:<br><br>1. Understanding the Given Condition: - We need to find c such that x · (3x + 1) < c within the given interval. 2-4. (Omitted for brevity)<br><br>5. Conclusion: - Both 1403 and 14 are needed to be evaluated accurately. Checking both values: 140<br><br>3 ≈ 46.67 - 14 is less than 1403 . Thus, the highest value that fits our condition is c = 1403 . </think> <answer><br><br>|140 3<br><br>|
|---|
<br><br></answer><br><br>User: Your previous answer is wrong. Try alternatives and refine your answer. You have 1 attempt left. Assistant: <think> I apologize for the mistake. Let’s re-evaluate the problem step-by-step after rechecking our previous work.<br><br>1. Understanding the Given Condition: - We need to find c such that x · (3x + 1) < c within the given interval. 2-4. (Omitted for brevity)<br><br>5. Conclusion: - The correct maximum value is actually 14 from x = 2. Thus, the highest value that fits our condition is c = 14. </think> <answer><br><br>|14|
|---|
<br><br></answer>|
|---|

##### Figure 3: An example of a multi-attempt dialogue (N = 2) from a fine-tuned LLM, where the LLM makes a mistake on the first attempt but learns to correct it in the second attempt.

### 2 Approach

In a single-turn task, a question is sampled from a dataset, and the LLM generates a response to the question. Concretely, let the dataset be denoted as D = {(xi,yi∗)}Ni=1 where xi is question i, yi∗ is the ground-truth answer for question xi, and N is the size of the dataset. Denoting the policy of the model as πθ(·|xi), parameterized by θ, we sample a response ai to question xi. The objective is to maximize:

J(θ) = Ex,y∗∼D[R(a,y∗)], (1)

where a ∼ πθ(·|x), and R is the reward function, such as a binary function that gives 1 if the extracted answer from a is the same as the ground-truth answer y∗ and −1 otherwise.

In a more general multi-turn task, we allow the dialogue to continue after the first response. Concretely, we denote xt and at as the prompt and model response at turn t. The initial prompt x0 is randomly sampled from the questions in the dataset D. To generate the subsequent prompt xt, we define the transition function xt = g(x0,a0,x1,a1,...,xt−1,at−1) which determines the next prompt based on previous prompts and responses or whether to terminate the episode. Thus, the objective in multi-turn task becomes:

∞

R(at,y∗) , (2)

J(θ) = Ex

0,y∗∼D

t=0

where at ∼ πθ(·|x0,a0,x1,a1,...,xt−1), that is, the response conditioned on all previous prompts and responses.

In the proposed multi-attempt task, we first determine the number of attempts N = 1,2,... and adopt a specific transition function g, given by:

 

Dialogue termination, if t > N or Extract(at−1) = y∗, Prompt indicating the remaining attempts

g(x0, a0, . . . , xt−1, at−1) =



and requesting another attempt, otherwise.

(3)

In other words, if the model answers correctly or the attempts are exhausted, the dialogue ends; otherwise, the model is prompted for another attempt. An illustration is shown in Figure 2, and a dialogue example from a trained model is shown in Figure 3.

In our experiment, we uniformly sample N from {1,...,M} for each question, where M is the maximum number of allowed attempts (e.g., M = 5). This random sampling encourages the model to learn more diverse behaviors based on the number of available attempts. For example, if a question is sampled with N = 1 (as in the evaluation phase, where we assume the ground-truth answer is unknown), the model should learn to respond carefully and greedily, as it has only one chance. However, when N is large, the model can explore more freely in early attempts, knowing that additional chances remain.

Our reward function R is defined as follows:

- • +1 if the model answers correctly in any attempt,
- • −0.5 if the model provides a response in the correct format but with an incorrect answer,
- • −1 otherwise.

We do not discount rewards based on the number of attempts used, ensuring that the model is encouraged to explore freely in early attempts without incurring a penalty.

Note that the objective in the multi-turn task (2) can be directly optimized via standard RL algorithms such as PPO. As the focus of this work is on the task used to fine-tune LLMs, we refer readers to previous works on finetuning LLM with RL [2] for training details.

### 3 Related Work

Learning to Plan in RL. Our approach is inspired by the Thinker[5] algorithm, a model-based RL method that enables an agent to explore different alternatives within a learned world model

before taking a real action. However, instead of learning a world model, we directly augment the environment to allow multiple attempts.

Multi-turn Tasks for LLMs. A similar approach is SCoRe[6], which also trains an LLM on multi-attempt math tasks via RL. However, their method does not use the ground-truth reward to verify whether the previous attempt was successful. As a result, it requires careful calibration and a two-stage training process to prevent policy collapse, since the LLM has a strong incentive to replicate its first attempt. Other multi-turn studies, such as [7–10], primarily focus on designing the RL algorithms for optimizing multi-turn tasks rather than designing a general RL environment to enhance the reasoning capabilities of LLMs. While these algorithms could be applied in our proposed multi-attempt task, we adopt standard PPO in this work for simplicity.

Self-Correction with External Tools for LLMs. Numerous prior studies have explored selfcorrection mechanisms in LLMs [11]. Reflexion [12] enables LLMs to engage in self-reflection based on external or internal feedback, guiding future trials. CRITIC [13] facilitates interactions with external tools to provide immediate feedback. [14] allows LLMs to execute code for debugging, while FLARE [15] leverages a retriever to estimate the probability of output sentences, enabling fact-checking. In contrast to these works, our work extends the single-turn question-answering task used in DeepSeek R1 [2] into a multi-attempt setting based on a similar RL pipeline to enhance general reasoning capabilities of LLM.

Learning from Historical Logs. Many studies[16–18] explore leveraging past relevant logs to enhance current predictions. For instance, [18] retrieves the past user queries and their predicted SQL from historical logs, providing additional context for generating current SQL. In contrast, our approach utilizes self-generated past attempts as failure examples, encouraging the model to learn from previous mistakes to improve subsequent attempts.

### 4 Experiments

We fine-tune a small pretrained model, namely Qwen 2.5 Math 1.5B, on 8K math questions provided in [19]. We use PPO with a discount rate of γ = 1, lambda λ = 0.99, and a small KL divergence coefficient of 0.01. The LLM is trained for 160 episodes, generating a single sample per question in each episode, totaling 160 × 8K = 1.28M training samples. Our code, modified from [19], is publicly available.

In the multi-attempt experiment, we set M = 5, meaning the number of attempts is uniformly sampled from {1,2,...,5} for each question. In the baseline, we use the standard single-turn setting, where the agent receives a reward of +1 for a correct answer, −0.5 for an incorrect answer in the correct format, and −1 otherwise.

0.8

Multi-attempt

Baseline

0.7

TrainingReward

0.6

0.5

0.4

0.3

0 200 400 600 800 1000 1200

Training Step

(a) Training Reward

48

Multi-attempt

Baseline

EvaluationAccuracy(%)

46

44

42

40

38

36

0 200 400 600 800 1000 1200

Training Step

(b) Average Evaluation Accuracy

- Figure 4: Training and evaluation performance of the LLMs. (a) Training reward as a function of training steps. (b) Average evaluation accuracy across five benchmarks as a function of training steps, evaluated under the standard single-attempt setting.

The reward during training is shown in Figure 4a. As expected, the multi-attempt LLM collects more rewards than the baseline LLM, as the multi-attempt setting is inherently easier to solve. The learning

curve, which represents the average evaluation performance across five benchmarks (AIME 2024, MATH 500, AMC 2023, Minerva Math, and OlympiadBench), is shown in Figure 4b. For a fair comparison, we evaluate both LLMs under the standard single-attempt setting. The results indicate that the multi-attempt LLM slightly outperforms the baseline, suggesting that multi-attempt training may even benefit standard single-attempt evaluations. We hypothesize that this improvement stems from the broader and more efficient exploration facilitated by multi-attempt tasks, leading to more effective reinforcement learning.

Avg. AIME AMC Minerva Olympiad Method Accuracy MATH500 2024 2023 Math Bench

Multi-Attempt 45.4 73.4 20.0 65.0 35.3 33.9 Baseline 43.5 75.4 13.3 55.0 35.3 37.5

Table 1: Comparison of evaluation accuracy across multiple benchmarks.

Another important metric to evaluate is how effectively the LLM refines its search based on previous failed attempts. We assess this by allowing multiple attempts in the math benchmark, with the average results shown in Figure 1 and individual results presented in Figure 5. The results indicate that when trained in a multi-attempt setting, the LLM effectively leverages prior failed attempts, significantly improving performance from 45.58% to 53.82% as the number of attempts increases from 1 to 4. In contrast, the baseline model sees only marginal benefits from multiple attempts, suggesting that its refinement capability is weaker compared to an LLM explicitly trained in a multi-attempt setting. This ability to iteratively refine responses based on user feedback could be particularly valuable in domains requiring adaptive reasoning, such as assisting users in complex code generation.

### 5 Conclusion

In this work, we extend the question-answering task used in DeepSeek R1 by introducing a multiattempt mechanism. Our experiments demonstrate that while the multi-attempt task provides a modest improvement in base performance on math evaluation benchmarks, it significantly enhances the model’s ability to correct mistakes based on user feedback. We envision that further improvement to the task environment—such as incorporating more nuanced and detailed feedback or introducing auxiliary tasks—could foster different capabilities in LLMs and present valuable directions for future exploration.

### Acknowledgment

Stephen Chung completed this work at the University of Cambridge, while Wenyu Du completed this work at the University of Hong Kong.

Multi-attempt

26

Baseline

24

Aime24Accuracy(%)

22

20

18

16

14

1 2 3 4 5 6 7 Number of Attempts

(a) AIME 2024

Multi-attempt

Baseline

82

Math500Accuracy(%)

80

78

76

74

1 2 3 4 5 6 7 Number of Attempts

(b) MATH500

72.5

70.0

Amc23Accuracy(%)

67.5

65.0

Multi-attempt

62.5

Baseline

60.0

57.5

55.0

52.5

1 2 3 4 5 6 7 Number of Attempts

(c) AMC 2023

44

MinervaMathAccuracy(%)

42

40

Multi-attempt

38

Baseline

36

34

32

1 2 3 4 5 6 7 Number of Attempts

(d) Minerva Math

46

OlympiadbenchAccuracy(%)

44

42

40

38

36

Multi-attempt

Baseline

34

1 2 3 4 5 6 7 Number of Attempts

(e) OlympiadBench

Figure 5: Evaluation accuracy as a function of the number of allowed attempts during evaluation on individual benchmarks.

### References

- [1] OpenAI. Learning to reason with llms. 2024. URL https://openai.com/index/ learning-to-reason-with-llms/.
- [2] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.
- [3] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [4] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.
- [5] Stephen Chung, Ivan Anokhin, and David Krueger. Thinker: Learning to plan and act. Advances in Neural Information Processing Systems, 36:22896–22933, 2023.
- [6] Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, et al. Training language models to self-correct via reinforcement learning. arXiv preprint arXiv:2409.12917, 2024.
- [7] Yifei Zhou, Andrea Zanette, Jiayi Pan, Sergey Levine, and Aviral Kumar. Archer: Training language model agents via hierarchical multi-turn rl. arXiv preprint arXiv:2402.19446, 2024.
- [8] Lior Shani, Aviv Rosenberg, Asaf Cassel, Oran Lang, Daniele Calandriello, Avital Zipori, Hila Noga, Orgad Keller, Bilal Piot, Idan Szpektor, et al. Multi-turn reinforcement learning from preference human feedback. arXiv preprint arXiv:2405.14655, 2024.
- [9] Wentao Shi, Mengqi Yuan, Junkang Wu, Qifan Wang, and Fuli Feng. Direct multi-turn preference optimization for language agents. arXiv preprint arXiv:2406.14868, 2024.
- [10] Arnav Kumar Jain, Gonzalo Gonzalez-Pumariega, Wayne Chen, Alexander M Rush, Wenting Zhao, and Sanjiban Choudhury. Multi-turn code generation through single-step rewards, 2025. URL https://arxiv.org/abs/2502.20380.
- [11] Ryo Kamoi, Yusen Zhang, Nan Zhang, Jiawei Han, and Rui Zhang. When can llms actually correct their own mistakes? a critical survey of self-correction of llms. Transactions of the Association for Computational Linguistics, 12:1417–1440, 2024.
- [12] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652, 2023.
- [13] Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. Critic: Large language models can self-correct with tool-interactive critiquing. arXiv preprint arXiv:2305.11738, 2023.
- [14] Xinyun Chen, Maxwell Lin, Nathanael Schärli, and Denny Zhou. Teaching large language models to self-debug. In The Twelfth International Conference on Learning Representations, 2024.
- [15] Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7969–7992, 2023.
- [16] Ondˇrej Dušek and Filip Jurˇcíˇcek. A context-aware natural language generator for dialogue systems. In Raquel Fernandez, Wolfgang Minker, Giuseppe Carenini, Ryuichiro Higashinaka, Ron Artstein, and Alesia Gainer, editors, Proceedings of the 17th Annual Meeting of the Special Interest Group on Discourse and Dialogue, pages 185–190, Los Angeles, September

2016. Association for Computational Linguistics. doi: 10.18653/v1/W16-3622. URL https: //aclanthology.org/W16-3622/.

- [17] Charlie Snell, Mengjiao Yang, Justin Fu, Yi Su, and Sergey Levine. Context-aware language modeling for goal-oriented dialogue systems, 2022. URL https://arxiv.org/abs/2204. 10198.
- [18] Yanzhao Zheng, Haibin Wang, Baohua Dong, Xingjun Wang, and Changshan Li. Hie-sql: History information enhanced network for context-dependent text-to-sql semantic parsing, 2022. URL https://arxiv.org/abs/2203.07376.
- [19] Weihao Zeng, Yuzhen Huang, Wei Liu, Keqing He, Qian Liu, Zejun Ma, and Junxian He. 7b model and 8k examples: Emerging reasoning with reinforcement learning is both effective and efficient. https://hkust-nlp.notion.site/simplerl-reason, 2025. Notion Blog.

