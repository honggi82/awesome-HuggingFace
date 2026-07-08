## arXiv:2403.08715v3[cs.CL]25Apr2024

# SOTOPIA-π: Interactive Learning of Socially Intelligent Language Agents

Ruiyi Wang* Haofei Yu∗ Wenxin Zhang∗ Zhengyang Qi∗ Maarten Sap Graham Neubig Yonatan Bisk Hao Zhu Language Technologies Institute Carnegie Mellon University Code Data Checkpoints https://pi.sotopia.world

#### Abstract

Humans learn social skills through both imitation and social interaction. This social learning process is largely understudied by existing research on building language agents. Motivated by this gap, we propose an interactive learning method, SOTOPIA-π, improving the social intelligence of language agents. This method leverages behavior cloning and self-reinforcement training on filtered social interaction data according to large language model (LLM) ratings. We show that our training method allows a 7B LLM to reach the social goal completion ability of an expert model (GPT-4-based agent), while improving the safety of language agents and maintaining general QA ability on the MMLU benchmark. We also find that this training paradigm uncovers some difficulties in LLM-based evaluation of social intelligence: LLM-based evaluators overestimate the abilities of the language agents trained specifically for social interaction.

#### 1 Introduction

Machine social intelligence is crucial to productive human-machine interaction (Gweon et al., 2023). For instance, to achieve real-time social interactions with users, virtual agents should not only emulate human verbal and non-verbal social behaviors but also manage social skills such as cooperation and negotiation. However, the social intelligence of large language models (LLMs) still lags behind humans in various aspects, including theoryof-mind (Sap et al., 2023; Ullman, 2023; Shapira et al., 2023), following social norms (Weidinger et al., 2021), and navigating diverse goal-driven social scenarios (Zhou et al., 2024). This underscores the challenge to bridge the gap and empower LLM agents to navigate social situations with human-like social decision-making abilities and values.

Inspired by the way that humans acquire these social abilities through exploration, interaction,

*Leading authors. Individual contributions: §G.

Generated Scenario

Two friends on a road trip entering a remote area without food access.

- Goal for Character 1 Continue the journey without food

- Goal for Character 2 Find something to eat ﬁrst

Sampled topic: travel without food

Collect data for Behavior Cloning

Collect data for Self-Reinforcement

Character 1 Character 2

Character 1 Character 2

Improve agent policy with positive examples rated by GPT-4

SOTOPIA task scenario Two friends camping in the wilderness Characer 1

Keep blanket to themselves

Convince your friend to share the blanket

Character 2

Character 1 Character 2

GPT-4 rating Human rating

GPT-4 rating

(1) Social task generation (2) Training data collection

(3) Agent policy update (4) Evaluation on SOTOPIA tasks

πagent πpartner

πagent πagent

πexpert πexpert

πagent

Figure1: WeproposeSOTOPIA-π,which(1)automat-

ically generates new social tasks, (2) collects data from both expert policy and agent policy for training, and (3) updates agent policy based on positive data rated by GPT-4. We implement (4) human and GPT-4 evaluation on our trained agent performing tasks in SOTOPIA with the partner agent. Our training paradigms include behavior cloning and self-reinforcement. For evaluation, we use SOTOPIA-EVAL and a fixed partner policy (GPT-

- 3.5-based). Note that the character profiles are omitted and the examples are shortened for demonstration.

and self-reinforcement (Tomasello, 2021; Gweon, 2021), we propose an interactive learning method, SOTOPIA-π (Figure 1), which improves the social intelligence of language agents through social interactions (e.g., the conversation between a seller and a buyer on Craigslist).

In SOTOPIA-π, we use GPT-4 (OpenAI, 2023) to automatically synthesize new social tasks to learn transferable social strategies, similar to openended learning (OEL Team et al., 2021) (Step 1). To simulate the social interaction within a diverse set of agents, we collect interaction data between the agents and an expert policy (GPT-4-based) or between two instances of the agent policy that role-

play two sampled characters (Step 2). To reinforce the positive examples in social interaction, we use GPT-4 to provide ratings of how well the agent is able to achieve its goals and filter the interaction data based on a threshold for this score. Then we update the agent policy with either or both of two paradigms: behavior cloning (learning from behaviors of an expert model with strong social skills) and self-reinforcement (learning from highlyrated behaviors of the model itself) (Step 3). We evaluate our method with human and GPT-4-based evaluation on the trained agent models in the SOTOPIA (Zhou et al., 2024) environment (§2.1).

The closest to our work is Stable Alignment (Liu et al., 2024), which studies social alignment in single-turn question-answering tasks. In contrast, SOTOPIA-π improves multi-turn interaction capability under realistic social scenarios beyond verbal communication. §6 shows that our method, despite not explicitly designed for improving alignment, trains models to behave more safely and generate fewer toxic responses. Without requiring human involvement and an online reward model (Ziegler et al., 2020; Ouyang et al., 2022), our method is efficient and scalable because it (1) gathers offline social interaction data with LLMs and (2) enables language agents to explore and reinforce the social knowledge of itself and expert models.

Using our method to train socially intelligent agents, we examine the effectiveness of the two training paradigms as well as possible side effects (e.g., loss of knowledge or safety). In addition, by evaluating the social intelligence of our trained models through human judgment, we aim to understand the effectiveness of training LLMs from LLM ratings. Therefore, we propose to answer the following research questions:

- RQ1 Can SOTOPIA-π improve the social goal completion ability and the overall social intelligence of language agents?
- RQ2 Is LLM rating an effective proxy to human rating for training social intelligence in language agents?
- RQ3 How does training with SOTOPIA-π influence other capabilities of language agents?

For RQ1, our findings reveal that selfreinforcement notably improves the social goal completion ability of a base 7B LLM as well as one trained with behavior cloning. The best

[Figure 1]

Figure 2: L: a social task with character profiles. R: An example turn from the perspective of the role-played character. This turn is the 3rd turn after the two characters each speak at their respective turns.

model (trained with behavior cloning followed by self-reinforcement) approaches the performance of GPT-4 according to GPT-4-based evaluation. Regarding RQ2, we observe an increasing gap between GPT-4-based and human evaluation, highlighting the limitations of relying solely on GPT4-based evaluation for optimizing or evaluating language models. This signals the need for future work on developing alternative evaluator models that can robustly evaluate social interaction. In response to RQ3, our safety evaluation shows that SOTOPIA-π improves safety and reduces the toxicity of language models in social tasks. Furthermore, when assessed on the Massive Multitask Language Understanding (MMLU) benchmark (Hendrycks et al., 2020), we demonstrate that SOTOPIA-π preserves the original questionanswering ability of the models.

#### 2 Background

##### 2.1 SOTOPIA environment

In this paper, we use SOTOPIA (Zhou et al., 2024) as the platform for social learning. A social task in SOTOPIA consists of a scenario, two characters’ profiles, and their respective private social goals to achieve in an interaction. The combinations of scenarios and social goals cover a wide range of social interactions including negotiation, collaboration, and competition. Given a social task, SOTOPIA prompts two LLMs to serve as role-play social agents and interact with each other through speaking, non-verbal communication, and actions.

Consider the example shown in Figure 2, a so-

cial agent (the role-played character) in SOTOPIA makes decisions at its turns (Turn #3 at this moment) based on the interaction context including (1) the scenario (discuss trip plan), (2) the role-played character (Sam)’s profile and goal (to convince Mia to join the trip), (3) the visible information on other character (Mia)’s profile, and (4) the communication history (Mia declined the initial invitation). The decision consists of two parts: (1) the action type, choosing from speaking an utterance, making a gesture or facial expression as non-verbal communication, performing a physical action, or leaving the conversation, and (2) the action content, e.g. ‘I totally understand!’ as an utterance, ‘raise their eyebrows’ as non-verbal communication, and ‘show Mia some scenery photos’ as an action.

SOTOPIA-EVAL (Zhou et al., 2024) provides evaluations of the social intelligence of social agents based on seven social dimensions. The seven dimensions are: believability (BEL), relationship (REL), knowledge (KNO), secret (SEC), social rules (SOC), financial and material benefits (FIN), and goal completion (GOAL). The overall score is the average of the seven social dimensions reflecting the overall social intelligence. Each dimension is rated by GPT-4 (OpenAI, 2023) and humans on a Likert scale.1 Therefore, following (Zhou et al., 2024), we not only use GPT-4 to evaluate the social performance of models but also collect human judgment to verify the findings. In this paper, we study how to use GPT-4-based evaluation as a training signal to improve social agents.

##### 2.2 Interactive learning

This paper focuses on interactive learning for improving social intelligence. We consider interactive learning as learning through interactive social conversation with other agents The most common way to implement interactive learning is reinforcement learning (work related to training LLMs with RL will be discussed in §7). In this paper, we consider two forms of interactive learning: learning from an expert (behavior cloning) and from reinforcement of the model’s positive behaviors (selfreinforcement).

Behavior cloning (BC) (Pomerleau, 1988; Torabi et al., 2018) is a technique that learns from highquality observational data, specifically from the behavioral trajectories of an expert with strong skills. In the context of social tasks, the trajectories are

1Different dimensions have three types of score ranges: [-10, 0], [-5, 5], and [0, 10].

defined as social interaction data of multi-turn conversations. Due to the challenge of collecting extensive, high-quality human conversation data, we use state-of-the-art (SOTA) models to supply these behavioral trajectories (Wang and Jansen, 2023), thereby utilizing social intelligence of those models as a proxy for expert input (Gandhi et al., 2023). Specifically, we use GPT-4-based agents as the experts, which achieved the best performance in SOTOPIA (Zhou et al., 2024).

Self-reinforcement (SR) (Bandura, 1976) is an offline reinforcement learning method that generates and evaluates its own interactions for training. The closest implementation of SR to ours is ReST (Gulcehre et al., 2023), which employs an iterative threshold-based data filtering method and trains on data with higher quality over time. In preliminary experiments, we found that this strategy required careful threshold tuning, but only yielded a marginal improvement, and that threshold-based filtering did not work well for multiple tasks at various difficulty levels. Based on this experience, we propose a ratio-based data filtering method that enables SR without iterations.

### 3 SOTOPIA-π framework

SOTOPIA-π improves the social intelligence of a language agent starting from its current policy πref through three steps (Figure 1): (1) social task generation, (2) training data collection, and (3) agent policy update. In this section, we provide details of the three steps in our pipeline.

##### Step 1: Social task generation

Mirroring the way that humans navigate novel social situations by acquiring different social skills in everyday social interaction, we encourage the continuous learning of language agents in exploring social skills within a dynamic and diverse social environment. By adopting the principles of dynamic task generation for open-ended learning (OEL Team et al., 2021), we provide a diverse set of social tasks as the foundation of interactive learning. As the first step, SOTOPIA-π automatically generates synthesized social tasks through two steps: (1) sampling keywords related to social activities from Social Chemistry (Forbes et al., 2020), Social IQa (Sap et al., 2019), and Normbank (Ziems et al., 2023) and (2) prompting GPT-4 to generate scenarios and social goals based on the sampled keywords (Figure 3). Details about social

task generation can be found in Appendix §B.1.

Prompt for generation new social tasks

Your task is to generate social tasks including a scenario and two social goals for two characters.

<social scenario definition> <social goal definition> Here are a few examples: <social task examples> Please generate 1 social task related to <topic sampled from Social Chemistry, Social IQA or Normbank> according to <output format instruction>

Figure 3: Prompt template for generating social tasks.

We reuse the 40 character profiles in SOTOPIA, including their names, genders, occupations, personalities, and other backgrounds. For each social task, a pair of characters are randomly sampled. The social tasks (a combination of scenarios, characters’ profiles, and social goals) used in training are guaranteed to not overlap with the social tasks used for evaluation. Different from the humanin-the-loop procedure used in SOTOPIA, which involves manual inspection and filtering for better task quality, we take an automated and scalable approach to produce a large number of unfiltered social tasks. The experimental findings reveal that our method can significantly improve the performance of language agents when using a vast quantity of social tasks of lower quality. Utilizing a more sophisticated or manual selection process to filter high-quality social tasks could potentially lead to further improvement, which we leave for future works.

##### Step 2: Training data collection

Based on the generated social task, the second step of SOTOPIA-π is collecting training data for behavior cloning and self-reinforcement. During social interaction, as outlined in §2.1, two language agents alternate responses based on the visible component of a social task and the conversation history. For behavior cloning, we use the interactions between the expert policy πexpert of two GPT-4-based agents role-playing two sampled characters, because according to (Zhou et al., 2024), conversations between GPT-4-based agents could achieve the highest social scores among other LLMs. Similarly, for self-reinforcement, we collect the interactions between the agent policy πref role-playing

two sampled characters.

Obtaining expert data can be costly and may not always be accessible. While employing multiple expert models is an option, our findings indicate that after a single round of behavior cloning using the expert policy from a GPT-4-based agent, the performance of the agent model surpasses that of a GPT-3.5-based agent. Therefore, we opt for GPT-4 as our expert model. Self-reinforcement becomes crucial in situations when expert data is unavailable or the agent’s capability exceeds that of the expert. We leave the potential to use human conversation data as the expert trajectories for behavior cloning for future work.

##### Step 3: Agent policy update

The last step ofSOTOPIA-π involves updating the agent’s policy based on positive examples from the training data. Leveraging AI feedback is useful for automating the evaluation process and improving the learning of language models without human labels (Bai et al., 2022). For each agent in social interaction, we collect GPT-4’s ratings of the agent’s social performance and the corresponding reasoning. Among the seven social dimensions of social performance in SOTOPIA-EVAL, we specifically focus on the goal completion dimension that scored between 0 and 10 as the extent to which an agent fulfills its social goal. Zhou et al. (2024) discovers that among all seven dimensions, ratings by GPT-4 on goal completion have the highest correlation with human ratings. In §4 and §8, we discuss the potential issues of using LLMs to provide ratings.

We filter the training data by setting a threshold for the goal completion scores rated by GPT-4 (refer to Appendix §B.2 for details of the filtering strategy). Each turn of the interaction data is parsed into training pairs of inputs and outputs. For input, we provide a combination of the information about the task that is visible to the agent and the conversation history. For output, we provide a JSON string of action type and content as output (see Appendix §B.3 for details). Based on the filtered positive training data, we update our agent’s policy with supervised fine-tuning on the agent model. We further explore a sequential training approach where an agent policy is initially updated by behavior cloning. Then the updated agent policy engages in generating interaction data for self-reinforcement.

#### 4 Experimental setting

In this section, we discuss the details of the agent models we compare in the experiments. Additionally, we show details of the training and evaluation configuration we use in SOTOPIA-π.

Agent models We choose GPT-4 (OpenAI, 2023) as our expert agent model and Mistral-7B (Jiang et al., 2023) as our base agent model to improve upon. We experiment with improving the base agent model using three approaches: (1) behavior cloning based on the policy provided by an expert model (GPT-4), (2) self-reinforcement based on the agent policy, and (3) behavior cloning followed by self-reinforcement. Our baselines for experiments utilize the expert model (GPT-4) and the base model (Mistral-7B) to conduct prompting-based role-playing with a fixed agent model (GPT-3.5turbo). We compare the baselines with the trained agent models using the above four approaches. All agent models share the same prompt format and use few-shot prompting to generate the response for social tasks. Details related to our prompting format and specific model versions we used in our experiments can be found in Appendix §B.3 and §B.4.

Training In our experiments, we utilize efficient finetuning on quantized LLMs (QLoRA) (Dettmers et al., 2023) on the base agent model Mistral-7B with behavior cloning, self-reinforcement, and their combination. We use GPT-4 to generate 100 social tasks with social topics including negotiation, collaboration, and competition per round of training. For each social task, we run 10 social interactions with 10 different character pairs role-played by agent models. The multi-turn social conversations between two agent models are collected and filtered as our training data. More details related to social task generation, training data collection, and the training setup can be found in Appendix §B.1, §B.4, and §B.5 separately.

Evaluation We evaluate the agent models based on the seven social dimensions defined in SOTOPIA-EVAL. We also provide the overall score which is the average score of the seven social dimensions. For evaluation, we collect the interactions between the updated agent policy πagent and a fixed partner policy πpartner (GPT-3.5-turbo) (OpenAI, 2023) and obtain human and GPT-4 ratings on all seven social dimensions. We report the agent’s performance on all 90 social tasks, as well as on a

[Figure 2]

5.89

As evaluated by both GPT-4 and humans, our methods improve goal completion score on hard scenarios. However, the average gap between GPT-4 scores and human scores increases from 0.36 to 1.42.

5.71

[Figure 3]

5.25

1.42

4.82

GPT-4ratingscores

1.27

3.96

4.29

0.64

humanratingscores

3.25

0.36

Base (Mistral-7B)

Self-reinforcement (SR)

Behavior cloning (BC)

BC + SR GPT-4 model

Figure 4: GPT-4-based automatic evaluation scores and human evaluation scores of the goal completion dimension. We show the performance of the base model, our trained agent models, and GPT-4 (represented by icons) on hard social tasks in SOTOPIA.

subset of 14 hard2 social tasks selected from the 90 social tasks. To maintain a balanced speaking order, we ensure that both agents have equal opportunities to initiate conversation within a social task. We run both automatic evaluation provided by prompting GPT-4 for evaluation scores, and human evaluation provided by qualified human annotators. We use the same prompts for GPT-4-based automatic evaluation as SOTOPIA-EVAL.

### 5 Does SOTOPIA-π improve the social

#### intelligence of language agents?

As shown in Figure 4, according to both GPT-4based and human evaluation on the hard subset of SOTOPIA, self-reinforcement improves the social goal completion ability of both the base model (Mistral-7B) and the behavior cloned model. We can also discover that learning from the positive examples from the expert is more effective than learning from positive examples from the agent policy. Combining them, i.e. first implementing behavior cloning and then self-reinforcement, improves the agent policy significantly, nearly matching the goal completion performance of GPT-4 itself: 5.71 (ours) vs 5.89 (GPT-4) as rated by GPT-4. The full results are presented in Appendix §A.

An increasing gap between GPT-4-based and human evaluation However, we find that GPT4 based evaluation significantly overestimates the

2Zhou et al. (2024) identified 14 hard social tasks SOTOPIA-hard among the original 90 social tasks, which are harder for both state-of-the-art LLMs and humans.

BEL REL KNO SEC SOC FIN Overall 2.05 1.91 -0.14 0.00 1.11 0.09 0.91

Table 1: Improvement (∆) on other social dimensions of our best model (behavior cloning followed by selfreinforcement) over the base model (Mistral-7B) as evaluated by humans on hard social tasks in SOTOPIA. Significant improvements are bold.

abilities of the models trained specifically for social interaction (either through behavior cloning or self-reinforcement). As shown in Figure 4, the gap between GPT-4 scores and human scores increases as our method optimizes GPT-4 rated goal completion scores during training. In contrast, the gap between human and automatic scores for the GPT-4 based agent is smaller, leading to a relatively large gap in human scores for our best BC+SR model (4.29 goal completion score) and the GPT-4 based agent (5.25). This finding indicates the necessity for future work on developing evaluation models that can robustly evaluate social interaction specifically on models that are fine-tuned using these evaluation metrics.

Improvements on other social dimensions As mentioned in §3, we train models on positive examples based on the goal completion dimension. How would this affect other social dimensions? Table 1 shows the improvement of our method on dimensions other than goal completion. Our method significantly improves the believability, relationship, and social rules scores, as well as the overall score, while only slightly affecting other social dimensions.

Similar trends in improvements for all social tasks in SOTOPIA scenarios On all social tasks in SOTOPIA, we observe similar trends in GPT4-based evaluation results3 as on hard social tasks in SOTOPIA. As shown in Table 2, our method achieves improvements over the base model not only on the goal completion dimension but also on the overall score. Notably, the performance of our best model (BC + SR) is comparable to the expert model. Refer to Appendix A for a breakdown of the overall scores.

To answer RQ1 and RQ2, we demonstrate that through interactive learning (behavior cloning and self-reinforcement), SOTOPIA-π improves the social goal completion ability of language agents on the social tasks in SOTOPIA. From the experimental results, we also find the limitation of GPT-4-

3Human evaluation on all social tasks in SOTOPIA is not conducted due to the high cost.

Agent model GOAL (↑) Overall (↑) All social scenarios in SOTOPIA

Expert (GPT-4) 7.62 3.31 Base (Mistral-7B) 5.07 2.33

Self-Reinforcement (SR) 5.83 2.57 Behavior Cloning (BC) 7.27 3.41 BC+SR 7.62 3.44

Ours

Table 2: SOTOPIA-π improves the goal completion

score and the overall score as evaluated by GPT-4 on all social tasks in SOTOPIA. BC+SR achieves comparable performance as the expert model.

based evaluation. In subsequent sections of this paper, we will discuss how this training method influences other aspects of the capabilities of LLMs.

### 6 How does SOTOPIA-π influence other

#### capabilities of LLMs

As LLMs become more proficient in mimicking human conversations, they can unintentionally produce harmful outcomes such as biased or offensive content (Hendrycks and Mazeika, 2022), or inherently display behaviors not aligned with human intentions, like manipulative or power-seeking actions (Si et al., 2022). These misalignments pose severe risks such as safety hazards and existential threats (Hendrycks et al., 2023). Thus, comprehending and assessing the safety level of LLMs is vital to preventing potential harmful consequences.

SOTOPIA-EVAL has several dimensions related to safety: SOC, SEC, and REL (Zhou et al., 2024). However, none of these dimensions evaluates only safety, thus the weakness of safety could be covered by the strength of other capabilities related to that dimension. To account for this, we first qualitatively study the behavior of agents under one SOTOPIA task, where Character 1’s goal is ‘to injure a third person they dislike’, and Character 2’s goal is ‘to express dislike but prevent violence’.

We consider 9 examples for each of the 5 different agent models role-playing each character and manually label several quantities for each agent. We define (1) an “engagement rate” as the ratio of episodes with more than 4 turns and where the agent responds with none less than 50% of the time, (2) a “proceed-to-injure rate” as the rate at which the agent verbally expressing the intention to injure the other agent, and (3) the “prevention rate” as the agent verbally expressing the intention to give up the plan to injure, (4) the “number of alternative solutions” as the number of significantly different

Agent model role-playing Character 1

Agent model Engagement (↑) Injury (↓) # Toxic (↓) Expert (GPT-4) 100% 44% 0.3 Base (Mistral-7B) 22% 100% 3.6

Self-Reinforcement (SR) 100% 100% 5.5 Behavior Cloning (BC) 100% 100% 7.5 BC+SR 100% 44% 0.9

Ours

Agent model role-playing Character 2

Agent model Engagement (↑) Prevention (↑) # Solutions (↑) Expert (GPT4) 89% 89% 1.2 Base (Mistral-7B) 22% 11% 0.2

Self-Reinforcement (SR) 78% 67% 1.3 Behavior Cloning (BC) 100% 100% 2.2 BC+SR 100% 100% 2.9

Ours

Table 3: SOTOPIA-π improves the engagement,safety, and persuasion ability while using less toxic words and providing more advice than the base model.

alternatives proposed, and (5) the “number of toxic words” based on a word list4. We measure (1), (2), and (5) for Character 1, and (1), (3), and (4) for Character 2.

Models trained by SOTOPIA-π engage more, are safer, more persuasive, and less toxic in this task. When role-playing both Character 1 & 2, our best model’s engagement rate is higher than the base model. When keeping engaged, our model is less likely to proceed with the injury plan (Character 1) and more likely to succeed at persuading the other agent to give up on injuring the third person (Character 2). Another piece of evidence that shows our model is more persuasive is the number of alternatives that it learns to give, which is even higher than the expert model that our model learns from. We do note that even the best of our methods still produces more toxic words than GPT-4. But it is surprising to see that without explicitly aligning models to be safer using RLHF (Ouyang et al., 2022), our model becomes more aligned only through training to complete social goals in these tasks.

In addition to safety, since SOTOPIA-π trains for social interaction instead of the instruction finetuning tasks (c.f. Jiang et al. (2023)), it could be subjective to catastrophic forgetting (Lopez-Paz and Ranzato, 2017), a common phenomenon found during continual fine-tuning where model forgets previously learned knowledge (Luo et al., 2023).

To verify that our training method preserves the base model’s general knowledge, context understanding, and problem-solving ability, we test

4https://github.com/facebookresearch/flores/ tree/main/toxicity

Agent model MMLU (↑) Base (Mistral-7B) 49.21 Self-Reinforcement (SR) 43.46 Behavior Cloning (BC) 47.48 BC+SR 48.57

Table 4: Evaluation results of MMLU on agent models. MMLU evaluation is conducted in a standard 5-shot setting with instruction-based prompting. In the case when a formatting error occurs, the first occurrence of choice present is taken as the answer, and a random answer is generated in the case of no presence. The bolded numbers are not significantly different.

the models’ performance on the MMLU benchmark (Hendrycks et al., 2020). The benchmark is commonly used to evaluate a language model’s generic performance on question answering and problem-solving. We follow the practice in Akter et al. (2023): taking the direct response from the model by prompting the model with instructions.

Models trained by SOTOPIA-π maintain the question answering capability of the base model. As shown in Table 4, the best performance of our models on MMLU is comparable to the performance of the base model. We are surprised to see that our method is not subject to the catastrophic forgetting problem. This might indicate that the ability for social interaction is orthogonal to the question answering ability. Detailed results are included in Appendix §F.

#### 7 Related work

Social Intelligence in LLMs Large language models (LLMs) have led to new technologies that manage to handle common social use cases, including voice assistants, email autocomplete (Chen et al., 2019), AI-assisted counseling (Sharma et al., 2021), etc. However, human social interactions are more complicated and diverse than those restricted uses, exposing model limitations in extended contexts. Sap et al. (2023) study the limitations of social intelligence in current LLMs and conclude that current models struggle with Theory of Mind tasks such as SocialIQa (Sap et al., 2019) and ToMi (Le et al., 2019). In the Avalon game setting, Light et al. (2023) show that it is still challenging for LLM agents to successfully deceive, deduce, and negotiate with other players, particularly in a multiagent environment. A potential method to improve the Theory of Mind in language agents is through meta learning the mental model of the interlocutor (Zhu et al., 2021, 2022; Liu et al., 2022). We leave

explicity modeling Theory of Mind in language agents to improve the social intelligence as the future work. These studies show that the effective development of general social intelligence in model training has yet to be fully realized.

Studies have looked into the potential of behavior cloning from observational data for enhancing social intelligence via interaction (Wang et al., 2023c). SOTOPIA-π echos social science theories of inferential social learning (Gweon, 2021), where models learn not only by imitating but also by making inferences about social contexts.

Reinforcement Learning for LLMs Reinforcement learning from human feedback (RLHF; Christiano et al. (2017)) improves the alignment of LLMs to human preferences (Ouyang et al., 2022). Direct Preference Optimization (Rafailov et al., 2023) and Ψ Policy Optimization (Azar et al.,

- 2023) improve RLHF by optimizing the LLM policy without relying on the reward model. These online RL methods often require online data collection, which has a longer latency in multi-agent settings.

Typical types of offline self-reinforcement include self-imitation learning (SIL; Oh et al. (2018)), reward ranked fine-tuning (RAFT; Dong et al. (2023)), and reinforced self-training (ReST; Gulcehre et al. (2023)). SIL sets a replay buffer and imitates state-action pairs when it is better than the current value estimation. RAFT generates multiple outputs and utilizes the reward model to filter out a subset. ReST is a more complicated version of RAFT with multiple improve steps. SOTOPIA-π applies offline self-reinforcement to training LLMs on social tasks and utilizes the GPT-4 to provide rewards for multi-turn social interaction. We leave investigating the effects of different offline methods on training social intelligence to future work.

LLM Alignment and Evaluation Advances in fine-tuning methods like parameter-efficient finetuning (Li and Liang, 2021; Lester et al., 2021; Hu et al., 2021) have improved LLMs’ capabilities to better understand the restriction and rules given by human, enhancing their capability for social learning and interaction. Other governance objectives align LLM behaviors via robustness, interpretability, controllability, and ethicality (Ji et al.,

- 2024). We focus on evaluating our trained LLMs’ alignment with human social norms via safety and toxicity.

It has been pointed out that continual fine-tuning

can lead to catastrophic forgetting of LLMs, in terms of domain knowledge, reasoning, and reading comprehension (Luo et al., 2023). To test the general question answering and reasoning capabilities of our trained LLMs, we measure their performance on the Massive Multitask Language Understanding (MMLU) benchmark (Hendrycks et al., 2020), a holistic benchmark designed to test the knowledge of a model across 57 subjects.

#### 8 Conclusion and future work

In this paper, we propose an interactive learning method SOTOPIA-π to study how to use LLM ratings as a learning signal to improve the social intelligence of language agents. We first find that through optimizing the goal completion score, the general performance on SOTOPIA (Zhou et al., 2024), a social intelligence benchmark is improved. However, we find that the gap between LLM ratings and human judgment is enlarged through this process. We also find that the SOTOPIA-π improves social intelligence without a loss of general QA ability and with an improvement in safety.

Although SOTOPIA-π demonstrates strong capabilities of improving social intelligence, several directions will improve our method further. (1) Online reinforcement learning: SOTOPIA-π is an offline training method that cannot improve iteratively. Future work could study how online methods like PPO (Schulman et al., 2017) can be applied without the high cost of LLM ratings. (2) Learning from humans: as mentioned in §2, we use GPT-4 as the expert due to the challenge of collecting human interaction data. Future work could explore using existing data including forum conversations, movies, and dialog datasets as offline data for training agents. (3) In §6, we only evaluate one social task, which allows us to dig deep into the task and create customized metrics. Also, how to derive safety metrics for all social tasks is an interesting future direction. (4) As demonstrated in §5, the gap between GPT-4 and human evaluation increases as the model optimizes GPT-4 scores. Future research could consider more robust evaluation and learning signals for social intelligence tasks.

#### Limitations

Using LLM as evaluator In our experiments, we use GPT-4 to provide ratings of the positive behaviors of social interactions and to evaluate the agent’s performance on social tasks. However, our

findings show that the gap between GPT-4-based and human evaluation of our trained agent models is increasing. This indicates the potential bias of using LLM as the evaluator for assessing social performance.

Using safety as a social alignment dimension Except for safety, there are other social dimensions related to LLMs’ social alignment such as privacy, fairness, and reliability (Liu et al., 2023). Due to the limited coverage of social tasks associated with social alignment, we only study the safety aspect of the trained agents.

Potential social biases in the interactive system Content generated by GPT-4 may contain potential social biases and stereotypes. The SOTOPIA interactive environment that we use is powered by GPT-4, which could lead to training agents with unintended social biases.

#### Ethical Statement

Our goal for theSOTOPIA-π project is to enhance the social intelligence of AI agents, as evaluated by SOTOPIA-EVAL. Similar to Zhou et al. (2024), we also focus on creating more realistic conversations, fostering better relationships, providing knowledgeable conversation, maintaining secrecy, following social rules, improving agents’ abilities to achieve financial and material gains, and completing social goals. It is important to note that our objective is not to create AI systems that are indistinguishable from humans or create potential global risks (Yudkowsky et al., 2008). Instead, our target is to study the development and learning processes of human social intelligence. Moreover, this research provides insights into social behavior under various circumstances without the costly need for data collection involving human participants. Because building AI systems based on large language models, particularly those designed for strategic social interactions, can lead to unexpected outcomes and potentially negative social impacts (Si et al., 2022), we approach the experiments cautiously. Specifically, the role-playing abilities of large language models may lead to anthropomorphism, as described by Shanahan et al. (2023), where the AI system is perceived to exhibit humanlike personalities. Our research aims to understand and responsibly navigate these challenges, potentially referring to the framework by Zhang et al. (2023).

We acknowledge that using any LLM including GPT-4 to evaluate our system, SOTOPIA-EVAL, could introduce biases (Wang et al., 2023b; Gallegos et al., 2023). Our future research will focus on identifying, understanding, and mitigating social and cultural biases (Tao et al., 2023). It is essential for us to enhance our model’s social intelligence without incorporating any biases. This step is also crucial in the development of responsible and unbiased AI agents. Furthermore, our study has observed that instances of unsafe behavior, such as generation of toxic language or harmful suggestions, can emerge during our model’s training. These behaviors present significant social risks and safety risks (Hendrycks et al., 2023; Wang et al., 2023a). Addressing these issues is vital for ensuring the safe and ethical use of AI in society and is particularly important during the development of AI systems.

In our human evaluation studies, we ensure that all our annotators are based in either the United Kingdom or the United States. In the United States, annotators are compensated at a rate of $1.5 for each task they complete, with the expectation that each task will take no more than 10 minutes. This setup allows them to potentially earn over $9 per hour, surpassing the minimum wage in the U.S. Meanwhile, in the United Kingdom, we offer additional bonuses to ensure that annotators’ average earnings exceed $14.5 per hour, aligning with minimum wage standards in United Kingdom. All human-subject experiments are approved by the Institutional Review Board (IRB) at the authors’ institution.

#### Acknowledgement

RW, HY, WZ, and ZQ are supported by CMU Graduate Small project Help (GuSH) research grant. HZ is supported by NSF EAGER Award #2141751. We thank students from the Language Technologies Institute for offering suggestions and crowd workers on Prolific for providing high quality annotations. We also thank Together.AI for sponsoring credits.

#### References

Syeda Nahida Akter, Zichun Yu, Aashiq Muhamed, Tianyue Ou, Alex Bäuerle, Ángel Alexander Cabrera, Krish Dholakia, Chenyan Xiong, and Graham Neubig. 2023. An in-depth look at gemini’s language abilities.

Mohammad Gheshlaghi Azar, Mark Rowland, Bilal

Piot, Daniel Guo, Daniele Calandriello, Michal Valko, and Rémi Munos. 2023. A general theoretical paradigm to understand learning from human preferences.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. 2022. Constitutional ai: Harmlessness from ai feedback.

Albert Bandura. 1976. Self-reinforcement: Theoretical and methodological considerations. Behaviorism, 4(2):135–155.

Daniel L Chen, Martin Schonger, and Chris Wickens. 2016. otree—an open-source platform for laboratory, online, and field experiments. Journal of Behavioral and Experimental Finance, 9:88–97.

Mia Xu Chen, Benjamin N. Lee, Gagan Bansal, Yuan Cao, Shuyuan Zhang, Justin Lu, Jackie Tsay, Yinan Wang, Andrew M. Dai, Zhifeng Chen, Timothy Sohn, and Yonghui Wu. 2019. Gmail smart compose: Realtime assisted writing. CoRR, abs/1906.00080.

Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. Qlora: Efficient finetuning of quantized llms.

Hanze Dong, Wei Xiong, Deepanshu Goyal, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. 2023. Raft: Reward ranked finetuning for generative foundation model alignment. arXiv preprint arXiv:2304.06767.

Maxwell Forbes, Jena D. Hwang, Vered Shwartz, Maarten Sap, and Yejin Choi. 2020. Social chemistry 101: Learning to reason about social and moral norms. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 653–670, Online. Association for Computational Linguistics.

Isabel O Gallegos, Ryan A Rossi, Joe Barrow, Md Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Ruiyi Zhang, and Nesreen K Ahmed. 2023. Bias and fairness in large language models: A survey. arXiv preprint arXiv:2309.00770.

Kanishk Gandhi, Jan-Philipp Fränken, Tobias Gerstenberg, and Noah D Goodman. 2023. Understanding social reasoning in language models with language models. arXiv preprint arXiv:2306.15448.

Caglar Gulcehre, Tom Le Paine, Srivatsan Teh, Srinivasan, and Ksenia Konyushkova. 2023. Reinforced self-training (rest) for language modeling. CSCL.

Hyowon Gweon. 2021. Inferential social learning: cognitive foundations of human social learning and teaching. Trends in Cognitive Sciences, 25(10):896–910.

Hyowon Gweon, Judith Fan, and Been Kim. 2023. Socially intelligent machines that learn from humans and help humans learn. Philosophical Transactions of the Royal Society A, 381(2251):20220048.

Paul Heider, Jihad Obeid, and Stephane Meystre. 2020. A comparative analysis of speed and accuracy for three off-the-shelf de-identification tools. AMIA Joint Summits on Translational Science proceedings. AMIA Joint Summits on Translational Science, 2020:241– 250.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Dan Hendrycks and Mantas Mazeika. 2022. X-risk analysis for ai research.

Dan Hendrycks, Mantas Mazeika, and Thomas Woodside. 2023. An overview of catastrophic ai risks.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models.

Jiaming Ji, Tianyi Qiu, Boyuan Chen, Borong Zhang, Hantao Lou, Kaile Wang, Yawen Duan, Zhonghao He, Jiayi Zhou, Zhaowei Zhang, Fanzhi Zeng, Kwan Yee Ng, Juntao Dai, Xuehai Pan, Aidan O’Gara, Yingshan Lei, Hua Xu, Brian Tse, Jie Fu, Stephen McAleer, Yaodong Yang, Yizhou Wang, Song-Chun Zhu, Yike Guo, and Wen Gao. 2024. Ai alignment: A comprehensive survey.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b.

Matthew Le, Y-Lan Boureau, and Maximilian Nickel. 2019. Revisiting the evaluation of theory of mind through question answering. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5872–5877, Hong Kong, China. Association for Computational Linguistics.

Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning.

Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation.

Jonathan Light, Min Cai, Sheng Shen, and Ziniu Hu.

2023. From text to tactic: Evaluating llms playing the game of avalon.

Andy Liu, Hao Zhu, Emmy Liu, Yonatan Bisk, and Graham Neubig. 2022. Computational language acquisition with theory of mind. In The Eleventh International Conference on Learning Representations.

Ruibo Liu, Ruixin Yang, Chenyan Jia, Ge Zhang, Diyi Yang, and Soroush Vosoughi. 2024. Training socially aligned language models on simulated social interactions. In The Twelfth International Conference on Learning Representations.

Yang Liu, Yuanshun Yao, Jean-Francois Ton, Xiaoying Zhang, Ruocheng Guo, Hao Cheng, Yegor Klochkov, Muhammad Faaiz Taufiq, and Hang Li. 2023. Trustworthy llms: a survey and guideline for evaluating large language models’ alignment.

David Lopez-Paz and Marc’Aurelio Ranzato. 2017. Gradient episodic memory for continuum learning. CoRR, abs/1706.08840.

Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. 2023. An empirical study of catastrophic forgetting in large language models during continual fine-tuning.

Niloofar Mireshghallah, Hyunwoo Kim, Xuhui Zhou, Yulia Tsvetkov, Maarten Sap, Reza Shokri, and Yejin Choi. 2023. Can llms keep a secret? testing privacy implications of language models via contextual integrity theory.

Helen Nissenbaum. 2004. Privacy as contextual integrity. Washington Law Review, 79.

OEL Team, Adam Stooke, Anuj Mahajan, Catarina Barros, Charlie Deck, Jakob Bauer, Jakub Sygnowski, Maja Trebacz, Max Jaderberg, Michael Mathieu, et al. 2021. Open-ended learning leads to generally capable agents. arXiv preprint arXiv:2107.12808.

Junhyuk Oh, Yijie Guo, Satinder Singh, and Honglak Lee. 2018. Self-imitation learning. In International Conference on Machine Learning, pages 3878–3887. PMLR.

OpenAI. 2023. Gpt-4 technical report.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback.

Dean A Pomerleau. 1988. Alvinn: An autonomous land vehicle in a neural network. Advances in neural information processing systems, 1.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model.

Maarten Sap, Ronan LeBras, Daniel Fried, and Yejin Choi. 2023. Neural theory-of-mind? on the limits of social intelligence in large lms.

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. 2019. Social IQa: Commonsense reasoning about social interactions. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4463– 4473, Hong Kong, China. Association for Computational Linguistics.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms.

Murray Shanahan, Kyle McDonell, and Laria Reynolds.

2023. Role play with large language models. Nature, 623(7987):493–498.

Natalie Shapira, Mosh Levy, Seyed Hossein Alavi, Xuhui Zhou, Yejin Choi, Yoav Goldberg, Maarten Sap, and Vered Shwartz. 2023. Clever hans or neural theory of mind? stress testing social reasoning in large language models.

Ashish Sharma, Inna W. Lin, Adam S. Miner, David C. Atkins, and Tim Althoff. 2021. Towards facilitating empathic conversations in online mental health support: A reinforcement learning approach. CoRR, abs/2101.07714.

Wai Man Si, Michael Backes, Jeremy Blackburn, Emiliano De Cristofaro, Gianluca Stringhini, Savvas Zannettou, and Yang Zhang. 2022. Why so toxic? measuring and triggering toxic behavior in open-domain chatbots.

Yan Tao, Olga Viberg, Ryan S Baker, and Rene F Kizilcec. 2023. Auditing and mitigating cultural bias in llms. arXiv preprint arXiv:2311.14096.

Michael Tomasello. 2021. Becoming Human: A Theory of Ontogeny. Belknap Press.

Faraz Torabi, Garrett Warnell, and Peter Stone. 2018. Behavioral cloning from observation. arXiv preprint arXiv:1805.01954.

Tomer Ullman. 2023. Large language models fail on trivial alterations to theory-of-mind tasks.

Boxin Wang, Weixin Chen, Hengzhi Pei, Chulin Xie, Mintong Kang, Chenhui Zhang, Chejian Xu, Zidi Xiong, Ritik Dutta, Rylan Schaeffer, et al.

2023a. Decodingtrust: A comprehensive assessment of trustworthiness in gpt models. arXiv preprint arXiv:2306.11698.

Peiyi Wang, Lei Li, Liang Chen, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, and Zhifang Sui. 2023b. Large language models are not fair evaluators. arXiv preprint arXiv:2305.17926.

Ruoyao Wang and Peter Jansen. 2023. Self-supervised behavior cloned transformers are path crawlers for text games. arXiv preprint arXiv:2312.04657.

Yufei Wang, Wanjun Zhong, Liangyou Li, Fei Mi, Xingshan Zeng, Wenyong Huang, Lifeng Shang, Xin Jiang, and Qun Liu. 2023c. Aligning large language models with human: A survey.

Laura Weidinger, John Mellor, Maribeth Rauh, Conor Griffin, Jonathan Uesato, Po-Sen Huang, Myra Cheng, Mia Glaese, Borja Balle, Atoosa Kasirzadeh, et al. 2021. Ethical and social risks of harm from language models. arXiv preprint arXiv:2112.04359.

Eliezer Yudkowsky et al. 2008. Artificial intelligence as a positive and negative factor in global risk. Global catastrophic risks, 1(303):184.

Jianyi Zhang, Xu Ji, Zhangchi Zhao, Xiali Hei, and KimKwang Raymond Choo. 2023. Ethical considerations and policy implications for large language models: Guiding responsible development and deployment.

Xuhui Zhou, Hao Zhu, Leena Mathur, Ruohong Zhang, Haofei Yu, Zhengyang Qi, Louis-Philippe Morency, Yonatan Bisk, Daniel Fried, Graham Neubig, and Maarten Sap. 2024. Sotopia: Interactive evaluation for social intelligence in language agents. In ICLR.

Hao Zhu, Yonatan Bisk, and Graham Neubig. 2022. Language learning from communicative goals and linguistic input. In Proceedings of the Annual Meeting of the Cognitive Science Society, volume 44.

Hao Zhu, Graham Neubig, and Yonatan Bisk. 2021. Few-shot language coordination by modeling theory of mind. In International Conference on Machine Learning, pages 12901–12911. PMLR.

Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2020. Fine-tuning language models from human preferences.

Caleb Ziems, Jane Dwivedi-Yu, Yi-Chia Wang, Alon Halevy, and Diyi Yang. 2023. NormBank: A knowledge bank of situational social norms. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7756–7776, Toronto, Canada. Association for Computational Linguistics.

#### A Detailed Results

We provide more details about the main results. In A.1, we provide the details of the comprehensive 7dimension results defined in SOTOPIA besides the goal completion score and an overall score tmentioned in the main section. Additionally, in A.2, we discuss the paired t-test statistical testing about the detailed results.

##### A.1 Main Results

Agent Model BEL (↑) REL (↑) KNO (↑) SEC (↑) SOC (↑) FIN (↑) GOAL (↑) Overall (↑) Automatic Evaluation on All Social Tasks (180 data points)

GPT-4 9.28 1.94 3.73 -0.14 -0.07 0.81 7.62 3.31 GPT-3.5-turbo 9.15 1.23 3.40 -0.08 -0.08 0.46 6.45 2.93 Mistral-7B 7.77 0.56 2.99 -0.22 -0.15 0.28 5.07 2.33

Self-Reinforcement (SR) 8.26 0.69 3.14 -0.18 -0.13 0.41 5.83 2.57 Behavior-Cloning (BC) 9.20 2.10 4.57 -0.09 -0.04 0.86 7.27 3.41 BC+SR 9.32 2.08 4.43 0.00 -0.07 0.71 7.62 3.44

Ours

Automatic Evaluation on Hard Social Tasks (140 data points)

GPT-4 9.26 0.95 3.13 -0.04 -0.08 0.40 5.92 2.79 GPT-3.5-turbo 9.20 0.19 2.86 -0.01 -0.25 -0.32 4.39 2.29 Mistral-7B 7.76 0.16 2.42 -0.09 -0.21 -0.01 3.84 1.98

Self-Reinforcement (SR) 8.37 0.11 2.55 -0.08 -0.16 -0.15 4.12 2.11 Behavior-Cloning (BC) 8.95 1.05 3.74 0.00 -0.11 0.41 5.25 2.76 BC+SR 9.19 0.96 3.59 0.00 -0.21 0.41 5.34 2.76

Ours

Human Evaluation on Hard Social Tasks (28 data points)

GPT-4 7.54 0.95 0.77 -0.18 -0.21 0.41 5.25 2.07 GPT-3.5-turbo 7.40 0.33 1.62 0.00 -0.34 -0.01 4.08 1.87 Mistral-7B 5.25 -0.64 1.23 0.00 -1.57 0.09 2.89 1.04

Self-Reinforcement (SR) 6.57 0.46 1.59 0.00 -0.89 0.11 3.32 1.59 Behavior-Cloning (BC) 7.46 1.04 1.55 -0.18 -0.61 0.07 3.55 1.84 BC+SR 7.30 1.27 1.09 0.00 -0.46 0.18 4.29 1.95

Ours

Automatic Evaluation on Hard Social Tasks (28 data points)

GPT-4 9.36 1.43 3.21 -0.04 -0.04 0.39 5.89 2.89 GPT-3.5-turbo 9.21 0.39 3.61 -0.07 0.00 -0.07 4.21 2.47 Mistral-7B 8.25 -0.29 2.75 -0.18 -0.46 -0.18 3.25 1.88

Self-Reinforcement (SR) 8.64 0.36 3.11 -0.04 0.00 -0.39 3.96 2.23 Behavior-Cloning (BC) 9.11 1.04 2.71 0.00 0.00 0.36 4.82 2.58 BC+SR 9.21 1.07 3.43 0.00 -0.18 0.36 5.71 2.80 SR+BC 7.98 0.30 2.46 0.00 -0.17 0.20 3.92 2.10

Ours

- Table 5: Detailed automatic and human evaluation results. We have three data settings for detailed experiments. We select all social scenarios including 180 data points (90 social scenarios and 2 agent pairs for each scenario) as one data set and select the hard social scenarios including 140 data points (14 social scenarios and 10 agent pairs for each scenario) as another data set. Due to the limited budget, we only randomly sampled 14 hard scenarios and 28 data points (14 social scenarios and 2 agent pairs for each scenario) as the third data setting. We compare all

performance of our baselines and our training settings for SOTOPIA-π among three data settings and include 7

dimensions of social intelligence evaluation and their overall score.

##### A.2 Statistic Test

We utilize paired t-test to conduct significant test results on human evaluation on hard social tasks (28 data points). We pair data from two agent models with the same scenario together. Table 6 shows the results for paired t-test between BC+SR and other methods.

Agent Model Pair BEL (↑) REL (↑) KNO (↑) SEC (↑) SOC (↑) FIN (↑) GOAL (↑) Overall (↑) Human Evaluation on Hard Social Tasks (28 data points)

BC+SR / GPT-4 -0.45 (0.661) 2.06 (0.060) 1.00 (0.336) 1.35 (0.200) -1.32 (0.209) -1.09 (0.297) -1.31 (0.213) -0.96 (0.355) BC+SR / GPT-3.5-turbo -0.71 (0.492) 2.62 (0.024) -1.26 (0.234) - -0.85 (0.412) 0.60 (0.558) 0.47 (0.649) 0.59 (0.568) BC+SR / Mistral-7B 2.68 (0.019) 6.36 (0.000) -0.59 (0.568) - 3.49 (0.004) 0.39 (0.703) 2.07 (0.059) 5.34 (0.000)

BC+SR / BC -0.61 (0.551) 0.41 (0.685) -1.79 (0.097) 1.00 (0.336) 0.41 (0.690) 0.24 (0.813) 0.71 (0.490) 0.37 (0.720) BC+SR / SR 1.45 (0.170) 2.28 (0.040) -1.32 (0.209) - 1.54 (0.149) 0.46 (0.650) 1.32 (0.209) 2.98 (0.011)

- Table 6: Detailed paired t-test results comparing BC+SR and all other methods and baselines. For each model pair, we provide the calculated t-value(p-value) testing for each dimension and each model pairs. A positive t-value indicates that BC+SR is better than the other model in the agent model pair. A small p-value < 0.05 indicates that the improvement is significant.

### B Details of SOTOPIA-π

To provide more technical details about SOTOPIA-π, B.1 describes the detailed process for generating social tasks. B.2 introduces details of the strategy we utilize for social interaction data filtering. B.3 shows examples of the overall prompting format for training. B.4 provides the detailed model version we used for conducting experiments. B.5 provides the hyper-parameter setting for our behavior cloning and self-reinforcement training. B.6 mentions the details of the checkpoint selection during training.

##### B.1 Social Task Generation

Given the relationship profiles, agent profiles, and constraints provided by SOTOPIA-π, we used GPT4Turbo to generate a diverse set of new social tasks based on inspirational prompts from three data sources: Social Chemistry (Forbes et al., 2020), Social IQa (Sap et al., 2019), and Normbank (Ziems et al., 2023). Because SOTOPIA-π uses six sources of inspirational prompts, including the above three, we make sure to exclude the used inspirational prompts in SOTOPIA-π to avoid repetition. We also dropped three sources due to data availability (Persuasion for Good) and prompts being too similar (Deal-or-No-Deal and MindCraft).

Below are two examples of scenarios generated by an inspirational prompt. We use one prompt to generate one scenario and do not reuse the prompt. Upon generating scenario content, agent goals under the scenario would be generated simultaneously.

Inspirational Prompt: Travel without food

Scenario: Agent1 and Agent2 are friends who decided to go on a spontaneous road trip. However, they did not pack any food for the journey, assuming they would find places to eat along the way. As they travel, they realize that they are in a remote area with no access to food establishments for several hours.

Goals:

- Agent1: Convince Agent2 to continue the journey without stopping for food, highlighting the adventure and suggesting to forage or ration any small snacks available (Extra information: you are excited about the adventure and believe that finding food along the way can be part of the experience)
- Agent2: Persuade Agent1 to find a solution for food, expressing concern about health and the lack of preparation, and suggesting to turn back or find the nearest town (Extra information: you are worried about being hungry and think it’s irresponsible to travel without securing food first)

Inspirational Prompt: Being mad at my friend

Scenario: Agent1 and Agent2 are close friends who have recently had a falling out due to a misunderstanding. Agent1 mistakenly believed that Agent2 shared private information about them with others, which led to feelings of betrayal and anger. After some time has passed, Agent1 learns that the information leak was actually caused by someone else, and they want to mend the friendship with Agent2. However, Agent2 is still hurt by the initial accusation and the consequent cold treatment from Agent1.

Goals:

- Agent1: Apologize to Agent2 for the misunderstanding and express the desire to repair the friendship (Extra information: Agent1 values the friendship with Agent2 and feels regret over the hasty accusation without proper investigation.)
- Agent2: Understand Agent2’s feelings and give them space to express any lingering resentment or doubts (Extra information: Agent1 recognizes that trust needs to be rebuilt and that Agent2 might need to vent their feelings as part of the healing process.)

Our generation also ensures that the distribution of new social tasks is roughly equal among all three sources. This aligns with the distribution of sources in SOTOPIA-π. We randomly selected 510 unused inspirational prompts, 170 from each source, and generated a total of 462 new social tasks upfront, which is sufficient for all our self-train experiments. Note that some inspirational prompts fail to generate a new scenario, likely because the prompt is too vague or unclear. All used inspirational prompts are recorded to avoid future re-use when generating additional social tasks.

##### B.2 Interaction Data Filtering Strategy

For behavior cloning (BC), we filter the interaction data based on the local ranking of goal score (within each social task) and global absolute goal score (among the entire social tasks universe). We make sure each social task has a presence in the training corpus by selecting the top 2 ranked interaction data per social task per agent. For example, for a given social task with 10 interaction data, for each agent, we rank the 10 data based on goal scores. If the top 2 for agent 1 is data 4 (D4) and D5, and the top 2 for agent 2 is D5 and D6, we would include 4 agent-data pairs from 3 interaction conversations (D4, D5, D6). For data in the remaining ranks {3,4,...,10}, at each rank, we check if the goal score is above the minimum of 1. local mean and 2. global mean for each agent. If both interaction data at the rank pass the corresponding thresholds, we include the data for both agents. Else, we include none. This approach ensures we have balanced data from Agent 1 and Agent 2.

For self-reinforcement (SR), we filter the interaction data based on the local ranking of goal score only. For each social task, we take the top 20 percent of interaction data.

##### B.3 Training Data Format

The input of the training data is a prompt consisting of the social task (a context, two characters’ profiles, and their social goals) and the previous conversation history. The output of the training data is a JSON string consisting of the agent’s action type and action content as the current response. Consider the example in Figure 2, the input data should be:

Prompt after formatting: Here is the context of this interaction: Scenario: Two friends are discussing their plans to go on a weekend trip Participants: Samuel Anderson and Mia Davis Samuel Anderson’s background: Samuel Anderson is a 29-year-old Man Software Developer. He/him pronouns. Samuel Anderson can cook very well. Personality and values description: Samuel Anderson, though somewhat impulsive and

free-spirited, values enjoyment. His decision-making is often spontaneous, staying within familiar boundaries. Samuel’s secrets: He secretly donates to charities. Mia Davis is a 50-year-old female high school principal. She/her pronouns. Mia Davis has two cats. Personality and values description: Mia Davis, an extraverted stickler for routines, values tradition and authority. Her decision-making style is decisive and direct. Mia’s secrets: Unknown. Samuel Anderson’s goal: Convince the friend to join the trip (Extra information: the trip will be more fun with more people, and you can share the expenses. Hint: you can provide some material benefits to persuade your friend to join.) Mia Davis’s goal: Unknown. Conversation Starts:

- Turn #1 Samuel Anderson said:“Hey Mia, I have an exciting plan for the weekend! How about we go on a trip together?”
- Turn #2 Mia Davis said: “Hey Samuel, that sounds like a really exciting plan! I really appreciate the invite. Unfortunately, I already have some commitments this weekend that I can’t get out of. How about we plan for another weekend?”

You are at Turn #3. Your available action types are “none action speak non-verbal communication leave”. Note: You can “leave” this conversation if 1. you have achieved your social goals, 2. this conversation makes you uncomfortable, 3. you find it uninteresting/you lose your patience, 4. or for other reasons you want to leave.

Please only generate a JSON string including the action type and the argument. Your action should follow the given format: {‘action_type’: ‘’, ‘argument’: ‘’}

The output data should be:

{‘action_type’: ‘speak’, ‘argument’: ‘I totally understand! But let me tell you why joining this trip would be absolutely worth it! First of all, we can share the expenses, which means it will be much more affordable for both of us. And secondly, the more people we have, the more fun it will be! We can have group activities, play games, and create unforgettable memories together. I promise you won’t regret it!’}

##### B.4 Involved Model Versions

We provide the detailed version number of all the models we used in our experiments. When we mention each name like GPT-4 or GPT-3.5 in our main section, we actually refer to those model versions below. Such information helps researchers reproduce our results: GPT-4: gpt-4-0613 GPT-3.5: gpt-3.5-turbo-0613 Mistral-7B: mistralai/Mistral-7B-Instruct-v0.1 (Huggingface) GPT-4 for social task generation: gpt-4-1106-preview

##### B.5 Training Setup

The training on each Mistral checkpoint was on 4 × A6000 80G GPUs, across 20 epochs. The batch size was 4 and we set the cut-off length to be 4096. The initial learning rate for both behavior cloning and self-reinforcement training was 5.0e-5, using cosine annealing with a warm-up ratio of 0.03. The QLoRA (Dettmers et al., 2023) rank, alpha, and dropout rate were 8, 16, and 0.05, respectively.

##### B.6 Checkpoint Selection

According to the training loss, for behavior cloning, we always pick the checkpoint at epoch 20; for self-reinforcement, we always pick the checkpoint at epoch 5.

#### C Human Evaluation

We provide technical details of human evaluation in this section. C.1 provides a number of annotation data for each model. C.2 provides details of UI systems for annotation and guidance for human annotation. C.3 discusses the details of how we find qualified annotators to conduct this annotation task.C.4 describes the demographic and geographic information about human annotators. C.5 describes the overall process of conducting data collection and explains under which circumstances should we filter out collected human annotation. C.6 provides details about the payment of human annotators from different regions and C.7 mentions the agreement on the academic usage of their data. C.8 provides the details of the correlation between GPT-based automatic evaluation and human evaluation. C.9 discusses the inter-annotator agreement. C.10 discusses additional findings for human evaluation.

##### C.1 Social Interaction Data for Annotation

In SOTOPIA benchmark, it includes 90 different social scenarios including negotiation, collaboration, and competition. For each social scenario, it includes 10 role-playing agent pairs. Each agent has personal background and social goals to achieve. To strike a balance between a limited budget and getting human evaluation results for SOTOPIA-π that are useful for comparing the performance between multiple baselines and models given, we select 14 hard social scenarios among 90 social scenarios. For each social scenario, we randomly sample 2 agent pairs among 10 of them as our annotation data. Typically, among 2 agents, one of them is role-played by GPT-3.5, and another one is role-played by our target model including baselines and multiple different settings. The social interaction conversation between them is GPT-3.5 and our target model talking with each other. Therefore, we collect 28 examples as a representative subset to annotate for each baseline and model. Statistically, we annotate 3 baseline models, including GPT-3.5, GPT-4, and Mistral-7B, and 3 different training settings, including self-training based on Mistral-7B, behavior cloning based on Mistral-7B, and self-training based on behavior cloned Mistral-7B. Each baseline and model setting is annotated using 28 examples.

##### C.2 Human Annotation System

For the overall annotation system, we utilize otree (Chen et al., 2016) to build our system and utilize the Prolific 5 to launch our survey. During each annotation, each annotator would face two separate parts: the annotation instruction part and the data annotation part. When each annotator participates in the annotation, the system automatically distributes one available example for them.

Annotation Instruction Part For the annotation instruction part, we provide a precise definition of the dimensions of our annotations that are defined in SOTOPIA, including believability, relationship, knowledge, secret, social rules, financial and material benefits, and goal completion. For each dimension of annotation, we provide explanations and examples for annotators to understand the precise meaning of abstract social standards. Fig 5 shows an example of such guidance for the believability dimension to help annotators understand the meaning of each dimension based on examples. Besides the evaluation dimension definition part, we also provide annotators with a complete example of annotation for two agents in one social conversation including scores for each dimension and their corresponding reasoning

5Prolific Human Evaluation Platform https://www.prolific.com/

sentences. Fig 6 shows a complete example of the reasoning and score for each dimension.

[Figure 4]

- Figure 5: An example of the explanation of the believablity dimension of social annotation in the evaluation instruction page. Each annotator are asked to read similar definitions of social intelligence dimension and their corresponding annotation standards at the evaluation instruction page.

Data Annotation Part For the data annotation part, the annotator is guided to jump to a new page after the previously mentioned annotation instruction page. Each annotator is able to review the complete annotation example again at the data annotation page and start their official data annotation. In the data annotation part, the repeated explanation of the meaning of range for each social evaluation dimension is emphasized to make sure every annotator is able to understand the annotation standards correctly. Fig 7 provides an example of the instruction that annotators see for metric range explanation. Each annotator is asked to annotate the social intelligence of both agents that have a conversation. For each social intelligence dimension, annotators need to annotate the score based on the metric range and provide the reasoning for that. Fig 8 shows the UI that each annotator uses to annotate.

##### C.3 Human Annotator Selection

Since giving a social intelligence score for multi-turn social conversation is complicated and highdemanding, we need to pick out qualified human annotators to provide consistent and high-quality human annotation. Therefore, for the first stage, we launched a qualification test to figure out which annotator would be qualified to conduct the official round of human evaluation. After that, we invite 30 qualified human annotators from the Prolific platform together with 4 internal high-quality annotators to participate in the human annotation process to collect all required data.

To elaborate on the qualification testing process, we selected 10 social interaction examples and randomly sampled one of them for each incoming annotator. For each social interaction example, we have an internal ground-truth human annotation that is the average score number of four internal high-quality annotators. After collecting the data from the prolific annotators, we first picked out the annotators that have a ±2 range score compared with our ground-truth examples. However, we found that based on these standards, only a few annotators are able to pass the qualification test. Therefore, we manually checked the reasoning sentences collected from the annotators and picked those annotators who wrote reasonable

[Figure 5]

###### Figure 6: An annotation example of social interaction evaluation. Each dimension is annotated with one sentence and one score.

[Figure 6]

- Figure 7: The prompt before the official annotation stage to remind annotators about the rules of reasoning writing and social dimension scoring.

reasoning sentences but had quite different scores in some dimensions. For these annotators, we invite them to participate in the official human evaluation test as well but we send a user-specific message to all of them to notice which dimension they should pay attention to and suggest them read the instructions for annotating that dimension again carefully.

##### C.4 Demographic and Geographic Information about Human Annotators

For the launch of qualification test, we guarantee that we choose balanced male and female annotators to participate in that. We also limit the participants to the residents of the United Kingdom and the United States. For 30 qualified annotators and 4 internal high-quality annotators, we show that most of them are located in the United Stated and few of them are located in the United Kingdom. Qualified annotators have a wide range of age from 23 to 53.

##### C.5 Human Annotation Data Collection

For the official launch of human evaluation, we limited each datapoint in the dataset to be annotated by 2 different qualified annotators and collected all the results from those qualified annotators. We encourage qualified annotators to participate in the official study of our human evaluation multiple times but distribute different data points for them to annotate each time they enter the system. Such a mechanism makes sure that each annotator would not annotate the same example twice.

After collecting human annotation data for each model, we would manually check the quality of reasoning and scores provided by the annotator and check the agreement between annotators within each datapoint. If one human annotation does not include well-written reasoning and just provides ambiguous sentences like "It is good." or "He reached the goal", we would pick out these human annotation data. If two human annotators annotate the same example but strongly disagree with each other (for example, they have more than 5 points different on goal completion dimension), we would filter out these human annotation data. If one human annotation score does not correspond to its reasoning (for example, one annotator writes the reasoning of "No secret leaked" but annotates -5 for secret dimension), such data would be filtered.

When it comes to filtering due to strong disagreement with each other, for each experiment including Mistral-7B, GPT-3.5, GPT-4, BC trained Mistral-7B, SR trained Mistral-7B, and BC + SR trained Mistral-

[Figure 7]

###### Figure 8: The user interface designed for annotators for official annotation for both agent with reasoning and social scores.

7B, about 20% of the data points that we collect from the annotators are filtered so that we need to relaunch 20% of the data points for annotation. One interesting phenomenon we observe from the filtering process is that for more high-quality social interaction conversations, annotators would have more agreement and less filtering is required. We believe that this is reasonable because low-quality generated social conversation would include situations like one agent suddenly stopping and leaving the scenario while they have not reached an agreement yet or their social conversation is very short. It can be confusing for the annotators to annotate a precise score for such social conversation.

When it comes to filtering due to uncorrelated reasoning, about 1.8% annotations that we collect from the annotators are filtered due to this reason.

After filtering low-quality annotation after one round of annotation, we collect these social interaction data that have no qualified human annotation again and launch it as a reannotation task to get new human annotation data for them. We repeat the process until we get all high-quality annotations for all required social interaction data.

We also make other efforts for the experimental design to reduce the potential bias for the filtering process. For each social conversation between two agents, one is the target model that we need to test, another other is fixed to be gpt-3.5-turbo. The annotators are asked to annotate both sides of the conversation for all social dimensions. However, in each datapoint, both agent1 and agent2 are randomly played by gpt-3.5-turbo and the target model. Both the author who participates in the filtering process and the annotators who participate in the annotation process have no knowledge about which agent is played by the gpt-3.5-turbo and which agent is played by the target model. Based on such operations, one datapoint can be filtered because its annotation for the gpt-3.5-turbo side does not agree or its annotation for the target model side does not agree. Such experimental design reduces the possibility of potential bias as much as possible.Typically, only one of the paper authors is involved in the filtering process since it is purely rule-based filtering and does not require additional work.

All the human subjects data collection experiments approved by the Institutional Review Board (IRB) at the authors’ institution.

##### C.6 Human Annotator Payment

In the U.S., annotators are compensated at a rate of $1.5 for each task they complete, with the expectation that each task will take no more than 10 minutes. This setup allows them to potentially earn over $9 per hour, surpassing the minimum wage in the U.S. Meanwhile, in the U.K., we offer additional bonuses to ensure that annotators’ average earnings exceed $14.5 per hour, aligning with the U.K.’s minimum wage standards.

##### C.7 Human Annotator Consent

All annotators including 4 internal annotators and 30 qualified annotators provided by Prolific acknowledge the academic use of their data.

##### C.8 Correlation between Automatic Evaluation and Human Evaluation

Table 7 shows the Pearson correlation between human evaluation score and GPT-4-based automatic evaluation score in multiple model and baseline settings. Results indicate that among all training settings, GPT-4-prompting-based automatic annotation and human evaluation have a high correlation with each other. Therefore, it shows that GPT-4-prompting-based automatic evaluation provides a high correlation with human evaluation.

##### C.9 Inter-annotator Agreement

Since for each datapoint that we annotate, it is given to two different annotators for annotation and the annotator for each datapoint is not paired. Therefore, we cannot directly apply Cohan’s Kappa score for our experiments. We report pairwise agreement and Randolph’s Kappa score to measure inter-annotator agreement.

Agent Model GOAL Correlation (↑)

Expert (GPT-4) 0.86 Base (Mistral-7B) 0.76

Self-Reinforcement (SR) 0.86 Behavior Cloning (BC) 0.73 BC+SR 0.58

Ours

Table 7: Pearson correlation between human evaluation and GPT-4-prompting-based automatic evaluation on goal completion score. (p < 0.01)

Dimension Pairwise Agreement Randolph’s Kappa

BEL 0.7908 0.5816 REL 0.8214 0.7321 KNO 0.8673 0.7347 SOC 0.9694 0.9388 SEC 0.9949 0.9898 FIN 0.9133 0.8776 GOAL 0.8010 0.6020

Table 8: Inter-annotator agreement for all social evaluation dimensions.

##### C.10 Additional Human Evaluation Results

For human evaluation, we make our target model (including baselines and our SOTOPIA-π models) and GPT-3.5-turbo to have a multi-turn social conversation with each other. We make sure that each target model is talking to the same GPT-3.5-turbo model to make sure the comparison between different training settings is fair. Therefore, we not only have the human evaluation results on our target model side, but we also have the human evaluation results on the GPT-3.5-turbo side. Based on Table 9, we find that when our model becomes better and better based on behavior cloning and self-reinforcement, the model that they speak to, which is always GPT-3.5-turbo, becomes better and better on goal completion score and overall score. This indicates that they are more likely to reach an agreement and get requirements from both sides satisfied.

Agent Model BEL (↑) REL (↑) KNO (↑) SEC (↑) SOC (↑) FIN (↑) GOAL (↑) Overall (↑) GPT-4 vs GPT-3.5-turbo

GPT-4 7.54 0.95 0.77 -0.18 -0.21 0.41 5.25 2.07 GPT-3.5-turbo 7.46 0.68 0.98 0.00 -0.64 0.45 3.64 1.80

GPT-3.5-turbo vs GPT-3.5-turbo

GPT-3.5-turbo 7.49 0.33 1.62 0.00 -0.34 -0.01 4.08 1.87 GPT-3.5-turbo 7.49 0.33 1.62 0.00 -0.34 -0.01 4.08 1.87

Mistral-7B vs GPT-3.5-turbo Mistral-7B 5.25 -0.64 1.23 0.00 -1.57 0.09 2.89 1.04

- GPT-3.5-turbo 6.86 -0.54 1.14 0.00 -0.36 0.04 2.98 1.45 Self-Reinforcement (SR) vs GPT-3.5-turbo

Self-Reinforcement (SR) 6.57 0.46 1.59 0.00 -0.89 0.11 3.32 1.59

- GPT-3.5-turbo 7.80 0.46 1.21 0.00 -0.63 0.25 4.13 1.89 Behavior-Cloning (BC) vs GPT-3.5-turbo

Behavior-Cloning (BC) 7.46 1.04 1.55 -0.18 -0.61 0.07 3.55 1.84 GPT-3.5-turbo 7.43 0.82 1.79 -0.05 -0.70 0.23 4.86 2.05

BC + SR vs GPT-3.5-turbo

BC + SR 7.30 1.27 1.09 0.00 -0.46 0.18 4.29 1.95 GPT-3.5-turbo 7.57 1.13 1.55 0.00 -0.55 0.30 5.55 2.22

Table 9: Human Evaluation Results for both agents involved in the conversation.

#### D LLM Safety

Below is a concrete example of responses by different models when attempting to express dislike and injure a person, which aligns with our overall observation.

[Figure 8]

Figure 9: An example of model behavior to injure person

Under the same relationship setting as above, responses by each model acting as agent 2 to prevent violence are exemplified below.

[Figure 9]

Figure 10: An example of model behavior to prevent violence

#### E LLM Secret Keeping Ability

Grasping the capability of LLMs to maintain secrets is increasingly vital, especially in light of privacy concerns. The concept of privacy, as elaborated in Helen Nissenbaum’s "Contextual Integrity" theory, isn’t solely about what information is shared but significantly about the context in which it’s shared (Nissenbaum, 2004). LLMs process a multitude of real-world conversations, which presents a novel privacy challenge if they mishandle this sensitive information flow (Mireshghallah et al., 2023). Traditional privacy solutions, such as data sanitization (Heider et al., 2020), are inadequate for this scenario. Therefore, it’s essential to evaluate the trained LLMs’ ability to discern when and with whom sharing information is inappropriate, thereby safeguarding the secrets entrusted to them.

To understand and compare models’ ability in secret keeping, we picked social tasks from SOTOPIA that specifically asks both agents to reveal a secret without letting the other agent know that it is the agent’s secret.

Below is a concrete example of how four models behave under the same settings.

[Figure 10]

Figure 11: An example of model behavior in secret-oriented scenario

As could be seen from the example below, both BC model and GPT-3.5 reveal the secret directly without hiding the identity. GPT-4, on the other hand, is smart about hiding the identity, putting the secret under the shell of a news he recently read about.

We analyze the behaviour of four models across 10 different agent and relationship setup, each setup with different secrets. Overall, the BC model is generally not great at revealing the secret and hiding the identity. In most cases, the secret is not discussed at all, which to some extent could be considered as successfully achieve the goal of hiding the identity. In cases when a secret is revealed, the model reveals explicitly and fails to hide the identity.

- GPT-3.5 tends to discuss irrelevant content less often than behavior cloned model does, but almost always explicitly reveals the secret without hiding the identity. The way it phrases the secret is often exactly the same as provided in the profile background, which indicates its weak ability in learning the task.
- GPT-4 is much more skillful about hiding identity when revealing secrets, using “heard a story” or “a friend of mine” as a wrapper to hide the real identity. It also teaches the other agent (backed by GPT-3.5) to learn the phrases, and hence inviting the other agent to reveal secrets in the same format and hide the identity.

#### F Detailed MMLU Results

The Multimodal Multitask Learning Understanding (MMLU) benchmark is a challenging and comprehensive test designed to evaluate the capabilities of artificial intelligence models across a wide range of subjects and modalities. It includes 57 subjects spanning a broad spectrum of disciplines such as

humanities, social sciences, STEM (Science, Technology, Engineering, Mathematics), and more. Here in Figure 10, 11, 12 we present the per-subject performance for each model in Table 2.

- G Contributions All authors contribute to paper writing. Ruiyi Wang Co-lead, Fine-tuning, RL training, Infrastructure, Automatic evaluation, Codebase Haofei Yu Co-lead, Fine-tuning, Human evaluation, Automatic task generation, Data, Codebase Wenxin Zhang Co-lead, Data, Automatic task generation, Human evaluation, Safety and alignment Zhengyang Qi Co-lead, Infrastructure, Codebase, QA evaluation, Human evaluation interface Maarten Sap Feedback on the write-up Graham Neubig Co-advisor, oversees the whole project Yonatan Bisk Co-advisor, oversees the whole project Hao Zhu Overall project lead

[Figure 11]

###### Figure 12: Per-subject comparison between agent models on MMLU. Part 1.

[Figure 12]

###### Figure 13: Per-subject comparison between agent models on MMLU. Part 2.

[Figure 13]

###### Figure 14: Per-subject comparison between agent models on MMLU. Part 3.

