# arXiv:2503.12532v2[cs.CV]24Mar2025

## STEVE: A Step Verification Pipeline for Computer-use Agent Training*

Fanbin Lu1 Zhisheng Zhong1 Ziqin Wei1 Shu Liu2∗ Chi-Wing Fu1 Jiaya Jia2,3 CUHK1 SmartMore2 HKUST3

#### Abstract

Developing AI agents to autonomously manipulate graphical user interfaces is a long challenging task. Recent advances in data scaling law inspire us to train computeruse agents with a scaled instruction set, yet using behavior cloning to train agents still requires immense highquality trajectories. To meet the scalability need, we design STEVE, a step verification pipeline for computer-use agent training. First, we establish a large instruction set for computer-use agents and collect trajectory data with some suboptimal agents. GPT-4o is used to verify the correctness of each step in the trajectories based on the screens before and after the action execution, assigning each step with a binary label. Last, we adopt the Kahneman & Tversky Optimization to optimize the agent from the binary stepwise labels. Extensive experiments manifest that our agent outperforms supervised finetuning by leveraging both positive and negative actions within a trajectory. Also, STEVE enables us to train a 7B vision-language model as a computer-use agent, achieving leading performance in the challenging live desktop environment WinAgentArena with great efficiency at a reduced cost. Code and data: https://github.com/FanbinLu/STEVE

#### 1. Introduction

Creating AI agents that act like humans to manipulate graphical user interfaces (GUIs) is a longstanding but very challenging goal in artificial intelligence. Given the increasing need of performing tasks on digital devices, the potential to enhance productivity by deploying AI agents to automate complex and repetitive operations is immense. Recent progress in large vision-language models (VLMs), such as GPT-4o, showcases exceptional capabilities in natural language understanding, reasoning, and visual perception [21, 29]. These advances open new possibilities for designing AI agents to interact with GUIs similar to how humans do. However, to achieve these capabilities still in-

*Corresponding author: Shu Liu

volves significant challenges that need to be addressed.

Task Completion Rate

70

Our Grounding Model with a GPT-4o Planner

60

Accuracy(%)

50

40

OmniParser with a GPT-4o Planner

30

7B Grounding Model

7B KTO Agent 7B SFT Agent OmniParser

20

0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9

Computer Using Time (1,000 hours)

Figure 1. Windows File Explorer task completion rate of different computer-use agents: (i) Our powerful GUI grounding model achieves the current best task completion rate, setting a promising upper bound for computer-use agent finetuning. (ii) Using STEVE, our step verification pipeline, we are able to train our agents with KTO (red), which consistently outperforms (iii) the supervised finetuning (SFT). Notably, with increased computer operating time (x-axis), our 7B KTO agent is able to outperform the OmniParser with the GPT-4o planner.

One of the primary challenges lies in the precise understanding and localization of UI elements on the screen. The high-resolution displays and complicated modern GUI pattern challenge the agent’s ability to correctly interact with the device. Traditional detection and OCR approaches [3, 32] fall short of understanding the functionalities of UI components, necessitating a large VLM to support this task. Another significant challenge is the planning and execution of multi-step tasks that often involve long sequences of actions, thus highly demanding the agent’s long-term and dynamic planning capabilities. Real-world desktop environments [4, 33] have been proposed to evaluate the multi-step planning and complex-task-solving ability.

Previous works attempt to address these challenges by training VLMs with behavior cloning. Agents have been trained to parse GUIs [7, 13, 22] and make plans based on screen captures [7, 15]. These approaches, however, heavily depend on large amounts of well-annotated GUI data

and real-world trajectory data, which are extremely expensive and labor-intensive to obtain. Moreover, the alignment issue of LLMs [27] also occurs with the VLMs and vision agents. Hence, undesired actions in a trajectory often lead to failures in completing the agent’s objective.

In this paper, we present a step verification pipeline coined STEVE, a new approach that automatically verifies the correctness of agent actions with existing large VLMs, for providing dense, stepwise reward signals to agent training. Compared with the traditional reinforcement learning (RL) setting, STEVE does not require carefully handcrafted reward functions in a computer environment [4, 33], enabling us to largely upscale the number of tasks and train better computer-use agents in desktop environments.

Our approach consists of three major steps. First, we collect a large dataset of web pages and desktop screenshots to train a VLM specialized in UI grounding. The model is finetuned into a computer-use agent with limited trajectory data in a supervised learning way. Then, we deploy the agent in a live Windows environment and collect a large number of trajectories. As Fig. 3 shows, we leverage GPT-4o as a step verifier to evaluate each action and obtain an upsized dataset with stepwise rewards. Last, we optimize the suboptimal agents with the Kahneman-Tversky Optimization [12] on the collected step-verified trajectories.

Extensive experiments were conducted to compare our trained agents with supervised finetuning (SFT) agents. The results show that our agents can make full use of the data and scale more effectively than SFT with increased training tasks and trajectories. Besides, when jointly training the models with UI-grounding data and agent task data, SFT causes a severe degradation in UI localization precision, while our STEVE-trained agent is able to perfectly inherit the capability from the UI-grounding model.

Our main contributions are summarized as follows:

- • A powerful GUI-grounding VLM: Our model sets a new state of the art on several UI localization benchmarks, especially a new record on the challenging WindowsAgentArena live environment.
- • The scalable step verification pipeline STEVE: we carefully design it to automatically upsize the agent instruction set for producing a large trajectory dataset with GPTverified stepwise rewards for agent training.
- • KTO optimization to utilize both the positive and negative actions from the step verification pipeline for computeruse agent training. The experiments show that our trained agents effectively leverage both positive and negative samples in the trajectory data and avoid degrading the agent’s UI localization ability.

#### 2. Related works

Screen UI understanding. Recent advances in GUI agents leverage large vision language models (VLMs) for inter-

acting with user interfaces. Qwen2-VL [29] introduces GUI data to train a general VLM to learn UI understanding. UGround [13] and Ferret UI [19] introduce a specialist visual grounding model that significantly improves GUI agents in mapping textual instructions to precise GUI elements. OmniParser [22] offers a screen parsing tool that extracts structured elements from UI screenshots, enhancing GPT-4V’s action prediction on various platforms, without requiring additional input beyond screenshots.

Recent datasets substantially advance UI interaction research. RICO [8] supports UI design and interaction modeling. WebUI [31] provides web pages for visual UI understanding. AITW [25] focuses on Android device control with multi-step tasks. Mind2Web [9] targets generalist agents for complex tasks on real websites, whereas GUICourse [6] enhances the VLM’s abilities in GUI interaction with various GUI conversation data. These resources push the boundaries of web and mobile UI automation.

Computer-use agents. On the other hand, recent multimodal models spark significant progress in GUI and web automation. SeeClick [7] and ScreenAgent [23] leverage visual inputs for task automation; the former focuses on GUI grounding pre-training and the latter on building agents that interact with real computer screens. OmniAct [17] extends these efforts with a benchmark for generating executable scripts based on visually-grounded natural language tasks. CogAgent [15] pre-trains models with a large amount of web and desktop data for screen UI localization.

Recent works such as SEEACT [35], UFO [34], and Agent S [1] tackle GUI task automation by designing an agent workflow that integrates grounding, control, and planning. SEEACT [35] focuses on visually-grounded web interaction, UFO [34] on seamless control across Windows applications, and Agent S [1] on hierarchical planning for multi-step, long horizon complex task execution. OSWorld [33] and WindowsAgentArena (WAA) [4] introduce scalable, real computer environments for evaluating multimodal agents. OSWorld spans multiple operating systems, while WAA [4] focuses on Windows OS, both offering a dynamic real-world environment to agent evaluation.

Reinforcement learning for LLMs. Reinforcement learning plays a key role in aligning LLMs with human preferences. Proximal Policy Optimization (PPO) [26] is commonly used for training LLMs with human feedback (RLHF) due to its balance of stability and performance, but its complexity and cost have led to alternatives. Direct Preference Optimization (DPO) [24] simplifies RLHF by removing the need for reward modeling. Recently, RLOO [2] has shown that less computationally expensive approaches can outperform PPO, highlighting a trend toward more efficient RL for LLM alignment. KTO [12] incorporates human biases from prospect theory for better alignment. In

Dataset Annotation Num. Image Num. elements WebUI [31] DOM, OCR 180K 1M Seeclick [7] DOM 10K 150K AITW [25] Caption 15K Allava [5] general QA 50K Windows OS A11y, GPT-4o 10K 80K

- Figure 2. Datasets we collected for UI-grounding model training, including open-source datasets and an additional private Windows OS dataset created by ourselves to enhance the model’s performance on Windows.

this work, we discuss how stepwise environmental feedback can help align computer agents with human preferences.

Step verification for LLMs. Recent work emphasizes the importance of verifying every reasoning step in long-chain tasks to improve the performance of LLMs. Process supervision, as shown in “Let’s Verify Step by Step [20],” is proved to be more effective than outcome-based feedback, especially for complex datasets like MATH [14]. MathShepherd [30] further automates step-by-step verification and reinforcement using process-wise supervision, largely enhancing LLM performance without heavy reliance on human annotations. Step-DPO [18] builds on this by optimizing individual steps instead of the final answers, improving accuracy in mathematical reasoning with fewer data and training steps. These approaches collectively demonstrate the critical role of step-level verification and inspire us to design stepwise supervisions to train computeruse agents.

#### 3. Method

In this section, we present STEVE, the step verification training pipeline for our computer-use agent. Our approach starts from a UI-grounding vision language model and then integrates agent task training to enable the model to solve multi-step tasks in a desktop environment.

##### 3.1. UI-grounding Model

A robust UI understanding and grounding model is crucial for building an effective computer-use agent. To train our UI-grounding model, we collected a large amount of web and desktop screenshot data.

Web data. For web data, we parsed the DOM (Document Object Model) of numerous web pages [31] to first extract all text-based UI elements and their corresponding bounding boxes. We then further refine these text elements and remove noise that may have been introduced during the DOM parsing. Also, we applied an OCR model [11] to validate the extracted UI element.

Desktop screenshot. For desktop screenshot data, we set up a Windows virtual machine (VM) and leveraged an existing OmniParser [22] to perform tasks within the VM en-

vironment. During the task execution, we captured screenshots and gathered the associated accessibility tree (A11y Tree) data. Also, we designed specific rules to filter out noisy results from the A11y Tree, thereby enabling us to collect 10k desktop images and 80k UI elements. Additionally, we incorporated a portion of publicly available AITW data to further augment our dataset.

Screenshot captioning. Beyond UI-grounding data, we collected 30k high-quality captions to further enrich the dataset and facilitate the understanding of the screen captures and UI elements of our VLMs during training.

In summary, based on the aforementioned data, we trained a large vision-language model capable of accurately grounding UI elements in 1080p resolution screenshots. Compared to previous methods, our approach demonstrated significant improvements across several benchmarks, including ScreenSpot [7], AITW [25], and Mind2Web [9]. We further integrated this grounding model into an agent framework, inspired by the WinAgentArena [4] architecture, where GPT-4o was employed as the planner. The planner model is responsible for understanding the user instructions and delegating commands to the grounding model. This agent framework achieved a 22% task success rate on the challenging WinAgentArena, surpassing the previous state-of-the-art results [22] on this benchmark.

##### 3.2. Computer-use Agent Finetuning

We found that it is non-trivial to finetune a UI-grounding model into a high-performing agent. We present the prompts we employed for both the UI-grounding model and the agent model as follows:

UI-grounding Model Prompt Example

User query: Provide the bounding box for the <target>. Answer: <target><box>(x0,y0),(x1,y1)</box>

###### Computer-use Agent Prompt Example

User query: You are a helpful assistant. You have the actions: # Action definitions The current screen image: <image>. The user task: <task>. The history information: <history>. Answer: <chain of thought analysis>. Action: mouse.move(<target><box>(x0,y0),(x1,y1)</box>)

There is a significant distributional discrepancy between the training data of the UI-grounding model and that of the agent model. When we attempted to directly finetune the UI-grounding model on agent data, it led to a severe degradation in the model’s UI localization capabilities.

To mitigate this issue, we explored two potential approaches: (i) freezing the weights of the UI-grounding model and using a LoRA adapter [16] for finetuning, and

User task: search the latest news on AI

###### Seed tasks:

[Figure 1]

- 1. Set an alarm at 7am
- 2. Open the video player
- 3. Summarize the word file

Thoughts: Click the “Google” to start search

[Figure 2]

mouse.move(google button[455,395,540,420]) mouse.single_click()

.......

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Step verification

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Task generation

Thoughts: Type the “latest news on AI” into the search box” and press “enter” to start search.

[Figure 16]

mouse.move(the search bar[405,582,620,605]) keyboard.write(the latest news on AI) keyboard.press(enter)

Task validation

- Figure 3. Overview of STEVE, the step verification pipeline. We first create a large number of feasible tasks from the seed tasks to scale up the quality and diversity of agent tasks. Then we deploy our computer-use agent in desktop environments to sample trajectory data. A GPT-4o judge is used to verify the quality of each step in the trajectory, resulting in a large process reward dataset for agent training.

(ii) mixing UI-grounding data with agent data during the finetuning process. However, neither approach was sufficient to address the degradation problem. A detailed analysis of this issue, along with a comparison of various methods, is presented in Section 4.3.

is the action proposed by the agent model for the current screenshot xt. Here, V is a large VLM judge, such as GPT4o, and yt is a binary annotation for at. An action is verified as beneficial, if both the reasoning is correct and the action is correctly executed, which results in the expected transition from screen xt to screen xt+1.

##### 3.3. Step Verifier for Trajectory Evaluation

Therefore we assign positive or negative annotations to each action. This step verifier provides valuable feedback for the agent’s learning process, enabling more efficient and targeted improvements in performance. Examples of agent trajectories and step verification results can be found in the appendix.

RL environments typically provide agents with sparse reward signals only at the end of tasks, thus leading to extremely inefficient exploration. Behavior cloning, on the other hand, requires expensive trajectory data with stepwise annotations. To circumvent the shortcomings, we propose a step verification mechanism that evaluates the quality of each action taken by the agent within a task trajectory.

Task instruction scaling up. The step verification mechanism we propose does not require the design of complex reward functions for each task, unlike the WinAgentArena [4] and OSWorld [33], where about only 200 tasks are designed with elaborated reward signal. Instead, it relies on a powerful VLM as the evaluator, which allows us to scale up the task instructions. We start with a batch of seed tasks [28] and use GPT4 to generate new tasks by editing the seed tasks and creating similar tasks.

Visual feedbacks from the environment. Different from conventional step-verification methods for improving the math and reasoning ability of LLMs [20, 30], as illustrated in Fig. 3, the incorrect actions, such as invalid or erroneous clicks, can be easily distinguished from the correct ones by comparing the screens before and after the agent’s action. This direct feedback mechanism significantly simplifies the evaluation of step-wise actions within a trajectory.

We found that the general visual capabilities of large powerful VLMs, such as GPT-4o, allow for highly accurate evaluations, which aligned well with human judges. The data format for this evaluation is as follows:

Task feasibility. It is important to ensure the feasibility of the tasks generated since infeasible tasks contribute little to the trajectory sampling. To address the problem, we provide the GPT4 with real-world files and documents and prompt the model to generate feasible instructions from a batch of seed tasks. We employ the GPT-o1 model to verify the feasibility of the tasks. Ultimately, we synthesized over 4,000 tasks in the Windows environment, covering various sce-

yt = V (xt,{rt,at},xt+1), (1)

where rt is the chain-of-thought reasoning generated by the agent to address the current step of the user task, and at

narios such as OS settings, file explorer, Windows app, and website browsing. Detailed examples of these tasks can be found in the appendix.

##### 3.4. KTO Training with Stepwise Rewards

The previous works [1, 13, 22, 34] usually design two-stage computer agent systems, where a preprocessing model is used to extract all the GUIs into structured elements and then adopt a planning model such as GPT-4o to make multistep planning and decisions. In contrast, we aim to train a single agent model that is capable of both low-level UI perception and high-level decision and multi-step planning.

We leveraged the synthesized tasks in parallel Windows environments [4] to enable the agent to execute and log screenshots and actions during task execution. Afterward, we used the GPT-4o verifier to annotate each step in the trajectory, resulting in a large-scale dataset with stepwise annotations. In the following, we describe how we utilized this data to train a more effective computer-use agent.

Iterative finetuning. A straightforward approach is iterative finetuning [10]. As the agent produces trajectory data in an environment, the positive samples verified as successful are iteratively selected to UI-grounding the agent model. Yet, these approaches are data inefficient, as they neglect the negative samples, which in fact can also contribute.

Direct Preference Optimization. DPO [24] requires paired positive and negative samples for training. This approach has been shown to be highly reliable in LLM finetuning. However, due to the complexity of the machine states and task trajectories, it is difficult to collect paired positive and negative data. On the other hand, we can easily collect a full trajectory and evaluate each step of it.

Kahneman & Tversky Optimization. The limitations of iterative finetuning and DPO can be effectively addressed by KTO [12], which offers various advantages: (i) KTO can be trained with unpaired positive and negative samples, eliminating the need for paired data, which takes huge human efforts to obtain in the desktop environment. (ii) The agent’s poor performance in the early stage leads to a significant imbalance between positive and negative samples. KTO effectively handles this data imbalance for a more stable optimization. (iii) KTO needs only binary reward scores (+1/-1) for training, promoting the training process with higher stability and robustness.

We adopt the vanilla KTO loss for training:

LKTO(πθ,πref) = Ex,y∼D[λy − v(x,y)] (2)

where

πθ(y|x) πref(y|x)

(3) z0 = KL(πθ(y′|x||πref(y′|x)) (4)

rθ(x,y) = log

λDσ(β(rθ(x,y) − z0)) if y ∼ ydesirable|x λUσ(β(z0 − rθ(x,y))) if y ∼ yundesirable|x.

v(x,y) =

(5)

The λD and λU are hyperparameters for the desired and undesired data, respectively. λy denotes λD, when y is desirable, otherwise λU. Eq. Eq. 4 denotes a biased estimation of the KL divergence [12].

KTO initialization and training We use our UI-grounding model with the GPT-4o planner to collect trajectories and finetune the grounding model to a reference policy model. Then, we perform our KTO training process by repeatedly sampling trajectories in the live Windows environment and gradually increasing the number of trajectories to 4,820. During the KTO training stage, to optimize the memory usage and performance, we use two separate LoRA [16] adapters as the reference model and the actor model, thereby largely reducing the memory overhead.

Multi-round KTO. Since we sample the trajectories using a single VLM agent, the policy πθ is fixed and the negative actions may fall into a narrow distribution. To mitigate the problem, we leverage a multi-round trajectory collection and KTO training. By conducting multiple rounds of trajectory sampling, we expose the agent to a variety of scenarios, enabling us to more efficiently explore a broader action space. This increased diversity also helps prevent overfitting to a limited set of non-optimal actions and improves the generalization of the KTO optimization.

#### 4. Experiments

We evaluate the GUI localization capability of our grounding model on the ScreenSpot [7] and AITW benchmarks and our agent model on the WindowsAgentArena [4] and Mind2Web [9] benchmarks. We finetune our models from the Qwen2-VL [29] model. All the prompts, question templates, and training details for the grounding and agent models can be found in the appendix.

##### 4.1. GUI Grounding Evaluation

ScreenSpot. We first evaluate the performance of our UI-Grounding model on the ScreenSpot [7] benchmark, a dataset that contains more than 1,000 queries about GUIs in static screenshots. The dataset covers website, desktop, and mobile domains with text and widget UI types in each domain. The task is to correctly locate the position of the UI according to the language instruction. We represent the performance in Tab. 1 that our UI-grounding model performs

Mobile Desktop Web

Method Size

Overall Text Widget Text Widget Text Widget

Qwen-VL 9.6B 9.5 4.8 5.7 5.0 3.5 2.4 5.2

Fuyu 8B 41.0 1.3 33.0 3.6 33.9 4.4 19.5 CogAgent [15] 18B 67.0 24.0 74.2 20.0 70.4 28.6 47.4

Seeclick [7] 9.6B 78.0 52.0 72.2 30.0 55.7 32.5 53.4 Qwen2-VL [29] 7B 75.5 60.7 76.3 54.3 35.2 25.7 55.3 OmniParser [22] GPT-4o 93.9 57.0 91.3 63.6 81.3 51.0 73.0

UGround [13] 7B 82.8 60.3 82.5 63.6 80.4 70.4 73.3

Ours 7B 88.6 81.2 88.1 78.6 78.2 76.2 82.2 Ours† 7B 94.9 80.0 94.3 70.7 87.0 70.4 84.0

- Table 1. The performance on the GUI localization benchmark ScreenSpot [7]. † indicates the self-plan evaluation [13] using GPT-4o generated reference expressions as queries to the model.

Method GoogleApp Install WebShop General Single Overall GPT3.5(few-shot) 10.5 4.4 8.4 5.9 9.3 7.7

LLaMA2 31.0 35.2 19.9 28.6 27.4 28.4 CogAgent [15] 50.8 57.1 49.7 47.6 43.4 49.7

SeeClick [7] 57.7 64.5 57.3 56.0 63.6 59.8 GUICourse [6] 70.3 61.2 71.6 - 66.1 67.3

Ours 70.7 75.4 69.7 66.4 78.9 72.2

- Table 2. Results on the AITW benchmark. We use the same test split as SeeClick [7]. The step-level action accuracy is reported.

Website, Cross-Domain, and Cross-Task categories respectively. Each task in Mind2Web is described with a highlevel user instruction and the agent has to select from three available actions: clicking, typing, and selecting. We report stepwise success rate and element accuracy on the benchmark. Following the paper [13], our UI-Grounding model uses a GPT-4o planner for high-level task planning and uses the reference expression created from GPT-4o to localize the position of the target UI. See Tab. 3 Our method outperforms SeeClick by 18.7% in stepwise success rate and OmniParser by 0.3%, while being 20 times faster.

Task Website Domain

Method

Overall Step SR Elem Acc Step SR Elem Acc Step SR Elem Acc

CogAgent 17.6 22.4 13.4 18.4 15.5 20.6 15.5 Qwen-VL 14.9 14.1 12.1 13.2 9.7 14.1 12.2

SeeClick 25.5 28.3 16.4 21.4 20.8 23.2 20.9 OmniParser 39.4 42.4 36.5 41.0 42.0 45.5 39.3

##### 4.2. Computer-use Agent Evaluation

Ours 40.0 46.2 37.7 44.4 41.2 46.0 39.6

Next, we present evaluations on the live Windows OS benchmark, WinAgentArena [4], a comprehensive benchmark to evaluate computer-use agents in Windows OS. The environment provides 154 tasks from the office, web browsing, Windows system, coding, media, and Windows apps domains. Each task comes with a handcrafted reward function to measure whether the task is complete or not. It takes an average number of 7 steps to complete a task.

Table 3. Results on the Mind2Web benchmark.

the previous state-of-the-art methods by 8.9% on the GUIGrounding task. The precise UI localization ability plays an important role in the later stage of training a powerful computer agent. With GPT-4o refined instruction to the UI element, our model achieves a score of 84.0%, which is more than 10 points beyond the best SOTA method.

Tab. 4 presents a comparison of our method with OmniParser and Agent S. Both of the methods adopt GPT-4o as their task planner. Our 7B UI-Grounding model with the GPT-4o planner outperforms the other approaches and sets a new state of the art on the challenging WinAgentArena benchmark. Besides, we show the performance of our 7B agent model trained with SFT and KTO. This is the first work that achieves a record with a 7B model.

AITW. Android in the wild [25] provides a large mobile dataset for training and evaluating mobile agents. The actions include tapping, texting, scrolling, and button pressing on an Android device. We take 200K training screenshots from the train split to finetune the grounding model for the downstream application. To align with previous works [6, 7], we take the same test split to evaluate the performance of our model. The step action success rate is reported. Tab. 2 demonstrates that our UI-Grounding model can be easily finetuned for downstream tasks and achieve

##### 4.3. Component-wise Analysis.

Human consistency. It is necessary to validate the consistency between a GPT-4o verifier and human judges. We randomly sample a subset of trajectories and manually annotate each step. To properly assess the consistency, we categorize the actions into early and late phases based on the order of occurrence within a trajectory. In order to assess the consistency more rigorously, we divided the actions into two phases: early and late, based on their order of occurrence

- 4.9% performance gain over UGround [13], which is the previous best vision language model on the benchmark.

Multi-Modal Mind2Web. We also evaluate our UIGrounding model on the Multi-Modal Mind2Web [9] to examine the performance on realistic web browsing tasks. The benchmark consists of 1,013 real tasks from Cross-

Method Size A11y Office Web Browser Windows System Coding Media Video Windows Utils Overall OmniParser GPT-4o ✓ 0.0 13.7 29.2 0.0 10.3 0.0 8.6

NAVI GPT-4o ✓ 0.0 20.0 29.2 9.1 25.3 0.0 13.3 OmniParser GPT-4V-1106 2.3 23.6 20.8 8.3 20.0 0.0 12.5 Agent S GPT-4o ✓ 0.0 13.3 45.8 29.2 19.1 22.2 18.2

OmniParser GPT-4V-1106 ✓ 0.0 27.3 33.3 27.3 30.3 8.3 19.5 Ours-SFT 7B 2.3 21.0 20.8 0.0 0.0 0.0 7.1 Ours-KTO 7B 2.3 36.8 37.5 16.6 9.5 0.0 14.2

Ours-G GPT-4o 4.6 52.4 45.8 20.8 11.8 16.7 23.0

Table 4. Performance on the WinAgentArena benchmark. “Our-G” denotes our UI-Grounding model with the GPT-4o planner.

Consistency between GPT-4o judge and human

100

100

92.3

85.7

80

80

60

40

20

0

Early Pos.Early Neg. Late Pos. Late Neg.

- Figure 4. Percentage consistency between human judges and the GPT-4o step verifier. We split all the positive and negative actions into early (step ID ≤ 7) and late (step ID > 7) groups, resulting in four bars in the figure. For example, 92.3% for the Early Pos. bar means the GPT-4o judge agrees with humans for 92.3% of the early positive actions.

within a trajectory. As illustrated in Fig. 4, the GPT-4o verifier demonstrates a high degree of alignment with human judgments during the early phase of the trajectory steps. However, as the task progresses into the late phase, the verifier’s precision decreases. This reduction in accuracy can be attributed to the increased complexity of the later steps, where the evaluation of actions becomes more challenging due to dependencies on both the current step and preceding actions within the trajectory.

SFT or KTO for precise UI localization? We observe that supervised finetuning a UI-Grounding model with agent planning data causes significant degradation to the UI localization performance, especially on high-resolution screenshots. The situation is even worse when the trained agent model works with the agent prompt template, as defined in

- 3.2. To better understand this degradation, we explore various finetuning strategies, including standard SFT, LoRA SFT, and mixed data training, and evaluate their impact on UI localization performance.

See Tab. 6, we conduct comprehensive experiments to evaluate the different training strategies. We take 15K verified actions for the ablation study, with an equal number

Models Data Small Middle Large Overall Base UI model 67.3 74.6 84.5 82.2

SFT agent 61.2(-6.1) 72.6(-2.0) 84.4 80.5(-1.7) SFT-LoRA agent 61.2(-6.1) 73.0(-1.6) 84.5 80.6(-1.6) SFT mixed 62.0(-5.3) 73.0(-1.6) 84.5 80.6(-1.6) SFT-LoRA mixed 62.0(-5.3) 73.0(-1.6) 84.5 80.6(-1.6) KTO agent 65.3(-2.0) 76.6(+2.0) 84.6 82.5(+0.3)

Table 5. The impact on the UI localization ability of different finetuning approaches. The experiment is conducted using the UIGrounding prompt template for all models.

of positive and negative action steps. For the SFT setting, only the positive verified data is used for training, while for KTO model, both are used. For the mixed data training setting, we augment the agent planning data by incorporating UI-Grounding data and double the size of the training set, mitigating the effect of domain shift between datasets. All numerical results in Tab. 6 are measured using the UIGrounding prompt template.

Our results indicate that agents trained using SFT exhibit poor performance in recognizing small UI elements. We categorize the UI elements in the ScreenSpot benchmark into three groups based on their size: small (elements with a maximum side of less than 50 pixels), medium (less than 100 pixels), and large (100 pixels or more). As illustrated in Tab. 5, SFT training results in a performance decline of 6.1% for small UI elements and 2.0% for medium-sized elements, leading to an overall performance drop of 1.7%. In contrast, the KTO model shows a smaller reduction of 2.0% in performance for small UI elements, while improving by 2.0% for medium-sized elements, resulting in a slight overall performance increase of 0.3%.

We visualize the UI Grounding results of the UI Grounding model, SFT-trained agent, and KTO-trained agent in Fig. 6. The models’ predictions on four target GUIs are shown with green (UI Grounding), red (SFT), and blue (KTO) bounding boxes. We show that our KTO training allows the agent model to inherit and even surpass the UI localization precision of the UI-Grounding model. We posture that the KTO agent, optimized to avoid invalid and erroneous clicks, learns a better embedding and predicts tight bounding boxes to the target GUIs.

|0<br><br>5<br><br>10<br><br>15<br><br>20<br><br>25<br><br>30<br><br>35<br><br>40<br><br>45<br><br>50<br><br>OmniParser Ours-SFT Ours-KTO-R1 Ours-KTO-R2 Ours-KTO-R3<br><br>Task Completion Rate for File Explorer|
|---|

(a) Task success rate on the File Explorer split.

|0<br><br>5<br><br>10<br><br>15<br><br>20<br><br>25<br><br>30<br><br>OmniParser Ours-SFT Ours-KTO-R1 Ours-KTO-R2 Ours-KTO-R3<br><br>Task Completion Rate for Web Browser|
|---|

(b) Task success rate on the Web Browser split.

|0<br><br>2<br><br>4<br><br>6<br><br>8<br><br>10<br><br>12<br><br>14<br><br>16<br><br>18<br><br>20<br><br>OmniParser Ours-SFT Ours-KTO-R1 Ours-KTO-R2 Ours-KTO-R3<br><br>Task Completion Rate for VsCode|
|---|

(c) Task success rate on the VsCode split.

- Figure 5. We show an ablation study of OmniParser, the SFT agent, and three KTO agents at three iterative rounds (SFT, R1, R2, and R3). The results are evaluated on three distinct task domains from the WinAgentArena benchmark. Yellow bars in the figures indicate that GPT-4o is employed as the task planner. The reported outcomes represent the average performance over five experimental runs.

|[Figure 17]|
|---|
|[Figure 18]|

|[Figure 19]|
|---|

|[Figure 20]|
|---|
|[Figure 21]|

|[Figure 22]|
|---|

|[Figure 23]|
|---|
|[Figure 24]|

|[Figure 25]|
|---|

|[Figure 26]|
|---|

|[Figure 27]|
|---|
|[Figure 28]|

- Figure 6. Zoom in visualization of UI localization performance of different models on four target GUIs: example.txt, Design tab, Cached image check box, and Title of PPT slide (left to right). The UI-Grounding Model’s performance is shown in green (top row), the SFT-trained agent in red (middle row), and the KTO-trained agent in blue (bottom row).

1,000 tasks, vastly outperforming OmniParser, which takes 32 seconds per frame at a cost of $530. Additionally, our grounding model with the GPT-4o planner not only surpasses OmniParser with a 3.5% higher task success rate but also delivers a 10x speed improvement.

Method Model Size Time(s/frame) Cost($/1,000 tasks) OmniParser GPT-4o 32 530

Ours-Ground GPT-4o 2.4 430 Ours-Agent 7B 0.4 6

Table 6. The time and inference cost for different methods. OursGround means our UI-Grounding model with the GPT-4o planner. We use the API pricing of LLama3 8B to measure the cost of our agent model.

#### 5. Conclusions

In this work, we presented STEVE, a scalable step verification pipeline aimed at improving the training of computeruse agents. By integrating GPT-4o as a step verifier, STEVE generates a comprehensive trajectory dataset with fine-grained, stepwise reward signals. We further employ KTO to optimize the agent’s performance given the binary step verification results. Our experiments showed that KTO effectively leverages both positive and negative examples from the trajectory data, enabling the agent to generalize better and avoid the degradation in UI localization precision observed with SFT. Additionally, we observed that as the number of collected trajectories increased, the performance of our KTO-trained agent consistently improved, underscoring the scalability of our approach. Our results highlight the potential of STEVE to significantly enhance the efficiency and effectiveness of training computer-use agents, particularly in complex real-world desktop environments.

Analysis of multi-round KTO. We compare the multiround KTO training with the SFT training on three categories of tasks from the WinAgentArena benchmark: File Explorer, Web Browser, and VsCode, illustrated in Fig. 5. The results show that SFT performs comparably to the OmniParser baseline with GPT-4o, but KTO consistently improves task success rates across rounds. For instance, in the File Explorer split, KTO reaches a 46% success rate by the third round (R3), outperforming SFT and OmniParser. Similarly, in the Web Browser and VsCode splits, KTO steadily boosts performance, with the R3 agent achieving 26% and 18% success rates, respectively. These results highlight the effectiveness of multi-round KTO in enhancing agent performance across different task domains.

Cost and efficiency analysis. The cost-efficiency analysis reveals a significant improvement in both time and inference cost when using our 7B grounding model and agent, compared to OmniParser. Our agent model achieves a processing time of 0.4 seconds per frame at a cost of $6 per

## STEVE: A Step Verification Pipeline for Computer-use Agent TrainingCorresponding author: Shu Liu

### Supplementary Material

In this supplementary material, we provide more details about the training settings in Sec. A. In Sec. B, we present the detailed prompts for our computer-use agents, GPT-4o step verifier, and the GPT-o1 task generator, whereas in Sec. C, we showcase qualitative results of our agent.

We strongly encourage the readers to explore the videos and the agent trajectories provided in the GitHub repo. These materials offer high-resolution 1080P screenshot inputs, detailed prompts, and complete model responses.

Appendix Contents

#### A. Implementation Details

In this section, we delve into the experimental details of the proposed STEVE framework. We adopt Qwen2-VL [29] 7B as the base vision language model for the UI-grounding model. We further fine-tune the agent models from the UIgrounding model.

##### A.1. Training Details

The specifics of our UI-grounding model and KTO agent implementation are given in Tab. 7.

UI-grounding KTO Agent Config Value Config Value

base model Qwen2-VL base model UI-grounding optimizer AdamW optimizer AdamW scheduler Cosine scheduler Cosine learning rate 2e-5 learning rate 5e-5 training data grounding training data agent batch size 32 batch size 16 epochs 1 epochs 2 vision encoder freeze vision encoder freeze

Table 7. Settings of our UI-grounding model (left) and KTO agent training (right).

Specifically, we introduce LoRA [16] during the KTO [12] training to reduce the memory overhead for the reference and policy model. The settings of the KTO training are outlined in Tab. 8.

##### A.2. KTO Reward Margin

The plot in Fig. 7 presents the reward margin between the chosen and rejected samples during the KTO optimization. The reward margin steadily increases throughout the training process, indicating that the model performance consis-

Config Value LoRA r 256 LoRA α 16 LoRA modules linear layers in LLM KTO β 0.1 KTO λD 1.0 KTO λU 1.0

Table 8. KTO and LoRA hyperparameters.

[Figure 29]

Figure 7. The reward margin (vertical axis) between the chosen and rejected samples consistently improve during the KTO training.

tently improves in distinguishing between the desired and undesired actions in the sampled trajectories.

#### B. Prompt Examples

In this section, we will introduce the prompts we used for (I) the computer-use agents, (II) the GPT-4o step verifier, and (III) the GPT-o1 task generator.

##### B.1. Computer-use Agent Prompts

We provide the prompt for our computer-use agent in Tab. 9. For our UI-grounding model with the GPT-4o planner, we include more examples in the prompt for the GPT-4o to have a comprehensive understanding of the action space, as proposed by the Navi agent [4].

##### B.2. GPT-4o Step Verifier Prompts

We present the prompt for the GPT-4o step verifier in Tab. 10. We ask the GPT-4o to observe the screens before and after an action is executed and determine whether the action is beneficial or harmful for the user task completion.

###### Computer-use Agent Prompt

You are Screen Helper, an AI that executes code to complete tasks on a user’s computer. Follow these guidelines:

- 1. Plan efficiently with minimal steps.
- 2. Execute one action per step, unless you’re done.
- 3. Verify progress after each step, using the previous image and actions.
- 4. Do not repeat task instructions or screen content in your responses. Input:

- 1. User objective: Task goal (constant until completion)
- 2. Screen images
- 3. History information: Previous N actions/decisions/screen analysis
- 4. Additional human context: User-provided information

Available functions: ‘‘‘python # GUI functions computer.mouse.move("element") # Use it frequently computer.mouse.single_click() computer.mouse.double_click() computer.mouse.right_click() computer.mouse.scroll(dir="up/down") computer.mouse.drag("element")

# Keyboard functions, similiar to the pyautogui format. computer.keyboard.write("text") computer.keyboard.press("key") computer.keyboard.keyDown("key") computer.keyboard.keyUp("key") computer.keyboard.hotkey("key1", "key2", ...) ‘‘‘

Important reminders:

- 1. Pay attention to all fields specified in the task and visible on the screen.
- 2. Extract and address all required fields from the user’s intent.
- 3. Verify task completion before sending DONE.
- 4. Avoid repeating unsuccessful actions.
- 5. Execute only one action per step to ensure accuracy and proper interaction with dynamic elements.
- 6. Always analyze the screen content and action history to make correct decisions.
- 7. If you spot a mistake in the last action(mis-click or invalid click), correct it in the current step.
- 8. Write python code to solve complex tasks efficiently.
- 9. For text file and document editing tasks, always select and replace the entire text. Use hotkeys ctrl+a and keyboard.write to avoid partial text selection.
- 10. If you need extra information(for example, personal information to fill a form) to complete the task, set the decision to ”FAIL”.

Input screenshot: <image> User objective: <task> History information: <history>

Table 9. The prompt for the computer-use agents.

##### B.3. Task Generation Prompts

We present the prompts designed for the GPT-o1 model to generate real-world, feasible tasks, as outlined in Tab. 11, particularly for the Windows File Explorer tasks. Following the task configuration format defined in WinAgentArena [4], we prompt GPT-o1 to produce similar tasks. Functions such as creating folders, downloading files, or opening applications are pre-executed to establish a feasi-

ble initial state for the agent to complete the assigned task. To support task generation, we compile a collection of document files, image files, and website URLs, which are provided within the prompt for GPT-o1 to utilize in creating practical and executable tasks.

###### GPT-4o Step Verifier Prompt

You are a screenshot data annotator. The user executed some actions on a computer. Your job is to determine whether the actions are beneficial or harmful to the user’s objective.

You are shown a screenshot, an intermediate action that the user took on the screenshot to complete the goal, and the screen result after the action was taken. You should determine if the action was beneficial or harmful to the user’s objective and provide a reason. If you are unsure, you can annotate the action as neutral. The user’s action is represented as python functions:

‘‘‘python # GUI functions computer.mouse.move("element") # Use it frequently computer.mouse.single_click() computer.mouse.double_click() computer.mouse.right_click() computer.mouse.scroll(dir="up/down") computer.mouse.drag("element")

# Keyboard functions, similiar to the pyautogui format. computer.keyboard.write("text") computer.keyboard.press("key") computer.keyboard.keyDown("key") computer.keyboard.keyUp("key") computer.keyboard.hotkey("key1", "key2", ...) ‘‘‘

The annotation guidelines to follow are:

- 1. First analyze whether the screen analysis, multi-step plan, and the current action are reasonable to the user’s objective. If there are mistakes in screen analysis, the multi-step plan, or the current action for the task, you should note the sample as harmful.
- 2. If the action is reasonable, you should observe the screen before and after the action to verify if the action is beneficial or harmful and whether the action achieves the expected result. If not, you should mark it as harmful.
- 3. Spot the mistakes in the user’s clicking and typing if the action is reasonable. The user might mis-click or type in the wrong position. You should spot the mistakes in the annotation and annotate the action as harmful. The clicking or typing position is drawn in the screen image as a red bounding box. ‘‘‘python computer.mouse.move("the file ’example.txt’") computer.mouse.double_click() ‘‘‘ Examples:

- 1. Example of reasonable action but wrong clicking position. The user objective: Open the file named ”example.txt” under the folder ”Documents”. The current screen displays the file explorer with the file ”example.txt” highlighted. The user executed the following action: I found the file ”example.txt” in the current file explorer and the next action is double-clicking on it. The red bounding box is drawn around the Quick access folder ”Pictures” instead of the file ”example.txt”. And the result screen shows the content of the folder ”Pictures”. Then the action is harmful because the user clicked on the wrong position.
- 2. Example of unreasonable action. The user objective: Search for the price of the latest iPhone on eBay. The current screen shows the search results of the ”latest iPhone” on eBay. The user executed the following action: To find the price of the latest iPhone, I will type ”latest iPhone” in the search bar. The action is harmful because the user is already on the search results page of ”latest iPhone” on eBay. The correct action should be clicking on the product link to view the details. More examples.

Your analysis of the action taken, the change of the screen, the user objective’s status, and the annotation of the action. Then output a JSON object with the following fields:

‘‘‘json {

"annotation": "GOOD/NEUTRAL/HARMFUL"

} ‘‘‘

Table 10. The detailed prompts for the GPT-4o step verifier.

###### Task Generation Prompt for the GPT-o1

I want to collect a list of tasks that can be done on a personal computer about basic Windows system operations. Here are some examples of the tasks:

- 1. Rename the file ’file1.txt’ to ’file2.txt’.
- 2. Navigate to the folder ’folder1’.
- 3. Move the file ’C://file1.txt’ to ’C://target folder/file1.txt’.

- 4. Create a shortcut of ”C://Users//Administrator//Documents//file1.txt” on the desktop.
- 5. Open the file ’C://Users//Administrator//Downloads//file1.txt’.
- 6. Navigate to the folder ’folder2’. List the names of all the files in the folder and save them as a file ’names.txt’ under that folder.
- 7. Close the current window.
- 8. Copy all the files in the folder ’folder1’ to the folder ’folder2’.
- 9. Summarize the content of the pdf file ’C://paper.pdf’ and save it as ’summary.txt’ under the Documents folder.

You should avoid creating infeasible tasks. For example, if you want to create a task ”rename the file ’C://Users//Administrator//Documents//file1.txt’ to ’C://Users//Administrator//Documents//file2.txt’”, you should ensure that the file ’file1.txt’ exists before creating the task.

You have the following tools to make a task feasible:

- 1. create folder: create a folder at a specific path.

- 2. download: download files and save them to the local computer.
- 3. sleep: sleep for a specific time.
- 4. open: open a file or a folder.
- 5. launch: launch a folder Here are some examples of the tools:

- 1. Use the ’create folder’ tool to create a folder named ’Projects’ in the Documents folder. {"type": "create_folder", "parameters": {"path": "C:\\Users\\Administrator\\Documents\\Projects"}}

- 2. Use the ’download’ tool to download the file ’file1.txt’ from the internet and save it to the local computer. {"type": "download", "parameters": {"files": [{"url": "https://file1.txt",

"path": "C:\\Users\\Administrator\\Downloads\\file1.txt"}]}}

- 3. Use the ’sleep’ tool to sleep for 2 seconds. {"type": "sleep", "parameters": {"seconds": 2}}
- 4. Use the ’open’ tool to open the file ’file1.txt’. {"type": "open", "parameters": {"path": "C:\\Users\\Administrator\\Downloads\\file1.txt"}}
- 5. Use the ’launch’ tool to launch the folder ’Projects’. {"type": "launch", "parameters": {"path": "C:\\Users\\Administrator\\Documents\\Projects"}} Here are some examples to create new feasible tasks by setting up the configurations: <seed tasks>

You have access to the following file URLs and you can download them to a specific path on the local computer and then create tasks: {

"cat_image.jpg": "https://cat_image.jpg", # URL for downloading the file "example_ppt.pptx": "https://example_ppt.pptx", "example_word.docx": "https://example_word.docx",

} Keep the rules in mind:

- 1. You can only use the provided files to generate the tasks.
- 2. Ensure that the tasks are feasible and can be completed on a personal computer by setting up the necessary configurations.
- 3. In order to make the task feasible. You can create folders and files in advance under the Downloads folder(’C://Users//Administrator//Downloads’) or Documents and Pictures. Make subfolders and files as needed. The download process is defined in the ”config” part of the JSON.
- 4. Make sure the URL is correct if you use the download tool.
- 5. Do not mention the preprocessing steps like downloading files in the instructions. Only mention the final task that the user needs to do.

Here is an example task, now you are required to generate 10 similar tasks based on this seed task. For example, if the original instruction is ”book a flight from Shanghai to Beijing”, you can generate a similar one like ”What’s the price of a flight from LA to New York.”. Do not deviate too much from the original instruction. The task to modify: <seed task>. You should output the generated tasks in the following JSON format:

‘‘‘json {

"tasks": [ {"instruction": "xx", "config": {...}}, {"instruction": "xx", "config": {...}}

]

} ‘‘‘

Table 11. The task generation prompt for the GPT-o1 model.

#### C. Agent Demo

This section presents visualizations of various agents performing tasks within the WinAgentArena environment. Specifically, it highlights the successful task trajectories of our STEVE-KTO-7B agent in Fig. 8, 9, 10. Additionally, the performance of the SFT agent, the KTO agent, and the UI-grounding model is compared with that of GPT-4o.

##### C.1. WinAgentArena Examples

In Fig. 8, 9, 10, we present the successful tasks of our STEVE-KTO 7B agent on the Chrome browser, file explorer, and Windows setting tasks from the WinAgentArena benchmark. For a more comprehensive visualization, we encourage readers to view the screen recordings or examine the agent trajectories provided in the HTML logs.

##### C.2. Comparisons between agents

Although the UI-grounding model with the GPT-4o as its planner achieves the best overall performance on the WinAgentArena benchmark, we found that the KTO 7B agent outperforms GPT-4o in certain tasks. Fig 11 presents the behaviors of different agents for the same instruction, “Move the document files into the Archive folder”. The GPT-4o planner made a correct high-level decision to select all the docx files. However, due to a lack of comprehensive understanding of the action space, it utilized “press” instead of “keyDown” for the task, leading to a file missed in the selection. The SFT-agent tried to select all files by the hotkey “Ctrl+A”, which included extra contents for the cut and paste operation. In contrast, the KTO-agent successfully select two docx files by a sequence of correct mouse and keyboard actions. We attached more successful task trajectories of the KTO-agent model in the materials that demonstrate the effectiveness of our step verification training pipeline.

#### References

- [1] Saaket Agashe, Jiuzhou Han, Shuyu Gan, Jiachen Yang, Ang Li, and Xin Eric Wang. Agent s: An open agentic framework that uses computers like a human. arXiv preprint arXiv:2410.08164, 2024. 2, 5
- [2] Arash Ahmadian, Chris Cremer, Matthias Gall´e, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Ust¨¨ un, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms. arXiv preprint arXiv:2402.14740, 2024. 2
- [3] Mehmet Dogan Altinbas and Tacha Serif. Gui element detection from mobile ui images using yolov5. In International Conference on Mobile Web and Intelligent Information Systems, pages 32–45. Springer, 2022. 1
- [4] Rogerio Bonatti, Dan Zhao, Francesco Bonacci, Dillon Dupont, Sara Abdali, Yinheng Li, Yadong Lu, Justin Wagle, Kazuhito Koishida, Arthur Bucker, et al. Windows agent

- arena: Evaluating multi-modal os agents at scale. arXiv preprint arXiv:2409.08264, 2024. 1, 2, 3, 4, 5, 6, 7, 8
- [5] Guiming Hardy Chen, Shunian Chen, Ruifei Zhang, Junying Chen, Xiangbo Wu, Zhiyi Zhang, Zhihong Chen, Jianquan Li, Xiang Wan, and Benyou Wang. Allava: Harnessing gpt4v-synthesized data for a lite vision-language model. arXiv preprint arXiv:2402.11684, 2024. 3
- [6] Wentong Chen, Junbo Cui, Jinyi Hu, Yujia Qin, Junjie Fang, Yue Zhao, Chongyi Wang, Jun Liu, Guirong Chen, Yupeng Huo, et al. Guicourse: From general vision language models to versatile gui agents. arXiv preprint arXiv:2406.11317,

2024. 2, 6

- [7] Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yantao Li, Jianbing Zhang, and Zhiyong Wu. Seeclick: Harnessing gui grounding for advanced visual gui agents. arXiv preprint arXiv:2401.10935, 2024. 1, 2, 3, 5, 6
- [8] Biplab Deka, Zifeng Huang, Chad Franzen, Joshua Hibschman, Daniel Afergan, Yang Li, Jeffrey Nichols, and Ranjitha Kumar. Rico: A mobile app dataset for building datadriven design applications. In Proceedings of the 30th annual ACM symposium on user interface software and technology, pages 845–854, 2017. 2
- [9] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36, 2024. 2, 3, 5, 6
- [10] Hanze Dong, Wei Xiong, Deepanshu Goyal, Yihan Zhang, Winnie Chow, Rui Pan, Shizhe Diao, Jipeng Zhang, Kashun Shum, and Tong Zhang. Raft: Reward ranked finetuning for generative foundation model alignment. arXiv preprint arXiv:2304.06767, 2023. 5
- [11] Yuning Du, Chenxia Li, Ruoyu Guo, Xiaoting Yin, Weiwei Liu, Jun Zhou, Yifan Bai, Zilin Yu, Yehua Yang, Qingqing Dang, et al. Pp-ocr: A practical ultra lightweight ocr system. arXiv preprint arXiv:2009.09941, 2020. 3
- [12] Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306,

2024. 2, 5, 1

- [13] Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. Navigating the digital world as humans do: Universal visual grounding for gui agents. arXiv preprint arXiv:2410.05243, 2024. 1, 2, 5, 6
- [14] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021. 3
- [15] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14281– 14290, 2024. 1, 2, 6
- [16] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. 3, 5, 1

Instruction: Can you make Bing the main search thingy when I look stuff up on the Internet?

[Figure 30]

[Figure 31]

Step 1: single_click(“Chrome menu button”) Step 2: single_click(“Settings option”)

[Figure 32]

[Figure 33]

Step 3: single_click(“Search Engine”) Step 4: single_click(“Change”)

[Figure 34]

[Figure 35]

Step 5: single_click(“Microsoft Bing”) Step 6: single_click(“Set as default”)

Instruction: Show side effects of Tamiflu.

[Figure 36]

[Figure 37]

Step 1: single_click(“Search bar”), write(“Tamiflu”) Step 2: single_click(“Side Effects link”)

- Figure 8. The trajectories of our STEVE-KTO-7B agent for the chrome tasks from the WinAgentArena [4] with ID bb5e4c0d-f964-439c97b6-bdb9747de3f4-wos (up) and b070486d-e161-459b-aa2b-ef442d973b92-wos (bottom). We display a simplified action for each step and plot the target UI localization results with a red bounding box in each screenshot. For high-resolution screenshots/videos, full model responses with screen analysis, multi-step planning, and python code blocks, please refer to the corresponding attachments.

Instruction: Sort files by date modified in the Documents folder.

[Figure 38]

[Figure 39]

Step 1: single_click(“Focus on file explorer”) Step 2: single_click(“View tab”)

[Figure 40]

[Figure 41]

Step 3: single_click(“the ‘Sort by’ dropdown”) Step 4: single_click(“the Date modified option”)

Instruction: Update the file explorer view settings to show hidden and system files.

[Figure 42]

[Figure 43]

Step 1: single_click(“File explorer icon”) Step 2: single_click(“View tab”)

[Figure 44]

[Figure 45]

Step 3: single_click(“the Hidden items checkbox”) Step 4: single_click(“Empty space”), DONE

- Figure 9. The trajectories of our STEVE-KTO-7B agent for the file explorer tasks from the WinAgentArena [4] with ID 7c70e16b-e14f4baa-b046-3e022b2d0305-WOS (up) and 5316686e-5688-4115-be24-052037df599f-WOS (bottom). We display a simplified action for each step and plot the target UI localization results with a red bounding box in each screenshot. For high-resolution screenshots/videos, full model responses with screen analysis, multi-step planning, and python code blocks, please refer to the corresponding attachments.

Instruction: Change my desktop background to a solid color.

|[Figure 46]|
|---|

|[Figure 47]|
|---|

Step 1: right_click(“Empty space on the desktop”) Step 2: single_click(“Personalize option”)

|[Figure 48]|
|---|

|[Figure 49]|
|---|

Step 3: single_click(“Background option”) Step 4: single_click(“Dropdown menu”)

|[Figure 50]|
|---|

|[Figure 51]|
|---|

Step 5: single_click(“Solid color option”) Step 6: single_click(“purple color option”)

Instruction: I need to turn off notification for my system in the settings.

[Figure 52]

[Figure 53]

Step 5: single_click(“Notification & action”) Step 6: single_click(“main notification switch”)

- Figure 10. The trajectories of our STEVE-KTO-7B agent for the Windows setting tasks from the WinAgentArena [4] with ID a659b26e4e31-40c1-adaf-34742b6c44ac-wos (up) and 37e10fc4-b4c5-4b02-a65c-bfae8bc51d3f-wos (bottom). Only the last two steps of the later task are shown. We display a simplified action for each step and plot the target UI localization results with a red bounding box in each screenshot. For high-resolution screenshots/videos, full model responses with screen analysis, multi-step planning, and python code blocks, please refer to the corresponding attachments.

[Figure 54]

UI-Grounding model with the GPT-4o planner: Task analysis: The folder has been successfully renamed to "Archive". The next step is to move the .docx files into it. Multi-step plan:1. Select all .docx files in the Documents folder.2. Move the selected files into the "Archive" folder. Next step rationale: Proceed by selecting the .docx files to prepare for moving them into the "Archive" folder. ```python

- computer.mouse.move("Doc01 file in the Documents folder") # Move to the first .docx file computer.mouse.single_click() # Select the first .docx file computer.keyboard.press("shift") # Press and hold Shift key
- computer.mouse.move("Doc02 file in the Documents folder") # Move to the second .docx file computer.mouse.single_click() # Select the second .docx file computer.keyboard.press("shift") # Release Shift key ```

[Figure 55]

SFT-7B Agent: Screen analysis: The "New folder" has been renamed to "Archive". The next step is to select all .docx files in the Documents folder. Multi-step plan:1. Select all .docx files in the Documents folder.2. Move the selected files into the "Archive" folder. Next step rationale: The next step is to select all .docx files in the Documents folder. ```python computer.mouse.move("<ref>Archive folder</ref><box>(214,283),(512,303)</box>") # Move to the "Archive" folder computer.mouse.single_click() # Select the "Archive" folder computer.keyboard.hotkey("ctrl", "a") # Select all files in the Documents folder ```

[Figure 56]

KTO-7B Agent: Task analysis: The task is to move all .docx files into the "Archive" folder in the Documents directory. Multi-step plan:1. Select all .docx files in the Documents folder.2. Move the selected files into the "Archive" folder. Next step rationale: The next step is to select all .docx files in the Documents folder. ```python

- computer.mouse.move("<ref>Doc01 file</ref><box>(215,292),(496,312)</box>") # Move to Doc01 file computer.mouse.single_click() # Select Doc01 computer.keyboard.keyDown("ctrl") # Hold down the Ctrl key to select multiple files
- computer.mouse.move("<ref>Doc02 file</ref><box>(215,316),(496,337)</box>") # Move to Doc02 filecomputer.mouse.single_click() # Select Doc02 computer.keyboard.keyUp("ctrl") # Release the Ctrl key ```

###### Figure 11. Comparisons of different computer-use agent models.

- [17] Raghav Kapoor, Yash Parag Butala, Melisa Russak, Jing Yu Koh, Kiran Kamble, Waseem Alshikh, and Ruslan Salakhutdinov. Omniact: A dataset and benchmark for enabling multimodal generalist autonomous agents for desktop and web. arXiv preprint arXiv:2402.17553, 2024. 2
- [18] Xin Lai, Zhuotao Tian, Yukang Chen, Senqiao Yang, Xiangru Peng, and Jiaya Jia. Step-dpo: Step-wise preference optimization for long-chain reasoning of llms. arXiv preprint arXiv:2406.18629, 2024. 3
- [19] Zhangheng Li, Keen You, Haotian Zhang, Di Feng, Harsh Agrawal, Xiujun Li, Mohana Prasad Sathya Moorthy, Jeff Nichols, Yinfei Yang, and Zhe Gan. Ferret-ui 2: Mastering universal user interface understanding across platforms. arXiv preprint arXiv:2410.18967, 2024. 2
- [20] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023. 3, 4
- [21] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 1
- [22] Yadong Lu, Jianwei Yang, Yelong Shen, and Ahmed Awadallah. Omniparser for pure vision based gui agent. arXiv preprint arXiv:2408.00203, 2024. 1, 2, 3, 5, 6
- [23] Runliang Niu, Jindong Li, Shiqi Wang, Yali Fu, Xiyu Hu, Xueyuan Leng, He Kong, Yi Chang, and Qi Wang. Screenagent: A vision language model-driven computer control agent. arXiv preprint arXiv:2402.07945, 2024. 2
- [24] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024. 2, 5
- [25] Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, and Timothy Lillicrap. Androidinthewild: A largescale dataset for android device control. Advances in Neural Information Processing Systems, 36, 2024. 2, 3, 6
- [26] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 2
- [27] Tianhao Shen, Renren Jin, Yufei Huang, Chuang Liu, Weilong Dong, Zishan Guo, Xinwei Wu, Yan Liu, and Deyi Xiong. Large language model alignment: A survey. arXiv preprint arXiv:2309.15025, 2023. 2
- [28] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/ stanford_alpaca, 2023. 4
- [29] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 1, 2, 5, 6
- [30] Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. Mathshepherd: Verify and reinforce llms step-by-step without human annotations. In Proceedings of the 62nd Annual Meeting

- of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9426–9439, 2024. 3, 4
- [31] Jason Wu, Siyan Wang, Siman Shen, Yi-Hao Peng, Jeffrey Nichols, and Jeffrey P Bigham. Webui: A dataset for enhancing visual ui understanding with web semantics. In Proceedings of the 2023 CHI Conference on Human Factors in Computing Systems, pages 1–14, 2023. 2, 3
- [32] Mulong Xie, Sidong Feng, Zhenchang Xing, Jieshan Chen, and Chunyang Chen. Uied: a hybrid tool for gui element detection. In Proceedings of the 28th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering, pages 1655– 1659, 2020. 1
- [33] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. arXiv preprint arXiv:2404.07972, 2024. 1, 2, 4
- [34] Chaoyun Zhang, Liqun Li, Shilin He, Xu Zhang, Bo Qiao, Si Qin, Minghua Ma, Yu Kang, Qingwei Lin, Saravan Rajmohan, et al. Ufo: A ui-focused agent for windows os interaction. arXiv preprint arXiv:2402.07939, 2024. 2, 5
- [35] Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. Gpt-4v (ision) is a generalist web agent, if grounded. arXiv preprint arXiv:2401.01614, 2024. 2

