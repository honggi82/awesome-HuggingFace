# arXiv:2502.14669v3[cs.CL]25Feb2025

## AlphaMaze: Enhancing Large Language Models’ Spatial Intelligence via GRPO

Alan Dao (Gia Tuan Dao)1, Dinh Bach Vu1 Menlo Research alan@menlo.ai, bach@menlo.ai 1Equal contribution.

February 20, 2025

### Abstract

Large Language Models (LLMs) have demonstrated impressive capabilities in language processing, yet they often struggle with tasks requiring genuine visual spatial reasoning. In this paper, we introduce a novel twostage training framework designed to equip standard LLMs with visual reasoning abilities for maze navigation. First, we leverage Supervised Fine-Tuning (SFT) on a curated dataset of tokenized maze representations to teach the model to predict step-by-step movement commands. Next, we apply Group Relative Policy Optimization (GRPO)—a technique used in DeepSeekR1—with a carefully crafted reward function to refine the model’s sequential decision-making and encourage emergent chain-of-thought behaviors. Experimental results on synthetically generated mazes show that while a baseline model fails to navigate the maze, the SFT-trained model achieves 86% accuracy, and further GRPO finetuning boosts accuracy to 93%. Qualitative analyses reveal that GRPO fosters more robust and self-corrective reasoning, highlighting the potential of our approach to bridge the gap between language models and visual spatial tasks. These findings offer promising implications for applications in robotics, autonomous navigation, and other domains that require integrated visual and sequential reasoning.

### 1 Introduction

The ability to reason about visual information, particularly in spatial contexts, is a hallmark of intelligent systems. From navigating physical environments to interpreting complex diagrams, visual spatial reasoning is crucial for a wide range of tasks. While Large Language Models (LLMs) have achieved impressive performance in natural language processing and code generation, their capacity for genuine visual reasoning, especially spatial understanding and sequential decision-making in visual environments, remains a significant open question [Zhang et al., 2024, Ma et al., 2024]. Current Vision-Language Models (VLMs) often excel at pattern recognition and object identification but may struggle with tasks requiring deeper spatial inference and step-by-step planning

MazeBench scores with GRPO steps

95

| | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |

MazeBench

90

85

80

0200400600800100012001400160018002000

GRPO steps

MazeBench Trendline (Linear Regression) ±1 Std Dev

Figure 1: MazeBench scores over GRPO steps with a linear regression trendline and its ±1 standard deviation bounds.

in visual domains [Ma et al., 2024]. Bridging this gap and endowing standard LLMs with robust visual reasoning capabilities is a critical step towards more versatile and human-like AI.

In this paper, we address the challenge of teaching visual spatial reasoning to a standard LLM, focusing on the task of maze navigation. We hypothesize that by providing an LLM with a tokenized visual representation of a maze, we can train it to learn step-by-step movement commands to navigate from a designated origin to a target. The core of our approach lies in a two-stage training framework. First, we employ Supervised FineTuning (SFT) to equip the LLM with the foundational skill of predicting movement tokens based on the visual maze input. Subsequently, we apply Group Relative Policy Optimization (GRPO), drawing inspiration from recent advancements in reinforcement learning for reasoning in LLMs, such as DeepSeek-R1 [Guo et al., 2025]. DeepSeek-R1 demonstrated that Reinforcement Learning (RL) can elicit emergent reasoning behaviors, including chain-of-thought, even without prior SFT. We adapt and extend these RL strategies, combined with carefully designed reward functions, to refine our model’s visual reasoning process for maze navigation.

To systematically evaluate LLM’s ability to solve maze,

we introduce MazeBench—a comprehensive benchmark on solving maze. MazeBench provides a controlled yet diverse environment that spans a range of maze sizes and complexities. By evaluating our model on MazeBench, we can rigorously measure both its maze-solving accuracy and the sophistication of its emergent reasoning behavior.

Our key contributions are as follows:

- • We present a novel training framework that combines Supervised Fine-Tuning and Group Relative Policy Optimization to enhance visual reasoning in standard LLMs, specifically for spatial tasks.
- • We empirically demonstrate that this framework, using a tokenized visual maze representation, enables an LLM to achieve improved maze navigation accuracy and exhibit emergent chain-of-thought reasoning in generating movement sequences.
- • We provide a detailed analysis of the design and impact of reward functions within the GRPO stage, highlighting their crucial role in shaping the model’s visual reasoning performance.
- • We draw comparisons with insights from state-ofthe-art reasoning models like DeepSeek-R1, both in terms of methodology and observed emergent behaviors, positioning our work within the context of current advancements in LLM reasoning.
- • We present MazeBench, a benchmark for visual maze navigation that captures a wide spectrum of spatial challenges.

### 2 Related Work

#### 2.1 Chain-of-Thought Reasoning in Language Models

Chain-of-Thought (CoT) prompting has emerged as a powerful technique to elicit complex reasoning from Large Language Models [Wei et al., 2022b]. By prompting LLMs to ”think step by step,” CoT encourages the generation of intermediate reasoning steps, leading to improved performance on tasks requiring multi-step inference. Prior research, including Wei et al. [2022b], Wei et al. [2023], Wang et al. [2023] prompting significantly enhances LLM performance on arithmetic, commonsense reasoning, and symbolic reasoning tasks. Our work builds upon the concept of CoT reasoning, aiming to induce a similar step-by-step thought process in LLMs, but within the domain of visual spatial reasoning for maze navigation.

#### 2.2 Supervised Fine-Tuning for Visual and Spatial Tasks

Supervised Fine-Tuning (SFT) is a widely adopted technique for adapting pre-trained LLMs to specific downstream tasks [Wei et al., 2022a]. By training on taskspecific datasets, SFT allows LLMs to acquire specialized skills and improve performance in targeted domains.

Jiang et al. [2024] recently highlighted the effectiveness of SFT in enhancing visual foundation models, demonstrating its utility in visual tasks. In our research, we leverage SFT as the initial stage of our training pipeline, using it to equip the LLM with the basic capability of processing tokenized visual maze inputs and predicting movement tokens. This SFT phase serves as a crucial foundation upon which we build more sophisticated reasoning through reinforcement learning.

#### 2.3 Reinforcement Learning and GRPO for Reasoning and Reward Shaping

Reinforcement Learning from Human Feedback (RLHF) and its variants have demonstrated significant efficacy in aligning Large Language Models (LLMs) with human preferences and enhancing their reasoning capabilities. However, RLHF faces substantial scalability challenges due to its resource-intensive nature and reliance on human feedback data. As an alternative approach, recent methodologies like Group Relative Policy Optimization (GRPO)[Kwon et al., 2023a] and Self-Play fIne-tuNing (SPIN) leverage self-play mechanisms, where models autonomously generate training signals and iteratively improve through self-competition Chen et al. [2024]. These self-play approaches show promise in achieving humanlevel performance without the need for extensive human feedback, potentially offering a more scalable solution to the alignment challenge. GRPO, as described by Shao et al. [2024] and implemented in DeepSeek-R1 [Guo et al., 2025], offers a computationally efficient approach to reinforcement learning by estimating advantages based on group scores, eliminating the need for a separate critic network. Reward function design is paramount in RLHF and GRPO, as it directly guides the model’s learning process. Carefully crafted reward functions can incentivize desired behaviors and shape the model’s policy towards optimal performance. Our work draws inspiration from the reward shaping strategies used in DeepSeek-R1 and adapts them to the context of visual maze navigation, designing reward components to encourage accuracy, valid movement sequences, and proper output formatting.

#### 2.4 DeepSeek-R1 and Emergent Reasoning through RL

The DeepSeek-R1 model [Guo et al., 2025] represents a significant advancement in using reinforcement learning to elicit sophisticated reasoning capabilities in LLMs. A key finding of DeepSeek-R1 is the demonstration that pure RL, specifically GRPO, can lead to the emergent development of chain-of-thought reasoning and even ”aha moments,” where the model re-evaluates previous steps and corrects its reasoning process. Furthermore, DeepSeek-R1 highlights the benefits of a multistage training pipeline, combining initial RL training with subsequent supervised fine-tuning to refine language coherence and readability. We directly adapt the GRPO optimization strategy and multi-stage training insights from DeepSeek-R1 to our visual maze navigation task.

[Figure 1]

Figure 2: Visual of the Example Maze

We hypothesize that similar RL techniques can drive the emergence of visual spatial reasoning in standard LLMs, enabling them to solve mazes through a step-by-step, selfcorrective process.

#### 2.5 Visual Reasoning and Maze Solving in AI

Maze solving has long been a benchmark task in Artificial Intelligence, serving as a testbed for various problem-solving and search algorithms [Janamian and Alam, 2023]. Traditional approaches include graph search algorithms like Depth-First Search, Breadth-First Search, and A* [Lester, 2014-2024]. More recently, AI techniques, particularly reinforcement learning and neural networks, have been applied to maze navigation [Zafrany, 2020]. While Chain-of-Thought (CoT) prompting has significantly enhanced complex reasoning capabilities in Large Language Models (LLMs) and Multimodal LLMs, it shows limitations in complex spatial reasoning tasks. Recent work by Microsoft introduces Multimodal Visualization-of-Thought (MVoT), which enables models to generate visual representations during their reasoning process, similar to human visual thinking [Li et al., 2025]. This breakthrough demonstrates the potential of combining verbal and visual reasoning in AI systems.

Our research builds upon these advances, focusing on teaching visual maze reasoning to standard language models through a tokenized visual representation and a combination of SFT and GRPO. This approach differs from traditional maze solvers by leveraging the inherent reasoning capabilities of LLMs and adapting them to process and reason about visual spatial information. Furthermore, research in neural-symbolic visual reasoning [Mao et al., 2023] explores combining neural networks with symbolic AI for visual tasks, offering a complementary perspective on integrating reasoning and visual processing.

3 Methodology

#### 3.1 Tokenized Visual Maze Representation

To enable the LLM to process maze information visually, we designed a tokenized input format that represents the maze grid, walls, origin, and target locations. Each cell in the maze is represented by a coordinate token

<|row-col|>, e.g., <|0-0|> for the top-left cell. Wall information for each cell is encoded using tokens such as <|no wall|>, <|up wall|>, <|up down wall|>, <|up down left right wall|>, .... The origin and target locations are marked with <|origin|> and <|target|> tokens, respectively. Empty spaces within the maze representation are filled with <|blank|> tokens for consistent grid structure. This tokenization scheme provides a visual representation by explicitly encoding the spatial relationships between cells and the presence of walls, allowing the LLM to “see” the maze structure in a symbolic, tokenized form.

Example Maze Tokenization 2:

|<|0-0|><|up_left_wall|><|blank|><|0-1|><|<br><br>up_down_wall|><|blank|><|0-2|><|<br><br>up_down_wall|><|blank|><|0-3|><| up_down_right_wall|><|blank|><|0-4|><| up_left_right_wall|><|blank|><br><br><br><|1-0|><|down_left_wall|><|blank|><|1-1|><| up_down_wall|><|blank|><|1-2|><| up_right_wall|><|blank|><|1-3|><| up_down_left_wall|><|blank|><|1-4|><| right_wall|><|blank|><br><br><|2-0|><|up_left_wall|><|blank|><|2-1|><| up_down_right_wall|><|origin|><|2-2|><| left_right_wall|><|blank|><|2-3|><| up_left_wall|><|blank|><|2-4|><| right_wall|><|blank|><br><br><|3-0|><|left_wall|><|blank|><|3-1|><| up_right_wall|><|blank|><|3-2|><| down_left_wall|><|blank|><|3-3|><| down_right_wall|><|blank|><|3-4|><| left_right_wall|><|blank|><br><br><|4-0|><|down_left_right_wall|><|blank<br><br><br>|><|4-1|><|down_left_wall|><|blank<br><br>|><|4-2|><|up_down_wall|><|target<br><br>|><|4-3|><|up_down_wall|><|blank<br><br>|><|4-4|><|down_right_wall|><|blank|><br><br><br>|
|---|

3.2 Baseline Models

To establish performance benchmarks for our approach, we employed three distinct baseline models, leveraging the DeepSeek-R1 [Guo et al., 2025] Distill-Qwen family of language models. We evaluate two distilled models: DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-DistillQwen-1.5B. Additionally, a Direct Prediction baseline was established using a Supervised Fine-Tuning (SFT) approach on the DeepSeek-R1-Distill-Qwen-1.5B architecture. This model was trained to directly predict the complete sequence of movement tokens representing the solution path through a given maze. The training objective was the minimization of cross-entropy loss between the predicted token sequence and the ground truth solution. This baseline assesses the performance of a standard language model trained to generate complete solutions without intermediate reasoning steps or reinforcement learning techniques.

We include these three baselines to provide a comprehensive comparison, examining the influence of model

[Figure 2]

Figure 3: Visualization of AlphaMaze’s step-by-step reasoning process while solving a maze.

size (7B vs. 1.5B) and the effectiveness of direct prediction versus our proposed step-by-step and reinforcement learning approaches. The subsequent sections will primarily focus on the customized direct prediction model and its enhancements through SFT for step-by-step reasoning and GRPO.

- 3.3 Supervised Fine-Tuning (SFT) for Step-byStep Reasoning

For the SFT stage, we curated a training dataset. Mazes were synthetically generated with fixed sizes (5x5) and varied complexity level. The Qwen 1.5B SFT model was then trained on this dataset. The training objective was to predict the next movement token at each step, conditioned on the maze input and the preceding movement tokens in the sequence as visually illustrated in Figure

- 3. This step-by-step prediction approach was designed to encourage the model to learn sequential reasoning for maze navigation.
- 3.4 Group Relative Policy Optimization (GRPO) for Enhanced Reasoning

Following SFT, we applied Group Relative Policy Optimization (GRPO) to further enhance the model’s mazesolving capabilities and encourage more robust reasoning. The GRPO training utilized a smaller set of data than SFT state. We designed a reward function 3 components.

Correctness Reward (+0.2 per solution step): This reward is scaled according to the number of steps in the maze solution. Each valid movement step adds 0.2 points to the total score. For example, a solution requiring 4 steps earns a reward of 0.2 × 4 = 0.8 points, incentivizing both accuracy and efficiency in navigation.

Integrity Reward (+0.5): This reward is given for each valid movement token (<|up|>, <|down|>, <|left|>, <|right|>) in the predicted sequence, encouraging the generation of meaningful and valid movement steps.

Thinking Reward (+0.25): This reward is given for correctly using the <think> tag in the output, ensuring completeness and consistency in the reasoning format.

These reward components were weighted to prioritize correctness while also encouraging valid movement sequences and proper reasoning formatting with <think> tag. We adapted the Group Relative Policy Optimization (GRPO) algorithm, as employed in DeepSeek-R1 [Guo et al., 2025], to perform reinforcement learning. GRPO estimates advantages based on relative group scores, offering computational efficiency compared to critic-based methods.

#### 3.5 Training Procedure and Pipeline

Our training pipeline consisted of two stages. First, Supervised Fine-Tuning (SFT) was performed on the Qwen 1.5B model using a curated maze dataset for 10 epochs to learn step-by-step movement prediction for maze navigation. This phase established a strong initial policy, ensuring that the model could effectively interpret and respond to sequential movement tasks.

Following SFT, Group Relative Policy Optimization (GRPO) was applied to refine the model’s performance. The SFT-trained model was further fine-tuned using LoRA Hu et al. [2021] with the GRPO method, implemented in Unsloth Daniel Han and team [2023] with VLLM Kwon et al. [2023b] for efficient inference. A carefully designed reward function guided the optimization process, and model checkpoints were saved every 200 steps to track improvements.

This two-stage pipeline mirrors the multi-stage training approach employed in DeepSeek-R1 [Guo et al., 2025], where initial RL training is followed by supervised finetuning for refinement. In our case, SFT provided a robust starting point for reinforcement learning (RL), allowing GRPO to focus on refining reasoning capabilities and enhancing task-specific performance.

4 Experiments and Results

#### 4.1 Dataset Details

The dataset is constructed through a multi-stage process involving generation, refinement, and augmentation. The process begins with the creation of a large initial pool of 530,000 synthetic mazes. These mazes are generated using the maze-dataset framework Ivanitskiy et al. [2023], which employs a randomized depth-first search algorithm. This algorithm ensures that every generated maze has a guaranteed solution path connecting the designated origin and target locations. Further details of algorithm used can be found at Appendix 1. All mazes within the dataset have a fixed size of 5x5 grids. From this extensive initial pool, a subset of 30,000 mazes is randomly selected and reserved as a held-out test set. This separation guarantees that the training and evaluation data are entirely distinct, preventing data leakage and enabling a robust assessment of model generalization.

The remaining 500,000 mazes form the basis for the various training datasets used in this work. This pool of mazes undergoes a multi-stage processing and augmentation procedure to create datasets specifically tailored for different training objectives.

Reset Dataset Creation: A significant portion of the training data focuses on teaching the model to recover from errors. To this end, a ”reset” dataset is created. This dataset is generated by algorithmically producing incorrect solution paths for the mazes. These incorrect paths are designed to be plausible but ultimately unsuccessful, either leading to dead ends or deviating from the correct solution. Importantly, they adhere to constraints: they do not revisit locations already visited within the incorrect attempt, and they avoid portions of the known correct solution path.

Associated with each incorrect path, a textual ”RESET” message is generated, simulating the feedback a system might provide upon encountering an error. The content of this message depends on whether the incorrect path terminates at a dead end (three surrounding walls) or simply deviates from the wrong route. These incorrect paths, along with their reset messages and the correct solution’s Chain-of-Thought (COT) reasoning, are combined. This process results in approximately 400,000 training examples where the model is presented with scenarios of initial failure(s) followed by a successful attempt after a ”reset.” The intent is to train the model to recognize incorrect trajectories and adapt its strategy. Illustrative examples of reset samples are provided in Appendix.

SFT Training Data Construction: The final Supervised Fine-Tuning (SFT) training dataset is a balanced combination of ”straight success” examples and ”retry” examples:

- • Straight Success Data (250,000 mazes): This portion consists of mazes where the model is expected to generate the correct solution path on the first attempt, without any resets.
- • Retry Data (250,000 mazes): This portion is drawn from the ”reset” dataset described above, providing examples where the model learns from incorrect attempts and subsequent resets. This combined 500,000-maze SFT set, encompassing success and error recovery, enables robust learning.

GRPO Training Data: The remaining 150,000 mazes from the original ”straight success” data pool are used to create a dataset for GRPO stage.

#### 4.2 MazeBench

To rigorously evaluate the spatial reasoning and planning capabilities of large language models (LLMs), we introduce MazeBench, a novel benchmark consisting of a curated collection of 100 maze-solving challenges. While existing benchmarks often assess logical reasoning or commonsense knowledge, MazeBench specifically targets the ability of LLMs to understand spatial relationships, plan multi-step paths, and execute sequential actions within a

constrained environment. This capacity is crucial for applications ranging from robotics and navigation to game playing and virtual agent control.

MazeBench is a collection of 100 unique mazes, randomly selected from a larger test set containing 30,000 mazes. It is designed to evaluate the performance of large language models (LLMs) by categorizing mazes into different difficulty levels. Each maze requires the model to determine an optimal path from the starting point to the target, with difficulty primarily based on the number of steps needed to reach the goal.

The benchmark is structured into three levels: Easy, Medium, and Hard, ensuring a progressive assessment of an LLM’s pathfinding and problem-solving abilities. The components are described in Table 1.

Table 1: Maze Configuration by Difficulty Level

|Category|Number of Mazes<br><br>|Steps|
|---|---|---|
|Easy Medium Hard<br><br>|50 40 10|1 – 4 5 – 8 9 – 13<br><br>|
|Total|100|1 – 13|

The Easy category consists of 50 mazes, each requiring between 1 and 4 steps to solve. These simpler mazes establish a baseline for evaluating fundamental navigation skills.

The Medium category includes 40 mazes that demand solution paths of 5 to 8 steps. These mazes introduce a higher level of complexity, requiring more advanced planning and spatial reasoning. Successfully solving them indicates an LLM’s ability to manage moderately intricate environments.

The Hard category comprises 10 mazes, each necessitating 9 to 13 steps to reach the target. These mazes present the greatest challenge, testing an LLM’s capacity to handle long-range dependencies and navigate complex spatial structures. Performance on this level reflects the model’s ability to process and reason over extended solution paths.

As mentioned previously, the mazes are presented to the LLM in a tokenized input format; the full details of this representation, including examples, are provided in Section 3.1. The LLM is expected to produce sequence of movement tokens. During evaluation, we will parse the LLM’s output to extract these tokens. The order of these tokens is crucial. The presence of extraneous characters, whitespace, or other tokens will not automatically invalidate the solution, provided that the correct sequence of movement tokens can be extracted. A solution is considered incorrect if the extracted sequence of movement tokens does not lead to the target or leads to an invalid state (e.g., attempting to move into a wall) is considered incorrect. The evaluation metric is the success rate: the percentage of mazes solved correctly.

- 4.3 Quantitative Results

- 4.3.1 Model Performance on MazeBench

As shown in Table 2, the baseline model, trained for direct path prediction without explicit reasoning, achieved 0% accuracy on MazeBench. This highlights the necessity of step-by-step reasoning for the task. The SFT-only model reached a baseline of 86.0%, demonstrating the effectiveness of supervised fine-tuning for learning step-bystep maze navigation. Further enhancement with GRPO led to significant improvement, reaching 93.0% after 1600 steps of GRPO training.

Table 2: Maze Solving Accuracy on MazeBench

|Model|SFT<br><br>|GRPO<br><br>|Score|
|---|---|---|---|
|Baseline-1.5B Baseline-7B Baseline-1.5B (SFT) AlphaMaze-SFT AlphaMaze|✗ ✗ ✓ ✓ ✓<br><br>|✗ ✗ ✗ ✗ ✓<br><br>|0.0 0.0 0.0 86.0 93.0|

- 4.3.2 Model Evolution During GRPO

Figure 1 displays the MazeBench scores (blue crosses) over GRPO steps along with a linear regression trendline (red dashed line) and its ±1 standard deviation bounds. The steady increase in the trendline indicates that GRPO effectively guides the model towards improved maze-solving policies.

#### 4.4 Qualitative Results

Qualitative analysis of model outputs revealed notable differences in reasoning behavior. The baseline model often produced nonsensical or incomplete movement sequences, frequently failing to reach the target and exhibiting ”hallucinations” by predicting movements invalid within the maze structure. The AlphaMaze-SFT model demonstrated improved coherence and step-by-step progression, but still struggled with longer or more complex mazes, sometimes becoming trapped in loops or making incorrect turns in later stages of the solution path.

In contrast, the AlphaMaze-SFT+GRPO model exhibited the most sophisticated reasoning. In many instances, emergent chain-of-thought patterns were observed, with AlphaMaze (two-stage) appearing to explicitly consider wall constraints and spatial relationships at each step before predicting the next movement. Furthermore, outputs occasionally displayed instances reminiscent of the ”aha moments” reported in prior work on DeepSeek-R1. For example, in some complex mazes, AlphaMaze (two-stage) would initially begin along one path, then appear to ”re-evaluate” its trajectory mid-sequence, correcting its course to find a more efficient or correct solution. Error analysis indicated that AlphaMaze (twostage) made fewer invalid moves and was more robust to long-context reasoning challenges compared to the AlphaMaze-SFT model. However, limitations remained,

particularly in mazes requiring backtracking or complex spatial planning beyond the immediate next step.

### 5 Discussion

#### 5.1 Analysis of GRPO’s Impact on Visual Maze Reasoning

Our results clearly demonstrate the incremental benefit of Group Relative Policy Optimization (GRPO) in enhancing visual maze reasoning within Large Language Models. While Supervised Fine-Tuning (SFT) establishes a strong foundation, enabling the model to achieve a 86% accuracy on MazeBench, the application of GRPO further elevates performance to 93% after 1600 training steps. This improvement, albeit seemingly modest in percentage points, is significant considering the already strong baseline established by SFT. It suggests that GRPO is effectively refining the model’s policy, leading to more robust and accurate maze navigation.

The qualitative analysis provides further insight into the nature of this improvement. The AlphaMazeSFT+GRPO model exhibited more pronounced chainof-thought reasoning patterns and instances of selfcorrection, indicating that GRPO is not merely finetuning the existing SFT policy, but rather encouraging more sophisticated reasoning processes. The reward function, designed to incentivize correctness, valid movements, and structured output, likely plays a crucial role in shaping this behavior. By rewarding successful navigation and penalizing invalid steps, GRPO encourages the model to learn more deliberate and considered movement strategies.

#### 5.2 Comparison with DeepSeek-R1 and RL for Reasoning

It is important to note that the base DeepSeek-R1 model, when operating with an extremely long context window, demonstrates emergent visual reasoning capabilities. However, our experiments reveal that the distilled variants (DeepSeek-R1 Distill-Qwen models) do not carry over these spatial reasoning abilities, as evidenced by their 0% accuracy on MazeBench. This suggests that the distillation process into Qwen or other smaller models is insufficient to preserve the emergent ability of visual spatial reasoning observed in the base model.

In contrast, our two-stage training approach—combining Supervised Fine-Tuning (SFT) to establish foundational step-by-step reasoning with Group Relative Policy Optimization (GRPO) for further refinement—effectively equips the distilled model with robust visual maze-solving skills. Even with only 1600 GRPO steps, the model achieves a notable improvement, reaching 93% accuracy and exhibiting clear chain-ofthought behaviors along with self-correction during navigation.

These findings underscore the necessity of specialized training to recover or enhance spatial reasoning in distilled models, highlighting that while the base DeepSeek-

R1 model is capable of visual reasoning with sufficient context, additional training stages are crucial to maintain or induce this capability in smaller, distilled variants.

#### 5.3 Limitations

Despite the encouraging results, our study is not without limitations. Firstly, the performance gain from GRPO, while statistically significant, is small (7% accuracy improvement in our reported experiment). Further investigation is needed to explore whether more extensive GRPO training, or modifications to the reward function, could lead to more substantial performance gains. It is possible that the current reward function, while effective, could be further optimized to better incentivize more complex reasoning strategies, such as backtracking or more proactive exploration of alternative paths.

Secondly, our evaluation, while including qualitative analysis, is primarily based on maze-solving accuracy. This metric, while important, provides a somewhat limited view of the model’s reasoning capabilities. Future work could benefit from more nuanced evaluation metrics that assess the efficiency of the generated paths, the robustness of the model to maze complexity variations, and the interpretability of the model’s internal reasoning process. Furthermore, while we observed qualitative signs of chain-of-thought reasoning, a more rigorous analysis, perhaps using techniques from interpretability research, is needed to definitively characterize the nature and depth of the model’s reasoning process.

Finally, our experiments are limited to synthetically generated mazes. While these mazes were designed to vary in size and complexity, they may not fully capture the intricacies and variability of real-world visual spatial reasoning tasks. Future research should explore the generalizability of our approach to more diverse and ecologically valid visual environments and tasks.

### 6 Conclusion

This paper introduced AlphaMaze, a novel approach to enhance Large Language Models’ spatial intelligence, focusing on maze navigation. We demonstrated the efficacy of a two-stage training framework, leveraging Supervised Fine-Tuning (SFT) followed by Group Relative Policy Optimization (GRPO). While initial pre-trained LLMs exhibited 0% accuracy on MazeBench, highlighting the need for task-specific adaptation, our approach successfully imbued a distilled LLM with robust spatial reasoning capabilities. SFT provided a crucial foundation by teaching step-by-step movement prediction from tokenized maze inputs, reaching 86% accuracy. This underscores the importance of structured input and targeted training for LLMs to effectively engage with visual spatial information.

Crucially, we adapted and applied the two-stage training methodology pioneered by DeepSeek-R1, demonstrating its generalizability beyond language-centric reasoning tasks. Following SFT, GRPO fine-tuning further elevated

performance to 93% on MazeBench after 1600 training steps, showcasing the power of reinforcement learning to refine reasoning processes in a novel domain. Qualitative analysis revealed that GRPO fostered more sophisticated and self-corrective reasoning strategies, including emergent chain-of-thought patterns, mirroring observations in DeepSeek-R1 and suggesting a common mechanism for enhanced reasoning through RL.

Our work contributes to the broader effort of expanding LLMs’ reasoning abilities beyond natural language, demonstrating the potential of a two-stage approach for visually grounded tasks. The success of GRPO, inspired by DeepSeek-R1’s advancements in language reasoning, highlights the transferability of these techniques to spatial domains. This suggests that carefully designed reinforcement learning, following an initial phase of supervised task learning, can be a powerful method to unlock and refine sophisticated reasoning capabilities in LLMs across diverse problem spaces. The implications extend beyond maze navigation to a wide array of applications demanding spatial understanding and sequential decisionmaking.

Future research will focus on further validating this two-stage GRPO approach across various reasoning domains beyond spatial tasks, exploring its potential to enhance LLMs’ capabilities in areas such as symbolic reasoning, logical deduction, and planning. Investigating the optimal configurations of SFT and GRPO stages, diversifying training data to encompass richer and more complex reasoning scenarios, and developing more refined reward functions tailored to different reasoning challenges are critical next steps. By pursuing these directions, we aim to establish the broader applicability of this two-stage training paradigm for imbuing standard LLMs with robust and versatile reasoning abilities, paving the way for more capable and generally intelligent language models.

### References

Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning converts weak language models to strong language models, 2024. URL https://arxiv.org/abs/2401.01335.

Michael Han Daniel Han and Unsloth team. Unsloth, 2023. URL http://github.com/unslothai/unsloth.

Duyu Guo, Guoxin Xu, Yuchen Chen, Chen Tang, and Others. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. URL https://arxiv.org/ abs/2501.12948.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. URL https://arxiv.org/abs/2106. 09685.

Michael Igorevich Ivanitskiy, Rusheb Shah, Alex F. Spies, Tilman R¨uker, Dan Valentine, Can Rager, Lu-

cia Quirke, Chris Mathwin, Guillaume Corlouer, Cecilia Diniz Behn, and Samy Wu Fung. A configurable library for generating and manipulating maze datasets, 2023. URL https://arxiv.org/abs/2309.10498.

Saba Janamian and MD Sahabul Alam. Maze solver robot using a* algorithm, 2023. URL https://scholarworks.calstate.edu/concern/ theses/0c483r787. ScholarWorks@CSUN.

Xiaohu Jiang, Yixiao Ge, Yuying Ge, Dachuan Shi, Chun Yuan, and Ying Shan. Supervised fine-tuning in turn improves visual foundation models, 2024. URL https: //arxiv.org/abs/2401.10222.

Minae Kwon, Sang Michael Xie, Kalesha Bullard, and Dorsa Sadigh. Reward design with language models, 2023a. URL https://arxiv.org/abs/2303.00001.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023b.

Patrick Lester. Pathfinding algorithms, 2014-2024. URL https://www.redblobgames.com/pathfinding/. Red Blob Games.

Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners, 2022a. URL https: //arxiv.org/abs/2109.01652.

Jason Wei, Denny Zhou, Quoc Le, Denny Zhou, Quoc Le, and Others. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35: 24824–24837, 2022b. URL https://proceedings. neurips.cc/paper_files/paper/2022/hash/ 9d8fc0533c2250385321d99c6a3f2f2c-Abstract-Conference. html.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models, 2023. URL https: //arxiv.org/abs/2201.11903.

Samy Zafrany. Deep reinforcement learning for maze solving, 2020. URL https://www.samyzaf.com/ML/rl/ qmaze.html. samyzaf.com.

Jingyi Zhang, Jiaxing Huang, Sheng Jin, and Shijian Lu. Vision-language models for vision tasks: A survey,

2024. URL https://arxiv.org/abs/2304.00685.

Chengzu Li, Wenshan Wu, Huanyu Zhang, Yan Xia, Shaoguang Mao, Li Dong, Ivan Vuli´c, and Furu Wei. Imagine while reasoning in space: Multimodal visualization-of-thought, 2025. URL https://arxiv. org/abs/2501.07542.

Yueen Ma, Zixing Song, Yuzheng Zhuang, Jianye Hao, and Irwin King. A survey on vision-language-action models for embodied ai, 2024. URL https://arxiv. org/abs/2405.14093.

Jiajun Mao, Chuang Gan, Fan Zhang, and Others. Neural-symbolic visual reasoning: A survey. 2023. URL https://arxiv.org/abs/2302.07200.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/ 2402.03300.

Boshi Wang, Sewon Min, Xiang Deng, Jiaming Shen, You Wu, Luke Zettlemoyer, and Huan Sun. Towards understanding chain-of-thought prompting: An empirical study of what matters. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2717–2739, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/ 2023.acl-long.153. URL https://aclanthology.org/ 2023.acl-long.153/.

### A Algorithm

This appendix details the algorithm used to generate the maze reasoning dataset with reset demonstrations. The algorithm processes a base dataset of maze navigation problems and augments it with demonstration of incorrect attempts followed by resets and correct solutions.

- Algorithm 1 Maze Reasoning Reset Data Generation - Main Process Require: Base dataset D containing maze problems with:

- 1: - Adjacency list representation of 5 × 5 maze grid
- 2: - Origin and target coordinates
- 3: - Correct solution path

Ensure: Augmented dataset with reset demonstrations

- 4: Initialize empty datasets D1 and D2
- 5: for all example e ∈ D do
- 6: Extract adjacency list A, origin O, target T, and path P from e
- 7: Count walls W around origin O
- 8: if W = 1 then
- 9: Add e to D1
- 10: Call ProcessOrder1(e) {See Algorithm 2}
- 11: else if W = 2 then
- 12: Add e to D2
- 13: Call ProcessOrder2(e) {See Algorithm 3}
- 14: end if
- 15: end for
- 16: Combine processed examples from D1 and D2 into the final dataset

- Algorithm 2 Order-1 Processing (1 wall at origin)

- 1: Procedure ProcessOrder1(example)
- 2: WP ← ∅ {Initialize wrong paths set}
- 3: for all adjacent node N to origin O do
- 4: if N ∈/ correct path P then
- 5: for n steps from max n steps down to 1 do

- 6: Attempt to extend path from N until a dead end or n steps are reached.

- 7: if path length = n steps or a dead end is reached then

- 8: WP ← WP ∪ {path}
- 9: break
- 10: end if
- 11: end for
- 12: end if
- 13: end for
- 14: for all path p ∈ WP do
- 15: Generate chain-of-thought steps for path p.
- 16: Add “Heading in wrong direction” message.
- 17: Add RESET marker.
- 18: end for
- 19: Append original correct solution (path P).
- 20: Format as conversation pairs.
- 21: End Procedure

- Algorithm 3 Order-2 Processing (2 walls at origin)

- 1: Procedure ProcessOrder2(example)
- 2: for n steps from max n steps down to 1 do

- 3: Generate wrong path WP of length n steps starting from O.

- 4: if a valid path WP is found then
- 5: Generate chain-of-thought for WP.
- 6: if WP ends at a dead end (3 walls) then
- 7: Add “Hit a dead end” message.
- 8: else
- 9: Add “Heading in wrong direction” message.
- 10: end if
- 11: Add RESET marker.
- 12: break
- 13: end if
- 14: end for
- 15: Append original correct solution (path P).
- 16: Format as conversation pairs.
- 17: End Procedure

