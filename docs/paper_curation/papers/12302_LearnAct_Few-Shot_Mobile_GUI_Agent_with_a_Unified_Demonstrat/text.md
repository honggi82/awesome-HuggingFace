# arXiv:2504.13805v1[cs.HC]18Apr2025

## LearnAct: Few-Shot Mobile GUI Agent with a Unified Demonstration Benchmark

Guangyi Liu†

Pengxiang Zhao†

Liang Liu‡

Zhejiang University Hangzhou, China

Zhejiang University Hangzhou, China

vivo AI Lab Hangzhou, China

Zhiming Chen

vivo AI Lab Hangzhou, China

Yuxiang Chai

vivo AI Lab Hangzhou, China

Shuai Ren

vivo AI Lab ShenZhen, China

Hao Wang

vivo AI Lab ShenZhen, China

Shibo He

Zhejiang University Hangzhou, China

Wenchao Meng

Zhejiang University Hangzhou, China wmengzju@zju.edu.cn

[Figure 1]

Figure 1: The LearnAct Framework and LearnGUI Benchmark focus on addressing the long-tail challenges in mobile GUI agent performance through demonstration-based learning. From rule-based automation to LLM-powered agents, mobile GUI automation has evolved significantly, yet still struggles with long-tail scenarios due to interface diversity. Our LearnAct framework introduces demonstrationbased learning to effectively handle these challenges, outperforming existing methods in both offline and online evaluations.

### ABSTRACT

human demonstrations. We further develop LearnAct, a sophisticated multi-agent framework that automatically extracts knowledge from demonstrations to enhance task completion. This framework integrates three specialized agents: DemoParser for knowledge extraction, KnowSeeker for relevant knowledge retrieval, and ActExecutor for demonstration-enhanced task execution. Our experimental results show significant performance gains in both offline and online evaluations. In offline assessments, a single demonstration improves model performance, increasing Gemini-1.5-Pro’s accuracy from 19.3% to 51.7%. In online evaluations, our framework enhances UI-TARS-7B-SFT’s task success rate from 18.1% to 32.8%. LearnAct framework and LearnGUI benchmark establish demonstrationbased learning as a promising direction for more adaptable, personalized, and deployable mobile GUI agents. The project resources are available at https://lgy0404.github.io/LearnAct.

Mobile GUI agents show promise in automating tasks but face generalization challenges in diverse real-world scenarios. Traditional approaches using pre-training or fine-tuning with massive datasets struggle with the diversity of mobile applications and user-specific tasks. We propose enhancing mobile GUI agent capabilities through human demonstrations, focusing on improving performance in unseen scenarios rather than pursuing universal generalization through larger datasets. To realize this paradigm, we introduce LearnGUI, the first comprehensive dataset specifically designed for studying demonstration-based learning in mobile GUI agents. It comprises 2,252 offline tasks and 101 online tasks with high-quality

† Equal Contribution, ‡ Project Lead, Corresponding Author.

to perform effectively. The prevailing approaches to building modern mobile GUI agents rely on either the inherent capabilities of general-purpose LLMs [18, 20, 28, 29, 34, 36, 37, 46] or fine-tuning with large volumes of data [11, 16, 41, 48]. However, these methods face fundamental limitations when confronted with diverse realworld usage scenarios. As of 2025, billions of users interact with 1.68 million applications on Google Play alone [17], each with unique task requirements and UI layouts [32, 43]. Pre-training or finetuning datasets cannot feasibly cover this immense variety, leading to poor performance in unseen scenarios and hindering the widespread adoption of mobile GUI agents [14], as illustrated in Figure 1 (left side). Traditional approaches simply cannot cover the entire spectrum of possible interactions and user-specific requirements across this heterogeneous landscape.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Support Task: Please check the temperature in the living room for me and adjust the windows and air conditioner to a suitable state. Query Task: Please check the humidity in the bedroom for me and adjust the humidifier and windows to a suitable state.

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

To address these limitations, we propose a novel paradigm that enhances mobile GUI agent capabilities through few-shot demonstration learning. Unlike traditional approaches that either lack flexibility or require massive datasets, our demonstrationbased approach achieves both robustness and personalization by learning from a small number of user-provided examples. We recognize that mobile users have unique, repetitive tasks with inherent variability—such as smart home control with dynamic configurations, health monitoring with personalized parameters, or enterprise software with company-specific layouts. These scenarios combine stable patterns with variable elements, creating a "personalization gap" that pre-trained models cannot bridge. By leveraging user-specific demonstrations, our approach enables personalized assistants that learn both consistent patterns and adaptation strategies, acquiring task-specific knowledge impossible to cover in general training datasets. This personalization allows mobile GUI agents to overcome performance bottlenecks and provide truly helpful automation for the tasks users most want to delegate.

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

- Figure 2: A toy example for demonstration learning on mobile GUI Agent. We build a benchmark named LearnGUI for demonstration learning on Mobile GUI Agent, which provides different few-shot task combinations and offers multi-dimensional metrics including task similarity, UI similarity, and action similarity between support tasks and query tasks.

### 1 INTRODUCTION

Mobile device automation has evolved significantly over time, from simple rule-based scripts to sophisticated AI-powered agents [17, 32, 38, 43]. Traditional automation approaches like Robotic Process Automation (RPA) [1] and rule-based shortcuts [10, 13] relied on predefined scripts to execute repetitive tasks, but they struggled with dynamic interfaces, required frequent maintenance when apps updated, and lacked understanding of complex user intentions.

To fill the gap in high-quality demonstration data, we introduce LearnGUI, the first dataset specifically designed to research and evaluate mobile GUI agents’ ability to learn from few-shot demonstrations. Built upon AMEX [5] and AndroidWorld [23], LearnGUI comprises 2,252 offline few-shot tasks and 101 online tasks with high-quality human demonstrations. This dataset enables systematic research into demonstration-based learning for mobile GUI agents. A toy example for LearnGUI is shown in Figure 2.

More recently, mobile Graphical User Interface (GUI) agents have emerged as a transformative technology with the potential to revolutionize how humans interact with mobile devices. These agents leverage Large Language Models (LLMs) to autonomously complete human tasks through environmental interaction [6, 18, 20, 28, 29, 33, 36, 37, 46]. They perceive phone states of mobile phone by observing screens (through screenshots or application UI trees) and generate actions (such as CLICK, TYPE, SWIPE, PRESS_BACK, PRESS_HOME, and PRESS_ENTER) that are executed via the phone user interface [17,32,38,43]. By harnessing the powerful perception and reasoning capabilities of LLMs, mobile GUI agents have the potential to fundamentally change how people interact with their mobile devices, bringing to life the "J.A.R.V.I.S." effect seen in science fiction.

Furthermore, we present LearnAct, a multi-agent framework that automatically understands human demonstrations, generates instructional knowledge, and uses this knowledge to assist mobile GUI agents in reasoning about unseen scenarios. LearnAct consists of three specialized agents: (1) DemoParser, a knowledge generation agent that extracts usable knowledge from demonstration trajectories to form a knowledge base; (2) KnowSeeker, a knowledge retrieval agent that searches the knowledge base for demonstration knowledge relevant to the current task; and (3) ActExecutor, a task execution agent that combines user instructions, real-time GUI environment, and retrieved demonstration knowledge to perform tasks effectively.

Despite these promising advances, mobile GUI agents continue to face significant challenges in real-world deployment scenarios. The immense diversity of mobile applications and user interfaces creates pervasive long-tail scenarios where current agents struggle

Our experimental results decisively validate the effectiveness of demonstration-based learning for mobile GUI agents, as shown in Figure 1 (right side). In offline evaluations, a single demonstration dramatically improves model performance across diverse scenarios,

Table 1: Comparison of different datasets and environments for benchmarking Mobile GUI agents. Column definitions: # Inst. (number of instructions), # Apps (number of applications), # Step (average steps per task), Env. (supports environment interactions), HL (has high-level instructions), LL (has low-level instructions), GT (provides ground truth trajectories), FS (supports few-shot learning).

Dataset # Inst. # Apps # Step Env. HL LL GT FS

PixelHelp [15] 187 4 4.2 ✗ ✓ ✗ ✓ ✗ MoTIF [4] 276 125 4.5 ✗ ✓ ✓ ✓ ✗ UIBert [3] 16,660 - 1 ✗ ✗ ✓ ✓ ✗ UGIF [27] 523 12 6.3 ✗ ✓ ✓ ✓ ✗ AITW [24] 30,378 357 6.5 ✗ ✓ ✗ ✓ ✗ AITZ [45] 2,504 70 7.5 ✗ ✓ ✓ ✓ ✗ AndroidControl [14] 15,283 833 4.8 ✗ ✓ ✓ ✓ ✗ AMEX [5] 2,946 110 12.8 ✗ ✓ ✗ ✓ ✗ MobileAgentBench [30] 100 10 - ✗ ✓ ✗ ✗ ✗ AppAgent [44] 50 10 - ✗ ✓ ✗ ✗ ✗

LlamaTouch [47] 496 57 7.01 ✓ ✓ ✗ ✓ ✗ AndroidWorld [23] 116 20 - ✓ ✓ ✗ ✗ ✗ AndroidLab [40] 138 9 8.5 ✓ ✓ ✗ ✗ ✗

LearnGUI (Ours) 2,353 73 13.2 ✓ ✓ ✓ ✓ ✓

with the most striking results seen in Gemini-1.5-Pro [26], whose accuracy increases from 19.3% to 51.7% (a 198.9% relative improvement). Performance gains are particularly pronounced in complex applications, with accuracy in CityMapper increasing from 14.1% to 69.4% and in To-Do apps from 17.4% to 69.2%. For real-world online evaluations, our framework demonstrates exceptional effectiveness, with Qwen2-VL-7B [31] with LearnAct achieving significant performance gains, while UI-TARS-7B-SFT [22]’s task success rate improves from 18.1% to 32.8% (+14.7%). These findings offer a practical pathway to developing more adaptable and personalized mobile GUI agents.

In summary, our contributions are as follows:

- • We develop LearnGUI, the first dataset designed for studying demonstration-based learning in mobile GUI agents, comprising 2,252 offline and 101 online tasks with high-quality human demonstrations.
- • We design and implement LearnAct, a sophisticated multiagent framework that systematically extracts, retrieves, and leveragesknowledgefromhumandemonstrations. This framework includes three specialized components: DemoParser (knowledge extraction), KnowSeeker (knowledge retrieval), and ActExecutor (task execution).
- • Our evaluations demonstrate unprecedented performance gains:asingle demonstration improves Gemini-1.5-Pro [26]’s accuracy by 198.9% in offline tests, while enhancing UI-TARS7B-SFT [22]’s online task success rate from 18.1% to 32.8%, advancing mobile GUI agents toward greater adaptability and practical deployability.

### 2 RELATED WORK

Mobile GUI Datasets and Environments. The development of mobile GUI agents relies heavily on high-quality datasets for training and evaluation. Table 1 compares LearnGUI and existing mobile

GUI datasets and benchmarks. These resources can be broadly categorized into static datasets and dynamic benchmarking environments. Static datasets [3–5, 14, 15, 24, 27, 30, 45] typically provide natural language task descriptions, UI states (screenshots and/or application UI trees), and corresponding user actions (CLICK, SWIPE, TYPE, and other standardized interactions). These datasets vary in scale, ranging from hundreds to tens of thousands of instructions across different applications. Recent work like AppAgent [44] has explored demonstration-based learning but without ground truth annotations or systematic analysis, providing only 50 tasks across 10 applications with high-level instructions. Notably, the average task length varies significantly across datasets, with AMEX [5] featuring substantially longer sequences (12.8 steps on average) compared to AndroidControl (4.8 steps) and AITW [24] (6.5 steps). Benchmarking environments, on the other hand, typically select a limited number of tasks and applications to provide dynamic testing environments [17]. These frameworks evaluate agent performance through metrics such as task completion rates, critical state achievements, and execution time. Examples include LlamaTouch [47], AndroidWorld [23], and AndroidLab [40], which offer interactive environments but lack few-shot demonstration capabilities. We present the first systematic study of demonstration-based learning for mobile GUI agents through LearnGUI, which distinguishes itself through three key innovations. First, it is designed to evaluate few-shot learning capabilities with a comprehensive collection of 2,252 offline tasks and 101 online tasks. Built upon AMEX [5] and AndroidWorld [23], which feature longer, more complex tasks ideal for out-of-distribution and demonstration-based learning scenarios, LearnGUI provides a unified framework for both offline and online evaluation. Second, while the original AMEX [5] dataset contains 2,946 independent tasks unsuitable for few-shot evaluation, we conducted detailed analyses to transform and enhance this resource. Specifically, we made three key modifications: (1) Action Space Standardization, refining the original action space by removing inconsistent TASK_IMPOSSIBLE actions, enhancing TASK_COMPLETE to support information retrieval tasks, and standardizing formats for consistency; (2) K-shot Task Combinations, constructing systematic task groupings by recovering application context, computing instruction similarity within applications, and creating k-shot combinations with similar tasks as support demonstrations; and (3) Similarity Measurement, computing UI and action similarity through descriptive representations, enabling analysis of how different similarity types affect learning efficacy. Third, regarding online evaluation, AndroidWorld [23] originally provides 116 dynamically constructed tasks without human demonstration trajectories. We collected 101 high-quality human demonstrations based on AndroidWorld’s environment and dynamic instructions, forming LearnGUI-Online for evaluating the few-shot capabilities of mobile GUI agents in real-time scenarios. By addressing the limitations of existing datasets, LearnGUI enables systematic research into few-shot learning for mobile GUI agents with varying k-shot configurations and controlled similarity conditions between support and query tasks.

Mobile GUI Agents. Mobile GUI agents are intelligent systems that leverage large language models to understand, plan, and execute tasks on mobile devices by integrating natural language

processing, multimodal perception, and action execution capabilities [32, 38]. Recent developments in this field have explored various approaches to enhance agent performance and generalizability. One prominent category of work focuses on designing effective prompting strategies to guide pre-trained LLMs without additional training [8, 35, 42]. By crafting prompts that incorporate task descriptions, interface states, and action histories, researchers can direct model behavior toward specific automation goals [25, 28, 29, 37]. These approaches leverage the inherent capabilities of generalpurpose LLMs but often struggle with complex tasks. A second category involves adapting LLMs specifically for mobile automation through fine-tuning techniques [7, 9, 11, 16, 19, 21, 41]. These methods train models on GUI-specific data to enhance their understanding of and interaction with graphical interfaces. While improving performance over pre-training approaches, these fine-tuned models require substantial training data and still face generalization challenges. Despite the progress made by both approaches, a fundamental limitation persists: the inability to generalize effectively to out-of-distribution scenarios. These methods both struggle with unseen applications, novel UI layouts, or unexpected task variations. These limitations stem from the impossibility of covering all potential real-world scenarios during training, creating significant bottlenecks in mobile GUI agent development. To address these critical challenges, we introduce LearnAct, a sophisticated multiagent framework that learns and reasons from screenshots without requiring UI tree information. The framework extracts, retrieves, and utilizes demonstration knowledge through three specialized components, enabling effective adaptation to new scenarios with minimal demonstrations.

### 3 LEARNGUI DATASET 3.1 Task Definition

Mobile GUI tasks require agents to interact with digital environments by executing actions to fulfill user instructions. These tasks can be formally described as a Partially Observable Markov Decision Process (POMDP), defined as M = (S, O, A, T, R), where S is the state space (current state of the mobile device), O is the observation space (instructions, screenshots, UI trees, etc.), A is the action space (e.g., click, type, swipe), T : S × A → S is the state transition function, and R : S × A → [0, 1] is the reward function. For example, a user might request the agent to "find the cheapest hotel in Paris for next weekend." The agent must perceive the current screen—either through an image or a UI tree—and execute a sequence of actions to complete the given task.

The key innovation in our approach is the integration of human demonstration knowledge into this POMDP framework. By incorporating demonstration knowledge D into the decision process, we enhance the agent’s ability to handle out-of-distribution scenarios. This knowledge influences the agent’s policy 𝜋 : O × D → A, which maps observations and relevant demonstration knowledge to actions, providing valuable examples of successful interaction patterns.

To study the impact of demonstration-based learning on mobile GUI agents, we need a dataset that provides various k-shot demonstrations with controlled similarity relationships between support and query tasks. This allows us to systematically investigate

how demonstration quantity and task similarity affect agent performance. While cross-application knowledge transfer remains an interesting research direction, we focus on within-application task learning, as this represents the most practical use case where users would provide demonstrations for applications they frequently use.

Our dataset design specifically enables research on three key dimensions:

- (1) Unified comprehensive evaluation framework: LearnGUI provides a standardized platform for studying few-shot demonstration learning in mobile GUI agents, featuring a unified action space and evaluation protocols that reflect real-world use cases
- (2) K-shot demonstration learning: The dataset systematically explores how varying quantities of demonstrations (k=1, 2, or 3) affect agent performance, enabling research on the optimal number of examples needed
- (3) Multi-dimensional similarity analysis: LearnGUI enables investigation of how different types of similarity between demonstration and query tasks influence learning efficacy and generalization capabilities

This comprehensive approach allows for a nuanced analysis of how mobile GUI agents can leverage human demonstrations to improve task performance, especially in scenarios not covered by their training data.

### 3.2 Data Collection

The LearnGUI dataset consists of two components: LearnGUIOffline for systematic evaluation of few-shot learning capabilities across varying similarity conditions, and LearnGUI-Online for real-time assessment in an interactive environment. Both components share a unified action space to ensure consistent evaluation, as detailed in Table 2.

###### Table 2: LearnGUI Action Space

Action Definition CLICK[x, y] Click at coordinates (x, y). TYPE[text] Type the specified text. SWIPE [direction] Swipe in the specified direction. PRESS_HOME Go to the home screen. PRESS_BACK Go back to the previous app screen. PRESS_ENTER Press the enter button. TASK_COMPLETE[answer] Mark the task as complete. Provide

answer inside brackets if required.

3.2.1 LearnGUI-Offline. We built LearnGUI-Offline by restructuring and enhancing the AMEX dataset [5], which contains 2,946 independent mobile tasks. To transform this resource for few-shot learning evaluation, we made several key modifications:

Action Space Standardization. We refined the original action space to better align with real-world scenarios. First, we removed

Table 3: Statistics of LearnGUI dataset splits. Each split is analyzed across multiple dimensions: Tasks (number of tasks), Apps (number of applications covered), Step actions (total action steps), similarity metrics (Avg Ins/UI/ActSim), and distribution across four similarity profiles categorized by high (SH) and low (SL) UI and action similarity.

Split K-shot Tasks Apps Step actions Avg InsSim Avg UISim Avg ActSim UISHActSH UISHActSL UISLActSH UISLActSL

- Offline-Train 1-shot 2,001 44 26,184 0.845 0.901 0.858 364 400 403 834

- Offline-Train 2-shot 2,001 44 26,184 0.818 0.898 0.845 216 360 358 1,067

- Offline-Train 3-shot 2,001 44 26,184 0.798 0.895 0.836 152 346 310 1,193

- Offline-Test 1-shot 251 9 3,469 0.798 0.868 0.867 37 49 56 109

- Offline-Test 2-shot 251 9 3,469 0.767 0.855 0.853 15 42 55 139

- Offline-Test 3-shot 251 9 3,469 0.745 0.847 0.847 10 36 49 156 Online-Test 1-shot 101 20 1,423 - - - - - - -

TASK_IMPOSSIBLE actions due to inconsistent labeling in the original dataset, which included errors such as tasks being incorrectly marked as impossible. Second, we enhanced TASK_COMPLETE to TASK_COMPLETE[answer] for information retrieval tasks. Many mobile tasks require returning specific information rather than just completion status. This aligns with both AMEX [5] and AndroidWorld [23] paradigms.

K-shot Task Combinations. We constructed systematic k-shot task combinations through a multi-step process. We began by recovering the application context for each task through instruction and screenshot analysis, as the original dataset lacked explicit app labels. Next, we computed instruction similarity between tasks within the same application using the all-MiniLM-L6-v2 model. Finally, we created k-shot combinations (k=1,2,3) for each query task by selecting the k most similar tasks within the same application as support demonstrations, ensuring that the average similarity exceeded a minimum threshold of 0.6. This process yielded 2,252 tasks with valid k-shot combinations.

Similarity Measurement. To enable multi-dimensional similarity analysis, we computed metrics across three key dimensions. For Instruction Similarity, we utilized the scores calculated during the K-shot Task Combinations process. For UI Similarity, we merged the UI trees from all steps of each task and calculated similarity using TF-IDF vectorization and cosine similarity, capturing the visual and structural similarity of interfaces. For Action Similarity, following the DemoParser approach detailed in Section 4.1, we generated descriptive representations of each action and computed embedding-based cosine similarity between task pairs.

3.2.2 LearnGUI-Online. For evaluating mobile GUI agents in realtime interactive scenarios, we developed LearnGUI-Online based on the AndroidWorld environment [23]. While AndroidWorld provides 116 dynamically constructed task templates, it lacks human demonstration trajectories essential for few-shot learning evaluation.

We identified 101 tasks suitable for human completion, excluding 15 tasks that proved challenging for human users. We then collected high-quality human demonstrations for these tasks. For tasks with dynamic elements, we generated specific instances and recorded corresponding demonstrations.

The resultingLearnGUI-Onlinedataset provides a realistic testbed for evaluating few-shot learning capabilities in mobile GUI agents under authentic conditions.

### 3.3 Dataset Statistics

Table 1 presents the comprehensive statistics of the LearnGUI dataset in comparison with existing datasets. With 2,353 instructions across 73 applications and an average of 13.2 steps per task, LearnGUI offers rich data for studying demonstration-based learning in mobile GUI agents. The dataset provides various k-shot combinations (k=1,2,3) for each task, along with multi-dimensional similarity metrics across instruction, UI, and action dimensions. This design enables systematic analysis of how different types and quantities of demonstrations affect learning outcomes. The similarity distribution reflects the natural variation in mobile tasks within applications, with a meaningful spread across similarity levels that allows for a detailed investigation of knowledge transfer under different conditions. A detailed visualization of these similarity distributions is provided in Appendix A.

### 3.4 Dataset Splits

We divided LearnGUI-Offline into training and testing splits to enable systematic evaluation of few-shot learning capabilities. Table 3 presents the detailed statistics of these splits, including the distribution of tasks across different similarity profiles.

The training set contains 2,001 tasks for each k-shot configuration (1, 2, and 3), spanning 44 applications with an average of 13.1 steps per task. The test set includes 251 tasks per k-shot configuration across 9 applications. Both splits maintain the same action space and similarity measurement methodology.

Based on empirical analysis, we established threshold values of 0.9447 for UI similarity and 0.9015 for action similarity to classify tasks into high (SH) and low (SL) similarity categories, enabling systematic analysis of how different similarity types affect learning from demonstrations.

As shown in Figure 3, we classify tasks into four categories based on UI and action similarity:

- • UISHActSH: High UI similarity and high action similarity. For example, in a smart home app, two tasks that both involve adjusting the brightness of different lights in the living room would navigate through similar UI screens.
- • UISHActSL: High UI similarity but low action similarity. For instance, in a smart home app, turning on all lights with a single button press versus adjusting each light’s color temperature.
- • UISLActSH: Low UI similarity but high action similarity. For example, setting a schedule for lights versus setting a

[Figure 42]

- Figure 3: Joint distribution of UI similarity and action similarity in LearnGUI-Offline. The scatter plot shows the relationship between UI and action similarity measures across task pairs. The quadrant divisions represent our categorization of tasks into

four profiles: UISHActSH, UISHActSL, UISLActSH, and UISLActSL, enabling analysis of how different similarity combinations affect learning transfer.

schedule for the thermostat—different UI screens but similar action patterns.

• UISLActSL: Low UI similarity and low action similarity. For instance, checking security camera footage versus creating a scene that coordinates multiple devices.

This categorization enables a detailed analysis of how different types of similarity affect learning efficacy. For instance, we can investigate whether UI similarity or action similarity has a greater impact on successful knowledge transfer from demonstrations.

Additionally, the LearnGUI-Online test set contains 101 tasks across 20 applications. Unlike the offline dataset, these tasks are evaluated in real time through direct interaction with the mobile environment.

The comprehensive structure of LearnGUI, with its carefully designed splits and similarity classifications, provides a resource for studying how mobile GUI agents can learn from demonstrations under varying conditions of task similarity and demonstration quantity.

- 4 METHOD: LEARNACT

Building on the insights from our LearnGUI dataset, we introduce LearnAct, a novel framework designed to break through the limitations of traditional training approaches for mobile GUI agents. Rather than pursuing universal generalization through extensive training data, LearnAct establishes demonstration-based learning as a paradigm for developing more adaptable, personalized, and practically deployable mobile GUI agents. As illustrated in Figure 4,

LearnAct is a sophisticated multi-agent framework that automatically understands human demonstrations, generates instructional knowledge, and leverages this knowledge to assist mobile GUI agents in reasoning about unseen scenarios. The LearnAct framework consists of three specialized components, each addressing a critical aspect of demonstration-based learning: (1) DemoParser (Section 4.1), a knowledge generation agent that extracts usable knowledge from demonstration trajectories to form a knowledge base; (2) KnowSeeker (Section 4.2), a knowledge retrieval agent that searches the knowledge base for demonstration knowledge relevant to the current task; and (3) ActExecutor (Section 4.3), a task execution agent that combines user instructions, real-time GUI environment, and retrieved demonstration knowledge to perform tasks effectively.

### 4.1 DemoParser

The DemoParser transforms raw human demonstrations into structured demonstration knowledge. It takes as input a raw action sequence (consisting of coordinates-based clicks, swipes, and text inputs) along with corresponding screenshots and task instructions. It then utilizes a vision-language model to generate semantically descriptive action descriptions that capture the essence of each demonstration step (e.g., “On Search Page, click the search box, to enter keywords”). Building on these descriptions, it constructs a structured knowledge base that records both the high-level action semantics and the contexts in which they occur, as shown in Figure 5.

Formally, DemoParser implements a knowledge generation function 𝐺 : I × S × A → K, where I represents the space of instructions, S is the space of screenshot sequences, A is the space of action sequences, and K is the knowledge space. For each demonstration trajectory (𝑖,𝑠,𝑎) ∈ I × S × A, DemoParser generates a knowledge entry 𝑘 ∈ K that encapsulates the demonstration in a semantically descriptive format, converting raw coordinate-based actions (e.g., CLICK[123,456]) into meaningful operation descriptions (e.g., "click search box").

The knowledge generation process is decomposed into a sequence of description generation steps for each action in the demonstration trajectory. Let 𝑑𝑗 represent the description for action 𝑎𝑗, which is generated using a context-aware description function 𝛿 : I × A𝑗 × V𝑗 × H𝑗−1 → D, where V𝑗 is the visual representation of action 𝑎𝑗 execution and H𝑗−1 = {𝑑1,𝑑2, . . .,𝑑𝑗−1} is the history of previous action descriptions.

Algorithm 1 in Appendix B.3 outlines the knowledge generation process. For each demonstration, DemoParser preserves the original task instruction and action sequence while generating semantically descriptive action descriptions. These descriptions provide crucial context about the purpose and significance of each action in the demonstration, enabling more effective knowledge transfer to new scenarios.

For intermediate actions, DemoParser analyzes a visual representation of the action execution, showing before-action and afteraction screenshots with the action visualized (e.g., click locations highlighted). The framework combines this visual input with the task instruction, action history, and current action to generate a description that follows a standardized format: "[On/In] [Screen

[Figure 43]

- Figure 4: Illustration of the overall framework of LearnAct. Architecture diagram showing the three main components (DemoParser, KnowSeeker, ActExecutor) and their interconnections within the LearnAct system, including data flow from human demonstrations to execution.

Name], [Action Details], to [Purpose]". For example: "On Home Screen, tap ’Settings’ icon, to access device configuration." For terminal actions, DemoParser processes the final screenshot, task instruction, and complete action history to generate a conclusion in the format: "[On/In] [Screen], complete task, [Reason/Answer]"

A distinctive feature of DemoParser is its memory mechanism, which captures critical information observed during task execution that may be necessary for future steps. The model identifies and annotates task-relevant information that is directly related to the user’s instruction, will likely be needed in subsequent steps, and has not been previously recorded. These memory annotations are included in the action descriptions when appropriate: "[On/In] [Screen], [Action], to [Purpose]. [Memory: important information for future steps]". For example, in a shopping task, a memory annotation might capture: "[Memory: iPhone 13 Pro costs $999 with 128GB storage]". The detailed prompt for this memory mechanism is provided in Appendix B.1.

This memory mechanism is particularly valuable for complex tasks requiring information retention across multiple steps, such as comparing prices, remembering account details, or tracking status changes. By transforming raw demonstrations into structured, semantically descriptive knowledge with memory capabilities, DemoParser enables effective knowledge transfer from human demonstrations to automated task execution.

### 4.2 KnowSeeker

KnowSeeker is the retrieval component of the LearnAct framework that identifies demonstration knowledge most relevant to the current task context. As depicted in Figure 6, this agent serves as the bridge between the knowledge base generated by DemoParser and the execution environment of ActExecutor. While DemoParser focuses on transforming demonstrations into structured knowledge, KnowSeeker specializes in efficiently accessing and selecting the most applicable knowledge for a specific task, addressing the critical challenge of knowledge relevance in few-shot learning scenarios.

Formally, KnowSeeker implements a retrieval function 𝑅 : I × K → K(𝑠), where I is the instruction space, K is the knowledge base, and K(𝑠) ⊂ K is a subset of knowledge entries determined to be relevant for the given instruction. This retrieval process is crucial for effective knowledge utilization, as it filters the potentially vast knowledge base to focus exclusively on demonstrations that offer valuable insights for the current task.

The core of KnowSeeker’s retrieval mechanism relies on semantic similarity measurement between the current task instruction and the instructions associated with demonstrations in the knowledge base. This similarity-based retrieval can be formally defined as:

#### (a) Mandate Execution

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

Instruction

Instruction

Instruction

User

[Figure 51]

Please check the temperature in the living room for me and adjust the windows and air conditioner to a suitable state.

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

Screenshot 1

Screenshot 2

Screenshot 2

[Figure 56]

[Figure 57]

[Figure 58]

Action 1

Action 2

Action 2

[Figure 59]

[Figure 60]

#### (b) Knowledge Generation

[Figure 61]

[Figure 62]

Action Descripution

Multimodal Prompts

Input

[Figure 63]

[Figure 64]

[Figure 65]

Before-after Steps

[Figure 66]

[Figure 67]

[Figure 68]

Intermediate Terminal Format:

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Reasoning

Format:

Actions Action Histories

Screenshots

[On/In] [Screen Name], task complete, [Reason] or the answer is [answer].

[On/In] [Screen Name], [Action], to [Purpose]. [Memory].

[Figure 76]

Action Descripution

[Figure 77]

Output

[Figure 78]

[Figure 79]

#### (c) Knowledge Data Base Instructions Actions Intermediate Descripution Terminal Descripution

[Figure 80]

[Figure 81]

Memory No memory Answer No answer

Instruction: User Input Action Space: CLICK[x, y], TYPE[text],

Action: CLICK[529,1751] Action Descripution: On the living room detail page, click the air conditioner icon, to turn on the air conditioner.

Descripution: <Answer> On the living room detail page, task complete, the answer is "The temperature in the living room is three degrees".

Descripution: <Memory> On the living room detail page, click the window icon, to open the window. [Memory: The temperature in the living room is three degrees].

SWIPE[direction], PRESS_HOME, PRESSBACK, PRESS_ENTER, TASK_COMPLET[answer].

- Figure 5: Pipeline of DemoParser Agent. Input instructions and corresponding actions and screenshots; output low-level action descriptions and create knowledge database. This process transforms high-level user instructions into precise operation sequences while building a reusable domain knowledge base to improve mobile interface interaction automation efficiency.

Please check the humidity in the bedroom for me and adjust the humidifier and windows to a suitable state.

Instruction Sentence Transformer

12×blocks

Avgpooling

Embedding Space

[Figure 82]

Maxmize the Distance

Minimize the Distance

HomeKit Embeddings Similarity Top K

- Instruction 1
- Instruction 2
- Instruction 3
- Instruction 4 Instruction n

Check the temperature in the bedroom now ...

- 1 =0.8

Check the humidity in the living room now ...

- 2 =0.6

Please check if the door is locked now ...

- 3 =0.4

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

- Figure 6: Pipeline of KnowSeeker Agent. The KnowSeeker Agent converts demo trajectories from the knowledge base into a vector database. When executing user tasks, KnowSeeker retrieves the top-k relevant demos from the vector database for subsequent use. This approach enables efficient retrieval of similar demonstrations to assist with new task execution.

similarity threshold, and 𝑡𝑜𝑝 − 𝑘 indicates selection of the 𝑘 most similar entries.

To implementthissimilaritymeasurement efficiently, KnowSeeker employs a two-phase approach:

- (1) Embedding Generation: Instructions are transformed into dense vector representations using a pre-trained sentence transformer model. Specifically, we utilize the all-MiniLML6-v2 model, which offers an optimal balance between computational efficiency and semantic representational power. This model has been fine-tuned on diverse natural language understanding tasks, making it particularly well-suited for capturing the semantic essence of mobile GUI task instructions.
- (2) Similarity Computation: The cosine similarity between embedding vectors is calculated to quantify the semantic relationship between instructions. For instructions 𝑖 and 𝑖𝑗 with corresponding embeddings 𝑒𝑖 and 𝑒𝑗, the similarity is computed as:

𝑒𝑖 · 𝑒𝑗 ||𝑒𝑖|| · ||𝑒𝑗 ||

(2)

𝑠𝑖𝑚(𝑖,𝑖𝑗) =

To optimize retrieval efficiency, KnowSeeker pre-computes embeddings for all instructions in the knowledge base during initialization. This approach transforms the potentially expensive operation of computing pairwise similarities during runtime into a more manageable vector comparison task. The pre-computation process is described as:

𝑅(𝑖,𝐾) = {𝑘𝑗 ∈ 𝐾 | 𝑠𝑖𝑚(𝑖,𝑖𝑗) ≥ 𝜏𝑠}𝑡𝑜𝑝𝑗=1−𝑘 (1) where 𝑖 is the current instruction, 𝑖𝑗 is the instruction associated

with knowledge entry 𝑘𝑗, 𝑠𝑖𝑚(·, ·) is a similarity function, 𝜏𝑠 is a

(a) ActExecutor Agent

[Figure 96]

[Figure 97]

[Figure 98]

Input

[Figure 99]

Prompt Engineering Support Task

[Figure 100]

[Figure 101]

User

[Figure 102]

[Figure 103]

ScreenShot

[Figure 104]

Environment

[Figure 105]

[Figure 106]

Click screens tab Click away mode icon Click security icon Click away icon Task complete

[Figure 107]

[Figure 108]

Instruction

Input

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

Input Memory Action History

[Figure 113]

[Figure 114]

Run the movie night scene and then set the alarm system to home mode for me.

[Figure 115]

ActExecutor

[Figure 116]

[Figure 117]

Input

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Inference

###### LLM

(b) Actual Tasks Execution

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

CLICK[545, 2289] CLICK[912, 1458] CLICK[756, 2282] CLICK[547, 546] TASK_COMPETE

- Figure 7: Pipeline of ActExecutor Agent. The ActExecutor Agent executes the low-level action descriptions generated by the Action Planner Agent. It uses the KnowSeeker Agent to retrieve relevant demonstrations from the knowledge base and execute the actions in the demonstrations. This approach enables efficient execution of low-level actions to assist with new task execution.

𝐸 = {𝑒𝑗 = 𝑓𝑒𝑚𝑏𝑒𝑑 (𝑖𝑗) | 𝑘𝑗 ∈ 𝐾} (3)

where 𝑓𝑒𝑚𝑏𝑒𝑑 is the embedding function implemented by the sentence transformer model.

During task execution, when presented with a new instruction 𝑖, KnowSeeker: 1. Computes the embedding 𝑒𝑖 = 𝑓𝑒𝑚𝑏𝑒𝑑 (𝑖) 2. Calculates similarities 𝑆 = {𝑠𝑖𝑚(𝑒𝑖,𝑒𝑗) | 𝑒𝑗 ∈ 𝐸} 3. Selects the top-𝑘 knowledge entries based on similarity scores

This approach ensures that knowledge retrieval scales efficiently with the size of the knowledge base, enabling rapid identification of relevant demonstrations even as the framework’s experiential knowledge grows over time. By systematically identifying the most relevant demonstration knowledge, KnowSeeker enables ActExecutor to perform tasks more effectively, particularly in unfamiliar scenarios.

### 4.3 ActExecutor

ActExecutor is the execution component of the LearnAct framework that translates retrieved demonstration knowledge into effective actions in the target environment. As illustrated in Figure 7, this agent represents the culmination of the LearnAct pipeline, integrating user instructions, real-time GUI observations, and demonstration knowledge to navigate even unfamiliar mobile applications successfully. While DemoParser creates structured knowledge and KnowSeeker retrieves relevant demonstrations, ActExecutor applies this knowledge to solve practical tasks, addressing the critical challenge of knowledge utilization in few-shot learning scenarios.

ActExecutor implements the POMDP framework introduced earlier, with the critical enhancement of incorporating demonstration knowledge into the decision-making process. The execution process can be formally described as a sequential decision-making loop that iteratively selects actions 𝑎𝑡 ∈ A based on current observations 𝑜𝑡 ∈ O and demonstration knowledge D, following policy 𝜋 : O × D → A.

The ActExecutor policy 𝜋 is implemented through a large visionlanguage model that processes a carefully constructed prompt integrating all available information sources. This prompt-based policy can be expressed as:

𝜋(𝑜𝑡, D) = 𝑓𝐿𝐿𝑀 (𝑃(𝑖,𝑜𝑡,ℎ𝑡−1, D)) (4)

where 𝑖 is the user instruction, 𝑜𝑡 is the current observation (screenshot), ℎ𝑡−1 is the action history up to time 𝑡 − 1, D is the retrieved demonstration knowledge, 𝑃 is a prompt construction function, and 𝑓𝐿𝐿𝑀 is the LLM-based decision function.

Algorithm 2 in Appendix B.3 outlines the execution process. For each task, ActExecutor processes the user instruction and screenshot observations through a sequence of perception, decision, and action phases until the task is completed or a maximum step limit is reached.

The execution process integrates three key phases:

- (1) Perception Phase: ActExecutor perceives the current state

of the mobile device through screenshot observations 𝑜𝑡. These observations provide the visual context essential for understanding the available interaction options and current application state.

- (2) Decision Phase: The agent constructs a comprehensive prompt that integrates the user instruction 𝑖, current obser-

vation 𝑜𝑡, action history ℎ, and retrieved demonstrations D. This prompt is processed by a large vision-language model using templates detailed in Appendix B.2, resulting in a selected action from the predefined action space described in Table 2.

- (3) Action Phase: The selected action 𝑎𝑡 is executed in the mobile environment, generating a state transition according to the transition function T of the POMDP. Additionally, the

agent generates a description𝑑𝑡 of the executed action using a process similar to DemoParser’s description generation, which serves as part of the action history for subsequent steps.

The prompt construction function 𝑃 plays a critical role in ActExecutor’s effectiveness. It integrates the agent’s role definition, demonstration examples, task and observation context, action history, and the action space definition into a comprehensive prompt that guides the model’s decision-making.

This approach enables ActExecutor to leverage demonstrations as exemplars that guide its decision-making process. When faced with a novel UI state, the agent identifies analogous situations from demonstrations and adapts the demonstrated actions to the current context. This capability is particularly valuable for handling out-ofdistribution scenarios where the agent lacks direct experience.

By closing the loop between demonstration knowledge and task execution, ActExecutor completes the LearnAct framework’s endto-end pipeline for demonstration-based learning. The combination of knowledge generation (DemoParser), knowledge retrieval (KnowSeeker), and knowledge-guided execution (ActExecutor) enables effective few-shot learning for mobile GUI agents, addressing the fundamental challenge of generalization to unseen scenarios with minimal examples.

### 5 EXPERIMENTS

We conducted comprehensive evaluations of the LearnAct framework through both offline and online experiments. The offline experiments were performed on the LearnGUI-Offline dataset to evaluate step-by-step task execution capabilities, while the online experiments utilized the LearnGUI-Online platform to assess end-to-end task completion in real-world interactive scenarios. We evaluated a diverse set of models, including both commercial (e.g., Gemini1.5-Pro [26]) and open-source models (e.g., UI-TARS-7B-SFT [22], Qwen2-VL-7B [31]), to demonstrate the broad applicability of our approach across different model architectures and capabilities.

### 5.1 Experiment Setup

The diverse similarity profiles in LearnGUI provide a unique opportunity to evaluate mobile GUI agents’ capabilities. Our experiments have two primary goals: (1) to evaluate the feasibility and effectiveness of enhancing mobile agents through few-shot demonstrations as a means to overcome the limitations of traditional pre-training or fine-tuning approaches; and (2) to investigate how different factors such as demonstration quantity (k=1,2,3) and various similarity aspects (instruction, UI, and action) influence the effectiveness of demonstration-based learning.

Implementation Details. We conducted experiments with threefoundationmodels: Gemini-1.5-Pro [26], UI-TARS-7B-SFT[22], and Qwen2-VL-7B [31]. For all models, we set the temperature to zero to obtain deterministic responses. For Qwen2-VL-7B [31] and UI-TARS-7B-SFT [22], we employed parameter-efficient fine-tuning using LoRA with rank 64, alpha 128, and dropout probability 0.1. We targeted all modules while freezing the vision encoder to ensure computational efficiency. Training used a learning rate of 1e-5 with cosine scheduling, batch size of 1, gradient accumulation over 8 steps, a warmup ratio of 0.001, and was conducted for 1 epoch. All fine-tuning experiments were conducted on 8 NVIDIA L40S GPUs. For offline experiments, Gemini-1.5-Pro [26] was evaluated directly on the LearnGUI-Offline test set without additional training. UI-TARS-7B-SFT [22] and Qwen2-VL-7B [31] were fine-tuned on the LearnGUI-Offline training set before evaluation. For online experiments, we deployed all models except Gemini-1.5-Pro [26] (which showed limited task completion capabilities in preliminary tests despite accuracy improvements) to the LearnGUI-Online environment, using 1-shot demonstration retrieval for all LearnActenhanced models.

Baselines. To rigorously evaluate our approach, we compared LearnAct against several baselines. These include: (1) SPHINXGUI Agent, the original agent developed for the AMEX dataset [5], providing a reference point for task execution on similar data; (2) Zero-shot inference versions of all models (Gemini-1.5-Pro [26], UI-TARS-7B-SFT [22], and Qwen2-VL-7B [31]) within the LearnAct framework but without demonstration knowledge, maintaining identical execution environments for fair comparison; and (3) For online evaluation, we additionally compared against GPT-4o, Gemini-Pro-1.5, Claude Computer-Use, and Aguvis to benchmark against current advanced systems.

Evaluation Metrics. For offline evaluation, we adopted mainstream evaluation protocols widely used in recent mobile GUI agent research, such as UI-TARS [22] and OS-ATLAS [39]. Specifically, we

- Table 4: Performance comparison of mobile GUI agents on LearnGUI-Offline dataset (action match accuracy %). Results show absolute values and relative improvements [in brackets] compared to baselines. Performance is evaluated across different models and number of support examples (1/2/3-shot).

Models Method Supports Average Gmail Booking Music SHEIN NBC CityMapper ToDo Signal Yelp SPHINX-GUI Agent[5] AMEX 0-shot 67.2 45.9 64.5 74.4 71.8 70.3 67.4 79.3 64.9 66.3

Baseline 0-shot 19.3 20.1 16.4 24.5 10.2 35.6 14.1 17.4 27.9 15.2

- 1-shot 51.7 [+32.4] 55.5 47.1 60.0 35.7 56.4 54.7 60.6 63.1 54.6

- 2-shot 55.6 [+36.3] 57.5 53.2 55.3 39.6 56.1 58.2 68.1 69.7 60.0

- 3-shot 57.7 [+38.4] 58.4 56.6 54.6 43.9 53.9 69.4 69.2 70.5 57.6

gemini-1.5-pro

LearnAct

Baseline 0-shot 77.5 68.1 81.0 81.1 72.9 80.9 70.6 66.0 92.6 82.4

- 1-shot 82.8 [+5.3] 79.9 82.9 86.6 75.7 86.3 79.4 84.0 89.3 83.0

- 2-shot 81.9 [+4.4] 80.1 80.7 86.2 76.1 87.2 80.0 83.7 84.4 84.2

- 3-shot 82.1 [+4.6] 79.9 80.9 86.2 75.7 86.9 81.2 85.8 84.4 84.2

UI-TARS-7B-SFT

LearnAct

Baseline 0-shot 71.8 60.8 73.9 76.0 65.5 75.5 62.9 78.7 82.8 69.1

- 1-shot 77.3 [+5.5] 75.0 77.5 77.8 69.8 83.5 72.9 78.0 83.6 78.8

- 2-shot 78.5 [+6.7] 75.0 78.0 77.8 73.3 86.0 73.5 81.9 87.7 77.6

- 3-shot 79.4 [+7.6] 75.0 78.8 78.6 72.6 87.8 77.1 82.6 87.7 80.6

Qwen2-VL-7B

LearnAct

measured step accuracy, which consists of two components: action type accuracy and action match accuracy. Action type accuracy measures the percentage of steps where the predicted action type (CLICK, TYPE, SWIPE, etc.) matches the ground truth. Action match accuracy measures the percentage of steps where both the action type and its parameters are correct, following standard evaluation criteria. For CLICK actions, coordinates are considered correct if they fall within 14% of the screen width from the ground truth. For TYPE actions, the content is correct if the F1 score between prediction and ground truth exceeds 0.5. For SWIPE actions, the direction must precisely match the ground truth. For other actions (e.g., PRESS_BACK), an exact match is required. For TASK_COMPLETE actions, we only verify the action type and ignore the answer field. For online evaluation, we measured the task success rate (SR), which represents the percentage of tasks completed successfully in the real-time interactive environment.

### 5.2 Main Results

- 5.2.1 Offline Agent Capability Evaluation. Table 4 presents the performance comparison of different models on the LearnGUIOffline dataset. The results demonstrate the substantial improvements achieved by the LearnAct framework across all tested models. Gemini-1.5-Pro [26] shows the most dramatic improvement, with performance increasing from 19.3% to 51.7% (+32.4%) with just a single demonstration, and further improving to 57.7% (+38.4%) with three demonstrations. This represents a 198.9% relative improvement, highlighting the powerful potential of demonstrationbased learning even for advanced foundation models. UI-TARS7B-SFT [22], despite already having strong zero-shot performance (77.5%), still achieves significant gains with LearnAct, reaching 82.8% (+5.3%) with a single demonstration. This indicates that even models specifically fine-tuned for GUI tasks can benefit from demonstration knowledge. Qwen2-VL-7B [31] demonstrates consistent improvement from 71.8% to 77.3% (+5.5%) with one demonstration, and to 79.4% (+7.6%) with three demonstrations, confirming that

the benefits of LearnAct generalize across models with different architectures and capabilities.

The results also reveal interesting patterns regarding the impact of demonstration quantity. For Gemini-1.5-Pro [26], performance scales monotonically with the number of demonstrations, suggesting that less specialized foundation models can benefit substantially from additional examples. In contrast, UI-TARS-7B-SFT [22] achieves its peak performance with just one demonstration, indicating that models already fine-tuned for GUI tasks may efficiently extract necessary information from minimal demonstrations.

Application-specific results highlight LearnAct’s consistent improvement across diverse scenarios, with particularly notable gains in complex applications like CityMapper (from 14.1% to 69.4% for Gemini-1.5-Pro [26]) and To-Do apps (from 17.4% to 69.2%). This suggests that demonstration-based learning is especially valuable for navigating applications with complex interactions and nonstandard interfaces.

To further understand the factors influencing LearnAct’s effectiveness, we analyzed performance across different similarity profiles, as shown in Table 5. Several important insights emerge: Gemini-1.5-Pro [26] shows substantial improvements across all similarity combinations, with the largest gains in action match accuracy (ranging from +29.3% to +39.6%). This indicates that demonstration knowledge significantly enhances the model’s ability to execute precise actions regardless of similarity conditions. UI-TARS-7BSFT [22] exhibits the most pronounced improvements in UISHActSH scenarios (+13.9% with 3-shot), suggesting that the model can extract maximum value from demonstrations when both UI and action patterns are similar to the target task. Qwen2-VL-7B [31] shows notably large improvements in action type accuracy for 2-shot settings (e.g., +67.4% for UISHActSH), potentially indicating a threshold effect where multiple demonstrations trigger significant pattern recognition improvements.

Interestingly, while UI similarity generally correlates with higher performance gains, we observe that action similarity also plays a

- Table 5: Performance breakdown of LearnAct-Offline on different UI and action combinations. Performance metrics (type and match accuracy) across four similarity quadrants showing absolute values and relative improvements [in brackets] compared to baselines. Results are grouped by model and number of support examples (1/2/3-shot).

Models Supports UISHActSH UISHActSL UISLActSH UISLActSL

type match type match type match type match

gemini-1.5-pro

- 1-shot 79.5 [+12.8] 50.2 [+35.6] 78.1 [+12.3] 47.8 [+33.2] 77.5 [+9.2] 52.3 [+30.5] 77.9 [+14.1] 44.2 [+29.3]

- 2-shot 77.7 [+13.0] 53.9 [+37.3] 73.2 [+10.8] 49.9 [+34.7] 80.0 [+9.0] 56.5 [+34.8] 77.2 [+12.9] 48.9 [+34.4]

- 3-shot 72.3 [+15.8] 53.5 [+39.6] 72.8 [+12.9] 49.5 [+34.6] 78.7 [+10.4] 60.0 [+38.4] 79.2 [+12.8] 51.6 [+36.3]

Qwen2-VL-7B

- 1-shot 86.0 [+5.3] 72.2 [+6.3] 85.4 [+4.9] 69.6 [+5.5] 86.0 [+2.0] 76.2 [+5.4] 82.9 [+1.3] 69.4 [+4.3]

- 2-shot 85.0 [+67.4] 75.6 [+9.3] 84.0 [+67.2] 71.2 [+5.7] 86.9 [+73.3] 76.8 [+6.3] 84.0 [+68.5] 70.5 [+5.5]

- 3-shot 80.2 [+5.0] 70.3 [+7.9] 82.9 [+4.7] 70.2 [+5.7] 85.6 [+1.9] 77.5 [+8.4] 85.6 [+3.4] 72.8 [+6.6]

UI-TARS-7B-SFT

- 1-shot 88.1 [+1.9] 77.8 [+6.6] 87.2 [+2.1] 75.3 [+6.4] 87.7 [+0.3] 80.1 [+5.9] 85.0 [-0.2] 75.0 [+2.8]

- 2-shot 85.5 [+2.1] 76.7 [+8.3] 85.7 [+1.6] 75.9 [+4.9] 87.3 [-0.4] 79.1 [+5.9] 84.9 [-0.8] 74.1 [+2.1]

- 3-shot 87.1 [+7.9] 78.2 [+13.9] 85.5 [+2.6] 75.4 [+4.9] 86.0 [-0.9] 78.9 [+6.8] 85.5 [-0.9] 75.2 [+2.7]

- Table 6: Performance comparison of different models on the LearnGUI-Online benchmark. Comparison of models with different inputs (Image, Image+AXTree) and parameters, measuring

that the benefits of demonstration-based learning translate effectively to real-world interactive scenarios. Qwen2-VL-7B [31] with LearnAct achieves 21.1% success rate, showing meaningful improvements over its baseline performance. This suggests that the quality and relevance of demonstrations are highly effective for enhancing model capabilities. UI-TARS-7B-SFT [22] with LearnAct achieves 32.8% success rate, approaching the performance of GPT4o (34.5%) despite using a much smaller model. This indicates that demonstration-based learning can help bridge the gap between smaller specialized models and large foundation models. Detailed visualizations of these performance comparisons are provided in Appendix C.1.To provide concrete examples of how LearnAct performs in real-world scenarios, we present three detailed case studies in Appendix C.2.

task success rate (LearnGUI-OnlineSR) with improvements shown in brackets for models with LearnAct enhancement.

Input Models # Params LearnGUI-OnlineSR

Image + AXTree GPT-4o[12] - 34.5 Image + AXTree Gemini-Pro-1.5[26] - 22.8

Image Claude Computer-Use[2] - 27.9 Image Aguvis[41] 72B 26.1

Image Qwen2-VL-7B + 0-shot 7B 9.9 Image Qwen2-VL-7B + LearnAct 7B 21.1 [+11.2]

Image UI-TARS-7B-SFT + 0-shot 7B 18.1 Image UI-TARS-7B-SFT + LearnAct 7B 32.8 [+14.7]

The moststrikingfindingisthe effectiveness of our demonstration-

based learning approach. The LearnAct framework provides significant performance improvements through its demonstration mechanism, with gains of up to 14.7% in task success rate. This demonstrates the power of high-quality demonstrations for enhancing model performance, highlighting the importance of relevant examples over simply increasing model size.

crucial role. For instance, Gemini-1.5-Pro [26] achieves its highest match accuracy in UISLActSH scenarios (+38.4% with 3-shot), suggesting that action similarity can sometimes compensate for UI differences. This finding highlights the importance of considering both UI and action similarity when designing demonstration-based learning approaches for mobile GUI agents.

These results confirm that the LearnAct framework provides a practical pathway to developing effective mobile GUI agents, making it particularly valuable for application-specific customization and personalization scenarios.

Theseresultsvalidateourhypothesized framework design, demon-

strating that LearnAct successfully leverages demonstration similarity to enhance performance across varying conditions, with the most substantial benefits observed when demonstrations can provide both perceptual and procedural knowledge relevant to the target task.

### 5.3 Ablation Study

To understand the contribution of each component in the LearnAct framework, we conducted ablation experiments on the LearnGUIOffline dataset using Gemini-1.5-Pro [26]. As shown in Table 7, we systematically evaluated the impact of removing either the DemoParser or KnowSeeker component while keeping all other settings constant.

- 5.2.2 Online Agent Capability Evaluation. While offline evaluations provide valuable insights into step-by-step execution capabilities, real-world deployment requires successful end-to-end task completion. Table 6 presents the results of our online evaluation on the LearnGUI-Online benchmark, which reveals several important findings. The LearnAct framework substantially improves performance for both evaluated models, with Qwen2-VL-7B [31] improving from 9.9% to 21.1% (+11.2%) and UI-TARS-7B-SFT [22] from 18.1% to 32.8% (+14.7%). These significant gains demonstrate

The results reveal several important insights. Both components are essential, as removing either component leads to substantial performance degradation compared to the full LearnAct framework. The complete framework achieves 51.7% accuracy, while removing DemoParser reduces performance to 40.6% (-11.1%) and removing

- Table 7: Ablation study of LearnAct components. Performance comparison across four configurations: baseline (no components), DemoParser only, KnowSeeker only, and both components combined. Results are presented as overall average accuracy and per-application breakdown across nine applications.

Ablation Setting

Average Gmail Booking Music SHEIN NBC CityMapper ToDo Signal Yelp DemoParser KnowSeeker

Baseline 19.3 20.1 16.4 24.5 10.2 35.6 14.1 17.4 27.9 15.2

✓ 40.6 47.7 31.3 55.4 29.1 47.0 43.0 58.2 48.8 50.7 ✓ 41.6 46.9 34.1 52.7 27.9 51.9 45.3 51.4 61.1 51.8 ✓ ✓ 51.7 55.5 47.1 60.0 35.7 56.4 54.7 60.6 63.1 54.6

KnowSeeker reduces it to 41.6% (-10.1%). Regarding DemoParser’s contribution, comparing "KnowSeeker only" (40.6%) to the baseline (19.3%), we observe that even without action descriptions, relevant demonstrations improve performance by 21.3%. However, the addition of DemoParser’s action descriptions further enhances performance by 11.1%, confirming the value of structured knowledge extraction. For KnowSeeker’s contribution, the "DemoParser only" configuration (41.6%) also substantially outperforms the baseline, indicating that detailed action descriptions are valuable even with randomly selected demonstrations. However, KnowSeeker’s retrieval of relevant demonstrations provides an additional 10.1% improvement, highlighting the importance of demonstration relevance.

The performance variations across applications are particularly informative. For instance, in the Signal application, DemoParser appears more important (61.1% vs. 48.8% for KnowSeeker only), suggesting that detailed action descriptions are crucial for applications with complex interaction patterns. Conversely, for the ToDo application, KnowSeeker seems more valuable (58.2% vs. 51.4% for DemoParser only), indicating that demonstration relevance may be more critical for applications with varied task types.

These findings validate our multi-agent framework design, confirming that both knowledge extraction (DemoParser) and relevant demonstration retrieval (KnowSeeker) play complementary and essential roles in enabling effective demonstration-based learning for mobile GUI agents.

### 6 DISCUSSION AND FUTURE WORK

Our experimental results demonstrate that demonstration-based learning significantly enhances mobile GUI agents’ capabilities. The substantial performance improvements across all evaluated models validate our core hypothesis that demonstration-based learning effectively addresses generalization challenges. Even advanced foundation models like Gemini-1.5-Pro [26] show dramatic improvements (198.9% relative improvement). Our multi-dimensional similarity analysis reveals that both UI similarity and action similarity influence learning efficacy, with action similarity sometimes compensating for UI differences.

Data Collection and Dataset Expansion. While our approach shows promising results, several limitations and future directions warrant consideration. First, regarding data collection, our current dataset, while comprehensive, could benefit from greater diversity and representativeness. The LearnGUI dataset, comprising 2,252

offline tasks and 101 online tasks, represents a significant step forward but remains limited in scale compared to the vast diversity of mobile applications and user interactions. Future work should expand the dataset to include a broader range of applications, particularly those with complex interaction patterns and specialized domains.

K-shot Learning Analysis. Second, our current investigation of k-shot learning is limited to k=1, 2, and 3 demonstrations. While these configurations provide valuable insights, a more comprehensive analysis of how demonstration quantity affects performance would be beneficial. Future research could explore the relationship between the number of demonstrations and performance gains, potentially identifying optimal demonstration counts for different scenarios and model architectures.

Enhanced Learning and Execution Strategies. Third, our learning and execution strategies could be enhanced to better leverage the relationship between support tasks and query tasks. While our current approach effectively retrieves relevant demonstrations, more sophisticated methods could be developed to extract and transfer knowledge more efficiently. For instance, techniques for abstracting common patterns across demonstrations, identifying critical decision points, and adapting demonstrated strategies to novel scenarios could further improve performance.

Agent Self-Learning. A promising direction for future research is to enable agents to learn from their own successful executions. Currently, our framework relies exclusively on human demonstrations, but agents could potentially learn from their own successful task completions. By incorporating these successful agent executions into the knowledge base, we could enable a form of "selflearning" where agents continuously improve their capabilities through their own experiences.

By addressing these limitations and pursuing these research directions, demonstration-based learning can evolve into a robust paradigm for developing adaptable, personalized, and practically deployable mobile GUI agents that effectively address the diverse needs of real-world users. The insights gained from our multidimensional similarity analysis provide valuable guidance for future research in this domain, suggesting that both UI similarity and action similarity play crucial roles in successful knowledge transfer.

### 7 CONCLUSION

This paper introduces a novel demonstration-based learning paradigm that fundamentally addresses the generalization challenges

faced by mobile GUI agents. Rather than pursuing universal coverage through ever-larger datasets, our approach leverages human demonstrations to enhance agent performance in unseen scenarios. We developed LearnGUI, the first comprehensive dataset for studying demonstration-based learning in mobile GUI agents, comprising 2,252 offline tasks and 101 online tasks with high-quality human demonstrations. We further designed LearnAct, a sophisticated multi-agent framework with three specialized components: DemoParser for knowledge extraction, KnowSeeker for relevant knowledge retrieval, and ActExecutor for demonstration-enhanced task execution. Our experimental results demonstrate remarkable performance gains, with a single demonstration increasing Gemini1.5-Pro [26]’s accuracy from 19.3% to 51.7% in offline tests and enhancing UI-TARS-7B-SFT [22]’s online task success rate from 18.1% to 32.8%. These findings establish demonstration-based learning as a promising direction for developing more adaptable, personalized, and practically deployable mobile GUI agents.

### REFERENCES

- [1] Simone Agostinelli, Andrea Marrella, and Massimo Mecella. 2019. Research challenges for intelligent robotic process automation. In Business Process Management Workshops: BPM 2019 International Workshops, Vienna, Austria, September 1–6, 2019, Revised Selected Papers 17. Springer, 12–18.
- [2] Anthropic. 2024. Developing a computer use model. https://www.anthropic. com/news/developing-computer-use
- [3] Chongyang Bai, Xiaoxue Zang, Ying Xu, Srinivas Sunkara, Abhinav Rastogi, Jindong Chen, et al. 2021. Uibert: Learning generic multimodal representations for ui understanding. arXiv preprint arXiv:2107.13731 (2021).
- [4] Andrea Burns, Deniz Arsan, Sanjna Agrawal, Ranjitha Kumar, Kate Saenko, and Bryan A Plummer. 2021. Mobile app tasks with iterative feedback (motif): Addressing task feasibility in interactive visual environments. arXiv preprint arXiv:2104.08560 (2021).
- [5] Yuxiang Chai, Siyuan Huang, Yazhe Niu, Han Xiao, Liang Liu, Dingyu Zhang, Peng Gao, Shuai Ren, and Hongsheng Li. 2024. Amex: Android multi-annotation expo dataset for mobile gui agents. arXiv preprint arXiv:2407.17490 (2024).
- [6] Yuxiang Chai, Hanhao Li, Jiayu Zhang, Liang Liu, Guozhi Wang, Shuai Ren, Siyuan Huang, and Hongsheng Li. 2025. A3: Android Agent Arena for Mobile GUI Agents. arXiv preprint arXiv:2501.01149 (2025).
- [7] Wentong Chen, Junbo Cui, Jinyi Hu, Yujia Qin, Junjie Fang, Yue Zhao, Chongyi Wang, Jun Liu, Guirong Chen, Yupeng Huo, et al. 2024. GUICourse: From General Vision Language Models to Versatile GUI Agents. arXiv preprint arXiv:2406.11317

(2024).

- [8] Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. 2022. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588 (2022).
- [9] Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yantao Li, Jianbing Zhang, and Zhiyong Wu. 2024. Seeclick: Harnessing gui grounding for advanced visual gui agents. arXiv preprint arXiv:2401.10935 (2024).
- [10] Tiago Guerreiro, Ricardo Gamboa, and Joaquim Jorge. 2008. Mnemonical body shortcuts: improving mobile interaction. In Proceedings of the 15th European conference on Cognitive ergonomics: the ergonomics of cool interaction. 1–8.
- [11] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. 2024. Cogagent: A visual language model for gui agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 14281–14290.
- [12] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024).
- [13] Courtney Kennedy and Stephen E Everett. 2011. Use of cognitive shortcuts in landline and cell phone surveys. Public Opinion Quarterly 75, 2 (2011), 336–348.
- [14] Wei Li, William Bishop, Alice Li, Chris Rawles, Folawiyo Campbell-Ajala, Divya Tyamagundlu, and Oriana Riva. 2024. On the Effects of Data Scale on Computer Control Agents. arXiv preprint arXiv:2406.03679 (2024).
- [15] Yang Li, Jiacong He, Xin Zhou, Yuan Zhang, and Jason Baldridge. 2020. Mapping natural language instructions to mobile UI action sequences. arXiv preprint arXiv:2005.03776 (2020).
- [16] Kevin Qinghong Lin, Linjie Li, Difei Gao, Zhengyuan Yang, Shiwei Wu, Zechen Bai, Weixian Lei, Lijuan Wang, and Mike Zheng Shou. 2024. Showui: One visionlanguage-action model for gui visual agent. arXiv preprint arXiv:2411.17465

(2024).

- [17] William Liu, Liang Liu, Yaxuan Guo, Han Xiao, Weifeng Lin, Yuxiang Chai, Shuai Ren, Xiaoyu Liang, Linghao Li, Wenhao Wang, et al. 2025. Llm-powered gui agents in phone automation: Surveying progress and prospects. (2025).
- [18] Zhe Liu, Cheng Li, Chunyang Chen, Junjie Wang, Boyu Wu, Yawen Wang, Jun Hu, and Qing Wang. 2024. Vision-driven Automated Mobile GUI Testing via Multimodal Large Language Model. arXiv preprint arXiv:2407.03037 (2024).
- [19] Quanfeng Lu, Wenqi Shao, Zitao Liu, Fanqing Meng, Boxuan Li, Botong Chen, Siyuan Huang, Kaipeng Zhang, Yu Qiao, and Ping Luo. 2024. GUI Odyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices. arXiv preprint arXiv:2406.08451 (2024).
- [20] Yadong Lu, Jianwei Yang, Yelong Shen, and Ahmed Awadallah. 2024. Omniparser for pure vision based gui agent. arXiv preprint arXiv:2408.00203 (2024).
- [21] Pawel Pawlowski, Krystian Zawistowski, Wojciech Lapacz, Marcin Skorupa, Adam Wiacek, Sebastien Postansque, and Jakub Hoscilowicz. 2024. TinyClick: Single-Turn Agent for Empowering GUI Automation. arXiv preprint arXiv:2410.11871 (2024).
- [22] Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. 2025. UI-TARS: Pioneering Automated GUI Interaction with Native Agents. arXiv preprint arXiv:2501.12326

(2025).

- [23] Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, et al. 2024. AndroidWorld: A dynamic benchmarking environment for autonomous agents. arXiv preprint arXiv:2405.14573 (2024).
- [24] Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, and Timothy Lillicrap. 2024. Androidinthewild: A large-scale dataset for android device control. Advances in Neural Information Processing Systems 36 (2024).
- [25] Yunpeng Song, Yiheng Bian, Yongtao Tang, and Zhongmin Cai. 2023. Navigating Interfaces with AI for Enhanced User Interaction. arXiv preprint arXiv:2312.11190

(2023).

- [26] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530 (2024).
- [27] Sagar Gubbi Venkatesh, Partha Talukdar, and Srini Narayanan. 2022. Ugif: Ui grounded instruction following. arXiv preprint arXiv:2211.07615 (2022).
- [28] Junyang Wang, Haiyang Xu, Haitao Jia, Xi Zhang, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. 2024. Mobile-Agent-v2: Mobile Device Operation Assistant with Effective Navigation via Multi-Agent Collaboration. arXiv preprint arXiv:2406.01014 (2024).
- [29] Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. 2024. Mobile-agent: Autonomous multi-modal mobile device agent with visual perception. arXiv preprint arXiv:2401.16158 (2024).
- [30] Luyuan Wang, Yongyu Deng, Yiwei Zha, Guodong Mao, Qinmin Wang, Tianchen Min, Wei Chen, and Shoufa Chen. 2024. MobileAgentBench: An Efficient and User-Friendly Benchmark for Mobile LLM Agents. arXiv preprint arXiv:2406.08184

(2024).

- [31] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. 2024. Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution. arXiv preprint arXiv:2409.12191 (2024).
- [32] Shuai Wang, Weiwen Liu, Jingxuan Chen, Weinan Gan, Xingshan Zeng, Shuai Yu, Xinlong Hao, Kun Shao, Yasheng Wang, and Ruiming Tang. 2024. GUI Agents with Foundation Models: A Comprehensive Survey. arXiv preprint arXiv:2411.04890

(2024).

- [33] Wenhao Wang, Zijie Yu, William Liu, Rui Ye, Tian Jin, Siheng Chen, and Yanfeng Wang. 2025. FedMobileAgent: Training Mobile Agents Using Decentralized Self-Sourced Data from Diverse Users. arXiv preprint arXiv:2502.02982 (2025).
- [34] Zhenhailong Wang, Haiyang Xu, Junyang Wang, Xi Zhang, Ming Yan, Ji Zhang, Fei Huang, and Heng Ji. 2025. Mobile-Agent-E: Self-Evolving Mobile Assistant for Complex Tasks. arXiv preprint arXiv:2501.11733 (2025).
- [35] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35

(2022), 24824–24837.

- [36] Hao Wen, Yuanchun Li, Guohong Liu, Shanhui Zhao, Tao Yu, Toby Jia-Jun Li, Shiqi Jiang, Yunhao Liu, Yaqin Zhang, and Yunxin Liu. 2024. Autodroid: Llm-powered task automation in android. In Proceedings of the 30th Annual International Conference on Mobile Computing and Networking. 543–557.
- [37] Hao Wen, Hongming Wang, Jiaxuan Liu, and Yuanchun Li. 2023. Droidbot-gpt: Gpt-powered ui automation for android. arXiv preprint arXiv:2304.07061 (2023).
- [38] Biao Wu, Yanda Li, Meng Fang, Zirui Song, Zhiwei Zhang, Yunchao Wei, and Ling Chen. 2024. Foundations and recent trends in multimodal mobile agents: A survey. arXiv preprint arXiv:2411.02006 (2024).
- [39] Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, et al. 2024. Os-atlas: A foundation action model for generalist gui agents. arXiv preprint arXiv:2410.23218

- (2024).
- [40] Yifan Xu, Xiao Liu, Xueqiao Sun, Siyi Cheng, Hao Yu, Hanyu Lai, Shudan Zhang, Dan Zhang, Jie Tang, and Yuxiao Dong. 2024. AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents. arXiv preprint arXiv:2410.24024 (2024).
- [41] Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. 2024. Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction. arXiv preprint arXiv:2412.04454 (2024).
- [42] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2024. Tree of thoughts: Deliberate problem solving with large language models. Advances in Neural Information Processing Systems 36

(2024).

- [43] Chaoyun Zhang, Shilin He, Jiaxu Qian, Bowen Li, Liqun Li, Si Qin, Yu Kang, Minghua Ma, Qingwei Lin, Saravan Rajmohan, et al. 2024. Large Language Model-Brained GUI Agents: A Survey. arXiv preprint arXiv:2411.18279 (2024).
- [44] Chi Zhang, Zhao Yang, Jiaxuan Liu, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. 2023. Appagent: Multimodal agents as smartphone users. arXiv preprint arXiv:2312.13771 (2023).
- [45] Jiwen Zhang, Jihao Wu, Yihua Teng, Minghui Liao, Nuo Xu, Xiao Xiao, Zhongyu Wei, and Duyu Tang. 2024. Android in the zoo: Chain-of-action-thought for gui agents. arXiv preprint arXiv:2403.02713 (2024).
- [46] Jiayi Zhang, Chuang Zhao, Yihan Zhao, Zhaoyang Yu, Ming He, and Jianping Fan.

2024. MobileExperts: A Dynamic Tool-Enabled Agent Team in Mobile Devices. arXiv preprint arXiv:2407.03913 (2024).

- [47] Li Zhang, Shihe Wang, Xianqing Jia, Zhihan Zheng, Yunhe Yan, Longxi Gao, Yuanchun Li, and Mengwei Xu. 2024. LlamaTouch: A Faithful and Scalable Testbed for Mobile UI Automation Task Evaluation. arXiv preprint arXiv:2404.16054 (2024).
- [48] Zhuosheng Zhang and Aston Zhang. 2023. You only look at screens: Multimodal chain-of-action agents. arXiv preprint arXiv:2309.11436 (2023).

### A ADDITIONAL LEARNGUI STATISTICS

Figure 8 illustrates the distribution of similarity scores across different dimensions in the LearnGUI-Offline dataset, enabling systematic analysis of how different types of similarity between demonstration and query tasks affect learning efficacy.

[Figure 146]

- Figure 8: Distribution of instruction, UI, and action similarity scores in LearnGUI-Offline. The histograms show the distribution of similarity scores across three dimensions: instruction similarity (top), UI similarity (middle), and action similarity (bottom). These distributions enable systematic analysis of how different types of similarity between demonstration and query tasks affect learning efficacy.

### B LEARNACT FRAMEWORK DETAILS

This section provides detailed descriptions of the components of our LearnAct framework, corresponding to the methods presented in Section 4 of the paper.

### B.1 DemoParser Prompts

We provide all of our prompt templates used in DemoParser for generating semantically descriptive action descriptions from demonstration data. These carefully designed prompts guide the vision-language model to produce structured knowledge that captures the essence of human demonstrations, as shown in Figures 9 and 10.

Prompt 1: Intermediate Action Description

System Prompt:

You are a mobile UI interaction analyst. Follow these rules: 1. Analyze the split-screen image (Before-action left, After-action right) 2. For click actions, a high-contrast red marker (white-bordered circle) shows the precise click location, with a green square surrounding it and a ’C’ label at the top-right corner of the square indicating the click. 3. Output JSON with ONLY ONE ’action_description’ field in this exact format: "[On/In] [Screen Name], [Action Details], to [Purpose]"

Action Types: - click [element] (e.g., ’Search button’) - swipe [up/down/left/right] - type [text] in [field] press [back/home/enter] Validation Rules: 1. Screen names should be 2-6 words 2. Keep purpose descriptions under 8 words 3. Never mention coordinates/IDs MEMORY RECORDING RULES: If the current screen contains information relevant to the user’s instruction that needs to be remembered for future steps, include a Memory part in your action description. The format should be: "[On/In] [Screen Name], [Action Details], to [Purpose]. [Memory: important information for future steps]" Memory should ONLY be added when: 1. The information is relevant to completing the user’s instruction 2. The information will likely be needed in future steps 3. This specific information has NOT been recorded in previous action history entries

Memory examples: 1. For a travel planning task: On Travel Blog, click ’Bali Beach Guide’, to read article. [Memory: Guide mentions Kuta Beach has surfing lessons for $25/hour] 2. For a shopping task: In Product Details, click ’Add to Cart’, to select item. [Memory: iPhone 13 Pro costs $999 with 128GB storage] 3. For a note-taking task: On Weather App, swipe down forecast, to view weekend. [Memory: Saturday will be rainy with 80% precipitation]

Avoid using Memory for: 1. Obvious UI changes that don’t contain task-relevant information 2. Information already captured in previous action steps 3. Generic observations not specific to the user’s task objective

- Figure 9: Prompt template for intermediate action descriptions. The template guides DemoParser to generate standardized descriptions for intermediate actions, including detailed rules for memory annotations that capture important information observed during task execution.

### B.2 ActExecutor Prompts

We provide the prompt templates used by ActExecutor to make decisions based on current observations, action history, and demonstration knowledge. These prompts guide the vision-language model to select appropriate actions for task execution, as shown in Figure 11.

### B.3 Algorithm Details

We provide the detailed algorithms for the DemoParser and ActExecutor components of our LearnAct framework, which are the core computational processes enabling knowledge extraction and task execution.

- C ADDITIONAL EXPERIMENTAL RESULTS AND ANALYSES This section provides additional experimental results and analyses that supplement the findings presented in Section 5 of the paper.

### C.1 Online Performance Comparisons

Figures 12 and 13 provide detailed comparisons of model performance with and without LearnAct enhancement in online evaluation scenarios.

### C.2 Case Studies of LearnAct Online Experiments

We present three detailed case studies from our online experiments to provide concrete examples of how LearnAct leverages demonstration knowledge to solve tasks in unseen mobile applications. These case studies highlight different scenarios where demonstration knowledge proves particularly beneficial for task execution.

###### Prompt 2: Terminal Action Description - Standard Completion

System Prompt for standard completion: Determine the final task status. Output rules: 1. Use ONLY ONE ’action_description’ field 2. Format: "[On/In] [Screen], complete task, [Reason]" Validation Rules: - Reason should be less than 10 words - Screen name must match previous context Examples: 1. Basic completion: On Payment Screen, complete task, successfully submit order 2. Failure case: In Search Results, cannot complete task, no nearby Vivo mobile phone stores found

###### Prompt 3: Terminal Action Description - With Answer

System Prompt for completion with answer: Determine the final task status with the given answer. Output rules: 1. Use ONLY ONE ’action_description’ field 2. Format: "[On/In] [Screen], complete task, the answer is [answer]" Validation Rules: - Screen name must match previous context - Use the exact answer provided in the TASK_COMPLETE action

Examples: 1. Answer is a price: On Checkout Screen, complete task, the answer is "$299.9". 2. Answer is a list: On Payment Options Screen, complete task, the answer is "google pay, check out with affirm, add credit/debit card".

- Figure 10: Prompt templates for terminal action descriptions. The templates provide specific formats for both standard task completion and information retrieval tasks, ensuring consistent output structure across different task types.

- Algorithm 1 DemoParser Knowledge Generation Process

Require: Demonstration dataset 𝐷 = {(𝑖𝑘,𝑠𝑘,𝑎𝑘)𝑘𝑁=1} where 𝑖𝑘 is instruction, 𝑠𝑘 is screenshot sequence, 𝑎𝑘 is action sequence Ensure: Knowledge base 𝐾 with semantically descriptive action descriptions

- 1: 𝐾 ← ∅ ⊲ Initialize empty knowledge base
- 2: for each demonstration (𝑖,𝑠,𝑎) in 𝐷 do
- 3: 𝑑 ← ∅ ⊲ Initialize empty description sequence
- 4: for 𝑗 = 1 to |𝑎| do
- 5: if 𝑗 < |𝑎| then ⊲ Intermediate action
- 6: Create visualization of action 𝑎𝑗 with before-after screenshots from 𝑠𝑗 and 𝑠𝑗+1
- 7: ℎ ← Previous action descriptions {𝑑1,𝑑2, . . .,𝑑𝑗−1}
- 8: 𝑑𝑗 ← GenerateDescription(𝑖,𝑎𝑗,𝑣𝑖𝑠𝑢𝑎𝑙𝑖𝑧𝑎𝑡𝑖𝑜𝑛,ℎ) using prompt format detailed in Appendix B.1
- 9: 𝑑𝑗 follows format: "[On/In] [Screen], [Action], to [Purpose]" with optional memory
- 10: else ⊲ Terminal action
- 11: ℎ ← Complete action history {𝑑1,𝑑2, . . .,𝑑|𝑎|−1}
- 12: 𝑑|𝑎| ← GenerateFinalDescription(𝑖,𝑠|𝑎|,ℎ,𝑎|𝑎|) using prompt detailed in Appendix B.1
- 13: 𝑑|𝑎| follows format: "[On/In] [Screen], complete task, [Reason/Answer]"
- 14: end if
- 15: Add 𝑑𝑗 to description sequence 𝑑
- 16: end for
- 17: Add (𝑖,𝑎,𝑑) to knowledge base 𝐾
- 18: end for
- 19: return 𝐾

###### Prompt 4: Task Execution Prompt

Role Definition: You are a smartphone assistant to help users complete tasks by interacting with apps. I will give you a screenshot of the current phone screen. Example Tasks: [Only when demonstrations are available] Example 1: [Demonstration instruction] Steps taken in this example: Step-1: [Action] [Action Description] Step-2: [Action] [Action Description] ... Background: This image is a phone screenshot. Its width is [width] pixels and its height is [height] pixels. The user’s instruction is: [instruction] History operations: [Only when action history is available]

Before reaching this page, some operations have been completed. You need to refer to the completed operations to decide the next operation. These operations are as follow: Step-1: [Action] [Action Description] Step-2: [Action] [Action Description] ...

Response requirements: Now you need to combine all of the above to decide just one action on the current page. You must choose one of the actions below:

"SWIPE[UP]": Swipe the screen up. "SWIPE[DOWN]": Swipe the screen down. "SWIPE[LEFT]": Swipe the screen left. "SWIPE[RIGHT]": Swipe the screen right. "CLICK[x,y]": Click the screen at the coordinates (x, y). x is the pixel from left to right and y is the pixel from top to bottom "TYPE[text]": Type the given text in the current input field. "PRESS_BACK": Press the back button. "PRESS_HOME": Press the home button. "PRESS_ENTER": Press the enter button. "TASK_COMPLETE[answer]": Mark the task as complete. If the instruction requires answering a question, provide the answer inside the brackets. If no answer is needed, use empty brackets "TASK_COMPLETE[]".

Response Example: Your output should be a string and nothing else, containing only the action type you choose from the list above. For example: "SWIPE[UP]" "CLICK[156,2067]" "TYPE[Rome]" "PRESS_BACK" "PRESS_HOME" "PRESS_ENTER" "TASK_COMPLETE[1h30m]" "TASK_COMPLETE[]"

- Figure 11: Task execution prompt template. This comprehensive prompt directs ActExecutor to generate actions based on current observations, action history, and retrieved demonstrations, with explicit formatting requirements to ensure consistent action outputs.

- Algorithm 2 ActExecutor Task Execution Process

Require: User instruction 𝑖, Knowledge base 𝐾, Maximum steps𝑇 Ensure: Task execution trajectory

- 1: 𝑡 ← 0 ⊲ Initialize time step
- 2: ℎ ← ∅ ⊲ Initialize action history
- 3: D ← KnowSeeker(𝑖,𝐾) ⊲ Retrieve relevant demonstrations
- 4: while 𝑡 < 𝑇 and not IsTaskComplete do
- 5: 𝑜𝑡 ← GetObservation() ⊲ Obtain current screenshot
- 6: 𝑃𝑡 ← ConstructPrompt(𝑖,𝑜𝑡,ℎ, D) ⊲ Construct decision prompt
- 7: 𝑎𝑡 ← 𝑓𝐿𝐿𝑀 (𝑃𝑡) ⊲ Generate action via LLM
- 8: 𝑑𝑡 ← GenerateDescription(𝑖,𝑎𝑡,𝑜𝑡,ℎ) ⊲ Generate action description
- 9: ℎ ← ℎ ∪ {(𝑎𝑡,𝑑𝑡)} ⊲ Update action history
- 10: ExecuteAction(𝑎𝑡) ⊲ Execute action in environment
- 11: 𝑡 ← 𝑡 + 1 ⊲ Increment time step
- 12: end while
- 13: return {(𝑎0,𝑑0), (𝑎1,𝑑1), . . ., (𝑎𝑡−1,𝑑𝑡−1)}

Qwen2-VL-7B Task Performance Comparison

0-shot

100

LearnAct

+14.3%

80

SuccessRate(%)

60

+14.3%

40

+26.3%

+8.4% +13.7%

20

+11.5%

+14.3%

+5.2%

+8.3%

0

ComplexUIUnderstanding DataEdit DataEntryInformationRetrieval MultiAppParameterized RepetitionRequiresSetupScreenReading Search Transcription Untagged Verification

Task Categories

- Figure 12: Detailed performance comparison of Qwen2-VL-7B with and without LearnAct on LearnGUI-Online. The figure shows the task success rates of Qwen2-VL-7B baseline versus Qwen2-VL-7B enhanced with LearnAct across different task dimensions in the LearnGUI-Online benchmark.

ComplexUIUnderstanding DataEdit DataEntryInformationRetrieval MultiAppParameterized RepetitionRequiresSetupScreenReading Search Transcription Untagged Verification

Task Categories

0

20

40

60

80

100

SuccessRate(%)

+5.5%

+21.1%

+4.4%

+5.3%

+6.3%

+11.0%

+8.3%

+25.0% +9.1%

+12.5%

+21.4%

+14.3%

UI-TARS-7B-SFT Task Performance Comparison

0-shot

LearnAct

- Figure 13: Detailed performance comparison of UI-TARS-7B-SFT with and without LearnAct on LearnGUI-Online. The figure presents a comprehensive breakdown of task success rates for UI-TARS-7B-SFT baseline versus UI-TARS-7B-SFT enhanced with LearnAct across multiple task dimensions in the LearnGUI-Online benchmark.

[Figure 147]

##### Baseline Failed

<instruction> What quantity of acai berries do I need for the recipe 'Tacos' in the Joplin app? Express your answer in the format <amount> <unit> where both the amount and unit exactly match the format in the recipe.

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

##### Support Mandate Execution

<instruction> What quantity of almond flour do I need for the recipe 'Tacos' in the Joplin app? Express your answer in the format <amount> <unit> where both the amount and unit exactly match the format in the recipe.

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

[Figure 172]

[Figure 173]

1/2 cup

Tacos

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

##### LearnAct Successed

<instruction> What quantity of buckwheat groats do I need for the recipe 'Tacos' in the Joplin app? Express your answer in the format <amount> <unit> where both the amount and unit exactly match the format in the recipe.

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

2 teaspoons

Tacos

[Figure 190]

- Figure 14: UI-TARS-7B-SFT with LearnAct vs. Baseline in NotesRecipeIngredientCount Task. Task template: "What quantity of {ingredient} do I need for the recipe ’{title}’ in the Joplin app? Express your answer in the format <amount> <unit> without using abbreviations."

[Figure 191]

###### Baseline Failed

<instruction> In Simple Calendar Pro, delete the calendar event on 2023-10-31 at 13h with the title 'Review session for Annual Report'

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

###### Support Mandate Execution

<instruction> In Simple Calendar Pro, delete the calendar event on 2023-10-26 at 17h with the title 'Workshop on Annual Report'.

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

[Figure 231]

###### LearnAct Successed

<instruction> In Simple Calendar Pro, delete the calendar event on 2023-10-25 at 0h with the title 'Review session for Annual Report'.

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

- Figure 15: UI-TARS-7B-SFT with LearnAct vs. Baseline in SimpleCalendarDeleteOneEvent Task. Task template: "In Simple Calendar Pro, delete the calendar event on {year}-{month}-{day} at {hour}h with the title ’{event_title}’"

[Figure 245]

###### Baseline Failed

<instruction> Delete the following expenses from pro expense: Undergarments, Event Tickets, Streaming Services.

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

[Figure 260]

[Figure 261]

###### Support Mandate Execution

<instruction> Delete the following expenses from pro expense: Mortgage, Concert Tickets, Home Insurance.

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

###### LearnAct Successed

<instruction> Delete the following expenses from pro expense: Museum Tickets, Rent Payment, Health Insurance.

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

###### Figure 16: Qwen2-VL-7B with LearnAct vs. Baseline in ExpenseDeleteMultiple Task. Task template: "Delete the following expenses from arduia pro expense: {expenses}."

23

