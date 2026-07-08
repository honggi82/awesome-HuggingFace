# arXiv:2602.11103v1[cs.AI]11Feb2026

## GameDevBench: Evaluating Agentic Capabilities Through Game Development

Wayne Chi1, Yixiong Fang1, Arnav Yayavaram1, Siddharth Yayavaram1, Seth Karten2, Qiuhong Anna Wei1, Runkun Chen1, Alexander Wang1, Valerie Chen1,

Ameet Talwalkar1, and Chris Donahue1 1Carnegie Mellon University 2Princeton University

Abstract

Despite rapid progress on coding agents, progress on their multimodal counterparts has lagged behind. A key challenge is the scarcity of evaluation testbeds that combine the complexity of software development with the need for deep multimodal understanding. Game development provides such a testbed as agents must navigate large, dense codebases while manipulating intrinsically multimodal assets such as shaders, sprites, and animations within a visual game scene. We present GameDevBench, the first benchmark for evaluating agents on game development tasks. GameDevBench consists of 132 tasks derived from web and video tutorials. Tasks require significant multimodal understanding and are complex—the average solution requires over three times the amount of lines of code and file changes compared to prior software development benchmarks. Agents still struggle with game development, with the best agent solving only 54.5% of tasks. We find a strong correlation between perceived task difficulty and multimodal complexity, with success rates dropping from 46.9% on gameplay-oriented tasks to 31.6% on 2D graphics tasks. To improve multimodal capability, we introduce two simple image and video-based feedback mechanisms for agents. Despite their simplicity, these methods consistently improve performance, with the largest change being an increase in Claude Sonnet 4.5’s performance from 33.3% to 47.7%. We release GameDevBench publicly to support further research into agentic game development.

Project Page https://waynechi.com/gamedevbench Github Repo https://github.com/waynchi/gamedevbench

### 1 Introduction

Progress on multimodal language model (LM) agents has lagged behind that of their unimodal counterparts [39, 13, 43, 17]. Agentic game development—despite its inherent multi-modality, increasing public interest, and a rich history combining artificial intelligence and games [33, 23, 27, 26, 12, 10, 37]—has largely been overlooked by the research community. Most prior works focus on specific goals within game development such as next frame prediction [30, 21], which replaces the graphics engine, procedural content generation [29, 24], which replaces both functional and cosmetic asset creation, or game playing agents [33, 26], which replaces the non-player characters (NPCs) and opponents. There has been little to no research on agentic use for general game development (i.e.,

Correspondence to waynechi@andrew.cmu.edu

[Figure 1]

- Figure 1: We present GameDevBench, a benchmark for evaluating an agent’s ability to solve complex and multimodal game development tasks in a modern game engine.

developing games within a game engine), most likely because it seemed inconceivable until recently. As LM agent capabilities continue to improve, it seems natural to ask: can agents develop video games?

Game development combines many desirable characteristics for a challenging benchmark in a modern agentic domain. First, tasks are complex and context-rich with projects often spanning large amounts of files, assets, and folders akin to that of traditional software development [39]. Second, tasks are inherently multimodal, requiring visual understanding of both static elements (e.g., map or scene layouts) and temporal dynamics (e.g., animations or movement) to accurately assess project state. Lastly, task solutions are deterministically verifiable through code which alleviates the need for approaches such as LLM-as-a-Judge [42] which are often subject to biases [34, 18]. For example, it is possible to verify that the correct animation was used by checking animation states at each frame. This combination of features makes game development an ideal environment to evaluate complex, multi-modal agentic capabilities.

In this work, we study an agent’s ability to solve complex game development tasks for a modern game engine. To our knowledge, this is the first work evaluating this capability. Game development typically involves creating and editing artifacts such as sprite sheet animations, collision shapes, game logic scripts, and scene layouts in a GUI (Graphical User Interface) called the game editor. A game engine then processes these artifacts into a runnable game build. Common examples of game engines include Unity, Unreal Engine, and Godot, each of which provides both an editor and an engine. Game development tasks are deceptively complex. For example, the “simple” task of creating an Italian plumber for a platformer game would require creating animations for various states such as idling, jumping, or running, setting up a collider to allow for jumping on enemies such as turtles, writing scripts to allow for control, adding sound effects for actions, and more.

We focus our work on the Godot environment for several reasons. First, Godot is fully open sourced under the MIT license which makes it easy to extend and release alongside the benchmark. Second, Godot is an increasingly popular game development engine, with 770 and 1185 releases on

Steam in 2024 and 2025. Third, Godot’s environment strongly resembles Unity, which is by far the most popular game development engine. Lastly, Godot projects (not including assets such as images) can be represented in code which makes it simple to extend existing LLM agent capabilities without having to construct specific tool-use APIs.

We present GameDevBench, the first benchmark for evaluating an agent’s ability to solve game development tasks. Tasks are created by analyzing and processing Godot YouTube and web tutorials. These tutorials span a wide range of topics such as 2D sprite animations, character controllers (i.e., character movement), colliders and platforms, shader usage, particle effects, among others. This ensures that tasks are not only diverse, but also align with common game development needs. Tasks are incredibly complex and content-rich. Not only do they require a deep understanding of various file types and assets (e.g., images), tasks on average require more than three times the number of lines of code changes compared to SWE-Bench [39]. For each task, agents are given a project folder with code and various assets, as well as an instruction as is standard in software benchmarks [39]. Task success is evaluated using tests built within Godot’s scripting framework. This allows us to deterministically test for features such as physics or polygonal shapes. Additionally, each task comes with a verified reference solution. All code and task project files for GameDevBench are released publicly.

We found that while agents are increasingly capable, they still struggle with the majority of game development tasks. Without additional support, the best agent succeeds at only 47.0% of the tasks. In particular, models perform significantly worse when the tasks require increased multimodal understanding. For example, agents perform almost two times better when tackling gameplay-oriented tasks compared to graphics tasks.

To improve agent multimodal capabilities, we propose two methods that provide agents with multi-modal feedback when solving a task. One method provides a screenshot view of the editor’s current state via a Model Context Protocol (MCP) server [1] while another records a video of the game scene. Despite their simplicity, we found that both methods are effective empirically, increasing agent performance across almost all models.

### 2 Benchmark Construction

GameDevBench consists of game-development tasks distilled from online tutorials (e.g., “add a walking animation using the given spritesheet”). GitHub repositories are a rich source of data, but can be noisy and poorly documented. Additionally, unlike prior work [13] on benchmarking general software development, there are no obvious popular open source game repositories to choose from. The game development community, however, has created an abundance of online tutorials—many of which come with solution repositories—that guide developers through common development use cases. We use a multi-step pipeline to construct game development tasks using these tutorials.

#### 2.1 Stage 1: Data Preparation

Game development tutorials primarily come in either text or video formats. For all tutorials, we search for and filter to only include Godot 4 tutorials that include a corresponding GitHub repository with permissive, open-source licenses.

Video. We source our video tutorials from YouTube. To convert video into text, we use a popular YouTube transcription API1 to extract the text transcript from each video. To search for a matching GitHub repository, we parse the video description for any GitHub links. The final result is a folder

1https://github.com/jdepoix/youtube-transcript-api

[Figure 2]

- Figure 2: This is an example task from GameDevBench that requests for the creation of a UI minimap. Top is the visual GUI representation and highlighted points of interest. Bottom is the same scenes and files represented in code. Tasks can be solved via the editor or entirely through code although either method requires understanding multimodal assets. Game development tasks are complex and require editing dense files, identifying and visually understanding various assets, and navigating various nodes (game elements) and scenes (a collection of nodes).

for each tutorial containing the transcript and the corresponding GitHub repository. We process 102 video tutorials which were selected based on view count. Each tutorial averages 29 minutes of content. In the end, we use 57 tutorials as not all tutorials are usable due to non-functional repositories or mislabeled Godot versioning.

Web. For web text tutorials, we source from “Godot Recipes by KidsCanCode” [16], which is listed on the community resources page in Godot’s official documentation. We scrape the webpages using a Python script with the goal of mirroring the structure of processed video tutorial folders. The end result is 99 tutorial folders, each of which contains the tutorial text content, a media directory of visual data downloaded from the webpage, a GitHub repository, as well as a metadata JSON containing information such as the tutorial URL. Finally, we ask an LLM to sort tutorials based on suitability for task creation and use the top 31 tutorials for subsequent task construction.

#### 2.2 Stage 2: Automatic Task Construction

Given the tutorial folder, the agent is asked to create tasks where 1) instructions adhere to the tutorial, 2) task files are created directly based on existing files in the repository, and 3) unit tests must only test for features explicitly requested in the instructions. Access to the solution repository is crucial as it allows the agent to create tasks that it would not normally have the capability to

solve or create. At the agent’s discretion, each tutorial is split into multiple tasks to capture more well-defined skills. For example, the agent could decompose a platformer tutorial into tasks on character animation, controls and colliders, and tilemap construction. We use the Codex Agent with the GPT-5 family of models to construct tasks from each tutorial. Codex was chosen primarily due to its API limits and availability at the time; we did not notice any significant differences between agents such as Claude Code. We create 202 initial tasks with an average of 1.3 tasks per tutorial. The full prompt can be found in Appendix A.

#### 2.3 Stage 3: Task Refinement

After stage 2, we found the majority of tasks to be sensible at a high level (i.e., task instructions were reasonable and matched the tutorial). However, similar to prior work [7], the agent was not able to perfectly create tasks and tests. We conducted a preliminary study on a small subset of 41 tasks where human annotators reviewed tasks and documented any issues observed. The study found that 43% of tasks were issue free, 50% of tasks had issues that required minor updates such

- as scenes being off-camera, tests asserting for non-existent instructions, or accidental references to other portions of the tutorial, and 7% of tasks contained major issues that made them difficult to fix. Since most of the errors were minor and easily caught, we employed a hybrid process to refine tasks. Based on the preliminary study, we constructed prompts and checklists to catch the most common mistakes. We then employed an agent to automatically verify and fix those mistakes based on the checklists. We re-use this prompt (Appendix B) when processing all future tutorials and tasks.

#### 2.4 Stage 4: Human Annotation.

Lastly, 8 human annotators, 5 of whom have prior game development experience, reviewed all tasks. Annotation served two primary goals. First is to ensure that tasks are verified for correctness and resolvability as is common practice [39, 7]. Annotators are instructed to look for and fix any ambiguous instructions, conflicting instructions, and overly strict tests. Additionally, annotators are asked to mark and remove any tasks that had any other issues. Second, we ask annotators to create variations of existing tasks similar to prior work [43]. An example would be two tasks that differ based on the requested animation used in a spritesheet (e.g, selecting the walking vs running animation frames). In total, we create 115 base tasks and 17 task variants. Annotation instructions can be found in Appendix C.

### 3 GameDevBench

Game development sits at the intersection of creative expression and software development. As such, GameDevBench features a diverse set of tasks that are inherently multi-modal, complex and context-rich.

#### 3.1 Task Categories

To our knowledge, there is no existing taxonomy of game development tasks performed within a game editor or game engine. To better understand our task diversity and enable deeper analysis, we categorize each task along two axes.

Categorization by skill set. We induce a task categorization based on the underlying game development skills required by each task (Table 5). Specifically, we adopt a bottom-up categorization procedure: we first obtain fine-grained skill annotations for each task by asking GPT-5-mini to label

Table 1: Skill categories for Godot-related development tasks.

Skill Definition Examples % Tasks

Tasks focused on implementing game rules and behaviors such as motion and collisions in Godot.

Enemy AI states, Signal-driven events, Collision detectors, Character controllers

Gameplay Logic

35.6%

Tasks involving the construction, rendering, and animation of threedimensional scenes in Godot.

- 3D Graphics and Animation

Material tuning, Skeletal animation, Camera rigs

25.7%

Tasks involving the creation, rendering, and animation of two-dimensional visual content in Godot.

2D Graphics and Animation

Sprite animation, TileMap setup, 2D shader effects

19.7%

Tasks concerning the design and implementation of interactive user interfaces in Godot.

HUD layout, Menu navigation, UI theming

15.9%

User Interface

each task. Labels are then abstracted into higher-level skill categories through a separate request to GPT-5-mini. These categories are subsequently reviewed and refined by game developers to ensure consistency and validity. This process yields four skill categories: 2D graphics and animation, 3D graphics and animation, gameplay logic, and user interface.

Categorization by editor type. Godot contains several different types of editors that users use to resolve various tasks. There are three main types of editors within Godot. The scene editor

- (Figure 3, top-left) allows the user to modify the game scene by constructing level maps or placing and editing objects. The script editor (Figure 3, top-right) is a built-in code editor. Contextual editors appear on the bottom panel depending on what the user is editing (Figure 3, bottom). For example, when editing an animation resource, the animation editor will appear. Some of the most common contextual editors include the animation, audio, shader, and tileset editors. We categorize each task in the benchmark by asking GPT-5-mini to determine the editors that a user would need to solve the task. While the agent may not directly interact with these editors, the type of editor a user would use provides a strong proxy for task categorization.

For simplicity, we assign each task one skill category and one editor type. We provide multiple examples with their skill category and editor type in Appendix E.

3.2 Features of GameDevBench

GameDevBench has a unique set of features which we describe as follows. We provide additional task statistics in Appendix F

Diverse file types across tasks. Unlike agentic benchmarks in the software domain [13], GameDevBench requires that agents handle a wide variety of filetypes across various modalities

- (Figure 4, left). In fact, the vast majority of tasks (82.4%) contain additional assets such as images (.png), text fonts (.ttf), shaders (.gdshader), audio (.wav), and other asset resources (.tres). As such, GameDevBench inherently tests the multimodal capabilities of agents.

Diverse task types. While there are other domains (such as frontend development [44, 25] or slide generation [6]) that intersect multimodality and code generation, most of these domains focus on tasks similar to user interface generation. On the other hand, GameDevBench features a diverse task set. Across the 132 benchmark tasks, domains are distributed as follows: (35.6%) Gameplay Logic, (25.7%) 3D Graphics and Animation, (19.7%) 2D Graphics and Animation, and (15.9%) User

[Figure 3]

- Figure 3: Types of editors in Godot. Top-left is the scene editor. Top-right is the script editor. The bottom contains various contextual editors. From left to right: tilemap, shader, animation, and audio editors. Contextual editors surface depending on use case. Typically, tasks that use contextual editors require deeper multi-modal understanding.

Interface.

Complex and context-rich solutions. Similar to software tasks, GameDevBench solutions require multi-location edits that weave together multiple files. Our reference solutions average 5 files and 106.2 lines of code changed across 3.4 distinct filetypes (Figure 4, right). This is more than triple the number of lines of code and file changes required compared to SWE-Bench [13], suggesting substantial complexity to the tasks and corresponding solutions.

Deterministic verification of multimodal solutions. Evaluating multimodal solutions is inherently challenging and solutions are typically evaluated through metrics such as CLIP [22] or a Visual LLM-as-a-Judge [41]. These methods are, however, either proxies to correctness or non-deterministic. GameDevBench instead uses Godot’s testing framework which allows us to directly test game behavior. For example, we can check to see if objects are in view of the camera or if object colliders have interacted purely through unit tests. Thus, tests are repeatable and verifiable similar to software benchmarks while testing multimodal problems.

Flexible solution methods. While the tests are deterministic, the methods used to derive solutions are flexible. In this work, for simplicity, we evaluate agents that attempt to solve tasks through code generation alone. However, it would be equally feasible to solve each task directly in the editor with approaches more similar to computer use. Our test-based verification allows for direct comparison of different solution strategies.

###### Task Assets

###### Mean Max

###### # File Types

# Files 74.3 1929 # Filetypes 6.4 18 # Lines of Code 500.5 20072 # Nodes 17.8 982

8200

| || |
|---|
<br><br>png tscn uid gd<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>tres svg wav (21 others)<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|---|
| | |
| | |
| | |
| | |
| | |
| | |

15.3%

17.6%

8000

Overview

7800

8.4%

7600

TotalCount

# Files Edited 5.0 17 # Filetypes Edited 3.4 6 # Lines Edited 106.2 1948 # Nodes Edited 3.3 24

7400

16.0%

7200

Gold Patch

42.7%

| | |
|---|---|
| | |
| | |

400

# Images 60.3 1920 Image Size (px) 121.0K 16.8M

200

No Assets: 23 Images Only: 56 Resources Only: 21

Both: 11 More than 2 Asset Types: 20

Images

| |
|---|

| |
|---|

0

| |
|---|

| |
|---|

| |
|---|

- Figure 4: GameDevBench features a diverse amount of filetypes (27 different types, left). The vast majority of tasks contain either images, resources (e.g., Shaders), or multiple asset types (middle). Each task contains multiple scripts and scenes, both of which are context-rich and require a significant amount of tokens to process (right).

Continually Renewable. While not unique to our benchmark, our pipeline is repeatable and thus the benchmark can be continuously renewed. Human validation is minor with each task taking under 10 minutes to validate.

### 4 Evaluation

We evaluate various models and agentic frameworks on GameDevBench.

Model Choices. From the Claude family of models we evaluate Claude Haiku 4.5, Claude Sonnet 4.5, and Claude Opus 4.5. From the Gemini family we evaluate Gemini 3 Flash and Gemini 3 Pro. We evaluate ChatGPT Codex 5.1 from the ChatGPT family. For open weights models we evaluate Qwen3-Vl-235B-Instruct and Kimi K2.5.

Agent Framework Choices. To allow agents access to both the project files and the Godot application itself, we focus on agentic frameworks that operate locally. We chose command-line interface (CLI) agentic frameworks due to their ability to directly read code, image, and other asset files. We evaluate each model in its respective agentic frameworks—claude-code for Claude models, gemini-cli for Gemini models, and codex for ChatGPT models. We evaluate Kimi K2.5 and Qwen3-Vl using OpenHands [35]. We also evaluate Claude Sonnet 4.5, Gemini 3 Flash, and ChatGPT Codex 5.1 using OpenHands to compare performance across frameworks.

#### 4.1 Multimodal Feedback

Here we outline two tooling configurations that allow agents to access richer multimodal information from Godot through editor screenshots and/or rendered video.

Baseline. As a baseline, each agent starts inside the project directory and is given the task instruction along with basic instructions on how to run Godot. We provide additional methods to support the agent, primarily to observe if additional visual context improves performance. We provide our full prompts in Appendix D.

Editor Screenshot MCP. We develop an MCP server that loads the Godot editor for the current task, takes a screenshot of the editor, then returns the image to the agent. This allows the agent to view the game scene, the node tree, the node inspector, as well as other information present in the editor. This method allows the agent to leverage additional visual feedback to validate its solution.

Runtime Video. We provide agents with instructions on how to generate gameplay videos using Godot’s built-in recording functionality, which is otherwise frequently ignored or misused. This differs from the MCP server as it captures both a) temporal elements only present in video and b) the current camera view (the editor does not show the camera view). Typically, models process videos into image frames using python rather than ingesting the video directly.

#### 4.2 Discussion of Results

Skills

Editors

100

100

Gemini 3 Pro Gemini 3 Flash Opus 4.5 Sonnet 4.5

Haiku 4.5 Codex 5.1 Kimi k2.5

| |
|---|

| |
|---|

| |
|---|

80

80

| |
|---|

| |
|---|

SuccessRate(%)

| |
|---|

60

60

40

40

20

20

0

0 Script Contextual Scene

Gameplay 3D Graphics User Interface

2D Graphics

Figure 5: In general, agents perform better on tasks that require skills focusing on gameplay functionality compared to tasks that require multimodal understanding such as 2D and 3D graphics tasks. Performance on editor categories is dependent on model performance. Stronger models (left 4 agents) tend to perform similarly across all editor types, while weaker models (right 3 agents) tend to perform worse on tasks requiring the scene and contextual editors. All success rates are taken from results where the agent has access to multimodal feedback.

We now discuss our findings from evaluating agents on GameDevBench (Table 2).

Game development proves challenging to even the most capable models and performance rapidly degrades when moving further from the frontier. The largest models from three different commercial model families (GPT, Claude, and Gemini) achieve baseline performances of 34.1%, 39.4%, and 46.2% respectively without additional multimodal feedback in their native agentic framework. Performance significantly degrades as we move further from the frontier. Claude Haiku 4.5 solves 22.0%, Kimi K2.5 solves 34.1%, and Qwen3-Vl-235B-Instruct proves almost completely incapable, solving only 8.3% of tasks. In contrast, Qwen3-Vl-235B-Instruct solves 92% of tasks in the frontend benchmark Design2Code [25].

Agent performance differs significantly across skill and editor categories. We observe a general trend where agents perform worse on tasks that are more multimodally demanding (Figure 5). For skills, agents perform best at gameplay tasks (average success rate of 46.9%) and perform worst at 2D graphics and animation (average success rate of 31.62%) which require agents to understand images or other assets for animations and effects. Performance across editor categories is instead dependent on model capabilities. The four models that are best overall—Gemini 3 Pro, Gemini

###### 3 Flash, Claude Opus 4.5, and Claude Sonnet 4.5—perform similarly on tasks regardless of the required editor type. However, Claude Haiku 4.5, ChatGPT Codex 5.1, and Kimi K2.5 perform worse on tasks requiring the scene and contextual editors which are typically more multimodally demanding compared to scripting tasks.

Multimodal tooling consistently improves agent performance. We find that providing an agent with either the MCP or video instructions improves performance. This trend holds across almost all models (with the exception of Claude Haiku 4.5 where performance remains similar). However, the best performing method differs between models. For example, Gemini 3 Flash benefits

Table 2: Results from evaluating various models and agent frameworks on GameDevBench. Screenshot indicates that the agent was given access to an MCP server that screenshots the editor state. Video indicates that the agent was given additional instructions on how to generate a video of the current game scene. Bold and italics indicate the best and second best model performance.

Framework Model w/ Screenshot w/ Video pass@1 (%)

× × 22.0 × ✓ 25.8 ✓ × 21.2 ✓ ✓ 24.2

claude-haiku-4-5-20251001

× × 33.3 × ✓ 47.7 ✓ × 33.3 ✓ ✓ 46.2

claude-code

claude-sonnet-4-5-20250929

× × 39.4 × ✓ 47.0 ✓ × 41.7 ✓ ✓ 50.0

claude-opus-4-5-20251101

× × 34.1 × ✓ 41.7 ✓ × 34.1 ✓ ✓ 41.7

codex gpt-5.1-codex-max

× × 47.0 × ✓ 48.5 ✓ × 50.8 ✓ ✓ 52.3

gemini-3-flash-preview

gemini-cli

× × 46.2 ✓ ✓ 54.5

gemini-3-pro-preview

× × 23.5 ✓ ✓ 31.3

claude-haiku-4-5-20251001

× × 43.2 ✓ ✓ 51.1

claude-sonnet-4-5-20250929

× × 36.4 ✓ ✓ 40.5

gemini-3-flash-preview

× × 45.5 ✓ ✓ 49.6

openhands

gpt-5.1-codex-max

× × 34.1 ✓ ✓ 38.9

kimi-k2.5

× × 8.3 × ✓ 8.3 ✓ × 7.6 ✓ ✓ 5.3

qwen3-vl-235b-a22b-instruct

from MCP, improving from 47.0% to 50.0% compared to 48.5% when using video. On the other

70%

60%

gemini-cli gemini-3-flash-preview

openhands

openhands claude-sonnet-4-5-20250929 gpt-5.1-codex-max

50%

###### SuccessRate

openhands kimi-k2.5

claude-code claude-sonnet-4-5-20250929

40%

claude-code claude-haiku-4-5-20251001

30%

claude-code claude-sonnet-4-5-20250929

claude-code claude-opus-4-5-20251101

openhands claude-haiku-4-5-20251001

openhands qwen3-vl-235b-a22b-instruct

20%

openhands claude-haiku-4-5-20251001

openhands claude-sonnet-4-5-20250929

Baseline

openhands gpt-5.1-codex-max

10%

Video

openhands gemini-3-flash-preview

Screenshot

openhands kimi-k2.5

Both

gemini-cli gemini-3-flash-preview

0%

$0.00 $0.20 $0.40 $0.60 $0.80 $1.00 $1.20 $1.40 $1.60

Median Cost per Task (USD)

- Figure 6: We capture the trade-off between performance and cost. In general, using multimodal feedback increases cost per task while increasing performance. Agents above the linear-fit outperform the average cost-to-success ratio. We find gemini-3-flash-preview to be the most cost-effective model. Since gemini-cli does not return the full agent trajectory, we use the OpenHands cost which is likely an overestimation as OpenHands is typically more costly.

hand, Claude Sonnet 4.5 improves from 33.3% to 41.7% using video while seeing no improvement when using the MCP. Using both MCP and video provides negligible additional benefit, achieving similar performance as the best of either method. Although all tasks can be verified through code, visual feedback allows agents to verify and amend mistakes. This behavior strongly resembles that seen in recent work [41], where visual feedback improves agentic performance.

Agentic framework choice can have significant impact on performance, but the effect varies depending on the model. We evaluate Claude Sonnet 4.5, Gemini 3 Flash, and GPT 5.1 Codex using both their original frameworks and OpenHands (Table 2). We observe that both Claude Sonnet 4.5 and GPT 5.1 Codex increase performance from 33.3% to 43.2% and 34.1% to 45.5% respectively, when switching from their native frameworks to OpenHands. On the other hand, Gemini 3 Flash’s performance decreases from 47.0% to 36.4%, rendering it the best of the three models in their native agentic frameworks and the worst in OpenHands. This is likely due to incompatible editing tools between Gemini models and OpenHands 2

Cost varies significantly depending on the model, framework, and whether multimodal feedback is provided. We find that enabling multimodal feedback almost always increases cost in exchange for increased performance (Figure 6). The only exception is Qwen3-Vl which struggles on the benchmark. Surprisingly, using both methods is more cost-effective across all of the claude-code agents while still maintaining improved performance. We suspect that models are able to dynamically choose the more effective feedback mechanism during execution. We find that Gemini

- 3 Flash is the most cost-efficient model while Claude models tend to be the least cost-efficient. We observe that model capacity and per-token cost does not necessarily correlate with final task cost. For example, when using claude-code, Claude Opus 4.5 (Figure 6, yellow) costs half as much as Claude Sonnet 4.5 (Figure 6, green) despite offering improved performance. As another example, when using OpenHands, Claude Haiku 4.5 costs around the same as ChatGPT Codex 5.1 despite

2https://github.com/OpenHands/OpenHands/issues/9454

being cheaper on a per-token basis and performing significantly worse.

#### 4.3 Error Analysis and Directions for Improvement.

We manually analyzed some of the most common errors that agents made when solving the task. These errors indicate potential gaps in capabilities and future directions for agent development. While there are a variety of errors, we observe two consistent error patterns.

Agents struggle with multimodal understanding. Perhaps the most consistent error pattern occurs from a lack of multimodal understanding. Specifically, it is often necessary to understand multimodal inputs to properly complete a game development task. For example, creating (or even simply picking) an animation requires that the agent either parse through multiple images or pick out specific sprites within a spritesheet. Currently, agents frequently pick the wrong images or sprites (e.g., picking walking motion sprites instead of attacking motions). It is clear that improvements to multimodal understanding would significantly improve performance of agentic game development.

Agents struggle with common game development patterns. In game development, there are many common development patterns. For example, game elements (called nodes in Godot) form a tree structure where specific nodes such as an AnimatedSprite2D and CapsuleCollider handle animations and physics respectively. Another example would be signals that trigger between various files when conditions are met such as when two colliders intersect with each other. Agents frequently add nodes to incorrect levels in the tree, drop necessary signals, or assign resources to the wrong elements. We provide an example of such an error in Appendix G. This reinforces a long-standing trend within model training—models must be trained on specific domains to excel within that domain.

### 5 Related Works

Agentic Benchmarks. Software development has been one of the premier frontiers of agentic development. SWE-Bench [13, 38] was perhaps the first benchmark and catalyst towards agentic software development. Over time, multiple new software benchmarks have been developed [5, 19, 40], but they remain largely unimodal. The few multimodal software benchmarks have largely focused on frontend JavaScript development [44, 25, 39]. Instead, the most common use case for multimodal agents has been computer use [36] and web navigation [43, 17]. Progress in this domain is challenging as agents must operate in an action space rather than simply writing code. Game development bridges the gaps between these domains by requiring multimodal input, but allowing for code output. GameDevBench is able to effectively reap benefits from both software and computer use domains, thus enabling effective multimodal evaluation.

Game Playing. There has always been significant interest in the application of artificial intelligence (AI) to games [11]; gameplay has been seen as a proxy for the capabilities or intelligence of an AI system, with projects ranging from Deep Blue [4], Alpha Go [26], and Cicero [9] to more recent generalists such as SIMA 2 [2]. Practically, games provide an interactive simulation environment with clear reward signals allowing researchers to experiment with methods—particularly from reinforcement learning—to improve model capabilities. The recent flux of LLMs playing Pokémon [14, 15, 8] uses game-playing agents to evaluate and explore the agentic reasoning capabilities of frontier models which is then used directly in game development to test games [20]. This transition from game playing agents as NPCs and opponents in games to becoming a portion of the game development process marks a timely need for benchmarks such as GameDevBench.

Game Development. Concordia [31] and other subsequent work on tabletop role-playing games [32] seek to replace interactable characters with a highly adaptive story created entirely

from interactions with LLMs. Other works try to fully replace the physics engine of the game to immediately generate frames based on player actions [3]. Procedural Content Generation has a long history of using AI for game asset creation [29, 24] and evolutionary level design [28]. However, these largely focus on a singular aspect of game development. Ultimately, each of these features still needs to be combined in a game engine to develop a full game, which is the capability GameDevBench directly evaluates.

### 6 Conclusion

We present GameDevBench, the first benchmark to evaluate an agent’s ability to solve game development tasks. We find that agents struggle with tasks in game development, especially when tasks require deeper multimodal understanding. The gap between frontier and non-frontier models is sharp, with absolute differences of up to 46.2% pass@1. Additionally, we observe that agentic frameworks can drastically affect model performance, although this is highly dependent on both model and framework. Lastly, we show that even simple tooling to provide multimodal feedback consistently improves agent performance, offering as much as a 42% relative improvement. Our findings highlight the need to improve multimodal capabilities of agents—either through training or methods of visual feedback. We speculate addressing these needs would improve agentic performance in domains even beyond software and game development.

### References

- [1] Anthropic. Introducing the model context protocol. https://www.anthropic.com/news/ model-context-protocol, November 2024.
- [2] Adrian Bolton, Alexander Lerchner, Alexandra Cordell, Alexandre Moufarek, Andrew Bolt, Andrew Lampinen, Anna Mitenkova, Arne Olav Hallingstad, Bojan Vujatovic, Bonnie Li, et al. Sima 2: A generalist embodied agent for virtual worlds. arXiv preprint arXiv:2512.04797, 2025.
- [3] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024.
- [4] Murray Campbell, A Joseph Hoane Jr, and Feng-hsiung Hsu. Deep blue. Artificial intelligence, 134(1-2):57–83, 2002.
- [5] Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, Lilian Weng, and Aleksander Mądry. Mle-bench: Evaluating machine learning agents on machine learning engineering, 2025.
- [6] Ying Chen, Guoan Wang, Yuanfeng Ji, Yanjun Li, Jin Ye, Tianbin Li, Ming Hu, Rongshan Yu, Yu Qiao, and Junjun He. Slidechat: A large vision-language assistant for whole-slide pathology image understanding, 2025.
- [7] Wayne Chi, Valerie Chen, Ryan Shar, Aditya Mittal, Jenny Liang, Wei-Lin Chiang, Anastasios Nikolas Angelopoulos, Ion Stoica, Graham Neubig, Ameet Talwalkar, and Chris Donahue. Edit-bench: Evaluating llm abilities to perform real-world instructed code edits, 2025.
- [8] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the

- frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.
- [9] Meta Fundamental AI Research Diplomacy Team (FAIR)†, Anton Bakhtin, Noam Brown, Emily Dinan, Gabriele Farina, Colin Flaherty, Daniel Fried, Andrew Goff, Jonathan Gray, Hengyuan Hu, et al. Human-level play in the game of diplomacy by combining language models with strategic reasoning. Science, 378(6624):1067–1074, 2022.
- [10] Aleksandar Filipović. The role of artificial intelligence in video game development. Kultura polisa, 20(3):50–67, 2023.
- [11] Roberto Gallotta, Graham Todd, Marvin Zammit, Sam Earle, Antonios Liapis, Julian Togelius, and Georgios N Yannakakis. Large language models and games: A survey and roadmap. IEEE Transactions on Games, 2024.
- [12] Dhanamma Jagli, Subhashchandra Nalla, Srinivasrao Danikonda, and Laxmi Nakirekanti. Artificial intelligence usage in game development. 2024.
- [13] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues?, 2024.
- [14] Seth Karten, Jake Grigsby, Stephanie Milani, Kiran Vodrahalli, Amy Zhang, Fei Fang, Yuke Zhu, and Chi Jin. The pokeagent challenge: Competitive and long-context learning at scale. NeurIPS Competition Track, 2025.
- [15] Seth Karten, Andy Luu Nguyen, and Chi Jin. Pokéchamp: an expert-level minimax language agent. arXiv preprint arXiv:2503.04094, 2025.
- [16] KidsCanCode. Godot Recipes. https://kidscancode.org/godot_recipes/4.x/. Version 4.x, accessed January 28, 20.
- [17] Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Chong Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Ruslan Salakhutdinov, and Daniel Fried. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks, 2024.
- [18] Ryan Koo, Minhwa Lee, Vipul Raheja, Jong Inn Park, Zae Myung Kim, and Dongyeop Kang. Benchmarking cognitive biases in large language models as evaluators, 2024.
- [19] Mike A. Merrill, Alexander G. Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, Lin Shi, Jeong Yeon Shin, Thomas Walshe, E. Kelly Buchanan, Junhong Shen, Guanghao Ye, Haowei Lin, Jason Poulos, Maoyu Wang, Marianna Nezhurina, Jenia Jitsev, Di Lu, Orfeas Menis Mastromichalakis, Zhiwei Xu, Zizhao Chen, Yue Liu, Robert Zhang, Leon Liangyu Chen, Anurag Kashyap, Jan-Lucas Uslu, Jeffrey Li, Jianbo Wu, Minghao Yan, Song Bian, Vedang Sharma, Ke Sun, Steven Dillmann, Akshay Anand, Andrew Lanpouthakoun, Bardia Koopah, Changran Hu, Etash Guha, Gabriel H. S. Dreiman, Jiacheng Zhu, Karl Krauth, Li Zhong, Niklas Muennighoff, Robert Amanfu, Shangyin Tan, Shreyas Pimpalgaonkar, Tushar Aggarwal, Xiangning Lin, Xin Lan, Xuandong Zhao, Yiqing Liang, Yuanli Wang, Zilong Wang, Changzhi Zhou, David Heineman, Hange Liu, Harsh Trivedi, John Yang, Junhong Lin, Manish Shetty, Michael Yang, Nabil Omi, Negin Raoof, Shanda Li, Terry Yue Zhuo, Wuwei Lin, Yiwei Dai, Yuxin Wang, Wenhao Chai, Shang Zhou, Dariush Wahdany, Ziyu She, Jiaming Hu, Zhikang Dong, Yuxuan Zhu, Sasha Cui, Ahson Saiyed, Arinbjörn Kolbeinsson, Jesse Hu, Christopher Michael Rytting, Ryan Marten, Yixin Wang, Alex Dimakis, Andy Konwinski, and Ludwig Schmidt. Terminal-bench: Benchmarking agents on hard, realistic tasks in command line interfaces, 2026.

- [20] Nunu AI. Beating the world record in pokémon emerald: An AI agent case study. https: //nunu.ai/case-studies/pokemon-emerald, 2024.
- [21] Junhyuk Oh, Xiaoxiao Guo, Honglak Lee, Richard L. Lewis, and Satinder Singh. Actionconditional video prediction using deep networks in atari games. In Neural Information Processing Systems, 2015.
- [22] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021.
- [23] Julian Schrittwieser, Ioannis Antonoglou, T. Hubert, K. Simonyan, L. Sifre, Simon Schmitt, A. Guez, Edward Lockhart, D. Hassabis, T. Graepel, T. Lillicrap, and David Silver. Mastering atari, go, chess and shogi by planning with a learned model. Nature, 588:604 – 609, 2019.
- [24] Noor Shaker, Julian Togelius, and Mark J Nelson. Procedural content generation in games. 2016.
- [25] Chenglei Si, Yanzhe Zhang, Ryan Li, Zhengyuan Yang, Ruibo Liu, and Diyi Yang. Design2code: How far are we from automating front-end engineering? ArXiv, abs/2403.03163, 2024.
- [26] David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George Van Den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, et al. Mastering the game of go with deep neural networks and tree search. nature, 529(7587):484–489, 2016.
- [27] David Silver, T. Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, A. Guez, Marc Lanctot, L. Sifre, D. Kumaran, T. Graepel, T. Lillicrap, K. Simonyan, and D. Hassabis. A general reinforcement learning algorithm that masters chess, shogi, and go through self-play. Science, 362:1140 – 1144, 2018.
- [28] Shyam Sudhakaran, Miguel González-Duque, Claire Glanois, Matthias Freiberger, Elias Najarro, and Sebastian Risi. Mariogpt: Open-ended text2level generation through large language models, 2023.
- [29] Adam Summerville, Sam Snodgrass, Matthew Guzdial, Christoffer Holmgård, Amy K. Hoover, Aaron Isaksen, Andy Nealen, and Julian Togelius. Procedural content generation via machine learning (pcgml), 2018.
- [30] Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. ArXiv, abs/2408.14837, 2024.
- [31] Alexander Sasha Vezhnevets, John P Agapiou, Avia Aharon, Ron Ziv, Jayd Matyas, Edgar A Duéñez-Guzmán, William A Cunningham, Simon Osindero, Danny Karmon, and Joel Z Leibo. Generative agent-based modeling with actions grounded in physical, social, or digital space using concordia. arXiv preprint arXiv:2312.03664, 2023.
- [32] Alexander Sasha Vezhnevets, Jayd Matyas, Logan Cross, Davide Paglieri, Minsuk Chang, William A Cunningham, Simon Osindero, William S Isaac, and Joel Z Leibo. Multi-actor generative artificial intelligence as a game engine. arXiv preprint arXiv:2507.08892, 2025.

- [33] O. Vinyals, Igor Babuschkin, Wojciech M. Czarnecki, Michaël Mathieu, A. Dudzik, Junyoung Chung, David Choi, Richard Powell, T. Ewalds, Petko Georgiev, Junhyuk Oh, Dan Horgan, M. Kroiss, Ivo Danihelka, Aja Huang, L. Sifre, Trevor Cai, J. Agapiou, Max Jaderberg, A. Vezhnevets, Rémi Leblond, Tobias Pohlen, Valentin Dalibard, D. Budden, Yury Sulsky, James Molloy, T. Paine, Caglar Gulcehre, Ziyun Wang, T. Pfaff, Yuhuai Wu, Roman Ring, Dani Yogatama, Dario Wünsch, Katrina McKinney, Oliver Smith, T. Schaul, T. Lillicrap, K. Kavukcuoglu, D. Hassabis, C. Apps, and David Silver. Grandmaster level in starcraft ii using multi-agent reinforcement learning. Nature, 575:350 – 354, 2019.
- [34] Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Qi Liu, Tianyu Liu, and Zhifang Sui. Large language models are not fair evaluators, 2023.
- [35] Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighoff, Yizhe Zhang, Binyuan Hui, Junyang Lin, Robert Brennan, Hao Peng, Heng Ji, and Graham Neubig. Openhands: An open platform for ai software developers as generalist agents, 2025.
- [36] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments, 2024.
- [37] Sayid Adli Yakan. Analysis of development of artificial intelligence in the game industry. International Journal of Cyber and IT Service Management, 2(2):111–116, 2022.
- [38] John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik R Narasimhan, and Ofir Press. SWE-agent: Agent-computer interfaces enable automated software engineering. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.
- [39] John Yang, Carlos E. Jimenez, Alex L. Zhang, Kilian Lieret, Joyce Yang, Xindi Wu, Ori Press, Niklas Muennighoff, Gabriel Synnaeve, Karthik R. Narasimhan, Diyi Yang, Sida I. Wang, and Ofir Press. Swe-bench multimodal: Do ai systems generalize to visual software domains?, 2024.
- [40] John Yang, Kilian Lieret, Joyce Yang, Carlos E. Jimenez, Ofir Press, Ludwig Schmidt, and Diyi Yang. Codeclash: Benchmarking goal-oriented software engineering, 2025.
- [41] Shaofeng Yin, Jiaxin Ge, Zora Zhiruo Wang, Xiuyu Li, Michael J. Black, Trevor Darrell, Angjoo Kanazawa, and Haiwen Feng. Vision-as-inverse-graphics agent via interleaved multimodal reasoning, 2026.
- [42] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena, 2023.
- [43] Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. Webarena: A realistic web environment for building autonomous agents, 2024.
- [44] Hongda Zhu, Yiwen Zhang, Bing Zhao, Jingzhe Ding, Siyao Liu, Tong Liu, Dandan Wang, Yanan Liu, and Zhaojian Li. Frontendbench: A benchmark for evaluating llms on front-end development via automatic evaluation. ArXiv, abs/2506.13832, 2025.

### A Task Construction Prompt

Below is the full prompt provided to the Codex agent for automatic task construction from YouTube tutorials (Stage 2). The agent receives this prompt along with a pointer to a specific tutorial folder containing a video transcript, metadata, and a GitHub repository URL.

###### Task Construction System Prompt

# YouTube Tutorial to Task Construction Guide This guide explains how to convert a single YouTube Godot tutorial (with transcript and GitHub repo) into GameDevBench tasks. GameDevBench is a multimodal LLM Agent benchmark to test if models can develop games or assist with game development. ## Godot Godot is installed and usable with the ‘godot‘ command. VERY IMPORTANT: Whenever you run ‘godot‘ please ensure you set a timeout of 1 minute. ## Context You will be working in a single tutorial folder at a time. Each folder contains:

{data_folder}/{channel_name}/{video_title}/

+-- transcript.txt # Full video transcript

+-- metadata.json # Video metadata

+-- github_repo.txt # GitHub repository URL ### Tips for YouTube Tutorial Processing

- 1. Transcript Context: Tutorials often explain "why" before "what" - look for action verbs

- 2. GitHub is Ground Truth: When transcript is unclear, GitHub repo shows what actually works

- 3. Simplify Complexity: If tutorial covers multiple concepts, break into multiple tasks

- 4. Test Repository First: Clone and run GitHub repo to understand expected behavior

- 5. Match Repo Structure: Use similar node names and organization as the repo

- 6. License Compliance: All repos already filtered for MIT/Apache-2.0/CC0-1.0 ### Common Pitfalls

- - Copying GitHub Repo Verbatim: Adapt to GameDevBench structure, don’t just copy

- - Ignoring Transcript: GitHub shows "what" but transcript explains "why" and learning objective

- - Overly Broad Tasks: Focus on one specific learning objective per task

- - Missing Assets: Ensure sprites/sounds from repo are included in both task directories

- - Weak Validation: Check everything that makes the task correct ### Key Principles for Single-Folder Processing

- 1. All analysis happens in the tutorial folder first

- - Clone repo to repo/ subdirectory

- - Create analysis_progress.md for documentation

- - Complete all analysis before creating tasks

- 2. Document everything as you go

- - Update analysis_progress.md after each step

- - Include transcript quotes, repo structure, task ideas

- - Track what works and what doesn’t

- 3. Test the GitHub repo before extracting tasks

- - Run godot --import-all --quit

- - Verify it’s a working Godot project

- - Check for missing assets or dependencies

- 4. Navigate to GameDevBench root for task creation

- - Don’t create tasks inside the tutorial folder

- - Copy assets from tutorial’s repo/ to task directories

- 5. Return to tutorial folder for final documentation

- - Update analysis_progress.md with completion status

- - Note which tasks were created

- - Record any issues for future reference

- ## Phase 1: Setup

- ### Step 1: Check for Godot 4. We only want to operate on Godot 4 tutorials. If the tutorial folder / github repo is for a Godot 3 project, stop and report that.

- ### Step 2: Set Up Your Workspace Create a progress tracking file to document your work.

- ### Step 3: Read Available Files Read the transcript, GitHub repo URL, and metadata. Document in analysis_progress.md:

- - Main topic, key concepts, estimated complexity

- - Has implementation steps: yes/no

- - GitHub repo available: yes/no

- - Suitable for tasks: yes/no/maybe

- ### Step 4: Analyze Transcript for Task Ideas Look for in transcript.txt:

- - Node creation mentions ("create a CharacterBody2D")

- - Node adjustments ("adjust the anchors of the container")

- - Property settings ("set the gravity to 980")

- - Script attachment steps ("attach a new script")

- - Signal connections ("connect the body_entered signal")

- - Scene organization instructions

- - Multimodal reasoning ("create an animation from the spritesheet") Task Categories:

- - Graphics & Animation

- - Physics & Movement

- - World Building

- - Programming

- - User Interface

- - Game Systems

- - Audio IMPORTANT:

- - The skillset required between tasks should be diverse. Focus on tasks that require adjusting node properties, adding new nodes, or adding sub-children.

- - Focus on keeping the tasks faithful to the tutorial.

- - Each task must be independent from each other.

- - Tasks that require multimodal reasoning (e.g., cutting up a spritesheet, adjusting sound to match animation) are especially desirable.

- ### Step 5: Clone and Examine GitHub Repository Clone to a repo/ subdirectory. Examine structure, key files, project.godot for Godot version. Document:

- - Main scene, key scenes, scripts, assets

- - Key nodes and configuration

- - Critical properties

- - Scripts summary Create a dependency graph mapping file and feature dependencies.

- ### Step 6: Test the GitHub Repository

Import assets, try to run the project, check for tests. Document import results, project status, and conclusion.

- ## Phase 2: Task Creation

- ### Step 1: Extract Actionable Tasks Based on transcript + GitHub repo, identify specific, testable tasks. Ensure tasks center on node creation

and/or inspector configuration. Difficulty Guidelines:

- - Easy: 1 to 3 individual steps

- - Medium: 4 to 8 individual steps

- - Hard: 9 or more individual steps For each task document:

- - Source (transcript line + repo file)

- - What to create (specific nodes/properties)

- - Validation criteria

- - File modifications from start to finish state

- - Difficulty, GitHub reference, categories, multimodal flag

- ### Step 2: Create Task Directories in GameDevBench Navigate to GameDevBench root. Determine next task number. Create directories for both tasks/ and tasks_gt /. Copy project template and needed assets from cloned repo.

- ### Step 3: Create task_config.json {

"task_id": XXXX, "name": "Descriptive Task Name from Video", "instruction": "Clear, specific instructions...", "difficulty": "easy|medium|hard", "template_id": X, "metadata": {

"tutorial_folder": "...", "tutorial_source": "YouTube: {channel} - {video_title}", "video_id": "...", "github_repo": "...", "transcript_excerpt": "...", "expected_nodes": ["NodeType1", "NodeType2"], "key_properties": {"property": "expected_value"}

}, "tags": ["youtube", "2d|3d", "category", "node-type"]

}

- ### Step 4: Reference GitHub Repo for Ground Truth

- - GitHub repo shows the completed task

- - Adapt to fit GameDevBench structure (don’t copy verbatim)

- - Document which files were copied and which were not

- ### Step 5: Create Ground Truth Implementation

Study GitHub repo scene structure, recreate key nodes and hierarchy, set properties, add scripts, simplify if needed, and test.

- ## Phase 3: Task Instruction ### Step 1: Create Task Instruction Key principles:

- - Concise, clear, and unambiguous

- - Solver must understand requirements to go from start state to ground truth

- - Solver will NOT have access to tests or ground truth

- - Self-contained: no references to transcript, tests, other tasks, or the tutorial name

- - Mention technical requirements (node types, APIs) but not usage details

- - NO tips, NO hints, NO test commands, NO code examples Each instruction step must have evidence pointing back to the original source (transcript or repository).

- ## Phase 4: Task Validation

- ### Step 1: Create Validation Script Create a GDScript validation that:

- - Asserts required nodes exist in the correct hierarchy

- - Confirms critical inspector values left unset in the starting point

- - Fails early when structural requirements are missing

- - Prints VALIDATION_PASSED or VALIDATION_FAILED messages

- - Copy to BOTH task and ground truth directories Document how each test maps to a specific instruction step.

- ### Step 2: Create Starting Point (Incomplete Version)

- - Provide basic scaffolding so the scene launches

- - Include required raw assets

- - Omit key implementation details:

- - Leave tutorial-created nodes absent

- - Skip scripts and signal connections

- - Leave tutorial-modified inspector properties unset

Goal: Starting point should fail validation but provide foundation.

- ### Step 3: Test and Validate

- - Starting point: should output VALIDATION_FAILED

- - Ground truth: should output VALIDATION_PASSED ## Final Quality Checklist

- - analysis_progress.md fully completed

- - Each instruction step lists transcript or repo evidence

- - Validation maps to instruction with file + line numbers

- - Multiple independent tasks created from the tutorial

- - Every task has both ground truth and starting point

- - Starting points fail validation

- - Ground truths pass validation

- - Each task includes valid main.tscn and test.tscn

- - Node/inspector-focused requirements asserted

- - At least one task requires multimodal reasoning

- - Asset transfer summary recorded

- - Deviations, blockers, and Godot version issues documented

### B Task Refinement Prompt

Below is the full prompt provided to the agent for automatic task validation and refinement (Stage 3). The prompt consists of two parts: (1) an instruction that describes the validation workflow and context, and (2) a checklist template that the agent must fill out with evidence for each criterion. If any criterion fails, the agent is instructed to fix the task accordingly.

A variant of this prompt omits scripting-related checks and adds the constraint “No .gd script editing is required,” which was used for tasks that focus exclusively on scene construction and inspector configuration.

###### Task Refinement System Prompt

# Benchmark Task Validation Guide This file documents instructions to validate a task for GameDevBench. GameDevBench is a multimodal LLM Agent benchmark to test if models can develop games or assist with game development. ## Context You will be working on a single task at a time. Each task has three related folders: ### Tutorial Folder

{data_folder}/{channel_name}/{video_title}/

+-- repo/ # YouTube tutorial repository

+-- analysis_progress.md # Notes from task generation

+-- transcript.txt # Full video transcript

+-- metadata.json # Video metadata

+-- github_repo.txt # GitHub repository URL ### Task (Starting Point) Folder

tasks/task_{number}_{name}/

+-- task_config.json # Contains the task instruction

+-- scripts/test.gd # Contains the test code

### Task (Ground Truth) Folder tasks_gt/task_{number}_{name}/ (Same structure as starting point, with completed solution)

## Instructions Your job is to:

- 1. Read and analyze the transcript
- 2. Read and examine the GitHub repository
- 3. Read and examine the task created
- 4. Document your progress
- 5. Validate whether the task satisfies each criterion 5a. Each criterion must have evidence for validation documented Copy the checklist template into the task starting point folder and fill it out as you validate.

--# Key Checklist

- - [ ] The task starting point runs with ‘uv run gamedevbench validate $TASK_NAME‘ and successfully outputs a test failure.
- - Evidence:
- - [ ] The task ground truth runs with ‘uv run gamedevbench --gt validate $TASK_NAME‘ and successfully outputs SUCCESS.
- - Evidence:
- - [ ] In every task, there exists a valid main.tscn and test.tscn similar to tasks_gt/task_0001_place_asset.
- - Evidence:
- - [ ] The task instruction matches the tutorial transcript. The task instructions must be a subset of the tutorial transcript.
- - Instruction 1 / Transcript 1

...

- - Instruction N / Transcript N
- - Instructions Missing from Transcript: Explanation
- - [ ] The task code is directly derived from the repository code. Please document where the derived code is.
- - Evidence:
- - [ ] The task instruction is clear, unambiguous, and self-contained. There are no references to the tutorial or other tasks.
- - Evidence:
- - [ ] The tests in test.gd match the instructions. All tests are contained in the instruction. Similarly, all instructions are in the tests. Explain how to adjust the tests themselves to match the instructions.
- - Test 1, Instruction 1
- - Test 2, Instruction 2

...

- - Test N, Instruction N
- - Missing Coverage: Explanation here
- - [ ] Each test in test.gd is unambiguously defined in the instructions. With just the instruction and the task code ( without looking at the tests), it is unambiguously possible to satisfy each test condition.
- - For EACH test, list EVERY assertion/check it makes, then verify the instruction specifies that EXACT detail:
- - Test 1, Assertions: [list each check], Instruction coverage: [exact instruction text]
- - Test 2, Assertions: [list each check], Instruction coverage: [exact instruction text]

...

- - CRITICAL AMBIGUITY CHECKS - For each test, verify:
- - [ ] String formatting (padding, delimiters, exact format) is specified in instruction
- - [ ] Exact string values/names are in instruction
- - [ ] Number formats (zero-padding, decimal places) are specified
- - [ ] Any comparison operators have clear criteria
- - [ ] Node names, paths, and types match instruction
- - [ ] Property values (numbers, booleans, strings) have exact values in instruction
- - Ambiguous Tests: [List any test checks that lack exact specification in instruction]
- - [ ] If there are multiple solutions to the problem, the tests in test.gd are flexible to allow multiple solutions. Mark as completed if there is only one solution and that solution is clearly decipherable from the instructions.
- - Evidence:
- - [ ] The folder and file names are consistent with other tasks (tasks_gt/task_0001_place_asset).
- - Evidence:
- - [ ] PROCEED. Check this box if the task is validated and all key checks pass successfully.

# Feature Checklist

- - [ ] The task contains instructions or goals that are Node/inspector-focused.
- - Evidence:
- - [ ] The task contains or requires multimodal reasoning or understanding to complete.
- - Evidence:
- - [ ] The task contains a multimodal input (such as an image) in the instruction.
- - Evidence:

# Identifying Ambiguous Tests (Examples)

- ## Example 1: AMBIGUOUS - Vague formatting requirement Instruction: "Format incremental and timer step text"

Test Code: var expected_text = "Destroy 5 ships 00/05" if do_label.text != expected_text:

issues.append("Initial text should be ’%s’" % expected) Assertions:

- - Checks text equals exactly "Destroy 5 ships 00/05"
- - Requires zero-padded format
- - Requires specific spacing and delimiter

Why AMBIGUOUS: Instruction says "format text" but does not specify zero-padding, exact delimiter, spacing, or format string structure.

How to fix: Change instruction to "Format incremental step text as ’{details} {collected:02d}/{required:02d}’" OR make test flexible to accept any reasonable format.

- ## Example 2: UNAMBIGUOUS - Specific requirement Instruction: "Set the ColorRect size to Vector2(screen_max_size, screen_max_size)"

Test Code: var expected = Vector2(screen_max_size, screen_max_size) assert(color_rect.size == expected)

Why UNAMBIGUOUS: Instruction explicitly states the exact Vector2 formula. No ambiguity about what value is expected.

- ## Example 3: AMBIGUOUS - Missing specific values

Instruction: "Connect to QuestManager signals" Test Code:

var required = ["step_updated", "step_complete", "quest_completed", "quest_failed"] for signal_name in required:

if not _has_connection(qm, signal_name, quest_ui): issues.append("Must connect to %s" % signal_name)

Why AMBIGUOUS: Instruction says "signals" (vague) but test checks for 4 specific signal names. A solver might connect only 2 signals and technically satisfy "connect to signals".

How to fix: Change instruction to "Connect to QuestManager signals: step_updated, step_complete, quest_completed, and quest_failed".

### C Human Annotation Instructions

Below are the instructions provided to human annotators during Stage 4. Annotators were asked to verify task correctness, fix common issues, and flag tasks that were unsalvageable.

###### Human Annotation Instructions

# Human Validation Goal: Ensure that tasks are solvable and not ambiguous. Essentially, we want to ensure that tasks are actually good tasks. Fixing a task should take 5-15 minutes per task at most. If it takes longer, then it may be too difficult for us to fix,

in which case we can skip the task and mark it as Blocked in the Status. ## Common Issues Most tasks require minor fixes. By far the most common are the following:

- - Ambiguous Instructions (e.g., model says to change the skybox, but there may be three different skybox variables possible)
- - Overly Strict Tests (e.g., tests require something that isn’t stated in the instruction)
- - Naming is a good one (tests for a named node, without requesting it)
- - Conflicting Instructions (e.g., model says to do two incompatible things)
- - Tutorial References - sometimes the instruction mentions the tutorial; just remove that reference

As with previous, the best way to find these errors is to actually implement the task. However, in this case you should feel free to use any tool (e.g., a coding agent directly) to support you or implement the task. The goal is NOT to see if you can solve the task (as it was before), but to identify errors in the task.

## General Procedure

- 1. Look at the task (base and ground truth versions) in the editor. Read the task instruction. See if it looks reasonable (multimodality) or if it’s clearly a scripting oriented task. What you’re looking for is something that just makes sense. Run this before to ensure everything loads properly:

godot --path /path/to/folder --editor

- 2. Change directory to the task. Ask your agent of choice to solve the task.
- 3. While the agent is solving the task, take a look at test.gd. See if each test matches the instruction. If not, fix. You can usually catch some easy errors, such as named node/function tests.
- 4. After the agent finishes, run validation. See if it passes / fails.
- 5. If the agent failed, pass the test.gd in. Ask the agent if it missed anything. Ask the agent if the things it missed are due to its own error or due to ambiguous instruction/overly strict tests. This is my usual prompt:

"Look at test.gd. Was there anything you missed? Did you miss it due to your mistake or were the instructions unclear?"

The agent is sometimes good, sometimes bad at this. It really depends on the agent. ## Multimodality You may encounter tasks that have poor multimodality. If it’s easy to fix, then just fix it (e.g., something is not in the camera, some default is incorrect). If it’s not, flag it and move on. Note that some tasks are inherently multimodal

in which case don’t worry about it. We have an automated script that is improving multimodality, so really just fix any obvious errors.

## Skipping Tasks

If the project doesn’t play, doesn’t load, or some other catastrophic issue, just skip the task. There should be very few tasks like this.

Please mark all the issues using the issues dropdown. This is important so we can double back to tasks if necessary. ## Important: Preventing Agent Contamination When using an agent to attempt the task, always run the task in the task folder itself and instruct the model:

- - Don’t look at test.gd, test.tscn or execute any tests
- - Don’t look outside of your current folder
- - Don’t look at task_config.json
- - Don’t look at task_validation.md

### D Prompt Templates

All task prompts are derived from the same base instruction, with optional extensions that provide additional multimodal feedback mechanisms. We report the exact prompt templates used for the baseline, MCP-enabled, and runtime-video-enabled settings below.

###### Baseline Prompt

<INSTRUCTION FROM task_config.json> You must complete the full task without any further assistance. Godot is installed and you can run godot using the ‘godot‘ command. It is recommended to run this with a timeout (e.g., ‘timeout 10 godot‘) to prevent hanging. You are a visual agent and can use images and videos to help you understand the state of the game.

###### Prompt with MCP

<INSTRUCTION FROM task_config.json> You must complete the full task without any further assistance. Godot is installed and you can run godot using the ‘godot‘ command. It is recommended to run this with a timeout (e.g., ‘timeout 10 godot‘) to prevent hanging. You are a visual agent and can use images and videos to help you understand the state of the game. You have access to a Godot MCP (Model Context Protocol) server that provides specialized tools for working with Godot projects. Available MCP Tools:

- - ‘godot-screenshot‘: Takes a screenshot of the Godot editor. Usage Guidelines:
- - Use screenshots before starting to understand project structure.
- - Use screenshots after making changes to verify correctness.
- - Use screenshots during debugging to inspect editor state.

Prompt with Runtime Video

<INSTRUCTION FROM task_config.json> You must complete the full task without any further assistance. Godot is installed and you can run godot using the ‘godot‘ command. It is recommended to run this with a timeout (e.g., ‘timeout 10 godot‘) to prevent hanging. You are a visual agent and can use images and videos to help you understand the state of the game. You can run the game and capture visual output using:

- - ‘godot --path . --quit-after 1 --write-movie output.png‘ You can capture short videos using:
- - ‘timeout 60s godot --path . --quit-after 60 --write-movie output.avi‘

Ensure that Godot exits after execution to avoid hanging. Use images or videos to verify that your changes worked as expected.

### E Task Examples

We provide examples of tasks in GameDevBench. Each task can be solved by taking actions in the editor as a human would or by directly editing code files.

#### E.1 Isometric Crusader Animation

###### In this example, the goal is to add physical collision and animation to the character. This is a 2D graphics and animations task that focuses on the animation editor which is a contextual editor.

Instruction: In scenes/Player.tscn ﬁnish the Player CharacterBody2D by switching motion_mode to Floating, adding an AnimatedSprite2D that uses the Isometric Mini-Crusader sprite set with animations idle0 through idle7 and run0 through run7 (default to idle0 at speed 2.0), each one fully using the 16 or 17-frame animation loop in the corresponding sprite set, and adding a CollisionShape2D rectangle positioned at (7, 44) with a 66x24 size so the collider hugs the feet. Name the nodes the same as each node's class names (AnimatedSprite2D and CollisionShape2D, case sensitive).

[Figure 4]

③ set default animation and speed_scale

④ Add Frames for animations

① Add Node to scene

[Figure 5]

② Add Animation idle0, ...

Editor

[Figure 6]

[Figure 7]

[Figure 8]

|①|
|---|
|③|

②

④

Code

- Figure 7: An example task from GameDevBench. In this example, the goal is to add physical collision and animation to the character. This can be achieved through either taking actions directly in the editor or editing code files. Each action in the editor is equivalent to specific modifications within the code files. Matching steps are denoted with the same numbers in our figure.

#### E.2 Floating Balls

In this example, the goal is to populate an empty 3D scene with a water depth visualization, including environment lighting, shader-driven water plane, background spheres, and a camera. This is a 3D graphics and animations task that focuses on the scene editor.

Instruction: Populate res://scenes/main.tscn so Main contains a water depth visualization scene: add a WorldEnvironment with a procedural sky and glow enabled, a DirectionalLight3D with shadows enabled, a MeshInstance3D named Water that uses a 10x10 PlaneMesh subdivided 20x20 with a ShaderMaterial pointing at res://scenes/WaterShader.tres, a Background Node3D holding exactly ﬁve sphere MeshInstance3Ds positioned near the water surface (y between -2 and 2) with green StandardMaterial3D overrides, and a Camera3D positioned at (-4.269, 0.54, -3.258) with a 70.9° FOV and all spheres are visible.

[Figure 9]

① Add WorldEnvironment & DirectionalLight3D & Water (MeshInstance3D)

[Figure 10]

| |[Figure 11]|
|---|---|
|② Conﬁ sky mod material|gure nodes: e, glow, mesh,<br><br>, shadow, ...|
| |[Figure 12]|

④ Add camera and

adjust positions

[Figure 13]

③ Add and conﬁgure background and spheres

Editor

[Figure 14]

[Figure 15]

[Figure 16]

④ Add camera and

③ Add and conﬁgure background and spheres

adjust positions

[Figure 17]

② Add and set enums, materials, resources for created nodes

[Figure 18]

[Figure 19]

[Figure 20]

① Add WorldEnvironment & DirectionalLight3D & Water (MeshInstance3D)

Code

- Figure 8: An example task from GameDevBench. In this example, the goal is to populate an empty

###### 3D scene with a water depth visualization, including environment lighting, shader-driven water plane, background spheres, and a camera. This can be achieved through either taking actions directly in the editor or editing the scene file (main.tscn). Each action in the editor is equivalent to specific modifications within the scene file. Matching steps are denoted with the same numbers in our figure

#### E.3 FPS User Interface

###### In this example, the goal is to build a complete three-screen menu system (Launch, Pause, and Restart) and signal connections to the menu handler script. This is a user interface task that focuses on the scene editor.

Instruction: Create res://scenes/menus.tscn as a Control named Menus that ﬁlls the viewport, keeps process mode Always, and uses scripts/ menu_handler.gd. Add three hidden panels, all centered with anchors at 0.5: LaunchMenu (offsets: -205, -196 to 205, 196), PauseMenu (offsets: -205, -196 to 205, 45), and RestartMenu (offsets: -205, -196 to 205, 63). Give LaunchMenu a VBoxContainer with 'FPS Horror' / 'By Bonkahe' labels; give PauseMenu a 'Paused' label; and give RestartMenu a 'You Died' label. Add the appropriate buttons to each (FullScreenButton, PlayButton, QuitButton, ResumeButton, RestartButton) using 31pt font overrides and exact node names. Finally, add a CanvasLayer on layer 2 with a TransitionOverlay ColorRect that ignores mouse input and stretches to the viewport; assign assets/materials/ menu_overlay_material.tres to it, export the LaunchMenu/PauseMenu/DeathMenu references on the root, and connect every button’s pressed signal to the proper menu handler method (ToggleFullScreen, BeginLaunch, HidePauseMenu, RestartLevel, ExitGame)."

[Figure 21]

###### ⑤ Link & set all button control signals

- ① Add Launch Menu

(place UI components in the editor)

- ② Add Pause Menu

- ③ Add Restart Menu

- ④ Add Transition

Overlay in Canvas Layer

Editor

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

④ Add Transition Overlay in Canvas Layer

[Figure 30]

① Add Launch Menu ② Add Pause Menu ③ Add Restart Menu

⑤ Link & set all button control signals

Code

- Figure 9: An example task from GameDevBench. In this example, the goal is to build a complete three-screen menu system (Launch, Pause, and Restart) with styled buttons, title labels, a shaderdriven transition overlay, and signal connections to the menu handler script. This can be achieved through either taking actions directly in the editor or editing the scene file (menus.tscn). Each action in the editor is equivalent to specific modifications within the scene file. Matching steps are denoted with the same numbers in our figure.

#### E.4 RTS Unit

###### In this example, the goal is to build a reusable RTS unit with a sprite, collision shapes, a detection area for neighbor avoidance, and an aura shader that highlights the unit when selected. The main focus is on the scripting, thus this is a gameplay logic task that focuses on the script editor.

Instruction: Build the reusable Unit scene: root it in a CharacterBody2D on collision layer 2/mask 3, keep it in the 'units' group. Add a Sprite2D child that uses the towerDefense spritesheet texture with region_enabled=true and region_rect=Rect2(960, 640, 64, 64), and apply a ShaderMaterial using the aura.gdshader. Add a CollisionShape2D child with a CircleShape2D of radius 14. Add an Area2D child named 'Detect' (collision layer 2, collision mask 2) with its own CollisionShape2D child using a CircleShape2D of radius 35. Implement unit.gd to export a speed variable, deﬁne a target_radius variable, include set_selected() and set_target() setters, an avoid() function that uses $Detect.get_overlapping_bodies(), and use move_and_collide() for movement. The set_selected() function must toggle the aura_width shader parameter (1.0 when selected=true, 0.0 when selected=false). Duplicate the Sprite2D material in _ready() to ensure each instance has its own material.

[Figure 31]

|[Figure 32]<br><br>④ Write Game logic|
|---|

① Add Sprite2D with shader

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

- ① Add Sprite2D with shader

② Add Collision

- ② Add Collision

③ Add Area2D

- ③ Add Area2D

[Figure 37]

Write Game Logic

[Figure 38]

- Figure 10: An example task from GameDevBench. In this example, the goal is to build a reusable RTS unit with a sprite, collision shapes, a detection area for neighbor avoidance, and an aura shader that highlights the unit when selected. Unlike purely scene-based tasks, this task requires both editing the scene file (player.tscn) and implementing gameplay logic in a script file (unit.gd). Each action in the editor is equivalent to specific modifications within the code files. Matching steps are denoted with the same numbers in our figure.

### F Task Statistics

We provide detailed statistics for GameDevBench. Different tasks test different skills, thus causing skewed distributions. For example, some sprite animation tasks have thousands of sprites that must be processed.

Table 3: Comprehensive GameDevBench Task Statistics. Mean (3σ) denotes the mean after excluding values more than 3 standard deviations from the mean.

##### Mean Mean (3σ) Median Max

Files 72.4 14.0 10.0 1929 Filetypes 6.4 6.2 6.0 18 Lines of Code 500.5 350.0 196.0 20072 Nodes 17.8 10.4 6.0 982

Overview

Scripting Files 2.5 2.2 1.0 42 Scripting Lines 236.9 165.3 108.0 9543

Godot Scripting

Scene Files 3.3 2.9 3.0 54 Scene Lines 194.5 116.9 30.0 10282

Godot Scenes

Images 60.3 1.8 1.0 1920 Image Size (px) 121.0K 73.9K 71.8K 16.8M Shaders 0.2 0.1 0.0 8 Audio 0.2 0.0 0.0 8 Resources 0.8 0.5 0.0 14

Assets

Files Edited 5.0 4.7 5.0 17 Filetypes Edited 3.4 3.4 3.0 6 Total Lines Edited 106.2 53.3 43.0 1948 Scripting Lines Edited 9.7 7.6 0.0 92 Scene Lines Edited 92.6 39.2 24.0 1948 Nodes Edited 3.3 3.0 2.0 24

Gold Patch

### G Case Study of Model Failure G.1 Common Game Development Patterns

- Figure 11 shows a representative failure of common game development. The task requires completing a Godot .tscn scene file for a rain particle system, including wiring the sub_emitter property on a GPUParticles2D node to a sibling Splash node. ChatGPT Codex 5.1 produces the correct property name and value (sub_emitter = NodePath("../Splash")), but places it under the ParticleProcessMaterial sub-resource instead of the GPUParticles2D node. The sub_emitter property is belonged to GPUParticles2D and has no meaning on a material resource, indicating that the model lacks the knowledge that this property must be placed under the GPUParticles2D node.

[Figure 39]

Figure 11: Example of Godot common game development task. GPT-Codex-5.1-Max places sub_emitter inside the ParticleProcessMaterial sub-resource (left, red) instead of on the GPUParticles2D node (right, green). The property is belonged to GPUParticles2D.

