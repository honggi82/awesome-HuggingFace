# arXiv:2410.16392v3[cs.CL]4Nov2025

### SCIENCE CHINA Information Sciences

. Review .

## Scaffolded Language Models with Language Supervision for Mixed-Autonomy: A Survey

Matthieu Lin1, Jenny Sheng1, Andrew Zhao2, Shenzhi Wang2, Yang Yue2, Victor Shea-Jay Huang3, Huan Liu4, Jun Liu4,

Gao Huang2 & Yong-Jin Liu1*

1Department of Computer Science and Technology, Tsinghua University, Beijing, China 2Department of Automation, Tsinghua University, Beijing, China 3Shanghai AI Lab, Shanghai, China 4Xi’an Jiaotong University, Xi’an, China

Abstract This survey organizes the intricate literature on the design and optimization of emerging structures around post-trained LMs. We refer to this overarching structure as scaffolded LMs and focus on a scaffold that integrates LMs in a multi-step process with tools. We view scaffolded LMs as semi-parametric models wherein we optimize prompts and tools, which we refer to as non-parametric variables. In particular, scaffolded LMs interpret instruction, use tools, and receive feedback all in language. Recent works use an LM as an optimizer to interpret language supervision and update non-parametric variables according to intricate objectives. In this survey, we refer to this paradigm as training of scaffolded LMs with language supervision. A key feature of non-parametric training is the ability to learn from language. Parametric training excels in learning from demonstration (supervised learning), exploration (reinforcement learning), or observations (unsupervised learning), using well-defined loss functions. Optimization in the space of language enables rich, interpretable, and expressive objectives, while mitigating issues like catastrophic forgetting and supporting compatibility with closed-source models. Furthermore, agents are increasingly deployed as co-workers in real-world applications—such as Copilot in Office tools or software development. In these mixed-autonomy settings, where control and decision-making are shared between humans and AI, users provide feedback by identifying errors or offering corrections. Accordingly, we discuss agents that inhabit streams of experience by learning from this language-based feedback.

Keywords Compound AI Systems, Language Agent, Mixed-Autonomy, Prompt Optimization, Experiential Learning Agents

Citation Scaffolded Language Models with Language Supervision for Mixed-Autonomy: A Survey. Sci China Inf Sci, for review

#### 1 Introduction

Scaffolded language models embed language models (LMs) in a framework that extends their ability beyond traditional NLP tasks to perform open-ended digital automation tasks. Notably, they mark the beginning of a new type of programs for digital automation [4]. It uses language as the interface between humans and digital systems. This opens up the opportunity for mixed autonomy where humans and AI share control and decision-making responsibilities for digital tasks [1,5].

Following the terminology from OpenAI [6], we define a scaffolded LM as a structure that integrates the LM into a multi-step process with tools. Specifically, the post-training interface of the LM enables the developer to specify the system behavior in natural language. The developer specifies available tools, and the LM executes them by generating the function calls and input arguments. Tools [7] correspond to external capabilities the model can invoke (see the glossary in Fig. 3). The scaffold refers to a surrounding structured code that manages control flow (multi-step process), maintains state (chat history), and executes tools based on responses of the LM. Thus, the resulting scaffolded LM can be viewed as a semi-parametric model, where the LM is the parametric component and non-parametric components correspond to tools and prompts. In particular, non-parametric components enables an interpretable and

* Corresponding author (email: liuyongjin@tsinghua.edu.cn)

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

- Figure 1 Non-parametric training of scaffolded LMs focuses on learning from language. Parametric training has excelled in learning from demonstration (supervised learning), exploration (reinforcement learning), and observation (unsupervised learning) by using well-defined loss functions. Instead, non-parametric training enables using an LM to optimize in the language space, allowing for efficient and interpretable learning with fuzzy objectives and rich textual feedback. As scaffolded LMs interact with digital systems operated by humans, mixed autonomy becomes increasingly important [1]. We anticipate that learning from language will serve as a crucial framework for scaffolded LMs that inhabit streams of experience from rich user feedback in mixed autonomy [2]. Importantly, it circumvents catastrophic forgetting, supports compatibility with closed-source models, and is interpretable [3]. We refer readers to Sec. 5 for further discussion.

efficient way to modify the behavior of the ‘program’. For instance, we can add new or modify tools corresponding to APIs to interact with the digital world, developer-defined code to perform computations, or even other entities such as the human or another (scaffolded) LM [8]. As well as creating detailed prompts with intricate constraints. The optimization of scaffolded LMs requires significant trial and error by expert developers. This involves identifying the LM’s failure modes by running the scaffolded LM on examples of user queries and then addressing them one by one by refining the system prompt, input template, available tools, or the scaffold’s code itself.

Accordingly, recent works [9,10] have proposed to use an LM as an optimizer that interprets intricate objectives, execution traces, and rich execution feedback to generate updated non-parametric variables. In practice, scaffolded LMs are naturally exposed to a wide variety of rich language feedback, such as user feedback or execution errors, that reveal about the failure modes. Beyond learning from execution feedback, the LM optimizer can also extract insights or reusable tools from successful execution traces [10,11].

As shown in Fig. 1, training of non-parametric variables focuses on learning from language. Notably, parametric training excels in learning from demonstrations (supervised learning) [12], exploration (reinforcement learning) [13] , and observations (unsupervised learning) [14], using well-defined loss functions [15]. In contrast, optimization in the language space [16] provides a learning system with intricate objectives expressed in natural language that is efficient and interpretable. The training of non-parametric variables circumvents key limitations of gradient descent, such as catastrophic forgetting [3], while being compatible with closed-source models [17,18]. In this survey, we refer to this framework as training of scaffolded language models with language supervision.

This survey presents a taxonomy for understanding this paradigm shift from parametric training to nonparametric training. In particular, it organizes the intricate literature on compound AI systems [19], LM pipelines [20], prompt optimization [21,22], language agents [23–26], AI workflow optimization [9,27,28], and experiential learning agents [10] under the concept of training of scaffolded language models with language supervision. It focuses on the multi-step aspect that involves optimization beyond the prompt of a single LLM call [21] to multiple LM calls and tools. Specifically, compound AI systems [19,29] focus on the systemic perspective (the interconnection of components) and overlook the multi-step aspect (agent vs workflow). Meanwhile, the literature in language agents [25] neglects the efficiency of LM pipelines (also called workflows) [20] that leverage the task structure to reduce the burden in decision-making and supervise the LM with clearer instructions.

As shown in Fig. 2, we organize this survey as follows:

- a) Sec. 2 positions this survey in the broader literature of scaffolded LMs and recent breakthroughs in parametric training with RL.

- b) Sec. 3 focuses on how the post-training interface of LMs interacts with the scaffold. We identify two types of scaffold—agents and workflows—and focus on important design aspects of each. Finally, we reflect on the trend of recent benchmarks and highlight the ability of scaffolded LMs for mixedautonomy, a setting where human and AI share control and decision-making.
- c) Sec. 4 organizes the intricate literature that uses an LM as an optimizer for non-parametric training. We categorize these methods into whether they train LMs, agents, or workflows.
- d) Sec. 5 discusses a key capability missing from current scaffolded LMs: the ability to inhabit streams of experience [2,30] and describes opportunities with non-parametric training.

Additionally, to help readers navigate this survey, we provide a glossary in Fig. 3.

Autonomous Agents §2.1

Related Works §2

Parametric Training §2.2

TrainingofScaffoldedLanguageModelsWithLanguageSupervision

Post-Trained Language Models §3.1

Tools §3.2

Scaffolded Language Models §3

Agent §3.3.1

Multi-step Process §3.3

Workflow §3.3.2 Benchmarks §3.4

Prompt Optimization §4.1

Insights §4.2.1

Experiential Learning §4.2

Tools §4.2.2

Training with Language Supervision §4

Graph Abstraction §4.3.1

AutoDiff frameworks §4.3

Graph Execution and Optimization §4.3.2

Scalability §5.1

Plasticity and Stability §5.2

Beyond Episodic Learning §5

Efficiency §5.3

Interpretability §5.4

- Figure 2 Organization of this survey. This survey introduces a unifying paradigm centered on learning from language using the non-parametric variables of a scaffolded language model. Despite the complexity and fragmentation of the literature across domains such as language agents, prompt optimization, experiential learning, and compound AI systems, we adopt a deliberately simple and clear organizational structure. This simplicity is not a limitation, but a reflection of our effort to distill a coherent framework from a rapidly evolving field. It enables us to outline key research opportunities in learning with non-parametric variables in mixed-autonomy settings.

#### 2 Related Works

This section clarifies the position of this survey in the broader literature of autonomous agents [25,31,32] and parametric training [33].

##### 2.1 Autonomous Agents

In this survey, we deliberately adopt the broader scaffold abstraction that encompasses both agents and pre-structured workflows. This is unlike existing surveys that focus on agents as the primary abstraction

Term Definition Scaffold (See Sec. 3) A surrounding structured code that manages control flow (multi-step

process), maintains chat history, and executes tools based on responses of the LM.

Chat format (See Sec. 3.1) A chat format is a protocol that turns a structured conversation into a single sequence of strings that a post-trained LM both expects as input and emits as output.

Tool (See Sec. 3.2) An external capability the model can invoke via the Tools API (e.g., developer-defined functions, search engine). A function is a specific tool defined by a JSON schema. Non-parametric training is used to implement or modify functions.

Tool call (See Sec. 3.2) A specific type of response from the LM that includes a call to one of the available tools and a corresponding plain text for that tool (e.g., a query for a search engine); for a function call, it corresponds to a structured JSON.

Mixed-autonomy (See Sec. 3.4)

A setting where humans and AI work together, sharing control and decision-making responsibilities to achieve a goal. Typically, the human delegates control to the AI when appropriate, dividing the task according to their respective strengths. For instance, in vibe coding, the human guides intent and judges quality, while the AI proposes implementations and refactors.

Prompt (See Sec. 4) The prompt includes the system and user message that is provided to an LM at the start of the chat. The user message specifies the task and data for the episode. The system message is shared across tasks, it sets the system behavior by specifying available tools, role of the scaffolded LM, high-level instructions, constraints, and any other information required to perform the task. Non-parametric training is used to improve the system prompt, e.g., by adding high-level insights or improving tool description.

Non-parametric variables (See Sec. 4)

Prompts and tools of a scaffolded LM. Agents have a single prompt while workflows use a different prompt at each step. Non-parametric variables are expressed either in language or code.

LM optimizer (See Sec. 4) An LM optimizer prompts an LM with an objective function expressed

in natural language to generate updated non-parametric variables.

Language supervision (See Sec. 4)

It refers to the objective function and execution feedback, both expressed in natural language, that are provided to the LM optimizer.

Episode (See Sec. 5) Each episode begins with a new chat history when the human delegates control to the scaffolded LM via an instruction and concludes once the scaffolded LM successfully completes the task or the human resumes control. Importantly, during an episode, the human can offer corrections or provide guidance. Its textual form corresponds to an execution trace and feedback, which is used to prompt the LM optimizer.

Figure 3 Glossary of core terms.

[25,31,32]. In particular, they focus on systematizing components such as perception, planning, action loops, tool use, memory, and environment interaction [25, 31], or feedback mechanisms (e.g., human preferences) to improve agent behavior [32]. We separate learning paradigms into (i) parametric training that updates LM weights (e.g., supervised learning, RL) and (ii) non-parametric training that updates non-parametric variables including the prompt and tools based on language-feedback. We argue that this separation is crucial for mixed-autonomy settings: non-parametric updates are interpretable and easily supervised by humans at high frequency, and they synergize with parametric post-training to deliver data-efficient learning. Using this taxonomy, we unify prompt optimization, experiential learning, and AutoDiff-style frameworks under a single lens.

##### 2.2 Parametric Training

The discussion of parametric training [34–36] is important to understand the relevance of non-parametric training in the context of recent breakthrough in post-training of LMs with RL [33–35]. We start with an example in prompt optimization where recent methods that use parametric training to learn to reason offer much better performance [33]. To understand this, we compare RL with verifiable reward [33] with previous approaches. Provided with a reward that measures the performance of a model on a downstream task by verifying that the model’s answer match the ground-truth response [37]:

r ≜ Ex∼D Ey∼p(x)[{y = y ⋆(x)}] , (1)

previously the following approaches have been used. One approach is to use an LM optimizer to generate a prompt and evaluate it against the reward. These methods use search methods such as MCTS [38] or beam search [39]. Another approach train the LM optimizer with RL to generate better prompts [40–43]. Similarly, another line of work [44] proposes test-time editing of the instruction, resulting in a querydependent LM optimizer. Tempera [44] uses a restricted action space (e.g., modify the verbalizer) to reduce variance in RL training. They define the reward as the relative performance before and after the edit on the downstream task. Recently, with stronger base models, LMs can directly optimize Eq. 1 with RL [33]. Importantly, doing so results in strong generalization, improvements beyond the domain at hand [45–47] and opens up a new axis for scaling, test-time compute. Similar examples can be found in scaffolded LMs trained with RL. SWE-RL [47] compares the generated patch with the GT patch on SWEBench. They do not use test cases as the reward signal, which can lead to reward hacking [48]. Specifically, SWE-RL is based on the Agentless [49] and applies RL on the patch generation step. They find that, unlike SFT, training with RL on SWE-Bench [50] leads to improvement across domains, including code generation with library use [51], code reasoning [52], math [53], and language understanding [54]. Thus, for tasks that are easily verifiable, parametric training works very well [55]. However, as we discuss in Sec. 5, non-parametric training offers interesting avenues of research in circumventing catastrophic forgetting and the plasticity-stability dilemma, allowing continuous learning across episodes [2].

#### 3 Scaffolded Language Model

A scaffolded LM refers to an LM embedded in a multi-step process with tools. Such a structure allows for extending an LM’s utility to interact with digital systems in similar ways as humans do [5]. Scaffolded LMs involve tools and thus differs from meta-generation techniques [56] that employ multiple calls to an LM [57–59]. Meta-generation mimics search algorithms [60] for reasoning tasks [37] where each step consists of a call to an LM. Importantly, in scaffolded LMs, each call to an LM expects a tool response.

To provide readers with a practical understanding of how this structure allows LMs to perform sophisticated cognitive tasks and improve from non-parametric training, we organize this section as follows. We introduce the post-training interface of language models that allows interaction with tools and defines their behavior in natural language. Next, we discuss multiple aspects of tools including compositionality, reliability, safety, and built-in tools. Furthermore, we show how the multi-step process can be controlled to implement priors into how the task is solved. Finally, we discuss benchmarks to understand the shift induced by scaffolded LMs. Specifically, these benchmarks highlight the omnipresence of scaffolded LMs in mixed-autonomy settings, where the design of the scaffold defines the interaction dynamics between human and AI agents for sharing control and decision-making.

- 3.1 Post-trained Language Models

System

You have access to the following functions. Respond in JSON format.

MODELINPUT

{

"name": "search_web", "description": "Search the web for current information", "parameters": {

"type": "object", "properties": {

"query": { "type": "string", "description": "The search query"

}

}, "required": ["query"]

} }

You are an assistant that provides the user with information from the web.

###### User

Find the latest trip of IShowSpeed

###### Assistant

{"name": "search_web", "parameters": {"query": "IShowSpeed latest trip"}}

###### Tool Response

IShowSpeed’s most recent trip was a high-profile tour across China and Mongolia in March and April 2025.

- Figure 4 Example of chat history in LLaMA-3. Adapted from [61]. A chat history starts with a prompt comprising a system and user message in LLaMA-3. The system message enumerates available tools (as JSON schemas) and sets high-level behavior. The user message specifies the task and the data. During the multi-step process, the chat history is updated with assistant messages (LM responses), user feedback, and tool responses.

From a probabilistic perspective, given an input (corresponding to the chat history) x = (x1,...,xT) consisting of a sequence of tokens xt ∈ V, a post-trained LM generates a response y = (y1,...,yN) conditioned on x in an autoregressive manner:

N

pθ(yt | x,y<t).

pθ(y | x) =

t=1

During post-training, the LM is trained to follow a chat format. Thus, the response follows the chat format and corresponds to a tool call or a response to the user. As shown in Fig. 4, the multi-step process in a scaffolded LM produces a chat history where messages come from different roles. Specifically, a chat format is a protocol that specifies how to convert a chat history into a single sequence of tokens x that the post-trained LM both expects as input and emits as output. For instance, the “tool” role is used for tool responses. In other words, we can understand a chat format as a protocol consisting of

1. a finite set of roles, e.g., system, user, assistant, tool.

1. a set of special tokens that act as markers, such as role headers, channel indicators, message delimiters, tool-call fences, and termination markers [62].

Importantly, the system message and user message specify how the model should behave as well as when and how to use tools. A chat starts with a system message and a user message, and we refer to them as the prompt. Non-parametric training steers the behavior of scaffolded LMs by optimizing the prompt and tools.

By default, it is recommended to adhere to the provider’s native chat format with explicit role headers (system, user, assistant, tool), placing tool outputs under the dedicated tool role and assistant generations under the assistant role. This aligns with how post-trained LMs are supervised and preserves the intended privilege ordering [63]. During post-training, the LM is explicitly trained to treat each message source (system → user → tool output) as having an explicit privilege order [63]. Thus, if there’s a conflict between messages from different sources, the LM is trained to prioritize them based on this privilege order. This avoids prompt-injection [64], jailbreaks [65], and system-prompt extraction [66]. This process is different from knowledge conflict [67, 68] that focuses on factuality. It arises when there’s a conflict between parametric knowledge (what the model “knows”), contextual knowledge (retrieved content), and instruction knowledge (user/system assumptions or edits). Accordingly, Claude’s system prompt includes explicit instructions to avoid invoking external tools on topics the model already “knows,” which could be a strategy to avoid knowledge conflicts to some extent [66,69].

##### 3.2 Tools

A tool can be virtually anything, including the interface to a digital system, a scratch pad for thinking [70], a request to the user [71], another LM [72], or developer-defined code. We define a tool as an external capability that the LM can invoke via the tool API [7,8]. Moreover, we define a function as a tool defined by a JSON schema. Similarly, we define tool calling as a specific type of response from the LM that includes a call to one of the available tools and a corresponding plain text for that tool (e.g., a query to a search engine). We define function calling as a specific type of tool call that corresponds to a structured JSON. It is important to note that the order of arguments in a function matters because post-trained LMs are autoregressive models [73,74].

During post-training, LMs may be trained to use built-in tools, which we call tool-integrated reasoning [75]. Built-in tools are executed remotely on the provider side within temporary secure sandboxes, rather than on the user’s local machine. Thus, built-in tools are only used for seeking external information to improve their reasoning process rather than advancing on the task itself. Built-in tools [23,70] typically include:

- • a web search API [76] allows the LM to tackle tasks with public data. The LM typically generates a response by citing the retrieved content.
- • a vector database [77–81] for storing data as embeddings that can be retrieved through top-k retrieval for a query. It is often used for accessing private data or as a memory module [82].
- • a computer use tool [83] that interacts with a headful browser. Unlike the web search API, it allows the LM to use interactive websites.

The multi-step process in scaffolded LMs naturally allows sequential tool composition, where tools are called sequentially in a coordinated manner to solve a task. In that sense, scaffolded LMs start by producing a plan in plain text to improve tool composition [75]. CodeAct [84] proposes to use executable Python code as a unified way to call tools. As shown in Fig. 6, code inherently supports both control flow (e.g., conditionals and loops) and data flow, allowing intermediate results to be stored and reused via variables when coordinating multiple tool calls. They [84] find that it achieves comparable or better performance on sequential tool composition. However, this approach is vulnerable to executing unsafe code and requires sandboxing. Even with sandboxes, strict capability scoping and comprehensive controls are needed: network and file-system egress restrictions, timeouts, resource quotas, import allow-lists, AST/static analysis, and immutable audit trails. Building and operating that infrastructure is nontrivial. Moreover, code-injection attacks become more dangerous because untrusted tool outputs (e.g., web pages or file contents) can be stitched into code and executed. Additionally, agents with access to the internet are exposed to a wide variety of attacks where tools return instructions, when followed by the LM, leads to leaking personal data [64].

Besides, LMs may also occasionally generate invalid tool calls (e.g., syntax errors), wrong tools, or irrelevant tools. For instance, irrelevant tool calls may be a result of reward hacking during post-training [85]. Similarly, the LM may face unexpected tool execution failures (e.g., rate-limit). To account for these, developers must implement error-handling mechanisms to catch exceptions during tool execution and provide feedback to the LM for recovery. Additionally, the way tools are defined and described inside the system message affects reliability. Clear, specific descriptions and intuitive names help the LM

choose the right tool. Additionally, clear description provides explicit guidance for the LM to learn when to use this tool as opposed to learning it through RL which is prone to reward hacking [85]. Thus, the optimization of prompts and tools is important.

##### 3.3 Multi-Step Process

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

- Figure 5 Multi-step process of a scaffolded LM. The distinctive trait between agent and workflow is that for each workflow step, a different prompt is written to pre-define the sub-task at that step. This makes workflows much more efficient and predictable than agents. However, workflows are engineered for specific scenarios. In this figure, we assume the case where the tool execution happens on the developer side. For built-in tools, tool execution happens on the provider’s side [23].

A scaffolded LM involves a multi-step process with tools. Each step consists of an interaction between the assistant and tool. As shown in Fig. 5, guiding the model at each step with tailored prompts reduces its decision-making burden. Thus, in practice, a workflow involves pre-defining the number of steps and crafting appropriate prompt for each step, while an agent focuses on engineering a single prompt and the set of tools.

These approaches mainly differ in how much autonomy the LM has, and we can also refer to this autonomy as the degree of agenticness [86]. An agent works well in unstructured scenarios, i.e., in cases where it is hard to anticipate all the scenarios that the LM might face. In contrast, a workflow approach works well when we know the solution path to a problem and can thus directly define each step required to solve it. A good example is solving GitHub issues in SWE-Bench [50]. A workflow such as Agentless [49] solves GitHub issues with the following steps. Roughly, it first calls the LLM with the GitHub repository and issue description to provide a list of files that potentially contain the bug. Those files are then used to prompt the LLM to choose relevant content inside those files. Then, it asks the LLM to generate multiple patches. In comparison, an agent such as Openhands [5] provides the LM with a terminal and a code interpreter to interact with the repository.

- 3.3.1 Agents

Agents solve tasks by deciding at each step which tool to use and how to use it. They also determine when a task is solved based on their own interpretation of results and goals. The chat of an agent starts with a prompt and each step updates the chat history. The agent continues until it decides to stop or it reaches the maximum number of steps, and returns a response to the user that summarize what it did. Besides providing the right tool to our agent, an important design of the agent is how it composes tools. In particular, code execution [84] offers a unified way to compose tools with control flow and data flow, and typically results in a shorter horizon. Another approach prompts the agent to provide a plan beforehand [75].

- 3.3.2 Workflows

Workflows [49,87,88] shine in applications where a solution path can be pre-determined. In contrast to agents, they are rendered more predictable and reliable by limiting the LM’s autonomy. At each step, the input x consists of a different prompt that specify the sub-task at that step. Thus, workflows consist of a pre-determined number of steps. It is more cost-efficient as it avoids any unnecessary exploration steps [49]. Unlike agents that scale with the number of steps [89], workflows scale by generating multiple responses in parallel [56,90,91]. Specifically, by fixing the solution path, the output format at each step is predictable. Thus, workflows use meta-generation techniques [56] such as best of n sampling or majority voting [92] to scale test-time compute. For instance, in [88], the AI scientist-v2 workflow can reach workshop-level papers by implementing tree-based methods on top of its previous version [87]. In particular, by generating multiple solutions at each step, it effectively generates multiple solution paths, and then uses heuristics (e.g., LLM as judge [93]) to select the best-performing path. The cost of parallelization in workflows mainly depends on the aggregation method. For instance LLM as judge adds one extra LM call, while pairwise judging adds quadratic cost. Meanwhile, aggregation can be done without an LM judge, e.g., unit tests [49] or majority voting [92].

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

- Figure 6 Tool compositionality. Left: tools are composed sequentially. As shown in the figure, some LMs support generating independent tool calls in parallel [7]. Right: executable Python code as a unified way to call tools [84]. Code inherently supports both control flow (e.g., conditionals and loops) and data flow, allowing intermediate results to be stored and reused via variables when coordinating multiple tool calls

##### 3.4 Benchmarks

We highlight a growing trend in recent benchmarks: from measuring pure intelligence (e.g., math olympiads) to measuring productivity gains (e.g., software engineering) [4]. The goal is no longer to assess whether a model solves a task perfectly in isolation, but how reliably and usefully it supports human workflows. Importantly, these benchmarks are used to compare LMs on products [5,94–96] that operate in mixed-autonomy. This is a setting where humans and AI work together, sharing control and

Benchmark Scenario Evaluation metrics SWE-Bench [50] Software engineering Unit test. SWE-Lancer [97] Software engineering pass@1 / pass@k, dollars earned, unit

test, SWE Manager by matching the original manager’s proposal.

HCAST [102] Machine learning, software engineering, cybersecurity

Binarized success rate by human-time buckets.

RE-Bench [103] AI Research Expert-graded task success; fraction of human pace; time-to-solution. PaperBench [101] AI research Rubric organized as a hierarchical tree of requirements. τ-Bench [98] Tool-agent-user interaction pass@k metric as a measure of reliability

of an agent. SWEET-RL [99] Multi-turn collaborative

Task success/win rate under simulated feedback.

reasoning

BrowseComp [100] Deep research Accuracy, calibration error.

Table 1 Benchmarks commonly used for scaffolded LMs.

decision-making responsibilities to achieve a goal. For instance, in vibe coding, the human guides intent and judges quality, while the AI proposes implementations and refactors.

To accurately gauge mixed-autonomy capability, these benchmarks align agent performance with human effort—measuring monetary return [97], simulating human collaborators [98, 99], exercising real software and APIs [50, 100, 101], and using time as a measure of task difficulty [102] because time is money.

SWE-bench [50] and SWE-Lancer [97] ground evaluation in software engineering tasks from GitHub or Upwork—tasks that are compensated in real-world settings and evaluated via functional test cases. Other benchmarks go beyond measuring raw accuracy. BrowseComp [100] evaluates accuracy and calibration error. In particular, the LM is asked to output a confidence (0–100) with its answer; BrowseComp then evaluates how well that confidence aligns with reality. HCAST [102] evaluates success rate as a function of human-calibrated task length (e.g., <15min, 15min-15h, 1-4h, 4h+). Additionally, they compute a “50% time-horizon”: the human task duration at which the agent has a 50% chance of success, fitted from per-task success vs. human time. RE-Bench [103] finds that currently humans scale better with increased time budgets to solve the task. For example, τ-Bench [98] and Sweet-RL [99] simulate human feedback using LMs to better reflect the back-and-forth nature of co-working. For tasks that are hard to verify, PaperBench [101] proposes a rubric organized as a hierarchical tree of requirements. The leaf nodes are precise, binary checks (pass/fail). Internal nodes aggregate their children by a manual, importance-based weighting, and the root score is the paper’s Replication Score. Tab. 1 summarizes these benchmarks.

#### 4 Training with Language Supervision

We view scaffolded LM π as semi-parametric models where the LM is the parametric component that is fixed. Instead, we consider the training of non-parametric variables, prompt and tools, that are represented in language and code. As shown in Fig. 8, non-parametric variables include the prompt and developer-defined code as tools. As shown in Fig. 7, we use an LM as an optimizer to generate the updated prompt or tool according to an objective (“improve prompt according to feedback on this execution trace”), an execution trace, and optionally execution feedback f, all provided in natural language.

- As in machine learning (ML), we assume a training, validation, and test dataset sampled from the same

(unknown) distribution. During training, we iteratively sample data point(s) and update prompt or tool.

- At inference, prompt(s) and tools are fixed (in the case of workflows there’s a multiple prompts, i.e., a different prompt for each step). Unlike in ML, where we train with Monte Carlo samples to prevent catastrophic forgetting [3], an LM optimizer works with variables represented in language or code as opposed to tensors. Thus, the role of batching is still an open research question [9]. The datasets D consist of user message corresponding to different tasks from which we obtain execution traces τ ∼ π. During training, the execution traces come from any scaffolded LM π′ or the current scaffolded LM

π. In that sense, it shares similarities with RL, where we have on/off-policy training and offline/online training [104]. However, the training dynamics significantly differ from those in RL. While RL is about finding good data (exploration) and imitating that good data (exploitation) [105], recent works in Sec.

- 4.3 learn only from bad execution traces. This is because failed execution traces often yield rich textual feedback, such as error messages or simulated user responses.

- Figure 7 LM optimizers interpret fuzzy objectives and execution trace with rich execution feedback. We illustrate a simple example with an agent. In practice, the objective contains the variables to optimize, and the execution trace shows how these variables were used. Optionally, execution feedback can be provided.

Central to this paradigm is an optimizer implemented with an LM as opposed to bootstrapping successful execution traces as in-context learning examples [106,107]. We further discuss how this paradigm differs from other approaches that have employed an LM and non-parametric components for improving a scaffolded LM. First, in Reflexion [108], the updates are not retrained across tasks and are only used for retrying a task. Second, CoALA [109] is an orthogonal approach that works at run-time with a specific scaffold. In particular, it requires changing the design of the agent with a memory module. This memory module not only serves as long-context memory but also for improving the agent over time. However, this requires the agent to make additional decisions on what and when to store or retrieve knowledge from this memory module. In training with language supervision, recent works focus on a setting where we are provided with a fixed dataset for training. Thus, it can be seen as amortizing the expensive inference of a cognitive architecture [109] into a training procedure.

Our organization follows the structure outlined in Fig. 8. Tracing their development, these methods have evolved to optimize more complex execution traces. Automatic prompt optimization focuses on an LM (as opposed to a scaffolded LM). It covers various types of objectives for improving the prompt of a target LM. Experiential learning trains agents where the tools and system prompts matter. Accordingly, the execution traces consist of interactions between the LM and the tools, from which the LM optimizer extracts reusable tools or insights. Finally, AutoDiff frameworks focus on workflows, which resemble classical software pipelines that include LM calls at key stages. These frameworks focus on recording the execution trace as a graph. The LM optimizer then uses this graph for execution and optimization.

##### 4.1 Prompt Optimization

In prompt optimization [110], we update the prompt of an LM (as opposed to a scaffolded LM). We prompt an LM optimizer with an objective (e.g., “improve this instruction according to the following examples of tasks where the model fails”) and (optionally) a textual feedback f. The updates made by the LM optimizer is evaluated on a validation set and the best one is selected. Thus, while these methods use a different objective in natural language, by heuristically selecting the best update on a validation set, they implicitly optimize for performance on the validation set. In its simplest form, we provide query and response examples and prompt the LM to generate the corresponding instruction [20,21].

Provided with different initial instructions, other methods define the objective as generating variations. Each variation is evaluated on a validation set, and the top-performing versions are retained. In [21], the objective is to generate semantically similar instructions. Evolutionary methods [111,112] combine parts of multiple instructions and then introduce random changes; we call this process crossover and mutation. In OPRO [113], we define the objective as generating instructions with a higher score using previous instructions, their scores, and a random sample of input-output pairs from the training data. To introduce fine-grained variations of instructions, SAMMO [114] uses a more granular representation of prompts by breaking them down into smaller sections, such as input formatting. This fine-grained structure allows SAMMO to adjust specific components, such as changing the input format from raw text to a structured format (e.g., XML). In a similar vein, ABO [115] implements the instruction as step-by-step guidelines. Thus, the LM optimizer only updates the specific step that failed.

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

- Figure 8 Evolution of LM optimization approaches. LM optimization approaches have evolved from prompt optimization, then agents, and finally workflows. In all approaches, an LM acts as an optimizer for generating updates on prompts or tools. Top: prompt optimization focuses on LMs as opposed to scaffolded LMs. The LM optimizer uses a single or multiple LM calls. Middle: experiential learning extends to agents and focuses on extracting insight from tasks and storing them in the system message, or refining tools. The LM optimizer uses a single LM call to update them from successful or unsuccessful execution traces. Bottom: AutoDiff frameworks focus on workflows where computation is represented as a graph interleaving LM calls and developer-defined code as tools. Using textual feedback, the LM optimizer can optimize nodes of the graph (e.g., system prompts).

Another line of work uses an LM to generate rich textual feedback f, and defines the objective as following the feedback f [38,39,116,117]. Importantly, textual feedback is obtained from sample queries where the target LM fails. Additionally, these works also propose summarizing multiple pieces of feedback [116] or updates [117].

Self-discover [118] focuses on the output structure of the instruction. The objective is to generate a JSON format with keys representing distinct reasoning steps. By filling in each key in an autoregressive manner during generation, it guides the target LM’s reasoning steps.

##### 4.2 Experiential Learning

We borrow the term experiential learning from ExpeL [10] to refer to the training of agents with language supervision. The LM optimizer considers an objective (e.g., “extract a general insight on why you failed this task”, “extract reusable tool from execution trace”), prompt s0, execution trace τ, optionally a textual feedback f, and generates an insight or a tool. In particular, the execution trace τ of agents consists of a sequence of tool calls/responses and ends with a response to the user. As discussed in Sec. 3.3.1, tools play an important role in agents. As shown in Fig. 9, the objective corresponds to either improving the set of tools or extract insights that are provided to the agent’s system prompt. Specifically, prior works leverage successful and unsuccessful execution traces with different objectives.

- 4.2.1 Insights Extraction

ExpeL [10] starts by collecting a dataset for training consisting of prompt s0 and corresponding execution trace τ. Using a Reflexion agent [108], multiple execution traces may be sampled for a prompt s0. In particular, the Reflexion agent allows the agent to retry the task upon failing, which allows the collection of unsuccessful and successful trajectories (τsucc,τfail) on the same prompt instance s0. The LM optimizer refines a list of insights from successful and unsuccessful execution traces (τsucc,τfail) for a given prompt s0 or from k successful trajectories (τ1succ,...,τksucc). At inference, those insights are fixed and provided to the scaffolded LM through its system prompt.

Follow-up works on ExpeL [10], such as AutoGuide [119], modify insights to be selectively added to the context at each step. Insights are specific to a state and contain a description of the state in which it is applicable. However, these methods require additional cost at inference since an LM summarizes the current state and then uses it to select the corresponding state-specific insight. AutoManual [120] modifies ExpeL [10] to prevent distributional shift in offline RL, where at test-time the state distribution is different [121]. However, their paper does not show strong evidence that training agents from language supervision suffers from distributional shift [121]. Additionally, MSI [122] proposes to categorize each insight and retrieve the relevant one at test-time based on pre-defined rules.

ExpeL [10] also explores new settings such as transfer learning, where we use insights learned from a source dataset {ssource0 } and transfer it to a target dataset {starget0 }. In parametric training [123], it consists of initializing the weights with the model trained on the source dataset. Then, finetuning these weights on the target dataset. In ExpeL, they initialize with insights extracted from a source dataset. Then, the LM optimizer’s objective is to generate refined insights to the target task provided with an execution trace (from a human). Note that the update is done in a single shot thanks to the LM’s ability to interpret intricate objectives. Additionally, they manually inspect execution traces from the agent with and without the insights. For instance, in ALFWorld [124], one insight would be “when searching for an item, consider its nature and its typical usage.” This leads the agent to update its belief [125] and avoid unnecessary actions to find objects. Furthermore, they find that another insight, “if an attempt to interact with an item fails [...] consider alternative actions or locations,” leads to self-correction. These findings are important, they showcase that simple high-level insights influence the behavior of agents in some predictable way.

- 4.2.2 Tools Optimization

Tool optimization [11,126,127] focuses on refining developer-defined code into more effective routines. The main challenge of tool optimization is ensuring that the skills are reusable and useful across episodes [128].

A related term to tool, but slightly more general is skill [129]. It refers to both a textual representation of a sequence of action and developer-defined code as tool. The term “skills” emphasizes two aspects of tool optimization. First, we want to make tools more efficient, thereby achieving tasks with fewer steps. Second, similarly to unsupervised RL (URL) [130,131], we want to extract the set of behaviors possible in the environment. Drawing an analogy with principal component analysis, before we are given any tasks, we want to know what is the basis of behavior that exists in some environment. Similarly, these methods [129,132] stores skills into a vector database and retrieve corresponding skill at inference. For instance, in CER [132] skills are represented as a sequence of tool calls with a high-level description of the skill. In AWM [129], they explore different textual representation of skills. We can also understand these methods as ‘curating’ in-context samples for retrieval of similar experiences in ExpeL [10]. However, the textual representation represents a challenge for formal verification [11].

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

- Figure 9 Experiential learning focuses on tools and insights. Successful execution traces are used for extracting reusable tools [11] while unsuccessful execution traces are used to refine tools [126]. Additionally, we can contrast successful and unsuccessful trajectories on the same task to extract insights [10].

At each step, LearnAct [127] and AgentOptimizer [126] sample an execution trace τ ∼ π(s0), and refine the set of tools for unsuccessful execution traces τfail. LearnAct uses execution feedback f corresponding to execution errors. The LM optimizer’s objective is to address f by either revising, adding a tool, or updating the system prompt with information about the tools. AgentOptimizer does not rely on the execution feedback f. The LM optimizer modifies the set of tools at each step by revising, removing, or adding a tool to solve the task that the model failed on. Moreover, they propose heuristics to account for the LM’s inability to correctly understand the failure modes and update tools. At each epoch, they evaluate the performance of the agent on the training data. If the performance is lower than the previous epoch, they roll back the set of tools. However, this is expensive and quickly leads to overfitting, where tools are not reusable.

In contrast, ASI [11] extracts reusable tools from a successful execution trace. One important challenge in extracting a reusable tool is to make sure that it is both useful and bug-free. AgentOptimizer [126] checks the performance on the training set with and without the updated tools by simulating again on the same query. An LM is used to rewrite the execution trace with only the extracted tools. Then, the agent attempts to finish the execution. If it successfully solves the task, then the extracted tool is kept. In contrast to transfer learning for insights, ASI explores domain adaptation [133] in web browsing. They find that the agent can effectively reuse tools and adapt tools to the new domain.

Similar to unsupervised RL [130], recent works propose to use an LM to both propose tasks and extract tools from rollouts on these tasks. In Voyager [128], the LM optimizer refines tools based on execution feedback and adds them to the skill set when successfully used to achieve a task. SkillWeaver [134] extracts tools directly from successful execution traces. Then, it produces test cases to check the longterm viability of the tool.

##### 4.3 AutoDiff Frameworks

The term autodiff is inspired by deep learning frameworks [135] in the sense that it implements elementary operations for defining and optimizing workflows [9,27,28,136]. The frameworks represent execution traces with a directed acyclic graph (DAG) where nodes correspond to variables or operations and edges represent dependencies. They roll out execution traces τ ∼ π and obtain execution feedback f. Subsequently, prior works [9,28,136,137] use the chain rule to back-propagate f through the graph. Each variable of the graph is updated by the LM optimizer according to its feedback. Alternatively, Trace [27] converts the DAG into text akin to a Python traceback. The LM optimizer updates variables in an autoregressive manner. We organize this section as follows. First, we cover graph abstraction to understand which variables of the workflow are optimized in different frameworks. Second, we focus on how execution traces are recorded and used for optimization.

- 4.3.1 Graph Abstraction

We refer to graph abstraction as the representation of nodes in the graph. This abstraction is foundational because it determines how workflows are formalized, traced, and subsequently optimized. There are two

types of nodes in the graph: variables and operations. Variables correspond to non-parametric components of the workflow (e.g., system prompt) or intermediate values (result from operations). Operations correspond to computation done on variables, e.g., Python functions, LM calls. As an analogy with graph representation for neural networks, variables would be weights, and operations would be layers. However, different from neural networks, any node of the computational graph can be optimized, including operations [27]. We distinguish between two lines of work, focusing on prompts or Python functions. It is important to note that in this survey, we focus on workflows within the scope of scaffolded LM. Thus, we do not consider applications of these frameworks for optimization instances (as opposed to workflows).

TextGrad [9] and other variants [28, 136, 137] focus on textual variables. In particular, operations include concatenations and LM calls [9]. Accordingly, trainable variables correspond to the system prompt or user input template of an LM call.

Trace [27] focuses on general-purpose workflows. Importantly, operations are not limited to LM calls, they can be arbitrary Python functions. In practice, we wrap Python functions with the Python decorator “bundle”. Additionally, they support the optimization of nodes corresponding to operations. In that sense, it goes beyond prompts to optimizing the workflow’s developer-defined tool.

##### Chain Rule

- 1 # step 1
- 2 output = workflow(user_message)
- 3 feedback = compute_feedback(output)
- 4
- 5 # step 2, compute the grad field of variables
- 6 feedback.backward()
- 7
- 8 # step 3, update each variable according to grad
- 9 optimizer.step()

##### Formatting of execution trace

- 1 # step 1
- 2 output = workflow(user_message)
- 3 feedback = compute_feedback(output)
- 4
- 5 # step 2, format prompt template of optimizer
- 6 output.backward(feedback)
- 7
- 8 # step 3, update variables autoregressively
- 9 optimizer.step()

- Figure 10 Graph execution and optimization using chain rule vs. formatting of execution trace. AutoDiff frameworks consist of three key steps: (1) forward computation, (2) backward feedback propagation, (3) variable updates. Both approaches differ during their backward pass. Left: using the chain rule, we compute a grad field on each variable corresponding to a feedback that addresses the feedback on its child variable. The LM optimizer updates each trainable variable following its grad field. Right: it performs a reverse topological sort on the computation graph starting from the output to format the prompt template of the optimizer. Together with the feedback, the LM optimizer generates trainable variables in an autoregressive manner.

- 4.3.2 Graph Execution and Optimization

We refer to graph execution and optimization as the process of sampling an execution trace τ ∼ π(s0) and optimizing variables of the graph according to the execution feedback f. As shown in Fig. 10, it involves three key steps: (1) forward computation, (2) backward feedback propagation, (3) variables update. During forward computation, we collect execution feedback f. Then, backward feedback propagation performs a reverse topological sort to either apply the chain rule on the execution feedback [9,28,136,137] or format a prompt template with the DAG [27]. On the one hand, the chain rule enables structured credit assignment by propagating feedback through the graph in a fine-grained manner. On the other hand, encoding the entire execution trace as a single prompt template is more token-efficient.

In the context of optimization in the language space, the chain rule does not refer exactly to the differentiation of composite functions in calculus. Instead, it points to the idea of when a transformation is applied to the output of another transformation. Then, the overall effect on an input can be understood by systematically combining the influence of each transformation in sequence. As shown in Fig. 11, and following the graph abstraction presented in Sec. 4.3.1, operations, including LM calls and concatenation, create nodes and attach a grad function. Most importantly, the grad function contains a prompt for an LM that informs about that operation’s role. Similarly, each variable has a role. These roles are provided in text to account for heterogeneous variables and operations in workflows. Accordingly, the backward feedback propagation calls the attached grad function at each operation in the reverse order of the DAG. Each gradient is a textual feedback with respect to the child node of that operation. For each trainable variable, the LM optimizer’s objective follows the feedback as well as the variable’s role. Follow-up work extends Textgrad [9] in the following ways. AdalFlow [28] extends operations to retrievers [77]. GASO [137] modifies the gradient operation to take into account neighboring nodes.

system message user message

concat(system message, user message)

input

LM call

response

evaluate(output)

feedback

- Figure 11 Chain rule in TextGrad. Forward computation is represented with solid black arrows. Calling “feedback.backward()” (dashed red arrows, bent) applies the chain rule on the feedback f. It computes a gradient for each variable (in green), provided with gradient functions attached to operations (in yellow).

A conceptually more efficient approach is Trace [27]. At each step of the optimization process, the chain rule requires multiple calls to an LM. Furthermore, the LM optimizer also requires multiple calls to an LM. In contrast, Trace [27] formats a prompt template for the execution trace during backward feedback propagation. This means the LM optimizer generates all updated variables in an autoregressive manner, provided with the execution trace and feedback. Additionally, as discussed in Sec. 4.3.1, Trace [27] uses a different graph abstraction that allows for arbitrary Python functions as operations. As opposed to Textgrad [9], this allows the LM optimizer to also consider operations during training. In that sense, we can optimize the workflow’s developer-defined tool.

#### 5 Beyond episodic learning

This section discusses non-parametric training as an approach towards agents that inhabit streams of experience1). We describe it as agents that continuously learn from experience. In particular, current models only learn and improve from their mistakes within an episode. At the end of the episode, the chat history is cleared and the model will recommit the same errors and rediscover the same solutions in future episodes.

This key capability of learning beyond episodic interaction is crucial for mixed-autonomy settings [95,96]. In this setting, there’s significant contextual recurrence across episodes; tasks exhibit temporal coherence, wherein humans and AI dynamically share control based on their respective strengths. It also features rich and interpretable feedback, where humans identify errors and offer guidance [98,99]. For instance, agents rarely succeed on the first attempt, as user prompts often omit crucial information—highlighting a mismatch between latent intent and explicit input [138]. As a result, each episode may involve multiple rounds of user feedback, where the agent corrects its behavior. As illustrated in Fig. 12, we propose a setting in continual learning with non-parametric updates as opposed to parametric updates. Unlike scalar-based update, language-based update is interpretable and does not suffer from catastrophic forgetting [3], while being compatible with closed-source models.

Below, we discuss key capabilities required for agents to inhabit streams of experience using nonparametric training with language supervision. To support such learning, we use an LM as an opti-

1) We purposely focus on agents, as mixed autonomy settings typically involve an agent [2,30].

mizer that interprets rich user feedback, execution traces, and intricate objectives (e.g., “minimize redundant steps based on prior interaction history”, “align with user-specified constraints”), and update non-parametric variables. Compared to parametric training, it uses rich textual feedback and does not suffer from sparse rewards [139] and catastrophic forgetting [3]. Each episode begins when the human delegates control to the AI via a user query. The AI then autonomously interacts with tools to attempt task completion. After producing an initial result, the human may intervene to offer feedback—correcting errors and providing guidance. The AI integrates this feedback and continues the task. The episode concludes once the AI successfully completes the task or the human resumes control to begin a new one. In formula, given the system message and user message s0, an episode τ is a sequence of LM’s response at, tool response ot, and human feedback f

τ = (s0,a0,o0,...,f,...,at,ot,...,f,...,aT).

At the end of each episode, a non-parametric update is applied to the scaffolded LM using an LM optimizer. We posit that an effective learning system in this setting must exhibit several key properties:

- • Scalable learning across similar tasks. The system should improve as it accumulates experience. This defines a new scaling axis—performance gains through recurrence. For instance, maintainers resolve GitHub issues 5-18x faster than external contributors [140], suggesting the value of task familiarity.
- • Plasticity and stability. The model must quickly adapt to new tasks (plasticity) while not forgetting how to solve tasks it previously learned (stability). If it forgets, it should recover rapidly from supervision.
- • Efficiency. It should avoid excessive context consumption to remain effective over long-term interactions spanning weeks to months.
- • Interpretability. Especially in mixed-autonomy settings, the AI’s internal state must remain legible so that humans can anticipate its behavior. Carrying over entire trajectories verbatim hinders interpretability.

##### 5.1 Scalability

In mixed-autonomy deployments, users repeatedly face related sub-problems under shared tools, codebases, or data silos. This induces recurrence: the same APIs, preconditions, and failure modes reappear across episodes. If a scaffolded LM fails to improve under recurrence, it repeatedly pays exploration cost and re-commits prior mistakes. We therefore target scaling for tasks that share similar context (e.g., same codebase): systematic gains as similar tasks recur, achieved by updating non-parametric variables (prompts, tools, step policies) with language supervision rather than gradient steps.

Streams of experience naturally provide textual signals—human corrections, rationales, constraint reminders, and execution errors—alongside verifiers (unit tests, checkers, compilers) when available. Such signals (i) localize error causes (wrong tool, argument, or precondition), (ii) support immediate, auditable edits to prompts and tools, and (iii) avoid catastrophic forgetting inherent to weight updates [3].

We highlight two mechanism for scaling in this context:

- • Insight caching (Sec. 4.2.1): distill state-invariant heuristics or decision rules from traces (e.g., “if login required, navigate to /signin before form submit”) and insert them into high-privilege prompts [10,119].
- • Skill/tool induction (Sec. 4.2.2): refactor recurring action patterns into reusable, typed tools with pre/post-conditions and minimal tests [11,126–128,134].

In-episode reflection [108] improves retries but does not persist if no artifact is updated. RL-tuned reflection controllers (e.g., better verbal feedback and plans) still pay the same exploration cost when a similar task reappears, unless insights or tools are written back [141]. In contrast, agents that inhabit streams of experience synthesize durable artifacts from traces: principle extraction from failures, explicit tool learning.

Prior work equips agents with a memory module [82]. This has long been seen as an important step towards self-evolving agents [142,143]. For instance, in ExpeL [10], they save successful trajectories in an

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

- Figure 12 Continual learning with non-parametric vs. parametric training. Top: we focus on non-parametric training where users provide rich textual feedback at each episode. This feedback is naturally present as users interact with the agent to adjust the answer. We update the agent with an LM optimizer after each episode, allowing agents to learn across episodes [2]. Bottom: parametric training uses scalar feedback and focuses on solving catastrophic forgetting that occurs when the i.i.d assumption doesn’t hold at each gradient step [3].

experience buffer. At inference, it retrieves similar experiences and appends them to the context of the agent. Similarly, Dynamic Cheatsheet [144] stores and retrieves accumulated strategies, code snippets, and general problem-solving insights at inference time. While this helps the agent’s performance, it does not scale well. An important property in the recent breakthrough of AI is the scaling with more data points [145]. We expect the agent to find more efficient solutions as it encounters similar tasks. Retrieving similar experiences is limited to a few experiences. Generative agents [82] use the memory module to simulate real human behaviors that plan and act over very long horizons. In particular, the focus of generative agents is to account for the limited context of LMs by “compressing” previous interactions and retrieving them later. However, these works focus on simply storing experiences [10] or compressing prior interactions [82] to fit multiple interactions beyond the context window.

##### 5.2 Plasticity and Stability

Agents that inhabit streams of experience must adapt quickly (plasticity) while not regressing on previously mastered patterns (stability). Language-level edits (prompts, tools, validators) help by localizing changes, preserving provenance, and enabling rollback—circumventing catastrophic forgetting common in parametric updates [3].

Plasticity and stability requires tracking beyond binary success [146,147]:

- • Assistance rate: mean human interventions per task. We want to reduce the assistance rate over time without affecting the assistance rate of previous tasks.
- • Backward/forward transfer (BWT/FWT): BWT measures accuracy on prior tasks after an edit; FWT measures improvement on new tasks after learning on previous tasks.

As an analogy, we want to implement new features to a codebase without breaking previous features [50]. We highlight some mechanism to preserve stability:

- • Scoped edits: restrict updates to the smallest locus (one tool, one step prompt) via graph credit assignment [9,27].
- • Gated promotion: accepte update only if they pass unit tests [50].
- • Versioned artifacts: keep prior versions + provenance (trace, feedback, objective) for audit and instant rollback [11,126].

##### 5.3 Efficiency

Dumping prior trajectories or long memories into context windows scales poorly as context-windows are limited. Non-parametric training instead compresses experience into durable artifacts—concise insights, executable tools, and step-specific prompts—so future episodes consume similar or fewer tokens while solving harder variants. For instance, we can update prompt and tools with

- • priors: encode stable facts about the environment: API maps, repo layout, naming conventions, data schemas, units, and invariants. This allows fewer exploratory calls.
- • constraints: specifies constraints according to user preferences.
- • docstring: short, copy-ready examples and counter-examples in the tool description.

Besides considering the efficiency of the scaffolded LMs, it is important to also consider the LLM optimizer’s efficiency. Inspired by sleep-time compute [148], we can imagine separating online learning from offline consolidation to preserve latency. In particular, recent work proposed the concept of sleeptime compute [148]. This paper is motivated by the fact that LM user queries consist of a shared task context c and queries. Thus, when the model is idle, it proactively updates the shared context by inferring likely future queries and restructures the context in ways that may be beneficial at test time.

##### 5.4 Interpretability

In mixed autonomy, humans must anticipate and steer agent behavior. Carrying opaque internal states or long, free-form memories across episodes hinders oversight. Non-parametric updates promote legibility: each change is a human-readable diff, on prompt and tools, linked to the trace and feedback that motivated it. In particular, working with language and code allows to easily track:

- • Provenance: instead of storing tensors, it stores pure text.
- • Diffs: relative changes that happen at each step are easily interpretable.
- • Information: the hierarchical structure of the chat format allows to control how information is interpreted [63].

Additionally, several effective approaches operate on the parametric component, offering adaptation but typically with weaker legibility than prompt/tool edits:

Continual learning. In parametric training, continual learning aims to address a core limitation of parametric training: catastrophic forgetting. Notably, models are optimized using i.i.d. batches sampled from a fixed dataset. However, consider a setting where a model is first trained to classify dogs, and later needs to be extended to also classify cats. Simply continuing training on the new data can lead to the model overwriting its knowledge of dogs—resulting in a phenomenon known as catastrophic forgetting. A naive solution might involve freezing the feature extractor and training only the output head. Yet this often leads to poor performance, as the learned features are not well-suited for the new task (e.g., distinguishing cats). Continual learning addresses this by developing methods that preserve previously learned knowledge while integrating new information, without requiring retraining on the full original dataset. Its motivation is different: to avoid the inefficiency of reprocessing previously seen data—especially when that data has already been well-classified. The key insight is that gradient-based learning on i.i.d. samples can be wasteful when the model only needs to adapt at the margin. Furthermore, gradient-based learning faces the plasticity-stability dilemma [149–152]: models are highly plastic early in training, quickly adapting to new data, but the plasticity decreases over time. As shown in Fig. 12, besides the aforementioned challenges, continual learning would also require humans to provide labels (in the form of reward or demonstration). In contrast, updating non-parametric variables does not

exhibit a decaying plasticity over time and mixed-autonomy settings naturally provides feedback in the form of textual corrections and guidance.

Test-time training. test-time training methods [153] function as memory mechanisms that compress information directly within the embedding space, rather than relying on text-based representations. For instance, Lattice [153] introduces a compression model that dynamically updates and maintains a compact representation of contextual history by operating on the key-value caches of a transformer in a streaming fashion. These methods typically involve architectural modifications to the neural network and offer an orthogonal approach to standard prompting or finetuning. Potentially, the right objective function for these methods could enable emergent behaviors—such as learning across episodes—in the embedding space.

#### 6 Conclusion

This survey introduces a paradigm called training of scaffolded LMs with language supervision. It organizes the intricate literature in prompt optimization, LM pipelines, experiential learning agents, AI workflow optimization, and LM agents. We focus on the structure around LMs where AI and human share control and decision-making responsibilities to achieve a common goal [4,5,94], and refer to this structure as the scaffold. We view scaffolded LMs as semi-parametric models where the scaffold represents the non-parametric component and the LM refers to the parametric component. From a learning perspective, by focusing on the training of non-parametric variables, we view this new paradigm as learning from language. In practice, scaffolded LMs receive instructions, interface with tools, and receive feedback all in natural language. Accordingly, we use an LM as an optimizer to interpret this rich language supervision. In contrast, parametric training has excelled in learning from demonstration (supervised learning), exploration (reinforcement learning), and observation (unsupervised learning), using well-defined loss functions. Furthermore, it discusses a key capability missing in current scaffolded LMs: the ability to continuously learn across episodes [2]. Parametric training inherently suffers from catastrophic forgetting. Instead, scaffolded LMs are naturally exposed to rich user feedback [5,94]. Therefore, we can use an LM optimizer to interpret fuzzy objectives [154] and rich feedback. Compared to parametric training, it is interpretable, efficient, and compatible with closed-source models.

We discussed agents that inhabit streams of experience as AI systems that can continuously learn in settings with significant contextual recurrent. From a product perspective [95], LMs are scaffolded to create product that works with the human in a mixed-autonomy settings. We hope this survey inspires future research in adaptive AI systems in mixed-autonomy settings, where AI continously adapts to human feedbacks.

#### 7 Limitations

Our work focuses on organizing a diverse and intricate body of literature, but it does not aim to exhaustively trace how each idea has been applied across all prior work. For instance, we highlight certain benchmarks that are not consistently used across the papers we discuss [155]. This is because these benchmarks are often too challenging for current non-parametric training methods. This selective approach reflects our intent to extract broader conceptual insights rather than follow any one established framework. With the recent surge of interest in RL applied to LMs, many earlier methods and assumptions are becoming less central. As a result, we intentionally de-emphasize some threads of the literature that, while once influential, no longer align with the direction of current research. This allows us to focus on outlining key future research areas that involve learning from language.

In addition to learning from language, there has been limited work in learning to incorporate visual information, as some concepts—find me this cup—are difficult to convey purely through text. Multimodal prompt-based learning has been explored in prior work on robotics [156], where prompts may include both text and images, e.g., “bring me image of the cup.” Inspired by this, scaffolded LMs can also interleave textual and visual tokens within their system prompt. This is particularly beneficial when images convey information more compactly or clearly than text alone.

Acknowledgements This work was supported by the National Natural Science Foundation of China (Grant Nos. W2442033, W2442032, and 62461160309).

###### References

- 1 Yuchen Cui, Siddharth Karamcheti, Raj Palleti, et al. No, to the right: Online language corrections for robotic manipulation via shared autonomy. In: Proceedings of the 2023 ACM/IEEE International Conference on Human-Robot Interaction, New York, NY, USA: Association for Computing Machinery. 2023, HRI ’23. 93–101. URL https://doi.org/10.1145/3568162. 3578623
- 2 David Silver, Richard S Sutton. The era of experience, 2025. URL https://storage.googleapis.com/deepmind-media/ Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf
- 3 Raia Hadsell, Dushyant Rao, Andrei A Rusu, et al. Embracing change: Continual learning in deep neural networks. Trends in Cognitive Sciences, 2020, 24: 1028–1040. URL https://www.sciencedirect.com/science/article/pii/S1364661320302199
- 4 OpenAI. Openai charter, 2018. URL https://openai.com/charter/, accessed: 2025-04-25
- 5 Xingyao Wang, Boxuan Li, Yufan Song, et al. OpenHands: An Open Platform for AI Software Developers as Generalist Agents, 2024. URL https://arxiv.org/abs/2407.16741
- 6 Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, et al. MLE-bench: Evaluating machine learning agents on machine learning engineering. 2024. URL https://arxiv.org/abs/2410.07095
- 7 OpenAI. Function calling, 2025. URL https://platform.openai.com/docs/guides/function-calling, accessed: 2025-10-19
- 8 Zhiruo Wang, Zhoujun Cheng, Hao Zhu, et al. What are tools anyway? a survey from the language model perspective. In: First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=Xh1B90iBSR
- 9 Mert Yuksekgonul, Federico Bianchi, Joseph Boen, et al. Optimizing generative ai by backpropagating language model feedback. Nature, 2025, 639: 609–616. URL https://doi.org/10.1038/s41586-025-08661-4
- 10 Andrew Zhao, Daniel Huang, Quentin Xu, et al. Expel: Llm agents are experiential learners. In: Proceedings of the AAAI Conference on Artificial Intelligence, 2024, volume 38. 19632–19642
- 11 Zora Zhiruo Wang, Apurva Gandhi, Graham Neubig, et al. Inducing programmatic skills for agentic tasks, 2025. URL https://arxiv.org/abs/2504.06821
- 12 Long Ouyang, Jeffrey Wu, Xu Jiang, et al. Training language models to follow instructions with human feedback. In: Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper_files/ paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html
- 13 DeepSeek-AI, Daya Guo, Dejian Yang, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948
- 14 Tom Brown, Benjamin Mann, Nick Ryder, et al. Language models are few-shot learners. In: Advances in Neural Information Processing Systems, Curran Associates, Inc.. 2020, volume 33. 1877–1901. URL https://proceedings.neurips.cc/paper_ files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf
- 15 Yann LeCun, Yoshua Bengio, Geoffrey Hinton. Deep learning. nature, 2015, 521: 436
- 16 Ching-An Cheng, Andrey Kolobov, Dipendra Misra, et al. Llf-bench: Benchmark for interactive learning from language feedback, 2023. URL https://arxiv.org/abs/2312.06853
- 17 Anthropic. Claude, 2024. URL https://www.anthropic.com/claude, accessed: Month, Day, 2024
- 18 OpenAI. Chatgpt, 2024. URL https://openai.com/chatgpt, accessed: Month, Day, 2024
- 19 Matei Zaharia, Omar Khattab, Lingjiao Chen, et al. The shift from models to compound ai systems, 2024. URL https: //bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/
- 20 Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, et al. Dspy: Compiling declarative language model calls into selfimproving pipelines. CoRR, 2023, abs/2310.03714. URL https://doi.org/10.48550/arXiv.2310.03714
- 21 Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, et al. Large language models are human-level prompt engineers. In: The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023, OpenReview.net. 2023. URL https://openreview.net/forum?id=92gvk82DE-
- 22 Andrew Zhao, Reshmi Ghosh, Vitor Carvalho, et al. Are my optimized prompts compromised? exploring vulnerabilities of llm-based optimizers, 2025. URL https://arxiv.org/abs/2510.14381
- 23 OpenAI. New tools for building agents, 2025. URL https://openai.com/index/new-tools-for-building-agents/, accessed: 2025-04-17
- 24 Shunyu Yao. Language agents: From next-token prediction to digital automation. Ph.D. thesis, Princeton University, Princeton, NJ, USA, 2024. URL https://www.cs.princeton.edu/people/profile/yaos, doctor of Philosophy (Ph.D.) Thesis
- 25 Zhiheng Xi, Wenxiang Chen, Xin Guo, et al. The rise and potential of large language model based agents: a survey. Science China Information Sciences, 2025, 68: 121101. URL https://link.springer.com/article/10.1007/s11432-024-4222-0
- 26 Tula Masterman, Sandi Besen, Mason Sawtell, et al. The landscape of emerging AI agent architectures for reasoning, planning, and tool calling: A survey. CoRR, 2024, abs/2404.11584. URL https://doi.org/10.48550/arXiv.2404.11584
- 27 Ching-An Cheng, Allen Nie, Adith Swaminathan. Trace is the new autodiff - unlocking efficient optimization of computational workflows. CoRR, 2024, abs/2406.16218. URL https://doi.org/10.48550/arXiv.2406.16218
- 28 Li Yin. AdalFlow: The Library for Large Language Model (LLM) Applications, 2024. URL https://github.com/ SylphAI-Inc/AdalFlow
- 29 Matthieu Lin, Jenny Sheng, Andrew Zhao, et al. Llm-based optimization of compound ai systems: A survey, 2024. URL https://arxiv.org/abs/2410.16392
- 30 Shunyu Yao. The second half, 2025. URL https://ysymyth.github.io/The-Second-Half/, blog post
- 31 Lei Wang, Chen Ma, Xueyang Feng, et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 2024, 18: 186345. URL https://link.springer.com/article/10.1007/s11704-024-40231-1
- 32 Zhipeng Liu, Xuefeng Bai, Kehai Chen, et al. A survey on the feedback mechanism of LLM-based AI agents. In: Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence (IJCAI), 2025. 10582–10592
- 33 DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv. org/abs/2501.12948
- 34 Guiyao Tie, Zeli Zhao, Dingjie Song, et al. A survey on post-training of large language models, 2025. URL https: //arxiv.org/abs/2503.06072
- 35 Kenny Shijian Lai, Jasur Mirzakhalov, Karan Singla, et al. A survey of post-training scaling in large language models. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Vienna, Austria: Association for Computational Linguistics. 2025. 2478–2510. URL https://aclanthology.org/2025.acl-long.140
- 36 Shangheng Du, Jiabao Zhao, Jinxin Shi, et al. A survey on the optimization of large language model-based agents, 2025. URL https://arxiv.org/abs/2503.12434
- 37 Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021

- 38 Xinyuan Wang, Chenxi Li, Zhen Wang, et al. Promptagent: Strategic planning with language models enables expert-level prompt optimization. In: The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024, OpenReview.net. 2024. URL https://openreview.net/forum?id=22pyNMuIoa
- 39 Reid Pryzant, Dan Iter, Jerry Li, et al. Automatic prompt optimization with "gradient descent" and beam search. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, Association for Computational Linguistics. 2023. 7957–7968. URL https://doi.org/10.18653/v1/2023. emnlp-main.494
- 40 Andrew Zhao, Quentin Xu, Matthieu Lin, et al. Diver-ct: Diversity-enhanced red teaming large language model assistants with relaxing constraints, 2024. URL https://arxiv.org/abs/2405.19026
- 41 Mingkai Deng, Jianyu Wang, Cheng-Ping Hsieh, et al. Rlprompt: Optimizing discrete text prompts with reinforcement learning, 2022. URL https://arxiv.org/abs/2205.12548
- 42 Weize Kong, Spurthi Amba Hombaiah, Mingyang Zhang, et al. Prewrite: Prompt rewriting with reinforcement learning,

2024. URL https://arxiv.org/abs/2401.08189

- 43 Minchan Kwon, Gaeun Kim, Jongsuk Kim, et al. Stableprompt: Automatic prompt tuning using reinforcement learning for large language models, 2024. URL https://arxiv.org/abs/2410.07652
- 44 Tianjun Zhang, Xuezhi Wang, Denny Zhou, et al. TEMPERA: test-time prompting via reinforcement learning. CoRR, 2022, abs/2211.11890. URL https://doi.org/10.48550/arXiv.2211.11890
- 45 Andrew Zhao, Yiran Wu, Yang Yue, et al. Absolute zero: Reinforced self-play reasoning with zero data, 2025. URL https://arxiv.org/abs/2505.03335
- 46 Yiping Wang, Qing Yang, Zhiyuan Zeng, et al. Reinforcement learning for reasoning in large language models with one training example, 2025. URL https://arxiv.org/abs/2504.20571
- 47 Yuxiang Wei, Olivier Duchenne, Jade Copet, et al. Swe-rl: Advancing llm reasoning via reinforcement learning on open software evolution. arXiv preprint arXiv:2502.18449, 2025
- 48 Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, et al. Is your code generated by chatGPT really correct? rigorous evaluation of large language models for code generation. In: Thirty-seventh Conference on Neural Information Processing Systems,

2023. URL https://openreview.net/forum?id=1qvx610Cu7

- 49 Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, et al. Agentless: Demystifying llm-based software engineering agents. CoRR, 2024, abs/2407.01489. URL https://doi.org/10.48550/arXiv.2407.01489
- 50 Carlos E Jimenez, John Yang, Alexander Wettig, et al. SWE-bench: Can language models resolve real-world github issues? In: The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id= VTF8yNQM66
- 51 Terry Yue Zhuo, Vu Minh Chien, Jenny Chim, et al. Bigcodebench: Benchmarking code generation with diverse function calls and complex instructions. In: The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=YrycTjllL0
- 52 Alex Gu, Baptiste Roziere, Hugh James Leather, et al. CRUXEval: A benchmark for code reasoning, understanding and execution. In: Proceedings of the 41st International Conference on Machine Learning, PMLR. 2024, Proceedings of Machine Learning Research, volume 235. 16568–16621. URL https://proceedings.mlr.press/v235/gu24c.html
- 53 Dan Hendrycks, Collin Burns, Saurav Kadavath, et al. Measuring mathematical problem solving with the MATH dataset. In: Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. URL https://openreview.net/forum?id=7Bywt2mQsCe
- 54 Dan Hendrycks, Collin Burns, Steven Basart, et al. Measuring massive multitask language understanding. In: International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=d7KBjmI3GmQ
- 55 Jason Wei. Asymmetry of verification and verifier’s rule, 2025. URL https://www.jasonwei.net/blog/ asymmetry-of-verification-and-verifiers-law, accessed: 2025-10-20
- 56 Sean Welleck, Amanda Bertsch, Matthew Finlayson, et al. From decoding to meta-generation: Inference-time algorithms for large language models. Transactions on Machine Learning Research, 2024. URL https://openreview.net/forum?id= eskQMcIbMS, survey Certification
- 57 Shengran Hu, Cong Lu, Jeff Clune. Automated design of agentic systems. CoRR, 2024, abs/2408.08435. URL https: //doi.org/10.48550/arXiv.2408.08435
- 58 Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, et al. AFlow: Automating agentic workflow generation. In: The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=z5uVAKwmjf
- 59 Jon Saad-Falcon, Adrian Gamarra Lafuente, Shlok Natarajan, et al. Archon: An architecture search framework for inferencetime techniques, 2024. URL https://arxiv.org/abs/2409.15254
- 60 David Silver, Aja Huang, Chris J Maddison, et al. Mastering the game of go with deep neural networks and tree search. Nature, 2016, 529: 484–489
- 61 Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. The llama 3 herd of models, 2024. URL https://arxiv.org/ abs/2407.21783
- 62 Dominik Kundel. Openai harmony response format. https://cookbook.openai.com/articles/openai-harmony, 2025. OpenAI Cookbook; accessed 2025-10-21
- 63 Eric Wallace, Kai Xiao, Reimar Leike, et al. The instruction hierarchy: Training llms to prioritize privileged instructions,

2024. URL https://arxiv.org/abs/2404.13208

- 64 OpenAI. Codex cloud: Internet access, 2025. URL https://developers.openai.com/codex/cloud/internet-access, accessed: October 22, 2025
- 65 Patrick Chao, Alexander Robey, Edgar Dobriban, et al. Jailbreaking black box large language models in twenty queries,

2023. URL https://arxiv.org/abs/2310.08419

- 66 Drew Breunig. Claude’s system prompt: Chatbots are more than just models, 2025. URL https://www.dbreunig.com/ 2025/05/07/claude-s-system-prompt-chatbots-are-more-than-just-models.html, accessed: 2025-05-11
- 67 Gaotang Li, Yuzhong Chen, Hanghang Tong. Taming knowledge conflicts in language models, 2025. URL https://arxiv. org/abs/2503.10996
- 68 Rongwu Xu, Zehan Qi, Zhijiang Guo, et al. Knowledge conflicts for llms: A survey. In: Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024, Association for Computational Linguistics. 2024. 8541–8565. URL https://aclanthology.org/2024.emnlp-main.486
- 69 Ásgeir Thor Johnson. Claude system prompt leak, 2025. URL https://github.com/asgeirtj/system_prompts_leaks/blob/ main/claude.txt, accessed: 2025-05-11
- 70 Anthropic. The "think" tool: Enabling claude to stop and think in complex tool use situations, 2025. URL https: //www.anthropic.com/engineering/claude-think-tool, accessed: 2025-04-22
- 71 LangChain. Human as a tool, 2025. URL https://python.langchain.com/docs/integrations/tools/human_tools/, accessed: 2025-05-11
- 72 OpenAI. Openai agents sdk, 2025. URL https://github.com/openai/openai-agents-python, accessed: 2025-05-11
- 73 R Thomas McCoy, Shunyu Yao, Dan Friedman, et al. Embers of autoregression: Understanding large language models through the problem they are trained to solve, 2023. URL https://arxiv.org/abs/2309.13638

- 74 Roger Grosse, Juhan Bae, Cem Anil, et al. Studying large language model generalization with influence functions, 2023. URL https://arxiv.org/abs/2308.03296, see page 45 for relevant details
- 75 OpenAI. gpt-oss-120b & gpt-oss-20b model card, 2025. URL https://arxiv.org/abs/2508.10925
- 76 Reiichiro Nakano, Jacob Hilton, Suchir Balaji, et al. Webgpt: Browser-assisted question-answering with human feedback,

2022. URL https://arxiv.org/abs/2112.09332

- 77 Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. In: Proceedings of the 34th International Conference on Neural Information Processing Systems, Red Hook, NY, USA: Curran Associates Inc.. 2020, NIPS ’20
- 78 Gautier Izacard, Edouard Grave. Leveraging passage retrieval with generative models for open domain question answering. In: Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, EACL 2021, Online, April 19 - 23, 2021, Association for Computational Linguistics. 2021. 874–880. URL https://doi.org/10.18653/v1/2021.eacl-main.74
- 79 Ori Ram, Yoav Levine, Itay Dalmedigos, et al. In-context retrieval-augmented language models. Trans. Assoc. Comput. Linguistics, 2023, 11: 1316–1331. URL https://doi.org/10.1162/tacl_a_00605
- 80 Anthropic. Introducing contextual retrieval, 2024. URL https://www.anthropic.com/news/contextual-retrieval, accessed: 2024-10-01
- 81 Yichen Jiang, Shikha Bordia, Zheng Zhong, et al. Hover: A dataset for many-hop fact extraction and claim verification. In: Findings of the Association for Computational Linguistics: EMNLP 2020, Online Event, 16-20 November 2020, Association for Computational Linguistics. 2020, Findings of ACL, volume EMNLP 2020. 3441–3460. URL https://doi.org/10.18653/ v1/2020.findings-emnlp.309
- 82 Joon Sung Park, Joseph C O’Brien, Carrie Jun Cai, et al. Generative agents: Interactive simulacra of human behavior. In: Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology, UIST 2023, San Francisco, CA, USA, 29 October 2023- 1 November 2023, ACM. 2023. 2:1–2:22. URL https://doi.org/10.1145/3586183.3606763
- 83 OpenAI. Operator system card. Technical Report, OpenAI, 2025. URL https://cdn.openai.com/operator_system_card.pdf, accessed: 2025-04-28
- 84 Xingyao Wang, Yangyi Chen, Lifan Yuan, et al. Executable code actions elicit better LLM agents. In: Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, OpenReview.net. 2024. URL https://openreview.net/forum?id=jJ9BoXAfFa
- 85 Shengjie Ma, Chenlong Deng, Jiaxin Mao, et al. Pou: Proof-of-use to counter tool-call hacking in deepresearch agents, 2025. URL https://arxiv.org/abs/2510.10931
- 86 LangChain. How to think about agent frameworks, 2025. URL https://blog.langchain.dev/ how-to-think-about-agent-frameworks/, accessed: 2025-04-22
- 87 Chris Lu, Cong Lu, Robert Tjarko Lange, et al. The AI Scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292, 2024
- 88 Yutaro Yamada, Robert Tjarko Lange, Cong Lu, et al. The ai scientist-v2: Workshop-level automated scientific discovery via agentic tree search. arXiv preprint arXiv:2504.08066, 2025
- 89 Jiayi Pan, Xingyao Wang, Graham Neubig, et al. Training software engineering agents and verifiers with swe-gym, 2024. URL https://arxiv.org/abs/2412.21139
- 90 Charlie Snell, Jaehoon Lee, Kelvin Xu, et al. Scaling llm test-time compute optimally can be more effective than scaling model parameters, 2024. URL https://arxiv.org/abs/2408.03314
- 91 Lingjiao Chen, Jared Quincy Davis, Boris Hanin, et al. Are more llm calls all you need? towards scaling laws of compound inference systems, 2024. URL https://arxiv.org/abs/2403.02419
- 92 Xuezhi Wang, Jason Wei, Dale Schuurmans, et al. Self-consistency improves chain of thought reasoning in language models. In: The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023, OpenReview.net. 2023. URL https://openreview.net/forum?id=1PL1NIMMrw
- 93 Shunyu Yao, Dian Yu, Jeffrey Zhao, et al. Tree of thoughts: Deliberate problem solving with large language models. In: Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/ hash/271db9922b8d1f4dd7aaef84ed5ac703-Abstract-Conference.html
- 94 Windsurf. Windsurf editor. URL https://windsurf.com/editor, accessed: 2025-05-10
- 95 Cursor. Cursor: The ai-powered code editor, 2023. URL https://www.cursor.com/, accessed: 2025-05-14
- 96 Microsoft. Copilot (gpt-4) [large language model], 2025. URL https://copilot.microsoft.com/, accessed: 2025-05-13
- 97 Samuel Miserendino, Michele Wang, Tejal Patwardhan, et al. Swe-lancer: Can frontier llms earn $1 million from real-world freelance software engineering?, 2025. URL https://arxiv.org/abs/2502.12115
- 98 Shunyu Yao, Noah Shinn, Pedram Razavi, et al. τ-bench: A benchmark for tool-agent-user interaction in real-world domains. CoRR, 2024, abs/2406.12045. URL https://doi.org/10.48550/arXiv.2406.12045
- 99 Yifei Zhou, Song Jiang, Yuandong Tian, et al. Sweet-rl: Training multi-turn llm agents on collaborative reasoning tasks,

2025. URL https://arxiv.org/abs/2503.15478

- 100 Jason Wei, Zhiqing Sun, Spencer Papay, et al. Browsecomp: A simple yet challenging benchmark for browsing agents, 2025. URL https://arxiv.org/abs/2504.12516
- 101 Giulio Starace, Oliver Jaffe, Dane Sherburn, et al. Paperbench: Evaluating ai’s ability to replicate ai research, 2025. URL https://arxiv.org/abs/2504.01848
- 102 David Rein, Joel Becker, Amy Deng, et al. Hcast: Human-calibrated autonomy software tasks, 2025. URL https://arxiv. org/abs/2503.17354
- 103 Hjalmar Wijk, Tao Lin, Joel Becker, et al. Re-bench: Evaluating frontier ai r&d capabilities of language model agents against human experts, 2024. URL https://arxiv.org/abs/2411.15114
- 104 Richard S Sutton, Andrew G Barto. Reinforcement Learning: An Introduction. Second edition. Cambridge, MA, USA: A Bradford Book, The MIT Press. 2018, 2018. URL http://incompleteideas.net/book/the-book-2nd.html
- 105 Ben Eysenbach, Aviral Kumar, Abhishek Gupta. Reinforcement learning is supervised learning on optimized data, 2020. URL https://bair.berkeley.edu/blog/2020/10/13/supervised-rl/, berkeley Artificial Intelligence Research Blog
- 106 Omar Khattab, Keshav Santhanam, Xiang Lisa Li, et al. Demonstrate-search-predict: Composing retrieval and language models for knowledge-intensive NLP. CoRR, 2022, abs/2212.14024. URL https://doi.org/10.48550/arXiv.2212.14024
- 107 Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, et al. Transformers learn in-context by gradient descent. In: Proceedings of the 40th International Conference on Machine Learning, PMLR. 2023, Proceedings of Machine Learning Research, volume 202. 35151–35174. URL https://proceedings.mlr.press/v202/von-oswald23a.html
- 108 Noah Shinn, Federico Cassano, Ashwin Gopinath, et al. Reflexion: language agents with verbal reinforcement learning. In: Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/ hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html
- 109 Theodore Sumers, Shunyu Yao, Karthik Narasimhan, et al. Cognitive architectures for language agents. Transactions on Machine Learning Research, 2024. URL https://openreview.net/forum?id=1i6ZCvflQJ, survey Certification

- 110 Wenwu Li, Xiangfeng Wang, Wenhao Li, et al. A survey of automatic prompt engineering: An optimization perspective,

2025. URL https://arxiv.org/abs/2502.11560

- 111 Chrisantha Fernando, Dylan Banarse, Henryk Michalewski, et al. Promptbreeder: Self-referential self-improvement via prompt evolution. In: Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, OpenReview.net. 2024. URL https://openreview.net/forum?id=9ZxnPZGmPU
- 112 Qingyan Guo, Rui Wang, Junliang Guo, et al. Connecting large language models with evolutionary algorithms yields powerful prompt optimizers. In: The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024, OpenReview.net. 2024. URL https://openreview.net/forum?id=ZG3RaNIsO8
- 113 Chengrun Yang, Xuezhi Wang, Yifeng Lu, et al. Large language models as optimizers. In: The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024, OpenReview.net. 2024. URL https://openreview.net/forum?id=Bb4VGOWELI
- 114 Tobias Schnabel, Jennifer Neville. Prompts as programs: A structure-aware approach to efficient compile-time prompt optimization. CoRR, 2024, abs/2404.02319. URL https://doi.org/10.48550/arXiv.2404.02319
- 115 Ruotian Ma, Xiaolei Wang, Xin Zhou, et al. Are large language models good prompt optimizers? CoRR, 2024, abs/2402.02101. URL https://doi.org/10.48550/arXiv.2402.02101
- 116 Weizhe Chen, Sven Koenig, Bistra Dilkina. Reprompt: Planning by automatic prompt engineering for large language models agents, 2024. URL https://arxiv.org/abs/2406.11132
- 117 Derek Austin, Elliott Chartock. Grad-sum: Leveraging gradient summarization for optimal prompt engineering, 2024. URL https://arxiv.org/abs/2407.12865
- 118 Pei Zhou, Jay Pujara, Xiang Ren, et al. Self-discover: Large language models self-compose reasoning structures, 2024. URL https://arxiv.org/abs/2402.03620
- 119 Yao Fu, Dong-Ki Kim, Jaekyeom Kim, et al. Autoguide: Automated generation and selection of state-aware guidelines for large language model agents. arXiv preprint arXiv:2403.08978, 2024
- 120 Minghao Chen, Yihang Li, Yanting Yang, et al. Automanual: Generating instruction manuals by llm agents via interactive environmental learning, 2024. URL https://arxiv.org/abs/2405.16247
- 121 Shenzhi Wang, Qisen Yang, Jiawei Gao, et al. Train once, get a family: State-adaptive balances for offline-to-online reinforcement learning. In: Advances in Neural Information Processing Systems, Curran Associates, Inc.. 2023, volume 36. 47081–47104. URL https://proceedings.neurips.cc/paper_files/paper/2023/file/ 9318763d049edf9a1f2779b2a59911d3-Paper-Conference.pdf
- 122 Dayuan Fu, Biqing Qi, Yihuai Gao, et al. Msi-agent: Incorporating multi-scale insight into embodied agents for superior planning and decision-making. arXiv preprint arXiv:2409.16686, 2024. URL https://api.semanticscholar.org/CorpusID: 272880637
- 123 Samira Pouyanfar, Saad Sadiq, Yilin Yan, et al. A survey on deep learning: Algorithms, techniques, and applications. ACM Computing Surveys, 2018, 51: 1–36
- 124 Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, et al. Alfworld: Aligning text and embodied environments for interactive learning. ArXiv, 2020, abs/2010.03768. URL https://api.semanticscholar.org/CorpusID:222208810
- 125 Jacob Andreas. Language models as agent models. In: Findings of the Association for Computational Linguistics: EMNLP 2022, Abu Dhabi, United Arab Emirates: Association for Computational Linguistics. 2022. 5769–5779. URL https: //aclanthology.org/2022.findings-emnlp.423/
- 126 Shaokun Zhang, Jieyu Zhang, Jiale Liu, et al. Offline training of language model agents with functions as learnable weights. In: Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024, OpenReview.net. 2024. URL https://openreview.net/forum?id=2xbkWiEuR1
- 127 Haiteng Zhao, Chang Ma, Guoyin Wang, et al. Empowering large language model agents through action learning. CoRR, 2024, abs/2402.15809. URL https://doi.org/10.48550/arXiv.2402.15809
- 128 Guanzhi Wang, Yuqi Xie, Yunfan Jiang, et al. Voyager: An open-ended embodied agent with large language models. Transactions on Machine Learning Research, 2024. URL https://openreview.net/forum?id=ehfRiF0R3a
- 129 Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, et al. Agent workflow memory, 2024. URL https://arxiv.org/abs/2409.07429
- 130 Andrew Zhao, Matthieu Lin, Yangguang Li, et al. A mixture of surprises for unsupervised reinforcement learning. In: Advances in Neural Information Processing Systems, Curran Associates, Inc.. 2022, volume 35. 26078–26090. URL https://proceedings.neurips.cc/paper_files/paper/2022/file/a7667ee5d545a43d2f0fda98863c260e-Paper-Conference.pdf
- 131 Benjamin Eysenbach, Abhishek Gupta, Julian Ibarz, et al. Diversity is all you need: Learning skills without a reward function. In: International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=SJx63jRqFm
- 132 Yitao Liu, Chenglei Si, Karthik R Narasimhan, et al. Contextual experience replay for continual learning of language agents,

2025. URL https://openreview.net/forum?id=RXvFK5dnpz

- 133 Yaniv Ovadia, Emily Fertig, Jie Ren, et al. Can you trust your model’s uncertainty? evaluating predictive uncertainty under dataset shift, 2019. URL https://arxiv.org/abs/1906.02530
- 134 Boyuan Zheng, Michael Y Fatemi, Xiaolong Jin, et al. Skillweaver: Web agents can self-improve by discovering and honing skills, 2025. URL https://arxiv.org/abs/2504.07079
- 135 Adam Paszke, Sam Gross, Francisco Massa, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, 2019, 32. URL https://papers.nips.cc/paper/2019/hash/ bdbca288fee7f92f2bfa9f7012727740-Abstract.html
- 136 Wangchunshu Zhou, Yixin Ou, Shengwei Ding, et al. Symbolic learning enables self-evolving agents. CoRR, 2024, abs/2406.18532. URL https://doi.org/10.48550/arXiv.2406.18532
- 137 Wenyi Wang, Hisham Abdullah Alyahya, Dylan R Ashley, et al. How to correctly do semantic backpropagation on languagebased agentic systems, 2025. URL https://openreview.net/forum?id=r1cbFEH0Df
- 138 Max van Duijn, Bram van Dijk, Tom Kouwenhoven, et al. Theory of mind in large language models: Examining performance of 11 state-of-the-art models vs. children aged 7–10 on advanced tests. In: Proceedings of the 27th Conference on Computational Natural Language Learning (CoNLL), Singapore: Association for Computational Linguistics. 2023. 389–402. URL https://aclanthology.org/2023.conll-1.25/
- 139 Eduardo Pignatelli, Johan Ferret, Matthieu Geist, et al. A survey of temporal credit assignment in deep reinforcement learning. Transactions on Machine Learning Research, 2024. URL https://openreview.net/forum?id=bNtr6SLgZf, survey Certification
- 140 Thomas Kwa, Ben West, Joel Becker, et al. Measuring ai ability to complete long tasks, 2025. URL https://arxiv.org/abs/ 2503.14499
- 141 Weiran Yao, Shelby Heinecke, Juan Carlos Niebles, et al. Retroformer: Retrospective large language agents with policy gradient optimization. In: The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024, OpenReview.net. 2024. URL https://openreview.net/forum?id=KOZu91CzbK
- 142 Bang Liu, Xinfeng Li, Jiayi Zhang, et al. Advances and challenges in foundation agents: From brain-inspired intelligence to evolutionary, collaborative, and safe systems, 2025. URL https://arxiv.org/abs/2504.01990
- 143 OpenAI. Openai memory announcement, 2025. URL https://x.com/OpenAI/status/1910378768172212636, accessed: 202505-13

- 144 Mirac Suzgun, Mert Yuksekgonul, Federico Bianchi, et al. Dynamic cheatsheet: Test-time learning with adaptive memory,

2025. URL https://arxiv.org/abs/2504.07952

- 145 Richard S Sutton. The bitter lesson, 2019. URL http://www.incompleteideas.net/IncIdeas/BitterLesson.html, accessed: 2025-05-12
- 146 Cheng-Kuang Wu, Zhi Rui Tam, Chieh-Yen Lin, et al. Streambench: Towards benchmarking continuous improvement of language agents. In: Advances in Neural Information Processing Systems, Curran Associates, Inc.. 2024, volume 37. 107039–107063. URL https://proceedings.neurips.cc/paper_files/paper/2024/file/ c189915371c4474fe9789be3728113fc-Paper-Datasets_and_Benchmarks_Track.pdf
- 147 Jiaxuan You, Mingjie Liu, Shrimai Prabhumoye, et al. Llm-evolve: Evaluation for llm’s evolving capability on benchmarks. In: Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2024. 16937–16942. URL https://aclanthology.org/2024.emnlp-main.940
- 148 Kevin Lin, Charlie Snell, Yu Wang, et al. Sleep-time compute: Beyond inference scaling at test-time, 2025. URL https: //arxiv.org/abs/2504.13171
- 149 Evgenii Nikishin, Max Schwarzer, Pierluca D’Oro, et al. The primacy bias in deep reinforcement learning, 2022. URL https://arxiv.org/abs/2205.07802
- 150 Alessandro Achille, Matteo Rovere, Stefano Soatto. Critical learning periods in deep neural networks, 2019. URL https: //arxiv.org/abs/1711.08856
- 151 Jonathan Frankle, David J Schwab, Ari S Morcos. The early phase of neural network training, 2020. URL https://arxiv. org/abs/2002.10365
- 152 Mohamed Elsayed, Gautham Vasan, A Rupam Mahmood. Streaming deep reinforcement learning finally works, 2024. URL https://arxiv.org/abs/2410.14606
- 153 Mahdi Karami, Vahab Mirrokni. Lattice: Learning to efficiently compress the memory, 2025. URL https://arxiv.org/abs/ 2504.05646
- 154 Seongyun Lee, Sue Hyun Park, Seungone Kim, et al. Aligning to thousands of preferences via system message generalization. In: The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/ forum?id=recsheQ7e8
- 155 Shangyin Tan, Lakshya A Agrawal, Arnav Singhvi, et al. Langprobe: a language programs benchmark, 2025. URL https://arxiv.org/abs/2502.20315
- 156 Yunfan Jiang, Agrim Gupta, Zichen Zhang, et al. Vima: General robot manipulation with multimodal prompts. In: Fortieth International Conference on Machine Learning, 2023

