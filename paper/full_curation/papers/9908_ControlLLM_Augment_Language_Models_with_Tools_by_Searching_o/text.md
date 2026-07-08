### ControlLLM: Augment Language Models with Tools by Searching on Graphs

# arXiv:2310.17796v3[cs.CV]18Dec2023

Zhaoyang Liu∗1,2 Zeqiang Lai∗2 Zhangwei Gao2 Erfei Cui2 Ziheng Li3 Xizhou Zhu2,3 Lewei Lu4 Qifeng Chen1 Yu Qiao2 Jifeng Dai2,3 Wenhai Wang2

1The Hong Kong University of Science and Technology 2OpenGVLab, Shanghai AI Laboratory 3Tsinghua University 4SenseTime

https://github.com/OpenGVLab/ControlLLM

#### Abstract

We present ControlLLM, a novel framework that enables large language models (LLMs) to utilize multi-modal tools for solving complex real-world tasks. Despite the remarkable performance of LLMs, they still struggle with tool invocation due to ambiguous user prompts, inaccurate tool selection and parameterization, and inefficient tool scheduling. To overcome these challenges, our framework comprises three key components: (1) a task decomposer that breaks down a complex task into clear subtasks with welldefined inputs and outputs; (2) a Thoughts-on-Graph (ToG) paradigm that searches the optimal solution path on a prebuilt tool graph, which specifies the parameter and dependency relations among different tools; and (3) an execution engine with a rich toolbox that interprets the solution path and runs the tools efficiently on different computational devices. We evaluate our framework on diverse tasks involving image, audio, and video processing, demonstrating its superior accuracy, efficiency, and versatility compared to existing methods.

#### 1. Introduction

Large-scale language models, such as ChatGPT [24] and LLaMA series [36, 37], have demonstrated impressive capability in understanding and generating natural language. Beyond their prowess in linguistic tasks, these models have been rapidly extended to interaction, planning, and reasoning, propelling the advancement of studies in multi-modal interaction [1, 16, 17, 21, 38, 39, 51].

One of the emerging examples of multi-modal interaction is tool-augmented language models [20, 31, 32, 43, 45], which strive to enhance the capabilities of language models to include diverse modalities beyond text such as image,

* Equal contribution. Corresponding authors (wangwenhai@pjlab.org.cn, cqf@ust.hk).

video, audio, etc. These models employ LLMs as primary controllers and incorporate tools with diverse functionalities as plugins, which solves a wide range of multi-modal tasks. However, challenges in this field still persist, covering task decomposition, task planning, and efficient tool scheduling.

With these challenges in mind, prior methods [20, 29, 32, 34, 43, 45, 46] made their endeavors in developing toolaugmented LLMs. They utilize LLMs with input-output prompting, CoT [41] or ToT [47] to perform task planning. These methods can solve problems by breaking them into a chain or tree of sub-tasks. Theoretically, as long as LLMs have strong generalization ability, these methods can also solve complex tasks. However, in practice, we found that these methods often suffer from inaccurate tool invocation problems when dealing with complex cases. This is due to the fact that solutions for complex tasks often contain tool invocations with intricate topological structures. It is insufficient for these methods to form a complex thought network and thus fail to solve complicated tasks. Therefore, it requires us to figure out a new paradigm beyond chainshaped or tree-shaped ones, which can generate solutions with intricate topology structures to solve more complicated problems (see Fig. 1 and Fig. 2).

To this end, we introduce ControlLLM, a new framework that assists large language models in accurately and efficiently controlling multi-modal tools and identifying comprehensive solutions for complex real-world tasks involving multi-modal inputs. Alongside a variety of improvements over previous works, our framework places particular emphasis on three aspects as follows:

Task Decomposition. A task decomposer is introduced to analyze the user prompt and breaks it down into a number of subtasks, each with well-defined attributes such as task description, task domain, arguments, and returned output. By decomposing complex tasks into manageable subtasks, the task decomposer significantly enhances the system’s ability to handle intricate user prompts, which paves the way for follow-up task planning and solution execution.

Input

Input

Input

Input

[Figure 1]

Output

Output

Output

Output

- 1. 3. +! , ≤ 1 & +" , ≤ 1 1.3. +! , ≤ 1 No +! , or +" , Constraints

- (a) Chain of Thought (CoT) (b) Tree of Thoughts (ToT) (c) ControlLLM (ours)

|Tool (Vertex, ,) Tool Dependency (Edge, /) Solution Path +! 0 In-Degree +" 0 Out-Degree|
|---|

Thoughtson Graph

self−consistency

- Figure 1. Comparisons of different paradigms for task planning. (a) Chain of Thought (CoT) [41], CoT with self-consistency [40] and

- (b) Tree of Thoughts [47] (ToT) essentially rely on the LLMs to perform task planning, where the edge is actually formed by LLMs at run time. (c) The Thoughts-on-Graph (ToG) paradigm in our method searches for solutions on a pre-built graph that captures the dependencies of tools, which avoids the hallucination problem in tool invocation.

Task Planning. This part handles tool selection and tool argument assignment. We propose a thoughts-on-graph (ToG) paradigm that traverses a topological tool graph to search for solutions. The nodes of the graph are tools that are connected based on their dependencies and relationships. ToG orchestrates the selected tools and controls the flow of resources among them to form possible solutions. ToG can find the optimal solution for each sub-task by applying diverse search strategies on the graph. Due to the concrete definition in subtask and explicit tool dependencies in a tool graph, ToG can effectively search all feasible solution paths in cases where the selected optimal solution fails to meet users’ preferences.

when tools change, we only need to rebuild the graph without re-training LLMs or updating in-context prompts.

To evaluate the effectiveness of ControlLLM in tasks of different complexities, we construct a benchmark with a series of tailored metrics. Specifically, we use irrelevant tool inclusion rate and necessary tool inclusion rate to measure tool selection. We employ the resource hallucination rate and resource type consistency rate to assess argument assignments. We also split the test set into three difficulty levels based on the number of APIs involved: easy (< 2 APIs), medium (2 or 3 APIs), and hard (> 3 APIs). We conducted various experiments, both quantitatively and qualitatively, to compare our method with existing ones. The results show that ControlLLM achieves a higher success rate in tool invocation, especially for complicated instructions.

Solution Execution. We design an execution engine that can execute the solution generated by ToG and craft informative and well-formatted responses. The engine has access to a versatile toolbox consisting of various tools from different sources, such as locally deployed APIs or cloud services. The engine can also parallelize the tool executions according to the topology of the solution path to reduce the latency and provide feedback during the execution process.

In summary, the main contributions are as follows:

- (1) We propose ControlLLM, a framework that lets

LLMs use various tools across different modalities to solve complex tasks in the real world. With a powerful toolbox, ControlLLM can be easily extended to tasks with natural language, images, audio, video, or any mix of them.

- (2) We design three tailored components in Control-

LLM: Task decomposition, which breaks down the user prompt into subtasks with well-defined inputs and outputs; ToG paradigm for task planning, searching the optimal solution path on a graph that depicts tool dependencies; And an execution engine with a powerful toolbox, which efficiently schedules and executes the solution path.

- (3) We construct a benchmark to assess the efficacy of

Our ControlLLM offers several advantages. (1) It can accurately handle complex real-world tasks that involve multimodal inputs and outputs, while previous methods [4, 19, 20, 32, 43, 45] usually fail to handle due to their capabilities of task planning; (2) It can overcome the token limitation of LLMs during task planning. Because our method searches the optimal solution path on the tool graph, instead of asking LLMs to generate a solution for the task; (3) It can easily scale up toolbox. Since all solutions lie in the tool graph,

ControlLLM on tasks with different complexity levels. The

experimental results demonstrate significant improvements in tool usage. Notably, ControlLLM achieves a success rate of 93% in the metric of overall solution evaluation on challenging tasks, while the best baseline only reaches 59%.

#### 2. Related Work

Planning, Reasoning, and Decision Making. It is a longstanding vision to empower autonomous agents with the abilities of planning, reasoning, and decision-making [15, 33, 42]. Despite progressive development, it was recent advancements in large language models (LLM) [3, 5, 25, 36, 50] that have taken a breakthrough step in addressing these problems on the broad user requests. Nevertheless, it is shown that LLMs still suffer from difficulties in dealing with knowledge-heavy and complex tasks [30]. To overcome these issues, Chain of Thoughts (CoT) [41] is introduced as a simple Tool Documentation Enables ZeroShot Tool-Usage with Large Language Modelsprompting technique to elite the complex reasoning capabilities of LLMs. Following this line of work, CoT with self consistency [40], Tree of Thoughts (ToT) [41], and other techniques [6, 12, 55], have been proposed to improve the reasoning abilities further. There are also several works [2, 48] that introduce techniques called Graph-of-Thought (GoT). They all share a common insight that relies on LLMs to generate thoughts for solving complicated NLP problems. In contrast, our ToG aims to endow the language model with the ability to use tools for a multi-modal dialogue system. Furthermore, ToG builds a tool-graph in advance without requiring LLMs and uses a search algorithm to form a complicated thought network for task planning.

Tool-Augmented LLM. Drawing inspiration from the evolving planning and decision-making capabilities observed in Large Language Model (LLM) systems, a new wave of research starts to enhance LLMs with external tools for accessing up-to-date information, reducing hallucination, multi-modal interactions, etc. Prominent examples include ReAct [46], VisProg [10], Visual ChatGPT [43], HuggingGPT [32], InternGPT [20], AutoGPT1, and Transformers Agent2. A distinctive trait of this line of research is its reliance on the zero-shot or few-shot in-context learning [8] capabilities inherent in LLMs [3]. These capabilities enable task decomposition, tool selection, and parameter completion without requiring explicit finetuning. However, due to the inherent limitations of LLMs, issues such as hallucination and challenges in effective decomposition and deduction can arise with substantial frequency. Furthermore, there are also instructiontuning methods [11, 26, 27, 29, 31, 45]. Whereas allevi-

- 1https://github.com/Significant-Gravitas/Auto-

GPT

- 2https : / / huggingface . co / docs / transformers /

transformers_agents

ating the above issues after being tuned on the text corpus involved tools, these methods are still limited at expanding the toolset, i.e., additional training is required to add tools. Among these methods, ToolLLM [29] proposes the depth first search-based decision tree to boost the planning ability of LLMs. However, it still has limitations similar to ToT, as shown in the Fig. 1.

Multi-Modal LLMs. Developing LLMs that inherently possess multi-modal capabilities is another approach to extending the usage boundary of LLMs for more complex real-world scenarios [7, 13, 18, 19, 22, 23, 28, 44]. For instance, BLIP-2 [17], LLava [18], and Mini-GPT4 [54] bind frozen image encoders and LLMs to enable the visionlanguage understanding and generation. Similarly, VisionLLM [39] and LISA [14] empower the LLMs with the visual perception capabilities such as object detection and segmentation. GILL [13], DreamLLM [9], and MiniGPT5 [53] extend LLM for interleaved image and text generation by jointly optimizing the LLM with off-theshelf Stable Diffusion model. Kosmos2 [28], Ferret [49], GPT4RoI [52], and etc., design various region-aware image encoders to augment LLMs with the abilities of grounding and referring. Nevertheless, these methods could only cover a limited range of modalities or tasks and often require huge effects on model finetuning.

#### 3. ControlLLM

The prevalence of LLMs has unprecedentedly boosted the development of human-computer interaction. It is feasible to empower the LLMs with abilities to interact with various modalities via tools. In response, we present an innovative framework, namely ControlLLM, characterized by its flexibility, and high performance. As depicted in Fig. 2, our framework consists of three sequential stages, i.e., task decomposition, task planning and solution execution. Next, we will illustrate the design of each stage in detail.

###### 3.1. Task Decomposition

ControlLLM starts with task decomposition – a stage for decomposing the user request r into a list of parallel subtasks. We here can utilize a language model M, e.g., ChatGPT or instruction-tuned LLaMA, to automatically decompose the user request as follows:

{s0,...,si,...,sn} = M(r), (1)

where si is the i-th subtask, n is the number of all subtasks. We will elaborate on the different choices of language model M in Sec. 3.4 and discuss their impacts in Sec. 4.5. The result of task decomposition is JSON format, and the output protocol is presented in Table 5.

Task decomposition is different from task planning. It only breaks down the user’s request into several parallel

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

## !

[Figure 6]

[Figure 7]

Stage1:TaskDecomposi2on!

###### Stage2:Thoughts-on-Graph(ToG)#

Can you dub the video.mp4 with a suitable background music and then generate a HTML web page for this video?

|SubTask 1<br><br>Descrip0on: Dub the video with a background music.<br><br>Domains: video-percepEon, audio-generaEon, videoediEng<br><br>Args: <video> Return: <video>|
|---|

|SubTask 2<br><br>Descrip0on: Generate a HTML page for this video.<br><br>Domains: video-percepEon, video-processing<br><br>Args: <video> Return: <HTML>|
|---|

input video.mp4

|[Figure 8]<br><br>[Figure 9]|
|---|

Input

|videocap+oning| |
|---|---|
| | |

|text-tomusic| |
|---|---|
| | |

|VideoCap+oning| |
|---|---|
| | |

|videodubbing| |
|---|---|
| | |

[Figure 10]

[Figure 11]

## "

Stage 3: Solu2on Execu2on

I have designed the HTML web page to introduce the video with light background music:

⚙

[Figure 12]

|Text-ToMusic| |
|---|---|
| | |

|videocap+oning|
|---|

|videotagging| |
|---|---|
| | |

|video-+tling|
|---|

[Figure 13]

ExecuEon Engine w/ Toolbox

Summarize

|[Figure 14]<br><br>[Figure 15]<br><br>[Figure 16]|
|---|

[Figure 17]

[Figure 18]

|video-towebpage| |
|---|---|
| | |

[Figure 19]

[Figure 20]

[Figure 21]

|VideoDubbing| |
|---|---|
| | |

output <HTML>

[Figure 22]

Op0mal Solu0on

|video-capEoning|
|---|

|video-tagging|
|---|

Searching op0mal solu0on on the tool graph

|text-to-music|
|---|

|video-Etling|
|---|

|VideoCap+oning|
|---|

|VideoTagging| |
|---|---|
| | |

|VideoTitling|
|---|

|input / output tools<br><br>| |
|---|
<br><br>soluEon path|
|---|

|video-dubbing|
|---|

...

|GenerateHTML| |
|---|---|
| | |

- Figure 2. System design of ControlLLM. The framework consists of three stages. The first stage is task decomposition, which parses the user input into several subtasks. Then, in Stage 2, ToG utilizes a depth-first search algorithm to find the optimal solution for each subtask. The execution engine in the last stage executes the solution and returns the output to users. We here use the example of generating a web page for the video to illustrate our method.

subtasks and summarizes the input resources for each subtask from the user request. It does not need to know what tools to use or how to use them. The objective of this stage is to achieve three aims. Firstly, it splits user requests into smaller and more manageable units, i.e., subtasks, thereby accelerating task planning. Secondly, it seeks to determine the task domain that is most relevant and appropriate for the given problem, thus further narrowing down the scope of task planning. Thirdly, it endeavors to infer the input and output resource types from the context, which identifies the start and end nodes for ToG to search in the next stage.

- 3.2. Task Planning with Thoughts-on-Graph

Output

a comprehensive tool graph that encapsulates the inherent relationship between tools.

There are two types of nodes i.e., Resource node and Tool node, in the graph. Resource node can be formally defined as one-tuple: ⟨type⟩, where type represents the specific type of resource, like image, mask, video, etc. Tool node can be expressed as a three-tuple: ⟨desc, args, ret⟩. The desc field encapsulates the description of the tool, elucidating its purpose, methodology, and intended applications. The args field represents a list of resource nodes that the tool accepts, thereby giving the prerequisites for utilizing this tool. Finally, the ret field designates the resource node that the tool returns. We elaborate on the definitions of resource types and tools in supplementary material (Sec. 10).

This stage is the key of the entire system. Given the results of task decomposition, we design a Thoughts-on-Graph (ToG) paradigm to find solutions on the graph heuristically.

Edge Definitions. Edges in the tool graph intricately connect the nodes, highlighting the relationships between different tools. We define two types of edges in the graph.

###### 3.2.1 Building the Tool Graph

(1) Tool-resource edge is established from the tool to its returned resource type. This signifies that the tool is capable of generating resources of the corresponding type. Mathematically, a tool-resource edge is represented as:

In this stage, we embark on constructing a Tool Graph G by simply using an adjacency matrix, which serves as a fundamental guideline for analyzing and optimizing the interactions between tools. Our motivation is driven by observing a discernible topological structure that inherently exists between the input and output of diverse tools, as demonstrated in Fig. 2. This compelling insight propels us to craft

true , if Ri equals to ret of Tj false, otherwise

, (2)

G(Tj,Ri) =

Algorithm 1 The Python pseudocode of depth-first solution search in Thoughts-on-Graph Input:

t: subtask obtained by Eq. 1 g: tool graph G constructed in Sec. 3.2.1

- r: available resources, initialized with subtask[“args”]
- s: recorded tools during searching

###### Output:

solutions: all possible solutions for the subtask t

- 1: function DFS SEARCH(t, g, r, s)

- 2: if len(s) > m:
- 3: return [] # F finds all tool candidates, explained in Sec. 3.2.2
- 4: available tools = F(t, g, r)

- 5: solutions = []
- 6: for tool in available tools:

- 7: s.append(tool)
- 8: r.append(tool[“returns”])
- 9: if tool[“returns”] == t[“returns”]:
- 10: solutions.append(s.copy())
- 11: results = DFS SEARCH(t, g, r, s)

- 12: solutions.extend(results)
- 13: r.remove(tool[“returns”])
- 14: s.remove(tool)
- 15: return solutions ▷ Return
- 16: end function

where Tj is j-th tool node, Ri is i-th resource node, “true” denotes two nodes are connected, and “false” denotes two nodes are disconnected.

(2) Resource-tool edge denotes the resource node that can be accepted as input arguments for its adjacent tool. This connection indicates how the resources flow to the tool. The resource-tool edge is mathematically defined as:

G(Ri,Tj) =

true , if Ri belongs to args of Tj false, otherwise

. (3)

Through the establishment of this graph, we can use diverse search strategies to make informed decisions regarding tool selection, and input resource assignments.

###### 3.2.2 Searching on the Graph

As described in Algorithm 1, our ToG is built upon a depthfirst search (DFS) algorithm where the tool selection function F is used to sample the tool nodes on the tool graph. The algorithm starts from the input resource nodes and explores all possible paths to the output node while keeping track of the intermediate resources and tools along the way. The algorithm stops when it reaches the expected output node or when it exceeds a maximum length limit m (m=10

by default). Finally, the algorithm returns all searched solutions as a list of tool sequences. Each step from resource node to tool node represents a thought process, as it involves a decision that determines whether to use this tool and how to assign its input arguments from available resources.

To find a trade-off between time and space complexities, we develop a tool assessment module in which the language model is leveraged to score the tools in each search step and then filter out some irrelevant tools. For details, please refer to Sec. 6.2 in supplementary material. With this assessment module, we design four search strategies for the function F to determine which tool nodes within the task domains to visit among all adjacent nodes when searching on the graph:

Greedy Strategy. This strategy selects the tool node with the highest score at each step, where the score indicates the relevance of the tool to the task. A higher score indicates that the tool is more helpful for solving the task. Greedy search is fast and simple, but it may not find the optimal solution or even any solution at all.

Beam Strategy. It only keeps the k best tools according to their assessment scores. Beam search can expand the search space but reduce the search efficiency slightly.

Adaptive Strategy. This is a variant of beam search where it dynamically adjusts the beam size by choosing the tools with scores higher than a fixed threshold, which is a trade-off between exploration and exploitation. It can widen the search space when there are many available choices and narrow down when there are few confident choices.

Exhaustive Strategy. This strategy explores all possible paths from the start node to the terminal node. The exhaustive search is guaranteed to find an optimal solution if one exists, but it may be very slow and consume a lot of computational resources during the search.

The impacts of different search strategies are studied in Sec. 4.5. By initiating a systematic traversal of tool graph, commencing at the “args” nodes and culminating at the “return” node, a diverse list of conceivable solutions is meticulously synthesized. This process, akin to a brainstorm or mind map, represents the spectrum of potential solutions.

###### 3.2.3 Solutions Post-processing

After ToG searches the solutions, we design solution expert and resource expert to post-process solutions, which both build upon the language model M. Specifically, solution expert to select the optimal one among all solution candidates and resource expert to infer the remaining arguments for tools, respectively. The overall details are shown in supplementary material (Sec. 6.3 and 6.4).

###### 3.3. Solution Execution

Once the task solutions are completed, they are passed to the execution engine obtain results, as shown in Fig. 2. During

this stage, the execution engine initially parses the solutions into a sequence of Actions. Each action is associated with particular tool services, which could be implemented via either handcrafted mapping tables or an automatic scheduler based on some strategies. Different from previous works [20, 43, 45] that adopt static tool mapping, our design empowers the system with the flexibility to schedule diverse tools based on users’ preferences.

The parsed actions are automatically executed by scheduling the action to the local, remote, or hybrid endpoints. Multiple independent subtasks would be executed in parallel to improve efficiency. Besides, we maintain a state memory storing all the intermediate results, including their values and types. This enables the running-time automatic correction for the action parameters.

Response Generation. With all the execution results in hand, we could respond to the user requests. The unprocessed results may lack comprehensiveness and clarity, potentially making it difficult for users to understand. To this end, we introduce a module to aggregate all the execution results and generate user-friendly responses. This is achieved by prompting the LLMs, such as ChatGPT, with the user request, action list, and execution results and asking them to summarize the answers intelligently. The prompt can be found in supplementary material (Table 10).

###### 3.4. The Choices of Language Model

One feasible yet direct choice is to use off-the-shelf large language models (LLMs) such as ChatGPT or Llama 2 [37], which are pre-trained on large-scale text corpora and can handle various NLP tasks. These LLMs are readily available. We design a series of elaborate prompts as shown in Sec. 6 for task decomposition, tool assessment, solution expert, and resource expert. We call this variant as ControlLLM-ChatGPT. In this way, we avoid training a language model from scratch. However, they may lead to low performance as they are not trained for our requirements.

The alternative choice of M, termed as ControlLLMLLaMA, is to finetune a language model (e.g., LLaMA [36]) by using self-instruct method [40]. More details of optimizing M can be referred to Sec. 7 in supplementary material. The advantage of this variant is that it can achieve high performance by adapting to the data and the task. Nevertheless, it requires lots of GPUs to train the model and may suffer from overfitting issues.

Regarding these choices, it is essential to carefully consider the trade-offs between readily available off-the-shelf LLMs with zero-shot capabilities and the potential for finetuning a model to achieve superior performance at the cost of computational resources. We will thus further discuss the impacts of different language models M in Sec. 4.4 and explore the optimal settings for our framework.

#### 4. Experiments

###### 4.1. Benchmark

We build a benchmark that is used to evaluate our proposed framework compared with other state-of-the-art methods. In order to make fair comparisons, we only evaluate and test on the intersection of toolsets from different methods [20, 32, 43, 45], all of which share comparable toolsets. The benchmark consists of a set of tasks that require various tools to solve complex problems collaboratively. It is designed to cover different task domains, such as question answering, image generation, image editing, image perception, visual question answering, etc. In this benchmark, the tasks involve more than 20 tools across different modalities.

This benchmark includes about 100 instructions which are classified into three levels of difficulty: easy (< 2 APIs), medium (2 or 3 APIs), and hard(> 3APIs). We use test instructions with various levels to meticulously validate the ability of different methods. We believe that this benchmark can provide a comprehensive comparison of the tool control capabilities of different methods. In Table 12, we showcase some instruction samples from our benchmark. It is noticeable that there is no absolute relationship between difficulty and length of instruction.

###### 4.2. Evaluation Protocol

Effectively evaluating the performance of tool-augmented LLMs remains a challenging task. The challenges stems from several factors, including the inherent ambiguities in natural language, the absence of shared benchmarks, and formatted solutions for systematically assessing different methods. Consequently, existing methods [20, 32, 43, 45] provide extensive case studies to validate the performance.

We found the APIs of tools in different methods are slightly inconsistent. It is hard to annotate all feasible solutions for each method. As such, we adopt an evaluation protocol via a multi-person voting approach with three annotation experts. The protocol breaks down the evaluation into three main aspects: tool selection, argument assignment, and overall solution evaluation. Please note that the evaluation protocol is independent of the tools’ capabilities. When the tools and their input arguments are correct, we do not account for the case where the output fails to satisfy the user’s expectations due to the limitations of tools.

Metrics for Tool Selection: A) Irrelevant Tool Inclusion Rate (abbr. IR): This metric gauges the performance of the method in excluding irrelevant tools. It measures the proportion of the predicted solutions that contain the irrelevant tools. A higher IR indicates that the method tends to include more unnecessary tools, potentially hindering effective task planning; B) Necessary Tool Inclusion Rate (abbr. NR): This metric assesses the inclusion of necessary tools in the predicted solution but without considering whether

ControlLLM (our work)

HuggingGPT [32]

Visual ChatGPT [43]

InternGPT [20]

GPT4Tools [45]

Features

Image Perception ✓ ✓ ✓ ✓ ✓ Image Editing ✓ ✓ ✓ ✓ ✓ Image Generation ✓ ✓ ✓ ✓ ✓ Video Perception ✓ ✓ ✗ ✓ ✗ Video Editing ✓ ✓ ✗ ✓ ✗ Video Generation ✓ ✓ ✗ ✓ ✗ Audio Perception ✓ ✓ ✗ ✗ ✗ Audio Generation ✓ ✓ ✗ ✗ ✗ Multi-Solution ✓ ✗ ✗ ✗ ✗ Pointing Device ✓ ✗ ✗ ✓ ✗ Resource Type Awareness ✓ ✗ ✗ ✗ ✗

Table 1. Comparisons of features between different methods. The table shows that our framework supports more features that facilitate the user experience of multi-modal interaction. It proves the high scalability of our framework.

the arguments of tools are correct. If NR is high, it indicates the method has strong capabilities in tool selection.

Metrics for Argument Assignment: A) Resource Hallucination Rate (abbr. HR): This indicator reveals the extent of hallucination in the method’s responses when inferring the arguments for tools. It measures whether all arguments of the tools used in the predicted solution exist physically. A lower HR suggests that the method is less prone to generating hallucinated content. B) Resource Type Consistency Rate (abbr. CR): This metric examines whether the types of input resources in the predicted solution match those of the corresponding tools. It evaluates the method’s ability to ensure consistency of input types of tools.

Solution Evaluation (abbr. SE) measures the success rate of all generated solutions on our benchmark. It only considers whether the output solution can effectively address the user’s problem, irrespective of whether it contains irrelevant tools. A higher score in the solution evaluation indicates a stronger capability of task planning.

In summary, these intuitive metrics together provide a comprehensive assessment of tool-augmented LLMs. The formal definitions of these metrics can refer to Sec. 11.

###### 4.3. Feature Comparisons

- Table 1 presents a comprehensive feature comparison among various methods [20, 32, 43, 45], highlighting ControlLLM’s distinct advantages in the landscape of multimodal interaction. Notably, “Multi-Solution” signifies the method’s ability to provide multiple feasible solutions, granting users more options. “Pointing Device” signifies support for pointing devices such as the mouse, to enhance user experience. “Resource Type Awareness” indicates the method’s capability to discern the type of resource in the context, ensuring more context-aware responses. In summary, ControlLLM emerges as the standout choice, ex-

celling in various features. It offers a comprehensive set of tools in the domains of image, video, and audio. Moreover, its support for resource type awareness, multiple solutions, and pointing inputs demonstrates its adaptability and scalability, making it the highly versatile framework for diverse multi-modal interaction scenarios.

###### 4.4. Quantitative Comparisons

In this section, we give a comprehensive analysis of ControlLLM to compare with state-of-the-art methods, as summarized in Table 2. We provide three implementations in supplementary materials for our method: a) ControlLLMChatGPT leverages the ChatGPT-3.5 as language model M; b) ControlLLM-LLaMA that finetunes a LLaMA-7B as a language model M; c) ControlLLM-Mix is regarded as our default setting, which finetunes LLaMA-7B as a task decomposer in the first stage while the remaining modules employ the ChatGPT to finish the tasks. ControlLLM-Mix combines the advantages of the other two variants and is abbreviated as ControlLLM in the following sections.

Our evaluation is based on a set of metrics assessing effectiveness of task planning. ControlLLM excels in several key aspects. Notably, it achieves the lowest Irrelevant Tool Inclusion Rate (IR) as well as the highest Necessary Tool Inclusion Rate, indicating its ability in effective yet efficient task planning. Furthermore, ControlLLM demonstrates superior performance in argument assignment, with the lowest Argument Hallucination Rate (HR) of 0.02 and the highest Argument Type Consistency Rate (CR) of 0.98. These results underscore its ability to generate accurate and consistent arguments, addressing a challenge in tool-augmented LLMs. In the solution evaluation, ControlLLM maintains its lead with a score of 0.93, indicating its effectiveness in resolving user requests. In summary, ControlLLM exhibits remarkable performance in all proposed metrics, evidently

|Methods|Tool IR ↓ NR ↑<br><br>|Argument HR ↓ CR ↑|Solution Evaluation ↑ All Easy Medium Hard|
|---|---|---|---|
|HuggingGPT [32] Visual ChatGPT [43] InternGPT [20] GPT4Tools [45]|0.45 0.64 0.26 0.58 0.12 0.51 0.19 0.44|0.16 0.69 0.09 0.76 0.49 0.43 0.28 0.72|0.59 0.73 0.50 0.33 0.57 0.73 0.63 0.10 0.44 0.60 0.46 0.00 0.43 0.64 0.33 0.00<br><br>|
|ControlLLM-ChatGPT ControlLLM-LLaMA ControlLLM-Mix∗|0.16 0.63 0.06 0.95 0.03 0.93|0.83 0.83 0.02 0.98 0.02 0.98|0.64 0.71 0.67 0.43 0.91 0.98 0.88 0.76 0.93 0.98 0.96 0.81|

- Table 2. Comparisons with the state-of-the-art methods. ↓ means the smaller the better, ↑ means the larger the better. The results of state-of-the-art methods [20, 32, 43, 45] are reproduced on our own benchmark. ∗ denotes the default setting of ControlLLM if not stated.

|Search Strategies|Tool IR ↓ NR ↑<br><br>|Argument HR ↓ CR ↑|Solution Evaluation ↑ All Easy Meduim Hard|Time Complexities|
|---|---|---|---|---|
|Greedy Beam (k = 3) Adaptive Exhaustive|0.19 0.49 0.14 0.88 0.03 0.93 0.06 0.97|0.24 0.76<br><br>0.01 0.99<br>0.02 0.98 0.01 0.99<br>|0.49 0.56 0.58 0.19 0.88 0.96 0.79 0.76 0.93 0.98 0.96 0.81 0.97 1.00 0.96 0.91|4.07 121.29 236.49 3444.23|

- Table 3. The evaluation for different search strategies. As introduced in Sec. 3.2.2, although exhaustive strategy achieves the best performance on most metrics, the adaptive strategy strikes a good balance between efficiency and effectiveness. We count the average number of visited tools to denote the time complexities for different search strategies.

|Task Decomp.|LLMs|Tool IR ↓ NR ↑<br><br>|Argument HR ↓ CR ↑|Solution Evaluation ↑ All Easy Meduim Hard|
|---|---|---|---|---|
|w/o PK|Llama2-13B ChatGPT-3.5 GPT-4<br><br>|0.28 0.71 0.13 0.84 0.06 0.91|0.01 0.99 0.01 0.99 0.03 0.97|0.68 0.87 0.50 0.38 0.83 0.99 0.67 0.57 0.91 0.98 0.83 0.81|
|w/ PK|Llama2-13B ChatGPT-3.5 GPT-4|0.12 0.83 0.03 0.93 0.01 0.98|0.04 0.95 0.02 0.98 0.02 0.98|0.82 0.95 0.71 0.62 0.93 0.98 0.96 0.81 0.98 1.00 1.00 0.91|

- Table 4. The effects of task decomposition with regard to different LLMs. PK denotes “prior knowledge”. We find, if adding prior knowledge, such as which tools might be used, into the subtask description, the performance of task planning can be evidently improved.

outperforming the state-of-the-art methods in this field.

###### 4.5. Ablation Studies

Table 3 investigates the impact of different search strategies within our Thoughts-on-Graph. We observe that the exhaustive search strategy outperforms others in most metrics, but this strategy is time-consuming. On the other hand, the greedy search strategy achieves the lowest performance. Because it can not search for a feasible path based on the tool with a high score due to inaccurate tool assessment. It thus usually fails to find the solution, especially in complicated cases. In addition, the adaptive strategy strikes a

balance between performance metrics and time complexities, offering competitive results in most aspects. To tradeoff between time and accuracy, we thus choose the adaptive strategy as our default setting.

In Table 4, we conduct ablation studies to evaluate the impact of different LLMs on task planning for ControlLLM-Mix. We find language models plays a decisive role in tool selection. The more powerful the language model, the higher the score of solution evaluation. Furthermore, we investigate the effects of incorporating prior knowledge into the subtask descriptions. The method without prior knowledge usually directly uses the user’s request

Case 1: Generate a new image similar to img_1.png.

ControlLLM HuggingGPT

input: img_1.png

input: img_1.png

| | | | |
|---|---|---|---|
| | | | |

|image-to-image|
|---|

|segment-anything| |
|---|---|
| | |

|image-to-edge|
|---|

|image-to-image|
|---|

image- captioning

|seg-text-to-image|
|---|

|text-to- image| |
|---|---|
| | |

|edge-text-to-image|
|---|

output: <image> output: <image>

output: <image> output: <image> output: <image>

Case 2: Are there dogs in the img_2.png?

ControlLLM HuggingGPT

input: img_2.png

input: img_2.png

| | |
|---|---|
| | |

| | | |
|---|---|---|
| | | |

|object-detection| |
|---|---|
| | |

|VQA| |
|---|---|
| | |

|VQA|
|---|

|image-classification| |
|---|---|
| | |

|object-detection| |
|---|---|
| | |

|select-category| |
|---|---|
| | |

select-bbox

output: <text> output: <text> output: <image>

|image-cropping| |
|---|---|
| | |

|highlight-object|
|---|

output: <category>

output: <image>

output: <image>

Figure 3. Qualitative comparisons of task planning. We here use two simple cases to illustrate the differences between two different methods in task planning. Here, each output node is generated by different solution paths.

as a subtask description and does not offer any hints or suggestions on tool selections in the subtask description. In contrast, in the variant with prior knowledge, we add prior knowledge into the subtask description. The prior knowledge indeed improves the necessary tool inclusion rate (NR) and reduces the chance of selecting irrelevant tools (IR) when using the same large language model.

across various domains, including text, image, audio, video, etc. The proposed framework consists of three key stages:

- (1) task decomposition to concrete the objective of the task,
- (2) a Thoughts-on-Graph (ToG) paradigm to search the optimal solution path on the constructed tool graph, (3) and an execution engine with a versatile toolbox to execute solution efficiently. We conduct extensive experiments and demonstrate that our ControlLLM achieves superior performance regarding tool selection, argument assignment, and overall solution effectiveness compared to existing methods.

###### 4.6. Qualitative Analyses

Fig. 3 shows two simple cases to illustrate the capabilities of our ControlLLM in task planning. In contrast to HuggingGPT [32], we find our method is able to generate more diverse solutions to meet users’ expectations, thanks to the Thoughts-on-Graph paradigm. In Sec. 12, we provide extensive case studies across different modalities to validate the user experience for our method in practice.

Nevertheless, this work still has some limitations. Since the goal of this work is to improve the accuracy of tool usage, even if the solution is theoretically feasible, we cannot guarantee that the output from tools is always correct. On the other hand, due to the inherent ambiguity in natural language, it is difficult to ensure that the optimal solution selected is consistent with the user’s goal. In this case, we can only provide more alternative solutions searched by ToG for users to choose from if the optimal solution fails.

#### 5. Conclusion

In this paper, we propose ControlLLM, a multi-modal interaction framework that can accurately control tool usage

#### References

- [1] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, et al. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691, 2022. 1
- [2] Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Michał Podstawski, Hubert Niewiadomski, Piotr Nyczyk, and Torsten Hoefler. Graph of Thoughts: Solving Elaborate Problems with Large Language Models, 2023. 3
- [3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. 3
- [4] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality, 2023. 2
- [5] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022. 3
- [6] Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416, 2022. 3
- [7] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards generalpurpose vision-language models with instruction tuning,

2023. 3

- [8] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui. A survey for in-context learning. arXiv preprint arXiv:2301.00234, 2022. 3
- [9] Runpei Dong, Chunrui Han, Yuang Peng, Zekun Qi, Zheng Ge, Jinrong Yang, Liang Zhao, Jianjian Sun, Hongyu Zhou, Haoran Wei, et al. Dreamllm: Synergistic multimodal comprehension and creation. arXiv preprint arXiv:2309.11499,

2023. 3

- [10] Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14953–14962, 2023. 3
- [11] Shibo Hao, Tianyang Liu, Zhen Wang, and Zhiting Hu. Toolkengpt: Augmenting frozen language models with massive tools via tool embeddings. arXiv preprint arXiv:2305.11554, 2023. 3
- [12] Namgyu Ho, Laura Schmid, and Se-Young Yun. Large language models are reasoning teachers. arXiv preprint arXiv:2212.10071, 2022. 3

- [13] Jing Yu Koh, Daniel Fried, and Ruslan Salakhutdinov. Generating images with multimodal language models. arXiv preprint arXiv:2305.17216, 2023. 3
- [14] Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. arXiv preprint arXiv:2308.00692,

2023. 3

- [15] Jean-Claude Latombe. Robot motion planning. Springer Science & Business Media, 2012. 3
- [16] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 12888–

12900. PMLR, 2022. 1

- [17] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023. 1, 3
- [18] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023. 3
- [19] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485,

2023. 2, 3

- [20] Zhaoyang Liu, Yinan He, Wenhai Wang, Weiyun Wang, Yi Wang, Shoufa Chen, Qinglong Zhang, Zeqiang Lai, Yang Yang, Qingyun Li, Jiashuo Yu, et al. Interngpt: Solving vision-centric tasks by interacting with chatbots beyond language. arXiv preprint arXiv:2305.05662, 2023. 1, 2, 3, 6, 7, 8
- [21] Lei Ma, Jincong Han, Zhaoxin Wang, and Dian Zhang. Cephgpt-4: An interactive multimodal cephalometric measurement and diagnostic system with visual large language model. arXiv preprint arXiv:2307.07518, 2023. 1
- [22] Seungwhan Moon, Andrea Madotto, Zhaojiang Lin, Tushar Nagarajan, Matt Smith, Shashank Jain, Chun-Fu Yeh, Prakash Murugesan, Peyman Heidari, Yue Liu, et al. Anymal: An efficient and scalable any-modality augmented language model. arXiv preprint arXiv:2309.16058, 2023. 3
- [23] Yao Mu, Qinglong Zhang, Mengkang Hu, Wenhai Wang, Mingyu Ding, Jun Jin, Bin Wang, Jifeng Dai, Yu Qiao, and Ping Luo. Embodiedgpt: Vision-language pretraining via embodied chain of thought. arXiv preprint arXiv:2305.15021, 2023. 3
- [24] OpenAI. Chatgpt (Mar 14 version) [large language model]. 6, 2023. 1
- [25] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35: 27730–27744, 2022. 3
- [26] Aaron Parisi, Yao Zhao, and Noah Fiedel. Talm: Tool augmented language models. arXiv preprint arXiv:2205.12255,

2022. 3

- [27] Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E Gonzalez. Gorilla: Large language model connected with massive apis. arXiv preprint arXiv:2305.15334, 2023. 3

- [28] Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023. 3
- [29] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. Toolllm: Facilitating large language models to master 16000+ real-world apis. arXiv preprint arXiv:2307.16789,

2023. 1, 3

- [30] Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446, 2021. 3
- [31] Timo Schick, Jane Dwivedi-Yu, Roberto Dess`ı, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. arXiv preprint arXiv:2302.04761, 2023. 1, 3
- [32] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in huggingface. arXiv preprint arXiv:2303.17580, 2023. 1, 2, 3, 6, 7, 8, 9, 14, 16
- [33] David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. nature, 550(7676): 354–359, 2017. 3
- [34] Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, and Le Sun. Toolalpaca: Generalized tool learning for language models with 3000 simulated cases. arXiv preprint arXiv:2306.05301, 2023. 1
- [35] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/ stanford_alpaca, 2023. 16
- [36] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023. 1, 3, 6, 13, 16
- [37] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 1, 6
- [38] Sai Vemprala, Rogerio Bonatti, Arthur Bucker, and Ashish Kapoor. Chatgpt for robotics: Design principles and model abilities. Microsoft Auton. Syst. Robot. Res, 2:20, 2023. 1
- [39] Wenhai Wang, Zhe Chen, Xiaokang Chen, Jiannan Wu, Xizhou Zhu, Gang Zeng, Ping Luo, Tong Lu, Jie Zhou, Yu Qiao, et al. Visionllm: Large language model is also an open-ended decoder for vision-centric tasks. arXiv preprint arXiv:2305.11175, 2023. 1, 3

- [40] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171,

2022. 2, 3, 6

- [41] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022. 1, 2, 3
- [42] Sholom M Weiss, Casimir A Kulikowski, Saul Amarel, and Aran Safir. A model-based method for computer-aided medical decision-making. Artificial intelligence, 11(1-2):145– 172, 1978. 3
- [43] Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. Visual chatgpt: Talking, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671, 2023. 1, 2, 3, 6, 7, 8
- [44] Shengqiong Wu, Hao Fei, Leigang Qu, Wei Ji, and Tat-Seng Chua. Next-gpt: Any-to-any multimodal llm. arXiv preprint arXiv:2309.05519, 2023. 3
- [45] Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu Li, and Ying Shan. Gpt4tools: Teaching large language model to use tools via self-instruction, 2023. 1, 2, 3, 6, 7, 8
- [46] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022. 1, 3
- [47] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601, 2023. 1, 2
- [48] Yao Yao, Zuchao Li, and Hai Zhao. Beyond chain-ofthought, effective graph-of-thought reasoning in large language models. arXiv preprint arXiv:2305.16582, 2023. 3
- [49] Haoxuan You, Haotian Zhang, Zhe Gan, Xianzhi Du, Bowen Zhang, Zirui Wang, Liangliang Cao, Shih-Fu Chang, and Yinfei Yang. Ferret: Refer and ground anything anywhere at any granularity, 2023. 3
- [50] Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, et al. Glm-130b: An open bilingual pre-trained model. arXiv preprint arXiv:2210.02414, 2022. 3
- [51] Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199,

2023. 1

- [52] Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-of-interest. arXiv preprint arXiv:2307.03601, 2023. 3
- [53] Kaizhi Zheng, Xuehai He, and Xin Eric Wang. Minigpt5: Interleaved vision-and-language generation via generative vokens. arXiv preprint arXiv:2310.02239, 2023. 3

- [54] Kaizhi Zheng, Xuehai He, and Xin Eric Wang. Minigpt5: Interleaved vision-and-language generation via generative vokens, 2023. 3
- [55] Denny Zhou, Nathanael Sch¨arli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, et al. Least-to-most prompting enables complex reasoning in large language models. arXiv preprint arXiv:2205.10625, 2022. 3

### ControlLLM: Augment Language Models with Tools by Searching on Graphs Supplementary Material

#### 6. ControlLLM-ChatGPT

In this variant, we implement language model M with ChatGPT. As such, we elaborately design a series of prompts for each module.

###### 6.1. Task Decomposition

The prompt in Table 6 is designed for task decomposition in ControlLLM-ChatGPT. It guides the ChatGPT to decompose the user request into several subtasks. Table 5 shows the output protocol of task decomposition.

###### 6.2. Tool Assessment

Table 7 outlines a prompt design for the task of tool assessment, where the AI assistant evaluates tools’ suitability for a given task. The output is captured in the JSON format, including reasons and a score. The scoring criteria range from 1 to 5, reflecting the tool’s relevance to the task. The prompt emphasizes the connection between tool descriptions and task requirements. This prompt guides AI in making informed decisions when assessing tools’ utility for a specific task.

###### 6.3. Solution Expert

In this section, we delve into the core concept of the solution expert that streamlines the process of evaluating and selecting optimal solutions from all possible candidates. By systematically converting each solution into a formatted string description, the solution expert enables us to make informed decisions based on evaluated scores.

Solution Description Formatting. To facilitate the solution expert to comprehend the solution, we need to generate the description for each solution candidate. This involves transforming raw solution data into structured, formatted string descriptions. These descriptions encapsulate the information including functionality, inputs and output.

Solution Evaluation. The solution expert capitalizes on prompt engineering techniques to assess each solution based on subtask descriptions and formatted solution descriptions. The designed prompts guide language model M to evaluate the feasibility of each solution against the objective of the subtask. Through this process, we can assign scores to solutions, gauging their effectiveness and relevance to the task. It must ensure that the evaluation process is focused, targeted, and aligned with the subtask. The prompt template is shown in the Table 8.

Solution Ranking. The final aim of this module is to select the top-performing solutions. The optimal solution is

identified as the highest score assessed in the solution evaluation. Given that sometimes the selected optimal solution may not meet the user requirements, we also provide several alternative solutions by setting a threshold score of 3. These solutions, which exhibit a higher degree of alignment with the subtask’s requirements, emerge as the most promising candidates for user preference.

Through collaborative efforts, the optimal solution expert ensures that solutions are appropriately tailored, optimized, and well-adapted to the task.

###### 6.4. Resource Expert

In the algorithm of ToG, we encounter a challenge stemming from the potential presence of multiple instances of the same resource type within the available resource list. This challenge introduces complexity, making it difficult to straightforwardly deduce certain arguments for tools using predefined rules. As a result, we design a solution expert.

This module transforms the task of argument assignment into a fill-in-the-blank exercise. To achieve this, we design a resource expert crafts with prompts that not only incorporate the task description but also include the available resource list. In this manner, a language model M is employed to dynamically complete the missing parameters within a solution by interacting with the contextual information presented. We put the prompt template in the Table 9.

#### 7. ControlLLM-LLaMA

For ControlLLM-LLaMA, we use the LLaMA-7b [36] as language model M to solve the problems in task decomposition, tool assessment, solution expert, resource expert.

###### 7.1. Instruction Generation

The first step to train M is to construct the instruction corpus. We here opt for ChatGPT-3.5 to generate the training corpus. The following steps will elaborate on the details of instructions generation for task decomposition, tool assessment, solution expert, and resource expert, respectively.

For task decomposition, we generate two different types of instructions as follows: 1) Basic instructions, where they only contain one subtask after task decomposition. We set some seed instructions with ground-truth results of task decomposition, which serve as initial templates for generating more diverse instructions. Then, we use ChatGPT to generate more diverse instructions based on the pre-defined seed instructions. During the generation process, we center on the seed instructions and produce more instructions

|Field<br><br>|Description|
|---|---|
|description|a brief summary of subtask. It gives some guidance on how to approach the problem for ToG.<br><br>|
|domains<br><br>|the domain scope that tools required by this task fall into. It helps ToG narrow down the search space and find the most relevant and suitable tools for the subtask. We showcase some domains in Table 11.|
|args|the inputs that the user provides for this subtask. It is usually in the form of key-value pairs, where the key is the type of the argument, and the value is the resource path or text you want to use. For example, [{“type”: “image”, “value”: “image 1.png”}, {“type”: “text”, “value”: “remove the dog in the picture”}].<br><br>|
|return|the expected output of the subtask. For example, the return is {“image”: “⟨GEN⟩-0”}, which means the expected output is an image and “⟨GEN⟩-0” is just a temporary placeholder.|

Table 5. The output protocol of task decomposition. We elaborate on each field in the output of task decomposition.

|The following is a friendly conversation between a human and an AI. The AI is professional and parses user input to several tasks with lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. The AI assistant can parse user input to several tasks with JSON format as follows: <Solution>[“description”: task description, “task”: [task domain 1, task domain 2], “id”: task id, “dep”: dependency task id, “args”: [“type”: “text”, “image” or audio, “value”: text, image url or <GEN>-dep id], “returns”:[“type”: “segmentation”, “value”: “<GEN>-task id”]]</Solution>. The ”description” should describe the task in detail, and AI assistant can add some details to improve the user’s request without changing the user’s original intention. The special tag “<GEN>-dep id” refers to the one generated text/image/audio/video/segmentation/mask in the dependency task (Please consider whether the dependency task generates resources of this type.) and “dep id” must be in “dep” list. The special tag “<GEN>-task id” refers to the one generated text/image/audio/video/segmentation/mask in this task and “task id” should be in line with field “id” of this task. The “dep” field denotes the ids of the previous prerequisite tasks, which generate a new resource that the current task relies on. The “args” field and the “returns” field denotes the input resources and output resources of this task, respectively. The type of resource must be in [“text”, “image”, “line”, “normal”, “hed”, “scribble”, “pose”, “edge”, “bbox”, “category”, “segmentation”, “audio”, “video”, “segmentation”, “mask”], nothing else. The “task” MUST be selected from the following options: “question-answering”, “visual-question-answering”, “imagegeneration”, “image-editing”, “image-perception”, “image-processing”, “audio-perception”, “audio-generation”, “audioediting”, “video-question-answering”, “video-perception”, “video-generation”, “video-editing”, nothing else. Think step by step about all the tasks that can resolve the user’s request. Parse out as few tasks as possible while ensuring that the user request can be resolved. Pay attention to the dependencies and order among tasks. If some inputs of tools are not found, you cannot assume that they already exist. You can think of a new task to generate those args that do not exist or ask for the user’s help. If the user request can’t be parsed, you need to reply empty JSON []. You should always respond in the following format:<br><br><Solution><YOUR SOLUTION></Solution> <YOUR SOLUTION>should be strict with JSON format described above.<br><br>|
|---|

Table 6. The prompt for task decomposition. It is inspired by [32].

using more diverse expressions and styles. These instructions need to share the task decomposition results with the seed instructions as ground truth. 2) Compound instructions, which involve multiple subtasks and intermediate resources. We simply assemble the basic instructions into the compound instructions in a coherent and logical manner. It aims to enhance the improve the complex interaction capability of the system by enabling the model to handle user requests that span multiple domains and require multiple steps of processing. We here generate almost 100k instructions for training. The instructions generated in this step will be used in the following tasks as well.

For the tool assessment, solution expert, and resource expert, we use prompts in Table 7, 8 and 9 to collect the output from ChatGPT by running ControlLLM on the instructions generated above. Unlike directly generating the solution, these tasks only involve making a decision, like scoring the tools or solutions based on the input, so they are relatively simple, and ChatGPT with strong zero-shot capabilities, can easily solve them. Therefore, we opt to directly distill the knowledge of ChatGPT by using prompt techniques. Through the experiments, we verify the feasibility of this strategy.

|Given a task and a tool, the AI assistant helps the system decide whether this tool can process the task. The assistant should focus more on the description of the model and give a score to each tool. The AI assistant respond with JSON format as follows: <Solution>“Thought”: thought, “Score”: score </Solution>. The “Thought” field records the model’s thinking process step by step within 80 words, which gives the reasons why giving this score. The “Score” field denotes a score that assesses whether this tool is useful for this task. Score is in [1, 2, 3, 4, 5]. Here are the scoring criteria: “Score”=1: The tool is totally not related to the task and does not provide any useful output for solving the<br><br>task. “Score”=2: The tool is somewhat not related to the task and may not provide any useful output for solving the<br>task. “Score”=3: The tool is probably related to the task and provides some intermediate output that is partially helpful for solving the task, but it may not be the optimal one. “Score”>3: The tool is closely or directly related to the task and provides an output that is mostly helpful for solving the task or that matches the returns of the task with regard to the type. In a nutshell, for the given task, the higher the score, the more useful the tool is. You should always respond in the following format: <Solution>SOLUTION </Solution>. \n‘SOLUTION‘ should strictly comply with the SON format described above. Task description: “{{task}}”.\n\n Here is the description of the tool “{{tool name}}”: \n{{tool name}}: {{tool description}}\nArgs: \n{{arguments}}\nReturns: \n{{returns}}\n\nThe above information may be useful for AI to make decision. Please refer to the scoring criteria and score the tool {{tool name}} for this task. Notice that If the tool description contains keywords from the task description, the score of this tool should be greater than or equal to 3.<br><br><br>|
|---|

- Table 7. The prompt for tool assessment.

|Given a task and a solution, The AI assistant needs to score the solution and respond in JSON format. Please notice that the AI assistant should think. The AI assistant should pay more attention to the relevance between the description of each tool in the solution and task. The AI assistant respond with JSON format as follows: <Solution>{“Thought”: “thought”, “Score”: score}</Solution>. “Thought” field records the model’s thinking process step by step within 80 words, which gives the reasons why giving this score. “Score” field denotes a score that assesses whether this tool is helpful for this task. “Score” is in [1, 2, 3, 4, 5]. Here are the scoring criteria: “Score”=1: The solution is totally not related to the user’s request and can not solve the task. “Score”=2: The solution is somewhat not related to the user’s request and may not solve the task. “Score”=3: The solution is probably related to the user’s intention and may solve the task, but it may not be the optimal one. “Score”>3: The solution is closely or directly related to what the user wants and could satisfactorily solve the task. In a nutshell, the higher the score, the greater the likelihood of the solution solving the given task. You should always respond in the following format: \n<Solution>‘SOLUTION‘ </Solution>\n‘SOLUTION‘ should strictly comply with the JSON format described above. \nUser’s request: “{{request}}” Task description: “{{task}}”. Here is the description of the solution: {{solution}} Please refer to the scoring criteria and score this solution based on the task description. You should think carefully before scoring the solution. Notice that If the keywords in the solution are close in meaning to the keywords in the task description, then the score of this solution is at least 3.|
|---|

- Table 8. The prompt for solution expert.

|Some tools have missing input arguments, and the AI assistant needs to infer these missing inputs from the context. Please notice that the AI assistant should never fake the resources that do not exist. The returned input arguments should be JSON format as follows: [{“image”: “xxx.png”}, {“bbox”: “<GEN>-detr-bbox-0”}, “text”: “<text>”]. AI assistant should always respond in the following format: \n“<Explanation>[briefly explain your choice here]</Explanation><Solution>‘SOLUTION‘ </Solution>”. \n‘SOLUTION‘ should be strictly in the JSON format described above. \nUser’s request: \n“{{request}}”\nTask: \n“{{task description}}”. \n<Resources>: \n{{resources}}. \nWe use {{tool name}} to solve this task: \n‘{{tool name}}‘: {{tool description}} \nArgs: {{arguments}} \nReturns: {{returns}} \nFor the type of “text”, the AI assistant should summarize the content from the context based on the task and the tool’s description. For other types of input, the AI assistant needs to select the inputs from <Resources>. Now we prepare the inputs for {{tool name}}: {{input}}. Please complete these inputs and return the completed inputs with the format described above like: <Solution>‘SOLUTION‘ </Solution>.<br><br>|
|---|

- Table 9. The prompt for resource expert.

|Your name is ControlLLM, an AI-powered assistant. For user’s request, the system executes the solution and collects the results based on the following workflow. You need to respond to user requests based on the following information. Here are the information for your reference. ## User Request {{request}} ## Workflow and Execution Results {{solution}} ## Summarized Results {{results}}<br><br>You must first answer the user’s request in a straightforward manner. Some of the results may not be accurate and need you to use your judgment in making decisions. Then please explain your workflow, including the tools and returned results for the request, in a friendly way. If the answers contain file paths, you have to repeat the complete file path. Only if there is nothing in the Summarized Results, you need to tell the user you can not finish the task.|
|---|

Table 10. The Prompt Design in Response Generation. We here refer to the prompts from [32] to generate a user-friendly response.

###### 7.2. Training Recipes

We follow the training protocol in [35], where LLaMA [36] is used as an alternative choice for our language model M. It is finetuned for three epochs with the initial learning rate 2e-5 and consine decay. We fix the training batch size as 128 by adaptively setting “gradient accumulation steps”. The whole training procedure is on 8xA100 GPUs.

#### 8. ControlLLM-Mix

In practice, we find ControlLLM-ChatGPT has difficulty in task decomposition, especially for hard instructions. In addition, ControlLLM-LLaMA is good at task decomposition due to an instruction-tuned language model M while other modules are slightly inferior to ChatGPT. Because we finetune M by distilling the knowledge from ChatGPT to assess tools, ranking solutions, and assign arguments. As a result, we design ControlLLM-Mix to integrate the benefits from the other variants. For task decomposition, we use the same method from ControlLLM-LLaMA to finetune LLaMA-7B to decompose user requests. For the remaining modules, we directly utilize the ChatGPT, sharing the same prompt design from ControlLLM-ChatGPT. In experiments, ControlLLM-Mix achieves the best performance.

#### 9. Response Generation

We design a prompt template for the Response Generation task in Table 10. In this task, the AI assistant is tasked with explaining the process and outcomes using input and inference results. The AI is instructed to respond directly to the user’s request, followed by describing the task’s procedure, offering analysis, and presenting model inference results using a first-person perspective. If the results involve file paths, the complete path should be provided, or if there are no results, the AI should communicate its inability. The prompt sets the context for generating informative and user-

understandable responses during response generation.

#### 10. Resource Types and Tools

We initially define a type set containing a series of resource type descriptors, such as “text”, “tags”, “html”, “image”, “video”, “audio”, “segmentation”, “edge”, “line”, “hed”, “canny”, “scribble”, “pose”, “depth”, “normal”, “mask”, “point”, “bbox” and “category”. The type set is easy to extend depending on the toolkit. The types of inputs of tools must fall into the pre-defined type set.

In Table 11, we exhibit lots of tools supported by our framework across different domains, including natural language, image, audio, video, etc. The whole system will continue to evolve.

##### 11. Metrics for Tool Selection A) Irrelevant Tool Inclusion Rate (abbr. IR):

F(sp) =

true, sp contains the irrelevant tools false, otherwise

, (4)

|Sp| i I(F(Sip))

, (5)

IR =

|Sp|

where I is indicator function, | · | represents the number of elements in a set, and Sp denotes all predicted solutions on our benchmark. It measures the proportion of the predicted solutions that contain the irrelevant tools. A higher IR indicates that the method tends to include more unnecessary tools, potentially hindering effective task planning. This metric gauges the performance of methods in excluding irrelevant tools.

B) Necessary Tool Inclusion Rate (abbr. NR):

H(sp) =

true, Solution sp contains necessary tools false, otherwise

, (6)

|Domains|Tools<br><br>|
|---|---|
|question-answering|question answering, image question answering<br><br>|
|natural-language-processing|summarization, title generation, text to tags, text to text generation, sentiment analysis<br><br>|
|image-perception<br><br>|object detection, image captioning, visual grounding, image classification, segment anything, instance segmentation, segment by points<br><br>|
|image-generation|text to image, image to image, line text to image, hed text to image, scribble text to image, pose text to image, segmentation text to image, edge text to image, depth text to image, normal text to image<br><br>|
|image-editing|text image editing, image inpainting, image cropping, mask image, highlight object on image<br><br>|
|image-processing<br><br>|image to edge, image to line, image to hed, image to scribble, image to pose, image to depth, image to normal<br><br>|
|video-perception<br><br>|video classification, video captioning<br><br>|
|video-processing|dub video, video to webpage<br><br>|
|video-generation<br><br>|image audio to video, image to video, text to video<br><br>|
|audio-perception<br><br>|audio classification<br><br>|
|audio-generation|text to music, text to speech, audio to Audio<br><br>|

Table 11. A subset of domains and tools in our tool kit.

|Sp| i I(H(Sip))

. (7)

NR =

|Sp|

The necessary tools play a critical role in solving the user request. For example, if users want to know the position of a specific object, the object detection tool is necessary. This metric measures the proportion of solutions that contain the necessary tools for solving the task but without considering whether the arguments of tools are correct. It checks whether the solution has all the necessary tools that can produce the expected output. A high NR value means that the method has a strong ability in task planning and finding the right tools for the user’s request.

###### 11.1. Metrics for Argument Assignment A) Resource Hallucination Rate (abbr. HR):

P(sp) =

true, spcontains false resources false, otherwise

, (8)

|Sp| i I(P(Sip))

. (9)

HR =

|Sp|

This indicator reveals the extent of hallucination when inferring the arguments for tools. It checks whether all the arguments of the tools exist in the input resources or not. A low HR value means that the method rarely leads to hallucinations that are common in LLMs.

B) Resource Type Consistency Rate (abbr. CR):

Q(sp) =

true, No resource type conflict in sp false, otherwise

, (10)

|Sp| i I(Q(Sip))

. (11)

CR =

|Sp|

This metric examines whether the types of resources used as inputs for the predicted solution match those of the corresponding tools. It evaluates the method’s ability to ensure consistency between argument types and tools. A high CR value means that the method can correctly infer and assign arguments for each tool.

###### 11.2. Solution Evaluation

The Solution Evaluation (abbr. SE):

W(sp) =

true, sp can solve the task false, otherwise

, (12)

|Sp| i I(W(Sip))

. (13)

SE =

|Sp|

This metric measures the success rate of all generated solutions on our benchmark, regardless of whether it contains irrelevant tools, as long as the chain of tool invoking outputs the information that is able to solve the task. A higher score in the solution evaluation indicates that the method is able to provide an effective solution to user requests.

In summary, these intuitive metrics together provide a comprehensive assessment of tool-augmented LLMs in terms of tool selection, argument assignment, and overall effectiveness in addressing user queries.

###### Easy

- 1. Please determine if the image 1.png contains a platyhelminth?

- 2. How can I design a sleep monitoring system in C# that can accurately detect a baby’s specific sleep stages and predict when they will enter a light sleep stage within the next hour? And once this prediction is made, how can I alert the parent or caregiver that the baby will be waking up soon and suggest soothing methods to ease the transition from sleep to wakefulness? Also, how can I modify the statement “The baby is sleeping” to reflect this predictive system in C#?
- 3. Please extract the scribble result for the image in image 2.png”

- 4. With the HED result image 3.png, generate a new image that features a zoo landscape with various animals, a couple with their children, and a fountain.

- 5. Given the image image 4.png, What is unique about the window in the room? Medium

- 6. Can you generate a new image that has a similar layout to the file named ’image 5.png’? I’m particularly interested in the positioning of the elements. The new image should have the same arrangement of elements and their positioning.

- 7. Generate a new image conditioned on the segmentation from image 6.png and the new image contains a majestic mountain range with a clear blue sky and a herd of wild horses running free.

- 8. Take away the umbrella from the picture image 7.png.

- 9. Crop out the baseball glove in image 8.png

- 10. Provide me with a mask that separates the bear from the rest of the image 9.png? Hard

- 11. provide the number of umbrellas presented in the image 10.png, image 11.png, image 12.png, image 13.png, image 14.png

- 12. Can you elaborate on the elements of the image 15.png, image 16.png and image 17.png provided?

- 13. Erase the laptop from the image 18.png,image 19.png and image 20.png

- 14. Create a new image using the segmentation from image 21.png that showcases a cozy cabin in the woods with a dog and a cat, surrounded by snow-covered trees. Can you crop out the dog from given image? I’m looking for the cat in the image file, can you guide me to it?

- 15. Can you determine whether image 22.png contains a mouse? Please provide a list of all the objects present in the image, with a special emphasis on the killer. Is image 23.png displaying a banana? As for the image, what skills are important for improving one’s performance in the depicted scenario?

Table 12. Test samples of instruction in the benchmark.

#### 12. Case Studies

We provide more cases across different modalities to validate the user experience for our method in practice. In Fig. 4, we show some cases of image perception, which involves analyzing and understanding the content of an image, such as detecting objects, counting objects, finding objects, segmenting objects, answering questions about the image, etc. These tasks require the system to invoke tools to process visual information and extract relevant features and labels from the image. Fig. 5 gives examples of image processing and image editing, which assist users in processing or editing the image according to some criteria or instructions. Fig. 6 mainly focuses on image question answering and image generation, showing the graphic dialogue capability. In Fig. 7, we provide some multi-modal interaction cases on image, video, and audio domains. In addition, we

also illustrate the capabilities of complicated scenarios with solutions searched by ToG during task planning in Fig. 8 and Fig. 9. These complex tasks involve combining multiple tools to find an advanced and creative solution path that can solve more challenging problems. It requires a system that can integrate different types of information and outputs from tools and generate comprehensive and meaningful responses based on execution results. These figures demonstrate the strong capabilities of ControlLLM in task planning for both simple and complicated scenarios. It thus leads to a better user experience.

image-perception User: Provide me with the count of bears in the input image

[Figure 23]

AI: The count of bears in the input image is 3. Here is a summary of the workflow and results for your request:

- 1. The image was processed using an object detection tool, which detected the presence of bears in the image.
- 2. The bounding boxes around the detected bears were selected by bounding boxes selection tool.
- 3. The count of bears was then determined using a count objects tool.
- 4. The result of the count objects tool indicated that there are 3 bears in the image.

[Figure 24]

User: I'm trying to find the umbrella in the image, can you assist me?

AI: Yes, I can assist you in finding the umbrella in the image. After analyzing the image, I have identified the location of the umbrella and cropped it for better visibility. To achieve this, I used the following tools in my workflow:

- 1. Object Detection: This tool was used to detect various objects in the image, including the umbrella. The tool provided a list of detected objects with their corresponding labels and bounding box coordinates.
- 2. Select Bounding Box: This tool was used to select the bounding box corresponding to the umbrella from the list of detected objects.
- 3. Image Cropping: This tool was used to crop the image based on the selected bounding box of the umbrella. The result is a new image that focuses specifically on the umbrella.

[Figure 25]

I hope this explanation clarifies the steps I took to find and extract the umbrella from the image. If you have any further questions, feel free to ask!

[Figure 26]

User: Give me the mask of elephant in the image.

AI: I have generated a mask of an elephant for you. To generate this mask, I used an image instance segmentation tool on the image. The tool identified different objects in the image and created masks for them. Then, I used a mask selection tool to specifically select the mask for the elephant based on the condition provided. Please note that the result may not be completely accurate, as it relies on the performance of the tools used.

[Figure 27]

[Figure 28]

User: Can you highlight the giraffe for me in the image?

AI: I have highlighted the giraffe for you. To fulfill your request, I followed the following workflow:

- 1.First, I used an object detection tool to locate the elephant in the image.
- 2.Then, I used a selection tool to choose the bounding box that contains the elephant.
- 3.Finally, I used an image highlighting tool to highlight the selected bounding box on the image. The output of this workflow is the highlighted image, which shows the giraffe highlighted.

[Figure 29]

###### Figure 4. Case study on different task domains (a).

image-processing User: Extract the edge information for this image.

[Figure 30]

AI: I have successfully extracted the edge information from the image you provided. To achieve this, I used an image processing tool called “image_to_edge”. I passed your input image as input to the tool. The tool processed the image and generated a new image file containing only the edge information.

[Figure 31]

[Figure 32]

User: Please get me the pose information for the image.

AI: Sure! Based on your request, I have obtained the pose result for the input image. To obtain this result, I used the “image_to_pose” tool. I provided the tool with the image file as input. The tool processed the image and generated the pose result.

[Figure 33]

[Figure 34]

User: Please segment anything without label information for this image.

AI: I have segmented anything without label information for the image you mentioned. Now, let me explain the workflow and the tools used for your request. I used a tool called "segment_anything" to perform the segmentation. The input for this tool was the image file named "47e56e_None_input_image.jpg". The tool processed the image and generated a segmentation result.

[Figure 35]

image-editing

[Figure 36]

User: Erase the sheep from the photo

AI: I have edited the picture and changed the color of the horse to blue.

[Figure 37]

To address your request, I used text_image_editing tool, which can edit the image guided by a textual prompt.

[Figure 38]

User: Erase the sheep from the photo

AI: I have successfully erased the sheep from the photo. Now, let me explain how I accomplished this task. First, I used an image instance segmentation tool to identify the sheep in the photo. This tool generated several mask images highlighting the sheep in different parts of the photo. Then, I used a select mask tool to choose the most accurate masks that represent the sheep. These selected masks were saved as separate images. Finally, I used an image inpainting tool to remove the sheep from the original photo, using the selected masks as a guide. Please let me know if there's anything else I can help you !

[Figure 39]

###### Figure 5. Case study on different task domains (b).

image-question-answering User: Why is this image so funny?

[Figure 40]

AI: The image is funny because it features two toy frogs, one of which is holding a camera and taking a picture of the other. The scene is set in a grassy field, and the toy frogs are posed in a way that mimics human behavior, specifically a photographer capturing a subject. The humor comes from the unexpected and anthropomorphic portrayal of the toy frogs, as it is not common to see inanimate objects, such as toys, engaging in activities like photography. This playful and whimsical scene can evoke amusement in the viewer. I used an image question answering tool. The tool analyzed the image and determined why this image is so funny.

User: Given the image, is there going to be a potentially dangerous situation in the picture?

[Figure 41]

AI: There is a potentially dangerous situation as a man is riding on the back of an elephant, while another elephant is walking through the water. The presence of two elephants in close proximity to each other and the man could pose a risk of accidents or injuries, especially if the elephants become agitated or frightened. It is essential for the man to maintain control over the elephant he is riding and ensure that the other elephant is calm and well-behaved to minimize any potential dangers. To solve your question, I used an image question answering tool.

User: how many people can be seen riding the elephant in the image?

AI: In the image, there is one person riding on the back of the elephant. I used an image question answering tool to answer your question.

image-generation

[Figure 42]

User: I want a new image that has mountains and sunset. Can you use the depth result from input image to create it?

AI: The new image is generated. The workflow involved two tools: "image_to_depth" and "depth_text_to_image". The first tool, “image_to_depth”, used an input image to generate a depth result. The second tool, "depth_text_to_image", took a text input of "The new image should have mountains and sunset." and a depth map (generated by the previous tool)

[Figure 43]

[Figure 44]

User: please generate a new image based the edge from this image. The new image shows a beautiful tower in the rain.

AI: The new image generated. Workflow: The image provided was processed using an image-to-edge tool, which generated an edge representation of the image. The edge representation was then used along with the text “The new image shows a beautiful tower in the rain” in an edge-text-to-image tool. This tool combined the edge representation and the text to generate a new image.

[Figure 45]

###### Figure 6. Case study on different task domains (c).

image & video & audio tasks

User: Convert the following sentence into speech: “Hope is the thing with feathers That perches in the soul, And sings the tune without the words, And never stops at all”

AI: I have converted the following sentence into speech for you: “Hope is the thing with feathers That perches in the soul, And sings the tune without the words, And never stops at all”. To do this, I used a text_to_speech tool that can generate speech from text. I gave it the input sentence and it returned an audio file with a humanlike voice reading the sentence. The tool can also adjust the speed, pitch, and accent of the voice to suit different preferences. I hope you like the results.

[Figure 46]

User: Generate an image where a woman is surfing. Then convert this image to a dynamic video.

[Figure 47]

AI: I have generated an image where a woman is surfing and converted it to a dynamic video for you. To do this, I used two tools that can create and manipulate visual content. The first tool called text_to_image can generate an image based on a text description. I gave it the input ‘a woman is surfing’ and it returned an image of a woman surfing on a blue ocean wave. The second tool is image_to_video that can convert an image to a video by adding motion and sound effects. I gave it the image from the first tool, and it returned a video of the woman surfing with realistic water and wind sounds. I hope you like the results.

[Figure 48]

[Figure 49]

[Figure 50]

User: What happened in this video?

[Figure 51]

AI: In this video, a basketball game is taking place, with a player attempting to shoot the ball. The player is holding the ball and getting ready to make a shot, while other players are on the court, possibly trying to defend or score. To solve your question, I used two tools: “action_recognition” and “video_captioning”, which both take the video as input.

User: Please generate music according to the following prompt: An 80s driving pop song with heavy drums and synth pads in the background

AI: Sure! I have generated a piece of music based on your prompt.

[Figure 52]

To create this music, I used a text-to-music tool. I provided the tool with the prompt you gave me, which was "An 80s driving pop song with heavy drums and synth pads in the background". The tool then generated a music composition that matches the description.

[Figure 53]

[Figure 54]

[Figure 55]

User: Can you help me dub this video with uploaded audio?

[Figure 56]

[Figure 57]

AI: Sure! Based on your request, I have dubbed the video with the audio for you to dub a video. I used a “dub_video” tool that can add or replace the audio track of a video. I gave it the input video file and the input audio file, and it returned a new video file with the audio synchronized with the video.

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

###### Figure 7. Case study on different task domains (d).

User: Please dub the video.mp4 with a suitable background music and then generate a HTML web page to introduce this video.

Task planning

[Action(tool_name='video_captioning‘, inputs={'video': 'video.mp4‘}, outputs=['<TOOL-GEN>-text-0‘]), Action(tool_name='text_to_music‘, inputs={'text': '<TOOL-GEN>-text-0‘}, outputs=['<TOOL-GEN>-text_to_music-audio0‘]), Action(tool_name='dub_video', inputs={'video': "video.mp4 ", 'audio': "<TOOL-GEN>-text_to_music-audio-0",}, outputs=['<TOOL-GEN>-dub_video-video-0’])] [Action(tool_name='title_generation', inputs={'text': '<TOOL-GEN>-text-0'}, outputs=['<TOOL-GEN>-text-1‘]), Action(tool_name='text_to_tags', inputs={'text': '<TOOL-GEN>-text-0'}, outputs=['<TOOL-GEN>-tags-0‘]), Action(tool_name='video_to_webpage', inputs={'video': '<TOOL-GEN>-dub_video-video-0', 'title': '<TOOL-GEN>-text-1', 'tags': '<TOOL-GEN>-tags-0', 'description': '<TOOL-GEN>-text-0'}, outputs=['<GEN>-0'])]

AI: Sure, I can generate a web page for your input video. I have executed a workflow using several tools to generate the desired web page. The video_captioning tool was used to generate a text description for the input video. The dub_video tool combined the input video with the generated audio to create a dubbed video. The title_generation tool was used to generate a title for the video based on the generated text. The text_to_tags tool generated a set of tags related to the video content. Finally, the video_to_webpage tool was used to generate a web page for the video, including the video itself, the generated title, tags, and description. The resulting web page includes the following elements:

|[Figure 62]<br><br>[Figure 63]|
|---|

1) A title prompt asking for a title to be generated within 10 words. 2) A video player to play the input video. 3) Tags related to the video content. 4) A video description.

User: Generate a new image based on prompt. Prompt: “A serene and beautiful landscape with a calm lake reflecting the blue sky and white clouds. There are some green hills and trees on the shore”. Then help me generate a promotional video for this generated image.

Task planning

[Action(tool_name='text_to_image', inputs={'text': 'A serene and beautiful landscape with a calm lake reflecting the blue sky and white clouds. There are some green hills and trees on the shore, and a small wooden boat floating on the water.'}, outputs=['<TOOL-GEN>-text_to_image-image-0‘]), Action(tool_name='image_captioning', inputs={'image': '<TOOL-GEN>-text_to_image-image-0'}, outputs=['<TOOL-GEN>-text-0‘]), Action(tool_name='text_to_speech', inputs={'text': '<TOOL-GEN>-text-0'}, outputs=['<TOOL-GEN>-text_to_speech-audio-0‘]), Action(tool_name='image_audio_to_video', inputs={'image': "<TOOL-GEN>-text_to_image-image-0", 'audio': "<TOOLGEN>-text_to_speech-audio-0"}, outputs=['<GEN>-0'])]

AI: Sure! I can help you generate a video to introduce an image. Based on the information provided, I have used a combination of tools to fulfill your request. First, I used the "text_to_image" tool to convert your description into an image. The tool created an image of a serene and beautiful landscape with a calm lake reflecting the blue sky and white clouds. There are also green hills and trees on the shore, as well as a small wooden boat floating on the water. Next, I used the "image_captioning" tool to generate a textual description of the image. The tool analyzed the image and produced a caption that best describes it. Then, I used the "text_to_speech" tool to convert the generated textual description into an audio file. This tool transformed the text into speech, allowing for a more engaging experience. Finally, I used the "image_audio_to_video" tool to combine the generated image and audio into a video. The tool merged the image and audio together, creating a video that introduces the image. The result of this workflow is a video file. This video showcases the image described in your initial request and includes the audio narration of its description.

[Figure 64]

[Figure 65]

The image features a serene scene of a boat …

###### Figure 8. Case study on different task domains (e).

User: Count horse in the image1.png, image2.png and image3.png

[Figure 66]

[Figure 67]

[Figure 68]

Task planning

[

- Action(tool_name='object_detection', inputs={'image': 'image1.png'}, outputs=['<TOOL-GEN>-object_detection-bbox0‘]), Action(tool_name='select_bbox', inputs={'bbox_list': '<TOOL-GEN>-object_detection-bbox-0', 'condition': 'bear'}, outputs=['<TOOL-GEN>-select_bbox-bbox-0‘]), Action(tool_name='count_objects', inputs={'bbox_list': '<TOOL-GEN>-select_bbox-bbox-0'}, outputs=['<GEN>-0']), ], [ Action(tool_name='image_question_answering', inputs={'text': 'Provide me with the count of bears in the input image',

- 'image': 'image1.png'}, outputs=['<GEN>-1']) ], [

Action(tool_name='object_detection', inputs={'image': 'image2.png'}, outputs=['<TOOL-GEN>-object_detection-bbox1']), Action(tool_name='select_bbox', inputs={'bbox_list': '<TOOL-GEN>-object_detection-bbox-0', 'condition': 'bear'}, outputs=['<TOOL-GEN>-select_bbox-bbox-1']), Action(tool_name='count_objects', inputs={'bbox_list': '<TOOL-GEN>-select_bbox-bbox-0'}, outputs=['<GEN>-2']), ], [ Action(tool_name='image_question_answering', inputs={'text': 'Provide me with the count of bears in the input image',

- 'image': 'image2.png'}, outputs=['<GEN>-3']) ] [

Action(tool_name='object_detection', inputs={'image': 'image3.png'}, outputs=['<TOOL-GEN>-object_detection-bbox1']), Action(tool_name='select_bbox', inputs={'bbox_list': '<TOOL-GEN>-object_detection-bbox-0', 'condition': 'bear'}, outputs=['<TOOL-GEN>-select_bbox-bbox-1']), Action(tool_name='count_objects', inputs={'bbox_list': '<TOOL-GEN>-select_bbox-bbox-0'}, outputs=['<GEN>-4']), ], [ Action(tool_name='image_question_answering', inputs={'text': 'Provide me with the count of bears in the input image',

- 'image': 'image3.png'}, outputs=['<GEN>-5']) ]

AI: Based on the information provided, the count of horses in the two images is eight. To provide you with this information, I followed a workflow using various tools. First, I used an object detection tool to detect objects in the images. Then, I used a tool called “select_bbox” to select the bounding boxes that contain the desired object, which in this case is a horse. Next, I used a tool called “count_objects” to count the number of horses based on the selected bounding boxes. For both two images, I used the same solution to count the horses. The “image_question_answering” tool is also used to double check the result. The results of this workflow indicate that there are four horses in image1.png, three horses in image2.png and one horse in image3.png. So, there are eight horses in total. However, please note that these results may not be completely accurate, as they are generated using automated tools and may require human judgement for accuracy.

###### Figure 9. Case study on different task domains (f).

