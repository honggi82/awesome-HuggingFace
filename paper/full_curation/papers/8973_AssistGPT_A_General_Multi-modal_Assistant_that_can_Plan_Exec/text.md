# arXiv:2306.08640v2[cs.CV]28Jun2023

## AssistGPT: A General Multi-modal Assistant that can Plan, Execute, Inspect, and Learn

[Figure 1]

#### Difei Gao, Lei Ji, Luowei Zhou, Kevin Qinghong Lin, Joya Chen, Zihan Fan, Mike Zheng Shou∗

Show Lab, National University of Singapore, https://showlab.github.io/assistgpt/

### Abstract

Recent research on Large Language Models (LLMs) has led to remarkable advancements in general NLP AI assistants. Some studies have further explored the use of LLMs for planning and invoking models or APIs to address more general multi-modal user queries. Despite this progress, complex visual-based tasks still remain challenging due to the diverse nature of visual tasks. This diversity is reflected in two aspects: 1) Reasoning paths. For many real-life applications, it is hard to accurately decompose a query simply by examining the query itself. Planning based on the specific visual content and the results of each step is usually required. 2) Flexible inputs and intermediate results. Input forms could be flexible for in-the-wild cases, and involves not only a single image or video but a mixture of videos and images, e.g., a user-view image with some reference videos. Besides, a complex reasoning process will also generate diverse multimodal intermediate results, e.g., video narrations, segmented video clips, etc. To address such general cases, we propose a multi-modal AI assistant, AssistGPT, with an interleaved code and language reasoning approach called Plan, Execute, Inspect, and Learn (PEIL) to integrate LLMs with various tools. Specifically, the Planner is capable of using natural language to plan which tool in Executor should do next based on the current reasoning progress. Inspector is an efficient memory manager to assist the Planner to feed proper visual information into a specific tool. Finally, since the entire reasoning process is complex and flexible, a Learner is designed to enable the model to autonomously explore and discover the optimal solution. We conducted experiments on A-OKVQA and NExT-QA benchmarks, achieving state-of-the-art results. Moreover, showcases demonstrate the ability of our system to handle questions far more complex than those found in the benchmarks.

### 1 Introduction

Large language models (LLMs) [1–4], especially ChatGPT [5], have made remarkable progress in recent months, significantly advancing the field of developing AI assistants. Despite these advances, a single LLM serving as an AI assistant still exhibits inherent limitations in certain abilities, such as understanding visual environments and comprehending complex tasks, which restrict their utility in real-world applications. To address these shortcomings, a promising solution is to explore the integration and collaboration of multiple domain experts e.g., pretrained models or APIs, to tackle complex tasks. Numerous efforts have been made in this direction. Some works [6–8] utilize language as a bridge and transform the visual input into pure texts using foundational visual models, such as captioner [9–11], object detectors [12–14], and OCR models [15, 16]. Subsequently, the extracted texts are fed into LLMs for reasoning tasks like question-answering. Nonetheless, as for complex

∗Corresponding author.

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

|[Figure 8]<br><br>[Figure 9]<br><br>visual-1|
|---|

Question: I am trying to add Midjourney to my own server. I have clicked the 'show member icon', what should I do in the next step? Show it on my provided screenshot.

[Figure 10]

Answer: Right-click on the “Midjourney Bot"

[Figure 11]

User AssistGPT

|[Figure 12]<br><br>[Figure 13]|
|---|

visual-0

[Figure 14]

Planner

[Figure 15]

|visual-0: a 627 seconds video, …, Midjourney Beginner Tutorial<br>visual-1: an image, …, the chat window with a painting on it.<br>|
|---|

[Figure 16]

LLM plans what tool

to use

[Figure 17]

Thought: Locate the video clip regarding adding Midjourney to one's personal server.

[Figure 18]

Executor

|Action: subtitle_ground("When was the video discussing adding Midjourney to a server?", visual[0])<br><br>|
|---|
|Observation: the target clip is from 03:58 – 04:15|
|visual-2: a 17 seconds video, …, target clip for query "When was the video …?"|

Run the selected tool

[Figure 19]

[Figure 20]

Object Detection

[Figure 21]

[Figure 22]

Video Narration

|[Figure 23]<br><br>[Figure 24]<br><br>visual-2|
|---|

[Figure 25]

[Figure 26]

Subtitle Grounding

[Figure 27]

Thought: Infer what is the next step after

selecting the 'show member' icon?

[Figure 28]

Region Grounding

|Action: subtitle_reason("What is the next step after selecting the 'show member' icon?", visual[2])<br><br>|
|---|
|Observation: Right-click on the "mid-journey bot"|

[Figure 29]

[Figure 30]

OCR Detection

[Figure 31]

ASR …

[Figure 32]

Thought: Find "mid-journey bot" on screenshot

|Action: text_ground("mid-journey bot", visual[1])<br><br>|
|---|
|Observation:the region is found|
|visual-3: an image, …, target region for query "mid-journey bot"|

[Figure 33]

Inspector

[Figure 34]

Generate brief summary of visual inputs and intermediate results

|[Figure 35]<br><br>[Figure 36]<br><br>visual-3|
|---|

Learner Thought: I know the final answer. Final Answer: (Right-click on the "mid-journey bot", visual[3])

[Figure 37]

Check the reasoning path to retry or save successful tries

|It is a successful try, save the whole reasoning process as an in-context example|
|---|

[Figure 38]

Figure 1: In-the-wild example of AssistGPT. AssistGPT can reason in an interleaved language and code format. Given a query input and visual inputs, AssistGPT plans the problem-solving path in language, using structured code to call upon various powerful tools. The Inspector, part of the system, can manage visual inputs and intermediate results, assisting the Planner to invoke tools. Meanwhile, the Learner can assess the reasoning process and collect in-context examples.

visual scenarios such as a long-form video with complicated scene switching, as shown in Fig. 1, the generated texts may go well beyond the query requirements. This can lead to an abundance of superfluous information while crucial details relevant to the query may be omitted.

Some other concurrent works propose decomposing user queries into subtasks and plan to sequentially call external models or APIs to answer them. Currently, there are two branches of methods. The first one is language-based planning [17–20]. For instance, HuggingGPT and Chameleon [17, 19] propose using an LLM as a controller, managing and organizing the cooperation of expert models. Another branch of work is code-based planning [21–23]. ViperGPT [21] proposes to use Codex to write the Python code to call visual-related APIs for handling multi-modal tasks. These approaches allow for invoking the models only when necessary, which allows models to only output useful information and optimize the use of computation resources.

Despite this progress, addressing high-level queries is still challenging. Specifically, current questions in existing benchmarks usually directly imply how to plan the reasoning. For example, for questions like "What is the red object used for?", no matter what the image is, the reasoning steps are relatively fixed, i.e., recognize the red object, then figure out its function. However, for more complex questions, there could be diverse reason paths. For example, for question “How much black pepper should I use for 700g beef?” in Fig. 2, the variations in the presentation of relevant information, whether it’s in the form of subtitles, actions, text within videos, or a combination of these, can result in distinct reasoning paths. Therefore, as shown in Fig. 2, once a reason-only approach makes a mistake, it becomes difficult for it to self-correct.

Similar approaches are already proposed in the NLP field, such as ReAct [24] and ToolFormer [25]. However, there is a unique challenge in multimodal tasks: How to handle non-textual intermediate

|video_segment = VideoSegment(video) for frame in video_segment.frame_iterator():<br><br>if frame.exists("beef") and \ frame.exists("black pepper"):<br><br>return frame.simple_query("How much black pepper should I use for 700g beef?")<br><br>code = generate_code(question)<br><br>Code:<br><br>Code Execution:<br><br>answer = execute_code(code, video)<br><br>Parse Question:<br><br>Risk: When introducing the measurements, beef and pepper may not be appeared together.<br><br>Reason-only with Code|
|---|

|[Figure 39]<br><br>Question: How much black pepper should I use for 700g beef?<br><br>Inputs<br><br>|
|---|

|Thought: … use Subtitle Ground Module to …<br><br>visual-0: a 197 seconds video, userprovided, sparse subtitle, How to Cook Perfect Beef Stir Fry Every Time<br><br>subtitle_ground("black pepper for beef", visual[0])<br><br>Observation: Clip Not Found. Thought: I need to know more information..<br><br>video_narration("Describe it", visual[0]) Observation: … 3-6: a person cutting a piece of beef on a cutting board. some texts on the frame (Beef 350 g Taste) …<br><br>narration_reason("how much ...", visual[0])<br><br>Thought: I need to infer how much black pepper should I use for 700g beef …<br><br>Observation: 1 tsp<br><br>ReAct with PEIL|
|---|

|Thought: I need to see if the video gives specific measurements for using black pepper on beef. Acton: Subtitle Ground Action Input: black pepper for beef<br><br>Thought: I need to infer the amount of black pepper for 700g beef based on the information from the video. Acton: Subtitle Reason Action Input: how much black pepper should I use for 700g beef?<br><br>Reason-only with Language<br><br>Risk: the subtitles may not provide relevant context.|
|---|

Figure 2: Comparison of PEIL and two mainstream reasoning methods in multi-modal tasks.

result? For ReAct and ToolFormer, the outputs of external models can be directly fed into the Planner and passed to subsequent models. While the intermediate results obtained in multimodal tasks usually are cropped regions for the image grounding module, and segmented video clips for the temporal location module, as shown in Fig. 1. In complex cases, it is hard for Planner to manage which information should be fed into the next module.

In this paper, we propose a multi-modal AI Assistant system, named AssistGPT (The design of our model’s icon is inspired by the HAL 9000 from the movie “A Space Odyssey”, a fictional artificial intelligence character), with interleaved language and code reasoning method, inheriting the advantages of flexible reasoning in ReAct and robust tool invocation in Program-based planning. Specifically, our system consists of four parts, Planner, Executor, Inspector, and Learner. We show how our system works in Fig. 1. Similar to ReAct, the Planner thinks about what needs to be done next based on the current reasoning progress and invoking external models. What sets our method apart is the use of formatted code to invoke external models. The Executor wraps external tools into a uniform input and output format, allowing the tool to be invoked with structural commands. Simultaneously, we have also proposed an Inspector, which manages visual inputs and intermediate results during the reasoning process. It provides the Planner with summaries and metadata of all currently available visual materials. The combination of the Inspector and the Executor allows the model to efficiently implement complex reasoning. Moreover, it is challenging for the model to ensure correct reasoning in a zero-shot scenario. The Planner might output invalid code or unreasonable paths. To enable the system to continuously improve, we proposed the Learner, which checks whether the prediction process is reasonable or judges the correctness of the predicted results based on annotations. It allows the system to try multiple times and record successful examples as in-context examples.

The current version of AssistGPT integrates 10+ tools for different functions, including image detection, captioning, region grounding, temporal grounding, OCR Module, object enumeration, speech-to-text, etc. By combining these functionalities, AssistGPT can accomplish a wide range of multi-modal tasks which are still hard for existing systems.

In summary, our contributions are as follows: 1) We constructed a general multimodal AI assistant that can accomplish diverse visual-related tasks with the cooperation of multiple models. 2) We propose a new compositional reasoning method that reasons over in an interleaved language and code manner. A simple learning mechanism is also proposed to improve the AssistGPT’s ability in planning. 3) We showcase AssistGPT’s capabilities not only the benchmark results but also some realistic applications for processing complex images and long-form videos, understanding high-level queries, and handling flexible inputs.

### 2 Related Work

Multi-modal Systems. Prior to the advent of LLM, remarkable works were done to design multimodal models for one or several specific tasks, such as focusing on visual appearance [26–31], visualrelated knowledge [32–37], action [38–40], ego-centric videos [41–44], instructional videos [45–47], scene text [48–51], etc. They have achieved commendable results in specific tasks, however, their

generalizability is relatively limited, making it challenging to address more complex and diverse questions in real-world scenarios.

Recently, two types of strategies are proposed for developing a general multi-modal system. One is pre-training LLM to support visual features as conditional inputs. The representative models are GPT4 [52], PaLM-E [53], BLIP-2 [54], and Mini-GPT4 [55]. Despite these methods being capable of directly processing multi-modal input, they still exhibit limitations in addressing advanced functional needs, such as image spatial grounding, long-form video grounding, and audio comprehension. Additionally, the computational cost of scaling these models can be extremely high. The alternative strategy aims to combine multiple models or APIs to accomplish complex multi-modal reasoning. For instance, models like the Socratic model [6] and Visual ChatGPT [8] achieve this by connecting ChatGPT with image generation models. HuggingGPT [17] combines a variety of Huggingface models with LLMs. ViperGPT [21] employs Codex [56] to call visual APIs via Python programming. Our AssistGPT falls into the second category by combining and invoking various modules for multi-modal reasoning, but we propose a new framework PEIL for integrating external tools and models.

Compositional Reasoning. Compositional reasoning methods in the field of visual question answering usually decompose questions into several subtasks, each addressed by a specific module. This kind of method offers strong interpretability due to its modular structure and the clear division of responsibilities among the individual components. This idea was initially put forward by [57]. Subsequently, [58, 59] introduced an end-to-end variant based on LSTM and CNN. Traditional compositional reasoning methods are limited by language models’ parsing capabilities, often requiring ground-truth question decomposition or reinforcement learning for optimal module usage.

With the advent of LLMs, question decomposition can be accomplished remarkably well in a zeroshot manner. Chain-of-thought prompts [60], Toolformer [25], and ReAct [24] enable models to plan how to solve an NLP problem. HuggingGPT [17] and ViperGPT [21] are multi-modal systems that use LLM to parse a question into a series of reasoning steps. However, for complex queries, the model needs to determine the subsequent steps based on not only questions but also visual inputs or feedback from previously executed modules. MMReAct [61] introduced the idea of ReAct to a multi-modal system to overcome it, while it is still under development and hasn’t demonstrated its effectiveness on the benchmark. Previous methods reason over either language reasoning or code, and as stated in the introduction, both have certain shortcomings. Our work first proposes an interleaved language and code reasoning manner which can better handle general queries and complex visual inputs.

Learning Schemes for Modular System. Early modular models primarily employed end-to-end Reinforcement Learning (RL) to train each module’s planning and acting from scratch. While this approach is practical for lightweight models, RL can introduce substantial overhead for systems where each module is an LLM. Toolformer [25] proposes a self-supervised technique that optimizes planning requiring only a handful of demonstrations for each API. Specifically, Toolformer attempts various APIs to find successful examples and then fine-tunes the model. In contrast, we propose a straightforward mechanism in the multi-modal field, which can guide the system to retry and preserve the successful explorations as in-context examples.

### 3 AssistGPT

Overview. AssistGPT is a general multi-modal AI assistant system that can dynamically engage various tools in an interleaved language and code manner. Specifically, given a general language query and reference images or videos as inputs, the goal of AssistGPT is to generate the desired answer. As shown in Fig. 3, AssistGPT is achieved by cooperation with four core modules: (a) Planner, (b) Executor, (c) Inspector, and (d) Learner. The Planner § 3.1 aims to control the whole reasoning process, with the Executor § 3.2 supplying valuable feedback to Planner by executing external tools. The Inspector § 3.3 manages the input and intermediate results and assists the Planner in feeding proper content to the Executor. The Learner § 3.4 is capable of assessing the system performance and record successful explorations as in-context examples. In the following sections, we will go through each module in detail.

[Figure 40]

[Figure 41]

[Figure 42]

###### Planner

###### Inspector

|1. Validation Check yes<br><br>Executor<br><br>Error Message Generation: <module> only takes video as input.<br><br>no<br><br>2. Module Execution:<br><br>[Figure 43]<br><br>3. Post processing: Crop image / Segment Video<br><br>[Figure 44]<br><br>[Figure 45]<br><br>BLIP<br><br>Grounding DINO<br><br>[Figure 46]<br><br>[Figure 47]<br><br>Observation: …<br><br>…<br><br>[Figure 48]<br><br>module_name(arg1, arg2)<br><br>|
|---|

[Figure 49]

Input Videos & Images

Instruction Prompt: Answer the following question as best as you can. [Tool Set Illustration] Text Detection Module: Used for detect the text [In-Context Example] Question: …. Thought: … Acton: … Observation:…

[Figure 50]

User

Processed Videos

[Figure 51]

❓ &Images

User

Input Query: When was the video … ?

###### Image/Keyframe Captioner and Metadata Tool

[Figure 52]

Executor

Summary of Input Videos and Images: visual-0, a 48.27 seconds video, …

[Figure 53]

Inspector

Attribute Value type video

[Figure 54]

source user-provided duration 17 summary [caption] … …

Thought: I need to find … Action:

module_name(arg1, arg2)

[Figure 55]

Executor

Observation: Output of a tool Summary: visual-1: a 17 seconds video, …

[Figure 56]

Inspector

Summary of Metadata: visual-1: a 17 seconds video, segmented video from visual-0, dense subtitle, target clip for query "When was the video …

[… repeat above 𝑁𝑁 times until getting final answer] Thought: I know the final answer Final Answer: …

[Figure 57]

✅User

[Figure 58]

|video_narration("Describe what happens near the end of the video.", visual[0])<br><br>Summary: visual-0, a 48.27 seconds video, sparse subtitle, user provided video, a toy train on the floor …<br><br>narration_reasoning("why did the train stopped a while near the end of the video?", visual[0])<br><br>Final Answer: no more space to push cart x<br><br>1-st Try<br><br>1. Self-Check or GT Comparison<br><br>video_ground("Locate the end of the video.", visual[0])<br><br>video_narration("Describe it.", visual[1])<br><br>Final Answer: blocked by boy’s hand<br><br>narration_reasoning("why did the train stopped a while near the end of the video?", visual[1])<br><br>visual-1: a 12 seconds video, sparse subtitle, segmented video from visual-0, target clip for query “Locate the end …<br><br>2. In-Context Memory Bank: Save as in-context example for better planning and module prompts<br><br>Learner<br><br>Input Query: why did the train stopped a while near the end of the video?<br><br>2-nd Try<br><br>if fail<br><br>if success<br><br>evaluate<br><br>evaluate<br><br>✅|
|---|

#### Figure 3: Diagrammatic illustration of AssistGPT system. It consists of four core modules:

● Planner: control the whole reasoning process; ● Executor: execute external tool and return feedback to Planner; ● Inspector: manage the input and intermediate outcomes; ● Learner: assess the system performance and record successful trials as in-context examples.

#### 3.1 Planner

The ● Planner employs a highly intelligent LLM i.e., GPT-4 [52] as the central brain to control the global reasoning planning. It begins the planning process by taking inputs from three types of information: an Instruction Prompt consisting of the [Tool Set Illustration] and [In-Context

Example]2, Input Query , and the Summary of Visual Inputs created by ● Inspector.

Then it generates the appropriate output for the next step, which consist of two parts: Thought: a language phrase indicates what should be done next. While it doesn’t affect the module or API call directly, it aids the LLM planning procedure. Action: a structural string obeys the pre-defined template provided in the instructions. It specifies which external tool to call and what arguments to input, e.g., caption("what color is the car?", visual[0]) .

After each time ● Executor call to an external tool, the tool returns outputs in the form of natural language, which we refer to as Observation . If the tool generates an intermediate outcome, e.g., a segmented video, our ● Inspector will store it and generate a Summary for it. Both the

Observation and Summary will be fed to the ● Planner to guide the planning of the next step. The following sections will introduce more details of Action, Observation , and Summary . Currently, we integrate 13 functional tools in AssistGPT to power multi-modal assistance, as shown in Tab. 1. These modules can be mainly categorized into three types:

• Descriptor: To effectively comprehend and utilize the data derived from intricate multimodal environments e.g., image, video, audio, and text, we employ a variety of fundamental models as basic descriptors for perception. These models, including (a) Image Caption, (b) Video Narration, (c) Object Detection, (d) Text Detection, and (e) ASR Translation, enable us to extract enough information from diverse source, thus enhancing our understanding of the multimodal sceanrio.

2The successful trials recorded by Learner, will be introduced later.

- Table 1: Module used in AssistGPT. A module may have different models, separated by a slash (/).

Module Usage Core Model Input Output Module Usage Core Model Input Output

- (a) Image Caption BLIP series [9–11] T, I T (h) Text Ground Program + SSA [62, 63] T, I T, I
- (b) Video Narration BLIP series [9–11] T, V T (i) Subtitle Ground GPT [5] T, Sub. T, V
- (c) Object Detection G. Dino [64] / GLIP [13] T, I T (j) Knowledge Reason GPT [5] T T
- (d) Text Detection Google OCR I T (k) Narration Reason GPT [5] T, Nar. T
- (e) ASR Translation Whisper [65] A T (l) Subtitle Reason GPT [5] T, Sub. T
- (f) Region Ground OFA [66] T, I T, I (m) Temporal Reason Rule-based T, V T, V

- (g) Narration Ground GPT / CLIP [67] T, Nar. T, V I: Image, V: Video, T: Text, A: Audio, Nar.: Narration, Sub.: Subtitle

- • Locator: As the saying goes, a picture is worth a thousand words. Images or videos typically contain a wealth of information - objects, attributes, actions, events, and so on. However, the abundance of information can sometimes obstruct our problem-solving process. One crucial solution is to pinpoint the most crucial and valuable information from the rich sea of visual, textual, and audio data This part incorporates several modules such as the (f) Region Ground, (g) Narration Ground, (h) Text Ground, and (i) Subtitle Ground.
- • Reasoner: The initial two sets of tools primarily deal with collect and identify of data, whereas the third set focuses on reasoning, utilizing the extracted information and external knowledge. This part incorporates modules such as (j) Knowledge Reason, (k) Narration Reason, (l) Subtitle reason, and (m) Temporal Reason modules. These modules primarily utilize LLM at their core by taking different types of information and prompts as inputs or a simple program.

#### 3.2 Executor

The ● Executor takes the code generated by the ● Planner as input, then call a module to produce the output by carrying out three steps to obtain the final result. These steps include validation check, module execution, and post-processing, as shown in Fig. 3.

- • Validation Check: Even powerful LLM like GPT-4 can sometimes generate illegal code. For example, an image caption module accept a long video as input. We have designed a legality check for each module to determine whether the code is executable. Moreover, if code includes errors, we do not interrupt the entire reasoning process. Instead, we return an error message as the output code to the ● Planner, allowing it to optimize the planning process in real-time.
- • Module Execution: We standard various modules or APIs into a unified interface using the code-style template i.e., [Module_Name](<text_query>, <visual_index>). Each module is designed to accept multiple text queries and visual data (images or videos) as input. In each standarded module, we provide instructions on its function and the requirements of the argument, which is used for [Tool Set Illustration] in ● Planner. Additionally, for the sake of simplicity and accuracy in Planning, the generated code is simplified. Later, a simple rule-based function will map it to the executable codes and then execute it to obtain the final result.
- • Post-processing: For all modules, the generated results will be translated into a language format to inform the ● Planner about the outcome, as the Observation part illustrated above. For instance, for the Narration Ground module, the model will return whether it has found the relevant segment. If so, output the start and end times of the segment. Additionally, many Ground-related modules will send their segmented video or cropped image region to the subsequent visual outcome manager i.e., ● Inspector.

#### 3.3 Inspector

The objective of the ● Inspector is to manage the visual inputs provided by the user and the intermediary results produced by our system to assist the ● Planner in deciding which source should be directed to which module. Specifically, the ● Inspector records the metadata of each visual element, which includes its type (image or video), source (provided by the user or generated by the system), and a brief description of the content (obtained from the caption model, or the title of an online video). For videos, there is some additional metadata, such as the duration of the video, whether it contains audio and subtitles. The ● Inspector monitors the inputs from the user and the outputs from the ● Executor. As soon as a new visual element is received, it appends the metadata, noted as Summary in the above, to the reasoning history of the ● Planner. With the cooperation of

the ● Planner, ● Executor, and ● Inspector, our system can generate answers to difficult queries with complex visual inputs.

#### 3.4 Learner

Despite the robot generalization capabilities of LLMs, they can still easily encounter errors when dealing with multi-modal type queries. Thus, it is essential for an AI assistant to have self-valuate mechanism. To achieve this goal, we hope that the model can self-check the reasonableness of its output. On the other hand, when ground truth is available, we intend to gather successful prediction instances as in-context examples. Specifically, AssistGPT will repeatedly attempt to provide the answer when the response is not satisfactory until either passes the self-check, or correct answer is given (when ground truth is available) or a predefined maximum number of attempts is reached. ● Learner includes an evaluator implemented by the LLM, which operates in two modes: selfassessment and ground-truth comparison. These modes are activated depending on the availability of ground truth, and we discuss the two of them separately.

- • Self-assessment mode is activated when there is no user feedback or ground truth available. It takes the reasoning trace and the results of each step as input, allowing GPT to assess whether the reasoning is complete, consistent, and adhere to the required format.
- • Ground-truth comparison mode is activated when annotators provide ground truth. In this mode, GPT evaluates whether the AssistGPT’s prediction is semantically consistent with the provided ground truth.

Furthermore, ● Learner encourages to keep trying until it receives positive feedback or reachs the maximum number of attempts. After conducting N times explorations, several outcomes may arise:

- • No adjustments required: If the model delivers the correct answer on its initial attempt, this suggests that AssistGPT can well-solve the current question effectively. Therefore, no improvement is required.
- • Plan Revision: If the model produces the correct answer after making n attempts, where 1 < n ≤ N, this implies that there is room for improving the model’s planning capabilities. Therefore, we save the successful reasoning trace to [In-Context Memory Bank]. Consequently, when the model comes across a similar query in the future, it can use this as an in-context example.
- • Function Updates: If the model still fails to provide the correct answer even after N attempts, it is highly probable that the problem resides in a specific module or API rather than the planning process. It may necessitate incremental updates to the module. We will leave this for future work.

### 4 Experiments

#### 4.1 Experimental Setting

Datasets. Our system is evaluated on A-OKVQA [33] and NExT-QA [68] benchmarks designed to test comprehensive multimodal capabilities, including visual facts, commonsense, temporal sequences, causality, etc. A-OKVQA [33] is an innovative benchmark for knowledge-aware visual question answering with 25K questions that demand a high-level comprehension of commonsense and world knowledge. These questions in A-OKVQA go beyond the information contained in the image and cannot be answered solely by querying a knowledge base. Besides, the question is diverse, spanning a wide range of domains such as commonsense reasoning, visually-grounded, knowledge-based, and physical understanding. In our experiments, we assess the model performance under the incontext learning setting on the validation set, which consists of 1,145 questions. NExT-QA [68] is a benchmark for evaluating the AI system’s causal reasoning, temporal action reasoning, and rich object interactions in video question answering. NExT-QA has a total of 5,440 videos, with averaging 44 seconds in length, and approximately 52K manually annotated question-answer pairs. In our experiments, we assess the model performance under the in-context learning setting on the validation set, which consists of 4,996 questions.

Implementation Details. In the following experiments, we use GPT-4 API provided by OpenAI [52] as Planner. In the A-OKVQA experiments, we set Caption Module as BLIP2 or InstructBLIP (abbreviated as Ins.BLIP), use the Gounding Dino for the Object Detection model, and Google

OCR for Text Detection. For the NExT-QA experiments, our Video Narration Module is based on InstructBLIP Vicuna-7B [11]. Our experiments are performed on 4 A5000 GPUs.

#### 4.2 Quantitative Results

- Table 2: Comparison of SOTAs on A-OKVQA dataset. D.A. and M.C. indicate direct answer and multi-choice. ICL: In-context Learning. ZS: Zero-shot inference.

Table 3: Ablation study of our AssistGPT on A-OKVQA dataset. Ins.BLIP used here is the pre-trained version.

Model D. A. M.C.

LLM Model D.A. M.C.

LXMERT [29] 30.7 51.4 KRISP [36] 33.7 51.9 GPV-2 [69] 48.6 60.3 InstructBLIP Vicuna-7B [11] 64.0 75.7

- Ins.BLIP 13.4 53.8 Ins.BLIP + GPT-4 27.9 55.2

Sup.

Reason only 28.8 65.9 ReAct 30.1 68.2 PIE 32.4 72.4 PEIL w. Self-Check 41.2 74.2 PEIL w. GT-Check 44.3 74.7

AssistGPT

PromptCap [70] 56.3 73.2 AssistGPT (BLIP2 FlanT5XL [54]) 42.6 73.7 AssistGPT (Ins.BLIP Vicuna-7B) 44.3 74.7

ICL

Comparison with State-of-the-arts. From the results in Table 2, it can be seen that in the multichoice track, our two versions of AssistGPT (i.e., with light-weight BLIP2 FlanT5XL and more powerful Ins.BLIP Vicuna-7B) achieve the best among all current methods on in-context learning setting. It’s worth noting that we use a pre-trained version of InstructBLIP, which performs at 53.3%, as shown in Table 3. When integrated into our system, it can enhance its performance to the level of fine-tuning model. For direct answer questions, while our performance on it may not match that of recently proposed models, it is still comparable to previous supervised SOTA, like GPV-2 [69].

Our performance on direct answers did not surpass previous methods. The main reason is that for open-ended questions, models relying on LLM tend to output complete phrases rather than a single word as the final answer, even when we prompt them to provide as concise an answer as possible. For instance, for a given question "What flag is represented on the wall?", AssistGPT outputted the answer, "United States flag", but the correct answer does not include the word "flag", therefore it’s deemed incorrect. This type of error is very common in AssistGPT. In the appendix, we show more examples to analyze the failure cases. Moreover, compared to the SOTA method, PromptCap [70], it specifically trained a caption model toward generating captions for A-OKVQA, which is also the reason for its good performance, while our system is more general.

From the results in Table 4, AssistGPT achieved higher performance than recently proposed supervised methods, demonstrating the effectiveness of our approach. We can see that our model’s performance mainly shows a more promising improvement in Causal and Descriptive questions, mainly due to our model continuously obtaining detailed information related to the question from the videos. Moreover, our method does not perform well on temporal questions. The main reason for this is that there are relatively few open-world temporal grounding models available, and mainstream work still involves fine-tuning on closed-world datasets. Therefore, we have to use the image captioner InstructBLIP with GPT-4 to achieve temporal grounding. The effect is not as good as that of fine-tuned models but has a more generalization ability. Furthermore, its performance is also very close to recent concurrent work, ViperGPT [21]. The ViperGPT is a little bit superior to ours, possibly because it has designed a sophisticated rule-based method, iteratively checking whether objects appear in the frame to perform temporal grounding.

Ablation Study. We have designed several variants of AssistGPT to test the effectiveness of our proposed method. The most basic baseline is InstructBLIP (note that all following models are using Vicuna-7B version), which is the main source of visual information in AssistGPT. Since InstructionBLIP cannot necessarily output the answer in the required format, we design a variant, InstructionBLIP+GPT-4 allows GPT-4 to further refine the output of InstructionBLIP. The Reasononly model directly plans all the steps the models need to run, similar to previous works [17]. The ReAct model is capable of executing language-based ReAct. However, without Inspector and Codelike invocation forms, a subsequent model can only accept the output of the previous model, which is similar to [61]. We also ablate the Learner, which has three versions, PIE (i.e., w/o. Learner), PEIL w. Self-Check and PEIL w. GT-Check.

From the results in Table 3, we can see that the Reason-only model, which plans all the steps the models need to execute, showed a notable improvement in D.A. and M.C. This indicates that

Table 4: Comparison of our AssistGPT with SOTAs on NExT-QA dataset.

Method Causal Temporal Descriptive All

HGA 44.22 52.49 44.07 49.74 VQA-T [71] 49.60 51.49 63.19 52.32 ATP [72] 53.10 50.20 66.80 54.30 VGT [73] 52.28 55.09 64.09 55.02 MIST [74] 54.62 56.64 66.92 57.18

Sup.

ViperGPT [21] - - - 60.00 AssistGPT 60.02 51.38 67.26 58.36

ICL

integrating multiple models can enhance model performance. The ReAct model, despite not having Inspector and Code-like invocation forms, showed a further improvement in both metrics, surpassing the Reason-only model. This suggests the effectiveness of ReAct manner. But involving our interleaved language and code, i.e., PIE, brings a more significant improvement on M.C. Finally, the two variants of PIE with partial ablations, PEIL w. Self-Check and PEIL w. GT-Check, scored the highest on both tracks, showing the effectiveness of the Learner. The Learner shows a more significant improvement on D.A. tracks because models on D.A. often fail to output extremely short answers as required by A-OKVQA. The Learner can mitigate it by collecting in-context examples.

#### 4.3 Qualitative Results

[Figure 59]

Question1 : What kind of room is on the left of the yellow train? You must choose one answer from: nursery room, bathroom, changing room, waiting room

visual-0: an image, user-provided, a yellow and white train pulling into a train station

|Thought: Check the visual<br><br>information about the room on the left of the yellow train. Action:<br><br>|caption("What kind of room is on the left of the yellow train?", visual[0])|
|---|
|Observation: There is no room on the left of the yellow train.<br><br>|
<br><br>Thought: The observation contradicts the visual description. I will try to locate the room in the image. Action:<br><br>|region_ground("room on the left of the yellow train", visual[0])|
|---|
|Observation: region found<br><br>|
|visual-1: an image, cropped …|
<br><br>Thought: Now that I have the cropped image of the room, I can check its type.<br><br>Action:<br><br>|caption("What kind of room is this?", visual[1])|
|---|
|Observation: Train station<br><br>|
<br><br>Thought: Now that I have the cropped image of the room, I can check its type. Action:<br><br>|caption("Is this a nursery room? Is this a bathroom? …",<br><br>visual[1])|
|---|
|Observation: The answers for each question are: no, no, no, no<br><br>|
<br><br>Thought: None of the given options match the observation. Final Answer: None of the given<br><br>options match the room.<br><br>AssistGPT: First Try|
|---|

||caption("Please provide more detailed information about this room?", visual[1])|
|---|
|Observation: The image depicts a group of people gathered at a train station, waiting for a train to arrive.<br><br>|
<br><br>AssistGPT: Successful Try<br><br>|knowledge_reason("Based on the detailed information about the room, which of the following options is the most likely match: nursery room, bathroom, changing room, waiting room??", visual[1])|
|---|
|Observation: waiting room<br><br>|
<br><br>|region_ground("room on the left of the yellow train", visual[0])|
|---|
<br><br>Thought: I know the final answer. Final Answer: waiting room<br><br>|
|---|

|Reason-only Baseline<br><br>[Figure 60]<br><br>|Describe the image in detail: The image depicts a busy train station, with several people standing on the platform waiting for their trains. There is a yellow and white train parked on one of the tracks, with passengers boarding and disembarking from the train. In the background, there are several other …. Answer the question based on the above information of the image: Question: … You must choose one answer from: …|
|---|
<br><br>[Figure 61]<br><br>InstructBLIP:<br><br>The information provided about the image does not specify what kind of room is on the left of the yellow train. Thus, it's impossible to definitively answer this question based on the given details.<br><br>GPT-4:<br><br>|
|---|

|Thought: Locate the final moment in the video. Action:<br><br>|temporal_reason("end", visual[0])|
|---|
|Observation: The target clip is from 02:05-02:21.<br><br>|
|visual-1: a 16 seconds video, cropped …|
<br><br>[Figure 62]<br><br>Thought: I need to know more information about the video. Action:<br><br>|Video_narration("did the baby<br><br>play with the toy? was the baby attracted by the chair? did the baby look at the person filming? did the baby trip? did the baby reading from paper?", visual[1])|
|---|
|Observation: The answers for each question are: yes, no, yes, no, no<br><br>|
<br><br>Thought: I need to infer why<br><br>does the baby stop for a while near the end of the video?<br><br>Action:<br><br>|knowledg_reason("why does the baby stop for a while near the end of the video? …", visual[1])|
|---|
|Observation: to look at the person filming<br><br>|
<br><br>Thought: I know the final answer. Final Answer: to look at the person filming<br><br>|
|---|

[Figure 63]

Question2 : why does the baby stop for a while near the end of the video? You must …

visual-0: a 81 seconds video, dense subtitle, user provided video, a little girl riding a tricycle on a

sidewalk

Figure 4: Qualitative results on A-OKVQA (Question 1) and NExT-QA dataset (Question 2).

In Fig. 4, we visualize some prediction cases from A-OKVQA (Question 1) and NExT-QA (Question 2). From both examples, it can be seen that AssistGPT can decompose the question into reasonable sub-tasks and then complete them step by step, ultimately obtaining the final answer. Moreover, due to the interleaved code and language reasoning method, the model can effectively invoke the necessary content as input. From the reasoning process of Question 1, we can also see AssistGPT’s selfcorrection ability. When the visual model output unsatisfactory results, AssistGPT can dynamically invoke other modules, like the ground module to reason over another path. In addition, for Question

1, the model’s first attempt did not yield effective results, and it will autonomously optimize the plan because it did not pass the self-check. In addition, we also present the result of the reason-only baseline. It first calls InstructBLIP to output a caption, then uses GPT-4 for inference. Since the information from the caption does not meet the requirements, resulting incorrect results. However, once the prediction fails, the model does not have a way to self-optimize. It’s worth mentioning that the most significant feature of our method is that it can solve more complex problems than those in the benchmark, as the example in Fig. 1. We show more in-the-wild examples in the Appendix.

### 5 Conclusions and Limitations

In this paper, we propose a novel multi-modal assistant system named AssistGPT that leverages an interleaved code and language reasoning approach, namely Plan, Execute, Inspect, and Learn (PEIL). This innovative system integrates LLM with various tools to address the challenges posed by complex visual-based tasks. Our experimental results on A-OKVQA and NExT-QA benchmarks demonstrate AssistGPT’s effectiveness. Furthermore, we showcase our system’s ability in handling diverse and intricate real-world scenarios. Our system also has some limitations. Our approach does not propose an end-to-end updating solution, which is crucial when the tools used make mistakes. Another limitation is that the planning process requires an extensive explanation of tools, resulting in a relatively large overhead, which could be improved by distilling a smaller size planner.

### Appendix

In the appendix, we provide additional details for the main paper:

- • More discussion with existing modular systems in Sec. A
- • More details of AssistGPT in Sec. B.
- • More qualitative results of A-OKVQA in Sec. C.
- • More in-the-wild examples in Sec. D.

### A Discussion with Existing LLM-driven Modular Systems

Table 5: Comparison of existing LLM-driven modular systems. We compare existing methods from four dimensions: Task Focus, Reasoning Method, Source Management (how they manage input and intermediate results), and whether they have learning capabilities. The term "ReAct" in the table does not strictly refer to using the ReAct [24], but rather it denotes planning and executing concurrently.

Task Focus Reasoning Source Management

Model

Learning NLP Image Video Format ReAct Input format Method

Toolformer [25] ✓ ✗ ✗ lang. & prog. ✓ text-only - ✓ WebGPT [75] ✓ ✗ ✗ program ✓ test-only - ✓

Visual ChatGPT [8] ✗ ✓ ✗ language ✗ multi. V. Filename ✗ ViperGPT [21] ✗ ✓ ✓ program ✗ single V. Variable ✗ VisProg [22] ✗ ✓ ✗ program ✗ single V. Variable ✗ MM-ReAct [61] ✗ ✓ ✓ language ✓ multi V. Filename ✗ Chameleon [19] ✓ ✓ ✗ language ✗ single V. Cache update ✗ HuggingGPT [17] ✗ ✓ ✓ language ✗ multi V. Filename ✗

AssistGPT (ours) ✗ ✓ ✓ lang. & prog. ✓ multi V. Inspector ✓

- In Table 5, we compare the existing LLM-driven modular systems with our AssistGPT from four perspectives:

Task Focus. From the perspective of Task Focus, there are currently three works that can handle videos: Hugging GPT [17], MM-ReAct [61], and ViperGPT [21]. Hugging GPT and MM-ReAct merely demonstrate their capabilities in handling videos through a few simple examples (thus we mark them with orange checkmarks ✓). For instance, Hugging GPT exhibits its video generation feature, while MM-ReAct showcases its ability to perform tasks such as summarization and localization based

on subtitles. However, these methods have not been validated on any benchmark. ViperGPT can handle questions based on visual content. Compared to these works, AssistGPT is capable of dealing with more complex and general video question-answering tasks, including understanding subtitles, visual content, and OCR, and demonstrating long video comprehension capabilities.

Reasoning. In terms of reasoning, existing Multi-modal models primarily adopt a reason-only style, that is, directly deriving the solution steps based on the question. This approach struggles with handling complex visual inputs, and when the intermediate results don’t meet expectations, the model also finds it hard to self-correct. MM-ReAct introduces the original ReAct for reasoning in Multi-modal tasks, but due to the original ReAct’s inadequacy in dealing with complex non-text intermediate results, its current planning scheme for addressing video-related issues is basically two steps: extracting all information from the video and then having an LLM answer the question. In contrast, this paper proposes a more general Plan, Execute, Inspect, and Learn (PEIL) reasoning scheme. In the case of complex videos, our interleaved language and code reasoning approach allows for flexible language planning for the next step, and structured code for invoking input and intermediate results, thereby facilitating the handling of complex questions and visual content.

Source Management. Handling complex input and a large number of intermediate results is often crucial in complex reasoning processes. Current language-based reasoning methods mainly use filenames to label resources. Chameleon proposes an update mechanism with a cache that constantly updates the current reasoning results. Program-based reasoning, on the other hand, uses variables to store intermediate results. A deficiency of these methods is the inability of the language-based Planner to quickly comprehend the content of visual sources, which impedes the effective use of different sources to complete different subtasks. As a result, existing work struggles to handle flexible input and intermediate results. Even though some work supports multiple visual sources as input, they are more often batch-processed for similar tasks, with each source requiring similar operations. For instance, in HuggingGPT, the task of calculating the sum of the number of zebras in several images involves counting the number of zebras in each image. In contrast, our work introduces the Inspector, which records the metadata and summary of each visual source and provides it to the Planner for reasoning. This design can support complex input. For example, a user view image that describes the current user’s problem, and a reference video as a source of knowledge, AssistGPT can then use these two different types of sources to jointly answer the user’s question.

Learning. Most multi-modal modular systems lack the capability for continuous optimization. This paper proposes a simple update mechanism that allows the model to self-check the reasonableness of its output and ultimately continues to collect in-context learning examples.

### B More details of AssistGPT

#### Table 6: Invoke Commands and Illustrations to the Modules in AssistGPT.

###### Module Invoke Command Illustration

- (a) Image Caption caption(query, visual[i]) extract the visual information in an image.

- (b) Video Narration video_narration(query, visual[i]) output narration based on video’s visual information.

- (c) Object Detection object_detect(query, visual[i]) detect required objects in an image.

- (d) Text Detection text_detect(None, visual[i]) extract the OCR in an image.

- (e) ASR Translation asr(None, visual[i]) transcribe audio to text.

- (f) Region Ground region_ground(query, visual[i]) locate the queried region in an image.

- (g) Narration Ground narration_ground(query, visual[i]) find the clip based on the narration of a video.

- (h) Text Ground text_ground(query, visual[i]) find the location of a specific text in an image.

- (i) Subtitle Ground subtitle_ground(query, visual[i]) find the clip based on the subtitle of a video.

- (j) Knowledge Reason knowledge_reason(query, []) infer the answer based on the commonsense.

- (k) Narration Reason narration_reason(query, visual[i]) infer the answer based on narration of a video.

- (l) Subtitle Reason subtitle_reason(query, visual[i]) infer the answer based on subtitle of a video.

- (m) Temporal Reason temporal_reason(query, visual[i]) find the clip based on temporal relationship words.

- In Table 6, we show the invoke commands and illustration of each module in AssistGPT. We provide more details of how each module is implemented.

- • Image Caption: The core model of this module is a text-conditioned captioning model, e.g., BLIP2 [54], InstructBLIP [11], similar to an open-ended Visual Question Answering model.

- • Video Narration: As the general video captioning models are not yet mature, we currently use the image captioning model [54, 11] to accomplish this function. Specifically, we sample image frames (1/3 FPS for current implementation) and perform text-conditioned captioning on each frame. We employ text-conditioned captioning because, if we use dense captioning, the output text will be excessively abundant, making it difficult for subsequent models to utilize. The Video Narration feature can also optionally read the OCR content within the frames. The extracted OCR will be appended to the caption of each frame.
- • Object Detection: The main function of this module is to determine whether the image contains the objects mentioned in the query and to address counting-related questions. Thus, it contains an open-set object detection model, e.g., Grounding DINO [64], which can output the bounding boxes of relevant objects based on the query. We also let the module calculate the number of related objects.
- • Text Detection: This model is used to extract OCR from images, and the extracted text is returned to the Planner. We use Google OCR to achieve this purpose.
- • ASR Translation: This model is used to convert audio from a video into text. We use OpenAI’s open-source ASR (Automatic Speech Recognition) model, Whisper [65], to accomplish this. The detected ASR organizes timestamps and text in a manner similar to subtitles. In the implementation, we automatically run this module as soon as we receive a video with audio.
- • Region Ground: The purpose of this module is to find a specific area of an image based on the query. We use the OFA-Large [66], which is fine-tuned on RefCOCO, to achieve it.
- • Narration Ground: This model’s function is to find time segments related to the query based on the video’s narration. We propose two implementations: 1) We use GPT-4 [5], taking the video’s narration and query as prompts, to output the timestamps of the time segments. 2) Another solution is using CLIP [67] to do that. We can split the video into several segments, and calculate the similarity between the frame in each segment and query. The time stamps of the segment with the highest similarity will be outputted. In our preliminary experiments, the first solution showed better interpretability and generalization ability, so it was adopted in the benchmark evaluation.
- • Text Ground: The purpose of this model is to locate specific areas of an image that correspond to a certain text. This capability can guide users in identifying crucial information in complex, textrich images, such as user interfaces. The query format is text[:object_name], wherein text signifies the text to be located, and object_name (which is optional) is used to locate the text on a specific object, for instance, "menu: button". Specifically, the model operates in two stages:

1) Based on the Optical Character Recognition (OCR) detection results, the model identifies areas of the image that match the text segment of the query. This is achieved by calculating the distance between the query and the OCR extracted, and when the edit distance is below a particular threshold, it is considered a match. 2) If more than one textual area is identified, we further refine the results based on the object’s name. We employ the Semantic Segment Anything (SSA) [63] to segment the image semantically, identifying regions that match the object’s name mentioned in the query.

- • Subtitle Ground: This model is similar to the narration grounding model, but it uses the video’s subtitles as input instead of the narration. Thus, we also use GPT-4 to achieve it.
- • Knowledge Reason: The purpose of this model is to enable the model to apply external knowledge to answer questions. We currently do not connect to the internet to retrieve knowledge, but use the knowledge that GPT-4 has itself learned. Specifically, this model enables GPT-4 to use its own knowledge to infer the answer based on the question and results of all previous reasoning steps.
- • Narration Reason: The aim of this module is to infer some information based on the visual content of the video. This module also uses GPT-4, taking the query and the input video’s narration as prompts, to infer the answer.
- • Subtitle Reason: The aim of this module is to infer some information based on the subtitle of the video. It is similar to Narration Reason, but takes the input video’s subtitle and query as prompts, to infer the answer.
- • Temporal Reason: This model is able to find a video clip based on some temporal relation words. The input to this module follows the following format: temporal_word: time stamps, e.g., after: 3 - 6. Temporal relation words include two types, one is absolute temporal relation words, such as in the middle/beginning/end of the video. The second type is relative temporal relation words, such as before and after. For the first type of words, we divide the video into 5 segments and then

output the time stamps of the corresponding segment according to the temporal_word. For the second type, we divide the video into 8 segments, and then, according to the input time stamps, we output the time stamps of the segment before or after it. The current hyperparameters, the division of video clips, are still preliminary. It would be much better to use the model to divide them semantically, and then perform temporal reasoning in the future.

### C Qualitative Results in A-OKVQA

In Figure 5, we showcase a successful instance along with several failure examples, illustrating the most frequent error patterns in A-OKVQA. As is evident, AssistGPT can produce highly interpretable answer processes. Moreover, even in cases where the questions are answered incorrectly, there are relatively reasonable explanations provided. In the following, we illustrate the common error patterns in detail:

- • Undesired output format: For Direct Answer questions, like Q2, the results of AssistGPT are the same as the correct answers in meaning, but the expression is different, which would be considered as incorrect under the existing metrics.
- • Fine-grained recognition: The recognition of fine-grained categories of some objects is still not well done by existing visual models, resulting in the incorrect final answer. For example, AssistGPT didn’t successfully recognize cough drops in Q3.
- • Pose-to-text: Currently, there are very few models that can map the fine-grained pose or actions of people or animals to natural language. For example, capturing the upward jump action of the cat in Q4 is a challenge. AssistGPT currently does not incorporate a related model to grasp such information. Instead, it makes prediction based on the surrounding objects in relation to the cat.
- • Inconsistent reasoning: Despite AssistGPT having some self-error correction mechanisms, it occasionally exhibits inconsistencies in its reasoning process, which can lead to final inaccuracies. For instance, in Q5, the model initially identifies the orange vehicle as a truck, but in subsequent steps, it is referred to as a shuttle bus. Unfortunately, AssistGPT fails to detect this inconsistency and does not proceed to make necessary corrections.

### D In-the-wild Prediction Examples

We show some examples of AssistGPT handling in-the-wild scenarios in Figure 6 and Figure 7. From various in-the-wild examples, it’s clear that AssistGPT can adeptly handle a range of video types, be it dense, subtitled instructional videos (Q2, Q3), or those featuring rich, visual content with sporadic on-frame text (Q1, Q4, Q5). Impressively, when faced with high-level queries (Q2 and Q3), the model exhibits a capacity to strategically locate useful content, accurately identify the correct responses, and offer comprehensive, multimodal answers. A notable self-error correction capability is also evident during its reasoning process, as demonstrated in Q2. Here, the narration model was unable to generate meaningful narrations and, therefore, opted to utilize the subtitle to answer the question.

Moreover, in Q5, we highlight that our model can effectively process multiple video inputs serving different functions. This includes a User view image and a couple of reference videos. It’s important to note that our model can accommodate any number of inputs. Consequently, with the incorporation of a YouTube video search function, the model could autonomously seek out several reference videos and then cross-reference them to discern the user’s intent.

In summary, we want to emphasize that AssistGPT is a comprehensive multi-modal assistant system, capable of managing a wide array of real-world application queries that are far more complex and comprehensive than the samples provided in benchmarks.

[Figure 64]

[Figure 65]

Thought: Check the visual information about the man in the gray suit on the left. Action:

Thought: Check what the man in the gray suit on the left is looking down to check. Action:

Question1: What is the man in the gray suit on the left looking down to check? You must choose one answer from: phone, tablet, notebook, pager

|caption("What is the man is looking down to check.?", visual[0])|
|---|
|Observation: The man is looking down to check his cell phone.<br><br>|

|region_ground("man in the<br><br>gray suit on the left",<br><br>visual[0])|
|---|
|Observation: region found<br><br>|
|visual-1: an image, cropped image from visual0, main content is man in the gray suit on the left<br><br>region|

visual-0: an image, user

provided image, main content is two men in

gray suits sitting at a table

Thought: I know the final answer. Final Answer: phone

[Figure 66]

[Figure 67]

Thought: Check the visual information about the birthday number in the image. Action:

Thought: I found the number 30 in the image. It could be the birthday number Final Answer: 30 Ground truth: ['thirty’, '30th’, 'thirty’,

Question2: Which number birthday is probably being

celebrated?

visual-0: an image, user provided image, main content is a teddy bear cake on a table

|text_detect(None,visual[0])|
|---|
|Observation: Figna,Troy,30,afo<br><br>|

'thirty’, 'thirty’, '30th’, 'thirty’, 'thirty’,

'thirty’, 'thirty']

[Figure 68]

[Figure 69]

- Question3: What item on the desk could help with a cold? You must choose one answer from: cough drops, syringe, pills, herbal tea

visual-0: an image, user provided image, main content is a woman reading a book in front of a laptop

Thought: Check the visual information about the items on the desk in the image. Action:

|caption("What items are on the desk in the image?", visual[0])|
|---|
|Observation: On the desk in the image,<br><br>there is a laptop computer,<br><br>a notebook, and a pen.|

Thought: None of the given options are present in the image. I need to choose the best option based on common sense knowledge. Action:

|knowledge_reason("Which item is most likely to help with a cold: cough drops, syringe,<br><br>pills, or herbal tea?", [])|
|---|
|Observation: herbal tea<br><br>|

Thought: I know the final answer. Final Answer: herbal tea

[Figure 70]

[Figure 71]

- Question4: What activity does the cat appear most likely to do? You must choose one answer from: drink, jump, eat, sleep

Thought: Determine the

Thought: Check the visual information about the cat and its surroundings.

most likely activity the cat will do based on the objects around it. Action:

Action:

|knowledge_reason("Given that the cat is sitting in front of an open oven with a bowl and a water bottle nearby, what activity is it most likely to do: drink, jump, eat, sleep?", [])|
|---|
|Observation: drink<br><br>|

|caption("What is the cat doing and what is around it?", visual[0])|
|---|
|Observation:<br><br>The cat is sitting on the floor in front of an open oven, staring up at the reflection of itself in the oven's glass door. There is also a bowl and a water bottle nearby.|

visual-0: an image, user

provided image, main content is a cat sitting on the floor looking at an oven

Thought: I know the final answer. Final Answer: drink

[Figure 72]

[Figure 73]

Question5: What are the orange vehicles for? You must choose one answer from: police, shuttle, passengers, air traffic

visual-0: an image, user provided image, main content is two airplanes on a runway

Thought: I need more information about the tow trucks to determine their purpose. Check the choice one by one. Action:

Thought: Check the visual information about the orange vehicles in the image. Action:

Thought: I know the final answer. Final Answer: shuttle

|caption("Are the orange vehicles for police? Are the orange vehicles for shuttle? Are the orange vehicles for passengers? Are the orange vehicles for air traffic?", visual[0])|
|---|
|Observation: The answers for each question are: no, Yes, the<br><br>orange vehicles are shuttle buses., No, Yes, the orange<br><br>vehicles are for air traffic control.|

|caption("What are the orange vehicles for in the image?", visual[0])|
|---|
|Observation: The orange vehicles in the image are tow trucks.<br><br>|

Figure 5: Reasoning process of AssistGPT on A-OKVQA. The choice colored with green in the question indicates the ground truth.

- Question1: Which country was Messi's second goal scored against?

visual-0: a 146.59 seconds video, no meaningful subtitle, user provided video, Every Lionel Messi Goal from the 2022 FIFA World Cup

[Figure 74]

|Thought: I need to find the segment of Messi's second goal Action:<br><br>|video_narration("Describe what Messi is doing.", visual[0])|
|---|
|Observation: … 9-12: he is celebrating with his team mates<br><br>12-15: he is kicking the<br><br>ball into the goal 15-18: he is dribbling the ball …|
<br><br>Thought: Find the segment of Messi's second goal. Action:<br><br>|video_ground("when does Messi score his second goal?", visual[0])| |
|---|---|
|Observation: the target<br><br>clip is from 12-27| |
|visual-1: a 15 seconds video, no meaningful subtitle, segmented from visual-0, …| |
| | |
|[Figure 75]| |
<br><br>Thought: Identify the country Messi scored his second goal against. Action:<br><br>|text_recognize(None, visual[0])|
|---|
|Observation: 63:27 ARG O O MEX WORLD'S BEST AIRLINE QATAR AIRWAYS QATAR AIRWAYS 16 VorldCr<br><br>FIFAⓇ<br><br>WORLD'S BEST AIRLINI|
<br><br>Thought: Infer which country was Messi‘s second goal scored against. Action:<br><br>|knowledge_reason("which<br><br>country was Messi's second goal scored against?", [])|
|---|
|Observation: Mexico<br><br>|
<br><br>Thought: I know the final answer. Final Answer: Mexico<br><br>AssistGPT|
|---|

[Figure 76]

- Question2: How to grip the racket when striking? And show me a video clip to illustrate it.

[Figure 77]

[Figure 78]

visual-0: a 279.08 seconds video, Dense Subtitle, user provided video, Common Beginner Badminton Mistakes - Do And Don'ts 2

|Thought: Find the video segment that shows how to grip the racket when striking.<br><br>Action:<br><br>subtitle_ground("How to grip the racket when striking?", visual[0]) Observation: The target clip is from 32.16 to 42.8 in subtitle.<br><br>Thought: Describe the video clip to provide the answer. Action:<br><br>|video_narration("Describe it.", visual[1])|
|---|
|Observation: 0-3: badminton court - how to play badminton 3-6: how to hit a badminton ball 6-9: how to play badminton<br><br>|
<br><br>visual-1: a 10.64 seconds<br><br>video, dense subtitle,<br><br>segmented video from visual-0, Talking about "How to grip the racket when striking"<br><br>Thought: The narration doesn't provide enough information about gripping the racket. I need to watch<br><br>the video more closely.<br><br>Action:<br><br>|video_narration("Describe how to grip the racket when<br><br>striking.", visual[1])|
|---|
|Observation: 0-3: Describe how to grip the racket when striking 3-6: how to grip the racket when hitting the ball 6-9: Describe how to grip the racket when striking<br><br>|
<br><br>Thought: The video narration still doesn't provide enough information about gripping the racket. I will try to find more information from<br><br>the subtitles.<br><br>Action:<br><br>|subtitle_reason("How to grip the racket when striking?", visual[1])|
|---|
|Observation: When striking,<br><br>grip the racket with a forehand grip, which allows you to rotate the forearm and generate more power in your shots. This grip also enables you to get into a side-on position and bring your elbow back, giving you more control and deception with your overhead shots.|
<br><br>Thought: I know the final answer.<br><br>Final Answer: When striking, grip the racket with a forehand grip, which allows you to rotate the forearm and generate more power<br><br>in your shots. This grip also enables you to get into a side-on position and bring your elbow back, giving you more control and deception with your overhead shots. To see a video clip illustrating this, please check the visual output. (visual[1])<br><br>AssistGPT<br><br>[Figure 79]<br><br>|
|---|

#### Figure 6: The reasoning process of AssistGPT when handling in-the-wild questions.

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Question3: Show me a video to illustrate what is the meaning of the setting in the image?

- visual-0: a 917.51 seconds video, Dense subtitle, user provided video, GoPro Hero 8 Black Best Tips and Tricks!

- visual-1: a 1783.08 seconds video, Dense subtitle, user provided video, GoPro Hero 10 Beginners Guide & Tutorial _ Getting Started

- visual-2: an image, user provided image, main content is a close up of a person holding a smart watch.

|Thought: First, I need to understand what the setting in the image is. I will use the text_detect tool to read<br><br>any visible settings or<br><br>text in the image. Action:<br><br>|text_detect(None,<br><br>visual[2])|
|---|
|Observation: Linear + Horizon Lock, Full 360 rotation.<br><br>|
<br><br>Thought: The text recognition tool provided the setting information, which is “Linear + Horizon Lock, Full 360 rotation.”<br><br>It appears to be a setting<br><br>or a feature of a device, possibly the smartwatch in the image. … I'll use the subtitle_ground tool on both videos to find a segment that mentions this setting. Action:<br><br>|subtitle_ground("Linear<br><br>+ Horizon Lock, Full 360 rotation", visual[0])|
|---|
|Observation: Warning: Relevant clip didn’t found.<br><br>|
<br><br>Thought: The first video doesn't contain the required information. I'll now try the same action with the second video.<br><br>Action:<br><br>subtitle_ground("Linear<br><br>+ Horizon Lock, Full 360 rotation", visual[1]) Observation: the target clip is from 18:02 – 18:44<br><br>Thought: The subtitle_ground tool found a clip from the second video that mentions "Linear<br><br>+ Horizon Lock, Full 360 rotation." This video segment should illustrate what these settings mean. Final Answer: (The meaning<br><br>of the setting 'Linear +<br><br>Horizon Lock, Full 360 rotation' is explained in the extracted video clip, visual[3])<br><br>AssistGPT<br><br>visual-3: a 42 seconds video, segmented video from visual-1, dense subtitle, target clip for query "Linear + Horizon Lock, Full 360 rotation"<br><br>[Figure 85]<br><br>[Figure 86]<br><br>|
|---|

[Figure 87]

[Figure 88]

Question4: How to Cook Perfect Beef Stir Fry?

visual-0: a 197.12 seconds video, no meaningful subtitle, user provided video, How to Cook Perfect Beef Stir Fry Every Time

|Thought: I need to watch the video and narrate the content.<br><br>Action:<br><br>|video_narration("Describe it.", visual[0])|
|---|
|Observation:<br><br>0-3: asian beef stir fry. Texts on the frame (Taste) 3-6: a chef is preparing a beef dish Texts on the frame (Beef 350 g Taste) 6-9: a person is cutting a piece of meat. Texts on the frame (Taste) 9-12: a person is cutting a piece of meat. Texts on the frame (e Taste)|
<br><br>Thought: I have the narration of the video, now I can<br><br>summarize the steps to cook perfect beef stir fry.<br><br>Final Answer: To cook perfect beef stir fry, follow these steps:<br><br>1. Slice 350g of beef.<br>2. Cut 100g of carrots and 200g of broccoli.<br>3. Mince 15g of garlic.<br>4. Heat a wok with cold oil to prevent sticking.<br>5. Cook the beef in the wok.<br>6. Add the vegetables and cook them.<br>7. Add 1 tsp of soy sauce and some water.<br>8. Mix water and cornstarch, then add it to the wok.<br>9. Add some lemon juice.<br>10. Serve the stir fry over cooked brown rice.<br><br><br>AssistGPT|
|---|

Question5: List all the ingredients for the dish and their weights.

|Thought: I need to find the ingredients and their weights in the video. Action:<br><br>|video_narration("Describe the ingredients and their<br><br>weights.", visual[0])|
|---|
|Observation: 0-3: asian beef stir fry. Texts on the frame (Taste) …<br><br>84-87: garlic. Texts on the frame (Garlic 15 g Taste) …<br><br>132-135: 1 cup of broccoli florets. Texts on the frame (Taste) …|
<br><br>Thought: I have found the ingredients and their weights in the video narration. Final Answer: The ingredients and their weights for the beef<br><br>stir fry are: Beef 350g, Soy Sauce 1 1/2 tsp,<br><br>Carrot 100g, Broccoli 200g, Garlic 15g, and Cooked Brown Rice.<br><br>AssistGPT|
|---|

#### Figure 7: The reasoning process of AssistGPT when handling in-the-wild questions.

### References

- [1] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.
- [2] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, page 9, 2019.
- [3] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. NeurIPS, pages 1877–1901, 2020.
- [4] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [5] OpenAI. Introducing chatgpt. OpenAI Blog, 09 2021.
- [6] Andy Zeng, Adrian Wong, Stefan Welker, Krzysztof Choromanski, Federico Tombari, Aveek Purohit, Michael Ryoo, Vikas Sindhwani, Johnny Lee, Vincent Vanhoucke, et al. Socratic models: Composing zero-shot multimodal reasoning with language. arXiv preprint arXiv:2204.00598, 2022.
- [7] Zhenhailong Wang, Manling Li, Ruochen Xu, Luowei Zhou, Jie Lei, Xudong Lin, Shuohang Wang, Ziyi Yang, Chenguang Zhu, Derek Hoiem, et al. Language models with image descriptors are strong few-shot video-language learners. arXiv preprint arXiv:2205.10747, 2022.
- [8] Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. Visual chatgpt: Talking, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671, 2023.
- [9] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML, pages 12888– 12900, 2022.
- [10] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023.
- [11] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500, 2023.
- [12] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In ECCV, pages 213–229, 2020.
- [13] Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, and Jianfeng Gao. Grounded language-image pre-training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 10965–10975, June 2022.
- [14] Jialian Wu, Jianfeng Wang, Zhengyuan Yang, Zhe Gan, Zicheng Liu, Junsong Yuan, and Lijuan Wang. Grit: A generative region-to-text transformer for object understanding. arXiv preprint arXiv:2212.00280, 2022.
- [15] Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, and Lijuan Wang. Git: A generative image-to-text transformer for vision and language. arXiv preprint arXiv:2205.14100, 2022.

- [16] Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Nan Ding, Keran Rong, Hassan Akbari, Gaurav Mishra, Linting Xue, Ashish Thapliyal, James Bradbury, Weicheng Kuo, Mojtaba Seyedhosseini, Chao Jia, Burcu Karagol Ayan, Carlos Riquelme, Andreas Steiner, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, and Radu Soricut. Pali: A jointly-scaled multilingual language-image model, 2022.
- [17] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in huggingface. arXiv preprint arXiv:2303.17580, 2023.
- [18] Yingqiang Ge, Wenyue Hua, Jianchao Ji, Juntao Tan, Shuyuan Xu, and Yongfeng Zhang. Openagi: When llm meets domain experts. arXiv preprint arXiv:2304.04370, 2023.
- [19] Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, and Jianfeng Gao. Chameleon: Plug-and-play compositional reasoning with large language models. arXiv preprint arXiv:2304.09842, 2023.
- [20] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023.
- [21] Dídac Surís, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. arXiv preprint arXiv:2303.08128, 2023.
- [22] Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. arXiv preprint arXiv:2211.11559, 2022.
- [23] Yaobo Liang, Chenfei Wu, Ting Song, Wenshan Wu, Yan Xia, Yu Liu, Yang Ou, Shuai Lu, Lei Ji, Shaoguang Mao, Yun Wang, Linjun Shou, Ming Gong, and Nan Duan. Taskmatrix.ai: Completing tasks by connecting foundation models with millions of apis. arXiv preprint arXiv:2303.16434, 2023.
- [24] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.
- [25] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. arXiv preprint arXiv:2302.04761, 2023.
- [26] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425–2433, 2015.
- [27] Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang. Bottom-up and top-down attention for image captioning and visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6077–6086, 2018.
- [28] Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee. Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. Advances in neural information processing systems, 32, 2019.
- [29] Hao Tan and Mohit Bansal. Lxmert: Learning cross-modality encoder representations from transformers. arXiv preprint arXiv:1908.07490, 2019.
- [30] Difei Gao, Ruiping Wang, Shiguang Shan, and Xilin Chen. Learning to recognize visual concepts for visual question answering with structural label space. IEEE Journal of Selected Topics in Signal Processing, 14(3):494–505, 2020.
- [31] Difei Gao, Ruiping Wang, Shiguang Shan, and Xilin Chen. Visual textbook network: Watch carefully before answering visual questions. In BMVC, 2017.

- [32] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pages 3195–3204, 2019.
- [33] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. arXiv, 2022.
- [34] Peng Wang, Qi Wu, Chunhua Shen, Anthony Dick, and Anton Van Den Hengel. Fvqa: Fact-based visual question answering. IEEE transactions on pattern analysis and machine intelligence, 40(10):2413–2427, 2017.
- [35] Liangke Gui, Borui Wang, Qiuyuan Huang, Alex Hauptmann, Yonatan Bisk, and Jianfeng Gao. Kat: A knowledge augmented transformer for vision-and-language. arXiv preprint arXiv:2112.08614, 2021.
- [36] Kenneth Marino, Xinlei Chen, Devi Parikh, Abhinav Gupta, and Marcus Rohrbach. Krisp: Integrating implicit and symbolic knowledge for open-domain knowledge-based vqa. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14111–14121, 2021.
- [37] Difei Gao, Ruiping Wang, Shiguang Shan, and Xilin Chen. Cric: A vqa dataset for compositional reasoning on vision and commonsense. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2022.
- [38] Max Bain, Arsha Nagrani, Gül Varol, and Andrew Zisserman. Frozen in time: A joint video and image encoder for end-to-end retrieval. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1728–1738, 2021.
- [39] Jie Lei, Linjie Li, Luowei Zhou, Zhe Gan, Tamara L. Berg, Mohit Bansal, and Jingjing Liu. Less is more: Clipbert for video-and-language learning via sparse sampling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7331–7341, June 2021.
- [40] Yuxuan Wang, Difei Gao, Licheng Yu, Weixian Lei, Matt Feiszli, and Mike Zheng Shou. Geb+: A benchmark for generic event boundary captioning, grounding and retrieval. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXV, pages 709–725. Springer, 2022.
- [41] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18995–19012, 2022.
- [42] Kevin Qinghong Lin, Jinpeng Wang, Mattia Soldan, Michael Wray, Rui Yan, Eric Z XU, Difei Gao, Rong-Cheng Tu, Wenzhe Zhao, Weijie Kong, et al. Egocentric video-language pretraining. Advances in Neural Information Processing Systems, 35:7575–7586, 2022.
- [43] Difei Gao, Ruiping Wang, Ziyi Bai, and Xilin Chen. Env-qa: A video question answering benchmark for comprehensive understanding of dynamic environments. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1675–1685, 2021.
- [44] Zhijian Hou, Wanjun Zhong, Lei Ji, Difei Gao, Kun Yan, Wing-Kwong Chan, Chong-Wah Ngo, Zheng Shou, and Nan Duan. Cone: An efficient coarse-to-fine alignment framework for long video temporal grounding. arXiv preprint arXiv:2209.10918, 2022.
- [45] Benita Wong, Joya Chen, You Wu, Stan Weixian Lei, Dongxing Mao, Difei Gao, and Mike Zheng Shou. Assistq: Affordance-centric question-driven task completion for egocentric assistant. In European Conference on Computer Vision, pages 485–501. Springer, 2022.
- [46] Weixian Lei, Difei Gao, Yuxuan Wang, Dongxing Mao, Zihan Liang, Lingmin Ran, and Mike Zheng Shou. Assistsr: Task-oriented video segment retrieval for personal ai assistant. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 319–338, 2022.

- [47] Joya Chen, Difei Gao, Kevin Qinghong Lin, and Mike Zheng Shou. Affordance grounding from demonstration video to target image. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6799–6808, 2023.
- [48] Ronghang Hu, Amanpreet Singh, Trevor Darrell, and Marcus Rohrbach. Iterative answer prediction with pointer-augmented multimodal transformers for textvqa. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9992–10002, 2020.
- [49] Zhengyuan Yang, Yijuan Lu, Jianfeng Wang, Xi Yin, Dinei Florencio, Lijuan Wang, Cha Zhang, Lei Zhang, and Jiebo Luo. Tap: Text-aware pre-training for text-vqa and text-caption. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8751–8761, 2021.
- [50] Difei Gao, Ke Li, Ruiping Wang, Shiguang Shan, and Xilin Chen. Multi-modal graph neural network for joint reasoning on vision and scene text. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.
- [51] Stan Weixian Lei, Difei Gao, Jay Zhangjie Wu, Yuxuan Wang, Wei Liu, Mengmi Zhang, and Mike Zheng Shou. Symbolic replay: Scene graph as prompt for continual learning on vqa task. arXiv preprint arXiv:2208.12037, 2022.
- [52] OpenAI. Gpt-4 technical report, 2023.
- [53] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.
- [54] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023.
- [55] Deyao Zhu, Jun Chen, Xiaoqian Shen, xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models, 2023.
- [56] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.
- [57] Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. Neural module networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 39–48, 2016.
- [58] Ronghang Hu, Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Kate Saenko. Learning to reason: End-to-end module networks for visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 804–813, 2017.
- [59] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Judy Hoffman, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Inferring and executing programs for visual reasoning. In Proceedings of the IEEE international conference on computer vision, pages 2989–2998, 2017.
- [60] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed Chi, Quoc Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903, 2022.
- [61] Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381, 2023.
- [62] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv preprint arXiv:2304.02643, 2023.

- [63] Jiaqi Chen, Zeyu Yang, and Li Zhang. Semantic segment anything. https://github.com/ fudan-zvg/Semantic-Segment-Anything, 2023.
- [64] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023.
- [65] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. arXiv preprint arXiv:2212.04356, 2022.
- [66] Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang. Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework. CoRR, abs/2202.03052, 2022.
- [67] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR, 2021.
- [68] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of questionanswering to explaining temporal actions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9777–9786, 2021.
- [69] Amita Kamath, Christopher Clark, Tanmay Gupta, Eric Kolve, Derek Hoiem, and Aniruddha Kembhavi. Webly supervised concept expansion for general purpose vision models. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXXVI, pages 662–681. Springer, 2022.
- [70] Yushi Hu, Hang Hua, Zhengyuan Yang, Weijia Shi, Noah A Smith, and Jiebo Luo. Promptcap: Prompt-guided task-aware image captioning. arXiv preprint arXiv:2211.09699, 2022.
- [71] Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Just ask: Learning to answer questions from millions of narrated videos. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1686–1697, 2021.
- [72] Shyamal Buch, Cristóbal Eyzaguirre, Adrien Gaidon, Jiajun Wu, Li Fei-Fei, and Juan Carlos Niebles. Revisiting the" video" in video-language understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2917–2927, 2022.
- [73] Junbin Xiao, Pan Zhou, Tat-Seng Chua, and Shuicheng Yan. Video graph transformer for video question answering. In European Conference on Computer Vision, pages 39–58. Springer, 2022.
- [74] Difei Gao, Luowei Zhou, Lei Ji, Linchao Zhu, Yi Yang, and Mike Zheng Shou. Mist: Multimodal iterative spatial-temporal transformer for long-form video question answering. arXiv preprint arXiv:2212.09522, 2022.
- [75] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

