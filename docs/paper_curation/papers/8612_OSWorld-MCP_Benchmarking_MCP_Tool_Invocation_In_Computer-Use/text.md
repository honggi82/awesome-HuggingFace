# arXiv:2510.24563v2[cs.CV]11Nov2025

## OSWORLD-MCP: BENCHMARKING MCP TOOL INVOCATION IN COMPUTER-USE AGENTS

### Hongrui Jia1,2,†, Jitong Liao2,†, Xi Zhang2,†, Haiyang Xu2,∗, Tianbao Xie2, Chaoya Jiang1, Ming Yan2,∗, Si Liu3, Wei Ye1,∗, Fei Huang2 1 Peking University 2 Tongyi Lab, Alibaba Group 3 Beijing Zhongguancun Academy

{jiahongrui, wye}@pku.edu.cn, shuofeng.xhy@alibaba-inc.com

ABSTRACT

With advances in decision-making and reasoning capabilities, multimodal agents show strong potential in computer application scenarios. Past evaluations have mainly assessed GUI interaction skills, while tool invocation abilities, such as those enabled by the Model Context Protocol (MCP), have been largely overlooked. Comparing agents with integrated tool invocation to those evaluated only on GUI interaction is inherently unfair. We present OSWorld-MCP, the first comprehensive and fair benchmark for assessing computer-use agents’ tool invocation, GUI operation, and decision-making abilities in a real-world environment. We design a novel automated code-generation pipeline to create tools and combine them with a curated selection from existing tools. Rigorous manual validation yields 158 high-quality tools (covering 7 common applications), each verified for correct functionality, practical applicability, and versatility. Extensive evaluations of state-of-the-art multimodal agents on OSWorld-MCP show that MCP tools generally improve task success rates (e.g., from 8.3% to 20.4% for OpenAI o3 at 15 steps, from 40.1% to 43.3% for Claude 4 Sonnet at 50 steps), underscoring the importance of assessing tool invocation capabilities. However, even the strongest models have relatively low tool invocation rates, Only 36.3%, indicating room for improvement and highlighting the benchmark’s challenge. By explicitly measuring MCP tool usage skills, OSWorld-MCP deepens understanding of multimodal agents and sets a new standard for evaluating performance in complex, tool-assisted environments. Our code, environment, and data are publicly available at https://osworld-mcp.github.io.

1 INTRODUCTION

Large Language Models (LLMs) such as GPT-5 (OpenAI, 2025a), DeepSeek-R1 (Guo et al., 2025a), and Qwen3 (Yang et al., 2025) have dramatically advanced reasoning and decision-making capabilities. Building on these advances, recent Large Multimodal Models (LMMs) are able to address complex computer-use tasks, which has stimulated considerable research interest (Qin et al., 2025; Lai et al., 2025; Song et al., 2025; Luo et al., 2025; Lu et al., 2025a;b; Ye et al., 2025; Wang et al., 2024b; 2025a; Zhu et al., 2025; Wang et al., 2024a). Consequently, how to reliably and robustly benchmark different LMMs in GUI-driven scenarios has become a key open question. Existing evaluation frameworks (Xie et al., 2024; Abhyankar et al., 2025; Xie et al., 2025; Bonatti et al., 2024; Kuntz et al., 2025; Rawles et al., 2024) primarily focus on assessing a model’s ability to perform GUI-based operations, by predefining a set of user-interface actions (e.g., click, type, and drag) and allowing the model to autonomously decide how to complete the task.

Although many benchmarks have been proposed for evaluating GUI agents, most neglect a crucial capability: the ability to invoke external tools such as the Model Context Protocol (MCP) (Anthropic, 2024). MCP is an open standard that connects AI applications with external systems. By using MCP, computer-use agents can access diverse resources, including data sources such as local files and databases, and tools such as search engines and calculator, thereby enabling them to obtain critical information and complete tasks more effectively. In many cases, performing a task through

†Equal contribution. ∗Corresponding author.

[Figure 1]

Figure 1: Comparison of completing the instruction “Please help me install the autoDocstring extension in VS Code” via GUI operations (a) and the MCP Tool (b).

MCP is more efficient than relying solely on GUI operations. For instance, as illustrated in Figure 1, when an agent is instructed to install the autoDocstring extension in VS Code, a GUI-based approach may require at least four steps, whereas an MCP tool can achieve the same result in a single step, offering both greater efficiency and higher robustness. Several recent agents (Lai et al., 2025; Song et al., 2025) have already incorporated autonomous tool invocation and have achieved notable performance gains. However, it is inherently inequitable to compare such agents with others that assess only GUI interaction. At present, there remains a lack of comprehensive and equitable benchmarks that jointly evaluate GUI operation skills, tool invocation capabilities, and the decisionmaking competence of computer-use agents in an integrated framework.

To bridge this gap, we introduce OSWorld-MCP, the first comprehensive and fair benchmark designed to evaluate the tool invocation capabilities of computer-use agents.Our primary motivation is to establish a unified standard for fair comparison of tool utilization abilities across different models, addressing the current lack of consistency in tool sets and evaluation metrics. Built upon a widely used real-world computer-use environment OSWorld (Xie et al., 2024), OSWorld-MCP significantly extends its capabilities by incorporating a curated set of 158 high-quality MCP tools. These tools cover 7 common applications such as LibreOffice Writer and VS Code, ensuring a diverse and realistic testing environment (Figure 4). A number of 25 non-target tools are included, serving as distractors in the OSWorld-MCP tasks. Besides, our designed MCP tools are applicable to 250 tasks, accounting for 69% of the entire benchmark, which underscores their broad applicability. Notably, 153 tasks, representing 42% of the benchmark, involve challenging multi-round tool invocations. Even the strongest model, Claude 4 Sonnet, achieves an accuracy of 0 when relying solely on GUI operations for tasks requiring four rounds of tool invocation, and only 16.7% accuracy when MCP tools are introduced. These results highlight the challenging of our benchmark and the broad utility of our tools in diverse scenarios.

Another distinguishing feature of OSWorld-MCP is its dynamic interaction between GUI operations and tool usage. Specifically, at every step of a task, the agent can autonomously choose between our MCP tools and direct GUI actions (e.g., click and type) to interact with the graphical interface. With this setting, OSWorld-MCP can provide a balanced and thorough assessment of LMM capabilities in hybrid decision-making skills. Here, the decision-making involves not only selecting the correct tools, but also choosing the most efficient execution path when both GUI operations and MCP tools are required to accomplish the task. Besides, to provide a more nuanced evaluation, we introduce two new metrics alongside task accuracy: Tool Invocation Rate (TIR) and Average Completion Steps (ACS). TIR measures the proportion of tasks successfully completed using MCP tools, offering insights into an agent’s tool utilization propensity, while ACS quantifies task completion efficiency. In conclusion, compared with existing text-based tool-use benchmarks (Liu et al., 2025; Mo et al., 2025; Gao et al., 2025; Wang et al., 2025b) and the above GUI relevant benchmarks, OSWorld-MCP has significant advantages in evaluating real-world computer-use scenarios. It challenges agents to interpret visual GUI information, perform GUI operations, invoke appropriate tools, and effectively

chain multiple tools. This combination of features makes OSWorld-MCP a more comprehensive and realistic benchmark for evaluating computer-use capabilities.

To develop the above-mentioned high-quality MCP tools, we design an automated code generation pipeline comprising three modules: the Code Generation Module, the Code Filter Module, and the Tool Wrap Module. By leveraging the advanced reasoning capabilities of OpenAI o3 (OpenAI, 2025b), this pipeline produces 72 functional tools. These tools are then combined with those curated from existing MCP servers (Lai et al., 2025), followed by a fine-grained manual verification procedure to remove functionally redundant items and highly task-specific ones that lack relevance to real-world applications. Through this process, we obtain a curated collection of 158 high-quality tools, each verified to be readily usable and aligned with the designed task difficulty. With this comprehensive tool set and our OSWorld-MCP, we conduct a detailed experimental study and analysis on a range of state-of-the-art LMMs and multimodal frameworks. Three key findings are identified during experiments: (1) MCP tools enhance agent accuracy and efficiency compared to the GUI-only setting. For example, the success rate of OpenAI o3 increases from 8.3% to 20.4% at 15 steps. (2) The tool invocation rate positively correlates with performance. We also observe that tool invocation rates for multimodal agents remain relatively low, indicating the significant potential in the tool utilization capabilities of current LMMs and multimodal frameworks. (3) The composition of multiple tools remains a significant challenge. Performance declines on complex task involving more tools. In the other hands, agent struggle on selecting the correct tool from a full list.

Our contributions are as follows:

- • We introduce OSWorld-MCP, a comprehensive, fair, and novel benchmark for evaluating computer-use agents that integrates 158 high-quality MCP tools (covering 7 common apps) with GUI operations in real-world scenarios. It bridges the gap between pure-GUI and text-based tool-use evaluations, offering a holistic and realistic assessment of computer-use capabilities.
- • We propose a novel pipeline combining automated code generation with rigorous manual curation to create MCP tools, enhancing our benchmark’s evaluation depth and fairness. We also introduce new metrics (i.e., TIR and ACS) for nuanced assessment of agents’ tool utilization propensity.
- • Our extensive experiments indicate that (1) MCP tools improve agent metrics; (2) higher tool invocation correlates with higher accuracy; (3) combining tools introduces significant challenges.

2 RELATED WORK

- 2.1 BENCHMARKS FOR MULTIMODAL AGENTS Current benchmarks (Deng et al., 2023; L`u et al., 2024; Kapoor et al., 2024; Zhou et al., 2023; Koh

- et al., 2024; Drouin et al., 2024; Tian et al., 2024; Bonatti et al., 2024; Xie et al., 2024) for multimodal agents primarily focus on evaluating their ability to complete tasks through GUI-based operations. Static benchmarks such as Mind2Web (Deng et al., 2023), WebLinx (L`u et al., 2024), and OmniAct (Kapoor et al., 2024) rely on manually collected static datasets to assess agent performance. These static benchmarks are built upon pre-defined trajectories of GUI actions, which make them incapable of evaluating alternative action paths that may arise when tool invocation is introduced. In contrast, dynamic interactive benchmarks operate in open-ended environments and provide reward signals upon task completion, enabling a more flexible and open-ended evaluation of agents. Notable examples of dynamic benchmarks for specific environments include WebArena (Zhou et al.,

- 2023), VisualWebArena (Koh et al., 2024), WorkArena (Drouin et al., 2024), MMInA (Tian et al.,
- 2024), WindowsAgentArena (Bonatti et al., 2024), and OSWorld (Xie et al., 2024). However, existing dynamic interactive benchmarks typically predefine only GUI actions for the agent to use, and therefore lack a comprehensive and fair evaluation framework that jointly measures multimodal agents’ tool invocation, GUI interaction, and decision-making capabilities.

- 2.2 MODEL CONTEXT PROTOCOL

With the rapid development of multimodal agents, increasing attention has been paid to their tool invocation capabilities (Song et al., 2025; Lai et al., 2025). Introduced by Anthropic in November 2024, the Model Context Protocol (MCP) is a JSON-RPC–based client–server interface designed

[Figure 2]

Figure 2: Overview of the OSWorld-MCP framework.

for secure context ingestion and structured tool invocation. MCP provides a standardized, modelagnostic interface that enables AI applications to connect to external systems such as tools, data resources, and workflows, thereby facilitating the integration of large language models with external data sources and utilities (Anthropic, 2024). This standardization addresses challenges arising from fragmented and highly customized integrations. MCP supports flexible plug-and-play tooling, secure infrastructure integration, and cross-LLM vendor compatibility (Ehtesham et al., 2025). Several contemporary MCP-related benchmarks for LLM evaluation have recently emerged (Gao

- et al., 2025; Liu et al., 2025; Mo et al., 2025; Wang et al., 2025b). MCPEval (Liu et al., 2025) and MCP-Radar (Gao et al., 2025) cover only a limited set of MCP servers, typically no more than a few dozen tools, which restricts task diversity. LiveMCPBench (Mo et al., 2025) employs large language model–based evaluation, an approach that is not well suited to tasks requiring real-time knowledge. MCP-Bench (Wang et al., 2025b) defines its tasks based on available tools, introducing constraints that create a gap between benchmark tasks and truly open-ended real-world problems. Currently, there is no benchmark that comprehensively evaluates multimodal agents in terms of GUI interaction, tool invocation, and decision-making capabilities within an integrated and fair framework.

- 3 OSWORLD-MCP BENCH

- 3.1 OVERVIEW

We propose OSWorld-MCP, a comprehensive benchmark for evaluating computer-use agents. OSWorld-MCP is built upon the OSWorld benchmark, which is a widely used dynamic and interactive evaluation framework designed to assess the performance of multimodal agents in realistic computing environments, including Ubuntu, Windows, and macOS. OSWorld covers nine applications and consists of a total of 369 real-world tasks that involve interaction through both graphical user interfaces (GUI) and command-line interfaces (CLI). As illustrated in Figure 2, OSWorld-MCP enables effective assessment of multimodal agents in authentic scenarios, capturing their tool invocation capability, GUI operation skills, and decision-making competence. Decision-making assessment includes the ability to choose between GUI and MCP pathways, as well as the ability to select the most appropriate MCP tool for a given task. In the following section, we first introduce our automated pipeline for tool generation and the procedures for collecting and filtering tools used to construct OSWorld-MCP. We then present a detailed analysis of the MCP tools we produce, demonstrating the high quality and rational design of the OSWorld-MCP tool set. Finally, we describe the evaluation metrics defined for the OSWorld-MCP dataset and outline how these metrics enable comprehensive performance measurement of different models.

- 3.2 TOOLS GENERATION AND FILTER

Existing MCP servers generally offer tools that are relatively simple and often have overlapping functionalities. Consequently, there is a shortage of high-quality MCP tools that can be readily

[Figure 3]

- Figure 3: Illustration of our tool generation process. (a) Prompt for wrapping code into MCP tools with OpenAI o3. (b) Workflow for generating high-quality tools.

applied in practical scenarios. To address this limitation, we design an automated tool generation pipeline composed of three modules: the Code Generation Module, the Code Filter Module, and the Tool Wrap Module. In the Code Generation Module, the user only needs to specify the target task, and the module automatically generates code to accomplish it. Specifically, we employ OpenAI

- o3 (OpenAI, 2025b) to produce code-based solutions for every task in OSWorld. Following the prompting strategy of CoAct (Song et al., 2025), we develop our own prompt and instruct OpenAI
- o3 to generate, whenever feasible, functional code solutions capable of completing the tasks. In the Code Filter Module, we use OpenAI o3 to summarize usable code obtained from multi-turn interactions. This summarized code is then applied to solve the corresponding tasks, and any code successfully completing the tasks is retained. Through this process, we obtain seventy-two verified solutions. In the Tool Wrap Module, we provide OpenAI o3 with a prompt that automates the packaging of these verified code solutions into 72 MCP tools, as illustrated in Figure 3.(a).

In addition, we carefully curate 192 tools from existing MCP servers. Since some generate or imported tools are tailored to solve a single specific task and thus unsuitable for real-world use, we conduct a manual inspection of all 264 collected tools to remove such task-specific items as well as functionality duplicates. Each tool is independently evaluated by at least two reviewers with extensive GUI agent development experience, and is retained only if both reviewers deemed it qualified. After this manual filtering process, we obtain 158 high-quality tools that are both applicable and valuable in real-world computing environments.

- 3.3 TOOLS ANALYSIS

We analyze the composition of the 158 high-quality tools designed for practical real-world use, with their distribution across application scenarios shown in Figure 4.(b). To further ensure that these tools have a tangible positive impact on task completion in realistic settings, we conduct an additional manual validation and find that 133 tools effectively contribute to improving task efficiency, while the remaining 25 tools originate from existing external MCP servers.

To verify that the effective tools can be actively utilized by models, we evaluate five state-of-theart large multimodal models, including Qwen2.5-VL-72B-Instruct (Bai et al., 2025) and Claude 4 Sonnet (Anthropic, 2025), on OSWorld-MCP, allowing them at each step to autonomously choose between performing GUI operations and invoking any of the 158 high-quality tools. The results indicate that 131 tools are invoked at least once during evaluation. The remaining two tools, which are manually re-verified for usability, are hypothesized to be absent from model usage due to the complexity of the associated tasks, which likely discouraged models from attempting to invoke them. Tool invocation frequencies are presented in Figure 4.(a).

We also conduct a manual analysis of all 361 OSWorld-MCP tasks (8 Google Drive tasks excluded) to annotate the set of available tools for each task and to record the total number of invocations of these tools across evaluations. The distribution of total available tool invocations per task is shown in Figure 4.(c). An available tool is defined as one whose invocation can make task execution substantially more efficient. Based on this definition, OSWorld-MCP is classified into two subsets:

(a) Number of times each tool was invoked by the five models in the OSWorld-MCP evaluation

(b) The percentage of available tools in each software

[Figure 4]

[Figure 5]

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
|osworld_mcp_code.update_extensions|osworld_mcp_code.list_extensions| | | |

Goolge Chrome 9%

450

[Figure 6]

osworld_mcp_libreoffice_calc.set_cell_value

VS Code 15%

OS 14%

[Figure 7]

400

LibreOffice Impress 17%

[Figure 8]

350

VLC 9%

[Figure 9]

[Figure 10]

300

LibreOffice Writer 12%

LibreOffice Calc 24%

osworld_mcp_libreoffice_calc.format_numbers_to_human_readable

250

osworld_mcp_libreoffice_calc.scale_first_sheet_and_export_pdf

osworld_mcp_google_chrome.open_search_engine_settings

osworld_mcp_libreoffice_writer.set_paragraph_alignment

osworld_mcp_google_chrome.open_appearance_settings

osworld_mcp_libreoffice_calc.copy_cells_between_sheets

osworld_mcp_libreoffice_impress.set_textbox_alignment

osworld_mcp_libreoffice_impress.set_text_strikethrough

osworld_mcp_libreoffice_impress.set_background_color

osworld_mcp_libreoffice_impress.set_slide_background

osworld_mcp_google_chrome.open_password_settings

osworld_mcp_libreoffice_writer.insert_image_at_cursor

osworld_mcp_libreoffice_impress.configure_auto_save

osworld_mcp_libreoffice_impress.set_slide_orientation

(c) Distribution of the total number of available tool invocations per task

200

osworld_mcp_os.copy_matching_files_with_hierarchy

osworld_mcp_google_chrome.delete_browsing_data

osworld_mcp_google_chrome.open_privacy_settings

osworld_mcp_code.set_python_diagnostics_override

osworld_mcp_google_chrome.open_profile_settings

osworld_mcp_libreoffice_writer.add_page_numbers

osworld_mcp_libreoffice_calc.adjust_column_width

osworld_mcp_libreoffice_impress.export_to_image

osworld_mcp_google_chrome.bring_back_last_tab

osworld_mcp_libreoffice_writer.insert_page_break

osworld_mcp_libreoffice_impress.convert_to_docx

osworld_mcp_libreoffice_calc.switch_active_sheet

osworld_mcp_libreoffice_calc.set_number_format

osworld_mcp_libreoffice_impress.get_slide_count

osworld_mcp_libreoffice_calc.get_workbook_info

osworld_mcp_libreoffice_writer.find_and_replace

osworld_mcp_libreoffice_calc.set_column_values

osworld_mcp_libreoffice_impress.duplicate_slide

osworld_mcp_libreoffice_impress.delete_content

osworld_mcp_libreoffice_calc.create_pivot_table

osworld_mcp_libreoffice_writer.set_line_spacing

osworld_mcp_libreoffice_writer.set_default_font

osworld_mcp_google_chrome.open_bookmarks

osworld_mcp_libreoffice_impress.set_slide_font

osworld_mcp_libreoffice_impress.set_text_color

osworld_mcp_google_chrome.open_extensions

osworld_mcp_libreoffice_calc.get_column_data

osworld_mcp_libreoffice_impress.write_text

osworld_mcp_google_chrome.bookmark_page

osworld_mcp_libreoffice_calc.reorder_columns

osworld_mcp_libreoffice_calc.transpose_range

osworld_mcp_libreoffice_impress.insert_image

osworld_mcp_libreoffice_impress.position_box

osworld_mcp_code.set_focus_editor_on_break

osworld_mcp_libreoffice_calc.highlight_range

osworld_mcp_libreoffice_calc.fill_blank_down

osworld_mcp_libreoffice_writer.export_to_pdf

osworld_mcp_libreoffice_impress.go_to_slide

osworld_mcp_libreoffice_calc.set_zoom_level

osworld_mcp_os.remove_image_background

osworld_mcp_libreoffice_calc.reorder_sheets

osworld_mcp_libreoffice_calc.hide_row_data

osworld_mcp_libreoffice_writer.set_font_size

osworld_mcp_libreoffice_calc.rename_sheet

osworld_mcp_libreoffice_calc.export_to_csv

osworld_mcp_libreoffice_calc.env_info

osworld_mcp_libreoffice_calc.format_range

osworld_mcp_libreoffice_impress.insert_file

osworld_mcp_code.set_word_wrap_column

osworld_mcp_os.get_do_not_disturb_status

osworld_mcp_libreoffice_calc.freeze_panes

osworld_mcp_os.set_default_terminal_size

120

osworld_mcp_libreoffice_impress.set_style

osworld_mcp_libreoffice_calc.create_chart

osworld_mcp_libreoffice_impress.env_info

osworld_mcp_libreoffice_calc.sort_column

osworld_mcp_libreoffice_calc.merge_cells

osworld_mcp_libreoffice_impress.save_as

osworld_mcp_libreoffice_writer.write_text

osworld_mcp_libreoffice_calc.copy_sheet

150

osworld_mcp_libreoffice_writer.set_color

osworld_mcp_code.set_auto_save_delay

osworld_mcp_libreoffice_writer.env_info

osworld_mcp_libreoffice_writer.set_font

osworld_mcp_os.convert_image_format

osworld_mcp_code.update_extensions

osworld_mcp_libreoffice_impress.save

osworld_mcp_os.ffmpeg_video_to_gif

osworld_mcp_code.add_files_exclude

osworld_mcp_os.configure_auto_lock

osworld_mcp_code.install_extension

osworld_mcp_libreoffice_writer.save

osworld_mcp_os.get_trash_directory

osworld_mcp_os.set_do_not_disturb

osworld_mcp_code.set_color_theme

100

osworld_mcp_os.change_text_scale

osworld_mcp_google_chrome.print

osworld_mcp_vlc.toggle_fullscreen

osworld_mcp_os.rename_directory

osworld_mcp_code.launch_vscode

osworld_mcp_libreoffice_calc.save

osworld_mcp_vlc.get_current_time

filesystem_list_allowed_directories

osworld_mcp_code.list_extensions

osworld_mcp_code.set_wrap_tabs

osworld_mcp_vlc.get_media_files

osworld_mcp_vlc.add_to_playlist

osworld_mcp_code.replace_text

osworld_mcp_os.get_text_scale

osworld_mcp_code.search_text

osworld_mcp_code.add_folder

osworld_mcp_os.git_operation

osworld_mcp_vlc.get_settings

osworld_mcp_vlc.set_settings

80

osworld_mcp_vlc.get_playlist

osworld_mcp_os.search_files

osworld_mcp_os.get_volume

osworld_mcp_os.set_volume

osworld_mcp_code.goto_file

osworld_mcp_os.restore_file

100

filesystem_create_directory

filesystem_read_media_file

filesystem_directory_tree

filesystem_read_text_file

osworld_mcp_vlc.pause

filesystem_list_directory

filesystem_get_file_info

filesystem_search_files

60

osworld_mcp_vlc.play

filesystem_move_file

filesystem_write_file

filesystem_read_file

filesystem_edit_file

40

git_git_commit

50

git_git_status

git_git_add

20

0

0

Tool Name

0 1 2 3 > 4

The total number of available tool invocations per task

- Figure 4: (a) illustrates the total number of times each tool was invoked by the five models in a single OSWorld-MCP evaluation. (b) depicts the distribution of our 158 high-quality tools across different usage scenarios for various OSWorld environments. Due to software version constraints within OSWorld, MCP servers were not developed for GIMP and Thunderbird. (c) presents the distribution of the total available tool invocations per task.

Tool-Beneficial Tasks, which include tasks for which at least one available tool can improve efficiency (250 tasks), and Non-Tool-Beneficial Tasks, which include tasks for which no available tool improves efficiency (111 tasks).

These findings demonstrate that the tools in OSWorld-MCP are genuinely relevant to real-world needs and are not artificially tailored for specific benchmark tasks. Furthermore, many tasks can be completed more efficiently through multiple tool invocations, requiring models to select appropriate tools and to exhibit strong decision-making capabilities. This confirms the soundness of our tool design and underscores the challenging nature of the benchmark.

- 3.4 METRICS

Building on the evaluation of GUI operation capabilities, we introduce three metrics in OSWorldMCP to assess a multimodal agent’s tool invocation and decision-making abilities: Task Accuracy, Tool Invocation Rate, and Average Completion Steps.

Task Accuracy. Similar to OSWorld, we use accuracy as an overall performance indicator to measure how well a multimodal agent completes tasks. This metric jointly reflects the agent’s decisionmaking ability, tool invocation skills, and GUI interaction performance.

Tool Invocation Rate (TIR). As described in Section 3.2, human reviewers classify each task into one of two categories: Tool-Beneficial Tasks or Non-Tool-Beneficial Tasks.Let Nt be the total number of Tool-Beneficial Tasks, and nt the number of such tasks in which the agent invoked a tool and successfully completed the task during evaluation. Let Ng be the total number of NonTool-Beneficial Tasks, and ng the number of such tasks in which the agent did not invoke a tool and successfully completed the task. We define TIR as:

TIR = (nt + ng)/(Nt + Ng) (1) TIR can effectively indicate the agent’s ability to decide when a tool should or should not be invoked. Average Completion Steps (ACS). This metric measures the average number of steps an agent takes to complete a task. For N tasks, if the number of execution steps for task i is Si, the Average Completion Steps is computed as:

N

ACS =

Si/N (2)

i=1

ACS reflects decision-making efficiency: the more accurate the decisions, the higher the rate of correct tool usage, and the more frequently the agent selects efficient tools, the lower ACS will be.

- 4 EXPERIMENTS 4.1 SETUP

We evaluate a series of state-of-the-art Large Multimodal Models (LMMs), including Qwen2.5VL-72B-Instruct (Bai et al., 2025), Qwen3-VL-Plus (QwenTeam, 2025), Seed1.5-VL (Guo et al.,

- 2025b), Claude 4 Sonnet (Anthropic, 2025), OpenAI o3 (OpenAI, 2025b), and Gemini-2.5-Pro (Comanici et al., 2025), by running a computer-use agent on real-world tasks in OSWorld-MCP involving nine different software applications. To facilitate comparison of performance differences across models, we standardized our evaluation using the GUI-Owl agent configuration. This may lead to some performance fluctuations for certain models under the original OSWorld configuration. Specifically, at each step, the core LMM performs visual perception of the current interface, generates a corresponding thought, and proposes the next action along with a reasoning summary. This reasoning history informs subsequent planning during task execution. Besides, we also evaluate a multi-agent framework Agent-S2.5 (Simular Research, 2025), and adopt OpenAI o3 as the main generation model with UI-TARS-1.5-72B as the grounding model. For each task, the agent is restricted to a fixed maximum number of steps to either complete the task or determine that it is infeasible. The temperature parameter is set to 1.0.

In the original OSWorld configuration, LMMs may only use a predefined set of 11 basic GUI operations, including key, type, mouse move, click, drag, right click, middle click, double click, scroll, wait and terminate, to complete tasks. With the introduction of MCP tools in our OSWorld-MCP, the LMM can, autonomously decide whether to invoke any MCP Tool or perform a GUI operation at each action step. In details, we employ the 158 high-quality MCP tools curated in Section

- 3.2. Since providing all 158 tools simultaneously would result in excessively long input contexts, we apply Retrieval-Augmented Generation (RAG) to select only the tools relevant to the current application. The model then chooses from these filtered MCP tools or GUI operations.
- 4.2 MAIN RESULTS

As shown in Table 1, we evaluate six advanced end-to-end models and one agent-based frameworks under both the original OSWorld configuration and OSWorld-MCP with our curated high-quality tools, using maximum step limits of 15 and 50 respectively. Due to fluctuations in the experimental results, we conducted three runs for each model or framework under each configuration. The results reported in Tables 1 are the averages over these three runs.

Among the end-to-end models, Claude 4 Sonnet achieves the highest accuracy in OSWorld-MCP for both step limits, reaching 35.3 (15 steps) and 43.3 (50 steps) respectively among LMMs. Claude 4 Sonnet also records the highest tool invocation rate in OSWorld-MCP. Seed-VL1.5 and Claude 4 Sonnet achieve the lowest average completion steps (ACS) at 15 and 50 steps respectively, with values of 10.2 and 20.1. Our results also reveal that current large multimodal models generally have low tool invocation rates in OSWorld-MCP: even the highest, achieved by Claude 4 Sonnet, is only 36.3 percent, while the lowest, from Qwen2.5-VL-72B-Instruct, is 10.9 percent. This shows that these models still face challenges in invoking tools both correctly and efficiently. For the multiagent frameworks, Agent-S2.5 achieves an accuracy of 42.1 at the 15-step limit and 49.5 at the 50-step limit, confirming its effectiveness in handling challenging tasks.

Impact of Tool Invocation on Accuracy and Efficiency. A comparison with the GUI-only setting shows that, with the exception of Qwen2.5-VL, all six end-to-end models and both agent frameworks achieve higher accuracies and lower ACS after the introduction of MCP tools. Qwen2.5-VL, despite achieving a slight accuracy improvement, shows increased ACS. It indicates poor tool invocation capability and weak decision-making ability, leading to longer average completion times.

Among the remaining models and frameworks, Gemini-2.5-Pro exhibits the most significant improvement. At 15 steps, its overall accuracy rises from 7.4 to 20.5, while ACS decreases from 13.8 to 11.4. Furthermore, a comparison of Tool-Beneficial and Non-Tool-Beneficial Tasks shows that, except for Qwen2.5-VL-72B-Instruct, all models demonstrate substantial accuracy improvement in Tool-Beneficial Tasks after the introduction of MCP tools. The largest gain is observed in Gemini-2.5-Pro at 15 steps, where accuracy increases from 6.3 to 24.9. In addition, accuracy changes for Non-Tool-Beneficial Tasks are minor. Two possible factors may explain these results: First, Non-Tool-Beneficial Tasks refer only to tasks without efficiency-enhancing tools, yet they may

Table 1: Performance on OSWorld-MCP (15 Steps).

Tool-Beneficial Tasks Non-Tool-Beneficial Tasks Overall Accuracy TIR ACS Accuracy TIR ACS Accuracy TIR ACS

Agent Model Actions Steps

Open Models

GUI 15 10.1 - 13.0 14.4 - 12.9 11.4 - 13.0

+ MCP 15 14.7 4.6↑ 11.6 13.6 0.6↑ 18.3 3.9↑ 16.5 13.1 0.2↑ 15.8 4.4↑ 13.1 13.5 0.5↑ GUI 50 12.3 - 30.6 17.7 - 30.4 13.9 - 30.5

Qwen2.5-VL

+ MCP 50 11.7 0.6↓ 7.3 38.6 8.0↑ 21.6 3.9↑ 18.9 34.2 3.8↑ 14.8 0.9↑ 10.9 37.2 6.7↑

GUI 15 24.5 - 11.5 27.3 - 11.7 25.4 - 11.6

+ MCP 15 30.5 6.0↑ 23.1 10.2 1.5↓ 33.0 5.7↑ 27.6 11.1 0.6↓ 31.3 5.9↑ 24.5 10.5 1.1↓ GUI 50 31.9 - 25.5 38.0 - 25.7 33.8 - 25.6

Qwen3-VL

+ MCP 50 37.0 5.1↑ 27.7 20.2 5.3↓ 43.8 5.8↑ 33.3 23.3 2.4↓ 39.1 5.3↑ 29.5 21.1 4.5↓ Proprietary Models

GUI 15 6.3 - 13.7 9.8 - 14.1 7.4 - 13.8

+ MCP 15 24.9 18.6↑ 20.4 10.2 3.5↓ 9.9 0.1↑ 8.7 13.9 0.2↓ 20.5 13.1↑ 16.8 11.4 2.4↓ GUI 50 11.5 - 39.7 17.5 - 41.7 13.3 - 40.3

Gemini-2.5-Pro

+ MCP 50 29.5 18.0↑ 23.2 25.2 14.5↓ 21.6 4.1↑ 17.7 40.2 1.5↓ 27.2 13.9↑ 21.5 29.7 10.6↓

GUI 15 8.5 - 13.8 7.8 - 14.4 8.3 - 14.0

+ MCP 15 25.4 16.9↑ 21.3 10.5 3.3↓ 9.1 1.3↑ 6.3 14.1 0.3↓ 20.4 12.1↑ 16.7 11.6 2.4↓ GUI 50 10.7 - 43.9 17.4 - 46.8 12.8 - 44.8

OpenAI o3

+ MCP 50 28.9 18.2↑ 24.8 27.0 16.9↓ 16.8 0.6↓ 12.3 43.7 3.1↓ 25.2 12.4↑ 21.0 32.1 12.7↓

GUI 15 27.3 - 10.6 29.3 - 11.5 27.9 - 10.9

+ MCP 15 31.4 4.1↑ 21.7 9.6 1.0↓ 33.3 4.0↑ 32.7 11.5 0.0− 32.0 4.1↑ 25.1 10.2 0.7↓ GUI 50 31.2 - 22.4 40.2 - 26.8 34.0 - 23.8

Seed1.5-VL

+ MCP 50 36.1 4.9↑ 22.7 20.8 1.6↓ 43.6 3.4↑ 43.2 27.8 1.0↑ 38.4 4.4↑ 29.0 23.0 0.8↓

GUI 15 29.0 - 11.8 32.9 - 12.0 30.2 - 11.9

+ MCP 15 35.6 6.6↑ 29.7 9.8 2.0↓ 34.5 1.6↑ 30.9 11.7 0.3↓ 35.3 5.1↑ 30.0 10.4 1.5↓ GUI 50 38.2 - 24.2 44.1 - 25.6 40.1 - 24.7

Claude-4-Sonnet

+ MCP 50 42.4 4.2↑ 35.3 18.8 5.4↓ 45.5 1.4↑ 38.7 22.8 2.8↓ 43.3 3.2↑ 36.3 20.1 3.6↓ Multi-agent Models

GUI 15 35.8 - 11.4 38.6 - 11.2 36.7 - 11.3

+ MCP 15 42.9 7.1↑ 28.1 9.6 1.8↓ 40.4 1.8↑ 34.2 10.9 0.3↓ 42.1 5.4↑ 30.0 10.0 1.3↓ GUI 50 46.9 - 21.1 47.3 - 8.4 47.1 - 20.2

Agent-S2.5

+ MCP 50 49.4 2.5↑ 32.9 16.3 4.8↓ 49.6 2.3↑ 40.5 18.7 0.3↑ 49.5 2.4↑ 35.3 17.0 3.2↓

still include tools that, while not improving efficiency, have a positive effect on task completion. Such tools can make it easier for the model to solve the problem, thereby increasing Acc. Second, for tasks that contain no tools beneficial to the task at all, the tools provided can help the model rule out irrelevant solution paths, making it easier to execute the task along the correct path and thus improving task Accuracy. This also explains why ACS often decreases on Non-Tool-Beneficial Tasks. From these results, we derive the following conclusion:

Finding 1: MCP tools significantly enhance LMMs’ performance in computer-use, improving accuracy and reducing completion steps for most models. The effectiveness varies across different LMMs, indicating disparities in tool utilization capabilities.

[Figure 11]

- Figure 5: (a): The relationships between TIR, Acc, and ACS. (b): The performance of Claude-4Sonnet across task sets with different numbers of available tools. GM: GUI + MCP , G: GUI Only.

Impact of Tool Invocation Rate on Accuracy and Average Completion Steps. Our experiments reveal that, across different models, the tool invocation rate (TIR) and accuracy (Acc) generally exhibit a positive correlation, whereas Average Completion Steps (ACS) show no obvious correlation

Table 2: Performance of Gemini-2.5-Pro on OSWorld-MCP under different configurations.

Tool-Beneficial Tasks Non-Tool-Beneficial Tasks Overall Accuracy TIR ACS Accuracy TIR ACS Accuracy TIR ACS

Agent Model Settings

Base (MCP) 24.9 20.4 10.2 9.9 8.7 13.9 20.5 16.8 11.4 w/ Tools Shuffle 24.7 0.2↓ 19.2 1.2↓ 10.4 0.2↑ 18.0 8.1↑ 17.1 8.4↑ 13.9 0.0− 22.7 2.2↑ 18.6 1.8↑ 11.5 0.1↑ w/o Tools RAG 18.0 6.9↓ 12.4 8.0↓ 8.7 1.5↓ 9.9 0.0− 9.9 1.2↑ 9.2 4.7↓ 15.5 5.0↓ 11.6 5.2↓ 8.8 2.6↓

Gemini-2.5-Pro

with TIR. We computed TIR, Acc, and ACS for each model under varying maximum step limits, across Tool-Beneficial Tasks, Non-Tool-Beneficial Tasks, and the entire task set. These results were aggregated into a single chart (Figure 5.(a)). As shown in the figure, for a given model, higher TIR values tend to correspond to higher accuracies, indicating a clear positive relationship between tool invocation and task success. Notably, this relationship remains stable regardless of differences in step limits or task sets. This strongly supports the soundness of our MCP tool design. In contrast, ACS does not show a clear correlation with TIR. Further analysis of ACS–TIR patterns across different task sets and step limits suggests that, under the same settings, an increase in TIR can sometimes coincide with a decrease in ACS. We hypothesize two possible reasons for this phenomenon: a). TIR reflects the proportion of correct tool invocations. A higher TIR indicates a higher proportion of correct tool usage, which can enable the model to complete tasks more efficiently. b). The complexity of OSWorld-MCP tasks varies significantly, and the overall task completion rate remains relatively low. When the maximum step limit is raised, models tend to make more attempts in solving complex tasks, which can counteract or obscure the efficiency gains that come from correct tool usage.

Finding 2: Tool Invocation Rate (TIR) positively correlates with task accuracy, but its relationship with ACS is complex and non-linear, suggesting that the impact of tool use on efficiency depends on various factors including task difficulty and model-specific strategies.

Impact of the number of available tools. As shown in Figure 5.(b), we conducted experiments using the best-performing model from the previous evaluations, Claude 4 Sonnet, on the manually annotated task set described in Section 3.3. The tasks were grouped according to the number of available tools. For each configuration, we computed Acc, TIR, and ACS.

Firstly, in the results for the GUI-only configuration, we observe that as the number of available tools increases, Acc tends to decrease and ACS tends to increase. This indicates that task difficulty rises with the number of available tools. Secondly,in the results for the GUI plus MCP Tools configuration, we found that when the number of available tools was relatively small, Acc increased and ACS decreased. However, as the number of available tools grew, both Acc and TIR dropped sharply, and ACS gradually rose. A possible explanation is that with fewer available tools, the tasks are relatively easier, and the model is more likely to select the tools that are useful for the task, thereby completing it with higher accuracy and efficiency. When the number of available tools increases, the tasks become more complex. Even though more efficient tools might be available, these tasks often require multiple-tool combinations, making it difficult for the model to accurately select the most relevant tools, leading to reduced task accuracy and higher average completion steps. Thirdly, when comparing the GUI-only configuration with the GUI plus MCP Tools configuration, the latter consistently achieved lower ACS and higher accuracy overall. However, in cases where ACS was similar in both configurations, we found that the GUI plus MCP Tools configuration sometimes resulted in lower accuracy. We suspect that the use of MCP tool combinations is more challenging for LMMs than combining GUI operations.

Finding 3: MCP tools generally improve performance in complex tasks, but their efficacy diminishes in extremely complex scenarios requiring tool combinations. This indicates that combining multiple tools is more challenging than combining GUI operations.

- 4.3 ABLATION STUDY

To more comprehensively evaluate the tool invocation and decision-making capabilities of LMMs, we conducted a series of ablation studies. We selected Gemini-2.5-Pro, the model with the largest accuracy gain from MCP tools over the GUI-only setting, as the test subject for these experiments.

Impact of the Number of Callable Tools on Model Accuracy. In the default OSWorld-MCP setup, the set of tools available at each step is a subset obtained via Retrieval-Augmented Generation (RAG), filtered to match the current application in use. To investigate the impact of the number of callable tools on model performance, we removed the RAG filtering and allowed the model to choose freely from all 158 tools for each task. As shown in Table 2, removing RAG led to a noticeable performance drop: the overall Acc decreased from 20.5 to 15.5. We attribute this to the fact that descriptions of all 158 tools result in excessively long tool contexts, which markedly reduce the model’s tendency to use tools, thereby impairing accurate tool invocation. For Tool-beneficial Tasks, removing RAG reduced Acc from 24.9 to 18.0 and TIR from 20.4 to 12.4, indicating a pronounced decline in the model’s inclination to invoke tools for problem solving. In contrast, for Non-tool-beneficial Tasks, although Acc remained unchanged after removing RAG, TIR increased from 8.7 to 9.9, suggesting that the model became more inclined to employ the GUI rather than tools to solve the tasks.

Impact of Tool Description Order in Prompts. In the default OSWorld-MCP configuration, tool descriptions are provided to the LMM in alphabetical order. To investigate the impact of tool description ordering on model performance, we randomly shuffled the descriptions prior to passing them to the LMM and re-evaluated on OSWorld-MCP. As shown in Table 2, random ordering increased overall Acc from 20.5 to 22.7. This result indicates that the ordering of tool descriptions in the prompt has a substantial effect on model performance in OSWorld-MCP. While the model’s performance on Tool-beneficial Tasks was nearly unchanged before and after shuffling, differences were considerable for Non-tool-beneficial Tasks. This may be because, with fewer tools, the model tends to invoke the corresponding tool when one is available, whereas in the absence of available tools, the description order may implicitly suggest alternative solution strategies. For consistency in evaluation, tool descriptions are presented to the LMM in lexicographical order in OSWorld-MCP.

- 4.4 CASE STUDY

In order to demonstrate the tool invocation capability of the GUI Agent, we present a complex example in LibreOffice Calc: Copy the “Revenue” column along with the header to a new sheet named “Sheet2”. This task requires creating a new sheet and copying the specified column. The endto-end Gemini-2.5-Pro agent accomplishes this by utilizing tools for creating a sheet and copying data, then switching to Sheet2 to verify the operation, as illustrated in Figure 9. In contrast, the agent without MCP tools fails to select the specific column, demonstrating how tools can serve as a valuable complement to the agent’s capabilities. More cases are analyzed in Appendix A.1.

- 5 CONCLUSION

We introduce OSWorld-MCP, a fair and comprehensive benchmark for evaluating Large Multimodal Models (LMMs) in computer-use scenarios by jointly assessing graphical user interface (GUI) operation skills and MCP tool-invocation capabilities. Using an automated pipeline combined with meticulous manual validation, we construct a high-quality and diverse collection of MCP tools that supports realistic and balanced evaluation within the OSWorld framework. Experiments on eight state-of-the-art LMMs demonstrate that tool invocation can substantially improve robustness and efficiency, while also revealing trade-offs between usage frequency and overall performance. Looking ahead, extending OSWorld-MCP to more complex, dynamic, and collaborative environments, as well as incorporating human-centred evaluation metrics, will further advance the development of general-purpose, efficient, and trustworthy computer-use agents.

ETHICS STATEMENT

The OSWorld-MCP dataset is constructed entirely from publicly available software environments and tasks in OSWorld. All MCP tools included in the benchmark are either automatically generated and manually validated by the authors or selected from existing open-source MCP servers, ensuring that no proprietary, confidential, or personally identifiable information is included. The dataset contains only synthetic interaction records between multimodal agents and computer application environments; no human subject data are collected. We release OSWorld-MCP solely for research

and educational purposes to advance the evaluation of multimodal agents in realistic computeruse scenarios. Researchers using this dataset should comply with all applicable laws, institutional guidelines, and license terms. The authors bear responsibility for ensuring that the dataset is free of harmful or unethical content and that its use will not compromise privacy or security.

REFERENCES

Reyna Abhyankar, Qi Qi, and Yiying Zhang. Osworld-human: Benchmarking the efficiency of computer-use agents. arXiv preprint arXiv:2506.16042, 2025.

Anthropic. What is the model context protocol (mcp)? https://modelcontextprotocol. io/docs/getting-started/intro, 2024.

Anthropic. Claude Sonnet 4. https://www.anthropic.com/claude/sonnet, 2025. [Accessed 31-08-2025].

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Rogerio Bonatti, Dan Zhao, Francesco Bonacci, Dillon Dupont, Sara Abdali, Yinheng Li, Yadong Lu, Justin Wagle, Kazuhito Koishida, Arthur Bucker, et al. Windows agent arena: Evaluating multi-modal os agents at scale. arXiv preprint arXiv:2409.08264, 2024.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36:28091–28114, 2023.

Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H Laradji, Manuel Del Verme, Tom Marty, L´eo Boisvert, Megh Thakkar, Quentin Cappart, David Vazquez, et al. Workarena: How capable are web agents at solving common knowledge work tasks? arXiv preprint arXiv:2403.07718, 2024.

Abul Ehtesham, Aditi Singh, Gaurav Kumar Gupta, and Saket Kumar. A survey of agent interoperability protocols: Model context protocol (mcp), agent communication protocol (acp), agent-toagent protocol (a2a), and agent network protocol (anp). arXiv preprint arXiv:2505.02279, 2025.

Xuanqi Gao, Siyi Xie, Juan Zhai, Shqing Ma, and Chao Shen. Mcp-radar: A multi-dimensional benchmark for evaluating tool use capabilities in large language models. arXiv preprint arXiv:2505.16700, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025a.

Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025b.

Raghav Kapoor, Yash Parag Butala, Melisa Russak, Jing Yu Koh, Kiran Kamble, Waseem AlShikh, and Ruslan Salakhutdinov. Omniact: A dataset and benchmark for enabling multimodal generalist autonomous agents for desktop and web. In European Conference on Computer Vision, pp. 161– 178. Springer, 2024.

Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Chong Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Ruslan Salakhutdinov, and Daniel Fried. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks. arXiv preprint arXiv:2401.13649, 2024.

Thomas Kuntz, Agatha Duzan, Hao Zhao, Francesco Croce, Zico Kolter, Nicolas Flammarion, and Maksym Andriushchenko. Os-harm: A benchmark for measuring safety of computer use agents. arXiv preprint arXiv:2506.14866, 2025.

Hanyu Lai, Xiao Liu, Yanxiao Zhao, Han Xu, Hanchen Zhang, Bohao Jing, Yanyu Ren, Shuntian Yao, Yuxiao Dong, and Jie Tang. Computerrl: Scaling end-to-end online reinforcement learning for computer use agents. arXiv preprint arXiv:2508.14040, 2025.

Zhiwei Liu, Jielin Qiu, Shiyu Wang, Jianguo Zhang, Zuxin Liu, Roshan Ram, Haolin Chen, Weiran Yao, Shelby Heinecke, Silvio Savarese, et al. Mcpeval: Automatic mcp-based deep evaluation for ai agent models. arXiv preprint arXiv:2507.12806, 2025.

Xing Han L`u, Zdenˇek Kasner, and Siva Reddy. Weblinx: Real-world website navigation with multiturn dialogue. arXiv preprint arXiv:2402.05930, 2024.

Zhengxi Lu, Yuxiang Chai, Yaxuan Guo, Xi Yin, Liang Liu, Hao Wang, Han Xiao, Shuai Ren, Guanjing Xiong, and Hongsheng Li. Ui-r1: Enhancing efficient action prediction of gui agents by reinforcement learning. arXiv preprint arXiv:2503.21620, 2025a.

Zhengxi Lu, Jiabo Ye, Fei Tang, Yongliang Shen, Haiyang Xu, Ziwei Zheng, Weiming Lu, Ming Yan, Fei Huang, Jun Xiao, et al. Ui-s1: Advancing gui automation via semi-online reinforcement learning. arXiv preprint arXiv:2509.11543, 2025b.

Run Luo, Lu Wang, Wanwei He, and Xiaobo Xia. Gui-r1: A generalist r1-style vision-language action model for gui agents. arXiv preprint arXiv:2504.10458, 2025.

Guozhao Mo, Wenliang Zhong, Jiawei Chen, Xuanang Chen, Yaojie Lu, Hongyu Lin, Ben He, Xianpei Han, and Le Sun. Livemcpbench: Can agents navigate an ocean of mcp tools? arXiv preprint arXiv:2508.01780, 2025.

OpenAI. Introducing gpt-5. Technical report, OpenAI, 2025a. URL https://openai.com/ zh-Hans-CN/index/introducing-gpt-5/.

OpenAI. Openai o3 and o4-mini system card. Technical report, OpenAI, 2025b. URL https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/ o3-and-o4-mini-system-card.pdf. System Card.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326, 2025.

QwenTeam. Qwen3-vl: Sharper vision, deeper thought, broader action. https: //qwen.ai/blog?id=99f0335c4ad9ff6153e517418d48535ab6d8afef&from= research.latest-advancements-list, 2025.

Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, et al. Androidworld: A dynamic benchmarking environment for autonomous agents. arXiv preprint arXiv:2405.14573,

- 2024.

Simular Research. Agent-s. https://github.com/simular-ai/Agent-S/tree/main,

- 2025. [Accessed 23-09-2025].

Linxin Song, Yutong Dai, Viraj Prabhu, Jieyu Zhang, Taiwei Shi, Li Li, Junnan Li, Silvio Savarese, Zeyuan Chen, Jieyu Zhao, et al. Coact-1: Computer-using agents with coding as actions. arXiv preprint arXiv:2508.03923, 2025.

Shulin Tian, Ziniu Zhang, Liangyu Chen, and Ziwei Liu. Mmina: Benchmarking multihop multimodal internet agents. arXiv preprint arXiv:2404.09992, 2024.

Junyang Wang, Haiyang Xu, Haitao Jia, Xi Zhang, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent-v2: Mobile device operation assistant with effective navigation via multi-agent collaboration. Advances in Neural Information Processing Systems, 37:2686–2710, 2024a.

Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent: Autonomous multi-modal mobile device agent with visual perception. arXiv preprint arXiv:2401.16158, 2024b.

Zhenhailong Wang, Haiyang Xu, Junyang Wang, Xi Zhang, Ming Yan, Ji Zhang, Fei Huang, and Heng Ji. Mobile-agent-e: Self-evolving mobile assistant for complex tasks. arXiv preprint arXiv:2501.11733, 2025a.

Zhenting Wang, Qi Chang, Hemani Patel, Shashank Biju, Cheng-En Wu, Quan Liu, Aolin Ding, Alireza Rezazadeh, Ankit Shah, Yujia Bao, et al. Mcp-bench: Benchmarking tool-using llm agents with complex real-world tasks via mcp servers. arXiv preprint arXiv:2508.20453, 2025b.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh J Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094, 2024.

Tianbao Xie, Mengqi Yuan, Danyang Zhang, Xinzhuang Xiong, Zhennan Shen, Zilong Zhou, Xinyuan Wang, Yanxu Chen, Jiaqi Deng, Junda Chen, Bowen Wang, Haoyuan Wu, Jixuan Chen, Junli Wang, Dunjie Lu, Hao Hu, and Tao Yu. Introducing osworld-verified. xlang.ai, July 2025. URL https://xlang.ai/blog/osworld-verified.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Jiabo Ye, Xi Zhang, Haiyang Xu, Haowei Liu, Junyang Wang, Zhaoqing Zhu, Ziwei Zheng, Feiyu Gao, Junjie Cao, Zhengxi Lu, et al. Mobile-agent-v3: Foundamental agents for gui automation. arXiv preprint arXiv:2508.15144, 2025.

Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854, 2023.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

A APPENDIX

- A.1 MORE CASES

We present and analyze additional tool call examples to demonstrate how tool calls enable agents to complete diverse and complex tasks more efficiently and accurately.

- A.1.1 CASE: VSCODE SETTINGS The goal of the task is “Please help me modify VS Code setting to hide all ‘ pycache ’ folders in the explorer view”. Executed actions are as follows:

- 1. osworld mcp code.add files exclude

• pattern="**/ pycache "

- 2. computer use

- • action="left click"

- • x=1131.0
- • y=712.5

- 3. computer use

- • action="terminate"
- • status="success"

Figure 6 showcases the executing process. According to the evaluation program, the specific folder is hidden in the explorer view.

Initial state: reset() Step 1: osworld mcp code.add files exclude(p...

[Figure 12]

[Figure 13]

Step 2: computer use(action="left click", x=... Step 3: computer use(action="terminate", sta...

[Figure 14]

[Figure 15]

- Figure 6: Please help me modify VS Code setting to hide all “ pycache ” folders in the explorer view.

- A.1.2 CASE: FILESYSTEM OPERATION

The goal of the task is “I have an image of my receipt located in /home/user. I’m looking to transform it into a PDF file. Can you assist me with this task? Save the resulting PDF as ‘receipt.pdf’ on the desktop”. Executed actions are as follows:

- 1. osworld mcp os.convert image format

- • image path="/home/user/receipt.jpg"

- • output format="pdf"

- • output path="/home/user/Desktop/receipt.pdf"

- 2. filesystem list directory

• path="/home/user"

- 3. osworld mcp os.convert image format

- • image path="/home/user/OIP.jpg"

- • output format="pdf"

- • output path="/home/user/Desktop/receipt.pdf"

- 4. computer use

- • action="terminate"
- • status="success"

The agent first utilized the tool to convert an image to PDF format, but the tool response indicated the specified file or directory did not exist. By listing the files in the home directory path, the agent obtained the correct filename. The subsequent conversion attempt successfully produced the desired PDF file from the original image, as demonstrated in Figure 7.

Initial state: reset() Step 1: osworld mcp os.convert image format(...

[Figure 16]

[Figure 17]

Step 2: filesystem list directory(path="/hom... Step 3: osworld mcp os.convert image format(...

[Figure 18]

[Figure 19]

Step 4: computer use(action="terminate", sta...

[Figure 20]

- Figure 7: I have an image of my receipt located in /home/user. I’m looking to transform it into a PDF file. Can you assist me with this task? Save the resulting PDF as “receipt.pdf” on the desktop.

#### A.1.3 CASE: SLIDE DUPLICATION

The goal of the task is “Please duplicate the last two slides and insert the copies in alternating order, so the sequence becomes: original slide A, original slide B, then duplicated slide A, duplicated slide B”. Executed actions are as follows:

- 1. osworld mcp libreoffice impress.get slide count

- 2. osworld mcp libreoffice impress.duplicate slide

- • slide index=24

- • target position=25

- 3. osworld mcp libreoffice impress.get slide count

- 4. osworld mcp libreoffice impress.duplicate slide

- • slide index=24

- • target position=25

- 5. osworld mcp libreoffice impress.duplicate slide

- • slide index=23

- • target position=24

- 6. osworld mcp libreoffice impress.duplicate slide

- • slide index=24

- • target position=25

- 7. osworld mcp libreoffice impress.save

- 8. computer use

- • action="terminate"
- • status="success"

The agent initially determined the positions of the last two slides (indexed 23 and 24) through tool calls and attempted to directly duplicate the slides. However, the operation failed at steps 2 and

#### 4 due to the incorrect parameter of target position. After several attempts, the agent successfully duplicated the final two slides to positions 25 at step 5, and 26 at step 6. As shown in Figure 8, the agent ultimately completing the task.

Initial state: reset() Step 1: osworld mcp libreoffice impress.get ...

[Figure 21]

[Figure 22]

Step 2: osworld mcp libreoffice impress.dupl... Step 3: osworld mcp libreoffice impress.get ...

[Figure 23]

[Figure 24]

Step 4: osworld mcp libreoffice impress.dupl... Step 5: osworld mcp libreoffice impress.dupl...

[Figure 25]

[Figure 26]

Step 6: osworld mcp libreoffice impress.dupl... Step 7: osworld mcp libreoffice impress.save...

[Figure 27]

[Figure 28]

Step 8: computer use(action="terminate", sta...

[Figure 29]

- Figure 8: Please duplicate the last two slides and insert the copies in alternating order, so the sequence becomes: original slide A, original slide B, then duplicated slide A, duplicated slide B.

A.1.4 CASE: SPREADSHEET COPYING The goal of the task is “Copy the ‘Revenue’ column along with the header to a new sheet named ‘Sheet2’”. Executed actions are as follows:

- 1. osworld mcp libreoffice calc.copy cells between sheets

- • source range="A1:A20"

- • target sheet name="Sheet2"

- • target start cell="A1"

- 2. osworld mcp libreoffice calc.copy cells between sheets

- • source range="A1:A20"

- • target sheet name="Sheet2"

- • target start cell="A1"

- 3. osworld mcp libreoffice calc.get workbook info

- 4. osworld mcp libreoffice calc.copy cells between sheets

- • source range="A1:A20"

- • target sheet name="Sheet2"

- • target start cell="A1"

- 5. osworld mcp libreoffice calc.switch active sheet

• sheet name="Sheet2"

- 6. osworld mcp libreoffice calc.switch active sheet

• sheet name="Sheet1"

- 7. osworld mcp libreoffice calc.copy cells between sheets

- • source range="A1:A20"

- • target sheet name="Sheet2"

- • target start cell="A1"

- 8. osworld mcp libreoffice calc.switch active sheet

• sheet name="Sheet2"

- 9. computer use

- • action="terminate"
- • status="success"

According to Figure 9, the “Revenue” column has been copied to “Sheet2” successfully.

Initial state: reset() Step 1: copy cells between sheets(source range...

[Figure 30]

[Figure 31]

Step 2: copy cells between sheets(source range... Step 3: get workbook info()

[Figure 32]

[Figure 33]

Step 4: copy cells between sheets(source range... Step 5: switch active sheet(sheet name="Sheet2")

[Figure 34]

[Figure 35]

Step 6: switch active sheet(sheet name="Sheet1") Step 7: copy cells between sheets(source range...

[Figure 36]

[Figure 37]

Step 8: switch active sheet(sheet name="Sheet2") Step 9: computer use(action="terminate", sta...

[Figure 38]

[Figure 39]

Figure 9: Copy the “Revenue” column along with the header to a new sheet named “Sheet2”.

- A.2 MCP TOOL LIST We present the complete list of 158 MCP tools as follows.

Table 3: Complete List of MCP Tools

### No. MCP Server Domain MCP Tool

- 1 osworld mcp VS Code add folder Adds a folder to the last active window in VSCode

- 2 osworld mcp VS Code compare files Compares two files in VSCode

- 3 osworld mcp VS Code disable extension Disables a specific extension for the next instance of VSCode

- 4 osworld mcp VS Code goto file Opens a file at a specific line and character position

- 5 osworld mcp VS Code install extension Installs an extension or updates it in VSCode

- 6 osworld mcp VS Code launch vscode Launches Visual Studio Code with the specified file path or directory

- 7 osworld mcp VS Code list extensions Lists installed extensions in VSCode

- 8 osworld mcp VS Code remove folder Removes a folder from the last active window in VSCode

- 9 osworld mcp VS Code toggle sync Toggles synchronization on or off in VSCode

- 10 osworld mcp VS Code uninstall extension Uninstalls an extension from VSCode

- 11 osworld mcp VS Code update extensions Updates all installed extensions in VSCode to the latest version

- 12 osworld mcp VS Code add files exclude Add a glob pattern to the Files: Exclude setting

- 13 osworld mcp VS Code replace text Open VSCode search and replace panel, input search and replacement text, and execute replacement

- 14 osworld mcp VS Code search text Open VSCode search panel, input the search term, and execute search

- 15 osworld mcp VS Code set auto save delay Set the auto save delay in milliseconds

- 16 osworld mcp VS Code set color theme Change the editor color theme

- 17 osworld mcp VS Code set focus editor on break Set the Debug: Focus Editor On Break setting to true or false

- 18 osworld mcp VS Code set python diagnostics override Override severity for a specific Python analysis diagnostic rule

- 19 osworld mcp VS Code set word wrap column Set the number of columns at which the editor will word wrap

### No. MCP Server Domain MCP Tool

- 20 osworld mcp VS Code set wrap tabs Enable or disable wrap tabs in the editor

- 21 osworld mcp Google Chrome bookmark page Bookmarks the current page in the browser (equivalent to Ctrl+D)

- 22 osworld mcp Google Chrome bring back last tab Restores the last-closed tab in the browser (equivalent to Ctrl+Shift+T)

- 23 osworld mcp Google Chrome chrome open tabs setup Opens the entered URL

- 24 osworld mcp Google Chrome delete browsing data Opens the ’Clear browsing data’ dialog in the browser (equivalent to Ctrl+Shift+Del)

- 25 osworld mcp Google Chrome open appearance settings Opens the appearance settings page in the browser

- 26 osworld mcp Google Chrome open bookmarks Opens the bookmarks page in the browser

- 27 osworld mcp Google Chrome open extensions Opens the extensions management page in the browser

- 28 osworld mcp Google Chrome open password settings Opens the password/autofill settings page in the browser

- 29 osworld mcp Google Chrome open privacy settings Opens the privacy settings page in the browser

- 30 osworld mcp Google Chrome open profile settings Opens the profile settings page in the browser

- 31 osworld mcp Google Chrome open search engine settings Opens the search engine settings page in the browser

- 32 osworld mcp Google Chrome print

Opens the print dialog for the current browser page (equivalent to Ctrl+P)

- 33 osworld mcp LibreOffice Calc adjust column width Adjust the width of specified columns

- 34 osworld mcp LibreOffice Calc adjust row height Adjust the height of specified rows

- 35 osworld mcp LibreOffice Calc copy sheet Create a copy of an existing worksheet in the workbook

- 36 osworld mcp LibreOffice Calc create chart Create a chart in the active worksheet based on the specified data range

- 37 osworld mcp LibreOffice Calc create pivot table Create a pivot table in the active worksheet based on data from the source sheet

- 38 osworld mcp LibreOffice Calc env info Get content of the specified or active sheet, including its name, headers, and data

- 39 osworld mcp LibreOffice Calc export to csv Export the current document to a CSV file with the same path and name as the original file

- 40 osworld mcp LibreOffice Calc export to pdf Export the current document or specified sheets to PDF

### No. MCP Server Domain MCP Tool

- 41 osworld mcp LibreOffice Calc format range Apply formatting to the specified range in the active worksheet

- 42 osworld mcp LibreOffice Calc freeze panes Freeze rows and/or columns in the active worksheet

- 43 osworld mcp LibreOffice Calc get column data Get all data from the specified column

- 44 osworld mcp LibreOffice Calc get workbook info Get workbook information, including file path, file name, sheets and active sheet

- 45 osworld mcp LibreOffice Calc hide row data Hide rows that contain the specified value

- 46 osworld mcp LibreOffice Calc highlight range Highlight the specified range with the specified color

- 47 osworld mcp LibreOffice Calc merge cells Merge cells in the specified range

- 48 osworld mcp LibreOffice Calc rename sheet Rename a worksheet in the workbook

- 49 osworld mcp LibreOffice Calc reorder columns Reorder the columns in the sheet according to the specified order

- 50 osworld mcp LibreOffice Calc reorder sheets Change the order of worksheets in the workbook

- 51 osworld mcp LibreOffice Calc save

Save the current workbook to its current location

- 52 osworld mcp LibreOffice Calc set cell value Set a value to a specific cell in the active worksheet

- 53 osworld mcp LibreOffice Calc set column values Set values to the specified column, cannot be used to set formulas

- 54 osworld mcp LibreOffice Calc set number format Apply a specific number format to a range of cells

- 55 osworld mcp LibreOffice Calc set zoom level Adjust the zoom level of the current worksheet

- 56 osworld mcp LibreOffice Calc sort column Sort the data in the specified column in ascending or descending order

- 57 osworld mcp LibreOffice Calc switch active sheet Switch to the specified sheet and make it active. Creates new sheet if it doesn’t exist

- 58 osworld mcp LibreOffice Calc transpose range Transpose the specified range and paste it to the target cell

- 59 osworld mcp LibreOffice Calc copy cells between sheets Copy cells from a specified rectangular source range to another sheet

- 60 osworld mcp LibreOffice Calc fill blank down Forward-fills blank cells in specified columns with value from cell above

### No. MCP Server Domain MCP Tool

- 61 osworld mcp LibreOffice Calc format numbers to human readable Convert numeric values to human-readable strings (M for millions, B for billions)

- 62 osworld mcp LibreOffice Calc scale first sheet and export pdf Scales the first sheet to specified pages and exports to PDF

- 63 osworld mcp LibreOffice Writer save

Save the current document to its current location

- 64 osworld mcp LibreOffice Writer write text Write text at the current cursor position in the document

- 65 osworld mcp LibreOffice Writer set color Changes the color of matched text in the document

- 66 osworld mcp LibreOffice Writer find and replace Finds and replaces text in the document

- 67 osworld mcp LibreOffice Writer set font Changes the font of text in the document

- 68 osworld mcp LibreOffice Writer set line spacing Sets the line spacing for specified paragraphs

- 69 osworld mcp LibreOffice Writer insert formula at cursor Inserts a formula at the current cursor position

- 70 osworld mcp LibreOffice Writer insert image at cursor Inserts an image at the current cursor position

- 71 osworld mcp LibreOffice Writer set font size Changes the font size of specified text

- 72 osworld mcp LibreOffice Writer export to pdf Exports the current document to PDF format

- 73 osworld mcp LibreOffice Writer set paragraph alignment Sets the text alignment for specified paragraphs

- 74 osworld mcp LibreOffice Writer set default font Sets the default font for new text without changing existing text

- 75 osworld mcp LibreOffice Writer add page numbers Adds page numbers to the document at specified position

- 76 osworld mcp LibreOffice Writer insert page break Inserts a page break at current cursor position

- 77 osworld mcp LibreOffice Writer env info Retrieve all paragraphs, truncate each to at most 500 characters

- 78 osworld mcp LibreOffice Impress configure auto save Enables or disables auto-save functionality

- 79 osworld mcp LibreOffice Impress delete content Deletes the specified textbox from a slide

- 80 osworld mcp LibreOffice Impress duplicate slide Creates a duplicate of a specific slide

- 81 osworld mcp LibreOffice Impress env info Get the content of the specified pages

- 82 osworld mcp LibreOffice Impress export to image Exports the current presentation or a specific slide to an image file

- 83 osworld mcp LibreOffice Impress get slide count Gets the total number of slides in the current presentation

### No. MCP Server Domain MCP Tool

- 84 osworld mcp LibreOffice Impress go to slide Navigates to a specific slide in the presentation

- 85 osworld mcp LibreOffice Impress insert file Insects a video or audio file into the current or specified slide

- 86 osworld mcp LibreOffice Impress insert image Inserts an image to a specific slide

- 87 osworld mcp LibreOffice Impress position box Positions a textbox or image on a slide at a specific location

- 88 osworld mcp LibreOffice Impress save

Save the current presentation to its current location

- 89 osworld mcp LibreOffice Impress save as Saves the current document to a specified location

- 90 osworld mcp LibreOffice Impress set background color Sets the background color for the specified textbox

- 91 osworld mcp LibreOffice Impress set slide background Sets the background color or image for a specific slide or all slides

- 92 osworld mcp LibreOffice Impress set slide font Sets the font style for all text elements in a specific slide

- 93 osworld mcp LibreOffice Impress set slide orientation Changes the orientation of slides between portrait and landscape

- 94 osworld mcp LibreOffice Impress set style Sets the style properties for the specified textbox

- 95 osworld mcp LibreOffice Impress set text color Sets the text color for the specified textbox

- 96 osworld mcp LibreOffice Impress set text strikethrough Applies or removes strike-through formatting to text

- 97 osworld mcp LibreOffice Impress set textbox alignment Sets the text alignment for the specified textbox

- 98 osworld mcp LibreOffice Impress write text writes text to a specific textbox on a slide

- 99 osworld mcp LibreOffice Impress convert to docx Transfers all text slide-by-slide into a new Writer document

- 100 osworld mcp OS change text scale Changes the text-scaling factor and returns the previous value

- 101 osworld mcp OS configure auto lock Configures the GNOME automatic-lock behaviour

- 102 osworld mcp OS copy matching files with hierarchy Copies all files matching a pattern preserving directory hierarchy

- 103 osworld mcp OS get do not disturb status Returns the current Do Not Disturb state

### No. MCP Server Domain MCP Tool

- 104 osworld mcp OS get text scale Returns the current GNOME text-scaling factor

- 105 osworld mcp OS get trash directory Returns the absolute path to the user Trash ’files’ directory

- 106 osworld mcp OS get volume Returns the current output volume (percentage)

- 107 osworld mcp OS open shell Open a new terminal directly

- 108 osworld mcp OS rename directory Safely renames a folder on the local filesystem

- 109 osworld mcp OS restore file Restore the specified file from the user Trash back to its original location

- 110 osworld mcp OS search files Recursively search all files under the given folder

- 111 osworld mcp OS set default terminal size Persists the given number of columns × rows as the default GNOME-Terminal window size

- 112 osworld mcp OS set do not disturb Enable or disable the GNOME/Ubuntu Do Not Disturb mode

- 113 osworld mcp OS set volume Sets the output volume of the default PulseAudio / PipeWire sink

- 114 osworld mcp OS convert image format Convert an input image file to a specified format

- 115 osworld mcp OS ffmpeg video to gif Extract a portion of a video file and save it as an animated GIF

- 116 osworld mcp OS git operation Execute a git command (clone, add, commit, pull, push, etc.)

- 117 osworld mcp OS git set user info Configure git user.name and user.email

- 118 osworld mcp OS remove image background Remove the background of an input image and save with transparency

- 119 osworld mcp OS calculator

Simple calculator.

- 120 osworld mcp VLC add to playlist Adds a media file to the VLC playlist

- 121 osworld mcp VLC get current time Gets the current playback time position in seconds

- 122 osworld mcp VLC get media duration Gets the total duration of the currently playing media file in seconds

- 123 osworld mcp VLC get media files Gets the media files for the specified path

- 124 osworld mcp VLC get playlist Gets the current VLC playlist with track information

### No. MCP Server Domain MCP Tool

- 125 osworld mcp VLC get settings Gets the current settings of the VLC player

- 126 osworld mcp VLC next

Switches to the next media item in the VLC playlist

- 127 osworld mcp VLC pause

Pauses the currently playing media in VLC player

- 128 osworld mcp VLC play

Starts playing the current media in VLC player

- 129 osworld mcp VLC previous

Switches to the previous media item in the VLC playlist

- 130 osworld mcp VLC set settings Sets the settings for the VLC player

- 131 osworld mcp VLC toggle fullscreen Toggles fullscreen mode for the currently playing video

- 132 filesystem OS read file Read the complete contents of a file as text (DEPRECATED)

- 133 filesystem OS read text file Read the complete contents of a file as text with encoding handling

- 134 filesystem OS read media file Read an image or audio file as base64 encoded data

- 135 filesystem OS read multiple files Read the contents of multiple files simultaneously

- 136 filesystem OS write file Create a new file or completely overwrite an existing file

- 137 filesystem OS edit file Make line-based edits to a text file

- 138 filesystem OS create directory Create a new directory or ensure a directory exists

- 139 filesystem OS list directory Get a detailed listing of all files and directories in a specified path

- 140 filesystem OS list directory with sizes Get a detailed listing with sizes of all files and directories

- 141 filesystem OS directory tree Get a recursive tree view of files and directories as JSON

- 142 filesystem OS move file Move or rename files and directories

- 143 filesystem OS search files Recursively search for files and directories matching a pattern

- 144 filesystem OS get file info Retrieve detailed metadata about a file or directory

### No. MCP Server Domain MCP Tool

- 145 filesystem OS list allowed directories Returns the list of directories that this server is allowed to access

- 146 git OS git status Shows the working tree status

- 147 git OS git diff unstaged Shows changes in the working directory that are not yet staged

- 148 git OS git diff staged Shows changes that are staged for commit

- 149 git OS git diff Shows differences between branches or commits

- 150 git OS git commit Records changes to the repository

- 151 git OS git add Adds file contents to the staging area

- 152 git OS git reset Unstages all staged changes

- 153 git OS git log Shows the commit logs

- 154 git OS git create branch Creates a new branch from an optional base branch

- 155 git OS git checkout Switches branches

- 156 git OS git show Shows the contents of a commit

- 157 git OS git init Initialize a new Git repository

- 158 git OS git branch List Git branches

- A.3 USE OF LLMS

Large language models (LLMs) are used solely to assist in the preparation of this manuscript. They help improve the clarity, coherence, and conciseness of the text, refine phrasing, and ensure that the language conforms to academic writing standards. All conceptual content, experimental design, data analysis, and conclusions are developed entirely by the authors.

