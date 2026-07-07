ComfyUI-R1: Exploring Reasoning Models for Workflow Generation

ZHENRAN XU∗, Harbin Institute of Technology (Shenzhen), China YIYU WANG∗, Alibaba International Digital Commerce, China XUE YANG, Alibaba International Digital Commerce, China LONGYUE WANG, Alibaba International Digital Commerce, China WEIHUA LUO, Alibaba International Digital Commerce, China KAIFU ZHANG, Alibaba International Digital Commerce, China BAOTIAN HU†, Harbin Institute of Technology (Shenzhen), China MIN ZHANG, Harbin Institute of Technology (Shenzhen), China

arXiv:2506.09790v1[cs.CL]11Jun2025

Node Selection: I will choose the nodes ['CLIPTextEncode', 'ControlNetLoader', ...]. Workflow Planning:

instantid_1 = InstantIDModelLoader( instantid_file="ip-adapter.bin") control_net_2 = ControlNetLoader(control_net_name="instan tid/diffusion_pytorch_model.safetensors") faceanalysis_3 = InstantIDFaceAnalysis(provider="CUDA") image_4, mask_4 = LoadImage(image=”image.jpg")

Design a workflow that creates a high-resolution image of a woman while preserving the facial features of the reference image. The workflow should analyze the facial structure of the reference image and integrate it into the generation process, ensuring the output aligns with the identity and facial features of the reference.

[Figure 1]

[Figure 2]

[Figure 3]

- 1. The workflow begins by loading an InstantID model (`InstantIDModelLoader`) and preparing a face analysis model (`InstantIDFaceAnalysis`);
- 2. The reference image is loaded via the `LoadImage` node, and its facial keypoints are extracted with `FaceKeypointsPreprocessor`;
- 3. The conditioned model undergoes sampling

Coding

Thinking

.... image_13 = VAEDecode(samples=latent_12, vae=vae_5) ...

User Instruction

Chain-of-Thought Reasoning

Code Representation of Workflow

[Figure 4]

{1": { "inputs": { "instantid_file":

[Figure 5]

[Figure 6]

"ip-adapter.bin" }, "class_type":

Load onto Canvas & Execute

"InstantIDModelLoader"

}, ...}

Workflow Visualization & Execution Result

Workflow JSON

Fig. 1. We introduce ComfyUI-R1, a large reasoning model for automated workflow generation. Given a user instruction, ComfyUI-R1 performs long chain-of-thought reasoning to generate a code representation of a ComfyUI workflow. The generated workflow adheres to the correct format, executes successfully, and produces an image that aligns with the user’s instruction. ComfyUI-R1 is integrated in https://github.com/AIDC-AI/ComfyUI-Copilot.

AI-generated content has evolved from monolithic models to modular workflows, particularly on platforms like ComfyUI, enabling customization in creative pipelines. However, crafting effective workflows requires great expertise to orchestrate numerous specialized components, presenting a steep learning curve for users. To address this challenge, we introduce ComfyUI-R1, the first large reasoning model for automated workflow generation. Starting with our curated dataset of 4K workflows, we construct long chain-of-thought (CoT) reasoning data, including node selection, workflow planning, and code-level workflow representation. ComfyUI-R1 is trained through a two-stage framework: (1) CoT fine-tuning for cold start, adapting models to the ComfyUI domain; (2) reinforcement learning for incentivizing reasoning capability, guided by a fine-grained rule-metric hybrid reward, ensuring format validity, structural integrity, and node-level fidelity. Experiments show that our 7B-parameter model achieves a 97% format validity rate, along with high pass rate, node-level and graph-level F1 scores, significantly surpassing prior state-of-the-art methods that employ leading closed-source models such as GPT-4o and Claude series. Further analysis highlights the critical role of the reasoning process and the advantage of transforming

∗Both authors contributed equally to this research. †Corresponding author.

Authors’ Contact Information: Zhenran Xu, Harbin Institute of Technology (Shenzhen), Shenzhen, China, xuzhenran@stu.hit.edu.cn; Yiyu Wang, Alibaba International Digital Commerce, Hangzhou, China, wangyiyu18@mails.ucas.ac.cn; Xue Yang, Alibaba International Digital Commerce, Hangzhou, China, yx9966@126.com; Longyue Wang, Alibaba International Digital Commerce, Hangzhou, China, vincentwang0229@ gmail.com; Weihua Luo, Alibaba International Digital Commerce, Hangzhou, China, weihua.luowh@alibaba-inc.com; Kaifu Zhang, Alibaba International Digital Commerce, Hangzhou, China, kaifu.zkf@alibaba-inc.com; Baotian Hu, Harbin Institute of Technology (Shenzhen), Shenzhen, China, hubaotian@hit.edu.cn; Min Zhang, Harbin Institute of Technology (Shenzhen), Shenzhen, China, zhangmin2021@hit.edu.cn.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

SA Conference Papers ’25, Hong Kong, China © 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/2018/06 https://doi.org/XXXXXXX.XXXXXXX

workflows into code. Qualitative comparison reveals our strength in synthesizing intricate workflows with diverse nodes, underscoring the potential of long CoT reasoning in AI art creation.

### CCS Concepts: • Information systems → Multimedia content creation;

• Theory of computation → Reinforcement learning. Additional Key Words and Phrases: ComfyUI, large reasoning model, automated workflow generation ACM Reference Format:

Zhenran Xu, Yiyu Wang, Xue Yang, Longyue Wang, Weihua Luo, Kaifu Zhang, Baotian Hu, and Min Zhang. 2025. ComfyUI-R1: Exploring Reasoning Models for Workflow Generation. In Proceedings of SIGGRAPH Asia 2025 Conference Ppaers (SA Conference Papers ’25). ACM, New York, NY, USA, 11 pages. https://doi.org/XXXXXXX.XXXXXXX

- 1 Introduction

Recent advancements in large language models (LLMs) and image generation methods have democratized AI-generated content (AIGC) production [Dhariwal and Nichol 2021; Ho et al. 2020; Li et al.

- 2024; Wu and la Torre 2023]. Instead of relying on an end-to-end diffusion model, recent AIGC production has evolved into building advanced workflows to further enhance image quality [Gal et al. 2024]. Open-source frameworks such as ComfyUI [comfyanonymous 2023] are emerging as important tools for low-code AI workflow development. Serving over 4 million active users and backed by a vibrant community contributing 12K components (e.g., SDXL [Podell et al. 2023], LoRA [Ryu 2023], ControlNet [Zhang et al. 2023], and IPAdapter [Ye et al. 2023]), ComfyUI enables flexible workflow orchestration via drag-and-drop components for multimodal tasks such as text-to-image generation, face swapping, and video editing.

Despite its convenient interactive interface, ComfyUI presents challenges for new users who may lack the knowledge and reasoning skills required to build effective workflows [Huang et al. 2025]. The large number of nodes, along with their interdependencies and complex configurations, demands extensive documentation knowledge to go through. Moreover, the intricate connections within workflows require strong planning abilities to coordinate modules with diverse functionalities. As a result, constructing a well-designed workflow often requires considerable expertise, making automated end-to-end workflow generation and user onboarding crucial for broader adoption.

Recently, there has been increasing research on automated ComfyUI workflow generation. These approaches leverage state-of-theart LLMs (e.g, GPT-4o) to generate valid JSON-formatted workflows and break down the task into sub-tasks for multi-agent collaboration [Guo et al. 2025; Huang et al. 2025; Xue et al. 2024]. However, these methods still face several limitations: (1) Most studies focus only on text-to-image generation [Gal et al. 2024; Sobania et al. 2024], limiting their applicability to more conditional image and video generation tasks; (2) The generated JSON files often fail to transform into structured workflows or contain hallucinated nodes, possibly due to GPT-4o’s limited knowledge of newly introduced nodes and cumulative errors in multi-agent systems [Cemri et al. 2025; Wang

- et al. 2024]. To address these issues, our work expands the scope to a wide range of multimodal tasks and focuses on post-training foundational LLMs rather than relying on commercial APIs.

The success of long chain-of-thought (CoT) reasoning has attracted widespread attention, exemplified by models such as OpenAI

- o1 [OpenAI 2024], o3 [OpenAI 2025b], and DeepSeek-R1 [DeepSeekAI 2025], which have achieved remarkable accuracy in solving Olympiad-level math problems. Inspired by this progress, we introduce ComfyUI-R1, the first large reasoning model designed for automated workflow generation. Specifically, we begin by collecting a substantial volume of workflows and node data from ComfyUI community websites, covering a diverse set of tasks including image, video, and 3D generation and editing. After a comprehensive cleaning process, we retain 4K workflows out of the original 27K. Each example includes JSON-formatted and code representations
- of the workflow, along with a description of its functionality. Then, we generate long CoT reasoning sequences with a simulated node retrieval process, node selection, and workflow planning, ultimately producing the code representation of the workflow.

With the data prepared, ComfyUI-R1 is trained using a two-stage pipeline: (1) Cold-start CoT fine-tuning to adapt models to the ComfyUI domain, addressing the knowledge gap by incorporating retrieved node documentation; (2) reinforcement learning (RL) for incentivizing reasoning capability in LLMs. Unlike math tasks, which typically have a single correct answer and have been extensively studied in RL settings [de Winter et al. 2024; Li et al. 2025c], the reward design of workflow generation introduces multiple layers of complexity: the generated code should contain no hallucinated nodes, and conform to a valid format that can form a directed acyclic graph (DAG). Here we propose a fine-grained rule-metric hybrid reward. If there is anything wrong with the format validity, structural integrity, or node hallucinations, it will receive a negative reward to penalize such errors. Only when the above rules are satisfied, indicating that the workflow is well-formed, a positive reward will be given based on the correctness of node selection. We empirically demonstrate the effectiveness of this reward mechanism using Group Relative Policy Optimization [Shao et al. 2024].

Experiments on our test set show that, building upon Qwen2.5Coder [Hui et al. 2024], our 7B ComfyUI-R1 model outperforms previous state-of-the-art methods that rely on top-tier closed-source LLMs such as GPT-4o and the Claude series. It achieves a high format validity rate of 97%, a substantial improvement over the original Qwen2.5-Coder model (which achieves only 41%), showing the effectiveness of our two-stage training. ComfyUI-R1 also delivers superior performance in node-level and graph-level F1 scores. We further evaluate end-to-end retrieval and generation performance on ComfyBench [Xue et al. 2024], where ComfyUI-R1 surpasses the previous state-of-the-art ComfyAgent by an absolute margin of 11% in pass rate. Additional analysis underscores the importance of the reasoning process and the advantage of using code-based workflow representations instead of JSON. Qualitative comparisons demonstrate ComfyUI-R1’s ability to synthesize complex workflows with diverse nodes, showcasing the potential of long CoT reasoning in AI-driven content creation.

In summary, our main contributions are as follows:

- • We present ComfyUI-R1, the first large reasoning model for automated workflow generation to the best of our knowledge, incentivizing long CoT reasoning capability for complex workflow planning.
- • We curate a comprehensive workflow and node knowledge base and develop a two-stage training framework that combines supervised fine-tuning for cold start and reinforcement learning with our proposed rule-metric hybrid reward.
- • Extensive experiments show the superior performance of the 7B ComfyUI-R1 over previous state-of-the-art methods based on GPT-4o and the Claude series, underscoring the potential of long CoT reasoning in AI art creation.

- 2 Related Work

- 2.1 General Workflow Generation

General workflow generation has seen significant advancements with the advent of AI systems, such as Deep Research [OpenAI

- 2025a; Zheng et al. 2025], OpenAI o3 [OpenAI 2025b], which are capable of converting natural language descriptions into structured execution plans. Early efforts in this domain focus on straightforward human-designed workflows that constrain the planning process of large language or multimodal models (LLMs/LMMs) to prevent hallucinations [Guo et al. 2024]. Through such frameworks, multiple models can collaborate effectively and demonstrate collective intelligence in various domains, ranging from sophisticated reasoning tasks [Chen et al. 2024; Liu et al. 2025a; Wang et al. 2025a; Xu et al. 2023] and software development [Hong et al. 2024; Qian et al. 2023], to long-document translation [Wu et al. 2024] and video generation [Li et al. 2024; Xu et al. 2025a]. However, these manually crafted workflows are time-consuming to develop, heavily dependent on domain expertise, and often lack flexibility.

To address this challenge, recent studies have focused on training models or developing language agents to automatically generate workflows [Hu et al. 2025; Niu et al. 2025; Shang et al. 2025; Xu et al. 2024]. For example, WorkflowLLM [Fan et al. 2025] proposes a data-centric framework specifically designed to enhance the workflow orchestration capabilities of LLMs by fine-tuning on a large-scale dataset of complex, real-world workflows. AFlow [Zhang

- et al. 2025] reformulates the problem of workflow optimization as a search task. By employing Monte Carlo Tree Search (MCTS), AFlow efficiently explores the expansive action space, iteratively refining workflows based on tree-structured experiences and execution feedback. WorFEval [Qiao et al. 2025] is a recently released benchmark for assessing the quality of generated workflows, underscoring the need for models capable of generating complex graph structures. These studies collectively reflect a shift towards building systems that can manage more intricate and realistic workflow demands. The ComfyUI platform [comfyanonymous 2023] stands out as a particularly challenging testbed in this context, due to the large number of components involved in AI art creation workflows and the intricate connections between different modules with diverse functionalities [Xue et al. 2024].

- 2.2 ComfyUI-based Workflow Generation

Recent advances in AI-generated content have transitioned from end-to-end diffusion models [Dhariwal and Nichol 2021; Ho et al. 2020; Podell et al. 2023; Wang et al. 2025b] to more sophisticated workflows on open-source WebUI platforms such as ComfyUI. Users can easily construct workflows by connecting a series of blocks, such as large language models (LLMs) for refining input prompts, LoRAs trained to introduce specific artistic styles, improved latent decoders for finer details, super-resolution blocks, and more [Hu et al. 2021; Mañas et al. 2024; Ning et al. 2021]. Creating a welldesigned workflow and selecting appropriate nodes requires significant expertise [Xu et al. 2025b], where automated workflow generation comes into help.

ComfyUI-based workflow generation has emerged as a trending research topic in AI-driven multimodal content synthesis. For example, Xue et al. [2024] introduce a multi-agent ComfyAgent framework to design workflows in a step-by-step manner, consisting of three independent modules: Memory, Planner, and Actions. Similarly, ComfyGPT [Huang et al. 2025] introduces a self-optimizing system of four specialized LLM agents, focusing on node connection logic. Despite these advancements, the field remains in its early stages and faces two key limitations. Firstly, some existing approaches are limited to text-to-image generation [Gal et al. 2024; Sobania et al. 2024], restricting applicability to broader multimodal tasks. Secondly, the generated JSON workflows often suffer from structural inconsistencies or contain incorrect, hallucinated components, likely due to LLMs’ limited understanding of newly introduced nodes and error propagation in multi-agent frameworks [Guo et al. 2024; Huang et al. 2025]. To overcome these issues, our work expands the scope to multimodal tasks and explores the post-training of foundational LLMs rather than relying on proprietary APIs, with the goal of improving robustness and generalization in automated workflow generation.

- 2.3 Large Reasoning Models.

Since the release of OpenAI o1 [OpenAI 2024], large reasoning models have rapidly become a focal point in AI research, simulating “System 2” slow thinking processes [de Winter et al. 2024; Li et al. 2025b]. Early attempts to replicate o1 rely on supervised fine-tuning (SFT) using chain-of-thought traces, augmented with tree search algorithms and self-reflection strategies [Li et al. 2025a; Shinn et al. 2023; Zhao et al. 2024]. Inspired by the success of DeepSeek-R1 —which demonstrates strong performance on challenging benchmarks in mathematics, coding, and multi-hop question answering, rivaling or even surpassing human experts [DeepSeek-AI 2025]— recent studies have integrated the reasoning process directly into the optimization loop via group relative policy optimization (GRPO) [Shao et al. 2024]. This line of work includes re-implementations of R1 [Face 2025], extensions to multimodal settings [Yingzhe et al. 2025; Zhao et al. 2025; Zhou et al. 2025], applications in search [Song et al. 2025], tool integration [Li et al. 2025c; Qian et al. 2025], and medicine and finance domains [Liu et al. 2025b; Pan et al. 2025]. Overall, the posttraining landscape has progressed from SFT with curated rationales to sophisticated reinforcement learning (RL) pipelines. In this work, we explore reasoning models designed specifically for automated

data contains a large amount of noise, we conduct a comprehensive cleaning process to filter and standardize the data.

Instruction

Create a workflow that generates a detailed and stylized image of a girl in Hanfu, set in a night scene under the moonlight. ⋯⋯, The final image should be visually rich and high-quality, capturing the specified mood and aesthetic.

Node KB. Starting with a collection of 40K nodes, we first apply exact-match deduplication. After filtering out nodes that lack input or output parameters, 7,238 nodes remain. For nodes without structured documentation, we use Claude 3.5 to generate detailed documentation by analyzing their GitHub repositories. The final documentation is standardized to include the node type, usage, and the meanings of input and output parameters.

## Knowledge Base Here are the meta information to the nodes that may be used in the workflow:

- - Node: SaveImage <identifier> SaveImage </identifier> <description> Saves the input images to your ComfyUI output directory. </description>

<input> [{‘name’: ‘images’, ‘identifier’: ‘images’, ‘comfy_type’: ‘IMAGE’, ‘description’:

‘The images to ⋯⋯’,⋯⋯] </input> <output> [] </output>

- - (More nodes omitted) ## Format

Workflow KB. Workflows in ComfyUI are natively represented on the canvas in JSON format. We implement a parser to extract a Directed Acyclic Graph (DAG) from the JSON file, which is then converted into a sequence of Python-like function calls in topological order. Another parser is implemented to reverse the conversion, ensuring that workflows can be expressed through code and remain fully compatible with the ComfyUI environment.

- 1. Node Selection: First, reply with the names of the nodes you have chosen;
- 2. Workflow Design Principle: Then, explain your thought process and reasoning;
- 3. Workflow Construction: Finally, present the workflow as Python code.

Now, please provide your response according to the format specified above.

Response

Starting with the collected 27K workflows, we perform a series of filtering and refinement steps. First, we discard workflows that do not meet ComfyUI’s execution standards. Second, we remove exact duplicates. Third, we verify whether each workflow can be successfully transformed between JSON and code representations, retaining only those that support both conversions. Next, we further refine workflows by removing irrelevant information, such as “Anything Anywhere” node. Finally, we discard workflows that contain nodes not present in the above node KB. After this cleaning process, the resulting workflow KB contains 3,917 workflows, with an average of 21 nodes per workflow.

I will choose the nodes in ['CLIPTextEncode', 'CheckpointLoaderSimple', 'KSampler', ⋯⋯] </selected_nodes> <design_principle>

<selected_nodes>

The workflow begins by loading a pre-trained diffusion model checkpoint, which includes the model, CLIP, and VAE components. An empty latent image is created with the specified dimensions (1024x512 pixels). ⋯⋯ Finally, the VAEDecode node ⋯⋯.

</design_principle> <workflow>

model_1, clip_1, vae_1 = CheckpointLoaderSimple(ckpt_name=”xxx.xxx") #

CheckpointLoaderSimple conditioning_3 = CLIPTextEncode(clip=clip_1, text="A girl with red hair") # CLIPTextEnc ⋯⋯ image_12 = VAEDecode(samples=latent_11, vae=vae_1) # VAEDecode _ = PreviewImage(images=image_12) # PreviewImage

Since community-sourced content tends to focus more on installation instructions, there is often a lack of detailed functional descriptions for workflows. To address this, we leverage the multimodal understanding capabilities of GPT-4o, by prompting it with community-sourced texts and accompanying images that typically illustrate workflow outcomes. This approach helps fill in the gaps in informative functional descriptions. Therefore, each entry in our workflow KB includes a JSON-formatted and code-based representation of the workflow, along with an explanation of its functionality. As shown in Figure 6, our workflow KB encompasses a diverse set of tasks, including text-to-image generation, image editing, style transfer, video editing and generation.

</workflow>

Fig. 2. An example of SFT data.

workflow generation, propose a novel reward mechanism for RL in this task, and finally develop ComfyUI-R1.

- 3 ComfyUI-R1

ComfyUI-R1 is a large reasoning model designed for automated workflow generation. Given a user query about task description, based on a set of retrieved nodes, ComfyUI-R1 first performs stepby-step reasoning to select relevant nodes and plan the workflow. It then generates the code representation of a ComfyUI workflow. In this section, we first introduce the construction of ComfyUI-related knowledge bases (KBs) in Sec. 3.1. Then ComfyUI-R1 is trained using a two-stage pipeline: (1) Cold-start supervised fine-tuning (SFT) with distilled long chain-of-thought (CoT) data to adapt the model to the ComfyUI domain (Sec. 3.2); (2) Reinforcement learning (RL) for incentivizing reasoning capability via a fine-grained hybrid reward (Sec. 3.3) and the GRPO algorithm (Sec. 3.4).

3.2 Supervised Fine-tuning

We generate long CoT reasoning sequences through a simulated node retrieval process, node selection, and workflow planning, ultimately producing a code representation of the workflow. Each entry in the workflow KB is represented as a pair (𝑑𝑒𝑠𝑐,𝑐), where 𝑑𝑒𝑠𝑐 is the workflow description and 𝑐 is its corresponding code representation. Let V𝑔 denote the set of nodes in the ground-truth workflow, and VKB denote the full set of nodes in the KB. To simulate the node retrieval process, we construct a candidate node set Vcand by combining V𝑔 with a randomly sampled set of nodes Vrandom, where Vrandom ⊂ VKB \ V𝑔, and:

- 3.1 Knowledge Bases

We begin by constructing KBs of ComfyUI workflows and nodes. The original data is sourced from popular platforms for sharing generative resources, ComfyUI-related GitHub repositories, and the ComfyUI website, with NSFW content filtered out. Since the raw

Vcand = V𝑔 ∪ Vrandom, with |Vrandom| = ⌊0.8 · |V𝑔|⌋.

Based on the workflow description 𝑑𝑒𝑠𝑐 and its corresponding code representation 𝑐, we generate user instructions 𝑞𝑢𝑒𝑟𝑦 and a

rationale 𝑟 about the workflow design principle using Qwen-Max, Claude 3.5 and GPT-4o. We split 3,717 workflows into training and 200 for testing. This results in 11,140 training samples and 600 testing samples. As shown in Figure 2, each sample consists of the user instruction 𝑞𝑢𝑒𝑟𝑦, a set of candidate nodes Vcand, the workflow planning rationale 𝑟, the nodes in the gold workflow V𝑔, and the final code representation 𝑐.

During the SFT phase, the model input includes the 𝑞𝑢𝑒𝑟𝑦 and the candidate nodes Vcand. The output is a long CoT reasoning sequence, denoted as 𝑠 = [V𝑔,𝑟,𝑐]. The training objective can be formulated as:

## ∑︁𝑇

Pr 𝑠,𝑡 = 𝑖 desc, Vcand, 𝑠<𝑡 (1)

LSFT = −log

𝑡=1

where 𝑠<𝑡 is the sequence of tokens generated before time step 𝑡, Pr 𝑠,𝑡 = 𝑖 desc, Vcand, 𝑠<𝑡 is the probability that the LLM predicted token 𝑠,𝑡 in step 𝑡.

- 3.3 Reward Design

After adapting LLMs to the ComfyUI domain via SFT, we conduct RL to further enhance the reasoning capability. For an RL algorithm to be effective, a well-designed reward is essential. Unlike mathematical tasks in prior work—which typically have a single correct answer and allow for simple rule-based rewards—the workflow generation task presents several layers of complexity. For example, the generated code should avoid hallucinated nodes, and adhere to a valid format that forms a directed acyclic graph (DAG). To this end, here we propose a fine-grained rule-metric hybrid reward 𝑅final, including format reward 𝑅format, structure reward 𝑅DAG, node fidelity 𝑅fidelity and precision reward 𝑅correct. This reward formulation penalizes invalid formats, incorrect graph structures, and hallucinated nodes, while promoting accurate node selection.

Format reward (𝑅format). This component verifies whether the output response adheres to the expected structure, including a sequence ofreasoningstepsenclosedwithinthe “<selected_nodes>...</selected_ nodes>” and “<design_principle>...</design_principle>” tags. The final workflow code is wrapped within “<workflow>...</workflow>”. A format score of 0 is given if all required tags are present and their contents can be successfully extracted. Otherwise, a penalty is applied. The format reward function 𝑅format is defined as follows:

0 if all required fields appear −1 otherwise

(2)

𝑅format =

Structure reward (𝑅DAG). This reward assesses whether the parsed “<workflow>” section of the generated output forms a valid DAG, which is a fundamental requirement for ComfyUI workflows. A penalty is assigned if the structure is not a valid DAG.

0 if the structure is a valid DAG −1 otherwise

(3)

𝑅DAG =

Node fidelity reward (𝑅fidelity). This component penalizes the inclusion of hallucinated or inconsistent nodes during node selection and workflow generation. A penalty of -1 is applied in either of the following cases:

- (1) Invalid Nodes: Any node listed in the predicted “<selected_ nodes>” block V𝑝 is not present in the provided candidate nodes Vcand.
- (2) Inconsistent Nodes: The set of nodes listed in “<selected_ nodes>” block does not exactly match the set of nodes parsed from the ‘<workflow>” block.

𝑅fidelity = −1 if invalid or inconsistent nodes are detected 0 otherwise

(4)

Node Selection Accuracy (𝑅correct). This reward measures the overlap between generated node set V𝑝 and the ground truth node set V𝑔, reflecting the model’s ability to select the correct nodes for the workflow.

𝑅correct = |V𝑝 ∩ V𝑔|

− 1 (5)

|V𝑔|

Combining the above rewards. The total reward 𝑅final aggregates all the individual reward components using a veto-based mechanism. If any of 𝑅format, 𝑅DAG, or 𝑅fidelity is -1, indicating a fundamental error in format validity, structural integrity, or node fidelity, the total reward is immediately set to -1, preventing any positive reward for an invalid output. Otherwise, the total reward is computed based on the node selection accuracy.

𝑅final = −1 if 𝑅format = −1 or 𝑅DAG = −1 or 𝑅fidelity = −1

4+𝑅correct

4.0 otherwise

(6)

3.4 RL Training with GRPO

Based on the above hybrid reward, ComfyUI-R1 is trained using the Group Relative Policy Optimization (GRPO) algorithm [Shao et al. 2024]. At each training iteration, given an input text q (including the 𝑞𝑢𝑒𝑟𝑦 and the candidate nodes Vcand), a group of 𝐺 candidate outputs, {𝑠1,𝑠2, . . .,𝑠𝐺}, is sampled from the policy model 𝜋𝜃old. The advantage is computed based on the group of hybrid rewards {𝑟1,𝑟2, . . .,𝑟𝐺}, defined as

𝑟𝑖 − mean({𝑟1,𝑟2, . . .,𝑟𝐺}) std({𝑟1,𝑟2, . . .,𝑟𝐺})

, (7)

𝐴𝑖 =

The GRPO algorithm then optimizes the policy 𝜋𝜃 by maximizing the following objective function:

𝐽GRPO(𝜃) = E𝑞∼𝑃(𝑄), {𝑠𝑖}𝐺

𝑖=1∼𝜋𝜃old (𝑂|𝑞)

## ∑︁𝐺

1 𝐺

𝜋𝜃 (𝑠𝑖 | 𝑞) 𝜋𝜃old(𝑠𝑖 | 𝑞)

min

𝐴𝑖,

𝑖=1

𝜋𝜃 (𝑠𝑖 | 𝑞) 𝜋𝜃old(𝑠𝑖 | 𝑞)

, 1 − 𝜀, 1 + 𝜀 𝐴𝑖

clip

(8)

− 𝛽 𝐷KL 𝜋𝜃 𝜋ref

,

𝜋ref (𝑜𝑖 | 𝑞) 𝜋𝜃 (𝑜𝑖 | 𝑞)

𝜋ref (𝑜𝑖 | 𝑞) 𝜋𝜃 (𝑜𝑖 | 𝑞)

DKL 𝜋𝜃 ∥ 𝜋ref =

− log

− 1, (9)

where 𝜀 specifies clipping range, and 𝛽 is a hyperparameter controlling the weight of the Kullback–Leibler (KL) divergence penalty.

Table 1. Evaluation results of all baselines on our test set. The node-level and graph-level precision, recall and F1 scores, together with Format Validity rate, are reported. The best results are highlighted in bold.

Node-level Graph-level P R F1 P R F1

Methods Models Format Validity

Qwen2.5-Coder-7B-Instruct 0.25 0.18 0.12 0.14 0.08 0.07 0.08 Qwen2.5-Max 0.50 0.35 0.25 0.29 0.19 0.17 0.18 GPT-4o 0.89 0.62 0.42 0.50 0.33 0.28 0.30 Claude 3.5 Sonnet 0.93 0.58 0.48 0.52 0.37 0.36 0.37 Claude 3.7 Sonnet 0.81 0.44 0.43 0.43 0.27 0.29 0.28

Few-shot

Qwen2.5-Coder-7B-Instruct 0.41 0.30 0.18 0.22 0.12 0.09 0.10 Qwen2.5-Max 0.64 0.46 0.31 0.36 0.25 0.21 0.23 GPT-4o 0.92 0.66 0.42 0.50 0.33 0.27 0.29 Claude 3.5 Sonnet 0.97 0.70 0.49 0.57 0.41 0.36 0.38 Claude 3.7 Sonnet 0.90 0.57 0.48 0.51 0.36 0.35 0.35

CoT

ComfyAgent GPT-4o 0.47 0.26 0.20 0.21 0.11 0.10 0.10 SFT + GRPO ComfyUI-R1 0.97 0.67 0.58 0.62 0.52 0.51 0.51

- 4 Experiments

- 4.1 Data and Evaluation Metric

Our experiments are conducted on the test set described in Sec. 3.2. Each task input consists of a 𝑞𝑢𝑒𝑟𝑦 and a set of candidate nodes Vcand, and the expected output is an executable workflow 𝑐 that addresses the given query. Our evaluation setup is designed to assess the model’s ability to reason over pre-retrieved node information, rather than its retrieval capability. To explore end-to-end performance—including both node retrieval and workflow generation—we conduct an additional experiment in Sec. 4.5, where no candidate set Vcand is provided.

The first evaluation metric is Format Validity rate. It provides an initial check of the syntactic and structural correctness of the generated workflows. Specifically, it verifies whether all node names referenced in the workflow exist and whether the resulting structure forms a valid DAG.

Following the evaluation protocol in WorFEval [Qiao et al. 2025], we quantitatively assess the matching of the predicted and gold workflows. Specifically, we identify the longest node chain via Longest Increasing Subsequence (LIS)1 and the largest workflow subgraph via Maximum Common Induced Subgraph (MCIS)2. Based on these matches, we compute and report node-level and graphlevel precision, recall and F1 scores3.

- 4.2 Baselines

Following previous workflow generation studies [Huang et al. 2025; Xue et al. 2024], we adopt two effective prompting methods for advanced commercial LLMs (Qwen-Max, Claude 3.5, Claude 3.7 and GPT-4o), as well as Qwen2.5-Coder-7B-Instruct, which serves as the base model for our training.

- 1https://en.wikipedia.org/wiki/Longest_increasing_subsequence
- 2https://en.wikipedia.org/wiki/Maximum_common_induced_subgraph 3For detailed derivations and formal equations of the evaluation metrics, please refer to the original WorFEval paper. Our implementation follows its official GitHub repository.

Table 2. Ablation results of ComfyUI-R1 with different settings. “SFT + GRPO” indicates ComfyUI-R1. Two variants only conduct the SFT stage, either with code or JSON-formatted workflows. “FV” means format validity.

Node-level Graph-level P R F1 P R F1

Methods FV

SFT + GRPO 0.97 0.67 0.58 0.62 0.52 0.51 0.51 SFT only 0.95 0.64 0.57 0.60 0.48 0.49 0.48 SFT only (JSON) 0.92 0.62 0.55 0.57 0.45 0.46 0.45

- (1) Few-shot learning provides a set of code-formatted workflows in the prompt to utilize the in-context learning capabilities of LLMs for workflow generation.
- (2) Chain-of-Thought (CoT) is improved based on the above few-shot learning baseline, incorporating the CoT reasoning process into the few-shot demonstrations.

In addition, we include the multi-agent baseline ComfyAgent [Xue et al. 2024] based on its official GitHub implementation.

- 4.3 Implementation Details

We use the Qwen2.5-Coder-7B-Instruct as the backbone of ComfyUIR1. The SFT training stage is conducted on 8×80G NVIDIA A100 GPUs over 1 epoch, with a learning rate of 1e-5 and a batch size of 1 on each GPU. During RL training, we utilize 8×80G NVIDIA A100 GPUs to train for a total of 300 steps, with a learning rate of 1e-6 and a total batch size of 64. In Eq. 8, the number of group computations 𝐺 is set to 4, the clipping coefficient 𝜀 to 0.2, the KL penalty weight 𝛽 to 0.001. The maximum tokens are set to 32,768 for both training and inference.

- 4.4 Results

Table 1 presents the experimental results on our test set. In terms of node-level and graph-level F1 scores, our 7B ComfyUI-R1 model

User Instruction: Create an anime-style portrait of a nurse character with pink hair. The image should focus on the upper body, showing the character looking at the viewer with a smile while wearing a nurse cap. The final image should have high resolution and maintain a cartoon aesthetic style.

[Figure 7]

ComfyAgent Generated Workflows & Execution Result:

[Figure 8]

ComfyUI-R1 Generated Workflows & Execution Result:

[Figure 9]

[Figure 10]

- Fig. 3. Comparison between ComfyAgent and our ComfyUI-R1. The execution result of ComfyUI-R1 accurately adheres to the “anime-style” and “cartoon” attributes in the user instruction. In contrast, ComfyAgent fails to follow these stylistic guidelines.

Table 3. Evaluation results on ComfyBench. The numbers of GPT-4o methods are taken from the original paper [Xue et al. 2024].

outperforms few-shot, CoT prompting and multi-agent methods, which all rely on top-tier closed-source LLMs such as GPT-4o and the Claude series. Compared to the original Qwen2.5-Coder model, ComfyUI-R1 achieves substantial improvements across all metrics (e.g., format validity rate increases from 41% to 97%), demonstrating the effectiveness of our two-stage training strategy. When comparing performance under few-shot and CoT settings, all models show consistent improvement with CoT reasoning in their in-context demonstrations, highlighting the importance of reasoning process.

# Methods Pass Rate

GPT-4o + Few-shot 0.23 GPT-4o + CoT 0.28 GPT-4o + RAG 0.52 GPT-4o + ComfyAgent 0.56

ComfyUI-R1 0.67

Table 2 reports results from two variants of ComfyUI-R1: (1) an SFT-only version, and (2) a version that replaces code-based workflow representations with a JSON format proposed by Huang et al. [2025]. The results show that RL training further improves the already high 95% format validity rate of the SFT-only model, validating the effectiveness of RL in the workflow generation task. On the other hand, switching from code to JSON for workflow representation leads to decreased performance, likely because the JSON format carries less semantic and logical structure. Code-based representations are more compact and semantically rich, highlighting their advantage for this task.

measures the ratio of tasks where generated workflows can be successfully executed in the ComfyUI server.

Our retrieval process for obtaining candidate nodes Vcand is as follows: (1) Given an instruction, we retrieve the top 3 most semantically relevant workflows from the workflow KB based on OpenAI’s text-embedding-3-small. (2) We then aggregate the nodes from these retrieved workflows into a set, which forms Vcand. Next, the 𝑞𝑢𝑒𝑟𝑦 and candidate nodes are input into ComfyUI-R1 for workflow generation. As shown in Table 3, our model outperforms previous state-of-the-art, the GPT-4o-based ComfyAgent, by an absolute margin of 11%, showing the effectiveness of our training scheme tailored for the workflow generation task. Sec. 4.6 provides more qualitative comparison of ComfyAgent and our ComfyUI-R1.

- 4.5 Additional Experiments on ComfyBench

To evaluate the end-to-end retrieval and generation performance, we conduct experiments on ComfyBench [Xue et al. 2024] without providing candidate nodes Vcand. We report the pass rate, which

4.6 Case Study

We present a comparison between ComfyAgent and our ComfyUIR1. In Figure 3, the image style generated by ComfyUI-R1’s workflow aligns more closely with the user instruction compared to that produced by ComfyAgent. From the multi-image combination example in Figures 4 and 5, ComfyUI-R1 successfully loads and seamlessly blends the two input images. In contrast, ComfyAgent’s workflow loads the second image but fails to utilize it further, resulting in an incomplete output. This highlights the importance of effective workflow planning. From these examples, we observe that ComfyUI-R1’s workflows typically include more nodes than those of ComfyAgent, demonstrating ComfyUI-R1’s capability to synthesize complex, executable, and instruction-aligned workflows with diverse nodes.

- 5 Conclusion

We introduce ComfyUI-R1, the first large reasoning model for automated workflow generation on the ComfyUI platform. We begin by curating workflow and node knowledge bases to construct long chain-of-thought reasoning data. Following this, we employ supervised fine-tuning for cold start and reinforcement learning for incentivizing reasoning capability, guiding the model to generate structurally sound and executable workflows. Experimental results demonstrate that our 7B ComfyUI-R1 model significantly outperforms existing SOTA methods based on GPT-4o and Claude series, achieving a 97% format validity rate and superior node-level and graph-level F1 scores. Qualitative analysis highlights the model’s ability to synthesize complex workflows, showcasing the power of structured reasoning in AI-driven content creation. Future directions include exploring more fine-grained reward signals during training to better guide the intricate reasoning required for workflow generation.

References

Mert Cemri, Melissa Z. Pan, Shuyi Yang, Lakshya A. Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E. Gonzalez, and Ion Stoica. 2025. Why Do Multi-Agent LLM Systems Fail? arXiv:2503.13657 [cs.AI] https://arxiv.org/abs/2503.13657

Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chi-Min Chan, Heyang Yu, Yaxi Lu, Yi-Hsin Hung, Chen Qian, Yujia Qin, Xin Cong, Ruobing Xie, Zhiyuan Liu, Maosong Sun, and Jie Zhou. 2024. AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id=EHg5GDnyq1

comfyanonymous. 2023. ComfyUI. https://github.com/comfyanonymous/ComfyUI. Joost C. F. de Winter, Dimitra Dodou, and Yke Bauke Eisma. 2024. System 2 thinking

in OpenAI’s o1-preview model: Near-perfect performance on a mathematics exam. CoRR abs/2410.07114 (2024). doi:10.48550/ARXIV.2410.07114 arXiv:2410.07114 DeepSeek-AI. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via

Reinforcement Learning. arXiv:2501.12948 [cs.CL] https://arxiv.org/abs/2501.12948

Prafulla Dhariwal and Alexander Nichol. 2021. Diffusion Models Beat GANs on Image Synthesis. In Advances in Neural Information Processing Systems, M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (Eds.), Vol. 34. Curran Associates, Inc., 8780–8794. https://proceedings.neurips.cc/paper_files/ paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf

Hugging Face. 2025. open-r1. https://github.com/huggingface/open-r1. GitHub Repository.

Shengda Fan, Xin Cong, Yuepeng Fu, Zhong Zhang, Shuyan Zhang, Yuanwei Liu, Yesai Wu, Yankai Lin, Zhiyuan Liu, and Maosong Sun. 2025. WorkflowLLM: Enhancing Workflow Orchestration Capability of Large Language Models. In The Thirteenth International Conference on Learning Representations. https://openreview.net/forum? id=3Hy00Wvabi

Rinon Gal, Adi Haviv, Yuval Alaluf, Amit H. Bermano, Daniel Cohen-Or, and Gal Chechik. 2024. ComfyGen: Prompt-Adaptive Workflows for Text-to-Image Generation. arXiv:2410.01731 [cs.CV] https://arxiv.org/abs/2410.01731

Litao Guo, Xinli Xu, Luozhou Wang, Jiantao Lin, Jinsong Zhou, Zixin Zhang, Bolan Su, and Ying-Cong Chen. 2025. ComfyMind: Toward General-Purpose Generation via Tree-Based Planning and Reactive Feedback. arXiv:2505.17908 [cs.AI] https: //arxiv.org/abs/2505.17908

Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V. Chawla, Olaf Wiest, and Xiangliang Zhang. 2024. Large Language Model based Multi-Agents: A Survey of Progress and Challenges. arXiv:2402.01680 [cs.CL]

Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising Diffusion Probabilistic Models. In Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (Eds.), Vol. 33. Curran Associates, Inc., 6840–6851. https://proceedings.neurips.cc/paper_files/paper/2020/file/ 4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. 2024. MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum? id=VtmBAGCN7o

Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2021. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations.

Shengran Hu, Cong Lu, and Jeff Clune. 2025. Automated Design of Agentic Systems. In The Thirteenth International Conference on Learning Representations. https: //openreview.net/forum?id=t9U3LW7JVX

Oucheng Huang, Yuhang Ma, Zeng Zhao, Mingrui Wu, Jiayi Ji, Rongsheng Zhang, Zhipeng Hu, Xiaoshuai Sun, and Rongrong Ji. 2025. ComfyGPT: A SelfOptimizing Multi-Agent System for Comprehensive ComfyUI Workflow Generation. arXiv:2503.17671 [cs.MA] https://arxiv.org/abs/2503.17671

Binyuan Hui, Jian Yang, Zeyu Cui, Jiaxi Yang, Dayiheng Liu, Lei Zhang, Tianyu Liu, Jiajun Zhang, Bowen Yu, Keming Lu, Kai Dang, Yang Fan, Yichang Zhang, An Yang, Rui Men, Fei Huang, Bo Zheng, Yibo Miao, Shanghaoran Quan, Yunlong Feng, Xingzhang Ren, Xuancheng Ren, Jingren Zhou, and Junyang Lin. 2024. Qwen2.5Coder Technical Report. arXiv:2409.12186 [cs.CL] https://arxiv.org/abs/2409.12186

Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou. 2025a. Search-o1: Agentic search-enhanced large reasoning models. arXiv preprint arXiv:2501.05366 (2025).

Xuefeng Li, Haoyang Zou, and Pengfei Liu. 2025c. ToRL: Scaling Tool-Integrated RL. arXiv:2503.23383 [cs.CL] https://arxiv.org/abs/2503.23383

Yunxin Li, Zhenyu Liu, Zitao Li, Xuanyu Zhang, Zhenran Xu, Xinyu Chen, Haoyuan Shi, Shenyuan Jiang, Xintong Wang, Jifang Wang, Shouzheng Huang, Xinping Zhao, Borui Jiang, Lanqing Hong, Longyue Wang, Zhuotao Tian, Baoxing Huai, Wenhan Luo, Weihua Luo, Zheng Zhang, Baotian Hu, and Min Zhang. 2025b. Perception, Reason, Think, and Plan: A Survey on Large Multimodal Reasoning Models. arXiv:2505.04921 [cs.CV] https://arxiv.org/abs/2505.04921

Yunxin Li, Haoyuan Shi, Baotian Hu, Longyue Wang, Jiashun Zhu, Jinyi Xu, Zhen Zhao, and Min Zhang. 2024. Anim-Director: A Large Multimodal Model Powered Agent for Controllable Animation Video Generation. In SIGGRAPH Asia 2024 Conference Papers (SA ’24). Association for Computing Machinery, New York, NY, USA, Article 34, 11 pages. doi:10.1145/3680528.3687688

Yexiang Liu, Jie Cao, Zekun Li, Ran He, and Tieniu Tan. 2025a. Breaking Mental Set to Improve Reasoning through Diverse Multi-Agent Debate. In The Thirteenth International Conference on Learning Representations. https://openreview.net/forum? id=t6QHYUOQL7

Zhaowei Liu, Xin Guo, Fangqi Lou, Lingfeng Zeng, Jinyi Niu, Zixuan Wang, Jiajie Xu, Weige Cai, Ziwei Yang, Xueqian Zhao, Chao Li, Sheng Xu, Dezhi Chen, Yun Chen, Zuo Bai, and Liwen Zhang. 2025b. Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement Learning. arXiv:2503.16252 [cs.CL] https://arxiv.org/abs/2503.16252

Oscar Mañas, Pietro Astolfi, Melissa Hall, Candace Ross, Jack Urbanek, Adina Williams, Aishwarya Agrawal, Adriana Romero-Soriano, and Michal Drozdzal. 2024. Improving Text-to-Image Consistency via Automatic Prompt Optimization. arXiv:2403.17804 [cs.CV] https://arxiv.org/abs/2403.17804

Qian Ning, Weisheng Dong, Guangming Shi, Leida Li, and Xin Li. 2021. Accurate and Lightweight Image Super-Resolution With Model-Guided Deep Unfolding Network. IEEE Journal of Selected Topics in Signal Processing 15, 2 (2021), 240–252. doi:10.1109/ JSTSP.2020.3037516

Boye Niu, Yiliao Song, Kai Lian, Yifan Shen, Yu Yao, Kun Zhang, and Tongliang Liu. 2025. Flow: Modularized Agentic Workflow Automation. In The Thirteenth International Conference on Learning Representations. https://openreview.net/forum?id= sLKDbuyq99

- OpenAI. 2024. Introducing OpenAI o1-preview. https://openai.com/index/introducingopenai-o1-preview/.
- OpenAI. 2025a. Introducing Deep Research. https://openai.com/index/introducingdeep-research/ Accessed: 2025-02-02.

OpenAI. 2025b. Introducing OpenAI o3 and o4-mini. https://openai.com/index/ introducing-o3-and-o4-mini/ Accessed: 2025-04-18.

Jiazhen Pan, Che Liu, Junde Wu, Fenglin Liu, Jiayuan Zhu, Hongwei Bran Li, Chen Chen, Cheng Ouyang, and Daniel Rueckert. 2025. MedVLM-R1: Incentivizing Medical Reasoning Capability of Vision-Language Models (VLMs) via Reinforcement Learning. arXiv preprint arXiv:2502.19634 (2025).

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach. 2023. SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis. arXiv:2307.01952 [cs.CV] https: //arxiv.org/abs/2307.01952

Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek HakkaniTür, Gokhan Tur, and Heng Ji. 2025. ToolRL: Reward is All Tool Learning Needs. arXiv:2504.13958 [cs.LG] https://arxiv.org/abs/2504.13958

Chen Qian, Xin Cong, Wei Liu, Cheng Yang, Weize Chen, Yusheng Su, Yufan Dang, Jiahao Li, Juyuan Xu, Dahai Li, Zhiyuan Liu, and Maosong Sun. 2023. Communicative Agents for Software Development. arXiv:2307.07924 [cs.SE]

Shuofei Qiao, Runnan Fang, Zhisong Qiu, Xiaobin Wang, Ningyu Zhang, Yong Jiang, Pengjun Xie, Fei Huang, and Huajun Chen. 2025. Benchmarking Agentic Workflow Generation. arXiv:2410.07869 [cs.CL] https://arxiv.org/abs/2410.07869

Simo Ryu. 2023. Low-rank adaptation for fast text-to-image diffusion fine-tuning. https://github.com/cloneofsimo/lora.

Yu Shang, Yu Li, Keyu Zhao, Likai Ma, Jiahe Liu, Fengli Xu, and Yong Li. 2025. AgentSquare: Automatic LLM Agent Search in Modular Design Space. In The Thirteenth International Conference on Learning Representations. https://openreview.net/ forum?id=mPdmDYIQ7f

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300 [cs.CL] https://arxiv.org/abs/2402.03300

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik R Narasimhan, and Shunyu Yao. 2023. Reflexion: language agents with verbal reinforcement learning. In Thirtyseventh Conference on Neural Information Processing Systems. https://openreview. net/forum?id=vAElhFcKW6

Dominik Sobania, Martin Briesch, and Franz Rothlauf. 2024. ComfyGI: Automatic Improvement of Image Generation Workflows. arXiv:2411.14193 [cs.CV] https: //arxiv.org/abs/2411.14193

Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. 2025. R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning. arXiv preprint arXiv:2503.05592 (2025).

Junlin Wang, Jue WANG, Ben Athiwaratkun, Ce Zhang, and James Zou. 2025a. Mixtureof-Agents Enhances Large Language Model Capabilities. In The Thirteenth International Conference on Learning Representations. https://openreview.net/forum?id= h0ZfDIrj7T

Jifang Wang, Xue Yang, Longyue Wang, Zhenran Xu, Yiyu Wang, Yaowei Wang, Weihua Luo, Kaifu Zhang, Baotian Hu, and Min Zhang. 2025b. A Unified Agentic Framework for Evaluating Conditional Image Generation. arXiv:2504.07046 [cs.CV] https: //arxiv.org/abs/2504.07046

Qineng Wang, Zihao Wang, Ying Su, Hanghang Tong, and Yangqiu Song. 2024. Rethinking the Bounds of LLM Reasoning: Are Multi-Agent Discussions the Key?. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand, 6106–6131. doi:10.18653/v1/2024.acl-long.331

Chen Henry Wu and Fernando De la Torre. 2023. A Latent Space of Stochastic Diffusion Models for Zero-Shot Image Editing and Guidance. In ICCV.

Minghao Wu, Yulin Yuan, Gholamreza Haffari, and Longyue Wang. 2024. (Perhaps) Beyond Human Translation: Harnessing Multi-Agent Collaboration for Translating Ultra-Long Literary Texts. arXiv:2405.11804 [cs.CL] https://arxiv.org/abs/2405.11804

Jia Xu, Weilin Du, Xiao Liu, and Xuejun Li. 2024. LLM4Workflow: An LLM-based Automated Workflow Model Generation Tool. In Proceedings of the 39th IEEE/ACM International Conference on Automated Software Engineering (Sacramento, CA, USA) (ASE ’24). Association for Computing Machinery, New York, NY, USA, 2394–2398. doi:10.1145/3691620.3695360

Zhenran Xu, Senbao Shi, Baotian Hu, Jindi Yu, Dongfang Li, Min Zhang, and Yuxiang Wu. 2023. Towards Reasoning in Large Language Models via Multi-Agent Peer Review Collaboration. arXiv:2311.08152 [cs.CL] https://arxiv.org/abs/2311.08152

Zhenran Xu, Longyue Wang, Jifang Wang, Zhouyi Li, Senbao Shi, Xue Yang, Yiyu Wang, Baotian Hu, Jun Yu, and Min Zhang. 2025a. FilmAgent: A Multi-Agent Framework for End-to-End Film Automation in Virtual 3D Spaces. arXiv:2501.12909 [cs.CL] https://arxiv.org/abs/2501.12909

Zhenran Xu, Xue Yang, Yiyu Wang, Qingli Hu, Zijiao Wu, Longyue Wang, Weihua Luo, Kaifu Zhang, Baotian Hu, and Min Zhang. 2025b. ComfyUI-Copilot: An Intelligent Assistant for Automated Workflow Development. arXiv:2506.05010 [cs.CL] https: //arxiv.org/abs/2506.05010

Xiangyuan Xue, Zeyu Lu, Di Huang, Zidong Wang, Wanli Ouyang, and Lei Bai. 2024. ComfyBench: Benchmarking LLM-based Agents in ComfyUI for Autonomously Designing Collaborative AI Systems. arXiv:2409.01392 [cs.CL] https://arxiv.org/ abs/2409.01392

Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. 2023. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721 (2023).

Peng Yingzhe, Zhang Gongrui, Zhang Miaosen, You Zhiyuan, Liu Jie, Zhu Qipeng, Yang Kai, Xu Xingzhong, Geng Xin, and Yang Xu. 2025. LMM-R1: Empowering 3B LMMs with Strong Reasoning Abilities Through Two-Stage Rule-Based RL. arXiv:2503.07536 [cs.CL]

Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xiong-Hui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, and Chenglin Wu. 2025. AFlow: Automating Agentic Workflow Generation. In The Thirteenth International Conference on Learning Representations. https://openreview.net/forum?id=z5uVAKwmjf

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding conditional control to text-to-image diffusion models. In CVPR. Jiaxing Zhao, Xihan Wei, and Liefeng Bo. 2025. R1-Omni: Explainable Omni-Multimodal Emotion Recognition with Reinforcement Learning. arXiv:2503.05379

Yu Zhao, Huifeng Yin, Bo Zeng, Hao Wang, Tianqi Shi, Chenyang Lyu, Longyue Wang, Weihua Luo, and Kaifu Zhang. 2024. Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions. CoRR abs/2411.14405 (2024). doi:10.48550/ARXIV.2411.14405 arXiv:2411.14405

Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. 2025. DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments. arXiv preprint arXiv:2504.03160 (2025).

Hengguang Zhou, Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, and Cho-Jui Hsieh. 2025. R1-Zero’s "Aha Moment" in Visual Reasoning on a 2B Non-SFT Model. arXiv:2503.05132 [cs.AI] https://arxiv.org/abs/2503.05132

User Instruction: Create a workflow that can seamlessly combine two images into one wider image by intelligently generating a transition area between them. The workflow should analyze the style and content of the first image and use it to create a natural-looking bridge to the second image. The result should look like one cohesive image rather than two separate images placed side by side.

[Figure 11]

[Figure 12]

ComfyUI-R1 Generated Workflows & Execution Result:

[Figure 13]

[Figure 14]

- Fig. 4. Case study: multi-image combination. Here ComfyUI-R1 seamlessly combines the input images by generating a correct and executable workflow.

ComfyAgent Generated Workflows & Execution Result:

[Figure 15]

[Figure 16]

- Fig. 5. Case study: multi-image combination (continued). This figure shows the workflow and the corresponding execution result generated by ComfyAgent [Xue et al. 2024].

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

Image Editing

Categories

Fig. 6. An illustration of the task categories in our constructed workflow KB, including the subcategories under image editing.

