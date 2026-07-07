# arXiv:2311.05437v1[cs.CV]9Nov2023

## LLAVA-PLUS: LEARNING TO USE TOOLS FOR CREATING MULTIMODAL AGENTS

Shilong Liu♠∗, Hao Cheng♣, Haotian Liu♢∗, Hao Zhang♡∗, Feng Li♡∗, Tianhe Ren , Xueyan Zou♢∗, Jianwei Yang♣, Hang Su♠, Jun Zhu♠, Lei Zhang , Jianfeng Gao♣, Chunyuan Li♣

♠Dept. of Comp. Sci. & Tech., Institute for AI, BNRist, Tsinghua University

♣Microsoft Research, Redmond

♢University of Wisconsin-Madison ♡HKUST IDEA Research ∗Work performed during an internship at Microsoft Project Lead

https://llava-vl.github.io/llava-plus/

ABSTRACT

This paper presents LLaVA-Plus (Large Language and Vision Assistants that Plug and Learn to Use Skills), a general-purpose multimodal assistant trained using an end-to-end approach that systematically expands the capabilities of large multimodal models (LMMs). LLaVA-Plus maintains a skill repository that contains a wide range of vision and vision-language pre-trained models (tools), and is able to activate relevant tools, given users’ multimodal inputs, to compose their execution results on the fly to fulfill many real-world tasks. To acquire the ability of using tools, LLaVA-Plus is trained on multimodal instruction-following data that we have curated. The training data covers many tool use examples of visual understanding, generation, external knowledge retrieval and their compositions. Empirical results show that LLaVA-Plus outperforms LLaVA in existing capabilities, and exhibits many new capabilities. Compared with tool-augmented LLMs, LLaVA-Plus is distinct in that the image query is directly grounded in and actively engaged throughout the entire human-AI interaction sessions, significantly improving tool use performance and enabling new scenarios.

1 INTRODUCTION

A long-standing aspiration in artificial intelligence is to develop general-purpose assistants that can effectively follow users’ (multimodal) instructions to complete a wide range of real-world tasks (Askell et al., 2021; Li et al., 2023c). Recently, the community has witnessed a growing interest in developing foundation models with emergent abilities of multimodal understanding and generation in open-world tasks (Gan et al., 2022; Li et al., 2022). While the recipes of using Large Language Models (LLMs) such as ChatGPT (OpenAI, 2023a) to develop general-purpose assistants for natural language tasks have been proved effective, the recipes of building general-purpose, multimodal assistants for computer vision and vision-language tasks remain to be explored.

Ongoing efforts of developing multimodal agents can be broadly categorized into two classes (Li et al., 2023c): (i) End-to-end training with LLMs, where image-text data and multimodal instructionfollowing data are collected to continually train LLMs to acquire the ability of processing visual information, resulting in a series of Large Multimodal Models (LMMs). Impressive visual understanding and reasoning performances have been demonstrated by both proprietary models such as Flamingo (Alayrac et al., 2022) and multimodal GPT-4 (OpenAI, 2023c), and open-sourced models such as LLaVA (Liu et al., 2023a) and MiniGPT-4 (Zhu et al., 2023). Although these end-to-end training methods are effective in helping LMMs to gain emergent abilities (such as in-context learning), it

Contact: liusl20@mails.tsinghua.edu.cn, {jfgao,chunyl}@microsoft.com

[Figure 1]

Visual Generation Visual Interaction

[Figure 2]

Input Image

Conditional Gen. Editing

Point to multi-level Seg Box to Seg Stroke to Seg

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

The image depicts a wooden pier extending into a large body of water, with a bench placed in the middle of the pier. The scene is set against a backdrop of trees and mountains, creating a tranquil and picturesque environment.

The lake in question is

The text "2024 16MONTH CALENDAR" is detected, suggesting that this calendar is a 16-month calendar for the year 2024.

This image is paw-some!

Lakebridge

Lost Lake, which is located in Whistler, British Columbia,

Who says cats can‘t be outdoorsy? !" #naturelover

Canada.

[Figure 13]

[Figure 14]

Object Seg and Det Semantic Seg

OCR Search

Social Media Post

LLaVA

External Knowledge

[Figure 15]

Visual Understanding Composition

Figure 1: Visual illustration of LLaVA-Plus’ capabilities enabled by learning to use skills.

remains challenging to develop a unified architecture that can seamlessly incorporate a wide range of skills, such as image segmentation and generation, which are crucial for real-world multimodal applications. (ii) Tool chaining with LLMs, where the prompts are meticulously crafted to enable LLMs (e.g., through LangChain lan (2022)) to invoke different tools (e.g., pre-trained vision models) to perform desired (sub-)tasks, without the need of additional model training. Some prominent works include VisProg (Gupta & Kembhavi, 2022), ViperGPT (Sur´ıs et al., 2023), Visual ChatGPT (Wu et al., 2023), X-GPT (Zou et al., 2023a), and MM-REACT (Yang et al., 2023b). The strength of these methods is the ability to perform a broad spectrum of visual tasks through the use of (new) tools, which can be incorporated into an AI agent with very low development cost. However, prompting is neither adaptable nor robust enough to allow multimodal agents to always accurately select and activate appropriate tools (from a large and diverse toolset) and compose their results to generate final answers on the fly for real-world multimodal tasks.

In this paper, we present LLaVA-Plus (Large Language and Vision Assistants that Plug and Learn to Use Skills), a general-purpose multimodal assistant that learns to use tools using an end-to-end training approach that systematically expands the capabilities of LMMs via visual instruction tuning. To the best of our knowledge, this is the first attempt reported to combine the strengths of the end-to-end training and tool chaining methods mentioned above. LLaVA-Plus is equipped with a skill repository that contains a wide range of vision and vision-language tools. The design is an embodiment of the “Society of Mind” scheme (Minsky, 1988), where each tool is originally designed for a specific skill and by itself is only useful for specific scenarios, but the combinations of these tools lead to emergent abilities that show signs of higher intelligence. For example, LLaVA-Plus is able to construct a new workflow on the fly, given users’ multimodal inputs, select and activate relevant tools from the skill repository, and compose their execution results to fulfill many real-world tasks that are unseen during model training.

LLaVA-Plus can be continually improved by incorporating new skills or tools via instruction tuning. Consider a new multimodal tool that has been developed for a specific scenario or skill. We collect pertinent user instructions that request this tool and their execution results (or following) to form instruction-following data for tuning. After instruction tuning, LLaVA-Plus expands its abilities as it learns to use this new tool to deal with the tasks that it cannot handle before. LLaVA-Plus also differs from those existing works on teaching LLMs to use tools (e.g., Yang et al., 2023a; Patil et al., 2023), where visual signals are only used when the multimodal tools are activated. In contrast, LLaVA-Plus uses the raw visual signals through the entire human-AI interaction sessions to improve LMM’s ability of planning and reasoning.

In summary, our paper makes the following contributions:

- • New multimodal instruction-following tool use data. We present a new pipeline for curating vision-language instruction-following data, dedicated for tool use in human-AI interaction sessions, leveraging ChatGPT and GPT-4 as labeling tools.
- • New large multimodal assistant. We have developed LLaVA-Plus, a general-purpose multimodal assistant that extends LLaVA (Liu et al., 2023a) by incorporating a large and diverse set of external tools that can be selected, composed, and activated on the fly for performing tasks. As shown in Figure 1, LLaVA-Plus significantly extends LMM’s capabilities. Our empirical

- study validates the effectiveness of LLaVA-Plus with consistently improved results on multiple benchmarks, and in particular, new SoTA on VisiT-Bench with a diverse set of real-life tasks.
- • Open-source. We will release the following assets to the public: the generated multimodal instruction data, the codebase, the LLaVA-Plus checkpoints, and a visual chat demo.

2 LEARNING TO USE TOOLS WITH VISUAL INSTRUCTION TUNING

Inspired by the impressive performance of multimodal GPT-4 and the open-source LMMs such as LLaVA/MiniGPT-4, the community has witnessed a surge in developing LMMs and the multimodal instruction-following data, following the instruction tuning paradigm (e.g., Liu et al., 2023a; Peng et al., 2023a). In this paper, we use LLaVA as a running example. But note that the proposed recipe can be easily applied to other LMMs. Starting with a user input image query Iq, existing LMMs such as LLaVA typically accept a natural language instruction input Xq from the user, and output a natural language response Xanswer. Therefore, we can use a unified scheme to represent multimodal instruction-following data as:

Human : Iq <\n> Xq<STOP> Assistant : Xanswer<STOP>, (1)

where Human and Assistant are special role tokens, < \n > and <STOP> are the line break token and sequence end token, respectively. It naturally covers any multimodal tasks that can be formulated as language-image input and language output, ranging from simple visual understanding tasks such as recognition, captioning, and visual question answering (VQA) to complex visual reasoning tasks. Due to its simplicity, the data pipeline is easy to construct and scale. By training a single Transformer-based model with an auto-regressive objective, the resulting LMM enables a seamless human-assistant interaction, proficiently completing many visual tasks in the wild. However, it is limited in flexibility regarding skill expansion and engagement in human-AI interactions.

- 2.1 LLAVA-PLUS

We propose a modularized system architecture that allows an LMM, working as a planner, to learn to use a wide range of skills at scale, and thus facilitating easy expansion of its capabilities and interface. Specifically, we build a skill repository, where the LMM can leverage a broad range of existing vision and vision-language specialist models as tools for their respective skills when needed, to complete various tasks in the wild. The LMMs in most existing multimodal agents typically perform user-oriented dialogues, where the LMMs are required to immediately respond to user instructions based solely on the knowledge encoded in model weights, as shown in equation 1 and the left part of Figure 2. In addition to this, the LMM in LLaVA-Plus also performs skill-oriented dialogues, where the LMM initiates requests to call appropriate tools from the skill repository, and subsequently aggregate the tool execution results after applying proper skills, as shown in the right part of Figure 2.

A Full Dialogue of LLaVA-Plus. We illustrate how LLaVA-Plus works with a full dialogue session in Figure 2. It proceeds in four steps: ⃝1 Humans provide a task instruction Xq related to an image Iq. ⃝2 The LMM-powered assistant analyzes both Xq and Iq, and outputs Xskill use that chooses the tool from skill repository and writes the appropriate prompt as the tool argument. ⃝3 By executing the tool, the result Xskill result is returned to the assistant. ⃝4 The assistant aggregates Xskill result with Xq and Iq, and outputs Xanwser to humans. The interaction can be represented as:

1 2

Instruction

Tool Use

Human Assistant Skill Repository

4 3

Tool Output

Response

Figure 2: The four-step LLaVA-Plus pipeline.

Human : Iq <\n> Xq<STOP> Assistant : Xskill use<STOP> Human : Xskill result<STOP> Assistant : Xanwser<STOP> (2)

Compared with equation 1 which is used to train LLaVA, the only newly introduced component for LLaVA-Plus training is the skill-oriented dialogue. Table 1 illustrates one sequence example of calling detection and segmentation skills in human-AI interactions. LLaVA-Plus is trained with an auto-regressive objective on the sequence of equation 2, where only the green sub-sequences (or tokens) are used to compute the loss, and thus the model learns to predict skill use, answers, and when to stop.

Can you segment the girl and the cart in the image and describe their relationship?

Human (Xq, Iq )

[Figure 16]

(The source image from COCO with ID 569536)

“thoughts” To accurately segment and identify the girl and the cart in the image, we can leverage the grounding dino model for open-set detection and the segmentation model SAM. Integrating these models will enable us to generate a comprehensive summary of the segmented subjects.

Assistant (Xskill use)

“actions” [”API name”: ”grounding dino+sam”, ”API params”: ”caption”: ”little girl and cart .”] “value” I will use grounding dino+sam to help to answer the question. Please wait for a moment.

Skill Repo. Tool execution: Grounding DINO + SAM Human (Xskill result )

grounding dino+sam model outputs: ’boxes’: [[0.35, 0.37, 0.66, 0.97], [0.0, 0.57, 0.69, 1.0]], ’logits’: [0.58, 0.41], ’phrases’: [’little girl’, ’cart’] Can you segment the girl and the cart in the image and describe their relationship?

Assistant “thoughts” Thanks to the output of grounding dino+sam. I can answer the question better. (Xanwser) “actions” []

“value” Sure! I segment and identify the little girl and the cart in the image.

- - Centered in the image, a little girl is discernible with bounding box coordinates [0.35, 0.37, 0.66, 0.97]. She’s seen actively maneuvering a cart within the airport’s ambiance.

[Figure 17]

- - To the bottom left of the image, a cart can be pinpointed, bounded by the coordinates [0.0, 0.57, 0.69, 1.0]. It’s laden with various luggage items, indicating travel preparations or arrivals.

The interaction between the girl and the cart epitomizes a typical airport scenario: a traveler utilizing the cart to facilitate her movement with luggage. This tableau, whether signifying an arrival or departure, encapsulates the vibrancy and expectation inherent to air travel.

- Table 1: An example of a LLaVA-Plus workflow that plugs and learns to use the skills of object detection and segmentation, enhanced by a rich region language description. The gray text is not in the training sequence.

Unified Prediction Format from LMMs. Figure 2 shows that the LMM of LLaVA-Plus needs to perform both user-oriented and skill-oriented dialogues. To this end, we use a unified model prediction format to represent dialogues with and without the need of calling the skill repository. Inspired by Yao et al. (2022), the format consists of three fields, as illustrated in Table 1: (i) Thought is a text sequence representing a reasoning process, which determines whether the skill repository is needed to follow the user instruction, and if so, which tools to use. (ii) Action is a list of function calls for the tools to execute the thought. The list is in the JSON format, with each item consisting of two sub-fields: API name to call the tool and API params for the corresponding function arguments if applicable. When action is an empty list, no skill is invoked. (iii) Value is a natural language response that LLaVA-Plus generates by aggregating tool execution results and the human-AI session history. When presented in Xskill use of user-oriented dialogues, it is the final response returned to human users. When presented in Xanwser of skill-oriented dialogues, it is a natural language description about tool execution. In the serving stage, we find it important to ensure a good user experience that we only return the content in the value field of Xanwser to human users, but hide the entire skill-oriented dialogues unless we need to debug the system.

- 2.2 SKILL REPOSITORY: MULTIMODAL TOOL USE INSTRUCT DATA GENERATION

The skill repository of LLaVA-Plus consists of multimodal tools of different skills. To allow the LMM to always activate the most appropriate tools to complete a task, the corresponding tool-use multimodal instruction-following data is needed for LMM tuning. We follow the self-instruct method to curate the data by using GPT-4 as the labeler. Without loss of generality, in this study we want LLaVA-Plus to deal with the scenarios that requires novel skills that LLaVA does not have, e.g., the individual skills for visual understanding, generation, and external knowledge retrieval and the compositions of these individual skills, as summarized in Table 2. In what follows, we treat visual understanding skills as core skills and the others as extended skills, and describe the way instruction data is curated.

###### Skills Tools Source Size

Detection/Grounding G-DINO (Liu et al., 2023b) COCO 13783

Semantic Segmentation OpenSeeD (Zhang et al., 2023a) COCO 5989 Instance Segmentation G-DINO+SAM COCO 5228 Caption + Grounding BLIP2+G-DINO COCO 4037 Tagging + Grounding RAM+G-DINO COCO 4439 Caption BLIP2 Li et al. (2023e) COCO 4064 Tagging RAM (Zhang et al., 2023d) COCO 6045 OCR EasyOCR (JaidedAI, 2023) Hiertext 6528

Understanding

IndividualSkills

External Knowledge Retrieval CLIP Retrieval (Radford et al., 2021) InfoSeek 4087 Generation

Image Generation Stable Diffusion (Rombach et al., 2021) JourneyDB 4694 Image Editing Instruct P2P (Brooks et al., 2023) Instruct P2P 6981

Interactive Segmentation SAM (Kirillov et al., 2023) COCO 5601 Multi-granularity Semantic SAM (Li et al., 2023d) COCO 5601 Example Based Segmentation SEEM (Zou et al., 2023b) COCO 5601

Visual Prompt

ComposedSkills

Mix of Detection, Segmentation, Tagging, Caption G-DINO, SAM, BLIP2, RAM COCO 37,431

Interactive Segmentation + Inpainting SAM + Stable Diffusion COCO 3063 Semantic Segmentation + Generation OpenSeeD + ControlNet (Zhang et al., 2023b) COCO 5989 Image Generation + Social Media Post Stable Diffusion JourneyDB 4694 Image Editing + Social Media Post Instruct P2P Brooks et al. (2023) Instruct P2P 5924

- Table 2: LLaVA-Plus skill repository and dataset statistics of our created visual instruction-following data for each tool use case. G-DINO indicates Grounding DINO (Liu et al., 2023b). HierText (Long et al., 2022; 2023), InfoSeek (Chen et al., 2023b), and JourneyDB (Pan et al., 2023) are datasets for OCR, external knowledge, and image generation, respectively.

- 2.2.1 CORE SKILLS: UNDERSTANDING

Visual understanding skills enable machines to interpret and comprehend visual signals. Existing LMMs have only a limited subset of visual understanding skills, constrained by language inputs and outputs. We expand them to a broader skill set with visual input prompts and visual outputs, including open-set detection and grounding, semantic/instance/interactive segmentation, tagging, captioning, OCR and their compositions, and so on. These understanding skills can be grouped into two categories, depending on whether additional function arguments are required.

Skills with Image-only. The skills without additional function arguments include captioning, tagging, semantic segmentation, caption+grounding, tagging+grounding, and OCR. We have curated training samples for each tool individually. To collect the training samples for a given skill, we fill in the four data variables in equation 2 using different strategies. (i) For Xq, we use GPT-4 to generate a set of instructions that require the use of tools for proper answers. For each sample, we randomly select a question and rewrite it to enhance data diversity. An rewriting example is shown in Table 9 in Appendix. (ii) For Xskill use, its thoughts and value are generated by randomly selecting from some preset responses with rewriting. The actions is known, so it can be directly assigned. (iii) Xskill result is generated with a fixed rule: first presenting the tool outputs and then repeating the initial question. (iv) For Xanwser, its thoughts is created in a similar way to thoughts in Xskill use, and action is set empty. The value of Xanwser is the most important field, as it is the visible response to humans in chat. We feed all previous information, including previous questions, the previous tool outputs, and context of the image to language-only GPT-4, which then generates responses to form instruction-following data. Inspired by LLaVA, we consider the ground-truth captions, object coordinates, and object categories as image contexts.

Skills with Additional Function Arguments. Visual skills such as object detection and instance segmentation often require humans to provide very specific instructions regarding the concepts of interests. Their instruction-following data is more challenging to create. We use two methods in this study. (i) The first method is similar to that in the image-only skill setting, where the initial Xq contains a placeholder concept, one or more categories presented in the image are randomly chosen to replace this placeholder, and the final Xq is obtained via rewriting, as shown in Table 9.

- (ii) To allow the LMM to learn more diverse prompts beyond category information, we use GPT-4 to generate questions. Specifically, we manually create two seed samples following the full dialogue in equation 2, send them, together with image contexts, to GPT-4, and ask GPT-4 to generate a full dialogue based on a new image context. An example is shown in Table 10 in Appendix.

- 2.2.2 EXTENDED SKILLS

The LLaVA-Plus recipe can be applied to any tools to improve the system capabilities. We demonstrate its versatility by onboarding multimodal tools of different categorizes. Due to the limited space, we describe the instruction-following data creation process in Section B in Appendix, and summarize the extended skills we have enabled.

- • External Knowledge. To enable LMMs to use knowledge beyond that encoded in pre-trained model weights, we use the CLIP search API to retrieve external knowledge from LIAON.
- • Generation. To allow LLaVA-Plus to output images, we use Stable Diffusion (SD) and Instruct-Pix2Pix for image generation and editing, respectively.
- • Visual Prompts. To better follow human intents, we support various visual prompts for humanAI interaction, such as user-drawn points, sketches and boxes. SAM, Semantic-SAM and SEEM are used for different interactive segmentation tasks.
- • Skill Composition. To allow LLaVA-Plus to deal with real-world compositional tasks. We curate data for the following scenarios: (i) The scenarios where various visual understanding results of the same image in a multi-turn human-AI interaction session are required. We generate instruction data by applying different tools (including detection, segmentation, tagging, and captioning). (ii) Interactive Segmentation + Inpainting. By combining the SAM segmentation results from the user pointing and SD, we enable inpainting with visual interaction. (iii) Semantic Segmentation + Generation. By combining the spatial layout from OpenSeed semantic segmentation and ControlNet, we enable instructional visual-conditioned generation. (iv) Image Generation/Editing + Social Media Post. It is time-consuming for human users to generate posts that contains both images and text. Thus, we use SD to generate an image, or Instruct Pix2Pix to edit an image, then combine the image with its description generated by a pre-trained LMM to create a multimodal post.

- 2.3 MODEL TRAINING AND SERVING

Training. To train LLaVA-Plus, we combine the curated tool use instruction data, as shownin Table 2, with the LLaVA-158K dataset. To convert LLaVA-158K into the unified prediction format as described in Section 2.1, we treat the responses in LLaVA-158K as value, and add the fields of thoughts and actions with templates, as illustrated in the example in Table 8 in Appendix. LLaVA-Plus are built in two settings. (i) LLaVA-Plus (All Tools), where tool use is cast as external knowledge. All visual understanding tools except segmentation in Table 2 are utilized to process the input image, and the extracted recognition results are organized as symbolic sequence representations to enrich the image features in both the training and evaluation stages. (ii) LLaVA-Plus (Fly), where tools are used on the fly. To reduce the cost of calling all tools, we only provide the execution results of related tools for a given instruction. When reporting quantitative numbers, we train models on the 81K understanding instruction data, because existing benchmarks focus mainly on understanding capabilities. When building demo systems, we train our models on the full dataset.

Serving. LLaVA-Plus is served using the FastChat (Vicuna, 2023) system, which is composed of web servers that interface with humans, model workers that host the LMM and multiple tools, and a controller to coordinate the web-server and model workers. The 7B LLaVA-Plus and all the tools can be loaded and served in a 80G GPU.

- 3 RELATED WORKS

We summarize the connections and differences between LLaVA-Plus and existing general-purpose multimodal systems in Table 3, where only representative methods are shown due to space constraint. They can be broadly categorized into two classes as discussed below.

AI Agents with Multimodal Tool Use. There is a growing interest in exploring a paradigm of building general-purpose AI agents that synergistically leverage multiple tools with LLMs to solve sophisticated, open-world problems. The idea is originated in NLP to invoke general tools whose skills are lacked from LLM (e.g., ToolFormer (Schick et al., 2023), ChatGPT-Plugin (OpenAI, 2023b)), and is recently extended to the multimodal space. There are two ways to leverage multimodal tools with the LLM as a planner to determine which tools to invoke: (i) tool chaining by prompt engineering

Capabilities

Too Use Input

|Image Understanding|Knowledge Image Gen.<br><br>|Visual Interaction<br><br>|Combined|
|---|---|---|---|
|(Text, Image) Text Box Mask Text Image<br><br>| |(Point, Box) (Text, Image, Mask)|All All<br><br>|

Allocator Training

Output MM-REACT ✓ ✓ ✓ LLM GPT4Tools ✓ ✓ ✓ ✓ LLM ✓ LLaVA-Plus ✓ ✓ ✓ ✓ ✓ ✓ ✓ LMM ✓ LLaVA/GPT-V ✓ Kosmos-2 ✓ ✓ CM3Leon ✓ ✓ ✓ ✓

Table 3: Comparison with existing multimodal systems. The empty cells indicate inapplicable. “Allocator” indicates which base model is used to invoke the tools, and “Training” indicates whether model training is needed to enable tool use.

and in-context-learning, such as Visual ChatGPT (Wu et al., 2023), MM-ReAct (Yang et al., 2023b), and (ii) instruction tuning of LLM with a focus on multimodal tool use, such as GPT4Tools (Yang et al., 2023a) and Gorilla (Patil et al., 2023). LLaVA-Plus represents the first work of utilizing the LMM as the planner for tool use, where image inputs are considered throughout the entire interaction sessions for improved user experience.

Unified Multimodal Models with Versatile Capabilities. Inspired by the success of a unified architecture of LLMs to complete many language tasks, the AI community has witnessed an increasing interest in building unified models with versatile multimodal capabilities. Proprietary models such as Flamingo (Alayrac et al., 2022) and multimodal GPT-4 (OpenAI, 2023c) (or GPT-4V (OpenAI, 2023d)) have demonstrated strong multimodal performance on zero-shot task transfer, which quickly inspired their open-source counterparts: LLaVA, MiniGPT-4, Open-Flamingo (Awadalla et al., 2023), Otter (Li et al., 2023a), to name a few. These LMMs can deal with the tasks with image-text input and text output. The capabilities have been extended to support the tasks with image-text output, such as image editing and segmentation, as demonstrated in CM3Leon (Yu & et al, 2023), Emu (Sun et al., 2023), and GILL (Koh et al., 2023). Bounding box outputs for grounding are recently supported, as shown in Kosmos-2 (Peng et al., 2023b), Shikra (Chen et al., 2023a) and DetGPT (Pi et al., 2023). GPT4ROI (Zhang et al., 2023c) allows users to select regions of interest with bounding boxes for human-AI visual chat. BubaGPT (Zhao et al., 2023) and LISA (Lai et al., 2023) use an extra referring segmentation model to enable the mask prediction capability. Compared with them, LLaVA-Plus enables a much wider range of multimodal skills and their compositions, as illustrated in Table 3.

- 4 EXPERIMENTS

- 4.1 THE EFFECTIVENESS OF LEARNING TO USE SKILLS

Tool Use Improves Existing Capabilities. We consider two benchmarks. LLaVA-Bench (Liu et al., 2023a) evaluates the visual chat of LMMs, with three types of questions: conversation, detailed description and visual reasoning. It consists of two datasets: the COCO set containing 30 COCO images and 90 chat questions, and the In-the-Wild set containing 24 web images with 60 questions. Language GPT-4 (gpt4-0314) is used to score the generated answers. The relative scores between the model output and gold response are reported. SEED-Bench (Li et al., 2023b) evaluates the image-level and instance-level perception and reasoning of LMMs, with 19K multi-choice questions. The results are shown in Table 4. Both LLaVA-Plus variants outperform LLaVA on these two benchmarks, demonstrating the effectiveness of adding visual recognition results of applying new skills in the LMM pipeline. LLaVA-Plus (All Tools) shows superior performance to LLaVA-Plus (Fly) because the former leverages more tools as additional contexts. We further conducted several ablations: (i) We tried to directly add the skill execution results in the testing stage of LLaVA, shown as the row of LLaVA (Tools in Test). The degraded performance compared with LLaVA demonstrates the necessity of learning to use skills in training. (ii) We removed thoughts in the unified data format and observed a performance drop, indicating chain-of-thoughts style data format is beneficial.

- (iii) GPT4Tools trains an LLM for multimodal tool use. Its lower performance indicates that visual instruction tuning of tool use in LLaVA-Plus is important.

LLaVA-Bench (Tools). To study the novel capabilities enabled by learning to use skills, we create an evaluation set LLavA-Bench (Tools), which measures four capabilities (grounding, tagging, caption, and OCR) with 10, 12, 12, and 10 samples in each. In Table 5, we also compare against

###### LLaVA-Bench (COCO) LLaVA-Bench (In-the-Wild)

Conv. Detail Reasoning All Conv. Detail Reasoning All

LLaVA 82.0 69.1 92.6 81.2 42.6 51.9 68.9 57.1 LLaVA (Tools in Test) 56.2 67.9 53.3 59.1 40.7 48.1 51.2 47.5 LLaVA-Plus (All Tools) 81.6 74.5 95.7 83.9 65.5 56.8 79.1 69.5 LLaVA-Plus (Fly) 76.2 72.2 92.3 80.4 45.2 50.4 72.6 59.1 LLaVA-Plus (Fly) (no thoughts) 76.6 70.4 90.7 79.4 38.8 39.8 59.8 48.7

GPT4Tools 75.3 53.8 86.9 72.1 31.1 27.1 54.1 40.7

(a) LLaVA-Bench.

Scene Identity Attribute Location Counting Spatial Interact. Reason. Text Average

LLaVA 59.50 54.29 56.06 42.54 39.35 33.03 43.30 41.39 30.59 44.45 LLaVA (Tools in Test) 67.13 56.85 45.24 47.24 45.69 40.18 60.82 70.09 30.59 51.54 LLaVA-Plus (All Tools) 68.94 56.80 58.89 47.34 48.14 45.21 60.82 71.30 37.65 55.01 LLaVA-Plus (Fly) 68.43 56.47 59.69 45.40 41.68 44.14 59.79 69.49 34.12 53.25

(b) SEED-Bench.

Table 4: LLaVA-Plus variants improves LLaVA on two LMM benchmarks.

Grounding Tagging Caption OCR All

LLaVA 47.1 87.1 77.0 23.6 58.7 LLaVA (Tools in Test) 41.7 48.5 72.0 31.9 48.5 LLaVA-Plus (All Tools) 89.3 94.4 96.7 48.8 82.3 LLaVA-Plus (Fly) 88.6 88.9 90.2 38.4 76.5

Bard (0730) 36.5 105.3 103.3 60.0 76.3 Bing Chat (0730) 56.0 84.0 96.0 44.8 70.2 MM-REACT 30.2 94.7 103.8 77.3 76.5 All Tools + GPT4 77.5 95.6 95.2 39.3 76.9

Table 5: LLaVA-Bench (Tool Use).

the commercial visual chat systems such as Microsoft BingChat and Google Bard. LLaVA-Plus significantly outperforms the others on this benchmark, mainly because the other systems are not equipped with some of these capabilities. By comparing with chaining tools with GPT-4 (row of “All tools + GPT4”) and MM-REACT, we demonstrate the advantage of training an open-source LMM as a planner for tool use.

- 4.2 COMPARISONS WITH SOTA LMM SYSTEMS

MMVet (Yu et al., 2023) contains 200 images and 218 questions, aiming to evaluate six core vision-language (VL) capabilities and their combinations. For evaluation, an LLM-based evaluator (gpt4-0613) is used to score open-ended outputs of different forms. The results are reported in Table 6. LLaVA-Plus consistently outperforms LLaVA on both 7B and 13B model sizes. The categories with most significant improvements are OCR and spatial, indicating the positive impact of the corresponding visual skills on LMM outputs.

VisIT-Bench (Bitton et al., 2023) is a real-world use oriented LMM benchmark, comprising 592 questions and 1,159 public images categorized into 70 instruction families. The results are shown in Table 7, which summarizes the battles between LMMs with GPT-analog human judgment. Elo ratings are computed by treating each pairwise human judgment as a “match”. The difference between the Elo ratings of two models provides an estimate for the win probability when pitting model A vs. model B. The “#matches” column indicates the number of total matches in which a particular model participates. Win-rate indicates the win rate of a model against the human-verified reference outputs. LLaVA-Plus significantly outperforms the leading method LLaVA by 100+ ELO score, achieving a new SoTA on the leaderboard.

- 4.3 VISUAL EXAMPLES OF NEW CAPABILITIES

In Table 3, we illustrate new capabilities of LLaVA-Plus with visual examples. Please see Section D in Appendix for many other interesting scenarios that demonstrate the versatile capabilities of LLaVA-Plus by learning to use skills and their compositions.

In the left example, the questions require identifying the precise object locations. LLaVA-Plus can successfully detect the frisbee’s coordinates, which help determine its status of flying in the

Model Rec OCR Knowledge Generation Spatial Math Total Results of various open-source LMM on reported in the MM-VET paper (Yu et al., 2023)

OpenFlamingo-9B (Awadalla et al., 2023) 24.6 14.4 13.0 12.3 18.0 15.0 21.8±0.1 BLIP-2-12B (Li et al., 2023e) 27.5 11.1 11.8 7.0 16.2 5.8 22.4±0.2 LLaVA-7B (Liu et al., 2023a) 28.0 17.1 16.3 18.9 21.2 11.5 23.8±0.6 MiniGPT-4-14B (Zhu et al., 2023) 29.9 16.1 20.4 22.1 22.2 3.8 24.4±0.4 Otter-9B (Li et al., 2023a) 28.4 16.4 19.4 20.7 19.3 15.0 24.6±0.2 InstructBLIP-14B (Dai et al., 2023) 30.8 16.0 9.8 9.0 21.1 10.5 25.6±0.3 MM-ReAct-GPT-3.5 (Yang et al., 2023b) 24.2 31.5 21.5 20.7 32.3 26.2 27.9±0.1 LLaMA-Adapter v2-7B (Gao et al., 2023) 32.9 20.1 19.0 20.1 22.9 3.9 31.4±0.1 LLaVA-13B (V1.3, 336px) (Liu et al., 2023a) 38.1 22.3 25.2 25.8 31.3 11.2 32.5±0.1 MM-ReAct-GPT-4 (Yang et al., 2023b) 33.1 65.7 29.0 35.0 56.8 69.2 44.6±0.2

Results with our own experiment runs

LLaVA-7B 30.4 13.3 19.2 20.1 18.7 8.1 24.1±0.0 LLaVA-Plus-7B (All Tools) 30.5 23.6 20.5 22.5 28.5 7.7 27.5±0.3 LLaVA-Plus-13B (All Tools, V1.3, 336px) 37.5 29.4 22.3 24.5 37.3 11.5 35.0±0.0

Table 6: Performance of various open-source LMM on MM-VET. Note that MM-ReAct is not a single multimodal model, it is a system built on chaining visual tools via GPT-3.5 or GPT-4, which we append as a reference. Our experiment running on LLaVA-7B yields very similar scores with the same checkpoint reported in MM-VET paper, indicating that our evaluation pipelines are consistent.

Model ELO Matches Win(#Ratings)

Human Reference 1382 5880 LLaVA-Plus (13B) 1203 678 35.07% (134) LLaVA (13B) 1095 5420 18.53% (475) mPLUG-Owl 1087 5440 15.83% (480) LlamaAdapter-v2 1066 5469 14.14% (488) Lynx(8B) 1037 787 11.43% (140) Idefics (9B) 1020 794 9.72% (144) InstructBLIP 1000 5469 14.12% (503) Otter 962 5443 7.01% (499) Visual GPT 941 5437 1.57% (510) MiniGPT-4 926 5448 3.36% (506) Octopus V2 925 790 8.90% (146) OpenFlamingo V1 851 5479 2.95% (509) PandaGPT (13B) 775 5465 2.70% (519) MultimodalGPT 731 5471 0.19% (527)

Table 7: Current ELO rankings on ViSiT-Bench leaderboard as of Sept. 27th, 2023.

air and thus describe the outdoor scene/activity. The same example is shown to Bard, Bing Chat, MM-REACT and LLaVA in Figure 6 in Appendix. They all fail, revealing the lack of grounding ability.

In the right example, we illustrate an interactive image editing scenario, where users aim to see the spatial layout of the scene first and then generate an image of a similar layout, but with a new “under water” scene. The LMM not only applies the correct skills, but also generates a function argument “A bicycle parked next to a bench under the sea” for conditional image generation. This reveals the appealing property of LMM as a planner, as it can see the raw image, and provide necessary image analysis results throughout the human-AI interaction process. More such examples are in Appendix Figure 11.

In the bottom example, we show that LLaVA-Plus can be used to help create multimodal social media posts. For example, when capturing an image, the user wants to post the same image in an autumn scene and associate the image with some attractive text to post Instagram. LLaVA-Plus can use the editing skills to revise the image, and combine the context of visual images and their related language topics to suggest several caption options. In Appendix Figure 12, we create all four seasons for the same scenarios, and observe that LLaVA-Plus can follow the instruction to easily switch among them while consistently maintaining the original image cue.

Is it possible to perform semantic segmentation for the image?

[Figure 18]

[Figure 19]

Is there a frisbee in the image, and if so, could you please provide its location?

"thoughts!" I will u/lize the OpenSeeD model to assist in conduc/ng seman/c segmenta/on for the image. "ac/ons"" [{"API_name": "openseed", "API_params": {}}] "value#"

[Figure 20]

[Figure 21]

[Figure 22]

I will use openseed to help to answer the question. Please wait for a moment. controlnet model outputs: {'message': 'The controlnet has processed the image.'} Please summarize the model outputs and answer my first question: Is it possible to perform semantic segmentation for the image?

"thoughts!" Thanks to the output of controlnet. I can answer the ques/on beIer. "ac/ons"" [] "value#”

[Figure 23]

[Figure 24]

"thoughts!" The task requires identifying the presence of a frisbee in the image and providing its location. "actions"" [{"API_name": "grounding_dino", "API_params": {"caption": "the frisbee ."}}] "value#” I will use grounding_dino to help to answer the question. Please wait for a moment. grounding_dino model outputs: {'boxes': [[0.76, 0.64, 0.82, 0.66]], 'logits': [0.49], 'phrases': ['the frisbee']} Please summarize the model outputs and answer my first question: Is there a frisbee in the image, and if so, could you please provide its location? "thoughts!" Thanks to the output of grounding_dino. I can answer the question better. "actions"" [] "value#”

[Figure 25]

[Figure 26]

[Figure 27]

Certainly! I can assist you with performing semantic segmentation on the image. To accomplish this, I will utilize the powerful OpenSeeD model. Please take a look below. Let me know if you need any further assistance!

[Figure 28]

[Figure 29]

[Figure 30]

Can you generate a similar image under sea?

"thoughts!" I will u/lize the ControlNet model for image genera/on. "ac/ons"" [{"API_name": "controlnet", "API_params": {"prompt": {"cap/on": "A bicycle parked next to a bench under the sea."}}}] "value#"

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

Yes, there is a frisbee in the image. Based on the results of the grounding_dino model, the frisbee is located in the middle of the image, with bounding box coordinates of [0.76, 0.64, 0.82, 0.66]. This suggests that the frisbee is in the air, possibly being thrown or caught by one of the players.

I will use controlnet to help to answer the question. Please wait for a moment. controlnet model outputs: {'message': 'The controlnet has processed the image.'} Please summarize the model outputs and answer my first question: Can you generate a similar image under sea?

[Figure 37]

"thoughts!" Thanks to the output of controlnet. I can answer the ques/on beIer. "ac/ons"" [] "value#"

[Figure 38]

[Figure 39]

[Figure 40]

Certainly! I've used the powerful ControlNet model to generate an image similar to the one you described under the sea. Please take a look at the results below. I hope you like it!

make the image like autumn. then generate some attractive texts for Instagram posts

[Figure 41]

"thoughts!" To complete the image editing task, I need the instruct pix2pix model. "actions"" [{"API_name": "instruct-pix2pix", "API_params": {"prompt": "make the image like autumn."}}] "value#" I will use instruct-pix2pix to help to answer the question. Please wait for a moment. instruct-pix2pix model outputs: {'message': 'The instruct-pix2pix has processed the image.'} Please summarize the model outputs and answer my first question: make the image like autumn. then generate some attractive texts for Instagram posts "thoughts!" Thanks to the output of instruct-pix2pix. I can answer the question better. "actions"" [] "value#”

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

Great! I'll use instruct pix2pix to edit the image, and you can see the changes below. Now, let's cra! some Instagram captions for this new image:

- 1."Embrace the cozy vibes of autumn with this stunning lake view! !" #AutumnFeels #LakeViews #NatureLover"

[Figure 49]

[Figure 50]

- 2."Who says fall's colors aren't pretty? Check out this vibrant view of the lake! !# #AutumnVibes #LakeLife #ScenicRiver"

[Figure 51]

[Figure 52]

- 3."Take a moment to soak in the vivid shades of autumn with this breathtaking view! !$ #AutumnHues #NaturePhotography #SereneLocation"

[Figure 53]

[Figure 54]

- 4."Why rush through the season when you have views like this? !% #AutumnDays #LakeEscape #GetawayGoals"

[Figure 55]

[Figure 56]

- 5."Get lost in the beauty of autumn nature with this captivating view of the lake! "# #AutumnWonder #LakeLife #NatureLover"

[Figure 57]

[Figure 58]

Figure 3: New capabilities in LLaVA-Plus. Human questions Xq are in purple, LLaVA-Plus responses Xanwser are in green. (Left) Object detection and visual chat; (Right) Semantic segmentation and mask-based conditional image generation; (Bottom) Multimodal social media post by editing an image and writing a message.

- 5 CONCLUSION

We have presented LLaVA-Plus, a general-purpose, multimodal assistant which is based on an LMM that plugs and learns to use skills to complete a wide range of vision-language tasks in the wild. The first visual instruction dataset specifically designed for multimodal tool use has been collected for model training. By incorporating the execution results of new skills, LLaVA-Plus consistently outperforms LLaVA across many benchmarks, creates a new SoTA and shows emergent multimodal interaction capabilities. However, LLaVA-Plus is limited due to hallucinations and tool use conflicts in practice. There are interesting problems yet to be addressed in future research on building reliable general-purpose multimodal AI agents.

Reproducibility To ensure the reproducibility of our research, we will publicly release a comprehensive set of assets including the generated multimodal instruction data, our codebase, the LLaVA-Plus checkpoints, and a visual chat demo. Additionally, we have ensured complete transparency by elaborating on every facet of our training data collection and model training within this paper, as shown in Sec. 2.

REFERENCES

Langchain. https://github.com/hwchase17/langchain, 2022. Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel

Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. arXiv preprint arXiv:2204.14198, 2022.

Amanda Askell, Yuntao Bai, Anna Chen, Dawn Drain, Deep Ganguli, Tom Henighan, Andy Jones, Nicholas Joseph, Ben Mann, Nova DasSarma, et al. A general language assistant as a laboratory for alignment. arXiv preprint arXiv:2112.00861, 2021.

Anas Awadalla, Irena Gao, Joshua Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, and Ludwig Schmidt. Openflamingo, March 2023. URL https: //doi.org/10.5281/zenodo.7733589.

Yonatan Bitton, Hritik Bansal, Jack Hessel, Rulin Shao, Wanrong Zhu, Anas Awadalla, Josh Gardner, Rohan Taori, and Ludwig Schimdt. Visit-bench: A benchmark for vision-language instruction following inspired by real-world use, 2023.

Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18392–18402, 2023.

Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. Shikra: Unleashing multimodal llm’s referential dialogue magic. arXiv preprint arXiv:2306.15195, 2023a.

Yang Chen, Hexiang Hu, Yi Luan, Haitian Sun, Soravit Changpinyo, Alan Ritter, and Ming-Wei Chang. Can pre-trained vision and language models answer visual information-seeking questions? Feb 2023b.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning. arXiv preprint arXiv:2305.06500, 2023.

Zhe Gan, Linjie Li, Chunyuan Li, Lijuan Wang, Zicheng Liu, and Jianfeng Gao. Vision-language pre-training: Basics, recent advances, and future trends. Foundations and Trends® in Computer Graphics and Vision, 2022.

Peng Gao, Jiaming Han, Renrui Zhang, Ziyi Lin, Shijie Geng, Aojun Zhou, Wei Zhang, Pan Lu, Conghui He, Xiangyu Yue, et al. Llama-adapter v2: Parameter-efficient visual instruction model. arXiv preprint arXiv:2304.15010, 2023.

Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning

without training. arXiv preprint arXiv:2211.11559, 2022. JaidedAI. Easyocr. https://github.com/JaidedAI/EasyOCR, 2023. Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete

Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. arXiv preprint arXiv:2304.02643, 2023.

Jing Yu Koh, Daniel Fried, and Ruslan Salakhutdinov. Generating images with multimodal language models. arXiv preprint arXiv:2305.17216, 2023.

Xin Lai, Zhuotao Tian, Yukang Chen, Yanwei Li, Yuhui Yuan, Shu Liu, and Jiaya Jia. Lisa: Reasoning segmentation via large language model. arXiv preprint arXiv:2308.00692, 2023.

Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. arXiv preprint arXiv:2305.03726, 2023a.

Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023b.

Chunyuan Li, Haotian Liu, Liunian Harold Li, Pengchuan Zhang, Jyoti Aneja, Jianwei Yang, Ping Jin, Houdong Hu, Zicheng Liu, Yong Jae Lee, and Jianfeng Gao. ELEVATER: A benchmark and toolkit for evaluating language-augmented visual models. In NeurIPS Track on Datasets and Benchmarks, 2022.

Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, and Jianfeng Gao. Multimodal foundation models: From specialists to general-purpose assistants. arXiv preprint arXiv:2309.10020, 2023c.

Feng Li, Hao Zhang, Peize Sun, Xueyan Zou, Shilong Liu, Jianwei Yang, Chunyuan Li, Lei Zhang, and Jianfeng Gao. Semantic-sam: Segment and recognize anything at any granularity. arXiv preprint arXiv:2307.04767, 2023d.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023e.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023a.

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023b.

Shangbang Long, Siyang Qin, Dmitry Panteleev, Alessandro Bissacco, Yasuhisa Fujii, and Michalis Raptis. Towards end-to-end unified scene text detection and layout analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.

Shangbang Long, Siyang Qin, Dmitry Panteleev, Alessandro Bissacco, Yasuhisa Fujii, and Michalis Raptis. Icdar 2023 competition on hierarchical text detection and recognition. arXiv preprint arXiv:2305.09750, 2023.

Marvin Minsky. Society of mind. Simon and Schuster, 1988. OpenAI. ChatGPT. https://openai.com/blog/chatgpt/, 2023a. OpenAI. Chatgpt plugins. https://openai.com/blog/chatgpt-plugins, 2023b. OpenAI. Gpt-4 technical report, 2023c. OpenAI. Gpt-4v(ision) system card. https://cdn.openai.com/papers/GPTV_System_

Card.pdf, 2023d.

Junting Pan, Keqiang Sun, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, Jifeng Dai, Yu Qiao, and Hongsheng Li. Journeydb: A benchmark for generative image understanding. Jul 2023.

Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E Gonzalez. Gorilla: Large language model connected with massive apis. arXiv preprint arXiv:2305.15334, 2023.

Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruction tuning with GPT-4. arXiv preprint arXiv:2304.03277, 2023a.

Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei. Kosmos-2: Grounding multimodal large language models to the world. arXiv preprint arXiv:2306.14824, 2023b.

Renjie Pi, Jiahui Gao, Shizhe Diao, Rui Pan, Hanze Dong, Jipeng Zhang, Lewei Yao, Jianhua Han, Hang Xu, and Lingpeng Kong Tong Zhang. Detgpt: Detect what you need via reasoning. arXiv preprint arXiv:2305.14167, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. Highresolution image synthesis with latent diffusion models, 2021.

Timo Schick, Jane Dwivedi-Yu, Roberto Dess`ı, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. arXiv preprint arXiv:2302.04761, 2023.

Quan Sun, Qiying Yu, Yufeng Cui, Fan Zhang, Xiaosong Zhang, Yueze Wang, Hongcheng Gao, Jingjing Liu, Tiejun Huang, and Xinlong Wang. Generative pretraining in multimodality. arXiv preprint arXiv:2307.05222, 2023.

D´ıdac Sur´ıs, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. arXiv preprint arXiv:2303.08128, 2023.

Vicuna. Vicuna: An open-source chatbot impressing gpt-4 with 90%* chatgpt quality. https: //vicuna.lmsys.org/, 2023.

Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. Visual chatgpt: Talking, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671, 2023.

Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu Li, and Ying Shan. Gpt4tools: Teaching large language model to use tools via self-instruction. arXiv preprint arXiv:2305.18752, 2023a.

Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv preprint arXiv:2303.11381, 2023b.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629,

- 2022.

Lili Yu and et al. Scaling autoregressive multi-modal models: Pretraining and instruction tuning.

- 2023.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.

Hao Zhang, Feng Li, Xueyan Zou, Shilong Liu, Chunyuan Li, Jianfeng Gao, Jianwei Yang, and Lei Zhang. A simple framework for open-vocabulary segmentation and detection. arXiv preprint arXiv:2303.08131, 2023a.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models, 2023b.

Shilong Zhang, Peize Sun, Shoufa Chen, Min Xiao, Wenqi Shao, Wenwei Zhang, Kai Chen, and Ping Luo. Gpt4roi: Instruction tuning large language model on region-of-interest. arXiv preprint arXiv:2307.03601, 2023c.

Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li, Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo, Yaqian Li, Shilong Liu, et al. Recognize anything: A strong image tagging model. arXiv preprint arXiv:2306.03514, 2023d.

Yang Zhao, Zhijie Lin, Daquan Zhou, Zilong Huang, Jiashi Feng, and Bingyi Kang. Bubogpt: Enabling visual grounding in multi-modal llms. arXiv preprint arXiv:2307.08581, 2023.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.

Xueyan Zou, Zi-Yi Dou, Jianwei Yang, Zhe Gan, Linjie Li, Chunyuan Li, Xiyang Dai, Harkirat Behl, Jianfeng Wang, Lu Yuan, Nanyun Peng, Lijuan Wang, Yong Jae Lee, and Jianfeng Gao. Generalized decoding for pixel, image, and language. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15116–15127, 2023a.

Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li, Jianfeng Gao, and Yong Jae Lee. Segment everything everywhere all at once. arXiv preprint arXiv:2304.06718, 2023b.

- A DATA

Augmenting LLaVA data. The original LLaVA data only consists of questions and answers. We need to augment this data to make it match with our regular data format. We transformed the original answers in LLaVA into a part of the values field, then added an empty list for actions, and generated a thoughts using ChatGPT. The thoughts should indicate that the model can answer the question without invoking any tools. An example is shown in Table 8 in Appendix. We found the model cannot invoke tools if we did not unify the two data formats.

LLaVA Instruction Data Augmented LLaVA Instruction Data

User What are the colors of the bus in the image? What are the colors of the bus in the image? LLaVA-Plus The bus in the image is white and red. “thoughts” The questions can be answered by

the information in the context, without need any external tools.

###### “actions” []

“value” The bus in the image is white and red.

Table 8: Augmenting LLaVA instruction data.

Details on data generation. The pipeline to generate questions for visual prompts is shown in Table 4. The pipeline to generate questions with image-related parameters is shown in Table 5. An example of rewriting questions using GPT4 is shown in Table 9. The self-instruct example to generate multi-turn conversation for detection is shown in Table 10.

Prompts for ChatGPT/GPT4 to rewrite questions: Prompt System You will be given a question. Please rephrase the question to convey the same meaning but with

greater diversity. The question is asking a model to generate a caption for an image. Human generate a caption for the image AI Please create a short phrase that captures the essence of this photo. Human What would be an appropriate caption for this picture? AI What caption would you write for this image? Human Please examine this picture closely and provide a comprehensive caption that elucidates all of the

pertinent subjects, actions, contexts, and nuances that are crucial for understanding what is being depicted. Be thorough and exact in your detailed textual description.

GPT AI Please study this photo carefully and generate a wordy caption that comprehensively describes what is going on, ensuring to include all pertinent subjects, actions, contexts, nuances, and other specifics necessary for comprehensively representing the essence of the image.

Table 9: Examples of rewriting questions using GPT4.

User actions: Click at point [0.45, 0.89] User question: Perform segmentation based on the point.

Reformulating input questions.

Input question: Perform segmentation based on the point. input point: [0.45, 0.89]

Figure 4: The pipeline to generate questions for visual prompts.

Can you help to detect all {{classname}} in the image?

Filling ground truth categories.

Can you help to detect all persons, cars, dogs in the image?

ChatGPT/GPT4 rewrite.

Could you assist in identifying all individuals, vehicles, and dogs in the photograph and provide their coordinates?

Figure 5: The pipeline to generate questions with image-related parameters.

###### Prompts for ChatGPT/GPT4 to rewrite questions:

Prompt System You are an AI visual assistant that can analyze a single image. You receive five sentences, each describing the same image you are observing. In addition, specific object locations within the image are given, along with detailed coordinates. These coordinates are in the form of bounding boxes, represented as (x1, y1, x2, y2) with floating numbers ranging from 0 to 1. These values correspond to the top left x, top left y, bottom right x, and bottom right y. Design a conversation between you and a person asking about this photo. The answers should be in a tone that a visual AI assistant is seeing the image and answering the question. In the first question question the user ask you to detect some objects in an image. To finish this, you will use an open-world object detection model grounding dino to detect objects in the image. The detection model used is grounding dino. It is a model that can detect objects with captions in an image. For each question, write an answer that you think would be appropriate. It must include the <thoughts>, the <actions>, and <actions>. <thoughts> is you thinking about the question and the answer. <actions> is what you do to answer the question. The API name should be ”grounding dino” or None.<values> is the value of the answer. You must also include the output of the model, and how you would use it to answer the question. Ask diverse questions and give corresponding answers. Only include questions that have definite answers. Do not ask any questions that cannot be answered confidently. Give the answers as you have seen the image. Do not reveal the input information of the image. DO NOT say that you are given the captions and the objects in the image, JUST answer the question as if you are seeing the image for the first time, as well as the output of the model you invoked.

Human A plate filled with food sitting next to three glasses. A plate of meats and cheese’s next to three wine glasses. A table displaying wine glasses and a platter of assorted meats and cheeses. A white plate containing sliced ham and cheese sitting on a table next to wine glasses. The wine has been poured and ready to drink dining table: [0.0, 0.334, 0.893, 0.898] person: [0.0003, 0.002, 0.111, 0.138] person: [0.020, 0.0, 0.106, 0.109] wine glass: [0.360, 0.109, 0.212, 0.501] wine glass: [0.010, 0.107, 0.259, 0.612] wine glass: [0.646, 0.101, 0.220, 0.487] fork: [0.279, 0.588, 0.488, 0.225]

AI Question: Is it feasible to call on a detection model to scan the image and generate a comprehensive summary of glasses? Answer: <thoughts> It would be beneficial to utilize an open-set detection model to identify the presence of glasses. Integrating the models into the conversation would enhance the overall outcome and provide more accurate results. <actions> [{’API name’: ’grounding dino’, ’API params’: {’image’: ’<imagepath>’, ’caption’: ’glasses .’}}] <values> Yes, it is feasible to leverage a detection model to scan the image and generate a comprehensive summary of glasses. In this case, I can utilize an open-set detection model called grounding dino to detect all glasses present in the image. Based on the bounding box coordinates detected, the image contains three wine glasses. Here’s a summary of their positions:

- 1. The first wine glass is located towards the middle of the image, with bounding box coordinates of [0.360, 0.109, 0.212, 0.501]. This suggests that the glass is somewhat centrally placed on the table.
- 2. The second wine glass is located towards the left of the image, with bounding box coordinates of [0.010, 0.107, 0.259,

- 0.612]. This indicates that the glass is closer to the edge of the table or possibly near a person, given the presence of person bounding boxes in the image.

3. The third wine glass is located towards the right of the image, with bounding box coordinates of [0.646, 0.101, 0.220, 0.487]. This suggests that the glass is also closer to the edge of the table or near another person.

The bounding box coordinates suggest that all three glasses are roughly in a line, possibly indicating a formal dining setup. The glasses are also likely filled with wine.

====== Question: Can a detection model be used to identify and summarize the presence of forks in the image? Answer: <thoughts> It would be beneficial to utilize an open-set detection model to identify the presence of forks. Integrating the models into the conversation would enhance the overall outcome and provide more accurate results. <actions> [{’API name’: ’grounding dino’, ’API params’: {’image’: ’<imagepath>’, ’caption’: ’forks .’}}] <values> Yes, it is feasible to leverage a detection model to scan the image and generate a comprehensive summary of forks. In this case, I can utilize an open-set detection model called grounding dino to detect all forks present in the image. Based on the bounding box coordinates detected, the image contains one fork. Here’s a summary of its position:

- 1. The fork is located towards the middle of the image, with bounding box coordinates of [0.279, 0.588, 0.488, 0.225]. This suggests that the fork is placed near the plate of food, possibly indicating that someone is about to eat or is in the process of eating.

Human A girl with some food and drink at a table. A YOUNG GIRL EATING PIZZA DRESSED AS A EGYPTION A girl sitting around a table in a restaurant eating a pizza. A young girl eating pizza in a restaurant. A girl in a Cleopatra outfit eating pizza at a table.

chair: [0.394, 0.491, 0.248, 0.148] dining table: [0.007, 0.573, 0.974, 0.416] dining table: [0.663, 0.219, 0.101, 0.059] person: [0.497, 0.229, 0.488, 0.634]

AI Question: Perform an exhaustive search for the book in the image, and count the number of girls. Answer: <thoughts> Based on the information provided, the task requires me to search for the book in the image and count the number of girls. <actions> [{’API name’: ’grounding dino’, ’API params’: {’image’: ’<imagepath>’, ’caption’: ’the book and the girls in the image.’}}] <values> Certainly! To accomplish this task, I will utilize an object detection model to identify the book in the image and count the number of girls. Based on the results of grounding dino, I can identify the book and girls in the image. According to the available data, there is one ’book’ object detected and there are three ’person’ objects detected. Hence, there are three girls in the image.

====== Question: Hi, can you tell me what’s happening in this image? Answer: <thoughts> I don’t need to utilize any specific tools to provide an answer to your question. I can answer the question by reading the image. <actions> [] <value> Hello! According to the information provided, it seems like the image depicts a young girl, dressed as Cleopatra, who is eating pizza in a restaurant.

###### Table 10: Self-instruct examples of generating multi-turns conversations with in-context learning.

- B EXTENDED SKILLS

External Knowledge. To enable LMMs to gain knowledge beyond that encoded in pre-trained model weights, we use the CLIP search API to retrieve external knowledge from LIAON. We utilize the images Iq and questions Xq from the InfoSeek dataset, and generate the other fields of the training sequence by following the image-only skill data creation pipeline. Input images are considered as queries, and image-to-text retrieval is performed to get top-K items for each query. To encourage the LMM to leverage external knowledge, we only consider the subset of questions whose ground truth answers can be extracted or derived from the retrieved knowledge. This subset can be selected using ChatGPT that compares the answers and retrieved knowledge.

Generation. For image generation, we employ Stable Diffusion (SD) as the tool, and generate instruction data based on the JourneyDB dataset due to its high quality in language prompt and images. We ask ChatGPT to generate human-like instructions based on the original, detailed prompt for image generation, focusing on the scenarios where human-specified instructions are ambiguous and short, and thus cannot easily align with the prompt distribution of SD. Similarly, we use Instruct-Pix2Pix for image editing. The Instruct Pix2Pix dataset contains both instructions and prompts of source and target images. We directly use their editing instructions and follow the image-only skill data creation pipeline to fill the other fields.

Visual Prompts. The visual prompt data is constructed similarly to that for visual understanding skills, except that additional visual inputs, such as user-drawn points, sketches and boxes, are required. Take SAM as an example. A point is required as input for interactive segmentation. We simply generate a random point and then convert it into a text sequence, and append it to a user question to form a concatenated text sequence Xq, which is a standard format that LMMs such as LLaVA can deal with. Sometimes, a user point might correspond to segmented masks at multiple levels. To support this skill, we use Semantic-SAM (Li et al., 2023d) to create training data where the multi-granularity segmentation functionality is explicitly specified by instructions.

Skill Composition. The scenarios described so far are designed to create training samples for single-skill tasks. However, many real-world scenarios often require some compositions of several skills. To allow LLaVA-Plus to deal with such compositional tasks, we have curated instructionfollowing data for compositional skills as follows. (i) Various visual understanding results of the same image can be requested. To teach an LMM to learn to use multiple skills in a multi-turn human-AI interaction session, we generate instruction data by applying different tools (including detection, segmentation, tagging, and captioning) to the same image from COCO, combining the results with LLaVA instruction data, and then randomly mixing these datasets. This produces instruction data that simulates users’ behavior of using multiple tools to deal with real-world tasks. (ii) Interactive Segmentation + Inpainting. In one editing scenario, we ask a user to specify an area of an image with visual pointing along with language instruction. We then combine the SAM segmentation results and the SD inpainting results to create an instruction-following sample. (iii) Semantic Segmentation + Generation. In another image editing scenario, we ask a user to specify the spatial layout of an image, using an user-provided image and a language instruction. We then combine the OpenSeed semantic segmentation results and ControlNet conditional generation results to create an instruction-following sample. (iv) Image Generation/Editing + Social Media Post. It is time-consuming for human users to generate posts that contains both images and text. Thus, we use existing tools to create large amounts of multimodal posts for model tuning as follows. We use SD to generate an image, or Instruct Pix2Pix to edit an image. We then combine the image with its description generated by a pre-trained LMM to create a multimodal post.

- C RESULTS

Comparisons on COCO Caption. We aim to investigate the potential of the LMM in enhancing existing tools. A comparison of three distinct models on the COCO caption benchmark is presented in Table 11. We employed BLIP2 as our primary captioning tool and hence, use it as the benchmark model. Additionally, the original LLaVA is also included for reference. The enhanced LLaVA-Plus model refines BLIP2’s outputs, leading to richer details.

The table reveals that LLaVA-Plus outperforms the others in terms of the CLIP score. Intriguingly, both language models exhibit subpar performance on language-language metrics. A striking observation is the significantly lower CIDEr scores for these models when juxtaposed with BLIP2.

bleu1 bleu2 bleu3 bleu4 meteor rouge l CIDEr SPICE CLIP score RefCLIP score

BLIP2 77.0 62.1 48.0 36.4 28.2 57.2 123.5 22.3 0.788 0.836 LLaVA-Plus-7B 50.8 35.4 23.8 15.7 27.7 44.5 31.0 22.9 0.815 0.813 LLaVA-7B 21.1 13.7 8.4 5.1 19.3 21.5 0.0 17.6 0.785 0.785

Table 11: Comparisons on COCO Caption.

False Positives of Grounding DINO Grounding DINO, despite its commendable object detection prowess, occasionally exhibits hallucinations, leading it to generate false positive instances. Our LLaVA-Plus model, capable of simultaneously analyzing model outputs and image content, holds the potential to reduce such false positives.

To harness this potential, we crafted examples using negative prompts from COCO and directed the model to eliminate false positive outputs. We subsequently evaluated the model on the first 100 images from the COCO validation set. By using all negative categories of an image as prompts, we gauged the presence of false positive objects. The results are tabulated in Table 12.

The results show that Grounding DINO has a high possibility of resulting in false positive examples. With the LLaVA-Plus model, it can help to reduce the false positive rate significantly.

#Ins. FP #Img. FP

Grounding DINO 90 41 LLaVA-Plus-7B 20 12

Table 12: Comparisons of false positive of Grounding DINO. ‘#Ins. FP’ is the number of false positive examples in a whole test set, while the ‘#Img. FP’ is the number of images that have false positive examples. The test set is the first 100 images of COCO val set.

- D EXAMPLE SCENARIOS We show more scenarios of LLaVA-Plus in leveraging new skills to improve visual chat experience.

Object Detection for Visual Chat. Figure 6 compares object localization capability of LLaVA-Plus with Bard, Bing Chat, MM-REACT and LLaVA. It turns out the commercial visual chat do not have the ability to tell the object spatial location, while LLaVA-Plus can successfully identify the object location and thus describe the outdoor scene and activity correctly.

Detection and Segmentation in Contexts. Figure 7 (a) shows an example to detect and count the number of objects. Figure 7 (b) shows a real-life scenarios to pick up the appropriate tools and teach the users how to use them. Compared langauge-output-only LMM such as LLaVA/GPT-V, identify and visualization the location of object is an more intuitive approach for users to comprehend. Figure 8 provides object segmentation results, but enriched with language descriotion at the instance level. It is the synergy of LMM and segmentation that improve the enhanced fine-grained understanding.

External Knowledge In Figure 9, we compare LLaVA-Plus and LLaVA in terms of generating response with detailed facts and entities. The retrieval external knowledge of LLaVA-Plus introduces more relevant information that allows LLaVA-Plus to ground in generation.

Image Generation. In Table 10, we show that LLaVA-Plus can produce detailed SD-favored language prompts for image generation, based on the high-level and brief requests. This can help improve image generation quality.

Interactive Image Editing. Figure 11 demonstrate the multi-turn interactive image segmentation and editing capabilities. By leveraging OpenSEED, LLaVA-Plus can apply the skill of full-image semantic segmentation to group pixels of the same object together, providing the spatial layout of the scene. With further requests to produce new images that follow the same layout but change other aspects, the corresponding editing skills can be executed, through InstructPix2Pix and ControlNet.

Multimodal Social Meida Post. In Figure 12, the four seasons of the same scene are used as instructions to ask LLaVA-Plus to provide the edited images and attractive texts. Another example on fireworks is shown in Figure 13

Visual Prompt for Interactions. Figure 14 demonstrates the use of semantic SAM to support visual pointing on the image from humans, after which multiple segmentation masks at different levels are shown. Figure 15 demonstrates the visual referring segmentation capabilities. LLaVA-Plus allows humans to specify the segmentation intents on the object of interest with the selected regions from another image. This is useful because some concepts can be hard described in language, but easier to express with reference visual regions.

Human: Is there a frisbee in the image, and if so, could you please provide its location?

[Figure 59]

[Figure 60]

Bard: I have examined the image you sent, and I can confirm that there is no frisbee in the image. The image shows two men standing in a park, but there is no flying disc or any other object that could be mistaken for a frisbee. I hope this answers your question! Let me know if you have any other questions.

Bing Chat: Yes, there is a frisbee in the image. It is located near the bottom left corner of the image, on the grass. It is orange in color and has a white stripe around it.

[Figure 61]

MM-REACT: To summarize, This is an image of a couple of men playing frisbee.One man is wearing a white shirt and black pants, and the other man is in a blue jacket. The scene is outdoors with grass, trees, and sky in the background. It seems like they are standing on a lawn or a field. Is there a frisbee in the image, and if so, could you please provide its location? I‘m sorry, but the information provided does not mention a frisbee or its location in the image.

[Figure 62]

[Figure 63]

LLaVA: Yes, there is a frisbee in the image. It is located on the ground, near the two men who are playing with it.

|LLaVA-Plus: "thoughts!" The task requires identifying the presence of a frisbee in the image and providing its location. "actions"" [{"API_name": "grounding_dino", "API_params": {"caption": "the frisbee ."}}] "value#”<br><br>[Figure 64]<br><br>[Figure 65]<br><br>[Figure 66]<br><br>I will use grounding_dino to help to answer the question. Please wait for a moment. grounding_dino model outputs: {'boxes': [[0.76, 0.64, 0.82, 0.66]], 'logits': [0.49], 'phrases': ['the frisbee']} Please summarize the model outputs and answer my first question: Is there a frisbee in the image, and if so, could you please provide its location?<br><br>"thoughts!" Thanks to the output of grounding_dino. I can answer the question better. "actions"" [] "value#”<br><br>[Figure 67]<br><br>[Figure 68]<br><br>[Figure 69]<br><br>Yes, there is a frisbee in the image. Based on the results of the grounding_dino model, the frisbee is located in the middle of the image, with bounding box coordinates of [0.76, 0.64, 0.82, 0.66]. This suggests that the frisbee is in the air, possibly being thrown or caught by one of the players.<br><br>[Figure 70]|
|---|

Figure 6: The comparison of detection capabilities and its impact on visual chat. LLaVA-Plus is the only system that is able to detect the frisbee and leverage the location information to tell the motion and status of the object as well as human activity, revealing the importance of object localization on the LMM response.

[Figure 71]

detect birds and human head in the image, then tell how many birds in it

"thoughts!" To answer the question, I must utilize the pre-trained grounding_dino model specifically designed for open-set detection. "actions"" [{"API_name": "grounding_dino", "API_params": {"caption": "bird . human . head"}}] "value#" I will use grounding_dino to help to answer the question. Please wait for a moment. grounding_dino model outputs: {'boxes': [[0.3, 0.12, 0.7, 0.91], [0.15, 0.77, 0.27, 0.94], [0.83, 0.14, 0.92, 0.28], [0.64, 0.05, 0.72, 0.17], [0.82, 0.77, 0.97, 0.97], [0.05, 0.06, 0.15, 0.2], [0.66, 0.71, 0.77, 0.87], [0.14, 0.2, 0.27, 0.39], [0.26, 0.04, 0.33, 0.13], [0.86, 0.35, 0.98, 0.51], [0.1, 0.45, 0.2, 0.59], [0.74, 0.39, 0.82, 0.5], [0.83, 0.58, 0.89, 0.67], [0.06, 0.66, 0.14, 0.77]], 'logits': [0.61, 0.45, 0.45, 0.45, 0.44, 0.44, 0.42, 0.41, 0.41, 0.41, 0.4, 0.39, 0.39, 0.39], 'phrases': ['head', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird', 'bird']} Please summarize the model outputs and answer my first question: detect birds and human head in the image, then tell how many birds in it "thoughts!" Thanks to the output of grounding_dino. I can answer the question better. "actions"" [] "value#”

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

Based on the output of the grounding_dino model, there are 14 boxes detected in the image. Out of these boxes, 13 have been labeled as "bird" and 1 has been labeled as "head". The coordinates of the boxes have also been provided. The boxes labeled as "bird" are located at [0.3, 0.12, 0.7, 0.91], [0.15, 0.77, 0.27, 0.94], [0.83,

[Figure 78]

0.14, 0.92, 0.28], [0.64, 0.05, 0.72, 0.17], [0.82, 0.77, 0.97, 0.97], [0.05, 0.06, 0.15, 0.2], [0.66, 0.71, 0.77, 0.87], [0.14, 0.2, 0.27, 0.39], [0.26, 0.04, 0.33, 0.13], [0.86, 0.35, 0.98, 0.51], [0.1, 0.45, 0.2, 0.59], [0.74, 0.39, 0.82, 0.5], [0.83, 0.58, 0.89, 0.67], [0.06, 0.66,

0.14, 0.77]. The box labeled as "head" is located at 0.3, 0.12, 0.7, 0.91. Therefore, the image contains 13 birds and 1 human head.

(a) Detection and counting

[Figure 79]

My bike is broken. I want to use a wrench to fix it. Can you show me what is the location of wrenches and how to use it

"thoughts!" I am unable to give a direct response to the user, so I will have to use the pre-trained open-set detection model grounding_dino to reply. "actions"" [{"API_name": "grounding_dino", "API_params": {"caption": "wrench . bicycle"}}] "value#" I will use grounding_dino to help to answer the question. Please wait for a moment. grounding_dino model outputs: {'boxes': [[0.64, 0.28, 0.69, 0.5], [0.69, 0.28, 0.73, 0.46], [0.81, 0.27, 1.0, 0.51], [0.31, 0.14, 0.69, 0.24], [0.08, 0.5, 0.21, 0.67]], 'logits': [0.42, 0.41, 0.4, 0.34, 0.31], 'phrases': ['wrench', 'wrench', 'wrench', 'wrench', 'wrench']} Please summarize the model outputs and answer my first question: My bike is broken. I want to use a wrench to fix it. Can you show me what is the location of wrenches and how to use it "thoughts!" Thanks to the output of grounding_dino. I can answer the question better. "actions"" [] "value#”

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

Based on the output of the grounding_dino model, there are five wrenches in the image, and their locations are as follows:

- - The first wrench is located in the upper right corner, with coordinates (0.64, 0.28) for the top le" point and (0.69, 0.5) for the bottom right point.
- - The second wrench is located in the upper middle area, with coordinates (0.69, 0.28) for the top le"

- point and (0.73, 0.46) for the bottom right point.

- The third wrench is located in the lower le" corner, with coordinates (0.81, 0.27) for the top le"

- point and (1.0, 0.51) for the bottom right point.

- - The fourth wrench is located in the middle le" area, with coordinates (0.31, 0.14) for the top le" point and (0.7, 0.24) for the bottom right point.
- - The fi"h wrench is located in the upper middle area, with coordinates (0.08, 0.5) for the top le" point and (0.21, 0.67) for the bottom right point.

[Figure 86]

As for how to use a wrench, it is a tool used to tighten or loosen nuts and bolts. To use a wrench, you need to hold it with the handle facing the direction you want to turn the nut or bolt, and then turn the handle to apply pressure on the fastener. The wrench is typically marked with numerical degrees to indicate the amount of torque applied.

(b) Detection for real-life actions Figure 7: Detection for counting and actions.

[Figure 87]

##### Segmentation

[Figure 88]

###### Please detect and segment the cat and computer from the image.

"thoughts!" To accurately detect and segment the cat and computer in the image, I can leverage the grounding_dino model for open-set detection and the segmentation model SAM. "actions"" [{"API_name": "grounding_dino+sam", "API_params": {"caption": "cat and computer ."}}] "value#" I will use grounding_dino+sam to help to answer the question. Please wait for a moment. grounding_dino+sam model outputs: {'boxes': [[0.0, 0.0, 0.59, 0.75], [0.23, 0.2, 1.0, 1.0], [0.01, 0.01, 0.99, 0.99]], 'logits': [0.55, 0.48, 0.31], 'phrases': ['cat', 'computer', 'cat']} Please summarize the model outputs and answer my first question: Please detect and segment the cat and computer from the image. "thoughts!" Thanks to the output of grounding_dino+sam. I can answer the question better. "actions"" [] "value#”

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

###### Segmentation

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

By using the grounding_dino model for open-set detection and the segmentation model SAM, we can effectively detect and segment the cat and computer in the image. The cat is laying down on a

pillow, resting its paw on the laptop, creating a cozy and captivating scene. The computer, in this case a laptop, is positioned next to the cat, reinforcing the connection between the two subjects. The segmentation model SAM enables us to isolate and analyze these elements, providing a detailed and comprehensive understanding of the image composition.

[Figure 97]

###### detect and segment dog and phone from the image, and describe them

"thoughts!" To accurately detect and segment the dog and phone in the image, we can leverage the grounding_dino model for open-set detection and the segmentation model SAM. Integrating these models will enable us to generate a comprehensive summary of the segmented objects. "actions"" [{"API_name": "grounding_dino+sam", "API_params": {"caption": "dog and phone ."}}] "value#" I will use grounding_dino+sam to help to answer the question. Please wait for a moment. grounding_dino+sam model outputs: {'boxes': [[0.18, 0.01, 0.99, 1.0], [0.42, 0.08, 0.76, 1.0]], 'logits': [0.72, 0.58], 'phrases': ['dog', 'phone']} Please summarize the model outputs and answer my first question: detect and segment dog and phone from the image, and describe them "thoughts!" Thanks to the output of grounding_dino+sam. I can answer the question better. "actions"" [] "value#”

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

By utilizing the grounding_dino model for open-set detection and the segmentation model SAM, we can eﬀectively segment and detect the dog and phone in the image. The dog is a small pug with a wrinkly face, and it is positioned close to the phone. The phone is held up to the dog's ear, as if the dog is talking on the phone. The segmentation model SAM allows us to isolate and analyze these objects, providing a detailed and comprehensive summary of the dog and phone in the image. Integrating these models enhances the accuracy and precision of the segmentation process, resulting in a deeper understanding of the scene depicted in the image.

[Figure 104]

###### Figure 8: Language-enriched detection and description.

LLaVA-Plus

[Figure 105]

Describe the food in the image? search on the internet

"thoughts!" To tackle that question, one needs the assistance of web research using clip retriever. "actions"" [{"API_name": "clip", "API_params": {}}] "value#" I will use clip to help to answer the question. Please wait for a moment. clip model outputs: {'retrieval_results': [{'caption': ' ? ', 'similarity': 0.98}, {'caption': '

[Figure 106]

[Figure 107]

[Figure 108]

', 'similarity': 0.95}, {'caption': 'Cách làm bánh nướng nhân đậu xanh 12', 'similarity': 0.95}, {'caption': 'Sự đặc biệt trong những chiếc bánh trung thu ngon và đẹp (2)', 'similarity': 0.94}, {'caption': '

', 'similarity': 0.94}, {'caption': 'Khám phá hương vị bánh trung thu các nước Châu Á', 'similarity': 0.94}, {'caption': ' , , . ', 'similarity': 0.94}, {'caption': ' ', 'similarity': 0.94}]} Please summarize the model outputs and answer my first question: Describe the food in the image? search on the internet "thoughts!" Thanks to the output of clip. I can answer the question better. "actions"" [] "value#"

[Figure 109]

[Figure 110]

[Figure 111]

Based on the retrieved information, it seems that the image is related to month cakes, which are a traditional food served during the Mid-Autumn Festival in China.

LLaVA

Describe the food in the image? The image features a variety of food items, including a plate of pastries, a bowl of soup, and a cup of tea. There are also some cakes and a flower arrangement on the table. what it is the food? The food in the image consists of a plate of pastries, a bowl of soup, a cup of tea, and some cakes. The pastries are likely a type of pastry or dessert, while the soup could be a savory dish. The cakes are also dessert items, and the tea is a beverage.

###### LLaVA-Plus

what's the image? search on the internet

[Figure 112]

"thoughts!" I need to search for more information on the internet, which can be done by using the clip retriever. "actions"" [{"API_name": "clip", "API_params": {}}] "value#"

[Figure 113]

[Figure 114]

[Figure 115]

I will use clip to help to answer the question. Please wait for a moment. clip model outputs: {'retrieval_results': [{'caption': 'Fresk Giotto - Pocałunek Judasza', 'similarity': 0.91}, {'caption': 'Северное возрождение живопись эпохи картины художники', 'similarity': 0.91}, {'caption': 'Kiss of Judas (1304–06), fresco by Giotto, Scrovegni Chapel, Padua, Italy', 'similarity': 0.91}, {'caption': 'Giotto di Bondone - 21b Prendimiento de Jesús.jpg', 'similarity': 0.91}, {'caption': 'Giotto - Kiss of Judas. Fresco in Scrovegni Chapel, Padua, Italy (1304–06)', 'similarity': 0.91}]} Please summarize the model outputs and answer my first question: what's the image? search on the internet "thoughts!" Thanks to the output of clip. I can answer the question better. "actions"" [] "value#”

[Figure 116]

[Figure 117]

[Figure 118]

Based on the retrieved information, it is highly likely that the image belongs to the fresco painting "Kiss of Judas" by Giotto, which is located in the Scrovegni Chapel in Padua, Italy.

###### LLaVA

what's the image? The image is a painting depicting a group of people gathered around Jesus, with some of them holding torches.

Figure 9: External knowledge retrieval help improve the entity and fact based responses.

generate a view of the city skyline of Seattle in a future and technique style

generate a view of the city skyline of Seattle in a future and technique style

generate a view of the city skyline of Shanghai in a cyberpunk style

generate a view of the city skyline of Shenzhen in a future and technique style

generate a view of the city skyline of

Human Instructions

downtown Seattle in a sketch style

seattle city skyline future technique

view of the city skyline of seattle, futuristic, digital art, technique, photorealistic, high quality, resolution, 8 k

view of the city skyline of shenzhen in a future and technique style, digital art, cinematic, hyperrealistic 3 5 mm lens, aerial photography, volumetric lighting, by alena aenami and max hay

view of the city skyline of shanghai, china, cyberpunk style, photorealistic, octane render, by wlop, pixiv

view of the city skyline of downtown seattle, sketch

SD Prompt Generated by LMM (LLaVA-Plus)

|[Figure 119]|
|---|

|[Figure 120]|
|---|

|[Figure 121]|
|---|

|[Figure 122]|
|---|

|[Figure 123]|
|---|

Generated Images

- Figure 10: LLaVA-Plus improves SD-favored language prompt from user instructions for image generation.

Is it possible to perform semantic segmentation for the image?

[Figure 124]

"thoughts!" I have decided to utilize the OpenSeeD model in order to improve the performance of semantic segmentation on the image. "actions"" [{"API_name": "openseed", "API_params": {}}] "value#"

[Figure 125]

[Figure 126]

[Figure 127]

I will use openseed to help to answer the question. Please wait for amoment. controlnet model outputs: {'message': 'The controlnet has processed the image.'}

Please summarize the model outputs and answer my first question: Is it possible to perform semantic segmentation for the image?

"thoughts!" Thanks to the output of controlnet. I can answer the question better. "actions"" [] "value#"

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

Yes, I can definitely help you with semantic segmentation for the image! I have utilized the powerful OpenSeeD model to enhance the accuracy and performance of the segmentation process. Here are the results I have obtained for you to analyze and explore.

Turnthe image into cartoonstyle.

[Figure 132]

"thoughts!" To enhance the image editing process, I have collaborated with the instruct pix2pix model. "actions"" [{"API_name": "instruct-pix2pix", "API_params": {"prompt": "Make the image look like a cartoon."}}] "value#"

[Figure 133]

[Figure 134]

I will use instruct-pix2pixto help to answer the question. Please wait for amoment. instruct-pix2pixmodel outputs: {'message': 'The instruct-pix2pixhas processed the image.'}

[Figure 135]

Please summarize the model outputs and answer myfirst question: Turnthe image into cartoonstyle.

"thoughts!" Thanks to the output of instruct-pix2pix. I can answer the question better. "actions"" [] "value#"

[Figure 136]

[Figure 137]

[Figure 138]

Sure, I can definitely help you with that! I've used the instruct pix2pix model to turn the image into a cartoon style. Here are the results for you to check out. Let me know if you need any further assistance!

Generate a similar image with cartoon style.

"thoughts!" I will utilize the ControlNet model in order to generate images. "actions"" [{"API_name": "controlnet", "API_params": {"prompt": {"caption": "A table topped with plates of food in a cartoon style."}}}] "value#"

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

I will use controlnet to help to answer the question. Please wait for amoment. controlnet model outputs: {'message': 'The controlnet has processed the image.'} Please summarize the model outputs and answer myfirst question: Generate asimilar image with cartoonstyle.

[Figure 143]

"thoughts!" Thanks to the output of controlnet. I can answer the question better. "actions"" [] "value#"

[Figure 144]

[Figure 145]

Certainly! I've used the ControlNet model to generate a similar image with a cartoon style. Please take a look at the results below.

[Figure 146]

Is it possible to perform semantic segmentation for the image?

"thoughts!" I will utilize the OpenSeeD model to assist in conducting semantic segmentation for the image. "actions"" [{"API_name": "openseed", "API_params": {}}] "value#" I will use openseed to help to answer the question. Please wait for a moment. controlnet model outputs: {'message': 'The controlnet has processed the image.'} Please summarize the model outputs and answer my first question: Is it possible to perform semantic segmentation for the image? "thoughts!" Thanks to the output of controlnet. I can answer the question better. "actions"" [] "value#"

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

Of course! I can definitely assist you with performing semantic segmentation for the image. To accomplish this, I will utilize the powerful OpenSeeD model. It has the capability to accurately segment objects in the image based on their semantic meaning. I have already triggered the model and the results are now available for you to review. Please take a look below.

[Figure 153]

Please generate a similar image of a street in India.

"thoughts!" I will utilize the ControlNet model for image generation. "actions"" [{"API_name": "controlnet", "API_params": {"prompt": {"caption": "A street in India with people and cars."}}}] "value#" I will use controlnet to help to answer the question. Please wait for a moment. controlnet model outputs: {'message': 'The controlnet has processed the image.'} Please summarize the model outputs and answer my first question: Please generate a similar image of a street in India. "thoughts!" Thanks to the output of controlnet. I can answer the question better. "actions"" [] "value#"

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

Certainly! I've used the powerful ControlNet model to generate an image of a street in India. The results are now ready for you to see below.

Please generate a similar image of a street in New York.

"thoughts!" I will utilize the ControlNet model for image generation. "actions"" [{"API_name": "controlnet", "API_params": {"prompt": {"caption": "A street in New York City with cars and people."}}}] "value#"

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

I will use controlnet to help to answer the question. Please wait for a moment. controlnet model outputs: {'message': 'The controlnet has processed the image.'} Please summarize the model outputs and answer my first question: Please generate a similar image of a street in New York.

[Figure 165]

"thoughts!" Thanks to the output of controlnet. I can answer the question better. "actions"" [] "value#"

[Figure 166]

[Figure 167]

Sure! I've used the advanced ControlNet model to generate an image of a street in New York. Here are the results for you to see.

- Figure 11: Semantic segmentation and mask-based conditional image generation of LLaVA-Plus. Purple is human questions, green is LLaVA-Plus response. The semantic segmentation task is fullfilled via OpenSEED. Based on the segmented images, new editing instructions and history, InstructPix2Pix and ControlNet can be called to complete the tasks. The captions of the target edited images are generated by LLaVA-Plus, revealing the unique advantage of LMM for tool use.

###### image like spring. then generate some attractive texts for Instagram posts

|[Figure 168]|make the<br><br>"thoughts!" To "actions"" "value#" I will use instruct-pix2pix Please summarize<br><br>[Figure 169]<br><br>[Figure 170]<br><br>[Figure 171]<br><br>"thoughts!" "actions"" [] "value#”<br><br>[Figure 172]<br><br>[Figure 173]<br><br>[Figure 174]<br><br>Absolutely! And now, let|
|---|---|
|4 seasons: summ|1."Spring has #SpringVibes<br>2."Feeling<br><br><br>er|
|[Figure 175]|#NaturePhotography<br><br>3."This image Let's take a<br>4."Who needs all its glory.<br>5."Spring has you feel the<br>|

accomplish the task, I need to utilize the specialized abilities of the instruct pix2pix model for the required image editing. [{"API_name": "instruct-pix2pix", "API_params": {"prompt": "make the image like spring."}}]

instruct-pix2pix to help to answer the question. Please wait for a moment. model outputs: {'message': 'The instruct-pix2pix has processed the image.'}

the model outputs and answer my first question: make the image like spring. then generate some attractive texts for Instagram posts Thanks to the output of instruct-pix2pix. I can answer the question better.

I've got you covered. Let's spruce up that image. See the enhancements below. me share some Instagram captions to go with the new image: s sprung, and this stunning lake view is proof! Can you feel the sunshine and fresh air? &'(

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

#NatureLover" rejuvenated by this breathtaking image of a serene lake in springtime. Mother Nature truly is a wonder! &) ography #SpringFeels"

[Figure 180]

[Figure 181]

transported us straight to a peaceful spring morning with its vibrant colors and tranquil surroundings. moment to appreciate the beauty of nature! %' #SpringMood #NatureLove"

[Figure 182]

[Figure 183]

s a spring break when you have a view like this? * This image perfectly captures the beauty of nature in &( #NatureIsAmazing #Springtime"

[Figure 184]

[Figure 185]

[Figure 186]

arrived with a bang, as evidenced by this picture-perfect view of pristine waters and lush greenery. Can excitement in the air? &'+ #SpringFeels #NatureBeauty"

[Figure 187]

[Figure 188]

[Figure 189]

make the image like summer. then generate some attractive texts for Instagram posts

| |"thoughts!<br><br>[Figure 190]<br><br>"actions" "value#"<br><br>[Figure 191]<br><br>[Figure 192]<br><br>"thoughts! "actions"" [] "value#”<br><br>[Figure 193]<br><br>[Figure 194]<br><br>[Figure 195]|
|---|---|
|4 seasons: autum|n|
|[Figure 196]|#PeacefulRetreat" #SummerCruising #IslandParadise"|

" In order to successfully conclude the task at hand, I must employ the specialized capabilities of the instruct pix2pix model to facilitate the necessary image editings.

" [{"API_name": "instruct-pix2pix", "API_params": {"prompt": "make the image like summer."}}]

I will use instruct-pix2pix to help to answer the question. Please wait for a moment. instruct-pix2pix model outputs: {'message': 'The instruct-pix2pix has processed the image.'} Please summarize the model outputs and answer my first question: make the image like summer. then generate some attractive texts for Instagram posts

" Thanks to the output of instruct-pix2pix. I can answer the question better.

Great news! I'll utilize the instruct pix2pix tool to help make the image look like a beautiful summer day. Check out the amazing result below. Now, let's explore some Instagram captions that perfectly capture the essence of this new image:

[Figure 197]

- 1."Feeling the summer vibes on this mountain pier! ☀-# #MountainPier #SummerFeels #NatureLovers"

[Figure 198]

[Figure 199]

[Figure 200]

- 2."Escape to paradise on this mountain pier piercing the water! %#) #IslandLife #SummerVibes #TravelGoals"

[Figure 201]

[Figure 202]

[Figure 203]

- 3."Basking in the tranquility of summer on this breathtaking mountain pier ☀.% #SummerDays #NatureLover treat"

[Figure 204]

[Figure 205]

[Figure 206]

- 4."Dive into the crystal clear waters of this stunning mountain pier this summer #/. #TropicalVibes uising #BucketListGoals"

[Figure 207]

[Figure 208]

[Figure 209]

- 5."Immerse yourself in the beauty of nature on this serene mountain pier '#% #SummerMood #NatureIsAmazing dise"

[Figure 210]

[Figure 211]

[Figure 212]

I hope these captions spark some creativity for your Instagram posts!

image like autumn. then generate some attractive texts for Instagram posts

| |make the<br><br>"thoughts!" To "actions"" "value#" I will use instruct-pix2pix Please summarize<br><br>[Figure 213]<br><br>[Figure 214]<br><br>[Figure 215]<br><br>"thoughts!" "actions"" [] "value#”<br><br>[Figure 216]<br><br>[Figure 217]<br><br>[Figure 218]|
|---|---|
|4 seasons: autum|Great! I'll captions for 1."Embrace<br><br>n|
|[Figure 219]|2."Who says #ScenicRiver"<br>3."Take a<br>4."Why rush<br>5."Get lost #NatureLover"<br>|

complete the image editing task, I need the instruct pix2pix model. [{"API_name": "instruct-pix2pix", "API_params": {"prompt": "make the image like autumn."}}]

instruct-pix2pix to help to answer the question. Please wait for a moment. model outputs: {'message': 'The instruct-pix2pix has processed the image.'}

e the model outputs and answer my first question: make the image like autumn. then generate some attractive texts for Instagram posts Thanks to the output of instruct-pix2pix. I can answer the question better.

[Figure 220]

use instruct pix2pix to edit the image, and you can see the changes below. Now, let's cra! some Instagram this new image: the cozy vibes of autumn with this stunning lake view! !" #AutumnFeels #LakeViews #NatureLover"

[Figure 221]

[Figure 222]

fall's colors aren't pretty? Check out this vibrant view of the lake! !# #AutumnVibes #LakeLife

[Figure 223]

[Figure 224]

r" moment to soak in the vivid shades of autumn with this breathtaking view! !$ #AutumnHues

[Figure 225]

[Figure 226]

#NaturePhotography #SereneLocation" through the season when you have views like this? !% #AutumnDays #LakeEscape #GetawayGoals" in the beauty of autumn nature with this captivating view of the lake! "# #AutumnWonder #LakeLife r"

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

make the image like winter. then generate some attractive texts for Instagram posts

"thoughts!" The AI will rely on the instruct pix2pix model for the image modifications. "actions"" [{"API_name": "instruct-pix2pix", "API_params": {"prompt": "make the image like winter."}}] "value#" I will use instruct-pix2pix to help to answer the question. Please wait for a moment. instruct-pix2pix model outputs: {'message': 'The instruct-pix2pix has processed the image.'} Please summarize the model outputs and answer my first question: make the image like winter. then generate some attractive texts for Instagram posts "thoughts!" Thanks to the output of instruct-pix2pix. I can answer the question better. "actions"" [] "value#”

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

Great! Let's enhance the image to make it look like winter. Please take a look at the edited version below. Additionally, I have also prepared some Instagram captions that can accompany the new image. Please see the options below:

[Figure 237]

- 1."Experience winter wonderland vibes with this stunning image captured by Andrew MacLean. #winterphotography #naturelover"
- 2."Embrace the winter season with this breathtaking image! Andrew MacLean never fails to amaze us. #winterwonderland #naturephotography"
- 3."Let this magical winter scene transport you to a world of wonder and awe. Andrew MacLean has truly captured the essence of winter in this image. #wintermagic #naturelovers"
- 4."This winter wonderland is sure to spread warmth and joy. Andrew MacLean's image captures the beauty of the season perfectly. #winterbliss #naturephotography"
- 5."Indulge in the serene beauty of winter with this captivating image by Andrew MacLean. It will surely leave you mesmerized and inspired. #winterchills #naturelover"

###### Figure 12: Multimodal social media post by editing an image and writing a message. Four season of the same image are considered to edit and further associate the text to attract attention of Instagram.

[Figure 238]

###### i want to post a message on Instagram. add some firework to the image, and write an attractive post for my ins.

"thoughts!" My success in completing the task depends on my capability to utilize the functionalities of the instruct pix2pix model, which has been tailored for this type of image editing. "actions"" [{"API_name": "instruct-pix2pix", "API_params": {"prompt": "add some fireworks"}}] "value#" I will use instruct-pix2pix to help to answer the question. Please wait for a moment. instruct-pix2pix model outputs: {'message': 'The instruct-pix2pix has processed the image.'} Please summarize the model outputs and answer my first question: i want to post a message on Instagram. add some firework to the image, and write an attractive post for my ins. "thoughts!" Thanks to the output of instruct-pix2pix. I can answer the question better. "actions"" [] "value#”

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

I have used Instruct pix2pix technique to make the necessary edits and the revised image is now available for your viewing pleasure. Now, let me share with you some exciting captions to go with our new image on Instagram:

[Figure 245]

- 1."Paris shines extra bright with the added sparkle of fireworks $✨ #CityOfLight #FireworksFest"

[Figure 246]

[Figure 247]

- 2."Bonjour, Paris! & It's even more magical with the dazzling fireworks illuminating the night sky ' #TravelGoals"

[Figure 248]

[Figure 249]

- 3."Dreaming of a Parisian adventure? (& Our stunning new image with fireworks will inspire your wanderlust )* #Wanderlust"

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

- 4."Who needs to wait for New Year's Eve to feel the magic of fireworks? $' Our new image of the Eiﬀel Tower sparkling in the night sky will transport you to the City of Love ❤ #BonJourParis"

[Figure 254]

[Figure 255]

[Figure 256]

- 5."Experience the enchanting beauty of Paris like never before! , Our new image with added fireworks will transport you to a fairytale setting - #MidnightMagic"

[Figure 257]

[Figure 258]

###### Figure 13: Multimodal social media post on fireworks.

[Figure 259]

The input point: [0.5, 0.32]

###### segment with multi granularity.

“thoughts!” I will use the Semantic SAM as my segmentation tool for these input point. "actions"" [{"API_name": "semantic-sam", "API_params": {"point": [0.5, 0.32]}, "value": "I will use semantic-sam to help to answer the question. Please wait for a moment."}] "value#" I will use semantic-sam to help to answer the question. Please wait for a moment. semantic-sam model outputs: {'message': 'The semantic-sam has processed the image.'} Please summarize the model outputs and answer my first question: segment with multi granularity. The input point: [0.5, 0.32] "thoughts!" Thanks to the output of semantic-sam. I can answer the question better. "actions"" [] "value#”

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

Certainly! We tackle the challenge by utilizing the Semantic-SAM, which divides subjects using the provided point. When 0.5 and 0.32 are inputted, 6 masks are generated. The outcomes are displayed below.

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

Figure 14: Multi-granularity segmentation with an user input point, using Semantic SAM.

Input

Is it possible to perform segmentation based on the stroke on the reference image?

[Figure 272]

"thoughts!" I have decided to utilize the SEEM model in order to improve the performance of segmentation on the image. "actions"" [{"API_name": "seem", "API_params": {}}] "value#"

[Figure 273]

[Figure 274]

[Figure 275]

Reference

I will use semmto help to answer the question. Please wait for a moment.

[Figure 276]

seem outputs: {'message': 'The seem has processed the image.'} Please summarize the model outputs and answer my first question: Is it possible to perform segmentation based on the stroke on the reference image?

"thoughts!" Thanks to the output of seem. I can answer the question better. "actions"" [] "value#”

[Figure 277]

Output

[Figure 278]

[Figure 279]

[Figure 280]

Yes, I can definitely help you with the segmentation for the image! I have utilized the powerful seem model to finish the task. See the results below!

Figure 15: Visual referring image segmentation of LLaVA-Plus. Purple is human questions, green is LLaVA-Plus response. Users can make a stroke on the reference image (a red curve) as the visual target to segment, LLaVA-Plus calls SEEM model to predict the corresponding masks in the target image.

