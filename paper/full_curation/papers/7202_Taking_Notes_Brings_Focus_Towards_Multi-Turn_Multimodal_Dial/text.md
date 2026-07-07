# arXiv:2503.07002v1[cs.CV]10Mar2025

## Taking Notes Brings Focus? Towards Multi-Turn Multimodal Dialogue Learning

Jiazheng Liu1* Sipeng Zheng2 B¨orje F. Karlsson2 Zongqing Lu1,2† 1School of Computer Science, Peking University 2Beijing Academy of Artificial Intelligence

### Abstract

Multimodal large language models (MLLMs), built on large-scale pre-trained vision towers and language models, have shown great capabilities in multimodal understanding. However, most existing MLLMs are trained on single-turn vision question-answering tasks, which do not accurately reflect real-world human conversations. In this paper, we introduce MMDiag, a multi-turn multimodal dialogue dataset. This dataset is collaboratively generated through deliberately designed rules and GPT assistance, featuring strong correlations between questions, between questions and images, and among different image regions; thus aligning more closely with real-world scenarios. MMDiag serves as a strong benchmark for multi-turn multimodal dialogue learning and brings more challenges to the grounding and reasoning capabilities of MLLMs. Further, inspired by human vision processing, we present DiagNote, an MLLM equipped with multimodal grounding and reasoning capabilities. DiagNote consists of two modules (Deliberate and Gaze) interacting with each other to perform Chainof-Thought and annotations respectively, throughout multiturn dialogues. We empirically demonstrate the advantages of DiagNote in both grounding and jointly processing and reasoning with vision and language information over existing MLLMs.

### 1. Introduction

In recent years, large language models (LLMs) have achieved remarkable advances in various natural language applications, including chatbots [1, 2, 35], programming assistants [12], and rhetorical aides [14]. The success has further spurred the development of multimodal large language models (MLLM) [25, 51]. However, most existing MLLMs are trained as single black-box systems to handle multimodal instructions, often struggling with inaccuracies

*Work done as an intern at BAAI. †Correspondence to <zongqing.lu@pku.edu.cn>

and hallucinations, especially in complex multi-turn dialogues [42, 50]. We hypothesize such challenges arise from the MLLM’s difficulty in maintaining focus on target regions throughout the conversation, especially for highresolution images with overly long visual tokens. In this paper, we seek to address these issues by moving beyond a black-box approach to an explicit target-grounding solution. Here, we summarize two key goals for multiturn multimodal dialogue learning: ❶ “saliency tracking”, where the MLLM must keep tracking different relevant regions over the course of the dialogue, and ❷ “saliency recall”, where the model needs to consistently retain focus on the same critical information across multiple questionanswering (QA) rounds. For example, in the dialogue illustrated in Figure 1, completing the Minigrid [10] task requires the MLLM to accurately locate both the agent (i.e. “red triangle”) and the target (i.e. “purple key”) to answer the initial question. The following question then builds upon this information, requiring the MLLM to reason about the agent’s starting position based on the previously identified location of the key. This example illustrates the need for sustained and explicit grounding to multiple specific visual details in multi-turn multimodal dialogue.

To achieve these two goals, we draw inspiration from how humans maintain focus while studying. For instance, when working through documents, people may lose concentration, but can quickly refocus by using simple techniques such as jotting down notes or highlighting key points. Even basic marks, such as circling or underlining, can significantly enhance focus without requiring elaborate explanations. These visual cues guide attention, making it easier to track, recall, and revisit important information. In contrast, existing MLLMs lack such tracking capabilities, prompting us to ask: “Can an MLLM be designed to equip similar attention-guiding abilities? If so, what would that model design entail?”

To answer this question, we first review existing tuning methods for MLLMs and identify a critical gap: the lack of quality multi-turn multimodal dialogue datasets that adequately reason over both visual and text information. Exist-

(a) HumanInput: What should the red triangle agent do if it wants to pick up the purple key?

[Figure 1]

###### MLLM Reasoning I

To achieve this goal, we should first locate the red triangle agent. Then, we need to find the purple key. Finally, we should plan the path for the agent.

MLLM Output: The agent should first turn back, go forward four times, turn back and move one step forward to pick up the purple key.

Human Input: After that, What should the agent do next if it wants to reach the red ball below the purple key?

(b)

###### MLLM Reasoning II

To achieve this goal, we should first locate the red triangle agent. Since the agent is located at the place it reached the last step, we should focus on the same region at the last step.

[Figure 2]

Then, we need to find the red ball below the purple key. Finally, we should plan the path for the agent.

MLLM Output: To reach the red ball below the purple key, the agent should go straight down to reach that red ball.

- Figure 1. Multi-turn multimodal dialogue: (a) Saliency tracking. The MLLM needs to focus on both the red triangle agent and the purple key, which scatter on the image, to answer the question correctly. (b) Saliency recall. The MLLM needs to retain focus on the region where the agent will stop after the last question.

ing datasets, such as MMDU [28] and SciGraphQA [22], primarily consist of single-turn QA pairs, where most questions can be answered independently without relying on prior context. To bridge this gap, we introduce a novel dataset, MMDiag, designed as a foundational benchmark for challenging multi-turn multimodal dialogue. This dataset offers visually detailed multi-turn dialogues across a range of scenarios.

Furthermore, recent studies have introduced various modules to help keep focus in multi-turn multimodal dialogues. However, these methods either “zoom in” to progressively narrow focus areas with the aid of external grounding and OCR tools [32], or identify a single region of interest per question before generating an answer [38]. These approaches lead to severe limitations: the zoom-in method restricts the focus to smaller regions, potentially missing broader context, while the single-region method isolates specific areas, overlooking multiple relevant details that could enrich responses. To address these limitations, we propose DiagNote, a model designed to enhance focus and reasoning in multi-turn multimodal dialogue. DiagNote comprises two main modules: Deliberate and Gaze. The Deliberate module guides the Gaze module in dynamically adjusting the region of visual focus, while the Gaze module highlights crucial areas for subsequent processing by the Deliberate module. These two modules interact across multiple dialogue turns, emulating human visual

processing to produce an answer accompanied by optional reasoning and grounding steps. Through this interactive mechanism, DiagNote can achieve more effective reasoning with multimodal information, resulting in accurate and context-aware responses throughout multi-turn dialogues.

Our main contributions are summarized as follows: ❶ To address the need for robust multimodal grounding and reasoning, we build a new large-scale multi-turn multimodal dialogue dataset – MMDiag – across several QA scenarios (e.g. daily life and tabular data), using rule-based searching and GPT-4o-mini [31] capabilities. ❷ Inspired by human visual processing, we propose DiagNote and its two key modules – Deliberate and Gaze – to enhance the model’s capacity for multimodal information integration and reasoning. ❸ We evaluate DiagNote’s reasoning and grounding abilities on MMDiag and other benchmarks and the results demonstrate that the introduction of MMDiag and DiagNote significantly improves performance in multimodal conversations, while the MMDiag itself can also serve as a more challenging benchmark for this area.

### 2. Related Work

#### 2.1. Multimodal Large Language Models

The introduction of Transformers [27, 46] and large-scale training has greatly enhanced model capabilities, leading to the development of advanced vision encoders [33] and large language models (LLMs) [11, 44]. Building on these advancements, multimodal large language models (MLLMs) [25, 50] have demonstrated impressive performance across a wide range of multimodal tasks, and potential applications from VR/AR to game agents [16, 48].

An MLLM typically consists of three main components: modality encoders, modality interfaces, and LLMs [49]. Modality encoders and LLMs process modality information and language separately, and then modality interfaces align other modalities with the representations of the language. For modality interfaces, most approaches [21, 25] rely on learnable connectors. For modality encoders, research indicates that visual information processing (especially in terms of image resolution [30]) significantly affects the performance of MLLM. Additionally, certain models incorporate generators to produce other multimodal data, such as low-level actions [15] or images [50].

MLLM training commonly follows a two-stage process. In the first stage, vision and language modalities are aligned with the modality interface, often through pre-training on large datasets of image-caption pairs [5, 25, 36]. The second stage involves fine-tuning with visual question-answering (VQA) tasks [25, 40] for better LLMs’ capabilities of instruction following. This two-stage process is widely used in MLLMs like PALI-X [8], Qwen-VL [3], and LLaVA [25], forming a strong foundation for subsequent

MLLM advancements.

#### 2.2. Grounding and Reasoning Benefit MLLMs

MLLMs can perform in-context learning [4], enabling generalization to new tasks from a few examples. The Chain-of-Thought (CoT) [47] reasoning mechanism also allows models to approach problem-solving step-by-step. However, when faced with unfamiliar tasks, MLLMs sometimes rely excessively on the generalization capabilities of the LLM component, leading to overlooking visual details and hallucinations. To address these limitations, models like CogCoM [32] introduce “Chain of Manipulations”, allowing MLLMs to perform CoT reasoning with external grounding and OCR models, which enable incremental task-solving. Although this approach improves performance, it is limited to zooming in on specific areas and may miss key scattered details. Similarly, Visual CoT [38] enhances performance by focusing on a single region of interest per question, improving both answer accuracy and visual grounding. However, a single grounding and reasoning round is often insufficient for complex, multistep problems. To overcome these challenges, we propose two modules: Deliberate for reasoning and Gaze for grounding, enabling multiple rounds of CoT reasoning. This iterative approach allows for better problem-solving by refining both grounding and reasoning across interactions, making it more effective in handling complex tasks, like multi-turn multimodal QAs.

#### 2.3. Multi-Turn Multimodal Dialogue

Multi-turn dialogue entails sustained interactions between a human and an MLLM-based agent. These range from casual interactions [39], to cooperative tasks with shared objectives [6] and structured question-answering scenarios [23, 40]. Our focus is on structured questionanswering in these dialogues. In language-only multiturn dialogues, a core challenge lies in managing question interdependence, where responses to earlier questions serve as contextual references in subsequent queries. To provide accurate responses, the model must interpret both the initial answer and the contextual references in follow-up questions. When a visual modality is introduced, the model faces added complexity: it must ❶ supplement language information with visual context, ❷ synchronize and integrate visual and linguistic data, and ❸ manage a reduction in visual focus over prolonged dialogues.

In dialogues where questions are independent, the interdependence challenge is absent, simplifying the interaction to single-turn question answering. Existing multiturn datasets [13, 22, 28] generally feature QA pairs with minimal interconnection. The MNIST Dialog [37] dataset incorporates spatial reasoning for correlated QA pairs, but tasks remain relatively simple. ChatterBox [43]

acknowledges the referential challenge but undermines coherence with rule-based substitutions, simply substituting words occurring repeatedly with “it”, introducing ambiguities. Our approach addresses these limitations by generating correlated question-answer drafts through rulebased methods, then refining them using GPT-4o-mini [31]. This produces a more complex and realistic multimodal, multi-turn dialogue dataset.

### 3. MMDiag: A New Benchmark for MultiTurn Multimodal Dialogue

In the following section, we first motivate the choice of scenarios. Next, we show details on how to construct the QA pairs for our MMDiag dataset. We then explain the evaluation process in Section 3.3. Finally, we compare MMDiag with existing multimodal dialogue datasets in Section 3.4. MMDiag contains three scenarios: everyday, tabular, and Minigrid. Examples of QA pairs are given in Section A.2. Both MMDiag and its generation code will be publicly released.

#### 3.1. Chosen Scenarios

The three selected scenarios — Everyday, Tabular, and Minigrid — are chosen to evaluate distinct yet complementary challenges in multimodal reasoning. Everyday scenes test common-sense understanding and multiturn interactions, reflecting real-world AI applications. Tabular scenarios require structured data comprehension and numerical reasoning, which many MLLMs struggle with. And Minigrid focuses on spatial reasoning and planning, essential for navigation and decision-making. This diverse selection ensures a comprehensive assessment of multimodal understanding. Empirically, all three settings pose significant challenges even for state-of-the-art models like GPT-4o (Figure 3), with notable failures, such as Visual CoT’s inability to generate positive grounding predictions in Tabular tasks (Table 2).

#### 3.2. Dataset Curation

Everyday Scene Subset. The raw source dataset [19] for this subset contains 108K images, each with detailed annotations. This allows us to construct a directed graph G = (V,E) for each image, where V represents the objects and E denotes their relationships. Then, each QA pair for an image is created and represented as a subgraph of G, i.e., Gqa = (Vqa,Eqa), with nodes and edges that belong to either the question or the answer. Note that if a QA pair lacks shared nodes or edges with other subgraphs, we classify it as independent, as it does not contribute to the dialogue’s complexity and does not require information from other QAs for a response. The created QA pairs are then extended into multi-turn QAs. We begin by constructing a subgraph pattern M = ni=1 Gqai , where

each Gqai represents a subgraph of a QA pair, ensuring ∀i,∃j ̸= i, s.t. Vqai ∩ Vqaj ̸= ∅. This design guarantees that answering any individual pair requires information from other QA pairs within the multi-turn dialogue.

We then apply subgraph matching to locate instances of M in the graph G for each image, enabling us to create diverse multi-turn QAs. We employ GPT-4o-mini [31] to generate various, natural questions, answers, and reasoning steps, while also providing ground truth location data for key objects. The specific prompt used in this process is detailed in Section A.1.

Tabular Scene Subset. This subset is sourced from ChartQA [29], which contains 18K real-world charts and 23.1K human-authored QA pairs. As ChartQA consists only of single-turn QA, it does not meet our multi-turn dialogue requirements. To generate multi-turn question answering, we use GPT-4o-mini, primarily relying on chart images due to the questionable reliability of tabletype metadata. To ensure interrelated dialogues, where certain regions are referenced as pronouns to increase complexity, we explicitly emphasize this requirement in the prompt. However, GPT-4o-mini struggles with maintaining this structure, requiring supplementary prompts to guide generation more effectively. Details on the prompt design are provided in Section A.1. Finally, we use EasyOCR [17] to match keywords with corresponding chart regions, enabling generation of bounding boxes for relevant areas.

Minigrid Scene Subset. Minigrid [10] is a Gymnasiumbased [45] collection of 2D grid-world environments with goal-oriented tasks. The agent, represented as a triangular figure with a discrete action space, navigates maze-like maps and interacts with objects such as doors, keys, and boxes. These tasks test the model’s ability to focus on image details, spatial reasoning, and action planning, with some requiring numerous steps to complete, making them particularly challenging. To construct this subset, we use Minigrid and BabyAI [9] to generate grid worlds, tasks, and step-by-step action plans, which are formatted as prompts for GPT-4o-mini. Minigrid creates environments based on specific constraints, saving grid world data as both rendered images and lists of special objects with bounding boxes. BabyAI then identifies feasible solutions by analyzing the agent’s field of view and determining subgoal-aligned actions. To simplify QA generation, we make the entire grid world visible, allowing MLLMs to guide the agent from a top-down perspective. GPT-4o-mini then generates natural questions, reasoning steps, key region queries, and concise final answers. Further details on environment generation and prompt design are in Section A.1.

Common Visual-Text Subset. To enable MLLMs with robust capabilities to answer the question, we also add additional visual-text pairs with high quality from previous works [25] to enhance their instruction-following ability.

#### 3.3. Multi-Turn Multimodal Dialogue Evaluation

The answers in MMDiag consist of three main components: the reasoning process, the corresponding grounded key region, and the final answer. Accordingly, we evaluate these three components separately. For the reasoning process and final answer, both of which are expressed in natural language and may vary in phrasing, we pass the image, question, ground-truth answer, and generated answer to a powerful MLLM for scoring, adhering to widely used evaluation practice. To mitigate the potential bias of using the same model for both dataset generation and evaluation, as MMDiag is generated using GPT-4o-mini [31], we instead use Gemini-1.5-Pro [35] in evaluation. Following prior studies [7, 20, 41], we evaluate the MLLMs through “ad-hoc” reasoning and scoring across five categories on a 0-10 scale, for greater consistency and interpretability. The complete evaluation prompt is provided in Section A.3. Additionally, we use the key region queries and their bounding boxes to constitute a grounding (GND) subset for evaluation. Since key region queries often involve detailed descriptions of objects or areas, including attributes and relationships, this GND subset can effectively measure grounding capability for complex queries. In this context, we use Intersection over Union (IoU) to evaluate the accuracy of grounding.

#### 3.4. Multimodal Dialogue Datasets Comparison

We compare MMDiag with prior datasets designed for vision-language understanding and reasoning. As shown in Table 1, MMDiag is the first to feature multi-turn, multiregion dialogues with strong QA dependencies, reinforced by a thorough generation process. In contrast, datasets like CB-300k [43] and MMDU [28] lack mechanisms to enforce such dependencies, reducing multi-turn dialogues to mere concatenations of independent QA pairs. Although MMDiag has relatively short dialogues, the inherent dependence between turns presents significant challenges for MLLMs, including GPT-4o, as demonstrated in Figure 3. The grounding and QA test splits include 1,000 unseen images and QA pairs, respectively.

### 4. DiagNote

In this section, we introduce our proposed DiagNote and its training process. Using two essential modules named Deliberate and Gaze, DiagNote is trained on the train split of MMDiag to meet the requirements for multiturn multimodal dialogue, which provides capabilities of stepwise reasoning and grounding corresponding salient visual regions for each dialogue.

Dataset QA Scale GND Scale Generation Process Average Turns Multi-Turn Multi-Region Dialogue Correlation CB-300k [43] 463k 254k GPT-4/Rule-based 5.49 ✓ ✗ ⃝ Visual CoT [38] 438k 438k GPT-4/OCR 1 ✗ ✗ ✗ CoM [32] 76k - GPT-4/Tree-Search/Human 1 ✗ ⃝ ✗ MMDU [28] 410k - LLM-filtered/GPT-4o 9 ✓ ✗ ✗

MMDiag 639k 1139k Graph-search/OCR/GPT-4o-mini 2.19 ✓ ✓ ✓

- Table 1. Comparison between MMDiag and other multimodal dialogue datasets. ⃝: Features are considered, but implemented weakly.

[Figure 3]

[Figure 4]

Gaze Module

Gaze Module

Gaze Module

Gaze Module

Step One Deliberate

Step One Deliberate

Step One D Q B

Bounding Box

Deliberate Step One

Query BBox

Query BBox

|lib Step|erat Two|
|---|---|
|Ga Qu|ze ery|
| | |

dishes [44,56,116,148]

Step Fin-2 D Q B Step Fin-1 D E Step Fin

Deliberate Step

Image

Image

Deliberate Step One

###### De e S

Deliberate Step Fin-1

Deliberate Step Fin

Dialogue Context

Dialogue Context

Step Two Deliberate

Gaze Query

Step Fin-1 Deliberate END

The 'them' refers to the dishes from the previous question. We should first locate dishes.

Gaze Query

Final Query

Could you describe them in detail?

What is on the shelf?

Query BBox

Bounding Box

END

There are dishes on the shelf.

Final_Ans Deliberate Step

They are white

Deliberate Module

Deliberate Module

Deliberate Module

Deliberate Module

in color, made of plastic,

Buffer

Buffer

Buffer

D B

Could you describe them in detail?

and stacked together.

- Figure 2. Model architecture of DiagNote. Regions with blue backgrounds represent a deliberation step and the interaction between the Deliberate and Gaze modules. At each turn, the Deliberate module processes the original image, dialogue context, and buffers from both modules. It produces two outputs: (1) a Deliberate step, stored in the Deliberate buffer, and (2) a Gaze query, which is processed by the Gaze module. The resulting bounding boxes are then stored in the Gaze buffer.

- 4.1. Model Architecture The overall framework of our model is illustrated in

- Figure 2. We use the same architecture (LLaVA-1.5 [24, 25]) for both Deliberate and Gaze modules, where the two modules do not share parameters. Considering the generalization capabilities of MLLMs, we choose not to use a dedicated grounding model like Grounding DINO [26] for the Gaze module. Specifically, the Deliberate module consists of an LLM as backbone, a pre-trained ViT [34] as vision encoder, and one MLP with a projection matrix to serve as the visual-text connector. The same structure applies for the Gaze module, with distinct parameters.

Given an input image Iv, we consider the entire dialogue contains T turns of questions and answers, which can

be represented as I1q,I1a,··· ,ITq ,ITa , where Itq and Ita respectively denote the question and the answer in the t-th dialogue turn.

At each turn t, given question Itq, DiagNote undergoes multiple interactive rounds between the Deliberate and Gaze modules for reasoning and to generate a reliable response Ita. To be specific, for the first interactive round, the Deliberate module takes as input the dialogue context Ct = I1q,I1a,··· ,Itq−1,Ita−1,Itq and image Iv and outputs a Deliberate step St1 and a Gaze query Qt1. St1 is then stored in the Deliberate buffer Btd. The Gaze module takes Gaze query Qt1 as input and outputs the corresponding bounding box ot1, which is stored in the Gaze buffer Btg. In each subsequent interactive round i of Deliberate and Gaze, the Deliberate module takes as input the image Iv, the dialogue context Ct, the Gaze buffer Btg = ot1,··· ,oti−1 , and

the Deliberate buffer Btd = St1,··· ,Sti−1 to generate Deliberate step Sti and Gaze query Qti. The Gaze module

, again, takes Gaze query Qti as input and outputs the annotation bounding box oti. This process continues until the Deliberate module outputs ‘END’ as the Gaze query

QkFin−1, indicating that the Deliberate and Gaze back-andforth process is complete.

Finally, the image, the dialogue context, and all the buffers are fed into the Deliberate module to produce the final answer StFin (i.e., Ita) and the Gaze query QtFin. The Gaze module then provides the bounding box of the salient area otFin for the t-th dialogue turn. The final output is StFin, along with the optional key region bounding box otFin, as well as the Deliberate process St1,··· ,StFin−1 , if required. The final answer Ita is then appended to the dialogue context for the next dialogue turn.

#### 4.2. Model Training

The training process of both Deliberate and Gaze modules follows that of LLaVA, and DiagNote provides two prompt templates pd and pg for Deliberate and Gaze respectively. At the i-th round of Deliberate and Gaze for Question Itq, the instruction Rindi for the Deliberate module is:

 

pd(Iv,Ct), i = 1 pd Iv,Ct,Btg,Btd , 1 < i < Fin pd Iv,Ct,Btg,Btd,Fin , i = Fin,

Rindi =

(1)



where Btd = St1,··· ,Sti−1 and Btg = Qt1,··· ,Qti−1 . The instruction Ringi for the Gaze module is:

Ringi = pg Iv,Qti , i ≤ Fin,i ̸= Fin − 1. (2)

MMDiag GND Testset GND Dataset

Average Everyday Tabular Minigrid MSCOCO RefCOCO

Model Train Data

Grounding DINO [26] - 0.384 0.001 0.209 0.715 0.469 0.356 LLaVA [25] LCS558K+Mixed665K 0.237 0.006 0.142 0.365 0.414 0.233 Visual CoT [38] VisCoT 0.220 0.003 0.160 0.321 0.362 0.213

DiagNote COCO 0.307 0.008 0.199 0.662 0.765 0.388 DiagNote MMDiag 0.369 0.466 1.0 0.259 0.257 0.471 DiagNote MMDiag + COCO 0.399 0.487 0.988 0.624 0.742 0.648 DiagNote MMDiag + COCO + VisCoT 0.433 0.281 0.910 0.662 0.837 0.625

- Table 2. Comparison results with existing MLLMs on Grounding benchmarks (GND) to demonstrate the challenging characteristics of our dataset MMDiag. We use Intersection over Union (IoU) as the evaluation metric.

We fine-tune the LLM on the prediction tokens, utilizing the auto-regressive training objective to optimize. We compute the probability of the target output Routxi with length L at i-th round by:

L

pθx rl | Rinxi ,Routx,<l ,

p(Routxi | Rinxi ) =

(3)

l=1

where x ∈ {d,g}.

θx is the trainable parameters of the Deliberate and Gaze

modules respectively, with x ∈ {d,g}. Rinxi are the input tokens of i-th round of the Deliberate and Gaze interaction

process. Routx,<l are the answer tokens before the current prediction token rl.

Our Deliberate and Gaze modules take LLaVA-1.5 [24] as base model. For the Gaze module, since grounding such salient areas as words and objects with detailed descriptions is quite challenging, we can first fine-tune it with an additional grounding dataset, and then fine-tune Deliberate and Gaze modules together. We combine the fine-tuning dataset from LLaVA [25] and the grounding datasets of MSCOCO [23] and RefCOCO [18, 43] with the augmentation grounding split of MMDiag to generate the grounding dataset; and we also combine the finetuning dataset from LLaVA with the training split of the MMDiag dataset to generate the entire training dataset. For data points in LLaVA, DiagNote does not add Deliberate prompts for the Deliberate module, thus instructing the Deliberate module to maintain the ability to output answers in general format.

### 5. Experiments 5.1. Implementation Details

We use LLaVA-1.5-7B [24] as the foundation model for both Deliberate and Gaze modules, with CLIP-ViT-LargePatch14-336 [34] as vision tower. Training is conducted on 8 × A800 GPUs with a learning rate of 2e-5. Deliberate and Gaze are optimized separately via supervised learning with

ground-truth outputs per round. During inference, the Gaze module signals reasoning completion by outputting “END” for turn Tx (Table 4), with the round number dynamically determined by DiagNote. Additional training details are provided in the Sections B and C.

#### 5.2. Results on MMDiag

##### 5.2.1. Visual Grounding

In this section, we focus primarily on how the MMDiag dataset benefits the grounding performance of MLLMs. Grounding is a crucial capability for MLLMs, enabling them to focus on relevant salient regions and reveal the visible reasoning process in dialogues, rather than functioning as a black box. We evaluate our DiagNote on several general grounding (GND) benchmarks [18, 23, 43], as well as our MMDiag GND benchmark. We use the average Intersection over Union (IoU) scores as the metric to assess GND performance, with results summarized in Table 2. When comparing DiagNote’s performance on established GND benchmarks like MSCOCO to its performance on MMDiag, we observe a significant decline on MMDiag, highlighting its increased difficulty relative to existing benchmarks. Existing models, such as Visual CoT, incorporate regions of interest for multimodal dialogue learning, but perform unsatisfactorily on GND datasets. For instance, Visual CoT scores -0.394 compared to Grounding DINO on the MSCOCO benchmark and performs worse than LLaVA. These results indicate a lack of robustness in explicitly grounding relevant areas in images. In contrast, DiagNote, trained with limited standard GND annotations provided by MMDiag and MSCOCO, shows significant improvements in both MSCOCO and RefCOCO benchmarks and performs better across the three subset scenarios of MMDiag. Notably, the MSCOCO data here is used solely to enhance grounding capability, and we intentionally limit the scale of GND data to prevent dataset size from influencing our conclusion. As shown in Row 4, DiagNote performs the poorest when trained exclusively on the MSCOCO dataset, underscoring the necessity and

MMDiag

Model Gaze Train Data

Average Everyday Tabular Minigrid

reasoning answer reasoning answer reasoning answer

LLaVA [25] ✗ LCS558K+Mixed665K 2.55 4.85 1.00 1.28 2.29 0.42 2.21 CogCoM [32] ✗ - 3.05 5.45 0.50 1.25 0.53 0.96 2.20 Visual CoT [38] ✗ VisCoT 4.15 4.90 1.23 1.95 1.09 2.50 2.81

DiagNote ✗ MMDiag 4.25 4.95 3.61 4.20 4.95 4.27 4.32 DiagNote ✓ MMDiag 5.82 6.15 3.95 4.05 5.10 4.15 4.92 DiagNote ✓ MMDiag+COCO 6.35 5.97 3.95 4.30 5.75 4.93 5.18 DiagNote ✓ GT 6.85 5.80 6.32 7.76 7.37 9.15 7.00

- Table 3. Comparison of the evaluation score with baselines to validate the Gaze module, we use Gemini-1.5-Pro to evaluate the performance of the reasoning process and the final answer. The evaluation process is detailed in Section 3.3.

Model

Tabular Reasoning Answer

T1 T2 T3 T4 T1 T2 T3 T4

CogCoM 0.55 0.91 1.15 0.67 1.75 0.73 0.85 0.35 Visual CoT 1.50 1.05 1.33 1.02 1.86 1.24 1.03 0.88 LLaVA 2.34 0.35 1.00 0.58 1.42 0.50 0.97 0.50

w/o Gaze 4.01 3.05 2.15 1.66 3.47 2.03 1.65 1.63 with Gaze 3.86 3.34 2.31 2.53 3.25 2.65 2.17 1.98

- Table 4. The Gemimi-1.5-Pro evaluation of the reasoning process and the final answer, scaling to 0-10, at turns 1 to 4 under the tabular scenario, where T∗ denotes the ∗-th turn in the dialogue.

benefits of our MMDiag dataset.

- 5.2.2. Multi-Turn Reasoning

on it easily and avoid such cases that it fails to locate the right target when the reasoning process moves on.

To further evaluate model performance, we compare our DiagNote with CogCoM [32] and Visual CoT [38], both of which can focus on specific regions and manage multimodal dialogues. Results show that DiagNote has significant advantages, especially in the tabular and Minigrid scenarios, reflecting the complexity of the MMDiag dataset and the strengths of DiagNote’s architecture, featured with the Deliberate and Gaze modules. To deepen the analysis, we show a comparison of results in tabular scenes under different numbers of dialogue turns in Table 4. DiagNote consistently outperforms the other models in the second, third, and fourth rounds under the tabular scenario, underscoring its superior capability in handling longcontext scenes with contextual and pronoun references. Meanwhile, the Gaze module shows more significant improvement, especially for increasingly long dialogues (e.g. T3 or T4), which further validates its effectiveness and benefits in long-context multimodal understanding. It is important to note that MMDiag’s tabular scenes in Table 3 include QA pairs of varying lengths (2–4), while Table 4 focuses only on dialogues with exactly 4 QA pairs.

We also evaluate our model’s multi-turn reasoning capabilities using the MMDiag benchmark. Beyond evaluating the correctness of the final answers, the evaluator also assesses the coherence and logic of the reasoning process within the Deliberate module. Detailed results are presented in Table 3. “GT” denotes scenarios where the Deliberate module receives ground-truth inputs for the reasoning step, serving as an upper bound. Except for the GT results, Gaze queries are generated by DiagNote, preventing information leakage. As expected, the GT setting significantly outperforms other settings, highlighting considerable room for improvement. To validate the effectiveness of our proposed module, we observe that the Gaze module enhances performance in specific reasoning tasks. For instance, in the everyday scenario, models utilizing the Gaze module achieve notably higher accuracy than those without it, demonstrating its ability to enhance focus and accuracy in reasoning. When there are multiple things, of the same kind, with different locations and attributes, in the image, chances are that the model cannot tell which object is exactly the one mentioned in the question. If the specific target is annotated on the image, the model can regain focus

#### 5.3. Qualitative Results.

In this section, we provide additional examples of the visual grounding and reasoning capabilities of DiagNote. More visualization results can be found in Sections D and F.

Visual Grounding. The Gaze module offers both grounding and OCR capabilities across diverse scenarios. As illustrated in Figure 4b, Grounding DINO [26] struggles in complex scenes where multiple objects of the same category exist with different attributes or relationships, therefore often failing to locate the target object precisely. In contrast, DiagNote’s Gaze module effectively manages such situations, as shown in Figure 4a. Additionally, when faced with tasks requiring text recognition, the Gaze module exhibits more robust OCR capabilities, accurately

The provided image depicts a grid world, with gray squares representing impassable walls, black squares as the background color, and the agent represented as a red triangle, with its pointed tip indicating the initial direction it is facing. Various squares may contain different objects such as colored balls, keys, boxes, and doors.

Question': 'How can the agent go to the yellow box?

To help the agent (the red triangle) reach the yellow box, you can follow these steps:

CoT:

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

|current box, which is| |
|---|---|
|should move toward the one more<br><br>then forward<br><br>|4<br><br>5<br><br><br>|

- 1.To start, we need to identify the agent's position.

- 2. Next, we need to locate the yellow b the target.

- 3.To reach the yellow box, the agent sh right.
- 4.Then, the agent should move forward yellow box.
- 5.Finally, the agent needs to move forward time to reach the yellow box.

- 1.Identify the Agent's Position: The agent starts in the bottom left corner of the grid.
- 2.Determine the Position of the Yellow Box: The yellow box is located in the middle column, second row from the top.
- 3.Plan the Path:

2

- ▪ Move up to the square directly above the agent.
- ▪ Move up again to the next row.
- ▪ Move right to the middle column.
- ▪ Finally, move up to reach the yellow box.

6

- 1

- 2

5

- 3

- 4

Final_Ans: The agent should move right, twice to reach the yellow box.

1

This path avoids impassable walls and allows the agent to reach the yellow box efficiently.

3

- Figure 3. Comparison for an example of the Minigrid scenario, one of the subsets in MMDiag. We give DiagNote (green) and GPT-4o (orange) the same environmental description and question. DiagNote focuses on the key regions and gives the correct reasoning process and the final answer. In contrast, GPT-4o fails to locate the object and thus gives the wrong answer. Examples for the MMDiag subsets of everyday scenarios and tabular scenes can be found in Section F.

[Figure 9]

(a) DiagNote (b) Grounding DINO

[Figure 10]

- Figure 4. A grounding comparison between Grounding DINO and DiagNote’s Gaze module , with the Gaze query “pink and white sign”. In (a), the red bounding box represents the ground-truth answer, while the blue one indicates the output generated by the Gaze module in DiagNote. In (b), the red bounding boxes show the outputs produced by Grounding DINO.

identifying and localizing specific keywords.

Multi-Turn Reasoning. With the incorporation of the Gaze module, our model can also more effectively focus on fine-grained details distributed across the image, offering a clear advantage in tasks that demand cohesive reasoning across both visual and linguistic information. As shown in Figure 3, a comparison between our DiagNote and GPT-4o within a simple Minigrid environment highlights this benefit. Despite detailed descriptions provided in the prompt, GPT-4o struggles with completing a shortrange, single-subgoal task, underscoring the strengths of our dataset and methodology.

- 5.4. Ablation Study

inputs. Failure cases show that when dialogues reference tiny key regions (under 0.2% of the image), Gaze often produces inaccurate bounding boxes, confusing the Deliberate module. The CLIP-ViT-Large-Patch14-336 encoder further limits resolution, contributing to errors. On standard multimodal benchmarks, DiagNote performs comparably or slightly lower, as it targets complex multiregion dialogues without in-domain training data. Ablation details are in Section E.

### 6. Conclusion

In this paper, we focus on a key challenging task scenario for MLLMs—multi-turn multimodal dialogue. To address it, we first introduce a specially designed dataset, MMDiag, where accomplishing tasks requires properly integrating visual information across different regions of an image and connecting multimodal information across various QA pairs. This setting closely resembles natural conversations and poses significant challenges to current MLLMs. To solve this, we construct the MMDiag dataset across three distinct scenarios—everyday, tabular, and Minigrid—using a combination of rule-based methods and GPT-4o-mini to ensure robustness and diversity. Experimental results highlight the challenges posed by MMDiag. Therefore, we propose DiagNote, an MLLM inspired by human visual processing, composed of two primary modules: Gaze and Deliberate. The Deliberate module performs CoT reasoning step by step, with the assistance of the Gaze module, which provides annotations of salient regions to focus on. Experiments show that this design enhances both grounding and reasoning capabilities, effectively addressing MMDiag challenges. We hope our work contributes to advancing the development of more intelligent MLLMs.

We observe a counterintuitive performance trend when comparing DiagNote with and without the Gaze module. To analyze its impact, we fine-tune DiagNote and Visual CoT on MMDiag and confirm Gaze’s effectiveness. However, its gains are limited, likely due to low-resolution image

### Limitations

Although MMDiag contains diverse data, our methods can be expected to generate even more scenarios and complex questions, resulting in even more challenging datasets for multi-turn multimodal dialogue. While qualitative results and case studies demonstrate the effectiveness of our approach, there remains considerable room for improvement. The potential performance drops with the introduction of Gaze module may stem from failures in queries involving extremely tiny objects. Fine-tuning Gaze to abstain from answering when uncertain or replacing the vision encoder backbone may enhance its robustness. Further exploration of training paradigms and model architecture could also potentially lead to enhanced performance.

### References

- [1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. 1
- [2] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen Technical Report. arXiv preprint arXiv:2309.16609, 2023. 1
- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond. arXiv preprint arXiv:2308.12966, 2023. 2
- [4] Tom B. Brown. Language Models are Few-Shot Learners. arXiv preprint arXiv:2005.14165, 2020. 3
- [5] Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12M: Pushing Web-Scale Image-Text Pre-Training To Recognize Long-Tail Visual Concepts. In CVPR, pages 3558–3568, 2021. 2
- [6] Guangyao Chen, Siwei Dong, Yu Shu, Ge Zhang, Jaward Sesay, B¨orje F. Karlsson, Jie Fu, and Yemin Shi. AutoAgents: A Framework for Automatic Agent Generation. In IJCAI, 2024. 3
- [7] Guiming Hardy Chen, Shunian Chen, Ziche Liu, Feng Jiang, and Benyou Wang. Humans or llms as the judge? a study on judgement biases. arXiv preprint arXiv:2402.10669, 2024. 4
- [8] Xi Chen, Josip Djolonga, Piotr Padlewski, Basil Mustafa, Soravit Changpinyo, Jialin Wu, Carlos Riquelme Ruiz, Sebastian Goodman, Xiao Wang, Yi Tay, et al. PaLI-X: On

- Scaling up a Multilingual Vision and Language Model. arXiv preprint arXiv:2305.18565, 2023. 2
- [9] Maxime Chevalier-Boisvert, Dzmitry Bahdanau, Salem Lahlou, Lucas Willems, Chitwan Saharia, Thien Huu Nguyen, and Yoshua Bengio. BabyAI: First Steps Towards Grounded Language Learning With a Human In the Loop. In ICLR, 2019. 4, 12
- [10] Maxime Chevalier-Boisvert, Bolun Dai, Mark Towers, Rodrigo de Lazcano, Lucas Willems, Salem Lahlou, Suman Pal, Pablo Samuel Castro, and Jordan Terry. Minigrid & Miniworld: Modular & Customizable Reinforcement Learning Environments for Goal-Oriented Tasks. CoRR, abs/2306.13831, 2023. 1, 4, 12
- [11] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality, 2023. 2
- [12] Cursor. The AI Code Editor. https://www.cursor. com/, 2024. 1
- [13] Abhishek Das, Satwik Kottur, Khushi Gupta, Avi Singh, Deshraj Yadav, Jose M. F. Moura, Devi Parikh, and Dhruv Batra. Visual Dialog. In CVPR, 2017. 3
- [14] DeepL. Better writing with DeepL Write. https://www. deepl.com/en/write, 2024. 1
- [15] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. PaLM-E: An Embodied Multimodal Language Model. arXiv preprint arXiv:2303.03378, 2023. 2
- [16] Yicheng Feng, Yuxuan Wang, Jiazheng Liu, Sipeng Zheng, and Zongqing Lu. LLaMA-Rider: Spurring Large Language Models to Explore the Open World. In NAACL, pages 4705– 4724, 2024. 2
- [17] JaidedAI. EasyOCR. https://github.com/ JaidedAI/EasyOCR, 2024. 4, 15
- [18] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. ReferItGame: Referring to Objects in Photographs of Natural Scenes. In EMNLP, pages 787–798,

2014. 6

- [19] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations. IJCV, 123:32–73, 2017. 3, 12
- [20] Noah Lee, Jiwoo Hong, and James Thorne. Evaluating the Consistency of LLM Evaluators. arXiv preprint arXiv:2412.00543, 2024. 4
- [21] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. In ICML, pages 19730–19742. PMLR, 2023. 2
- [22] Shengzhi Li and Nima Tajbakhsh. SciGraphQA: A LargeScale Synthetic Multi-Turn Question-Answering Dataset for Scientific Graphs. arXiv preprint arXiv:2308.03349, 2023. 2, 3

- [23] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft COCO: Common Objects in Context. In ECCV, pages 740–755. Springer, 2014. 3, 6
- [24] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved Baselines with Visual Instruction Tuning. In CVPR, pages 26296–26306, 2024. 5, 6, 19
- [25] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual Instruction Tuning. NeurIPS, 36, 2024. 1, 2, 4, 5, 6, 7, 13
- [26] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. arXiv preprint arXiv:2303.05499, 2023. 5, 6, 7, 15
- [27] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows. In ICCV, pages 10012–10022, 2021. 2
- [28] Ziyu Liu, Tao Chu, Yuhang Zang, Xilin Wei, Xiaoyi Dong, Pan Zhang, Zijian Liang, Yuanjun Xiong, Yu Qiao, Dahua Lin, et al. MMDU: A Multi-Turn Multi-Image Dialog Understanding Benchmark and Instruction-Tuning Dataset for LVLMs. arXiv preprint arXiv:2406.11833, 2024. 2, 3, 4, 5
- [29] Ahmed Masry, Do Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning. In ACL, 2022. 4, 12
- [30] Brandon McKinzie, Zhe Gan, Jean-Philippe Fauconnier, Sam Dodge, Bowen Zhang, Philipp Dufter, Dhruti Shah, Xianzhi Du, Futang Peng, Floris Weers, et al. MM1: Methods, Analysis & Insights from Multimodal LLM Pretraining. arXiv preprint arXiv:2403.09611, 2024. 2
- [31] OpenAI. GPT-4o mini: advancing cost-efficient intelligence. https://openai.com/index/gpt- 4o- miniadvancing- cost- efficient- intelligence/,

2024. 2, 3, 4, 12

- [32] Ji Qi, Ming Ding, Weihan Wang, Yushi Bai, Qingsong Lv, Wenyi Hong, Bin Xu, Lei Hou, Juanzi Li, Yuxiao Dong, et al. CogCoM: Train Large Vision-Language Models Diving into Details through Chain of Manipulations. arXiv preprint arXiv:2402.04236, 2024. 2, 3, 5, 7, 19
- [33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning Transferable Visual Models From Natural Language Supervision. In ICML, pages 8748–8763. PMLR, 2021. 2
- [34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning Transferable Visual Models From Natural Language Supervision. In ICML, pages 8748–8763. PMLR, 2021. 5, 6
- [35] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal

- understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 1, 4, 12
- [36] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. LAION-5B: An open large-scale dataset for training next generation image-text models. NeurIPS, 35: 25278–25294, 2022. 2
- [37] Paul Hongsuck Seo, Andreas Lehrmann, Bohyung Han, and Leonid Sigal. Visual Reference Resolution using Attention Memory for Visual Dialog. NeurIPS, 30, 2017. 3
- [38] Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chainof-Thought Reasoning. arXiv preprint arXiv:2403.16999,

2024. 2, 3, 5, 6, 7, 19

- [39] Kurt Shuster, Samuel Humeau, Antoine Bordes, and Jason Weston. Image Chat: Engaging Grounded Conversations. arXiv preprint arXiv:1811.00945, 2018. 3
- [40] Amanpreet Singh, Vivek Natarjan, Meet Shah, Yu Jiang, Xinlei Chen, Devi Parikh, and Marcus Rohrbach. Towards VQA Models That Can Read. In CVPR, pages 8317–8326,

2019. 2, 3

- [41] Rickard Stureborg, Dimitris Alikaniotis, and Yoshi Suhara. Large language models are inconsistent and biased evaluators. arXiv preprint arXiv:2405.01724, 2024. 4
- [42] Weihao Tan, Ziluo Ding, Wentao Zhang, Boyu Li, Bohan Zhou, Junpeng Yue, Haochong Xia, Jiechuan Jiang, Longtao Zheng, Xinrun Xu, et al. Towards General Computer Control: A Multimodal Agent for Red Dead Redemption II as a Case Study. In ICLR 2024 Workshop on Large Language Model (LLM) Agents, 2024. 1
- [43] Yunjie Tian, Tianren Ma, Lingxi Xie, Jihao Qiu, Xi Tang, Yuan Zhang, Jianbin Jiao, Qi Tian, and Qixiang Ye. ChatterBox: Multi-round Multimodal Referring and Grounding. arXiv preprint arXiv:2401.13307, 2024. 3, 4, 5, 6
- [44] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. LLaMA: Open and Efficient Foundation Language Models. arXiv preprint arXiv:2302.13971, 2023. 2
- [45] Mark Towers, Ariel Kwiatkowski, Jordan Terry, John U Balis, Gianluca De Cola, Tristan Deleu, Manuel Goul˜ao, Andreas Kallinteris, Markus Krimmel, Arjun KG, et al. Gymnasium: A Standard Interface for Reinforcement Learning Environments. arXiv preprint arXiv:2407.17032,

2024. 4

- [46] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention Is All You Need. NeurIPS, 30,

2017. 2

- [47] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS, 35:24824–24837, 2022. 3

- [48] Xinrun Xu, Yuxin Wang, Chaoyi Xu, Ziluo Ding, Jiechuan Jiang, Zhiming Ding, and B¨orje F. Karlsson. A Survey on Game Playing Agents and Large Models: Methods, Applications, and Challenges. arXiv preprint arXiv:2403.10249,

2024. 2

- [49] Shukang Yin, Chaoyou Fu, Sirui Zhao, Ke Li, Xing Sun, Tong Xu, and Enhong Chen. A Survey on Multimodal Large Language Models. arXiv preprint arXiv:2306.13549, 2023. 2
- [50] Sipeng Zheng, Jiazheng Liu, Yicheng Feng, and Zongqing Lu. Steve-Eye: Equipping LLM-based Embodied Agents with Visual Perception in Open Worlds. In ICLR, 2024. 1, 2, 13
- [51] Sipeng Zheng, Bohan Zhou, Yicheng Feng, Ye Wang, and Zongqing Lu. UniCode: Learning a Unified Codebook for Multimodal Large Language Models. In ECCV, pages 426–

443. Springer, 2025. 1

### A. Dataset

We use GPT-4o-mini [31] to generate our MMDiag dataset. Our dataset mainly consists of three parts: everyday scenes, tabular scenes, and Minigrid settings. We adopt different prompts for the generation of datasets under different scenes.

#### A.1. Dataset Collection

We design prompts for different scenarios, and the same devising ideas can be used in other scenarios for data collection.

Everyday Scenes. For everyday scenes, we generate our dataset from the Visual Genome dataset [19]. Since the original dataset has human-annotated attributes and relationship data, we extract the subsets that represent the QA pairs and feed them to GPT-4o-mini to generate corresponding dialogues. Figures 5 to 7 show several example prompts.

Please generate a new list based on a dictionary (`dict`) structured as follows: [Image_Dict]

The resulting list should be structured as follows: [Result_Dict]

### Explanation: There are two dictionaries in the generated list.

- - The first dictionary's question is based on the relation to the first object in the `answer`. The first two items in the `CoT` (Chain of Thought) list correspond to the first list in `gnd`, breaking the question down into two steps of grounding reasoning. The final `CoT` item provides a complete and concise answer to the question.
- - The second dictionary’s question refers to the attributes of the object from the first question's answer and is presented using a pronoun. The first `CoT` item deduces the referent, the second extracts the attribute information, and the last item provides a complete and concise answer to the question. The `Question` and `CoT` answers should be diverse and natural. The `Query` contains a concise, detailed description of the object in that step, and `Bbox` includes the object's coordinates from `obj_info`. Only output the dict in JSON format.

**IMPORTANT**: The order of objects in the CoT reasoning should follow the order of objects in the `gnd` list.

###### Human:{Current_Image_Dict}

Figure 5. The first example prompt for generating data samples in everyday scenes.

Tabular Scenes. For tabular scenes, we generate our dataset from the ChartQA dataset [29]. In general, we use different types of graphs to capture various visualization intuitions, providing corresponding chart examples in the prompts. Figure 8 illustrates the main structure of the prompt, while Figures 9 to 11 show examples for line, pie, and bar charts, respectively.

Minigrid Settings. For Minigrid settings, we generate our dataset from the Minigrid database [10]. Since we observe that GPT-4o-mini struggles to solve the mission without ground-truth planning, we first use BabyAI [9] to collect the plan needed to complete the mission for each environment generated by the Minigrid database. We then combine the positions of all objects with the mission and plan, as shown

Please generate a new `dict` based on the provided one. The provided `dict` is structured as follows: [Image_Dict]

The generated `dict` should look like this: [Result_Dict]

### Explanation:

- - The `Question` should be generated based on the `relation` predicates and the `attributes` of the last object in the `gnd`.
- - The `CoT` (Chain of Thought) list's first three entries MUST correspond to the `gnd` objects list, which break the problem into three steps of grounding reasoning. The `Query` MUST correspond to the `gnd` objects list.
- - The fourth item in the `CoT` list refers to the attributes of the target object.
- - The last `CoT` entry provides a concise final answer to the question.
- - The `Question` and `CoT.Ans` should be varied and natural. `Query` is a brief, specific description of the object, while `Bbox` corresponds to the object’s `coordinates` in `obj_info`. Only output the dict in JSON format.

**IMPORTANT**: The order of objects in the CoT reasoning should follow the order of objects in the `gnd` list.

###### Human:{Current_Image_Dict}

- Figure 6. The second example prompt for generating data samples in everyday scenes.

Please generate a new `dict` based on the given one. The provided `dict` is structured as follows: [Image_Dict]

The new `dict` should follow this structure: [Result_Dict]

### Explanation:

- - The first `dict` asks a question based on the first object in the `relation[0]` and uses the first object from the `answer`. The `CoT` list contains step-by-step reasoning, aligning with the first item in `gnd`, breaking the problem into two steps of grounding reasoning. The final item in the `CoT` list provides a simple and concise answer to the question.
- - The second `dict` asks about the attributes of the object answered in the first question, referring to it with a pronoun. The first `CoT` item infers the referred object, the second item extracts the attributes, and the final item provides a full, concise answer.
- - The third `dict` asks a question about the related object from `relation[1]`, again referring to it with a pronoun. The `CoT` steps involve reasoning to identify the referred object and then the related object, ending with a complete, concise answer.

**IMPORTANT**: The order of objects in the CoT reasoning must match the order of objects in the `gnd` list.

Human:{Current_Image_Dict}

- Figure 7. The third example prompt for generating data samples in everyday scenes.

in Figure 12, and feed them to GPT-4o-mini. The prompt structure is illustrated in Figure 13.

#### A.2. Dataset Format

Examples of the final MMDiag dataset are shown in Figures 14 to 16. Figures 14a, 15a and 16a display the original images from the source datasets and environments, while Figures 14b, 15b and 16b show the data format of MMDiag generated by GPT-4o-mini and standardized according to specific rules.

#### A.3. Evaluation

Since GPT-4o-mini contributes to generating our datasets, we use Gemini-1.5-Pro [35] for evaluation. There are multiple reasons for choosing it for this task: answer

Please generate a new list based on the provided chart and table data. The main reference should be the chart content, as the table content might contain errors. The format of the new list should be similar to the following example: [QA_and_CoT]

This list consists of two dictionaries corresponding to two rounds of Q&A. Each question is based on the chart, providing a reasoning process and an answer. The CoT (Chain of Thought) consists of multiple steps with "Ans" representing the answer broken down into steps, and "Query" indicating the key terms in the chart relevant to that step. The final step of CoT provides a complete and concise answer to the question, and the "Query" highlights the key terms in the chart that are relevant to the question.

The Question and CoT answers should be diverse and natural.

**Important**: The second question should refer back to the answer from the first question, meaning that you can’t answer the second question unless you know the answer of the first question. The answer of the first question is presented using a pronoun in the second question, and shouldn’t appear in the second question. You only need to output the list in JSON format.

###### Human:{Current_QA_and_CoT}

- Figure 8. The prompt structure to generate samples in tabular scenes.

[

{

"Question": "In which year did the highest percentage of voters care about the election outcome, and what was the percentage?",

"CoT": [ {

"Ans": "To solve this, we should first find the highest point of the brown line, which is 83.",

"Query": "83"

}, {

"Ans": "Next, we can identify that this occurred in 2020.", "Query": "2020"

}, {

"Ans": "In 2020, 83% of voters cared the most about the election result.", "Query": "83"

} ]

}, {

"Question": "What percentage of voters didn't care about the election result four years before that year?",

"CoT": [ {

"Ans": "The referenced year is 2020 from the previous question, and four years earlier would be 2016.",

"Query": "2016"

}, {

"Ans": "The yellow line in 2016 indicates a value of 22.", "Query": "22"

}, {

"Ans": "In 2016, 22% of voters did not care about the election outcome.", "Query": "22"

} ]

} ]

- Figure 9. The question-answer (QA) and Chain-of-Thought (CoT) examples for line charts.

formatting and the Chain of Thought (CoT) processes may be diverse, making a simple similarity score insufficient for evaluation. Additionally, recent works [25, 50] commonly apply LLMs for judgment. We provide the MLLM with images, ground-truth answers, and generated responses, and ask it to score the accuracy of the generated answers across five categories. We notice that the MLLM provides more reasonable rankings when asked to explain the ‘ad-hoc’ reason before their final score. As a result, we include this reasoning step in the prompt, as shown in Figure 17.

[

{

"Question": "What did most Americans favor when it comes to spending on policing, and what was the percentage?",

"CoT": [ {

"Ans": "To solve this, we should first locate the largest part of the pie chart, which is 42%.",

"Query": "42"

}, {

"Ans": "Next, we can see that this part represents people who favored maintaining the same level of spending on policing.",

"Query": "Stay about the same"

}, {

"Ans": "The largest group, with 42%, favored maintaining current spending levels on policing.",

"Query": "42" }

]

}, {

"Question": "How does this group compare to those who favored reduced spending?",

"CoT": [ {

"Ans": "This group refers to the one mentioned in the previous answer, which represents 42%.",

"Query": "42"

}, {

"Ans": "Now, we need to compare it with those who favored reduced spending, indicated by the label 'Decreased'.",

"Query": "Decreased"

}, {

"Ans": "The portion of people who favored reduced spending is represented by the purple section of the pie chart, at 25%.",

"Query": "25"

}, {

"Ans": "The difference in percentage is 42 - 25 = 17.", "Query": ""

}, {

"Ans": "This group is 17 percentage points larger than those who favored reduced spending.",

"Query": "17" }

] }

]

Figure 10. The question-answer (QA) and Chain-of-Thought (CoT) examples for pie charts.

### B. DiagNote

Our DiagNote consists of two MLLMs, one for Deliberate, and one for Gaze. For each input question, DiagNote appends buffer information and queries to the respective prompts for Deliberate and Gaze. For images from Minigrid, a description of the Minigrid environment, as shown in Figure 20, is included in both training and testing. The remaining components of the Deliberate prompt and Gaze prompt are consistent across all three scenes.

Deliberate Prompt. For deliberating, DiagNote provides the dialogue context and Chain of Thought (CoT) history for the current question in the prompt, as shown in Figure 21. When the ‘END’ token appears in the latest ‘Query’ from the Deliberate module, signaling the end of the CoT process, DiagNote provides a new prompt, as shown in Figure 22, to the Deliberate module for generating the final answer.

Gaze Prompt. For gazing, DiagNote extracts the ‘Query’ from the output of the Deliberate module and provides it to

[

{

"Question": "Which region had the second smallest consumption of Ozone-Depleting Substances in tonnes in 1998?",

"CoT": [ {

"Ans": "To solve this, we first need to find the second smallest consumption in tonnes, which is 143 tonnes.",

"Query": "143 tonnes"

}, {

"Ans": "Next, we can determine that this bar refers to Malta.", "Query": "Malta"

}, {

"Ans": "In 1998, Malta had the second smallest consumption of Ozone-Depleting Substances, with 143 tonnes.",

"Query": "143 tonnes" }

]

}, {

"Question": "How many times greater was the highest consumption of Ozone-Depleting Substances compared to that region?",

"CoT": [ {

"Ans": "The region in question is Malta, with 143 tonnes.", "Query": "143 tonnes"

}, {

"Ans": "The highest consumption to compare it with is 2,262 tonnes.", "Query": "2,262 tonnes"

}, {

"Ans": "The ratio is calculated as 2,262 / 143 = 15.8.", "Query": ""

}, {

"Ans": "The region with the highest consumption used 15.8 times more Ozone-Depleting Substances than Malta.",

"Query": "" }

] }

]

Figure 11. The question-answer (QA) and Chain-of-Thought (CoT) examples for bar charts.

hyper-parameters value deepspeed zero3 base model LLaVA-1.5-7B conversation template Vicuna v1 vision tower CLIP-ViT-Large-

Patch14-336 modality projector type mlp2x gelu image aspect ratio pad training epochs 1 training batch size 16 learning rate 2e-5 weight decay 0 warm-up ratio 0.03 model max length 2048 data loader workers 4

Table 5. The implementation details of the Deliberate module.

the Gaze module along with the prompt shown in Figure 23. The output from the Gaze module, which includes the

{

"mission": "open the grey door, then open the green door", "object": {

"grey door": "[256, 320, 288, 352]", "red triangle agent": "[288, 288, 320, 320]", "green door": "[320, 288, 352, 320]"

}, "plan_list": [

[

"Actions.left", "(GoNextToSubgoal: grey door None, reason: Open)"

], [

"Actions.forward", "(GoNextToSubgoal: grey door None, reason: Open)"

], [

"Actions.left", "(GoNextToSubgoal: grey door None, reason: Open)"

], [

"Actions.toggle", "(OpenSubgoal)"

], [

"Actions.left", "(GoNextToSubgoal: green door None, reason: Open)"

], [

"Actions.forward", "(GoNextToSubgoal: green door None, reason: Open)"

], [

"Actions.toggle", "(OpenSubgoal)"

] ]

}

- Figure 12. The mission and plan input example of Minigrid settings.

Based on the provided image and the given mission and object information, generate a new dict. The provided image is a grid world, where gray squares represent impassable walls, black squares are the background color, and the agent is a red triangle, with the pointed tip indicating the initial direction the agent is facing. Different squares may contain various objects such as colored balls, keys, boxes, doors, etc. The mission provides the task that the agent needs to accomplish, the plan list provides the action and subgoal for each step, and the object provides the coordinates of these objects. The format of mission and object is as follows: [Mission_and_Plan]

The format of the new dict should be similar to the following example: [QA_and_CoT]

Each dict should consist of a Question, a CoT (Chain of Thought) process, and a Final_Ans. The Question is generated based on the mission. The CoT consists of multiple steps, where each step has "Ans" for the explanation ,"Query" for identifying the key elements in the image relevant to that step and "Bbox" for the coordinates of the object in "Query". The Final_Ans provides a clear and concise solution to the question, with the "Query" highlighting the key terms in the image corresponding to the solution.

Ensure the Question, CoT answers, and Final_Ans are diverse and natural. The Bbox should contains all the bounding boxes of the Query. Output the dict in JSON format only.

Human:{Current_QA_and_CoT}

- Figure 13. The prompt structure to generate data samples in Minigrid settings.

bounding box of the query, is then saved in the Deliberate buffer to support the next turn of Deliberating.

### C. Implementation

The detailed parameters of implementation are shown in Tables 5 and 6.

[Figure 11]

(a) the original image

{

"QA_pairs": [ {

"Question": "What's the woman holding?", "CoT": [

{

"Ans": "To address this question, we should first identify the woman.", "Query": "woman", "Bbox": [211, 46, 478, 255]

}, {

"Ans": "Next, we can observe that she is holding a cup.", "Query": "END"

}, {

"Final_Ans": "The woman is holding a cup.", "Query": "cup", "Bbox": [309, 118, 338, 154]

} ]

}, {

"Question": "Could you describe it in detail?", "CoT": [

{

"Ans": "The 'it' in the question refers to the cup from the previous question, so

we first need to locate the cup.", "Query": "cup", "Bbox": [309, 118, 338, 154]

}, {

"Ans": "We can see that the cup is made of paper.", "Query": "END"

}, {

"Final_Ans": "The cup is a paper cup.", "Query": "cup", "Bbox": [309, 118, 338, 154]

} ]

}

], "image": "2353699.jpg", "question_id": 16

}

(b) the sample format

Figure 14. One example of the original image and the generated sample from Visual Genome in JSON format.

### D. Qualitative Comparison of Grounding

Figures 18 and 19 show a comparison of grounding ability between DiagNote and Grounding DINO [26]. As illustrated in Figure 18b, Grounding DINO struggles with grounding tasks involving Optical Character Recognition (OCR). In contrast, DiagNote leverages the generalization capability of LLMs, enabling it to effectively locate the target words, as shown in Figure 18a. Figure 19b

[Figure 12]

(a) the original image

{

"QA_pairs": [ {

"Question": "What was the average advertisement cost during Super Bowl XXXVI

(2002)?", "CoT": [ {

"Ans": "To find the average advertisement cost for Super Bowl XXXVI, we look at its specific entry.",

"Query": "Super Bowl XXXVI (2002)"

}, {

"Ans": "The average cost listed is 2.3 million U.S. dollars.", "Query": "END"

}, {

"Final_Ans": "Thus, the average advertisement cost during Super Bowl XXXVI was 2.3 million U.S. dollars.",

"Query": "2.3 million U.S. dollars" }

]

}, {

"Question": "How much more did the average advertisement cost for Super Bowl LV (2021) compared to that event?",

"CoT": [ {

"Ans": "The average advertisement cost for Super Bowl LV is 5.6 million U.S. dollars.",

"Query": "5.6 million U.S. dollars"

}, {

"Ans": "The cost difference can be calculated as 5.6 - 2.3.", "Query": ""

}, {

"Ans": "This results in a difference of 3.3 million U.S. dollars.", "Query": "END"

}, {

"Final_Ans": "Therefore, the average advertisement cost for Super Bowl LV was 3.3 million U.S. dollars more than that event.",

"Query": "" }

] }

], "image": "two_col_383.png"

}

(b) the sample format

Figure 15. One example of the original image and the generated data point from ChartQA in JSON format. The bounding boxes of the queries are generated using EasyOCR [17] and thus are not shown in the example.

illustrates that Grounding DINO fails to handle objects with attributes. Although the grey key has a marginally higher confidence, accurately locating the ‘grey’ key in the

[Figure 13]

(a) the original image

{

"id": "BabyAI-OpenDoorsOrderN4-v0_185", "QA_pairs": {

"Question": "How can the agent open the green door first, and then open the grey door?",

"CoT": [ {

"Ans": "To solve this, we first need to locate the agent's position.", "Query": "red triangle agent", "Bbox": [288, 224, 320, 256]

}, {

"Ans": "Next, we need to find the green door.", "Query": "green door", "Bbox": [256, 160, 288, 192]

}, {

"Ans": "To open the green door, the agent should move forward, then turn right,

move forward again, and finally toggle to open the door.", "Query": "green door", "Bbox": [256, 160, 288, 192]

}, {

"Ans": "Now, we need to locate the grey door.", "Query": "grey door", "Bbox": [288, 320, 320, 352]

}, {

"Ans": "To go to the grey door, the agent should turn right, move forward, turn right again, and move forward several times to reach the grey door, then toggle to open it.",

"Query": "END"

}, {

"Final_Ans": "The agent first needs to move forward, turn right, move forward again to open the green door. Then, it should turn right, move forward, turn right again, move forward several times, and finally open the grey door.",

"Query": "grey door", "Bbox": [288, 320, 320, 352]

} ]

}, "image": "BabyAI_frame_0_with_action_full_obs_with_attr/BabyAI-OpenDoor-

sOrderN4-v0/185.jpg" }

(b) the sample format

Figure 16. One example of the original image and the generated sample from Minigrid in JSON format.

You are an evaluator. Your task is to assess the given answer based on its accuracy in response to the provided picture, related question, and the ground truth answer. Your evaluation should be based on ad-hoc reasoning. First, provide a detailed reasoning for your judgment, then explicitly state the final category in the format: Reason: ... Judgment: ... Use the following five categories for your judgment: Incorrect: The answer is entirely wrong or unrelated. Partially Correct: The answer contains some relevant elements but is mostly incorrect. Medium: The answer captures partial correctness but lacks significant details or has notable inaccuracies. Almost Correct: The answer is mostly accurate but has minor errors or omissions. Correct: The answer is fully accurate and aligns well with the ground truth. [Please give a detailed Chain-of-Thought process.]

Question: {Question} Ground Truth Answer: {GroundTruthAnswer} Given Answer: {GivenAnswer}

Figure 17. The evaluation prompt structure given to Gemini1.5-Pro. The content in ‘[]’ is added when the CoT process is evaluated.

hyper-parameters value deepspeed zero3 base model LLaVA-1.5-7B conversation template Vicuna v1 vision tower CLIP-ViT-Large-

Patch14-336 modality projector type mlp2x gelu layer selected for -2 fine-tuning vision tower image aspect ratio pad training epochs 1 training batch size 32 learning rate 2e-5 weight decay 0 warm-up ratio 0.03 model max length 2048 data loader workers 4 fine-tune vision tower True/False

Table 6. The implementation details of the Gaze module.

image confuses Grounding DINO. In contrast, DiagNote accurately identifies the grey key in Figure 19a, which aids the subsequent actions of the Deliberate module.

### E. Ablation Study

We observe a counterintuitive performance trend in Table 3 in the main paper: Gaze provides only limited performance gains and, in some cases, even reduces performance, particularly in tabular and Minigrid scenarios. As shown in Figure 24, Gaze incorrectly identifies the bounding box for a critical but tiny piece of information—the year 2019—misleading Deliberate to focus on the wrong color

[Figure 14]

[Figure 15]

(a) DiagNote (b) Grounding DINO

- Figure 18. The grounding comparison between Grounding DINO and the Gaze module of DiagNote in Tabular Scene. The grounding query is “Cyprus”. The red bounding box in (a) is the ground-truth answer, while the blue one is the bounding box generated by our Gaze module. The red bounding box in (b) is the output of Grounding DINO.

[Figure 16]

(a) DiagNote (b) Grounding DINO

[Figure 17]

- Figure 19. The grounding comparison between Grounding DINO and the Gaze module of DiagNote in Minigrid Scene. The grounding query is “grey key”. The blue bounding box in (a) is generated by the Gaze module of DiagNote, which overlaps the ground-truth red bounding box. Meanwhile, the red bounding box in (b) is the output of Grounding DINO.

The provided image depicts a grid world, with gray squares representing impassable walls, black squares as the background color, and the agent represented as a red triangle, with its pointed tip indicating the initial direction it is facing.

Various squares may contain different objects such as colored balls, keys, boxes, and doors.

###### Figure 20. The description of Minigrid Scene added to the prompts.

This is the context, which includes the previous questions and answers related to the image. {Context}

Current problem and its reasoning process (CoT) includes reasoning, focused objects, and their bounding boxes (bbox): {Question_and_CoT}

Please provide the detailed reasoning process and focused objects for the question in dictionary form, outputting one dictionary in the following format each time: {{'Ans': , 'Query': }}. Here, Ans represents the current reasoning step, and Query indicates the object of focus. If the reasoning is complete, set 'Query' to END.

###### Figure 21. The prompt structure of the Deliberate module when the last Query output of the Deliberate module is not ‘END’.

This is the context, which includes the previous questions and answers related to the image. {Context}

Current problem and its complete reasoning process (CoT) includes reasoning, focused objects, and their bounding boxes (bbox): {Question_and_CoT}

Please provide a concise answer to the question in dictionary form, outputting a dictionary in the following format: {{'Final_Ans': , 'Query': }}, where Final_Ans is the concise answer to the question, and Query is the core object of focus in the image related to the question.

###### Figure 22. The prompt structure of the Deliberate module when the last Query output of the Deliberate module is ‘END’.

Please refer to the image and provide the exact rectangular coordinates of the {Query} in the format of a four-dimensional integer array [x1, y1, x2, y2], where x1 and y1 represent the coordinates of the top-left corner of the rectangle, and x2 and y2 represent the coordinates of the bottom-right corner.

Figure 23. The prompt structure of the Gaze module.

bar. This issue accounts for most failure cases.

To further analyze this, we evaluate the proportion of tiny key regions across different scenarios in MMDiag (Table 9). In tabular and Minigrid scenes, nearly all key regions occupy less than 3% of the total image area, making them particularly challenging for Gaze to detect accurately. To mitigate this, we curate an alternative test dataset for tabular scenes, excluding questions that require attention to extremely small regions. We then fine-tune Visual CoT and DiagNote with MMDiag and evaluate them on this revised tabular split. As shown in Table 7, Gaze’s impact becomes more pronounced. Table 8 demonstrates that DiagNote performs comparably or slightly lower on standard multimodal benchmarks, as it targets complex multi-region dialogues without in-domain training data.

Which country had the highest rate per million population in 2019?

Questions

To find the highest rate in 2019, we look at the values for that year.

Query: 2019 values

Deliberate One

Bbox: [337, 969, 373, 985]

[Figure 18]

Bounding Box

Figure 24. The second example of comparison between different MLLMs under everyday scenes.

Model Fine-tuning Data Gaze T1 T2 T3 T4

Visual CoT-13B MMDiag - 2.00 1.43 0.40 0.95 DiagNote-14B MMDiag ✗ 3.15 2.35 1.78 1.23 DiagNote-14B MMDiag ✓ 4.20 3.10 2.55 1.95

Table 7. Tabular scenes results of MLLMs fine-tuned on MMDiag, using the same evaluation metrics as the previous evaluation.

Benchmark MMBench MM-Vet RefCOCO+ RefCOCOg DiagNote-14B 63.7 28.5 0.834 0.775

- Table 8. DiagNote performance on general datasets.

Scenario ≤ 0.2% ≤ 1% ≤ 3% ≤ 5% ≤ 10% Everyday 7.57% 27.62% 47.99% 57.49% 69.91% Tabular 87.17% 99.24% 99.80% 99.92% 100% Minigrid 6.98% 66.61% 96.99% 99.41% 100%

- Table 9. MMDiag tiny key regions percentage.

### F. Qualitative Comparison of Multi-Turn Multimodal Dialogue

We present several cases comparing models in everyday scenarios and tabular scenes. Figures 25 and 26 show examples from unseen everyday scenarios. In Figure 25,

[Figure 19]

Input Image

How many orange plates are there in front of the girl? What are placed above them?

Questions

There are two orange plates in front of the girl. There are two cups placed above the orange plates.

LLaVA-1.5-13B

There are two orange plates in front of the girl. There are two bowls placed above the orange plates.

Visual CoT

1

CogCoM

The image captures a lively scene in a restaurant. A family of four is seated at a table, their attention directed towards the young boy who sits between them. He's engaged with his food, holding a fork and spoon in his hands as he enjoys his meal. \n\nThe table is adorned with plates filled with various dishes, each one adding to the vibrant color scheme of the room. There are glasses on the table, perhaps used for drinking or serving. \n\nIn the background, you can see other diners, suggesting that this is not just a family outing but also an event where others are present. The walls are painted in a warm yellow hue, providing a welcoming atmosphere to all those who enter.\n\nOverall, the image paints a picture of a happy moment shared by a family in a cozy setting.

There are two orange plates in front of the girl. The two white plates are placed above the orange plates.

DiagNote

Figure 25. The first example of comparison between different MLLMs under everyday scenes.

CogCoM [32] completely fails to answer the two-turn questions correctly. Despite the assistance of the counting expert, CogCoM is unable to answer the first counting question. Although LLaVA-1.5-13B [24] and Visual CoT [38] can answer the first questions accurately, both encounter hallucinations when responding to the second question, mistakenly identifying white plates as cups and bowls, respectively. In contrast, our DiagNote performs well on both questions, demonstrating the effectiveness of the Gaze module in ensuring DiagNote stays grounded in visual details. In Figure 26, CogCoM fails to provide a clear answer to the first question, instead offering a confusing single word ‘jean’. Again, LLaVA-1.5-13B and Visual CoT answer the first question correctly, but imagine the man was holding a frisbee. Both CogCoM and DiagNote understand the context, with DiagNote accurately describing the can based on the visual details. In contrast, CogCoM mistakenly assumes it is a can of beer, which may not be the case.

[Figure 20]

Input Image

What type are the trousers worn by the man with the headband? What is he holding in hands? The man with the headband is wearing jeans. The man with the headband is holding a beer in one hand and a frisbee in the other hand.

Questions

LLaVA-1.5-13B

The man with the headband is wearing jeans. The man with the headband is holding a beer in one hand and a frisbee in the other hand.

Visual CoT

jean CogCoM

The man is holding a can of beer.

The trousers worn by the man with the headband are jeans. He is holding a green can in his hands.

DiagNote

- Figure 26. The second example of comparison between different MLLMs under everyday scenes.

Questions

What percentage of LGBT Americans consider religion to be very important in their lives?

In comparison, how much does the percentage change of those who consider religion to be less important?

According to the survey, do more LGBT Americans tend to be religiously affiliated or not?

Input Image

[Figure 21]

According to the Pew Research Center, 17% of LGBT Americans consider religion to be very important in their lives.

In comparison, 34% of LGBT Americans consider religion to be less important in their lives.

The survey shows that a majority of LGBT Americans are religiously affiliated, with 34% being religiously affiliated and 44% being unaffiliated.

LLaVA-1.5-13B

Visual CoT

17% 1% Not

CogCoM The percentage of those who consider religion to be less important has increased from 34% to 50%.

Based on the data provided by Pew Research Center, 17% of LGBT Americans consider religion to be very important in their lives.

The survey shows that more LGBT Americans tend to be religiously affiliated than not.

DiagNote

Therefore, the percentage of those who consider religion less important is 17 percentage points higher than those who consider it very important.

Thus, 17% of LGBT Americans consider religion to be very important in their lives.

Thus, more LGBT Americans tend to be religiously affiliated than not.

- Figure 27. One example of comparison between different MLLMs under tabular scenes.

Figure 27 presents examples of unseen tabular scenes. All models answer the first question correctly. However, Visual CoT provides a completely incorrect answer to the second question, while CogCoM introduces an unfounded ‘50%’. LLaVA-1.5-13B correctly identifies the visual detail ‘34%’, but overlooks the keyword ‘change’ in the question, which requires a calculation between two percentages. Only DiagNote answers the question precisely. The final

question requires the models to understand the entire pie chart. The model should compare the sum of two parts on the right side of the pie chart with the left part to obtain the final answer ‘yes’. Visual CoT fails to provide this correct answer, and LLaVA-1.5-13B misinterprets the unaffiliated percentage and derives an incorrect affiliated percentage. Both CogCoM and DiagNote reach the right conclusion. Overall, DiagNote performs well on all questions, demonstrating its ability to focus on both visual and language details and to comprehend the full picture the chart conveys. This strong ability can be attributed to the Gaze and Deliberate structure, which enables it to zoom in on specific details while integrating multimodal information for a holistic understanding.

