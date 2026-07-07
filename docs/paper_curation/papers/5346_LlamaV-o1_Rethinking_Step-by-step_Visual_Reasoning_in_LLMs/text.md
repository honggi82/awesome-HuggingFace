# arXiv:2501.06186v1[cs.CV]10Jan2025

[Figure 1]

## LlamaV-o1: Rethinking Step-by-step Visual Reasoning in LLMs

Omkar Thawakar1∗ Dinura Dissanayake1* Ketan More1* Ritesh Thawkar1* Ahmed Heakl1* Noor Ahsan1* Yuhao Li1* Mohammed Zumri1* Jean Lahoud1* Rao Muhammad Anwer1 Hisham Cholakkal1 Ivan Laptev1 Mubarak Shah2 Fahad Shahbaz Khan1,3 Salman Khan1,4

1Mohamed bin Zayed University of AI, 2University of Central Florida,

3Linköping University, 4Australian National University

[Figure 2]

LlamaV-o1 Project: https://mbzuai-oryx.github.io/LlamaV-o1/

LlamaV-o1 Model: https://huggingface.co/omkarthawakar/LlamaV-o1

LlamaV-o1 Code: https://github.com/mbzuai-oryx/LlamaV-o1

VRC-Bench https://huggingface.co/datasets/omkarthawakar/VRC-Bench

### Abstract

Reasoning is a fundamental capability for solving complex multi-step problems, particularly in visual contexts where sequential step-wise understanding is essential. Existing approaches lack a comprehensive framework for evaluating visual reasoning and do not emphasize step-wise problem-solving. To this end, we propose a comprehensive framework for advancing step-by-step visual reasoning in large language models (LMMs) through three key contributions. First, we introduce a visual reasoning chain benchmark specifically designed to evaluate multi-step reasoning tasks. The benchmark presents a diverse set of challenges with eight different categories ranging from complex visual perception to scientific reasoning with over 4k reasoning steps in total, enabling robust evaluation of LLMs’ abilities to perform accurate and interpretable visual reasoning across multiple steps. Second, we propose a novel metric that assesses visual reasoning quality at the granularity of individual steps, emphasizing both correctness and logical coherence. The proposed metric offers deeper insights into reasoning performance compared to traditional end-task accuracy metrics. Third, we present a new multimodal visual reasoning model, named LlamaV-o1, trained using a multi-step curriculum learning approach, where tasks are progressively organized to facilitate incremental skill acquisition and problem-solving. The proposed LlamaV-o1 is designed for multi-step reasoning and learns step-by-step through a structured training paradigm. Extensive experiments show that our LlamaV-o1 outperforms existing open-source models and performs favourably against close-source proprietary models. Compared to the recent Llava-CoT, our LlamaV-o1 achieves an average score of 67.3 with an absolute gain of 3.8% across six benchmarks while being 5× faster during inference scaling. Our benchmark, model, and code are publicly available.

∗Equal Contribution

Technical Report of LlamaV-o1.

|Claude-3.5-sonnet|
|---|
|Reasoning Steps: Let me solve this step by step.<br><br>Step 1: Analyze the first row of images. Action 1: In the first box, there are three symbols:<br>Step 2: Analyze the second row (incomplete pattern). Action 2: There are two symbols followed by a question mark:<br>Step 3: Identify the pattern from the first row to the second row. Action 3: Looking at how the first row transitions to the second row:<br>Step 4: Evaluate each option against the pattern Action 4: Looking at options A through D:<br>Step 5: Compare pattern requirements with options. Action 5: The answer should: Maintain a star or similar geometric theme. Final Answer: None of the given options A, B, C, or D follows the established pattern of transitioning from complex sun-like shapes to simpler star-based designs.<br>|

|Gemini-1.5-Flash|
|---|
|Reasoning Steps:<br><br>Step 1: Analyze the first row. Action 1: The first row shows a progression of shapes. It starts with a sun-like shape, then an octagonlike shape.<br>Step 2: Analyze the second row. Action 2: The second row starts with a smaller version of the sun-like shape from the first row. Next to it is a bigger version of that same sun-like shape. This suggests a size progression. The last shape is a star, taking the concept of points than the previous two.<br>Step 3: Determine the next shape in the pattern. Action 3: Following the logic of the first row, after a pointed shape (the star) should come a larger version of it, maintaining the pattern from the second row.<br>Step 4: Examine the options. Action 4: None of the provided options (A, B, C, or D) depict a star. Final Answer: None of the provided choices matched.<br>|

|LlamaV-o1 (Ours)|
|---|
|Reasoning Steps: Step 1: Analyze the pattern in the first group of images. Action 1: The first row contains images that have a circular shape with additional features inside, such as a sun pattern, a geometric star shape, and a smiley face.', 'Step 2: Determine what feature is consistent across all the images in the first group. Action 2: Each image maintains a central circular shape with symmetrical, repeated internal designs or symbols.', 'Step 3: Determine the rule across the shapes<br><br>Action 3: The second sequence is using the shapes of the first sequence and adding one shape. ', 'Step 4: Compare the options in the second group to the established pattern.<br>Action 4: The last shape in the first sequence is two smiley faces, and Option D is showing three smiley faces.’ Final Answer: D) Picture D follows the same pattern or rule established by the previous pictures.<br>|

###### Question

[Figure 3]

Which one picture follows the same pattern or rule established by the previous pictures? choices :

- A) Picture A ,
- B) Picture B ,
- C) Picture C ,
- D) Picture D

- Figure 1: Comparison of the reasoning abilities of our model (LlamaV-o1) with closed-source Gemini-1.5-Flash and Claude-3.5-Sonnet on an example in pattern recognition task from our proposed VRC-Bench. While Claude-3.5-Sonnet concludes "none of the options," its reasoning steps lack full alignment with the observed logic (highlighted in red). Gemini-1.5-Flash demonstrates weaker reasoning with less logical coherence details (highlighted in red). Our LlamaV-o1 provides better and more systematic reasoning, identifying that option D follows the established pattern, thereby showcasing its logical reasoning capability. Additional results are presented in Fig. 5.

### 1 Introduction

Large Language Models (LLMs) are designed to understand and generate text, enabling them to handle a wide range of tasks such as translation [6, 74, 12, 8, 55, 42], summarization [20, 75, 60], and question-answering [55, 33, 56]. The integration of visual data through Large Multimodal Models (LMMs) extends capabilities further by combining text, images, or videos, allowing for more complex multimodal tasks like image captioning, visual question answering, and video analysis. To effectively solve these tasks, visual reasoning is essential for LMMs to process and connect diverse information, ensuring logical coherence and sequential problem-solving. The ability to reason across multiple modalities is crucial to addressing complex real-world problems.

To improve the problem-solving ability of LLMs, step-by-step reasoning is desired to break down complex tasks into easier components. This approach resembles human cognitive processes, enabling models to track their thought processes and ensure logical consistency throughout their reasoning. By following a structured reasoning path, models can reach more accurate and interpretable conclusions. To this end, previous works have demonstrated that prompting or fine-tuning LLMs to generate stepby-step rationales can lead to improvements in reasoning tasks [32, 61, 59]. These methods encourage models to explicitly reason through each step, focusing on improving their ability to tackle complex tasks. However, most existing works struggle to handle step-by-step multimodal reasoning tasks (see Fig. 1). Further, a notable gap in current visual reasoning benchmarks is their lack of emphasis on step-by-step reasoning. Most benchmarks focus primarily on end-task accuracy, neglecting the quality of intermediate reasoning steps. Moreover, the absence of standardized evaluation can likely lead to inaccurate comparisons between models, making it difficult to assess their true visual reasoning capabilities.

Our work strives to bridge the aforementioned gaps by introducing a holistic approach to evaluating step-by-step visual reasoning capabilities. To this end, we introduce Visual Reasoning-Chain (VRCBench), a comprehensive a benchmark specifically designed to assess multi-step visual reasoning tasks. The benchmark spans 8 diverse categories: Visual Reasoning, Math & Logic Reasoning, Social & Cultural Context, Medical Imaging (Basic Medical Science), Charts & Diagram Understanding, OCR & Document Understanding, Complex Visual Perception, and Scientific Reasoning. It includes over 1,000 challenging samples, meticulously curated to evaluate reasoning capabilities across various domains. Furthermore, the benchmark features 4,173 manually verified reasoning steps, ensuring accuracy and reliability in assessing step-by-step logical reasoning. Recognizing that measuring end-task accuracy is insufficient, we present a new metric that evaluates visual reasoning quality at the granularity of individual steps, focusing on both correctness and logical coherence. Further, we explore the combined advantages of integrating Beam Search with Multi-Step curriculum learning in the training of visual reasoning models. By leveraging the efficiency of Beam Search alongside the progressive structure of curriculum learning, the proposed model incrementally acquires skills,

starting with simpler tasks such as summary of the approach and question derived captioning and advancing to more complex multi-step reasoning scenarios, ensuring both optimized inference and robust reasoning capabilities. We observe this structured training paradigm not only to enhance the model’s performance but also to improve interpretability and adaptability in handling different visual reasoning tasks (see Fig. 1). Our extensive experiments demonstrate that our approach, named LlamaV-o1, outperforms existing open-source methods, including the recent Llava-CoT [66] model, across multiple evaluation metrics.

In summary, this paper presents a unified framework that aims to advance step-by-step visual reasoning capabilities through a new benchmark, a novel metric, and a new model trained with curriculum learning. Our contributions are as follows:

- • Step-by-Step Visual Reasoning Benchmark: To the best of our knowledge, the proposed benchmark is the first effort designed to evaluate multimodal multi-step reasoning tasks across diverse topics. The proposed benchmark, named VRC-Bench, spans around eight different categories (Visual Reasoning, Math & Logic Reasoning, Social & Cultural Context, Medical Imaging (Basic Medical Science), Charts & Diagram Understanding, OCR & Document Understanding, Complex Visual Perception and Scientific Reasoning) with over 1,000 challenging samples and more than 4k reasoning steps.
- • Novel Evaluation Metric: A metric that assesses the reasoning quality at the level of individual steps, emphasizing both correctness and logical coherence.
- • Combined Multi-Step Curriculum Learning and Beam Search Approach: A multimodal reasoning method, named LlamaV-o1, that combines the structured progression of curriculum learning with the efficiency of Beam Search. The proposed approach ensures incremental skill development while optimizing reasoning paths, enabling the model to be effective in complex multi-step visual reasoning tasks in terms of both accuracy and efficiency. Specifically, the proposed LlamaV-o1 achieves an absolute gain of 3.8% in terms of average score across six benchmarks while being 5× faster, compared to the recent Llava-CoT [66].

### 2 Related Works

Reasoning with LLMs: The development of robust reasoning capabilities in Large Language Models (LLMs) has been a focal point of research. Early work often relied on neural-symbolic methods for explicit modeling of the reasoning process using formal language instead of natural language [53, 11, 3]. However, the emergence of powerful LLMs has prompted new approaches that leverage their inherent reasoning abilities [63]. For example, inference time computing is scaled in recent models to perform reasoning before giving the final answer [65, 62, 24, 49]. Techniques like Chain-of-Thought (CoT) prompting, where a complex question is decomposed into intermediate reasoning steps, have shown promise in guiding LLMs to structured solutions [61, 69]. Nevertheless, maintaining logical consistency, especially in tasks requiring multi-step inference, poses a significant challenge, leading to errors and hallucinated outputs [67, 43]. LLMs, even with CoT guidance, might generate unfaithful explanations, deviate from logical reasoning paths, and struggle with verifying and selecting correct reasoning steps [61]. These approaches are further extended to VLMs.

Reasoning with VLMs: Visual reasoning tasks require models to possess visual perception and high-level cognitive abilities [22, 31, 67]. The visual reasoning skills have broad applicability across domains such as science [40], mathematics [37], robotic planning [23] and advanced question answering [72]. Similar to the case of LLMs, the conventional approaches employed neural-symbolic methods to explicitly model the reasoning process [17, 58, 5]. For example, [4] propose differentiable logic formalism to decouple the reasoning aspect of VQA from visual perception. More recent VLMs leverage the reasoning capabilities of LLMs for visual tasks. Visual programming [22] provides a modular neuro-symbolic system based on computer vision models as functions and GPT-3 LLM for compositional visual reasoning. Zhang et al. [73] argue that VLM training with concise answers results in reduced generalization to more complex problems requiring reasoning. They use GPT-4o model to create rationales and use correct and incorrect reasoning chains in training to enhance model’s reasoning ability via reinforcement learning (RL) [51]. In contrast, LlaVA-o1 [67] does not use RL and advocates for stage-wise reasoning instead of CoT prompting, where the answer is worked out sequentially via summarization, interpretation, reasoning, and conclusion steps. Our work builds on [67] and shows the importance of curriculum learning and path search for visual reasoning.

|Visual reasoning|
|---|
|Question : Who painted this subject matter from 16th-17th Century? Choices:<br><br>A) Peter Paul Rubens 1675<br>B) Caravaggio 1635<br>C) Artemisia Gentileschi 1612-13<br>D) Francisco de Zurbarán 1614 Reasoning:<br><br><br>• Step 1: Identify the depiction in the image.<br>• Action 1: This painting depicts "Judith Slaying Holofernes," a scene often represented by artists like Caravaggio and Artemisia Gentileschi.<br>• Step 2: Match the painting style and date to an artist.<br>• Action 2: This specific depiction is known for its dramatic intensity and style that aligns with Artemisia Gentileschi's work from the Baroque period.<br>• Step 3: Verify the time period of each artist's work.<br>• Action 3: Artemisia Gentileschi painted "Judith Slaying Holofernes" around 1612-13, which fits the style in the image.<br><br><br>Final Answer: C) Artemisia Gentileschi<br><br>|[Figure 4]|
|---|
|

|Social and Cultural Contexts|
|---|
|Question : What sport is being played in the image?<br><br>Reasoning:<br><br>• Step 1: Identify the key elements of the sport illustrated.<br>• Action 1: The image shows people playing in the mud, and a goal post resembles soccer.<br>• Step 2: Identify the sport being played.<br>• Action 2: The sport being played is known as Swamp football.<br><br><br>Final Answer: The sport being played is known as Swamp football.<br><br>|[Figure 5]|
|---|
|

|Math and logic reasoning|
|---|
|Question : As shown in the figure, points A, O, and B are collinear, and DO is perpendicular to CO. If ∠BOC = 67°, what is ∠AOD? Reasoning:<br><br>• Step 1: Use the fact that ∠AOC and ∠BOC form a linear pair.<br>• Action 1: ∠AOC + ∠COB = 180°<br>• Step 2: Calculate ∠COA using ∠BOC = 67°<br>• Action 2: ∠AOC = 180° - 67° = 113°<br>• Step 3: ∠COD = 90°Action 3: Use this to find ∠ AOD<br>• Step 4: Determine ∠AOD given ∠AOC = 113 °,∠ COD<br><br>= 90 °<br><br>• Action 4: Since ∠AOD = ∠AOC - ∠DOC, calculate ∠AOD = 113 °- 90 ° = 23 °<br><br><br>Final Answer: 23 °<br><br>|[Figure 6]|
|---|
|

|Medical Imaging (Basic Medical Science)|
|---|
|Question : What kind of tissue does this<br><br>image depict? Choices:<br><br>A) Cardiac muscle<br>B) Skeletal muscle<br>C) C) Cartilage Reasoning:<br><br><br>• Step 1: Identify the characteristics of tissue in image.<br>• Action 1: Observe the striations, elongated fibers, and any presence of intercalated discs or nuclei alignment.<br>• Step 2: Compare these characteristics to the tissue types.<br>• Action 2: Cardiac muscle tissue typically has striations and intercalated discs. Skeletal muscle tissue has striations and multiple peripheral nuclei. Cartilage is not striated.<br>• Step 3: Match the observed characteristics with given options.<br>• Action 3: The image shows striations and elongated fibers typical of muscle tissue, specifically cardiac muscle, due to the presence of intercalated discs.<br><br><br>Final Answer: A) Cardiac muscle<br><br>|[Figure 7]|
|---|
|

|Chart and Diagram Understanding|
|---|
|Question : What percentage of global non-renewable energy reserves did conventional natural gas account for in 2018?<br><br>Reasoning:<br><br>• Step 1: Identify the section related to conventional natural gas in the pie chart.<br>• Action 1: Locate the segment labeled "Conventional natural gas”.<br>• Step 2: Determine the percentage contribution of conventional natural gas<br>• Action 2: Read the percentage labeled alongside the "Conventional natural gas" segment.<br><br><br>Final Answer: Conventional natural gas accounted for 17.8% of global non-renewable energy reserves in 2018.<br><br>|[Figure 8]|
|---|
<br><br>|

|OCR and Document Understanding|
|---|
|Question : What is the name of the choco fills advertised? Reasoning:<br><br>• Step 1: Locate the advertisement for choco fills.<br>• Action 1: Examine the image to find the section where choco fills are advertised. Look for any visual or textual cues that indicate a choco fills product.<br>• Step 2: Identify the brand name in the advertisement.<br>• Action 2: Once the choco fills advertisement is located, read the text to find the brand name associated with the product. Look for prominent text that typically indicates the product name.<br>• Step 3: Confirm the product name.<br>• Action 3: Verify that the identified name is indeed the product name by checking for any additional context or branding elements that support this identification.<br><br><br>Final Answer: Dark fantasy<br><br>|[Figure 9]|
|---|
|

|Complex Visual Perception|
|---|
|Question : Which figure is a reflection of the object? Reasoning:<br><br>• Step 1: Analyze the original object. Action 1: Observe the orientation and arrangement of the blocks in the original object. Notice which parts are protruding.<br>• Step 2: Determine the axis of reflection. Action 2: Visualize a vertical reflection over the left or right side of the image to understand how the shape transforms.<br>• Step 3: Compare each option with the reflected image. Action 3: Compare each of the options A, B, C, and D to see which one matches the reflection of the original shape.<br>• Step 4: Check the alignment and orientation of the blocks. Action 4: Focus on specific parts of the structure, such as the top and bottom blocks and how they align after reflection.<br><br><br>Final Answer: B)<br><br>|[Figure 10]|
|---|
|

|Scientific reasoning|
|---|
|Question : Complete the statement.<br><br>Ethane is __. Choices: (A) an elementary substance (B) a compound Reasoning:<br><br>• Step 1: Identify the structure depicted in the image. Action 1: Count the different elements in molecular.<br>• Step 2: Determine if the substance is a compound or an elementary substance. Action 2: Recognize that ethane (C₂H₆) consists of two different types of atoms: carbon (C) and hydrogen (H).<br>• Step 3: Classify substance based on its atomic composition. Action 3: According to chemistry definitions, a compound is a substance made from two or more different elements that are chemically bonded together.<br><br><br>Final Answer: B) a Compound<br><br>|[Figure 11]|
|---|
|

- Figure 2: The proposed VRC-Bench examples show the diverse and challenging reasoning tasks our benchmark encompasses, spanning a wide range of modalities and contexts. Each example emphasizes step-by-step reasoning, starting from task comprehension and progressing to logical inference and answer generation. The tasks include mathematical reasoning using geometric principles, scientific classification based on molecular structures, visual interpretation of charts and diagrams, artistic identification from historical paintings, and medical diagnosis from tissue images. For instance, one example demonstrates the calculation of an angle in a geometric diagram by leveraging linear pairs and perpendicular relationships. Another highlights scientific reasoning by identifying ethane as a compound based on its molecular composition. Visual perception tasks challenge the model to analyze pie charts for global energy reserves or recognize reflected shapes. Artistic and cultural tasks require identifying paintings and sports based on visual and contextual cues. Finally, tasks in medical imaging and advertisement recognition test the model’s ability to classify tissue types or extract product names through careful observation.

Benchmarks for Visual Reasoning: Several datasets and benchmarks have been developed to evaluate and advance visual reasoning in VLMs. These datasets vary in complexity, visual context, and reasoning skills required. Some notable examples are as follows. CLEVR (Compositional Language and Elementary Visual Reasoning) tests visual reasoning abilities like counting, comparisons, and logical inference through rendered images and automatically generated questions [25]. StrategyQA is a multi-hop question-answering dataset on Wikipedia that necessitates implicit decompositions and diverse reasoning strategies [18]. ScienceQA offers a large-scale multimodal science dataset with multi-modal contexts, diverse science topics, and annotated answers with corresponding lectures and explanations [40]. A consolidated mathematical reasoning benchmark in diverse visual contexts called MathVista incorporates 28 existing multimodal datasets and three new datasets [37]. Zhang et al. [73] propose ShareGPT-4o-Reasoning, a comprehensive CoT dataset containing 193k examples covering various VQA tasks, designed to improve CoT reasoning in VLMs. However, these benchmarks do not provide step-by-step reasoning in complex evaluation scenarios and generally judge the correctness based on only the final answer. In this work, our goal is to provide a comprehensive benchmark that assesses the reasoning chains as well as the final outcome in complex reasoning scenarios.

- 3 Step-by-Step Visual Reasoning Benchmark: VRC-Bench

To facilitate a thorough assessment of the reasoning capabilities in complex scenarios, we introduce a step-by-step visual reasoning benchmark. This benchmark serves as a structured tool to assess both the logical progression of reasoning chains and the accuracy of the final outcomes generated by LMMs.

Comparison of LMM’s on VRC-Bench

MMMU, BLINK-Art

MMMU-Medical(29)

(23)

G G G C G G Ll M Ll Ll

GPT-4o Gemini-2.0-Flash

ista(231)

GPT-4o-mini Claude-3.5-Sonnet

Reasoningisual

MathV

MedicalImaging

V

Doc-VQA

(61)

Math& Logic

OCR& Document

DynaMath(171)

Gemini-1.5-Pro Gemini-1.5-Flash

Reasoning

[Figure 12]

VComplex isual

Charts& Diagrams

Llama-3.2-Vision

LogicV

Perception

ista(158)

ScientificReasoning

ContextCulturalSocial &

Mulberry LlamaV-o1 Llava-CoT

(107)

ChartVQA

Blink-IQ(35)

ALM-Bench (104)

(83)

50 60 70

40 50 60

ScienceQA

Final Answer Accuracy

Steps Score

- Figure 3: The figure illustrates our comprehensive benchmark structure and comparative performance of LMMs on the proposed ReasoningChain-Bench. (Left) The dataset spans multiple domains, including carefully selected samples for mathematical and logical reasoning (e.g., MathVista [38] with 231 samples and LogicVista with 158 samples), scientific reasoning (e.g., Science-QA [40] with 83 samples), and visual perception (e.g., Blink-IQ-Test [15] with 35 samples). Additionally, it includes specialized areas such as medical imaging (e.g., MMMU-Medical [72] with 29 samples), cultural and social understanding (e.g., ALM-Bench [57] with 104 samples), and document understanding through OCR (e.g., Doc-VQA [46] with 61 samples). By integrating tasks like chart and diagram comprehension (e.g., Chart-VQA [44] with 107 samples), our dataset not only covers a broad spectrum of real-world applications but also expand LMM’s ability to reason, perceive, and interpret complex multimodal information. (Right) The bar chart compares various SoTA reasoning models on the VRC-Bench, highlighting both final answer accuracy and step-by-step reasoning scores. The models evaluated for complex reasoning tasks include GPT-4o, Gemini-2.0-Flash, Claude-3.5-Sonnet, and Llava-CoT. Our benchmark evaluates models not only on their ability to generate accurate final answers but also on the coherence and logical flow of their reasoning steps. Our approach, LlamaV-o1, outperforms GPT-4o-mini, Gemini-1.5-Flash and Llava-CoT in the VRC-Bench, achieving superior results in final answer accuracy across complex multimodal reasoning tasks.

By integrating diverse datasets that include a diverse range of topics, such as science, mathematics, medical knowledge, social sciences, and data interpretation, we ensure that our evaluation benchmark captures the diverse aspects of reasoning.

#### 3.1 Benchmark Creation

Benchmark Domains: To ensure a comprehensive assessment of reasoning capabilities, our step-bystep visual reasoning benchmark incorporates samples from several specific datasets across various domains. Figure 2 shows examples of the questions and answers included in our benchmark. The data distribution is shown in Figure 3. By integrating these diverse sources, we capture a wide range of reasoning scenarios, allowing for an extensive evaluation of the models’ abilities to respond to complex inquiries. Based on these diverse data samples, we generate step-by-step reasoning steps using a semi-automated annotation pipeline with detailed rationales. Next, we outline the main domains covered in the benchmark and then explain the annotation process.

Mathematical and Logical Reasoning: This category includes datasets focused on mathematical and logical tasks. MathVista [38], provides a variety of mathematical problems, while DynaMath [76] offers dynamic mathematical challenges. Additionally, ChartQA [44] encompasses tasks related to chart and diagram comprehension, allowing evaluation of visual reasoning in logical contexts.

Scientific Reasoning: For scientific reasoning, we include samples from Science-QA [40] to test the model’s ability to answer questions based on scientific knowledge and reasoning. Furthermore, MMMU-Medical [72], focuses on medical imaging tasks, assessing the model’s capability in interpreting complex multimodal medical data.

Cultural and Social Understanding: To assess the model’s ability to recognize and interpret diverse cultural scenarios, we include samples from ALM-Bench [57], which is designed to assess understanding of the social and cultural context.

Other Visual Reasoning Scenarios: We further include samples from other visual reasoning datasets. LogicVista [64] and Blink-IQ [15] focus on complex visual perception, providing challenges that require the model to analyze and interpret intricate visual information. Doc-VQA [46] targets OCR and document understanding, evaluating the model’s ability to extract information from text-based documents. Lastly, MMMU [72] and BLINK [15] (Art Split) contribute to visual reasoning tasks.

Semi-Automatic Step-by-Step Reasoning Generation: We adopt a semi-automatic approach to generate step-by-step reasoning responses. We begin by using the GPT-4o model to create detailed reasoning steps and answers for the various questions in our dataset. This involves crafting specific prompts to guide the model in producing detailed logical reasoning. In this way, we efficiently generate various reasoning chains with a consistent format, where the step-by-step reasoning includes all the required steps and actions to reach the desired answer. Additional details are provided in

- Appendix (Section. A.1).

Manual Verification: Since automated responses are not always reliable, we perform manual verification to ensure that all reasoning steps are accurate and correct. In this stage, a team of verifiers meticulously reviewed the generated reasoning chains and final answers, making necessary adjustments to enhance clarity and correctness. Our benchmark consists of examples spanning over 8 diverse categories as shown in Fig. 3. We ask the verifiers to add missing reasoning steps when necessary, and we drop examples with less than three reasoning steps after the verification except some samples from MathVista as they can be addressed with 2 steps. Over 25% of the data was corrected during the manual verification resulting in more than 1,000 samples and carefully verified 4,173 reasoning steps. The manual verification stage is essential for establishing a trustworthy ground truth, which serves as the benchmark for evaluating LMMs performance in our evaluations.

- Table 1: An overview of comprehensive set of attributes considered in our evaluation to assess the quality of reasoning in LMMs. These attributes focus on critical aspects such as faithfulness, informativeness, and logical coherence of reasoning steps. Key measures include ensuring alignment of reasoning steps with the source (Faithfulness-Step and Token), completeness of information (Informativeness-Step), and identifying issues like hallucinations, redundancy, or missing steps. Additional metrics, such as Semantic Coverage and Reasoning Alignment, evaluate the logical and semantic integrity of the response. Together, these metrics provide a robust framework for evaluating the accuracy, completeness, and reliability of LLM-generated reasoning.

Metric Definition Faithfulness-Step Measures how well the reasoning steps in the LMM response

align with the source reasoning steps. Faithfulness-Token Extends Faithfulness-Step to token-level granularity, checking if the content within each step is accurate. Informativeness-Step Measures how well the reasoning steps extract all relevant information from the context.

Repetition-Token Identifies repeated or unnecessarily paraphrased reasoning steps. Hallucination Detects irrelevant or fabricated reasoning steps not aligned with

the source. Redundancy Identifies redundant reasoning steps that do not add value. Semantic Coverage-Step Measures how well the response covers the essential semantic

elements of the source reasoning steps. Reasoning Alignment Overall alignment between the hypothesis and reference reasoning chain. Commonsense Checks for missing commonsense reasoning are required to solve the problem. Missing Step Identifies if any necessary reasoning steps are missing.

#### 3.2 Evaluation Framework

While multiple previous methods have been proposed for evaluating reasoning chains [19, 50], these methods exhibit various limitations. These methods adopt a reference-free approach, as they do not depend on a set ground truth. While this allows for flexibility in evaluation, it can lead to significant problems. For example, even if the reasoning steps are logically sequenced, a minor error can lead to a major disruption in the reasoning chain but still result in a high score. This compromises the accuracy of the assessment, as it does not truly reflect the quality of the reasoning. In our research, we stress on the importance of having a ground truth for scoring. By comparing generated responses to a reliable reference, we aim to improve the accuracy of our evaluations. Using a ground truth reasoning chain allows us to better identify and address inaccuracies.

Evaluation Metric: To overcome the shortcomings of reference-free metrics, we use GPT-4o [2] to compare the predictions generated by the model against a ground truth. This method allows us to evaluate reasoning quality using specific metrics that focus on different aspects of alignment and accuracy. We base our metric on the reference-free ROSCOE suite of metrics [19] and propose a reference-based metric. We show the details of the measures used in our metric in Table 1. For instance, we use the Faithfulness-Step and Faithfulness-Token metrics to assess how well the reasoning aligns with the source. The Faithfulness-Step metric scores alignment on a scale from 1 to 10, providing clear feedback on the accuracy of each reasoning step. We also measure Informativeness-Step, which checks if all critical information is included. By incorporating attributes like Hallucination and Redundancy, we can spot irrelevant or repetitive reasoning that reduces clarity. The final scoring process averages all attribute scores to give a comprehensive evaluation of reasoning quality. Additional details such as system prompt used for scoring are provided in

- Appendix (Section. A.2).

### 4 Proposed Step-by-Step Visual Reasoning Model: LlamaV-o1

Our proposed approach introduces several key contributions to advance multimodal reasoning in LMMs. First, we leverage curriculum learning to train the model progressively, starting with foundational tasks like summarization of approach and question based caption generation before advancing to detailed, multi-step reasoning. This structured approach helps the model manage complexity, improve logical coherence, and generalize effectively to challenging scenarios. Second, we scale inference efficiently with a simple yet effective Beam Search technique, which generates multiple beams in parallel and selects the most optimal one, ensuring both efficiency and high-quality outputs. This method significantly reduces computational costs, achieving constant scaling compared to the linear scaling of traditional approaches in terms of model calls.

#### 4.1 Curriculum Learning for Large Multimodal Models

LMMs are powerful tools for understanding and generating content across different data types, such as text, images, and video. However, reasoning in such models, especially in complex multistep scenarios, presents unique challenges. Models often struggle to handle step-by-step reasoning because reasoning requires not only understanding the input but also maintaining consistency and logical clarity across multiple steps. This is where curriculum learning becomes an essential strategy. Curriculum learning, inspired by human education systems, involves training a model progressively, starting from simpler tasks and gradually introducing more complex ones. This approach has shown significant benefits in improving model performance across various tasks, particularly when the tasks require reasoning over multiple modalities. For instance, curriculum learning has been successfully applied in multimodal learning such as Visual Question Answering (VQA) [30] and captioning tasks [26]. These studies demonstrate that models trained with simpler examples first and later gradually increasing task difficulty can generalize better to more complex problems.

Curriculum learning is a powerful approach for enhancing reasoning capabilities in LMMs by adopting a progressive training strategy. Starting with complex task and gradually introducing more difficult complex challenges, it helps models to build foundational skills incrementally. In the case of multimodal models, this structured progression allows to manage complexity effectively, as they first learn to interpret basic relationships between modalities, such as connecting text with images, before tackling more intricate scenarios. By ensuring a strong basis for logical reasoning, curriculum learning improves the coherence of multi-step tasks, enabling models to maintain consistency and alignment

across steps. In addition, curriculum learning addresses challenges like catastrophic forgetting, which can occur when models are directly fine-tuned on complex tasks, leading to overfitting and poor generalization. By focusing initially on simpler tasks, models consolidate fundamental patterns before progressing to more advanced problems. This approach mirrors human learning, where basic skills are mastered before tackling complex concepts, ensuring better generalization and adaptability. Overall, curriculum learning establishes a robust framework for developing reasoning capabilities, making multimodal models more reliable and effective across a wide range of real-world applications.

#### 4.2 Multi-Step Chain-of-Thought for Improved Reasoning

Multi-step chain-of-thought reasoning is crucial for tackling complex tasks that require sequential decision-making and logical coherence. Unlike single-step reasoning, which often overlooks intermediate steps, multi-step reasoning allows models to break down problems into smaller, manageable parts, ensuring transparency and consistency throughout the process. This step-by-step approach mirrors how humans solve complex problems by systematically reasoning through each step. For instance, answering a multifaceted question about an image might involve identifying objects, understanding their relationships, and synthesizing this information to form a coherent answer. Embracing multi-step reasoning not only enhances the interpretability of multimodal models but also aligns them closer to human-like problem-solving, setting the stage for more robust and versatile AI systems.

#### 4.2.1 Multi-Step Reasoning Stages

Multi-step reasoning in chain-of-thought (CoT) frameworks enables models to break down complex tasks into incremental steps, mirroring the human approach to problem-solving. This structured process ensures that each reasoning step is explicit and logical, improving transparency and accuracy. Our multi-step reasoning comprises of following steps also depicted as an example in the box below:

- • Task Understanding: The model begins by understanding the question and the context.
- • Task Summarization: The next step involves generating a summary of the visual data to ensure the model has a holistic understanding. This stage prepares the model to focus on relevant action items to be taken to get the final answer.
- • Detailed Caption Generation: To narrow the scope further, the model generates a detailed caption, which identifies specific labels and their corresponding values in the chart. This step ensures that the model accurately interprets the visual elements.
- • Logical Reasoning: The model then formulates a logical reasoning process to locate and interpret the required data. This reasoning step breaks the task into sub-goals, ensuring a systematic approach.
- • Final Answer Generation: Finally, the model outputs the final answer based on the reasoning process and the extracted context.

Breaking down reasoning into these stages ensures that models process complex queries methodically, reducing errors and enhancing interpretability. multi-step training reinforces the ability to handle each stage effectively, enabling LMMs to perform better on tasks that require step-by-step logical reasoning. This approach not only improves accuracy but also provides a transparent pathway for verifying and refining the model’s outputs.

#### Multi-Step Training Example

[Figure 13]

Human: What was the value of the middle-income share in 1971? Answer the question using a single word or phrase. Please generate a summary of the picture. Assistant: I will examine the image to find the relevant data for the middle-income share in 1971 and present the answer in the specified format. Human: Please generate a detailed caption for the image. Assistant: The image displays a bar chart comparing the percentage of adults in the lower, middle, and upper-income tiers for the years 2015 and 1971. It shows that in 1971, the middle-income share was 61%. Human: Please generate a detailed reasoning to answer the question given the caption.

Assistant: To solve the problem, I will look for the specific percentage associated with the middleincome group in the year 1971 from the provided chart. The image clearly shows that in 1971, the middle-income tier accounted for 61% of adults.

Human: Please generate the final answer based on reasoning steps. Do not output anything else. Assistant: 0.61 Human: How many did the share of the middle-class change from 1971 to 2015? Answer the question using a single word or phrase. Please generate a summary of the picture.

Assistant: To solve this problem, I will analyze the image data to determine the change in the share of adults living in middle-income households from 1971 to 2015. I will compare the percentages provided for each year and calculate the difference.

Human: Please generate a detailed caption for the image. Assistant: The image is a bar graph from Pew Research Center showing the share of adults in lower, middle, and upper-income households for the years 1971 and 2015. For 1971, 61% were in the middle income, and for 2015, 50% were in the middle income. Human: Please generate a detailed reasoning to answer the question given the caption.

Assistant: To find the change in the share of middle-class adults from 1971 to 2015, subtract the 2015 percentage from the 1971 percentage. The middle-income share in 1971 was 61%, and in 2015 it was 50%. Thus, the change is calculated as 61% - 50% = 11%. The difference in share, expressed as a decimal, is 0.11.

Human: Please generate the final answer based on reasoning steps. Do not output anything else. Assistant: 0.11

#### 4.2.2 Data Preparation and Model Training

To implement our curriculum learning strategy effectively, we divide the model training process into two stages, each designed to incrementally enhance the model’s reasoning capabilities while ensuring a robust understanding of multimodal inputs. This structured approach allows the model to acquire foundational reasoning skills in the first stage and progressively refine its ability to provide detailed, step-by-step answers in the second stage.

- Stage 1: Training for Summarization and Caption Generation: In the first stage, the model is trained to generate two critical components: (1) a summary of the approach needed to answer the question and (2) a detailed caption describing the relevant aspects of the input data, such as visual

- elements in an image. The training data for this stage is derived from 18K samples from the Cap-QA split of PixMo dataset [13] and 57K samples from Geo170K dataset from G-LLaVa [16]. Each sample includes a question paired with input data (e.g., an image or chart). Cap-QA split of PixMo dataset contains examples having grounded captions based on input question whereas the Geo170K dataset contains question-answer pairs with their reasoning steps. This stage ensures that the model learns to contextualize the input and outline a high-level reasoning plan before diving into detailed steps. The focus in this stage is to help the model gather the structure of reasoning tasks, improving its ability to decouple the problem into simpler elements. By focusing on structured training, the model develops the ability to handle multi-step tasks while maintaining a clear and organized flow of thought.
- Stage 2: Training for Detailed Reasoning and Final Answer Generation: In the second stage, the model builds upon the foundation established in Stage 1. Here, the model is trained not only to generate the summary and caption but also to provide detailed reasoning based on these components. Finally, the model outputs the correct answer derived from the reasoning process. For this stage, we use the original Llava-CoT dataset [66], which contains 99K structured samples comprising of various domains such as General VQA and Science-Targeted VQA from multiple sources. The General VQA data sources includes ShareGPT4V [8], ChartQA [44], A-OKVQA [54], DocVQA [45], PISC [27], CLEVR [25] whereas, Science-Targeted VQA sources consists GeoQA+ [7], AI2D [28], ScienceQA [40] and CLEVR-Math [34] respectively. Each sample consists of a summary, caption, detailed reasoning, and the final answer. The training process in this stage involves multi-step interactions, where the model progressively learns to break down the approach into incremental reasoning steps. This incremental learning ensures that the model refines its logical flow and systematically integrates the information from summaries and captions into actionable reasoning steps.

In second stage, the multi-step training methodology is key to the model’s success. During Stage 1, the model learns to organize its thoughts and outline a strategy, effectively setting the stage for the detailed reasoning required in Stage 2. By the time the model reaches Stage 2, it is already equipped with the ability to outline a structured approach, making it easier to focus on breaking down complex tasks into step-by-step solutions. This approach improves the interpretability, accuracy, and robustness of the model, enabling it to excel in complex multimodal reasoning tasks. Our results demonstrate that by leveraging datasets such as PixMo and Llava-CoT in a curriculum learning framework, the model can effectively transition from high-level problem understanding to detailed, step-by-step reasoning, achieving state-of-the-art performance in multi-step reasoning benchmarks.

Model Training: We leverage the PixMo and LLaVA-CoT-100k datasets to train our model using a curriculum learning strategy combined with a Supervised Fine-Tuning (SFT) approach. For this work, we select Llama-3.2-11B-Vision-Instruct [47] as our base model due to its strong foundation in multimodal reasoning and instruction-following capabilities. The fine-tuning process involves full-parameter optimization, allowing the model to adapt effectively to the structured reasoning tasks provided by the PixMo and LLaVA-CoT-100k dataset. Training is performed on a high-performance computing node equipped with 8 NVIDIA A100 (80GB) GPUs, ensuring efficient handling of the large-scale dataset and the computational requirements of the model. During the initial stage of curriculum learning, the model is fine-tuned on the PixMo dataset to develop foundational reasoning skills, such as generating summaries and captions. Further training details, such as the number of epochs, learning rate, optimizer settings, and batch size, are outlined in Appendix for reproducibility of our work.

#### 4.2.3 Optimizing Inference Efficiency: Beam Search

Inference efficiency is a critical factor in deploying large multimodal models for real-world applications, particularly when handling complex reasoning tasks. To address this, we adopt a Beam Search strategy that significantly improves inference efficiency and reasoning quality compared to existing approaches like LLava-CoT [67]. Our method is designed to balance computational complexity with output quality, enabling faster and more reliable inference.

Simplified Output Design: Unlike LLava-CoT [67], our approach does not require a highly structured output format. This flexibility simplifies the reasoning process, allowing the model to focus on generating high-quality outputs without the overhead of rigid structural constraints. This design choice

- Table 2: Comparison of models based on Final Answer accuracy and Reasoning Steps performance on the proposed VRC-Bench. The best results in each case (closed-source and open-source) are in bold. Our LlamaV-o1 achieves superior performance compared to its open-source counterpart (Llava-CoT) while also being competitive against the closed-source models.

Close-Source Open-Source

###### Model GPT-4o Claude-3.5 Gemini-2.0 Gemini-1.5 Gemini-1.5 GPT-4o Llama-3.2 Mulberry Llava-CoT LlamaV-o1

[2] Sonnet [1] Flash Pro [52] Flash [52] mini [48] Vision [47] [68] [66] (Ours)

Final Answer 59.28 61.35 61.16 61.35 54.99 56.39 48.40 51.90 54.09 56.49 Steps 76.68 72.12 74.08 72.12 71.86 74.05 58.37 63.86 66.21 68.93

###### GPT-4o

###### Claude-3.5-Sonnet

###### Gemini-1.5-Pro

###### gemini-2.0-flash-reasoning

|73.71| | | | |
|---|---|---|---|---|
| | | | | |

|67.64| | | | |
|---|---|---|---|---|
| | | | | |

|70.33| | | | |
|---|---|---|---|---|
| | | | | |

|75.03| | | |
|---|---|---|---|
| | | | |

59.28

61.35

61.35

61.16

Overall

76.68

72.12

72.12

74.08

85.25

78.69

89.66

78.69

OCR & Document Understanding

86.70

83.19

83.19

85.48

47.83

60.87

60.87

43.48

Visual Reasoning

79.09

69.28

69.28

63.61

61.54

65.38

65.85

70.19

Social & Cultural Context

79.58

72.79

72.79

77.26

###### Categories

79.31

58.62

65.52

50.00

Medical Imaging

83.57

77.40

77.40

68.80

82.24

82.24

78.00

76.42

Chart & Diagram Understanding

87.40

81.61

81.61

82.80

61.51

53.37

85.54

58.55

Complex Visual Perception

77.73

65.09

76.51

67.08

80.72

24.10

62.32

20.48

Scientific Reasoning

83.05

76.51

72.14

66.96

50.50

57.21

51.80

63.59

Math & Logic Reasoning

0 20 40 60 80 100

0 20 40 60 80 100

0 20 40 60 80 100

0 20 40 60 80 100

###### gemini-1.5-flash

###### GPT-4o-mini

###### LLAVA Cot

Our Model

|71.84| | | |
|---|---|---|---|
| | | | |

|72.79| | | |61.65| | | | |
|---|---|---|---|---|---|---|---|---|
| |Steps Score|Final A|nswer|Score| | | | |

|63.18| | | | |
|---|---|---|---|---|
| | | | | |

54.99

56.39

54.09

56.49

Overall

71.86

74.05

66.21

68.93

83.61

85.25

93.44

93.44

OCR & Document Understanding

87.50

88.55

81.17

83.69

43.48

60.87

73.91

73.91

Visual Reasoning

59.91

65.89

72.53

74.32

73.08

70.19

65.38

80.77

Social & Cultural Context

75.82

78.00

70.68

72.67

###### Categories

41.38

65.52

62.07

65.52

Medical Imaging

65.48

71.50

73.63

72.34

81.31

82.24

82.24

83.18

Chart & Diagram Understanding

83.83

86.78

77.36

80.28

45.60

48.70

28.50

29.53

Complex Visual Perception

64.00

66.87

54.17

61.93

13.25

14.46

84.31

86.75

Scientific Reasoning

63.91

68.01

80.96

80.24

53.73

53.48

42.04

42.54

Math & Logic Reasoning

0 20 40 60 80 100

0 20 40 60 80 100

0 20 40 60 80 100

0 20 40 60 80 100

- Figure 4: The comprehensive comparison of category-wise and overall performance scores achieved by various models on diverse reasoning tasks. The evaluation spans multiple domains, including Math & Logic Reasoning, Scientific Reasoning, Complex Visual Perception, Chart & Diagram Understanding, Medical Imaging, Social & Cultural Context, Visual Reasoning, and OCR & Document Understanding. The models assessed include GPT-4o, Claude-3.5-Sonnet, Gemini variants, LLAVA-CoT, and our proposed model. Our model demonstrates consistently superior performance in critical categories such as Math & Logic Reasoning, Chart & Diagram Understanding, and Medical Imaging, achieving a balanced improvement across both step-by-step reasoning (Step Scores) and final answer accuracy (Final Answer Scores). Compared to LLAVA-CoT, our approach excels in maintaining high accuracy across tasks while showcasing robustness and interpretability in multi-step reasoning challenges.

makes our method more adaptable to a wide range of reasoning scenarios, improving generalization across tasks.

Improved Efficiency with Beam Search: The Beam Search technique allows us to generate multiple reasoning paths in parallel and select the most optimal one. This approach enhances both the quality and consistency of the model’s outputs. By evaluating multiple candidates and selecting the best, we ensure that the final answer is logical and robust. One of the key advantages of our method is its computational efficiency. The inference time scaling of our approach has a complexity of O(n), which is significantly more efficient than LLava-CoT’s O(n2) scaling. This linear complexity ensures that our method is scalable to larger datasets and more complex reasoning tasks without a proportional increase in computational cost.

- 5 Experiments

In this section, we evaluate the performance of our proposed model trained with a curriculum learning strategy. We use Llama-3.2-11B-Vision-Instruct [47] as the baseline model, leveraging its robust vision-language understanding as a foundation. Training is conducted using the PixMo [13] subset and LLaVA-CoT-100k [66] datasets, carefully curated to support step-by-step reasoning tasks. The

curriculum learning framework ensures the model progresses from simpler tasks, such as generating summaries and captions, to more complex multi-step reasoning tasks involving detailed logic and final answer generation.

To assess the effectiveness of our approach, we evaluate the model on our newly proposed reasoning benchmark, designed to test multi-step chain-of-thought capabilities in multimodal contexts. Additionally, we benchmark our model against six commonly used multimodal benchmarks employed in LLaVA-CoT [66], covering diverse domains such as visual reasoning, mathematical problem-solving, and scientific reasoning. This comprehensive evaluation demonstrates the robustness of our method, highlighting significant improvements in both reasoning quality and inference efficiency compared to the baseline.

#### 5.1 Experimental Setup

For our experiments, we use Llama-3.2-11B-Vision-Instruct as the baseline model, fine-tuned using the llama-recipes framework in a Supervised Fine-Tuning (SFT) manner. This robust foundation enables us to implement our curriculum learning strategy effectively, training the model on reasoning tasks that progressively increase in complexity. The training datasets include PixMo and LLaVACoT-100k, which we specifically tailored to support multi-step reasoning tasks and detailed chain-ofthought explanations.

For evaluation, we utilize a diverse set of benchmarks, including both our proposed reasoning benchmark and six established multimodal benchmarks: MMStar [9], MMBench [35], MMVet [71], MathVista [39], AI2D [29], and Hallusion [21]. These benchmarks comprehensively evaluate the model’s capabilities across general visual question answering, mathematical and scientific reasoning, and handling language hallucinations and visual illusions. Our proposed benchmark is designed to assess step-by-step reasoning and final answer as described in the Section 3.2, and evaluations are conducted using a fuzzy evaluation strategy where GPT-4o acts as the judge, ensuring robust assessments of generated outputs. To maintain consistency and fairness, we adopt the VLMEvalKit [14] framework, as used in LLaVA-CoT, to evaluate all models on six established multimodal benchmarks. This open-source toolkit ensures reproducibility and allows for direct comparison of performance metrics across different models. Our rigorous experimental setup highlights the effectiveness of our approach in advancing multimodal reasoning capabilities.

#### 5.2 Results and Discussion

Our model demonstrates significant improvements over existing methods on our proposed reasoning benchmark, as shown in Table 2. The evaluation compares final answer accuracy and step-by-step reasoning performance across state-of-the-art models. While models like GPT-4o, Claude-3.5-Sonnet and Gemini-2.0-Flash exhibit strong reasoning capabilities, our approach achieves the better final answer accuracy (56.49) compared to GPT-4o-mini and LLava-CoT and competitive step scores (68.93%). This highlights the model’s ability to generate accurate outputs while maintaining logical coherence in multi-step tasks. The structured curriculum learning strategy and effective inference design have been critical in achieving these results.

- Figure 4 illustrates the category-wise performance of our model compared to leading reasoning models in various domains from our benchmark such as Math & Logic, Scientific Reasoning, and Complex Visual Perception. Our model outperforms others in several challenging categories, including Chart & Diagram Understanding (83.18%), Scientific Reasoning (86.75%) and OCR & Document Understanding (93.44%). These improvements outline the model’s ability to handle tasks requiring visual and logical reasoning in accordance. The results also highlight balanced performance across all categories, reflecting the versatility of our approach.

- Table 3 summarizes the performance of various models on six established benchmarks: MMStar, MMBench, MMVet, MathVista, AI2D, and HallusionBench. Among open-source models, our method achieves the highest average score of 67.33%, surpassing other prominent models like LLaVA-CoT (63.50). Notably, our model demonstrates significant strengths in reasoning-intensive benchmarks such as MMVet (65.40%) and Hallusion (63.51%). These results validate the effectiveness of our model in handling diverse and complex multimodal tasks.

- Table 3: Performance comparison on six benchmark datasets (MMStar [9], MMBench [35], MMVet [71], MathVista [39], AI2D [29], and Hallusion [21]) along with their average scores. The comparison includes both close-source and open-source models. The best performing closesource model is GPT-4o with an average score of 71.8%. Among open-source models, our proposed LlamaV-o1 achieves the best performance with an average score of 67.33% outperforming the recent Llava-CoT by 3.8%.

Model MMStar MMBench MMVet MathVista AI2D Hallusion Average Close-Source

GPT-4o-0806 [2] 66.0 82.4 80.8 62.7 84.7 54.2 71.8 Claude3.5-Sonnet-0620 [1] 64.2 75.4 68.7 61.6 80.2 49.9 66.7 Gemini-1.5-Pro [52] 56.4 71.5 71.3 57.7 79.1 45.6 63.6 GPT-4o-mini-0718 [48] 54.9 76.9 74.6 52.4 77.8 46.1 63.8

Open-Source

InternVL2-8B [10] 62.50 77.40 56.90 58.30 83.60 45.00 64.00 Ovis1.5-Gemma2-9B [41] 58.70 76.30 50.90 65.60 84.50 48.20 64.00 MiniCPM-V2.6-8B [70] 57.10 75.70 56.30 60.60 82.10 48.10 63.30 Llama-3.2-90B-Vision-Inst [47] 51.10 76.80 74.10 58.30 69.50 44.10 62.30 VILA-1.5-40B [36] 53.20 75.30 44.40 49.50 77.80 40.90 56.90 Mulberry-7B [68] 61.30 75.34 43.90 57.49 78.95 54.10 62.78 Llava-CoT [66] 57.60 75.00 60.30 54.80 85.70 47.80 63.50

Our Models

Llama-3.2-11B-Vision-Inst [47] (baseline) 49.80 65.80 57.60 48.60 77.30 40.30 56.90 LlamaV-o1 (Ours) 59.53 79.89 65.40 54.40 81.24 63.51 67.33

- Table 4: Impact of our proposed contributions on multimodal reasoning tasks across six benchmarks: MMStar, MMBench, MMVet, MathVista, AI2D, and Hallusion. Starting with Curriculum Learning combined with Multi-Step CoT reasoning (2nd row), the model achieves a 9.14% absolute gain compared to base model Llama-3.2-11B-Vision-Inst [47], demonstrating its ability to handle complex multi-step reasoning effectively. This baseline approach leverages structured training to improve performance across diverse tasks, including logical reasoning and visual understanding. By incorporating Beam Search, the model’s performance further improves (3rd row). This enhancement is particularly noticeable in benchmarks such as MMVet (65.40% vs. 61.88%), MathVista (54.40% vs. 53.20%), and AI2D (81.24% vs. 80.18%), showcasing the model’s ability to generalize better with more accurate reasoning. Our final approach that combines curriculum learning with optimized inference strategies achieves an overall average gain of 10.43%, compared to the baseline.

Model MMStar MMBench MMVet MathVista AI2D Hallusion Average

Llama-3.2-11B-Vision-Inst (baseline) 49.80 65.80 57.60 48.60 77.30 40.30 56.90 + Curriculum with Multi-Step CoT Reasoning 58.13 79.55 61.88 53.20 80.18 63.31 66.04 + Beam Search 59.53 79.89 65.40 54.40 81.24 63.51 67.33

The results demonstrate that our approach outperforms recent open-source visual reasoning methods while achieving favorable results against its close-source counterparts. By leveraging curriculum learning and optimizing inference efficiency with Beam Search, our model effectively balances reasoning accuracy and computational complexity. Our performance improvements in reasoning tasks are complemented by robust handling of logical errors and visual illusions, as evidenced in benchmarks like HallusionBench. Fig. 5 presents a qualitative comparison between the recent LlavaCoT and our LlamaV-o1 on different examples from the VRC-Bench. Our LlamaV-o1 achieves superior performance both in reasoning steps and the final answer, compared to Llava-CoT.

#### 5.3 Ablations

Impact of Proposed Components: The Table 4 showcases the impact of our proposed components of LlamaV-o1 on improving performance in complex visual reasoning tasks across six multimodal benchmarks: MMStar, MMBench, MMVet, MathVista, AI2D, and Hallusion. Starting with a curriculum learning strategy combined with multi-step Chain-of-Thought (CoT) reasoning, the model achieves an average score of 66.08%, demonstrating its ability to handle reasoning-intensive tasks effectively. By incorporating Beam Search, which optimizes the selection of reasoning paths, the performance further improves, achieving the highest average score of 67.33%. This improvement is particularly significant in benchmarks such as MMVet (65.40% vs. 61.88%), MMStar (59.53% vs. 58.13%), and AI2D (81.24% vs. 80.18%), which evaluate the model’s logical, visual, and contextual

|Llava-CoT|
|---|
|Reasoning Steps: First, I will locate the data for Boomers and Generation X who have used the service. For Boomers, the percentage is 80%, and for Generation X, it is 91%. Adding these two percentages gives 80% + 91% = 171%.<br><br>Final Answer: 171|

|LlamaV-o1|
|---|
|Reasoning Steps: First, I will locate the percentage of Boomers who have used the service, which is 3%. Next, I will find the percentage of Generation X who have used the service, which is 19%. Finally, I will add these two percentages together: 3% (Boomers) + 19% (Generation X) = 22%.<br><br>Final Answer: 22|

[Figure 14]

Question : What is the sum of the people of Boomers and Generation X who have used this service? Answer : 22

|Llava-CoT|
|---|
|Reasoning Steps: The presence of trees and a paved surface suggests an outdoor setting, typical of urban parks or public squares. The casual attire and informal gathering indicate a relaxed, public environment. These factors are commonly associated with a park or similar outdoor recreational area.<br><br>Final Answer: Park|

|LlamaV-o1|
|---|
|Reasoning Steps: The presence of a sidewalk, trees, and parked cars suggests that the game is being played in a public outdoor area, likely in a city or town. The casual attire of the players and the informal setting further support this conclusion.<br><br>Final Answer: The game is being played on a sidewalk in a city or town.|

[Figure 15]

Question : Where is this game being played? Answer : The game is being played outdoors on a sidewalk.

|Llava-CoT|
|---|
|Reasoning Steps:<br><br>1. Identify the brown SUV in the image, which is located on the left side. 2. Identify the brown rubber school bus in the image, which is located on the right side. 3. Count the number of brown SUVs, which is 1.<br><br>4. Count the number of brown rubber school buses, which is 1. 5. Compare the counts: 1 (brown SUV) is less than 1 (brown rubber school bus). 6. Determine that the number of brown SUVs is indeed less than the number of brown rubber school buses. Final Answer: A) Yes|

|LlamaV-o1|
|---|
|Reasoning Steps: Step-by-Step Process: Step 1: Identify the number of brown SUVs in the image. Action 1: There is only one brown SUV present in the image.<br><br>Step 2: Identify the number of brown rubber school buses in the image.<br><br>Action 2: There are zero brown rubber school buses present in the image. The bus is teal reddish-pink.<br><br>Step 3: Compare the number of brown SUVs and brown school buses.<br><br>Action 3: The number of brown SUVs<br><br><br><br><br>(1) is greater than the number of brown school buses (0). Final Answer: B) No|

[Figure 16]

Question : Is the number of brown suvs less than the number of brown rubber school buss? Choices: A) Yes B) No. Please select the correct option by its letter. Answer : B

- Figure 5: Qualitative comparison between Llava-CoT and the proposed LlamaV-o1 on examples from the VRC-Bench. First row: the example shows visual reasoning capabilities on an example chart. Here, Llava-CoT makes mistakes (highlighted in red) for both the intermediate steps and the final answer. In Comparison, our LlamaV-o1 provides an accurate description of the steps as well as the final answer. Second row: While both Llava-CoT and our LlamaV-o1 provide accurate step descriptions on an example real-world VQA, Llava-CoT fails to infer the final answer. Last row: Llava-CoT fails to accurately answer for the counting task, while also missing the intermediate counting steps. In contrast, our LlamaV-o1 model performs better in intermediate reasoning steps while also providing the accurate final answer.

reasoning abilities. These results highlight the effectiveness of combining progressive training with optimized inference, enabling the model to generalize better across complex tasks and consistently deliver accurate and coherent reasoning. This validates the importance of our proposed contributions in advancing multimodal reasoning systems.

- Table 5: Comparison of inference scaling techniques on the MMVet benchmark, evaluated using a single NVIDIA A100 GPU. Left: Llava-CoT with stage-level beam search shows improved MMVet scores with more beams but suffers from quadratic scaling, significantly increasing inference time. Right: Performance of our approach utilizing Beam Search achieving higher MMVet scores with much lower inference time, due to its linear scaling efficiency. For instance, our method scores 65.40 with four beams in 6.1 GPU hours, compared to Llava-CoT’s 62.9 score requiring 46.1 GPU hours. This demonstrates the efficiency and practicality of our approach for real-world applications.

Inference Scaling # Beams MMVet Score Time (GPU Hours) No Scaling 1 60.3 3.8

- Stage-level 2 61.7 20.1

- Stage-level 3 62.3 38.5

- Stage-level 4 62.9 54.1

Inference Scaling # Beams MMVet Score Time (GPU Hours) No Scaling 1 63.63 2.7

- Beam Search 2 64.26 4.8

- Beam Search 3 64.92 5.7

- Beam Search 4 65.40 6.1

Effectiveness of Inference Scaling Techniques: The Table 5 focuses on the efficiency and effectiveness of inference scaling techniques on the MMVet benchmark. We compare the newly introduced stage-level beam search used in Llava-CoT with Beam Search in our proposed approach. Both approaches are evaluated based on MMVet scores and inference time, measured on a single NVIDIA A100 GPU (80GB). Stage-Level Beam Search (Llava-CoT): Increasing the number of beams improves the MMVet score incrementally (from 60.3% with 1 beam to 62.9% with 4 beams). However, this improvement comes at a significantly higher computational cost due to linear scaling (time complexity of O(n)) based on model calls, with inference time rising from 3.8 GPU hours for 1 beam to 54.1 GPU hours for 4 beams. This scaling inefficiency limits the practicality of the stage-level approach for real-world applications. Beam Search (Ours): In contrast, our method achieves significantly better MMVet scores while maintaining a constant scaling (time complexity of O(1)) of inference time in terms of model calls. With 1 beam, our model already outperforms Llava-CoT (63.63% vs. 60.3%). As the number of beams increases, the MMVet score improves further, reaching 65.40 with 4 beams in just 4.2 GPU hours, a fraction of the computational cost of Llava-CoT. This demonstrates that Beam Search is not only more efficient with higher accuracy but also suitable for real-world applications.

- 6 Conclusion

In this paper, we proposed a comprehensive approach for advancing multimodal reasoning by introducing a new benchmark, a novel metric, and an innovative model trained using curriculum learning. Our model demonstrates significant improvements over existing methods, achieving stateof-the-art performance on challenging benchmarks while maintaining efficiency in inference. The incorporation of curriculum learning enabled the model to develop foundational reasoning skills progressively, resulting in improved generalization and robustness across diverse tasks. Our results highlight the effectiveness of our design choices, including the structured training strategy, efficient inference mechanism, and rigorous evaluation using both our benchmark and widely recognized datasets. Our contributions aim to provide a capable multimodal reasoning model, emphasizing the importance of interpretable, step-by-step reasoning in improving AI’s ability to handle complex, multi-step tasks.

### References

- [1] Claude 3.5 sonnet, 2024. Available at: https://www.anthropic.com/news/ claude-3-5-sonnet.
- [2] OpenAI (2024). Gpt-4o system card, 2024.
- [3] Aida Amini, Saadia Gabriel, Shanchuan Lin, Rik Koncel-Kedziorski, Yejin Choi, and Hannaneh Hajishirzi. MathQA: Towards interpretable math word problem solving with operation-based formalisms. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of

- the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2357– 2367, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.
- [4] Saeed Amizadeh, Hamid Palangi, Alex Polozov, Yichen Huang, and Kazuhito Koishida. Neurosymbolic visual reasoning: Disentangling. In International Conference on Machine Learning, pages 279–290. Pmlr, 2020.
- [5] Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. Neural module networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 39–48, 2016.
- [6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- [7] Jie Cao and Jing Xiao. An augmented benchmark dataset for geometric question answering through dual parallel text encoding. In Proceedings of the 29th International Conference on Computational Linguistics, pages 1511–1520, 2022.
- [8] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. In European Conference on Computer Vision, pages 370–387. Springer, 2025.
- [9] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024.
- [10] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024.
- [11] Ting-Rui Chiang and Yun-Nung Chen. Semantically-aligned equation generation for solving and reasoning math word problems. In Jill Burstein, Christy Doran, and Thamar Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2656–2668, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.
- [12] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1– 113, 2023.
- [13] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. arXiv preprint arXiv:2409.17146, 2024.
- [14] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, Dahua Lin, and Kai Chen. Vlmevalkit: An open-source toolkit for evaluating large multi-modality models, 2024.
- [15] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. arXiv preprint arXiv:2404.12390, 2024.
- [16] Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, et al. G-llava: Solving geometric problem with multi-modal large language model. arXiv preprint arXiv:2312.11370, 2023.

- [17] Artur d’Avila Garcez, Marco Gori, Luis C Lamb, Luciano Serafini, Michael Spranger, and Son N Tran. Neural-symbolic computing: An effective methodology for principled integration of machine learning and reasoning. arXiv preprint arXiv:1905.06088, 2019.
- [18] Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. Transactions of the Association for Computational Linguistics, 9:346–361, 2021.
- [19] Olga Golovneva, Moya Peng Chen, Spencer Poff, Martin Corredor, Luke Zettlemoyer, Maryam Fazel-Zarandi, and Asli Celikyilmaz. Roscoe: A suite of metrics for scoring step-by-step reasoning. In The Eleventh International Conference on Learning Representations.
- [20] Tanya Goyal, Junyi Jessy Li, and Greg Durrett. News summarization and evaluation in the era of gpt-3. arXiv preprint arXiv:2209.12356, 2022.
- [21] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14375–14385, June 2024.
- [22] Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14953–14962, 2023.
- [23] Yingdong Hu, Fanqi Lin, Tong Zhang, Li Yi, and Yang Gao. Look before you leap: Unveiling the power of gpt-4v in robotic vision-language planning. arXiv preprint arXiv:2311.17842, 2023.
- [24] Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. Large language models can self-improve. arXiv preprint arXiv:2210.11610, 2022.
- [25] Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2901–2910, 2017.
- [26] Justin Johnson, Andrej Karpathy, and Li Fei-Fei. Densecap: Fully convolutional localization networks for dense captioning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4565–4574, 2016.
- [27] Li Junnan, Wong Yong Kang, Zhao Qi, and Mohan S Kankanhalli. People in social context (pisc) dataset. 2017.
- [28] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235–251. Springer, 2016.
- [29] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235–251. Springer, 2016.
- [30] Aniruddha Kembhavi, Minjoon Seo, Dustin Schwenk, Jonghyun Choi, Ali Farhadi, and Hannaneh Hajishirzi. Are you smarter than a sixth grader? textbook question answering for multimodal machine comprehension. In Proceedings of the IEEE Conference on Computer Vision and Pattern recognition, pages 4999–5007, 2017.
- [31] Salman Hameed Khan, Mohammed Bennamoun, Ferdous Sohel, and Roberto Togneri. Geometry driven semantic labeling of indoor scenes. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part I 13, pages 679–694. Springer, 2014.

- [32] Andrew K Lampinen, Nicholas Roy, Ishita Dasgupta, Stephanie CY Chan, Allison Tam, James Mcclelland, Chen Yan, Adam Santoro, Neil C Rabinowitz, Jane Wang, et al. Tell me why! explanations support learning relational and causal structure. In International Conference on Machine Learning, pages 11868–11890. PMLR, 2022.
- [33] Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ili´c, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. Bloom: A 176b-parameter open-access multilingual language model. 2023.
- [34] Adam Dahlgren Lindström and Savitha Sam Abraham. Clevr-math: A dataset for compositional language, visual and mathematical reasoning. arXiv preprint arXiv:2208.05358, 2022.
- [35] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European Conference on Computer Vision, pages 216–233. Springer, 2025.
- [36] Zhijian Liu, Ligeng Zhu, Baifeng Shi, Zhuoyang Zhang, Yuming Lou, Shang Yang, Haocheng Xi, Shiyi Cao, Yuxian Gu, Dacheng Li, et al. Nvila: Efficient frontier visual language models. arXiv preprint arXiv:2412.04468, 2024.
- [37] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.
- [38] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR), 2024.
- [39] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR), 2024.
- [40] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022.
- [41] Shiyin Lu, Yang Li, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, and Han-Jia Ye. Ovis: Structural embedding alignment for multimodal large language model. arXiv:2405.20797, 2024.
- [42] Yinquan Lu, Wenhao Zhu, Lei Li, Yu Qiao, and Fei Yuan. Llamax: Scaling linguistic horizons of llm by enhancing translation capabilities beyond 100 languages. arXiv preprint arXiv:2407.05975, 2024.
- [43] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems, 36, 2024.
- [44] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022.
- [45] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021.
- [46] Minesh Mathew, Dimosthenis Karatzas, R Manmatha, and CV Jawahar. Docvqa: A dataset for vqa on document images. corr abs/2007.00398 (2020). arXiv preprint arXiv:2007.00398, 2020.
- [47] Meta AI. Llama 3.2: Revolutionizing edge ai and vision with open, customizable models. https: //ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/, 2024.

- [48] OpenAI. Gpt-4o mini: advancing cost-efficient intelligence. https://openai.com/index/ gpt-4o-mini-advancing-cost-efficient-intelligence/, 2024.
- [49] OpenAI. Introducing openai o1-preview, 2024. Accessed: 2024-12-16.
- [50] Archiki Prasad, Swarnadeep Saha, Xiang Zhou, and Mohit Bansal. Receval: Evaluating reasoning chains via correctness and informativeness. In The 2023 Conference on Empirical Methods in Natural Language Processing.
- [51] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.
- [52] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, and et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context, 2024.
- [53] Subhro Roy and Dan Roth. Solving general arithmetic word problems. In Lluís Màrquez, Chris Callison-Burch, and Jian Su, editors, Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 1743–1752, Lisbon, Portugal, September 2015. Association for Computational Linguistics.
- [54] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In European conference on computer vision, pages 146–162. Springer, 2022.
- [55] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.
- [56] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [57] Ashmal Vayani, Dinura Dissanayake, Hasindri Watawana, Noor Ahsan, Nevasini Sasikumar, Omkar Thawakar, Henok Biadglign Ademtew, Yahya Hmaiti, Amandeep Kumar, Kartik Kuckreja, Mykola Maslych, Wafa Al Ghallabi, Mihail Mihaylov, Chao Qin, Abdelrahman M Shaker, Mike Zhang, Mahardika Krisna Ihsani, Amiel Esplana, Monil Gokani, Shachar Mirkin, Harsh Singh, Ashay Srivastava, Endre Hamerlik, Fathinah Asma Izzati, Fadillah Adamsyah Maani, Sebastian Cavada, Jenny Chim, Rohit Gupta, Sanjay Manjunath, Kamila Zhumakhanova, Feno Heriniaina Rabevohitra, Azril Amirudin, Muhammad Ridzuan, Daniya Kareem, Ketan More, Kunyang Li, Pramesh Shakya, Muhammad Saad, Amirpouya Ghasemaghaei, Amirbek Djanibekov, Dilshod Azizov, Branislava Jankovic, Naman Bhatia, Alvaro Cabrera, Johan Obando-Ceron, Olympiah Otieno, Fabian Farestam, Muztoba Rabbani, Sanoojan Baliah, Santosh Sanjeev, Abduragim Shtanchaev, Maheen Fatima, Thao Nguyen, Amrin Kareem, Toluwani Aremu, Nathan Xavier, Amit Bhatkal, Hawau Toyin, Aman Chadha, Hisham Cholakkal, Rao Muhammad Anwer, Michael Felsberg, Jorma Laaksonen, Thamar Solorio, Monojit Choudhury, Ivan Laptev, Mubarak Shah, Salman Khan, and Fahad Khan. All languages matter: Evaluating lmms on culturally diverse 100 languages, 2024.
- [58] Ramakrishna Vedantam, Karan Desai, Stefan Lee, Marcus Rohrbach, Dhruv Batra, and Devi Parikh. Probabilistic neural symbolic models for interpretable visual question answering. In International Conference on Machine Learning, pages 6428–6437. PMLR, 2019.
- [59] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023.
- [60] Yiming Wang, Zhuosheng Zhang, and Rui Wang. Element-aware summarization with large language models: Expert-aligned evaluation and chain-of-thought method. arXiv preprint arXiv:2305.13412, 2023.

- [61] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [62] Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Shengping Liu, Bin Sun, Kang Liu, and Jun Zhao. Large language models are better reasoners with self-verification. arXiv preprint arXiv:2212.09561, 2022.
- [63] Siwei Wu, Zhongyuan Peng, Xinrun Du, Tuney Zheng, Minghao Liu, Jialong Wu, Jiachen Ma, Yizhi Li, Jian Yang, Wangchunshu Zhou, et al. A comparative study on reasoning patterns of openai’s o1 model. arXiv preprint arXiv:2410.13639, 2024.
- [64] Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts, 2024.
- [65] Tianyi Xiong, Xiyao Wang, Dong Guo, Qinghao Ye, Haoqi Fan, Quanquan Gu, Heng Huang, and Chunyuan Li. Llava-critic: Learning to evaluate multimodal models. arXiv preprint arXiv:2410.02712, 2024.
- [66] Guowei Xu, Peng Jin, Li Hao, Yibing Song, Lichao Sun, and Li Yuan. Llava-o1: Let vision language models reason step-by-step. arXiv preprint arXiv:2411.10440, 2024.
- [67] Guowei Xu, Peng Jin, Li Hao, Yibing Song, Lichao Sun, and Li Yuan. Llava-o1: Let vision language models reason step-by-step. arXiv preprint arXiv:2411.10440, 2024.
- [68] Huanjin Yao, Jiaxing Huang, Wenhao Wu, Jingyi Zhang, Yibo Wang, Shunyu Liu, Yingjie Wang, Yuxin Song, Haocheng Feng, Li Shen, et al. Mulberry: Empowering mllm with o1-like reasoning and reflection via collective monte carlo tree search. arXiv preprint arXiv:2412.18319, 2024.
- [69] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in Neural Information Processing Systems, 36, 2024.
- [70] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024.
- [71] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.
- [72] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of CVPR, 2024.
- [73] Ruohong Zhang, Bowen Zhang, Yanghao Li, Haotian Zhang, Zhiqing Sun, Zhe Gan, Yinfei Yang, Ruoming Pang, and Yiming Yang. Improve vision language model chain-of-thought reasoning. arXiv preprint arXiv:2410.16198, 2024.
- [74] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.
- [75] Tianyi Zhang, Faisal Ladhak, Esin Durmus, Percy Liang, Kathleen McKeown, and Tatsunori B Hashimoto. Benchmarking large language models for news summarization. Transactions of the Association for Computational Linguistics, 12:39–57, 2024.
- [76] Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models, 2024.

### A Appendix

#### A.1 Generating reasoning Steps from Closed Sourced Models

We designed a structured system prompt to guide closed-source models like GPT-4o [2], Claude [1], and Gemini [52] in generating detailed, step-by-step reasoning for complex tasks. The prompt requires the model to describe each action to be taken and explain how it is executed, ensuring a clear and logical progression throughout the reasoning process. To account for varying levels of complexity, the prompt allows the model to take as many steps as necessary, ensuring that the solution is systematically derived. Additionally, the prompt emphasizes the use of visual elements, guiding the model to reference provided images or diagrams explicitly in its reasoning steps.

The prompt is further designed to handle ambiguity effectively by instructing the model to respond with "None of the choices provided" when no valid options are available. This ensures robustness and prevents the generation of forced or inaccurate conclusions. By enforcing a logical flow, grounding the reasoning in visual inputs, and providing explicit instructions for ambiguous scenarios, this prompt enables consistent, interpretable, and reliable reasoning outputs across various multimodal tasks.

#### System Prompt used for the generation of reasoning steps

When answering the question based on the provided image(s), follow a structured reasoning process and provide the final answer after solving it step by step. Use the following format for your response:

Step -by-Step Process:

- Step 1: Describe the action to be taken.

- Action 1: Explain the execution of the first action.

Step 2: Describe the next action to be taken.

- Action 2: Explain the execution of the second action.

Step 3: Describe the next action to be taken.

- Action 3: Explain the execution of the second action.

...continue as needed...take as many steps you want. Step n: Describe the final action to be taken. Action n: Execute the final action leading to the conclusion. Final Answer: Provide the final solution or conclusion derived

from the reasoning process.

Ensure each step logically follows the previous one , and explicitly detail how the image(s) guide the solution at every stage. Also if options are present and none of options are correct. Please response None of the choices provided.

#### A.2 System Prompt used to Evaluate Reasoning Steps

The following system prompt was used to evaluate the reasoning steps of the target model. It defines a structured framework to assess the alignment, coherence, and quality of reasoning through multiple metrics, including faithfulness, informativeness, repetition, hallucination, redundancy, semantic coverage, reasoning alignment, commonsense, and completeness of steps. Each metric is scored on a scale of 1-10, with detailed guidelines ensuring consistent and objective evaluations.

#### System Prompt used to evaluate the reasoning steps

You are a reasoning evaluator designed to assess the alignment ,

coherence , and quality of reasoning steps in text responses. Your task is to evaluate reasoning steps between the *ground truth* and the *LLM response* using the following metrics:

- 1. **Faithfulness -Step (1-10):**

- - Definition: Measures how well the reasoning steps in the LLM response align with the source reasoning steps.
- - Scoring Guidelines:
- - 9-10: All or almost all steps match or closely reflect the ground truth reasoning.
- - 7-8: Most steps are aligned , with minor deviations.
- - 5-6: Some steps align , but several are missing or significantly altered.
- - 3-4: Few steps align correctly; most are off or missing.
- - 1-2: The majority of steps are not aligned with the source.

- 2. **Faithfulness -Token (1-10):**

- - Definition: Extends Faithfulness -Step to a token -level granularity , checking if the content within each reasoning step is true to the source.
- - Scoring Guidelines:
- - 9-10: Token -level details mirror the ground truth closely.
- - 7-8: Minor token -level deviations but largely faithful.
- - 5-6: Noticeable inaccuracies in token -level details.
- - 3-4: Several token -level discrepancies.
- - 1-2: Most token -level details are incorrect or fabricated.

- 3. **Informativeness -Step (Info -Step) (1-10):**

- - Definition: Measures how well the reasoning steps extract all relevant information from the source.
- - Scoring Guidelines:
- - 9-10: Almost all critical information steps are present and accurate.
- - 7-8: Most important points are included , with minor omissions.
- - 5-6: Some key information is missing or underdeveloped.
- - 3-4: Limited inclusion of critical content.
- - 1-2: Very poor extraction of relevant information.

- 4. **Repetition -Token (1-10):**

- - Definition: Identifies repeated or unnecessarily paraphrased reasoning steps within the hypothesis.
- - Scoring Guidelines:
- - 9-10: No or minimal unnecessary repetition.
- - 7-8: Minor repetition that doesn ’t impede clarity.
- - 5-6: Noticeable repetition that doesn ’t add value.
- - 3-4: Frequent repetition that disrupts coherence.
- - 1-2: Excessive repetition reducing the quality of reasoning.

- 5. **Hallucination (1-10):**

- - Definition: Detect irrelevant or invented reasoning steps not aligned with the source.
- - Scoring Guidelines:
- - 9-10: No hallucinations; all reasoning is grounded in the source.
- - 7-8: One or two minor hallucinations.
- - 5-6: Several steps contain invented or irrelevant details.
- - 3-4: Many hallucinations , but some grounding remains.
- - 1-2: Mostly hallucinated reasoning.

- 6. **Redundancy (1-10):**

- - Definition: Identify redundant reasoning steps that do not add value.
- - Scoring Guidelines:
- - 9-10: No unnecessary steps; very concise.
- - 7-8: Minor redundancy.
- - 5-6: Some steps clearly unnecessary.
- - 3-4: Many redundant steps.
- - 1-2: Excessive redundancy that hampers clarity.

- 7. **Semantic Coverage -Step (1-10):**

- - Definition: How well the hypothesis covers the essential semantic elements from the source reasoning steps.
- - Scoring Guidelines:
- - 9-10: Almost complete semantic coverage of all important elements.
- - 7-8: Good coverage but some minor elements are missing.
- - 5-6: Partial coverage with noticeable gaps.
- - 3-4: Significant semantic gaps.
- - 1-2: Very poor coverage of essential meaning.

- 8. **Reasoning Alignment (1-10):**

- - Definition: Overall alignment between the hypothesis and the reference reasoning chain.
- - Scoring Guidelines:
- - 9-10: Very closely aligned , minimal divergence.
- - 7-8: Mostly aligned , with some minor issues.
- - 5-6: Some alignment , but also several misalignments.
- - 3-4: Poor alignment , though occasional matches.
- - 1-2: Fundamentally misaligned reasoning.

- 9. **Commonsense (1-10):**

- - Definition: Check for missing commonsense reasoning required to solve the problem.
- - Scoring Guidelines:
- - 9-10: Adequate commonsense reasoning present.
- - 7-8: Minor commonsense gaps but mostly adequate.
- - 5-6: Noticeable commonsense gaps.
- - 3-4: Many commonsense steps missing.
- - 1-2: Almost entirely lacking necessary commonsense.

- 10. **Missing Step (1-10):**

- - Definition: Identify if any necessary reasoning steps are missing.
- - Scoring Guidelines:
- - 9-10: No critical steps missing.
- - 7-8: Minor missing steps that don ’t significantly affect the conclusion.
- - 5-6: Some important steps absent , affecting the outcome.
- - 3-4: Several crucial missing steps.
- - 1-2: Major gaps; the reasoning chain is incomplete.

**Additional Instructions for Consistency:**

- - Always follow the above scoring guidelines strictly.
- - Before scoring , re-read both the ground truth and the LLM response carefully.
- - Compare the reasoning steps directly to determine where they align or diverge.
- - Use the provided scoring benchmarks (anchor examples , if any) as a reference to maintain consistency across evaluations.
- - Avoid subjective interpretation and adhere to the given thresholds.
- - Once scores for all metrics are determined , compute the Overall Score as the average of all metric scores.
- - Provide the final output as a Python dictionary with the structure only don’t add a anything extra , beacuase your out will be used in code pipeline. So single change in you output will crash whole system. :

# Example output : {’Faithfulness -Step ’: 8.0, ’Faithfulness -Token ’:

7.5, ’Informativeness -Step ’: 8.5, ’Repetition -Token ’: 9.0, ’ Hallucination ’: 9.5, ’Redundancy ’: 8.0, ’Semantic Coverage -Step ’: 8.5, ’Reasoning Alignment ’: 8.0, ’Commonsense ’: 9.0, ’Missing

Step ’: 8.5 , ’Overall Score ’: 8.65} # Do not give output in following format : ‘‘‘python {

’Faithfulness -Step ’: 1.0, ’Faithfulness -Token ’: 1.0, ’Informativeness -Step ’: 1.0, ’Repetition -Token ’: 9.0, ’Hallucination ’: 1.0, ’Redundancy ’: 9.0, ’Semantic Coverage -Step ’: 1.0, ’Reasoning Alignment ’: 1.0, ’Commonsense ’: 1.0, ’Missing Step ’: 1.0, ’Overall Score ’: 2.6

} ‘‘‘

#### A.3 Response format used to generate structured evaluation scores

To further ensure the evaluation framework generates consistent and interpretable outputs, we designed the response format using a well-defined JSON schema. This schema serves as a blueprint, enforcing strict adherence to a structured format while capturing detailed scores for each metric in a systematic and transparent manner. By standardizing the output structure, the schema facilitates easier comparison between models, reduces ambiguity, and enhances the reproducibility of results.

The JSON schema is carefully tailored to accommodate the unique aspects of our evaluation process, such as step-by-step reasoning, metric-specific scores, and logical flow validation. Each response is divided into key components, including reasoning steps, metric scores, and final answers, ensuring that all critical aspects of the model’s performance are systematically captured. This level of detail not only improves interpretability but also enables fine-grained analysis of strengths and weaknesses across models.

Additionally, the schema supports modularity, allowing seamless integration of new metrics or evaluation criteria as the benchmark evolves. By adopting this structured approach, we ensure that the evaluation framework remains robust, scalable, and adaptable to future advancements in multimodal reasoning research.

#### Response format provided to LLM’s which supports structured-output

response_format = { "type": "json_schema", "json_schema": {

"name": "EvaluationScores", "strict": True , "schema": {

"type": "object", "properties": {

"Faithfulness -Step": {"type": "number"}, "Faithfulness -Token": {"type": "number"}, "Informativeness -Step": {"type": "number"}, "Repetition -Token": {"type": "number"}, "Hallucination": {"type": "number"}, "Redundancy": {"type": "number"}, "Semantic Coverage -Step": {"type": "number"}, "Reasoning Alignment": {"type": "number"}, "Commonsense": {"type": "number"}, "Missing Step": {"type": "number"}, "Overall Score": {"type": "number"}

}, "required": [

"Faithfulness -Step", "Faithfulness -Token", "Informativeness -Step", "Repetition -Token", "Hallucination", "Redundancy", "Semantic Coverage -Step", "Reasoning Alignment", "Commonsense", "Missing Step", "Overall Score"

], "additionalProperties": False

} }

}

#### A.4 Evaluating reasoning steps using gpt-4o as a judge

The evaluate_steps function is designed to rigorously assess the quality of reasoning steps generated by models against ground truth data using the GPT-4o-mini model. It takes the task question, ground truth reasoning, and model response as inputs and processes them within a structured conversation context. By leveraging a predefined system prompt and parameters like deterministic temperature (0.0) and a maximum token limit of 500, the function ensures consistent and reliable evaluations. The output provides clear feedback on alignment, logical flow, and coherence of reasoning steps, enabling precise analysis of model performance. This automated and standardized approach enhances objectivity, reproducibility, and detailed insight into multimodal reasoning capabilities.

#### Reasoning steps evaluation using gpt-4o as a judge

def evaluate_steps(question , ground_truth , llm_response):

messages = [ {"role": "system", "content": system_prompt}, {

"role": "user", "content": [

{"type": "text", "text": question + "\n" + f"Ground Truth : {ground_truth}" + "\n" + f"LLM Response : {llm_response}" },

], }

] response = client.chat.completions.create(

model="gpt -4o-mini", messages=messages , response_format=response_format , max_tokens=500, temperature = 0.0,

) return response.choices[0].message.content

#### A.5 Evaluating final answer accuracy

To objectively assess how well the model’s final answer predictions align with the ground truth, we developed a comparison function that utilizes a secondary system prompt to evaluate response accuracy. This function analyzes the semantic similarity between the ground truth and the model’s output, assigning a binary score: 1 for a match and 0 for a mismatch. By exclusively producing numeric scores, this approach ensures a precise and quantifiable evaluation of the model’s performance, effectively complementing the structured framework outlined earlier.

#### Secondary system prompt to evaluate the final answer accuracy

system_prompt_2 = """ You are a helpful Assistant. Provide helpful response to the user ’s question.

"""

#### Evaluate the Final Answer

def compare_results(question , ground_truth , llm_response):

messages = [ {"role": "system", "content": system_prompt_2}, {

"role": "user", "content": [

{ "type": "text", "text": f"""

Evaluate the following answer based on Accuracy: Question: {question} Ground Truth: {ground_truth} Model Prediction: {llm_response} Match the meaning of the ground truth with the model prediction and if it matches give a 1. Otherwise 0. Strictly return only the numeric score , without any additional commentary

""" },

], }

] response = client.chat.completions.create(

model="gpt -4o-mini", messages=messages , max_tokens=10, temperature =0.0

) return response.choices[0].message.content

