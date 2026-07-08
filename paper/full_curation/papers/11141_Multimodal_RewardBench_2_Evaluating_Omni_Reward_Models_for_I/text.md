# arXiv:2512.16899v3[cs.CL]19Jan2026

## Multimodal RewardBench 2: Evaluating Omni Reward Models for Interleaved Text and Image

Yushi Hu∗, Reyhane Askari-Hemmat∗, Melissa Hall, Emily Dinan, Luke Zettlemoyer, Marjan Ghazvininejad FAIR at Meta Superintelligence Labs

∗Equal Contribution

Reward models (RMs) are essential for training large language models (LLMs), but remain underexplored for omni models that handle interleaved image and text sequences. We introduce Multimodal RewardBench 2 (MMRB2), the first comprehensive benchmark for reward models on multimodal understanding and (interleaved) generation. MMRB2 spans four tasks: text-to-image, image editing, interleaved generation, and multimodal reasoning (“thinking-with-images”), providing 1,000 expertannotated preference pairs per task from 23 models and agents across 21 source tasks. MMRB2 is designed with: (1) practical but challenging prompts; (2) responses from state-of-the-art models and agents; and (3) preference pairs with strong human-expert consensus, curated via an ensemble filtering strategy. Using MMRB2, we study existing judges for each subtask, including multimodal LLM-as-a-judge and models trained with human preferences. The latest Gemini 3 Pro attains 75-80% accuracy. GPT-5 and Gemini 2.5 Pro reach 66–75% accuracy, compared to >90% for humans, yet surpass the widely used GPT-4o (59%). The best performing open-source model Qwen3-VL-32B achieves similar accuracies as Gemini 2.5 Flash (64%). We also show that MMRB2 performance strongly correlates with downstream task success using Best-of-N sampling and conduct an in-depth analysis that shows key areas to improve the reward models going forward.

Date: January 21, 2026 Correspondence: Yushi Hu at yushihu@meta.com, Reyhane Askari-Hemmat at reyhaneaskari@meta.com, Marjan Ghazvininejad at ghazvini@meta.com Code and data: https://github.com/facebookresearch/MMRB2

1 Introduction

Reward models are central to the development of LLMs (Christiano et al., 2017; Bai et al., 2022; Jaech et al.,

- 2024; Guo et al., 2025; Lambert et al., 2024; Yuan et al., 2024). They enable scalable evaluation that tracks model performance and surfaces systematic weaknesses (Zheng et al., 2023). They can be used to assess data quality, which is crucial for building synthetic data pipelines (Wang et al., 2022b). And, as reinforcement learning becomes increasingly important in post training, high quality reward models are crucial for surfacing or suppressing a range of different model capabilities (Christiano et al., 2017; Wu et al., 2023c; Guo et al., 2025). Recent work has focused on developing new classes of omni models, which enable understanding, generation, and reasoning with interleaved text and images (OpenAI, 2024; Chameleon Team, 2024; Ge et al.,
- 2025; Zhou et al., 2024; Deng et al., 2025; Chen et al., 2025c; Wang et al., 2024b; Chen et al., 2025a; Google DeepMind, 2025a). However, reward modeling for omni models remains largely unexplored.

This omission is at least in part because there is no existing benchmark for omni reward models, making it nearly impossible to measure model quality. Unlike text-only models, omni models can generate and understand any number of texts and images together in a single arbitrarily ordered sequence. This generality creates unique challenges for reward modeling. Unlike domains such as math or coding, images are difficult to verify automatically (Hessel et al., 2021; Hu et al., 2023; Lin et al., 2024), and high-quality preference data requires carefully designed annotation protocols (Liang et al., 2024). Omni models can also be used for a very broad range of real-world applications, demanding diverse task coverage for both training and evaluation (Liu et al., 2024; Chen et al., 2024; Yao et al., 2025). Finally, gathering high-quality responses needed to train and evaluate omni reward models can be challenging, since omni model capabilities are not always as strong as

Text-to-Image

Image Editing:

Prompt: Create a whimsical advertisement for a summer road trip, featuring the yellow van from the ﬁrst image parked at a seaside location with the butterﬂy from the second image enlarged and gently landing on the van’s roof, with a bright, playful atmosphere and vibrant, inviting colors. Add fun text like “Adventure Awaits!” and stylized graphics that evoke freedom and travel.

[Figure 1]

[Figure 2]

Prompt: A cloudy sky frames a tall metal pole with a green directional sign. The sign displays two paths: "AIRPORT," “CITY CENTER," and "mathematicsIN STATION" with arrows pointing forward, while "MUSEUM," "SHOPPING PLAZA," and "TECH PARK" direct travelers to the left.

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

(A) FLUX 1.1 (B) Imagen 4

Human: (B) is better. In (A) the details of the van is missing. Both image has similar quality.

[Figure 7]

[Figure 8]

Human: (B) is better. In (A) the words are not organized on the correct sign, with errors in “MUSEM”, and garbled text in the top sign.

[Figure 9]

GPT-4o: (A) is better. The signs are similar, the sky in (A) is more aesthetically pleasing.

GPT-4o: (A) is better. The style is more coherent.

[Figure 10]

Interleaved Generation

[Figure 11]

(A) GPT-Image-1 (B) Nano Banana

Prompt: I will give you an image of a farm ﬁeld at the beginning of the planting season. Please show me the growth process of crops over time using a combination of 4 images and text.

Multimodal Reasoning (“thinking with images”):

[Figure 12]

(A) Nano Banana

(B) GPT-Flux agent

[Figure 13]

[Figure 14]

Prompt: if I was positioned where the stacked chairs, facing the same direction as it, what would be the nearest object to my right? A. Red chair upside down on a small red table B. Green chair

Here's the growth process of crops over time: Early Growth: [...]

At the start of planting season, [...]

(A) GPT-4.1

The green chair is on the right side of you.

[...]

[...]

(B) o3-Sketchpad-Agent

[Figure 15]

[Figure 16]

[Figure 17]

Here is the same image with an arrow showing the direction of “your right” when you’re in the stacked chairs: [IMAGE] The ﬁrst thing in that direction is the red upside-down chair on the red table.

Harvest Time: Abundance and Reward […]

By harvest time [...]

[IMAGE]

[Figure 18]

[Figure 19]

Human: (A) is better. The images are consistent with the text, and the crops are growing. In (B) the images are not following the texts, and the crops are not “harvesting” in the end.

Human: (B) got the correct answer with correct and consistent reasoning. (A) is wrong.

[Figure 20]

GPT-4o: (B) is has the correct reasoning.

GPT-5: (A) is better. The images and accompanying text demonstrate a higher degree of consistency with the prompt…

[Figure 21]

Figure1 Examples of multimodal preference pairs in MMRB2 across four subtasks: text-to-image generation, interleaved generation, image editing, and multimodal reasoning, showing human and model judgments on challenging prompts.

the models used to develop previous text-only reward benches.

We introduce Multimodal RewardBench 2 (MMRB2) which overcomes all of these challenges to establish a foundation for future research on omni reward modeling. MMRB2 follows Multimodal RewardBench (MMRB1) (Yasunaga et al., 2025), which covered image-text-to-text tasks for multimodal large language models (MLLMs). MMRB2 instead covers the much more challenging case of omni models over four subtasks (Figure 1): text-to-image, image editing, interleaved generation, and multimodal reasoning (“thinking with images” (OpenAI, 2025d)). Each subtask contains 1,000 expert-annotated preference pairs, consisting of a task prompt, a preferred response, and a rejected response. To ensure that MMRB2 is comprehensive, reliable, and highly predictive of reward model quality, we design it with three key characteristics: (1) diverse, practical, yet challenging prompts near the capability boundary of frontier models, drawn from 21 existing and newly created tasks; (2) responses generated by state-of-the-art multimodal models, ranging from SD3.5 (Stability AI, 2024) to GPT-Image (OpenAI, 2025c) and Gemini 2.5 Flash Image (Google DeepMind, 2025a), along with specialized agents (Hu et al., 2024) for interleaved generation and visual reasoning tasks where even the best models often fail; and (3) preference pairs that have >90% agreement among human experts but which remain challenging for current judges (both MLLM-based judges and trained reward models), curated via an ensemble filtering strategy. A summary of all the prompts and multimodal models covered in MMRB2 are in Table 1.

Using MMRB2, we conduct a comprehensive study of reward models for multimodal understanding and generation, including multimodal LLM-as-a-judge, task-specific metrics, and reward models trained with human preferences. Experiments show that:

Category Source Response Models Task Description

Image Gemini 2.0 and 2.5 Flash Image (Google DeepMind, 2025c,a)

Text-to-Image WISE (Niu et al., 2025) EvalMuse (Han et al., 2024) OneIG-Bench (Chang et al., 2025) R2IBench (Chen et al., 2025b) RealUnify (Shi et al., 2025)

Image generation from text assessing creativity, composition, reasoning, text rendering, etc.

- Imagen 3 (Baldridge et al., 2024)
- Imagen 4 and Ultra (Google DeepMind, 2025d) FLUX (Labs et al., 2025) GPT-image-1 (OpenAI, 2025c) SD 3.5-L (Stability AI, 2024)

Image Editing DreamBench (Peng et al., 2025) Emu-Edit (Sheynin et al., 2024) HQ-Edit (Hui et al., 2024) RISE-Bench (Zhao et al., 2025) Text-heavy edit Multi-Image edit

Image Gemini-2.0 and 2.5 Flash Image (Google DeepMind, 2025c,a) Imagen3-Edit (Baldridge et al., 2024) FLUX-Kontext (Labs et al., 2025) GPT-image-1 (OpenAI, 2025c)

Object replacement, scene modification, style change, entitypreserving editing, reasoning-heavy editing, text-heavy editing, multiimage editing, etc.

Text+Image Gemini 2.0 and 2.5 Flash Image (Google DeepMind, 2025c,a)

Interleaved Generation Chameleon (Chameleon Team, 2024) Interleaved-Eval (Liu et al., 2024) ISG-Bench (Chen et al., 2024) MMMG (Yao et al., 2025)

Interleaved text-image generation, storytelling, open-ended visual question answering, scene composition, 3D prediction, temporal prediction, etc.

Agents:

GPT-Gemini-agent GPT-GPT-image-agent GPT-Imagen-agent GPT-FLUX-agent

Reasoning BLINK (Fu et al., 2024) MindCube (Yin et al., 2025) VisuLogic (Xu et al., 2025) V∗ (Wu and Xie, 2023) MuirBench (Wang et al., 2024a) RealUnify (Shi et al., 2025)

Text(+Image) GPT-5 (OpenAI, 2025b) GPT-4.1 (OpenAI, 2025a) GPT-4o (OpenAI, 2024)

Thinking with images, spatial reasoning, multi-image reasoning, perception-heavy tasks, etc.

- o3 (OpenAI, 2025d) Gemini 2.5 Flash (Gemini Team, 2025) Gemini 2.5 Pro (Gemini Team, 2025) Sketchpad Agents (Hu et al., 2024):
- o3-sketchpad-agent GPT-5-sketchpad-agent

- Table 1 Overview of the four subtask categories in MMRB2, including their prompt sources, response modalities, model that were used to synthesize the data, and task descriptions. The benchmark draws from a diverse set of public and newly created datasets to cover text-to-image generation, image editing, interleaved text–image generation, and multimodal reasoning ("thinking with images").

- • MMRB2 poses significant challenges to current MLLM-as-a-judge approaches: the latest Gemini 3 Pro (Google DeepMind, 2025b) reaches 74-80% accuracy across all subtasks. GPT-5 (OpenAI, 2025b) and Gemini 2.5 Pro (Gemini Team, 2025) achieve only moderate performance (66-75% accuracy across all subtasks) compared to >90% for humans. The best open-source model, Qwen3-VL-32B (Qwen Team, 2025), achieves 55%-69% accuracy. Notably, GPT-4o (OpenAI, 2024), which is commonly used as an evaluator in existing benchmarks, attains only 51-65% accuracy, suggesting that it is no longer suitable for evaluating frontier multimodal models, especially on reasoning-heavy tasks.
- • We study task-specific metrics (e.g., VQAScore (Lin et al., 2024)) and reward models trained on human preferences (e.g., ImageReward (Xu et al., 2023), UnifiedReward (Wang et al., 2025b)), and find that they are no longer reliable on the challenging prompts and frontier models in MMRB2. For instance, VQAScore (with Qwen2.5-VL backbone) and ImageReward achieve 58.3% and 54.0% on text-to-image evaluation, respectively, well below MLLM-as-a-judge baselines such as Qwen3-VL-32B (64.1%) and Gemini 3 Pro (74.4%). While human preference training improves performance over heuristic metrics, these models still fall short of frontier MLLMs.
- • We show that performance on MMRB2 strongly correlates with performance on GenAI-Bench (Li et al., 2024), GEdit-Bench (Liu et al., 2025), ISGBench (Chen et al., 2024), and EMMA (Hao et al., 2025) when using different reward models for best-of-N selection, suggesting that MMRB2 is a good proxy for downstream effectiveness.
- • Further analysis of test-time scaling and fine-grained error patterns reveals substantial remaining headroom for omni model reward modeling and highlights concrete failure modes that future methods should address. Judges show notably higher agreement with human preferences on different-model pairs than on same-model pairs, with differences of up to 12%. Moreover, in multimodal reasoning tasks, judges exhibit a strong bias toward responses that include images, with performance gaps of 27.7–49.3% between pairs where annotators preferred image-containing responses and those where the preferred response contained only text.

Overall, MMRB2 establishes a challenging and informative benchmark that we hope will serve as a foundation for future research on omni model reward modeling, evaluation, and post-training.

### 2 Related Work

Reward modeling for visual generation. Building on RLHF, reward modeling has been extended beyond text. ImageReward (Xu et al., 2023), HPSv2 and v3 (Wu et al., 2023b; Ma et al., 2025), PickScore (Kirstain et al., 2023) learn human preferences for text-to-image generation, improving correlation with human judgments and guiding diffusion models beyond CLIP-based proxies. For image editing, EditScore (Luo et al., 2025) and EditReward (Wu et al., 2025) adopt similar preference-learning frameworks. Unified approaches aim for cross-task generalization: Wang et al. (2025b) train a single multimodal reward across image, video, and understanding tasks. Despite progress, most multimodal RMs remain task-specific and lack a unified, stress-testing evaluation.

Evaluating reward models. Benchmarking reward models has become an active research direction. In the text domain, RewardBench and RewardBench 2 (Lambert et al., 2025; Malik et al., 2025) systematically compare LLM reward functions across diverse axes (e.g., instruction following, reasoning, safety). VL-RewardBench (Li et al., 2025) and Multimodal RewardBench (Yasunaga et al., 2025) assess reward models for multimodal LLM. Llava-Critic series (Xiong et al., 2025; Wang et al., 2025a) focus on developing reward models for these reward benchmarks. EvalPlanner (Saha et al., 2025) and J1 (Whitehouse et al., 2025) further improve reward modeling by incentivizing test-time scaling in LLM-as-a-judge. However, existing benchmarks and judge training efforts still largely focus on image-text-to-text tasks. For image generation, researchers develop automatic evaluation metrics for text-to-image generation. CLIPScore (Hessel et al., 2021) offers a reference-free image–text similarity measure that correlates with human judgments but often misses compositional errors; TIFA (Hu et al., 2023), DSG (Cho et al., 2024), and VQAScore (Lin et al., 2024) address this by probing alignment via VQA, improving robustness on compositional cases. OmniVerifier (Zhang et al., 2025) further investigate on training better visual-outcome verification mdiirhtdkltrnlldrnvhbccugtflukgivodels. The human annotations collected in these works are often used as reward model evaluations. Most existing reward model evaluations focus either on text or text-to-image generation, offering little insight into interleaved text and image. To bridge this gap, Multimodal RewardBench 2 (MMRB2) provides a unified and challenging framework for assess reward modeling for omni models.

### 3 Multimodal RewardBench 2

MMRB2 (Figure 3) is a comprehensive omni reward model evaluation benchmark spanning a range of tasks (§3.1) of four types: text-to-image generation, image editing, interleaved generation, and multimodal reasoning. Each datapoint in MMRB2 contains a task prompt (§3.2) and two model responses, chosen and rejected (§3.3). Reward models are evaluated based on their agreement with human annotators (§3.4).

- 3.1 Tasks in MMRB2

- Task 1. Text-to-Image. Text-to-image generation provides natural language prompts for which generators produce candidate images. Reward models see the prompt and the candidate images, and must prefer the human-preferred image based on factors such as object composition, spatial relationships, attribute binding, text rendering, and adherence to complex multi-object instructions.
- Task 2. Image Editing. Image editing provides 1-3 input images and a textual edit instruction, along with candidate edited images from generators. Reward models must select the edit that best matches human preference, balancing faithfulness to the edit request with preservation of irrelevant regions. The edits include both single-image operations (e.g., changing attributes, scene modifications, adding/removing elements) and multi-image compositions where multiple inputs must be integrated.
- Task3. InterleavedGeneration. Interleaved generation provides multimodal prompts that elicit mixed image–text sequences from generators (e.g., for storytelling, how-to guides, educational content, or multi-step reasoning).

Interleaved Tasks:

- Multimodal Reasoning Tasks:
- • Multi-image reasoning
- • Multi-view reasoning
- • Spatial reasoning
- • Navigation
- • Correspondence tasks
- • Attentional focusing
- • IQ tests
- • Solving puzzles
- • Mazes
- • Mental reconstruction
- • Mental tracking

- • Visual storytelling
- • Explanation & "how-to"
- • Report generation
- • QA with image + text
- • Market material generation
- • 3D synthesis
- • Sequential image editing
- • Multi-concept composition
- • Scene decomposition
- • Temporal prediction …

Multimodal RewardBench 2

…

Image Editing Tasks:

- • Additions
- • Removals
- • Attribute edits
- • Action edits
- • Style change
- • Subject-driven generation
- • Text-heavy edits
- • Multiple-image composition
- • Reasoning (spatial, causal, temporal, logical)

Text-to-Image Tasks:

- • General objects
- • Text rendering
- • Anime & Stylization
- • Compositional generation
- • Knowledge-informed generation
- • Reasoning (commonsense, logical, mathematical, scientific, code to image)

…

…

- Figure 2 Breakdown of MMRB2 by task type and source, and detailed categories under each task.

Reward models are asked to rank candidate interleaved outputs, capturing human preferences for coherence, global planning, and effective coordination between visual and textual content.

Task 4. Multimodal Reasoning (Thinking with images). Multimodal reasoning provides complex problems that require visual understanding, logical inference, and multi-step problem solving. Generators may produce both text and intermediate thinking or sketchpad images; reward models must judge which candidate reasoning trajectory and final answer better aligns with human preference, emphasizing accurate perception, spatial reasoning, and clear explanation.

See Figure 1 for examples of multimodal preference pairs in MMRB2 across these four subtasks.

- 3.2 Prompt and response collection

For each task, we sample prompts from existing benchmarks via stratified sampling over difficulty and subtask type, using only test splits to avoid train–test leakage. We additionally design new, practical tasks (e.g., multi-image editing) that are not covered in prior benchmarks. Benchmarks are weighted by coverage and difficulty, yielding 1,000 prompts per task. For each prompt, we generate multiple responses from 7–11 state-of-the-art models, including both API and open-source systems. We observe that even strong models such as Gemini 2.5 Flash Image struggle on interleaved generation and multimodal reasoning; for such cases, we further construct agents that can call Python and image generation/editing tools (Hu et al., 2024). Table 1 summarizes prompt sources and candidate models, with additional details in Appendix C.

- 3.3 Human preference annotations Given prompts and responses, we developed methods to gather human preferences for each task type.

- 3.3.1 Image generation, editing & interleaved tasks

We adopt a unified annotation protocol to ensure consistency across text-to-image generation, image editing, and interleaved generation tasks.

Ensemblefiltering. To focus human annotation on the most informative comparisons, we first apply an ensemble filtering pipeline that removes easy preference pairs where one response is almost unanimously preferred. We collect judgments from nine multimodal judges spanning API models (GPT-5, GPT-4.1, GPT-4o, Gemini 2.5

Response Generation Filtering Human Annotation Final Benchmark Evals

- 1. Construct pairs
- 2. An ensemble of MLLM judges vote their preference.

Text-to-Image, Editing, Interleaved Generation

Task Prompt, Response A, Response B

[Figure 22]

Prompt: Remove the trees to clean up the image and make it easier to see the cows.

[Figure 23]

Q: Remove the trees to clean up the image and make it easier to see the cows.

[Figure 24]

###### Reward Model / Judge

Select their preference given a pair. Then remove samples with high disagreement.

High agreement pairs are ﬁltered out.

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

RMs

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

A) Gemini 2.5 B) GPT-image

Prediction: A > B Accuracy: 1.0

Multimodal reasoning

Construct pairs (correct reasoning + answer v.s. incorrect reasoning / answer)

Q: … A: … Reasoning: …

[Figure 36]

[Figure 37]

Q: Is the red balloon above or below the white balloon?

Ranking:

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

- (A)Above
- (B) Below

Prompt: …

Samples with incorrect answers are ﬁltered out.

[Figure 42]

[Figure 43]

- Response A: …
- Response B: … Better Response: …

Verify the reasoning of a correct response

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

- Figure 3 Overview of the MMRB2 data pipeline. The process combines ensemble MLLM judging, human verification, and multi-stage filtering to ensure high-quality, reasoning-consistent preference pairs across tasks.

Flash, Gemini 2.5 Pro) and open-source VL models (Gemma-3-27B/12B/4B, Qwen-2.5-VL-7B), covering a wide range of capability. Each judge evaluates every pair twice, once in forward order (A vs. B) and once in reverse order (B vs. A), to mitigate position bias (see Appendix C for the exact prompts).

We define easy pairs as those where the majority label appears in at least 90% of all judge evaluations across both orderings, and discard them because they provide little signal about fine-grained differences between reward models. While ensemble filtering can in principle introduce bias, the diversity of the judges and the high 90% threshold restrict filtering to near-trivial cases and mitigate systematic bias from any single model.

Human Preference Annotation. We employed professional annotators via the Surge AI platform to collect high-quality human preferences.1 Each pair is independently evaluated by three annotators who have no knowledge of which model generated each response. Annotators assess each response using a comprehensive evaluation framework with different criteria tailored to each task category. Finally for each pair, annotators provide their overall preference for answer A vs B on a 7-point Likert scale where we convert these ratings to preferences. See details in Appendix C. We implement several additional quality control measures. First, we filter out annotations with high inter-annotator disagreement, specifically removing pairs where the rating spread (maximum rating minus minimum rating) exceeds 4 points on the 7-point scale. We also exclude ambiguous annotations where the average rating falls too close to the scale midpoint (within the 3.0-4.0 range), as these indicate genuine uncertainty rather than clear preferences from human annotations. Finally, we remove pairs where the majority vote results in a tie, as these provide limited signal for evaluating judge agreement.

For the three generative tasks, we collected approximately 17,700 human preference judgments, each evaluated by three independent annotators, resulting in 5900 judgments overall. After filtering, we retain 1,000 pairs per task (approximately 50% of the initial set). Inter-annotator agreement on these filtered pairs is high: 95.6% overall (excluding ties), with task-specific rates of 95.3% for image generation, 96.3% for image editing, and 95.2% for interleaved generation.

- 3.3.2 Multimodal reasoning task

Because multimodal reasoning prompts have ground truth answers, we collect human annotations per model response (rather than pairwise) then construct pairs.

Human annotation. We filter generated model responses from §3.2 to those that contain both the correct answer to the prompt and some form of reasoning. We then balance samples across responses that include text-only reasoning and those that reason with both images and text. We collect three human annotations

1Annotators were compensated at an hourly rate of $85.

per response that indicate whether the reasoning contained in the model response is correct. The annotator instructions are listed in Appendix B.2.

Pair construction. With the annotated responses, we construct preference pairs. For the human-preferred sample of each pair, we select model responses in which all three human annotators agree that the reasoning contains no major errors and the model answer is correct. For the non-preferred sample of each pair, we utilize two kinds of responses: Correct answer, incorrect reasoning, where the model answer is correct but all three annotators consider the reasoning to contain major errors, and Incorrect answer, with reasoning, where the model answer is incorrect and some form of reasoning is included. For each pair, the two model responses may share the same modality (both text-only or both image+text) or be a combination. No model response is duplicated across pairs. For more details, see Appendix B.2.

- 3.4 Evaluation Method Finally, we use the preference pairs to evaluate reward models on MMRB2.

Positional consistent dual evaluation Position bias is common problem; models have a systematic preference for the first item in a pair (Min et al., 2022; Tan et al., 2025). To mitigate this, each pair is evaluated twice per judge: once in its original order (A vs. B) and once with responses swapped (B vs. A). Both forward and reverse judgments are retained as independent data points, doubling judge-human comparison instances. This protocol improves agreement statistics by increasing sample size and penalizes judges with high position bias.

Judge-HumanAgreementComputation We measure judge-human agreement by comparing each judge evaluation against the human preference for the corresponding pair. Human preference is determined by majority vote across three annotators for Tasks 1-3 and unanimous agreement of reasoning and answer correctness in Task

- 4. For each judge evaluation (whether forward or reverse), we compute a binary agreement score: 1.0 if the judge’s preference matches the human preference (including tie-to-tie matches), and 0.0 otherwise.

### 4 Experiments

We conduct a comprehensive study of omni reward modeling with MMRB2 along a number of dimensions: evaluation of MLLM-as-a-judge (§4.1), evaluation of other task-specific evaluators (§4.2), and in-depth analysis on various aspects of the benchmark and omni model reward modeling (§4.3 - 4.5).

- 4.1 Performance of MLLM-as-a-judge

Setup. We evaluate all tasks on API-based models GPT-4o, GPT-4.1, GPT-5, Gemini 2.5 Flash, Gemini 2.5 Pro, Gemini 3 Pro and open-source models Qwen 2.5-VL (7B and 72B) (Bai et al., 2025), Qwen 3-VL (8B, 32B, 30BA3B, 235BA22B) (Qwen Team, 2025) and Gemma 3 (4B, 12B, and 27B) (Gemma Team et al., 2025). For each task type, we design task-specific evaluation prompts with detailed rubrics (see Appendix C). We follow the positional consistent dual evaluation method in §3.4 to mitigate positional bias.

Results. Table 2 reveals substantial variation in judge-human agreement across models and tasks. APIbased models generally outperform open-source alternatives, with Gemini 3 Pro achieving the strongest overall performance across all tasks. GPT-5 and Gemini 2.5 Pro also achieves decent accuracy on text-to-image generation, image editing, and interleaved generation (70 - 75% accuracy). Notably, multimodal reasoning proves to be the most challenging task across all models except Gemini 3 Pro, with even top API models achieving only 52-70% agreement on reasoning tasks (compared to 63-75% on multimodal generation tasks). This difficulty may stem from multiple valid solution paths, varying levels of explanation detail that humans may value differently, or the challenge of assessing both correctness and reasoning quality simultaneously.

Gemma 3, Qwen2.5-VL, and Qwen3-VL families of models all performbetteronMMRB2asnumberofparameters scales. Additionally, the performance gap between API-based and open-source models has narrowed with recent open-source advances. The top API models (Gemini 3 Pro, Gemini 2.5 Pro, GPT-5) achieve agreement rates of 65-80% across most tasks, while the best open-source models now reach competitive performance levels. Qwen3-VL-32B is the strongest open-source model, achieving 64.1-70.5% across tasks. Notably, its 70.5% agreement rate for interleaved generation approaches API-based model performance. While the

Text to Image

Image Editing

Interleaved Generation

Multimodal Reasoning

Judge

Avg. Open-source multimodal LLMs

Gemma 3 4B (Gemma Team et al., 2025) 51.7 51.0 51.3 48.8 50.7 Gemma 3 12B (Gemma Team et al., 2025) 56.0 58.0 58.0 49.3 55.3 Gemma 3 27B (Gemma Team et al., 2025) 58.3 60.2 61.1 49.4 57.3 Qwen2.5-VL-7B (Bai et al., 2025) 50.4 57.1 48.4 47.5 50.9 Qwen2.5-VL-72B (Bai et al., 2025) 59.1 64.6 62.3 50.0 59.0 Qwen3-VL-8B (Qwen Team, 2025) 59.4 61.7 61.5 54.6 59.3 Qwen3-VL-32B (Qwen Team, 2025) 64.1 67.3 70.5 56.6 64.6 Qwen3-VL-30BA3B (Qwen Team, 2025) 60.0 59.5 57.3 57.3 58.5 Qwen3-VL-235BA22B (Qwen Team, 2025) 62.0 64.8 69.0 55.9 62.9

API-based Models

- GPT-4o (OpenAI, 2024) 60.3 65.0 61.5 51.9 59.7 GPT-4.1 (OpenAI, 2025a) 65.8 68.2 67.0 53.0 63.5
- GPT-5 (OpenAI, 2025b) 70.5 73.8 74.4 70.2 72.2 Gemini 2.5 Flash (Gemini Team, 2025) 63.1 66.5 69.4 57.5 64.1 Gemini 2.5 Pro (Gemini Team, 2025) 70.5 71.3 75.1 66.6 70.9 Gemini 3 Pro (Google DeepMind, 2025b) 74.4 74.9 76.4 79.5 76.3

- Table 2 MLLM-as-a-judge accuracies on MMRB2. The best numbers are bolded and the second best are underlined. Gemini 3 Pro is the best across all tasks. Qwen3-VL-32B is the best open-source model.

Qwen3-VL series generally outperforms the Gemma 3 and Qwen2.5 families on image-related tasks, even some of the Gemma 3 and Qwen2.5 variants are within a few percentage points of API-based models. However, open-source models still show large gaps with API-based models on multimodal reasoning: the strongest, Qwen3-VL 30BA3B at 57.3%, trails Gemini 3 Pro by 22 percentage points.

- 4.2 Performance of supervised reward models

Judge Text to Image Image Editing* Multimodal Reasoning* MLLM-as-a-judge

Qwen2.5-VL-7B (Bai et al., 2025) 50.4 57.8 53.7 Qwen3-VL-32B (Qwen Team, 2025) 64.1 66.4 69.9 GPT-5 (OpenAI, 2025b) 70.5 74.3 83.8

CLIP-based evaluators

CLIPScore (Hessel et al., 2021) 51.0 - ImageReward (Xu et al., 2023) 54.0 - HPSv2 (Wu et al., 2023a) 54.7 - PickScore (Kirstain et al., 2023) 58.6 - -

Qwen2.5-VL-7B-based evaluators

VQAScore (Lin et al., 2024) 58.3 - HPSv3 (Ma et al., 2025) 60.2 - EditReward (Wu et al., 2025) - 67.2 UnifiedReward (Wang et al., 2025b) 59.8 - 55.1

- Table3 Other evaluators’ accuracies on MMRB2. Note that all task-specific evaluators except CLIPScore and VQAScore have been trained with human preference pairs. *For editing we use the single-image subset; for reasoning we use the text-only-output subset, ensuring fair comparison among evaluators.

Besides directly prompting MLLMs to act as judges, prior work has proposed a range of automatic metrics and preference-trained reward models targeting the tasks in MMRB2. We evaluate these methods on the three MMRB2 tasks—text-to-image generation, image editing, and multimodal reasoning. To the best of our knowledge, there are currently no evaluators specifically tailored for interleaved text–image outputs.

Setup. Unless otherwise noted, we adopt the default metaprompt provided by each official library. For

text-to-image, we consider two families of evaluators. The first is CLIP-based (Radford et al., 2021), including CLIPScore (Hessel et al., 2021) and its preference-trained variants ImageReward (Xu et al., 2023), HPSv2 (Wu et al., 2023a), and PickScore (Kirstain et al., 2023). The second family is based on Qwen2.5-VL-7B (Bai et al., 2025). We evaluate VQAScore (Lin et al., 2024), which scores generated images using model logits, as well as the preference-trained reward models HPSv3 (Ma et al., 2025) and UnifiedReward (Wang et al., 2025b). We evaluate all of the above models on the MMRB2 text-to-image task. Qwen2.5-VL-7B has also been used as the backbone for reward models on other tasks, including EditReward (Wu et al., 2025) for image editing and UnifiedReward (Wang et al., 2025b) for multimodal understanding. Because EditReward is trained only on single-image editing, and UnifiedReward is trained on single-image image-to-text tasks, we evaluate them on the corresponding single-image subsets of MMRB2 to ensure a fair comparison among evaluators. Table 3 summarizes the results.

Preference training substantially improves reward-model accuracy. Several reward models share the same base architecture as our MLLM baselines (e.g., EditReward, UnifiedReward, and HPSv3 are based on Qwen2.5VL-7B), and some are CLIP-based (ImageReward, HPSv2, PickScore). Relative to the Qwen2.5-VL-7B judge, EditReward yields a +9.4% gain on editing (57.8 → 67.2), and UnifiedReward improves text-to-image by +9.4% (50.4 → 59.8) and reasoning by +1.4% (53.7 → 55.1). Similarly, compared to CLIPScore (51.0), CLIP-based preference models show consistent gains: ImageReward 54.0 (+3.0 %), HPSv2 54.7 (+3.7 %), and PickScore 58.6 (+7.6 %). These results indicate that training with human preferences is an effective way to boost evaluator performance on multimodal tasks.

Reward models can be out-of-distribution; strong MLLMs remain strong judges. Despite the above gains, most preference-trained reward models still underperform a larger open-source judge such as Qwen3-VL-32B across tasks; a notable exception is EditReward, which is competitive on editing (67.2 vs. 66.4). One plausible explanation is a distribution shift: several reward models were trained on data from earlier-generation systems (e.g., SD 2.1–era), and their accuracy diminishes when judging outputs from more capable, recent models. Overall, newer reward models (HPSv3, EditReward, UnifiedReward) are far better than older ones, yet stronger MLLM still set a high bar through simple prompting.

- 4.3 Correlation with downstream tasks

[Figure 48]

Figure 4 Downstream best-of-N score v.s. MMRB2 performance. We perform best-of-N sampling with 2 base models each on 4 tasks (GenAI-Bench (Li et al., 2024), GEdit-Bench (Liu et al., 2025), ISG-Bench (Chen et al., 2024), and EMMA (Hao et al., 2025)). A judge’s score on MMRB2 strongly correlates with improvement in downstream tasks when it is used in best-of-N sampling, highlighting MMRB2’s utility for downstream task success.

A key research question is whether MMRB2 performance can predict downstream task performance. To address this question, following prior works (Lightman et al., 2023; Li et al., 2025), we conduct best-of-N sampling with different rewards. We experiment with 4 tasks: GenAI-Bench (Li et al., 2024), GEdit-Bench (Liu et al., 2025), ISG-Bench (Chen et al., 2024), and EMMA (Hao et al., 2025), each corresponds to one task in MMRB2. For each query, we generate N = 8 candidate responses from two base models, one strong model and a weaker one, and we use 7 different MLLM-as-a-judge to select the best one via knockout tournaments. Then we evaluate the selected response with each task’s metrics.

- Figure 4 shows all the results. The x-axis is the MMRB2 performance, and Y-axis is the score of the best-of-N response selected by different rewards. We exclude GPT-4o and 4.1 because they are often used as evaluators. For interleaved generation, we remove ISGBench preference pairs when computing MMRB2 scores to avoid leakage. The results show that there is a strong correlation between best-of-N performance and MMRB2 scores. A good reward model can give great gains on downstream performance, even with the simple best-of-N sampling. For example, FLUX’s GenAI-Bench score improved from 73% to 79%, and GPT-4o’s accuracy on EMMA improved from 32% to 45%, when using GPT-5 as best-of-N selector. We can still see consistent gains even for strong base models like Gemini 2.5 Flash Image and o3-Sketchpad. The strong correlation (>0.8 Pearson’s r for all tasks and models) between MMRB2 and downstream task performance validates that MMRB2 is a good proxy for downstream effectiveness.

- 4.4 Fine-grained analysis of errors

Same-model pairs vs. different-model pairs. Our benchmark contains 57.4% same-model pairs (comparing two outputs from the same model) and 42.6% different-model pairs (comparing outputs from different models), allowing us to assess judge performance across both scenarios. See results in Table 4.

Task Judge Overall (%) Same-M (%) Diff-M (%)

Gemini 3 Pro 74.4 70.4 79.7 Gemini 2.5 Pro 70.5 68.4 73.4 GPT-5 70.5 66.8 75.6 GPT-4.1 65.8 61.6 71.4 Qwen3-VL-32B 64.1 59.1 71.0

Image Generation

Gemini 3 Pro 74.9 71.0 79.3 GPT-5 73.8 71.7 76.2 Gemini 2.5 Pro 71.3 66.7 76.6 GPT-4.1 68.2 65.6 71.3 Qwen3-VL-32B 67.3 64.5 70.5

Image Editing

Gemini 3 Pro 76.4 72.8 82.0 Gemini 2.5 Pro 75.1 70.7 81.9 GPT-5 74.4 69.4 82.1 Qwen3-VL-32B 70.5 66.7 76.4 Gemini 2.5 Flash 69.4 65.0 76.3

Interleaved

Gemini 3 Pro 79.5 78.7 79.8 GPT-5 70.2 68.4 70.8 Gemini 2.5 Pro 66.6 70.5 65.4 Gemini 2.5 Flash 57.5 59.9 56.7 Qwen3-VL-30BA3B 57.3 57.4 57.3

Reasoning

- Table 4 Detailed performance breakdown of top 5 judges per task showing overall agreement, same-model pairs, and different-model pairs with human preferences.

Across the image generation, editing, and interleaving tasks, we observe a consistent pattern for all judges: judges achieve higher agreement with human on different-model pairs compared to same-model pairs. For the best-performing judges, this gap ranges from 5-13 percentage points. For example, on image generation, Gemini 3 Pro achieves 79.7% agreement on different-model pairs but only 70.4% on same-model pairs (9.3 point gap). This pattern holds across tasks: same-model pairs demand fine-grained judgments within one model’s outputs, while different-model pairs reveal larger gaps rooted in capability differences.

##### Same-modality pairs vs mixed-modality pairs. For the multimodal reasoning task, we study how judges perform differentially for pairs constructed with responses from the same modality (e.g., text response vs. text response) versus mixed modalities (e.g., text response vs. text-image response). Full results are reported in Table 5.

Same modality: Image+text

Same modality: Text

Mixed modality: Pref: Image+text; Not Pref: Text

Mixed modality: Pref: Text; Not Pref: Image+text

Correct vs. incorrect reason

Correct vs. incorrect answer

Correct vs. incorrect reason

Correct vs. incorrect answer

Correct vs. incorrect reason

Correct vs. incorrect answer

Correct vs. incorrect reason

Correct vs. incorrect answer

Model

Open-source models

Gemma3 4B 47.39 50.63 50.00 48.32 63.64 57.00 38.68 36.00 Gemma3 12B 49.57 51.47 54.02 52.10 81.82 73.50 15.09 11.50 Gemma3 27B 51.30 50.21 51.79 51.68 87.50 79.50 10.38 10.50 Qwen2.5-VL-7B 49.12 48.10 51.34 50.00 52.27 39.00 47.17 40.31 Qwen2.5-VL-72B 52.63 48.10 53.57 54.41 78.41 68.00 16.04 23.98 Qwen3-VL-8B 57.46 52.53 58.48 54.20 71.59 73.00 34.91 36.73 Qwen3-VL-32B 62.28 54.43 60.71 56.93 78.41 80.00 25.47 33.16 Qwen3-VL-30BA3B 58.77 55.72 57.59 56.30 75.00 78.00 36.79 43.37 Qwen3-VL-235BA22B 58.11 57.02 55.80 57.14 85.23 81.96 23.58 26.02

API-based models

- GPT-4o 50.43 50.42 55.80 56.51 81.82 80.00 18.87 18.00 GPT-4.1 56.09 50.42 58.04 58.61 93.18 81.50 10.38 13.50
- GPT-5 69.57 67.02 75.89 80.25 88.64 88.00 36.79 40.00 Gemini 2.5 Flash 60.53 58.47 56.25 59.03 86.36 76.00 16.98 38.42 Gemini 2.5 Pro 73.91 66.18 62.95 65.55 84.09 79.00 43.40 58.00 Gemini 3 Pro 71.88 84.75 75.45 82.49 84.88 87.00 66.98 72.00

- Table 5 Multimodal reasoning performance breakdown by pair modality and pair type.

We find that for mixed-modal pairs, all judges exhibit a strong bias towards the response that contains images. This is true even of the highest performing models: the accuracy of GPT-5 for mixed-modal pairs when the preferred response contains an image is 49.3 points higher than pairs where the preferred response contains text (88.2% vs. 38.9%), and Gemini 2.5 Pro and Qwen3-VL-30BA3B have gaps of 27.7 and 36.0 points respectively. Gemini 3 Pro performs much better, but still has a 17.9 point gap. Additionally, we find that this trend holds for both pair types: those constructed with an incorrect response vs. a correct response and those with incorrect reasoning vs. correct reasoning.

- 4.5 Test-time scaling of rewards

[Figure 49]

- Figure 5 Majority-vote accuracy of each MLLM as the number of samples K varies. Test-time scaling yields small gains for GPT and Gemini models but no improvement for Qwen3-VL.

Prior work (Wang et al., 2022a; Brown et al., 2024) shows that test-time scaling can substantially improve LLM performance. We ask whether similar gains transfer to multimodal reward models. For each judge,

we draw K ∈ {1,3,5,7,9} independent judgments and take a majority vote as the final decision. We report majority-vote accuracy averaged over the four MMRB2 tasks (300 examples per task) in Fig. 5.

The effects are model-dependent, echoing trends in prior observations (Li et al., 2025). Qwen3-VL models show no measurable improvement as K increases. In contrast, GPT-4o, Gemini 2.5 Flash, GPT-5, and Gemini 2.5 Pro improve by roughly 0.8–1.2% at K=9, with Gemini 2.5 Pro showing the largest gain (from 71.3% to 72.5%). Overall, test-time scaling provides only modest returns for multimodal reward models compared with text-only LLMs, suggesting that alternative scaling methods are needed for multimodal rewards.

- 5 Conclusion

We introduce Multimodal RewardBench 2, the first comprehensive benchmark for omni reward models spanning four tasks: text-to-image, image editing, interleaved generation, and multimodal reasoning. Our analysis suggests that current omni reward models, particularly the latest Gemini 3 Pro, can serve as proxies for human evaluation on multimodal generation tasks, achieving 74-80% agreement. However, the substantial disagreement remaining (20-26%) indicates that human evaluation remains essential, , and that other models, including GPT-5, lag significantly behind Gemini 3 Pro. Overall, MMRB2 establishes a challenging and informative benchmark that we hope will serve as a foundation for future research on omni model reward modeling, evaluation, and post-training.

Limitations and future extensions. As the first comprehensive benchmark targeting omni reward models, MMRB2 focuses on core settings and overall human preferences in interleaved text–image scenarios. The construction pipeline is modular and can be extended to additional evaluation dimensions (e.g., safety- and bias-sensitive preferences), richer task formats (e.g., multilingual tasks, in-the-wild prompts, multi-turn and agentic interactions), and further modalities (e.g., video and audio). Further discussion is provided in Appendix D.

- 6 Acknowledgements

We would like to thank Mason Yu, Christophe Ropers, Nate Ekberg, Cynthia Gao, Justin Hovey, Jaimie Hsu, Samantha Snowden, and all annotators from Surge AI for their invaluable contributions to data annotation. We also thank Jonea Gordon and Vanessa Stark for their assistance with the approval process. Additionally, we are grateful to Mary Williamson, Xiaochuang Han, Adriana Romero Soriano, Michal Drozdzal, Xudong Wang, Michihiro Yasunaga, Ishan Misra, Amita Kamath, Inna Lin, and Karen Chen for their insightful discussions and support throughout this project.

References

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Jason Baldridge, Jakob Bauer, Mukul Bhutani, Nicole Brichtova, Andrew Bunner, et al. Imagen 3. arXiv preprint arXiv:2408.07009, 2024. https://arxiv.org/abs/2408.07009.

Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, Christopher Ré, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787, 2024.

Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024. doi: 10.48550/arXiv.2405.09818.

Jingjing Chang, Yixiao Fang, Peng Xing, Shuhan Wu, Wei Cheng, Rui Wang, Xianfang Zeng, Gang Yu, and Hai-Bao Chen. Oneig-bench: Omni-dimensional nuanced evaluation for image generation. arXiv preprint arxiv:2506.07977, 2025.

Dongping Chen, Ruoxi Chen, Shu Pu, Zhaoyi Liu, Yanru Wu, Caixi Chen, Benlin Liu, Yue Huang, Yao Wan, Pan Zhou, et al. Interleaved scene graphs for interleaved text-and-image generation assessment. arXiv preprint arXiv:2411.17188, 2024.

Jiuhai Chen, Zhiyang Xu, Xichen Pan, Yushi Hu, Can Qin, Tom Goldstein, Lifu Huang, Tianyi Zhou, Saining Xie, Silvio Savarese, Le Xue, Caiming Xiong, and Ran Xu. Blip3-o: A family of fully open unified multimodal models-architecture, training and dataset, 2025a. https://arxiv.org/abs/2505.09568.

Kaijie Chen, Zihao Lin, Zhiyang Xu, Ying Shen, Yuguang Yao, Joy Rimchala, Jiaxin Zhang, and Lifu Huang. R2I-bench: Benchmarking reasoning-driven text-to-image generation. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Proceedings of the 2025 Conference on Empirical Methods in Natural

Language Processing, pages 12606–12641, Suzhou, China, November 2025b. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.636. https://aclanthology.org/2025.emnlp-main.636/.

Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025c. doi: 10.48550/arXiv.2501.17811.

Jaemin Cho, Yushi Hu, Roopal Garg, Peter Anderson, Ranjay Krishna, Jason Baldridge, Mohit Bansal, Jordi PontTuset, and Su Wang. Davidsonian Scene Graph: Improving Reliability in Fine-Grained Evaluation for Text-to-Image Generation. In ICLR, 2024.

Paul F. Christiano, Jan Leike, Tom B. Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems, volume 30, 2017.

Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, Guang Shi, and Haoqi Fan. Emerging properties in unified multimodal pretraining. CoRR, abs/2505.14683, May 2025. https://arxiv.org/abs/2505.14683.

Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A Smith, Wei-Chiu Ma, and Ranjay Krishna. Blink: Multimodal large language models can see but not perceive. arXiv preprint arXiv:2404.12390, 2024.

Yuying Ge, Sijie Zhao, Jinguo Zhu, Yixiao Ge, Kun Yi, Lin Song, Chen Li, Xiaohan Ding, and Ying Shan. Seed-x: Multimodal models with unified multi-granularity comprehension and generation, 2025. https://arxiv.org/abs/2404. 14396.

Gemini Team. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities, July 2025. https://arxiv.org/abs/2507.06261.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.

- Google DeepMind. Gemini 2.5 flash & 2.5 flash image: Model card. https://storage.googleapis.com/deepmind-media/ Model-Cards/Gemini-2-5-Flash-Model-Card.pdf, September 2025a.
- Google DeepMind. Gemini 3 pro model card. Technical report, Google DeepMind, November 2025b. https: //storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Pro-Model-Card.pdf. Accessed: 2025-11-24.

Google DeepMind. Gemini 2.0 flash image: Native image generation with gemini 2.0 flash. https://developers. googleblog.com/experiment-with-gemini-20-flash-native-image-generation/, 2025c.

Google DeepMind. Google gemini imagen 4: AI image generation. https://gemini.google/overview/image-generation/, 2025d.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Shuhao Han, Haotian Fan, Jiachen Fu, Liang Li, Tao Li, Junhui Cui, Yunqiu Wang, Yang Tai, Jingwei Sun, Chunle Guo, and Chongyi Li. Evalmuse-40k: A reliable and fine-grained benchmark with comprehensive human annotations for text-to-image generation model evaluation, 2024. https://arxiv.org/abs/2412.18150.

Yunzhuo Hao, Jiawei Gu, Huichen Will Wang, Linjie Li, Zhengyuan Yang, Lijuan Wang, and Yu Cheng. Can mllms reason in multimodality? emma: An enhanced multimodal reasoning benchmark. arXiv preprint arXiv:2501.05444, 2025.

Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 conference on empirical methods in natural language processing, pages 7514–7528, 2021.

Yushi Hu, Benlin Liu, Jungo Kasai, Yizhong Wang, Mari Ostendorf, Ranjay Krishna, and Noah A Smith. Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering. arXiv preprint arXiv:2303.11897, 2023.

Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. arXiv preprint arXiv:2406.09403, 2024.

Mude Hui, Siwei Yang, Bingchen Zhao, Yichun Shi, Heng Wang, Peng Wang, Yuyin Zhou, and Cihang Xie. Hq-edit: A high-quality dataset for instruction-based image editing, 2024. https://arxiv.org/abs/2404.09990.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. In Advances in Neural Information Processing Systems (NeurIPS), 2023.

Black Forest Labs, Stephen Batifol, Andreas Blattmann, Frederic Boesel, Saksham Consul, Cyril Diagne, Tim Dockhorn, Jack English, Zion English, Patrick Esser, Sumith Kulal, Kyle Lacey, Yam Levi, Cheng Li, Dominik Lorenz, Jonas Müller, Dustin Podell, Robin Rombach, Harry Saini, Axel Sauer, and Luke Smith. FLUX.1 kontext: Flow matching for in-context image generation and editing in latent space, 2025. https://arxiv.org/abs/2506.15742.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, et al. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.

Nathan Lambert, Valentina Pyatkin, Jacob Morrison, Lester James Validad Miranda, Bill Yuchen Lin, Khyathi Chandu, Nouha Dziri, Sachin Kumar, Tom Zick, Yejin Choi, et al. Rewardbench: Evaluating reward models for language modeling. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 1755–1797, 2025.

Baiqi Li, Zhiqiu Lin, Deepak Pathak, Jiayao Li, Yixin Fei, Kewen Wu, Tiffany Ling, Xide Xia, Pengchuan Zhang, Graham Neubig, et al. Genai-bench: Evaluating and improving compositional text-to-visual generation. arXiv preprint arXiv:2406.13743, 2024.

Lei Li, Yuancheng Wei, Zhihui Xie, Xuqing Yang, Yifan Song, Peiyi Wang, Chenxin An, Tianyu Liu, Sujian Li, Bill Yuchen Lin, et al. Vl-rewardbench: A challenging benchmark for vision-language generative reward models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24657–24668, 2025.

Youwei Liang, Junfeng He, Gang Li, Peizhao Li, Arseniy Klimovskiy, Nicholas Carolan, Jiao Sun, Jordi Pont-Tuset, Sarah Young, Feng Yang, Junjie Ke, Krishnamurthy Dj Dvijotham, Katie Collins, Yiwen Luo, Yang Li, Kai J Kohlhoff, Deepak Ramachandran, and Vidhya Navalpakkam. Rich human feedback for text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, 2023.

Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with image-to-text generation. arXiv preprint arXiv:2404.01291, 2024.

Minqian Liu, Zhiyang Xu, Zihao Lin, Trevor Ashby, Joy Rimchala, Jiaxin Zhang, and Lifu Huang. Holistic evaluation for interleaved text-and-image generation, 2024. https://arxiv.org/abs/2406.14643.

Shiyu Liu, Yucheng Han, Peng Xing, Fukun Yin, Rui Wang, Wei Cheng, Jiaqi Liao, Yingming Wang, Honghao Fu, Chunrui Han, et al. Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761, 2025.

Xin Luo, Jiahao Wang, Chenyuan Wu, Shitao Xiao, Xiyan Jiang, Defu Lian, Jiajun Zhang, Dong Liu, and Zheng Liu. Editscore: Unlocking online rl for image editing via high-fidelity reward modeling. arXiv preprint arXiv:2509.23909, 2025.

Yuhang Ma, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. Hpsv3: Towards wide-spectrum human preference score. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.

Saumya Malik, Valentina Pyatkin, Sander Land, Jacob Morrison, Noah A Smith, Hannaneh Hajishirzi, and Nathan Lambert. Rewardbench 2: Advancing reward model evaluation. arXiv preprint arXiv:2506.01937, 2025.

Sewon Min, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Noisy channel language model prompting for few-shot text classification. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Proceedings of

the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5316– 5330, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.365. https://aclanthology.org/2022.acl-long.365/.

Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Kunpeng Ning, Chaoran Feng, Bin Zhu, and Li Yuan. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.

- OpenAI. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

OpenAI. Introducing gpt-4.1 in the api. https://openai.com/index/gpt-4-1/, April 2025a. Blog post (no standalone technical report/system card published as of this date).

- OpenAI. Gpt-5 system card. https://cdn.openai.com/gpt-5-system-card.pdf, August 2025b. Version dated Aug 13, 2025.

OpenAI. Gpt-image 1 (gpt-image-1) model card. https://platform.openai.com/docs/models/gpt-image-1, April

2025c. API documentation and model card for GPT-Image 1, an image-generation model introduced April 2025. OpenAI. Openai o3 and o4-mini system card. https://cdn.openai.com/pdf/2221c875-02dc-4789-800b-e7758f3722c1/

o3-and-o4-mini-system-card.pdf, April 2025d.

Yuang Peng, Yuxin Cui, Haomiao Tang, Zekun Qi, Runpei Dong, Jing Bai, Chunrui Han, Zheng Ge, Xiangyu Zhang, and Shu-Tao Xia. Dreambench++: A human-aligned benchmark for personalized image generation. In The Thirteenth International Conference on Learning Representations, 2025. https://dreambenchplus.github.io/.

Qwen Team. Qwen3-vl: Sharper vision, deeper thought, broader action. Qwen Blog. Accessed, pages 10–04, 2025. Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda

Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR, 2021.

Swarnadeep Saha, Xian Li, Marjan Ghazvininejad, Jason Weston, and Tianlu Wang. Learning to plan & reason for evaluation with thinking-llm-as-a-judge. 2025. doi: 10.48550/arXiv.2501.18099. https://arxiv.org/abs/2501.18099.

Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. Emu edit: Precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8871–8879, 2024.

Yang Shi, Yuhao Dong, Yue Ding, Yuran Wang, Xuanyu Zhu, Sheng Zhou, Wenting Liu, Haochen Tian, Rundong Wang, Huanqian Wang, Zuyan Liu, Bohan Zeng, Ruizhe Chen, Qixun Wang, Zhuoran Zhang, Xinlong Chen, Chengzhuo Tong, Bozhou Li, Chaoyou Fu, Qiang Liu, Haotian Wang, Wenjing Yang, Yuanxing Zhang, Pengfei Wan, Yi-Fan Zhang, and Ziwei Liu. Realunify: Do unified models truly benefit from unification? a comprehensive benchmark, 2025. https://arxiv.org/abs/2509.24897.

Stability AI. Introducing stable diffusion 3.5. https://stability.ai/news/introducing-stable-diffusion-3-5, 2024. Sijun Tan, Siyuan Zhuang, Kyle Montgomery, William Y. Tang, Alejandro Cuadron, Chenguang Wang, Raluca Ada

Popa, and Ion Stoica. Judgebench: A benchmark for evaluating llm-based judges, 2025. https://arxiv.org/abs/ 2410.12784.

Fei Wang, Xingyu Fu, James Y Huang, Zekun Li, Qin Liu, Xiaogeng Liu, Mingyu Derek Ma, Nan Xu, Wenxuan Zhou, Kai Zhang, et al. Muirbench: A comprehensive benchmark for robust multi-image understanding. arXiv preprint arXiv:2406.09411, 2024a.

Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, Yingli Zhao, Yulong Ao, Xuebin Min, Tao Li, Boya Wu, Bo Zhao, Bowen Zhang, Liangdong Wang, Guang Liu, Zheqi He, Xi Yang, Jingjing Liu, Yonghua Lin, Tiejun Huang, and Zhongyuan Wang. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024b. doi: 10.48550/arXiv.2409.18869.

Xiyao Wang, Chunyuan Li, Jianwei Yang, Kai Zhang, Bo Liu, Tianyi Xiong, and Furong Huang. Llava-critic-r1: Your critic model is secretly a strong policy model. arXiv preprint arXiv:2509.00676, 2025a.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022a.

Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025b.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions, 2022b.

Chenxi Whitehouse, Tianlu Wang, Ping Yu, Xian Li, Jason Weston, Ilia Kulikov, and Swarnadeep Saha. J1: Incentivizing thinking in llm-as-a-judge via reinforcement learning. 2025. doi: 10.48550/arXiv.2505.10320. https: //arxiv.org/abs/2505.10320.

Keming Wu, Sicong Jiang, Max Ku, Ping Nie, Minghao Liu, and Wenhu Chen. Editreward: A human-aligned reward model for instruction-guided image editing. arXiv preprint arXiv:2509.26346, 2025.

Penghao Wu and Saining Xie. V*: Guided visual search as a core mechanism in multimodal llms. arXiv preprint arXiv:2312.14135, 2023.

Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023a.

Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score: Better aligning text-toimage models with human preference. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2096–2105, 2023b.

Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj Ammanabrolu, Noah A Smith, Mari Ostendorf, and Hannaneh Hajishirzi. Fine-grained human feedback gives better rewards for language model training. arXiv preprint arXiv:2306.01693, 2023c.

Tianyi Xiong, Xiyao Wang, Dong Guo, Qinghao Ye, Haoqi Fan, Quanquan Gu, Heng Huang, and Chunyuan Li. Llava-critic: Learning to evaluate multimodal models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13618–13628, 2025.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.

Weiye Xu, Jiahao Wang, Weiyun Wang, Zhe Chen, Wengang Zhou, Aijun Yang, Lewei Lu, Houqiang Li, Xiaohua Wang, Xizhou Zhu, Wenhai Wang, Jifeng Dai, and Jinguo Zhu. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models. arXiv preprint arXiv:2504.15279, 2025. https://arxiv.org/abs/2504.15279.

Jihan Yao, Yushi Hu, Yujie Yi, Bin Han, Shangbin Feng, Guang Yang, Bingbing Wen, Ranjay Krishna, Lucy Lu Wang, Yulia Tsvetkov, Noah A. Smith, and Banghua Zhu. Mmmg: a comprehensive and reliable evaluation suite for multitask multimodal generation, 2025. https://arxiv.org/abs/2505.17613.

Michihiro Yasunaga, Luke Zettlemoyer, and Marjan Ghazvininejad. Multimodal rewardbench: Holistic evaluation of reward models for vision language models. arXiv preprint arXiv:2502.14191, 2025.

Baiqiao Yin, Qineng Wang, Pingyue Zhang, Jianshu Zhang, Kangrui Wang, Zihan Wang, Jieyu Zhang, Keshigeyan Chandrasegaran, Han Liu, Ranjay Krishna, et al. Spatial mental modeling from limited views. In Structural Priors for Vision Workshop at ICCV’25, 2025.

Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, and Jason E Weston. Self-rewarding language models. In Forty-first International Conference on Machine Learning, 2024.

Xinchen Zhang, Xiaoying Zhang, Youbin Wu, Yanbin Cao, Renrui Zhang, Ruihang Chu, Ling Yang, and Yujiu Yang. Generative universal verifier as multimodal meta-reasoner. arXiv preprint arXiv:2510.13804, 2025.

Xiangyu Zhao, Peiyuan Zhang, Kexian Tang, Xiaorong Zhu, Hao Li, Wenhao Chai, Zicheng Zhang, Renqiu Xia, Guangtao Zhai, Junchi Yan, et al. Envisioning beyond the pixels: Benchmarking reasoning-informed visual editing. arXiv preprint arXiv:2504.02826, 2025.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-bench and chatbot arena. In Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track, 2023.

Chunting Zhou, Lili Yu, Arun Babu, Kushal Tirumala, Michihiro Yasunaga, Leonid Shamis, Jacob Kahn, Xuezhe Ma, Luke Zettlemoyer, and Omer Levy. Transfusion: Predict the next token and diffuse images with one multi-modal model. arXiv preprint arXiv:2408.11039, 2024. doi: 10.48550/arXiv.2408.11039.

## Appendix

Contents of Supplementary Material

Additional experimental results ......................................................................... 18 Details of annotation and pair construction ............................................................. 20 Additional details for prompts, response generation, and MLLM-as-a-judge .............................. 26 Limitations and Future Directions.......................................................................37 Examples ............................................................................................... 38

- A Additional Experimental Results

- A.1 Performance by task source and pair type

The pairwise evaluation results presented in Tables 6, 7, 8, 9 distinct performance patterns across multimodal tasks and model capabilities.

ImageGeneration: Performance varies moderately across benchmarks, with realunify (48-78.5%) and oneigbench (73-74% for top models) showing higher judge agreement rates, while wise consistently yields the lowest scores across all models (47-66%).

Image Editing: The breakdown reveals stark differences in benchmark difficulty, with text-based editing benchmarks (text: 54-83%, risebench: 48-83%) showing significantly higher agreement rates compared to general editing tasks (emu-edit: 49-72%, multi-image editing: 51-71%). This pattern holds consistently across all judge models, indicating that text rendering or text-focused editing provides clearer discriminative signals for pairwise evaluation than open-ended creative edits.

Interleaved: Performance is relatively uniform across benchmarks for top models, with isgbench consistently scoring highest (76-79% for frontier models) and all benchmarks clustering within a 5-8 percentage point range for leading judges.

Reasoning: This task exhibits the most dramatic benchmark-level variance, with muirbench showing substantially higher scores (36-76%) compared to other benchmarks, while vstar proves exceptionally challenging (30-52%). The performance on blink and mindcube clusters tightly (44-55% for most models), suggesting these represent a baseline reasoning difficulty.

Judge Model Overall % evalmuse oneigbench r2ibench realunify wise (n=390) (n=278) (n=128) (n=93) (n=111)

Gemini 3 Pro 74.4% 74.5% 74.5% 72.7% 80.1% 71.2% Gemini 2.5 Pro 70.5% 69.6% 73.0% 70.7% 77.4% 61.7% GPT5 70.5% 67.3% 73.2% 72.7% 78.5% 66.2% GPT 4.1 65.8% 62.4% 69.6% 66.8% 75.8% 58.6% Qwen-3 vl32b 64.1% 63.3% 68.3% 60.9% 65.6% 59.0% Gemini25flash 63.1% 60.3% 64.4% 62.1% 75.3% 60.8% Qwen-3 vl235ba22b 62.0% 59.2% 65.5% 60.2% 68.8% 59.0% GPT 4o 60.3% 58.1% 65.5% 59.4% 65.1% 52.3% Qwen-3 vl30ba3b 60.0% 57.6% 64.2% 57.8% 59.1% 61.3% Qwen-3 vl8b 59.4% 59.2% 62.8% 57.4% 62.4% 50.9% Qwen25vl72b 59.1% 56.8% 62.1% 56.2% 67.7% 55.9% Gemma 3-27b 58.3% 56.7% 59.5% 60.2% 66.7% 51.8% Llama4-17b 56.7% 56.3% 58.1% 53.4% 58.0% 58.1% Gemma 3-12b 56.0% 54.6% 58.6% 53.1% 62.9% 52.3% Gemma 3-4b 51.7% 50.1% 53.2% 50.8% 54.3% 52.7% Qwen25vl7b 50.4% 48.8% 55.9% 47.7% 48.4% 46.8%

###### Table 6 Image Generation: Pairwise model evaluation breakdown by benchmark.

Judge Model Overall % dreambench emu-edit hq-edit multi-image editing risebench text (n=242) (n=329) (n=53) (n=178) (n=84) (n=114)

Gemini 3 Pro 74.9% 70.0% 75.8% 79.2% 68.5% 82.1% 84.6% GPT5 73.8% 71.3% 71.6% 72.6% 71.1% 82.7% 83.3% Gemini 2.5 Pro 71.3% 70.2% 68.4% 76.0% 64.9% 79.8% 83.3% GPT 4.1 68.2% 67.6% 63.4% 68.9% 66.0% 74.4% 82.0% Qwen-3 vl32b 67.3% 63.6% 65.8% 72.6% 64.9% 71.4% 77.6% Gemini25flash 66.5% 63.8% 67.6% 67.9% 59.6% 72.6% 74.6% Qwen-3 vl235ba22b 66.0% 66.7% 64.1% 67.0% 57.9% 72.0% 77.6% GPT 4o 65.0% 66.5% 60.6% 60.4% 64.6% 66.1% 76.3% Qwen25vl72b 64.6% 63.6% 63.5% 61.3% 60.1% 63.7% 79.4% Qwen-3 vl8b 61.7% 59.3% 61.9% 61.3% 54.8% 62.5% 76.3% Llama4-17b 61.1% 60.2% 59.8% 65.3% 57.3% 64.4% 67.9% Gemma 3-27b 60.2% 61.2% 60.2% 58.5% 57.6% 53.0% 68.4% Qwen-3 vl30ba3b 59.5% 58.1% 58.5% 59.4% 56.5% 61.9% 68.4% Gemma 3-12b 58.0% 58.9% 55.0% 56.6% 57.6% 60.7% 64.0% Qwen25vl7b 57.1% 59.5% 53.8% 57.5% 53.7% 52.4% 70.2% Gemma 3-4b 51.0% 53.3% 49.1% 49.1% 51.4% 47.6% 54.4%

- Table 7 Image Editing: Pairwise model evaluation breakdown by benchmark.

Judge Model Overall % chameleon interleavedeval isgbench mmmg (n=284) (n=267) (n=421) (n=28)

Gemini 3 Pro 76.4% 76.4% 76.8% 76.1% 76.8% Gemini 2.5 Pro 75.1% 73.4% 71.5% 78.5% 75.0% GPT5 74.4% 72.9% 72.8% 76.5% 71.4% Qwen-3 vl32b 70.5% 66.9% 70.4% 73.3% 66.1% Gemini25flash 69.4% 64.4% 70.8% 71.5% 75.0% GPT 4.1 67.0% 65.3% 66.3% 69.1% 60.7% Qwen-3 vl235ba22b 66.7% 63.5% 66.3% 68.9% 69.6% Qwen25vl72b 62.3% 59.9% 61.4% 64.7% 58.9% GPT 4o 61.5% 60.9% 60.1% 63.3% 53.6% Qwen-3 vl8b 61.5% 59.3% 63.7% 61.3% 66.1% Gemma 3-27b 61.1% 59.9% 59.4% 62.4% 69.6% Gemma 3-12b 58.0% 57.6% 58.1% 58.2% 58.9% Qwen-3 vl30ba3b 57.3% 55.8% 56.8% 57.7% 69.6% Llama4-scout-17b 54.4% 55.7% 54.9% 52.5% 66.7% Gemma 3-4b 51.3% 50.4% 52.8% 51.1% 50.0% Qwen25vl7b 48.4% 48.4% 46.4% 49.8% 48.2%

- Table 8 Interleaved: Pairwise model evaluation breakdown by benchmark.

- A.2 Win rate analysis on generations

We also report the generation capabilities of MLLMs as content producers (models generating the multimodal content being evaluated, reported in Table 10). Judging requires discriminative understanding and alignment with human preferences, while generation requires creative synthesis and technical execution. A model may excel at one role while underperforming at the other, as we observe in our results.

Table 10 presents the win rates of generative models across MMRB2’s three generation tasks (Tasks 1–3), where win rate is computed as (wins + 0.5 × ties)/total comparisons based on human majority preferences. These are the same model outputs that judges evaluate in Table 2, allowing us to assess both generation quality and judgment accuracy within a unified framework.

Image Generation. GPT-Image-1 (60.4%) narrowly leads text-to-image generation, closely followed by Imagen

- 4 (57.4%), Imagen 4 Ultra (56.5%), and Gemini 2.5 Flash (54.3%), indicating a highly competitive landscape among top proprietary models with less than 6 points separating the leaders. Open-source models lag substantially: Stable Diffusion 3.5 Large (41.0%) and FLUX (36.8%) trail by 19–24 points.

Image Editing. Interestingly, general-purpose multimodal models such as Gemini 2.5 Flash (59.2%) and GPTImage-1 (53.2%) outperform specialized models. While Imagen Edit achieves only a 35.2% win rate despite

Judge Model Overall % blink mindcube muirbench realunify visulogic vstar (n=355) (n=367) (n=137) (n=55) (n=49) (n=37)

Qwen-3 vl32b 56.6% 52.4% 55.6% 71.5% 56.4% 61.2% 45.9% Qwen-3 vl30ba3b 56.5% 54.6% 52.9% 70.3% 56.9% 62.1% 51.1% Qwen-3 vl235ba22b 55.9% 52.6% 54.2% 76.1% 52.8% 59.2% 30.6% Qwen-3 vl8b 53.7% 51.9% 50.9% 64.2% 54.4% 60.4% 48.5% Qwen25vl72b 50.2% 46.7% 52.0% 57.7% 50.9% 54.1% 29.7% Llama4-scout-17b 44.5% 43.5% 47.7% 35.8% 49.1% 44.9% 48.6%

- Table 9 Reasoning: Pairwise model evaluation breakdown by benchmark.

being purpose-built for editing, the gap is less severe than earlier reports suggested. FLUX-Kontext (49.0%) demonstrates competitive performance for an open-source solution, though it still trails the leaders. These results suggest that strong vision–language understanding provides significant advantages for instruction-based editing, even if specialized architectures are not entirely obsolete.

Interleaved Generation. Agent-based systems dominate, with GPT-Gemini Agent (57.1%) and GPT-Image Agent (56.9%) leading by narrow margins. Native multimodal models like Gemini 2.5 Flash (53.2%) perform competitively, narrowing the gap with agent architectures. GPT-FLUX Agent’s improved but still modest performance (40.4%) confirms that agent quality depends critically on component model quality, though the improvement suggests that better integration strategies can help.

Rank Task Model Win Rate (%)

- 1
- 2
- 3
- 4
- 5
- 6
- 7
- 8

Image Gen.

GPT-Image-1 60.4 Imagen 4 57.4 Imagen 4 Ultra 56.5 Gemini 2.5 Flash 54.3 Imagen 3 49.2 Gemini 2.0 Flash 45.6 SD 3.5 Large 41.0 FLUX 36.8

- 1
- 2
- 3
- 4
- 5

Image Editing

Gemini 2.5 Flash 59.2 GPT-Image-1 53.2 FLUX-Kontext 49.0 Gemini 2.0 Flash 47.1 Imagen Edit 35.2

- 1
- 2
- 3
- 4
- 5
- 6

Interleaved

GPT-Gemini Agent 57.1 GPT-Image Agent 56.9 Gemini 2.5 Flash 53.2 Gemini 2.0 Flash 46.2 GPT-Imagen Agent 42.1 GPT-FLUX Agent 40.4

- Table 10 Model win rates (%) on Multimodal RewardBench 2 ranked by performance within each task. Win rate is computed as (wins + 0.5 × ties) / total comparisons.

### B Details for Annotation and Pair Construction

- B.1 Tasks 1-3

- Figure 6 shows a sample of the annotation interface for the MMRB2 text-to-image task. In this section we provide additional details on the human annotation procedure. For each annotation task, we provide a prompt and two responses, A and B, and the goal is to assess the

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

###### Figure 6 Annotation interface for the MMRB2 text-to-image task. Note that for image editing and interleaved tasks, there are more fine-grained questions.

quality of each response and then rate them. Annotators answer the following questions:

- • Prompt Quality Check: Indicate whether the prompt is correct (Yes/No).
- • Pointwise Evaluation for Response A and Response B: For each response, rate the following dimensions on a 4-point scale (see Section B.1.1 for details):

- – Faithfulness to the text instruction
- – (Tasks 2 and 3) Faithfulness to the input image
- – Overall quality of the generated image
- – (Task 3 only) Cross-generation image congruence
- – (Task 3 only) Generated text-image alignment
- – (Task 3 only) Technical quality of generated text
- – (Conditional) Correctness of text rendering

- • Rationales: Provide a brief rationale for the overall quality rating of both Response A and Response B.
- • Overall Preference: Indicate your overall preference between Response A and Response B, choosing one of the following:

- – A is significantly better
- – A is marginally better
- – Unsure or A is negligibly better
- – Unsure or B is negligibly better
- – B is marginally better
- – B is significantly better

- • Rationale for Preference: Provide a brief explanation for your overall preference.

- B.1.1 Details of each question

##### 1. (For all tasks) Faithfulness to the text instruction: How accurately and completely does the output follow the explicit and implicit text instructions in the prompt?

Rating Label Description

- 0 Major deviations

Key elements are missing, altered, or contradicted

- 1 Some mismatch Some key elements are missing or altered.
- 2 Minor mismatch

Most key elements are present, but others are missing, incorrect, or incomplete

- 3 Full match

All key elements are represented exactly as described, with no significant omissions or contradictions

##### 2. (For task 2 and 3) Faithfulness to the input image: When using an input image as context (e.g., editing, continuation, transformation), how well does the output incorporate the relevant elements of the input according to the instructions?

- 0 Fails to use the input meaningfully

Key elements are ignored, misinterpreted, or contradicted

- 1 Partial mismatch to the input

Some elements are carried over or transformed correctly, but those are not key elements or important aspects

- 2 Minor mismatch to the input

Most relevant elements are carried over or transformed correctly, but a few aspects are missing or incorrectly handled

- 3 Uses input fully

All relevant elements from the input are accurately incorporated, extended, or transformed exactly as instructed

- 3. (For all tasks) Overall quality of generated image: Does the image contain significant technical errors that break composition (including style coherence and realism) or make it visually unappealing? For example, issues with impossible geometry, strange objects, garbled text, incorrect human anatomy (limbs bending the wrong way, wrong number of fingers) or unappealing aesthetics (distorted faces, large asymmetry in bodies)?

Rating Label Description

- 0 Severe flaws , Very unappealing

Obvious errors that strongly affect usability: Major physical or visual errors that most viewers would notice immediately, unbalanced composition, clashing colors, heavy jarringness

- 1 Some flaws, Somewhat unappealing

Some errors that noticeably disrupt the image and jeopardize its usability regardless: Clear physical or visual errors that most viewers would eventually notice, the image isn’t an eye sore but something is wrong with its overall composition or color balance

- 2 Minor flaws, Somewhat appealing

Small inaccuracies that are noticeable but are not strongly disruptive: Mostly plausible, but minor inconsistencies reduce believability, acceptable composition and color balance, but lacks artistic quality

- 3

No noticeable technical or logical flaws | Very appealing

The image is free of noticeable technical errors: Fully coherent and physically plausible (if photorealistic, could be mistaken for a real photo; if stylized, maintains internal logic), strong composition, harmonious colors, and captivating style

- 4. (For task 3) Cross-generation image congruence: How well do the images relate to each other in a coherent way, maintaining consistency in recurring elements, style, and context, while allowing for appropriate variation when required?

- 0 Very incoherent

Many recurring elements change in unrealistic or unexplained ways, significantly breaking visual or thematic coherence

- 1 Rather incoherent

Some recurring elements change in unrealistic or unexplained ways, breaking visual or thematic coherence

- 2 Mostly coherent

Most recurring elements match, but there are noticeable mismatches or shifts that reduce cohesion

- 3 Full coherence

Recurring elements, style, and context remain consistent where appropriate, and variations are clearly intentional and coherent

- 5. (For task 3) Generated Text-image alignment: How well does the generated text align with the visual content of the image(s), without contradictions or unsupported details?

Rating Label Description

- 0 Very inconsistent

Text contradicts or misrepresents key elements of the image(s)

- 1 Rather inconsistent

Text aligns with some image content, but contains major mismatches or omissions

- 2 Mostly consistent

Text aligns with most image content, but contains minor mismatches or omissions

- 3 Full consistency

Text accurately and completely reflects the relevant details of the image(s) with no contradictions

- 6. (For task 3) Technical quality of generated text: Does the text contain serious issues such as hallucinations, omissions, or logical errors that undermine accuracy or coherence? Is the tone of the generated text appropriate and congruent with the overall context, style, and intent of the generation task?

Rating Label Description

- 0 Severe flaws (including tone)

Contains clear hallucinations, major omissions, or serious logical inconsistencies; tone is clearly mismatched to the intended context or style, or contradicts the task’s purpose

- 1 Some flaws (including tone)

Some factual gaps, unsupported claims, or reasoning errors: would be considered incorrect and incoherent overall; has some mismatches or inconsistencies in tone, and does not generally fit the context well

- 2 Minor flaws (including tone)

Mostly correct and coherent, but has small factual gaps, minor unsupported claims, or slight reasoning errors; tone generally fits the context in spite of occasional minor mismatches

- 3 No noticeable flaws (including tone)

Text is factually accurate, logically sound, and complete with no unsupported content; tone matches the intended context, style, and purpose throughout

- 7. (For all tasks) Correctness of text rendering: (only if there are texts rendered in the image) Does the image render text correctly? For example, issues with misspellings, distorted text, and inconsistent capitalization?

- 0 Major deviations | Many obvious errors

The text is unreadable, severely distorted, or not rendered

- 1 Partial match | some errors

The text rendered has major misspellings or distorted

- 2 mostly match | minor errors

The text rendered is mostly correct, has minor misspellings or inconsistent capitalization

- 3 Full match | No noticeable error

The rendered text is free of noticeable technical errors

For each pair, after answering the above pointwise evaluation questions, annotators provide their overall preference for answer A vs. B on a 7-point Likert scale, and we convert these ratings to pairwise preferences using the following mapping: ratings 5–7 indicate preference for A, ratings 1–2 indicate preference for B, and ratings 3–4 are treated as ties. The final preference for each pair is determined by majority vote across the three annotators. This rich annotation scheme allows us to capture both the direction and magnitude of preferences while maintaining interpretability.

To ensure high-quality annotations, the annotator vendor applied a post-processing step designed to ensure accuracy, high quality, and oversight, blending automation with human review. Automated checks flagged cases of disagreement, and human reviewers conducted manual reviews. In this process, annotators compared sibling tasks, examined whether disagreements were well founded, and corrected judgments when necessary.

- B.2 Task 4 For the multimodal reasoning task, annotators are asked the following question with answer choices:

Is the model’s reasoning / rationale for the answer correct and consistent?

- • Answer is correct and reasoning has no major errors, omissions, or inaccuracies affecting its correctness or completeness, with no additional improvement needed
- • Answer is correct and reasoning has no major errors, omissions, or inaccuracies affecting its correctness or completeness, but could benefit from minor improvements in reasoning
- • Answer is correct but reasoning has major errors, omissions, or inaccuracies affecting its correctness or completeness
- • Answer is correct, outputs did not include reasoning information
- • Answer is not correct / I cannot verify it

- Figure 7 shows the annotation interface for MMRB2 multimodal reasoning tasks. We also collect free-form rationales from annotators explaining their choices.

Pair construction. We construct preference pairs from annotated model responses. For the human-preferred sample of each pair, we select model responses in which all three human annotators agree that the reasoning contains no major errors and the model answer is correct (i.e., all annotators select either the first or second answer choice above). For the non-preferred sample of each pair, we utilize two kinds of responses: Correct answer, incorrect reasoning, where the model answer is correct but all three annotators consider the reasoning to contain major errors (the third answer choice above), and Incorrect answer, with reasoning, where the model answer is incorrect and some form of reasoning is included. We discard responses for which annotators disagree about the accuracy of the model reasoning. For each pair, the two model responses may share the same modality (both text-only or both image+text) or be a combination. No model response is duplicated across pairs. Table 11 shows the breakdown of pairs across modalities and pair types.

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

- Figure 7 Annotation interface for the multimodal reasoning tasks.

|Pair Type<br><br>|Same Modality<br><br>| |Mixed Modality| |
|---|---|---|---|---|
| |Text<br><br>|Image+Text<br><br>|Pref: Text Not Pref: Image+Text|Pref: Image+Text Not Pref: Text|
|Correct reason vs. Incorrect reason<br><br>|112<br><br>|115|53<br><br>|44|
|Correct answer vs. Incorrect answer|238<br><br>|238<br><br>|100|100|

- Table 11 Number of samples for each reasoning pair type and modality combination.

- C Details for Prompts, Response Generation, and MLLM-as-a-judge

- C.1 Task Prompts Here we provide additional details for the newly synthesized tasks in MMRB2.

Text-Heavy Editing. Text rendering has become increasingly important in practical applications (e.g., designing a product poster), yet it is not well covered in existing image-editing benchmarks. To construct this task, we first curate a set of object-centric images. We collect 200 real images from DreamBench++ (Peng et al., 2025), and additionally create 500 synthetic object images using state-of-the-art text-to-image models GPT-Image (OpenAI, 2025c) and Gemini-2.5-Flash-Image (Google DeepMind, 2025a). The synthetic images can be more creative, such as a newly designed spaceship or a cyberpunk horse. We manually inspect all images to ensure that they are of high quality.

Given an object image, we prompt GPT-4o (OpenAI, 2024) to generate an editing instruction that heavily relies on text rendering, using the following prompt:

#### PROMPT

I am making a text-heavy image-editing benchmark. I provided one images. Generate an edit instruction that take this image as input and output a new image.

The instruction should be realistic and practical. Think about very diverse and creative edits.

This benchmark mainly focuses on the text-heavy editing. Explicitly contain the text you want the model

to render in the prompt. There should be 10 - 50 words in the instruction. Here are some examples, you can think many more:

- 1. create a four-panel comic about an object in the image

- 2. create a poster about the objects in the image

- 3. create a ppt slide about the objects in the image

- 4. add some text to the image

- 5. put a banner or a blackboard with text on the image etc.

**Important**: must contain enough text (10 - 50 words) in the instruction. Devise what texts you want to render in the image. For example, you can create a poster, and the poster can have a bulk of text in several paragraphs.

Use this format: INSTRUCTION: <edit instruction>.

The final MMRB2 image-editing benchmark contains 114 text-heavy editing examples.

Multi-ImageEditing. Recent models such as Gemini 2.5 Flash Image support taking multiple images as input for editing. This enables new use cases such as virtual try-on and composing multiple photos. However, existing image-editing benchmarks mostly cover only single-image editing. We therefore synthesize new multi-image editing examples. Each example consists of 2–3 input images and a textual editing instruction, and the output is a single image (the output image is not included in the benchmark).

We generate the task prompts with our interleaved agents (§C.2), which can produce interleaved text and image responses given arbitrary multimodal prompts. We consider multiple settings for this task. For example, the three input images can all be sampled from the image set used in the text-heavy editing tasks; alternatively, we sample one image from this set and let the agent generate two additional images together with the editing instruction. Each of the 2–3 input images can be either real or synthetic. Below we show the prompt for the setting with one real and one synthetic image: given one real image, the agent is asked to provide another image and an editing instruction:

#### PROMPT

I am making a multi-image image-editing benchmark. I provided one image. First think of how a user may use this image to create a new image/poster/comic/

etc.

Then, think of another image that may be also used to create this. Create the new image based on this. Due to legal concern, do not generate images with human faces. Also, do not leak the components of the original image to the new image.

This do not necessarily need to have the same style as the original image. Finally, generate an edit instruction that take the input image and the generated image as input and

output a new image. The edit instruction can specify the style of the new image. Think diversely on the images, and what they can be used for. For example, a new product, a scene, a style to reference, etc. You don’t have to use everything in the images. For example, you can take one object from each image, and then put it ina new image with completely different styles, or even a comic.

The instruction should be realistic and practical. Think about very diverse and creative edits. Here are some examples, you can think many more:

- 1. Make a multi-panels comic that tell a story

- 2. put the objects together in a new scene

- 3. put them together in a advertisement

- 4. have a image with new style containing all the objects

- 5. reference the style of one image to modify the other etc.

**Important**: Make sure the instruction is reasonable. For example, be careful about the sizes of the objects. Specifiy them carefully when you generate the images, so that the edit instruction is reasonable.

The edit instruction should not contain image index like "image #0" or "<image 0>", rather, you should refer to them as "the first image" or "the second image".

In your response, first give the new image you generated, and then the edit instruction, using this format: <new_image> INSTRUCTION: <edit instruction>

Altogether, there are 178 multi-image editing examples in the MMRB2 image-editing task, among which 79 have 2 input images and 99 have 3 input images.

- C.2 Response generation

All responses are stored in a unified format that supports interleaved text and image content. For all model generations—including LLMs, diffusion models, and unified models—we use the default sampling parameters from the official implementations; in most cases, the sampling temperature is set to 1.0.

Agents. Many interleaved and multimodal reasoning tasks in MMRB2 remain challenging for existing models. For example, we observe that Gemini 2.5 Flash Image, although very strong at generating and editing images, often fails to produce the correct number of images specified by the task prompt. To address these failure modes, we build multimodal tool-using agents for these tasks and collect their responses as additional model outputs.

Specifically, we follow the implementation of Visual Sketchpad (Hu et al., 2024), in which an LLM can write Python code and call tools to generate or edit images. All tool outputs, including both text and images, are returned to the LLM, enabling further planning and reasoning based on these multimodal signals. In all our tool definitions, each generated image is assigned an integer index, and the model can refer to these indices in its answer to produce interleaved text–image outputs. We use GPT-4.1 (OpenAI, 2025a), o3 (OpenAI, 2025d), and GPT-5 (OpenAI, 2025b) as the LLM backbone in these experiments.

We instantiate multiple agent variants that differ in their image-generation components so that MMRB2 can cover a wide variety of interleaved outputs. For GPT-FLUX-agent, we use FLUX.1-dev as the text-to-image tool and FLUX.1-Kontext for image editing (Labs et al., 2025); for GPT-Imagen-agent, we use Imagen-4Ultra (Google DeepMind, 2025d) as the text-to-image tool and Imagen-3-Edit (Baldridge et al., 2024) as the editing tool; for GPT-GPT-Image-Agent, we use GPT-Image-1 (OpenAI, 2025c) for both text-to-image generation and image editing; and for GPT-Gemini-Agent, we use Gemini 2.5 Flash Image (Google DeepMind, 2025a) as the image tool. The tool definitions are as follows.

- 1 tools = [

- 2 {

- 3 "type" : " function " ,

- 4 " function " : {

- 5 "name" : "python_exec" ,

- 6 " description " : "A python code executor that can run your code . Use common python l i b r a r i e s l i k e numpy, matplotlib , PIL , etc . The code can use the load_image ( index ) function to load an image from the image store and the save_image ( image ) function to save an image to the image store . The tool returns stdout / stderr and any generated images . " ,

- 7 "parameters" : {

- 8 "type" : " object " ,

- 9 " properties " : {"code" : {"type" : " string "}} ,

- 10 " required " : [ "code" ] ,

- 11 } ,

- 12 } ,

- 13 } ,

- 14 {

- 15 "type" : " function " ,

- 16 " function " : {

- 17 "name" : "generate_image" ,

- 18 " description " : "Generate an image given a text prompt ( operatiion : generate ) , or generate an image by referencing existing images ( operation : edit ) . Note that edit can be used in a lot of cases , l i k e change style , keep e n t i t i e s consistent , add/remove objects , continue a story / video frame , etc . This tool does not have access to previous images in the conversation history , unless you e x p l i c i t l y reference them in arguments . " ,

- 19 "parameters" : {

- 20 "type" : " object " ,

- 21 " properties " : {

- 22 "prompt" : {

- 23 "type" : " string " ,

- 24 " description " : " for image generation , a detailed description of what to generate / edit (15 -30 words ) . For image editing , a detailed description of what to edit (15 -30 words ) . " ,

- 25 } ,

- 26 " references " : {

- 27 "type" : " array " ,

- 28 "items " : {"type" : " integer "} ,

- 29 " description " : " for edit operation , a l i s t of image references . The f i r s t image in the whole dialogue ( including both user and a s s i s t a n t messages ) i s at index 0 , the second image i s at

index 1 , etc . Use the index to reference the image . " ,

- 30 } ,

- 31 } ,

- 32 " required " : [ "prompt" ] ,

- 33 " additionalProperties " : False ,

- 34 } ,

- 35 } ,

- 36 } ,

- 37 ]

For interleaved tasks, we use the following system prompt. These tasks generally do not require running Python code, so we do not mention that capability in the system prompt.

#### PROMPT

You are a multimodal assistant capable of generating both text and images. When visual content would enhance your response or is specifically requested, you can generate or edit images through

advanced diffusion models. As a helpful assistant, you should generate images in your response to better help the user. Follow user’s multimodal instruction carefully. For example, if user is describing a process, using one text,

one image per step, you should follow this format, generate one text and one image per step. If user asks for three steps, you should generate three pairs of text and image.

## Image Generation Instructions When you need to generate images, use the ‘generate_image‘ function declaration to structure your

response. This function allows you to

**Generate new images** conditioned on detailed prompts and existing images. ## How to Use the Function Declaration

- Use the ‘generate_image‘ function with a detailed prompt and references to existing images. For multistep processes in the SAME SCENE (same kitchen, same objects, same location),you can reference existing images to maintain visual consistency.

## Function Parameters The ‘generate_image‘ function accepts:

- - ‘prompt‘: Detailed description of what to generate/edit (15-30 words)

- - ‘references‘: Array of image references to edit (optional) You can codition on multiple images. ## Formatting of the response

The user want to see text and image that are interleaved in the correct order. In your response you need to use tags like <image #0>, <image #1>, to represent the position of the image in the output. The

number is the index of the image in the whole dialogue (including both user and assistant messages). For example, if you are generating a story, it can be like this: "<image #0> A little cat is sleeping. <

image #1> She woke up and is looking around."

## Best Practices

- - Write clear, specific prompts with visual details

- - Include style preferences and composition elements

- - Reference images by their index

- - The tool does not have access to previous images in the conversation history, unless you explicitly reference them in the function arguments.

- - In most cases, you do not need to include user’s input images in your response.

Provide concise, direct responses that use the function calling system to structure image generation requests. The system will automatically handle the actual image generation based on your function calls.

**DO NOT ask for permission to continue with multi-step processes. Complete the entire requested sequence automatically.**

For the multimodal reasoning task, we use the following system prompt.

#### PROMPT

You are a multimodal assistant capable of generating both text and images. You can use visual tools (python code execution, and image generation tools) to help you reason about

images, and help enhance your response.

For example, if the user asks about some small details in the image, you can crop the image using python codes to zoom in on the image. In your response, include the zoomed image to better show your reasoning process.

The image generation tool is very powerful and can condition on existing images. For example, if you want to see the other angle of an object, you can crop it out first and use the image generation tool to

generate the other angle. ## Tool Instructions All images, including the user’s input images, and your generated images, are stored in a list. You can

access the images by their index. The index starts from 0. You can use "python_exec" to execute python code. You can only use numpy, matplotlib, PIL, and

seaborn beyond the standard library in your code. There are two built in functions: load_image(index:int) -> PIL.image: to load an image from the image list save_image(image:PIL.image) -> int: to save an image to the image list, and return the index of this

image. You can use them directly in your code without importing them. Note that the sandbox cannot show any image. You can use save_image to save the image, and the tool will return the image and its index to the system. You can use "generate_image" to generate an image, conditioned on detailed prompts and arbitrary

number of existing images. ## Function Parameters

The "python_exec" function has one parameter:

- "code": the python code you want to execute. For example, you can load an image, crop it, and save the cropped image. You can also plot additional things (like lines, boxes, labels, etc.) on the image using matplotlib to help

you reason about the image. The ‘generate_image‘ function accepts:

- - ‘prompt‘: Detailed description of what to generate/edit (15-30 words)

- - ‘references‘: Array of image references to condition on (optional) You can codition on multiple images. The ‘generate_image‘ function does not have access to previous images in the conversation history, unless you explicitly reference them in the function arguments. ## Best Practices
- - The user likes to see both text and image in the response.

- - The user wants to see the reasoning process that leads to the final result.

- - Use at most 10 tool calls that I gave you in your reasoning process. ## Response

Show user not only the final result, but also the reasoning process that leads to the final result, which is illustrated by interleaved text and image (which you generated in your reasoning process).

In your response you need to use tags like <image #0>, <image #1>, to represent the image in the output. The number is the index of the image in the whole dialogue (including both user and assistant messages).

For example, if you are answering a math question, it can be like this: "Look closer to the option A, < image #0> We can see that the square is above the triangle. Take a closer look to option B, <image #1> we can see that it is not the case. Thus, the answer is A."

**DO NOT ask for permission to continue with multi-step processes. Complete the entire requested sequence automatically.**

**Use at most 10 tool calls, or you will be terminated.**

**DO NOT ONLY give a final answer. Also show user how you get the final answer.**

**Important: illustrate the reasoning process in your response, with interleaved text and image. For example, if user asks you to put the answer choice in a box, you should first generate the reasoning, and then the answer choice in the box.**

We set the maximum number of turns for these agents to 15. As seen above, the system prompts specify an output format, and we automatically parse the LLM output into an interleaved text–image sequence.

- C.3 MLLM-as-a-judge details For the image-generation task, we use the following system prompt for the MLLM-as-a judge.

#### PROMPT

"""You are an expert in multimodal quality analysis and generative AI evaluation. Your role is to act as an objective judge for comparing two AI-generated responses to the same prompt. You will evaluate which response is better based on a comprehensive rubric.

**Important Guidelines:**

- - Be completely impartial and avoid any position biases

- - Ensure that the order in which the responses were presented does not influence your decision

- - Do not allow the length of the responses to influence your evaluation

- - Do not favor certain model names or types

- - Be as objective as possible in your assessment

- - Consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail

- **Understanding the Content Structure:**
- - **[ORIGINAL PROMPT TO MODEL:]**: This is the instruction given to both AI models

- - **[INPUT IMAGE FROM PROMPT:]**: This is the source image provided to both models (if any)

- - **[RESPONSE A:]**: The first model’s generated response (text and/or images)

- - **[RESPONSE B:]**: The second model’s generated response (text and/or images)

Your evaluation must be based on a fine-grained rubric that covers the following criteria. For each criterion, you must provide detailed step-by-step reasoning comparing both responses. You will use a 1-6 scoring scale.

**Evaluation Criteria:**

- 1. **faithfulness_to_prompt:** Which response better adheres to the composition, objects, attributes, and spatial relationships described in the text prompt?

- 2. **text_rendering:** If either response contains rendered text, which one has better text quality ( spelling, legibility, integration)? If no text is rendered, state "Not Applicable."

- 3. **input_faithfulness:** If an input image is provided, which response better respects and incorporates the key elements and style of that source image? If no input image is provided, state "Not Applicable

."

- 4. **image_consistency:** If multiple images are generated, which response has better visual consistency between images (character appearance, scene details)? If no multiple images are provided, state "Not Applicable."

- 5. **text_image_alignment:** Which response has better alignment between text descriptions and visual content?

- 6. **text_quality:** If text was generated, which response has better linguistic quality (correctness, coherence, grammar, tone)?

- 7. **overall_quality:** Which response has better general technical and aesthetic quality, realism, coherence, and fewer visual artifacts or distortions?

**Scoring Rubric:**

- - Score 6 (A is significantly better): Response A is significantly superior across most criteria

- - Score 5 (A is marginally better): Response A is noticeably better across several criteria

- - Score 4 (Unsure or A is negligibly better): Response A is slightly better or roughly equivalent

- - Score 3 (Unsure or B is negligibly better): Response B is slightly better or roughly equivalent

- - Score 2 (B is marginally better): Response B is noticeably better across several criteria

- - Score 1 (B is significantly better): Response B is significantly superior across most criteria

**Confidence Assessment:** After your evaluation, assess your confidence in this judgment on a scale of 0.0 to 1.0:

**CRITICAL**: Be EXTREMELY conservative with confidence scores. Most comparisons should be in the 0.2-0.5 range.

- - **Very High Confidence (0.8-1.0)**: ONLY for absolutely obvious cases where one response is dramatically better across ALL criteria with zero ambiguity. Use this extremely rarely (less than 10%

of cases).

- - **High Confidence (0.6-0.7)**: Clear differences but some uncertainty remains. Use sparingly (less than 20% of cases).

- - **Medium Confidence (0.4-0.5)**: Noticeable differences but significant uncertainty. This should be your DEFAULT range.

- - **Low Confidence (0.2-0.3)**: Very close comparison, difficult to distinguish. Responses are roughly equivalent or have conflicting strengths.

- - **Very Low Confidence (0.0-0.1)**: Essentially indistinguishable responses or major conflicting strengths.

**IMPORTANT GUIDELINES**:

- - DEFAULT to 0.3-0.5 range for most comparisons

- - Only use 0.6+ when you are absolutely certain

- - Consider: Could reasonable people disagree on this comparison?

- - Consider: Are there any strengths in the "worse" response?

- - Consider: How obvious would this be to a human evaluator?

- - Remember: Quality assessment is inherently subjective

After your reasoning, you will provide a final numerical score, indicate which response is better, and assess

your confidence. You must always output your response in the following structured JSON format: {

"reasoning": { "faithfulness_to_prompt": "YOUR REASONING HERE", "text_rendering": "YOUR REASONING HERE", "input_faithfulness": "YOUR REASONING HERE", "image_consistency": "YOUR REASONING HERE", "text_image_alignment": "YOUR REASONING HERE", "text_quality": "YOUR REASONING HERE", "overall_quality": "YOUR REASONING HERE", "comparison_summary": "YOUR OVERALL COMPARISON SUMMARY HERE"

}, "score": <int 1-6>, "better_response": "A" or "B", "confidence": <float 0.0-1.0>, "confidence_rationale": "YOUR CONFIDENCE ASSESSMENT REASONING HERE"

}

For the image-editing task, we use the following system prompt for the MLLM-as-a judge.

#### PROMPT

You are an expert in image editing quality analysis and AI evaluation. Your role is to act as an objective judge for comparing two AI-generated image editing responses to the same prompt. You

will evaluate which response is better based on a comprehensive rubric specifically designed for image editing tasks.

**Important Guidelines:**

- - Be completely impartial and avoid any position biases

- - Ensure that the order in which the responses were presented does not influence your decision

- - Do not allow the length of the responses to influence your evaluation

- - Do not favor certain model names or types

- - Be as objective as possible in your assessment

- - Focus on image editing specific factors: faithfulness to editing instructions, preservation of input image elements, and overall editing quality **Understanding the Content Structure:**
- - **[ORIGINAL PROMPT TO MODEL:]**: This is the image editing instruction given to both AI models

- - **[INPUT IMAGE FROM PROMPT:]**: This is the source image provided to both models for editing

- - **[RESPONSE A:]**: The first model’s edited image response

- - **[RESPONSE B:]**: The second model’s edited image response

Your evaluation must be based on a fine-grained rubric that covers the following criteria. For each criterion, you must provide detailed step-by-step reasoning comparing both responses. You will use a

1-6 scoring scale.

**Evaluation Criteria:**

- 1. **text_faithfulness:** Which response better adheres to the text editing instruction? Consider how well each response follows the specific editing instructions (e.g., adding objects, changing colors, modifying scenes).

- 2. **image_faithfulness:** Which response better respects and incorporates the key elements of the input image? Consider how well each response preserves important aspects of the original image ( composition, lighting, style, background elements) while making the requested changes.

- 3. **overall_image_quality:** Which response has better general technical and aesthetic quality, with fewer visual artifacts, distortions, or inconsistencies introduced during the editing process?

- 4. **text_rendering:** If either response contains rendered text, which one has better text quality ( spelling, legibility, integration with the image)? If no text is rendered, state "Not Applicable."

**Scoring Rubric:**

- - Score 6 (A is significantly better): Response A is significantly superior across most criteria

- - Score 5 (A is marginally better): Response A is noticeably better across several criteria

- - Score 4 (Unsure or A is negligibly better): Response A is slightly better or roughly equivalent

- - Score 3 (Unsure or B is negligibly better): Response B is slightly better or roughly equivalent

- - Score 2 (B is marginally better): Response B is noticeably better across several criteria

- - Score 1 (B is significantly better): Response B is significantly superior across most criteria

**Confidence Assessment:** After your evaluation, assess your confidence in this judgment on a scale of 0.0 to 1.0:

**CRITICAL**: Be EXTREMELY conservative with confidence scores. Most comparisons should be in the 0.2-0.5 range.

- - **Very High Confidence (0.8-1.0)**: ONLY for absolutely obvious cases where one response is dramatically better across ALL criteria with zero ambiguity. Use this extremely rarely (less than 10%

of cases).

- - **High Confidence (0.6-0.7)**: Clear differences but some uncertainty remains. Use sparingly (less than 20% of cases).

- - **Medium Confidence (0.4-0.5)**: Noticeable differences but significant uncertainty. This should be your DEFAULT range.

- - **Low Confidence (0.2-0.3)**: Very close comparison, difficult to distinguish. Responses are roughly equivalent or have conflicting strengths.

- - **Very Low Confidence (0.0-0.1)**: Essentially indistinguishable responses or major conflicting strengths.

**IMPORTANT GUIDELINES**:

- - DEFAULT to 0.3-0.5 range for most comparisons

- - Only use 0.6+ when you are absolutely certain

- - Consider: Could reasonable people disagree on this comparison?

- - Consider: Are there any strengths in the "worse" response?

- - Consider: How obvious would this be to a human evaluator?

- - Remember: Quality assessment is inherently subjective

After your reasoning, you will provide a final numerical score, indicate which response is better, and assess

your confidence. You must always output your response in the following structured JSON format: {

"reasoning": { "text_faithfulness": "YOUR REASONING HERE", "image_faithfulness": "YOUR REASONING HERE",

"overall_image_quality": "YOUR REASONING HERE", "text_rendering": "YOUR REASONING HERE", "comparison_summary": "YOUR OVERALL COMPARISON SUMMARY HERE"

}, "score": <int 1-6>, "better_response": "A" or "B", "confidence": <float 0.0-1.0>, "confidence_rationale": "YOUR CONFIDENCE ASSESSMENT REASONING HERE"

}

For the interleaved generation task, we use the following system prompt for the MLLM-as-a judge.

#### PROMPT

You are an expert in multimodal interleaved generation quality analysis and AI evaluation. Your role is to act as an objective judge for comparing two AI-generated interleaved responses (text and images) to the same prompt. You will evaluate which response is better based on a comprehensive rubric specifically designed for interleaved generation tasks.

**Important Guidelines:**

- - Be completely impartial and avoid any position biases

- - Ensure that the order in which the responses were presented does not influence your decision

- - Do not allow the length of the responses to influence your evaluation

- - Do not favor certain model names or types

- - Be as objective as possible in your assessment

- - Focus on interleaved generation specific factors: faithfulness to instructions, quality of both text and images, and coherence between modalities

**Understanding the Content Structure:**

- - **[ORIGINAL PROMPT TO MODEL:]**: This is the interleaved generation instruction given to both AI models

- - **[INPUT IMAGE FROM PROMPT:]**: This is the source image provided to both models (if any)

- - **[RESPONSE A:]**: The first model’s interleaved response (text and/or images)

- - **[RESPONSE B:]**: The second model’s interleaved response (text and/or images)

Your evaluation must be based on a fine-grained rubric that covers the following criteria. For each criterion, you must provide detailed step-by-step reasoning comparing both responses. You will use a 1-6 scoring scale.

**Evaluation Criteria:**

- 1. **text_faithfulness:** Which response better adheres to the text instruction? Consider how well each response follows the specific text generation instructions and requirements.

- 2. **image_faithfulness:** Which response better respects and incorporates the key elements of the input image? Consider how well each response preserves important aspects of the original image ( composition, lighting, style, background elements) while making the requested changes. If no input image is provided, state "Not Applicable."

- 3. **overall_image_quality:** Which response has better overall quality of generated image? Consider technical and aesthetic quality, with fewer visual artifacts, distortions, or inconsistencies.

- 4. **congruence:** If multiple images are generated, which response has better cross-generation image congruence? Consider visual consistency between images (character appearance, scene details, style consistency). If no multiple images are provided, state "Not Applicable."

- 5. **text_image_alignment:** Which response has better generated text-image alignment? Consider how well the text and images work together as a coherent multimodal response.

- 6. **text_quality:** If text was generated, which response has better technical quality of generated text? Consider linguistic quality (correctness, coherence, grammar, tone, clarity). If no text is generated, state "Not Applicable."

- 7. **text_rendering:** If either response contains rendered text within images, which one has better correctness of text rendering? Consider text quality (spelling, legibility, integration with the image). If no text is rendered in images, state "Not Applicable."

**Scoring Rubric:**

- - Score 6 (A is significantly better): Response A is significantly superior across most criteria

- - Score 5 (A is marginally better): Response A is noticeably better across several criteria

- - Score 4 (Unsure or A is negligibly better): Response A is slightly better or roughly equivalent

- - Score 3 (Unsure or B is negligibly better): Response B is slightly better or roughly equivalent

- - Score 2 (B is marginally better): Response B is noticeably better across several criteria

- - Score 1 (B is significantly better): Response B is significantly superior across most criteria

**Confidence Assessment:** After your evaluation, assess your confidence in this judgment on a scale of 0.0 to 1.0:

**CRITICAL**: Be EXTREMELY conservative with confidence scores. Most comparisons should be in the 0.2-0.5 range.

- - **Very High Confidence (0.8-1.0)**: ONLY for absolutely obvious cases where one response is dramatically better across ALL criteria with zero ambiguity. Use this extremely rarely (less than 10%

of cases).

- - **High Confidence (0.6-0.7)**: Clear differences but some uncertainty remains. Use sparingly (less than 20% of cases).

- - **Medium Confidence (0.4-0.5)**: Noticeable differences but significant uncertainty. This should be your DEFAULT range.

- - **Low Confidence (0.2-0.3)**: Very close comparison, difficult to distinguish. Responses are roughly equivalent or have conflicting strengths.

- - **Very Low Confidence (0.0-0.1)**: Essentially indistinguishable responses or major conflicting strengths.

**IMPORTANT GUIDELINES**:

- - DEFAULT to 0.3-0.5 range for most comparisons

- - Only use 0.6+ when you are absolutely certain

- - Consider: Could reasonable people disagree on this comparison?

- - Consider: Are there any strengths in the "worse" response?

- - Consider: How obvious would this be to a human evaluator?

- - Remember: Quality assessment is inherently subjective

After your reasoning, you will provide a final numerical score, indicate which response is better, and assess

your confidence. You must always output your response in the following structured JSON format: {

"reasoning": { "text_faithfulness": "YOUR REASONING HERE", "image_faithfulness": "YOUR REASONING HERE", "overall_image_quality": "YOUR REASONING HERE", "congruence": "YOUR REASONING HERE", "text_image_alignment": "YOUR REASONING HERE", "text_quality": "YOUR REASONING HERE", "text_rendering": "YOUR REASONING HERE", "comparison_summary": "YOUR OVERALL COMPARISON SUMMARY HERE"

}, "score": <int 1-6>,

"better_response": "A" or "B", "confidence": <float 0.0-1.0>, "confidence_rationale": "YOUR CONFIDENCE ASSESSMENT REASONING HERE"

}

For the reasoning task, we use the following system prompt for the MLLM-as-a judge.

#### PROMPT

Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user’s instructions and answers the user’s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, and level

of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not

influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.

After your reasoning, you will provide a final judgement, indicate which response is better. You must

always output your response in the following structured JSON format: {

"reasoning": "YOUR REASONING HERE", "better_response": "A" or "B"

}

As shown, these prompts are very close to the rubrics that were used for human annotations.

- D Limitations and Future Directions

MMRB2 is designed as a first comprehensive benchmark for omni-model reward evaluation in text–image settings. In this section, we clarify the scope of the current release and outline natural extensions that our pipeline can support.

Scope and focus. The current version of MMRB2 focuses on core use cases for omni models: text-to-image generation, image editing, interleaved text–image generation, and multimodal reasoning over images. We also focus on overall human preference, rather than more fine-grained dimensions. By concentrating on this space, MMRB2 offers a focused yet diverse benchmark that is immediately useful for training and evaluating multimodal reward models.

Modalitiesandtaskformats. While MMRB2 is grounded in text–image interactions, the underlying construction pipeline is modality-agnostic. The same recipe of prompt curation, multi-model candidate generation, ensemble filtering, and expert preference collection can be applied to additional modalities such as video, audio, or 3D content as these use cases and tools become more standardized. Likewise, our current tasks are predominantly single-turn; extending MMRB2 to multi-turn and agentic interaction trajectories, where reward models must evaluate sequences rather than single responses, is a natural next step.

Data sources and coverage. Our prompts are sourced primarily from established benchmarks and carefully designed task variants. This choice ensures clear task definitions and strong coverage of core capabilities. At the same time, it leaves room for complementary extensions focusing on in-the-wild user queries, domainspecific applications, and multilingual settings. We view MMRB2 as the backbone that more specialized or application-driven subsets can build upon.

Evaluationdimensions. The present benchmark emphasizes overall task-level preference quality: which response better satisfies the user’s instruction in a given multimodal setting. Our pipeline can also support additional evaluation dimensions, including safety- and bias-sensitive preferences, robustness to adversarial prompts, or fairness across demographic attributes, by appropriately adapting the prompt sources and annotation guidelines. We expect such specialized subsets to further broaden the applicability of MMRB2 for alignment and safety research.

[Figure 61]

Evolvingjudgesandbenchmarks. Finally, MMRB2 uses a diverse ensemble of contemporary judges in its filtering stage to focus human effort on informative comparisons. As frontier and open-source models continue to evolve, the same modular design allows future versions of MMRB2 to refresh the judge ensemble, incorporate new model families, and add new tasks, while retaining compatibility with the core benchmark principles introduced here.

- E Examples

Here we show two examples from each task in MMRB2. For each task prompt, there is a Response A and a Response B. The human-preferred output is indicated with a green checkmark next to it. We also label which model the response comes from, for illustration purposes.

- Figure 8 An example of MMRB2 text-to-image task. Response A, generated by GPT-Image-1, is preferred over Response B, generated by FLUX. The rationale is that Response B is not a railway underpass.

[Figure 62]

- Figure 9 An example of MMRB2 text-to-image task. Responses A and B are both generated by Gemini 2.5 Flash

- Image, while B is preferred over A. The rationale is that Response A only has five people, which does not align with the user input.

[Figure 63]

- Figure 10 An example of MMRB2 image-editing task. Response A, generated by Gemini 2.5 Flash Image, is preferred over Response B, generated by GPT-Image. The rationale is that many important texts are missing in Response B. Response A also has some rendering mistakes in the small texts, but this is a smaller issue compared to B.

[Figure 64]

- Figure 11 An example of MMRB2 image-editing task. Responses A and B are both generated by Gemini 2.0 Flash

- Image, while B is preferred over A. The rationale is that Response B follows the instruction better, and the backpack is more “anime-styled.”

[Figure 65]

###### Figure 12 An example of MMRB2 interleaved task. Responses A and B are both generated by the agent with GPT-Image, while B is preferred over A. The rationale is that Response B better follows the instruction and is more consistent with the original image.

[Figure 66]

[Figure 67]

###### Figure 13 An example of MMRB2 interleaved task. Responses A and B are both generated by Gemini 2.5 Flash Image, while A is preferred over B. The rationale is that in Response B the cat is barely changed across the images, while in A the cats are more natural while remaining consistent.

[Figure 68]

###### Figure 14 An example of MMRB2 multimodal reasoning task. Response A, generated by Gemini 2.5 Pro, is preferred over Response B, which is generated by GPT-4.1. Response A has correct reasoning and answer, while Response B’s reasoning has apparent problems. For example, “2nd circle: a veritcal and a diaglonal line” is incorrect.

[Figure 69]

[Figure 70]

###### Figure 15 An example of MMRB2 multimodal reasoning task. Responses A and B are both generated by sketchpad agents. A uses GPT-5 as the LLM backbone, and B uses o3 as the backbone. A is preferred over B. The rationale is that B does not contain analysis for the third image, so the reasoning process is incomplete.

