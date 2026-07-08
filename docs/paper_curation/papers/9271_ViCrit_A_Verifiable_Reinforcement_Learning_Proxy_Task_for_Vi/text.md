arXiv:2506.10128v1[cs.CV]11Jun2025

# ViCrit: A Verifiable Reinforcement Learning Proxy Task for Visual Perception in VLMs

Xiyao Wang1,2†, Zhengyuan Yang2† , Chao Feng3† Yongyuan Liang1, Yuhang Zhou1, Xiaoyu Liu1, Ziyi Zang4, Ming Li1 Chung-Ching Lin2, Kevin Lin2, Linjie Li2‡, Furong Huang1‡, Lijuan Wang2‡ 1University of Maryland, College Park 2Microsoft 3University of Michigan 4Cardiff University †First Authors ‡Equal Advising Project Lead

Reinforcement learning (RL) has shown great effectiveness for fine-tuning large language models (LLMs) using tasks that are challenging yet easily verifiable, such as math reasoning or code generation. However, extending this success to visual perception in vision–language models (VLMs) has been impeded by the scarcity of vision-centric tasks that are simultaneously challenging and unambiguously verifiable. To this end, we introduce ViCrit (Visual Caption Hallucination Critic), an RL proxy task that trains VLMs to localize a subtle, synthetic visual hallucination injected into paragraphs of human-written image captions. Starting from a 200-word captions, we inject a single, subtle visual description error—altering a few words on objects, attributes, counts, or spatial relations—and task the model to pinpoint the corrupted span given the image and the modified caption. This formulation preserves the full perceptual difficulty while providing a binary, exact-match reward that is easy to compute and unambiguous. Models trained with the ViCrit Task exhibit substantial gains across a variety of VL benchmarks. Crucially, the improvements transfer beyond naturalimage training data to abstract image reasoning and visual math, showing promises of learning to perceive rather than barely memorizing seen objects. To facilitate evaluation, we further introduce ViCrit-Bench, a category-balanced diagnostic benchmark that systematically probes perception errors across diverse image domains and error types. Together, our results demonstrate that fine-grained hallucination criticism is an effective and generalizable objective for enhancing visual perception in VLMs.

Date: Jun 11, 2025

Code Repository: https://github.com/si0wang/ViCrit

Model Weights: https://huggingface.co/collections/russwang/ViCrit

ViCrit Training Dataset: https://huggingface.co/datasets/zyang39/ViCrit-Train

ViCrit-Bench: https://huggingface.co/datasets/russwang/ViCrit-Bench

## 1. Introduction

Reinforcement learning (RL) has recently emerged as a dominate paradigm (Guo et al., 2025, Jaech et al., 2024) for fine-tuning large language models (LLMs) when training tasks are both challenging and automatically verifiable. Successful examples include mathematical reasoning tasks with concise numerical answers (Hendrycks et al., 2021, of America, 2024), and software engineering problems (Zheng et al., 2023, Miserendino et al., 2025) whose correctness can be checked in a sandboxed environment. By focusing on tasks that strike this balance—sufficiently challenging to have room for improvements yet straightforward to

Corresponding author(s): Xiyao Wang xywang@umd.edu; Zhengyuan Yang zhengyang@microsoft.com

[Figure 1]

###### Synthetic Hallucination Injection VLM + RL via ViCrit Task

[Figure 2]

###### Object Description Perturbation

Challenging in visual perception Unambiguously verifiable outcome reward Internalize fine-grained visual perception

[Figure 3]

❖ Original: an iPad on the table

❖ Hallucination: a MacBook Pro on the table

[Figure 4]

❖ Original: black flag [...] reads both "true" and "false."

[Figure 5]

[Figure 6]

[Figure 7]

❖ Hallucination: black flag [...] reads "tree"

MathVision VLMsAreBlind Charxiv

[Figure 8]

[Figure 9]

[Figure 10]

ViCrit Task: Hallucination Prediction

###### Human-written caption:

The image showcases a social gathering of Caucasian individuals, both male and female, ranging from middle age to about 60, seated at multiple tables inside a room that appears to be a café or restaurant. The café’s walls are a light brown to mustard yellow, adorned with an eclectic mix of picture frames and flags, including one particularly striking black flag with curved white stitching that reads both "true" and "false." There is a tall vertical window on the left side, offering a view of trees and parked cars outside. Hanging from the ceiling are two distinct light fixtures: a black wrought iron chandelier with six gold-colored bulbs, and a single glass pendant light with a black wire. Additionally, a lamp occupies the corner on the left side. Near this window, a woman dressed in black and wearing glasses is seated alone with an iPad on the table, a coffee cup beside her, and she is gazing out the window. Nearby, a group of four individuals, predominantly young men, are engaged in conversation and one is looking at his phone. To the right, there are smaller tables, where pairs of people, including some young women, are conversing. At one table in the lower right corner, a man with headphones and a blue jacket looks down, perhaps immersed in his own world. The atmosphere is lively, with a mix of discussions and some quiet moments of individual focus.

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

|RL reward: +1 for correct Hallucination spans, 0 o.w.|
|---|

###### Verifiable reward via string match

- Figure 1: Overview of the ViCrit framework. Starting from high-quality image–caption pairs, we synthetically inject visual hallucinations by minimally altering noun phrases. The model is trained to localize incorrect spans in the caption given the image, receiving a verifiable reward through exact string matching. This finegrained perceptual objective improves visual perception in vision-language models (VLMs) and generalizes to downstream reasoning tasks across diverse visual domains.

grade deterministically—RL can explore the solution space effectively, extract genuinely useful strategies, and transfer those gains to broader domains.

Despite its success in textual reasoning, RL training with verifiable rewards has yet to demonstrate a comparable significance in improving the visual perception abilities of vision–language models (VLMs). This is largely due to the lack of vision-centric tasks that are both perceptually challenging and automatically gradable. Whereas multi-hop math problems naturally compress numerous premises into a single verifiable answer, the semantic elements within an image rarely collapses into such a tidy question-answer pair. Even advanced visual-question-answering benchmarks (Antol et al., 2015, Hudson and Manning, 2019, Singh et al., 2019, Marino et al., 2019) often probe only fragments of a scene, allowing shallow perception to suffice. Attempts to increase the perceptual difficulty, such as exhaustive image captioning that enumerates every visual element (Pont-Tuset et al., 2020, Deitke et al., 2024, Lian et al., 2025, Chen et al., 2024a, Awadalla et al., 2024, Wu et al., 2024a), yield paragraph-length outputs (200+ words) that are nearly impossible to grade unambiguously. The central challenge, therefore, is to craft a task that forces the model to perceive the full scene yet produces a concise, deterministically verifiable response.

To bridge this gap, we propose ViCrit (Visual Caption Hallucination Critic), a reinforcement learning proxy task that offers both perceptual difficulty and evaluation simplicity. ViCrit trains VLMs to localize synthetic visual hallucinations injected into paragraph-length image captions. It is designed to be both challenging, requiring fine-grained visual perception, and verifiable, enabling rule-based deterministic reward signals for scalable RL training. As shown in Figure 1, the task begins with human-annotated detailed image captions with more than 200 words (Deitke et al., 2024), and synthetically injects visual hallucinations. Such subtle errors misdescribe object, attribute, count, scene text, or spatial relation as its visually similar alternative. The model is trained to act as a critic: given an image and its corrupted caption, it must identify the specific

tokens that are incorrect. This token-level span detection can be easily graded via string matching yet requires fine-grained visual perception across the entire image, encouraging the model to internalize robust visual perception strategies extracted during the RL exploration trajectories.

Training Qwen2.5-VL-7B-Instruct and 72B-Instruct with the proposed ViCrit RL task yields consistent gains across ten vision-language benchmarks. In addition to better hallucination benchmark results, these improvements extend well beyond the natural-image domain seen during ViCrit RL training, onto abstract image reasoning and visual math: Qwen2.5-VL-72B-Instruct improves from 35.2% to 40.1% on MathVision (Wang et al., 2024a), from 61.3% to 65.8% on VLMsAreBlind (Rahmanzadehgervi et al., 2024), and from 45.5% to 49.4% on Charxiv (Wang et al., 2024d). These cross-domain improvements indicate that the learned perceptual strategies transfer effectively to general VL domains. By training models to pinpoint fine-grained errors, ViCrit encourages the development of internal perception strategies that cross-check textual claims against visual evidence. Unlike supervised fine-tuning on captioning data (Sariyildiz et al., 2020, Desai and Johnson, 2021, Tschannen et al., 2023), which can lead to surface-level memorization, our RL task rewards perceptual correctness and penalizes hallucinations directly. As a result, the model moves beyond merely memorizing the seen object lists, towards learning to decide how to perceive an image. Comprehensive analyses on how ViCrit-induced chain-of-thoughts generalize to a broad spectrum of downstream VLM tasks further reveals the effectiveness and working mechanism of the ViCrit RL training.

In addition to ViCrit training, we present a benchmark named ViCrit-Bench for evaluating VLMs on hallucination detection. We group images into four categories and hallucination types into eight fine-grained hallucination classes, enabling detailed diagnostic analysis. We then manually curate a set of images selected from PixMo-Cap (Deitke et al., 2024) and inject eight types of hallucinations into their corresponding captions. This process results in a high-quality, fine-grained, and highly challenging hallucination detection benchmark, containing 607 samples. The benchmark supports zero-shot evaluation and exposes clear correlations with downstream perception tasks, making it a powerful probe of VLMs’ perception limitations. We benchmark a range of state-of-the-art open-source and closed-source vision-language models on ViCrit-Bench. Even proprietary systems such as OpenAI-o3 and Gemini-2.5-Pro achieve only 47.7% and 45.2% accuracy. After in-domain reinforcement learning with the ViCrit task, Qwen2.5-VL-72B attains an improved accuracy of 43.0%. The gains are uniform across all four image categories and are especially pronounced on document and abstract images, highlighting the efficacy of ViCrit-based RL for strengthening generalizable visual perception.

Our contributions are summarized as follows:

- • We introduce ViCrit, an RL task for visual perception that requires VLMs to identify token-level visual hallucinations in paragraph-length image captions. The task is both perceptually challenging and automatically verifiable, enabling scalable RL training with precise, unambiguous supervision.
- • Training VLMs with the ViCrit Task significantly enhances their performance on a wide range of VL benchmarks. The improvements also generalize to other image domains such as abstract image reasoning and visual math, which shows the advantage of ViCrit incentivizing models to verify visual detail against text, rather than merely memorize seen objects.
- • We present ViCrit-Bench that systematically probes eight hallucination types across four image domains. The benchmark supports zero-shot evaluation and serves as a diagnostic tool for assessing fine-grained visual perception capabilities in VLMs. Furthermore, its scores track averaged VLM accuracy monotonically, making it a strong indicator of the overall VLM performance.

- 2. Related Work Large language model reasoning. Prompting-based Chain-of-thought methods (Wei et al., 2022, Kojima

- et al., 2022) first explored the reasoning abilities of large language models (LLMs) (Brown et al., 2020, Chowdhery et al., 2023) by eliciting chains of intermediate thoughts, markedly improving arithmetic and commonsense benchmarks (Cobbe et al., 2021, Patel et al., 2021, Koncel-Kedziorski et al., 2016). Subsequent decoding strategies aim to further improve test-time performance with extra test-time computation. For example, Self-Consistency sampling (Wang et al., 2022b) that votes over multiple thought paths to boost reliability. Expanding beyond linear traces, structured search frameworks like Tree-of-Thoughts (Yao et al.,

- 2023) and Graph-of-Thoughts (Jin et al., 2024) let the model explore a branching space of candidate “thought” states before committing to an answer. Studies (Muennighoff et al., 2025) also explore hacking the thought process to generate long CoT that is beyond the CoT length distribution. Moving from test-time scaling to training, process reward models (Lightman et al., 2023, Uesato et al., 2022, Wang et al., 2023) grade each reasoning step rather than only final answers, which can be coupled with Monte Carlo Tree Search (Xie et al.,
- 2024) for fine-grained value estimates. Most recently, large-scale reinforcement learning (Guo et al., 2025, Jaech et al., 2024) with outcome-based rewards alone can induce emergent multi-step reasoning skills. The key to its success is challenging tasks that can be automatically verified, such that the RL can be effectively scaled up with minimal noise in its reward signals. The goal for this study is to find such tasks for VLMs’ visual perception. VLM reasoning. Based on the modern vision language models (OpenAI, 2023, Wang et al., 2022a, Liu et al.,

- 2023, Hurst et al., 2024, Liu et al., 2024, Bai et al., 2025, Chen et al., 2024c, Tong et al., 2024, Li et al., 2024b, Yang et al., 2023a), recent studies explore the use of multimodal CoT to further improve vision-language reasoning tasks (Hao et al., 2025, Wang et al., 2024a, Lu et al., 2024, Yu et al., 2024a) with both grounded textual thoughts (Lu et al., 2022, Zhang et al., 2024c) and multimodal thoughts (Rose et al., 2023, Wu et al.,
- 2024b, Fu et al., 2025). Techniques like rationale distillation and self-reflection further boost these models’ reasoning capabilities (Zhang et al., 2024b, Yang et al., 2023b, Zhou et al., 2024b, Wang et al., 2024b, Xiong et al., 2024, Wang et al., 2024c, Deng et al., 2024). Inspired by the success on outcome-based reward based RL in LLMs, recent studies (Deng et al., 2025, Huang et al., 2025, Meng et al., 2025b,a, Wang et al., 2025, Peng et al., 2025, Chen et al., 2025, Zheng et al., 2025, Ni et al., 2025) applies similar techniques to visual math and other visual-question-answering benchmarks. Despite the improvements in visual math and STEM questions, they still fall short of significantly advancing fine-grained visual perception.

Visual-centric VLMs and reasoning. One threads of works aim to improve visual perception in VLMs via text-supervised visual representation learning (Sariyildiz et al., 2020, Desai and Johnson, 2021, Tschannen

- et al., 2023), which trains the model to generate a good description of the image. This line of work show great promises with the recent success in obtaining ultra-descriptive image captions (Pont-Tuset et al., 2020, Deitke et al., 2024, Lian et al., 2025, Chen et al., 2024a, Awadalla et al., 2024, Betker et al., 2023, Wu et al.,
- 2024a). However, supervised fine-tuning on captioning data may lead to superficial object memorization, while paragraph captioning task does not have reliable rewards for RL training. In this work, we present an RL proxy task to close this gap.

- 3. ViCrit RL Training

Recent progress in outcome-based reinforcement learning shows that LLMs learn richer reasoning procedures when trained with hard questions whose answers can be unambiguously verified. The same recipe, however, is not immediately available to visual perception in VLMs. The traditional caption-supervision objective

###### ViCrit Task: Which noun phrase is replaced by the hallucinated noun phrase in this image description?

[Figure 15]

Perturbed caption: (209 words) The image captures a detailed overhead view of a grey and white granite or formica tabletop. Central to the composition, towards the bottom, is a rectangular yellow styrofoam tray commonly found in grocery store meat departments. The tray is neatly lined with two rows of raw chicken drumsticks. Just to the right edge of the tray, a small black jar labeled "Fly and Saucy Medium" in red and white lettering is positioned. Adjacent to this, a red-capped container with a label reading "Zatarain's Creole Seasoning" is placed prominently. Near the top right corner of the tray, a large bottle of Frank's RedHot Wing Sauce, filled to the brim with its fiery red contents, stands upright. In the background, slightly to the right, a charming small statue of a brown corgi dog adds a whimsical touch. The dog wears a sign around its neck, albeit unreadable, and is flanked by salt and pepper shakers nestled in tiny ornate dog bowls. In the upper right corner of the image, a brown bottle with a white plastic wrap around its top can be seen. Its white label, featuring "Mark's Lone Star Certified Good Stuff Honey Barbecue" in white lettering against a tan background, completes the scene with a hint of delectable flavor.

Visual hallucination (GT): (2 words)

dog corgi

[Figure 16]

Perturbed caption: (226 words) In this outdoor photograph, a musical band is performing beneath a white festival tent with open walls, supported by gray struts and metal supports. A prominent black banner with the text "Friday Nights Live" in pink, white, and yellow is displayed at the front of the tent. The banner also features logos and the word "Hospital" in white text. Spotlights, including orange and reddish hues, cast a glow over the band. The band features a variety of musicians: On the left, a Caucasian man in his 30s with dark hair, wearing a gray long-sleeved shirt and blue jeans, plays a golden-colored saxophone. Beside him, to the right, the lead singer stands front and center, dressed entirely in black with a black fedora and a lighter gray belt. He holds a microphone close to his mouth and a tambourine at his hip. To the right of the lead singer, a man whose face is obscured plays a guitar. In the deep background, a young woman with a black billed baseball cap, gray hoodie, and blue jeans smiles, her role in the band undetermined. There are also black music stands placed around the musicians. Behind the open tent walls, a green tree line stretches across the background, meeting a clear blue sky at the horizon. Small white text at the bottom corner of the photograph reads "Chuck Willie."

Visual hallucination (GT): (2 words)

Willie Chuck

- Figure 2: Instead of asking the model to write a paragraph-long caption that is hard to grade (e.g., the 200-word example above), ViCrit feeds the model an almost-correct caption containing a single, deliberately inserted visual hallucination and trains it to locate that error. The short, token-level response is just as demanding in terms of visual perception, yet it is far easier to verify automatically.

optimizes a model for recalling a fixed list of objects, but never for deciding where to look next. Our goal is therefore to turn visual perception into an RL problem whose reward (i) compels the model to interrogate every visual details and (ii) remains as cheap and deterministic to evaluate as in code or math.

Examples of our proposed ViCrit task is shown in Figure 2. Instead of asking a model to generate a perfect, paragraph-length caption (200+ words), which is difficult to grade, we present it with an almost-correct caption containing a single, synthetically injected visual error and reward the model for pinpointing the mistaken span (2 words). Solving this task is as hard as perfect captioning: a critic that can reliably spot any hallucination must perceive and understand the entire scene; yet the answer collapses into a few words that can be matched exactly. This simple reshape of the objective delivers the two missing ingredients for perception-centric RL: a genuinely challenging perception task and an evaluation rule that reduces to simple string equality.

#### 3.1. ViCrit Task

Task description. For every training instance we start with an image I and its exhaustive, human-annotated caption C drawn from the PixMo-Cap dataset (Deitke et al., 2024), with an average caption length of 196 words. We then prompt GPT-4 (OpenAI, 2023) to select one object description o within that 200-word-length paragraph and perturb it into a visual hallucination o˜, such that the perturbed object is visually similar, semantically plausible, and without ambiguity. We also sample two examples from a small set of manually crafted in-context examples when prompting the LLM. The complete prompt is in Appendix. The desired

output is a minimally modified caption C˜ that differs from C by exactly one visual span (e.g., two words

- in Figure 2). We instruct for diverse types of selected objects o and resulted hallucination o˜, such as object substitution, attribute flip, scene-text error, relation swap, etc..

After data generation we task the model to identity the visual hallucination o˜ given the image I and caption C˜. A positive reward is given if predicted words matches ground-truth o˜. Because the reward depends purely on exact string match, it is deterministic and easy to scale in RL training.

Discussion on task difficulty. Perfectly performing the ViCrit task requires the model to perceive the entire visual scene, which is the same level of visual perception demanded of an “oracle” visual captioner that can exhaustively describe every image elements. Indeed, a flawless ViCrit critic could be repurposed into such an oracle by iteratively proposing refinements to an image caption. Thus, ViCrit imposes the same visual perceptual requirements as paragraph captioning, yet its single-span output is easily verifiable, enabling scalable outcome reward based reinforcement learning.

Data. We build the image I caption C˜ starting from all samples in the PixMo-Cap dataset (Deitke et al., 2024). Filtering out the invalid image URLs yields 384K image caption pairs. We then prompt LLM to create the visual hallucination o˜ and use it to replace the original object description o to create the minimally modified caption C˜. In the end, we collect 875K pairs of images and modified captions.

#### 3.2. Model Training

We use Qwen2.5-VL as the base VLM for experiments and finetune all model parameters via the ViCrit RL proxy task. We train the model with Group Relative Policy Optimization (GRPO) (Shao et al., 2024):

ℒGRPO = Ei[min(Ai⋅ρi, Ai⋅clip(ρi,1 − ϵ,1 + ϵ))], Ai = (ri − r¯), ρi = ππθ(yi∣x)

θold(yi∣x) The sample reward ri is computed based on a deterministic string matching between the injected visual hallucination string o˜ and the model prediction oˆ˜: ranswer = {

+1, if o˜ == oˆ˜, 0, otherwise.

We relax the string matching

such that the model is not penalized for copying additional words before or after the selected span o, as long as they are an exact copy from the original caption C. In addition to answer correctness reward, we also follow the standard practice to instruct the model to follow a specific prompt format, which group thoughts with special tokens <think>...</think> and final answers with special tokens \boxed{}. The format reward rformat is 1 if it correctly uses the special format tokens and 0 otherwise. The final reward for sample i is ri = 0.9 ∗ ranswer + 0.1 ∗ rformat.

## 4. ViCrit Benchmark

Motivated by the substantial gains yielded by reinforcement learning with the ViCrit task, we hypothesize that zero-shot ViCrit accuracy also correlates with a VLM’s perception capability and can therefore anchor a diagnostic benchmark. We thus present ViCrit-Bench, a high-quality, fine-grained, and highly challenging benchmark for hallucination detection. In this section, we first present the image domains and hallucination task categories defined in ViCrit-Bench. We then describe the human annotation procedure and dataset construction pipeline. Finally, we provide comprehensive statistics and distributional insights of ViCrit-Bench.

##### 1. Natural Image 2. Document 3. Scene Text Heavy Image 4. Abstract Image

|[Figure 17]|
|---|

|[Figure 18]|
|---|

|[Figure 19]|
|---|

|[Figure 20]|
|---|

|[Figure 21]|
|---|

Category: Color Original: yellow license plates

Category: Material Original: a solid white paper cup Hallucination: a solid white glass cup

Category: Object Original: A large, bloody sword Hallucination: A large, bloody crutch

|[Figure 22]|
|---|

|[Figure 23]|
|---|

|[Figure 24]|
|---|

Category: Text Original: Nike Air More Uptempo Hallucination: Nike Air More Upttempo

Hallucination:

orange license plates

|[Figure 25]|
|---|

|[Figure 26]|
|---|

|[Figure 27]|
|---|

|[Figure 28]|
|---|

|[Figure 29]|
|---|

Category: Spatial Original: In the upper left, a glimpse of green foliage Hallucination: In the upper right, a glimpse of green foliage

|[Figure 30]|
|---|

Category: Condition Original: arrow that goes up Hallucination: arrow that goes down

|[Figure 31]|
|---|

[Figure 32]

Category: Count Original: twelve red pillars Hallucination: eleven red pillars

Category: Shape Original: a yellow rectangle Hallucination: a yellow ellipse

- Figure 3: Data examples from ViCrit-Bench, which involve four image categories and eight visual hallucination types. We manually verify each image’s long caption, and carefully inject different kinds of proper visual hallucinations.

#### 4.1. Image Domains and Hallucination Categories

ViCrit-Bench partitions its images into four broad domains, each chosen to probe a complementary slice of visual perception: (1) Natural images: everyday photos of landscapes, animals, people, and objects captured in the wild; (2) Documents: images dominated by structured content such as tables, charts, plots, diagrams, or dense textual screenshots; (3) Scene-text–heavy images: images where scene text is appeared in the scene, such as street signs, memes, comic panels, and illustrative layouts; (4) Abstract images: images that do not directly depict real-world objects or scenes, but instead convey meaning through geometric shapes, symbols, color patterns, synthetic compositions, or artistic illustrations; these images emphasize structure, style, or conceptual abstraction rather than natural realism or textual content.

We note that the training data distribution from the PixMo-Cap dataset contains dominated natural images. The annotation on a subset of 4k randomly sampled PixMo-Cap images shows a category percentage of 59%, 10%, 24%, and 7% for image domains 1-4, respectively. Human annotators thus go through additional candidate images to find the proper sources for categories 2 and 4. Based on the image domains above, we then systematically categorized all visual hallucinations into eight distinct hallucination task types, defined as follows: (1) Count: evaluates whether the quantity of objects or elements is incorrectly described; (2) Material: assesses the model’s ability to accurately recognize the material composition of objects; (3) Spatial: determines whether the spatial configuration and relative positioning of entities are misrepresented; (4) Color: examines the consistency between the described and actual color attributes of visual elements; (5) Object: identifies cases where objects are incorrectly classified into wrong semantic

categories; (6) Condition: checks whether the physical state, dynamic action, or emotional expression of entities is appropriately conveyed; (7) Shape: measures the accuracy of describing the geometric structure or contour of objects; (8) Text: verifies whether embedded textual content within the image is correctly detected and interpreted.

#### 4.2. Annotation Pipeline

All samples in ViCrit-Bench originate from PixMo-Cap, whose 200-word, detailed captions provide a fertile substrate for hallucination synthesis. The construction pipeline proceeds in three stages.

- Stage 1: Image selection and caption sanitization. For the four image categories, we first employ OpenAIo3 model to perform an initial classification over the entire PixMo-Cap dataset, identifying candidate images that align with the definitions of natural images, document images, scene-text–heavy images, and abstract images. Subsequently, human annotators manually filter the candidates and select 20 images for each hallucination task with each image category, ensuring images strictly adhere to the domain-specific criteria.

Besides, given that some image captions in PixMo-Cap contain annotation errors, we first used o3 to review and automatically correct the captions, addressing the majority of semantic and factual issues. During the final annotation phase, each caption is individually reviewed and validated by human annotators to guarantee its accuracy and consistency. A total of four human annotators are involved in this process.

- Stage 2: Hallucination injection. Each image category is assigned to a dedicated annotator, who injects the selected hallucination types by surgically replacing a single noun phrase. This is done by replacing noun phrases with semantically plausible but misleading alternatives that are visually similar, as shown

in Figure 3. Each image is allowed to be modified with only one hallucination. For each image category, the number of images per hallucination task is capped at 20. However, in certain categories, some types of hallucinations may be inherently rare or difficult to instantiate—for example, material hallucinations in abstract images—which may result in fewer than 20 finalized examples for those specific tasks.

- Stage 3: Cross-validation. A final round of cross-validation by two independent annotator ensures the correctness and clarity of the injected hallucinations across all task types.

#### 4.3. Statistics

Following the aforementioned image selection and hallucination injection procedures, the final ViCrit-Bench contains 607 images, each paired with manually verified and edited captions containing a total of 607 fine-grained hallucination instances. The distribution of hallucination tasks across the dataset is illustrated in Figure 4. All hallucination task types, except for Material, are relatively balanced, each comprising around 13% the total instances. This reflects the comprehensive and well-balanced design of ViCrit-Bench. Due to its unique nature, the Material task appears mostly in first three categories, resulting in a lower overall proportion of 7.9%.

#### 4.4. Metric and Evaluation

[Figure 33]

Figure 4: Hallucination task distribution of ViCrit-Bench.

For each sample, we combine the image I and corrupted caption C˜ with a predefined evaluation prompt template (see Appendix A.2) to form the final evaluation model input. Given this prompt, a VLM must locate

- Table 1: Comparison between ViCrit-RL-7B and ViCrit-RL-72B with other open-source VLMs. After training on the ViCrit task using Qwen2.5-VL-7B-Instruct and Qwen2.5-VL-72B-Instruct as base models, hallucination rates are significantly reduced, achieving the best performance across all three hallucination benchmarks. Moreover, training on the ViCrit task substantially improves general vision-language performance. On eight general VL benchmarks, ViCrit-RL-72B achieves SOTA results on seven tasks, with the average accuracy increasing from 59.78 to 63.16.

Hallucination Benchmark General benchamrk

MathVision

MathVerse

MathVista

reasoning

MMStar

testmini

CHAIRs

MM-Vet

Charxiv

CHAIRi

MMMU

MMHal

Blind

mini

mini

Model Size Model

↑ Avg.

↓

↓

↑

↑

↑

↑

↑

↑

↑

↑

GPT-4o – – – 63.8 36.8 50.2 69.1 64.7 69.1 50.4 52.7 57.10

–

o1 – – – 73.9 58.2 57.0 78.2 – – 57.0 55.1 –

Molmo-7B-D-0924 36.7 6.0 3.03 54.1 19.5 23.2 40.2 52.6 59.2 43.3 30.8 40.38 LLaVA-OneVision-7B 35.0 5.5 3.12 63.2 17.4 26.2 48.8 61.7 57.5 40.1 31.3 43.28 InterVL2.5-8B 29.2 5.4 3.65 64.4 22.0 39.5 54.9 62.8 68.8 47.6 32.9 49.11 Qwen2.5-VL-7B-Instruct 28.0 5.1 3.74 67.8 23.6 44.5 50.6 61.7 66.0 49.3 41.4 50.61

7B

ViCrit-RL-7B 25.2 4.5 3.77 70.7 25.7 46.3 52.0 61.9 67.1 52.6 47.8 53.01 ∆ (Ours - Qwen2.5-7B) -2.8 -0.6 +0.03 +2.9 +2.1 +1.8 +1.4 +0.2 +1.1 +3.3 +6.4 +2.40

Molmo-72B-0924 28.8 5.7 3.54 61.1 24.7 30.9 48.3 58.4 65.5 46.9 35.2 46.38 LLaVA-OneVision-72B 27.4 4.9 3.71 67.5 29.3 39.1 56.8 66.1 63.7 49.6 38.2 51.29 InterVL2.5-78B 25.9 5.2 3.89 72.3 34.9 51.7 68.7 68.9 72.3 59.8 42.4 58.75 Qwen2.5-VL-72B-Instruct 26.4 4.8 3.82 74.8 35.2 53.3 63.4 68.4 76.3 61.3 45.5 59.78

72B

ViCrit-RL-72B 21.0 3.9 3.91 77.3 40.1 59.8 66.0 69.8 77.1 65.8 49.4 63.16 ∆ (Ours - Qwen2.5-72B) -5.4 -0.9 +0.09 +2.5 +4.9 +6.5 +2.6 +1.4 +0.8 +4.5 +3.9 +3.38

the hallucinated span inside C˜ as an open-ended QA task. A prediction is considered correct if the model’s prediction oˆ˜ exactly matches o˜. We take this string exact match accuracy as the metric for ViCrit-bench.

## 5. Experiments

#### 5.1. Effectiveness of ViCrit RL Training

We evaluate the effectiveness of ViCrit-based RL on various VL benchmarks. Through extensive comparisons with SOTA VLMs, we demonstrate the effectiveness of ViCrit as an RL training task and show that reinforcement fine-tuning on this task leads to general VL performance improvements.

Baseline VLMs. We start from the Qwen2.5-VL-7B-Instruct and Qwen2.5-VL-72B-Instruct checkpoints. Applying RL training with the ViCrit task produces our ViCrit-RL-7B and ViCrit-RL-72B, respectively. Qwen2.5 models thus constitute our primary models of interest as well as a fairly comparable baselines for ablation. For external comparison, we report benchmark results for three widely used open-source VLMs: Molmo (Deitke et al., 2024), LLaVA-OneVision (Li et al., 2024a), and InternVL2.5 (Chen et al., 2024c), including both their 7B-level and 72B-level variants. We also reference proprietary models include GPT-4o and o1. All training and evaluation is conducted with 8×80G A100 GPUs.

Evaluation benchmarks. (i) Hallucination mitigation: we first quantify ViCrit training’s impact on dedicated visual hallucination benchmarks. (ii) Broad VLM generalization: we then examine if the perceptual skills instilled by ViCrit-based RL transfer to general vision–language benchmarks.

- • (i) Hallucination mitigation. We adopt two widely used benchmarks: CHAIR (Rohrbach et al., 2018)

and MMHal (Sun et al., 2023). Specifically, CHAIR quantifies the proportion of hallucinated content in image captions. Following the setting in previous works (Zhou et al., 2024a,b), we randomly sample 500 images from the COCO Val2014 dataset and use prompts from the LLaVA-150k detailed description dataset, and

calculate CHAIR as follows: CHAIRI = ∣{∣{allhallucinatedmentionedobjectsobjects}∣}∣,CHAIRS = ∣{captions with∣{allhallucinatedcaptions}∣ objects}∣. MMHal serves as a complementary benchmark for evaluating hallucination in VLMs on VQA tasks. We employ GPT-4 as the scoring model to assess the hallucination severity in model responses.

- • (ii) Broad generalization. We use 8 widely adopted VLM benchmarks covering mathematical reasoning

(MathVista (Lu et al., 2024), MathVision (Wang et al., 2024a), MathVerse (Zhang et al., 2024a)), general knowledge (MMMU (Yue et al., 2024), MMStar (Chen et al., 2024b), MMVet (Yu et al., 2024a,b)), visual understanding (Blind (Rahmanzadehgervi et al., 2024)), and chart reasoning (ChartXiv (Wang et al., 2024d)).

Finding 1: RL with the ViCrit task significantly reduce visual hallucination.

The middle three columns of Table 1 compares our ViCrit-RL with other VLMs on visual hallucination benchmarks (Rohrbach et al., 2018, Sun et al., 2023). At the 7B scale, compared with the baseline Qwen2.5-VL7B, ViCrit-based RL training reduces CHAIRs and CHAIRi from 28.0 and 5.1 to 25.2 and 4.5, respectively. The MMHal increases to 3.77, which surpassing multiple 72B-level models. At the 72B scale, the improvement is even more pronounced: CHAIRs and CHAIRi reaches 21.0 and 3.9, and MMHal improves to 3.91, outperforming all SOTA VLMs across all three hallucination metrics. The consistency of the improvements across scales and benchmarks validates ViCrit’s effectiveness in reducing visual hallucination and improving perception across various description types.

Finding 2: RL with the ViCrit task improves VL performance in general.

Beyond curbing hallucinations, the right side of Table 1 shows that ViCrit-RL consistently lifts accuracy on the eight heterogeneous VLM benchmarks that constitute our “general vision–language” suite. Because the ViCrit proxy forces the model to verify every noun phrase against the image, it refines low-level perception and yields more faithful intermediate representations. These improvements appear to propagate to downstream reasoning tasks, with an averaged improvement of 2.4% on 7B scale and 3.4% on 72B scale. More importantly, the improvements generalize well onto the low-source training image domains. For example, the 72B model improves +4.9% on MathVision, +4.5% on VLMsareBlind and +3.9% on ChartXiv, despite math and abstract images only account for 7% of the PixMo-Cap training data, and 10% for chart and document images. This indicates that the model is not merely memorizing object lists but has learned a transferable strategy for “how to look” at an image before generating text. We provide a qualitative chain-of-thought analysis in Section 5.3 to probe this generalization pattern further.

#### 5.2. ViCrit-Bench Results

We benchmark a broad range of SOTA VLMs on our ViCrit-Bench, which probes eight fine-grained visual hallucination types across four image domains. For closed-source models, we evaluate OpenAI-GPT-series which includes 4o, o1 and o3, and Gemini-series which includes 2.0-Flash, 2.5-Flash, and 2.5-Pro. For opensource models, we follow the same experimental setup as Section 5.1 and evaluate Molmo, LLaVA-OneVision, InternVL2.5, and Qwen2.5-VL series. Table 2 shows that ViCrit-Bench is markedly challenging: the best model o3 reaches only 47.7% correctness, while the best open-source model Qwen2.5-VL-72B-Instruct achieves 42.4%.

Spatial hallucination emerges as the dominant failure mode, with the top-performing model achieving

- Table 2: Evaluation results of a range of closed-source and open-source VLMs on ViCrit-Bench. The results indicate that ViCrit-Bench poses a substantial challenge to current models—even the best-performing model, OpenAI-o3, achieves only 47.7 accuracy. After reinforcement fine-tuning on the ViCrit task, ViCrit-RL-72B achieves the highest accuracy of 43.0 over all opensouce models on the benchmark. Moreover, we observe a strong correlation between performance on ViCrit-Bench and the average accuracy on general visionlanguage tasks for open-source models. Models that score higher on ViCrit-Bench tend to perform better on general benchmarks, suggesting that ViCrit-Bench serves as a reliable indicator of overall reasoning and understanding capabilities.

Hallucination Type Image Type

HeavyImage

GeneralTask

SceneText

Document

Condition

Material

Abstract

Natural

Spatial

Object

Image

Image

Count

Shape

Color

Avg.

Text

Models

Overall

OpenAI-GPT-4o – 23.3 47.50 17.50 27.08 13.58 16.25 18.75 26.58 20.25 27.04 25.00 15.09 26.17 OpenAI-o1 – 45.8 60.00 48.10 64.58 25.93 43.75 40.51 57.69 32.91 53.46 39.29 42.14 47.26 OpenAI-o3 – 47.7 67.50 46.25 60.42 22.22 50.00 62.50 54.43 22.78 51.57 49.29 43.40 46.31 Gemini-2.0-Flash – 19.3 30.00 22.50 39.58 6.17 16.25 13.75 18.99 15.19 25.16 19.29 16.35 16.11 Gemini-2.5-Flash – 44.4 60.00 41.25 60.42 28.40 47.50 40.00 50.63 32.91 48.43 42.86 44.65 40.94 Gemini-2.5-Pro – 45.2 68.75 50.00 66.67 20.99 43.75 50.00 39.24 30.38 46.54 46.43 52.83 34.23

Molmo-7B-D-0924 40.48 9.6 25.00 8.75 10.42 9.88 5.00 3.75 6.33 7.59 5.66 13.57 8.81 10.74 LLaVA-OneVision-7B 43.28 12.4 20.00 11.25 10.42 12.35 6.25 15.00 12.65 10.13 17.61 7.86 9.43 14.09 InternVL-2.5-8B 49.11 20.0 26.25 11.25 25.00 22.22 12.50 15.00 30.38 18.99 27.04 15.00 13.21 24.16 Qwen-2.5-VL-7B 50.61 21.9 30.00 23.75 39.58 9.88 12.50 8.75 45.57 12.66 35.22 12.86 20.13 18.12

ViCrit-RL-7B 53.01 35.6 47.50 46.25 68.75 6.17 38.75 37.50 21.52 31.65 41.51 30.00 38.99 30.87 Molmo-72B 46.38 18.2 36.25 11.25 10.42 11.11 6.25 13.75 25.32 27.85 23.90 18.57 13.21 16.78 LLaVA-OneVision-72B 51.29 24.5 42.50 20.00 25.00 22.22 17.50 21.25 22.78 24.05 30.82 19.29 24.53 22.82 InternVL-2.5-78B 58.75 32.7 46.25 21.25 45.83 22.22 20.00 26.25 46.84 37.97 38.99 29.29 31.45 30.20 Qwen-2.5-VL-72B 59.78 42.4 57.50 36.25 58.33 28.40 26.25 46.25 54.43 37.97 47.17 40.71 44.03 36.91 ViCrit-RL-72B 63.16 43.0 60.00 48.75 70.83 17.28 40.00 56.25 25.32 39.24 47.80 41.43 44.65 37.58

only 28.40%, whereas object hallucination and material hallucination looks easier on paper. However, the higher number is because of an easier question subset on foreground objects, and it remains nontrivial to perform perfectly on any one of these eight classes (cf. the “corgi” object example in Figure 2.) With respect to image types, “Document Image” and “Abstract Image” are the most challenging ones, as nearly all models exhibited significantly lower accuracy on these two types compared to image types.

Furthermore, RL training with ViCrit task leads to substantial gains on ViCrit-Bench. ViCrit-RL-7B and ViCrit-RL-72B achieves an improved accuracy of 35.6% and 43.0%, respectively. Among four image categories, the largest gains occur on the Document and Abstract image domains, precisely the areas where baseline models struggle. This improvement also foreshadows the model’s enhanced performance on downstream benchmarks involving multimodal mathematical reasoning and chart understanding after ViCrit-based training, as shown in Table 1.

However, we observe a significant drop in accuracy for the Spatial and Text tasks after RL training. We attribute this to data imbalance in the construction of the training set. Furthermore, RL training led to substantial performance gains for the 7B model on ViCrit-Bench, whereas the improvements for the 72B model are relatively marginal. We hypothesize that this is due to the constructed training set being insufficiently challenging for Qwen-2.5-VL-72B, which already possesses strong visual perception capabilities. To further enhance the performance of the 72B model, more complex and demanding data may be required.

45

| | | | | |
|---|---|---|---|---|
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |

40

ViCrit-BenchPerformance

35

30

25

20

15

10

45 50 55 60

Average VLM Task Performance

- Figure 5: Correlation between average VLM task performance and ViCrit-Bench performance (Task Avg. and Overall columns in Table 2). Each point represents a different model, and the fitted linear regression line highlights a strong positive relationship, indicating that better ViCrit-Bench results are associated with higher stronger VLM capabilities.

Finding 3: ViCrit-Bench results well foreshadow the VLM performance.

Beyond the raw results, Table 2 exposes a monotonic link between the ViCrit-Bench results and the general VLM performance. Models rank in precisely the same order in the ViCrit-Bench Overall column, as in the averaged VLM performance column quoted from Table 1. Figure 5 demonstrates a strong positive linear correlation between average VLM task performance and ViCrit-Bench scores. ViCrit-Bench scores rise almost linearly with a model’s average performance across eight vision-language tasks (r = 0.96), showing that the benchmark effectively evaluates the visual perception, and is a strong proxy for overall VLM capability. In the 7B class, performance rises step-wise from Molmo-7B through LLaVA-OneVision-7B, InternVL 2.5-8B, and Qwen2.5-VL-7B, to ViCrit-RL-7B that tops every metric. The pattern repeats at the 72B scale where ViCrit-RL-72B achieves the best performance on both ViCrit-bench and general VLM evaluation.

This finding echoes our original motivation in building ViCrit-bench, with the hypothesis that ViCrit accuracy could foreshadow a VLM’s perception capability as well as the overall multimodal performance. Furthermore, the consistency of the ordering across scales suggests that ViCrit-RL’s better hallucination performance in Finding 1 is not a by-product of merely training on hallucination detection. Rather, models that learn to “cross-check textual claims against visual evidence” via ViCrit also perceive and reason better on general VL problems, such as chart, math, and abstract images.

#### 5.3. Qualitative Results

We showcase two representative cases (Figure 6) that reveal how training on the ViCrit task sharpens visual perception and, consequently, improving VLM performance. Example 1. ViCrit-RL-72B is able to accurately identify all objects in the image in a clockwise order, capturing detailed attributes such as color and shape, and successfully making the correct calculation. Example 2. ViCrit-RL-72B correctly identifies all relevant

###### Qwen-2.5-VL-72B-Instruct

Question: Subtract all small blue rubber objects. Subtract all brown shiny balls. How many objects are left?

To solve the problem, we need to follow these steps:\n\n1. Identify all objects in the image.\n2. Identify the small blue rubber objects and subtract them.\n3. Identify the brown shiny balls and subtract them.\n4. Count the remaining objects.\n\nStep 1: Identify all objects in the image.\n- There are 9 objects in total: 2 cubes, 1 red cylinder, 1 yellow sphere, 1 green sphere, 1 blue sphere, 1 purple sphere, 1 brown shiny sphere, and 1 cyan cube.\n\nStep 2: Identify the small blue rubber objects and subtract them.\n- There is 1 small blue rubber sphere.\n- Subtracting this leaves us with 8 objects.\n\nStep 3: Identify the brown shiny balls and subtract them.\n- There is 1 brown shiny sphere.\nSubtracting this leaves us with 7 objects.\n\nStep 4: Count the remaining objects.\n- After subtracting the small blue rubber sphere and the brown shiny sphere, we are left with 7 objects. The final answer is: 7

[Figure 34]

###### ViCrit-RL-72B

<think>\nTo solve the problem, I need to follow these steps:\n\n1. Identify all objects in the image.\n2. Subtract all small blue rubber objects.\n3. Subtract all brown shiny balls.\n4. Count the remaining objects.\n\nStep 1: Identify all objects:\n- Small yellow metallic cube\n- Small purple rubber sphere\n- Small red rubber cylinder\n- Large cyan metallic cube\n- Small brown shiny sphere\n- Large yellow rubber sphere\n- Small green rubber sphere\n- Small blue rubber sphere\n\nStep 2: Subtract all small blue rubber objects:\n- There is 1 small blue rubber sphere.\n- Removing this leaves 7 objects.\n\nStep 3: Subtract all brown shiny balls:\n- There is 1 small brown shiny sphere.\n- Removing this leaves 6 objects.\n\nStep 4: Count the remaining objects:\n- Small yellow metallic cube\n- Small purple rubber sphere\n- Small red rubber cylinder\n- Large cyan metallic cube\n- Large yellow rubber sphere\n- Small green rubber sphere\n\nThere are 6 objects left.\n</think> \boxed{6}

###### Qwen-2.5-VL-72B-Instruct

Question: Can you find the missing shape in this picture puzzle?

To find the missing shape in the picture puzzle, let‘s analyze the pattern:\n\n1. The first shape is a black hexagon.\n2. The second shape is a white hexagon.\n3. The fourth shape is a white trapezoid.\n\nThe pattern seems to alternate between black and white shapes, and the shapes themselves are changing. The sequence of shapes is hexagon, hexagon, and then trapezoid. \n\nGiven this pattern, the missing shape should be a black trapezoid to maintain the alternating color pattern and the sequence of shapes.\n\nLooking at the options:\n- A is a black pentagon.\n- B is a white pentagon.\n- C is a black hexagon.\n- D is a black trapezoid.\n\nThe correct option is D, as it is a black trapezoid. The final answer is: D

[Figure 35]

###### ViCrit-RL-72B

<think>\nTo find the missing shape in the picture puzzle, let's analyze the pattern:\n\n1. The first shape is a black heptagon (7-sided polygon).\n2. The second shape is a white

hexagon (6-sided polygon).\n3. The third shape is missing.\n4. The fourth shape is a white trapezoid.\n\nWe need to determine the pattern in terms of the number of sides and the color of the shapes.\n\n- The heptagon has 7 sides and is black.\n- The hexagon has 6 sides and is white.\n- The trapezoid has 4 sides and is white.\n\nThe pattern seems to be alternating between black and white shapes, and the number of sides is decreasing. Following this pattern:\n\n- The next shape should have 5 sides (since 7, 6, 5, 4 is a

decreasing sequence).\n- The color should be black (since the sequence is black, white, black, white).\n\nLooking at the options:\n- A is a black pentagon (5 sides).\n- B is a white pentagon (5 sides).\n- C is a black hexagon (6 sides).\n- D is a black trapezoid (4 sides).\n\nThe correct shape is a black pentagon, which is option A.\n</think> \boxed{A}

- Figure 6: Two examples demonstrate the behavioral differences between models before and after training with the ViCrit task. It can be seen that ViCrit-RL-72B pays closer attention to image details and arrives at the correct final reasoning through its enhanced perception capabilities.

visual details—including colors and the number of edges of each object—and uses this information to derive the correct answer. In contrast, Qwen2.5-VL-Instruct fails to capture the complete visual content due to its limited perception ability, leading to incorrect reasoning. These examples demonstrate that training on ViCrit task significantly improves visual perception, which is a crucial foundation for enhancing the VLM performance.

## 6. Conclusion

We have presented ViCrit, an RL proxy task that trains VLMs to pinpoint fine-grained, synthetically injected visual hallucinations in paragraph-length captions. Because each targeted span is unambiguously verifiable, ViCrit provides a challenging yet noise-free reward signal that compels models to internalize stronger perceptual strategies, yielding consistent gains across a broad suite of benchmarks. Furthermore, we release ViCrit-Bench, a carefully curated dataset that enables rigorous evaluation of VLM perception. We hope this new task will spark further breakthroughs in multimodal RL, from standalone VLMs to end-to-end-trained tool-augmented multimodal agents.

## Acknowledgment

Wang, Liang, Zhou, Liu, and Huang are supported by DARPA Transfer from Imprecise and Abstract Models to Autonomous Technologies (TIAMAT) 80321, DARPA HR001124S0029-AIQ-FP-019, DOD-AFOSR-Air Force Office of Scientific Research under award number FA9550-23-1-0048, National Science Foundation NSF-IIS-2147276 FAI, National Science Foundation NAIRR240045, National Science Foundation TRAILS Institute (2229885). Private support was provided by Peraton.

## References

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C Lawrence Zitnick, and Devi Parikh. Vqa: Visual question answering. In Proceedings of the IEEE international conference on computer vision, pages 2425–2433, 2015.

Anas Awadalla, Le Xue, Manli Shu, An Yan, Jun Wang, Senthil Purushwalkam, Sheng Shen, Hannah Lee, Oscar Lo, Jae Sung Park, et al. Blip3-kale: Knowledge augmented large-scale dense captions. arXiv preprint arXiv:2411.07461, 2024.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn. openai. com/papers/dall-e-3. pdf, 2(3):8, 2023.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Liang Chen, Lei Li, Haozhe Zhao, Yifan Song, and Vinci. R1-v: Reinforcing super generalization ability in vision language models with less than $3. https://github.com/Deep-Agent/R1-V, 2025. Version v0.2 released 3 Feb 2025.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. In European Conference on Computer Vision, pages 370–387. Springer, 2024a.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024b.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024c.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. arXiv preprint arXiv:2409.17146, 2024.

Yihe Deng, Pan Lu, Fan Yin, Ziniu Hu, Sheng Shen, Quanquan Gu, James Y Zou, Kai-Wei Chang, and Wei Wang. Enhancing large vision language models with self-training on image comprehension. Advances in Neural Information Processing Systems, 37:131369–131397, 2024.

Yihe Deng, Hritik Bansal, Fan Yin, Nanyun Peng, Wei Wang, and Kai-Wei Chang. Openvlthinker: An early exploration to complex vision-language reasoning via iterative self-improvement, 2025. URL https: //arxiv.org/abs/2503.17352.

Karan Desai and Justin Johnson. Virtex: Learning visual representations from textual annotations. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11162–11173, 2021.

Xingyu Fu, Minqian Liu, Zhengyuan Yang, John Corring, Yijuan Lu, Jianwei Yang, Dan Roth, Dinei Florencio, and Cha Zhang. Refocus: Visual editing as a chain of thought for structured image understanding. arXiv preprint arXiv:2501.05452, 2025. URL https://arxiv.org/abs/2501.05452.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, and many others. Deepseekr1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948,

### 2025. URL https://arxiv.org/abs/2501.12948.

Yunzhuo Hao, Jiawei Gu, Huichen Will Wang, Linjie Li, Zhengyuan Yang, Lijuan Wang, and Yu Cheng. Can mllms reason in multimodality? emma: An enhanced multimodal reasoning benchmark. arXiv preprint arXiv:2501.05444, 2025.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset, 2021. Includes AIME and other competition-level problems.

Wenxuan Huang, Bohan Jia, Zijie Zhai, Shaosheng Cao, Zheyu Ye, Fei Zhao, Zhe Xu, Yao Hu, and Shaohui Lin. Vision-r1: Incentivizing reasoning capability in multimodal large language models, 2025. URL https://arxiv.org/abs/2503.06749.

Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Bowen Jin, Chulin Xie, Jiawei Zhang, Kashob Kumar Roy, Yu Zhang, Zheng Li, Ruirui Li, Xianfeng Tang, Suhang Wang, Yu Meng, and Jiawei Han. Graph chain-of-thought: Augmenting large language models by reasoning on graphs. arXiv preprint arXiv:2404.07103, 2024. URL https://arxiv.org/abs/2404. 07103.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022.

Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. Mawps: A math word problem repository. In Proceedings of the 2016 conference of the north american chapter of the association for computational linguistics: human language technologies, pages 1152–1157, 2016.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024a.

Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, Jianfeng Gao, et al. Multimodal foundation models: From specialists to general-purpose assistants. Foundations and Trends® in Computer Graphics and Vision, 16(1-2):1–214, 2024b.

Long Lian, Yifan Ding, Yunhao Ge, Sifei Liu, Hanzi Mao, Boyi Li, Marco Pavone, Ming-Yu Liu, Trevor Darrell, Adam Yala, et al. Describe anything: Detailed localized image and video captioning. arXiv preprint arXiv:2504.16072, 2025.

Hunter Lightman, Karl Cobbe, Vineet Kosaraju, Yura Burda, Harri Edwards, Jan Leike, and Ilya Sutskever. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023. URL https://arxiv.org/abs/2305. 20050.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306, 2024.

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR), 2024.

Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge, 2019. URL https://arxiv.org/abs/1906. 00067.

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, Ping Luo, Yu Qiao, Qiaosheng Zhang, and Wenqi Shao. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025a. URL https://arxiv.org/abs/2503.07365.

Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, et al. Mm-eureka: Exploring visual aha moment with rule-based large-scale reinforcement learning. arXiv preprint arXiv:2503.07365, 2025b.

Samuel Miserendino, Michele Wang, Tejal Patwardhan, and Johannes Heidecke. Swe-lancer: Can frontier llms earn $1 million from real-world freelance software engineering? arXiv preprint arXiv:2502.12115, 2025.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025.

Minheng Ni, Zhengyuan Yang, Linjie Li, Chung-Ching Lin, Kevin Lin, Wangmeng Zuo, and Lijuan Wang. Point-rft: Improving multimodal reasoning with visually grounded reinforcement finetuning. arXiv preprint arXiv:2505.19702, 2025.

Mathematical Association of America. American invitational mathematics examination (aime) 2024: Competition problems. https://artofproblemsolving.com/wiki/index.php/2024_AIME_I_Problems,

2024. Accessed 13 May 2025.

### OpenAI. Gpt-4v(ision) system card. 2023. URL https://api.semanticscholar.org/CorpusID: 263218031.

Arkil Patel, Satwik Bhattamishra, and Navin Goyal. Are nlp models really able to solve simple math word problems? arXiv preprint arXiv:2103.07191, 2021.

Yingzhe Peng, Gongrui Zhang, Miaosen Zhang, Zhiyuan You, Jie Liu, Qipeng Zhu, Kai Yang, Xingzhong Xu, Xin Geng, and Xu Yang. Lmm-r1: Empowering 3b lmms with strong reasoning abilities through two-stage rule-based rl. arXiv preprint arXiv:2503.07536, 2025. URL https://arxiv.org/abs/2503.07536.

Jordi Pont-Tuset, Jasper Uijlings, Soravit Changpinyo, Radu Soricut, and Vittorio Ferrari. Connecting vision and language with localized narratives. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part V 16, pages 647–664. Springer, 2020.

Pooyan Rahmanzadehgervi, Logan Bolton, Mohammad Reza Taesiri, and Anh Totti Nguyen. Vision language models are blind. In Proceedings of the Asian Conference on Computer Vision, pages 18–34, 2024.

Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. Object hallucination in image captioning. arXiv preprint arXiv:1809.02156, 2018.

Daniel Rose, Vaishnavi Himakunthala, Andy Ouyang, Ryan He, Alex Mei, Yujie Lu, Michael Saxon, Chinmay Sonar, Diba Mirza, and William Yang Wang. Visual chain of thought: bridging logical gaps with multimodal infillings. arXiv preprint arXiv:2305.02317, 2023.

Mert Bulent Sariyildiz, Julien Perez, and Diane Larlus. Learning visual representations with caption annotations. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part VIII 16, pages 153–170. Springer, 2020.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019.

Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, et al. Aligning large multimodal models with factually augmented rlhf. arXiv preprint arXiv:2309.14525, 2023.

Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310–87356, 2024.

Michael Tschannen, Manoj Kumar, Andreas Steiner, Xiaohua Zhai, Neil Houlsby, and Lucas Beyer. Image captioners are scalable vision learners too. Advances in Neural Information Processing Systems, 36:46830– 46855, 2023.

Jonathan Uesato, Nate Kushman, Ramana Kumar, Francis Song, Noah Siegel, Lisa Wang, Antonia Creswell, Geoffrey Irving, and Irina Higgins. Solving math word problems with process-and outcome-based feedback. arXiv preprint arXiv:2211.14275, 2022.

Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, and Lijuan Wang. Git: A generative image-to-text transformer for vision and language. arXiv preprint arXiv:2205.14100, 2022a.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024a. URL https://openreview. net/forum?id=QWTCcxMpPA.

Peiyi Wang, Lei Li, Zhihong Shao, Rui Xia, Damai Dai, Yifei Li, De Li Chen, Yu Wu, and Zhifang Sui. Math-shepherd: Verify and reinforce llms step-by-step without human annotations. arXiv preprint arXiv:2312.08935, 2023. URL https://arxiv.org/abs/2312.08935.

Xiyao Wang, Jiuhai Chen, Zhaoyang Wang, Yuhang Zhou, Yiyang Zhou, Huaxiu Yao, Tianyi Zhou, Tom Goldstein, Parminder Bhatia, Furong Huang, et al. Enhancing visual-language modality alignment in large vision language models via self-improvement. arXiv preprint arXiv:2405.15973, 2024b.

Xiyao Wang, Zhengyuan Yang, Linjie Li, Hongjin Lu, Yuancheng Xu, Chung-Ching Lin, Kevin Lin, Furong Huang, and Lijuan Wang. Scaling inference-time search with vision value model for improved visual comprehension. arXiv preprint arXiv:2412.03704, 2024c.

Xiyao Wang, Zhengyuan Yang, Chao Feng, Hongjin Lu, Linjie Li, Chung-Ching Lin, Kevin Lin, Furong Huang, and Lijuan Wang. Sota with less: Mcts-guided sample selection for data-efficient visual reasoning self-improvement. arXiv preprint arXiv:2504.07934, 2025.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022b. URL https://arxiv.org/abs/2203.11171.

Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, et al. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. Advances in Neural Information Processing Systems, 37:113569–113697, 2024d.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Jialian Wu, Jianfeng Wang, Zhengyuan Yang, Zhe Gan, Zicheng Liu, Junsong Yuan, and Lijuan Wang. Grit: A generative region-to-text transformer for object understanding. In European Conference on Computer Vision, pages 207–224. Springer, 2024a.

Wenshan Wu, Shaoguang Mao, Yadong Zhang, Yan Xia, Li Dong, Lei Cui, and Furu Wei. Mind’s eye of llms: Visualization-of-thought elicits spatial reasoning in large language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024b.

Yuxi Xie, Anirudh Goyal, Wenyue Zheng, Min-Yen Kan, Timothy Lillicrap, Kenji Kawaguchi, and Michael Q. Shieh. Monte carlo tree search boosts reasoning via iterative preference learning. arXiv preprint arXiv:2405.00451, 2024. URL https://arxiv.org/abs/2405.00451.

Tianyi Xiong, Xiyao Wang, Dong Guo, Qinghao Ye, Haoqi Fan, Quanquan Gu, Heng Huang, and Chunyuan Li. Llava-critic: Learning to evaluate multimodal models. arXiv preprint arXiv:2410.02712, 2024.

Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v (ision). arXiv preprint arXiv:2309.17421, 9(1):1, 2023a.

Zhengyuan Yang, Jianfeng Wang, Linjie Li, Kevin Lin, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. Idea2img: Iterative self-refinement with gpt-4v (ision) for automatic image design and generation. arXiv preprint arXiv:2310.08541, 2023b.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36:11809–11822, 2023.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities, 2024a. URL https: //arxiv.org/abs/2308.02490.

Weihao Yu, Zhengyuan Yang, Lingfeng Ren, Linjie Li, Jianfeng Wang, Kevin Lin, Chung-Ching Lin, Zicheng Liu, Lijuan Wang, and Xinchao Wang. Mm-vet v2: A challenging benchmark to evaluate large multimodal models for integrated capabilities. arXiv preprint arXiv:2408.00765, 2024b.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multidiscipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of CVPR, 2024.

Renrui Zhang, Dongzhi Jiang, Yichi Zhang, Haokun Lin, Ziyu Guo, Pengshuo Qiu, Aojun Zhou, Pan Lu, Kai-Wei Chang, Peng Gao, et al. Mathverse: Does your multi-modal llm truly see the diagrams in visual math problems? arXiv preprint arXiv:2403.14624, 2024a.

Ruohong Zhang, Bowen Zhang, Yanghao Li, Haotian Zhang, Zhiqing Sun, Zhe Gan, Yinfei Yang, Ruoming Pang, and Yiming Yang. Improve vision language model chain-of-thought reasoning. arXiv preprint arXiv:2410.16198, 2024b.

Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-ofthought reasoning in language models. Trans. Machine Learning Research, 2024c.

Sage Zheng, Darwin Hou, Yujie Pan, Xinyun Li, Amanpreet Singh, Dawn Song, Percy Liang, Jason Wei, et al. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023.

Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, and Yuwen Xiong. Easyr1: An efficient, scalable, multi-modality rl training framework. https://github.com/hiyouga/EasyR1, 2025.

Yiyang Zhou, Chenhang Cui, Rafael Rafailov, Chelsea Finn, and Huaxiu Yao. Aligning modalities in vision large language models via preference fine-tuning. arXiv preprint arXiv:2402.11411, 2024a.

Yiyang Zhou, Zhiyuan Fan, Dongjie Cheng, Sihan Yang, Zhaorun Chen, Chenhang Cui, Xiyao Wang, Yun Li, Linjun Zhang, and Huaxiu Yao. Calibrated self-rewarding vision language models. arXiv preprint arXiv:2405.14622, 2024b.

Appendix

- A. Prompts used in experiments

- A.1. Prompt for Training Data Generation We provide the prompt used for generating ViCrit task training data in Table 5.
- A.2. Prompt for ViCrit-Bench Evaluation We provide the prompt used for ViCrit-Bench evaluation in Table 3.

Table 3: Prompt template used for ViCrit-Bench evaluation.

Prompt Template: You are provided with an image and the description corresponding to this image. There is one hallucination in this description. Find out the hallucination phase and answer with the hallucination phase directly in a list. Your output should only be a list that contains the hallucination phase you find. Description:

- B. Comparison with SFT

In this section, we perform SFT on Qwen-2.5-VL-7B and 72B using 900k captioning samples from PixMo-Cap, and compare the results with ViCrit-RL models trained using the same amount of data through ViCrit task RFT. As shown in Table 4, we find that although SFT significantly reduced hallucination in VLMs, it do not lead to notable performance improvements on general benchmarks—in fact, the 7B model even shows a performance drop. This highlights the effectiveness of ViCrit task RFT, which not only reduces hallucinations but also generalizes well to enhance VLM performance on general reasoning tasks.

- Table 4: Comparison between ViCrit-RL and ViCrit-RL with using same captioning data for SFT. We find that although hallucinations in the VLM are significantly reduced after SFT, the performance improvement is difficult to generalize to general tasks.

Hallucination Benchmark General benchamrk

MathVision

MathVerse

MathVista

reasoning

MMStar

testmini

CHAIRs

MM-Vet

Charxiv

CHAIRi

MMMU

MMHal

Blind

mini

mini

Model

↑ Avg.

↓

↓

↑

↑

↑

↑

↑

↑

↑

↑

Qwen2.5-VL-7B-Instruct 28.0 5.1 3.74 67.8 23.6 44.5 50.6 61.7 66.0 49.3 41.4 50.61 Qwen2.5-VL-7B-CapSFT 25.5 4.4 3.78 67.4 20.1 44.3 52.1 53.4 64.7 47.3 38.0 48.41 ViCrit-RL-7B 25.2 4.5 3.77 70.7 25.7 46.3 52.0 61.9 67.1 52.6 47.8 53.01

∆ (Ours - Qwen2.5-7B) -2.8 -0.6 +0.03 +2.9 +2.1 +1.8 +1.4 +0.2 +1.1 +3.3 +6.4 +2.40

Qwen2.5-VL-72B-Instruct 26.4 4.8 3.82 74.8 35.2 53.3 63.4 68.4 76.3 61.3 45.5 59.78 Qwen2.5-VL-72B-CapSFT 21.6 3.6 3.89 76.1 34.8 57.9 65.3 68.9 76.5 63.0 44.7 60.78 ViCrit-RL-72B 21.0 3.9 3.91 77.3 40.1 59.8 66.0 69.8 77.1 65.8 49.4 63.16

∆ (Ours - Qwen2.5-72B) -5.4 -0.9 +0.09 +2.5 +4.9 +6.5 +2.6 +1.4 +0.8 +4.5 +3.9 +3.38

- Table 5: Prompt used for training data generation.

You are a helpful assistant designed to manipulate text with precision. Your task is as follows:

- 1. Identify all noun phrases in a given paragraph. A noun phrase consists of a noun and its modifiers (e.g., "the wooden bridge," "a flock of birds"). Noun phrase is two to five words long. Do not output a list of multiple noun phrases.
- 2. Randomly select one noun phrase from the list, it can be small background objects, scene text, foreground objects. Try to select scene text and small background objects more often when possible.
- 3. Replace the chosen noun phrase with another phrase that is visually similar, such as changing the object attributes, replacing the object with a visually similar noun, or adding and removing characters within the scene text. The replacement should be visually similar but not identical to the original phrase. Be creative and don’t always focus on the most obvious or common replacements such as color.
- 4. However, the replacement should introduce clear change, such that it is impossible to be ambiguous. The change should be directly related to image and be a visual description. Do not only change words to its synonyms or make ambigious changes. Do not merely change words to its synonyms. Do not merely change words to its synonyms. Do not merely change words to its synonyms.
- 5. Ensure the edited paragraph is still be a plausible image description, and the change is not too obvious.
- 6. Group the original phrase in <Before>original</Before>, and changed phrase in <After>changed</After>. <Caption> is used to give input caption and should not be generated. Perform this transformation accurately and naturally. Here are some examples:

- 1. <Caption>This image appears to be a screenshot taken from an iPhone displaying the interface of a food delivery app, likely DoorDash, around the Chicago and Gary, Indiana area. The top of the screen indicates the time as 2:30 PM, with the phone connected to an LTE network. The battery icon suggests a low battery level of 15-20 Central to the image is a map highlighting various regions with color codes: red areas represent high traffic or demand, likely meaning those areas are "busy" for delivery drivers, as indicated by a red text banner. Lighter red and green sections represent varying levels of demand. At the top of the image, a black banner labeled "Promos" is displayed, accompanied by a blue notification bell icon with the number two beside it, indicating two notifications. The bottom of the screen shows a black navigation bar. It contains options for "Dash," "Schedule," "Account," "Ratings," and "Earnings." The "Dash" option is highlighted in red, suggesting it is currently selected. Centrally located in this bar is a red "Dash Now" button, implying that the user can begin delivering immediately. An additional black banner, located just above the navigation bar, reads "In... Hammond." Overall, this detailed caption gives a comprehensive idea of the app’s functionality, likely indicating areas of high demand where food delivery services are needed the most.</Caption> <Before>a low battery level of 15-20%</Before> <After>a high battery level of 75-80%</After>
- 2. <Caption>The image depicts a screenshot from a strategy video game with a third-person, aerial view. The central character, named Anselm, navigates through a complex, industrial-style building that evokes the aesthetic of games like Metal Gear. The environment is dark and futuristic, with certain areas illuminated, revealing various paths and stairs. The top left corner displays the yellow text “Instructor Eastwood,” alongside a graph-like design. The upper center features game-related instructions in white text stating “Defensive Measures – Use Range to Your Advantage,” with the notations “5” and “5.1” accompanying the instructions. Additionally, a map of the area is situated in the lower left corner, while the bottom right corner features interactive elements or a key potentially indicating available weapons. The overall scene suggests a mission-focused gameplay scenario requiring strategic maneuvering and tactical decision-making.</Caption> <Before>text “Instructor Eastwood,”</Before> <After>text “Instructor Westwood,”</After>
- 3. <Caption>The image depicts a three-dimensional panoramic view of a conference room where a business meeting is taking place. The setting is a typical meeting room with white walls, fluorescent lighting, and windows equipped with blinds on some, including wooden slats on one. At the center of the room is a round, yellow table that appears slightly distorted due to the panoramic effect. On this table, there are various items, including a box of tissues, a white mug, a teacup, and pamphlets. Surrounding the table, seated in a circle, are eight individuals. They appear to be a mixed group of men and women, predominantly of Asian descent, and are dressed in a variety of attire ranging from business to casual. All attendees are wearing name tags on the left side of their chests, indicating their participation in the meeting. Their seating arrangement includes black chairs with green backs, and each person either has their hands folded in their laps or is holding something, possibly a drink. From left to right, the attendees include a woman in a peach-colored t-shirt with writing, a man in a blue shirt, a woman in a gray sweater, a woman in a green shirt, another woman in a blazer, an empty chair, a man in a yellowish shirt, a man wearing a ball cap, and a man in a blue and green jacket. One notable aspect is that the meeting environment, though professional, is quite understated with minimalistic decor and standard conference room furnishings.</Caption> <Before>eight individuals</Before> <After>seven individuals</After> Here is the input caption: <Caption>{CAPTION}</Caption>

23

