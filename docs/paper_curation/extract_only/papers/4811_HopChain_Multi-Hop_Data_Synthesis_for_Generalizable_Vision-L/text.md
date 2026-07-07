[Figure 1]

March 19, 2026

# arXiv:2603.17024v2[cs.CV]19Mar2026

## HopChain: Multi-Hop Data Synthesis for Generalizable Vision-Language Reasoning

Shenzhi Wang1,2,∗, Shixuan Liu1,∗, Jing Zhou1, Chang Gao1, Xiong-Hui Chen1, Binghai Wang1, An Yang1, Shiji Song2, Bowen Yu1, ,†, Gao Huang2, , Junyang Lin1 1 Qwen Team, Alibaba Inc. 2 LeapLab, Tsinghua University

#### Abstract

Vision-language models (VLMs) show strong multimodal capabilities but still struggle with fine-grained vision-language reasoning. We find that long chain-of-thought (CoT) reasoning exposes diverse failure modes, including perception, reasoning, knowledge, and hallucination errors, which can compound across intermediate steps. However, most existing vision-language data used for reinforcement learning with verifiable rewards (RLVR) does not involve complex reasoning chains that rely on visual evidence throughout, leaving these weaknesses largely unexposed. We therefore propose HopChain, a scalable framework for synthesizing multi-hop vision-language reasoning data for RLVR training of VLMs. Each synthesized multi-hop query forms a logically dependent chain of instance-grounded hops, where earlier hops establish the instances, sets, or conditions needed for later hops, while the final answer remains a specific, unambiguous number suitable for verifiable rewards. We train Qwen3.5-35B-A3B and Qwen3.5-397B-A17B under two RLVR settings: the original data alone, and the original data plus HopChain’s multi-hop data, and compare them across 24 benchmarks spanning STEM and Puzzle, General VQA, Text Recognition and Document Understanding, and Video Understanding. Although this multi-hop data is not synthesized for any specific benchmark, it improves 20 of 24 benchmarks on both models, indicating broad and generalizable gains. Consistently, replacing full chained queries with half-multi-hop or single-hop variants reduces the average score across five representative benchmarks from 70.4 to 66.7 and 64.3, respectively. Notably, multi-hop gains peak in long-CoT vision-language reasoning, exceeding 50 points in the ultra-long-CoT regime. These experiments establish HopChain as an effective, scalable framework for synthesizing multi-hop data that improves generalizable vision-language reasoning.

#### 1 Introduction

Vision-language models (VLMs) have achieved impressive performance across a diverse set of multimodal benchmarks by integrating visual encoders with large language models (Liu et al., 2023b;a; Bai et al., 2023; 2025b; Chen et al., 2024c; OpenAI, 2023; Gemini Team et al., 2023). Recent advances in reinforcement learning with verifiable rewards (RLVR) have further improved VLMs’ reasoning abilities by training them to produce step-by-step chain-of-thought solutions for questions with objectively verifiable answers (Shao et al., 2024; DeepSeek-AI, 2025). Despite this progress, a critical gap remains: VLMs frequently struggle with tasks that require fine-grained, multi-step vision-language reasoning, where the correct answer depends on carefully attending to multiple visual elements and their relationships within an image (Bigverdi et al., 2025; Ye et al., 2025; Jiang et al., 2025).

A natural question arises: what prevents current VLMs from performing robust vision-language reasoning? In Section 3, we analyze a central challenge: VLMs exhibit diverse and compounding failure modes during long chain-of-thought (CoT) reasoning, as longer reasoning chains may cause models to attend less faithfully to the image, execute incorrect intermediate reasoning steps, rely on incomplete knowledge, or produce hallucinated and weakly grounded intermediate content that then compounds across subsequent steps. Related failure patterns have also been reported in prior analyses of amplified multimodal hallucination, evidential drift, object hallucination, visual illusion, and image-context reasoning errors (Liu et al., 2025; Luo et al., 2025; Rohrbach et al., 2018; Guan et al., 2024; Leng et al., 2024). Critically, as depicted in Figure 1(b), most existing vision-language training data does not involve particularly complex reasoning chains that depend on visual evidence throughout the process. As a result, these long-CoT weaknesses remain largely unexposed during training. This observation suggests that relying on, or simply expand-

∗Equal contribution : Corresponding authors †: Project lead Emails: wangshenzhi99@gmail.com, liushixuan66@gmail.com, yubowen.ph@gmail.com, gaohuang@tsinghua.edu.cn

(a) HopChain Framework for Multi-Hop Data Synthesis (b)

Typical training data: lacks complex reasoning chains relying on visual evidence throughout.

Evaluated with hard problems: minor errors in steps compound into final mistakes.

[Figure 2]

[Figure 3]

Identify Categories

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

Segment Instances

[Figure 8]

[Figure 9]

[Figure 10]

Sampled Combinations with Spatial Localization

Behind Right of

[Figure 11]

[Figure 12]

[Figure 13]

White Paper Black Sheep Doll

(c)

Synthesized multi-hop data: enforces dependencylinked hops with repeated visual grounding.

Evaluated with hard problems: reduces error cascades and yields more reliable final answers.

Multi-Hop Query Generation

Identify a sheep

Count its eyes

[Figure 14]

[Figure 15]

PerceptionLevel Hop

[Figure 16]

[Figure 17]

[Figure 18]

(Single-object perception)

(Multi-object perception)

Any readable text?

InstanceChain Hop

Behind

[Figure 19]

[Figure 20]

[Figure 21]

Ground-Truth Annotation

Difficulty

[Figure 22]

+ Calibration

- Figure 1: Overview of HopChain and the motivation for multi-hop vision-language reasoning data. (a) HopChain synthesizes multi-hop data through four stages: category identification, instance segmentation, multi-hop query generation, and ground-truth annotation with difficulty calibration. (b) Typical visionlanguage training data often does not require complex reasoning chains that rely on visual evidence throughout; on hard problems, minor mistakes introduced at intermediate long-CoT steps can compound into final errors. (c) In contrast, the multi-hop data synthesized by HopChain forms logically dependent hop chains in which later hops depend on instances, sets, or conditions established by earlier hops, while nearly every hop requires fresh visual re-grounding. This training signal encourages continual visual evidence seeking during long-CoT reasoning, improving robustness and reducing compounding errors.

ing, existing vision-language RLVR training data is insufficient; what is needed is training data that structurally forces the model to seek visual evidence at each step of long-CoT reasoning, strengthening step-by-step vision-language reasoning and improving generalization across diverse scenarios.

Motivated by this insight, we propose HopChain1, a scalable framework for synthesizing multi-hop vision-language reasoning data specifically for RLVR training of VLMs. Each synthesized multi-hop query forms a logically dependent chain of instance-grounded hops, where earlier hops establish the instances, sets, or conditions needed for later hops, forcing repeated visual re-grounding throughout training rather than language-only shortcuts. At the same time, each query terminates in a specific, unambiguous numerical answer that is easy to verify for RLVR; moreover, because the hops are logically dependent, obtaining the correct final number usually also requires the intermediate reasoning steps to be correct. Rather than targeting a narrow synthetic task, we use the multi-hop data synthesized by HopChain as a complementary source of RLVR training data to strengthen fundamental vision-language reasoning under long CoT and support broad, generalizable gains across diverse domains. Importantly, this synthesized data is not tailored to any particular downstream benchmark; instead, it is constructed to strengthen general-purpose vision-language reasoning, which we later show transfers broadly across benchmark families. We define this multi-hop structure through two hop types: perception-level hops and instance-chain hops. A perception-level hop switches between single-object perception (e.g., read text, identify color, determine position) and multi-object relationship reasoning (e.g., compare sizes, count objects satisfying a condition, determine spatial arrangement), while remaining grounded in the instances, sets, or conditions established by earlier hops. An instance-chain hop follows an explicit dependency chain (e.g., instance A → B → C), where the next instance can be identified only from the instances, sets, or conditions established by earlier hops. We require each query to combine both hop types in a logically dependent chain, where earlier hops establish the instances, sets, or conditions needed for later hops. This design forces fresh visual grounding at every step, blocks language-only shortcuts, and exposes diverse long-CoT failure modes during training, as shown in Figure 1(c).

To construct such multi-hop data at scale, HopChain adopts a scalable four-stage data synthesis pipeline: (1) category identification via Qwen3-VL-235B-A22B-Thinking (Bai et al., 2025a) to enumerate semantic categories in each image; (2) instance segmentation via SAM3 (Carion et al., 2025) to localize individual

1Throughout this paper, HopChain refers to our synthesis framework, while multi-hop vision-language reasoning data, or simply multi-hop data, refers to the training data synthesized by HopChain.

instances for the identified semantic categories; (3) multi-hop query generation via Qwen3-VL-235B-A22BThinking that constructs logically chained questions over combinations of instances; and (4) human-in-theloop verification where multiple annotators independently solve each query, and only queries with the same final numerical answer are retained as valid training examples. The pipeline provides a scalable data synthesis workflow, scaling to broad image collections with sufficient detectable instances while maintaining strict quality control, as summarized in Figure 1(a).

We apply RLVR with Soft Adaptive Policy Optimization (SAPO) (Gao et al., 2025) on the multi-hop data synthesized by HopChain to train VLMs. We validate the effectiveness of the multi-hop data synthesized by HopChain on Qwen3.5-35B-A3B and Qwen3.5-397B-A17B (Qwen Team, 2026) across 24 benchmarks spanning STEM and Puzzle, General VQA, Text Recognition and Document Understanding, and Video Understanding. Compared with RLVR on the original RLVR data alone, adding this multi-hop data improves 20 out of 24 benchmarks on both Qwen3.5-35B-A3B and Qwen3.5-397B-A17B, despite the fact that the synthesized data is not designed for any specific benchmark, indicating broad and generalizable performance gains across diverse scenarios. To demonstrate that full chained queries are important, we replace them with half-multi-hop or single-hop variants on Qwen3.5-35B-A3B, reducing the average score across five representative benchmarks from 70.4 to 66.7 and 64.3, respectively. Multi-hop training substantially strengthens long-CoT vision-language reasoning, with gains peaking at more than 50 accuracy points in the ultra-long-CoT regime on Qwen3.5-397B-A17B. When we independently sample each synthesized query multiple times on both Qwen3.5-35B-A3B and Qwen3.5-397B-A17B, more than half of the queries are partially solved and the outcomes are distributed relatively evenly from fully incorrect to fully correct, indicating a broad difficulty range suitable for both smaller and larger models. The corrected error types also closely follow the original error distribution before adding this multi-hop data, indicating gains that are broad rather than confined to a single narrow failure mode. These extensive experiments establish HopChain as an effective framework for synthesizing multi-hop data that improves generalizable vision-language reasoning capabilities beyond the synthesized training distribution.

Our main contributions are as follows:

- • We identify diverse and compounding failure modes during long CoT reasoning as a key barrier to vision-language generalization, and show why relying on or simply expanding existing visionlanguage RLVR training data is insufficient.
- • We formalize multi-hop vision-language reasoning with perception-level and instance-chain hops, and build HopChain, a scalable synthesis pipeline whose queries form logically dependent chains in which earlier hops establish the instances, sets, or conditions needed for later hops, forcing repeated visual re-grounding throughout training while keeping the final answer directly verifiable for RLVR.
- • Through extensive experiments, we verify that RLVR on the multi-hop data synthesized by HopChain yields broad, generalizable gains: 20 out of 24 benchmarks improve on both Qwen3.5-35B-A3B and Qwen3.5-397B-A17B, while additional analyses show that preserving full chained queries is important, multi-hop data strengthens long-CoT vision-language reasoning, and the synthesized data spans a broad difficulty range while enabling trained models to correct a broad range of errors.

#### 2 Preliminaries

##### 2.1 Reinforcement Learning with Verifiable Rewards for Vision-Language Models

The reinforcement learning with verifiable rewards (RLVR) framework for vision-language models (VLMs) closely parallels that for large language models (LLMs), with the primary distinction being that RLVR for VLMs processes both an image and a text query as input to generate a textual chain-of-thought culminating in a verifiable answer prediction, whereas RLVR for LLMs operates solely on text queries. Specifically, RLVR for VLMs aims to maximize the following objective:

J(π) = E(I,q,a)∼D,o∼π(·|I,q)[R(o, a)], (1) where R(o, a) =

1.0 if is_equivalent(o, a), 0.0 otherwise.

(2)

Here, I, q, and a denote the image, text query, and ground-truth answer, respectively, sampled from dataset D, and o represents the response generated by policy π conditioned on I and q.

##### 2.2 Soft Adaptive Policy Optimization

Soft Adaptive Policy Optimization (SAPO, (Gao et al., 2025)) is introduced to mitigate the potential instability and inefficiency caused by hard clipping in prior RLVR algorithms such as GSPO (Zheng et al.,

(a) Baseline Error Types (b) Error Types Mitigated by Multi-Hop

[Figure 23]

[Figure 24]

Perception Error

Reasoning Error

Knowledge Error

Hallucination Error

Other

- Figure 2: Error-type distributions before and after adding multi-hop data in RLVR. Subfigure (a) shows the error-type distribution of RLVR w/o Multi-Hop on the benchmarks in Tables 1 and 2. Subfigure (b) shows the distribution of baseline error types mitigated after adding multi-hop data (i.e., cases corrected by RLVR w/ Multi-Hop). In (a), errors are diverse, with perception and reasoning errors as the dominant categories. In (b), the distribution remains similar to (a), indicating that multi-hop data mitigates failures in a generalizable way rather than improving only a narrow error type. Figure 3 provides representative perception reasoning failures, together with responses corrected after adding multi-hop data.

2025) and GRPO (Shao et al., 2024). Concretely, SAPO substitutes hard clipping with a temperaturecontrolled soft gate and optimizes the following objective for VLMs:

J (θ) = E(I,q,a)∼D,{o

i}iG=1∼πold(·|I,q)

1 G

G

∑

i=1

1 |oi|

|oi|

∑

t=1

fi,t(ri,t(θ)) Aˆi,t , (3)

where ri,t(θ) =

πθ(oi,t | I, q, oi,<t) πθold(oi,t | I, q, oi,<t)

, Aˆi,t = Aˆi =

Ri − mean {Rj}Gj=1 std {Rj}Gj=1

, (4)

fi,t(x) = σ(τi,t(x − 1)) ·

4 τi,t

, τi,t =

τpos, if Aˆi,t > 0, τneg, otherwise.

(5)

Here, i, j denote the sample indices in D, and t is the token index within a sequence. The parameters of the currently trained policy and the old rollout policy are denoted by θ and θold, respectively. The temperatures for positive and negative tokens are τpos and τneg, respectively. Moreover, Ri is computed according to Equation (2) for the i-th sample, and σ(x) = 1+1e−x denotes the sigmoid function.

- 3 Diverse Failure Modes in Long Chain-of-Thought Reasoning

Long CoT reasoning places a much higher demand on visual grounding than short, single-step QA. To answer correctly, a VLM must repeatedly return to the image, recover the right object, attribute, relation, or numeric evidence at each step, and use that evidence to determine the next step of reasoning. This makes long CoT reasoning inherently fragile: multiple error types can emerge along the chain, including perception, reasoning, knowledge, and hallucination errors, among others. Once any such error appears at an intermediate step, the remaining reasoning can still look coherent while operating on flawed intermediate evidence, eventually producing an incorrect final answer. Recent analyses in multimodal reasoning report similar patterns: longer reasoning traces can drift away from image-grounded evidence, reduce attention to visual inputs, and amplify hallucinated intermediate content (Liu et al., 2025; Luo et al., 2025). More generally, diagnostic studies of VLMs have also documented persistent object hallucination and visual illusion under image-context reasoning (Rohrbach et al., 2018; Guan et al., 2024).

To obtain the error breakdown in Figure 2, we analyze responses produced on the benchmarks in Table 2 by Qwen3.5-397B-A17B under the RLVR w/o Multi-Hop setting. For each benchmark, we randomly sample 20 incorrect responses and ask annotators to identify the primary failure type by comparing the model output against the ground-truth answer. The resulting breakdown shows that long-CoT failures are diverse rather than concentrated in a single category: perception errors are the largest group, while reasoning, knowledge, and hallucination errors are also present in the breakdown. Qualitative examples in Figure 3 further show that these failures are not limited to a narrow benchmark type. The baseline model (RLVR w/o Multi-Hop) miscounts small local details in the ladybird example, misjudges the contact relation between the gripper and the dress strap, misreads the sign shape in the driving scene, reads

[Figure 25]

[Figure 26]

Which season does C represent? Baseline: Highest arc = summer [...] Lowest arc = winter [...] Middle arc = equinox/spring [...] Label C points to middle arc [...] → Answer: spring.

How many dots do all ladybirds have together? Baseline: [...] Top-middle: 4 dots [...] Bottom-left: 6 dots [...] Bottom-right: 7 dots [...] Total: 2 + 4 + 3 + 6 + 7 = 22. Ours: [...] Top-middle: 3 dots [...] Bottom-left: 5 dots [...] Bottom-right: 6 dots [...] Total: 2 + 3 + 3 + 5 + 6 = 19. Error analysis: Perception Error (Miscounted ladybird dots: top-middle 3 to 4, bottom-left 5 to 6, bottom-right 6 to 7).

[Figure 27]

[Figure 28]

Ours: Highest arc (labeled C) = summer solstice [...] Lowest arc = winter [...] Middle arc = spring/fall [...] Label C points to highest arc [...] → Answer: summer.

[Figure 29]

[Figure 30]

Error analysis: Perception Error (Misjudged the arrow pointing to the highest arc as pointing to the middle arc).

[Figure 31]

[Figure 32]

According to the positive sequence of the images, does the blue car attempt to leave the parking space? Baseline: Top: car parked parallel [...] Middle: wheels turned angling outward [...] Bottom: straightened in lane [...] interprets sequence as leaving → Answer: Yes. Ours: Top: car positioning alongside [...] Middle: reversing at angle to back in [...] Bottom: straightened and stopped parked [...] interprets sequence as entering → Answer: No. Error analysis: Reasoning Error (Correctly perceived positions but logically misinterpreted entry as exit).

What percentage of the zebra's stripes are horizontal? Baseline: Neck/body stripes vertical [...] legs/rump stripes horizontal [...] vertical is dominant pattern [...] percentage of horizontal = 0 → Answer: 0.

[Figure 33]

[Figure 34]

Ours: Front body stripes vertical [...] hindquarters/legs stripes horizontal [...] roughly half and half distribution [...] percentage of horizontal = 50 → Answer: 50.

[Figure 35]

[Figure 36]

Error analysis: Reasoning Error (Misidentification of dominant as 100%, ignoring horizontal stripes).

- Figure 3: Qualitative examples of unreliable visual perception in long CoT reasoning. We show failure cases from the benchmarks in Tables 1 and 2 using “RVLR w/o Multi-Hop” (baseline), alongside correct answers from “RVLR w/ Multi-Hop” (ours). For brevity, unimportant parts of the long CoT are omitted with [. . . ]. The baseline error is highlighted in red, and the failure reason is given in the error analysis.

chart values incorrectly, follows the wrong arc in the astronomy diagram, and selects the wrong body part in the fish illustration. These examples cover natural images, charts, and scientific diagrams, but they share the same structure: one faulty intermediate step appears in the middle of a long reasoning chain, and the later reasoning steps inherit that mistake. In this sense, long-CoT errors are often coupled rather than isolated: a mistaken visual judgment can trigger faulty reasoning, unsupported inference, or other downstream failures. By contrast, RLVR w/ Multi-Hop is more likely to recover the correct visual evidence at each hop and therefore reaches the correct final prediction.

This analysis suggests that the central challenge is not merely producing a longer textual CoT, but maintaining reliable, step-wise reasoning over visual evidence throughout the chain across diverse visual scenarios. This interpretation is also aligned with recent work arguing that multimodal reasoning is often bottlenecked by perception quality and can benefit from stronger intermediate perception, repeated image-grounded observation, or iterative revisiting of visual regions (Bigverdi et al., 2025; Ye et al., 2025; Jiang et al., 2025). Consequently, improving long CoT reasoning requires an RLVR data construction method that is applicable to diverse images, forces repeated grounding during reasoning, and trains the model to use the result of one step to locate, verify, or constrain the next. This is exactly the goal of the HopChain framework introduced in the next section.

#### 4 Boosting Vision-Language Generalization by Synthesizing Multi-Hop Data

The analysis in Section 3 shows that long-CoT reasoning does not fail for a single reason: models may make perception, reasoning, knowledge, hallucination, and other errors, and these errors often propagate once an incorrect intermediate step is carried forward through the rest of the chain. This motivates training data beyond simply relying on, or expanding, existing vision-language RLVR training data. Specifically, the desired training data should structurally force the model to seek visual evidence at each step of long-CoT reasoning, thereby strengthening step-by-step vision-language reasoning and improving generalization across diverse scenarios. Accordingly, we synthesize multi-hop vision-language reasoning data whose hops are designed to instantiate exactly this requirement, while ensuring that each query terminates in a specific, unambiguous numerical answer compatible with RLVR, as illustrated in Figure 4.

##### 4.1 Multi-Hop Vision-Language Reasoning Definition

We first define the structure of the target multi-hop queries. To do so, we use three reasoning levels: Levels 1 and 2 describe what an individual reasoning step asks the model to do, while Level 3 describes a query that chains multiple Level 1 and Level 2 steps together. Level 1 is single-object perception, such as reading text or identifying object attributes, including color, shape, size, position, and category. Level 2 is multi-object perception, such as spatial, comparative, or counting relations across objects. Level 3 is multi-hop reasoning, where multiple Level 1 and Level 2 steps are chained into one query.

Within such a Level 3 query, consecutive hops can be linked in two complementary dimensions. Perception-level hop: the next step changes the kind of perception being performed, for example from a Level 1 single-object judgment to a Level 2 relational judgment, or vice versa, while remaining grounded in the instances, sets, or conditions established by earlier hops. Instance-chain hop: the next step moves to a new instance along an explicit dependency chain (e.g., instance A → B → C), where the next instance can only be identified from the instances, sets, or conditions established by earlier hops.

Each query must satisfy three structural conditions: (i) it must be Level 3, (ii) it must combine both hop types, and (iii) its hops must form a logically dependent chain in which earlier hops establish the instances, sets, or conditions needed for later hops. We further prefer instance dependency chains and perception-level transitions to be intertwined as tightly as possible. In addition, because the synthesized data is intended for RLVR, each query must terminate in a specific, unambiguous numerical answer. This makes the final answer easy to verify for RLVR, while the logical dependence among hops means that obtaining the correct final number usually also requires the intermediate reasoning chain to be correct. This definition is intended to rule out pseudo-multi-hop queries in which substeps are loosely connected or can be bypassed with shallow shortcuts. Intuitively, it targets exactly the kind of failures shown in Figure 3: the model must repeatedly identify the correct visual evidence, then use that evidence to determine the next object, relation, or computation.

##### 4.2 Data Synthesis Pipeline

Given the query definition in Section 4.1, we next describe how candidate multi-hop queries are synthesized. We build the dataset through a three-stage synthesis pipeline that operationalizes this design goal into trainable RLVR data (see Figure 1(a) for the full four-stage workflow, where this subsection covers Stages 1–3 and Section 4.3 covers Stage 4). The pipeline is designed so that the final queries are not only complex, but also structurally aligned with the long-CoT setting we care about: multiple intermediate hops, each grounded in visual evidence and required for later hops.

- Stage 1: Category Identification. We first identify the candidate visual entities that later hops can operate over. Given an input image, we use a VLM (Qwen3-VL-235B-A22B-Thinking) to identify semantic categories present in the image, yielding a list of semantic categories (e.g., “car,” “person,” “sign”) without localization.
- Stage 2: Instance Segmentation. We then resolve these categories into concrete instances, because hop-based reasoning must be anchored to particular objects rather than generic labels. For each identified semantic category, we use SAM3 to generate segmentation masks and bounding boxes for candidate instances of that category where possible, producing a set of individual instances with spatial localization.
- Stage 3: Multi-Hop Query Generation. This stage is where we convert detected instances into training questions that enforce chained visual reasoning rather than isolated recognition. We form combinations of 3–6 instances and use Qwen3-VL-235B-A22B-Thinking to generate multi-hop queries for each combination. The model receives the original image plus cropped patches of each instance, where the patches are used only at design time to help identify the appearance and location of each instance. The full prompt used for multi-hop query synthesis is provided in Appendix A. Crucially, to address the

- Multi-Hop Query #1:

- H1: Start by examining the introductory text paragraph located on the right side of the image, directly below the main 'CENSORSHIP' title banner. Identify the URL mentioned at the very end of this paragraph and count the number of letters in the portion of the URL strictly after the forward slash.
- H2: Use this letter count to find the category in the 'WHO INITIATES CHALLENGES TO MATERIALS?' donut chart that has a percentage value exactly equal to this number. Identify the specific color used for this category's slice in the chart.
- H3: Next, look at the large illustration in the top-left corner featuring a book and a megaphone. Determine if the primary color of the book's cover matches the color identified from the chart slice in the previous step.
- H4: If they match, locate the facility type in the 'WHERE DO CHALLENGES TAKE PLACE?' section (on the right) whose corresponding building icon is drawn in this same color. Note its percentage. If they don’t match, the final number is 0.
- H5: Then, find the 'BEYOND BOOKS' section in the bottom-left and identify the percentage of challenges that target 'Databases'. Sum this 'Databases' percentage with the percentage from the facility type found earlier.
- H6: Finally, return to the 'WHO INITIATES' donut chart, identify the category whose percentage is numerically closest to this calculated sum, and subtract the initial letter count (from the URL step) from this category's percentage. What is the final number?

[Figure 37]

- H1: The number of letters strictly after the forward slash is 6. Answer: 6.

Ground-Truth Answer #2:

Multi-Hop Query #2:

[Figure 38]

[Figure 39]

- H2: The donut chart's 6% segment is red. Answer: red.

| |
|---|

[Figure 40]

- H3: The book cover's primary red color matches the chart slice. Answer: yes.

| |
|---|

[Figure 41]

- H4: Since H3 answered yes, the red building icon's percentage is 16. Answer: 16.

| |
|---|

[Figure 42]

- H5: Databases account for 18% of challenges. Answer: H4 + 18 = 34.

| |
|---|

[Figure 43]

- H6: The donut chart percentage closest to H5 is 32. Answer: 32 – H1 = 26.

- H1: Start by locating the black sheep toy on the left side of the image. Count the number of white eyes visible on this sheep.
- H2: Next, inspect the large area of crumpled white paper that forms the background strictly behind and above the sheep; determine if there is any readable text on this specific background paper. If text is present, set a reference value to 5; if no text is visible, set the reference value to 10.
- H3: Then, move right to identify the first doll. Count the number of eyes visible on this doll's face.
- H4: Continue to the right to find a second doll. Locate the crumpled paper positioned directly in front of this second doll, read the bold title word starting with 'T', and count its letters.
- H5: Now, perform the calculation: add the sheep's eye count to the first doll's eye count, multiply this sum by the letter count of the word, and then add the reference value from the background paper check.
- H6: Finally, multiply this total by the number of distinct toy figures present in the image. What is the final number?

###### Ground-Truth Answer #1:

[Figure 44]

- H1: The black sheep has 2 visible eyes. Answer: 2.

[Figure 45]

- H2: No readable text. Answer: 10.

[Figure 46]

- H3: The doll to the sheep’s right has no visible eyes. Answer: 0.

[Figure 47]

- H4: The bold title word starting with ‘T’ has 7 letters. Answer: 7.
- H5: (H1 + H3) x H4 + H2 = 24. Answer: 24.

- H6: The number of distinct toy figures is 3. Answer: H5 x 3 = 72.

[Figure 48]

[Figure 49]

- Figure 4: Examples of synthesized multi-hop data. In each query, purple-highlighted text denotes the instance chain, meaning that the instance required at the current hop can only be identified from instances established in earlier hops. In the corresponding image, we mark the instance regions involved in each hop with colored rectangles, and the rectangle colors are aligned with the text colors of the corresponding hop-wise ground-truth answers.

failure modes identified in Section 3, we impose several constraints on each synthesized query: it must (i) involve as many instances as possible from the selected combination while remaining answerable from the original image alone; (ii) describe objects only by spatial, contextual, or visual attributes; (iii) terminate in a specific, unambiguous numerical answer; and (iv) avoid any reference to segmentation masks, bounding boxes, or patch images. These constraints prevent the synthesis process from leaking shortcuts that would let the model solve the query from auxiliary cues unavailable at training or test time. In addition, reasoning hops must form a logically dependent chain: each hop must either change the perception level or move along an instance dependency chain, while earlier hops establish the instances, sets, or conditions needed for later hops. As a result, the synthesized queries better mirror the structure of long-CoT reasoning: the model must recover and retain intermediate visual evidence and use it to determine the next step, rather than rely on an early guess or a language-only heuristic.

##### 4.3 Ground-Truth Annotation and Difficulty Calibration

After the synthesis pipeline in Section 4.2 produces candidate queries, we introduce a separate Stage 4 with two complementary components: ground-truth annotation and difficulty calibration. In this stage, ground-truth annotation filters out low-quality queries and provides reliable numerical ground-truth answers for the retained ones, while difficulty calibration uses these verified answers to remove queries that are too easy.

- Stage 4: Ground-Truth Annotation and Difficulty Calibration. For each generated query, four annotators from our data annotation team independently solve it. During ground-truth annotation, we discard queries where annotators report ambiguous references or other quality issues, and for queries that pass, we take the shared final numerical answer (all four must agree) as the ground truth. For difficulty calibration, we evaluate the retained queries on a weaker model with eight sampled responses per query, using the verified ground-truth answers as reference; queries on which the weaker model achieves 100% accuracy are removed as too easy, and the remaining queries form our final dataset with verified difficulty and reliable ground-truth annotations.

Together, the pipeline functions as a scalable data synthesis workflow: from raw images to verified multi-hop queries, it can be applied to broad image collections with sufficient detectable instances, while maintaining quality control through human and model-based verification. In this way, the synthesized data is not merely harder data. It is constructed to expose the model to the same kind of chained evidence use that long-CoT reasoning requires in practice, thereby providing a more suitable training signal for improving robustness and generalization across diverse visual scenarios. A direct comparison between the typical vision-language reasoning question in Figure 3 and the synthesized multi-hop data in Figure 4 makes this difference concrete: simply scaling existing RLVR-style data often still leaves shallow or loosely coupled solution paths, whereas our synthesized data enforces hop-wise dependency and repeated visual re-grounding throughout the chain. This structural gap is precisely why relying on, or merely expanding, existing vision-language RLVR training data is insufficient.

- 5 Experiments

We evaluate HopChain from three complementary angles. First, we test whether adding multi-hop data to RLVR training based on the original RLVR data improves general vision-language reasoning across model scales and benchmark families (Section 5.2). Second, we examine whether the full multi-hop formulation is necessary, rather than weaker single-hop or truncated-hop variants (Section 5.3). Third, we analyze whether the observed gains align with our central claims about long-CoT robustness, data difficulty coverage, and broad correction of diverse failure modes (Section 5.4).

##### 5.1 Experimental Setup

We evaluate two models, Qwen3.5-35B-A3B and Qwen3.5-397B-A17B (Qwen Team, 2026), under three settings: Before RLVR, RLVR w/o Multi-Hop, and RLVR w/ Multi-Hop. Here, Before RLVR denotes the model after SFT and before RLVR, which serves as the shared starting point for both RLVR settings; RLVR w/o Multi-Hop denotes RLVR training on the original RLVR data only, while RLVR w/ Multi-Hop denotes RLVR training on a mixture of the original RLVR data and our synthesized multi-hop data. Unless otherwise stated, the SFT data and the original RLVR data used in these experiments are pre-final internal versions, rather than the final official Qwen3.5 data recipe. For Qwen3.5-35B-A3B, each rollout samples 16 responses for each of 256 queries, uses a mini-batch size of 64 queries, and runs for 1000 gradient steps. Qwen3.5-397B-A17B follows the same rollout configuration in general, except that the mini-batch size is increased to 128 queries and the total number of gradient steps is 800. For both models, RLVR training uses the SAPO algorithm with a learning rate of 2.0 × 10−6, while other hyperparameters follow the

- Table 1: Results of Qwen3.5-35B-A3B. Before RLVR denotes the model after SFT but before RLVR. RLVR w/o Multi-Hop denotes RLVR training on the original RLVR data only. RLVR w/ Multi-Hop denotes RLVR training on the original RLVR data plus the multi-hop data synthesized by HopChain. Benchmark Before RLVR RLVR w/o Multi-Hop RLVR w/ Multi-Hop

STEM and Puzzle MathVision 61.97 73.71 76.05 MMMU Pro 66.07 69.25 70.64 MMMU 75.89 78.89 78.33 Mathvista(mini) 82.80 85.50 85.00 BabyVision 14.95 21.91 22.68 ZEROBench 1 1 3 EMMA(mini) 41.88 53.00 58.00 LogicVista 63.85 74.66 75.56

##### General VQA

MMBenchCN-DEV-V1.1 87.46 90.17 90.48 MMBenchEN-DEV-V1.1 88.47 90.63 91.49 RealWorldQA 75.42 78.17 79.35 MMStar 75.20 78.53 78.60 HallusionBench 66.49 66.64 66.50 AI2D_TEST 88.44 90.87 91.29 ERQA 44.50 48.25 51.38

##### Text Recognition and Document Understanding

CharXiv 61.30 69.00 73.10 DocVQA_VAL 95.00 95.13 95.55 InfoVQA_VAL 86.81 87.44 90.17

##### Video Understanding

VideoMMEw/o sub. 73.41 74.63 75.00 VideoMMMU 70.67 73.33 74.78 MMVUCOT 63.70 65.80 68.90 MVBench 69.18 69.95 70.73 LVBench 51.13 54.49 53.20 MLVU (M-Avg) 77.92 77.69 79.53

original SAPO paper (Gao et al., 2025).

Image Filtering. Before multi-hop query synthesis, we filter images to retain cases that are genuinely useful for long vision-language reasoning. The selection prompt in Appendix B is designed to favor images that are perceptually challenging for standard vision models, such as those involving occlusion, dense but still analyzable objects, unusual poses, complex interactions, fine-grained distinctions, or difficult lighting, while filtering out images that are too low-quality or too annotation-impractical to support verifiable reasoning. To balance quality and throughput, we use a two-stage pipeline. We first apply Qwen3-VL-235B-A22B-Thinking to a small subset of images with the image-selection prompt and keep the accepted images as the initial set of selected images. We then use this initial set of selected images together with the large model’s outputs to perform supervised fine-tuning (SFT) on a smaller model, Qwen3-VL-30B-A3B-Thinking, specialized for image filtering. Next, we run this smaller model over all remaining images to obtain a coarse screening result. Finally, we send the images retained by the coarse screening back to Qwen3-VL-235B-A22B-Thinking for a second, finer filtering pass, yielding a second set of selected images. The union of the initial set of selected images and the second set of selected images forms the final pool of images used for multi-hop data construction. Based on the filtered image pool above, we construct multi-hop queries with HopChain and then filter out overly easy samples using a weaker model and the pre-RLVR model, yielding about 6k–8k multi-hop RLVR samples for each model. In addition, we use a similar amount of math RLVR data as the original RLVR data.

Benchmarks. We evaluate 24 benchmarks across 4 categories. For STEM and puzzle reasoning, we use MathVision (Wang et al., 2024a), MMMU-Pro (Yue et al., 2025), MMMU (Yue et al., 2024), MathVista (Lu et al., 2024), BabyVision (Chen et al., 2026), ZeroBench (Roberts et al., 2025), EMMA (Hao et al., 2025), and LogicVista (Xiao et al., 2024). For general VQA, we use MMBench (Liu et al., 2023d), RealWorldQA (xAI,

- Table 2: Results of Qwen3.5-397B-A17B, the larger model counterpart of Table 1. Before RLVR denotes the model after SFT but before RLVR. RLVR w/o Multi-Hop denotes RLVR training on the original RLVR data only. RLVR w/ Multi-Hop denotes RLVR training on the original RLVR data plus the multi-hop data synthesized by HopChain. Benchmark Before RLVR RLVR w/o Multi-Hop RLVR w/ Multi-Hop

STEM and Puzzle MathVision 77.38 81.68 83.71 MMMU Pro 73.03 75.06 76.47 MMMU 79.78 81.67 82.89 Mathvista(mini) 87.50 88.30 89.00 BabyVision 17.01 28.61 32.22 ZEROBench 3 4 8 EMMA(mini) 58.13 66.25 69.00 LogicVista 75.62 80.69 81.59

##### General VQA

MMBenchCN-DEV-V1.1 89.47 91.41 91.72 MMBenchEN-DEV-V1.1 90.71 92.49 91.56 RealWorldQA 79.87 79.87 81.70 MMStar 80.00 81.73 80.67 HallusionBench 65.76 67.48 67.86 AI2D_TEST 91.45 92.81 92.97 ERQA 53.75 60.50 60.00

##### Text Recognition and Document Understanding

CharXiv 70.00 74.60 77.20 DocVQA_VAL 95.93 95.98 96.03 InfoVQA_VAL 90.42 90.83 92.20

##### Video Understanding

VideoMMEw/o sub. 76.56 78.30 80.41 VideoMMMU 76.67 78.89 80.00 MMVUCOT 70.20 72.30 72.50 MVBench 69.63 73.03 73.31 LVBench 58.30 59.13 59.07 MLVU (M-Avg) 81.46 82.43 82.52

2024), MMStar (Chen et al., 2024a), HallusionBench (Guan et al., 2024), AI2D (Kembhavi et al., 2016), and ERQA (Gemini Robotics Team et al., 2025). For text recognition and document understanding, we use CharXiv (Wang et al., 2024c), DocVQA (Mathew et al., 2021), and InfographicVQA (reported as InfoVQA_VAL) (Mathew et al., 2022). For video understanding, we use Video-MME (Fu et al., 2025), Video-MMMU (Hu et al., 2025), MVBench (Li et al., 2024), LVBench (Wang et al., 2024b), MLVU (Zhou et al., 2024), and MMVU (reported as MMVUCOT) (Zhao et al., 2025).

##### 5.2 Main Benchmark Results

Tables 1 and 2 report the main benchmark results on Qwen3.5-35B-A3B and Qwen3.5-397B-A17B, respectively. RLVR on the original RLVR data (i.e., RLVR w/o Multi-Hop) already improves the Before RLVR models on many benchmarks, but augmenting the original RLVR data with the multi-hop data synthesized by HopChain (i.e., RLVR w/ Multi-Hop) yields further gains on the majority of benchmarks for both model scales. These gains are broad rather than isolated. Importantly, our multi-hop data is not synthesized to target any specific benchmark. Even so, it delivers broad gains on the majority of benchmarks, suggesting that HopChain improves generalizable vision-language reasoning rather than overfitting to benchmark-specific patterns. Quantitatively, RLVR w/ Multi-Hop improves 20 out of 24 benchmarks for both Qwen3.5-35B-A3B and Qwen3.5-397B-A17B. For Qwen3.5-35B-A3B, the gains cover 6/8 STEM-and-puzzle benchmarks (MathVision, MMMU-Pro, BabyVision, ZeroBench, EMMA, and LogicVista), 6/7 general-VQA benchmarks (MMBench-CN, MMBench-EN, RealWorldQA, MMStar, AI2D, and ERQA), all 3 text-and-document benchmarks (CharXiv, DocVQA, and InfoVQA), and 5/6 video benchmarks (Video-MME, VideoMMMU, MMVUCOT, MVBench, and MLVU). For Qwen3.5-397B-A17B, the gains are similarly broad: all 8 STEM-and-puzzle benchmarks improve (MathVision, MMMU-Pro,

RLVR w/ Single Hop

RLVR w/ Half-Multi-Hop

RLVR w/ Multi-Hop

| |
|---|

| |
|---|

| |
|---|

79.35

80

78.30

76.05

74.90

74.78

73.03

72.76

70.78

70.64

BenchmarkScore

70

65.78

62.72

60.87

60

51.38

50

48.50

47.25

40

MathVision MMMU Pro RealWorldQA ERQA VideoMMMU

- Figure 5: Benchmark-level comparison on Qwen3.5-35B-A3B under three training-query settings. RLVR w/ Single Hop simplifies each multi-hop training query to only its final hop, RLVR w/ Half-Multi-Hop removes the first half of the hops and keeps only the latter half, and RLVR w/ Multi-Hop uses the full multi-hop training queries. We evaluate the resulting models on five representative benchmarks, namely MathVision, MMMU Pro, RealWorldQA, ERQA, and VideoMMMU, and plot the benchmark score for each setting. Across all five benchmarks, the full multi-hop setting performs best, showing that preserving the complete multi-hop structure during training is more effective than shortening the query chain.

MMMU, MathVista, BabyVision, ZeroBench, EMMA, and LogicVista), along with 4/7 general-VQA benchmarks (MMBench-CN, RealWorldQA, HallusionBench, and AI2D), all 3 text-and-document benchmarks (CharXiv, DocVQA, and InfoVQA), and 5/6 video benchmarks (Video-MME, VideoMMMU, MMVUCOT, MVBench, and MLVU). These broad quantitative gains are also reflected qualitatively: Figure 3 shows representative cases in which adding multi-hop data corrects failures made by RLVR w/o Multi-Hop. Notably, although our multi-hop data is synthesized from images, the improvement transfers strongly to video understanding: video benchmarks improve on 5 out of 6 tasks for both model scales, indicating substantial cross-domain generalization rather than narrow overfitting to image-only training signals. Across the two tables, the improvements cover all four benchmark families, with only a small number of regressions, indicating that multi-hop data acts as a composable RLVR training signal rather than a narrow task-specific trick.

##### 5.3 Ablation on Hop Structure: Single-Hop vs. Half-Multi-Hop vs. Multi-Hop

To test whether preserving complex, long hop chains during training is necessary, we compare three training-query settings on the subset of benchmarks shared by all three runs. In RLVR w/ Single Hop, each multi-hop training query is reduced to only its final hop; in RLVR w/ Half-Multi-Hop, the first half of the chain is removed and only the latter half is kept; and in RLVR w/ Multi-Hop, the full query is kept.

A benchmark-level comparison on five representative tasks is shown in Figure 5. On MathVision, MMMU Pro, RealWorldQA, ERQA, and VideoMMMU, the ordering is consistent: RLVR w/ Multi-Hop performs best, followed by RLVR w/ Half-Multi-Hop, and then RLVR w/ Single Hop. This per-benchmark ordering is also reflected in the aggregate: averaged across these five representative benchmarks, the score drops from 70.4 for RLVR w/ Multi-Hop to 66.7 for RLVR w/ Half-Multi-Hop and 64.3 for RLVR w/ Single Hop. This pattern shows that the benefit does not come from collapsing the training queries to the last hop or to a shortened chain; preserving longer cross-hop dependencies is important.

##### 5.4 Analysis

Analysis by Reasoning Length. If HopChain primarily strengthens long-CoT reasoning, its advantage should persist as response chains become longer. Figure 6 tests exactly this on Qwen3.5-397B-A17B by binning benchmark responses by response token count. The advantage of RLVR w/ Multi-Hop over RLVR w/o Multi-Hop remains visible across the bins and is often larger in the ultra-long-response regime. This supports the interpretation that HopChain improves robust chained reasoning over long outputs, rather than only helping on short or easy responses.

Difficulty Coverage Across Model Scales. A useful multi-hop training set should cover a broad range of difficulties, rather than collapsing to queries that are either trivial or essentially unsolvable. Figure 7

RLVR w/o Multi-Hop

AccuracyImprovement(%)

80

50

RLVR w/ Multi-Hop

70

40

Accuracy(%)

60

30

50

20

40

10

30

0

20

10

10

20

6000 8000 10000 12000 14000 16000 18000 20000

Response Token Count

- Figure 6: Performance as a function of response length, aggregated from the benchmark evaluation results of Qwen3.5-397B-A17B in Table 2. The blue dashed line and the orange solid line show the accuracies of RLVR w/o Multi-Hop and RLVR w/ Multi-Hop, respectively, across bins of increasing response token count, while the bars show the corresponding accuracy improvement of RLVR w/ Multi-Hop over RLVR w/o Multi-Hop on the right axis. The advantage of multi-hop training remains evident on long responses and is especially pronounced in the ultra-long-CoT region.

0/8 1/8 2/8 3/8 4/8 5/8 6/8 7/8 8/8

Success Rate (Correct/Total)

0

10

20

30

40

50

PercentageofQueries(%)

13.1%

10.6% 9.7% 9.2% 9.6% 9.5%

10.9%11.8%

15.6%

All Correct (8/8): 15.57% Partially Correct (1-7/8): 71.34% All Incorrect (0/8): 13.10%

Qwen3.5-35B-A3B

0/8 1/8 2/8 3/8 4/8 5/8 6/8 7/8 8/8

Success Rate (Correct/Total)

0

10

20

30

40

50

PercentageofQueries(%)

8.5%

5.3% 5.5% 5.3% 6.0% 7.2%

8.9%

13.2%

40.0%

All Correct (8/8): 39.99% Partially Correct (1-7/8): 51.49% All Incorrect (0/8): 8.52%

Qwen3.5-397B-A17B

All Incorrect (0/8)

| |
|---|

Partially Correct (1-7/8)

| |
|---|

All Correct (8/8)

- Figure 7: Query-level success-rate distributions for Qwen3.5-35B-A3B and Qwen3.5-397B-A17B. For each multi-hop query, we independently sample eight responses and bucket the query by how many of the eight are correct, from 0/8 to 8/8. For both models, more than half of the queries fall into the Partially Correct regime, and the queries are distributed across multiple success buckets. This suggests that the synthesized multi-hop data spans a broad difficulty range and is suitable for RLVR training across models of different sizes and capability levels.

measures this by independently sampling eight responses for each multi-hop query and grouping the query by how many sampled responses are correct. For both Qwen3.5-35B-A3B and Qwen3.5-397B-A17B, more than half of the queries fall into the Partially Correct regime, and the distribution spans multiple success buckets. This indicates that the synthesized multi-hop data covers a broad difficulty range and can provide a useful RLVR training signal to models of different sizes and capability levels.

Error-Type Analysis. Finally, we ask whether multi-hop augmentation fixes only a narrow corner of the error space or yields broader gains. Figure 2 analyzes the original error types of the cases where RLVR w/ Multi-Hop corrects an error made by RLVR w/o Multi-Hop. Compared with the baseline error profile in Figure 2, the corrected-case distribution is broadly similar: perception errors remain the largest group, reasoning errors remain the second largest, and knowledge, hallucination, and other errors are also represented. The subtype breakdowns in Figure 8 further show that these improvements extend across diverse subcategories, covering chart and text misreads, object misidentification, spatial and counting errors, as well as logic, math, temporal, and causal reasoning errors. Taken together, the close

### Ours相对于Baseline提升的分析

###### Perception Error Types Major Error Types Reasoning Error Types

[Figure 50]

[Figure 51]

[Figure 52]

Spatial Error Chart Misread

Perception Error Reasoning Error

Logic Error Temporal Error

Counting Error

Text Misread

Knowledge Error

Hallucination Error

Math Error Causal Error

Other

Spatial Error

Object Misidentify

Color Error

Temporal Misread

- Figure 8: Error distribution of the original RLVR w/o Multi-Hop errors among the cases where RLVR w/ Multi-Hop improves over RLVR w/o Multi-Hop on the benchmarks in Tables 1 and 2. Center pie chart: the major error categories of the corrected cases, indicating which types of baseline errors are most often fixed after adding multi-hop data. Left pie chart: the subtype breakdown of corrected perception errors. Right pie chart: the subtype breakdown of corrected reasoning errors. With Figure 2, this figure shows that the multi-hop data synthesized by HopChain can improve models in broad, generalizable domains.

[Figure 53]

correspondence to the baseline error distribution in Figure 2 and the broad subtype gains in Figure 8 suggest that HopChain does not only patch a single narrow failure type; instead, it yields broad gains across the diverse failure modes in long-CoT vision-language reasoning.

[Figure 54]

#### 6 Related Work

- 6.1 Vision-Language Models

The integration of visual encoders with large language models has driven rapid progress in multimodal understanding. LLaVA (Liu et al., 2023b) introduced the visual instruction tuning paradigm by projecting visual features into the language model’s embedding space, a design subsequently refined in LLaVA1.5 (Liu et al., 2023a) with stronger projectors and higher-resolution inputs. This architectural template has been adopted and scaled by both open-source and proprietary efforts, including InternVL (Chen et al., 2024c;b), Qwen-VL (Bai et al., 2023; 2025b), GPT-4V (OpenAI, 2023), and Gemini (Gemini Team et al.,

- 2023). Alongside these advances, diagnostic studies show that even strong VLMs remain vulnerable to object hallucination, visual illusion, and language-prior-driven errors (Rohrbach et al., 2018; Guan et al., 2024; Leng et al., 2024). Recent analyses further suggest that these failures can become more pronounced when multimodal reasoning chains grow longer and drift away from image-grounded evidence (Liu et al., 2025). While these models achieve strong aggregate scores on diverse benchmarks, their visual perception can still become unreliable during multi-step reasoning. This is the bottleneck our work targets through synthesized multi-hop data.

6.2 Reinforcement Learning for Language and Vision-Language Models

Reinforcement learning from human feedback (RLHF) (Ouyang et al., 2022) first demonstrated the effectiveness of reinforcement learning (RL)-based alignment for language models, using proximal policy optimization (PPO) (Schulman et al., 2017) as the default optimizer. More recently, reinforcement learning with verifiable rewards (RLVR) (Shao et al., 2024) has emerged as a powerful alternative to RLHF, eliminating the need for a learned reward model by using programmatically verifiable answers. DeepSeek-R1 (DeepSeek-AI, 2025) further showed that pure RL can induce strong chain-of-thought reasoning in LLMs, which has motivated parallel extensions to VLMs such as VLM-R1 (Shen et al., 2025) and TikArt (Ding et al., 2026). In parallel, the optimization toolkit has also evolved: GRPO (Shao et al.,

- 2024) and GSPO (Zheng et al., 2025) use group-based advantage estimation with hard clipping, while SAPO (Gao et al., 2025) replaces hard clipping with a temperature-controlled soft gate for improved stability. Beyond algorithm design, recent mechanistic and robustness analyses suggest that RL’s effects are selective rather than uniform. In LLMs, improvements are concentrated on a minority of high-entropy tokens (Wang et al., 2025); in VLMs, RL appears to primarily refine vision-to-reasoning alignment in midto-late layers rather than uniformly strengthening visual perception (Li et al., 2026), and RL-finetuned

models can still exhibit weak visual grounding and over-reliance on textual cues (Zhao et al., 2026). In related multimodal reasoning settings, recent work has also begun to explicitly reward visually grounded reasoning traces, for example through visually grounded reinforcement finetuning for visual document reasoning (Ni et al., 2025) and evidence-grounded RL for video reasoning (Luo et al., 2025). Most existing RL and RLVR works remain tied to pre-existing task or benchmark data, including math- and scienceoriented reasoning sets as well as task-specific multimodal datasets for documents or videos. In contrast, our pipeline generates a benchmark-agnostic proxy task designed to force repeated visual grounding and yield broad, generalizable improvements.

##### 6.3 Vision-Language Reasoning and Multi-Hop Reasoning

Compositional visual reasoning has been studied through synthetic diagnostic datasets such as CLEVR (John-

son et al., 2017) and real-image benchmarks such as GQA (Hudson and Manning, 2019), with early architectural innovations including neural module networks (Andreas et al., 2016) and visual programming (Gupta and Kembhavi, 2023). In the language domain, multi-hop question answering requires chaining evidence across multiple passages. This setting has been formalized by benchmarks such as HotpotQA (Yang et al., 2018). Chain-of-thought (CoT) prompting (Wei et al., 2022) and its multimodal extension (Zhang et al., 2023) have shown that eliciting step-by-step reasoning significantly improves both LLM and VLM performance. At the same time, several recent works argue that strong multimodal reasoning requires more than language-space CoT: it depends on fine-grained observation, stronger intermediate perception representations, repeated image inspection, or iterative revisiting of visual regions (Bigverdi et al., 2025; Ye et al., 2025; Jiang et al., 2025). Our work differs from prior visual reasoning research in three respects. (1) We formalize multi-hop vision-language reasoning with two complementary hop types: perception-level hops (switching between single-object and multi-object perception) and instance-chain hops (A → B → C dependency chains), instead of treating it as a single-axis chain. (2) We use multi-hop reasoning as a proxy task to improve general VLM capabilities rather than as an end goal. (3) Our data is synthesized on real images at scale, which bridges the gap between synthetic diagnostics and manually curated benchmarks.

##### 6.4 Scalable Data Synthesis for Model Training

Using strong models to generate training data for weaker ones has become a central scaling strategy. Self-Instruct (Wang et al., 2023) and Alpaca (Taori et al., 2023) pioneered LLM-generated instruction data, while ShareGPT4V (Chen et al., 2023) extended this idea to vision-language captions. Representative multimodal data synthesis pipelines additionally integrate vision foundation models. They often combine a VLM for object detection with SAM/SAM2 (Kirillov et al., 2023; Ravi et al., 2024) for instance segmentation. They may also use open-set detection approaches such as Grounding DINO (Liu et al., 2023c) to construct structured multi-hop queries from raw images. Unlike prior data synthesis efforts that aim to approximate the target task distribution, our pipeline generates a benchmark-agnostic proxy task designed to force repeated visual grounding for generalizable improvement.

#### 7 Conclusion

In this paper, we identify diverse and compounding failure modes during long CoT reasoning as a central barrier to robust vision-language reasoning. To address this challenge, we propose HopChain, a scalable framework for synthesizing multi-hop vision-language reasoning data for RLVR training, where earlier hops establish the instances, sets, or conditions needed for later hops, forcing repeated visual re-grounding throughout training while terminating in specific, unambiguous numerical answers suitable for verifiable rewards. By augmenting the original RLVR data with HopChain-synthesized multi-hop data, we obtain broad and generalizable gains on 20 out of 24 benchmarks on both Qwen3.535B-A3B and Qwen3.5-397B-A17B. Importantly, these gains arise even though the synthesized data is benchmark-agnostic rather than designed for any specific benchmark. Additional analyses further show that preserving full chained queries is important, that multi-hop data substantially strengthens long-CoT vision-language reasoning, and that the synthesized data spans a broad difficulty range while enabling trained models to correct a broad range of errors. These results establish HopChain as an effective and scalable framework for improving generalizable vision-language reasoning in VLMs.

Despite these gains, the current pipeline still depends on successful instance segmentation, so images with no detectable objects (and thus no SAM3-segmentable instances) cannot be processed and are excluded from the current synthesis workflow. A natural next step is to reduce this dependency by introducing complementary data-construction routes for images with few or no segmentable objects, while preserving the core design principle of chained visual grounding for long-CoT RLVR training.

#### References

Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. Neural module networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 39–48, 2016.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966, 2023.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

Mahtab Bigverdi, Zelun Luo, Cheng-Yu Hsieh, Ethan Shen, Dongping Chen, Linda G. Shapiro, and Ranjay Krishna. Perception tokens enhance visual reasoning in multimodal language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 3836–3845, 2025.

Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, Shoubhik Debnath, Ronghang Hu, Didac Suris, Chaitanya Ryali, Kalyan Vasudev Alwala, Haitham Khedr, Andrew Huang, Jie Lei, Tengyu Ma, Baishan Guo, Arpit Kalla, Markus Marks, Joseph Greer, Meng Wang, Peize Sun, Roman Rädle, Triantafyllos Afouras, Effrosyni Mavroudi, Katherine Xu, Tsung-Han Wu, Yu Zhou, Liliane Momeni, Rishi Hazra, Shuangrui Ding, Sagar Vaze, François Porcher, Feng Li, Siyuan Li, Aishwarya Kamath, Ho Kei Cheng, Piotr Dollár, Nikhila Ravi, Kate Saenko, Pengchuan Zhang, and Christoph Feichtenhofer. Sam 3: Segment anything with concepts, 2025. URL https://arxiv.org/abs/2511.16719.

Liang Chen, Weichu Xie, Yiyan Liang, Hongfeng He, Hans Zhao, Zhibo Yang, Zhiqi Huang, Haoning Wu, Haoyu Lu, Y. Charles, Yiping Bao, Yuantao Fan, Guopeng Li, Haiyang Shen, Xuanzhong Chen, Wendong Xu, Shuzheng Si, Zefan Cai, Wenhao Chai, Ziqi Huang, Fangfu Liu, Tianyu Liu, Baobao Chang, Xiaobo Hu, Kaiyuan Chen, Yixin Ren, Yang Liu, Yuan Gong, and Kuan Li. Babyvision: Visual reasoning beyond language. arXiv preprint arXiv:2601.06521, 2026.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Conghui He, Jiaqi Wang, Feng Zhao, and Dahua Lin. Sharegpt4v: Improving large multi-modal models with better captions. arXiv preprint arXiv:2311.12793, 2023.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024a.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024b.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks via progressive image-text alignment. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024c.

DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Hao Ding, Zhichuan Yang, Weijie Ge, Ziqin Gao, Chaoyi Lu, and Lei Zhao. Tikart: Stabilizing aperture-guided fine-grained visual reasoning with reinforcement learning. arXiv preprint arXiv:2602.14482, 2026.

Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

Chang Gao, Chujie Zheng, Xiong-Hui Chen, Kai Dang, Shixuan Liu, Bowen Yu, An Yang, Shuai Bai, Jingren Zhou, and Junyang Lin. Soft adaptive policy optimization. arXiv preprint arXiv:2511.20347, 2025.

Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, et al. Gemini robotics: Bringing ai into the physical world. arXiv preprint arXiv:2503.20020, 2025. URL https://arxiv.org/abs/2503.20020.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Sorber, Johan Schalkwyk, Andrew M. Dai, et al. Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14375–14385, 2024.

Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14953–14962, 2023.

Yunzhuo Hao, Jiawei Gu, Huichen Will Wang, Linjie Li, Zhengyuan Yang, Lijuan Wang, and Yu Cheng. Can mllms reason in multimodality? emma: An enhanced multimodal reasoning benchmark. arXiv preprint arXiv:2501.05444, 2025.

Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos. arXiv preprint arXiv:2501.13826, 2025. URL https://arxiv.org/abs/2501.13826.

Drew A. Hudson and Christopher D. Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6700–6709, 2019.

Chaoya Jiang, Yongrui Heng, Wei Ye, Han Yang, Haiyang Xu, Ming Yan, Ji Zhang, Fei Huang, and Shikun Zhang. Vlm-r3: Region recognition, reasoning, and refinement for enhanced multimodal chain-of-thought. arXiv preprint arXiv:2505.16192, 2025.

Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2901–2910, 2017.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision – ECCV 2016, pages 235–251. Springer International Publishing, 2016.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4015–4026, 2023.

Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. Mitigating object hallucinations in large vision-language models through visual contrastive decoding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13872–13882, 2024.

Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, and Yu Qiao. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

Xirui Li, Ming Li, and Tianyi Zhou. What does rl improve for visual reasoning? a frankenstein-style analysis. arXiv preprint arXiv:2602.12395, 2026.

Chengzhi Liu, Zhongxing Xu, Qingyue Wei, Juncheng Wu, James Zou, Xin Eric Wang, Yuyin Zhou, and Sheng Liu. More thinking, less seeing? assessing amplified hallucination in multimodal reasoning models. arXiv preprint arXiv:2505.21523, 2025.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023a.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, 2023b.

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. arXiv preprint arXiv:2303.05499, 2023c.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023d.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR), 2024.

Mi Luo, Zihui Xue, Alex Dimakis, and Kristen Grauman. When thinking drifts: Evidential grounding for robust video reasoning. arXiv preprint arXiv:2510.06077, 2025.

Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 2200–2209, 2021.

Minesh Mathew, Viraj Bagal, Ruben Tito, Dimosthenis Karatzas, Ernest Valveny, and C. V. Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2022.

Minheng Ni, Zhengyuan Yang, Linjie Li, Chung-Ching Lin, Kevin Lin, Wangmeng Zuo, and Lijuan Wang. Point-rft: Improving multimodal reasoning with visually grounded reinforcement finetuning. arXiv preprint arXiv:2505.19702, 2025.

OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini

Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, pages 27730–27744, 2022.

Qwen Team. Qwen3.5: Accelerating productivity with native multimodal agents, February 2026. URL https: //qwen.ai/blog?id=qwen3.5.

Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, Eric Mintun, Junting Pan, Kalyan Vasudev Alwala, Nicolas Carion, Chao-Yuan Wu, Ross Girshick, Piotr Dollár, and Christoph Feichtenhofer. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.

Jonathan Roberts, Mohammad Reza Taesiri, Ansh Sharma, Akash Gupta, Samuel Roberts, Ioana Croitoru, SimionVlad Bogolin, Jialu Tang, Florian Langer, Vyas Raina, Vatsal Raina, Hanyi Xiong, Vishaal Udandarao, Jingyi Lu, Shiyang Chen, Sam Purkis, Tianshuo Yan, Wenye Lin, Gyungin Shin, Qiaochu Yang, Anh Totti Nguyen, David I. Atkinson, Aaditya Baranwal, Alexandru Coca, Mikah Dang, Sebastian Dziadzio, Jakob D. Kunz, Kaiqu Liang, Alexander Lo, Brian Pulfer, Steven Walton, Charig Yang, Kai Han, and Samuel Albanie. Zerobench: An impossible visual benchmark for contemporary large multimodal models. arXiv preprint arXiv:2502.09696, 2025.

Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. Object hallucination in image captioning. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4035–4045, 2018.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, Ruochen Xu, and Tiancheng Zhao. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_ alpaca, 2023.

Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. In The Thirty-eighth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024a. URL https://openreview.net/forum?id= QWTCcxMpPA.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xiong-Hui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for LLM reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https:

//openreview.net/forum?id=yfcpdY4gMP. Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Xiaotao Gu, Shiyu Huang, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. Lvbench: An extreme long video understanding benchmark, 2024b.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), pages 13484–13508, 2023.

Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Haotian Liu, Sadhika Malladi, Alexis Chevalier, Sanjeev Arora, and Danqi Chen. Charxiv: Charting gaps in realistic chart understanding in multimodal llms. arXiv preprint arXiv:2406.18521, 2024c.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, pages 24824–24837, 2022.

xAI. Grok-1.5 vision preview. xAI blog, April 2024. URL https://x.ai/blog/grok-1.5v.

Yijia Xiao, Edward Sun, Tianyu Liu, and Wei Wang. Logicvista: Multimodal llm logical reasoning benchmark in visual contexts. arXiv preprint arXiv:2407.04973, 2024.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2369–2380, 2018.

Junyan Ye, Dongzhi Jiang, Jun He, Baichuan Zhou, Zilong Huang, Zhiyuan Yan, Hongsheng Li, Conghui He, and Weijia Li. Blink-twice: You see, but do you observe? a reasoning benchmark on visual perception. arXiv preprint arXiv:2510.09361, 2025.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge Zhang, Huan Sun, Yu Su, Wenhu Chen, and Graham Neubig. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL), 2025.

Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models. arXiv preprint arXiv:2302.00923, 2023.

Rosie Zhao, Anshul Shah, Xiaoyu Zhu, Xinke Deng, Zhongyu Jiang, Yang Yang, Joerg Liebelt, and Arnab Mondal. On robustness and chain-of-thought consistency of rl-finetuned vlms. arXiv preprint arXiv:2602.12506, 2026.

Yilun Zhao, Lujing Xie, Haowei Zhang, Guo Gan, Yitao Long, Zhiyuan Hu, Tongyan Hu, Weiyuan Chen, Chuhan Li, Junyang Song, Zhijian Xu, Chengye Wang, Weifeng Pan, Ziyao Shangguan, Xiangru Tang, Zhenwen Liang, Yixin Liu, Chen Zhao, and Arman Cohan. Mmvu: Measuring expert-level multi-discipline video understanding, 2025. URL https://arxiv.org/abs/2501.12380.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: Benchmarking multi-task long video understanding. arXiv preprint arXiv:2406.04264, 2024.

Appendix

- A Full Prompt for Multi-Hop Query Design

We provide the complete prompt used for synthesizing multi-hop vision-language reasoning queries below. Placeholders such as {num_queries_word}, {object_list}, {num_queries}, and {target_hop_count_info} are filled at runtime.

##### Multi-Hop Query Design Prompt (Full)

#### **Role & Goal** You are a top-tier AI multimodal capability evaluation expert. Your task is to design a set of **{

num_queries_word}** independent, high-difficulty **multi-hop vision-language reasoning** subqueries based on a **complex image**. These sub-queries will serve as "capability modules" to

evaluate a model's comprehensive visual intelligence. #### **Understanding VLM Perception Capabilities** VLM (Vision-Language Model) perception capabilities can be categorized into three levels:

- 1. **Single-Object Perception (Level 1):** Perceiving attributes of a single object (e.g., color, shape, size, text content, position, category).
- 2. **Multi-Object & Relationship Perception (Level 2):** Perceiving multiple objects and their relationships (e.g., spatial relationships like "A is to the left of B", comparative relationships like "A is larger than B", counting objects that satisfy certain conditions).
- 3. **Multi-Hop Reasoning (Level 3):** Chaining multiple perception steps together. **Multi-hop can occur in TWO dimensions:**

- - **Perception-Level Hops:** Moving between Level 1 and Level 2 perception tasks
- - **Instance-to-Instance Hops:** Reasoning from instance_1 -> instance_2 -> instance_3 -> ...
- -> instance_n, where **finding the next instance DEPENDS ON the previous instance**. The key is that instance_N+1 can ONLY be located by using information from or relationships with instance_N.

**YOUR TASK: Design queries that are strictly Level 3 (Multi-Hop Reasoning).** Each query must chain together multiple perception steps across BOTH dimensions - hopping between different perception levels AND hopping across different instances with DEPENDENCY relationships.

#### **Types of Multi-Hop Reasoning (Design queries using ALL types)**

- **Type A - Instance Dependency Chain (MOST IMPORTANT):**

- - The NEXT instance can ONLY be found by using its relationship with the PREVIOUS instance
- - Each hop must establish a dependency: "the X that is [relationship] to the previous instance"
- - Example: "Start from the red shirt -> find the chair that THIS shirt is draped over -> find the table that THIS chair is placed next to -> count objects on THIS table"
- - Example: "Find person A -> locate the bag that A is holding -> identify the brand logo on THIS bag -> count similar logos in the image"
- - **BAD (no dependency):** "Find the largest car, then find the tallest tree" (tree doesn't depend on car)
- - **GOOD (with dependency):** "Find the largest car, then find the tree closest to THIS car" ( tree depends on which car was found)

- **Type B - Perception Level Hop:**

- - Level 1 (single object) -> Level 2 (multi-object relationship) -> Level 1 -> Level 2 -> ...
- - Each level transition should maintain the instance dependency chain
- - Example: "Read the number on sign A (L1), find objects to the right of THIS sign (L2), among THESE objects find the closest to B (L2), read text on THIS object (L1)"

- **Type C - Combined with Strong Dependencies (PREFERRED):**

- - Combine instance dependency chains with perception level changes for maximum complexity
- - Every instance transition must have a clear dependency relationship
- - Example: "Find shirt A -> locate the chair THIS shirt is draped on -> find the person sitting on THIS chair -> read the text on the book THIS person is holding -> count pages visible in THIS book"

**You are provided with MULTIPLE images:**

- 1. **Image 1 (Original):** The original image containing all visual information. **This is the

- ONLY image that will be available when answering the queries.**
- 2. **Instance Patch Images (Image 2, 3, 4, ...):** Cropped patches of each individual instance from the original image. Each patch shows one specific instance extracted using its bounding box. These patches help you precisely identify each instance's appearance and content. ** These patches are ONLY for your reference during question design and will NOT be available when answering queries.**

**CRITICAL: You are designing queries for a SPECIFIC COMBINATION of objects. The following object

instances are the ONLY instances you should consider for this task. Each instance is provided as a separate cropped patch with its bounding box location in the original image ( coordinates in 0-1000 range):**

{object_list}

**[WARNING] MANDATORY REQUIREMENT - INVOLVE ALL INSTANCES:**

- - **Each sub-query MUST involve AS MANY instances as possible from the combination list above.**
- - **Ideally, every query should reference or require reasoning about ALL or MOST of the listed instances.**
- - If the combination has N instances, each query should involve at least (N-1) instances, preferably all N instances.
- - The `involved_objects` field in your output MUST list ALL instances that are referenced or reasoned about in that query.
- - **DO NOT design queries that only involve 2-3 objects when 5-6 objects are available in the combination.**

**CRITICAL CONSTRAINT - NO PATCH/BOX REFERENCES & UNAMBIGUOUS INSTANCE REFERENCES:**

- - The cropped patches and bounding box coordinates are **ONLY** visual aids to help you understand which object is which during question design.
- - **When writing the actual query text, you MUST NEVER mention:**
- - Detection borders, bounding boxes, patches, or any detection-related visual markers
- - Bounding box coordinates or positions in the 0-1000 coordinate system
- - Any reference to the cropped patch images or instance visualization aids
- - **You MUST describe objects using ONLY:**
- - Spatial positions (e.g., "the object on the left", "the topmost item")
- - Contextual relationships (e.g., "the object next to the table", "the item above the text")
- - Functional descriptions (e.g., "the button", "the icon", "the label")
- - Visual attributes from the original image itself (e.g., actual colors of objects, sizes, shapes)
- - **[WARNING] CRITICAL: UNAMBIGUOUS INSTANCE REFERENCES (MOST IMPORTANT):**
- - **Each instance reference in your query MUST uniquely identify exactly ONE instance in the original image (Image 1) without any borders.**
- - **Test each reference:** Imagine the query is given to someone looking ONLY at Image 1 (no colored borders). Can they identify the EXACT instance you mean without any confusion?
- - **BAD (ambiguous):** "the car" when there are multiple cars in the image
- - **BAD (ambiguous):** "the person on the right" when there are two people on the right side
- - **BAD (ambiguous):** "the red object" when multiple objects are red
- - **GOOD (unambiguous):** "the car closest to the bottom-left corner"
- - **GOOD (unambiguous):** "the person wearing a blue shirt standing next to the lamppost"
- - **GOOD (unambiguous):** "the largest red object that is positioned above the text area"
- - **GOOD (unambiguous):** "the bicycle located between the tree and the bench"
- - **Use multiple discriminating features:** Combine spatial position + visual attributes + contextual relationships to ensure uniqueness
- - **Verify uniqueness:** For each instance reference, mentally check if ANY other object in Image 1 could match that description. If yes, add more distinguishing details.
- - **Reason:** The final questions will be evaluated on Image 1 (the original image) **without any borders, markers, or detection aids**. Any ambiguous reference will make the question

unsolvable or lead to multiple valid interpretations. #### **Core Principles: Requirements for Each Sub-query**

**Every** sub-query you design must strictly adhere to all of the following standards:

- 1. **Independence:** Each query must be completely self-contained. Solving it must not depend on information or answers from any other query.
- 2. **[VERY IMPORTANT] MANDATORY Multi-Hop Structure (MOST CRITICAL):**

- - **Each query MUST be a genuine multi-hop reasoning problem with {target_hop_count_info}.**
- - **Number of hops = {target_hop_count_info}.** Push the boundaries of complexity within this

- range.
- - Each hop can be:
- - **Level 1 (Single-Object Perception):** Extract information from one specific object (e.g ., read text, identify color, determine position)
- - **Level 2 (Multi-Object Relationship):** Reason about relationships between multiple

objects (e.g., compare sizes, count objects satisfying a condition, determine spatial arrangement)

- - **Instance-to-Instance Chaining is REQUIRED:** The reasoning should hop from instance_1 -> instance_2 -> instance_3 -> ... involving as many instances as possible in a chain.
- - **The hops must be logically chained:** The result of each hop must be REQUIRED for the next hop. No hop should be skippable.
- - **Example of a HIGH-QUALITY multi-hop structure (5 hops, all instances involved):**
- - Hop 1 (L1 on A): Read the number on object A -> get value X
- - Hop 2 (L2 on A,B,C): Find objects to the right of A that are smaller than B -> get set S containing C
- - Hop 3 (L1 on C): Identify the color of object C (from set S) -> get color Y
- - Hop 4 (L2 on D,E,F): Count objects among D,E,F that share color Y -> get count Z
- - Hop 5 (L2 on result): Compute X x Z and compare with the area ratio of E to F -> final

answer

- - **INVALID (too few hops):** "How many red objects are there?" (1 hop)
- - **INVALID (too simple):** "What is the color of the largest object?" (2 hops at most)
- - **INVALID (no instance chain):** Query that only touches 2 instances

- 3. **[IMPORTANT] Maximum Instance Coverage (CRITICAL):**

- - **Each query MUST involve ALL instances from the provided combination list - no exceptions

.**

- - Design queries where EVERY instance plays a meaningful role in the reasoning chain.
- - **Good pattern:** "Starting from A, find the instance closest to it (hop to B), then find instances sharing B's color (hop to C,D), compare their sizes with E (hop to E), and count those larger than F (hop to F) -> final answer uses info from ALL instances"
- - **Bad pattern:** "Compare object 1 and object 2" (ignores other instances)

- 4. **[IMPORTANT] High Perception & Logic Difficulty (CRITICAL):**

- - **Perception difficulty:** Each hop should require non-trivial perception:
- - Reading small or partially occluded text
- - Distinguishing subtle color differences
- - Determining complex spatial relationships (e.g., "between", "surrounded by", "aligned with")
- - Counting objects under multiple constraints
- - **Logic difficulty:** The reasoning chain should require:
- - Conditional logic (if-then-else based on visual observations). **IMPORTANT: Ensure that the "if" conditions are NOT always true. Aim for a balanced and random distribution of "Yes" and "No" outcomes for these conditions across different queries and hops to avoid bias.**
- - Set operations (intersection, union, difference of object sets)
- - Numerical computations (ratios, differences, products)
- - Comparative reasoning with multiple criteria
- - **Avoid trivial hops** like "identify that this is a car" - instead require "identify the car that is closest to the pedestrian and has a license plate starting with a letter"

- 5. **Deterministic Numerical Answer:** The final answer to each query must be a specific, unambiguous **number**.
- 6. **Visual Grounding:** All information required to solve the query must be obtainable **only** from the image and must not rely on external knowledge.
- 7. **Avoid "Shortcut" Hints:** Strictly prohibit using low-level visual features (e.g., "the red text," "the largest icon") to locate information. Guidance should be provided through the

object's **function, context, or logical relationships**.

- 8. **Avoid Visual Information Leakage:** The wording of the query **must not contain** any specific text or numbers that the model would need to acquire from the image via visual recognition (e.g., OCR). This ensures the model finds the answer by understanding the image content, not through text matching.
- 9. **Unambiguous Phrasing & Deterministic Solution:**

- - **The query itself must be phrased precisely, allowing for only one logical, unambiguous interpretation within the context of the image.**
- - **The solution process must be completely deterministic with no room for dispute:**

- - Every instance reference must uniquely identify exactly one object (see CRITICAL CONSTRAINT above)
- - Every spatial relationship must have a clear, unambiguous definition (e.g., "to the left of" means the object's center is to the left of the reference object's center)
- - Every comparison criterion must be objective and measurable (e.g., "larger" means larger area, not perceived importance)
- - Every counting task must have clear inclusion/exclusion criteria
- - Every numerical operation must be well-defined
- - **BAD (ambiguous solution):** "Count objects near the tree" - what distance counts as "near "?
- - **GOOD (deterministic):** "Count objects whose center is within a distance of twice the tree's width from the tree's center"
- - **BAD (ambiguous):** "Find the most prominent object" - subjective judgment
- - **GOOD (deterministic):** "Find the object with the largest area"
- - **Verification test:** Two different people with good vision looking at Image 1 should arrive at the EXACT SAME numerical answer through the EXACT SAME reasoning steps.

- 10. **Diversity of Reasoning:** For a set of sub-queries designed for the same image, avoid using the same reasoning logic or template. Each sub-query should test different reasoning paths,

hop combinations, and instance chains. **This includes diversifying the outcomes of conditional steps (mixing "if true" and "if false" paths) so that the correct answer is neither always "Yes" nor always "No".**

- 11. **Capability Coverage:** The {num_queries_word} queries you design, **as a whole**, should cover the following core capability dimensions:

- * **Color Comprehension**
- * **Spatial Reasoning**
- * **Counting & Aggregation**
- * **Text & Symbol Recognition**
- * **Logical & Conditional Inference**

- 12. **No Detection Border/Color References:** **STRICTLY FORBIDDEN** to mention detection borders, border colors, bounding boxes, or any detection-related visual markers in the query text.
- 13. **Balanced Conditional Outcomes (CRITICAL):** For queries involving conditional logic (ifthen-else), you MUST ensure a balanced and random distribution of "Yes" and "No" outcomes for

the conditions. **Do not make the "if" condition always true or always false.** Across all queries and hops, roughly half of the conditional checks should evaluate to "No" (triggering the 'else' or 'otherwise' path). For example, if an object is red, you might design a condition like "if this object is blue, add 5; otherwise, add 2", forcing the reasoning to follow the "otherwise" path. This prevents the model from developing a bias and ensures it actually performs the visual verification.

#### **Workflow & Output Format** Your final output should be a **single JSON object**. This object contains an array named `

sub_queries`, with {num_queries_word} elements, each representing a meticulously designed sub

-query.

**IMPORTANT: The `involved_objects` field MUST list ALL instances from the combination that are

referenced in the query. Aim to involve ALL available instances in each query.** Please strictly follow the JSON structure below: "`json {{

"sub_queries": [ {{

"id": 1, "primary_capability": "Spatial Reasoning + Counting + Conditional Logic", "involved_objects": ["instance_1", "instance_2", "instance_3", "instance_4", "instance_5"], "query": "Your complex multi-hop query here. It should naturally chain through multiple

instances: Start from A, hop to B based on some relationship, then hop to C, then to D, and finally involve E to compute the answer.",

"instance_chain": "A -> B -> C -> D -> E (describe how reasoning flows between instances)", "reasoning_hops": [

{{

"hop_number": 1, "hop_type": "Level 1 (Single-Object)",

"from_instance": "instance_1", "to_instance": null, "description": "Extract specific information from instance_1", "objects_involved": ["instance_1"], "output": "value_X extracted from instance_1"

}}, {{

"hop_number": 2, "hop_type": "Level 2 (Multi-Object Relationship)", "from_instance": "instance_1", "to_instance": "instance_2", "description": "Use value_X to find/filter among other instances, identifying

instance_2", "objects_involved": ["instance_1", "instance_2", "instance_3"], "output": "instance_2 identified based on relationship with instance_1"

}}, {{

"hop_number": 3, "hop_type": "Level 2 (Multi-Object Relationship)", "from_instance": "instance_2", "to_instance": "instance_4", "description": "From instance_2, determine relationship with instance_4", "objects_involved": ["instance_2", "instance_4"], "output": "relationship_result between instance_2 and instance_4"

}}, {{

"hop_number": 4, "hop_type": "Level 1 (Single-Object)", "from_instance": "instance_4", "to_instance": null, "description": "Extract value from instance_4", "objects_involved": ["instance_4"], "output": "value_Y from instance_4"

}}, {{

"hop_number": 5, "hop_type": "Level 2 (Multi-Object Relationship)", "from_instance": "instance_4", "to_instance": "instance_5", "description": "Compare or compute using instance_5 as reference", "objects_involved": ["instance_4", "instance_5"], "output": "final_numerical_answer"

}}

], "hypothetical_answer": "42", "design_rationale": "Explain: (1) The instance chain A->B->C->D->E, (2) Why each hop is

necessary, (3) What makes the perception/logic difficult" }}, {{

"id": 2, "primary_capability": "Text Recognition + Logical Inference + Conditional", "involved_objects": ["ALL instances - list them all"], "query": "Another complex multi-hop query with instance chain...", "instance_chain": "Describe the instance flow", "reasoning_hops": [

- {{"hop_number": 1, "hop_type": "L1", "from_instance": "...", "to_instance": "...", "

description": "...", "objects_involved": [...], "output": "..."}},

- {{"hop_number": 2, "hop_type": "L2", "from_instance": "...", "to_instance": "...", "

description": "...", "objects_involved": [...], "output": "..."}},

- {{"hop_number": 3, "hop_type": "...", "from_instance": "...", "to_instance": "...", "

description": "...", "objects_involved": [...], "output": "..."}},

- {{"hop_number": 4, "hop_type": "...", "from_instance": "...", "to_instance": "...", "

description": "...", "objects_involved": [...], "output": "..."}} ], "hypothetical_answer": "150", "design_rationale": "..."

}}, {{

"id": 3, "primary_capability": "Counting & Aggregation + Spatial + Set Operations", "involved_objects": ["ALL instances"], "query": "Complex multi-hop query...", "instance_chain": "...", "reasoning_hops": ["... {target_hop_count_info} ..."], "hypothetical_answer": "24", "design_rationale": "..."

}}, {{

"id": 4, "primary_capability": "Color + Comparison + Conditional Branching", "involved_objects": ["ALL instances"], "query": "Complex multi-hop query...", "instance_chain": "...", "reasoning_hops": ["... {target_hop_count_info} ..."], "hypothetical_answer": "3", "design_rationale": "..."

}}, {{

"id": {num_queries}, "primary_capability": "Comprehensive: All Capabilities Combined", "involved_objects": ["ALL instances"], "query": "Most complex multi-hop query combining all capability types with longest instance

chain...", "instance_chain": "...", "reasoning_hops": ["... {target_hop_count_info} ..."], "hypothetical_answer": "1800", "design_rationale": "..."

}} ]

}} "`

#### **Examples of Well-Designed Multi-Hop Queries** Suppose the combination contains: person_1, car_1, bicycle_1, sign_1, tree_1

---

**EXCELLENT Example (5 hops, instance chain, all 5 objects, high difficulty, UNAMBIGUOUS references):**

**Context:** Assume in the image: sign is in top-right corner, car is red sedan in center, bicycle is yellow near bottom, person wearing blue shirt in left, tree is large with green leaves in right background.

Query (with unambiguous references): "Starting from the traffic sign located in the top-right corner of the image, read the speed limit number displayed on it. Then find which vehicle either the red car in the center or the yellow bicycle near the bottom edge - has its center point closest to the sign's center. Check whether that vehicle's center is closer to the person wearing the blue shirt on the left side, or to the large tree with green leaves in the

right background. If it's closer to the person, count all objects whose centers lie on the straight line connecting the person and the tree; if closer to the tree, count all objects positioned to the left of the tree's leftmost edge. Finally, multiply the speed limit number by this count, then subtract the total number of wheels visible on both vehicles (the red car

and the yellow bicycle). What is the result?"

- - **Hop 1** (L1 -> sign_1): Read number from "traffic sign in top-right corner" -> speed_limit = 30
- - **Hop 2** (L2 -> sign_1, car_1, bicycle_1): Compare distances: "red car in center" vs "yellow bicycle near bottom" to sign -> closest = car_1
- - **Hop 3** (L2 -> car_1, person_1, tree_1): Compare distances: car's center to "person in blue shirt on left" vs "large tree in right background" -> closer to person_1
- - **Hop 4** (L2 -> person_1, tree_1, car_1, bicycle_1): Count objects on line between "person in blue" and "tree in right" -> count = 2 (car and bicycle)
- - **Hop 5** (L1+L2 -> car_1, bicycle_1): Count wheels: "red car" (4) + "yellow bicycle" (2) -> wheels = 6

- - **Final**: 30 x 2 - 6 = 54

**Why this is unambiguous:**

- - Every instance has multiple distinguishing features (color + position + distinctive attributes)
- - "Red car in center" - unique by color and position
- - "Yellow bicycle near bottom" - unique by color and position
- - "Person wearing blue shirt on left" - unique by clothing and position
- - "Traffic sign in top-right corner" - unique by function and position
- - "Large tree with green leaves in right background" - unique by size, color, position

Instance Chain: sign_1 -> car_1 -> person_1/tree_1 -> car_1,bicycle_1 -> final involved_objects: ["person_1", "car_1", "bicycle_1", "sign_1", "tree_1"] [OK] ALL 5!

- ---

**GOOD Example (4 hops, instance chain, all objects):** Query: "Find the object that is furthest from the person. Then identify all objects that share at

least one visual attribute (color, shape, or size category) with it. Among those objects, find the one closest to the bicycle. Read any text or number visible on or near that object. If no text exists, use the count of objects to its left. What is this value?"

- - **Hop 1** (L2 -> person_1, all others): Find furthest object from person_1 -> sign_1
- - **Hop 2** (L2 -> sign_1, car_1, bicycle_1, tree_1): Find objects sharing attributes with sign_1

-> result set contains tree_1 (both have vertical structure)

- - **Hop 3** (L2 -> tree_1, bicycle_1): Find which object in result set is closest to bicycle_1 -> tree_1
- - **Hop 4** (L1 -> tree_1): Check for text on tree_1, if none, count objects to its left -> 2

Instance Chain: person_1 -> sign_1 -> tree_1 -> bicycle_1 -> tree_1 -> final involved_objects: ["person_1", "car_1", "bicycle_1", "sign_1", "tree_1"] [OK] ALL 5!

- ---

**BAD Example (too few hops, too simple):**

- - Query: "What color is the car?"
- - Problems: Only 1 hop, only 1 object, trivial perception

**BAD Example (few objects, weak chain):**

- - Query: "Read the number on the sign and add it to the count of people."
- - Problems: Only 2 objects involved, only 2 hops, no instance chain

**BAD Example (hops not logically dependent):**

- - Query: "Count red objects and count blue objects. What is the sum?"
- - Problems: The two counts are independent (can be done in parallel), not a true chain

**BAD Example (AMBIGUOUS instance references - CRITICAL FAILURE):**

- - Query: "Find the car, then count objects to its right."
- - Problems:
- - "the car" - which car? If there are 2+ cars, this is ambiguous
- - "to its right" - to the right of what point? Edge? Center? Entire bounding region?
- - Cannot be solved deterministically without borders

**BAD Example (SUBJECTIVE/NON-DETERMINISTIC):**

- - Query: "Find the most important object, then count nearby objects."
- - Problems:
- - "most important" - subjective, not measurable
- - "nearby" - no clear distance threshold defined
- - Two people would give different answers

**GOOD Fix (making it unambiguous and deterministic):**

- - Query: "Find the red sedan car located in the center-bottom area of the image (the only car with visible license plate), then count all objects whose center points are within a horizontal distance of 100 pixels to the right of this car's rightmost edge."
- - Why it works:
- - "Red sedan car in center-bottom with visible license plate" - multiple features ensure uniqueness
- - "Center points" - clear reference point

- - "Horizontal distance of 100 pixels to the right of rightmost edge" - precise, measurable criterion
- - Completely deterministic

#### **Final Checklist Before Outputting** For EACH of your {num_queries_word} sub-queries, verify:

- - [ ] **Hop Count:** Does it have {target_hop_count_info}?
- - [ ] **Instance Chain:** Does the reasoning hop from instance to instance (A->B->C->...)?
- - [ ] **Full Coverage:** Does `involved_objects` contain ALL instances from the combination?
- - [ ] **Logical Dependency:** Is the result of each hop REQUIRED for the next hop? (no parallel/ skippable hops)
- - [ ] **Perception Difficulty:** Does each hop require non-trivial perception (not just "identify object type")?
- - [ ] **Logic Complexity:** Does the chain include conditionals, set operations, or multi-step computations?
- - [ ] **Numerical Answer:** Is the final answer a specific, computable number?
- - [ ] **No Border References:** Does it avoid mentioning detection borders or visualization colors?
- - [ ] **[VERY IMPORTANT] Unambiguous Instance References (CRITICAL):** For EACH instance reference in the query, verify:
- - Can someone looking ONLY at the original image (no borders) uniquely identify this exact instance?
- - Would ANY other object in the image match this description? If yes, add more distinguishing details.
- - Does the description use multiple discriminating features (spatial + visual + contextual)?
- - [ ] **[VERY IMPORTANT] Deterministic Solution Process (CRITICAL):** Verify:
- - Would two different people arrive at the EXACT SAME answer?
- - Are all spatial relationships clearly defined? (e.g., "left of" = center is to the left)
- - Are all comparisons objective? (e.g., "larger" = larger area, not importance)
- - Are all counting criteria explicit? (what counts as "between", "near", etc.)
- - Is there NO room for subjective interpretation?

**Quality Scoring (aim for 10+):**

- - 1 point per hop
- - 1 point if ALL instances involved
- - 1 point if instance chain covers 4+ different instances
- - 1 point if includes conditional logic (if-then)
- - 1 point if includes numerical computation beyond counting
- - **2 points if ALL instance references are unambiguous and solution is deterministic (CRITICAL)**

**Please Note:**

- - Your task is to generate the JSON content that conforms to the format and requirements above.
- - **CRITICAL REMINDERS:**
- - Each `query` field must be answerable using ONLY the original image (Image 1) without any detection borders or markers.
- - **EVERY query MUST be multi-hop with {target_hop_count_info}.**
- - **EVERY query MUST involve ALL or MOST of the instances in the provided combination list.**
- - The `involved_objects` field must accurately list all instances referenced in the query.

- B Full Prompt for Image Filtering We also provide the complete prompt used to screen images before multi-hop query synthesis.

##### Image Filtering Prompt (Full)

# Role and Goal You are a professional AI Image Analyst. Your primary goal is to evaluate the complexity of a

given image from the perspective of a standard computer vision model. This analysis will identify images that pose a challenge to AI perception, especially in tasks like object detection, counting, and recognition.

# Core Task For the image I provide, you must:

- 1. Evaluate its overall perceptual complexity on a scale of **1 to 10**.
- 2. Assess the image quality (High / Medium / Low).
- 3. Identify the specific objects or areas that contribute to the complexity.
- 4. Output the results as a **single JSON object** as required, with no explanatory text outside of the JSON.

# Key Definitions

- ## 1. Image Complexity Factors (What makes an image complex for AI?)

- * **Occlusion**: Objects are partially hidden by other objects. The more a key object is occluded, the higher the complexity.
- * **Object Count & Density**: The image contains a large number of distinct objects, especially of the same category (e.g., a crowd of people, a fleet of cars), making counting and

individual identification difficult.

- * **Unusual Pose/Angle**: Objects are shown from unconventional viewpoints (e.g., a person doing a handstand, a car seen directly from above).
- * **Complex Interaction**: Objects interact in a way that blurs their individual boundaries (e. g., people hugging, tangled wires, a pile of clothes).
- * **Fine-grained Recognition**: Requires distinguishing between very similar sub-categories (e. g., telling two different bird species apart, or different models of the same car brand).
- * **Challenging Lighting/Shadows**: Harsh shadows, reflections, or poor lighting obscure the shape and details of objects.

- ## 2. Image Quality Guidelines (How to handle low-quality or unusable images) Your analysis must focus on clear, perceptible objects.

- * **High/Medium Quality**: The image is generally clear, and its main subjects are recognizable, even if the scene is complex.
- * **Low Quality**: An image should be rated "Low Quality" when it is unusable for precise perceptual analysis due to any of the following reasons:

- * **Technical Flaws:** The image is too blurry, noisy, severely overexposed, or underexposed, making objects unrecognizable.
- * **Annotation Impracticality:** **Even if the image itself is clear, the extreme number and density of objects make meaningful, verifiable annotation impossible (e.g., it's impossible to draw a bounding box for or accurately count every single object). Such images lack a "Ground Truth" for model evaluation.**

- * **Crucial**:
- * **Case 1 (Technical Flaws):** If a scene is chaotic due to blur or lighting issues, making it impossible to distinguish individual objects, its quality should be rated **"Low"**

with a **low complexity score (1-3)**.

- * **Case 2 (Annotation Impracticality):** **If a scene contains a vast, uncountable collection of objects (e.g., a massive but clear crowd where individuals cannot be boxed or counted; a bookshelf or rack packed with countless small items), its quality should also be rated "Low," even if the image is technically clear.**
- * **Conclusion:** **For both of the "Low Quality" cases described above, the perceptual complexity score should be low (1-3)**, as they are not good test cases for perception but rather issues of data usability or quality itself.

# Output Format Requirements The output must be a structured **JSON object** (with no additional text). The structure is as

follows:

```json {

"overall_complexity_score": <Integer, 1-10>, "overall_quality_rating": "<High | Medium | Low>", "complexity_analysis": "<String, a brief explanation for the score, referencing the complexity

factors above>", "complex_objects": [ {

"object_name": "<Description of the specific object>", "generalized_name": "<General category like Person, Animal, Vehicle, Plant, etc.>", "reason_for_complexity": [

"<Select from the list of complexity factors, e.g., 'Occlusion', 'Fine-grained Recognition'>"

] }

]

} ```

# Final Instruction After I provide an image, strictly follow all the rules above, perform the analysis based on the

actual image content, and output a JSON object that strictly adheres to the specified format.

