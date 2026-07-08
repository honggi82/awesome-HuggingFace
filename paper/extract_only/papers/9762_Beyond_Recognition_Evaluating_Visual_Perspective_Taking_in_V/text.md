## Beyond Recognition: Evaluating Visual Perspective Taking in Vision Language Models

# arXiv:2505.03821v2[cs.CV]28Mar2026

Gracjan G´oral1,2,6∗ Alicja Ziarko1,2 Piotr Miło´s1,2 Michał Nauman1,4 Maciej Wołczyk5 Michał Kosi´nski3

1University of Warsaw 2Polish Academy of Sciences 3Stanford University 4University of California, Berkeley 5 IDEAS NCBR 6Lute

∗Corresponding author: gp.goral@uw.edu.pl

#### Abstract

We investigate the ability of Vision Language Models (VLMs) to perform visual perspective taking using a new set of visual tasks inspired by established human tests. Our approach leverages carefully controlled scenes in which a single humanoid minifigure is paired with a single object. By systematically varying spatial configurations—such as object position relative to the minifigure and the minifigure’s orientation—and using both bird’s-eye and surfacelevel views, we created 144 unique visual tasks. Each task is paired with a series of 7 diagnostic questions designed to assess three levels of visual cognition: scene understanding, spatial reasoning, and visual perspective taking. We evaluate several high-performing models, including Gemini Robotics-ER 1.5, Llama-3.2-11B-Vision-Instruct, and variants of Claude Sonnet, GPT-4, and Qwen3, and find that while they excel at scene understanding, performance declines markedly on spatial reasoning and deteriorates further on perspective taking. Our analysis suggests a gap between surface-level object recognition and the deeper spatial and perspective reasoning required for complex visual tasks, pointing to the need for integrating explicit geometric representations and tailored training protocols in future VLM development.

#### 1. Introduction

Recent advances in vision–language modeling have produced systems capable of jointly reasoning over images and text [1, 6, 8]. These models are being applied across diverse domains, including robotics and autonomous driving [12, 42], as well as healthcare [14, 17, 50]. However, excelling in such applications requires the ability to reason about what others can and cannot see, going beyond tra-

ditional scene recognition or language understanding. For instance, an autonomous vehicle must anticipate what a nearby driver can or cannot see, while a surgical or industrial robot should assess whether a human coworker can visually locate an object before passing it. Reasoning about visual perspectives is therefore essential for safe, collaborative, and socially aware embodied systems.

In humans, the ability to adopt another’s visual vantage point, known as visual perspective taking (VPT) [13, 26, 39], is a core component of theory of mind [18]. VPT supports a wide range of cognitive and social functions, from spatial navigation to joint action coordination, and impairments in VPT have been linked to difficulties in both domains [35, 37]. Many cognitive abilities that have been central to the development of machine learning, such as perception and reasoning, were first characterized in humans, with their testing paradigms later adapted to evaluate machines. Because VPT has been extensively studied in psychology with well-established tasks, it provides a natural foundation for assessing similar abilities in foundation models.

Recent studies suggest that large language models can track the beliefs of different characters, effectively reasoning about the world through the eyes of their minds in the linguistic domain [16, 22, 46]. This raises the question of whether similar perspective-taking abilities extend to the visual domain. To address this, we build on the rich psychological literature on human VPT [21, 24] and introduce Isle-Bric-V2, a controlled dataset of 144 manually created visual tasks by systematically varying spatial configurations of humanoid minifigures and objects, with each image accompanied by 7 questions testing scene understanding, spatial reasoning, and visual perspective taking. We use this dataset to investigate whether VLMs can take others’ visual perspectives—that is, whether they can see the world through another’s eyes

100

Correctness(%)

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

75

Prediction

50

25

0

Scene Understanding Spatial Reasoning Visual Perspective Taking

Models

GPT-4-turbo Claude-3-Sonnet Qwen-3-2B-Instruct

GPT-4o Claude-3.5-Sonnet Qwen-3-4B-Instruct

Gemini Robotics ER 1.5 Llama-3.2-11B-Vision Qwen-3-8B-Instruct

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| |
|---|

- Figure 1. Prediction correctness across three categories of growing difficulty: scene understanding, spatial reasoning, and visual perspective taking. Error bars represent 95% confidence intervals (estimated using bootstrapping (10,000 iterations)). The random classifier is a baseline choosing an answer uniformly at random, see A.3.

Unlike large-scale benchmarks that prioritize breadth and are susceptible to data contamination [7], our approach emphasizes experimental control and variable isolation. We adopt a minimal-contrast methodology inspired by cognitive science: conditions are constructed so that they differ in only a single cognitively relevant factor (e.g., the humanoid minifigure’s orientation or the camera viewpoint), while all other aspects of the scene are held constant. This is analogous to classic dot-perspective paradigms in psychology, where otherwise identical displays differ only in what an avatar can see [36, 40], and to cognition-inspired benchmarks for language models such as COMPS, which use conceptual minimal pairs to probe structured knowledge and world modeling [30].

Contributions Our work makes three key contributions:

- (1) We introduce Isle-Brick-V2, a manually created, psy-

chologically grounded diagnostic benchmark for visual perspective taking that uses open-ended questions to cleanly separate scene recognition, spatial reasoning, and perspective taking abilities on the same set of stimuli.

- (2) We systematically evaluate nine recent VLMs on this

benchmark, revealing consistent VPT failures and a persistent directional prior that remains robust under multiple visual and text interventions.

- (3) We present empirical evidence that accurate ori-

entation detection does not guarantee correct perspectivetaking. Our findings are consistent with a partial dissociation between orientation detection and perspective-taking in our setting, motivating future architectures and training protocols that incorporate explicit egocentric or geometric representations.

#### 2. Contributions in Context of Related Work

Psychological foundations. Within the psychological literature, VPT is often conceptualized along two hierarchically organized levels [15, 31, 41]. Level 1 concerns understanding what others can or cannot see (e.g., Do they

see the object?), whereas Level 2 involves mentally adopting another’s vantage point to determine how objects appear from that perspective (e.g., From their viewpoint, is the object to the right or left?). In adults, Level-1 judgments are typically supported by rapid line-of-sight computations [20, 29], whereas Level-2 judgments require more effortful mental transformations and are associated with mental rotation costs [19, 47]. The mere presence of others can trigger spontaneous perspective changes [52], although the degree of automaticity in VPT remains debated [47].

Our contribution. We explicitly ground our diagnostic tests in the Level-1 / Level-2 framework to separate qualitatively different forms of VPT failure. The same visual stimuli are probed with questions targeting visibility (Level 1) and viewpoint-dependent spatial relations (Level 2), yielding a testing procedure that is in principle applicable to both human participants and VLMs.

Spatial reasoning and VPT in VLMs. Recent VLMs such as GPT-4, Gemini, and Claude report strong performance on a variety of visual and spatial benchmarks probing object relations, counting, and compositional reasoning [2, 4, 34]. Specialized approaches like SpatialVLM and SpatialRGPT further improve spatial understanding by leveraging internet-scale 3D data, metric depth, and regionlevel grounding [9, 10], while ScanQA and SQA3D extend visual question answering to RGB-D and 3D environments [5, 27]. The 3D-PC benchmark evaluates visual perspective taking via depth-ordering and line-of-sight classification in natural scenes [23], and Omni-Perspective scales multimodal ToM-style evaluation to thousands of question–context–answer tuples over real images and a multilevel hierarchy of social and mental-state reasoning [54].

Our contribution. In contrast to 3D-PC and OmniPerspective, which probe VPT in natural scenes via depthorder or multi-choice ToM-style queries [23, 54], we introduce a psychology-grounded visual task benchmark. By asking a hierarchy of seven questions that range from scene recognition to visual perspective taking on the same stimuli, we can disentangle failures of basic object recognition from

[Figure 1]

[Figure 2]

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

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

- Figure 2. Sixteen tasks involving a single humanoid minifigure–object pair. Tasks vary by the object’s placement (left, right, front, back); the orientation of the humanoid minifigure (facing toward or away from the object); and camera angle (surface-level and bird’s-eye views). All images had the same dimensions, but some are enlarged here for presentation purposes.

failures of spatial reasoning and perspective taking in a way that is difficult to achieve with existing large-scale 3D VPT benchmarks.

- 3. Experimental Setup

nipulation. Thus, our approach complemented previous work relying on web-scraped images that may be part of training data [49] or used abstract visual elements such as dots and arrows overlaid on real images [23].

##### 3.2. Diagnostic Questions

##### 3.1. Visual Tasks

Question Design. Each visual task was paired with a set of seven questions, see Table 1. Questions Q1, Q2, and

We developed 144 novel visual tasks inspired by the established human VPT tests [25, 32] to ensure the models had not encountered them during training and could not simply recall memorized solutions. The visual tasks represented a humanoid minifigure and an object placed on the same surface. We used nine distinct humanoid minifigures (varying in hairstyle, clothing, and accessories) and nine distinct objects: a plant, a wardrobe, a cat, a dog, a goblet, a chair, a desk, a bat, and a computer monitor. For each minifigureobject pair, we systematically varied their spatial positions, see Figure 2. These variations resulted in a total of 144 tasks (9 pairs × 4 spatial positions × 2 orientations × 2 viewpoints), which can be downloaded from [LINK].

- Q3 focused on scene understanding: Q1 asked for the total number of objects, Q2 asked for the number of humanoid minifigures, and Q3 inquired whether the humanoid minifigure and objects shared the same surface. Questions
- Q4 and Q5 tested spatial reasoning: Q4 required identifying the object’s location relative to the humanoid minifigure, and Q5 queried the humanoid minifigure’s facing direction. Finally, Q6 and Q7 evaluated visual perspective taking: Q6 checked if the humanoid minifigure could see the object (Level-1), while Q7 asked for the object’s location from the humanoid minifigure’s perspective (Level-2). Open-Ended Format. To reduce the influence of guessing and to avoid position-based artifacts common in multiple-choice evaluations [38], all questions were pre-

The use of LEGO elements further enabled precise control over image and scene properties without post hoc ma-

sented in an open-ended format. Models were not constrained to choose from a fixed set of options and could produce free-form text, although we only scored the final answer. This setup matches typical user interactions and avoids hand-crafted, model-specific prompting strategies. Each question was answered independently in a zero-shot manner (with the context cleared between questions), using temperature 0 to minimize variance, and the maximum response length was capped at 128 tokens.

Gold-Standard Answers. To create the gold-standard answers for each of the 144 visual tasks paired with each of the 7 questions, three research assistants independently responded, with the initial agreement rate exceeding 99%. The rare disagreements were resolved through discussion. Multiple gold-standard answers were allowed for certain questions, e.g., in Q5, where multiple cardinal directions could be correct (north and east for the humanoid minifigure facing northeast). The distribution of gold-standard answers for each question is presented in A.1.

##### 3.3. Evaluation Procedure

Two research assistants processed the model’s responses, extracting the key answer components. This included details like object counts (Q1, Q2), spatial relationship confirmations (Q3), line-of-sight judgments (Q6), egocentric locations (Q7), and the cardinal directions (Q4, Q5). Rare disagreements, less than 1%, were resolved by a third person. Combined answers (like northeast) were transformed into sets of basic directions (north, east). We evaluated performance using averaged prediction correctness. This metric calculates precision for each answer: it is the fraction of components in the model’s answer that are present in the gold-standard answer set. For example, if the model predicted northeast (components: north, east) against a goldstandard answer of north, the prediction correctness is one correct component divided by two predicted components, yielding 0.5. This approach quantifies partial correctness and reduces to standard accuracy for single-component answers. Further details are in Appendix A.2.

#### 4. Results

Models Evaluated. We tested nine popular models, including open source models – Llama-3.2-11B-VisionInstruct (11 December 2024) [28], Qwen3 (2B, 4B, and 8B)-Instruct [55] (1 November 2025) – and five closed source models – GPT-4-Turbo (9 April 2024), GPT-4o (6 August 2024) [33], Claude 3 Sonnet (29 February 2024), Claude 3.5 Sonnet (20 June 2024) [3] and Gemini RoboticsER 1.5 (March 12, 2025) [51]. Models’ performance for each question is presented in Table 2. Figure 1 presents the performance averaged by categories. All error bars repre-

sent 95% CIs calculated using bootstrapping (10,000 iterations).

Random Classifier. Figure 1 also plots a random classifier baseline, i.e. the performance achieved by selecting answers uniformly at random from the permissible answer pool, for more details, see A.3.

Scene Understanding. All nine models performed strongly on scene understanding tasks, reflecting their ability to recognize objects and count humanoid minifigures. GPT-4o and Gemini Robotics-ER 1.5 achieved perfect performance at 100.0% +0−0..00 prediction correctness, closely followed by Qwen3-4B-Instruct (99.8% +0−0..25), Qwen3-8B-Instruct (99.1% +1−1..22), and GPT-4-Turbo (97.5% +1−2..91). Claude-3.5-Sonnet (96.5% +2−3..52), Claude 3 Sonnet (95.8% +3−3..07), and Llama-3.2-11B-Vision-Instruct (92.4% +3−3..59) also demonstrated high performance. Qwen32B-Instruct achieved 76.4% +3−3..79 overall, with particularly low performance on Q1 (34.7% +7−7..66) driven by occasional misclassification of the white background as a non-humanoid minifigure or as a white brick, though it recovered to near-perfect performance on Q2 and Q3. This suggests that identifying what is in the scene – in this instance, how many humanoid minifigures or objects are present, and whether they share the same surface – has become a relatively routine task for modern VLMs, with smaller models occasionally struggling with certain visual ambiguities.

Spatial Reasoning. Models fare considerably weaker on spatial reasoning tasks. Although they performed relatively well when localizing objects relative to the humanoid minifigure (Q4), the prediction correctness dropped significantly when the models had to determine the humanoid minifigure’s facing direction (Q5). We hypothesize this discrepancy arises because Q4 involved an extrinsic reference frame, where objects are localized relative to the humanoid minifigure using the fixed cardinal directions established by the image orientation. Q5, however, required understanding an intrinsic reference frame based on the humanoid minifigure’s own orientation, demanding that models interpret body posture cues to determine facing direction.

For Q5, we observed that GPT-4-Turbo, Claude 3 Sonnet, Qwen3 (2B, 4B, and 8B)-Instruct, and Llama-3.2-11BVision-Instruct were susceptible to directional bias, see Figure 3. Namely, they favored certain cardinal directions, for example, GPT-4-Turbo focused on east and south, completely omitting other directions. For this model, we ran additional detailed experiments in Section 4.2. Namely, we systematically tested variations such as removing secondary objects, zooming in on humanoid minifigures, explicitly labeling cardinal directions (N, S, E, W) in the visual tasks, and even replacing humanoid minifigures with human faces.

- Table 1. List of the diagnostic questions with an example answer by GPT-4-Turbo for the task shown in the top-left panel of Figure 2.

Question Ability GPT-4-Turbo’s answer Scene understanding

- Q1: List and count all objects in the image that are not humanoid minifigures.

Object recognition, counting [45]

In the image, there is one object that is not a humanoid minifigure: 1. A small black dog figure. So, there is a total of 1 non-humanoid object in the image.

- Q2: How many humanoid minifigures are in the picture? Object recognition, counting [45]

There is one humanoid minifigure in the picture.

- Q3: Are the humanoid minifigure and the object on the same surface? Object placement [44] Yes, both the humanoid minifigure and the dog are on the same flat, white surface.

Spatial reasoning

- Q4: Assuming the top of the image is north, in which cardinal direction (i.e., north, west, east, or south) is the object located relative to the humanoid minifigure?

Spatial reasoning [53] The dog is located to the west of the humanoid minifigure, assuming the top of the image is north.

- Q5: Assuming the top of the image is north, which cardinal direction (i.e., north, west, east, or south) is the humanoid minifigure facing?

Mental rotation, spatial reasoning [43]

The humanoid minifigure is facing east.

Visual perspective taking

- Q6: Assuming that the humanoid minifigure can see and its eyes are open, does it see the object?

Understanding what others see (Level-1) [41]

Yes, assuming the humanoid minifigure can see and its eyes are open, it appears to be facing the dog, so it would be able to see the dog.

- Q7: From the perspective of the humanoid minifigure, where is the object located relative to it? Please use terms such as front, left, right, or back.

Adopting others’ perspective (Level-2) [48]

From the perspective of the humanoid minifigure, the dog is located to its left side.

- Table 2. Each question was evaluated across 144 visual tasks in three categories: Scene Understanding (random baseline: 38.9%), Spatial Reasoning (random baseline: 31.7%), and Visual Perspective Taking (random baseline: 41.1%). indicates the best closed model performance, while indicates the best open-source model performance. For VPT tasks, the best closed model (GPT-4o) achieves

+32.15pp above random, while the best open-source model (Qwen3-4B-Instruct) achieves +4.75pp above random.

Gemini Robotics ER 1.5

Llama 3.2-11B Vision-Instruct

Question GPT-4 Turbo GPT-4o

Claude 3 Sonnet Claude 3.5 Sonnet

Qwen3-2B-Instruct Qwen3-4B-Instruct Qwen3-8B-Instruct

Scene Understanding

- Q1 97.2% +2−2..18 100.0% +0−0..00 100.0% +0−0..00 96.5% +2−3..85 95.8% +2−3..85 98.6% +1−2..41 34.0% +7−7..66 99.3% +0−1..74 97.2% +2−2..18
- Q2 95.1% +3−3..55 100.0% +0−0..00 100.0% +0−0..00 94.4% +3−4..52 97.9% +2−2..18 95.8% +2−3..85 95.1% +3−3..55 100.0% +0−0..00 100.0% +0−0..00
- Q3 100.0% +0−0..00 100.0% +0−0..00 100.0% +0−0..00 96.5% +2−3..85 95.8% +2−3..85 82.6% +5−6..62 100.0% +0−0..00 100.0% +0−0..00 100.0% +0−0..00 Spatial Reasoning

- Q4 83.3% +5−6..62 98.6% +1−1..47 95.8% +2−3..85 79.9% +6−6..29 89.9% +4−4..59 84.8% +4−5..82 91.0% +4−4..29 97.2% +2−2..18 87.5% +4−5..96
- Q5 41.7% +8−8..33 72.9% +6−7..96 64.6% +7−7..66 29.9% +6−7..96 55.6% +7−8..63 38.6% +7−8..63 36.1% +7−8..63 46.5% +8−8..33 47.2% +8−8..33 Visual Perspective Taking

- Q6 48.6% +7−8..63 87.5% +4−5..96 59.0% +7−8..63 38.9% +7−8..63 56.2% +8−8..33 49.3% +8−8..33 49.3% +8−8..33 54.2% +8−8..33 54.2% +7−8..63

- Q7 30.2% +7−7..33 59.0% +7−8..60 39.6% +7−8..63 34.0% +7−7..66 35.1% +6−6..99 31.9% +7−7..66 20.1% +6−6..29 37.5% +7−7..66 36.8% +7−7..66

None of these was able to significantly mitigate the GPT-4Turbo’s directional bias. This suggests that some models may rely on linguistic priors or memorized defaults rather than genuinely engaging in spatial reasoning.

Visual Perspective Taking. This task assessed models on two VPT levels: determining if the humanoid minifigure saw an object (Q6; Level-1) and identifying the object’s relative position from the humanoid minifigure’s perspective (Q7; Level-2). Among open-source models, Qwen3-

4B-Instruct achieved the highest performance, reaching 54.2%+8−8..33 on Q6, though this represented only a modest 4.2 percentage point improvement over the random baseline. In comparison, GPT-4o performed well in Q6 (87.5%+4−5..96), making fewer errors than its peers. By contrast, Claude 3 Sonnet often failed Q6 by rejecting the question’s premise

- – insisting that a humanoid minifigure cannot really see
- – a recurring error pattern that occurred systematically in 39/144 instances. These issues, largely fixed in Claude

Qwen-3-2BInstruct

Qwen-3-4BInstruct

Qwen-3-8BInstruct

Qwen-3-2BInstruct

Qwen-3-4BInstruct

Qwen-3-8BInstruct

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

West

Right

South

Left

North

Front

East

Back

100

Gemini Robotics ER 1.5

Gemini Robotics ER 1.5

|[Figure 23]| |
|---|---|
| | |

Claude-3.5Sonnet

Claude-3Sonnet

Claude-3.5Sonnet

Claude-3Sonnet

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

West

Right

Row%

South

Left

50

North

Front

East

Back

0

Llama-3.2-11BVision

Llama-3.2-11BVision

GPT-4-turbo

GPT-4o

GPT-4-turbo

GPT-4o

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

West

Right

South

Left

North

Front

East

Back

BackFrontLeftRight

BackFrontLeftRight

BackFrontLeftRight

EastNorthSouthWest

EastNorthSouthWest

EastNorthSouthWest

- Figure 3. Row-normalized co-occurrence heatmaps for nine models on Q5 (cardinal directions; left) and Q7 (egocentric directions; right). Q5 reveals strong directional bias: GPT-4-Turbo collapses onto East (with South as a frequent fallback); GPT-4o strongly favors East (and somewhat South); Llama-3.2-11B-Vision concentrates on South; and Claude-3-Sonnet almost always predicts West. Qwen3-(2B/4B/8B)Instruct systematically underuse North, with 2B over-predicting East and 4B particularly favoring West/North. In contrast, Q7 exhibits milder egocentric asymmetries, with probability mass drifting toward Right/Left (e.g., GPT-4-Turbo, Llama-3.2-11B-Vision) or Front/Right (GPT-4o). Collectively, these patterns reveal a systematic preference for specific default bearings rather than balanced spatial reasoning.

- 3.5 Sonnet, primarily involved rejecting the premise or misidentifying objects. For example, common premise rejection errors included statements like ...inanimate LEGO toy, it does not possess actual vision... Similarly, object misclassification occurred, such as when the model stated There is no dog visible; the black piece appears to be a weapon... However, for Q7, all models had difficulties, frequently misclassifying objects located behind the humanoid minifigure as being to its left or right, as indicated on cooccurrence matrices shown in Figure 3.

This discrepancy between detecting scene content and simulating the humanoid minifigure’s true perspective highlights a critical shortfall in current VLMs. Recognizing objects does not necessarily equate to robust geometric reasoning or an inferential grasp of spatial relationships – cognitive skills in humans linked to mental rotation and perspective taking. Understanding these shortcomings requires further studies. In this work, we performed a simple analysis starting from the observation that both Q6 and Q7 can be perceived as a combination of Q4 and Q5 and very simple reasoning (e.g. for Q6, the answer is positive iff the object is located in the same direction as the humanoid minifig-

ure is facing, these are determined answering Q4 and Q5 resp.). Following that, one could hypothesize that poor performance on Q5 is the root cause of problems with Q6 and Q7. To test it, we added the ground truth Q5 answer (i.e. the humanoid minifigure’s facing direction) to the prompt for Q6. It turned out that this results in modest improvements, suggesting that the aforementioned decomposition is not entirely accurate. We provide more discussion in 4.1.

##### 4.1. Orientation vs Perspective Test

We started by looking at how models handle questions about what a humanoid minifigure in an image might be seeing (Q6). One plausible route to answering such questions is to first infer which direction the humanoid minifigure is facing (Q5). For example, if an object is to the north and the humanoid minifigure is facing north, then the humanoid minifigure likely sees the object. This led us to ask: are the difficulties models have with visual perspective taking (Q6) mainly driven by problems identifying the humanoid minifigure’s facing direction (Q5)?

To investigate this, we ran an experiment across 144 visual tasks. We specifically tested the VLMs on Q6, but

surface-level bird’s eye

100

Q6

Q6 with cardinal direction the humanoid minifigure was facing

80

[Figure 36]

[Figure 37]

PredictionCorrectness(%)

withobjectwithoutobject

60

40

20

0

GPT-4o GPT-4TurboClaude3.5SonnetClaude3SonnetQwen3-2B-InstructQwen3-4B-InstructQwen3-8B-Instruct

[Figure 38]

[Figure 39]

- Figure 4. Comparison of VLM performance on visual perspective taking (Q6) with and without an explicit orientation hint (goldstandard Q5 answer), showing only marginal prediction correctness improvement.

with a helpful modification to the prompt: we explicitly told the model which cardinal direction the humanoid minifigure was facing, using the gold-standard answer from Q5 as a hint. Our reasoning was that if the struggle with Q5 was the primary bottleneck for Q6, providing this directional information should lead to a significant performance increase. However, as shown in Figure 4, adding this hint resulted in only marginal improvements. This suggests that the models’ difficulties with visual perspective taking might be more complex than simply getting the orientation wrong, and solving Q5 alone does not automatically lead to correctly answering Q6.

##### 4.2. Directional Bias

In the results section, we described how models struggle with spatial judgments, particularly when determining the facing direction of a humanoid minifigure in Q5. We observed that some VLMs, such as GPT-4-Turbo, suffer from directional bias, e.g., favoring directions like east or south. We aim to determine the source of the observed directional biases: do they originate primarily from challenges in visual perception, from how the prompt language is interpreted, or inherently biased, or from more fundamental issues within the model’s spatial reasoning capabilities?

To this end, we are systematically investigating how specific interventions influence their judgments. We explored two paths: modifying the visual input and manipulating the textual prompts.

Removing Objects. We hypothesized that extraneous objects might contribute to GPT-4-Turbo’s directional bias. To test this, we removed items like cats, chairs, etc., from the test images, see Figure 5, and re-ran the 36 visual tasks assessing humanoid minifigure orientation (e.g., surfacetowards-left, birds-eye-towards-right). Although removing objects caused pointwise changes in 5 predictions and

- Figure 5. Illustration of the object removal process, used in investigating if the presence of contextual objects impacts the model’s orientation predictions (Q5). Top: visual tasks with objects. Bottom: corresponding visual tasks without objects. Left/Right: surface-level/bird’s-eye.

slightly increased the frequency of south answers (from 1 to 5), east remained the vastly predominant response (31 times). Therefore, object removal did not substantially reduce the model’s tendency towards east, indicating this bias is robust and not merely an artifact of contextual items.

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

bird’seyesurface-level

- Figure 6. Examples of humanoid minifigure orientation tasks (Q5) at different zoom levels. Left to right: original image with 10% zoom, 30% zoom, and 50% zoom.

Zoom. To test if the bias was related to perceiving details, we conducted an experiment gradually zooming in (10%, 30%, 50%) on the visual task without object, see Figure 6. Performance on the 36 tasks remained poor, with prediction correctness consistently around 44.4%+16−16..77 (e.g., 44.4%+16−16..77/ 41.7%+16−16..77/ 47.2%+16−16..77 for 10%/30%/50%

zoom). Importantly, this low accuracy was accompanied by the same persistent directional bias: GPT-4-Turbo continued to predominantly output east, irrespective of zoom level. This persistence, even when orientation details were magnified, suggests the bias is not merely a perceptual limitation regarding fine details, but potentially points to deeper issues in the model’s spatial reasoning.

###### surface-level bird’s eye

[Figure 44]

[Figure 45]

- Figure 7. Surface-level (left) and bird’s-eye (right) views of the visual task, showing N S E W marks on the image used for the on-visual task cardinal hints experiment.

On-Visual Task Cardinal Hints. To isolate referenceframe ambiguity, we overlaid N E S W markers on each visual task, as illustrated in Figure 7. Even with these cues, prediction correctness remained low at 34.3%+14−17..31, and the model still selected east in 27 of the 36 trials. Because every stimulus explicitly provided both the scene geometry and its coordinate system, any residual error could have arisen from the model’s internal mapping between visual layouts and directional tokens rather than from mis-perceiving north. In this light, the persisting bias may reflect a hardwired prior embedded somewhere in the model’s spatialreasoning pipeline.

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

surface-levelbird’seye

- Figure 8. Surface-level (top row) and bird’s-eye (bottom row) views of a real person used in the influence of subject type experiment.

Influence of Subject Type. To investigate if the observed directional bias was specific to the subject type (i.e., a plastic humanoid minifigure versus a real person), we used 8 images of a real person, consisting of 4 surface-level and 4

bird’s-eye view shots, as shown in Figure 8. Accordingly, we modified the prompt to ask about the direction the person was facing, replacing humanoid minifigure. Notably, for all 8 images, GPT-4-Turbo responded that the person was facing east (e.g., answering The person in the image is facing east). This unanimous east prediction mirrored the strong bias observed with the humanoid minifigure images. Consequently, this persistence suggests that the directional bias is not solely attributable to the specific nature or structure of the humanoid minifigure itself, but may stem from a more general aspect of the model’s processing.

#### 5. Discussion

Our synthetic setting represents a lower bound for VLM spatial reasoning failures: the controlled environment provides perfect lighting, no occlusions, and distinct object separation. That VLMs fail under these optimal conditions suggests deficits fundamental to their spatial reasoning capabilities, not artifacts of perceptual noise in natural datasets (e.g., ScanNet [11]). The persistent directional biases and failure of multiple interventions—including explicit orientation cues, zooming, cardinal markers, and real human subjects—indicate models cannot compute perspective-dependent spatial relations, instead relying on linguistic priors over visual facts. We uncover a systematic mismatch between model predictions and ground truth on VPT tasks, in which models privilege linguistic priors over visual information; this may have implications for downstream alignment. Addressing these limitations requires architectural innovations that properly ground spatial reasoning in visual perception rather than learned textual associations. These failures highlight potential risks for applications that depend critically on reliable perspective-taking, such as autonomous driving and robotic assistance.

#### 6. Conclusions

Our psychology-grounded diagnostic reveals a critical gap between VLMs’ strong scene understanding capabilities and their markedly weaker performance on spatial reasoning and visual perspective taking. While models achieved near-perfect accuracy identifying objects and humanoid minifigures, their performance degraded substantially when determining spatial relationships and perspective-dependent locations. Future work should focus on architectures incorporating explicit geometric representations, specialized training protocols emphasizing mental rotation, and hybrid approaches combining symbolic spatial reasoning with learned representations to bridge the gap between recognition and true spatial understanding.

#### 7. Limitations

Our diagnostic focuses on simple spatial configurations with single humanoid minifigures and objects in static images, which may not fully capture the complexity of realworld perspective taking involving multiple agents, dynamic scenarios, and complex occlusions. The limited spatial coverage (four cardinal positions, two orientations) and reliance on LEGO minifigures, while enabling experimental control, restricts generalization to more diverse and naturalistic settings. Additionally, we evaluated nine models in their out-of-the-box configurations without extensive prompt engineering, and our interventions were primarily conducted on GPT-4-Turbo, leaving room for exploring alternative reasoning strategies and newer model architectures.

#### References

- [1] Armen Aghajanyan, Lili Yu, Alexis Conneau, et al. Scaling laws for generative mixed-modal language models. In International Conference on Machine Learning, pages 265–279. PMLR, 2023. 1
- [2] Rohan Anil, Sebastian Borgeaud, Yonghui Wu, et al. Gemini: A family of highly capable multimodal models. CoRR, abs/2312.11805, 2023. 2
- [3] Anthropic. Claude 3 family. https : / / www . anthropic.com/news/claude-3-family, 2023. Accessed: 2023-08-28. 4
- [4] Anthropic. Introducing claude 3. https://www. anthropic.com/news/claude-3-family, 2024. Accessed: 2025-11-09. 2
- [5] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, et al. Scanqa: 3d question answering for spatial scene understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022. 2
- [6] Jinze Bai et al. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond,

2023. 1

- [7] Simone Balloccu, Patr´ıcia Schmidtov´a, Mateusz Lango, et al. Leak, cheat, repeat: Data contamination and evaluation malpractices in closed-source LLMs. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 67–93, St. Julian’s, Malta, 2024. Association for Computational Linguistics. 2
- [8] S´ebastien Bubeck, Varun Chandrasekaran, Ronen Eldan, et al. Sparks of artificial general intelligence: Early experiments with GPT-4. CoRR, abs/2303.12712, 2023. 1
- [9] Boyuan Chen, Zhuo Xu, Sean Kirmani, et al. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14455– 14465, 2024. 2
- [10] An-Chieh Cheng, Hongxu Yin, Yang Fu, et al. Spatialrgpt: Grounded spatial reasoning in vision language model. CoRR, abs/2406.01584, 2024. 2

- [11] Angela Dai, Angel X. Chang, Manolis Savva, et al. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proc. Computer Vision and Pattern Recognition (CVPR), IEEE, 2017. 8
- [12] Pengxiang Ding, Han Zhao, Wenjie Zhang, et al. QUARVLA: vision-language-action model for quadruped robots. In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part V, pages 352–367. Springer, 2024. 1
- [13] Thorsten M. Erle and Topolinski Sascha. The grounded nature of psychological perspective-taking. Journal of Personality and Social Psychology, 112(5):683–695, 2017. 1
- [14] Daniel Ferber, Gregor W”olflein, Isabel C Wiest, et al. Incontext learning enables multimodal large language models to classify cancer pathology images. Nature Communications, 15:10104, 2024. 1
- [15] John H. Flavell, Frances L. Everett, Karen Croft, et al. Young children’s knowledge about visual perception. Developmental Psychology, 17(1):99–103, 1981. 2
- [16] Kanishk Gandhi, Jan-Philipp Fr¨anken, Tobias Gerstenberg, et al. Understanding social reasoning in language models with language models, 2023. 1
- [17] Iryna Hartsock and Ghulam Rasool. Vision-language models for medical report generation and visual question answering: A review. CoRR, abs/2403.02469, 2024. 1
- [18] Cecilia M Heyes and Chris D Frith. The cultural evolution of mind reading. Science, 344(6190):1243091, 2014. 1
- [19] M. Janczyk. Level 2 perspective taking entails two processes: Evidence from PRP experiments. Journal of Experimental Psychology: Learning, Memory, and Cognition, 39

(6):1878–1887, 2013. 2

- [20] Klaus Kessler and Hannah Rutherford. The two forms of visuo-spatial perspective taking are differently embodied and subserve different spatial prepositions. Frontiers in Psychology, 1, 2010. 2
- [21] Klaus Kessler and Konstantina E. Rutherford. The two forms of visual perspective taking are differently embodied and subserve different spatial prepositions. Frontiers in Psychology, 5(2):102, 2014. 1
- [22] Michal Kosinski. Evaluating large language models in theory of mind tasks. Proceedings of the National Academy of Sciences, 121(45):e2405460121, 2024. 1
- [23] Drew Linsley, Peisen Zhou, Alekh Ashok, et al. The 3dpc: a benchmark for visual perspective taking in humans and machines. In International Conference on Representation Learning, pages 30472–30494, 2025. 2, 3
- [24] J. M. Loomis. Spatial updating in humans. Trends in cognitive sciences, 7(3):103–111, 2003. 1
- [25] I. Lukosiunaite, A.M. Kovacs, and N. Sebanz. The influence of another’s actions and presence on perspective taking. Scientific Reports, 14:4971, 2024. 3
- [26] Ieva Lukoˇs¯unait˙e, Agnes´ M. Kov´acs, and Natalie Sebanz. The influence of another’s actions and presence on perspective taking. Scientific Reports, 14(1):4971, 2024. 1
- [27] Xiaojian Ma, Silong Yong, Zilong Zheng, et al. Sqa3d: Situated question answering in 3d scenes. In International Conference on Learning Representations, 2023. 2

- [28] Meta AI. Llama 3.2: Revolutionizing edge AI and vision with open, customizable models, 2024. 4
- [29] Pascale Michelon and Jeffrey Zacks. Two kinds of visual perspective taking. Perception & psychophysics, 68:327–37,

2006. 2

- [30] Kanishka Misra, Julia Rayz, and Allyson Ettinger. COMPS: Conceptual minimal pair sentences for testing robust property knowledge and its inheritance in pre-trained language models. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 2928–2949, Dubrovnik, Croatia, 2023. Association for Computational Linguistics. 2
- [31] Henrike Moll and Michael Tomasello. Level 1 perspectivetaking at 24 months of age. British Journal of Development Psychology, 24:603–613, 2006. 2
- [32] Cathal O’Grady, Thomas Scott-Phillips, Susannah Lavelle, et al. Perspective-taking is spontaneous but not automatic. Quarterly Journal of Experimental Psychology (Hove), 73: 1605–1628, 2020. 3
- [33] OpenAI. Gpt-4v system card. https://openai. com/index/gpt-4v-system-card, 2023. Accessed: 2023-08-28. 4
- [34] OpenAI, Josh Achiam, Steven Adler, et al. Gpt-4 technical report, 2024. 2
- [35] Camilla Orefice, Ramona Cardillo, Isabella Lonciari, et al. “picture this from there”: spatial perspective-taking in developmental visuospatial disorder and developmental coordination disorder. Front. Psychol., 15:1349851, 2024. 1
- [36] Cathleen O’Grady, Thom Scott-Phillips, Suilin Lavelle, et al. Perspective-taking is spontaneous but not automatic. Quarterly Journal of Experimental Psychology, 73(10):1605– 1628, 2020. PMID: 32718242. 2
- [37] Amy Pearson, Danielle Ropar, and Antonia F de C. Hamilton. A review of visual perspective taking in autism spectrum disorder. Frontiers in human neuroscience, 7:652, 2013. 1
- [38] Pouya Pezeshkpour and Estevam Hruschka. Large language models sensitivity to the order of options in multiple-choice questions. In Proceedings of the 2023 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACLHLT), 2023. 3
- [39] Jean Piaget and B¨arbel Inhelder. The Child’s Conception of Space. Routledge & Kegan Paul, 1956. 1
- [40] Paula Rubio-Fernandez, Madeleine Long, Vishakha Shukla, et al. Visual perspective taking is not automatic in a simplified dot task: Evidence from newly sighted children, primary school children and adults. Neuropsychologia, 172:108256,

2022. 2

- [41] Dana Samson, Ian Apperly, Jason Braithwaite, et al. Seeing it their way: Evidence for rapid and involuntary computation of what other people see. Journal of Experimental Psychology: Human Perception and Performance, 36:1255–1266,

2010. 2, 5

- [42] Adarsh Jagan Sathyamoorthy, Kasun Weerakoon, Mohamed Elnoor, et al. Convoi: Context-aware navigation using vision language models in outdoor and indoor environments. CoRR, abs/2403.15637, 2024. 1

- [43] Roger N Shepard and Jacqueline Metzler. Mental rotation of three-dimensional objects. Science, 171(3972):701–703,

1971. 5

- [44] Elizabeth S. Spelke. Principles of object perception. Cognitive Science, 14(1):29–56, 1990. 5
- [45] Elizabeth S Spelke and Katherine D Kinzler. Core knowledge. Developmental Science, 10(1):89–96, 2007. 5
- [46] J. W. A. Strachan, D. Albergo, G. Borghini, et al. Testing theory of mind in large language models and humans. Nature Human Behaviour, 8:1285–1295, 2024. 1
- [47] Andrew Surtees, Ian Apperly, and Dana Samson. Similarities and differences in visual and spatial perspective-taking processes. Cognition, 129:426–438, 2013. 2
- [48] Andrew DJ Surtees, Stephen A Butterfill, and Ian A Apperly. Direct and indirect measures of level-2 perspective-taking in children and adults. British Journal of Developmental Psychology, 30(1):75–86, 2012. 5
- [49] Samy Tafasca, Anshul Gupta, Victor Bros, et al. Toward semantic gaze target detection. In The Thirty-eighth Annual Conference on Neural Information Processing Systems,

2024. 3

- [50] Ryutaro Tanno, David GT Barrett, August Sellergren, et al. Collaboration between clinicians and vision–language models in radiology report generation. Nature Medicine, 2024. 1
- [51] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, et al. Gemini robotics: Bringing ai into the physical world, 2025. 4
- [52] Barbara Tversky and Bridgette Hard. Embodied and disembodied cognition: Spatial perspective-taking. Cognition, 110:124–9, 2009. 2
- [53] David H Uttal, Nathaniel G Meadow, Elizabeth Tipton, et al. The malleability of spatial skills: a meta-analysis of training studies. Psychological Bulletin, 139(2):352–402, 2013. 5
- [54] Bingyang Wang, Yijiang Li, Qingyang Zhou, et al. Do vision language models infer human intention without visual perspective-taking? towards a scalable ”one-image-probeall” dataset. In ICML 2025 Workshop on Assessing World Models, 2025. 2
- [55] An Yang, Anfeng Li, Baosong Yang, et al. Qwen3 technical report, 2025. 4

## Beyond Recognition: Evaluating Visual Perspective Taking in Vision Language Models

### Supplementary Material

#### A. Data

- A.1. Gold-Standard Answer In Table 3 we list the gold-standard answers distribution.

Table 3. Gold–standard answer distribution for each question type (Q1–Q7).

- Q Gold-Standard Answers (% of items)

- Q1 1 (100.0)
- Q2 1 (100.0)
- Q3 yes (100.0)
- Q4 east (15.3), east, north (10.4), east, south (8.3), north (14.6), north, west (10.4), south (18.1), south, west (6.9), west (16.0)
- Q5 east (15.3), east, south (9.7), north (25.0), south (25.0), south, west (7.6), west (17.4)
- Q6 no (50.0), yes (50.0)
- Q7 back (31.9), back, left (4.9), back, right (13.2), front (39.6), front, left (7.6), front, right (2.8)

A.2. Prediction Correctnesses

Models sometimes generated compound answers – for instance, northeast in Q4 and Q5, or back and slightly to the left in Q7 (see Figure 9 and Table 4). Because these responses contained multiple components, our evaluation needed to acknowledge partial as well as fully correct answers.

To tackle this, we employed a precision-based metric that rewarded models for each correctly identified component while tolerating omissions. Assume that we evaluate

- R responses for a given diagnostic questions and a model, like in Table 2. The score P reported in the table is called prediction correctness and is defined as the mean precision across all R responses:

Table 4. Frequency of answer types for Q4, Q5, and Q7 across models. Comp. (compound) denotes compound answers (e.g., northwest to Q5); Sing. (Single) denotes single-word answers (e.g., back to Q7); Disc. (Disclaimer) marks instances in which the model failed to provide a relevant answer (e.g., claiming no object is present when one is).

Model Comp. Sing. Disc.

- Question 4

GPT-4-o 21 123 0 GPT-4-Turbo 0 144 0 Claude 3.5 Sonnet 21 117 6 Claude 3 Sonnet 0 127 17 Llama-3.2-11B-Vision 33 111 0 Qwen3-2B-Instruct 0 144 0 Qwen3-4B-Instruct 0 144 0 Qwen3-8B-Instruct 0 144 0 Gemini Robotics ER 1.5 0 144 0

- Question 5

GPT-4-o 0 144 0 GPT-4-Turbo 0 144 0 Claude 3.5 Sonnet 0 144 0

- Claude 3 Sonnet 0 144 0

- Llama-3.2-11B-Vision 1 143 0 Qwen3-2B-Instruct 0 144 0 Qwen3-4B-Instruct 0 144 0 Qwen3-8B-Instruct 0 144 0 Gemini Robotics ER 1.5 0 144 0 Question 7

GPT-4-o 10 134 0 GPT-4-Turbo 16 127 1 Claude 3.5 Sonnet 68 76 0 Claude 3 Sonnet 1 121 22

- Llama-3.2-11B-Vision 2 139 3 Qwen3-2B-Instruct 0 144 0 Qwen3-4B-Instruct 0 144 0 Qwen3-8B-Instruct 0 144 0 Gemini Robotics ER 1.5 0 144 0

R

Pi, where Pi = P(Mi,Gi) = |Mi ∩ Gi| |Mi|

1 R

- P =

,

i=1

where Mi (resp. Gi) is the set of components in the model’s prediction (resp. gold-standard answer) for the i response.

The value of P ranges from 0 to 1. We note that for the questions with single answers, like e.g., Q6, this metric is equivalent to standard accuracy. Moreover, we experimented with several other metrics that take into account partial correctness (e.g., the Jaccard index), and they all yielded similar results.

##### A.3. Random Baseline

To establish a chance performance benchmark, we defined a random baseline classifier selecting one answer uniformly at random from the Nq distinct choices available for question type q. For a given instance of question type q, the probability of this random choice being evaluated as correct is Prob(successq) = Nkq

, where kq is the number (kq ≥ 1) of acceptable answers specified in the gold-standard answers for that instance out of Nq total options. This formulation accurately quantifies the expected success rate of uninformed random guessing under our evaluation proto-

q

[Figure 54]

[Figure 55]

GPT-4-Turbo Claude 3.5 Sonnet

From the perspective of the humanoid minifigure, where is the cat located relative to it? Please use terms such as front, left, right, or back.

Assuming the top of the image is north, in which cardinal direction (i.e., north, west, east, or south) is the plant located relative to the humanoid minifigure?

The plant is located southeast of the LEGO minifigure. The minifigure is positioned in the upper right (northeast) part of the image, while the small plant or sprout is in the lower right (southeast) corner, diagonally down and to the left from the minifigure's position.

From the perspective of the humanoid minifigure, the cat is located to its front and slightly to the right.

- Figure 9. Example model responses to Q5 and Q7 questions. Left: GPT-4 Turbo answers front and slightly to the right (gold-standard answer: front). Right: Claude 3.5 Sonnet answers southeast (gold-standard answer: south).

col, accommodating multiple correct answers, based on the empirical distribution of kq values observed in our goldstandard answers dataset (detailed in A.1, Table 3). For instance, for Q4 (N4 = 4), where 64% of instances have k4 = 1 (Prob = 14) and 36% have k4 = 2 (Prob = 12), the weighted average yields a chance level for this question type of (0.64 × 14) + (0.36 × 21) = 0.16 + 0.18 = 0.34.

Category-level chance performance was computed by averaging the chance levels of the constituent question types. For example, the scene understanding category comprises Q1 (chance = 13), Q2 (chance = 31), and Q3 (chance = 12), resulting in an average category chance level of (13 + 13 + 12)/3 = 187 ≈ 0.389. Following this methodology across all categories yields the following random baseline classifier levels: scene understanding (0.389), spatial reasoning (0.317), and visual perspective taking (0.411).

##### A.4. Co-occurrence Matrix

Our co-occurrence matrices presented in the result section show how the model’s predictions line up with each goldstandard answer. To build it, we take all questions whose gold-standard answers include a particular label – for example, north in Q5, and within that subset simply count how often the model produced north, east, south, or west. Those four counts become the row for north. Because a single question can have several gold answers and the model may mention several answers at once (such as northeast in

- Q5 or back and slightly to the left in Q7), one question can be counted in more than one row or column, so the values in a row may exceed the total number of questions. When every question has exactly one gold label and the model also

outputs exactly one label, this co-occurrence table collapses to the ordinary single-label confusion matrix, with each row summing to the number of items.

