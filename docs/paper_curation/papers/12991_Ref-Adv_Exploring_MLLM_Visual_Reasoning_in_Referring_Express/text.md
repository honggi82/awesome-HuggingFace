# arXiv:2602.23898v1[cs.CV]27Feb2026

## REF-ADV: EXPLORING MLLM VISUAL REASONING IN REFERRING EXPRESSION TASKS

### Qihua Dong Kuo Yang Lin Ju Handong Zhao Yitian Zhang Yizhou Wang Huimin Zeng Jianglin Lu Yun Fu Northeastern University https://ref-adv.github.io/

ABSTRACT

Referring Expression Comprehension (REC) links language to region level visual perception. Standard benchmarks (RefCOCO, RefCOCO+, RefCOCOg) have progressed rapidly with multimodal LLMs but remain weak tests of visual reasoning and grounding: (i) many expressions are very short, leaving little reasoning demand; (ii) images often contain few distractors, making the target easy to find; and (iii) redundant descriptors enable shortcut solutions that bypass genuine text understanding and visual reasoning. We introduce Ref-Adv, a modern REC benchmark that suppresses shortcuts by pairing linguistically nontrivial expressions with only the information necessary to uniquely identify the target. The dataset contains referring expressions on real images, curated with hard distractors and annotated with reasoning facets including negation. We conduct comprehensive ablations (word order perturbations and descriptor deletion sufficiency) to show that solving Ref-Adv requires reasoning beyond simple cues, and we evaluate a broad suite of contemporary multimodal LLMs on Ref-Adv. Despite strong results on RefCOCO, RefCOCO+, and RefCOCOg, models drop markedly on Ref-Adv, revealing reliance on shortcuts and gaps in visual reasoning and grounding. We provide an in depth failure analysis and aim for Ref-Adv to guide future work on visual reasoning and grounding in MLLMs.

1 INTRODUCTION

Referring expression comprehension (REC) is the task of grounding a natural language expression to a specific region in an image (Mao et al., 2016; Kazemzadeh et al., 2014; Yu et al., 2016). It has important applications in real world systems and downstream tasks, and it has become a key benchmark for evaluating multimodal large language models (MLLMs) because it probes fine grained correspondence between language and vision. Recent MLLMs (Google, 2025a; Bai et al., 2025; Zhu et al., 2025), both closed source and open source, have made substantial progress, achieving over 90% accuracy on classic REC benchmarks, i.e., RefCOCO(+/g) (Kazemzadeh et al., 2014; Yu et al., 2016; Mao et al., 2016).

Despite this near saturated performance, we identify critical limitations of the classic REC benchmarks that motivate a modern benchmark capable of more challenging and comprehensive evaluation of MLLMs. We view modern REC for MLLMs as a multistep reasoning task with two coupled components: (1) textual reasoning—understanding the referring expression, identifying the target, and identifying its descriptors; and (2) visual reasoning—searching for candidates and establishing correspondence between descriptors and image regions. The order of these steps can vary across models, but a meaningful benchmark should require both textual and visual reasoning. From this perspective, we highlight the following limitations of RefCOCO(+/g).

First, most of the referring expressions are extremely short, as shown in Figure 1. For RefCOCO and RefCOCO+, the average expression length is around 3 words. Such short expressions lead to two issues: (1) minimal linguistic effort is required, and (2) they typically entail less visual reasoning because fewer descriptors must be verified in the image. Second, there are few distractors in the images in RefCOCO(+/g), as shown in Figure 2 (b), with most cases of only 1 distractors. Here we define a distractor as an object of the same category as the target but a different instance. When few

[Figure 1]

[Figure 2]

|[Figure 3]|
|---|

[Figure 4]

[Figure 5]

Find “a DELL mouse connected to the laptop”

Find “a man in the background jumping to return the tennis ball”

Find “pizza”

User

|(answer without reasoning)|
|---|

|Only one ‘mouse’, I don’t need to know the rest.|
|---|

|I don’t know what is ‘in the background’, but I see one jumping.|
|---|

[Figure 6]

MLLM (a) Expression too short? (b) Few distractors in image? (c) Expression too descriptive?

- Figure 1: Common limitations of classic referring expression benchmarks that reduce the reasoning challenge. These include very short expressions, few visual distractors, and overspecified descriptors that enable shortcut matching without requiring genuine reasoning. The cyan box highlights the ground truth region.

[Figure 7]

distractors exist, the task requires far less textual and visual reasoning: models need only infer the target category and select from a small set of candidates. Figure 2 (b) reveals a negative correlation between the number of distractors and model performance.

a man wearing a brown sweater and holding a game control

It is worth noting that for reasoning assessment, task difficulty does not monotonically increase with referring expression length due to ”grounding shortcuts”. These shortcuts occur when a long, descriptive expression is paired with few distractors, rendering many descriptors redundant. Consequently, can pa This hig ons that

|a model can locali<br><br>paradoxically lead to higher accu hlights the need for modern RE are concise and carefully bala|ze the target by matching only racy for longer expressions, a C benchmarks to mitigate such nced against the available distr|a subset of descriptors, which s illustrated in Figure 2 (a). shortcuts by designing expressi actors.|
|---|---|---|
|Meanwhile, prior work has ackno (2024) point out the length limitat<br><br>distractors. However, the pro data with average length ≥ numerous shortcuts since the num|wledged aspects of these limitat ions of RefCOCO(+/g), and Ch posed datasets also raise new 90 words, which may be unnat<br><br>bers of descriptors and distrac|ions: Wei et al. (2024); Chen et en et al. (2020) highlights the concerns. The former introduces ural and, more importantly,<br><br>tors are heavily imbalanced.|
|(a) proposesExpressionsettingstoo short?including|referring from a set of images,|which shifts away from the|

M al. (2 lack of ces REC enable nu The latter classic REC setting, and the referring expressions are sampled from GQA (Hudson & Manning, 2019) scene graphs with fixed templates, reducing naturalness.

We therefore aim to build a REC benchmark that preserves the classic REC setting and natural expressions while substantially increasing the reasoning challenge aligned with the capabilities of modern LLMs. To this end, we introduce Ref-Adv, a modern REC benchmark that avoids short reasoning paths and imposes both reasoning and grounding challenges on contemporary MLLMs. To validate the quality of the benchmark, we conduct comprehensive ablation studies in section 2 that examine what makes a rigorous modern REC benchmark and compare its reasoning and grounding difficulty with RefCOCO(+/g). In section 3, we evaluate 13 contemporary MLLMs on Ref-Adv, spanning both closed source and open source models. We report performance changes and provide detailed analyses. We publicly release Ref-Adv-s, a curated subset of 1,142 cases, to enable reproducible benchmarking.

- 2 THE REF-ADV DATASET

- 2.1 DATA SOURCE

We sample from the validation and test splits of COCO (Lin et al., 2014) and OpenImages v7 (Kuznetsova et al., 2020). We filter the images and only use those with panoptic instance annotations, since this is important for our later pipeline. For the bounding box annotations, we convert

[Figure 8]

[Figure 9]

(a) (b)

- Figure 2: Accuracy@0.5 (IoU ≥ 0.5) of Qwen on the RefCOCO/+/g validation sets. Marker size is proportional to the number of samples in each bin. (a) is the Acc@0.5 on number of words in expressions, (b) is on distractor count. We can see most cases have short expressions and few distractors.

- Table 1: Basic statistics of the validation+test sets of RefCOCO, RefCOCO+, RefCOCOg, and Ref-Adv (Ours). The instance size is represented by its square root. Avg. length: average length of annotations. Vocab.: vocabulary size. Avg. distractors: average number of same category distractors per image. Negation ratio: percentage of expressions using explicit negation.

Benchmark Images Instances Avg. Length Avg. Distractors Negation Ratio Instance Size Vocab. RefCOCO 2014 3,000 7,596 3.6 3.99 0.99% 105–607 3,525 RefCOCO+ 2016 3,000 7,578 3.6 3.96 3.36% 105–607 4,387 RefCOCOg 2016 3,900 7,596 8.4 1.64 1.41% 83–610 5,050 Ref-Adv (Ours) 2,833 5,000 11.5 4.01 21.25% 30-607 5,308

all to using the absolute coordinates in the format of [x1, y1, x2, y2]. The input for our data pipeline is the image, the bounding box annotations and category name of each instance, and we will output the referring expression paired with the target instance.

- 2.2 COLLECTION GUIDELINES

As shown in Figure 1, we aim to collect referring data that requires visual reasoning, avoids shortcut solutions, and challenges models. Based on these observations, we propose the following guidelines to mitigate these limitations and yield cases requiring advanced reasoning.

Accuracy@0.5 (IoU ≥ 0.5) of Qwen (Qwen‑VL‑Chat) on the RefCOCO, RefCOCO+, and RefCOCOg validation sets as a function of reference expression length (number of words). Each point is the mean accuracy for a given length, with marker size proportional to the number of samples; accuracy declines with longer expressions on RefCOCO/RefCOCO+, while RefCOCOg remains comparatively stable across lengths.

Accuracy@0.5 (IoU ≥ 0.5) of Qwen‑VL on the RefCOCO, RefCOCO+, and RefCOCOg validation sets as a function of distractor count. Marker size is proportional to the number of samples in that bin. Accuracy declines with increasing distractors.

Distractor Pressure Distractors are instances of the same category as the target but different instances. To avoid easy grounding based solely on the target category, we select images that have at least 3 candidate instances of the same category as the target, based on the instance annotations of each dataset.

Language Complexity RefCOCO(+/g) has an average expression length of around 3 words, which limits language complexity and requires much less visual reasoning. Meanwhile, fixed templates that extract referring information from scene graphs limit diversity in the referring expressions. Therefore, we employ LLMs (e.g., GPT-4o) with carefully designed pipelines to generate more natural and diverse referring expressions while maintaining linguistic complexity.

Hard Distractors Simply increasing the number of distractors and the length of the referring expression does not necessarily make the task more challenging because of the ”grounding shortcut” illustrated in Figure 1 (c). To reduce such shortcuts (i.e., reliance on redundant descriptors), we ensure the presence of ”hard distractors” in the images, defined as distractors that partially match, but do not exactly satisfy, the referring expression. Identifying such pairs and composing expressions around them is central to our data collection process.

Manual Check It is laborious and time-consuming to manually select images with hard distractors and generate the referring expressions, so we use LLMs to assist generation. However, LLMs can

[Figure 10]

Find the most similar pair and provide descriptors that differ the pair and the rest, and the ones inside the pair.

Generate a refer. expression for

- ‘1’. Use discrimators to differ with
- ‘2’, and then with ‘3’. Consider using negation.

[Figure 11]

- 1. The expression is accurate.
- 2. There exists hard distractors.

Distractors >= 3 ?

[Figure 12]

[Figure 13]

{similar group: ‘1’ and ‘3’, others: ‘2’, group discriminators: [ “hair …”, “necklace …”,],

“The person with short hair, and wearing sunglass”

SoM Tagging

GPT-4o GPT-4o

[Figure 14]

“Not wearing necklace, the person in the light-colored shirt”

individual discriminators: [“sunglass …”, “shirt …”,]

2 3

Ref-Adv Data

1

…

}

(a) Prepare Image (b) Similarity Judgement (c) Expression Generation (d) Human Verification

- Figure 3: LLM-authored data curation pipeline for Ref-Adv. (a) Prepare Image: filter images, ensure ≥ 3 distractors, and add number tags to candidate instances. (b) Similarity Judgement: use GPT-4o to identify the most similar pair and elicit group-level and instance-level discriminators. (c) Expression Generation: compose minimally sufficient referring expressions using discriminators and optional negation. (d) Human Verification: verify expression accuracy and confirm the existence of hard distractors before inclusion.

make mistakes or hallucinate. To ensure accuracy, we perform a human verification pass to confirm the existence of hard distractors and the correctness and unambiguity of the referring expression.

- 2.3 REFERRING EXPRESSION GENERATION PROCESS

As shown in Figure 3, the whole generation process is conducted in four stages. The prompts we use are provided in section 5.

Input Preparation We first filter the images to only keep those with at least 3 candidate instances. We then put a number tag on each instance, similar to Set-of-Marks (Yang et al., 2023), but since we already have instance annotations, we only need to add the number tag to the candidate instances.

- 2.3.1 LLM-AUTHORED PIPELINE

Before detailing the pipeline, we note an important design choice. We first attempted single step prompting of GPT-4o to directly produce complete referring expressions from the image and candidate instances. In practice, GPT-4o frequently produced overspecified descriptions with many redundant descriptors, which enabled shortcut grounding and weakened the need to understand the whole expression. To avoid this behavior, we adopt a two stage procedure: we first elicit discriminative attributes (between group A and group B and within group A), and then compose the final expression from a minimal yet sufficient subset of those attributes.

Similarity Judgement If there is a hard distractor and a target instance, they will be similar in some ways. To encourage the LLMs to identify any such similar pair in the image, we define two groups, group A and group B, where group A contains the hard distractor and the target instance, and group B contains the other distractors. We then prompt the LLMs to identify the two groups and to describe (1) attributes that distinguish the groups and (2) attributes that distinguish the two instances within group A. We ask for multiple alternative descriptions for each distinction. This could help us generate multiple diverse referring expressions for one image and allow us to select the high quality ones.

Referring Expression Generation After the similarity judgement, we obtain a list of paired descriptors that distinguish (1) group A from group B and (2) the two instances within group A. To ensure naturalness and diversity in phrasing, we prompt LLMs to compose referring expressions from combinations of these descriptors. Specifically, we use two alternative strategies: (1) employ the target’s descriptors and (2) use the negation of the hard distractor’s descriptors. This promotes more diverse and natural expressions. We also explicitly instruct the LLMs to not include number tag related descriptions. Although the elicited descriptors alone are sufficient for generation, we find that including the image input at this stage yields more diverse and accurate expressions, so

[Figure 15]

[Figure 16]

[Figure 17]

(a) (b) (c)

- Figure 4: Dataset statistics across REC benchmarks. (a) Expression length comparison. (b) Distribution of distractor counts. (c) Instance size on a log area scale.

- Table 2: Accuracy@0.5 after replacing the original referring expressions with the fixed “the one” prompt. ∆ is Fixed@0.5 minus Ref-Adv Fixed@0.5 (shown in blue). With fixed prompt, models achieve higher accuracy on RefCOCO, RefCOCO+, and RefCOCOg than Ref-Adv.

RefCOCO RefCOCO+ RefCOCOg Ref-Adv Model Fixed@0.5 ∆ vs Ref-Adv Fixed@0.5 ∆ vs Ref-Adv Fixed@0.5 ∆ vs Ref-Adv Fixed@0.5

Qwen2.5-VL-72B 35.1% +13.7% 39.4% +18.0% 38.3% +16.9% 21.4% InternVL-3-14B 35.9% +13.1% 38.0% +15.2% 38.2% +15.4% 22.8%

we include the image. After this stage, we obtain multiple candidate referring expressions for each target instance.

- 2.3.2 HUMAN-AUTHORED PIPELINE

We also collect a subset of human-authored referring expressions. For each filtered image, annotators first confirm whether there is a hard distractor pair and, if so, write a referring expression for it. Annotators are instructed to produce diverse and natural phrasing.

- 2.3.3 VERIFICATION PROTOCOL

We verify each image–text pair. Three annotators answer two questions: (1) whether the expression is correct and unambiguous and (2) whether hard distractors are present in the image. Annotators first attempt grounding on the original image (without number tags) using the LLM generated expression. We then show the ground truth box overlaid on the image for reference, allowing reflection if their initial grounding was incorrect. Afterward, annotators record their final decisions on correctness/unambiguity and on the presence of hard distractors. Pairs are presented in a random order per annotator, and a pair is kept only if all three annotators agree. The keep rate is 18.7% for LLM-authored expressions.

- 2.4 QUALITY ANALYSIS

Despite verification to ensure correctness, there remain potential issues for an REC benchmark that could affect fairness and the evaluation of reasoning skills. To further assess the quality of our data, we conduct the following analyses.

Statistics As shown in Figure 4 and table 1, Ref-Adv exhibits clear advantages in expression length, vocabulary size, distractor counts, and the negation ratio.

Model Bias Test Inspired by Cirik et al. (2018); Chen et al. (2020), we conduct a bias test of modern MLLMs (Qwen2.5-VL-72B and InternVL-3) on RefCOCO(+/g) and Ref-Adv. Here, bias refers to statistical regularities that may arise if training data comes from the same source as an evaluation benchmark, which can benefit performance. We design the test as follows: we replace the referring expression with a fixed prompt (“the one”), keep the same image, and prompt the model to output a bounding box. This test reveals whether model bias helps localize the target. The results are shown in table 2. They suggest that Ref-Adv is less affected by this bias than other benchmarks.

#### Table 3: Bag-of-words ablation on RefCOCO, RefCOCO+, RefCOCOg, and Ref-Adv. Acc@0.5 with original expressions vs bag-of-words (word order removed). ∆ denotes (BoW − Original).

RefCOCO RefCOCO+ RefCOCOg Ref-Adv

Model Orig@0.5 BoW@0.5 ∆ Orig@0.5 BoW@0.5 ∆ Orig@0.5 BoW@0.5 ∆ Orig@0.5 BoW@0.5 ∆ Qwen2.5-VL-72B 92.7% 82.8% -9.9% 88.9% 78.2% -10.7% 89.9% 75.3% -14.6% 58.3% 41.5% -16.8% InternVL-3-14B 92.0% 84.7% -7.3% 87.6% 81.0% -6.6% 88.5% 74.9% -13.6% 52.3% 38.6% -13.7%

#### Table 4: One descriptor deletion ablation on RefCOCO, RefCOCO+, RefCOCOg, and Ref-Adv. Acc@0.5 with original expressions vs one descriptor deletion (removing a single descriptor in expression). ∆ denotes (1-Desc − Original).

RefCOCO RefCOCO+ RefCOCOg Ref-Adv

Model Orig@0.5 1D@0.5 ∆ Orig@0.5 1D@0.5 ∆ Orig@0.5 1D@0.5 ∆ Orig@0.5 1D@0.5 ∆ Qwen2.5-VL-72B 92.7% 88.0% -4.7% 88.9% 83.6% -5.3% 89.9% 85.3% -4.6% 58.3% 51.9% -6.4% InternVL-3-14B 92.0% 87.1% -4.9% 87.6% 82.4% -5.2% 88.5% 83.8% -4.7% 52.3% 45.2% -7.1%

Textual Reasoning Necessity Test Prior work (Akula et al., 2020) shows that shuffling word order in RefCOCOg often leaves performance largely intact, indicating weak necessity for textual reasoning in prior REC benchmarks. This lack of degradation could stem from two factors: (1) expressions that only mention the target (or its parts) without referencing distractors and (2) images with no or very few distractors. Both factors reduce the reasoning demand in REC. To validate that RefAdv requires reasoning, we extend the test to RefCOCO(+/g) and Ref-Adv for comparison. Rather than shuffling while preserving meaning, we propose a simpler test: we convert the expression to a bag of words and randomize its order in the prompt (e.g., “a red ball with yellow stripes” becomes “with yellow red ball stripes a”). We evaluate Qwen2.5-VL-72B and InternVL-3 under this setting. Results are shown in table 3, indicating that Ref-Adv indeed requires textual understanding and reasoning to follow the referring expression exactly.

Avoidance of ”Grounding Shortcut” As illustrated in Figure 1, RefCOCO(+/g) admits a ”grounding shortcut,” where a model can localize the target by checking a small subset of descriptors, without reasoning over the entire expression. To validate that Ref-Adv avoids this shortcut, we conduct a descriptor-deletion sufficiency test. For a given referring expression, we first use Qwen2.572B (Yang et al., 2024) to extract all descriptors, randomly delete one, and ask Qwen2.5-72B to rewrite the expression with that descriptor removed. We then evaluate MLLMs on the modified image–text pair. If deleting a descriptor does not affect performance, the descriptor is unnecessary, suggesting a shortcut that succeeds without understanding the full expression. Such shortcuts are exacerbated in datasets with imbalanced numbers of descriptors and distractors. Results are shown in table 4, indicating that Ref-Adv has far fewer grounding shortcuts than others.

- 3 EXPERIMENT

3.1 EVALUATION SETUP

Evaluated Models We evaluate contemporary state of the art MLLMs, both closed source and open source, on Ref-Adv. The suite includes Qwen2.5-VL series (Bai et al., 2025), InternVL3 series (Zhu et al., 2025), Gemini 2.5-Flash (Google, 2025a), Gemini 2.5-Pro (Google, 2025b), CogVLM-Grounding (Hong et al., 2024), GLM-4.5V (Team et al., 2025b), GPT-4o (OpenAI, 2024), and Claude-3.5 Sonnet (Anthropic, 2024).

Evaluation Methods Set-of-Marks (SoM) overlays numbered marks on candidate objects in the image and leverages a specialized segmenter to provide fine-grained localization, avoiding the need for the MLLM to perform grounding itself. Because GPT-4o and Claude-3.5 have limited grounding ability, we evaluate them using SoM (Yang et al., 2023) with Semantic-SAM (Li et al., 2023). We use Semantic-SAM due to its strong performance on COCO images, one of the sources of Ref-Adv.

For each model (except CogVLM-Grounding which does not support CoT), we evaluate both with and without Chain-of-Thought (CoT). While CoT is uncommon in classic REC benchmark evaluation, Ref-Adv requires more reasoning, so we include CoT in our setup. Table 6 and Table 7 report results on Ref-Adv with and without CoT.

##### LLM-Authored Human-Authored

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

Image Expression The cup that is empty

The young elephant near the water not covered in more dust.

The car is brown or black with a lighter tint on its windows.

The person sitting farther to the one holding a goose feather

and positioned at the center of the pizzas.

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

Image Expression The frisbee holder who

The occupied chair with the person’s arm not resting on the lap.

The sailboat, identified by being anchored, is the one with a flag.

The frisbee held by the person on the right hand side of the person in red

is not facing forward.

- Table 5: Examples from Ref-Adv. Columns 1 to 3 are LLM generated; column 4 is human authored.

[Figure 26]

|[Figure 27]<br><br>[Figure 28]<br><br>[Figure 29]<br><br>[Figure 30]<br><br>[Figure 31]<br><br>[Figure 32]<br><br>[Figure 33]<br><br>[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>The user is asking to determine the bounding box coordinates of the target object based on the given referring expression.<br><br>1. **Identify the glasses:** ...<br>2. **Determine "less full":** ...<br>3. **Locate the corner:** ...<br>4. **Proximity to the corner:** ... Therefore, the target is the less-full glass on the right, closer to the corner. Bounding box coordinates (visually estimated):<br><br><br>- Top-left (x1, y1): ...<br>- Bottom-right (x2, y2): ... … \boxed{[390, 380, 450, 460]}<br><br><br>The user is asking to identify a specific sofa in the image.<br><br>1. **Identify all sofas:** ...<br>2. **Analyze the first part of the referring expression: "The modern sofa"**: ...<br>3. **Analyze the second part of the referring expression: "not the one with a metal frame"**: ...<br>4. **Reconcile the two parts:** ... However, ... The key differentiator is the "metal frame". ... Therefore, "the modern sofa, not the one with a metal frame" refers to the dark, high-backed sofa on the left. … The final answer is \boxed{[30, 132, 230, 230]}<br><br><br>The user is asking to identify the bounding box coordinates of the plant in the clay pot that is positioned lower on the ledge.<br><br>1. **Clay Pot:** ...<br>2. **Lower on the Ledge:** ... Step-by-Step Analysis:<br><br><br>- Focus only on pots that appear clay (orange-brown).<br>- Interpret the ledge as the stone shelf behind the bench.<br>- Compare vertical positions of the clay pots on that ledge and pick the lowest.<br>- Verify the candidate is on the right side and contains a broad-leaf green plant. … \boxed{[490, 280, 630, 450]}<br><br><br>The user is asking to identify the chair referred to as “the chair near the middle, taller rather than shorter.”<br><br>1. **Near the middle:** ...<br>2. **Taller rather than shorter:** ...<br><br><br>Step-by-Step Analysis:<br><br>- Identify the central vertical axis as the middle area.<br>- List chairs intersecting that middle band.<br>- Compare visible heights; select the tallest among them.<br><br><br>Observations: The target chair is behind the short-haired person using a laptop and is taller than nearby chairs.<br><br>Bounding Box Coordinates: Approximate edges noted from visual inspection. … \boxed{[230, 180, 330, 350]}<br><br>The user is asking to determine the bounding box for “the one not leaning.”<br><br>1. **Identify the vehicle:** ...<br>2. **Identify the people near the vehicle:** ...<br>3. **Determine “the one not leaning”:** ...<br>4. **Locate this person:** On the left of the carriage, standing upright in dark clothing, facing the vehicle.<br><br><br>Bounding Box Coordinates: Approximate edges from visual inspection. … \boxed{[60, 200, 110, 290]}<br><br>With Cot Without CoT|
|---|

|[Figure 37]<br><br>[Figure 38]<br><br>[Figure 39]<br><br>[Figure 40]<br><br>[Figure 41]<br><br>[Figure 42]<br><br>[Figure 43]<br><br>[Figure 44]<br><br>The user is asking to identify a specific glass in the image based on its fullness and position.<br><br>1. **Identify all glasses:** ...<br>2. **Filter by "less full":** ...<br>3. **Filter by "closer to the corner rather than further from it":** ... Let's re-evaluate "less full". ... Comparing the two "less full" glasses that are relatively close to the corner: The red wine glass is closer to the bottom-left corner of the table area. Therefore, the target is the red wine glass that is to the left of the wine bottle and to the right of the person on the left. … Let's re-check the "closer to the corner" part. ...<br><br><br>... \boxed{[196, 343, 245, 416]}<br><br>[Figure 45]<br><br>[Figure 46]<br><br>The user is asking to identify a specific sofa in the image.<br><br>1. **Identify all sofas:** ...<br>2. **Analyze the first part of the referring expression: "The modern sofa"**: ...<br>3. **Analyze the second part of the referring expression: "not the one with a metal frame"**: ...<br>4. **Reconcile the two parts:** ... However, ... The key differentiator is the "metal frame". ... Therefore, "the modern sofa, not the one with a metal frame" refers to the dark, high-backed sofa on the left. … The final answer is \boxed{[30, 132, 230, 230]}<br><br><br>The user is asking to identify a specific plant in the image.<br><br>1. **"The plant in the clay pot"**: ...<br>2. **"that's positioned lower on the ledge"**: ... Therefore, the target is the plant in the brown clay pot on the right side of the image, sitting on a small wooden stand. … x1: ..., y1: ..., x2: ..., y2: ... Let's refine the coordinates to be more precise. ... Let's re-evaluate the bounding box to encompass the entire plant and its pot. ... So the coordinates are ... The final answer is \boxed{[486, 336, 602, 432]}<br><br><br>The user is asking to identify a specific chair in the image based on its location and relative height.<br><br>1. **"The chair near the middle"**: ...<br>2. **"taller rather than shorter"**: ... Let's examine the chairs in the image: ... Considering the descriptions, the chair that stands out is the one occupied by the person with long, grey, curly hair wearing a blue patterned shirt. ... Considering the entire chair, including the seat and legs... ...<br><br><br>... Final check:<br><br>- Is it near the middle? Yes, ...<br>- Is it taller rather than shorter? Yes, ... The bounding box for the chair occupied by the person with long grey hair. … The final answer is \boxed{[270, 214, 330, 281]}<br><br><br>The user is asking to identify the bounding box of a specific object in the image based on the referring expression "Near the vehicle is the one not leaning."<br><br>1. **Identify the "vehicle":** ...<br>2. **Identify objects "near the vehicle":** ...<br>3. **Identify who is "not leaning":** ...<br>4. **Resolve ambiguity:** ... Therefore, the target is the person sitting on the red vehicle. … x1: ..., y1: ..., x2: ..., y2: ... … \boxed{[250, 103, 314, 180]}<br><br><br>With Cot Without CoT|
|---|

[Figure 47]

###### Case #397: img402720_cat46_obj 6

Expression: "The less full glass closer to the corner rather than further from it."

The less full glass closer to the corner rather than further from it.

###### Case #470: img94871_cat20_obj4

Expression: "The white sheep facing away that is smaller rather than bigger."

[Figure 48]

[Figure 49]

The modern sofa, not the one with a metal frame.

[Figure 50]

[Figure 51]

The plant in the clay pot that's positioned lower on the ledge.

[Figure 52]

[Figure 53]

The chair near the middle, taller rather than shorter.

[Figure 54]

[Figure 55]

Near the vehicle is the one not leaning.

(a) Input (b) Gemini (c) Qwen

- Figure 5: Performance of representative multimodal LLMs on Ref-Adv. We include qualitative examples with and without CoT for Gemini 2.5-Flash and Qwen2.5-VL-72B. CoT answers are shown in a gray box. Hard distractors in Ref-Adv challenge current MLLMs.

Evaluation Prompts Models differ in prompt format and output conventions. For example, Qwen2.5-VL-72B uses absolute coordinates, while others use normalized coordinates; CogVLMGrounding requires the question to strictly follow the form ”Where is the ’referring expression’?” to output boxes. To ensure fairness, we adopt best-practice prompts for each model.

Near the vehicle is the one not leaning.

- Table 6: Results on Ref-Adv-s, a publicly released subset of 1,142 cases, across Qwen2.5-VL, Qwen3-VL, and Qwen3.5 model families. Columns report accuracy at IoU thresholds 0.5, 0.75, and 0.9. For distractor groups (2–3, 4–6, and ≥7), we report Acc0.5 and the delta relative to overall Acc0.5. CoT denotes Chain-of-Thought prompting (via thinking mode, think-first prompt, or native in Qwen3.5).

Model CoT? Acc0.5 Acc0.75 Acc0.9 Distractors (Acc0.5)

2–3 ∆ 4–6 ∆ ≥7 ∆

- Qwen2.5-VL

- 3B-Instruct ✗ 23.8 18.1 8.8 25.9 +2.1 21.9 -1.9 17.1 -6.8

- 3B-Instruct ✓ 25.3 19.1 9.5 28.2 +2.9 22.9 -2.4 15.5 -9.8

- 7B-Instruct ✗ 39.3 29.2 12.5 42.8 +3.5 36.8 -2.5 26.4 -13.0

- 7B-Instruct ✓ 39.0 28.8 11.6 43.0 +4.0 35.2 -3.7 26.4 -12.6 32B-Instruct ✗ 48.0 35.5 16.0 51.6 +3.6 43.8 -4.2 38.8 -9.2 32B-Instruct ✓ 50.6 37.7 16.0 55.2 +4.5 44.8 -5.9 40.3 -10.3 72B-Instruct ✗ 54.0 40.1 18.0 57.0 +3.0 52.7 -1.3 41.1 -12.9 72B-Instruct ✓ 52.4 39.0 18.3 56.9 +4.5 47.9 -4.4 38.8 -13.6 Qwen3-VL 2B-Instruct ✗ 23.5 19.2 11.0 26.1 +2.6 20.0 -3.5 17.8 -5.6 2B-Instruct ✓ 25.2 20.6 11.4 28.1 +2.9 21.3 -3.9 19.4 -5.8 2B-Thinking ✓ 44.4 36.8 21.8 48.6 +4.2 40.6 -3.8 31.0 -13.4

4B-Instruct ✗ 41.9 34.9 20.7 46.4 +4.5 36.2 -5.8 31.8 -10.2 4B-Instruct ✓ 42.5 34.9 20.6 46.6 +4.1 36.5 -6.0 34.9 -7.6 4B-Thinking ✓ 57.6 45.5 27.8 63.0 +5.4 52.7 -4.9 40.3 -17.3

- 8B-Instruct ✗ 47.2 37.0 19.1 51.3 +4.1 44.1 -3.1 32.6 -14.6

- 8B-Instruct ✓ 52.3 38.9 19.9 55.7 +3.5 50.2 -2.1 38.8 -13.5 8B-Thinking ✓ 59.5 48.2 27.3 63.5 +4.0 55.6 -3.9 47.3 -12.2 30B-A3B-Instruct ✗ 44.0 37.6 23.4 47.6 +3.5 40.3 -3.7 34.1 -9.9 30B-A3B-Instruct ✓ 52.1 43.1 27.4 54.7 +2.6 48.9 -3.2 45.7 -6.4 30B-A3B-Thinking ✓ 64.1 52.6 31.6 67.3 +3.2 62.2 -1.9 51.2 -12.9 32B-Instruct ✗ 53.4 44.7 27.1 56.3 +2.9 50.2 -3.3 45.7 -7.7 32B-Instruct ✓ 59.0 47.5 27.6 60.9 +1.9 57.5 -1.6 52.7 -6.3 32B-Thinking ✓ 65.6 52.8 31.6 67.9 +2.3 65.7 +0.1 52.7 -12.9 235B-A22B-Instruct ✗ 57.3 47.5 30.0 63.3 +6.1 51.7 -5.5 38.0 -19.3 235B-A22B-Instruct ✓ 59.3 48.9 29.9 63.5 +4.2 54.9 -4.4 47.3 -12.0 235B-A22B-Thinking ✓ 67.1 53.6 31.8 69.6 +2.6 65.7 -1.4 56.6 -10.5

- Qwen3.5 27B ✓ 67.3 54.9 32.7 69.9 +2.7 65.7 -1.5 56.6 -10.7 35B-A3B ✓ 66.7 54.4 34.9 68.9 +2.2 65.4 -1.3 58.1 -8.6 122B-A10B ✓ 67.2 55.0 35.1 69.9 +2.8 66.3 -0.8 54.3 -12.9 397B-A17B-FP8 ✓ 68.0 55.6 34.2 70.1 +2.1 67.9 -0.0 56.6 -11.4

- 3.2 EVALUATION METRICS

Accuracy serves as a widely adopted metric for evaluating existing REC models. A referring expression instance is deemed successfully grounded when the Intersection over Union (IoU) between the predicted bounding box and the ground truth annotation surpasses 0.5. This conventional evaluation metric is designated as Acc0.5. Here, we implement multiple evaluation protocols, i.e., Accuracy computed under different IoU thresholds such as Acc0.5, Acc0.75, Acc0.9, and mean Accuracy (mAcc) across different IoU criteria, to thoroughly evaluate the precision and robustness.

- 3.3 ANALYSIS

Effect of Distractor Count In Ref-Adv, each expression is paired with at least 2 same-category distractors, and images contain roughly 4 distractors on average. Compared with overall Acc0.5, most models show a modest change in the 4–6 group but a larger drop in the ≥7 group (e.g., Qwen2.5-VL72B+CoT: −0.2 and −2.7). This trend indicates that handling larger numbers of similar distractors remains a key challenge for current MLLMs.

Effect of CoT Table 6 and Table 7 show that CoT generally improves performance on Ref-Adv. We attribute the improvement on Ref-Adv to its heavier reasoning demand; for RefCOCO(+/g), where

- Table 7: Main results on Ref-Adv. Rows list models; columns report accuracy at IoU thresholds 0.5, 0.75, and 0.9, and mean accuracy (mAcc). For distractor groups (4–6 and ≥7), we report Acc0.5 and the delta relative to overall Acc0.5.

Model Setting Acc0.5 Acc0.75 Acc0.9 mAcc Distractors (Acc0.5)

CoT? SoM? 4–6 ∆ ≥7 ∆

GPT-4o 2024 ✗ ✓ 52.3 31.2 13.4 27.8 53.4 +1.1 51.7 -0.6 GPT-4o 2024 ✓ ✓ 63.7 38.4 19.7 34.1 62.9 -0.8 60.5 -3.2

Claude-3.5 Sonnet 2024 ✗ ✓ 40.8 22.1 3.8 22.4 39.0 -1.8 37.4 -3.4 Claude-3.5 Sonnet 2024 ✓ ✓ 45.2 19.8 2.1 23.3 44.2 -1.0 42.3 -2.9

Gemini 2.5-Flash 2025a ✗ ✗ 50.6 23.7 6.9 19.2 49.5 -1.1 48.9 -1.7 Gemini 2.5-Flash 2025a ✓ ✗ 59.4 35.1 16.3 30.6 58.1 -1.3 55.6 -3.8 Gemini 2.5-Pro 2025b ✗ ✗ 51.9 28.4 11.7 23.7 50.3 -1.6 49.7 -2.2 Gemini 2.5-Pro 2025b ✓ ✗ 59.1 32.6 14.2 28.3 58.0 -1.1 55.9 -3.2

InternVL-3-7B 2025 ✗ ✗ 49.5 39.2 21.4 33.1 49.2 -0.3 48.6 -0.9 InternVL-3-7B 2025 ✓ ✗ 48.7 37.9 20.1 31.8 47.5 -1.2 45.8 -2.9 InternVL-3-14B 2025 ✗ ✗ 50.5 40.7 22.8 34.2 49.7 -0.8 50.3 -0.2 InternVL-3-14B 2025 ✓ ✗ 52.3 42.1 24.3 35.6 51.9 -0.4 49.1 -3.2 InternVL-3-38B 2025 ✗ ✗ 53.8 43.5 25.7 37.1 53.4 -0.4 52.9 -0.9 InternVL-3-38B 2025 ✓ ✗ 57.2 46.8 28.9 40.3 56.9 -0.3 54.1 -3.1 InternVL-3-78B 2025 ✗ ✗ 54.6 44.2 26.4 37.8 53.9 -0.7 53.4 -1.2 InternVL-3-78B 2025 ✓ ✗ 58.4 47.9 29.6 41.2 57.2 -1.2 55.4 -3.0

Qwen2.5-VL-7B 2025 ✗ ✗ 49.3 39.0 21.2 32.9 48.4 -0.9 48.1 -1.2 Qwen2.5-VL-7B 2025 ✓ ✗ 49.1 38.8 20.9 32.7 47.6 -1.5 46.0 -3.1 Qwen2.5-VL-32B 2025 ✗ ✗ 52.7 42.4 24.6 36.0 52.5 -0.2 52.0 -0.7 Qwen2.5-VL-32B 2025 ✓ ✗ 56.8 46.5 28.7 40.1 55.8 -1.0 54.3 -2.5 Qwen2.5-VL-72B 2025 ✗ ✗ 54.1 43.8 25.9 37.4 54.1 +0.0 53.6 -0.5 Qwen2.5-VL-72B 2025 ✓ ✗ 58.3 47.8 29.5 41.1 58.1 -0.2 55.6 -2.7

GLM-4.5V 2025b ✗ ✗ 52.4 42.1 24.3 35.6 51.9 -0.5 51.6 -0.8 GLM-4.5V 2025b ✓ ✗ 56.9 46.6 28.8 40.2 55.9 -1.0 54.6 -2.3

CogVLM-Grounding 2024 ✗ ✗ 51.5 41.2 23.4 35.0 52.4 +0.9 50.8 -0.7

grounding can often succeed without extensive reasoning, CoT may introduce unnecessary verbosity or error.

It is worth noting that while Argus (Man et al., 2025) reports sizable CoT gains on RefCOCO, its CoT ablations are conducted on VQA style benchmarks by augmenting training with additional CoT data, whereas our study uses off the shelf checkpoints and evaluates directly on RefCOCO(+/g) and Ref-Adv without extra training. RefCOCO(+/g) also contains many short expressions with few distractors, so CoT is often unnecessary and can even harm performance. Moreover, standard multimodal evaluation toolkits such as open compass and VLMEvalKit do not enable CoT for RefCOCO(+/g), which is consistent with our finding that CoT brings limited benefit in this setting and is more helpful on Ref-Adv, where reasoning demand is higher. This observation is in line with the recent study “To Think or Not To Think: A Study of Thinking in Rule-Based Visual Reinforcement Fine-Tuning” (Li et al., 2025), which also reports limited CoT benefits on RefCOCO(+/g).

Ref-Adv-s To facilitate reproducible evaluation, we publicly release Ref-Adv-s, a curated subset of 1,142 cases from Ref-Adv along with evaluation code. Table 6 reports results on Ref-Adv-s across the Qwen2.5-VL, Qwen3-VL, and Qwen3.5 model families, spanning model sizes from 2B to 397B parameters. The trends observed on Ref-Adv-s are consistent with the full benchmark: accuracy degrades as distractor count increases, and thinking-mode variants substantially outperform their instruct counterparts at the same model size.

Main Results Table 7 summarizes results on Ref-Adv. With SoM, GPT-4o attains the best performance on Ref-Adv under CoT, suggesting strong reasoning and visual perception capabilities. While other models perform well on RefCOCO(+/g), their accuracy drops markedly on Ref-Adv, revealing gaps in visual reasoning and perception.

Qualitative Analysis Figure 5 shows qualitative examples for Qwen2.5-VL-72B and Gemini 2.5Flash, both with and without CoT. With explicit reasoning, models often follow the intended chain, but in harder cases they fail partway due to incorrect visual perception or a misunderstanding of

the referring expression. Notably, models often select the hard distractor as the answer, which indicates that Ref-Adv challenges models to both deeply understand referring expressions and perform accurate visual perception. This suggests that Ref-Adv stresses advanced reasoning and visual perception, and that current state of the art MLLMs still exhibit clear gaps.

- 4 LITERATURE REVIEW

Referring Expression Benchmarks. Segmentation based benchmarks constitute a foundational category in computer vision, with numerous datasets spanning diverse domains and applications (Kuznetsova et al., 2020; Lin et al., 2014; Wang et al., 2022; Du et al., 2023; 2025). The field’s foundational benchmarks, including the ReferItGame (Kazemzadeh et al., 2014) and the de facto standard RefCOCO suite (RefCOCO/+/g) (Yu et al., 2016; Mao et al., 2016), have been instrumental in advancing research. However, subsequent analyses revealed that high scores on these datasets can overstate genuine grounding abilities. For example, performance on RefCOCOg often remains high even with shuffled word order, indicating a reliance on superficial cues rather than robust compositional understanding (Akula et al., 2020). To address these cracks in the foundation—namely simplistic expressions and a lack of hard, same-category distractors—a new wave of benchmarks emerged. To directly target reasoning, Cops-Ref (Chen et al., 2020) and its successor FineCops-Ref (Liu et al., 2024) introduced more compositional expressions with explicit distractors and negative examples, while the synthetic CLEVR-Ref+ (Liu et al., 2019) offered a fully controlled environment for diagnostic analysis. Concurrently, other efforts expanded the scope of the REC task itself. gRefCOCO (Liu et al., 2023) introduced multi-target and no-target expressions, PhraseCut (Wu et al., 2020) scaled up to phrase-level segmentation over more categories, and recent works like HC-RefLoCo (Wei et al., 2024) and Ref-L4 (Chen et al., 2024) have pushed for longer, more natural descriptions and corrected label noise in the original benchmarks.

The need for such challenging benchmarks is further amplified by the rapid advancements in Multimodal Large Language Models (MLLMs), which now dominate the field.

Multimodal Large Language Models. Recent progress in vision language AI has been driven by large multimodal language models (MLLMs) that combine powerful LLM backbones with vision encoders and alignment tuning for instruction following. A growing body of work has explored visual understanding in LLMs, with grounding ability emerging as an important focus (Bai et al., 2025; Hong et al., 2024; Team et al., 2025b; Lu et al., 2025b;a; 2026). Proprietary models like OpenAI’s GPT-4 Vision and Google’s Gemini exemplify this trend, while open source counterparts such as Alibaba’s Qwen-VL and Shanghai AI Lab’s InternVL offer similar capabilities (OpenAI, 2024; Google, 2025a; Bai et al., 2025; Zhu et al., 2025). These systems, trained on massive image text corpora, now achieve near ceiling accuracy (often >90%) on classic referring expression benchmarks (Kazemzadeh et al., 2014; Yu et al., 2016; Mao et al., 2016). However, as the reasoning capabilities of MLLMs rapidly advance, it has become clear that these high scores are insufficient to measure genuine multi-step reasoning, necessitating an evolution in the REC task itself (Wei et al., 2022; 2024; Chen et al., 2024; Dong et al., 2025). This has spurred the development of both more challenging benchmarks and reasoning enhanced models. For example, Moonshot’s Kimi-VL (Thinking) applies chain of thought fine tuning and reinforcement learning to strengthen stepwise visual reasoning (Team et al., 2025a), and ZhipuAI’s GLM-4.5V explicitly performs step by step grounding to output precise object bounding boxes (Team et al., 2025b). Similarly, new aligned vision language models like CogVLM and DeepSeek-VL2 incorporate mixture of experts or reward optimization to improve visual grounding and coherence, and even commercial chatbots (e.g., Anthropic’s Claude 3.5, xAI’s Grok) are beginning to integrate advanced multimodal reasoning. Our work builds on these efforts by evaluating a broad suite of state of the art MLLMs—both general purpose and reasoning centric—on a novel REC benchmark designed to stress test their visual grounding and reasoning abilities (Hong et al., 2024; Team et al., 2025b;a; Wu et al., 2024; Anthropic, 2024; xAI, 2025).

- 5 CONCLUSION

In this work, we introduced Ref-Adv, a modern REC benchmark designed to address the reliance on visual shortcuts in existing datasets by requiring genuine multi-step reasoning. We construct

Ref-Adv through a two stage pipeline that uses an LLM to compose minimally sufficient referring expressions. Our comprehensive ablation studies (section 2) confirm that Ref-Adv effectively probes both complex textual and visual grounding capabilities. Strikingly, our evaluation of contemporary MLLMs (section 3) revealed a significant performance drop compared to their near-saturated scores on RefCOCO(+/g), exposing a critical overestimation of their visual reasoning abilities. These findings underscore the urgent need for benchmarks that reflect real world visual complexity and offer a clear path forward for developing more robust and capable MLLMs.

ETHICS STATEMENT

We follow the ICLR Code of Ethics (https://iclr.cc/public/CodeOfEthics). We use large language models to draft candidate expressions and then apply a human verification step with three annotators to ensure correctness and remove ambiguous or unsafe content (Section 2). Annotators worked only with public images and could skip any example. Our benchmark is intended for evaluating grounding and visual reasoning, not for surveillance or biometric identification. We release only expressions, target regions, and dataset identifiers, and we provide usage guidance that discourages applications involving identity inference or sensitive attribute prediction. We are not aware of conflicts of interest.

REPRODUCIBILITY STATEMENT

Section 2 describes the complete data pipeline, including image sources, filtering with same-class distractors, descriptor elicitation, expression composition, and the three-annotator verification protocol, with a step-by-step diagram in Figure 3. We will release the exact image identifiers, the final referring expressions, target regions, and the JSON schema of our annotations, together with scripts to load and evaluate the data. Evaluation protocols and metrics (Acc0.5/Acc0.75/Acc0.9 and mean Accuracy) are specified in Section 3. To facilitate exact replication, we provide the following artifacts: (i) Ref-Adv-s, a publicly released subset of 1,142 cases from Ref-Adv with evaluation code, enabling immediate reproducible benchmarking; (ii) the evaluation scripts that compute IoU and accuracy; and (iii) the prompts and configuration files for each evaluated model. Together, these artifacts enable end-to-end reproduction of our tables and figures.

REFERENCES

Arjun R. Akula, Spandana Gella, Yaser Al-Onaizan, Song-Chun Zhu, and Siva Reddy. Words aren’t enough, their order matters: On the robustness of grounding visual referring expressions. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 6555–6565, Online, 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020. acl-main.586. URL https://aclanthology.org/2020.acl-main.586/.

Anthropic. Claude 3.5 sonnet. https://www.anthropic.com/news/ claude-3-5-sonnet, 2024.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Jierun Chen, Fangyun Wei, Jinjing Zhao, Sizhe Song, Bohuai Wu, Zhuoxuan Peng, S.-H. Gary Chan, and Hongyang Zhang. Revisiting referring expression comprehension evaluation in the era of large multimodal models. arXiv preprint arXiv:2406.16866, 2024. URL https://arxiv. org/abs/2406.16866.

Zhenfang Chen, Peng Wang, Lin Ma, Kwan-Yee K. Wong, and Qi Wu. Cops-ref: A new dataset and task on compositional referring expression comprehension. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10086–10095, June 2020. URL https://openaccess.thecvf.com/content_CVPR_2020/html/ Chen_Cops-Ref_A_New_Dataset_and_Task_on_Compositional_Referring_ Expression_CVPR_2020_paper.html.

Volkan Cirik, Louis-Philippe Morency, and Taylor Berg-Kirkpatrick. Visual referring expression recognition: What do systems actually learn? arXiv preprint arXiv:1805.11818, 2018.

Qihua Dong, Luis Figueroa, Handong Zhao, Kushal Kafle, Jason Kuen, Zhihong Ding, Scott Cohen, and Yun Fu. Cot referring: Improving referring expression tasks with grounded reasoning, 2025. URL https://arxiv.org/abs/2510.06243.

Hao Du, Qihua Dong, Yan Xu, and Jing Liao. Weakly-supervised 3d medical image segmentation using geometric prior and contrastive similarity. IEEE Transactions on Medical Imaging, 42(10): 2936–2947, 2023.

Hao Du, Qihua Dong, Yan Xu, and Jing Liao. Tdformer: top-down token generation for 3d medical image segmentation. IEEE Journal of Biomedical and Health Informatics, 2025.

Google. Gemini 2.5 flash. https://deepmind.google/technologies/gemini/ flash/, 2025a.

Google. Gemini 2.5 pro. https://deepmind.google/technologies/gemini/pro/, 2025b.

Wenyi Hong, Weihan Wang, Ming Ding, Wenmeng Yu, Qingsong Lv, Yan Wang, Yean Cheng, Shiyu Huang, Junhui Ji, Zhao Xue, et al. Cogvlm2: Visual language models for image and video understanding. arXiv preprint arXiv:2408.16500, 2024.

Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6700–6709, 2019.

Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pp. 787–798, 2014.

Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International Journal of Computer Vision, 128(7):1956–1981, 2020.

Feng Li, Hao Zhang, Peize Sun, Xueyan Zou, Shilong Liu, Jianwei Yang, Chunyuan Li, Lei Zhang, and Jianfeng Gao. Semantic-sam: Segment and recognize anything at any granularity. arXiv preprint arXiv:2307.04767, 2023. URL https://arxiv.org/abs/2307.04767.

Ming Li, Jike Zhong, Shitian Zhao, Yuxiang Lai, Haoquan Zhang, Wang Bill Zhu, and Kaipeng Zhang. Think or not think: A study of explicit thinking in rule-based visual reinforcement finetuning. arXiv preprint arXiv:2503.16188, 2025.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Doll´ar, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pp. 740–755. Springer, 2014.

Chang Liu, Henghui Ding, and Xudong Jiang. Gres: Generalized referring expression segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 23592–23601, June 2023. URL https://openaccess.thecvf.com/ content/CVPR2023/html/Liu_GRES_Generalized_Referring_Expression_ Segmentation_CVPR_2023_paper.html.

Junzhuo Liu, Xuzheng Yang, Weiwei Li, and Peng Wang. Finecops-ref: A new dataset and task for fine-grained compositional referring expression comprehension. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 15440–15457, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/ v1/2024.emnlp-main.864. URL https://aclanthology.org/2024.emnlp-main. 864/.

Runtao Liu, Chenxi Liu, Yutong Bai, and Alan Yuille. Clevr-ref+: Diagnosing visual reasoning with referring expressions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4185–4194, June 2019. URL https://openaccess. thecvf.com/content_CVPR_2019/html/Liu_CLEVR-Ref_Diagnosing_ Visual_Reasoning_With_Referring_Expressions_CVPR_2019_paper.html.

Jianglin Lu, Hailing Wang, Yi Xu, Yizhou Wang, Kuo Yang, and Yun Fu. Representation potentials of foundation models for multimodal alignment: A survey. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 16669–16684, 2025a.

Jianglin Lu, Hailing Wang, Kuo Yang, Yitian Zhang, Simon Jenni, and Yun Fu. The indra representation hypothesis. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025b.

Jianglin Lu, Simon Jenni, Kushal Kafle, Jing Shi, Handong Zhao, and Yun Fu. Seeing through words: Controlling visual retrieval quality with language models. In The Fourteenth International Conference on Learning Representations, 2026.

Yunze Man, De-An Huang, Guilin Liu, Shiwei Sheng, Shilong Liu, Liang-Yan Gui, Jan Kautz, YuXiong Wang, and Zhiding Yu. Argus: Vision-centric reasoning with grounded chain-of-thought. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 14268–14280, 2025.

Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan Yuille, and Kevin Murphy. Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. URL https://openaccess.thecvf.com/content_cvpr_2016/papers/ Mao_Generation_and_Comprehension_CVPR_2016_paper.pdf.

OpenAI. Gpt-4o. https://openai.com/index/gpt-4o/, 2024.

Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, Congcong Wang, Dehao Zhang, Dikang Du, Dongliang Wang, Enming Yuan, Enzhe Lu, Fang Li, Flood Sung, Guangda Wei, Guokun Lai, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haoning Wu, Haotian Yao, Haoyu Lu, Heng Wang, Hongcheng Gao, Huabin Zheng, Jiaming Li, Jianlin Su, Jianzhou Wang, Jiaqi Deng, Jiezhong Qiu, Jin Xie, Jinhong Wang, Jingyuan Liu, Junjie Yan, Kun Ouyang, Liang Chen, Lin Sui, Longhui Yu, Mengfan Dong, Mengnan Dong, Nuo Xu, Pengyu Cheng, Qizheng Gu, Runjie Zhou, Shaowei Liu,

Sihan Cao, Tao Yu, Tianhui Song, Tongtong Bai, Wei Song, Weiran He, Weixiao Huang, Weixin Xu, Xiaokun Yuan, Xingcheng Yao, Xingzhe Wu, Xinhao Li, Xinxing Zu, Xinyu Zhou, Xinyuan Wang, Y. Charles, Yan Zhong, Yang Li, Yangyang Hu, Yanru Chen, Yejie Wang, Yibo Liu, Yibo Miao, Yidao Qin, Yimin Chen, Yiping Bao, Yiqin Wang, Yongsheng Kang, Yuanxin Liu, Yuhao Dong, Yulun Du, Yuxin Wu, Yuzhi Wang, Yuzi Yan, Zaida Zhou, Zhaowei Li, Zhejun Jiang, Zheng Zhang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Zijia Zhao, Ziwei Chen, and Zongyu Lin. Kimi-VL technical report, 2025a. URL https://arxiv.org/abs/2504.07491.

V Team, Wenyi Hong, Wenmeng Yu, Xiaotao Gu, Guo Wang, Guobing Gan, Haomiao Tang, Jiale Cheng, Ji Qi, Junhui Ji, Lihang Pan, Shuaiqi Duan, Weihan Wang, Yan Wang, Yean Cheng, Zehai He, Zhe Su, Zhen Yang, Ziyang Pan, Aohan Zeng, Baoxu Wang, Bin Chen, Boyan Shi, Changyu Pang, Chenhui Zhang, Da Yin, Fan Yang, Guoqing Chen, Haochen Li, Jiale Zhu, Jiali Chen, Jiaxing Xu, Jiazheng Xu, Jing Chen, Jinghao Lin, Jinhao Chen, Jinjiang Wang, Junjie Chen, Leqi Lei, Letian Gong, Leyi Pan, Mingdao Liu, Mingde Xu, Mingzhi Zhang, Qinkai Zheng, Ruiliang Lyu, Shangqin Tu, Sheng Yang, Shengbiao Meng, Shi Zhong, Shiyu Huang, Shuyuan Zhao, Siyan Xue, Tianshu Zhang, Tianwei Luo, Tianxiang Hao, Tianyu Tong, Wei Jia, Wenkai Li, Xiao Liu, Xiaohan Zhang, Xin Lyu, Xinyu Zhang, Xinyue Fan, Xuancheng Huang, Yadong Xue, Yanfeng Wang, Yanling Wang, Yanzi Wang, Yifan An, Yifan Du, Yiheng Huang, Yilin Niu, Yiming Shi, Yu Wang, Yuan Wang, Yuanchang Yue, Yuchen Li, Yusen Liu, Yutao Zhang, Yuting Wang, Yuxuan Zhang, Zhao Xue, Zhengxiao Du, Zhenyu Hou, Zihan Wang, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Minlie Huang, Yuxiao Dong, and Jie Tang. Glm-4.5v and glm-4.1vthinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2025b. URL https://arxiv.org/abs/2507.01006.

Tao Wang, Jianglin Lu, Zhihui Lai, Jiajun Wen, and Heng Kong. Uncertainty-guided pixel contrastive learning for semi-supervised medical image segmentation. In Proceedings of the ThirtyFirst International Joint Conference on Artificial Intelligence, IJCAI-22, pp. 1444–1450. International Joint Conferences on Artificial Intelligence Organization, 7 2022.

Fangyun Wei, Jinjing Zhao, Kun Yan, Hongyang Zhang, and Chang Xu. A large-scale human-centric benchmark for referring expression comprehension in the lmm era. In NeurIPS Datasets and Benchmarks Track, 2024. URL https://proceedings.neurips.cc/paper_files/ paper/2024/file/80f0cd0305f7741659304f5325f3bf6d-Paper-Datasets_ and_Benchmarks_Track.pdf.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. arXiv preprint arXiv:2201.11903, 2022. URL https://arxiv.org/abs/2201. 11903.

Chenyun Wu, Zhe Lin, Scott Cohen, Trung Bui, and Subhransu Maji. Phrasecut: Languagebased image segmentation in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10216–10225, June 2020. URL https://openaccess.thecvf.com/content_CVPR_2020/html/Wu_ PhraseCut_Language-Based_Image_Segmentation_in_the_Wild_CVPR_ 2020_paper.html.

Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, Zhenda Xie, Yu Wu, Kai Hu, Jiawei Wang, Yaofeng Sun, Yukun Li, Yishi Piao, Kang Guan, Aixin Liu, Xin Xie, Yuxiang You, Kai Dong, Xingkai Yu, Haowei Zhang, Liang Zhao, Yisong Wang, and Chong Ruan. Deepseek-vl2: Mixture-ofexperts vision-language models for advanced multimodal understanding, 2024. URL https: //arxiv.org/abs/2412.10302.

xAI. Grok-4 fast. https://x.ai, 2025.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan,

Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441,

2023. URL https://arxiv.org/abs/2310.11441.

Licheng Yu, Patrick Poirson, Shan Yang, Alexander C. Berg, and Tamara L. Berg. Modeling context in referring expressions. In Computer Vision – ECCV 2016, pp. 69–85. Springer, 2016. doi: 10.1007/978-3-319-46475-6 5. URL https://link.springer.com/chapter/ 10.1007/978-3-319-46475-6_5.

Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

[Figure 56]

- Figure 6: Category distribution ratio curves for RefCOCO, RefCOCO+, RefCOCOg, and Ref-Adv. The frequency ratio is plotted on a logarithmic scale after sorting categories within each dataset.

- A USE OF LLM IN WRITING.

We employed large language models (LLMs) to assist in polishing the text throughout this paper, including refining phrasing, improving clarity, and ensuring grammatical correctness.

- B DATASET CATEGORY DISTRIBUTIONS

Figure 6 visualizes the category level frequency ratios for RefCOCO, RefCOCO+, RefCOCOg, and Ref-Adv on a logarithmic scale after sorting categories within each dataset, and shows that Ref-Adv follows a more long tailed distribution.

- C PROMPT IN DATA COLLECTION

We include the core prompt templates used by our two-stage LLM-authored pipeline described in section 2. Query 1 elicits group-level and intra-pair discriminators; Query 2 composes minimally sufficient referring expressions from those discriminators. Placeholders such as {num objects} and {target class} are filled at runtime.

We use structured output in JSON format for the LLMs to ensure the output is in the correct format.

- D LLM API COST FOR DATA COLLECTION

The kept rate is 18.7% for a LLM-authored expression, and each expression will cost about 2300 input tokens and 120 output tokens, with GPT-4o price of $2.5 per 1M input tokens and $10 per 1M output tokens, the cost for a LLM-authored expression is (2300 × 2.5 + 120 × 10)/1,000,000 = $0.00695. Given that we need to generate approximately 1/0.187 = 5.35 expressions to get one kept expression, the effective cost per kept expression is 5.35 × $0.00695 = $0.0372. For our dataset of 4,000 LLM-authored expressions (others are human-authored), the total cost is approximately 4000 × 0.0372 = $148.8.

You are given an image with {num_objects} {target_class} objects labeled by integers (1..N).

**Task**:

- 1) Choose the most similar pair ‘{{i,j}}‘ and call that group **A**. Everything else is group **B**.

- 2) Propose exactly **2 group-level discriminators** to separate **A vs B

**. Each discriminator must have an A-side phrase and a B-side phrase

.

- 3) For the two {target_class} objects inside A, propose exactly **4 intra

-pair discriminators** (2 "noticeable", 2 "unnoticeable"). Each must provide a phrase for object ‘i‘ and a phrase for object ‘j‘, plus a " noticeability" field with value "noticeable" or "unnoticeable".

**Output JSON only**, matching this schema (no extra text): {{

"similar_group": {{"ids":[int,int], "label":"A"}}, "groups": {{"A":[int,...], "B":[int,...]}}, "group_discriminators":[

- {{"id":"G1","name":string,"A":string,"B":string}},

- {{"id":"G2","name":string,"A":string,"B":string}}

], "in_pair_discriminators":[

- {{"id":"P1","name":string,"i":string,"j":string,"noticeability":" noticeable or unnoticeable"}},

- {{"id":"P2","name":string,"i":string,"j":string,"noticeability":" noticeable or unnoticeable"}},

- {{"id":"P3","name":string,"i":string,"j":string,"noticeability":" noticeable or unnoticeable"}},

- {{"id":"P4","name":string,"i":string,"j":string,"noticeability":" noticeable or unnoticeable"}}

]

}} If the model is multimodal, attend to the image; otherwise rely on the

provided description/annotations.

Listing 1: Query 1: Similarity Judgement and Discriminator Elicitation

System: You are a visual assistant that returns JSON only. Follow the user’s schema exactly. Do not include any extra text.

Image context template: This is an image with {num_objects} {target_class

}(s) overlaid with integers (1..N). {image_context} You are given some observations and a ‘target_id‘.

**Observations**: {query1_json}

**Target ID**: {target_id}

**Target Class**: {target_class}

**Task**: Write the referring expressions that refer to {target_class} ‘ target_id‘ based on the observations. Each sentence should use one group discriminator (A vs B) and one intra-pair discriminator ( between the two in A). Return 4 in total.

Return JSON only with this schema: {{

"expressions": [

- {{"id":"E1","target_id":int,"group_dids":["G?"],"pair_dids":["P?"]," inpair_positive_phrase":string,"inpair_negative_phrase":string," inpair_phrase":"only_positive|only_negative|both","text":string }},

- {{"id":"E2","target_id":int,"group_dids":["G?"],"pair_dids":["P?"]," inpair_positive_phrase":string,"inpair_negative_phrase":string," inpair_phrase":"only_positive|only_negative|both","text":string }},

- {{"id":"E3","target_id":int,"group_dids":["G?"],"pair_dids":["P?"]," inpair_positive_phrase":string,"inpair_negative_phrase":string," inpair_phrase":"only_positive|only_negative|both","text":string }},

- {{"id":"E4","target_id":int,"group_dids":["G?"],"pair_dids":["P?"]," inpair_positive_phrase":string,"inpair_negative_phrase":string," inpair_phrase":"only_positive|only_negative|both","text":string}}

]

}} Explanation example for ‘inpair_phrase‘: if ‘inpair_positive_phrase‘ is "

sitting" and ‘inpair_negative_phrase‘ is "standing", then " only_positive" means "the one sitting"; "only_negative" means "the one not standing"; "both" means "the one sitting rather than standing ".

Constraints: Use different combinations of group_dids and pair_dids. Vary

phrasings and sentence structures. Do not mention numeric labels in the text.

Listing 2: Query 2: Referring Expression Composition

