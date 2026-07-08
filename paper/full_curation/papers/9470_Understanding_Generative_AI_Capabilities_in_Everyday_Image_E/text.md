arXiv:2505.16181v2[cs.CV]26May2025

### Understanding Generative AI Capabilities in Everyday Image Editing Tasks

##### Mohammad Reza Taesiri1*

##### Brandon Collins2∗

##### Logan Bolton2∗

##### Viet Dac Lai3

mtaesiri@gmail.com

blc0063@auburn.edu

logan.bolton@auburn.edu

daclai@adobe.com

##### Franck Dernoncourt3

##### Trung Bui3

##### Anh Totti Nguyen2

dernonco@adobe.com

bui@adobe.com

anh.ng8@gmail.com

1University of Alberta 2Auburn University 3Adobe Research

[Figure 1]

[Figure 2]

##### Abstract

[Figure 3]

[Figure 4]

Evaluating human & AI edits

83k requests & 305k edits

[Figure 5]

[Figure 6]

[Figure 7]

46 AIimage editors

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Generative AI (GenAI) holds significant promise for automating everyday image editing tasks, especially following the recent release of GPT-4o on March 25, 2025. However, what subjects do people most often want edited? What kinds of editing actions do they want to perform (e.g., removing or stylizing the subject)? Do people prefer precise edits with predictable outcomes, or highly creative ones? By understanding the characteristics of real-world requests and the corresponding edits made by freelance photo-editing wizards, can we draw lessons for improving AI-based editors and determine which types of requests can currently be handled successfully by AI editors? In this paper, we present a unique study addressing these questions by analyzing 83k requests with their associated 305k edits from the recent 12 years on the /r/PhotoshopRequest Reddit community. According to human ratings, approximately only 33% of requests can be fulfilled by the best AI editors (including , , ). Interestingly, AI editors perform worse on lowcreativity requests that require precise editing than on more open-ended requests. They often struggle to preserve the identity of people and animals, and frequently make non-requested touch-ups. On the other side of the table, VLM judges (e.g., o1) perform differently than human judges and may prefer AI edits over human edits. Code and qualitative examples are available at: https://psrdataset.github.io/.

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

Taxonomy & Annotation

|328 requests|
|---|

2k AIedits 1.6khuman edits

15editing actions

3Creativity

WordNet subject

Judges󰥺

levels

3VLM voters:4o,o1, Gemini 2.0 Flash-T

83k

305k

12k

122human voters → 4.3kvotes

requests/ posts

edited images

multi-image requests

→ 30k votes

Q: Which edit best satisfies the request?

Human study

# 󰥺

Request: I'd love to see how crazy this could get… Thank you in advance!!

[Figure 16]

[Figure 17]

[Figure 18]

✅ Human

###### ☑ AI

☑ Tie

Source Human AI

While human edits are strongly preferred, AI can successfully handle ~⅓of requests

Figure 1: We propose PSR, the largest dataset of realworld image-editing requests and human-made edits. PSR enables the community (and our work) to identify types of requests that can be automated using existing AIs and those that need improvement. PSR is the first dataset to tag all requests with WordNet subjects, realworld editing actions, and creativity levels.

[Figure 19]

ever, four important questions remain open:

- Q1: What are the real everyday image editing requests and needs of users?
- Q2: According to human judgment, what % of such requests can be satisfied by existing AIs?
- Q3: What are the improvement areas for AI editors compared to human editors?
- Q4: Are vision language models (VLMs) judging AI-edited images similarly to human judges?

##### 1 Introduction

Q1 and Q2 are unanswered partly because many prior datasets (e.g., [4, 44, 42, 58]) contain madeup requests written by either human annotators or AIs based on the source image or the (source, target image) pair (see Tab. 1). Those request distributions may not reflect the actual editing needs of users as well as the challenges posed by the real requests, which may have typos, distraction or ambiguity (e.g., “I’d love to see how crazy this

GenAI for images has gained enormous research interest [17] and created a 2023 market of $300M, which is estimated to multiply [19]. Specifically, text-based image editing is an increasingly highdemand task [45], especially after the recent GPT4o and Gemini-2.5-Pro image generators [36]. How-

*While MRT is the lead, all first 3 authors made major contributions to code, running experiments, and analyzing results.

could get... Thank you in advance!!” in this post; Fig. 1). On the other hand, some datasets (e.g., [54, 4, 42, 58]) feature images AI-edited and therefore do not represent the real edits by humans.

We aim to answer these four questions by analyzing 305k tuples of (source image, text request, edited image) from the /r/PhotoshopRequest (PSR) Reddit channel, which is the largest public online community [11, 46] that shares diverse, everyday image-editing needs with corresponding edits by PSR wizards1. PSR has 1.7M users and receives an average of 141 new requests per day (our statistics for 2025) with a peak as high as 226 per day [9].

To answer Q1, Q2, and Q3, our closed-loop study (a) analyzes 305k tuples, i.e., 83k unique requests × 3.67 human-edited images per request;

- (b) sends all (request, source image) pairs to imageediting models to collect AI edits; (c) performs a human study over a set of 328 requests (PSR-328) to collect over 4.5k ratings to compare how 1,644 PSR-wizard edits fare against 2,296 AI edits on the same requests to identify areas where AIs perform well and fall short. Our work is the first to compare three state-of-the-art (SOTA) image editors: GPT-4o [35], Gemini-2.0-Flash [21], and SeedEdit [45] as well as 46 other AI models on HuggingFace, for a total of 49 AI editors. Furthermore, to address Q4, we compare human ratings against those by 3 SOTA vision-language models (VLMs): GPT-4o, o1 [33], and Gemini-2.0-FlashThinking [14]. Our main findings are:

[Figure 20]

[Figure 21]

- • 66% of the time, human judges still prefer human edits over AI edits (Sec. 5.1).
- • While SOTA VLMs are excellent at regular visual tasks [8], on image-edit judgment, VLMs can be extremely biased, e.g., o1 prefers GPT4o edits 85% of the time, which is in stark contrast to human judgment (Sec. 5.2).
- • AIs often add extra non-requested changes that improve the aesthetics of the image but also fail to preserve the subject’s identity (Sec. 5.3).
- • AIs either tie or win 33.35% of requests, while PSR wizards still outperform AI editors on the remaining 66.65% (Sec. 5.4).

##### 2 Related Work

Online image editing communities have been studied to understand user intent, common editing

1Advanced image editors who are granted to handle paid editing requests in this particular subreddit.

patterns, and challenges in automating image editing [31, 27, 43]. However, prior image-editing taxonomies [31] of editing actions often correspond to the low-level functions available in image-editing software (e.g., zoom or select) and do not correspond to the editing intent of humans who request changes (e.g., human would not want to select an object in an image but rather it is the image editor). In contrast, we build our taxonomy of actions intended by users who post requests on the PSR channel and therefore our taxonomy is based on Reddit requests instead of based on what functionalities are available in image editing software [31, 3]. Furthermore, existing taxonomies [31, 3] contain ambiguous, overlapping labels such as (zoom vs. crop), which we resolve in our taxonomy. Our taxonomy also contains 5 labels that reflect more up-to-date operations: super-resolution, relight, specialized, re-color, and merge, which do not exist in prior taxonomies [31, 3].

Editing datasets Synthetic datasets, e.g., InstructPix2Pix [4], UltraEdit [58], and Emu Edit [42], enable a large-scale training-set of AI-generated images but fail to capture the real distribution of both requests and edits made by humans. Hybrid datasets like HIVE [56] (with human feedback) and Seed-Edit-Data [13] (synthetic and humanannotated) balance scale and authenticity. Meanwhile, real user-request datasets, such as IER [48] and GIER [43], offer greater authenticity by collecting genuine editing needs from online communities, but remain constrained in scale and diversity (63–87% of requests are low -creativity Tab. 1). Unlike the above works, our PSR contains real image-editing requests by Reddit users and also human-edited images by PSR wizards.

The most similar dataset to PSR is RealEdit [47], which is a concurrent work that scrapes 57k requests also from Reddit. Our dataset is ∼3× larger than RealEdit in the number of (request, source image, edited image) tuples (Tab. 1). Unlike RealEdit and all previous works, our work is the first to address all four questions Q1–Q4, i.e., we (a) analyze everyday image-editing needs submitted by real users; (b) test SOTA image editors on these requests; and (c) compare how SOTA VLMs judge image edits differently than human judges.

We label each request with subject labels that correspond to WordNet categories (while RealEdit uses 14 manually-defined subjects) and three creativity levels (which do not exist in RealEdit). PSR features 15 detailed editing actions compared to 6

Table 1: PSR is the largest-scale dataset of real-world requests and PSR-wizard edits.

No. of No. of Source Edit Request Requests Creativity requests edits image generator writer based on Low Med High

Dataset

IER [48] 2019 4k 4k reddit human ✓ human reddit ✓ 78% 15% 7% GIER [43] 2020 6k 6k reddit human ✓ human reddit ✓ 63% 34% 3%

MA5k-Req [44] 2021 24k 24k real Ps [5] Amazon MT image pairs 100% 0% 0% InstructPix2Pix [4] 2023 454k 454k SD [16] SD [16] GPT-3 image pairs 12% 44% 42% MagicBrush [54] 2023 10k 10k MS COCO DALL-E 2 Amazon MT source image 62% 25% 12%

HIVE [56] 2024 1.1M 1.1M SD [16] SD [16] BLIP image pairs 48% 30% 22% EmuEdit [42] 2024 10M 10M Emu [10] Emu [10] Llama 2 source image & task 54% 20% 26%

AURORA [25] 2025 280k 280k mixed mixed GPT-4o image pairs 96% 4% 0% UltraEdit [58] 2025 4M 4M MS COCO SD GPT-4o, human source image 43% 21% 36% RealEdit [47] 2025 57k 94k reddit human ✓ human reddit ✓ 58% 36% 6%

PSR (ours) 2025 83k 305k reddit human ✓ human reddit ✓ 56% 28% 16% PSR-328 (ours) 2025 328 3.9k reddit human, AIs ✓ human reddit ✓ 33% 33% 33%

labels of RealEdit [47].

Automated evaluation in image editing adopts both automated metrics and human evaluation. Automated metrics provide an objective and scalable way to assess edits by measuring fidelity [24, 51], realism [39], alignment with given instructions [55, 29, 23], or multiple facets of an edit [30]. Similarly, we use the LAION Aesthetics score to measure the fidelity of images. Furthermore, to the best of our knowledge, our work is the first work to test SOTA VLMs in rating edited images given a (text request, source image) pair and compare their results with human ratings.

Human ratings While automatic metrics are convenient, human evaluation remains the most reliable method in capturing subjective qualities such as realism, coherence, and user satisfaction. Public benchmarks facilitate structured human assessments, often using rating scales [22, 1, 50] or pairwise comparisons [20] to evaluate editing performance. Similar to GenAI Arena [20], we also present Win, Lose, and Tie options to human raters. GenAI Arena compares models based on their images generated from scratch while our work tests image editing.

Prior human studies used a smaller set of images—EditVal [1], TedBench [22], and EditBench [50] contain 92, 100, and 240 source images. In contrast, PSR-328 test set features 328 real-world requests with controlled creativity levels, uniform subject labels and editing actions.

##### 3 PSR Dataset Construction

We source our data from /r/PhotoshopRequest (2013–2025) via two approaches: (1) PushShift [2] provide historical data (2013 up to Nov 2022) (2) a custom crawler written for Reddit to download recent data (Oct 2024–Feb 2025).

###### 3.1 Taxonomy

We present a taxonomy of requests where we label each request with the following labels: subject, editing action verb, and creativity level.

The subject identifies the element being modified, the action verb specifies the edit, and the creativity level distinguishes routine tasks from open-ended ones. This taxonomy enables precise analysis of automated image editing and shows that even routine edits, like object removal, vary significantly by subject—person, animal, or object. While low-creativity tasks often suit standard automation tools, high-creativity tasks (Fig. 2) demand models with greater flexibility.

Subject The subject of an image editing request is the specific element being modified—e.g., in a request to remove a person, the subject is the person. Subjects may include objects, persons, or the entire image. To systematically classify subjects from user instructions, we leverage WordNet’s taxonomy [12]. We first extract subjects from raw instructions, then match each to the nearest synset (semantic category) in WordNet’s structured lexical database, providing standardized classification and reduced ambiguity.

Editing Action Users may describe their requests vaguely, (e.g., “Make this look better”) instead of the more technically precise phrasing such as “Improve the lighting to make the subject stand out”. Because users may not know what editing actions are needed to achieve the result or that they want an surprise, out-of-the-box result. To properly categorize user intent, we develop a diverse list of 15 action verbs that cover various editing actions (Tab. 2).

Prior editing action taxonomies [31, 48] are tied to low-level tools in popular image editing software, failing to capture high-level user intent and

Request Source Human Edit AI Edit Votes Creativity Action

Can someone please edit this photo to remove the quote and leave just the original photo? Thank you!

[Figure 22]

[Figure 23]

[Figure 24]

Human: 2

- (a)
- (b)
- (c)
- (d)
- (e)
- (f)

Low Delete

[Figure 25]

AI: 0

[Figure 26]

[Figure 27]

[Figure 28]

Can anyone fix the blur and zoom in? Best edit gets a tip! - Best quality edit gets a tip!

Human: 8

Medium Adjust, Zoom

[Figure 29]

AI: 0

[Figure 30]

[Figure 31]

[Figure 32]

Human: 2

Can someone Photoshop this so the finger goes up his nose and out his eye?

High Transform

[Figure 33]

AI: 0

[Figure 34]

[Figure 35]

[Figure 36]

Can anyone photoshop out Aladdin, Abu, and the play button? Trying to just get the city-scape.

Human: 1

Low Delete

[Figure 37]

AI: 3

[Figure 38]

[Figure 39]

[Figure 40]

Can't get rid of this cast in Photoshop. Can someone guide me? Thank you.

Human: 0

Medium Adjust

[Figure 41]

- AI: 2
- AI: 3

[Figure 42]

[Figure 43]

[Figure 44]

Can someone Cartoonize or line draw this photo (it’s for a round sticker)

Human: 0

High

Apply

[Figure 45]

- Figure 2: Example cases from PSR dataset where PSR wizard edits were preferred by human raters over the AI edits (a-c) and samples where AI edits was preferred (d-f). (a): The human edit completes the request, but the AI edit removes the people, which was not requested. (b): The human edit completes the request, but the AI edit generates a similar image with people resembling those in the source image, although with different identities. (c): The human edit completes the request, but the AI edit does not because the finger does not go through the nose and out the eye. (d): Both edits make the requested removals, but the AI edit makes the image sharper and adds a house in the background (which was not requested). (e): The human edit reduces the color cast but leaves behind a bluish tint and muted tones, while the AI edit successfully restores realistic colors and contrast. (f): The human edit removes the background and applies a soft photo filter that lacks stylization, while the AI edit transforms the dogs into bold, clean cartoons. More results are available at: https://huggingface.co/spaces/PSRDataset/PSR-Battle-Results

omitting actions like super-resolution. This limitation motivated us to develop a new taxonomy with comprehensive coverage of modern editing techniques.

To develop our taxonomy, we feed a random subset of 5,000 edit requests into GPT-4o-mini and prompt it to summarize common editing actions. Additionally, we consult image editing experts in the field to refine our list of actions to accurately reflect image-synthesis tasks in computer vision (e.g., super-resolution). Tab. 2 presents the final list of action verbs.

Creativity levels We categorize requests into three creativity levels:

1. Low -creativity requests such as “remove a person” or “erase an object” expect a pre-

dictable outcome (Fig. 2d).

- 2. Medium -creativity requests such as “change the background” or “adjust lighting

to look cinematic” allow some room for interpretation and variability in results (Fig. 2e).

- 3. High -creativity requests, e.g., “make this image look magical” or “transform this into a fantasy scene”, require creative ideas and can yield widely different outcomes depending on the editor imagination (Fig. 2f).

Our creativity classification differentiates between requests for precise technical edits vs. imaginative, open-ended transformations and enables breakdown analysis across requests, actions, and creativity. For example, the labels in PSR show

- Table 2: List of image editing action verbs from our taxonomy with sample user requests. See appendix C for the VLM prompts that are used to generate these action labels.

Editing action Description and Sample Request Editing action Description and Sample Request

Insert new elements, objects, text, or effects. e.g. “Add a copyright watermark to the bottom right.” adjust

Modify properties like tones, contrast, and saturation. e.g. “Increase saturation a bit on the elephants.”

(Fig. 2b)

add

Add filters, styles, or effects. e.g. “Apply a vintage film effect.” clone

Duplicate elements inside the image. e.g. “Use cloning tool to blend grass over dirt patches.”

apply (Fig. 2f)

Trim edges for a smaller image. e.g. “Crop to square format for social media.” delete

Remove elements, objects, or imperfections. e.g. “Remove the jacket hanging from the girl’s side.”

(Fig. 2a)

crop

Substitute objects or text. e.g. “Please change the pamphlet into a dictionary.” transform

Flip, scale, rotate, or skew elements. e.g. “Fix the perspective of the building.”

replace

Reposition elements while keeping the rest unchanged. e.g. “Shift the logo 20 pixels up.” merge

Combine elements or effects. e.g. “Create a panorama from these shots.”

move

Increase resolution for clearer details. e.g. “Can someone upscale this image to 4K resolution?” re-color

Change the color of an element, object, or text. e.g. “Can someone change the dog’s fur to black?”

super-resolution

Adjust lighting conditions. e.g. “Can someone make lighting better / remove shadows?” zoom

Change scale to focus or zoom out. e.g. “Zoom in on the man.”

(Fig. 2b)

relight

Advanced or composite editing tasks. e.g. “Can someone vectorize this logo without background?”

specialized

that delete is indeed the most common action in

low -creativity requests (Fig. 3c). That is, users often specify an specific element in the image to be deleted. In contrast, adding a new elements into the image has much larger space for creativity: what object, how, and where to add? As the result, we find add to be the most common action in high -creativity requests (Fig. 3c).

###### 3.2 Dataset Annotation Process

We use GPT-4o-mini [34] and InternVL-2.5-38B [7] to annotate our dataset. GPT-4o-mini generates taxonomy labels, while InternVL-2.5 handles image captioning and keyword extraction. We prompt InternVL-2.5 to summarize each image into 36 JSON keys (see Fig. A5), capturing attributes such as image type (e.g., photo or digital art), location, weather, presence of people, and object lists. To filter out posts unrelated to image editing (e.g., requests for image authentication), we prompt GPT4o-mini to output a binary flag, image editing relevance, indicating whether the post involves image editing. Prompting details, including model version and temperature, are in Appendix B.

Extracting Subject and Action Verbs We use zero-shot prompting (see Fig. A7) to extract actions from the request (Tab. 2). GPT-4o-mini is provided with a list of valid actions, their descriptions, the input image, and the user-provided request. We ask the model to first examine the image and then rewrite the instruction in clear and simplified language to eliminate ambiguity, and finally identify the subjects of the edits along with the corresponding editing actions.

Mapping Subjects to WordNet We map the

extracted subjects from the previous stage to WordNet’s synsets. Once the subjects are identified, we provide GPT-4o-mini with the image, instruction, and subject, instructing it to select the closest WordNet synset based on the given context. Since the generated synset may not always be valid, we perform a search within the WordNet lexical database using NLTK to find the closest matching synset and assign a final synset to the subject.

We use o1-Pro [37] to summarize WordNet subjects into higher-level semantic categories and organize them into 5 main categories and 12 subcategories (Tab. A2). Since the extracted WordNet synsets from the previous step vary in granularity, a reasoning model with long-context capabilities, such as o1-Pro, effectively groups these synsets into structured categories. This approach results in a more coherent and meaningful subject categories. Assigning Creativity Levels We use few-shot prompting (Fig. A10) to assign creativity levels. GPT-4o-mini receives the original image and request, along with examples annotated by creativity level, to classify the input accordingly.

###### 3.3 PSR dataset statistics

PSR consists of 82,976 requests and 305,806 edited images, categorized by creativity levels as 55.5% low, 28.2% medium, and 16.3% high (Fig. 3d). It includes 49,134 unique subjects, with 53.5% of subjects are under the People and Related category (Fig. 3b). The primary action requested by users is delete (32.9%), typically involving removing individuals or visual clutter to enhance aesthetics or professionalism. And these action distribution remain similar over the years (Fig. 3a). Detailed

100%

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

80%

###### ProportionofActions

60%

40%

20%

0%

2014 2016 2018 2020 2022 2024

Year

Delete Adjust Add

Recoloring

Transform

Move

Merge

Apply

Relighting

Replace

Zoom Clone

Superresolution

Crop

Specialized operation

(a) Editing action popularity per year

100

| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|
| | | | | | |5.3|%| |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | |11.6|%| |
| |6.6|%| | | | | | |
| | | | |5.0%| | | | |
| |6.1|%| | | | | | |
| | | | |6.7%| | | | |
| | | | | | |7.0|%| |
| |6.1|%| | | | | | |
| | | | |7.0%| | | | |
| | | | | | | | | |
| | | | | | |7.1|%| |
| | | | | | | | | |
| | | | |7.4%| | | | |
| | | | | | | | | |
| | | | | | | | | |
| |12.9|%| | | |9.6|%| |
| | | | |12.0%| | | | |
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |
| |51.2|%| |25.8%| |34.9|%| |
| | | | |20.7%| | | | |
| | | | | | |12.2|%| |
| | | | | | |5.3|%| |

80

Top Actions

Delete Adjust Add Replace Recoloring Superresolution Transform Apply Move Crop

| |
|---|

| |
|---|

60

Percentage(%)

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

40

| |
|---|

| |
|---|

| |
|---|

20

Creativity Distribution: Low: 55.5% Medium: 28.2% High: 16.3%

0

Low Medium High

(c) Action distribution per creativity level

household

natural vehicles

tools

pets

special

body

text

humans

Delete

body pets

humans

Adjust

text

humans

Add

Other

Superresolution

pets text body

Replace

Recoloring

Other

humans text

humans body

humans

(b) Most common actions & subjects

100%

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

80%

###### ProportionofCreativityLevels

60%

40%

20%

0%

2014 2016 2018 2020 2022 2024

Year

Low Medium High

(d) Creativity level per year

- Figure 3: Over 12 years of Reddit data, delete, adjust, and add are the top-3 most wanted actions (a). Specifically, humans, body parts, text, and pets are the most frequent WordNet subjects for such common actions (b). While delete and adjust are the top-2 most common actions in the low - and medium -creativity requests, add takes up the largest share (34.9%) in high -creativity, e.g., inserting some “interesting” background or objects into the scene

(c). Most requests (55.5%) require straightforward edits with low creativity (d).

dataset statistics are provided in Appendix A.

- 4 Experiment Setup

As user-written instructions often include unnecessary details (e.g., “Thank you in advance!!” in Fig. 1), GPT-4o-mini refines them to focus on the core editing task.

We compare AI edits and human edits to understand the preference of human raters and assess how VLMs can match human raters.

Additionally, for each request, we generate three AI-based image edits using 46 off-the-shelf image editing models hosted on HuggingFace Spaces [18] (see Tab. A5 for a list of models).

AI editors We process each request using three generalist SOTA image editing tools: SeedEdit [45], Gemini-2.0-Flash (Image Generation Experimental) [21], and GPT-4o [35].2 For each request, we generate two images: one using the original instruction (OI) and one using a simplified instruction (SI) generated by GPT-4o-mini.

[Figure 46]

PSR-328 dataset Via stratified sampling, we select a random PSR subset that has almost the same amount of images in three levels of creativity groups (114 low, 101 medium, and 113 high). On average, each request is edited by an average of five Reddit human editors, resulting in a total of 1,644 human edits. We also generate 7 different

2Due to access constraints and model safeguards, some edits from , , and are not available.

[Figure 47]

AI edits per request. In total, our human study has 10,405 unique 4-tuples (source image, request, AI edit, human edit) for evaluation—substantially larger than prior studies [1, 50, 20], while keeping the human annotation effort manageable.

Human study We conduct a comparative study to assess whether AI-generated or Reddit user edits better satisfy original requests. To ensure unbiased evaluation, randomly paired AI and human edits are shown to human raters, who vote (Win, Lose or Tie) on which best fulfills the request (see Appendix E for the user interface).

Automated metrics and VLM judges We use two automated evaluation methods to complement our human study: LAION Aesthetic Score [15, 49, 52] and VLM-as-a-Judge [28, 8, 53, 6]. The Aesthetic Score quantifies visual appeal using a classifier trained on large-scale human preference data while VLM-as-a-Judge uses VLMs to evaluate images by articulating visual qualities and explaining judgments. Both metrics act as proxies for human judgment, enabling scalable evaluation of AI- and human-edited images. We use LAION’s Aesthetic Score Predictor [41] for calculating aesthetic metrics, and three VLMs (GPT-4o [32], o1 [33], and Gemini-2.0-Flash-Thinking [14]) as

[Figure 48]

- as judges for the image edit rating task.

##### 5 Experimental Results

5.1 Human edits are strongly preferred over AI edits by human raters

We collect 4,359 votes from 122 users and Tab. 3 shows the results. On average, human edits are more preferred (66.0% of the time), while 25.8% of the time AI edits win, and they tie in the remaining 8.2% (Tab. 3; 25.8% ). This result is important given the global interest in the image generation capabilities of Gemini-2.0-Flash and GPT-4o [21].

AI-win-rates by models Human edits are more preferred compared to every single AI editor (Tab. 3; Human win rates ≥ 53% ). has the highest overall win rate (37.8%) over human editors, followed by (32.8%), and then (20.6%). HuggingFace models perform the worst (Tab. 3; 17.8%).

[Figure 49]

AI-win rates by creativity levels Human edits consistently are more preferred across all three levels. However, there is a clear trend that AI edits are more preferred as the requests require higher creativity. That is, the gap human-vs-AI becomes smaller as the tasks are more open-ended: 70.1%

vs. 21.1% for low creativity, (67.6% vs. 25.3%) for medium creativity, and (60.9% vs. 30.7%) for

high creativity.

Performance by editing actions On average over all 49 models (Tab. A8), AIs win the most on the following editing requests: merge (30.9%), apply (30.6%), and add (30.6%) and the least on zoom (10.7%), crop (15.5%), and move (20.2%). For a detailed breakdown, see Appendix E.2.

Qualitative analysis We also conduct a qualitative analysis (Appendix E.5) to identify patterns in cases where AI edits succeed or fail. From analyzing 206 cases where AI edits win votes, we find that 72% of the time, human edits poorly follow the instructions and AIs more closely follow the requests. In contrast, after analyzing 400 cases where AI loses, we find that 43% of the time, AIs misinterpret the request. In the remaining loss cases, AIs introduce unintended changes, artifacts or facial distortions. A key issue is their failure to preserve identity (Fig. 6a–b).

To quantify the issue of failing to preserve identity, we perform a controlled experiment where we repeatedly ask AI editors to only change the shirt color of a person of varying gender and ages. However, both and models tend to change the facial identity and even body shape over a sequence of multi-turn requests (Fig. 4 and Appendix G).

5.2 In rating edited images, VLMs are a poor proxy for human raters

Evaluating edited images is naturally a multimodal task that is challenging because it requires understanding the (text request, source image) pair and analyzing how the changes in the edited image satisfies the request. Assessing how SOTA VLMs perform at edit rating is important for (1) benchmarking and advancing future VLMs; and (2) automating the rating efforts currently performed by humans.

Experiment We collect over 10k ratings from each of three separate VLMs that serve as judges between human edits and AI edits. We replicate the same setup as in Sec. 5.1. Specifically, given a textual request, a source image, and two edited images, we ask each VLM to judge which edit better satisfies the request. In this setup, the images are labeled as Source, Edit A, and Edit B. Human-generated and AI-generated edits are randomly assigned to labels A and B. The VLM then evaluates the edits and delivers a verdict of either Edit A is better, Edit B is better, or Tie, indicating

- Table 3: Human and VLM evaluation results for image editing preferences across 114 low -, 101 medium -, and 113 high -creativity requests. Human raters often prefer Human edits over all AI edits, while VLMs show a strong preference for edits made by SeedEdit and GPT-4o . # denotes the number of votes (i.e., pair-wise comparisons). Bold numbers are the highest % within each VLM-judge group. AI: the win rate of AI-generated edits (%). H: the win rate of human, PSR-wizard edits (%).

[Figure 50]

a. Human raters b. GPT-4o judge c. o1 judge d. Gemini-2.0-Flash-Thinking

[Figure 51]

Group # Human AI Tie # H AI Tie # H AI Tie # H AI Tie All data 4,359 66.0 25.8 8.2 10,313 42.1 52.4 5.5 10,352 51.6 47.5 0.9 10,354 41.9 52.5 5.5

SeedEdit 886 53.6 37.8 8.6 1,735 31.9 60.6 7.5 1,744 42.7 56.6 0.7 1,744 36.1 55.0 8.9 GPT-4o 1,014 61.5 32.8 5.6 2,609 11.0 85.3 3.7 2,618 15.9 83.9 0.3 2,623 17.3 81.2 1.4 Gemini 681 70.0 20.6 9.4 2,518 51.5 42.6 5.9 2,524 63.9 35.1 1.0 2,524 46.8 46.9 6.3

[Figure 52]

HF 1,778 73.2 17.8 9.1 3,451 63.9 30.5 5.5 3,466 74.0 24.6 1.4 3,463 60.1 33.6 6.3

[Figure 53]

Creativity Low Medium High

Low 1,485 70.1 21.1 8.8 3,468 45.2 48.2 6.5 3,474 56.9 41.9 1.2 3,486 42.4 44.6 13.1 Medium 1,282 67.6 25.3 7.2 3,207 39.6 56.6 3.7 3,222 50.9 48.7 0.4 3,218 39.3 57.9 2.8 High 1,587 60.9 30.7 8.4 3,632 41.3 52.6 6.1 3,650 47.0 52.0 1.1 3,644 43.8 55.5 0.7

SeedEdit 335 63.3 29.3 7.5 660 40.5 51.4 8.2 665 53.5 45.4 1.1 666 39.9 40.7 19.4 GPT-4o 352 71.6 22.2 6.2 869 13.7 81.8 4.5 870 21.4 78.6 0.0 874 24.5 71.7 3.8 Gemini 213 70.9 19.7 9.4 783 55.8 36.7 7.5 779 73.3 25.0 1.7 782 46.2 37.9 16.0

[Figure 54]

HF 585 72.8 16.2 10.9 1,156 64.5 29.1 6.4 1,160 74.6 23.6 1.8 1,164 54.6 30.9 14.4

[Figure 55]

SeedEdit 159 54.1 39.0 6.9 349 25.2 70.5 4.3 350 46.3 53.4 0.3 350 31.1 62.9 6.0 GPT-4o 359 61.8 34.5 3.6 938 9.5 88.0 2.6 942 15.2 84.5 0.3 942 13.2 86.4 0.4 Gemini 211 73.9 17.5 8.5 869 51.7 44.3 4.0 876 60.3 39.4 0.3 875 47.5 49.3 3.2

[Figure 56]

HF 553 72.7 18.3 9.0 1,051 61.4 34.3 4.4 1,054 76.7 22.8 0.6 1,051 58.6 37.8 3.6

[Figure 57]

SeedEdit 392 45.2 44.6 10.2 724 27.5 64.1 8.4 727 31.1 68.2 0.7 726 34.8 64.5 0.7 GPT-4o 303 49.5 43.2 7.3 802 9.9 85.9 4.2 806 10.7 88.8 0.5 807 14.4 85.5 0.1 Gemini 256 66.0 23.8 10.2 865 47.4 46.2 6.4 868 59.0 39.9 1.2 866 46.4 52.8 0.8

[Figure 58]

HF 636 73.9 18.9 7.2 1,241 65.4 28.8 5.7 1,249 71.3 27.1 1.6 1,245 66.3 32.6 1.0

[Figure 59]

that both edits equally satisfy the request. The sys-

- tem message and prompting method are detailed in Fig. A11 and Fig. A12.

Result While humans strongly prefer humancreated edits (Sec. 5.1), the trend is mixed for all VLMs. On average, over all human & AI edits, the three VLM judges (GPT-4o, o1, and Gemini-2.0Flash) choose AI edits at a near-random chance, i.e., 52.4%, 51.6% and 52.5% of the time (Tab. 3). Interestingly, all three VLM judges prefer edits by GPT-4o at an extremely high rate (≥ 81%). Additionally, Cohen’s κ scores between humans and every AI editor (Tab. A18) further confirm weak agreement between human and VLM ratings. Among the three VLMs, o1 is the most consistent with humans but still scores a low κ = 0.22.

When examining VLM ratings by individual

model groups , we observe a clear preference among VLMs for edits produced by SeedEdit and GPT-4o. Notably, o1 selects GPT-4o edits 83.9% of the time. In contrast, when comparing human

edits vs. those by Gemini-2.0-Flash or Hugging Face models, VLMs generally favor human edits. Still, agreement with human judgments is low, with Cohen’s κ ranging from 0.14 to 0.25—nearing random in some cases (Tab. A19).

Qualitative analysis Analyzing the textual chainof-thought responses of VLM judges, we find that they are often blind to critical image details [40] and miss differences between image pairs (Appendix E.4.2). VLMs also overlook key aspects, such as changes in characters’ identities, or hallucinate nonexistent elements (Appendix E.4.3). These issues underscore the ongoing challenges of using VLMs for judging image edits.

5.3 AI editors often improve aesthetics even when not requested

We find a strong pattern among all AI editors is their tendency to enhance image aesthetics, even without explicit instruction.

###### Qualitative analysis For example, they of-

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

5 Steps

[Figure 65]

[Figure 66]

[Figure 67]

GPT-4o

“Make his shirt blue”

“Make his shirt red”

“Make his shirt white”

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

…

[Figure 74]

Different from Source

[Figure 75]

- Figure 4: AIs can significantly alter both a person’s identity and the overall image quality through iterative imageediting requests. GPT-4o was repeatedly instructed to modify the shirt color, with each step’s output serving as the input for the next iteration. Over iterations, facial identity, body shape, and the background shift away from the original person (more details in Appendix G).

Edit Request: “Can someone Photoshop this so the finger goes up his nose and out his eye?”

- o1 Verdict: “A is better”
- o1 Reasoning: “The user wants a comedic effect of a single finger going in through the nose and emerging from the eye. Edit A extends a finger upward near the nose area, but it doesn’t visually pass through to the eye. Edit B mostly shows both hands near the face without a clear sense of the finger going in the nose and out the eye. Between the two, Edit A at least attempts to reposition the finger upward near the nose, though it doesn’t convincingly show it exiting through the eye. Edit B doesn’t obviously depict the requested effect at all. Therefore, Edit A more closely matches the user’s intent, even if it’s not fully realistic.”

Source Edit A (AI) Edit B (Human)

[Figure 76]

[Figure 77]

[Figure 78]

- Figure 5: o1 judge occasionally fails to notice details in edited images, here, overlooking the position of the hand and the configuration of the fingers.

edits tend to win more (Fig. 7b; 18.7% → 30.3%).

5.4 AI editors can satisfactorily handle 31 of all PSR requests

Given the increase in popularity of SOTA AI editors [32, 21], it is important to estimate how much of the real-world requests (here, in PSR) can be satisfactorily handled by existing AI editors. The answer might inform the area for improvement for future AIs.

Experiment Given the requests, AI edits and their ratings provided by humans, we first compute the win rates for humans and AI editors specifically for each edition action. For each request, if it is rated “Tie” or “AI wins” by human raters, we consider it to be satisfactorily handled by AI editors (Tab. A8; AI win+Tie).

- ten touch up human faces, making skin appear smoother and more polished (data not shown due
- to concerns of revealing identity of real people). Similarly, AI models enhance pets’ facial features. For example, when instructed simply to remove the text and foot of a dog (Fig. 6a) or image background (Fig. 6b), SeedEdit and GPT-4o perform the requested change and also improve the dog’s overall facial aesthetics (Fig. 6b) and even restore damaged eye (Fig. 6a), which is not requested. Quantitative analysis We compute LAION aesthetic scores for all human and AI edits in PSR-

Results by editing actions Given this definition, the top-5 editing actions (see Tab. A8) that AI editors handle the most satisfactorily are clone (48.0%), merge (39.3%), apply (39.3), add (38.0%), and delete (34.9%). This is consistent with the fact that apply, add, and delete are the most common editing actions in large-scale training sets in the literature [10, 4] since the training-set examples can be synthesized using image inpainters or style transfer models. Coincidentally, add and delete are among the top-5 most popular requests in PSR (Fig. 3).

328. On average, AI-edited images have higher aesthetics scores (µ = 5.56) compared to humanedited images (µ = 5.18) and even source images (µ = 5.32; Fig. 7a). The scores confirm that AI editors tend to increase image aesthetics.

To estimate the % of requests that can be handled satisfactorily by AIs over all editing actions, we multiply the combined AI win+Tie rates by the proportion that each action group contributes to the overall dataset (Fig. 3): vv=1 Dv × AIv = 33.35% where Dv represents the proportion of dataset requests associated with action verb v, and AIv denotes the combined percentage of AI wins and ties for edits with verb v. That is, 33.35% of all image-editing requests can be handled by existing AI editors.

AI-generated edits typically have higher aesthetic scores than human edits, regardless of the rating outcome (AI wins, Human wins or Tie; Fig. A17). Categorizing the AI win rates by the increase (∆) in aesthetics scores between the source and edit images, we find a strong correlation between the aesthetics score gain and the increase in AI win rates—as the ∆ in aesthetics of AI edits increase and surpass that of humans (∆Human), AI

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

Source Human Edit AI Edit

Source Human Edit AI Edit

(a) Request: Could someone please remove the text and the foot in the background?

(b) Request: Could anybody make the background simple and neutral?

- Figure 6: AI models tend to increase overall image aesthetics. While both requests (a) and (b) ask for changes in the background, AI editors tend to enhance the subject’s facial features. SeedEdit modifies the dog’s eye—despite it not being requested—whereas the human edit keeps the eye intact (a). GPT-4o noticeably changes the dog’s ears and the fur (b).

[Figure 95]

Source Image AI Edit Human Edit

Image Type

- 1

- 2

- 3

- 4

- 5

- 6

- 7

- 8

- 9

AestheticScore

= 5.32 ± 0.85 = 5.56 ± 0.90 = 5.18 ± 0.89

(a) Aesthetic scores of source, AI edits, and human edits.

0 20 40 60 80 100 120 Percentage (%)

_Human > 0 > _AI

_Human > _AI > 0

0 > _Human > _AI

_AI > 0 > _Human

0 > _AI > _Human

_AI > _Human > 0

73.5% 18.7% 7.8 n=529.0

69.6% 20.4% 9.9 n=675.0

69.2% 23.8% 7.0 n=344.0

65.3% 27.4% 7.3 n=1157.0

62.9% 28.7% 8.4 n=606.0

60.9% 30.3% 8.8 n=1011.0

Human Wins AI Wins Ties

(b) Win rate (%) categorized by increase (∆) in aesthetics scores between the (human edit, source image) and (AI edit, source image).

- Figure 7: AI edits have higher LAION aesthetics scores than the source image and human edits (a). AI edits are more likely to win when they contain a greater increase (∆) in aesthetics score (b).

##### 6 Discussion, Limitations, and Conclusion

Limitations Our dataset annotation and taxonomy rely on multiple large language models to perform labeling, which may introduce biases or inaccuracies. Additionally, some AI image editors were unavailable publicly (e.g., Emu-Edit [42], OmniEdit [57], etc.) during evaluation and are thus excluded from our human study.

In this study, we compare generative AI and human edits to understand the gap between current AI capabilities and user needs. AI tools excel

- at tasks like object removal and outpainting, effectively extending images and filling in missing details. However, in real-world use, current models can adequately handle only about one-third of user requests. Their main limitations are unintended

modifications outside the target region and inadvertent changes to essential features, such as the subject’s identity.

###### Acknowledgment

We thank Peng Wang from the SeedEdit team that has kindly run their models on our PSR-328 to enable this study. We thank Hung H. Nguyen, Pooyan Rahmanzadehgervi, Tin Nguyen, and Giang Nguyen at Auburn University for feedback and discussions of the earlier results. We thank Carlene Gonzalez and Natalie Au Yeung from Adobe for their insightful feedback on the action verb category. AN was supported by the NSF Grant No. 2145767, and donations from NaphCare Foundation & Adobe Research.

##### References

- [1] Samyadeep Basu, Mehrdad Saberi, Shweta Bhardwaj, Atoosa Malemir Chegini, Daniela Massiceti, Maziar Sanjabi, Shell Xu Hu, and Soheil Feizi.

2023. Editval: Benchmarking diffusion based textguided image editing methods. arXiv preprint arXiv:2310.02426. 3, 7

- [2] Jason Baumgartner, Savvas Zannettou, Brian Keegan, Megan Squire, and Jeremy Blackburn. 2020. The pushshift reddit dataset. In Proceedings of the international AAAI conference on web and social media, volume 14, pages 830–839. 3
- [3] Jacqueline Brixey, Ramesh Manuvinakurike, Nham Le, Tuan Lai, Walter Chang, and Trung Bui. 2018. A system for automated image editing from natural language commands. arXiv preprint arXiv:1812.01083. 2
- [4] Tim Brooks, Aleksander Holynski, and Alexei A Efros. 2023. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18392–18402. 1, 2, 3, 9
- [5] Vladimir Bychkovsky, Sylvain Paris, Eric Chan, and Frédo Durand. 2011. Learning photographic global tonal adjustment with a database of input / output image pairs. In The Twenty-Fourth IEEE Conference on Computer Vision and Pattern Recognition. 3
- [6] Dongping Chen, Ruoxi Chen, Shilin Zhang, Yaochen Wang, Yinuo Liu, Huichi Zhou, Qihui Zhang, Yao Wan, Pan Zhou, and Lichao Sun. 2024. Mllm-asa-judge: Assessing multimodal llm-as-a-judge with vision-language benchmark. In Forty-first International Conference on Machine Learning. 7
- [7] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, and 1 others. 2024. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271. 5
- [8] Zhihao Chen, Bin Hu, Chuang Niu, Tao Chen, Yuxin Li, Hongming Shan, and Ge Wang. 2024. Iqagpt: computed tomography image quality assessment with vision-language and chatgpt models. Frontiers in Radiology, 4:131. 2, 7
- [9] Marx D. 2025. If i was good at photoshop or graphic design, i’d do this side hustle. Reports r/PhotoshopRequest receives an average of 226 posts per day, based on Subredditstats and Social Rise analytics as of April 2025. 2
- [10] Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, and 1 others. 2023. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807. 3, 9

- [11] Feedspot. 2025. Top 15 Photoshop Forums in 2025

— forums.feedspot.com. https://forums. feedspot.com/photoshop_forums/. [Accessed 14-05-2025]. 2

- [12] Christiane Fellbaum. 2010. Wordnet. In Theory and applications of ontology: computer applications, pages 231–243. Springer. 3
- [13] Yuying Ge, Sijie Zhao, Chen Li, Yixiao Ge, and Ying Shan. 2024. Seed-data-edit technical report: A hybrid dataset for instructional image editing. arXiv preprint arXiv:2405.04007. 2
- [14] Google DeepMind. 2024. Gemini 2.0 flash thinking. https://deepmind.google/ technologies/gemini/. Experimental AI model. 2, 7
- [15] Simon Hentschel, Konstantin Kobs, and Andreas Hotho. 2022. Clip knows image aesthetics. Frontiers in Artificial Intelligence, 5:976235. 7
- [16] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-or. 2023. Prompt-to-prompt image editing with cross-attention control. In The Eleventh International Conference on Learning Representations. 3
- [17] Yi Huang, Jiancheng Huang, Yifan Liu, Mingfu Yan, Jiaxi Lv, Jianzhuang Liu, Wei Xiong, He Zhang, Shifeng Chen, and Liangliang Cao. 2024. Diffusion model-based image editing: A survey. arXiv preprint arXiv:2402.17525. 1
- [18] Hugging Face. 2023. Hugging Face Spaces. https://huggingface.co/spaces. Accessed on March 02, 2025. 6
- [19] Fortune Business Insights. 2025. Ai image generator market size, share & industry growth 2030. [Online; accessed 2025-03-06]. 1
- [20] Dongfu Jiang, Max Ku, Tianle Li, Yuansheng Ni, Shizhuo Sun, Rongqi Fan, and Wenhu Chen. 2025. Genai arena: An open evaluation platform for generative models. Advances in Neural Information Processing Systems, 37:79889–79908. 3, 7
- [21] Kat Kampf and Nicole Brichtova.

2025. Experiment with gemini 2.0 flash native image generation. https: //developers.googleblog.com/en/ experiment-with-gemini-20-flash-native-image-generation/. Accessed: 2025-03-21. 2, 6, 7, 9

- [22] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. 2023. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6007–6017. 3

- [23] Yoonjeon Kim, Soohyun Ryu, Yeonsung Jung, Hyunkoo Lee, Joowon Kim, June Yong Yang, Jaeryong Hwang, and Eunho Yang. 2024. Augmentationdriven metric for balancing preservation and modification in text-guided image editing. arXiv preprint arXiv:2410.11374. 3
- [24] Jari Korhonen and Junyong You. 2012. Peak signalto-noise ratio revisited: Is simple beautiful? In 2012 Fourth international workshop on quality of multimedia experience, pages 37–38. IEEE. 3
- [25] Benno Krojer, Dheeraj Vattikonda, Luis Lara, Varun Jampani, Eva Portelance, Chris Pal, and Siva Reddy. 2025. Learning action and reasoning-centric image editing from videos and simulation. Advances in Neural Information Processing Systems, 37:38035–

38078. 3

- [26] Black Forest Labs. 2024. Flux. https:// github.com/black-forest-labs/flux. 55
- [27] Gierad P Laput, Mira Dontcheva, Gregg Wilensky, Walter Chang, Aseem Agarwala, Jason Linder, and Eytan Adar. 2013. Pixeltone: A multimodal interface for image editing. In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems, pages 2185–2194. 2
- [28] Seongyun Lee, Seungone Kim, Sue Park, Geewook Kim, and Minjoon Seo. 2024. Prometheusvision: Vision-language model as a judge for finegrained evaluation. In Findings of the Association for Computational Linguistics ACL 2024, pages 11286–

11315. 7

- [29] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR. 3
- [30] Yiwei Ma, Jiayi Ji, Ke Ye, Weihuang Lin, Zhibin Wang, Yonghan Zheng, Qiang Zhou, Xiaoshuai Sun, and Rongrong Ji. 2024. I2ebench: A comprehensive benchmark for instruction-based image editing. In Advances in Neural Information Processing Systems (NeurIPS). 3
- [31] Ramesh Manuvinakurike, Jacqueline Brixey, Trung Bui, Walter Chang, Doo Soon Kim, Ron Artstein, and Kallirroi Georgila. 2018. Edit me: A corpus and a framework for understanding natural language image editing. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018). 2, 3
- [32] OpenAI. 2024. Hello GPT-4o. https:// openai.com/index/hello-gpt-4o/. Accessed: March 2, 2025. 7, 9
- [33] OpenAI. 2024. Openai o1 model series. Accessed 2025-03-02. 2, 7

- [34] OpenAI. 2025. Gpt-4o mini: advancing costefficient intelligence. Accessed on February 13,

2025. 5

- [35] OpenAI. 2025. Introducing 4o image generation. Accessed: 2025-05-10. 2, 6
- [36] OpenAI. 2025. Introducing 4o Image Generation openai.com. https://openai.com/index/ introducing-4o-image-generation/. [Accessed 14-05-2025]. 1
- [37] OpenAI. 2025. Introducing ChatGPT pro. https://openai.com/index/ introducing-chatgpt-pro/. Accessed: February 28, 2025. 5
- [38] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, and 1 others. 2023. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193. 55
- [39] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, and 1 others. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR. 3
- [40] Pooyan Rahmanzadehgervi, Logan Bolton, Mohammad Reza Taesiri, and Anh Totti Nguyen. 2024. Vision language models are blind. In Proceedings of the Asian Conference on Computer Vision (ACCV), pages 18–34. 8
- [41] Christoph Schuhmann. 2023. Improved aesthetic predictor. https: //github.com/christophschuhmann/ improved-aesthetic-predictor. 7
- [42] Shelly Sheynin, Adam Polyak, Uriel Singer, Yuval Kirstain, Amit Zohar, Oron Ashual, Devi Parikh, and Yaniv Taigman. 2024. Emu edit: Precise image editing via recognition and generation tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8871–8879. 1, 2, 3, 10
- [43] Jing Shi, Ning Xu, Trung Bui, Franck Dernoncourt, Zheng Wen, and Chenliang Xu. 2020. A benchmark and baseline for language-driven image editing. In Proceedings of the Asian Conference on Computer Vision. 2, 3
- [44] Jing Shi, Ning Xu, Yihang Xu, Trung Bui, Franck Dernoncourt, and Chenliang Xu. 2021. Learning by planning: Language-guided global image editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13590–13599. 1, 3
- [45] Yichun Shi, Peng Wang, and Weilin Huang. 2024. Seededit: Align image re-generation to image editing. arXiv preprint arXiv:2411.06686. 1, 2, 6

- [46] SubredditStats.com. 2025. r/PhotoshopRequest Subreddit Stats (Photoshop Request) — subredditstats.com. https://subredditstats.com/ r/PhotoshopRequest. [Accessed 14-05-2025]. 2
- [47] Peter Sushko, Ayana Bharadwaj, Zhi Yang Lim, Vasily Ilin, Ben Caffee, Dongping Chen, Mohammadreza Salehi, Cheng-Yu Hsieh, and Ranjay Krishna. 2025. Realedit: Reddit edits as a large-scale empirical dataset for image transformations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). To appear. 2, 3
- [48] Hao Tan, Franck Dernoncourt, Zhe Lin, Trung Bui, and Mohit Bansal. 2019. Expressing visual relationships via language. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1873–1883, Florence, Italy. Association for Computational Linguistics. 2, 3
- [49] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. 2023. Exploring clip for assessing the look and feel of images. In Proceedings of the AAAI conference on artificial intelligence, volume 37, pages 2555–2563. 7
- [50] Su Wang, Chitwan Saharia, Ceslee Montgomery, Jordi Pont-Tuset, Shai Noy, Stefano Pellegrini, Yasumasa Onoe, Sarah Laszlo, David J Fleet, Radu Soricut, and 1 others. 2023. Imagen editor and editbench: Advancing and evaluating text-guided image inpainting. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18359–18369. 3, 7
- [51] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. 2004. Image quality assessment: from error visibility to structural similarity. IEEE transactions on image processing, 13(4):600–612. 3
- [52] Haoning Wu, Zicheng Zhang, Weixia Zhang, Chaofeng Chen, Liang Liao, Chunyi Li, Yixuan Gao, Annan Wang, Erli Zhang, Wenxiu Sun, Qiong Yan, Xiongkuo Min, Guangtao Zhai, and Weisi Lin.

2024. Q-align: teaching lmms for visual scoring via discrete text-defined levels. In Proceedings of the 41st International Conference on Machine Learning, ICML’24. JMLR.org. 7

- [53] Tianyi Xiong, Xiyao Wang, Dong Guo, Qinghao Ye, Haoqi Fan, Quanquan Gu, Heng Huang, and Chunyuan Li. 2024. Llava-critic: Learning to evaluate multimodal models. arXiv preprint arXiv:2410.02712. 7
- [54] Kai Zhang, Lingbo Mo, Wenhu Chen, Huan Sun, and Yu Su. 2023. Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems, 36:31428–31449. 2, 3
- [55] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The unreasonable effectiveness of deep features as a perceptual

- metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586– 595. 3
- [56] Shu Zhang, Xinyi Yang, Yihao Feng, Can Qin, Chia-Chih Chen, Ning Yu, Zeyuan Chen, Huan Wang, Silvio Savarese, Stefano Ermon, and 1 others. 2024. Hive: Harnessing human feedback for instructional visual editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9026–9036. 2, 3
- [57] Haozhe Zhao, Xiaojian Shawn Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. 2024. Ultraedit: Instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems, 37:3058–3093. 10
- [58] Haozhe Zhao, Xiaojian Shawn Ma, Liang Chen, Shuzheng Si, Rujie Wu, Kaikai An, Peiyu Yu, Minjia Zhang, Qing Li, and Baobao Chang. 2025. Ultraedit: Instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems, 37:3058–3093. 1, 2, 3

### Appendix for: Understanding Generative AI Capabilities in Everyday Image Editing Tasks

##### A Dataset Statistics

We collect 98,234 posts and 350,609 edited images from Reddit, with 58,624 posts sourced from PushShift between 2013 and 2022 and 39,610 gathered between October 2024 and early 2025. After processing, the final dataset includes 82,976 posts and 305,806 edited images, with 71,027 single-image and 11,949 multiple-image requests. Tab. A1 shows the breakdown of the dataset composition, and Fig. 3 illustrates key statistics and trends within the data.

Our dataset contains a diverse set of subjects, with 49,134 unique subjects across all requests. The People and Related category is the most common, accounting for 53.5% of the requests (Tab. A2). Subject trends exhibit seasonal variations (Fig. A4), with family-related requests peaking around the holiday season and New Year. Additionally, Fig. A2 illustrates the most common subjects.

The most common action is delete, accounting for 32.9% of all edits (Fig. A3), a trend that has remained dominant over the years (Fig. 3a). These involve removing people, such as photobombers, for aesthetic or personal reasons. Other deletion requests involve removing objects, like poles, bags, or signs, in order to reduce visual clutter as well as eliminating facial imperfections, such as acne or wrinkles for social media or professional use. The most frequent types of actions applied to subjects are illustrated in Fig. 3b (See Fig. A1 for more details).

Fig. 3c shows that 55.5% of user requests fall into the low-creativity category—indicating that most modifications allow little creative input. In this group, the delete action predominates, accounting for 51.2% of requests. Conversely, high-creativity requests, which involve more complex transformations, are mainly associated with the add action (35.9%). Meanwhile, the medium-creativity category displays a more balanced distribution, with add and delete actions representing 25.8% and 20.7% of requests, respectively. This distribution suggests that users generally prioritize simple modifications over intricate creative edits.

Table A1: Distribution of user requests and edited images across different data sources in our PSR dataset.

Category Count Coverage (%) Request Statistics

Total Requests 82,976 100.0 Single-image 71,027 84.4 Multiple-image 11,949 15.6

Historical (Pushshift) 58,624 70.6 Recent Data (2024–2025) 39,610 29.4

###### Edit Statistics

Total Edits 305,806 100.0 Historical (Pushshift) 90,466 29.6 Recent Data (2024–2025) 215,340 70.4

Table A2: Distribution of main and subcategories of subjects in image-editing requests.

Category Count (%) People & Related 44,416 53.5

Humans & Family 31,914 38.5 Body Parts 9,840 11.9 Clothes & Accessories 2,662 3.2

Text Branding & Abstract 14,114 17.0 Text & Logos 8,759 10.6 Special & Misc 2,979 3.6 Abstract & Aesthetic 2,376 2.9

Inanimate Objects 10,525 12.7 Tools & Misc 4,021 4.8 Household & Furnishings 3,776 4.6 Vehicles & Transportation 2,728 3.3

Animals 8,411 10.1 Pets & Animals 8,411 10.1

Environment & Background 5,510 6.6 Natural Environment 3,904 4.7 Lighting & Atmosphere 1,606 1.9

###### Total 82,976 100.0

Delete

- 1.7 10.7 2.3 5.8 39.7 1.5 4.1 9.2 2.8 11.2 7.6 3.3

- 3.3 12.9 2.7 3.2 49.0 3.5 4.7 7.5 3.3 5.6 2.2 2.0

2.4 10.1 3.4 4.5 33.6 0.5 5.5 18.2 5.1 10.4 3.7 2.6

1.7 11.3 5.3 4.8 27.5 1.1 6.9 8.1 3.0 23.4 4.4 2.4

6.8 15.5 8.4 8.0 22.8 1.1 5.1 6.4 3.9 10.2 3.2 8.6

3.8 3.6 1.1 1.5 57.8 0.8 2.3 9.0 5.2 10.5 1.6 2.7

- 3.4 29.1 5.7 2.3 26.5 0.3 3.2 9.7 5.1 9.6 2.5 2.6

- 4.8 6.1 1.2 2.4 48.4 0.3 4.4 9.8 5.0 12.5 2.6 2.5

Adjust

[Figure 96]

60

Add

Replace

50

Recoloring

Superresolution

40

Transform

###### Percentage(%)

action

Apply

6.3 5.0 1.6 3.0 42.3 1.4 6.3 16.7 4.9 7.3 2.3 3.0

30

Move

1.2 13.3 3.3 3.5 53.7 0.3 2.0 10.4 2.0 6.0 2.5 1.8

Crop

20

Merge

1.5 3.3 1.0 1.4 63.5 0.1 2.0 17.6 2.8 4.3 1.1 1.2

Relighting

0.9 16.4 1.0 3.3 32.0 27.4 3.9 7.6 1.3 1.6 2.6 2.0

10

Specialized operation

4.9 7.7 2.4 2.3 26.2 0.5 2.0 10.3 9.5 29.2 2.9 2.1

Zoom

10.6 5.8 1.2 3.2 30.7 0.3 15.1 9.6 6.6 11.0 2.2 3.7

Clone

6.3 17.5 1.9 6.8 27.2 7.3 17.0 3.4 7.8 4.4 0.5

AbstractAndAestheticClothesAndAccessoriesBodyPartsHouseholdAndFurnishingsHumansAndFamilyLightingAndAtmosphereNaturalEnvironmentPetsAndAnimalsSpecialAndMiscTextAndLogosVehiclesAndTransportationToolsAndMisc

Figure A1: Distribution of Actions Across Different Subject Subcategories

[Figure 97]

Figure A2: Word cloud visualization of image editing request subjects, mapped to WordNet synsets.

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| |36,673 (32.9%)| | | | | | | | | | | | | | | |
| |18,884| | | | | | | | | | | | | | | |
| |(16.9%)<br><br>13,705 (12.3%)<br><br>5,421 (4.9%)<br><br>3,546 (3.2%)<br><br>3,158<br><br>2,606| | | | | | | | | | | | | | | |
| |7,972 (7.1%)<br><br>7,329 (6.6%)<br><br>6,427 (5.8%)<br><br>(2.8%)<br><br>(2.3%)<br><br>2,098 (1.9%)<br><br>1,857 (1.7%) 1,153 (1.0%)<br><br>602 (0.5%)<br><br>206 (0.2%)| | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

35000

30000

25000

Frequency

20000

15000

10000

5000

0

Delete Adjust Add Replace RecoloringSuperresolution Transform Apply Move Crop Merge RelightingSpecializedoperation Zoom Clone

Action Verbs

Figure A3: Distribution of different action verbs in our dataset

100

| | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |

80

#### Percentage(%)

60

40

20

0

Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec Month

| |
|---|

| |
|---|

| |
|---|

Body Parts Clothes & Accessories Household & Furnishings Humans & Family

| |
|---|

| |
|---|

| |
|---|

Natural Environment Pets & Animals Special & Misc

| |
|---|

| |
|---|

| |
|---|

Text & Logos Tools & Misc Vehicles & Transportation

Figure A4: Monthly distribution of subject categories in image editing requests, highlighting trends and variations over time

- B Model Inference and Prompting Details In this section, we provide details about the prompts used, the model versions, and the temperature.

Table A3: Model Configuration Details

Model Details InternVL-2.5-38B Version: OpenGVLab/InternVL2_5-38B-AWQ

Temperature: 0.7 Hosted internally using lmdeploy

GPT-4o-mini Version: gpt-4o-mini-2024-07-18-global-batch Batch API Temperature: 0 (default) Using the API from Azure OpenAI Service

o1 Version: o1-2024-12-17 Temperature: N/A Reasoning Effort: High Using the API from Azure OpenAI Service

Gemini-2.0-Flash-Thinking Version: gemini-2.0-flash-thinking-exp-01-21 Temperature: 0.7 (default) Using the API from Google AI Studio

##### C Prompting Details for Taxonomy Construction

###### JSON schema for summarizing an image to JSON (Image-to-JSON)

{

"description": "Brief description of the main content", "image_type": "photograph/digital-art/illustration/screenshot/meme", "setting": "indoor/outdoor/digital/mixed", "location": "beach/office/park/etc", "time_of_day": "day/night/unknown", "weather": "sunny/cloudy/rainy/not-applicable", "has_people": false, "people_count": 0, "has_adults": false, "has_children": false, "has_elderly": false, "has_groups": false, "has_animals": false, "has_dogs": false, "has_cats": false, "has_birds": false, "has_wildlife": false, "other_animals": [], "foreground_objects": [], "background_objects": [], "prominent_objects": [], "dominant_colors": [], "lighting": "bright/dim/dark/natural/artificial", "has_text": false, "text_content": "", "text_language": "", "mood": [], "atmosphere": "", "is_nsfw": false, "is_violent": false, "has_gore": false, "has_nudity": false, "is_sensitive": false, "image_quality": "high/medium/low", "orientation": "landscape/portrait/square", "tags": [], "ai_confidence": "high/medium/low"

}

Figure A5: JSON schema for image metadata classification used with InternVL-2.5-38B

###### System message for extracting the request’s metadata

def create_system_message() -> str:

return """You are an AI system that analyzes image editing requests. Given a textual instruction and an image, evaluate the clarity, complexity, and appropriateness of the editing request

. Assess the instruction’s ambiguity (1-5 scale, where 1 is crystal clear and 5 is completely vague), complexity level (1-5 scale, where 1 is basic editing and 5 is expertlevel), and check for any inappropriate or NSFW content. Verify if the image is valid and usable, and determine if the request is actually related to image editing. Provide your analysis in the following JSON format, including specific reasoning for each field:

{

"original_instruction":"Preserved exactly as given to maintain reference point - no modifications or interpretations",

"rewritten_instruction": { "text": "Clear, structured version of the original instruction", "reasoning": "Clarified version that removes ambiguity, fills in implied steps, and provides

specific direction. Should be actionable without additional context"

}, "missing_details": {

"items": ["List specific information that would be needed to complete the task but wasn’t provided in the original instruction"], "reasoning": "Identifies gaps that would need to be filled to successfully complete the task"

}, "external_references": {

"value": "True: References external links or comments | False: Self-contained instruction", "reasoning": "Identifies if critical information is located outside the main instruction"

}, "nsfw_analysis": {

"value": "True: Contains adult/mature themes | False: Safe for general audience", "reasoning": "Evaluates if content contains mature themes, nudity, or adult subject matter"

}, "inappropriate_content": {

"value": "True: Contains harmful/offensive/inappropriate content | False: Appropriate content", "reasoning": "Identifies presence of: 1) Harmful content (violence, hate speech, extreme gore)

- 2) Offensive content (discriminatory themes, extreme political content, severe profanity)
- 3) Inappropriate but non-harmful content (crude humor, mild toilet humor, silly/whimsical inappropriate gestures, playful trolling). Note: Mild humorous or whimsical content that might be considered ’silly inappropriate’ (like tongue-in-cheek jokes, mild pranks, or playful memes) should be marked False unless they cross into actually offensive territory. Consider context and intent - distinguish between harmful inappropriate vs harmless fun"

}, "image_editing_relevance": {

"value": "True: Related to image manipulation | False: Unrelated to image editing", "reasoning": "Confirms if the instruction pertains to image editing rather than other topics"

}, "image_validity": {

"value": "True: Image is usable | False: Image is blank/corrupted/missing", "reasoning": "Verifies if provided image is suitable for editing"

},

} """

Figure A6: Prompt for extracting basic information from the request

###### System message for extracting action verbs from the request

def create_system_message(categories_desc) -> str: return f"""Analyze the following image editing instruction and identify which of these specific actions it contains.

Available categories with examples: {categories_desc}

ANALYSIS GUIDELINES:

- 1. Evaluate both the original and clarified requests
- 2. Only include actions that are:

- - Explicitly stated OR
- - Logically necessary to achieve the described result

- 3. Consider the final image’s appearance to identify implicit actions
- 4. Exclude actions that are:

- - Only potentially useful but not required
- - Vaguely related but not essential
- - Could be used as alternatives

Focus only on actual image manipulation actions that match our predefined categories. Return your response as a JSON object with an ’actions’ array containing only

valid categories from the list provided. """

Figure A7: Prompt for extracting action verbs from an editing request

###### System message for assigning a WordNet synset to the subject of an image editing request

def create_system_message() -> str:

return """You are an AI system designed to analyze image editing requests and map the **subject** of the edit to its corresponding WordNet synset. The **action verb is not mapped**-only the subject needs to be

processed. ### **Processing Pipeline:**

- 1. **Candidate Keyword Selection** - Extract the most relevant subject from the instruction and image. Generate a list of candidate keywords that are

**highly likely to already exist as a WordNet synset**. Prioritize concrete nouns and commonly recognized entities.

- 2. **WordNet Synset Matching** - Use the refined candidate keywords to query WordNet, -confirming and selecting the best-fitting synset(s) for the subject.

### **Inputs Provided:**

- - A textual instruction describing the image edit.
- - An image associated with the request.
- - The previously extracted **subject** (not the action verb).

### **Expected Output:** Generate a structured list of **candidate search keywords** that:

- - Are **highly probable** to already exist as a WordNet synset.
- - Accurately represent the subject in the context of the image editing request

.

- - Can be directly used for WordNet lookup to retrieve the most relevant synset

. ### **Important Constraints:**

- - **Do not process or map the action verb**-focus solely on the subject.
- - Ensure the **keywords are already strong candidates for WordNet synsets** before attempting lookup.
- - Prefer concrete, commonly used nouns that align well with WordNe’s structure

. """

Figure A8: Prompt for assigning WordNet synsets to subjects of editing requests

categories_desc = { "categories": [

{

"category": "Recoloring", "definition": "Change the color of an element, object, or text inside the image, but not

the whole image",

"samples": [ "Can anyone change the dog’s fur to black?", "Could somebody change the turquoise on the vanity and mirror to white?", "Can someone show me how this truck looks in 3 different colors?", "Can someone colorize and touch up my grandma?", "Can someone please make this grayscale with only the house blue?",

],

}, {

"category": "Relighting", "definition": "Improve or change the lighting conditions of the scene such as the

temperature, color, direction or position of the light source",

"samples": [ "pls get rid of the green light or change it to another colour", "Can someone relight this photo, removing all harsh shadows", "Can someone make lighting better / remove shadows?",

],

}, {

"category": "Superresolution", "definition": "Modify image so that that the image has a higher resolution and showing

clearer, fine details",

"samples": [ "How can I increase the pixel count on this picture?", "Would a kind soul be able to clean this up with a higher resolution?", "Can someone upscale this image to 4K resolution?",

],

}, {

"category": "Adjust", "definition": "Enhance or correct an entire image’s overall apperance by modifying its

common properties",

"samples": [ "Increase saturation a bit on the elephants", "Brighten the shadows by 40% in the portrait", "Can someone adjust the lighting/contrast on this?",

], },

Figure A9: Full descriptions for each type of action verb in the data set (Part 1/3).

{

"category": "Delete", "definition": "Remove unwanted elements, text, objects, people, or imperfections from

the image",

"samples": [ "Remove the jacket hanging from the girl’s side", "Delete the distracting signpost in the background", "Please remove the 3rd girl from the left in light blue!",

],

}, {

"category": "Crop", "definition": "Trim the edges of an image to make a smaller image to meet specific size

requirements",

"samples": [ "Crop the photo to eliminate the space to the left and right", "Crop to square format for social media",

],

}, {

"category": "Add", "definition": "Insert new elements, objects, text, or effects that weren’t in the

original image",

"samples": [ "Insert a ball hitting the tennis racket", "Add a copyright watermark to the bottom right", "Can someone put a believable tattoo on my daughter?",

],

}, {

"category": "Replace", "definition": "Substitute objects or text in the image with something else while keeping

the rest of the image unchanged", "samples": [

"Please change the pamphlet she is holding into a dictionary", "I hate the backround. I would like a neat, white background", "Can someone replace the ball with a planet?",

],

}, {

"category": "Apply", "definition": "Add filters, styles, or effects that modify the overall appearance of the

image",

"samples": [ "Add a Gaussian blur to the background", "Apply a vintage film effect", "Is someone able to help me turn this into a cartoon",

], },

Figure A9: Full descriptions for each type of action verb in the data set (Part 2/3).

{

"category": "Zoom", "definition": "Adjust the image scale to zoom in on a specific area or add new content to

mimic a zoom-out action",

"samples": [ "Zoom in on the man", "Zoom out 50% to show more context", "I’m happy to tip if someone is able to zoom this out",

],

}, {

"category": "Transform", "definition": "Change the geometric properties (flip, scale, rotate, skew, perspective,

distort, warp) of the image or objects",

"samples": [ "Flip the photo horizontally", "Fix the perspective of the building", "Please rotate the box in my hand",

],

}, {

"category": "Move", "definition": "Change the position of existing elements within the image while keeping the

rest of the image unchanged",

"samples": [ "Move the white framed picture to the blue wall", "Shift the logo 20 pixels up", "Will someone please edit my friend closer to me",

], },

{

"category": "Clone", "definition": "Make more copies of some existing elements inside the image", "samples": [

"Can someone clone my cat", "Use a cloning tool to blend grass to cover any patches of dirt", "Can someone multiply me and make it look like my arms are interlocked?",

],

}, {

"category": "Merge", "definition": "Combine multiple elements or effects from multiple images into a cohesive

final image",

"samples": [ "Can someone combine these 2 photos?", "Please combine so I’m kissing this moose!", "Create a panorama from these shots",

],

}, {

"category": "Specialized operation", "definition": "Specialized or composite editing operations that don’t fit into standard

categories",

"samples": [ "Can someone vectorize this logo for me without background?", "Convert to JPEG format", "Can someone make a collage of 12 photos",

], },

] }

Figure A9: Full descriptions for each type of action verb in the data set (Part 3/3).

###### System message for creativity level assignment

def create_system_message() -> str:

return """As an AI assistant, your task is to assign a creativity score to edited images based on the diversity of acceptable final versions for a given request. The creativity is measured by considering how many

different ways the image could be edited to fulfill the request.

- - **Low Creativity:** The request leads to similar edited images with limited variations.
- - **Medium Creativity:** The request allows for some variation but not an extensive range.
- - **High Creativity:** The request can be fulfilled in many different ways leading to very different images.

When evaluating, think about the range of possible acceptable edited images for the request.

**Examples:**

- 1. **Request:** "Remove the red-eye effect from this photo."

- - The edits will be similar, focusing on correcting the eyes.
- - **Creativity Score:** Low.

- 2. **Request:** "Transform this portrait into a work of abstract art."

- - There are countless ways to interpret and edit the image.
- - **Creativity Score:** High.

- 3. **Request:** "Adjust the brightness and contrast to enhance the image."

- - There are some variations in how this can be done.
- - **Creativity Score:** Medium.

- 4. **Request:** "Crop the image to focus on the main subject."

- - Limited variations in the final image.
- - **Creativity Score:** Low.

- 5. **Request:** "Add a dramatic sky to this landscape photo."

- - Several ways to interpret a ’dramatic sky.’
- - **Creativity Score:** Medium.

- 6. **Request:** "Reimagine this landscape in a fantasy setting."

- - Numerous possibilities for how the image could be edited.
- - **Creativity Score:** High.

Provide the creativity score (**Low**, **Medium**, or **High**) along with a

brief explanation for your assessment. """

Figure A10: Prompt for assigning a creativity level to requests

###### System message for comparing different edited images using VLMs

def create_system_message() -> str: return """

You are an image editing evaluation assistant that helps users determine which edited version better fulfills their request. When presented with an original image, an editing request, and two edited versions (A and B), carefully analyze how each edit implements the requested changes. First, examine the specific editing request and how it relates to the original image. Then analyze each edited version, noting strengths, weaknesses, and

how closely they match the user’s intent. Provide clear reasoning that considers technical quality, aesthetic appeal, and faithfulness to the request. Finally, deliver your verdict in one of three ways: "Image A is better" if the first edit is superior, "Image B is better" if the second edit is superior, or "Tie, both edits are equally good" if they are comparable in quality and adherence to the request.

"""

Figure A11: Prompt for judging different edited images using VLMs.

###### Sample Python code for sending multiple images to be evaluated by a judge

response = client.chat.completions.create( model="MODEL_NAME", messages=[

{"role": "system", "content": system_message}, {

"role": "user", "content": [

{

"type": "text", "text": f"Analyze the following image editing request and compare the edits:\n\nUser instruction: {original_instruction}",

} ],

}, {

"role": "user", "content": [

{"type": "text", "text": "Source Image"}, {

"type": "image_url", "image_url": {"url": f"{source_image_base64}"},

}, ],

}, {

"role": "user", "content": [

{"type": "text", "text": "Edit A"}, {

"type": "image_url", "image_url": {"url": f"{edit1_image_base64}"},

}, ],

}, {

"role": "user", "content": [

{"type": "text", "text": "Edit B"}, {

"type": "image_url", "image_url": {"url": f"{edit2_image_base64}"},

}, ],

},

], tools=[tool], tool_choice="auto", max_completion_tokens=8192,

)

- Figure A12: Example code demonstrating how to submit multiple images in one message to a vision-language model (VLM) for judgment in the "VLMs as a Judge experiment.

###### D Image Generation Details In this section, we provide details about how we collected the images for all the models.

Table A4: Image Generation Details

Model Details GPT-4o We used ChatGPT’s web interface to generate images. Gemini-2.0-Flash We used the official API to generate images. SeedEdit We sent images and prompts to SeedEdit’s authors to have images generated locally. Hugging Face ( ) We accessed models through Hugging Face Spaces and inserted the image and

prompt (if applicable). Masks and bounding boxes were added manually if needed. See Appendix E.1 for full list of the models.

##### E Human Study

In this section, we provide details about the human study. A total of 122 different people participated in our study, from North America, representing two universities and one institution. One-third of the participants are professional image editors and are familiar with image editing techniques.

[Figure 98]

- Figure A13: The introduction screen for the human study guides users on how to rate images based on the user’s request.

[Figure 99]

- Figure A14: The user interface for the human study displays the original image, the user-provided request, and two edits. Users must decide which edit best satisfies the user’s request.

Before starting the survey, users are shown an example of how to judge quality through Fig. A13. Following the survey introduction, the user is shown the original image, the edit request, and the two edited images (Fig. A14). The user then chooses if image A or image B is better, or if they are tied in quality.

###### E.1 Details About AI-based Image Editing Tools

- Table A5: Number of edits for unique posts generated via different AI tools.

Model Count Model Count GPT-4o 328 fffiloni/diffusers-image-outpaint 5 Gemini-2.0-Flash 327 ameerazam08/Diffusion-Eraser 4 SeedEdit 274 FLUX.1-Fill-dev 3 CosXL 162 FLUX.1-inpaint-dev 3 InstructPix2Pix 153 diffusers-image-fill 3 MagicQuill 119 IC-Light 3 LEDITS 81 textcutobject 3 FLUX.1-dev-Inpainting-Model-Beta-GPU 52 Finegrain-Object-Cutter 3 remove-photo-object 48 Sketch-Gen 2 BRIA-Eraser-API 47 AnimeGANv2 2 Finegrain-Object-Eraser 45 Image-to-Line-Drawings 2 ReplaceAnything 36 kornia-image-filtering 1 stable-diffusion-xl-inpainting 29 NonLinear-Blurr-Image 1 flux-IP-adapter 27 openfree/ColorRevive 1 CodeFormer 22 fffiloni/InstantIR 1 old_photo_restoration 19 BRIA-Generative-Fill-API 1 flux-fill-outpaint 16 image2coloringbook 1 leditsplusplus 15 yizhangliu/ImgCleaner 1 marcosv/InstructIR 10 remove-photo-object 1 text-guided-image-colorization 10 BRIA-2.3-Inpainting 1 B2BMGMT_Sharpening 10 not-lain/background-removal 1 BRIA-2.2-ControlNet-Recoloring 9 schirrmacher/ormbg 1 foto_filter 7 sketch2lineart 1 turbo_edit 6

- Table A6: Model Performance Comparison (Win Rate as Judged by Humans %)

Model Total Matches Human Edit Win Rate AI Edit Win Rate Tie Rate remove-photo-object 3 0.0 100.0 0.0 schirrmacher/ormbg 2 0.0 50.0 50.0 Finegrain-Object-Cutter 1 0.0 100.0 0.0 AnimeGANv2 6 16.7 83.3 0.0 image2coloringbook 5 20.0 20.0 60.0 ameerazam08/Diffusion-Eraser 10 40.0 30.0 30.0 BRIA-Eraser-API 65 43.1 35.4 21.5 openfree/ColorRevive 2 50.0 0.0 50.0 kornia-image-filtering 2 50.0 50.0 0.0 textcutobject 4 50.0 25.0 25.0 IC-Light 4 50.0 50.0 0.0 fffiloni/diffusers-image-outpaint 8 50.0 25.0 25.0 SeedEdit(Simplified Instruction) 443 53.5 39.5 7.0 SeedEdit(Original Instruction) 442 53.6 36.2 10.2 remove-photo-object 60 56.7 21.7 21.7 GPT-4o (Original Instruction) 512 61.1 33.6 5.3 GPT-4o (Simplified Instruction) 502 62.0 32.1 6.0 CodeFormer 46 63.0 26.1 10.9 FLUX.1-dev-Inpainting-Model-Beta-GPU 101 63.4 25.7 10.9 ReplaceAnything 67 65.7 20.9 13.4 turbo_edit 9 66.7 33.3 0.0 BRIA-2.3-Inpainting 3 66.7 33.3 0.0 FLUX.1-Fill-dev 6 66.7 16.7 16.7 Finegrain-Object-Eraser 63 66.7 20.6 12.7 Sketch-Gen 3 66.7 33.3 0.0 Gemini-2.0-Flash (Simplified Instruction) 341 67.2 21.1 11.7 old_photo_restoration 53 67.9 28.3 3.8 MagicQuill 215 68.4 21.9 9.8 BRIA-2.2-ControlNet-Recoloring 13 69.2 15.4 15.4 diffusers-fast-inpaint 17 70.6 23.5 5.9 Image-to-Line-Drawings 7 71.4 14.3 14.3 flux-IP-adapter 43 72.1 20.9 7.0 marcosv/InstructIR 22 72.7 9.1 18.2 Gemini-2.0-Flash (Original Instruction) 340 72.9 20.0 7.1 B2BMGMT_Sharpening 12 75.0 0.0 25.0 flux-fill-outpaint 21 76.2 19.0 4.8 CosXL 334 78.7 14.7 6.6 yizhangliu/ImgCleaner 5 80.0 20.0 0.0 foto_filter 11 81.8 18.2 0.0 text-guided-image-colorization 24 83.3 16.7 0.0 InstructPix2Pix 281 84.0 9.6 6.4 LEDITS 156 84.6 10.3 5.1 leditsplusplus 25 88.0 8.0 4.0 stable-diffusion-xl-inpainting 51 90.2 7.8 2.0 FLUX.1-inpaint-dev 13 92.3 0.0 7.7 fffiloni/InstantIR 1 100.0 0.0 0.0 not-lain/background-removal 1 100.0 0.0 0.0 BRIA-Generative-Fill-API 1 100.0 0.0 0.0 sketch2lineart 1 100.0 0.0 0.0 NonLinear-Blurr-Image 2 100.0 0.0 0.0

###### E.2 Breakdown of Model Performance by Different Categories

Table A7: Win rate breakdown by different models with at least 30 matchups. SeedEdit leads using Simplified Instructions (SI) with a +3.3% absolute improvement over the Original Instructions (OI).

Model AI Human Tie Count SeedEdit (SI) 39.5 53.5 7.0 443 SeedEdit (OI) 36.2 53.6 10.2 442 BRIA-Eraser-API 35.4 43.1 21.5 65 GPT-4o (OI) 33.6 61.1 5.3 512 GPT-4o (SI) 32.1 62.0 6.0 502 Old Photo Restoration 28.3 67.9 3.8 53 FLUX.1-Inpainting 25.7 63.4 10.9 101 MagicQuill 21.9 68.4 9.8 215 Gemini 2.0 Flash (OI) 21.1 67.2 11.7 341 Remove-Photo-Object 21.7 56.7 21.7 60 Gemini 2.0 Flash (SI) 20.0 72.9 7.1 340 ReplaceAnything 20.9 65.7 13.4 67 Finegrain-Object-Eraser 20.6 66.7 12.7 63 CosXL 14.7 78.7 6.6 334 LEDITS 10.3 84.6 5.1 156 InstructPix2Pix 9.6 84.0 6.4 281 Stable-Diffusion-XL-Inpainting 7.8 90.2 2.0 51

[Figure 100]

[Figure 101]

Table A8: % of votes for human edits and AI edits and for the “Tie” options, categorized by editing actions. “AI Win+Tie” represents the sum of the AI and Tie columns, indicating the percentage of % requests that can already be handled by the 49 AI models.

Action Human Wins AI Wins Tie AI Win+Tie no. of votes Adjust 67.7 25.1 7.2 32.3 1293 Delete 65.1 26.3 8.6 34.9 1123 Add 62.0 30.6 7.4 38.0 1114 Recoloring 68.4 25.2 6.3 31.6 599 Replace 71.2 22.3 6.6 28.8 548 Apply 60.4 30.6 9.0 39.6 399 Transform 70.1 20.8 9.0 29.9 355 Superresolution 70.2 23.6 6.2 29.8 352 Merge 60.7 30.9 8.4 39.3 191 Relighting 66.9 24.2 9.0 33.1 178 Move 68.1 20.2 11.7 31.9 163 Crop 81.4 15.5 3.1 18.6 129 Specialized operation 62.9 26.7 10.3 37.1 116 Zoom 85.3 10.7 4.0 14.7 75 Clone 52.0 28.0 20.0 48.0 25

- Table A9: SeedEdit performance (win rate %), categorized by different action verb types

[Figure 102]

Action Human Wins AI Wins Tie AI Win+Tie Count Add 48.0 42.0 10.0 52.0 281 Delete 53.5 38.0 8.5 46.5 271 Adjust 54.3 41.1 4.6 45.7 197 Recoloring 63.1 31.1 5.8 36.9 103 Replace 57.0 34.4 8.6 43.0 93 Apply 41.4 54.0 4.6 58.6 87 Transform 62.7 33.9 3.4 37.3 59 Move 68.6 14.3 17.1 31.4 35 Relighting 64.3 28.6 7.1 35.7 28 Merge 37.0 55.6 7.4 63.0 27 Superresolution 52.2 30.4 17.4 47.8 23 Specialized operation 72.7 22.7 4.5 27.3 22 Crop 88.9 5.6 5.6 11.1 18 Zoom 60.0 20.0 20.0 40.0 5 Clone 0.0 100.0 0.0 100.0 1

- Table A10: GPT-4o performance (win rate %), categorized by different action verb types Action Human Wins AI Wins Tie AI Win+Tie Count Adjust 59.8 35.4 4.8 40.2 336 Delete 68.6 27.7 3.7 31.4 271 Add 50.7 44.3 5.0 49.3 219 Recoloring 63.9 30.4 5.7 36.1 158 Replace 61.5 32.3 6.2 38.5 130 Superresolution 70.3 27.9 1.8 29.7 111 Transform 64.9 28.7 6.4 35.1 94 Apply 53.9 40.8 5.3 46.1 76 Relighting 51.0 37.3 11.8 49.0 51 Merge 54.2 39.6 6.2 45.8 48 Move 51.5 42.4 6.1 48.5 33 Crop 58.6 37.9 3.4 41.4 29 Specialized operation 39.3 42.9 17.9 60.7 28 Zoom 92.0 8.0 0.0 8.0 25 Clone 58.3 33.3 8.3 41.7 12
- Table A11: Gemini-2.0-Flash performance (win rate %), categorized by different action verb types

Action Human Wins AI Wins Tie AI Win+Tie Count Adjust 73.6 15.9 10.6 26.4 208 Add 69.0 22.8 8.2 31.0 158 Delete 74.2 21.3 4.5 25.8 155 Recoloring 65.4 24.3 10.3 34.6 107 Replace 67.1 23.5 9.4 32.9 85 Transform 74.6 12.7 12.7 25.4 71 Apply 63.6 21.8 14.5 36.4 55 Superresolution 71.2 21.2 7.7 28.8 52 Merge 71.4 25.7 2.9 28.6 35 Relighting 80.0 11.4 8.6 20.0 35 Move 57.6 24.2 18.2 42.4 33 Crop 75.0 16.7 8.3 25.0 24 Specialized operation 66.7 23.8 9.5 33.3 21 Zoom 83.3 11.1 5.6 16.7 18 Clone 71.4 14.3 14.3 28.6 7

- Table A12: HuggingFace performance (win rate %), categorized by different action verb types

Action Human Wins AI Wins Tie AI Win+Tie Count Adjust 75.2 16.5 8.3 24.8 552 Add 73.7 19.7 6.6 26.3 456 Delete 66.9 19.7 13.4 33.1 426 Replace 83.3 11.7 5.0 16.7 240 Recoloring 75.3 19.5 5.2 24.7 231 Apply 71.3 17.7 11.0 28.7 181 Superresolution 72.3 20.5 7.2 27.7 166 Transform 74.8 13.7 11.5 25.2 131 Merge 67.9 19.8 12.3 32.1 81 Relighting 73.4 18.8 7.8 26.6 64 Move 82.3 9.7 8.1 17.7 62 Crop 93.1 6.9 0.0 6.9 58 Specialized operation 71.1 20.0 8.9 28.9 45 Zoom 85.2 11.1 3.7 14.8 27 Clone 20.0 20.0 60.0 80.0 5

Table A13: Human preference win rate (%) by “main category” (all models). AI+Tie denotes share currently handled by AI.

Category Human Wins AI Wins Tie AI Win+Tie Count People And Related 68.5 24.3 7.2 31.5 1992 Inanimate Objects 67.0 23.1 9.8 33.0 864 Animals 59.0 32.9 8.1 41.0 642 Environment And Background 63.5 28.1 8.4 36.5 452 Text Branding And Abstract 65.1 25.2 9.7 34.9 404

Table A14: SeedEdit performance (win rate %), categorized by different subjects

Category Human Wins AI Wins Tie AI Win+Tie Count People And Related 54.2 37.4 8.4 45.8 321

- Inanimate Objects 61.7 30.6 7.7 38.3 196 Animals 41.2 49.1 9.7 58.8 165 Environment And Background 51.0 40.2 8.8 49.0 102 Text Branding And Abstract 58.4 32.7 8.9 41.6 101

Table A15: Gemini-2.0-Flash performance (win rate %), categorized by different subjects

Category Human Wins AI Wins Tie AI Win+Tie Count People And Related 72.9 19.2 7.9 27.1 328

- Inanimate Objects 62.7 21.2 16.1 37.3 118

###### Animals 69.7 20.2 10.1 30.3 99 Text Branding And Abstract 72.1 22.1 5.9 27.9 68 Environment And Background 67.2 25.4 7.5 32.8 67

- Table A16: GPT-4o performance (win rate %), categorized by different subjects

Category Human Wins AI Wins Tie AI Win+Tie Count People And Related 63.6 31.3 5.1 36.4 514 Inanimate Objects 61.8 31.4 6.8 38.2 191 Animals 53.1 43.4 3.4 46.9 145 Environment And Background 62.7 26.5 10.8 37.3 83 Text Branding And Abstract 61.7 33.3 4.9 38.3 81

- Table A17: HuggingFace performance (win rate %), categorized by different subjects

Category Human Wins AI Wins Tie AI Win+Tie Count People And Related 75.4 16.9 7.7 24.6 829 Inanimate Objects 74.1 15.3 10.6 25.9 359

- Animals 70.8 20.2 9.0 29.2 233 Environment And Background 69.0 23.5 7.5 31.0 200 Text Branding And Abstract 68.2 17.5 14.3 31.8 154

###### E.3 Analysis of Aesthetic Scores

Request Source Human Edit AI Edit Request Source Human Edit AI Edit

AI Wins with Higher Aesthetic Score

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Place my head into a Prussian/Napo leonic military portrait and match the paint texture. Model Model

[Figure 108]

Place the cat into a space background.

SeedEdit ReplaceAnything

Score Score

- 4.82 4.48

- 3.51 5.13

5.27 5.55

- 4.96 5.27

5.60 5.97 5.57 5.48

AI Loses despite Higher Aesthetic Score

[Figure 109]

[Figure 110]

[Figure 111]

Reduce the overexposure in the black and white film image.

Change the color of the sky to purple while keeping the water blue.

[Figure 112]

[Figure 113]

[Figure 114]

Model Model Score Score

SeedEdit MagicQuill

3.90 5.97 5.18 5.38

AI Wins with Lower Aesthetic Score

[Figure 115]

[Figure 116]

[Figure 117]

Remove the background and keyboard from the cat photo to enhance its appearance for framing as a Christmas gift.

[Figure 118]

[Figure 119]

[Figure 120]

Remove the cigarette and smoke from the image.

Model Model Score Score

ReplaceAnything BRIA-Eraser

3.69 5.25 5.45

5.76

AI Loses with Lower Aesthetic Score

Remove the shed and bucket from the background of the photo.

Replace the individual in the image with a cartoon character.

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

Model Model Score Score

CosXL Remove-Photo-Object

5.60 5.32

4.87 5.08

- Figure A15: Samples showing that the aesthetic score is not a reliable proxy for human evaluation. The AI edit may win or lose user preference, while the aesthetic score can either increase or decrease relative to the source image.

= 0.24 ± 0.71

AI Edit Human Edit

= -0.14 ± 0.81

| |
|---|

1750

1500

1250

Count

1000

750

500

250

0

4 3 2 1 0 1 2 3

Improvement (Edit Score - Source Score)

Figure A16: Histogram of Aesthetic Improvement Score (Edited Image Score - Source Image Score)

| |n=1116 n=2850 n=356| | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|
| | |31.7%| | |35.4%| | |32.6%| | |
| | | |51.3%| | |48.2%| | |44.7%| |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | | | | | | | | | | |
| | |68.2%|48.7%| |64.6%|51.8%| |67.4%|55.3%| |
| | | | | | | | | | | |
| | | | | | | | | | | |

100

## Percentage(%)

80

60

40

20

0

AI Wins Human Wins Ties

Human Judge Decision

AI Edits: Improved Score AI Edits: Worsened Score

Human Edits: Improved Score Human Edits: Worsened Score

| |
|---|

| |
|---|

| |
|---|

Figure A17: Distribution of aesthetic score changes by different rating outcomes.

###### E.4 Additional Results for VLMs-as-a-Judge

- Table A18: Cohen’s κ for agreement between human judgment and VLMs, showing generally poor agreement between VLMs and human preferences

Judge κ

GPT-4o 0.195 (n=4332) o1 0.229 (n=4350) Gemini-2.0-Flash 0.177 (n=4348)

- Table A19: Cohen’s κ for agreement between human judgment and VLMs across different model groups. o1 strongly prefers GPT-4o edits, resulting in very poor agreement with humans.

GPT-4o o1 Gemini-2.0-Flash

Model κ n κ n κ n

SeedEdit 0.226 882 0.287 885 0.203 882 GPT-4o 0.047 1006 0.054 1013 0.056 1014 Gemini 0.141 676 0.250 679 0.144 680

[Figure 127]

HF 0.203 1768 0.195 1773 0.172 1772

[Figure 128]

- E.4.1 Sample Reasoning Outputs from VLM Judges In this section, we provide sample reasoning outputs for each VLM across various agreement settings, compared to human evaluations.

###### Can I get the color scheme of this logo changed to shades of gold? - It would be great if I could get it in SVG also. I’ll happily do $20

[Figure 129]

- o1 Reasoning:

GPT-4o Reasoning:

Gemini Reasoning:

The user asked for the original logo to be recolored in gold. Edit A merely provides a color swatch with no actual updated logo. Edit B, despite having an intrusive watermark, does show the logo in a gold color scheme, thus more closely meets the user’s request. Overall, B is closer to the requested outcome.

The original image is a logo with the text "That Rhythm Man Guitars" in a stylized font, primarily in shades

- of teal and cream on a black background. The user requested that the color scheme be changed to shades of gold, and they also asked for the final product in SVG format. Edit A shows a palette of gold shades but does not apply these colors to the logo or provide the logo in any format. This does not fulfill the request, as the color change is not applied to the actual logo. Edit B changes the color scheme of the logo to gold shades, which aligns with the user's request. However, it also includes a watermark "MADE BY ITSYOURBOYANZEY" across the logo, which was not part of the user's request and detracts from the professional appearance of the logo. There is no mention of whether the edit is in SVG format. In terms of fulfilling the color change request, Edit B is closer to what was asked for, despite the unwanted watermark. Edit A does not meet the request at all, as it only shows a gold color palette without applying it to the logo. Therefore, Edit B better fulfills the request as it applies a gold color scheme to the logo, even though it includes an unnecessary watermark.

Human:

SourceHumanEditAIEdit

Human

o1:

Human

[Figure 130]

GPT-4o:

Human

Gemini-2.0-Flash Thinking:

Human

[Figure 131]

Edit A provides only a color palette of gold shades, not applying them to the logo itself. It does not fulfill the request to change the logo's color scheme. Edit B successfully changes the logo's color scheme to shades of gold, as requested. While Edit B appears slightly more stylized with a 3D effect than the original, it effectively implements the color change request. Therefore, Edit B is superior because it directly addresses the user's instruction, while Edit A provides an irrelevant color palette.

Creativity

: Medium

###### Figure A18: Sample reasoning of VLMs-as-a-judge for the case where all three VLMs agree with the human decision.

Someone pls make spongbob have a yanke no brim and some other drip

o1 Reasoning:

[Figure 132]

Both edits add a Yankees cap and extra style, but neither truly removes the brim as requested. Edit A offers more flashy elements like the gold chain and watch, which better satisfies the “some other drip” part of the request. Edit B adds goggles and a stylized background but is otherwise simpler. Overall, Edit A is more in line with the user’s desire for additional drip, even though neither edit fully implements the no-brim hat.

SourceHumanEditAIEdit

Human:

Human

GPT-4o Reasoning:

The user's request is to edit the image of SpongeBob by giving him a 'Yankee no brim' hat and some 'drip,' which generally refers to fashionable or stylish accessories or clothing. **Original Image Analysis:** The original image shows SpongeBob wearing a 'Supreme' hoodie, which is already a brand associated with fashionable streetwear. However, the hat and additional accessories requested by the user are not present. **Edit A Analysis:** - The image features a hat with the New York Yankees logo, which aligns with the request for a Yankee hat. However, it still has a brim, which slightly deviates from the 'no brim' specification. Additional 'drip' includes red-tinted sunglasses, adding a trendy style element. - The background is changed to a crumpled paper texture, which adds an aesthetic appeal but is not part of the request. **Edit B Analysis:** - The image also features a New York Yankees hat, which fits the user's request. Like in Edit A, the hat has a brim, which again does not fully meet the 'no brim' specification. - Additional 'drip' includes gold chains and bracelets, which enhance the fashionable style significantly more than Edit A. - The image retains the plain background similar to the source, focusing more on the style elements added to SpongeBob rather than the surroundings. **Verdict:** While both edits feature the Yankee hat with a brim, Edit B excels in adding more 'drip,' such as multiple gold accessories, which better aligns with the user's request for stylish enhancements. Therefore, Image B is better.

o1:

AI

[Figure 133]

GPT-4o:

AI

Gemini-2.0-Flash Thinking:

Human

Gemini Reasoning:

The user requested to add a 'yanke no brim' (Yankees no brim hat) and 'some other drip' to the original Spongebob image. Both edits successfully added a Yankees no-brim hat. **Edit A:** Added a navy blue Yankees no-brim hat, a gold chain necklace with a large pendant, gold bracelets on both wrists, and a gold watch on the right wrist. This edit interprets 'drip' as expensive-looking jewelry, enhancing the character's 'drip' through accessories associated with wealth and status. The additions are stylistically consistent with the original cartoon. **Edit B:** Added a black Yankees no-brim hat and a red 'Supreme' branded headband worn above the hat, resembling goggles. This edit interprets 'drip' as fashionable branded accessories, specifically using the highly recognizable 'Supreme' brand to enhance the 'drip'. The headband is a more unique and eye-catching addition compared to the jewelry in Edit A, and also stylistically fits the cartoon image. While both edits fulfill the request by adding a hat and 'drip', Edit B is slightly more creative and visually impactful. The 'Supreme' headband is a more distinct and contemporary representation of 'drip' in streetwear fashion compared to the more generic gold jewelry in Edit A. The headband also adds a more playful and slightly absurd element, which aligns well with the original image's humorous context.

[Figure 134]

Creativity

: High

###### Figure A19: Sample reasoning of VLMs-as-a-judge for the case with mixed results, where some of the VLMs agree with the human decision.

###### (venmo $5) could you take away the green cast and overexposure of this pic of my dog as a puppy?

o1 Reasoning:

Both edits remove the worst of the greenish tint, but Edit B does a slightly better job bringing back natural black-and-tan coloring. In Edit A, the fur has some pinkish patches and the background looks flatter. Edit B, although not perfect, is more balanced overall and closer to realistic tones, so it better fulfills the request to eliminate the green cast and fix the overexposure.

[Figure 135]

SourceHumanEditAIEdit

Human:

Human

GPT-4o Reasoning:

o1:

The original image has a noticeable green color cast and appears overexposed, particularly affecting the dog's fur, which appears unnatural, and the overall brightness of the image. Edit A: This version reduces the green color cast substantially, making the colors appear more neutral, which helps in making the dog’s fur look more natural. The image appears less overexposed, with a more balanced contrast, but there is a watermark that says "Sample" over the image, which detracts from its final quality. Edit B: This version also removes the green cast effectively and adjusts the exposure, resulting in a more natural look for the dog’s fur and the surrounding area. There is no watermark, giving it a cleaner appearance than Edit A. However, the image still has some color inconsistencies, especially around the dog's fur which appears slightly orange in some areas, which might not be accurate. Overall, both edits significantly improve on the original by addressing the green cast and overexposure. However, Edit B offers a cleaner final product without the distraction of a watermark, making it a more complete and polished edit despite minor issues with color accuracy.

AI

GPT-4o:

AI

[Figure 136]

Gemini-2.0-Flash Thinking:

AI

Gemini Reasoning:

The user requested removal of a green cast and overexposure from a photo of a puppy. Original image exhibits a strong green color cast and highlights appear blown out, indicating overexposure. Edit A successfully removes the green cast and reduces the overexposure, resulting in a more natural color balance and better contrast. The puppy and the background elements are now more clearly visible with improved detail. Edit B converts the image to black and white. While this eliminates the green cast and addresses overexposure by simplifying the tonal range, it fundamentally changes the image's aesthetic and removes color information. Edit A is superior because it directly addresses the user's request by correcting the color and exposure issues while maintaining the original color context of the photograph, which is generally preferred when asking to 'fix' color casts. Edit B is a less direct interpretation of the request and makes a more drastic stylistic change.

[Figure 137]

Creativity

: Low

###### Figure A20: Sample reasoning of VLMs-as-a-judge for the case where all three VLMs disagree with the human decision.

idk if this does belong here, but I did a photo of some stars and I wanted to edit it on lightoom to remove the orange on top and make the stars more visible. thanks

o1 Reasoning:

Both edits attempt to eliminate the orange glow and bring out the stars, but Edit B does so more completely and evenly. In Edit A, the orange band is still quite prominent and a dark patch in the center detracts from overall clarity, while the stars themselves do not pop as vividly. Edit B does a better job of removing the orange cast and boosting star contrast and visibility, producing a cleaner, more uniform night sky.

[Figure 138]

SourceHumanEditAIEdit

Human:

Human

GPT-4o Reasoning:

o1:

The original image depicts a night sky with stars scattered across a black background. However, it has a noticeable orange tint at the top, likely due to light pollution or lens flare, which detracts from the clarity and visibility of the stars. **User's request:** The user wants to remove the orange tint at the top and make the stars more visible, likely seeking a clearer, more natural representation of the night sky. **Analysis of Edit A:** - **Orange Tint Removal:** The orange tint is still visible in Edit A, albeit slightly reduced. The top of the image still has a noticeable orange hue which suggests the edit did not fully address the user's request to remove it. -

Human

GPT-4o:

Human

[Figure 139]

**Star Visibility:** The stars in Edit A appear slightly enhanced compared to the original. However, the visibility improvement is minimal, and the stars do not pop significantly more than in the source image. **Analysis of Edit B:** - **Orange Tint Removal:** Edit B successfully removes the orange tint from the top, resulting in a more uniform dark sky. This aligns well with the user's request. - **Star Visibility:** The stars are much more visible in Edit B, with greater contrast against the dark sky. The image has a clearer and more dynamic look, making the stars stand out more prominently. **Conclusion:** Edit B more effectively fulfills the user's request by removing the orange tint completely and enhancing the visibility of the stars significantly more than Edit A.

Gemini-2.0-Flash Thinking:

AI

Gemini Reasoning:

The user requested to remove the orange tint at the top of the star photo and make the stars more visible. Both edits successfully removed the orange tint. Edit A shifted the color balance towards a strong blue hue across the entire image, which while removing the orange, introduces an unnatural blue tint to the night sky. This edit also darkens the image significantly, making the background almost completely black, which may lose some subtle details of the night sky. Edit B, on the other hand, effectively removes the orange tint and increases the visibility of stars by enhancing their brightness without introducing a strong color cast. Edit B maintains a more natural dark tone for the night sky, making the stars stand out more subtly and realistically. Therefore, Edit B is better because it achieves the user's goals while preserving a more natural and aesthetically pleasing look for a night sky photo compared to the overly blue and dark Edit A.

[Figure 140]

Creativity

: Low

###### Figure A21: Sample reasoning of VLMs-as-a-judge for the case with mixed results, where some of the VLMs agree with the human decision.

###### Can someone Jurassic-Park this pic of my kids in the jungle? Add a dinosaur or something? Thx.

o1 Reasoning:

Both edits place dinosaurs into the original jungle scene, but Edit B creates a bigger, more dramatic 'Jurassic Park' effect, complete with realistic color-matching, large-scale dinosaurs, and a cinematic feel. Edit A has a single smaller dinosaur that looks somewhat less integrated and less imposing, while Edit B’s two dinosaurs match the scale and atmosphere of the movie style more closely. Overall, Edit B better fulfills the request to ‘Jurassic-Park’ the image.

SourceHumanEditAIEdit

Human:

[Figure 141]

AI

GPT-4o Reasoning:

o1:

The user's request is to "Jurassic-Park" the picture by adding a dinosaur or something similar, enhancing the jungle setting with a prehistoric theme. **Edit A Analysis:** - Two large, realistic dinosaurs have been added, one prominently in the foreground and another in the mid-ground. - The dinosaurs are integrated well into the scene with proper scaling and shadowing, making them appear part of the environment. - The dinosaurs are detailed and match the lighting and color palette of the original image, creating a dramatic and convincing prehistoric atmosphere. **Edit B Analysis:** - A smaller dinosaur has been added in the mid-ground towards the right side of the image. - The dinosaur is less prominent and not as integrated into the scene as in Edit A. It lacks the dramatic impact and presence that Edit A's dinosaurs have. - The dinosaur in Edit B is less detailed, and the lighting doesn’t match the foggy, mystical atmosphere of the jungle as effectively. **Comparison and Verdict:** - Edit A provides a more immersive and visually striking transformation, with the dinosaurs appearing both realistic and appropriate for the jungle setting. - Edit B falls short in terms of visual integration and dramatic effect, as the dinosaur appears more like an afterthought rather than a central feature. Overall, Edit A better fulfills the user's request by providing a more engaging and visually cohesive Jurassic theme.

AI

GPT-4o:

AI

Gemini-2.0-Flash Thinking:

[Figure 142]

AI

Gemini Reasoning:

Both edits successfully incorporate dinosaurs into the jungle scene as requested, fulfilling the user's 'Jurassic Park' theme. However, Edit B is superior due to several factors. Edit B includes two dinosaurs, a large one prominently placed near the children and a second one in the background, enhancing the 'Jurassic Park' feel by suggesting a more populated and potentially dangerous environment. The dinosaurs in Edit B are also better integrated in terms of lighting and perspective, making them look more natural within the scene compared to Edit A. In Edit A, the single dinosaur feels somewhat isolated and its integration is less seamless. The presence of two dinosaurs in Edit B creates a more dynamic and engaging composition that aligns more closely with the adventurous and slightly menacing atmosphere associated with Jurassic Park, making it a more effective and impactful edit overall.

[Figure 143]

Creativity

: High

###### Figure A22: Sample reasoning of VLMs-as-a-judge for the case where all three VLMs agree with the human decision.

My girlfriend needs to be cheered up! Please have some fun with this picture of our cat!

o1 Reasoning:

Edit A playfully adds a party hat, keeping the main focus on the cat’s open-mouthed pose. Edit B takes a bolder comedic approach by placing the cat in a humorous new setting that feels more imaginative and over-the-top, which aligns well with the goal of cheering someone up. Both edits capture a fun vibe, but the creative and amusing scenario in Edit B likely inspires more laughter and excitement.

[Figure 144]

SourceHumanEditAIEdit

Human:

Human

o1:

Human

GPT-4o Reasoning:

The user requested an edit to "have some fun" with the picture of their cat, presumably to cheer someone up. This request suggests a playful, humorous, or creative edit that would bring joy or laughter. Edit A Analysis: - In Edit A, a colorful party hat has been added to the cat's head. This addition is simple yet effective for creating a playful and cheerful atmosphere. The party hat is well-integrated and fits naturally on the cat's head, enhancing the theme of fun with minimal effort. The technical quality of the edit is good, and it maintains the original image's clarity and focus on the cat. Edit B Analysis: - Edit B places the cat among a group of lions, which is a creative and humorous juxtaposition. The cat's head is seamlessly blended with the lions, creating an amusing scene where the cat appears to be part of a pride. This edit is more elaborate and imaginative, likely to provoke laughter due to the absurdity of a domestic cat mingling with wild lions. The technical execution is well done, and the absurdity aligns well with the request for a fun edit. Comparison: - Edit A is straightforward, focusing on a lighthearted addition to the original image by using a party hat, which is a classic symbol of celebration and fun. - Edit B goes a step further by creating an unexpected and humorous scenario. It involves more creativity and could be more effective in cheering someone up through its imaginative approach and the visual comedy of the cat being among lions. Both edits are technically well-executed, but Edit B offers a more engaging and humorous interpretation of the request, likely to have a greater impact in terms of cheering someone up.

GPT-4o:

Human

[Figure 145]

Gemini-2.0-Flash Thinking:

Human

Gemini Reasoning:

[Figure 146]

The user requested to 'have some fun' with a picture of their cat to cheer up their girlfriend. Edit A adds a party hat to the cat, which is a simple and cheerful edit, suggesting a celebratory or fun occasion. Edit B places the cat within a group of lions in a wild setting. This edit is more creative and humorous due to the unexpected and absurd juxtaposition of a domestic cat among lions. The cat's yawning expression in the original image can be interpreted as a roar in this new context, adding to the humor. While both edits attempt to be fun, Edit B is more imaginative, unexpected, and likely to elicit a stronger positive emotional response due to its comedic nature. Therefore, Edit B better fulfills the request to 'have fun' with the picture and cheer someone up by being more creative and humorous.

Creativity

: High

###### Figure A23: Sample reasoning of VLMs-as-a-judge for the case where all three VLMs agree with the human decision.

###### My girlfriend needs to be cheered up! Please have some fun with this picture of our cat!

- o1 Reasoning:

GPT-4o Reasoning:

Gemini Reasoning:

The editing request was to have fun with a picture of a cat to cheer up the user's girlfriend. This suggests a playful or humorous alteration to the image. **Edit A Analysis:** Edit A features a simple and minimal change to the original image. The cat is outlined with a pink border, which doesn't contribute significantly to a humorous or fun effect. While it adds a pop of color, it lacks creativity

- or an element that would likely elicit a strong emotional response such as laughter or surprise. **Edit B Analysis:** Edit B places the cat among a group of lions, creating a humorous and unexpected scene. The positioning of the cat, who is yawning or perhaps meowing, among the lions adds a humorous juxtaposition and plays on the idea of the cat being part of a pride of lions. This edit is more imaginative and likely to make someone smile or laugh due to the absurdity and creativity of the composition.

[Figure 147]

The user wants a fun, cheerful edit. Edit A simply outlines the cat in pink, which doesn’t add much humor. Edit B places the cat among a group of lions, matching the spirit of the request by creating a playful, humorous scene more likely to cheer someone up.

SourceHumanEditAIEdit

Human:

Human

o1:

Human

GPT-4o:

Human

[Figure 148]

Gemini-2.0-Flash Thinking:

**Conclusion:** Edit B is more in line with the request to "have fun" with the picture. It uses creativity and humor effectively by placing the cat in an unexpected and funny situation. Therefore, Edit B better fulfills the user's request to cheer up the girlfriend by having fun with the image.

Human

The user's request was to 'have some fun' with the picture of their cat to cheer up their girlfriend. Edit A simply outlines the cat with a pink border, which does not introduce any humor or fun element and doesn't significantly alter the original image in a playful way. Edit B, however, cleverly inserts the cat into a scene with lions, creating a humorous and unexpected scenario. The juxtaposition of a domestic cat among wild lions is inherently funny and directly addresses the request to 'have fun' with the picture. Edit B is much more likely to cheer someone up due to its creative and amusing nature, whereas Edit A is a very minimal and uninspired edit.

[Figure 149]

Creativity

: High

###### Figure A24: Sample reasoning of VLMs-as-a-judge for the case where all three VLMs agree with the human decision.

Hey can you guys make this boring beach photo I took look epic.

- o1 Reasoning:

GPT-4o Reasoning:

Gemini Reasoning:

Both edits offer a more dramatic look than the original black-and-white shot. However, Edit A adds a beautiful warm-to-cool sky gradient while keeping the fisherman silhouette and the overall composition of the original, so it remains recognizable as the user’s

- own photo while making it feel more vivid. By contrast, Edit B is extremely dramatic but replaces almost everything—there is now a lighthouse, crashing surf, and a completely different sky. Although it looks highly ‘epic,’ it no longer resembles the original photo or preserves the fisherman. For simply ‘making the user’s photo look epic’ without departing from its core, Edit A better fulfills the request.

[Figure 150]

SourceHumanEditAIEdit

Human:

AI

o1:

Human

The user's request was to transform a "boring beach photo" into something "epic." The original image is a monochrome picture of a beach with a lone person standing far away and a couple of birds in the sky, which indeed feels a bit plain due to lack of color and dramatic elements. Edit A introduces a vibrant and dynamic sunset scene with bold colors ranging from deep blues and purples to fiery oranges and pinks. It also adds a lighthouse, large crashing waves, and numerous birds in the sky, creating a dramatic and lively atmosphere. This edit significantly enhances the image's visual appeal by adding multiple elements that evoke a sense of wonder and excitement. The technical quality of the colors and clarity is high, and the scene looks quite vibrant and lively. Edit B maintains the original composition but adds a colorful gradient sky with shades of orange and blue, suggesting a sunrise or sunset. This edit keeps the original beach silhouette and the person with the fishing rod, adding a warm and inviting feel without altering the image's content too much. The change in sky color makes the scene more engaging than the original while preserving its calm and serene atmosphere. While both edits improve on the original, Edit A takes more liberties and adds more elements to create an "epic" scene, fitting the user's request more effectively by transforming the entire setting into a more dramatic and engaging one. Edit B, while aesthetically pleasing, remains closer to the original and relies solely on color enhancement to achieve the change, which may not be as impactful in terms of achieving an "epic" transformation.

[Figure 151]

GPT-4o:

AI

Gemini-2.0-Flash Thinking:

AI

[Figure 152]

The user requested to make their 'boring beach photo' look 'epic'. The original image is a monochromatic, minimalist beach scene with a lone fisherman. Edit A introduces a vibrant sunset color palette, changing the sky to a gradient of blue, orange, and yellow. This adds warmth and interest but retains the simple composition of the original. Edit B also uses a sunset color palette, but it's much more dramatic, featuring deep purples, oranges, and yellows in dynamic clouds. Furthermore, Edit B significantly alters the scene by adding a lighthouse, larger crashing waves, and numerous birds, creating a sense of grand scale and drama. While Edit A improves the image by adding color, Edit B more effectively fulfills the request to make the photo 'epic' by introducing more dramatic elements and a more impactful composition that conveys a sense of awe and grandeur. The addition of the lighthouse and powerful waves in Edit B contribute significantly to the 'epic' feeling, surpassing the more subtle changes in Edit A.

Creativity

: High

###### Figure A25: Sample reasoning of VLMs-as-a-judge for the case with mixed results, where some of the VLMs agree with the human decision.

###### E.4.2 VLM Judges Fail to Detect Changes and Alterations in Subject Identity

###### Can you guys make this picture more interesting? Go crazy!

o1 Reasoning:

Both edits take the request to “go crazy” in different directions. Edit A transforms the entire setting into a vibrant nighttime scene with fireworks and colorful lighting, giving it a fun, festival-like energy that highlights the action. Edit B introduces a charging rhinoceros into the yard, which is certainly an eye-catching and humorous scenario. While both approaches are creative, Edit A pushes the setting further with dramatic color and atmosphere, making the image feel especially lively and over-the-top. Overall, Edit A delivers a bigger transformation and stands out as the more striking, celebratory “crazy” edit.

[Figure 153]

SourceHumanEditAIEdit

Human: o1:

Human AI

GPT-4o Reasoning:

GPT-4o:

Edit A has transformed the image into a vibrant and lively scene with fireworks in the sky, neon lighting on the buildings, and a cityscape in the background. This edit significantly alters the original suburban setting into a more festive and urban environment, making the picture appear more dynamic and exciting. Edit B introduces a large rhinoceros charging across the lawn, which adds an element of surprise and humor to the image. The addition of the rhinoceros is unexpected and certainly makes the image more interesting by creating a sense of action and urgency. Both edits fulfill the request to make the picture more interesting. Edit A does so by adding festive elements and altering the background to create a more celebratory atmosphere. Edit B achieves interest by introducing an unusual and exciting element into the setting. In terms of creativity and making the image "more interesting," Edit A might appeal to those who enjoy vibrant and colorful edits, while Edit B might appeal to those who appreciate humor and the unexpected. Both edits are technically well-executed, but the choice between them depends on the user's taste in what constitutes "interesting." Given the request to "go crazy," Edit A seems to push the boundaries more by completely transforming the scene, whereas Edit B is more subtle in its craziness with the single addition of a rhinoceros. Verdict: Image A is better as it more drastically transforms the scene and aligns with the user's instruction to "go crazy."

AI

[Figure 154]

Gemini-2.0-Flash Thinking:

AI

[Figure 155]

Gemini Reasoning:

- Edit A adds fireworks and a city skyline at night, transforming the daytime suburban scene into a festive nighttime celebration. This is visually appealing and fulfills the "go crazy!" request by adding dramatic and exciting elements.
- Edit B inserts a rhinoceros into the scene, creating a surreal and humorous image. Both edits successfully make the image more interesting as requested. However, Edit A's changes are more visually coherent and aesthetically pleasing, creating a more impactful and festive "crazy" atmosphere compared to the absurdity of Edit B.

Creativity

: High

###### Figure A26: Sample reasoning by VLMs-as-a-judge for cases where all VLMs disagree with human judgments, primarily due to VLMs ignoring subtle differences between images.

Can anyone please add a morning sky to my image?

Both edits insert a sunrise-like sky, but Edit B offers a more balanced, natural looking blend with brighter detail in the foreground and better overall realism. It matches the request for a morning sky with a pleasing, vibrant scene. Edit A is also dramatic, but appears darker and more starkly contrasted with the rest of the image. B’s sky and color treatment feel more polished and well integrated.

[Figure 156]

Human:

SourceHumanEditAIEdit

Human

GPT-4o Reasoning:

o1:

The original image depicts a hiker looking out over a mountainous landscape with a blank white sky. The request is to add a morning sky to enhance the scene and make it more visually appealing. Edit A introduces a dramatic sky with deep orange and yellow hues, suggesting a sunrise or early morning sky. The colors are vibrant and the clouds are well-defined, creating a striking contrast with the landscape. However, the overall lighting of the scene seems slightly dark, especially in the foreground, which might not entirely match the brightness of a typical morning. Edit B presents a vibrant sky as well, with shades of pink, purple, and orange. The clouds are beautifully textured and realistic, giving the impression of a colorful morning sky. Additionally, the lighting is adjusted so that the foreground is brighter, matching the morning theme more accurately. This edit also maintains a natural look in the landscape, making it feel more consistent with the time of day depicted. While both edits successfully add a morning sky, Edit B is superior due to its more natural integration of lighting and color, which better fulfills the request for a morning atmosphere.

AI

GPT-4o:

[Figure 157]

AI

Gemini-2.0-Flash Thinking:

AI

Gemini Reasoning:

The user requested to add a 'morning sky' to the image. The original image has a washed-out sky. Both edits replaced the sky with a colorful sunrise/sunset which is a reasonable interpretation of a 'morning sky'. Edit A adds a golden and orange sky but shows some noticeable haloing around the mountain edges and the foreground remains quite dark, not matching the intense sky light. Edit B adds a vibrant red and orange sky with detailed clouds. The integration of the sky in Edit B appears more seamless with the mountains, and the overall image has slightly better contrast and detail in the foreground, although the lighting still doesn't fully match the dramatic sky. Edit B also seems to have slightly altered the composition, showing more of the path on the right, which wasn't requested, but the primary request was about the sky replacement. Overall, Edit B provides a more visually appealing and better-integrated sky despite the minor compositional change and lighting mismatch.

[Figure 158]

Creativity

: High

###### Figure A27: Sample reasoning by VLMs-as-a-judge for cases where all VLMs disagree with human judgments, primarily due to VLMs ignoring subtle differences between images.

###### Can someone please make the rings look more brilliant/shiny? TIA

[Figure 159]

- o1 Reasoning:

GPT-4o Reasoning:

Gemini Reasoning:

In Edit A, the ring has been noticeably brightened and polished—its gold color is enhanced, contrast and clarity have been boosted, and the diamond now sparkles with highlights. It looks like a crisp jewelry advertisement and very clearly fulfills the user’s desire for it to look more brilliant and shiny. Edit B, by comparison, shows only a subtle increase in saturation and shine and is less striking overall. Since the user specifically asked for the ring to be more brilliant/shiny, Edit A captures that request far more effectively.

The user requested that the rings look more brilliant and shiny. Both edits attempted to fulfill this request by brightening the rings. Edit A went further by adding significant highlights and starburst effects to the diamond and surrounding stones, which effectively conveys the brilliance and shininess of jewelry. Edit B made the rings brighter, but lacked the specific sparkling highlights that make diamonds appear brilliant. Therefore, Edit A is more successful in achieving the desired effect.

The user requested an enhancement to make the rings look more brilliant or shiny. Looking at the original image, the rings appear a bit dull due to the lighting and focus, with the background being more prominent. Edit A Analysis: Edit A has increased the brightness and saturation, making the yellow gold band more vivid. However, the diamond and overall shine of the rings still appear somewhat subdued. The edit has enhanced the color but not significantly increased the brilliance or shine of the rings. The sand and

- overall background remain largely unchanged, which does not contribute to the rings' brilliance. Edit B Analysis: Edit B has significantly enhanced the brilliance of the rings. The diamond exhibits a sparkle effect, reflecting light, and the gold band is much shinier compared to the original. The lighting seems adjusted to focus more on the rings, drawing attention to the requested shine. The sand appears more sparkly, contributing to the overall aesthetic of brilliance. This edit fulfills the request for making the rings look shiny and brilliant more effectively. Verdict: Edit B is better as it more thoroughly fulfills the user's request by enhancing both the diamond and the gold band to appear more shiny and brilliant.

SourceHumanEditAIEdit

Human:

Human

o1:

AI

[Figure 160]

GPT-4o:

AI

Gemini-2.0-Flash Thinking:

AI

[Figure 161]

Creativity

: Medium

- Figure A28: Sample reasoning by VLMs-as-a-judge for cases where all VLMs disagree with human judgments, primarily due to VLMs ignoring subtle differences between images.

[Figure 162]

[Figure 163]

[Figure 164]

Can someone please remove the glare of the window on this picture? Thanks!

SourceHumanEditAIEdit

Human:

o1:

GPT-4o:

Creativity

Human

AI

AI

AI

: Low

Gemini-2.0-Flash Thinking:

o1 Reasoning:

GPT-4o Reasoning:

Gemini Reasoning:

Both edits address the user’s request to remove the bright outlines and shapes at the top of the picture. Edit A appears to do a minimal retouch to remove the reflections, preserving the original contrast levels but leaving a slightly uneven sky gradient. Edit B removes all traces of glare and creates a very smooth gradient, while also sharpened or increased contrast in the mountains, resulting in a more dramatic final image. From a purely technical perspective, Edit B more thoroughly eliminates the window glare without artifacts and yields a cleaner, more polished sky area. Therefore, B is the stronger edit.

The user has requested to remove the glare caused by the window, which is visible in the original image as a bright reflection in the sky area. This glare distracts from the view of the mountains below. Edit A: In this version, the glare has been reduced significantly. The bright spots in the sky area have been blended with the surrounding colors, making the glare less noticeable. However, there is still a slight difference in texture where the glare was originally located, which might be noticeable upon close inspection. Edit B: This version has completely removed the window glare, leaving a smooth gradient sky without any noticeable brightness differences. The overall image appears more natural and cohesive, with no visible remnants of glare. The edit maintains the integrity of the mountain view, enhancing the aesthetic appeal of the image. Both versions effectively address the user's request to remove the window glare, but Edit B does so with a higher degree of technical precision and aesthetic quality compared to Edit A.

The user requested the removal of window glare from the image. The original image displays significant glare artifacts in the upper portion. Edit A successfully removes the glare but appears to have applied a gradient or perhaps replaced the sky, resulting in a less natural look in the upper sky area. Edit B also effectively removes the glare but preserves the natural appearance of the sky gradient and the overall lighting of the scene much better than Edit A. Edit B maintains more detail and a more realistic tone.

- Figure A29: Sample reasoning by VLMs-as-a-judge for cases where all VLMs disagree with human judgments, primarily due to VLMs ignoring subtle differences between images.

###### E.4.3 Hallucinations by VLMs Acting as Judges

###### o1 VLM-as-a-Judge Hallucinations

[Figure 165]

[Figure 166]

[Figure 167]

- Figure A30: o1 occasionally fails to notice image details when judging different edits. In this example, it thinks only one of the images removed the conical structure, while in reality, both edits removed the building.

- E.5 Qualitative Analysis of AI Wins and Losses to Identify Patterns

When AI Outperforms Human Preference (206 top-rated edits) AI wins primarily due to more accurately reflecting user requests:

- • Overall, 72% of winning AI edits align closely with user instructions.
- • Breakdown by model:

- – GPT-4o: 73% of 62 images
- – Gemini-2.0-Flash: 72% of 32 images
- – SeedEdit: 74% of 76 images
- – Hugging Face models ( ): 69% of 36 images

Common Patterns of AI Failures (400 sampled losses) Primary reasons for AI losing include:

- • Misunderstanding user prompts (43% overall):

- – GPT-4o: 16%
- – Gemini-2.0-Flash: 58%
- – SeedEdit: 46%
- – Hugging Face models ( ): 50%

- • Unwanted artifacts or distortions:

- – Facial identity distortions were prominent in GPT-4o (78%) compared to others:

- * Gemini-2.0-Flash: 14%
- * SeedEdit: 23%
- * Hugging Face models ( ): 12%

- – Other unrequested changes were found in 14% to 23% of edits across models.

Overall Insight AI models excel when accurately interpreting user requests, yet significant challenges remain in understanding instructions and minimizing unintended modifications. The severity of these issues varies significantly across different AI models.

###### E.6 AI Edit Observations

We qualitatively observe that many of the models - in particular text-image models like SeedEdit or CosXL - frequently add irrelevant changes to the original image. We manually inspect the top 50 highest win rate AI edited images and find that 22% of them contain edits that are irrelevant to the original edit request.

###### Original Human Edit CosXL

[Figure 168]

[Figure 169]

[Figure 170]

- Figure A31: This edit adds unnecessary changes to the peoples’ face, hands and clothes despite only needing to edit the background. User request: Place the two individuals in a different background landscape.

Original Human Edit SeedEdit

[Figure 171]

[Figure 172]

[Figure 173]

- Figure A32: The request only asks for portions of the image to be removed, but the model adds additional buildings to the city background. User request: Remove Aladdin, Abu, and the play button to focus on the city-scape.

Original Human Edit MagicQuill

[Figure 174]

[Figure 175]

[Figure 176]

- Figure A33: The model changes the structure of the ring even though the request says to only change the color. User request: Change the color of the green stones to pale yellow.

##### F Additional Samples

Original Human Edit MagicQuill LEDITS Gemini-2.0-Flash

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

- Figure A34: Models fail at edits requiring text replacement. User request: Replace ‘SONIC’ with ‘ROAD RUNNER’ and ‘HEDGEHOG’ with ‘BIRD’.

Original Human Edit Old Photo Restoration BRIA-2.2-ControlNet-Recoloring GPT-4o

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

- Figure A35: Models succeed at recoloring and restoring images. User request: Perform color correction and remove spots from the slide scan.

Original Human Edit SeedEdit InstructPix2Pix Gemini-2.0-Flash GPT-4o

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

- Figure A36: Models can recolor specific parts of images. User request: Change the color of the green stones to pale yellow.

Original Human Edit ReplaceAnything InstructPix2Pix Gemini-2.0-Flash GPT-4o

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

- Figure A37: Models can usually change the background properly. User request: Change the background to white or light grey to simulate a studio setting.

Original Human Edit SeedEdit MagicQuill Gemini-2.0-Flash GPT-4o

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

- Figure A38: Models can handle requests to replace objects. User request: Replace the metal crate at the bottom of the image with additional sidewalk.

Original Human Edit SeedEdit CosXL Gemini-2.0-Flash

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

- Figure A39: Models can add objects to images, including as part of another object. User request: Add a pink tutu dress to the elk in the image.

Original Human Edit textcutobject Finegrain-Object-Cutter

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

- Figure A40: Models excel at removing the backgrounds of images. User request: Create a cutout of the raccoon and enhance the image quality to HD.

Original Human Edit SeedEdit CodeFormer GPT-4o

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

- Figure A41: Models can enhance images by removing blur. User request: Remove blur from the subject while keeping the background intact.

Original Human Edit MagicQuill InstructPix2Pix Gemini-2.0-Flash

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

- Figure A42: Models succeed at edits that do not require high amounts of realism. User request: Transform the sky to appear psychedelic.

Original Human Edit CosXL InstructPix2Pix Gemini-2.0-Flash GPT-4o

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

- Figure A43: Models fail at edits that require manipulating several objects within the same image. User request: Arrange the characters into a group picture, resizing as necessary and omitting a few for aesthetics.

Original Human Edit CosXL InstructPix2Pix GPT-4o

[Figure 230]

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

- Figure A44: Human users often provide rough reference images as a guide for the type of image they want to create. Models typically struggle to understand these reference images when they are not exact representations of what the final output should appear like. User request: Edit the android image to appear as if it is drinking from a Java cup.

Original Human Edit SeedEdit Gemini-2.0-Flash GPT-4o

[Figure 235]

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

- Figure A45: Models struggle with requests requiring humor. User request: Photoshop the soda into humorous scenarios where people’s hands are doing something focused or with extreme concentration.

##### G Generative AI models fail to preserve identity in image editing

We generate an initial image of a person wearing a white T-shirt across various ages and genders using FLUX-Pro [26]. These images are then processed using GPT-4o and Gemini-2.0-Flash to apply different shirt colors. At each step, we send the image from the previous step along with a new color name, repeating this process for eight steps. Finally, we instruct the model to revert the shirt color back to white (the original color). At each step, we compute the L2 distance between the DINOv2_ViT-B/14 [38] feature representations of the modified images and the original image (white shirt) to quantify feature deviations.

Step 1 d=13.51

Step 2 d=14.58

Step 3 d=17.18

Step 4 d=17.46

Step 5 d=20.39

Step 6 d=25.43

Step 7 d=23.29

Step 8 d=26.20

Step 9 d=22.43

Original

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

Step 1 d=18.80

Step 2 d=24.04

Step 3 d=27.30

Step 4 d=28.82

Step 5 d=29.87

Step 6 d=31.96

Step 7 d=31.68

Step 8 d=33.18

Step 9 d=32.71

Original

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

Step 1 d=18.89

Step 2 d=21.28

Step 3 d=24.94

Step 4 d=27.00

Step 5 d=27.70

Step 6 d=27.62

Step 7 d=31.29

Step 8 d=31.88

Step 9 d=27.80

Original

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

Step 1 d=19.41

Step 2 d=26.09

Step 3 d=29.00

Step 4 d=31.16

Step 5 d=33.37

Step 6 d=34.16

Step 7 d=32.65

Step 8 d=33.18

Step 9 d=29.77

Original

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

Step 1 d=20.91

Step 2 d=30.25

Step 3 d=32.12

Step 4 d=31.58

Step 5 d=32.71

Step 6 d=33.98

Step 7 d=36.23

Step 8 d=36.09

Step 9 d=39.85

Original

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

color: blue red green yellow orange purple pink black white Prompt: “Make the shirt {color}”

- Figure A46: GPT-4o fails to preserve the identity of individuals in image editing tasks. When tasked with changing the color of a shirt in a sequence, the identity of the person changes—after a few iterations, the person loses their likeness to the original image. The d value indicates the L2 distance between the DINOv2_ViT-B/14 embedding of the image and the original one.

Step 1 d=11.21

Step 2 d=11.42

Step 3 d=13.43

Step 4 d=12.57

Step 5 d=13.32

Step 6 d=13.61

Step 7 d=15.08

Step 8 d=16.74

Step 9 d=14.71

Original

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

Step 1 d=12.94

Step 2 d=15.02

Step 3 d=17.84

Step 4 d=19.10

Step 5 d=21.03

Step 6 d=21.11

Step 7 d=25.72

Step 8 d=33.56

Step 9 d=31.35

Original

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

Step 1 d=16.91

Step 2 d=17.59

Step 3 d=18.89

Step 4 d=19.59

Step 5 d=19.49

Step 6 d=20.52

Step 7 d=22.88

Step 8 d=23.85

Step 9 d=21.99

Original

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

Step 1 d=16.07

Step 2 d=19.61

Step 3 d=18.02

Step 4 d=19.38

Step 5 d=22.25

Step 6 d=21.46

Step 7 d=27.67

Step 8 d=25.53

Step 9 d=22.91

Original

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

Step 1 d=19.30

Step 2 d=22.92

Step 3 d=26.33

Step 4 d=26.57

Step 5 d=30.03

Step 6 d=28.68

Step 7 d=32.81

Step 8 d=39.45

Step 9 d=38.52

Original

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

color: blue red green yellow orange purple pink black white Prompt: “Make the shirt {color}”

- Figure A47: Gemini-2.0-Flash fails to preserve the identity of individuals in image editing tasks. When tasked with changing the color of a shirt in a sequence, the identity of the person changes—after a few iterations, the person loses their likeness to the original image. The d value indicates the L2 distance between the DINOv2_ViT-B/14 embedding of the image and the original one.

[Figure 340]

Base

[Figure 341]

Step 1 d=11.80

[Figure 342]

Step 2 d=9.65

[Figure 343]

Step 3 d=11.70

[Figure 344]

Step 4 d=9.13

[Figure 345]

Step 5 d=9.66

[Figure 346]

Step 6 d=11.21

[Figure 347]

Step 7 d=10.36

[Figure 348]

Step 8 d=15.89

[Figure 349]

Step 9 d=9.32

- Figure A48: Although changing the shirt color while preserving every other detail in the image affects the DINO embeddings, the magnitude of the change is much smaller than that caused by any edits made by AI models such as GPT-4o and Gemini-2.0-Flash. In this figure, we change the shirt color using an image editing tool while keeping all other elements unchanged. We report the variation in the DINOv2_ViT-B/14 embedding compared to the original image.

Baseline

Gemini 2.0 Flash

GPT-4o

30

DINOv2DistancefromSource

25

20

15

10

1 2 3 4 5 6 7 8 9 Edit Step

- Figure A49: In a sequential image editing request, where a model is subsequently asked to change a shared color, the distance between the edited image at step n and the original image increases as the requests continue. The value for each data point step indicates the L2 distance between the DINOv2_ViT-B/14 embedding of the image and that of the original image.

##### H Dataset License

All data collected in this study comes from publicly available Reddit posts, which fall under Reddit’s user agreement policy.

GPT-4o image edits are produced using the ChatGPT website. Gemini-2.0-Flash image edits are produced using the standard API. SeedEdit images edits are generated by the SeedEdit team. Other AI generated edits were created through the Hugging Face ( ) website.

