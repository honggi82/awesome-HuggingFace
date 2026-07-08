# arXiv:2411.16034v2[cs.CV]18Oct2025

## VisualLens: Personalization through Task-Agnostic Visual History

Wang Bill Zhu1,2,∗, Deqing Fu1,2,∗, Kai Sun1, Yi Lu1, Zhaojiang Lin1, Seungwhan Moon1, Kanika Narang1, Mustafa Canim1, Yue Liu1, Anuj Kumar1, Xin Luna Dong1

1Meta, 2University of Southern California

∗Work done at Meta

Existing recommendation systems either rely on user interaction logs, such as online shopping history for shopping recommendations, or focus on text signals. However, item-based histories are not always accessible, and are not generalizable for multimodal recommendation. We hypothesize that a user’s visual history — comprising images from daily life — can offer rich, task-agnostic insights into their interests and preferences, and thus be leveraged for effective personalization. To this end, we propose VisualLens, a novel framework that leverages multimodal large language models (MLLMs) to enable personalization using task-agnostic visual history. VisualLens extracts, filters, and refines a spectrum user profile from the visual history to support personalized recommendation. We created two new benchmarks, Google Review-V and Yelp-V, with task-agnostic visual histories, and show that VisualLens improves over state-of-the-art item-based multimodal recommendations by 5-10% on Hit@3, and outperforms GPT-4o by 2-5%. Further analysis shows that VisualLens is robust across varying history lengths and excels at adapting to both longer histories and unseen content categories.

Date: November 25, 2024 Correspondence: Wang Bill Zhu at wangzhu@usc.edu

1 Introduction

Imagine a personal assistant, similar to Vannevar Bush’s Memex (Bush, 1945), observing what you do in your daily life. With her keen insight, she can make informed guesses about what you may enjoy or find intriguing. When you ask for recommendations on anything from restaurants and activities to movies, books, and products, based on her in-depth understanding of you she will provide suggestions tailored specifically to your tastes.

While the concept is intuitive, a truly comprehensive personal assistant capable of making recommendations across all aspects of life has yet to be realized. Most existing multimodal recommendation systems remain domain-specific and rely heavily on item-based interaction histories (Tan and Jiang, 2023; Papadakis et al., 2023). For example, an e-commerce platform may suggest products based on past purchases but ignore dining habits or interests outside shopping.

This work explores how can we leverage such a user’s visual record to better understand individual preferences and enable more general, personalized recommendations. Achieving task-agnostic recommendations from visual history poses several challenges. First, visual histories are often diverse and noisy, containing images unrelated to any specific recommendation task, entities that fail to accurately represent user preference, or non-informative elements (e.g., background objects like trash cans). This creates a trade-off between preserving rich visual content and extracting clean, interpretable representations of user interests. Second, most current MLLMs can only process a limited number of images, requiring selective retrieval of relevant user profile information for each query. Third, existing benchmarks are inadequate for evaluating task-agnostic visual recommendation systems, highlighting the need for new, purpose-built evaluation datasets.

We propose a novel framework VisualLens, as a first step towards harnessing a user’s visual history for MLLM recommendation. VisualLens begins by extracting an offline spectrum user profile, compressing each image in the visual history into a triplet: (raw image, caption, aspect words). This representation spans a spectrum from

[Figure 1]

[Figure 2]

###### Recommendation Question

GoogleReview-V Yelp-V

Human Oracle

VisualLens

Recommend some museums for me in Los Angeles.

70

66.0

59.1 54.5

###### User Provided Photo History

48.5

- 1. Los Angeles County Museum of Art (Art Museum)
- 2. California Science Center (Local Science Museum)
- 3. …

[Figure 3]

[Figure 4]

[Figure 5]

45.0

37.3 38.9 34.1

Hit@3

[Figure 6]

[Figure 7]

[Figure 8]

0

UniMP GPT-4o VisualLens

- Figure 1 VisualLens leverages a user’s task-agnostic visual history to provide personalized recommendations. Our method outperforms GPT-4o by 1.6%∼4.6% on Hit@3.

rich but noisy content (raw image) to concise but clean semantic cues (aspect words). To improve the quality of aspect words, we employ an iterative refinement process that progressively enhances their alignment with user interests. Next, to efficiently incorporate multiple images during runtime recommendation, VisualLens retrieves the most relevant segments of the user profile based on the query context. These selected images are organized into a d×d visual grid, accompanied by their corresponding captions and aspect words, enabling the model to jointly process and reason over a compact yet informative representation of the user’s visual history. Finally, we train the system to perform both aspect refinement and recommendation question-answering (QA) in a unified model. This design not only reduces parameter overhead but also strengthens the model’s ability to interpret and utilize visual history for accurate, personalized recommendations.

To facilitate the evaluation, we created two new benchmarks, Google Review-V and Yelp-V, providing a foundation for personalization assessments. We leverage user-taken photos to address the challenge of history availability. Unlike extensive Memex video logs, these photos require less storage, often available in reviews and social media posts, and provide more insights into a user’s interests and preferences. Each benchmark includes a standard test set targeting generalization to new users, along with two additional test sets for transferring to longer histories and unseen categories.

Our experimental study shows promising recommendation quality of VisualLens. It achieved 82-91% Hit@10 on the Google Review-V and Yelp-V benchmarks, outperforming state-of-the-art (UniMP Wei et al. (2024a)) by ∼10%. Even comparing with GPT-4o, our 8B model improves Hit@3 by 1.6% and 4.6% respectively on the two benchmarks. Further analysis reveals that VisualLens excels in adapting to longer histories and unseen categories, while maintaining robustness with shorter histories.

### 2 Related Works

Recommendation system with large language models. Large Language Models (LLMs) have demonstrated strong potential in recommendation systems with their advanced language processing capabilities (Tan and Jiang, 2023). On item-based recommendation, studies such as LLM4RS (Dai et al., 2023), LLMRank (Hou et al., 2024), CLLM4Rec (Zhu et al., 2024b), P5 (Geng et al., 2022), and Sanner et al. (2023) explored various LLM prompt and bootstrapping strategies, showing competitive performance, especially in cold-start scenarios. Generative recommendation with open-domain prompts is explored by GenRec (Ji et al., 2023), though fine-tuning remains crucial. Fine-tuning approaches include personalized aspect extraction (Li et al., 2023c), and multitasking on LLaMa models (Yang et al., 2023). Retrieval-enhanced models, such as ReLLa, improve recommendation by retrieving relevant behavior sequences (Lin et al., 2024). Instruction-tuning and graph augmentation approaches are explored in InstructRec (Zhang et al., 2023), LLMRec (Wei et al., 2024b), and LKPNR (Hao et al., 2023). Recent advances in personalized conversation systems have utilized

- Table 1 Unlike representative related works, VisualLens is a novel framework leveraging multimodal LLM for recommendation with task-agnostic visual content. CTR: click-through rate.

Multimodal New Eval User History User Profile Objective

LLM4RS (Dai et al., 2023) ✗ ✗ Textual item features Text sequence Items ReLLa (Lin et al., 2024) ✗ ✗ Textual item features Top-k behaviors CTR ONCE (Liu et al., 2023) ✗ ✗ Task-agnostic text Content embedding Items

PC2L (Yan et al., 2023) ✓ ✓ Multimodal item features Selected images Explanation COURIER (Yang et al., 2024) ✓ ✗ Multimodal item features Joint embedding CTR UniMP (Wei et al., 2024a) ✓ ✗ Multimodal item features Joint embedding Multi-task

VisualLens ✓ ✓ Task-agnostic images Spectrum QA

task-agnostic conversation logs to provide personalized answers (Li et al., 2023a; Harper and Konstan, 19; Ni et al., 2019; Liu et al., 2023; Xu et al., 2024). However, these content-based recommendation approaches predominantly rely on textual data.

Multimodal recommendation systems. Multimodal recommendation systems (Wang et al., 2022; Yang et al., 2024; Sheng et al., 2024) leverage multiple data types, such as text and images, to improve recommendation relevance and personalization. Before the LLM era, Lee and Abu-El-Haija (2017) proposed systems for contentonly video recommendation using similarity learning. PC2L (Yan et al., 2023) develop an LLM model that provides multimodal explanations for recommendations. On modalities beyond image and text, MMRF (Zhou et al., 2024) built a joint recommendation system integrating comments with video items (Davidson et al., 2010; Du et al., 2020; Gao et al., 2017; Zheng et al., 2024). RecFormer (Li et al., 2023b) and UniSRec (Hou et al., 2022) convert images to short captions to utilize text-only models. The current state-of-the-art image-text recommendation, UniMP (Wei et al., 2024a), extended single-task multimodal personalization (He and McAuley, 2016; Tang et al., 2020; Wei et al., 2019, 2023) on multitask website-based shopping. However, most existing multimodal recommendation approaches rely on item-based user history, which is not always available. To address this limitation, we propose VisualLens, a novel framework that leverages MLLMs to enable personalization using task-agnostic visual history.

- 3 Multimodal Task-Agnostic Recommendation

Consider a recommendation QA task, where the user asks a recommendation question q, and the recommender answers q with a ranked list of candidate items. Good recommenders shall rank the items that the user is more likely to be interested in or willing to try early in the list.

In multimodal task-agnostic recommendation, the recommender is facilitated with a task-agnostic visual history Hu for each user u, which contains a series of photos, taken or posted by the user, not necessarily related to q. We state three assumptions to allow generalization. First, the photos may not be directly relevant to the question. Second, an image may not be associated with any candidate item, and even so, the candidate ID is not given. Third, a photo does not necessarily imply strong preferences. Figure 1 shows an example question, visual history, and candidates.

To simplify the problem, we assume a candidate retriever exists to retrieve all candidates that satisfy the user’s question. Each candidate s is represented with a (xs,Is) pair, where xs is the name and text descriptions, and Is is an optional image set for the item.

Traditional recommendation setting considers two more types of signals. The first is a task-specific set of items in the candidate set, which captures user interest or at least user history. The second is a set of user-specific attributes such as the user’s age, gender, and interests. This paper focuses on task-agnostic visual history and leaves integration of these traditional signals for future extensions.

|Captions<br><br>Image 1. A large, white building three-story building with a clock tower…<br>Image 2. the entrance to a museum, with a sign “MUSEUM ENTRANCE”…<br>Image 3. …<br>|
|---|

O F F L I N E

###### History augmentation Caption generation

User history

[Figure 9]

[Figure 10]

Frozen

|[Figure 11]|
|---|

[Figure 12]

Tunable

(LLaVA-v1.6)

[Figure 13]

Shared Weights

|Aspect words<br><br>Image 1. museum, arches, bench, dome, light, streetlight, night, balcony, ...<br>Image 2. museum, entrance, glass, black, iron, gate, reﬂection, plants, …<br>Image 3. …<br>|
|---|

###### Aspect generation

|[Figure 14]| |
|---|---|
| | |

(MiniCPM-v2.5)

[Figure 15]

R U N T I

|[Figure 16]|
|---|

Rank candidates with the output probability

History retrieval

###### Preference proﬁling

Images Captions Aspect words

(CLIP ViT-L/14@336px)

(Gridify / Concatenation)

Question

Relevant history

|<I3> <I1> <I2> …<br><br>Prediction|
|---|

[Figure 17]

What is the museum the user most likely to go to next?

###### Candidate matching

- M

<I1> Vulcan Park and Museum... <I2> Sloss Furnaces Visitor Center… <I3> Historic Cahaba Pumping Station… …

(MiniCPM-v2.5)

Candidates

- E

- Figure 2 VisualLens inference pipeline: the offline process augments images with captions and aspect words to generate a spectrum user profile; the runtime recommendation process retrieves relevant images, generate query-specific user profile accordingly, and then predict candidate preferences.

### 4 VisualLens Framework

VisualLens framework contains two parts in inference (Figure 2): offline user profile generation, and runtime recommendation. Offline user profile generation (§4.1) augments each image in the visual history with captions and aspect words, and builds a spectrum user profile. Runtime recommendation (§4.2) answers a recommendation query q in three steps. First, the history retrieval step retrieves only images relevant to q, since the visual history can be diverse and not all photos are relevant to every query. Second, the preference profiling step uses the retrieved images and their augmented captions and aspects to generate a query-specific profile of the user. Third, the candidate matching step matches the query-specific profile with each candidate, to generate the confidence score for each candidate for ranking. We discuss the joint training algorithm of VisualLens in §4.3.

- 4.1 Offline user profile generation

To build a user profile that retains rich visual content while offering clean, interpretable cues about user interests, we augment each image with a caption and a set of aspect words. Each image is thus represented as a spectrum triplet — (raw image, caption, aspect words) — ordered by decreasing information richness and increasing semantic clarity. Ablation results (Table 4) confirm that these augmentations improve recommendation performance.

Image encoding. Each image is encoded using the CLIP ViT-L/14@336px model (Radford et al., 2021), producing embeddings used for history retrieval at query time.

Caption generation. Captions are generated using a frozen LLaVA-v1.6-8B model (Liu et al., 2024), prompted to produce concise (≤30 words) and grounded descriptions to minimize hallucinations.

Aspect word generation. Aspect words are concise descriptors of key image attributes (e.g., dome, balcony). We prompt the model to list relevant terms without constraining the count, allowing flexibility based on image complexity.

All modules are plug-and-play and can be replaced with stronger alternatives. While LLaVA-v1.6 occasionally produces irrelevant or generic aspect words (e.g., blue, sky), we describe in Section 4.3.3 how joint finetuning improves their utility for recommendation tasks.

- Table 2 Dataset statistics of Google Review-V and Yelp-V.

Dataset Train Dev Test Categories Avg. # of images Avg. # of GT Avg. # of candidates

GR-V 15.69M 2K 200K 66 157.0 2.7 43.1 Yelp-V 4.12M 2K 100K 35 263.6 8.2 66.7

#### 4.2 Runtime recommendation

History retrieval. Given a query q and a user’s visual history Hu, VisualLens first retrieves images that are related to q, denoted by Iu,q. We choose up to w images to cap the number of images we process at runtime, and ensure that only the most contextually relevant images are retained for further processing, thereby reducing noise.

In general, we can use any image retrieval method such as DELG (Cao et al., 2020). Here, we present a method for categorical recommendations such as restaurants and museums, popular in recommendation tasks. For each category c, we randomly select a set of candidate items in the category and average the visual embeddings of their images as the image embedding of the category, denoted by vc. Specifically, the category embedding is calculated as vc = n1 nj=1 vc(j), where n is the number of candidates (see Appendix C for sensitivity), and vc(j) indicates the visual embedding of the j-th item image in category c. The retrieval step measures the cosine similarity between the visual embedding vi of each image i ∈ H in the user’s history and the image embedding vc of the relevant category c. We then select the top-w images based on the cosine similarity scores.

Preference profiling. Given a set of retrieved images Iu,q, VisualLens then generates user’s query-specific profile. A critical part of this step is image encoding. Even after retrieval, the number of images w is still large. Most MLLMs allow context windows of limited sizes, constraining the number of images we can process. For example, for an input image of resolution 896 × 896, the PaliGemma model would generate an embedding of up to 4,096 tokens. A typical LLM with a context window of 8,192 tokens can take at most 2 images.

We propose to group relevant images Iu,q into a d × d grid, where d2 = w, and treat all images in the grid as a single image. If we have retrieved fewer than w images, we pad with a black background. Let h be the maximum available resolution in a multimodal LLM. The gridify process G takes the d × d grid and generates an image of fixed size Rh×h×3. Additionally, we number each image from 1 to d2 to ensure the images are grounded to the corresponding caption and aspect words in the input to the candidate matching. We denote a user u’s profile on question q by (iu,q,xu,q), where iu,q denotes the gridified image, and xu,q denotes the concatenated captions and aspect words of relevant images.

Candidate matching. Finally, VisualLens takes the query-specific user profile (iu,q,xu,q) and a set of candidates, each represented by (xs,Is), and generates the matching score for each candidate, which will then be used for ranking. This is achieved by prompting the multimodal candidate predictor, where we packed the user profile and candidates to the prompt through the image channel and the text channel separately (see the prompt template in Appendix D).

#### 4.3 Iterative Refinement and Joint Training

##### VisualLens relies on LLMs for image encoding, caption generation, aspect word extraction, and profile–candidate matching. Current off-the-shelf models perform suboptimally for these tasks, so we apply continued pretraining and task-specific fine-tuning to enhance performance.

- 4.3.1 Multi-image caption pretraining

##### To facilitate the model to ground to each grid faithfully, we perform a LoRA continual pretraining on dense captions. We adopt the dense captioning dataset DOCCI dataset (Onoe et al., 2024), which contains over 15,000 images and their corresponding dense captions. Each time, we randomly sample w images I = {i1,··· ,iw} and their corresponding caption C = {x1,··· ,xw}, and then we construct a gridified input image G(I) and a target output text description T(C) = “Image 1: x1, ···, Image w: xw”. Then we LoRA finetune the

##### pretrained backbone model (e.g., MiniCPM-V2.5) on all image-caption pairs {G(I),T(C)} so that the model is able to process the gridified user history grid by grid. We then use the continual pretrained model as the starting point to apply joint training described in §4.3.3.

- 4.3.2 Iterative aspect word refinement

Unlike image captioning, aspect word generation is not a standard multimodal task and lacks the extensive pretraining data. Hence, with zero-shot prompting, the generated aspect words have a large quality gap across different images, and the extracted aspects may not indicate user preferences.

To finetune aspect word generation, we first generate the training data. For each image i, we start from an initial set of aspect words, denoted as Wi(0), which is generated by LLaVA-v1.6. In the jth round, we prompt a separate Llama-3.1 70B model with Wi(j−1) candidates, and ground truths, and ask it to select useful aspect words that are helpful in ground truth prediction, which constitute Wi(j). This refinement process continues for several rounds, and the iterations allow for converging extracted aspect words toward a more accurate and relevant subset. Empirically, we observe that the refinement converges after approximately 4 rounds, and denote the 4th refined aspect word set Wi(4) as Wi, which serves as the training target. The backbone model with parameter θ is finetuned to optimize the cross-entropy (CE) loss over all images I,

Lasp =

1 |I| i∈I

CE(Wi,pθ(xasp,i)), (1)

where xasp is the prompt for aspect words generation.

- 4.3.3 Joint training of aspect word generation and candidate matching

To take advantage of multitask training in multimodal recommendation (Wei et al., 2024a), we jointly train the aspect word generator and the candidate predictor on the backbone model. This joint training strategy allows the model to simultaneously learn to identify useful aspect words and make accurate predictions, thus improving overall performance.

The joint loss function balances aspect word generation and candidate matching with a weighting factor λ, where the candidate matching is optimized with binary cross-entropy (BCE) loss to handle multiple ground truth labels.

N

1 N

BCE(Sj,pθ(xpred,j,iu

j,qj)), (2)

Lpred =

j=1

Ljoint = Lasp + λLpred, (3)

where uj,qj,Sj is the user, the question, and the ground truth set of candidates of the j-th example. The text prompt xpred,j consists of the question qj, the text query-specific profiles xu

j,qj, and candidates. We LoRA finetune the model under the joint loss Ljoint.

### 5 Benchmarks and Experiments Setups

- 5.1 Benchmark creation

To the best of our knowledge, there is no existing benchmark (Harper and Konstan, 19; Ni et al., 2019; Wu et al., 2020; Wan and McAuley, 2018; Salemi et al., 2023) to evaluate personalization with task-agnostic visual history. We created two benchmarks, Google Review-V and Yelp-V, leveraging publicly available data from Google Local Review (Li et al., 2022) and Yelp (Asghar, 2016). Table 2 gives the benchmark statistics. The ratio of an average number of candidates and that of ground truths is 16:1 for Google Review-V and 8:1 for Yelp-V. By default, the train, dev, and test data have disjoint users. We discuss other splitting in Table 5. We list the details on dataset creation in Appendix A.

- Table 3 Hit rates and MRRs of VisualLens vs. multiple baselines on Google Review-V and Yelp-V. The result shows (a) VisualLens outperforms other baselines, though has a gap with the human oracle; (b) model size greatly affects the performance; (c) simply rank by rating is a worse design than the random baseline. Due to the large test set size (200K), an MRR difference greater than 0.4 yields a p-value < 0.04.

Google Review-V Yelp-V

Modality Size Hit@1 Hit@3 Hit@10 MRR Hit@1 Hit@3 Hit@10 MRR Naive baselines

Random - - 7.6 21.0 55.0 21.2 13.0 33.6 72.7 30.0 Rank by rating - - 3.9 15.8 55.5 17.7 8.7 28.0 72.3 25.9

Fine-tuned models

UniMP (Wei et al., 2024a) T + I 3B 13.8 34.1 73.0 30.5 22.4 48.5 85.0 38.3 Llama-3.1-8B-Instruct (Meta, 2024) T 8B 15.8 36.3 77.2 32.9 24.1 52.2 88.5 39.6 PaliGemma (Beyer et al., 2024) T + I 3B 13.0 32.0 70.1 28.4 20.8 46.7 82.0 37.5 MiniCPM-V2.5 (Yao et al., 2024) T + I 8B 16.1 36.4 78.4 33.2 24.8 53.0 89.3 40.3

Direct inference

Llama-3.1-70B-Instruct (Meta, 2024) T 70B 16.2 35.9 75.7 33.1 25.2 53.2 88.5 40.6 GPT-4o (OpenAI, 2024) T + I - 17.1 37.3 80.1 34.3 26.1 54.5 90.5 41.7

Our method

VisualLens (PaliGemma) T + I 3B 16.7 36.3 77.1 33.5 27.8 58.8 90.4 44.3 VisualLens (MiniCPM-V2.5) T + I 8B 18.5 38.9 82.3 35.4 28.3 59.1 91.0 44.9

Human annotations - - 22.0 45.0 - - 36.0 66.0 - -

- 5.2 Evaluation measures We use two metrics to evaluate recommendation quality. Hit@k. Hit@k = N1 Ni=1 [rank(ri) ≤ k] checks if any relevant item is within the top-k ranked results, where

N is the number of examples, and rank(ri) is the rank of the first relevant item. We check Hit@3 (e.g., voice recommendations) and Hit@10 ( e.g., on-screen recommendations).

MeanReciprocalRank(MRR). MRR = N1 Ni=1 rank1(r

i) measures the ranking quality by averaging the reciprocal ranks of the first relevant item for each example. The MRR ranges from 1/S to 1, where S is the number of candidates.

- 5.3 Implementation and baselines

We ran VisualLens with two backbone models, a smaller 3B model PaliGemma and a larger 8B model MiniCPM-V2.5. For optimal performance, we selected w = 64 (112x112 each sub-image for PaliGemma, best in practice), corresponding to an 8 × 8 image grid, a candidate count of n = 10k, and a loss weighting factor of λ = 2 (Eqn. 3). We compared VisualLens with solutions below.

- • Baselines: Randomly rank or select top-k ratings.
- • Fine-tuned models: We fine-tuned three state-of-the-art solutions: multimodal personalization model UniMP (Wei et al., 2024a) (RedPajama 3B), with the adaptation to replace item images and attributes with image tokens in the visual history; Llama-3.1-8B-Instruct (Meta, 2024) with text-only user preference profiles; PaliGemma 3B and MiniCPM-V2.5 8B (Yao et al., 2024) with image profiles.
- • Direct inference: We compared with two out-of-box models: Llama-3.1-70B-Instruct (Meta, 2024) with text-only preference profiles; GPT-4o (OpenAI, 2024) with multimodal profiles. To control the API cost, we subsample 1k instances from the test sets.
- • Human annotation: Finally, we subsampled 50 examples from the test set for human annotation.

- Table 4 Ablation study on PaliGemma. Different components of VisualLens model: joint training (Joint), iterative refinement (Iter), aspect words (Asp.), captions (Cap.), image embedding (Img.), and relevant image retrieval (Ret.). An MRR difference greater than 0.4 yields a p-value < 0.04.

# Representation Ret. Training Google Review-V Yelp-V Asp. Cap. Img. Iter. Joint Hit@1 Hit@3 Hit@10 MRR Hit@1 Hit@3 Hit@10 MRR

- 1 ✓ ✓ ✓ ✓ ✓ ✓ 16.7 36.3 77.1 33.5 27.8 58.8 90.4 44.3
- 2 ✓ ✓ ✓ ✓ ✓ 16.1 35.8 76.2 33.0 27.2 57.9 88.9 43.3
- 3 ✓ ✓ ✓ ✓ 15.7 35.2 75.4 32.5 26.9 57.5 88.2 42.9

- 4 ✓ ✓ ✓ 15.2 34.7 74.2 31.9 25.8 55.3 86.1 41.2
- 5 ✓ ✓ ✓ 14.8 33.9 73.0 31.2 25.0 53.9 84.9 40.4
- 6 ✓ ✓ 13.5 32.5 71.9 29.6 22.0 48.2 83.6 38.8

- 7 ✓ ✓ ✓ 11.5 27.9 67.3 25.9 20.1 45.7 81.7 36.8

| |Google Review-V<br><br>Yelp-V| | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |
| | | | | | | | | | | | |

0.8

0.7

0.6

0.5

MRR

0.4

0.3

0.2

0.1

0.0

10 20 30 40 50 60 70 80 90 100

# of Candidates

| |Google Review-V<br><br>Yelp-V| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | |

0.5

0.4

0.3

MRR

0.2

0.1

0.0

10 20 30 40 50 100 150 200 250 300 400 500 600 700 800 900

# of Images

(a)

(b)

- Figure 3 (a) MRR distribution over number of candidates, (b) MRR distribution over number of images. Both are on the User ID test set. We find (1) MRR converges when number of candidates exceeds 50; (2) MRR increases and flattens after reaching ∼100 images.

- 6 Results and Analysis We conducted experiments to answer four questions:

- Q1: Can we effectively leverage a user’s visual history to improve personalization?
- Q2: How does each element of VisualLens contribute to the recommendation quality?
- Q3: Can VisualLens transfer across users, unseen categories, longer history, and new benchmark?
- Q4: What is the robustness of VisualLens?

- 6.1 Recommendation effectiveness (Q1)

VisualLens significantly outperforms baselines. Our first observation from Table 3 is that all recommendation models that leverage the visual history significantly outperform baseline solutions on all metrics. In particular, the best version of VisualLens improves Hit@3 over Random by 18% on Google Review-V and by 26% on Yelp-V, and even more over Rank by rating (apparently, overall ratings do not cater to specific users). There is still a gap between VisualLens and human annotations, but comparing w. Random, it fills ∼75% of the gaps for hit@3 on both datasets.

VisualLens outperforms state-of-the-art solutions. VisualLens outperforms UniMP with the same number of trainable parameters. With the 8B MiniCPM-V2.5 backbone, VisualLens outperforms MiniCPM-V2.5 itself by 2.5% on Hit@3 on Google Review-V, and by 6% on Yelp-V, and we observe a similar trend for the 3B models. Even comparing with significantly larger models, including Llama-3.1-70B-Instruct and GPT-4o

- Table 5 Transferability: MRR of VisualLens models to different test setups. LongHis: train until a certain timestamp and test afterwards. Category: held-out a set of categories for testing per user. Use ID: train and test set share no common user ID.

Google Review-V Yelp-V LongHis Category User ID LongHis Category User ID

VisualLens (PaliGemma) 35.9 34.9 33.5 46.6 45.2 44.3 VisualLens (MiniCPM-V2.5) 38.0 37.1 35.4 47.2 45.5 44.9

Google Review-V Categories

Yelp-V Categories

restaurant bar tourist attraction park cafe grill shopping mall bakery supermarket

steak house station hotel brewery market museum deli theater takeout

pub area hall historical landmark club brewpub salon beach garden

gastropub golf course atm nature preserve church winery car wash campground lodge

restaurant bars & pubs burgers & sandwiches breakfast & brunch coffee & tea seafood pizza desserts salad bakeries

steakhouses yogurt & cream & cupcakes barbeque grocery soup bubble tea buffets chocolatiers hotel

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

| |
|---|

44

- 104

- 105

- 106

- 107

50

42

- 103

- 104

- 105

- 106

Numberoftrainingexamples

40

48

Numberoftrainingexamples

38

46

36

MRR

MRR

44

34

32

42

30

40

28

26

38

Categories

Categories

(a)

(b)

- Figure 4 (a) MRR distribution over categories on Google Review-V, (b) MRR distribution over categories on Yelp-V. We find (1) the performance per category is loosely correlated with number of training data; (2) when a category is more general and less ambiguous, the performance on the category is better.

without fine-tuning, VisualLens 7B improves by 1.6-5.6% on Hit@3.

- 6.2 Ablation studies (Q2)

We evaluate the usefulness of each component in VisualLens in Table 4, with PaliGemma as the backbone. We find history retrieval significantly improves the results, and can reduce Hit@3 by 7% on Google Review-V and by 12% on Yelp-V (#7 vs. #3). Besides, all three representations of the images (embedding, caption, aspects) play an important role. Removing captions and aspect words can reduce Hit@3 by 3% on Google Review-V and by 9% on Yelp-V, even without fine-tuning (#6 vs. #3). Between the two, aspect words play a more important role than captions (#5 vs. #4). Moreover, both iterative training and joint multi-task training improve the recommendation quality. Removing both of them lowers Hit@3 by 1%+ on both data sets (#3 vs #1).

- 6.3 Transferability (Q3)

We tested transferability over users (default setting), over categories, and over different (longer) history lengths. Table 5 compares the MRR of VisualLens on both benchmarks, showing good transferability. MRR is highest when applied to longer history, with the effectiveness of history retrieval. Transferability is higher across categories than across users, both demonstrating the promise of leveraging task-agnostic signals, and illustrating slightly more challenges to transfer between users of different interest patterns. We also present the generalization results across benchmarks in Appendix C.

- 6.4 Robustness and qualitative analysis (Q4) We conducted several robustness tests as follows.

Candidate count: Figure 3a shows that as the number of candidates grows, the recommendation becomes harder and MRR gets lower. However, when the number of candidates exceeds 50, MRR converges.

Image count in user history: Figure 3b shows that as the history grows, MRR increases and flattens after reaching ∼100 images. This trend is related to our grid size 8 × 8, as smaller than 64 images will not leverage all spaces in the grid. On the other hand, flat MRR after 100 images shows robustness against history noises with the retrieval step.

Category distribution: Figure 4 plots MRR for different categories. There are a few factors that can affect recommendation quality. First and foremost, ambiguous categories like “area”, “station”, and “market” get the lowest MRR in Google Review-V. Second, general categories (e.g., “museum”, “hotel”) with bigger category sizes obtain higher MRR than specific ones (e.g., “historical landmark”). Third, transfer learning happens between neighbor categories; for example, “deli” and “takeout” achieve the top-2 and top-3 performance with less training data, since they are similar to the largest category “restaurant”. We provide more qualitative analysis in Appendix C.

### 7 Conclusion and Discussion

In this paper, we proposed a novel approach VisualLens for personalized recommendation using task-agnostic visual history. We advanced MLLMs to extract spectrum signals from images to serve as user profile. We created two benchmarks, Google Review-V and Yelp-V, to evaluate VisualLens, affirming the efficacy and robustness of VisualLens.

VisualLens offers a promising first step toward MLLM recommendation systems that leverage task-agnostic visual history. Several future directions remain. First, we could integrate VisualLens with additional data, such as image timestamps, locations, recognized fine-grained entities (e.g., specific products), and user profiles. Second, we aim to extend the recommendation problems explored here to encompass broader QA tasks. Finally, we plan to investigate privacy-preserving techniques, such as federated learning, during training.

Limitation

While VisualLens demonstrates strong performance on task-agnostic multimodal recommendation, several limitations remain.

Modular design without optimal components. Our framework prioritizes modularity and extensibility, using a unified architecture across modules including image encoding, captioning, aspect word generation, and preference matching. However, we do not use the best-performing model for each subtask. Each component can be replaced with more advanced or domain-specific alternatives, which could potentially boost overall performance.

Limited modality and domain coverage. Our experiments focus exclusively on static visual history (images) and do not cover more complex modalities such as video, audio, or multimodal narratives over time. These richer formats are important for building truly comprehensive, lifelong user models, as envisioned by early ideas like Memex (Bush, 1945), but remain out of scope for this work.

Evaluation scope. We restrict the recommendation task to a QA format for consistency and interpretability. While this setup allows for controlled benchmarking and comparison, it does not cover other common recommendation forms such as ranked lists, interactive dialogues, or implicit feedback modeling. Additionally, our proposed benchmarks, Google Review-V and Yelp-V, capture a subset of the full task-agnostic recommendation landscape and may not reflect the full range of real-world personalization scenarios.

Future work could explore richer modalities, more diverse recommendation formats, and further improvements to the modular components of VisualLens.

Acknowledgements

The robot icon used in Figure 1 is from https://www.flaticon.com/free-icon/robot_2432846. The padlocks icons used in Figure 2 are from https://www.flaticon.com/free-icon/lock_996365 and https://www. flaticon.com/free-icon/padlock_535143. The illustration images in Figures 1, 2 are from DOCCI (Onoe

et al., 2024).

Social Impact

The ability to model user preferences from visual histories raises important considerations around privacy, consent, and data usage. While our work uses publicly available and anonymized images, real-world deployments would need to carefully address how user data is collected, stored, and interpreted, especially when recommendations extend beyond a single domain. There is also potential for reinforcing biases or overfitting to superficial cues (e.g., location, aesthetics) that may not reflect deeper user intent. We encourage future research to incorporate fairness auditing, privacy-preserving training, and transparency mechanisms to ensure responsible use of multimodal personalization systems.

References

Anthropic. Claude 3.5 sonnet. https://www.anthropic.com/news/claude-3-5-sonnet, 2024. https://www.anthropic. com/news/claude-3-5-sonnet.

Nabiha Asghar. Yelp dataset challenge: Review rating prediction, 2016. https://arxiv.org/abs/1605.05362.

Lucas Beyer, Andreas Steiner, André Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, Thomas Unterthiner, Daniel Keysers, Skanda Koppula, Fangyu Liu, Adam Grycner, Alexey Gritsenko, Neil Houlsby, Manoj Kumar, Keran Rong, Julian Eisenschlos, Rishabh Kabra, Matthias Bauer, Matko Bošnjak, Xi Chen, Matthias Minderer, Paul Voigtlaender, Ioana Bica, Ivana Balazevic, Joan Puigcerver, Pinelopi Papalampidi, Olivier Henaff, Xi Xiong, Radu Soricut, Jeremiah Harmsen, and Xiaohua Zhai. Paligemma: A versatile 3b vlm for transfer, 2024. https://arxiv.org/abs/2407.07726.

Vannevar Bush. As we may think. The Atlantic Monthly, 176(1):101–108, July 1945. Bingyi Cao, Andre Araujo, and Jack Sim. Unifying deep local and global features for image search. In Computer

Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XX 16, pages 726–743. Springer, 2020.

Sunhao Dai, Ninglu Shao, Haiyuan Zhao, Weijie Yu, Zihua Si, Chen Xu, Zhongxiang Sun, Xiao Zhang, and Jun Xu. Uncovering chatgpt’s capabilities in recommender systems. In Proceedings of the 17th ACM Conference on Recommender Systems, RecSys ’23, page 1126–1132. ACM, September 2023. doi: 10.1145/3604915.3610646. http://dx.doi.org/10.1145/3604915.3610646.

James Davidson, Benjamin Liebald, Junning Liu, Palash Nandy, Taylor Van Vleet, Ullas Gargi, Sujoy Gupta, Yu He, Mike Lambert, Blake Livingston, and Dasarathi Sampath. The youtube video recommendation system. In Proceedings of the Fourth ACM Conference on Recommender Systems, RecSys ’10, page 293–296, New York, NY, USA, 2010. Association for Computing Machinery. ISBN 9781605589060. doi: 10.1145/1864708.1864770. https://doi.org/10.1145/1864708.1864770.

Xingzhong Du, Hongzhi Yin, Ling Chen, Yang Wang, Yi Yang, and Xiaofang Zhou. Personalized video recommendation using rich contents from videos. IEEE Transactions on Knowledge and Data Engineering, 32(3):492–505, 2020. doi: 10.1109/TKDE.2018.2885520.

Deqing Fu, Ruohao Guo, Ghazal Khalighinejad, Ollie Liu, Bhuwan Dhingra, Dani Yogatama, Robin Jia, and Willie Neiswanger. IsoBench: Benchmarking multimodal foundation models on isomorphic representations. In First Conference on Language Modeling (COLM), 2024.

Deqing Fu, Tong Xiao, Rui Wang, Wang Zhu, Pengchuan Zhang, Guan Pang, Robin Jia, and Lawrence Chen. TLDR: Token-level detective reward model for large vision language models. In The Thirteenth International Conference on Learning Representations, 2025. https://openreview.net/forum?id=Zy2XgaGpDw.

Junyu Gao, Tianzhu Zhang, and Changsheng Xu. A unified personalized video recommendation via dynamic recurrent neural networks. In Proceedings of the 25th ACM International Conference on Multimedia, MM ’17, page 127–135, New York, NY, USA, 2017. Association for Computing Machinery. ISBN 9781450349062. doi: 10.1145/3123266.3123433. https://doi.org/10.1145/3123266.3123433.

Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5). Proceedings of the 16th ACM Conference on Recommender Systems, 2022. https://api.semanticscholar.org/CorpusID:247749019.

Google. Gemini: A family of highly capable multimodal models, 2023.

Yulong Gu, Zhuoye Ding, Shuaiqiang Wang, Lixin Zou, Yiding Liu, and Dawei Yin. Deep multifaceted transformers for multi-objective ranking in large-scale e-commerce recommender systems. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management, CIKM ’20, page 2493–2500, New York, NY, USA, 2020. Association for Computing Machinery. ISBN 9781450368599. doi: 10.1145/3340531.3412697. https://doi.org/10.1145/3340531.3412697.

Chen Hao, Xie Runfeng, Cui Xiangyang, Yan Zhou, Wang Xin, Xuan Zhanwei, and Zhang Kai. Lkpnr: Llm and kg for personalized news recommendation framework, 2023. https://arxiv.org/abs/2308.12028.

- F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS), 5(4), 19.

Ruining He and Julian McAuley. Vbpr: visual bayesian personalized ranking from implicit feedback. In Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, AAAI’16, page 144–150. AAAI Press, 2016.

Yupeng Hou, Shanlei Mu, Wayne Xin Zhao, Yaliang Li, Bolin Ding, and Ji-Rong Wen. Towards universal sequence representation learning for recommender systems. In KDD, 2022.

Yupeng Hou, Junjie Zhang, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian McAuley, and Wayne Xin Zhao. Large language models are zero-shot rankers for recommender systems, 2024. https://arxiv.org/abs/2305.08845.

Jianchao Ji, Zelong Li, Shuyuan Xu, Wenyue Hua, Yingqiang Ge, Juntao Tan, and Yongfeng Zhang. Genrec: Large language model for generative recommendation, 2023. https://arxiv.org/abs/2307.00457.

Joonseok Lee and Sami Abu-El-Haija. Large-scale content-only video recommendation. In 2017 IEEE International Conference on Computer Vision Workshops (ICCVW), pages 987–995, 2017. doi: 10.1109/ICCVW.2017.121.

Cheng Li, Mingyang Zhang, Qiaozhu Mei, Yaqing Wang, Spurthi Amba Hombaiah, Yi Liang, and Michael Bendersky. Teach llms to personalize – an approach inspired by writing education, 2023a. https://arxiv.org/abs/2308.07968.

Jiacheng Li, Jingbo Shang, and Julian McAuley. UCTopic: Unsupervised contrastive learning for phrase representations and topic mining. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6159– 6169, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.426. https://aclanthology.org/2022.acl-long.426.

Jiacheng Li, Ming Wang, Jin Li, Jinmiao Fu, Xin Shen, Jingbo Shang, and Julian McAuley. Text is all you need: Learning language representations for sequential recommendation. Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2023b.

Pan Li, Yuyan Wang, Ed H. Chi, and Minmin Chen. Prompt tuning large language models on personalized aspect extraction for recommendations, 2023c. https://arxiv.org/abs/2306.01475.

Jianghao Lin, Rong Shan, Chenxu Zhu, Kounianhua Du, Bo Chen, Shigang Quan, Ruiming Tang, Yong Yu, and Weinan Zhang. Rella: Retrieval-enhanced large language models for lifelong sequential behavior comprehension in recommendation, 2024. https://arxiv.org/abs/2308.11131.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024. https://llava-vl.github.io/blog/2024-01-30-llava-next/.

Qijiong Liu, Nuo Chen, Tetsuya Sakai, and Xiao-Ming Wu. Once: Boosting content-based recommendation with both

open- and closed-source large language models, 2023. https://arxiv.org/abs/2305.06566. Meta. The llama 3 herd of models, 2024. https://arxiv.org/abs/2407.21783. Jianmo Ni, Jiacheng Li, and Julian McAuley. Justifying recommendations using distantly-labeled reviews and fine-

grained aspects. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 188–197, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1018.

Yasumasa Onoe, Sunayana Rane, Zachary Berger, Yonatan Bitton, Jaemin Cho, Roopal Garg, Alexander Ku, Zarana Parekh, Jordi Pont-Tuset, Garrett Tanzer, Su Wang, and Jason Baldridge. Docci: Descriptions of connected and contrasting images, 2024. https://arxiv.org/abs/2404.19753.

OpenAI. Gpt-4o, 2024. https://openai.com/index/hello-gpt-4o/. Harris Papadakis, Antonis Papagrigoriou, Eleftherios Kosmas, Costas Panagiotakis, Smaragda Markaki, and Paraskevi

Fragopoulou. Content-based recommender systems taxonomy. Foundations of Computing and Decision Sciences, 48

(2):211–241, 2023. doi: doi:10.2478/fcds-2023-0009. https://doi.org/10.2478/fcds-2023-0009. Letitia Parcalabescu and Anette Frank. Do vision & language decoders use images and text equally? how self-consistent are their explanations? ArXiv, abs/2404.18624, 2024.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 2021. https: //api.semanticscholar.org/CorpusID:231591445.

Alireza Salemi, Sheshera Mysore, Michael Bendersky, and Hamed Zamani. LaMP: When large language models meet personalization, 2023.

Scott Sanner, Krisztian Balog, Filip Radlinski, Ben Wedin, and Lucas Dixon. Large language models are competitive near cold-start recommenders for language- and item-based preferences, 2023. https://arxiv.org/abs/2307.14225.

Xiang-Rong Sheng, Feifan Yang, Litong Gong, Biao Wang, Zhangming Chan, Yujing Zhang, Yueyao Cheng, Yong-Nan Zhu, Tiezheng Ge, Han Zhu, Yuning Jiang, Jian Xu, and Bo Zheng. Enhancing taobao display advertising with multimodal representations: Challenges, approaches and insights. In Proceedings of the 33rd ACM International

Conference on Information and Knowledge Management, CIKM ’24, page 4858–4865, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400704369. doi: 10.1145/3627673.3680068. https://doi.org/10. 1145/3627673.3680068.

Fei Sun, Jun Liu, Jian Wu, Changhua Pei, Xiao Lin, Wenwu Ou, and Peng Jiang. Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management, CIKM ’19, page 1441–1450, New York, NY, USA, 2019. Association for Computing Machinery. ISBN 9781450369763. doi: 10.1145/3357384.3357895. https://doi.org/10.1145/3357384.

3357895. Zhaoxuan Tan and Meng Jiang. User modeling in the era of large language models: Current research and future directions, 2023. https://arxiv.org/abs/2312.11518.

Jiaxi Tang and Ke Wang. Personalized top-n sequential recommendation via convolutional sequence embedding. In Proceedings of the Eleventh ACM International Conference on Web Search and Data Mining, WSDM ’18, page 565–573, New York, NY, USA, 2018. Association for Computing Machinery. ISBN 9781450355810. doi: 10.1145/3159652.3159656. https://doi.org/10.1145/3159652.3159656.

Jinhui Tang, Xiaoyu Du, Xiangnan He, Fajie Yuan, Qi Tian, and Tat-Seng Chua. Adversarial training towards robust multimedia recommender system. IEEE Transactions on Knowledge and Data Engineering, 32(5):855–867, 2020. doi: 10.1109/TKDE.2019.2893638.

Mengting Wan and Julian J. McAuley. Item recommendation on monotonic behavior chains. In Sole Pera, Michael D. Ekstrand, Xavier Amatriain, and John O’Donovan, editors, Proceedings of the 12th ACM Conference on Recommender Systems, RecSys 2018, Vancouver, BC, Canada, October 2-7, 2018, pages 86–94. ACM, 2018. doi: 10.1145/3240323.

3240369. https://doi.org/10.1145/3240323.3240369.

Jie Wang, Fajie Yuan, Mingyue Cheng, Joemon M. Jose, Chenyun Yu, Beibei Kong, Xiangnan He, Zhijin Wang, Bo Hu, and Zang Li. Transrec: Learning transferable recommendation from mixture-of-modality feedback, 2022. https://arxiv.org/abs/2206.06190.

Tianxin Wei, Bowen Jin, Ruirui Li, Hansi Zeng, Zhengyang Wang, Jianhui Sun, Qingyu Yin, Hanqing Lu, Suhang Wang, Jingrui He, and Xianfeng Tang. Towards unified multi-modal personalization: Large vision-language models for generative recommendation and beyond, 2024a. https://arxiv.org/abs/2403.10667.

Wei Wei, Chao Huang, Lianghao Xia, and Chuxu Zhang. Multi-modal self-supervised learning for recommendation. In Proceedings of the ACM Web Conference 2023, pages 790–800, 2023.

Wei Wei, Xubin Ren, Jiabin Tang, Qinyong Wang, Lixin Su, Suqi Cheng, Junfeng Wang, Dawei Yin, and Chao Huang. Llmrec: Large language models with graph augmentation for recommendation, 2024b. https://arxiv.org/abs/2311. 00423.

Yinwei Wei, Xiang Wang, Liqiang Nie, Xiangnan He, Richang Hong, and Tat-Seng Chua. Mmgcn: Multi-modal graph convolution network for personalized recommendation of micro-video. In Proceedings of the 27th ACM International Conference on Multimedia, pages 1437–1445, 2019.

Fangzhao Wu, Ying Qiao, Jiun-Hung Chen, Chuhan Wu, Tao Qi, Jianxun Lian, Danyang Liu, Xing Xie, Jianfeng Gao, Winnie Wu, and Ming Zhou. MIND: A large-scale dataset for news recommendation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 3597–3606, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.331.

Wentao Xu, Qianqian Xie, Shuo Yang, Jiangxia Cao, and Shuchao Pang. Enhancing content-based recommendation via large language model. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, CIKM ’24, page 4153–4157, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400704369. doi: 10.1145/3627673.3679913. https://doi.org/10.1145/3627673.3679913.

An Yan, Zhankui He, Jiacheng Li, Tianyang Zhang, and Julian McAuley. Personalized showcases: Generating multi-modal explanations for recommendations. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’23, page 2251–2255, New York, NY, USA, 2023.

Association for Computing Machinery. ISBN 9781450394086. doi: 10.1145/3539618.3592036. https://doi.org/10. 1145/3539618.3592036.

Fan Yang, Zheng Chen, Ziyan Jiang, Eunah Cho, Xiaojiang Huang, and Yanbin Lu. Palr: Personalization aware llms for recommendation, 2023. https://arxiv.org/abs/2305.07622.

Jia-Qi Yang, Chenglei Dai, Dan OU, Dongshuai Li, Ju Huang, De-Chuan Zhan, Xiaoyi Zeng, and Yang Yang. Courier: Contrastive user intention reconstruction for large-scale visual recommendation, 2024. https://arxiv.org/abs/2306. 05001.

Yuhao Yang, Chao Huang, Lianghao Xia, Yuxuan Liang, Yanwei Yu, and Chenliang Li. Multi-behavior hypergraphenhanced transformer for sequential recommendation. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, KDD ’22, page 2263–2274, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450393850. doi: 10.1145/3534678.3539342. https://doi.org/10.1145/3534678. 3539342.

Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, Qianyu Chen, Huarong Zhou, Zhensheng Zou, Haoye Zhang, Shengding Hu, Zhi Zheng, Jie Zhou, Jie Cai, Xu Han, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint 2408.01800, 2024.

Junjie Zhang, Ruobing Xie, Yupeng Hou, Wayne Xin Zhao, Leyu Lin, and Ji-Rong Wen. Recommendation as instruction following: A large language model empowered recommendation approach, 2023. https://arxiv.org/abs/2305.07001.

Bowen Zheng, Zihan Lin, Enze Liu, Chen Yang, Enyang Bai, Cheng Ling, Wayne Xin Zhao, and Ji-Rong Wen. A large language model enhanced sequential recommender for joint video and comment recommendation, 2024. https://arxiv.org/abs/2403.13574.

Peilun Zhou, Xiaoxiao Xu, Lantao Hu, Han Li, and Peng Jiang. A model-based multi-agent personalized short-video recommender system, 2024. https://arxiv.org/abs/2405.01847.

Wang Zhu, Jesse Thomason, and Robin Jia. Generalization differences between end-to-end and neuro-symbolic vision-language reasoning systems. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, Findings of the Association for Computational Linguistics: EMNLP 2022, pages 4697–4711, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-emnlp.345. https://aclanthology.org/2022.findings-emnlp.345.

Wang Zhu, Alekh Agarwal, Mandar Joshi, Robin Jia, Jesse Thomason, and Kristina Toutanova. Efficient end-to-end visual document understanding with rationale distillation. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8401–8424, Mexico City, Mexico, June 2024a. Association for Computational Linguistics. https://aclanthology.org/2024.naacl-long.465.

Yaochen Zhu, Liang Wu, Qi Guo, Liangjie Hong, and Jundong Li. Collaborative large language model for recommender systems. In Proceedings of the ACM Web Conference 2024, WWW ’24, page 3162–3172. ACM, May 2024b. doi: 10.1145/3589334.3645347. http://dx.doi.org/10.1145/3589334.3645347.

## Appendix

- A Benchmark Creation Details 16

- A.1 Algorithm for candidate generation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- A.2 Processing and filtering . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- A.3 Training data distribution . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18

- B More Related Works 18

- B.1 Traditional recommendation methods. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- B.2 Multimodal large language models. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- B.3 Datasets and evaluating multimodal recommendation. . . . . . . . . . . . . . . . . . . . . . . 18

- C More Analysis 19

- C.1 Sensitivity of random sampling for candidate embedding . . . . . . . . . . . . . . . . . . . . . 19
- C.2 Generalization between Google Review-V and Yelp-V . . . . . . . . . . . . . . . . . . . . . . 19
- C.3 Qualitative results . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- D Prompt Template 19
- E Additional Implementation Details 20
- F Licenses 21

### A Benchmark Creation Details

- A.1 Algorithm for candidate generation

We list the complete candidate and ground truth sets generation algorithm in Algorithm 1, where the function nearest(G,loc,m) returns the m nearest businesses of around certain location loc based on the graph G. The function unique_name(S) removes the businesses in the set S with redundant names.

In both Google Review-V and Yelp-V, rcmin = 30, rcmax = 100, fcmin = 10.

Data distribution

Categories

- 104

- 105

- 106

- 107

restaurant

museum

gastropub

bistro

waterfall

bar

deli

golf course

marina

pet groomer

tourist attraction

theater takeout pub

atm

lake gym hospital

mountain peak

park cafe grill

nature preserve

zoo

Numberofexamples

church winery car wash

pharmacy

area

spa

roasters

shopping mall

hall

farm

boutique

bakery

historical landmark

campground

florist

arena

supermarket

club

lodge

bank

therapist

steak house

brewpub

casino

bowling alley

grocer airport nursery

station

salon

playground

clinic

hotel

beach

gallery

bridge library

brewery

garden

stadium

parking lot

market

Categories

###### Figure 5 The Google Review-Vision (Google Review-V) training data consists of 66 categories.

Algorithm 1 Candidate and Ground Truth Sets Generation

Input: Business geographics graph G = (V,E), Visit set B of a user, Category c, Minimum random candidate count rcmin, Maximum random candidate count rcmax, Minimum final candidate count fcmin, Minimum ground truth count gtcmin.

Output: Candidate sets S1CD..n, Ground truth sets S1GT..n. # Initialization for each business b ∈ B do

flag fb ← unselected

end for count ← 0 # Main algorithm for each business b ∈ B do

if b is in category c and fb = unselected then count ← count + 1 Candidate count m ← random(rcmin,rcmax) Candidate set ScountCD ← nearest(G,bloc,m) Candidate set ScountCD ← unique_name(ScountCD ) Ground truth set ScountGT ← ScountCD ∩ B if |ScountCD | < fcmin or |ScountGT | < gtcmin then

count ← count − 1

continue end if for each business b′ ∈ ScountGT do

flag fb′ ← selected

end for

end if end for return (S1CD..count,S1GT..count)

- A.2 Processing and filtering

User logs: For each user in the two datasets, we take the list of reviews in chronological order. Each review is associated with a business name, categories and description. In Google Review-V, each review is associated with a few photos, used for image logs. Yelp-V does not associate a review with photos, so we randomly subsample one-third of the store profile pictures, such that different reviews for the same business can be associated with different images.

Category removal and merge: For Google Review-V and Yelp-V, the categories are selected as the last word of the annotated tags in the business. However, some category requires multiple words to express its meaning, such as “tourist attraction”, “steak house”, “historical landmark", “nature preserve”, etc. We select and keep these multi-word categories.

In Google Review-V, we remove smaller categories with less than 10k occurrence in the dataset. Then, we remove ambiguous or non-differentiable categories, including “shop”, “store”, “complex”, “service”, “company”, “supplier”, “caterer”, “agency”, “center”, “organization”, “attraction”, “house”, “mall”, “landmark”, “wash”, “course”, “preserve”, “alley”, “groomer”, “field”, “peak”, “venue”, “delivery”, “dealer”, “lounge”, “office”, “arcade”, “court”, “spot”, “stop”, “maintenance”, “trainer”, “wholesaler”, “planner”, “place”, “facility”, “school”, “stand”, “range”, “consultant”, “designer”, “veterinarian”, “ground”, “contractor”, “manufacturer”, “studio”, “point”, “lot”.

In Yelp-V, as the category is very centralized, we remove smaller categories with less than 50k occurrence in the dataset. Then, we remove ambiguous or non-differentiable categories, including“planning”, “nightlife”, “services”, “wings”, “arts”, “dogs”, “tacos”, “caribbean”, “beer”, “spirits”, “wine”, “venues”, “fusion”, “entertainment”, “southern”, “spaces”, “lounges”, “breweries”, “shopping”, “smoothies”, “flavor”, “plates”, “eastern”, “tex-mex”, “shop”, “noodles”, “markets”, “market”, “donuts”, “gelato”, “sum”, “veggies”, “fruits”, “trucks”,

“bagels”, “cheesesteaks”, “clubs”, “cuban”, “ramen”, “life”, “roasteries”, “stands”, “brewpubs”, “gluten-free”, “gardens”, “travel”.

Questions and visual history: We consider a special type of questions, category recommendations, like "Recommend a nearby museum". Such questions are both popular in real applications, and hard as there are many candidates satisfying the constraint. We remove small categories and most ambiguous categories, such as “place”, and “spot”.

For each review r regarding a business of category c, we create a question asking to recommend businesses in category c. We take all (and only) photos in the reviews before r to generate the visual history. We consider the visual history task-agnostic since the categories are highly diverse (see Figure 4), and the photos are quite diverse too (e.g., a park photo to illustrate happiness mentioned in the review). We filter cases where the history is too short (<10) or does not contain the questioned category. By doing so, we ensure that the user history contains at least 10 relevant images (i.e., from the same category as the query) for each example in both Google Review-V and Yelp-V.

Candidates and ground truths: For a review r we take all reviews starting from r to generate candidates and ground truths. To be realistic, we consider only nearby businesses of the same category as the candidate set, and the number of candidates is a random number in [30,100]. Candidates that also appear in the user’s future reviews are considered as ground truths. To avoid falling into a classification problem, we filter examples with only 1 ground truth in Google Review-V and fewer than 5 in Yelp-V.

- A.3 Training data distribution We list the training data distribution of Google Review-V in Figure 5.

### B More Related Works

- B.1 Traditional recommendation methods.

Before the LLM era, there are also a line of work for recommendation with other networks, smaller language models, or ensembling methods. For example, Sun et al. (2019) uses a bidirectional Transformer to model the item sequence. Tang and Wang (2018) uses CNN to model user preference with convolutional filters. Gu et al. (2020) and Yang et al. (2022) uses ensembling methods for recommendation.

- B.2 Multimodal large language models.

Multimodal LLMs are becoming increasingly powerful in processing images and videos and in generating human-like text, exemplified by models like GPT-4 family (OpenAI, 2024), Claude 3.5 family (Anthropic, 2024), Gemini and PaliGemma (Google, 2023; Beyer et al., 2024), LLaVA model family (Liu et al., 2024), and Llama 3 Vision models (Meta, 2024). However, they still suffer from a strong language prior (Fu et al., 2024; Parcalabescu and Frank, 2024), generalization (Zhu et al., 2022) and hallucinations (Fu et al., 2025), or require heavy text transcription (Zhu et al., 2024a). The extent to which they can understand user history, in analogy to LLM recommendation systems, is still unclear.

- B.3 Datasets and evaluating multimodal recommendation.

The development of multimodal recommendation systems has been facilitated by the availability of diverse datasets that incorporate various types of data, including text, images, and user interaction histories. Notable datasets in this domain include MovieLens-1M (Harper and Konstan, 19), which is extensively used for movie recommendations, and the Amazon dataset (Ni et al., 2019), which includes user reviews and metadata for product recommendations. For news and books recommendations, the MIND (Wu et al., 2020) and Goodreads (Wan and McAuley, 2018) datasets offer insights into user preferences in news and literature. Recently, LaMP (Salemi et al., 2023) introduces 7 text classification and text generation tasks across long contexts to evaluate LLM’s personalization capablity. In the context of outdoor activities, datasets such as Google Local Data (Li et al., 2022; Yan et al., 2023) and Yelp (Asghar, 2016) are crucial. These datasets not only provide

textual reviews but also include user ratings and geospatial data, which are essential for recommending local businesses and services. These benchmarks utilize the traditional recommendation systems that focus on ranking user preferences based on historical IDs. Instead, we introduce new visual history benchmarks Google Review-V and Yelp-V, using data from Google Local and Yelp, to evaluate the multimodal recommendation systems under a more realistic visual history for task-agnostic setups.

- C More Analysis

- C.1 Sensitivity of random sampling for candidate embedding

We show that the recommendation results are not sensitive to the random sampling of images in candidate embedding. Table 6 below shows that the robustness against sampling randomness: 1) the standard deviation is <0.1 when we do 5 runs of sampling; 2) as we reduce the size of the samples to 1K and 100, the MRR stays similar, with <0.5 difference.

n Google Review-V Yelp-V 10k 33.5 ± 0.1 44.3 ± 0.0 1k 33.4 ± 0.1 44.3 ± 0.1 100 33.0 ± 0.2 44.1 ± 0.1

- Table 6 MRR and its standard deviation (5 runs) of VisualLens (PaliGemma) on different sampling sizes.

C.2 Generalization between Google Review-V and Yelp-V

We report the generalization quality between Google Review-V and Yelp-V. Table 7 below shows that cross-source recommendation reduces MRR by 2-4% respectively vs. in-domain, in an acceptable range, highlighting the transferability from the shared feature in the embedding space. Besides, transferring from Google Review-V to Yelp-V is still better than the best baseline model in-domain (40.2), while the other side is much worse (33.2). This is as expected, since Google Review-V has much more categories and much lower photo quality.

Train data Test data Google Review-V Yelp-V

Google Review-V 33.5 41.9 Yelp-V 29.4 44.3

- Table 7 MRR of VisualLens (PaliGemma) on in-domain test and cross-domain transfer.

- C.3 Qualitative results

We show the input and output of a positive example in Figure 6 and Figure 7. We show the input and output of a negative example in Figure 8 and Figure 9. Please note that due to image licensing restrictions, we are only able to provide the image names in JPG format. For more details, you can download the image data from https://business.yelp.com/data/resources/open-dataset/.

- D Prompt Template

We list the prompt template for aspect word generation in Figure 10, and the prompt template for candidate matching in Figure 11. The model will fill in category information in the [[Category]] slot and predict the answer at the [[Answer]] slot.

| |Input| |
|---|---|---|
| | | |

Question: What is the seafood the user most likely to go to next? Images: vJqN7cQXBVKHJJPXLZhoiQ.jpg, r4AfL_MgfmF3kmPMFs8TFg.jpg, UKMOX-Zvg6jMFNw7q65oEA.jpg,

wPn4exXYYNKP3lzSqyvlLg.jpg, 2ArnN2RJkdSRsiTfTdEK5A.jpg, pRwlK1-CwGNcuaj68ajV_A.jpg, tX20F76pfX54eOh1kkznsg.jpg, YQiidvr8-Pb3Bl7PS1EvTw.jpg, 0nuzLrJuSMfbuwK_eu20CQ.jpg, TXGOYaPCse3jSjNb22qBbg.jpg, CNF90JgRzw8OSDCtUWOwJQ.jpg, H0lVV7AqqNHqIxg5W-7VqA.jpg, hC5YoVNzvbsSTIBzYUw6gQ.jpg, 2t1gOf8J6RdGAN08YtzK6A.jpg, wyomfq33QG7Lcsm2I1keYg.jpg, xBEVh2jWRaHFo-GVDkdYww.jpg, 8rXB-RWfztDFTUXd4v0AjQ.jpg, 7jT9OQ6iWhysKmfKtSLjSw.jpg, dMq7TX-_HAZevi0ec2_l_w.jpg, h2vbvkl-qVtFykWSnXMA6A.jpg, MQgymZxJgwvZi9ceTKhJqw.jpg, BYGg4bPM9RyOlcpbj_pdPA.jpg, tfxriiwBgyz-kTYhWygpCg.jpg, Ara9kbHnH_kiwWf03Nr-dA.jpg, yFE4KT49_8uY5CRD4N8ZYQ.jpg, ANXYMnTSlIdcSSxBTaeuyg.jpg, S3jQKlAJqGC4TU0gDK66Hw.jpg, spTRlxA-jfJWTyZTw2cgVg.jpg, qJoah_eRYngBqr9loubCbA.jpg, MQuNacSddcwXFtoblQ1GkQ.jpg, EhkCMYoscdfMpGubTBfl3A.jpg, y4r4dgzZfPo4q_5YAiGHvg.jpg, jxKIXDjCqAsqyakhCUC3Lw.jpg, 2W7reraU4LA0XrP78y64sw.jpg, FhVHa7bRVrzMda_Dm1wnTg.jpg, KWzV50vfln0_ag5qs989OQ.jpg, Mc25dRycrpy2SHGihHOW0g.jpg, FmT3RYVQY6Rkm5MlYt1EAQ.jpg, zIRzC3yzhcSHO8VLTemtFg.jpg, _hkeputL9COjS_sXrRDeIg.jpg, 0oyPpSHInW-lnnToM-xeXQ.jpg, 3Bkrs4_XOysf5EBx7tE_lg.jpg, deQ74JW91K-St4RgULT-nQ.jpg, WPKSSCdLPteqrwpeS9hZvA.jpg, 9biNDFQDY8r9BlasGKk2GA.jpg, wi4Tm0FMuQ2okQwYtHPyWw.jpg, tsjxColf4XZmBfq-B7WJLQ.jpg, Y0D5OiZPVikxP3nXzSE19Q.jpg, YQuc3-t3p2lnQyaWEE0xGQ.jpg, ezf9bs1GI44t8XkBiYOHkQ.jpg, fOe1E9JsDCGGxTuJaxJUQg.jpg, a8cVlIBv2ztpNq-zU93Ghw.jpg, RfO6Kr6H0RrA5T_L50B1mQ.jpg, OkelD8PM3aYV_oiT_OOfPQ.jpg, Q6ZABL2V_WkUcjZAQS48Cg.jpg, VaOy8CE5EO0-ZArdyQQkIA.jpg, dLvdOgh417CZBqO3NUDriQ.jpg, 8tX8MDZ7nNBsbOw4M7Qfiw.jpg, QIYfwuQc-Pw2erU501ipYQ.jpg, yvvlCHgMZIa8Vg9UlIQ9Iw.jpg, mWjD72pB7TnoSYCOW_EYQw.jpg, TK5g5F19WJZuTsXxf77XVg.jpg, NPsJmoYIdvoRzTv4l-jjmg.jpg, 2MlS2-V5AJcVfHInXUxCDg.jpg, UlJkD0OHYsXjPbBUXMIvCg.jpg, oowr6hZezzRrAf15FfNaRQ.jpg, ifEeKdHzwpI4He54LylxFw.jpg, S5zpR2aJD5opF3TSyr5GHg.jpg, f0jBRH-6c0C7LKRMZ_7CHA.jpg, iIpBj0_0F-FnGu5byUXHHw.jpg

Candidates: <I1> Stoney River Steakhouse and Grill <I2> JWB Grill <I3> Morton's The Steakhouse <I4> The Hampton Social - Nashville <I5> Char Restaurant <I6> Batter'd & Fried Boston Seafood House <I7> Captain D's <I8> Willie B's Kitchen & Lounge <I9> Ruth's Chris Steak House <I10> Bob's Steak & Chop House <I11> Fish & Co <I12> Poke Bros <I13> Rudie's Seafood & Sausage <I14> Far East Nashville <I15> Demos' Restaurant <I16> Sehrt Seafood Company <I17> Firebirds Wood Fired Grill <I18> Reyes Mexican Grill & Mariscos <I19> East Side Fish <I20> Crab Fever <I21> Sitar Indian Restaurant <I22> Carrabba's Italian Grill <I23> Jeff Ruby's Steakhouse- Nashvill <I24> Fleming’s Prime Steakhouse & Wine Bar <I25> Amerigo Italian Restaurant <I26> The River House <I27> Etch <I28> Fannie Mae's <I29> The Hook <I30> Green Hills Grille <I31> Pemrose <I32> The Southern Steak & Oyster <I33> Oak Steakhouse Nashville <I34> El Tapatio #2 <I35> Red Perch <I36> Marsh House <I37> Sushi 88 <I38> RuSan's Sushi and Seafood <I39> Skull's Rainbow Room <I40> No 1 Chinese <I41> Little Octopus <I42> The Cheesecake Factory - Nashville <I43> Juicy Seafood <I44> House of Cards <I45> Joe's Crab Shack <I46> Henrietta Red <I47> Urban Grub <I48> Bombay Palace Restaurant <I49> The Gumbo Bros <I50> The Diner <I51> Redlands Grill - Nashville <I52> The Green Pheasant <I53> Nomzilla! Sushi Et Cetera <I54> The Smiling Elephant <I55> Sperry's Restaurant <I56> Louie's Wine Dive <I57> Eddie V's Prime Seafood <I58> The Optimist - Nashville <I59> The Silly Goose <I60> Cousins Maine Lobster - Nashville <I61> Jimmy Kelly's Steakhouse <I62> J. Alexander's Restaurant <I63> 360 Wine Bar Bistro <I64> The Palm - Nashville <I65> Blue Moon Waterfront Grille <I66> Bourbon Steak by Michael Mina <I67> Boston Commons <I68> White Limozeen <I69> Saltine Fine Food and Drink <I70> Seafood Sensation <I71> Siam Cafe <I72> The Continental <I73> South Street

- Figure 6 A successful example of VisualLens on Hit@1. We list the input question, images and candidates.

- E Additional Implementation Details

We present the hyperparameters for training VisualLens in PaliGemma and MiniCPM-V2.5 in Table 8 and Table 9. Note that since there are at most 100 candidates, we add special tokens <I1>, ..., <I100> as

| |Output| |
|---|---|---|
| | | |

Candidate Ranking: <I66> Bourbon Steak by Michael Mina <I33> Oak Steakhouse Nashville

- <I61> Jimmy Kelly's Steakhouse <I23> Jeff Ruby's Steakhouse- Nashville <I9> Ruth's Chris Steak House <I24> Fleming's Prime Steakhouse & Wine Bar <I10> Bob's Steak & Chop House <I3> Morton's The Steakhouse <I64> The Palm - Nashville <I72> The Continental

- <I47> Urban Grub <I57> Eddie V's Prime Seafood

- <I58> The Optimist - Nashville <I32> The Southern Steak & Oyster

- <I17> Firebirds Wood Fired Grill <I1> Stoney River Steakhouse and Grill <I5> Char Restaurant <I51> Redlands Grill - Nashville <I36> Marsh House <I69> Saltine Fine Food and Drink

- <I26> The River House <I44> House of Cards

<I62> J. Alexander's Restaurant <I46> Henrietta Red <I55> Sperry's Restaurant <I2> JWB Grill <I4> The Hampton Social - Nashville <I6> Batter'd & Fried Boston Seafood House <I13> Rudie's Seafood & Sausage <I16> Sehrt Seafood Company

- <I60> Cousins Maine Lobster - Nashville <I70> Seafood Sensation <I29> The Hook <I35> Red Perch <I43> Juicy Seafood <I45> Joe's Crab Shack <I49> The Gumbo Bros <I11> Fish & Co <I38> RuSan's Sushi and Seafood <I20> Crab Fever <I41> Little Octopus <I65> Blue Moon Waterfront Grille

<I27> Etch <I39> Skull's Rainbow Room <I68> White Limozeen <I56> Louie's Wine Dive <I63> 360 Wine Bar Bistro <I30> Green Hills Grille <I50> The Diner <I73> South Street <I31> Pemrose <I12> Poke Bros

- <I52> The Green Pheasant <I67> Boston Commons

<I59> The Silly Goose <I28> Fannie Mae's <I8> Willie B's Kitchen & Lounge <I37> Sushi 88

- <I53> Nomzilla! Sushi Et Cetera <I15> Demos' Restaurant

<I18> Reyes Mexican Grill & Mariscos <I19> East Side Fish <I22> Carrabba's Italian Grill <I25> Amerigo Italian Restaurant <I34> El Tapatio #2 <I42> The Cheesecake Factory - Nashville

- <I54> The Smiling Elephant <I71> Siam Cafe <I7> Captain D's <I14> Far East Nashville

- <I21> Sitar Indian Restaurant <I40> No 1 Chinese

<I48> Bombay Palace Restaurant

| |Ground Truth| |
|---|---|---|
| | | |

<I12> Poke Bros

- <I22> Carrabba's Italian Grill <I33> Oak Steakhouse Nashville <I57> Eddie V's Prime Seafood

- <I61> Jimmy Kelly's Steakhouse <I66> Bourbon Steak by Michael Mina <I68> White Limozeen

- Figure 7 A successful example of VisualLens on Hit@1. We list the output candidate ranking and ground truth. The ranking follows the same left-right order as the input candidates.

identifier of the candidates in the vocab of the models. By doing so, we can predict the rank by taking the probability of only those special tokens in the first output token.

- F Licenses We list the licenses involved in this work as follows.

- • Our usage of Google Local Data is under the license of Google Maps Platform Terms of Service.
- • Our usage of Yelp Data is under Yelp Dataset License.
- • PaliGemma models are under a custom license the Gemma Terms of Use (https://ai.google.dev/ gemma/terms).

| |Input| |
|---|---|---|
| | | |

Question: What is the pizza the user most likely to go to next? Images: hhRkpdongqf8m5EE10YQRQ.jpg, 9ToJiDVDJy5pUnSoMNGuBg.jpg, KFx5LzegrM_8TFwliDk9aQ.jpg,

742SAAE7x0gmDRGsKHQ4UA.jpg, 6q5oq-bpGqEQ6jEnwy-myA.jpg, KC6EHtjGVe-Dvx8v5R3tEA.jpg, VSJRG9TWrNukgVuk9MS1cQ.jpg, 5pYM0N5hx0el5R1C1DCYew.jpg, UPQQfA1mrMEJSYPl8_QjBw.jpg, 9z4Pu_zbWb-W7Y_j-7H1Yg.jpg, Ido4zXZ7KOQeRY5hNuBAsg.jpg, Rmtu1zLSG68i1d4JT62CQQ.jpg, ItWufsYqPm0JQxoohSTiUA.jpg, dJtqcFC8PRud3yk8s1TvvQ.jpg, c586W6qreQigu6EgltB30w.jpg, c7g6ciD8KYu3vb6Cw0DGCA.jpg, 3Ob64cs6HgG0CCUY-ppj4A.jpg, rOzJH8qZYacB2Xa-AbzE9A.jpg, mO4d4KAlxdvNdUGu0av7kQ.jpg, xou5bMV0GxYrVUU3tCDDBg.jpg, uzbaUWOJnC3f9V35pJUn-A.jpg, QVGyJPteap1ueoMz8o5JsA.jpg, cTeR6ZQh-KQgcaO1jXMbpQ.jpg, 66YOiOkDwsvFYkCiOBa3KQ.jpg, By97Yxw5e4Ab5DfpEMQ6uw.jpg, t_Vr-IjXTgE6PZg7qUpafA.jpg, EbsYH1VMwXtQuLXwdFw1Zw.jpg,

- -MvxNZN4RIkzNRBc003DwQ.jpg, 0nCG8FvsObbrpYulUmnuTQ.jpg, 5Zg5zYaGlpkApTIVFTqeGQ.jpg, _AFOqfRIUIwyJM1FSzI0EA.jpg, Hi3drxwhhP_qvHeC8VbGAg.jpg, RHyRk_8lHMzEtlh2-TLdrQ.jpg, z_xa3u4wx4NHjW8eoZHUvw.jpg, G9B1SMaiYILP8AjC4pTQ1A.jpg, aEgf8X4R4bJDuAE_kYjtKw.jpg, R96I_J2M81SYz0WdXQe_oA.jpg, x9uHc7Y9Bzy7KkwDeJx2Kw.jpg, uQ5xcTpP2JAsUOohT8DI2g.jpg, UseElxL5U7wYbmlf9Az7kQ.jpg, QySJ5zZhePEuQLrixs5O7w.jpg, HqJ_3OzutHD-yr2UajmA8w.jpg, i8JBFOQ907Akk-VutW5gSA.jpg, ZELwmcr7KW1MNqMkAN9E2Q.jpg, BgVWrsi78V-BxiLxaj9PGA.jpg, zoeOP-t0QR_1b4iDC6j2HQ.jpg, SnkcEBSZMuQCpbWCjVkchw.jpg, VeBxRrP3g2T3lJD_dOTxGQ.jpg,
- -mIOvTzcHsjr1drlapEITA.jpg, hs4OYvrita3-pM4ags835A.jpg, HhoyiXcspjVILME9pGG00g.jpg, FHCf0kMZMzO9DCHC26QjOQ.jpg, 3b5QJj8GQA8g5jipS-WWjw.jpg, IGVKYExRpkDoQBWupcIRDQ.jpg, HDELx9GYX6iUvtZxQ8ctcA.jpg, aws74_-4V-xybaXhOp7KxQ.jpg, BUlr9EAkvHjA68HGkTjzpQ.jpg, 33NfUzGpm0pojsul6xrnnA.jpg, hcztlNNBk1OpGj5lXBbfRg.jpg, uWj0ff2rembupvVUJnfyWQ.jpg, 1lxJBNjg7ZzZhNUWUpqbVA.jpg, wNBmdFh-m-owSFOVdlz3xw.jpg, V1sHwJCt_1-WRl3_bgnuCA.jpg, eCi5nuodtwhbMhNzBDCXvw.jpg, YqOF3R95qupsBzlaDBSW_g.jpg, vwFLRRbHDAq8qI4d0_hslg.jpg, N0aSm4quN2s-XUxz2LnhxA.jpg, qz3bDurS9JjWX6v2_pJBKA.jpg, _ykibX3snmvSYQctujwEWg.jpg, OcF6jXTNLUC20WhUVSfcpg.jpg, 3jGXj-oz5VtbqjIm9L93xA.jpg, Tsnd-cwmKWNfdkpcI-U8yw.jpg, QaysW7lx3PpfwuoXuYH-vg.jpg, 7bDOOxi0KAz9GMgP3bYDhg.jpg, MymG2Zc5A9QWUE1AYuL33Q.jpg, fp9rdA0RghLISyH9-atgyw.jpg, bBCGH3vssPXOM-gHzvL-KA.jpg, 9D4AZeD-8qPRMesOWX4OWA.jpg ...

Candidates: <I1> La Cucina Italian Eatery <I2> Chuck E. Cheese <I3> M3 Restaurant <I4> Yummy's <I5> Pizzeria Lupo <I6> Brugos <I7> Huntsman Brewing <I8> Godfather's Pizza <I9> Pizzava <I10> Little Caesars <I11> Electric Blue Elephant <I12> Eclipse Pizza Co. <I13> Knockouts Bar and Hookah Lounge <I14> Firetrail Pizza <I15> Gold 'N Silver Inn <I16> J J's Pie <I17> Smiling With Hope Pizza <I18> Chicago's Pizza With A Twist Reno - S Virginia St, NV <I19> Monaciello <I20> Rick's Pizza, Beer, & More <I21> Noble Pie Parlor - Downtown <I22> Food + Drink <I23> Korean Barbie Q <I24> Pizza Baron - Reno <I25> Semenza's Pizzeria <I26> Wild Garlic Pizza & Pub <I27> Casale's Halfway Club <I28> The Pizza Collective <I29> Pub N' Sub <I30> Noble Pie Parlor - Midtown Reno <I31> Little Waldorf Saloon <I32> Mountain Mike's Pizza <I33> Sizzle Pie <I34> West Street Market <I35> Cafe Whitney <I36> Piezzetta Pizza Kitchen <I37> Pizanos Pizza <I38> Pizza Hut <I39> Rattlesnake Club <I40> zpizza <I41> Taste of Chicago <I42> Playfield 76 <I43> Round Table Pizza <I44> The Blind Onion Pizza & Pub <I45> Opa Gourmet Pizza Cuisine <I46> Paulie's Pizza <I47> The Blind Onion Pizza & Wings <I48> The Rack <I49> Boulevard Pizza <I50> MOD Pizza <I51> Calafuria <I52> Atlantis Café Alfresco <I53> Domino's Pizza <I54> Wild Garlic <I55> Chicago's Pizza With A Twist - Reno, NV <I56> Peluso's Apizza <I57> Liberty Food & Wine Exchange <I58> Opa Cafe <I59> California Pizza Kitchen <I60> Chicago Dogs! Eatery <I61> Hidden Pizza <I62> Pizza Factory

- Figure 8 A failed example of VisualLens on Hit@3. We list the input question, images and candidates.

- • MiniCPM-V2.5 is under Apache License 2.0
- • Our usage of OpenAI’s models for prompting is under the license of OpenAI"s Terms of Service.

| |Output| |
|---|---|---|
| | | |

###### Candidate Ranking:

- <I40> zpizza <I42> Playfield 76 <I50> MOD Pizza <I57> Liberty Food & Wine Exchange

- <I33> Sizzle Pie <I28> The Pizza Collective

- <I26> Wild Garlic Pizza & Pub <I54> Wild Garlic <I61> Hidden Pizza <I36> Piezzetta Pizza Kitchen

- <I17> Smiling With Hope Pizza <I5> Pizzeria Lupo <I21> Noble Pie Parlor - Downtown <I30> Noble Pie Parlor - Midtown Reno

- <I12> Eclipse Pizza Co. <I56> Peluso's Apizza <I14> Firetrail Pizza <I62> Pizza Factory <I37> Pizanos Pizza <I9> Pizzava <I25> Semenza's Pizzeria <I46> Paulie's Pizza

- <I44> The Blind Onion Pizza & Pub <I47> The Blind Onion Pizza & Wings <I1> La Cucina Italian Eatery <I51> Calafuria <I19> Monaciello <I27> Casale's Halfway Club
- <I45> Opa Gourmet Pizza Cuisine <I58> Opa Cafe <I59> California Pizza Kitchen <I20> Rick's Pizza, Beer, & More

<I18> Chicago's Pizza With A Twist Reno - S Virginia St, NV<I55> Chicago's Pizza With A Twist - Reno, NV <I41> Taste of Chicago <I60> Chicago Dogs! Eatery <I3> M3 Restaurant <I22> Food + Drink <I39> Rattlesnake Club <I6> Brugos <I7> Huntsman Brewing <I11> Electric Blue Elephant <I34> West Street Market <I49> Boulevard Pizza

- <I16> J J's Pie <I29> Pub N' Sub <I48> The Rack <I31> Little Waldorf Saloon

<I13> Knockouts Bar and Hookah Lounge <I43> Round Table Pizza <I24> Pizza Baron - Reno <I8> Godfather's Pizza <I32> Mountain Mike's Pizza <I4> Yummy's <I35> Cafe Whitney <I15> Gold 'N Silver Inn

- <I52> Atlantis Café Alfresco <I23> Korean Barbie Q <I10> Little Caesars <I38> Pizza Hut
- <I53> Domino's Pizza <I2> Chuck E. Cheese

| |Ground Truth| |
|---|---|---|
| | | |

<I57> Liberty Food & Wine Exchange <I23> Korean Barbie Q <I27> Casale's Halfway Club

- <I17> Smiling With Hope Pizza <I30> Noble Pie Parlor - Midtown Reno

- Figure 9 A failed example of VisualLens on Hit@3. We list the output candidate ranking and ground truth. The ranking follows the same left-right order as the input candidates.

Hyperparameters for training on PaliGemma

Parameter Size 3B Image Resolution 896 × 896 Number of Image Tokens 4096 Hidden Dimension Size 2048 LoRA Rank 16 LoRA α 16 LoRA dropout 0.1 GPU 8 × NVIDIA H100 Batch Size 8 Gradient Accumulation Steps 8 Warmup Steps 200 Learning Rate 1e-3

- Table 8 Hyperparameters for training VisualLens with PaliGemma Backbone.

| |Aspect Word Generation| |
|---|---|---|
| | | |

Here is a list of aspect words for a given image that the user took at a [[Category]]: industrial, rust, pipes, towers, metal, structure, machinery, abandoned, overgrown, bluesky.

What are useful aspects to predict the [[Category]] the user will go after?

Candidates of [[Category]]: <I1> Vulcan Park and Museum, Statue symbolizing city's industries sits atop Red Mountain, surrounded by a park & a museum. <I2> Sloss Furnaces Visitor Center, Blast furnace plant where iron was made from 1882–1971, now an arts & education center with tours. <I3> Historic Cahaba Pumping Station, ...

... Ground Truth: <I2> Sloss Furnaces Visitor Center

Provide the useful aspects to help predict the ground truth, separated by comma: [[Answer]]

###### Figure 10 The prompt template for aspect word generation.

| |Candidate Matching| |
|---|---|---|
| | | |

What is the [[Category]] the user most likely to go to next? Aspect words: industrial, metal, structure, machinery, ... Captions:

- Image 1: A large, rusted metal structure, possibly a piece of industrial machinery or equipment.
- Image 2: Industrial landscape with tall, rusted metal structures under a clear blue sky.

... Candidates: <I1> Vulcan Park and Museum, Statue symbolizing city's industries sits atop Red Mountain, surrounded by a park & a museum. <I2> Sloss Furnaces Visitor Center, Blast furnace plant where iron was made from 1882–1971, now an arts & education center with tours. <I3> Historic Cahaba Pumping Station, ...

... Answer: [[Answer]]

###### Figure 11 The prompt template for candidate matching.

Hyperparameters for training on MiniCPM-V2.5

Parameter Size 8B Image Resolution 980 × 980 Number of Image Tokens 96 Hidden Dimension Size 4096 LoRA Rank 64 LoRA α 64 LoRA dropout 0.1 GPU 8 × NVIDIA H100 Batch Size 8 Gradient Accumulation Steps 8 Warmup Steps 200 Learning Rate 1e-3

- Table 9 Hyperparameters for training VisualLens with MiniCPM-V2.5 Backbone.

