## VISUALOVERLOAD

### Probing Visual Understanding of VLMs in Really Dense Scenes

# arXiv:2509.25339v3[cs.CV]24May2026

Paul Gavrikov1,4 Wei Lin2 M. Jehanzeb Mirza3 Soumya Jahagirdar4 Muhammad Huzaifa4 Sivan Doveh5 Serena Yeung-Levy5 James Glass3 Hilde Kuehne4,6

1Independent Researcher 2JKU Linz 3MIT CSAIL 4Tübingen AI Center 5Stanford 6MIT-IBM Watson AI Lab

#### Abstract

Is basic visual understanding really solved in state-of-the-art VLMs? We present VisualOverload, a slightly different visual question answering (VQA) benchmark comprising 2,720 question–answer pairs, with privately held ground-truth responses. Unlike prior VQA datasets that typically focus on near global image understanding, VisualOverload challenges models to perform simple, knowledge-free vision tasks in densely populated (or, overloaded) scenes. Our dataset consists of high-resolution scans of public-domain paintings that are populated with multiple figures, actions, and unfolding subplots set against elaborately detailed backdrops. We manually annotated these images with questions across six task categories to probe for a thorough understanding of the scene. We hypothesize that current benchmarks overestimate the performance of VLMs, and encoding and reasoning over details is still a challenging task for them, especially if they are confronted with densely populated scenes. Indeed, we observe that even the best model (o3) out of 37 tested models only achieves 19.6% accuracy on our hardest test split and overall 69.5% accuracy on all questions. Beyond a thorough evaluation, we complement our benchmark with an error analysis that reveals multiple failure modes, including a lack of counting skills, failure in OCR, and striking logical inconsistencies under complex tasks. Altogether, VisualOverload exposes a critical gap in current vision models and offers a crucial resource for the community to develop better models.

Benchmark: https://paulgavrikov.github. io/visualoverload

#### 1. Introduction

Visual question answering (VQA) [2, 3, 25] has emerged as a common benchmark for image understanding in VLMs. Recent state-of-the-art models achieve surprisingly strong

[Figure 1]

Figure 1. Example scenes from VisualOverload. All artworks (public domain) in our benchmark show incredibly rich scenes filled with details, which we utilize to test the perception of stateof-the-art VLMs. Please zoom in for details.

results on established VQA datasets [31, 68], suggesting that basic forms of visual understanding might already be “solved”. In turn, several benchmarks have shifted from generic image understanding towards the probing of domainspecific knowledge [53, 69].

However, have today’s VLMs really solved core vision tasks? We argue that current benchmarks are poor indicators of this, as most of them fail to capture the complexity of realworld applications, where safety and reliability depend on fine-grained perception in dense and high-resolution scenes.

[Figure 2]

Activity

What is the person at the center back looking to the right doing?

A. sitting B. playing with animals C. cooking D. ordering

OCR What year is inscribed? (Answer in free form)

Attribute What color are the pants of the person painted on the wall on the right? A. blue B. white C. black D. red

Reasoning

Did the women ﬁnish at least one bottle of wine? Yes/No Did the women drink less than one bottle of wine? Yes/No

###### Scene Is it day? Yes/No Is it night? Yes/No

Counting How many bottles can be seen? (Answer in free form)

Figure 2. Example questions from VisualOverload. Our benchmark consists of images displaying densely populated scenes paired with handcrafted questions (multiple-choice and free-form) covering six core vision tasks. All yes/no questions are paired with questions asking for a logical opposite question to decrease the random chance and to provide an additional signal for measuring logical consistency.

Current benchmarks instead emphasize simple foreground reasoning in low-resolution scenes [31, 35, 68] or utilize higher resolution only for needle-in-a-haystack-like retrieval tasks [32, 54, 64], falling short of testing broader capabilities, and potentially overestimating performance.

Instead, we expect that model performance will severely drop “under pressure”, and modulate this through the angle of visual complexity and dense–visually overloaded–scenes. We motivate our analysis by suggesting that the vision representation and its multi-modal alignment are a bottleneck in modern VLMs. Encoders are designed to compress visual input from spatial to a semantic space, where empirical risk minimization encourages the retention of necessary features. This process imposes an inherent upper bound on fine-grained perception and attention to features that are not commonly seen during training. While random noise illustrates an extreme case of this, we expect sufficiently densely populated scenes to already trigger these limits.

To verify our expectations, we introduce a new dataset explicitly designed to probe image understanding in dense and high-resolution scenes. Our dataset comprises 150 high-resolution scans of artworks featuring highly dense scenes, along with 2,720 manually curated question–answer pairs spanning six fundamental tasks of visual comprehension: activity recognition, attribute recognition, counting, optical character recognition (OCR), visual reasoning, and global scene classification (see Fig. 1 for examples of images and Fig. 2 for example questions about one scene). Unlike many prior benchmarks that recycle existing image datasets, all of our images are newly sourced from public domain artworks, resulting in a fresh source of data and free of copyright concerns.

Our empirical study of 37 VLMs reveals that state-ofthe-art models, while often competent at global scene clas-

sification, consistently struggle in fine-grained recognition in dense scenes. To better characterize these challenges, we split our benchmark into three difficulty levels (easy, medium, hard), calibrated by average model performance. Even the strongest model we tested (o3) achieves only 19.6% accuracy on the hardest split and 69.5% overall, underscoring the difficulty of the benchmark and the underlying challenge.

Finally, we conduct a detailed error analysis and uncover striking failures: for instance, we observe strong failures in counting tasks for high ground-truth values and in OCR tasks requiring precise textual recognition, such as the recognition of typos. Furthermore, we observe that models frequently provide logically inconsistent answers to logically opposite paired questions, with this instability intensifying as the complexity of such queries increases. Such inconsistencies sometimes even degrade performance to random or even subrandom baselines, suggesting that these models rely heavily on shortcuts rather than robust reasoning. Taken together, these findings highlight the urgent need for benchmarks like ours that reflect the realities of dense, high-resolution perception and reveal fundamental limitations of current VLMs.

We summarize our contributions as follows:

• We introduce a new benchmark for VQA in dense, highresolution (visually overloaded) scenes. Our benchmark contains 2,720 manually curated question–answer pairs across six fundamental categories (activity recognition, attribute recognition, counting, OCR, visual reasoning, and global scene classification) as described in Sec. 2. Ground truths are held private to avoid target leakage. All images are sourced entirely from public domain artwork collections to provide a fresh image dataset free of copyright issues.

- • We evaluate a range of state-of-the-art models in Sec. 3 and show that, while they perform well on global scene classification, they struggle significantly in fine-grained understanding in dense settings, particularly for counting and OCR. We provide a three-level difficulty split, calibrated by average model performance, showing that even the strongest tested model (o3) reaches only 19.6% accuracy on the hardest split.
- • We perform a detailed task-level error analysis in Sec. 4, uncovering systematic inconsistencies and shortcut biases that further hinder robust performance in visually overloaded settings.

#### 2. The VisualOverload Benchmark

Our goal is to create a benchmark that tests basic image recognition skills that we expect to be present in any frontier models. However, unlike many previous benchmarks, we design our benchmark around fine-grained recognition in dense scenes to stress test the vision encoders’ representation and their multi-modal alignment. In the following subsections, we discuss the dataset curation (Sec. 2.1), evaluation process (Sec. 2.2), and discuss differences to other benchmarks in detail (Sec. 2.3).

##### 2.1. Dataset Curation

Image collection. We collected 150 high-resolution digitizations of paintings, curated from collections held by museums around the world and made available through Google Arts & Culture (see Fig. 1 for examples). We specifically selected paintings that depict visually complex scenes–densely composed narratives filled with numerous figures, actions, and subplots, often unfolding simultaneously within richly detailed environments. As complexity is hard to quantify [15, 36, 42, 44]1, we picked artworks that tend to overwhelm our eyes and demanded significant time and attention to fully absorb their intricate details. We only selected paintings in the public domain, automatically granted for artworks where the original creators passed away more than 100 years ago [56].

Due to the inherent complexity of the scenes, the images in the dataset are typically of extreme resolution and exceed 4K resolution (3840 × 2160 pixels). To standardize the dataset, we downsampled all images to match the nearest number of pixels in 4K, while preserving their original aspect ratios. 28 images were originally below 4K resolution and were therefore not downsampled; however, all images remain above Full HD resolution (1920 × 1080 pixels).

Question annotation. Six human annotators manually annotated the resized images with questions and answer options. The annotators were instructed to generate questions that are

1In Appendix A we show an experiment to estimate the complexity of all samples in our dataset via VLMs.

clearly formulated and specific, leaving no ambiguity about the information being requested. To avoid language priors, the questions are also explicitly mandated to be grounded in the content of the accompanying image and should not be answered from text alone [1, 2, 9, 25, 70]. In addition, we restricted questions to probe for details that can be directly observed or reasonably inferred from the image, excluding any question–answer pairs based on beliefs or subjective interpretations. Finally, we requested questions to be solvable without external or expert knowledge beyond a basic level of everyday “world” knowledge, as we are only concerned with image understanding in this work (unlike prior art-based benchmarks [6, 17]).

We employ two answer formats: multiple-choice and freeform. Multiple-choice questions either offer four options, with only one correct answer, or are binary yes/no questions. We pair each of the latter kind of questions with a logical opposite (e.g., "Is it day?" and "Is it night?") [70]. This not only helps calibrate against random guessing but also provides an additional signal for identifying logical inconsistencies in generated responses (see Sec. 4). For selected tasks, we use freeform answers to raise the level of difficulty (see below).

Our annotated questions each fall into one of the following six categories, resulting in approximately 18 questions per image:

- • Activity recognition (N=150): multiple-choice questions about actions or activities occurring in the scene. These questions will refer to a single or a group of subjects, typically paired with a constraint. For instance, "What is the person dressed in brown at the front of the table in the leftmost house doing?".
- • Attribute recognition (N=149): multiple-choice queries about the color attribute of objects are typically paired with a constraint probing for spatial, attribute, or activity recognition. For instance, "What is the color of the leftmost ship flag?".
- • Counting (N=559): freeform inquiries about details that involve determining the number of objects present. The questions may be related to the entire scene or spatially constrained, requiring mild visual reasoning to provide a correct answer. For instance, "How many roses are lying on the floor?".
- • OCR (N=118): freeform queries about written text in the image. Languages include English, Latin, Chinese, Dutch, and Greek. Some questions are specified to probe for parts of the text, which can be seen as a mild form of text reasoning, e.g., "What is the last name of the signature?", or require some minimal visual reasoning efforts, e.g., "What does the word below the main character read?".
- • Reasoning (N=356): multiple-choice queries that require a medium to high load of visual reasoning to be answerable. In principle, we expect that a "chain of thought" is

necessary to provide a correct answer. For instance, these questions may require functional or intent understanding, distance or path estimation, light- or wind-source estimation, occupancy detection, and numerical comparisons based on the image’s content. Some example questions are: "Do you have to cross the water to reach the two windmills on the right?", "I am allergic to seafood, is all of the food on the table safe for me?", or "Does capital punishment appear to be legal in this scene?".

• Scene classification (N=1388): multiple-choice queries about the overall scene or setting of the image. These questions typically do not require a fine-grained understanding or complex visual reasoning of the scene. Thus, we primarily see these samples as a test for shallow image understanding, and we expect all models to perform well on them. Yet, we still observe that some models struggle with them. For instance, "Are there animals in the scene?".

Quality control. After annotation, we evaluated 37 VLMs on our dataset and manually verified the correctness of ground truths if the question was only solved by a small number of models. Furthermore, we evaluated the performance of 3 of the strongest models from our leaderboard (InternVL3-38B, Qwen2.5-VL 32B, LLaVA-OV 72B) on our dataset while ablating the image (i.e., a blind question answering) to probe for hidden biases due to linguistic cues in the question or answer options of multiple-choice questions. We detected a number of questions where all 3 models were able to answer the question without seeing the image. We then prompted Gemini 2.5 Pro to detect potential language biases in each question (see Appendix E for the prompt) and removed instances with severe biases, such as cases where the correct answer was an oddity or was implied by the context of the question. Please note that this is not necessary for freeform answers (counting, OCR) or binary questions, which are self-balanced by their logical opposites. The final “blind” performance on the remaining questions is shown in Appendix Tab. 2. Overall, our quality control resulted in a reduction of blind performance to near-chance baselines for most tasks. However, we still observe above-random performance for the attribute recognition and counting tasks. These gains stem primarily from statistical irregularities in the distribution of ground-truth answers (e.g., small object counts being more frequent). Such distributional priors are unavoidable in real-world datasets and do not confer a generalizable shortcut that undermines evaluation. In practice, models must still extract and process visual content to achieve strong performance on all of our tasks.

Difficulty splits. We divide our questions into three difficulty levels–easy, medium, and hard–based on the ratio of correct responses by all models described in Sec. 3. The thresholds are defined by the percentage of correct responses per question: [0,20] for hard, (20,90) for medium, and [90,100] for

easy.

##### 2.2. Evaluation Process

Metrics. We rely on the average accuracy as the principal metric for our benchmark, scored over all questions, each difficulty split, as well as each task category. We define an answer as accurate if it precisely matches the ground truth label. For binary questions, we measure pair-wise accuracy, and score a pair as correct if both questions are correct, and false otherwise.

Answer extraction. Although our default prompts (see Appendix C) aim to constrain output format, VLMs do not always follow these instructions. To address this, we apply simple heuristic-based preprocessing to extract and normalize responses across tasks.

For multiple-choice questions, we detect the option letter and map it to the corresponding label, or directly match the label when possible. For counting questions, we extract either numeral or lexical integer forms, defaulting to the last-mentioned integer if multiple candidates appear. For OCR tasks, we extract the relevant text, then normalize it by removing diacritics, punctuation, and spacing, converting to lowercase, and replacing ‘V’ with ‘U’ and ‘J’ with ‘I’ to reduce ambiguity in Latin texts.

Evaluation server. To prevent test leakage into future VLMs, we hold out the ground truth and only release the image samples and questions. We do not provide a development split, as our tasks do not require any specialized knowledge or skills, and we expect decent foundational vision models to solve these tasks without finetuning. Instead, we provide an evaluation server that scores generated answers, and we maintain an opt-in leaderboard of those. Evaluations are made by submitting a JSON file with model predictions to our public evaluation server. The server applies our extraction heuristics as outlined above, but users are free to apply their own preprocessing of any kind before submitting their predictions. We rate-limit the server per user and day to prevent ground-truth extraction attacks.

##### 2.3. Comparison With Existing Benchmarks

Many existing VQA datasets underestimate the true difficulty of visual reasoning and may probe only for shallow pattern matching rather than genuine scene understanding. Furthermore, they often rely on low-resolution images, recycled content, and automatically generated questions. Our benchmark is intentionally designed to correct these shortcomings and to set a higher standard for evaluation. Its distinguishing features are:

• Dense scenes in high-resolution images. We collect detailed images of complex scenes, enabling questions that demand fine-grained perception and long-range reasoning. Unlike prior benchmarks, which often reduce vision to global features, our dataset forces models to engage with

the full richness of the scene–naturally, this is correlated with high-resolution.

- • Manual annotation. All questions are crafted by human annotators. Automated pipelines used in other datasets may scale cheaply, but they also introduce biases, trivial patterns, and low-quality queries. Our human-centered approach ensures natural, challenging, and unbiased evaluation.
- • Fresh image data. Rather than recycling existing dataset sources, we provide entirely new images. This prevents leakage from pretraining corpora and eliminates the domain biases that plague benchmarks built from reused datasets.
- • Public domain licensing. Every image is sourced from the public domain, removing legal barriers that limit distribution or usage. Unlike benchmarks with restrictive or unclear licensing due to web crawling, ours is openly and universally accessible.

In sum, where existing benchmarks compromise on difficulty, reliability, or ethics, our dataset sets a new bar: more challenging, more trustworthy, and more responsible.

#### 3. Experiments

In the following subsections, we evaluate the performance of different VLMs on VisualOverload. In Sec. 3.1 we introduce the models, and assess their performance in Sec. 3.2.

##### 3.1. Baselines

We evaluate 37 recent VLMs, including variously sized openweight models ranging from 450M to 109B parameters, designed for low- and high-resolution image understanding, that we separate into three parameter bands, specialized highresolution understanding models, and 4 proprietary frontier models. To simplify the answer extraction, we add small postfixes to the benchmark questions outlined in Appendix C. We generate answers using greedy decoding for all models, except for proprietary models and models where greedy decoding failed to generate useful outputs (e.g., Llama 4), as highlighted in the result tables.

Additionally, we compare the results to random chance (we assume no priors for counting and OCR), as well as consistent chance, where we assume that a model is guessing, but gives consistent guesses for logically opposite questions.

##### 3.2. Main Results

The results in Tab. 1 show vast differences between models and some of the tasks in VisualOverload. First off, we notice that all models struggle with our freeform counting and OCR tasks. The best accuracy in counting is achieved by Gemini 2.0 Flash, but is only at 41.7%. OCR performance is overall better, but even the best model, o4-mini, only achieves 62.7%. This is also the task with the highest discrepancy

between proprietary commercial models and open-weight ones.

For activity and attribute recognition, we see an improved accuracy (albeit at a higher random chance), but still far from satisfactory performance, even with the strongest models. For reasoning tasks, we find that almost all models struggle and make rather small improvements compared to the consistent random chance, while some of the smaller models even underperform it. The only positive outlier here is o3, which achieves a significant advantage compared to other models, presumably due to its reasoning mode. Unsurprisingly, we find that frontier models achieve a high accuracy on scene understanding, as it primarily relies on a superficial understanding of the scenes, as is common in many of the existing VQA datasets. However, rather surprisingly, the task can still be challenging for many other models, even for large ones. Yet, 8B parameters seem already to be sufficient to achieve 93.4%. In a few cases, the accuracy even fell below a consistent chance, suggesting a fallback to shortcut features (see also Sec. 4).

Averaged over all tasks, the best model (o3) achieves only 19.6% on the hardest test split, and 69.5% overall. The strongest open-weight model is InternVL3 38B with 7.2% and 67.6%, respectively. Interestingly, we found that specialized HD models perform significantly worse than equally sized regular models. We attribute this primarily to the fact that most VLMs already apply methodologies such as AnyRes [40] to support high-resolution images and, thus, performance is rather dependent on the backbones and training, therefore showing that modern VLMs outperform specialized VLMs built on older backbones2. Finally, we also find some counter-intuitive scaling trends, where performance decreases with parameter size (often for the largest model of the family, i.e., in InternVL3 and PaliGemma 2).

We encourage the community to explore advanced prompting techniques and invite them to submit these to our leaderboard.

#### 4. Error Analysis

In this section, we aim to better outline the errors that models make. With the protection of our private ground-truth in mind, we will rely on average statistics over all models described in Sec. 3.1.

Counting. To analyze errors in counting tasks, we plot the distribution of predictions versus ground truths in Fig. 3. Models are generally accurate when the ground truth is low, but errors increase substantially as the ground truth rises. Although some errors stem from incorrect predictions, many are also due to refusals (which we treat as 0) or blank responses (e.g., “too many objects to count”). In all cases,

2Please find an ablation of performance under different resolutions of VisualOverload in Appendix D.

Table 1. Benchmark results. We report the accuracy as the fraction of correct responses after processing, including the accuracy normalization for binary questions for each of the categories in our benchmark, as well as the average. For counting and OCR, we additionally report the RMSE and Normalized Levenshtein Edit Distance, respectively. Legend: S Completions were generated using stochastic sampling at default parameters.

Params Activity Attributes Counting OCR Reasoning Scene Easy Medium Hard Total Model [B] (150) (149) (559) (118) (356) (1388) (986) (1304) (430) (2720)

Random Chance - 25.0 25.0 0.0 0.0 25.0 25.0 24.5 16.7 3.7 16.0 Consistent Chance - 25.0 25.0 0.0 0.0 42.5 50.0 47.2 26.2 4.7 27.2

Small Open-Weight Models (< 7B)

PaliGemma 2 3B [55] 3.0 42.0 53.0 20.4 (8.5) 8.5 (0.69) 24.9 32.7 51.9 28.3 5.0 29.0 LLaVA 1.5 7B [38] 7.0 35.3 43.6 13.2 (8.2) 3.4 (0.76) 39.5 43.2 69.7 24.6 1.9 30.8 Gemma 3n E2B [24] 5.0 32.0 26.2 15.0 (12.2) 19.5 (0.56) 35.6 53.2 74.6 25.7 7.9 33.9 LLaVA-NeXT 7B [40] 7.0 44.7 41.6 19.1 (8.6) 8.5 (0.66) 40.5 54.0 81.8 31.5 2.2 37.5 LFM2 VL 450M [37] 0.4 35.3 47.0 22.9 (7.1) 20.3 (0.57) 27.8 59.5 83.1 32.4 8.6 39.7 DeepSeek VL2 Tiny [65] 1.0 54.7 47.7 22.5 (8.4) 35.6 (0.43) 37.1 54.2 82.5 38.0 2.6 41.2 SmolVLM [43] 2.0 42.7 41.6 17.2 (7.1) 28.0 (0.47) 32.2 67.3 83.5 38.8 3.1 42.0 Gemma 3n E4B [24] 5.0 40.0 23.5 19.3 (8.1) 23.7 (0.54) 41.0 73.9 87.8 38.4 8.9 44.2

- InternVL3 1B [76] 1.0 48.0 57.0 27.2 (6.2) 25.4 (0.52) 35.1 77.5 94.9 48.9 5.0 50.6 LFM2 VL 1.6B [37] 1.6 49.3 55.7 25.2 (6.2) 28.0 (0.51) 44.4 79.5 97.4 50.4 4.8 51.9 Qwen2.5-VL 3B [5] 3.0 60.7 61.7 25.9 (6.1) 49.2 (0.33) 43.9 77.5 94.0 56.0 4.8 54.1

- InternVL3 2B [76] 2.0 50.0 58.4 30.4 (8.0) 39.0 (0.43) 49.8 80.3 98.9 55.6 5.7 55.3 DeepSeek VL2 [65] 4.5 65.3 63.8 25.9 (8.2) 46.6 (0.35) 58.5 81.8 99.4 60.6 4.1 57.7

Medium Open-Weight Models (7–13B)

LLaVA-OV 7B [33] 7.0 60.7 57.7 28.4 (6.1) 29.7 (0.45) 54.1 88.2 95.5 63.6 4.3 58.3 Qwen2.5-VL 7B [5] 7.0 63.3 69.1 34.9 (5.5) 55.9 (0.27) 49.8 85.3 97.9 66.2 9.6 61.5 LLaVA 1.5 13B [38] 13.0 41.3 39.6 13.8 (8.1) 3.4 (0.72) 42.9 71.6 94.0 34.0 2.6 42.0 LLaVA-NeXT 13B [40] 13.0 44.0 43.6 17.0 (8.3) 6.8 (0.66) 41.5 75.8 97.4 38.1 2.9 45.1 Gemma 3 12B [23] 12.0 48.7 42.3 16.5 (9.6) 31.4 (0.47) 47.8 82.7 98.3 45.6 6.2 50.0 PaliGemma 2 10B [55] 10.0 48.7 52.3 23.6 (7.3) 5.1 (0.75) 42.4 81.8 91.9 49.5 5.7 50.3 InternVL3 8B [76] 8.0 66.0 67.8 32.2 (5.6) 42.4 (0.37) 59.0 93.4 99.6 70.8 7.9 63.9

Large Open-Weight Models (> 13B)

PaliGemma 2 28B [55] 28.0 40.0 49.0 17.4 (9.2) 5.9 (0.73) 40.0 66.1 81.2 37.7 6.0 41.5 Gemma 3 27B [23] 27.0 51.3 46.3 18.1 (9.1) 40.7 (0.41) 50.7 86.3 98.5 50.6 8.9 53.2 Llama 4 ScoutS [47] 109.0 58.7 65.8 31.1 (4.2) 37.3 (0.44) 62.0 78.8 95.7 57.9 13.6 57.5 InternVL3 14B [76] 14.0 66.7 69.1 30.6 (5.1) 41.5 (0.42) 57.1 91.1 98.5 69.7 5.3 62.5 LLaVA-OV 72B [33] 72.0 66.0 69.8 30.9 (6.0) 39.0 (0.41) 57.1 91.8 97.6 71.0 4.1 62.7 Qwen2.5-VL 32B [5] 32.0 60.0 70.5 30.8 (4.7) 61.0 (0.23) 61.5 90.3 98.5 68.7 12.4 63.6 Qwen2.5-VL 72B [5] 72.0 67.3 74.5 35.1 (5.7) 72.9 (0.16) 53.2 90.5 97.6 72.6 13.4 65.7 InternVL3 78B [76] 78.0 78.0 80.5 34.7 (5.0) 31.4 (0.52) 65.4 93.7 97.6 76.9 8.1 66.8 InternVL3 38B [76] 38.0 76.7 78.5 35.4 (5.3) 45.8 (0.33) 69.8 92.2 98.3 78.6 7.2 67.6

Specialized High-Resolution Models

VILA HD 4KS [54] 8.0 54.0 48.3 22.5 (6.3) 11.0 (0.73) 49.3 74.5 91.2 47.1 4.1 48.5 VILA HD 1.5KS [54] 8.0 54.0 57.7 25.9 (6.5) 21.2 (0.52) 52.2 79.4 94.2 54.3 4.1 53.1 ILM-XC2-4KHD [12] 7.0 50.7 53.7 25.4 (5.8) 31.4 (0.46) 42.4 83.6 94.4 53.8 6.7 53.4 ILM-XC2.5 [71] 7.0 48.0 51.7 22.7 (6.0) 35.6 (0.46) 45.9 87.3 95.9 53.7 9.1 54.3

Proprietary Models

Horizon AlphaS [26] – 57.3 74.5 35.6 (5.4) 48.3 (0.36) 63.9 93.2 99.4 72.9 10.8 65.7 Gemini 2.0 Flash – 76.0 71.1 41.7 (5.8) 57.6 (0.26) 56.6 92.1 99.1 74.0 19.1 68.1

- o4-miniS [51] – 70.0 76.5 38.3 (4.3) 62.7 (0.22) 67.8 93.7 98.1 77.4 17.2 69.1

- o3S [51] – 74.0 69.8 36.7 (4.0) 61.0 (0.24) 75.1 94.7 99.4 76.4 19.6 69.5

150

Prediction

100

correct

50

0

0 10 20 30 40 50 60 70 80 Ground Truth

###### Figure 3. Counting prediction vs. ground truths.

0.15

Probability

0.10

0.05

0.00

0.0 0.2 0.4 0.6 0.8 1.0 Normalized Levenshtein Distance

Figure 5. OCR prediction error distance.

| | |
|---|---|
| | |
| | |
| | |
| | |
| | |

100

misinterpretation of text flow, such as side-by-side multi-line paragraphs or non-standard layouts like banners. For errors with low edit distance, we frequently observe that models’ auto-correct spelling or generally fall back to more probable token sequences rather than reproducing the actual text (e.g., “accidunt” becomes “accident”), particularly in non-English or non-Latin scripts.

80

Accuracy(%)

60

40

Logical consistency. As described in Sec. 2.1, our dataset contains binary questions, where each such question is paired with a logical opposite. A strong model should argue logically consistently, even if the answer is wrong. For instance, if a model answers “yes” to “Is it day?” it should answer “no” to “Is it night?”. We measure the ratio of logically consistent answer pairs per model and task (reasoning and global scene understanding) and visualize the results in Fig. 6.

20

0 100 200 300 400 Allowed Error (%)

Figure 4. Counting accuracy under tolerance (mean±std).

We observe that frontier models answer fairly logically consistent for the easier scene questions, but their performance rapidly drops on the harder reasoning questions. On average, consistency falls from 83.3% or 60.6% between the tasks. For some models, a well-above-chance consistency drops a near-random baseline for reasoning, suggesting that models are now guessing independently of the context, while providing well-grounded answers on the original task. In some cases, we also find a well-below random chance consistency, suggesting that the model is relying on shortcuts for shortcuts rather than the visual inputs. Alarmingly, we find PaliGemma2 3B to be susceptible to these for both tasks.

models tend to err on the low side and underestimate the ground truth. Yet, our analysis also contained outliers showing severe overestimation.

To quantify the magnitude of these errors, we measured accuracy under varying tolerance levels, shown in Fig. 4. Prediction errors are typically severe: even with a 10% tolerance, average accuracy improves by only 1.6%. Larger tolerances, such as 50% or 100%, yield more substantial improvements, but such levels are impractical for real-world applications.

OCR. Similar to counting, we aim to quantify the magnitude of errors in OCR predictions. To do this, we measure the Levenshtein edit distance [30] between preprocessed predictions (as described in Sec. 2.2) and ground truths for incorrect answers. We normalize the distance by the maximum sequence length and visualize the distribution in Fig. 5. The distribution’s center of mass is around 0.7, indicating that sequences require substantial edits to be correct, highlighting severe errors.

#### 5. Related Work

Large multi-modal models. Recent progress in VLMs has significantly advanced the integration of visual and linguistic modalities, enabling more sophisticated multi-modal understanding and generation. Early approaches connect pretrained vision encoders with large language models via lightweight modules, achieving competitive performance with relatively few trainable parameters [34, 72, 75]. The LLaVA series [33, 38–40] improves visual instruction tuning, demonstrating stronger performance on fine-grained

Manual inspection of a subset of errors reveals three main causes: hallucinations, extraction of irrelevant text, and, in a few severe cases, failure to follow the instruction to respond only with the text. Errors of the second type often arise from

1.0

Task

Scene Reasoning

0.8

LogicalConsistency

0.6

Random

0.4

0.2

0.0

PaliGemma210B

PaliGemma228B

PaliGemma23B

Qwen2.5-VL32B

Qwen2.5-VL72B

Qwen2.5-VL7B

Qwen2.5-VL3B

o3

HorizonAlpha

SmolVLM

LFM2VL450M

o4mini

Llama4Scout

Gemma3nE4B

Gemma3nE2B

Gemma327B

Gemma312B

InternLM-XC2-4KHD

LFM2VL1.6B

InternLM-XC2.5

DeepSeekVL2

LLaVA-OV72B

LLaVA-OV7B

Gemini2.0Flash

LLaVA1.513B

LLaVA1.57B

VILAHD4K

VILAHD1.5K

LLaVA-NeXT13B

LLaVA-NeXT7B

InternVL378B

InternVL314B

InternVL338B

InternVL38B

InternVL32B

InternVL31B

DeepSeekVL2Tiny

Figure 6. Logical consistency on scene understanding and reasoning.

visual tasks. More recent models extend these capabilities to multi-image contexts, enabling richer scene understanding and more coherent textual reasoning [5, 23, 33, 55, 76]. Proprietary VLMs, including GPT, the o-series [50, 51], and Gemini [21, 22], further highlight progress in versatile, context-aware multi-modal learning frameworks, sometimes even including multi-modal reasoning [51].

Despite these advancements, many VLMs still exhibit notable weaknesses in visual understanding [16, 61]. Prior work has shown that they struggle with counting, spatial reasoning, concept binding, and dense scene understanding [10, 13, 14, 28, 52, 66, 67] and even image classification tasks [48, 49, 73]. In our work, we build on these findings by introducing a benchmark of densely populated publicdomain paintings, designed to probe such vulnerabilities and evaluate the capacity of VLMs to perform basic visual tasks in more challenging, visually overloaded scenes.

Multi-modal vision benchmarks. The rapid progress of VLMs has spurred a surge of benchmarks evaluating their ability to integrate vision and language across tasks such as VQA, captioning, reasoning, and instruction following. Extending classic VQA datasets [3, 25], modern benchmarks vary in scope, from real-world instruction following in VisitBench [7] to conversational reasoning in LLaVA-Bench [8], zero-shot capability assessment across 16 capabilities, including OCR and spatial reasoning in MMVet [68], and multiple-choice probing in 12 dimensions in SeedBench [31]. Broader frameworks such as MM-Bench [41], TouchStone [4], OmniBench [35], and MMStar [11] aim for holistic multi-modal evaluation by covering a wide array of tasks and domain-specific knowledge. MMMU [69] pushes toward expert-level multi-modal reasoning. As performance

on most of these benchmarks seems to saturate, more carefully designed benchmarks [27, 28, 59, 63] reveal persistent weaknesses in multiple dimensions, highlighting a discrepancy between many seemingly positive benchmark results and actual visual capabilities.

While these efforts nonetheless provide valuable insights, most emphasize global understanding, a very broad task coverage, or require domain-specific expertise, while often overlooking basic perception in more challenging settings, such as visually overloaded scenes. Recently, multiple benchmarks started the exploration of small details in high-resolution scenes [32, 54, 64] or documents [45, 57], showing another hurdle in the development of vision models. Our work complements these benchmarks with VisualOverload, a human-annotated dataset of VQA pairs grounded in high-resolution, but also densely populated artworks. A key differentiator of high-resolution benchmarks is that VisualOverload aims at exploiting the full complexity of the scene, while previous works mostly model needle-in-thehaystack-style retrieval of small details. By focusing on six basic tasks in overloaded scenes, VisualOverload reveals systematic error modes in state-of-the-art open and proprietary VLMs, highlighting critical gaps even in basic, knowledgefree visual understanding.

#### 6. Conclusion

In this work, we introduced VisualOverload, a novel VQA benchmark designed to expose the limitations of state-ofthe-art VLMs in complex, detail-rich scenes. Our findings demonstrate that while these models perform well on global tasks, they consistently struggle with simple, fine-grained questions within visually overloaded environments. This performance gap highlights a critical area for future research, suggesting that the problem of fundamental visual understanding is far from solved. Ultimately, our dataset offers a crucial resource for the community to develop more robust and perceptive VLMs.

Limitations. Our findings are based on a dataset composed exclusively of art, which means they may not directly generalize to benchmarks of natural images. Artistic representations often diverge significantly from photorealism, emphasizing abstractness over detail [60]. This introduces a unique set of challenges: while models trained on natural images may excel at texture-based recognition [18, 20], they often struggle with the high stylistic variance and strong reliance on global shape inherent in art. We therefore consider this focus a feature, not a drawback. We posit that a truly foundational model, aiming to match or exceed humanlevel perception, must also be robust to these variations and comprehend representations just as humans do. We further elaborate on this in Appendix B.

#### References

- [1] Aishwarya Agrawal, Dhruv Batra, and Devi Parikh. Analyzing the Behavior of Visual Question Answering Models. arXiv preprint arXiv:1606.07356, 2016.
- [2] Aishwarya Agrawal, Dhruv Batra, Devi Parikh, and Aniruddha Kembhavi. Don’t Just Assume; Look and Answer: Overcoming Priors for Visual Question Answering. In Proceedings of the Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
- [3] Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh. VQA: Visual Question Answering. In Proceedings of the International Conference on Computer Vision (ICCV), 2015.
- [4] Shuai Bai, Shusheng Yang, Jinze Bai, Peng Wang, Xingxuan Zhang, Junyang Lin, Xinggang Wang, Chang Zhou, and Jingren Zhou. Touchstone: Evaluating vision-language models by language models. arXiv preprint arXiv:2308.16890, 2023.
- [5] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923, 2025.
- [6] Federico Becattini, Pietro Bongini, Luana Bulla, Alberto Del Bimbo, Ludovica Marinucci, Misael Mongiovì, and Valentina Presutti. VISCOUNTH: A Large-scale Multilingual Visual Question Answering Dataset for Cultural Heritage. ACM Trans. Multimedia Comput. Commun. Appl., 19(6), 2023.
- [7] Yonatan Bitton, Hritik Bansal, Jack Hessel, Rulin Shao, Wanrong Zhu, Anas Awadalla, Josh Gardner, Rohan Taori, and Ludwig Schmidt. Visit-bench: A benchmark for visionlanguage instruction following inspired by real-world use. arXiv preprint arXiv:2308.06595, 2023.
- [8] Florian Bordes, Richard Yuanzhe Pang, Anurag Ajay, Alexander C Li, Adrien Bardes, Suzanne Petryk, Oscar Mañas, Zhiqiu Lin, Anas Mahmoud, Bargav Jayaraman, et al. An introduction to vision-language modeling. arXiv preprint arXiv:2405.17247, 2024.
- [9] Remi Cadene, Corentin Dancette, Hedi Ben younes, Matthieu Cord, and Devi Parikh. RUBi: Reducing Unimodal Biases for Visual Question Answering. In Advances in Neural Information Processing Systems (NeurIPS), 2019.
- [10] Declan Campbell, Sunayana Rane, Tyler Giallanza, Nicolò De Sabbata, Kia Ghods, Amogh Joshi, Alexander Ku, Steven M. Frankland, Thomas L. Griffiths, Jonathan D. Cohen, and Taylor Webb. Understanding the Limits of Vision Language Models Through the Lens of the Binding Problem. In Advances in Neural Information Processing Systems (NeurIPS), 2024.
- [11] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? Advances in Neural Information Processing Systems (NeurIPS), 37, 2024.
- [12] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Songyang Zhang, Haodong Duan, Wenwei Zhang, Yining Li, et al. Internlm-xcomposer2-4khd: A pioneering large vision-language model handling resolutions

- from 336 pixels to 4k hd. Advances in Neural Information Processing Systems (NeurIPS), 37, 2024.
- [13] Sivan Doveh, Assaf Arbelle, Sivan Harary, Roei Herzig, Donghyun Kim, Paola Cascante-Bonilla, Amit Alfassy, Rameswar Panda, Raja Giryes, Rogerio Feris, et al. Dense and aligned captions (dac) promote compositional reasoning in vl models. Advances in Neural Information Processing Systems (NeurIPS), 36, 2023.
- [14] Sivan Doveh, Assaf Arbelle, Sivan Harary, Eli Schwartz, Roei Herzig, Raja Giryes, Rogerio Feris, Rameswar Panda, Shimon Ullman, and Leonid Karlinsky. Teaching structured vision & language concepts to vision & language models. In Proceedings of the Conference on Computer Vision and Pattern Recognition (CVPR), 2023.
- [15] A. Forsythe, M. Nadal, N. Sheehy, C. J. Cela-Conde, and M. Sawey. Predicting beauty: Fractal dimension and visual complexity in art. British Journal of Psychology, 102(1), 2011.
- [16] Xingyu Fu, Yushi Hu, Bangzheng Li, Yu Feng, Haoyu Wang, Xudong Lin, Dan Roth, Noah A. Smith, Wei-Chiu Ma, and Ranjay Krishna. BLINK: Multimodal Large Language Models Can See but Not Perceive. In Proceedings of the European Conference on Computer Vision (ECCV), 2025.
- [17] Noa Garcia, Chentao Ye, Zihua Liu, Qingtao Hu, Mayu Otani, Chenhui Chu, Yuta Nakashima, and Teruko Mitamura. A Dataset and Baselines for Visual Question Answering on Art. In Proceedings of the European Conference on Computer Vision Workshops (ECCVW), 2020.
- [18] Paul Gavrikov, Jovita Lukasik, Steffen Jung, Robert Geirhos, Muhammad Jehanzeb Mirza, Margret Keuper, and Janis Keuper. Can We Talk Models Into Seeing the World Differently? In International Conference on Learning Representations (ICLR), 2025.
- [19] Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé III, and Kate Crawford. Datasheets for datasets. Commun. ACM, 64

(12), 2021.

- [20] Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A. Wichmann, and Wieland Brendel. ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness. In International Conference on Learning Representations (ICLR), 2019.
- [21] Gemini Team. Gemini: A Family of Highly Capable Multimodal Models. arXiv preprint arXiv:2312.11805, 2024.
- [22] Gemini Team. Gemini 2.0 Flash Model Card, 2025. [Online; accessed 28. Aug. 2025].
- [23] Gemma Team. Gemma 3 Technical Report. arXiv preprint arXiv:2503.19786, 2025.
- [24] Lucas Gonzalez and Rakesh Shivanna. Announcing Gemma 3n preview: powerful, efficient, mobile-first AI, 2025. [Online; accessed 28. Aug. 2025].
- [25] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering. In Proceedings of the Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

- [26] Horizon Alpha Team. Horizon Alpha - Advanced AI Language Model, 2025. [Online; accessed 28. Aug. 2025].
- [27] Cheng-Yu Hsieh, Jieyu Zhang, Zixian Ma, Aniruddha Kembhavi, and Ranjay Krishna. Sugarcrepe: Fixing hackable benchmarks for vision-language compositionality. Advances in Neural Information Processing Systems (NeurIPS), 36, 2023.
- [28] Irene Huang, Wei Lin, Muhammad Jehanzeb Mirza, Jacob Hansen, Sivan Doveh, Victor Butoi, Roei Herzig, Assaf Arbelle, Hilde Kuehne, Trevor Darrell, et al. Conme: Rethinking evaluation of compositional reasoning for modern vlms. Advances in Neural Information Processing Systems (NeurIPS), 37, 2024.
- [29] Takeshi Kojima, Shixiang (Shane) Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large Language Models are Zero-Shot Reasoners. In Advances in Neural Information Processing Systems (NeurIPS), 2022.
- [30] Vladimir Iosifovich Levenshtein. Dvoichnye kody s ispravleniem vypadenii, vstavok i zameshchenii simvolov. Doklady Akademii Nauk SSSR, 163(4), 1965.
- [31] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023.
- [32] Bo Li, Peiyuan Zhang, Jingkang Yang, Yuanhan Zhang, Fanyi Pu, and Ziwei Liu. Otterhd: A high-resolution multi-modality model. arXiv preprint arXiv:2311.04219, 2023.
- [33] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. LLaVA-OneVision: Easy Visual Task Transfer. arXiv preprint arXiv:2408.03326, 2024.
- [34] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In Proceedings of the International Conference on Machine Learning (ICML),

- 2023.

[35] Yizhi Li, Ge Zhang, Yinghao Ma, Ruibin Yuan, Kang Zhu, Hangyu Guo, Yiming Liang, Jiaheng Liu, Zekun Wang, Jian Yang, et al. Omnibench: Towards the future of universal omni-language models. arXiv preprint arXiv:2409.15272,

- 2024.

[36] Kylie Lin, Sean Sheng-tse Ru, David N. Rapp, Hui Guan, and Cindy Xiong Bearfield. What Makes a Visualization Visually Complex? In Proceedings of the Extended Abstracts of the CHI Conference on Human Factors in Computing Systems,

- 2025.

- [37] Liquid AI. Introducing LFM2: The Fastest On-Device Foundation Models on the Market | Liquid AI, 2025. [Online; accessed 28. Aug. 2025].
- [38] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023.
- [39] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in Neural Information Processing Systems (NeurIPS), 36, 2023.
- [40] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. LLaVA-NeXT: Improved reasoning, OCR, and world knowledge, 2024.

- [41] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an allaround player? In Proceedings of the European Conference on Computer Vision (ECCV). Springer, 2024.
- [42] Louis Mahon and Thomas Lukasiewicz. Minimum Description Length Clustering to Measure Meaningful Image Complexity. arXiv preprint arXiv:2306.14937, 2023.
- [43] Andrés Marafioti, Orr Zohar, Miquel Farré, Merve Noyan, Elie Bakouch, Pedro Cuenca, Cyril Zakka, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, Vaibhav Srivastav, Joshua Lochner, Hugo Larcher, Mathieu Morlon, Lewis Tunstall, Leandro von Werra, and Thomas Wolf. SmolVLM: Redefining small and efficient multimodal models. arXiv preprint arXiv:2504.05299, 2025.
- [44] Manuela M. Marin and Helmut Leder. Examining Complexity across Domains: Relating Subjective and Objective Measures of Affective Environmental Scenes, Paintings and Music. PLOS ONE, 8(8), 2013.
- [45] Minesh Mathew, Dimosthenis Karatzas, and C.V. Jawahar. DocVQA: A Dataset for VQA on Document Images. In IEEE Winter Conference on Applications of Computer Vision (WACV), 2021.
- [46] Leland McInnes, John Healy, Nathaniel Saul, and Lukas Großberger. UMAP: Uniform Manifold Approximation and Projection. Journal of Open Source Software, 3(29), 2018.
- [47] Meta AI. The Llama 4 herd: The beginning of a new era of natively multimodal AI innovation, 2025. [Online; accessed

28. Aug. 2025].

- [48] Muhammad Jehanzeb Mirza, Leonid Karlinsky, Wei Lin, Horst Possegger, Mateusz Kozinski, Rogerio Feris, and Horst Bischof. LaFTer: Label-Free Tuning of Zero-shot Classifier using Language and Unlabeled Image Collections. Advances in Neural Information Processing Systems (NeurIPS), 36, 2023.
- [49] Muhammad Jehanzeb Mirza, Mengjie Zhao, Zhuoyuan Mao, Sivan Doveh, Wei Lin, Paul Gavrikov, Michael Dorkenwald, Shiqi Yang, Saurav Jha, Hiromi Wakaki, Yuki Mitsufuji, Horst Possegger, Rogerio Feris, Leonid Karlinsky, and James R. Glass. GLOV: Guided Large Language Models as Implicit Optimizers for Vision Language Models. Transactions on Machine Learning Research (TMLR), 2025.
- [50] OpenAI. GPT-4 Technical Report. arXiv preprint arXiv:2303.08774, 2024.
- [51] OpenAI. OpenAI o3 and o4-mini System Card, 2025. [Online; accessed 28. Aug. 2025].
- [52] Roni Paiss, Ariel Ephrat, Omer Tov, Shiran Zada, Inbar Mosseri, Michal Irani, and Tali Dekel. Teaching CLIP to Count to Ten. In Proceedings of the International Conference on Computer Vision (ICCV), 2023.
- [53] Long Phan et al. Humanity’s Last Exam. arXiv preprint arXiv:2501.14249, 2025.
- [54] Baifeng Shi, Boyi Li, Han Cai, Yao Lu, Sifei Liu, Marco Pavone, Jan Kautz, Song Han, Trevor Darrell, Pavlo Molchanov, and Hongxu Yin. Scaling Vision Pre-Training to 4K Resolution. arXiv preprint arXiv:2503.19903, 2025.

- [55] Andreas Steiner, André Susano Pinto, Michael Tschannen, Daniel Keysers, Xiao Wang, Yonatan Bitton, Alexey Gritsenko, Matthias Minderer, Anthony Sherbondy, Shangbang Long, Siyang Qin, Reeve Ingle, Emanuele Bugliarello, Sahar Kazemzadeh, Thomas Mesnard, Ibrahim Alabdulmohsin, Lucas Beyer, and Xiaohua Zhai. PaliGemma 2: A Family of Versatile VLMs for Transfer. arXiv preprint arXiv:2412.03555, 2024.
- [56] Richard Stim. How Long Does Copyright Protection Last? Nolo, 2023.
- [57] Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito. SlideVQA: a dataset for document visual question answering on multiple images. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2023.
- [58] Qwen Team. Qwen3.5: Accelerating Productivity with Native Multimodal Agents, 2026.
- [59] Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. Winoground: Probing vision and language models for visiolinguistic compositionality. In Proceedings of the Conference on Computer Vision and Pattern Recognition (CVPR), 2022.
- [60] Barbara Tversky. What do sketches say about thinking. In 2002 AAAI Spring Symposium, Sketch Understanding Workshop, Stanford University, AAAI Technical Report SS-02-08, 2002.
- [61] An Vo, Khai-Nguyen Nguyen, Mohammad Reza Taesiri, Vy Tuong Dang, Anh Totti Nguyen, and Daeyoung Kim. Vision Language Models are Biased. arXiv preprint arXiv:2505.23941, 2025.
- [62] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. Chain of Thought Prompting Elicits Reasoning in Large Language Models. In Advances in Neural Information Processing Systems (NeurIPS), 2022.
- [63] Haoning Wu, Zicheng Zhang, Erli Zhang, Chaofeng Chen, Liang Liao, Annan Wang, Chunyi Li, Wenxiu Sun, Qiong Yan, Guangtao Zhai, et al. Q-bench: A benchmark for general-purpose foundation models on low-level vision. arXiv preprint arXiv:2309.14181, 2023.
- [64] Penghao Wu and Saining Xie. V*: Guided visual search as a core mechanism in multimodal llms. In Proceedings of the Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [65] Zhiyu Wu, Xiaokang Chen, Zizheng Pan, Xingchao Liu, Wen Liu, Damai Dai, Huazuo Gao, Yiyang Ma, Chengyue Wu, Bingxuan Wang, Zhenda Xie, Yu Wu, Kai Hu, Jiawei Wang, Yaofeng Sun, Yukun Li, Yishi Piao, Kang Guan, Aixin Liu, Xin Xie, Yuxiang You, Kai Dong, Xingkai Yu, Haowei Zhang, Liang Zhao, Yisong Wang, and Chong Ruan. DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding. arXiv preprint arXiv:2412.10302, 2024.
- [66] Peng Xu, Wenqi Shao, Kaipeng Zhang, Peng Gao, Shuo Liu, Meng Lei, Fanqing Meng, Siyuan Huang, Yu Qiao, and Ping Luo. LVLM-EHub: A Comprehensive Evaluation Benchmark for Large Vision-Language Models. IEEE Transactions on

- Pattern Analysis and Machine Intelligence (TPAMI), 47(3), 2025.
- [67] Zhenfei Yin, Jiong Wang, Jianjian Cao, Zhelun Shi, Dingning Liu, Mukai Li, Xiaoshui Huang, Zhiyong Wang, Lu Sheng, LEI BAI, Jing Shao, and Wanli Ouyang. LAMM: Language-Assisted Multi-Modal Instruction-Tuning Dataset, Framework, and Benchmark. In Advances in Neural Information Processing Systems (NeurIPS), 2023.
- [68] Weihao Yu, Zhengyuan Yang, Lingfeng Ren, Linjie Li, Jianfeng Wang, Kevin Lin, Chung-Ching Lin, Zicheng Liu, Lijuan Wang, and Xinchao Wang. Mm-vet v2: A challenging benchmark to evaluate large multimodal models for integrated capabilities. arXiv preprint arXiv:2408.00765, 2024.
- [69] Xiang Yue, Yuansheng Ni, Tianyu Zheng, Kai Zhang, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. MMMU: A Massive Multi-Discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI. In Proceedings of the Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [70] Peng Zhang, Yash Goyal, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Yin and Yang: Balancing and Answering Binary Visual Questions. In Proceedings of the Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
- [71] Pan Zhang, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Rui Qian, Lin Chen, Qipeng Guo, Haodong Duan, Bin Wang, Linke Ouyang, Songyang Zhang, Wenwei Zhang, Yining Li, Yang Gao, Peng Sun, Xinyue Zhang, Wei Li, Jingwen Li, Wenhai Wang, Hang Yan, Conghui He, Xingcheng Zhang, Kai Chen, Jifeng Dai, Yu Qiao, Dahua Lin, and Jiaqi Wang. InternLM-XComposer-2.5: A Versatile Large Vision Language Model Supporting Long-Contextual Input and Output. arXiv preprint arXiv:2407.03320, 2024.
- [72] Renrui Zhang, Jiaming Han, Chris Liu, Peng Gao, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199, 2023.
- [73] Yuhui Zhang, Alyssa Unell, Xiaohan Wang, Dhruba Ghosh, Yuchang Su, Ludwig Schmidt, and Serena Yeung-Levy. Why are visually-grounded language models bad at image classification? arXiv preprint arXiv:2405.18415, 2024.
- [74] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. arXiv preprint arXiv:2506.05176, 2025.
- [75] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models. In International Conference on Learning Representations (ICLR), 2023.
- [76] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, Zhangwei Gao, Erfei Cui, Xuehui Wang, Yue Cao,

Yangzhou Liu, Xingguang Wei, Hongjie Zhang, Haomin Wang, Weiye Xu, Hao Li, Jiahao Wang, Nianchen Deng, Songze Li, Yinan He, Tan Jiang, Jiapeng Luo, Yi Wang, Conghui He, Botian Shi, Xingcheng Zhang, Wenqi Shao, Junjun He, Yingtong Xiong, Wenwen Qu, Peng Sun, Penglong Jiao, Han Lv, Lijun Wu, Kaipeng Zhang, Huipeng Deng, Jiaye Ge, Kai Chen, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. InternVL3: Exploring Advanced Training and TestTime Recipes for Open-Source Multimodal Models. arXiv preprint arXiv:2504.10479, 2025.

## VISUALOVERLOAD

### Probing Visual Understanding of VLMs in Really Dense Scenes Supplementary Material

#### Overview

- A. Quantifying Image Complexity 13
- B. Discussion of Domain Shift 13
- C. Benchmark Prompts 14
- D. Ablation of Resolution 14
- E. Language Bias Detection 14
- F. Performance with Advanced Prompting 15
- G. Embedding Space of Benchmark Questions 15
- H. Datasheet 17
- I. Image References 20

#### A. Quantifying Image Complexity

Measuring visual density or complexity remains a non-trivial challenge for standard metrics [15, 36, 42, 44]. To obtain complexity measurements of our dataset, we chose to quantify complexity through relative, pairwise comparisons across the full set of 150 images. We automate this process using Qwen3.5-27B (thinking) [58] to minimize human bias, sampling responses at temperature 1.0 (p = 0.95,k = 20,penalty = 1.0). The model was provided with the following prompt:

Prompt for Image Complexity (Qwen3.5-27B)

<Image 1><Image 2> You are an expert in image analysis. Given two images, your task is to determine which image has a higher visual density and complexity. Do not attempt to identify the name of the painting. Respond with ’A’ if the first image has higher complexity, ’B’ if the second image has higher complexity.

The prompt explicitly instructs the model to ignore the identity of the artwork, encouraging a focus on visual composition rather than semantic recognition. While we observed that the model occasionally attempted identification within its reasoning traces, its final selections remained focused on density.

Ultimately, an image’s complexity score is defined as its empirical win-rate: the proportion of pairings in which the model selected it as the more complex image. Fig. 7

[Figure 3]

[Figure 4]

least complex (0.00 %) most complex (99.3 %)

Figure 7. Extreme ends of complexity distribution. Shown are the least and most complex images according to our LLM-assisted measurements and their respective win-rate.

14

12

10

Count

8

6

4

2

0 20 40 60 80 100 Win-rate (%)

Figure 8. Distribution of Image Complexity. The distribution shows the win-rate over all 150 images in VisualOverload.

illustrates the images at the lowest and highest extremes of this calculated distribution (shown in Fig. 8).

#### B. Discussion of Domain Shift

VisualOverload consists of paintings, which will display a domain shift compared to natural images. We think of this shift as beneficial because we expect foundation models aiming for human-level perception to also be robust to stylistic abstractions.

The domain of historical paintings is a good testbed for this kind of robustness, as it offers distinct practical advantages: first, the paintings depict naturalistic scenes and objects before the time of photography. As such, they deviate from digital photos in various ways (style, color, etc.) while still maintaining the core features of objects, providing a good trade-off between realism and avoiding the biases of web-crawled data (center-focus, etc.). Second, we assume that this benchmark might be harder to solve than

|DeepSeek VL2 InternVL3 8B o3<br><br>| |
|---|
<br><br>| |
|---|
| | |
|---|---|---|
| | | |

Default Prompt for Counting Questions {Question} Answer with a number directly.

Density

#### D. Ablation of Resolution

We distribute VisualOverload at a resolution that matches the pixels of 4K (with a few outliers). Additionally, we downsampled images to match the number of pixels of VGA (640 × 480 pixels), HD (1280 × 720 pixels), FHD (1920 × 1080 pixels), QHD (2560 × 1440 pixels), and measured task-level performances on various instances of InternVL3 models in comparison to our original resolution. The results are shown in Fig. 10.

0 20 40 60 80 100 Accuracy per image (%)

Figure 9. Distribution of accuracy per image. Three diverse models show distributions with clear peaks beyond average accuracy, thus proving that the challenge is caused by the content of the image and question, not the domain.

benchmarks consisting of natural images, by just crawling more data (or even generating), as the number of historical paintings is finite and not easily scalable. To improve performance, models will always, to a certain extent, have to be able to generalize. Finally, it is a domain that provides high-resolution images without depth-of-field blur (ensuring details are recognizable across the full scene) and provides a clean copyright status.

Generally, performance improves with resolution, but at a minor rate. However, it is visible that improvements are differently correlated with tasks. Text (especially small one) is poorly compressible, and it is, thus, unsurprising to see a strong correlation between resolution and OCR performance. The opposite is modeled by scene recognition, which, for the most part, is solvable by global features that should be detectable even at extreme compression. This is backed by the lack of significant performance deviation throughout our tested resolutions. For the other tasks, we typically see an increase in performance with resolution, which seems to plateau after Full HD resolution.

Finally, we investigate whether the domain shift may be correlated with the low performance of some models. As shown in the figure (Fig. 9), the per-image accuracy distributions for 3 distinct models lack the density near zero performance that would characterize a domain shift. Instead, they show a well-above random performance on most images, thus indicating that the performance is not tied to the domain but to the question and visual details. This is further backed by the measurements of logical consistency in Fig. 6, which show stark differences in performance for tasks differing in the required level of image understanding.

This is likely not a shortcoming of our benchmark, but rather attributed to the model’s architecture. By default, InternVL3 splits the input image into at most 12 patches (each 448 × 448 pixels) plus a thumbnail [76]. Thus, the model only supports a resolution slightly above FHD without downsampling. While it is possible to increase the number of patches, this significantly increases the inference time and memory. For instance, even for InternVL3-8B, increasing the number of patches from 12 to 40, which should be sufficient to process VisualOverload without downsampling, requires 8 × 40 GB GPUs, instead of just one, making such an experiment impossible for us. In theory, we, however, expect model performance to scale with resolution, assuming no downsampling. Consequently, we also expect higher performance using more patches (assuming a sufficient context window and proper training).

#### C. Benchmark Prompts

We used the following prompts in our main evaluation, depending on the question type (multiple-choice, counting, or OCR):

Default Prompt for Multiple-Choice Questions {Question} Options:

- A. {Option A}
- B. {Option B} · · · Answer with the option’s letter from the given choices directly.

#### E. Language Bias Detection

Default Prompt for OCR Questions {Question} Answer directly.

We use Gemini 2.5 Pro with the following prompt to detect language bias:

Task Activity Attributes Counting OCR Reasoning Scene Score

InternVL3-8B

InternVL3-14B

InternVL3-38B

80

Accuracy(%)

60

40

VGA HD FHD QHD Original Resolution

VGA HD FHD QHD Original Resolution

VGA HD FHD QHD Original Resolution

###### Figure 10. Resolution ablation.

Prompt for Language Bias Detection (Gemini 2.5 Pro)

Below you will find a CSV with an excerpt of questions from a visual question answering benchmark. The benchmark is supposed to be only solvable by looking at the image, however for the questions below, most models are able to guess the correct option (ground_truth). Your task is to look at each questions, the options, and ground_truth and to determine if the models were just lucky or there is some kind of shortcut or language bias. Provide an answer and rationale for each question_id.

question_id, question, options, ground_truth

{CSV}

Tab. 2 contains the quality control results discussed in Sec. 2.1.

#### F. Performance with Advanced Prompting

Our evaluation in Sec. 3 utilizes simple prompts. In this section, we additionally ablate zero-shot chain-of-thought (CoT) [29, 62] on InternVL3 8B, the strongest 8B model on our benchmark, and an overall strong model. To this end, we modified the prompts as follows:

CoT Prompt for Multiple-Choice Questions {Question} Options:

- A. {Option A}
- B. {Option B} · · · Think step by step. Answer with the option’s letter from the given choices wrapped in <answer></answer>.

CoT Prompt for OCR Questions {Question} Think step by step. Answer with the extracted text wrapped in <answer></answer>

CoT Prompt for Counting Questions {Question} Think step by step. Answer with a number wrapped in <answer></answer>

The results in Tab. 3 show that at least for this model, CoT decreased performance on average. However, it significantly improved performance on the hardest split and for OCR. Since CoT prompting is primarily effective in large-scale LLMs [62], we hypothesize that the tested LLM may have been too small to benefit from CoT.

#### G. Embedding Space of Benchmark Questions

We show a UMAP [46] reduced embedding generated by Qwen3-embedding-4B [74] of all questions (without answers) colored by task in Fig. 11. A clear separation of tasks is visible, except for the reasoning task, which overlaps with multiple other tasks as intended. The OCR questions form the most disconnected cluster.

Table 2. Blind benchmark results. We benchmark three models on VisualOverload without the images to measure a potential language bias.

Activity Attributes Counting OCR Reasoning Scene Easy Medium Hard Total Model Params [B] (150) (149) (559) (118) (356) (1388) (986) (1304) (430) (2720)

Random Chance - 25.0 25.0 0.0 0.0 25.0 25.0 24.5 16.7 3.7 16.0 Consistent Chance - 25.0 25.0 0.0 0.0 42.5 50.0 47.2 26.2 4.7 27.2

InternVL3 38B [76] 38 30.0 34.9 15.6 0.8 36.6 24.2 32.5 24.5 8.1 22.8 Qwen2.5-VL 32B [5] 32 32.0 26.2 8.8 0.0 29.3 38.0 39.7 25.1 6.2 24.5 LLaVA-OV 72B [33] 72 29.3 40.3 18.1 0.8 36.1 38.6 24.6 41.1 6.5 29.2

###### Table 3. Comparison with CoT prompting.

Params Activity Attributes Counting OCR Reasoning Scene Easy Medium Hard Total Model [B] (150) (149) (559) (118) (356) (1388) (986) (1304) (430) (2720)

InternVL3 38B [76] 38 76.7 78.5 35.4 45.8 69.8 92.2 99.7 81.8 7.2 67.6 + CoT 38 74.0 69.8 34.5 50.0 62.4 91.4 98.9 77.1 14.4 65.5

Task

Scene OCR

Activity Reasoning

Counting Attributes

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

###### Figure 11. Question embeddings.

#### H. Datasheet

In the following, we provide a datasheet [19]. We have anonymized some entries for the review process and will update these upon release.

|Motivation|
|---|

For what purpose was the dataset created?

VisualOverload was created to test basic visual recognition skills of VLMs in densely populated scenes, as most prior VQA datasets often probe skills of superficial features.

Who created this dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?

The dataset was created by the authors of this paper on behalf of their institutions. Who funded the creation of the dataset?

All authors were funded by their respective institutions.

|Composition|
|---|

What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?

The dataset consists of images associated with multiple questions.

How many instances are there in total (of each type, if appropriate)?

The dataset consists of 150 images and a total of 2720 questions.

Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?

The images are a subset of public domain artworks hosted on https://artsandculture.google.com filtered to display visually complex and dense scenes.

What data does each instance consist of? “Raw” data (e.g., unprocessed text or images) or features?

Each sample is a collection of the following items:

- • question_id: Unique identifier of each question.
- • image: A PIL JPEG image. Most of our images were resized to match the total pixel count of 4k (3840x2160 px) in different aspect ratios.
- • question: A question about the image.
- • question_type: Type of question. Will be one of choice (response expected to be "A", "B", "C", or "D"), counting (freeform), or ocr (freeform). You can use this information to request a suitable output format.
- • options: This is the list of options for question_type=choice and empty otherwise. Please treat the options as answer options A, B, C, D (4 options) or A, B (2 options).

- • difficulty: Meta-data about the difficulty of the question. One of easy, medium, or hard.
- • category: Meta-data about the question task. One of activity, attributes, counting, ocr, reasoning, or scene.
- • default_prompt: You can use this prompt to stay compliant with our results. It is a simple combination of the question and answers, with some additional output format constraints. This should work well for most models.

Further, we provide a database linking each image to its respective complexity_win_rate, as defined in Appendix A.

###### Is there a label or target associated with each instance?

Each question is associated with a ground-truth. This ground-truth is hidden from the public to avoid test leakage.

###### Is any information missing from individual instances?

We obfuscate image file names and question IDs to reduce knowledge priors.

Are relationships between individual instances made explicit (e.g., users’ movie ratings, social network links)? The samples in the dataset shall be treated independently.

Are there recommended data splits (e.g., training, development/validation, testing)?

All the samples in our dataset shall be exclusively treated as a test set. We do not provide development sets, as we consider all questions to be solvable with a basic set of skills that should be present in frontier VLMs.

Are there any errors, sources of noise, or redundancies in the dataset?

All questions and ground truths are manually annotated and, thus, may contain errors. To reduce the error rate, we doublechecked all questions where multiple models provided wrong answers.

Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?

The dataset is self-contained.

Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals non-public communications)?

No.

Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?

The dataset contains samples that show religious beliefs, (partial) nudity, and/or injury and death. Does the dataset relate to people?

The dataset contains artworks that may depict people.

Does the dataset identify any subpopulations (e.g., by age, gender)?

The dataset does not identify any subpopulations.

Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?

Some of the individuals are of historical, biblical, or mythical origin and may be identified. No living individuals can be identified from the dataset.

Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?

No.

|Collection Process|
|---|

How was the data associated with each instance acquired?

Please see Sec. 2.

What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?

Please see Sec. 2.

If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?

n/a.

Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?

The dataset was collected and annotated by the authors of this paper. No crowdworkers, students, or contractors, etc., were involved.

Over what timeframe was the data collected? Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)?

The images were collected between April and May 2025, and annotated and cleaned between May and August 2025.

Were any ethical review processes conducted (e.g., by an institutional review board)?

No.

Does the dataset relate to people?

The dataset contains artworks that may depict people.

Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?

n/a.

Were the individuals in question notified about the data collection?

All depicted individuals are no longer alive.

Did the individuals in question consent to the collection and use of their data?

n/a.

If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?

n/a.

Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?

n/a.

|Preprocessing/cleaning/labeling|
|---|

Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-ofspeech tagging, SIFT feature extraction, removal of instances, processing of missing values)?

Yes, see Sec. 2.

Was the “raw” data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?

The raw data can be requested from the authors.

Is the software used to preprocess/clean/label the instances available?

The images were obtained using https://github.

com/lovasoa/dezoomify-rs. All further processing scripts were developed by the authors and are not available publicly.

|Uses|
|---|

Has the dataset been used for any tasks already?

The dataset has been used to evaluate basic visual skills of frontier VLMs in Sec. 3.

Is there a repository that links to any or all papers or systems that use the dataset?

We will list relevant papers at https://github.com/ paulgavrikov/visualoverload. We encourage authors to contact us to list their works.

###### What (other) tasks could the dataset be used for?

The dataset is primarily designed for visual question answering (VQA), but we encourage users to apply it to other tasks as desired.

Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?

No.

###### Are there tasks for which the dataset should not be used?

This dataset is released exclusively for academic research and educational use. It must not be applied to purposes that could lead to harm, including surveillance, discrimination, exploitation, harassment, or the generation of misleading or offensive content. Users are expected to uphold the highest standards of research integrity and ethics, and to ensure that their work with this dataset aligns with responsible AI principles.

|Distribution|
|---|

Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created?

The dataset is publicly available.

How will the dataset be distributed (e.g., tarball on website, API, GitHub)

The dataset is distributed through HuggingFace datasets,

which currently uses a PyArrow format. When will the dataset be distributed?

The dataset is immediately distributed through: https: //huggingface.co/datasets/paulgavrikov/ visualoverload.

Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?

The dataset is distributed under the Creative Commons Attribution-ShareAlike 4.0 International license without any further terms of use.

Have any third parties imposed IP-based or other restrictions on the data associated with the instances?

No.

Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?

No.

|Maintenance|
|---|

Who will be supporting/hosting/maintaining the dataset?

The authors will be supporting/hosting/maintaining the dataset.

How can the owner/curator/manager of the dataset be contacted (e.g., email address)?

The authors can be contacted via GitHub issues at: https : / / github . com / paulgavrikov / visualoverload/issues.

###### Is there an erratum?

No.

Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?

The dataset will not be modified to ensure comparability of results. Corrected or derived datasets will be released independently.

If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?

n/a.

Will older versions of the dataset continue to be supported/hosted/maintained?

The dataset will remain available as long as it continues to be hosted by the third-party platforms on which it is stored.

If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?

Users can extend/augment/build upon the dataset, but must publish their new work as a standalone derivative. We kindly request that users communicate any releases to the authors.

#### I. Image References

Table 4. List of artworks. This table contains all artworks present in VisualOverload in random order. The metadata is taken from the references with minor postprocessing by us.

###### Creator Title Date URL

Unknown Wood of the Philosophers 1800/1830 Link Pieter Aertsen The Fat Kitchen. An Allegory 1565-1575 Link Unknown A handscroll painting of the porcelain production process

early 19th century Link

(right half)

Avercamp, Hendrick Enjoying the Ice ca. 1615-1620 Link Charles M. Russell The Medicine Man 1908 Link Pieter van der Heyden after Pieter Bruegel the Elder

The Big Fish Eat the Little Fish published 1557 Link

Charles Fairfax Murray Allegory of Good Government, after Ambrogio Lorenzetti 1873 Link Philip Galle after Pieter Bruegel the Elder

Prudence published 1559 Link

Pieter Bruegel the Elder, Frans Hogenberg

The Kermis at Hoboken ca. 1559 Link

Joan Antigó, Honorat Borrassà i Francesc Vergós

Altarpiece of Saint Miquel de Castelló d’Empúries (detail) 1448 Link

Greek artist from the end of the 18th century

St. George 1798/1798 Link

Anger (Ira) from The Seven Deadly Sins 1558 Link

Pieter Bruegel the Elder, Pieter van der Heyden, Hieronymus Cock

Dirck Franchoisz Hals, Dirck van Delen

Festive Company in a Renaissance Room 1628 Link

Philips Galle, Pieter Bruegel the Elder, Hieronymus Cock

Charity (Charitas) from The Virtues 1559 Link

Unknown A Sunday on La Grande Jatte 1884-1886 Link Charles William Sharpe The Death of Nelson at the Battle of Trafalgar 1806/1876 Link Ast, Balthasar van der Still Life with Fruit and Flowers 1620-1621 Link School of Canaletto St. Marks, Venice unknown Link Pieter Bruegel the Elder The Sermon of Saint John the Baptist 1566 Link Pieter Aertsen Market Scene 1569 Link Thomas Matthews Rooke Washing Sheds at Chartres 1885 Link Severin Roesen Still Life of Flowers and Fruit with a River Landscape in

1867 Link

the Distance

Pere Mates Final Judgment. Altarpiece of Santa Maria de Segueró (Garrotxa)

1500/1550 Link

Steen, Jan Havicksz Villagers Merrymaking Outside an Inn 1652 Link

Hieronymus Bosch Ecce Homo 1500 Link Jan Steen Beware of Luxury (“In Weelde Siet Toe”) 1663 Link Unknown Christ in the House of Martha and Mary 1553 Link Ostade, Adriaen van Peasants in an Interior 1661 Link Pieter Bruegel the Elder Desidia (Sloth) 1557 Link Avercamp, Hendrick Enjoying the Ice near a Town ca. 1620 Link Rijn, Rembrandt van The Night Watch 1642 Link Ditlev Blunck Danish artists at the Osteria La Gensola in Rome 1837 Link Konstantin Makovsky A Boyar Wedding Feast 1883 Link Unknown The trial of the Neptune’s seamen 1807 Link Albrecht Altdorfer Christ taking Leave of his Mother probably 1520 Link Baines, Thomas Kaffirs and Rebel Hottentotts Attacking a Wagon Train 1851/1852 Link Unknown Bucentaur’s return to the pier by the Palazzo Ducale 1728/1729 Link Neer, Aert van der Winter Landscape near a Town with Kolf Players and

ca. 1650-1655 Link

Horse-Drawn Sleighs

Philip Galle after Pieter Bruegel the Elder

Faith published 1559 Link

Francesco Hayez Pope Urban II Preaching the First Crusade in the Square of Clermont

1835/1835 Link

Pieter Bruegel the Elder The Fall of the Rebel Angels 1562 Link Atelier de Paris Psyché rapporte la laine des brebis 1650 Link Hals, Dirck The Fête champêtre 1627 Link Baines, Henry Fisherfleet Looking East 1823/1894 Link Jan Cornelisz. Vermeyen The Spanish Brothel 1545 Link Marià Vayreda i Vila Gambeto dance in Riudaura 1890 Link Paolo De Matteis St. Nicolas of Bari Felling a Tree Inhabited by Demons 1727/1727 Link Philip Galle, Pieter Bruegel The Resurrection of Christ ca. 1562 Link Pieter Aertsen Market Scene 1550 Link Pieter Aertsen Christ with Mary and Martha 1552 Link Philip Galle after Pieter Bruegel the Elder

The Parable of the Wise and Foolish Virgins ca. 1560/1563 Link

Unknown Dragon Boats at Aberdeen Hong Kong showing Careening Island

1923 Link

Pieter Bruegel the Elder, Pieter van der Heyden, Hieronymus Cock

Avarice (Avaritia), from the series The Seven Deadly Sins 1558 Link

Hendrick Avercamp A Scene on the Ice ca. 1625 Link Attributed to Jan van Belcamp

The Great Picture 1646 Link

Aleksander Ivanov The Apparition of Christ to the People (The Apparition of the Messiah)

1837-1857 Link

Hendrick Avercamp Frozen River with Skaters 1620s Link Jan Rost The Pharaoh Welcomes Joseph 1553 Link Aertsen, Pieter Wing of an Altarpiece with Adoration of the Magi, on the

1560-1565 Link

reverse is Presentation in the Temple

Van Aachen, Hans The Rape of Proserpine 1589 Link Avercamp, Hendrick Winter Landscape with Ice Skaters ca. 1608 Link Estevão Silva Untitled 1887/1887 Link Pieter Bruegel, J. Liefrinck, H. Hondius

The fat kitchen 1563 Link

Unknown Moses descends from Mount Siniai with the Ten Commandments

1662 Link

Unknown British forces receiving Commissioner Keying at Canton’s British Factories for conference with Sir J. F. Davies

1847 Link

Unknown Le peuple rend les honneurs à Psyché 1650 Link Pieter Bruegel the Elder Peasant Wedding 1566-1569 Link Jan Miense Molenaer Self-Portrait with Family Members 1630/1640 Link Philip Galle after Pieter Bruegel the Elder

Temperance published 1559 Link

Anonymous Christ as the Good Shepherd 1505 Link Dirck Jacobsz Vellert The Flood 1544 Link Cornelis Cornelisz van Haarlem

The Golden Age (Bacchanal) or the Garden of Love 1614 Link

Aertsen, Pieter The Adoration of the Magi ca. 1560 Link William Duffield Still Life 1859 Link Balthasar van der Ast Still Life of Flowers, Fruit, Shells, and Insects About 1629 Link Albrecht Dürer Feast of Rose Garlands 1506 Link Jan Steen The Dancing Couple 1663 Link Jan Steen The Worship of the Golden Calf ca. 1672-1675 Link Carl Bloch In a Roman Osteria 1866 Link Avercamp, Hendrick Ice-Skating in a Village ca. 1610 Link After Pieter Bruegel the Elder

The Festival of Fools after 1570 Link

The Witch of Malleghem published 1559 Link

Pieter van der Heyden after Pieter Bruegel the Elder

Pieter van der Heyden, Pieter Bruegel the Elder, Hieronymus Cock

Patience (Patientia) 1557 Link

Francisco de Goya El Entierro de la Sardina 1808/1812 Link

Master of Okoliˇcno Holy Kinship 1510 Link Maarten van Heemskerck The Gods of the Olympus 1556 Link Gerrit van Honthorst Apollo and Diana 1628 Link Pieter Bruegel the Elder Children’s Games 1560 Link Hendrick Avercamp Skating Scene 1620s Link Pieter Bruegel the Elder The Adoration of the Magi undated Link Bol, Ferdinand Consul Titus Manlius Torquatus Orders the Beheading of

1661-1663 Link

his Son

Balthasar van der Ast Still Life with Basket of Fruit 1622 Link

Pieter Bruegel the Elder, Philips Galle, Hieronymus Cock

The Alchemist after 1558 Link

Claesz., Pieter Still Life with a Turkey Pie 1627 Link Pieter Bruegel the Elder The Dirty Bride or the Marriage of Mopsus and Nisa 1570 Link Western artist painted from the deck of HMS Vulcan

British and French fleets in Victoria Harbour 1860 Link

Pieter Bruegel the Elder Hunters in the Snow (Winter) 1565 Link

Philips Galle, Pieter Bruegel the Elder, Hieronymus Cock

Justice (Justicia) from The Virtues ca. 1559–60 Link

Giulio Romano Chamber of the Giants - Ceiling 1532-1534 Link

Pieter Bruegel the Elder, Pieter van der Heyden, Hieronymus Cock

The Descent of Christ Into Limbo ca. 1561 Link

Hendrick Avercamp Winter Landscape 1600/1620 Link Arcadi Mas i Fondevila The Corpus Christi procession 1887 Link Unknown Winter Scene on a Frozen Canal 1620 Link Caroline Le Souef Home Life of the Victorian Aborigines 1895-1895 Link Andrej Janez Herrlein Ljubljana Šempeter 1798 Link Edward Roper Gold Diggings, Ararat 1855-1860 Link Pieter Bruegel the Elder Massacre of the Innocents 1565-1567 Link Lucas van Leyden Christ presented to the people 1510 Link Hans Bol Goose Snatching 1560/1580 Link Edouard Hildebrand Menino com patos 1800/1800 Link Pieter Bruegel the Elder The Census at Bethlehem 1566 Link Frans Huys Ice Skating before the Gate of Saint George in Antwerp 1558 Link Pieter Aertsen A Meat Stall with the Holy Family Giving Alms 1551 Link Philip Galle after Pieter Bruegel the Elder

Fortitude published 1559 Link

Lucas Cranach the Elder The Crucifixion 1506-1520 Link Bol, Ferdinand Aeneas Crowning Cloanthus ca. 1661-1663 Link

Pelegrí Clavé i Roqué Jacob receives the bloody tunic of his son Joseph 1842 Link Aertsen, Pieter The Egg Dance 1552 Link Unknown The Village Wedding 1653 Link Joos van Craesbeeck The Temptation of Saint Anthony 1650 Link Pieter Bruegel the Elder Winter Landscape with Skaters and Birds Trap 1565 Link Caroline Le Souef Native Fight on the Lower Goulburn River in 1842 1895-1895 Link Joachim Wtewael The Annunciation to the Shepherds 1606 Link Joachim Beuckelaer Woman Selling Vegetables second half of

Link

16th century

Unknown Dragon boat racing at Spring Festival 1860 Link Karl Brullov The Last Day of Pompeii 1830/1833 Link Caroline Le Souef Corroboree on the Goulburn River 1895-1895 Link

Pieter Bruegel the Elder, Hieronymus Cock, Philips Galle

Hope (Spes) from The Virtues ca. 1559–60 Link

Steen, Jan Havicksz. The Feast of St Nicholas 1665-1668 Link Hans Memling Virgin and Child with Saints Catherine of Alexandria and

early 1480s Link

Barbara

Gerard Dou The Quack 1652 Link Baines, Thomas Forces under General Cathcart crossing the Orange river

1854 Link

to attack Moshesh 1852

Namcheong Whampoa’s earliest mud dock, Canton ca. 1850s Link Jan Fyt Still life with parrot ca. 1645 Link Oldmeadow, William H. Reffley Spring 1818 Link Bartholomeus van Bassen, Esaias van den Velde

Renaissance Interior with Banqueters 1618/1622 Link

Gift of Mr. Antony J. Hardy Bamboo Town and Anchorage at Whampoa Island 1860 Link Unknown A handscroll painting of the porcelain production process

early 19th century Link

(left half)

Pieter Brueghel, Pieter van der Heyden, Hieronymus Cock

Luxuria, uno de los siete vicios 1558 Link

Pieter Brueghel La soberbia Serie de los siete pecados capitales 1558 Link Pieter Bruegel the Elder Landscape with the Fall of Icarus undated Link

Pieter Bruegel the Elder, Pieter van der Heyden, Hieronymus Cock

The Last Judgment 1558 Link

Hans Memling The Annunciation 1480–89 Link James Taylor Panorama du Port Jackson et de la ville de Sidney 1820/1825 Link Avercamp, Hendrick Frolicking on a Frozen Canal in a Town ca. 1615-1620 Link

After Pieter Bruegel the Elder

Gluttony (Gula) from The Seven Deadly Sins 1558 Link

Felip Massó i Falp The Procession of St. Bartolomew 1884 Link Pieter Bruegel the Elder The Dutch Proverbs 1559 Link Maertan Van Heemskerck Concert of Apollo and the Muses on Mount Helicon 1565 Link David Teniers the Younger The Surgeon 1670s Link

