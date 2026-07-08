# arXiv:2411.16508v4[cs.CV]1May2025

## All Languages Matter: Evaluating LMMs on Culturally Diverse 100 Languages

Ashmal Vayani1,2*, Dinura Dissanayake2, Hasindri Watawana2, Noor Ahsan2, Nevasini Sasikumar2, Omkar Thawakar2, Henok Biadglign Ademtew7, Yahya Hmaiti1, Amandeep Kumar8, Kartik Kuckreja2, Mykola Maslych1, Wafa Al Ghallabi2, Mihail Mihaylov2, Chao Qin2, Abdelrahman M Shaker2, Mike Zhang9, Mahardika Krisna Ihsani2, Amiel Esplana10, Monil Gokani11, Shachar Mirkin12, Harsh Singh2, Ashay Srivastava13, Endre Hamerlik14, Fathinah Asma Izzati2, Fadillah Adamsyah Maani2, Sebastian Cavada2, Jenny Chim15, Rohit Gupta1, Sanjay Manjunath2, Kamila Zhumakhanova2, Feno Heriniaina Rabevohitra16, Azril Amirudin17, Muhammad Ridzuan2, Daniya Kareem2, Ketan More2, Kunyang Li1, Pramesh Shakya1, Muhammad Saad2, Amirpouya Ghasemaghaei1, Amirbek Djanibekov2, Dilshod Azizov2, Branislava Jankovic2, Naman Bhatia11, Alvaro Cabrera2, Johan Obando-Ceron18, Olympiah Otieno19, Fabian Farestam20, Muztoba Rabbani21, Sanoojan Baliah2, Santosh Sanjeev2, Abduragim Shtanchaev2, Maheen Fatima22, Thao Nguyen2, Amrin Kareem2, Toluwani Aremu2, Nathan Xavier23, Amit Bhatkal2, Hawau Toyin2, Aman Chadha3, Hisham Cholakkal2 , Rao Muhammad Anwer2,4 , Michael Felsberg6 , Jorma Laaksonen4 , Thamar Solorio2 , Monojit Choudhury2 , Ivan Laptev2, Mubarak Shah1, Salman Khan2,5, Fahad Shahbaz Khan2,6

1University of Central Florida, 2Mohamed bin Zayed University of AI, 3Amazon, 4Aalto University, 5Australian National University, 6Linköping University, 7Ehtiopian AI Institute, 8Johns Hopkins University, 9Aalborg University, 10University of the West of England, 11IIT Hyderadabad, 12Alpinference, 13University of Maryland, 14HUN-REN Institute for CS and Control, 15Queen Mary University of London, 16Chongqing University, 17University of the People, 18University of Montreal, 19JKUAT, 20ETH Zurich, 21University of California, Merced, 22Air University, 23Universidade Federal de Minas Gerais

[Figure 1]

Figure 1. ALM-bench comprises a diverse set of 100 languages with manually verified annotations by respective native language experts. Here, qualitative examples highlight the comprehensive set of 13 cultural aspects covered in the benchmark, such as heritage, customs, architecture, literature, music, and sports. It also evaluates visual understanding for six generic aspects. The ALM-bench focuses on lowresource languages and different regions, spanning 73 countries across five continents and 24 distinct scripts. ALM-bench covers diverse questions, such as multiple choice questions (MCQs), true/false (T/F), short and long visual question answers (VQAs).

1

### Abstract

Existing Large Multimodal Models (LMMs) generally focus on only a few regions and languages. As LMMs continue to improve, it is increasingly important to ensure they understand cultural contexts, respect local sensitivities, and support low-resource languages, all while effectively integrating corresponding visual cues. In pursuit of culturally diverse global multimodal models, our proposed All Languages Matter Benchmark (ALM-bench) represents the largest and most comprehensive effort to date for evaluating LMMs across 100 languages. ALM-bench challenges existing models by testing their ability to understand and reason about culturally diverse images paired with text in various languages, including many low-resource languages traditionally underrepresented in LMM research. The benchmark offers a robust and nuanced evaluation framework featuring various question formats, including true/false, multiple choice, and open-ended questions, which are further divided into short and long-answer categories. ALM-bench design ensures a comprehensive assessment of a model’s ability to handle varied levels of difficulty in visual and linguistic reasoning. To capture the rich tapestry of global cultures, ALM-bench carefully curates content from 13 distinct cultural aspects, ranging from traditions and rituals to famous personalities and celebrations. Through this, ALM-bench not only provides a rigorous testing ground for state-of-the-art open and closed-source LMMs but also highlights the importance of cultural and linguistic inclusivity, encouraging the development of models that can serve diverse global populations effectively. Our benchmark is publicly available at https://mbzuai-oryx.github.io/ ALM-Bench/.

### 1. Introduction

In recent years, large multimodal models (LMMs) have seen significant progress across a variety of vision-language tasks [2, 5, 9, 28, 36, 42]. Despite these advances, a major limitation remains in the models’ capacity to understand and respond accurately to cultural and linguistic diversity [37]. Current LMMs perform well in widely spoken languages with substantial training data but struggle with lowresource languages and nuanced cultural contexts, limiting their effectiveness on a global scale [29].

Existing benchmarks for evaluating multilingual and multicultural capabilities in LMMs are limited in scope, often representing only a small subset of cultures and languages. Such benchmarks generally focus on highresource languages and overlook the cultural richness of low-resource languages, which hinders the development

*corresponding author: ashmal.vayani@ucf.edu

and evaluation of models that can serve diverse populations across the globe. For instance, CulturalVQA [32] and GD-VCR [45] focus on culturally specific content in English and do not reflect global cultural diversity. Similarly, the Henna benchmark [4] includes cultural information from 11 countries but is restricted to the Arabic language. In terms of multilingualism, existing benchmarks [11, 14, 27, 41, 43, 48] include multilingual data but remain constrained by limited cultural diversity and are often restricted to only single question type. This gap highlights the need for a more comprehensive benchmark that addresses both linguistic and cultural inclusivity in LMMs.

To address these challenges, we introduce a culturally and linguistically diverse multimodal benchmark: All Languages Matter Benchmark (ALM-bench). ALM-bench is specifically designed to evaluate LMMs across a diverse set of 100 languages, including many low-resource ones, with a strong emphasis on cultural depth. The complete list of these languages is presented in the Tab. A. 3. (suppl. material). In comparison to the recent CVQA benchmark [37], ALM-bench offers 3x more languages, more diverse question types, and ∼2x more image domains and samples with an extensive focus on fine-grained cultural aspects (Fig. 1). With these enhancements, ALM-bench aims to assess the next generation of massively multilingual multimodal models in a standardized way, pushing the boundaries of LMMs towards better cultural understanding and inclusivity.

Our evaluations with 16 recent state-of-the-art LMMs show that existing models suffer from low-resource languages and cultures (Fig. 2). The analysis also shows a significant gap between the open and closed-source models in understanding cultural and linguistic nuances in a multimodal setting; e.g., GPT4o performs 27% better than the best open-source model (GLM-4V). Our evaluations across scripts and language groups highlight the need to improve the understanding of underrepresented regions, such as Southeast Asia and West Africa. We also show how simple prompting techniques can help improve LMM performance.

Our contributions can be summarized as follows:

- • We introduce ALM-bench, a culturally diverse multilingual and multimodal VQA benchmark covering 100 languages with 22.7K question-answers. ALM-bench encompasses 19 generic and culture-specific domains for each language, enriched with four diverse question types.
- • ALM-bench is meticulously curated and verified with native-language experts (over 800 hours of human annotators), ensuring cultural relevance and accuracy across low- and high-resource languages alike.
- • We benchmark existing LMMs on ALM-bench, identifying performance gaps and areas for improvement, especially in culturally complex multilingual scenarios.

- 0

10

20

30

40

50

60

70

80

Performance(%)

26.4%

31.1% 31.8% 32.9% 33.0% 34.0% 34.3% 35.0%

37.4%

39.0%

44.8% 44.9%

48.8%

51.8%

74.3%

78.8%

15.2%

18.0%

16.8% 18.1% 19.9% 20.3% 21.8% 20.8%

22.6%

22.3%

24.9% 25.2%

28.3%

30.5%

38.3%

41.7%

11.2% 13.2% 15.0% 14.8% 13.0% 13.6% 12.5% 14.2% 14.8% 16.7%

19.9% 19.7% 20.5% 21.3%

36.0% 37.1%

| |
|---|

Specification

High Resource Languages

Low Resource Languages

(a) Performance comparison of open vs. closed-source models.

Architecture

Customs

Economy

Heritage

Festivals

Food

Literature Lifestyle

Media

Music

Notable Key Figures

Religion

Sports

38.6

49.2

59.8

70.4

35.8

46.6

57.4

68.2

36.6

46.2

55.8

65.4

41.6 52.2 62.8 73.4 40.6

51.2

61.8

72.4

34.2

45.4

56.6

67.8

37.2

48.4

59.6

70.8

39.6

49.2

58.8

68.4

38.0

49.0

60.0

71.0

40.2 50.4

60.6 70.8

64.0 55.0 46.0 37.039.6

50.2

60.8

71.4

38.0

49.0

60.0

71.0

GPT-4o

Eagle-X5-13B-Chat

Gemini-1.5-Pro

GLM-4V-9B

InternVL2-8B

(b) Performance on various cultural categories.

- Figure 2. Benchmarking LMMs on diverse languages & cultures: (a) Performance of various open versus closed-sourced LMMs on ALM-bench. For each LMM, we also report performance on low versus high-resource languages. All these carefully selected models were released after 2023. (b) Performance of high-performing LMMs on our culturally curated 13 categories available in our ALM-bench.

2. Related Work

To align large language models (LLMs) with human preferences, various approaches have been developed [35, 39, 50]. Over time, these methods are extended to enable LLMs to be sensitive to geo-diverse cultural values [1, 24, 31, 40]. This evolution has broadened their applicability to both multilingual and multimodal LLMs, enabling greater engagement with users from diverse linguistic and cultural backgrounds. Despite this progress, LMM benchmarks often fail to adequately address cultural aspects, especially for low-resource languages. Models such as GPT-4V [2] and Gemini 1.5 Pro [42] performs well on existing multimodal benchmarks, but these evaluations often lack in capturing the intricacies of multilingual and culturally diverse interactions. Open-source models, in particular, face challenges in accurately representing geodiverse values (e.g., local beliefs, rituals, and traditions), underscoring the need for a culturally aware evaluation framework for LMMs (see Tab.

- 1).

InternVL2-8BLLaVA-v1.5-7BmBLIP-mt0-xlPhi3-Vision-128k Pixtral-12B PALOEagle-X5-13B-ChatMiniCPM-V-2 LlaVa-v1.6 Qwen-VL MolmoLLaVA-OneVision Qwen-2-VL GLM-4V-9BGemini-1.5-Pro GPT-4o

lack cultural nuance and are constrained to multiple-choice questions (MCQ) which only assess the LMMs in one dimension. Although MM-Vet [46] and MMMU [47] offer a variety of question types, they primarily address basic perceptual abilities without demanding in-depth cultural reasoning or expert-level domain knowledge.

For multilingual LMM benchmarking, CVQA [37] represents a recent effort, collecting cultural samples across 31 languages. However, it is limited to a single question type, MCQ, and features only a small proportion of samples on traditions and rituals. Similarly, MaRVL [25] covers five languages but lacks probing into cultural common sense and primarily focuses on non-Western cultural information, resulting in limited diversity. Other benchmarks, including MMMB [41], MMBench [27], Exams-V [11], M3Exam [48], M4U [43], and MME [14], do include multilingual evaluation samples. However, they are constrained by a lack of cultural diversity in their datasets, cover fewer languages, and are restricted to a single question type (e.g., MCQs).

Our proposed benchmark, ALM-bench, bridges the aforementioned gaps (see Tab. 1) by introducing a culturally diverse dataset covering Western and non-Western regions, low and high-resource languages (100 in total), and specific regional dialects (e.g., Emirati Arabic, Egyptian Arabic). ALM-bench provides a robust evaluation framework for Multilingual LMMs, enabling a comprehensive assessment of their generic and cultural adaptability and promoting inclusivity in multilingual multimodal AI research.

Several recent benchmarks have been developed to evaluate the cultural aspects of multilingual LMMs, such as, the Henna benchmark [4] representing cultural information from 11 countries with nearly 1,132 samples. However, it is restricted to the Arabic language, covering only five domains while being closed-source. Other benchmarks, such as CulturalVQA [32] and GD-VCR [45] are also smallscale and focus only on culturally specific content. These benchmarks are limited to English and struggle to encompass the full spectrum of global cultural diversity. Their samples are unevenly distributed across countries and domains, which may lead to skewed assessment.

### 3. ALM Benchmark

ALM-bench provides global coverage, featuring both generic and culture-specific image samples that preserve the nuances of native languages in VQAs (Fig. 4). It includes diverse QA formats such as MCQs, True/False, and openended VQA (long and short) under 19 domains, grouped

Benchmarks such as SEED-Bench [21, 22] and MMStar [8] extend the CC3M dataset [33] to cover an extended set of domains. Despite their broader scope, these benchmarks

Benchmark Multilingual # of # of Total Total Question Question Annotation Cultural Specific Open- Bias Diversity Languages Scripts Domains Samples Types Forms Type Content Source Correction

Henna [4] ✗ 1 1 5 1132 SVQA, LVQA - Auto+Manual ✓ ✗ ✓ ✗ CulturalVQA [32] ✗ 1 1 5 2378 SVQA - Manual ✓ ✗ - ✗ GD-VCR [45] ✗ 1 1 1 886 MCQ Fixed Manual ✓ ✓ ✗ ✓ SEED-Bench [21] ✗ 1 1 12 19242 MCQ Diverse Auto+Manual ✗ ✓ ✗ ✗ SEED-Bench2 [22] ✗ 1 1 34 24371 MCQ Diverse Auto+Manual ✗ ✓ ✗ ✓ MMStar [8] ✗ 1 1 18 1500 MCQ Fixed Auto+Manual ✗ ✓ ✓ ✗ MM-Vet [46] ✗ 1 1 16 218 SVQA,LVQA Fixed Manual ✗ ✓ ✗ ✗ MMMU [47] ✗ 1 1 30 11550 MCQ,SVQA Fixed Manual ✗ ✓ ✓ ✓ CVQA [37] ✓ 31 13 10 10000 MCQ Fixed Manual ✓ ✓ ✗ ✓ MMMB [41] ✓ 6 4 15 12000 TF, MCQ Fixed Auto+Manual ✗ ✓ ✓ ✓ MMBench [27] ✓ 2 2 20 2974 MCQ Fixed Auto+Manual ✗ ✓ ✓ ✗ EXAMS-V [11] ✓ 11 4 20 20932 MCQ Diverse Manual ✗ ✓ ✗ ✓ MaRVL [25] ✓ 5 3 11 7630 TF Fixed Manual ✓ ✓ ✓ ✗ M3Exam [48] ✓ 9 3 4 12317 MCQ Diverse Auto+Manual ✗ ✓ ✗ ✓ MaXM [7] ✓ 7 5 - 2142 SVQA Fixed Auto+Manual ✗ ✓ ✓ ✓ xGQA [34] ✓ 8 5 - 12578 Y/N,SVQA Fixed Manual ✗ ✓ ✗ ✓ M4U [43] ✓ 3 2 64 8931 MCQ Fixed Auto+Manual ✗ ✓ ✓ ✗ MME [14] ✓ 2 2 14 2370 Y/N Fixed Manual ✗ ✓ ✗ ✗ Ours ✓ 100 24 19 22763 MCQ,SVQA,LVQA,TF Diverse Auto+Manual ✓ ✓ ✓ ✓

Table 1. Comparison of various LMM benchmarks with a focus on multilingual and cultural understanding. The Domains indicate the range of aspects covered by the dataset for each language. Question Form is categorized as "Diverse" if the questions phrasing varies, and "Fixed" otherwise. Annotation Types are classified as "Manual" if questions were originally in the local language, "Manual+Auto" if questions were generated or translated using GPT-4/Google API and subsequently validated by human experts, and "Auto" if generated or translated automatically without human validation. Bias Correction reflects whether the dataset is balanced across cultures and countries, while Diversity indicates whether the dataset includes both Western and non-Western minority cultures. ‘-’ means information not available.

into generic and cultural categories Tab. A. 1. (suppl. material).

The generic category includes six domains: indoor and outdoor scenes, memes, paintings, food items, and sketches following [26]. The cultural category spans 13 comprehensive domains—Architecture, Customs, Economy, Festivals, Food, Heritage, Lifestyle, Literature, Media, Music, Notable Figures, Religion, and Sports—adding depth to cultural representation across languages. While some of these domains have been previously explored [3, 30, 32, 37], they are investigated in limited languages and do not fully capture the depth of cultural nuances. Our work introduces six unique domains, including literature, music, festivals, economy, media, and notable key figures to better capture cultural richness across diverse languages. More details are in Sec. A (suppl. material)

ALM-bench comprises of 22,763 diverse questions across all subsets in 100 languages (see Fig. 4). The dataset spans 73 countries across five continents, capturing the cultural nuances of both underrepresented and predominant languages from different regions. We include 24 language scripts (Fig. 7) and 15 language families, with Latin being the most prevalent script, covering 53 languages, followed by Cyrillic with 10 languages; and Arabic with 9 languages. In terms of language families, Indo-European is the largest group, encompassing 55 languages, with Afro-Asiatic and Austronesian each representing eight languages (Fig. 9).

#### 3.1. Data Collection and Annotation

Country-Language pairs. Our cultural data samples are curated from the country with the most prominent presence

of each specific culture. We carefully selected 100 languages across 73 countries and five continents, striking a balance between low- and high-resource languages and between Western and non-Western cultures to ensure linguistic and cultural diversity. The selection criteria also consider the number of native speakers and the availability of speakers to validate and annotate the curated samples.

To accurately capture the cultural nuances of each language, we follow [32] and link each language to a country based on the World Values Survey [16] to cover social, ritualistic, and cultural values. For multilingual countries like India, multiple languages are associated to capture their cultural diversity. For each country-language pair, we collect cultural samples for all 19 generic and cultural domains.

Generic VQA Curation. Our ALM-bench has both generic and cultural questions. To evaluate visual understanding in 100 languages on generic scenes, e.g., indoor and outdoor images, we source examples from a well-known Englishonly LMM benchmark LLaVA-Bench (In-the-Wild) [26]. These English instructions are extended to the remaining 99 languages using GPT-4o followed by manual correction from native speakers. In some cases where the GPT-4o translations are poor, the annotators also translate the instructions from scratch. It results in a total of 6,000 openended samples across 100 languages for generic VQA.

Cultural Image Curation. To curate diverse cultural image-QA pairs, we collect open-licensed images and their corresponding metadata from the Internet, focusing on specific country-language pairs across various cultural domains. For each domain, we go through several filtration steps, e.g., removing low-resolution images and removing

|General VQA Pairs (English)|
|---|

||[Figure 2]|
|---|
<br><br>|[Figure 3]|
|---|
<br><br>|[Figure 4]|
|---|
<br><br>|[Figure 5]|
|---|
<br><br>Festivals Architecture Religion<br><br>Notable Key Figures Food Sports<br><br>|[Figure 6]|
|---|
<br><br>|[Figure 7]|
|---|
|
|---|

|ALM-Bench (Ours)<br><br>• Diverse Question Types<br>• 6K general VQA Pairs<br>• 16.7K cultural VQA Pairs<br>|
|---|

|Generated Cultural QA Pairs (English)|
|---|

[Figure 8]

Image Captioning Retrieval

[Figure 9]

[Figure 10]

[Figure 11]

Automated

Blurring Personal Information

Image Retrieval

[Figure 12]

100 Languages 73 Countries 13 Cultural Domains

[Figure 13]

|Translated VQA Pairs (100 Languages)|
|---|

|Translated Cultural VQA Pairs|
|---|

Manual

Human Verification & Translation Correction

Filtration

Cultural Images (x100 languages/dialects)

- Figure 3. Data collection and verification pipeline. Our benchmark features both cultural specific content sourced from the web (left) and generic image understanding collection sourced from existing LMM benchmark. The cultural part is carefully filtered to remove noisy samples and private information. We use GPT4o for translations which are manually verified and corrected with over 800 hours of human annotators (native speakers). Our ALM-bench has diverse question types and features approximately 23K QA pairs in total in 100 languages.

[Figure 14]

[Figure 15]

2,929

Images

22,763

Q/As

MCQs → 5,619 True/False → 2,805 Short QAs → 5,624 Long QAs → 8,715

Avg. Ques/Image → 4.43 Avg. Words/Ques →12.05 Avg. Words/Ans → 42.65 Total Categories → 18

100 languages 73 countries 24 scripts

[Figure 16]

[Figure 17]

[Figure 18]

- Figure 4. Data statistics of our ALM-bench showing the diversity of the scripts, global coverage, comprehensive categories, and various question types. Our dataset contains 22.7K high-quality question-answers in total, covering 100 languages and 24 scripts. All the samples are manually verified by native speakers.

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Language error

Semantic error

Cultural error

Grammatical error

0 50 100 150 200 250 300

Number of Mistakes

Figure 5. Analysis of the type of mistakes GPT-4o made while language translations via human-feedback. GPT-4o encounters more issues with respect to semantic and grammatical errors.

sen option rather than just the leading alphabet. The detailed instructions are presented in Sec. I (suppl. material.)

Filtration and Translation Errors. Given the QA pairs, we translate them to remaining 99 languages using GPT4o, followed by manual correction from native speakers. Similar to image filtering, native speakers assess translation quality and make detailed corrections to provide culturally grounded answers focusing on core cultural aspects. They also discard redundant and culturally irrelevant QAs.

redundant samples to ensure diversity. To maintain highquality and cultural relevance, we ask expert native speakers of each language to manually verify the quality and cultural diversity of these images. Images lacking cultural relevance are removed from the dataset. This manual filtration process discarded nearly 10.7% of the images. To ensure privacy, we blur any personally identifiable details (PIDs) and textual information in image that may directly leak the answer. Fig. 3 shows our data collection and verification pipeline.

We note several challenges in GPT-4o’s translations across 100 languages. Specifically, GPT-4o demonstrates difficulty in generating culturally grounded QA pairs across diverse languages. To analyze these issues, mistakes from the GPT-4o model in translations are categorized into four types: semantic error, cultural error, language error, and grammatical error. We sample 57 question-answer pairs from 51 randomly selected languages and plotted error distribution in Fig. 5. Notably, GPT-4o encounters more issues with semantic and grammatical accuracy when translating into different languages. We show examples of such errors in Fig. A. 2. (suppl. material).

Cultural QA Generation. ALM-bench dataset covers different question types. To generate high-quality cultural questions, we utilize GPT-4o to create QA pairs based on the provided images and their metadata. Notably, images and QAs are not shared across languages in the cultural set. For each image, we generate two MCQs, two short questions, a long question, and a true/false question in English. We instruct GPT-4o to emphasize the cultural concepts depicted in each image, generate questions that require a visual understanding of an image, and not perpetuate bias and stereotypes. To mitigate the randomness in multiple choice and true/false, we ensure that these questions can also be answered when converted to open-ended questions [37]. We also evaluate with shuffled orders, predicting the entire cho-

### 4. Benchmarking LMMs on ALM-bench

As discussed earlier, ALM-bench comprises four different question types: MCQs, T/F, Short and Long VQA. Therefore, we employ different prompts for each type of question. We use GPT-4o as a judge and prompt it to score responses of various question types with different criterion.

GPT-4o Gemini

[Figure 19]

87 84 51 79 70 79 81 81 81 79 82 83 84 80 85 78 83 85 86 86 87 83 88 82 85 81 86 87 81 87 84 76 70 69 84 84 85 82 63 85 78 88 68 81 72 80 66 81 57 80 62 79 85 83 79 86 74 85 70 80 80 80 61 82 85 63 72 79 86 84 81 86 83 66 81 77 86 67 76 63 83 84 77 84 78 81 87 78 73 75 84 86 85 82 58 83 84 79 71 66 84 78 53 77 72 72 77 78 75 72 78 78 70 76 80 74 67 79 85 81 80 60 85 67 81 78 69 84 80 83 77 78 70 73 66 69 80 79 71 83 74 85 59 76 70 74 60 78 71 64 70 63 79 82 79 83 69 79 76 64 79 75 70 75 81 68 63 74 83 72 80 67 81 64 81 73 81 71 71 70 81 81 74 78 60 76 82 77 70 59 80 78 81 73 61 80 79 76 73 68 70 52 16 48 41 57 44 53 46 44 64 63 64 47 77 69 64 70 72 69 57 54 80 53 56 63 75 70 49 75 62 40 44 36 54 45 64 46 40 62 34 75 58 61 37 38 28 58 46 39 20 68 49 51 55 65 30 68 42 43 42 27 29 41 67 43 40 47 71 69 42 71 64 32 59 45 68 44 28 26 69 58 41 68 60 40 73 35 33 46 43 68 70 41 28 59 69 49 36 31 61 38 27 41 33 46 33 53 44 39 61 58 60 40 75 65 57 65 62 68 62 57 77 40 45 50 73 70 45 72 49 45 31 35 53 47 53 38 39 68 33 73 54 60 44 33 25 56 32 39 37 58 47 49 49 60 32 68 50 36 43 35 33 47 59 38 39 39 64 69 41 59 64 30 59 30 63 30 33 24 59 54 37 71 51 33 68 34 34 46 61 61 63 42 33 41 65 36 38 34 53 47 32 24 37 32 41 47 41 50 52 50 49 51 61 62 49 54 53 56 38 46 77 32 53 39 64 57 31 58 39 34 39 41 37 44 33 32 34 57 45 58 61 53 36 33 33 52 47 31 39 58 46 44 44 55 43 57 37 47 44 36 23 48 50 39 44 42 58 54 31 50 61 40 43 47 44 47 37 39 53 51 38 58 50 41 54 31 23 24 40 51 50 44 33 41 50 36 46 42 60 34 24 34 39 33 31 43 48 45 51 48 55 59 71 63 45 58 54 64 48 44 76 38 54 38 71 65 30 67 37 37 33 40 43 43 38 33 40 65 29 63 58 50 38 29 29 51 33 37 33 55 42 40 44 52 34 66 40 39 42 32 24 42 49 42 43 42 56 70 34 54 55 35 47 31 49 43 38 30 50 42 37 65 48 33 58 43 31 35 48 30 52 38 32 34 59 38 45 36 45 38 22 32 32 30 33 29 37 35 47 44 50 39 64 54 47 47 50 58 29 32 71 40 38 33 58 50 38 57 37 26 33 36 31 41 41 39 35 51 38 56 51 45 32 26 29 43 34 29 33 46 39 34 45 44 34 53 31 39 30 24 28 38 48 25 31 27 45 53 23 48 49 32 30 39 50 38 31 25 42 44 32 58 44 30 55 28 29 34 38 36 44 30 24 38 40 36 27 29

80

GLM-4V-9B Qwen-2-VL

70

Molmo LLaVA-OneVision

60

AverageScore

Qwen-VL LlaVa-v1.6

50

- 53 37 14 20 24 31 26 41 18 21 54 50 55 35 56 52 52 51 59 56 30 35 68 29 40 50 59 55 17 60 28 18 35 36 32 30 46 37 37 57 21 64 47 47 16 27 28 41 26 36 19 52 32 34 43 44 28 57 29 35 21 28 12 29 57 23 35 33 57 62 13 52 51 25 34 26 49 35 22 18 49 52 29 59 37 23 60 38 18 18 23 43 55 25 23 28 48 32 28 37

- 45 40 19 28 15 34 31 32 16 22 44 40 45 35 61 59 42 48 47 54 35 32 66 37 39 37 54 52 23 50 34 17 24 30 32 27 35 37 31 50 35 57 44 41 18 23 25 37 26 25 18 45 35 35 41 37 26 42 22 38 20 25 19 27 48 17 30 26 47 50 25 45 46 17 35 35 48 29 24 21 42 39 32 51 39 31 53 30 19 33 31 42 47 29 25 30 41 31 26 27

54 25 16 8 13 29 26 37 15 23 55 49 53 32 55 52 53 56 58 59 31 30 67 26 38 48 63 55 10 59 24 21 27 23 22 30 48 26 22 56 18 64 44 40 20 19 16 44 25 23 15 50 25 20 41 49 15 55 17 29 22 24 7 22 58 18 27 40 53 58 17 53 50 19 27 24 54 22 17 15 48 49 26 58 31 25 61 22 18 44 19 40 55 27 19 27 48 22 16 18

- 46 23 8 12 30 24 22 45 41 45 43 47 47 29 54 49 42 41 49 56 51 52 64 26 24 37 56 50 18 54 26 8 26 31 28 52 35 30 18 48 21 56 42 39 11 31 19 32 20 34 12 53 23 18 43 50 27 44 11 37 33 22 8 36 46 9 37 39 44 48 8 46 50 34 50 28 44 26 30 5 40 38 32 58 38 23 49 36 10 39 10 28 48 43 23 34 38 27 24 18

MiniCPM-V-2 Eagle-X5-13B-Chat

40

PALO Pixtral-12B

30

- 40 32 8 26 15 35 28 37 21 33 38 40 50 34 43 39 39 33 43 44 21 25 74 30 38 33 69 64 27 64 29 14 28 30 25 37 34 31 25 42 28 69 46 38 24 30 23 35 16 25 7 42 34 30 42 37 22 41 33 26 25 23 14 33 41 9 29 26 39 66 22 40 56 19 24 26 39 29 15 9 32 35 27 66 30 30 43 28 21 20 28 36 51 22 19 34 38 30 19 24
- 41 33 21 18 21 28 30 22 22 30 32 39 44 37 49 40 29 35 33 43 33 28 72 29 41 25 65 53 24 58 29 30 34 30 28 29 23 28 32 40 24 59 39 34 27 18 25 30 30 18 25 47 29 30 30 33 29 44 28 26 19 25 20 23 34 31 34 28 33 60 28 34 44 28 26 28 34 34 31 21 29 36 36 60 38 33 38 29 26 30 26 31 28 28 25 33 27 34 34 32 31 31 28 34 29 29 34 34 34 33 33 34 31 28 28 26 28 34 35 36 33 34 36 36 28 39 38 35 35 36 29 27 27 29 36 36 31 33 28 33 25 38 28 36 33 31 25 30 31 34 35 29 35 39 30 39 25 34 33 24 37 26 32 32 34 24 31 35 34 33 27 37 36 29 33 25 34 26 24 31 37 34 26 35 36 35 40 23 37 35 39 35 37 33 16 32 30 24 32 22 41 33 22 22 24 25 19 29 18 22 40 39 39 37 40 42 40 40 48 45 31 30 58 26 32 37 44 41 22 49 24 28 20 34 24 28 31 30 24 41 20 47 41 38 21 28 21 35 23 32 23 43 27 24 34 30 17 43 30 29 26 24 14 25 49 26 27 31 43 48 19 40 43 20 28 24 34 30 27 19 33 38 19 47 29 29 51 24 23 35 22 24 44 21 19 25 37 26 25 19

Phi3-Vision-128k mBLIP-mt0-xl LLaVA-1.5-7B InternVL2-8B

20

10

29 25 16 15 19 22 18 28 24 27 29 26 35 28 50 39 27 34 33 35 25 22 51 26 28 24 45 41 19 40 22 26 20 17 27 26 24 23 17 34 18 43 39 29 19 20 19 30 21 20 17 27 22 24 25 29 17 37 21 22 20 18 15 25 31 24 23 25 30 38 24 32 31 24 25 25 32 22 18 17 30 28 20 41 25 18 35 21 22 22 23 34 36 24 21 24 25 24 20 23

Yiddish

Polish

Persian

Portuguese

Telugu

Hebrew

Greek

Russian

Uyghur

Romanian

Sindhi

Igbo

Shona

French

Croatian

Pashto

Hungarian

Sundanese

Hindi

Spanish

Icelandic

Serbian

English

Cebuano

Czech

Dutch

Odia

Afrikaans

Sinhala

Slovak

Slovenian

Yoruba

Irish

Indonesian

Swedish

Thai

Hausa

Macedonian

Uzbek

Bhojpuri

Malay

Nepali

Japanese

Javanese

Kinyarwanda

Basque

Albanian

Mongolian

SaudiArabic

Swahili

Danish

Lao

Bulgarian

Bengali

Norwegian

EgyptianArabic

Estonian

Vietnamese

Ukrainian

Sanskrit

Bosnian

Italian

Malagasy

Amharic

Galician

Lithuanian

Hawaiian

Latin

Somali

Latvian

Marathi

Catalan

Korean

Urdu

Gujarati

Belarusian

Tajik

Kannada

Azerbaijani

Maltese

Malayalam

Finnish

ScotsGaelic

Punjabi

Tamil

Kazakh

Chinese(Traditional)

Georgian

Filipino

EmiratiArabic

Assamese

German

Armenian

Turkish

Kyrgyz

Luxembourgish

Chinese(Simplified)

Welsh

Kurdish

Myanmar(Burmese)

Figure 6. ALM-bench Performance comparison of different open and closed-sourced models (y-axis) on the 100 languages (x-axis) of our ALM-bench. The performance is represented as an average accuracy across all questions in a language. The actual performance of a model on a language is shown in each respective box, where the higher accuracy is highlighted with a high color intensity. The closed-source propriety models generally perform better across languages compared to their open-sourced counterparts. The performance on several high-resource languages (e.g., English, French, Chinese, and Spanish) is generally higher throughout different models, whereas all opensource models struggle on low resource languages (e.g., Amharic, Kinyarwanda, Burmese) fs QAW a Overall, GPT-4o and GLM-4V-9B performs better in terms of closed-source and open-source models, respectively. Best viewed zoomed in.

We use accuracy for MCQs and T/F, correctness for short VQAs (SVQAs), and consistency, fluency, and relevance for long VQAs (LVQAs). Here, correctness refers to how closely the model’s output aligns with the ground-truth [26]. For LVQAs, the consistency assesses whether the predicted answer is coherent across the entire generated output [19]. The fluency metric measures the naturalness and readability of model prediction [38], whereas the relevancy metric determines whether model prediction provides answers directly related to ground truth [6]. To ensure fair evaluation for decision-making questions (T/F and MCQs) using GPT4o as a judge, we also show consistent performance using LLaMA-3.8.1B-Instruct [13] in Tab. A.2. (suppl. material). Overall Results: Fig. 6 presents the per-language performance of 16 recent LMMs on the ALM-bench. The results offer a number of insights: (a) the closed-source propriety models (GPT-4o [2] and Gemini-1.5-Pro [42]) generally perform better across the 100 languages, compared to their open-source counterparts. The best closed-source model, GPT-4o, achieves an overall accuracy of 78.8% compared to 51.9% overall accuracy obtained by the best performing open-source model GLM-4V-9B [15]. Both closed-source models struggle with several low-resource languages (e.g., Amharic, Kinyarwanda, Burmese, and Sanskrit). For instance, the performance of GPT-4o significantly drops from 88.4% for the English language to 50.8% for the Amharic language. (b) Several open-source models (GLM-4V-9B [15], Qwen2-VL [44], Molmo [12], LLaVA-OneVision [23]) achieve comparable overall performance. Similar to their closed-source counterparts, open-source models also struggle with low-resource languages. For instance, the performance of GLM-4V-9B [15] dramatically drops from 80.3% for the English language to 15.6% for the Amharic language. Fig. 2a presented earlier further shows the overall performance breakdown in terms of high-and low-resource languages across different models. It is worth mentioning that our ALM-bench comprises 50 high-resource and 50 low-resource languages following the definition of lowresource languages as in [10]. (c) Fig. 2a reveals a sub-

[Figure 20]

Figure 7. Performance comparison of GPT-4o and Qwen2-VL on different language scripts. The count above each bar indicates the number of languages present in that script.

stantial performance gap between high-resource and lowresource languages consistent across different models, with GPT-4o showing a 6% drop. This performance gap extends to more than 8% in case of some open-source models (e.g., GLM-4V-9B [15], Qwen2-VL [44]). A notable exception is the performance of closed-source Gemini-1.5-Pro [42], which does not deteriorate significantly for low-resource languages. Next, we present further results analysis.

How important is visual context? To investigate this question, we ran an experiment with only the base LLM of various LMMs used in our evaluation. We randomly sample 50 languages and prompt the respective LLM with a textual question without the visual input. Our results demonstrate that the visual context is important to answer several questions in our benchmark and the LLMs struggle without input images. For instance, GPT-4o [2] performance drops by absolute 27.3% when prompted with textual questions alone. Specifically, it shows a significant performance gain in languages such as Sinhala (38.7%), Sanskrit (50%), and Dutch (40%) when images are included. Similarly, Qwen27B gives a 13% absolute and 24.8% relative drop in performance without image inputs. This substantial performance drop underscores the robustness of our visual benchmark, revealing that without images, even the best-performing proprietary and open-source models LLMs struggle to answer accurately.

##### Comparison across scripts: We further group the 100 lan-

|Bengali: Bengali Script<br><br>[Figure 21]<br><br>Lack of Cultural Understanding 65%<br><br>Lack of Knowledge 27%<br><br>Reasoning Error 2%<br><br>Translation Error 1%<br><br>Perceptual Error 4% Language Error 1%<br><br>|Sinhala: Sinhalese Script<br><br>[Figure 22]<br><br>Lack of Knowledge 31%<br><br>Lack of Cultural Understanding 18%<br><br>Language Error 36%<br><br>Perceptual Error 5%<br><br>Translation Error 3%<br><br>Reasoning Error 5%<br><br>|Portugese: Latin Script<br><br>[Figure 23]<br><br>Reasoning<br><br>Error 29%<br><br>Lack of Knowledge 26%<br><br>Lack of Cultural Understanding 23%<br><br>Translation Error 6%<br><br>Perceptual Error 11%<br><br>Language Error 6%<br><br>|Russian: Cyrillic Script<br><br>[Figure 24]<br><br>Lack of Knowledge<br><br>68%<br><br>Lack of Cultural Understanding 14%<br><br>Language Error 5%<br><br>Perceptual Error 5%<br><br>Translation Error 3%<br><br>Reasoning<br><br>Error 5%<br><br>|
|---|---|---|---|

- Figure 8. Error analysis across 4 diverse language scripts, including Bengali, Sinhalese, Latin and Cyrillic on GPT-4o results, demonstrates significant challenges for even the top-performing closed-source models, particularly in cultural and reasoning comprehension. The ALM-bench highlights these gaps, especially in languages with complex dialectal variations.

| |(5) (1)<br><br>(1)<br><br>(6) (4) (1) (8)<br><br>(3) (2) (8) (55)<br><br>(1) (1) (3)<br><br>(1)<br><br>(5) (1)<br><br>(1) (6) (4) (1) (8) (3) (2) (8) (55) (1) (1) (3) (1)<br><br>Model<br><br>Qwen2-VL<br><br>GPT-4o| | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

Atlantic-CongoMongolic-KhitanIsolate TurkicDravidianKartvelianAfro-Asiatic UralicTai-KadaiAustronesianIndo-EuropeanJaponicKoreanicSino-TibetanAustroasiatic

0

20

40

60

80

Performance(%)

- Figure 9. Performance comparison of GPT-4o and Qwen2-VL on different language families. The count above each bar indicates the number of languages present in that script.

its role in Buddhist festivals, the response is incorrectly written in the Sinhala language. This highlights GPT-4o’s language error with the Sinhalaese script, demonstrating a gap in script proficiency despite grasping cultural context.

Comparison across language families: We also analyze the performance of LMMs by grouping the different languages with respect to 15 families taken from Britannica* as shown in Fig. 9. Results show that performance on several African (Atlantic-Congo) languages such as Igbo, Kinyarwanda, Shona, Swahili, and Yoruba is inferior compared to several Asian (e.g., Chinese, Korean, Vietnamese) and Western languages (e.g., English, French, German).

Effect of question types: We perform this analysis on two closed-source (GPT-4o [2] and Gemini-1.5-Pro [42]) and two open-source (GLM-4V-9B [15], Qwen2-VL [44]) models, as shown in Fig. 11. Overall, we observe that all models fare better on decision-making questions (MCQs and T/F questions). Closed-source models perform better on long VQAs (LVQAs) than short VQAs (SVQA). We observe this trend to be the opposite in open-source models as they struggle to generate accurate and fluent long responses in a multilingual setting with 100 languages.

guages in ALM-bench by script [10]. We use data from Ethnologue [20] and the Glottolog database [17] on language scripts, origins, and families. This resulted in 24 distinct script groups. Fig. 7 shows GPT-4o and Qwen2-VL performance across these scripts, with both models struggling particularly on low-resource scripts such as, Ge’ez (Amharic), Sinhalese (Sinhala), Oriya (Odia), and Myanmar (Myanmar-Burmese). While the results are not surprising as they highlight that existing models are being predominantly trained on high-resource languages, the extent of decline for low-resource scripts is noteworthy.

Are LMMs Culturally Aware? We study the cultural understanding of LMMs on our 13 domains and report results in Fig. 2b. Overall, GPT-4o achieves the best results with an overall score of 80.3%. However, we observe a significant variation in performance across different cultural domains. For instance, GPT-4o achieves 83.7% on Education and heritage but drops to 72.7% on Notable key figures category. We conjecture this variation in performance likely arises since the higher-performing categories are covered in LMM training datasets, leading to better representation learning. In contrast, categories like Notable Key Figures and Customs are often culturally specific and nuanced across languages and regions.

We conduct an error analysis on the cultural examples from ALM-bench by selecting one language per script and having native speakers review GPT-4o (best-performing model) responses. Errors were categorized into six types: lack of knowledge, reasoning error, perceptual error [47], language error, translation error, and lack of cultural understanding. Fig. 8 summarizes these error types across four scripts, showing that the primary issues involve knowledge gaps, reasoning errors, and cultural misunderstandings. Fig. 10 further illustrates specific errors across scripts. One such example in Sinhala asks about the type of music group that uses two traditional drums, the Thammattama and Dhawula, which are commonly played by Hewisi music groups associated with traditional Buddhist festivals. Although GPT-4o correctly answered Hewisi and recognized

Impact of location-aware information in prompts: We show the performance of LMMs when evaluated with addi-

*https://www.britannica.com/browse/Languages

|Saudi Arabic: Arabic Script|Afrikaans: Latin Script|Indonesian: Latin Script|Tamil: Tamil Script|
|---|---|---|---|
|Question: ؟ةيديلقتلا سبلاملا نودتري ةروصلا يف لاجرلا له<br><br>Ground Truth Answer: حص<br><br>Predicted Answer:<br><br>حص Score:<br><br>10<br><br>|[Figure 25]|
|---|
|Question:<br><br>In watter provinsie van Suid-Afrika is die persoon op die beeld gebore?<br><br>Ground Truth Answer:<br><br>Oos-Kaap (Opsies: Oos-Kaap, Wes-Kaap, Gauteng, KwaZulu-Natal)<br><br>Predicted Answer: Oos-Kaap<br><br>Score:<br><br>10<br><br>|[Figure 26]|
|---|
|Question:<br><br>Di manakah pria Muslim Indonesia kemungkinan besar beribadah?<br><br>Ground Truth Answer:<br><br>Di alam terbuka (Pilihan: Di alam terbuka, Di dalam masjid, Di pasar, Di ruang kelas)<br><br>Predicted Answer: Di alam terbuka<br><br>Score:<br><br>10<br><br>|[Figure 27]|
|---|
|Question:<br><br>படத்தில் காட்டப்பட்டுள்ள படத்தின் தலலப்பு என்ன?<br><br>Ground Truth Answer:<br><br>தளபதி<br><br>Predicted Answer:<br><br>படத்தின் தலலப்பு "தலபதி".<br><br>Score:<br><br>9<br><br>|[Figure 28]|
|---|
|
|Category: Lifestyle|Category: Notable Key Figure|Category: Religion|Category: Media|

|Urdu: Arabic Script|Sinhala: Sinhalese Script|Amharic: Geʽez Script|Japanese: Japanese Script|
|---|---|---|---|
|Question:<br><br>؟ےہ اہر اج ايانم راوہت اس نوک ںيم ريوصت<br><br>Ground Truth Answer:<br><br>ںاغارچ لايم<br><br>Predicted Answer:<br><br>یک نشج ےک یبنلا دلايم ديع ريوصت ہي ۔ےہ ايگ اياجس ےس ںولوھپ روا ںوينشور وک ترامع ںاہج ےہ یتاھکد بيرقت<br><br>Reason for Error:<br><br>Lack of Cultural Understanding<br><br>|[Figure 29]|
|---|
||[Figure 30]|
|---|
<br><br>Question:<br><br>මෙෙ මෙර වර්ග භා විතා කරන සංගීත කණ්ඩා යෙක් හ ඳුන්වන නෙ කුෙක්ද?<br><br>Ground Truth Answer:<br><br>මේවිසි කණ්ඩායෙක් වන අතර, සාොනයමයන් මෙෞද්ධ උත්සව වලදී වා දනය කරයි.<br><br>Predicted Answer:<br><br>මේවිසි මලස හදුන්වන අතර,<br><br>මෙෞද්ධ උත්සව වල තිමේ<br><br>Reason for Error:<br><br>Language Error|Question:<br><br>በምስሉ ላይ የትኛው ባህላዊ አለባበስ ይታያል?<br><br>Ground Truth Answer: ነጭ ሻርኮች (አማራጮች፡ ነጭ ሻርኮች፣ ሰማያዊ ሳሪስ፣ ቀይ ማቀፊያዎች፣ አረንጓዴ ኪሞኖዎች)<br><br>Predicted Answer: ነጭ ሻርኮች።<br><br>Reason: Language Error<br><br>|[Figure 31]|
|---|
|Question: この画像に描かれているのは誰ですか?<br><br>Ground Truth Answer: この画像には織田信長が描かれていま す。<br><br>Predicted Answer: この画像に描かれている人物が誰かは わかりません。<br><br>Reason: Lack of Knowledge<br><br>|[Figure 32]|
|---|
|
|Category: Festivals|Category: Music|Category: Lifestyle|Category: Notable Key Figures|

- Figure 10. We present the qualitative examples of the success cases in the first row and failure cases of GPT-4o in the second row on different languages & domains in ALM-bench. For the failure cases, we specify different error types. For instance, the Urdu language question asks about the festival depicted in the image. The image specifically refers to Mela Chiraghan (Festival of Lights), a celebration held in honor of the Sufi saint Shah Jamal’s shrine. Since the decoration in the image closely resembles that of Eid Milad un Nabi — another religious festival—the model erroneously associates it with this wrong event. This constitutes a lack of cultural understanding since the model fails to distinguish between the theme behind the decorations. Eid Milad un Nabi typically features more modest, reverential lighting with green lights, whereas the lighting in Mela Chiraghan is brighter and more colorful. Additionally, people typically dress for the Eid Milad un Nabi event in a traditional outfit which is absent in the image. These examples highlight the model’s gap in cultural knowledge and its limitations in terms of accurately interpreting the cultural context of the given sample. Additional results are in the suppl. material.

| |81%<br><br>76%<br><br>48%<br><br>35% 34%<br><br>72%<br><br>66%<br><br>46%<br><br>43%<br><br>31%<br><br>90%<br><br>81%<br><br>67% 66%<br><br>57%<br><br>91%<br><br>79% 79%<br><br>74%<br><br>67%<br><br>Question Type<br><br>Long Question<br><br>Short Question<br><br>Multiple Choice<br><br>True False| | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

GPT-4o Gemini-1.5-Pro GLM-4V-9B Qwen2-VL LLaVA-OneVision

0

20

40

60

80

Performance(%)

- Figure 11. Performance of top-performing VLMs on ALM-bench according to different question types. Compared to open-ended questions, VLMs perform better on MCQs and T/F questions.

Models With Country Info. Without Country Info.

GPT-4o 83.57% 80.96% Gemini-1.5-Pro 81.52% 76.19% GLM-4V-9B 56.78% 56.41% Qwen2-VL 53.97% 52.57%

Table 2. Performance of LMMs with and without additional country location information. Proprietary models show a notable performance boost of 2.6% to 5% when location-aware prompts are used, while open-source models exhibit a marginal improvement.

derrepresented domains. However, we believe ALM-bench represents a step forward in standardized multilingualLMM evaluation, paving the way for greater inclusion of diverse cultural domains, languages, scripts, and origins.

tional country-specific information in Tab. 2. We randomly select 50 languages from the ALM-bench and include country information in the prompts to assess any potential performance gains. Here, we observe that closed-source models, including GPT-4o and Gemini-1.5-Pro, better utilize the added geographic context and indicate higher cultural specificity across languages than the open-source models.

Translations. We use GPT-4o [2] for translations in 100 languages. The initial translations generated by GPT-4o undergo a thorough review by native speakers to ensure linguistic accuracy and cultural relevance to maintain highquality standards across all language pairs. This combined approach enables consistent and accurate translations.

### 5. Ethical Consideration

Human Annotations. To ensure high-quality image filtering and language annotation, we collaborated with linguistic groups and finalized over 60 volunteers for bench-

Data. Our cultural images in ALM-bench are sourced from the internet. Therefore, there may be potential biases in un-

mark curation. For most languages, annotators were required to be either native or bilingual speakers. However, for certain low-resource languages like Igbo, Hungarian, and Afrikaans, proficiency was also acceptable. Annotators were required to be familiar with the cultural context of the specific country-language pair they were working with. Demographic details about the expert annotators are provided in Sec. B (suppl. material). Among them, 80.3% are native speakers, and 87.9% have lived for over 15 years in countries where the target language is spoken, thereby ensuring a deeper cultural insight. Furthermore, 46.7% of annotators both male and female, fall within the 18–25 age bracket.

### 6. Conclusion

In this paper, we introduce ALM-bench, a novel multilingual multimodal cultural benchmark for evaluation with over 22.7k humanly verified samples across 19 domains. Our benchmark encompasses cultural nuances from 73 countries in 24 language scripts and 15 language families. We conduct empirical analysis on 16 vision-language models with various question types (MCQs, T/F, SVQA, and LVQAs) and highlight notable disparities in their performance.

The performance difference between the bestperforming open-source model and the proprietary model, GPT-4o, is 27%. Our results also highlight that the models perform superior on predominant language scripts such as Latin, Cyrillic, and Devanagari and under-performs on underrepresented scripts such as Ge’ez, Lao, Sinhalese, and Oriya. Moreover, cultural understanding of prominent language families such as Indo-European, Austronesian and Afro-Asiatic are well represented by GPT-4o as compared to Atlantic-Congo and Turkic families. Our work highlights the limitations of state-of-the-art LMMs in multilingual and multicultural settings, showing key areas for improvement.

### 7. Acknowledgment

We would like to thank Tafar Mab for providing highquality annotations for Afrikaans and Kinyarwanda languages, Georgios Ioannides for verifying Greek language, Marek Suppa for annotating and verifying Czech, Slovak and Polish language, and Feno Heriniaina Rabevohitra for annotating Malagasy language. We also thank Yuhao Li for reviewing and assisting with the annotations for Chinese Simplified language. Finally, we extend our gratitude to the anonymous reviewers for their invaluable suggestions and feedback, which helped improve this paper. The computations were enabled by resources provided by NAISS at Alvis partially funded by Swedish Research Council through grant agreement no. 2022-06725, LUMI hosted by CSC (Finland) and LUMI consortium, and by Berzelius resource provided by the Knut and Alice Wallenberg Foundation at the NSC. This work was partially supported by

by the Swedish Research Council (2022-04266) and from KAW (DarkTree project; 2024.0076).

### References

- [1] Mohammad Amin Abbasi, Arash Ghafouri, Mahdi Firouzmandi, Hassan Naderi, and Behrouz Minaei Bidgoli. Persianllama: Towards building first persian large language model. arXiv preprint arXiv:2312.15713, 2023. 3
- [2] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774,

2023. 2, 3, 6, 7, 8

- [3] Muhammad Farid Adilazuarda, Sagnik Mukherjee, Pradhyumna Lavania, Siddhant Singh, Alham Fikri Aji, Jacki O’Neill, Ashutosh Modi, and Monojit Choudhury. Towards measuring and modeling" culture" in llms: A survey. arXiv preprint arXiv:2403.15412, 2024. 4, 12
- [4] Fakhraddin Alwajih, El Moatez Billah Nagoudi, Gagan Bhatia, Abdelrahman Mohamed, and Muhammad AbdulMageed. Peacock: A family of arabic multimodal large language models and benchmarks. arXiv preprint arXiv:2403.01031, 2024. 2, 3, 4
- [5] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023. 2
- [6] Souradip Chakraborty, Soumya Suvra Ghosal, Ming Yin, Dinesh Manocha, Mengdi Wang, Amrit Singh Bedi, and Furong Huang. Transfer q star: Principled decoding for llm alignment, 2024. 6
- [7] Soravit Changpinyo, Linting Xue, Idan Szpektor, Ashish V Thapliyal, Julien Amelot, Michal Yarom, Xi Chen, and Radu Soricut. Maxm: Towards multilingual visual question answering. arXiv preprint arXiv:2209.05401, 2022. 4
- [8] Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330,

2024. 3, 4

- [9] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024. 2
- [10] Marta R Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, et al. No language left behind: Scaling human-centered machine translation. arXiv preprint arXiv:2207.04672, 2022. 6, 7
- [11] Rocktim Jyoti Das, Simeon Emilov Hristov, Haonan Li, Dimitar Iliyanov Dimitrov, Ivan Koychev, and Preslav Nakov. Exams-v: A multi-discipline multilingual multimodal exam benchmark for evaluating vision language models. arXiv preprint arXiv:2403.10378, 2024. 2, 3, 4

- [12] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, Jiasen Lu, Taira Anderson, Erin Bransom, Kiana Ehsani, Huong Ngo, YenSung Chen, Ajay Patel, Mark Yatskar, Chris CallisonBurch, Andrew Head, Rose Hendrix, Favyen Bastani, Eli VanderBilt, Nathan Lambert, Yvonne Chou, Arnavi Chheda, Jenna Sparks, Sam Skjonsberg, Michael Schmitz, Aaron Sarnat, Byron Bischoff, Pete Walsh, Chris Newell, Piper Wolters, Tanmay Gupta, Kuo-Hao Zeng, Jon Borchardt, Dirk Groeneveld, Jen Dumas, Crystal Nam, Sophie Lebrecht, Caitlin Wittlif, Carissa Schoenick, Oscar Michel, Ranjay Krishna, Luca Weihs, Noah A. Smith, Hannaneh Hajishirzi, Ross Girshick, Ali Farhadi, and Aniruddha Kembhavi. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models, 2024. 6
- [13] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783,

2024. 6

- [14] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models, 2024. 2, 3, 4
- [15] Team GLM, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Diego Rojas, Guanyu Feng, Hanlin Zhao, Hanyu Lai, et al. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793, 2024. 6, 7
- [16] C Haerpfer, R Inglehart, A Moreno, C Welzel, K Kizilova, J Diez-Medrano, M Lagos, P Norris, E Ponarin, and B Puranen. World values survey: Round seven-country-pooled datafile version 3.0. jd systems institute & wvsa secretariat,

2022. 4

- [17] Harald Hammarström, Robert Forkel, Martin Haspelmath, and Sebastian Bank. Glottolog database 4.6, 2022. 7
- [18] Fajri Koto, Haonan Li, Sara Shatnawi, Jad Doughman, Abdelrahman Boda Sadallah, Aisha Alraeesi, Khalid Almubarak, Zaid Alyafeai, Neha Sengupta, Shady Shehata, et al. Arabicmmlu: Assessing massive multitask language understanding in arabic. arXiv preprint arXiv:2402.12840,

2024. 15

- [19] Wojciech Kry´sci´nski, Nitish Shirish Keskar, Bryan McCann, Caiming Xiong, and Richard Socher. Neural text summarization: A critical evaluation, 2019. 6
- [20] Melvyn Lewis. Ethnologue: Languages of the World. 2009. 7
- [21] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125, 2023. 3, 4
- [22] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13299–13308, 2024. 3, 4

- [23] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 6
- [24] Yen-Ting Lin and Yun-Nung Chen. Taiwan llm: Bridging the linguistic divide with a culturally aligned language model. arXiv preprint arXiv:2311.17487, 2023. 3
- [25] Fangyu Liu, Emanuele Bugliarello, Edoardo Maria Ponti, Siva Reddy, Nigel Collier, and Desmond Elliott. Visually grounded reasoning across languages and cultures. arXiv preprint arXiv:2109.13238, 2021. 3, 4
- [26] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36, 2024. 4, 6, 13
- [27] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023. 2, 3, 4
- [28] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024), 2024. 2
- [29] Muhammad Maaz, Hanoona Rasheed, Abdelrahman Shaker, Salman Khan, Hisham Cholakal, Rao M Anwer, Tim Baldwin, Michael Felsberg, and Fahad S Khan. Palo: A polyglot large multimodal model for 5b people. arXiv preprint arXiv:2402.14818, 2024. 2
- [30] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and Roozbeh Mottaghi. Ok-vqa: A visual question answering benchmark requiring external knowledge. In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition, pages 3195–3204, 2019. 4, 12
- [31] Vishal Narnaware, Ashmal Vayani, Rohit Gupta, Sirnam Swetha, and Mubarak Shah. Sb-bench: Stereotype bias benchmark for large multimodal models. arXiv preprint arXiv:2502.08779, 2025. 3
- [32] Shravan Nayak, Kanishk Jain, Rabiul Awal, Siva Reddy, Sjoerd van Steenkiste, Lisa Anne Hendricks, Karolina Sta´nczak, and Aishwarya Agrawal. Benchmarking vision language models for cultural understanding. arXiv preprint arXiv:2407.10920, 2024. 2, 3, 4
- [33] Edwin G. Ng, Bo Pang, Piyush Sharma, and Radu Soricut. Understanding guided image captioning performance across domains. arXiv preprint arXiv:2012.02339, 2020. 3
- [34] Jonas Pfeiffer, Gregor Geigle, Aishwarya Kamath, JanMartin O Steitz, Stefan Roth, Ivan Vuli´c, and Iryna Gurevych. xgqa: Cross-lingual visual question answering. arXiv preprint arXiv:2109.06082, 2021. 4
- [35] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024. 3
- [36] Hanoona Rasheed, Muhammad Maaz, Sahal Shaji, Abdelrahman Shaker, Salman Khan, Hisham Cholakkal, Rao M.

- Anwer, Eric Xing, Ming-Hsuan Yang, and Fahad S. Khan. Glamm: Pixel grounding large multimodal model. The IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. 2
- [37] David Romero, Chenyang Lyu, Haryo Akbarianto Wibowo, Teresa Lynn, Injy Hamed, Aditya Nanda Kishore, Aishik Mandal, Alina Dragonetti, Artem Abzaliev, Atnafu Lambebo Tonja, et al. Cvqa: Culturally-diverse multilingual visual question answering benchmark. arXiv preprint arXiv:2406.05967, 2024. 2, 3, 4, 5, 12
- [38] Ananya B Sai, Akash Kumar Mohankumar, and Mitesh M Khapra. A survey of evaluation metrics used for nlg systems. ACM Computing Surveys (CSUR), 55(2):1–39, 2022. 6
- [39] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 3
- [40] Swetha Sirnam, Jinyu Yang, Tal Neiman, Mamshad Nayeem Rizve, Son Tran, Benjamin Yao, Trishul Chilimbi, and Mubarak Shah. X-former: Unifying contrastive and reconstruction learning for mllms. In European Conference on Computer Vision, pages 146–162. Springer, 2024. 3
- [41] Hai-Long Sun, Da-Wei Zhou, Yang Li, Shiyin Lu, Chao Yi, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, DeChuan Zhan, et al. Parrot: Multilingual visual instruction tuning. arXiv preprint arXiv:2406.02539, 2024. 2, 3, 4
- [42] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 2, 3, 6, 7
- [43] Hongyu Wang, Jiayu Xu, Senwei Xie, Ruiping Wang, Jialin Li, Zhaojie Xie, Bin Zhang, Chuyan Xiong, and Xilin Chen. M4u: Evaluating multilingual understanding and reasoning for large multimodal models. arXiv preprint arXiv:2405.15638, 2024. 2, 3, 4
- [44] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 6, 7
- [45] Da Yin, Liunian Harold Li, Ziniu Hu, Nanyun Peng, and Kai-Wei Chang. Broaden the vision: Geo-diverse visual commonsense reasoning. arXiv preprint arXiv:2109.06860,

2021. 2, 3, 4

- [46] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023. 3, 4
- [47] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9556– 9567, 2024. 3, 4, 7
- [48] Wenxuan Zhang, Mahani Aljunied, Chang Gao, Yew Ken Chia, and Lidong Bing. M3exam: A multilingual, multimodal, multilevel benchmark for examining large language

- models. Advances in Neural Information Processing Systems, 36:5484–5505, 2023. 2, 3, 4
- [49] Zeliang Zhang, Mingqian Feng, Zhiheng Li, and Chenliang Xu. Discover and mitigate multiple biased subgroups in image classifiers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10906– 10915, 2024. 14
- [50] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019. 3

### Appendix A. ALM-Bench Categories

Our ALM-bench benchmark dataset consists of 19 categories. Among these categories, we include Food, Lifestyle, Religion, Architecture, Sports, Customs and Heritage from [30, 37] with some minor modifications to it. We further include six additional categories including Literature, featuring prominent authors, poets, and their notable works; Music, showcasing traditional music and dance through visual samples; Festivals, covering major cultural celebrations; Economy, representing local industries and businesses; Media, highlighting cultural icons, entertainment figures, and popular TV shows; and Notable Key Figures, featuring influential historical leaders who serve as representatives for country-language association. Table A. A.1 entails all our ALM-bench categories.

Following [3], we adopt an approach to group cultural attributes based on the country for each language. Additionally, we group culturally grounded elements representing shared knowledge, values, and objectives among the people in that culture that are collectively understood. We curated a culturally diverse dataset by collecting data from many cultural aspects (see Fig. A. A.1) and manually curated the caption for each image from web-sources. We also show the most frequent words from these categories in Fig. A.

- A.15.

ALM-bench Categories

- 1. Indoor
- 2. Outdoor
- 3. Food Items
- 4. Memes
- 5. Painting
- 6. Sketch
- 7. Food
- 8. Lifestyle
- 9. Religion
- 10. Literature

- 11. Music
- 12. Customs
- 13. Festivals
- 14. Heritage
- 15. Economy
- 16. Media
- 17. Architecture
- 18. Sports
- 19. Notable Key Figures

Table A.1. The 19 categories present in our ALM-bench dataset. Note that Food Items and Food appear in both generic and cultural categories, but have a different context as described in Sec. F.

- B. Annotators Demographics

Our ALM-bench is the most comprehensive benchmark to date, capturing cultural nuances across 100 languages. Since building such large-scale datasets is challenging, especially due to the limited availability of high-quality native language samples, we enlisted over 60 volunteers to provide expert feedback on our curated examples. In addition to our in-house team, we collaborated with multilingual research communities to bring in external volunteers. These

[Figure 33]

Figure A.1. A breakdown of the cultural categories from our ALM-bench is depicted. We ensure consistent samples across all subsets, except for the Economy category, where culturally unique images were challenging to find. These visual and questionanswer samples are verified and filtered by the native speakers, removing any culturally irrelevant and redundant information.

volunteers, representing 50 different countries, bring a deep understanding of their languages’ cultural elements and insights unique to their country-language pairs.

Each annotator was given detailed instructions along with examples to guide them in curating high-quality datasets. We also hosted a UI interface as well as gave them the option to directly use the dataset. An example of our annotation platform is shown in Fig. A. A.4. Annotators who made substantial contributions are recognized as co-authors of this paper to support high-quality language verification.

We summarize the annotator demographics in Fig. A. A.3. To summarize, we show the Age and Gender distribution of our volunteers (top-left) indicating the presence of one-fourth of Female annotators and over 46% fall in the 18-25 age limit. We show their levels of language proficiency (top-right). Notably, over 80% of the annotators were either native speakers or bilingual, with an additional 14.8% at a proficient level in their chosen language. This high level of expertise ensures that our cultural samples have been meticulously reviewed, filtered, and refined by knowledgeable volunteers. Additionally, we pre-select annotators who are accustomed to the cultural values of their languages. Moreover, (bottom-left) depicts that over 88.5% are culturally familiar with their languages, and bottomright highlights that 87.9% of the annotators have lived in the country where we have crafted the cultural samples, making them ideal participants for participating in this re-

words (e.g., country name, language name, cultural category) for each language. We also fetch meta-data along with the image, including image caption, image size, and image license. We use such information for post-processing to keep only high-quality images. Then, we perform manual filtering and blurring of personally identifiable data (PID). We discuss this step in detail in Sec. F. The cleaned images are used to generate and translate QA pairs using GPT-4o. To ensure quality, contributors with expertise in the respective native languages manually verify the cultural relevance of images, as well as the accuracy, relevance, and consistent translation of the generated QAs. This section outlines the instructions provided to contributors for verifying the relevance of questions generated for cultural domains and for reviewing the QA translations across both general and cultural domains. Additionally, we use GPT-4o as a judge to score the response and assess the accuracy of answers for cultural category images. This process is detailed further in Sec. J and K.

|Hindi: Devanagari Script|
|---|
|Original Question:<br><br>इस चित्र में चिखाए गए मंचिर की चिशेषता क्या है?<br><br>Original Answer:<br><br>कोणाकक सूर्क मंचिर में पत्थर पर की गई जचिल नक्काशी और सूर्किेि को समचपकत रथ का आकार प्रिचशकत है।<br><br>Corrected Question:<br><br>छचि में चिखार्ा गर्ा मंचिर कौन सी चिचशष्ट चिशेषता प्रिचशकत करता है?<br><br>Corrected Answer:<br><br>कोणाकक सूर्क मंचिर जचिल पत्थर की नक्काशी और भगिान सूर्क को समचपकत एक रथ की आकृ चत को प्रिचशकत करता है।<br><br>|[Figure 34]|
|---|
|
|Cultural Category: Architecture|

|Japanese: Japanese Script|
|---|
|Original Question:<br><br>画像に表示されている像に描かれた歴史的人物は誰ですか？<br><br>Original Answer:<br><br>徳川家康 (オプション: 徳川家康、 織田信長、豊臣秀吉、源頼朝)<br><br>Corrected Question:<br><br>画像に写っている像は、どの歴 史上の人物を表していますか？<br><br>Corrected Answer:<br><br>徳川家康 (選択肢：徳川家康、織田信長、豊臣秀吉、源頼朝)<br><br>|[Figure 35]|
|---|
|
|Cultural Category: Notable Key Figures|

|Sanskrit: Devanagari Script|
|---|
|Original Question:<br><br>कस्मिन्नेचह चिने रामनिमी इचत उत्सिः चित्रे आचिस्मिर्ते?<br><br>Original Answer:<br><br>िैत्रे उज्ज्वलपक्षस्य निमः । (चिकल्ाः : िैत्रे उज्ज्वलपक्षस्य निमः ।, िीपािली, मकरसंक्रास्मतः , िैत्रे कृ ष्णपक्षस्य पञ्चमी ।)<br><br>Corrected Question:<br><br>चित्रे िचशकतं रामनिमी कस्मिन् चिने आिर्कते?<br><br>Corrected Answer:<br><br>िैत्रे उज्ज्वलपक्षस्य निमी। (चिकल्ाः : िैत्रे<br><br>उज्ज्वलपक्षस्य निमी ।, िीपािली, मकरसंक्रास्मतः , िैत्रे कृ ष्णपक्षस्य पञ्चमी ।)<br><br>|[Figure 36]|
|---|
|
|Cultural Category: Food|

|Spanish: Latin Script|
|---|
|Original Question:<br><br>¿Cuál es el monumento icónico que se muestra en la imagen conocido como?<br><br>Original Answer:<br><br>La Alcazar por Sevilla.<br><br>Corrected Question:<br><br>¿Cómo se conoce el Monumento icónico que se muestra en la imagen?<br><br>Corrected Answer: El Alcázar de Sevilla.<br><br>|[Figure 37]|
|---|
|
|Cultural Category: Heritage|

- Figure A.2. Qualitative examples of various mistakes in GPT-4o translations including semantic, cultural, language and grammatical errors. We employ expert human-feedback to rewrite the correct translations for all samples in our ALM-bench dataset. search effort.

|[Figure 38]<br><br>|[Figure 39]|
|---|---|
|[Figure 40]|[Figure 41]|

- Figure A.3. top-left. depicts the Age and Gender distribution of

Each annotator is provided access to a dedicated verification platform that guides them through the list of QA pairs for the selected language (examples are presented in Sec D). Each QA pair includes the following components: Image, Category, English Question, English Answer, Translated Question, and Translated Answer. Contributors are required to follow the two steps outlined below.

- Step 1 (Question Relevance): Verify that the question is

phrased such that it cannot be answered without the use of a Vision-Language Model (VLM) analyzing the image. For example, a question like “Where is Aura Church located?” can be answered without any visual input. A more appropriate question would be, “Where is the church shown in the image located?"

- Step 2 (Translation Verification): If the translated ques-

tion or answer is incorrect, provide the correct version in the designated space and classify the error into one of the following four categories.

- our volunteers. We have one-fourth of the Female speakers and it is also shown that almost 48.3% of our volunteers fall in the age-bracket of 18-25. top-right. depicts the volunteer’s native language proficiency in years. bottom-left. depicts the volunteer’s Cultural Literacy proficiency in years. bottom-right. depicts the volunteer’s native language proficiency in years.

Translation Error Types

- • Semantic error: Translation hasn’t captured the semantic meaning
- • Cultural error: Correct but unusual words uncommon in local language context are used
- • Language error: Translation is provided using characters from a different alphabet or most commonly not translated and still given in English
- • Grammatical error: Grammar-related errors present

### C. Guidelines for Data Verification

For each country-language pair (e.g., Sri Lanka-Sinhala), QA pairs for general domains are created by translating the original English QA pairs from LLaVA-Bench (In-the-Wild) dataset [26]. For cultural domains, we curate the dataset by collecting images from online sources using targeted key-

### D. Verification Platform and UI Examples

Cultural Domain Descriptions

- • Food: Specific Dishes, Dining Dishes
- • Lifestyle: Daily Attire, Daily Life, Modern Lifestyle
- • Literature: Famous literature, authors and poets
- • Music: Traditional music and dance
- • Religion: Major religions and religious festivals
- • Customs: Social etiquette and traditional greetings
- • Festivals: Important cultural festivals and celebrations
- • Heritage: Popular historical heritage sites and iconic landmarks
- • Economy: Major economic industries
- • Media: Iconic Entertainment Figures and Popular TV Shows, Movies
- • Architecture: Traditional Art and Architecture
- • Sports: Famous traditional sports
- • Notable key figures: Famous historical leaders

[Figure 42]

[Figure 43]

- Figure A.4. User interface for translation verification hosted on Gradio, allowing contributors to classify incorrect entries and provide accurate translations.

### G. Instruction Prompt for QA Generation

Fig. A. A.5 illustrates the prompt messages used to guide GPT-4o in generating culturally relevant QA pairs. Each prompt includes an image, its manually added caption, and the search query used during image scraping. The model is instructed to generate two short QA pairs, two multiplechoice QA pairs, one true/false question, and one long QA pair. Detailed instructions ensure that the generated QAs are culturally grounded and specific to the image, avoiding answers based solely on common knowledge.

### E. Image Web Searching

We automated the process of web scraping for collecting cultural images. A custom URL, embedded with our encoded search query and licensing information, is passed to the Google Search Engine, and images are downloaded from the retrieved links. For each specific cultural domain within a given country-language pair, the search query is constructed as “{language} {cultural domain description} in {country}”. For privacy and copyright issues, our benchmark only includes images that are either in the public domain or licensed under the Creative Commons license.

### H. Instruction Prompt for QA Translations

After the curation of question-answer pairs using GPT-4o, we use the same open-source model to translate the English QA pairs into the native languages. We show the prompt that we used for this task in Fig. A. A.6.

### F. Blurring the PIDs

Recent research highlights the importance of blurring personally identifiable details (PIDs) in image datasets to mitigate privacy risks and reduce bias [49]. Thus, we filter the collected cultural images for cultural relevance, and faces are blurred using a face detection model, except for the ‘Media’ and ‘Notable Key Figures’ categories where we have images of public figures or celebrities. The blurred images are then manually reviewed to identify any discrepancies, such as blurring personally identifiable data (PIDs) and removing watermark images, and any remaining issues are addressed when using the PicdeFacer tool.

|prompt_messages = [ { "role":"system", "content": f "You're a helpful assistant that translates English text to target_lang”}, { "role":"user", “content": English_input} ]|
|---|

Figure A.6. Prompts used for translating cultural QAs using GPT4o.English_input refers to either the English question or answer to be translated into the target_lang, the desired target language.

|prompt_messages = [ { "role":"user", "content": [ { "type": “text", "text": search_query + "\n" + caption }, {“type”: “image_url", “image_url": image_url” } ] }, { "role":"system", "content": f """Here is an image and a caption I have on hand. I'd like you to generate two short questions and answers, two multiple choice questions and answers, one true/false question, and one long question and answer. Refer to the caption for the context/hint. Take into account the cultural diversity of cultural category. Follow the following rules while designing questions and answers:<br><br>1. The question must be answerable only by looking at the image.<br>2. Ensure that the questions are culturally relevant and specific to the image.<br>3. Provide answers that are concise, accurate, and directly related to the question.<br>4. You will also need to provide 1 correct option and 3 other incorrect options (distractors). For the distractors, choose options that are relevant, not obvious wrong answers.<br>5. The question must be answerable even without the multiple-choice. Example of the invalid question: (“What song is not performed by this musician” – not answerable if you don’t know the choices).<br>6. Make sure the questions are written fluently in English.<br>7. Be mindful of cultural sensitivities and avoid stereotyping or misrepresenting cultural aspects.<br>8. Ensure there are variations in your questions. Identity questions are fine, eg “What is this”, or “where is this”. But additionally. For example, multi-hop reasoning, counting, referencing, or questions that require local commonsense knowledge to be answered.<br>9. Just generate these in English.<br>10. For short questions and answers, don't keep it very short, include at least 2 sentences.<br>11. Make the questions distinct and unique from each other.<br><br><br>Give the answers in the following JSON format and make sure to only output a valid JSON, { "short_questions": [ { "question": <question>, "answer": <answer> }],<br><br>"multiple_choice_questions": [ { "question": <question>, "answer": <answer>, “options” <4 options> }], "true_false_question": { "question": <question>, "answer": <answer> }, "long_question": { "question": <question>, "answer": <answer> }""" }<br><br>]|
|---|

- Figure A.5. Prompts used for generation of cultural QA pairs. search_query refers to the query used to search the image online. It includes language, country and cultural category details. caption is a manually added textual description specific to the image. cultural category indicates the domain to which the image belongs, selected from the 13 cultural domains in our ALM-bench dataset.

### I. Instruction Prompt for LMM Answer Generation

We conduct a comprehensive study by evaluating various state-of-the-art Large Multimodal Models (LMMs), including both open-weight and closed-weight models, on our ALM-bench benchmark. We used different prompts for each question type as shown in Fig. A. A.7. We prompt all questions with English system instruction as highlighted by [18], suggesting that the use of prompts in English results in the best performance. Finally, we score each model’s generated answer with the human-annotated ground truth answers through a scoring system to assess each model’s performance on the benchmark.

### J. Prompts for GPT-Scoring

In addition to using GPT-4o as a judge in scoring the predicted answers, we also do scoring with multiple other LMMs to ensure a fair evaluation. Results are reported in Table A. A.2.

### K. Guidelines for Error Analysis on GPT-4o Output

Each annotator is provided with an Excel file specific to a given country-language pair, containing the following

|Model Name<br><br>|GPT-4o|Llama-3.1-8B-Instruct|
|---|---|---|
|GPT-4o Gemini-1.5 Pro GLM-4V-9B|90.16% 80.21% 71.35%|90.34% 80.65% 73.89%|

Table A.2. We evaluated decision-making questions (both True/False and multiple-choice) across a sample of 20 randomly selected languages using the LLama-3.1-8B-Instruct model. This assessment aims to ensure consistency in performance on T/F and MCQs when scored using GPT-4o as a judge.

columns: Image URL, Question, Ground Truth Answer, Predicted Answer (all in the native language), and GPT4o Score. The ‘Predicted Answer’ column records the response generated by GPT-4o when presented with the Question in the native language. The ‘GPT-4o Score’ column reflects the evaluation score assigned by GPT-4o, acting as a judge, based on the comparison between the Predicted Answer and the Ground Truth Answer.

Each annotator is required to complete two additional columns: (1) ‘Is the GPT Score Justified?’ with a binary response Yes/No, and (2) if the GPT-4o score is not justified, the ‘Reason for GPT-4o Failure’ column, where they select an appropriate reason from a predefined dropdown menu provided below.

messages = [{"role":"user", "content": [{“type": "image_url", "image_url": image_url }, {"type": "text", "text": prompt}] }]

messages = [{"role":"system", "content": " You are a helpful Assistant. Provide helpful response to the user's question."},{"role": "user", "content": prompt_eval}]

messages = [{"role":"user", "content": [{“type": "image_url", "image_url": image_url }, {"type": "text", "text": prompt}] }]

messages = [{"role":"system", "content": " You are a helpful Assistant. Provide helpful response to the user's question."},{"role": "user", "content": prompt_eval}]

True/False & Multiple Choice Questions prompt_eval = (

Multiple Choice Questions prompt = (

True/False & Multiple Choice Questions prompt_eval = (

Multiple Choice Questions prompt = (

f "Evaluate the following answer based on Accuracy:" f "Question: {question}" f "Ground Truth: {ground_truth}" f "Model Prediction: {predicted_answer}" f "Match the meaning of ground truth with model prediction, if it matches give a 10. Otherwise 0." f "Strictly return only the numeric score, without any additional commentary.” )

f "For the given the Multiple Choice Question Answer below, analyze the question and answer strictly from one of the options below. Strictly answer the choice only. No additional text." + question + choices )

f "Evaluate the following answer based on Accuracy:" f "Question: {question}" f "Ground Truth: {ground_truth}" f "Model Prediction: {predicted_answer}" f "Match the meaning of ground truth with model prediction, if it matches give a 10. Otherwise 0." f "Strictly return only the numeric score, without any additional commentary.” )

f "For the given the Multiple Choice Question Answer below, analyze the question and answer strictly from one of the options below. Strictly answer the choice only. No additional text." + question + choices )

True/False Questions prompt = ( question + f "The above question is a True/False question. Please provide the answer as one word in target_language" ) Long Questions prompt = ( question + f "Answer the question in detail in target_language language”

Long Questions prompt_eval = (

True/False Questions prompt = ( question + f "The above question is a True/False question. Please provide the answer as one word in target_language" ) Long Questions prompt = ( question + f "Answer the question in detail in target_language language”

f "Evaluate the following answer based on Consistency, Fluency, and Relevance based on the Ground Truth answer."

Long Questions prompt_eval = (

f "A high score example when predicted response match closely with ground truth and a low

score example when predicted response lacks knowledge, or is not related to the ground truth." f "Question: {question}" f "Ground Truth: {ground_truth}" f "Model Prediction: {predicted_answer}" f "Provide a single overall score out of 10 based on these three criteria." f "Strictly return only the numeric score, without any additional commentary.” )

f "Evaluate the following answer based on Consistency, Fluency, and Relevance based on the Ground Truth answer."

) Short Questions prompt = ( question + f "Please provide brief, clear responses in target_language language”

f "A high score example when predicted response match closely with ground truth and a low

score example when predicted response lacks knowledge, or is not related to the ground truth." f "Question: {question}" f "Ground Truth: {ground_truth}" f "Model Prediction: {predicted_answer}" f "Provide a single overall score out of 10 based on these three criteria." f "Strictly return only the numeric score, without any additional commentary.” )

Short Questions prompt_eval = (

)

f "Evaluate the following answer based on accuracy and correctness based on the Ground Truth answer."

) Short Questions prompt = ( question + f "Please provide brief, clear responses in target_language language”

f "A high score example when predicted response match closely with ground truth and a low

score example when predicted response lacks knowledge, or is not related to the ground truth." f "Question: {question}" f "Ground Truth: {ground_truth}" f "Model Prediction: {predicted_answer}" f "Provide a single overall score out of 10 based on these three criteria." f "Strictly return only the numeric score, without any additional commentary.” )

Short Questions prompt_eval = (

)

f "Evaluate the following answer based on accuracy and correctness based on the Ground Truth answer."

f "A high score example when predicted response match closely with ground truth and a low

Figure A.8. Prompts used to generate a score between 0 and 10, with GPT-4o acting as the evaluator to compare an LMM’s predicted answers against ground truth answers. The terms question, ground_truth, and predicted_answer refer to the cultural question, the ground truth answer generated by GPT-4o and verified by experts, and the model’s predicted answer, respectively.

score example when predicted response lacks knowledge, or is not related to the ground truth." f "Question: {question}" f "Ground Truth: {ground_truth}" f "Model Prediction: {predicted_answer}" f "Provide a single overall score out of 10 based on these three criteria." f "Strictly return only the numeric score, without any additional commentary.” )

Figure A.7. Prompts used to generate answers for cultural questions using multiple Large Multimodal Models evaluated in our study. Different prompts are designed for different question types. question refers to the cultural question associated with the given image, previously generated using GPT-4o. choices represents the four options provided in multiple choice question type, and target_language is the desired local language for the response.

### M. Country Specific Prompts

For the country-specific experiment as discussed in Section 4, Impact of location-aware information in prompts, Tab. 2 (main paper), we show the prompt used to conduct this experiment below. In this setting, we aim to assess how incorporating location-aware context affects visionlanguage model outputs, particularly in terms of potential cultural biases. To do this, we construct prompts using structured triplets comprising (language, country, cultural domain) — for example, (“Hindi,” “India,” “Food”) or (“Japanese,” “Japan,” “Religion”). This allows us to systematically evaluate whether models respond differently to culturally grounded prompts, and whether location-specific context influences model behavior or introduces/reduces bias.

Reason for GPT-4o Failure

- • Lack of Knowledge: The model fails to find an answer to the question.
- • Lack of Cultural Understanding: The model fails to understand the cultural aspect of the answer.
- • Language Error: Some words in the GPT-4o’s output are wrong for the language.
- • Reasoning Error: The reason it gave does not match that language.
- • Translation Error: A few native words were not properly translated.
- • Perceptual Error: The model fails to understand where a certain entity is (for eg, the top of the image, the bottom of the image, etc).

Location Aware Prompts

prompt = ques + “Provide brief, clear responses in ‘lang’ language. The image represents the ‘cultural category’ in ‘country’.”

### L. Qualitative examples with different question types

Next, we present some qualitative examples showing vari-

- ous question types for each image sample from our dataset and in Fig. A. A.9, A. A.10, A. A.11, A. A.11, A. A.13, and A. A.14.

|[Figure 44]<br><br>[Figure 45]<br><br>Language: Urdu Category: Festivals|Short Question Answer<br><br>|Ques: ؟ےہ ایگ ایاھکد وش اک مسق سک ںیم ریوصت (What type of show is depicted in the image?) Ans: وش اک ںویشیوم روا ںوڑوھگ (A horse and cattle show)<br><br>|
|---|
<br><br>|Ques: ؟ےہ ایگ ایاھکد رپ روط ںایامن روناج اس نوک ںیم ریوصت (What animal is prominently featured in the image?) Options:<br><br>A) اڑوھگHorse<br>B) ٹنواCamel<br>C) یھتاہElephant<br>D) Cattle یشیوم<br>|
|---|
<br><br>|Ques: ؟ںیہ ےہر آ رظن تلاآ ےک یقیسوم ےس نوک ںیم ریوصت (What kind of musical instruments are visible in the image?) Options:<br><br>A) Drums لوھڈ<br>B) Flute یرسناب<br>C) Sitar راتس<br>D) Tabla لابط<br>|
|---|
<br><br>|Ques: ےہر لچ ھتاس ےک ںوڑوھگ ےنپا نارود ےک بیرقت گول ؟ےہ ایگ ایاھکد ںیم ریوصت ہک اسیج ںیہ<br><br>(People are walking with their horses during the event shown in the image?) Ans: حیحص(TRUE)<br><br>|
|---|
<br><br>|Ques: ؟ےہ یہر وہ ںیم رہش سک بیرقت ہی<br><br>(In which city is this event taking place?)<br><br>Ans: روہلا (Lahore)<br><br>|
|---|
<br><br>True/False<br><br>|Ques: تیمہا یک زوش ےک ںویشیوم روا ےڑوھگ ںیم تفاقث یناتسکاپ یماقم اک نا ،تاعقاو ےلاو ےنوہ ًامومع ںیم نج ،ںیرک لایخ ہلدابت رپ ۔ںیہ ےتید غورف وک ےثرو یتفاقث حرط سک ہو روا رثا رپ ںویردارب (Discuss the significance of horse and cattle shows in Pakistani culture, including the types of events that typically take place, their impact on local communities, and how they promote cultural heritage.)<br><br>Ans:<br><br>وج ںیہ تابیرقت مہا زوش ےک ںویشیوم روا ےڑوھگ ںیم تفاقث یناتسکاپ رہاظ وک ںوتیحلاص یک ےنلاپ یشیوم روا تراہم یتعارز یک یٹنویمک نج ،ںیہ ےتوہ لماش ےلباقم فلتخم رپ روط ماع ںیم تابیرقت نا ۔ںیہ یترک زوش سنمرافرپ روا ،ڈیرپ یک ںوروناج ،ےلباقم ےک لسن نیرتہب ںیم ےنپا وک زرڈیرب روا ںوناسک زوش ہی ۔ںیہ لماش صقر اک ںوڑوھگ ےسیج یماقم ،ںیہ ےترک مہارف مراف ٹیلپ اک ےنرک شیپ وک ںوروناج نیرتہب ےک ےناھڑب لسن یک ںویشیوم روا ںیہ ےتید غورف وک رابوراک یعرز ٹہ ےس ولہپ یداصتقا ۔ںیہ ےترک یئازفا ہلصوح یک ےنانب رتہب وک رایعم یتیاور ںیم نج ،ںیہ ےتوہ روپرھب ےس ںورہظم یتفاقث راوہت ہی ،رک بیرق وک یٹنویمک ہی ۔ںیہ یتوہ لماش ںیٹواجس رادناش روا ،صقر ،یقیسوم روا ءاکرش ،ںیہ ےتید بیغرت یک ظفحت ےک ےثرو یتفاقث روا ںیہ ےتلا<br><br>۔ںیہ ےترک مہارف ہبرجت یمیلعت روا یحیرفت کیا ےئل ےک ںونود نیرظان|
|---|
<br><br>Multiple Choice Question Answers Long Question Answer<br><br>|
|---|---|
|[Figure 46]<br><br>[Figure 47]<br><br>Language: Chinese(Simplified) Category: Lifestyle|Short Question Answer<br><br>|Ques: 图片中描绘的是什么风格的婚纱? (What type of clothing is the person in this image wearing?) Ans: 那个人穿着现代汉服。 (The person is wearing modern Chinese Hanfu clothing.)<br><br>|
|---|
<br><br>|Ques: 这个人头上戴着什么? (What is this person wearing on her head?) Options:<br><br>A) 凤冠 FengGuan<br>B) 贝雷帽 Beret<br>C) 头纱 Veil<br>D) 头盔 Helmet<br>|
|---|
<br><br>|Ques: 图片中描绘的是哪种文化服饰？ (Which cultural attire is depicted in the image?) Options:<br><br>A) 现代汉服 Modern Chinese Hanfu<br>B) 和服 Kimono<br>C) 纱丽 Sari<br>D) 韩服Hanbok<br>|
|---|
<br><br>|Ques: 图片中的衣服是传统汉服吗？ (Is the clothing in the image traditional Chinese Hanfu?) Ans: 是的，是传统汉服。( True, it is traditional Chinese Hanfu. )<br><br>|
|---|
<br><br>|Ques: 这个人穿着的衣服主要是什么颜色？ (What colors are predominant in the person's outfit?) Ans: 衣服主要是红色和白色。 (The outfit predominantly features red and white colors.)<br><br>|
|---|
<br><br>True/False<br><br>|Ques: 描述图片中的服装上的具体元素和图案，并解释 现代汉服在当代中国社会中的文化意义<br><br>(Describe the specific elements and patterns seen on the outfit in the image and explain the cultural<br><br>significance of modern Hanfu attire in contemporary<br><br>Chinese society.) Ans:<br><br>图片中的服装有一件白色外衣，宽袖口和一件红色内衣，内衣包括 一条高腰裙子和一件有花卉图案的上衣。上衣上的花卉刺绣为整个 服装增添了精致的细节。现代汉服虽然起源于传统中国文化，但在 当代社会中得到了复兴。许多中国年轻人以穿戴汉服的方式来与自 己的文化遗产建立联系，表达文化自豪感。这种流行的复苏在各种 文化节日、社交媒体和汉服社区中也都有所体现，热爱汉服的人士 在这些平台上分享对这种复杂而意义深远的服饰的热情。<br><br>(The outfit in the image features a white outer garment with wide sleeves and a red inner garment composed of a high-waisted skirt and a floral top. The floral embroidery on the top adds elaborate detail to the ensemble. Modern Hanfu attire, while rooted in traditional Chinese culture, has seen a revival in contemporary society. Many young people in China embrace Hanfu as a way to<br><br>connect with their heritage and express cultural pride. This<br><br>resurgence in popularity is also evident in various cultural festivals, social media, and Hanfu communities, where individuals share their passion for this intricate and significant form of attire.)|
|---|
<br><br>Multiple Choice Question Answers Long Question Answer<br><br>|

###### Figure A.9. Qualitative examples of various question types in Urdu and Chinese Simplified Language.

|[Figure 48]<br><br>[Figure 49]<br><br>Language: Hindi Category: Heritage|Short Question Answer<br><br>|Ques: यह ऐतिहातिक स्थल कहााँ स्थस्थि है? (Where is this historic site located?)<br><br>Ans: नई तिल्ली, भारि। (New Delhi, India.)<br><br>|
|---|
<br><br>|Ques: तित्र में तिखाए गए तकले का नाम क्या है? (What is the name of the fort shown in the image?)<br><br>Options:<br><br>A) लाल किला Red Fort<br>B) अम्बर तकला Amber Fort<br>C) आगरा तकला Agra Fort<br>D) ग्वातलयर तकला Gwalior Fort<br>|
|---|
<br><br>|Ques: लाल तकला तकि शहर में स्थस्थि है? (In which city is the Red Fort located?) Options:<br><br>A) नई किल्ली New Delhi<br>B) जयपुर Jaipur<br>C) आगरा Agra<br>D) लखनऊ Lucknow<br>|
|---|
<br><br>|Ques: क्या तित्र में तिखाया गया तकला लाल बलुआ पत्थर िे बना है? (Is the fort in the image constructed using red sandstone?) Ans: िही (TRUE)<br><br>|
|---|
<br><br>|Ques: तित्र में तिखाए गए इि भवन का नाम क्या है? (What is the name of this building depicted in the image?)<br><br>Ans: लाल तकला। (Red Fort.)<br><br>|
|---|
<br><br>True/False<br><br>|Ques: लाल तकले के ऐतिहातिक महत्व पर ििाा करें, तजिमें मुग़ल युग के िौरान इिकी भूतमका, इिकी वास्तुकला तवशेषिाएाँ और आधुतनक भारि में इिका महत्व शातमल हैं।<br><br>(Discuss the historical significance of the Red Fort, including its role during the Mughal era, its architectural features, and its importance in modern India.)<br><br>Ans:<br><br>लाल तकला, जो नई तिल्ली में स्थस्थि है, का तनमााण मुग़ल िम्राट शाहजहााँ ने 17वीीं ििी के मध्य में करवाया था। यह लगभग 200 वषों िक मुग़ल िम्राटोीं का मुख्य आवाि रहा। तकला मुग़ल<br><br>वास्तुकला का प्रिीक है, तजिे इिकी तवशाल लाल बलुआ पत्थर<br><br>की िीवारोीं और जतटल तिजाइनोीं द्वारा पहिाना जािा है। मुग़ल युग के िौरान, यह एक राजनीतिक और िाींस्कृ तिक केंद्र के रूप में भारिीय इतिहाि में महत्वपूणा भूतमका तनभािा था। आधुतनक िमय में, लाल तकला उि स्थल के रूप में महत्वपूणा है जहााँ हर िाल स्विींत्रिा तिवि पर भारि के प्रधानमींत्री राष्ट्ीय ध्वज फहरािे हैं और भाषण िेिे हैं, जो भारि के इतिहाि और िींस्कृति में तकले की स्थायी तवरािि का प्रिीक है।|
|---|
<br><br>Multiple Choice Question Answers Long Question Answer<br><br>|
|---|---|
|[Figure 50]<br><br>[Figure 51]<br><br>Language: Sinhala Category: Sports|Short Question Answer<br><br>|Ques: ඡායාරූපයේ දක්වා ඇති සටන් කලාව<br><br>කුමක්ද?<br><br>(What martial art is being demonstrated in the image?) Ans: අංගමයපාර. (Angampora.)<br><br>|
|---|
<br><br>|Ques: ඡායාරූපයේ දක්වා ඇති ශ්‍රී ලංකායේ පුරාණ සටන් ක්‍රමය කුමක්ද? (What ancient martial art from Sri Lanka is shown in the image?)<br><br>Options:<br><br>A )අංගම්පොර Angampora<br>B) කලාරිපයට්තු Kalaripayattu<br>C) මුයි තායි Muay Thai<br>D) තයියකාන්යඩා Taekwondo<br>|
|---|
| |
|Ques: ඡායාරූපය අනුව යමම සටන්කලායේ භාවිතා වන අවි වර්ගය මින් කුමක්ද? (Which of these weapons is used in angampora according to the image?) Options:<br><br>A) ෂුරියක්න් Shuriken<br>B) නන්චක්කු Nunchaku<br>C) ය ෝ රිට Bo Staff<br>D) කඩු Swords<br>|
<br><br>|Ques: ඡායාරූපයේ යපන්නුම කර ඇති සමමතය සමරදායික ශ්‍රී ලංකායේ සටන් කලාවකි. (The image shows a form of traditional Sri Lankan martial arts.) Ans: ඔේ(TRUE)<br><br>|
|---|
<br><br>|Ques: සටන් කටයුුවලට අමතරව අංගමයපාර වල<br><br>රගුණ කරන ක්‍රියාව කුමක්ද? (What distinct activity apart from combat is integrated into angampora?) Ans: භාවනාව. (Meditation.)<br><br>|
|---|
<br><br>True/False<br><br>|Ques: ඡායාරූපයේ දක්වා ඇති සටන්කලායේ රධාන සංරචක යදක විස්තර කර සහ එම සංරචකවල වැදගත්කම පැහැදිලි කරන්න.<br><br>(Describe the two main components of angampora depicted in the image and explain their significance in the martial art.)<br><br>Ans:<br><br>අංගමයපාරයේ 'අංගම' හා 'ඉලලාංගම' යන රධාන සංරචක යදක අන්තර්ගත යේ. 'අංගම' යනු අත්වලින් සටන් කිරීමයි. 'ඉලලංගම' යනු කඩු සහ අයනකුත් ආයුධ රයයෝජනයට ගැනීමයි. යමම අංග ආත්මාරක්ශාව සඳහා වැදගත් වනවා යමන්ම, අතින් කරන සටන් තමායේ යේගවත් ව හා තම කුසලතාව වැඩිදියුණු කරනවා යමන්ම, ආයුධ වලින් කරන සටන්, සටන් කලාවට නව මානයක් එක් කරයි.|
|---|
<br><br>Multiple Choice Question Answers Long Question Answer<br><br>|

र

- Figure A.10. Qualitative examples of various question types in Hindi and Sinhala Language.

|[Figure 52]<br><br>[Figure 53]<br><br>Language: Italian Category: Music|Short Question Answer<br><br>|Ques: Che tipo di evento potrebbe rappresentare l'immagine? (What kind of event does the image likely depict?) Ans: L'immagine probabilmente rappresenta un evento di danza folcloristica italiana tradizionale.<br><br>(The image likely depicts a traditional Italian folk<br><br>dance event.)|
|---|
<br><br>|Ques: Quali abiti tradizionali indossano i ballerini nell'immagine? (What traditional clothing are the dancers wearing in the image?) Options:<br><br>A) Costumi tradizionali folcloristici italiani<br><br>Traditional Italian folk costumes<br><br>B) Gonne scozzesi Scottish kilts<br>C) Kimoni giapponesi Japanese kimonos<br>D) Sari indiani ndian sarees)<br>|
|---|
<br><br>|Ques: L'immagine mostra una performance di danza moderna?<br><br>Does the image show a modern dance performance?) Ans: Falso (False)<br><br>|
|---|
<br><br>|Ques: Cosa tengono in mano i ballerini nell'immagine? (What are the dancers in the image holding in their hands?) Ans: I ballerini nell'immagine tengono fazzoletti in mano. (The dancers in the image are holding handkerchiefs in their hands.)<br><br>|
|---|
<br><br>True/False<br><br>|Ques: Descrivi gli elementi dell'abbigliamento e della danza tradizionali italiani raffigurati nell'immagine, compresi i dettagli sull'abbigliamento, i movimenti della danza e il significato culturale di tali performance.<br><br>( Describe the elements of traditional Italian dress and dance depicted in the image, including details about the attire, the dance moves, and the cultural significance of such performances.)<br><br>Ans:<br><br>L'immagine mostra ballerini in costumi tradizionali folcloristici<br><br>italiani, che comprendono elementi come gilet ricamati, grembiuli<br><br>colorati e gonne con bordi di pizzo per le donne, e cappelli a tesa larga e bandane per gli uomini. I movimenti della danza includono passi e gesti sincronizzati, spesso con l'uso di mani o fazzoletti. Queste performance sono culturalmente significative perché rappresentano tradizioni regionali, celebrano eventi storici e mantengono viva l'eredità culturale trasmettendola di generazione in generazione. Tali eventi sono spesso accompagnati da musica dal vivo, suonata con strumenti tradizionali come la fisarmonica o il violino, che arricchiscono l'atmosfera festosa.۔|
|---|
<br><br>Multiple Choice Question Answers Long Question Answer<br><br>|Ques: Quale tipo di accompagnamento musicale è probabilmente presente in questo evento di danza tradizionale?<br><br>(What kind of musical accompaniment is likely<br><br>present in this traditional dance event?) Options:<br><br>A) Musica folcloristica dal vivo con strumenti<br><br>come la fisarmonica o il violino. (Live folk music from instruments like the accordion or violin)<br><br>B) Rock band (Rock band)<br>C) Ensemble di jazz (Jazz ensemble)<br>D) Orchestra sinfonica classica (Classical symphony orchestra)<br>|
|---|
|
|---|---|
|[Figure 54]<br><br>[Figure 55]<br><br>Language: German Category: Notable Key Figures|Short Question Answer<br><br>|Ques: Wer ist der Nazi-Führer auf diesem Bild? (Who is the Nazi leader depicted in this image?) Ans: Adolf Hitler (Adolf Hitler)<br><br>|
|---|
|Ques: In welchem Zeitraum regierte der Führer auf dem Bild Deutschland?<br><br>(During what period did the leader in the image rule<br><br>Germany?) Ans: Adolf Hitler regierte Deutschland von 1933 bis 1945.. (Adolf Hitler ruled Germany from 1933 to 1945.)<br><br>|
<br><br>|Ques: Welche politische Partei wurde von der Person auf dem Bild angeführt? (Which political party was led by the person in the image?) Options:<br><br>A) NSDAP (Nazi Party)<br>B) Kommunistische Partei (Communist Party)<br>C)Sozialdemokratische Partei (Social Democratic Party)<br>D)Christlich-Demokratische-Union(Christian Democratic Union)<br>|
|---|
<br><br>|Ques:Welcher Krieg wird mit der Führung der Person auf dem Bild in Verbindung gebracht?<br><br>(Which war is associated with the leadership of the person in the image?)<br><br>Options:<br><br>A) Zweiter Weltkrieg (World War II)<br>B) Erster Weltkrieg (World War I)<br>C) Kalter Krieg (Cold War)<br>D) Koreakrieg (Korean War)<br>|
|---|
<br><br>|Ques: Hat die Person auf dem Bild eine bedeutende Rolle im Zweiten Weltkrieg gespielt?<br><br>Did the person in the image play a significant role in World War II?) Ans: Wahr (TRUE)<br><br>|
|---|
<br><br>True/False<br><br>|Ques: Erörtern Sie die Auswirkungen und Konsequenzen der Herrschaft Adolf Hitlers auf Deutschland und die Welt, einschließlich der Ereignisse des Zweiten Weltkriegs und des Holocausts.<br><br>(Discuss the impact and consequences of Adolf Hitler's rule on Germany and the world, including the events of World War II and the Holocaust.)<br><br>Ans:<br><br>Die Herrschaft Adolf Hitlers hatte tiefgreifende und verheerende Auswirkungen auf sowohl Deutschland als auch die Welt. Als Führer des Nazi-Deutschlands führten Hitlers aggressive expansionistische Politiken direkt zum Ausbruch des Zweiten Weltkrieges im Jahr 1939. Sein Regime verübte den Holocaust, bei dem sechs Millionen Juden sowie Millionen anderer, die vom Nazi-Staat als unerwünscht angesehen wurden, systematisch in Konzentrationslagern ermordet wurden. Der Krieg selbst führte zum Tod von über 60 Millionen Menschen weltweit, zu massiver Zerstörung in ganz Europa und zu bedeutenden politischen, sozialen und wirtschaftlichen Folgen. Nach dem Krieg wurde Deutschland von den Alliierten besetzt und aufgeteilt, was schließlich zur Teilung in Ost- und Westdeutschland führte. Hitlers Handlungen veränderten die globale Geopolitik grundlegend, führten zur Gründung der Vereinten Nationen und bereiteten den Weg für den Kalten Krieg.|
|---|
<br><br>Multiple Choice Question Answers Long Question Answer<br><br>|

###### Figure A.11. Qualitative examples of various question types in Italian and German Language.

|[Figure 56]<br><br>[Figure 57]<br><br>Language: Spanish Category: Customs|Short Question Answer<br><br>|Ques: ¿Qué tipo de evento se muestra en la imagen? (What type of event is shown in the image?) Ans: La imagen muestra una corrida de toros. (The image shows a bullfight.)<br><br>|
|---|
<br><br>|Ques: ¿En qué ciudad se está llevando a cabo este<br><br>evento? (In which city is this event taking place?) Options:<br><br>A) Barcelona, España (Barcelona, Spain)<br>B) Madrid, España (Madrid, Spain)<br>C) Sevilla, España (Seville, Spain)<br>D) Bilbao, España (Bilbao, Spain)<br>|
|---|
<br><br>|Ques: ¿Qué papel desempeña la persona con el traje tradicional en este evento?<br><br>( What role does the person in the traditional costume play in<br><br>this event?)<br><br>Options:<br><br>A) Matador (Matador)<br>B) Picador (Picador)<br>C) Banderillero (Bandarillero)<br>D) Rejoneador (Rejoneador)<br>|
|---|
<br><br>|Ques: ¿El evento representado en la imagen es una<br><br>corrida es una corrida de toros?<br><br>(Is the event depicted in the image a traditional Spanish bullfight?) Ans: Verdadero (TRUE)<br><br>|
|---|
<br><br>|Ques: ¿Qué objeto está sosteniendo el matador en la imagen?<br><br>(What item is the matador holding in the image?) Ans: El matador está sosteniendo una capa. (The matador is holding a cape.)<br><br>|
|---|
<br><br>True/False<br><br>|Ques: Explica la importancia de las corridas de toros en la cultura española, especialmente su historia, los roles involucrados en una corrida, y cómo es vista en la sociedad española contemporánea.<br><br>(Explain the significance of bullfighting in Spanish culture, particularly its history, the roles involved in a bullfight, and how it is viewed in contemporary Spanish society.)<br><br>Ans:<br><br>Las corridas de toros, conocidas como 'corrida de toros', han sido una parte significativa de la cultura española durante siglos, con orígenes que se remontan a la antigua Roma. Involucra una serie de etapas, cada una<br><br>con roles específicos: el matador, que realiza la estocada<br><br>final; los picadores, que debilitan al toro; y los banderilleros, que colocan palos coloridos (banderillas) en el toro. A pesar de sus profundas raíces históricas y su fuerte presencia en las tradiciones y festivales españoles, las corridas de toros se han convertido en un tema controvertido en la España moderna. Algunas regiones, como Cataluña, las han prohibido, reflejando un cambio en los valores sociales donde los derechos de los animales y las consideraciones éticas están tomando precedencia sobre las prácticas tradicionales.|
|---|
<br><br>Multiple Choice Question Answers Long Question Answer<br><br>|
|---|---|
|[Figure 58]<br><br>[Figure 59]<br><br>Language: Emirati Arabic Category:Architecture|Short Question Answer<br><br>|Ques: ؟ةروصلا يف دوجوم وش<br><br>(What is depicted in the image?) Ans: افارتلااك وغايتناس هممص يللا تاراملإا حانج نيبت ةروصلا. (The image shows Santiago Calatrava’s UAE Pavilion.)<br><br>|
|---|
<br><br>|Ques: ؟ةروصلا يف تاراملإا حانج ممص يلإ يرامعملا وه نم (Which architect designed the UAE Pavilion shown in the image?) Options:<br><br>A) Santiago Calatrava افارتلااك وغايتناس<br>B) Frank Gehry يريج كنارف<br>C) Zaha Hadid ديدح اهز<br>D) Norman Foster رتسوف نامرون<br>|
|---|
<br><br>|Ques: ؟تاراملإاحانج ميمصتل عقوتملا ماهللإا وهوش (What inspiration is most likely behind the design of the UAE Pavilion?) Options:<br><br>A) Falcon wings رقصلا ةحنجأ<br>B) Waves جاوملأا<br>C) Mountains لابجلا<br>D) Palm trees ليخنلا راجشأ<br>|
|---|
<br><br>|Ques: ؟ةيديلقت ةيتارامإ رصانع تاراملإا حانج ميمصت سكعي له (Does the design of the UAE Pavilion reflect traditional Emirati elements?) Ans: سكعي ،حص(TRUE, it does)<br><br>|
|---|
<br><br>|Ques: ؟ىنبملا يف ةزرابلا ةيرامعملا ةزيملا وش (What unique architectural feature is prominent in the building?) Ans: ةحنجلأا هبشت لكايهب زيمتي ىنبملا. (The building prominently features wing-like structures.)<br><br>|
|---|
<br><br>True/False<br><br>|Ques: يف تاراملإا حانجل افارتلااك وغايتناس ميمصت سكعي فيك نع ملكت وبسكإ2020 يفاقثلا تاراملإا ةلود ثارتو تاعلطت (Discuss how Santiago Calatrava’s architectural design of the UAE Pavilion at Expo 2020 reflects the aspirations and cultural heritage of the United Arab Emirates.)<br><br>Ans: ةزجعم ربتعي2020 وبسكإ يف تاراملإا حانجل افارتلااك وغايتناس ميمصت هبشت يللا لكايهلا. يفاقثلا تاراملإا ثارتو تاحومط سكعت ةثيدح ةيرامعم ةينطوو ةيفاقث ةيزمر اهل يللا ،ناريطلا يف روقصلا ةكرح نم ةاحوتسم ةحنجلأا ميمصتلا. ةلودلل ةحومطلا حورلاو ثارتلاو ةوقلا لثمتو ،تاراملإا يف ةريبك ةمادتسلااب اهمازتلا نيبيو ،ةيلبقتسملا تاراملإا ةيؤر نع ربعي حانجلل ركتبملا نيب جيزملاه. ةقيرعلا ةيفاقثلا اهديلاقت ىلع اهظافح عم روطتلاو ايجولونكتلاو ىلع تاراملإل ةيكيمانيدلا ةيوهلا زربي ةيفاقثلا ةيزمرلاو يرصعلا ميمصتلا<br><br>.ملاعلا ىوتسم|
|---|
<br><br>Multiple Choice Question Answers Long Question Answer<br><br>|

ُ

- Figure A.12. Qualitative examples of various question types in Spanish and Emirati Arabic Language.

|[Figure 60]<br><br>[Figure 61]<br><br>Language: Saudi Arabic Category: Media|Short Question Answer<br><br>|Ques: ؟ةروصلا يف ضورعملا ينويزفلتلا لسلسملا ناونع وه ام (What is the title of the TV series shown in the image?) Ans: رفس ةكس (Sekket Safar)<br><br>|
|---|
<br><br>|Ques: ؟رفس ةكس ينويزفلتلا لسلسملا يف نيلثمملا دحأ وه نم (Who is one of the actors in the TV series Sekket<br><br>Safar?)<br><br>Options:<br><br>A) Mohammed Al-Shehri يرهشلا دمحم<br>B) Nasser Al Qasabi يبصقلا رصان<br>C) Abdulmohsen Al-Nemr رمنلا نسحملادبع<br>D) Rashed Essa ىسيع دشار<br>|
|---|
<br><br>|Ques: ؟رفس ةكس ينويزفلتلا لسلسملا عون وه ام (What genre is the TV series Sekket Safar?)<br><br>Options:<br><br>A) Comedy ايديموك<br>B) Drama امارد<br>C) Horror بعر<br>D) Thriller ةراثإ<br>|
|---|
<br><br>|Ques: ماع يف ضرع ينويزفلت لسلسم وه رفس ةكس له2022؟ (Is Sekket Safar a TV series that aired in 2022?) Ans: حيحص(TRUE)<br><br>|
|---|
<br><br>|Ques: ؟ةروصلا يف ينويزفلتلا لسلسملا تجتنأ ةينويزفلت ةكبش يأ (Which TV network produced the TV series in the image?) Ans: MBC (MBC)<br><br>|
|---|
<br><br>True/False<br><br>|Ques: ةيبرعلا ةكلمملا يف رفس ةكس لسلسمل ةيهيفرتلاو ةيفاقثلا ةيمهلأا شقان<br><br>لبق نم هجاتنإ ىلإ رظنلاب ،ةيدوعسلاMBC ةايحلا طامنلأ هريوصتو ةيدوعسلا.<br><br>(Discuss the cultural and entertainment significance of the TV series Sekket Safar in Saudi Arabia, considering its production by<br><br>MBC and the portrayal of Saudi lifestyles.)<br><br>Ans: ةيمهأ لمحي ، 2022ماع يفMBC هتجتنأ يديموك لسلسم وه رفس ةكس بناوج لسلسملا روصي. ةيدوعسلا ةيبرعلا ةكلمملا يف ةريبك ةيهيفرتو ةيفاقث ةربعمو ةفيرط ةرظن نيدهاشملل ًامدقم ،ةيدوعسلا ةفاقثلاو ةايحلا نم ةعونتم ،لسلسملا زربي. ةيمويلا براجتلاو ةيعامتجلاا فارعلأاو ةيلحملا تاداعلا نع ،ورمع وبأ حلاصو زيزع دعسو يرهشلا دمحم لثم نولثمم هيف كراشي يذلا لثم ةدئار ةكبش لبق نم لسلسملا جاتنإ. ةقطنملا يف هيفرتلا ةعانص ومن سكعي امم ،يدوعسلا روهمجلا عم لعافتت ةيلاع ىوتحم ةدوج نمضيMBC ءارثإو عيونت يف رفس ةكس مهسي ،اذل. مهب اطبترم اهيفرت مهل رفويو مهصصق<br><br>.يدوعسلا يملاعلإا دهشملا|
|---|
<br><br>Multiple Choice Question Answers Long Question Answer<br><br>|
|---|---|
|[Figure 62]<br><br>[Figure 63]<br><br>Language:Afrikaans Category: Economy|Short Question Answer<br><br>|Ques: Watter stad word as hoofstad op die beeld vertoon? (What is the main city shown in the image?) Ans: Johannesburg (Johannesburg)<br><br>|
|---|
<br><br>|Ques: Watter stad in Suid-Afrika word op die beeld uitgebeeld? (What South African city is depicted in the image?) Options:<br><br>A) Johannesburg (Johannesburg)<br>B) Kaapstad (Cape Town)<br>C) Durban (Durban)<br>D) Pretoria (Pretoria)<br>|
|---|
<br><br>|Ques: In watter bedryf is Johannesburg veral bekend? (Which industry is Johannesburg widely known for?) Options:<br><br>A) Mynbou (Mining)<br>B) Landbou (Agriculture)<br>C) Toerisme (Tourism)<br>D) Mode (Fashion)<br>|
|---|
<br><br>|Ques: Die beeld toon die skyline van Johannesburg. (The image represents the skyline of Johannesburg.) Ans: Waar (TRUE)<br><br>|
|---|
<br><br>|Ques: Watter toringagtige struktuur is prominent op die beeld? (Which structure is prominently featured with a tower-like design?) Ans: Die Hillbrow Toring (The Hillbrow Tower)<br><br>|
|---|
<br><br>True/False<br><br>|Ques: Verduidelik die rol van Johannesburg in die Suid-<br><br>Afrikaanse ekonomie, met spesifieke klem op sy historiese<br><br>agtergrond in mynbou en ander belangrike ekonomiese sektore wat tot sy welvaart bydra. (Explain the significance of Johannesburg in South Africa's economy, highlighting its historical background in mining and other major economic industries that contribute to its prosperity.)<br><br>Ans:<br><br>Johannesburg, dikwels beskryf as die ekonomiese spilpunt van Suid-Afrika, het vinnig gegroei na die ontdekking van goud in die laat 19de eeu en het steeds 'n noemenswaardige ekonomiese rol. Behalwe mynbou, het Johannesburg oor tyd gediversifiseer na ander sektore,<br><br>insluitend finansies, vervaardiging, telekommunikasie en<br><br>kleinhandel. Die stad huisves die Johannesburgse Effektebeurs (JSE), wat sy finansiële markposisie versterk. Met sy skyline as simbool van 'n moderne en dinamiese ekonomie, trek Johannesburg beleggers plaaslik en internasionaal, en speel dit 'n sleutelrol in Suid-Afrika se ekonomiese landskap.|
|---|
<br><br>Multiple Choice Question Answers Long Question Answer<br><br>|

ّ

ّ

ّ

ّ

ّ

ُ ّ

ّ

- Figure A.13. Qualitative examples of various question types in Saudi Arabic and Afrikaans Language.

###### South Africa-Afrikaans

###### Italy- Italian

###### France-French

###### China-Chinese Simplified

Category: Music

Category: Food

Category: Heritage

Category: Media

[Figure 64]

[Figure 65]

[Figure 66]

[Figure 67]

Multiple Choice Question Answers

Short Question Answers

Short Question Answers True False

(哪个电视频道播出 “中国好声音”？

Waarop speel die Hausamusikante in hierdie beeld?

Comment s'appelle le monument dans l’image?

###### Descrivi gli ingredienti tipici e il metodo di preparazione della Zuppa di Minestrone raffigurata nell'immagine.

(Which television network broadcasts 'The Voice of China'?)

(What is the monument in the image called?)

(What are Hausa musicians in this image performing on?)

Hulle speel op verskeie tradisionele instrumente, insluitend dié wat van dierehorings gemaak is.

- a) cathédrale Notre-Dame de Paris ( Cathedral Notre-Dame of Paris)
- b) Tour Eiffel (Eiffel Tower)
- c) Arc de Triomphe), (Arc de Triomphe)
- d) Pyramide du Louvre(Louvre Pyramid))

(The bread shown in the image is a

type of Italian bread.)

浙江电视台

Vero (True)

(They are performing on various traditional instruments, including those made from animal horns.)

(Zhejiang Television)

###### India-Hindi

###### Germany-German

###### Bulgaria-Bulgarian

###### Russia-Russian

Category: Religion

Category: Literature

Category: Architecture

Category: Festivals

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Short Question Answers Multiple Choice Question Answers Short Question Answers

True False

Каква традиционна архитектурна

###### इस चित्र में कौन सा त्यौहार मनाया जा रहा है?

Что изображено на картинке?

Befindet sich die Skulptur im Bild in Dresden?

особеност може да се види в сградата отляво?

(What is depicted in the image?)

(What festival is being celebrated in this image?)

(What traditional architectural feature can be observed in the building on the left?)

(Is the sculpture in the image located in Dresden?.)

На изображении изображено празднование Масленицы с центральной фигурой, олицетворяющей Масленицу в традиционной одежде.

- a) Сградата отляво се отличава с богато украсени дървени резби и цветни фрески. (The building on the left features ornate wooden carvings and colorful frescoes.
- b) минималистични бетонни конструкции (Minimalist concrete structures)

मेवाड़ महोत्सव.

Falsch (False)

(The Mewar Festival.)

(The image images a celebration of Maslenitsa with a central figure representing Maslenitsa in traditional clothing.)

Sri Lanka-Sinhala Category: Customs

###### Kazakhstan-Kazakh

###### Bangladesh-Bengali

###### India-Marathi

Category: Economy

Sports

Category: Notable key figures

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

Multiple Choice Question Answers

True False

Short Question Answers

Short Question Answers

මෙෙ රූපමේ සිටින ෙධ්‍යෙ යුවලමේ වම් පසින් සාම්රදායික ඇඳුමින් සැරසී සිටින පුද්ගලයන් ම ාම ෝ විට නිමයෝජනය කරන්මන් කවුරුන්ද?

Бәйтерек монументі Қазақстанда орналасқан ба?

বাচ্চারা ছববতে বি খেলা খেলতছ?

चिवाजीिा पुतळा कोठे आहे?

(What game are the kids playing in the image?)

(Where is this statue of Shivaji located?)

(Is the Baiterek monument located in Kazakhstan?

(Who are the people in traditional attire to the left of the central couple in this image most likely representing?)

हा चिवाजीिा पुतळा भारतातील महाराष्ट्रातील रायगड चकल्ल्यावर आहे

িাাঁটার উপর ঝাাঁবপত়ে পডা। (Jumping over thorns.)

- a) සාම්රදායික ම ර වාදකයින් ස නර්තන ශිල්පීන් (Traditional drummers and dancers)
- b) විවාහ අමුත්තන් (Wedding guests)
- c) යුවලගේ පවුගේ සාමාජිකයන්(Family members of the couple)

ШЫН (True)

(This statue of Shivaji is located at Raigarh Fort in Maharashtra, India.)

Figure A.14. Some more qualitative examples of various question types from our benchmark.

[Figure 76]

[Figure 77]

[Figure 78]

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

###### Figure A.15. Word clouds depicting prominent concepts from 19 categories in our ALM-bench. For intriguing results, we plot and demonstrate the results on English samples of the plot representing both the cultural and generic elements for the entire 100 languages.

###### S#No Language Country Script Family Subgrouping Specification

- 1 Afrikaans South Africa Latin Indo-European Germanic High
- 2 Albanian Albania Latin Indo-European Albanian High
- 3 Amharic Ethiopia Ge’ez Afro-Asiatic Semitic Low
- 4 Armenian Armenia Armenian Indo-European Armenic Low
- 5 Assamese India Bengali Indo-European Indo-Aryan Low
- 6 Azerbaijani Azerbaijan Latin Turkic Common Turkic Low
- 7 Basque Spain Latin Isolate – High
- 8 Belarusian Belarus Cyrillic Indo-European Balto-Slavic Low
- 9 Bengali Bangladesh Bengali Indo-European Indo-Aryan High
- 10 Bhojpuri India Devanagari Indo-European Indo-Aryan Low
- 11 Bosnian Bosnia Latin Indo-European Balto-Slavic High
- 12 Bulgarian Bulgaria Cyrillic Indo-European Balto-Slavic High
- 13 Catalan Spain Latin Indo-European Italic High
- 14 Cebuano Philippines Latin Austronesian Malayo-Polynesian Low
- 15 Chinese Simplified China Chinese Sino-Tibetan Sinitic High
- 16 Chinese Traditional Hong Kong Chinese Sino-Tibetan Sinitic High
- 17 Croatian Croatia Latin Indo-European Balto-Slavic High
- 18 Czech Czech Republic Latin Indo-European Balto-Slavic High
- 19 Danish Denmark Latin Indo-European Germanic High
- 20 Dutch Netherlands Latin Indo-European Germanic High
- 21 Egyptian Arabic Egypt Arabic Afro-Asiatic Semitic Low
- 22 Emirati Arabic United Arab Emirates Arabic Afro-Asiatic Semitic High
- 23 English United Kingdom Latin Indo-European Germanic High
- 24 Estonian Estonia Latin Uralic Finnic High
- 25 Filipino Philippines Latin Austronesian Malayo-Polynesian Low
- 26 Finish Finland Latin Uralic Finnic High
- 27 French France Latin Indo-European Italic High
- 28 Galician Spain Latin Indo-European Italic Low
- 29 Georgian Georgia Georgian Kartvelian Kartvelian Low
- 30 German Germany Latin Indo-European Germanic High
- 31 Greek Greece Greek Indo-European Graeco-Phrygian High
- 32 Gujarati India Gujarati Indo-European Indo-Aryan Low
- 33 Hausa Nigeria Latin Afro-Asiatic Chadic Low
- 34 Hawaiian United States Latin Austronesian Malayo-Polynesian Low
- 35 Hebrew Israel Hebrew Afro-Asiatic Semitic High
- 36 Hindi India Devanagari Indo-European Indo-Aryan High
- 37 Hungarian Hungary Latin Uralic – High
- 38 Icelandic Iceland Latin Indo-European Germanic High
- 39 Igbo Nigeria Latin Atlantic-Congo Benue-Congo Low
- 40 Indonesian Indonesia Latin Austronesian Malayo-Polynesian High
- 41 Irish Ireland Latin Indo-European Celtic Low
- 42 Italian Italy Latin Indo-European Italic High
- 43 Japanese Japan Kanji/Kana Japonic Japanese Ryukyuan High
- 44 Javanese Indonesia Latin Austronesian Malayo-Polynesian Low
- 45 Kannada India Kannada Dravidian South Dravidian Low
- 46 Kazakh Kazakhstan Cyrillic Turkic Common Turkic High
- 47 Kinyarwanda Rwanda Latin Atlantic-Congo Benue-Congo Low
- 48 Korean South Korea Hangul Koreanic Korean High
- 49 Kurdish Turkey Arabic Indo-European Iranian Low
- 50 Kyrgyz Kyrgyzstan Cyrillic Turkic Common Turkic Low

- S#No Language Country Script Family Subgrouping Specification
- 51 Lao Thailand Lao Tai-Kadai Kra-Dai Low
- 52 Latin Vatican City Latin Indo-European Italic Low
- 53 Latvian Latvia Latin Indo-European Balto-Slavic High
- 54 Lithuanian Lithuania Latin Indo-European Balto-Slavic High
- 55 Luxembourgish Luxembourg Latin Indo-European Germanic Low
- 56 Macedonian North Macedonia Cyrillic Indo-European Balto-Slavic High
- 57 Malagasy Madagascar Latin Austronesian Malayo-Polynesian Low
- 58 Malay Malaysia Latin Austronesian Malayo-Polynesian High
- 59 Malayalam India Malayalam Dravidian South Dravidian Low
- 60 Maltese Malta Latin Afro-Asiatic Semitic High
- 61 Marathi India Devanagari Indo-European Indo-Aryan Low
- 62 Mongolian Mongolia Cyrillic Mongolic-Khitan Mongolic Low
- 63 Myanmar (Burmese) Myanmar Myanmar Sino-Tibetan Burmo-Qiangic Low
- 64 Nepali Nepal Devanagari Indo-European Indo-Aryan Low
- 65 Norwegian Norway Latin Indo-European Germanic Low
- 66 Odia (Oriya) India Oriya Indo-European Indo-Aryan Low
- 67 Pashto Pakistan Arabic Indo-European Iranian Low
- 68 Persian Iran Arabic Indo-European Iranian High
- 69 Polish Poland Latin Indo-European Balto-Slavic High
- 70 Portuguese Portugal Latin Indo-European Italic High
- 71 Punjabi Pakistan Gurmukhi Indo-European Indo-Aryan Low
- 72 Romanian Romania Latin Indo-European Italic High
- 73 Russian Russia Cyrillic Indo-European Balto-Slavic High
- 74 Sanskrit India Devanagari Indo-European Indo-Aryan Low
- 75 Saudi Arabic Saudi Arabia Arabic Afro-Asiatic Semitic High
- 76 Scots Gaelic Scotland Latin Indo-European Celtic Low
- 77 Serbian Serbia Cyrillic Indo-European Balto-Slavic Low
- 78 Shona Zimbabwe Latin Atlantic-Congo Benue-Congo Low
- 79 Sindhi Pakistan Arabic Indo-European Indo-Aryan Low
- 80 Sinhala Sri Lanka Sinhala Indo-European Indo-Aryan Low
- 81 Slovak Slovakia Latin Indo-European Balto-Slavic High
- 82 Slovenian Slovenia Latin Indo-European Balto-Slavic High
- 83 Somali Somalia Latin Afro-Asiatic Cushitic Low
- 84 Spanish Spain Latin Indo-European Italic High
- 85 Sundanese Indonesia Latin Austronesian Malayo-Polynesian Low
- 86 Swahili Tanzania Latin Atlantic-Congo Benue-Congo High
- 87 Swedish Sweden Latin Indo-European Germanic High
- 88 Tajik Tajikistan Cyrillic Indo-European Iranian Low
- 89 Tamil India Tamil Dravidian South Dravidian Low
- 90 Telugu India Telugu Dravidian South Dravidian Low
- 91 Thai Thailand Thai Tai-Kadai Kam-Tai High
- 92 Turkish Turkey Latin Turkic Common Turkic High
- 93 Ukrainian Ukraine Cyrillic Indo-European Balto-Slavic High
- 94 Urdu Pakistan Arabic Indo-European Indo-Aryan Low
- 95 Uyghur China Arabic Turkic Common Turkic Low
- 96 Uzbek Uzbekistan Latin Turkic Common Turkic High
- 97 Vietnamese Vietnam Latin Austroasiatic Vietic High
- 98 Welsh United Kingdom Latin Indo-European Celtic Low
- 99 Yiddish Israel Hebrew Indo-European Germanic Low
- 100 Yoruba Nigeria Latin Atlantic-Congo Benue-Congo Low

Table A.3. A comprehensive list of 100 languages, their associated country, language scripts, families, subgrouping, and the resource specification.

|No.|Author Name|Affiliation<br><br>|Email|
|---|---|---|---|
|1<br>2<br>3<br>4<br>5<br>6<br>7<br>8<br>9<br>10<br>11<br>12<br>13<br>14<br>15<br>16<br>17<br>18<br>19<br>20<br>21<br>22<br>23<br>24<br>25<br>26<br>27<br>28<br>29<br>30<br>31<br>32<br>33<br>34<br>35<br>36<br>37<br>38<br>39<br>40<br>41<br>42<br>43<br>44<br>45<br>46<br>47<br>48<br>49<br>50<br>51<br>52<br>|Henok Biadglign Ademtew Yahya Hmaiti Amandeep Kumar Kartik Kuckreja Mykola Maslych Wafa Al Ghallabi Mihail Mihaylov Chao Qin Abdelrahman M Shaker Mike Zhang Mahardika Krisna Ihsani Amiel Esplana Monil Gokani Shachar Mirkin Harsh Singh Ashay Srivastava Endre Hamerlik Fathinah Asma Izzati Fadillah Adamsyah Maani Sebastian Cavada Jenny Chim Rohit Gupta Sanjay Manjunath Kamila Zhumakhanova Feno Heriniaina Rabevohitra Azril Amirudin Muhammad Ridzuan Daniya Kareem Ketan More Kunyang Li Pramesh Shakya Muhammad Saad Amirpouya Ghasemaghaei Amirbek Djanibekov Dilshod Azizov Branislava Jankovic Naman Bhatia Alvaro Cabrera Johan Obando-Ceron Olympiah Otieno Fabian Farestam Muztoba Rabbani Sanoojan Baliah Santosh Sanjeev Abduragim Shtanchaev Maheen Fatima Thao Nguyen Amrin Kareem Toluwani Aremu Nathan Xavier Amit Bhatkal Hawau Olamide Toyin|Ethiopian Artificial Intelligence Institute University of Central Florida Johns Hopkins University MBZUAI University of Central Florida MBZUAI MBZUAI MBZUAI MBZUAI Aalborg University MBZUAI University of the West of England, Bristol IIT, Hyderabad Alpinference MBZUAI University of Maryland, College Park HUN-REN Institute for CS and Control MBZUAI MBZUAI MBZUAI Queen Mary University of London, UK University of Central Florida MBZUAI MBZUAI Chongqing University University of the People MBZUAI MBZUAI MBZUAI University of Central Florida University of Central Florida MBZUAI University of Central Florida MBZUAI MBZUAI MBZUAI Indian Institute of Technology, Hyderabad MBZUAI University of Montreal, Mila JKUAT ETHZ - ETH Zurich University of California, Merced MBZUAI MBZUAI MBZUAI Air University MBZUAI MBZUAI MBZUAI Universidade Federal de Minas Gerais MBZUAI MBZUAI|henokb2124@gmail.com yohan.hmaiti@ucf.edu kumar.amandeep015@gmail.com kartik.kuckreja@mbzuai.ac.ae mykola.maslych@ucf.edu wafa.alghallabi@mbzuai.ac.ae mihail.mihaylov@mbzuai.ac.ae 1746625542@qq.com abdelrahman.youssief@mbzuai.ac.ae jjz@cs.aau.dk mahardika.ihsani@mbzuai.ac.ae amiel2.esplana@live.uwe.ac.uk monilgokani08@gmail.com shacharmirkin@gmail.com harsh.singh@mbzuai.ac.ae ashays06@umd.edu hamerlik@sztaki.hu fathinah.izzati@mbzuai.ac.ae fadillah.maani@mbzuai.ac.ae sebastian.cavada@mbzuai.ac.ae jennychim@gmail.com rohitgupta.hpf@gmail.com sanjay.manjunath@mbzuai.ac.ae kamila.zhumakhanova@mbzuai.ac.ae fenoheriniaina@gmail.com azrilamirudin@my.uopeople.edu 20020084@mbzuai.ac.ae daniya.kareem@mbzuai.ac.ae ketan.more@mbzuai.ac.ae kunyang.li@ucf.edu pramesh.shakya@ucf.edu muhammad.saad@mbzuai.ac.ae aghaei.ap@ucf.edu amirbek.djanibekov@mbzuai.ac.ae dilshod.azizov@mbzuai.ac.ae branislava.jankovic@mbzuai.ac.ae naman.219311175@muj.manipal.edu alvaro.berobide@mbzuai.ac.ae jobando0730@gmail.com anikaolympiah@gmail.com ffarestam@student.ethz.ch mrabbani@ucmerced.edu sanoojan.baliah@mbzuai.ac.ae santosh.sanjeev@mbzuai.ac.ae abduragim.shtanchaev@mbzuai.ac.ae 231659@students.au.edu.pk thao.nguyen@mbzuai.ac.ae amrin.kareem@mbzuai.ac.ae toluwani.aremu@mbzuai.ac.ae nathanxavier@ufmg.br amitbhatkal12@gmail.com hawau.toyin@mbzuai.ac.ae|

###### Table A.4. List of affiliations for all the volunteer co-authors who contributed to construct ALM-Bench.

