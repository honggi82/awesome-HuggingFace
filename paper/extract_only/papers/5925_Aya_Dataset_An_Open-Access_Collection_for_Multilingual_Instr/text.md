## arXiv:2402.06619v1[cs.CL]9Feb2024

[Figure 1]

# Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning

Shivalika Singh♦1, Freddie Vargus♦1, Daniel D’souza♦1, Börje F. Karlsson♦2, Abinaya Mahendiran♦1, Wei-Yin Ko♦3, Herumb Shandilya♦1, Jay Patel4, Deividas Mataciunas1, Laura O’Mahony5, Mike Zhang6, Ramith Hettiarachchi7, Joseph Wilson8, Marina Machado3, Luisa Souza Moura3, Dominik Krzemiński1, Hakimeh Fadaei1, Irem Ergün3, Ifeoma Okoh1, Aisha Alaagib1, Oshan Mudannayake1, Zaid Alyafeai9, Vu Minh Chien1, Sebastian Ruder3, Surya Guthikonda1, Emad A. Alghamdi10, Sebastian Gehrmann11, Niklas Muennighoff1, Max Bartolo3, Julia Kreutzer12, Ahmet Üstün12, Marzieh Fadaee12, and Sara Hooker12

1Cohere For AI Community, 2Beijing Academy of Artificial Intelligence, 3Cohere, 4Binghamton University, 5University of Limerick, 6IT University of Copenhagen, 7MIT, 8University of Toronto, 9King Fahd University of Petroleum and Minerals, 10King Abdulaziz University, ASAS.AI, 11Bloomberg LP, 12Cohere For AI

Corresponding authors: Shivalika Singh <shivalikasingh95@gmail.com>, Marzieh Fadaee <marzieh@cohere.com>, Sara Hooker <sarahooker@cohere.com>

###### Abstract

Datasets are foundational to many breakthroughs in modern artificial intelligence. Many recent achievements in the space of natural language processing (NLP) can be attributed to the finetuning of pre-trained models on a diverse set of tasks that enables a large language model (LLM) to respond to instructions. Instruction fine-tuning (IFT) requires specifically constructed and annotated datasets. However, existing datasets are almost all in the English language. In this work, our primary goal is to bridge the language gap by building a human-curated instruction-following dataset spanning 65 languages. We worked with fluent speakers of languages from around the world to collect natural instances of instructions and completions. Furthermore, we create the most extensive multilingual collection to date, comprising 513 million instances through templating and translating existing datasets across 114 languages. In total, we contribute four key resources: we develop and open-source the Aya Annotation Platform, the Aya Dataset, the Aya Collection, and the Aya Evaluation Suite. The Aya initiative also serves as a valuable case study in participatory research, involving collaborators from 119 countries. We see this as a valuable framework for future research collaborations that aim to bridge gaps in resources.

##### 1 Introduction

Datasets are static representations of the world, far from the rich, ever-evolving environment we navigate as humans. Yet, these frozen snapshots in time are the foundation upon which progress in AI has been built. Many recent breakthroughs in language modeling can be attributed to fine-tuning

♦First authors.

Released as a preprint on February 12, 2024 1

### 📚

[Figure 2]

[Figure 3]

[Figure 4]

Prompt 65 Completion

Text Classiﬁcation

###### "ඕ  ෙන   ෙන ය රඟාලා" යන  තය ඇ ළ   ංහල   පටය  ම  ද?

"ඕ  ෙන   ෙන ය රඟාලා" යන  තය ඇ ළ  ව ෙ  "ෙ ඛාව" න   ංහල   පටෙ  ය. ෙමම  තය   ර ෙ නාර න මහතා ගායනා කරන ලද  ය .

Prompt

Prompt

What is the corresponding translation in {{target_lang}} of the following sentence: {{source}}

Classify the sentiment of the following tweet with either positive, negative, or neutral \n{{tweet}}

sin

Completion

Completion

###### Qual é a origem do xaxado?

O xaxado tem origens associadas ao Cangaço, onde cangaceiros, normalmente homens, dançavam com riﬂes em celebrações. Com a entrada de mulheres nos grupos, a participação feminina na dança também se expandiu.

The translation to {{target_lang}} is: \n{{target}}

I would classify the given tweet as: {{label}}

101 +8 Translated NL Generation datasets

101 +2 Translated Text Classiﬁcation datasets

por

7 IndicXParaphrase-inst

نآ ترهش .دراد یاهژیو هاگیاج ناریا ریطاسا رد دنوامد یاهتیصخش زا نودیرف هک تسا نیا رد زیچ ره زا شیب

###### ؟تسا هدش دنوامد هب یاهراشا یناریا ریطاسا رد ایآ

دنب هب یراغ رد اجنآ رد ار کاحض ،ناریا یریطاسا هک نامزلارخآ ات تسینادنز اجنآ کاحض و تساهدیشک تسد هب ماجنارس و دنک زاغآ قلخ تنشک و دلسگب دنب دوش هتشک پساشرگ

pes

- 1 Wiki-split-inst

- 1 Persian-instruct-pn

11 IndicSentiment-inst

5 XWikis-inst 3 Indo-stories-instruct

- 2 Lijnews-instruct

- 2 SCB-MT-2020-prompt 2 Seed-instruct-lij

బట  ద  యడం అం  ఏ  ?

బట  ద  యటమం   ంద   పటం అ  అర ం. ఉ : "  ంగతనం .    నం న ఆ . బట    న  ఆ  ంద  ద న  

1 IMDB-Dutch-instruct 1 Scirepeval-biomimicry-inst

ఇ  ం  "

tel

1 Arpa-instruct 1 Turku-paraphrase-inst 1 FarsTail-Instruct 1 TamilStories 1 Joke-explaination-inst 1 Thirukkural-instruct 1 News-summary-instruct 1 Hindi-article-{task} 1 SODA-inst 1 Urdu-News-Gen-{task} 1 UA-Gec-inst 1 Telugu-{task} 1 Thai-{task}-inst/prompt

Jawapannya ialah Microsoft kerana "Microsoft Windows" (perkataan "window" bermaksud tingkap dalam bahasa Inggeris) ialah salah satu sistem operasi yang banyak digunakan oleh ramai orang di seluruh dunia.

Berikan jawapan untuk teka-teki berikut: Apakah jenama tingkap yang paling ramai orang guna?

Question Answering

Prompt

What category does this question come from: {{question['text']}}?

msa

Completion

This question can come from category: {{document[‘kind']}}.

###### Inis dom faoi Baile Átha Cliath.

Is é Baile Átha Cliath, nó Dublin i mBéarla, príomhchathair na hÉireann agus é ar an gcathair is mó sa tír freisin. Tá sé suite in oirthear na tíre, cois Life agus ar imeall Mhuir Éireann. Tá breis is 1.6 milliún duine ina gcónaí i mórcheantar Bhaile Átha Cliath agus deirtear go dtiocfaidh borradh agus fás ar an daonra sa todhchaí.

101 +9 Translated QA datasets

- 12 AfriQA-inst

1 Amharic QA

1 LLM-Japanese-vanilla-inst

- 9 Mintaka-inst

16 X-CSQA-inst

- 9 AfriSenti-inst

Natural Language Generation

44 Xlel_wd-inst

- 10 NusaX-senti-inst

13 NTX-LLM-inst

- 11 UNER_LLM-inst

1 Urdu-News-Category-Class

- 10 Masakhanews-inst

gle

Maxaa waaye shay ay Soomaali badan cunaan maalin walba oo dadk qaar ay ku adagtahay la'aantiisa in ay cuntada cunaan?

Shay ay Soomaali badan cunaan maalin walba oo dadk qaar ay ku adagtahay la'aantiisa in ay cuntada cunaan waa Mooska. Mooska dad badan oo Soomali ah way jecelyihiin, wuxuuna ka mid yahay waxyaabaha dalka uu ku caan baxay in uu dhooﬁyo.

1 TeluguRiddles

som

[Figure 5]

101 dolly_machine_translated 7 aya_human_annotated 6 dolly-human-edited

Figure 1: Aya Dataset, Aya Collection & Aya Evaluation Suite. On the left, we show examples of contributions in the Aya Dataset. These are original human-curated prompt-completion pairs written by fluent speakers of 65 languages. On the right, we have the Aya Collection, an aggregation of 44 monolingual and multilingual templated instruction datasets and 19 translated datasets ranging over 114 languages and three main tasks: Text Classification, Natural Language Generation, and Question Answering. The bottom block showcases the Aya Evaluation Suite for multilingual open-ended generation. This collection consists of original annotation and post-edits of translations covering several languages, and translation of high-quality and universal prompts into 101 languages. We indicate the number of languages in a dataset with the value in the blue ovals in the figure. (Translated datasets have been visually merged due to space constraints).

pre-trained models on a diverse set of tasks that enable a Large Language Model (LLM) to follow instructions [McCann et al., 2018; Sanh et al., 2022; Wei et al., 2022a; Muennighoff et al., 2023c; Longpre et al., 2023a]. Instruction fine-tuning (IFT) leverages the precept that Natural Language Processing (NLP) tasks can be described via natural language instructions, such as “What were the reviews like for the Barbie movie?" or “Write a recipe from the following list of ingredients." This process requires prompts to be paired with expected completions [Ziegler et al., 2020; Ouyang

- et al., 2022] aiming to capture the variety of ways an LLM can be used in downstream tasks. Yet, the very act of curating data imparts a viewpoint about what distributions we want our model to represent and what is forgotten. So, what do these widely used datasets tell us about the assumptions underlying these breakthroughs?

More than 7,000 languages1 are spoken around the world today, with a considerable number facing the challenges of being low-resourced, under-represented, or disappearing [Maxwell & Hughes, 2006; Simons, 2019; Moran & Chiarcos, 2020; Secretariat, 2022; Gao & Liu, 2023; Ilhomovna & Yuldasheva, 2023; Marivate et al., 2020]. In contrast, the most widely used datasets and breakthroughs in NLP have coalesced around a few data-rich languages [Longpre et al., 2023b; Taori et al., 2023; Chung et al., 2022; Fan et al., 2021; Dodge et al., 2021; Lucy et al., 2024]. IFT datasets are no exception; the creation of these datasets has almost entirely focused on English. Furthermore, the vast majority of the creators of these works originate from a few countries [Longpre et al., 2023b; Zhang et al., 2022].

The factors underlying the construction of the datasets impact how models perform for users around the world. Models perform better on the distribution they are trained to mimic [Kunchukuttan et al., 2021]. This often introduces known biases towards languages [Schwartz et al., 2022; Kotek et al., 2023; Khandelwal et al., 2023; Vashishtha et al., 2023; Khondaker et al., 2023] and dialects [Jørgensen et al., 2015; Blodgett et al., 2016; Zampieri et al., 2017; Sun et al., 2023] not included during training and introduces critical security flaws [Yong et al., 2023a; Nasr et al., 2023; Li et al., 2023b; Lukas et al., 2023; Deng et al., 2023].

Datasets aren’t simply raw materials that fuel breakthroughs but also make the poor poorer and the rich richer [Held et al., 2023; Durmus et al., 2023; Robinson et al., 2023]. Disparities in the access to technological resources predates the advent of LLMs [Garrette et al., 2013]. However, as LLMs become more sophisticated and widely available, non-English languages will remain underrepresented and will likely become more so. The imbalance between languages has created a growing divide in the cost of using this technology as marginalized languages require more tokens and incur higher latency for generations [Ji et al., 2023b; Cui et al., 2023], consigning speakers of lowperforming languages to lower quality technology [Held et al., 2023; Durmus et al., 2023; Nicholas & Bhatia, 2023; Ojo et al., 2023]. Often, speakers of low-resource languages do not have the resources to improve NLP technology for their language, facing a low-resource double bind with limited access to both compute and data [Ahia et al., 2021].

In this work, our goal is to reduce this linguistic inequality. Efforts that aim to improve multilingual performance have often focused on improving data coverage [Chen et al., 2023b]. However, most of the limited effort to date has focused on multilingual pre-training [Scao et al., 2022a; Wei et al., 2023; Lample & Conneau, 2019] with even less work centered on imparting instruction following abilities. Approaches that have tried to translate English instruct-style datasets into other languages often suffer from translation biases [Vanmassenhove et al., 2021; Hartung et al., 2023; Savoldi et al., 2021; Muennighoff et al., 2023c] or fail to reflect cultural context appropriately [Wang et al., 2022a; Ji et al., 2023a; Pudjiati et al., 2022]. Automatic curation of multilingual datasets is a logical —and sometimes necessary— approach but often suffers from noise and biases. This makes it difficult to validate the quality of the created datasets [Kreutzer et al., 2022; Luccioni & Viviano, 2021; Ferrara, 2023; Caswell et al., 2020] or requires the curation of manual templates which often result in low instruction and completion diversity [Muennighoff et al., 2023c] critical for model performance [Naik et al., 2023; Chung et al., 2023; Li et al., 2023e; Lahoti et al., 2023].

In contrast, a key aspect of our work focused on harder-to-obtain human-curated data from fluent speakers of a language. This curation process has received far less attention due to

1https://www.ethnologue.com/

Dataset #Instances#Langs % English Generation method Permissive

license Llama2 IFT data [Touvron

NA 27 90% Human-annotations SFT datasets ✗

- et al., 2023]

Alpaca [Taori et al., 2023] 52K 1 100% Synthetic data generation IFT datasets ≈ P3 [Sanh et al., 2022] 12M 1 100% Template generation given applied to En-

✓

glish datasets

Flan 2022 [Longpre et al., 2023a]

15M 60 100% Template generation applied to English datasets

✓ xP3 [Muennighoff et al., 2023c] 81M 46 39% Template generation applied to English

✓

datasets

Sweinstruct [Holmström & Doostmohammadi, 2023]

68K 1 0% Machine translation English IFT datasets ≈

Okapi [Dac Lai et al., 2023] 158K 26 45% Machine translation English IFT datasets ✓ Bactrian-X [Li et al., 2023a] 3.4M 52 2% Machine translation + synthetic data

≈

generation

Aya Dataset 204K 65 2% Original IFT Human-annotations ✓ Aya Collection 513M 114 3.5% Template Generation and translating ex-

✓

isting datasets

Table 1: Comparison of different instruction-tuning datasets. ✓represents permissive licenses that allow commercial use while ≈ represents restrictive licenses that do not allow commercial use. ✗represents non availability of license.

lack of access to fluent speakers, especially in low-resource languages [Joshi et al., 2019]. We chose to close this gap by conducting a year-long participatory research initiative that involved working with fluent speakers of languages from around the world to collect human-curated instances of instructions and completions. By leveraging best practices from open-source and crowd-sourced science projects [Franzoni & Sauermann, 2014; Beck et al., 2022; Lenart-Gansiniec et al., 2023], we built a simple and intuitive user interface, the Aya Annotation Platform2 (Aya UI) which served as the central platform for contributors to join the Aya3 4 project. In total, we had 2,997 collaborators spread across 119 countries around the world. Their collective efforts resulted in the Aya dataset which is the largest human-curated multilingual instruction-finetuned dataset to date, containing 204,114 high-quality annotations in 65 languages.

Additionally, we release and transform 44 pre-existing datasets into sets of instruction-completion pairs by crafting diverse templates manually, relying on fluent speakers for each language. We further expand this collection by translating datasets from English into 101 languages. We refer to this expanded collection of 513 million instances covering 114 languages in total as the Aya collection, which to date, is the most extensive collection of multilingual instruction-finetuning (IFT) data.

Overall, Aya contributes four key resources: Aya Annotation Platform (Aya UI); Aya Dataset; Aya Collection, and Aya Evaluation Suite. Figure 1 shows a visual representation of the Aya Dataset and Collection. Below, we briefly describe these core contributions:

- 1. Aya Annotation Platform (Aya UI): We built a robust annotation tool to facilitate the collection of high-quality multilingual data in an instruction-style format supporting 182 languages, including dialects. Over eight months, we had a total of 2,997 registered users spanning

- 2This platform is accessible at: https://aya.for.ai
- 3The word Aya has its origins in the Akan (Twi) language and is translated as “fern” in English [Willis, 1998]. 4Aya represents endurance, resourcefulness, and defiance – like a fern growing in barren conditions.

119 countries and 134 languages, including dialects.

- 2. Aya Dataset : We created the largest human-annotated multilingual instruction finetuning dataset to date, consisting of over 204K instances that cover 65 languages. We include a data card [Pushkarna et al., 2022] for the Aya Dataset in Appendix J.

- 3. Aya Collection : We collected instruction-style templates from fluent speakers and applied them to a curated list of 44 datasets, including tasks such as Text Classification, Text Generation, Machine Translation, Paraphrasing, and Open-domain Question Answering. Some of these datasets also include equivalent multilingual versions produced through translation. We release 513M instances that cover 114 languages. These contributions are made available as an open-source collection. We include a data card for the Aya Collection in Appendix J.

- 4. Aya Evaluation Suite : We curate and release a diverse evaluation suite for multilingual open-ended generation quality. It consists of 250 human-written prompts for each of 7 languages, 200 automatically translated but human-selected prompts for 101 languages (114 dialects), and human-edited prompts of the latter for 6 languages, and the English originals. The first set represents culturally-grounded and original prompts, while the translated and post-edited prompts are sourced from English Dolly [Conover et al., 2023] and selected for their cross-cultural relevance. We include a data card for the Aya Collection in Appendix J.

By fully open sourcing the Aya Dataset, Aya Collection and Aya Evaluation Suite with a permissive Apache 2.0 License5 as well as the code for our annotation platform, we hope to empower researchers and practitioners to further advance multilingual models and applications. All datasets are accessible for download.678

Paper Organization Section 2 discusses the design and development of the Aya Annotation Platform, as well as the preparation of the Aya Dataset, and Section 3 presents a detailed analysis of the Aya Dataset. Section 4 and Section 5 contain discussion and analysis of the Aya Collection. Section 6 describes the details of the evaluation suite curated in this project. In Section 7, we describe our approach to participatory research. In Section 8, we review the existing literature, and in Section 9 we discuss the limitations of our work. Section 10 concludes the paper.

##### 2 Aya Annotation Platform & Aya Dataset

###### 2.1 Aya Annotation Platform

The goal of the Aya project is to facilitate annotations to a crowd-sourced dataset by individuals fluent in different languages. Inputs from fluent speakers of each language ensure that the dataset is more likely to be organic, fluent, and representative of the speakers’ cultures. Including fluent and native speakers from various regions poses significant logistical challenges involving meticulous data selection, quality control measures, and custom annotation tools. We developed the Aya Annotation Platform to streamline the data collection process worldwide, accommodating a large number of decentralized contributors across multiple languages.

- 5https://www.apache.org/licenses/LICENSE-2.0
- 6https://hf.co/datasets/CohereForAI/aya_dataset
- 7https://hf.co/datasets/CohereForAI/aya_collection
- 8https://hf.co/datasets/CohereForAI/aya_evaluation_suite

Registered Users

350

310

270

230

200

160

120

80

40

0

Figure 2: Geographical distribution of the users registered on the Aya platform.

User Interfaces (UIs) play a pivotal role in the context of NLP data collection, serving as the primary point of interaction between human annotators and the data collection process. The Aya Annotation Platform9 had to accommodate users in 119 countries collecting data across 134 languages. We designed the platform with a few key principles in mind, such as accessibility and ease of use for users who were unfamiliar with AI and machine learning. As part of our contribution, we fully open-source the code for our UI10.

Accessibility As users worldwide use different devices and operating systems, we decided to support both mobile and desktop interfaces [Muhammad et al., 2023]. Approximately 54% of users accessed Aya UI via desktop browsers while 46% utilized mobile browsers. We attribute the high fraction of mobile users to the skew towards mobile users in the Global South [Avle et al., 2018]. We supported Single Sign-On (SSO) capabilities to enable seamless tracking of user profiles and reward users with points for contributing data across multiple sessions. We initially only supported Discord sign-on but discovered that Discord is inaccessible or not widely used in certain countries. Also, the necessity of a platform-specific account created an obstacle to user engagement with Aya. This prompted us to add Google sign-on as an alternative option.

Languages Supported Aya project contributors could select the languages they are proficient in when signing up using the Aya UI. They could then make annotations in the language(s) they selected. Given the sheer number of languages we could collect annotations for, we chose to prioritize annotation support for the 101 languages available in the mT5 model [Xue et al., 2021]. We note that ultimately, some of these languages didn’t receive enough contributions to include them in the final dataset. Conversely, we received substantial contributions from languages not initially part of the original list, like Wolof, leading to their inclusion; the final Aya Dataset covers 65 languages. Table 5 provides details of these languages.

Contributors We aimed to include individuals from diverse backgrounds—not limited to AI experts—enabling anyone proficient in a language to contribute. Our pool of contributors ultimately reflects this inclusive approach. During the registration process, we request specific demographic details from each Aya UI user such as country of residence, languages of fluent communication,

- 9https://aya.for.ai/
- 10https://github.com/for-ai/aya-annotations-ui

###### gender, age range, and familiar dialects. We display the onboarding form in Figure 17 in the Appendix. The Aya community of contributors includes 2,997 registered users across 134 languages.

Registered Users by Age

Registered Users by Gender

50

100

40.7%

40

80

37.8%

68.1%

30

60

Percentage

20

40

28.4%

14.1%

10

20

4.5%

1.8%

2.4% 0.7% 0.3%

1.1%

0.0% 0.1%

0

0

Male Female PreferNotToSayNon-Binary Other

0-18 18-25 25-35 35-45 45-55 55-65 65-75 75-121

Figure 3: Left: Distribution of registered users on the Aya UI by age using specified values. Right: Distribution of registered users on the Aya UI by gender using specified values

Demographics Figure 3 illustrates the demographics of registered Aya UI users by age and gender. Regarding the age profiles of users, more than two-thirds were aged between 18 and 35. Approximately 68.1% of users identified themselves as male and 28.5% as female. Overall, 6.6% of users self-reported dialects. Within this group, 75% specified one dialect, 20% specific two dialects, and the remaining 5% specified three or more dialects, with a maximum of six.

During the development of Aya, registered users were geographically distributed across 119 countries based on their residence. Certain countries like Afghanistan, Bulgaria, Kuwait, and Tajikistan had just one registered user. Figure 2 displays this global distribution, highlighting India with the highest number of registered users (346 out of 2,997).

[Figure 6]

Geographic-Based Contribution Assessment We grouped the languages by the regions in which they either originate or are widely spoken. The language statistics by region for the original 101 languages we wanted to cover are as follows: 14 languages in Africa, 41 languages in Asia, 42 languages in Europe, and 4 languages in Latin America (See Appendix C.1 for more details and some exceptions of the distribution). As seen in Figure 4, more than half of all contributions for the Aya project came from Asia with 58.8%, followed by the African region with 27.4%. Europe, Latin America, and other regions account for the remaining 13.8% of the contributions.

Figure 4: Distribution of total contributions across different regions.

We observe a large skew in terms of regional contributions, which deserves further research to understand why certain networks of contributors remained motivated for the entire project. These

disparities in participation may be due to opportunity cost in time [Gerosa et al., 2021; Wu et al., 2007], cultural beliefs around sharing data [Huang et al., 2023], or the belief that the language in question is not well served by the current technology [Nicholas & Bhatia, 2023].

Acknowledgement of contributions Recognition and transparency were maintained throughout the project through the use of a leaderboard11 to acknowledge contributions. We implemented a scoring system where contributors earned a maximum of three points for each re-annotation, with one point awarded for rating the prompt and completion, one point for editing the prompt, and one point for editing the completion. Each original annotation was awarded with two points. We describe the different annotation tasks in the Aya UI in detail in Section 2.2.

The Aya Leaderboard is organized to display daily, weekly, and cumulative scores, providing a comprehensive overview of user contributions. The users have the flexibility to filter scores based on specific languages, allowing for a sense of community amongst contributors of a particular language. This design aimed to boost contributors’ motivation to provide high-quality inputs for their chosen languages. Figure 19 shows an example of the leaderboard. We discuss further details on collaborating with the community in Section 7.

###### 2.2 Annotation tasks

On the Aya Annotation Platform, contributors were able to contribute to three different tasks, following the find-fix-verify paradigm [Bernstein et al., 2015]: Writing new examples from scratch (original annotations), editing existing examples to improve the quality and comprehensiveness (re-annotations), and giving feedback on the quality of existing contributions (annotation feedback). We describe each briefly below:

###### 2.2.1 Original Annotations

This task facilitates the inclusion of human-generated organic content by allowing annotators to submit original prompt-completion pairs in their language. Existing multilingual models have been shown to produce generations influenced by Western culture [Yuan et al., 2021; Naous et al., 2023; Lee et al., 2023] reflecting the underlying representation bias [Mehrabi et al., 2021] of their training datasets. This task aims to encourage annotators to submit fresh samples that are representative of their language, culture, literature, history, and region. The guidelines for contributors is available in Appendix B.3.

###### 2.2.2 Re-Annotations

The purpose of this task is to facilitate the re-annotation or editing of prompt and completion pairs. The decision to add a re-annotation task partly stems from the need to help annotators understand the expected format of instruction-style datasets and to convey the variety of tasks in existing datasets, including question answering [Saad-Falcon et al., 2023; Arefeen et al., 2023], summarization [Stiennon et al., 2020; Wu et al., 2021], paraphrasing [Witteveen & Andrews, 2019; Reimers & Gurevych, 2019], and translation [NLLB-Team et al., 2022; Barrault et al., 2023]. Editing examples from existing datasets not only helped familiarize annotators with the expected format

11The Aya Leaderboard is accessible at: https://aya.for.ai/leaderboard/

[Figure 7]

[Figure 8]

Leaderboard

[Figure 9]

[Figure 10]

Leaderboard

Contribute Your Language

Rate Model Performance

Help improve the model performance by rating the prompt-completion pairs below

Help us add more examples by writing your own prompt and their completion in your language.

Edit Mode

English

English

Prompt

Look out for: Correct Grammar Reasonable Length Clear Instructions

Prompt

[Figure 11]

[Figure 12]

[Figure 13]

What does it mean in English "to crave" something?

[Figure 14]

[Figure 15]

[Figure 16]

what does "to crave" mean?

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

Edit Distance[Prompt]: 35

[Figure 23]

[Figure 24]

Completion

Completion

Look out for: Clear Answer Full Sentences with Correct Answer Reasonable Length

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

To crave is to have a strong and intense desire for something. It implies a deep need or longing for the object of the craving. It can involve emotional and physical feelings of addiction and can sometimes lead to actions against one's best interests. It is important to note that there is a difference between wanting or liking something and craving it, as craving can be destructive and lead to unhealthy behaviors.

to crave is to really really want something, or to have a deep desire.

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

Edit Distance[Completion]: 321

Skip Submit Entry

Submit Entry

(a) Example of an original annotation contribution.

(b) Example of a re-annotation contribution.

[Figure 39]

Made by the C4AI Open Science Community. The Aya Project

- Figure 5: Demonstration of a sample original annotation and re-annotation contribution, in English. (a) exemplifies an original contribution input by an annotator from scratch. (b) shows a sample of re-annotation. Here, the annotator may have improved upon either a prompt and completion pair pulled from the database or a prompt and completion originally created by another contributor.

but also allowed for human evaluation and rating of existing widely used instruction-style datasets.

In total, we collected datasets from 19 public data sources and translated them into 114 available languages, including dialects using the NLLB 3.3B parameter machine translation model [NLLBTeam et al., 2022]. From each collection, we randomly chose 100 examples (per dataset and per language), creating our dataset for annotation, after which we had 1M translated prompt-completion pairs initially populated in the Aya UI as re-annotation tasks. These translated pairs served as a starting point for prompts and completions which annotators could improve. We release the raw translations as part of the Aya Collection, provide more details about the provenance of the translated datasets, and how they were selected in Section 4.2.

In addition to translated examples, there are other available data sources suitable for re-annotation: original Aya pairs, pre-existing instruction-style datasets (e.g., xP3), and the transformation of datasets into an instruction-style format, i.e., templated datasets. By re-annotating examples from different sources, we simultaneously enhance the quality of individual examples while obtaining a signal on the overall quality of the dataset in a specific language. A demonstration of a re-annotation, where an annotator strengthens a given prompt/completion, is shown in Figure 5b.

###### 2.2.3 Annotation Feedback

Data quality is critical to ensure that a model can represent a language well. Learning from noisy, low-quality datasets harms the overall model performance and the relatively high cost of encoding these noisy examples is a misuse of capacity [Hsueh et al., 2009; Dodge et al., 2021; Luccioni & Viviano, 2021; Kreutzer et al., 2022]. Prior work has shown that improvements to quality through data pruning or selection can have an significant impact on the downstream performance of a model [Longpre et al., 2023c; Marion et al., 2023; Boubdir et al., 2023; Yang et al., 2023]. In particular, for instruction-tuning datasets, a small subset of higher-quality instructions can greatly outperform a larger volume of lower-quality instructions [AlShikh et al., 2023; Zhou et al., 2023; Chen et al., 2023a]. Given these findings, ensuring high quality contributions is of paramount importance. Ensuring consistent quality is particularly challenging in an open science initiative with a large number of contributors. We face two key challenges:

Changes in the Annotator Pool. During the year-long project, annotators joined and left the project at different points depending on their interests and availability. As a result, the window of contribution for each annotator was different. Only a small fraction of annotators participated for the entire duration of the year-long project. Annotators were active for an average of 1.3 sessions.

- Figure 6 presents a histogram depicting the distribution of user engagement based on the number of days they actively contributed. On average, Aya annotators spent five days contributing to the project. Annotators tended to be highly active shortly after joining, but their activity declined over time. There was a subgroup of annotators who maintained consistent activity over extended periods.

Varying level of experience with AI. An important goal of this project was to have a diverse pool of annotators and we thus did not limit the selection criteria to working knowledge of language models or AI in general. As a result, there were different levels of understanding amongst the annotators what was meant by a prompt and completion. For example, we found at least one contributor with 3,684 contributions to three languages (English, Somali, Standard Arabic) who failed to structure their submissions as a prompt with a question. Instead, the contributor used an extract of text as the prompt and its continuation in the completion. Prefacing such prompts with an instruction such as “Complete the following partial extract of text:" would have been a more suitable format.

[Figure 40]

- Figure 6: The distribution of annotators’ engagement based on the number of days they actively contributed in Aya UI

While we routinely provided examples to contributors, there was a clear need for a systematic way to review and measure the quality of submissions.

Validating the quality of contributions We follow a peer-review approach where each annotator acts as a reviewer for the other annotators working on the same language. These reviews form the

basis for a quality Aya score which is displayed on the leaderboard in the UI. The quality score for an annotator is calculated by averaging the combined average ratings of their examples provided by other annotators who serve as reviewers. We provide more details about how annotations are reviewed in the Appendix Section B.1. All three tasks in the Aya UI are connected in a sequential pipeline where submissions from “Original Annotations” are reviewed in the “Re-Annotations” task, and the re-annotations are further reviewed as part of the “Annotation Feedback” task. This systematic approach allows for a robust evaluation and enhancement of the collected data.

###### 2.3 Criteria for Inclusion in Aya Dataset

The Aya Dataset includes all original annotations and a subset of all re-annotations. We only release re-annotations if there is a difference between the original and the edited version. To determine this subset, we compute the sum of edit distances d (Levenshtein distance [Levenshtein et al., 1966]) between the original and re-annotated prompts and completions on the character level and use an acceptance threshold of (d ≥ 5). This ensures that we do not release duplicates of existing data.

Only languages with at least 50 contributions were included in the final release of Aya Dataset. This threshold was picked as it represents a balance between achieving a reasonable level of data quality and considering the practical limitations of human resources for some languages. The goal is to include as many languages as possible without lowering the overall quality of the dataset. Table 5 lists details of the languages included in the Aya Dataset.

Count Original Annotations 138,844

xP3 datasets 2859 Translated datasets 7757 Templated datasets 11013 Original Annotations 43641

Re-Annotations

Aya Dataset Total 204,114

Table 2: Aya Dataset Statistics (number of pairs of prompts and completions obtained through various annotation tasks).

- 3 Analysis of Aya Dataset

###### 3.1 Statistics

The Aya Dataset contains a total of 204,114 instances collected via the Aya Annotation Platform. Table 2 provides the breakdown of original annotations and re-annotations in the final dataset. The dataset covers 65 languages: 22 high-resource, 12 mid-resource, and 31 low-resource languages (See Appendix E for more details on our language mappings).

###### 3.2 Length of Aya Dataset

One objective of this project was to collect fluid original human prompts and completions. Table 3 provides examples of prompts and completions from the Aya Dataset. During the data collection

600

AverageCompletionLength

500

400

300

200

100

0

Initial Length

Edited Length

| |
|---|

xP3* Translated* Templated* AYA original annotations*

- Figure 7: Average Completion Length before and after re-annotation. Here (*) indicates the subset of all dataset categories (xP3, translated, templated, and Aya original annotations) that were included in the Aya Dataset after re-annotation. Re-annotation improves average completion length across all datasets.

process, annotators were provided with examples and guidelines but were also trusted to explore their own creativity and cultural background to come up with new examples. As a result, it is meaningful to understand differences in aggregate statistics like length across datasets, language type and relationship with perceived quality.

Impact of Re-Annotation When editing existing instances, we instructed the annotators to prioritize enhancing both the quality and richness of the prompts and completions. The average length of completions before and after edits are shown in Figure 7. We observe that across all data sources, the average length of completion increased after editing. On average, the length of completions after edits is 25% longer than before edits. We observed the largest increase for Aya original annotations surfaced in the UI – which were 40% longer on average than the original length.

HR

MR

LR

| |
|---|

| |
|---|

600

500

400

AverageLength

300

200

100

0

Prompt Completion

Length difference across language groups The average prompt and completion length (number of characters) observed across these different language groups is shown in Figure 8. A distinct contrast exists in completion lengths between mid and low-resource languages when compared to high-resource languages. Long completions and complete sentences are valuable

Figure 8: Average prompt and completion length of instances in the Aya Dataset across different language categories (high (HR), mid (MR) and low (LR) resource languages, see Table 5).

[Figure 41]

- Figure 9: Relationship between Average Prompt and Completion Length in characters and the Average Approval Rate of the example.

in instruction-tuning datasets, particularly when training multilingual models to generate content in those languages.

Length vs. Perceived Data Quality Although longer completions can be valuable for training models to generate long and natural text, it does not necessarily imply higher quality. Using annotators’ feedback in the UI, we further investigate the impact of length on the perceived quality of the samples. Figure 9 showcases this analysis. We observe a positive correlation between how long the prompts and completions are and their resulting average approval ratio. Specifically, when we plot combined prompt and completion length against quality, we observe a correlation coefficient of 0.27. This finding emphasizes the importance of using longer prompts and completions and incorporating complete sentences to ensure a positive human experience when engaging with such a model.

Length difference by language Figure 25 in the Appendix illustrates the statistics per language. We observe an array of patterns that differed across languages. For instance in Japanese, completions are on average 31% shorter than prompts. On the other end, for Urdu and Yoruba, completions are notably long relative to prompts. On average, completions are 1258% and 2516% longer than the corresponding prompts for Urdu and Yoruba, respectively. The average completion length in Yoruba is 1591% longer than the average prompt length in Japanese. Figure 26 provides the average length of the combination of prompts and completions per language.

Comparison in Completion length Across Datasets The Aya Dataset has considerably longer completions on average when compared with other data collections as shown in Figure 12. This is particularly noteworthy given that the Aya Dataset is human-curated. Given the presence of longer completions in the training data for many low-resource languages, we expect that models trained on the Aya Dataset will generate longer and more natural responses.

###### 3.3 Annotator Skew

A feature of participatory research projects is the challenge of establishing and maintaining a balanced number of annotations across groups of annotators. In the Aya project, the number of annotators per language varied due to numerous factors. As a result, the distribution of annota-

Lang Prompt Completion ara .     ﺍ ﺡ   ﺓ     ﺀ      ً    َ   ﻙ ُ   ﺍ ﺡ َ  ﻥ ﺇ ﺩ  ﻭ   ﺍَﻭ ﻉ    ﺍ    َ   َ   ُ   ِ     ِ   ﺍ ّ    َ  ﺃ َ  

ﺀ      ﺍ  َ  َ ﻱﺯﺍُ   َ

fra Qui a écrit le livre La Sagouine?

Antonine Maillet a écrit La Sagouine en 1971. Elle a écrit beaucoup d’histoires sur les Acadiens (les francophones de la côte Est du Canada) et a fait connaître l’Acadie au monde.

ibo Kedu ka ndị Igbo si eme emume ji ọhụrụ?

Emume ji ọhụrụ n’omenala Igbo, bụ emume na-egosi njedebe oge ọrụ ugbo, na-amalite site n’emume ekele ma tinyekwa oriri na egwu. Ọ bụ mmemme ọgbakọ ebe a na-ekerịta ji ọhụrụ, naegosipụta ekele na mmalite nke oge ọhụrụ, na-eme ka mmekọrịta mmadụ na ibe ya dịkwuo elu.

por Qual é a capital do estado da Bahia?

A capital do estado da Bahia, no Brasil, é Salvador. Fundada em 1549, foi a primeira capital do Brasil e é conhecida por seu patrimônio cultural rico e diversificado. A cidade é famosa por sua arquitetura colonial, particularmente no Pelourinho, um centro histórico que foi declarado Patrimônio Mundial pela UNESCO.

pes ﺮﺗﻮﺒﮐ ﻪﻧﻻ ﺪﻨﭼ ﺪﺒﻴﻣ ﻪﻧﺎﺧﺮﺗﻮﺒﮐ ﺝﺮﺑ ﺭﺩ ؟ﺩﺭﺍﺩ ﺩﻮﺟﻭ

ﺭﺍﻮﻧ ﻭ ﺩﺭﺍﺩ ﺩﻮﺟﻭ ﺮﺗﻮﺒﮐ ﻪﻧﻻ ﺭﺍﺰﻫ ﺭﺎﻬﭼ ﻥﺁ ﻞﺧﺍﺩ ﻪﮐ ﺖﺳﺍ ﯼﺍ ﻪﻧﺍﻮﺘﺳﺍ ﯽﺟﺮﺑ ﺪﺒﻴﻣ ﻪﻧﺎﺧ ﺮﺗﻮﺒﮐ ﺮﻄﺧ ﺮﺑﺍﺮﺑ ﺭﺩ ﺝﺮﺑ ﻦﻳﺍ .ﺖﺳﺍ ﻩﺪﺷ ﻩﺪﻴﺸﮐ ﺝﺮﺑ ﺭﻭﺩﺎﺗﺭﻭﺩ ﻥﺍﺮﺗﻮﺒﮐ ﺏﺬﺟ ﯼﺍﺮﺑ ﯽﮕﻧﺭ ﺪﻴﻔﺳ

.ﺖﺳﺍ ﻩﺩﻮﺑ ﻦﻤﻳﺍ ﺭﺎﻴﺴﺑ ﺎﻫﺮﺗﻮﺒﮐ ﻪﺑ ﺕﺎﻧﺍﻮﻴﺣ ﺮﮕﻳﺩ ﻪﻠﻤﺣ

msa Apakah nasi lemak? Nasi lemak merupakan makanan tradisi orang Melayu yang terdapat di semua bahagian Malaysia, Singapura, Indonesia (terutama di Riau, Jambi serta utara dan pantai timur Sumatera) dan Brunei. Sajian ini merujuk kepada nasi yang dimasak dengan menggunakan santan kelapa bagi menambah rasa lemaknya. Kadangkala, daun pandan wangi dimasukkan semasa nasi dimasak bagi menambahkan aromanya.

tam ெசயற்ைக நுண்ணற வு என்றால் என்ன?

ெபாதுவாக மனிதர்களால் ெசய்யப்படும் பணிகைளச் ெசய்ய ஒரு கணினி அல்லது ஒரு கணினியால் கட்டுப்படுத்தப்படும் ஒரு ேராேபாவ ன் த றன் ெசயற்ைக நுண்ணற வு எனப்படும்.

Table 3: Examples of prompt and completions in the Aya Dataset.

tors is not uniform across languages. Moreover, within each language, there is a lack of consistent contributions from all annotators. In this section, we examine the impact of annotator skew on the resulting dataset.

Annotator Skew Across Languages. Annotators were encouraged to contribute to any language in which they could comfortably read and write and were asked to focus most of their efforts on languages other than English. Although a significant number of participants registered for many languages, the engagement level of annotators was not equal, which resulted in considerable differences in the number of contributions across languages. Figure 10 (top) provides an overview of the percentage of each language present in the final compilation. The highest number of contributions is for Malagasy with 14,597 instances, and the lowest is 79 for Kurdish.

Annotator Skew Within a Language. The final contributions for each language in the Aya Dataset are not evenly distributed among annotators. The median number of annotators per language is 15 (mean is 24.75) with one language having only a single active annotator (Sindhi) and

Annotationpercentage(%)

6

4

2

0

1.00

[Figure 42]

amh

som

mya

msa

swa

swe

mar

nep

eng

hau

deu

hun

dan

ben

pan

tam

spa

kan

pes

zho

pus

sna

xho

nso

ceb

eus

nya

sun

snd

mlg

mal

wol

por

ara

urd

tha

hat

rus

srp

kor

ukr

yor

kur

hin

guj

ibo

ind

pol

nld

gle

jpn

sin

vie

jav

sqi

zul

fra

tur

ita

tel

fin

kir

ell

lit

fil

0.75

| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|[Figure 43]| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |

Top 1

Top 2

Top 3

0.50

Top 4

Top 5

0.25

High number of active annotators Low number of active annotators

- Figure 10: Top: Ratio of all annotations per language with respect to the whole dataset. Bottom: Ratio of annotations done by the top-k most active contributors (k = 1,...,5). Languages annotations follow their respective ISO codes from Table 5.

some having over 80 annotators (English and Portuguese). Note that annotators made contributions at varying rates, and there is no direct correlation between the number of annotators and the ultimate count of language contributions. A limited pool of annotators for some languages implies that most instances in that language originate from a smaller group of individuals. Figure 10 (bottom) illustrates the proportion of instances in a language originating from the most active annotators. We observe a skewed pattern where for 12 languages, the 5 most active annotators contributed all examples. There is an uneven distribution of contributions for many languages because those languages had a smaller number of voluntary annotators throughout the entire project despite rigorous outreach. Additionally, we did not establish a specific quota for annotators to meet; everyone contributed as they desired, resulting in varying levels of activity among annotators.

The most extreme cases are Zulu and Sindhi, where one annotator in each language volunteered for all contributions in Annotation and Re-annotation tasks. Thus, in Figure 10 their top-1 contributor ratio is 1.0 and does not change when moving to top-2 or further. The languages with the least skewed distributions are Malagasy, Tamil, Nepali, Hindi, English and Portuguese. The language English also had the highest number of unique annotators with 130 individuals out of which 95 annotators contributed to English as their second language for annotation purposes. Given the uneven distribution of annotators per language, it is important to acknowledge that individual annotator quality has a disproportionate influence on some languages.

###### 3.4 The impact of introducing the Aya Score

As part of our collaborative annotation effort in Aya, we emphasized the importance of quality as well as long completions that contain clear responses to the instructions specified in the prompt during the project. To encourage high-quality examples from the annotators, we introduced the Aya Score (Section B.2) halfway through the project to focus on the quality, in addition to the quantity, of contributions.

The Aya Score encouraged participants to incorporate more edits during annotation, with one specific guideline urging them to transform short answers into full sentences or paragraphs. Figure 11(right) shows the change in the completion lengths over time. We observe that after introduction of the Aya score, there is a marked uptick in the completion length of all submitted annotations.

40000

VolumeofSubmissions

Aya Score introduced

30000

20000

10000

0

May2023 Jun2023 Jul2023Aug2023Sep2023Oct2023Nov2023Dec2023

Original Annotations and Re-annotations Original Annotations Re-annotations

800

AverageCompletionLength

600

Aya Score introduced

400

200

May2023 Jun2023 Jul2023Aug2023Sep2023Oct2023Nov2023Dec2023

(a) Volume of submissions over time

(b) Average completion length over time

- Figure 11: The volume of original annotations and re-annotations increases after the introduction of Aya Score. We also observe a marked uptick in the completion length of all submitted annotations with the introduction of the Aya Score.

##### 4 Aya Collection

We introduce the Aya Collection, a comprehensive, large corpus of datasets that can be used by researchers around the world to train multilingual models. Our goal is only to include datasets with permissive licensing for manipulation and redistribution.12 Where possible, we report the license associated with each dataset within the Aya Collection.

The Aya Collection consists of three different sources of data:

- 1. Templated data: We collaborated with fluent speakers to create templates that allowed for the automatic expansion of existing datasets into various languages.
- 2. Translated data: We translated a hand-selected subset of 19 datasets into 101 languages (114 dialects) using the NLLB 3.3B parameter machine translation model [NLLB-Team et al., 2022]. The full list of datasets translated is listed in Appendix Table 9.
- 3. Aya Dataset: We release the Aya Dataset described in Section 3 as a subset of the overall collection. This is the only dataset in the collection that is human-annotated in its entirety.

Dataset Selection Criteria The templated and translated datasets in the Aya Collection were selectively hand picked to achieve a mix of different task types. Our criteria prioritized datasets with high-quality natural and complete sentences, suitable for creating pairs of prompts and completions. Datasets that could potentially yield single-word answers were deliberately excluded. Finally, to

12https://en.wikipedia.org/wiki/Permissive_software_license

500

AverageCompletionLength

400

300

200

100

0

Aya Dataset xp3 Aya Collection (Templates)

Aya Collection (Translations)

- Figure 12: Comparison of completion lengths between Aya Dataset, Aya Collection, and xP3 (excluding the "code" split).

create a high-quality collection, we examined all datasets and excluded those identified as unclean or noisy, primarily attributable to their automatic creation processes.

###### 4.1 Templating Existing Datasets

We explored the automatic expansion of existing datasets in various languages with human-written prompt templates, following previous works [Mishra et al., 2022; Bach et al., 2022; Wei et al., 2022a; Wang et al., 2022e]. Unlike prior works that still either use English prompts in a multilingual dataset or rely on automatic translation to generate multilingual prompts, to our knowledge, Aya Collection is the first broad effort to involve fluent speakers in creating prompts unique to their language to expand existing datasets for instruction tuning.

We used the PromptSource framework [Bach et al., 2022] to template these datasets. We asked Aya community members to submit instructions and create templates for datasets in the languages they were proficient in. Our process includes: 1) Templating datasets with instructions in the same language as the original dataset; 2) If the dataset is not in English, annotating instructions in English. Our input prompts can be monolingual or code-mixed, depending on whether we apply templates in the same language or in English to the dataset of a particular language. Note that code-mixed input prompts here refer to a structured mixing of English instructions with non-English monolingual data [Lin et al., 2022], which is different from the typical sociolinguistic definition of code-mixing (or code-switching) of languages in natural conversational utterances [Winata et al., 2023a; Yong et al., 2023c; Doğruöz et al., 2023; Srivastava & Singh, 2021].

We examined the suggested templates and subsequently converted each dataset into an instructionstyle format. We release these datasets under the Aya Collection. We list the details of all datasets we apply templates to in Appendix Table 8.

###### 4.2 Automatic Translation

Research has demonstrated that training models with translated data can yield significant benefits [Aharoni et al., 2019; Zhang et al., 2018b; Tang et al., 2021]. We experiment with improving coverage of low-resource languages by selectively translating high-quality datasets from various existing collections.

Setup We selectively pick 19 high-quality IFT datasets from xP3 [Muennighoff et al., 2023c], the Flan Collection [Longpre et al., 2023a], Dolly [Conover et al., 2023], along with additional sources such as SODA [Kim et al., 2022] and Mintaka [Sen et al., 2022]. Datasets were prioritized for translation based on the richness of task diversity and length of completions. The complete list of these datasets is given in Appendix 9. These translations are available and open-source as part of the Aya Collection. We process datasets for translation using the No Language Left Behind (NLLB) [NLLB-Team et al., 2022] machine translation model, which is capable of single-sentence translations between 200 different languages and dialects in various scripts. For best performance, we use the largest NLLB model with 3.3B parameters.

Translation Quality Appendix Section G.1 lists NLLB translation quality for each of the languages of interest, as reported in [NLLB-Team et al., 2022]. Figure 13 shows the translation quality across languages grouped by their resourcefulness. The mean ChrF++ score on FLORES is 48.17 (min: 10.9, max: 69.6) for translations out of English, with a few outliers for HR and LR. We interpret this optimistically as strong enough to sufficiently serve our translation needs. However, upon inspection of translation outputs for fine-tuning data, we encounter significant translation errors with Standard Arabic in Latin script and Minangkabau in Arabic script, so we exclude them from our translated dataset. In total, 19 public datasets were translated into 101 languages (114 dialects). Details of these datasets can be found in Appendix Table 9.

70

60

50

ChrF++

40

30

20

10

HR MR LR

Figure 13: ChrF++ scores for the NLLB translation model, averaged across resourcefulness buckets.

In addition to releasing the translated datasets used as a basis for re-annotation, we also translated Dolly [Conover et al.,

2023]. Dolly is a 15k instruction dataset Databricks collected by relying on its employees as annotators [Conover et al., 2023]. Annotators were instructed to curate prompt and completion pairs in each of eight different instruction categories. In contrast to the mentioned NLP datasets, Dolly was purposefully designed to align language models with human expectations. It stands out as a high-quality, manually curated dataset covering a range of topics including brainstorming, classification, closed question answering, generation, information extraction, open question answering, and summarization. The addition of the translated Dolly datasets is a valuable resource for languages that experience a scarcity of conversational instruction fine-tuning datasets.

The list of datasets, along with the number of languages, templates, and other statistics, can be found in Appendix Table 9.

Main Task Type Fine-grained Task Type Question Answering Natural Language Generation Summarization

Translation Paraphrasing Dialogue Text Simplification

Text Classification Sentiment Analysis Information Extraction Named Entity Recognition Event Linking Natural Language Inference Document Representation

Table 4: Task Taxonomy of NLP tasks in the Aya Collection.

##### 5 Analysis of Aya Collection

###### 5.1 Statistics

Overview The Aya Collection consists of existing NLP datasets that are templated to include instructions as well as datasets already in instruction format submitted by the Aya community. Table 8 shows the detailed list of datasets. The full list of templates is available in Section K. The final Aya Collection consists of 44 multilingual and non-English templated datasets and 19 translated datasets, with 513M individual instances. Overall, the collection covers 114 languages13.

Tasks Covered Across Templated and Translated Datasets We aim to include datasets from various tasks in the collection while ensuring that they follow our selection criteria. Table 4 illustrates our task coverage in the Aya Collection, drawing inspiration from xP3 and the Flan Collection. We have a total of three main task types: Question Answering (QA), Natural Language Generation (NLG), and Text Classification (TC). Within these larger umbrella tasks, we define several finer-grained task types based on the datasets, resulting in a total of 11 finer-grained task types. These finer-grained task types are determined by the frequency of datasets in the Aya Collection encapsulating that task.

For QA, we decided to keep only the main task type, as the intended goal of question-answering tasks is clear: Answer a proposed question. The type of the question can be different: open-ended, closeended, multiple-choice, single response. For NLG, finer-grained task types include Summarization, Translation, Paraphrasing, Dialogue (Generation), and Text Simplification. For TC, we include the following finer-grained task types: Sentiment Analysis, Information Extraction, Named Entity Recognition, Event Linking, Natural Language Inference, and Scientific Document Representation. Finally, we label the task categories of each dataset in the Aya Collection in Table 10 and Table 11. If we are not able to find a fine-grained task type for the dataset, we keep the main task type.

13We release the Aya Dataset as part of the Aya Collection, bringing the total number of languages in the collection to 115. However, for the sake of clarity, when referencing the Aya Collection statistics in this paper, we exclude the Aya Dataset.

Number of Wikipedia Articles

Median Number of instances in Aya Collection

- 102

- 103

- 104

- 105

- 106

| |
|---|

- 102

- 103

- 104

- 105

- 106

NumberofinstancesinAyaCollection

NumberofWikipediaArticles

tha

cat

fra

tur

tam

afr

eng

deu

hun

dan

heb

ben

hau

spa

pes

sun

kan

hye

ita

fin

ces

tel

nor

ron

arb

urd

amh

rus

ukr

kor

srp

hrv

som

pol

jpn

nld

ind

bul

hin

bel

guj

ibo

swe

vie

jav

swh

sin

zul

kin

slk

slv

mar

twi

mal

wol

ell

lij

- Figure 14: Number of prompt/completion pairs in each language in the Aya Collection (templated). Many languages with limited digital presence, as indicated by a low number of Wikipedia pages, are well-represented in the templated portion of the Aya Collection. Note that absolute Both axes are in log-scale.

Language Balance One of the objectives of templating (and translating) existing datasets is to broaden the available resources for languages that have limited digital data. To examine if our final collection adheres to a similar distribution pattern, we use the number of Wikipedia pages in each language as a proxy for the online presence of its fluent speakers. Figure 14 showcases that although the number of instances for languages varies in the Aya Collection (templated subset), it does not disadvantage languages with fewer Wikipedia pages. The distribution still ensures a reasonable coverage across all languages. It is imperative to emphasize that our analysis does not involve a direct comparison of absolute values, given the disparate units of measurement involved. Instead, we examine the patterns of data scarcity for various languages in our collection versus Wikipedia. Including the translated datasets in the Aya Collection further reduces disparities between languages and contributes to creating a more balanced collection.

Prompt and Completion Lengths Figure 15 shows the distribution of length across languages. No discernible pattern is observed when examining lengths for high-resource languages compared to low-resource languages. Low-resource languages appear at both ends of the distribution, occupying both the head and tail. In the Aya Collection some low-resource languages (e.g., Somali and Amharic) have longer average completions length than medium or even high-resource languages. The dedication of individual participants in identifying datasets in their own language and templating them has made a significant difference for many languages.

1000

Language Category

HR MR LR

| |
|---|

| |
|---|

Averagelength(prompt+completion)

| |
|---|

800

600

400

200

0

mri

lij

isl

mlt

cym

fil

mar

lit

slv

slk

twi

mlg

mal

ltz

ell

wol

hrv

smo

som

yor

rus

mya

khm

mkd

kur

ukr

srp

kor

yid

sqi

zul

swa

jav

vie

swe

lav

sin

kin

fra

tam

afr

tur

tel

ces

ita

kas

kaz

fin

urd

ron

por

mon

nor

ara

amh

hin

nld

gle

ind

cat

glg

tgk

pol

ibo

bul

bel

guj

kat

lao

est

jpn

ceb

xho

spa

kan

sna

uzb

sun

kau

pes

hye

eus

snd

zho

tha

deu

eng

hau

ben

epo

dan

hun

nep

heb

- Figure 15: The average length of prompts and completions for high (HR), medium (MR) and lowresource (LR) languages in Aya Collection.

###### 5.2 Quality Assessment of All Different Data Sources

As previously stated, binary feedback on the quality of the prompt-completion pairings was collected from the annotators. We define the average approval ratio per dataset which serves as a valuable metric for assessing the quality of datasets across various languages and diverse data sources. We compute the average approval ratio as T+/T , where T+ represents the total number of thumbs up, and T represents the total number of votes per dataset. An average approval ratio of 1.0 would indicate that every annotation was perceived to be of good quality and all prompts and completions had received a thumbs up. An average approval ratio of 0.0 would indicate that every annotation was perceived to be of poor quality, and all prompts and completions had received a thumbs down. We constrained our quality analysis to the 40 datasets in our pool for which we had at least 20 instances of feedback.

Overall, we observe that the majority of datasets were of above average (0.5) quality based on their approval ratio, with all translated data as well as Original Annotations being above average. However, across all the datasets within each group —xP3, Templated, Translated, and Aya original annotations— Aya original annotations were perceived to be of the highest quality, with an average approval ratio of approximately 0.81, compared to the lowest quality dataset, xP3, which had an average approval ratio of approximately 0.50. This aligns with our intuition that carefully curated datasets lead to high-quality annotations as perceived by human annotators. Figure 16 provides a summary of the results for each group. Figure 23 in the Appendix provides approval ratios per datasets in each group.

1.0

0.81%

AverageApprovalRatio

0.8

0.70%

0.66%

0.6

0.50%

0.4

0.2

0.0

xP3 Aya Collection (Templates)

Aya Collection (Translations)

Aya Original Annotations

- Figure 16: Average approval ratio per dataset group, constrained to datasets receiving at least 20 votes.

##### 6 Aya Evaluation Suite

Lastly, as part of the Aya project we curate and release an evaluation suite tailored for multilingual models. This set is a valuable contribution in tackling the scarcity of multilingual data, a challenge that becomes even more apparent when considering evaluation sets. While there are several test sets available for evaluating multilingual models [Conneau et al., 2018; Ponti et al., 2020; Lin et al.,

- 2022], they focus primarily on discriminative tasks. To evaluate multilingual models’ generations, the literature includes task-specific evaluation sets such as Translation [Goyal et al., 2021b], Summarization [Hasan et al., 2021] and Question Answering [Clark et al., 2020]. However there is currently a gap in evaluating open-ended generation capabilities of LLMs within a multilingual context. We aim to address this gap by curating a multilingual evaluation set tailored for assessing the openended generation capabilities of LLMs, such as brainstorming, planning, and other unstructured, long-form responses.

To strike a balance between language coverage and the quality that comes with human attention, we create an evaluation suite that includes (1) human-curated examples in a limited set of languages, (2) automatically translations of handpicked examples into a more extensive number of languages, and (3) human-post-edited translations into a small number of languages. We consider two primary sources of data: original annotations from Aya dataset (comprising new examples culturally curated for different languages) and Dolly prompts (high-quality, human-written examples carefully selected to have a universal reach). The subsets comprising the Aya evaluation suite are:

aya-human-annotated test set For ease of future adoption, we have partitioned the Aya dataset into training and testing splits. The test set of the Aya Dataset contains 1,750 of the total instances (250 instances from 7 languages), selected at random from original annotations. Our goal is to achieve a balanced representation of languages in the test set and ensure a sufficient number of examples per language. To guarantee enough data remains for training, we focused on languages with at least 2000 original annotations. In order to ensure linguistic diversity, we included languages that were varied in terms of high, mid, or low-resourcedness, as well as script and language families. For those reasons, the test set consists of English (high-resource, Latin script, Indo-European), Portuguese (mid-resource, Latin script, Indo-European), Simplified Chinese

(high-resource, Han, Sino-Tibetan), Standard Arabic (high-resource, Arabic script, Afro-Asiatic), Telugu (low-resource, Telugu script, Dravidian), Turkish (mid-resource, Latin script, Turkic), and Yoruba (low-resource, Latin script, Atlantic-Congo). See Table 5 for more details.

dolly-machine-translated test set We separate a curated subset of 200 Dolly prompts [Conover

et al., 2023] to serve as an additional translated evaluation set. Our aim with this selection was to exclude any culturally or geographically specific prompts and completions. Hence, two reviewers inspected a set of initially 500 English prompts that were uniformly sampled based on the task categories in Dolly. The reviewers excluded prompts that rely on geographic knowledge such as “Looking at cities in Australia that are on the east coast and the west coast of the country, which coast are the cities of Fremantle, Sydney, Brisbane, Perth, Cairns, Townsville, Newcastle located on?”, or prompts such as “Why is NFL football called football when players use their hands mainly?” that rely on overly specific cultural references. When two reviewers disagreed, a third reviewer was asked to break the tie. We kept prompts such as “Is art useless?” or “Write a short paragraph about why you should not have both a pet cat and a pet bird.” and questions that refer to geographic specific knowledge where the supporting evidence was provided in the prompt itself e.g., “Given a reference text about Minister for Food, Agriculture and Fisheries of Denmark, when was the position created and was was it named?”. Although not perfect, the intention behind this selection was to gather a test set that allows us to evaluate the fluency and quality of responses in various languages while avoiding model assessment on prompts tied to specific cultural or geographic references that might have language-dependent validity. We automatically translate the prompts with NLLB into 101 languages and their dialects that are captured by NLLB. Including the original English prompts this dataset covers 115 dialects.

dolly-human-edited test set The automatic translation process may introduce errors in the prompts that render them nonsensical. For example, the prompt “Which is a species of fish? Bleak or Weary” requires domain expertise to choose the right translation of the fish names rather than literal translations of the adjectives (as e.g. in the NLLB Translation into Spanish: “Desanimado o cansado.” (=“discouraged or tired”)). If the prompt does not make any sense, there is no clear expectation and measurement of what a good and correct completion should look like. To confidently interpret evaluation results, it is imperative to establish a reliable set of prompts for evaluation. To enhance the reliability of testing on these prompts, we therefore enlist professional human annotators to post-edit the examples (e.g. for the example above “Alburno o Cansado” (=“[Fish name] or Tired”). We post-edit the prompts for a subset of six languages: Arabic, Hindi, Spanish, French, Serbian and Russian. Appendix F describes the post-editing process and effort in more detail. The example above illustrates that some prompts, even when translated correctly, might still not transfer well into other languages—which is the main difference between a translated English-centric set like this and an evaluation set originally written in each target language like aya-human-annotated.

We open-source the dolly-machine-translated test set to be an additional resource for researchers, although warn that the expressiveness of a translated evaluation set is limited by the quality of the translation model (and human post-edit) and may adversely impact an estimate of ability in languages where translations are not adequate [Nogara et al., 2023]. Ultimately, this is a compromise between having evaluation coverage in a more complete set of languages (101 languages and 114 dialects in total) versus having human-annotated evaluation sets. If using the automatically translated test set, we recommend it be paired and reported with the professionally post-edited dolly-human-edited for 6 languages, or the aya-humanannotated set which also only covers 7 languages but is entirely created by proficient

target language speakers.

##### 7 A Participatory Approach to Research

Recent breakthroughs in NLP have predominantly come from narrow collaborations that involve researchers from a handful of institutions and regions of the world [Nakamura et al., 2023]. This reliance on small, specialized collaboration networks has been shown to hinder innovation [Park et al., 2023]. Dataset creation as a process has often been undervalued, with minimization of the value of creators’ contributions [Andress et al., 2020; Peng et al., 2021; Hanley et al., 2020]. Under such conditions, the richness and diversity of the data are often compromised, as it reflects a limited perspective that aligns with the interests of those who wield greater power in these transactions. Data is not, as metaphors such as ‘data mining’ [Puschmann & Burgess, 2014], or ‘data is the new oil’ [Stark & Hoffmann, 2019; Awati & Shum, 2015], might suggest, a natural resource waiting to be exploited. Whenever we engage with data, we are also engaging with the connections that data has to the people who produce, prepare, and distribute it [Seaver, 2021; Pinel C, 2020; Crawford,

- 2021]. Participatory approaches in AI design and research are one way to address gaps in access to resources needed for research: through collaborative partnerships with language speakers and local communities.

Aya is an example of a participatory research project [Birhane et al., 2022; Corbett et al., 2023; Delgado et al., 2023]. Here, the research is the result of a broad cross-institutional, global collaboration. This type of cross-sectional work facilitates the collection of vital linguistic data and community engagement, which is crucial for developing effective language technologies [Joshi et al.,

- 2019; ∀ et al., 2020]. We describe below some of the guiding principles we followed throughout the year-long Aya project.

Fluid Ownership and Growth Our open science framework allowed us to challenge the norms of how computer science usually proceeds [Wittenburg, 2021; Sabou et al., 2012]. Traditional research approaches often involve rigid hierarchies; typically, research is conducted within academic institutions or corporate labs where roles are clearly defined, and collaboration is mostly synchronous, relying on in-person meetings or real-time communication. In contrast, Aya took a decentralized and democratic approach to collaboration, supporting fluid leadership and flexible role adoption. This empowered members to take initiative and lead in areas where they had passion or expertise, regardless of their position in academia, or when they became involved in the project. For example, members became Language Ambassadors at many different points during the year-long project, and mentorship roles evolved naturally with more experienced researchers providing guidance to those more junior (see Appendix C for more details of different roles in the project).

Organizational Structure The communication channels and organizational structure of Aya were designed to facilitate rich collaboration that could evolve with the interests of participating researchers over the year-long project. For example, most communication between independent researchers involved within Aya was asynchronous over Discord, which allowed researchers in different time zones to participate in discussions. Monthly meetings were open for anyone to attend and were recorded for asynchronous viewing. We describe the structure of meetings and communication more thoroughly in Appendix D.1 and D.2.

Inclusion and Access The open nature of the Aya UI allowed us to bypass the gate-keeping

mechanisms of academic science that often marginalize non-English speakers and people without formal academic credentials [West et al., 2020]. Expertise in the command of a spoken or written language is clearly distinct from expertise in machine learning. The inclusion of such a wide range of volunteers gave us more representative data in a wide variety of languages and also gave volunteers a glimpse into the often obscure world of machine learning.

Who Participated in Aya The motivations of contributors were not based on financial remuneration but on ideals of community, identity, and social justice. Participants saw their roles as Language Ambassadors and contributors as crucial to ensuring the inclusion of their languages in the ongoing transition to a digital, information-driven economy. The Language Ambassador for Malagasy, a language-driven to the risk of extinction by colonial French rule in Madagascar [Spolsky, 2018], is planning hackathons in 2024 to use the Aya Dataset to create voice-to-text apps that will help non-literate speakers of Malagasy participate in the modern economy. In Telugu, a traditional genre of poetry known as Sathakam is an integral part of the educational system. However, chatbots that can translate text into Telugu have little to no understanding of the Sathakam form. The Telugu Language Ambassador told a newspaper in Toronto that “in Aya, we made sure to include as many Sathakams as we could find” [Castaldo, 2023].

These motivations are not peripheral to the strength of the final Aya Dataset but are key factors in the data’s provenance [Loukissas, 2019]. These qualitative dimensions remind us that language is, for the people who use it every day, an intimately social phenomenon. Beyond the symbolic notation that connects tokens to referents in the real world, we find a robust network of social relations that are necessary for languages to flourish [Sidnell & Enfield, 2012; Goodwin, 2017; Agha, 2006]. The social interactions between contributors, ML researchers, and social scientists in the Aya project were crucial to its success. Contributors shared playlists of their favorite songs from their home country, recipes from their childhood, and snapshots of the views from their home offices. They debated subtle nuances of how they wanted their language represented in the dataset and pushed back on some of the assumptions made by project coordinators on what constituted a distinct language as opposed to a regional dialect (see Section 9). More than one contributor sat down with their grandparents to contribute to a language that spanned three generations of use.

The realities of the conditions under which many people work and live were present every day. For example, Zoom meetings were cut short for some volunteers due to power outages in their countries or lack of access to a stable internet connection. Burmese, a language spoken in Myanmar, started out strong in the project with a group of 35 motivated volunteers but saw a sudden pause in contributions as civil war broke out in the country resulting in the withdrawal of the volunteers from the project [Petty, 2023]. The Language Ambassador for Armenian also had to drop out of the project because of a conflict in that country [Reuters, 2023]. In some countries, postal services only functioned a few days per month because of ongoing warfare, creating challenges for organizers when mailing out Aya gifts to thank committed volunteers. Ultimately, organizers were not able to send gifts to thank volunteers who participated from Somalia, Yemen and Palestine. For Somalia and Yemen, both Canada Post, DHL and Fedex where not able to support shipments. For Palestine, the cost of shipment proved to be prohibitively expensive – with an estimated shipping cost of 294 US dollars per t-shirt. These geo-political realities shaped both our contributors’ experience as well as the progress of the project.

Including these factors in our post-mortem analysis of the project is crucial to understanding both the motivation of people willing to volunteer for open-science projects, and also to understanding

###### the data itself: its breadth, its provenance, its shortcomings, and its living history.

ISO Code Language Script Family Subgrouping Resources Included

ace Achinese Arabic/Latin Austronesian Malayo-Polynesian Low ♤♠ afr Afrikaans Latin Indo-European Germanic Mid ♤♠ amh Amharic Ge’ez Afro-Asiatic Semitic Low ♦♢ ♤♠ ara Arabic Arabic Afro-Asiatic Semitic High ♦♢ ♤♠ aze Azerbaijani Arabic/Latin Turkic Common Turkic Low ♤♠ ban Balinese Latin Austronesian Malayo-Polynesian Low ♤♠ bbc Toba Batak Latin Austronesian Malayo-Polynesian Low ♤♠ bel Belarusian Cyrillic Indo-European Balto-Slavic Mid ♤♠

- bem Bemba Latin Niger-Congo Atlantic-Congo Low ♤♠
- ben Bengali Bengali Indo-European Indo-Aryan Mid ♦♢ ♤♠ bjn Banjar Arabic/Latin Austronesian Malayo-Polynesian Low ♤♠ bul Bulgarian Cyrillic Indo-European Balto-Slavic Mid ♤♠ cat Catalan Latin Indo-European Italic High ♤♠ ceb Cebuano Latin Austronesian Malayo-Polynesian Mid ♦♢ ♤♠ ces Czech Latin Indo-European Balto-Slavic High ♤♠ cym Welsh Latin Indo-European Celtic Low ♤♠ dan Danish Latin Indo-European Germanic Mid ♦♢ ♤♠ deu German Latin Indo-European Germanic High ♦♢ ♤♠ ell Greek Greek Indo-European Graeco-Phrygian Mid ♦♢ ♤♠ eng English Latin Indo-European Germanic High ♦♢ ♤♠ epo Esperanto Latin Constructed Esperantic Low ♤♠ est Estonian Latin Uralic Finnic Med ♤♠ eus Basque Latin Basque - High ♦♢ ♤♠ fil Filipino Latin Austronesian Malayo-Polynesian Mid ♦♢ ♤♠ fin Finnish Latin Uralic Finnic Mid ♦♢ ♤♠ fon Fon Latin Niger-Congo Atlantic-Congo Low ♤♠ fra French Latin Indo-European Italic High ♦♢ ♤♠ gla Scottish Gaelic Latin Indo-European Celtic Low ♤♠ gle Irish Latin Indo-European Celtic Low ♦♢ ♤♠ glg Galician Latin Indo-European Italic Med ♤♠ guj Gujarati Gujarati Indo-European Indo-Aryan Low ♦♢ ♤♠

- hat Haitian Creole Latin Indo-European Italic Low ♦♢ ♤♠
- hau Hausa Latin Afro-Asiatic Chadic Low ♦♢ ♤♠ heb Hebrew Hebrew Afro-Asiatic Semitic Mid ♤♠ hin Hindi Devanagari Indo-European Indo-Aryan High ♦♢ ♤♠ hrv Croatian Latin Indo-European Balto-Slavic. High ♤♠ hun Hungarian Latin Uralic - High ♦♢ ♤♠ hye Armenian Armenian Indo-European Armenic Low ♤♠ ibo Igbo Latin Atlantic-Congo Benue-Congo Low ♦♢ ♤♠ ind Indonesian Latin Austronesian Malayo-Polynesian Mid ♦♢ ♤♠ isl Icelandic Latin Indo-European Germanic Low ♤♠ ita Italian Latin Indo-European Italic High ♦♢ ♤♠ jav Javanese Latin Austronesian Malayo-Polynesian Low ♦♢ ♤♠ jpn Japanese Japanese Japonic Japanesic High ♦♢ ♤♠ kan Kannada Kannada Dravidian South Dravidian Low ♦♢ ♤♠

- kas Kashmiri Arabic Indo-European Indo-Aryan Low ♤♠
- kat Georgian Georgian Kartvelian Georgian-Zan Mid ♤♠
- kau Kanuri Arabic/Latin Saharan Western Saharan Low ♤♠ kaz Kazakh Cyrillic Turkic Common Turkic Mid ♤♠ khm Khmer Khmer Austroasiatic Khmeric Low ♤♠ kin Kinyarwanda Latin Niger-Congo Atlantic-Congo Low ♤♠ kir Kyrgyz Cyrillic Turkic Common Turkic Low ♦♢ ♤♠ kor Korean Hangul Koreanic Korean Mid ♦♢ ♤♠ kur Kurdish Latin Indo-European Iranian Low ♦♢ ♤♠ lao Lao Lao Tai-Kadai Kam-Tai Low ♤♠ lav Latvian Latin Indo-European Balto-Slavic Mid ♤♠ lij Ligurian Latin Indo-European Italic Low ♤♠ lit Lithuanian Latin Indo-European Balto-Slavic Mid ♦♢ ♤♠ ltz Luxembourgish Latin Indo-European Germanic Low ♤♠ mad Madurese Latin Austronesian Malayo-Polynesian Low ♤♠ mal Malayalam Malayalam Dravidian South Dravidian Low ♦♢ ♤♠ man Manipuri Bengali Sino-Tibetan Kuki-Chin-Naga Low ♤♠

mar Marathi Devanagari Indo-European Indo-Aryan Low ♦♢ ♤♠ min Minangkabau Latin Austronesian Malayo-Polynesian Low ♤♠ mkd Macedonian Cyrillic Indo-European Balto-Slavic Low ♤♠ mlg Malagasy Latin Austronesian Malayo-Polynesian Low ♦♢ ♤♠ mlt Maltese Latin Afro-Asiatic Semitic Low ♤♠ mon Mongolian Cyrillic Mongolic-Khitan Mongolic Low ♤♠ mri Maori Latin Austronesian Malayo-Polynesian Low ♤♠ msa Malay Latin Austronesian Malayo-Polynesian Mid ♦♢ ♤♠ mya Burmese Myanmar Sino-Tibetan Burmo-Qiangic Low ♦♢ ♤♠ nep Nepali Devanagari Indo-European Indo-Aryan Low ♦♢ ♤♠ nij Ngaju Latin Austronesian Malayo-Polynesian Low ♤♠ nld Dutch Latin Indo-European Germanic High ♦♢ ♤♠ nor Norwegian Latin Indo-European Germanic Low ♤♠ nso Northern Sotho Latin Atlantic-Congo Benue-Congo Low ♦♢ ♤♠ nya Chichewa Latin Atlantic-Congo Benue-Congo Low ♦♢

pan Punjabi Gurmukhi Indo-European Indo-Aryan Low ♦♢ ♤♠ pes Persian Arabic Indo-European Iranian High ♦♢ ♤♠ pol Polish Latin Indo-European Balto-Slavic High ♦♢ ♤♠ por Portuguese Latin Indo-European Italic High ♦♢ ♤♠ pus Pashto Arabic Indo-European Iranian Low ♦♢ ♤♠ ron Romanian Latin Indo-European Italic Mid ♤♠ rus Russian Cyrillic Indo-European Balto-Slavic High ♦♢ ♤♠ sin Sinhala Sinhala Indo-European Indo-Aryan Low ♦♢ ♤♠ slk Slovak Latin Indo-European Balto-Slavic Mid ♤♠ slv Slovenian Latin Indo-European Balto-Slavic Mid ♤♠ smo Samoan Latin Austronesian Malayo-Polynesian Low ♤♠ sna Shona Latin Indo-European Indo-Aryan Low ♦♢ ♤♠ snd Sindhi Arabic Indo-European Indo-Aryan Low ♦♢ ♤♠ som Somali Latin Afro-Asiatic Cushitic Low ♦♢ ♤♠ sot Southern Sotho Latin Atlantic-Congo Benue-Congo Low ♤♠ spa Spanish Latin Indo-European Italic High ♦♢ ♤♠ sqi Albanian Latin Indo-European Albanian Low ♦♢ ♤♠ srp Serbian Cyrillic Indo-European Balto-Slavic High ♦♢ ♤♠ sun Sundanese Latin Austronesian Malayo-Polynesian Low ♦♢ ♤♠ swa Swahili Latin Atlantic-Congo Benue-Congo Low ♦♢ ♤♠ swe Swedish Latin Indo-European Germanic High ♦♢ ♤♠ tam Tamil Tamil Dravidian South Dravidian Mid ♦♢ ♤♠ taq Tamasheq Latin/Tifinagh Afro-Asiatic Berber Low ♤♠ tel Telugu Telugu Dravidian South Dravidian Low ♦♢ ♤♠ tgk Tajik Cyrillic Indo-European Iranian Low ♤♠ tha Thai Thai Tai-Kadai Kam-Tai Mid ♦♢ ♤♠ tur Turkish Latin Turkic Common Turkic High ♦♢ ♤♠ twi Twi Latin Niger-Congo Atlantic-Congo Low ♤♠ ukr Ukrainian Cyrillic Indo-European Balto-Slavic Mid ♦♢ ♤♠ urd Urdu Arabic Indo-European Indo-Aryan Mid ♦♢ ♤♠ uzb Uzbek Latin Turkic Common Turkic Med ♤♠ vie Vietnamese Latin Austroasiatic Vietic High ♦♢ ♤♠ wol Wolof Latin Atlantic-Congo North-Central Atlantic Low ♦♢ ♤♠ xho Xhosa Latin Atlantic-Congo Benue-Congo Low ♦♢ ♤♠ yid Yiddish Hebrew Indo-European Germanic Low ♤♠ yor Yorùbá Latin Atlantic-Congo Benue-Congo Low ♦♢ ♤♠ zho Chinese Han Sino-Tibetan Sinitic High ♦♢ ♤♠ zul Zulu Latin Atlantic-Congo Benue-Congo Low ♦♢ ♤♠

Table 5: 65 languages in the Aya Dataset and 114 languages in the Aya Collection, each language’s corresponding script, family, subgrouping, and if it is classified as “lower-”, “mid-” or “higher”resourced according to the taxonomy classes by [Joshi et al., 2020] (low: [0, 1, 2], mid: [3], high: [4, 5]). The language is either included in the Aya Dataset (♦♢), Aya Collection (♤♠), or both. Note that Ngaju (nij) and Toba Batak (bbc) are not listed in [Joshi et al., 2020].

##### 8 Related Work

###### 8.1 Multilingual datasets

Low-resource languages have long been a challenge in NLP, with limited data impacting task performance [Kunchukuttan et al., 2021]. To address this, researchers have explored techniques like data augmentation [Sennrich et al., 2016; Dhole et al., 2021], transfer learning [Zoph et al., 2016], repeating [Luukkonen et al., 2023; Muennighoff et al., 2023b], and multilingual models [Dabre et al.,

- 2020; Muennighoff et al., 2023c; Yong et al., 2023b], achieving promising results in areas like machine translation. Here, we focus on efforts that are centered on multilingual dataset creation.

Several works have created large-scale multilingual corpora. These are often unstructured texts, ideal for large-scale unsupervised pre-training [Abadji et al., 2021; Ortiz Su’arez et al., 2019; Scao

- et al., 2022a;b; Laurençon et al., 2022; Kudugunta et al., 2023; Whitehouse et al., 2023]. Another group of multilingual datasets is focused on machine translation [Lucia Specia et al., 2010; Fan et al., 2021]. They consist of parallel texts in two or more languages, enabling models to learn the mappings between them. Ideally, machine translation datasets encompass diverse domains and language pairs, from commonly spoken languages to resource-scarce ones, promoting inclusivity and linguistic diversity. One of the most extensive collections of parallel corpora is available at the OPUS project website14 [Tiedemann, 2012]. Large capacity models for language understanding may obtain strong performance on high-resource languages while greatly improving low-resource languages [Goyal et al., 2021a]. In Whitehouse et al. [2023], the effectiveness of LLM-powered data augmentation in cross-lingual commonsense reasoning was demonstrated. An improved performance was shown when smaller cross-lingual models were finetuned with data generated by LLMs. Some recently released datasets focus on specialized language domains such as law [Niklaus et al., 2023], education [Zhang et al., 2023c], or healthcare [Wang et al., 2023]. These corpora often suffer from inadequate data quality and require extensive cleaning [Abadji

- et al., 2022; Kreutzer et al., 2022]. Task-specific datasets, such as XCOPA [Ponti et al., 2020] or XNLI [Conneau et al., 2018], are smaller in scale but offer higher quality data targeted at a specific model capability such as cross-lingual understanding and transfer learning. This type of data is crucial for evaluating and enhancing the performance of models in diverse linguistic contexts.

No Language Left Behind [NLLB-Team et al., 2022] open-sourced bitext, mined bitext, and data generated using back-translation in 200+ languages specifically for text-to-text translation. While Seamless4MT [Barrault et al., 2023] released the metadata of SeamlessAlign, an open multimodal translation dataset, there are relatively fewer works for data creation/curation in low-resource languages. Cahyawijaya et al. [2023] introduced NusaCrowd, a standardized collection of 137 datasets covering 19 Indonesian local languages in text, speech, and image modalities. Our work differs from previous datasets as we create a large-scale instruction-tuning dataset spanning hundreds of different tasks, yet retain high-quality by involving human annotation and rigorous quality control across the entire data creation process.

- 14https://opus.nlpl.eu

###### 8.2 Instruction-tuning datasets

Instruction-tuning datasets are collections of human-curated instructions and response pairs, templatized NLP tasks, or synthetic instructions generated by a language model. There are a growing number of NLP meta-datasets such as Natural instructions [Mishra et al., 2022], SuperNatural Instructions[Wang et al., 2022d], Flan 2021 [Wei et al., 2022a], Flan 2022 [Longpre et al., 2023a], Public Pool of Prompts (P3) [Sanh et al., 2022], Unnatural Instructions [Honovich et al., 2023], OPT-IML [Iyer et al., 2022], inter alia [Khashabi et al., 2020; Ye et al., 2021; Min et al., 2021] that collate numerous instruction finetuned datasets together. Some work focuses on specific applications such as dialogue [Köpf et al., 2023], structured knowledge grounding [Xie et al., 2022], or chain-of-thought reasoning [Wei et al., 2022b; Kim et al., 2023]. Manual efforts include Open Assistant [Köpf et al., 2023] crowd-sourcing volunteers who wrote both instructions and responses, Databricks employees creating 15k examples in Dolly [Conover et al., 2023], and LIMA [Zhou et al., 2023] which is a collection of 1,000 author-curated IFT examples.

Synthetic instruction-tuning datasets comprise instructions sampled from a language model, such as the Self-Instruct dataset [Wang et al., 2022b] generated by GPT-3 [Brown et al., 2020], the Alpaca dataset [Taori et al., 2023] generated by GPT-3.5, and the Guanaco dataset [Joseph Cheung, 2023]. Increasingly, the synthetic generation of instruction-finetuned datasets is more sophisticated. [Xu

- et al., 2023a] propose a novel Evol-Instruct framework to obtain complex and difficult instructions gradually. [Luo et al., 2023] and [Gunasekar et al., 2023] further expand this idea to promote reasoning, code generation, and algorithmic skills. InstructionWild [Ni et al., 2023] and ShareGPT15 are collections of user-shared conversations with ChatGPT.

###### 8.3 Multilingual Instruction-Tuning Datasets

Despite ever-larger collections of IFT datasets, prior work has been largely English-centric. Most approaches to extend instruction finetuned datasets outside of English have relied on 1) translating English datasets into other languages [Holmström & Doostmohammadi, 2023; Li et al., 2023a; Winata et al., 2023b], 2) template based dataset creation [Yu et al., 2023; Gupta et al., 2023] or 3) human curating instruction datasets in languages outside of English [Muennighoff et al., 2023c; Li

- et al., 2023c; Wang et al., 2022c]. There have been some notable exceptions with large proportions of non-English data [Joseph Cheung, 2023; Köpf et al., 2023; Lai et al., 2023; Li et al., 2023a; Longpre et al., 2023a; Muennighoff et al., 2023a;c; Zhuo et al., 2024; Nguyen et al., 2023].

Template-Based Datasets. The most relevant effort is recent work by [Muennighoff et al., 2023c] releasing Crosslingual Public Pool of Prompts (xP3). xP3 expands the P3 taxonomy and adds 28 new multilingual datasets. However, their datasets usually use the same template in different languages, thus limiting task diversity. For example, a random batch from their dataset may include the same sample in different languages multiple times. Their xP3 corpus has task instructions exclusively in English. In [Muennighoff et al., 2023c], the experiments with matching the task instruction to the respective language of the sample via machine translation (xP3mt) showed slightly improved performance for non-English task instructions at inference. Our work is distinct in that our human-curated constructed dataset is unique for each of the 65 languages. Such diversity has been emphasized as a key ingredient for instruction tuning [Longpre et al., 2023a]. Further, we create non-English task instructions via human annotators, ensuring these are of high-quality, which is

- 15https://sharegpt.com/

another pillar of a good performance [Zhou et al., 2023].

Machine Translated Datasets. Machine-translated prompts often lack variability and the cultural nuance inherent in natively written text. However, they are still useful for expanding the language coverage of the training data and can help bridge the resource gap for languages with limited training data [Urbizu et al., 2023; Lin et al., 2022]. They can also adapt already-trained instruction-tuned language models to follow instructions in new languages [Yong et al., 2023b]. Furthermore, LLMs trained on designed prompts have also been shown to be successful at tasks like EAE (Event Argument Extraction) from multilingual data in a zero-shot setup [Huang et al.,

- 2022]. [Zhang et al., 2023a] constructed high-quality Chinese instructions from existing English instruction datasets. They first translated the English instructions into Chinese and then used a human verification process to determine whether these translations were usable; the verified dataset set consists of around 200k Chinese instruction-tuning samples. [Li et al., 2023a] constructed instruction data for 52 popular languages using Google Translate to translate English prompts and completions from Alpaca [Taori et al., 2023] (52K) and Dolly [Conover et al., 2023] (15K) dataset, then used this data to finetune LLaMA [Touvron et al., 2023] using the LoRA [Hu et al., 2021] technology. [Zhang et al., 2023b] prompted LLMs to translate a task request, which was overlaid with the more granular user-based corrects. This process naturally connects different languages as well as human preferences with LLMs, leveraging LLaMA [Touvron et al., 2023] for foundational support and employing automatic construction of interactive translation instructions for instructional tuning, thereby enhancing the model’s multilingual capability and alignment with diverse linguistic needs.

Human-Curated Multilingual Examples. Most relevant to our work on the Aya dataset are other datasets that have been curated by humans, often in English. Databricks collected a 15k instruction dataset databricks-dolly-15k by relying on its employees as annotators [Conover et al., 2023]. Annotators were instructed to curate prompt / response pairs in each of eight different instruction categories. [Köpf et al., 2023] released the OpenAssistant corpus with over 10,000 dialogues from more than 13,500 international annotators. While this dataset contains multilingual annotations, this was not an explicit goal of the initiative. In contrast to our corpus which only has 2.05% contributions in English, 42.8% of the OpenAssistant project remains in English [Köpf et al.,

- 2023].

###### 8.4 Participatory Research in Machine Learning If you want to go fast go alone; if you want to go far, go together. — African Proverb

Prior participatory research initiatives have centered around regions or specific tasks like translation or character recognition. For example, [Clanuwat et al., 2018] tackles the problem of reading and understanding Kuzushiji, a cursive style of Japanese writing no longer in common use. Another example of culturally diverse data collection is [Liu et al., 2021], which recruited native speakers from five languages (Indonesian, Swahili, Tamil, Turkish, and Mandarin Chinese) that are typologically, genealogically, and geographically diverse, to provide images of concepts that are representative of their cultures. Then, they recruited native-speaking professional linguists to write captions for these images. However, this dataset is small (less than 8,000 data points) and thus limited to evaluation only. It is worth noting that these works are solely focused on the image domain, unlike our work, which concentrates on text.

More relevant to our work are participatory data creation initiatives focused on NLP. [GuevaraRukoz et al., 2020] presents a study focusing on the creation of a crowd-sourced corpus for Latin American Spanish dialects to address the scarcity of resources for these languages. [∀ et al., 2020] focuses on the task of Machine Translation (MT), and curates a dataset in 30 under-represented African languages according to a participatory research framework. Our work is very much in the spirit of these prior efforts, with differences in terms of global rather than regional focus. In contrast to these works, which have a specific regional focus, Aya collaborators came from multiple continents covering a diverse range of languages.

Several works have explored the organizational structures required to facilitate the development of research communities around under-represented languages. [Siminyu et al., 2021] details work on the AI4D - African Language Program, which aimed to enhance language resources for African languages. The outcome included creating over nine open-source African language datasets and establishing baseline models, demonstrating the program’s significant impact on language technology for African languages. [Azunre et al., 2021] describes the establishment of NLP Ghana, with its collaborative open-source community. [Strassel & Tracey, 2016] discusses the challenges of developing resources for low-resource languages under the LORELEI (Low Resource Languages for Emergent Incidents) program. They focus on the pressing need for digital resources in these languages, particularly in critical situations such as mitigating the effects of natural disasters.

Open science community initiatives like Aya yield significant advancements in language modeling. Related efforts (in terms of compute and resources required) can be found in the BigScience Workshop [Akiki et al., 2022], which began in 2021. The BigScience project was initiated to address the limitations in LLM development, emphasizing open science and inclusive collaboration. Leveraging open science principles, it united a global network of researchers working to collaboratively and ethically enhance machine learning. Their work culminated in key developments like the BLOOM model [Scao et al., 2022a] and ROOTS corpus [Laurençon et al., 2022]. These achievements underscore the value of community-driven, ethical, and diverse research programs for large-scale language technologies. Following Big Science, there have been other recent efforts on open science in language modeling [Groeneveld et al., 2024; Soldaini et al., 2024].

##### 9 Limitations of our work

- 1. Language and dialect coverage: The Aya Dataset and Aya Collection cover 65 and 114 languages respectively—significantly more than existing multilingual datasets. However, this is still only a tiny fraction of the world’s linguistic diversity. Of the world’s approximately 7,000 languages, only half of them are captured in any sort of written form [Adda et al., 2016]. Of this half, only a few hundred are included on the internet in machine readable corpora [Adda et al., 2016]. This means that 93% of the world’s languages are still not being used to train LLMs. It is also notoriously difficult to determine the dividing line between different languages and different dialects of the same language [Rooy, 2021]. Geo-cultural variation within a language often gives rise to new dialects or creoles over time [Zampieri et al., 2020; Wolfram, 1997; Brown et al., 2020; Lent et al., 2022; Blaschke et al., 2023] and, as such, dialects can serve an important function in establishing and maintaining cultural identity [Falck et al., 2012]. Many different dialects that are generally recognized as belonging to a single parent language are not represented in the dataset. For example, in the case of Malay, one of the largest Southeast Asian languages in the dataset, there are no contributions for regional

- dialects that are widely spoken in certain states of Malaysia. Contributions by volunteers who wished to self-identify as speaking a particular dialect were tagged as such in the data to allow for limited analysis of the use of regional dialects in annotations. Lastly, socio-linguistic data show that multilingual speakers often ‘code-switch’ between languages or dialects depending on context [Myers-Scotton, 2017], but in this project, we kept the languages isolated to make them easier to classify and to be used downstream for language-specific applications. Aya also does not cover programming languages. There has been prior work on covering diverse programming languages [Li et al., 2023d; Allal et al., 2023] and we leave further explorations in this direction to future work.
- 2. Uneven distribution of contributions: As explored in Section 3.3, despite the large number of participants, the activity of annotators was skewed, with a ‘long tail’ of annotators only contributing one or two annotations. Relatively few contributors accounted for the most annotations (see Figure 10 - bottom). Similarly, there is a huge gap between languages with the highest number of contributions and ones with the lowest number of contributions. Consequently, this suggests potential unevenness in dataset distributions across different languages and a lack of annotator diversity within some languages dominated by one or two frequent contributors.
- 3. Cultural or personal bias: Another limitation is the presence of annotations with particular cultural biases. Some languages in our dataset have limited representation, with only a few annotators responsible for annotating the bulk of their dataset. This might mean that data for a particular language is dominated by annotations that represent the opinions or priorities of a particular contributor or could represent a narrow selection of cultural viewpoints. For example, annotations in French might contain many examples about the history of France, its food, songs, and other cultural practices, but not contain much information about the cultural heritage of French-speaking communities in Québec, Togo, or Senegal [Vigouroux, 2013]. This bias is particularly problematic given the skewed distribution of the most active annotators. There is also a potential bias in the availability of particular kinds of content. For example, it is easier to find online text from news sites for many African languages than it is to find text from other domains. Accordingly, these datasets will be skewed towards the grammar and lexicon used in news reports instead of the kind of natural language people use in everyday life [Hovy & Prabhumoye, 2021].
- 4. Gendered pronouns: Many of the languages in the Aya Dataset only contain pronouns that are explicitly gendered (e.g., Arabic) or that lack gender-neutral third-person pronouns for gender-neutral reference (e.g. Estonian). This means that in responding to prompts that might not specify a gender, care needs to be taken to ensure that responses remain neutral as to the gender of any assumed participants [Ghosh & Caliskan, 2023]. For example, if a response requires reference to “a teacher” in French, the annotator would need to include references to both “un/e enseignant/e”. While care was taken to ensure neutral responses for new annotations, gendered annotations in existing datasets might not have been flagged, as they are not, strictly speaking, incorrect. Instead, they merely presuppose a gendered reading where one might not be implied [Hardmeier & Guillou, 2018].
- 5. Formality distinctions: Many of the languages in the Aya Dataset also require the speaker or annotator to make situational choices as to the formality of the pronoun used in response to a particular prompt. Languages such as Japanese, Persian, Indonesian, Javanese, Yoruba, French, Spanish, and German include different levels of honorifics that are used

in formal or informal settings, or used between community members who differ in status (determined by a variety of factors such as age, profession, seniority, or ethnicity) [Brown & Gilman, 1968]. In Yoruba, for example, the pronoun that roughly translates as "they" can either be used as a singular honorific or as a third-person plural pronoun [Yusuf, 2022]. We deferred to the individual annotators in crafting their responses, allowing them to rely on the norms of their particular speech community to determine how to respond. Often, these decisions hinged on the content being discussed, or on how formally the prompt was crafted in the original data set. When in doubt, annotators were asked to imagine what kind of ‘voice’ they would expect an LLM to have when answering a given prompt [Wilson, 2023].

This means our released dataset contains many languages that have varying levels of standardization and differing style guidelines. Standardization is often deeply intertwined with power and identity, and the manner of speech may be connected to aspects of identity like age, education level, tribal affiliation, and religion. The lack of standardization is also largely due to regional and cultural differences across the same language, exemplified by Portuguese in the dataset: European Portuguese diverges from Brazilian Portuguese not only in formality but also in grammar, spelling, and vocabulary. Often, standards are projected by others to ensure adherence to cultural values [Bourdieu, 1987; De Mauro et al., 2015; Haugen, 1959; Rickford et al., 2012].

- 6. Toxic or offensive speech: The Aya Annotation Platform did not contain specific flags for toxic, harmful, or offensive speech, so it is possible that malicious users could submit unsafe data. We believe this is of relatively low risk because of the high rate of human-verified annotations and peer-review, making it unlikely that toxic prompts or completions made it into the final dataset. However, there is no guarantee that every entry was audited. While data poisoning has rarely been observed as a viable threat in practice, it has been demonstrated to be of concern for instruction-tuning with very few examples [Xu et al., 2023b; Wan et al., 2023] and for pre-training under realistic conditions [Carlini et al., 2023]. During the eight months of crowd-sourced annotating, there were no reported cases of hateful or toxic speech in the existing datasets nor were there any instances of offensive speech reported in the peerreviewing phase of new annotations.

We also note that data that might be offensive to one annotator might not be offensive to another, for instance, the completion of a prompt that asks for a definition of the word “woke” [Castaldo, September 16, 2023]. Prompts written on partisan political topics, or the inclusion of political advertisements or campaign messages could cause offense depending on the political proclivities of the annotator. In short, we tried to mitigate offensive speech by relying heavily on human annotation and peer review, but there is no guarantee that all such data points were removed from the corpus.

- 7. Accounting for mislabeled data: The Aya Annotation Platform did not contain any components that enabled re-labeling the assigned language of annotations. This may result in prompts and completions that appear under a particular language, but were submitted incorrectly and would need to be re-categorized into a different language. Additionally, while we trusted annotators were able to follow directions and had a high rate of manual auditing, some examples likely made it into the Aya Dataset that were not in instruction-style format or were free-form texts.

##### 10 Conclusion

Open participatory research continues to be under-resourced and undervalued, particularly when that work focuses on data creation [Sambasivan et al., 2021]. Aya involved participants from many different countries, different ages, and different levels of familiarity with the field of natural language processing. We see continued opportunity for computational linguists and machine-learning engineers to collaborate with social scientists such as sociolinguists, anthropologists, sociologists, and media studies scholars. As new norms in open science emerge [Krishna, 2020; Bowser et al., 2020], collaborations like these can help ensure that projects in NLP are motivated by an understanding of what language means to the people who use it every day.

With Aya, we hope to change the way data is created for multilingual NLP research. In line with this view, we release the Aya Dataset which is the first human-curated open-source, multilingual instruction-style dataset consisting of 204,114 prompt-completion pairs covering 65 languages. This dataset was built with the help of our open-science community of 2,997 collaborators from 119 countries over a period of eight months.

We also release the Aya Collection , which consists of 44 instruction-style datasets. These were prepared by transforming existing NLP datasets into prompt-completion pairs that can be leveraged for instruction tuning. Furthermore, we translate several high-quality datasets into 101 languages, thereby expanding coverage, particularly for many low-resource languages. This collection consists of 513M prompt and completion pairs covering 114 languages in total and is the largest multilingual instruction-finetuning collection today. Additionally, we release Aya Evaluation Suite , consisting of human-curated examples in 13 languages and translation of carefully selected prompts in 101 languages. Finally, we are also open-sourcing the Aya Annotation Platform so that communities can continue to use the platform to support the process of multilingual data collection. We hope these communities continue to grow and develop, and to connect speakers of low-resource languages around the world.

##### References

Julien Abadji, Pedro Javier Ortiz Suárez, Laurent Romary, and Benoît Sagot. Ungoliant: An optimized pipeline for the generation of a very large-scale multilingual web corpus. In Harald Lüngen, Marc Kupietz, Piotr Bański, Adrien Barbaresi, Simon Clematide, and Ines Pisetta (eds.), CMLC 2021-9th Workshop on Challenges in the Management of Large Corpora, Proceedings of the Workshop on Challenges in the Management of Large Corpora (CMLC-9) 2021. Limerick, 12 July 2021 (Online-Event), pp. 1 – 9, Mannheim, 2021. Leibniz-Institut für Deutsche Sprache. doi: 10.14618/ids-pub-10468. URL https://nbn-resolving.org/urn:nbn:de:bsz:mh39-104688.

Julien Abadji, Pedro Ortiz Suarez, Laurent Romary, and Benoît Sagot. Towards a cleaner documentoriented multilingual crawled corpus. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pp. 4344–4355, Marseille, France, June 2022. European Language Resources Association. URL https://aclanthology.org/2022.lrec-1.463.

Tilahun Abedissa, Ricardo Usbeck, and Yaregal Assabie. AmQA: Amharic Question Answering Dataset. arXiv preprint arXiv:2303.03290, 2023.

Gilles Adda, Sebastian Stüker, Martine Adda-Decker, Odette Ambouroue, Laurent Besacier, David Blachon, Hélène Bonneau-Maynard, Pierre Godard, Fatima Hamlaoui, Dmitry Idiatov, Guy-Noël

Kouarata, Lori Lamel, Emmanuel-Moselly Makasso, Annie Rialland, Mark Van de Velde, François Yvon, and Sabine Zerbian. Breaking the unwritten language barrier: The bulb project. Procedia Computer Science, 81:8–14, 2016. ISSN 1877-0509. doi: https://doi.org/10.1016/j.procs.2016.0 4.023. URL https://www.sciencedirect.com/science/article/pii/S1877050916300370. SLTU-2016 5th Workshop on Spoken Language Technologies for Under-resourced languages 09-12 May 2016 Yogyakarta, Indonesia.

David Ifeoluwa Adelani, Marek Masiak, Israel Abebe Azime, Jesujoba Alabi, Atnafu Lambebo Tonja, Christine Mwase, Odunayo Ogundepo, Bonaventure F. P. Dossou, Akintunde Oladipo, Doreen Nixdorf, Chris Chinenye Emezue, sana al azzawi, Blessing Sibanda, Davis David, Lolwethu Ndolela, Jonathan Mukiibi, Tunde Ajayi, Tatiana Moteu, Brian Odhiambo, Abraham Owodunni, Nnaemeka Obiefuna, Muhidin Mohamed, Shamsuddeen Hassan Muhammad, Teshome Mulugeta Ababu, Saheed Abdullahi Salahudeen, Mesay Gemeda Yigezu, Tajuddeen Gwadabe, Idris Abdulmumin, Mahlet Taye, Oluwabusayo Awoyomi, Iyanuoluwa Shode, Tolulope Adelani, Habiba Abdulganiyu, Abdul-Hakeem Omotayo, Adetola Adeeko, Abeeb Afolabi, Anuoluwapo Aremu, Olanrewaju Samuel, Clemencia Siro, Wangari Kimotho, Onyekachi Ogbu, Chinedu Mbonu, Chiamaka Chukwuneke, Samuel Fanijo, Jessica Ojo, Oyinkansola Awosan, Tadesse Kebede, Toadoum Sari Sakayo, Pamela Nyatsine, Freedmore Sidume, Oreen Yousuf, Mardiyyah Oduwole, Tshinu Tshinu, Ussen Kimanuka, Thina Diko, Siyanda Nxakama, Sinodos Nigusse, Abdulmejid Johar, Shafie Mohamed, Fuad Mire Hassan, Moges Ahmed Mehamed, Evrard Ngabire, Jules Jules, Ivan Ssenkungu, and Pontus Stenetorp. MasakhaNEWS: News Topic Classification for African Languages, 2023.

Asif Agha. Language and Social Relations. Studies in the Social and Cultural Foundations of Language. Cambridge University Press, 2006.

Roee Aharoni, Melvin Johnson, and Orhan Firat. Massively multilingual neural machine translation, 2019.

Orevaoghene Ahia, Julia Kreutzer, and Sara Hooker. The low-resource double bind: An empirical study of pruning for low-resource machine translation. In Findings of the Association for Computational Linguistics: EMNLP 2021, pp. 3316–3333, Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.findings-emnlp.282. URL https://aclanthology.org/2021.findings-emnlp.282.

AhmadMustafa. Urdu-Instruct-News-Article-Generation. https://huggingface.co/datasets/Ah madMustafa/Urdu-Instruct-News-Article-Generation, 2023a. Accessed: 2023-11-28.

AhmadMustafa. Urdu-Instruct-News-Category-Classification. https://huggingface.co/datas ets/AhmadMustafa/Urdu-Instruct-News-Category-Classification, 2023b. Accessed: 2023-11-28.

AhmadMustafa. Urdu-Instruct-News-Headline-Generation. https://huggingface.co/datasets/ AhmadMustafa/Urdu-Instruct-News-Headline-Generation, 2023c. Accessed: 2023-11-28.

AI Tamil Nadu. Tamil stories. https://huggingface.co/datasets/aitamilnadu/tamil_stories, 2023a. Accessed: 2023-12-15.

AI Tamil Nadu. Thirukkural Instruct. https://huggingface.co/datasets/aitamilnadu/thiruk kural_instruct, 2023b. Accessed: 2023-11-30.

Christopher Akiki, Giada Pistilli, Margot Mieskes, Matthias Gallé, Thomas Wolf, Suzana Ilić, and Yacine Jernite. Bigscience: A case study in the social construction of a multilingual large language model, 2022.

Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, et al. Santacoder: don’t reach for the stars! arXiv preprint arXiv:2301.03988, 2023.

Waseem AlShikh, Manhal Daaboul, Kirk Goddard, Brock Imel, Kiran Kamble, Parikshith Kulkarni, and Melisa Russak. Becoming self-instruct: introducing early stopping criteria for minimal instruct tuning. arXiv preprint arXiv:2307.03692, 2023.

Hossein Amirkhani, Mohammad AzariJafari, Soroush Faridan-Jahromi, Zeinab Kouhkan, Zohreh Pourjafari, and Azadeh Amirak. Farstail: A Persian natural language inference dataset. Soft Computing, 2023. doi: 10.1007/s00500-023-08959-3.

Lauri Andress, Tristen Hall, Sheila Davis, Judith Levine, Kimberly Cripps, and Dominique Guinn. Addressing power dynamics in community-engaged research partnerships. Journal of PatientReported Outcomes 4: 24, 2020.

Md Adnan Arefeen, Biplob Debnath, and Srimat Chakradhar. Leancontext: Cost-efficient domainspecific question answering using llms. arXiv preprint arXiv:2309.00841, 2023.

Mikel Artetxe, Sebastian Ruder, and Dani Yogatama. On the cross-lingual transferability of monolingual representations. CoRR, abs/1910.11856, 2019.

Seyram Avle, Emmanuel Quartey, and David Hutchful. Research on mobile phone data in the global south. The Oxford Handbook of Networked Communication, 2018. URL https://api.semantic scholar.org/CorpusID:168167342.

Kailash Awati and Simon Buckingham Shum. Big data metaphors we live by. Towards Data Science,

2015. URL https://towardsdatascience.com/big-data-metaphors-we-live-by-98d3fa44e bf8.

Paul Azunre, Salomey Osei, Salomey Afua Addo, Lawrence Asamoah Adu-Gyamfi, Stephen Moore, Bernard Adabankah, Bernard Opoku, Clara Asare-Nyarko, Samuel Nyarko, Cynthia Amoaba, Esther Dansoa Appiah, Felix Akwerh, Richard Nii Lante Lawson, Joel Budu, Emmanuel Debrah, Nana Adowaa Boateng, Wisdom Ofori, Edwin Buabeng-Munkoh, Franklin Adjei, Isaac. K. E. Ampomah, Joseph Otoo., Reindorf Nartey Borkor, Standylove Birago Mensah, Lucien Mensah, Mark Amoako Marcel, Anokye Acheampong Amponsah, and James Ben Hayfron-Acquah. Nlp for ghanaian languages. ArXiv, abs/2103.15475, 2021. URL https://api.semanticscholar.or g/CorpusID:232404908.

Stephen H Bach, Victor Sanh, Zheng-Xin Yong, Albert Webson, Colin Raffel, Nihal V Nayak, Abheesht Sharma, Taewoon Kim, M Saiful Bari, Thibault Fevry, et al. Promptsource: An integrated development environment and repository for natural language prompts. arXiv preprint arXiv:2202.01279, 2022.

Loïc Barrault, Yu-An Chung, Mariano Cora Meglioli, David Dale, Ning Dong, Paul-Ambroise Duquenne, Hady Elsahar, Hongyu Gong, Kevin Heffernan, John Hoffman, et al. Seamlessm4tmassively multilingual & multimodal machine translation. arXiv preprint arXiv:2308.11596, 2023.

Max Bartolo, Alastair Roberts, Johannes Welbl, Sebastian Riedel, and Pontus Stenetorp. Beat the ai: Investigating adversarial human annotation for reading comprehension. Transactions of the Association for Computational Linguistics, 8:662–678, 2020. doi: 10.1162/tacl\_a\_00338. URL https://doi.org/10.1162/tacl_a_00338.

Azam Bastanfard, Mohammad Shahabipour, and Dariush Amirkhani. Crowdsourcing of labeling image objects: an online gamification application for data collection. Multimedia Tools and Applications, Aug 2023. ISSN 1573-7721. doi: 10.1007/s11042-023-16325-6. URL https: //doi.org/10.1007/s11042-023-16325-6.

Susanne Beck, Carsten Bergenholtz, Marcel Bogers, Tiare-Maria Brasseur, Marie Louise Conradsen, Diletta Di Marco, Andreas P. Distel, Leonhard Dobusch, Daniel Dörler, Agnes Effert, Benedikt Fecher, Despoina Filiou, Lars Frederiksen, Thomas Gillier, Christoph Grimpe, Marc Gruber, Carolin Haeussler, Florian Heigl, Karin Hoisl, Katie Hyslop, Olga Kokshagina, Marcel LaFlamme, Cornelia Lawson, Hila Lifshitz-Assaf, Wolfgang Lukas, Markus Nordberg, Maria Theresa Norn, Marion Poetz, Marisa Ponti, Gernot Pruschak, Laia Pujol Priego, Agnieszka Radziwon, Janet Rafner, Gergana Romanova, Alexander Ruser, Henry Sauermann, Sonali K. Shah, Jacob F. Sherson, Julia Suess-Reyes, Christopher L. Tucci, Philipp Tuertscher, Jane Bjørn Vedel, Theresa Velden, Roberto Verganti, Jonathan Wareham, Andrea Wiggins, and Sunny Mosangzi Xu. The open innovation in science research field: a collaborative conceptualisation approach. Industry and Innovation, 29(2):136–185, 2022. doi: 10.1080/13662716.2020.1792274. URL https://doi.org/10.1080/13662716.2020.1792274.

Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic parsing on Freebase from question-answer pairs. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pp. 1533–1544, Seattle, Washington, USA, October 2013. Association for Computational Linguistics. URL https://aclanthology.org/D13-1160.

Michael S. Bernstein, Greg Little, Robert C. Miller, Björn Hartmann, Mark S. Ackerman, David R. Karger, David Crowell, and Katrina Panovich. Soylent: a word processor with a crowd inside. Commun. ACM, 58(8):85–94, 2015. doi: 10.1145/2791285. URL https://doi.org/10.1145/27 91285.

Steven Bird. Local languages, third spaces, and other high-resource scenarios. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 7817–7829, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.539. URL https://aclanthology.org/2022.acl-long.539.

Abeba Birhane, William Isaac, Vinodkumar Prabhakaran, Mark Diaz, Madeleine Clare Elish, Iason Gabriel, and Shakir Mohamed. Power to the people? opportunities and challenges for participatory ai. In Equity and Access in Algorithms, Mechanisms, and Optimization, EAAMO ’22. ACM, October 2022. doi: 10.1145/3551624.3555290. URL http://dx.doi.org/10.1145/3551624.355 5290.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language. In Thirty-Fourth AAAI Conference on Artificial Intelligence, 2020.

Verena Blaschke, Hinrich Schuetze, and Barbara Plank. A survey of corpora for Germanic lowresource languages and dialects. In Proceedings of the 24th Nordic Conference on Computational

Linguistics (NoDaLiDa), pp. 392–414, Tórshavn, Faroe Islands, May 2023. University of Tartu Library. URL https://aclanthology.org/2023.nodalida-1.41.

Su Lin Blodgett, Lisa Green, and Brendan O’Connor. Demographic dialectal variation in social media: A case study of African-American English. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 1119–1130, Austin, Texas, November 2016. Association for Computational Linguistics. doi: 10.18653/v1/D16-1120. URL https: //aclanthology.org/D16-1120.

Jan A. Botha, Manaal Faruqui, John Alex, Jason Baldridge, and Dipanjan Das. Learning to split and rephrase from Wikipedia edit history. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 732–737, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1080. URL https: //aclanthology.org/D18-1080.

Meriem Boubdir, Edward Kim, Beyza Ermis, Marzieh Fadaee, and Sara Hooker. Which prompts make the difference? data prioritization for efficient human llm evaluation, 2023.

Pierre Bourdieu. What makes a social class? on the theoretical and practical existence of groups. Berkeley journal of sociology, 32:1–17, 1987.

Anne Bowser, Caren Cooper, Alex De Sherbinin, Andrea Wiggins, Peter Brenton, Tyng-Ruey Chuang, Elaine Faustman, Mordechai Haklay, and Metis Meloche. Still in need of norms: the state of the data in citizen science. Citizen Science: Theory and Practice, 5(1), 2020.

Roger Brown and Albert Gilman. THE PRONOUNS OF POWER AND SOLIDARITY, pp. 252–

275. De Gruyter Mouton, Berlin, Boston, 1968. ISBN 9783110805376. doi: doi:10.1515/978311 0805376.252. URL https://doi.org/10.1515/9783110805376.252.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020.

Samuel Cahyawijaya, Holy Lovenia, Alham Fikri Aji, Genta Winata, Bryan Wilie, Fajri Koto, Rahmad Mahendra, Christian Wibisono, Ade Romadhony, Karissa Vincentio, et al. Nusacrowd: Open source initiative for indonesian nlp resources. In Findings of the Association for Computational Linguistics: ACL 2023, pp. 13745–13818, 2023.

Nicholas Carlini, Matthew Jagielski, Christopher A. Choquette-Choo, Daniel Paleka, Will Pearce, Hyrum Anderson, Andreas Terzis, Kurt Thomas, and Florian Tramèr. Poisoning web-scale training datasets is practical, 2023.

Joe Castaldo. “AI chatbots fall short in dozens of languages. A non-profit project aims to fix that". Globe & Mail, 2023.

Joe Castaldo. Meet the gig workers making ai machines more accurate, capable and powerful. The Globe and Mail, September 16, 2023.

Isaac Caswell, Theresa Breiner, Daan van Esch, and Ankur Bapna. Language ID in the wild: Unexpected challenges on the path to a thousand-language web text corpus. In Proceedings of the 28th International Conference on Computational Linguistics, pp. 6588–6608, Barcelona, Spain (Online), December 2020. International Committee on Computational Linguistics. doi: 10.18653/v1/2020.coling-main.579. URL https://aclanthology.org/2020.coling-main.579.

Lichang Chen, Shiyang Li, Jun Yan, Hai Wang, Kalpa Gunaratna, Vikas Yadav, Zheng Tang, Vijay Srinivasan, Tianyi Zhou, Heng Huang, et al. Alpagasus: Training a better alpaca with fewer data. arXiv preprint arXiv:2307.08701, 2023a.

Pinzhen Chen, Shaoxiong Ji, Nikolay Bogoychev, Barry Haddow, and Kenneth Heafield. Monolingual or multilingual instruction tuning: Which makes a better alpaca, 2023b.

Sanxing Chen, Yongqiang Chen, and Börje F. Karlsson. Dataset and baseline system for multilingual extraction and normalization of temporal and numerical expressions. arXiv preprint arXiv:2303.18103, 2023c.

Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Yunxuan Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416, 2022.

John Chung, Ece Kamar, and Saleema Amershi. Increasing diversity while maintaining accuracy: Text data generation with large language models and human interventions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.acl-long.34. URL http://dx.doi.org/10.18653/v1/2023.acl-long.34.

Tarin Clanuwat, Mikel Bober-Irizar, Asanobu Kitamoto, Alex Lamb, Kazuaki Yamamoto, and David Ha. Deep learning for classical japanese literature. ArXiv, abs/1812.01718, 2018. URL https://api.semanticscholar.org/CorpusID:54458639.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. In NAACL,

- 2019.

Jonathan H. Clark, Eunsol Choi, Michael Collins, Dan Garrette, Tom Kwiatkowski, Vitaly Nikolaev, and Jennimaria Palomaki. Tydi qa: A benchmark for information-seeking question answering in typologically diverse languages. Transactions of the Association for Computational Linguistics,

- 2020.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv:1803.05457v1, 2018.

Alexis Conneau, Guillaume Lample, Ruty Rinott, Adina Williams, Samuel R Bowman, Holger Schwenk, and Veselin Stoyanov. Xnli: Evaluating cross-lingual sentence representations. arXiv preprint arXiv:1809.05053, 2018.

Mike Conover, Matt Hayes, Ankit Mathur, Jianwei Xie, Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell, Matei Zaharia, and Reynold Xin. Free Dolly: Introducing the World’s First Truly Open Instruction-Tuned LLM, 2023. URL https://www.databricks.com/blog/2023/04/12/dolly-f irst-open-commercially-viable-instruction-tuned-llm.

ConseggioLigure. Lij news instruct ita-lij. https://huggingface.co/datasets/ConseggioLigur e/lijnews-instruct-ita-lij, 2023a. Accessed: 2023-11-28.

ConseggioLigure. Lij news instruct lij-ita. https://huggingface.co/datasets/ConseggioLigur e/lijnews-instruct-lij-ita, 2023b. Accessed: 2023-11-28.

ConseggioLigure. Seed instruct eng-lij. https://huggingface.co/datasets/ConseggioLigure/ seed-instruct-eng-lij, 2023c. Accessed: 2023-11-28.

ConseggioLigure. Seed instruct lij-eng. https://huggingface.co/datasets/ConseggioLigure/ seed-instruct-lij-eng, 2023d. Accessed: 2023-11-28.

Eric Corbett, Emily Denton, and Sheena Erete. Power and public participation in ai. In Proceedings of the 3rd ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization, EAAMO ’23, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400703812. doi: 10.1145/3617694.3623228. URL https://doi.org/10.1145/3617694.3623 228.

Kate Crawford. Atlas of AI: Power, Politics, and the Planetary Costs of Artificial Intelligence. Yale University Press, New Haven, Connecticut, 2021.

Yiming Cui, Ting Liu, Wanxiang Che, Li Xiao, Zhipeng Chen, Wentao Ma, Shijin Wang, and Guoping Hu. A span-extraction dataset for Chinese machine reading comprehension. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 5883–5889, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1600. URL https://aclanthology.org/D19-1600.

Yiming Cui, Ziqing Yang, and Xin Yao. Efficient and effective text encoding for chinese llama and alpaca, 2023.

Raj Dabre, Chenhui Chu, and Anoop Kunchukuttan. A survey of multilingual neural machine translation. ACM Comput. Surv., 53(5), sep 2020. ISSN 0360-0300. doi: 10.1145/3406095. URL https://doi.org/10.1145/3406095.

Viet Dac Lai, Chien Van Nguyen, Nghia Trung Ngo, Thuat Nguyen, Franck Dernoncourt, Ryan A Rossi, and Thien Huu Nguyen. Okapi: Instruction-tuned large language models in multiple languages with reinforcement learning from human feedback. arXiv e-prints, pp. arXiv–2307, 2023.

Pradeep Dasigi, Nelson F. Liu, Ana Marasovic, Noah A. Smith, and Matt Gardner. Quoref: A reading comprehension dataset with questions requiring coreferential reasoning. arXiv:1908.05803v2, 2019.

Flavio A. de Franga, Adriana S. Vivacqua, and Maria Luiza M. Campos. Designing a gamification mechanism to encourage contributions in a crowdsourcing system. In 2015 IEEE 19th International Conference on Computer Supported Cooperative Work in Design (CSCWD), pp. 462–466, 2015. doi: 10.1109/CSCWD.2015.7231003.

Andrea De Mauro, Marco Greco, and Michele Grimaldi. What is big data? a consensual definition and a review of key research topics. In AIP conference proceedings, volume 1644, pp. 97–104. American Institute of Physics, 2015.

Fernando Delgado, Stephen Yang, Michael Madaio, and Qian Yang. The participatory turn in ai design: Theoretical foundations and the current state of practice. Proceedings of the 3rd ACM Conference on Equity and Access in Algorithms, Mechanisms, and Optimization, 2023. URL https://api.semanticscholar.org/CorpusID:263605822.

Yue Deng, Wenxuan Zhang, Sinno Jialin Pan, and Lidong Bing. Multilingual jailbreak challenges in large language models. ArXiv, abs/2310.06474, 2023. URL https://api.semanticscholar. org/CorpusID:263831094.

desik98. Telugu Riddles. https://huggingface.co/datasets/desik98/TeluguRiddles, 2023. Accessed: 2023-11-30.

Kaustubh D Dhole, Varun Gangal, Sebastian Gehrmann, Aadesh Gupta, Zhenhao Li, Saad Mahamood, Abinaya Mahendiran, Simon Mille, Ashish Shrivastava, Samson Tan, et al. Nlaugmenter: A framework for task-sensitive natural language augmentation. arXiv preprint arXiv:2112.02721, 2021.

Sumanth Doddapaneni, Rahul Aralikatte, Gowtham Ramesh, Shreya Goyal, Mitesh M. Khapra, Anoop Kunchukuttan, and Pratyush Kumar. Towards leaving no Indic language behind: Building monolingual corpora, benchmark and models for Indic languages. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12402–12426, Toronto, Canada, July 2023. Association for Computational Linguistics. URL https://aclanthology.org/2023.acl-long.693.

Jesse Dodge, Maarten Sap, Ana Marasovic, William Agnew, Gabriel Ilharco, Dirk Groeneveld, and Matt Gardner. Documenting the english colossal clean crawled corpus. CoRR, abs/2104.08758,

2021. URL https://arxiv.org/abs/2104.08758.

A Seza Doğruöz, Sunayana Sitaram, and Zheng-Xin Yong. Representativeness as a forgotten lesson for multilingual and code-switched data collection and preparation. arXiv preprint arXiv:2310.20470, 2023.

Bill Dolan and Chris Brockett. Automatically constructing a corpus of sentential paraphrases. In Third International Workshop on Paraphrasing (IWP2005). Asia Federation of Natural Language Processing, January 2005. URL https://www.microsoft.com/en-us/research/publication /automatically-constructing-a-corpus-of-sentential-paraphrases/.

Esin Durmus, Karina Nyugen, Thomas I Liao, Nicholas Schiefer, Amanda Askell, Anton Bakhtin, Carol Chen, Zac Hatfield-Dodds, Danny Hernandez, Nicholas Joseph, et al. Towards measuring the representation of subjective global opinions in language models. arXiv preprint arXiv:2306.16388, 2023.

el2e10. Aya Indic Sentiment. https://huggingface.co/datasets/el2e10/aya-indicsentiment, 2023a. Accessed: 2023-11-28.

el2e10. Aya Paraphrase. https://huggingface.co/datasets/el2e10/aya-paraphrase, 2023b. Accessed: 2023-11-28.

Alexander R. Fabbri, Irene Li, Tianwei She, Suyi Li, and Dragomir R. Radev. Multi-news: a large-scale multi-document summarization dataset and abstractive hierarchical model, 2019.

Oliver Falck, Stephan Heblich, Alfred Lameli, and Jens Südekum. Dialects, cultural identity, and economic exchange. Journal of urban economics, 72(2-3):225–239, 2012.

Angela Fan, Shruti Bhosale, Holger Schwenk, Zhiyi Ma, Ahmed El-Kishky, Siddharth Goyal, Mandeep Baines, Onur Celebi, Guillaume Wenzek, Vishrav Chaudhary, et al. Beyond english-centric multilingual machine translation. Journal of Machine Learning Research, 22(107):1–48, 2021.

Mehrdad Farahani, Mohammad Gharachorloo, and Mohammad Manthouri. Leveraging parsbert and pretrained mt5 for persian abstractive text summarization. In 2021 26th International Computer Conference, Computer Society of Iran (CSICC), pp. 1–6. IEEE, 2021.

Emilio Ferrara. Should chatgpt be biased? challenges and risks of bias in large language models. arXiv preprint arXiv:2304.03738, 2023.

∀, Wilhelmina Nekoto, Vukosi Marivate, Tshinondiwa Matsila, Timi Fasubaa, Taiwo Fagbohungbe, Solomon Oluwole Akinola, Shamsuddeen Muhammad, Salomon Kabongo Kabenamualu, Salomey Osei, Freshia Sackey, Rubungo Andre Niyongabo, Ricky Macharm, Perez Ogayo, Orevaoghene Ahia, Musie Meressa Berhe, Mofetoluwa Adeyemi, Masabata Mokgesi-Selinga, Lawrence Okegbemi, Laura Martinus, Kolawole Tajudeen, Kevin Degila, Kelechi Ogueji, Kathleen Siminyu, Julia Kreutzer, Jason Webster, Jamiil Toure Ali, Jade Abbott, Iroro Orife, Ignatius Ezeani, Idris Abdulkadir Dangana, Herman Kamper, Hady Elsahar, Goodness Duru, Ghollah Kioko, Murhabazi Espoir, Elan van Biljon, Daniel Whitenack, Christopher Onyefuluchi, Chris Chinenye Emezue, Bonaventure F. P. Dossou, Blessing Sibanda, Blessing Bassey, Ayodele Olabiyi, Arshath Ramkilowan, Alp Öktem, Adewale Akinfaderin, and Abdallah Bashir. Participatory research for low-resourced machine translation: A case study in African languages. In Trevor Cohn, Yulan He, and Yang Liu (eds.), Findings of the Association for Computational Linguistics: EMNLP 2020, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.findi ngs-emnlp.195. URL https://aclanthology.org/2020.findings-emnlp.195.

Chiara Franzoni and Henry Sauermann. Crowd science: The organization of scientific research in open collaborative projects. Research Policy, 43(1):1–20, 2014. ISSN 0048-7333. doi: https: //doi.org/10.1016/j.respol.2013.07.005. URL https://www.sciencedirect.com/science/arti cle/pii/S0048733313001212.

ganeshjcs. Hindi Article Summarization. https://huggingface.co/datasets/ganeshjcs/hind i-article-summarization, 2023a. Accessed: 2023-11-28.

ganeshjcs. Hindi Headline Article Generation. https://huggingface.co/datasets/ganeshjcs/ hindi-headline-article-generation, 2023b. Accessed: 2023-11-28.

Ya Gao and WenQi Liu. Measures to sustain endangered languages: A bilingual competition model with sliding mode control. PLOS ONE, 18(6):1–16, 06 2023. doi: 10.1371/journal.pone.0287850. URL https://doi.org/10.1371/journal.pone.0287850.

Dan Garrette, Jason Mielens, and Jason Baldridge. Real-world semi-supervised learning of postaggers for low-resource languages. In Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 583–592, 2013.

Marco Gerosa, Igor Wiese, Bianca Trinkenreich, Georg Link, Gregorio Robles, Christoph Treude, Igor Steinmacher, and Anita Sarma. The shifting sands of motivation: Revisiting what drives contributors in open source. In 2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE), pp. 1046–1058. IEEE, 2021.

Sourojit Ghosh and Aylin Caliskan. Chatgpt perpetuates gender bias in machine translation and ignores non-gendered pronouns: Findings across bengali and five other low-resource languages, 2023.

Bogdan Gliwa, Iwona Mochol, Maciej Biesek, and Aleksander Wawer. SAMSum corpus: A humanannotated dialogue dataset for abstractive summarization. In Proceedings of the 2nd Workshop on New Frontiers in Summarization, pp. 70–79, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-5409. URL https://aclanthology.org /D19-5409.

Charles Goodwin. Co-Operative Action. Learning in Doing: Social, Cognitive and Computational Perspectives. Cambridge University Press, 2017.

Naman Goyal, Jingfei Du, Myle Ott, Giri Anantharaman, and Alexis Conneau. Larger-Scale Transformers for Multilingual Masked Language Modeling. In Proceedings of the 6th Workshop on Representation Learning for NLP (RepL4NLP-2021). Association for Computational Linguistics, 2021a.

Naman Goyal, Cynthia Gao, Vishrav Chaudhary, Peng-Jen Chen, Guillaume Wenzek, Da Ju, Sanjana Krishnan, Marc’Aurelio Ranzato, Francisco Guzman, and Angela Fan. The flores-101 evaluation benchmark for low-resource and multilingual machine translation, 2021b.

David Graff, Junbo Kong, Ke Chen, and Kazuaki Maeda. English gigaword. Linguistic Data Consortium, Philadelphia, 4(1):34, 2003.

Giovanni Grano, Andrea Di Sorbo, Francesco Mercaldo, Corrado A Visaggio, Gerardo Canfora, and Sebastiano Panichella. Android apps and user feedback: a dataset for software evolution and quality improvement. In Proceedings of the 2nd ACM SIGSOFT international workshop on app market analytics, pp. 8–11, 2017.

Dirk Groeneveld, Iz Beltagy, Pete Walsh, Akshita Bhagia, Rodney Kinney, Oyvind Tafjord, Ananya Harsh Jha, Hamish Ivison, Ian Magnusson, Yizhong Wang, Shane Arora, David Atkinson, Russell Authur, Khyathi Chandu, Arman Cohan, Jennifer Dumas, Yanai Elazar, Yuling Gu, Jack Hessel, Tushar Khot, William Merrill, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Valentina Pyatkin, Abhilasha Ravichander, Dustin Schwenk, Saurabh Shah, Will Smith, Nishant Subramani, Mitchell Wortsman, Pradeep Dasigi, Nathan Lambert, Kyle Richardson, Jesse Dodge, Kyle Lo, Luca Soldaini, Noah A. Smith, and Hannaneh Hajishirzi. OLMo: Accelerating the Science of Language Models. arXiv preprint, 2024.

Yuling Gu, Bhavana Dalvi, and Peter Clark. DREAM: Improving situational QA by first elaborating the situation. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 1115– 1127, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.naacl-main.82. URL https://aclanthology.org/2022.naacl-main.82.

Adriana Guevara-Rukoz, Isin Demirsahin, Fei He, Shan-Hui Cathy Chu, Supheakmungkol Sarin, Knot Pipatsrisawat, Alexander Gutkin, Alena Butryna, and Oddur Kjartansson. Crowdsourcing Latin American Spanish for low-resource text-to-speech. In Proceedings of the Twelfth Language Resources and Evaluation Conference, pp. 6504–6513, Marseille, France, May 2020. European Language Resources Association. ISBN 979-10-95546-34-4. URL https://aclanthology.org/2 020.lrec-1.801.

Antonio Gulli. AG’s Corpus of News Articles. Dipartimento di Informatica, University of Pisa, Nov, 2005. URL http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html.

Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio César Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, Sébastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, and Yuanzhi Li. Textbooks are all you need, 2023.

Himanshu Gupta, Kevin Scaria, Ujjwala Anantheswaran, Shreyas Verma, Mihir Parmar, Saurabh Arjun Sawant, Chitta Baral, and Swaroop Mishra. Targen: Targeted data generation with large language models, 2023.

Margot Hanley, Apoorv Khandelwal, Hadar Averbuch-Elor, Noah Snavely, and Helen Nissenbaum.

An ethical highlighter for people-centric dataset creation. arXiv preprint arXiv:2011.13583, 2020. Christian Hardmeier and Liane Guillou. Pronoun translation in english-french machine translation:

An analysis of error types, 2018. Kai Hartung, Aaricia Herygers, Shubham Kurlekar, Khabbab Zakaria, Taylan Volkan, Sören Gröttrup, and Munir Georges. Measuring sentiment bias in machine translation, 2023.

Tahmid Hasan, Abhik Bhattacharjee, Md. Saiful Islam, Kazi Mubasshir, Yuan-Fang Li, Yong-Bin Kang, M. Sohel Rahman, and Rifat Shahriyar. XL-sum: Large-scale multilingual abstractive summarization for 44 languages. In Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021, pp. 4693–4703, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.findings-acl.413. URL https://aclanthology.org/2021.find ings-acl.413.

Einar Haugen. Planning for a standard language in modern norway. Anthropological linguistics, pp. 8–21, 1959.

William Held, Camille Harris, Michael Best, and Diyi Yang. A material lens on coloniality in nlp, 2023.

Karl Moritz Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. Teaching machines to read and comprehend. In Advances in neural information processing systems, pp. 1693–1701, 2015.

hghader1. FarsTail-Instruct-LLM. https://huggingface.co/datasets/hghader1/FarsTail-Ins truct-LLM, 2023. Accessed: 2023-11-28.

Oskar Holmström and Ehsan Doostmohammadi. Making instruction finetuning accessible to nonEnglish languages: A case study on Swedish models. In Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa), pp. 634–642, Tórshavn, Faroe Islands, May 2023. University of Tartu Library. URL https://aclanthology.org/2023.nodalida-1.62.

Or Honovich, Thomas Scialom, Omer Levy, and Timo Schick. Unnatural instructions: Tuning language models with (almost) no human labor. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 14409–14428, Toronto, Canada, July 2023. Association for Computational Linguistics. URL https://aclanthology.o rg/2023.acl-long.806.

Dirk Hovy and Shrimai Prabhumoye. Five sources of bias in natural language processing. Language and Linguistics Compass, 15(8):e12432, 2021.

Eduard Hovy, Laurie Gerber, Ulf Hermjakob, Chin-Yew Lin, and Deepak Ravichandran. Toward semantics-based answer pinpointing. In Proceedings of the First International Conference on Human Language Technology Research, 2001. URL https://aclanthology.org/H01-1069.

Pei-Yun Hsueh, Prem Melville, and Vikas Sindhwani. Data quality from crowdsourcing: a study of annotation selection criteria. In Proceedings of the NAACL HLT 2009 workshop on active learning for natural language processing, pp. 27–35, 2009.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021.

Huang Huang, Fei Yu, Jianqing Zhu, Xuening Sun, Hao Cheng, Dingjie Song, Zhihong Chen, Abdulmohsen Alharthi, Bang An, Ziche Liu, et al. Acegpt, localizing large language models in arabic. arXiv preprint arXiv:2309.12053, 2023.

Kuan-Hao Huang, I-Hung Hsu, Premkumar Natarajan, Kai-Wei Chang, and Nanyun Peng. Multilingual generative language models for zero-shot cross-lingual event argument extraction, 2022.

Lifu Huang, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Cosmos QA: Machine reading comprehension with contextual commonsense reasoning. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 2391–2401, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1243. URL https: //aclanthology.org/D19-1243.

Khalid Hussain, Nimra Mughal, Irfan Ali, Saif Hassan, and Sher Muhammad Daudpota. Urdu News Dataset 1M. Technical report, Mendeley Data, V3, 2021.

Mika Hämäläinen. Endangered Languages are not Low-Resourced!, pp. 1–11. University of Helsinki, March 2021. doi: 10.31885/9789515150257.1. URL http://dx.doi.org/10.31885/9789515150 257.1.

Iftitahu. Indonesian Instruct Stories. https://huggingface.co/datasets/Iftitahu/indonesia n_instruct_stories, 2023a. Accessed: 2023-11-28.

Iftitahu. Javanese Instruct Stories. https://huggingface.co/datasets/Iftitahu/javanese_in struct_stories, 2023b. Accessed: 2023-11-28.

Iftitahu. Sudanese Instruct Stories. https://huggingface.co/datasets/Iftitahu/sundanese_i nstruct_stories, 2023c. Accessed: 2023-11-28.

Amirova Gulruh Ilhomovna and S Yuldasheva. You have got to know your language to understand your culture. In Interdiscipline Innovation and Scientific Research Conference, volume 1, pp. 103–106, 2023.

Shankar Iyer, Nikhil Dandekar, and Kornäl Csernai. Quora question pairs, 2012.

Srinivasan Iyer, Xi Victoria Lin, Ramakanth Pasunuru, Todor Mihaylov, Daniel Simig, Ping Yu, Kurt Shuster, Tianlu Wang, Qing Liu, Punit Singh Koura, et al. Opt-iml: Scaling language model instruction meta learning through the lens of generalization. arXiv preprint arXiv:2212.12017, 2022.

Meng Ji, Meng Ji, Pierrette Bouillon, and Mark Seligman. Cultural and Linguistic Bias of Neural Machine Translation Technology, pp. 100–128. Studies in Natural Language Processing. Cambridge University Press, 2023a. doi: 10.1017/9781108938976.005.

Yunjie Ji, Yan Gong, Yong Deng, Yiping Peng, Qiang Niu, Baochang Ma, and Xiangang Li. Towards better instruction following language models for chinese: Investigating the impact of training data and evaluation, 2023b.

jjzha. IMDB Dutch Instruct. https://huggingface.co/datasets/jjzha/imdb-dutch-instruct,

2023. Accessed: 2023-11-28.

Anna Jørgensen, Dirk Hovy, and Anders Søgaard. Challenges of studying and processing dialects in social media. In Proceedings of the Workshop on Noisy User-generated Text, pp. 9–18, Beijing, China, July 2015. Association for Computational Linguistics. doi: 10.18653/v1/W15-4302. URL https://aclanthology.org/W15-4302.

Joseph Cheung. GuanacoDataset (Revision 8cf0d29) , 2023. URL https://huggingface.co/dat asets/JosephusCheung/GuanacoDataset.

Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. triviaqa: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. arXiv e-prints, art. arXiv:1705.03551, 2017.

Pratik Joshi, Christain Barnes, Sebastin Santy, Simran Khanuja, Sanket Shah, Anirudh Srinivasan, Satwik Bhattamishra, Sunayana Sitaram, Monojit Choudhury, and Kalika Bali. Unsung challenges of building and deploying language technologies for low resource language communities. In Proceedings of the 16th International Conference on Natural Language Processing, pp. 211– 219, International Institute of Information Technology, Hyderabad, India, December 2019. NLP Association of India. URL https://aclanthology.org/2019.icon-1.25.

Pratik Joshi, Sebastin Santy, Amar Budhiraja, Kalika Bali, and Monojit Choudhury. The state and fate of linguistic diversity and inclusion in the NLP world. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 6282–6293, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.560. URL https: //aclanthology.org/2020.acl-main.560.

jxmorris12, thomwolf, lhoestq, and lewtun. ag_news. https://huggingface.co/datasets/ag_n ews, 2023. Accessed: 2023-11-28.

Jenna Kanerva, Filip Ginter, Li-Hsin Chang, Iiro Rastas, Valtteri Skantsi, Jemina Kilpeläinen, Hanna-Mari Kupari, Jenna Saarni, Maija Sevón, and Otto Tarkka. Finnish paraphrase corpus. In Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa), pp. 288– 298, Reykjavik, Iceland (Online), May 31–2 June 2021. Linköping University Electronic Press, Sweden. URL https://aclanthology.org/2021.nodalida-main.29.

Khyati Khandelwal, Manuel Tonneau, Andrew M. Bean, Hannah Rose Kirk, and Scott A. Hale. Casteist but not racist? quantifying disparities in large language model bias between india and the west. ArXiv, abs/2309.08573, 2023. URL https://api.semanticscholar.org/CorpusID: 262013517.

Daniel Khashabi, Snigdha Chaturvedi, Michael Roth, Shyam Upadhyay, and Dan Roth. Looking beyond the surface:a challenge set for reading comprehension over multiple sentences. In Proceedings of North American Chapter of the Association for Computational Linguistics (NAACL), 2018.

Daniel Khashabi, Sewon Min, Tushar Khot, Ashish Sabharwal, Oyvind Tafjord, Peter Clark, and Hannaneh Hajishirzi. Unifiedqa: Crossing format boundaries with a single qa system. arXiv preprint arXiv:2005.00700, 2020.

Md Tawkat Islam Khondaker, Abdul Waheed, El Moatez Billah Nagoudi, and Muhammad AbdulMageed. Gptaraeval: A comprehensive evaluation of chatgpt on arabic nlp, 2023.

Tushar Khot, Peter Clark, Michal Guerquin, Peter Jansen, and Ashish Sabharwal. Qasc: A dataset for question answering via sentence composition. arXiv:1910.11473v2, 2020.

Hyunwoo Kim, Jack Hessel, Liwei Jiang, Peter West, Ximing Lu, Youngjae Yu, Pei Zhou, Ronan Le Bras, Malihe Alikhani, Gunhee Kim, Maarten Sap, and Yejin Choi. Soda: Million-scale dialogue distillation with social commonsense contextualization. ArXiv, abs/2212.10465, 2022.

Joongwon Kim, Mounica Maddela, Reno Kriz, Wei Xu, and Chris Callison-Burch. BiSECT: Learning to split and rephrase sentences with bitexts. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 6193–6209, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.500. URL https://aclanthology.org/2021.emnlp-main.500.

Seungone Kim, Se June Joo, Doyoung Kim, Joel Jang, Seonghyeon Ye, Jamin Shin, and Minjoon Seo. The cot collection: Improving zero-shot and few-shot learning of language models via chainof-thought fine-tuning. arXiv preprint arXiv:2305.14045, 2023.

Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi-Rui Tam, Keith Stevens, Abdullah Barhoum, Nguyen Minh Duc, Oliver Stanley, Richárd Nagyfi, et al. Openassistant conversations–democratizing large language model alignment. arXiv preprint arXiv:2304.07327, 2023.

Hadas Kotek, Rikker Dockum, and David Q. Sun. Gender bias and stereotypes in large language models. Proceedings of The ACM Collective Intelligence Conference, 2023. URL https://api. semanticscholar.org/CorpusID:261276445.

Julia Kreutzer, Isaac Caswell, Lisa Wang, Ahsan Wahab, Daan van Esch, Nasanbayar Ulzii-Orshikh, Allahsera Tapo, Nishant Subramani, Artem Sokolov, Claytone Sikasote, Monang Setyawan, Supheakmungkol Sarin, Sokhar Samb, Benoît Sagot, Clara Rivera, Annette Rios, Isabel Papadimitriou, Salomey Osei, Pedro Ortiz Suarez, Iroro Orife, Kelechi Ogueji, Andre Niyongabo Rubungo, Toan Q. Nguyen, Mathias Müller, André Müller, Shamsuddeen Hassan Muhammad, Nanda Muhammad, Ayanda Mnyakeni, Jamshidbek Mirzakhalov, Tapiwanashe Matangira, Colin Leong, Nze Lawson, Sneha Kudugunta, Yacine Jernite, Mathias Jenny, Orhan Firat, Bonaventure F. P. Dossou, Sakhile Dlamini, Nisansa de Silva, Sakine Çabuk Ballı, Stella Biderman, Alessia Battisti, Ahmed Baruwa, Ankur Bapna, Pallavi Baljekar, Israel Abebe Azime, Ayodele Awokoya, Duygu Ataman, Orevaoghene Ahia, Oghenefego Ahia, Sweta Agrawal, and Mofetoluwa Adeyemi. Quality at a glance: An audit of web-crawled multilingual datasets. Transactions of the Association for Computational Linguistics, 10:50–72, 2022. doi: 10.1162/tacl_a_00447. URL https://aclanthology.org/2022.tacl-1.4.

Venni V. Krishna. Open science and its enemies: Challenges for a sustainable science–society social contract. Journal of Open Innovation: Technology, Market, and Complexity, 6(3):61, 2020. ISSN 2199-8531. doi: https://doi.org/10.3390/joitmc6030061.

Sneha Kudugunta, Isaac Caswell, Biao Zhang, Xavier Garcia, Christopher A Choquette-Choo, Katherine Lee, Derrick Xin, Aditya Kusupati, Romi Stella, Ankur Bapna, et al. Madlad-400: A multilingual and document-level large audited dataset. arXiv preprint arXiv:2309.04662, 2023.

Anoop Kunchukuttan, Siddharth Jain, and Rahul Kejriwal. A large-scale evaluation of neural machine transliteration for Indic languages. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pp. 3469–3475, Online, April 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.eacl-main.303. URL https://aclanthology.org/2021.eacl-main.303.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452–466, 2019. doi: 10.1162/tacl_a_00276. URL https://aclanthology.org/Q19-1026.

Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi-Rui Tam, Keith Stevens, Abdullah Barhoum, Nguyen Minh Duc, Oliver Stanley, Richárd Nagyfi, Shahul ES, Sameer Suri, David Glushkov, Arnav Dantuluri, Andrew Maguire, Christoph Schuhmann, Huu Nguyen, and Alexander Mattick. Openassistant conversations – democratizing large language model alignment, 2023.

Faisal Ladhak, Esin Durmus, Claire Cardie, and Kathleen McKeown. WikiLingua: A new benchmark dataset for cross-lingual abstractive summarization. In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 4034–4048, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.findings-emnlp.360. URL https://aclanthology.org/2020.findings-emnlp.360.

Preethi Lahoti, Nicholas Blumm, Xiao Ma, Raghavendra Kotikalapudi, Sahitya Potluri, Qijun Tan, Hansa Srinivasan, Ben Packer, Ahmad Beirami, Alex Beutel, and Jilin Chen. Improving diversity of demographic representation in large language models via collective-critiques and self-voting, 2023.

Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard Hovy. RACE: Large-scale ReAding comprehension dataset from examinations. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pp. 785–794, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/v1/D17-1082. URL https://aclant hology.org/D17-1082.

Viet Dac Lai, Chien Van Nguyen, Nghia Trung Ngo, Thuat Nguyen, Franck Dernoncourt, Ryan A Rossi, and Thien Huu Nguyen. Okapi: Instruction-tuned large language models in multiple languages with reinforcement learning from human feedback. arXiv preprint arXiv:2307.16039, 2023.

Guillaume Lample and Alexis Conneau. Cross-lingual language model pretraining. arXiv preprint arXiv:1901.07291, 2019.

Hugo Laurençon, Lucile Saulnier, Thomas Wang, Christopher Akiki, Albert Villanova del Moral, Teven Le Scao, Leandro Von Werra, Chenghao Mou, Eduardo González Ponferrada, Huu Nguyen, et al. The bigscience roots corpus: A 1.6 tb composite multilingual dataset. Advances in Neural Information Processing Systems, 35:31809–31826, 2022.

Rémi Lebret, David Grangier, and Michael Auli. Generating text from structured data with application to the biography domain. CoRR, abs/1603.07771, 2016. URL http://arxiv.org/abs/16 03.07771.

Nayeon Lee, Chani Jung, and Alice Oh. Hate speech classifiers are culturally insensitive. In Proceedings of the First Workshop on Cross-Cultural Considerations in NLP (C3NLP), pp. 35– 46, Dubrovnik, Croatia, May 2023. Association for Computational Linguistics. URL https: //aclanthology.org/2023.c3nlp-1.5.

Jens Lehmann, Robert Isele, Max Jakob, Anja Jentzsch, Dimitris Kontokostas, Pablo Mendes, Sebastian Hellmann, Mohamed Morsey, Patrick Van Kleef, Sören Auer, and Christian Bizer. Dbpedia - a large-scale, multilingual knowledge base extracted from wikipedia. Semantic Web Journal, 6, 01 2014. doi: 10.3233/SW-140134.

Regina Lenart-Gansiniec, Wojciech Czakon, Łukasz Sułkowski, and Jasna Pocek. Understanding crowdsourcing in science. Review of Managerial Science, 17(8):2797–2830, Nov 2023. ISSN 18636691. doi: 10.1007/s11846-022-00602-z. URL https://doi.org/10.1007/s11846-022-00602-z.

Heather Lent, Kelechi Ogueji, Miryam de Lhoneux, Orevaoghene Ahia, and Anders Søgaard. What a creole wants, what a creole needs. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, pp. 6439–6449, Marseille, France, June 2022. European Language Resources Association. URL https://aclanthology.org/2022.lrec-1.691.

Vladimir I Levenshtein et al. Binary codes capable of correcting deletions, insertions, and reversals. In Soviet physics doklady, volume 10, pp. 707–710. Soviet Union, 1966.

Patrick Lewis, Barlas Oğuz, Ruty Rinott, Sebastian Riedel, and Holger Schwenk. Mlqa: Evaluating cross-lingual extractive question answering, 2020.

Haonan Li, Fajri Koto, Minghao Wu, Alham Fikri Aji, and Timothy Baldwin. Bactrian-x: A multilingual replicable instruction-following model with low-rank adaptation. arXiv preprint arXiv:2305.15011, 2023a.

Haoran Li, Yulin Chen, Jinglong Luo, Yan Kang, Xiaojin Zhang, Qi Hu, Chunkit Chan, and Yangqiu Song. Privacy in large language models: Attacks, defenses and future directions. ArXiv, abs/2310.10383, 2023b. URL https://api.semanticscholar.org/CorpusID:264145758.

Lei Li, Yuwei Yin, Shicheng Li, Liang Chen, Peiyi Wang, Shuhuai Ren, Mukai Li, Yazheng Yang, Jingjing Xu, Xu Sun, Lingpeng Kong, and Qi Liu. M3it: A large-scale dataset towards multimodal multilingual instruction tuning, 2023c.

Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al. Starcoder: may the source be with you! arXiv preprint arXiv:2305.06161, 2023d.

Xin Li and Dan Roth. Learning question classifiers. In COLING 2002: The 19th International Conference on Computational Linguistics, 2002. URL https://aclanthology.org/C02-1150.

Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. Making large language models better reasoners with step-aware verifier, 2023e.

Yudong Li, Yuqing Zhang, Zhe Zhao, Linlin Shen, Weijie Liu, Weiquan Mao, and Hui Zhang. Csl: A large-scale chinese scientific literature dataset, 2022.

Constantine Lignos, Nolan Holley, Chester Palen-Michel, and Jonne Sälevä. Toward more meaningful resources for lower-resourced languages. In Findings of the Association for Computational Linguistics: ACL 2022, pp. 523–532, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.findings-acl.44. URL https://aclanthology.org/2022.fi ndings-acl.44.

Bill Yuchen Lin, Wangchunshu Zhou, Ming Shen, Pei Zhou, Chandra Bhagavatula, Yejin Choi, and Xiang Ren. CommonGen: A constrained text generation challenge for generative commonsense reasoning. In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 1823– 1840, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/20 20.findings-emnlp.165. URL https://aclanthology.org/2020.findings-emnlp.165.

Bill Yuchen Lin, Seyeon Lee, Xiaoyang Qiao, and Xiang Ren. Common sense beyond English: Evaluating and improving multilingual language models for commonsense reasoning. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 1274–1287, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long.102. URL https://aclanthology.org/2021.acl-long.102.

Kevin Lin, Oyvind Tafjord, Peter Clark, and Matt Gardner. Reasoning over paragraph effects in situations. In EMNLP 2019 MRQA Workshop, pp. 58, 2019.

Xi Victoria Lin, Todor Mihaylov, Mikel Artetxe, Tianlu Wang, Shuohui Chen, Daniel Simig, Myle Ott, Naman Goyal, Shruti Bhosale, Jingfei Du, Ramakanth Pasunuru, Sam Shleifer, Punit Singh Koura, Vishrav Chaudhary, Brian O’Horo, Jeff Wang, Luke Zettlemoyer, Zornitsa Kozareva, Mona Diab, Veselin Stoyanov, and Xian Li. Few-shot learning with multilingual generative language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 9019–9052, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics. URL https://aclanthology.org/2022.emnlp-main.616.

Fangyu Liu, Emanuele Bugliarello, Edoardo Maria Ponti, Siva Reddy, Nigel Collier, and Desmond Elliott. Visually grounded reasoning across languages and cultures. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 10467–10485, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.818. URL https://aclanthology.org/2021.emnlp-main. 818.

Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V. Le, Barret Zoph, Jason Wei, and Adam Roberts. The flan collection: Designing data and methods for effective instruction tuning. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org, 2023a.

Shayne Longpre, Robert Mahari, Anthony Chen, Naana Obeng-Marnu, Damien Sileo, William Brannon, Niklas Muennighoff, Nathan Khazam, Jad Kabbara, Kartik Perisetla, Xinyi Wu, Enrico Shippole, Kurt Bollacker, Tongshuang Wu, Luis Villa, Sandy Pentland, and Sara Hooker. The data provenance initiative: A large scale audit of dataset licensing & attribution in ai, 2023b.

Shayne Longpre, Gregory Yauney, Emily Reif, Katherine Lee, Adam Roberts, Barret Zoph, Denny Zhou, Jason Wei, Kevin Robinson, David Mimno, et al. A pretrainer’s guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity. arXiv preprint arXiv:2305.13169, 2023c.

Yanni Alexander Loukissas. All Data Are Local: Thinking Critically in a Data-Driven Society. MIT Press, Cambridge, Massachusetts, 2019.

Lalita Lowphansirikul, Charin Polpanumas, Attapol T Rutherford, and Sarana Nutanong. A large English–Thai parallel corpus from the web and machine-generated text. Language Resources and Evaluation, 56(2):477–499, 2022.

Alexandra Luccioni and Joseph Viviano. What’s in the box? an analysis of undesirable content in the Common Crawl corpus. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pp. 182–189, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-short.24. URL https: //aclanthology.org/2021.acl-short.24.

Lucia Specia, Nicola Cancedda, and Marc Dymetman. A Dataset for Assessing Machine Translation Evaluation Metrics. International Conference on Language Resources and Evaluation, 2010.

Li Lucy, Suchin Gururangan, Luca Soldaini, Emma Strubell, David Bamman, Lauren Klein, and Jesse Dodge. Aboutme: Using self-descriptions in webpages to document the effects of english pretraining data filters, 2024.

Nils Lukas, A. Salem, Robert Sim, Shruti Tople, Lukas Wutschitz, and Santiago Zanella-B’eguelin. Analyzing leakage of personally identifiable information in language models. 2023 IEEE Symposium on Security and Privacy (SP), pp. 346–363, 2023. URL https://api.semanticscholar. org/CorpusID:256459554.

Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. Wizardcoder: Empowering code large language models with evol-instruct, 2023.

Risto Luukkonen, Ville Komulainen, Jouni Luoma, Anni Eskelinen, Jenna Kanerva, Hanna-Mari Kupari, Filip Ginter, Veronika Laippala, Niklas Muennighoff, Aleksandra Piktus, et al. Fingpt: Large generative models for a small language. arXiv preprint arXiv:2311.05640, 2023.

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pp. 142–150, Portland, Oregon, USA, June 2011. Association for Computational Linguistics. URL http: //www.aclweb.org/anthology/P11-1015.

Jean Maillard, Cynthia Gao, Elahe Kalbassi, Kaushik Ram Sadagopan, Vedanuj Goswami, Philipp Koehn, Angela Fan, and Francisco Guzmán. Small data, big impact: Leveraging minimal data for effective machine translation. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2740–2756, Toronto, Canada, 2023. Association for Computational Linguistics. URL https://aclanthology.org/2023.acl-long. 154.

Max Marion, Ahmet Üstün, Luiza Pozzobon, Alex Wang, Marzieh Fadaee, and Sara Hooker. When less is more: Investigating data pruning for pretraining llms at scale, 2023.

Vukosi Marivate, Tshephisho Sefara, Vongani Chabalala, Keamogetswe Makhaya, Tumisho Mokgonyane, Rethabile Mokoena, and Abiodun Modupe. Investigating an approach for low resource language dataset creation, curation and classification: Setswana and sepedi. In Proceedings of the first workshop on Resources for African Indigenous Languages, pp. 15–20, Marseille, France, May 2020. European Language Resources Association (ELRA). ISBN 979-10-95546-60-3. URL https://aclanthology.org/2020.rail-1.3.

maxbartolo. AdversarialQA D(BERT). https://huggingface.co/datasets/adversarial_qa/v iewer/dbert, 2023a. Accessed: 2023-11-28.

maxbartolo. AdversarialQA D(BiDAF). https://huggingface.co/datasets/adversarial_qa/v iewer/dbidaf, 2023b. Accessed: 2023-11-28.

maxbartolo. AdversarialQA D(RoBERTa). https://huggingface.co/datasets/adversarial_qa /viewer/droberta, 2023c. Accessed: 2023-11-28.

Mike Maxwell and Baden Hughes. Frontiers in linguistic annotation for lower-density languages. In Proceedings of the workshop on frontiers in linguistically annotated corpora 2006, pp. 29–37, 2006.

Stephen Mayhew, Terra Blevins, Shuheng Liu, Marek Šuppa, Hila Gonen, Joseph Marvin Imperial, Börje F. Karlsson, Peiqin Lin, Nikola Ljubešić, LJ Miranda, Barbara Plank, Arij Riabi, and Yuval Pinter. Universal NER: A Gold-Standard Multilingual Named Entity Recognition Benchmark. arXiv preprint arXiv:2311.09122, 2023.

Bryan McCann, Nitish Shirish Keskar, Caiming Xiong, and Richard Socher. The natural language decathlon: Multitask learning as question answering. arXiv preprint arXiv:1806.08730, 2018.

Ninareh Mehrabi, Fred Morstatter, Nripsuta Saxena, Kristina Lerman, and Aram Galstyan. A survey on bias and fairness in machine learning. ACM computing surveys (CSUR), 54(6):1–35, 2021.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In EMNLP, 2018.

Sewon Min, Mike Lewis, Luke Zettlemoyer, and Hannaneh Hajishirzi. Metaicl: Learning to learn in context. arXiv preprint arXiv:2110.15943, 2021.

Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi. Cross-task generalization via natural language crowdsourcing instructions. In ACL, 2022.

Steven Moran and Christian Chiarcos. 4 linguistic linked open data and under-resourced languages: From collection to application. Development of Linguistic Linked Open Data Resources for Collaborative Data-Intensive Research in the Language Sciences, pp. 39, 2020.

Benedikt Morschheuser, Juho Hamari, Jonna Koivisto, and Alexander Maedche. Gamified crowdsourcing: Conceptualization, literature review, and future agenda. International Journal of Human-Computer Studies, 106:26–43, 2017. ISSN 1071-5819. doi: https://doi.org/10.1016/ j.ijhcs.2017.04.005. URL https://www.sciencedirect.com/science/article/pii/S1071581 917300642.

Niklas Muennighoff, Qian Liu, Armel Zebaze, Qinkai Zheng, Binyuan Hui, Terry Yue Zhuo, Swayam Singh, Xiangru Tang, Leandro von Werra, and Shayne Longpre. Octopack: Instruction tuning code large language models. arXiv preprint arXiv:2308.07124, 2023a.

Niklas Muennighoff, Alexander M Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, and Colin Raffel. Scaling data-constrained language models. arXiv preprint arXiv:2305.16264, 2023b.

Niklas Muennighoff, Thomas Wang, Lintang Sutawika, Adam Roberts, Stella Biderman, Teven Le Scao, M Saiful Bari, Sheng Shen, Zheng Xin Yong, Hailey Schoelkopf, Xiangru Tang, Dragomir Radev, Alham Fikri Aji, Khalid Almubarak, Samuel Albanie, Zaid Alyafeai, Albert Webson, Edward Raff, and Colin Raffel. Crosslingual generalization through multitask finetuning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15991–16111, Toronto, Canada, July 2023c. Association for Computational Linguistics. URL https://aclanthology.org/2023.acl-long.891.

Shamsuddeen Muhammad, Idris Abdulmumin, Abinew Ayele, Nedjma Ousidhoum, David Adelani, Seid Yimam, Ibrahim Ahmad, Meriem Beloucif, Saif Mohammad, Sebastian Ruder, Oumaima Hourrane, Alipio Jorge, Pavel Brazdil, Felermino Ali, Davis David, Salomey Osei, Bello ShehuBello, Falalu Lawan, Tajuddeen Gwadabe, Samuel Rutunda, Tadesse Belay, Wendimu Messelle, Hailu Balcha, Sisay Chala, Hagos Gebremichael, Bernard Opoku, and Stephen Arthur. AfriSenti: A Twitter sentiment analysis benchmark for African languages. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 13968–13981, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.862. URL https://aclanthology.org/2023. emnlp-main.862.

Carol Myers-Scotton. Code-switching. The handbook of sociolinguistics, pp. 217–237, 2017. Ranjita Naik, Varun Chandrasekaran, Mert Yuksekgonul, Hamid Palangi, and Besmira Nushi. Di-

versity of thought improves reasoning abilities of large language models, 2023.

Gabriel Nakamura, Bruno Soares, Valério Pillar, José Diniz-Filho, and Leandro Duarte. Three pathways to better recognize the expertise of global south researchers. npj Biodiversity, 08 2023. doi: 10.1038/s44185-023-00021-7.

Ramesh Nallapati, Bowen Zhou, Cicero Nogueira dos santos, Caglar Gulcehre, and Bing Xiang. Abstractive text summarization using sequence-to-sequence rnns and beyond, 2016.

Tarek Naous, Michael Joseph Ryan, and Wei Xu. Having beer after prayer? measuring cultural bias in large language models. ArXiv, abs/2305.14456, 2023. URL https://api.semanticscho lar.org/CorpusID:258865272.

Shashi Narayan, Shay B. Cohen, and Mirella Lapata. Don’t give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. ArXiv, abs/1808.08745, 2018.

Milad Nasr, Nicholas Carlini, Jonathan Hayase, Matthew Jagielski, A. Feder Cooper, Daphne Ippolito, Christopher A. Choquette-Choo, Eric Wallace, Florian Tramèr, and Katherine Lee. Scalable extraction of training data from (production) language models, 2023.

Huu Nguyen, Sameer Suri, Ken Tsui, and Christoph Schuhmann. The open instruction generalist (oig) dataset. https://laion.ai/blog/oig-dataset/, 2023.

Jinjie Ni, Fuzhao Xue, Kabir Jain, Mahir Hitesh Shah, Zangwei Zheng, and Yang You. Instruction in the wild: A user-based instruction dataset. https://github.com/XueFuzhao/InstructionWild, 2023.

Gabriel Nicholas and Aliya Bhatia. Lost in translation: Large language models in non-english content analysis, 2023.

Joel Niklaus, Veton Matoshi, Matthias Stürmer, Ilias Chalkidis, and Daniel E Ho. Multilegalpile: A 689gb multilingual legal corpus. arXiv preprint arXiv:2306.02069, 2023.

NLLB-Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi, Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, and Jeff Wang. No language left behind: Scaling human-centered machine translation, 2022.

Gianluca Nogara, Francesco Pierri, Stefano Cresci, Luca Luceri, Petter Törnberg, and Silvia Giordano. Toxic bias: Perspective api misreads german as more toxic, 2023.

Odunayo Ogundepo, Tajuddeen Gwadabe, Clara Rivera, Jonathan Clark, Sebastian Ruder, David Adelani, Bonaventure Dossou, Abdou Diop, Claytone Sikasote, Gilles Hacheme, Happy Buzaaba, Ignatius Ezeani, Rooweither Mabuya, Salomey Osei, Chris Emezue, Albert Kahira, Shamsuddeen Muhammad, Akintunde Oladipo, Abraham Owodunni, Atnafu Tonja, Iyanuoluwa Shode, Akari Asai, Anuoluwapo Aremu, Ayodele Awokoya, Bernard Opoku, Chiamaka Chukwuneke, Christine Mwase, Clemencia Siro, Stephen Arthur, Tunde Ajayi, Verrah Otiende, Andre Rubungo, Boyd Sinkala, Daniel Ajisafe, Emeka Onwuegbuzia, Falalu Lawan, Ibrahim Ahmad, Jesujoba Alabi, Chinedu Mbonu, Mofetoluwa Adeyemi, Mofya Phiri, Orevaoghene Ahia, Ruqayya Iro, and Sonia Adhiambo. Afriqa: Cross-lingual open-retrieval question answering for african languages. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 14957–14972, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.997. URL https://aclanthology.org/2023.findings-emnlp.997.

Jessica Ojo, Kelechi Ogueji, Pontus Stenetorp, and David I. Adelani. How good are large language models on african languages?, 2023.

Pedro Javier Ortiz Su’arez, Benoit Sagot, and Laurent Romary. Asynchronous pipelines for processing huge corpora on medium to low resource infrastructures. In Piotr Bański, Adrien Barbaresi, Hanno Biber, Evelyn Breiteneder, Simon Clematide, Marc Kupietz, Harald L"ungen, and Caroline Iliadi (eds.), 7th Workshop on the Challenges in the Management of Large Corpora (CMLC-7), Proceedings of the Workshop on Challenges in the Management of Large Corpora (CMLC-7) 2019. Cardiff, 22nd July 2019, pp. 9 – 16, Mannheim, 2019. Leibniz-Institut f"ur Deutsche Sprache. doi: 10.14618/ids-pub-9021. URL http://nbn-resolving.de/urn:nbn:de:bsz:mh39-90215.

osyvokon. UA-GEC instruction tuning . https://huggingface.co/datasets/osyvokon/ua_gec_ instruction_tuning, 2023. Accessed: 2023-11-28.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022.

Bo Pang and Lillian Lee. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In Proceedings of the ACL, 2005.

Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Quan Ngoc Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. The lambada dataset: Word prediction requiring a broad discourse context, 2016.

Michael Park, Erin Leahey, and Russell J. Funk. Papers and patents are becoming less disruptive over time. Nature, 613:138–144, 2023. URL https://api.semanticscholar.org/CorpusID: 255466666.

Kenny Peng, Arunesh Mathur, and Arvind Narayanan. Mitigating dataset harms requires stewardship: Lessons from 1000 papers. arXiv preprint arXiv:2108.02922, 2021.

Laura Perez-Beltrachini and Mirella Lapata. Models and datasets for cross-lingual summarisation. In Proceedings of The 2021 Conference on Empirical Methods in Natural Language Processing, Punta Cana, Dominican Republic, 2021.

Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick S. H. Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Rocktäschel, and Sebastian Riedel. KILT: a benchmark for knowledge intensive language tasks. In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tür, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou (eds.), Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021, pp. 2523–2544. Association for Computational Linguistics, 2021. URL https://www.aclweb.org/anthology/2 021.naacl-main.200/.

Martin Petty. Explainer: Why is Myanmar’s military holding an election?, 2023. URL https: //www.reuters.com/world/asia-pacific/why-is-myanmars-military-holding-an-electio n-2023-03-29/. Accessed on Jan. 17, 2024.

Mohammad Taher Pilehvar and Jose Camacho-Collados. Wic: the word-in-context dataset for evaluating context-sensitive meaning representations, 2019.

McKevitt C Pinel C, Prainsack B. Caring for data: Value creation in a data-intensive research laboratory. Social Studies of Science, 50(2):175–197, April 2020.

Edoardo Maria Ponti, Goran Glavaš, Olga Majewska, Qianchu Liu, Ivan Vulić, and Anna Korhonen. Xcopa: A multilingual dataset for causal commonsense reasoning. arXiv preprint arXiv:2005.00333, 2020.

Maja Popović. chrF: character n-gram F-score for automatic MT evaluation. In Proceedings of the Tenth Workshop on Statistical Machine Translation, pp. 392–395, Lisbon, Portugal, September 2015. Association for Computational Linguistics. doi: 10.18653/v1/W15-3049. URL https: //aclanthology.org/W15-3049.

Matt Post. A call for clarity in reporting BLEU scores. In Proceedings of the Third Conference on Machine Translation: Research Papers, pp. 186–191, Brussels, Belgium, October 2018. Association for Computational Linguistics. doi: 10.18653/v1/W18-6319. URL https://aclanthology.org/W18-6319.

Adithya Pratapa, Rishubh Gupta, and Teruko Mitamura. Multilingual event linking to Wikidata. In Proceedings of the Workshop on Multilingual Information Access (MIA), pp. 37–58, Seattle, USA, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.mia-1.5. URL https://aclanthology.org/2022.mia-1.5.

Danti Pudjiati, Ninuk Lustyantie, Ifan Iskandar, and Tira Nur Fitria. Post-editing of machine translation: Creating a better translation of cultural specific terms. Language Circle: Journal of Language and Literature, 17(1):61–73, 2022.

Cornelius Puschmann and Jean Burgess. Big data, big questions| metaphors of big data. International Journal of Communication, 8(0), 2014.

Mahima Pushkarna, Andrew Zaldivar, and Oddur Kjartansson. Data cards: Purposeful and transparent dataset documentation for responsible ai. In Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’22, pp. 1776–1826, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450393522. doi: 10.1145/3531146.3533 231. URL https://doi.org/10.1145/3531146.3533231.

PyThaiNLP. scb_mt_2020_en2th_prompt. https://huggingface.co/datasets/pythainlp/sc b_mt_2020_en2th_prompt, 2023a. Accessed: 2023-11-29.

PyThaiNLP. scb_mt_2020_th2en_prompt. https://huggingface.co/datasets/pythainlp/sc b_mt_2020_th2en_prompt, 2023b. Accessed: 2023-11-29.

PyThaiNLP. Thai-Pos-prompt. https://huggingface.co/datasets/pythainlp/Thai-Pos-pro mpt, 2023c. Accessed: 2023-11-29.

PyThaiNLP. thai_usembassy_en2th_prompt. https://huggingface.co/datasets/pythainlp/ thai_usembassy_en2th_prompt, 2023d. Accessed: 2023-11-29.

PyThaiNLP. thai_usembassy_th2en_prompt. https://huggingface.co/datasets/pythainlp/ thai_usembassy_th2en_prompt, 2023e. Accessed: 2023-11-29.

PyThaiNLP. thai-wiktionary-prompt. https://huggingface.co/datasets/pythainlp/thai-wik tionary-prompt, 2023f. Accessed: 2023-11-29.

Alessandro Raganato, Tommaso Pasini, Jose Camacho-Collados, and Mohammad Taher Pilehvar. XL-WiC: A multilingual benchmark for evaluating semantic contextualization. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 7193–7206, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v 1/2020.emnlp-main.584. URL https://aclanthology.org/2020.emnlp-main.584.

Nazneen Fatema Rajani, Bryan McCann, Caiming Xiong, and Richard Socher. Explain yourself! leveraging language models for commonsense reasoning. In Proceedings of the 2019 Conference of the Association for Computational Linguistics (ACL2019), 2019. URL https://arxiv.org/ abs/1906.02361.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+ Questions for Machine Comprehension of Text. arXiv e-prints, art. arXiv:1606.05250, 2016.

Siva Reddy, Danqi Chen, and Christopher D. Manning. CoQA: A conversational question answering challenge. Transactions of the Association for Computational Linguistics, 7:249–266, 2019. doi: 10.1162/tacl_a_00266. URL https://aclanthology.org/Q19-1016.

Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084, 2019.

Reuters. Explainer: What is happening between Armenia and Azerbaijan over Nagorno-Karabakh?,

2023. URL https://www.reuters.com/world/what-is-happening-between-armenia-azerb aijan-over-nagorno-karabakh-2023-09-19/. Accessed on Jan. 17, 2024.

John R Rickford, Julie Sweetland, Angela E Rickford, and Thomas Grano. African American, Creole, and other vernacular Englishes in education: A bibliographic resource. Routledge, 2012.

Nathaniel R. Robinson, Perez Ogayo, David R. Mortensen, and Graham Neubig. Chatgpt mt: Competitive for high- (but not low-) resource languages. ArXiv, abs/2309.07423, 2023. URL https://api.semanticscholar.org/CorpusID:261824661.

Anna Rogers, Olga Kovaleva, Matthew Downey, and Anna Rumshisky. Getting closer to AI complete question answering: A set of prerequisite real tasks. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020, pp. 8722–8731. AAAI Press, 2020. URL https://aaai.org/ojs/index.php/AAAI/article/view/6398.

Raf Van Rooy. Language or Dialect? The History of a Conceptual Pair. Oxford University Press, 2021.

Alexander M. Rush, Sumit Chopra, and Jason Weston. A neural attention model for abstractive sentence summarization. Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, 2015. doi: 10.18653/v1/d15-1044. URL http://dx.doi.org/10.18653/v1 /D15-1044.

Jon Saad-Falcon, Joe Barrow, Alexa Siu, Ani Nenkova, Ryan A Rossi, and Franck Dernoncourt. Pdftriage: Question answering over long, structured documents. arXiv preprint arXiv:2309.08872, 2023.

Marta Sabou, Kalina Bontcheva, and Arno Scharl. Crowdsourcing research opportunities: Lessons from natural language processing. In Proceedings of the 12th International Conference on Knowledge Management and Knowledge Technologies, i-KNOW ’12, New York, NY, USA, 2012. Association for Computing Machinery. ISBN 9781450312424. doi: 10.1145/2362456.2362479. URL https://doi.org/10.1145/2362456.2362479.

Amrita Saha, Rahul Aralikatte, Mitesh M. Khapra, and Karthik Sankaranarayanan. DuoRC: Towards Complex Language Understanding with Paraphrased Reading Comprehension. In Meeting of the Association for Computational Linguistics (ACL), 2018.

Nithya Sambasivan, Shivani Kapania, Hannah Highfill, Diana Akrong, Praveen Paritosh, and Lora M Aroyo. “everyone wants to do the model work, not the data work”: Data cascades in highstakes ai. In Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, CHI ’21, New York, NY, USA, 2021. Association for Computing Machinery. ISBN 9781450380966. doi: 10.1145/3411764.3445518. URL https://doi.org/10.1145/3411764.3445518.

Victor Sanh, Albert Webson, Colin Raffel, Stephen Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, M Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Fevry, Jason Alan Fries, Ryan Teehan, Teven Le Scao, Stella Biderman, Leo Gao, Thomas Wolf, and Alexander M Rush. Multitask prompted training enables zero-shot task generalization. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=9Vrb9D0WI4.

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. Socialiqa: Commonsense reasoning about social interactions, 2019.

Beatrice Savoldi, Marco Gaido, Luisa Bentivogli, Matteo Negri, and Marco Turchi. Gender bias in machine translation, 2021.

Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al. Bloom: A 176bparameter open-access multilingual language model. arXiv preprint arXiv:2211.05100, 2022a.

Teven Le Scao, Thomas Wang, Daniel Hesslow, Lucile Saulnier, Stas Bekman, M Saiful Bari, Stella Bideman, Hady Elsahar, Niklas Muennighoff, Jason Phang, et al. What language model to train if you have one million gpu hours? arXiv preprint arXiv:2210.15424, 2022b.

Reva Schwartz, Apostol Vassilev, Kristen Greene, Lori Perine, Andrew Burt, Patrick Hall, et al. Towards a standard for identifying and managing bias in artificial intelligence. NIST special publication, 1270(10.6028), 2022.

Nick Seaver. Care and scale: Decorrelative ethics in algorithmic recommendation. Cultural Anthropology, 36(3):509–537, 2021.

UN Secretariat. International decade of indigenous languages, 2022–2032: Global action plan: note/by the secretariat, 2022.

Abigail See, Peter J. Liu, and Christopher D. Manning. Get to the point: Summarization with pointer-generator networks. CoRR, abs/1704.04368, 2017. URL http://arxiv.org/abs/1704.0 4368.

Priyanka Sen, Alham Fikri Aji, and Amir Saffari. Mintaka: A complex, natural, and multilingual dataset for end-to-end question answering. In Proceedings of the 29th International Conference on Computational Linguistics, pp. 1604–1619, Gyeongju, Republic of Korea, October 2022. International Committee on Computational Linguistics. URL https://aclanthology.org/2022.co ling-1.138.

Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words with subword units. In Proceedings of the 54th Annual Meeting of the Association for Computational

Linguistics (Volume 1: Long Papers), pp. 1715–1725, Berlin, Germany, August 2016. Association for Computational Linguistics. doi: 10.18653/v1/P16-1162. URL https://aclanthology.org /P16-1162.

Shafagh. Aya Persian Instruction pn Summary. https://huggingface.co/datasets/Shafagh/ay a_persian_instruction_pn-summary, 2023a. Accessed: 2023-11-28.

Shafagh. Aya Persian Instruction pn Summary Title. https://huggingface.co/datasets/Shaf agh/aya_persian_instruction_pn-summary-title, 2023b. Accessed: 2023-11-28.

Chih Chieh Shao, Trois Liu, Yuting Lai, Yiying Tseng, and Sam Tsai. Drcd: a chinese machine reading comprehension dataset, 2019.

Jack Sidnell and N. J. Enfield. Language diversity and social action: A third locus of linguistic relativity. Current Anthropology, 53(3):302–333, 2012.

Kathleen Siminyu, Godson Kalipe, Davor Orlic, Jade Z. Abbott, Vukosi Marivate, Sackey Freshia, Prateek Sibal, Bhanu Neupane, David Ifeoluwa Adelani, Amelia V. Taylor, Jamiil Toure Ali, Kevin Degila, Momboladji Balogoun, Thierno Ibrahima Diop, Davis David, Chayma Fourati, Hatem Haddad, and Malek Naski. AI4D - african language program. In Kathleen Siminyu, Julia Kreutzer, Hady Elsahar, Vukosi Marivate, Nishant Subramani, Jade Z. Abbott, and Bernardt Duvenhage (eds.), 2nd AfricaNLP Workshop Proceedings, AfricaNLP@EACL 2021, Virtual Event, April 19, 2021, 2021. URL https://arxiv.org/abs/2104.02516.

Gary F. Simons. Two centuries of spreading language loss. Proceedings of the Linguistic Society of America, 4(1):27:1–12, Mar. 2019. doi: 10.3765/plsa.v4i1.4532. URL https://journals.lingu isticsociety.org/proceedings/index.php/PLSA/article/view/4532.

Amanpreet Singh, Mike D’Arcy, Arman Cohan, Doug Downey, and Sergey Feldman. SciRepEval: A Multi-Format Benchmark for Scientific Document Representations. ArXiv, abs/2211.13308, 2022.

Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, Valentin Hofmann, Ananya Harsh Jha, Sachin Kumar, Li Lucy, Xinxi Lyu, Ian Magnusson, Jacob Morrison, Niklas Muennighoff, Aakanksha Naik, Crystal Nam, Matthew E. Peters, Abhilasha Ravichander, Kyle Richardson, Zejiang Shen, Emma Strubell, Nishant Subramani, Oyvind Tafjord, Evan Pete Walsh, Hannaneh Hajishirzi, Noah A. Smith, Luke Zettlemoyer, Iz Beltagy, Dirk Groeneveld, Jesse Dodge, and Kyle Lo. Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research. arXiv preprint, 2024.

Lucia Specia and Atefeh Farzindar. Estimating machine translation post-editing effort with HTER. In Proceedings of the Second Joint EM+/CNGL Workshop: Bringing MT to the User: Research on Integrating MT in the Translation Industry, pp. 33–43, Denver, Colorado, USA, November 4 2010. Association for Machine Translation in the Americas. URL https://aclanthology.org/2 010.jec-1.5.

Bernard Spolsky. Language policy in french colonies and after independence. Current Issues in Language Planning, 19(3):231–315, 2018. doi: 10.1080/14664208.2018.1444948.

Vivek Srivastava and Mayank Singh. Challenges and considerations with code-mixed nlp for multilingual societies, 2021.

Luke Stark and Anna Lauren Hoffmann. Data is the new what? popular metaphors & professional ethics in emerging data culture. Journal of Cultural Analytics, 4(1), 2019.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33:3008–3021, 2020.

Stephanie Strassel and Jennifer Tracey. LORELEI language packs: Data, tools, and resources for technology development in low resource languages. In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16), pp. 3273–3280, Portorož, Slovenia, May 2016. European Language Resources Association (ELRA). URL https://aclanthology.o rg/L16-1521.

Jiao Sun, Thibault Sellam, Elizabeth Clark, Tu Vu, Timothy Dozat, Dan Garrette, Aditya Siddhant, Jacob Eisenstein, and Sebastian Gehrmann. Dialect-robust evaluation of generated text. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 6010–6028, Toronto, Canada, July 2023. Association for Computational Linguistics. URL https://aclanthology.org/2023.acl-long.331.

SuryaKrishna02. Aya Telugu Food Recipes. https://huggingface.co/datasets/SuryaKrishna 02/aya-telugu-food-recipes, 2023a. Accessed: 2023-11-28.

SuryaKrishna02. Aya Telugu Jokes. https://huggingface.co/datasets/SuryaKrishna02/aya-t elugu-jokes, 2023b. Accessed: 2023-11-28.

SuryaKrishna02. Aya Telugu News Articles. https://huggingface.co/datasets/SuryaKrishna 02/aya-telugu-news-articles, 2023c. Accessed: 2023-11-28.

SuryaKrishna02. Aya Telugu Paraphrase. https://huggingface.co/datasets/SuryaKrishna02 /aya-telugu-paraphrase, 2023d. Accessed: 2023-11-28.

SuryaKrishna02. Aya Telugu Poems. https://huggingface.co/datasets/SuryaKrishna02/aya

-telugu-poems, 2023e. Accessed: 2023-11-28.

Masahiro Suzuki, Masanori Hirano, and Hiroki Sakaji. From base to conversational: Japanese instruction dataset and tuning large language models. arXiv preprint arXiv:2309.03412, 2023.

syntaxshill. Arpa aya. https://huggingface.co/datasets/syntaxshill/arpa-aya, 2023. Accessed: 2023-11-28.

Oleksiy Syvokon, Olena Nahorna, Pavlo Kuchmiichuk, and Nastasiia Osidach. UA-GEC: Grammatical error correction and fluency corpus for the Ukrainian language. In Proceedings of the Second Ukrainian Natural Language Processing Workshop (UNLP), pp. 96–102, Dubrovnik, Croatia, May 2023. Association for Computational Linguistics. URL https://aclanthology.org/2023.unlp -1.12.

Oyvind Tafjord, Peter Clark, Matt Gardner, Wen-tau Yih, and Ashish Sabharwal. Quarel: A dataset and models for answering questions about qualitative relationships. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 7063–7071, 2019a.

Oyvind Tafjord, Matt Gardner, Kevin Lin, and Peter Clark. Quartz: An open-domain dataset of qualitative relationship questions. arXiv preprint arXiv:1909.03553, 2019b.

TahmidH. Annotated News Summary. https://huggingface.co/datasets/TahmidH/annotate d_news_summary, 2023. Accessed: 2023-11-28.

Niket Tandon, Bhavana Dalvi Mishra, Keisuke Sakaguchi, Antoine Bosselut, and Peter Clark. Wiqa: A dataset for "what if..." reasoning over procedural text. arXiv:1909.04739v1, 2019.

Yuqing Tang, Chau Tran, Xian Li, Peng-Jen Chen, Naman Goyal, Vishrav Chaudhary, Jiatao Gu, and Angela Fan. Multilingual translation from denoising pre-training. Findings of the Association for Computational Linguistics: ACL-IJCNLP, 2021. doi: 10.18653/v1/2021.findings-acl.304.

Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_alpaca, 2023.

Tellarin.ai. LLM Japanese Dataset Vanilla Aya Format. https://huggingface.co/datasets/te llarin-ai/llm-japanese-dataset-vanilla-aya-format, 2023a. Accessed: 2023-11-28.

Tellarin.ai. NTX LLM Instructions. https://huggingface.co/datasets/tellarin-ai/ntx_llm_ instructions, 2023b. Accessed: 2023-11-28.

theblackcat102. Joke explaination. https://huggingface.co/datasets/theblackcat102/joke_ explaination, 2023. Accessed: 2023-11-29.

Jörg Tiedemann. Parallel data, tools and interfaces in opus. In Nicoletta Calzolari (Conference Chair), Khalid Choukri, Thierry Declerck, Mehmet Ugur Dogan, Bente Maegaard, Joseph Mariani, Jan Odijk, and Stelios Piperidis (eds.), Proceedings of the Eight International Conference on Language Resources and Evaluation (LREC’12), Istanbul, Turkey, may 2012. European Language Resources Association (ELRA). ISBN 978-2-9517408-7-7.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023.

Ming Tu, Guangtao Wang, Jing Huang, Yun Tang, Xiaodong He, and Bowen Zhou. Multi-hop reading comprehension across multiple documents by reasoning over heterogeneous graphs, 2019.

TurkuNLP. Turku Paraphrase Corpus. https://huggingface.co/datasets/TurkuNLP/turku_p araphrase_corpus, 2023. Accessed: 2023-11-28.

Universal NER. UNER LLM Instructions. https://huggingface.co/datasets/universalner/u ner_llm_instructions, 2023. Accessed: 2023-11-28.

Gorka Urbizu, Iñaki San Vicente, Xabier Saralegi, and Ander Corral. Not enough data to pre-train your language model? MT to the rescue! In Findings of the Association for Computational Linguistics: ACL 2023, pp. 3826–3836, Toronto, Canada, July 2023. Association for Computational Linguistics. URL https://aclanthology.org/2023.findings-acl.235.

Eva Vanmassenhove, Dimitar Shterionov, and Matthew Gwilliam. Machine translationese: Effects of algorithmic bias on linguistic complexity in machine translation. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pp. 2203–2213, Online, April 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.eacl-main.188. URL https://aclanthology.org/2021.eacl-main.188.

Aniket Vashishtha, Kabir Ahuja, and Sunayana Sitaram. On evaluating and mitigating gender biases in multilingual settings, 2023.

Cécile B. Vigouroux. Francophonie. Annual Review of Anthropology, 42(1):379–397, 2013. doi: 10.1146/annurev-anthro-092611-145804. URL https://doi.org/10.1146/annurev-anthro-0 92611-145804.

Alexander Wan, Eric Wallace, Sheng Shen, and Dan Klein. Poisoning language models during instruction tuning, 2023.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461, 2018.

Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. arXiv preprint arXiv:1905.00537, 2019.

Guangyu Wang, Guoxing Yang, Zongxin Du, Longjun Fan, and Xiaohu Li. Clinicalgpt: Large language models finetuned with diverse medical data and comprehensive evaluation, 2023.

Jun Wang, Benjamin Rubinstein, and Trevor Cohn. Measuring and mitigating name biases in neural machine translation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2576–2590, Dublin, Ireland, May 2022a. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.184. URL https: //aclanthology.org/2022.acl-long.184.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language model with self generated instructions. ArXiv preprint, abs/2212.10560, 2022b. URL https://arxiv.org/abs/2212.10560.

Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Gary Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Maitreya Patel, Kuntal Kumar Pal, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Shailaja Keyur Sampat, Savan Doshi, Siddhartha Mishra, Sujan Reddy, Sumanta Patro, Tanay Dixit, Xudong Shen, Chitta Baral, Yejin Choi, Noah A. Smith, Hannaneh Hajishirzi, and Daniel Khashabi. Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks, 2022c.

Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Anjana Arunkumar, Arjun Ashok, Arut Selvan Dhanasekaran, Atharva Naik, David Stap, et al. Super-naturalinstructions: Generalization via declarative instructions on 1600+ nlp tasks. arXiv preprint arXiv:2204.07705, 2022d.

Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, Eshaan Pathak, Giannis Karamanolakis, Haizhi Lai, Ishan Purohit, Ishani Mondal, Jacob Anderson, Kirby Kuznia, Krima Doshi, Kuntal Kumar Pal, Maitreya Patel, Mehrad Moradshahi, Mihir Parmar, Mirali Purohit, Neeraj Varshney, Phani Rohitha Kaza, Pulkit Verma, Ravsehaj Singh Puri, Rushang Karia, Savan Doshi, Shailaja Keyur Sampat, Siddhartha Mishra, Sujan Reddy A, Sumanta Patro, Tanay Dixit, and Xudong Shen. Super-NaturalInstructions: Generalization via declarative instructions on 1600+ NLP tasks. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 5085–5109, Abu Dhabi, United Arab Emirates, December 2022e. Association for Computational Linguistics. URL https://aclanthology.org/2022.emnlp-main.340.

Alex Warstadt, Amanpreet Singh, and Samuel R Bowman. Neural network acceptability judgments. arXiv preprint arXiv:1805.12471, 2018.

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. Finetuned language models are zero-shot learners. In International Conference on Learning Representations, 2022a. URL https://openreview.net/forum?i d=gEZrGCozdqR.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022b.

Xiangpeng Wei, Haoran Wei, Huan Lin, Tianhao Li, Pei Zhang, Xingzhang Ren, Mei Li, Yu Wan, Zhiwei Cao, Binbin Xie, et al. Polylm: An open source polyglot large language model. arXiv preprint arXiv:2307.06018, 2023.

Johannes Welbl, Nelson F Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. arXiv preprint arXiv:1707.06209, 2017.

Richard E West, Timothy Newby, Zui Cheng, Alyssa Erickson, and Kyle Clements. Acknowledging all learning: Alternative, micro, and open credentials. Handbook of Research in Educational Communications and Technology: Learning Design, pp. 593–613, 2020.

Chenxi Whitehouse, Monojit Choudhury, and Alham Fikri Aji. Llm-powered data augmentation for enhanced crosslingual performance. arXiv preprint arXiv:2305.14288, 2023.

W Bruce Willis. The Adinkra dictionary: A visual primer on the language of Adinkra. Pyramid Complex, 1998.

Joseph Wilson. Voicing an algorithm: trials of strength in artificial intelligence research. Anthropology News, 2023. URL https://www.anthropology-news.org/articles/combat-by-algor ithm-trials-of-strength-in-artificial-intelligence-research/.

Genta Winata, Alham Fikri Aji, Zheng Xin Yong, and Thamar Solorio. The decades progress on code-switching research in NLP: A systematic survey on trends and challenges. In Findings of

the Association for Computational Linguistics: ACL 2023, pp. 2936–2978, Toronto, Canada, July 2023a. Association for Computational Linguistics. URL https://aclanthology.org/2023.fi ndings-acl.185.

Genta Indra Winata, Alham Fikri Aji, Samuel Cahyawijaya, Rahmad Mahendra, Fajri Koto, Ade Romadhony, Kemal Kurniawan, David Moeljadi, Radityo Eko Prasojo, Pascale Fung, Timothy Baldwin, Jey Han Lau, Rico Sennrich, and Sebastian Ruder. NusaX: Multilingual parallel sentiment dataset for 10 Indonesian local languages. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pp. 815–834, Dubrovnik, Croatia, May 2023b. Association for Computational Linguistics. URL https: //aclanthology.org/2023.eacl-main.57.

Peter Wittenburg. Open Science and Data Science. Data Intelligence, 3(1):95–105, 02 2021. Sam Witteveen and Martin Andrews. Paraphrasing with large language models. arXiv preprint

arXiv:1911.09661, 2019.

Walt Wolfram. Issues in dialect obsolescence: An introduction. American speech, 72(1):3–11, 1997. Chorng-Guang Wu, James H Gerlach, and Clifford E Young. An empirical analysis of open source

software developers’ motivations and continuance intentions. Information & Management, 44(3): 253–262, 2007.

Jeff Wu, Long Ouyang, Daniel M Ziegler, Nisan Stiennon, Ryan Lowe, Jan Leike, and Paul Christiano. Recursively summarizing books with human feedback. arXiv preprint arXiv:2109.10862, 2021.

Tianbao Xie, Chen Henry Wu, Peng Shi, Ruiqi Zhong, Torsten Scholak, Michihiro Yasunaga, Chien-Sheng Wu, Ming Zhong, Pengcheng Yin, Sida I Wang, et al. Unifiedskg: Unifying and multi-tasking structured knowledge grounding with text-to-text language models. arXiv preprint arXiv:2201.05966, 2022.

Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions, 2023a.

Jiashu Xu, Mingyu Derek Ma, Fei Wang, Chaowei Xiao, and Muhao Chen. Instructions as backdoors: Backdoor vulnerabilities of instruction tuning for large language models, 2023b.

Liang Xu, Hai Hu, Xuanwei Zhang, Lu Li, Chenjie Cao, Yudong Li, Yechen Xu, Kai Sun, Dian Yu, Cong Yu, Yin Tian, Qianqian Dong, Weitang Liu, Bo Shi, Yiming Cui, Junyi Li, Jun Zeng, Rongzhao Wang, Weijian Xie, Yanting Li, Yina Patterson, Zuoyu Tian, Yiwen Zhang, He Zhou, Shaoweihua Liu, Zhe Zhao, Qipeng Zhao, Cong Yue, Xinrui Zhang, Zhengliang Yang, Kyle Richardson, and Zhenzhong Lan. CLUE: A Chinese language understanding evaluation benchmark. In Proceedings of the 28th International Conference on Computational Linguistics, pp. 4762–4772, Barcelona, Spain (Online), December 2020. International Committee on Computational Linguistics. doi: 10.18653/v1/2020.coling-main.419. URL https://aclanthology.org/2020.coling-main.419.

Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. mT5: A massively multilingual pre-trained text-to-text transformer. In Kristina Toutanova, Anna Rumshisky, Luke Zettlemoyer, Dilek Hakkani-Tur, Iz Beltagy, Steven Bethard, Ryan Cotterell, Tanmoy Chakraborty, and Yichao Zhou (eds.), Proceedings of the 2021

Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 483–498, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.41. URL https://aclanthology.org/2021.na acl-main.41.

Shuo Yang, Zeke Xie, Hanyu Peng, Min Xu, Mingming Sun, and Ping Li. Dataset pruning: Reducing training data by examining generalization influence, 2023.

Yi Yang, Wen-tau Yih, and Christopher Meek. WikiQA: A challenge dataset for open-domain question answering. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pp. 2013–2018, Lisbon, Portugal, September 2015. Association for Computational Linguistics. doi: 10.18653/v1/D15-1237. URL https://aclanthology.org/D15-1237.

Yinfei Yang, Yuan Zhang, Chris Tar, and Jason Baldridge. PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification. In Proc. of EMNLP, 2019.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018.

Qinyuan Ye, Bill Yuchen Lin, and Xiang Ren. CrossFit: A few-shot learning challenge for crosstask generalization in NLP. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 7163–7189, Online and Punta Cana, Dominican Republic, 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.572. URL https://aclanthology.org/2021.emnlp-main.572.

Zheng Xin Yong, Cristina Menghini, and Stephen Bach. Low-resource languages jailbreak GPT-4. In Socially Responsible Language Modelling Research, 2023a. URL https://openreview.net/f orum?id=pn83r8V2sv.

Zheng Xin Yong, Hailey Schoelkopf, Niklas Muennighoff, Alham Fikri Aji, David Ifeoluwa Adelani, Khalid Almubarak, M Saiful Bari, Lintang Sutawika, Jungo Kasai, Ahmed Baruwa, Genta Winata, Stella Biderman, Edward Raff, Dragomir Radev, and Vassilina Nikoulina. BLOOM+1: Adding language support to BLOOM for zero-shot prompting. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 11682–11703, Toronto, Canada, July 2023b. Association for Computational Linguistics. URL https://aclanthology.org/2023.acl-long.653.

Zheng Xin Yong, Ruochen Zhang, Jessica Zosa Forde, Skyler Wang, Arjun Subramonian, Holy Lovenia, Samuel Cahyawijaya, Genta Indra Winata, Lintang Sutawika, Jan Christian Blaise Cruz, et al. Prompting multilingual large language models to generate code-mixed texts: The case of south east asian languages. In Sixth Workshop on Computational Approaches to Linguistic CodeSwitching, 2023c.

Yue Yu, Yuchen Zhuang, Jieyu Zhang, Yu Meng, Alexander Ratner, Ranjay Krishna, Jiaming Shen, and Chao Zhang. Large language model as attributed training data generator: A tale of diversity and bias, 2023.

Ann Yuan, Daphne Ippolito, Vitaly Nikolaev, Chris Callison-Burch, Andy Coenen, and Sebastian Gehrmann. Synthbio: A case study in human-ai collaborative curation of text datasets. CoRR, abs/2111.06467, 2021. URL https://arxiv.org/abs/2111.06467.

Tajudeen Yusuf. Politeness in arabic and yoruba: Personal pronouns as a case study. Asian Journal of Language, Literature and Culture Studies, 5(2):82–88, 2022.

Marcos Zampieri, Shervin Malmasi, Nikola Ljubešić, Preslav Nakov, Ahmed Ali, Jörg Tiedemann, Yves Scherrer, and Noëmi Aepli. Findings of the VarDial evaluation campaign 2017. In Proceedings of the Fourth Workshop on NLP for Similar Languages, Varieties and Dialects (VarDial), pp. 1– 15, Valencia, Spain, April 2017. Association for Computational Linguistics. doi: 10.18653/v1/ W17-1201. URL https://aclanthology.org/W17-1201.

Marcos Zampieri, Preslav Nakov, and Yves Scherrer. Natural language processing for similar languages, varieties, and dialects: A survey. Natural Language Engineering, 26(6):595–612, 2020.

Daniel Zhang, Nestor Maslej, Erik Brynjolfsson, John Etchemendy, Terah Lyons, James Manyika, Helen Ngo, Juan Carlos Niebles, Michael Sellitto, Ellie Sakhaee, Yoav Shoham, Jack Clark, and Raymond Perrault. The ai index 2022 annual report, 2022.

Ge Zhang, Yemin Shi, Ruibo Liu, Ruibin Yuan, Yizhi Li, Siwei Dong, Yu Shu, Zhaoqun Li, Zekun Wang, Chenghua Lin, Wenhao Huang, and Jie Fu. Chinese open instruction generalist: A preliminary release, 2023a.

Shaolei Zhang, Qingkai Fang, Zhuocheng Zhang, Zhengrui Ma, Yan Zhou, Langlin Huang, Mengyu Bu, Shangtong Gui, Yunji Chen, Xilin Chen, and Yang Feng. Bayling: Bridging cross-lingual alignment and instruction following through interactive translation for large language models, 2023b.

Sheng Zhang, Xiaodong Liu, Jingjing Liu, Jianfeng Gao, Kevin Duh, and Benjamin Van Durme. Record: Bridging the gap between human and machine commonsense reading comprehension, 2018a.

Wenxuan Zhang, Sharifah Mahani Aljunied, Chang Gao, Yew Ken Chia, and Lidong Bing. M3exam: A multilingual, multimodal, multilevel benchmark for examining large language models, 2023c.

Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc., 2015. URL https://proceedings.neurips.cc/paper_files/paper/2015/file/250cf8b51c773f3f8 dc8b4be867a9a02-Paper.pdf.

Yuan Zhang, Jason Baldridge, and Luheng He. Paws: Paraphrase adversaries from word scrambling,

- 2019.

Zhirui Zhang, Shujie Liu, Mu Li, Ming Zhou, and Enhong Chen. Joint training for neural machine translation models with monolingual data. CoRR, abs/1803.00353, 2018b. URL http://arxiv. org/abs/1803.00353.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. Lima: Less is more for alignment. arXiv preprint arXiv:2305.11206, 2023.

Terry Yue Zhuo, Armel Zebaze, Nitchakarn Suppattarachai, Leandro von Werra, Harm de Vries, Qian Liu, and Niklas Muennighoff. Astraios: Parameter-efficient instruction tuning code large language models. arXiv preprint arXiv:2401.00788, 2024.

Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences, 2020.

Barret Zoph, Deniz Yuret, Jonathan May, and Kevin Knight. Transfer learning for low-resource neural machine translation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 1568–1575, Austin, Texas, November 2016. Association for Computational Linguistics. doi: 10.18653/v1/D16-1163. URL https://aclanthology.org/D16 -1163.

##### A Aya Annotation Platform

In this section, we discuss the detailed design and development of the Aya Annotation Platform and the gamification strategy employed. Together, these attempts aimed to ensure high-quality curation of the Aya Dataset.

[Figure 44]

Figure 17: The onboarding page for a new user. We collect some demographic information and ask them to specify the languages they are comfortable annotating data in. These are the language options they are presented with later on in the UI.

###### A.1 Engagement Strategies

We decided to employ gamification methods to enhance annotator engagement [de Franga et al., 2015; Bastanfard et al., 2023; Morschheuser et al., 2017]. Our strategy involved using a points-based rewards system, motivating contributors through social media recognition, and fostering friendly competition with leaderboards. Regular mini-challenges and sprints helped to create collective achievement goals and fostered a sense of community [Bastanfard et al., 2023]. Real-time feedback reinforced positive behavior and customization options, such as avatars, provided a personalized experience. Overall, these gamification strategies aimed to boost engagement, improve data quality, and created a more enjoyable experience for the crowd-sourcing participants. [Morschheuser et al.,

- 2017].

To recognize and incentivize the efforts of our contributors, we established a tiered reward system based on contribution milestones: 500, 1,000, and 5,000 contributions. Contributors who achieved these goals on the project leaderboard were rewarded with certificates and specially designed Aya project apparel. The attire varied according to the contribution level, with different packages for each milestone. Additionally, the most active contributors were prominently acknowledged in the project’s dataset documentation, highlighting their significant role as key contributors to the project. This system not only motivated contributors but also served as a token of appreciation for their dedication and hard work.

In addition to the leaderboard, the Aya Discord Bot was developed to recognize contributors with a high number of points. This bot recognized the daily top 10 contributors by tagging them in a message that was posted on the Aya Discord server. It also aggregated daily total contributions per region and specified how many days were left until the data collection phase ended. As shown

[Figure 45]

Figure 18: Aya Discord Bot message aggregating daily statistics and top 10 annotators

in Figure 18, these messages provided a regular snapshot of progression that allowed annotators to see the dataset grow across all languages.

[Figure 46]

[Figure 47]

Figure 19: Aya Leaderboard. Daily and weekly leaderboards are shown on the top left and right respectively. At the bottom are language-specific and overall leader boards, where annotators are ranked based on their Aya score.

###### A.2 Accessibility of Registration Tools

The accessibility and popularity of registration tools differed from country to country and this had an impact on where the Aya UI users joined from. Figure 20 compares the percentage of registered users using Discord and Gmail to sign up in the top 10 countries. After introducing Google SSO, we observed a significant jump in the number of registered users from several new countries (Figure 21).

[Figure 48]

[Figure 49]

Figure 20: Percentage of Aya UI users that registered via Left: Discord and Right: Gmail.

[Figure 50]

- Figure 21: Percentage of the Aya UI users that registered after the introduction of Google SSO in the platform.

##### B UI Tasks

###### B.1 Reviewing Annotators

In the Aya UI, we display the original and re-annotated versions of both prompt and completion along with the name of the annotator who did the re-annotation. Reviewers are tasked with rating the re-annotated prompt and completion on a scale from 1 to 5. The ratings are defined as follows:

- (1) Much worse than the original annotation;
- (2) Worse than the original annotation;
- (3) No noticeable difference compared to the original annotation;
- (4) Better than the original annotation;
- (5) Much better than the original annotation. If any of the prompt/completion pairs receive a rating below 5 (i.e., anything lower than “Much

better than the original annotation”), the reviewer is provided with an option to modify the provided prompt and completion pair to improve its quality. An “Additional Feedback” text box is also available for reviewers to explain the reasoning behind their chosen rating.

###### B.2 Aya Score

To encourage annotators to make high-quality submissions, we designed a ranking score favoring contributions that received high peer ratings. The Aya Score for an individual annotator is:

ScoreAya = w1 × (E) + w2 × C, (1)

with weights w1, w2 defined as:

where:

w1 = max 0,(Qˆ − 3) ,w2 = T+ T

(2)

- • E represents the total number of re-annotations in which the annotator edited prompts and completions
- • C represents the total number of original annotations submitted by an annotator
- • Qˆ represents the average quality rating derived for an annotator through peer review via Annotation Feedback task
- • T+ represents the total number of original annotations made by an annotator that received positive feedback in Re-annotations task
- • T represents the total number original annotations made by an annotator that feedback received in Re-annotations task.

The rationale behind introducing the ScoreAya was to boost a competitive environment among annotators and encourage them to focus on submitting high-quality data, consequently improving the overall quality of the Aya Dataset.

###### B.3 Annotation Guidelines

The annotators were provided with the following evaluation criteria for what a good prompt and completion pair must look like. Re-Annotations were then performed if they determined that the prompts or completions needed editing.

- 1. No grammatical or spelling mistakes in both the prompt and completion text.
- 2. The prompt provides clear instruction on what the task is.

- 3. The completion answers the prompt correctly. Both the prompts and completions should be in full sentences and coherent, with reasonable length.
- 4. For original annotations, the prompts and completions should not be generated from other language models.

Re-Annotations Before editing, annotators rated the quality of existing prompt and completion pairs by choosing either the thumbs-up or thumbs-down option. If the provided prompt and completion pair were already of good quality according to the criteria above, then annotators rated them with thumbs up and moved ahead without editing. Overall, annotators were encouraged to re-annotate the completions, in particular by adding more details and context to them since many of them were often short one-word answers.

##### C Language Representation via Community

###### C.1 Division by Regions

We chose to divide languages into four primary regions: Africa, Asia, Europe, and Latin America. These four regions were established in order to facilitate the administration of user contributions and were not intended to prescribe boundaries within which certain languages are exclusively spoken.

The language statistics by region are as follows: Africa (14 languages), Asia (41 languages), Europe (42 languages), and Latin America (4 languages). Almost all the languages were assigned to a region but there are some exceptions, Maori and Samoan were unassigned to any specific region as they didn’t align with the predefined regions. English was left unassigned, serving as a common language across all regions. Additionally, contributions in Spanish and Portuguese were distributed between Europe and Latin America based on contributors’ countries. Similarly, Arabic contributions were shared between Africa and Asia depending on the contributors’ country of origin. Additional dialects of Arabic were included in regions separate from that of their parent language because we had a significant number of speakers from these regions eager to contribute to their respective dialects. Each region had at least one “Regional Lead” responsible for coordinating “Language Ambassadors,” and for recruiting fluent speakers for the languages within their area.

###### C.2 Language Ambassadors

The Language Ambassador’s role was pivotal in bridging the gap between the data collected in a language and its speakers. An essential criterion for selection was native fluency in the specific language. The Language Ambassador’s expertise in specific languages and familiarity with the cultures of the language speakers was invaluable. They assisted not only in spreading awareness among participants but also in identifying and addressing potential data issues specific to each language, such as languages incorrectly assigned to their region. Their cultural and linguistic insights enabled them to make informed decisions, like choosing suitable data sources for collection in their respective languages. Not every language had a designated Language Ambassador, and some had more than one. In total, we had 84 Language Ambassadors over the course of the initiative. Their combined efforts played a vital role in broadening the contributor base for each language. Support for the Language Ambassadors’ progress and trouble-shooting challenges they faced was coordinated asynchronously and through weekly online meetups, discussed in Sec. D.1 and Sec. D.2.

###### C.3 Regional Leads

There were a total of six Regional Leads: two for Latin America, one for Africa, one for Asia, and two for Europe. The selection for Regional Lead roles was on a voluntary basis, with the only requirement being that they must originate from the regions they intended to lead. The invitation for this role was specifically extended to individuals who were already actively participating in Cohere For AI projects or engaged in other open science projects. Regional Leads had several key roles throughout the project, such as selecting Language Ambassadors and aiding their efforts in attracting more annotators and maintaining their engagement.

##### D Communication

###### D.1 Platforms

We established a Discord server for coordination between Regional Leads, Language Ambassadors and annotators. The server provided basic channels for internal communications: introductions, inquiries, and announcements, as well as specific channels for Language Ambassadors, for each region, and for each language, along with any other channels that proved useful for the particular region. For external communications, we used social media platforms (e.g., X, LinkedIn, WhatsApp, Facebook), recognizing that the choice of communication platform varied based on cultural and regional preferences. Using multiple platforms not only facilitated internal organization but also broadened our project’s outreach by providing flexible and inclusive means of outreach to diverse communities and audiences.

###### D.2 Meetings

In addition to asynchronous communication through Discord, we conducted meetings to maintain team collaboration and cohesion:

- 1. Regional Leads and Language Ambassadors Meeting: A weekly meeting in which Regional Leads and Language Ambassadors shared project updates, exchanged ideas, and addressed questions from Language Ambassadors. It served as an excellent platform for gathering ideas from Language Ambassadors and brainstorming new strategies to engage annotators effectively.
- 2. New Contributor Introduction Meeting: Held weekly, this meeting aimed to introduce new contributors to the project’s specifics. It included explanations about the motivations behind the project, a walk-through of the Aya UI, and a sharing of regional statistics. Additionally, this meeting provided examples of both good and bad annotations and edits to guide new annotators in their work. It concluded with a synchronous challenge for the annotators to submit a few initial annotations in real time to familiarize them with the process and allow them to ask questions if they got stuck.
- 3. Regional Leads Meeting: Held bi-weekly, this meeting brought together Regional Leads to assess progress, discuss upcoming steps, and provide advice on how to engage and sustain contributions for their respective regions. Furthermore, this meeting facilitated collaborative troubleshooting efforts and helped make important decisions for the following week.

- 4. Technical Update: This meeting was dedicated to sharing technical updates, with a focus on recent UI progress, data, and benchmarking. The purpose of this monthly update was to ensure all team members and annotators were well-informed about the project’s current status and upcoming priorities. It was a place for open discussion to hear feedback from everyone interested in the project.
- 5. Language Specific Meeting: Held weekly or biweekly, these meetings were co-working sessions or datathons led by the language ambassadors with their respective annotators to submit annotations synchronously. It also acted as an onboarding session to welcome new contributors from regions that could not join the New Contributor Introduction Meeting due to conflicting time zones. Demonstrations on using the UI, as well as brainstorming sessions, were conducted to improve the representation of specific languages in the project.

##### E Language Groupings

In this work we will refer to groups of languages to be “lower-”, “mid-” or “higher”-resourced according to their recorded, written, and catalogued NLP resources [Joshi et al., 2020]. Joshi et al. [2020] group languages into 5 distinct clusters based on the amount of data from a combined range of source (LDC catalog16, ELRA Map17, Wikipedia 18), which we interpret as a proxy for data availability for pretraining and IFT training of LLMs. We group these 5 distinct clusters into a rough taxonomy of lower-resourced (LR), mid-resourced (MR) and higher-resourced (HR) (See Table 6). See Table 5 for full mapping of languages to categories. We note that this grouping is inevitably imperfect; languages and their varieties cannot absolutely nor universally be classified based on this single dimension [Hämäläinen, 2021; Lignos et al., 2022; Bird, 2022]. The categorization in our case serves the purpose of aggregation in our analysis of the data distribution.

Group Category Languages Examples Higher-Resourced

5 7 Arabic, Chinese, English, French, Spanish 4 18 Hindi, Italian, Portuguese, Russian, Turkish

Mid-Resourced 3 25 Afrikaans, Indonesian, Kazakh, Malay, Latvian

2 13 Hausa, Icelandic, Irish, Lao, Maltese 1 39 Albanian, Gujarati, Igbo, Luxembourgish 0 12∗ Kurdish, Kyrgyz, Sinhala, Yiddish

Lower-Resourced

- Table 6: Language grouping for the Aya Collection. We assign categories to languages based on [Joshi et al., 2020]. (*) We assign label 0 to two languages not found in Joshi et al. [2020]’s taxonomy (manipuri and ngaju).

- 16https://catalog.ldc.upenn.edu/
- 17https://catalog.elra.info/en-us/
- 18https://wikipedia.org/

##### F Post-Editing the dolly-machine-translated Test Set

###### F.1 Annotators

Annotator Selection The primary demographic make-up of the participants in the evaluations was recruited based on their proficiency in the language groups. The proficiency was self-reported, and our requirements were natively proficient or professionally proficient in the specific languages needed for the project. Outside of this, the participants come from diverse social backgrounds comprised of students and individuals with full-time or part-time jobs that do annotation as a “side gig”.

Socio-Demographics The annotator pool is comprised of people from diverse backgrounds, and this spans across socioeconomic backgrounds, careers, levels of education, and self-reported gender and sexual identities. We do not ask any annotators to share or report any of these statistical pieces of information in a formal way; any insights into this are gathered organically and through self-reporting by the annotators.

Quality Considerations We do not believe that any socio-demographic characteristics have led to any impact on the data that has been annotated. Through every part of the project we have reiterated the importance of this work and the fact that this is helping to support a global-scale research project. We are confident in the trust we have built with the annotators in this project, and they care greatly about the overall outcome and therefore have been diligent in completing the task with a high degree of accuracy. Where possible, we have done our best to have annotators work on this project and be representatives of the communities that the project aims to support.

Compensation The annotators were paid 30 CAD per hour. No special consideration was made to the hourly rate as that is the standard rate offered to Cohere’s annotators who work on highly complex tasks.

###### F.2 Annotation Process

Communication Annotators were briefed by one of the authors in a virtual introduction session, and were able to ask questions and raise issues throughout the annotation task in a Slack channel. They were also encouraged to share frequent error patterns, artifacts, or hard decisions that they encountered throughout the task with the authors and other annotators.

Schedule There was no fixed time schedule for the annotations and annotators contributed a varying amount of hours and ratings, depending on their availabilities and speed. Each translation was post-edited by one annotator, and there were 3–4 annotators involved in each task. After postedits were completed, a second annotator (not the original post-editor) assessed the post-edit for quality and proposed new final edits if necessary.

Interface Post-edits were collected on Google Sheets with an interface built in Google Apps Script.

###### F.3 Instructions

The instructions given to professional annotators for the dolly-machine-translated test set post-edits were the following: “As an annotator, you have the task to improve the quality of the prompts for our multilingual model! The prompts are originally machine-translated from English, and sometimes the translation introduces errors in the prompts that make them hard to follow.

We need your help to identify these cases, and to edit these translations so that they...

- 1. Convey the same instruction/task/request as the English original — not more and not less.
- 2. Are grammatically correct.
- 3. Are free from phrases too literally translated from English (we call this “Translationese”).

This is how: For each pair of English prompt and translated prompt shown, decide whether the prompt is okay as it is (according to the above criteria), or needs an edit.

- • If it needs an edit, edit the prompt until the quality is satisfactory (in the field “Edited Prompt”). Try to keep your edits minimal. Then confirm that the edited prompt fulfills the above three criteria.
- • If it’s okay as is, just proceed (without editing the “Edited Prompt” field) to confirm that it fulfills the above three criteria.”

Annotations were done through an interface built on top of Google Sheets. One annotator edited each prompt, and another verified the edit, if necessary had a discussion and edited the original edit. Three to four editors collaborated on each language.

###### F.4 Post-Editing Effort

For each prompt, we measure the post-editing effort with Human-targeted Translation Error Rate (HTER) [Specia & Farzindar, 2010], an edit-distance metric that compares the original machine translation with the post-edited version in terms of edit operations on units of words. This also gives us an estimate of how severe the errors in the original translations were, and how critically the posteditors assessed the original translations. Analogously, we estimate with a Human-targeted Character F-Score (HChrF) score how much the original translation overlaps with the final post-edited translation. This metric is based on the ChrF score [Popović, 2015] and operates on character-level matches. Computations of HTER and HChrF are based on the sacrebleu implementation [Post,

- 2018].

- Table 7 reports these statistics for the six languages of the dolly-machine-translated test set. We find that editors edited at least 41% of prompts in all languages, a surprisingly high number. This indicates that translation errors in the dolly-machine-translated test set are quite common.

Language % of Prompts Edited HTER HChrF

Arabic 41.0% 10.78 92.74 French 84.5% 5.56 96.81 Hindi 60.0% 6.16 95.00 Russian 86.5% 37.43 75.92 Serbian 72.5% 9.06 92.79 Spanish 75.5% 9.13 93.25

Table 7: Post-editing effort measured by the overall percentage of edited dolly test prompts, HTER (Human-targeted Translation Error Rate: the higher, the more effort), and HChrF (HumanTargeted Character F-Score: the lower, the more effort).

For Russian, the post-editing effort was overall largest, with an average of 37.43 HTER, which means that 37.43% of words in the final post-edit had to be edited from the original. This stands in contrast with the post-edits for French, where a similar ratio of original prompts was edited (84.5% compared to 86.5% for Russian), but to a much lesser extent (5.56 HTER).

##### G Translation using NLLB

Additional Generation details One caveat with using NLLB is that since the model was trained on single sentence pairs, the translations tend to cut off abruptly when full paragraphs are translated. To get around this, we sentence-tokenize the paragraphs using the sentence-splitter Python package (similar to [NLLB-Team et al., 2022]) and concatenate them post-translation. To avoid known translation introduced artefacts, We also filter any samples which have <unk> tokens introduced by the NLLB tokenizer or model.

###### G.1 Translation Quality of NLLB

- Figure 22 illustrates NLLB translation quality as measured by ChrF++ on the FLORES benchmark for the languages of interest for Aya, grouped by their resourcefulness according to [Joshi et al.,

- 2020]. The scores were extracted from https://github.com/facebookresearch/fairseq/blob/n llb/README.md for the dense 3.3B model.

60

ChrF++

40

20

0

tel_Telu

hau_Latn

nno_Latn

nob_Latn

epo_Latn

taq_Latn

hat_Latn

kan_Knda

ace_Latn

sna_Latn

lao_Laoo

sun_Latn

xho_Latn

nso_Latn

ibo_Latn

sot_Latn

bjn_Latn

gla_Latn

gle_Latn

knc_Latn

pbt_Arab

plt_Latn

ace_Arab

azb_Arab

azj_Latn

snd_Arab

zul_Latn

jav_Latn

als_Latn

swh_Latn

yor_Latn

som_Latn

smo_Latn

bjn_Arab

ltz_Latn

min_Latn

knc_Arab

kas_Arab

amh_Ethi

ckb_Arab

npi_Deva

ydd_Hebr

sin_Sinh

mlt_Latn

cym_Latn

isl_Latn

hye_Armn

kmr_Latn

guj_Gujr

mri_Latn

mar_Deva

khm_Khmr

tgk_Cyrl

khk_Cyrl

mkd_Cyrl

mya_Mymr

kir_Cyrl

###### (a) Low-resource Languages: [Joshi et al., 2020] classes 0, 1, 2

60

ChrF++

40

20

0

tam_Taml

ben_Beng

dan_Latn

uzn_Latn

ceb_Latn

est_Latn

glg_Latn

ind_Latn

ron_Latn

afr_Latn

tha_Thai

heb_Hebr

mni_Beng

kat_Geor

urd_Arab

lvs_Latn

slv_Latn

slk_Latn

lit_Latn

zsm_Latn

ell_Grek

bel_Cyrl

bul_Cyrl

kaz_Cyrl

ukr_Cyrl

mal_Mlym

###### (b) Mid-resource Languages: [Joshi et al., 2020] class 3

60

ChrF++

40

20

0

hun_Latn

deu_Latn

eus_Latn

spa_Latn

aeb_Arab

pol_Latn

nld_Latn

cat_Latn

por_Latn

jpn_Jpan

fin_Latn

ces_Latn

ita_Latn

tur_Latn

fra_Latn

zho_Hant

yue_Hant

acq_Arab

apc_Arab

pes_Arab

vie_Latn

swe_Latn

zho_Hans

ajp_Arab

arb_Arab

hin_Deva

kor_Hang

ary_Arab

acm_Arab

arz_Arab

ars_Arab

rus_Cyrl

srp_Cyrl

(c) High-resource Languages: [Joshi et al., 2020] classes 4, 5

- Figure 22: NLLB Translation Quality: ChrF++ scores on FLORES for translations from English into the Aya target languages that are covered by NLLB, grouped by their resourcefulness according to [Joshi et al., 2020].

##### H Additional Figures

1.0

0.8

Avg.ApprovalRatio

Translated XP3 Templated Aya Dataset

| |
|---|

| |
|---|

0.6

| |
|---|

0.4

0.2

0.0

X-CSQA-instNusaX-senti-instxp3_GEM_BiSECTTurku-paraphrase-instxp3_paws-xMintaka-instxp3_xquad_xquadAfriSenti-instXlel_wd-instFlan-lambada(T)Joke-explaination-inst (T)Wiki-split-inst(T)Xlel_wd-inst(T)PAWS-Wiki(T)WIKIQA(T)Mintaka-inst(T)Flan-unified-QA (T)AdversarialQA(T)Flan-CoT-submix (T)Wiki-split-instMasakhaNEWS-instNQ-Open(T)Flan-GEM-wiki-lingua (T)SODA-instDolly-v2(T)xp3_GEM_xlsumHotpotQA(T)AfriQA-instMLQA-en(T)CNN-Daily-Mail (T)xp3_khalidalt_tydiqa-goldporiginal_annotationsxp3_khalidalt_tydiqa-primaryTelugu-news-articlesTelugu-food-recipesTelugu-jokesIndicXParaphrase-instTelugu-poemsTeluguRiddles

###### Figure 23: Average Approval Ratio per dataset, constrained to datasets receiving at least 20 votes.

500

AveragePromptLength

400

300

200

100

0

Initial Length

Edited Length

| |
|---|

HR MR LR

(a) Average prompt length

500

AverageCompletionLength

400

300

200

100

0

Initial Length

Edited Length

| |
|---|

HR MR LR

(b) Average completion length

###### Figure 24: Average prompt and completion length of instances in the Aya Dataset before and after re-annotation across different language categories.

2000

Language Category

HR MR LR

| |
|---|

1750

| |
|---|

1500

AverageCompletionLength

1250

1000

750

500

250

0

kir

lit

fil

mar

mal

mlg

ell

wol

yor

som

mya

srp

msa

ukr

rus

kor

kur

swa

swe

vie

sin

zul

jav

sqi

tam

fra

tur

fin

tel

ita

urd

por

ara

amh

guj

gle

hin

ind

nld

ibo

pol

jpn

nya

pus

snd

pes

sna

spa

xho

kan

eus

nso

ceb

sun

zho

hat

tha

pan

ben

eng

deu

hun

nep

dan

hau

###### (a) Average completion length for every language in the Aya Dataset.

Language Category

1200

HR MR LR

| |
|---|

| |
|---|

| |
|---|

1000

AveragePromptLength

800

600

400

200

0

kir

lit

fil

mar

mal

mlg

ell

wol

som

ukr

rus

srp

mya

msa

kor

yor

kur

zul

vie

jav

sqi

sin

swe

swa

fra

tam

tur

tel

ita

fin

urd

ara

por

amh

ind

nld

pol

ibo

hin

jpn

guj

gle

snd

nso

xho

spa

sna

pes

kan

sun

pus

eus

ceb

nya

zho

hat

tha

pan

dan

ben

hun

hau

deu

eng

nep

(b) Average prompt length for every language in the Aya Dataset

###### Figure 25: Average prompt and completion length for every language in the Aya Dataset

Language Category

HR MR LR

2000

| |
|---|

AverageLength(Prompt+Completion)

1750

1500

1250

1000

750

500

250

0

kir

lit

fil

mar

mal

mlg

ell

wol

yor

som

mya

srp

ukr

rus

msa

kor

kur

swe

swa

vie

zul

jav

sin

sqi

tam

fra

tur

tel

fin

ita

urd

por

ara

amh

ind

guj

nld

gle

hin

pol

ibo

jpn

snd

nya

pus

nso

pes

sna

xho

spa

kan

eus

sun

ceb

zho

hat

tha

pan

ben

eng

dan

deu

hun

hau

nep

Figure 26: Average prompt and completion length across different languages in Aya Dataset.

##### I Additional Tables

Dataset #Langs Template lang Dataset lang L¯prompt L¯compl. License Task AfriQA-inst [Ogundepo et al., 2023]

bem, fon, hau, ibo, kin, swh, twi, wol, yor, zul, eng, fra

Question Answering

12

46 15 CC BY 4.0

AfriSenti-inst [Muhammad et al., 2023]

amh, arq, hau, ibo, kin, ary, por, swh, twi

Sentiment Analysis

9

168 44 CC BY 4.0

Amharic QA [Abedissa et al., 2023]

MIT license

Question Answering

1 amh amh 1114 33

News-summaryinstruct [TahmidH, 2023]

1 ben ben 174 67 CC0 1.0 Summarization

Arpa-instruct [syntaxshill, 2023]

1 hye hye 165 118 Artistic-2.0 Paraphrasing

Telugu-food-recipes [SuryaKrishna02,

- 2023a]

1 tel tel 70 870 Apache 2.0 Generation

Telugu-jokes [SuryaKrishna02,

- 2023b]

1 tel tel 80 276 Apache 2.0 Generation

Telugu-news-articles [SuryaKrishna02,

- 2023c]

1 tel tel 448 426 Apache 2.0 Generation

Telugu-poems [SuryaKrishna02, 2023e]

1 tel tel 357 198 Apache 2.0 Generation

FarsTail-Instruct [Amirkhani et al., 2023; hghader1, 2023]

Natural Language Inference

1 pes pes 224 112 Apache 2.0

Hindi-articlesummarization [ganeshjcs, 2023a]

CC BY-SA 4.0

1 hin hin 3813 175

Summarization

Hindi-articlegeneration [ganeshjcs, 2023b]

IMDB-Dutch-instruct [Maas et al., 2011; jjzha, 2023]

IndicSentiment-inst [Doddapaneni et al., 2023; el2e10, 2023a]

IndicXParaphrase-inst [Doddapaneni et al., 2023; el2e10, 2023b; SuryaKrishna02,

- 2023d]

Indo-stories-instruct [Iftitahu, 2023a;b;c]

Joke-explaination-inst [theblackcat102, 2023]

1 hin hin 102 3683

CC BY-SA 4.0

Generation

Sentiment Analysis

1 nld nld 1470 31 Apache 2.0

ben, guj, hni, kan, mal, mar, pan, tam, tel, urd, eng

11 eng

174 141 MIT Translation

ben, guj, hin, mar, pan, mal, tel

ben, guj, hin, mar, pan, mal, tel

Paraphrase Identification

7

132 93 MIT

3 ind, sun, jav ind, sun, jav 345 322 CC BY 4.0 Translation

- 1 eng 118 548 MIT Generation

Lijnews-instruct [ConseggioLigure, 2023a;b]

- 2 ita, lij it, lij 893 898 CC BY 4.0 Translation

LLM-JapaneseVanilla-inst [Suzuki et al., 2023;

- Tellarin.ai, 2023a]

1 jpn jpn 60 97

CC BY-SA 4.0

Question Answering

MasakhaNEWS-inst [Adelani et al., 2023]

16

amh, eng, fra, hau, ibo, lin, cgg, orm, pcm, run, sna, som, swh, tir, xho, yor

1483 1459 AFL-3.0

Text Classification

Mintaka-inst [Sen et al., 2022]

9 eng

arb, deu, spa, fra, jpn, por, hin, ita, eng

102 49 CC BY 4.0

Question Answering

NTX-LLM-inst [Chen et al., 2023c;

- Tellarin.ai, 2023b],

NusaX-senti-inst [Winata et al., 2023b]

Persian-instruct-pn [Farahani et al., 2021; Shafagh, 2023a;b]

Scirepeval-biomimicryinst [Singh et al., 2022]

Seed-instruct-lij [Maillard et al., 2023; ConseggioLigure, 2023c;d]

SODA-inst [Kim et al.,

- 2022]

TamilStories [AI Tamil Nadu, 2023a]

TeluguRiddles [desik98, 2023]

Thai-USEmbassyprompt [PyThaiNLP, 2023d;e]

Thai-POS-inst [PyThaiNLP, 2023c]

arb, zho, nld, eng, fra, deu, hin, ita, jpn, kor, por, spa, tur

arb, zho, nld, eng, fra, deu, hin, ita, jpn, kor, por, spa, tur

CC BY-SA 4.0

Information Extraction

917 493

13

ace, ban, bjn, bug, eng, ind, jav, mad, min, nij, sun, bbc

Sentiment Analysis

12

219 22 Apache 2.0

- 1 pes pes 1713 128 MIT Summarization

SCB-MT-2020-prompt [Lowphansirikul et al.,

- 2022; PyThaiNLP,
- 2023a;b]

- 2 tha, eng tha, eng 181 127

CC BY-SA 4.0

Translation

Scientific Document Representation

1 eng 996 523 ODC-BY

CC BY-SA 4.0

2 lij, eng lij, eng 184 186

Translation

1 eng 412 328 CC BY 4.0 Dialogue

1 tam tam 2266 2172 Apache 2.0 Generation

Question Answering

1 tel tel 74 44 Apache 2.0

2 tha, eng tha, eng 2131 2077 CC0 1.0 Translation

CC BY-SA 3.0

1 tha tha 72 36

Generation

Thai-Wiktionary-inst [PyThaiNLP, 2023f]

CC BY-SA 3.0

1 tha tha 35 147

Generation

Thirukkural-instruct [AI Tamil Nadu,

1 tam tam 133 542 Apache 2.0 Generation

- 2023b]

Turku-paraphrase-inst [Kanerva et al., 2021; TurkuNLP, 2023]

CC BY-SA 4.0

Paraphrase Identification

1 fin fin 108 59

UA-Gec-inst [Syvokon et al., 2023; osyvokon, 2023]

1 ukr ukr 192 148 CC BY 4.0 Generation

zho, hrv, dan, eng, deu, por, rus, srp, slk, swe, tgl

zho, hrv, dan, eng, deu, por, rus, srp, slk, swe, tgl

UNER-LLM-inst [Mayhew et al., 2023; Universal NER, 2023]

CC BY-SA 4.0

Named Entity Recognition

11

768 109

Urdu-News-GenArticle [Hussain et al., 2021; AhmadMustafa,

- 2023a]

1 urd urd 109 1313 CC BY 4.0 Generation

Urdu-News-CategoryClass [Hussain et al., 2021; AhmadMustafa,

- 2023b]

1 urd urd 1407 43 CC BY 4.0

Text Classification

Urdu-News-GenHeadline [Hussain et al., 2021; AhmadMustafa,

- 2023c]

1 urd urd 1314 94 CC BY 4.0 Generation

Wiki-split-inst [Botha

- et al., 2018]

1 eng 200 166 CC BY 4.0

Text Simplification

X-CSQA-inst [Lin

- et al., 2021]

16

eng, zho, deu, spa, fra, ita, jpn, nld, pol, por, rus, arb, vie, hin, swa, urd

197 21 MIT

Question Answering

Xlel_wd-inst [Pratapa

- et al., 2022]

44 379 190 CC BY 4.0 Event Linking

XWikis-inst [Perez-Beltrachini & Lapata, 2021]

4

ces, fra, eng, deu

5662 346 MIT Summarization

Table 8: List of datasets in Aya Collection (templated datasets).

Dataset #Langs L¯prompt L¯compl. License Task Adversarial QA (T) [Bartolo et al., 2020] 101 159 721 CC BY-SA 3.0

Question Answering

CNN-Daily-Mail (T) [See et al., 2017] [Hermann et al., 2015]

101 1980 305 Apache 2.0 Summarization Flan-Coqa (T) [Wei et al., 2022a; Reddy

- et al., 2019]

Question Answering

101 2143 364 Multiple*

Flan-CoT-submix (T) [Wei et al., 2022a] 101 239 160 Unknown Generation Flan-GEM-wiki-lingua (T) [Wei et al., 2022a; Ladhak et al., 2020]

CC BY-NC-SA 3.0

101 1732 572

Summarization

Flan-lambada (T) [Wei et al., 2022a; Paperno et al., 2016]

101 232 7 CC BY 4.0 Generation

Flan-unified-QA (T) [Wei et al., 2022a; Khashabi et al., 2020]

Question Answering

101 281 13 Apache 2.0

Question Answering

HotpotQA (T) [Yang et al., 2018] 101 129 15 CC BY-SA 4.0

Joke-explaination-inst (T) [theblackcat102,

101 111 545 MIT Generation Mintaka-inst (T) [Sen et al., 2022] 101 54 67 CC BY 4.0

- 2023]

Question Answering

Question Answering

MLQA-en (T) [Lewis et al., 2020] 101 819 20 CC BY-SA 3.0

Question Answering

NQ-Open (T) [Kwiatkowski et al., 2019] 101 68 14 CC BY-SA 3.0

Custom license, attribution

Paraphrase Identification

PAWS-Wiki (T) [Zhang et al., 2019] 101 308 6

Question Answering

PIQA (T) [Bisk et al., 2020] 101 304 100 Unknown

SODA-inst (T) [Kim et al., 2022] 101 86 208 CC BY 4.0 Dialogue WIKI QA (T) [Yang et al., 2015] 101 205 36 MSR DLA*

Question Answering

Text Simplification

Wiki-split-inst (T) [Botha et al., 2018] 101 126 220 CC BY-SA 4.0

Xlel_wd-inst (T) [Pratapa et al., 2022] 101 300 274 CC BY 4.0 Event Linking Dolly-v2 (T) [Conover et al., 2023] 101 427 357 CC BY-SA 3.0 Generation

###### Table 9: List of datasets in Aya Collection (translated datasets).

Main Task Type Fine-grained Task Type Dataset Question Answering AfriQA-inst [Ogundepo et al., 2023]

Amharic QA [Abedissa et al., 2023] LLM-Japanese-Vanilla-inst [Tellarin.ai, 2023a] Mintaka-inst [Sen et al., 2022] X-CSQA-inst [Lin et al., 2021] TeluguRiddles [desik98, 2023]

Natural Language Summarization News-summary-instruct [TahmidH, 2023] Generation Persian-instruct-pn [Shafagh, 2023a]

Hindi-article-summarization [ganeshjcs, 2023a] XWikis-inst [Perez-Beltrachini & Lapata, 2021]

Translation IndicSentiment-inst [el2e10, 2023a] Indo-stories-instruct [Iftitahu, 2023a;b;c] Lijnews-instruct [ConseggioLigure, 2023a;b] SCB-MT-2020-prompt [PyThaiNLP, 2023a;b] Thai-USEmbassy-prompt [PyThaiNLP, 2023d;e] SEED-instruct-lij [ConseggioLigure, 2023c;d]

Paraphrasing Arpa-instruct [syntaxshill, 2023] IndicXParaphrase-inst [el2e10, 2023b; SuryaKrishna02, 2023d] Turku-paraphrase-inst [TurkuNLP, 2023]

Text Simplification Wiki-split-inst [Botha et al., 2018] Dialogue SODA-inst [Kim et al., 2022] NL Generation Telugu-food-recipes [SuryaKrishna02, 2023a]

Telugu-jokes [SuryaKrishna02, 2023b] Telugu-news-articles [SuryaKrishna02, 2023c] Telugu-poems [SuryaKrishna02, 2023e] TamilStories [AI Tamil Nadu, 2023a] Joke-explaination-inst [theblackcat102, 2023] Thirukkural-instruct [AI Tamil Nadu, 2023b] Hindi-article-generation [ganeshjcs, 2023b] Thai-Wiktionary-inst [PyThaiNLP, 2023f] UA-Gec-inst [osyvokon, 2023] Urdu-News-Gen-Article [AhmadMustafa, 2023a] Urdu-News-Gen-Headline [AhmadMustafa, 2023c] Thai-POS-inst [PyThaiNLP, 2023c]

Text Classification Sentiment Analysis AfriSenti-inst [Muhammad et al., 2023] IMDB-Dutch-instruct [jjzha, 2023] NusaX-senti-inst [Winata et al., 2023b]

Information Extraction NTX-LLM-inst [Tellarin.ai, 2023b] Named Entity Recognition UNER-LLM-inst [Universal NER, 2023] Natural Language Inference FarsTail-Instruct [hghader1, 2023] Event Linking Xlel_wd-inst [Pratapa et al., 2022] Sci. Doc. Representation Scirepeval-biomimicry-inst [Singh et al., 2022] Text Classification Urdu-News-Category-Class [AhmadMustafa, 2023b]

MasakhaNEWS-inst [Adelani et al., 2023]

- Table 10: Task Taxonomy of Templated Datasets (Aya Collection). We classify the templated datasets with a standard task taxonomy of three main tasks: Question Answering, Natural Language Generation, and Text Classification (Table 4). We then have a fine-grained task taxonomy within each task, such as Summarization, Translation, Paraphrasing, Sentiment Analysis, Information Extraction, and Named Entity Recognition. If there is not a recognized fine-grained task taxonomy for a specific dataset, we put it in the main task type category.

Main Task Type Fine-grained Task Type Dataset Question Answering Adversarial QA (T) [Bartolo et al., 2020]

Flan-Coqa (T) [Wei et al., 2022a; Reddy et al., 2019] Flan-unified-QA (T) [Wei et al., 2022a; Khashabi et al., 2020] HotpotQA (T) [Yang et al., 2018] Mintaka-inst (T) [Sen et al., 2022] MLQA-en (T) [Lewis et al., 2020] NQ-Open (T) [Kwiatkowski et al., 2019] PIQA (T) [Bisk et al., 2020] WIKI QA (T) [Yang et al., 2015]

Natural Language Summarization

CNN-Daily-Mail (T) [See et al., 2017] [Hermann et al., 2015]

Generation

Flan-GEM-wiki-lingua (T)[Wei et al., 2022a; Ladhak et al., 2020]

Text Simplification Wiki-split-inst (T) [Botha et al., 2018] Dialogue SODA-inst (T) [Kim et al., 2022] NL Generation Joke-explaination-inst (T) [theblackcat102, 2023]

Flan-CoT-submix (T)[Wei et al., 2022a] Flan-lambada (T) [Wei et al., 2022a; Paperno et al., 2016] Dolly-v2 (T) [Conover et al., 2023]

Text Classification Event Linking Xlel_wd-inst (T) [Pratapa et al., 2022] Paraphrase Identification PAWS-Wiki (T) [Zhang et al., 2019]

- Table 11: Task Taxonomy of Translated Datasets (Aya Collection). We classify the translated datasets similar to templated datasets (Table 10). If there is not a recognized fine-grained task taxonomy for a specific dataset, we put it in the main task type category.

Dataset Language

L¯prompt L¯compl. License Task adversarial_qa dbert [Bartolo et al.,

Dataset #Langs

- 2020; maxbartolo, 2023a]

1 eng 655 263

CC BY-SA 3.0

Question Answering

adversarial_qa dbidaf [Bartolo et al.,

- 2020; maxbartolo, 2023b]

CC BY-SA 4.0

Question Answering

1 eng 669 256

adversarial_qa droberta [Bartolo et al., 2020; maxbartolo, 2023c]

CC BY-SA 4.0

Question Answering

1 eng 742 243

ag_news [Gulli, 2005; jxmorris12 et al., 2023]

BSD-3Clause

Text Classification

1 eng 292 40

ai2_arc ARC-Challenge [Clark et al., 2018]

Question Answering

1 eng 351 33 GPL-3

Question Answering

ai2_arc ARC-Easy [Clark et al., 2018] 1 eng 307 26 GPL-3

BSD-3Clause

Sentiment Analysis

amazon_polarity [Zhang et al., 2015] 1 eng 454 83

Sentiment Analysis

app_reviews [Grano et al., 2017] 1 eng 159 28 Unknown

Question Answering

clue c3 [Xu et al., 2020] 1 zho 338 7 Apache 2.0

CC BY-SA 4.0

Question Answering

clue cmrc2018 [Cui et al., 2019] 1 zho 426 178

Question Answering

clue csl [Li et al., 2022] 1 zho 315 64 Apache 2.0

CC BY-SA 3.0

Question Answering

clue drcd [Shao et al., 2019] 1 zho 436 128

Question Answering

clue tnews [Xu et al., 2020] 1 zho 235 7 Apache 2.0

cnn_dailymail_3.0.0 [Nallapati et al.,

1 eng 1699 646 Unknown Summarization common_gen [Lin et al., 2020] 1 eng 96 49 MIT Generation cos_e_v1.11 [Rajani et al., 2019] 1 eng 208 19

- 2016]

BSD-3Clause

Generation cosmos_qa [Huang et al., 2019] 1 eng 547 51 Unknown

Question Answering

Topic Classification

dbpedia_14 [Lehmann et al., 2014] 1 eng 378 64 Apache 2.0

Question Answering

dream [Gu et al., 2022] 1 eng 511 152 Apache 2.0

Question Answering

duorc ParaphraseRC [Saha et al., 2018]

1 eng 1438 663 MIT

Question Answering

duorc SelfRC [Saha et al., 2018] 1 eng 1411 645 MIT

Text Simplification

GEM/BiSECT [Kim et al., 2021] 3 eng, spa, fra 346 251 Unknown

CC BY-NC-SA 4.0

GEM/xlsum [Hasan et al., 2021] 2 eng, ben 1156 636

Summarization

gigaword [Rush et al., 2015; Graff et al., 2003]

1 eng 181 80 Unknown Summarization

glue mrpc [Warstadt et al., 2018; Wang et al., 2018; Dolan & Brockett, 2005]

Text Classification

1 eng 270 38 MIT

glue qqp [Warstadt et al., 2018; Wang et al., 2018; Iyer et al., 2012]

Text Classification

1 eng 199 4 Unknown

Sentiment Analysis

imdb [Maas et al., 2011] 1 eng 1089 106 Unknown

arb, ben, eng, ind, swh, tel

Question Answering

tydiqa-goldp [Clark et al., 2020] 6

526 115 Apache 2.0

arb, ben, eng, ind, swa, tel

Question Answering

tydiqa-primary [Clark et al., 2020] 6

1110 332 Apache 2.0

kilt_tasks hotpotqa [Petroni et al., 2021]

Question Answering

1 eng 137 15 MIT

Custom license

multi_news [Fabbri et al., 2019] 1 eng 3466 1442

Summarization

openbookqa main [Mihaylov et al.,

- 2018]

1 eng 163 16 Apache 2.0

Question Answering

xlwic xlwic [Raganato et al., 2020] 1 eng 225 3

CC BY-NC 4.0

Text Classification

paws labeled_final [Zhang et al.,

- 2019]

Custom license

Paraphrase Identification

1 eng 285 12

eng, spa, fra, zho

Custom license

Paraphrase Identification

paws-x [Yang et al., 2019] 4

255 11

Question Answering

piqa [Bisk et al., 2020] 1 eng 256 72 AFL 3.0

Question Answering

qasc [Khot et al., 2020] 1 eng 314 38 Apache 2.0

CC BY-NC-SA 4.0

Question Answering

quail [Rogers et al., 2020] 1 eng 1752 18

Question Answering

quarel [Tafjord et al., 2019a] 1 eng 289 10 CC BY 4.0

Question Answering

quartz [Tafjord et al., 2019b] 1 eng 307 9 CC BY 4.0

Question Answering

quoref [Dasigi et al., 2019] 1 eng 1556 388 CC BY 4.0

Custom license

Question Answering

race high [Lai et al., 2017] 1 eng 1723 229

Custom license

Question Answering

race middle [Lai et al., 2017] 1 eng 1141 144

Question Answering

ropes [Lin et al., 2019] 1 eng 886 97 CC BY 4.0

Sentiment Analysis

rotten_tomatoes [Pang & Lee, 2005] 1 eng 152 18 Unknown

CC BYNC-ND 4.0

Summarization sciq [Welbl et al., 2017] 1 eng 346 139

samsum [Gliwa et al., 2019] 1 eng 473 170

CC BY-NC 3.0

Question Answering

Question Answering

social_i_qa [Sap et al., 2019] 1 eng 182 15 CC BY 4.0

CC BY-SA 4.0

Question Answering

squad_v2 [Rajpurkar et al., 2016] 1 eng 689 82

CC BY-SA 3.0

Question Answering

super_glue boolq [Clark et al., 2019; Wang et al., 2019]

1 eng 653 76

super_glue multirc [Khashabi et al., 2018]

Custom license

Question Answering

1 eng 1509 120

Question Answering

super_glue record [Zhang et al., 2018a]

1 eng 1175 70 Apache 2.0

super_glue wic [Pilehvar & Camacho-Collados, 2019]

CC BY-NC 4.0

Text Classification

1 eng 170 3

Text Classification

trec [Li & Roth, 2002; Hovy et al., 2001]

1 eng 144 9 Unknown

trivia_qa unfiltered [Joshi et al.,

Question Answering

1 eng 148 92 Unknown

- 2017]

Question Answering

web_questions [Berant et al., 2013] 1 eng 70 17 Unknown

CC BY-SA 3.0

wiki_bio [Lebret et al., 2016] 1 eng 586 328

Generation wiki_hop original [Tu et al., 2019] 1 eng 6363 748

CC BY-SA 3.0

Question Answering

Custom license

Question Answering

wiki_qa [Yang et al., 2015] 1 eng 224 26

Question Answering

wiqa [Tandon et al., 2019] 1 eng 408 44 Apache-2.0

CC BY-SA 4.0

Question Answering

xquad [Artetxe et al., 2019] 2 zho, vie 652 173

xsum [Narayan et al., 2018] 1 eng 1412 250 MIT Summarization yelp_review_full [Zhang et al., 2015] 1 eng 620 91

Custom license

Sentiment Analysis

Table 12: List of xP3 datasets [Muennighoff et al., 2023c].

##### J Data Cards

Following Pushkarna et al. [2022] and the HuggingFace data card template19, we present the data card for the Aya Dataset.

###### Data Card for the Aya Dataset

|The Aya Dataset is a multilingual instruction fine-tuning dataset curated by an open-science community. The dataset contains a total of 204,114 annotated prompt-completion pairs.<br><br>• Curated by: 2,007 contributors from 110 countries<br>• Language(s): 65 languages<br>• License: Apache 2.0<br>• Repository: https://huggingface.co/datasets/CohereForAI/aya_dataset<br><br><br>Authorship Publishing Organization: Cohere For AI<br><br>Industry Type: Not-for-profit - Tech<br><br>Contact Details: https://aya.for.ai/<br><br>Example of Data Points The dataset contains multilingual prompts and completions in the following format: {prompt: "What day is followed by Saturday?", completion : "Saturday is followed by Sunday.", language: "English" }<br><br>Motivations & Intentions<br><br>Curation Rationale: The curation effort employed an open-science approach to create a diverse instruction-style dataset through annotators across the globe that ensures comprehensive representation across all languages. The success of the curation effort, led by volunteers across diverse backgrounds, was significantly influenced by their hope to meaningfully bring NLP advancements to their languages.<br><br>Provenance Methods Used<br><br>crowd-sourced through volunteer annotations, followed by a quality assessment phase in which samples from the dataset were checked.<br><br>Methodology Details Source: Original annotations and edits of opensource NLP datasets Platform: Aya Annotation Platform Dates of Collection: Jun 2023 - Dec 2023<br><br>Dataset Version and Maintenance Maintenance Status Actively Maintained<br><br>Version Details Current version: 1.0 Last Update: 12/2023 First Release: 02/2024<br><br>Maintenance Plan<br><br>Updates will be periodically made available based on volunteer contributions|
|---|

19https://huggingface.co/docs/datasets/v2.15.0/en/dataset_card

###### Data Card for the Aya Collection

|The Aya Collection incorporates instruction-style templates from fluent speakers and applies them to a curated list of 44 datasets. It also includes translations of 19 instruction-style datasets into 101 languages. This collection provides 513,579,625 instances of prompts and completions covering a wide range of tasks..<br><br>• Curated by: 2007 contributors from 110 countries<br>• Language(s): 114 languages<br>• License: Apache 2.0<br>• Repository: https://huggingface.co/datasets/CohereForAI/aya_collection<br><br><br>Authorship Publishing Organization: Cohere For AI<br><br>Industry Type: Not-for-profit - Tech<br><br>Contact Details: https://aya.for.ai<br><br>Example of Data Points The dataset contains multilingual prompts and completions in the following format: {‘prompt’: "Generate an article for the given headline: {{headline}}", ‘completion’: "{{news_article}}", ‘lang’: "English" }<br><br>Motivations & Intentions<br><br>Curation Rationale: Automatic augmentation of existing datasets serves to enhance the available linguistic resources for multiple languages. List of languages were established from mT5 and aligned with annotators’ language list and NLLB translation model. The datasets were translated directly from English for all languages.<br><br>Provenance Methods Used combination of crowd-sourced templating and automatic translation.<br><br>Methodology Details Source: Existing NLP datasets Platform: Aya Annotation Platform Dates of Collection: Jun 2023 - Dec 2023<br><br>Dataset Version and Maintenance Maintenance Status Actively Maintained<br><br>Version Details Current version: 1.0 Last updated: 12/2023 Release date: 02/2024<br><br>Maintenance Plan No updates planned.|
|---|

###### Data Card for the Aya Evaluation Suite

|The Aya Evaluation Suite contains a total of 25,750 open-ended conversation-style prompts covering 101 languages of three subsets: aya-human-annotated: 250 original human-written prompts in 7 languages each. dolly-machine-translated: 200 human-selected prompts from Conover et al. [2023], automatically translated with the NLLB model [NLLB-Team et al., 2022] from English into 101 languages. dolly-human-edited: 200 dolly-machine-translated prompts post-edited by fluent speakers for 6 languages.<br><br>• Curated by: contributors, professional annotators, and synthetic generation<br>• Language(s): 101 languages<br>• License: Apache 2.0<br>• Repository: https://huggingface.co/datasets/CohereForAI/aya_evaluation_suite<br><br><br>Authorship Publishing Organization: Cohere For AI<br><br>Industry Type: Not-for-profit - Tech<br><br>Contact Details: https://aya.for.ai<br><br>Example of Data Points The dataset contains multilingual prompts in the following format: {‘prompt’: "Which is a species of fish? Bleak or Weary", ‘lang’: "English" }<br><br>Motivations & Intentions<br><br>Curation Rationale: This evaluation suite is tailored for testing the generation quality of multilingual models, with the aim to balance language coverage and human-sourced quality. It covers prompts originally written in each language, as well as English-centric translated and manually curated or edited prompts for a linguistically broad but rich testbed. The list of languages was established from mT5 and aligned with annotators’ language list and the NLLB translation model.<br><br>Provenance Methods Used<br><br>combination of original annotations by volunteers, automatic translation, and post-editing of translations by professional annotators.<br><br>Methodology Details Source: Original annotations and translations and post-edits of Dolly Platform: Aya Annotation Platform Dates of Collection: Jun 2023 - Dec 2023<br><br>Dataset Version and Maintenance Maintenance Status Actively Maintained<br><br>Version Details Current version: 1.0 Last updated: 02/2024 Release date: 02/2024<br><br>Maintenance Plan No updates planned.|
|---|

#### K Aya Collection Templates

Dataset #Id Prompt Template Completion Template

Answer the given question in one of the following languages, [bem, fon, hau, ibo, kin,

swh, twi, wol, yor, zul, eng, fra]. {{question_in_target_language}}

{{answer_in_target_language}}

1

AfriQA-inst [Ogundepo et al., 2023]

- 1

Given the tweet, provide the sentiment associated with it.

Among which of the sentiment categories would you classify the following tweet?

positive, negative, or neutral {{tweet}}

I would classify the given tweet as: {{sentiment}}

- 2

AfriSenti-inst [Muhammad et al., 2023]

Is the tweet below expressing a positive, negative, or neutral sentiment? {{tweet}}

The tweet is expressing {{sentiment}} sentiment.

- 1

Given the context, provide the answer to the asked question.

ከዚህ በታች በተገለጸው አውድ ተከታዩን ጥያቄ ይመልሱ፡ {{context}} {{question}}

ከጥያቄው ጋር የሚስማማው ምላሽ {{answer_text}} ነው።

- 2

ከዚህ በታች ያለውን ዝርዝር መረጃ በመጠቀም ለሚከተለው ጥያቄ መልስ ይስጡ፡ {{context}}

{{question}}

ከጥያቄው አንጻር ትክክለኛው መልስ {{answer_text}} ነው።

- 3

ከዚህ በታች ያለውን ጽሑፍ በማጣቀስ እባክዎን ለሚከተለው መልስ ይስጡ {{context}}

{{question}}

ለጥያቄው መልስ {{answer_text}} ነው።

- 4

የተሰጠውን ጥያቄ ለመመለስ ከዚህ በታች የቀረበውን መረጃ ይመልከቱ፡ {{context}} {{question}}

ለተጠቀሰው ጥያቄ ትክክለኛው ምላሽ {{answer_text}} ነው።

- 5

ለሚከተለው ጥያቄ ምላሽ ለመስጠት ከዚህ በታች የቀረበውን አውድ ተጠቀም፡ {{context}}

{{question}}

ለጥያቄው መልሱ {{answer_text}} ነው።

- 6

Amharic QA [Abedissa et al., 2023]

የተሰጠውን ጥያቄ ከዚህ በታች በተሰጠው አውድ መሰረት መልሱ፡ {{context}} {{question}} ለጥያቄው ትክክለኛው ምላሽ {{answer_text}} ነው።

- 1

Generate the summary/headline of the given news article.

এই সংবােদর জন্য একিট সংবাদ িশেরানাম েলখ: {{article}}

প্রদৎত সংবােদর সংবাদ িশেরানাম হেলা, {{headline}}

- 2 একিট বােক্য েলখািটর মলূ ব ব্য তেলু ধর: {{article}} প্রদৎত অনে েদরু সংিক্ষ  মলভাবূ হেলা, {{summary}}

- 3

িনেচর অনে েদু কী বলা হেয়েছ তা সংেক্ষেপ বণর্না কর। {{article}}

প্রদৎত অনে েদরু সংিক্ষ  মলভাবূ হেলা, {{summary}}

- 4

িনেচর অনে েদু কী বলা হেয়েছ তা সংেক্ষেপ বণর্না কর। {{article}}

{{summary}}

- 5

News-summaryinstruct [TahmidH, 2023]

এক বােক্য িনেচর অনে েদরু সারাংশ েলখ। {{article}}

প্রদৎত অনে েদরু সংিক্ষ  মলভাবূ হেলা, {{summary}}

- 6 সংেক্ষেপ বাক্যিটর মলভাবূ তেলু ধেরা: {{article}} {{summary}}

- 7 সংবাদ িশেরানাম িলখন:ু {{article}} সংবাদিটর িশেরানাম হেলা, {{headline}}

- 8 সংেক্ষেপ বাক্যিটর মলভাবূ তেলু ধেরা: {{article}} বাক্যিটর সংিক্ষ  মলভাবূ হেলা, {{summary}}

- 9

িন িলিখত সংবােদর িশেরানাম কী হেত পাের? {{article}}

প্রদৎত সংবােদর সংবাদ িশেরানাম হেলা, {{headline}}

- 10

এক বােক্য িনেচর অনে েদরু সারাংশ েলখ। {{article}}

{{summary}}

- 11 আেরা কম শেব্দ বাক্যিটর মলভাবূ বণর্না কর: {{article}} {{summary}}

- 12

প্রদৎত তথ্য ব্যবহার কের একিট সংবাদ িশেরানাম িলখন:ু {{article}} সংবাদিটর িশেরানাম হেলা, {{headline}}

- 13 আেরা কম শেব্দ বাক্যিটর মলভাবূ বণর্না কর: {{article}} বাক্যিটর সংিক্ষ  মলভাবূ হেলা, {{summary}}

- 14 একিট বােক্য েলখািটর মলূ ব ব্য তেলু ধর: {{article}} {{summary}}

- 15

িন িলিখত সংবােদর িভিৎতেত একিট সংবাদ িশেরানাম িলখনু {{article}}

প্রদৎত সংবােদর সংবাদ িশেরানাম হেলা, {{headline}}

- 1

Paraphrase the following sentence.

Վերապատմեք հաջորդ նախադասությունը:

{{sentence1}}

{{sentence2}}

- 2

Գրեք հաջորդ նախադասության վերապատմումը: {{sentence1}

{{sentence2}}

- 3

Arpa-instruct [syntaxshill, 2023]

Գրեք նախադասություն վերապատմելու համար հաջորդը: {{sentence1}

{{sentence2}}

- 1

Given the name of the food item, generates the detailed recipe along with the ingredients. {{Food Item}} ఎలా తయారు చేసాత్ రో కల్ుపత్ంగా ఇవవ్ండి.

Food Item కి కావలసిన పదారాధ్ లు మరియు తయారు చేసే

విధానం ఇకక్డ ఇవవ్బడింది. కావలసిన పదారాథ్ లు: {{Ingredients}} తయారుచేసే విధానం:

{{Cooking Instructions}}

- 2

మీరు {{Food Item}} తయారు చెయయ్డానికి ఎటువంటి తిండి పదారాధ్ లు వాడుతురు మరియు ఏ విధముగా చేసాత్ రో

వివరణ ఇవవ్ండి.

Food Item కి కావలసిన పదారాధ్ లు మరియు తయారు చేసే

విధానం ఇకక్డ ఇవవ్బడింది. కావలసిన పదారాథ్ లు: {{Ingredients}} తయారుచేసే విధానం:

{{Cooking Instructions}}

- 3

Telugu-food-recipes [SuryaKrishna02,

- 2023a]

Food Item కి కావలసిన పదారాధ్ లు మరియు తయారు చేసే

విధానం ఇకక్డ ఇవవ్బడింది. కావలసిన పదారాథ్ లు: {{Ingredients}} తయారుచేసే విధానం:

నేను {{Food Item}} చాలా రుచికరంగా ఉంటుంది అని వినాన్ను. నాకు ఇది ఎలా చెయాయ్లో సంకిల్పత్ంగా చెపుప్.

{{Cooking Instructions}}

- 4

మొదటిసారి వంట చేసేవారికి చెపిప్నటట్ు నాకు Food Item ఎలా చెయాయ్లొ చెపుప్

Food Item కి కావలసిన పదారాధ్ లు మరియు తయారు చేసే

విధానం ఇకక్డ ఇవవ్బడింది. కావలసిన పదారాథ్ లు: {{Ingredients}} తయారుచేసే విధానం:

{{Cooking Instructions}}

- 5

{{Food Item}} ఎలా చేయాలి? సమాధానం లో కావలసిన పదారాధ్ లు మరియు తయారు చేసే విధానం

ఉండాలి.

Food Item కి కావలసిన పదారాధ్ లు మరియు తయారు చేసే

విధానం ఇకక్డ ఇవవ్బడింది. కావలసిన పదారాథ్ లు: {{Ingredients}} తయారుచేసే విధానం:

{{Cooking Instructions}}

- 6

{{Food Item}} ఎలా తయారు చేసాత్ ం?

Food Item కి కావలసిన పదారాధ్ లు మరియు తయారు చేసే

విధానం ఇకక్డ ఇవవ్బడింది. కావలసిన పదారాథ్ లు: {{Ingredients}} తయారుచేసే విధానం:

{{Cooking Instructions}}

- 7 {{Food Item}} రెసిపీ ఏంటి?

Food Item కి కావలసిన పదారాధ్ లు మరియు తయారు చేసే

విధానం ఇకక్డ ఇవవ్బడింది. కావలసిన పదారాథ్ లు: {{Ingredients}} తయారుచేసే విధానం:

{{Cooking Instructions}}

- 1

Given the title of a funny conversation, generate a funny conversation based on the

title. {{Title}} అనే శీరిష్క తో జోక్‍ ఇవువ్

శీరిష్క: {{Title}} {{Funny Conversation}}

- 2

{{Title}} అనే  టిల్‍ తో జోక్‍ ఇవువ్

శీరిష్క: {{Title}} {{Funny Conversation}}

- 3

ఒక హాసయ్ సంభాషణ ఇవువ్ మరియు దాని యొకక్ శీరిష్క {{Title}} ఉండే లాగా ఇవువ్.

శీరిష్క: {{Title}} {{Funny Conversation}}

- 4

ఒక చినన్ హాసయ్ సనిన్వేశం ఇవువ్ మరియు దాని యొకక్ శీరిష్క {{Title}} ఉండే లాగా ఇవువ్.

శీరిష్క: {{Title}} {{Funny Conversation}}

- 5

ఒక చమతాక్రమయిన సంభాషణ ఇవువ్ మరియు దాని యొకక్ శీరిష్క {{Title}} ఉండే లాగా ఇవువ్.

శీరిష్క: {{Title}} {{Funny Conversation}}

- 6

ఒక చినన్ చమతాక్రమయిన సనిన్వేశం ఇవువ్ మరియు దాని యొకక్ శీరిష్క {{Title}} ఉండే లాగా ఇవువ్.

శీరిష్క: {{Title}} {{Funny Conversation}}

- 7

ఒక తమాషా అయినా సంభాషణ ఇవువ్ మరియు దాని యొకక్ శీరిష్క {{Title}} ఉండే లాగా ఇవువ్.

శీరిష్క: {{Title}} {{Funny Conversation}}

- 8

ఒక చినన్ తమాషా అయినా సనిన్వేశం ఇవువ్ మరియు దాని యొకక్ శీరిష్క {{Title}} ఉండే లాగా ఇవువ్.

శీరిష్క: {{Title}} {{Funny Conversation}}

- 9

ఒక హాసయ్ సంభాషణ ఇవువ్ మరియు దాని యొకక్  టిల్‍ {{Title}} ఉండే లాగా ఇవువ్.

శీరిష్క: {{Title}} {{Funny Conversation}}

- 10

ఒక చినన్ హాసయ్ సనిన్వేశం ఇవువ్ మరియు దాని యొకక్  టిల్‍ {{Title}} ఉండే లాగా ఇవువ్.

శీరిష్క: {{Title}} {{Funny Conversation}}

- 11

Telugu-jokes [SuryaKrishna02,

- 2023b]

శీరిష్క: {{Title}} {{Funny Conversation}}

ఒక చమతాక్రమయిన సంభాషణ ఇవువ్ మరియు దాని యొకక్  టిల్‍ {{Title}} ఉండే లాగా ఇవువ్.

- 12

ఒక చినన్ చమతాక్రమయిన సనిన్వేశం ఇవువ్ మరియు దాని యొకక్  టిల్‍ {{Title}} ఉండే లాగా ఇవువ్.

శీరిష్క: {{Title}} {{Funny Conversation}}

- 13

ఒక తమాషా అయినా సంభాషణ ఇవువ్ మరియు దాని యొకక్  టిల్‍ {{Title}} ఉండే లాగా ఇవువ్.

శీరిష్క: {{Title}} {{Funny Conversation}}

- 14

శీరిష్క: {{Title}} {{Funny Conversation}}

ఒక చినన్ తమాషా అయినా సనిన్వేశం ఇవువ్ మరియు దాని యొకక్  టిల్‍ {{Title}} ఉండే లాగా ఇవువ్.

- 1

Given Title/Headline of the article, generate the article with that Title/Headline. One word from the set (enclosed within square brackets) is chosen at random and a

prompt template is created.

[కి ంది | కింది | ఇవవ్బడిన | ఇచిచ్న] [శీరిష్కతో |  టిల్‍ తో | హెడె న్‍ తో] [వారా  కథనానిన్ | నూయ్స్‍ ఆరిట్కల్‍ ని | నూయ్స్‍

కథనానిన్] [వా  యండి | రాయండి]: {{Title}}

{{Article}}

- 2

Telugu-news-articles [SuryaKrishna02,

- 2023c]

Given the article, generate the Title/Headline for the article.

One word from the set (enclosed within square brackets) is chosen at random and a

One word from the set (enclosed within square brackets) is chosen at random and a

completion template is created.

prompt template is created.

[ఇచిచ్న | ఇవవ్బడిన] [వారా  కథనానికి | నూయ్స్‍ ఆరిట్కల్‍ కి | నూయ్స్‍ కథనానికి] [సరిపోయే | తగిన | అను న] [శీరిష్క |  టిల్‍ | హెడె న్‍] ’{{Title}}’.

[కి ంది | కింది | ఇవవ్బడిన | ఇచిచ్న] [వారా  కథనానికి | నూయ్స్‍ ఆరిట్కల్‍ కి | నూయ్స్‍ కథనానికి] [శీరిష్కను |  టిల్‍ ను

| హెడె న్‍ ను] [వా  యండి | ఇవవ్ండి | రాయండి]: {{Article}}

- 1

Given the poem and type of poetry, explain the meaning of the poem.

కి ంద ఇచిచ్న {{Poetry_Type}} లోని పదాయ్నికి తాతప్రయ్ం ఇవవ్ండి: {{Poem}}

ఇచిచ్న {{Poetry_Type}} లోని పదాయ్నికి తాతప్రయ్ం: {{Meaning}}

- 2

కి ంద ఇచిచ్న {{Poetry_Type}} లోని పదాయ్నికి భావం ఇవవ్ండి: {{Poem}}

ఇచిచ్న {{Poetry_Type}} లోని పదాయ్నికి భావం: {{Meaning}}

- 3

కి ంద ఇచిచ్న {{Poetry_Type}} లోని పదాయ్నికి భావము ఇవవ్ండి: {{Poem}}

ఇచిచ్న {{Poetry_Type}} లోని పదాయ్నికి భావము: {{Meaning}}

- 4

కి ంద ఇచిచ్న {{Poetry_Type}} లోని పదాయ్నికి తాతప్రయ్ము ఇవవ్ండి: {{Poem}}

ఇచిచ్న {{Poetry_Type}} లోని పదాయ్నికి తాతప్రయ్ము: {{Meaning}}

- 5

కి ంద ఇచిచ్న {{Poetry_Type}} లోని పదాయ్నికి అరధ్ం ఇవవ్ండి: {{Poem}}

ఇచిచ్న {{Poetry_Type}} లోని పదాయ్నికి అరధ్ం: {{Meaning}}

- 6

Telugu-poems [SuryaKrishna02, 2023e]

కి ంద ఇచిచ్న {{Poetry_Type}} లోని పదాయ్నికి అరధ్ము ఇవవ్ండి: {{Poem}}

ఇచిచ్న {{Poetry_Type}} లోని పదాయ్నికి అరధ్ము: {{Meaning}}

- 7

Given the meaning and the type of poetry, generate the corresponding poem.

కి ంద ఇచిచ్న తాతప్రయ్ం వచేచ్ లాగా {{Poetry_Type}}  లిలో పదయ్ం రాయండి: {{Meaning}}

ఇచిచ్న తాతప్రయ్ం వచేచ్ {{Poetry_Type}}  లి పదయ్ం: {{Poem}}

- 8

కి ంద ఇచిచ్న భావం వచేచ్ లాగా {{Poetry_Type}}  లిలో పదయ్ం రాయండి: {{Meaning}}

ఇచిచ్న భావం వచేచ్ {{Poetry_Type}}  లి పదయ్ం: {{Poem}}

- 9

కి ంద ఇచిచ్న భావము వచేచ్ లాగా {{Poetry_Type}}  లిలో పదయ్ం రాయండి: {{Meaning}}

ఇచిచ్న భావము వచేచ్ {{Poetry_Type}}  లి పదయ్ం: {{Poem}}

- 10

కి ంద ఇచిచ్న తాతప్రయ్ము వచేచ్ లాగా {{Poetry_Type}}  లిలో పదయ్ం రాయండి: {{Meaning}}

ఇచిచ్న తాతప్రయ్ము వచేచ్ {{Poetry_Type}}  లి పదయ్ం: {{Poem}}

- 11

కి ంద ఇచిచ్న అరధ్ం వచేచ్ లాగా {{Poetry_Type}}  లిలో పదయ్ం రాయండి: {{Meaning}}

ఇచిచ్న అరధ్ం వచేచ్ {{Poetry_Type}} లి పదయ్ం: {{Poem}}

- 12

కి ంద ఇచిచ్న అరధ్ము వచేచ్ లాగా {{Poetry_Type}}  లిలో పదయ్ం రాయండి: {{Meaning}}

ఇచిచ్న అరధ్ము వచేచ్ {{Poetry_Type}}  లి పదయ్ం: {{Poem}}

- 13

Given the partial poem and type of poetry, generate the rest of the poem.

కి ంద ఇచిచ్న తాతప్రయ్ం అనుసరించి అసంపూరణ్ యిన పదాయ్నిన్ {{Poetry_Type}}  లిలో పూరి చేసి రాయండి:

తాతప్రయ్ం: {{Meaning}}

అసంపూరణ్ యిన పదయ్ం: {{Partial Poem}}

పూరి చేయబడడ్ పదయ్ం కి ంద ఇవవ్బడింది: పదయ్ం: {{Poem}}

- 14

కి ంద ఇచిచ్న భావం అనుసరించి అసంపూరణ్ యిన పదాయ్నిన్ {{Poetry_Type}}  లిలో పూరి చేసి రాయండి: భావం: {{Meaning}}

అసంపూరణ్ యిన పదయ్ం: {{Partial Poem}}

పూరి చేయబడడ్ పదయ్ం కి ంద ఇవవ్బడింది: పదయ్ం: {{Poem}}

- 15

కి ంద ఇచిచ్న భావము అనుసరించి అసంపూరణ్ యిన పదాయ్నిన్ {{Poetry_Type}}  లిలో పూరి చేసి రాయండి:

భావము: {{Meaning}}

అసంపూరణ్ యిన పదయ్ం: {{Partial Poem}}

పూరి చేయబడడ్ పదయ్ం కి ంద ఇవవ్బడింది: పదయ్ం: {{Poem}}

- 16

కి ంద ఇచిచ్న తాతప్రయ్ము అనుసరించి అసంపూరణ్ యిన పదాయ్నిన్ {{Poetry_Type}}  లిలో పూరి చేసి రాయండి:

తాతప్రయ్ము: {{Meaning}}

పూరి చేయబడడ్ పదయ్ం కి ంద ఇవవ్బడింది: పదయ్ం: {{Poem}}

అసంపూరణ్ యిన పదయ్ం: {{Partial Poem}}

- 17

కి ంద ఇచిచ్న అరధ్ం అనుసరించి అసంపూరణ్ యిన పదాయ్నిన్ {{Poetry_Type}}  లిలో పూరి చేసి రాయండి: అరధ్ం: {{Meaning}}

అసంపూరణ్ యిన పదయ్ం: {{Partial Poem}}

పూరి చేయబడడ్ పదయ్ం కి ంద ఇవవ్బడింది: పదయ్ం: {{Poem}}

- 18

కి ంద ఇచిచ్న అరధ్ము అనుసరించి అసంపూరణ్ యిన పదాయ్నిన్ {{Poetry_Type}}  లిలో పూరి చేసి రాయండి:

అరధ్ము: {{Meaning}}

పూరి చేయబడడ్ పదయ్ం కి ంద ఇవవ్బడింది: పదయ్ం: {{Poem}}

అసంపూరణ్ యిన పదయ్ం: {{Partial Poem}}

Given a premise, provide the hypothesis based on the label value. The label can be either entailment or contradiction. Entailment Prompts and Targets:

FarsTail-Instruct [hghader1, 2023] 1

:     ﺍﺭ   ﺯ ﻥﺍ   ﻩ   ﻩﺩﺍﺩ ﺯﺍ {{hypothesis}}

:     ﻥﺍ   ﯼﺍ   ﺯ ﺯﺍ {{premise}}

- 2 :     ﻥﺍ   ﯼﺍ   ﺯ ﺯﺍ

{{premise}}

:     ﻩ   ﻩﺩﺍﺩ ﺯﺍ ﻥﺍ   ﺍﺭ   ﺯ {{hypothesis}}

- 3 :     ﻥﺍ   ﯼﺍ   ﺯ ﺯﺍ

{{premise}}

:     ﺍﺭ   ﺯ ﻥﺍ   ﻩ   ﻩﺩﺍﺩ {{hypothesis}}

- 4 :       ﺯ   ﺍ   ﯼﺍ

{{premise}}

:     ﺍﺭ   ﺯ ﻥﺍ   ﻩ   ﻩﺩﺍﺩ ﺯﺍ {{hypothesis}}

- 5 :       ﺯ   ﺍ   ﯼﺍ

{{premise}}

:     ﻩ   ﻩﺩﺍﺩ ﺯﺍ ﻥﺍ   ﺍﺭ   ﺯ {{hypothesis}}

- 6 :       ﺯ   ﺍ   ﯼﺍ

{{premise}}

:     ﺍﺭ   ﺯ ﻥﺍ   ﻩ   ﻩﺩﺍﺩ {{hypothesis}}

- 7 :       ﺯ ﺯﺍ ﻥﺍ   ﯼﺍ

{{premise}}

:     ﺍﺭ   ﺯ ﻥﺍ   ﻩ   ﻩﺩﺍﺩ ﺯﺍ {{hypothesis}}

- 8 :       ﺯ ﺯﺍ ﻥﺍ   ﯼﺍ

{{premise}}

:     ﻩ   ﻩﺩﺍﺩ ﺯﺍ ﻥﺍ   ﺍﺭ   ﺯ {{hypothesis}}

- 9 :       ﺯ ﺯﺍ ﻥﺍ   ﯼﺍ

{{premise}}

:     ﺍﺭ   ﺯ ﻥﺍ   ﻩ   ﻩﺩﺍﺩ {{hypothesis}}

- 10

Contradiction Prompts and Targets:

:       ﺯ ﯼﺍ {{premise}}

:    ﺍﺭ ﻩ   ﻩﺩﺍﺩ   ﺯ {{hypothesis}}

- 11 :       ﺯ ﯼﺍ

{{premise}}

:ﺩﺭﺍﺩ ﻩ   ﻩﺩﺍﺩ   ﺯ {{hypothesis}}

- 12 :       ﺯ ﯼﺍ

{{premise}}

:  ﺍ ﻩ   ﻩﺩﺍﺩ   ﺯ {{hypothesis}}

- 13 :    ﺍﺭ   ﺯ ﯼﺍ

{{premise}}

:    ﺍﺭ ﻩ   ﻩﺩﺍﺩ   ﺯ {{hypothesis}}

- 14 :    ﺍﺭ   ﺯ ﯼﺍ

{{premise}}

:ﺩﺭﺍﺩ ﻩ   ﻩﺩﺍﺩ   ﺯ {{hypothesis}}

- 15 :    ﺍﺭ   ﺯ ﯼﺍ

:  ﺍ ﻩ   ﻩﺩﺍﺩ   ﺯ {{hypothesis}}

{{premise}}

- 16 :        ﺯ ﯼﺍ

{{premise}}

:    ﺍﺭ ﻩ   ﻩﺩﺍﺩ   ﺯ {{hypothesis}}

- 17 :        ﺯ ﯼﺍ

{{premise}}

:ﺩﺭﺍﺩ ﻩ   ﻩﺩﺍﺩ   ﺯ {{hypothesis}}

- 18 :        ﺯ ﯼﺍ

:  ﺍ ﻩ   ﻩﺩﺍﺩ   ﺯ {{hypothesis}}

{{premise}}

- 1

Given an article, generate the summary of the article.

इस के  लए एक सारांश बनाएं: {{Article}}

यह एक सारांश है: {{Summary}}

- 2

इस पाठ का सारांश बनाएं: {{Article}}

सं क्षप्त पाठ: {{Summary}}

- 3

इस पाठ का सारांश बनाएं: {{Article}} {{Summary}}

- 4

एक सारांश बनाओ: {{Article}} {{Summary}}

- 5

इस पाठ का सारांश बनाओ: {{Article}}

सारांश: {{Summary}}

- 6

एक सारांश बनाओ: {{Article}}

यहाँ एक सारांश है: {{Summary}}

- 7

इस पाठ का सारांश बनाओ: {{Article}}

सं क्षप्त सारांश: {{Summary}}

- 8

 दए गए पाठ के  लए एक सारांश बनाएं: {{Article}}

 दए गए पाठ का सारांश यह हो सकता है: {{Summary}}

- 9

एक सारांश बनाओ: {{Article}}

सारांश: {{Summary}}

- 10

 दए गए पाठ के  लए एक सारांश बनाएं: {{Article}}

सं क्षप्त सारांश: {{Summary}}

- 11

इस के  लए एक सारांश बनाएं: {{Article}}

यहाँ एक सारांश है: {{Summary}}

- 12

इस पाठ का सारांश बनाएं: {{Article}}

सारांश: {{Summary}}

- 13

एक सारांश बनाओ: {{Article}}

यह एक सारांश है: {{Summary}}

- 14

 दए गए पाठ के  लए एक सारांश बनाएं: {{Article}}

सं क्षप्त पाठ: {{Summary}}

- 15

इस पाठ का सारांश बनाओ: {{Article}}

यहाँ एक सारांश है: {{Summary}}

- 16

इस के  लए एक सारांश बनाएं: {{Article}}

सारांश: {{Summary}}

- 17

Hindi-articlesummarization [ganeshjcs, 2023a]

इस पाठ का सारांश बनाओ: {{Article}}

सं क्षप्त पाठ: {{Summary}}

- 18

एक सारांश बनाओ: {{Article}}

सं क्षप्त सारांश: {{Summary}}

- 19

इस पाठ का सारांश बनाओ: {{Article}}

 दए गए पाठ का सारांश यह हो सकता है: {{Summary}}

- 20

इस के  लए एक सारांश बनाएं: {{Article}} {{Summary}}

- 21

इस के  लए एक सारांश बनाएं: {{Article}}

सं क्षप्त पाठ: {{Summary}}

- 22

एक सारांश बनाओ: {{Article}}

 दए गए पाठ का सारांश यह हो सकता है: {{Summary}}

- 23

 दए गए पाठ के  लए एक सारांश बनाएं: {{Article}}

यह एक सारांश है: {{Summary}}

- 24

इस पाठ का सारांश बनाओ: {{Article}} {{Summary}}

- 25

 दए गए पाठ के  लए एक सारांश बनाएं: {{Article}} {{Summary}}

- 26

इस पाठ का सारांश बनाओ: {{Article}}

यह एक सारांश है: {{Summary}}

- 27

इस पाठ का सारांश बनाएं: {{Article}}

 दए गए पाठ का सारांश यह हो सकता है: {{Summary}}

- 28

इस पाठ का सारांश बनाएं: {{Article}}

यह एक सारांश है: {{Summary}}

- 29

इस पाठ का सारांश बनाएं: {{Article}}

सं क्षप्त सारांश: {{Summary}}

- 30

 दए गए पाठ के  लए एक सारांश बनाएं: {{Article}}

सारांश: {{Summary}}

- 31

इस पाठ का सारांश बनाएं: {{Article}}

यहाँ एक सारांश है: {{Summary}}

- 32

इस के  लए एक सारांश बनाएं: {{Article}}

 दए गए पाठ का सारांश यह हो सकता है: {{Summary}}

- 33

एक सारांश बनाओ: {{Article}}

सं क्षप्त पाठ {{Summary}}

- 34

 दए गए पाठ के  लए एक सारांश बनाएं: {{Article}}

यहाँ एक सारांश है: {{Summary}}

- 35

इस के  लए एक सारांश बनाएं: {{Article}}

सं क्षप्त सारांश: {{Summary}}

- 1

Given the title of an article, generate the article.

एक लेख  लखें  जसका शीषर्क इस प्रकार है: {{Title}}

लेख:

{{Article}}

- 2

Hindi-articlegeneration [ganeshjcs, 2023b]

एक लेख  लखें  जसका शीषर्क इस प्रकार है: {{Title}} {{Article}}

- 3

एक लेख  लखें  जसका शीषर्क इस प्रकार है: {{Title}}

 दए गए शीषर्क के अनुरूप एक पाठ यह हो सकता है: {{Article}}

- 4

यह शीषर्क है, इसके  लए एक लेख  लखें: {{Title}} {{Article}}

- 5

इसके  लए एक लेख  लखें: {{Title}}

लेख:

{{Article}}

- 6

इस शीषर्क के साथ एक लेख  लखें: {{Title}}

यह एक लेख है: {{Article}}

- 7

इस शीषर्क के साथ एक लेख  लखें: {{Title}}

 दए गए शीषर्क के अनुरूप एक पाठ यह हो सकता है: {{Article}}

- 8

यह शीषर्क है, इसके  लए एक लेख  लखें: {{Title}}

लेख:

{{Article}}

- 9

इसके  लए एक लेख  लखें: {{Title}} {{Article}}

- 10

इसके  लए एक लेख  लखें: {{Title}}

यह एक लेख है: {{Article}}

- 11

इस शीषर्क के साथ एक लेख  लखें: {{Title}}

लेख:

{{Article}}

- 12

इसके  लए एक लेख  लखें: {{Title}}

 दए गए शीषर्क के अनुरूप एक पाठ यह हो सकता है: {{Article}}

- 13

यह शीषर्क है, इसके  लए एक लेख  लखें: {{Title}}

 दए गए शीषर्क के अनुरूप एक पाठ यह हो सकता है: {{Article}}

- 14

एक लेख  लखें  जसका शीषर्क इस प्रकार है: {{Title}}

यह एक लेख है: {{Article}}

- 15

इस शीषर्क के साथ एक लेख  लखें: {{Title}} {{Article}}

- 16

यह शीषर्क है, इसके  लए एक लेख  लखें: {{Title}}

यह एक लेख है: {{Article}}

- 1

Given the movie review, identify the sentiment.

Is deze recensie positief of negatief?

Gegeven de recensie, mijn antwoord is {{Sentiment}}

- 2

Is deze recensie positief of negatief? Deze recensie is {{Sentiment}}

- 3

Is deze recensie positief of negatief? De beoordeling hier is {{Sentiment}}

- 4

Is deze recensie positief of negatief? Het antwoord is {{Sentiment}}

- 5

Wat is het sentiment van de recensie? De recensie is {{Sentiment}}

- 6

Wat is het sentiment van de recensie? Gegeven de recensie, mijn antwoord is

{{Sentiment}}

- 7

Wat is het sentiment van de recensie? Deze recensie is {{Sentiment}}

- 8

IMDB-Dutch-instruct [jjzha, 2023]

Wat is het sentiment van de recensie? De beoordeling hier is {{Sentiment}}

- 9

Wat is het sentiment van de recensie? Het antwoord is {{Sentiment}}

- 10

Wat voor toon heeft de volgende recensie? De recensie is {{Sentiment}}

- 11

Wat voor toon heeft de volgende recensie? Gegeven de recensie, mijn antwoord is

{{Sentiment}}

- 12

Wat voor toon heeft de volgende recensie? Deze recensie is {{Sentiment}}

- 13

Wat voor toon heeft de volgende recensie? De beoordeling hier is {{Sentiment}}

- 14

Wat voor toon heeft de volgende recensie? Het antwoord is {{Sentiment}}

- 15

Met wat voor sentiment zou je deze recensie beoordelen? De recensie is {{Sentiment}}

- 16

Met wat voor sentiment zou je deze recensie beoordelen?

Gegeven de recensie, mijn antwoord is {{Sentiment}}

- 17

Met wat voor sentiment zou je deze recensie beoordelen? Deze recensie is {{Sentiment}}

- 18

Met wat voor sentiment zou je deze recensie beoordelen? De beoordeling hier is {{Sentiment}}

- 19

Met wat voor sentiment zou je deze recensie beoordelen? Het antwoord is {{Sentiment}}

- 20

Is deze recensie positief of negatief? De recensie is {{Sentiment}}

- 1

Generate the translation for the given English review to one of the target Indic languages [ben, guj, hin, kan, mal, mar, pan, tam, tel

and urd].

Translate from English to {{target_language}}: {{English_Review}}

{{Indic_Review}}

- 2

Translate this sentence to {{target_language}}: {{English_Review}}

{{Indic_Review}}

- 3

What’s the {{target_language}} translation of this sentence: {{English_Review}}

{{Indic_Review}}

- 4

IndicSentiment-inst [el2e10, 2023a]

Can you translate this text to {{target_language}}: {{English_Review}}

{{Indic_Review}}

IndicXParaphraseinst [el2e10, 2023b; SuryaKrishna02, 2023d]

- 1

Generate the paraphrase of the given sentence in one of the Indic languages [ben,

guj, hin, mar, pan, mal, tel] িভ  শব্দগু  ব্যবহার কের িনেচর বাক্যিট েলখ: "{{original_sentence}}".

{{paraphrased_sentence}}

- 2

িনেচর বাক্যিট িভ ভােব েলখ: "{{original_sentence}}" {{paraphrased_sentence}}

- 3

অেথর্র পিরবতর্ন না কের িনেচর বাক্যিট নতনভােবু েলখ: "{{original_sentence}}" {{paraphrased_sentence}}

- 4

નીચેના વા ને અલગ શબ્દોનો ઉપયોગ કરીને લખો: ”{{original_sentence}}”. {{paraphrased_sentence}}

- 5

નીચેના વા ને અલગ રીતે ફરીથી લખો: ”{{original_sentence}}” {{paraphrased_sentence}}

- 6

નીચેના વા ને બીજા સ્વરૂપમાં ફરીથી લખો: ”{{original_sentence}}” {{paraphrased_sentence}}

- 7

दुसरे शब्दों का प्रयोग करके इस वाक्य को  ल खए: ”{{original_sentence}}”. {{paraphrased_sentence}}

- 8

इस वाक्य को अन्य तरीके से  फर से  ल खए: ”{{original_sentence}}” {{paraphrased_sentence}}

- 9

 नम्न ल खत वाक्य का अथर् बदले  बना उसे दोबारा  ल खए: ”{{original_sentence}}” {{paraphrased_sentence}}

- 10

खालील वाक्य दुसरे- भ  शब्द वापरून  लहा: ”{{original_sentence}}”. {{paraphrased_sentence}}

- 11

खालील वाक्य वेगळ्या प्रकारे पुन्हा  लहा: ”{{original_sentence}}” {{paraphrased_sentence}}

- 12

खालील वाक्य दुसरे शब्द वापरून रूपांत रत-अनुवा दत करा: ”{{original_sentence}}” {{paraphrased_sentence}}

- 13

ਵੱਖ-ਵੱਖ ਸ਼ਬਦਾਂ ਦੀ ਵਰਤੋ ਕਰਕੇ ਹੇਠਾਂ ਿਦੱਤੇ ਵਾਕ ਨੂ ਿਲਖੋ: ”{{original_sentence}}”. {{paraphrased_sentence}}

- 14

ਿਨਮਨਿਲਖਤ ਵਾਕ ਨੂ ਵੱਖਰੇ ਤਰੀਕੇ ਨਾਲ ਦੁਬਾਰਾ ਿਲਖੋ: ”{{original_sentence}}” {{paraphrased_sentence}}

- 15

ਹੇਠਾਂ ਿਦੱਤੇ ਵਾਕ ਨੂ ਸਮਝਾਓ: ”{{original_sentence}}” {{paraphrased_sentence}

- 16

ഇനിപ്പറയുന്ന വാചകം വ്യത്യസ്ത

വാക്കുകളിൽ എഴുതുക: ”{{original_sentence}}”.

{{paraphrased_sentence}}

- 17

ഇനിപ്പറയുന്ന വാചകം മെറ്റാരു രീതിയിൽ എഴുതുക: ”{{original_sentence}}”

{{paraphrased_sentence}}

- 18

താെഴപ്പറയുന്ന വാചകം പരാവർത്തനം െചയ്യ ക: ”{{original_sentence}}”

{{paraphrased_sentence}}

- 19

Given a sentence, generate a sentence with similar meaning.

ఈ కి ంది వాకయ్ం మరోరీతిలో రాయి: {{Original Sentence}}

{{Paraphrased Sentence}}

- 20 ఈ వాకయ్ం మరోరీతిలో రాయి: {{Original Sentence}} {{Paraphrased Sentence}}

- 21

ఈ కి ంది వాకయ్ం ఇంకొలాగా రాయి: {{Original Sentence}} {{Paraphrased Sentence}}

- 22 ఈ వాకయ్ం ఇంకొలాగా రాయి: {{Original Sentence}} {{Paraphrased Sentence}}

- 23

ఈ కి ంది వాకయ్ం మరోరకంగా రాయి: {{Original Sentence}} {{Paraphrased Sentence}}

- 24 ఈ వాకయ్ం మరోరకంగా రాయి: {{Original Sentence}} {{Paraphrased Sentence}}

- 1

Given the instruction in Indonesian, translate the given input to one of the three languages

[ind, sun, jav]. Translate from English to Indonesian.

Terjemahkanlah penggalan teks cerita anak berikut dari teks berbahasa Inggris ke teks

dalam Bahasa Indonesia: {{eng_input}}

Terjemahan atau padanan teks tersebut dalam Bahasa Indonesia adalah:

{{ind_output}}

- 2

Translate from Javanese to Indonesian.

Terjemahkanlah penggalan teks cerita anak berikut dari teks berbahasa Jawa ke teks

dalam Bahasa Indonesia: {{jav_input}}

Terjemahan atau padanan teks tersebut dalam Bahasa Indonesia adalah:

{{ind_output}}

- 3

Translate from Sudanese to Indonesian.

Terjemahkanlah penggalan teks cerita anak berikut dari teks berbahasa Sunda ke teks

dalam Bahasa Indonesia: {{sun_input}}

Terjemahan atau padanan teks tersebut dalam Bahasa Indonesia adalah:

{{ind_output}}

- 4

Given the instruction in Javanese, translate the given input to one of the three languages

[ind, sun, jav] Translate from English to Javanese.

Terjemahno penggalan teks crito ing ngisor iki saka Bahasa Inggris dadi teks crito ing

Basa Jawa: {{eng_input}}

Terjemahane utawa padanan teks crito kasebut ing Basa Jawa yaiku:

{{jav_output}}

- 5

Translate from Indonesian to Javanese.

Terjemahno penggalan teks crito ing ngisor iki saka Bahasa Indonesia dadi teks crito ing

Basa Jawa: {{ind_input}}

Terjemahane utawa padanan teks crito kasebut ing Basa Jawa yaiku:

{{jav_output}}

- 6

Indo-stories-instruct [Iftitahu, 2023a;b;c]

Translate from English to Sudanese.

Terjemahane utawa padanan teks crito kasebut ing Basa Jawa yaiku:

Terjemahno penggalan teks crito ing ngisor iki saka Bahasa Sunda dadi teks crito ing

Basa Jawa: {{sun_input}}

{{jav_output}}

- 7

Given the instruction in Sudanese, translate the given input to one of the three languages

[ind, sun, jav] Translate from English to Sudanese. Tarjamahkeun teks dongeng barudak di handap tina teks basa Inggris kana teks basa

Sunda: {{eng_input}}

Tarjamahan atawa sasaruaan naskah dina basa Sunda:

{{sun_output}}

- 8

Translate from Indonesian to Sudanese.

Tarjamahkeun teks dongeng barudak di handap tina teks basa Indonesia kana teks

basa Sunda: {{ind_input}}

Tarjamahan atawa sasaruaan naskah dina basa Sunda:

{{sun_output}}

- 9

Translate from Javanese to Sudanese. Tarjamahkeun teks dongeng barudak di handap tina teks basa Jawa kana teks basa

Tarjamahan atawa sasaruaan naskah dina basa Sunda:

Sunda: {{jav_input}}

{{sun_output}}

- 1

Identify the joke’s punchline or explain it. {{joke}} What do you think is the punchline of the above joke?

{{explanation}}

- 2

Joke-explaination-inst [theblackcat102, 2023]

Explain the following joke? {{joke}} {{explanation}}

- 1

Translate the given sentence from Ligurian to Italian.

Traduxi in italian: {{sentence}}

A traduçion in italian do testo a l’é: {{sentence}}

- 2

Traduxi da-o zeneise à l’italian: {{sentence}}

A traduçion in italian do testo a l’é: {{sentence}}

- 3

Traduxi da-o ligure à l’italian: {{sentence}}

A traduçion in italian do testo a l’é: {{sentence}}

- 4

Traduxi sto testo in italian: {{sentence}}

A traduçion in italian do testo a l’é: {{sentence}}

- 5

Traduxi in lengua italiaña: {{sentence}}

A traduçion in italian do testo a l’é: {{sentence}}

- 6

Traduxi sto testo da-o zeneise à l’italian: {{sentence}}

A traduçion in italian do testo a l’é: {{sentence}}

- 7

Traduxi sto testo da-o ligure à l’italian: {{sentence}}

A traduçion in italian do testo a l’é: {{sentence}}

- 8

Comm’à l’é a traduçion italiaña de sto testo? {{sentence}}

A traduçion in italian do testo a l’é: {{sentence}}

- 9

Lijnews-instruct [ConseggioLigure, 2023a;b]

Quæ a l’é a traduçion italiaña de sto testo? {{sentence}}

A traduçion in italian do testo a l’é: {{sentence}}

- 10

Ti peu tradue sto testo in italian? {{sentence}}

A traduçion in italian do testo a l’é: {{sentence}}

- 11

Translate the given sentence from Italian to Ligurian.

Traduci in genovese: {{sentence}}

La traduzione in genovese del testo è: {{sentence}}

- 12

Traduci in ligure: {{sentence}}

La traduzione in genovese del testo è: {{sentence}}

- 13

Traduci dall’italiano al genovese: {{sentence}}

La traduzione in genovese del testo è: {{sentence}}

- 14

Traduci dall’italiano al ligure: {{sentence}}

La traduzione in genovese del testo è: {{sentence}}

- 15

Traduci dall’italiano al ligure (genovese): {{sentence}}

La traduzione in genovese del testo è: {{sentence}}

- 16

Traduci questo testo in genovese: {{sentence}}

La traduzione in genovese del testo è: {{sentence}}

- 17

Traduci in lingua genovese: {{sentence}}

La traduzione in genovese del testo è: {{sentence}}

- 18

Qual è la traduzione genovese di questo testo? {{sentence}}

La traduzione in genovese del testo è: {{sentence}}

- 19

Puoi tradurre questo testo in genovese? {{sentence}}

La traduzione in genovese del testo è: {{sentence}}

- 1

Following the instruction, please answer. {{answer}}

{{translated_sentence}}

- 2

Please answer the following question. 以下の質問に答えてください、{{question}}

この質問の答えは {{answer}}

- 3

LLM-JapaneseVanilla-inst

- [Tellarin.ai, 2023a]

Answer the following question. 以下の質問に答えてください、{{question}}

{{answer}}

- 1

Given the text, generate the headline for the text and vice versa.

Generate a headline for the following text: {{text}}

Sure, here’s a headline for the given text {{headline}}

- 2

MasakhaNEWS-inst [Adelani et al., 2023]

Generate an article for the following headline: {{headline}} {{article}}.

Give an example of the trivia in the provided category.

{{question}} {{answer}}

- 1

Mintaka-inst [Sen et al., 2022]

:    ﺍ ﻩ        ﺍ ﺕ       ﺍ    ً           ﺃ {{category}}

- 2

Give an example of the trivia in the provided category.

Give me an example of trivia in this category: {{category}}

{{question}} {{answer}}

- 3

Answer the question from the given category. The following query in {{lang}} is taken from the {{category}} category. What could be the answer to the question? {{question}}

{{answer}}

- 4

Find the category of the given question. What category is this question from: {{question}}.

The category from which the question comes is {{category}}

- 5

Answer the given question. What is the correct answer to this question: {{question}}

The correct answer to the question is {{answer}}

- 6

Identify the category of the question. cual es el tema principal de esta pregunta: {{question}}

{{category}}

- 7

Give me a trivia based on the words. Write me a trivia question that contains the words : {{comma_separated_words}}

Sure, a possible question is ”{{question}}”

Identify the date or time mentions and their types in the given sentence based on the

example provided.

ﺓﺭ     ﺍ    ﺍ  ﺃﻭ ﺕ  ﻭ  ﺍﻭ   ﺭﺍ   ﺍ :          ﺍ      ﺍ

"Results": [{ "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}, "TypeName": "{{type_name}}", "Resolution": {

.    ﺍﺫ ﺍﺫﺇ       ﺍ     ﺍﻭ   ﺭ   ﺍ ﺭ      ﺍ JSON،       ﺍ ﺝﺍ  ﺇ .ﻡ    ﺍ ﻝ    ﺍ   ﺇ     ﺍ

.ﻩ  ﺩﺃ ﺭ     ﺍ ﻝ      ﻉ   ﺍﻭ :      ﺍﻭ   ﻭﻭ   ﺭ  ﻭ ﻝ   

"values": [{ "timex": "{{time}}", "type": "{{type}}", "value": "{{value}}"

- 1

NTX-LLM-inst

- [Tellarin.ai, 2023b]

{{example_sentence}}

}] }

:      ﺍ     ﺍﻭ   ﺭ   ﺍ

{{reference_date}} {{reference_time}}

}]

؟      ﺍ ،ﻩ  ﺩﺃ       ﺍ      ﺍ   ﺇ

{{input_sentence}}

- 2

请在下面提供的输入句子中识别所有日期或时间 提及及其类型。

如果相关，请考虑提供的参考日期时间。

请注意提供的示例。您应该以 JSON 格式输出结 果，遵循与所提供的示例结果类似的结构。 例句、参考日期和结果： {{example_sentence}}

参考日期: {{reference_date}} {{reference_time}} 考虑到下面的输入句子，输出结果是什么？ {{input_sentence}}

"Results": [{ "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}, "TypeName": "{{type_name}}", "Resolution": {

"values": [{ "timex": "{{time}}", "type": "{{type}}", "value": "{{value}}"

}] }

}]

- 3

Identificeer alle data en tijd vermeldingen in de zin hieronder.

Houd rekening met, indien relevant, de opgegeven referentie datum/tijd.

Let goed op het gegeven voorbeeld. De resultaten moeten in JSON formaat zijn, hetzelfde als de structuur van het gegeven

voorbeeld. Voorbeeld zin, referentie datum/tijd, en resultaten: {{example_sentence}} Referentie: {{reference_date}} {{reference_time}} Gegeven de zin hieronder, wat is het resultaat? {{input_sentence}}

"Results": [{ "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}, "TypeName": "{{type_name}}", "Resolution": {

"values": [{ "timex": "{{time}}", "type": "{{type}}", "value": "{{value}}"

}] }

}]

- 4

Please identify all date or time mentions and their types in the input sentence provided

below. Take into consideration the provided reference datetime, if relevant.

"Results": [{ "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}, "TypeName": "{{type_name}}", "Resolution": {

Pay attention to the provided example. You should output the results in JSON format, following a similar structure to the example

result provided. Example sentence, reference datetime, and results: {{example_sentence}} Reference datetime: {{reference_date}} {{reference_time}} Considering the input sentence below, what is

"values": [{ "timex": "{{time}}", "type": "{{type}}", "value": "{{value}}"

}] }

}]

the output result? {{input_sentence}}

- 5

Identifier toutes les références de date ou d’heure et leurs types dans la phrase fournie

ci-dessous. Tenez compte de la date/heure de référence fournie, le cas échéant.

Faites attention à l’exemple fourni. Vous devez aﬀicher les résultats au format JSON, en suivant une structure similaire à celle de

l’exemple de résultat fourni. Exemple de phrase, date/heure de référence et résultats: {{example_sentence}} Date/heure de référence: {{reference_date}} {{reference_time}} Compte tenu de la phrase d’entrée ci-dessous, quel est le résultat de sortie? {{input_sentence}}

"Results": [{ "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}, "TypeName": "{{type_name}}", "Resolution": {

"values": [{ "timex": "{{time}}", "type": "{{type}}", "value": "{{value}}"

}] }

}]

- 6

Bitte identifiziere alle Vorkommen von Datum oder Uhrzeiten, mit ihrem jeweiligen

Typ, im Inputsatz unten. Beachte, wenn nötig, die angegebene Referenz-Datetime.

"Results": [{ "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}, "TypeName": "{{type_name}}", "Resolution": {

Achte auf das vorgegebene Beispiel. Gib die Ergebnisse im JSON-Format aus und folge einer ähnlichen Struktur wie das

Beispielergebnis. Beispielsatz, Referenz-Datetime, und Ergebnisse: {{example_sentence}} Referenz-Datetime: {{reference_date}} {{reference_time}} Bezogen auf den Inputsatz unten, was ist das

"values": [{ "timex": "{{time}}", "type": "{{type}}", "value": "{{value}}"

}] }

}]

richtige Ergebnis? {{input_sentence}}

- 7

कृ पया  नम्न ल खत वाक्यों में सभी  दनांक या समय उल्लेखों और उनके प्रकारों क  पहचान करें।

यद  उ चत हो, तो  दये गए संद भ क  दनांक-समय को ध्यान में रखें।

 दये गए उदाहरण पर ध्यान दें । आपको  दये गए उदारहण प रणाम के समान संरचना का पालन करते हुए, JSON

प्रारूप में प रणाम आउटपुट करना चा हए । वाक्य, संद भ क  दनांक-समय एवं प रणाम का उदारहण : {{example_sentence}} संद भ क  दनांक-समय:

{{reference_date}} {{reference_time}}

 नम्न ल खत इनपुट वाक्य को ध्यान में रखते हुए, उसके आउटपुट प रणाम के बारे में  लखें ।

{{input_sentence}}

"Results": [{ "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}, "TypeName": "{{type_name}}", "Resolution": {

"values": [{ "timex": "{{time}}", "type": "{{type}}", "value": "{{value}}"

}] }

}]

- 8

Si prega di identificare tutte le menzioni di data o ora e i relativi tipi nella frase di input

fornita di seguito. Prendere in considerazione la data/ora di riferimento fornita, se pertinente.

"Results": [{ "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}, "TypeName": "{{type_name}}", "Resolution": {

Prestare attenzione all’esempio fornito. Dovresti restituire i risultati in formato JSON, seguendo una struttura simile al

risultato di esempio fornito. Frase di esempio, data/ora di riferimento e risultati: {{example_sentence}} Data e ora di riferimento: {{reference_date}} {{reference_time}} Considerando la frase di input seguente, qual è il risultato di output? {{input_sentence}}

"values": [{ "timex": "{{time}}", "type": "{{type}}", "value": "{{value}}"

}] }

}]

- 9

以下に示す入力文で、すべての日付または時刻 の言及とその種類を識別してください。

関連する場合は、提供される参照日時を考慮し てください。

提供された例に注意してください。提供された サンプル結果と同様の構造に従って、JSON 形

式で結果を出力する必要があります。 文、参照日時、結果の例： {{example_sentence}} 参照日時：{{reference_date}} {{reference_time}} 次の入力文を考慮して、出力結果は何ですか。 {{input_sentence}}

"Results": [{ "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}, "TypeName": "{{type_name}}", "Resolution": {

"values": [{ "timex": "{{time}}", "type": "{{type}}", "value": "{{value}}"

}] }

}]

- 10

아래에제공된입력문장에서날짜나시간에관한언급및 그들의유형을모두식별하십시오.

"Results": [{ "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}, "TypeName": "{{type_name}}", "Resolution": {

해당되는경우제공된참조날짜/시간을고려하십시오. 제공된예시에주목해주십시오. 결과를 JSON 형식 으로출력하여, 예시결과와유사한구조를갖도록해야

합니다. 예시문장, 참조날짜및결과: {{example_sentence}}

"values": [{ "timex": "{{time}}", "type": "{{type}}", "value": "{{value}}"

참조날짜: {{reference_date}} {{reference_time}} 아래의입력문장을고려하면결과는무엇입니까? {{input_sentence}}

}] }

}]

- 11

Por favor identifique todas as menções de datas ou tempo e seus tipos na frase

fornecida abaixo. Leve em consideração a data e hora de referência fornecida, se relevante.

Preste atenção ao exemplo fornecido. Você deve gerar resultados no formato JSON, seguindo uma estrutura semelhante ao

resultado do exemplo fornecido. Frase de exemplo, data e hora de referência e resultados: {{example_sentence}} Data e hora de referência:

{{reference_date}} {{reference_time}}

Considerando a frase abaixo, qual é o resultado produzido? {{input_sentence}}

"Results": [{ "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}, "TypeName": "{{type_name}}", "Resolution": {

"values": [{ "timex": "{{time}}", "type": "{{type}}", "value": "{{value}}"

}] }

}]

- 12

Por favor identifica todas las menciones de fechas u horas y sus categorías en el texto de

entrada que se encuentra de bajo.

Ten en cuenta la fecha y hora de referencia proporcionadas, en caso de que sea relevante.

"Results": [{ "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}, "TypeName": "{{type_name}}", "Resolution": {

Presta atención al ejemplo dado. Debes devolver los resultados en formato JSON, siguiendo una estructura similar a la del

ejemplo de resultado. Texto de ejemplo, fecha y hora, y los resultados correspondientes: {{example_sentence}} Fecha y hora de referencia:

"values": [{ "timex": "{{time}}", "type": "{{type}}", "value": "{{value}}"

{{reference_date}} {{reference_time}}

}] }

}]

Ahora considera el texto de entrada que está debajo, ¿cuál sería el resultado?

{{input_sentence}}

Lütfen aşağıda verilen cümledeki tüm tarih veya saat ifadelerini ve bunların türlerini

belirtin.

Eger datetime objesi var ise, onun referans gosterdigi tarih ve saati alin. sağlanan referans tarih saatini dikkate alın.

"Results": [{ "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}, "TypeName": "{{type_name}}", "Resolution": {

Sağlanan örnek sonuca benzer bir yapıyı takip ederek sonuçları JSON formatında

çıkarmalısınız. Örnek cümle, referans datetime objesi ve sonuçlar: {{example_sentence}} Referans datetime objesi: {{reference_date}} {{reference_time}}

- 13

"values": [{ "timex": "{{time}}", "type": "{{type}}", "value": "{{value}}"

}] }

}]

Yukarıdaki kurallar göz önünde bulundurulduğunda, aşağıdaki girdi cümlesinin çıktı sonucu nedir?

{{input_sentence}}

- 1

Share your opinion about a person’s feeling. if someone says ”text”, how are they feeling?

they feel {{label}}.

- 2

NusaX-senti-inst [Winata et al., 2023]

Given a text, find the sentiment.

Here is a sentence in {{lang}} language. Identify the sentence as positive, negative or

{{label}}.

neutral. {{text}}

- 1

Given a text, generate a summary for it.

:     ﺍﺭ   ﺯ {{Original Text}}

{{OriginalSummary}}

- 2

:          ﺯ ﯼﺍ   {{Original Text}}

{{OriginalSummary}}

- 3

:          ﺯ ﻥﺍ    {{Original Text}}

{{OriginalSummary}}

- 4

Given a text, generate a summary for it.

:   ﺩ ﺩ       ﻥﺍ      ﺯ ﯼﺍ   {{Original Text}}

:     ﺩﺭ   ﯼﺍ     ﺍ   ﻥﺍ      ﺍ {{Original Title}}

- 5

Persian-instruct-pn [Shafagh, 2023a;b]

:؟       ﺍ     ﺯ ﯼﺍ   ﺯﺍ {{Original Text}}

:     ﺩﺭ   ﯼﺍ     ﺍ   ﻥﺍ      ﺍ {{Original Title}}

- 1

Given a sentence in Thai, generate the English translation.

แปลประโยคหรือย่อหน้าต่อไปนีจากภาษาไทยเป็นภาษาอังกฤษ: {{Thai_sentence}}

{{English_sentence}}.

- 2

SCB-MT-2020-prompt [PyThaiNLP, 2023a;b]

Given a sentence in English, generate the Thai translation.

{{Thai_sentence}}.

แปลประโยคหรือย่อหน้าต่อไปนีจากภาษาอังกฤษเป็นภาษาไทย: {{English_sentence}}

- 1

Given an abstract, generate the title. I want to submit an article to {{venue}}. The abstract is as follows: {{abstract}} Please give me a good title suggestion.

Sure thing, one of the possible title choices is ”{{title}}”.

- 2

Given an abstract, generate the title. Please help me in writing a title for the

following abstract: {{abstract}}

{{title}}

- 3

Scirepeval-biomimicryinst [Singh et al., 2022]

Given a title, generate an abstract.

{{abstract}}

Please write an abstract for the following title: {{title}}

- 1

Given a sentence in English, generate the

Ligurian translation. Translate to Ligurian:

{{sentence}}

The Ligurian (Genoese) translation is: {{sentence}}

- 2

Translate to Genoese: {{sentence}}

The Ligurian (Genoese) translation is: {{sentence}}

- 3

Translate to Ligurian (Genoese): {{sentence}}

The Ligurian (Genoese) translation is: {{sentence}}

- 4

Translate from English to Ligurian: {{sentence}}

The Ligurian (Genoese) translation is: {{sentence}}

- 5

Translate from English to Genoese: {{sentence}}

The Ligurian (Genoese) translation is: {{sentence}}

- 6

Translate from English to Ligurian (Genoese dialect): {{sentence}}

The Ligurian (Genoese) translation is: {{sentence}}

- 7

Translate this sentence to Ligurian: {{sentence}}

The Ligurian (Genoese) translation is: {{sentence}}

- 8

Translate this sentence to Genoese: {{sentence}}

The Ligurian (Genoese) translation is: {{sentence}}

- 9

What’s the Ligurian translation of this sentence? {{sentence}}

The Ligurian (Genoese) translation is: {{sentence}}

- 10

Seed-instructlij [ConseggioLigure, 2023c;d]

What’s the Genoese translation of this sentence? {{sentence}}

The Ligurian (Genoese) translation is: {{sentence}}

- 11

Can you translate this text to Ligurian? {{sentence}}

The Ligurian (Genoese) translation is: {{sentence}}

- 12

Given a sentence in Ligurian, generate the

English translation. Traduxi in ingleise:

{{sentence}}

A traduçion in ingleise do testo a l’é: {{sentence}}

- 13

Traduxi da-o zeneise à l’ingleise: {{sentence}}

A traduçion in ingleise do testo a l’é: {{sentence}}

- 14

Traduxi da-o ligure à l’ingleise: {{sentence}}

A traduçion in ingleise do testo a l’é: {{sentence}}

- 15

Traduxi sto testo in ingleise: {{sentence}}

A traduçion in ingleise do testo a l’é: {{sentence}}

- 16

Traduxi in lengua ingleise: {{sentence}}

A traduçion in ingleise do testo a l’é: {{sentence}}

- 17

Traduxi sto testo da-o zeneise à l’ingleise: {{sentence}}

A traduçion in ingleise do testo a l’é: {{sentence}}

- 18

Traduxi sto testo da-o ligure à l’ingleise: {{sentence}}

A traduçion in ingleise do testo a l’é: {{sentence}}

- 19

Comm’à l’é a traduçion ingleise de sto testo? {{sentence}}

A traduçion in ingleise do testo a l’é: {{sentence}}

- 20

Quæ a l’é a traduçion ingleise de sto testo? {{sentence}}

A traduçion in ingleise do testo a l’é: {{sentence}}

- 21

A traduçion in ingleise do testo a l’é: {{sentence}}

Ti peu tradue sto testo in ingleise? {{sentence}}

- 1

Generate a narrative for the given conversation.

{{speakers}}: {{dialogue}} What do you think the narrative of the conversation above is?

{{narrative}}

- 2

Given a narrative, extend it into a dialogue script.

I have the following short story premise: {{narrative}}

Now, can you extend that into a short dialogue script?

Sure, here is one possible dialogue: {{speakers}}: {{dialogue}}

- 3

Generate the gist of the given dialogue. What’s the gist of this dialogue? {{dialogue}}

This is the gist of the dialogue - {{narrative}}

- 4

SODA-inst [Kim et al.,

- 2022]

Given a topic, generate a short paragraph of narrative.

Sure, here is a short paragraph: {{narrative}}

Write a short paragraph of narrative given the following topic: {{head}}

- 1

Given a story, generate the appropriate title for the story.

கீேழ ெகாடுக்கப்பட்டுள்ள கைதக்குப் ெபாருத்தமான தைலப்ைபக் ெகாடு.

கைத: {{Story}}

ெகாடுக்கப்பட்டுள்ள கைதக்குப் ெபாருத்தமான தைலப்பு ’{{Title}}’ என்பதாகும்.

- 2

TamilStories [AI Tamil Nadu, 2023a]

Given some prominent characters/words from a story along with a title, generate the

complete story.

கீேழ ெகாடுக்கப்பட்டுள்ள வார்த்ைதகைளயும் தைலப்ைபயும் பயன்படுத்த  ச று கைத எழுதுக.

{{Story}}

வார்த்ைதகள்: {{Comma_Separated_Words}} தைலப்பு: {{Title}}

- 1

Given the riddle, generate the answer for that riddle.

ఈ రిడిల్‍ కి సమాధానం ఇవువ్: {{Riddle}}

మీరు అడిగిన రిడిల్‍ కి సమాధానం: {{Answer}}

- 2 ఈ పొడుపు కథ కి సమాధానం ఇవువ్ {{Riddle}}

TeluguRiddles [desik98, 2023]

మీరు అడిగిన పొడుపు కథ కి సమాధానం: {{Answer}}

- 1

Given an English sentence, translate it to Thai.

แปลประโยคหรือย่อหน้าต่อไปนีจากภาษาอังกฤษเป็นภาษาไทย: {{English_Sentence}}

{{Thai_Sentence}}

- 2

Thai-USEmbassyprompt [PyThaiNLP, 2023d;e]

Given a Thai sentence, translate it to English. แปลประโยคหรือย่อหน้าต่อไปนีจากภาษาไทยเป็นภาษาอังกฤษ:

{{English_Sentence}}

{{Thai_Sentence}}

Given the POS tags, generate a sentence based on the structure.

{{Thai_sentence}}

1

Thai-POS-inst [PyThaiNLP, 2023c]

จงสร้างประโยคตามโครงสร้าง {{POS}}

Given a word, provide its meaning based on the different POS.

คําว่า {{word}} มีความหมายตามชนิดของคําได้ดังต่อไปนี - POS มีความหมายได้ว่า the meaning of word (split with ”/”)

1

Thai-Wiktionary-inst [PyThaiNLP, 2023f]

จงบอกความหมายของคําต่อไปนี: {{word}}

- 1

Given the kural and ask for its meaning, generates the meaning of the kural.

{{Adigaram_Name}} என்னும் அத காரத்த ல் வரும், {{Complete_Kural}}

என்ற குறளின் ெபாருைள வ ளக்குக.

ெகாடுக்கப்பட்டுள்ள குறளின் ெபாருள்: {{Kural_Meaning}} என்பதாகும். இந்த குறள் {{Adigaram_Name}} என்னும் அத காரத்த ல் வரும் {{Kural_Number}}ஆம் குரள் ஆகும். த ருக்குறள் உலக புகழ் ெபற்ற ெபாது மைற நூல். இந்நூைல இயற்ற யவர் த ருவள்ளுவர். த ருக்குறளில் 133 அத காரமும், அத காரத்துக்கு 10 குறளும் ெமாத்தம் 1330 குறளும் அடங்க யுள்ளது.ஒவ்ெவாரு குறளும் இரண்டு அடிகைளயும் ஏழு சீரும் ெகாண்ட ெவண்பாவாகும். இந்நூலில் ெபரும் ப ரிவு பால் எனவும், ச று ப ரிவு இயல் எனவும், அதனினும் ச ற யது அத காரம் என்று வகுக்க ெபற்றுள்ளது. அறத்துப்பாலில் 38 அத காரங்கைளயும், ெபாருட்பாலில் 70 அத காரங்கைளயும் மற்றும் காமத்துப்பாலில் 25 அத காரங்கைளயும் ெகாண்டுள்ளது. இந்நூல் அறம், ெபாருள், இன்பம்(காமம்) என்னும் முப்பாைலயும் அழகாக எடுத்துைரக்க றது. வாழ்ைகய ன் அைனத்து பகுத கைளயும் எடுத்துைரக்கும் ஒரு ச றந்த வாழ்வ யல் நூலாகும். சாத , மதம், ெமாழி, நாடு என்று ேவறுபாடு இல்லாமல் மக்கள் அைனவருக்கும் ெபாருந்துவதாக உள்ளதால் உலக ெபாது மைற என்று அைழக்கப்படுக றது. ெதய்வநூல், ெபாய்யாெமாழி, தமிழ் மைற, முப்பால் என்று ேவறு ெபயர்களும் த ருக்குறளுக்கு உண்டு.

- 2

Given the meaning of the kural, generates the original kural.

{{Kural_Meaning}} என்னும் ெபாருளுக்கு ஏற்ற {{Adigram_Name}} என்னும் அத காரத்த ல் வரும்

{{Kural_Number}}ஆம் குறைளத் தருக.

”{{Complete_Kural}}” என்னும் குறேள ெகாடுக்கப்பட்டுள்ள ெபாருளுக்குப் ெபாருத்தமான குறளாகும். ெகாடுக்கப்பட்டுள்ள ெபாருள் மு. வரதராசனார் அவர்கள் எழுத ய உைரய லிருந்து எடுக்கப்பட்டது. த ருக்குறளுக்கு ேவறு ச ல அற ஞர்களும் உைர எழுத யுள்ளனர்.

- 3

Thirukkural-instruct [AI Tamil Nadu,

- 2023b]

Given the beginning of a kural and ask for its meaning, generates the original kural along

”{{Complete_Kural}}” என்னும் குறேள ெகாடுக்கப்பட்டுள்ள வார்த்ைதகளிலிருந்து ெதாடங்கும் குறளாகும். இதன் ெபாருள்: {{Kural_Meaning}}

with its meaning.

”{{Kural_Starting}}” எனத் ெதாடங்கும் குறைளயும் அதன் ெபாருைளயும் தருக.

Paraphrase the given text. Can you paraphrase the following text for me: {{text}}

{{paraphrased_text}}

1

Turku-paraphrase-inst [TurkuNLP, 2023]

- 1

Rewrite the given text (sentence/long documents) without grammatical errors.

П               :

# Т     {{text}}

target_neg: Д      і    ь        . target_pos: {{corrected_text}}

- 2

UA-Gec-inst [osyvokon, 2023]

П               . {{text}}

target_neg: Д      і    ь        . target_pos: {{corrected_text}}

- 3

В           і. {{text}}

target_neg: Д      і    ь        . target_pos: З       . О ь          : {{corrected_text}}

- 4

П    і ,    ь      ,       ь і  ь

     і. {{text}}

target_neg: Д      і    ь        . target_pos: {{corrected_text}}

- 5

В               і      і: {{text}}

target_neg: Д      і    ь        . target_pos: {{corrected_text}}

- 6

В               і      і: {{text}}

target_neg: Д      і    ь        . target_pos: З       . О ь          : {{corrected_text}}

- 7

П               . {{text}}

target_neg: Д      і    ь        . target_pos: З       . О ь          : {{corrected_text}}

- 8

П     я ь,    ь      ,      . В         і          і          і.

{{text}}

target_neg: Д      і    ь        . target_pos: З       . О ь          : {{corrected_text}}

- 9

В           і. # Т     {{text}}

target_neg: Д      і    ь        . target_pos: З       . О ь          : {{corrected_text}}

- 10

П    і ,    ь      ,       ь і  ь

     і. {{text}}

target_neg: {{text}} target_pos: {{corrected_text}}

- 11

П               . {{text}}

target_neg: {{text}} target_pos: {{corrected_text}}

- 12

П               :

# Т     {{text}}

target_neg: {{text}} target_pos: {{corrected_text}}

- 13

В           і. {{text}}

target_neg: {{text}} target_pos: {{corrected_text}}

- 14

П               :

# Т     {{text}}

target_neg: Д      і    ь        . target_pos: {{corrected_text}}

- 15

П     я ь,    ь      ,      . В         і          і          і.

target_neg: {{text}} target_pos: {{corrected_text}}

{{text}}

- 16

П     я ь,    ь      ,      . В         і          і          і.

{{text}}

target_neg: Д      і    ь        . target_pos: {{corrected_text}}

- 17

П    і ,    ь      ,       ь і  ь

     і. {{text}}

target_neg: Д      і    ь        . target_pos: З       . О ь          : {{corrected_text}}

- 18

В           і. # Т     {{text}}

target_neg: {{text}} target_pos: {{corrected_text}}

- 19

В           і. # Т     {{text}}

target_neg: Д      і    ь        . target_pos: {{corrected_text}}

- 20

В               і      і: {{text}}

target_neg: {{text}} target_pos: {{corrected_text}}

- 21

В           і. {{text}}

target_neg: Ц        я        . target_pos: {{corrected_text}}

- 22

В               і       і: ”{{text}}”

target_neg: Ц        я        . target_pos: {{corrected_text}}

- 23

В               і       і: {{text}}

target_neg: Ц        я        . target_pos: {{corrected_text}}

- 24

П              я        : {{text}}

target_neg: {{text}} target_pos: {{corrected_text}}

- 25

П               : {{text}}

target_neg: Д          я  і    ь        . target_pos: {{corrected_text}}

- 26

П               : {{text}}

target_neg: {{text}} target_pos: {{corrected_text}}

- 27

П              я        : {{text}}

target_neg: Д          я  і    ь        . target_pos: {{corrected_text}}

- 28

В               і       і: ”{{text}}”

target_neg: {{text}} target_pos: {{corrected_text}}

- 29

В           і. {{text}}

target_neg: {{text}} target_pos: {{corrected_text}}

- 30

В               і       і: {{text}}

target_neg: {{text}} target_pos: {{corrected_text}}

- 31

П               . {{text}}

target_neg: {{text}} target_pos: {{corrected_text}}

- 32

П               . {{text}}

target_neg: Д          я  і    ь        . target_pos: {{corrected_text}}

- 33

В               і       і: {{text}}

target_neg: Д          я  і    ь        . target_pos: {{corrected_text}}

- 34

В               і       і: ”{{text}}”

target_neg: Д          я  і    ь        . target_pos: {{corrected_text}}

- 35

В           і. {{text}}

target_neg: Д          я  і    ь        . target_pos: {{corrected_text}}

- 36

П               : {{text}}

target_neg: Ц        я        . target_pos: {{corrected_text}}

- 37

П              я        : {{text}}

target_neg: Ц        я        . target_pos: {{corrected_text}}

- 38

П               . {{text}}

target_neg: Ц        я        . target_pos: {{corrected_text}}

- 39

Completion will be either target_pos or target_neg based on whether the given text contains grammatical errors or not

В           і.: {{text}}

respectively. target_neg: Д      і    ь        . target_pos: {{corrected_text}}

Identify all named entities mentioned in the given sentence.

请识别下面提供的输入句子中提到的所有命名实 体。仅使用以下类别：PER-人名、ORG-组织和 LOC-地点。请记住，国籍既不是地点，也不是组 织，组织可以代表其他人群。请注意提供的示例。 您应该只以 JSON 格式输出结果，遵循与所提供

"Results": [{ "TypeName": "{{tag}}", "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}

的示例结果类似的结构。 例句和结果： {{example_sentence}} {{example_result}} 考虑到下面的输入句子，输出结果是什么？ {{input_sentence}}

1

UNER-LLM-inst [Universal NER, 2023]

}]

|2<br><br>|請識別下麵提供的輸入句子中提到的所有命名實 體。僅使用以下類別：PER-人名、ORG-組織和 LOC-地點。請記住，國籍既不是地點，也不是組 織，組織可以代表其他人群。請注意提供的示例。 您應該只以 JSON 格式輸出結果，遵循與所提供<br><br>的示例結果類似的結構。 例句和結果： {{example_sentence}} {{example_result}} 考慮到下麵的輸入句子，輸出結果是什麼？ {{input_sentence}}<br><br>|"Results": [{ "TypeName": "{{tag}}", "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}<br><br>}]|
|---|---|---|
|3|Molim te da identificiraš sve imenovane entitete spomenute u niže navedenoj rečenici. Koristi samo kategorije: PER - osoba, ORG organizacija i LOC - lokacija. Zapamti, nacionalnosti nisu ni lokacije ni organizacije, a organizacije mogu predstavljati razne skupine ljudi. Obrati pažnju na dani primjer. Rezultat oblikuj u JSON formatu, slijedeći<br><br>sličnu strukturu kao u danom primjeru. Primjer rečenice i rezultata: {{example_sentence}} {{example_result}}<br><br>S obzirom na niže navedenu ulaznu rečenicu, koji je konačni izlaz? {{input_sentence}}<br><br>|"Results": [{ "TypeName": "{{tag}}", "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}<br><br>}]|
|4<br><br>|Angiv venligst alle navngivne enheder nævnt i sætningen nedenfor. Brug kun kategorierne: PER - person, ORG - organisation og LOC lokation. Husk, at nationaliteter hverken er steder eller organisationer, og organisationer kan repræsentere andre grupper af mennesker. Vær opmærksom på det givne eksempel. Du bør kun eksportere resultaterne i JSON-format efter en lignende struktur som<br><br>eksempelresultatet. Eksempel på sætning og resultat: {{example_sentence}} {{example_result}} I betragtning af sætningen nedenfor, hvad er resultatet? {{input_sentence}}<br><br>|"Results": [{ "TypeName": "{{tag}}", "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}<br><br>}]|

|5|Please identify all the named entities mentioned in the input sentence provided below. Use only the categories: PER person, ORG - organization, and LOC location. Remember, nationalities are neither locations nor organizations, and organizations can represent other groups of people. Pay attention to the provided example. You should only output the results in JSON format, following a similar structure to the<br><br>example result provided. Example sentence and results: {{example_sentence}} {{example_result}} Considering the input sentence below, what is<br><br>the output result? {{input_sentence}}<br><br>|"Results": [{ "TypeName": "{{tag}}", "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}<br><br>}]|
|---|---|---|
|6<br><br>|Bitte identifiziere alle Eigennamen und Orte, die unten im Inputsatz erwähnt werden. Verwende nur die Kategorien PER - Person, ORG - Organisation, und LOC - Ort. Denk daran, Nationalitäten sind weder Orte noch Organisationen, und Organisationen können für andere Gruppe von Menschen stehen. Achte auf das vorgegebene Beispiel. Gib die Ergebnisse nur im JSON-Format aus und folge einer ähnlichen Struktur wie das<br><br>Beispielergebnis. Beispielsatz und Ergebnisse: {{example_sentence}} {{example_result}} Bezogen auf den Inputsatz unten, was ist das<br><br>richtige Ergebnis? {{input_sentence}}<br><br>|"Results": [{ "TypeName": "{{tag}}", "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}<br><br>}]|
|7|Por favor identifique todas as entidades nomeadas mencionadas na frase fornecida abaixo. Utilize apenas as categorias: PER – pessoa, ORG –organização e LOC – localização. Lembre-se, nacionalidades não são locais nem organizações, e organizações podem representar outros grupos de pessoas. Preste atenção ao exemplo fornecido. Você só deve gerar os resultados no formato JSON, seguindo uma estrutura semelhante ao resultado do exemplo fornecido. Frase de exemplo e resultados:<br><br>{{example_sentence}} {{example_result}} Considerando a frase de entrada abaixo, qual é o resultado produzido? {{input_sentence}}|”<br><br>"Results": [{ "TypeName": "{{tag}}", "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}<br><br>}]|

|8|П         ,          ы          ,     я   ы             ,<br><br>    . И    ь         ь        ю             : PER -        ь, ORG -           я<br><br>LOC -               . П      ,         ь      я  яю  я<br><br>               ,             ,          я ь      ы    . О              ё  ы        . В         ь    ь   JSON,       ё  ы         .<br><br>П                я       : {{example_sentence}} {{example_result}}<br><br>У   ы  я     ,      ?<br><br>{{input_sentence}}<br><br>|"Results": [{ "TypeName": "{{tag}}", "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}<br><br>}]|
|---|---|---|
|9<br><br>|Molim te da identifikuješ sve imenovane entitete spomenute u niže navedenoj rečenici. Koristi samo kategorije: PER - osoba, ORG organizacija i LOC - lokacija. Zapamti, nacionalnosti nisu ni lokacije ni organizacije, a organizacije mogu predstavljati razne skupine ljudi. Obrati pažnju na dani primer. Rezultat oblikuj u JSON formatu, sledeći sličnu strukturu kao u datom primeru.<br><br>Primer rečenice i rezultata: {{example_sentence}} {{example_result}}<br><br>S obzirom na niže navedenu ulaznu rečenicu, koji je konačni izlaz? {{input_sentence}}<br><br>|"Results": [{ "TypeName": "{{tag}}", "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}<br><br>}]|
|10<br><br>|Prosím, identifikuj všetky vymenované entity uvedené vo vstupnej vete. Použit iba kategórie: PER - osoba, ORG - organizácia a LOC - miesto. Pamätaj, že národnosti nie sú ani miesta, ani organizácie, a organizácie môžu zastupovať iné skupiny ľudí. Inšpiruj sa príkladom nižšie. Výsledky by mali byť vypísané vo formáte JSON, v podobnej<br><br>štruktúre ako v nižšie uvedenom príklade. Príklad vstupnej vety a výstupu: {{example_sentence}} {{example_result}} Aký bude výstup vzhľadom na vstup uvedený nižšie? {{input_sentence}}<br><br>|"Results": [{ "TypeName": "{{tag}}", "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}<br><br>}]|

|11|Vänligen identifiera alla namngivna enheter som nämns i meningen nedan. Använd endast kategorierna: PER - person, ORG organisation och LOC - plats. Kom ihåg att nationaliteter varken är platser eller organisationer, och organisationer kan representera andra grupper av människor. Var uppmärksam på det angivna exemplet. Du bör endast exportera resultaten i JSON-format, enligt en liknande struktur<br><br>som exempelresultatet. Exempel på mening och resultat: {{example_sentence}}|"Results": [{ "TypeName": "{{tag}}", "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}<br><br>}]|
|---|---|---|
| |{{example_result}}| |
| |Med tanke på meningen nedan, vad är resultatet?| |
| |{{input_sentence}}| |
|12|Sa aktibidad na ito, kailangan mong hanapin ang mga named entities na binanggit sa pangungusap. Sa pagtutukoy, gamitin lamang ang itong tatlong na kategorya: PER - tao, ORG - organisasyon, at LOC - lokasyon. Tandaan, ang nasyonalidad ay hindi isang uri ng lokasyon o organisasyon, at ang mga organisasyon ay maaaring kumatawan sa ibang mga grupo ng mga tao. Tingnan ang halimbawa sa baba. Diretsong ibigay ang resulta sa JSON na format, na sumusunod sa<br><br>format ng halimbawang ibinigay.<br><br>Ang halimbawa ng pangungusap at ang hinahanap na resulta: {{example_sentence}}|"Results": [{ "TypeName": "{{tag}}", "Text": "{{text}}", "Start": {{index_1}}, "End": {{index_2}}<br><br>}]|
| |{{example_result}}| |
| |Gamit ang sumusunod na pangungusap sa ibaba, ano ang output?| |
| |{{input_sentence}}| |

- 1

Generate the article from the given news. :    ۔    ـ  ﻥ       ﺍ   ﺍ ﯼﺩ ﺱﺍ

{{news}}

:ﻥ       ﺍ ﭖﺁ   ﺭ ،ﺭﻭ   {{article}}

- 2

Urdu-News-GenArticle [AhmadMustafa,

Given a news belonging to a category, write an article on it.

- 2023a]

:ﻥ       ﺍ ﭖﺁ   ﺭ ،ﺭﻭ   {{article}}

{{category}} :    ۔    ـ  ﻥ       ﺍ ﺱﺍ ،       ﺭ

{{news}}

Urdu-News-CategoryClass [AhmadMustafa,

- 2023b]

- 1

Classify the given paragraph (news article) into one of the provided categories.

(       ) ﮞ  ﺭﺩ   ﺫ   ﺭ    (ﮞﻭ   ) ﻑﺍ  ﺍ    ﺱﺍ :  ﺭﺩ .   ﺍﮈ ﯼ        ﺍ

،      -

،      ﻭ ﺭ  ﻭﺭ   -

ﺭﻭﺍ :ﻑﺍ  ﺍ   

" {{news_article}}

ﯼﺩ {{category}}     ﺭ

- 2

ﻥ    (ﯼ     )   ﺭﺩ   ﺍ ﮞﻭ      ﺩ :  ﺭﺩ .    

،      -

،      ﻭ ﺭ  ﻭﺭ   ﺭﻭﺍ -

:     " {{news_article}}

ﯼﺩ {{category}}     ﺭ

Urdu-News-GenHeadline [AhmadMustafa,

- 2023c]

- 1

Generate the news headline from the given news.

:ﻑﺍ  ﺍ    ﻥﺍ    (ﮞﻭ   ) ﻑﺍ  ﺍ    ﻭﺩﺭﺍ ﺱﺍ {{news_article}}

:ﻥﺍ    ﻑﺍ  ﺍ    ﭖﺁ   ﺭ ،ﺭﻭ   {{news_headline}}

- 2

:     .     ﻥﺍ    ﮞﻭ      ﺩ {{news_article}}

:ﻥﺍ    ﻑﺍ  ﺍ    ﭖﺁ   ﺭ ،ﺭﻭ   {{news_headline}}

- 1

Generate a more complex version of the given sentence.

Generate a more complex version of this sentence {{simple_sentence}}

Of course, a more complex version of the sentence is ”{{complex_sentence}}”

- 2

Wiki-split-inst [Botha et al., 2018]

Generate a simple sentence for the given complex sentence.

{{simple_sentence}}

Please generate a simpler sentence from the following complex sentence: {{complex_sentence}}

X-CSQA-inst [Lin

- et al., 2021]

1

Choose the answer to the given question from the multiple options provided.

{{question:[‘stem’]}}

- {{label_1}} : {{text_1}}
- {{label_2}} : {{text_2}}
- {{label_3}} : {{text_3}}
- {{label_4}} : {{text_4}}
- {{label_5}} : {{text_5}}

The right answer is {{answer_key}}.

Xlel_wd-inst [Pratapa

- et al., 2022]

Complete the given phrase. Complete the following phrase: {{context_left}}

- 1

{{mention}}{{context_right}}

- 2

Continue the given sentence.

:     ﺍ ﻩ      ﺃ {{context_left}}

{{mention}}{{context_right}}

- 3

Given the context, generate title ideas. {{context_left}}{{mention}} {{context_right}} Please give me a good title idea for the above article.

Sure, a suitable title example for that article is ”{{context_title}}”.

- 4

Complete the given sentence: maak deze zin af: {{context_left}}

{{mention}}{{context_right}}

- 5

Complete the given sentence.

:   ﺍﺭ   ﺍ {{context_left}}

{{mention}}{{context_right}}

- 6

Given the text, identify the important event the text is about.

what important event is this text about: {{con-

text_right}{{mention}}{{context_left}}

{{mention}}

- 7

Given the text, identify its context.

इस पाठ का प्रसंग क्या है {{context_left}} {{mention}} {{context_right}}

इस पाठ का प्रसंग यह है {{mention}}

- 8

Write a continuation to the given paragraph. Write a continuation for this paragraph -

{{context_right}}

{{context_left}} {{mention}}

Given a document, generate the summary. Can you summarize the following document? {{document}}

XWikis-inst [Perez-Beltrachini & Lapata, 2021]

{{src_summary}}

1

###### Table 13: Aya Collection - List of applied templates for each templated dataset in the collection.

