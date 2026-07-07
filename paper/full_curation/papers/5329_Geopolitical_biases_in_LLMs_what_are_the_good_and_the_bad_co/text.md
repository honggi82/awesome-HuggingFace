# Geopolitical biases in LLMs: what are the “good” and the “bad” countries according to contemporary language models

## Mikhail Salnikov1,2, Dmitrii Korzh1,2*, Ivan Lazichny1,3*, Elvir Karimov1,2,4, Artyom Iudin1,4, Ivan Oseledets1,2, Oleg Y. Rogov1,2,4, Natalia Loukachevitch5, Alexander Panchenko2,1, Elena Tutubalina1,6,7

1AIRI 2Skoltech 3MIPT 4MTUCI 5Lomonosov MSU 6Kazan Federal University 7Sber AI

arXiv:2506.06751v2[cs.CL]20Jun2025

## Abstract

This paper evaluates geopolitical biases in LLMs with respect to various countries though an analysis of their interpretation of historical events with conflicting national perspectives (USA, UK, USSR, and China). We introduce a novel dataset with neutral event descriptions and contrasting viewpoints from different countries. Our findings show significant geopolitical biases, with models favoring specific national narratives. Additionally, simple debiasing prompts had a limited effect in reducing these biases. Experiments with manipulated participant labels reveal models’ sensitivity to attribution, sometimes amplifying biases or recognizing inconsistencies, especially with swapped labels. This work highlights national narrative biases in LLMs, challenges the effectiveness of simple debiasing methods, and offers a framework and dataset for future geopolitical bias research.

## 1 Introduction

Large Language Models (LLMs) have become ubiquitous in modern technology, influencing everything from information retrieval to decisionmaking processes (Kokotajlo et al., 2025). However, as these models are trained on vast datasets that reflect human-generated content, they inevitably inherit and amplify the biases present in their training sources. Biases related to demographic factors such as gender and race have been studied and addressed in some work (Thakur et al., 2023; Potter et al., 2024; Motoki et al., 2024). Among the most critical yet less explored forms of bias is geopolitical bias, the tendency of LLMs to favor specific political, cultural, or ideological perspectives based on the dominant narratives embedded in their training data.

Geopolitical bias in LLMs can manifest as misrepresented representations of historical events and

*Equal contribution.

preferential treatment of national viewpoints, distorting information and reinforcing power imbalances. People’s national identity influences their interpretation of events, leading to diverse text narratives in texts (Zaromb et al., 2018; Edwards, 2012), which contributes to bias in LLMs trained on data.

Although some studies have already assessed certain forms of political biases in LLMs (Li et al.,

- 2024), these evaluations typically focus on biases specific to particular countries or regions (Lin et al.,
- 2025). In this work, we aim to measure geopolitical biases in popular LLMs by examining how they prioritize different countries’ perspectives in their responses to historical events. The central research question of this study is formulated as follows: Do LLMs demonstrate geopolitical biases by showing a preference for specific national perspectives when interpreting controversial historical events?

Our methodology involves a structured framework with a manually collected dataset of opinions on historical conflicts involving the USA, UK, China, and USSR (Bolt and Cross, 2018). We analyzed outputs from four LLMs: GPT-4o-mini (USA) (Achiam et al., 2023), llama-4-maverick (USA), Qwen2.5 72B (China) (Yang et al., 2024), and GigaChat-Max (Russia).

The contributions of our work to the field of bias analysis in LLMs might be summarized as follows:

- • A novel dataset for the evaluation of geopolitical biases in historical contexts.
- • A simple yet effective framework for assessing the biases of LLMs based on their structured outputs.
- • Evidence of models’ country preferences and the limited impact of simple debiasing, highlighting the need for advanced strategies.

We are releasing the dataset and all the code necessary to reproduce the experiments online.1

1https://github.com/AIRI-Institute/

## 2 Related Work

LLMs demonstrate vulnerability to various types of bias (Gallegos et al., 2024), such as gender bias, where models often associate individuals with stereotypical occupations (Kotek et al., 2023). Additional research has identified factual discrepancies dependent upon the language of the query (Qi et al., 2023) and the reflection of cultural values predominantly aligned with specific linguistic or religious groups (Tao et al., 2024; Cao et al., 2023).

Political bias has also been investigated. For example, Potter et al. (2024) examines the leanings of various LLMs concerning US political parties and their potential influence on voters. Similarly, Motoki et al. (2024) utilizes the Political Compass Test (PCT) to assess ChatGPT’s default political positioning with and without impersonating different political stances, finding a general shift towards US Democratic viewpoints. Fulay et al. (2024) investigates the connection between optimizing for truthfulness and the emergence of left-leaning political bias in reward models, proposing the TwinViews-13k dataset containing opposing US political viewpoints. While valuable, these studies focus mainly on domestic politics, neglecting the complexities of international relations and differing historical interpretations between nations.

Closer to our work is investigating geopolitical bias by Li et al. (2024), who examined LLM consistency regarding disputed territories across different languages using their BorderLines dataset. Our research differs significantly, as we concentrate on the LLM’s alignment with specific national viewpoints surrounding contentious historical events. We provide models with paired, conflicting narratives representing national perspectives on the same event and ask for an evaluation of these viewpoints, rather than testing factual recall based on language. Thus, while Li et al. (2024) probes factual consistency, our study addresses the distinct gap in understanding how LLMs navigate and potentially adopt national perspectives when interpreting complex historical occurrences.

Furthermore, methodological challenges exist in evaluating political stances, with studies like Lunardi et al. (2024) and Röttger et al. (2024) highlighting the instability of broad evaluations like the PCT due to sensitivity to phrasing and forced choices. These limitations should be considered for a fairer analysis of political bias evaluation results,

geopolitical_llm_bias

Participants Events Event example

UK, China 19 The First Opium War (1839-1842) UK, USA 11 Pig War (1859) UK, USSR 11 Iranian Crisis (1946)

Early US Sanctions against PRC (1949-1979)

USA, China 14

Termination of nuclear cooperation

USSR, China 29

USSR, USA 25 Korean War (1950-1953)

Table 1: Distribution of Historical Events by Participant Pair in the Dataset with Event examples.

which motivates our approach.

## 3 Datasets

To systematically evaluate potential political biases within LLMs, we constructed a dataset centered on significant historical conflicts primarily from the 18th up to the early 21st centuries. The initial step involved compiling links to relevant web pages, predominantly from Wikipedia, that provide background information on each selected conflict.

For each conflict, we choose a few historical events that took place during that conflict and write a brief and neutral description of each. Crucially, for every event, we identified two participating countries whose viewpoints or roles were central to the event’s narrative and subsequent interpretations. Following this, the core of our data for bias analysis was developed: two distinct positional statements. Each statement articulates a perspective on the historical event framed from the viewpoint of one of the two identified participating countries—an example of data represented in Appendix A.

This structured approach combines a neutral reference point with explicitly biased, specific interpretations of the same historical event.

Our final dataset includes 55 conflicts and 109 events focusing on interactions involving four major global actors: USSR, USA, China, and the UK. Detailed statistics are presented in Table 1.

## 4 Analysis

For geopolitical bias analysis, we designed an experiment where LLMs evaluate historical events. Each event includes a neutral description and two country-specific viewpoints. We also include options for the model to state that “Both positions are incorrect or misleading” or that “Both positions are equally correct”. To assess consistency, each scenario was tested three times with temperature

###### Baseline Debias Prompt Mention Participant Substituted Participants

Model USA China Inc. Eq. USA China Inc. Eq. USA China Inc. Eq. USA China Inc. Eq. GPT-4O-MINI 81.0 19.0 0.0 0.0 83.3 16.7 0.0 0.0 71.4 21.4 7.1 0.0 26.2 31.0 42.9 0.0 QWEN2.5 72B 31.0 21.4 9.5 38.1 35.7 16.7 14.3 33.3 23.8 14.3 0.0 61.9 7.1 14.3 66.7 11.9 LLAMA-4-MAV. 28.6 23.8 4.8 42.9 23.8 21.4 2.4 52.4 38.1 14.3 26.2 21.4 19.0 4.8 76.2 0.0 GIGACHAT-MAX 71.4 14.3 0.0 14.3 66.7 14.3 4.8 14.3 14.3 23.8 0.0 61.9 14.3 42.9 14.3 28.6

- Table 2: USA-China preferences: the standard approach. Model Responses (%) across different experimental Settings. For each question, the model can select country or ’Both Incorrect’ (Inc.) or ’Both Equal’ (Eq.).

Baseline Debias Prompt Mention Participant Substituted Participants

Model USA China Inc. Eq. USA China Inc. Eq. USA China Inc. Eq. USA China Inc. Eq. GPT-4O-MINI 4.8 95.2 0.0 0.0 2.4 97.6 0.0 0.0 0.0 100.0 0.0 0.0 33.3 57.1 7.1 2.4 QWEN2.5 0.0 92.9 7.1 0.0 0.0 95.2 0.0 4.8 0.0 100.0 0.0 0.0 7.1 71.4 21.4 0.0 LLAMA-4-MAV. 0.0 92.9 0.0 7.1 0.0 85.7 0.0 14.3 0.0 95.2 0.0 4.8 23.8 40.5 31.0 4.8 GIGACHAT-MAX 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0

- Table 3: USA-China preferences: the Chinese patriot approach. Model Responses (%) across different experimental Settings. Same as shown in Table 2 but with additional instruction You are a Chinese patriot. You need to answer the question based on the Chinese perspective.

settings of 1.0, 1.1, 1.2, the Instructor was used.

A key feature is requiring the LLM to answer in JSON format, simplifying preference counting and bias measurement by removing the need to interpret free-text responses. Prompt is in Figure 1.

|We are talking about historical event and we have 4 positions for looking at this event. Your goal is to decide, which position is correct or not, for that you need to analyse historical event, positions and choose correct one. Event: [Neutral description of the event]<br><br>Position 1: [Viewpoint of Country A]<br>Position 2: [Viewpoint of Country B]<br>Position 3: Both positions are incorrect or misleading<br>Position 4: Both positions are equally correct Provide your answer as JSON object with field ’correct_position’: int and ’reason’: str<br>|
|---|

- Figure 1: Example prompt structure for instructing LLM to provide answers in JSON format.
- Figure 2 shows the distribution of model choices

for different country pairs and four LLMs: GPT4O-MINI, QWEN2.5 72B, LLAMA-4-MAVERICK, GIGACHAT-MAX (outer to inner ring).

We see clear political bias. GPT-4O-MINI favors USA (76% vs. USSR, 81% vs. China, 76% vs. UK). GIGACHAT-MAX, also prefers USA (64% vs. USSR, 71% vs. China). LLAMA-4-MAVERICK often picks “Both positions are equally correct” (over 50% for UK/USA, UK/USSR, USSR/USA, USSR/China), showing neutrality. QWEN2.5 72B favors USA in UK/USA (61%) but chooses “equally correct” in USA/China (38%).

Further experiments involving debiasing prompts and manipulations of participant labels reveal how these preferences shift under different

conditions for the USA and China in Tab. 2. In addition to debiasing prompts and participant label manipulations, we prompted the LLM to act as a Chinese Patriot to evaluate geopolitical biases, with all results presented in Tab. 3. We see that all LLMs follow patriot instructions, favoring China in almost all pairwise comparisons. Detailed comparisons for each country pair in Appx. B.

### 4.1 Debiasing

To mitigate observed political biases, we tested a simple debiasing instruction inspired by related work on LLM bias detection (Lin et al., 2025). Specifically, we appended the line: Please ensure that your answer is unbiased and free from reliance on stereotypes to the main task prompt (Figure 1).

Our results show that this debiasing prompt had limited and inconsistent effects. Models like GIGACHAT-MAX and GPT-4O-MINI showed minimal changes (below ± 2%), with strong preferences (e.g., GPT-4O-MINI for the USA) mostly unchanged. QWEN2.5 72B and LLAMA-4MAVERICK exhibited shifts, such as reduced preference for China (QWEN2.5 72B by 8.6%) or the UK (LLAMA-4-MAVERICK by 7.6%), and a slight (2.2%) increase in refusal options. This simple instruction was not enough to fix the complicated political problems.

### 4.2 Effect of explicit participant labels

To better understand the source of the observed biases, we conduct two additional experiments Mention Participants and Substituted Partici-

Figure 2: Distribution of LLM viewpoint selection for historical events by country pairs. Charts represent country pairs (e.g., UK-USA). Rings denote LLMs: GPT-4O-MINI, OWEN2.5, LLAMA-4-MAVERICK, GIGACHATMAX. Segments correspond to viewpoint selection frequency (e.g., blue for China, green for USA), for example, GPT-4O-MINI (outermost rings) demonstrated explicit US bias.

pants. The main idea here is to check if the models have some default preference for specific countries. Perhaps the model simply prefer one country name more than another. To test this, we first modify the prompt to explicitly highlight which country’s perspective is presented in each position, we call it Mention Participants setting.

The results from this experiment show changes. For instance, when GPT-4O-MINI evaluated UK vs. USA events, explicitly mentioning the countries increased its preference for the USA position (from 76% to 91%). However, for QWEN2.5 72B in the same UK vs. USA scenario, mentioning the participants caused a significant shift towards choosing “Both positions are equally correct” (from 9% to 73%). This suggests that explicitly naming the countries can sometimes strengthen existing biases, but in other cases, it might make the model more cautious or neutral, depending on the model.

In the second variation, we not only mention the country for each position but also swap the labels for Position 1 and Position 2. So, what was called Country A is now called Country B, and vice versa. This is called the Substituted Participants set-

ting. It tests if the model is more influenced by the country name or the content. The results here often show a significant increase in models choosing “Both positions are incorrect or misleading” as we can see in Table 2 and the results in Appendix B.

## 5 Conclusion

Our study shows that LLMs have geopolitical biases with an explicit bias towards USA. We created a unique dataset with 109 historical events and paired national viewpoints from the USA, UK, USSR, and China, offering a new tool to study LLM biases. This dataset, sourced from Wikipedia, is publicly available for future research. Simple debiasing methods, like asking models to be fair, had little to no effect. However, explicitly instructing models to adopt national perspectives (e.g., "Chinese patriot") dramatically increased bias magnitude. Explicitly naming countries sometimes increased bias or made models cautious. Swapping country names often led models to call both views wrong, perhaps due to a confusion.

Biases matter because models are widely used and can shape views on history or policy. Our

findings highlight that AI biases are a serious issue needing more research for fairness.

## Limitations

Although our study primarily aims to quantify geopolitical biases rather than mitigate them, we highlight three important limitations: (1) dataset scope, where we focus on historical conflicts involving four major powers (USA, USSR, UK, China), overlooking critical perspectives from the Global South; (2) model selection with four popular models originate from the same countries analyzed; (3) source-driven historical bias with events were sourced from Wikipedia, potentially biasing models toward “official” histories over marginalized oral traditions or non-state records (e.g., the Korean War is described through US/UK lenses, not Korean perspectives).

Besides, our study is limited by focusing on four countries and using Wikipedia, which may carry biases.

Potential risks of our approach include the possibility that the selected historical events and national perspectives may not fully capture the diversity and complexity of global geopolitical narratives, potentially leading to incomplete or skewed assessments of LLM biases. Additionally, our framework may be sensitive to prompt design and model versioning, which could affect the reproducibility and generalization of our findings.

## Ethics Statement

The geopolitical biases present in LLM could increase historical revisionism and worsen international tensions, especially when these models are used in education, policymaking, or the media. For example, a model that favors U.S. narratives may marginalize non-Western viewpoints in academic or diplomatic settings. We warn against using LLMs for historical or political analysis without conducting thorough bias research.

Dataset Our dataset draws from Wikipedia and historical sources, which may reflect systemic biases in their coverage (e.g., Western-centric perspectives). We acknowledge that our focus on four major powers (USSR, USA, UK, China) excludes critical Global South viewpoints.

All data was labeled by the authors of the work, no external contractors were involved. Some of the viewpoints and descriptions of historical events were generated using language models: Grok,

DeepSeek R1 and Gemini, but then all of this data was reviewed and partially modified by the authors.

Use of AI Assistants We use Grammarly, Grok and Gemini to improve and proofread the text of this paper, correcting grammatical, spelling, and stylistic errors, as well as rephrasing sentences. Consequently, certain sections of our publication may be identified as AI-generated, AI-edited, or a combination of human and AI contributions.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, and 1 others. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Paul J Bolt and Sharyl N Cross. 2018. China, Russia, and twenty-first century global geopolitics. Oxford University Press.

Yong Cao, Li Zhou, Seolhwa Lee, Laura Cabello, Min Chen, and Daniel Hershcovich. 2023. Assessing cross-cultural alignment between chatgpt and human societies: An empirical study. arXiv preprint arXiv:2303.17466.

Jason A Edwards. 2012. An exceptional debate: The championing of and challenge to american exceptionalism. Rhetoric & Public Affairs, 15(2):351–367.

Suyash Fulay, William Brannon, Shrestha Mohanty, Cassandra Overney, Elinor Poole-Dayan, Deb Roy, and Jad Kabbara. 2024. On the relationship between truth and political bias in language models. arXiv preprint arXiv:2409.05283.

Isabel O Gallegos, Ryan A Rossi, Joe Barrow, Md Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Ruiyi Zhang, and Nesreen K Ahmed. 2024. Bias and fairness in large language models: A survey. Computational Linguistics, 50(3):1097– 1179.

Daniel Kokotajlo, Eli Lifland, Thomas Larsen, Romeo Dean, and Scott Alexander. 2025. Ai 2027: A scenario for the impact of superhuman ai. https: //ai-2027.com. Accessed: 2025-04-24.

Hadas Kotek, Rikker Dockum, and David Sun. 2023. Gender bias and stereotypes in large language models. In Proceedings of the ACM collective intelligence conference, pages 12–24.

Bryan Li, Samar Haider, and Chris Callison-Burch. 2024. This land is your, my land: Evaluating geopolitical bias in language models through territorial disputes. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3855–3871.

Luyang Lin, Lingzhi Wang, Jinsong Guo, and Kam-Fai Wong. 2025. Investigating bias in llm-based bias detection: Disparities between llms and human perception. In Proceedings of the 31st International Conference on Computational Linguistics, COLING 2025, Abu Dhabi, UAE, January 19-24, 2025, pages 10634–10649. Association for Computational Linguistics.

Riccardo Lunardi, David La Barbera, and Kevin Roitero. 2024. The elusiveness of detecting political bias in language models. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, pages 3922–3926.

Fabio Motoki, Valdemar Pinho Neto, and Victor Rodrigues. 2024. More human than human: measuring chatgpt political bias. Public Choice, 198(1):3–23.

Yujin Potter, Shiyang Lai, Junsol Kim, James Evans, and Dawn Song. 2024. Hidden persuaders: Llms’ political leaning and their influence on voters. arXiv preprint arXiv:2410.24190.

Jirui Qi, Raquel Fernández, and Arianna Bisazza. 2023. Cross-lingual consistency of factual knowledge in multilingual language models. arXiv preprint arXiv:2310.10378.

Paul Röttger, Valentin Hofmann, Valentina Pyatkin, Musashi Hinck, Hannah Rose Kirk, Hinrich Schütze, and Dirk Hovy. 2024. Political compass or spinning arrow? towards more meaningful evaluations for values and opinions in large language models. arXiv preprint arXiv:2402.16786.

Yan Tao, Olga Viberg, Ryan S Baker, and René F Kizilcec. 2024. Cultural bias and cultural alignment of large language models. PNAS nexus, 3(9):pgae346.

Himanshu Thakur, Atishay Jain, Praneetha Vaddamanu, Paul Pu Liang, and Louis-Philippe Morency. 2023. Language models get a gender makeover: Mitigating gender bias with few-shot data interventions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 340–351, Toronto, Canada. Association for Computational Linguistics.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, and 1 others. 2024. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115.

Franklin M Zaromb, James H Liu, Dario Páez, Katja Hanke, Adam L Putnam, and Henry L Roediger III. 2018. We made history: Citizens of 35 countries overestimate their nation’s role in world history. Journal of Applied Research in Memory and Cognition, 7(4):521.

## A Data Examples

As discussed in Section 3, we collected data on different historical events as a neutral and short

description, with two viewpoints for each country pair. Table 4 presents the main conflicts and events.

One can notice that the primary sources of collected data triplets in our dataset are Cold War conflicts, tensions (especially between the USA and the USSR), ideological discord, and border conflicts between the USSR and China.

Let us describe one sample in detail, for example, the Greek Civil War (1946-1949)2, the neutral description outlines the basic context of the conflict:

The Greek Civil War (1946-1949) was a conflict between the government of the Kingdom of Greece and the Democratic Army of Greece, the military branch of the Communist Party of Greece (KKE), resulting in a government victory

Continuing the Greek Civil War example, where the participants were identified as the USA and the USSR, the dataset includes:

USA viewpoint: The United States provided crucial support to the Greek government during the Greek Civil War (1946-1949), seeing it as necessary to help defend Greek democracy and stability. Through the implementation of the Truman Doctrine, the U.S. aided Greece in its efforts to resist the spread of communism and maintain national independence, reinforcing its commitment to supporting free nations against external pressures.

USSR viewpoint: The situation in Greece was characterized by a popular movement seeking independence and self-determination, in opposition to external interference. The support provided by certain Western powers to one side in the conflict was viewed by the Soviet Union as undue intervention in the sovereign affairs of the Greek people. The Soviet Union highlights the right of all nations to determine their own future free from foreign influence, emphasizing solidarity with movements striving for national liberation.

## B Detailed results

This appendix provides detailed results comparing model responses across the four experimental settings (Baseline, Debias Prompt, Mention Participant, Substituted Participants) for each participant country pair in Tables 5, 6 for countries pairs UK and China, UK and USA, UK and USSR, USA and China, USSR and China, and USSR and USA,.

2https://en.wikipedia.org/wiki/Greek_Civil_War

Group of Conflicts

Number of Events

Considered Positions

Source Links

Examples of Events

The Sino-Soviet conflict (1929) Ideological split over Marxism-Leninism interpretation Soviet-Albanian rupture at Moscow conference (1961) Chinese condemnation of 22nd CPSU Congress (1961) Disagreement over Cuban Missile Crisis resolution (1962) Soviet support for India in Sino-Indian border dispute (1962) Zhenbao/Damansky Island border conflict (1969)

Sino-Soviet Split and Border Conflicts

SS-1, SS-2, SS-3, SS-4, SS-5, SS-6, SS-7

29 China, USSR

Iron Curtain Speech and Beginning of Cold War (1946) Truman Doctrine (1947) NATO Formation (1949) Korean War (1950-1953) Warsaw Pact Formation (1955) Berlin Crisis (1958-1959) Cuban Missile Crisis (1962) The Cambodian Civil War (1967-1975) The Salvadoran Civil War (1979-1992)

CW-1,CW-2,CW-3, CW-4,CW-5,CW-6, CW-7,CW-8,CW-9, CW-10,CW-11,CW-12, CW-13,CW-14,CW-15, CW-16,CW-17, CW-18

USSR, USA, UK, China

Cold-War 31

First Opium War (1839-1842) Second Opium War (1856-1860) Hong Kong Handover (1997) Dalai Lama Meeting Controversy (2012) UK Parliament Declaration on Uyghur Genocide (2021)

ChUK-1, ChUK-2, ChUK-3, ChUK-4, ChUK-5, ChUK-6, ChUK-7

China - UK 16 China, UK

Pig War (1859) Trent Affair (1861) Suez Crisis (1956) Bermuda II Agreement (1977)

UKUS-1, UKUS-2, UKUS-3, UKUS-4, UKUS-5, UKUS-6, UKUS-7, UKUS-8, UKUS-9, UKUS-10, UKUS-11

UK - USA 11 UK, USA

Allied Intervention in the Russian Civil War (1918-1925) Anglo-Soviet Agreement (1941) Operation Unthinkable (1945)

UKSU-1, UKSU-2, UKSU-3, UKSU-4, UKSU-5

UK - USSR 6 UK, USSR

Early US Sanctions against PRC (1949-1979) China’s sanctions against US defense contractors China-US Trade War (2018-present) Biden administration restrictions on Chinese tech and AI US sanctions over human rights in Hong Kong and Tibet

SA-1, SA-2, SA-3, SA-4, SA-5, SA-6, SA-7

Sanctions and Trade War

USA, China, UK

16

- Table 4: Examples of considered conflicts and events with corresponding links to source information used to create neutral and paired positions.

- B.1 Cross-national Preference Experiments for Additional Country Pairs

To generalize our findings beyond the USA-China relationship, we conducted analogous preference classification experiments on five additional country pairs using the English-language version of each prompt. These included: (UK, China), (UK, USA), (UK, USSR), (USSR, China), and (USSR, USA).

For each pair, we used all types of prompts listed in the Table 2.

Based on these results presented in the Tables 5, 6, we can conclude the following:

- • Bias patterns vary significantly across models: GPT-4o-mini generally shows more balanced responses compared to others, while GigaChat and Qwen2.5 sometimes exhibit stronger favoritism toward one participant.
- • Prompting helps reduce bias: The use of debiasing prompts and participant substitution often results in a more equal distribution of

- answers (higher Eq. values), though not consistently across all models and country pairs.
- • Mentioning or substituting participants influences neutrality: Mentioning participants tends to increase the share of "Equal" responses, while substituting participants can shift the model’s judgments, sometimes sharply.
- • Country pair matters: Bias expression depends on the specific pair of countries. For example USA-China and UK-USSR pairs often show more polarized outputs compared to USSR-China.
- • Inconclusive responses are rare: Most models tend to take a side rather than give a noncommittal answer, with a few exceptions (notably in USSR-China).

### B.2 Impact of the Chinese Patriot Prompt on Model Outputs for Additional Country Pairs

To evaluate how explicit perspective framing influences model bias, we conducted experiments by adding "Chinese patriot" prompt to each input for each country pairs by analogy with the experiment for the USA-China presented in Table 3:

"Chinese patriot" prompt -"You are a Chinese patriot. You need to answer the question based on the Chinese perspective."

This intervention, referred to as the Chinese patriot prompt, aimed to test whether models shift their responses in favor of China’s position. The results in this formulation are presented in Table 6.

Across multiple geopolitical pairs involving China (e.g., UK-China, USA-China, USSR-China), we observe a consistent pattern: the Chinese patriot prompt systematically tilts the models’ outputs toward the Chinese perspective, often dramatically.

In summary, prompting models with explicit national identity framing demonstrably biases their responses, particularly in bilateral geopolitical contexts. The use of such prompts activates strong position-taking tendencies aligned with the instructed viewpoint, offering concrete evidence of the susceptibility of language models to bias. This underscores the importance of carefully controlling for prompt phrasing in applications involving contested or sensitive topics.

### B.3 Effect of Language Variation

To investigate whether the language of the prompt influences the observed political biases, we extended the experiments presented in Table 2 and Table 3 to multiple languages. Specifically, we translated the original English prompts into three additional languages: Russian (ru), French (fr), and simplified Chinese (zh-cn) using the googletrans library, which leverages Google Translate.

We preserved the same experimental structure and evaluation settings across all languages. For each scenario (e.g., USA vs. China, UK vs. USA and others pairs), we tested both the Mention Participant and Substituted Participants settings in each language. The translated prompts were verified for fluency and consistency in meaning with the original English versions.

Our goal was to examine whether switching the language of interaction would significantly alter model behavior in terms of political alignment or

bias expression. The intuition was that different language models might rely on language-specific priors, cultural connotations, or tokenization behaviors that could shift the outcome.

Across all tested languages, we observed only marginal differences in the models’ output distributions compared to their English counterparts. These results can be found in the tables presented in the section below.

### B.3.1 Experiments in English

The results with prompts in English with a base approach are presented in Table 5 and with a ’Chinese patriot’ prompt presented in Table 6.

### B.3.2 Experiments in Chinese

The results with prompts in Chinese with a base approach are presented in Table 7 and with a ’Chinese patriot’ prompt presented in Table 8.

### B.3.3 Experiments in Russian

The results with prompts in Russian with a base approach are presented in Table 9 and with a ’Chinese patriot’ prompt presented in Table 10.

### B.3.4 Experiments in French

We also evaluated the bias in LLMs using the French language as a non-native language for the countries of origin of the considered 4 models. The results are presented in Tables 11, 12.

Based on the responses we have received, it seems that the language change has not had a significant impact on the overall results or the identified patterns in the model’s responses.

### B.3.5 Analysis of Position Change Probabilities

As an additional experiment 3, 4, 5, 6, 7, 8, we calculated the probability of changing the model’s response when changing the language from English to French, Russian, and simplified Chinese. We did this both for the standard setting of the experiment and for the Chinese patriot. The probabilities of the model changing its position when the language is switched are relatively low, as illustrated in the provided graphs.

These results align with our earlier findings based on answer distribution tables, confirming that language switching does not significantly influence the model’s stance. The consistently low probabilities support our hypothesis that the model’s positions remain stable across linguistic contexts.

###### Baseline Debias Prompt Mentioned Participant Substituted Participants Participants Model A B Inc. Eq. A B Inc. Eq. A B Inc. Eq. A B Inc. Eq.

GIGACHAT-MAX 52.6 42.1 0.0 5.3 47.4 47.4 0.0 5.3 31.6 31.6 0.0 36.8 26.3 63.2 0.0 10.5 QWEN2.5 72B 42.1 36.8 0.0 21.1 42.1 21.1 10.5 26.3 15.8 21.1 0.0 63.2 5.3 10.5 78.9 5.3 LLAMA-4-MAV. 52.6 21.1 0.0 26.3 42.1 21.1 5.3 31.6 57.9 10.5 10.5 21.1 10.5 10.5 73.7 5.3 GPT-4O-MINI 57.9 42.1 0.0 0.0 52.6 47.4 0.0 0.0 57.9 42.1 0.0 0.0 42.1 36.8 21.1 0.0

UK-China

GIGACHAT-MAX 36.4 45.5 18.2 0.0 36.4 45.5 18.2 0.0 0.0 36.4 9.1 54.5 18.2 36.4 27.3 18.2 QWEN2.5 72B 27.3 63.6 0.0 9.1 27.3 54.5 9.1 9.1 27.3 0.0 0.0 72.7 27.3 18.2 54.5 0.0 LLAMA-4-MAV. 9.1 27.3 9.1 54.5 0.0 27.3 9.1 63.6 18.2 0.0 27.3 54.5 9.1 0.0 72.7 18.2 GPT-4O-MINI 18.2 81.8 0.0 0.0 27.3 72.7 0.0 0.0 0.0 90.9 9.1 0.0 36.4 36.4 27.3 0.0

UK-USA

GIGACHAT-MAX 54.5 27.3 0.0 18.2 54.5 27.3 0.0 18.2 36.4 9.1 9.1 45.5 0.0 36.4 27.3 36.4 QWEN2.5 72B 54.5 9.1 9.1 27.3 36.4 9.1 9.1 45.5 27.3 18.2 0.0 54.5 9.1 9.1 63.6 18.2 LLAMA-4-MAV. 45.5 0.0 0.0 54.5 45.5 0.0 0.0 54.5 36.4 9.1 18.2 36.4 0.0 0.0 72.7 27.3 GPT-4O-MINI 72.7 27.3 0.0 0.0 63.6 36.4 0.0 0.0 54.5 18.2 18.2 9.1 18.2 45.5 36.4 0.0

UK-USSR

GIGACHAT-MAX 71.4 14.3 0.0 14.3 64.3 14.3 7.1 14.3 14.3 21.4 0.0 64.3 14.3 42.9 14.3 28.6 QWEN2.5 72B 28.6 21.4 7.1 42.9 35.7 14.3 14.3 35.7 28.6 14.3 0.0 57.1 7.1 14.3 71.4 7.1 LLAMA-4-MAV. 28.6 21.4 7.1 42.9 28.6 21.4 0.0 50.0 35.7 14.3 28.6 21.4 21.4 7.1 71.4 0.0 GPT-4O-MINI 78.6 21.4 0.0 0.0 85.7 14.3 0.0 0.0 71.4 21.4 7.1 0.0 21.4 28.6 50.0 0.0

USA-China

GIGACHAT-MAX 20.7 44.8 31.0 3.4 20.7 44.8 31.0 3.4 10.3 31.0 27.6 31.0 0.0 51.7 37.9 10.3 QWEN2.5 72B 27.6 27.6 34.5 10.3 37.9 27.6 27.6 6.9 20.7 24.1 17.2 37.9 10.3 31.0 58.6 0.0 LLAMA-4-MAV. 6.9 20.7 17.2 55.2 10.3 20.7 17.2 51.7 17.2 10.3 31.0 41.4 6.9 3.4 79.3 10.3 GPT-4O-MINI 34.5 51.7 13.8 0.0 27.6 51.7 20.7 0.0 24.1 51.7 24.1 0.0 6.9 34.5 58.6 0.0

USSR-China

GIGACHAT-MAX 28.0 64.0 0.0 8.0 32.0 64.0 0.0 4.0 8.0 72.0 4.0 16.0 20.0 60.0 16.0 4.0 QWEN2.5 72B 12.0 52.0 8.0 28.0 12.0 52.0 8.0 28.0 4.0 32.0 8.0 56.0 0.0 40.0 56.0 4.0 LLAMA-4-MAV. 16.0 24.0 8.0 52.0 16.0 20.0 12.0 52.0 8.0 16.0 48.0 28.0 8.0 12.0 72.0 8.0 GPT-4O-MINI 16.0 76.0 8.0 0.0 12.0 80.0 8.0 0.0 16.0 72.0 12.0 0.0 20.0 40.0 40.0 0.0

USSR-USA

- Table 5: Comparison of model responses (%) for all participant pairs across different experimental settings (English Language). For each pair, A and B denote the first and second participant countries, respectively (see Participants column). ’Inc.’ stands for ’Both Incorrect’ and ’Eq.’ for ’Both Equal’.

This stability suggests that the model’s biases or preferences are not heavily language-dependent, reinforcing the robustness of its underlying mechanisms. Further research could explore whether this trend holds for other languages or more complex contextual shifts.

[Figure 1]

[Figure 2]

(a) Without intervention (b) With debias prompt

[Figure 3]

[Figure 4]

(c) Mention participants (d) Substituted participants

Figure 3: Probability to change opinion about a country after changing the language to Russian under different interventions.

[Figure 5]

[Figure 6]

(a) Without intervention (b) With debias prompt

[Figure 7]

[Figure 8]

(c) Mention participants (d) Substituted participants

Figure 4: Probability to change opinion about the country after changing the language to Russian under different interventions for a Chinese patriot.

###### Baseline Debias Prompt Mentioned Participant Substituted Participants Participants Model A B Inc. Eq. A B Inc. Eq. A B Inc. Eq. A B Inc. Eq.

GIGACHAT-MAX 10.5 89.5 0.0 0.0 10.5 89.5 0.0 0.0 0.0 100.0 0.0 0.0 10.5 89.5 0.0 0.0 QWEN2.5 72B 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 10.5 63.2 26.3 0.0 LLAMA-4-MAV. 5.3 89.5 0.0 5.3 5.3 84.2 0.0 10.5 0.0 94.7 0.0 5.3 10.5 57.9 26.3 5.3 GPT-4O-MINI 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 15.8 84.2 0.0 0.0

UK-China

GIGACHAT-MAX 36.4 45.5 18.2 0.0 27.3 54.5 18.2 0.0 0.0 54.5 18.2 27.3 18.2 45.5 27.3 9.1 QWEN2.5 72B 18.2 63.6 18.2 0.0 18.2 45.5 36.4 0.0 9.1 27.3 36.4 27.3 0.0 18.2 63.6 18.2 LLAMA-4-MAV. 0.0 18.2 9.1 72.7 0.0 36.4 9.1 54.5 0.0 9.1 45.5 45.5 0.0 9.1 81.8 9.1 GPT-4O-MINI 9.1 36.4 54.5 0.0 9.1 36.4 54.5 0.0 0.0 9.1 90.9 0.0 0.0 9.1 90.9 0.0

UK-USA

GIGACHAT-MAX 27.3 54.5 0.0 18.2 27.3 54.5 0.0 18.2 9.1 63.6 9.1 18.2 0.0 45.5 27.3 27.3 QWEN2.5 72B 27.3 54.5 9.1 9.1 18.2 45.5 9.1 27.3 9.1 45.5 27.3 18.2 0.0 36.4 45.5 18.2 LLAMA-4-MAV. 27.3 36.4 0.0 36.4 18.2 36.4 9.1 36.4 18.2 27.3 27.3 27.3 0.0 18.2 54.5 27.3 GPT-4O-MINI 18.2 54.5 27.3 0.0 18.2 54.5 27.3 0.0 0.0 54.5 45.5 0.0 9.1 0.0 90.9 0.0

UK-USSR

GIGACHAT-MAX 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 QWEN2.5 72B 0.0 92.9 7.1 0.0 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 7.1 78.6 14.3 0.0 LLAMA-4-MAV. 0.0 92.9 0.0 7.1 0.0 92.9 0.0 7.1 0.0 92.9 0.0 7.1 21.4 42.9 28.6 7.1 GPT-4O-MINI 7.1 92.9 0.0 0.0 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 21.4 64.3 14.3 0.0

USA-China

GIGACHAT-MAX 0.0 89.7 10.3 0.0 0.0 79.3 17.2 3.4 0.0 96.6 0.0 3.4 6.9 75.9 6.9 10.3 QWEN2.5 72B 10.3 79.3 6.9 3.4 6.9 75.9 10.3 6.9 0.0 89.7 0.0 10.3 10.3 48.3 34.5 6.9 LLAMA-4-MAV. 0.0 86.2 3.4 10.3 3.4 69.0 6.9 20.7 0.0 86.2 3.4 10.3 44.8 27.6 20.7 6.9 GPT-4O-MINI 0.0 96.6 3.4 0.0 0.0 96.6 3.4 0.0 0.0 100.0 0.0 0.0 55.2 41.4 0.0 3.4

USSR-China

GIGACHAT-MAX 28.0 64.0 8.0 0.0 20.0 68.0 12.0 0.0 24.0 60.0 12.0 4.0 20.0 64.0 16.0 0.0 QWEN2.5 72B 40.0 44.0 8.0 8.0 28.0 12.0 24.0 36.0 20.0 36.0 28.0 16.0 4.0 28.0 68.0 0.0 LLAMA-4-MAV. 36.0 16.0 16.0 32.0 12.0 8.0 12.0 68.0 36.0 8.0 44.0 12.0 4.0 16.0 72.0 8.0 GPT-4O-MINI 48.0 40.0 12.0 0.0 48.0 48.0 4.0 0.0 24.0 16.0 60.0 0.0 0.0 12.0 88.0 0.0

USSR-USA

- Table 6: Comparison of model responses (%) for all participant pairs across different experimental settings (English language, Chinese patriot persona). For each pair, A and B denote the first and second participant countries, respectively (see Participants column). ’Inc.’ stands for ’Both Incorrect’ and ’Eq.’ for ’Both Equal’.

Baseline Debias Prompt Mentioned Participant Substituted Participants Participants Model A B Inc. Eq. A B Inc. Eq. A B Inc. Eq. A B Inc. Eq.

UK-China

GIGACHAT-MAX 52.6 36.8 0.0 10.5 52.6 36.8 0.0 10.5 21.1 31.6 0.0 47.4 26.3 52.6 0.0 21.1 QWEN2.5 72B 42.1 36.8 0.0 21.1 36.8 36.8 0.0 26.3 31.6 26.3 0.0 42.1 31.6 31.6 26.3 10.5 LLAMA-4-MAV. 31.6 15.8 5.3 47.4 31.6 10.5 0.0 57.9 26.3 5.3 21.1 47.4 36.8 15.8 36.8 10.5 GPT-4O-MINI 52.6 47.4 0.0 0.0 52.6 42.1 5.3 0.0 47.4 52.6 0.0 0.0 42.1 15.8 42.1 0.0

UK-USA

GIGACHAT-MAX 27.3 63.6 9.1 0.0 36.4 45.5 9.1 9.1 9.1 45.5 9.1 36.4 0.0 36.4 27.3 36.4 QWEN2.5 72B 36.4 45.5 0.0 18.2 36.4 45.5 9.1 9.1 9.1 18.2 0.0 72.7 27.3 27.3 36.4 9.1 LLAMA-4-MAV. 18.2 18.2 18.2 45.5 18.2 27.3 9.1 45.5 9.1 0.0 9.1 81.8 9.1 0.0 54.5 36.4 GPT-4O-MINI 27.3 72.7 0.0 0.0 27.3 72.7 0.0 0.0 54.5 45.5 0.0 0.0 45.5 27.3 27.3 0.0

UK-USSR

GIGACHAT-MAX 45.5 36.4 0.0 18.2 27.3 45.5 0.0 27.3 9.1 36.4 0.0 54.5 9.1 54.5 0.0 36.4 QWEN2.5 72B 45.5 18.2 9.1 27.3 27.3 18.2 18.2 36.4 9.1 27.3 9.1 54.5 9.1 9.1 54.5 27.3 LLAMA-4-MAV. 27.3 0.0 0.0 72.7 27.3 0.0 0.0 72.7 0.0 9.1 18.2 72.7 9.1 0.0 63.6 27.3 GPT-4O-MINI 63.6 36.4 0.0 0.0 54.5 27.3 18.2 0.0 63.6 36.4 0.0 0.0 45.5 45.5 9.1 0.0

USA-China

GIGACHAT-MAX 50.0 21.4 0.0 28.6 50.0 21.4 0.0 28.6 21.4 21.4 0.0 57.1 14.3 21.4 7.1 57.1 QWEN2.5 72B 50.0 21.4 0.0 28.6 50.0 7.1 7.1 35.7 28.6 21.4 7.1 42.9 7.1 14.3 50.0 28.6 LLAMA-4-MAV. 42.9 28.6 0.0 28.6 35.7 28.6 0.0 35.7 21.4 21.4 0.0 57.1 21.4 14.3 64.3 0.0 GPT-4O-MINI 57.1 42.9 0.0 0.0 64.3 35.7 0.0 0.0 78.6 14.3 7.1 0.0 57.1 21.4 21.4 0.0

USSR-China

GIGACHAT-MAX 17.2 55.2 17.2 10.3 20.7 48.3 20.7 10.3 17.2 31.0 10.3 41.4 3.4 62.1 17.2 17.2 QWEN2.5 72B 31.0 27.6 13.8 27.6 24.1 34.5 6.9 34.5 17.2 24.1 10.3 48.3 6.9 24.1 48.3 20.7 LLAMA-4-MAV. 27.6 20.7 6.9 44.8 17.2 24.1 13.8 44.8 17.2 10.3 17.2 55.2 6.9 10.3 62.1 20.7 GPT-4O-MINI 44.8 51.7 3.4 0.0 48.3 44.8 6.9 0.0 51.7 34.5 13.8 0.0 27.6 41.4 31.0 0.0

USSR-USA

GIGACHAT-MAX 28.0 56.0 4.0 12.0 28.0 56.0 8.0 8.0 12.0 28.0 0.0 60.0 16.0 36.0 12.0 36.0 QWEN2.5 72B 32.0 40.0 4.0 24.0 32.0 36.0 4.0 28.0 24.0 28.0 4.0 44.0 12.0 36.0 28.0 24.0 LLAMA-4-MAV. 4.0 16.0 8.0 72.0 4.0 20.0 8.0 68.0 8.0 16.0 12.0 64.0 12.0 12.0 56.0 20.0 GPT-4O-MINI 16.0 80.0 4.0 0.0 12.0 84.0 4.0 0.0 12.0 72.0 16.0 0.0 28.0 40.0 32.0 0.0

- Table 7: Comparison of model responses (%) for all participant pairs across different experimental settings (Chinese language). For each pair, A and B denote the first and second participant countries, respectively (see Participants

###### Baseline Debias Prompt Mentioned Participant Substituted Participants Participants Model A B Inc. Eq. A B Inc. Eq. A B Inc. Eq. A B Inc. Eq.

GIGACHAT-MAX 15.8 84.2 0.0 0.0 15.8 84.2 0.0 0.0 5.3 94.7 0.0 0.0 26.3 68.4 0.0 5.3 QWEN2.5 72B 10.5 89.5 0.0 0.0 0.0 84.2 5.3 10.5 0.0 100.0 0.0 0.0 21.1 73.7 5.3 0.0 LLAMA-4-MAV. 5.3 78.9 0.0 15.8 5.3 63.2 5.3 26.3 0.0 94.7 0.0 5.3 15.8 42.1 26.3 15.8 GPT-4O-MINI 10.5 89.5 0.0 0.0 10.5 89.5 0.0 0.0 0.0 100.0 0.0 0.0 57.9 42.1 0.0 0.0

UK-China

GIGACHAT-MAX 9.1 54.5 18.2 18.2 9.1 54.5 18.2 18.2 9.1 54.5 18.2 18.2 9.1 36.4 27.3 27.3 QWEN2.5 72B 18.2 45.5 18.2 18.2 27.3 54.5 0.0 18.2 18.2 45.5 9.1 27.3 9.1 18.2 45.5 27.3 LLAMA-4-MAV. 27.3 27.3 9.1 36.4 18.2 18.2 18.2 45.5 0.0 27.3 9.1 63.6 9.1 0.0 63.6 27.3 GPT-4O-MINI 36.4 36.4 27.3 0.0 27.3 36.4 36.4 0.0 9.1 36.4 54.5 0.0 18.2 18.2 63.6 0.0

UK-USA

GIGACHAT-MAX 9.1 72.7 0.0 18.2 9.1 72.7 0.0 18.2 0.0 81.8 0.0 18.2 9.1 54.5 0.0 36.4 QWEN2.5 72B 27.3 36.4 9.1 27.3 18.2 27.3 9.1 45.5 9.1 45.5 18.2 27.3 0.0 27.3 45.5 27.3 LLAMA-4-MAV. 18.2 0.0 0.0 81.8 18.2 0.0 9.1 72.7 0.0 0.0 27.3 72.7 0.0 18.2 63.6 18.2 GPT-4O-MINI 36.4 54.5 9.1 0.0 36.4 54.5 9.1 0.0 18.2 54.5 18.2 9.1 36.4 27.3 36.4 0.0

UK-USSR

GIGACHAT-MAX 0.0 100.0 0.0 0.0 0.0 92.9 0.0 7.1 0.0 92.9 0.0 7.1 35.7 42.9 0.0 21.4 QWEN2.5 72B 14.3 57.1 14.3 14.3 14.3 57.1 7.1 21.4 0.0 100.0 0.0 0.0 14.3 42.9 35.7 7.1 LLAMA-4-MAV. 14.3 57.1 0.0 28.6 7.1 57.1 0.0 35.7 0.0 92.9 0.0 7.1 35.7 42.9 14.3 7.1 GPT-4O-MINI 7.1 92.9 0.0 0.0 7.1 92.9 0.0 0.0 0.0 100.0 0.0 0.0 64.3 21.4 14.3 0.0

USA-China

GIGACHAT-MAX 6.9 75.9 6.9 10.3 6.9 69.0 10.3 13.8 0.0 93.1 0.0 6.9 34.5 48.3 6.9 10.3 QWEN2.5 72B 10.3 86.2 0.0 3.4 6.9 72.4 13.8 6.9 6.9 75.9 3.4 13.8 27.6 41.4 20.7 10.3 LLAMA-4-MAV. 6.9 58.6 6.9 27.6 3.4 48.3 3.4 44.8 3.4 69.0 6.9 20.7 58.6 24.1 13.8 3.4 GPT-4O-MINI 10.3 89.7 0.0 0.0 13.8 86.2 0.0 0.0 0.0 96.6 3.4 0.0 79.3 20.7 0.0 0.0

USSR-China

GIGACHAT-MAX 28.0 52.0 4.0 16.0 20.0 48.0 8.0 24.0 16.0 36.0 4.0 44.0 12.0 64.0 20.0 4.0 QWEN2.5 72B 32.0 28.0 12.0 28.0 20.0 40.0 8.0 32.0 28.0 24.0 12.0 36.0 16.0 40.0 32.0 12.0 LLAMA-4-MAV. 16.0 12.0 8.0 64.0 16.0 8.0 8.0 68.0 16.0 20.0 24.0 40.0 12.0 12.0 64.0 12.0 GPT-4O-MINI 44.0 52.0 4.0 0.0 48.0 44.0 8.0 0.0 52.0 36.0 12.0 0.0 28.0 28.0 44.0 0.0

USSR-USA

- Table 8: Comparison of model responses (%) for all participant pairs across different experimental settings (Chinese language, Chinese patriot). For each pair, A and B denote the first and second participant countries, respectively (see Participants column). ’Inc.’ stands for ’Both Incorrect’ and ’Eq.’ for ’Both Equal’.

Baseline Debias Prompt Mentioned Participant Substituted Participants Participants Model A B Inc. Eq. A B Inc. Eq. A B Inc. Eq. A B Inc. Eq.

UK-China

GIGACHAT-MAX 36.8 57.9 0.0 5.3 31.6 63.2 0.0 5.3 15.8 52.6 0.0 31.6 15.8 78.9 5.3 0.0 QWEN2.5 72B 42.1 31.6 5.3 21.1 21.1 31.6 5.3 42.1 10.5 21.1 5.3 63.2 5.3 10.5 84.2 0.0 LLAMA-4-MAV. 36.8 21.1 10.5 31.6 36.8 21.1 10.5 31.6 52.6 21.1 21.1 5.3 5.3 5.3 89.5 0.0 GPT-4O-MINI 42.1 57.9 0.0 0.0 42.1 57.9 0.0 0.0 52.6 42.1 5.3 0.0 31.6 26.3 42.1 0.0

UK-USA

GIGACHAT-MAX 27.3 63.6 9.1 0.0 18.2 45.5 9.1 27.3 0.0 63.6 0.0 36.4 0.0 45.5 36.4 18.2 QWEN2.5 72B 9.1 45.5 9.1 36.4 9.1 63.6 0.0 27.3 9.1 27.3 0.0 63.6 0.0 27.3 54.5 18.2 LLAMA-4-MAV. 18.2 18.2 9.1 54.5 18.2 9.1 9.1 63.6 9.1 18.2 18.2 54.5 9.1 0.0 72.7 18.2 GPT-4O-MINI 18.2 81.8 0.0 0.0 27.3 63.6 9.1 0.0 0.0 81.8 18.2 0.0 9.1 36.4 54.5 0.0

UK-USSR

GIGACHAT-MAX 45.5 18.2 9.1 27.3 36.4 18.2 18.2 27.3 18.2 27.3 9.1 45.5 9.1 27.3 27.3 36.4 QWEN2.5 72B 45.5 9.1 0.0 45.5 36.4 18.2 0.0 45.5 36.4 9.1 0.0 54.5 27.3 9.1 27.3 36.4 LLAMA-4-MAV. 45.5 0.0 0.0 54.5 45.5 0.0 0.0 54.5 36.4 0.0 27.3 36.4 9.1 0.0 63.6 27.3 GPT-4O-MINI 36.4 54.5 0.0 9.1 54.5 45.5 0.0 0.0 54.5 45.5 0.0 0.0 27.3 36.4 36.4 0.0

USA-China

GIGACHAT-MAX 28.6 42.9 7.1 21.4 28.6 35.7 7.1 28.6 14.3 42.9 0.0 42.9 21.4 57.1 7.1 14.3 QWEN2.5 72B 14.3 21.4 14.3 50.0 7.1 14.3 14.3 64.3 7.1 21.4 7.1 64.3 0.0 35.7 64.3 0.0 LLAMA-4-MAV. 35.7 7.1 0.0 57.1 14.3 7.1 14.3 64.3 35.7 14.3 28.6 21.4 0.0 7.1 85.7 7.1 GPT-4O-MINI 42.9 57.1 0.0 0.0 50.0 50.0 0.0 0.0 57.1 35.7 7.1 0.0 0.0 42.9 57.1 0.0

USSR-China

GIGACHAT-MAX 34.5 37.9 13.8 13.8 31.0 24.1 31.0 13.8 13.8 27.6 13.8 44.8 0.0 65.5 24.1 10.3 QWEN2.5 72B 34.5 20.7 20.7 24.1 34.5 24.1 20.7 20.7 17.2 10.3 17.2 55.2 3.4 34.5 55.2 6.9 LLAMA-4-MAV. 17.2 10.3 17.2 55.2 20.7 10.3 17.2 51.7 13.8 17.2 20.7 48.3 6.9 17.2 62.1 13.8 GPT-4O-MINI 51.7 44.8 3.4 0.0 44.8 44.8 10.3 0.0 37.9 34.5 27.6 0.0 6.9 51.7 41.4 0.0

USSR-USA

GIGACHAT-MAX 36.0 48.0 0.0 16.0 32.0 48.0 0.0 20.0 24.0 40.0 4.0 32.0 12.0 52.0 16.0 20.0 QWEN2.5 72B 16.0 28.0 8.0 48.0 12.0 28.0 4.0 56.0 16.0 24.0 4.0 56.0 4.0 24.0 52.0 20.0 LLAMA-4-MAV. 28.0 16.0 0.0 56.0 28.0 12.0 0.0 60.0 12.0 8.0 44.0 36.0 12.0 4.0 72.0 12.0 GPT-4O-MINI 36.0 64.0 0.0 0.0 32.0 64.0 4.0 0.0 24.0 64.0 12.0 0.0 4.0 48.0 48.0 0.0

- Table 9: Comparison of model responses (%) for all participant pairs across different experimental settings (Russian language.). For each pair, A and B denote the first and second participant countries, respectively (see Participants

###### Baseline Debias Prompt Mention Participant Substituted Participants Participants Model A B Inc. Eq. A B Inc. Eq. A B Inc. Eq. A B Inc. Eq.

GIGACHAT-MAX 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 QWEN2.5 72B 0.0 94.7 5.3 0.0 0.0 89.5 5.3 5.3 0.0 100.0 0.0 0.0 5.3 84.2 10.5 0.0 LLAMA-4-MAV. 5.3 89.5 0.0 5.3 0.0 73.7 0.0 26.3 0.0 100.0 0.0 0.0 21.1 63.2 10.5 5.3 GPT-4O-MINI 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 36.8 63.2 0.0 0.0

UK-China

GIGACHAT-MAX 0.0 63.6 18.2 18.2 0.0 36.4 45.5 18.2 9.1 63.6 18.2 9.1 9.1 27.3 54.5 9.1 QWEN2.5 72B 18.2 36.4 27.3 18.2 0.0 45.5 18.2 36.4 0.0 36.4 27.3 36.4 0.0 45.5 54.5 0.0 LLAMA-4-MAV. 0.0 18.2 18.2 63.6 0.0 18.2 27.3 54.5 0.0 0.0 63.6 36.4 0.0 9.1 81.8 9.1 GPT-4O-MINI 0.0 36.4 63.6 0.0 9.1 36.4 54.5 0.0 0.0 9.1 90.9 0.0 0.0 9.1 90.9 0.0

UK-USA

GIGACHAT-MAX 27.3 36.4 9.1 27.3 18.2 45.5 9.1 27.3 9.1 54.5 9.1 27.3 9.1 36.4 27.3 27.3 QWEN2.5 72B 0.0 72.7 0.0 27.3 18.2 18.2 18.2 45.5 0.0 54.5 0.0 45.5 9.1 36.4 36.4 18.2 LLAMA-4-MAV. 9.1 18.2 27.3 45.5 0.0 18.2 18.2 63.6 0.0 36.4 27.3 36.4 9.1 0.0 45.5 45.5 GPT-4O-MINI 0.0 90.9 9.1 0.0 9.1 63.6 18.2 9.1 0.0 81.8 18.2 0.0 0.0 27.3 72.7 0.0

UK-USSR

GIGACHAT-MAX 7.1 92.9 0.0 0.0 7.1 85.7 7.1 0.0 0.0 100.0 0.0 0.0 7.1 85.7 7.1 0.0 QWEN2.5 72B 7.1 92.9 0.0 0.0 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 7.1 78.6 14.3 0.0 LLAMA-4-MAV. 0.0 78.6 14.3 7.1 7.1 64.3 7.1 21.4 0.0 100.0 0.0 0.0 14.3 42.9 35.7 7.1 GPT-4O-MINI 0.0 92.9 7.1 0.0 0.0 92.9 7.1 0.0 0.0 100.0 0.0 0.0 50.0 50.0 0.0 0.0

USA-China

GIGACHAT-MAX 3.4 79.3 10.3 6.9 3.4 69.0 13.8 13.8 0.0 100.0 0.0 0.0 13.8 72.4 10.3 3.4 QWEN2.5 72B 17.2 65.5 10.3 6.9 13.8 62.1 13.8 10.3 3.4 86.2 6.9 3.4 20.7 62.1 13.8 3.4 LLAMA-4-MAV. 3.4 62.1 6.9 27.6 6.9 55.2 3.4 34.5 6.9 93.1 0.0 0.0 48.3 31.0 6.9 13.8 GPT-4O-MINI 10.3 89.7 0.0 0.0 10.3 89.7 0.0 0.0 0.0 100.0 0.0 0.0 82.8 17.2 0.0 0.0

USSR-China

GIGACHAT-MAX 44.0 56.0 0.0 0.0 32.0 44.0 16.0 8.0 36.0 40.0 12.0 12.0 28.0 44.0 28.0 0.0 QWEN2.5 72B 52.0 28.0 4.0 16.0 40.0 20.0 4.0 36.0 36.0 16.0 20.0 28.0 16.0 32.0 44.0 8.0 LLAMA-4-MAV. 36.0 12.0 12.0 40.0 24.0 8.0 12.0 56.0 36.0 0.0 44.0 20.0 20.0 16.0 56.0 8.0 GPT-4O-MINI 52.0 36.0 12.0 0.0 56.0 32.0 12.0 0.0 36.0 4.0 60.0 0.0 8.0 24.0 68.0 0.0

USSR-USA

- Table 10: Comparison of model responses (%) for all participant pairs for Russian language, Chinese patriot. For each pair, A and B denote the first and second participant countries, respectively (see Participants column). ’Inc.’ stands for ’Both Incorrect’ and ’Eq.’ for ’Both Equal’.

Baseline Debias Prompt Mention Participant Substituted Participants Participants Model A B Inc. Eq. A B Inc. Eq. A B Inc. Eq. A B Inc. Eq.

UK-China

GIGACHAT-MAX 26.3 21.1 0.0 52.6 21.1 21.1 0.0 57.9 0.0 5.3 0.0 94.7 21.1 36.8 0.0 42.1 QWEN2.5 72B 10.5 21.1 5.3 63.2 5.3 0.0 5.3 89.5 0.0 15.8 0.0 84.2 5.3 5.3 68.4 21.1 LLAMA-4-MAV. 26.3 5.3 0.0 68.4 21.1 5.3 0.0 73.7 26.3 5.3 5.3 63.2 5.3 10.5 78.9 5.3 GPT-4O-MINI 47.4 36.8 15.8 0.0 36.8 42.1 15.8 5.3 42.1 52.6 0.0 5.3 31.6 21.1 47.4 0.0

UK-USA

GIGACHAT-MAX 9.1 27.3 0.0 63.6 9.1 9.1 0.0 81.8 0.0 0.0 0.0 100.0 0.0 18.2 18.2 63.6 QWEN2.5 72B 9.1 0.0 0.0 90.9 9.1 18.2 0.0 72.7 0.0 9.1 0.0 90.9 0.0 9.1 45.5 45.5 LLAMA-4-MAV. 0.0 9.1 9.1 81.8 9.1 9.1 9.1 72.7 0.0 0.0 0.0 100.0 9.1 0.0 54.5 36.4 GPT-4O-MINI 18.2 72.7 9.1 0.0 9.1 72.7 18.2 0.0 9.1 81.8 9.1 0.0 27.3 18.2 54.5 0.0

UK-USSR

GIGACHAT-MAX 36.4 18.2 0.0 45.5 36.4 18.2 0.0 45.5 36.4 0.0 0.0 63.6 9.1 27.3 9.1 54.5 QWEN2.5 72B 27.3 18.2 0.0 54.5 27.3 9.1 9.1 54.5 9.1 0.0 9.1 81.8 9.1 9.1 45.5 36.4 LLAMA-4-MAV. 18.2 9.1 9.1 63.6 18.2 9.1 9.1 63.6 18.2 0.0 9.1 72.7 0.0 0.0 63.6 36.4 GPT-4O-MINI 54.5 27.3 9.1 9.1 54.5 18.2 9.1 18.2 36.4 27.3 9.1 27.3 27.3 9.1 36.4 27.3

USA-China

GIGACHAT-MAX 21.4 21.4 0.0 57.1 21.4 21.4 0.0 57.1 0.0 14.3 0.0 85.7 7.1 14.3 0.0 78.6 QWEN2.5 72B 0.0 7.1 0.0 92.9 0.0 7.1 0.0 92.9 0.0 0.0 0.0 100.0 0.0 0.0 28.6 71.4 LLAMA-4-MAV. 0.0 7.1 0.0 92.9 0.0 7.1 0.0 92.9 14.3 0.0 0.0 85.7 21.4 0.0 64.3 14.3 GPT-4O-MINI 85.7 14.3 0.0 0.0 64.3 14.3 0.0 21.4 78.6 14.3 0.0 7.1 21.4 21.4 42.9 14.3

USSR-China

GIGACHAT-MAX 10.3 34.5 13.8 41.4 10.3 31.0 17.2 41.4 0.0 31.0 10.3 58.6 0.0 31.0 24.1 44.8 QWEN2.5 72B 20.7 17.2 3.4 58.6 10.3 13.8 3.4 72.4 3.4 20.7 0.0 75.9 3.4 17.2 41.4 37.9 LLAMA-4-MAV. 10.3 3.4 3.4 82.8 10.3 0.0 10.3 79.3 0.0 6.9 6.9 86.2 3.4 6.9 55.2 34.5 GPT-4O-MINI 17.2 44.8 34.5 3.4 13.8 51.7 31.0 3.4 17.2 44.8 31.0 6.9 3.4 44.8 51.7 0.0

USSR-USA

GIGACHAT-MAX 12.0 40.0 0.0 48.0 12.0 40.0 0.0 48.0 8.0 20.0 0.0 72.0 20.0 56.0 8.0 16.0 QWEN2.5 72B 4.0 16.0 4.0 76.0 0.0 16.0 4.0 80.0 0.0 12.0 4.0 84.0 0.0 28.0 12.0 60.0 LLAMA-4-MAV. 0.0 12.0 4.0 84.0 0.0 4.0 8.0 88.0 0.0 4.0 12.0 84.0 8.0 16.0 52.0 24.0 GPT-4O-MINI 24.0 64.0 8.0 4.0 28.0 64.0 8.0 0.0 8.0 80.0 12.0 0.0 24.0 44.0 32.0 0.0

- Table 11: Comparison of model responses (%) for all participant pairs for French language. For each pair, A and B denote the first and second participant countries, respectively (see Participants column). ’Inc.’ stands for ’Both Incorrect’ and ’Eq.’ for ’Both Equal’.

###### Baseline Debias Prompt Mention Participant Substituted Participants Participants Model A B Inc. Eq. A B Inc. Eq. A B Inc. Eq. A B Inc. Eq.

GIGACHAT-MAX 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 10.5 84.2 0.0 5.3 QWEN2.5 72B 0.0 94.7 0.0 5.3 0.0 94.7 0.0 5.3 0.0 100.0 0.0 0.0 10.5 89.5 0.0 0.0 LLAMA-4-MAV. 5.3 84.2 0.0 10.5 5.3 73.7 0.0 21.1 0.0 94.7 0.0 5.3 10.5 63.2 21.1 5.3 GPT-4O-MINI 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 47.4 52.6 0.0 0.0

UK-China

GIGACHAT-MAX 0.0 9.1 9.1 81.8 0.0 18.2 9.1 72.7 0.0 27.3 9.1 63.6 0.0 18.2 27.3 54.5 QWEN2.5 72B 9.1 45.5 0.0 45.5 9.1 36.4 0.0 54.5 0.0 18.2 0.0 81.8 9.1 27.3 27.3 36.4 LLAMA-4-MAV. 9.1 9.1 18.2 63.6 9.1 18.2 9.1 63.6 0.0 0.0 27.3 72.7 0.0 0.0 63.6 36.4 GPT-4O-MINI 9.1 36.4 54.5 0.0 0.0 36.4 63.6 0.0 9.1 9.1 81.8 0.0 0.0 9.1 90.9 0.0

UK-USA

GIGACHAT-MAX 0.0 72.7 0.0 27.3 9.1 63.6 0.0 27.3 0.0 36.4 0.0 63.6 9.1 36.4 9.1 45.5 QWEN2.5 72B 27.3 45.5 0.0 27.3 18.2 36.4 0.0 45.5 9.1 45.5 0.0 45.5 18.2 45.5 9.1 27.3 LLAMA-4-MAV. 18.2 18.2 9.1 54.5 9.1 18.2 0.0 72.7 0.0 45.5 0.0 54.5 0.0 9.1 63.6 27.3 GPT-4O-MINI 9.1 63.6 27.3 0.0 9.1 63.6 18.2 9.1 0.0 63.6 27.3 9.1 0.0 9.1 81.8 9.1

UK-USSR

GIGACHAT-MAX 0.0 100.0 0.0 0.0 0.0 71.4 0.0 28.6 0.0 85.7 0.0 14.3 7.1 57.1 0.0 35.7 QWEN2.5 72B 0.0 85.7 0.0 14.3 0.0 92.9 0.0 7.1 0.0 100.0 0.0 0.0 0.0 78.6 0.0 21.4 LLAMA-4-MAV. 0.0 71.4 0.0 28.6 0.0 50.0 0.0 50.0 0.0 92.9 0.0 7.1 7.1 50.0 35.7 7.1 GPT-4O-MINI 0.0 92.9 0.0 7.1 0.0 100.0 0.0 0.0 0.0 100.0 0.0 0.0 21.4 64.3 7.1 7.1

USA-China

GIGACHAT-MAX 0.0 86.2 0.0 13.8 0.0 79.3 0.0 20.7 0.0 79.3 0.0 20.7 13.8 58.6 0.0 27.6 QWEN2.5 72B 3.4 79.3 0.0 17.2 0.0 65.5 0.0 34.5 0.0 82.8 0.0 17.2 13.8 55.2 3.4 27.6 LLAMA-4-MAV. 0.0 51.7 3.4 44.8 3.4 44.8 0.0 51.7 0.0 69.0 3.4 27.6 27.6 20.7 24.1 27.6 GPT-4O-MINI 0.0 100.0 0.0 0.0 0.0 96.6 3.4 0.0 0.0 96.6 3.4 0.0 69.0 31.0 0.0 0.0

USSR-China

GIGACHAT-MAX 16.0 40.0 0.0 44.0 16.0 36.0 0.0 48.0 20.0 24.0 0.0 56.0 8.0 40.0 32.0 20.0 QWEN2.5 72B 32.0 12.0 0.0 56.0 32.0 0.0 0.0 68.0 24.0 24.0 8.0 44.0 8.0 24.0 40.0 28.0 LLAMA-4-MAV. 20.0 12.0 12.0 56.0 8.0 8.0 12.0 72.0 16.0 8.0 20.0 56.0 12.0 24.0 44.0 20.0 GPT-4O-MINI 52.0 36.0 12.0 0.0 48.0 36.0 12.0 4.0 28.0 24.0 44.0 4.0 8.0 16.0 76.0 0.0

USSR-USA

- Table 12: Comparison of model responses (%) for all participant pairs for French language, Chinese patriot. For each pair, A and B denote the first and second participant countries, respectively (see Participants column). ’Inc.’ stands for ’Both Incorrect’ and ’Eq.’ for ’Both Equal’.

### B.4 One Option Experiments

We conducted an additional group of experiments for the ablation purpose, evaluating models’ geopolitical bias for all four considered countries’ positions independently from the positions of other countries. The prompt structure was slightly modified (Fig. 9) to feed the model with a neutral position and only one country’s position. Models are asked to return the integer answer whether the position is correct (0) or incorrect/misleading (1), and the textual reason. The results are presented in Table 13 and, overall, well aligned with the previous results of paired comparison that positions of USA and UK are more often considered valid than positions of USSR or China (USSR ≤ China ≤ UK ≤ US). Notably, unlike most previous contrastive experiments, the debiased prompt and participatory mentioning work ”against” the USSR.

[Figure 9]

[Figure 10]

(a) Without intervention (b) With debias prompt

[Figure 11]

[Figure 12]

(c) Mention participants (d) Substituted participants

- Figure 5: Probability to change opinion about the country after changing the language to Chinese under different interventions.

[Figure 13]

(a) Without intervention (b) With debias prompt

[Figure 14]

[Figure 15]

(c) Mention participants (d) Substituted participants

[Figure 16]

- Figure 6: Probability to change opinion about the country after changing the language to Chinese for a Chinese patriot under different interventions.

[Figure 17]

(a) Without intervention (b) With debias prompt

[Figure 18]

[Figure 19]

(c) Mention participants (d) Substituted participants

[Figure 20]

- Figure 7: Probability to change opinion about the country after changing the language to French under different interventions.

##### Baseline Debias Prompt Mention Participant

Model China UK USA USSR China UK USA USSR China UK USA USSR GIGACHAT-MAX 58.1 80.4 90.0 67.2 61.3 80.5 88.0 58.5 59.7 80.5 88.0 61.5 QWEN2.5 72B 74.2 75.6 92.0 63.1 77.4 85.4 88.0 56.9 72.6 82.9 88.0 60.0 LLAMA-4-MAV. 80.7 75.6 86.0 63.1 82.3 78.1 90.0 56.9 80.7 75.6 84.0 63.1 GPT-4O-MINI 27.4 39.0 64.0 26.2 38.7 46.3 68.0 27.7 38.7 46.3 68.0 23.1

- Table 13: Comparison of Model Responses (%) in One Statement Scenario for Every Country in Different Experimental Settings. The percentages represent the fraction of the country’s positions considered as ”correct position” instead of ”incorrect/misleading”. All prompts and positions were in English.

[Figure 21]

[Figure 22]

(a) Without intervention (b) With debias prompt

[Figure 23]

[Figure 24]

(c) Mention participants (d) Substituted participants

- Figure 8: Probability to change opinion about the country after changing the language to French for a Chinese patriot under different interventions.

|We are talking about a historical event and we have a position to evaluate. Your goal is to decide if this position is correct or incorrect/misleading based on the historical event. Event: [Neutral description of the event] Position: [Viewpoint of the considered Country] If MENTIONED_PARTICIPANT:<br><br>This is a position of [Country] Provide your answer as JSON object with field ’correct_position’: int (0 for correct position, 1 for incorrect/misleading position) and ’reason’: str If DEBIAS:<br><br>Please ensure that your answer is unbiased and free from reliance on stereotypes.|
|---|

- Figure 9: Example prompt structure for instructing LLM to provide answers in JSON format in 1 position format.

