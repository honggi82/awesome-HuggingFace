# arXiv:2502.00094v2[cs.CV]4Feb2025

[Figure 1]

## AIN: The Arabic INclusive Large Multimodal Model

Ahmed Heakl1∗ Sara Ghaboura1* Omkar Thawakar1 Fahad Shahbaz Khan1,2 Hisham Cholakkal1 Rao Muhammad Anwer1,3 Salman Khan1,4

1Mohamed bin Zayed University of AI, 2Linköping University, 3Aalto University, 4Australian National University

[Figure 2]

##### AIN Demo AIN Webpage AIN GitHub

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

[Figure 17]

OCR & Doc. Understanding Remote Sensing Understanding Agricultural Understanding Total (All Domains)

Figure 1: Cross-domain performance analysis on the Camel-Bench Benchmark. Our AIN-7B achieves promising performance compared to significantly bigger models (GPT-4o and Gemini-1.5-Pro) in both domainspecific and aggregate settings. Despite its smaller size, our AIN-7B achieves competitive performance across all 38 sub-domains with significantly superior capabilities on OCR & document understanding.

### Abstract

Amid the swift progress of large language models (LLMs) and their evolution into large multimodal models (LMMs), significant strides have been made in high-resource languages such as English and Chinese. While Arabic LLMs have seen notable progress, Arabic LMMs remain largely unexplored, often narrowly focusing on a few specific aspects of the language and visual understanding. To bridge this gap, we introduce AIN—the Arabic Inclusive Multimodal Model—designed to excel across diverse domains. AIN is an English-Arabic bilingual LMM designed to excel in English and Arabic, leveraging carefully constructed 3.6 million high-quality Arabic-English multimodal data samples. AIN demonstrates state-of-the-art Arabic performance, while also possessing strong English-language visual capabilities. On the recent CAMEL-Bench benchmark comprising 38 sub-domains including, multi-image understanding, complex visual perception, handwritten document understanding, video understanding, medical imaging, plant diseases, and remote sensing-based land use understanding, our AIN demonstrates strong performance with the 7B model outperforming GPT-4o by an absolute gain of 3.4% averaged over eight domains and 38 sub-domains. AIN’s superior capabilities position it as a significant step toward empowering Arabic speakers with advanced multimodal generative AI tools across diverse applications.

Technical Report of AIN: The Arabic INclusive Large Multimodal Model.

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

- Figure 2: AIN: A versatile LMM excelling in visual and contextual understanding across diverse domains, including VQA on complex topics, OCR for various fonts and handwriting, cultural insights (traditions, food, places), agricultural tasks (crop identification, fruit classification, disease detection), remote sensing (multi-scale objects), medical imaging (various modalities), and video analysis (animation, human activities).

### 1 AIN Capabilities

The AIN model is an advanced Arabic Large Multimodal Model (See Figure 2) with strong English proficiency, built on 7 billion parameters derived from the Qwen-2-VL-7B architecture [2]. Its performance highlights significant progress in multimodal understanding, excelling in complex reasoning, cross-lingual tasks, and detailed image-text alignment across diverse benchmarks.

#### 1.1 Quantitative Results

We present the quantitative performance of the model across benchmarks and domains. Figure 3 showcases a comprehensive performance analysis of AIN-7B across CAMEL-Bench [1] domains, comparing it with a variety of models including, GPT-4o [3], GPT-4o-mini [4], Gemini-1.5-Pro [5], Qwen2-VL-7B [2], LlaVA-NeXt-7B [6], and Pangea-7B [7]. The plot demonstrates the superior performance of AIN-7B in most domains. Table 1 shows that our AIN-7B achieves the best overall performance across all domains, compared to existing models.

We conduct a comprehensive evaluation of the bilingual capabilities of AIN in Arabic and English using well-established benchmarks. To further assess AIN’s proficiency in Arabic beyond CAMELBench, we evaluate its performance across all topics of ArabicMMLU [13]. As shown in Table 2, AIN outperforms in 14 out of 19 evaluated categories, achieving a significant 3% overall improvement compared to Qwen2-VL-7B [2]. Similarly, as presented in Table 3, AIN demonstrates strong English language capabilities on ten benchmarks that cover general VQA, mathematics, science, and visual chart interpretation.

#### 1.2 Qualitative Results

The qualitative assessment of AIN’s contextual response generation reveals its proficiency across diverse domains, as illustrated in Figure 4. The model excels in general VQA tasks, demonstrating advanced capabilities in OCR and document analysis, including both machine-printed text and handwriting interpretation. Additionally, it achieves prominent performance in medical image interpretation, scientific visualization comprehension, and remote sensing analysis. The model’s capabilities extend to intricate data visualization interpretation, where it demonstrates skilled extrapolative understanding from charts and diagrams. Notably, AIN exhibits strong cultural-specific understanding through object recognition across diverse contexts, such as places, food items, and

∗Equal Contribution

VQA

|Medical|OCR<br><br>|
|---|---|
|Agro|RS|

AIN-7B (ours)

GPT-4o

GPT-4o-mini

Gemini-1.5-Pro

Qwen2-VL-7B

LLaVa-NeXt-7B

Pangea-7B

Cultural

Video

CDT

- Figure 3: AIN compared to existing LMMs across CAMEL-Bench benchmark [1] domains: OCR: “OCR & Document Understanding”, Video: “General Video & Multi-Image Understanding”, RS: “Remote Sensing Understanding”, CDT:“Chart, Diagram & Table Understanding”, Agro.: “Agricultural Image Understanding”, Cultural: “Cultural-Specific Understanding”, Medical: “Medical Image Understanding”.

- Table 1: Performance comparison of AIN and different closed- and open-source LMMs across CAMEL-Bench domains [1]. Best performance is highlighted in green; second-best performance is underlined. OCR*: “OCR & Document Understanding”, Video*: “General Video & Multi-Image Understanding”, RS*: “Remote Sensing Understanding”, CDT*:“Chart, Diagram & Table Understanding”, Agro.*: “Agricultural Image Understanding”, Cult.*: “Cultural-Specific Understanding”, Med.*: “Medical Image Understanding”.

Models VQA OCR Video RS CDT Agro. Cult. Med. Total GPT-4o [3] 55.15 54.98 69.65 27.36 62.35 80.75 80.86 49.91 60.13 GPT-4o-mini [4] 48.83 39.38 66.28 16.93 56.37 78.80 65.92 47.37 52.49 Gemini-1.5-Pro [8] 46.68 28.68 42.95 17.07 47.06 72.14 56.24 33.78 52.38 Gemini-1.5-flash [8] 45.59 27.58 53.31 14.95 48.26 76.07 46.54 42.87 44.40 InternVL-8B [9] 30.41 15.91 51.42 5.36 30.27 44.47 20.88 29.48 28.52 InternVL2.5-1B [10] 27.22 19.45 38.20 3.39 30.75 39.53 35.68 21.27 26.94 Qwen-VL-2B [2] 41.02 22.93 38.90 12.56 27.83 52.02 34.28 29.12 32.33 Qwen2-VL-7B [2] 48.76 42.73 61.97 21.30 54.67 79.32 75.96 35.81 52.56 LLaVa-NeXt-7B [6] 26.33 19.12 44.90 8.33 27.56 42.00 28.30 22.54 27.39 LLaVa-OneVision [11] 42.90 31.35 29.41 10.72 40.86 75.03 66.02 27.29 40.45 Pangea-7B [7] 40.09 17.75 49.01 6.67 38.87 74.51 20.34 31.99 34.90 Maya-8B [12] 39.07 26.70 47.23 27.53 34.25 70.61 57.42 31.57 41.80 AIN-7B (ours) 56.78 72.35 64.09 45.92 64.10 85.05 78.09 43.77 63.77

- Table 2: ArabicMMLU all categories’ performance comparison across various models. Best performance is highlighted in green. Comp. Sci*: Computer Science, Gen. Knldge*: General Knowledge, Islamic Std.*: Islamic Studies, Managmt: Management, Nat. Sci*: Natural Science, Political Sci.*: Political Science, Social Sci.*: Social Science.

Model Accounting Arabic Lang. Biology Civics Comp. Sci.*

- Qwen2-VL-7B [2] 52.7 51.34 44 48.92 65.87 AIN-7B (ours) 59.46 55.41 43.72 47.37 69

Model Driving Test Economics Gen. Knldge* Geography History Qwen2-VL-7B [2] 67.88 59.59 55.18 49.67 43.1 AIN-7B (ours) 69.69 59.25 57.01 55.16 45.16

Model Islamic Std.* Law Managmt* Math Nat. Sci.*

Qwen2-VL-7B [2] 55.93 58.92 68 64.79 71.45 AIN-7B (ours) 58.64 71.66 64 66.99 78.37

Model Philosophy Physics Political Sci.* Social Sci.* Total

- Qwen2-VL-7B [2] 53.85 39.61 54.29 65.75 56.36 AIN-7B (ours) 56.41 45.1 60 65.43 59.36

- Table 3: Comprehensive performance comparison of AIN-7B against Qwen2-VL-7B across 10 English benchmarks, with relative performance gain (↑) indicated. Best performance is highlighted in green. Our AIN-7B achieves promising performance on English language across these diverse benchmarks. Benchmarks: MMBench [14], MME [15], MMMU [16], POPE[17], SEED[18], MathVista [19], ScienceQA [20], ChartQA [21], AI2D [22], MMT-Bench [23].

Model MMBench MME MMMU POPE SEED Qwen2-VL-7B [2] 81.78 1,675.90 52 86.13 76.42 AIN-7B (ours) 93.76 1,689.02 54.2 87.59 78.35 Improved by ↑ ↑ 11.98 ↑ 13.12 ↑ 2.2 ↑ 1.46 ↑ 1.93

###### Model MathVista ScienceQA ChartQA AI2D MMT-Bench

Qwen2-VL-7B [2] 61.19 85.87 83.16 82.77 63.29 AIN-7B (ours) 63.9 91.82 85.12 83.29 63.64 Improved by ↑ ↑ 2.71 ↑ 5.95 ↑ 1.96 ↑ 0.52 ↑ 0.35

###### celebratory scenes. Throughout all tasks, AIN consistently produces accurate, contextually relevant, and comprehensive responses, underscoring its versatility in handling complex visual-linguistic tasks.

|General VQA / Reasoning and Relative Position|
|---|
||[Figure 44]|
|---|
<br><br>[Figure 45]<br><br>[Figure 46]<br><br>[Figure 47]<br><br>[Figure 48]|

|Medical Image Understanding<br><br>(Diagnosis Support)|
|---|
||[Figure 49]|
|---|
<br><br>[Figure 50]<br><br>[Figure 51]<br><br>[Figure 52]<br><br>[Figure 53]|

|General VQA/ Identify and Count<br><br>(Short Answers)|
|---|
||[Figure 54]|
|---|
<br><br>[Figure 55]<br><br>[Figure 56]<br><br>[Figure 57]<br><br>[Figure 58]<br><br>[Figure 59]<br><br>[Figure 60]|

|Cultural-Specific Understanding<br><br>(Food Identification)|
|---|
||[Figure 61]|
|---|
<br><br>[Figure 62]<br><br>[Figure 63]<br><br>[Figure 64]<br><br>[Figure 65]|

|OCR & Document Understanding<br><br>(Font style/ Handwriting/ Language/ Special Characters)|
|---|
|[Figure 66]<br><br>[Figure 67]<br><br>[Figure 68]<br><br>ﮫﺗﻧﯾﻌﻣ جارﺧإ ﻰﻠﻋ ﺎﯾﻧدﻟا ﻲﻓ دﺣأ ردﻘﯾ ﻻ يذﻟا ءﺎﯾﺷﻷا ﻊﯾﻣﺟﻟ :باوﺟﻟا<br><br>مﻧﻐﻟا ضﺑارﻣ ﻲﻓ دﺟﺳﻣﻟا ﻰﻧﺑﯾ نأ لﺑﻗ ﻲﻠﺻﯾ مﻠﺳو ﮫﯾﻠﻋ ﷲ ﻰﻠﺻ ﻲﺑﻧﻟا :باوﺟﻟا<br><br>وﮭﻓ .ﺔﯾﻧورﺗﻛﻟﻹا ةرﺎﺟﺗﻟا ﻰﻟا E-commerce ﺢﻠطﺻﻣ زﻣرﯾ : باوﺟﻟا<br><br>[Figure 69]<br><br>اوﻟﺎﻗ: ”\ نإو ،  لوﺳر ﺎﯾ :باوﺟﻟا<br><br>[Figure 70]<br><br>[Figure 71]<br><br>[Figure 72]<br><br>[Figure 73]|

|Math and Sciences / Chemistry|
|---|
|[Figure 74]<br><br>[Figure 75]<br><br>[Figure 76]|

ّ

|Chat, Diagrams & Table Understanding<br><br>(Bar Charts)|
|---|
||[Figure 77]|
|---|
<br><br>[Figure 78]<br><br>[Figure 79]<br><br>[Figure 80]<br><br>[Figure 81]<br><br>[Figure 82]<br><br>[Figure 83]<br><br>[Figure 84]|

|Remote Sensing Understanding<br><br>(Type/Color Identification)|
|---|
||[Figure 85]|
|---|
<br><br>[Figure 86]<br><br>[Figure 87]<br><br>[Figure 88]<br><br>[Figure 89]<br><br>[Figure 90]<br><br>[Figure 91]|

|Agricultural Image Understanding<br><br>(Plants Diseases)|
|---|
||[Figure 92]|
|---|
<br><br>[Figure 93]<br><br>[Figure 94]<br><br>[Figure 95]<br><br>[Figure 96]<br><br>[Figure 97]<br><br>[Figure 98]<br><br>[Figure 99]|

[Figure 100]

|Cultural-Specific Understanding<br><br>(Places Identification)|
|---|
||[Figure 101]|
|---|
<br><br>[Figure 102]<br><br>[Figure 103]<br><br>[Figure 104]<br><br>[Figure 105]|

|Cultural-Specific Understanding<br><br>(Celebraties Identification)|
|---|
||[Figure 106]|
|---|
<br><br>؟ﺺﺨﺸﻟا اﺬھ ﺪﻠﺑ يأ ﻦﻣ :لاﺆﺴﻟا<br><br>:تارﺎﯿﺨﻟا ﺖﯾﻮﻜﻟا .أ<br><br>ﺔﯾدﻮﻌﺴﻟا .ب ﻦﯾﺮﺤﺒﻟا .ج تارﺎﻣﻹا .د<br><br>[Figure 107]<br><br>[Figure 108]<br><br>[Figure 109]|

|General VQA/ Traffic lights<br><br>(Yes/No Question)|
|---|
||[Figure 110]|
|---|
<br><br>[Figure 111]<br><br>[Figure 112]<br><br>[Figure 113]<br><br>[Figure 114]<br><br>[Figure 115]|

- Figure 4: Qualitative results demonstrating AIN’s comprehensive capabilities across diverse domains. The results show its proficiency in handling both multiple-choice and open-ended questions. Our proposed AIN exhibits robust performance in addressing queries related to visual attributes (shape, color, quantity), while maintaining appropriate response formats (single character, word, or complete sentence) according to task requirements.

For qualitative comparison, Figure 5 highlights AIN’s performance relative to open-source and closed-source LMMs (GPT-4o [3] and LLaVA [11], respectively) across various domains. Unlike its counterparts, which frequently provided incorrect, incomplete answers or failed to adhere to the required format, AIN consistently delivers accurate and contextually appropriate responses. The model demonstrates proficiency in handling diverse query formats, effectively addressing both multiple-choice and open-ended questions with precision and reliability.

|Domain|
|---|
|Question|

|Medical Image Understanding|
|---|
||[Figure 116]|
|---|
<br><br>[Figure 117]<br><br>[Figure 118]|

|OCR & Document Understanding|
|---|
||[Figure 119]|
|---|
<br><br>[Figure 120]|

|General VQA|
|---|
||[Figure 121]|
|---|
<br><br>؟زﻮﺠﻌﻟا ﻞﺟﺮﻟا ﻒﻘﯾ ثﺎﺛأ ﺔﻌﻄﻗ يأ بﻧﺎﺟﺑ :لاؤﺳﻟا<br><br>.ةدﺣاو ةرﺎﺑﻋ وأ ﺔﻣﻠﻛ مادﺧﺗﺳﺎﺑ لاؤﺳﻟا نﻋ بﺟأ<br><br>[Figure 122]|

Question

|Ground Truth|
|---|

|"A" / "أ"|
|---|

|ﻦﻣ مﻮﻗ ﻰﻠﻋ ﺮﻤﻓ ﻰﻠﺻ ﺎﻣ ﺪﻌﺑ جﺮﺧ ﻢﺛ ﻞﺟر ﻢﻠﺳو ﮫﯿﻠﻋ" "سﺪﻘﻤﻟا ﺖﯿﺑ ﻮﺤﻧ ﺮﺼﻌﻟا ةﻼﺻ ﻲﻓ رﺎﺼﻧﻷا|
|---|

|"ﻲﺳﺮﻛ"|
|---|

|[Figure 123]<br><br>GPT-4o|
|---|

|"C" / "ج"<br><br>[Figure 124]|
|---|

|ﻦﻣ مﺪﻗ ﻢﻗ ﻰﻠﺻ ﺎﻣ ﺪﻌﺑ ﺦﺷ ﻢﺛ ﻞﺟر مﺎﺳو ﮫﯿﻠﻋ" "ﺮﯿﺴﻐﺘﻟا ﺖﯿﺑ ﻮﺤﻧ ﺮﺼﻌﻟا ةﻼﺻ ﻲﻓ ﺎﻀﺤﻟا<br><br>[Figure 125]|
|---|

|"ﺔﻟوﺎط"<br><br>[Figure 126]|
|---|

|LlaVA<br><br>[Figure 127]|
|---|

|"1.1.1.1.1.1.1....."<br><br>[Figure 128]|
|---|

|"ﻲھ ﺔﺑﻮﺘﻜﻤﻟا ﺔﻠﻤﺠﻟا"<br><br>[Figure 129]|
|---|

| |"ﺔﻧاﺰﺨﻟا ﺐﻧﺎﺠﺑ زﻮﺠﻌﻟا ﻞﺟﺮﻟا ﻒﻘﯾ"<br><br>[Figure 130]| |
|---|---|---|

|Ours<br><br>[Figure 131]|
|---|

|"A" / "أ"<br><br>[Figure 132]|
|---|

|ﻦﻣ مﻮﻗ ﻰﻠﻋ ﺮﻤﻓ ﻰﻠﺻ ﺎﻣ ﺪﻌﺑ جﺮﺧ ﻢﺛ ﻞﺟر ﻢﻠﺳو ﮫﯿﻠﻋ" "سﺪﻘﻤﻟا ﺖﯿﺑ ﻮﺤﻧ ﺮﺼﻌﻟا ةﻼﺻ ﻲﻓ رﺎﺼﻧﻷا<br><br>[Figure 133]|
|---|

|"ﻲﺳﺮﻛ"<br><br>[Figure 134]|
|---|

ﻦ

|Domain|
|---|
|Question|

|Agricultural Image Understanding|
|---|
||[Figure 135]|
|---|
<br><br>[Figure 136]<br><br>[Figure 137]|

|Cultural Specific Understanding|
|---|
||[Figure 138]|
|---|
<br><br>[Figure 139]<br><br>[Figure 140]|

|Remote Sensing Understanding|
|---|
||[Figure 141]|
|---|
<br><br>[Figure 142]<br><br>[Figure 143]<br><br>hint: the car is parking near the tree in the leftbottom side quadrant|

Question

|Ground Truth|
|---|

|"C" / "ج"|
|---|

|"A" / "أ"|
|---|

|[Figure 144]|
|---|

|[Figure 145]<br><br>GPT-4o|
|---|

|"D" / "د"<br><br>[Figure 146]|
|---|

|"C" / "ج"<br><br>[Figure 147]|
|---|

|[Figure 148]<br><br>[Figure 149]|
|---|

|LlaVA<br><br>[Figure 150]|
|---|

|"[A.]." / "[أ.]."<br><br>[Figure 151]|
|---|

|"[1]"<br><br>[Figure 152]|
|---|

|No Answer / باوﺟ ﻻ<br><br>[Figure 153]|
|---|

|Ours<br><br>[Figure 154]|
|---|

|"C" / "ج"<br><br>[Figure 155]|
|---|

|"A" / "أ"<br><br>[Figure 156]|
|---|

|[Figure 157]<br><br>[Figure 158]|
|---|

- Figure 5: Comparison of AIN with GPT-4o [3] and LLaVA [11] across diverse tasks. The evaluation demonstrates AIN’s proficiency in handling both multiple-choice and open-ended questions while maintaining appropriate response formats.

#### 1.3 Human Feedback

To further evaluate our AIN model, we conduct a qualitative assessment through human feedback, comparing it against closed- and open-source LMMs in a blind setup, where model identities are not revealed to participants. The survey, as shown in Figure 6, covers various real-world domains, including medical diagnosis, road signs, and other scenarios such as low-resolution settings. Targeted at Arabic native speakers, it consists of 10 questions evaluating a range of topics, each with corresponding ground-truth answers. Participants are tasked with selecting the response from the three models that they believe are closest to the ground-truth.

In the survey, “Model 1” represented AIN-7B (ours), “Model 2” corresponds to GPT-4o [3], and “Model 3” is LLaVA [11]. Delivered in MSA, the survey also includes an additional question to gather feedback on language clarity, MSA/ dialect preferences, and the survey itself.

Survey Participation. More than 200 participants from 17 Arab countries (Figure 7), selected from diverse sectors and educational background, completed the survey. The highest contributions came from Saudi Arabia (30%), followed by Egypt (25%), the UAE (13.3%), and Lebanon (13.3%).

Model Preferences. Figure 8 shows that participants predominantly favor Model 1 (AIN-7B (ours)), which received 76% of the votes. GPT-4o followed with 15%, and LLaVA garnered 9%, underscoring AIN’s significant preference among respondents.

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

[Figure 163]

[Figure 164]

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

[Figure 169]

[Figure 170]

[Figure 171]

- Figure 6: AIN human evaluation survey, illustrating assessment criteria and multi-domain questions designed to evaluate multi-task and complex reasoning capabilities. The survey includes evaluations on specific food items, road signs in low-resolution settings, celebrities, charts, remote sensing tasks, and other diverse topics to comprehensively assess performance across multiple domains and challenges.

[Figure 172]

[Figure 173]

- Figure 7: Nationality of AIN Survey Participants: Participants represent 17 Arab nations, with the highest contributions from Saudi Arabia (30%), followed by Egypt (25%), the UAE (13.3%), and Lebanon (13.3%).

[Figure 174]

- Figure 8: User Model Preferences. Participant preferences for the three models in the survey, with Model 1 (AIN (ours)) receiving 76% of the votes, GPT-4o 15%, and LLaVA 9%, demonstrating AIN’s strong performance.

Survey Results and Analysis. The survey results reveal that AIN outperforms human participants in several questions, demonstrating notable advantages in accuracy and adherence to the response formats. For instance, in Q1, although both Model 3 and human participants provided correct answers, they failed to comply with the required response format, a criterion that AIN successfully fulfilled. Similarly, in Q9, 24.9% of participants, and in Q6, 20.5%, were unable to respond appropriately, whereas AIN provides accurate responses. In Q2, AIN exhibits superior precision by correctly identifying the shape as a disc rather than a circle, outperforming 18.6% of human respondents. Furthermore, in Q3 and Q5, participants struggle to recognize small details in the images, highlighting AIN’s ability to detect subtle features. Notably, in Q10, AIN demonstrates its capacity to solve complex reasoning tasks by extracting value beyond the visible content of the image. This highlights its ability to handle abstract problem-solving, further showcasing its comparative advantage over other models. Detailed results and comparative analyses are illustrated in Figures 10, 11, and 12.

Dialect Preferences. Regarding language suitability, depicted in Figure 9, 74.3% of participants found MSA appropriate for the survey. An additional 11% were comfortable with MSA but expressed a preference for their local dialect. Only 4.3% strongly preferred their local dialect over MSA, while 10.5% reported challenges unrelated to language.

[Figure 175]

.ﺔﺑﺎﺘﻜﻟاو ةءاﺮﻘﻠﻟ ﻰﺤﺼﻔﻟا ﺔﯿﺑﺮﻌﻟا ﺔﻐﻠﻟا ﻞﻀﻓأ ﺎﻧأ ؛ﻢﮭﻔﻟا ﺔﻠﮭﺳو ﺔﺤﺿاو ﺖﻧﺎﻛ ،ﻻ

74.3%

No, it was clear and easy to understand; I prefer MSA for reading and writing.

.(ﺔﯿﻣﺎﻌﻟا) ﺔﺻﺎﺨﻟا ﺔﯿﺑﺮﻌﻟا ﻲﺘﺠﮭﻟ ﻞﻀﻓأ ﻲﻨﻨﻜﻟ ،ةﺪﯿﺟ ﺖﻧﺎﻛ ،ﻻ

11%

No, it was fine, but I prefer my own Arabic dialect.

###### .ىرﺧأ ﺔﯾوﻐﻟ لﻣاوﻌﺑ لﺑ ﺔﺟﮭﻠﻟﺎﺑ ﺔطﺑﺗرﻣ رﯾﻏ ﺔﺑوﻌﺻﻟا تﻧﺎﻛ ،مﻌﻧ

10.5%

Yes, the difficulty was unrelated to dialect but to other linguistic factors.)

ﺔﺻﺎﺧﻟا ﺔﯾﺑرﻌﻟا ﻲﺗﺟﮭﻟ لﺿﻓأ ﺎﻧأو ،ﻰﺣﺻﻔﻟا ﺔﯾﺑرﻌﻟا ﺔﻐﻠﻟا مادﺧﺗﺳا بﻌﺻﻟا نﻣ نﺎﻛ ،مﻌﻧ .(ﺔﯾﻣﺎﻌﻟا)

4.3%

Yes, it was challenging to use MSA, and I prefer my own Arabic dialect.

- Figure 9: User Preferences for MSA and Local Dialects: The majority (74.3%) preferred MSA for reading and writing. An additional 11% are comfortable with MSA but favored their local dialects, while 4.3% found MSA challenging and preferred using their dialect. A further 10.5% reported difficulties unrelated to linguistic aspects.

### 2 Data Inspection and Selection

Our data collection comprises publicly available MSA Arabic and English datasets, with a portion specifically curated for model training. Notably, 35% of the Arabic data is authentic. To achieve scalability and address specific domain requirements, translation was utilized to complement the existing data.

#### 2.1 Data Translation

In pursuit of optimal data translation, we select three models from the GPT-4v suite—specifically, GPT-4 [24], GPT-4o [3], and GPT-4o-mini [4]. A comprehensive experiment is conducted to evaluate the performance of these models across key criteria, including translation correctness, translation accuracy, and translation efficiency. For this evaluation, a variety of random samples of original English content are selected, and an English prompt was meticulously curated. To ensure accuracy and cultural adequacy, the corresponding Arabic prompts are crafted by a native Arabic speaker. Additionally, the same samples are manually translated by native Arabic speakers with high proficiency, serving as reference translations for benchmarking.

We evaluate the translation performance of GPT-4, GPT-4o, and GPT-4o-mini using both Arabic and English prompts under identical settings. Translation accuracy is assessed by native speakers who rated the outputs against manually translated references on a scale from 0 (fail) to 1 (accurate). The results indicate that GPT-4o and GPT-4omini perform closely in speed and accuracy across both prompt types, with the Arabic prompt outperforming in both metrics. Notably, GPT-4o-mini achieves the highest accuracy with the Arabic prompt. Contributors generally found that GPT-4o-mini successfully translates all terms, including brand names such as “Boeing”, whereas GPT-4o frequently fails to provide complete or accurate translations for these sentences.

Based on these findings, GPT-4o-mini is selected for data translation. All reading and results are recorded in Table 4 and Table 5 for Arabic and English prompts, respectively.

#### 2.2 Data Quality Verification and Filtering

High-performance models inherently require high-quality data [25]. Therefore, in addition to selecting appropriate data, we have implemented a multi-step data translation verification procedure alongside rigorous toxicity-free filtering (Figure 13).

Data Semantic Translation Verification. To identify the optimal model for translation verification, we design a set of sentences reflecting common linguistic challenges in Arabic, including punctuation alignment with English, direct and semantic translation accuracy, masculine/feminine tone differentiation, and handling of diacritics. This evaluation involves 21 sentence pairs, categorized as simple

[Figure 176]

(a) Q1: Domain: Agricultural Image Understanding / Plant diseases. Purpose: Ability to detect diseased plant areas and identify their color.

[Figure 177]

(b) Q2: Domain: Cultural-Specific Image Understanding / Food. Purpose: Ability to recognize food and precisely determine its shape.).

[Figure 178]

[Figure 179]

(c) Q3: Domain: Remote Sensing Image Understanding / Roads & Constructions. Purpose: Ability to identify specific constructions among similar ones.

(d) Q4: Domain: General VQA/ Binary Question. Purpose: Ability to identify tiny details in ambiguous scenes and answer binary questions.

- Figure 10: Survey Feedback - Part 1: Questions 1 to 4 explored diverse domains, including agriculture, cultural-specific topics (e.g., food), remote sensing, and general VQA. Tasks included agro-disease detection, food recognition, shape identification, specific construction detection, and recognizing tiny details in ambiguous scenes, using various question formats such as MCQ, binary, and open-ended short answers.

[Figure 180]

(a) Q5: Domain: General VQA / Traffic Signs. Purpose: Ability to spot traffic signs at a distance and in low resolution.

[Figure 181]

(b) Q6: Domain: OCR & Document Understanding. Purpose: Ability to discern Arabic characters and extract text from images.

[Figure 182]

[Figure 183]

(c) Q7: Domain: General VQA / Short Answer Question.

Purpose: Ability to pinpoint the required item among several items + provide a short answer as required.

(d) Q8: Domain: Medical Image Understanding / Diseases Diagnoses.

Purpose: Ability to diagnose organ health by reasoning its condition (normal or abnormal) for a specific disease.

- Figure 11: Survey Feedback - Part 2: Questions 5 to 8 focus on domains such as traffic sign recognition, OCR and document understanding, general VQA, and medical imaging. Tasks include identifying traffic signs, extracting correct text from images, pinpointing specific items among several options, and diagnosing organ conditions, using various formats such as MCQ and short answers.

[Figure 184]

[Figure 185]

(a) Q9: Domain: General VQA / Grounding and Celebrities. Purpose: Ability to determine a person’s identity in a specific location.

(b) Q10: Domain: Chart, Diagram & Table Understanding / Bar Charts. Purpose: Ability to extract values from charts, even when not explicitly shown.

- Figure 12: Survey Feedback - Part 3: Questions 9 and 10 focus on domains such as celebrities, grounding, and charts and diagrams. Tasks include identifying a celebrity in a specific location within the image based on the question, and extrapolating values from charts where the information is not explicitly written, using formats such as MCQ and short answers.

- Table 4: Comparison of translation performance for Arabic prompts across three GPT-4v models, evaluated on a variety of samples. Avg time/iteration*: average time per sample.

Arabic Prompt Model Time Avg time/ iteration* Accuracy GPT-4o [3] 1 min, 43 sec 04.16 sec 90% GPT-4 [24] 6 min, 39 sec 15.99 sec 85% GPT-4o-mini [4] 1 min, 28 sec 03.52 sec 92%

- Table 5: Comparison of translation performance for English prompts across three GPT-4v models, evaluated on variety of samples. Avg time/iteration*: average time per sample.

English Prompt Model Time Avg time/ iteration * Accuracy GPT-4o [3] 2 min, 01 sec 04.87 sec 88% GPT-4 [24] 8min, 21 sec 20.08 sec 50% GPT-4o-mini [4] 1 min, 52 sec 4.48 sec 87%

sentences (Table 6), complex sentences with tone ambiguity (Table 7), and affirmative clauses with question (Table 8).

The sentence pairs are processed using five multilingual models—M-BERT [26], Paraphrase-XLM-R [27], all-mpnet-base-v2 [28], LaBSE [29], and AraBERT [30]—to evaluate semantic similarity between English and Arabic translations. Cosine similarity is used as the scoring metric to quantify the alignment between the translations, providing a robust basis for selecting the most suitable model.

- Table 6: Translation quality check - Sentence 1: A simple English sentence with different settings including accurate direct translation, semantic translation, mismatched translation, punctuation, and diacritics.

Ref. Original Translation Translation Criteria

- 1.1 This is an example sentence Accurate direct translation
- 1.2 This is an example sentence Completely mismatched translation.
- 1.3 This is an example sentence Translation with semantic meaning.
- 1.4 This is an example sentence Accurate direct translation + diacritics.
- 1.5 This is an example sentence! Translation with semantic meaning + punctuation, no diacritics

- Table 7: Translation quality check - Sentence 2: The English sentence consists of a polite request with different settings including accurate direct translation, semantic translation, masculine/ Feminine tone, mismatched translation, punctuation, and diacritics.

Ref. Original Translation Translation Criteria

- 2.1 Please, sit down Accurate direct translation
- 2.2 Please sit down No punctuation, no diacritics.
- 2.3 Please sit down No punctuation, with diacritics
- 2.4 Please sit down No punctuation, with diacritics, feminine tone
- 2.5 Please sit down No punctuation, with diacritics, masculine tone
- 2.6 Please, sit down Completely mismatched translation.
- 2.7 Please, sit down Translation with semantic meaning.
- 2.8 Please, sit down Punctuation + diacritics/ masculine tone.
- 2.9 Please, sit down

Accurate direct translation + diacritics/ partial feminine tone.

- 2.10 Please, sit down Accurate direct translation + diacritics/ feminine tone.
- 2.11 Please, sit down. Accurate direct translation + punctuation + diacritics.
- 2.12 Please, sit down. Accurate direct translation + punctuation, no diacritics.
- 2.13 Please, sit down. With punctuation and only “hamzat al kaser”

Figure 14a presents a heatmap of the similarity scores of the models in evaluating translations semantic correctness, considering punctuation, tone, and diacritics. While high similarity scores indicate good performance, a robust model must also assign low scores to poor or irrelevant translations. To assess this, a second experiment tested model behavior on mismatched translations ( Figure 14b).

Due to the close performance observed between LaBSE [29] and Paraphrase-XLM-R [27] in initial evaluations, an additional experiment is conducted to further assess their capabilities. This experiment utilized 50 samples of high-quality translations and 50 samples of moderate-to-poor translations. LaBSE demonstrated superior consistency, providing higher similarity scores for accurate translations and relatively lower scores for poor translations compared to Paraphrase-XLM-R. This reliability in

- Table 8: Translation quality check - Sentence 3: The English sentence consists of an affirmative clause followed by a question of accurate direct translation in different settings including semantic translation and punctuation.

Ref. Original Translation Translation Criteria 3.1

It is raining today should we stay at home

No punctuation.

It is raining today. Should we stay at home

Semantic meaning + punctuation.

3.2

It is raining today. Should we stay at home?

3.3

With punctuation.

distinguishing translation quality (Figures 15a and 15b) led to the selection of LaBSE for the full dataset. Translations scoring below 80% similarity were excluded, accounting for less than 2% of the data.

Translation Semantic Similarity Verification

EN

EN Embeddings

[Figure 186]

[Figure 187]

Yes

Data ready to next step

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

Satisfied?

[Figure 192]

Semantic

[Figure 193]

ع

[Figure 194]

LaBSE

Score 80%

No

[Figure 195]

AR

Isolate out of the data pool

GPT-4o-mini

Cosine Similarity

Original English Textual Data

Translated Arabic Textual Data

AR Embeddings

AR

Visual Data Toxicity Filtering

[Figure 196]

Translation Quality, Correlation, Overlap Verification

[Figure 197]

EN (Reverse Translation)

(Original"Reference")

[Figure 198]

GPT-4o-mini

tonextstep

re-translation Dataready

[Figure 199]

Yes

Data ready for training

[Figure 200]

[Figure 201]

BLEU Score

60%

BLEU

[Figure 202]

Visual Data

EN

Yes

AllSatisfied?

[Figure 203]

[Figure 204]

No

METEOR Score

[Figure 205]

[Figure 206]

Isolate out of GPT-4o the data pool

METEOR

80%

Considerdata

Safe?

No

[Figure 207]

[Figure 208]

Precision Recall F1-Score

[Figure 209]

ROUGE 80%

LlaVAGuard Safety Policy

- Figure 13: Data verification and filtering pipeline for textual and visual data. Textual data underwent semantic similarity checks using LaBSE [29] (80% threshold) and quality evaluation using BLEU [31] (60% threshold), METEOR [32] (80% threshold), and ROUGE [33] (80% threshold). Visual data was screened for toxicity using LLavaGuard [34] policies with GPT-4o [3], discarding unsafe images to ensure quality and safety.

Data Quality Verification. To ensure a comprehensive quality verification process, particularly given the use of generative AI for translation, additional checks are necessary. These include assessing the quality of machine-generated text, its correlation with the original content, and the degree of overlap between the generated and original text. To achieve this, we employ three specialized evaluation metrics: BLEU (2-gram and 4-gram) [31] for text quality, METEOR [32] for translation correlation, and ROUGE (unigram and ROUGE-L) [33] for overlap measurement.

Our data quality verification experiment involves analyzing randomly selected 50 translated samples that are translated back to English using GPT-4o-mini [4] for comparison against the original English text (reference data).

The evaluation metrics demonstrated strong performance across multiple dimensions: BLEU scores [31] of 71.11% (2-gram) and 60.20% (4-gram) indicated high local coherence and good fluency in the translated text. The METEOR score [32] of 86.10% suggested high-quality translation with effective handling of both exact matches and linguistic variations. ROUGE metrics [33] were particularly strong, with unigram scores showing 87.80% precision and 87.30% recall, indicating excellent

[Figure 210]

Direct translation, exact punctuation

no diacritics

Arabic translation with feminine tone

Different punctuation marks between English and Arabic

Translation with semantic meaning, with diacritics (not showing in the chart), and different punctuation.

Translation with diacritics (not showing in the chart)

###### (a) Similarity scores for different settings for correct translation. The higher the better.

[Figure 211]

Complete mismatched translaation.

(b) Similarity scores for different settings for incorrect translation. The lower the worse.

- Figure 14: Similarity scores for diverse settings, including direct correct translation, incorrect translation, semantic translation, diacritics, and punctuation.

word-level accuracy and comprehensive content capture. Similarly, ROUGE-L scores [33] (precision: 86.20%, recall: 85.90%, F1: 85.80%) confirmed strong structural similarity between the translated and reference texts, demonstrating that the essential meaning and structure are well-preserved throughout the translation process (Table 9).

With the data translation quality checks completed, a final step of visual toxicity inspection is required to ensure the data is ready for model training.

Table 9: Data Quality Verification and Evaluation Metrics

Metric Scores Precision Recall F1-score BLEU (2-gram) [31] 71.11% BLEU (4-gram) [31] 60.20% METEOR [32] 86.10% ROUGE (unigram) [33] 87.80% 87.30% 87.30% ROUGE-L [33] 86.20% 85.90% 85.80%

Toxicity Filtering: To ensure model safety in vision, toxicity inspection is a critical component of our evaluation process. We utilize LLavaGuard’s safety taxonomy [34], a well-curated prompt specifically designed to verify visual data against predefined safety criteria, in combination with

[Figure 212]

- (a) Comparison of LaBSE and Paraphrase-XLM-R: Evaluating 50 high-quality translated samples.

[Figure 213]

- (b) Comparison of LaBSE and Paraphrase-XLM-R: Assessing 50 low-quality translated samples.

Figure 15: Comparison of LaBSE and Paraphrase-XLM-R to identify the optimal model.

[Figure 214]

Figure 16: Visual Data Toxicity Filtering. Using GPT-4o [3] and LLavaGuard [34] policies, about 96% of the data is classified as safe, while the remainder was deemed unsafe. The unsafe data was distributed across four categories: “Weapon, or Substance Abuse” (3.25%), “Hate, Humiliation, Harassment” (0.55%), “Animal Cruelty” (1.09%), and “Violence, Harm, or Cruelty” (0.55%).

GPT-4o for the inspection process. The dataset undergoes a comprehensive assessment to ensure compliance with safety policies. The evaluation covers key categories, including “Hate, Humiliation, or Harassment”; “Violence, Harm, or Cruelty”; “Sexual Content”; “Nudity”; “Weapons or Substance Abuse”; “Self-Harm”; “Animal Cruelty”; and “Disasters or Emergencies”.

The results reveal that 95.63% of the data is deemed safe, while 4.37% is classified as unsafe. The unsafe data is distributed across four main categories: “Weapons or Substance Abuse”, “Hate, Humiliation, or Harassment”, “Animal Cruelty”, and “Violence, Harm, or Cruelty”, as illustrated in Figure 16. This rigorous evaluation ensured that our data met safety standards, further preparing it for downstream applications.

Following the completion of the data verification steps and toxicity filtering, our dataset, comprising 3.6 million safe and curated entries, is prepared for model training.

### 3 Experiments

The AIN model is trained on 8 GPU nodes, each equipped with 8 NVIDIA A100 GPU cards, each with 80 GB memory. The GPUs within each node are interconnected using 8 NVLink links, ensuring high bandwidth and low latency. To facilitate efficient cross-node communication, each node is equipped with dual-port 200 Gbps (4×HDR) InfiniBand connections, achieving an aggregate interconnect bandwidth of 800 Gbps. This robust infrastructure was crucial in handling the computational and memory-intensive large-scale LMM training.

Our approach leverages the Qwen2-VL-7B [2] model as the base, which we fine-tuned on our English-Arabic bilingual dataset. The dataset comprises 3.6 million high-quality text samples curated from diverse sources, ensuring comprehensive coverage of linguistic, cultural, and domain-specific nuances. For fine-tuning, we employed a full-parameter fine-tuning strategy, conducting training for one epoch. This approach allowed us to adapt the pre-trained model to better capture the semantic properties of both Arabic and English languages.

To optimize training efficiency and scalability, we employ the flash-attention mechanism, which significantly reduces memory overhead during attention computation. Additionally, we adhere to the hyper-parameter configurations established by LLaMA-Factory [35], including an optimized learning rate schedule, batch size, and weight decay strategies tailored for large-scale transformer-based models.

### 4 Conclusion

This work introduces AIN, an Arabic-inclusive LMM, as a step toward bridging the gap in AI solutions for Arabic, a low-resource yet globally significant language. AIN is trained on a large-scale bilingual dataset with 35% of authentic Arabic data. Through rigorous evaluation, we show that AIN achieves state-of-the-art performance across a wide range of tasks, including VQA, OCR and document understanding, cultural understanding, and domain-specific applications such as medical imaging and remote sensing, surpassing even bigger and more sophisticated models. AIN further demonstrates superior accuracy, contextual understanding, and human-like reasoning in MSA, as validated by extensive evaluations and human judgments. By integrating advanced data curation, robust translation pipelines, and stringent quality control, AIN provides a new state-of-the-art multimodal AI model tailored to Arabic speakers.

### References

- [1] Sara Ghaboura, Ahmed Heakl, Omkar Thawakar, Ali Alharthi, Ines Riahi, Abduljalil Saif, Jorma Laaksonen, Fahad S Khan, Salman Khan, and Rao M Anwer. Camel-bench: A comprehensive arabic lmm benchmark. arXiv preprint arXiv:2410.18976, 2024.
- [2] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen-vl: A frontier large vision-language model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.
- [3] OpenAI. Gpt-4o model. https://openai.com, 2024. Accessed: 2024.
- [4] OpenAI. Gpt-4o-mini model. https://openai.com, 2024. Accessed: 2024-10-14.
- [5] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.
- [6] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024.
- [7] Xiang Yue, Yueqi Song, Akari Asai, Seungone Kim, Jean de Dieu Nyandwi, Simran Khanuja, Anjali Kantharuban, Lintang Sutawika, Sathyanarayanan Ramamoorthy, and Graham Neubig. Pangea: A fully open multilingual multimodal llm for 39 languages. arXiv preprint arXiv:2410.16153, 2024.
- [8] Google AI. Gemini: A family of highly capable multimodal models, 2023.
- [9] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. arXiv preprint arXiv:2312.14238, 2023.
- [10] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.
- [11] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [12] Nahid Alam, Karthik Reddy Kanjula, Surya Guthikonda, Timothy Chung, Bala Krishna S Vegesna, Abhipsha Das, Anthony Susevski, Ryan Sze-Yin Chan, SM Uddin, Shayekh Bin Islam, et al. Maya: An instruction finetuned multilingual multimodal model. arXiv preprint arXiv:2412.07112, 2024.
- [13] Fajri Koto, Haonan Li, Sara Shatnawi, Jad Doughman, Abdelrahman Boda Sadallah, Aisha Alraeesi, Khalid Almubarak, Zaid Alyafeai, Neha Sengupta, Shady Shehata, et al. Arabicmmlu: Assessing massive multitask language understanding in arabic. arXiv preprint arXiv:2402.12840, 2024.
- [14] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European Conference on Computer Vision, pages 216–233. Springer, 2025.
- [15] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.
- [16] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.

- [17] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023.
- [18] Bohao Li, Yuying Ge, Yixiao Ge, Guangzhi Wang, Rui Wang, Ruimao Zhang, and Ying Shan. Seed-bench: Benchmarking multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.
- [19] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.
- [20] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022.
- [21] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022.
- [22] Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part IV 14, pages 235–251. Springer, 2016.
- [23] Kaining Ying, Fanqing Meng, Jin Wang, Zhiqian Li, Han Lin, Yue Yang, Hao Zhang, Wenbo Zhang, Yuqi Lin, Shuo Liu, jiayi lei, Quanfeng Lu, Peng Gao, Runjian Chen, Peng Xu, Renrui Zhang, Haozhe Zhang, Yali Wang, Yu Qiao, Ping Luo, Kaipeng Zhang, and Wenqi Shao. MMT-bench: A comprehensive multimodal benchmark for evaluating large vision-language models towards multitask AGI. In Proceedings of the International Conference on Machine Learning (ICML), 2024.
- [24] OpenAI. Gpt-4 model. https://openai.com, 2024. Accessed: 2024.
- [25] Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. arXiv preprint arXiv:2409.17146, 2024.
- [26] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of naacL-HLT, volume 1. Minneapolis, Minnesota, 2019.
- [27] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bertnetworks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 11 2019.
- [28] Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. Mpnet: Masked and permuted pre-training for language understanding. In Advances in Neural Information Processing Systems, 2020.
- [29] Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, and Wei Wang. Languageagnostic bert sentence embedding. arXiv preprint arXiv:2007.01852, 2020.
- [30] Wissam Antoun, Fady Baly, and Hazem Hajj. Arabert: Transformer-based model for arabic language understanding. arXiv preprint arXiv:2003.00104, 2020.
- [31] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318, 2002.
- [32] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In Proceedings of the acl workshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization, pages 65–72, 2005.

- [33] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81, 2004.
- [34] Lukas Helff, Felix Friedrich, Manuel Brack, Kristian Kersting, and Patrick Schramowski. Llavaguard: Vlm-based safeguards for vision dataset curation and safety assessment. arXiv preprint arXiv:2406.05113, 2024.
- [35] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, Zheyan Luo, Zhangchi Feng, and Yongqiang Ma. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), Bangkok, Thailand, 2024. Association for Computational Linguistics.

