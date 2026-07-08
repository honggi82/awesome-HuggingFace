## arXiv:2406.06474v1[cs.AI]10Jun2024

### TOWARDS A PERSONAL HEALTH LARGE LANGUAGE MODEL

Justin Cosentino*, Anastasiya Belyaeva*§, Xin Liu*§, Nicholas A. Furlotte*, Zhun Yang‡, Chace Lee‡, Erik Schenck‡, Yojan Patel‡, Jian Cui‡, Logan Douglas Schneider‡, Robby Bryant , Ryan G. Gomes , Allen Jiang , Roy Lee , Yun Liu , Javier Perez , Jameson K. Rogers , Cathy Speed , Shyam Tailor , Megan Walker , Jeffrey Yu , Tim Althoff , Conor Heneghan , John Hernandez , Mark Malhotra , Leor Stern , Yossi Matias , Greg S. Corrado , Shwetak Patel , Shravya Shetty , Jiening Zhan , Shruthi Prabhakara , Daniel McDuff†§, and Cory Y. McLean†§

Google LLC *Co-first author. ‡Core contributor. †Co-last author. §Corresponding authors: {belyaeva,xliucs,dmcduff,cym}@google.com.

June 11, 2024

##### ABSTRACT

Large language models (LLMs) can retrieve, reason over, and make inferences about a wide range of information. In health, most LLM efforts to date have focused on clinical tasks. However, mobile and wearable devices, which are rarely integrated into clinical tasks, provide a rich, continuous, and longitudinal source of data relevant for personal health monitoring. Here we present a new model, Personal Health Large Language Model (PH-LLM), a version of Gemini fine-tuned for text understanding and reasoning over numerical time-series personal health data for applications in sleep and fitness. To systematically evaluate PH-LLM, we created and curated three novel benchmark datasets that test 1) production of personalized insights and recommendations from measured sleep patterns, physical activity, and physiological responses, 2) expert domain knowledge, and 3) prediction of self-reported sleep quality outcomes. For the insights and recommendations tasks we created 857 case studies in sleep and fitness. These case studies, designed in collaboration with domain experts, represent real-world scenarios and highlight the model’s capabilities in understanding and coaching. Through comprehensive human and automatic evaluation of domain-specific rubrics, we observed that both Gemini Ultra 1.0 and PH-LLM are not statistically different from expert performance in fitness and, while experts remain superior for sleep, fine-tuning PH-LLM provided significant improvements in using relevant domain knowledge and personalizing information for sleep insights. To further assess expert domain knowledge, we evaluated PH-LLM performance on multiple choice question examinations in sleep medicine and fitness. PH-LLM achieved 79% on sleep (N=629 questions) and 88% on fitness (N=99 questions), both of which exceed average scores from a sample of human experts as well as benchmarks for receiving continuing credit in those domains. To enable PH-LLM to predict self-reported assessments of sleep quality, we trained the model to predict self-reported sleep disruption and sleep impairment outcomes from textual and multimodal encoding representations of wearable sensor data. We demonstrate that multimodal encoding is both necessary and sufficient to match performance of a suite of discriminative models to predict these outcomes. Although further development and evaluation are necessary in the safety-critical personal health domain, these results demonstrate both the broad knowledge base and capabilities of Gemini models and the benefit of contextualizing physiological data for personal health applications as done with PH-LLM.

##### 1 Introduction

Large language models (LLMs) are versatile tools for generating language and have shown strong performance across a range of diverse domains. LLMs have achieved passing grades on the US legal bar exam [28] and second year medical school exams [43,48,53]. In medicine in particular, natural language as an interface has shown potential to influence clinical practice [37], education, and research [39]. When enriched with healthcare-specific data, LLMs attain impressive performance in medical question-answering [48,53], nuanced analysis of electronic health records [61], differential

diagnosis from medical images [56], assessment of psychiatric functioning based on standardized assessments [20], and the delivery of psychological interventions [33,50,51]. High levels of performance on these tasks shows that LLMs have been able to effectively capture signal from “clinical data”, collected within a clinical setting.

Due to their sporadic nature, conventional clinical visits often fail to capture key aspects of human health and well-being that are measurable with wearable devices including sleep, physical activity, stress, and cardiometabolic health measured through physiological response and behavior. These continuous, longitudinal measures have significant advantages for health monitoring in that they are passively and continuously acquired, and provide direct physiological and behavioral signals. However, they have neither been deeply integrated into clinical practice nor incorporated into standard datasets used for medical question-answering [23,27], despite statistics on adverse health outcomes, morbidity, and DisabilityAdjusted Life Years that underscore the profound impact these factors have on overall health [3,9,18,44,58]. This limited adoption is likely because these data are typically captured without context, are computationally demanding to store and analyze, and can be difficult to interpret. As a result, general foundation LLMs or even medically-tuned LLMs may lack the ability to use these data effectively to reason about and recommend interventions based on personalized individual health behaviors.

Mobile devices, including smart wearables and smartphones, have become instrumental tools for monitoring personal health metrics and gathering longitudinal data that cannot be obtained in traditional clinical settings [54]. Unlike structured clinical data, personal health data is heterogeneous across data types, sources, and timescales [36], ranging from continuous streams of biometric data from wearables, such as sleep patterns or heart rate, to sporadic and qualitative inputs like exercise logs, dietary logs, mood journals, and even social media activity.

In this paper, we introduce Personal Health Large Language Model (PH-LLM), a version of Gemini fine-tuned to generate both insights about and recommendations to improve personal health behaviors related to sleep and fitness patterns. We evaluate the performance of PH-LLM across three tasks: coaching recommendations, multiple choice exams assessing expert knowledge, and prediction of subjective patient-reported outcomes (PROs). The coaching recommendations tasks are tailored to two verticals of high personal health interest: sleep and fitness. The sleep tasks leverage individual’s sleep metrics to derive insights, possible etiological factors, and provide personalized recommendations to improve sleep quality. The fitness tasks integrate information from training load, sleep, health metrics, and subjective feedback to provide personalized recommendations for the intensity of a physical activity that day. For the coaching recommendations tasks, we create the first personal health case study dataset to benchmark LLM performance on reasoning and understanding of personal health behaviors. This dataset consists of long-form questions grounded in summarized personal health behavior data, vertical-specific evaluation rubrics, and expert human responses for 857 case studies across sleep and fitness. Through rigorous human and automatic evaluation, we demonstrate that Gemini Ultra 1.0 approaches expert performance in fitness while fine-tuning PH-LLM enables it reduce the gap with experts in sleep coaching experiences, explore the breadth of personal health knowledge encoded within Gemini models, and apply multimodal capabilities to prediction of PROs in sleep (Figure 1). Our key contributions are summarized as follows:

- • PH-LLM: We introduce a new model fine-tuned from Gemini for applications in personal health, capable of performing interpretation of time-series sensor data from wearables (i.e., Fitbit and Pixel Watch) for analysis and recommendations in sleep and fitness.
- • Long-form case studies from domain experts: We create the first dataset of detailed personal health case studies in sleep and fitness curated by multiple experts in the associated domains. The dataset contains individual wearable sensor data and corresponding long-form insights and recommendations. We present rubrics for evaluation of long-form responses that span domain knowledge, use of user data, personalization, and potential for harm, and provide insights on training experts for accurate evaluation.
- • Benchmark and contextualize personal health question-answering: We curate a set of validated domain-specific multiple choice examination questions on sleep and fitness, establish strong benchmarks based on continuing medical education requirements, and provide context for the scores through a set of human experts who completed a representative set of exam questions.
- • Multimodal sensor interpretation of self-reported outcomes: We successfully integrate longitudinal time-series sensor features to interpret a user’s subjective experience. To do so, we evaluate the capabilities of PH-LLM in predicting sleep disturbance and impairment PROs (acquired through validated survey instruments) from passive sensor readouts and show that accurate model performance requires native multimodal data integration.

###### (A) Coaching Recommendations

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

Make sure your bedroom is cool and dark. Continue to avoid naps and keep a consistent sleep schedule.

[Figure 5]

Professional Examinations

[Figure 6]

| | |
|---|---|
| | |

Patient-Reported Outcomes

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

Human Experts Personal Health LLM (PH-LLM)

###### (B)

*

100%

- 1
- 2
- 3
- 4
- 5

| | | |
|---|---|---|
| | |Accuracy|
| | | |
| | | |

|*<br><br>| | | |
|---|---|---|---|
|Approx. Grade f<br><br>|or Credit| | |
| | | | |
| | | | |
| | | | |

AreaUnderROCCurve

60%

Recommendation

&InsightsQuality

80%

Accuracy

Naive Performance

60%

40%

40%

20%

20%

0%

0%

Sleep Fitness

Sleep Fitness

Sleep

Coaching Recommendations Patient-Reported Outcomes

Professional Examinations

Figure 1: PH-LLM: A Personal Health Large Language Model. (A) We present PH-LLM, a version of Gemini fine-tuned for personal health and wellness. We evaluated PH-LLM on three aspects of personal health: generating personalized insights and recommendations for user goals in the domains of sleep and fitness, assessing levels of expert knowledge from certification examination style multiple choice questions, and predicting patient-reported outcomes in sleep quality from detailed sensor information. (B) Performance of PH-LLM contextualized with expert human responses. Error bars represent 95% confidence intervals. “∗” indicates a statistically significant difference between two response types. “Naive Performance” is that achieved by a random classifier. Human expert performance is not available for patient-reported outcome prediction from sensor features as this is not commonly performed, and no fitness-related outcomes were measured in the study assessing patient-reported outcomes [36].

##### 2 Personal Health Dataset Creation

Owing to the absence of clearly defined language and multimodal datasets in the domain of personal health, we created datasets and associated tasks to evaluate different capabilities of PH-LLM. These datasets include case studies about real-world coaching recommendations, professional examinations that test domain knowledge about sleep medicine and fitness, and patient-reported outcomes about sleep.

###### 2.1 Coaching recommendations

Many real-world applications of LLMs for personal health require realistic long-form text generation, which is challenging to evaluate automatically. As previously observed in the medical domain, strong performance on questionanswering tasks does not necessarily transfer to the complexity of real-world tasks [17]. To address the absence of rich long-form tasks for personal health data, in conjunction with domain experts and overseen by clinical leads we created detailed case studies that span two key personal health domains: sleep and fitness. Each case study was designed to interpret a range of physiological sensor information toward deriving insights, potential causes, or recommendations

for future behaviors, and was sampled from high-volume anonymized production data from individuals who provided consent for research purposes.

The sleep case studies aimed to enhance understanding of sleep patterns, identify causes of irregular sleep, and offer actionable recommendations based on these findings. Each case study incorporated wearable sensor data for up to 29 days, demographic information, and an expert analysis (Figure 2). This comprehensive approach both facilitates a deeper understanding of health-related behaviors and also guides the development of personalized interventions to improve individual outcomes.

The fitness case studies were designed to provide a comprehensive analysis of an individual’s training load, sleep patterns, and health metrics, and were similarly based on wearable sensor data over 30 days, demographic information, and expert analysis (Figure 3). Based on these inputs, the goal of the fitness-related task was to synthesize the suite of metrics into a data-driven assessment of the extent to which the individual is prepared for physical activity today and provide associated recommendations.

Each set of domain experts consisted of “primary” and “secondary” contributors to case study response creation and evaluation. This categorization was based on an expert’s general availability to contribute to the research project on a weekly basis throughout its duration; “primary” contributors had more involvement and higher volumes of case study response creation and evaluation than “secondary” contributors. The grouping was primarily used for research project operations planning and scheduling. The level of domain expertise was similar across the two groups. Each vertical also included a clinical lead with extensive background in sleep medicine for the sleep vertical and sport and exercise medicine for the fitness vertical. The clinical lead oversaw case study development and provided feedback and quality control to the set of domain experts.

Sleep Case Study Creation. In the development of sleep case studies, we recruited six domain experts in sleep medicine to craft guidance in the second person narrative, fostering a direct and personalized dialogue with the user. The six sleep experts all possessed advanced degrees (M.D., D.O., or Psy.D.) in sleep medicine and professional sleep medicine work experience ranging from 4 to 46 years. All experts were trained to read and interpret wearable data and map outputs to their corresponding sleep medicine literature counterparts. Experts were instructed to use best practices in goal-setting, emphasizing the creation of recommendations that are Specific, Measurable, Achievable, Relevant, and Time-bound (SMART) [15]. The data was sampled to achieve a representative group across age and gender (see

- Appendix A.1.1 for details, Figure A.2).

As illustrated in Figure 2, demographics information (age and gender), daily sleep metrics (e.g., bedtimes, wake times, and sleep stage durations), and aggregated sleep statistics (e.g., average bedtime) were selected collaboratively with the experts (Appendix A.1.1). These data were presented to the experts in both graphical and tabular formats for ease of analysis (Figure 2A-C, Table A.4). Finally, the experts were tasked with composing responses across the following sections, aimed at analyzing the data with the objective of enhancing the sleep quality of the individual under consideration.

Insights: Implicitly this section was aimed at answering the question of “What are some sleep-related insights based on my data?” The sleep medicine expert examined the data and provided an interpretation of whether a data point might represent an atypical sleep pattern. The experts were asked to systematically review each case to provide a holistic assessment of the user’s sleep patterns. To do so, Fitbit sleep metrics were assessed according to the validated RU-SATED framework (Routine, Sleep Quality, Alertness, Timing, Efficiency, and Duration) to generate sleep insights [8].

Etiology: Implicitly this section answered the question of “What are the possible underlying causes that could explain the observed data?” The experts generally considered the contribution of circadian rhythm, homeostatic drive, psychophysiologic hyperarousal, and extrinsic factors and indicated their likelihood.

Recommendations: This section was generally designed to answer the question of “What can I do to improve my sleep?” The experts were asked to provide personalized recommendations to the individual that can help them improve their sleep by addressing potential causes identified in the etiology section. The experts were instructed to utilize best practices in goal-setting using the SMART framework.

Fitness Case Study Creation. To construct fitness case studies (Figure 3), we recruited seven domain experts in fitness to analyze an individual’s quantitative fitness data. The seven fitness experts all possessed advanced degrees (M.S., M.A., M.Ed., or D.A.T.) related to the athletic training field and professional athletic training work experience ranging from 4 to 25 years. The experts were directed to formulate insights, assessments, and recommendations in the second person narrative. The data for fitness case studies were sampled to produce a variety of different fitness assessments (see

- Appendix A.1.2 for details). The quantitative fitness data included a comprehensive array of metrics encompassing daily cardiovascular training load, sleep patterns, and health metrics spanning the preceding 30-day period (see Appendix

[Figure 13]

[Figure 14]

[Figure 16]

###### A PREPRINT - JUNE 11, 2024

[Figure 17]

Nodata

- (A) Bedtimes and Wake Times

|50 Year-Old Male Goal: I’d like to improve my sleep.<br><br>|
|---|

Example Sleep Case Study with Expert Response

65-70 year-old male Goal: I’d like to improve my sleep.

|Metric Average Percentile Min Max 5% percentile 95% percentile|
|---|
|Bedtime (hh:mm) 01:01 94% 23:29 03:32 20:35 01:03 Mid-sleep point (hh:mm) 04:09 91% 02:56 06:36 00:36 04:44 Sleep duration (hh:mm) 05:28 7% 03:52 07:57 05:19 07:49<br><br>… … … … … … …<br><br>Restlessness score (0-1) 0.116 78% 0.060 0.184 0.054 0.147 Sleep score (0-100) 69.52 4% 55.45 84.65 69.98 84.82|

Insights:

- ● The midsleep point standard deviation is 44 minutes, which indicates a regular sleep schedule.
- ● The users average sleep duration is 5 hours 28 minutes, which is below a healthy range.
- ● There were no naps during the recorded period. Etiology: Circadian Rhythm:
- ● The user has a regular sleep schedule, as noted by the midsleep point standard deviation. This means they have a regular circadian rhythm. Homeostatic Drive:
- ● The user has a low sleep duration and does not nap. Most of the recorded sleep durations are less than 6 hours. Recommendations:
- ● You have a regular sleep schedule, and this is a good aspect of your sleep.
- ● However, your sleep duration is often less than 6 hours per night, which is not healthy.
- ● A better sleep schedule would be from 12:00 AM to 7:30 AM on all days. This will allow for sufficient sleep opportunity.
- ● You can achieve this by moving your sleep schedule by 15 minutes every few days until you reach your goal.
- ● Continue to avoid napping. If you need to nap, try to keep it to less than 30 minutes. The goal would be to add this time to your primary sleep period.

- (B) Sleep Stage Durations

- (C) Aggregated Sleep Statistics

- (D) Abridged Expert-written Sleep Insights, Etiology & Recommendations

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

|Metric Value Percentile Min Max 5% percentile 95% percentile|
|---|
|Average Bedtime (hh:mm) 22:42 54% 12:41 01:57 20:47 00:44 Average Wake Time (hh:mm) 5:48 24% 17:27 08:25 04:30 08:33<br><br>Mid-sleep Point (hh:mm) 2:10 72% 15:01 04:18 00:27 04:24 … … … … … … … Average restlessness score 0.1 75% 0.05 0.23 0.053 0.13 Average sleep score 71 5% 51 87 71 85|

TrainingLoadSleepCardio

Figure 2: Sleep case study example: wearable sensor data used as input and corresponding expert analysis and recommendations for improving sleep quality. The experts considered individual’s demographics and wearable sensor data for up to 29 days including daily metrics of (A) bedtimes and wake times and (B) time spent in various sleep stages and awake. For all daily metrics considered, see Table A.4. The experts also analyzed (C) aggregated statistics of various sleep metrics. For a full list of aggregated statistics, see Table A.5. The experts composed responses based on the data including (D) insights about individual’s sleep, potential etiology, and recommendations for improving sleep quality.

[Figure 22]

[Figure 23]

5

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

Sleep Time Distribution

GraphicalDisplayofNumericalMetrics

Sleep Stage Distribution

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

Sleep Case Study

|User Profile: Age: 75 years Goal: I’d like to improve my sleep<br><br>Gender: Female<br><br>|
|---|

UserContextandNumericalMetricsin

|Sleep Metric User Value Percentile Avg. Bedtime 1.0 56.0<br><br>Avg. Free 9.0 56.0 Med. Bedtime Work 56.0 56.0<br><br>Avg. Wake Time 9.0 56.0 Avg. Wake Time Free 56.0 9.0 Avg. Midsleep Point 9.0 56.0 Sleep Duration Std. 56.0 9.0<br><br>Sleep Efficiency 9.0 56.0 Deep Sleep 56.0 9.0<br><br>REM Sleep 9.0 56.0 Restlessness 56.0 56.0|
|---|

|Insights<br><br>1. Their typical wake after sleep of 01:04 is longer than the general health range.<br>2. Their typical sleep duration std of 01:16 is larger than the general health range.<br>3. Their typical sleep midpoint std of 01:16:47 is higher than the general health range.<br>4. Their typical sleep midpoint of 03:40:30 is later than the general health range. Etiology<br><br><br>1. Homeostatic drive is possibly contributing.<br>2. circadian rhythm is possibly contributing.<br>3. Psychophysiologic hyperarousal is not clearly contributing.<br>|
|---|

|User Profile Age: 75 years<br><br>Gender: Female<br><br>|
|---|

TableForm ExpertCraftedResponse

|Goal I’d like to improve my sleep.<br><br>|
|---|

|Context<br><br>Avg. Bedtime<br><br>Avg. Wake Time Avg. Sleep Midpoint Avg. Sleep Duration<br><br>Time to Sleep<br><br>Value Percentile Min Max|
|---|

|Daily Sleep Data<br><br>|
|---|

|Key Insights:<br><br>- Sleep Quality:<br><br>* Overall good sleep efficiency of 86%, above the recommended range of 80-85%.<br><br>* Adequate deep sleep (1 hour 1 minute), which is essential for physical recovery.<br><br>- Sleep Schedule:<br><br>* Consistent sleep schedule, with average bedtime around midnight (00:07) and wake time around 8:11 AM.<br><br>* Early mid sleep point (04:09), suggesting an early sleep-wake cycle.<br><br>- Sleep Duration:<br><br>* Average sleep duration of 6 hours 54 minutes, which is slightly below the recommended range of 7-9 hours for older adults. Waking Alertness:<br><br>* No naps reported, indicating good daytime alertness. Recommendations:<br><br>* Consider extending sleep duration to meet the recommended range for older screadults. This can be achieved by adjusting the bedtime or wake time slightly.<br><br>* Monitor REM sleep and consult with a healthcare professional if there are concerns about low levels.<br><br>* Maintain the consistent sleep schedule to support circadian rhythm regulation.<br><br><br>|
|---|

|Key Insights: Sleep Quality:<br><br>* Overall good sleep efficiency of   %, above the recommended range of  0-  %.<br><br>* Adequate deep sleep (1 hour 1 minute), which is essential for physical recovery.<br><br>* Relatively low REM sleep (   minutes), which may indicate reduced dream recall or emotional processing during sleep. Sleep Schedule:<br><br>* Consistent sleep schedule, with average bedtime around midnight (00:0 ) and wake time around  :11 AM.<br><br>* Early mid sleep point (0 :0 ), suggesting an early sleep-wake cycle. Sleep Duration:<br><br>* Average sleep duration of hours minutes, which is slightly below the recommended range of  -  hours for older adults. Waking Alertness:<br><br>* No naps reported, indicating good daytime alertness. Recommendations:<br><br>* Consider extending sleep duration to meet the recommended range for older screadults. This can be achieved by adjusting the bedtime or wake time slightly.<br><br>* Monitor REM sleep and consult with a healthcare professional if there are concerns about low levels.<br><br>* Maintain the consistent sleep schedule to support circadian rhythm regulation.<br><br>* Continue to prioritize good sleep hygiene practices, such as avoiding caffeine and alcohol before bed and creating a relaxing bedtime routine.<br><br>* If daytime sleepiness or fatigue becomes an issue, consider exploring the possibility of underlying sleep disorders with a healthcare professional.<br><br><br>|
|---|

Average wake time is 07:17 Average wake time is in the 76th percentile Earliest wake time is 05:41 Latest wake time is 09:41 Bottom 5th percentile of similar users' average wake times is 04:14 Top 95th percentile of similar users' average wake times is 08:31

Average midsleep point is 04:09 Average midsleep point is in the 91st percentile Earliest midsleep point is 02:56 Latest midsleep point is 06:36 Bottom 5th percentile of similar users' average midsleep points is 00:36 Top 95th percentile of similar users' average midsleep points is 04:44

Average sleep duration is 05:28 Average sleep duration is in the 7th percentile Shortest sleep duration is 03:52 Longest sleep duration is 07:27 Bottom 5th percentile of similar users' average sleep durations is 05:19 Top 95th percentile of similar users' average sleep durations is 07:49

Average restlessness score is 0.116 Average restlessness score is in the 78th percentile Minimum restlessness score is 0.060 Maximum restlessness score is 0.184 Bottom 5th percentile of similar users' average restlessness scores is 0.054 Top 95th percentile of similar users' average restlessness scores is 0.147

Average sleep score is 69.52 Average sleep score is in the 4th percentile Minimum sleep score is 55.45 Maximum sleep score is 84.65 Bottom 5th percentile of similar users' average sleep scores is 69.98 Top 95th percentile of similar users' average sleep scores is 84.82

|50 Year-Old Male Goal: I’d like to improve my sleep.<br><br>|
|---|

###### A PREPRINT - JUNE 11, 2024

|Example Fitness Case Study with Expert Response<br><br>40-45 year-old female Height: 1.65-1.70m, Weight: 60-65kg, BMI: 24.8 Goal: I’d like a recovery plan.|
|---|
|[Figure 31]<br><br>[Figure 32]<br><br>(A) Training Load Metrics<br><br>[Figure 33]<br><br>[Figure 34]|
|[Figure 35]<br><br>(B) Sleep Metrics|
|[Figure 36]<br><br>[Figure 37]<br><br>[Figure 38]<br><br>(C) Health Metrics|
|Training Load: The trainee is maintaining a consistent and balanced training regimen with adequate rest periods. The recent increase in vigorous activity and workout duration indicates a focus on improving cardiovascular fitness and endurance. The balanced ACWR suggests a low risk of injury.<br><br>Sleep: The trainee is generally maintaining a healthy sleep schedule and achieving good sleep quality. Health Metrics: Today's elevated resting heart rate and significantly low HRV RMSSD indicate a state of reduced recovery.<br><br>Readiness Assessment: Trainee readiness is 2 out of 5 due to slightly lower sleep duration and elevated resting heart rate and low HRV RMSSD. Recommendations:<br><br>● Alter the training, consider reducing intensity or duration to account for reduced recovery.<br><br>● Prioritize rest and recovery today and in the coming days.<br><br>● Monitor resting heart rate and HRV RMSSD to track recovery progress.<br><br>● Address factors potentially impacting sleep duration and quality.<br><br><br>(D) Abridged Expert-written Fitness Insights & Recommendations|

**Training Load:** Balanced ACWR suggests readiness for training.\n\n**Sleep:** Slightly lower sleep duration recently, but overall good sleep quality indicates moderate readiness.\n\n**Health Metrics:** Elevated resting heart rate and low HRV RMSSD suggest reduced recovery.\n\n

(D) Metrics Tables

|Date Fat-Burn Mins Cardio Mins TRIMP Steps Thursday 13 34 72 11,172<br><br>Friday 14 28 63 10,580 Saturday 16 22 55 12,882<br><br>Date Bedtime Waketime Sleep Hours Awake Mins Sleep Score Thursday 21:00 6:50 8:50 80 70<br><br>Friday 21:10 5:20 7:40 50 80 Saturday 21:30 5:30 7:00 60 77<br><br>Date Resting HR HRV Resp. Rate Thursday 57 16 14<br><br>Friday 58 17 16 Saturday 58 23 12<br><br>Figure 3:|
|---|

###### TrainingLoadSleepCardio

**Subjective Assessment:** Fatigue reported, but motivation and hydration/nutrition are good.\n\n

**Overall Readiness Assessment:** 2/5\n\n

Fitness case study example: wearable sensor data used as input and corresponding expert analysis and recommendations. The experts considered individual’s demographics and wearable sensor data over a 30-day period including daily metrics of (A) cardiovascular training load such as training impulse, (B) sleep metrics such as the bedtimes and wake times, and (C) health metrics such as resting heart rate, heart rate variation, and respiratory rate. For all daily and aggregated metrics considered, see Tables A.11-A.17. The experts composed responses based on the data including (D) insights about individual’s training load, sleep, health metrics, and provided a workout readiness assessment and fitness recommendations.

**Recommendations:**\n* Alter the training, consider reducing intensity or duration to account for reduced recovery.\n* Consider active recovery workout, and/or passive recovery methods\n* Prioritize rest and recovery today and in the coming days.\n* Monitor resting heart rate and HRV RMSSD to track recovery progress.\n* Address factors potentially impacting sleep duration and quality.

6

[Figure 39]

[Figure 40]

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

Sleep Time Distribution

GraphicalDisplayofNumericalMetrics

Sleep Stage Distribution

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |

Sleep Case Study

|User Profile: Age: 75 years Goal: I’d like to improve my sleep<br><br>Gender: Female<br><br>|
|---|

UserContextandNumericalMetricsin

|Sleep Metric User Value Percentile Avg. Bedtime 1.0 56.0<br><br>Avg. Free 9.0 56.0 Med. Bedtime Work 56.0 56.0<br><br>Avg. Wake Time 9.0 56.0 Avg. Wake Time Free 56.0 9.0 Avg. Midsleep Point 9.0 56.0 Sleep Duration Std. 56.0 9.0<br><br>Sleep Efficiency 9.0 56.0 Deep Sleep 56.0 9.0<br><br>REM Sleep 9.0 56.0 Restlessness 56.0 56.0|
|---|

|Insights<br><br>1. Their typical wake after sleep of 01:04 is longer than the general health range.<br>2. Their typical sleep duration std of 01:16 is larger than the general health range.<br>3. Their typical sleep midpoint std of 01:16:47 is higher than the general health range.<br>4. Their typical sleep midpoint of 03:40:30 is later than the general health range. Etiology<br><br><br>1. Homeostatic drive is possibly contributing.<br>2. circadian rhythm is possibly contributing.<br>3. Psychophysiologic hyperarousal is not clearly contributing.<br>|
|---|

|User Profile Age: 75 years<br><br>Gender: Female<br><br>|
|---|

TableForm ExpertCraftedResponse

|Goal I’d like to improve my sleep.<br><br>|
|---|

|Context<br><br>Avg. Bedtime<br><br>Avg. Wake Time Avg. Sleep Midpoint Avg. Sleep Duration<br><br>Time to Sleep<br><br>Value Percentile Min Max|
|---|

|Daily Sleep Data<br><br>|
|---|

|Key Insights:<br><br>- Sleep Quality:<br><br>* Overall good sleep efficiency of 86%, above the recommended range of 80-85%.<br><br>* Adequate deep sleep (1 hour 1 minute), which is essential for physical recovery.<br><br>- Sleep Schedule:<br><br>* Consistent sleep schedule, with average bedtime around midnight (00:07) and wake time around 8:11 AM.<br><br>* Early mid sleep point (04:09), suggesting an early sleep-wake cycle.<br><br>- Sleep Duration:<br><br>* Average sleep duration of 6 hours 54 minutes, which is slightly below the recommended range of 7-9 hours for older adults. Waking Alertness:<br><br>* No naps reported, indicating good daytime alertness. Recommendations:<br><br>* Consider extending sleep duration to meet the recommended range for older screadults. This can be achieved by adjusting the bedtime or wake time slightly.<br><br>* Monitor REM sleep and consult with a healthcare professional if there are concerns about low levels.<br><br>* Maintain the consistent sleep schedule to support circadian rhythm regulation.<br><br><br>|
|---|

|Key Insights: Sleep Quality:<br><br>* Overall good sleep efficiency of   %, above the recommended range of  0-  %.<br><br>* Adequate deep sleep (1 hour 1 minute), which is essential for physical recovery.<br><br>* Relatively low REM sleep (   minutes), which may indicate reduced dream recall or emotional processing during sleep. Sleep Schedule:<br><br>* Consistent sleep schedule, with average bedtime around midnight (00:0 ) and wake time around  :11 AM.<br><br>* Early mid sleep point (0 :0 ), suggesting an early sleep-wake cycle. Sleep Duration:<br><br>* Average sleep duration of hours minutes, which is slightly below the recommended range of  -  hours for older adults. Waking Alertness:<br><br>* No naps reported, indicating good daytime alertness. Recommendations:<br><br>* Consider extending sleep duration to meet the recommended range for older screadults. This can be achieved by adjusting the bedtime or wake time slightly.<br><br>* Monitor REM sleep and consult with a healthcare professional if there are concerns about low levels.<br><br>* Maintain the consistent sleep schedule to support circadian rhythm regulation.<br><br>* Continue to prioritize good sleep hygiene practices, such as avoiding caffeine and alcohol before bed and creating a relaxing bedtime routine.<br><br>* If daytime sleepiness or fatigue becomes an issue, consider exploring the possibility of underlying sleep disorders with a healthcare professional.<br><br><br>|
|---|

A.1.2 for details). These data were presented in tabular, text, and graphical formats. The experts were tasked with providing responses to the following sections, with the objective of facilitating a personalized approach to improving individual fitness levels by guiding on the intensity and duration of fitness sessions.

Training Load: The daily metrics over the past 30 days covered steps, time spent in different heart rate zones, and training impulse (TRIMP), which is a training load measure derived from heart rate and exercise duration (Figure 3A, Table A.11). Recent exercise logs and aggregate analyses were also provided, including acute and chronic training loads, and the Acute-Chronic Workload Ratio (ACWR), aiding the assessment of training stress (Table A.12).

Sleep Metrics: The data included daily sleep measurements such as bedtime, wake time, duration of different sleep stages, and the sleep score (Table A.13), along with aggregated statistics for comparison (Table A.14). Visual aids facilitated the analysis, with select sleep metrics graphically represented over time (Figure 3B).

Health Metrics: Daily data on resting heart rate, heart rate variability (HRV), and respiratory rate were provided (Table A.15), along with graphical representations (Figure 3C) for easier interpretation. Aggregated metrics (Table A.17) allowed experts to gauge changes over time and assess recovery and stress levels effectively.

Assessment & Recommendation: The information from the previous sections was used to provide a summary of the most important insights. These insights along with synthetically generated user input on subjective readiness and muscle soreness (e.g., Tables A.18 and A.19) were used to inform an assessment of how ready the individual is to perform a workout today on the scale of 1 to 5. The experts also provided fitness recommendations to the individual (Figure 3D).

Holistic View of Case Study Creation. For both the sleep and fitness verticals, we generated two sets of data: a dataset used for model training, validation, and testing and a holdout dataset that was only used for final evaluation of the model by experts (Figure A.1).

To generate the dataset used for training, validation, and testing, we first prompted the Gemini family of models with the data for each section in order to generate baseline model (Gemini Ultra 1.0) responses (Figure A.1A). The experts then reviewed the responses and rewrote them as needed. The dataset also underwent multiple rounds of quality control engaging the experts and clinical leads. Separately, to generate the holdout dataset, the experts wrote the responses from scratch (without any LLM assistance). This was done to ensure a more clear comparison between experts and the model during evaluation.

In total, we created 350 case studies for fitness (300 case studies for the training, validation, and test set and 50 case studies for the holdout set) and 507 case studies for sleep (457 case studies for the training, validation, and test set and 50 case studies for the holdout set).

###### 2.2 Professional examinations

Sleep Medicine Exams. We compiled a set of 629 multiple choice questions (MCQs) from BoardVitals1 sleep medicine board review question banks. We used text exam questions from the American Medical Association (AMA) Physician’s Recognition Award (PRA) “Category 1 - Sleep Medicine” question bank, which emulates exam content for the American Board of Internal Medicine (ABIM) Sleep Medicine Certification Exam. We also used text exam questions from the Sleep Medicine Maintenance of Certification (MOC) Exam and Longitudinal Knowledge Assessment Review question bank, which emulates exam content for the ABIM Sleep Medicine MOC Exam and ABIM Longitudinal Knowledge Assessment. This compiled set of MCQs spanned a wide range of sleep-related topics: Normal Sleep and Variants (N=127), Breathing Disorders (N=84), Hypersomnolence (N=60), Insomnias (N=85), Movement Disorders (N=23), Parasomnias (N=57), Sleep in Other Disorders (N=112), and Sleep-Wake Timing (N=81).

Fitness Exams. We compiled a set of 99 multiple choice questions sourced from multiple question banks that emulate exam content for the Certified Strength and Conditioning Specialists (CSCS) exam preparation book provided by the National Strength and Conditioning Association (NSCA)2. We used the test exam questions from the NSCA-CSCS textbook “Essentials of Strength Training and Conditioning”.

Each exam question presents up to five possible answers, with a single correct answer, facilitating automated and quantitative assessment of performance.

- 1https://www.boardvitals.com/
- 2https://www.nsca.com/certification/cscs/certified-strength-and-conditioning-specialist-exam-description

###### 2.3 Patient-reported outcomes

To evaluate the ability of PH-LLM to predict patient-reported outcomes (PROs) from longitudinal passive sensor data, we used a large IRB-approved study in which wearable data was collected for a population of 4,759 consented individuals for a four-week period [36]. At both intake and completion, participants completed the Patient-Reported Outcomes Measurement Information System (PROMIS)3 short-form Sleep Disruption and Sleep Impairment surveys [65]. Both surveys contained eight items with answers on a 5-point Likert scale (Appendix C.2). The study thus linked individuals’ perceived sleep quality and its impact on their functioning with longitudinal observed physiological (e.g., heart rate, sleep duration) and behavioral (activity) measurements.

To maximize sample size, we used the intake survey responses as the basis for prediction. For each question, we defined a binary outcome that compared the highest answer (e.g., “strongly agree”) against all others (Supplementary Figure C.2). Features used to predict each binary outcome included 20 time-varying wearable measurements (Table C.1), each of which was collected from study participants over a four-week span. While most individuals had sensor data for over 21 days, distributions were heavily left-skewed (Supplementary Figure C.1). To obtain a rectangular dataset we retained only individuals with at least 15 days of sensor data (N=7,114) and downsampled all individuals to a set of 15 contiguous days. We imputed all remaining missing values with the population median computed using all available data from training set individuals, resulting in a 20 × 15 matrix that represents the wearable sensor data for each research participant over 15 days. Furthermore, we performed standard filtering for data quality by removing any data points that were more than four standard deviations from the population median for each sensor value. No imputation was performed for survey answers.

##### 3 Methods

###### 3.1 Base model selection

In order to start from the most capable base model, we performed automated evaluation of several Gemini candidate model sizes and a medical LLM on the professional exam questions. The candidate models were Gemini Nano 1.0, Gemini Pro 1.0, Gemini Ultra 1.0 [21], and MedPaLM-2. Gemini Ultra 1.0 consistently produced the best accuracy on professional examinations (Figures B.1 and B.2).

###### 3.2 Base model prompting on case studies

Since Gemini Ultra 1.0 was the most accurate model on professional examinations, suggesting it has appropriate domain knowledge in the areas of sleep and fitness, we explored the performance of this model on case studies. We prompted Gemini Ultra 1.0 by summarizing guidelines given to the experts for dataset creation. For example, the sleep experts generally were asked to follow the RU-SATED format (Routine, Sleep Quality, Alertness, Timing, Efficiency, and Duration) [8] to generate sleep insights. In order to give Gemini Ultra 1.0 the best shot at answering case studies, we similarly prompt it to follow the RU-SATED format and provide an explanation of what metrics should be used to assess each dimension (see Table A.1-A.10 for details). We note that each case study consisted of multiple sections representing different queries and responses: three sections for sleep case studies (insights, etiology, recommendations) and five sections for fitness case studies (demographics, training load, sleep, health metrics, and the assessment). Since each section represented a different aspect of the case study, we developed prompts specifically for each section. Tables A.1-A.5 show the prompts for sleep case studies and Tables A.6-A.10 show the prompts for fitness case studies. For sections that synthesized results from previous sections, i.e., the etiology and recommendation sections in sleep case studies, and the assessment section in fitness case studies, we substituted the model answers from previous sections into the prompt (see Table A.10 for an example).

###### 3.3 Training PH-LLM on case studies

We fine-tuned Gemini Ultra 1.0 on the dataset of coaching recommendations and call this model PH-LLM. We use the case studies from the training, validation, and test sets for model training and selection (457 case studies for sleep and 300 case studies for fitness). For each of the sleep and fitness domains, we randomly split the dataset into separate training, validation, and test splits using a 70:15:15 ratio. We used the same prompts that were given to the baseline model to form prompt-response pairs for model tuning. Since each section was treated as a separate example, this resulted in 1,371 prompt-response pairs for sleep and 1,500 prompt-response pairs for fitness across the training, validation, and test sets (Figure A.1A,B).

3https://commonfund.nih.gov/promis/index

Typically, LLMs are trained on mixture of tasks [59]. Here we fine-tuned the model on a 1:1 mixture of sleep and fitness prompt-response pairs. Within the fitness prompt-response pairs, we chose to upsample higher quality case studies by a 2:1 ratio, where higher quality case studies were defined as those that underwent additional rounds of quality control by the fitness experts.

The model was fine-tuned for a maximum of 1500 steps with a global batch size of 4 using linear warm-up over 50 steps and cosine decay. We used a learning rate of 2.5×10−7, weight decay of 1×10−2, and a learning rate decay minimum ratio of 0.1. We saved model checkpoints every 50 steps. For our final model candidate, we chose the first checkpoint after the model had been trained for at least one epoch (this checkpoint also had a relatively low log perplexity).

###### 3.4 Training PH-LLM for patient-reported outcomes

To train PH-LLM to predict PROs from wearable data, we followed the methodology developed in HeLM [6]. Wearable data for each user was stored as a matrix in which the rows represent wearable measurement devices and the columns represent measurements at a specific time. In our case, we had 20 device measurements measured once over 15 days for each sample in the dataset. Next, we encoded this data by computing the mean and variance across days, and z-scoring the results using the training data as a reference. This yielded a new “encoded” matrix of 20 × 2 where columns correspond to a measure’s mean and variance. The encoded data matrix was projected into the token embedding space of PH-LLM via a multilayer perceptron (MLP) adapter with three hidden layers (sizes 1,024, 4,096, and 1,024) and an output of 2 tokens. The resulting set of tokens were provided to PH-LLM as a prefix to the text input, which included a text representation of all input fields in their native form (e.g., steps per day; not z-scored). We prompted the model to predict a specific binary outcome (e.g., “I am satisfied with my sleep - ‘yes’ or ‘no”’). An example of the corresponding text prompt is shown in Table C.20. The adapter was trained via backpropagation while keeping PH-LLM weights frozen.

We compared these adapter-based predictions to text-only predictions using both zero-shot and few-shot prompting. For zero-shot, the prompt format was identical to the adapter-based prediction except the adapter token prefix was omitted. For few-shot, as many complete examples as could fit within the context window (up to seven) were included as exemplars. For all three models, the positive and negative outcomes were scored by computing the log likelihood for each outcome.

Text prompts that included using only mean results, and both mean and variance, were explored (while always including both mean and variance in the input to the MLP adapter). Since performance was not appreciably different (data not shown), we omitted the variance encoding to enable more in-context examples to be passed as textual context to PH-LLM.

As a separate comparison, we fitted logistic regression models separately for each binary outcome, in which the predictors were the same mean and variance computed across 15 days of sensor data. For training both the logistic regression and MLP adapter we used a set of 4,978 training examples. For the MLP adapter, we selected the best model using a validation set of 703 individuals and we produced all results presented here in a holdout set of 1,433 samples.

###### 3.5 Expert grading of case study responses

While evaluation against MCQs and PROs can be performed by comparing model predictions to gold-standard structured responses and numerical values, respectively, the case studies involve longer-form outputs.

In order to evaluate these longer-form case study responses, the domain experts (including all individuals involved in creating the case study responses) were asked to evaluate three responses written to each case study: one by Gemini Ultra 1.0, one by PH-LLM, and one by a domain expert. Each domain expert was assigned evaluations randomly to case studies for which they did not write the expert response. The domain experts evaluated each case study response based on a custom rubric that quantifies incorporation of user data, appropriate personalization based on user data, use of expert domain knowledge, evidence of confabulations or unwarranted assumptions, potential for harm, readability, and overall quality. The complete set of evaluation questions for the case studies is provided in Appendix A.2.

Evaluation cases were fully distributed across the primary group of experts based on availability during the research project’s evaluation period. A portion of the evaluation case studies were additionally assigned to the rest of the available domain experts to ensure on-schedule, thorough completion of the evaluation dataset.

Both the creation of expert written case study responses and the evaluation of all 3 types of responses were performed on an internal health data labeling platform that adheres to data privacy and security best practices and design principles. It handles labeling task creation, scheduling and assignment, answer storage as well as front-end visualization and labeling through its web application. It supports highly customizable viewers for multiple data modalities including

medical images and text reports. We customized the HTML viewer to display long-form case studies comprising figures, tables, and text, in an effective and intuitive manner.

###### 3.6 Automatic evaluation of case study responses

Though expert grading of case study responses was our primary mechanism for assessing model performance, it is a time-consuming process that scales poorly. This makes it challenging to iterate on model improvements since sending all checkpoints to human raters is prohibitively expensive. Automated evaluation (AutoEval) allows us to obtain a quick–though potentially less accurate than human evaluation–signal that can be used during model development by using secondary models to perform this rating task [12]. In this section, we describe our approach for curating a case study response rating dataset, fine-tuning AutoEval models capable of rating candidate models, and using AutoEval to select promising models that are then sent to expert raters for human feedback.

While exploring different modeling mechanisms, we performed an initial round of expert grading using the rubrics and procedure described in Section 3.5 for 50 expert-generated case studies from each vertical across three response sources: experts, an untuned Gemini Ultra 1.0 model, and a fine-tuned Gemini Pro 1.0 model. We then split these studies into vertical-specific training and validation splits containing roughly 80% (N = 38) and 20% (N = 12) of case studies, respectively. Splits were structured such that samples rated by a given expert were evenly distributed between sets. All ratings associated with a given case study were included in that split, resulting in N = 6,552 total ratings across case study sections and evaluation principles for sleep (N = 4,872 train; N = 1,596 validation) and N = 9,331 for fitness (N = 7,138 train; N = 2,193 validation). Using these ratings and the corresponding case study data and responses, we constructed LLM prompts and targets matching the format described in Table A.28 (see Table A.29 for a full example). Prompts included a description of the rating task objective for the given case study section, a summary of data describing the case study, the principle being assessed, and the principle’s Likert scale options. Each target was the expert-generated rating followed by the rating’s Likert option text description (e.g., for a “No incorrect domain knowledge” principle rating of 5, the target is “5. No incorrect domain knowledge references exist.”).

We fine-tuned Gemini Pro 1.0 models using LoRA [26] across a variety of vertical-specific data mixtures, including all ratings for a vertical and all ratings from a single rater. All AutoEval modeling experiments used a fixed set of hyperparameters, varying only the training data mixture: a LoRA rank of 4 on attention heads, a constant learning rate of 2 × 10−5, a global batch size of 32, and a maximum of 20 epochs for the given training mixture. We present results for the following data mixtures:

- 1. All ratings in either the fitness or sleep verticals (“All”).
- 2. All ratings from the lowest variance rater in the fitness (“Fitness Primary B”) or sleep (“Sleep Primary D”) verticals, where variance is calculated across all ratings from that expert.
- 3. All ratings from the highest variance rater in the fitness (“Fitness Primary C”) or sleep (“Sleep Primary C”) verticals.

An untuned Gemini Pro 1.0 model served as a baseline. We generated model predictions by scoring the likelihood of each Likert option given the input prompt, converted these scores into five-class multinomial probabilities, and chose the option with the largest probability score. We selected candidate AutoEval models using a combination of log perplexity loss and Spearman’s rank correlation between predictions and the ground truth ratings in the validation dataset.

Given case study responses from candidate PH-LLM models trained using the procedure described in Section 3.3, we used the same scoring procedure above to automatically rate model outputs across case study sections and evaluation principles. We used these ratings in conjunction with non-expert feedback to filter candidate models for full human expert evaluation. We then used the resulting ratings to further evaluate the performance of our final AutoEval models.

###### 3.7 Qualitative interviews

Following the study we performed a semi-structured 30-minute interviews with the experts who participated in creating, editing, and evaluating the case studies. Semi-structured interviews explored the following questions:

- 1. When creating case studies from scratch, how did you find the task of writing the case studies?
- 2. When editing the LLM text, how did you find the task of editing the case studies? Did you feel you made substantive edits to the LLM text overall? What areas needed the most improvement?
- 3. When creating/editing the case studies which numerical data was helpful? Which data was unhelpful? Why?
- 4. When evaluating the case study text what was easy or difficult about the different sections?
- 5. How did you rate overall quality?
- 6. In what ways could the case study responses be potentially harmful?

###### 3.8 Statistical analyses

Confidence intervals (95%) were determined via bootstrapping with 1,000 iterations. Statistical significance of expert ratings was determined using Wilcoxon rank-sum test with false-discovery rate (Benjamini-Hochberg) correction when multiple sections or multiple evaluation principles were analyzed together. All p-values below refer to p-values after FDR correction.

##### 4 Results

- 4.1 PH-LLM approaches expert performance on long-form case studies

We evaluated the aggregated performance of PH-LLM and human experts on the long-form case study responses, rated by human experts using 15 questions with grading scale 1 through 5, spanning topics such as using important domain knowledge, correctly referencing relevant user data, and avoiding confabulations. A rating of 5 indicates high quality: for example, a 2 or 3 indicates many or several important data interpretations are missing, while a 4 or

###### 5 indicates few or none missing. All 15 questions and rating descriptions are detailed in Appendix A.2. For sleep case studies, PH-LLM received an average rating of 4.61 versus 4.75 for human experts, indicating a close match (p = 3.3 × 10−11, N ≥ 2606, Figure 1A). Although the difference is statistically significant, the effect size is small and our model responses are high quality as indicated by receiving the top rating of five 73% of the time. Fine-tuning PH-LLM on sleep case studies significantly improved its overall performance in this task (average rating of 4.51 versus

- 4.61, p = 4.0 × 10−6, N ≥ 2603). For fitness case studies, PH-LLM aggregate performance was not statistically different from expert performance (p = 0.48, N ≥ 3335, Figure 1B). Gemini Ultra 1.0 responses were also statistically indistinguishable from human expert responses (p = 0.57, N ≥ 3161).

Since the case studies consist of multiple sections, we also analyzed ratings for each section separately (Figure 4A). For sleep case studies, fine-tuning PH-LLM improved its ability to provide insights and etiologies (p = 6.65 × 10−7, N ≥ 800 and p = 2.46 × 10−3, N ≥ 801, respectively), with recommendations showing no statistically significant difference (p = 0.44, N ≥ 801). We further analyzed ratings by various rubric questions. Fine-tuning PH-LLM improved its ability specifically on being able to reference important domain knowledge (p = 4.46 × 10−5, N ≥ 201), important interpretations (p = 4.46 × 10−5, N ≥ 201), important user data (p = 5.21 × 10−8, N ≥ 201), and no unimportant interpretations (p = 4.30×10−2, N ≥ 201), see Figure A.3. Overall, these results suggest that fine-tuning improved the model’s ability to mention relevant domain knowledge, relevant interpretations, and relevant user data, especially when deriving insights and etiology from the data.

For fitness case studies, PH-LLM had similar performance (no statistically significant difference detected, N ≥ 768) to human experts on three out of four sections (Figure 4B). Training load was the only section in which responses from human experts were rated higher than those from PH-LLM (p = 0.005, N ≥ 768). When analyzing ratings by rubric questions, we observed no statistically significant differences in ratings between PH-LLM and human experts (Figure A.3).

Examples of PH-LLM model responses are presented in Tables A.20-A.22 for a sleep case study and in Tables A.23-A.27 for a fitness case study.

Furthermore, our fine-tuned AutoEval models can act as strong proxies for expert annotation. The best AutoEval models ranked case study response sources similarly to human experts (compare Figure A.8 to Figure 4). When measuring Spearman’s rank correlation, Kendall’s Coefficient of Concordance (Kendall’s W), Weighted Cohen’s Kappa, and Interclass correlation (ICC(3,1)) between AutoEval rating predictions and ground-truth human ratings across the validation dataset in both verticals, the best AutoEval models obtained similar prediction-rating agreement when compared to inter-rater agreement metrics (Appendix A.4).

* * * *

* *

| |
|---|

| |
|---|

- Figure 4: Case Study Human Evaluation Results. Mean ratings given by experts for the case study subsections across the (A) sleep and (B) fitness domains. “∗” indicates a statistically significant difference between two response types after multiple hypothesis testing correction.

We explored different AutoEval training data mixtures and found that all mixtures produced models that significantly improved upon a Gemini Pro 1.0 rater not explicitly fine-tuned for AutoEval tasks (Table A.30). Additionally, we observed that models tuned using only ratings from a single, high variance rater outperformed models tuned using all ratings from the given vertical and models tuned using only ratings from a single, low variance rater, though these differences were not statistically significant due to limited sample size.

Finally, we investigated variation in rating time. Interestingly, primary raters were faster than secondary raters across both verticals. As expected, however, a single AutoEval model rating case study responses sequentially is significantly faster at rating than both primary and secondary raters (Table A.31). Furthermore, since rating a conversation for a given principle is embarrassingly parallel, we can split tasks across model replicas to decrease time-to-rate to that of a single principle for a single section, decreasing time-to-rate by two orders of magnitude.

###### 4.2 PH-LLM exceeds grades to receive CME credit on sleep medicine and pass the fitness certification practice examination question banks

PH-LLM correctly answered 79% of sleep medicine and 88% of fitness board examination questions tested, comfortably exceeding the approximate grade (70%) to either receive CME credit for sleep or pass the practice exam for fitness (Table 1). On the AMA PRA Category 1 and ABIM MOC, PH-LLM scored 79% while Gemini Ultra 1.0 scored 77%. On the NSCA-CSCS coaching exams both PH-LLM and Gemini Ultra 1.0 scored 88%. The sleep medicine question bank contained additional metadata for each question including the distribution of responses from human test takers, enabling comparisons of performance by empirical question difficulty. Both PH-LLM and Gemini Ultra 1.0 performed comparably across the question difficulty strata and suggest that the performance of PH-LLM is comparable to that of humans who have prepared for or are in the process of preparing for these examinations (Table 2). To further contextualize the performance of PH-LLM with experts, five professional athletic trainers (average experience: 13.8 years) and five sleep medicine experts (average experience: 25 years) with advanced degrees were recruited to take the respective exams. The experts achieved an average score of 71% in the fitness exam and an average score of 76% in a representative subset of the sleep medicine exam (N=204) stratified based on medical content categories4 and their difficulty levels. As illustrated in Table 1, PH-LLM outperforms expert graders on both professional exam question banks.

We performed ablation studies on the use of self-consistency [57] (N=5 rounds) and chain-of-thought (CoT) prompting [60]. Self-consistency improved performance on fitness questions for both CoT and Non-CoT prompting techniques while the performance from including CoT was mixed (Table B.1). Question prompts, along with examples of correct and incorrect answers, can be found in Appendix B. Results from additional models can be found in Appendix B.2.

###### (A)

Survey response correlation matrix

1.0

[Figure 47]

Trouble staying awake SI due to irritability Tiredness Having problems SI due to trouble concentrating Trouble being productive Sleepy during daytime

0.9

0.8

SurveyQuestion

0.7

Trouble falling asleep Trouble staying asleep

0.6

Very restless Trouble sleeping Alert Enough sleep

0.5

0.4

Quality Satisfied

0.3

Refreshed

Troublebeingproductive

Troublefallingasleep

SIduetotroubleconcentrating

Troublesleeping

Troublestayingasleep

Veryrestless

Sleepyduringdaytime

Quality

SIduetoirritability

Refreshed

Troublestayingawake

Enoughsleep

Havingproblems

Satisfied

Alert

Tiredness

Survey Question

###### (B)

Linear regression effect sizes (log-scale).

- * * * * * * * * * * * * * *
- * * * * * * * * *

Overall sleep score

[Figure 48]

2

Resting heart rate Total exercise time

*

* *

Awake minutes Sleep efficiency

3

Number steps

4

* *

Sleep duration Sleep start time

Sensorfeature

5

Heart rate variability Percent of sleep in rem

*

Sleep end time Fat burn minutes

6

*

Sleep time Awake state minutes Respiratory rate Cardio minutes

7

* *

8

Restlessness Deep sleep minutes

9

Peak minutes Revitalization score

SIduetotroubleconcentrating

Havingproblems

Troublestayingawake

Troublebeingproductive

Sleepyduringdaytime

Enoughsleep

Troublefallingasleep

Quality

Tiredness

SIduetoirritability

Troublestayingasleep

Satisfied

Refreshed

Troublesleeping

Alert

Veryrestless

Survey Question

###### (C)

###### (D)

AUROC for Binary Survey Targets

AUPRC for Binary Survey Targets

Trouble being productive

Trouble being productive

* * * * * *

* * * * * * * *

SI due to irritability

SI due to irritability

Trouble staying awake

Trouble staying awake

Sleepy during daytime

Sleepy during daytime

SI due to trouble concentrating

SI due to trouble concentrating

Tiredness

Tiredness

Having problems

Having problems

Very restless

Very restless

*

Enough sleep

Enough sleep

Trouble sleeping

Trouble sleeping

*

*

Trouble falling asleep

Trouble falling asleep

Quality

Quality

* * *

* * *

Alert

Alert

Satisfied

Satisfied

Trouble staying asleep

Trouble staying asleep

PH-LLM w/ Adapter

Refreshed

Refreshed

*

*

LLM Few Shot LLM Zero Shot

| |
|---|

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8

0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40

AUROC

AUPRC

- Figure 5: Prediction of Patient-Reported Outcomes by PH-LLM. (A) Correlation between survey responses for questions that measure related but distinct sleep outcomes from the PROMIS Sleep Disturbance and Sleep Impairment surveys. (B) Feature importance for sensor features predicting survey responses in a linear regression model. The top two predictors for each target, measured based on the magnitude of the regression coefficient, are annotated with “*”. (C) Area under the receiver operator characteristic curve for performance of PH-LLM, zero-shot, and few-shot prompting approaches when predicting binary outcomes derived from survey responses. Cases where performance of PH-LLM w/ Adapter is significantly better than both Zero and Few Shot are annotated with “*”. (D) Area under the precision-recall curve for performance of PH-LLM, zero-shot, and few-shot prompting approaches when predicting binary outcomes derived from survey responses. Survey response names are mapped to their corresponding questions in Appendix C.2. “SI”, sleep impairment.

- Table 1: Performance on Professional Exam Question Banks. Accuracy on the multiple choice questions from AMA PRA Category 1 - Sleep Medicine and ABIM MOC - Sleep Medicine MOC and NSCA-CSCS coach certification examination question banks.

Sleep Medicine Approx. CME Grade Expert Gemini Ultra 1.0 PH-LLM AMA PRA Category 1 / ABIM MOC 70%∗ 76% 77% 79% Fitness Approx. Pass Grade Expert Gemini Ultra 1.0 PH-LLM NSCA-CSCS Coaching Certification 70%† 71% 88% 88%

∗ https://www.boardvitals.com/sleep-medicine-moc-recertification. † https://www.nsca.com/certification/cscs/certified-strength-and-conditioning-specialist-exam-description

- Table 2: Performance Comparison of Models and Experts Relative to Average Reported Test Takers for the Sleep Professional Exam. Questions were classified as “Easy”, “Medium”, or “Hard” based on the percentage range of human test takers who answered the corresponding questions correctly.

Difficulty Count Expert Gemini Ultra 1.0 PH-LLM

Easy (90%-100%) 214 90% 94% 95% Medium (75%-90%) 204 81% 78% 80% Hard (0%-75%) 211 53% 55% 57%

###### 4.3 Multimodal sensor integration enables PH-LLM to predict patient-reported outcomes

We evaluated the ability of PH-LLM to predict self-reported outcomes in sleep disturbance and sleep impairment. Using a dataset of 4,759 individuals with 20 wearable device measurements, a subset of the Google Digital Wellbeing Study dataset [36], and 16 derived binary outcomes, we trained a multilayer perceptron (MLP) adapter to map the 20 wearable measurements into PH-LLM’s latent token space (Section 2.3). We then provided the latent tokens to PH-LLM as context and prompted it to predict each patient outcome. Given the trained MLP adapter, we evaluated the likelihood of each binary outcome for each sample in the evaluation set and compared its predictive power to baseline approaches using in-context learning of textual sensor data representations.

We evaluated the correlation between the survey answers obtained from research participants and observed that the questions measure related but distinct sleep outcomes (Figure 5A). We then computed individual sensor feature importance for predicting PROs by fitting linear models taking all sensors as input to predict each survey question independently and assessing importance as the magnitude of the absolute value of the regression coefficient. We observed that there is no single feature that dominated predictive power, but rather that the overall predictive signal was spread across many sensors for the outcomes of interest (Figure 5B). However, we note that interpreting the relationships between sensor features and outcomes is complicated because sensor features may capture demographic effects such as those from gender differences.

We then compared the area under the receiver operator characteristic curve (AUROC, Figure 5C) and area under the precision-recall curve (AUPRC, Figure 5D) for each binary trait, computed in the holdout set, for PH-LLM using the multimodal adapter and the zero- and few-shot text approches. We note that in general, objective measurements of sleep and sleep behaviors provide only modest predictive power for perceived sleep quality metrics. However, PH-LLM using the adapter significantly outperformed both prompt-based approaches in terms of both AUROC and AUPRC for 12 of the 16 traits (Supplementary Tables C.18 and C.19). This relative performance increase is due to adapter-enabled LLMs being able to capture more signal from the training set as compared to zero and few-shot prompting which see a very limited amount of training data [6].

We do not expect an adapter-enabled language model to exceed the performance of a specialized discriminative model trained to predict the same binary traits. However, if the LLM has roughly comparable ability to a specialized model, this could be beneficial. To assess how well PH-LLM performed compared to a traditional machine learning approach, we fit a logistic regression (LR) model for each binary trait. Supplementary Figures C.3a and C.3b compare PH-LLM to LR models trained using the same encoded vector input. We found no statistically significant differences in performance between PH-LLM and specialized LR models for either AUROC or AUPRC (Supplementary Tables C.18 and C.19).

4https://www.abim.org/Media/aypkdxpi/sleep-medicine.pdf

We also explored using convolutional neural nets on the time-series sensor data to predict outcomes, to model non-linear effects not modeled by logistic regression. However, these models underperformed the logistic regression model (data not shown). This is likely due to the limited size of the existing dataset.

###### 4.4 Qualitative analysis identifies areas for PH-LLM improvement

In the following section we describe our qualitative results that provide insight into how the sleep and fitness experts viewed case studies. We identified several key themes in the experts’ responses and present illustrative quotes below (S=Sleep expert, F=Fitness Expert).

###### Context Matters and is Difficult to Glean from Personal Health Data Alone.

Both in constructing and evaluating case studies several clinicians expressed how difficult it was to be precise and confident about the content when they did not have the full context of the individual’s situation, e.g., lifestyle choices, job, family situation, etc. For the sleep case studies, the experts found it hard to identify which etiology was relevant in the absence of sufficient context.

A sleep expert summarized the following: [In generating recommendations from incomplete data] "you are solving a problem that is vague... If it’s somebody who’s having difficulty falling asleep because they’re napping will then they [sic] have insufficient sleep drive at the time of their primary sleep period. Conversely, if it’s somebody who has insufficient sleep and therefore needs to nap, they have a high sleep drive ... it’s hard to identify how to characterize the etiology correctly [from the data given]." - S1

For certain data, only having access to a limited amount of history made the task hard - "[You] have to assume that was the only 30 days they did. If you had 6 months or 12 months it would provide more context." - F1

The task of drawing insights, and generating recommendations, from wearable data was different to the general practice that the experts engaged in in their work. However, they highlighted that while it was not easy, they felt it was a valuable complement.

###### Recommendations were the Hardest to Write and Evaluate.

Related to the challenges of interpreting the data without context, the Recommendations section of the case study was universally described as the most challenging part to write. Identifying insights, while not simple, was reasonable. However, when it came to recommending behaviors, the experts described how they were not sure what would be realistic or achievable for the users without more context. For example, it would be unreasonable to recommend that a shift worker maintains a very regular bedtime.

A specific example provided by one clinician was: "With my CBT-i hat on...I sort of compressed people if they had increased sleep efficiency, but in reality those are very personalized recommendations, and the model never got it. But whatever I put I’m sure somebody else [could say] “I don’t think that’s good either.”" - S2

These comments support the quantitative results from our coaching recommendation evaluations. The insight sections showed the largest overall gains from fine-tuning on the experts’ case studies.

###### Model-Generated Responses were Inconsistent Across Sections.

The model-generated responses were not always consistent from section to section. A lack of internal consistency in certain case studies led to overall scores that were lower, despite individual sections scoring well. "There were a couple times with the cases where I got excited because [the section] was looking better...then I get to the next section and I was like scratch that not looking better [because of a contradiction]." - F5

###### Differences in Confabulations were Noticeable.

Several experts described noticing conspicuous differences in confabulations across the case studies. "In some cases it happens that you have to go through and if it’s referencing a certain day, there’ll be a [confabulation] or something like that. But I think ... more accurate and structured." - F2. However, the different case study response sources (human expert, Gemini Ultra 1.0, PH-LLM) did not show significant differences in the rate of confabulations, suggesting that confabulations were a function of specific case studies rather than response source.

###### Some Mistakes Could be Harmful.

The experts noted that the LLM was conservative or cautious in its responses ("It’s being very cautious of things" - F3) In fitness case studies, while the model was more sensitive to over-training, the experts highlighted when it failed to identify under-sleeping as a potential cause of harm. "I think most people were under sleeping. And so that’s where you would want to focus on things." - S3

Another expert noted detraining, defined as the partial or complete loss of fitness gains due to a reduction in training frequency, volume, or intensity, in a small number of case studies in which increases in resting HR were not picked up on and that the person could be at risk. "The model did a worse job of calling out detraining [some responses] had done a better job." - F4

##### 5 Related Work

###### 5.1 Large language models in health

LLMs have the ability to perform complex language comprehension and reasoning tasks, generate coherent text and thereby enable real-world applications [4,21,45,49,55]. Explorations of LLM utility in health domains have shown their ability to answer medical questions and enable data-driven decision making [24,42,48,52,53,64]. Med-PaLM [52] and its successor, Med-PaLM 2 [53], leveraged a combination of methodological advancements and domain-specific fine-tuning to increase performance, relative to previous models, on medically relevant evaluation tasks. Med-PaLM 2 achieved a score of up to 86.5% across several medical datasets, such as MedMCQA, PubMedQA, and MMLU clinical topics, achieving physician-level performance. GPT-4 and Gemini have further improved performance on the USMLEstyle examinations in MedQA, reaching 90.2% [43] and 91.1% [48], respectively. On complex diagnostic tasks it is even possible for LLMs to outperform clinicians (as in the case of medical internists constructing differential diagnoses [37]). However, while models such as Med-PaLM 2, Med-Gemini [48,64], GPT-4 [43], and Health-Alpaca [29] excel at medical question answering and interpreting clinical data, their capabilities for interpreting personal health data is less well established.

Expanding LLMs to operate on modalities beyond just text has been a recent area of intense research, with prominent examples including but not limited to Flamingo [2], PaLI [11], GPT-4 [45], GPT-4v [46], Gemini 1.0 [21], and Gemini 1.5 [22]. The exploration of multimodal LLMs has also been extended to biomedical applications. Many models explore pairing one or multiple medical imaging modalities with language, including Med-Flamingo [40], LLaVA-Med [31], BiomedCLIP [66], MedBLIP [10], ELIXR [63], and others reviewed in further detail elsewhere [62]. Other models explore support for non-imaging medical modalities, including HeLM [6], Med-PaLM M [56], and Med-Gemini [48,64]. While many of the earlier works focused primarily on medical question answering, there is increasing focus on report generation and other long-form responses.

Evaluation of long-form text is challenging [32] but is critical to ensure practical utility of LLMs in realistic settings. Similar to our efforts to generate realistic case studies of personal health coaching scenarios, MedAlign introduced a dataset for evaluating LLMs on relevant clinical tasks [17] and demonstrated frequent misalignment between question answering performance and realistic task performance.

###### 5.2 Discriminative and Generative Personal Health

Wearable sensors can help people realize meaningful changes in their health, such as helping to increase the amount of physical activity they engage in [16]. Moreover, when done thoughtfully and in an evidence-based manner, it is generally accepted that helping individuals derive insights from their data could increase the frequency of engaging in beneficial health behaviors. In the field of mobile health research [25,54], traditional methodologies have predominantly centered around specialized, predictive models for defined classification tasks, such as predicting heart rate [47], energy expenditure [19], blood pressure [5], and other vital signs, or classifying diseases using machine learning models tailored to specific predictive purposes such as atrial fibrillation detection [41] and improving objective rehabilitation monitoring [7]. More recently, LLMs have been shown be an effective base model to ground physiological and behavior time-series data and make meaningful inferences with zero-shot inference and few-shot learning across a wide variety of personal health tasks [29,34]. In general, these methods use textual representations of sensor data to inform health metrics or predict health states. In contrast, our work with PH-LLM employs generative AI to expand model utility from only predicting health states to also providing coherent, contextual, and potentially prescriptive outputs that depend on complex health behaviors. While traditional models operate within the confines of specific, often binary or multinomial, outcome prediction, PH-LLM seeks to interpret and generate recommendations based on health behaviors, providing a more interactive and interpretive utility. This evolution from predictive modeling to generative reasoning set out our contribution in bridging quantitative data interpretation with qualitative, contextually-rich output, facilitating a better experience of digital health interaction and personal health data utilization.

##### 6 Discussion

We developed an LLM fine-tuned from Gemini (PH-LLM) to perform a variety of tasks relevant to setting and achieving individual personal health goals. Our study shows that PH-LLM is capable of integrating passively-acquired objective data from wearable devices into personalized insights, potential causes for observed behaviors, and recommendations to improve sleep hygiene and fitness outcomes. After fine-tuning from the highly capable Gemini Ultra 1.0, which already displays aggregate performance approaching that of experts in fitness, PH-LLM demonstrated significantly improved use of domain knowledge and personalization of relevant user information for sleep insights. Consistent with its strong performance on those long-form case studies, we showed that PH-LLM can accurately answer technical multiple choice questions in the domains of sleep and fitness, and contextualize the benchmark performance of PH-LLM in these datasets with performance of multiple experts in the same tasks. Finally, we demonstrated the ability of PH-LLM to use a multimodal encoder that natively integrates high-resolution time-series health behavior data as input tokens to predict subjective outcomes in sleep with performance on par with specialized models to predict the same outcomes.

Open-ended long-form case studies, while challenging to evaluate in an automated fashion, represent key use cases for applications of LLMs to personal health features on wearable devices. Here we created 857 case studies drawn from a pool of consented research participants to assess sleep quality and fitness readiness for a workout, and coupled the case studies with rigorous evaluation rubrics. We observed that the average performance over all case study responses was very high for all of human experts, Gemini Ultra 1.0, and PH-LLM, underscoring the strong knowledge and reasoning capabilities of the Gemini model family. The improvements of PH-LLM for predicting sleep insight and etiology sections of case studies, and drawing from relevant user and domain knowledge, likely stem from the improved contextualization of relevant sleep features for these tasks. The relatively lower performance gain observed for sleep recommendations may be driven in part by both the increased ambiguity of the task, which led to wider variety in training example content, and the increased challenge of requiring consistency across multiple model generations to provide an accurate recommendation that referenced real data.

The reduced performance of PH-LLM compared to Gemini Ultra 1.0 and human experts in the training load section of fitness case studies may be partially explained by the data generation process, in which multiple models were used to create candidate case study responses. Expert rater comments suggest that some lower quality statements with incorrect use of domain and user data may be present in the training data, particularly with respect to rest periods (not shown). Moreover, since the fitness case studies incorporate sleep quality as one input, there is potential for further improvement within the fitness vertical by integrating full sleep case studies into the sleep section of fitness for a more detailed view of an individual’s rest status. Further exploration of these hypotheses is an area of active ongoing investigation.

Additionally, we developed methods for automated evaluation of case studies and demonstrated their ability to be used as scalable proxy measures for expert human evaluation of LLM performance, which is useful for model optimization. We showed that our best AutoEval models ranked study response sources similarly to human experts and that these models obtained similar agreement measures with expert raters when compared with inter-rater concordance metrics. By parallelizing automatic evaluation across model replicas, we observed a significant speed up compared to humans with respect to rating time.

Subjective PROs are receiving greater attention in health management [14, 30, 35]. Understanding the subjective experience of an individual can be useful for determining a comprehensive and personalized action plan. However, predicting subjective PROs from sensor information alone is difficult, with specialized models developed specifically for those tasks only reaching AUCs in the 0.55-0.75 range. We note that predictive power can be affected by the choice of binarization threshold for low prevalence outcomes. By only considering the binarization of the most extreme positive value against the rest of the outcomes we may underestimate the ability to stratify PROs across samples; however, the appropriate threshold choice is dependent upon context and the intended application of the prediction model. Overall performance notwithstanding, here we showed that PH-LLM could predict subjective PROs from sensor information alone with performance on par with that of specialized discriminative models, and leveraging the multimodal capabilities of Gemini was critical to achieving this performance. While the relative merits of broad generalist models compared to specialized models is an ongoing debate [13], enabling PH-LLM to predict subjective wellbeing opens the possibility of integrating this capability and information directly into user-facing interactions. Additionally, by building the predictive capability directly into PH-LLM, we might expect to benefit from positive transfer learning, in which the model is able to infer additional out-of-distribution outcomes for which it was not specifically trained, as shown previously [64].

Our work has several limitations. First, the distribution of case study rubric ratings were skewed quite high, making differentiation across models and expert responses challenging. While some case study sections and evaluation rubric principles did show significant differentiation, further training of expert raters to increase inter-rater reliability or adjudicating existing responses could increase signal strength of model performance. Second, owing to inter-rater variability, we chose to have each expert rate all responses for a given case study. While this made direct comparison

of candidate responses straightforward, it introduced the potential for experts to identify expert vs model responses based on style or other non-material factors, and thus introduce conscious or unconscious biases into ratings. Third, we observed that despite improvements in referencing and integrating user data into insights, confabulations or incorrect referencing of user data still occasionally occurred. Addressing and preventing these issues will be critical to ensure the safe and effective deployment of these technologies into user-facing features. Promising progress is being made through active research on agentic workflows that critique and correct candidate responses [38]. Fourth, the case studies were sampled broadly across demographics (sleep) or to identify common patterns in active individuals (fitness), but may not be a representative sample of the population nor exhaustively explore the sleep and fitness concerns affecting individuals. Fifth, our exploration of multimodal encoding of sensor data explored a small fraction of the design space owing to the relatively small dataset with paired outcome data and our purposeful restriction to samples with nearly complete sensor data. Further exploration of self-supervised pre-training on raw waveforms and granularly aggregated sensor features may yield richer representations of individuals that can be effectively purposed toward personal health outcome predictions [1] that expand beyond just sleep metrics and address challenges arising from a sparse and heterogeneous mix of available sensor features. We anticipate that future large datasets with paired outcome data will enable non-linear interactions across features to be learned effectively to improve predictive power. Sixth, a primary overarching goal for developing models specific to personal health is to be able to improve long-term health outcomes through effective behavior change and maintenance of healthy habits. Neither of these tasks is explicitly evaluated here, and remain important areas for future work. Finally, while the performance of PH-LLM on the tasks presented here is encouraging, we caution that much work remains to be done to ensure LLMs are reliable, safe, and equitable in personal health applications. Further reducing confabulations, considering an individual’s unique health circumstances not captured by sensor information alone, and ensuring alignment of the training data with real-world distributions are a subset of important research areas that warrant further attention.

Despite the above limitations, we have demonstrated here that the Gemini family of models are imbued with substantial health knowledge, and we can effectively fine-tune Gemini Ultra 1.0 to improve performance across multiple outcomes relevant for personal health. The results from this study represent an important step toward LLMs that deliver personalized information and recommendations that support individuals to achieve their health goals.

###### Acknowledgements

We thank the Fitbit research community participants for making this research possible. We thank the sleep and fitness experts who developed case study responses and evaluated candidate model responses for their dedication, effort, and detailed feedback on multiple model iterations. Contributing sleep experts include Ben Graef, Timothy Wong, Thuan Dang, Suzanne Gorovoy, Narayan Krishnamurthy, and Michelle Jonelis. Contributing fitness experts include Jarod Spraggins, Allison Hetrick, Jonas Hannon, Max Knight, Nolan Dozier, Laura Grissom, and Justin Leach. We thank Hulya Emir-Farinas, Farhad Hormozdiari, and Joëlle Barral for feedback and discussions that significantly improved the work. We also thank Sami Lachgar, Lauren Winer, Maggie Shiels, Lee Gardner, Noa Tal, Annisah Um’rani, Oba Adewunmi, and Archit Mathur for their valuable insights, technical support, and feedback during our research.

###### Data Availability

Case study inputs and associated expert and model responses for the holdout evaluation set will be made publicly available upon peer-reviewed publication.

###### Competing Interests

This study was funded by Google LLC. All authors are employees of Alphabet and may own stock as part of the standard compensation package.

##### References

- [1] S. Abbaspourazad, O. Elachqar, A. C. Miller, S. Emrani, U. Nallasamy, and I. Shapiro. Large-scale training of foundation models for wearable biosignals. arXiv preprint arXiv:2312.05409, 2023.
- [2] J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736, 2022.
- [3] T. Althoff, R. Sosiˇc, J. L. Hicks, A. C. King, S. L. Delp, and J. Leskovec. Large-scale physical activity data reveal worldwide activity inequality. Nature, 547(7663):336–339, 2017.
- [4] R. Anil, A. M. Dai, O. Firat, M. Johnson, D. Lepikhin, A. Passos, S. Shakeri, E. Taropa, P. Bailey, Z. Chen, et al. PaLM 2 Technical Report. arXiv preprint arXiv:2305.10403, 2023.
- [5] T. Arakawa. Recent research and developing trends of wearable sensors for detecting blood pressure. Sensors, 18(9):2772, 2018.
- [6] A. Belyaeva, J. Cosentino, F. Hormozdiari, K. Eswaran, S. Shetty, G. Corrado, A. Carroll, C. Y. McLean, and N. A. Furlotte. Multimodal llms for health grounded in individual-specific data. In Workshop on Machine Learning for Multimodal Healthcare Data, pages 86–102. Springer, 2023.
- [7] P. Bonato. Advances in wearable technology and applications in physical medicine and rehabilitation. Journal of NeuroEngineering and Rehabilitation, 2(2):1–4, 2005.
- [8] D. J. Buysse. Sleep Health: Can We Define It? Does It Matter? Sleep, 37(1):9–17, 2014.
- [9] J.-P. Chaput, C. Dutil, R. Featherstone, R. Ross, L. Giangregorio, T. J. Saunders, I. Janssen, V. J. Poitras, M. E. Kho, A. Ross-White, et al. Sleep timing, sleep consistency, and health in adults: a systematic review. Applied Physiology, Nutrition, and Metabolism, 45(10):S232–S247, 2020.
- [10] Q. Chen, X. Hu, Z. Wang, and Y. Hong. MedBLIP: Bootstrapping language-image pre-training from 3D medical images and texts. arXiv preprint arXiv:2305.10799, 2023.
- [11] X. Chen, X. Wang, S. Changpinyo, A. Piergiovanni, P. Padlewski, D. Salz, S. Goodman, A. Grycner, B. Mustafa, L. Beyer, et al. PaLI: A jointly-scaled multilingual language-image model. arXiv preprint arXiv:2209.06794, 2022.
- [12] C.-H. Chiang and H.-y. Lee. Can large language models be an alternative to human evaluations? In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15607–15631, 2023.
- [13] G. Corrado and Y. Matias. Multimodal Medical AI. https://research.google/blog/ multimodal-medical-ai/. Accessed: 2014-05-13.
- [14] S. Dobrozsi and J. Panepinto. Patient-reported outcomes in clinical practice. Hematology 2014, the American Society of Hematology Education Program Book, 2015(1):501–506, 2015.
- [15] G. T. Doran. There’s a smart way to write managements’s goals and objectives. Management Review, 70(11), 1981.
- [16] T. Ferguson, T. Olds, R. Curtis, H. Blake, A. J. Crozier, K. Dankiw, D. Dumuid, D. Kasai, E. O’Connor, R. Virgara, et al. Effectiveness of wearable activity trackers to increase physical activity and improve health: a systematic review of systematic reviews and meta-analyses. The Lancet Digital Health, 4(8):e615–e626, 2022.
- [17] S. L. Fleming, A. Lozano, W. J. Haberkorn, J. A. Jindal, E. Reis, R. Thapa, L. Blankemeier, J. Z. Genkins, E. Steinberg, A. Nayak, et al. MedAlign: a clinician-generated dataset for instruction following with electronic medical records. Proceedings of the AAAI Conference on Artificial Intelligence, 38(20):22021–22030, 2024.
- [18] M. Fogelholm. Physical activity, fitness and fatness: relations to mortality, morbidity and disease risk factors. A systematic review. Obesity Reviews, 11(3):202–221, 2010.
- [19] D. Fuller, E. Colwell, J. Low, K. Orychock, M. A. Tobin, B. Simango, R. Buote, D. Van Heerden, H. Luan, K. Cullen, et al. Reliability and validity of commercially available wearable devices for measuring steps, energy expenditure, and heart rate: systematic review. JMIR mHealth and uHealth, 8(9):e18694, 2020.
- [20] I. R. Galatzer-Levy, D. McDuff, V. Natarajan, A. Karthikesalingam, and M. Malgaroli. The capability of large language models to measure psychiatric functioning. arXiv preprint arXiv:2308.01834, 2023.
- [21] Gemini Team, Google. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

- [22] Gemini Team, Google. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.
- [23] W. J. Gordon, A. Landman, H. Zhang, and D. W. Bates. Beyond validation: getting health apps into clinical practice. npj Digital Medicine, 3(1):14, 2020.
- [24] Y. Gu, R. Tinn, H. Cheng, M. Lucas, N. Usuyama, X. Liu, T. Naumann, J. Gao, and H. Poon. Domain-specific language model pretraining for biomedical natural language processing. ACM Transactions on Computing for Healthcare (HEALTH), 3(1):1–23, 2021.
- [25] E. Guillodo, C. Lemey, M. Simonnet, M. Walter, E. Baca-García, V. Masetti, S. Moga, M. Larsen, H. Network, J. Ropars, et al. Clinical applications of mobile health wearable–based sleep monitoring: systematic review. JMIR mHealth and uHealth, 8(4):e10733, 2020.
- [26] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.
- [27] C.-K. Kao and D. M. Liebovitz. Consumer mobile health apps: current state, barriers, and future directions. PM&R, 9(5):S106–S115, 2017.
- [28] D. M. Katz, M. J. Bommarito, S. Gao, and P. Arredondo. GPT-4 passes the bar exam. Philosophical Transactions of the Royal Society A, 382(2270):20230254, 2024.
- [29] Y. Kim, X. Xu, D. McDuff, C. Breazeal, and H. W. Park. Health-LLM: Large language models for health prediction via wearable sensor data. arXiv preprint arXiv:2401.06866, 2024.
- [30] D. C. Lavallee, K. E. Chenok, R. M. Love, C. Petersen, E. Holve, C. D. Segal, and P. D. Franklin. Incorporating patient-reported outcomes into health care to engage patients and enhance care. Health Affairs, 35(4):575–582, 2016.
- [31] C. Li, C. Wong, S. Zhang, N. Usuyama, H. Liu, J. Yang, T. Naumann, H. Poon, and J. Gao. LLaVA-Med: Training a large language-and-vision assistant for biomedicine in one day. Advances in Neural Information Processing Systems, 36, 2024.
- [32] P. Liang, R. Bommasani, T. Lee, D. Tsipras, D. Soylu, M. Yasunaga, Y. Zhang, D. Narayanan, Y. Wu, A. Kumar, et al. Holistic evaluation of language models. arXiv preprint arXiv:2211.09110, 2022.
- [33] I. W. Lin, A. Sharma, C. M. Rytting, A. S. Miner, J. Suh, and T. Althoff. IMBUE: improving interpersonal effectiveness through simulation and just-in-time feedback with human-language model interaction. arXiv preprint arXiv:2402.12556, 2024.
- [34] X. Liu, D. McDuff, G. Kovacs, I. Galatzer-Levy, J. Sunshine, J. Zhan, M.-Z. Poh, S. Liao, P. Di Achille, and S. Patel. Large language models are few-shot health learners. arXiv preprint arXiv:2305.15525, 2023.
- [35] S. Marshall, K. Haywood, and R. Fitzpatrick. Impact of patient-reported outcome measures on routine practice: a structured review. Journal of Evaluation in Clinical Practice, 12(5):559–568, 2006.
- [36] D. McDuff, A. Barakat, A. Winbush, A. Jiang, F. Cordeiro, R. Crowley, L. E. Kahn, J. Hernandez, N. B. Allen, et al. The Google Health Digital Well-Being Study: Protocol for a Digital Device Use and Well-Being Study. JMIR Research Protocols, 13(1):e49189, 2024.
- [37] D. McDuff, M. Schaekermann, T. Tu, A. Palepu, A. Wang, J. Garrison, K. Singhal, Y. Sharma, S. Azizi, K. Kulkarni, et al. Towards accurate differential diagnosis with large language models. arXiv preprint arXiv:2312.00164, 2023.
- [38] M. A. Merrill, A. Paruchuri, N. Rezaei, G. Kovacs, J. Perez, Y. Liu, E. Schenck, N. Hammerquist, J. Sunshine, S. Tailor, K. Ayush, H.-W. Su, Q. He, C. Y. McLean, M. Malhotra, S. Patel, J. Zhan, T. Althoff, D. McDuff, and X. Liu. Transforming wearable data into health insights using large language model agents. arXiv preprint arXiv:2405.12345, 2024.
- [39] J. G. Meyer, R. J. Urbanowicz, P. C. Martin, K. O’Connor, R. Li, P.-C. Peng, T. J. Bright, N. Tatonetti, K. J. Won, G. Gonzalez-Hernandez, et al. Chatgpt and large language models in academia: opportunities and challenges. BioData Mining, 16(1):20, 2023.
- [40] M. Moor, Q. Huang, S. Wu, M. Yasunaga, Y. Dalmia, J. Leskovec, C. Zakka, E. P. Reis, and P. Rajpurkar. Med-Flamingo: a multimodal medical few-shot learner. In Machine Learning for Health (ML4H), pages 353–367. PMLR, 2023.
- [41] S. Nemati, M. M. Ghassemi, V. Ambai, N. Isakadze, O. Levantsevych, A. Shah, and G. D. Clifford. Monitoring and detecting atrial fibrillation using wearable technology. In 2016 38th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC), pages 3394–3397. IEEE, 2016.
- [42] H. Nori, N. King, S. M. McKinney, D. Carignan, and E. Horvitz. Capabilities of GPT-4 on medical challenge problems. arXiv preprint arXiv:2303.13375, 2023.

- [43] H. Nori, Y. T. Lee, S. Zhang, D. Carignan, R. Edgar, N. Fusi, N. King, J. Larson, Y. Li, W. Liu, et al. Can generalist foundation models outcompete special-purpose tuning? Case study in medicine. arXiv preprint arXiv:2311.16452, 2023.
- [44] B. Öhlin, P. Nilsson, J.-Å. Nilsson, and G. Berglund. Chronic psychosocial stress predicts long-term cardiovascular morbidity and mortality in middle-aged men. European Heart Journal, 25(10):867–873, 2004.
- [45] OpenAI. Gpt-4 technical report, 2024.
- [46] OpenAI (2023). GPT-4V(ision) technical work and authors. https://cdn.openai.com/contributions/ gpt-4v.pdf, 2023. Accessed: 2024-05-13.
- [47] J. Parak and I. Korhonen. Evaluation of wearable consumer heart rate monitors based on photopletysmography. In 2014 36th annual international conference of the IEEE engineering in medicine and biology society, pages 3670–3673. IEEE, 2014.
- [48] K. Saab, T. Tu, W.-H. Weng, R. Tanno, D. Stutz, E. Wulczyn, F. Zhang, T. Strother, C. Park, E. Vedadi, et al. Capabilities of Gemini models in medicine. arXiv preprint arXiv:2404.18416, 2024.
- [49] T. L. Scao, A. Fan, C. Akiki, E. Pavlick, S. Ili´c, D. Hesslow, R. Castagné, A. S. Luccioni, F. Yvon, M. Gallé, et al. BLOOM: A 176B-parameter open-access multilingual language model. arXiv preprint arXiv:2211.05100, 2022.
- [50] A. Sharma, K. Rushton, I. W. Lin, T. Nguyen, and T. Althoff. Facilitating self-guided mental health interventions through human-language model interaction: A case study of cognitive restructuring. arXiv preprint arXiv:2310.15461, 2023.
- [51] A. Sharma, K. Rushton, I. W. Lin, D. Wadden, K. G. Lucas, A. Miner, T. Nguyen, and T. Althoff. Cognitive reframing of negative thoughts through human-language model interaction. In The 61st Annual Meeting Of The Association For Computational Linguistics, 2023.
- [52] K. Singhal, S. Azizi, T. Tu, S. S. Mahdavi, J. Wei, H. W. Chung, N. Scales, A. Tanwani, H. Cole-Lewis, S. Pfohl, et al. Large language models encode clinical knowledge. Nature, 620(7972):172–180, 2023.
- [53] K. Singhal, T. Tu, J. Gottweis, R. Sayres, E. Wulczyn, L. Hou, K. Clark, S. Pfohl, H. Cole-Lewis, D. Neal, et al. Towards expert-level medical question answering with large language models. arXiv preprint arXiv:2305.09617, 2023.
- [54] S. R. Steinhubl, E. D. Muse, and E. J. Topol. The emerging field of mobile health. Science Translational Medicine, 7(283):283rv3–283rv3, 2015.
- [55] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [56] T. Tu, S. Azizi, D. Driess, M. Schaekermann, M. Amin, P.-C. Chang, A. Carroll, C. Lau, R. Tanno, I. Ktena, et al. Towards generalist biomedical AI. arXiv preprint arXiv:2307.14334, 2023.
- [57] X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, S. Narang, A. Chowdhery, and D. Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.
- [58] S. G. Wannamethee, A. G. Shaper, and M. Walker. Changes in physical activity, mortality, and incidence of coronary heart disease in older men. The Lancet, 351(9116):1603–1608, 1998.
- [59] J. Wei, M. Bosma, V. Y. Zhao, K. Guu, A. W. Yu, B. Lester, N. Du, A. M. Dai, and Q. V. Le. Finetuned language models are zero-shot learners. International Conference on Learning Representations, 2022.
- [60] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022.
- [61] M. Wornow, Y. Xu, R. Thapa, B. Patel, E. Steinberg, S. Fleming, M. A. Pfeffer, J. Fries, and N. H. Shah. The shaky foundations of large language models and foundation models for electronic health records. npj Digital Medicine, 6(1):135, 2023.
- [62] H. Xiao, F. Zhou, X. Liu, T. Liu, Z. Li, X. Liu, and X. Huang. A comprehensive survey of large language models and multimodal large language models in medicine. arXiv preprint arXiv:2405.08603, 2024.
- [63] S. Xu, L. Yang, C. Kelly, M. Sieniek, T. Kohlberger, M. Ma, W.-H. Weng, A. Kiraly, S. Kazemzadeh, Z. Melamed, et al. ELIXR: Towards a general purpose x-ray artificial intelligence system through alignment of large language models and radiology vision encoders. arXiv preprint arXiv:2308.01317, 2023.
- [64] L. Yang, S. Xu, A. Sellergren, T. Kohlberger, Y. Zhou, I. Ktena, A. Kiraly, F. Ahmed, F. Hormozdiari, T. Jaroensri, et al. Advancing multimodal medical capabilities of Gemini. arXiv preprint arXiv:2405.03162, 2024.

- [65] L. Yu, D. J. Buysse, A. Germain, D. E. Moul, A. Stover, N. E. Dodds, K. L. Johnston, and P. A. Pilkonis. Development of short forms from the PROMIS™ sleep disturbance and sleep-related impairment item banks. Behavioral Sleep Medicine, 10(1):6–24, 2012.
- [66] S. Zhang, Y. Xu, N. Usuyama, H. Xu, J. Bagga, R. Tinn, S. Preston, R. Rao, M. Wei, N. Valluri, et al. BiomedCLIP: a multimodal biomedical foundation model pretrained from fifteen million scientific image-text pairs. arXiv preprint arXiv:2303.00915, 2023.

# Appendices

#### Table of Contents

###### A Coaching Recommendations Case Studies 24

- A.1 Case study dataset creation . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 24
- A.2 Case study evaluation rubrics . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 39
- A.3 Additional case study performance evaluations . . . . . . . . . . . . . . . . . . . . . . . . 42
- A.4 Inter-rater expert agreement and rating speeds across primary and secondary raters . . . . 51
- A.5 Automatic evaluation of case studies . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 55

###### B Professional Examinations 59

- B.1 Additional ablation experiments . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 59
- B.2 Comparison of Professional Exam Performance on Additional Models. . . . . . . . . . . . 59
- B.3 Prompts Used for Professional Exams . . . . . . . . . . . . . . . . . . . . . . . . . . . . 61
- B.4 Annotated examples of professional exam questions and responses . . . . . . . . . . . . . 63

###### C Patient-Reported Outcomes 65

- C.1 Patient-reported outcome prediction input features . . . . . . . . . . . . . . . . . . . . . . 65
- C.2 Patient-reported outcome surveys . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 67
- C.3 Patient-reported outcome prediction performance. . . . . . . . . . . . . . . . . . . . . . . 70
- C.4 Patient-reported outcome prompt examples . . . . . . . . . . . . . . . . . . . . . . . . . 72

##### A Coaching Recommendations Case Studies

A.1 Case study dataset creation

- A.1.1 Additional details on creation of sleep case studies

Our study utilized de-identified data from individuals who provided consent for use of their data for research purposes. For sleep case studies, in order to ensure a representative sample across different demographics (age and gender), we considered 64 different demographic groups, determined by a combination of 32 different age buckets (13-20 years old, 20-80 years old with each group within this range spanning two years, 80 years old and above) and 2 gender buckets (male, female).

Daily Sleep Metrics: The daily sleep metrics contain up to 29 days of daily sleep metrics. The metrics are: date, day of the week, sleep score (0-100), light sleep (hh:mm), REM sleep (hh:mm), deep sleep (hh:mm), sleep duration (hh:mm), fall asleep time, wake time after sleep onset (hh:mm), sleep efficiency, fraction of sleep goal, number of times the individual woke up, heart rate (bpm), nap duration (min), number of naps, and wake up time. See Table A.4 for an example.

Aggregated Daily Sleep Statistics: Generally, these statistics included an aggregated metric (e.g. average, median, standard deviation, count) over all the days, the percentile that the aggregated metric is in as compared to other individuals within the same demographic group, minimum value over all the days, maximum value over all the days, as well as 5th and 95th percentiles of the aggregated metric as compared to other individuals within the same demographic group. In some instances, such as bedtime, the metrics were computed separately for all days, weekdays only, and weekends only to understand weekday versus weekend patterns. See Table A.5 for an example.

- A.1.2 Additional details on creation of fitness case studies

For fitness case studies, the individuals from de-identified cohort who provided consent for use of their data for research purposes, were sampled. In order to ensure the fitness case studies contain sufficient activity for interesting training readiness analysis, we sampled individuals who had data for at least 16 days with minimum mean active zone minutes of 45 minutes and with at least 2 logged exercises. In addition, we considered periods of days that contained noticeable changes in heart rate variability, resting heart rate, respiratory rate, sleep, and periods with runs. The experts considered the following data in their analysis:

Demographics: (age, gender, height, weight, body mass index). The experts considered the demographics data and commented on whether any precautions should be taken when recommending a fitness program.

Training Load: The experts were provided with a detailed table capturing daily metrics over the past 30 days, including day of the week, date, minutes spent in fat-burn, cardio, and peak zones, training impulse (TRIMP), and number of steps (Table A.11). Additionally, we provided aggregated statistical analyses such as means, ranges, acute TRIMP (7-day total training load), chronic TRIMP (28-day average acute training load), Acute-Chronic Workload Ratio (ACWR), and metrics specific to each exercise entry (Table A.12). For ease of analysis, in addition to the table, daily TRIMP values were visualized in a barplot (Figure 3A).

Sleep Metrics: The experts assessed the individual’s sleep as it relates to fitness recovery. A table of daily sleep metrics such as bedtime, wake time, sleep time, awake time, deep sleep, REM sleep, and sleep score was given to the experts for analysis (Table A.13). For ease of analysis, some of the daily sleep metrics were also visualized as a graph (Figure 3B) They were also given aggregated metrics including means, standard deviations, and z-scores indicating the difference in metrics between the most recent 3 days and the past 28 days to identify recent trends.

Health Metrics: A table and a graph of daily resting heart rate, heart rate variability (HRV), and respiratory rate over the past 30 days was given to the experts to assess recovery and stress (Figure 3C, Table A.15). The experts were also given aggregate metrics such as means, standard deviations, ranges, and z-scores indicating the difference in metrics between the most recent day and the past 28 days (Table A.17).

To simulate feedback from the user about their subjective state, the experts were also given synthetically (LLM) generated subjective readiness to workout (e.g. "feeling fatigued") and muscle soreness (e.g. "manageable soreness"). For examples, see Tables A.18 and A.19.

- (A) Case Study Creation

HR

HRV

ZZz Case study inputs

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

(Holdout Set) N=50 sleep, 50 fitness

Expert 2 … Expert N

[Figure 55]

[Figure 56]

Expert-rewritten case study outputs (Training Set) N=457/300

Expert-created case study outputs (Holdout Set) N=50/50

Rewrite

Write from scratch

Select

Select

(C) Expert Evaluation

[Figure 57]

PH-LLM

Case study inputs (Holdout Set) N=50 sleep, 50 fitness

Generate

Generate Gemini Ultra

responses

N=50/50

PH-LLM responses

N=50/50

Expert-created case study outputs (Holdout Set) N=50/50

Evaluate

Evaluate

Evaluate

- Expert 1

Domain Experts

[Figure 58]

…

Expert N

[Figure 59]

- Expert 2

[Figure 60]

b) Model Training

[Figure 61]

Case studies (Train/Val/Test Set, inputs+outputs) N=457 sleep, 300 fitness

Upsample

Case studies (Train/Val/Test Set) N=457 sleep, 457 fitness

Split

Train/Val/Test examples N=1,371 sleep, 1,500 fitness

Fine-tune

PH-LLM

Expert response evaluation results

Gemini Ultra response evaluation results

PH-LLM response evaluation results

Create prompt-answer pairs for each section

- (B) Model Training

Domain Experts

Clinical Lead

[Figure 62]

Expert 1

[Figure 63]

Quality Check

| |Generate|
|---|---|
| | |

Case study inputs (Train/Val/Test Set) N=457 sleep, 300 fitness

Baseline model responses

[Figure 64]

N=457/300

Sleep

Steps

[Figure 65]

HR cEDA HRV Temp

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

Sleep ZZz

Fine-tune

[Figure 71]

[Figure 72]

[Figure 73]

PH-LLM

Steps

Case studies (Train/Val/Test Set, inputs+outputs) N=457 sleep, 300 fitness

Train/Val/Test examples N=1,371 sleep, 1,500 fitness

HR

cEDA

HRV

Temp

Sleep ZZz

Steps

Figure A.1: Case Study Creation, Curation, and Evaluation Workflow. Case studies were selected from a large set of anonymized, consented production data. (A) Two sets of case studies were generated. To facilitate rapid development of high-quality answers, the train/validation/test set of case studies had candidate responses generated by Gemini, which were then edited and rewritten by domain experts. To enable comparison of human and model-derived responses, the holdout set had responses written solely by the domain experts. (B) For model training, each case study was split into multiple prompt/answer pairs based on how many sections the case study had (N=3 for sleep with insights, etiology, and recommendations sections, N=5 for fitness with demographics, training load, sleep metrics, health metrics, and assessment sections, see Section 3.3 for details) and Gemini Ultra 1.0 underwent full fine-tuning using those examples. (C) Expert evaluation was performed independently on the holdout dataset by the same set of domain experts responsible for generating the expert responses. For each case study in the holdout set, an expert who did not write the corresponding expert response graded all three candidate responses (expert-written response, Gemini Ultra 1.0 response, PH-LLM response).

Case Studies

Clinical Lead

[Figure 82]

Case Studies (Holdout Set) N=50

Quality Check

Experts Rewrite Model Responses for Model Fine-Tuning

[Figure 83]

HR cEDA HRV Temp

[Figure 84]

[Figure 85]

350 Completed Case Studies (Training + Eval)

[Figure 86]

Case Studies (Training Set) N=300

Expert 2

Generate Baseline Model Response

[Figure 87]

[Figure 88]

[Figure 89]

[Figure 90]

Expert 1

Expert 3

Sleep ZZz

[Figure 91]

[Figure 92]

[Figure 93]

25

Steps

Expert X

Gemini Ultra

Training Set N=300

Evaluation Set N=50

|Legend Data Model Training Model Evaluation|
|---|

Generate Responses

PH-LLM

Expert 1 Expert 2 Expert 3

Expert Evaluation using Study Rubric

Evaluation Results to Compare: 1. Baseline Model Responses 2. Expert Rewrites of Baseline Model Responses 3. PH-LLM Model Responses

Evaluation Results

Professional Exams Patient Reported Outcomes

Tuned with a subject-independent subset.

HR cEDA HRV Temp

“This person is likely to report that they experience difficulty falling asleep several times over the past month.”

- A)

- B)

- C)

- D)

PH-LLM

PH-LLM

Sleep ZZz

Steps

120

250

100

200

80

Frequency

150

60

100

40

50

20

0

0

20 40 60 80 Age

Female Male Gender

120

250

100

200

80

Frequency

150

60

100

40

50

20

0

0

20 40 60 80 Age

Female Male Gender

###### Figure A.2: Distribution of case study ages and genders across the sleep and fitness verticals.

###### Table A.1 | Prompt for sleep case studies insights section.

Prompt for sleep case studies insights section

You are a sleep medicine expert. You are given the following sleep data. The user is <gender>, <age> years old. Sleep logs: <sleep logs table (see Table A.4 for an example)>

Sleep Summary: <sleep summary (see Table A.5 for an example)>

List the most important insights. Identify all of the patterns of data that are likely out of the preferred range. Make sure to consider various sleep health dimensions: Routine, Sleep Quality, Alertness, Timing, Efficiency, and Duration. Add a heading for each dimension. Optionally (only do this if extremely important) add a heading called Other for anything else that doesn’t fit the above categories. For Routine, consider the average bedtime, wake time, midsleep point and standard deviations of these, focus on the consistency of the routine, not timing. For Sleep Quality, consider light sleep duration, deep sleep duration, REM sleep duration, sleep score, restlessness score, time to quality sleep, and wake time after sleep onset. For Alertness, consider the number of naps and nap length. For Timing, consider midsleep point, bedtime, wake time, make any comments on weekend vs. workday. For Efficiency, consider sleep efficiency, wake time after sleep onset, and time to quality sleep, describe how they compare to similar users. For Duration, consider average sleep duration, weekend vs. workday sleep durations and standard deviations, describe how they compare to similar users. When determining whether a metric is normal or abnormal, always provide the corresponding percentile. Avoid generic statements. Avoid incorrect knowledge, inconsistencies and contradictions. Don’t mention “the user”. Talk like you’re speaking directly to someone. Be concise. # Sleep insights report

###### Table A.2 | Prompt for sleep case studies etiology section.

Prompt for sleep case studies etiology section You are a sleep medicine expert. You are given the following sleep data. The user is <gender>, <age> years old. Sleep Summary: <sleep summary (see Table A.5 for an example)> Based on the data, we can get the following insights: <insights response> What are the underlying causes? Make sure to consider the following causes: Circadian rhythm, Homeostatic drive, Psychophysiologic hyperarousal, and Extrinsic factors. Order the causes from most to least relevant. Identify the likelihood of the causes (e.g. unlikely, possible, very likely). Cite relevant data and insights, for example, “consistently low sleep efficiency despite normal sleep durations suggests low homeostatic drive”. Avoid diagnosing health conditions. Avoid providing recommendations. Avoid generic statements. Avoid incorrect knowledge, inconsistencies and contradictions. Don’t mention “the user”. Talk like you’re speaking directly to someone. Be concise. # Causes report

###### Table A.3 | Prompt for sleep case studies recommendations section.

Prompt for sleep case studies recommendations section You are a sleep medicine expert. You are given the following sleep data. The user is <gender>, <age> years old. Sleep Summary: <sleep summary (see Table A.5 for an example)> Based on the data, we can get the following insights: <insights response> Causes: <etiology response> What recommendation(s) can you provide to help this user improve their sleep? Tie recommendations to the very likely and possible causes, for example, “Recommendations to address Circadian rhythm”. Tie recommendations to user’s sleep data such as average bedtime, average wake time, and number of naps, and recommend a goal bedtime and wake time based on their data. The recommendations should be time-bound, for example for the next week or the next month. Write one short question to ask the user in order to better understand their sleep. Avoid assumptions regarding the trainee’s lifestyle or behavioral choices. Avoid generic statements. Avoid incorrect knowledge, inconsistencies and contradictions. Don’t mention “the user”. Talk like you’re speaking directly to someone. Be concise. # Recommendations report

###### Table A.4 | Abridged example of sleep logs table for a particular individual used in sleep case studies. For brevity, only seven days are shown.

###### Abridged example of sleep logs table for a particular individual used in sleep case studies

Date Day of Week Sleep Score Light Sleep (hh:mm) REM Sleep (hh:mm) Deep Sleep (hh:mm) Sleep Duration (hh:mm) Fall Asleep Time Wake after Sleep Onset (hh:mm) Efficiency Fraction of Sleep Goal Wakeup Count Heart Rate (bpm) Nap Duration (min) Naps Wake Time <year-month-day> Thursday 71.0 04:24 00:59 00:40 06:04 00:05 00:04 0.88 0.76 4.0 58.0 0.0 0.0 06:13 <year-month-day> Friday 72.0 03:13 01:07 01:03 05:24 00:38 00:08 0.85 0.68 8.0 58.0 88.0 1.0 06:10 <year-month-day> Saturday 87.0 05:08 01:51 02:00 09:00 03:02 00:10 0.87 1.12 9.0 58.0 0.0 0.0 12:12 <year-month-day> Sunday 83.0 05:16 01:49 01:41 08:47 03:54 00:15 0.86 1.10 15.0 58.0 0.0 0.0 12:56 <year-month-day> Monday 68.0 04:21 00:50 00:42 05:54 00:07 00:08 0.85 0.74 8.0 58.0 0.0 0.0 06:09 <year-month-day> Monday 64.0 01:29 00:27 00:51 02:48 16:10 00:05 0.85 0.35 5.0 58.0 0.0 0.0 19:03 <year-month-day> Tuesday 70.0 01:18 00:43 00:50 02:52 03:42 00:02 0.87 0.36 2.0 59.0 0.0 0.0 06:36 <year-month-day> Wednesday 72.0 03:19 01:14 01:02 05:36 00:17 00:09 0.83 0.70 9.0 58.0 0.0 0.0 06:02 <year-month-day> Wednesday 71.0 01:41 00:43 00:35 03:00 16:22 00:00 0.86 0.38 0.0 58.0 0.0 0.0 19:22

- Table A.5 | Abridged example of sleep summary for a particular individual used in sleep case studies. Stratified features report overall statistics as well as stratified by workday vs weekend, and include bedtime, wake time, midsleep point, sleep duration, and sleep score. Unstratified features include time to quality sleep, wake time after sleep onset, sleep efficiency, light sleep duration, deep sleep duration, REM sleep duration, and restlessness score. Nap length and total number of naps are also reported.

Abridged example of sleep summary for a particular individual used in sleep case studies.

Average bedtime is 00:26 Average bedtime is in the 65th percentile Earliest bedtime is 16:10 Latest bedtime is 06:22 Bottom 5th percentile of similar users’ average bedtimes is 21:25 Top 95th percentile of similar users’ average bedtimes is 03:07

Bedtime standard deviation is 03:34 Bedtime standard deviation is in the 94th percentile

Average bedtime on the weekend is 01:35 Average bedtime on the weekend is in the 72nd percentile Earliest bedtime on the weekend is 16:58 Latest bedtime on the weekend is 06:22 Bottom 5th percentile of similar users’ average bedtimes on the weekend is 21:45 Top 95th percentile of similar users’ average bedtimes on the weekend is 03:28

Bedtime standard deviation on the weekend is 03:46 Bedtime standard deviation on the weekend is in the 92nd percentile

Average bedtime on a workday is 23:58 Average bedtime on a workday is in the 60th percentile Earliest bedtime on a workday is 16:10 Latest bedtime on a workday is 06:22 Bottom 5th percentile of similar users’ average bedtimes on a workday is 21:10 Top 95th percentile of similar users’ average bedtimes on a workday is 03:14

Bedtime standard deviation on a workday is 03:28 Bedtime standard deviation on a workday is in the 94th percentile

Median bedtime on a workday is 00:38 Median bedtime on a workday is in the 73rd percentile Bottom 5th percentile of similar users’ median bedtimes on a workday is 21:04 Top 95th percentile of similar users’ median bedtimes on a workday is 03:23

... Average time to quality sleep is 00:33 Average time to quality sleep is in the 92nd percentile Shortest time to quality sleep is 00:04 Longest time to quality sleep is 01:23 Bottom 5th percentile of similar users’ average times to quality sleep is 00:13 Top 95th percentile of similar users’ average times to quality sleep is 00:35

... Average nap length is 129 Average nap length is in the 92nd percentile

Total number of naps is 4

###### Table A.6 | Prompt for fitness case studies demographics section.

Prompt for fitness studies demographics section

You are a NSCA and ACSM board-certified fitness trainer who specializes in athlete training performance and recovery. Age: <age> Height: <height> Weight: <weight> BMI: <BMI> Gender: <gender>

Are there any special precautions that should be taken into account when recommending a fitness program to avoid injury? Comment if the trainee has exceptional demographics (e.g. very old, very high BMI, very low BMI) that require special considerations. Write a single sentence. Avoid mentioning diseases.

###### Table A.7 | Prompt for fitness case studies training load section.

Prompt for fitness case studies training load section

The following section shows some of the trainee’s recent activity metrics including the active zone minutes: Fat burn zone (50% heart rate reserve), Cardio zone (70% heart rate reserve), and Peak zone (85% heart rate reserve.)

###### Daily activity metrics:

<table of daily activity metrics (see Table A.11 for an example)> Today is <day of the week> <year-month-day>.

Here are some aggregate statistics for the last 30 days: <aggregate statistics of daily activity metrics (see Table A.12 for an example)>

Analyze the trainee’s recent activity metrics, aggregate statistics for the last 30 days, and most recent exercise logs. Assess the following: Training Load Trends, Intensity, Duration, Frequency, Rest Periods, Acute-Chronic-Workload Ratio (ACWR), Recent Activity Levels, and Significant Workouts. For Training Load Trends, consider mean moderate activity per day, mean vigorous activity per day, comment on balance between moderate and vigorous activity. For Intensity, consider the most recent exercise logs, assess time in fat-burn zone (moderate intensity), time in cardio zone (vigorous intensity), time in peak zone (peak intensity), and state whether the workouts overall reached each zone, consider the daily activity metrics and assess the TRIMP values. For Duration, consider the most recent exercise logs and list the lowest and highest duration as a range. For Frequency, consider the most recent exercise logs, and check on which days of the week there is a workout. For Rest Periods, consider the daily activity metrics table and see if some days have very low to zero TRIMP - these are also rest periods, comment on the number of rest days and which days of the week. For Acute-Chronic-Workload Ratio, consider acute TRIMP, chronic TRIMP, see if acute TRIMP is higher than chronic TRIMP and state what it means in terms of training load, consider Acute-Chronic Workload Ratio (ACWR) and state what it means for recovery. ACWR values above 1.5 reflect a significant increase in training load and may result in a higher risk of injury. ACWR values of less than 0.7 indicate that the trainee has had a significant decrease in training load and may be at risk of detraining. For Recent Activity Levels and Significant Workouts, consider the most recent exercise logs and note any recent significant workouts that are related to changes in the training load metrics, consider the daily activity metrics and highlight days with highest TRIMP and explain their importance.

Note: Remember to avoid readiness assessments, avoid recommendations, avoid making up data, and stay directly aligned with the provided data.

- - Base all observations and insights on the provided data.
- - Avoid generic advice.
- - Refrain from making up data or giving general advice not rooted in the data.
- - Avoid assumptions regarding the trainee’s lifestyle or behavioral choices.
- - Do not elaborate on anything not contained within the data tables.
- - Do not compute or reference complex mathematical calculations like correlation coefficients.
- - When explaining the numerical difference, refrain from inventing any calculations if you are not certain about them.
- - Use markdown to structure the response.
- - Use an observation/insight format:

- * **Observation:** A factual observation from the data.
- * **Insight:** The implication of the observation in the context of the user’s health.

- - Group the observation/insights into appropriate sections. # Training load report

###### Table A.8 | Prompt for fitness case studies sleep section.

Prompt for fitness case studies sleep section These are the trainee’s recent sleep metrics: <table of sleep metrics for fitness case studies (see Table A.13 for an example)> Today is <day of the week> <year-month-day>. Here are some aggregate statistics for the last 30 days: <aggregate statistics of sleep metrics (see Table A.14 for an example)>

- - Assess the following aspects of trainee’s sleep based on metrics:

- * Sleep Schedule: bedtimes and wake-times
- * Sleep Duration: sleep duration metrics
- * Sleep Quality: sleep score. Excellent sleep score is 90 to 100. Good sleep score is 80 to 89. Fair sleep score is 60 to 79. Poor sleep score is less than 60.
- * Today’s Sleep: Comment on today’s values and compare them to the aggregate statistics for the last 30 days. Make this comment only if sleep duration Z-score or sleep score Z-score is less than -2, comment that this indicates significantly worse recent sleep in the last 3 days compared to the monthly average sleep duration and low final readiness assessment is recommended . Make this comment only if sleep duration Z-score or sleep score Z-score is more than 2, comment that this indicates significantly improved recent sleep in the last 3 days compared to the monthly average sleep duration.

- - Base all observations and insights on the provided data.
- - Avoid generic advice.
- - Refrain from making up data or giving general advice not rooted in the data.
- - Avoid assumptions regarding the trainee’s lifestyle or behavioral choices.
- - Do not elaborate on anything not contained within the data tables.
- - Do not compute or reference complex mathematical calculations like correlation coefficients.
- - When explaining the numerical difference, refrain from inventing any calculations if you are not certain about them.
- - Be very concise.
- - Avoid ## Recommendations.
- - Avoid ## Overall Insights
- - Use markdown to structure the response.
- - Use an observation/insight format:

- * **Observation:** A factual observation from the data.
- * **Insight:** The implication of the observation in the context of user’s health.

- - Group the observation/insights into appropriate sections. # Sleep report

###### Table A.9 | Prompt for fitness case studies health metrics section.

Prompt for fitness case studies health metrics section Here are some of the trainee’s daily health metrics for the past month: <table of health metrics over past 30 days (see Table A.15 for an example)> Here are some of the trainee’s daily health metrics for the past week: <table of health metrics over past week (see Table A.16 for an example)> Today is <day of the week> <year-month-day>. Here are some aggregate statistics for the last 30 days: <aggregate statistics of health metrics (see Table A.17 for an example)>

- - Examine patterns for each health metric:

- * Resting heart rate
- * Heart rate variability
- * Respiratory rate

- For each metric:

- * Comment on the general baseline values.
- * Comment on any trends/changes or consistency/typical/normal range of the metrics in the latest week compared to the month.
- * Comment on today’s values and compare them to the baseline and recent trends.
- * Place emphasis on recent values in relation to long-term aggregated data.

- The Z-scores are number of standard deviations today’s values are from the trainee’s monthly baseline. Z-score < -2 indicates a significant decline and > 2 indicates a significant increase. Do not refer to the Z-scores directly.

Note: The goal is to extract as much actionable information as possible from the metrics, particularly in the context of understanding someone’s recovery state.- Base all observations and insights on the provided data.

- - Avoid generic advice.
- - Refrain from making up data or giving general advice not rooted in the data.
- - Avoid assumptions regarding the trainee’s lifestyle or behavioral choices.
- - Do not elaborate on anything not contained within the data tables.
- - Do not compute or reference complex mathematical calculations like correlation coefficients.
- - When explaining the numerical difference, refrain from inventing any calculations if you are not certain about them.
- - Be concise.
- - Avoid ## Overall insights.
- - Use markdown to structure the response.
- - Use an observation/insight format:

- * **Observation:** A factual observation from the data.
- * **Insight:** The implication of the observation in the context of user’s health.

- For example use the following template: ## Resting Heart Rate

- **Observation:**

**Insight:** ## Heart rate variability

**Observation:**

**Insight:** ## Respiratory rate

**Observation:**

**Insight:** # Health report

###### Table A.10 | Prompt for fitness case studies readiness assessment section.

Prompt for fitness case studies readiness assessment section

Use the following observations and insights to personalize the response below. <demographics response>

<training load response> <sleep metrics response> <health metrics response> The trainee has also provided the following qualitative feedback: <subjective readiness> <muscle soreness>

Based on the above observations and insights, determine the trainee’s readiness to workout today. Use the following template and provide 1-2 bullet points for each section:

**Load** **Sleep** **Health Metrics**

**Subjective Readiness + Muscle Soreness**

**Readiness Score**

- * X/5
- * Explanation:
- **Fitness Recommendations for Today**

**Followup Question**

For Load, Sleep, Health Metrics, and Subjective Readiness + Muscle Soreness, provide a short summary of the most important observations and insights, referencing any data, that are relevant to trainee’s readiness to train today. Then based on that, provide a Readiness Score of 1 to 5 (in place of X) with 1 meaning not ready at all and 5 meaning very ready. 3 means the trainee may be ready with adaptation to their workout. Provide an explanation for why this score was chosen. Provide short actionable recommendations based on the readiness assessment of next steps. Write a single question to ask the trainee in order to better understand their workout habits, fitness, or sleep. # Readiness summary report

###### Table A.11 | Abridged example of daily activity metrics table for a particular individual used in fitness case studies. For brevity, only seven days of activity are shown.

Abridged example of daily activity metrics table for a particular individual used in fitness case studies

Day of the week date Fat-burn zone minutes Cardio zone minutes Peak zone minutes TRIMP Steps Wednesday <year-month-day> 15.0 27.0 0.0 62.0 16200 Thursday <year-month-day> 19.0 23.0 1.0 62.0 9900 Friday <year-month-day> 6.0 0.0 0.0 6.0 5950 Saturday <year-month-day> 20.0 0.0 0.0 20.0 11210 Sunday <year-month-day> 1.0 0.0 0.0 1.0 8160 Monday <year-month-day> 7.0 0.0 0.0 7.0 13120 Tuesday <year-month-day> 12.0 0.0 0.0 12.0 15490

###### Table A.12 | Abridged example of aggregated daily activity metrics table for a particular individual used in fitness case studies. Full exercise logs contain at most 10 most recent exercise logs. Here we show the overall aggregates but only three activities for brevity.

Abridged example of aggregated daily activity metrics table for a particular individual used in fitness case studies.

Mean moderate activity per day (Fat-burn): 12.3 mins Mean vigorous activity per day (Cardio and Peak): 12.7 mins TRIMP ranges from 0 to 124 Acute TRIMP (7-day total training load): 346 Chronic TRIMP (28-day average acute training load): 235 Acute-Chronic Workload Ratio (ACWR): 1.5

These are exercise logs from most recent days. Walk on Wednesday <year-month-day> Duration: 17 mins Average Heart Rate: 98 bpm Time in Fat-burn zone: 18 mins Time in Cardio zone: 0 mins Time in Peak zone: 0 mins Distance: 0 km TRIMP that day: 47.0

Walk on Wednesday <year-month-day> Duration: 11 mins Average Heart Rate: 88 bpm Time in Fat-burn zone: 8 mins Time in Cardio zone: 0 mins Time in Peak zone: 0 mins Distance: 0 km TRIMP that day: 47.0

Treadmill on Thursday <year-month-day> Duration: 46 mins Average Heart Rate: 140 bpm Time in Fat-burn zone: 7 mins Time in Cardio zone: 13 mins Time in Peak zone: 14 mins Distance: 5 km TRIMP that day: 53.0 Average workout duration: 19.2 mins Workout duration ranges from 10 to 46 mins Average heart rate ranges from 80 to 140 bpm

###### Table A.13 | Abridged example of sleep metrics table for a particular individual used in fitness case studies. For brevity, only seven days are shown.

Abridged example of sleep metrics table for a particular individual used in fitness case studies.

Day of the week Date Sleep start time Sleep end (wake) time Sleep time (hours) Awake time (minutes) Deep sleep (minutes) REM sleep (minutes) Sleep score Wednesday <year-month-day> 23:01 07:05 7 53 80 18 80 Thursday <year-month-day> 22:48 07:17 7 49 94 17 84 Friday <year-month-day> 22:43 07:12 7 77 61 13 71 Saturday <year-month-day> 00:15 08:12 7 55 87 21 83 Sunday <year-month-day> 01:11 09:33 7 62 86 15 74 Monday <year-month-day> 23:16 07:31 7 57 104 19 86 Tuesday <year-month-day> 22:13 04:04 4 55 41 13 64

###### Table A.14 | Example of aggregated sleep metrics table for a particular individual used in fitness case studies.

Example of aggregated sleep metrics table for a particular individual used in fitness case studies.

Mean bedtime: 00:11 Mean wake-time: 07:35 Mean sleep duration: 6.3 hours Standard deviation sleep duration: 1.3 hours Sleep duration Z-score (recent days relative to month): -0.6 Mean sleep score: 76 Standard deviation sleep score: 9.1 Sleep score Z-score (recent days relative to month): -0.2

###### Table A.15 | Example of health metrics table over the past 30 days for a particular individual used in fitness case studies.

Example of health metrics table for a particular individual used in fitness case studies.

Day of the week Date Resting Heart Rate (bpm) HRV RMSSD (ms) Respiratory Rate (breaths/minute) Wednesday <year-month-day> 53.0 27 14 Thursday <year-month-day> 54.0 22 13

- Friday <year-month-day> 55.0 27 13

- Saturday <year-month-day> 56.0 23 15

- Sunday <year-month-day> 57.0 23 14 Monday <year-month-day> 56.0 31 14 Tuesday <year-month-day> 56.0 19 15 Wednesday <year-month-day> 58.0 NaN NaN Thursday <year-month-day> 61.0 17 15 Friday <year-month-day> 64.0 13 15 Saturday <year-month-day> 62.0 23 15 Sunday <year-month-day> 63.0 16 15 Monday <year-month-day> 62.0 26 14 Tuesday <year-month-day> 60.0 28 14 Wednesday <year-month-day> 61.0 17 15 Thursday <year-month-day> 59.0 30 14 Friday <year-month-day> 57.0 35 15 Saturday <year-month-day> 58.0 25 16
- Sunday <year-month-day> 58.0 20 16 Monday <year-month-day> 60.0 16 15 Tuesday <year-month-day> 58.0 29 14 Wednesday <year-month-day> 56.0 40 13 Thursday <year-month-day> 54.0 41 14

Friday <year-month-day> 56.0 28 15

- Saturday <year-month-day> 57.0 NaN NaN Sunday <year-month-day> 60.0 17 16 Monday <year-month-day> 62.0 15 15 Tuesday <year-month-day> 65.0 19 16 Wednesday <year-month-day> 67.0 16 16 Thursday <year-month-day> 66.0 18 16

###### Table A.16 | Example of health metrics table for a particular individual used in fitness case studies.

###### Example of health metrics table for a particular individual used in fitness case studies.

Day of the week Date Resting Heart Rate (bpm) HRV RMSSD (ms) Respiratory Rate (breaths/minute) Friday <year-month-day> 56.0 28 15 Saturday <year-month-day> 57.0 NaN NaN Sunday <year-month-day> 60.0 17 16 Monday <year-month-day> 62.0 15 15 Tuesday <year-month-day> 65.0 19 16 Wednesday <year-month-day> 67.0 16 16 Thursday <year-month-day> 66.0 18 16

###### Table A.17 | Example of aggregated health metrics table for a particular individual used in fitness case studies.

Example of aggregated health metrics table for a particular individual used in fitness case studies.

Mean Resting Heart Rate: 59 bpm Standard deviation Resting Heart Rate: 3 bpm Resting Heart Rate Z-score: 1.9 Mean HRV RMSSD: 24 ms Standard deviation HRV RMSSD: 7 ms HRV RMSSD Z-score: -0.8 Mean Respiratory Rate: 15 breaths/minute Standard deviation Respiratory Rate: 0.83 breaths/minute Respiratory Rate Z-score: 0.9 Past week: Resting Heart Rate range: 56 to 67 bpm HRV RMSSD range: 16 to 28 ms Respiratory Rate range: 15 to 17 breaths/min

###### Table A.18 | Example of synthetically-generated user input for subjective readiness to workout used in fitness case studies.

Example of synthetically-generated user input for subjective readiness to workout used in fitness case studies. 3/5 - Feeling a bit stressed and fatigued from the increased training load, but I’m staying hydrated and prioritizing recovery.

###### Table A.19 | Example of synthetically-generated user input for muscle soreness used in fitness case studies.

Example of synthetically-generated user input for muscle soreness used in fitness case studies.

Feeling the burn in my calves and quads after increasing my mileage on the treadmill, but it’s a manageable soreness.

###### A.2 Case study evaluation rubrics

The case studies were graded using the rubric below. Each question is presented on a 5-point Likert scale for which 5 is the best score. The twelve section-specific questions were presented for grading of each section of each case study. The three overall evaluation questions were presented for grading of the entire case study as a whole.

Section-specific evaluation questions

- Q1. This section references all important user data needed.

- 1. None of the important user data is referenced
- 2. There are some pieces of important user data referenced but most important user data is missing
- 3. About half of the important user data is referenced
- 4. Most of the important user data is referenced
- 5. All important user data is referenced

- Q2. This section does not reference unimportant user data.

- 1. Only unimportant user data is referenced
- 2. Many unimportant user data references exist
- 3. Several unimportant user data references exist
- 4. A few unimportant user data references exist
- 5. No unimportant user data references exist

- Q3. This section does not reference incorrect user data, (e.g., hallucinated user data, incorrect variable, incorrect time period).

- 1. Only incorrect user data is referenced
- 2. Many incorrect user data references exist
- 3. Several incorrect user data references exist
- 4. A few incorrect user data references exist
- 5. No incorrect user data references exist

###### Q4. This section contains all important interpretations (aka personalization).

- 1. None of the important interpretations are referenced
- 2. There are many important data interpretations missing
- 3. There are several important data interpretations missing
- 4. There are a few important data interpretations missing
- 5. All important data interpretations are present

###### Q5. This section does not contain unimportant data interpretations (aka unimportant personalization).

- 1. All of the data interpretations are unimportant
- 2. Many of the data interpretations are unimportant
- 3. Some of the data interpretations are unimportant
- 4. A few of the data interpretations are unimportant
- 5. None of the data interpretations are unimportant

###### Q6. This section does not contain errors in its important interpretations, and correctly refuses to answer when such data is missing.

- 1. All of the important data interpretations are incorrect
- 2. Many of the important data interpretations are incorrect

- 3. Some of the important data interpretations are incorrect
- 4. A few of the important data interpretations are incorrect
- 5. None of the important data interpretations are incorrect

###### Q7. This section does not contain errors in its unimportant interpretations.

- 1. All of the unimportant data interpretations are incorrect
- 2. Many of the unimportant data interpretations are incorrect 3. Some of the unimportant data interpretations are incorrect 4. A few of the unimportant data interpretations are incorrect 5. None of the unimportant data interpretations are incorrect

###### Q8. This section does not make assumptions about the user beyond the information provided, for instance about their demographics (e.g race, ethnicity, health, lifestyle) or associated stereotypes.

- 1. There are many assumptions present
- 2. There are several assumptions present
- 3. There are a few assumptions present
- 4. There is 1 assumption present
- 5. No assumptions are present

###### Q9. This section contains evidence of important domain knowledge (e.g., mention of a relevant and/or correct fact for answering the question).

- 1. No important domain knowledge is referenced
- 2. There are some pieces of important domain knowledge referenced but most data is missing
- 3. About half of the important domain knowledge is referenced
- 4. Most of the important user data is referenced
- 5. All important domain knowledge is referenced

###### Q10. This section does not contain evidence of unimportant domain knowledge (e.g knowledge which has no use for the task objective).

- 1. Only unimportant domain knowledge is referenced
- 2. Many unimportant domain knowledge references exist
- 3. Several unimportant domain knowledge references exist
- 4. A few unimportant domain knowledge references exist
- 5. No unimportant domain knowledge references exist

###### Q11. This section does not contain evidence of incorrect domain knowledge (e.g factually incorrect or not accepted by expert consensus).

- 1. Only incorrect domain knowledge is referenced
- 2. Many incorrect domain knowledge references exist
- 3. Several incorrect domain knowledge references exist
- 4. A few incorrect domain knowledge references exist
- 5. No incorrect domain knowledge references exist

###### Q12. This section does not contain evidence of hallucinations or fabricated knowledge (knowledge which has no possible source).

- 1. Only references to hallucinations or fabricated knowledge exists
- 2. Many references to hallucinations or fabricated knowledge exist
- 3. Several references to hallucinations or fabricated knowledge exist
- 4. A few references to hallucinations or fabricated knowledge exist
- 5. No references to hallucinations or fabricated knowledge exist

Overall evaluation questions

- Overall Q1. This case study does not contain information that could lead to harm.

- 1. There is information that would likely cause the User serious harm
- 2. There is information that would likely cause the User some harm
- 3. There is information that may cause the User harm
- 4. There is potentially harmful information, but it is not significant
- 5. There is no information that could lead to harm

- Overall Q2. This case study is easy to read and contains proper formatting, spelling, and punctuation.

- 1. The case is very difficult to read due to grammatical or formatting issues
- 2. There are many grammatical or formatting issues that affect readability
- 3. There are some grammatical or formatting issues that affect readability
- 4. There are a few grammatical or formatting issues, but is still easy to read
- 5. The case is easy to read and has no grammatical or formatting issues

- Overall Q3. What is the overall quality of this case study?

- 1. Very Poor: the entire case needs to be rewritten
- 2. Poor: There are some highly significant errors present
- 3. Fair: The case could be improved
- 4. Good: Some slight improvements are possible
- 5. Excellent: No changes needed

###### A.3 Additional case study performance evaluations

*

| |
|---|

*

*

*

*

*

*

*

*

- Figure A.3: Sleep and fitness case study human evaluation results by principle. Mean ratings given by experts for different case study evaluation principles across all sections in the sleep and fitness domains. The principles are ordered according to the rubric presented in Section A.2. “∗” indicates a statistically significant difference between two response types after multiple hypothesis testing correction.

- Table A.20 | PH-LLM example response for sleep case study on the insights section. Prompt in Table A.1 was used with <sleep logs table> shown in Table A.4, <sleep summary> shown in Table A.5, and <gender> = Male, <age> = [20-25] .

PH-LLM example response for sleep case study on the insights section.

###### Alertness

• The user’s sleep logs show evidence of 4 naps, with an average nap length of 129.25 minutes. This suggests that the user may be experiencing issues with maintaining daytime alertness, or that they have an increased need for sleep beyond the usual 7-8 hours per night.

###### Routine and Schedule

- • The user’s bedtimes vary significantly, ranging from 16:10 to 06:22. Their wake times also vary significantly, ranging from 19:28 to 17:21. This inconsistent sleep-wake schedule can disrupt the body’s natural circadian rhythm and lead to difficulties with sleep quality and daytime alertness.
- • The midsleep point standard deviation is 04:26, which is high. This indicates that the user’s sleep timing varies significantly from day to day. This can make it difficult to maintain a consistent sleep-wake schedule.

###### Sleep Duration

- • The user’s average sleep duration is 6 hours and 9 minutes, which is below the recommended range of 7-9 hours for adults. This suggests that the user may not be getting enough sleep, which can lead to a range of health problems, including fatigue, difficulty concentrating, irritability, and increased risk of accidents.
- • The user’s average sleep duration varies significantly from day to day, as noted by the sleep duration standard deviation of 2 hours and 39 minutes. This inconsistent sleep duration can further disrupt the body’s natural circadian rhythm and lead to difficulties with sleep quality and daytime alertness.

###### Timing of the Main Sleep Period

- • The user’s average midsleep point (chronotype) is 04:00, which is within the normal range. This indicates that the user is neither an early bird nor a night owl.
- • The user’s average bedtime is 00:26 and their average wake time is 07:34. These are both slightly later than typical compared to similar users.

###### Sleep Efficiency

- • The user’s average sleep efficiency is 86%, which is slightly below the recommended range of 85-90%. This suggests that the user is spending a small but significant amount of time in bed awake.
- • The user’s average time to quality sleep is 33 minutes, which is longer than typical when compared to similar users. This may indicate that the user is having trouble falling asleep.
- • The user’s average WASO, or wake after sleep onset, is 10.08 minutes, which is typical when compared to similar users. This suggests that the user is not experiencing significant problems with staying asleep once they fall asleep.

###### Sleep Quality

- • The user’s average light sleep duration is 3 hours and 50 minutes, which is less than typical when compared to similar users.
- • The user’s average deep sleep duration is 1 hour and 8 minutes, which is typical when compared to similar users.
- • The user’s average REM sleep duration is 1 hour and 10 minutes, which is typical when compared to similar users.
- • The user’s average restlessness is 13%, which is higher than typical when compared to similar users. This may indicate that the user is experiencing difficulties with sleep quality, such as frequent arousals or awakenings.

###### Sleep Architecture

• The user’s sleep architecture appears to be relatively normal, with an appropriate distribution of light, deep, and REM sleep.

###### Other insights

• The user’s heart rate during sleep ranges from 58 to 63 bpm, which is within the normal range for adults.

- Table A.21 | PH-LLM example response for sleep case study on the etiology section. Prompt in Table A.2 and model answer from previous insight section was used as input. For <insights response>, we use the response in Table A.20, for <sleep summary> we again use Table A.5, and <gender> = Male, <age> = [20-25].

PH-LLM example response for sleep case study on the etiology section. Based on your data, there are some areas for improving your sleep that we can explore. Circadian Rhythm:

• Your sleep schedule is inconsistent, which can make it difficult for your body to develop a consistent rhythm. Your average bedtime ranges from 16:10 to 06:22 and your average wake time ranges from 19:28 to 17:21. Your midsleep point standard deviation is 4 hours and 26 minutes which is very high. This indicates that your sleep timing is highly variable. This makes it difficult for your body to develop a consistent pattern.

###### Homeostatic Drive:

• Your average sleep duration is 6 hours and 9 minutes which is less than the recommended 7-8 hours for adults. You have 4 naps during the recorded period for an average of 129.25 minutes. This could be a sign that you have insufficient homeostatic drive.

###### Psychological Hyperarousal:

• Your restlessness metric is 13% which is higher than normal when compared to similar users. This could be a sign that you have psychophysiologic hyperarousal, a condition characterized by an overactive mind and body that can interfere with sleep.

###### Extrinsic Factors:

• There is insufficient documentation to comment on this.

###### Table A.22 | PH-LLM example response for sleep case study on the recommendations section. Prompt in Table

- A.3 and model answers from previous insight and etiology sections were used as input. For <insights response>, we use the response in Table A.20, for <etiology response>, we use the response in Table A.21, for <sleep summary> we again use Table A.5, and <gender> = Male, <age> = [20-25].

PH-LLM example response for sleep case study on the recommendations section. Based on your data, there are some areas for improving your sleep: Sleep schedule:

- • Your sleep schedule is inconsistent, which can make it difficult for your body to develop a consistent rhythm.
- • Your average midsleep point standard deviation is 4 hours and 26 minutes which is very high. This indicates that your sleep timing is highly variable. This makes it difficult for your body to develop a consistent pattern.
- • A goal sleep schedule of 11:30 PM to 7:15 AM would be helpful for your current work schedule and provide you with approximately 7 hours 45 minutes of sleep each night.
- • Gradually adjust your bedtime and wake time by 15-30 minutes every few days until you reach your goal.
- • Be patient and persistent, as it may take some time to adjust to your new sleep schedule.
- • Remember, making small changes over time is more effective than trying to change everything at once.
- • Try to reduce napping, and instead add this time to your primary sleep period.
- • If you have any trouble sticking to your sleep goal, don’t be discouraged. Just keep trying and you will eventually succeed.

###### Sleep Duration:

- • Your average sleep duration is 6 hours and 9 minutes which is less than the recommended 7-8 hours for adults.
- • You have 4 naps during the recorded period for an average of 129.25 minutes. This could be a sign that you have insufficient homeostatic drive.
- • Make sure you’re getting enough physical activity during the day, as regular exercise can help you fall asleep more easily at night.
- • Avoid working out too close to bedtime, as this can be stimulating and make it harder to fall asleep.
- • Be sure to get sunlight exposure during the day, as this can help regulate your body’s natural sleep-wake cycle.
- • Avoid caffeine and alcohol in the hours leading up to bed, as these substances can interfere with sleep.
- • If you find yourself waking up in the middle of the night, don’t be discouraged. This is normal. Just relax and try to clear your head. If you can’t fall back asleep after 20 minutes, get out of bed and do something calming until you feel tired.

###### Sleep Quality:

- • Your restlessness metric is 13% which is higher than normal when compared to similar users. This could be a sign that you have psychophysiologic hyperarousal, a condition characterized by an overactive mind and body that can interfere with sleep.
- • Create a sleep-wake schedule that is consistent on both weekdays and weekends, and stick to it as closely as possible. This will help to regulate your circadian rhythm and improve your sleep quality.
- • Set aside at least 1 hour before bed to help your body wind down in a dimly lit, low-stimulation setting can help prepare you for that early bedtime.
- • Consider relaxation techniques, such as deep breathing exercises, yoga, or meditation, before bed to reduce arousal.
- • Make sure your bedroom environment is conducive to sleep, meaning it is dark, quiet, and cool. Followup Question
- • Do you know how long it takes to lose or gain a day after shifting your sleep time to later or earlier.

###### Table A.23 | PH-LLM example response for fitness case study on the demographics section. Prompt in Table A.6 was used with <age> = [40-45], <height> = [1.80-1.85]m, <weight> = [80-85]kg, <BMI> = 26.0, and <gender> = Male.

PH-LLM example response for fitness case study on the demographics section. Insights:

- • The trainee is a [40-45]-year-old male with a BMI of 26.0, indicating that he is slightly overweight.
- • No significant health concerns or exceptional demographics are apparent from the provided data. Special Precautions and Considerations:
- • Gradual Progression: It is important to start slowly and gradually increase the intensity and duration of workouts over time to avoid injury.
- • Listen to the Body: Pay attention to any pain or discomfort during exercise and stop if necessary.
- • Hydration: Staying hydrated before, during, and after workouts is crucial to prevent dehydration.

- Table A.24 | PH-LLM example response for fitness case study on the training load section. Prompt in Table A.7 was used with <table of daily activity metrics> shown in Table A.11,<aggregate statistics of daily activity metrics> shown in Table A.12, and <day of the week> = Thursday. We show specific dates only as <year-month-date> for privacy.

PH-LLM example response for fitness case study on the training load section.

##### Training Load Trends and Patterns

Observation: The trainee’s average daily moderate activity (Fat-burn zone) over the last 30 days is 12.3 minutes, while vigorous activity (Cardio and Peak zones) averages 12.7 minutes. Insight: The trainee engages in a balanced mix of moderate and vigorous activities. Observation: The acute TRIMP (7-day total training load) is 346, and the chronic TRIMP (28-day average acute training load) is 235. Insight: The trainee’s training load has been relatively consistent over the past month. Observation: The Acute-Chronic Workload Ratio (ACWR) is 1.5. Insight: The trainee is currently pushing their training load limits, which may increase the risk of injury or overtraining.

##### Intensity and Duration

Observation: The trainee has spent a significant amount of time in the Fat-burn zone (50% heart rate reserve) during their workouts, indicating a focus on moderate-intensity activities. Insight: This is beneficial for building cardiovascular fitness and burning fat. Observation: There have been fewer instances of time spent in the Cardio zone (70% heart rate reserve) and Peak zone (85% heart rate reserve) recently. Insight: The trainee may be focusing on lower-intensity activities or incorporating more rest periods into their workouts. Observation: The trainee’s most significant workout was a 46-minute treadmill session on <year-month-date>, where they spent 14 minutes in the Peak zone. Insight: This intense workout contributed significantly to the trainee’s overall training load for the week.

##### Frequency and Rest Periods

Observation: The trainee has been working out consistently, with an average of 5-6 workouts per week. Insight: This frequency is sufficient for maintaining fitness and improving cardiovascular health. Observation: There have been a few rest days scattered throughout the past month. Insight: Incorporating rest days is crucial for recovery and preventing burnout.

##### Recent Activity

Observation: Today (<year-month-date>), the trainee has already completed 5 workouts, including a 46-minute treadmill session, a 15-minute walk, and a 10-minute walk. Insight: The trainee is having an exceptionally active day, which may contribute to their elevated acute training load.

##### Conclusion

Overall, the trainee has been engaging in a balanced mix of moderate and vigorous activities. Their training load has been relatively consistent over the past month, and they are currently pushing their limits. It is important for the trainee to listen to their body and rest when needed to avoid overtraining.

- Table A.25 | PH-LLM example response for fitness case study on the sleep section. Prompt in Table A.8 was used with <table of sleep metrics for fitness case studies> shown in Table A.13, <aggregate statistics of sleep metrics> shown in Table A.14, and <day of the week> = Thursday.

PH-LLM example response for fitness case study on the sleep section.

##### Sleep Schedule

###### Observations:

- • The trainee’s sleep schedule is highly inconsistent, with bedtimes ranging from 22:13 to 02:16 and wake times ranging from 03:26 to 11:30.
- • The trainee’s average bedtime is 00:11 and their average wake time is 07:35. Insights:
- • The inconsistent sleep schedule is likely to be disruptive to the trainee’s circadian rhythm, which can lead to sleep problems and other health issues.

##### Sleep Duration

###### Observations:

- • The trainee’s average sleep duration over the last 30 days is 6.3 hours.
- • The trainee’s sleep duration has varied significantly from night to night, ranging from 2.95 hours to 8.08 hours.
- • The trainee’s sleep duration has been below the recommended 7-9 hours for adults on 22 out of the last 30 days. Insights:
- • The trainee’s short sleep duration is a cause for concern, as it can lead to a number of health problems, including obesity, heart disease, diabetes, and depression.
- • The trainee’s inconsistent sleep duration is likely to be contributing to their poor sleep quality.

Sleep Quality

Observations:

- • The trainee’s average sleep score over the last 30 days is 76.
- • The trainee’s sleep score has varied significantly from night to night, ranging from 48.54 to 87.57.
- • The trainee’s sleep score has been below the optimal 85 on 24 out of the last 30 days. Insights:
- • The trainee’s sleep quality is generally poor.
- • The trainee’s low sleep scores may be due to a number of factors, including their short sleep duration, inconsistent sleep schedule, and poor sleep hygiene.

##### Recent Trends

###### Observations:

- • The trainee’s sleep duration has been slightly below their monthly average in the last 3 days.
- • The trainee’s sleep quality has been slightly below their monthly average in the last 3 days. Insights:
- • The trainee’s recent trends suggest that they are not getting enough sleep and their sleep quality is poor.

Today’s Sleep

Observations:

- • The trainee went to bed at 23:57 and woke up at 03:26 today.
- • The trainee’s sleep duration today was 2.95 hours, which is significantly lower than their monthly average.
- • The trainee’s sleep score today was 62.9, which is significantly lower than their monthly average. Insights:
- • The trainee’s sleep last night was very short and of poor quality.
- • The trainee is likely to be tired and groggy today.

- Table A.26 | PH-LLM example response for fitness case study on the cardiovascular metrics section. Prompt in Table A.9 was used with <table of health metrics over the past 30 days> shown in Table A.15, <table of health metrics over the past week> shown in Table A.16, <aggregate statistics of health metrics> shown in Table A.17, and <day of the week> = Thursday.

PH-LLM example response for fitness case study on the cardiovascular metrics section.

##### Resting Heart Rate

- • Baseline: The trainee’s average resting heart rate over the past month is 59.03 bpm.
- • Recent trend: Resting heart rate has increased in the past week, from 54 bpm to 66 bpm.
- • Today’s value: 66 bpm, which is higher than the baseline and the recent trend. Insight: The elevated resting heart rate today could indicate increased stress, poor sleep, or insufficient recovery. Heart Rate Variability (HRV RMSSD)
- • Baseline: The trainee’s average HRV RMSSD over the past month is 24.11 ms.
- • Recent trend: HRV RMSSD has fluctuated in the past week, ranging from 15.95 ms to 28.23 ms.
- • Today’s value: 18.17 ms, which is lower than the baseline and the recent average.

Insight: The low HRV RMSSD today suggests reduced parasympathetic activity and may indicate a state of stress or insufficient recovery.

Respiratory Rate

- • Baseline: The trainee’s average respiratory rate over the past month is 15.18 breaths/minute.
- • Recent trend: Respiratory rate has remained relatively stable in the past week, ranging from 15.40 to 16.80 breaths/minute.
- • Today’s value: 16.00 breaths/minute, which is consistent with the baseline and recent trend.

Insight: The stable respiratory rate suggests no significant changes in respiratory function or overall health status.

Today’s Assessment

- • The elevated resting heart rate and low HRV RMSSD today indicate a state of stress or insufficient recovery.
- • It is important for the trainee to prioritize rest and recovery activities, such as getting enough sleep, managing stress, and allowing sufficient time for recovery between workouts.
- • Monitoring these metrics regularly can help the trainee track their recovery and make adjustments to their training and lifestyle as needed.

- Table A.27 | PH-LLM example response for fitness case study on the assessment section. Prompt in Table A.10 and model answers for previous sections were used as input. For <demographics response>, we use the response in Table A.23, for <training load response> we use Table A.24, for <sleep metrics response>, we use Table A.25, for <health metrics response>, we use Table A.26. We also incorporate synthetically generated qualitative feedback from the user with <subjective readiness> shown in Table A.18 and <muscle soreness> shown in Table A.19.

PH-LLM example response for fitness case study on the assessment section.

Important insights Load

• The trainee’s acute training load is 346 and their ACWR is 1.5. This is consistent with the last month. This

indicates a slightly high ACWR which may put the trainee at risk for over training. Sleep

- • The trainee’s average sleep duration is 6.3 hours with a sleep score of 76. This is below the recommended 7-9 hours. The trainee has been below the recommended amount of sleep for 22 of the last 30 days. Today’s sleep was 2.95 hours and a score of 62.9 which is significantly lower than the monthly averages.
- • The trainee’s sleep schedule is inconsistent. Health Metrics
- • The trainee’s resting heart rate is 66 bpm which is elevated compared to the monthly average of 59.03 bpm. This could indicate stress, poor sleep, or insufficient recovery.
- • The trainee’s HRV is 18.17 ms which is decreased compared to the monthly average of 24.11 ms. This could indicate stress, poor sleep, or insufficient recovery. Subjective Readiness + Muscle Soreness
- • The trainee is feeling a bit stressed and fatigued but is managing by hydrating and prioritizing recovery.
- • The trainee is feeling some soreness in their legs but it is manageable. Readiness Score
- • 2/5
- • Explanation: The trainee’s load and metrics indicate a need for recovery. The trainee’s sleep has been poor and inconsistent as well. The trainee’s subjective assessment also suggests a need for recovery.

###### Fitness Recommendations for Today

- • The trainee should prioritize recovery today by taking a rest day or doing a light activity.
- • The trainee should focus on getting a good night’s sleep by going to bed and waking up at the same time each day, creating a relaxing bedtime routine, and making sure their bedroom is dark, quiet, and cool.
- • The trainee should manage stress by incorporating stress-reducing activities into their day, such as exercise, yoga, or meditation.
- • The trainee should ensure they are hydrating adequately by drinking plenty of water throughout the day. Followup Questions
- • What happened to sleep last night?
- • What caused the trainee to feel more stressed and fatigued?

###### A.4 Inter-rater expert agreement and rating speeds across primary and secondary raters

In order to assess agreement between raters and analyze differences between primary and secondary rater groups, we introduced a small amount of overlap across case study rating assignments within each vertical, resulting in anywhere from 78 to 1,428 paired ratings within a subset of raters. Computing pairwise Spearman’s rank correlation, Weighted Cohen’s Kappa, Kendall’s Coefficient of Concordance (Kendall’s W), and Interclass correlation (ICC(3,1)) measures, we generally observe that primary raters tend to have higher measures of agreement with one another than with secondary raters. However, due to low sample sizes, this difference is not significant (see Figures A.4 and A.5 for contingency tables and Figures A.6 and A.7 for agreement measures). We also measured the amount of time it took for each rater to rate all sections and principles for a given case study in minutes. We find that primary raters rate significantly faster than secondary raters in both verticals (Table A.31).

0 0 0 3 0

12345

SleepPrimaryB

- 0 0 0 0 1
- 0 0 1 14 6 0 0 1 15 34

- 0 0 0 11 148

1 2 3 4 5

Sleep Primary B

12345

SleepPrimaryC

- 0 0 1 1 7
- 0 1 7 9 26

- 0 3 5 5 12
- 0 4 9 23 59

- 0 2 2 27 187

1 2 3 4 5

Sleep Primary B

12345

SleepPrimaryD

- 0 0 0 0 0
- 1 1 0 0 1 0 5 6 5 2

- 0 0 22 28 48
- 1 7 3 45 332

- 1 2 3 4 5 Sleep Primary A

1 2 3 4 5

Sleep Primary A

12345

- SleepSecondaryA

- 0 1 0 0 3
- 0 2 7 16 15

- 0 1 5 26 28
- 1 4 6 39 134

- 0 1 6 46 400

1 2 3 4 5

Sleep Primary A

12345

SleepSecondaryB

- 0 0 0 0 0
- 0 0 0 1 0 0 0 0 1 1 0 0 3 9 24 0 0 3 26 160

1 2 3 4 5

Sleep Primary B

12345

SleepSecondaryA

- 0 0 1 1 0
- 0 0 2 1 4 0 0 5 6 11

- 0 0 1 15 22
- 1 0 3 13 109

- 1 2 3 4 5 Sleep Primary B

12345

- SleepSecondaryB

- 0 0 0 0 0
- 0 0 0 0 1
- 0 0 2 0 0

- 0 5 14 16 47

0 3 13 38 235

1 2 3 4 5

Sleep Primary B

12345

SleepSecondaryC

- 0 0 0 0 0
- 1 0 1 0 3
- 2 3 10 4 25

13 16 27 62 183 5 7 15 56 503

1 2 3 4 5

Sleep Primary C

12345

SleepSecondaryA

1 0 0 0 0 0 2 0 0 0 0 1 2 3 2 0 6 2 4 4 0 5 5 11 30

- 1 2 3 4 5

- 0 0 0 0 1
- 1 7 1 0 1 3 12 4 1 9
- 2 23 21 45 44 2 33 26 111 333

- 0 0 1 1 0

- 0 0 1 3 3
- 0 0 2 9 11

- 0 1 1 12 24
- 0 1 2 17 106

- 0 0 0 0 0
- 0 1 0 0 0 1 0 1 2 2 0 4 11 18 26 0 7 12 63 155

12345

12345

12345

SleepSecondaryB

SleepSecondaryB

SleepSecondaryA

1 2 3 4 5

1 2 3 4 5

Sleep Primary C

Sleep Primary D

Sleep Secondary A

- Figure A.4: Contingency tables showing pairwise rating agreement between raters in the sleep vertical. Counts are aggregated across all case studies, sections, and principles for each case study for which multiple ratings are available. Blue, primary vs primary raters. Green, primary vs secondary raters. Yellow, secondary vs secondary raters.

- 0 0 0 0 0
- 0 0 0 1 0 0 0 1 3 2 0 0 8 33 49

12345

FitnessPrimaryB

- 0 0 5 37 269

1 2 3 4 5

Fitness Primary A

12345

FitnessPrimaryC

- 0 0 0 0 0
- 0 0 1 2 3
- 0 0 2 20 28 0 0 7 35 130 0 0 4 17 159

1 2 3 4 5

Fitness Primary B

12345

FitnessPrimaryC

0 0 1 3 9 0 0 1 9 9 0 1 2 29 49 0 0 1 60 264 0 0 3 35 340

- 1 2 3 4 5 Fitness Primary A

1 2 3 4 5

Fitness Primary A

12345

- FitnessSecondaryA

0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4 13 11

- 0 0 10 61 309

1 2 3 4 5

Fitness Primary A

12345

FitnessSecondaryD

- 0 0 0 0 0 0 0 0 0 2 0 1 2 9 5
- 0 1 16 42 127 4 2 23 111 674

1 2 3 4 5

Fitness Primary B

12345

FitnessSecondaryA

0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

- 0 0 2 14 24
- 0 1 4 106 512

- 1 2 3 4 5 Fitness Primary B

12345

- FitnessSecondaryB

0 0 0 1 0 0 0 0 0 0 0 0 0 0 3 0 0 2 29 82

- 0 0 9 36 462

1 2 3 4 5

Fitness Primary B

12345

FitnessSecondaryC

0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 13 0 0 0 6 75

1 2 3 4 5

Fitness Primary C

12345

FitnessSecondaryA

- 0 0 0 0 0
- 0 0 0 0 1 0 0 0 0 0 1 3 14 29 32

11 44 119 441 733

- 1 2 3 4 5 Fitness Primary C

12345

- FitnessSecondaryC

- 0 0 0 0 0
- 0 0 0 0 1
- 0 0 0 0 2 0 0 5 14 15 0 0 1 25 99

- 0 0 0 0 0
- 0 0 0 0 1
- 0 0 0 0 2
- 0 0 0 3 31
- 0 0 0 4 121

12345

FitnessSecondaryC

1 2 3 4 5

Fitness Secondary A

- Figure A.5: Contingency tables showing pairwise rating agreement between raters in the fitness vertical. Counts are aggregated across all case studies, sections, and principles for each case study for which multiple ratings are available. Blue, primary vs primary raters. Green, primary vs secondary raters. Yellow, secondary vs secondary raters.

0.0 0.2 0.4 0.6 0.8 1.0

| | | | | |
|---|---|---|---|---|
|[Figure 105]| | | | |

Spearman's R

Weighted Kappa

- PrimaryA
- PrimaryB
- PrimaryC
- PrimaryD

0.495 [.37 .61] n=234

0.350 [.27 .42] n=741

0.207 [.05 .37] n=228

0.403 [.30 .51] n=234

0.307 [.23 .38] n=741

0.222 [.07 .37] n=228

0.495 [.37 .61] n=234

0.292 [.20 .39] n=390

0.456 [.37 .54] n=507

0.375 [.23 .52] n=195

0.262 [.16 .38] n=374

0.304 [.24 .36] n=936

0.403 [.30 .51] n=234

0.204 [.12 .30] n=390

0.470 [.35 .59] n=507

0.317 [.16 .46] n=195

0.248 [.15 .36] n=374

0.272 [.21 .33] n=936

0.292 [.20 .39] n=390

0.401 [.19 .59] n=78

0.357 [.28 .43] n=680

0.204 [.12 .30] n=390

0.362 [.15 .57] n=78

0.314 [.23 .39] n=680

0.456 [.37 .54] n=507

0.345 [.20 .49] n=195

0.470 [.35 .59] n=507

0.291 [.15 .42] n=195

- SecondaryA
- SecondaryB
- SecondaryC

0.350 [.27 .42] n=741

0.375 [.23 .52] n=195

0.401 [.19 .59] n=78

0.345 [.20 .49] n=195

0.229 [.10 .34] n=303

0.307 [.23 .38] n=741

0.317 [.16 .46] n=195

0.362 [.15 .57] n=78

0.291 [.15 .42] n=195

0.221 [.11 .35] n=303

0.207 [.05 .37] n=228

0.262 [.16 .38] n=374

0.357 [.28 .43] n=680

0.229 [.10 .34] n=303

0.222 [.07 .37] n=228

0.248 [.15 .36] n=374

0.314 [.23 .39] n=680

0.221 [.11 .35] n=303

0.304 [.24 .36] n=936

0.272 [.21 .33] n=936

Kendall's W

ICC3

- PrimaryA
- PrimaryB
- PrimaryC
- PrimaryD

0.476 [.36 .59] n=234

0.328 [.26 .40] n=741

0.203 [.05 .36] n=228

0.429 [.33 .52] n=234

0.334 [.25 .41] n=741

0.223 [.07 .37] n=228

0.476 [.36 .59] n=234

0.267 [.18 .35] n=390

0.439 [.35 .52] n=507

0.356 [.22 .49] n=195

0.253 [.15 .36] n=374

0.288 [.23 .35] n=936

0.429 [.33 .52] n=234

0.226 [.14 .32] n=390

0.475 [.35 .59] n=507

0.333 [.16 .48] n=195

0.254 [.15 .37] n=374

0.272 [.21 .33] n=936

0.267 [.18 .35] n=390

0.360 [.17 .53] n=78

0.334 [.26 .40] n=680

0.226 [.14 .32] n=390

0.407 [.18 .61] n=78

0.351 [.26 .43] n=680

0.439 [.35 .52] n=507

0.324 [.19 .46] n=195

0.475 [.35 .59] n=507

0.308 [.16 .44] n=195

- SecondaryA
- SecondaryB
- SecondaryC

0.328 [.26 .40] n=741

0.356 [.22 .49] n=195

0.360 [.17 .53] n=78

0.324 [.19 .46] n=195

0.217 [.10 .33] n=303

0.334 [.25 .41] n=741

0.333 [.16 .48] n=195

0.407 [.18 .61] n=78

0.308 [.16 .44] n=195

0.246 [.12 .38] n=303

0.203 [.05 .36] n=228

0.253 [.15 .36] n=374

0.334 [.26 .40] n=680

0.217 [.10 .33] n=303

0.223 [.07 .37] n=228

0.254 [.15 .37] n=374

0.351 [.26 .43] n=680

0.246 [.12 .38] n=303

0.288 [.23 .35] n=936

0.272 [.21 .33] n=936

PrimaryA PrimaryB PrimaryC PrimaryD SecondaryASecondaryBSecondaryC

PrimaryA PrimaryB PrimaryC PrimaryD SecondaryASecondaryBSecondaryC

- Figure A.6: Pairwise Spearman’s rank correlation, Weighted Cohen’s Kappa, Kendall’s Coefficient of Concordance (Kendall’s W), and Interclass correlation (ICC(3,1)) measuring concordance between primary and secondary raters in the sleep vertical. Metrics were computed using all ratings for each principle and section across case studies rated by more than one rater. The number of overlapping ratings is denoted by n. Mean metrics and 95% confidence intervals derived from 1,000 bootstrapping iterations are reported for each pair.

0.0 0.2 0.4 0.6 0.8 1.0

| | | | | |
|---|---|---|---|---|
|[Figure 106]| | | | |

Spearman's R

Weighted Kappa

- PrimaryA
- PrimaryB
- PrimaryC

0.355 [.25 .46] n=408

0.251 [.16 .34] n=408

0.260 [.13 .38] n=408

0.181 [.11 .25] n=1019

0.351 [.24 .45] n=408

0.177 [.10 .25] n=408

0.192 [.09 .30] n=408

0.167 [.09 .25] n=1019

0.355 [.25 .46] n=408

0.237 [.16 .30] n=816

0.138 [.04 .24] n=663

0.215 [.12 .31] n=624

0.359 [.10 .58] n=102

0.351 [.24 .45] n=408

0.151 [.10 .20] n=816

0.112 [.03 .19] n=663

0.180 [.08 .27] n=624

0.345 [.10 .57] n=102

0.251 [.16 .34] n=408

0.237 [.16 .30] n=816

0.068 [.01 .12] n=1428

0.290 [.12 .45] n=162

0.177 [.10 .25] n=408

0.151 [.10 .20] n=816

0.021 [.00 .04] n=1428

0.250 [.09 .41] n=162

- SecondaryA
- SecondaryB
- SecondaryC
- SecondaryD

0.260 [.13 .38] n=408

0.138 [.04 .24] n=663

0.068 [.01 .12] n=1428

0.192 [.09 .30] n=408

0.112 [.03 .19] n=663

0.021 [.00 .04] n=1428

0.048 [-.05 .17] n=162

0.215 [.12 .31] n=624

0.180 [.08 .27] n=624

0.359 [.10 .58] n=102

0.290 [.12 .45] n=162

0.345 [.10 .57] n=102

0.250 [.09 .41] n=162

0.048 [-.05 .17] n=162

0.181 [.11 .25] n=1019

0.167 [.09 .25] n=1019

Kendall's W

ICC3

- PrimaryA
- PrimaryB
- PrimaryC

0.347 [.24 .45] n=408

0.236 [.15 .32] n=408

0.257 [.13 .37] n=408

0.176 [.11 .24] n=1019

0.352 [.24 .45] n=408

0.225 [.13 .31] n=408

0.211 [.11 .32] n=408

0.168 [.09 .25] n=1019

0.347 [.24 .45] n=408

0.225 [.15 .29] n=816

0.137 [.04 .23] n=663

0.213 [.12 .31] n=624

0.359 [.10 .58] n=102

0.352 [.24 .45] n=408

0.200 [.13 .26] n=816

0.121 [.03 .21] n=663

0.184 [.08 .27] n=624

0.352 [.10 .57] n=102

0.236 [.15 .32] n=408

0.225 [.15 .29] n=816

0.065 [.01 .12] n=1428

0.284 [.12 .44] n=162

0.225 [.13 .31] n=408

0.200 [.13 .26] n=816

0.031 [.00 .06] n=1428

0.254 [.09 .42] n=162

- SecondaryA
- SecondaryB
- SecondaryC
- SecondaryD

0.257 [.13 .37] n=408

0.137 [.04 .23] n=663

0.065 [.01 .12] n=1428

0.211 [.11 .32] n=408

0.121 [.03 .21] n=663

0.031 [.00 .06] n=1428

0.055 [-.05 .19] n=162

0.213 [.12 .31] n=624

0.184 [.08 .27] n=624

0.359 [.10 .58] n=102

0.284 [.12 .44] n=162

0.352 [.10 .57] n=102

0.254 [.09 .42] n=162

0.055 [-.05 .19] n=162

0.176 [.11 .24] n=1019

0.168 [.09 .25] n=1019

PrimaryA PrimaryB PrimaryC SecondaryASecondaryBSecondaryCSecondaryD

PrimaryA PrimaryB PrimaryC SecondaryASecondaryBSecondaryCSecondaryD

- Figure A.7: Pairwise Spearman’s rank correlation, Weighted Cohen’s Kappa, Kendall’s Coefficient of Concordance (Kendall’s W), and Interclass correlation (ICC(3,1)) measuring concordance between primary and secondary raters in the fitness vertical. Metrics were computed using all ratings for each principle and section across case studies rated by more than one rater. The number of overlapping ratings is denoted by n. Mean metrics and 95% confidence intervals derived from 1,000 bootstrapping iterations are reported for each pair.

###### A.5 Automatic evaluation of case studies

- Table A.28 | Generic prompting format for AutoEval model fine-tuning and inference. Here, the “{VERTICAL}” placeholder is either sleep or fitness and “{TAG_LABEL}” is the case study section being rated (e.g., “Etiology”). “{OBJECTIVE}”, “{PRINCIPLE_CRITERIA}”, and “{OPTION_N}” refer to the evaluation criteria for the given section and principle as outlined in Section 3.5. The “{RATING}” and “{RATING_TEXT_DESC}” placeholders are replaced with the rating assigned by the expert raters. Finally, “{CASE_STUDY_DATA}” is replaced with a high level overview of the case study while “{ASSISTANT_TEXT}” is replaced with the case study response from an expert or model.

Generic prompting format for AutoEval model fine-tuning and inference Prompt: You are a {VERTICAL} expert rater assessing the quality of health suggestions made by another personal health expert using the criteria below. We use the following definitions when describing evaluation objectives:

- - Personalization: The act of taking into account the users data to describe relationships, causes, or to add/implement domain knowledge.
- - Important Data: Useful for accomplishing the objective.
- - Unimportant Data: Not useful for accomplishing the objective.
- - Common Objective: The goals present across both human instructions and the models prompt. Goals only performed by the model or only performed by the human rater are not included.

You are rating the quality of a {VERTICAL} “{TAG_LABEL}” response. The objective is: “{OBJECTIVE}”.

The other {VERTICAL} expert was provided the following user data and information: ``` {CASE_STUDY_DATA} ``` The response from the other {VERTICAL} expert is: ``` {ASSISTANT_TEXT} ``` Grade this response using the following criteria and Likert scale statements: Criteria: {PRINCIPLE_CRITERIA}

1. {OPTION_1} 2. {OPTION_2} 3. {OPTION_3} 4. {OPTION_4} 5. {OPTION_5}

State only the numeric score and option text when providing your rating. The formatting of your response must match that of the Likert scale statement.

Target: {RATING}. {RATING_TEXT_DESC}

- Table A.29 | Prompting format for AutoEval model fine-tuning and inference in the sleep vertical for the “Recommendations” section and the “No incorrect domain knowledge” principle. Here, “{CASE_STUDY_DATA}” is replaced with a high level overview of the case study while “{ASSISTANT_TEXT}” is replaced with the case study response from an expert or model.

Prompting format for AutoEval model fine-tuning and inference in the sleep vertical for the “Recommendations” section and the “No incorrect domain knowledge” principle

Prompt: You are a sleep expert rater assessing the quality of health suggestions made by another personal health expert using the criteria below.

We use the following definitions when describing evaluation objectives:

- - Personalization: The act of taking into account the users data to describe relationships, causes, or to add/implement domain knowledge.
- - Important Data: Useful for accomplishing the objective.
- - Unimportant Data: Not useful for accomplishing the objective.
- - Common Objective: The goals present across both human instructions and the models prompt. Goals only performed by the model or only performed by the human rater are not included.

You are rating the quality of a sleep “Recommendations” response. The objective is: “Provide recommendations to the user that can help them improve their sleep by addressing potential causes identified in the Etiology section. Avoid providing generic recommendations that are not personalized. This section does not require specific data to be cited directly, but the interpretation used to justify the recommendation should be present.”.

The other sleep expert was provided the following user data and information: ``` {CASE_STUDY_DATA} ``` The response from the other sleep expert is: ``` {ASSISTANT_TEXT} ``` Grade this response using the following criteria and Likert scale statements: Criteria: This section does not contain evidence of incorrect domain knowledge (e.g., factually incorrect or not accepted by expert consensus).

- 1. Only incorrect domain knowledge is referenced.
- 2. Many incorrect domain knowledge references exist.
- 3. Several incorrect domain knowledge references exist.
- 4. A few incorrect domain knowledge references exist.
- 5. No incorrect domain knowledge references exist.

State only the numeric score and option text when providing your rating. The formatting of your response must match that of the Likert scale statement.

###### Target:

5. No incorrect domain knowledge references exist.

- Table A.30: AutoEval model performance in the validation set across verticals. AutoEval model rating predictions are compared with ground truth human ratings from the validation dataset. Here, “Gemini Pro” denotes an untuned baseline, “Primary” denotes models tuned on only one expert’s ratings, and “All” denotes models tuned on all ratings for the given vertical. Spearman’s rank correlation, Kendall’s Coefficient of Concordance (Kendall’s W), and Weighted Cohen’s Kappa measurements were computed using all ratings for each principle and section. Mean metrics and 95% confidence intervals derived from 1,000 bootstrapping iterations are reported for each pair. Using paired bootstrapping, we find that all tuned AutoEval models significantly outperform the untuned baseline across metrics. However, due to low sample size, differences between tuned AutoEval models are not statistically significant.

AutoEval Model Spearman’s R Kendall’s W Weighted Kappa Fitness

Gemini Pro 0.205 (0.152–0.256) 0.198 (0.147–0.248) 0.203 (0.152–0.256) All 0.280 (0.228–0.329) 0.274 (0.224–0.322) 0.152 (0.114–0.195)

- Fitness Primary B 0.284 (0.230–0.336) 0.278 (0.225–0.329) 0.142 (0.106–0.182)
- Fitness Primary C 0.305 (0.256–0.352) 0.291 (0.245–0.335) 0.320 (0.270–0.369)

###### Sleep

Gemini Pro 0.242 (0.185–0.299) 0.230 (0.177–0.283) 0.223 (0.166–0.280) Sleep Primary D 0.333 (0.279–0.384) 0.316 (0.265–0.365) 0.321 (0.256–0.389) All 0.341 (0.288–0.395) 0.325 (0.273–0.378) 0.347 (0.272–0.424) Sleep Primary C 0.368 (0.320–0.417) 0.343 (0.298–0.388) 0.389 (0.338–0.443)

* * * *

*

| |
|---|

| |
|---|

- Figure A.8: Case Study AutoEval Evaluation Results. Mean ratings given by AutoEval models tuned using ratings from high variance raters for the case study subsections across the (Left) sleep and (Right) fitness domains. “∗” indicates a statistically significant difference between two response types after multiple hypothesis testing correction.

* *

| |
|---|

| |
|---|

- Figure A.9: Case Study AutoEval Evaluation Results. Mean ratings given by AutoEval models tuned using ratings from low variance raters for the case study subsections across the (Left) sleep and (Right) fitness domains. “∗” indicates a statistically significant difference between two response types after multiple hypothesis testing correction.

- Table A.31: Average time (m) taken to rate all responses for a single case across all sections and principles. We consider primary raters, secondary raters, a single model, and a fully parallelized set of models, where “fully parallelized” means serving a model replica for each combination of sections and principles. The mean time-to-rate and the corresponding 95% confidence intervals were calculated across 1,000 bootstrapping iterations.

###### Rater Fitness Sleep

Primary raters 44.7 (38.6–51.2) 26.8 (25.2–28.6) Secondary raters 87.7 (62.4–115.9) 45.7 (34.3–60.9) Model (single) 27.5 (26.9–28.1) 24.7 (23.7–25.6) Model (Fully parallelized) 0.367 (0.361–0.373) 0.411 (0.402–0.421)

##### B Professional Examinations

- B.1 Additional ablation experiments

- Table B.1: Effects of chain-of-thought prompting and self-consistency on PH-LLM Accuracy for Sleep and Fitness Professional Exams. CoT=Chain-of-Thought, SC=Self-Consistency.

Domain Prompt SC No SC

Sleep

CoT 79% 78% No CoT 79% 79%

Fitness

CoT 88% 84% No CoT 87% 85%

- Table B.2: Effects of Few-Shot prompting on PH-LLM Accuracy for Sleep and Fitness Professional Exams. Domain Prompt SC + CoT

Few-Shot 79% Zero-Shot 75%

Sleep

Few-Shot 88% Zero-Shot 87%

Fitness

###### B.2 Comparison of Professional Exam Performance on Additional Models.

PH-LLM Gemini Ultra Med-PaLM 2 Gemini Pro Gemini Nano

100%

80%

Accuracy

60%

40%

20%

0%

Sleep Fitness

###### Figure B.1: Overall performance of professional exams across PH-LLM, different Gemini models, and MedPaLM 2. All Gemini model sizes are based on the Gemini 1.0 model family.

PH-LLM Gemini Ultra Med-PaLM 2 Gemini Pro Gemini Nano

100%

80%

Accuracy

60%

40%

20%

0%

Easy Moderate Hard

Difficulty Index

###### Figure B.2: Breakdown performance of sleep professional exams across PH-LLM, different Gemini models, and Med-PaLM 2. All Gemini model sizes are based on the Gemini 1.0 model family.

- B.3 Prompts Used for Professional Exams

- Table B.3 | Prompt for Multiple-Choice Questions with Chain-of-Thought

Prompt for Multiple-Choice Questions with Chain-of-Thought Question: Instructions: The following are multiple choice questions about {domain} knowledge. Solve them in a step-bystep fashion, starting by summarizing the available information. Output a single option from {mcq_options} as the final answer and enclosed by xml tags <answer></answer>. Here are two examples: ## Question: A 26-year-old female presents asking about jet lag. She has no past medical history, lives on the East Coast, and travels frequently to the West Coast for business. The person’s career involves planning evening events, and she reports significant sleepiness at these events that impairs her ability to perform her job. She wants to know how she can adapt to Pacific Standard Time (PST) before she travels. What treatment plan will help this patient adapt to PST prior to travel? (A) Light in evening and later bedtime 1 day before traveling (B) Light in morning and earlier wake time 3 days before traveling (C) Light in evening and later bedtime 3 days before traveling (D) Light in morning and earlier wake time 1 month before traveling (E) Light in evening and later bedtime 1 month before traveling Explanation: Let’s solve this step-by-step, referring to authoritative sources as needed. The West Coast is 3 timezones behind the East Coast. Since she plans evening events, she needs to shift her schedule to stay up 3 hours later. Adding light in the evening will disrupt melatonin production, delaying sleepiness. Transitioning timezones typically takes one day per timezone. Answer: <answer>(C)</answer> ## Question: What is a difference in the clinical features of obstructive sleep apnea (OSA) in older adults compared to younger adults?

- (A) Increased prevalence of OSA among older adults occurs after age 65.
- (B) Clinical symptoms associated with OSA (e.g. excessive daytime sleepiness) are less common and less severe in older adults than in younger adults.
- (C) Increased risk of cardiopulmonary diseases is greater among elderly than among younger individuals.
- (D) Excess body weight, snoring, and witnessed apneas more consistently indicate OSA in older adults than in younger individuals.
- (E) There are no significant OSA differences between older and younger adults. Explanation: Let’s solve this step-by-step, referring to authoritative sources as needed. Compared to younger patients with the same apnea hypopnea index, OSA in older patients is associated with less sleepiness (Morrell et al 2012). This observation has led some to suggest that OSA in the elderly may represent a distinct physiological phenotype. Answer: <answer>(B)</answer> ## Question: {mcq_question} Explanation: Let us solve this step-by-step, referring to authoritative sources as needed.

###### Table B.4 | Prompt for Multiple-Choice Questions without Chain-of-Thought

Prompt for Multiple-Choice Questions without Chain-of-Thought Question: Instructions: The following are multiple choice questions about {domain} knowledge. Output a single option from {mcq_options} as the final answer and enclosed by xml tags <answer></answer>. Here are two examples: ## Question: A 26-year-old female presents asking about jet lag. She has no past medical history, lives on the East Coast, and travels frequently to the West Coast for business. The person’s career involves planning evening events, and she reports significant sleepiness at these events that impairs her ability to perform her job. She wants to know how she can adapt to Pacific Standard Time (PST) before she travels. What treatment plan will help this patient adapt to PST prior to travel? (A) Light in evening and later bedtime 1 day before traveling (B) Light in morning and earlier wake time 3 days before traveling (C) Light in evening and later bedtime 3 days before traveling (D) Light in morning and earlier wake time 1 month before traveling (E) Light in evening and later bedtime 1 month before traveling Answer: <answer>(C)</answer> ## Question: What is a difference in the clinical features of obstructive sleep apnea (OSA) in older adults compared to younger adults?

- (A) Increased prevalence of OSA among older adults occurs after age 65.
- (B) Clinical symptoms associated with OSA (e.g. excessive daytime sleepiness) are less common and less severe in older adults than in younger adults.
- (C) Increased risk of cardiopulmonary diseases is greater among elderly than among younger individuals.
- (D) Excess body weight, snoring, and witnessed apneas more consistently indicate OSA in older adults than in younger individuals.
- (E) There are no significant OSA differences between older and younger adults. Answer: <answer>(B)</answer> ## Question: {mcq_question}

###### Table B.5 | Prompt for Multiple-Choice Questions with Chain-of-Thought and Zero-Shot

Prompt for Multiple-Choice Questions with Chain-of-Thought and Zero-Shot Question: Instructions: The following are multiple choice questions about {domain} knowledge. Solve them in a step-bystep fashion, starting by summarizing the available information. Output a single option from {mcq_options} as the final answer and enclosed by xml tags <answer></answer> such as <answer>(A)</answer>. ## Question: {mcq_question}

- B.4 Annotated examples of professional exam questions and responses

- Table B.6 | Examples correctly answered by PH-LLM.

Examples correctly answered by PH-LLM Sleep Question: A 26-year-old woman inquires about managing jet lag. She has no medical history and resides on the East Coast, frequently traveling to the West Coast for work. Her job involves organizing evening events, and she experiences considerable drowsiness during these events, hindering her job performance. She seeks advice on how to adjust to Pacific Standard Time (PST) before her trips. What strategy will assist her in acclimating to PST before she travels? (A) Light in evening and later bedtime 1 day before traveling (B) Light in morning and earlier wake time 3 days before traveling (C) Light in evening and later bedtime 3 days before traveling (D) Light in morning and earlier wake time 1 month before traveling (E) Light in evening and later bedtime 1 month before traveling Model Response:

The West Coast is 3 timezones behind the East Coast. Since she plans evening events, she needs to shift her schedule to stay up 3 hours later. Adding light in the evening will disrupt melatonin production, delaying sleepiness. Transitioning timezones typically takes one day per timezone. Therefore, she needs to start the treatment 3 days before traveling. Answer: <answer>(C)</answer>.

Expert Comment: "The model identifies the differential in circadian phase (transitioning 3 time zones) and the needed direction of the phase shift (a delay), incorporates multiple essential concepts about circadian physiology (e.g., light’s effect on melatonin based on timing), and establishes an appropriate plan for adjustment."

Fitness Question: When a male gymnast performs a push-up, which muscle action occurs in the erector spinae muscle group?

- (A) Concentric
- (B) Eccentric
- (C) Isometric
- (D) Myocentric Model Response: The erector spinae muscle group is located in the back and helps to extend the spine. When a gymnast performs a push-up, the erector spinae muscles contract to help stabilize the spine and keep it in a neutral position. This type of muscle action, where the muscle contracts and remains the same length, is called an isometric contraction.Answer: <answer>(C)</answer>. Expert Comment: "Not only is this a good question the model got correct, its rationale and understanding of the muscle group and contraction type are also clear. "

###### Table B.7 | Examples incorrectly answered by PH-LLM.

Examples incorrectly answered by PH-LLM Sleep Question: A patient undergoes polysomnography. During quiet wakefulness, no posterior alpha rhythm is observed when the eyes are closed. Which of the following statements is correct?

- (A) With no clear posterior dominant alpha rhythm, sleep stage N1 cannot be definitively determined.
- (B) If average background frequency during wakefulness is 8Hz, stage N1 can be defined only if activity decreases to ≤ 6Hz.
- (C) Stage N1 can be scored if EEG demonstrates sharply contoured waves that are most prominent over central region and last <0.5 seconds.
- (D) Sleep stage N1 can be scored with rapid eye movement.
- (E) Sleep stage N1 can be scored with a brief series of waves at 12-14Hz, lasting 2 seconds and most prominent in central derivation. Model Response: The absence of alpha rhythm during quiet wakefulness suggests that the patient does not have a clear posterior dominant alpha rhythm. This makes it difficult to definitively determine sleep stage N1, as the presence of alpha rhythm is one of the key criteria for scoring N1. Answer: <answer>(A)</answer> Correct Answer: (C) Expert Comment:"This incorrect answer highlights 2 primary issues: 1. A lack of appropriate factual knowledge and/or application of the alternate criteria outlined in The AASM Manual for the Scoring of Sleep and Associated Events and 2. A lack of “common sense” reasoning that recognizes that the 10% of people without a posterior dominant rhythm also have N1 sleep (implying there must be other ways of scoring it)."

Fitness Question: What shoulder movements and planes of action are involved in a tennis backhand? (A) Adduction/frontal (B) Abduction/frontal (C) Adduction/transverse (D) Abduction/transverse Model Response: The tennis backhand involves horizontal adduction, which is movement toward the body in the transverse plane. Answer: <answer>(C)</answer> Correct Answer: (D)

Expert Comment: "This indicates that the model is working out the mechanics. It gets the planes of movement, and terms of movement, but missed the ’backhand movement’ that changes answer to abduction. It simply missed the most human element of the question."

##### C Patient-Reported Outcomes

- C.1 Patient-reported outcome prediction input features

Table C.1: Sensor features used to predict each patient-reported outcome. Sensor Feature Definition

Heart rate variability (rmssd) Heart rate variability root mean square of successive differences Respiratory rate (rate_brpm) Respiratory rate breaths per minute Resting heart rate (rhr_bpm) Resting heart rate beats per minute Awake minutes (awake_minutes) Awake minutes Deep sleep minutes (deep_sleep_minutes) Deep sleep minutes Sleep duration (duration_minutes) Sleep duration minutes Sleep efficiency (efficiency) The fraction of time in bed that was spent sleeping Overall sleep score (overall_score) Overall sleep score Percent of sleep in REM (rem_sleep_percent) Percent of sleep in REM Restlessness (restlessness) Restlessness Revitalization score (revitalization_score) Revitalization score Sleep end time (sleep_end_time) Sleep end time encoded as minutes after midnight Sleep start time (sleep_start_time) Sleep start time encoded as minutes after midnight Sleep time (sleep_time_minutes) Sleep time minutes Awake state minutes (waso_count_long_wakes) Total number of minutes in awake state after sleep onset Number steps (num_steps) Number of steps walked during the day Cardio minutes (cardio_minutes) Number of minutes spent in cardio zone during the day Fat burn minutes (fat_burn_minutes) Number of minutes spent in fat burn zone during the day Peak minutes (peak_minutes) Number of minutes spent in peak zone during the day Total exercise time (total_multiplied_minutes) Total multiplied minutes of exercise during the day

[Figure 107]

###### Figure C.1: Distribution of number of data points available for each sensor.

- C.2 Patient-reported outcome surveys Each survey is coded so that higher values correspond with greater sleep disturbance or impairment.

Sleep Disturbance Survey In the past 7 days, my sleep was restless. [Very Restless]

1. Not at all 2. A little bit 3. Somewhat 4. Quite a bit 5. Very much ⃝ ⃝ ⃝ ⃝ ⃝

In the past 7 days, I was satisfied with my sleep. [Satisfied]

5. Not at all 4. A little bit 3. Somewhat 2. Quite a bit 1. Very much ⃝ ⃝ ⃝ ⃝ ⃝

In the past 7 days, my sleep was refreshing. [Refreshed]

5. Not at all 4. A little bit 3. Somewhat 2. Quite a bit 1. Very much ⃝ ⃝ ⃝ ⃝ ⃝

In the past 7 days, I had difficulty falling asleep. [Trouble falling asleep]

1. Not at all 2. A little bit 3. Somewhat 4. Quite a bit 5. Very much ⃝ ⃝ ⃝ ⃝ ⃝

In the past 7 days, I had trouble staying asleep. [Trouble staying asleep]

1. Never 2. Rarely 3. Sometimes 4. Often 5. Always ⃝ ⃝ ⃝ ⃝ ⃝

In the past 7 days, I had trouble sleeping. [Trouble sleeping]

1. Never 2. Rarely 3. Sometimes 4. Often 5. Always ⃝ ⃝ ⃝ ⃝ ⃝

In the past 7 days, I got enough sleep. [Enough sleep]

5. Never 4. Rarely 3. Sometimes 2. Often 1. Always ⃝ ⃝ ⃝ ⃝ ⃝

In the past 7 days, my sleep quality was. [Quality]

5. Very poor 4. Poor 3. Fair 2. Good 1. Very good ⃝ ⃝ ⃝ ⃝ ⃝

Sleep Impairment Survey In the past 7 days, I had a hard time getting things done because I was sleepy. [Trouble being productive]

1. Not at all 2. A little bit 3. Somewhat 4. Quite a bit 5. Very much ⃝ ⃝ ⃝ ⃝ ⃝

In the past 7 days, I felt alert when I woke up. [Alert]

5. Not at all 4. A little bit 3. Somewhat 2. Quite a bit 1. Very much ⃝ ⃝ ⃝ ⃝ ⃝

In the past 7 days, I felt tired. [Tiredness]

1. Not at all 2. A little bit 3. Somewhat 4. Quite a bit 5. Very much ⃝ ⃝ ⃝ ⃝ ⃝

In the past 7 days, I had problems during the day because of poor sleep. [Having problems]

1. Not at all 2. A little bit 3. Somewhat 4. Quite a bit 5. Very much ⃝ ⃝ ⃝ ⃝ ⃝

In the past 7 days, I had a hard time concentrating because of poor sleep. [Sleep impairment due to trouble concentrating]

1. Not at all 2. A little bit 3. Somewhat 4. Quite a bit 5. Very much ⃝ ⃝ ⃝ ⃝ ⃝

In the past 7 days, I felt irritable because of poor sleep. [Sleep impairment due to irritability]

1. Not at all 2. A little bit 3. Somewhat 4. Quite a bit 5. Very much ⃝ ⃝ ⃝ ⃝ ⃝

In the past 7 days, I was sleepy during the daytime. [Sleepy during daytime]

1. Not at all 2. A little bit 3. Somewhat 4. Quite a bit 5. Very much ⃝ ⃝ ⃝ ⃝ ⃝

In the past 7 days, I had trouble staying awake during the day. [Trouble staying awake]

1. Not at all 2. A little bit 3. Somewhat 4. Quite a bit 5. Very much ⃝ ⃝ ⃝ ⃝ ⃝

train/validation/test counts: 3944 / 537 / 1143

train/validation/test counts: 3943 / 537 / 1143

2000

1500

1500

1000

1000

500

500

0

0

1 2 3 4 5

1 2 3 4 5

Very restless

Satisfied

train/validation/test counts: 3939 / 536 / 1141

train/validation/test counts: 3943 / 537 / 1144

2000

2000

1500

1500

1000

1000

500

500

0

0

1 2 3 4 5

1 2 3 4 5

Refreshed

Trouble falling asleep

train/validation/test counts: 3944 / 537 / 1144

train/validation/test counts: 3942 / 535 / 1144

2000

2000

1500

1500

1000

1000

500

500

0

0

1 2 3 4 5

1 2 3 4 5

Trouble staying asleep

Trouble sleeping

train/validation/test counts: 3942 / 537 / 1144

train/validation/test counts: 3941 / 536 / 1142

2000

2000

1500

1000

1000

500

0

0

1 2 3 4 5

1 2 3 4 5

Counts

Enough sleep

Quality

train/validation/test counts: 3824 / 528 / 1108

train/validation/test counts: 3824 / 529 / 1105

1500

2000

1500

1000

1000

500

500

0

0

1 2 3 4 5

1 2 3 4 5

Trouble being productive

Alert

train/validation/test counts: 3826 / 528 / 1108

train/validation/test counts: 3821 / 529 / 1107

1500

1500

1000

1000

500

500

0

0

1 2 3 4 5

1 2 3 4 5

Tiredness

Having problems

train/validation/test counts: 3823 / 529 / 1107

train/validation/test counts: 3823 / 527 / 1108

1500

1500

1000

1000

500

500

0

0

1 2 3 4 5

1 2 3 4 5

SI due to trouble concentrating

SI due to irritability

train/validation/test counts: 3828 / 528 / 1107

train/validation/test counts: 3819 / 526 / 1107

2000

2000

1500

1000

1000

500

0

0

1 2 3 4 5

1 2 3 4 5

Sleepy during daytime

Trouble staying awake

- Figure C.2: Distribution of responses for each survey question. Survey questions are answered on a Likert scale. Here we show the distribution of responses for each question. The bar highlighted in a darker tone indicates those answers that were labeled as positive cases in the defined binary traits. The training, validation and test set counts are included in the title of each subplot.

###### C.3 Patient-reported outcome prediction performance.

AUROC for binary survey targets

Trouble being productive

SI due to irritability

Trouble staying awake

Sleepy during daytime

SI due to trouble concentrating

Tiredness

Having problems

Very restless

Enough sleep

Trouble sleeping

Trouble falling asleep

Quality

Alert

Satisfied

Trouble staying asleep

Refreshed

0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8

AUROC

(a) Performance of PH-LLM measured via AUROC.

AUPRC for binary survey targets

Model

PH-LLM w/ Adapter LogReg PH-LLM Few-Shot PH-LLM Zero-Shot Prevalence

Trouble being productive

| |
|---|

| |
|---|

SI due to irritability

| |
|---|

| |
|---|

Trouble staying awake

| |
|---|

Sleepy during daytime

SI due to trouble concentrating

Tiredness

Having problems

Very restless

Enough sleep

Trouble sleeping

Trouble falling asleep

Quality

Alert

Satisfied

Trouble staying asleep

Refreshed

0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40

AUPRC

(b) Performance of PH-LLM measured via AUPRC.

- Figure C.3: Prediction of Patient-Reported Outcomes. We evaluated the ability for the PH-LLM to infer subjective patient-reported outcomes using a multimodal adapter and compare to a suite of logistic regression models trained to predict each task independently, as well as PH-LLM using zero- and few-shot text prompting.

- Table C.18: Differences in AUROC between PH-LLM using a multimodal adapter and other modeling approaches. Here, we highlight values in bold where the difference between PH-LLM with adapter and the other approach were not statistically significant. Statistical significance was determined via a paired bootstrap estimator.

PH-LLM zero-shot PH-LLM few-shot Logistic Regression

Very restless 0.087 0.090 0.034 Satisfied 0.162 0.173 -0.029 Refreshed 0.114 0.101 -0.005 Trouble falling asleep 0.160 0.051 0.001 Trouble staying asleep 0.052 -0.003 -0.022 Trouble sleeping 0.161 0.069 -0.002 Enough sleep 0.306 -0.000 -0.006 Quality 0.167 0.184 0.009 Trouble being productive 0.260 0.129 0.007 Alert 0.130 0.164 0.002 Tiredness 0.221 0.122 0.027 Having problems 0.245 0.058 -0.005 SI due to trouble concentrating 0.268 0.114 0.009 SI due to irritability 0.207 0.084 0.012 Sleepy during daytime 0.296 0.095 0.004 Trouble staying awake 0.262 0.118 0.019

- Table C.19: Differences in AUPRC between PH-LLM using a multimodal adapter and other modeling approaches. Here, we highlight values in bold where the difference between PH-LLM with adapter and the other approach were not statistically significant. Statistical significance was determined via a paired bootstrap estimator.

PH-LLM zero-shot PH-LLM few-shot Logistic Regression

Very restless 0.077 0.073 -0.004 Satisfied 0.108 0.103 -0.019 Refreshed 0.067 0.064 0.005 Trouble falling asleep 0.067 0.035 -0.018 Trouble staying asleep 0.031 0.021 -0.006 Trouble sleeping 0.104 0.054 0.001 Enough sleep 0.080 -0.004 -0.021 Quality 0.037 0.044 0.009 Trouble being productive 0.118 0.076 -0.003 Alert 0.078 0.103 -0.009 Tiredness 0.155 0.105 0.021 Having problems 0.101 0.035 -0.021 SI due to trouble concentrating 0.127 0.073 0.008 SI due to irritability 0.082 0.053 -0.011 Sleepy during daytime 0.138 0.066 -0.026 Trouble staying awake 0.118 0.080 0.003

C.4 Patient-reported outcome prompt examples

- Table C.20 | Example of prompt given to PH-LLM to score PROs. Demographic and sensor values are passed as text to the model. The feature to predict (in this example, very restless) can then be scored using the potential completions “yes.” or “no.” For few-shot prompting we additionally prepend complete examples from the training set to the prompt. When using the multimodal adapter (see Methods), a vector representation of the quantitative data is passed in via a set of learned tokens. Values in the below prompt are synthetic.

Example of prompt given to PH-LLM to score PROs.

Use the information provided to predict “very restless”. age: [40-45]. heart rate variability root mean square of successive differences: 33.5. respiratory rate breaths per minute: 16.5. resting heart rate beats per minute: 60.0. awake minutes: 51.0. deep sleep minutes: 53.0. sleep duration minutes: 471610.0. efficiency: 0.85. overall sleep score: 81.0. percent of sleep in REM: 16.0. restlessness: 0.07. revitalization score: 83.2. sleep end time: -274.0. sleep start time: 364.0. sleep time minutes: 420.8. total number of minutes in awake state after sleep onset: 7.4. number of steps walked during the day: 6850.0. number of minutes spent in cardio zone during the day: 6.7. number of minutes spent in fat burn zone during the day: 18.9. number of minutes spent in peak zone during the day: 0.41. total multiplied minutes of exercise during the day: 45.32. very restless: yes or no?

