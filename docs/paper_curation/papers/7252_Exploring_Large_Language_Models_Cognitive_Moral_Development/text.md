# arXiv:2309.13356v2[cs.CL]7Oct2023

## Probing the Moral Development of Large Language Models through Defining Issues Test

#### Kumar Tanmay∗

Microsoft t-ktanmay@microsoft.com

#### Utkarsh Agarwal∗

Microsoft t-utagarwal@microsoft.com

#### Aditi Khandelwal∗

Microsoft t-aditikh@microsoft.com

Monojit Choudhury Microsoft monojitc@microsoft.com

### Abstract

In this study, we measure the moral reasoning ability of LLMs using the Defining Issues Test [1]- a psychometric instrument developed for measuring the moral development stage of a person according to the Kohlberg’s Cognitive Moral Development Model [2]. DIT uses moral dilemmas followed by a set of ethical considerations that the respondent has to judge for importance in resolving the dilemma, and then rank-order them by importance. A moral development stage score of the respondent is then computed based on the relevance rating and ranking. Our study shows that early LLMs such as GPT-3 exhibit a moral reasoning ability no better than that of a random baseline, while ChatGPT, Llama2-Chat, PaLM2 and GPT-4 show significantly better performance on this task, comparable to adult humans. GPT-4, in fact, has the highest post-conventional moral reasoning score, equivalent to that of typical graduate school students. However, we also observe that the models do not perform consistently across all dilemmas, pointing to important gaps in their understanding and reasoning abilities.

### 1 Introduction

The rapid paced developments and adoption of Large Language Models (LLMs) have led to fierce debates on the ethical concerns and potential harms that these models pose [3, 4, 5, 6], which include but are not limited to copyright, data and user privacy violations [7], linguistic inequality [8], hallucination [9, 10, 11], and toxic content generation [12]. The mainstream and most popular approaches to mitigate the harms related to the LLM-generated content, such as toxic, offensive [13], stereotyping, and exclusionary statements [14], and hate speech [15], have mainly involved alignment of model output to certain pre-determined values through techniques such as RLHF [16, 17], fair decoding [18], or post-processing/editing of the outputs [15, 19]. While these techniques are effective in achieving the underlying alignment goals [20], the goals themselves are often difficult, if not impossible, to define. This is because the ethical or moral values that must be upheld by a model or an AI system depend on the specific application, the user, the usage context, the cultural and geographical context, language and many other factors. In other words, it is impossible to design a universally-aligned LLM.

The problem of alignment becomes further complicated due to value pluralism – a condition where different moral values are in conflict with each other and any choice made by the model will have to jeopardize one value in favor of another [21, 22]. Philosophers capture this idea through “moral

∗These authors contributed equally to this work.

Preprint. Under review.

dilemmas" – situations that require one to choose one value over another to arrive at a resolution [23]. In fact, it would not be an overstatement to say that most real world situations involve some kind of value pluralism that requires one to chose between conflicting values. Thus, as LLMs become more ubiquitous and power various everyday applications, they have to face and resolve moral dilemmas arising from value pluralism [21]. Many have argued, therefore, that LLMs should ideally be trained as generic ethical reasoners rather than aligned for certain specific values [24].

To what extent LLMs can carry out deep ethical reasoning, and how can we systematically probe this? In this paper, we borrow ideas from the field of moral psychology to test the ethical or moral understanding and reasoning abilities of several popular LLMs. More specifically, we use the Defining Issues Test (DIT) [25] which is based on Kohlberg’s Cognitive Moral Development Model [26], to assess the moral development stage of the LLMs. In this test, a moral dilemma is presented along with 12 different statements on ethical considerations; the respondent (in our case, the LLM) is asked to rank these statements in the order of importance for resolving the dilemma. The outcome of the test is a set of scores that tells about the respondent’s moral development stage.

We study seven prominent models: GPT-3 [27], GPT-3.5, GPT-4 [28], ChatGPTv1, ChatGPTv2, PaLM-2 [29] and Llama2-Chat (70B version) [30], with 5 moral dilemmas from DIT and 4 newly designed dilemmas that extend the cultural context and diversity of the probes and precludes the possibility of training data contamination. We observe that GPT-4 achieves the highest moral development score in the range of that of a graduate school student, which according to Kohlberg’s model of cognitive moral development indicates a post-conventional moral understanding. GPT-3, on the other hand, performs no better than a random baseline. Performance of other models lie in between these two extremes, that roughly corresponds to the score range of adult humans and college students on DIT, and indicates a conventional moral understanding (as dictated by the moral norms and conventions of the society). Interestingly, for 2 of the 9 dilemmas, no model performs better than the random baseline, and for one of the newly designed dilemmas, GPT-4 performs worse than most other models. This shows that there is a lack of consistency in ethical reasoning across these models, implying the need for deeper investigation, understanding and improvement of LLMs’ moral reasoning abilities. This work also leads to several interesting technical, practical and philosophical questions, which are discussed in the last section.

### 2 Background and Related Work

In this section, we provide an overview of Morality, Moral Psychology and models of Cognitive Moral Development, from which we draw inspirations and materials to design this study. We also discuss current treatment of ethics in NLP literature, with a particular focus on LLMs.

#### 2.1 Morality and Moral Development

Morality is the study of what is right and wrong, and has been a central concern in philosophy [31]. Over the years, numerous theories have been proposed to explain how individuals develop their moral reasoning and judgments. Of these, the Cognitive Moral Development (CMD) model [2] proposed by Lawrence Kohlberg in 1969 remains one of the most influential accounts of moral development. Building upon Piaget’s work [32], Kohlberg developed a comprehensive theory that consists of six stages divided into three main levels: pre-conventional, conventional, and post-conventional morality.

At Stage 1, individuals are concerned with avoiding punishment and make moral decisions based on fear of consequences and self-interest. At Stage 2, individuals focus on their own needs and interests but recognize that others have similar needs. Moral judgments are influenced by reciprocity, such as “You scratch my back, I’ll scratch yours". Stages 1 and 2 are pre-conventional morality. At Stage 3, individuals seek approval and conform to social (and religious) norms. Moral decisions are made to maintain positive relationships and avoid disapproval. At Stage 4, individuals are concerned with law, rules, and authority figures and their moral reasoning revolves around maintaining social order and upholding the greater good. These two stages fall under the realm of conventional morality. At Stage 5, individuals recognize different groups may have different moral perspectives and base their decisions on principles of fairness, justice, and individual rights, even if these principles conflict with social norms or laws. This stage is further divided into sub-stages - 5A and 5B. Stage 5A suggests that moral obligation derives from voluntary commitments of society’s members to cooperate whereas Stage 5B is more concerned with procedures which exists for selecting laws that maximize welfare

as discerned in the majority will. At Stage 6, individuals develop their moral principles based on universal ethical values. They act according to a personal ethical code that transcends societal rules and laws. These principles often align with the concepts of justice, equality, and human rights. Stages 5A, 5B and 6 are, thus, called post-conventional morality.

The CMD model emphasizes the importance of moral reasoning and the development of an individual’s moral principles. It posits that as individuals mature, their moral reasoning becomes more sophisticated and abstract, allowing them to make ethical decisions based on principles rather than mere rules. It may be noted that this theory has been criticized for bias towards individualistic and self-expressionistic cultures (mostly prevalent in the Global North), overlooking the diversity of moral development across cultures [33, 34], for having gender bias [35], and for ignoring the role of intuitions and emotions in moral decision making [36]. Despite these criticisms, Kohlberg’s theory has played a vital role in advancing our understanding of moral development and remains influential in the field of moral psychology.

#### 2.2 Rest’s Defining Issues Test

In line with Kohlberg’s framework, James Rest introduced the Defining Issues Test (DIT) [1] as a way to measure an individual’s moral development. In this test the respondents are presented with moral dilemmas, and their moral reasoning abilities are assessed by analyzing the justifications provided by them for their decisions. Rest’s DIT draws upon Kohlberg’s stages to categorize individuals into stages of moral development, offering insights into ethical decision-making processes. For over three decades, the DIT has remained the most popular tool for assessing CMD.2. It includes either three (short-form DIT) or six (original DIT) moral dilemmas, each followed by 12 ethical considerations corresponding to different stages of CMD. The respondent has to first provide a resolution to the dilemma (it has three options: two horns of the dilemma and “can’t decide") and then rate the significance (“great", “much", “some", “little" and “no importance") of each item in resolving the moral dilemma, and then select and rank the four most important items.

The ethical consideration statements can also belong to A or M categories instead of the stages of CMD [25]. The A items are intended to typify an “anti-establishment" orientation, a point of view which condemns tradition and the existing social order. The M items are meant to be meaningless nonsense statements. The “M" statements were added as a reliability check as any valid respondent would be expected to rate the statement quite low, while for the purposes of any study, the “A" statements and it’s score are simply disregarded.

The Post Conventional Morality Score (abbreviated as P-score), stands as the most widely utilized metric, serving as an indicator of the “relative significance an individual places on principled moral considerations, specifically those associated with Stages 5 and 6, when deliberating moral dilemmas" [25]. If the most vital (top ranked) statement corresponds to either Stage 5 or 6, four points are added to the P-score. Similarly, if the second, third and fourth ranked statements belong to these post-conventional stages, three, two and one points are added respectively to the P-score. Thus, higher the P-score of a respondent, more the importance they pay to universal ethical values and human rights while making moral judgments.

Apart from P-score, DIT also measures Personal Interest Schema Score which reflects choices influenced by personal interests (Stages 2 and 3 in Kohlberg’s model), and Maintaining Norms Schema Score that indicates choices driven by societal norms, including legal systems, established roles, and organizational structures. The percentage of “can’t decide" choices measures the respondent’s decisiveness, reflecting the ease of processing moral information.

The Moral Judgment Test (MJT) [38], developed by Georg Lind to assess one’s moral judgment competencies, is also based on Kohlberg’s CMD. However, it measures the degree to which one can consistently employ the same moral value across moral dilemmas rather than the stage of moral development.

2Between 1974 and 1988, an estimated 400 studies have used DIT. It has been used in over 40 countries, across various professions and with about 150 new studies each year [37]

#### 2.3 Recent Theories in Moral Philosophy

In recent years, moral philosophy has seen the emergence of innovative theories developed by social psychologists, that expand our understanding of moral decision-making. Moral Foundations Theory [39], proposed by Jonathan Haidt and Jesse Graham, posits that human morality is shaped by a set of innate moral foundations or intuitions. These foundations include care/harm, fairness/cheating, loyalty/betrayal, authority/subversion, and sanctity/degradation. According to this theory, individuals vary in the extent to which they prioritize these moral foundations, leading to differences in moral judgments and values. Dual Process Theory [40], rooted in psychology and neuroscience, posits that moral decision-making involves two cognitive processes: System 1 (intuitive) and System 2 (reflective). System 1 operates quickly and automatically, relying on gut feelings and emotions, while System 2 involves deliberate reasoning and critical thinking. This theory suggests that moral judgments often result from the interplay between these two systems, and the balance can vary among individuals and situations. Though beyond the scope of our current study, these theories can provide novel frameworks for assessing the ethical reasoning abilities of LLMs.

#### 2.4 Current Approaches to Ethics of LLMs

AI alignment is a research field that aims to ensure that AI systems advance the intended goals, preferences, or ethical principles of humans [41]. Numerous scholarly works have contributed significantly to the development of ethical frameworks, principles, guidelines, methodologies, and tools essential for the responsible and ethical design, evaluation, and deployment of LLMs. Additionally, some datasets have been curated for the explicit purpose of training and assessing LLMs in their comprehension of ethical considerations, societal contexts, and norms, as well as their capacity to analyze these complex scenarios [42, 43, 44, 45, 46]. These studies have shed light on the notable ability of LLMs to understand and elucidate toxic content. However, it is important to underscore a salient limitation within these investigations, namely, the inherent bias embedded within the collected data. This bias stems from the geographical locations, cultural backgrounds, and political orientations of the annotators, casting a shadow on the universality of the findings [47].

Some recent works demonstrate how in-context learning [24] and supervised tuning [48, 49] can help aligning LLMs with moral instructions. These works aim to ensure that LLMs respect human values and norms, such as fairness, accountability, transparency, privacy, safety, etc. They also suggest ways to identify, measure, mitigate, and prevent the potential harms of LLMs to individuals and society. Some of these works propose ethical datasets [49] and guidelines [50, 51] to help researchers and practitioners assess and improve the ethical capabilities of LLMs.

However, ethics is not a monolithic or universal concept. Different people may have different ethical views, beliefs, values, preferences, etc. depending on their cultural, social, religious, and political backgrounds [52, 53, 21]. Therefore, it is important to acknowledge and respect the diversity and pluralism of human ethics and values when developing and using LLMs. This means that LLMs should not impose or favor a single or dominant ethical perspective or value system over others but rather allow for multiple and diverse ethical perspectives and value systems to coexist and interact.

Ethical issues often involve shades of gray and require nuanced reasoning that cannot be adequately captured with a binary decision. Most of the current approaches to AI alignment fail to capture the multifaceted nature of ethical reasoning. Ethical decisions often involve multiple dimensions, including fairness, justice, harm, and cultural context, which may not be fully addressed in a binary setup. Binary choices may lack explanatory power. They don’t provide insights into why a model made a particular ethical decision, making it challenging to assess the quality of its ethical reasoning. It may not adequately capture the complexities of ethical trade-offs. In real-world scenarios, ethical decisions often involve weighing competing values, which binary tasks may not address effectively.

### 3 Data and Method

In this section, we describe our experimental setup, the datasets, LLMs tested, prompt structure and metrics. We present the LLMs with a prompt that contains the moral dilemma along with the 12 ethical considerations followed by three questions. Based on the responses to these questions, we compute the P-score and individual stage scores for each LLM.

#### 3.1 Dataset

We used five dilemmas from DIT-13 and constructed four novel moral dilemmas. Each author designed one dilemma (story and the ethical consideration statements) similar in structure to the original DIT dilemmas. The statements of each dilemma were then independently annotated by all the authors for the Kohlberg’s CMD stages that they represent. Cases of disagreements were discussed and if for a statement no clear consensus was reached, the statement was edited or redesigned to avoid any ambiguity. A brief summary of the dilemmas are described below, and Appendix A presents the four new dilemmas.

The complete DIT-1 consists of six dilemmas: (1) Heinz dilemma - Should Heinz steal a drug from an inventor in town to save his wife who is dying and needs the drug?, (2) Newspaper dilemma Should a student newspaper be stopped by a Principal of a high school when the newspaper stirs controversy in the community?, (3) Student dilemma - Should students take over an administration building in protest of the Vietnam war?, (4) Webster dilemma - Should a minority member be hired for a job when the community is biased?, (5) Prisoner dilemma - Should a man who escaped from prison but has since been leading an exemplary life be reported to authorities? and (6) Doctor dilemma - Should a doctor give an overdose of pain-killer to a suffering patient?

The four novel moral dilemmas are: (1) Monica’s Dilemma - Should Monica give the first authorship to Aisha despite having the major contribution?, (2) Timmy’s Dilemma - Should Timmy attend his friend’s wedding instead of fixing an urgent bug that could put customers’ privacy at risk?, (3) Rajesh’s Dilemma - Should Rajesh rent a house by hiding the secret of his non-vegetarian consumption at home from the vegetarian neighborhood? and (4) Auroria Dilemma - Should the country Auroria share its innovations and resources to it’s poor neighbor or profit off it’s huge investments in research?

The dilemmas are associated with conflicting values such as interpersonal vs. societal (Heinz dilemma), interpersonal vs. professional (Timmy’s and Monica’s dilemmas), and community vs. personal values placed in diverse cultural and situational contexts. We exclude the Doctor’s dilemma from all experiments as most LLMs do not generate a response for it, presumably due to their content filtering policies.

#### 3.2 Experimental Setup

We study seven popular LLMs: GPT-4 (size undisclosed), PaLM-2 (size undisclosed), ChatGPT (July 2023) (henceforth referred to as ChatGPTv2, 175B params), ChatGPT (December 2022) (henceforth referred to as ChatGPTv1, 175B params), GPT-3.5 (text-davinci-003)(175B params), GPT-3 (175B params) and Llama2-Chat (70B params). All these models are trained on massive amounts of text data from various sources and domains and have different training methods and capabilities.

- Figure 1 shows the prompt structure. The text in black are fixed, whereas those in blue are dilemma specific. Since LLMs might have positional bias while ranking the ethical consideration statements for a dilemma, or in choosing one of the three options (O1, O2 and O3) as a resolution for the dilemma, we consider 8 different predefined permutations of the 12 statements (out of 12! possibilities) and all, i.e., 6, permutations of the options. This amounts to 48 distinct prompts per dilemma. For all experiments, we set temperature to 0, presence penalty to 1, top_p to 0.95, and max_tokens to 2000 (except GPT-3 where it is set at 1000 due it’s smaller context length).

#### 3.3 Metrics

We used the metric P-score, henceforth pscore, as proposed by the DIT authors which indicates the "relative importance a subject gives to principled moral considerations (Stages 5 and 6)". pscore is calculated by assigning points to the four most important statements the respondent (the LLM in our case) has selected that correspond to the post conventional stages. 4, 3, 2 and 1 points are added to the score if the first, second, third and fourth ranked statements belong to Stage 5 or 6 respectively. The final score is obtained by multiplying the sum by 10. As an illustration, suppose that the model predicts 12, 7, 3, 9 as the most important statements of consideration in descending order, of which only items 12 and 3 belong to the post-conventional stages. Then, the pscore will be 10·(4+2) = 60.

3DIT-1 dilemmas are not freely available; we purchased the dataset from The University of Alabama through the official website: https://ethicaldevelopment.ua.edu/ordering-information.html

|[Figure 1]|
|---|

Figure 1: Prompt structure illustrated for the Monica’s Dilemma.

Similarly, we also calculate stage-wise scores, scoreθ, as

scoreθ = 10 ·

4

((5 − i) · Si,θ) where Si,θ =

i=1

1 if ith ranked statement is from Stage-θ 0 otherwise

(1)

Thus, pscore = score5 + score6. We also compute the random baseline scores for each dilemma, i.e., the score a respondent will receive on average if they were ranking the items randomly. These baseline numbers depend only on the number of items that belong to a certain stage for a dilemma. Heinz, Prisoner and Newspaper dilemmas have 3 items in Stages 5 and 6, giving a random baseline pscore of 25. All other dilemmas have 4 items in Stages 5 and 6, and a random baseline pscore of 33.33. Thus, the average random pscore over all dilemmas is 30.56.

The maximum possible pscore is 90 for the Heinz, Prisoner and Newspaper dilemmas and 100 for the others. Thus, the pscore averaged on all dilemmas ranges from 0 to 96.67. Higher the pscore, deeper the moral understanding and better the moral reasoning ability of a model (or equivalently, of a human respondent). Various surveys conducted on human subjects using DIT [25] report a pscore of around 20 and 30 for junior and senior high school children respectively (mostly pre-conventional stage), between 40 and 46 for college students as well as average adults (mostly at the conventional stage), and between 53 and 63 for graduate school students (early post-conventional stage).

### 4 Results and Observations

The results of our experiments are summarized in two plots: Fig. 2 shows the pscore for each LLM as violin plots grouped by dilemmas. Fig. 3a shows the stage-wise scores for the LLMs averaged over all dilemmas; this provides insight into the overall performance and staging of the models. The three key observations from these results are as: (a) Overall, GPT-3 has the lowest and close to random pscore, while GPT-4 has the highest pscore; the other models in ascending order of pscore are: GPT-3.5, ChatGPTv2, PaLM-2, Llama2-Chat, ChatGPTv1. Our study shows that except for GPT-3, all models investigated have a pscore equivalent to an average adult human or college student; only GPT-4 achieves a pscore (= 55.68) in the range of a graduate student and shows post-conventional moral reasoning abilities. (b) All models perform poorly on the Prisoner and Webster dilemmas, while most models perform well on the Timmy and Newspaper dilemmas; and (c) There is significant variability in the responses of all the models over different runs (as shown by the violin plots), as well

- as specific dilemmas where they perform exceptionally well (e.g., GPT-4 on Newspaper dilemma) or poorly (e.g., GPT-4 on Rajesh’s dilemma).

[Figure 2]

- Figure 2: Dilemma wise pscore comparison across LLMs. The dotted line shows the random baseline pscore for the dilemma.

Fig 3b shows the resolutions proposed by the models for each dilemma. Two interesting observations emerge from it: (a) All models agree perfectly for the Webster dilemma. A majority of models agree for the Heinz, Newspaper, Rajesh and Auroria dilemmas. (b) Contrary to other models, ChatGPTv2, does not favor any particular resolution (except in Webster). In the subsequent paragraphs, we present model-specific observations.

- GPT-3. The prompt structure described in Fig. 1 did not work with GPT-3, as the model failed to generate any cogent response. Through trial-and-error, we constructed a prompt where only the resolution of the moral dilemma and the selection of top four statements (out of 12) were asked for, which seemed to work for the model. Even then, we observed that it frequently ranks the statements

- at position 1, 3, 5 and 7 as most significant options, irrespective of the stages the sentences belonged

to. This explains why the average pscore for GPT-3, 29.84, is close to that of the random baseline. In conclusion, GPT-3 is incapable of moral reasoning and also, of following complex multistage instructions. Incidentally, we also tested text-davinci-002, but could not make it generate cogent responses. Therefore, the model is excluded from the study.

GPT-3.5, ChatGPT (both v1 & v2) and GPT-4 demonstrate a greater ability of understanding the instructions, presumably due to the RLHF training. Therefore, these models responded consistently to the prompt questions, and also perform significantly better than the random baseline. We observe a general trend that the bigger and the newer models have higher pscore, except for ChatGPTv2 that has a slightly lower pscore than its previous version ChatGPTv1. Incidentally, there are anecdotal (but contested) claims [54] that the performance of ChatGPT is degrading over time as newer versions are being released, which is consistent with our observation. With a pscore of 55.68, GPT-4 is the only model that clearly shows post-conventional moral reasoning abilities equivalent of graduate students.

Llama2-Chat, even though a much smaller model compared to GPT-3.x series, achieves an unexpectedly high pscore which is less than only GPT-4 and ChatGPTv1. This points to the possibility of

[Figure 3]

(a) Stage-wise scores comparison of different models.

| | | | | | | |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |
| | | | | | | |

Heinz

Student

Newspaper

Webster

Prisoner

Timmy

Rajesh

Monica

Auroria

GPT-3.5 ChatGPTv2Llama2-ChatChatGPTv1 GPT-4 PaLM-2

(b) Dilemma-wise models’ resolution for the dilemma.

- Figure 3: Model-wise scores and their dilemma-wise resolutions. PaLM-2 results are from 8 dilemmas (Sec. 4). In Fig-(b), the colors’ RGB components depict the fraction of runs with corresponding resolutions (Green - O1(Should do), Blue - O2(Can’t Decide), Red - O3(Shouldn’t do))

building smaller models with strong moral reasoning capabilities. PaLM-2 exhibited superior moral reasoning capability with a pscore of 52.24. However, it did not generate a response to the Prisoner dilemma. Therefore, the total pscore is averaged over 8 instead of 9 dilemmas. When averaged over the same 8 dilemmas, the pscore of the other models are (in descending order): GPT-4 – 58.81, ChatGPTv1 – 56.44, Llama2-Chat – 52.85, ChatGPTv2 – 51.55, GPT-3.5 – 49.48 and GPT-3 – 31.20. Thus, PaLM-2 performs worse than GPT-4 and ChatGPTv1, but is comparable to Llama2-Chat and ChatGPTv2. Note that the average pscore is significantly higher for all the models when Prisoner dilemma is removed from the set because all models perform poorly on this dilemma.

### 5 Discussion and Conclusion

In this study, we propose an effective evaluation framework to measure the ethical reasoning capability of LLMs based on Kohlberg’s Cognitive Moral Development model and Defining Issues Test. Apart from the 6 moral dilemmas included in DIT-1, we propose 4 novel dilemmas partly to expand the socio-cultural contexts covered by the dilemmas, and partly to ensure that the LLMs were not already exposed to them. Our study shows that GPT-4 exhibits post-conventional moral reasoning abilities at the level of human graduate students, while other models like ChatGPT, LLama2-Chat and PaLM-2 exhibit conventional moral reasoning ability equivalent to that of an average adult human being or college student.

We are aware of several limitations of this study, including the known criticisms of the DIT framework [55, 56], that provides us with enough reasons not to take the numbers at their face value. More investigation is necessary to firmly establish the moral reasoning abilities and limitations of LLMs. Nevertheless, it is interesting to ponder on some of the repercussions of these findings. While one could explain the conventional moral reasoning abilities observed in the LLMs as an effect of the training data [57] at pre-training , instruction fine-tuning and RLHF phases, which certainly contains several instances of conventionalized and codified ethical values, one wonders how an LLM (e.g,

- GPT-4 ) could exhibit post-conventional moral reasoning abilities. Since the training data and the architectural details of GPT-4 are undisclosed, one can only speculate the reasons. Either the data (most likely the one used during RLHF) consisted of many examples of post-conventional moral reasoning, or it is an emergent property of the model. In the latter case, a deeper philosophical question that arises is whether moral reasoning can emerge in LLMs, and if so, whether it is just a special case of general reasoning ability.

There are other open problems around the dilemmas and types of moral questions where the current models are lagging (e.g., Prisoner and Webster dilemma), what makes these dilemmas difficult, and

how can we train models with the specific objective of improving their moral reasoning capability. One might also ask that since many of the models, especially GPT-4, is as good or better than an average adult human in terms of their moral development stage scoring, does it then make sense to leave the everyday moral decision making tasks to LLMs. In the future, if and when we are able to design LLMs with pscore higher than expert humans (e.g., lawyers and justices), should we replace judges and jury members by LLMs?

### References

- [1] J. Rest. Development in Judging Moral Issues. University of Minnesota Press, Minneapolis, MN, 1979.
- [2] Lawrence Kohlberg. The philosophy of moral development: Essays on moral development. San Francisco, 1981.
- [3] Laura Weidinger, Jonathan Uesato, Maribeth Rauh, Conor Griffin, Po-Sen Huang, John Mellor, Amelia Glaese, Myra Cheng, Borja Balle, Atoosa Kasirzadeh, Courtney Biles, Sasha Brown, Zac Kenton, Will Hawkins, Tom Stepleton, Abeba Birhane, Lisa Anne Hendricks, Laura Rimell, William Isaac, Julia Haas, Sean Legassick, Geoffrey Irving, and Iason Gabriel. Taxonomy of risks posed by language models. In 2022 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’22, page 214–229, New York, NY, USA, 2022. Association for Computing Machinery.
- [4] Laura Weidinger, John Mellor, Maribeth Rauh, Conor Griffin, Jonathan Uesato, Po-Sen Huang, Myra Cheng, Mia Glaese, Borja Balle, Atoosa Kasirzadeh, Zac Kenton, Sasha Brown, Will Hawkins, Tom Stepleton, Courtney Biles, Abeba Birhane, Julia Haas, Laura Rimell, Lisa Anne Hendricks, William Isaac, Sean Legassick, Geoffrey Irving, and Iason Gabriel. Ethical and social risks of harm from language models, 2021.
- [5] Rebecca L Johnson, Giada Pistilli, Natalia Menédez-González, Leslye Denisse Dias Duran, Enrico Panai, Julija Kalpokiene, and Donald Jay Bertulfo. The ghost in the machine has an american accent: value conflict in gpt-3, 2022.
- [6] Patrick Schramowski, Cigdem Turan, Nico Andersen, Constantin A. Rothkopf, and Kristian Kersting. Large pre-trained language models contain human-like biases of what is right and wrong to do, 2022.
- [7] Siwon Kim, Sangdoo Yun, Hwaran Lee, Martin Gubri, Sungroh Yoon, and Seong Joon Oh. Propile: Probing privacy leakage in large language models. arXiv preprint arXiv:2307.01881, 2023.
- [8] Monojit Choudhury and Amit Deshpande. How linguistically fair are multilingual pre-trained language models? In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 12710–12718, 2021.
- [9] Yejin Bang, Samuel Cahyawijaya, Nayeon Lee, Wenliang Dai, Dan Su, Bryan Wilie, Holy Lovenia, Ziwei Ji, Tiezheng Yu, Willy Chung, et al. A multitask, multilingual, multimodal evaluation of chatgpt on reasoning, hallucination, and interactivity. arXiv preprint arXiv:2302.04023, 2023.
- [10] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, et al. Siren’s song in the ai ocean: A survey on hallucination in large language models. arXiv preprint arXiv:2309.01219, 2023.
- [11] Nick McKenna, Tianyi Li, Liang Cheng, Mohammad Javad Hosseini, Mark Johnson, and Mark Steedman. Sources of hallucination by large language models on inference tasks. arXiv preprint arXiv:2305.14552, 2023.
- [12] Xingxuan Li, Yutong Li, Linlin Liu, Lidong Bing, and Shafiq Joty. Is gpt-3 a psychopath? evaluating large language models from a psychological perspective. arXiv preprint arXiv:2212.10529, 2022.
- [13] Paula Fortuna, Juan Soler, and Leo Wanner. Toxic, hateful, offensive or abusive? what are we really classifying? an empirical analysis of hate speech datasets. In Proceedings of the 12th language resources and evaluation conference, pages 6786–6794, 2020.

- [14] Su Lin Blodgett, Gilsinia Lopez, Alexandra Olteanu, Robert Sim, and Hanna Wallach. Stereotyping norwegian salmon: An inventory of pitfalls in fairness benchmark datasets. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1004–1015, 2021.
- [15] Fabio Del Vigna12, Andrea Cimino23, Felice Dell’Orletta, Marinella Petrocchi, and Maurizio Tesconi. Hate me, hate me not: Hate speech detection on facebook. In Proceedings of the first Italian conference on cybersecurity (ITASEC17), pages 86–95, 2017.
- [16] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.
- [17] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.
- [18] Thomas Hartvigsen, Saadia Gabriel, Hamid Palangi, Maarten Sap, Dipankar Ray, and Ece Kamar. Toxigen: A large-scale machine-generated dataset for adversarial and implicit hate speech detection. arXiv preprint arXiv:2203.09509, 2022.
- [19] Shaoxiong Ji, Shirui Pan, Xue Li, Erik Cambria, Guodong Long, and Zi Huang. Suicidal ideation detection: A review of machine learning methods and applications. IEEE Transactions on Computational Social Systems, 8(1):214–226, 2020.
- [20] Jing Yao, Xiaoyuan Yi, Xiting Wang, Jindong Wang, and Xing Xie. From instructions to intrinsic human values–a survey of alignment goals for big models. arXiv preprint arXiv:2308.12014, 2023.
- [21] Taylor Sorensen, Liwei Jiang, Jena Hwang, Sydney Levine, Valentina Pyatkin, Peter West, Nouha Dziri, Ximing Lu, Kavel Rao, Chandra Bhagavatula, et al. Value kaleidoscope: Engaging ai with pluralistic human values, rights, and duties. arXiv preprint arXiv:2309.00779, 2023.
- [22] William James. The moral philosopher and the moral life. The International Journal of Ethics, 1(3):330, 1891.
- [23] Michael Slote. Utilitarianism, moral dilemmas, and moral cost. American Philosophical Quarterly, 22(2):161–168, 1985.
- [24] Jingyan Zhou, Minda Hu, Junan Li, Xiaoying Zhang, Xixin Wu, Irwin King, and Helen Meng. Rethinking machine ethics–can llms perform moral reasoning through the lens of moral theories? arXiv preprint arXiv:2308.15399, 2023.
- [25] J.R. Rest and University of Minnesota. Center for the Study of Ethical Development. DIT Manual: Manual for the Defining Issues Test. Center for the Study of Ethical Development, University of Minnesota, 1990.
- [26] Cheryl E Sanders. Lawrence Kohlberg’s stages of moral development. technical report, April 2023.
- [27] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.
- [28] OpenAI. Gpt-4 technical report, 2023.
- [29] Rohan Anil, Andrew M. Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu, Jonathan H. Clark, Laurent El Shafey, Yanping Huang, Kathy Meier-Hellstern, Gaurav Mishra, Erica Moreira, Mark Omernick, Kevin Robinson, Sebastian Ruder, Yi Tay, Kefan Xiao, Yuanzhong Xu, Yujing Zhang, Gustavo Hernandez Abrego, Junwhan Ahn, Jacob Austin, Paul Barham, Jan Botha, James Bradbury, Siddhartha Brahma, Kevin Brooks, Michele Catasta, Yong Cheng, Colin Cherry, Christopher A. Choquette-Choo, Aakanksha Chowdhery, Clément Crepy, Shachi Dave, Mostafa Dehghani, Sunipa Dev, Jacob Devlin, Mark Díaz, Nan Du, Ethan Dyer, Vlad Feinberg, Fangxiaoyu Feng, Vlad Fienber, Markus Freitag, Xavier Garcia, Sebastian Gehrmann, Lucas Gonzalez, Guy Gur-Ari, Steven Hand, Hadi Hashemi, Le Hou, Joshua Howland, Andrea Hu, Jeffrey Hui, Jeremy Hurwitz, Michael Isard, Abe Ittycheriah, Matthew Jagielski, Wenhao Jia,

- Kathleen Kenealy, Maxim Krikun, Sneha Kudugunta, Chang Lan, Katherine Lee, Benjamin Lee, Eric Li, Music Li, Wei Li, YaGuang Li, Jian Li, Hyeontaek Lim, Hanzhao Lin, Zhongtao Liu, Frederick Liu, Marcello Maggioni, Aroma Mahendru, Joshua Maynez, Vedant Misra, Maysam Moussalem, Zachary Nado, John Nham, Eric Ni, Andrew Nystrom, Alicia Parrish, Marie Pellat, Martin Polacek, Alex Polozov, Reiner Pope, Siyuan Qiao, Emily Reif, Bryan Richter, Parker Riley, Alex Castro Ros, Aurko Roy, Brennan Saeta, Rajkumar Samuel, Renee Shelby, Ambrose Slone, Daniel Smilkov, David R. So, Daniel Sohn, Simon Tokumine, Dasha Valter, Vijay Vasudevan, Kiran Vodrahalli, Xuezhi Wang, Pidong Wang, Zirui Wang, Tao Wang, John Wieting, Yuhuai Wu, Kelvin Xu, Yunhan Xu, Linting Xue, Pengcheng Yin, Jiahui Yu, Qiao Zhang, Steven Zheng, Ce Zheng, Weikang Zhou, Denny Zhou, Slav Petrov, and Yonghui Wu. Palm 2 technical report, 2023.
- [30] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
- [31] The definition of morality.
- [32] Jean Piaget. The moral judgment of the child. Routledge, 2013.
- [33] Dora Shu-Fang Dien. A chinese perspective on kohlberg’s theory of moral development. Developmental Review, 2(4):331–341, 1982.
- [34] John R Snarey. Cross-cultural universality of social-moral development: a critical review of kohlbergian research. Psychological bulletin, 97(2):202, 1985.
- [35] Muriel J Bebeau and Mary M Brabeck. Integrating care and justice issues in professional moral education: A gender perspective. Journal of moral education, 16(3):189–203, 1987.
- [36] Jonathan Haidt. The emotional dog and its rational tail: a social intuitionist approach to moral judgment. Psychological review, 108(4):814, 2001.
- [37] James R Rest et al. Moral development in the professions: Psychology and applied ethics. Psychology Press, 1994.
- [38] Georg Lind. An introduction to the moral judgment test (mjt). Unpublished manuscript. Konstanz: University of Konstanz. http://www. uni-konstanz. de/ag-moral/pdf/MJT-introduction. PDF, 1998.
- [39] Jesse Graham, Jonathan Haidt, Sena Koleva, Matt Motyl, Ravi Iyer, Sean P Wojcik, and Peter H Ditto. Moral foundations theory: The pragmatic validity of moral pluralism. In Advances in experimental social psychology, volume 47, pages 55–130. Elsevier, 2013.
- [40] Derek Tam, Anisha Mascarenhas, Shiyue Zhang, Sarah Kwan, Mohit Bansal, and Colin Raffel. Evaluating the factual consistency of large language models through summarization. arXiv preprint arXiv:2211.08412, 2022.
- [41] Kelsey Piper. The case for taking ai seriously as a threat to humanity, Oct 15, 2020.
- [42] Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, and Jacob Steinhardt. Aligning ai with shared human values. arXiv preprint arXiv:2008.02275, 2020.
- [43] Maxwell Forbes, Jena D. Hwang, Vered Shwartz, Maarten Sap, and Yejin Choi. Social chemistry 101: Learning to reason about social and moral norms. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 653–670, Online, November 2020. Association for Computational Linguistics.
- [44] Denis Emelin, Ronan Le Bras, Jena D Hwang, Maxwell Forbes, and Yejin Choi. Moral stories: Situated reasoning about norms, intents, actions, and their consequences. arXiv preprint arXiv:2012.15738, 2020.
- [45] Nicholas Lourie, Ronan Le Bras, and Yejin Choi. Scruples: A corpus of community ethical judgments on 32,000 real-life anecdotes. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 13470–13479, 2021.
- [46] Maarten Sap, Saadia Gabriel, Lianhui Qin, Dan Jurafsky, Noah A. Smith, and Yejin Choi. Social bias frames: Reasoning about social and power implications of language. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 5477–5490, Online, July 2020. Association for Computational Linguistics.

- [47] Alexandra Olteanu, Carlos Castillo, Fernando Diaz, and Emre Kıcıman. Social data: Biases, methodological pitfalls, and ethical boundaries. Frontiers in big data, 2:13, 2019.
- [48] Liwei Jiang, Jena D Hwang, Chandra Bhagavatula, Ronan Le Bras, Jenny Liang, Jesse Dodge, Keisuke Sakaguchi, Maxwell Forbes, Jon Borchardt, Saadia Gabriel, et al. Can machines learn morality? the delphi experiment. arXiv preprint arXiv:2110.07574, 2021.
- [49] Dan Hendrycks, Collin Burns, Steven Basart, Andrew Critch, Jerry Li, Dawn Song, and Jacob Steinhardt. Aligning ai with shared human values, 2023.
- [50] Zeerak Talat, Hagen Blix, Josef Valvoda, Maya Indira Ganesh, Ryan Cotterell, and Adina Williams. On the machine learning of ethical judgments from natural language. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 769–779, Seattle, United States, July 2022. Association for Computational Linguistics.
- [51] Patrick Schramowski, Cigdem Turan, Sophie Jentzsch, Constantin Rothkopf, and Kristian Kersting. Bert has a moral compass: Improvements of ethical and moral values of machines, 2019.
- [52] William A Galston and William Arthur Galston. Liberal pluralism: The implications of value pluralism for political theory and practice. Cambridge University Press, 2002.
- [53] Mark C. Navin and Katie Attwell. Vaccine mandates, value pluralism, and policy diversity. Bioethics, 33(9):1042–1049, 2019.
- [54] Lingjiao Chen, Matei Zaharia, and James Zou. How is chatgpt’s behavior changing over time? arXiv preprint arXiv:2307.09009, 2023.
- [55] Richard M Martin, Michael Shafto, and William Vandeinse. The reliability, validity, and design of the defining issues test. Developmental Psychology, 13(5):460, 1977.
- [56] Stanley R Kay. Kohlberg’s theory of moral development: Critical analysis of validation studies with the defining issues test. International Journal of Psychology, 17(1-4):27–42, 1982.
- [57] Patrick Schramowski, Cigdem Turan, Sophie Jentzsch, Constantin Rothkopf, and Kristian Kersting. The moral choice machine. Frontiers in artificial intelligence, page 36, 2020.

### A Dilemmas

The dilemmas we have crafted and illustrated can be found in Figures 4 through 7.

[Figure 4]

- Figure 4: Story and 12 statements for Rajesh’s Dilemma

[Figure 5]

##### Figure 5: Story and 12 statements for Monica’s Dilemma

[Figure 6]

##### Figure 6: Story and 12 statements for Timmy’s Dilemma

[Figure 7]

##### Figure 7: Story and 12 statements for Auroria Dilemma

[Figure 8]

##### Figure 8: Radar Plot

