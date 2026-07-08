# arXiv:2406.16338v1[cs.CV]24Jun2024

## VideoHallucer: Evaluating Intrinsic and Extrinsic Hallucinations in Large Video-Language Models

Yuxuan Wang1,2 wangyuxuan1@bigai.ai

Yueqian Wang2,3 wangyueqian@.pku.edu

Dongyan Zhao2,3 zhaodongyan@pku.edu

Cihang Xie4 cixie@ucsc.edu

#### Zilong Zheng1,2

zlzheng@bigai.ai

- 1 Beijing Institute for General Artificial Intelligence, Beijing China
- 2 State Key Laboratory of General Artificial Intelligence, Beijing, China

3 Wangxuan Institute of Computer Technology, Peking University, Beijing, China 4 Computer Science and Engineering, University of California, Santa Cruz

https://VideoHallucer.github.io

### Abstract

Recent advancements in Multimodal Large Language Models (MLLMs) have extended their capabilities to video understanding. Yet, these models are often plagued by “hallucinations”, where irrelevant or nonsensical content is generated, deviating from the actual video context. This work introduces VideoHallucer, the first comprehensive benchmark for hallucination detection in large video-language models (LVLMs). VideoHallucer categorizes hallucinations into two main types: intrinsic and extrinsic, offering further subcategories for detailed analysis, including object-relation, temporal, semantic detail, extrinsic factual, and extrinsic non-factual hallucinations. We adopt an adversarial binary VideoQA method for comprehensive evaluation, where pairs of basic and hallucinated questions are crafted strategically. By evaluating eleven LVLMs on VideoHallucer, we reveal that i) the majority of current models exhibit significant issues with hallucinations; ii) while scaling datasets and parameters improves models’ ability to detect basic visual cues and counterfactuals, it provides limited benefit for detecting extrinsic factual hallucinations; iii) existing models are more adept at detecting facts than identifying hallucinations. As a byproduct, these analyses further instruct the development of our self-PEP framework, achieving an average of 5.38% improvement in hallucination resistance across all model architectures.

### 1 Introduction

Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in both visual understanding and language generation [Alayrac et al., 2022, Li et al., 2023a, Liu et al., 2023, OpenAI et al., 2024]. However, despite their strong performance on standard benchmarks [Antol et al., 2015, Lin et al., 2014, Xu et al., 2017, 2016, Krishna et al., 2017], these models frequently produce incorrect or unsubstantiated responses w.r.t. visual inputs [Li et al., 2023b, Tong et al., 2024, Petryk et al., 2024]. This issue, often referred to as “hallucination” [Rohrbach et al., 2018], means that MLLMs can generate irrelevant or nonsensical content that deviates from the original visual context. Given these challenges, a natural question arises: How can we examine the vulnerability to

Preprint. Under review.

###### Intrinsic Hallucination Extrinsic Hallucination

###### Factual Hallucination

###### Object-Relation Hallucination Temporal Hallucination

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

Basic Question: Does the video provide information that allows us to learn that …… responsibility, as these elements are crucial for your AI strategy? Answer: Yes Hallucinated Question: Does the video provide information that allows us to learn that …… responsibility in AI government policy? Answer: No Explanation: not mention government policy

Basic Question: Does ‘a man talks to a woman behind a glass wall’ happen later than ‘a man shows up with flowers at a woman's door’? Answer: Yes Hallucinated Question: Does 'a man

Basic Question: Is the cat above the sofa in the video? Answer: Yes Hallucinated Question: Is the cat below the sofa in the video? Answer: No

talks to a woman behind a glass wall' happen earlier than 'a man shows up with flowers at a woman's door'?

Answer: No

Non-factual Hallucination

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

Semantic Detail Hallucination

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Basic Question: Based on the video, should we cut up the garlics when we smash garlic? Answer: Yes Hallucinated Question: Based on the video, should we peel the bananas when we smash garlic? Answer: No

Hallucinated Question: Is there someone wearing a green hat in the video? Answer: No

Basic Question: Is there someone wearing a green hat in the video? Answer: Yes

- Figure 1: Example tasks in VideoHallucer. Each datapoint consists of a basic question, used to test the basic ability of LVLM, and a hallucinated question, containing hallucinated content to evaluate the models’ ability to detect hallucination. Intrinsic Hallucinations (left) occur when the responses are inconsistent with the original videos. Extrinsic hallucinations (right) happen when the responses cannot be confirmed by the source video. For extrinsic factual hallucination questions, we especially add explanation annotation for clarity and further exploration.

hallucinations in MLLMs? Addressing this need not only reveals the extent of hallucination in these models, but also helps identify underlying causes and develop methods to further enhance models.

To address this need, researchers have begun to benchmark hallucinations related to objects [Rohrbach et al., 2018, Li et al., 2023b], relationships [Han et al., 2024], and attributes [Sun et al., 2023, Wang et al., 2024a, Chen et al., 2023a, Wang et al., 2024b], as well as factual information [Cui et al., 2023, Jing et al., 2023, Chen et al., 2024]. However, most existing studies focus on hallucinations involving basic static visual attributes from images in large image-language models. They often overlook potential hallucination issues arising from dynamic content, such as actions, events, and stories, in large video-language models (LVLMs). Furthermore, video-language tasks, such as video summarization [Ghauri et al., 2020, Song et al., 2015], due to their higher complexity, are in lack of careful evaluation in existing datasets. As a result, we are still unsure of the extent, severity, characteristics, and causes of hallucination issues within LVLMs. Additionally, existing benchmarks focus on certain attributes of hallucinations, lacking a comprehensive and robust evaluation.

To tackle these issues, we introduce VideoHallucer, the first comprehensive benchmark designed to assess hallucination in LVLMs. Within VideoHallucer, we establish a clear taxonomy of hallucinations, distinguishing between two primary categories: intrinsic and extrinsic. Specifically, Intrinsic hallucinations involve generated content that directly contradicts information present in the source video, and can be categorised into three subtypes: object-relation, temporal, and semantic detail hallucinations. While extrinsic hallucinations involve content that cannot be verified from the source, and can be classified as either extrinsic factual, aligning with general knowledge but not present in the source video, or extrinsic non-factual, which includes all the others. To avoid confounding factors from LLMs [Li et al., 2023b, Zhang et al., 2023a], our benchmark focuses on identifying hallucinations in video-language grounding using a binary VQA-based method [Li et al., 2023b, Cui et al., 2023, Chen et al., 2023a], Specifically, we introduce an adversarial evaluation[Tong et al., 2024] with paired questions—one basic and one intentionally hallucinated—to rigorously test models. Moreover, we balance ’yes’ and ’no’ responses to reduce language biases and provide clear explanations to minimize misinterpretation. Comparisons with existing multimodal hallucination benchmark are discussed in Table 1 and Sec. 2.

By comprehensively evaluating twelve LVLMs on VideoHallucer, our analysis leads to three significant insights: First, our analysis revealed a widespread issue of hallucinations across LVLMs, and more critically, the performance gaps between humans and models in all VideoHallucer settings are

Table 1: Comparison with existing Vision Hallucination Benchmark VideoHallucer is the first comprehensive benchmark that focuses on hallucination issues in video-language understanding across five different settings. It employs both QA and adversarial formats for robust evaluation.

Hallucination Type

Benchmark Modality # of Ques/Img

Evaluation Adversarial Obj. Rel. Seman. Temp. Fact. Nonfact.

POPE [Li et al., 2023b] Image 3000/500 ✓ ✗ - - - - BinaryQA ✗ HallusionBench [Guan et al., 2024] Image, Video (4 frames) 1129/346 ✓ ✓ ✓ - - - LLM ✗ MMHal-Bench [Sun et al., 2023] Image 96/96 ✓ ✓ ✓ - - - LLM ✗ Bingo [Cui et al., 2023] Image 370/370 ✓ - ✓ - - ✓ LLM ✓ AMBER [Wang et al., 2024a] Image 15,220/1,004 ✓ ✓ ✓ - - - BinaryQA/CHAIR ✗ EasyDetect [Chen et al., 2024] Image 420/420 ✓ ✓ ✓ - - ✓ LLM ✗ VHTest [Huang et al., 2024] Image 1,200/1,200 ✓ ✓ ✓ - - - LLM ✗ PhD [Liu et al., 2024] Image 53,976/7,020 ✓ ✓ ✓ - - ✓ BinaryQA ✗ VALOR [Qiu et al., 2024] Image 211/211 ✓ ✓ ✓ - - - LLM ✗

VideoHallucer (Ours) Video 1,800/948 ✓ ✓ ✓ ✓ ✓ ✓ BinaryQA ✓

significant (Sec. 4.2). Second, we confirm the benefits brought by scaling on hallucination — increasing the size of the training dataset or/and the model’s parameters improves hallucination detection related to basic visual cues and counterfactuals. However, we found that this approach has a limited impact on the models’ ability to detect extrinsic factual hallucinations (Sec. 4.2). Third, we observed that current models are more proficient at recognizing facts than detecting hallucinations, where the latter requires the models to discern facts within the context of the source material (Sec. 5.1). We also found that these shortcomings in recognizing extrinsic factual hallucinations could be partially mitigated by implementing high-quality explanatory mechanisms (Sec. 5.2).

Building on the above observations, in Sec. 5, we devise Self-PEP, a plug-and-play framework that bolsters the self-improvement capabilities of models through the integration of explanatory processes. By applying Self-PEP, most models demonstrated enhanced performance on the VideoHallucer benchmark, with an average improvement of 5.38%. We believe this framework can streamline future research and development in detecting and mitigating hallucinations in video.

### 2 Related Work

Hallucination in NLG Generative models in NLP, particularly Large Language Models (LLMs), have shown remarkable proficiency across various language generation tasks. Despite their capabilities, a significant issue persists: the text they produce can sometimes be irrelevant or nonsensical. This issue is known in the field of NLP as “hallucination.” Hallucination refers to instances where the content generated by these models does not make sense or deviates from the intended meaning of the source material it is based on [Filippova, 2020, Maynez et al., 2020, Parikh et al., 2020, Zhou et al., 2021, Ji et al., 2023]. The concept of hallucination in NLG tasks can vary slightly, but it is generally categorized into two types based on the relationship between the generated content and the source material. These types are known as Intrinsic Hallucination and Extrinsic Hallucination [Dziri et al., 2021, Huang et al., 2021, Maynez et al., 2020, Ji et al., 2023]. Intrinsic hallucination refers to instances where the content generated by a model directly contradicts the information provided in the source material. On the other hand, extrinsic hallucination occurs when the generated content cannot be confirmed or refuted by the source material. In such cases, the content may either align with or contradict extrinsic knowledge. Building on this, Cao et al. [2022] further categorize extrinsic hallucinations into two types: factual and non-factual. Factual hallucinations produce content that can be verified against real-world knowledge, while non-factual hallucinations generate content that is at odds with what is known about the world. Within specialized research domains, there is a divergence of opinion regarding the value of factual hallucinations. Some studies [Maynez et al., 2020, Thomson and Reiter, 2020], suggest that factual hallucinations can be beneficial. They argue that the additional knowledge introduced by these hallucinations can enhance the informational quality of the generated content.

Vision Hallucination Benchmark Advancements in image-language modeling, particularly in generative tasks such as image captioning and image-based question answering, have led researchers to investigate the phenomenon of hallucination in this field. The concept of object hallucination in image captioning was first introduced by [Rohrbach et al., 2018]. They highlighted the issue of captions erroneously including objects that are absent from the images. To address this, they developed the CHAIR metric, an automatic evaluation tool designed to measure the accuracy of object references in captions by calculating the precision of the hallucinated objects. Following this development, numerous studies have begun to explore hallucination within image-language models.

However, the CHAIR metric itself has come under scrutiny. Li et al. [2023b] identified certain limitations in CHAIR, noting that the metric’s results could be skewed by the way instructions are designed and that the reliance on human-crafted parsing rules could severely restrict its applicability. To overcome these challenges, POPE introduced a binary VQA benchmark specifically tailored for the detection of object hallucination, aiming to provide a more robust and reliable means of evaluation in this area. Recent advancements have been made in the development of evaluation toolkits for object hallucination with the aid of LLMs, as evidenced by research [Wang et al., 2023, Zhai et al., 2023, Petryk et al., 2024, Qiu et al., 2024]. Concurrently, there is a growing body of work aimed at expanding the concept of object hallucination to include additional visual features. [Sun et al., 2023, Wang et al., 2024a, Chen et al., 2023a, Wang et al., 2024b, Han et al., 2024] have begun to explore the relationships, attributions, and other visual cues, including counting, OCR and etc. Moreover, [Jing

- et al., 2023, Guan et al., 2024, Cui et al., 2023, Chen et al., 2024, Liu et al., 2024] are pioneering the detection of factual hallucinations within image-language understanding models, marking a significant step forward in the field. Despite the availability of benchmarks for image-language models, there is a notable lack of a clear and comprehensive framework specifically designed to evaluate LVLMs. To address this gap, our work introduces a thorough benchmark dedicated to assessing the performance of LVLMs, with a particular focus on their susceptibility to hallucination. Moreover, Table 1 presents a comparison of VideoHallucer with other existing datasets designed for the detection of hallucinations in vision-language models. VideoHallucer stands out as the first and most extensive benchmark specifically tailored for LVLM hallucination detection.

3 The VideoHallucer Benchmark

We will provide a detailed introduction to VideoHallucer in the following sections. Sec. 3.1 will introduce the construction process, including the different types of questions. Then, in Sec. 3.2, we will display the statistical information of VideoHallucer. Finally, in Sec. 3.3, we will demonstrate how we evaluate LVLMs on VideoHallucer.

- 3.1 Dataset Construction

To evaluate hallucination issues in detail, we split our benchmark into intrinsic and extrinsic categories, resulting in five settings. In the following section, we introduce how we construct question-answer pairs for each of these five settings separately. The detailed annotation procedure is discussed in Appendix A.2.

- 3.1.1 Intrinsic Hallucination

Object-Relation Our work creates the object-relation hallucination setting that concentrates on the objects and their interactions over time, which we organize into three categories: subject, relation, and object. We construct this setting from existing datasets VidOR [Shang et al., 2019, Thomee et al., 2016] and VidVRD [Shang et al., 2017]. We follow Li et al. [2023b] and use templates to generate basic questions about objects and their relations. Annotators then generate semantically distinct yet visually analogous alternatives for these questions to identify hallucinated content. Through this semi-automated annotation process, we produced 400 question-answer pairs across 183 videos.

Temporal To benchmark the temporal hallucination issue in LVLMs, we design a setting to evaluate models’ hallucination issues from three dimensions: absolute temporal, relative temporal, and event length. Utilizing the ActivityNet dataset [Krishna et al., 2017], we create 400 question-answer pairs spanning 165 videos. For absolute temporal, we select events located in the first or last of 50 videos and asking if they occur at the beginning or end. For relative temporal, we choose 75 pairs of events with clear temporal separation and ask which occurs first. For event length, we compare the durations of 75 pairs of events, questioning which is longer.

Semantic Detail Recent research [Sun et al., 2023, Wang et al., 2024a, Chen et al., 2023a, Wang

- et al., 2024b, Han et al., 2024] underscores the importance of hallucinating object attributes in image-language models, such as OCR, object counting, and scene details, which we summarize as "semantic details". To benchmark this in LVLMs, we developed a setting focused on detecting hallucinations related to these details. We’ve employed a contrastive learning-inspired method using the HawkEye dataset [Wang et al., 2024c], which breaks down long videos into short video segments by semantic similarity. By computing the CLIP score of these video segments, we select segment

pairs with a score above 0.85 as the source video. Annotators then identify semantic differences between these pairs and create basic-hallucinated question-answer pairs from different perspective, resulting in a collection of 400 pairs and corresponding videos.

#### 3.1.2 Extrinsic Hallucination

Factual Our research focuses on detecting extrinsic factual hallucinations—facts consistent with reality but not verifiable from the source material—which can be contextually beneficial or detrimental. In text summarization, hallucinations are unwanted due to the need for source accuracy, while in conversational agents, they may enhance creativity and are more acceptable. We do not assess the overall value of such hallucinations but aim to identify them. At VideoHallucer, we curate instructional videos and course lectures, selecting content that stands alone clearly. Specifically, we select videos from YouCook [Zhou et al., 2018], COIN [Tang et al., 2019], and EDUVSUM [Ghauri et al., 2020] datasets as our source videos. We prefer shorter videos to minimize complexity. For instructional videos, we analyze tutorial steps and create questions about whether certain steps should be taken to finish a task. To prevent ambiguity, we focus on the final steps to detect hallucinations. We edit videos to exclude these steps and pose similar questions as factual hallucinated questions. For course lectures, annotators summarize the content and create questions to determine if the video contains the summary’s content. To create hallucinated questions, annotators modify the summary or add unrelated knowledge, ensuring the summary remains factual but not directly sourced from the video. They also provide explanations for these hallucinated questions to aid further research and improve clarity. As a result, we’ve created a dataset of 200 paired basic and hallucinated questions and answers, totaling 400 pairs from 200 videos—150 instructional videos and 50 course lectures.

Non-facutal Hallucinations that are not based on factual information can be particularly harmful due to their distorted content. Recently, there has been a growing interest among researchers in addressing this issue. At VideoHallucer, we specifically develop a setting to aid in the detection of non-factual hallucinations. To ensure consistency with the factual hallucinations, we select the same set of videos and corresponding basic questions used in the factual hallucination setting. For the creation of hallucinated content, we instructed annotators to manually alter the basic questions or infuse them with counterfactual information. The format of these questions remains identical to that used in the factual context. As a result, we created a comprehensive setting consisting of 400 question-answer pairs linked to 200 videos.

#### 3.2 Dataset Statistics

##### Table 2: VideoHallucer dataset statistics.

Statistic ORH TH SDH EFH ENFH All # of Ques. 400 400 400 400 400 1800 # of Vids. 183 165 400 200 200 948 Avg Ques. len 23.7 69.2 27.3 92.3 94.7 61.4 Avg Vids. len (sec.) 7.0 33.8 13.5 187.0 187.0 85.6

[Figure 17]

- Figure 2: Word cloud of questions in VideoHallucer.

in

is

wrestler

stand

car

wearing

continue

stand

explode

brushes

in

y

prepares

wooden

a

appears

in

p

s

man

d

drinking

es

a

a

a

alking

a

ach

someone

h

he

hold

u

engaged

o

in

man red

p

followed

g

w o

wrestlers

ers o n

d o g

g e

re

finishes

standing

d

- m a
- n

in

aracter

er

h

girl

c

g

ult

ele p h a

g

o

touching

child

g

h

m

b

c

hold

d

se e

cut

o

a

h o

n

nt

person

in

d

d

are

in

ma

dog

man

show

dol

wo

man

wolf

person

an he

baby

a

mannequin

w e

red toy

so meo ne

they

adult

bird

cat

monkey

white

bench

gorilla

pro vide

the

bathtub

burning

Does

###### a

elephant

hammer handbag

video

there

someone

Is

Do

wearing

Can

Are

an

you

see

the

the

there

many

a the

should

###### event

the

he

tarpaulin

person

two

we

we

a

she

1

table

w

they

adult

m

- m

e

o

n

e

a

- n

m

so

###### in

top

prepare

str

m

open

m

mash

the

reeze

whisk

transfer

the

ske wer

the

a

rub

r e

peel

the

spread

n

rinsed

perso

boil

seaso n

the

slice

add

g

polishing

do

pour

ble n d

m o v e

mix

pla c e

fan

ta k e

wip e

the

in

fin ely

cover

c ut

fry

h e

the

p ut

a

at

he

the

pointing

a

mustard

butter

the

e

he

a

in

o ut

th

th e

th e

a

th e

alt

c h

the

e

- o
- p

t h e

m

e

s

th e

o

a

t h e

chick hotdogs

s

w

m

th e

h e

1

a

clams

cream

in

onion

macaroni

ingredients

some

Figure 3: Question distribution in VideoHallucer

Quantitative Analysis There are five settings in VideoHallucer, each corresponding to a different type of hallucination in video understanding. For each setting, we create 400 question-answer pairs, including 200 basic questions and 200 hallucinated questions. To ensure a fair evaluation, the basic

questions in both factual and non-factual settings are identical. As a result, we obtain 1,800 questionanswer pairs with an average length of 61.4 words. Our dataset comprises 948 videos, ranging from 7 seconds to 187 seconds of different settings, with an average length of 85.6 seconds. These videos encompass most existing common video understanding benchmarks [Xu et al., 2017, 2016, Jang et al., 2017, Yu et al., 2019] as well as long video benchmarks [Xiao et al., 2021, Mangalam et al., 2023].

Qualitative Analysis To present VideoHallucer more intuitively, we display the word cloud of our benchmark in Fig. 2 and the sunburst chart in Fig. 3. As shown in these figures, the questions within our benchmark are informative and contain three types: “Does”, “Is”, and “Should”. We believe our benchmark can directly and effectively reveal potential hallucination problems within LVLMs.

#### 3.3 Evaluation

Hallucination Evaluation In this work, we opt for the VQA-based benchmark for the following reasons: (i) Influence of External Factors: Similar to metrics such as BLEU [Papineni et al., 2002] and ROUGE [Lin, 2004], the values in caption-based benchmarks can be affected by factors such as caption prompt and length [Li et al., 2023b]. (ii) Complexity: Methods like CHAIR [Rohrbach et al., 2018] require intricate, human-crafted parsing rules. (iii) LLM Hallucinations: The potential hallucinations existing in LLMs’ generation make it unconvincing to use themselves for self-evaluation [Wang et al., 2024a]. To ensure the credibility of our benchmark, we show a positive correlation between our QA-based evaluation and caption-based methods. More details are discussed in Sec. 6.

To mitigate biases such as the distribution of answers and language bias, we develop VideoHallucer using an adversarial approach [Tong et al., 2024]. Specifically, for each evaluation item, we formulate two types of questions: a basic question and a hallucinated question. The basic question assesses the core capabilities of LVLMs, while the hallucinated question includes deliberately hallucinated content. We then calculate the overall accuracy by considering both the basic and hallucinated questions as a paired set, marking it as a hit only if both questions are answered correctly. We posit that enhancing a model’s ability to recognize and counter hallucinations should not compromise its performance on fundamental tasks. This dual-question structure is designed to ensure that improvements in counter-hallucination do not detract from the model’s original competencies.

Bias Evaluation In addition to the accuracy, we calculate the Yes Percentage Difference (Pct. Diff) and False Positive Ratio (FP Ratio) [Guan et al., 2024] to reveal the bias of these LVLMs. Specifically, the Yes Percentage Difference is calculated as

dy = |{M(v,q) = “yes”}(v,q)∈V | − |{GT(v,q) = “yes”}(v,q)∈V | |V |

(1)

- where V is the set of video question pairs, M(v,q) is the prediction from models, GT(v,q) is the

ground truth. A smaller dy indicates the number of “yes” responses from models is closer to the ground truth, revealing less language bias. the False Positive Ratio is calculated as

.rfp = |{M(v,q) = “yes”}(v,q)∈W| |W|

(2)

- where W is the set of wrongly answered video question pairs. rfp demonstrates the percentage of “yes” in all wrongly predicted answers. A value closer to 50% indicates less bias from the models.

### 4 Experiment

In this section, we evaluate the most popular LVLMs on our VideoHallucer. We first present the setups of these models (Sec. 4.1), followed by the main results and performance analysis (Sec. 4.2). Additionally, we compare the effectiveness of current Image-Language Models on the object-relation and semantic details settings of VideoHallucer (Sec. 4.3). Finally, we reveal the human performance on our benchmark (Sec. 4.4).

#### 4.1 Setups

We assess twelve LVLMs, comprising ten open-source models (7B unspecified), including VideoChatGPT, Valley2, Video-LLaMA2, VideoChat2, Video-LLaVA, LLaMA-VID, VideoLaVIT, MiniGPT4-

- Table 3: Performance comparison of existing LVLMs on VideoHallucer with additional Yes/No bias analysis. To evaluate the accuracy, we present the performance of all these models on basic questions, hallucinated questions, and the overall score. We highlight the Top 2 models of opensource LVLMs.

Yes/No Bias Accuracy on VideoHallucer

Models Language Model Pct. Diff (∼ 0) FP Ratio (∼ 0.5) Basic ↑ Hallucinated ↑ Overall ↑ Open-source LVLMs

VideoChatGPT [Maaz et al., 2023] LLaMA-7B 0.40 0.89 92.8 10.4 6.4 Valley2 [Luo et al., 2023] LLaMA2-7B -0.07 0.29 44.4 11.5 2.8 Video-LLaMA2 [Zhang et al., 2023b] LLaMA2-7B 0.36 0.84 90.9 12.7 10 VideoChat2 [Li et al., 2023c] Vicuna-7B-v0 -0.24 0.15 29.7 25.8 7.8 Video-LLaVA [Lin et al., 2023] Vicuna-7B-v1.5 0.36 0.91 95.1 20.3 17.8 LLaMA-VID [Li et al., 2023d] Vicuna-7B-v1.5 0.29 0.83 89.9 26.6 21 VideoLaVIT [Jin et al., 2024] LLaMA2-7B 0.36 0.91 94.9 21.3 18.9 MiniGPT4-Video [Ataallah et al., 2024] Mistral-7B 0.18 0.62 79.4 28.6 22.3 PLLaVA [Xu et al., 2024] Vicuna-7B-1.5 0.06 0.53 75.1 55.5 38.1 LLaVA-NeXT-Video-DPO [Zhang et al., 2024] Vicuna-7B-1.5 -0.04 0.40 62.5 60.9 32.0

Video-LLaMA2-13B [Zhang et al., 2023b] LLaMA2-13B 0.36 0.79 88.3 3.8 3.3 LLaMA-VID-13B [Li et al., 2023d] Vicuna-13B-v1.5 0.21 0.72 85.2 36.9 29.2 PLLaVA-13B [Xu et al., 2024] Vicuna-13B-1.5 0.17 0.72 87.5 48.6 41.2 PLLaVA-34B [Xu et al., 2024] Yi-34B 0.18 0.78 90.8 50.8 45 LLaVA-NeXT-Video-DPO-34B [Zhang et al., 2024] Yi-34B 0.07 0.55 73.6 51.6 32.3

Closed-source LVLMs

Gemini-1.5-Pro [Reid et al., 2024] - 0.15 0.62 83.6 42.3 37.8 GPT-4o [OpenAI, 2024] - -0.02 0.43 75.1 74.2 53.3 Human - 0.02 0.42 90 88.8 85

Video, PLLaVA, and LLaVA-NeXT-Video-DPO, and two closed-source models, Gemini-1.5-Pro and GPT-4o. To make a fair comparison, we set all these baselines following their original setting including the number of frames and generation hyper-parameters.

#### 4.2 Main Benchmark Results

The overall results are delineated in Table 3. We find that although all models demonstrate strong capabilities in answering basic questions, they experience a significant decline in accuracy when confronted with hallucinated questions. The overall accuracy significantly drops compared to the accuracy on basic and hallucinated questions due to a mistake in one of the question pairs. This pattern implies a widespread susceptibility to hallucination issues among the current models. Regarding the “Yes/No Bias”, models with an obvious bias are more likely to have hallucination problems. Specifically, we find that most models tend to generate “Yes” answers, except for VideoChat2, which is more likely to generate “No” answers. Since PLLaVA and VideoChat2 share the same video tuning data, the difference stems from the image data. Therefore, we believe the bias and hallucination issues in VideoChat2 originate from the training image data. Additionally, LLaVA-NeXT-Video-DPO shares similar tuning data with Video-LLaVA, and as a result, the DPO strategy significantly reduces bias and hallucination. Generally, we do not find a huge gap between open-source models and closed-source models, but all models lack significant human-like hallucination detection capabilities.

Here, we will dig deeper into different types of hallucinations in LVLMs. We illustrated the more detailed analysis in Figure 4. We have the following findings.

First, when comparing the various dimensions of the radar chart, we observe that most models exhibit fewer hallucinations in the object-relation setting (ORH) than in other areas. Specifically, the performance of most models in this setting is centered around 50%. This observation points to a deficiency in existing modeling techniques to detect hallucinations beyond elementary visual cues.

Second, regarding semantic detail hallucination (SDH), when comparing LLaMA-VID, LLaVANeXT-Video, and PLLaVA with training data increasing, we find models with more training data significantly perform better than others. Moreover, we compare the different scales of these models on non-factual settings (ENFH), and we find larger models markedly surpass smaller models. We ascribe this superiority to the effects of data and model parameter scaling, which appear to bolster the model’s prowess in discerning visual details and retaining world knowledge.

Finally, for extrinsic factual hallucination (EFH), most models demonstrate inability in this setting. To be more specific, most existing models can’t discern hallucination issues that align with world knowledge but contradict the video context. Given the significant differences in model performance

|ORH<br><br>TH<br><br>SDH<br><br>EFH<br><br>ENFH<br><br>0 10 20 30 40 50 60|
|---|

TH

LLaMA-VID

LLaMA-VID-13B

SDH

PLLaVA

PLLaVA-13B PLLaVA-34B LLaVA-NeXT-Video

ORH

0 20 40 60 80 100

LLaVA-NeXT-Video-34B

Gemini-1.5-pro

GPT-4o Human

EFH

ENFH

Figure 4: Comparative analysis of models using VideoHallucer across various settings. The left displays the complete set of results, while the right provides a magnified view to facilitate a closer examination of the performance details among the open-source models of different scales.

in the two settings of extrinsic hallucinations, we will explore this issue in greater depth in the Sec. 5 to uncover the reasons behind this phenomenon.

#### 4.3 Comparison between Image-Language Models

Table 4: Comparison between Video-langauge models and Image-language models. We present the results for basic questions, hallucinated questions, and the overall score. We highlight the Top 2 model of accuracy on VideoHallucer.

Object-Relation Semantic Detail

Models Basic Hallucinated Overall Basic Hallucinated Overall Video-Language Models

LLaMA-VID [Li et al., 2023d] 78.5 59 43.5 89 24 17 MiniGPT4-Video [Ataallah et al., 2024] 80.5 34.5 27.5 78.5 27.5 23.5 PLLaVA [Xu et al., 2024] 76 76.5 60 83 71.5 57 LLaVA-NeXT-Video-DPO [Zhang et al., 2024] 72 73 51.5 63.5 69 38 Gemini-1.5-Pro [Reid et al., 2024] 84.5 56 52 89 63 53.5 GPT-4o [OpenAI, 2024] 81 82.5 66 63 87.5 55.5

Image-Language Models

LLaVA-1.5 [Liu et al., 2023] 79.5 79 61.5 75.5 63.5 43 GPT4V [OpenAI et al., 2024] 65 81.5 55 59 87 49.5

We carried out a comparative study of two types of multimodal models: image-language models and video-language models. For this analysis, we selected the top-performing LLaVA-1.5 and GPT4V. To facilitate a fair comparison, we used the middle frame from each video as the input for the image-language models. The findings, presented in Table 4, show that open-source image-language models have a superior performance in detecting object-relation hallucination, even though the dataset includes dynamic interactions. We believe this discrepancy in performance is due to two main reasons. First, there is a significantly larger amount of training data available for images than for videos. This means that image-language models benefit more from the scaling law. Second, videos tend to include more noise than images, making them more likely to encounter the hallucination problem. In light of these insights, we suggest that future development of LVLMs could be enhanced by integrating image datasets [Lin et al., 2023] or by building upon the foundations of existing image-language models [Zhang et al., 2024, Xu et al., 2024].

#### 4.4 Human Evaluation

We recruited three individuals proficient in English to evaluate the VideoHallucer benchmark. Each evaluator possesses basic computer knowledge, which is essential since part of the extrinsic dataset

includes computer science courses. To mitigate potential bias from the evaluators, we randomized the sequence of question-answer pairs to ensure that basic and hallucination pairs did not appear consecutively. We then computed the Pearson correlation coefficient among all evaluators’ scores, which resulted in a moderate agreement with a value of 0.557.

### 5 Self-PEP: Self-improvement with Predict-Explain-Predict

In Sec. 4.2, we find that most models perform worse in the factual hallucination setting compared to others on VideoHallucer. In this section, we aim to understand the reasons behind this by comparing the fact detection and hallucination detection abilities of these models (Sec. 5.1). Based on our findings, we propose a simple yet effective method with explanation to mitigate the models’ hallucination issues (Sec. 5.2).

#### 5.1 Fact Detection vs. Hallucination Detection

One of our key findings suggests that these models struggle to identify external factual hallucinations within video content. As a result, we have redirected our focus in this section to assess the models’ capability to discern factual knowledge. To this end, we crafted additional QA pairs derived from questions in the extrinsic hallucination setting. These questions aim to determine the factual accuracy of statements in the original extrinsic hallucination questions.

In our experiment, we utilized course videos from this context due to their rich factual content. For instance, we paraphrased extrinsic factual hallucination questions to “Does the following course summary include any non-factual information? {summary}”, and non-factual hallucination questions to “Does the following course summary encompass all essential factual information? {summary}”. By setting the correct answer to these questions as "no", we sought to counteract any language biases present in language models. The experimental results, illustrated in Fig. 5, reveal that most models are more adept at detecting factual knowledge than at detecting hallucinations. This indicates that current methods can effectively recognize counterfactual content. However, their ability to detect hallucinations in video data is markedly limited, highlighting a significant potential for improvement in this area. Among these models, we find the LLaVA-NeXT-Video-DPO performs better on hallucination detection than fact detection, we believe this gain comes from the DPO. Therefore, we believe that human feedback could mitigate hallucination issues in LVLMs.

80

Hallucination Detection

Fact Detection

70

60

Accuracy

50

40

30

20

10

0

VideoChatGPTValley2Video-LLaMA2VideoChat2VideoLLaVALLaMA-VIDVideoLaVITMiniGPT4-VideoLLaVA-NeXT-VideoGemini-1.5-pro

Figure 5: Results Comparison of Hallucination Detection and Fact Detection for Extrinsic Hallucination. Most models are more adept at detecting factuality than detecting hallucination.

##### 5.2 Self-PEP Framework Advances in LLMs have seen significant improvements through costly human feedback [Ouyang

- et al., 2022, Sun et al., 2023]. To reduce expenses, recent studies [Madaan et al., 2023, Shinn
- et al., 2023, Ye et al., 2023, Yan et al., 2023, Pan et al., 2023, Chen et al., 2023b, Zhang et al., 2023c, You et al., 2023, Lightman et al., 2023] focus on self-improvement of LLMs for better performance and clarity. Our research in Sec. 5.1 shows LLMs are better at identifying facts than hallucination, with the latter demanding a firm grasp of factual context. Building on these insights, we investigate further in this section. We examine how explanation can affect a model’s ability to detect hallucinations. Our methodology involves prompting the model to generate a prediction and then provide an explanation for its output. We then use the explanation to refine the model’s predictive accuracy. This experiment is conducted within the context of extrinsic factual hallucination detection, and the results are illustrated in Fig. 7. Although the self-generated explanations are less impactful compared to those derived from ground truth explanations, our results demonstrate that

Self Improve

###### Video Model

###### Visual Knowledge

Self Explain (P-E-P)

Input Explain Answer

Predict

Predict

Answer

Figure 6: The Self-PEP Framework. The self-improvement strategy (top-row) leverages extracted visual knowledge to introduce related information visibly. The self-explanation strategy (bottom-row) employs a predict-explain-predict scheme to mitigate hallucination issues with selfgenerated explanations.

80

GT-Explain

Self-Explain

70

Predict

60

Accuracy

50

40

30

20

10

0

VideoChatGPTValley2Video-LLaMA2VideoChat2VideoLLaVALLaMA-VIDVideoLaVITMiniGPT4-VideoLLaVA-NeXT-VideoGemini-1.5-pro

Figure 7: Results of Self-explain Strategy on Extrinsic Factual Hallucination. Most models could benefit from the self-explain strategy, and the ground-truth explanation would improve models’ performance on extrinsic hallucination significantly.

- Table 5: Results of Self-PEP Framework. We highlight the best performing model after applying our framework. For a clear comparison, we annotate all the models after applying our framework.

Models Object-Relation Temporal Semantic Detail Factual Non-factual Overall Accuracy VideoChatGPT [Maaz et al., 2023] 6 0 2 7 17 6.4

+Self-PEP 33.5+27.5 4.54.5 22.5+20.5 14+7 30+13.0 20.9+14.5 Valley2 [Luo et al., 2023] 4.5 3 2.5 0.5 3.5 2.8

+Self-PEP 10+5.5 5.5+2.5 1.5-1 1.5+1 4.5+1 4.6+1.8 Video-LLaMA-2 [Zhang et al., 2023b] 18 7.5 1 6.5 17 10

+Self-PEP 12-6 5.5-2 8+7 15.5+9 26+9 13.4+3.4 VideoChat2 [Li et al., 2023c] 10.5 7.5 9 7 5 7.8

+Self-PEP 34+23.5 15+7.5 27.5+18.5 19.5+12.5 21.5+16.5 23.5+15.7 VideoLLaVA [Lin et al., 2023] 34.5 13.5 12 3 26 17.8

+Self-PEP 52+17.5 5.5-8 36+24 11+8 34+8 27.7+10 LLaMA-VID [Li et al., 2023d] 43.5 21 17 2.5 21 21

+Self-PEP 44.5+1 14-7 36.5+19.5 22+19.5 33+12 30+9 VideoLaVIT [Jin et al., 2024] 35.5 25.5 10.5 4 19 18.9 +Self-PEP 24-11.5 0.5-25 22+11.5 6+1 13-6 13-6 MiniGPT4-Video [Ataallah et al., 2024] 27.5 18 23.5 12 30.5 22.3 +Self-PEP 47+19.5 21.5+3.5 34.5+11 14+2 35.5+5 30.5+8.2

PLLaVA [Xu et al., 2024] 60 23.5 57 9.5 40.5 38.1

+Self-PEP 52-8 7-15.5 46.5-10.5 16.5+7 36.5-1 31.7-5.6 LLaVA-NeXT-Video-DPO [Zhang et al., 2024] 51.5 28 38 14 28.5 32

+Self-PEP 51-0.5 15-13 37.5-0.5 11-3 16-12.5 26.1-5.9 Gemini-1.5-Pro [Reid et al., 2024] 52 18.5 53.5 16.5 48.5 37.8

+Self-PEP 56+4 44+25.5 64+10.5 33+16.5 63+14.5 52+14.2

they do contribute to improving the model’s capability in hallucination detection. This improvement is related to the model’s proficiency in fact detection.

Building on these insights, as depicted in Fig. 6, we devise an innovative framework called SelfImprovement with Predict-Explain-Predict (Self-PEP). This framework is designed to enhance the model’s resilience against the tendency to produce hallucinations. It capitalizes on the model’s established strength in fact detection over hallucination identification. The Self-PEP framework operates in two phases: self-improvement and self-explanation. In the self-improvement phase, the model autonomously extracts visual knowledge [Yin et al., 2023], while the self-explanation phase involves a three-step process: predict, then explain, and finally refine the prediction using the explanation. The implementation details of the Self-PEP framework and qualitative results are provided in Appendix A.3.3.

The efficacy of this framework is demonstrated in Table 5, where we note that Self-PEP significantly boosts the performance of most models on the VideoHallucer benchmark, yielding substantial improvements. In addition, we find the improvements on hallucinated questions are much more significant than the basic questions. When comparing all different settings, we find our methods could consistently improve all models’ performance on extrinsic factual hallucination. Our method,

for instance, could potentially harm PLLaVA in other settings while improving its performance solely on the factual setting. Consequently, we believe our method could reduce hallucination issues within existing LVLMs, particularly factual hallucinations, which are consistently present in LVLMs.

### 6 Conclusion and Discussion

In this work, we introduce VideoHallucer, a novel and comprehensive benchmark for detecting hallucinations in LVLMs. Our adversarial approach, which challenges models with paired questions, ensures a thorough evaluation of a model’s ability to discern hallucinations. Through rigorous testing of 12 LVLMs, we have identified the pervasive nature of spurious hallucinations and the limitations of model scaling in addressing certain types of hallucinations. Our innovative Self-PEP framework has demonstrated the potential to significantly enhance model performance against hallucinations.

Hallucination vs. Adversarial Attacks Adversarial attacks are deliberately crafted inputs designed to provoke erroneous outputs from a model, potentially leading to various security breaches such as unauthorized data access, fraud, system intrusion, malware deployment, content manipulation, and service disruption. In contrast, our tool, VideoHallucer, is specifically tailored to assess perceptual challenges. It evaluates whether a model can produce accurate and reliable responses based on the provided context, without being misled by superficially plausible but incorrect information.

VideoHallucer vs. Other Video-Language Benchmarks VideoHallucer is primarily designed to evaluate issues related to hallucinated content and the faithfulness of generated text from LVLM. This approach is relatively direct and specific. In comparison, other benchmarks may concentrate on basic visual understanding or more abstract cognitive tasks such as logical reasoning or advanced scene comprehension.

VQA-based benchmark vs. Caption-based benchmark Existing hallucination benchmarks primarily encompass two types of benchmarks and evaluation pipelines found in existing research: VQA-based and Caption-based. The VQA-based benchmark, exemplified by methods like POPE [Li et al., 2023b], employs binary question-answering to detect instances of object hallucination. In contrast, caption-based benchmarks [Rohrbach et al., 2018, Wang et al., 2023, Zhai et al., 2023] assess the precision or recall of hallucinated objects within captions, with evaluations conducted either through rule-based parsing or LLMs. In our study, we opt for the VQA-based benchmark for several reasons. Firstly, akin to established evaluation metrics for content generation such as BLEU [Papineni et al., 2002] and ROUGE [Lin, 2004], these values can be influenced by extrinsic factors, including the nature of the caption prompt and the caption’s length [Li et al., 2023b]. Secondly, methods like CHAIR [Rohrbach et al., 2018] depend heavily on intricate, human-crafted rules for parsing, which increases complexity. Thirdly, although recent advancements have incorporated LLMs for evaluation, employing a tool that itself is prone to have hallucination issues for assessment purposes is problematic. To establish the credibility of our benchmark, we demonstrate a positive correlation between our QA-based evaluation and the caption-based methods. Further details on this analysis are available in Appendix A.1.

### References

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karén Simonyan. Flamingo: a visual language model for few-shot learning. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 19730–19742. PMLR, 2023a.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems (NeurIPS), 2023.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, Irwan Bello, Jake Berdine, Gabriel Bernadett-Shapiro, Christopher Berner, Lenny Bogdonoff, Oleg Boiko, Madelaine Boyd, Anna-Luisa Brakman, Greg Brockman, Tim Brooks, Miles Brundage, Kevin Button, Trevor Cai, Rosie Campbell, Andrew Cann, Brittany Carey, Chelsea Carlson, Rory Carmichael, Brooke Chan, Che Chang, Fotis Chantzis, Derek Chen, Sully Chen, Ruby Chen, Jason Chen, Mark Chen, Ben Chess, Chester Cho, Casey Chu, Hyung Won Chung, Dave Cummings, Jeremiah Currier, Yunxing Dai, Cory Decareaux, Thomas Degry, Noah Deutsch, Damien Deville, Arka Dhar, David Dohan, Steve Dowling, Sheila Dunning, Adrien Ecoffet, Atty Eleti, Tyna Eloundou, David Farhi, Liam Fedus, Niko Felix, Simón Posada Fishman, Juston Forte, Isabella Fulford, Leo Gao, Elie Georges, Christian Gibson, Vik Goel, Tarun Gogineni, Gabriel Goh, Rapha Gontijo-Lopes, Jonathan Gordon, Morgan Grafstein, Scott Gray, Ryan Greene, Joshua Gross, Shixiang Shane Gu, Yufei Guo, Chris Hallacy, Jesse Han, Jeff Harris, Yuchen He, Mike Heaton, Johannes Heidecke, Chris Hesse, Alan Hickey, Wade Hickey, Peter Hoeschele, Brandon Houghton, Kenny Hsu, Shengli Hu, Xin Hu, Joost Huizinga, Shantanu Jain, Shawn Jain, Joanne Jang, Angela Jiang, Roger Jiang, Haozhun Jin, Denny Jin, Shino Jomoto, Billie Jonn, Heewoo Jun, Tomer Kaftan, Łukasz Kaiser, Ali Kamali, Ingmar Kanitscheider, Nitish Shirish Keskar, Tabarak Khan, Logan Kilpatrick, Jong Wook Kim, Christina Kim, Yongjik Kim, Jan Hendrik Kirchner, Jamie Kiros, Matt Knight, Daniel Kokotajlo, Łukasz Kondraciuk, Andrew Kondrich, Aris Konstantinidis, Kyle Kosic, Gretchen Krueger, Vishal Kuo, Michael Lampe, Ikai Lan, Teddy Lee, Jan Leike, Jade Leung, Daniel Levy, Chak Ming Li, Rachel Lim, Molly Lin, Stephanie Lin, Mateusz Litwin, Theresa Lopez, Ryan Lowe, Patricia Lue, Anna Makanju, Kim Malfacini, Sam Manning, Todor Markov, Yaniv Markovski, Bianca Martin, Katie Mayer, Andrew Mayne, Bob McGrew, Scott Mayer McKinney, Christine McLeavey, Paul McMillan, Jake McNeil, David Medina, Aalok Mehta, Jacob Menick, Luke Metz, Andrey Mishchenko, Pamela Mishkin, Vinnie Monaco, Evan Morikawa, Daniel Mossing, Tong Mu, Mira Murati, Oleg Murk, David Mély, Ashvin Nair, Reiichiro Nakano, Rajeev Nayak, Arvind Neelakantan, Richard Ngo, Hyeonwoo Noh, Long Ouyang, Cullen O’Keefe, Jakub Pachocki, Alex Paino, Joe Palermo, Ashley Pantuliano, Giambattista Parascandolo, Joel Parish, Emy Parparita, Alex Passos, Mikhail Pavlov, Andrew Peng, Adam Perelman, Filipe de Avila Belbute Peres, Michael Petrov, Henrique Ponde de Oliveira Pinto, Michael, Pokorny, Michelle Pokrass, Vitchyr H. Pong, Tolly Powell, Alethea Power, Boris Power, Elizabeth Proehl, Raul Puri, Alec Radford, Jack Rae, Aditya Ramesh, Cameron Raymond, Francis Real, Kendra Rimbach, Carl Ross, Bob Rotsted, Henri Roussez, Nick Ryder, Mario Saltarelli, Ted Sanders, Shibani Santurkar, Girish Sastry, Heather Schmidt, David Schnurr, John Schulman, Daniel Selsam, Kyla Sheppard, Toki Sherbakov, Jessica Shieh, Sarah Shoker, Pranav Shyam, Szymon Sidor, Eric Sigler, Maddie Simens, Jordan Sitkin, Katarina Slama, Ian Sohl, Benjamin Sokolowsky, Yang Song, Natalie Staudacher, Felipe Petroski Such, Natalie Summers, Ilya Sutskever, Jie Tang,

Nikolas Tezak, Madeleine B. Thompson, Phil Tillet, Amin Tootoonchian, Elizabeth Tseng, Preston Tuggle, Nick Turley, Jerry Tworek, Juan Felipe Cerón Uribe, Andrea Vallone, Arun Vijayvergiya, Chelsea Voss, Carroll Wainwright, Justin Jay Wang, Alvin Wang, Ben Wang, Jonathan Ward, Jason Wei, CJ Weinmann, Akila Welihinda, Peter Welinder, Jiayi Weng, Lilian Weng, Matt Wiethoff, Dave Willner, Clemens Winter, Samuel Wolrich, Hannah Wong, Lauren Workman, Sherwin Wu, Jeff Wu, Michael Wu, Kai Xiao, Tao Xu, Sarah Yoo, Kevin Yu, Qiming Yuan, Wojciech Zaremba, Rowan Zellers, Chong Zhang, Marvin Zhang, Shengjia Zhao, Tianhao Zheng, Juntang Zhuang, William Zhuk, and Barret Zoph. Gpt-4 technical report, 2024.

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh. VQA: visual question answering. In 2015 IEEE International Conference on Computer Vision, ICCV 2015, Santiago, Chile, December 7-13, 2015, pages 2425–2433. IEEE Computer Society, 2015.

Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. In David J. Fleet, Tomás Pajdla, Bernt Schiele, and Tinne Tuytelaars, editors, European Conference on Computer Vision (ECCV), volume 8693 of Lecture Notes in Computer Science, pages 740–755. Springer, 2014.

D. Xu, Zhou Zhao, Jun Xiao, Fei Wu, Hanwang Zhang, Xiangnan He, and Yueting Zhuang. Video question answering via gradually refined attention over appearance and motion. 2017.

Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. Conference on Computer Vision and Pattern Recognition (CVPR), pages 5288–5296, 2016.

Ranjay Krishna, Kenji Hata, Frederic Ren, Li Fei-Fei, and Juan Carlos Niebles. Dense-captioning events in videos. In International Conference on Computer Vision (ICCV), pages 706–715. IEEE Computer Society, 2017.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Annual Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 292–305. Association for Computational Linguistics, 2023b.

Shengbang Tong, Zhuang Liu, Yuexiang Zhai, Yi Ma, Yann LeCun, and Saining Xie. Eyes wide shut? exploring the visual shortcomings of multimodal llms. CoRR, abs/2401.06209, 2024.

Suzanne Petryk, David M. Chan, Anish Kachinthaya, Haodi Zou, John Canny, Joseph E. Gonzalez, and Trevor Darrell. Aloha: A new measure for hallucination in captioning models, 2024.

Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. Object hallucination in image captioning. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii, editors, Annual Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4035–4045. Association for Computational Linguistics, 2018.

Tianyang Han, Qing Lian, Rui Pan, Renjie Pi, Jipeng Zhang, Shizhe Diao, Yong Lin, and Tong Zhang. The instinctive bias: Spurious images lead to hallucination in mllms. CoRR, abs/2402.03757, 2024.

Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, Kurt Keutzer, and Trevor Darrell. Aligning large multimodal models with factually augmented RLHF. CoRR, abs/2309.14525, 2023.

Junyang Wang, Yuhang Wang, Guohai Xu, Jing Zhang, Yukai Gu, Haitao Jia, Jiaqi Wang, Haiyang Xu, Ming Yan, Ji Zhang, and Jitao Sang. Amber: An llm-free multi-dimensional benchmark for mllms hallucination evaluation, 2024a.

Zhiyang Chen, Yousong Zhu, Yufei Zhan, Zhaowen Li, Chaoyang Zhao, Jinqiao Wang, and Ming Tang. Mitigating hallucination in visual language models with visual supervision. CoRR, abs/2311.16479, 2023a.

Lei Wang, Jiabang He, Shenshen Li, Ning Liu, and Ee-Peng Lim. Mitigating fine-grained hallucination by fine-tuning large vision-language models with caption rewrites. In Stevan Rudinac, Alan Hanjalic, Cynthia C. S. Liem, Marcel Worring, Björn Þór Jónsson, Bei Liu, and Yoko Yamakata, editors, MultiMedia Modeling - 30th International Conference, MMM 2024, Amsterdam, The Netherlands, January 29 - February 2, 2024, Proceedings, Part IV, volume 14557 of Lecture Notes in Computer Science, pages 32–45. Springer, 2024b.

Chenhang Cui, Yiyang Zhou, Xinyu Yang, Shirley Wu, Linjun Zhang, James Zou, and Huaxiu Yao. Holistic analysis of hallucination in gpt-4v(ision): Bias and interference challenges. CoRR, abs/2311.03287, 2023.

Liqiang Jing, Ruosen Li, Yunmo Chen, Mengzhao Jia, and Xinya Du. FAITHSCORE: evaluating hallucinations in large vision-language models. CoRR, abs/2311.01477, 2023.

Xiang Chen, Chenxi Wang, Yida Xue, Ningyu Zhang, Xiaoyan Yang, Qiang Li, Yue Shen, Lei Liang, Jinjie Gu, and Huajun Chen. Unified hallucination detection for multimodal large language models. CoRR, abs/2402.03190, 2024.

Junaid Ahmed Ghauri, Sherzod Hakimov, and Ralph Ewerth. Classification of important segments in educational videos using multimodal features. In Stefan Conrad and Ilaria Tiddi, editors, Proceedings of the CIKM 2020 Workshops co-located with 29th ACM International Conference on Information and Knowledge Management (CIKM 2020), Galway, Ireland, October 19-23, 2020, volume 2699 of CEUR Workshop Proceedings. CEUR-WS.org, 2020.

Yale Song, Jordi Vallmitjana, Amanda Stent, and Alejandro Jaimes. Tvsum: Summarizing web videos using titles. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2015, Boston, MA, USA, June 7-12, 2015, pages 5179–5187. IEEE Computer Society, 2015.

Muru Zhang, Ofir Press, William Merrill, Alisa Liu, and Noah A. Smith. How language model hallucinations can snowball. CoRR, abs/2305.13534, 2023a.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. Hallusionbench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large visionlanguage models, 2024.

Wen Huang, Hongbin Liu, Minxin Guo, and Neil Zhenqiang Gong. Visual hallucinations of multimodal large language models. CoRR, abs/2402.14683, 2024.

Jiazhen Liu, Yuhan Fu, Ruobing Xie, Runquan Xie, Xingwu Sun, Fengzong Lian, Zhanhui Kang, and Xirong Li. Phd: A prompted visual hallucination evaluation dataset. CoRR, abs/2403.11116, 2024.

Haoyi Qiu, Wenbo Hu, Zi-Yi Dou, and Nanyun Peng. Valor-eval: Holistic coverage and faithfulness evaluation of large vision-language models, 2024.

Katja Filippova. Controlled hallucinations: Learning to generate faithfully from noisy data. In Trevor Cohn, Yulan He, and Yang Liu, editors, Annual Conference on Empirical Methods in Natural Language Processing (EMNLP), volume EMNLP 2020 of Findings of ACL, pages 864–870. Association for Computational Linguistics, 2020.

Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan T. McDonald. On faithfulness and factuality in abstractive summarization. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel R. Tetreault, editors, Annual Meeting of the Association for Computational Linguistics (ACL), pages 1906–1919. Association for Computational Linguistics, 2020.

Ankur P. Parikh, Xuezhi Wang, Sebastian Gehrmann, Manaal Faruqui, Bhuwan Dhingra, Diyi Yang, and Dipanjan Das. Totto: A controlled table-to-text generation dataset. In Bonnie Webber, Trevor Cohn, Yulan He, and Yang Liu, editors, Annual Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1173–1186. Association for Computational Linguistics, 2020.

Chunting Zhou, Graham Neubig, Jiatao Gu, Mona T. Diab, Francisco Guzmán, Luke Zettlemoyer, and Marjan Ghazvininejad. Detecting hallucinated content in conditional neural sequence generation. In Chengqing Zong, Fei Xia, Wenjie Li, and Roberto Navigli, editors, Annual Meeting of the Association for Computational Linguistics (ACL), volume ACL/IJCNLP 2021 of Findings of ACL, pages 1393–1404. Association for Computational Linguistics, 2021.

Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM Comput. Surv., 55(12):248:1–248:38, 2023.

Nouha Dziri, Andrea Madotto, Osmar Zaïane, and Avishek Joey Bose. Neural path hunter: Reducing hallucination in dialogue systems via path grounding. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, Annual Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 2197–2214. Association for Computational Linguistics, 2021.

Yi-Chong Huang, Xia-Chong Feng, Xiao-Cheng Feng, and Bing Qin. The factual inconsistency problem in abstractive text summarization: A survey. CoRR, abs/2104.14839, 2021.

Meng Cao, Yue Dong, and Jackie Chi Kit Cheung. Hallucinated but factual! inspecting the factuality of hallucinations in abstractive summarization. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Annual Meeting of the Association for Computational Linguistics (ACL), pages 3340–3354. Association for Computational Linguistics, 2022.

Craig Thomson and Ehud Reiter. A gold standard methodology for evaluating accuracy in datato-text systems. In Brian Davis, Yvette Graham, John D. Kelleher, and Yaji Sripada, editors, Proceedings of the 13th International Conference on Natural Language Generation, INLG 2020, Dublin, Ireland, December 15-18, 2020, pages 158–168. Association for Computational Linguistics, 2020.

Junyang Wang, Yiyang Zhou, Guohai Xu, Pengcheng Shi, Chenlin Zhao, Haiyang Xu, Qinghao Ye, Ming Yan, Ji Zhang, Jihua Zhu, Jitao Sang, and Haoyu Tang. Evaluation and analysis of hallucination in large vision-language models. CoRR, abs/2308.15126, 2023.

Bohan Zhai, Shijia Yang, Xiangchen Zhao, Chenfeng Xu, Sheng Shen, Dongdi Zhao, Kurt Keutzer, Manling Li, Tan Yan, and Xiangjun Fan. Halle-switch: Rethinking and controlling object existence hallucinations in large vision language models for detailed caption. CoRR, abs/2310.01779, 2023.

Xindi Shang, Donglin Di, Junbin Xiao, Yu Cao, Xun Yang, and Tat-Seng Chua. Annotating objects and relations in user-generated videos. In Proceedings of the 2019 on International Conference on Multimedia Retrieval, pages 279–287. ACM, 2019.

Bart Thomee, David A Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and Li-Jia Li. Yfcc100m: The new data in multimedia research. Communications of the ACM, 59(2):64–73, 2016.

Xindi Shang, Tongwei Ren, Jingfan Guo, Hanwang Zhang, and Tat-Seng Chua. Video visual relation detection. In ACM International Conference on Multimedia, Mountain View, CA USA, October 2017.

Yueqian Wang, Xiaojun Meng, Jianxin Liang, Yuxuan Wang, Qun Liu, and Dongyan Zhao. Hawkeye: Training video-text llms for grounding text in videos. CoRR, abs/2403.10228, 2024c.

Luowei Zhou, Chenliang Xu, and Jason J. Corso. Towards automatic learning of procedures from web instructional videos. In Sheila A. McIlraith and Kilian Q. Weinberger, editors, AAAI Conference on Artificial Intelligence (AAAI), pages 7590–7598. AAAI Press, 2018.

Yansong Tang, Dajun Ding, Yongming Rao, Yu Zheng, Danyang Zhang, Lili Zhao, Jiwen Lu, and Jie Zhou. COIN: A large-scale dataset for comprehensive instructional video analysis. In Conference on Computer Vision and Pattern Recognition (CVPR), pages 1207–1216. Computer Vision Foundation / IEEE, 2019.

Y. Jang, Yale Song, Youngjae Yu, Youngjin Kim, and Gunhee Kim. Tgif-qa: Toward spatio-temporal reasoning in visual question answering. Conference on Computer Vision and Pattern Recognition (CVPR), pages 1359–1367, 2017.

Zhou Yu, D. Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. Activitynet-qa: A dataset for understanding complex web videos via question answering. AAAI Conference on Artificial Intelligence (AAAI), 2019.

Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of questionanswering to explaining temporal actions. In Conference on Computer Vision and Pattern Recognition (CVPR), pages 9777–9786. Computer Vision Foundation / IEEE, 2021.

Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems (NeurIPS), 2023.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Annual Meeting of the Association for Computational Linguistics (ACL), pages 311–318. ACL, 2002.

Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics.

Muhammad Maaz, Hanoona Abdul Rasheed, Salman H. Khan, and Fahad Shahbaz Khan. Videochatgpt: Towards detailed video understanding via large vision and language models. CoRR, abs/2306.05424, 2023.

Ruipu Luo, Ziwang Zhao, Min Yang, Junwei Dong, Minghui Qiu, Pengcheng Lu, Tao Wang, and Zhongyu Wei. Valley: Video assistant with large language model enhanced ability. CoRR, abs/2306.07207, 2023.

Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. In Yansong Feng and Els Lefever, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023 - System Demonstrations, Singapore, December 6-10, 2023, pages 543–553. Association for Computational Linguistics, 2023b.

Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, and Yu Qiao. Mvbench: A comprehensive multi-modal video understanding benchmark. CoRR, abs/2311.17005, 2023c.

Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. CoRR, abs/2311.10122, 2023.

Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. CoRR, abs/2311.17043, 2023d.

Yang Jin, Zhicheng Sun, Kun Xu, Kun Xu, Liwei Chen, Hao Jiang, Quzhe Huang, Chengru Song, Yuliang Liu, Di Zhang, Yang Song, Kun Gai, and Yadong Mu. Video-lavit: Unified video-language pre-training with decoupled visual-motional tokenization. CoRR, abs/2402.03161, 2024.

Kirolos Ataallah, Xiaoqian Shen, Eslam Abdelrahman, Essam Sleiman, Deyao Zhu, Jian Ding, and Mohamed Elhoseiny. Minigpt4-video: Advancing multimodal llms for video understanding with interleaved visual-textual tokens, 2024.

Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava : Parameter-free llava extension from images to videos for video dense captioning, 2024.

Yuanhan Zhang, Bo Li, haotian Liu, Yong jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Llava-next: A strong zero-shot video understanding model, April 2024. URL https://llava-vl.github.io/blog/2024-04-30-llava-next-video/.

Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy P. Lillicrap, JeanBaptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, Ioannis Antonoglou, Rohan Anil, Sebastian Borgeaud, Andrew M. Dai, Katie Millican, Ethan Dyer, Mia Glaese, Thibault Sottiaux, Benjamin Lee, Fabio Viola, Malcolm Reynolds, Yuanzhong Xu, James Molloy, Jilin Chen, Michael Isard, Paul Barham, Tom Hennigan, Ross McIlroy, Melvin Johnson, Johan Schalkwyk, Eli Collins, Eliza Rutherford, Erica Moreira, Kareem Ayoub, Megha Goel, Clemens Meyer, Gregory Thornton, Zhen Yang, Henryk Michalewski, Zaheer Abbas, Nathan Schucher, Ankesh Anand, Richard Ives, James Keeling, Karel Lenc, Salem Haykal, Siamak Shakeri, Pranav Shyam, Aakanksha Chowdhery, Roman Ring, Stephen Spencer, Eren Sezener, and et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. CoRR, abs/2403.05530, 2024.

OpenAI. Gpt-4o system card, 2024.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 December 9, 2022, 2022.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Sean Welleck, Bodhisattwa Prasad Majumder, Shashank Gupta, Amir Yazdanbakhsh, and Peter Clark. Self-refine: Iterative refinement with self-feedback. ArXiv, abs/2303.17651, 2023.

Noah Shinn, Federico Cassano, Beck Labash, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. 2023.

Seonghyeon Ye, Yongrae Jo, Doyoung Kim, Sungdong Kim, Hyeonbin Hwang, and Minjoon Seo. Selfee: Iterative self-revising llm empowered by self-feedback generation. Blog post, May 2023.

Hao Yan, Saurabh Srivastava, Yintao Tai, Sida I. Wang, Wen tau Yih, and Ziyu Yao. Learning to simulate natural language feedback for interactive semantic parsing. ArXiv, abs/2305.08195, 2023.

Liangming Pan, Michael Stephen Saxon, Wenda Xu, Deepak Nathani, Xinyi Wang, and William Yang Wang. Automatically correcting large language models: Surveying the landscape of diverse selfcorrection strategies. ArXiv, abs/2308.03188, 2023.

Zhenfang Chen, Qinhong Zhou, Yikang Shen, Yining Hong, Hao Zhang, and Chuang Gan. See, think, confirm: Interactive prompting between vision and language models for knowledge-based visual reasoning. ArXiv, abs/2301.05226, 2023b.

Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alexander J. Smola. Multimodal chain-of-thought reasoning in language models. ArXiv, abs/2302.00923, 2023c.

Haoxuan You, Rui Sun, Zhecan Wang, Long Chen, Gengyu Wang, Hammad A. Ayyubi, Kai-Wei Chang, and Shih-Fu Chang. Idealgpt: Iteratively decomposing vision and language reasoning via large language models, 2023.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. ArXiv, abs/2305.20050, 2023.

Shukang Yin, Chaoyou Fu, Sirui Zhao, Tong Xu, Hao Wang, Dianbo Sui, Yunhang Shen, Ke Li, Xing Sun, and Enhong Chen. Woodpecker: Hallucination correction for multimodal large language models. CoRR, abs/2310.16045, 2023.

Jie Lei, Tamara L. Berg, and Mohit Bansal. Revealing single frame bias for video-and-language learning. In Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki, editors, Annual Meeting of the Association for Computational Linguistics (ACL), pages 487–507. Association for Computational Linguistics, 2023.

##### Junbin Xiao, Angela Yao, Yicong Li, and Tat-Seng Chua. Can I trust your answer? visually grounded video question answering. CoRR, abs/2309.01327, 2023.

### A Appendix

We provide appendices and supplementary materials as follows:

- • In Section A.1, we show the detailed analysis of two ways of evaluation methods for vision hallucination benchmark.
- • Section A.2 demonstrates the detailed annotation procedures.
- • Section A.3 outlines the implementation details of our study

- – In A.3.1, we show baseline configurations
- – In A.3.2, we demonstrate the evaluation prompts
- – In A.3.3, we reveal the implementation details of Self-PEP

- • Section A.4, we show the analysis of the coherence of the self-generated explanation and the groud-truth explanation.
- • A.5. Detailed Quantitative Results

- – In Section A.5.1, we present the comprehensive results for the VideoHallucer benchmark.
- – Section A.5.2 details the performance of Self-PEP on VideoHallucer

- • A.6. Example Questions. We provide examples from VideoHallucer that cover various types of hallucinations: object, spatial relation, temporal relation, absolute temporal, relative temporal, and semantic detail (including attribution, event, count, OCR, camera, and scene), as well as extrinsic factual (instruction, course) and non-factual (instruction, course) hallucinations.
- • We show the limitations and ethic states in Appendix A.7.

#### A.1 Evaluation Method Analysis

In this study, we investigate the relationship between VideoHallucer (QA-based mehtod) and the caption-based method. To determine the effectiveness of our QA-based method in comparison to the caption-based approach, we use the Pearson correlation coefficient to measure the coherence. Our focus is on two well-established caption-based evaluation metrics: the CHAIR score [Rohrbach et al., 2018] and the Coverage Score [Zhai et al., 2023]. Specifically, we randomly select a sample of 100 images from the COCO dataset to investigate the correlation between our newly introduced accuracy on VideoHallucer and the traditional metrics employed for image caption. We apply the experiments on LLaVA-1.5. To prevnet the issue of hallucination in LLMs, we implement a rule-based method to calculate the Coverage Score similar to CHAIR. Our findings reveal a moderate positive correlation with the Coverage Score (r = 0.477) and a weak negative correlation with the CHAIR score (r = −0.139). We illustrate the correlation between the overall accuracy on VideoHallucer and the Coverage Score in Fig. 8. These results indicate that overall accuracy on VideoHallucer correlates positively with the caption-based methods, suggesting that VideoHallucer is a reliable and robust alternative to caption-based evaluation methods for assessing hallucinations in generative models.

1.0

0.8

CoverageScore

0.6

0.4

0.2

0.0

0.0 0.2 0.4 0.6 0.8 1.0

Accuracy on VideoHallucer

Figure 8: Correlation of Accuracy on VideoHallucer and Coverage score

#### A.2 Annotation Details

To enhance the quality and reduce the annotation costs of our benchmark, we have developed a semi-automatic pipeline for constructing VideoHallucer. The process begins with the use of readily available tools and templates to create or collect initial pairs of basic and hallucinated content. Subsequently, we engage annotators to meticulously review and refine the question-answer pairs, ensuring they are well-grounded in the video content. This review focuses on three key aspects: correctness, ambiguity, and fluency. To guarantee the high standard of VideoHallucer, we implement a

two-stage human verification process. In the following paragraphs, we will give a detailed introduction to different sets of VideoHallucer.

Object-Relation Hallucination In the domain of vision-language modeling, object hallucination is a persistent challenge, particularly within the context of image captioning tasks. These tasks demand that the model accurately identifies and describes objects in an image. A significant body of research has been dedicated to addressing this issue, encompassing various approaches such as the development of benchmarks, the refinement of evaluation metrics, and the exploration of both analytical and practical solutions. More recently, the focus of research has expanded from object hallucination to include relation hallucination, with a particular emphasis on spatial relationships between objects. Building on this foundation, our work takes a novel step by applying these concepts to video content. We aim to capture not only the objects but also the dynamic interactions between them, considering both spatial and temporal dimensions. We posit that the ability to detect basic visual concepts is a fundamental skill for LVLMs.

We posit that objects and their relations are fundamentally intertwined in the basic visual comprehension of videos. To reflect this in our benchmark, we have constructed an object-relation hallucination set. This set is divided into three distinct categories: subject, relation, and object. To elaborate, there is a wealth of prior research that has focused on annotating objects and relations within video content. In order to leverage this existing body of work and avoid duplicating efforts, we have developed our object-relation hallucination sets based on these pre-existing datasets. Specifically, we have utilized the VidOR [Shang et al., 2019, Thomee et al., 2016] and VidVRD [Shang et al., 2017] datasets as the foundation for our set, ensuring that we build upon the annotations they provide without unnecessary redundancy in human annotation labor. For subject and object, we following the paradigm of POPE [Li et al., 2023b], use the template “Is there a/an {object} in the video?”. For relation, for spatial relation, we use the template “Is {subject} {relation} {object} in the video?”, for temporal relation, or action, we use the template “Does/Do {subject} {relation} {object} in the video?”. For hallucinated questions, instead of randomly selecting from existing object pools, we ask annotator to write semantic different but visually similar object, and anatomy of the relation word. After obtaining the question automatically, we manually check the question answer pairs. As a result, we get 200 basic-hallucination question-answer pairs, totally 400 question-answer pairs and 183 videos.

Temporal Hallucination Temporal information is the most fundamental difference between video input and image input. However, most recent video-text LLMs mainly focus on factual contents in short videos, ignoring the temporal information of multiple events in long videos. Recent studies [Lei et al., 2023, Xiao et al., 2023] found that they have almost no ability to perform time-related tasks such as video grounding.

The understanding of temporal information in videos contains two aspects: where in the video timeline is an event located, and how long it lasts. Therefore, we plan to test 3 time-related capabilities of video-text models: (1) 50 examples for absolute temporal understanding, where the model is asked to answer the position of an event in the video, such as whether it is located at the beginning or at the end of the video, (2) 75 examples for relative temporal understanding, where the model is asked to determine the order of two events, and (3) 75 examples for event length understanding, where the models is asked to compare the duration length of two events.

We construct the temporal hallucination benchmark based on the val set of ActivityNet Captions [Krishna et al., 2017], a popular temporal video grounding dataset with event-level video annotations: each event is labeled with a text description and a time span (start sec, end sec) in the video. For absolute temporal understanding examples, we sampled 50 events from 50 different videos that are entirely located in the first or last third of its video. For events in the first third of its video, we use “Does {event} happen at the beginning of the video?” as the basic question, and “Does {event} happen at the end of the video?” as the hallucinated question, and vise versa for the events in the last thirds of its video. For relative temporal understanding examples, we sampled 75 pairs of events from 75 different videos which satisfies that the end time of the first event is at least 0.1t earlier than the start time of the second event, where t is the duration of the video. We randomly choose “Does {event1} happen earlier than {event2}?” or “Does {event2} happen later than {event1}?” as the basic question, and “Does {event1} happen later than {event2}?” or “Does {event2} happen earlier than {event1}?” as the hallucinated question. For event length understanding examples, we also sampled 75 pairs of events from 75 different videos which satisfies that the duration of the first event is at least

2 times longer than the duration of the second event. We randomly choose “Is the event {event1} longer than {event2}?” or “Is the event {event2} shorter than {event1}?” as the basic question, and “Is the event {event1} shorter than {event2}?” or “Is the event {event2} longer than {event1}?” as the hallucinating question. As a result, we get 200 basic-hallucination question-answer pairs, totally 400 question-answer pairs and 165 videos.

Semantic Detail Hallucination Recent studies increasingly emphasize the importance of object attributes in visual analysis. A number of these works [Sun et al., 2023, Wang et al., 2024a, Chen et al.,

- 2023a, Wang et al., 2024b, Han et al., 2024] delve deeper, attempting to uncover a broader spectrum of visual details such as optical character recognition (OCR), object counting, and environmental context. Given the challenge of creating a definitive classification for the diverse range of visual details, our work consolidates these attributes under the umbrella term “other semantic details.” This encompasses OCR, attribute recognition, viewpoint identification, and scene understanding, among others. It has been observed that most vision-language models exhibit deficiencies in accurately identifying these nuanced details. To address this gap, we have specifically developed a dataset aimed at enhancing the detection of hallucinations related to these other semantic details.

To enhance our understanding of the detailed semantic content in videos, we have adopted a technique inspired by contrastive learning, which involves identifying distinctions by drawing comparisons. In particular, we utilize video clips from the HawkEye dataset [Wang et al., 2024c], which segments lengthy videos into shorter clips based on PySceneDetect, and then merging clips that share high semantic similarity. These clips are designed to encapsulate distinct semantic meanings. Once we have these video clips, we compute the CLIP score among them to measure their semantic relatedness. We then select pairs of video clips with a CLIP score exceeding 0.85 for further analysis. Subsequently, we engage annotators to pinpoint the semantic disparities between these selected pairs of clips. Based on these differences, the annotators are tasked with creating a set of basic-hallucinated question-answer pairs. Throughout this process, we instruct the annotators to pay particular attention to various aspects of the semantic details within the clips. As a result, we get 200 basic-hallucination question-answer pairs, totally 400 question-answer pairs and 400 videos.

Extrinsic Factual Hallucination Extrinsic factual hallucinations, which are unverifiable against the original source material yet remain factually consistent, may sometimes be considered beneficial. Certain studies suggest that these hallucinations could enhance the richness and informativeness of the generated content by providing additional background knowledge. However, the usefulness of such hallucinations is highly context-dependent. In the field of text summarization, any form of hallucination is undesirable, as accuracy and fidelity to the source are paramount. Conversely, in conversational agents, there is an expectation for the generation of varied and imaginative responses, where factual hallucinations might be more acceptable. Therefore, in our research, we do not seek to determine the overall value of factual hallucinations. Instead, our focus is on developing models that can detect and identify these hallucinations.

At VideoHallucer, we meticulously curate two distinct categories of videos: instructional videos and course lectures. Our selection criteria prioritize content that is rich in knowledge and can be understood independently of extrinsic information, thereby minimizing potential ambiguity. For the instructional category, we extract samples from the YouCook [Zhou et al., 2018] and COIN [Tang et al., 2019] datasets. These include cooking recipes and guides for various activities. In the realm of course lectures, our focus is on the EDUVSUM dataset [Ghauri et al., 2020], which features online courses in fields like computer science, the history of science, and engineering. To simplify matters, we intentionally choose shorter videos from these sources. When dealing with instructional videos, our process begins by identifying the steps outlined in each tutorial. To formulate basic questions, we adopt a template that reads: “Based on the video, should we {step} when we {target}?” For hallucinated questions, we pay special attention to the final steps, as they typically contain detailed information and dropping these steps won’t make the instructional video incomplete. We select these as the focal point for VideoHallucer. Specifically, we trim videos and remove the content related the these stepsand pose questions in the same format.

For course lectures, we first instruct annotators to condense the content into a summary that captures the key points. We then generate basic questions following the pattern: “Does the video provide information that allows us to learn that {summary}?” For hallucinated questions, we challenge annotators to alter the summary or incorporate knowledge not covered in the video. This ensures that while the summary remains factual, it cannot be solely derived from the video content. Moreover, we

ask the annotator to write an explanation for these hallucinated questions. The explanation point out the reason and the location of the hallucinated question. Through this meticulous process, we have compiled a dataset of 200 paired basic and hallucination questions and answers, resulting in a total of 400 question-answer pairs and 200 videos, where 150 videos are instructional videos and 50 videos are online courses.

Extrinsic Non-facutal Hallucination Hallucinations that are not based on factual information can be particularly harmful due to their distorted content. Recently, there has been a growing interest among researchers in addressing this issue. At VideoHallucer, we have specifically developed a dataset to aid in the detection of non-factual hallucinations.

To ensure consistency with the detection of factual hallucinations, we have selected the same set of videos and corresponding basic questions used in the factual hallucination dataset. For the creation of hallucinated content, we instructed annotators to manually alter the basic questions or infuse them with counterfactual information. The format of these questions remains identical to that used in the factual context. As a result of this process, we have compiled a comprehensive dataset consisting of 400 question-answer pairs linked to 200 videos.

- A.3 Implementation Details

- A.3.1 Setups for baselines

In our experiment, we choose the 7B-level video language models for fair comparison, including VideoChatGPT [Maaz et al., 2023], Valley2 [Luo et al., 2023], Video-LLaMA2 [Zhang et al., 2023b], VideoChat2 [Li et al., 2023c], VideoLLaVA [Lin et al., 2023], LLaMA-VID [Li et al., 2023d], VideoLaVIT [Jin et al., 2024], and MiniGPT4-Video [Ataallah et al., 2024]. To assure a fair comparison, we take the default hyper-parameter of these models, including “max_new_tokens”, “do_sample”, “temperature”, “num_beams”, and “num_of_frames”. To further reveal the potential of scaling law, we additionally add the existing best-performed closed-source model Gemini-1.5Pro [Reid et al., 2024] as the current upbound of existing LVLMs. For the Gemini-1.5-Pro, we take the fps as 1, and to improve the evaluation efficiency, we set the max number of frames as 128.

- A.3.2 Prompt for the Overall Results

To evaluate responses, we append the prompt “Answer the question using ’yes’ or ’no’.” to the end of each question. We then compare the answer to “yes” or “no” to calculate accuracy.

- A.3.3 Implementation Detail of the Self-PEP

As illustrated in Fig. 6, there are two main components of the Self-PEP: the self-improvement and the self-explain. For the self-improvement, we ask the model to extract the visual knowledge [Yin et al., 2023]. In our implementation, we ask the model to generate the caption for simplicity, the prompt is “Describe the video: ”. After obtaining the caption, we ask the model to predict the answer the question with the input of both the video and the self-generated caption. The prompt is “Description: {description} Please provide a clear response to the question below by watching the video. If necessary, you can also use the accompanying Description to help refine your answer. Your response should be a simple ’yes’ or ’no’. Question: {question} Answer the question using ’yes’ or ’no’: ”. We further apply the self-explain to the model, by asking the model to explain-then-predict to get the final answer. The prompt is “Description: {description} Please offer a detailed explanation for your answer to the following question. After explaining, verify the accuracy of the information you’ve used in your explanation. Once you’ve confirmed the facts, please respond to the question with a simple ’yes’ or ’no’. Question: {question} Answer: {predict} Answer the question using ’yes’ or ’no’: ”. Furthermore, we illustrate the qualitative study of the Self-PEP framework on Gemini-1.5-Pro in Fig. 9, we find that with the explanation, the model is able to correct its answer.

- A.4 Explanation Analysis

Upon comparing Fig. 7 with Fig. 5, it becomes evident that there is a correlation between the model’s self-explanation capability and its ability to detect factual information. To further explore this relationship, we evaluate how well the self-generated explanations align with the established ground truth explanations. For this purpose, we employ GPT-4 to conduct a consistency analysis,

[Figure 18]

Figure 9: Qualitative Analysis: In the video observed, the woman initially drizzles olive oil solely on the shrimp, neglecting to do the same for the kabobs. A self-generated explanation can accurately identify the rationale behind this action and provide a clarified response.

utilizing the prompt outlined in Fig. 10. The results, presented in Table 6, indicate that the quality of the explanations has a significant impact on the model’s proficiency in identifying instances of hallucination. Consequently, our goal is to enhance the model’s capacity to produce accurate and reliable explanations as a means to mitigate the issue of hallucination.

Table 6: Coherence of the self-generated explanation and the annotated explanation Models Consistent Rate GPT4 Score

VideoChatGPT [Maaz et al., 2023] 33.5 2.555 Valley2 [Luo et al., 2023] 39 2.34 Video-LLaMA-2 [Zhang et al., 2023b] 8.5 1.175 VideoChat2 [Li et al., 2023c] 38 2.42 VideoLLaVA [Lin et al., 2023] 36.5 2.45 LLaMA-VID [Li et al., 2023d] 42 2.625 VideoLaVIT [Jin et al., 2024] 36 2.435 MiniGPT4-Video [Ataallah et al., 2024] 31.5 2.085 Gemini-1.5-Pro [Reid et al., 2024] 37 2.54

[Figure 19]

Figure 10: Prompt for Analysis of the Explanations

#### A.5 Detailed Quantitative Results

In this section, we provide the detailed quantitative results on VideoHallucer, including the overall results (Appendix A.5.1) and the results with Self-PEP (Appendix A.5.2).

#### A.5.1 The Detailed Results on VideoHallucer

In this section, we present the comprehensive results of various LVLMs across different settings of VideoHallucer. Table 7 details the outcomes for object-relation hallucination; Table 8 displays the findings for temporal hallucination; Table 9 outlines the results for semantic detail hallucination; Table 10 depicts the results for extrinsic factual hallucination; and Table 11 provides the results for extrinsic non-factual hallucination.

#### Table 7: Overall Results on Object Relation Hallucination

###### Models Basic Question Hallucinated Question Overall Accuracy

- VideoChatGPT [Maaz et al., 2023] 95.5 7 6 Valley2 [Luo et al., 2023] 80.5 10.5 4.5 Video-LLaMA2 [Zhang et al., 2023b] 88.5 21.5 18 VideoChat2 [Li et al., 2023c] 26 41.5 10.5 VideoLLaVA [Lin et al., 2023] 95 38 34.5 LLaMA-VID [Li et al., 2023d] 78.5 59 43.5 LLaMA-VID-13B [Li et al., 2023d] 87.5 55.5 44.5 VideoLaVIT [Jin et al., 2024] 94.5 39 35.5 MiniGPT4-Video [Ataallah et al., 2024] 80.5 34.5 27.5 PLLaVA [Xu et al., 2024] 76 76.5 60 LLaVA-NeXT-Video-DPO [Zhang et al., 2024] 72 73 51.5 Gemini-1.5-Pro [Reid et al., 2024] 84.5 56 52

Table 8: Overall Results on Temporal Hallucination

Models Basic Question Hallucinated Question Overall Accuracy

VideoChatGPT [Maaz et al., 2023] 100 0 0 Valley2 [Luo et al., 2023] 25 11.5 3 Video-LLaMA2 [Zhang et al., 2023b] 91.5 8.5 7.5 VideoChat2 [Li et al., 2023c] 23.5 25 7.5 VideoLLaVA [Lin et al., 2023] 97.5 13.5 13.5 LLaMA-VID [Li et al., 2023d] 86 25 21 LLaMA-VID-13B [Li et al., 2023d] 78.5 35 27 VideoLaVIT [Jin et al., 2024] 88.5 27 25.5 MiniGPT4-Video [Ataallah et al., 2024] 68.5 27 18 PLLaVA [Xu et al., 2024] 46.5 58 23.5 LLaVA-NeXT-Video-DPO [Zhang et al., 2024] 53 61 28 Gemini-1.5-Pro [Reid et al., 2024] 81.5 19 18.5

- Table 9: Overall Results on Semantic Detail Hallucination

Models Basic Question Hallucinated Question Overall Accuracy

- VideoChatGPT [Maaz et al., 2023] 96.5 4 2 Valley2 [Luo et al., 2023] 86.5 6.5 2.5 Video-LLaMA2 [Zhang et al., 2023b] 99 1.5 1

- VideoChat2 [Li et al., 2023c] 33 26 9 VideoLLaVA [Lin et al., 2023] 97 14 12 LLaMA-VID [Li et al., 2023d] 89 24 17 LLaMA-VID-13B [Li et al., 2023d] 90.5 30 25.5

- VideoLaVIT [Jin et al., 2024] 96.5 13 10.5 MiniGPT4-Video [Ataallah et al., 2024] 78.5 27.5 23.5 PLLaVA [Xu et al., 2024] 83 71.5 57 LLaVA-NeXT-Video-DPO [Zhang et al., 2024] 63.5 69 38 Gemini-1.5-Pro [Reid et al., 2024] 89 63 53.5

#### Table 10: Overall Results on extrinsic Factual Hallucination

Models Basic Question Hallucinated Question Overall Accuracy

VideoChatGPT [Maaz et al., 2023] 86.5 13.5 7 Valley2 [Luo et al., 2023] 14 15 0.5 Video-LLaMA2 [Zhang et al., 2023b] 88 8.5 6.5 VideoChat2 [Li et al., 2023c] 32 16.5 7 VideoLLaVA [Lin et al., 2023] 93 4.5 3 LLaMA-VID [Li et al., 2023d] 98 2.5 2.5 LLaMA-VID-13B [Li et al., 2023d] 85 17.5 12.5

- VideoLaVIT [Jin et al., 2024] 97.5 6 4 MiniGPT4-Video [Ataallah et al., 2024] 86 16.5 12 PLLaVA [Xu et al., 2024] 85 18 9.5 LLaVA-NeXT-Video-DPO [Zhang et al., 2024] 62.5 41 14 Gemini-1.5-Pro [Reid et al., 2024] 82 19 16.5

#### Table 11: Overall Results on extrinsic Non-factual Hallucination

###### Models Basic Question Hallucinated Question Overall Accuracy

VideoChatGPT [Maaz et al., 2023] 85.5 27.5 17 Valley2 [Luo et al., 2023] 16 14 3.5 Video-LLaMA2 [Zhang et al., 2023b] 87.5 23.5 17

- VideoChat2 [Li et al., 2023c] 34 20 0.5 VideoLLaVA [Lin et al., 2023] 93 31.5 26 LLaMA-VID [Li et al., 2023d] 98 22.5 21 LLaMA-VID-13B [Li et al., 2023d] 84.5 46.5 36.5 VideoLaVIT [Jin et al., 2024] 97.5 21.5 19 MiniGPT4-Video [Ataallah et al., 2024] 83.5 37.5 30.5 PLLaVA [Xu et al., 2024] 85 53.5 40.5 LLaVA-NeXT-Video-DPO [Zhang et al., 2024] 61.5 60.5 28.5 Gemini-1.5-Pro [Reid et al., 2024] 81 54.5 48.5

#### A.5.2 The Detailed Results of Self-PEP

In this section, we further show the detailed results of Self-PEP on different settings of VideoHallucer. Table Table 12 details the outcomes for object-relation hallucination; Table Table 13 displays the findings for temporal hallucination; Table Table 14 outlines the results for semantic detail hallucination; Table Table 15 depicts the results for extrinsic factual hallucination; and Table Table 16 provides the results for extrinsic non-factual hallucination.

#### Table 12: Self-PEP Results on Object Relation Hallucination

###### Models Basic Question Hallucinated Question Overall Accuracy

- VideoChatGPT [Maaz et al., 2023] 95.5 7 6

+Self-PEP 72.5 53 33.5 Valley2 [Luo et al., 2023] 80.5 10.5 4.5

- +Self-PEP 20.5 31 10 Video-LLaMA2 [Zhang et al., 2023b] 88.5 21.5 18

+Self-PEP 75 27.5 12 VideoChat2 [Li et al., 2023c] 26 41.5 10.5

+Self-PEP 61 58.5 34 VideoLLaVA [Lin et al., 2023] 95 38 34.5

+Self-PEP 72 75 52 LLaMA-VID [Li et al., 2023d] 78.5 59 43.5 +Self-PEP 60 74.5 44.5 VideoLaVIT [Jin et al., 2024] 94.5 39 35.5 +Self-PEP 90.5 29 24 MiniGPT4-Video [Ataallah et al., 2024] 80.5 34.5 27.5

+Self-PEP 72 64.5 47 PLLaVA [Xu et al., 2024] 76 76.5 60 +Self-PEP 67.5 74 52 LLaVA-NeXT-Video-DPO [Zhang et al., 2024] 72 73 51.5

+Self-PEP 67 79.5 51 Gemini-1.5-Pro [Reid et al., 2024] 84.5 56 52

- +Self-PEP 62.5 84 56

Table 13: Self-PEP Results on Temporal Hallucination

Models Basic Question Hallucinated Question Overall Accuracy VideoChatGPT [Maaz et al., 2023] 100 0 0

+Self-PEP 95.5 8 4.5 Valley2 [Luo et al., 2023] 25 11.5 3

- +Self-PEP 21.5 15 5.5 Video-LLaMA2 [Zhang et al., 2023b] 91.5 8.5 7.5

+Self-PEP 57.5 23 5.5 VideoChat2 [Li et al., 2023c] 23.5 25 7.5 +Self-PEP 47 34 15 VideoLLaVA [Lin et al., 2023] 97.5 13.5 13.5

+Self-PEP 97.5 5.5 5.5 LLaMA-VID [Li et al., 2023d] 86 25 21 +Self-PEP 87.5 19.5 14 VideoLaVIT [Jin et al., 2024] 88.5 27 25.5

+Self-PEP 100 0.5 0.5 MiniGPT4-Video [Ataallah et al., 2024] 68.5 27 18

+Self-PEP 82 26 21.5 PLLaVA [Xu et al., 2024] 46.5 58 23.5 +Self-PEP 67.5 34 8 LLaVA-NeXT-Video-DPO [Zhang et al., 2024] 53 61 28 +Self-PEP 52 54.5 15 Gemini-1.5-Pro [Reid et al., 2024] 81.5 19 18.5

+Self-PEP 66 57 44

#### Table 14: Self-PEP Results on Semantic Detail Hallucination

Models Basic Question Hallucinated Question Overall Accuracy VideoChatGPT [Maaz et al., 2023] 96.5 4 2

+Self-PEP 72 42 22.5 Valley2 [Luo et al., 2023] 86.5 6.5 2.5 +Self-PEP 12.5 16.5 1.5 Video-LLaMA2 [Zhang et al., 2023b] 99 1.5 1

- +Self-PEP 79.5 10.5 8

- VideoChat2 [Li et al., 2023c] 33 26 9

+Self-PEP 75 39.5 27.5 VideoLLaVA [Lin et al., 2023] 97 14 12 +Self-PEP 74 58.5 36 LLaMA-VID [Li et al., 2023d] 89 24 17

- +Self-PEP 50 80 36.5

- VideoLaVIT [Jin et al., 2024] 96.5 13 10.5

+Self-PEP 92 27 22 MiniGPT4-Video [Ataallah et al., 2024] 78.5 27.5 23.5 +Self-PEP 77.5 44.5 34.5 PLLaVA [Xu et al., 2024] 83 71.5 57

- +Self-PEP 64 78 46.5 LLaVA-NeXT-Video-DPO [Zhang et al., 2024] 63.5 69 38

+Self-PEP 50 80 37.5 Gemini-1.5-Pro [Reid et al., 2024] 89 63 53.5 +Self-PEP 72.5 88.5 64

Table 15: Self-PEP Results on extrinsic Factual Hallucination

Models Basic Question Hallucinated Question Overall Accuracy VideoChatGPT [Maaz et al., 2023] 86.5 13.5 7

- +Self-PEP 74 28 14 Valley2 [Luo et al., 2023] 14 15 0.5

+Self-PEP 11.5 15.5 1.5 Video-LLaMA2 [Zhang et al., 2023b] 88 8.5 6.5 +Self-PEP 58 37 15.5 VideoChat2 [Li et al., 2023c] 32 16.5 7

+Self-PEP 57.5 38.5 19.5 VideoLLaVA [Lin et al., 2023] 93 4.5 3

+Self-PEP 80.5 23 11 LLaMA-VID [Li et al., 2023d] 98 2.5 2.5 +Self-PEP 65.5 44 22

- VideoLaVIT [Jin et al., 2024] 97.5 6 4

+Self-PEP 94.5 6.5 6 MiniGPT4-Video [Ataallah et al., 2024] 86 16.5 12 +Self-PEP 80 26.5 14 PLLaVA [Xu et al., 2024] 85 18 9.5

- +Self-PEP 63.5 45 16.5 LLaVA-NeXT-Video-DPO [Zhang et al., 2024] 62.5 41 14

+Self-PEP 28.5 72 11 Gemini-1.5-Pro [Reid et al., 2024] 82 19 16.5

- +Self-PEP 61.5 55 33

#### Table 16: Self-PEP Results on extrinsic Non-factual Hallucination

Models Basic Question Hallucinated Question Overall Accuracy VideoChatGPT [Maaz et al., 2023] 85.5 27.5 17

- +Self-PEP 75 49 30 Valley2 [Luo et al., 2023] 16 14 3.5

+Self-PEP 10.5 14.5 4.5 Video-LLaMA2 [Zhang et al., 2023b] 87.5 23.5 17

- +Self-PEP 65 52.6 26

- VideoChat2 [Li et al., 2023c] 34 20 0.5

+Self-PEP 47 50.5 21.5 VideoLLaVA [Lin et al., 2023] 93 31.5 26 +Self-PEP 80.5 53 34 LLaMA-VID [Li et al., 2023d] 98 22.5 21

- +Self-PEP 62.5 65 33 VideoLaVIT [Jin et al., 2024] 97.5 21.5 19

+Self-PEP 94.5 15 13.5 MiniGPT4-Video [Ataallah et al., 2024] 83.5 37.5 30.5 +Self-PEP 80 48.5 35.5 PLLaVA [Xu et al., 2024] 85 53.5 40.5

- +Self-PEP 63.5 72 39.5 LLaVA-NeXT-Video-DPO [Zhang et al., 2024] 61.5 60.5 28.5

+Self-PEP 31.5 82.5 16 Gemini-1.5-Pro [Reid et al., 2024] 81 54.5 48.5

+Self-PEP 67.5 92 63

- A.6 Example Questions of VideoHallucer In this section, we provide more cases from VideoHallucer.

[Figure 20]

Figure 11: An example Basic-Hallucinated question pair of Object Hallucination

[Figure 21]

- Figure 12: An example Basic-Hallucinated question pair of Spatial Relation Hallucination

[Figure 22]

- Figure 13: An example Basic-Hallucinated question pair of Temporal Relation Hallucination

- A.7 Limitations and Ethic States

Limitation Although we take a lot of strategies to make sure the quality of VideoHallucer, there is noise introduced by human annotations. In addition, currently, the scalability of this benchmark is still limited.

[Figure 23]

- Figure 14: An example Basic-Hallucinated question pair of Absolute Temporal Hallucination

[Figure 24]

- Figure 15: An example Basic-Hallucinated question pair of Relative Temporal Hallucination

Societal Impacts VideoHallucer is designed to counter hallucination in LVLMs, however, the hallucinated questions in this benchmark could lead to misinterpretation in related research area.

Responsibility and License We bear all responsibility in case of violation of rights and our dataset is under the license of CC BY-NC-SA (Attribution-NonCommercial-ShareAlike).

[Figure 25]

##### Figure 16: An example Basic-Hallucinated question pair of Semantic Detail (Attribution) Hallucination

[Figure 26]

##### Figure 17: An example Basic-Hallucinated question pair of Semantic Detail (Event) Hallucination

[Figure 27]

##### Figure 18: An example Basic-Hallucinated question pair of Semantic Detail (Count) Hallucination

[Figure 28]

##### Figure 19: An example Basic-Hallucinated question pair of Semantic Detail (OCR) Hallucination

[Figure 29]

##### Figure 20: An example Basic-Hallucinated question pair of Semantic Detail (Camera) Hallucination

[Figure 30]

##### Figure 21: An example Basic-Hallucinated question pair of Semantic Detail (Scene) Hallucination

[Figure 31]

##### Figure 22: An example Basic-Hallucinated question pair of Extrinsic Factual (Instruction) Hallucination

[Figure 32]

##### Figure 23: An example Basic-Hallucinated question pair of Extrinsic Factual (Course) Hallucination

[Figure 33]

##### Figure 24: An example Basic-Hallucinated question pair of Extrinsic Non-factual (Instruction) Hallucination

[Figure 34]

##### Figure 25: An example Basic-Hallucinated question pair of Extrinsic Non-factual (Course) Hallucination

