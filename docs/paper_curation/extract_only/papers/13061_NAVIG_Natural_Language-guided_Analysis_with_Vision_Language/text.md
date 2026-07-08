## NAVIG: Natural Language-guided Analysis with Vision Language Models for Image Geo-localization

Zheyuan Zhang1* Runze Li2* Tasnim Kabir3 Jordan Boyd-Graber3 1Tsinghua University 2Nanjing University 3University of Maryland zheyuan-22@mails.tsinghua.edu.cn jbg@umiacs.umd.edu

# arXiv:2502.14638v1[cs.CL]20Feb2025

[Figure 1]

### Abstract

[Figure 2]

###### Darlington, United Kingdom

Image geo-localization is the task of predicting the specific location of an image and requires complex reasoning across visual, geographical, and cultural contexts. While prior Vision Language Models (VLMs) have the best accuracy at this task, there is a dearth of highquality datasets and models for analytical reasoning. We first create NAVICLUES, a highquality dataset derived from GeoGuessr, a popular geography game, to supply examples of expert reasoning from language. Using this dataset, we present NAVIG, a comprehensive image geo-localization framework integrating global and fine-grained image information. By reasoning with language, NAVIG reduces the average distance error by 14% compared to previous state-of-the-art models while requiring fewer than 1000 training samples. Our dataset and code will be available at https:

Reasoning

The climate appears temperate with lush greenery suggesting a region with moderate rainfall, the architecture includes brick buildings and stone structures typical of Northern European styles. The overall environment is peaceful and rural, typical of the countryside in the UK, the speciﬁc style of the buildings and road infrastructure aligns with those in the Yorkshire region.

[Figure 3]

Map Search

Guidebook

[Figure 4]

[Figure 5]

The chevrons are black with white arrows in the United Kingdom.

The Croft Hotel, Northallerton Road, Darlington, North Yorkshire, England, United Kingdom", "lat": "54.4824", "lon": "-1.5561"

//github.com/SparrowZheyuan18/Navig/.

###### Location Prediction:

(54.4824, -1.5561). Darlington, U.K.

### 1 Introduction

Figure 1: In image geo-localization, models need to find both cultural and geographical clues to infer correct locations. External tools like maps and guidebooks can also be helpful by providing extra knowledge.

Image geo-localization—the task of predicting the location where an image was taken (Hays and Efros, 2008)—remains a challenging multimodal problem. For example, to say Figure 1 is a picture from Darlington (in England) requires reading the name of the hotel to determine possible candidates and excluding—for instance—the Croft hotel in Ontario based on architecture. Directly predicting the exact location of an image (Weyand et al., 2016; Haas et al., 2023; Cepeda et al., 2023) is difficult for computer vision models and requires extensive training on large image-location datasets.

suggest an Asian region; large spikes atop concrete poles narrow it down to Japan and Korea, and the black and yellow guardrails rule out Japan. While recent research integrates textual knowledge (Luo et al., 2022) and explicit clues (Zhang et al., 2024; Mendes et al., 2024; Li et al., 2024) with Vision Language Models (VLMs) to enhance accuracy, the reasoning in these models is often limited to a few words related to landmarks and does not provide a concrete analysis, as human experts would.

In contrast, human experts infer locations by reasoning. For example, in a GeoGuessr1 game video, an expert player, zi8gzag, explained how he identified a location in Korea: the presence of single yellow road lines and the language on the road signs

To date, these models’ reasoning remains more superficial than humans’ for two reasons: (1) Lack of high-quality reasoning datasets: Existing geotagged datasets lack linguistic reasoning elements,

* Equal contribution. 1http://www.geoguessr.com

while constructing a dataset that involves reasoning based on image details is resource-intensive. (2) Complexity of diverse information retrieval: Images often contain rich details, such as road signs, texts, and building styles, requiring additional tools for accurate retrieval and interpretation.

To address these questions, we introduce NAVICLUES, a detailed and high-quality reasoning dataset for image geo-localization, and NAVIG, a framework that combines both visual analysis and external knowledge to perform analytical reasoning. Inspired by the popular game GeoGuessr, NAVICLUES has over 2000 instances from five experienced YouTubers, recording their process of analyzing image details to infer locations, which trains VLMs to generate reasoning that mimics professional human players. With tools like public maps and expert-written guidebooks, we design a pipeline that dives into fine-grained details and retrieves relevant information to further enhance accuracy. We evaluate NAVIG against state-of-the-art models on two open benchmarks using five levels of prediction and ablate each component to investigate their contributions. NAVIG outperforms previous state-of-the-art models by a 14% reduction in average distance error while using less than 1000 training samples. We further illustrate the reasoning of NAVIG by providing examples of both successful and challenging cases. We release our dataset and framework to advance the use of reasoning in the field of image geo-localization.

### 2 Collecting NAVICLUES: Linking Places to Images

This section explains how we process the reasoning of GeoGuessr players to construct NAVICLUES. In addition, we analyze their reasoning and identify fifteen key clues humans use in geo-localization.

#### 2.1 Data Collection

Despite previous datasets containing image– location pairs (Hays and Efros, 2008; Vo et al., 2017; Astruc et al., 2024) and reasoning insights from guidebooks (Luo et al., 2022; Li et al., 2024), there is still a lack of datasets that capture the analytical reasoning process used to deduce locations from image details. To train NAVIG to generate reasoning (Section 3.1), we use the data from GeoGuessr, a popular game where players infer locations from street views, which preserve experts’ knowledge and strategies for image geo-

localization. We mine game data from “play along” videos of five popular YouTubers, along with transcripts of their reasoning during gameplay.

Data Mining. In a typical GeoGuessr game, there are multiple rounds of guessing the location from a new image. To segment the video transcript, we identify the timestamps of each round’s result pages by using Qwen-VL (Bai et al., 2023) to match the buttons and extract the corresponding scores. For images, we retrieve images from Google StreetView (GSV) API based on the coordinates of each round, omitting any unavailable ones.2 Following Haas et al. (2024), we capture images from four different directions and combine them to create 360-degree panoramic views (Table 1), which contain the same details as in the games. For reasoning data, we split the transcripts by round timestamps. The raw dataset contains 2637 images and respective locations.

Data Processing. To ensure data quality, we apply several processing steps: (1) we manually review and remove games where the visual content differs between the players’ view and the GSV data, such as games that allow movement or feature satellite view; (2) we exclude games with poor reasoning quality, including those with a time limit of less than 30 seconds, transcripts shorter than 100 words, or incorrect answers (where the GeoGuessr Score is lower than 3400, approximately 575 km); (3) answer-guided reasoning (Mendes et al., 2024) generates the step-by-step reasoning process from both the transcripts and images, guided by the correct locations and details mentioned by humans.

After processing, our final dataset, NAVICLUES, consists of 1120 images, each associated with a corresponding location (continent, country, and coordinates), reasoning, and scores. NAVICLUES is geographically well-distributed (Table 6), covering a diverse set of images in various countries to reduce potential data biases. NAVICLUES is useful for analyzing human geo-localization strategies and training models for related tasks. Prompts, examples, and other details are in the Appendix B.

Guidebook Data. For SEARCHER (Section 3.2), we collect guidebook data from two popular GeoGuessr community sources,3 which include tips from top players of GeoGuessr. Following Luo

- 2As GSV updates the images from time to time, some coordinates are deleted.
- 3https://somerandomstuff1.wordpress.com/2019/ 02/08/geoguessr-the-top-tips-tricks-and-techniq ues and https://www.plonkit.net

Image

plate

[Figure 6]

road

sign

car

pole

flag

mountain

Location Israel, Ashkelon. 31.66671, 34.59127

island

guy

language

###### Query

line

<image> Given an image, craft a brief and cohesive reasoning path that deduces this location based on the visual clues present in the image. Using a tone of exploration ...

location

tree

house

sun

###### Response

The scene reveals a blend of urban and natural features typical of a Mediterranean climate. Notably, the presence of palm trees suggests a warm, temperate region. The architecture of the buildings, characterized by flat roofs and a modern, functional style, often indicates environments with minimal snowfall. ... the neatly paved roads and sidewalks with occasional dust align with urban areas in regions with dry summers. ... the collective visual attributes and environmental clues point to a location consistent with Israel’s landscape and architectural style.

0 200 400 600 800

- Figure 2: Top clues in human reasoning. Humans identify roads, cars, poles, and linguistic clues—specifically the languages on plates, signs and houses.

et al. (2022) and Li et al. (2024), each instance in our dataset contains a clue (e.g., The chevrons are black with white arrows in the U.K.) paired with an image. The final dataset contains 6227 image– clue pairs that have distinguishable geographical features, which could be further integrated into geolocalization frameworks as additional knowledge (Examples are in Appendix B).

Table 1: Fine-tuning data format. The reasoning process leverages visual information in the images to deduce the correct location, such as climate, vegetation, building, and infrastructure. Complete prompts and response are in the Appendix.

#### 2.2 Data Analysis

We further analyze the reasoning of human players and investigate common patterns among human experts in geo-localization tasks. We use spaCy (Honnibal and Johnson, 2015) for noun extraction, allowing us to explore the specific information these experts focus on. We manually filter out irrelevant words and phrases, retaining only content pertinent to geographic reasoning.

This process identifies fifteen core clues frequently mentioned by experts reflecting common analytical patterns and reasoning strategies (Figure 2). The keywords cover cultural clues (e.g., language, flag, road, house) and natural geographical features (e.g., mountain, island, tree). This distribution conforms to categories in the guidebooks and further guides the implementation of our framework in Section 3.

### 3 NAVIG: Localizing Images with Reasoning and Tools

This section presents NAVIG: image geolocalization with reasoning about cultural and geographical clues and using external tools. YouTube experts both reason with image elements (the driving is on the left) and interpret image details

with guidebooks or maps (the Paria Main Road is in Toco) to locate an image. Based on this observation, NAVIG (Figure 3) has three components: (1) REASONER, which focuses on generating a reasoning process that analyzes the general information present in the image (Section 3.1); (2) SEARCHER, which uses additional knowledge sources and tools to explore the details (Section 3.2); and (3) GUESSER, which takes the concatenated outputs from both analyzers to determine the final location, which can be configured to any granularity of locations (Section 3.3).

3.1 Training VLMs to REASON about Image Locations

Recent VLMs can—sometimes—reason about the location of an image. (Li et al., 2024). However, the reasoning is limited to only a few words and does not help localization (Zhang et al., 2024). To enhance VLMs to reason location-relevant information in images, we create NAVICLUES and fine-tune VLMs using it to build REASONER. The reasoning includes geographical information such as climate, vegetation, building, and infrastructure (Table 1). This approach enables models to deduce locations from geographically pertinent details, ex-

###### Reasoner

###### Guesser

The climate appears tropical, given the lush greenery and palm trees, the scene is brightly lit by the sun which suggests a location near the equator; the signs feature English text, driving appears to be on the left side, typical in former British colonies, ... this place resembles regions in the Caribbean, speciﬁcally Trinidad and Tobago.

| |
|---|

[Figure 7]

Query: <image> Suppose you are an expert in image geo-localization, ... Here are some analysis for your reference:

Query

LoRA

Vision Language Model

R

[Figure 8]

Input Image VLM

Reasoning

- K1

- K2

Grounding Searcher

Vision Language Model

| |
|---|

[Figure 9]

House: Caribbean, Central America, and South America frequently use vibrant colors like bright red for buildings, to ...

House

[Figure 10]

| |
|---|

Road Sign

###### ...

Road Sign: search with OSM and ﬁnd

Guidebook

''Paria Main Road'' in Toco, Sangre

###### Grande, Trinidad and Tobago

###### ...

...

Country: Trinidad and Tobago CIty: Toco Latitude: 10.838 Longitude: -60.938

...

Kn

Map VLM

Building Sign

|[Figure 11]|
|---|

Building Sign: ''TOCO'' could be referring to a village on the coast of Trinidad.

Tools

Concat

Image Details

External Knowledge

- Figure 3: The framework of NAVIG comprises three main components: the REASONER, which handles general reasoning; the SEARCHER, which leverage external knowledge for detail-specific analysis, and the GUESSER, which combines outputs from both analyzers to generate predictions.

panding the depth and applicability.

After training, REASONER can generate a rationale for images, where given an image I, the fine-tuned VLM produces a reasoning R. However,

- as the reasoning relies solely on VLMs constrained by their parameterized knowledge, it lacks the information to understand specific details. For instance, human experts can search maps for text on buildings or road signs and consult guidebooks to identify the house style of a particular country, which goes beyond the intrinsic knowledge within VLMs. To emulate this process, an additional module, SEARCHER, integrates external tools, enabling more accurate interpretation of nuanced details.

#### 3.2 SEARCHING Image Details

The SEARCHER module extracts fine-grained details from images to enhance the reasoning by integrating relevant knowledge. It crops the image, generates queries, and retrieves external knowledge.

Grounding Image Details. As highlighted in Section 2.2, human experts often concentrate on specific elements in images, such as signs, houses, and roads, which provide crucial location-based clues. A precise grounding model generates highquality queries: given an image I and a predefined set of elements E = {e1,e2,...,en}, SEARCHER uses GroundingDino (Liu et al., 2023) to crop the

image according to E. Since each image may contain multiple instances of an element, the cropped images is defined as C = {ci,j | ei ∈ E,j ∈ [1,mi]}, where mi is the count of element ei in I. Specifically, we select road sign, building sign, and house from Figure 2 as elements, which align well with GroundingDino, since alternatives could yield overly large figures or uninformative results. Each cropped image ci,j is a query for specific tools. Additionally, if ci,j is a sign that contains text, textbased queries are generated with Optical Character Recognition (OCR) from Qwen2-VL (Wang et al., 2024). Therefore, the query set Q is:

##### Q = ⋃︂

{ci,j,OCR(ci,j) if ci,j ∈ signs}

i,j

Tools. The query set Q is then fed into a Tool Set T, which retrieves relevant knowledge. We use three tools for information retrieval: (1) The GeoGuessr Guidebook contains rich information for locating images (Section 2.1). Following prior research (Luo et al., 2022; Zhou et al., 2024), we frame Guidebook using as a Retrieval-Augmented Generation problem. Given an input image (e.g., a house as in Figure 3), we retrieve the most similar images. (2) Map. The map is a critical tool in image geo-localization: text in images (e.g., a

name on a sign) can pinpoint a location. We use OpenStreetMap4 for location retrieval, providing the top three search results, with the place name and multi-level location details. (3) VLM. We use an additional VLM as a tool by prompting it to identify details that might be overlooked in the REASONER. The VLM generates descriptions for details to narrow down potential locations (Figure 3). Each tool t in the Tool Set T contributes to the retrieval of additional knowledge K:

##### K = ⋃︂

t(Q)

t∈T

Further implementation details in the Appendix A.

#### 3.3 GUESSING the Final Location

The GUESSER uses all prior information to generate the final prediction. It concatenates the reasoning R from the REASONER with the external knowledge K retrieved by the SEARCHER, forms them into a prompt template p along with the image i, and makes the location prediction with a VLM:

(︁

)︁

yˆloc = VLMp

I,concat(R,K)

where yˆloc is the model’s generated location. The prompt p is configurable to flexibly adjust to specific output formats, such as various location levels (e.g., country, city, and coordinates).

### 4 How Well Does NAVIG Reason Image Locations?

We compare NAVIG against prior state-of-the-art image geo-localization models and other baseline approaches (Section 4.2), ablate each module to evaluate their contributions (Section 4.3), and provide qualitative examples to highlight successful and challenging cases (Section 4.4).

#### 4.1 Experimental Setup

Implementation. We use three open-source models in NAVIG: MiniCPM-V (Yao et al., 2024), LLaVA (Liu et al., 2024), and Qwen2-VL (Wang et al., 2024). These models serve as VLMs for REASONER, SEARCHER, and the GUESSER components within the NAVIG framework. (1) For REASONER, Low-Rank Adaptation (LoRA) (Hu et al., 2022) fine-tunes models using NAVICLUES. We use minicpm-v-2.6, llava-1.6-vicuna-7b, and qwen2-vl-7b for their advanced performance and

4https://www.openstreetmap.org/

mid-range size, which align with our cost constraints. (2) For SEARCHER, we select the top three cropped clues as the basis for generation (e.g., if multiple houses are cropped, only will the three with the highest similarity be analyzed). CLIP (Radford et al., 2021) encodes guidebook images and query images, retrieving guidebook data by the Euclidean distance d between image embeddings (FAISS (Johnson et al., 2019)), returning associated text if d is below a threshold dt (set to 30). (3) We prompt the GUESSER to predict locations at the coordinates level. Training hyperparameters, model configurations, and prompts are in Appendix A.

Baselines. We compare NAVIG with two baselines: (1) Geo-localization Models: we select top-performing open-source models from prior research in image geo-localization: G3 (Luo et al., 2022), GeoCLIP (Cepeda et al., 2023), and StreetCLIP (Haas et al., 2023). (2) Vision Language Models: we select vanilla MiniCPM-V, LLaVA, Qwen2VL as baselines, consistent with the backbone models used in NAVIG. The prompts for these VLM baselines are identical to those in NAVIG but lack analyses. We do not include commercial closedsource models (as discussed in Limitations).

Dataset and Metrics. Following previous work (Hays and Efros, 2008; Astruc et al., 2024; Haas et al., 2024), we evaluate our framework on two public datasets, including GWS5K sampled from GWS15K (Clark et al., 2023) due to cost constraints, and Im2GPS3k (Hays and Efros, 2008). We first computed the haversine distance between predicted and ground truth coordinates. For models limited to city level outputs, we use the coordinates of the predicted city as their predictions. Next, we evaluated the prediction accuracy—the percentage of guesses that fall within a distance threshold from the correct location—at five geographic levels: Street (1 km), City (25 km), Region (200 km), Country (750 km), and Continent (2,500 km). In addition, we calculated the average error distance and GeoGuessr Score, a metric from the original GeoGuessr game that quantifies guess accuracy, with a scoring range of 0 to 5000. Details about metric computation are in Appendix C.

#### 4.2 Main Experiments

Accuracy. We compare NAVIG with state-ofthe-art image geo-localization models and Vision Language Models (GWS5k results in Table 2). (1) NAVIG (Qwen2-VL) has the highest accuracy across all metrics, beating specialized geo-

Continent 2,500 km

Country 750 km

Region 200 km

City 25 km

Street 1 km

Distance↓ Score↑

Model

G3 50.9 14.6 2.3 0.1 0.0 4,341 1,304 GeoCLIP 78.2 46.5 17.1 3.5 0.4 2,099 2,613 StreetCLIP 79.4 43.4 13.4 1.7 0.3 2,060 2,543

MiniCPM-V 27.1 15.9 6.7 1.6 0.1 7,320 909 LLaVA 43.9 23.1 7.0 1.2 0.0 5,096 1,418 Qwen2-VL 89.4 66.7 31.8 6.1 0.1 1,124 3,344

###### NAVIG

- - MiniCPM-V 71.5 44.1 16.9 3.5 0.3 2,956 2,413
- - LLaVA 74.7 39.4 12.0 1.9 0.3 2,243 2,354
- - Qwen2-VL 91.1 66.9 31.9 6.7 0.7 965 3,389

- Table 2: Accuracy and scores on GWS5k. The data from Continent to Street represents the accuracy (%) at each level. The three sections are geo-localization models, VLMs, and NAVIG. Bold font indicates the best performance. NAVIG (Qwen2-VL) achieves the highest accuracy across all metrics.

Model ROUGE

F1 R1 R2 RL

REASONER (MiniCPM-V) 51.0 14.8 24.6 MiniCPM-V 46.4 12.6 22.1 REASONER (LLaVA) 49.8 13.9 24.0 LLaVA 44.7 10.8 21.8 REASONER (Qwen2-VL) 51.4 14.6 24.3 Qwen2-VL 45.2 12.3 22.1

- Table 3: ROUGE F1 scores for reasoning generated by models and humans (%). REASONER models reason more similarly to humans.

Level Frequency Accuracy Country 100.0 79.6 Others 50.7 3.0

Table 4: Frequency and accuracy (exact match) of REASONER. “Others” indicates more detailed predictions, which are challenging.

combination of these elements point towards a city like Chaco, Argentina” while the correct answer is “Trelew, Argentina.” This indicates the importance of SEARCHER for precise predictions.

localization models trained on domain-specific datasets, despite its relatively compact size of only 7 billion parameters. (2) All VLMs generate effective analytical reasoning trained with only around 1,000 samples and beat their vanilla models. These findings underscore the quality of training data and the efficacy of NAVIG. Similar results on Im2GPS3k are in Appendix D.

Comparison with Humans. We also compare NAVIG’s performance against human players in fifty randomly sampled GeoGuessr games (collected in Appendix B), focusing on common metrics for country, city, and street level predictions. Filtered human player data have time and access to additional knowledge resources (e.g., maps and guidebooks) for a fair comparison with NAVIG. NAVIG outperforms humans in overall scores (Table 5), although humans excel at finer-grained predictions by iteratively cross-referencing maps and comparing terrain and features within the game. This highlights a future direction to use non-textual features to refine map-based searches and enhance street-level accuracy of models.

Reasoning. We evaluate the linguistic reasoning quality generated by the model on a reserved test set of 50 games. To measure the alignment between model and human reasoning, we compute their ROUGE scores (Lin, 2004) which illustrate whether the model simulates human reasoning. REASONER achieves higher ROUGE scores across all models and metrics after training (Table 3).

#### 4.3 Ablation Study

We apply GPT-4o to label the granularity and accuracy (measured by exact match) of the reasoning by NAVIG (Qwen2-VL) on GWS5k (Table 4). REASONER predicts country with an accuracy of 79.6%, while it’s challenging (3.0%) when it makes finer-grained predictions (e.g., city, town, or street), as these predictions require additional information. For example, REASONER outputs “the

To illustrate the contributions of each component in NAVIG, we ablate the reasoning training, the impact of REASONER, and SEARCHER. Table 6 presents the three VLMs’ accuracy on GWS5k. In this setup, NAVIG represents our framework, “w/o training” denotes results with the same prompt but without training on NAVICLUES, “w/o Macro” and

Model Country City Street Score↑ NAVIG

- - MiniCPM-V 56.0 18.0 0.0 2,863
- - LLaVA 48.0 14.0 0.0 2,690
- - Qwen2-VL 86.0 32.0 4.0 4,202

Human Players 76.0 48.0 42.0 3,757

- Table 5: Performance between humans and NAVIG. The data from City to Street represents accuracy (%). Our best model beats humans with a higher overall score but still struggles to achieve fine-grained accuracy.

Model Country City Street NAVIG (MiniCPM-V) 44.1 3.5 0.3

- - w/o training - 3.3 - 0.4 - 0.2
- - w/o REASONER - 10.2 - 0.7 - 0.0
- - w/o SEARCHER - 0.3 - 0.3 - 0.2
- - MiniCPM-V - 14.9 - 0.5 - 0.2

NAVIG (LLaVA) 39.4 1.9 0.3

- - w/o training - 25.8 - 1.2 - 0.3
- - w/o REASONER - 20.2 - 0.8 - 0.0
- - w/o SEARCHER + 0.4 - 0.2 - 0.2
- - LLaVA - 16.3 - 0.7 - 0.3

NAVIG (Qwen2-VL) 66.9 6.7 0.7

- - w/o training - 6.0 - 0.9 - 0.5
- - w/o REASONER - 4.0 - 0.6 - 0.2
- - w/o SEARCHER + 0.1 - 0.9 - 0.5
- - Qwen2-VL - 0.2 - 0.6 - 0.6

- Table 6: Ablation results of NAVIG on the GWS5k dataset. Each component contributes to model accuracy, with their removal leading to notable declines across Country, City, and Street levels.

“w/o Micro” refer to the results without the REASONER and SEARCHER modules, respectively.

Results. (1) Each module contributes to improving the model’s accuracy. (2) Surprisingly, when the model is prompted to zero-shot generate reasoning, it can be misleading and decrease final accuracy. This highlights the necessity of training with NAVICLUES. (3) REASONER plays a critical role in coarse-grained localization, with improvements

- at the country level and decreases without it, as the reasoning in NAVICLUES is limited to the country and city level. (4) SEARCHER substantially enhances fine-grained reasoning. Precise street-level localization on the GWS dataset is challenging, but the SEARCHER narrows the scope within 1 km for images containing textual information using map searches (Table 2). Results on Im2GPS3k are in Appendix D, which is consistent with GWS.

#### 4.4 Qualitative Analysis

This section examines how the analytical reasoning derived from images contributes to NAVIG ’s inference process. NAVIG closely examines details within the image (e.g., climate, orientation, and “Lower Mill” in Figure 4 (top)) to determine the location. This detailed reasoning narrows down the possible range, while integration with OpenStreetMap data further aids the model in finding the restaurant, with an error distance of under 1 meter, improving its estimate by 144 km.

However, image elements can also mislead the model. In Figure 4 (middle), the model fixates on a shop name in the image, “KLICK”, which can be interpreted as a German word. This leads the reasoning process astray, resulting in an incorrect localization. OpenStreetMap can also lead to false predictions when there are places with the same name, such as “Bradesco”, a well-known Brazilian bank (Figure 4, bottom). The reasoning makes image geo-localization models more interpretable by revealing how image elements influence decisions.

### 5 Related Work

Image geo-localization. Image geo-localization falls into three methods: (1) Retrieval-based methods retrieves the most similar images (Hays and Efros, 2008; Zhu et al., 2023). Various retrievers (Vo et al., 2017; Pramanick et al., 2022; Haas et al., 2023) and gallery types (Cepeda et al., 2023) have been proposed. (2) Classification-based methods divide geographical maps into distinct classes and train models to classify the images into these categories with different model structures (Radford et al., 2021; Wu and Huang, 2022) and map division strategies (Weyand et al., 2016; Theiner et al., 2022; Haas et al., 2024). (3) Generationbased methods use visual understanding and generation in Vision Language Models (VLMs) to directly generate the location or coordinates for geolocalization. Aligning visual content with rich text descriptions and reasoning (Jia et al., 2024; Li et al., 2024; Zhang et al., 2024) and incorporating external knowledge through Retrieval-Augmented Generation (Luo et al., 2022; Zhou et al., 2024) are state-of-the-art. However, challenges persist in effectively using VLMs, including limited reasoning data and relying on constrained knowledge sources.

Visual Reasoning. Unlike previous methods, we treat geo-localization as a complex visual reasoning task that deduces the correct location with

[Figure 12]

###### City:

New York, Country:

THE LOWER MILL RESTAURANGT & GALLERIES

United States, Lat:

42.9559925 Lon:

-77.588744

Reasoning: The area appears to be in a temperate climate, ... the road signs and vehicles suggest driving on the right side, ..., the overall setting feels quaint and peaceful, typical of a small town in the northeastern United States, ... the speciﬁc name 'Lower Mill' is associated with a town in New York, USA, known for its historical and cultural signiﬁcance.

Search ("The Lower Mill Restaurant") → {"place_name": "The Lower Mill", "location": "The Lower Mill, New York, United States", "lat": "42.9560001", "lon": "-77.5887408"}

Prediction: (42.9560, -77.5887) Distance: 0.0009km

[Figure 13]

KLICK

City:

Russia, Country:

Ivangorod, Lat:

59.3797404 Lon:

28.1873023

Reasoning: The location is in Europe, ... the language on signs suggests a Germanic inﬂuence, the presence of a shopping mall indicates a commercial area, the name on one of the signs is 'Klick', which is a German word, the presence of a 'Silver Dream' sign suggests ... the overall setting and language suggest a city in Germany, possibly Berlin.

VLM (House) → The image depicts a shopping mall interior with ... This type of setting is commonly found in urban areas, particularly in Europe or North America, where ...

Prediction: (52.52, 13.405) Distance: 1190.79km

[Figure 14]

City:

Brazil, Country:

Campos Belos, Lat:

Bradesco

- -13.9364983 Lon:
- -46.7731332

Reasoning: The image shows a small town with distinct features that suggest a South American setting. The presence of Portuguese signage, such as "Bradesco," a well-known Brazilian bank, strongly indicates Brazil. The architecture ... The road infrastructure and vehicles ... The combination of these elements points conﬁdently to Brazil as the location.

Search ("Bradesco") → {"place_name": "Bradesco", "location": "Bradesco, Ulianópolis, Pará, Região Norte, Brasil", "lat": "-3.5893533", "lon": "-47.4905467"}

Prediction: (-3.5893, -47.4905) Distance: 1053.43km

- Figure 4: Top: The model uses visual details and OpenStreetMap to accurately determine the location. Middle: The model is misled by linguistic elements—the shop name, resulting in an incorrect inference. Bottom: The model found a namesake when using OpenStreetMap.

language, requiring both visual understanding and reasoning capabilities (Hudson and Manning, 2019; Gupta and Kembhavi, 2023). As VLMs have demonstrated exceptional accuracy in visual reasoning tasks (Alayrac et al., 2022; Lu et al., 2023), methods enhancing the visual reasoning of VLMs in specific tasks include: (1) High-quality reasoning data, which researchers have shown to be particularly effective in improving the performance of VLMs (Du et al., 2023; Chen et al., 2023); (2) Vision grounding, which enables models to ground in the details of the image and perform step-by-step reasoning (Qi et al., 2024; Wu and Xie, 2023; Zhang et al., 2024); and (3) Tool us-

ing and retrieval-augmented generation, which aid the model by leveraging tools to retrieve additional knowledge (Yang et al., 2023; Marino et al., 2021; Chen et al., 2022) and reduce hallucinations. We integrate these insights to improve NAVIG.

### 6 Conclusion

We introduce a reasoning dataset NAVICLUES and a novel framework NAVIG with detailed visual reasoning and knowledge retrieval for image geolocalization. Evaluated with relatively small backbone models, NAVIG is highly accurate by using the reasoning chains of human experts, providing

interpretable, verifiable evidence that a downstream user can use to validate its predictions. Future work could include detailed annotations to images more than street views, expanding more tools to improve finer-grained predictions, and using interpretable reasoning to assist geo-localization applications.

### Limitations

#### Dataset

(1) Limited data size. In this work, we utilize data from human players in the GeoGuessr game to train Vision Language Models for performing geographic reasoning on images. The copyright and usage rights of the images are subject to that of Google Street View. However, the size of NAVICLUES is limited due to the scarcity of available data on YouTube and the data noise.

- (2) Panoramic street view images. To simulate the perspective of players in the GeoGuessr game, we use stitched panoramic images as the input to the model. Furthermore, nearly all images in the data from GeoGuessr are street views, despite our efforts to ensure a geographically balanced distribution of data across countries. This limits its distribution, as there’s more weather, street, car, and vegetation information in street views than in other images. Models trained with NAVICLUES might be weak at images with less street-level information.
- (3) Future work could consider expanding the training dataset by incorporating images of different sizes and types, including more detailed annotations to create dataset s more than street views, to further enhance the performance of image geolocalization tasks with better reasoning.

#### Models

(1) Limited model sizes. Due to cost constraints, we are unable to train larger models and conduct our experiments using top-performing, mediumsized open-source models (around 7B parameters). While this choice may result in performance that is not as competitive as larger models, it ensures a practical balance between computational feasibility and model efficacy. We also refrain from using closed-source models, as their lack of transparency regarding training data and inability to be trained on NAVICLUES make them unsuitable for fair comparison.

(2) Limited tool sets. We evaluated only a limited set of tools and grounding words in SEARCHER. Identifying more geographic features such as cars,

road markings, and poles would require more precise recognition methods and more sophisticated model designs, which could potentially improve performance.

- (3) Complexity of subsystems. We employ a

pipeline approach to construct our model, aiming to maximize the performance of each component at every stage. However, this process introduces knowledge from different resources, which might conflict with each other. Currently, we implement a Guesser to handle the potential conflict and show the contribution of each ablated subsystems. We also examine the reasoning from REASONER to show the necessity of SEARCHER.

- (4) Future works can focus on including larger

backbone models to further improve the performance, adding more tools, and conduct end-to-end training to better integrate the information, or add another fact-checking module to better discern information.

### Ethical Considerations

#### Data Collection

In this work, we use the data from GeoGuessr players on YouTube to train our models. We carefully process the data and remove the personal information of the players, using all data for academic and non-commercial purposes, and giving appropriate credit to them in this paper. We make sure the use of our data is acceptable under YouTube’s copyright policies and the Fair Use guidelines.

#### Model Usage

While the task of image geo-localization has the potential to enable innovative applications in fields such as navigation and tourism, the misuse of these models could also lead to risks such as privacy breaches and surveillance. In our work, we ensured that all training and testing data came from publicly available sources, with no involvement of private or personal images or location data. Currently, as shown in our experiments, these models have not yet reached a level of precision to accurately predict coordinates-level locations. For the future development of this field, it is crucial for researchers to ensure that these models are used within appropriate boundaries to prevent the leakage of private information.

### References

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karén Simonyan. 2022. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

Guillaume Astruc, Nicolas Dufour, Ioannis Siglidis, Constantin Aronssohn, Nacim Bouia, Stephanie Fu, Romain Loiseau, Van Nguyen Nguyen, Charles Raude, Elliot Vincent, et al. 2024. Openstreetview5m: The many roads to global visual geolocation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21967– 21977.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-vl: A versatile visionlanguage model for understanding, localization, text reading, and beyond.

Vicente Vivanco Cepeda, Gaurav Kumar Nayak, and Mubarak Shah. 2023. Geoclip: Clip-inspired alignment between locations and images for effective worldwide geo-localization. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Keqin Chen, Zhao Zhang, Weili Zeng, Richong Zhang, Feng Zhu, and Rui Zhao. 2023. Shikra: Unleashing multimodal llm’s referential dialogue magic. ArXiv preprint, abs/2306.15195.

Zhuo Chen, Yufeng Huang, Jiaoyan Chen, Yuxia Geng, Yin Fang, Jeff Z Pan, Ningyu Zhang, and Wen Zhang. 2022. Lako: Knowledge-driven visual question answering via late knowledge-to-text injection. In Proceedings of the 11th International Joint Conference on Knowledge Graphs, pages 20–29.

Brandon Clark, Alec Kerrigan, Parth Parag Kulkarni, Vicente Vivanco Cepeda, and Mubarak Shah. 2023. Where we are and what we’re looking at: Query based worldwide image geo-localization using hierarchies and scenes. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 23182–23190. IEEE.

Yifan Du, Hangyu Guo, Kun Zhou, Wayne Xin Zhao, Jinpeng Wang, Chuyuan Wang, Mingchen Cai, Ruihua Song, and Ji-Rong Wen. 2023. What makes

for good visual instructions? synthesizing complex visual reasoning instructions for visual instruction tuning. ArXiv preprint, abs/2311.01487.

Tanmay Gupta and Aniruddha Kembhavi. 2023. Visual programming: Compositional visual reasoning without training. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023, Vancouver, BC, Canada, June 17-24, 2023, pages 14953–14962. IEEE.

Lukas Haas, Silas Alberti, and Michal Skreta. 2023. Learning generalized zero-shot learners for opendomain image geolocalization. ArXiv preprint, abs/2302.00275.

Lukas Haas, Michal Skreta, Silas Alberti, and Chelsea Finn. 2024. Pigeon: Predicting image geolocations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12893– 12902.

James Hays and Alexei A. Efros. 2008. IM2GPS: estimating geographic information from a single image. In 2008 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2008), 24-26 June 2008, Anchorage, Alaska, USA. IEEE Computer Society.

Matthew Honnibal and Mark Johnson. 2015. An improved non-monotonic transition system for dependency parsing. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 1373–1378, Lisbon, Portugal. Association for Computational Linguistics.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.

Drew A. Hudson and Christopher D. Manning. 2019. GQA: A new dataset for real-world visual reasoning and compositional question answering. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2019, Long Beach, CA, USA, June 16-20, 2019, pages 6700–6709. Computer Vision Foundation / IEEE.

Pengyue Jia, Yiding Liu, Xiaopeng Li, Xiangyu Zhao, Yuhao Wang, Yantong Du, Xiao Han, Xuetao Wei, Shuaiqiang Wang, and Dawei Yin. 2024. G3: An effective and adaptive framework for worldwide geolocalization using large multi-modality models. ArXiv preprint, abs/2405.14702.

Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2019. Billion-scale similarity search with gpus. IEEE Transactions on Big Data, 7(3):535–547.

Ling Li, Yu Ye, Bingchuan Jiang, and Wei Zeng. 2024. GeoReasoner: Geo-localization with reasoning in street views using a large vision-language model. In Proceedings of the 41st International Conference on

Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 29222–29233. PMLR.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. 2024. Llavanext: Improved reasoning, ocr, and world knowledge.

Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, et al. 2023. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. ArXiv preprint, abs/2303.05499.

Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, KaiWei Chang, Ying Nian Wu, Song-Chun Zhu, and Jianfeng Gao. 2023. Chameleon: Plug-and-play compositional reasoning with large language models. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Grace Luo, Giscard Biamby, Trevor Darrell, Daniel Fried, and Anna Rohrbach. 2022. G3: Geolocation via guidebook grounding. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 5841–5853, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Kenneth Marino, Xinlei Chen, Devi Parikh, Abhinav Gupta, and Marcus Rohrbach. 2021. KRISP: integrating implicit and symbolic knowledge for opendomain knowledge-based VQA. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021, pages 14111–14121. Computer Vision Foundation / IEEE.

Ethan Mendes, Yang Chen, James Hays, Sauvik Das, Wei Xu, and Alan Ritter. 2024. Granular privacy control for geolocation with vision language models. ArXiv preprint, abs/2407.04952.

Shraman Pramanick, Ewa M Nowara, Joshua Gleason, Carlos D Castillo, and Rama Chellappa. 2022. Where in the world is this image? transformer-based geo-localization in the wild. In European Conference on Computer Vision, pages 196–215. Springer.

Ji Qi, Ming Ding, Weihan Wang, Yushi Bai, Qingsong Lv, Wenyi Hong, Bin Xu, Lei Hou, Juanzi Li, Yuxiao Dong, et al. 2024. Cogcom: Train large visionlanguage models diving into details through chain of manipulations. ArXiv preprint, abs/2402.04236.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International

Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR.

Jonas Theiner, Eric Müller-Budack, and Ralph Ewerth. 2022. Interpretable semantic photo geolocation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 750–760.

Nam N. Vo, Nathan Jacobs, and James Hays. 2017. Revisiting IM2GPS in the deep learning era. In IEEE International Conference on Computer Vision, ICCV 2017, Venice, Italy, October 22-29, 2017, pages 2640– 2649. IEEE Computer Society.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. 2024. Qwen2vl: Enhancing vision-language model’s perception of the world at any resolution. ArXiv preprint, abs/2409.12191.

Tobias Weyand, Ilya Kostrikov, and James Philbin. 2016. Planet-photo geolocation with convolutional neural networks. In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part VIII 14, pages 37–55. Springer.

Meiliu Wu and Qunying Huang. 2022. Im2city: image geo-localization via multi-modal learning. In Proceedings of the 5th ACM SIGSPATIAL International Workshop on AI for Geographic Knowledge Discovery, pages 50–61.

Penghao Wu and Saining Xie. 2023. v*: Guided visual search as a core mechanism in multimodal llms. ArXiv preprint, abs/2312.14135.

Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. 2023. Mmreact: Prompting chatgpt for multimodal reasoning and action. ArXiv preprint, abs/2303.11381.

Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. 2024. Minicpm-v: A gpt-4v level mllm on your phone. ArXiv preprint, abs/2408.01800.

Gengyuan Zhang, Yurui Zhang, Kerui Zhang, and Volker Tresp. 2024. Can vision-language models be a good guesser? exploring vlms for times and location reasoning. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 636–645.

Zhongliang Zhou, Jielu Zhang, Zihan Guan, Mengxuan Hu, Ni Lao, Lan Mu, Sheng Li, and Gengchen Mai. 2024. Img2loc: Revisiting image geolocalization

using multi-modality foundation models and imagebased retrieval-augmented generation. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2749–2754.

Sijie Zhu, Linjie Yang, Chen Chen, Mubarak Shah, Xiaohui Shen, and Heng Wang. 2023. R2former: Unified retrieval and reranking transformer for place recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19370–19380.

### A Implementation Details

#### A.1 Training Parameters

We trained the REASONER on Nvidia RTX 6000 Ada (48G), with CUDA 12.4, Transformers 4.45.1, and Pytorch 2.1.2.

Parameters Value

Max Length 2048 LoRA Rank 8 LoRA Alpha 32 Optimizer AdamW

- Adam Beta1 0.9
- Adam Beta2 0.95 Learning Rate 1e-4 Warmup Ratio 0.05 LR Scheduler Type cosine Batch Size 1 Weight Decay 0.1

Table 7: Training parameters for REASONER.

#### A.2 Other Parameters.

For reproducibility, we also provide the parameters used in other modules and VLMs within our framework.

GroundingDino. We utilize GroundingDino to crop detailed information from the images, such

- as signs and houses. We observe variation in the features of images across different datasets. For instance, the GWS5k dataset focuses on street scenes, and other datasets contain considerable noise (e.g., animals). Consequently, to reduce noise that could potentially affect model performance, we empirically set the thresholds as follows: Box-Threshold

= 0.5 and Text-Threshold = 0.5 for GWS5K, and Box-Threshold = 0.8 and Text-Threshold = 0.6 for Im2GPS3k.

Retrieval-Augmented Generation. We employ CLIP as the image encoder for guidebook clues, using ViT-B-32 as the vision encoder. The guidebook database is deployed with FAISS, and similarity is calculated using Euclidean Distance. The number of most relevant retrieved images, k, is set to 3, with a similarity threshold of 30.

OpenStreetMap. We use the Nominatim Search API to process map searches, which takes text queries, and return the most relevant results along with the place name, address, and coordinates.

Vision Language Models. We use VisionLanguage Models in our framework for reasoning and location inference. The three models are minicpm-v-2.6, llava-1.6-vicuna-7b, and

qwen2-vl-7b. Each model is configured with a temperature of 0 and an output length of 2048.

#### A.3 Prompts for VLMs.

In Table 8 and Table 9, we present the prompts used in NAVIG for Vision Language Models. Four distinct prompts are employed: (1) the Data Processing Prompt, which employed an answer guided reasoning generation method to prompt VLMs in extracting step-by-step reasoning from YouTube transcripts; (2) the REASONER Prompt, which is the same as the query in the training data, prompting VLMs to generate a coherent reasoning process to infer the location within an image; (3) the SEARCHER Prompt, which generates additional knowledge from image details, and (4) the GUESSER Prompt, which synthesizes all prior information to make a final prediction.

Data Processing Prompt

<image> Given an image and the known location details (Country: country, Latitude: lat, Longitude: lon), and an expert’s analysis of the location (transcript), craft a brief and cohesive reasoning path that deduces this location based on the visual clues present in the image. Begin your reasoning without revealing that you know the exact location, using a tone of exploration and inference. Carefully analyze and link observations of natural features (climate, vegetation, terrain), man-made structures (roads, buildings, signage), and distinct landmarks. Allow these observations to naturally lead you to the correct country, enhancing the accuracy of your deductions. Ensure that while the narrative seems to be guessing, it aligns with the known country, confirming the reliability of your reasoning without stating the specific coordinates. Start the reasoning without any intro, and make sure to make it brief.

Table 8: The prompts used in NAVIG.

B Data.

In this section, we present the data processing workflows and provide more detailed information on the various types of data used in the system.

#### B.1 Data Processing.

YouTubers. We utilized the scripts of five professional GeoGuessr players’ YouTube videos as the starting data for our reasoning generation. We thank these five players for their contributions to knowledge dissemination and promotion of image geo-localization: zi8gzag, GeoWizard, GeoPeter, Geogasm, and RAINBOLT TWO.

Data Processing. We used the Google Street View5 API to retrieve images for our dataset. We

5https://www.google.com/streetview/

selected a resolution of 640×640 pixels (the maximum resolution accepted by GSV), a field of view (FOV) of 90, and headings of 0, 90, 180, and 270 degrees to obtain four images. Stitching them together produces a complete street view image, providing the same amount of information that a GeoGuessr player would see.

Next, we split the videos for retrieving the transcripts or each round. After a player submits their final guess, the game reveals the distance between their guessed location and the actual coordinates, where the player can choose to either proceed to the next round or end the challenge. We use precise pixel coordinates in conjunction with OCR technology to detect the presence of the “Next” or “End” buttons and split the videos. We sample frames

- at a rate of 1/6 per second to ensure no scene is missed. Additionally, we extract the GeoGuessr Score displayed beside the button and collect human players’ scores. Next, due to the noise in the data (with many informal language from players), we provide GPT-4o with the correct locations for paraphrasing and generating higher quality and more coherent data.

REASONER Prompt

<image> Given an image, craft a brief and cohesive reasoning path that deduces this location based on the visual clues present in the image. Using a tone of exploration and inference. Carefully analyze and link observations of natural features (climate, vegetation, terrain), man-made structures (roads, buildings, signage), and distinct landmarks. Allow these observations to naturally lead you to the correct country, enhancing the accuracy of your deductions. Start the reasoning without any intro, and make sure to make it brief.

SEARCHER Prompt

<image> Analyze the {item} images to determine the region with the highest likelihood of finding this type of {item}. For each image, provide only the core reasoning in one sentence. Don’t say you can’t determine, try your best as it’s a geo-localization game

###### GUESSER Prompt

<image> <information> Using the provided information as a reference, estimate the location depicted in the image with as much accuracy and precision as possible. Generally, you might use the reasoning to roughly locate the coarse-grained location, and use other information to help you decide more precisely. Use your own knowledge as well. Aim to deduce the exact coordinates whenever feasible. Format your response strictly as JSON in the following structure:{“country”: “<country_name>”, “city”: “<city_name>”, “latitude”: <Latitude Coordinate>, “longitude”: <Longitude Coordinate>} Ensure the JSON output is correctly formatted. Provide a well-informed estimate for each value, avoiding any empty fields. Do not include additional information or commentary.

Table 9: The prompts used in NAVIG.

#### B.2 Data Demonstration.

In this section, we present examples and key statistics for both NAVICLUES and guidebook datasets.

NAVICLUES. Each data includes a panoramic image, the corresponding location, and a highquality reasoning process that shows how geographical and cultural information is used to infer the location (Figure 10). To reduce hallucination and bias, the model is not required to generate specific street-level locations or coordinates directly, but carefully analysis about image elements (e.g., climate) that collectively lead to the prediction. NAVICLUES is geographically well-distributed, covering various countries across the globe (Figure 5).

[Figure 15]

Figure 5: Location distribution of NAVICLUES, covering a wide range of countries around the world.

Guidebooks. As shown in Table 11, each guidebook entry consists of an image-text pair, where the text describes the location and features depicted in the image. We utilize RAG in NAVIG to identify the most similar image and use the accompanying text to support the reasoning process.

Statistical Results. We present the statistical results of NAVICLUES in Figure 6. Most distances are relatively small, indicating that players are often successful in deducing the correct location. The reasoning process averages 842 words in length, demonstrating high quality with extensive details.

C Metrics Definition

#### C.1 GeoGuessr Score

The GeoGuessr Score is a metrics introduced in the GeoGuessr game to quantify the accuracy of a guess. It is defined as:

GeoGuessr Score = 5000 × e−1492d .7 (1)

where d represents the distance between the guessed coordinates and the actual coordinates in kilometers. Therefore, a more accurate guess yields a GeoGuessr Score closer to 5,000.

###### Image

[Figure 16]

Location Chile. -27.1265479, -109.2876917

###### Reasoning

These imposing stone figures are situated on a grassy hillside overlooking the ocean, suggesting a coastal location. The sparse vegetation and the presence of unique megalithic structures indicate a historical and possibly isolated environment. The statues, known as Moai, are characteristic of an island rich in archaeological history. Such a distinct culture is associated with an island famous for these figures, which is set far away from mainland destinations. This points toward the Polynesian island in the southeastern Pacific known for these world-renowned statues, aligning closely with a location associated with Chile.

###### Image

[Figure 17]

Location China. 22.27992557066081, 114.1648415981852

###### Reasoning

Tall, sleek skyscrapers dominate the cityscape, suggesting a prominent urban environment. The distinct yellow license plates on vehicles are a key hint, typically associated with regions influenced by British rule. The street signs feature both English and Chinese characters, which points to a bilingual city. The presence of a distinctive red double-decker bus is highly characteristic of a city with British influence. All these clues, combined with the modern architecture and dense urban vibes, strongly suggest this is Hong Kong, likely within its central business district.

###### Image

[Figure 18]

Location Italy. 42.71658113303754, 13.01051150781922

###### Reasoning

The landscape features rolling hills and a dense cover of greenery, indicative of a temperate climate typically found in parts of Southern Europe. The road layout with its narrow curve and the types of vehicles suggest a European setting. The presence of Italian-language signage such as Cascia and recognizable Italian road markers, like narrow front plates, strongly points to Italy. The architecture, with its rustic and modest buildings in the distance, complements the rural Italian countryside vibe. Hence, these visual cues collectively affirm the location as Italy, likely in a more central to southern region given the sign for Cascia.

Table 10: Demonstration of NAVICLUES.

Mean: 159.41 km

Mean: 842.23 characters

300

80

70

250

60

200

Frequency

Frequency

50

150

40

30

100

20

50

10

0

0

0 100 200 300 400 500 600 Distance (kilometers)

400 600 800 1000 1200 1400 Length of Reasoning (characters)

Figure 6: Distribution of Distance (left) and Length of Reasoning (right) in NAVICLUES.

Image

[Figure 19]

Text (Toptips) Houses in Brittany, a western region of France, are coloured white with dark roofs.

###### Image

[Figure 20]

Text (Plonkit) Tunisia has a fairly unique stop sign with Arabic and Latin script.

Table 11: Examples of clues in the two guidebooks, Toptips and Plonkit. The data is constructed as imagetext pairs in the guidebooks.

Image

[Figure 21]

Location Israel, Ashkelon. 31.66671, 34.59127

###### Query

<image> Given an image, craft a brief and cohesive reasoning path that deduces this location based on the visual clues present in the image. Using a tone of exploration and inference. Carefully analyze and link observations of natural features (climate, vegetation, terrain), man-made structures (roads, buildings, signage), and distinct landmarks. Allow these observations to naturally lead you to the correct country, enhancing the accuracy of your deductions. Start the reasoning without any intro, and make sure to make it brief.

###### Response

The scene reveals a blend of urban and natural features typical of a Mediterranean climate. Notably, the presence of palm trees suggests a warm, temperate region. The architecture of the buildings, characterized by flat roofs and a modern, functional style, often indicates environments with minimal snowfall. Observing the infrastructure, the neatly paved roads and sidewalks with occasional dust align with urban areas in regions with dry summers. The use of earth-toned paint on buildings matches the aesthetic found in Middle Eastern locales. Further exploration of the layout reveals hints of both residential and possibly suburban planning, with open spaces that are common in balanced urban settings. Taking all these elements into consideration, the collective visual attributes and environmental clues point to a location consistent with Israel’s landscape and architectural style.

Table 12: Fine-tuning data format. The reasoning process leverages visual information in the images to deduce the correct location, such as climate, vegetation, building, and infrastructure.

Continent 2, 500 km

Country 750 km

Region 200 km

City 25 km

Street 1 km

Distance↓ Score↑

Model

G3 47.3 23.9 6.0 1.6 0.0 4,938 1,451 GeoCLIP 82.3 66.5 48.0 32.2 13.0 1,762 3,402 StreetCLIP 68.2 51.2 29.6 19.0 4.2 3,161 2,640

MiniCPM-V 33.2 27.8 22.4 15.9 2.3 6,624 1,433 LLaVA 61.2 43.2 25.9 16.5 2.6 3,387 2,338 Qwen2-VL 75.0 65.0 48.9 29.9 5.3 2,483 3,237

###### NAVIG

- MiniCPM-V 68.5 51.7 36.5 23.1 3.0 3,149 2,726 - LLaVA 70.4 47.8 26.8 16.7 2.8 2,851 2,592 - Qwen2-VL 84.0 68.3 49.1 28.9 5.5 1,631 3,482

Table 13: Performance on Im2GPS3k.

#### C.2 Haversine Distance

We calculate the Haversine Distance of the models with the following formulas:

∆ = √︄sin2 (︃

)︃ + cos(latcor) cos(latpred) sin2 (︃

)︃

δlat 2

δlon 2

(2) d = 2r · arcsin(∆) (3)

where:

- • r is the Earth’s radius, which we set as 6,371,
- • δlat is the difference in latitude between the true and predicted coordinates,
- • δlon is the difference in longitude between the true and predicted coordinates,
- • latcor and loncor are the correct coordinates,
- • latpred and lonpred are the predicted coordinates.

### D Supplementary Experiments

In this section, we present supplementary experiments, including results from the experiments on Im2GPS3k, and SEARCHER details.

As shown in Table 13, NAVIG outperforms prior models on Im2GPS3k in terms of Average Distance and GeoGuessr Score. However, GeoCLIP achieves better performance at the City and Street level, likely due to its training on coordinates datasets. The ablation results demonstrated in Table 14 are consistent with those in Table 6.

We also analyze the usage of each tool across the datasets and the number of grounding images. This analysis illustrates how frequently NAVIG leverages each tool and image detail to deduce locations.

Model Country City Street NAVIG (MiniCPM-V) 51.7 23.1 3.0

- - w/o training - 1.6 - 1.8 - 0.1
- - w/o REASONER - 10.6 - 3.5 - 0.2
- - w/o SEARCHER - 0.3 - 0.2 - 0.0
- - MiniCPM-V - 23.9 - 7.2 - 0.7

###### NAVIG (LLaVA) 47.8 16.7 2.8

- - w/o training - 15.3 - 4.7 - 0.8
- - w/o REASONER - 8.1 - 1.3 - 0.1
- - w/o SEARCHER + 0.1 - 0.2 - 0.2
- - LLaVA - 4.5 - 0.2 - 0.1

NAVIG (Qwen2-VL) 68.3 28.9 5.5

- - w/o training - 4.3 - 1.2 - 0.3
- - w/o REASONER - 2.9 + 0.5 - 0.1
- - w/o SEARCHER + 0.1 - 0.0 - 0.2
- - Qwen2-VL - 3.3 + 1.0 - 0.2 Table 14: Ablation results on Im2GPS3k.

As shown in Table 15 and Table 16, houses are the most frequently identified items in the testing dataset, as images often contain multiple houses. In contrast, signs, though less common, play a critical role by generating queries for OSM. The distribution of items directly influences the frequency of tool usage for knowledge retrieval.

road sign

building sign

Dataset house

GWS5k 3,451 20 104 Im2GPS3k 465 52 24

- Table 15: The frequency of how each item is grounded.

Dataset N RAG MAP VLM GWS5k 5,000 128 70 1,978 Im2GPS3k 2,997 213 21 493

- Table 16: The usage of each tool in each dataset, where N denotes the size of the dataset.

