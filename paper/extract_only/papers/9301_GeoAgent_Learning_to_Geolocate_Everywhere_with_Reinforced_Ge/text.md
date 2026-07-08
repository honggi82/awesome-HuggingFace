## arXiv:2602.12617v1[cs.AI]13Feb2026

# GeoAgent: Learning to Geolocate Everywhere with Reinforced Geographic Characteristics

Modi Jin1, Yiming Zhang1, Boyuan Sun1, Dingwen Zhang2, MingMing Cheng1, Qibin Hou†1 1VCIP, School of Computer Science, Nankai University 2School of Automation, Northwestern Polytechnical University †Corresponding author.

Abstract: This paper presents GeoAgent, a model capable of reasoning closely with humans and deriving fine-grained address conclusions. Previous RL-based methods have achieved breakthroughs in performance and interpretability but still remain concerns because of their reliance on AI-generated chain-of-thought (CoT) data and training strategies, which conflict with geographic characteristics. To address these issues, we first introduce GeoSeek, a new geolocation dataset comprising CoT data annotated by geographic experts and professional players. We further thoroughly explore the inherent characteristics of geographic tasks and propose a geo-similarity reward and a consistency reward assessed by a consistency agent to assist training. This encourages the model to converge towards correct answers from a geographic perspective while ensuring the integrity and consistency of its reasoning process. Experimental results show that GeoAgent outperforms existing methods and a series of general VLLMs across multiple grains, while generating reasoning that closely aligns with humans.

Project Page: https://ghost233lism.github.io/GeoAgent-page Code: https://github.com/HVision-NKU/GeoAgent

HVision@Nankai

### 1 Introduction

Geolocation [33, 83, 89] is an important computer vision task that aims to infer the geographical location of the image solely from its visual content [42, 56, 23, 51, 85]. Benefiting from its competitive nature, it attracts large player communities, such as GeoGuesser [2] and TuXun [4].

Early approaches [49] attempted to address geolocation by considering it as classification [44, 6, 5, 72, 66] or retrieval [33, 90, 30, 40, 41]. More recently, benefiting from the advances of large language models (LLMs) , vision large language models (VLLMs) [102, 52, 61, 104, 75] capable of integrating visual signals have demonstrated superior abilities in scene perception and understanding [46, 101, 63] compared to traditional approaches. Some approaches thus employ VLLMs [77] to tackle the geolocation task [47, 41, 22, 31, 74], aiming to achieve better performance in open environments. Particularly, with the emergence of reinforcement learning–based methods [50, 59, 28, 71], some approaches [84, 99] incorporate chain-of-thought (CoT) data [88, 43, 15, 100, 86] and reasoning processes [26, 16, 48] into geolocation. By interpreting visual cues, these methods construct rational and trustworthy logical chains for localization [48, 92], and have achieved breakthroughs in both performance and interpretability.

Table 1 Comparison of Geolocation Datasets. CoT: chainof-thought data present. * denotes AI-generated core reasoning process. Location: natural-language place descriptions provided. Global: global geolocation dataset. Sampling: data sampling algorithm to eliminate geographic bias.

###### Dataset CoT Location Global Sampling

MP16 [45] ✘ ✘ ✓ ✘ OSV-5M [10] ✘ ✓ ✓ ✘ GeoGlobe [31] ✘ ✓ ✓ ✓ MP16-Pro [40] ✘ ✓ ✓ ✘ SF-IAL [93] ✘ ✓ ✘ ✓ Georanking [41] ✘ ✓ ✓ ✓ MG-GEO [22] ✓∗ ✓ ✓ ✘ MP16-Reason [48] ✓∗ ✓ ✓ ✘ GeoComp [74] ✘ ✘ ✓ ✘ GRE30k [84] ✓∗ ✘ ✓ ✘ GeoSeek(Ours) ✓ ✓ ✓ ✓

However, existing methods mostly rely on AI-generated CoTs to train the reasoning processes of VLLMs [84, 22], which may not fully align with genuine human reasoning and could potentially amplify the inherent biases of VLLMs. A typical case is that traditional geolocation datasets usually provide only GPS coordinates or lack sufficiently finegrained annotations [33, 45, 10]. These annotations differ substantially from the natural language descriptions humans

[Figure 1]

[Figure 2]

Val-bench

Proportion

[Figure 3]

[Figure 4]

Number

Locatability Extremely Low Low Medium High

30 938 1585

442

Number 1759 1416 881

Manmade

Traffic Buildings Sign Language

[Figure 5]

456

Natural Number

sample weight

- Figure 1 GeoSeek Dataset. We train GeoAgent with GeoSeek, a geolocation dataset with bias-reducing sampling and a val-bench annotated with locatability and geographic elements. Remarkably, a single image may contain multiple geographic elements.

typically use to express geographic locations. Moreover, inferring precise locations from limited visual cues requires human-like reasoning, whereas CoTs generated purely by AI may deviate from authentic logical processes. Prior studies [97, 91, 17] have also noted that RL-based VLLMs often learn superficial formatting patterns rather than true reasoning capabilities.

while semantic similarity measures the degree of similarity between the prediction and the ground truth from a textual perspective step by step. These components encourage the model to converge toward correct answers both physically and semantically.

Considering the issue of maintaining the integrity and consistency of CoT, we introduce the consistency agent as a means of enhancing the learning process of our GeoAgent. More precisely, the consistency agent attempts to derive answers from GeoAgent’s CoT without task-specific prior knowledge, thereby incentivizing GeoAgent to generate higher-quality reasoning processes that better support its conclusions. To sum up, the contributions of our research can be concluded as follows.

To overcome these limitations, we first construct a new geolocation dataset, named GeoSeek. We collaborate with a large number of geography experts and experienced geolocation game players to annotate the data with three levels of human-understandable reasoning granularity, including country, city, and precise location. These human reasoning processes are then standardized into a unified CoT format using GPT-4o [60]. This dataset enables our approach to achieve more fine-grained and interpretable geolocation performance while aligning the reasoning behavior of VLLMs more closely with human cognition.

- • We propose a novel geolocation dataset, GeoSeek, that features CoT data labeled by geographic experts and geolocation game players, alongside a benchmark obtained through a more rational sampling strategy.
- • We propose a new RL-based training paradigm incorporating a geo-similarity reward to judge predictions and a consistency reward computed by a dedicated consistency agent to ensure the integrity and consistency of CoT.
- • We propose GeoAgent, a method that not only enhances performance on the geolocation but also achieves finer granularity in geographic location output while enhancing CoT quality.

For the training strategy, most RL-based approaches incorporate reward functions that evaluate positioning accuracy based solely on whether the texts are equal [84, 48]. However, this does not align with the characteristics of the geographic task because different natural language descriptions may refer to the same geographic location (e.g., Parvis NotreDame, 4 Place Jean-Paul-II and Notre-Dame de Paris). This reward is unreasonable because it overlooks the model’s efforts to converge on the correct answer. To solve the issue of non-unique mappings between natural language and geographic locations, we introduce geo-similarity. Specifically, geo-similarity comprises two components: spatial similarity and semantic similarity. Spatial similarity is a function related to the distance between actual and predicted locations,

Step I : Data Construction

Step II: SFT Fine-tuning

Step III: GRPO Fine-tuning

Geo-Similarity

Cold Start

[Figure 6]

GeoSeek-CoT Expert/Player

[Figure 7]

[Figure 8]

GPS Coordinate

GT

Country Identification Clues: [buildings, signs, Siberian climate] Reasoning: Russian-style builds and Siberian setting, plus Cyrillic script place it in Russia. Conclusion: Russia

Spatial Similarity

40.738° -78.125°

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

Pred Pred

40.738° -78.125°

[Figure 17]

v

Seek For Clues

[Figure 18]

[Figure 19]

Country Region Precise

Reasoning

Semantic

[Figure 20]

[Figure 21]

[Figure 22]

Guide

Similarity

GT

Regional Guess Clues: [Water body , bare rock opposite] Reasoning: North water plus opposite rocks match Baikal’s relief, pointing to Oblast. Conclusion : Oblast

Country Region Precise

Russian-style buildings place us in Russia, yet the open water to the north and the band of exposed rock on the far shore are Lake Baikal scenery; the spot therefore be Olkhon Island. Field-checking quickly zeroes in on the island’s only real settlement, the village of Khuzhir.

Address

Geo-Similarity Reward

Data Base Model

[Figure 23]

GeoAgent

[Figure 24]

[Figure 25]

[Figure 26]

###### Precise Localization:

[Figure 27]

Clues: [Olkhon-type terrain, road angle] Reasoning: Road angle match Khuzhir on Olkhon, Baikal proximity seals it. Conclusion: Khuzhir, Olkhon Island

[Figure 28]

[Figure 29]

Standardize

GPT-4o

[Figure 30]

Country Identification Regional Guess

[Figure 31]

Answer

|[Figure 32]<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

Clues Reason

[Figure 33]

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Precise Localization Final Answer

[Figure 38]

Final Answer: Russia , Oblast , Khuzhir, Olkhon Island

Level-Judge

[Figure 39]

[Figure 40]

[Figure 41]

[Figure 42]

[Figure 43]

Reply Answer

Consistency Reward

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

SFT-Training

[Figure 48]

GeoSeek-Loc

[Figure 49]

highway territorial population

[Figure 50]

[Figure 51]

ConsistencyAgent

GeoAgent-SFT

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

- Figure 2 Data construction and training pipeline of GeoAgent. GeoSeek-CoT contains 10k high-quality reasoning processes labeled by geography experts and geolocation game players. GeoSeek-Loc includes 20k images for the cold start of GeoAgent-SFT. During the GRPO-based training, based on GeoAgent-SFT, we design the geo-similarity reward to encourage the model to converge towards correct answers both physically and semantically. Also, the consistency reward is introduced to keep the integrity and consistency of CoT.

### 2 Related Work

#### 2.1 Image Geolocation

Image geolocation is defined as the process of inferring geographic locations by observing geographical information in images. Many traditional approaches define this task as either classification [89, 83, 42] or retrieval [33, 9, 90, 54, 53]. Classification-based methods divide the Earth into a grid of cells, assigning images to corresponding labels [44, 6, 5, 72, 66] while retrieval-based methods [103, 98, 18, 12, 64, 35] use images as queries to retrieve results from the database [37, 32, 13, 27, 24]. Recently, some studies have started to enhance this paradigm by adopting CLIP [93, 29, 14, 30] or RAG techniques [40, 41]. However, these approaches conflict with the natural reasoning processes of humans. To solve this problem, recent methods use VLLMs to achieve breakthroughs in both task performance and interpretability [74, 31, 94]. Some other methods also start to prompt VLLMs to output not only geographic locations but also the reasoning process [47, 22]. More recent methods train VLLMs with the GRPO strategy [73, 84, 48]. This RL-based training paradigm has been applied to enhance performance and aligning with human thinking. However, the challenges also exist in AI-generated CoT data and unreasonable training strategies. Based on this, we design a novel training strategy from geographic characteristics, encouraging the model to converge toward the correct answer both spatially and semantically while keeping the integrity

and consistency of CoT.

#### 2.2 Geolocation Dataset

Existing relevant datasets [80, 87, 82, 81], like IMG2GPS [33], suffer from excessive regional bias and low locatability. Although datasets such as MP16 [45, 40], OSV5M [10], GeoGlobe [31], GeoComp [74], and Georanking [41] attempt to alleviate the shortage of high-quality data, the lack of reasoning process makes them incapable of supporting RL-based approaches. More recently, MG-GEO [22], MP16-Reason [48], and GRE30k [84] have made contributions by introducing AI-annotated CoT data. Although they offer performance gains, the absence of fine-grained location annotations along with the biases inherited from VLLMs continues to impede their deployment in open-world settings. To conquer these issues, as shown in Tab. 1, we introduce GeoSeek, a novel geolocation dataset, which contains CoT annotations from geographic experts and professional geolocation game players alongside finer-grained addresses. We also employ a rational sampling methodology to mitigate regional data distribution biases.

### 3 GeoSeek Dataset

Previous datasets [33, 80] suffer from coarse granularity, AI-generated CoT data, and suboptimal sampling strategies. Most existing datasets only provide city-level location an-

notations [10, 45], limiting the ability of models to learn fine-grained geographic prediction. Furthermore, if GRPObased methods rely solely on AI-generated CoT data, they may inherit biases present in the base models. Additionally, the uniform area-based sampling used in these datasets overlooks the fact that the distribution of street-view images is more strongly correlated with multiple geographic characteristics. To overcome the limitations , we introduce a new dataset named GeoSeek, aimed at enhancing reasoning quality, annotation granularity, and sampling balance for geolocation tasks. It is a combination of human-labeled CoT, fine-grained locations, and a stratified sampling strategy considering population, land area and road mileage.

Consistency Agent

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

Inconsistent Consistent

[Figure 70]

[Figure 71]

[Figure 72]

###### RegionalGuess:

RegionalGuess: Clues: [Green Signs Hanging on Street Lamp Poles, Green Belts on Both Sides of the Road], Reasoning: Green Signs and Green Belt Styles are in line with the

Clues: [Green Signs Hanging on Street Lamp Poles, Green Belts on

Both Sides of the Road]

[Figure 73]

Reasoning: Pale-green directional signs and contiguous trimmed green belts are typical of

|Anhui|
|---|

|South of the Yangtze River|
|---|

[Figure 74]

[Figure 75]

[Figure 76]

Conclusion: Uncertainty: Low

Conclusion: Uncertainty: Low,

|Anhui Province|
|---|

Anhui Province Hefei City

Anhui Hefei

|Anhui Province|
|---|

Shushan District

Shushan

PreciseLocalization:

PreciseLocalization:

|[Figure 77]<br><br>| |
|---|
<br><br>|
|---|

Clues: [Vegetation and streetlight design on both sides of the road,

Clues: [Vegetation and streetlight design on both sides of the road, Mist in the sky] Reasoning: Plane-tree canopies and evergreen shrub hedges, together with mist, match the landscaping of

Mist in the sky]

Reasoning: Mist and all vegetation characteristics conform to the

|Yangtze River Delta region|
|---|

[Figure 78]

|Hefei, Shushan|
|---|

[Figure 79]

Conclusion: Uncertainty: Low

|Hefei City, Shushan|
|---|

Conclusion: Uncertainty: Low

[Figure 80]

|Hefei City, Shushan|
|---|

Semantic

#### 3.1 GeoSeek-CoT

Similarity

FinalAnswer: China ; Anhui Province ; Hefei City, Shushan District

FinalAnswer: China ; Anhui Province ; Hefei City, Shushan District

To obtain high-quality CoT data, we establish a volunteer platform and invite a great number of geographic experts and professional geolocation game players to provide their reasoning processes. Additionally, we obtain verified and publicly available reasoning process data from the GeoGuessr [2] and TuXun [4] communities. This results in 10k CoT data constructed jointly by human experts and experienced players and each entry consists of street-view images, GPS coordinates and three-level location labels with reasoning process. After annotation, we apply GPT-4o [36] for linguistic normalization and structural formatting, forming a unified CoT template. Additional details about GeoSeekCoT and the annotation process are provided in the supplementary materials.

[Figure 81]

Spatial Similarity

Figure 3 Inconsistent CoT and different descriptions of the same location. Left: incomplete and inconsistent CoT, Right: consistent CoT after training with the consistency agent. Meanwhile, different final answers probably refer to the same location (e.g., Hefei and Hefei City). Therefore, Geo-Similarity is introduced to solve this problem.

mark, which emphasizes the evaluation model’s capability in street-view localization. Each sample is annotated with a locatability score automatically assessed by GPT-4o [36], ranging from 0 to 10, where higher scores indicate easier localization. In addition, we categorize scenes based on their geographic elements such as manmade structures, natural landscapes, and other visual cues. Detailed statistics of difficulty and scene categories are presented in Fig. 1.

#### 3.2 GeoSeek-Loc

GeoSeek-Loc is designed for RL-based finetuning, making it crucial to develop a well-designed sampling strategy that eliminates geographic bias [57, 76]. Most existing datasets perform uniform or area-based sampling, which neglects the impact of geographic characteristics such as population [8] and road mileage [39], especially given the widespread focus on street-view geolocation within the player community. In contrast, we propose a multi-level hierarchical sampling strategy. First, the strategy calculates sampling weights for each country based on population, land area, and highway mileage. Then, each country is divided into equally sized grid cells. Each cell is assigned a logarithmically proportional sampling weight to its population to reduce the excessive sampling concentration. The final GeoSeek-Loc contains 20,000 high-resolution street-view samples with global distribution. For the specific sampling formula, please refer to the supplementary materials.

### 4 Methodology

Our GeoAgent pipeline is illustrated in Fig. 2. It contains a two-stage training process. In the first stage, a cold start is employed using the high-quality reasoning dataset GeoSeekCoT. During this stage, we fine-tune Qwen2.5-VL-7B [11] for 2 epochs. The second stage involves GRPO [19] reinforcement learning for 1 epoch using the GeoSeek-Loc dataset. This stage enhances the model’s reasoning process through the geo-similarity reward function, which consists of spatial and semantic similarity, facilitating the convergence of the model toward correct spatial and semantic answers. In addition, we propose the consistency reward assessed by a consistency agent , which is designed to direct GeoAgent towards the generation of reasoning processes of higher quality and supporting its conclusions.

#### 3.3 GeoSeek-Val

#### 4.1 Overall Rewards

To evaluate model performance, we adopt the same stratified sampling strategy and extract an additional 3k samples from the OSV5M [10] dataset to form the GeoSeek-Val bench-

In GRPO-based approaches, the policy is refined by comparing candidates that are drawn as a group. For every query,

[Figure 82]

[Figure 83]

[Figure 84]

Consistent

Consistent

Deviate

Deviate

Inconsistent

Inconsistent

Consistency Reward

w/o Con. w/o Con.

Semantic Similarity Spatial Similarity

(b) Correlation between Spatial Similarity Reward and (c) Reward function value over training steps Semantic Similarity Reward

(a) Correlation between Spatial Similarity Reward

and Directly-Judge Reward

Figure 4 Discussion of reward functions. The two scatter plots respectively reveal the reasons for the unreasonable directly-judge reward and the positive effect of semantic similarity reward. We also demonstrate the curve of reward value changes over training steps, reflecting the importance of applying consistency reward to enable the model to establish a complete reasoning framework.

the algorithm first samples a batch of G candidate replies and scores them with verifiable rewards {Ri}Gi=1, thereby enabling intra-group comparison. The normalized advantage of reply i is then computed as:

inverse geocoding. Then, the spatial distance D between the predicted location (λ,ˆ ϕˆ) and the corresponding ground truth location (λ,ϕ) is calculated as below [38]:

∆ϕ 2

∆λ 2

+ cosϕˆcosϕsin2

D = 2r arcsin sin2

, (4)

Ri − mean(Rj) std(Rj)

, (1)

Ai =

where r = 6371km denotes the mean Earth radius and ∆ is the difference. The spatial similarity reward is defined as:

which quantifies how much this reply outperforms or underperforms its peers. During optimization, the same advantage Ai is broadcast to every token of reply i, and the policy network is updated with a PPO-like objective:

D (λ,ˆ φˆ), (λ,φ) τ

, (5)

Rspa = exp −

JG(θ) = E min ri,tAi, clip(ri,t,1 − ε,1 + ε)Ai , (2)

where ri,t is the importance weight between the new and old policies, and the clipping threshold ε keeps the update conservative for stability.

Previous works [48, 84, 96] design reward functions that directly-judge whether texts are equal to enhance model’s geolocation capabilities. However, this overlooks the characteristics of the geographic task because multiple locations can correspond to the same geographic position. Based on this insight, we develop the geo-similarity reward that measures the similarity between the model’s response and the correct answer. Specifically, it contains two components: spatial similarity Rspa and semantic similarity Rsem. Additionally, we also design the consistency reward Rcon to maintain the integrity and consistency of CoT. Finally, the reward function can be expressed as:

##### R = a1Rspa + a2Rsem + a3Rcon, (3)

where a1,a2,a3 are weights controlling the importance of three components, which are 1.5, 1.0 and 0.5, respectively.

#### 4.2 Geo-Similarity Reward

Spatial similarity. To overcome the issue of non-unique mapping, we obtain the corresponding latitude λˆ and longitude ϕˆ for the model’s predictions through OpenCage [3]

where τ is a temperature parameter controlling the reward sharpness. Rspa exhibits a nonlinear relationship with distance, specially increasing more slowly as the distance to the ground truth decreases. This encourages the model to identify a broad range of predictions before progressively narrowing it down, which aligns with human thinking.

Semantic similarity. To further enhance the model’s ability to generate standardized address names and improve its robustness against ambiguous or alias geographic names, we introduce the semantic similarity reward [58, 21, 69]. We employ the multilingual semantic encoding model [69, 70] to encode each level i of the address into vector representations hpredi and hgti . Then, we compute the cosine similarity between each pair of embeddings:

hpredi · hgti ∥hpredi ∥∥hgti ∥

. (6)

si =

To suppress over low-quality results, we apply a threshold to filter similarity sˆi greater than δ. Additionally, a hierarchical strategy is employed, which ensures that lower-level address components are only rewarded when higher-level predictions are not entirely wrong. Finally, the semantic similarity reward is computed as:

3

Rsem =

αi sˆi,

i=1

i

αi = 1, (7)

Table 2 Zero-shot Geolocation results on IM2GPS3K [33] and GeoSeek-Val. ♢ indicates the model is not publicly accessible. ⋆, ♣ and ♠ represent LoRA [34], QLoRA [20] and full fine-tuning, respectively. Best and second-best results are in bold and underlined.

IM2GPS3K (% @km) GeoSeek-Val (% @km)

Models Dataset Size

City Region Country Continent City Region Country Continent Geoscore 25km 200km 750km 2500km 25km 200km 750km 2500km 0-5k

GeoDecoder [18]♢ MP-16 4M 33.50 45.90 61.00 76.10 - - - - GeoCLIP [14] MP-16 4M 34.47 50.65 69.97 83.82 11.82 29.04 56.13 79.13 3172.3 Translocator [65] ♢ MP-16 4M 31.10 46.70 58.90 80.10 - - - - PIGEOTTO [30] ♢ MP-16 4M 36.70 53.80 72.40 85.30 - - - - G3 [40] ♢ MP-16 4M 40.94 55.56 71.24 84.68 - - - - -

Qwen2.5-VL-7B [11] - - 21.93 29.93 43.08 60.68 3.57 7.95 20.03 44.96 1622.0 Qwen2.5-VL-32B [11] - - 23.36 30.35 43.46 59.70 4.06 7.43 16.10 39.06 1413.5 Gemma3-27B [78] - - 28.53 41.74 57.95 73.64 11.69 25.48 49.52 71.79 2704.0 InternVL3-14B [105] - - 20.10 27.51 40.89 59.01 3.43 7.67 18.69 43.50 1549.1

GeoReasoner⋆ [47] GSV 133K 26.94 36.63 52.27 78.65 13.55 27.86 53.54 77.63 3083.9 GaGA [22] ♢ ♣ MG-Geo 5M 33.00 48.00 67.10 82.10 - - - - GRE-Suite-SFT [84] ♢ ♠ GRE-CoT 20K 29.30 44.78 62.43 78.81 - - - - GRE-Suite [84] ♢ ♠ GRE 30K 35.30 51.70 69.30 85.67 - - - - GLOBE [48] ♢ ♠ MP-16 Reason 33K 40.18 56.19 71.45 82.38 10.75 21.20 39.17 61.44 2412.5 GeoAgent-SFT (Ours)⋆ GeoSeek-CoT 10K 32.77 45.42 62.65 81.33 10.36 23.84 47.12 64.98 2968.9 GeoAgent (Ours) ⋆ GeoSeek 30K 40.75 58.57 76.21 89.90 15.69 33.39 60.37 81.72 3314.1

L

- g

- h

- h

- i

m

o

where αi represents the weight assigned to each level i. The semantic similarity reward effectively compensates for the limitations of spatial similarity in semantic understanding. It enables the model to achieve correct reward signals even in the presence of aliases, abbreviations, or translation variations. This encourages the generation of more standardized and semantically coherent geographic descriptions.

e

c

d

e

i

4158

u

a

r

m

- t

- u

t

l

[Figure 85]

u

- a

- b

c

3392

i

r

g

i

3481

l

a

3515

i

t

3260

l

o w

y

2716

3401

2649

e

3261

- x t r e m e

l

- y l o w

l

a

3506

2008

r

- m

- o

u

n

t

a

i

n

s

/

r

i

v

e

- r

- s

- p

- t

- u

1928

a

N

3171

s

Discussion. In Fig. 4, we visualize the correlation between the directly-judge reward proposed by previous methods and our proposed semantic similarity reward and spatial similarity reward under the same dataset. Spatial similarity reward is a function directly related to distance difference, reflecting the model’s geographic positioning capability. Therefore, the reward trend directly impacting the model’s geolocation capabilities should align with it. However, Fig. 4 (a) clearly demonstrates that directly-judge reward exhibits severe inconsistencies with spatial similarity due to the non-unique mapping between descriptions and locations. Therefore, it is unreasonable to use it to enhance geolocation capabilities, and the results in the Tab. 3 also prove this. In contrast, as shown in Fig. 4 (b), the trends of semantic similarity rewards and spatial similarity rewards are highly consistent. This further explains the positive effect of semantic similarity rewards on the model.

t

n

a

3409

3327

l

c

i

f

f

3570

a

r

t

3177

3564

w e

3331

a

3798

t

h

3459 3588

n

e

g

- d

- e

i

r

s

a

m

3669 3901

n

e

b

a

g

u

i

a

M

l

d

u

i

g

n

g

n

s

a

l

GeoAgent (Ours) GeoCLIP GeoReasoner

InternVL3-8B Qwen2.5VL-7B

Figure 5 GeoScore on GeoSeek-Val. We compare different models in multiple locatabilities and geographic elements.

However, we also need to establish the relationship between images and geographic clues. This involves progressively narrowing down potential geographic areas before establishing a connection to a specific location. Therefore, we propose the consistency reward, assessed by a consistency agent that only obtains reasoning processes output by the GeoAgent but not its conclusions, as shown in Fig. 2. The consistency reward Rcon is defined as:

#### 4.3 Consistency Reward

As shown in Fig. 13, one possible reason for inconsistent reasoning processes is that the geo-similarity reward represents the relationship between images and geographic locations.

wi = 1, (8)

1 y ˆi = yi · wi · pi,

Rcon =

i

i

Table 3 Ablation study results reported on GeoSeek-Val. Best and second-best results are in bold and underlined. Models SFT COT

GRPO Metrics

Spa. Reward Sem. Reward Con. Reward City Region Country

Qwen2.5-VL-7B [11] 1.39 3.36 11.13 GeoAgent-SFT ✓ 10.36 23.84 47.12 GeoAgent w/o Spa & Con ✓ ✓ ✓ 11.99 26.56 55.55 GeoAgent w/o Sem & Con ✓ ✓ ✓ 14.46 31.51 59.86 GeoAgent w/o Con ✓ ✓ ✓ ✓ 14.69 31.39 60.20 GeoAgent w/o Spa & Sem ✓ ✓ ✓ 9.08 20.03 40.43 GeoAgent w/o SFT ✓ ✓ ✓ ✓ 13.39 23.94 58.23 GeoAgent ✓ ✓ ✓ ✓ ✓ 15.69 33.39 60.37

Table 4 Discussion of geo-similarity and directly-judge on GeoSeek-Val. The judge means replacing the geo-similarity reward with directly-judge reward.

Metrics City Region Country

Models

Qwen2.5-VL-7B [11] 1.39 3.36 11.13 GeoAgent-SFT 10.36 23.84 47.12 GeoAgent-SFT + Judge 12.35 26.87 50.81 GeoAgent-SFT + Judge + Con 11.04 26.13 51.76 GeoAgent w/o Con 14.69 31.39 60.20 GeoAgent 15.36 32.53 60.37

where 1[·] denotes the indicator function that evaluates to 1 if the model’s predicted conclusion yˆi matches the ground-truth conclusion yi, and 0 otherwise. The weight wi corresponds to the importance of the i-th geographic granularity level (i.e., country, region, or precise location).

Additionally, pi is a penalty term introduced to discourage the model from generating overly simplistic reasoning processes that could evade detection by the consistency agent (such as outputting only a final conclusion).

This factor is a value proportional to the i-th level length ℓi of the reasoning process and can be expressed as:

ℓi − min(ℓ) max(ℓ) − min(ℓ)

1 1 + exp −λ(ℓˆ − µ)

,ℓˆ=

, (9)

pi =

where λ and µ are the parameters of the control curve. The consistency reward models the relationships from images to geographic clues, from clues to layer-by-layer analyses, and finally to geographic locations. This ensures the integrity and consistency of the CoT at each level.

Discussion. As shown in Fig. 4, when training progresses, the consistency reward converges first. This enables the model to gradually establish a complete and consistent reasoning process. Compared to the model without consistency reward, both spatial and semantic similarity rewards remain

lower than the existing model before the consistency reward converges. However, after convergence, these rewards begin to increase and surpass the model without consistency reward. This explains how consistency rewards positively impact the model’s geolocation capabilities.

### 5 Experiments

#### 5.1 Experimental details

Implementation details. GeoAgent is fine-tuned upon Qwen2.5-VL-7B [11] using LoRA [34] with rank of 64 and alpha of 128, while the consistency agent utilizes the GPTQ-INT4 [25] quantized version of Qwen3-32B [95].

Evaluation settings. An evaluation of GeoAgent is conducted using a public benchmark IMG2GPS3K [33] and GeoSeek-Val. In accordance with previous studies [48], we employ the City (25km), Region (200km), Country (750km) and Continent (2500km) accuracy on IM2GPS3K [33]. Furthermore, on GeoSeek-Val, we report the GeoScore, a widely utilized metric in the geolocation community. The scale ranges from 0 to 5000. The formula is as follows:

10d dmax

GeoScore = 5000 × exp −

, (10)

where d represents the distance between the predicted location and the actual location (in kilometers), while dmax denotes the scale. For global geolocation, dmax is typically set to 18,050 km. We utilize OpenCage [3] for geocoding and inverse geocoding.

#### 5.2 Main Results

Performance on Public Benchmarks. We perform a comparative analysis of GeoAgent with multiple approaches in Tab. 2, including traditional methods, general VLLMs, and geolocation-specific VLLMs. With only using LoRA [34] fine-tuning (1.91% Paras), GeoAgent achieves a considerable performance improvement on the IM2GPS3K [103]

Santiago, Chile

GeoAgent-SFT: Valparaiso, Chile

GeoAgent: Santiago, Chile

[Figure 86]

| |
|---|

[Figure 87]

CountryIdentification: The architectural style with its

CountryIdentification: The architectural style and the use of Spanish language strongly indicate that this location is in a

distinct red and yellow colors is reminiscent of Latin American colonial architecture, which is a strong clue pointing towards Chile. The Latin script on the building further supports this identification.

Spanish-speaking country. The Roman numerals suggest a

| |
|---|

historical context, which aligns with South American countries. Conclusion: Chile.

| |
|---|

v

v

RegionalGuess: The red and yellow colonial-style architecture is a hallmark of Chilean cities, particularly Santiago, which is known for its vibrant and ornate buildings. The license plates

| |
|---|

RegionalGuess: The architectural style and the historical reference point towards Valparaíso, a city in Chile known for its colonial-era buildings and cultural heritage., Conclusion: Valparaíso

[Figure 88]

with a single-digit prefix are a strong indicator of Santiago.

[Figure 89]

[Figure 90]

The subtropical vegetation and European-style urban planning are characteristic of Santiago's urban environment.

[Figure 91]

Qwen: Lima, Peru

Kimi: Valparaiso, Chile

Gemma3: Montevideo, Uruguay

[Figure 92]

[Figure 93]

[Figure 94]

GPT-5: Salta, Argentina

The inscription 'MDCCCLV' (1855 in Roman numerals) suggests a Spanish-speaking country, The style of the street light is

I cannot definitively determine the exact location of the building in the photograph.

The combination of architectural elements and the historical feel strongly suggests a region with a

This photograph depicts the Cathedral Basilica of

Salta, the cathedral of Salta, Argentina. The

church features a typical facade design blending Neoclassical and Baroque styles: symmetrical structure, massive arched porticoes, doorways framed by twin columns, and an ornate triangular pediment at the top. The exterior walls alternate between pinkish-orange and beige hues, a signature color scheme for Salta‘s churches. Above the lintel

common in Latin American countries. The

H owev er, ju dg i ng by i ts

rich colonial history, such as parts

architectural details and color scheme are particularly suggestive of Argentina or Uruguay. The street sign visible in the background is consistent with Uruguayan traffic signs. After researching the architectural style and inscription, the building is identified as the Catedral Basílica

architectural style and the Roman numerals “MDCCCLV”

of South America, particularly Peru or Argentina. The specific year '1755' and the architectural style point to a historically significant building in a city known for its colonial heritage. Lima, Peru, is renowned for such structures

(1855), it is likely a historic structure somewhere in Latin America—perhaps a significant landmark in Valparaíso, Chilepossibly associated with

is inscribed “MDCCCLV”—the Roman numerals for

1855—marking the cathedral’s completion years. Construction began in the mid-19th century (1858).

de San José de Montevideo.

municipal or religious functions.

and matches these characteristics.

Figure 6 Reasoning comparison of six different models (GPT-5 [62], Gemma3 [78], Kimi [79], Qwen2.5-VL-32B [11], GeoAgent and GeoAgent with only SFT). Green: correct reasoning, Yellow: reasoning with certain issues, Red: reasoning that deviates significantly. Notably, brown highlights instances where the reasoning process is incomplete or inconsistent with the result.

#### 5.3 Ablation Analysis

(e.g., 72.40 → 76.21 on country accuracy ), surpassing a series of full fine-tuning methods. It is also worth noting that GeoAgent demonstrates significantly greater performance improvements at the macro level compared to the fine level. This is because the geographical similarity reward gradually decelerates as distance decreases, prompting the model to first determine a coarse-grained location. In fact, in geolocation competition, country-level precision positioning is sufficient to outperform the most exceptional players.

Ablation study of components. In Tab. 3, we discuss the effectiveness of the reward function we propose. It can be seen that Rspa, Rsem, and Rcon each contribute to performance improvements. Among these, Rspa delivers a more significant performance improvement than Rsem. This is primarily because Rspa is a distance-dependent function that serves as a direct reward signal, whereas Rsem focuses on semantic robustness. When applied independently, Rcon causes a slight decrease in performance. This can be attributed to it imposes constraints on consistency rather than directly addressing ge-

Performance on GeoSeek-Val. We compare the accuracy of GeoAgent and other methods on GeoSeek-Val in Tab. 2. Also, we report their GeoScore on splits categorized by locatability and geographic elements in Fig. 5. The results demonstrate that GeoAgent significantly exceeds other models on the street-view geolocation benchmark (e.g., 56.13 → 60.37 on country accuracy). In addition, the results shown in Fig. 5 demonstrate that GeoAgent surpasses other models across all locatability and geographic element splits. This indicates that GeoAgent possesses a superior understanding of various geographic elements. Furthermore, the phenomenon where GeoAgent outperforms other models more significantly on splits with higher locatability suggests that GeoAgent’s thinking patterns are closer to those of humans.

olocation capabilities. However, when combined with Rspa and Rsem, it brings noticeable improvements, especially at the regional and city levels, where inconsistencies are more frequent than at the country level.

Cold Start. Using GeoSeek-CoT for SFT-based cold start enables the model to establish a reasoning framework, which aligns closely with human thinking patterns. The results in Tab. 3 indicate that cold starts can significantly enhance the base model’s performance (e.g., 11.13 → 47.12 on country accuracy). At the same time, models lacking cold start capabilities will experience a decline in performance. This demonstrates the importance of establishing a reasoning framework during the cold start phase.

Dataset Quality. As shown in Tab. 2, our GeoAgent with only SFT achieves performance surpassing other methods with only SFT (GeoReasoner [47], GaGA [22] and GREsuite-SFT [84]) despite operating on significantly less data (10K vs 20K, 133K, 5M). This fully demonstrates the high quality of the data annotated by GeoSeek-CoT’s geographic experts and geolocation game players.

Discussion of Geo-Similarity and Directly-Judge. To demonstrate that the geo-similarity reward function better aligns with the characteristics of geographic tasks, we compare it with a reward function that directly-judges whether texts are identical at each level. As shown in Tab. 6, the performance improvement from the directly-judge reward is

significantly smaller than that from the geo-similarity reward (e.g., 50.81 and 60.37 on country accuracy). This is because the directly-judge reward is overly strict, overlooking the model’s efforts when it is not entirely correct but is moving toward the correct answer. Geo-similarity, on the other hand, evaluates the similarity between predictions and answers from both distance and semantic perspectives. It encourages the model’s positive efforts while mitigating the negative impact of cross-language and aliasing issues on stability in geographic tasks. This underscores the necessity of designing training methods tailored to geographic characteristics for geographic tasks.

#### 5.4 Qualitative Comparisons

As shown in Fig. 6, we compare the reasoning processes of GeoAgent, GeoAgent with only SFT, and other general VLLMs to the same image. Compared to other models, GeoAgent exhibits a clearer hierarchical reasoning process, enabling it to better capture the relationship between geographic characteristics and geolocation. Notably, after GRPO training, GeoAgent not only enhances its ability to identify geographic features but also overcomes the shortcomings of incomplete and inconsistent reasoning process.

### 6 Conclusions

This paper proposes GeoAgent, a model capable of thinking like humans and providing a geographic location. We propose GeoSeek, a novel dataset with annotated CoT from human geographic experts and geolocation game players. In training, we employ a two-stage training combining SFT with GRPO fine-tuning. Considering the non-uniqueness of descriptions and location mappings , we introduce the geo-similarity reward. In addition, we introduce the consistency reward to ensure the consistency of CoT. Experimental results demonstrate that GeoAgent achieves outstanding performance on multiple benchmarks.

### 7 Acknowledgments

This work was supported by NSFC (No. 62225604 and No. 62495061), and the Fundamental Research Funds for the Central Universities (Nankai University) under Grant 070-63253220.

We sincerely thank Yue Zhang, H.M., Haowen He, Yuke Jun, and other experts in geography, as well as outstanding geolocation game players, for their valuable guidance, prompt design suggestions, and data support throughout the construction of the GeoSeek dataset.

We also thank Zhixiang Wang, Chilin Chen, Jincheng Shi, Liupeng Zhang, Yuan Gu, Yanghang Shao, Jinhua Zhang, Jiachen Zhu, Gucheng Qiuyue, Qingyang Guo, Jingchen

Yang, Weilong Kong, Xinyuan Li, and Dawei Xu for their outstanding contributions in providing high-quality reasoning process data.

### Appendix

In order to provide a more complete illustration of the details and capabilities of GeoAgent, a comprehensive appendix has been developed. This appendix includes detailed descriptions of the implementation, evaluation benchmarks, our proposed dataset GeoSeek, additional experiments analysis, a variety of success or failure cases and the discussion of our model.

### A More Implementation Details

- A.1 Configuration and Hyper-parameters

During training, we employ the AdamW [55] optimizer with an initial learning rate of 1e-5 and utilize deepseedzero3 [68, 67] for GPU memory optimization. Training and testing are performed on CentOS 7. For training, we utilize 8 NVIDIA A40 GPUs, while all evaluations are conducted on 2 NVIDIA A40 GPUs. The meanings and specific values of the hyperparameters mentioned in this paper and used in training and evaluation are shown in Tab. 5.

- A.2 Prompts of Different Stages

In Fig. 7, we present the prompts used in the SFT and GRPO stages. In Fig. 8, we present the prompts used for locatability scoring and category annotation.

- A.3 Introduction to IM2GPS3K

The IM2GPS3K dataset is a subset of the IM2GPS [33] dataset, comprising 3,000 geo-tagged images commonly used as a benchmark dataset for image geolocation. It is extracted from a collection of 6 million images sourced from Flickr [1]. This dataset is widely employed in numerous image geolocation studies to evaluate model performance across different geographic thresholds.

### B Details of GeoSeek

#### B.1 Sampling Algorithm

In contrast to traditional uniform sampling, we propose a multi-level hierarchical sampling strategy during the construction of GeoSeek-Loc and GeoSeek-Val. First, the strategy calculates sampling weights for each country based on population [8], land area, and highway mileage [39]. Subsequently, each country is divided into equally sized grid cells. Each cell is assigned a logarithmically proportional sampling weight to its population to reduce the excessive sampling concentration. At the country level, the number of

Table 5 The values of each hyperparameter. Symbol Description Value

- a1 weight of Rspa 1.5
- a2 weight of Rsem 1.0
- a3 weight of Rcom 0.5 r Earth’s radius 6371km τ temperature parameter of Rspa 200

- α1 weight of Country level in Rsem 0.1
- α2 weight of Region level in Rsem 0.6
- α3 weight of Precise level in Rsem 0.3

- δ1 Country-level threshold in Rsem 0.7
- δ2 Region-level threshold in Rsem 0.5

- w1 weight of Country level in Rcon 0.1
- w2 weight of Region level in Rcon 0.6
- w3 weight of Precise level in Rcon 0.3 G Group size of GRPO 8

t Temperature of GRPO 0.7 β KL coefficient of GRPO 0.001

Table 6 SFT results of different base models on GeoSeek-Val. Models

Metrics City Region Country

Qwen2.5-VL-7B [11] 1.39 3.36 11.13 InternVL3-8B [105] 3.79 7.35 16.10 Gemma3-12B [78] 10.28 22.37 46.51

GeoAgent-SFT 10.36 23.84 47.12 InternVL3-8B + SFT [105] 6.14 14.94 32.86 Gemma3-12B + SFT [78] 13.07 27.64 57.67 GeoAgent 15.36 32.53 60.37

samples mi assigned to each country i is defined as:

Ri j Rj

Pi j Pj

Ai j Aj

mi = M λ1

+ λ2

+ λ3

, (11)

where M is the total number of samples, Ri, Pi, and Ai denote the total road length, population, and land area of country i, respectively. The coefficients λi(i = 1,2,3) are weight factors, with specific values assigned as λ1 = 0.5, λ2 = 0.2 , and λ3 = 0.3 .

Within each country, the globe is divided into a 360 × 180 longitude–latitude grid (each cell spans 1◦ × 1◦). For each grid cell c, the local sampling probability pc is computed as:

pc ∝ log(1 + Pc), (12)

where Pc represents the predicted population within cell c. This logarithmic adjustment enhances coverage of populated regions without oversampling urban clusters.

#### B.2 Data Sources

GeoSeek consists of three components which are GeoSeekCoT, GeoSeek-Loc, and GeoSeek-Val. GeoSeek-CoT con-

tains 10,000 CoT data points with high-quality human annotations, approximately 6,000 of which originate from our collaborating geographic experts and volunteer geolocation game players. Approximately 4,000 are sourced from publicly available geolocation game guide websites [2, 4, 7].

To ensure regional balance, when constructing GeoSeek-Loc, we combine samples from OSV5M [10] and GeoComp [74] using sampling algorithm proposed in B.1. Specifically, 12,500 samples are drawn from the OSV5M [10] dataset and 7,500 from the GeoComp [74] dataset, both following the aforementioned multi-level hierarchical sampling approach. The final GeoSeek-Loc contains 20,000 highresolution street-view samples with global distribution.

Similarly, our evaluation set GeoSeek-Val also adopts the stratified sampling algorithm, selecting 3,000 data samples from OSV5M [10].

#### B.3 Data Collection and Annotation Process

In developing GeoSeek-CoT, we collaborate with a group of geographic experts and professional geolocation gamer players to build a volunteer platform. Volunteers are tasked with providing three-levels geographic resolution for images (country, region, precise location). Country-level resolution was mandatory, while the reasoning behind regional and precise location classifications is left to the volunteers’ discretion based on the image’s locatability. Subsequently, reasoning processes are processed through GPT-4o [60] to standardize it into the CoT format shown in Fig. 7.

When configuring GeoSeek-Loc, we extract images from the sample set via stratified sampling, integrate the returned results using the OpenCage [3] reverse geocoding service, and obtain a structured three-level address.

For GeoSeek-Val configuration, in addition to acquiring the address in the same format as mentioned above, we also utilize GPT-4o [60] to analyze the locatability and geographic elements of the images, ultimately integrating them into a comprehensive evaluation dataset.

#### B.4 Comparison with AI-annotated Data

- In Fig. 9, we present the reasoning processes provided by volunteers, the cleaned CoT data formatted to standard specifications, and the CoT data directly annotated by AI. We have highlighted certain errors in the AI-annotated CoT data in red. It is evident that the cleaned data annotated by humans significantly outperforms the AI-annotated data in both accuracy and the soundness of the reasoning process.

B.5 More Cases

- In Fig. 10, we present additional examples of GeoSeekCoT, GeoSeek-Loc, and GeoSeek-Val, with each dataset

containing fine-grained geolocation annotations. GeoSeekCoT also includes human-annotated high-quality CoT data, while GeoSeek-Loc contains locatability and geographically annotated clues for localization.

### C Different Base Models

In Tab. 6, we compare the performance of different base models on GeoSeek-Val after SFT cold start. Experimental results demonstrate that different base models exhibit significant performance improvements after undergoing GeoSeekCoT’s SFT-based cold start. This further validates the effectiveness of GeoSeek-CoT’s human-annotated CoT data.

### D Case Study

- D.1 More Typical Cases

As shown in Fig. 11 and Fig. 12, we present additional typical cases of GeoAgent on the IM2GPS3K [33] and GeoSeekVal evaluation datasets. Qualitative results indicate that GeoAgent can capture more geographically relevant features aligned with human cognition, engage in stepwise reasoning, and ultimately provide fine-grained geographic locations.

- D.2 Failure Cases

In Fig. 13, we illustrate several instances where GeoAgent encountered failures. These failures primarily occurred when there were too few locatable geographic features (such as only plants) or when geographic features could potentially appear in multiple locations.

- D.3 Robustness of the Model

We examine GeoAgent’s robustness by masking different geographical cues to observe whether the model could still infer locations. As shown in Fig. 14, we mask utility poles, text, shops, and foreground elements respectively. The model continue to infer locations using remaining geographical cues while also discovering new cues to aid inference, fully demonstrating GeoAgent’s robustness.

### E Discussion

This paper introduces GeoAgent, a model capable of performing hierarchical reasoning based on geographical clues within images to ultimately provide fine-grained geographic locations. The geolocation task itself possesses competitive and recreational attributes, attracting a large number of passionate and skilled players who have formed numerous gaming communities, such as Geoguessr [2] and TuXun [4]. The model proposed in this paper not only offers clear guidance for these players but also holds significant practical application value. Image geolocation plays a vital role in

fields like criminal tracking, emergency response, social media, and cultural dissemination especially when image metadata, particularly GPS data, is unavailable. Simultaneously, geolocation serves as an effective alternative to GNSS, addressing limitations in satellite signal availability.

However, the proposed model still has limitations. For instance, since data is sourced from player communities, it primarily focuses on street-view geolocation and performs poorly in indoor scenarios or outdoor environments with minimal geographical cues. Furthermore, the model’s societal impact warrants consideration, requiring careful use and attention to the potential for geolocation models to infringe on personal privacy. This prevents them from becoming tools that assist criminal activities or violate individual privacy.

###### GRPO Prompt

###### SFT Prompt

Prompt: “You are an expert with extensive experience in the field of geolocation, skilled at accurately determining the geographical location of images by analyzing various clues such as traffic signs, architectural styles, and natural landscapes. At the same time, you also serve as a mentor in constructing thought chains, capable of organizing complex ideas into clear and standardized patterns. With multidisciplinary knowledge in geography, cartography, transportation, architecture, and more, you can identify the characteristics of different countries, regions, and locations. Meanwhile, you possess logical analysis skills and the ability to construct thought chains. Task: Generate a thought chain and final answer based on the user's input image. The thought chain includes: CountryIdentification /RegionGuess/PreciseLocalization. Possible clues include: national clues (examples: traffic sign shapes/colors, language and writing, driving direction, architectural style, vegetation and climate features, etc.) Regional Clues: (Emblems/Enterprises, Topography, Vegetation Types, Regional Traffic Signage, Dialect/Spelling, License Plate Styles, Area Codes/Postal Codes, Infrastructure Characteristics, etc.)

Prompt: "You are an expert with rich experience in the

field of geolocation, skilled at accurately locating the

geographic location of images through various clues in the images, such as traffic signs, architectural styles, natural landscapes, etc. At the same time, you are also a mentor in building the chain of thought, able to organize complex ideas into clear and standardized patterns. You possess knowledge in multiple disciplines such as geography, cartography, transportation, and architecture, and are able to identify the characteristics of different countries, regions, and locations. At the same time, you have the ability to analyze logic and construct a chain of

thought.

Task: Output the thought chain and final answer based on the images input by the user The thought chain includes: Country Identification/Regional Guess/Precise Localization. Possible clues include: National clues: (Example: traffic sign shape/color, language and text, driving direction, architectural style, vegetation and climate characteristics, etc.) Regional clues: (logo/enterprise, topography, vegetation type, regional traffic signs, dialect/spelling, license plate style, area code/postal code, infrastructure features, etc.) Accurate positioning: (road sign text, street name, house number, landmark building, river and lake water system, place attributes such as park/city/commercial district, shop name and storefront, etc.) Output strictly in JSON format:\n {\n 'ChainOfThought':{\n CountryIdentification \": {\" Clues \": [],\" Reasoning \":\" \",\" Conclusion \":\" \",\" Uncertainty \":\" \"},\n RegionalGuess \": {\" Clues \": [],\" Reasoning \":\" \",\" Conclusion \":\" \",\" Uncertainty \":\" \"},\n PreciseLocalization \": {\"

Precise Positioning: (Road signs, street names, house numbers, landmark

buildings, river and lake systems, venue attributes such as parks/downtown areas/commercial districts, shop names and storefronts, etc.) Do not output objects that do not exist in the image. The reasoning process and the answer are strictly enclosed within <think> </think> and <answer> </answer> tags, respectively. For example, <think> reasoning process </think> , <answer> FinalAnswer </answer>\n The reasoning process within <think></think> strictly follows JSON format:\n{\ChainOfThought": {\CountryIdentification": {\n "Clues":[],\n "Reasoning": "",\n "Conclusion": "",\n "Uncertainty": ""\n},\n "RegionGuess": {\n "Clues": [],\n "Reasoning": "",\n "Conclusion": "",\n "Uncertainty": ""\n},\n "PreciseLocalization": {\n

"Clues": [],\n "Reasoning": "",\n "Conclusion": "",\n "Uncertainty":

""\n}\n },\n "FinalAnswer": "Country; Region; Specific Location"\n } The content within <answer> is strictly text, formatted as: Inferred country name; First-level or second-level administrative divisions; Further positioning without the conclusion of the previous stage.

Clues \": [],\" Reasoning \":\" \",\" Conclusion \":\"

\",\" Uncertainty \":\" \"}\n },\n FinalAnswer: \"Country; Region; Specific Location\"\n }"

- Example1:<answer> China; Hong Kong; Kowloon, Jordan Valley </answer>
- Example2:<answer> United States; MN, Saint Louis County; Ash Lake </answer>”

Figure 7 Prompts in SFT and GRPO.

###### Locatability Prompt

###### Classify Prompt

Prompt: “You are an expert in visual geography and scene understanding. You will be shown an image from a global geolocation dataset. Your task is to analyze the image and identify which geographical or contextual elements are visible. Please carefully look at the image and mark whether each of the following categories is present ("yes" or "no"). If the element is partially visible or ambiguous, mark "maybe". Categories:

Prompt: “Please act as a geographic information analysis expert and conduct a locatability assessment of the following images. Evaluation Criteria (1-10 Point Scale): 10 Points - Extremely High Locatability: Contains clear landmark buildings or iconic landscapes. Has legible road signs, house numbers, or store signs Includes GPS coordinates or map information. Features unique geographical characteristics (e.g., famous mountains, rivers, coastlines). 8-9 Points - High Locatability: Contains partial landmarks or well-known buildings. Has blurred but recognizable road signs or store names. Includes city skylines or distinctive building complexes Features unique topographical characteristics.

- 1. manmade — artificial structures or human-made objects (e.g., bridges, roads, fences)
- 2. traffic — vehicles, traffic lights, road signs, crosswalks, or road markings
- 3. buildings — any residential, commercial, or religious buildings
- 4. sign — textual or symbolic signage such as shop names, billboards, or direction signs
- 5. language — readable text that indicates a specific language (e.g., Chinese, English, Arabic)
- 6. natural — natural landscapes such as forests, grasslands, beaches, or deserts
- 7. plants — visible vegetation (trees, grass, flowers, bushes)
- 8. mountains and rivers — mountains, hills, rivers, lakes, or oceans
- 9. weather — visible weather conditions (e.g., snow, rain, fog, clear sky)
- 10. agricultural — farmlands, crops, agricultural machinery, or rural fields Please output your results in the following JSON format:\n {\n "manmade": "yes/no/maybe",\n "traffic": "yes/no/maybe",\n "buildings": "yes/no/maybe",\n "sign": "yes/no/maybe",\n "language": "yes/no/maybe",\n "natural": "yes/no/maybe",\n "plants": "yes/no/maybe",\n "mountains_and_rivers": "yes/no/maybe",\n "weather": "yes/no/maybe",\n "agricultural": "yes/no/maybe"\n }”

6-7 Points - Moderate Locatability:

Contains general buildings or street scenes. Has partial text information that is not sufficiently clear. Includes vehicle license plates (enabling inference of country/region). Features seasonal or temporal characteristics. 4-5 Points - Low Locatability: Primarily natural landscapes with a lack of man-made identifiers. Has blurred building outlines without specific features. Contains partial environmental clues that are not sufficiently clear. 1-3 Points - Extremely Low Locatability: Indoor photos with no geographical features. Abstract or artistic images. Solid-color backgrounds or minimalist designs. No geographical information

clues whatsoever.

0 Points - Unlocatable: Completely unable to obtain geographical information. Damaged or unrecognizable images. Please output in the following JSON format:\n [\n {\n"Image Name": "Image File Name",\n"Locatability Score": 8.5,\n"Image Clarity": 9,\n"Amount of Text in Image": 8,\n"Distinctiveness of Cultural Landscape": 7,\n"Distinctiveness of Natural Landscape": 6,\n“ Vehicle and Pedestrian Information": 9,\n "Main Geographical Clues": ["Specific descriptions of geographical information"]}\n ] Please begin analyzing the images.”

|Output: “[\n {\n"Locatability Score": 5,\n "Image Clarity": 6,\n“ Amount of Text in Image": 0,\n“ Distinctiveness of Cultural Landscape": 3,\n"Distinctiveness of Natural Landscape": 8,\n "Vehicle and Pedestrian Information": 4,\n "Main Geographical Clues": [\n "Unpaved gravel road",\n "Dense surrounding vegetation, possibly a temperate rainforest or subtropical forest",\n "Obvious hillsides and vegetation cover on both sides of the road",\n "Cyclists visible in the distance, indicating the road may be in a rural or remote<br><br>area",\n "Lighting indicates a sunny day, likely daytime"\n]}\n ]”|
|---|

|Output: "```json\n {\n \"manmade\": \"yes\",\n \"traffic\": \"yes\",\n \"buildings\": \"no\",\n \"sign\": \"yes\",\n \"language\": \"yes\",\n \"natural\": \"yes\",\n \"plants\": \"yes\",\n \"mountains_and_rivers\": \"no\",\n \"weather\": \"yes\",\n \"agricultural\": \"no\"\n }\n ```"|
|---|

[Figure 95]

[Figure 96]

Figure 8 Prompts in locatability scoring and category annotation.

|v<br><br>GeoSeek<br><br>Clues: [Most buildings are Soviet-style, with sharp edges and simple decorations., The Arch of Triumph is a landmark of Pyongyang., In the distance, the triangular skyscraper Ryugyong Hotelalso a landmark of Pyongyang]<br><br>CountryIdentification: Based on the architectural style and the characteristics of landmark buildings, it can be judged that this is Pyongyang, North Korea.<br><br>RegionalGuess: Both the Arch of Triumph and Ryugyong Hotel are located in Pyongyang, so it can be<br><br>PreciseLocalization: The specific location of the Arch of Triumph is on the roundabout of Chilsongmun Street.<br><br>|
|---|

|Volunteer<br><br>Most buildings are Soviet-style,<br><br>characterized by sharp edges and simple decorations. The Arch of Triumph is a landmark of Pyongyang, and in the distance,<br><br>the triangular skyscraper Ryugyong Hotel which is also a iconic structure of Pyongyang can be seen.<br><br>|
|---|

|[Figure 97]<br><br>| | |
|---|---|
| | |
<br><br>| |
|---|
|
|---|

|[Figure 98]<br><br>Qwen:<br><br>Clues: [ The architectural style of the monument resembles a triumphal arch, but with distinct Soviet-influenced<br><br>design elements., modernist and socialist realist aesthetic, a distinctive pyramid-shaped skyscraper is visible.]<br><br>CountryIdentification: The combination of the monumental arch and the unique pyramid-shaped building in the distance strongly suggests a location in a post-Soviet state. The architectural style, particularly the large-scale neoclassical monument, aligns with those found in former Soviet republics. The pyramid-shaped structure is identifiable as the 'Tower of Ismoil Somoni’.<br><br>RegionalGuess: The central placement of the arch and the surrounding urban infrastructure suggest this is a major civic or governmental area. The visibility of the Tower of Ismoil Somoni confirms that this is in the capital city of Tajikistan, which is Dushanbe.<br><br>PreciseLocalization: : The Arch of Triumph in Dushanbe is a well-known landmark located near the main square and facing the Tower of Ismoil Somoni. The image shows the arch from the front with the tower visible in the background, confirming the exact location<br><br>|
|---|

|GPT-4o<br><br>[Figure 99]<br><br>Clues: [The architecture resembles the Arch of Triumph, but it is distinct from the one in Paris., The surrounding buildings have a distinct socialist architectural style., The pyramid-shaped building in the<br><br>background is the Ryugyong Hotel, a well-known landmark., The overall urban design and landscaping are characteristic of North Korea.]<br><br>CountryIdentification: The presence of the Ryugyong Hotel and the unique socialist architecture strongly suggest this is Pyongyang, North Korea.<br><br>RegionalGuess: The combination of the Arch of Triumph and the Ryugyong Hotel confirms the region as Pyongyang. PreciseLocalization: The image is taken directly in front of the Arch of Triumph in Pyongyang, which is a precise location.|
|---|

|v<br><br>GeoSeek<br><br>Clues: [ The two-way road divider uses yellow concrete blocks, which are commonly seen in Myanmar., The bus ahead has rounded Burmese script visible on it, The tropical vegetation visible in the image suggests a southern region of Myanmar.]<br><br>CountryIdentification: The use of yellow concrete dividers and the presence of Burmese script strongly indicate that this location is in Myanmar RegionalGuess: The combination of tropical vegetation and well-maintained asphalt roads suggests the location is likely Yangon, a major city in southern Myanmar. PreciseLocalization: The urban environment and infrastructure indicate that this is likely within the city limits of Yangon.<br><br>|
|---|

|Volunteer<br><br>The two-way road divider uses yellow concrete blocks, which are commonly seen in Myanmar. Rounded Burmese script is visible on the bus ahead. Yangon is located in the southern part of Myanmar, so it has a more pronounced tropical atmospheremost of the vegetation visible in the image is tropical. Additionally, Yangon is a more economically<br><br>developed city in Myanmar; the road in the image is<br><br>asphalt and the road infrastructure is wellmaintained.<br><br>|
|---|

|[Figure 100]<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
|---|

|[Figure 101]<br><br>Qwen:<br><br>Clues: [The bus has a design and color scheme commonly seen in Southeast Asian countries., The<br><br>road signs and vehicles appear to be right-hand drive, which is common in many countries including Vietnam., The vegetation includes tropical trees and plants, suggesting a tropical climate. The architecture of the buildings along the street is typical of urban areas in Southeast Asia.]<br><br>CountryIdentification: The combination of tropical vegetation, right-hand drive vehicles, and architectural style points towards a country in Southeast Asia. The bus design, with its colorful rear and layout, is particularly reminiscent of public transportation in Vietnam.<br><br>RegionalGuess: The urban infrastructure and dense building pattern suggest a city in a region that experiences heavy rainfall. Southern Vietnam, such as Ho Chi Minh City.<br><br>PreciseLocalization: Despite the poor visibility due to rain, the road configuration and traffic patterns suggest a main urban road. However, without clear text or landmarks, precise localization is not possible.<br><br>|
|---|

|GPT-4o<br><br>[Figure 102]<br><br>Clues: [The bus design and markings suggest a public transport system common in Southeast Asia. The vegetation and climate indicate a tropical region, likely experiencing monsoon rains. The road infrastructure, including the orange barriers, is typical of developing countries in Asia., The vehicles are driving on the right side of the road.]<br><br>CountryIdentification: The combination of tropical vegetation, monsoon weather, and road infrastructure suggests a Southeast Asian country. Driving on the right side of the road narrows it down further.<br><br>RegionalGuess: The tropical climate and semi-urban infrastructure suggest a<br><br>region in the Philippines that experiences monsoon rains, likely in Luzon or Visayas.<br><br>PreciseLocalization: The bus markings and signage could indicate a specific city or municipality, but the image lacks clear text or identifiers.<br><br>|
|---|

|Volunteer<br><br>The combination of Arabic language, all<br><br>buildings made of sandstone, chaotic traffic (vehicles driving disorderly on roads with extremely worn markings, making it impossible to distinguish the direction of traffic), and blue-andwhite license plates (blue on top, white on bottom) indicates that this location is in Egypt.<br><br>|
|---|

|v<br><br>GeoSeek<br><br>Clues: [ Arabic language on signs, Buildings predominantly made of sandstone, Chaotic traffic with vehicles driving on heavily worn road markings, Blue and white license plates, Poor road conditions, Low-rise sandstone buildings, High building density, Chaotic traffic patterns]<br><br>CountryIdentification: The use of Arabic language, the architectural style with sandstone buildings, and the specific license plate style are indicative of Egypt. RegionalGuess: The combination of poor road conditions, high building density, and chaotic<br><br>traffic is characteristic of Cairo, the capital city of Egypt<br><br>PreciseLocalization: The image does not provide specific landmarks, street names, or other unique identifiers to pinpoint an exact location within Cairo.|
|---|

|[Figure 103]<br><br>| |
|---|
<br><br>|
|---|

|[Figure 104]<br><br>Qwen:<br><br>Clues: : [The road signs are in Arabic and English, indicating a bilingual region., The architecture consists of low-<br><br>rise buildings with flat roofs, typical of arid regions., The presence of palm trees and clear blue sky suggests a<br><br>warm, dry climate., The style of vehicles and road infrastructure is modern and well-maintained.]<br><br>CountryIdentification: The use of Arabic script on the signs is a strong indicator of a Middle Eastern country. The combination of modern infrastructure and desert-like environment narrows it down to countries like Saudi Arabia, UAE, or Qatar.<br><br>RegionalGuess: The phone number on the red sign (0555191110) follows the format used in Saudi Arabia, where 05 is the mobile prefix. The urban layout and construction style resemble areas in Riyadh or Jeddah.<br><br>PreciseLocalization: The road design and signage style match those seen in cities like Riyadh, particularly in newer developments such as King Abdullah Economic City (KAEC) or Olaya Street areas. However, no unique landmarks or named streets are visible to confirm an exact location.<br><br>|
|---|

|GPT-4o<br><br>[Figure 105]<br><br>Clues: [Arabic text on billboards, Driving on the right side of the road, Architecture style typical of Middle Eastern urban areas, Phone number format with '015' prefix, common in Egypt, Urban environment with modern infrastructure, Billboards and advertisements indicating a commercial area]<br><br>CountryIdentification: : The presence of Arabic text, driving direction, and phone number format strongly suggest a Middle Eastern country, specifically Egypt.<br><br>RegionalGuess: The urban setting and infrastructure suggest a major city in Egypt, likely Cairo or its surrounding areas. PreciseLocalization: The specific road layout and advertisements suggest this is a commercial or residential area within Cairo.|
|---|

###### Figure 9 Comparison between human-annotated and AI-standardized thought processes versus direct AI annotation.

GeoSeek-CoT "```json { "ChainOfThought ": { "CountryIdentification ": { "Clues ": [ "Prayer flags in the lower left corner, which are a cultural and religious symbol commonly found in Tibetan regions. ", "The landscape features high-altitude terrain with barren sand dunes and gravel plains. ", "Snow-capped peaks in the background suggest a high-altitude region, likely above 5000m. "], "Reasoning ": "The presence of prayer flags and the high-altitude mountainous terrain strongly indicate the Tibetan region in China. ", "Conclusion ": "China ", "Uncertainty ": "Low " }, "RegionalGuess ": { "Clues ": [ "The combination of high-altitude terrain, snow-capped peaks, and the presence of prayer flags suggests the southern part of the Tibet Autonomous Region. ", "The winding road visible in the image is characteristic of mountain roads in this area. "], "Reasoning ": "The geographical and cultural features align with the southern part of Tibet, near the Himalayan range. ", "Conclusion ": "Tibet, Tingri ", "Uncertainty ": "Low " }, "PreciseLocalization ": { "Clues ": [ "The winding road visible in the image resembles the S515 (Everest Road), which is a known route in Tingri County leading towards Mount Everest Base Camp. "], "Reasoning ": "The specific road layout and the surrounding terrain match the S515 road in Tingri County. ", "Conclusion ": "Tingri County, S515 (Everest Road) ", "Uncertainty ": "Low " } }, "FinalAnswer ": "China; Tibet, Tingri; Tingri County, S515 (Everest Road) " } ```"

[Figure 106]

[Figure 107]

"```json { "ChainOfThought ": { "CountryIdentification ": { "Clues ": [ "The architecture is made of rammed earth, indicating a dry climate with low precipitation and possibly significant wind and sand. ", "Thick walls and small windows suggest cold winters. ", "The crescent-shaped decorations on the railings are a symbol of Islam, indicating a region with a significant Muslim population. " ], "Reasoning ": "In China, regions with a dry climate, cold winters, and a significant Muslim population are primarily located in southern and eastern Xinjiang. The architectural style and cultural elements align with this region. ", "Conclusion ": "China ", "Uncertainty ": "Low " }, "RegionalGuess ": { "Clues ": [ "The dense arrangement of buildings suggests an urban area. ", "The green, intricate decorations on the distant building rooftops indicate a traditional Islamic architectural influence. ", "Rammed earth buildings are typically older and more likely to be found in historic or older parts of cities. " ], "Reasoning ": "In Xinjiang, Kashgar City is known for its well-preserved traditional rammed earth architecture, particularly in the old city area. The features in the image strongly align with Kashgar's architectural and cultural characteristics. ", "Conclusion ": "Xinjiang; Kashgar City ", "Uncertainty ": "Low " }, "PreciseLocalization ": { "Clues ": [ "The specific style and preservation of the rammed earth buildings, along with the intricate Islamic decorations, are characteristic of Kashgar's old city. ", "The dense urban layout and visible architectural details suggest a location within the historic Yawagh Subdistrict of Kashgar. " ], "Reasoning ": "The image likely depicts a part of Kashgar's old city, specifically the Yawagh Subdistrict, which is known for its traditional Islamic architecture and dense urban layout. ", "Conclusion ": "Yawagh Subdistrict, Kashgar City ", "Uncertainty ": "Moderate " } }, "FinalAnswer ": "China; Xinjiang; Kashgar City, Yawagh Subdistrict " } ```"

[Figure 108]

[Figure 109]

GeoSeek-Loc

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

Indonesia; South Sumatra; Karang Agung

United States; PA, Lackawanna County; Clifton Township

Russia; Samara Oblast; Novosemeykino

Indonesia; Bali; Selat

GeoSeek-Val

[Figure 114]

[Figure 115]

"solution": "United States; PR, Luquillo; Luquillo ",

- "Locatability Score": 6.5, "cls":{"manmade", "traffic", "buildings", "natural", "plants", "weather" }

- "lat": 18.379386389062, "lng": -65.743879999414

"solution": "<answer> United States; PR; Río Piedras </answer>", "Locatability Score": 8.5, "cls":{"manmade", "traffic, "sign", "language", "natural", "plants", "weather", } "lat": 18.440560193504, "lng": -66.069041327237

[Figure 116]

[Figure 117]

"solution": "<answer> Guinea-Bissau;

Barro; </answer>",

"Locatability Score": 6, "cls": { "manmade", "buildings", "natural", "plants", "weather", } "lat": 12.460572985091, "lng": -15.302325972063

[Figure 118]

[Figure 119]

[Figure 120]

"solution": "<answer> Zambia; Lusaka Province; </answer>", "Locatability Score": 4, "cls": {"manmade", "natural", "plants", "weather“ } "lat": -15.440055364801658, "lng": 28.05866456484767

"solution": "<answer> India;

Maharashtra; Mumbai -,H/E Ward </answer>", "Locatability Score": 7.5, "cls": {"manmade", "traffic", "buildings", "natural", "plants", "weather", }

- "lat": 19.0723762446242, "lng": 72.86419881681877

"solution": "<answer> South Africa; Limpopo; Blouberg Local Municipality </answer>", "Locatability Score": 4.5, "cls": {"manmade", "natural", "plants", "weather"} "lat": -22.740179663893,

"lng": 28.488901166989

- Figure 10 More cases of GeoSeek Dataset.

|Spain; Catalonia; Barcelona<br><br>[Figure 121]<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

|v<br><br>GeoAgent: Spain; Catalonia; Barcelona<br><br>Clues: [ Gothic architectural style, Stone statues with religious themes, Architectural details with pointed arches and ornate carvings]<br><br>CountryIdentification: The Gothic architectural style is a strong indicator of Europe, as this style was prevalent in medieval and early modern Europe.<br><br>The stone statues with religious themes and the intricate carvings further suggest a European context, as these elements are common in historical European churches and cathedrals. These clues strongly point to Spain, as it is known for its rich Gothic heritage.<br><br>RegionalGuess: The Gothic architectural style is a hallmark of Spain, especially in cities like Barcelona and Seville. The stone statues with religious themes and the intricate carvings are consistent with Spanish Gothic architecture, which is renowned for its ornate details. Additionally, the cultural and historical significance of Barcelona aligns with the regional guess of Catalonia, which is famous for its Gothic heritage.<br><br>PreciseLocalization: The stone statues with religious themes are highly characteristic of Barcelona, the capital city of Catalonia. The intricate carvings and pointed arches are typical of Catalan Gothic architecture, which is unique to this city.<br><br>|
|---|

|China; Sichuan ; Chengdu<br><br>[Figure 122]<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
<br><br>[Figure 123]|
|---|

|v<br><br>GeoAgent: China; Sichuan ; Chengdu<br><br>Clues: [Traditional Chinese architecture, Chinese characters on the sign, Tropical vegetation visible outside, Religious or cultural site setting]<br><br>CountryIdentification: The traditional Chinese architecture with its distinctive roof tiles, wooden structures, and ornate carvings is a strong indicator of China. The Chinese characters on the sign further confirm this, as they are unique to the Chinese language. Additionally, the tropical vegetation visible<br><br>outside aligns with the subtropical climate typical of China, which has a warm and humid environment.<br><br>RegionalGuess: The traditional Chinese architecture with dragon motifs and the use of Chinese characters on the sign are highly characteristic of Sichuan Province, known for its unique blend of cultural and historical elements. The subtropical vegetation and the cultural site setting further support this regional guess, as these features are common in Chengdu, the capital city of Sichuan.<br><br>PreciseLocalization: The subtropical vegetation and the cultural site setting further confirm the precise localization to Chengdu, as this city is renowned for its rich cultural heritage and unique architectural style.<br><br>|
|---|

|Finland; Tampere; Tampere<br><br>[Figure 124]<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

|v<br><br>GeoAgent: Finland; Tampere; Tampere<br><br>Clues: [Modern amusement park design, Tall tower structure, Waterfront setting, European-style architecture]<br><br>CountryIdentification: The modern amusement park design with roller coasters and other attractions is characteristic of European countries, which often feature such facilities. The tall tower structure resembles the iconic Näsinneula Tower in Tampere, Finland, known for its unique design. The waterfront setting and the overall urban environment suggest a developed country with a focus on tourism and leisure. These clues strongly point to Finland<br><br>RegionalGuess: The Näsinneula Tower is a distinctive feature of Tampere, which is the only city in Finland with this iconic structure. The amusement park's design and layout are consistent with Finnish urban planning, which often incorporates recreational areas near water bodies.<br><br>PreciseLocalization: The amusement park's design and the waterfront setting are also specific to Tampere, as these features are not found in any other Finnish city. The modern urban planning aligns with the city's reputation for being a hub of innovation and culture.<br><br>|
|---|

|China; Beijing ; Beijing<br><br>[Figure 125]<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
|---|

|v<br><br>GeoAgent: China; Beijing ; Beijing<br><br>Clues: [Chinese characters on signage, Traditional Chinese script, Urban environment with small shops, Cultural elements like red lanterns and traditional architecture]<br><br>CountryIdentification: The presence of Chinese characters strongly indicates the country is China, as these are unique to this nation. The urban setting with small shops and cultural elements further supports this identification, as they are common in Chinese cities.<br><br>RegionalGuess: The use of Chinese characters and traditional script is a strong indicator of Beijing, as this is the only major city in China where such script is commonly used. The urban environment with small shops and cultural elements aligns with Beijing's characteristics. The electric poles with a<br><br>specific design further confirm the regional guess of Beijing, which is known for its unique utility pole style.<br><br>PreciseLocalization: The urban environment with small shops and cultural elements matches the typical scene in Beijing. Additionally, the electric poles with their specific design are a hallmark of Beijing.<br><br>|
|---|

|Argentina; Mendoza; Mendoza<br><br>[Figure 126]<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

|v<br><br>GeoAgent: Argentina; Mendoza; Mendoza<br><br>Clues: [Wooden wine press, Subtropical vegetation, Architectural style of buildings, Clear blue sky and sunny weather] CountryIdentification: The subtropical vegetation and the architectural style of the buildings pointing towards Argentina, as they align with the characteristics of South American countries. RegionalGuess: The wooden wine press is a distinctive feature associated with Mendoza, a province known for its wine production. The subtropical vegetation and the architectural style of the buildings, with their pinkish-tan color and red roofs, are characteristic of Argentine cities like Mendoza. PreciseLocalization: The wooden wine press is a unique feature found in Mendoza, which is the only city in Argentina with such a structure. The subtropical vegetation and the architectural style of the buildings, with their pinkish-tan color and red roofs, are exclusively associated with Mendoza.<br><br>|
|---|

|Mali; Bamako; Bamako<br><br>[Figure 127]<br><br>|[Figure 128]<br><br>[Figure 129]|
|---|
<br><br>| |
|---|
<br><br>[Figure 130]<br><br>[Figure 131]|
|---|

|v<br><br>GeoAgent: Mali; Bamako; Bamako<br><br>Clues: [Dark skin tones, Traditional clothing, Nighttime setting with minimal lighting, Social gathering on the ground]<br><br>CountryIdentification: The dark skin tones suggest a sub-Saharan African country, as these features are common in this region. The nighttime setting with minimal lighting often found in Africa, where electricity is less widespread during power outages. The social gathering on the ground is typical of African cultural practices. These clues strongly indicate the country is Mali.<br><br>RegionalGuess: The traditional clothing are characteristic of Mali, a country known for its Sahelian culture. The nighttime setting with minimal lighting suggests the arid climate of Mali, which is consistent with the region's geography. The social gathering on the ground is a common practice in Mali, further supporting the regional guess of Bamako.<br><br>PreciseLocalization: The social gathering on the ground is a common practice in Mali, and the arid landscape visible in the background matches the environment of Bamako. These clues strongly suggest the precise localization is Bamako.<br><br>|
|---|

- Figure 11 More cases of IM2GPS3K [103].

|[Figure 132]<br><br>Canada; Quebec; Montreal<br><br>| |
|---|
<br><br>|
|---|

|v<br><br>GeoAgent: Canada; Quebec; Montreal<br><br>Clues: [Modern urban architecture, North American-style road infrastructure, Yellow traffic lights, Temperate deciduous trees visible in the background ] CountryIdentification: The image depicts a modern urban environment with North American-style road infrastructure, including yellow traffic lights, which are characteristic of Canada. The temperate deciduous trees suggest a temperate climate, which is typical of eastern North America.<br><br>RegionalGuess: The modern urban architecture with high-rise buildings and the temperate deciduous trees suggests a city located in a temperate region, likely in the northeastern part of Canada due to the climate. The yellow traffic lights are a distinctive feature of Montreal,. This combination of urban features and the temperate climate points to the metropolitan area of Montreal, which is in the province of Quebec.<br><br>PreciseLocalization: The modern urban architecture with high-rise buildings and the temperate deciduous trees is characteristic of Montreal, which is known for its distinctive yellow traffic lights and urban environment. The temperate climate and the urban setting further confirm the location within the<br><br>metropolitan area of Montreal.|
|---|

|[Figure 133]<br><br>Argentina; Corrientes; Corrientes<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

|v<br><br>GeoAgent: Argentina; Corrientes; Corrientes<br><br>Clues: [Tropical vegetation, Mediterranean-style architecture, Flat terrain, Urban environment]<br><br>CountryIdentification: The tropical vegetation and Mediterranean-style architecture suggest a subtropical or tropical climate, which is characteristic of South America. The flat terrain and urban environment further support this, as these features are common in the urban areas of Paraguay, a country<br><br>known for its subtropical climate and flat landscape.<br><br>RegionalGuess: The flat terrain and urban environment align with the urban areas of Asunción, the capital city of Paraguay, which is located in the subtropical region. The electric poles with horizontal wires are a distinctive feature of the urban infrastructure in this region. These clues strongly suggest the regional guess of Corrientes, as it is the only nearby province with this combination of features.<br><br>PreciseLocalization: The flat terrain and urban environment are consistent with the city of Corrientes, which is known for its subtropical climate and Mediterranean-style architecture. The white buildings with red-tiled roofs are a distinctive feature of the city's colonial-era architecture.<br><br>|
|---|

|[Figure 134]<br><br>Mauritania; Nouakchott; Nouakchott<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
|---|

|v<br><br>GeoAgent: Mauritania; Nouakchott; Nouakchott<br><br>Clues: [Arid desert environment, Tropical vegetation, Flat terrain, Brick buildings, Road infrastructure], CountryIdentification: The arid desert environment and tropical vegetation suggest a location in a semi-arid region, which is characteristic of North<br><br>Africa. The flat terrain and brick buildings are consistent with the Sahelian climate zone, which is found in Mali. The road infrastructure, including the<br><br>visible signage, aligns with the developed infrastructure seen in urban areas in this region. These clues strongly indicate the country of Mauritania RegionalGuess: The flat terrain and brick buildings are consistent with the urban infrastructure seen in the city of Nouakchott, the capital of Mauritania. The Arabic script on the signage indicates a Muslim-majority country, which further supports the identification of Mauritania PreciseLocalization: The Arabic script on the signage confirms the use of Arabic, which is the official language of Mauritania. These clues strongly suggest the precise localization of Nouakchott, the capital city of Mauritania.<br><br>|
|---|

|[Figure 135]<br><br>China; Hainan; Sanya<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

|v<br><br>GeoAgent: China; Hainan; Sanya<br><br>Clues: [Tropical vegetation, Road design with white center lines, License plates with blue background, Mountainous terrain visible in the distance ],<br><br>CountryIdentification: The tropical vegetation and mountainous terrain suggest a warm, subtropical environment, which is characteristic of East Asia.<br><br>The road design with white center lines and the blue license plates are consistent with the design standards used in China. These clues strongly indicate that the location is in China.<br><br>RegionalGuess: The tropical vegetation and subtropical climate suggest a warm, humid environment, which is typical of the subtropical region of southern China. The mountainous terrain visible in the distance aligns with the Hainan Island's hilly landscape. These clues strongly suggest the regional guess of Hainan Province.<br><br>PreciseLocalization: The mountainous terrain visible in the distance aligns with the hilly landscape of Hainan Island. The blue license plates and the road design with white center lines are consistent with Sanya City, the capital city of Hainan Province, which has a unique license plate design and maintains a clean, modern urban appearance.<br><br>|
|---|

|[Figure 136]<br><br>Australia; Victoria; Melbourne<br><br>| |
|---|
<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

|v<br><br>GeoAgent: Australia; Victoria; Melbourne<br><br>Clues: [Tropical vegetation, Flat terrain, Modern residential architecture, Blue satellite dish, Subtropical climate] CountryIdentification: The tropical vegetation and flat terrain suggest a warm, humid environment typical of tropical climates. The modern residential<br><br>architecture and the presence of a blue satellite dish are common in urban areas, which narrows it down to a developed country. The subtropical climate<br><br>further supports this, as it is characteristic of coastal regions in tropical Africa. These clues strongly indicate Nigeria.<br><br>RegionalGuess: The tropical vegetation and flat terrain suggest a coastal or semi-coastal region, which is characteristic of Lagos State, the only major city in Nigeria with such features. The modern residential architecture and the blue satellite dish are consistent with the urban environment of Lagos, which is known for its developed infrastructure. The subtropical climate aligns with the tropical savanna climate found in this region.<br><br>PreciseLocalization: The modern residential architecture and the blue satellite dish are consistent with the urban environment of Lagos City, the capital and largest city in Nigeria. The specific architectural style with metal gates is typical of the urban residential areas in Lagos.<br><br>|
|---|

|[Figure 137]<br><br>Australia; Victoria; Melbourne<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

|v<br><br>GeoAgent: Australia; Victoria; Melbourne<br><br>Clues: [Flat, marshy landscape, Vegetation typical of temperate climates, Clear blue sky with contrails, Rural setting]<br><br>CountryIdentification: The blue license plates are a strong indicator of Australia, as this is the only country in Australia's state of Victoria that uses blue plates for passenger cars. The left-hand driving is consistent with Australian road conventions, which further supports this identification. The modern vehicles and urban setting suggest a developed metropolitan area, which aligns with the urban infrastructure typically found in major cities in Australia.<br><br>RegionalGuess: The blue license plates are a distinctive feature of Victoria, one of the states in Australia, as it is the only state that uses blue plates for passenger cars. The urban environment and the flat terrain with some trees are characteristic of the Melbourne metropolitan area, which is known for its suburban settings and temperate climate. The modern vehicle design is consistent with the urban and suburban areas.<br><br>PreciseLocalization: The urban environment with modern vehicles and the flat terrain with some trees is characteristic of the Melbourne metropolitan area, which is located in a temperate climate zone.<br><br>|
|---|

- Figure 12 More cases of GeoSeek-Val.

|Jamaica; Portland; Portland<br><br>[Figure 138]<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

|v<br><br>GeoAgent: Indonesia; Bali; Bandung<br><br>Clues: [Tropical vegetation, Rugged terrain, Rainforest environment ]<br><br>CountryIdentification: The tropical vegetation and rainforest environment are characteristic of Southeast Asia, which narrows down the country to<br><br>Indonesia due to its equatorial position and lush tropical forests. The rugged terrain further supports this identification, as Indonesia is known for its mountainous regions. These clues strongly suggest the country is Indonesia, with no significant uncertainty.<br><br>RegionalGuess: The cultural context with Balinese elements suggests the presence of the Hindu culture associated with Bali, a nearby island. These clues point to the regional guess of West Java, as it is the only major province on Java with a significant mountainous region and proximity to Bali. The conclusion is West Java, with low uncertainty.<br><br>PreciseLocalization: West Java’s rainforest, volcanic terrain, and Balinese influences converge at Tangkuban Perahu volcanic area near Bandung, fitting all prior clues.|
|---|

|Dominica; StDavid; StDavid<br><br>|[Figure 139]|
|---|
<br><br>[Figure 140]<br><br>|
|---|

|v<br><br>GeoAgent: Indonesia; West Java ; Ubud<br><br>Clues: [Tropical vegetation, Hilly terrain, Palm trees, Rural setting]<br><br>CountryIdentification: The tropical vegetation and hilly terrain suggest a warm, humid environment typical of tropical or subtropical regions. The presence of palm trees further indicates a tropical climate, which is characteristic of Southeast Asia. These clues strongly suggest the country is<br><br>Indonesia, as it is the only major Southeast Asian country with this combination of features. The conclusion is Indonesia, with low uncertainty. RegionalGuess: The palm trees and rural setting align with the tropical climate in Southeast Asia. The electric poles with wires suggest a developed infrastructure, which is in the urbanized areas of Indonesia's main islands. These clues point to the regional guess of West Java. PreciseLocalization: The tropical vegetation, hilly terrain, and palm trees are characteristic of the volcanic islands in Indonesia, which are part of the Sundaland region. The rural setting, which appears to be a traditional Balinese-style structure, suggest the city of Ubud.|
|---|

|Brazil; São Paulo; Conchas<br><br>[Figure 141]<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

|v<br><br>GeoAgent: China; Guangdong; Guangzhou<br><br>Clues: [Tropical vegetation, Hilly terrain, Subtropical climate, Rural setting] CountryIdentification: The tropical vegetation and subtropical climate suggest a warm, humid environment typical of tropical or subtropical regions. The hilly terrain and rural setting indicate a developing country with agricultural activities. These clues point to China, as a major country in this region. RegionalGuess: The clues suggest a warm, humid environment, which is characteristic of the subtropical regions in southern China. These clues point to Guangdong Province, as it is the only province in this region with these characteristics. The conclusion is Guangdong Province, with low uncertainty.<br><br>PreciseLocalization: The hilly terrain and rural setting align with the agricultural landscape found in the Pearl River Delta region. The specific tree<br><br>species visible in the image, which appears to be a type of bamboo, is common in this region. These clues point to the city of Guangzhou, which has this type of vegetation.|
|---|

|Netherlands; Aruba; Aruba<br><br>[Figure 142]<br><br>| | | |
|---|---|---|
| | | |
| | | |
| | | |
|
|---|

|v<br><br>GeoAgent: Indonesia; Bali ; Kuta<br><br>Clues: [Underwater environment, Corals and marine life, Tropical climate indicators] CountryIdentification: The underwater scene with corals and marine life suggests a tropical or subtropical environment, which is characteristic of Southeast Asian countries like Indonesia and the Philippines. These clues point to the country of Indonesia, as it has extensive coral reefs.<br><br>RegionalGuess: The underwater scene with vibrant corals and marine life is a strong indicator of Indonesia, as this country is renowned for its extensive coral reefs and tropical marine environments. The clarity of the water and the specific types of corals present align with the natural beauty found in Indonesian islands. These clues suggest the regional guess of Bali.The conclusion is Bali, with low uncertainty.<br><br>PreciseLocalization: The underwater scene with vibrant corals and the tropical marine environment is unmistakably associated with Bali, as this island<br><br>is famous for its coral reefs and unique marine biodiversity. These clues strongly suggest the precise localization of Kuta, a popular tourist destination in Bali known for its beaches and coral reefs. The conclusion is Kuta, with no uncertainty.|
|---|

|Costa Rica; Limón; Talamanca<br><br>[Figure 143]<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

|v<br><br>GeoAgent: Indonesia; WestJava; Bogor<br><br>Clues: [Tropical vegetation, Spider species, Nighttime setting] CountryIdentification: The tropical vegetation and the spider species strongly suggest a Southeast Asian country. These clues point towards countries like Indonesia, Malaysia, or Thailand, which are known for their biodiversity and tropical climates. RegionalGuess: The tropical rainforest environment and the presence of the Nephila antipodiana spider, which is native to Southeast Asia, strongly indicate the regional guess of West Java. Above clues lead to the regional guess of West Java with low uncertainty.<br><br>PreciseLocalization: The tropical rainforest environment and the specific identification of the Nephila antipodiana spider, which is endemic to Indonesia, strongly suggest the precise localization of Bogor. This city is famous for its botanical gardens and is located in West Java. These clues lead to the precise localization of Bogor with high confidence.|
|---|

|Panama; Coclé; La Pintada<br><br>[Figure 144]<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

|v<br><br>GeoAgent: Indonesia; Jakarta; Jakarta<br><br>Clues: [Orchid flower, Tropical vegetation, Lush green environment]<br><br>CountryIdentification: The image shows a vibrant orchid flower with a tropical appearance, which is characteristic of Southeast Asian countries known for their diverse flora. The lush green environment and the presence of palm trees further suggest a tropical climate, which is typical of Indonesia, Malaysia, and Thailand.<br><br>RegionalGuess: The lush green vegetation and tropical climate further align with Indonesia's geography. The architectural style visible in the background, with its modern design and tropical setting, matches Jakarta, the capital city.<br><br>PreciseLocalization: The orchid flower is a national symbol of Indonesia and is commonly associated with Jakarta, the capital city. The lush green environment and tropical vegetation are consistent with Jakarta's urban and suburban areas. The modern architectural style visible in the background is typical of Jakarta's contemporary urban development. These clues strongly suggest the precise localization is Jakarta.|
|---|

- Figure 13 Failure cases of GeoAgent.

|China; Beijing ; Beijing<br><br>[Figure 145]<br><br>| |
|---|
<br><br>| |
|---|
<br><br>|
|---|

|v<br><br>GeoAgent: China; Beijing ; Beijing<br><br>Clues: [ Chinese characters on signage, Traditional Chinese script, Urban environment with small shops, Cultural elements like red lanterns and traditional architecture]<br><br>CountryIdentification: The presence of Chinese characters strongly indicates the country is China, as these are unique to this nation. The urban setting with small shops and cultural elements further supports this identification, as they are common in Chinese cities.<br><br>RegionalGuess: The use of Chinese characters and traditional script is a strong indicator of Beijing, as this is the only major city in China where such script is commonly used. The urban environment with small shops and cultural elements aligns with Beijing's characteristics. The electric poles with a specific design further confirm the regional guess of Beijing, which is known for its unique utility pole style.<br><br>PreciseLocalization: The urban environment with small shops and cultural elements matches the typical scene in Beijing. Additionally, the electric poles with their specific design are a hallmark of Beijing.<br><br>|
|---|

|China; Beijing ; Beijing<br><br>[Figure 146]<br><br>| |
|---|
|
|---|

|v<br><br>GeoAgent: China; Beijing ; Beijing<br><br>Clues: [ Chinese characters on the signage , Traditional Chinese script , Urban environment , Cultural elements like red decorations ] CountryIdentification: The presence of Chinese characters and traditional script strongly suggests that this is in China, as these are unique to the Chinese language and culture. The urban environment and cultural elements further confirm this. RegionalGuess: The urban setting and the red decorations, which are commonly associated with Chinese culture, suggest a major city. These clues point towards Beijing, as it is the capital city of China and known for its urban environment and cultural elements. PreciseLocalization: The urban environment and the red decorations, which are commonly associated with Chinese culture, suggest a major city. These clues point towards Beijing,<br><br>|
|---|

|China; Beijing ; Beijing<br><br>[Figure 147]<br><br>| |
|---|
<br><br>| |
|---|
|
|---|

|v<br><br>GeoAgent: China; Beijing ; Beijing<br><br>Clues: [ Red brick buildings , Traditional Chinese architecture , Winter clothing , Overhead power lines ]<br><br>CountryIdentification: The red brick buildings and traditional Chinese architecture suggest an East Asian country, as these are characteristic of Chinese urban environments. The winter clothing indicates a temperate climate, which is common in northern China. The overhead power lines are consistent with urban infrastructure found in many cities around the world. These clues strongly suggest the country is China due to its unique blend of traditional architecture and modern urban infrastructure.<br><br>RegionalGuess: The red brick buildings and traditional Chinese architecture are characteristic of Beijing's historic hutong districts, which are located in a major city known for its historical significance and cultural heritage. The overhead power lines are typical of urban infrastructure. These clues strongly suggest the regional guess of Beijing.<br><br>PreciseLocalization: The red brick buildings and traditional Chinese architecture are characteristic of Beijing's historic hutong districts, which are located in the capital city of China. The winter clothing and temperate climate are consistent with the northern part of China, where Beijing is situated.|
|---|

|China; Beijing ; Beijing<br><br>[Figure 148]<br><br>| |
|---|
<br><br>|
|---|

|v<br><br>GeoAgent: China; Beijing ; Beijing<br><br>Clues: [ Red brick buildings, Bare trees , Overhead power lines , Urban environment ]<br><br>CountryIdentification: The red brick buildings and bare trees suggest a temperate climate, which is common in northern China. The overhead power lines and urban environment are consistent with developed infrastructure found in major cities. These clues strongly indicate that the location is in China<br><br>RegionalGuess: The red brick buildings and bare trees narrows the regional guess to Beijing or Tianjin, as these are the only two major cities in China's northern region with such features. The overhead power lines and urban environment further confirm the presence of a developed city. These clues strongly suggest the regional guess of Beijing, as Tianjin has a more coastal appearance<br><br>PreciseLocalization: The red brick buildings and bare trees are characteristic of traditional Chinese architecture, which is commonly seen in Beijing, a major city known for its historical buildings and winter climate.<br><br>|
|---|

|China; Beijing ; Beijing<br><br>[Figure 149]<br><br>|
|---|

|v<br><br>GeoAgent: China; Beijing ; Beijing<br><br>Clues: [ Electric poles with horizontal wires , Bare trees , Winter-like environment ],<br><br>CountryIdentification: The electric poles with horizontal wires are a distinctive feature commonly found in China, where this type of wiring is prevalent for power distribution. The bare trees and winter-like environment suggest a temperate climate, which aligns with the northern part of China.<br><br>RegionalGuess: The horizontal electric pole wiring is a characteristic feature of urban areas in China, particularly in Beijing, which is known for its modern infrastructure and temperate climate. The bare trees and winter-like environment further support this regional guess, as they are consistent with the temperate climate of northern China. These clues strongly suggest the regional guess of Beijing<br><br>PreciseLocalization:. The bare trees and winter-like environment are consistent with the temperate climate of northern China's capital city. These clues strongly suggest the precise localization of Beijing, as it is the only major city in China with this type of urban infrastructure and temperate climate.<br><br>|
|---|

- Figure 14 Robustness of GeoAgent.

### References

- [1] Flickr. Accessed: 2025-11-19.
- [2] Geoguessr - let’s explore the world! Accessed: 2025-10-14.
- [3] OpenCage Geocoder. Accessed: 2025-10-21.
- [4] Tuxun:explore the world. Accessed: 2025-10-14.
- [5] Geolocation Estimation of Photos Using a Hierarchical Model and Scene Classification, pages 575–592. Springer International Publishing, Cham, 2018.
- [6] Exploiting the Earth’s Spherical Geometry to Geolocate Images, pages 3–19. Springer International Publishing, Cham, 2020.
- [7] Plonkit, 2025. Accessed: 2025-11-20.
- [8] The spatial distribution of population in 2015–2030 at a resolution of 30 arc (approximately 1 km at the equator), r2025a version v1, 2025. WorldPop, University of Southampton. DOI: 10.5258/SOTON/WP00845. Accessed: 2025-10-04.
- [9] Relja Arandjelovi´c, Petr Gronat, Akihiko Torii, Tomas Pajdla, and Josef Sivic. Netvlad: Cnn architecture for weakly supervised place recognition, May 2016.
- [10] Guillaume Astruc, Nicolas Dufour, Ioannis Siglidis, Constantin Aronssohn, Nacim Bouia, Stephanie Fu, Romain Loiseau, Van Nguyen Nguyen, Charles Raude, Elliot Vincent, Lintao XU, Hongyu Zhou, and Loic Landrieu. Openstreetview-5m: The many roads to global visual geolocation, Apr. 2024.
- [11] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.
- [12] Gabriele Berton, Carlo Masone, and Barbara Caputo. Rethinking visual geo-localization for large-scale applications, Apr. 2022.
- [13] Gabriele Moreno Berton, Valerio Paolicelli, Carlo Masone, and Barbara Caputo. Adaptive-attentive geolocalization from few queries: A hybrid approach. pages 2917–2926, Jan. 2021.
- [14] Vicente Vivanco Cepeda, Gaurav Kumar Nayak, and Mubarak Shah. Geoclip: Clip-inspired alignment between locations and images for effective worldwide geolocalization, Nov. 2023.
- [15] Qiguang Chen, Libo Qin, Jinhao Liu, Dengyun Peng, Jiannan Guan, Peng Wang, Mengkang Hu, Yuhang Zhou, Te Gao, and Wanxiang Che. Towards reasoning era: A survey of long chain-of-thought for reasoning large language models. arXiv preprint arXiv:2503.09567, 2025.
- [16] Zhenfang Chen, Qinhong Zhou, Yikang Shen, Yining Hong, Zhiqing Sun, Dan Gutfreund, and Chuang Gan. Visual chain-of-thought prompting for knowledge-based visual reasoning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 1254–1262, 2024.
- [17] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. Deep reinforcement learning from human preferences. Advances in neural information processing systems, 30, 2017.
- [18] Brandon Clark, Alec Kerrigan, Parth Parag Kulkarni, Vicente Vivanco Cepeda, and Mubarak Shah. Where we are

- and what we’re looking at: Query based worldwide image geo-localization using hierarchies and scenes, Mar. 2023.
- [19] DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.
- [20] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. Advances in neural information processing systems, 36:10088–10115, 2023.
- [21] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171– 4186, 2019.
- [22] Zhiyang Dou, Zipeng Wang, Xumeng Han, Guorong Li, Zhipei Huang, and Zhenjun Han. Gaga: Towards interactive global geolocation assistant, Apr. 2025.
- [23] Elias Dritsas and Maria Trigka. Remote sensing and geospatial analysis in the big data era: A survey. Remote Sensing, 17(3):550, 2025.
- [24] Nicolas Dufour, David Picard, Vicky Kalogeiton, and Loic Landrieu. Around the world in 80 timesteps: A generative approach to global visual geolocation, Dec. 2024.
- [25] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323, 2022.
- [26] Yao Fu, Litu Ou, Mingyu Chen, Yuhao Wan, Hao Peng, and Tushar Khot. Chain-of-thought hub: A continuous effort to measure large language models’ reasoning performance. arXiv preprint arXiv:2305.17306, 2023.
- [27] Yixiao Ge, Haibo Wang, Feng Zhu, Rui Zhao, and Hongsheng Li. Self-supervising fine-grained region similarities for large-scale image localization, July 2020.
- [28] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pages 1861–1870. Pmlr, 2018.
- [29] Lukas Haas, Silas Alberti, and Michal Skreta. Learning generalized zero-shot learners for open-domain image geolocalization, Feb. 2023.
- [30] Lukas Haas, Michal Skreta, Silas Alberti, and Chelsea Finn. Pigeon: Predicting image geolocations. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 12893–12902, Seattle, WA, USA, June 2024. IEEE.
- [31] Xiao Han, Chen Zhu, Xiangyu Zhao, and Hengshu Zhu. Swarm intelligence in geo-localization: A multi-agent large vision-language model collaborative framework, Oct. 2024.
- [32] Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford, and Tobias Fischer. Patch-netvlad: Multi-scale fusion of locally-global descriptors for place recognition, Mar. 2021.
- [33] James Hays and Alexei A. Efros. Im2gps: Estimating geographic information from a single image. In 2008 IEEE Conference on Computer Vision and Pattern Recognition, pages 1–8, Anchorage, AK, USA, June 2008. IEEE.

- [34] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan AllenZhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.
- [35] Wenmiao Hu, Yichen Zhang, Yuxuan Liang, Xianjing Han, Yifang Yin, Hannes Kruppa, See-Kiong Ng, and Roger Zimmermann. Petalview: Fine-grained location and orientation extraction of street-view images via cross-view local search. In Proceedings of the 31st ACM International Conference on Multimedia, MM ’23, page 56–66. ACM, Oct. 2023.
- [36] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.
- [37] Sarah Ibrahimi, Nanne van Noord, Tim Alpherts, and Marcel Worring. Inside out visual place recognition, Nov. 2021.
- [38] James Inman. Navigation and nautical astronomy: For the use of British seamen. F. and J. Rivington, 1849.
- [39] International Road Federation (IRF). World Road Statistics Data Warehouse, 2025. Accessed: 2025-11-12.
- [40] Pengyue Jia, Yiding Liu, Xiaopeng Li, Yuhao Wang, Yantong Du, Xiao Han, Xuetao Wei, Shuaiqiang Wang, Dawei Yin, and Xiangyu Zhao. G3: An effective and adaptive framework for worldwide geolocalization using large multimodality models, Oct. 2024.
- [41] Pengyue Jia, Seongheon Park, Song Gao, Xiangyu Zhao, and Yixuan Li. Georanker: Distance-aware ranking for worldwide image geolocalization, May 2025.
- [42] Hyo Jin Kim, Enrique Dunn, and Jan-Michael Frahm. Learned contextual feature reweighting for image geolocalization. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, July 2017. IEEE.
- [43] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022.
- [44] Giorgos Kordopatis-Zilos, Panagiotis Galopoulos, Symeon Papadopoulos, and Ioannis Kompatsiaris. Leveraging efficientnet and contrastive learning for accurate global-scale location estimation, May 2021.
- [45] Martha Larson, Mohammad Soleymani, Guillaume Gravier, Bogdan Ionescu, and Gareth JF Jones. The benchmarking initiative for multimedia evaluation: Mediaeval 2016. IEEE MultiMedia, 24(1):93–96, 2017.
- [46] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.
- [47] Ling Li, Yu Ye, Bingchuan Jiang, and Wei Zeng. Georeasoner: Geo-localization with reasoning in street views using a large vision-language model, June 2024.
- [48] Ling Li, Yao Zhou, Yuxuan Liang, Fugee Tsung, and Jiaheng Wei. Recognition through reasoning: Reinforcing image geo-localization with large vision-language models, June 2025.

- [49] Xiang Li, Congcong Wen, Yuan Hu, and Nan Zhou. Rs-clip: Zero shot remote sensing scene classification via contrastive vision-language supervision. International Journal of Applied Earth Observation and Geoinformation, 124:103497, 2023.
- [50] Yuxi Li. Deep reinforcement learning: An overview. arXiv preprint arXiv:1701.07274, 2017.
- [51] Yujie Li, Wenjia Xu, Guangzuo Li, Zijian Yu, Zhiwei Wei, Jiuniu Wang, and Mugen Peng. Unirs: Unifying multitemporal remote sensing tasks through vision language models. arXiv preprint arXiv:2412.20742, 2024.
- [52] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.
- [53] Liu Liu and Hongdong Li. Lending orientation to neural networks for cross-view geo-localization, Mar. 2019.
- [54] Liu Liu, Hongdong Li, and Yuchao Dai. Stochastic attraction-repulsion embedding for large scale image localization. In 2019 IEEE/CVF International Conference on Computer Vision (ICCV), pages 2570–2579, Seoul, Korea (South), Oct. 2019. IEEE.
- [55] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.
- [56] Mojgan Madadikhaljan and Michael Schmitt. Geolocationaware deep coding: Infusing geolocation information into deep neural networks for remote sensing. PFG–Journal of Photogrammetry, Remote Sensing and Geoinformation Science, 93(1):3–18, 2025.
- [57] Mohammad Sultan Mahmud, Joshua Zhexue Huang, Salman Salloum, Tamer Z Emara, and Kuanishbay Sadatdiynov. A survey of data partitioning and sampling methods to support big data analysis. Big Data Mining and Analytics, 3(2):85– 101, 2020.
- [58] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems, 26, 2013.
- [59] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. nature, 518(7540):529–533, 2015.
- [60] OpenAI. Gpt-4o system card, 2024.
- [61] OpenAI. Gpt-4.1, 2025. Accessed: 2025-05.
- [62] OpenAI. Introducing GPT-5, 2025. Accessed: 2025-11-07.
- [63] Chao Pang, Xingxing Weng, Jiang Wu, Jiayu Li, Yi Liu, Jiaxing Sun, Weijia Li, Shuai Wang, Litong Feng, Gui-Song Xia, et al. Vhm: Versatile and honest vision language model for remote sensing image analysis. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 6381–6388, 2025.
- [64] Guohao Peng, Yufeng Yue, Jun Zhang, Zhenyu Wu, Xiaoyu Tang, and Danwei Wang. Semantic reinforced attention learning for visual place recognition, Aug. 2021.
- [65] Shraman Pramanick, Ewa M. Nowara, Joshua Gleason, Carlos D. Castillo, and Rama Chellappa. Where in the world is this image? transformer-based geo-localization in the wild, July 2022.

- [66] Filip Radenovi´c, Giorgos Tolias, and Ondˇrej Chum. Finetuning cnn image retrieval with no human annotation, July 2018.
- [67] Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, and Yuxiong He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16. IEEE, 2020.
- [68] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, pages 3505–3506, 2020.
- [69] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084, 2019.
- [70] Nils Reimers and Iryna Gurevych. paraphrase-multilingualMiniLM-L12-v2: Multilingual Sentence Transformer Model, 2020. Accessed: 2025-11-12.
- [71] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.
- [72] Paul Hongsuck Seo, Tobias Weyand, Jack Sim, and Bohyung Han. Cplanet: Enhancing image geolocalization by combinatorial partitioning of maps, Aug. 2018.
- [73] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.
- [74] Zirui Song, Jingpu Yang, Yuan Huang, Jonathan Tonglet, Zeyu Zhang, Tao Cheng, Meng Fang, Iryna Gurevych, and Xiuying Chen. Geolocation with real human gameplay data: A large-scale dataset and human-like reasoning framework, Apr. 2025.
- [75] Boyuan Sun, Jiaxing Zhao, Xihan Wei, and Qibin Hou. Llava-scissor: Token compression with semantic connected components for video llms. arXiv preprint arXiv:2506.21862, 2025.
- [76] Hamed Taherdoost. Sampling methods in research methodology; how to choose a sampling technique for research. International journal of academic research in management (IJARM), 5, 2016.
- [77] Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.
- [78] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.
- [79] Kimi Team, Angang Du, Bohong Yin, Bowei Xing, Bowen Qu, Bowen Wang, Cheng Chen, Chenlin Zhang, Chenzhuang Du, Chu Wei, et al. Kimi-vl technical report. arXiv preprint arXiv:2504.07491, 2025.
- [80] Bart Thomee, David A. Shamma, Gerald Friedland, Benjamin Elizalde, Karl Ni, Douglas Poland, Damian Borth, and

- Li-Jia Li. Yfcc100m: The new data in multimedia research. Communications of the ACM, 59(2):64–73, Jan. 2016.
- [81] Akihiko Torii, Relja Arandjelovic, Josef Sivic, Masatoshi Okutomi, and Tomas Pajdla. 24/7 place recognition by view synthesis. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1808–1817, 2015.
- [82] Akihiko Torii, Josef Sivic, Tomas Pajdla, and Masatoshi Okutomi. Visual place recognition with repetitive structures. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 883–890, 2013.
- [83] Nam Vo, Nathan Jacobs, and James Hays. Revisiting im2gps in the deep learning era, May 2017.
- [84] Chun Wang, Xiaoran Pan, Zihao Pan, Haofan Wang, and Yiren Song. Gre suite: Geo-localization inference via fine-tuned vision-language models and enhanced reasoning chains, Sept. 2025.
- [85] Qixuan Wang, Ning Li, Yiheng Chen, and Hainiu Zhu. Multitrans-lc: Multimodal fusion transformer for remote sensing land cover classification. The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, 48:133–138, 2025.
- [86] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022.
- [87] Frederik Warburg, Soren Hauberg, Manuel LopezAntequera, Pau Gargallo, Yubin Kuang, and Javier Civera. Mapillary street-level sequences: A dataset for lifelong place recognition. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2623–2632, Seattle, WA, USA, June 2020. IEEE.
- [88] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.
- [89] Tobias Weyand, Ilya Kostrikov, and James Philbin. Planet photo geolocation with convolutional neural networks. pages 37–55. 2016.
- [90] Scott Workman, Richard Souvenir, and Nathan Jacobs. Wide-area image geolocalization with aerial reference imagery, Oct. 2015.
- [91] Rihui Xin, Han Liu, Zecheng Wang, Yupeng Zhang, Dianbo Sui, Xiaolin Hu, and Bingning Wang. Surrogate signals from format and length: Reinforcement learning for solving mathematical problems without ground truth answers. arXiv preprint arXiv:2505.19439, 2025.
- [92] Chenhui Xu, Fuxun Yu, Michael J Bianco, Jacob Kovarskiy, Raphael Tang, Qi Zhang, Zirui Xu, Will LeVine, Brandon Dubbs, Heming Liao, et al. Geo-r1: Unlocking vlm geospatial reasoning with cross-view reinforcement learning. arXiv preprint arXiv:2510.00072, 2025.
- [93] Shixiong Xu, Chenghao Zhang, Lubin Fan, Gaofeng Meng, Shiming Xiang, and Jieping Ye. Addressclip: Empowering vision-language models for city-wide image address localization, July 2024.

- [94] Shixiong Xu, Chenghao Zhang, Lubin Fan, Yuan Zhou, Bin Fan, Shiming Xiang, Gaofeng Meng, and Jieping Ye. Addressvlm: Cross-view alignment tuning for image address localization using large vision-language models, 2025.
- [95] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.
- [96] Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig, and Xiang Yue. Demystifying long chain-of-thought reasoning in llms. arXiv preprint arXiv:2502.03373, 2025.
- [97] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025.
- [98] Xiaohan Zhang, Xingyu Li, Waqas Sultani, Yi Zhou, and Safwan Wshah. Cross-view geo-localization via learning disentangled geometric layout correspondence, June 2023.
- [99] Zilun Zhang, Zian Guan, Tiancheng Zhao, Haozhan Shen, Tianyu Li, Yuxiang Cai, Zhonggen Su, Zhaojun Liu, Jianwei Yin, and Xiang Li. Geo-r1: Improving few-shot geospatial referring expression understanding with reinforcement finetuning. arXiv preprint arXiv:2509.21976, 2025.
- [100] Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-ofthought reasoning in language models. arXiv preprint arXiv:2302.00923, 2023.
- [101] Zilun Zhang, Tiancheng Zhao, Yulong Guo, and Jianwei Yin. Rs5m and georsclip: A large scale vision-language dataset and a large vision-language model for remote sensing. IEEE Transactions on Geoscience and Remote Sensing, 2024.
- [102] Jiaxing Zhao, Qize Yang, Yixing Peng, Detao Bai, Shimin Yao, Boyuan Sun, Xiang Chen, Shenghao Fu, Xihan Wei, Liefeng Bo, et al. Humanomni: A large vision-speech language model for human-centric video understanding. arXiv preprint arXiv:2501.15111, 2025.
- [103] Zhongliang Zhou, Jielu Zhang, Zihan Guan, Mengxuan Hu, Ni Lao, Lan Mu, Sheng Li, and Gengchen Mai. Img2loc: Revisiting image geolocalization using multi-modality foundation models and image-based retrieval-augmented generation. pages 2749–2754, July 2024.
- [104] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023.
- [105] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

