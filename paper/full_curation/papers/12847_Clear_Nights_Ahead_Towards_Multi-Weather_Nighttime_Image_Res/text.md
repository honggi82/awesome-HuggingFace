# arXiv:2505.16479v2[cs.CV]12Nov2025

## Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration

### Yuetong Liu1,3, Yunqiu Xu4,*, Yang Wei2,3, Xiuli Bi2,3, Bin Xiao2,3,5,∗

1School of Computer Science and Technology, Chongqing University of Posts and Telecommunications 2School of Artificial Intelligence, Chongqing University of Posts and Telecommunications 3Chongqing Key Laboratory of Image Cognition, Chongqing University of Posts and Telecommunications 4ReLER Lab, CCAI, Zhejiang University 5Jinan Inspur Data Technology Co., Ltd. d230201022@stu.cqupt.edu.cn, imyunqiuxu@gmail.com, {weiyang, bixl, xiaobin}@cqupt.edu.cn

###### Abstract

Uneven-Illumination Nighttime Inputs

Multiple Weather Nighttime Inputs

[Figure 1]

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

| |
|---|

| |
|---|

Restoring nighttime images affected by multiple adverse weather conditions is a practical yet under-explored research problem, as multiple weather degradations usually coexist in the real world alongside various lighting effects at night. This paper first explores the challenging multi-weather nighttime image restoration task, where various types of weather degradations are intertwined with flare effects. To support the research, we contribute the AllWeatherNight dataset, featuring large-scale nighttime images with diverse compositional degradations. By employing illumination-aware degradation generation, our dataset significantly enhances the realism of synthetic degradations in nighttime scenes, providing a more reliable benchmark for model training and evaluation. Additionally, we propose ClearNight, a unified nighttime image restoration framework, which effectively removes complex degradations in one go. Specifically, ClearNight extracts Retinex-based dual priors and explicitly guides the network to focus on uneven illumination regions and intrinsic texture contents respectively, thereby enhancing restoration effectiveness in nighttime scenarios. Moreover, to more effectively model the common and unique characteristics of multiple weather degradations, ClearNight performs weatheraware dynamic specificity and commonality collaboration that adaptively allocates optimal sub-networks associated with specific weather types. Comprehensive experiments on both synthetic and real-world images demonstrate the necessity of the AllWeatherNight dataset and the superior performance of ClearNight.

| |
|---|

| |
|---|

| |
|---|

| |
|---|

Daytime Model Nighttime Task-Specific Model

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

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

Doubtful Outputs with Unnatural Lighting

Doubtful Outputs with Residual Distortions

Uneven-Illumination Multiple Weather Nighttime Inputs

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

Nighttime Unified Model ( ClearNight )

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Credible Outputs with Clear Details and Natural Lighting

Figure 1: ClearNight is the first multi-weather nighttime image restoration framework, which effectively removes complex and coupled weather and flare degradations in one go.

lighting conditions (e.g., flare effects) and their intricate interplay with weather degradations, limiting the effectiveness in real-world applications.

In particular, nighttime scenes exacerbate such problem, where adverse weather degradations and uneven illumination are tightly coupled, severely obscuring background contents. Although a few works (Zhang et al. 2020; Liu et al. 2023a; Zhang et al. 2023a) have explored nighttime adverse weather image restoration, they seldom take into account these unique characteristics. More critically, existing methods are limited to handling a single type of degradations, which leads to unsatisfactory performance in complex real-world scenarios where multiple adverse weather conditions frequently co-occur (see Fig. 1). To better meet the demands of real-world applications, this paper first explores a highly practical yet remains largely under-explored task: multi-weather nighttime image restoration. In this context, two critical challenges must be addressed: ❶ the scarcity of realistic multi-weather training samples and ❷ the insufficient ability of existing models to effectively address entangled degradations in nighttime scenes.

Project Page: — https://henlyta.github.io/ClearNight/

### Introduction

Image restoration under adverse weather conditions is a vital preprocessing step in many computer vision applications, such as autonomous driving (Yu et al. 2020; Quan et al. 2021b; Wang et al. 2024; Xu, Zhu, and Yang 2025; Xu et al. 2022) and video surveillance (Liu et al. 2023c; Li et al. 2024, 2025; Wu et al. 2026, 2025b). Although significant efforts have been made to tackle the issues posed by adverse weather images, prior works (Sun et al. 2024a; Yang et al. 2024; Yue et al. 2024) often neglect the complexities of

*Corresponding authors. Copyright © 2026, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

To the best of our knowledge, there is no existing dataset

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

|Datasets<br><br>|H RS RD S F<br><br>|Syn Real<br><br>|Night|
|---|---|---|---|
|Outdoor-Rain RainDS BID-II WeatherStream CDD-11<br><br>[Figure 29]|✔ ✔ ✘ ✘ ✘<br><br>✘ ✔ ✔ ✘ ✘<br><br>✔ ✔ ✔ ✔ ✘<br>✔ ✔ ✔ ✔ ✘<br>✔ ✔ ✘ ✔ ✘<br><br><br>Real-World Adverse Weathe|✔ ✘<br><br>✔ ✔<br><br>✔ ✘<br><br>✘ ✔<br><br>✔ ✘<br><br><br>r Nighttime Images<br><br>[Figure 30]|✘ ✘ ✘ ✘ ✘<br><br>[Figure 31]|
|Raindrop Clarity Night/YellowHaze NHC/NHM/NHR GTA5 UNREAL-NH GTAV-NightRain RVSD AllWeatherNight<br><br>[Figure 32]<br><br>[Figure 33]|✘ ✘ ✔ ✘ ✘<br><br>✔ ✘ ✘ ✘ ✘<br>✔ ✘ ✘ ✘ ✘<br>✔ ✘ ✘ ✘ ✘<br>✔ ✘ ✘ ✘ ✔<br><br>✘ ✔ ✘ ✘ ✘ ✘ ✘ ✘ ✔ ✘<br><br>✔ ✔ ✔ ✔ ✔<br><br><br>Ground-Truths and Synthetic Adverse|✘ ✔<br><br>✔ ✘<br><br>✔ ✔<br><br>✔ ✘<br><br>✔ ✔<br><br>✔ ✘<br><br>✔ ✘<br><br>✔ ✔<br><br><br>[Figure 34]<br><br>[Figure 35]<br><br>[Figure 36]<br><br>[Figure 37]<br><br>Weather Nighttime|✔ ✔ ✔ ✔ ✔ ✔ ✔ ✔<br><br>Images|

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

Real-World Adverse Weather00 Nighttime Images

[Figure 42]

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

Ground-Truths and Synthetic Adverse Weather Nighttime Images

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Figure 2: Real-world and synthetic samples in our AllWeatherNight dataset. The synthetic images effectively simulate real-world nighttime scenes with various degradations.

Table 1: Comparison to related adverse weather datasets. H, RS, RD and S indicate haze, rain streak, raindrop and snow, respectively. F denotes the presence of synthesized flare images. Syn denotes synthesized images. Night denotes the presence of nighttime data.

[Figure 57]

[Figure 58]

[Figure 59]

structor, which identifies weather degradations and implicitly guides their dynamic allocation. Thus, the dynamic allocator becomes weather-aware and is encouraged to consistently select appropriate units for the same types of weather conditions.

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

that is applicable to multi-weather nighttime image restoration. As summarized in Tab. 1, all previous datasets (Guo et al. 2024; Jin et al. 2024; Liu et al. 2023a; Zhang et al. 2023a; Chen et al. 2023) overlook nighttime scenarios with multiple weather degradations under non-uniform illumination. To facilitate the research, we construct AllWeatherNight, a dataset for restoring nighttime images affected by multiple adverse weather effects and flares (see Fig. 2). Specifically, we collect diverse nighttime images from multiple sources and design an illumination-aware degradation generation method to faithfully emulate degradations in realworld nighttime photos, yielding images characterized by the realistic intertwined effects of uneven illumination and multiple adverse weather degradations. Consequently, training with our synthesized images enables models to generalize better in real-world nighttime scenarios.

Real-World Adverse Weather Nighttime Images

00

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

Extensive experiments and analysis demonstrate the superiority of our ClearNight on both synthetic and real-world images. The main contributions are summarized as follows:

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

- • This work pioneers multi-weather image restoration in nighttime conditions. We contribute a new dataset featuring 10K realistic illumination-aware synthetic images with multi-degradation alongside real-world samples.
- • We propose ClearNight, the first unified framework for multi-weather nighttime image restoration, which integrates Retinex-based dual prior guidance and weatheraware dynamic specificity-commonality collaboration, tailored to address challenging uneven lighting and diverse weather effects entangled in nighttime scenes.
- • Our ClearNight effectively removes various degradations and outperforms state-of-the-art approaches on both synthetic and real-world adverse weather nighttime images.

Ground-Truths and Synthetic Adverse Weather Nighttime Images

[Figure 78]

[Figure 79]

[Figure 80]

Real-World Adverse Weather Nighttime Images

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

[Figure 85]

[Figure 86]

Ground-Truths and Synthetic Adverse Weather Nighttime Images

To effectively handle the intertwined degradations caused by artificial lighting and multiple adverse weather effects at night, we present ClearNight, the first unified framework tailored for multi-weather nighttime image restoration. We first leverage Retinex-based dual prior guidance to explicit disentangle the lighting and texture information within degraded images. Specifically, the decoupled illumination prior guides the model to focus on the uneven lit regions for effective restoration in nighttime scenes, while the reflectance prior enhances texture representations to mitigate weatherinduced degradations and recover clear background details.

### Related Work

Adverse Weather Nighttime Image Restoration. Early works (Zhang, Cao, and Wang 2014; Zhang et al. 2017) typically relied on physical priors and statistical assumptions, limiting their effectiveness and robustness in handling realworld scenarios. To address the limitations, several datadriven methods have been developed, achieving impressive results in restoring nighttime images degraded by haze (Pei and Lee 2012; Liu et al. 2023a; Cong et al. 2024; Wang, Wang, and Liu 2022; Liu et al. 2023b, 2022; Lin et al. 2025), rain (Zhang et al. 2023a; Lin et al. 2024) and snow (Chen et al. 2023). However, existing works focus on single-type degradations and overlook the fact that real-world nighttime scenes often involve multiple simultaneous degradations caused by adverse weather and uneven lighting.

In order to better represent complex degradations consisting of multiple adverse weather conditions, ClearNight employs a specificity-commonality dual-branched architecture, where the specificity branch is dynamically constructed and synergizes with the commonality branch. The dynamic specificity branch contains diverse sub-networks, formed by selecting different combinations of candidate units (Shazeer et al. 2017; Zhong et al. 2024; Jia et al. 2024; Luo and Yang 2024). To further enhance the capabilities of handling multiweather degradations, we associate these candidate units with specific weather types using an auxiliary weather in-

Multi-Weather Image Restoration. Multi-weather image restoration aims to restore various weather-degraded scenes

Neither Weather Only

Neither Weather Only

Ground-Truth

Ground-Truth

Flare Only Both (ours)

Flare Only Both (ours)

Rain Scene Snow Scene

[Figure 102]

[Figure 104]

Raindrop Snow

[Figure 105]

[Figure 106]

Neither Weather Only

Rain Streak Haze

Flare

Snow Scene

[Figure 107]

Rain Scene

Haze + Snow

100

Real-Raindrop

Real-Rain Streak

[Figure 108]

Ground-Truth

Real-Haze

Real-Snow

[Figure 109]

[Figure 110]

Haze

1000

Haze + Raindrop

Snow

RainSceneSnowScene

Training Data

Flare Only Both (ours)

Raindrop

Haze + Rain Streak

Rain Streak

Figure 3: Visualization of four synthesized variants of complex rain scene, where Weather Only and Flare Only denote synthesis with illumination-aware weather degradation and flare, respectively. Ours involves both degradations.

Rain Streak + Raindrop Haze + Rain Streak + Raindrop

Figure 4: Distribution of the our AllWeatherNight dataset.

2019) as ground truths. These candidate images undergo a two-step process: selecting high-quality nighttime samples using average brightness, average gradient and grayscale variance; and manually selecting the 1,000 diverse images from each dataset. We also collect 1,000 real-world nighttime images with various weather conditions from the Internet and existing datasets (Li et al. 2019; Wang et al. 2019; Liu et al. 2018; Zhang, Sindagi, and Patel 2020; Fu et al. 2017; Jin et al. 2023; Yang et al. 2020).

using a single model (Kulkarni, Phutke, and Murala 2023; Kim et al. 2024; Yang et al. 2024; Ai et al. 2024; Wu et al. 2024; Liu et al. 2024; Wu et al. 2025a; Hu et al. 2025; Dong et al. 2025). Recently, numerous Transformerbased approaches are developed, which investigate weather queries (Valanarasu, Yasarla, and Patel 2022), adverse weather pixel categorization (Sun et al. 2024b), textureguided appearance flow (Wang et al. 2023), codebook prior fusion (Ye et al. 2023), and etc. Moreover, several recent works (Ozdenizci¨ and Legenstein 2023; Chen et al. 2024; Zheng et al. 2024) leverage remarkable generative capabilities of diffusion models for restoration performance boosting. To the best of our knowledge, all the previous methods focus on daytime scenes and ignore the entanglement of multiple weather conditions with non-uniform lighting in nighttime scenes.

Illumination-Aware Degradation Generation. We observe that uneven lighting conditions in real-world nighttime scenes often interact with adverse weather degradations, yet this phenomenon is largely overlooked by prior works (Zhang et al. 2020, 2023a). To synthesize more realistic nighttime images with weather effects, we introduce an illumination-aware degradation generation approach that first modulates the lighting effects and then superimposes an illumination-aware combination of weather degradations.

Adverse Weather Datasets. Most previous datasets, like Outdoor-Rain (Li, Cheong, and Tan 2019), RainDS (Quan et al. 2021a), BID-II (Han et al. 2022), CDD-11 (Guo et al. 2024) and WeatherStream (Zhang et al. 2023b), only focus on adverse weather image restoration in daytime scenarios. A few recent studies explore this task in nighttime scenes and build datasets for haze (e.g. Night/YellowHaze (Liao et al. 2018), NHC/NHM/NHR (Zhang et al. 2020), UNREAL-NH (Liu et al. 2023a)), rain (e.g. GTA5 (Yan, Tan, and Dai 2020), GTAV-NightRain (Zhang et al. 2022, 2023a), Raindrop Clarity (Jin et al. 2024)) and snow (e.g. RVSD (Chen et al. 2023)). However, each existing nighttime dataset only considers a single type of weather condition, ignoring the challenge of mixed weather conditions commonly found in the real world.

Given a clean nighttime image X, we generate flaredegraded image Xflare based on light regions:

Xflare = α · X + β · (L ∗ KAPSF), (1)

where α controls clean image preservation. β regulates flare blending and is adaptively set by the light source pixel ratio. L is a light source map devised via thresholding and alpha matting refinement (Levin, Lischinski, and Weiss 2008). KAPSF is a 2D kernel from the atmospheric point spread function (Jin et al. 2023), and ∗ is the convolution operator.

Subsequently, building upon Xflare, the illuminationaware weather-degraded image Xd is synthesized by:

Xd = Xflare +

ωe · FeG(Xflare), (2)

### AllWeatherNight Dataset

e∈E

Real-world nighttime scenes are often degraded by a complex interplay of co-occurring weather and flares, challenging models for faithful restoration and visibility enhancement. To address this gap, we introduce AllWeatherNight, a new large-scale dataset featuring nighttime images with coupled degradations from mixed weather and uneven lighting. Data Collection and Filtering. We repurpose images from BDD100K (Yu et al. 2020) and ExDark (Loh and Chan

where E is the set of selected weather effects for generating a specific degraded image, which is a subset of the universal set of all possible effects, i.e., E ⊆ {H,RS,RD,S} (Haze, Rain Streak, Raindrop, and Snow). We devise a weight map ωe to better simulate weather degradations under uneven lighting, where ωe̸=RD is set to the Retinex decomposed illumination map, while ωe=RD is empirically set to 1, as raindrops are primarily affected by local background

fa

###### Retinex-Based Dual Prior Extraction

###### Weather-Aware Dynamic Selection Block (WDS)

##### ClearNight

i1 i2 i3

fa

Illumination Prior

[Figure 111]

|[Figure 112]|
|---|

M

Degraded Images

C a

###### CommonalitySpecificity

Reconstruction

×4

×8

UnEmbedding

rn

Embedding

×4

×8

×8

Retinex Decomposition

Patch

Patch

[Figure 113]

|[Figure 114]|
|---|

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

Illumination Reflectance

###### Dynamic Selection Module

|[Figure 121]|
|---|

[Figure 122]

Knowledge-Sharing Unit

| | |
|---|---|
| | |

fa′

fa

fa

Degraded Images

Restored Images

###### Multi-Scale Prior Extraction Unit

###### Weather Instructor Dynamic Allocator

###### Weather-Aware Dynamic Specificity and Commonality Collaboration

Dilated Conv

Haze

Rain Streak

###### Gating

Down-Sample

fa

Retinex-Based Dual Prior Extraction

Up-Sample

###### C

f2 f3 f4 f5

Element-Wise Addition

Conv Conv Conv Conv

{w}

Weighted Sum

f1

Ma

###### Guidance

C Concatenation

Candidate Units

Forward

Illumination Prior

Raindrop

Snow

Supervision

r1 r2 r3

r1 r2 r3

i1 i2 i3

TFB Transformer Blocks

bce lb

Reflectance Prior

Reflectance Prior

Reflectance Prior

Illumination Prior

- Figure 5: Overview of our ClearNight framework. ClearNight primarily comprises Retinex-based dual prior guidance as well as weather-aware dynamic specificity and commonality branches. The Retinex-based dual priors explicitly guide the network to focus on illumination regions and intrinsic textures. The weather-aware dynamic specificity branch adaptively accommodates various weather effects and collaborates with the commonality branch to effectively handle complex multi-weather scenes.

Guidance

rather than distant lighting. FeG(·) represents corresponding weather generation functions. We simulate adverse weather using established models (Li, Cheong, and Tan 2019; Soboleva and Shipitko 2021; Liu et al. 2018; Chen et al. 2021). As shown in Fig. 3, unlike simplistic, spatially uniform synthesis, illumination-aware method generates more natural degradations.

Dataset Statistics. Overall, we synthesize 8,000 nighttime training images, covering multiple weather with varying scales, directions, patterns and intensities. As summarized in Fig. 4, the test dataset comprises synthetic and real-world subsets, each containing 1,000 images for model evaluation.

### ClearNight

ClearNight is a unified nighttime image restoration framework designed to simultaneously remove multiple weather degradations. As depicted in Fig. 5, ClearNight integrates Retinex-based dual prior guidance and weather-aware dynamic specificity-commonality collaboration. The former explicitly decouples uneven lighting and textures to guide the network in recovering clear images with natural lighting and rich background details. The latter effectively captures the unique and shared characteristics of diverse weather conditions, enabling powerful multi-weather image restoration.

#### Retinex-Based Dual Prior Guidance

Decoupling illumination and texture information is critical for nighttime image restoration, as it allows the model to separately handle non-uniform lighting and adverse weather degradations. Retinex theory (Edwin 1977) provides a classic physical model for this decomposition, formulating the input degraded image Xd as:

Xd = Rd · Id, (3)

where Rd and Id are the reflectance and illumination components, respectively. The decomposed components prtx ∈

{Id,Rd} are then fed into shared-weight multi-scale prior extraction units (MPE):

p1,p2,p3 = FMPE(prtx) with pn ∈ {in,rn}, (4)

where pn denotes the illumination or reflectance prior at the n-th scale, with n ∈ {1,2,3}. Within each MPE unit, a dilated convolution first projects the prtx into three scales. Then, resulting multi-scale features are interactively fused to produce the final illumination/reflectance priors pn.

More specifically, the illumination priors in are successively injected into the Transformer blocks (TFBs) of the first three stages, which guide the network to focus on the uneven lighting regions in nighttime images, thereby facilitating the handling of lighting-influenced weather degradations. As the reflectance priors rn contain rich intrinsic textures that not only capture background details but also reveal degradation types, we incorporate them into each weatheraware dynamic selection block (WDS) to enhance weather type discrimination and improve multi-weather restoration.

#### Weather-Aware Dynamic Specificity-Commonality Collaboration

Dynamic Specificity and Commonality Synergy. As different weather degradations (e.g., snow and rain streaks) exhibit both shared and unique patterns, we introduce a synergistic design of dynamic specificity and commonality to effectively model complex multi-weather degradations. The commonality branch consists of sequential Transformer blocks (TFBs), while the specificity branch incorporates multiple weather-aware dynamic selection blocks (WDS) as residuals for each stage of TFBs.

As shown in the right of Fig. 5, each WDS consists of an encoder-decoder structure and a dynamic selection module, which jointly process the merged features from two branches and the reflectance prior. The encoder-decoder structure, in-

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

- Figure 6: Comparison of 4 synthesized variants of nighttime images, where W and F indicate synthesis with illuminationaware weather degradation and flare effects respectively.

spired by (Zhong et al. 2024), learns compact representations, while the dynamic selection module constructs inputtailored sub-networks by sparsely selecting candidate units (FkU(·)), where k is the unit index.

Besides the candidate units, the dynamic selection module includes a gating (FW(·)) that computes the importance weights assigned to each candidate unit, and a router (FR(·)) that estimates the probability of each unit being selected. Given the input feature f¯a, the output of the a-th dynamic selection module can be calculated by

FW(f¯a)k · FkU(f¯a), (5)

M¯a =

k∈T

where T = TopK(FR(f¯a)) denotes the set of selected topK candidate unit indices, and FW(f¯a)k is the importance weight for the k-th unit.

Dynamic Weather Degradation Modeling. Relying solely on visual content would limit the ability to capture the correlations and distinctive characteristics of different weather types, thereby hindering the module’s effectiveness in complex multi-weather scenarios. To better associate distinct weather types with designated candidate units, we reorganize the dynamic selection module and introduce a new component, the weather instructor (FWI(·)).

In the new design, features first pass through a knowledge-sharing unit (FKSU(·)), which is composed of shared linear layers preceding the dynamic allocator. The output of the knowledge-sharing unit is then utilized by the weather instructor, which classifies degradations and learns weather-specific prototypes to aggregate features from the same weather conditions:

ya = FWI(f¯a) with f¯a = FKSU(fa′), (6)

where fa′ is the output of the encoder, and ya is the weather type predicted by the a-th dynamic selection module.

The weather instructor performs multi-label classification

using binary cross-entropy loss Lbce, guiding the prototypes to attract features of the corresponding weather.

#### Network Optimization

Multiple losses are utilized to jointly optimize ClearNight, where L1 loss L1 and perceptual loss Lp (Johnson, Alahi, and Li 2016) are used to ensure that restored results resemble ground-truths closely. To enhance the structure and details of outputs, depth loss Ld is exploited via minimizing

5.5

| | | | | | |
|---|---|---|---|---|---|
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |
| | | | | | |

5.0

4.5

4.0

3.5

3.0

Figure 7: Models trained on AllWeatherNight achieve superior NIQE↓ results on real-world images compared to those trained on the combination of existing nighttime datasets.

the differences between ground-truths and predictions of a pre-trained depth estimation model (Liu et al. 2021). In addition, we use a load balancing loss Llb (Shazeer et al. 2017) to balance the utilization of the candidate units. We jointly optimize the network using the total loss function:

Ltotal = L1 + λpLp + λbceLbce + λlbLlb + λdLd, (7) where λp, λbce, λlb and λd are the loss weights.

### Experiments

#### Implementation Details

The synthetic image size is 640 × 360. During training, the input image is randomly cropped to 256 × 256. We adopt DehazeFormer (Song et al. 2023) as our baseline for its advanced spatial aggregation, allowing effective natural lighting restoration. We use Adam as optimizer and the initial learning rate is set to 2 × 10−4 for 100 epochs. The learning rate is adjusted using the cosine annealing scheme. The loss weight λp, λbce, λlb and λd are empirically set to 0.1, 0.001, 0.01 and 0.02, respectively. α is set to 0.995.

#### Dataset Analysis

To demonstrate the effectiveness of illumination-aware degradation generation, we analyze four different synthetic variants of nighttime images (showcased in Fig. 3). Ten volunteers are recruited to rank image quality on authenticity, lighting complexity and task relevance. As shown in the left of Fig. 6, we can observe that the combination of illumination-aware weather and flare achieves the best human preference. Furthermore, the right part of Fig. 6 demonstrates that training with our illumination-aware data improves performance in real-world scenes, with four representation models (Zhu et al. 2023; Choi et al. 2024; Zheng et al. 2024) exhibiting consistent improvements.

To further validate the proposed dataset, we compare the performance of models trained on our AllWeatherNight dataset with those trained on a composite dataset constructed from existing nighttime datasets, as shown in Tab. 1. As shown in Fig. 7, models trained on AllWeatherNight exhibit superior performance on real-world samples, which is attributable to our illumination-aware synthetic images simulating degradations more realistically.

#### Comparison with the State-of-the-Art

Results on Synthetic Data. We compare ClearNight with state-of-the-art adverse weather image restoration methods

[Figure 123]

[Figure 124]

[Figure 125]

[Figure 126]

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

[Figure 131]

[Figure 132]

[Figure 133]

- (a)SyntheticTestingSubset

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

AWRaCLe

Inputs TKL FSDGN WeatherDiff WGWS RLP RAMiT DiffUIR DEA-Net ClearNight (ours)

[Figure 138]

[Figure 139]

[Figure 140]

[Figure 141]

[Figure 142]

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

[Figure 147]

[Figure 148]

[Figure 149]

[Figure 150]

[Figure 151]

[Figure 152]

[Figure 153]

[Figure 154]

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

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

[Figure 172]

[Figure 173]

[Figure 174]

[Figure 175]

[Figure 176]

[Figure 177]

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

- (b)Real-WorldTestingSubset

[Figure 198]

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

[Figure 203]

[Figure 204]

[Figure 205]

[Figure 206]

[Figure 207]

[Figure 208]

[Figure 209]

[Figure 210]

[Figure 211]

[Figure 212]

[Figure 213]

[Figure 214]

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

[Figure 219]

[Figure 220]

[Figure 221]

[Figure 222]

[Figure 223]

[Figure 224]

[Figure 225]

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

Inputs Ground-Truths TKL FSDGN WeatherDiff WGWS RLP RAMiT DiffUIR DEA-Net ClearNight (ours)

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

[Figure 236]

AWRaCLe

Figure 8: Qualitative results on AllWeatherNight synthetic and real-world testing subset. Please zoom in for more details.

|Method<br><br>|Multi-Degradation Rain Scene Snow Scene PSNR↑ / SSIM↑ PSNR↑ / SSIM↑<br><br>|Single-Degradation Haze Rain Streak Raindrop Snow Flare PSNR↑ / SSIM↑ PSNR↑ / SSIM↑ PSNR↑ / SSIM↑ PSNR↑ / SSIM↑ PSNR↑ / SSIM↑<br><br>|
|---|---|---|
|TKL FSDGN WeatherDiff WGWS RLP RAMiT DiffUIR DEA-Net AWRaCLe<br><br>|29.0919 / 0.8769 26.2657 / 0.8456<br><br>29.9850 / 0.8730 28.0147 / 0.8474<br><br>29.7631 / 0.8936 27.1109 / 0.8537<br>30.5811 / 0.8961 27.9023 / 0.8658 21.2180 / 0.6641 19.3059 / 0.6124<br><br><br>30.4565 / 0.9106 29.1169 / 0.8889 27.6676 / 0.8040 25.8151 / 0.7892<br>31.4244 / 0.9202 29.2500 /<br><br><br>|0.8956|
|---|
<br><br>|31.5392|
|---|
<br><br>/<br><br>|0.9210|
|---|
<br><br>|29.4270|
|---|
<br><br>/ 0.8738|31.6136 / 0.9401 30.4227 / 0.8844 32.5573 / 0.9561 31.0471 / 0.9247 36.7239 / 0.9741<br><br>34.1807 / 0.9378 31.3309 / 0.8851 33.9535 / 0.9638 31.9022 / 0.9288 38.7223 / 0.9798<br><br>30.5020 / 0.9256<br><br>|33.3630|
|---|
<br><br>/ 0.9352 35.9385 / 0.9726 33.6573 / 0.9488<br><br>|40.2057|
|---|
<br><br>/ 0.9826<br><br>31.6132 / 0.9181 32.8077 / 0.9221 34.3616 / 0.9554 32.1325 / 0.9196 32.1085 / 0.9017<br><br><br>18.6348 / 0.6379 31.1586 / 0.8867 32.4204 / 0.9322 31.5645 / 0.9101 34.7658 / 0.9451 / 0.9738 31.9433 / 0.9204 33.8452 / 0.9632 32.8934 / 0.9491 43.0080 / 0.9934 28.4763 / 0.8341 30.2547 / 0.8624 29.6778 / 0.8740 26.9833 / 0.8055 31.2789 / 0.8816<br><br>|36.4414|
|---|
<br><br>35.8174 / 0.9612 32.7631 / 0.9285 34.8406 / 0.9704 33.5811 / 0.9493 38.6533 / 0.9807<br><br>36.4315 / 0.9599 33.1078 / 0.9317 35.2985 / 0.9686<br><br><br>|33.6716|
|---|
<br><br>/<br><br>|0.9532|
|---|
<br><br>40.1014 / 0.9836|
|Baseline ClearNight<br><br>|28.7976 / 0.8825 27.1337 / 0.8452 32.5937 / 0.9223 30.6464 / 0.9041<br><br>|30.2905 / 0.9257 30.5615 / 0.8994 33.0758 / 0.9598 31.3318 / 0.9182 36.0821 / 0.9738 36.4655 /<br><br>|0.9621|
|---|
<br><br>33.6238 /<br><br>|0.9331|
|---|
<br><br>|35.4282|
|---|
<br><br>/<br><br>|0.9723|
|---|
<br><br>33.9747 / 0.9539 38.7707 /<br><br>|0.9838|
|---|
|

- Table 2: Quantitative results on AllWeatherNight synthetic testing subset. The best and

results are highlighted.

|second-best|
|---|

on AllWeatherNight, including TKL (Chen et al. 2022), FSDGN (Yu et al. 2022), RLP (Zhang et al. 2023a), WeatherDiff (Ozdenizci¨ and Legenstein 2023), WGWS (Zhu et al.

- 2023), RAMiT (Choi et al. 2024), DiffUIR (Zheng et al.
- 2024), DEA-Net (Chen, He, and Lu 2024) and AWRaCLe (Rajagopalan and Patel 2025). As shown in Fig. 8a, TKL (Chen et al. 2022), FSDGN (Yu et al. 2022) and DiffUIR (Zheng et al. 2024) produce overly smooth results such as the clock and railing regions, while other approaches often lose structural details or leave residual artifacts. In contrast, ClearNight preserves rich background details and restores natural lighting, illustrating robust performance across diverse and complex nighttime scenarios.

Tab. 2 reports quantitative results on multi-degradation and single-degradation scenes, evaluated using PSNR (Hor´e and Ziou 2010) and SSIM (Zhou et al. 2004) for synthetic samples. Among these methods, FSDGN, RLP and DEANet are tailored for daytime/nighttime task-specific restoration, but struggle in other task scenes due to limited weather

feature extraction. In contrast, our method targets robust nighttime weather effects removal. As the training data lack dedicated flare samples, its performance in flare removal remains moderate. Nevertheless, ClearNight achieves the best results on the multi-degradation subset and delivers competitive performance on single-degradation samples.

Results on Real-World Data. As shown in Fig. 8b, ClearNight effectively removes most weather effects on real images and mitigates flares, producing more natural results compared to state-of-the-art methods. Furthermore, we use NIQE (Mittal, Soundararajan, and Bovik 2013) to assess the restored images in Tab. 3. The experimental results demonstrate that ClearNight can predict highly realistic images under various adverse weather conditions.

Comparison with Cascade Solutions. As demonstrated in Fig. 9, we compare ClearNight against two pre-trained taskspecific models (Lin et al. 2025; Zhang et al. 2023a) on realworld nighttime rain streak images from AllWeatherNight. ClearNight not only achieves superior results visually and

|Method<br><br>|Haze Rain Streak Raindrop Snow<br><br>|
|---|---|
|TKL FSDGN WeatherDiff WGWS RLP RAMiT DiffUIR DEA-Net AWRaCLe<br><br>|4.1872 3.7765 4.2780 4.4694 4.9149 4.1528 4.1964 3.7842 3.9254 3.3451 4.1879<br><br>|3.9238|
|---|
<br><br>|3.2680|
|---|
<br><br>|3.7732|
|---|
<br><br>3.9635 3.2769 4.9699 4.1882 5.6240 4.7669 4.1655 3.9298 4.0808 3.2497 4.1063 3.8728 4.0471 3.3547 4.1665 3.9826 4.0889 3.4201 4.1681 3.9516 4.0936 3.3398|
|Baseline ClearNight<br><br>|4.2054 3.9983 4.0778 3.4605 3.7653 3.8882 3.2191<br><br>|4.1623|
|---|
|

- Table 3: Quantitative results on AllWeatherNight real-world testing subset, evaluated using the commonly used NIQE↓.

|#|IP DA RP WI<br><br>|PSNR↑ / SSIM↑<br><br>|
|---|---|---|
|1<br><br>2<br><br>3<br><br>4<br><br>5<br><br>6<br><br>7<br><br><br>|✘ ✘ ✘ ✘<br><br>✔ ✘ ✘ ✘<br><br>✘ ✔ ✘ ✘<br><br>✔ ✔ ✘ ✘<br><br>✘ ✔ ✔ ✔<br><br>✔ ✔ ✔ ✘<br>✔ ✔ ✔ ✔<br>|28.7976 / 0.8825 32.1304 / 0.9176<br><br>31.7075 / 0.9113<br><br>32.2393 / 0.9179<br><br><br>32.4430 / 0.9214 32.2528 / 0.9184 32.5937 / 0.9223<br><br>|

- Table 4: Ablation study of key component. IP and RP denote illumination and reflectance priors. DA and WI indicate dynamic allocator and weather instructor in WDS.

quantitatively, but requires significantly less inference time.

#### Ablation Studies

Ablation on Model Components. We evaluate the effectiveness of the illumination prior, reflectance prior, dynamic allocator and weather instructor on the Rain Scene testing subset. As shown in Tab. 4, dual priors significantly enhance the restoration performance over the baseline, while the dynamic allocator, guided by the weather instructor, optimizes candidate unit selection to handle multi-weather scenes. Quantitative results demonstrate that the integration of all components achieves the best performance.

Effectiveness of WDS. Fig. 10 illustrates the relationships between various degradations and selected units, demonstrating the effectiveness of WDS in capturing the specificity of different degradations. The WDS differentiates weather features, selecting similar unit combinations for degraded scenes with the same weather, enabling the network to efficiently eliminate diverse distortions. We visualize the feature f¯a of different weather types in Fig. 11. Despite the visual similarity between rain and snow, ClearNight still differentiates them. Notably, the features of “H + S” lie between haze and snow features, indicating that our model learns the correlations among multiple weather effects in complex nighttime scenarios.

### Conclusion

This paper explores a practical yet under-explored task, i.e., multi-weather nighttime image restoration. To facilitate this task, we introduce an illumination-aware degradation gener-

[Figure 237]

[Figure 238]

[Figure 239]

NIQE↓: 3.7653 Time: 0.5175s

NIQE↓: 5.8596 Time: 37.1206s

|[Figure 240]|
|---|

|[Figure 241]|
|---|

|[Figure 242]|
|---|

Input ClearNight (ours) Dehazing

[Figure 243]

[Figure 244]

[Figure 245]

NIQE↓: 4.1642 Time: 3.7198s

NIQE↓: 5.8356 Time: 40.8404s

NIQE↓: 6.6005 Time: 40.8404s

|[Figure 246]|
|---|

|[Figure 247]|
|---|

|[Figure 248]|
|---|

Deraining+Dehazing

Deraining Dehazing+Deraining

- Figure 9: ClearNight outperforms a cascade of two latest task-specific nighttime restoration methods. Quantitative results are averaged over all the real-world rain streak images.

[Figure 249]

- Figure 10: Correlation of selected units and diverse degradations. WDS associates various distortions with candidate units.

| | | |
|---|---|---|

| | |
|---|---|
| | |

| | |
|---|---|

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

| |
|---|

| |
|---|

| |
|---|

| |
|---|

| | |
|---|---|

| |
|---|

| |
|---|

80 60 40 20 0 20 40 60 80

80

60

40

20

0

20

40

60

80

Four Different Single-Degradations

- RD - 1

- RD - 2

| |
|---|

- RD - 3

- RD - 4

- RD - 5

- RS - 1

- RS - 2

| |
|---|

- RS - 3

- RS - 4

- RS - 5

- S - 1

- S - 2

| |
|---|

- S - 3

- S - 4

- S - 5

- H - 1

- H - 2

| |
|---|

- H - 3

- H - 4

- H - 5

| |
|---|

| |
|---|

| | | |
|---|---|---|
| | | |

| |
|---|

| |
|---|
| |

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

| | |
|---|---|

| |
|---|

| |
|---|
| |

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

60 40 20 0 20 40 60 80

75

50

25

0

25

50

75

Single- and Multi-Degradations

- H - 1

- H - 2

| |
|---|

- H - 3

- H - 4

- H - 5

- RS - 1

- RS - 2

| |
|---|

- RS - 3

- RS - 4

- RS - 5

- S - 1

- S - 2

| |
|---|

- S - 3

- S - 4

- S - 5

- H+S - 1

- H+S - 2

| |
|---|

- H+S - 3

- H+S - 4

- H+S - 5

- Figure 11: T-SNE visualization of feature distributions across distinct weather. H, RS, RD and S indicate haze, rain streak, raindrop and snow. The number is the index of WDS.

ation approach and construct a new dataset featuring 10K high-quality nighttime images with various compositional adverse weather and lighting conditions. In addition, we propose ClearNight, a unified framework, tailored for the new task, capable of removing multiple degradations in one go. ClearNight leverages Retinex-based dual priors to explicitly guide the network to focus on illumination regions and intrinsic textures respectively. Moreover, ClearNight incorporates weather-aware dynamic specific-commonality collaboration to better capture the characteristics of diverse weather, enabling effective multi-weather degradation removal. Comprehensive experiments on both synthetic and real-world images demonstrate the superiority of our ClearNight.

Clear Nights Ahead: Towards Multi-Weather Nighttime Image Restoration Supplementary Material

虽然我们的方法有效地处理了夜间照明和各种天气退化，但它在极其 动态的照明条件下（如快速变化的人工照明）的性能有待提高。如图 所示，尽管可以去除大部分的天气退化，但快速变化的光照难以去除。 未来的工作可以侧重于提高对这种极端情况的鲁棒性。 AllWeatherNight的原始数据来源于BDD100K\cite{BDD100K}和 Exdark\cite{Exdark}，它们专门关注驾驶场景和物体检测。虽然我们的 数据集包含了各种夜间恶劣天气退化，但其中自然场景比较单一，限 制了其环境多样性。我们考虑扩展AllWeatherNight，以包括更广泛的 现实世界场景，从而进一步增强模型的泛化能力。

Additional Implementation Details

[Figure 250]

[Figure 251]

As detailed in the main text, the model was trained on 256 × 256 randomly cropped patches. For inference, fullresolution images were used. The model was trained with a batch size of 1. To stabilize training, we adopt a cosine annealing warm restarts strategy with a restart period of 50 epochs. We use the Adam optimizer (Kingma and Ba 2015) with β1=0.9 and β2=0.99. The proposed framework is implemented in Pytorch and was trained and tested on an NVIDIA RTX A6000.

| |
|---|

Synthetic Image Restored Result

[Figure 252]

[Figure 253]

The architecture of ClearNight, applies a multi-branch structure (Zhang et al. 2023a; Park, Lee, and Chun 2023; Zhu et al. 2023), integrating illumination and reflectance priors into the backbone. The backbone consists of a commonality branch and a weather-aware dynamic specificity branch. The commonality branch employs five-stage Transformer blocks (Song et al. 2023) with modified normalization layers, activation functions, and spatial feature aggregation to enhance global feature extraction for improved restoration performance. The weather-aware dynamic specificity branch includes five weather-aware dynamic selection blocks (WDS) tailored to adverse weather degradations. After each interaction between unique and common features, we use upsampling and downsampling operations to adjust feature map scales, doubling or halving the width and height as needed.

| |
|---|

| |
|---|

Real-World Image Restored Result

Figure 12: Adverse weather samples and their restored results under rapidly changing lighting conditions.

and Chan 2019), which exclusively focus on driving and object detection. Although our dataset incorporates various nighttime adverse weather degradations, the natural scenes in it are relatively homogeneous, limiting its environmental diversity. We consider expanding the AllWeatherNight to include a wider range of real-world scenarios to further enhance the model’s generalization ability.

### Further Analysis of AllWeatherNight Dataset

### General Discussions

#### Authenticity Assessment

#### Dataset License and Intended Use

To validate the authenticity of the AllWeatherNight dataset, we conduct a statistical analysis of feature distribution, comparing it against existing synthetic adverse weather nighttime datasets and real-world samples. The nighttime synthetic datasets include various haze datasets (UNREAL-NH (Liu et al. 2023a), GTA5 (Yan, Tan, and Dai 2020), NHC/NHR/NHM (Zhang et al. 2020), NightHaze/YellowHaze (Liao et al. 2018)), rain datasets (GTAVNightRain (Zhang et al. 2022), GTAV-NightRain-H (Zhang et al. 2023a)) and snow dataset (RVSD (Chen et al. 2023)). As shown in Fig. 13, the t-SNE visualization reveals that AllWeatherNight’s adverse weather degradation distributions are more closely aligned with real-world nighttime images than any other synthetic nighttime dataset. From left to right, the subfigures depict the feature distributions for haze, rain, and snow, respectively. Notably, AllWeatherNight’s haze distribution closely matches that of realworld nighttime haze, while its rain and snow distributions exhibit the highest similarity to real-world adverse weather samples. These findings demonstrate that AllWeatherNight outperforms existing synthetic datasets in approximating real-world nighttime degradation features, making it highly suitable for training models that generalize to diverse realworld scenarios. Consequently, AllWeatherNight provides a robust foundation for enhancing model performance in multi-weather nighttime image restoration tasks.

The generated images and labels in AllWeatherNight dataset are released under the BSD 3-Clause License, a permissive open-source license that grants users the freedom to use, copy, modify, and distribute the dataset, whether in its original form or as part of derivative works. The ground-truths are released under the BSD 3-Clause License.

The AllWeatherNight dataset is designed to advance research in multi-weather nighttime image restoration. It provides a comprehensive resource for developing, training, and evaluating algorithms to restore nighttime images degraded by diverse adverse weather degradations and nonuniform flare effects. Furthermore, it serves as a standardized benchmark for comparing methods in adverse weather image restoration field.

#### Limitations and Future Works

While our approach effectively handles nighttime illumination and various adverse weather degradations, its performance under extremely dynamic lighting conditions, such as rapidly changing artificial lighting, needs improvement. As shown in Fig. 12, while most weather-induced degradations can be removed, rapidly changing flares still remain. Future work could focus on improving the robustness to such extreme scenarios. The original data for AllWeatherNight are sourced from BDD100K (Yu et al. 2020) and Exdark (Loh

- Figure 13: The t-SNE map of various nighttime adverse weather datasets, including haze, rain and snow degradations. The feature distribution of adverse weather degradations in our AllWeatherNight is closer to real-world nighttime scenes than existing nighttime synthetic datasets.

#### Effectiveness on Real-World Data

To validate the effectiveness of our AllWeatherNight, we train baseline and ClearNight models on our dataset and on a composite dataset, which includes all existing nighttime single weather datasets listed in Tab. 1 of the main text. As shown in Fig. 14, both the baseline and our ClearNight models trained on AllWeatherNight consistently produce highquality results on real-world nighttime scenarios. In contrast, models trained on the composite dataset tend to generate artifacts and often leave adverse weather degradations in the restored results. This underperformance can be attributed to the composite dataset’s fundamental failure to account for the co-occurrence of multiple weather conditions and the complex entanglement of these degradations with uneven lighting, consequently lacking real-world fidelity and realism. Consistent with quantitative results in Fig. 7 of the main text, these visual results demonstrate the superior quality and effectiveness of our AllWeatherNight for multi-weather nighttime image restoration.

#### Additional Visualizations of Illumination-Aware Degradation Generation

Existing synthetic adverse weather nighttime datasets typically model a single type of weather distortion and rarely account for background blurring induced by artificial lighting. In contrast, the proposed AllWeatherNight encompasses a diverse range of degradation, including haze, rain streaks, raindrops, snow, flare, and their combinations. Moreover, we design an illumination-aware degradation generation method to simulate realistic adverse weather and flare effects in nighttime scenes. We synthesized datasets under four settings, with supplemental visualizations provided in Fig. 15. Each type of degradation in AllWeatherNight varies in scales, directions, densities, and styles, ensuring comprehensive coverage. To simulate these degradations, we apply the atmospheric scattering model and interpolated Gaussian noise for haze and rain streaks (Li, Cheong, and Tan 2019), utilize Bezier curves for raindrop synthesis (Soboleva and

Shipitko 2021), and add snow degradation using snowflake masks from (Liu et al. 2018; Chen et al. 2021). The generated flare preserves background clarity in the original image to some extent, enabling the network to effectively restore natural lighting and fine details.

### Further Analysis of ClearNight

#### ClearNight vs. Cascaded Solutions

To further validate the necessity of our unified framework, ClearNight, for restoring complex nighttime images under multiple adverse weather conditions, we present comprehensive analysis comparing our unified model against two pretrained task-specific models (Zhang et al. 2023a; Lin et al. 2025) and their cascaded pipeline. For the cascaded pipeline, we evaluate two configurations: one that applies the dehazing model followed by the deraining model (dehazing + deraining) and the other in the reverse order (deraining + dehazing). As illustrated in Fig. 16, task-specific models fail to remove multiple weather degradations, and their cascaded models struggle to produce satisfactory results. In contrast,

- our ClearNight generates more visually pleasing results and achieves the best average NIQE scores with superior computational efficiency.

Additional Ablation Studies

Influence of Retinex-Based Dual Prior Guidance. To evaluate the impact of multi-scale dual prior, we conduct ablation studies on illumination and reflectance priors at vari-

- ous scales, using # 1 and # 3 (in Tab. 4 of the main text) as baselines, respectively. As shown in Tabs. 5-6, incorporating multi-scale illumination and reflectance priors into the network enables it to focus on uneven lighting and intrinsic texture features of different scales, significantly enhancing nighttime image restoration performance. As the inclusion of illumination components across all five stages brings only marginal gains (PSNR + 0.1628, SSIM + 0.0022), we adopt them only in the first three stages to achieve a better trade-off between performance and model complexity.

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

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

|[Figure 259]|
|---|

|[Figure 260]|
|---|

|[Figure 261]|
|---|

|[Figure 262]|
|---|

|[Figure 263]|
|---|

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

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

|[Figure 269]|
|---|

|[Figure 270]|
|---|

|[Figure 271]|
|---|

|[Figure 272]|
|---|

|[Figure 273]|
|---|

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

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

|[Figure 279]|
|---|

|[Figure 280]|
|---|

|[Figure 281]|
|---|

|[Figure 282]|
|---|

|[Figure 283]|
|---|

Baseline + Existing Datasets

Baseline + AllWeatherNight

ClearNight + Existing Datasets

ClearNight + AllWeatherNight

Input

- Figure 14: Models trained with AllWeatherNight achieve superior performance on real-world images compared to those trained with the combination of existing nighttime adverse weather datasets in Tab. 1.

Table 5: Ablation on Retinex-based Illumination Prior.

Table 6: Ablation on Retinex-based Reflection Prior.

|#|i1 i2 i3<br><br>|PSNR↑ / SSIM↑|
|---|---|---|
|1<br>2<br>3<br>4<br>5<br>6<br>|✘ ✘ ✘<br><br>✔ ✘ ✘<br><br>✘ ✔ ✘ ✘ ✘ ✔<br><br>✔ ✔ ✘<br><br>✔ ✔ ✔<br>|28.7976 / 0.8825 31.6843 / 0.9109 31.6714 / 0.9114 31.6429 / 0.9110<br><br>31.7217 / 0.9115<br><br>32.1304 / 0.9176<br>|

|#|r1 r2 r3<br><br>|PSNR↑ / SSIM↑|
|---|---|---|
|1<br>2<br>3<br>4<br>5<br>6<br>|✘ ✘ ✘<br><br>✔ ✘ ✘<br><br>✘ ✔ ✘ ✘ ✘ ✔<br><br>✔ ✔ ✘<br><br>✔ ✔ ✔<br><br><br>|31.7075 / 0.9113 31.9680 / 0.9198<br><br>31.9374 / 0.9201<br><br>31.8438 / 0.9172<br>32.0335 / 0.9203<br><br><br>32.0594 / 0.9205<br>|

Effectiveness of DSM. We investigate the number of candidate units (B) and top-K selection parameter (K). As shown in Fig. 19, increasing the number of candidate units enhances selective capacity, with an optimal K being critical for superior performance. Notably, performance improves with larger K up to a threshold, beyond which irrelevant information causes model degradation.

tor and weather instructor, respectively, enabling effective auxiliary selection tasks. To evaluate their impact, we conduct experiments on AllWeatherNight, with results visualized in Fig. 17. Evidently, Llb optimizes the allocation of candidate units, ensuring efficient utilization. Without Llb, many units remain underutilized, degrading performance in complex multi-weather scenes. Similarly, applying Lb enhances the correlation between diverse weather conditions and candidate units selection, improving the model’s ability

Analysis of the Loss Functions. Our method employs the loss functions Llb and Lb to supervise the dynamic alloca-

[Figure 284]

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

[Figure 289]

[Figure 290]

[Figure 291]

[Figure 292]

[Figure 293]

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

[Figure 298]

[Figure 299]

[Figure 300]

[Figure 301]

[Figure 302]

[Figure 303]

[Figure 304]

[Figure 305]

[Figure 306]

[Figure 307]

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

Ground-Truths Neither Weather Only Flare Only Both (ours)

- Figure 15: Visualization of 4 different synthesized variants of adverse weather nighttime images, where Weather only and Flare only indicate synthesis with illumination-aware weather degradation and flare degradation respectively. Ours includes both degradation synthesis.

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

to address multiple adverse weather effects. Without Lb, the selected units for different weather distortions tend to converge, impairing multi-weather nighttime image restoration. In contrast, the combined use of Lb and Llb enables precise identification of different adverse weather degradations and balanced utilization of candidate units.

significantly degrades model’s performance, as it fails to distinguish diverse adverse weather degradation features, impairing the guidance for unit selection. When the weight was set to 0.01, the excessively strong constraint of Lb limits its restoration performance on different adverse weather degradations with similar patterns. Conversely, with an excessively low weight of 0.0001, various adverse weather features cannot be effectively distinguished, resulting in poor performance across multiple weather conditions. Weights of 0.001 achieve the best performance. Therefore, we set the λb to 0.001 for our model.

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

As depicted in the bottom-right of Fig. 17, the correlation visualization reveals that distinct units are selected for different adverse weather conditions, with similar sets of units consistently chosen for the same weather type, and all units are effectively utilized, maintaining balanced load distribution. Consequently, this dual-loss collaboration strategy significantly enhances the model’s performance in restoring complex nighttime scenarios with diverse adverse weather conditions and flare effects, achieving high-quality results.

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

#### Inference Efficiency

Fig. 20 compares the average parameters and inference time of our method with recent competitive approaches across all synthetic scenes. While diffusion-based method such as WeatherDiff (Ozdenizci¨ and Legenstein 2023) and DiffUIR (Zheng et al. 2024) achieve impressive performance, they suffer from higher inference time, which limits their practicality in real-world scenes. On the other hand, lightweight models such as RAMiT (Choi et al. 2024)

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

The weighting of the loss function critically influences the performance of the network during training, particularly for multi-weather nighttime scenes with flare effects. To determine an optimal weighting strategy for Lb, we conduct an ablation study on the Rain Scene in AllWeatherNight dataset, with results shown in Fig. 18. The absence of Lb

Ground-Truths Neither Weather only Flare only Both (ours)

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

|[Figure 340]|
|---|

|[Figure 341]|
|---|

|[Figure 342]|
|---|

|[Figure 343]|
|---|

|[Figure 344]|
|---|

|[Figure 345]|
|---|

NIQE↓: 5.8356 Time: 40.8404s

NIQE↓: 6.6005 Time: 40.8404s

NIQE↓: 4.1642 Time: 3.7198s

NIQE↓: 3.7653 Time: 0.5175s

NIQE↓: 5.8596 Time: 37.1206s

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

|[Figure 352]|
|---|

|[Figure 353]|
|---|

|[Figure 354]|
|---|

|[Figure 355]|
|---|

|[Figure 356]|
|---|

|[Figure 357]|
|---|

Input ClearNight (ours) Dehazing Deraining Dehazing+Deraining Deraining+Dehazing

- Figure 16: ClearNight outperforms a cascade of two latest task-specific nighttime adverse weather image restoration methods. Quantitative results are averaged over all the real-world rain streak images.

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

- Figure 17: Correlation between degradation types and candidate units correlation under different loss supervision. Top left: withoutLb and Llb. Top right: with only Lb. Bottom left: with only Llb. Bottom right: with both Lb and Llb.

[Figure 362]

[Figure 367]

and DEA-Net (Chen, He, and Lu 2024) demonstrate fast inference, but often compromise restoration quality. Our ClearNight achieves a good trade-off between performance and efficiency. With only 2.84M parameters and an inference time of 0.32s, it is significantly faster than most diffusionbased models while offering much stronger restoration performance than lightweight models. In summary, ClearNight maintains a relatively low computational cost, making it a practical and effective solution for multi-weather nighttime

image restoration.

#### Additional Qualitative Comparisons

Multi-Degradation. We have provided more visual results of our model on Rain Scene and Snow Scene in AllWeatherNight, as shown in Fig. 21. The Rain and Snow Scene testing subsets contain degraded images with complex multi-weather degradations, effectively simulating realworld nighttime scenes. The visual results demonstrate that

| || |
|---|
| | | | | | |
|---|---|---|---|---|---|---|---|
| || |
|---|
| | | | | | |
| | | | | | | | |
| || |
|---|
| | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| || |
|---|
| | | | | | |
| | | | | | | | |
| || |
|---|
| | | | | | |
| | | | | | | | |

b

Figure 18: Ablation study on λb.

| | | | | | | | |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Figure 19: Ablation study of the number of candidate units B, and selected top-K units. We set B and K to 25 and 10 by default.

ClearNight excels in removing the influence of adverse weather while restoring rich background details. Especially when the background is similar to the adverse weather degradation, ClearNight accurately distinguishes these factors, producing high-quality images.

Single-Degradation. Although multiple adverse weather conditions frequently occur in an intertwined manner, the challenge of restoring scenes under simple degradations remains a crucial and non-trivial aspect of our work. Thus, we evaluate the generalization capability of ClearNight on single-degradation conditions. As shown in Fig. 22, we can see that our method is equally effective in single-degradation scenarios.

Real-World. Nighttime real-world degraded images often contain complex weather interferences and diverse flares. As shown in Fig. 23, we provide more comparison result of realworld scenes, e.g., haze (1st and 2nd rows), rain (3rd and 4th rows), raindrop (5th and 6th rows), snow (7th and 8th rows). The rain and snow scenes often exhibit subtle haze effects, as evidenced by real-world nighttime images, consistent with multi-degradation scenarios in our dataset. The restored results show that ClearNight effectively removes the interference of multiple weather conditions while recovering natural nighttime lighting. Compared to existing adverse weather image restoration approaches, our method is superior in reconstructing scene content and natural lighting conditions.

50

40

30

0.7

0.6

0.5

0.4

0.3

0.2

0.1

24 26 28 30 32 34

Figure 20: Comparison of ClearNight with other methods on AllWeatherNight synthetic testing subset. The marker size reflects the number of model parameters.

### References

Ai, Y.; Huang, H.; Zhou, X.; Wang, J.; and He, R. 2024. Multimodal prompt perceiver: Empower adaptiveness, generalizability and fidelity for all-in-One image restoration. In CVPR.

Chen, H.; Ren, J.; Gu, J.; Wu, H.; Lu, X.; Cai, H.; and Zhu, L. 2023. Snow removal in video: A new dataset and a novel method. In ICCV.

Chen, S.; Ye, T.; Zhang, K.; Xing, Z.; Lin, Y.; and Zhu, L. 2024. Teaching tailored to talent: Adverse weather restoration via prompt pool and depth-anything constraint. In ECCV.

Chen, W.-T.; Fang, H.-Y.; Hsieh, C.-L.; Tsai, C.-C.; Chen,

- I.-H.; Ding, J.-J.; and Kuo, S.-Y. 2021. All snow removed: Single image desnowing algorithm using hierarchical dualtree complex wavelet representation and contradict channel loss. In ICCV. Chen, W.-T.; Huang, Z.-K.; Tsai, C.-C.; Yang, H.-H.; Ding,
- J.-J.; and Kuo, S.-Y. 2022. Learning multiple adverse weather removal via two-stage knowledge learning and multi-contrastive regularization: Toward a unified model. In CVPR.

Chen, Z.; He, Z.; and Lu, Z.-M. 2024. DEA-Net: Single image dehazing based on detail-enhanced convolution and content-guided attention. IEEE TIP.

Choi, H.; Na, C.; Oh, J.; Lee, S.; Kim, J.; Choe, S.; Lee, J.; Kim, T.; and Yang, J. 2024. Reciprocal attention mixing transformer for lightweight image restoration. In CVPRW.

Cong, X.; Gui, J.; Zhang, J.; Hou, J.; and Shen, H. 2024. A semi-supervised nighttime dehazing baseline with spatialfrequency aware and realistic brightness constraint. In CVPR.

Dong, L.; Fan, Q.; Guo, Y.; et al. 2025. TSD-SR: One-step diffusion with target score distillation for real-world image super-resolution. In CVPR.

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

Truths RLPRAMiTDiffUIRWeatherDiffTKLInputsClearNightWGWS

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

[Figure 385]

[Figure 386]

[Figure 387]

[Figure 388]

[Figure 389]

[Figure 390]

[Figure 391]

Ground-

[Figure 392]

[Figure 393]

[Figure 394]

[Figure 395]

[Figure 396]

[Figure 397]

[Figure 398]

[Figure 399]

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

[Figure 418]

[Figure 419]

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

(ours) FSDGNDEA-NetAWRaCLe

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

[Figure 428]

[Figure 429]

[Figure 430]

[Figure 431]

[Figure 432]

[Figure 433]

[Figure 434]

[Figure 435]

[Figure 436]

[Figure 437]

[Figure 438]

[Figure 439]

[Figure 440]

[Figure 441]

[Figure 442]

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

[Figure 452]

[Figure 453]

[Figure 454]

[Figure 455]

[Figure 456]

[Figure 457]

[Figure 458]

[Figure 459]

[Figure 460]

[Figure 461]

[Figure 462]

[Figure 463]

[Figure 464]

[Figure 465]

[Figure 466]

[Figure 467]

[Figure 468]

[Figure 469]

[Figure 470]

[Figure 471]

[Figure 472]

[Figure 473]

[Figure 474]

[Figure 475]

[Figure 476]

[Figure 477]

[Figure 478]

[Figure 479]

[Figure 480]

[Figure 481]

[Figure 482]

[Figure 483]

[Figure 484]

[Figure 485]

[Figure 486]

[Figure 487]

[Figure 488]

[Figure 489]

[Figure 490]

[Figure 491]

[Figure 492]

[Figure 493]

[Figure 494]

[Figure 495]

[Figure 496]

[Figure 497]

[Figure 498]

[Figure 499]

[Figure 500]

[Figure 501]

[Figure 502]

[Figure 503]

[Figure 504]

[Figure 505]

[Figure 506]

[Figure 507]

[Figure 508]

[Figure 509]

[Figure 510]

[Figure 511]

[Figure 512]

[Figure 513]

[Figure 514]

[Figure 515]

[Figure 516]

[Figure 517]

[Figure 518]

[Figure 519]

[Figure 520]

[Figure 521]

[Figure 522]

[Figure 523]

[Figure 524]

[Figure 525]

[Figure 526]

[Figure 527]

[Figure 528]

[Figure 529]

[Figure 530]

[Figure 531]

[Figure 532]

[Figure 533]

[Figure 534]

[Figure 535]

[Figure 536]

[Figure 537]

[Figure 538]

[Figure 539]

[Figure 540]

[Figure 541]

[Figure 542]

[Figure 543]

[Figure 544]

[Figure 545]

[Figure 546]

[Figure 547]

[Figure 548]

[Figure 549]

[Figure 550]

[Figure 551]

[Figure 552]

[Figure 553]

[Figure 554]

[Figure 555]

[Figure 556]

[Figure 557]

[Figure 558]

[Figure 559]

Figure 21: Qualitative results of rain and snow scenes in the AllWeatherNight.

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

Truths TKLRAMiTInputsClearNightWeatherDiffRLP

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

[Figure 572]

[Figure 573]

[Figure 574]

Ground-

[Figure 575]

[Figure 576]

[Figure 577]

[Figure 578]

[Figure 579]

[Figure 580]

[Figure 581]

[Figure 582]

[Figure 583]

[Figure 584]

[Figure 585]

[Figure 586]

[Figure 587]

[Figure 588]

[Figure 589]

[Figure 590]

[Figure 591]

[Figure 592]

[Figure 593]

[Figure 594]

(ours) FSDGNDEA-NetDiffUIRAWRaCLeWGWS

[Figure 595]

[Figure 596]

[Figure 597]

[Figure 598]

[Figure 599]

[Figure 600]

[Figure 601]

[Figure 602]

[Figure 603]

[Figure 604]

[Figure 605]

[Figure 606]

[Figure 607]

[Figure 608]

[Figure 609]

[Figure 610]

[Figure 611]

[Figure 612]

[Figure 613]

[Figure 614]

[Figure 615]

[Figure 616]

[Figure 617]

[Figure 618]

[Figure 619]

[Figure 620]

[Figure 621]

[Figure 622]

[Figure 623]

[Figure 624]

[Figure 625]

[Figure 626]

[Figure 627]

[Figure 628]

[Figure 629]

[Figure 630]

[Figure 631]

[Figure 632]

[Figure 633]

[Figure 634]

[Figure 635]

[Figure 636]

[Figure 637]

[Figure 638]

[Figure 639]

[Figure 640]

[Figure 641]

[Figure 642]

[Figure 643]

[Figure 644]

[Figure 645]

[Figure 646]

[Figure 647]

[Figure 648]

[Figure 649]

[Figure 650]

[Figure 651]

[Figure 652]

[Figure 653]

[Figure 654]

[Figure 655]

[Figure 656]

[Figure 657]

[Figure 658]

[Figure 659]

[Figure 660]

[Figure 661]

[Figure 662]

[Figure 663]

[Figure 664]

[Figure 665]

[Figure 666]

[Figure 667]

[Figure 668]

[Figure 669]

[Figure 670]

[Figure 671]

[Figure 672]

[Figure 673]

[Figure 674]

[Figure 675]

[Figure 676]

[Figure 677]

[Figure 678]

[Figure 679]

Figure 22: Qualitative results of single degradation in the AllWeatherNight, including haze, rain streak, raindrop, snow and flare.

[Figure 680]

[Figure 681]

[Figure 682]

[Figure 683]

[Figure 684]

[Figure 685]

[Figure 686]

[Figure 687]

[Figure 688]

[Figure 689]

[Figure 690]

[Figure 691]

[Figure 692]

[Figure 693]

[Figure 694]

[Figure 695]

[Figure 696]

[Figure 697]

[Figure 698]

[Figure 699]

[Figure 700]

[Figure 701]

[Figure 702]

[Figure 703]

[Figure 704]

[Figure 705]

[Figure 706]

[Figure 707]

[Figure 708]

[Figure 709]

[Figure 710]

[Figure 711]

[Figure 712]

[Figure 713]

[Figure 714]

[Figure 715]

[Figure 716]

[Figure 717]

[Figure 718]

[Figure 719]

[Figure 720]

[Figure 721]

[Figure 722]

[Figure 723]

[Figure 724]

[Figure 725]

[Figure 726]

[Figure 727]

[Figure 728]

[Figure 729]

[Figure 730]

[Figure 731]

[Figure 732]

[Figure 733]

[Figure 734]

[Figure 735]

[Figure 736]

[Figure 737]

[Figure 738]

[Figure 739]

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

[Figure 756]

[Figure 757]

[Figure 758]

[Figure 759]

[Figure 760]

[Figure 761]

[Figure 762]

[Figure 763]

[Figure 764]

[Figure 765]

[Figure 766]

[Figure 767]

[Figure 768]

[Figure 769]

[Figure 770]

[Figure 771]

[Figure 772]

[Figure 773]

[Figure 774]

[Figure 775]

[Figure 776]

[Figure 777]

[Figure 778]

[Figure 779]

[Figure 780]

[Figure 781]

[Figure 782]

[Figure 783]

[Figure 784]

[Figure 785]

[Figure 786]

[Figure 787]

[Figure 788]

[Figure 789]

[Figure 790]

[Figure 791]

[Figure 792]

[Figure 793]

[Figure 794]

[Figure 795]

[Figure 796]

[Figure 797]

[Figure 798]

[Figure 799]

[Figure 800]

[Figure 801]

[Figure 802]

[Figure 803]

[Figure 804]

[Figure 805]

[Figure 806]

[Figure 807]

[Figure 808]

[Figure 809]

[Figure 810]

[Figure 811]

[Figure 812]

[Figure 813]

[Figure 814]

[Figure 815]

[Figure 816]

[Figure 817]

[Figure 818]

[Figure 819]

[Figure 820]

[Figure 821]

[Figure 822]

[Figure 823]

[Figure 824]

[Figure 825]

[Figure 826]

[Figure 827]

[Figure 828]

[Figure 829]

[Figure 830]

[Figure 831]

[Figure 832]

[Figure 833]

[Figure 834]

[Figure 835]

[Figure 836]

[Figure 837]

[Figure 838]

[Figure 839]

[Figure 840]

[Figure 841]

[Figure 842]

[Figure 843]

[Figure 844]

[Figure 845]

[Figure 846]

[Figure 847]

[Figure 848]

[Figure 849]

[Figure 850]

[Figure 851]

[Figure 852]

[Figure 853]

[Figure 854]

[Figure 855]

Inputs TKL FSDGN WeatherDiff WGWS RLP RAMiT DiffUIR DEA-Net AWRaCLe ClearNight (ours)

Figure 23: Qualitative results on real-world samples.

Edwin, H. L. 1977. The Retinex theory of color vision. Scientific American.

Fu, X.; Huang, J.; Zeng, D.; Huang, Y.; Ding, X.; and Paisley, J. 2017. Removing rain from single images via a deep detail network. In CVPR.

Guo, Y.; Gao, Y.; Lu, Y.; Zhu, H.; Liu, R. W.; and He, S. 2024. OneRestore: A universal restoration framework for composite degradation. In ECCV.

Han, J.; Li, W.; Fang, P.; Sun, C.; Hong, J.; Armin, M. A.; Petersson, L.; and Li, H. 2022. Blind image decomposition. In ECCV.

Hor´e, A.; and Ziou, D. 2010. Image quality metrics: Psnr vs. ssim. In ICPR.

Hu, J.; Jin, L.; Yao, Z.; and Lu, Y. 2025. Universal image restoration pre-training via degradation classification. In ICLR.

Jia, H.; Xu, Y.; Zhu, L.; Chen, G.; Wang, Y.; and Yang, Y.

- 2024. MoS2: Mixture of Scale and Shift Experts for TextOnly Video Captioning. In ACM MM. Jin, Y.; Li, X.; Wang, J.; Zhang, Y.; and Zhang, M. 2024. Raindrop clarity: A dual-focused dataset for day and night raindrop removal. In ECCV. Jin, Y.; Lin, B.; Yan, W.; Yuan, Y.; Ye, W.; and Tan, R. T.

2023. Enhancing visibility in nighttime haze images using guided APSF and gradient adaptive convolution. In ACM MM.

Johnson, J.; Alahi, A.; and Li, F.-F. 2016. Perceptual losses for real-time style transfer and super-resolution. In ECCV.

Kim, Y.; Cho, Y.; Nguyen, T.-T.; Hong, S.; and Lee, D. 2024. MetaWeather: Few-shot weather-degraded image restoration. In ECCV.

Kingma, D.; and Ba, J. 2015. Adam: A method for stochastic optimization. In ICLR.

Kulkarni, A.; Phutke, S. S.; and Murala, S. 2023. Unified transformer network for multi-weather image restoration. In ECCV.

Levin, A.; Lischinski, D.; and Weiss, Y. 2008. A closed-form solution to natural image matting. IEEE TPAMI.

Li, B.; Ren, W.; Fu, D.; Tao, D.; Feng, D.; Zeng, W.; and Wang, Z. 2019. Benchmarking single-image dehazing and beyond. IEEE TIP.

Li, R.; Cheong, L.-F.; and Tan, R. T. 2019. Heavy rain image restoration: Integrating physics model and conditional adversarial learning. In CVPR.

Li, X.; Liu, J.; Chen, Z.; Zou, Y.; Ma, L.; Fan, X.; and Liu, R. 2024. Contourlet residual for prompt learning enhanced infrared image super-resolution. In ECCV.

Li, X.; Wang, Z.; Zou, Y.; Chen, Z.; Ma, J.; Jiang, Z.; Ma, L.; and Liu, J. 2025. Difiisr: A diffusion model with gradient guidance for infrared image super-resolution. In CVPR.

Liao, Y.; Su, Z.; Liang, X.; and Qiu, B. 2018. HDP-net: Haze density prediction network for nighttime dehazing. In Advances in Multimedia Information Processing–PCM.

Lin, B.; Jin, Y.; Yan, W.; Ye, W.; Yuan, Y.; and Tan, R. T.

- 2025. Nighthaze: Nighttime image dehazing via self-prior learning. In AAAI.

Lin, B.; Jin, Y.; Yan, W.; Ye, W.; Yuan, Y.; Zhang, S.; and Tan, R. T. 2024. Nightrain: Nighttime video deraining via adaptive-rain-removal and adaptive-correction. In AAAI.

Liu, L.; Song, X.; Wang, M.; Liu, Y.; and Zhang, L. 2021. Self-supervised Monocular Depth Estimation for All Day Images using Domain Separation. In ICCV.

- Liu, X.; Zhang, Z.; Wu, X.; Feng, C.; Wang, X.; Lei, L.; and Zuo, W. 2024. Learning real-world image de-weathering with imperfect supervision. In AAAI.
- Liu, Y.; Yan, Z.; Chen, S.; Ye, T.; Ren, W.; and Chen, E. 2023a. NightHazeFormer: Single nighttime haze removal using prior query transformer. In ACM MM.

Liu, Y.; Yan, Z.; Tan, J.; and Li, Y. 2023b. Multi-purpose oriented single nighttime image haze removal based on unified variational Retinex model. IEEE TCSVT.

Liu, Y.; Yan, Z.; Wu, A.; Ye, T.; and Li, Y. 2022. Nighttime image dehazing based on variational decomposition model. In CVPRW.

- Liu, Y.-F.; Jaw, D.-W.; Huang, S.-C.; and Hwang, J.-N.

2018. DesnowNet: Context-aware deep network for snow removal. IEEE TIP.

- Liu, Z.; Wu, X.-M.; Zheng, D.; Lin, K.-Y.; and Zheng, W.S. 2023c. Generating anomalies for video anomaly detection with prompt-based feature mapping. In CVPR.

Loh, Y. P.; and Chan, C. S. 2019. Getting to know low-light images with the exclusively dark dataset. CVIU.

Luo, Y.; and Yang, Y. 2024. Large language model and domain-specific model collaboration for smart education. Front Inform Technol Electron Eng.

Mittal, A.; Soundararajan, R.; and Bovik, A. C. 2013. Making a “completely blind” image quality analyzer. IEEE SPL. Park, D.; Lee, B. H.; and Chun, S. Y. 2023. All-in-one image restoration for unknown degradations using adaptive discriminative filters for specific degradations. In CVPR.

Pei, S.-C.; and Lee, T.-Y. 2012. Nighttime haze removal using color transfer pre-processing and Dark Channel Prior. In ICIP.

Quan, R.; Yu, X.; Liang, Y.; and Yang, Y. 2021a. Removing raindrops and rain streaks in one go. In CVPR. Quan, R.; Zhu, L.; Wu, Y.; and Yang, Y. 2021b. Holistic LSTM for pedestrian trajectory prediction. IEEE TIP.

Rajagopalan, S.; and Patel, V. M. 2025. AWRaCLe: Allweather image restoration using visual in-context learning. In AAAI.

Shazeer, N.; Mirhoseini, A.; Maziarz, K.; Davis, A.; and Dean, J. 2017. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In ICLR.

Soboleva, V.; and Shipitko, O. 2021. Raindrops on windshield: Dataset and lightweight gradient-based detection algorithm. In SSCI.

Song, Y.; He, Z.; Qian, H.; and Du, X. 2023. Vision transformers for single image dehazing. IEEE TIP.

Sun, S.; Ren, W.; Gao, X.; Wang, R.; and Cao, X. 2024a. Restoring images in adverse weather conditions via histogram transformer. In ECCV.

Sun, S.; Ren, W.; Gao, X.; Wang, R.; and Cao, X. 2024b. Restoring images in adverse weather conditions via histogram transformer. In ECCV.

Valanarasu, J. M. J.; Yasarla, R.; and Patel, V. M. 2022. TransWeather: Transformer-based restoration of images degraded by adverse weather conditions. In CVPR.

Wang, C.; Zheng, Z.; Quan, R.; Sun, Y.; and Yang, Y. 2023. Context-aware pretraining for efficient blind image decomposition. In CVPR.

Wang, T.; Yang, X.; Xu, K.; Chen, S.; Zhang, Q.; and Lau, R. W. 2019. Spatial attentive single-image deraining with a high quality real rain dataset. In CVPR.

- Wang, W.; Wang, A.; and Liu, C. 2022. Variational single nighttime image haze removal with a gray haze-line prior. IEEE TIP.
- Wang, X.; Zhu, Z.; Huang, G.; Chen, X.; Zhu, J.; and Lu, J. 2024. DriveDreamer: Towards Real-World-Drive World Models for Autonomous Driving. In ECCV.

Wu, G.; Jiang, J.; Jiang, K.; and Liu, X. 2024. Learning from history: Task-agnostic model contrastive learning for image restoration. In AAAI.

Wu, G.; Jiang, J.; Wang, Y.; Jiang, K.; and Liu, X. 2025a. Debiased all-in-one image restoration with task uncertainty regularization. In AAAI.

Wu, Z.; Chen, K.; Li, K.; Fan, H.; and Yang, Y. 2025b. BVINet: Unlocking blind video inpainting with zero annotations. In ICCV.

Wu, Z.; Li, K.; Xu, Y.; Fan, H.; and Yang, Y. 2026. DLVINet: Advancing dual-lens video inpainting beyond parallax constraints. In AAAI.

Xu, Y.; Sun, Y.; Yang, Z.; Miao, J.; and Yang, Y. 2022. H2FA R-CNN: Holistic and hierarchical feature alignment for cross-domain weakly supervised object detection. In CVPR.

Xu, Y.; Zhu, L.; and Yang, Y. 2025. Mc-bench: A benchmark for multi-context visual grounding in the era of mllms. In ICCV.

Yan, W.; Tan, R. T.; and Dai, D. 2020. Nighttime defogging using high-low frequency decomposition and grayscalecolor networks. In ECCV.

Yang, H.; Pan, L.; Yang, Y.; and Liang, W. 2024. Languagedriven all-in-one adverse weather removal. In CVPR.

Yang, W.; Tan, R. T.; Feng, J.; Guo, Z.; Yan, S.; and Liu, J. 2020. Joint rain detection and removal from a single image with contextualized deep networks. IEEE TPAMI.

Ye, T.; Chen, S.; Bai, J.; Shi, J.; Xue, C.; Jiang, J.; Yin, J.; Chen, E.; and Liu, Y. 2023. Adverse weather removal with codebook priors. In ICCV.

Yu, F.; Chen, H.; Wang, X.; Xian, W.; Chen, Y.; Liu, F.; Madhavan, V.; and Darrell, T. 2020. BDD100K: A diverse driving dataset for heterogeneous multitask learning. In CVPR.

Yu, H.; Zheng, N.; Zhou, M.; Huang, J.; Xiao, Z.; and Zhao, F. 2022. Frequency and spatial dual guidance for image dehazing. In ECCV.

Yue, C.; Peng, Z.; Ma, J.; Du, S.; Wei, P.; and Zhang, D. 2024. Image restoration through generalized OrnsteinUhlenbeck bridge. In ICML.

- Zhang, F.; You, S.; Li, Y.; and Fu, Y. 2022. Gtav-nightrain: Photometric realistic large-scale dataset for night-time rain streak removal. arXiv:2210.04708.
- Zhang, F.; You, S.; Li, Y.; and Fu, Y. 2023a. Learning rain location prior for nighttime deraining. In ICCV.

Zhang, H.; Ba, Y.; Yang, E.; Mehra, V.; Gella, B.; Suzuki, A.; Pfahnl, A.; Chandrappa, C. C.; Wong, A.; and Kadambi, A. 2023b. WeatherStream: Light transport automation of single image deweathering. In CVPR.

Zhang, H.; Sindagi, V.; and Patel, V. M. 2020. Image deraining using a conditional generative adversarial network. IEEE TCSVT.

Zhang, J.; Cao, Y.; Fang, S.; Kang, Y.; and Chen, C. W. 2017. Fast haze removal for nighttime image using maximum reflectance prior. In CVPR.

Zhang, J.; Cao, Y.; and Wang, Z. 2014. Nighttime haze removal based on a new imaging model. In ICIP. Zhang, J.; Cao, Y.; Zha, Z.-J.; and Tao, D. 2020. Nighttime dehazing with a synthetic benchmark. In ACM MM.

Zheng, D.; Wu, X.-M.; Yang, S.; Zhang, J.; Hu, J.-F.; and Zheng, W.-S. 2024. Selective hourglass mapping for universal image restoration based on diffusion model. In CVPR.

Zhong, Z.; Tang, Z.; He, T.; Fang, H.; and Yuan, C. 2024. Convolution meets LoRA: Parameter efficient finetuning for segment anything model. In ICLR.

Zhou, W.; Bovik, A.; Sheikh, H.; and Simoncelli, E. 2004. Image quality assessment: from error visibility to structural similarity. IEEE TIP.

Zhu, Y.; Wang, T.; Fu, X.; Yang, X.; Guo, X.; Dai, J.; Qiao, Y.; and Hu, X. 2023. Learning weather-general and weatherspecific features for image restoration under multiple adverse weather conditions. In CVPR.

Ozdenizci,¨ O.; and Legenstein, R. 2023. Restoring vision in adverse weather conditions with patch-based denoising diffusion models. IEEE TPAMI.

