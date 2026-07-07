# arXiv:2512.08560v3[cs.CV]3Jun2026

### BrainExplore: Large-Scale Discovery of Interpretable Visual Representations in the Human Brain

Navve Wasserman1,∗ Matias Cosarinsky1,∗ Yuval Golbari1 Aude Oliva2 Antonio Torralba2 Tamar Rott Shaham2 Michal Irani1

1Weizmann Institute of Science 2Massachusetts Institute of Technology ∗Equal contribution

###### Abstract

Understanding how the human brain represents visual concepts, and in which brain regions these representations are encoded, remains a long-standing challenge. Decades of work have advanced our understanding of visual representations, yet brain signals remain large and complex, and the space of possible visual concepts is vast. As a result, most studies remain small-scale, rely on manual inspection, focus on specific regions and concepts, and rarely include systematic validation. We present a large-scale, automated framework for discovering and explaining visual representations across the human cortex. Our method comprises two main stages. First, we discover candidate interpretable patterns in fMRI activity through unsupervised, data-driven decomposition methods. Next, we explain each pattern by identifying the set of natural images that most strongly elicit it and generating a natural-language description of their shared visual meaning. To scale this process, we introduce an automated pipeline that tests multiple candidate explanations, assigns reliability scores, and selects the best description for each voxel pattern. Our framework reveals thousands of interpretable patterns spanning many distinct visual concepts, including fine-grained representations previously unreported. For a demo, models and labeled data, see our project page.

###### 1 Introduction

Understanding how the human brain represents visual information is a long-standing challenge. The visual cortex encodes a rich hierarchy of features that support object recognition, scene understanding, and visual reasoning. Yet, the structure of these representations and their underlying organization in the brain remain largely unknown. Functional Magnetic Resonance Imaging (fMRI) has become a dominant tool for studying how the human brain processes visual information, providing a noninvasive window into cortical activity while participants view natural images [1, 2]. It measures brain activity across the brain, parceled into tiny volume elements (“voxels”). Over the past decades, fMRI researchers have sought to interpret these complex voxel-level patterns to uncover the structure of visual representations in the brain. Early work targeted retinotopic maps and low-level features [3– 7], while later studies examined higher-level semantics using category-selective analyses [8–12], data-driven fMRI decompositions [13–17] and image–brain models [18–21].

However, fMRI signals are high-dimensional, comprising tens of thousands of brain voxels per subject that reflect a vast range of possible visual concepts, while available image-to-fMRI datasets remain relatively small. As a result, current studies often focus on specific properties (e.g., faces, places, food), or on particular brain regions, such as face-selective (FFA) or place-selective (PPA). Moreover, analyses typically rely on manual inspection, making it difficult to scale interpretations across many fMRI patterns and visual concepts. These limitations highlight the need for scalable, automated tools capable of discovering and explaining visual representations across the entire cortex.

Recent advances in interpretability of artificial neural networks, such as Language and Vision models, have shown that high-level concepts can be automatically discovered from internal activations [22–

Preprint.

(b) Discovering interpretable patterns and explaining them

(a) Per Brain Region fMRI Decomposition

Decomposition Components (Brain Patterns)

[Figure 1]

Hands in Action

[Figure 2]

[Figure 3]

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

[Figure 10]

Decompose

𝑎

+ 𝑎 +⋯

###### Interpret

𝑝

𝑝

Brain Activity

EBA region

[Figure 11]

Cages Cuddling

(c) Revealing interpretable Patterns Across the Visual Cortex

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

[Figure 16]

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

[Figure 23]

[Figure 24]

Animal Interaction

Two Identical

[Figure 25]

[Figure 26]

[Figure 27]

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

###### OPA

[Figure 34]

Open Mouth EBA

Forest

[Figure 35]

[Figure 36]

[Figure 37]

[Figure 38]

[Figure 39]

[Figure 40]

[Figure 41]

hV4

[Figure 42]

FFA-1 VWFA-1

[Figure 43]

PPA FFA-2

Kitchen

Black and White

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

- Figure 1: BrainExplore: Discovering Interpretable Visual Representations in the Human Brain. (a) Per–region fMRI decomposition learns component patterns such that any response is approximated as a linear combination of these patterns. (b) For each pattern, we retrieve its top-N activating fMRI responses and corresponding images, then automatically select the best matching concept and assign an alignment score. We then use these scores to identify the most interpretable patterns and their explanations. (c) Examples of discovered interpretable patterns across regions, showing pattern activations projected onto cortex, top activating images, and their textual explanations.

24]. These results suggest that similar data-driven approaches might offer new insight into the representations that emerge in the brain. Inspired by this, we introduce BrainExplore, an unsupervised, data-driven framework for large-scale discovery and evaluation of visual concept representations across the visual cortex. BrainExplore automatically discovers interpretable patterns of brain activity and explains them in a natural language.

Our pipeline comprises two main steps: (i) deriving candidate interpretable fMRI patterns via per–brain-region decomposition (Fig. 1a); and (ii) discovering interpretable patterns and explaining them (Fig. 1b). The first step performs per brain region decomposition of fMRI activity to extract components that may correspond to meaningful visual representations. Conceptually, each decomposition method learns a set of fMRI patterns (components) such that any fMRI response in a given region of interest (“ROI”) can be approximated as a linear combination of these patterns. For each component, we retrieve the top-N fMRI responses with the highest coefficient for that component and collect their corresponding images. This yields a set of images per component that visualize the visual concept that most strongly activates it. We then discover interpretable patterns by automatically identifying, for each component, the concept that best aligns with it and assigning an alignment score. This allows us to surface the most interpretable components together with their explanations.

Using BrainExplore, we discover thousands of interpretable patterns throughout the brain. Since the method automatically evaluates the quality of each explanation, it enables integration of findings across different decomposition methods rather than relying on a single one. We exploit established techniques (PCA [25], NMF [26], ICA [27]) as well as Sparse Autoencoders [28] (SAEs), a method widely used to interpret artificial neural networks via projection into high-dimensional space and sparsity constraints. SAE yields a large number of interpretable patterns and reveals visual representations that are complementary to other methods, including representations not captured elsewhere. Moreover, to overcome limited data, we leverage an image-to-fMRI prediction model [29] that synthesizes brain responses for unseen images. This allows us to expand the dataset from approximately 10k images with measured fMRI to over 120k images with measured or predicted responses. This augmentation improves decomposition quality and increases the diversity of discovered representations.

Altogether, BrainExplore reveals a large set of nuanced representations across the brain. These include interpretable patterns selective for object identity; for people, body parts, and pose (e.g., open mouths, hands holding objects, bent knees, identical objects); as well as distinct indoor and

outdoor scenes (e.g., nature, streets, oceans, rooms, toilets; see Fig. 1). All findings are quantitatively evaluated on measured fMRI not used during interpretation.

###### Our contributions are as follows:

- • We propose BrainExplore, a large-scale, automated framework, that discovers thousands of interpretable fMRI patterns including previously unreported ones.
- • We enrich fMRI decomposition with predicted fMRI, which substantially improves interpretability.
- • We propose an SAE fMRI decomposition that finds interpretable patterns beyond standard methods.
- • We will release the brain-inspired concept dictionary, the large-scale dataset of image–fMRIexplanations ranking, and the code, providing a benchmark for future work.

###### 2 Related Works

Contrasting predefined stimulus categories. A large body of work has studied category-selective activations by contrasting responses to hand-picked stimulus classes. For example, researchers have mapped the extrastriate body area (EBA) and related regions using body-selective contrasts [8, 9], and characterized scene-selective regions such as the parahippocampal place area (PPA) [30, 31]. Other studies have identified face-selective regions including the fusiform face area (FFA) [12], examined overlapping but separable responses to faces and food in fusiform cortex [10, 11], or investigated object-related properties [32]. These approaches have been highly informative, but they are limited by the need to predefine a small set of categories, to use relatively “clean” images dominated by a single concept, and to test each hypothesis separately. Moreover, the same voxels and regions can participate in many concepts, making it difficult to discover nuanced or overlapping representations using only category contrasts.

Decomposing fMRI signals. Unsupervised decomposition methods provide an alternative route to fMRI interpretability. Commonly used techniques include Principal Component Analysis (PCA) [25], Independent Component Analysis (ICA) [27], and Non-negative Matrix Factorization (NMF) [26]. These have been widely applied to fMRI research such as resting-state analysis and connectivity [33–38]. Stimulus-driven decompositions for interpretability have also been explored, in domains such as speech and audio [39–41] and visual stimuli [13, 14, 42]. For visual concept discovery, early work used PCA to capture large-scale semantic maps [13, 14, 42], while later studies leveraged NMF to obtain more interpretable components for categories such as bodies and food [15–17, 16]. ICA has seen more limited use, mainly for demonstrating retinotopic organization and low-level visual mapping [43]. Most of these studies focus on specific regions (e.g., early visual cortex, EBA, PPA) or on a small set of concepts (e.g., food, social vs. non-social images), and typically examine a single decomposition method and only the first few components via manual inspection. We systematically compare multiple decomposition methods for brain interpretability, and scale analysis to tens of thousands of components across many regions and methods. We also leverage image-to-fMRI models to greatly expand the effective training set and image pool used for decomposition, and we introduce SAE [28] for fMRI decomposition.

Image–fMRI models. Recent years have brought substantial progress in models that link images and brain activity. This includes encoding models that predict fMRI responses from images [44, 45, 29], transformations between subjects or brains [46, 47], and image reconstruction from fMRI [48, 49]. Beyond pure prediction, several works have used such models to explore cortical representations. Early studies related voxel activations to predefined semantic concepts, for example in face- and place-selective regions [18, 50]. More recent work has learned functional voxel clusters and visualized the kinds of images each cluster corresponds to [29]. Another line of work uses image-to-fMRI encoders together with generative image models to synthesize images that strongly activate particular regions or voxels [19–21, 51]. However, these approaches still operate at the level of individual voxels or entire regions: regions are too coarse to capture fine-grained mixtures of concepts, while voxels are too local and can participate in many concepts, making nuanced representations difficult to disentangle. Our approach is complementary. We use image-to-fMRI models primarily as a tool for data augmentation and interpretability at the pattern level: predicted responses for large image sets greatly expand the pool of images used to learn decompositions and to retrieve maximally activating examples for each component, enabling more robust and fine-grained hypothesis testing.

Automated interpretability. Interpretability of neural networks advanced rapidly in recent years [22–24, 52]. Recognizing that manual inspection does not scale to modern models, several works have proposed automated interpretability pipelines [53–56]. The human brain shares many of the same

[Figure 53]

[Figure 54]

[Figure 55]

[Figure 56]

Decompose Goal: Discover interpretable patterns

###### Visualize & Explain

[Figure 57]

[Figure 58]

(a) (b)

Goal: Interpreting each pattern 𝑝 based on its top N most activating images

Voxels Activation

𝐼 𝐹 𝑝

= 𝑎

Activation( ,𝑝 =Coeff( ,

#### ) )

+ 𝑎 …+ 𝑎

[Figure 59]

###### = 𝑎

𝐹

for Image .

𝐼

𝑝

𝐹

𝑝

𝑝 𝑝

Visualizing : top activating images

VLM-LLM

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

[Figure 64]

[Figure 65]

[Figure 66]

Water; Nature;

Each ROI voxel activation is represented as a linear combination of learned fMRI Components (Patterns)

Explain

[Figure 67]

[Figure 68]

Upscale

[Figure 69]

- (c)
- (d)

Goal: Scaling to an unlimited number of brain patterns without the need to assess their stimuli separately Brain-InspiredHypothesesDictionaryGeneration

Hypothesis-ImageLabeling

[Figure 70]

[Figure 71]

|Water;Sky|
|---|

Pattern Subset

Sky,Dog,Skateboard,… Surfer,Legs Bent,…

[Figure 72]

Hypothesis Dictionary

Water,Street,Inair,… Airplane,Sky,Green,…

[Figure 73]

Explain

[Figure 74]

[Figure 75]

|Street;Cars|
|---|

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

Discover

[Figure 80]

Goal:Scoreeachpattern–hypothesisalignmentforsystematicdiscoveryofinterpretablepatterns

###### PatternSearch ״Sittingaroundatable״

Hypothesis-Pattern Scoring

top activating images for 𝒑𝒊

[Figure 81]

[Figure 82]

=Mean(

[Figure 83]

Score( ,

[Figure 84]

Which patterns are most interpretable? (ranked across allROIs)

[Figure 85]

[Figure 86]

[Figure 87]

## (

## (

=0.66

Street

1 0 1

𝑝

ℎ

Scores for “Street”Labels

[Figure 88]

[Figure 89]

[Figure 90]

[Figure 91]

[Figure 92]

Whichpattern represents“resting”?

- Figure 2: BrainExplore Framework. (a) Decompose: per-region fMRI decomposition to discover interpretable patterns (Sec. 3.2). (b) Visualize & Explain: retrieve the top activating images for each pattern and interpret its semantics (Sec. 3.3). (c) Upscale: scale to an unlimited number of patterns by building a brain-inspired dictionary and labeling each image with respect to each hypothesis (Sec. 3.4). (d) Discover: score each pattern–hypothesis alignment, enabling systematic discovery of the most interpretable patterns and the pattern that best explains any given hypothesis (Sec. 3.5).

challenges: it is a large, complex system with many units (voxels) and a vast repertoire of concepts. However, as far as we know, no large-scale, automatic interpretability pipeline has been proposed for the brain. Our work bridges these directions. We combine unsupervised fMRI decomposition with an automatic interpretability pipeline. Together with data augmentation from image-to-fMRI models and the use of SAEs, this yields a scalable framework that discovers many interpretable brain patterns and provides a benchmark for evaluating and improving future decomposition methods.

- 3 Methods

We first describe the experimental setting used in our framework (Sec. 3.1). We then present BrainExplore, our brain interpretability framework with four parts (see Fig. 2): (i) Decompose: per-region fMRI decomposition into pattern components (Sec. 3.2); (ii) Visualize & Explain: retrieve the top activating images for each pattern and provide an interpretation of its semantics (Sec. 3.3); (iii) Upscale: scale to an unlimited number of brain patterns without interpreting each stimulus separately (Sec. 3.4); (iv) Discover: score each pattern–hypothesis alignment for systematic discovery of interpretable patterns (Appendix 3.5).

###### 3.1 Experimental Setting

We use the Natural Scenes Dataset (NSD) [1], a large publicly available 7-Tesla fMRI dataset that records responses from 8 subjects viewing diverse natural images drawn from COCO [57]. The dataset contains ∼73k image–fMRI pairs in total, with around 10k images per subject (some images are shared across subjects). We adopt the post-processed version of NSD released by Gifford et al. [58], which includes ∼40k voxels per subject together with a predefined division into mainly visionrelated ROIs. These ROIs are used for our per-ROI decomposition and subsequent analyses. To enrich the training data, we additionally use predicted fMRI responses for images not viewed by any subject. Specifically, we take ∼120K additional natural images from the unlabeled portion of COCO and employ the image-to-fMRI encoder of Beliy et al. [29] to predict fMRI responses for each subject. This augmentation produces a substantially larger set of image–fMRI pairs, which we use both for training decompositions and for retrieving maximally activating images. Importantly, all interpretations are evaluated and verified on measured fMRI.

Surfing

Soccer Tennis Frisbee

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

[Figure 100]

[Figure 101]

[Figure 102]

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

BrainRegionEBA

Jumping

Brushing teeth Hands Tied neckwear

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

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

[Figure 134]

[Figure 135]

[Figure 136]

[Figure 137]

[Figure 138]

[Figure 139]

[Figure 140]

Holding object Airplanes Black and white Lighting contrast

BrainRegionV4

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

- Figure 3: Discovered Interpretable Patterns (EBA and V4). We show patterns of subject 1 with their top activating images and selected explanations. EBA is known to process bodies and actions; V4 is known to encode mid-level features (e.g., color, shape).

###### 3.2 fMRI Decomposing

The first step in our pipeline performs per-ROI decomposition of fMRI activity to extract component patterns that may correspond to meaningful visual representations. Conceptually, each decomposition method learns a set of fMRI patterns (components) such that any fMRI response in a given ROI can be approximated as a linear combination of them (Fig. 2a). Each trained decomposition yields two outputs: (i) a component matrix, where each column is a learned fMRI pattern (for PCA, NMF, and ICA this is the standard loading matrix; for SAEs it corresponds to the decoder weights); and (ii) a coefficient matrix, which contains the linear coefficients reconstructing each fMRI response from these patterns. Importantly, all decompositions are learned purely from fMRI responses; no image features or labels are used, ensuring that all inferred visual representations arise from brain activity rather than externally imposing image semantics.

We train three standard decomposition methods (PCA, NMF, and ICA) as well as SAEs. For each method, we train two variants: (i) using only measured fMRI responses (∼10k responses per subject), and (ii) using the combined dataset of measured fMRI and 120k predicted responses. For each method, we adopt default parameter settings and vary a few hyperparameters (e.g., variance thresholds for PCA/NMF/ICA and different seeds; sparsity settings and expansion factors for SAE). Each decomposition produces a large pool of candidate patterns per ROI, later used in the Discover step. As a baseline, we also treat individual voxel activations as a single-component “decomposition”, effectively serving as a one-hot voxel basis. Training details are provided in (Appendix C.1).

###### 3.3 From fMRI Patterns to Visual Concepts

To interpret each decomposed brain pattern, we first connect it to the set of stimuli that drive it (Fig. 2b). Each fMRI response is directly associated with a stimulus — the natural image viewed by the subject. After decomposition, every fMRI response is represented by a vector of coefficients, one per pattern component, indicating how strongly each pattern contributes to its reconstruction. We assume that the responses with the highest coefficients for a given pattern are the most informative about its underlying visual semantics. Thus, for every pattern, we select its top-N activating responses and collect their corresponding images, producing a set of images that visualize what most strongly activates that pattern. We perform this separately for measured-fMRI responses and predicted responses. To explain each pattern, we take the six most activating images from the measured-fMRI pool and the ten most activating images from the predicted-fMRI pool (16 images total). We generate captions for these images using a Vision–Language Model (VLM) and then prompt an LLM to synthesize 5–10 hypotheses capturing what is shared across them (e.g., object identity, pose, texture, scene category). These hypotheses serve as candidate semantic explanations for the pattern.

Stone building Collage Screens

Landscape

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

PPABrainRegion

Indoor Kitchen Commercial buildings

Healthy food

[Figure 189]

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

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

- Figure 4: Discovered Interpretable Patterns (PPA). We show Subject 1 patterns with top activating images and selected explanations. PPA encodes scenes and places (e.g., indoor/outdoor, landmarks).

###### 3.4 Scaling to an Unlimited Number of Patterns

Explaining every component in every ROI and decomposition method is costly and inefficient, especially with high-dimensional SAEs, multiple hyperparameters and methods. To scale interpretability, we introduce two steps (Fig. 2c): (i) Hypothesis dictionary generation, which constructs a brainactivity-inspired set of candidate hypotheses; and (ii) Hypothesis–image labeling, which assigns binary labels for each hypothesis and image. Together, these two steps create large-scale associations between visual stimuli and candidate descriptions of brain activity, independent of any specific pattern. Once the dictionary and image labels are in place, any new brain pattern can be evaluated by measuring how consistently its top-activating images express concepts from the dictionary.

Hypothesis dictionary generation. We first identify patterns most likely to be interpretable. To do so, we compute a CLIP-based consistency score for every pattern: we embed its top activating images, compute pairwise similarities, and use their mean as a proxy to semantic coherence. We then select the top 40 patterns per ROI and decomposition method and run the Visualize & Explain step on this subset (> 10k patterns), yielding more than 1k unique candidate explanations. These are aggregated into a brain-inspired dictionary of visual concepts. To remove duplicates we embed each hypothesis with a BGE text encoder [59] and merge concepts with high embedding similarity. This produces a dictionary of ∼1,300 concepts that we can use to label all stimuli offline with respect to every candidate concept, creating large-scale associations between images and candidate brain activity descriptions, independent of any specific pattern. Further details and prompts appear in Appendix C. Hypothesis–image labeling. Our goal is to obtain, for every image in the dataset (both for images with measured and predicted fMRI), a binary vector indicating which concepts from the dictionary apply to that image. Directly querying a VLM with all 1,300 hypotheses per image is infeasible and often leads to inaccurate results, therefore we use a two-stage procedure. First, we use CLIP to shortlist a small set of candidate hypotheses per image: We embed each hypothesis under multiple prompt templates, compute its similarity to the CLIP image embedding, and retain the top 300 hypotheses per image. Second, we perform VLM-based verification. For each shortlisted candidate hypothesis, we prompt a VLM to output a binary decision (0/1) indicating whether the hypothesis holds for a given image. To improve reliability, we run a second verification pass with a different prompt for all hypotheses initially marked as positive, and keep only those confirmed twice. This produces a sparse but high-quality binary image–hypothesis label matrix, used later for large-scale pattern–hypothesis scoring. Additional details and labeling examples are provided in Appendix C.

###### 3.5 Discover Interpretable Brain Patterns

The goal of the Discover step (Fig. 2d) is to determine which visual concept each fMRI pattern represents. It uses the outputs of Upscale (the hypothesis dictionary and image–hypothesis labels) to evaluate any decomposition component pattern by checking whether the images that activate it express consistent visual concepts.

Hypothesis–pattern scoring. For each pattern, we quantify how strongly it aligns with each candidate hypothesis. Every fMRI response has a coefficient indicating how much a given pattern contributes to reconstructing that response. We rank all images by this coefficient and select the top 0.2% most activating images for that pattern. This is done separately for the measured and predicted fMRI pools, excluding responses that do not meaningfully activate the pattern (SAE coefficients

Table 1: Average Number of Interpretable Hypotheses. The number/fraction of hypotheses with an alignment score above a threshold (0.5 or 0.8) for at least one pattern, averaged across 8 subjects.

Measured fMRI + Predicted fMRI Method > 0.5 > 0.8 > 0.5 > 0.8

Voxels 42 / 3.17% 4 / 0.34% 86 / 6.50% 15 / 1.19% PCA 20 / 1.53% 4 / 0.32% 78 / 5.90% 11 / 0.84% NMF 19 / 1.45% 3 / 0.23% 61 / 4.62% 10 / 0.78% ICA 11 / 0.88% 2 / 0.22% 207 / 15.56% 34 / 2.62% SAE 104 / 7.87% 18 / 1.33% 219 / 16.46% 51 / 3.80% SAE+ICA 105 / 7.86% 18 / 1.34% 250 / 18.81% 58 / 4.36%

< 0.01; non-SAE coefficients ≤ 0). Since each image has binary labels for all hypotheses, we can directly measure how well a hypothesis matches the pattern. For a given pattern and hypothesis

h, we compute its score as #{top-activating imagesN labeled with h}, where N is the total number of selected top-activating images. This measures how consistently concept h appears among the images that

drive the pattern. Some hypotheses are globally rare, so even a small number of positives may indicate strong alignment. To avoid penalizing such cases, we apply a normalization factor based on the global frequency of the hypothesis (the proportion of images labeled “1” for that hypothesis across the dataset), capping the factor at 2. The resulting normalized score reflects how unexpectedly frequent the hypothesis is within the pattern’s top images. Scores are computed independently for the measured and predicted fMRI pools, and the final hypothesis-pattern score is defined as their average.

Pattern search. Given the pattern–hypothesis scores, we support two complementary searches: (i) pattern search: find the most interpretable pattern(s); and (ii) hypothesis search: for a given hypothesis (e.g., “open mouth”), find the best-explained pattern. Both searches can be run per ROI (best pattern within an ROI) or across ROIs (best pattern over all ROIs), enabling region-specific (if/where a hypothesis is represented) and method-level (which decompositions capture it) analyses.

###### 4 Experiments

###### 4.1 Evaluation Protocol

To ensure fair evaluation, we split each pool into disjoint ranking and evaluation halves: the ranking set is used to assign best pattern-hypothesis pairs; the evaluation set is used to report final metrics at Sec. 4. In the main paper, we present results for Subject 1 from the NSD dataset. Quantitative and qualitative results for the other subjects are provided in Appendices D and E, demonstrating the effectiveness of our pipeline across subjects. Since every pattern receives a score, searches can be conducted not only within a single decomposition, but also across methods and hyperparameters, allowing integrated comparison and selection.

###### 4.2 Quantitative Evaluation

We report two metrics: (i) Number of Interpretable Hypotheses (Table 1): the number/fraction of hypotheses that achieve an alignment score above a specified threshold (0.5 or 0.8) with at least one pattern. This can be computed either per ROI or across all ROIs. (ii) Number of Interpretable Patterns (Appendix Table T5): the number of patterns whose best hypothesis alignment exceeds the same threshold (0.5 or 0.8). To avoid counting near-duplicate patterns, we remove any component whose voxel-wise correlation with a higher-scoring pattern exceeds 0.5. This table shows that SAE yields far more interpretable patterns than other methods: approximately 9k patterns across the brain at a 0.5 threshold, compared to ∼6k for the Voxels baseline and 226 for ICA.

Predicted fMRI substantially improves interpretability. Table 1 compares decomposition methods trained on measured fMRI alone versus augmented with predicted fMRI, used both to learn the decomposition and to retrieve top-activating images. Predicted fMRI yields large gains across methods: ICA increases from an average of around 12 to 207 interpretable hypotheses at the 0.5 threshold, and SAE increases from around 105 to 218.

SAE provides the largest number of interpretable hypotheses. SAE yields the highest number of interpretable hypotheses, both with and without predicted fMRI.

[Figure 213]

- Figure 5: ROI–concept alignment. Applying our pipeline to average ROI activation shows scores consistent with the known functional selectivity of visual regions. For example, FFA/OFA align with faces and place regions (e.g. PPA) with indoor/outdoor scenes.

ROI k = 2 k = 4 k = 8 EBA 0.84* 0.72* 0.61* FBA 0.76 0.60* 0.48* FFA 0.73 0.55 0.42* OFA 0.69 0.50 0.38 PPA 0.75 0.58* 0.44* OPA 0.80* 0.66* 0.55* RSC 0.67 0.47 0.35 VWFA 0.73 0.55 0.41 OWFA 0.70 0.51 0.38*

Table 2: Concept consistency across subjects. We collected the interpretable concepts discovered in each ROI and measured their overlap across subject subsets of size k, compared to a null baseline. Asterisks indicate p < 0.05.

Combining decompositions yields the best performance. Integrating across methods (ICA+SAE) yields the strongest performance. This is enabled by our automatic scoring, which evaluates any number of pattern–hypothesis pairs and supports aggregation across decompositions.

###### 4.3 Alignment with Established Neuroscientific Knowledge

Our method reveals a substantially larger number of interpretable patterns and visual concepts than prior work. These include previous reported findings, concepts aligned with known functionality of specific brain regions, but to the best of our knowledge not previously shown, and additional less expected concepts, reflecting that many aspects of cortical organization remain not fully understood. We next validate our results by showing consistency with established neuroscientific knowledge.

Alignment with known category selectivity in functional regions. We conducted two experiments to validate alignment between our discovered concepts and established functional selectivity of visual regions. (i) ROI–concept alignment: We applied our scoring procedure at the ROI level, using the average activation of each ROI instead of individual voxels or decomposition components, to compute an ROI–concept alignment score. Since ROIs are known to be polysemantic, we do not expect very high alignment scores. Nevertheless, we expect the dominant concepts in each ROI to reflect its known functional role. As shown in Fig. 5, the resulting alignments are consistent with established knowledge: for example, EBA aligns strongly with actions and sports [8, 9]; FFA aligns with people and faces [12]; PPA, OPA, and RSC align with locations [30, 50]; and OWFA and VWFA align with text-related concepts [60, 61]. (ii) LLM-based relevance labeling: We further automatically labeled each concept–ROI pair according to whether it is generally consistent with the known functional role of that ROI. We then measured, for each ROI, the fraction of concepts labeled as relevant that were found to be interpretable in that ROI. The results, shown in Fig. S1 and discussed in more detail in Appendix B, further support the validity of the discovered patterns.

Consistency across subjects. The functional roles of visual regions are expected to show some consistency across subjects. To evaluate this, for each ROI and subject, we collected the set of interpretable concepts discovered in that ROI and measured their overlap across subjects. In Table 2, we report the fraction of concepts in each ROI consistent across all subject pairs, and groups of 4 and 8 subjects. As expected, consistency is high but not perfect, reflecting inter-subject variability. We further analyze the cross-subject consistency of patterns in Appendix B.3 and Fig. S2.

###### 4.4 Visual Representations in the Human Brain

BrainExplore reveals fine-grained patterns across the brain. BrainExplore identifies interpretable patterns effectively. For any hypothesis of interest, we can retrieve the best aligned pattern within a specific ROI or across ROIs, and visualize its representation via the top activating images. Figs. 1 and 3 show examples from the top 16 images per pattern (8 from measured, 8 from predicted fMRI; full grids appear in Appendix E). Although the predicted fMRI pool is much larger and often yields clearer visualizations, we validate each explanation on measured fMRI as well. Overall, BrainExplore uncovers many fine-grained patterns, including ones not previously identified. For example, in EBA, classically linked to body perception and action, we find patterns selective for specific sports (surfing, soccer, tennis, frisbee), actions (jumping, tooth brushing), and body oriented concepts (hands, neckwear). In PPA, known to respond to scenes, we observe a more nuanced division

|[Figure 214]|
|---|

|[Figure 215]|
|---|

EBA

PPA

- Figure 6: Concepts best explained by each ROI (EBA, PPA). Each concept is assigned to a single ROI—the one with the highest alignment score. Only concepts with alignment > 0.5 are shown; word size reflects the alignment score within the assigned ROI.

than prior indoor-outdoor contrasts, including distinct outdoor concepts for landscapes, commercial buildings, and stone architecture. Additional concepts and ROIs are shown in Appendix E.

Different ROIs represent different semantics. Different ROIs are known to support distinct functions. Many of our discovered patterns align strongly with these known roles, which both validates our approach and lends credibility to novel findings. Localizing established functions remains valuable—especially when we reveal finer semantics. To examine this systematically, we first find, for each hypothesis, the best matching pattern across ROIs; then, for each ROI, we list the hypotheses it best explains. In Fig. 6, we show examples where word size reflects the hypothesis–pattern score in that ROI. As shown, our method recovers many concepts, with different ROIs best explaining different ones, as expected.

Interpretable patterns are more localized with SAE. The brain exhibits meaningful spatial organization at both regional and subregional scales. While low-level maps are well studied, higherlevel semantic organization remains less clear. We find that SAEs yield interpretable patterns that are notably more spatially localized, despite receiving only 1D voxel vectors and no spatial information or spatial constraint. In Fig. 3, we show patterns from both ICA and SAE. In EBA, for example, soccer, frisbee, and jumping are derived using ICA, whereas the remaining patterns come from SAE. Additional side-by-side comparisons of pattern localization appear in Appendix E. SAE patterns are visibly more compact and clustered rather than scattered. Greater localization is both biologically plausible and of strong interest to the neuroscience community.

###### 4.5 Ablations & Analysis

In Appendix A we analyze several aspects of our pipeline, including: (i) SAE ablations - the importance of sparsity and of expanding dimensionality; (ii) Predicted fMRI pool size - results with varying image–fMRI pool sizes for retrieving top-activating images and their effect on interpretability; (iii) Per-ROI quantitative results - detailed metrics reported per ROI; and (iv) Complementarity across decompositions - the gains from combining different methods and an analysis of concepts uniquely captured by each.

###### 5 Discussion & Limitations

We introduced BrainExplore, a scalable framework for discovering visual concept representations across the visual cortex. By combining multiple decomposition methods within an automatic pipeline and enriching both training and retrieval with a large pool of predicted fMRI signals, BrainExplore reveals thousands of interpretable patterns capturing fine-grained visual concepts across regions. Despite the advantages of our large-scale pipeline, several limitations remain. Our approach uses VLM-based labeling, which can be noisy and may reduce the accuracy of the resulting scores. We mitigate this through a careful two-stage procedure, performing one-concept-at-a-time labeling followed by a second VLM verification step. Moreover, while our extensive evaluation supports the validity of the discovered patterns, definitive validation of newly identified representations will require future targeted neuroscience experiments. We hope that BrainExplore will help guide the design of such experiments. Overall, our work takes a meaningful step toward understanding visual representations in the brain and provides a practical tool for large-scale discovery. It uncovers multiple previously unreported visual representations and can help guide future computational and experimental studies.

###### Acknowledgments

This research was partially supported by the European Research Council (ERC) under the Horizon programme, grant number 101142115. We are also grateful for the support of ARL, MIT-IBM Watson AI Lab, Hyundai Motor Company and ONR MURI.

###### References

- [1] Emily J Allen, Ghislain St-Yves, Yihan Wu, Jesse L Breedlove, Jacob S Prince, Logan T Dowdle, Matthias Nau, Brad Caron, Franco Pestilli, Ian Charest, et al. A massive 7t fmri dataset to bridge cognitive neuroscience and artificial intelligence. Nature neuroscience, 25(1):116–126, 2022.
- [2] Tomoyasu Horikawa and Yukiyasu Kamitani. Generic decoding of seen and imagined objects using hierarchical visual features. Nature Communications, 8:15037, 2017. doi: 10.1038/ ncomms15037.
- [3] Stephen A. Engel, Gary H. Glover, and Brian A. Wandell. Retinotopic organization in human visual cortex and the spatial precision of functional mri. Cerebral Cortex, 7(2):181–192, 1997. doi: 10.1093/cercor/7.2.181.
- [4] Martin I. Sereno, Anders M. Dale, John B. Reppas, Kai K. Kwong, John W. Belliveau, Thomas J. Brady, Bruce R. Rosen, and Roger B. H. Tootell. Borders of multiple visual areas in humans revealed by functional magnetic resonance imaging. Science, 268(5212):889–893, 1995. doi: 10.1126/science.7754376.
- [5] E. A. DeYoe, G. J. Carman, P. Bandettini, S. Glickman, J. Wieser, R. Cox, D. Miller, and J. Neitz. Mapping striate and extrastriate visual areas in human cerebral cortex. Proceedings of the National Academy of Sciences, 93(6):2382–2386, 1996. doi: 10.1073/pnas.93.6.2382.
- [6] Yukiyasu Kamitani and Frank Tong. Decoding the visual and subjective contents of the human brain. Nature Neuroscience, 8(5):679–685, 2005. doi: 10.1038/nn1444.
- [7] Roger B. H. Tootell, John B. Reppas, Anders M. Dale, Rodney B. Look, Martin I. Sereno, Rafael Malach, Thomas J. Brady, and Bruce R. Rosen. Visual motion aftereffect in human cortical area mt revealed by functional magnetic resonance imaging. Nature, 375(6527):139–141, 1995. doi: 10.1038/375139a0.
- [8] Kevin S Weiner and Kalanit Grill-Spector. Not one extrastriate body area: using anatomical landmarks, hmt+, and visual field maps to parcellate limb-selective activations in human lateral occipitotemporal cortex. Neuroimage, 56(4):2183–2199, 2011.
- [9] Paul E Downing, Yuhong Jiang, Miles Shuman, and Nancy Kanwisher. A cortical area selective for visual processing of the human body. Science, 293(5539):2470–2473, 2001.
- [10] Kayleigh Adamson and Vanessa Troiani. Distinct and overlapping fusiform activation to faces and food. NeuroImage, 174:393–406, 2018. doi: 10.1016/j.neuroimage.2018.02.064.
- [11] Nidhi Jain, Aria Wang, Margaret M. Henderson, Ruogu Lin, Jacob S. Prince, Michael J. Tarr, and Leila Wehbe. Selectivity for food in human ventral visual cortex. Communications Biology, 6(1):175, 2023. doi: 10.1038/s42003-023-04546-2. URL https://doi.org/10.1038/ s42003-023-04546-2.
- [12] Nancy Kanwisher and Galit Yovel. The fusiform face area: a cortical region specialized for the perception of faces. Philosophical Transactions of the Royal Society B: Biological Sciences, 361(1476):2109–2128, 2006.
- [13] Christine Ecker, Emanuelle Reynaud, Steven C Williams, and Michael J Brammer. Detecting functional nodes in large-scale cortical networks with functional magnetic resonance imaging: A principal component analysis of the human visual system. Human brain mapping, 28(9): 817–834, 2007.

- [14] Alexander G. Huth, Shinji Nishimoto, An T. Vu, and Jack L. Gallant. A continuous semantic space describes the representation of thousands of object and action categories across the human brain. Neuron, 76(6):1210–1224, 2012. doi: 10.1016/j.neuron.2012.10.014.
- [15] Leonard van Dyck, Martin N Hebart, and Katharina Dobs. Core neural dimensions of functionally selective areas in the human visual cortex. In European Conference on Visual Perception (ECVP), 2024.
- [16] Yuanfang Zhao, Emalie McMahon, and Leyla Isik. Separate neural representations for physical and communicative social interactions along the lateral visual pathway: evidence from datadriven voxel decomposition. Journal of Vision, 24(10):362–362, 2024. doi: 10.1167/jov.24.10. 362.
- [17] Meenakshi Khosla, N. Apurva Ratan Murty, and Nancy Kanwisher. A highly selective response to food in human visual cortex revealed by hypothesis-free voxel decomposition. Current Biology, 32(19):4159–4171.e9, 2022. ISSN 0960-9822. doi: https://doi.org/ 10.1016/j.cub.2022.08.009. URL https://www.sciencedirect.com/science/article/ pii/S0960982222012866.
- [18] Tolga Çukur, Alexander G Huth, Shinji Nishimoto, and Jack L Gallant. Functional subdomains within human ffa. Journal of Neuroscience, 33(42):16748–16766, 2013.
- [19] Ethan Hwang, Hossein Adeli, Wenxuan Guo, Andrew Luo, and Nikolaus Kriegeskorte. In silico mapping of visual categorical selectivity across the whole brain. arXiv preprint arXiv:2510.21142, 2025.
- [20] Andrew Luo, Maggie Henderson, Leila Wehbe, and Michael Tarr. Brain diffusion for visual exploration: Cortical discovery using large scale generative models. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 75740–75781. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper_files/paper/2023/file/ ef0c0a23a1a8219c4fc381614664df3e-Paper-Conference.pdf.
- [21] Zijin Gu, Keith Wakefield Jamison, Meenakshi Khosla, Emily J. Allen, Yihan Wu, Ghislain St-Yves, Thomas Naselaris, Kendrick Kay, Mert R. Sabuncu, and Amy Kuceyeski. Neurogen: Activation optimized image synthesis for discovery neuroscience. NeuroImage, 247:118812,

2022. ISSN 1053-8119. doi: https://doi.org/10.1016/j.neuroimage.2021.118812. URL https: //www.sciencedirect.com/science/article/pii/S1053811921010831.

- [22] David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, and Antonio Torralba. Network dissection: Quantifying interpretability of deep visual representations. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6541–6549, 2017.
- [23] Evan Hernandez, Sarah Schwettmann, David Bau, Teona Bagashvili, Antonio Torralba, and Jacob Andreas. Natural language descriptions of deep visual features. In International Conference on Learning Representations, 2021.
- [24] Yossi Gandelsman, Alexei A Efros, and Jacob Steinhardt. Interpreting clip’s image representation via text-based decomposition. arXiv preprint arXiv:2310.05916, 2023.
- [25] Harold Hotelling. Analysis of a complex of statistical variables into principal components. Journal of educational psychology, 24(6):417, 1933.
- [26] Daniel D Lee and H Sebastian Seung. Learning the parts of objects by non-negative matrix factorization. nature, 401(6755):788–791, 1999.
- [27] Pierre Comon. Independent component analysis, a new concept? Signal processing, 36(3): 287–314, 1994.
- [28] Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models, 2023. URL https:// arxiv.org/abs/2309.08600.

- [29] Roman Beliy, Navve Wasserman, Amit Zalcher, and Michal Irani. The wisdom of a crowd of brains: A universal brain encoder. arXiv preprint arXiv:2406.12179, 2024.
- [30] Russell A Epstein and Chris I Baker. Scene perception in the human brain. Annual review of vision science, 5(1):373–397, 2019.
- [31] Soojin Park, Timothy F Brady, Michelle R Greene, and Aude Oliva. Disentangling scene content from spatial boundary: complementary roles for the parahippocampal place area and lateral occipital complex in representing real-world scenes. Journal of Neuroscience, 31(4): 1333–1340, 2011.
- [32] Talia Konkle and Aude Oliva. Examining how the real-world size of objects is represented in ventral visual cortex. Journal of Vision, 10(7):982–982, 2010.
- [33] Christian F. Beckmann and Stephen M. Smith. Probabilistic independent component analysis for functional magnetic resonance imaging. IEEE Trans. Med. Imaging, 23(2):137–152, 2004. doi: 10.1109/TMI.2003.822821.
- [34] V. D. Calhoun, Tülay Adali, G. D. Pearlson, and J. J. Pekar. A method for making group inferences from functional MRI data using independent component analysis. Hum. Brain Mapp., 14(3):140–151, 2001. doi: 10.1002/hbm.1048.
- [35] Stephen M. Smith, Peter T. Fox, Karla L. Miller, David C. Glahn, P. Mickle Fox, Clare E. Mackay, Nicola Filippini, Kate E. Watkins, Roberto Toro, Angela R. Laird, and Christian F. Beckmann. Correspondence of the brain’s functional architecture during activation and rest. Proceedings of the National Academy of Sciences, 106(31):13040–13045, 2009. doi: 10.1073/ pnas.0905267106. URL https://www.pnas.org/doi/abs/10.1073/pnas.0905267106.
- [36] Roberto Viviani, Georg Grön, and Manfred Spitzer. Functional principal component analysis of fmri data. Hum. Brain Mapp., 24(2):109–129, 2005. doi: 10.1002/hbm.20074.
- [37] Yuan Zhong, Huinan Wang, Guangming Lu, Zhiqiang Zhang, Qing Jiao, and Yijun Liu. Detecting functional connectivity in fmri using pca and regression analysis. Brain topography, 22(2): 134–144, 2009.
- [38] Martin J McKeown and Terrence J Sejnowski. Independent component analysis of fmri data: examining the assumptions. Human brain mapping, 6(5-6):368–372, 1998.
- [39] Alicia Zeng and Jack L. Gallant. Disentangling superpositions: Interpretable brain encoding model with sparse concept atoms. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2026. URL https://openreview.net/forum?id=3aNvX9TQTo.
- [40] Sam Norman-Haignere, Nancy G Kanwisher, and Josh H McDermott. Distinct cortical pathways for music and speech revealed by hypothesis-free voxel decomposition. neuron, 88(6):1281– 1296, 2015.
- [41] Fatma Deniz, Anwar O Nunez-Elizalde, Alexander G Huth, and Jack L Gallant. The representation of semantic information across human cerebral cortex during listening versus reading is invariant to stimulus modality. Journal of Neuroscience, 39(39):7722–7736, 2019.
- [42] Gijs Joost Brouwer and David J Heeger. Decoding and reconstructing color from responses in human visual cortex. Journal of Neuroscience, 29(44):13992–14003, 2009.
- [43] VG van de Ven, B Jans, M Been, R Goebel, and P de Weerd. Intrinsic functional organization of retinotopic visual fields in human occipital cortex: A 3t fmri study. NeuroImage, 47:S63, 2009.
- [44] Kendrick N. Kay, Thomas Naselaris, Ryan J. Prenger, and Jack L. Gallant. Identifying natural images from human brain activity. Nature, 452(7185):352–355, 2008. doi: 10.1038/nature06713.
- [45] Thomas Naselaris, Kendrick N. Kay, Shinji Nishimoto, and Jack L. Gallant. Encoding and decoding in fmri. NeuroImage, 56(2):400–410, 2011. ISSN 1053-8119. doi: https://doi.org/ 10.1016/j.neuroimage.2010.07.073. URL https://www.sciencedirect.com/science/ article/pii/S1053811910010657. Multivariate Decoding and Brain Reading.

- [46] Navve Wasserman, Roman Beliy, Roy Urbach, and Michal Irani. Functional brain-to-brain transformation with no shared data. arXiv preprint arXiv:2404.11143, 2024.
- [47] Kentaro Yamada, Yoichi Miyawaki, and Yukiyasu Kamitani. Inter-subject neural code converter for visual image representation. NeuroImage, 113:289–297, 2015.
- [48] Roman Beliy, Amit Zalcher, Jonathan Kogman, navve wasserman, and michal Irani. BrainIT: Image reconstruction from fMRI via brain-interaction transformer. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview. net/forum?id=9KjXqkfbPw.
- [49] Paul S Scotti, Mihir Tripathy, Cesar Kadir Torrico Villanueva, Reese Kneeland, Tong Chen, Ashutosh Narang, Charan Santhirasegaran, Jonathan Xu, Thomas Naselaris, Kenneth A Norman, et al. Mindeye2: Shared-subject models enable fmri-to-image with 1 hour of data. arXiv preprint arXiv:2403.11207, 2024.
- [50] Tolga Çukur, Alexander G Huth, Shinji Nishimoto, and Jack L Gallant. Functional subdomains within scene-selective cortex: parahippocampal place area, retrosplenial complex, and occipital place area. Journal of Neuroscience, 36(40):10257–10273, 2016.
- [51] Andrew Luo, Jacob Yeung, Rushikesh Zawar, Shaurya Rajat Dewan, Margaret Marie Henderson, Leila Wehbe, and Michael J. Tarr. Brain mapping with dense features: Grounding cortical semantic selectivity in natural images with vision transformers. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=yJ9QNbpMi2.
- [52] Yossi Gandelsman, Alexei A Efros, and Jacob Steinhardt. Interpreting the second-order effects of neurons in clip. arXiv preprint arXiv:2406.04341, 2024.
- [53] Amirata Ghorbani, James Wexler, James Y Zou, and Been Kim. Towards automatic conceptbased explanations. Advances in neural information processing systems, 32, 2019.
- [54] Tuomas Oikarinen and Tsui-Wei Weng. Clip-dissect: Automatic description of neuron representations in deep vision networks. arXiv preprint arXiv:2204.10965, 2022.
- [55] Sarah Schwettmann, Tamar Shaham, Joanna Materzynska, Neil Chowdhury, Shuang Li, Jacob Andreas, David Bau, and Antonio Torralba. Find: A function description benchmark for evaluating interpretability methods. Advances in Neural Information Processing Systems, 36: 75688–75715, 2023.
- [56] Tamar Rott Shaham, Sarah Schwettmann, Franklin Wang, Achyuta Rajaram, Evan Hernandez, Jacob Andreas, and Antonio Torralba. A multimodal automated interpretability agent. In Forty-first International Conference on Machine Learning, 2024.
- [57] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.
- [58] Alessandro T Gifford, Benjamin Lahner, Sari Saba-Sadiya, Martina G Vilas, Alex Lascelles, Aude Oliva, Kendrick Kay, Gemma Roig, and Radoslaw M Cichy. The algonauts project 2023 challenge: How the human brain makes sense of natural scenes. arXiv preprint arXiv:2301.03198, 2023.
- [59] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216, 2024.
- [60] Laurent Cohen, Stanislas Dehaene, Lionel Naccache, Stéphane Lehéricy, Ghislaine DehaeneLambertz, Marie-Anne Hénaff, and François Michel. The visual word form area: spatial and temporal characterization of an initial stage of reading in normal subjects and posterior split-brain patients. Brain, 123(2):291–307, 2000.
- [61] Bruce D McCandliss, Laurent Cohen, and Stanislas Dehaene. The visual word form area: expertise for reading in the fusiform gyrus. Trends in cognitive sciences, 7(7):293–299, 2003.

### Appendix

###### A Ablations and Additional Results

SAE ablations. The two main hyperparameters in our SAEs are (i) the latent-space dimensionality, controlled by the expansion factor, and (ii) the sparsity coefficient, which weights the sparsity regularization on the latent codes. Sparsity encourages each fMRI sample to be reconstructed using only a small number of patterns (i.e., a sparse linear combination of decoder columns). We evaluate four expansion factors: 0.5, 1, 2, and 4. The settings 0.5 and 1 correspond to “regular” autoencoders that reduce or roughly preserve the dimensionality, while 2 and 4 yield overcomplete representations that expand it. We also evaluate four sparsity coefficients: 0, 1, 2, and 4, where 0 corresponds to no sparsity penalty (a standard autoencoder).

In Table T1 we report, for each configuration, the rank score used for model selection (the evaluation results reported in the main paper are computed on a separate held-out set), together with the percentage of interpretable hypotheses above two thresholds (0.5 and 0.8). Sparsity has a clear and consistent effect: higher sparsity generally produces more interpretable patterns. This is expected, since enforcing that each fMRI sample is reconstructed from only a few patterns encourages those patterns to capture meaningful structure rather than overfitting. The expansion factor shows a similar trend: larger expansion factors tend to perform better, with 0.5 performing noticeably worse in most cases. Overall, both hyperparameters are important for SAE interpretability. Expansion factors larger than 4 did not yield consistent additional gains and were computationally expensive. Similarly, increasing the sparsity coefficient beyond 4 did not lead to clear improvements.

- Table T1: SAE Ablation by Expansion Factor and Sparsity. Percentage of Interpretable Hypotheses. The fraction of hypotheses that achieve an alignment score above a threshold (0.5 or 0.8) with at least one pattern (rank score).

Sparsity Threshold Exp. Factor 0 1 2 4

> 0.5

0.5 12.1% 19.2% 18.2% 22.0%

- 1 14.4% 18.6% 19.2% 22.5%
- 2 16.8% 19.8% 20.9% 22.5% 4 18.2% 21.1% 23.3% 25.0%

> 0.8

0.5 1.1% 3.0% 2.6% 3.6%

- 1 1.4% 2.6% 2.9% 3.6%
- 2 1.7% 3.8% 3.5% 3.7% 4 2.2% 2.9% 3.5% 3.8%

Predicted fMRI pool size. We aim to disentangle the influence of the retrieval pool size (the pool from which the top-N most activating images are retrieved) on interpretability performance. In

- Table T2 we report results for retrieval pools enriched with different amounts of predicted fMRI. All decomposition methods are trained on the same combined data (measured + predicted fMRI). Unlike the main paper, where we also compare models trained only on measured fMRI, here we fix the training data and vary only the retrieval pool. Importantly, increasing the pool size also increases the number of images used to compute the hypothesis scores, since the number of top activating images is defined as a fraction of the pool (0.2%). Therefore, larger pools not only tend to yield higher interpretability scores, but also make the scores more robust. Even when scores are similar, we therefore prefer larger pools. As shown in the table, enriching the pool with predicted fMRI leads to a clear improvement compared to using measured fMRI alone, and for the best-performing model the largest pool yields the highest and most stable interpretability scores.

Per-ROI quantitative results. We report, for each ROI, the percentage of explained hypotheses above a 0.5 threshold for the averaged results shown in the paper. V1–V3 are averaged across ventral and dorsal partitions, and FBA, FFA, and VWFA are averaged across their sub-areas. Higher-level visual regions show greater interpretability than early visual areas—likely because they encode higher-level semantics that are both easier to decompose and to describe using natural language within our pipeline.

- Table T2: Percentage of Interpretable Hypotheses (different retrieval pools). All decomposition methods are trained on the same combined data (measured + predicted fMRI). Unlike the main paper, where we also compare models trained only on measured fMRI, here we fix the training data and vary only the retrieval pool. We report results for pools enriched with different amounts of predicted fMRI (threshold > 0.5).

Method

Measured fMRI

+Pred 30k

+Pred 60k

+Pred 90k

+Pred 120k

Voxels 3.8% 7.1% 6.8% 7.1% 6.7% PCA (Single) 1.8% 7.1% 7.7% 7.7% 7.4% NMF (Single) 2.4% 5.6% 5.8% 5.9% 5.5%

ICA (Single) 10.4% 18.0% 18.5% 17.7% 18.1% ICA (Multiple) 10.0% 17.4% 17.6% 17.9% 18.3% SAE (Single) 5.6% 16.0% 14.6% 15.5% 15.7% SAE (Multiple) 8.2% 17.8% 16.8% 17.4% 17.4% SAE+ICA 9.8% 20.7% 20.6% 20.9% 21.5%

- Table T3: Per-ROI quantitative results. Per-ROI percentage of explained hypotheses (> 0.5) on the pool enriched with all predicted fMRI (+120k predicted fMRI). V1–V3 are averaged across ventral/dorsal; FBA, FFA, and VWFA are averages of their sub-areas.

Method V1 V2 V3 hV4 EBA OFA OPA OWFA PPA FFA RSC FBA VWFA

Voxels 0.2% 0.5% 0.6% 1.2% 5.8% 1.1% 2.8% 3.1% 2.1% 2.7% 1.0% 2.8% 3.0% PCA (Single) 0.3% 0.5% 0.5% 0.5% 4.4% 0.7% 2.3% 2.7% 2.9% 2.0% 1.8% 2.6% 3.2% NMF (Single) 0.2% 0.1% 0.4% 0.4% 2.0% 0.8% 1.1% 1.2% 1.7% 1.6% 2.2% 2.4% 1.1% ICA (Single) 1.9% 2.7% 5.7% 11.4% 15.3% 7.1% 13.9% 14.5% 12.8% 8.2% 6.2% 9.7% 0.2% SAE (Single) 0.6% 0.9% 1.3% 4.8% 14.5% 1.9% 9.2% 10.1% 6.8% 5.7% 2.9% 7.6% 7.6% SAE+ICA 1.8% 2.8% 6.8% 11.0% 18.8% 7.5% 16.8% 17.2% 13.6% 8.6% 5.2% 10.7% 11.8%

Complementarity across decompositions. We aim to understand which decomposition methods are most beneficial to combine, i.e., which ones provide complementary patterns that explain hypotheses the others do not. To this end, we measure, for every pair of decompositions, the change in the percentage of explained hypotheses (threshold 0.5) when combining their patterns. “Combining” a pair means pooling the patterns from both methods and recomputing the fraction of explained hypotheses; for reference, combining a method with itself corresponds to using multiple runs (different seeds) of the same decomposition. The reported value is the difference between the combined score and the maximum score of the two methods when used separately. As shown in Table T4, SAE and ICA are the most complementary pair, each substantially improving the other.

- Table T4: Pairwise complementarity between decompositions. Values show the gain in explained hypotheses (threshold 0.5) when combining each pair, relative to the better single method.

Voxels PCA NMF ICA SAE

Voxels 0.0% 0.5% 1.4% 0.1% -0.2% PCA 0.5% 0.0% 0.0% 0.4% 0.0% NMF 1.4% 0.0% 0.0% 0.6% 0.1% ICA 0.1% 0.4% 0.5% 0.2% 2.1% SAE -0.2% 0.0% 0.1% 2.1% 0.7%

###### B Extended Neuroscientific Validations

###### B.1 Alignment between ROI-discovered representations and known functionality

In our method, we can ask, for each ROI, which concepts are represented in it, meaning which concepts have a brain pattern in this ROI with a high interpretability score. In parallel, we can also ask which concepts from our hypothesis dictionary align with the ROI’s known functionality. We can then measure, for each ROI, what fraction of these functionally aligned concepts are indeed found to be interpretable by our method.

To create the ROI–concept alignment list, we use a strong LLM (GPT-5-mini1). We first ask it to describe, for each ROI, the general known functionality according to neuroscientific knowledge, considering only clear and widely accepted functional roles. Then, for each ROI and concept in our dictionary, we ask whether the concept is well aligned with the functionality of that ROI. The model is instructed to assign a concept to an ROI only when the concept clearly matches the ROI’s functional category. For example, the known functionality of FFA is faces, so both general concepts such as “face” or “human face” are included in the relevant concept list for this ROI, as well as more nuanced ones such as “smiling face”. EBA is known to correspond to body parts and actions, so concepts such as “hands,” “running,” and “skateboarding” are all aligned as relevant. Concepts that do not clearly correspond to a known functional domain are not assigned to any ROI. This procedure provides, for each ROI, a set of functionally relevant concepts, i.e., concepts that are assigned to that ROI by the LLM based on known functional selectivity.

We then use the score produced by our pipeline for each ROI and concept, namely the alignment score of the best-matching pattern in that ROI for that concept. We examine the relevant concept list for each ROI and measure what fraction of those concepts receives a score greater than 0.5 from our pipeline. To ensure robustness across subjects, we restrict the analysis to globally interpretable concepts, defined as concepts with interpretability score > 0.5 in at least three subjects (regardless of ROI). For each subject and ROI, we then define the set of interpretable concepts as those with score > 0.5 in that specific subject and ROI. Both the relevant and interpretable sets are restricted to the global concept set. Finally, for each subject and ROI, we compute the alignment score as the fraction of relevant concepts (as defined above) that are also interpretable. ROIs with fewer than five relevant concepts are excluded from the analysis. Results (Fig. S1) show consistently high values across subjects and ROIs, indicating that concepts identified as interpretable within an ROI tend to match its known functional specialization.

Interpretability of Functionally Relevant Concepts

1.0

[Figure 216]

- S1

- S2

- S3

- S4

- S5

- S6

- S7

- S8

- 0.82 0.82 0.90 0.75 0.83 0.75 0.86 0.67

0.87 0.74 0.80 0.88 0.83 0.71 1.00 0.39

- 0.83 0.89 0.70 0.88 0.83 0.71 0.86 0.58
- 0.84 0.82 0.90 1.00 0.94 0.77 0.86 0.72

0.9

0.8

0.93 0.84 0.60 1.00 0.94 0.74 0.71 0.72

0.7

0.92 0.84 0.80 0.88 0.89 0.78 0.86 0.67

0.6

0.86 0.76 0.80 0.62 0.83 0.77 0.86 0.53

0.72 0.79 0.80 0.62 0.72 0.62 0.86 0.67

0.5

EBA FBA FFA OFA OPA PPA VWFA hV4

- Figure S1: Alignment between ROI-discovered representations and known functionality. For each subject (rows) and ROI (columns), we report the proportion of concepts labeled as functionally relevant to that ROI (based on automatic LLM assignment) that are also found to be represented in that ROI (score > 0.5). Higher values indicate stronger agreement between the known functional role of an ROI and the concepts identified by our method.

1https://developers.openai.com/api/docs/models/gpt-5-mini

###### B.2 Cross-subject consistency of interpretable concepts

To assess the robustness of discovered concepts across subjects, we measure the overlap of interpretable concepts identified in each ROI using an intersection-over-union (IoU) metric. For each ROI, we consider subsets of k subjects and compute the IoU of their corresponding concept sets, defined as the size of the intersection divided by the size of the union. We report the average IoU across subsets of size k = 2,4,8 (Table 2 in the main text).

To evaluate whether the observed overlap exceeds chance, we compare against a permutation-based null model, where for each subset we randomly assign ROIs to subjects and recompute the IoU. We use this null distribution to assess statistical significance.

###### B.3 Cross-subject consistency of interpretable patterns

To further validate the biological relevance of the discovered SAE patterns, we asked whether patterns found in one ROI tend to have similar counterparts in the same or functionally related ROIs, both within and across subjects. For this analysis, we considered each SAE pattern in Subject 1 and measured its overlap with patterns from other ROIs using the intersection-over-union (IoU) between their sets of top-activating images. For every pattern in Subject 1, and for each target ROI separately, we identified the pattern with the highest IoU in that ROI. We then averaged these best-match scores across all patterns belonging to a given ROI in Subject 1, producing an ROI-to-ROI similarity matrix.

We visualize these results as heatmaps. Since within-Subject-1 comparisons contain especially high values on the diagonal, reflecting similarity between different SAE patterns learned in the same ROI, we do not show the Subject-1-to-Subject-1 diagonal.

Although the absolute IoU values are modest, since having exactly the same top-activating images is unlikely, the resulting structure is informative. First, the heatmaps are broadly consistent across subjects, suggesting that the discovered SAE patterns capture reproducible aspects of functional organization. Second, ROIs with similar functionality tend to show higher mutual similarity. In particular, early visual ventral regions cluster more strongly with each other than with dorsal regions, and additional ROIs with related functionality also exhibit stronger overlap. This indicates that the SAE-derived patterns are not only interpretable in terms of fine-grained visual concepts, but also capture the broader functional organization of the cortex, both within individual subjects and across subjects.

Sub1-Sub1

Sub1-Sub2

V1d V2d V3d V1v V2v V3v hV4 EBA FBA

V1d V2d V3d V1v V2v V3v hV4 EBA FBA

[Figure 217]

[Figure 218]

FFA OFA OPA PPA RSC

FFA OFA OPA PPA RSC

|[Figure 219]| |
|---|---|
| | |
| | |
| | |
| | |
| | |

OWFA VWFA

OWFA VWFA

OFA

OFA

FFA

FFA

OWFA

OWFA

VWFA

VWFA

RSC

RSC

EBA

EBA

V1d

V2d

V3d

V1d

V2d

V3d

hV4

hV4

V1v

V2v

V3v

V1v

V2v

V3v

FBA

FBA

OPA

OPA

PPA

PPA

0.10

Sub1-Sub3

Sub1-Sub4

V1d V2d V3d V1v V2v V3v hV4 EBA FBA

V1d V2d V3d V1v V2v V3v hV4 EBA FBA

[Figure 220]

[Figure 221]

0.08

FFA OFA OPA PPA RSC

FFA OFA OPA PPA RSC

OWFA VWFA

OWFA VWFA

OFA

OFA

FFA

FFA

OWFA

OWFA

VWFA

VWFA

RSC

RSC

EBA

EBA

V1d

V2d

V3d

V1d

V2d

V3d

hV4

hV4

V1v

V2v

V3v

V1v

V2v

V3v

FBA

FBA

OPA

OPA

PPA

PPA

0.06

Sub1-Sub5

Sub1-Sub6

V1d V2d V3d V1v V2v V3v hV4 EBA FBA

V1d V2d V3d V1v V2v V3v hV4 EBA FBA

[Figure 222]

[Figure 223]

0.04

FFA OFA OPA PPA RSC

FFA OFA OPA PPA RSC

OWFA VWFA

OWFA VWFA

OFA

OFA

FFA

FFA

OWFA

OWFA

VWFA

VWFA

RSC

RSC

EBA

EBA

V1d

V2d

V3d

V1d

V2d

V3d

hV4

hV4

V1v

V2v

V3v

V1v

V2v

V3v

FBA

FBA

OPA

OPA

PPA

PPA

0.02

Sub1-Sub7

Sub1-Sub8

V1d V2d V3d V1v V2v V3v hV4 EBA FBA

V1d V2d V3d V1v V2v V3v hV4 EBA FBA

[Figure 224]

[Figure 225]

0.00

FFA OFA OPA PPA RSC

FFA OFA OPA PPA RSC

OWFA VWFA

OWFA VWFA

OFA

OFA

FFA

FFA

OWFA

OWFA

VWFA

VWFA

RSC

RSC

EBA

EBA

V1d

V2d

V3d

V1d

V2d

V3d

hV4

hV4

V1v

V2v

V3v

V1v

V2v

V3v

FBA

FBA

OPA

OPA

PPA

PPA

- Figure S2: Cross-subject consistency of SAE patterns. For each SAE pattern in Subject 1, we find the pattern with the highest IoU of top-activating images in each target ROI, within the same subject and across other subjects, and average these scores across patterns from each source ROI. Functionally related ROIs show higher similarity, and the overall structure is consistent across subjects.

###### C Additional Details

###### C.1 SAE Training Details

The Sparse Autoencoder (SAE) was trained to learn a compact and interpretable latent representation of fMRI activations, enabling decomposition into meaningful spatial patterns.

Architecture. We used a simple, fully linear autoencoder, where both encoder and decoder are implemented as matrix multiplications. A ReLU activation is applied after the encoder to enforce non-negativity in the latent activations. Sparsity is encouraged via an ℓ1 loss applied to the latent space, added to the reconstruction mean-squared error (MSE) loss. The expansion factor controls the latent dimensionality (the ratio between latent and input size), while the sparsity coefficient balances the sparsity loss relative to the reconstruction loss.

Combining with predicted fMRI. We combine the real measured fMRI data (∼10k samples) with predicted fMRI activations from the image-to-fMRI encoder (∼120k samples). Real fMRI measurements are inherently noisy, and many voxels exhibit low signal-to-noise ratio (SNR). Consequently, the predicted encoder often outputs values close to zero for those voxels, leading to a mismatch in distribution between measured and predicted fMRI. Training a single SAE on this mixed data causes an imbalance, resulting in higher sparsity for predicted samples and reduced sparsity for measured ones. To address this, we use separate encoder networks for the measured and predicted data, while sharing a common decoder. This allows both data types to be decomposed into the same set of components (dictionary patterns) while enabling independent control over their activations. During training, each batch contained an equal number of measured and predicted samples to ensure balanced gradients from both data sources.

Hyperparameters. As reported in the ablation studies, we experimented with multiple combinations of sparsity coefficients and expansion factors. For the main hyperparameter search, we considered values {1,2,4} for both. The top four models selected according to the rank score threshold of 0.8 on Subject 1, were used in the multiple model setup: sparsity coefficient 4 with expansion factors {1,2,4} and sparsity coefficient 1 with expansion factor 2. The single model configuration used sparsity coefficient 4 and expansion factor 4. The same hyperparameters were applied across all subjects.

###### C.2 Baseline Decomposition Training Details

All baseline decomposition methods were trained in two configurations: (i) using only the measured fMRI data (∼10k samples), and (ii) using the combined pool of measured and predicted fMRI activations (∼120k samples). This setup enables consistent comparison between the baselines and the Sparse Autoencoder (SAE), both when relying solely on measured data and when leveraging the large scale predicted data.

Voxels. The voxel baseline uses the raw voxel activations without any decomposition or learned transformation. Each voxel is treated as an independent component. For visualization and interpretability, every voxel’s activation pattern was duplicated into two components (the original and its negation), allowing retrieval of top activating images for both positive and negative responses. This ensures interpretability is consistent across all decomposition methods, including those that produce only nonnegative activations.

PCA. Principal Component Analysis (PCA) serves as a deterministic, orthogonal decomposition baseline. It was trained on both measured only and combined fMRI datasets, with the number of components determined by the cumulative explained variance. To ensure consistency for interpretability visualization, each PCA component was duplicated into two versions, positive and negative, allowing activation maps to always correspond to positively responsive image sets.

ICA. Independent Component Analysis (ICA) was applied using numbers of components corresponding to the dimensionalities that achieve 90%, 95%, and 98% explained variance in PCA. Higher explained variance targets (>98%) were found to be unstable or unnecessary due to overfitting. ICA was run multiple times with different random seeds to capture variability arising from its nondeterministic optimization process. Similar to PCA, each independent component was duplicated into its positive and negative versions for interpretability visualization.

NMF. Nonnegative Matrix Factorization (NMF) was trained with the same component counts as ICA (matching the 0.9, 0.95, and 0.98 explained variance configurations). Since NMF produces strictly nonnegative components, no positive or negative duplication was needed. NMF was also trained on both measured and combined fMRI data. Different hyperparameter settings (for example,

initialization and regularization) showed minimal impact on the final interpretability results, consistent with its stable convergence properties under nonnegativity constraints.

###### C.3 Visualize & Explain

As described in the paper, in the Visualize step we select, for every pattern, its top-N activating responses and collect their corresponding images, producing a set of images that visualize what most strongly activates that pattern. This is done separately for measured fMRI responses and for predicted responses. We take the six most activating images from the measured fMRI pool and the ten most activating images from the predicted fMRI pool (16 images in total).

For each image, we first generate a detailed caption using Qwen2.5-VLM-32B2. The VLM is instructed (see prompt in Fig. S17) to produce rich descriptions including objects, scene or room type, dominant colors, ongoing activities, and body postures.

Given the set of N captions, we then ask a language model Qwen2.5-32B3 to generate between 3 and 12 hypotheses that may explain what is shared among at least half of those images (Fig. S18). Requesting multiple hypotheses provides diversity, since both the captions and the LLM can be biased toward a dominant concept, whereas smaller but meaningful shared attributes might otherwise be missed.

We intentionally separate the VLM and LLM stages. We found that current vision–language models often struggle to infer what is shared across multiple images, frequently producing incorrect or overly general hypotheses. By contrast, separating the process allows each VLM to focus on a single image, producing high-quality detailed captions, while the LLM operates only on textual descriptions—an easier and more reliable input for identifying cross-image commonalities.

###### C.4 Scaling to an Unlimited Number of Patterns

Explaining every component in every ROI and for every decomposition method is costly and inefficient. To scale interpretability, we introduce two steps: (i) Hypothesis dictionary generation, and (ii) Hypothesis–image labeling. Once the dictionary and the image labels are in place, any new brain pattern can be evaluated by measuring how consistently its top-activating images express concepts from the dictionary. The prompts used for hypothesis–image labeling are shown in Fig. S19, and examples of images with positively labeled hypotheses appear in Fig. S20.

###### C.5 Region of Interest (ROI)

We analyze activity within predefined Regions of Interest (ROIs), which are standard cortical parcels used to summarize responses within broader anatomical and functional subdivisions of the visual system. In the Algonauts/NSD data, these ROIs are supplied by the dataset releases based on independent localizers and anatomical parcellations. The full set used here is: V1v, V1d, V2v, V2d, V3v, V3d, hV4, VWFA-1, VWFA-2, EBA, FBA-1, FBA-2, FFA-1, FFA-2, OFA, OPA, OWFA, PPA, and RSC. Early visual areas (V1–V3, separated into ventral and dorsal subregions) correspond to the low-level visual cortex and are primarily involved in processing basic visual attributes such as orientation, color, and spatial frequency. Area hV4 is associated with color and shape processing. Higher visual regions show more specialized selectivity: EBA, FBA, and OFA are body- and faceselective regions; OPA, PPA, and RSC respond to scenes, places, and navigation-related information; VWFA and OWFA are selective to word and object forms, respectively. Together, these ROIs span the visual hierarchy from low-level visual representations to high-level semantic and categorical processing.

###### C.6 Compuational Resourences

The hypothesis generation over image sets was performed using large LLM and VLM models for the upscaling stage on an H200 GPU, requiring around 100 GPU hours. The image labeling was performed on an L40S GPU using a smaller 8B VLM model, and required around 100 GPU hours. After the upscaling stage, the pipeline itself was run on L40S GPUs; the full pipeline over all decompositions and concepts took around 4 hours.

- 2https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct
- 3https://huggingface.co/Qwen/Qwen2.5-32B-Instruct

###### D Additional Quantitative Evaluation

- Table T5: Number of Interpretable Patterns. The number of patterns whose best hypothesis alignment exceeds the same threshold (0.5 or 0.8). To avoid counting near-duplicate patterns, we remove any component whose voxel-wise correlation with a higher-scoring pattern exceeds 0.5

Measured fMRI + Predicted fMRI Method > 0.5 > 0.8 > 0.5 > 0.8 Voxels 5234 355 5905 291 PCA (Single) 1045 6 76 6 NMF (Single) 27 2 19 2

ICA (Single) 580 4 226 44 ICA (Multiple) 424 41 305 61

SAE (Single) 17242 586 8858 286 SAE (Multiple) 30005 1074 15748 617 SAE+ICA 30583 1077 16051 679

###### Table T6: Average across subjects: Interpretable Hypotheses (count) with standard error. Measured fMRI + Predicted fMRI

Method > 0.5 > 0.8 > 0.5 > 0.8

Voxels 42.1 ± 4.9 4.5 ± 0.8 86.5 ± 7.0 15.9 ± 1.9 PCA 20.4 ± 4.8 4.2 ± 1.5 78.5 ± 7.0 11.1 ± 2.1 NMF 19.2 ± 4.2 3.0 ± 0.9 61.5 ± 9.0 10.4 ± 2.2 ICA 11.8 ± 3.7 2.9 ± 1.5 207.0 ± 19.8 34.9 ± 9.4 SAE 104.6 ± 10.5 17.8 ± 4.0 218.9 ± 14.0 50.5 ± 6.3 SAE+ICA 104.5 ± 10.4 17.9 ± 4.1 250.1 ± 18.2 58.0 ± 9.7

###### Table T7: Average across subjects: Interpretable Hypotheses (percentage) with standard error. Measured fMRI + Predicted fMRI

Method > 0.5 > 0.8 > 0.5 > 0.8

Voxels 3.17 ± 0.37% 0.34 ± 0.06% 6.50 ± 0.52% 1.19 ± 0.15% PCA 1.53 ± 0.36% 0.32 ± 0.12% 5.90 ± 0.53% 0.84 ± 0.16% NMF 1.45 ± 0.32% 0.23 ± 0.07% 4.62 ± 0.68% 0.78 ± 0.16% ICA 0.88 ± 0.28% 0.22 ± 0.11% 15.56 ± 1.49% 2.62 ± 0.71% SAE 7.87 ± 0.79% 1.33 ± 0.30% 16.46 ± 1.05% 3.80 ± 0.48% SAE+ICA 7.86 ± 0.78% 1.34 ± 0.31% 18.81 ± 1.37% 4.36 ± 0.73%

Table T8: Interpretable hypotheses across all subjects (count / percentage). Subject Method Measured fMRI + Predicted fMRI

>0.5 >0.8 >0.5 >0.8

- S1

Voxels 39 / 2.9% 5 / 0.4% 88 / 6.6% 18 / 1.4% PCA 32 / 2.4% 11 / 0.8% 97 / 7.3% 13 / 1.0% NMF 18 / 1.4% 3 / 0.2% 73 / 5.5% 10 / 0.8% ICA 11 / 0.8% 0 / 0.0% 244 / 18.3% 72 / 5.4% SAE 82 / 6.2% 17 / 1.3% 231 / 17.4% 51 / 3.8% SAE+ICA 83 / 6.2% 16 / 1.2% 286 / 21.5% 78 / 5.9%

- S2

Voxels 54 / 4.1% 5 / 0.4% 107 / 8.0% 19 / 1.4% PCA 30 / 2.3% 5 / 0.4% 96 / 7.2% 18 / 1.4% NMF 28 / 2.1% 5 / 0.4% 84 / 6.3% 17 / 1.3% ICA 15 / 1.1% 2 / 0.2% 282 / 21.2% 48 / 3.6% SAE 115 / 8.6% 13 / 1.0% 247 / 18.6% 61 / 4.6% SAE+ICA 115 / 8.6% 13 / 1.0% 310 / 23.3% 70 / 5.3%

- S3

Voxels 31 / 2.3% 4 / 0.3% 72 / 5.4% 13 / 1.0% PCA 17 / 1.3% 2 / 0.2% 72 / 5.4% 8 / 0.6% NMF 14 / 1.1% 1 / 0.1% 44 / 3.3% 5 / 0.4% ICA 5 / 0.4% 1 / 0.1% 175 / 13.1% 14 / 1.1% SAE 82 / 6.2% 7 / 0.5% 191 / 14.4% 46 / 3.5% SAE+ICA 82 / 6.2% 7 / 0.5% 218 / 16.4% 45 / 3.4%

- S4

Voxels 41 / 3.1% 4 / 0.3% 83 / 6.2% 13 / 1.0% PCA 20 / 1.5% 2 / 0.2% 60 / 4.5% 9 / 0.7% NMF 20 / 1.5% 2 / 0.2% 36 / 2.7% 6 / 0.5% ICA 7 / 0.5% 2 / 0.2% 168 / 12.6% 31 / 2.3% SAE 103 / 7.7% 15 / 1.1% 189 / 14.2% 40 / 3.0% SAE+ICA 103 / 7.7% 16 / 1.2% 202 / 15.2% 42 / 3.2%

- S5

Voxels 71 / 5.3% 4 / 0.3% 122 / 9.2% 25 / 1.9% PCA 43 / 3.2% 11 / 0.8% 109 / 8.2% 17 / 1.3% NMF 43 / 3.2% 7 / 0.5% 107 / 8.0% 21 / 1.6% ICA 36 / 2.7% 13 / 1.0% 269 / 20.2% 74 / 5.6% SAE 159 / 11.9% 42 / 3.2% 262 / 19.7% 85 / 6.4% SAE+ICA 159 / 11.9% 43 / 3.2% 303 / 22.8% 110 / 8.3%

- S6

Voxels 30 / 2.3% 4 / 0.3% 66 / 5.0% 17 / 1.3% PCA 8 / 0.6% 1 / 0.1% 66 / 5.0% 4 / 0.3% NMF 15 / 1.1% 5 / 0.4% 59 / 4.4% 5 / 0.4% ICA 6 / 0.5% 1 / 0.1% 210 / 15.8% 14 / 1.1% SAE 94 / 7.1% 12 / 0.9% 226 / 17.0% 36 / 2.7% SAE+ICA 93 / 7.0% 12 / 0.9% 251 / 18.9% 36 / 2.7%

- S7

Voxels 38 / 2.9% 9 / 0.7% 87 / 6.5% 16 / 1.2% PCA 3 / 0.2% 1 / 0.1% 75 / 5.6% 17 / 1.3% NMF 3 / 0.2% 0 / 0.0% 57 / 4.3% 13 / 1.0% ICA 10 / 0.8% 3 / 0.2% 193 / 14.5% 19 / 1.4% SAE 132 / 9.9% 26 / 2.0% 257 / 19.3% 58 / 4.4% SAE+ICA 131 / 9.8% 26 / 2.0% 266 / 20.0% 58 / 4.4%

- S8

Voxels 33 / 2.5% 1 / 0.1% 67 / 5.0% 6 / 0.5% PCA 10 / 0.8% 1 / 0.1% 53 / 4.0% 3 / 0.2% NMF 13 / 1.0% 1 / 0.1% 32 / 2.4% 6 / 0.5% ICA 4 / 0.3% 1 / 0.1% 115 / 8.6% 7 / 0.5% SAE 70 / 5.3% 10 / 0.8% 148 / 11.1% 27 / 2.0% SAE+ICA 70 / 5.3% 10 / 0.8% 165 / 12.4% 25 / 1.9%

###### E Additional Visualizations

Spatial localization of interpretable patterns (SAE vs. ICA). Extending the main-text observation that SAE patterns are more spatially localized, Fig. S3 presents side-by-side brain activation maps for multiple ROIs, comparing patterns derived from ICA and from SAE. Despite being trained on onedimensional voxel vectors with no spatial priors, SAE produces compact, clustered patterns, whereas ICA frequently yields more diffuse or scattered maps. All maps were generated using identical preprocessing, color scaling, and selection criteria (top-activating images per pattern), ensuring that visual differences reflect the decomposition method rather than visualization settings.

[Figure 226]

[Figure 227]

[Figure 228]

[Figure 229]

[Figure 230]

##### SAEICA

[Figure 231]

[Figure 232]

[Figure 233]

[Figure 234]

[Figure 235]

- Figure S3: Brain activation map (SAE vs ICA). We show different activation patterns derived from a Sparse Autoencoder (SAE) and Independent Component Analysis (ICA) for different ROIs. SAE patterns are notably more compact and clustered and spatially localized, despite both receiving only 1D voxel vectors and no spatial information.

Additional concepts across ROIs. We present additional interpretable patterns from Subject 1 together with representative top-activating images. We show further results for EBA and RSC in Fig. S4, for OPA in Fig. S5, and for FBA-1 and FBA-2 in Fig. S6.

Additional subjects visualization. We present discovered interpretable patterns and concepts for two additional subjects with a larger number of examples (Subjects 2 and 5) in Figs. S7 to S10. All subjects’ trained decompositions and discovered patterns will be publicly available, enabling further research and discoveries.

Full top 16 images per pattern. We present full visual grids showing the top 16 activating images for the concepts from EBA discussed in the main paper (Figs. S11 and S12). For each concept, the top row corresponds to predicted fMRI responses and the bottom row to measured fMRI responses. Images are ordered by activation strength, from left to right. We also show all 16 top-activating images for the patterns and concepts presented in the main paper for PPA (Figs. S13 and S14).

Concepts best explained by each ROI. We summarize which semantic concepts are most strongly represented across different Regions of Interest (ROIs). Fig. S15 shows, for each ROI, the concepts achieving an alignment score above 0.5, allowing a concept to appear in multiple ROIs if it is represented across regions. In contrast, Fig. S16 presents an exclusive version in which each concept is assigned only to the ROI where it achieves its highest alignment score. Together, these visualizations reveal both the distributed and the region-specific organization of conceptual representations across the visual cortex.

Prompts. Figures S17 to S19 depict the prompts used in our pipeline: first, detailed per-image captions are elicited (Fig. S17); second, a language model infers shared hypotheses across the image set (Fig. S18); and, distinct from the two-stage explanation, the image–hypothesis labeling prompt is used in the upscale stage to decide whether each hypothesis is supported by each image (Fig. S19).

Labeled examples. Figure S20 presents representative images together with the hypotheses labeled as positive for each, demonstrating the output produced by the hypothesis–image labeling stage.

Dining table Baseball Legs in motion

Open mouth

[Figure 236]

[Figure 237]

[Figure 238]

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

[Figure 243]

[Figure 244]

[Figure 245]

[Figure 246]

[Figure 247]

[Figure 248]

[Figure 249]

[Figure 250]

[Figure 251]

[Figure 252]

[Figure 253]

[Figure 254]

[Figure 255]

[Figure 256]

[Figure 257]

[Figure 258]

[Figure 259]

BrainRegionEBA

Public transport

Airplanes Computer Bent knees

[Figure 260]

[Figure 261]

[Figure 262]

[Figure 263]

[Figure 264]

[Figure 265]

[Figure 266]

[Figure 267]

[Figure 268]

[Figure 269]

[Figure 270]

[Figure 271]

[Figure 272]

[Figure 273]

[Figure 274]

[Figure 275]

[Figure 276]

[Figure 277]

[Figure 278]

[Figure 279]

[Figure 280]

[Figure 281]

[Figure 282]

[Figure 283]

Mirror Toilet Indoor Outdoor

BrainRegionRSC

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

- Figure S4: Discovered Interpretable Patterns (EBA and RSC). We show additional patterns for Subject 1 with top activating images and selected explanations. EBA is known to encode bodies and actions, whereas RSC processes scene information.

BrainRegionOPA

Toilet Horizon Road

Vivid colors Reflecting Cluttered

Indoor

Arms for balance Ball Hands in action Human interaction

Artificial lighting

[Figure 308]

[Figure 309]

[Figure 310]

[Figure 311]

[Figure 312]

[Figure 313]

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

[Figure 323]

[Figure 324]

[Figure 325]

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

[Figure 335]

[Figure 336]

[Figure 337]

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

[Figure 342]

[Figure 343]

[Figure 344]

[Figure 345]

[Figure 346]

[Figure 347]

[Figure 348]

[Figure 349]

[Figure 350]

[Figure 351]

[Figure 352]

[Figure 353]

[Figure 354]

[Figure 355]

[Figure 356]

[Figure 357]

[Figure 358]

[Figure 359]

[Figure 360]

[Figure 361]

[Figure 362]

[Figure 363]

[Figure 364]

[Figure 365]

[Figure 366]

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

- Figure S5: Discovered Interpretable Patterns (OPA). We show additional patterns for Subject 1 with top activating images and selected explanations. OPA is known to process scene layout and navigability.

Legs extended Extended arms Utensils Shirt and tie

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

###### FBA-1BrainRegion

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

Hands relaxed Clothes Dim lighting Room

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

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

###### FBA-2BrainRegion

Smiling Eyes Animals Fabric

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

- Figure S6: Discovered Interpretable Patterns (FBA-1 and FBA-2). We show additional patterns for Subject 1 with top activating images and selected explanations. FBA is a body-selective area, encompassing two sub-areas shown here, FBA-1 and FBA-2.

BrainRegionOPA

Skies Indoor Urban background Vehicle

Extended arms Human interaction Bowl Fruit

VWFA-1BrainRegion

Eating Smiling Flying Text

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

- Figure S7: Discovered Interpretable Patterns of Subject 2 (OPA and VWFA-1). We show patterns for Subject 2 with top activating images and selected explanations. OPA is known to process scene layout and navigability, and VWFA is involved in processing word shapes and visual text.

Knife Holding object Open mouth Business attire

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

Back facing Resting Tennis

Seated people

BrainRegionEBA

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

[Figure 560]

[Figure 561]

[Figure 562]

[Figure 563]

[Figure 564]

[Figure 565]

[Figure 566]

[Figure 567]

[Figure 568]

[Figure 569]

[Figure 570]

[Figure 571]

Cooking Bed

Street view Tracks

[Figure 572]

[Figure 573]

[Figure 574]

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

[Figure 595]

- Figure S8: Discovered Interpretable Patterns of Subject 2 (EBA). We show patterns for subject 2 with top activating images and selected explanations. EBA is known to encode bodies and actions.

BrainRegionOPA

City street Indoor Clock Computer

Skiing Brushing teeth Seated at table Collage

BrainRegionFFA-1

Hands in action Holding object Eating Tie

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

- Figure S9: Discovered Interpretable Patterns of Subject 5 (OPA and FFA-1). We show patterns for subject 5 with top activating images and selected explanations. OPA is known to process scene layout and navigability, and FFA is primarily known for face processing.

Travel Cooking Relaxing Specific attire

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

EBABrainRegion

Expressive face Hands Jumping Arms for balance

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

Closed mouth Cluttered Forest Black and white

V4BrainRegion

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

- Figure S10: Discovered Interpretable Patterns of Subject 5 (EBA and hV4). We show patterns for subject 5 with top activating images and selected explanations. EBA is known to encode bodies and actions and V4 is known to encode mid-level features (e.g., color, shape).

[Figure 740]

[Figure 741]

[Figure 742]

[Figure 743]

###### Figure S11: Image grid EBA (Part 1). Full visual grid with top 16 activating images for concepts Tied neckwear, Brushing teeth, Hands, and Jumping found in the EBA region. For each concept, the top 8 images corresponding to Measured and Predicted fMRI responses are shown.

[Figure 744]

[Figure 745]

[Figure 746]

[Figure 747]

###### Figure S12: Image grid EBA (Part 2). Full visual grid with top 16 activating images for concepts Frisbee, Soccer, Surfing, and Tennis found in the EBA region. For each concept, the top 8 images corresponding to Measured and Predicted fMRI responses are shown.

[Figure 748]

[Figure 749]

[Figure 750]

[Figure 751]

###### Figure S13: Image grid PPA (Part 1). Full visual grid with top 16 activating images for concepts Stone building, Commercial buildings, Kitchen, and Indoor found in the PPA region. For each concept, the top 8 images corresponding to Measured and Predicted fMRI responses are shown.

[Figure 752]

[Figure 753]

[Figure 754]

[Figure 755]

###### Figure S14: Image grid PPA (Part 2). Full visual grid with top 16 activating images for concepts Collage, Screen, Healthy food, and Landscape found in the PPA region. For each concept, the top 8 images corresponding to Measured and Predicted fMRI responses are shown.

|[Figure 756]|
|---|

|[Figure 757]|
|---|

|[Figure 758]|
|---|

FBA-1

FBA-2

EBA

|[Figure 759]|
|---|

|[Figure 760]|
|---|

|[Figure 761]|
|---|

FFA-1

FFA-2

hv4

|[Figure 762]|
|---|

|[Figure 763]|
|---|

|[Figure 764]|
|---|

OWFA

OPA

OFA

|[Figure 765]|
|---|

|[Figure 766]|
|---|

|[Figure 767]|
|---|

RSC

PPA

V1v

|[Figure 768]|
|---|

|[Figure 769]|
|---|

|[Figure 770]|
|---|

V3d

V2d

V2v

|[Figure 771]|
|---|

|[Figure 772]|
|---|

|[Figure 773]|
|---|

VWFA-1

VWFA-2

V3v

- Figure S15: Concepts best explained by every ROI (Non-Exlusive). Concepts represented in each Region of Interest (ROI) that achieve an alignment score > 0.5. A concept may appear in multiple ROI word clouds, reflecting its representation across different brain regions. Word size reflects the alignment score of the concept within the assigned ROI.

|[Figure 774]|
|---|

|[Figure 775]|
|---|

|[Figure 776]|
|---|

FBA-1

FBA-2

EBA

|[Figure 777]|
|---|

|[Figure 778]|
|---|

|[Figure 779]|
|---|

FFA-2

FFA-1

hV4

|[Figure 780]|
|---|

|[Figure 781]|
|---|

|[Figure 782]|
|---|

OPA

RSC

PPA

|[Figure 783]|
|---|

|[Figure 784]|
|---|

|[Figure 785]|
|---|

V3d

V1v

V2v

|[Figure 786]|
|---|

|[Figure 787]|
|---|

|[Figure 788]|
|---|

VWFA-1

VWFA-2

V3v

- Figure S16: Concepts best explained by every ROI (Exlusive). Each concept is assigned to a single ROI—the one with the highest alignment score. Only concepts with alignment > 0.5 are shown. Word size reflects the alignment score within the assigned ROI.

Image Captioning Prompt

|You are given one image. Output: return ONLY a single JSON object (no prose, no code fences). Keep all fields SHORT strings. If unknown, use "" (empty string). Emit "shapes" only when clearly visible. Schema: { "caption": "2–3 short sentences. Include key objects, activities, background/room, zoom, and any visible TEXT/LOGO with its color.", "objects": "comma-separated items like: girl — smiling, holding apple; stove — metal, black, shiny", "number_of_objects": "none | one | two | multiple | many | two identical <object>", "room_type": "kitchen | bedroom | bathroom | living room | office | gym | none", "scene_type": "indoor | outdoor | mixed | none", "dominant_colors": "comma-separated color words",<br><br>"activity": "high/mid/low-level actions in a short phrase, e.g., cooking — stirring; sport/football — running; commute — walking",<br><br>"body_postures": "legs:<detailed specific posture>; hands:<- please detailed posture >; mouth:<detailed state>; in_air:<true/false>; stretching:<true/false>",<br><br>"background": "very short phrase (e.g., ocean horizon; tile backsplash; stadium crowd)",<br><br>"shapes": "only if clear: straight lines — fence/train track/shore line; clear edges — plate edge/road curb; round — plate/ball/wheel",<br><br>"food": "if present, note food types/tableware/body interaction, e.g., desert, Asian food, bread type food, pizza, vegetables; tableware plate, fork; body — hand",<br><br>"main_object_location": "one of: left/right/center + up/middle/down (e.g., left down, center middle, right up)" }|
|---|

###### Figure S17: Image Captioning Prompt.

###### Hypothesis Generation Prompt

|You are given structured captions for {N} images of the SAME SAE feature (its top activations). Images are NOT ordered by significance. Your goal: propose concise, shared hypotheses about what this feature likes. Think at multiple levels and combine cues when common:<br><br>- Low-level: colors (≤2 shared colors), textures/materials, lighting, edges/lines, shapes (round objects, squared etc.)<br>- Mid-level: objects & types, counts (one/two/multiple), actions/verbs, body parts & POSTURES, relationships (X on Y, X in Y), scene elements that co-occur<br>- High-level: style/design, scene type, overall mood/context<br><br>STRICT RULES<br><br>- Do NOT output the location "center middle" (it’s too common). Other locations are allowed.<br>- Colors: if you include colors, list at most TWO (e.g., "blue", "white and black"), not more. If 2 colors are always together in all images list them as one ( e.g. "black and white" for Zebra).<br>- Return between 3 and 12 hypotheses.<br>- Be BOTH general and nuanced: mix simple terms with specific combos when truly common (e.g., "airplane", "flying", "airplane flying"; "cows", "grass", "cows on grass", "farm animals"; "person", "skateboard", "jumping", "skateboarder jumping").<br>- Don’t stay only generic: if you see "pizza", also consider "round things"; if you see "sky", note what’s IN the sky; for "toilet", also notice "sink", "tiles", "counter", "wood".<br>- For sports/people: mention body posture and role (serving? running? legs crossed/bent/straight), hand vs leg sport, field/court type, uniform/common wear, count of people, in-action vs static.<br>- You may infer obvious shared facts even if not spelled out in every caption (e.g., pizza is round; cows eat grass; zebra is black and white).<br>- No placeholders, no explanations, no evidence lists. COUNTING RULE<br>- Only include a hypothesis if it appears in at least {support} of the {N} images (~{ratio}%). FORMAT<br>- Phrases only, no sentences, no explanations, no evidence lists, no duplicates. NOW USE THIS DATA: CAPTIONS: {captions} First, silently think about the shared concepts across these images. Then, OUTPUT (JSON only): {{"hypotheses": ["...", "..."]}}<br>|
|---|

- Figure S18: Hypothesis Generation Prompt. Given a set of 10 images and their detailed captions, an LLM is instructed to identify what is common across the images and to generate hypotheses that may explain what is shared among them.

###### Image-Hypothesis Labeling Prompt 1

|You are given ONE image and ONE short visual concept (hypothesis).\n Answer with EXACTLY one word: true or false (lowercase).\n\n Be STRICT:\n<br><br>- true = clearly visible and unmistakably present.\n<br>- false = absent, ambiguous, weakly implied, too small/occluded/background to be sure, or inferred from text only.\n<br>- ACTIONS must be clearly occurring (e.g., 'washing hands' is false if only a sink is visible).\n<br>- VAGUE terms (e.g., 'play') are false unless the action is clearly happening, not just a toy present.\n<br>- COLORS: answer true only if this is a DOMINANT color region, not a tiny patch.\n\n Return ONLY: true OR false.<br>|
|---|

###### Image-Hypothesis Labeling Prompt 2

|You are given ONE image and ONE short visual concept (hypothesis).\n This is a SECOND PASS to confirm a previous 'true' decision. Be EXTRA STRICT.\n Answer with EXACTLY one word: true or false (lowercase).\n\n Rules (even stricter than before):\n<br><br>- true = the concept is CLEAR, DOMINANT, and UNAMBIGUOUS in the image.\n<br>- false = any ambiguity, partial/occluded/tiny evidence, inference from context, "or near-match that does not exactly satisfy the concept.\n<br>- For actions: the action must be obviously occurring (not implied by objects).\n<br>- For colors: only if the color is clearly dominant (not a small patch).\n\n Return ONLY: true OR false.<br>|
|---|

- Figure S19: Image–Hypothesis Labeling Prompt. For each image–hypothesis pair, we label whether the hypothesis is supported by the image or not.

|[Figure 789]|
|---|

|[Figure 790]|
|---|

|[Figure 791]|
|---|

|[Figure 792]|
|---|

|[Figure 793]|
|---|

|[Figure 794]|
|---|

- Figure S20: Images with per-hypothesis labeling. We show six example images and list all hypotheses labeled as positive for each. The hypotheses are taken from our brain-inspired hypothesis dictionary, and the images are drawn from the set used with predicted fMRI.

