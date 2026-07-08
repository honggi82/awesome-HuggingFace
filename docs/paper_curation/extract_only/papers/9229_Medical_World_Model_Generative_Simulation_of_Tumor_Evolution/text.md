## Medical World Model: Generative Simulation of Tumor Evolution for Treatment Planning

Yijun Yang1,*, Zhao-Yang Wang2, Qiuping Liu3, Shuwen Sun3, Kang Wang4, Rama Chellappa2, Zongwei Zhou2, Alan Yuille2, Lei Zhu1,5,†, Yu-Dong Zhang3, Jieneng Chen2,†

1The Hong Kong University of Science and Technology (Guangzhou) 2Johns Hopkins University

3The First Affiliated Hospital of Nanjing Medical University 4University of California, San Francisco 5The Hong Kong University of Science and Technology

arXiv:2506.02327v1[cs.CV]2Jun2025

Project page: https://yijun-yang.github.io/MeWM

### Abstract

[Figure 1]

Survival Analysis

[Figure 2]

[Figure 3]

Providing effective treatment and making informed clinical decisions are essential goals of modern medicine and clinical care. We are interested in simulating disease dynamics for clinical decision-making, leveraging recent advances in large generative models. To this end, we introduce the Medical World Model (MeWM), the first world model in medicine that visually predicts future disease states based on clinical decisions. MeWM comprises (i) vision-language models to serve as policy models, and (ii) tumor generative models as dynamics models. The policy model generates action plans, such as clinical treatments, while the dynamics model simulates tumor progression or regression under given treatment conditions. Building on this, we propose the inverse dynamics model that applies survival analysis to the simulated post-treatment tumor, enabling the evaluation of treatment efficacy and the selection of the optimal clinical action plan. As a result, the proposed MeWM simulates disease dynamics by synthesizing post-treatment tumors, with state-of-the-art specificity in Turing tests evaluated by radiologists. Simultaneously, its inverse dynamics model outperforms medical-specialized GPTs in optimizing individualized treatment protocols across all metrics. Notably, MeWM improves clinical decision-making for interventional physicians, boosting F1-score in selecting the optimal TACE protocol by 13%, paving the way for future integration of medical world models as the second readers.

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

State(𝒔𝟎) Actor

[Figure 9]

[Figure 10]

[Figure 11]

[Figure 12]

[Figure 13]

[Figure 14]

[Figure 15]

State(𝒔𝟏···)

[Figure 16]

Progression Generative Model

Perception

[Figure 17]

[Figure 18]

[Figure 19]

[Figure 20]

[Figure 21]

[Figure 22]

###### Imaging Observation

[Figure 23]

[Figure 24]

[Figure 25]

Recovery-Conditioned Policies

[Figure 26]

[Figure 27]

Action

Figure 1. Formulation of Medical World Model. It integrates imaging observations with perception modules to form an initial state, which is then processed by a progression generative model to predict future states of disease under different treatment conditions. Recovery-conditioned policies guide treatment decisions, creating a feedback loop for optimizing clinical interventions.

tors including medical images and patient history to determine the best course of action for each patient. Artificial intelligence (AI) models are increasingly assuming a crucial role in this process by analyzing the complex multimodal data, revealing patterns that may be difficult to detect with human observation alone, and suggesting tailored treatment strategies based on predictive analytics.

Foundation models [9] such as large language models (LLMs) [37, 64] present a new frontier in medical AI research and development. However, recent studies [29] show that LLMs, even those specifically tailored for medicine [16, 61, 62], diagnose significantly worse than clinicians and make less informed treatment decisions. This is due to several challenges. First, the complexity of diseases themselves, such as tumors that evolve under the influence of diverse biological and chemotherapy factors, calls for models that can adapt and account for disease progression. Second, clinical decision-making necessitates not only accurate predictions but also visually trackable insights

### 1. Introduction

Clinical decision-making is at the heart of patient care, driving outcomes and shaping the trajectory of healthcare interventions. Physicians constantly weigh the multimodal fac-

*Work done while visiting at JHU. †Corresponding authors.

that physicians can trust.

Recent breakthroughs in world models (WMs) [4, 10, 43, 51, 69] provide a promising avenue for overcoming these obstacles. By generating a predictive distribution of how the world states evolve, WMs mirror the way human planners imagine future scenarios and then make informed decisions via inverse dyanmics [4, 21]. Although they remain largely underexplored in the medical domain, world models hold significant potential for generating clinically realistic images and simulating disease progression, which in turn can facilitate more effective and visually trackable treatment planning. Figure 1 illustrates our formulation of introducing WMs into generalized medical scenarios and how WMs integrate these capabilities to support clinical decision-making.

In this work, we introduce Medical World Model (MeWM) to address these challenges and push the boundaries of AI-driven clinical decision support. MeWM comprises three primary components: (1) a Policy Model powered by vision-language architectures, which generates the potential action combos from a patient’s current state and specific clinical scenario; (2) a Dynamics Model that forwards and simulates tumor dynamics, predicting how tumors could progress or regress under different treatment conditions by generative modeling; (3) an Inverse Dynamics Model that performs survival risk analysis on the simulated post-treatment tumor, and quantitatively evaluates treatment efficacy. Beyond forward simulation, this system heuristically explores the optimal plan with the assistance of a segmentation model. By uniting these elements, MeWM delivers a holistic framework for decision-making: it can synthesize realistic post-treatment tumors that pass Turing tests against radiologists, and it outperforms specialized GPT-like models on Transarterial Chemoembolization (TACE) Protocol Exploration (over 10%↑ in F1-score).

Overall, our contributions are threefold.

- • We propose the medical world models, where we develop a multimodal policy model that leverages visionlanguage capabilities to propose a tailored set of treatment action combos, and we design a generative dynamics model that accurately captures potential evolution of tumors, enabling forward-looking simulations for different interventions.
- • We integrate an inverse dynamics model that translates these action-conditioned simulation into survival analysis metrics, thereby offering a transparent and evidencebased tool for choosing the optimal treatment protocol.
- • We demonstrate a substantial leap in AI-driven decision support for interventional medicine, improving the F1score in selecting the optimal treatment protocol by 13% and offering a compelling glimpse into the future of precision healthcare.

### 2. Related Work

#### 2.1. Generative World Modeling

World models [26, 43] aim to simulate dynamic environments by predicting future states and rewards based on current observations and actions. Originally developed for constrained settings like Atari games [27], their ability to model state transitions has been extended to real-world scenarios through joint learning of policies and world models, improving sample efficiency in simulated robotics [59], real-world robots [70] and autonomous driving [33, 68]. While early world models focused on simple state transitions, modern approaches integrate structured action-object relationships [66] and multi-modal conditioning [7, 24]. For instance, Du et al. [20] present long-horizon video plans by synergising vision-language models and text-to-video models. Luo et al. [52] propose to ground video models to continuous action by leveraging video-guided goal-conditioned exploration to learn a goal-conditioned policy. In embodied decision-making, Lu et al. [51] enables agents to imaginatively explore the world with high generation quality and exploration consistency using video generative models. However, there is still no work investigating the applicability of world modeling in medical image analysis and clinical decision-making.

#### 2.2. Tumor Synthesis

Tumor synthesis has emerged as an attractive research topic across various medical imaging modalities, such as CT [14, 53, 74], MRI [6, 35, 71], and endoscopic videos [15, 45]. There are also many works on synthesizing non-cancerous lesions including chest CT synthesis [8, 53, 74, 75], and diabetic lesion synthesis in retinal images [18, 78]. Recent studies focus on improving the realism of synthetic tumors in the liver, kidney and pancreas [14, 34, 42] by leveraging the large generative models like diffusion models [32, 58, 63]. While these methods are conditioned only on shape masks, Li et al. [47] propose text-driven tumor synthesis by descriptive reports and conditional diffusion models. However, most of these works implemented tumor synthesis as a data augmentation to improve tumor detection tasks. They overlook its potential to empower clinical decision-making in treatment planning. In this work, we delve into the relatively unexplored field of tumor dynamics simulation by generating post-treatment tumors using pretreatment scans and treatment actions.

#### 2.3. Prognosis and Clinical Decision-making

Post-treatment prognosis in medical imaging is essential for evaluating therapy effectiveness, predicting disease recurrence, and guiding further clinical decisions. CT is widely used to assess structural and functional changes in tumors following interventions such as surgery, chemotherapy, ra-

[Figure 28]

[Figure 29]

[Figure 30]

[Figure 31]

[Figure 32]

[Figure 33]

Forward Tumor Dynamics

[Figure 34]

[Figure 35]

[Figure 36]

[Figure 37]

Simulated

[Figure 38]

Pre-Treatment

[Figure 39]

[Figure 40]

[Figure 41]

Post-Treatment Action Combo

[Figure 42]

Risk Score

[Figure 43]

[Figure 44]

[Figure 45]

[Figure 46]

[Figure 47]

[Figure 48]

[Figure 49]

[Figure 50]

[Figure 51]

[Figure 52]

[Figure 53]

- 1. 1.
- 2.

- 2.

3.

- 3.

- 1.
- 2.

Cisplatin PVA

Raltitrexed Lipiodol Gelatin Sponge

- 3. LobaplatinPVA

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

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

[Figure 70]

[Figure 71]

[Figure 72]

[Figure 73]

[Figure 74]

[Figure 75]

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

[Figure 82]

[Figure 83]

[Figure 84]

Treatment Action Base Drugs Embolisms

[Figure 85]

[Figure 86]

[Figure 87]

[Figure 88]

[Figure 89]

Survival Analysis Model <Heuristic Function>

Tumor Generative Model <Dynamics Model>

[Figure 90]

[Figure 91]

[Figure 92]

[Figure 93]

[Figure 94]

GPTs

<Policies>

[Figure 95]

[Figure 96]

[Figure 97]

[Figure 98]

[Figure 99]

Recommend a treatment plan for the HCC patient, given the pre-treatment CT.

[Figure 100]

Raltitrexed Lipiodol Gelatin Sponge

[Figure 101]

###### Inverse Dynamics

2.

Optimal Treatment

- Figure 2. Overview of TACE Protocol Exploration by Medical World Model. (1) GPTs (Policy Model): construct the TACE action combos by the observation of pre-treatment CT, integrating clinical guidelines and policies. (2) Tumor Generative Model (Dynamics Model): simulates post-treatment tumor based on different TACE intervention protocols, predicting treatment outcomes. (3) Survival Analysis Model (Heuristic Function): assesses risk scores from both simulated post-treatment CT and pre-treatment CT to determine the most effective TACE protocol. Note that the 3D tumor masks (colored in red) can be extracted using a well-trained segmentation network (as Assistant Model). The framework enables visually trackable protocol optimization by iterating between clinical policy guidance, generative modeling, and survival analysis.

#### 3.1. Policy Model

diation therapy, transarterial chemoembolization (TACE), and immunotherapy [28, 36, 44, 48, 72, 73]. Lee et al. [44] employ a CNN-based model to predict the post-treatment survival of patients with hepatocellular carcinoma (HCC) using CT images and clinical information. In addition, LLMs are increasingly being explored to assist in clinical decision-making [11, 29, 40, 46]. However, little attention has been given to applying LLMs for post-treatment prognosis. They did not leverage the feedback from survival analysis to achieve prompt intervention as well.

Vision-language models [13, 77] have emerged as a powerful source of prior knowledge about the clinical world, providing rich information about how to complete promising treatment from large-scale internet data and clinical guidelines. Based on TACE clinical guidelines, we set up the exploration configurations, including all potential chemotherapy drugs (e.g., Raltitrexed, Cisplatin) and embolism materials (e.g., Lipiodol, Gelatin Sponge). The two parts constitute the action base, which provides possible TACE protocols for Generative Dynamics Models as conditions. Then, we adopt a pre-trained large multimodal model (LMM), e.g., GPT-4o, to serve as policies. Given a highlevel goal g (e.g., “What TACE treatment protocols are recommended for a patient diagnosed with hepatocellular carcinoma (HCC) given the pre-treatment CT?”), the policy model πVLM(x0,g) extracts the visual observation and tumor context from the given pre-treatment CT x0 to prompt the proper Transarterial Chemoembolization (TACE) actions. To constrain the excessively large tree search in the action base, we further prompt the Large Language Reasoning Model, i.e., Deepseek-R1 [25], to refine the drug set and embolism set by the clinical policies, whose final cardinalities are D and E, respectively.

### 3. Medical World Models

Overall Framework. As shown in Fig. 2, our MeWM takes a visual observation of pre-treatment CT x0, a language treatment goal g to simulate the future state and explore the best treatment protocol. Policy model (§ 3.1) acquires the descriptive observation based on the visual state, and constructs a set of treatment protocols by the language goal and clinical guidelines. To perform the exploration, given the pre-treatment CT and an action combo, the dynamics model (§ 3.2) predicts the concrete resulting state, i.e., generating post-treatment CT. Finally, inverse dynamics model (§ 3.3) driven by Heuristic Function predicts the risk score from pre-treatment CT and simulated post-treatment CT with tumor masks from Assistant Model, to effectively prune branches in search and heuristically determine the optimal solution.

#### 3.2. Dynamics Model

Radiotherapy Report Extraction and Generation. While most existing studies focus on human-authored radiology reports, we aim to address radiotherapy reports to extract

[Figure 102]

(a) Radiotherapy Report Extraction and Generation

|[Figure 103]<br><br>GPT-4o Deepseek-R1<br><br>[Figure 104]<br><br>Keywords Extraction|
|---|

|[Figure 105]<br><br>GPT-4o Deepseek-R1<br><br>[Figure 106]<br><br>Action Generation|
|---|

Raltitrexed 4 mg; Adasone 30 mg; 100 ml Oxaliplatin; 10 ml ultra-liquid Lipiodol; Gelatin Sponge

Routine pre-procedural preparation for interventional surgery was completed. The patient was placed in the supine position. After sterilization and draping, the right femoral artery was punctured under local anesthesia, and a 5F catheter sheath was inserted …… The following agents were infused via the celiac trunk: Raltitrexed (4 mg), Idarubicin (30 mg), and Oxaliplatin (100 mL). …… Under fluoroscopic guidance, a slow embolization was performed using a mixture of approximately 9 mL of Lipiodol emulsion (10 mL Lipiodol and 10 mg Idarubicin) combined with an appropriate amount of gelatin sponge particles…… The final angiography showed occlusion of the tumor's blood supply artery. After the operation, the catheter was withdrawn, the catheter sheath was removed, the right femoral artery was compressed to stop bleeding, and pressure bandage was applied. The operation was successfully completed.

Pre-Treatment CT

|[Figure 107]| |
|---|---|
|Text-d -Gaussi|riven an|
|[Figure 108]| |
| | |

Raltitrexed 4 mg, Idarubicin 3 0 m g , a n d 1 0 0 m l Oxaliplatin were infused through the catheter;

10 ml ultra-liquid Lipiodol and 10 mg Idarubicin were mixed to create an emulsion for embolization;

9 ml of the emulsion was slowly injected under fluoroscopicAn appropriateguidance; amount of

[Figure 109]

[Figure 110]

Gelatin Sponge was used for embolization;

Morph Attenuation

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

|[Figure 117]|
|---|

[Figure 118]

Action Encoder

[Figure 119]

|[Figure 120]|
|---|

[Figure 121]

Raltitrexed 4mg Epirubicin 5mg Lipiodol 5ml

[Figure 122]

[Figure 123]

CT Encoder

Action Fusion

[Figure 124]

[Figure 125]

|[Figure 126]|
|---|

[Figure 127]

[Figure 128]

[Figure 129]

[Figure 130]

Push Pull

[Figure 131]

[Figure 132]

Action-driven 3D Diffusion Model

|[Figure 133]|
|---|

Lobaplatin20mg Idarubicin20mg

[Figure 134]

[Figure 135]

ml

[Figure 136]

Sponge

10

[Figure 137]

CT Decoder

Lipiodol

[Figure 138]

EpirubicinRaltitrexed4mg Lipiodol1010mgml

[Figure 139]

Gelatin

|[Figure 140]|
|---|

|[Figure 141]|
|---|

|[Figure 142]|
|---|

Synthesized Post-Treatment CT

(c) Combo Contrastive Learning (b) Post-Treatment Tumor Generation

- Figure 3. Dynamics Model based on Tumor Generative Model. The training framework consists of three parts: (a) Radiotherapy Report Extraction and Generation: GPT-4o and Deepseek-R1 extract key treatment details from radiotherapy reports and generate corresponding TACE surgical actions. (b) Post-Treatment Tumor Generation: An Action-driven 3D Diffusion Model is conditioned by fused action embeddings and attenuated CT features to generate post-treatment tumors that simulate treatment outcomes. (c) Combo Contrastive Learning (CCL): The model learns from treatment variations by pushing apart dissimilar combos and pulling together similar ones, improving its ability to generate realistic and action-aware post-treatment tumor appearances.

a pair of tumor-present pre-treatment CT volume x0 and the mask of its tumor region m0, we condition the denoising model on the masked pre-treatment latent representation z

more comprehensive information on treatment protocols. However, raw radiotherapy reports pose significant challenges due to noise and fragmented information, which hinder controlled tumor synthesis. To mitigate these issues, we propose a two-stage text preprocessing framework consisting of data cleaning and augmentation. In the first stage, we perform keyword extraction by aggregating the outputs of both GPT-4o and DeepSeek-R1, focusing on key entities such as drugs, embolic agents, and their corresponding dosages. In the second stage, we leverage the same tools for text generation, constructing a structured core action description based on the extracted keywords. This approach enhances the consistency and informativeness of processed reports, facilitating downstream tasks in tumor synthesis and treatment analysis.

′

′

′

0 is the attenuated mask from m0 by our proposed Text-driven Morpho-Gaussian Attenuation. Specifically, to mimic the effects of TACE treatment, the process begins with occlusion assessment on radiotherapy reports. The textual descriptions (e.g., occluded, reduced, disappear) are extracted and analyzed to determine the attenuation level l ∈ {1,2,3,4}, where a higher value corresponds to better tumor curative effects. Then, morphological erosion and dilation with the adaptive kernel by l are applied to m0, simulating occlusion-induced tumor structural dynamics. Simultaneously, adaptive Gaussian blurring is employed to exhibit the characteristics of heterogeneous intensity changes due to lipiodol deposition, necrotic transformation, and reduced perfusion. The final attenuated mask m

0 ⊙ x0) where m

0 = E(m

Post-Treatment Tumor Generation. We adopt Latent Diffusion Models (LDMs) [58] for latent feature extraction from 3D Pre-treatment CT volumes and integrate textual action embedding for controlled tumor synthesis. Each 3D Post-treatment CT volume x1 ∈ RH×W×D is encoded into a lower dimensional latent representation z1 = E(x1) using a 3D VQGAN autoencoder [22]. In the latent space, following the spirit of DiffTumor [14], we define a diffusion process that progressively adds noise to the latent representation z1 over discrete time steps t = 1,...,T. Given

′

0 is computed by the three steps, ensuring a smooth transition between tumor and organ tissues. Note that this attenuation is only used during training.

Also, we condition the denoising model on the generated textual action. Given the action combo a = {a1,...,aH}, each sub-action, respectively, undergoes encoding through a CLIP [57] text encoder ϕ(·) followed by linear projection σ1(·), enabling dimension reduction to a latent clinical con-

cept space. To enhance the semantics of action conditions in D different drug and E embolism keywords, we introduce learnable concept embeddings c, which extract keyword representations from the given action combo. This explicit pharmaceutical grounding enables precise modeling of therapeutic components while maintaining robustness to context variations. The final action condition τ(a) is the fusion of holistic text embeddings and concept embeddings by fully connected layers σ2, :

τ(a) = σ2([[σ1(ϕ(a1)),...,σ1(ϕ(aH))],c]), (1) where [·] denote concatenation operation.

The training objective of diffusion model is as follows:

′

Ez

1,ϵ∼N(0,1),t ϵ − ϵθ zt,z

0,m0,τ(a),t

2 2

, (2)

where ϵθ(·,t) is a 3D U-Net with interleaved self-attention layers and convolutional layers [14, 32, 56] that predict the noise given the input variable and conditions.

Combo Contrastive Learning. We adopt a contrastive learning strategy that aligns action combos with tumor evolution to enhance the realism and discrimination of posttreatment tumor synthesis. Given a pre-treatment anchor pair (x0,m0), along with an action combo a, the goal is to generate a post-treatment CT xˆ by generative model fDM(·).

For each anchor, positive samples, xˆ+ = fDM(x+0 ,m+0 ,a+), are defined as synthetic post-treatment CT from another pair but its action combo contains the same drug/embolism keywords. Negative samples, xˆ− = fDM(x0,m0,a−), in contrast, are generated using the same pair but action combos with diverse keywords, leading to distinct tumor evolution patterns. This contrastive loss is incorporated into the tumor generative model:

exp(sim(ˆx,xˆ+)/δ) exp(sim(ˆx,xˆ−)/δ) − exp(sim(ˆx,xˆ+)/δ)

E −log

,

(3) where sim(·) denotes cosine similarity, δ is the temperature scaling factor. This contrastive learning strategy ensures that tumors simulated from similar treatment protocols exhibit consistent attenuation effects, while those derived from distinct protocols remain differentiable.

#### 3.3. Inverse Dynamics Model

Inverse Dynamics Model, which empowers our full framework, aims to infer the most effective treatment strategy by analyzing relationships between pre-treatment conditions, intervention actions, and expected post-treatment outcomes. We unfold its essence in three aspects: (1) Assitant Model; (2) Heuristic Function and (3) TACE Protocol Exploration. Assistant Model. To better discriminate the tumor in synthesized post-treatment CT xˆ, we introduce a tumor

segmentation model as Assistant Model Hseg(·). Posttreatment tumors are characterized by heterogeneous highintensity regions due to calcification/lipiodol deposition, irregular shapes reflecting necrotic tissue changes, and reduced or absent contrast enhancement in viable tumor areas, in contrast to traditional pre-treatment CT tumors. Thus, we adapt a pre-trained nnUNet-based [38] model to this posttreatment context by finetuning it on our ground truth pairs of post-treatment CT and mask. Given the well-trained Assistant Model, the simulated post-treatment CT xˆ from Tumor Generative Model is processed for the segmentation of liver and tumor. The post-treatment CT with the predicted mask mˆ is subsequently utilized for survival analysis.

Heuristic Function. We use survival analysis model to implement a heuristic function Hsurv(x0,m0,x,mˆ 0,g), which quantifies the efficacy under the specific TACE action combo by the output risk score. Inspired by DeepSurv [39], we utilize a 3D convolution-based model structure, the 3D ResNet (MC3) [65], as the feature extractor of survival analysis model. Given the pre-treatment pair (x0,m0) and simulated post-treatment pair (ˆx,mˆ ), we extract their concatenated CT and mask, respectively, and bidirectionally align the semantics of pre- and post-treatment by Cross-Attention Transformer [41]. After that, we adopt an attention-based aggregator to fuse pre- and post-features, followed by fc layers to determine the risk score. The action combo with a lower risk score should bring greater efficacy for the patient. Note that, for training, we leverage multi-task learning strategy, i.e., CoxPH [39] and OS regression, to improve the generalization of survival analysis.

TACE Protocol Exploration. Given a combination of the proposed models above, we are able to predict TACE protocol from any Pre-treatment CT x by a language treatment goal g. To reason the optimal action combo, we propose to search for a list of actions to reach g, corresponding to finding a treatment plan consisting of both drug and embolism components, which optimizes:

xˆ∗1:H = arg min

Hsurv(x0,m0,x,Hˆ seg(ˆx),g), (4)

xˆ1:H∼πVLM,fDM

where H = Hd + He.

With this objective in mind, we exhibit a tree-search exploration procedure. Our exploration algorithm initializes a set of B parallel protocol beams. We sample the potential action space composed of D drugs and E embolisms and clinical rules using πVLM. The clinical rules are introduced to prune unreasonable branches, e.g., concomitant use of multiple platinum-based agents is contraindicated due to the risk of cumulative toxicity and myelosuppression. We sequentially explore the two parts to ensure that TACE protocol contains both drugs and embolism. For each current action combo, we synthesize T post-treatment tumors from fDM(x0,m0,a) to obtain a more reliable simulation. We

Algorithm 1 TACE Protocol Exploration with MeWM

- 1: Input: Pre-treatment CT x0, Pre-treatment tumor mask m0, Language treatment goal g
- 2: Functions: VLM Policy Model πVLM, Dynamics Model fDM, Heuristic Function Hsurv, Assistant Model Hseg
- 3: Hyperparameters: Drug Actions factor D, Embolism Actions factor E, Tumor Generative factor T, Protocol Beams B, Drug horizon Hd, Embolism horizon He
- 4: plans ← [[x0] ∀ i ∈ {1 . . . B}] # Initialize B Different TACE Protocol Beams
- 5: drug1:D, embo1:E, rule ← πVLM(x0, g) # Generate D Different Drug, E Embolism Actions, Clinical Rules
- 6: for h = 1 . . . Hd do
- 7: for b = 1 . . . B do
- 8: tumors ← [fDM(x, drugi) for j in (1 . . . T) for i in (1 . . . D) if rule] # Generate tumors from x and plans[b] under rule
- 9: plans[b].append(argmin(tumors, Hsurv, Hseg)) # Add Tumor with Lowest Risk to Plan
- 10: end for
- 11: max idx, min idx ← argmax(plans, Hsurv, Hseg), argmin(plans, Hsurv, Hseg)

- 12: plans[max idx] ← plans[min idx] # Periodically Replace the Plan with High Risk

- 13: end for
- 14: for h = 1 . . . He do
- 15: for b = 1 . . . B do
- 16: tumors ← [fDM(x, emboi) for j in (1 . . . T) for i in (1 . . . E) if rule] # Generate tumors from x and plans[b] under rule
- 17: plans[b].append(argmin(tumors, Hsurv, Hseg)) # Add Tumor with Lowest Risk to Plan
- 18: end for
- 19: max idx, min idx ← argmax(plans, Hsurv, Hseg), argmin(plans, Hsurv, Hseg)

- 20: plans[max idx] ← plans[min idx] # Periodically Replace the Plan with High Risk

- 21: end for
- 22: plan ← argmin(plans, Hsurv, Hseg) # Return Plan with Lowest Risk

then use our heuristic function Hsurv(x0,m0,x,ˆ m,gˆ ) with the assistance of Hseg(ˆx) to select the generated tumor with the best average survival score of T replicas among D or E actions. After every step of extending all beams, we discard the beam with the worst survival score and replace its action combo with the best beam. To prevent cumulative toxicity and organ dysfunction, we prohibit over-exploration by drug horizon Hd and embolism horizon He. Our final action combo is taken from the beam with the best survival score and adopted as TACE protocol for the patient. The pseudocode of our method is also provided in Algorithm 1.

### 4. Experiments

HCC-TACE In-house Dataset. We collect a large repository of 338 longitudinal paired pre- and post-treatment CT scans with well-annotated liver/tumor masks and clinical records, such as TACE radiotherapy reports as gold action and Overall Survival (OS) time. We split the training set (validation set included) and testing sets in the 9:1 ratio.

HCC-TACE-Seg Public Dataset [55]. For external validation, we use patients from HCC-TACE-Seg public dataset referring to a single-institution collection with confirmed HCC treated at The University of Texas MD Anderson Cancer Center. We conduct data curation and preprocessing to collect 78 cases containing pre-treatment CT, posttreatment CT, TACE Gold Action, and OS time. We use 80% cases to fine-tune and validate MeWM and leave 20% cases for the exploration evaluation.

#### 4.1. Evaluation on Generation Quality

Visual Turing Test (Human Evaluation). We conduct an action-driven Visual Turing Test on 240 CT scans of post-treatment tumors, where 120 scans contain real posttreatment tumors, and 120 scans contain synthetic posttreatment tumors generated by different tumor synthesis models. Three radiologists (R1-R3) participated in this study, independently evaluating five groups of 48 CT scans each and classifying them as either real or synthetic. It is important to note that the radiologists’ evaluations are based on whether the synthetic tumor closely resembles a post-treatment tumor, which typically contains a mixture of lipiodol deposition, necrotic, and viable tumor regions, distinguishing it from ordinary pre-treatment tumors. The test results are summarized in Table 1. The sensitivity scores of all radiologists remain high (above 91%), demonstrating their ability to correctly identify real post-treatment tumors. However, specificity scores vary among the methods, indicating different levels of realism in the synthetic tumors. Notably, our method MeWM achieves the lowest specificity scores (79.17% for R1, 70.83% for R2, and 75.00% for R3), suggesting that a large proportion of synthetic tumors generated by our approach are mistaken as real. This indicates superior realism compared to other methods such as SynTumor [34], Pixel2Cancer [42], DiffTumor [14], and TextoMorph [47]. Figure 4 illustrates examples from the test, where a real tumor is compared with synthetic tumors that

- Table 1. Action-driven Visual Turing Test involves three radiologists (R1-R3) each evaluating five groups of 48 CT scans each, with 24 real post-treatment tumors and 24 synthetic post-treatment tumors from a tumor generative model, respectively. They were tasked with categorizing each CT scan as either real or synthetic. A higher sensitivity score indicates better discriminative ability of radiologists, while a lower specificity score indicates a higher number of synthetic tumors being identified as real. We also provide perceptual evaluation using FID and LPIPS compared to corresponding real post-treatment scans. Lower FID and LPIPS indicate better simulation results.

|Methods<br><br>|R1|R2|R3|Perceptual metrics|
|---|---|---|---|---|
| |sensitivity specificity ↓ accuracy|sensitivity specificity ↓ accuracy|sensitivity specificity ↓ accuracy<br><br>|FID↓ LPIPS↓|
|SynTumor [34] Pixel2Cancer [42] DiffTumor [14] TextoMorph [47]|100.0 95.83 97.92 95.83 100.0 97.92 100.0 91.67 95.83 100.0 91.67 95.83<br><br>|87.50 95.83 91.67 91.67 95.83 93.75 95.83 87.50 91.67 91.67 83.33 87.50<br><br>|100.0 95.83 97.92 100.0 100.0 100.0 100.0 87.50 93.75 95.83 87.50 91.67<br><br>|3.33 0.6832<br>3.34 0.6831 1.40 0.7660 1.03 0.9111<br>|
|MeWM (Ours)|100.0 79.17 89.58<br><br>|91.67 70.83 81.25<br><br>|91.67 75.00 83.33<br><br>|0.71 0.6120<br><br>|

[Figure 143]

[Figure 144]

[Figure 145]

[Figure 146]

True Risk Score Cox Model Risk Score Our Model Risk Score

[Figure 147]

[Figure 148]

[Figure 149]

𝒙𝟏

𝒙𝟏

𝒙𝟏

[Figure 150]

[Figure 151]

𝒙𝟎

𝒙𝟎

𝒙𝟎

[Figure 152]

[Figure 153]

Cox Model Error (MSE:0.3550)

Our Model Error (MSE:0.2142)

[Figure 154]

###### Real

𝒙𝟏

𝒙𝟏

- Figure 4. Examples of Visual Turing Test. We present one real tumor alongside examples of synthetic tumors that were correctly and incorrectly identified. A red dot indicates the radiologist classified the post-treatment tumor as synthetic, while a green dot signifies it was identified as real.

𝒙𝟎

𝒙𝟎

- Figure 5. Performance of heuristic function on survival analysis. The first three heatmaps show the true risk distribution, Cox model predictions, and our heuristic function predictions. The last two depict prediction errors, with lower MSE (0.2142) for our model compared to the Cox model (0.3550), demonstrating improved accuracy in capturing localized risk patterns.

[Figure 155]

Radiomics-based Cox Model: c-index=0.472, p-value=9.255e-01 Our Deep Model: c-index=0.752, p-value=6.740e-05

[Figure 156]

SurvivalProbability

Time (months) Time (months)

- Figure 6. Kaplan-Meier Survival Curves: Radiomics-based Cox Model [67] vs. our deep model. The left shows the survival curves predicted by the Cox model based on Radiomics features. The right presents the survival curves from our model based on deep features, which achieves a significantly higher c-index of 0.752 and a log-rank p-value of 6.74e − 5, demonstrating a stronger ability to distinguish between high- and low-risk groups. Shaded areas represent confidence intervals.

radiologists correctly or incorrectly classified. This highlights synthetic tumors closely resemble real post-treatment tumors.

Perceptual Evaluation. We perform perceptual evaluation using FID and LPIPS scores, where lower values indicate better simulation quality. Our method achieves the best FID (0.71) and LPIPS (0.6120), demonstrating the highest fidelity in synthetic tumor generation. These results confirm that MeWM effectively synthesizes realistic post-treatment tumors, making it more challenging for radiologists to distinguish between real and synthetic cases.

#### 4.2. Survival Analysis

In Figure 5, we evaluate survival risk regression between the popular Cox Proportional Hazards model [23] and our heuristic function model on the HCC-TACE-Seg dataset. The true risk distribution (left) is estimated using the Nelson-Aalen estimator [17]. The Cox model fails to accurately distinguish between high- and low-risk samples from low-dimensional deep features, resulting in an overly smoothed risk distribution. In contrast, our model produces a risk map that better aligns with the true distribution, effectively capturing variations in risk levels. Error analysis shows higher Mean Square Error (MSE), i.e., 0.3550 for Cox and lower MSE, 0.2142 for our model, indicating superior accuracy. Figure 6 further presents Kaplan-Meier survival curves comparing risk stratification performance be-

tween the Radiomics-based Cox model and our deep model. These results demonstrate that our heuristic function better estimates survival risks, reduces prediction errors, and captures complex patterns beyond the capabilities of the Cox model and radiomics features.

- Table 2. TACE Protocol Exploration Evaluation on HCC-TACE in-house dataset and public dataset. F1-score, Jaccard index, Precision, and Recall are computed between the predicted action combo and gold action. MeWM significantly advances multimodal GPTs in exploring optimal individualized treatment protocol across all metrics, even comparable to interventional physicians.

|In-house dataset Methods<br><br>F1-score↑ Jaccard↑ Precision↑ Recall↑|Public dataset<br><br>F1-score↑ Jaccard↑ Precision↑ Recall↑|
|---|---|
|Physician w/ Pre-CT 48.81 38.44 46.67 54.67 Physician w/ MeWM 61.51 (+13%) 49.89 60.89 65.44<br><br>|71.43 63.10 66.67 78.57 80.00 (+9%) 73.81 76.16 85.71<br><br>|
|Qwen2.5-VL [3] 37.09 24.49 34.44 41.83 GPT-4o [37] 41.97 27.81 35.93 52.78 Claude-3.7-sonnet [2] 40.93 28.55 45.83 37.78|47.14 34.40 53.57 42.86 44.29 32.74 57.14 38.10 44.76 33.81 64.29 35.71<br><br>|
|CT2Rep [30] 27.75 17.21 30.83 25.83 MedGPT [19, 54] 37.51 25.57 32.78 45.21 HuatuoGPT-Vision [76] 40.13 29.08 40.11 42.28|43.61 28.57 53.57 37.50 47.14 40.48 50.00 45.24 52.62 42.26 54.76 51.19<br><br>|
|MeWM(Ours) 52.38 38.59 63.06 46.17<br><br>|64.08 48.45 72.62 58.93<br><br>|

#### 4.3. Results on TACE Protocol Exploration

Radiologist TACE Protocol

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

Drug: Raltitrexed; Lobaplatin; Idarubicin Embolism: Lipiodol; Gelatin Sponge

Diagnose

Evaluation Strategy. For treatment planning evaluation, we utilize four metrics in Table 2: (1) F1-score: harmonizes Precision and Recall, balancing redundancy and omissions; (2) Jaccard Index: measures prediction overlap with gold actions, emphasizing category-level alignment; (3) Precision: reflects recommendation purity, penalizing incorrect or redundant drugs/embolisms; (4) Recall: captures therapeutic coverage, highlighting critical omissions.

[Figure 162]

[Figure 163]

[Figure 164]

Drug: Raltitrexed; Lobaplatin Embolism: Lipiodol; Gelatin Sponge

[Figure 165]

[Figure 166]

[Figure 167]

[Figure 168]

Post-treatment Simulation tumor

[Figure 169]

[Figure 170]

[Figure 171]

###### Final TACE Protocol

Intervention

Drug: Raltitrexed; Lobaplatin Embolism: Lipiodol

[Figure 172]

MeWM TACE Protocol

Figure 7. Example of MeWM intervention in clinical applications. The radiologist initially proposes a TACE protocol with Raltitrexed, Lobaplatin, Idarubicin, and embolization using Lipiodol and Gelatin Sponge. MeWM simulates a protocol with Raltitrexed, Lobaplatin, and Lipiodol. After intervention, the optimized protocol removes Idarubicin but restores Gelatin Sponge, aligning with gold action.

Partial Observation Misleads GPTs. For Multimodal Large Language Models (e.g., GPT-4o, MedGPT, HuatuoGPT-Vision), they are prompted with pre-treatment CT slices and allowed to predict the action combo from the given action set. These inferior results (over -10% in F1score) to MeWM demonstrate that it tends to make deficient planning relying solely on vision-language models and their commonsense reasoning. This also validates the necessity of simulation from pre-treatment to post-treatment.

Table 3. Ablation studies of TACE protocol exploration on both datasets. “AM” denotes Assistant Model, while “CCL” denotes Combo Contrastive Learning. Two components significantly contribute to better exploring the optimal treatment.

MeWM as A Clinical Decision-support Tool. MeWM demonstrates significant potential in augmenting the capabilities of radiologists and physicians, underscoring its clinical relevance in optimizing TACE planning. Reliance solely on pre-treatment CT often results in partial observation and suboptimal targeting due to heterogeneous pathological conditions. By incorporating MeWM’s recommended protocol, clinical decision-making is markedly enhanced, yielding performance improvements of 12.70, 11.45, 14.22, and 10.77 in F1-score, Jaccard, Precision, and Recall on our dataset. MeWM facilitates accurate tumor localization and enables predictive assessment of postembolization outcomes, thereby reducing procedural uncertainty. Moreover, its synthetic post-treatment CT projections help anticipate embolization efficacy, optimize TACE distribution, and mitigate non-target embolization risks, contributing to enhanced therapeutic precision and individualized strategies. MeWM serves as a critical decisionsupport tool in interventional oncology, bridging anatomi-

|Metrics ↑<br><br>In-house dataset w/o AM w/o CCL<br><br>|Public dataset w/o AM w/o CCL|
|---|---|
|F1-score 49.13 (-3.9) 50.97 (-1.4) Jaccard 35.40 (-3.2) 36.76 (-1.8) Precision 56.39 (-6.7) 60.57 (-2.5) Recall 45.22 (-1.0) 45.36 (-0.8)<br><br>|60.03 (-4.1) 62.90 (-1.2) 45.36 (-3.1) 46.97 (-1.5) 75.00 (+2.4) 70.10 (-2.5) 52.38 (-5.6) 56.72 (-2.2)|

cal imaging with functional assessment for meaningful clinical outcomes. As shown in Figure 7, interventional physicians refine TACE protocols by MeWM intervention, aligning treatment with expert practices.

Ablation Study. As shown in Table 3, we ablate the effectiveness of Assitant Model and Combo Contrastive Learning (CCL) in TACE Protocol Exploration. The results demonstrate that both the Assistant Model and Combo Contrastive Learning (CCL) contribute significantly to its performance. Removing the assistant model, which provides

the location information of tumors for heuristic function, leads to a critical drop in F1-score (52.38→49.13) on our dataset, as well as on public dataset. Similarly, omitting CCL reduces F1-score (e.g., 52.38→50.97), indicating that CCL enhances the model’s discrimination on action units. Overall, MeWM achieves the best results across all metrics, even outperforming radiologists in some areas, validating its effectiveness.

### 5. Conclusion

We present Medical World Model, which marks a step toward AI-driven precision medicine by simulating disease evolution and optimizing clinical strategies. By bridging generative modeling with medical decision-making, it enables a deeper understanding of treatment outcomes and refines intervention planning. The advancements of MeWM in tumor synthesis and survival analysis set the stage for future AI systems that seamlessly integrate with clinical workflows, driving the next generation of longitudinal datadriven healthcare.

### References

- [1] deedsbcv. 2, 5
- [2] AI Anthropic. Claude 3.5 sonnet model card addendum. Claude-3.5 Model Card, 2024. 8
- [3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 8
- [4] Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. Navigation world models. In CVPR, 2025. 2
- [5] Patrick Bilic, Patrick Christ, Hongwei Bran Li, Eugene Vorontsov, Avi Ben-Cohen, Georgios Kaissis, Adi Szeskin, Colin Jacobs, Gabriel Efrain Humpire Mamani, Gabriel Chartrand, et al. The liver tumor segmentation benchmark (lits). Medical image analysis, 84:102680, 2023. 2
- [6] Benjamin Billot, Douglas N Greve, Oula Puonti, Axel Thielscher, Koen Van Leemput, Bruce Fischl, Adrian V Dalca, Juan Eugenio Iglesias, et al. Synthseg: Segmentation of brain mri scans of any contrast and resolution without retraining. Medical image analysis, 86:102789, 2023. 2
- [7] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 2
- [8] Christian Bluethgen, Pierre Chambon, Jean-Benoit Delbrouck, Rogier van der Sluijs, Małgorzata Połacin, Juan Manuel Zambrano Chaves, Tanishq Mathew Abraham, Shivanshu Purohit, Curtis P Langlotz, and Akshay S Chaudhari. A vision–language foundation model for the generation

- of realistic chest x-ray images. Nature Biomedical Engineering, pages 1–13, 2024. 2
- [9] Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021. 1
- [10] Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024. 2
- [11] Felix Busch, Philipp Prucker, Alexander Komenda, Sebastian Ziegelmayer, Marcus R Makowski, Keno K Bressem, and Lisa C Adams. Multilingual feasibility of gpt-4o for automated voice-to-text ct and mri report transcription. European Journal of Radiology, 182:111827, 2025. 3
- [12] M Jorge Cardoso, Wenqi Li, Richard Brown, Nic Ma, Eric Kerfoot, Yiheng Wang, Benjamin Murrey, Andriy Myronenko, Can Zhao, Dong Yang, et al. Monai: An open-source framework for deep learning in healthcare. arXiv preprint arXiv:2211.02701, 2022. 5
- [13] Junying Chen, Chi Gui, Ruyi Ouyang, Anningzhe Gao, Shunian Chen, Guiming Hardy Chen, Xidong Wang, Ruifei Zhang, Zhenyang Cai, Ke Ji, et al. Huatuogpt-vision, towards injecting medical visual knowledge into multimodal llms at scale. arXiv preprint arXiv:2406.19280, 2024. 3
- [14] Qi Chen, Xiaoxi Chen, Haorui Song, Zhiwei Xiong, Alan Yuille, Chen Wei, and Zongwei Zhou. Towards generalizable tumor synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11147–11158, 2024. 2, 4, 5, 6, 7
- [15] Tong Chen, Shuya Yang, Junyi Wang, Long Bai, Hongliang Ren, and Luping Zhou. Surgsora: Decoupled rgbd-flow diffusion model for controllable surgical video generation. arXiv preprint arXiv:2412.14018, 2024. 2
- [16] Zeming Chen, Alejandro Hern´andez Cano, Angelika Romanou, Antoine Bonnet, Kyle Matoba, Francesco Salvi, Matteo Pagliardini, Simin Fan, Andreas K¨opf, Amirkeivan Mohtashami, et al. Meditron-70b: Scaling medical pretraining for large language models. arXiv preprint arXiv:2311.16079, 2023. 1
- [17] Enrico Colosimo, Fla´ vio Ferreira, Maristela Oliveira, and Cleide Sousa. Empirical comparisons between kaplanmeier and nelson-aalen survival function estimators. Journal of Statistical Computation and Simulation, 72(4):299–308,

2002. 7

- [18] Pedro Costa, Adrian Galdran, Maria Ines Meyer, Meindert Niemeijer, Michael Abr`amoff, Ana Maria Mendonc¸a, and Aur´elio Campilho. End-to-end adversarial retinal image synthesis. IEEE transactions on medical imaging, 37(3):781– 791, 2017. 2
- [19] Michael D Moor. Medgpt, 2025. Accessed: March 7, 2025. 8
- [20] Yilun Du, Mengjiao Yang, Pete Florence, Fei Xia, Ayzaan Wahid, Brian Ichter, Pierre Sermanet, Tianhe Yu, Pieter Abbeel, Joshua B Tenenbaum, et al. Video language planning. arXiv preprint arXiv:2310.10625, 2023. 2
- [21] Yilun Du, Mengjiao Yang, Pete Florence, Fei Xia, Ayzaan

- Wahid, Brian Ichter, Pierre Sermanet, Tianhe Yu, Pieter Abbeel, Joshua B Tenenbaum, et al. Video language planning. In ICML, 2024. 2
- [22] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 4
- [23] John Fox and Sanford Weisberg. Cox proportional-hazards regression for survival data. An R and S-PLUS companion to applied regression, 2002, 2002. 7
- [24] Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, and Ishan Misra. Emu video: Factorizing text-to-video generation by explicit image conditioning. arXiv preprint arXiv:2311.10709, 2023. 2
- [25] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025. 3, 5
- [26] David Ha and J¨urgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018. 2
- [27] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023. 2
- [28] Amr Hagag, Ahmed Gomaa, Dominik Kornek, Andreas Maier, Rainer Fietkau, Christoph Bert, Yixing Huang, and Florian Putz. Deep learning for cancer prognosis prediction using portrait photos by stylegan embedding. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 198–208. Springer,

2024. 3

- [29] Paul Hager, Friederike Jungmann, Robbie Holland, Kunal Bhagat, Inga Hubrecht, Manuel Knauer, Jakob Vielhauer, Marcus Makowski, Rickmer Braren, Georgios Kaissis, et al. Evaluation and mitigation of the limitations of large language models in clinical decision-making. Nature medicine, 30(9):2613–2622, 2024. 1, 3
- [30] Ibrahim Ethem Hamamci, Sezgin Er, and Bjoern Menze. Ct2rep: Automated radiology report generation for 3d medical imaging. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 476–

486. Springer, 2024. 8

- [31] Jan B Hinrichs, Hoen-Oh Shin, Daniel Kaercher, Davut Hasdemir, Tim Murray, Till Kaireit, Carolin Lutat, Arndt Vogel, Bernhard C Meyer, Frank K Wacker, et al. Parametric response mapping of contrast-enhanced biphasic ct for evaluating tumour viability of hepatocellular carcinoma after tace. European radiology, 26:3447–3455, 2016. 5
- [32] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020. 2, 5
- [33] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023. 2
- [34] Qixin Hu, Yixiong Chen, Junfei Xiao, Shuwen Sun, Jieneng Chen, Alan L Yuille, and Zongwei Zhou. Label-free liver

- tumor segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7422–7432, 2023. 2, 6, 7
- [35] Pu Huang, Dengwang Li, Zhicheng Jiao, Dongming Wei, Bing Cao, Zhanhao Mo, Qian Wang, Han Zhang, and Dinggang Shen. Common feature learning for brain tumor mri synthesis by context-aware generative adversarial network. Medical Image Analysis, 79:102472, 2022. 2
- [36] Xiaoyu Huang, Yong Huang, Ping Li, and Kai Xu. Ct-based deep learning predicts prognosis in esophageal squamous cell cancer patients receiving immunotherapy combined with chemotherapy. Academic Radiology, 2025. 3
- [37] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024. 1, 8
- [38] Fabian Isensee, Paul F Jaeger, Simon AA Kohl, Jens Petersen, and Klaus H Maier-Hein. nnu-net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2):203–211, 2021. 5, 2
- [39] Jared L Katzman, Uri Shaham, Alexander Cloninger, Jonathan Bates, Tingting Jiang, and Yuval Kluger. Deepsurv: personalized treatment recommender system using a cox proportional hazards deep neural network. BMC medical research methodology, 18:1–12, 2018. 5
- [40] Yubin Kim, Chanwoo Park, Hyewon Jeong, Yik Siu Chan, Xuhai Xu, Daniel McDuff, Hyeonhoon Lee, Marzyeh Ghassemi, Cynthia Breazeal, Hae Park, et al. Mdagents: An adaptive collaboration of llms for medical decisionmaking. Advances in Neural Information Processing Systems, 37:79410–79452, 2024. 3
- [41] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4015–4026, 2023. 5
- [42] Yuxiang Lai, Xiaoxi Chen, Angtian Wang, Alan Yuille, and Zongwei Zhou. From pixel to cancer: Cellular automata in computed tomography. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 36–46. Springer, 2024. 2, 6, 7
- [43] Yann LeCun. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62(1):1–62,

2022. 2

- [44] Kyung Hwa Lee, Jungwook Lee, Gwang Hyeon Choi, Jihye Yun, Jiseon Kang, Jonggi Choi, Kang Mo Kim, and Namkug Kim. Deep learning-based prediction of posttreatment survival in hepatocellular carcinoma patients using pre-treatment ct images and clinical data. Journal of Imaging Informatics in Medicine, pages 1–12, 2024. 3
- [45] Chenxin Li, Hengyu Liu, Yifan Liu, Brandon Y Feng, Wuyang Li, Xinyu Liu, Zhen Chen, Jing Shao, and Yixuan Yuan. Endora: Video generation models as endoscopy simulators. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 230–240. Springer, 2024. 2
- [46] Jia Li, Zichun Zhou, Han Lyu, and Zhenchang Wang. Large language models-powered clinical decision support: enhancing or replacing human expertise?, 2025. 3

- [47] Xinran Li, Yi Shuai, Chen Liu, Qi Chen, Qilong Wu, Pengfei Guo, Dong Yang, Can Zhao, Pedro RAS Bassi, Daguang Xu, et al. Text-driven tumor synthesis. arXiv preprint arXiv:2412.18589, 2024. 2, 6, 7, 5
- [48] Junhao Liang, Weisheng Zhang, Jianghui Yang, Meilong Wu, Qionghai Dai, Hongfang Yin, Ying Xiao, and Lingjie Kong. Deep learning supported discovery of biomarkers for clinical prognosis of liver cancer. Nature Machine Intelligence, 5(4):408–420, 2023. 3
- [49] Jie Liu, Yixiao Zhang, Jie-Neng Chen, Junfei Xiao, Yongyi Lu, Bennett A Landman, Yixuan Yuan, Alan Yuille, Yucheng Tang, and Zongwei Zhou. Clip-driven universal model for organ segmentation and tumor detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21152–21164, 2023. 5
- [50] Jie Liu, Yixiao Zhang, Kang Wang, Mehmet Can Yavuz, Xiaoxi Chen, Yixuan Yuan, Haoliang Li, Yang Yang, Alan Yuille, Yucheng Tang, et al. Universal and extensible language-vision models for organ segmentation and tumor detection from abdominal computed tomography. Medical image analysis, 97:103226, 2024. 5
- [51] Taiming Lu, Tianmin Shu, Alan Yuille, Daniel Khashabi, and Jieneng Chen. Generative world explorer. arXiv preprint arXiv:2411.11844, 2024. 2
- [52] Yunhao Luo and Yilun Du. Grounding video models to actions through goal conditioned exploration. arXiv preprint arXiv:2411.07223, 2024. 2
- [53] Fei Lyu, Mang Ye, Jonathan Frederik Carlsen, Kenny Erleben, Sune Darkner, and Pong C Yuen. Pseudo-label guided image synthesis for semi-supervised covid-19 pneumonia infection segmentation. IEEE Transactions on Medical Imaging, 42(3):797–809, 2022. 2
- [54] Michael Moor, Oishi Banerjee, Zahra Shakeri Hossein Abad, Harlan M Krumholz, Jure Leskovec, Eric J Topol, and Pranav Rajpurkar. Foundation models for generalist medical artificial intelligence. Nature, 616(7956):259–265, 2023. 8
- [55] Ali Morshid, Khaled M Elsayes, Ahmed M Khalaf, Mohab M Elmohr, Justin Yu, Ahmed O Kaseb, Manal Hassan, Armeen Mahvash, Zhihui Wang, John D Hazle, et al. A machine learning model to predict hepatocellular carcinoma response to transcatheter arterial chemoembolization. Radiology: Artificial Intelligence, 1(5):e180021, 2019. 6, 2
- [56] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, pages 8162–8171. PMLR,

2021. 5

- [57] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 4
- [58] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bj¨orn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 2, 4
- [59] Younggyo Seo, Danijar Hafner, Hao Liu, Fangchen Liu,

- Stephen James, Kimin Lee, and Pieter Abbeel. Masked world models for visual control. In Conference on Robot Learning, pages 1332–1344. PMLR, 2023. 2
- [60] Zhuchen Shao, Hao Bian, Yang Chen, Yifeng Wang, Jian Zhang, Xiangyang Ji, et al. Transmil: Transformer based correlated multiple instance learning for whole slide image classification. Advances in neural information processing systems, 34:2136–2147, 2021. 6
- [61] Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. Large language models encode clinical knowledge. Nature, 620(7972):172–180, 2023. 1
- [62] Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Mohamed Amin, Le Hou, Kevin Clark, Stephen R Pfohl, Heather Cole-Lewis, et al. Toward expert-level medical question answering with large language models. Nature Medicine, pages 1–8, 2025. 1
- [63] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International conference on machine learning, pages 2256–2265. PMLR, 2015. 2
- [64] Gemini Team, Rohan Anil, Sebastian Borgeaud, JeanBaptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 1
- [65] Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, and Manohar Paluri. A closer look at spatiotemporal convolutions for action recognition. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pages 6450–6459, 2018. 5
- [66] Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. Mocogan: Decomposing motion and content for video generation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1526–1535,

2018. 2

- [67] Joost van Griethuysen et. al. Radiomics-based cox model,

2025. Accessed: March 7, 2025. 7

- [68] X Wang, Z Zhu, G Huang, X Chen, and J Drivedreamer Lu. Towards real-world-driven world models for autonomous driving. arXiv preprint arXiv:2309.09777, 2023. 2
- [69] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards real-worlddrive world models for autonomous driving. In European Conference on Computer Vision, pages 55–72, 2024. 2
- [70] Philipp Wu, Alejandro Escontrela, Danijar Hafner, Pieter Abbeel, and Ken Goldberg. Daydreamer: World models for physical robot learning. In Conference on robot learning, pages 2226–2240. PMLR, 2023. 2
- [71] Zhaohu Xing, Sicheng Yang, Sixiang Chen, Tian Ye, Yijun Yang, Jing Qin, and Lei Zhu. Cross-conditioned diffusion model for medical image to image translation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 201–211. Springer,

2024. 2

- [72] Jiawen Yao, Yu Shi, Kai Cao, Le Lu, Jianping Lu, Qike Song, Gang Jin, Jing Xiao, Yang Hou, and Ling Zhang.

- Deepprognosis: Preoperative prediction of pancreatic cancer survival and surgical margin via comprehensive understanding of dynamic contrast-enhanced ct imaging and tumor-vascular contact parsing. Medical image analysis, 73:102150, 2021. 3
- [73] Jiawen Yao, Yu Shi, Le Lu, Jing Xiao, and Ling Zhang. Deepprognosis: Preoperative prediction of pancreatic cancer survival and surgical margin via contrast-enhanced ct imaging. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pages 272–282. Springer, 2020. 3
- [74] Qingsong Yao, Li Xiao, Peihang Liu, and S Kevin Zhou. Label-free segmentation of covid-19 lesions in lung ct. IEEE transactions on medical imaging, 40(10):2808–2819, 2021. 2
- [75] Wenfang Yao, Chen Liu, Kejing Yin, William Cheung, and Jing Qin. Addressing asynchronicity in clinical multimodal fusion via individualized chest x-ray generation. Advances in Neural Information Processing Systems, 37:29001–29028,

2025. 2

- [76] Hongbo Zhang, Junying Chen, Feng Jiang, Fei Yu, Zhihong Chen, Jianquan Li, Guiming Chen, Xiangbo Wu, Zhiyi Zhang, Qingying Xiao, et al. Huatuogpt, towards taming language model to be a doctor. arXiv preprint arXiv:2305.15075, 2023. 8
- [77] Yanxin Zheng, Wensheng Gan, Zefeng Chen, Zhenlian Qi, Qian Liang, and Philip S Yu. Large language models for medicine: a survey. International Journal of Machine Learning and Cybernetics, pages 1–26, 2024. 3
- [78] Yi Zhou, Xiaodong He, Shanshan Cui, Fan Zhu, Li Liu, and Ling Shao. High-resolution diabetic retinopathy image synthesis manipulated by grading and lesions. In International conference on medical image computing and computer-assisted intervention, pages 505–513. Springer,

2019. 2

# Appendix

## Table of Contents

- A. Data Preprocessing 2

- A.1. HCC-TACE-Seg dataset preprocessing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
- A.2. HCC-TACE dataset preprocessing . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

- B. Implementation Details 5

- B.1. Policy Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- B.2. Dynamics Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- B.3. Assistant Model . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- B.4. Heuristic Function . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5

- C. Comparison against Multi-Modal GPTs 6

### A. Data Preprocessing

#### A.1. HCC-TACE-Seg dataset preprocessing

The HCC-TACE-Seg dataset [55] refers to a single-institution collection of patients with confirmed hepatocellular carcinoma (HCC) who were treated at The University of Texas MD Anderson Cancer Center. Data preprocessing for HCC-TACE-Seg involves resampling the provided CT images to a standardized spatial resolution while preserving the integrity of the original data structure. Specifically, images and masks are resampled to a target spacing of 0.8mm × 0.8mm × 3.0mm to standardize voxel dimensions across different cases.

Longitudinal Registration: Accurate image registration is essential to ensure that tumor boundaries are clearly defined across both the liver and HCC regions in different imaging modalities, such as arterial phase (AP) and portal venous phase (PVP) scans. The longitudinal registration process involves aligning the post-AP image to the pre-AP image, and the postPVP to the pre-PVP image, addressing any misalignments between scans. Both linear and non-linear registration methods are employed through the open-sourced registration framework deedsBCV [1] for optimal alignment.

Liver and HCC Cancer Segmentation: We utilize a nnUNet-based [38] mode trained on the public LiTS dataset [5] for liver and HCC cancer segmentation. For postprocessing, we adopt connected component analysis to extract the liver and HCC regions precisely. This approach ensures that the tumor and liver boundaries are defined clearly, which is crucial for downstream analysis. An example of pre- and post-treatment CT images, along with liver and tumor segmentation generated from the HCC-TACE-Seg dataset, is shown in Figure 8.

We also conduct a Component Size Filtering strategy. A component size filtering step is applied, with a minimum threshold of 300 voxels, ensuring the accurate identification of tumor and liver regions. This step helps to remove noise or irrelevant small regions, improving the precision of segmentation results. Only the image data paired with the following meta-information are selected for further analysis:

- • Chemotherapy: Information about whether the patient underwent chemotherapy treatment, including details about the type of chemotherapy regimen.
- • Overall survival: Overall survival time in months.
- • Survival status: 0 indicates that the patient is alive or lost to follow-up, while 1 indicates death. Details are in Table 4.

ID 011 ID 025 ID 058

[Figure 173]

[Figure 174]

[Figure 175]

HCC Pre-CT

[Figure 176]

[Figure 177]

[Figure 178]

HCC Post-CT

- Figure 8. Example of HCC-TACE-Seg dataset. The first row shows HCC Pre-CT images, and the second row shows HCC Post-CT images. The red mask represents the liver, while the green mask represents the HCC tumor.

##### Patient ID Chemotherapy Overall Survival (months) Survival Status

HCC 009 Cisplatin; Doxorubicin; Mitomycin; Lipiodol 4.7 1.0 HCC 011 Cisplatin; Doxorubicin; Mitomycin; Lipiodol 19.3 1.0 HCC 025 Cisplatin; Doxorubicin; Mitomycin; Lipiodol 30.0 1.0 HCC 034 Doxorubicin; Lipiodol; LC beads 18.9 1.0 HCC 042 Cisplatin; Mitomycin; Lipiodol 34.1 1.0 HCC 051 Cisplatin; Mitomycin; Lipiodol 12.9 1.0 HCC 058 Cisplatin; Mitomycin; Lipiodol 87.0 0.0 HCC 067 Cisplatin; Doxorubicin; Mitomycin; Lipiodol 90.9 0.0 HCC 079 Doxorubicin; LC beads; Lipiodol 42.5 1.0 HCC 091 Doxorubicin; LC beads; Lipiodol 25.3 0.0

- Table 4. An example of HCC-TACE-Seg dataset metadata, including chemotherapy, overall survival, and survival status. For survival status, a value of 0 indicates that the patient is alive or lost to follow-up, while a value of 1 indicates death.

ID 623595 ID 657727 ID 705855

[Figure 179]

[Figure 180]

[Figure 181]

HCC Pre-CT

[Figure 182]

[Figure 183]

[Figure 184]

HCC Post-CT

- Figure 9. Example of HCC-TACE dataset. The first row shows HCC Pre-CT images, and the second row shows HCC Post-CT images. The red mask represents the liver, while the green mask represents the HCC tumor. In post-treatment CT imaging of HCC, particularly after Transarterial Chemoembolization (TACE), the viable tumor region and its enhancement intensity decrease due to Lipiodol accumulation and treatment-induced necrosis. Lipiodol appears hyperdense (bright) on post-treatment CT, indicating areas that have been successfully embolized.

##### Patient ID Processed Chemotherapy OS (months) Survival Status

HCC 08116730 Raltitrexed 4 mg was infused through the catheter; 5 ml ultra-liquid Lipiodol and 5 mg Epirubicin were mixed to create an emulsion for embolization; the emulsion was slowly injected under fluoroscopic guidance; an appropriate amount of Gelatin Sponge particles was used to embolize the tumor-feeding branches of the S8 segment of the right hepatic artery; Lipiodol deposition in the tumor was satisfactory; tumor-feeding arteries were occluded on the final angiography.

1.4 0.0

HCC 01061677 THP 10 mg was infused through the catheter; 10 mg THP and 10 ml ultra-liquid Lipiodol were mixed to create an emulsion for embolization; 12 ml of the emulsion was slowly injected under fluoroscopic guidance; Lipiodol deposition in the tumor and satellite lesions was satisfactory; tumor-feeding arteries were occluded on the final angiography.

75.7 1.0

84.6 1.0

HCC 01192613 THP 40 mg and 30 ml ultra-liquid Lipiodol, along with a small amount of contrast agent, were mixed to create an emulsion for embolization; 30 ml of the emulsion was slowly injected under fluoroscopic guidance; a small amount of Gelatin Sponge particles was used for embolization; Lipiodol deposition in the tumor was satisfactory; no tumor staining was observed on the final angiography.

17.1 0.0

HCC 01204059 Cisplatin 40 mg was infused through the catheter; 10 ml ultra-liquid Lipiodol was slowly injected under fluoroscopic guidance for embolization of the right hepatic artery tumor-feeding branches; 3 ml ultra-liquid Lipiodol was injected for protective embolization of the segment II branch of the left hepatic artery; Lipiodol deposition in the tumor was acceptable; tumor staining mostly disappeared on the final angiography.

HCC 01532843 Oxaliplatin 100 mg and Epirubicin 30 mg were infused through the catheter; 10 mg Epirubicin and 10 ml ultraliquid Lipiodol were mixed to create an emulsion for embolization; 10 ml of the emulsion was slowly injected under fluoroscopic guidance; an appropriate amount of Gelatin Sponge particles was used to embolize the tumorfeeding branches of the right hepatic artery; Lipiodol deposition in the tumor was satisfactory; tumor staining disappeared on the final angiography.

29.3 1.0

HCC 01532843 Epirubicin 40 mg and Oxaliplatin 100 mg were infused through the catheter; 10 ml ultra-liquid Lipiodol was slowly injected under fluoroscopic guidance for embolization; Lipiodol deposition in the tumor was satisfactory; tumor-feeding arteries were mostly occluded on the final angiography.

21.6 1.0

- Table 5. An example of HCC-TACE dataset metadata, including chemotherapy, overall survival, and survival status. For survival status, a value of 0 indicates that the patient is alive or lost to follow-up, while a value of 1 indicates death.

#### A.2. HCC-TACE dataset preprocessing

The HCC-TACE dataset is a large-scale, self-collected repository containing 338 longitudinal pairs of pre- and post-treatment CT scans, along with well-annotated liver and tumor masks, as well as clinical records. These records include TACE radiotherapy reports (considered the gold action) and Overall Survival (OS) time. Details are presented in Table 5. The dataset is split into training (including validation) and testing sets in a 9:1 ratio. All images and masks are resampled to a target spacing of 0.8mm × 0.8mm × 3.0mm to standardize voxel dimensions across different cases.

Longitudinal Registration: We also employ deedsBCV [1] to align the post-AP image with the pre-AP image, and the post-PVP image with the pre-PVP image, addressing any misalignments between the scans.

Liver and HCC Cancer Annotation: In this dataset, all liver and tumor masks for each CT scan are carefully annotated by radiologists. For postprocessing, we also apply connected component analysis to accurately extract the liver and HCC regions. An example of pre- and post-treatment CT images, along with liver and tumor segmentation generated from the HCC-TACE dataset, is shown in Figure 9.

### B. Implementation Details

#### B.1. Policy Model

We adopt GPT-4o to obtain the initial observation from the given pre-treatment CT scans and collect the individualized potential drugs and embolism during TACE treatment. An example is presented in Figure 10. Then, we refine the action set using DeepSeek-R1 [25], which reasons the clinical conflicts in the current action set and summarizes a better action set using clinical guidelines for individuals (e.g., Multiple platinum-based drugs cannot be used simultaneously).

#### B.2. Dynamics Model

In this study, we implement Dynamics Model by training the corresponding Diffusion Model [47] specifically from pretreatment liver tumors to post-treatment liver tumors. The CT scans are oriented according to specific axcodes and resampled to achieve isotropic spacing of 1.0 × 1.0 × 1.0 mm3. 96 × 96 × 96 patches are randomly cropped around either foreground voxels based on a set ratio. Their intensities are truncated to the range [−175,600] to maintain the discrimination of lipiodol/necrosis/viable areas [31], then linearly normalized to [-1, 1]. We utilize the Adam optimizer with hyperparameters β1 = 0.9 and β2 = 0.999, a learning rate of 0.0001, and a batch size of 10 per GPU. The training is conducted on A6000 GPUs for 2 days, over a total of 2,000 iterations.

#### B.3. Assistant Model

We employ a nnUNet-based [38] segmentation model for the segmentation of liver and tumor in post-treatment CT. As suggested by Chen et al. [14], we generate realistic tumor-like shapes using ellipsoids, and combine these generated tumor masks with the healthy CT volumes to create a range of realistic liver tumors. We pre-train the model on the generated and real tumors for robust generalization. Then, we finetune it on post-treatment CT scans as well as liver and tumor masks. The implementation is in Python, leveraging MONAI*. The CT scans are oriented according to specific axcodes and resampled to achieve isotropic spacing of 1.0 × 1.0 × 1.0 mm3. Their intensities are truncated to the range [−175,600] to maintain the discrimination of lipiodol/necrosis/viable areas, then linearly normalized to [-1, 1]. During training, 96 × 96 × 96 patches are randomly cropped around either foreground or background voxels based on a set ratio. Each patch is subjected to a 90◦ rotation with probability 0.1 and an intensity shift of 0.1 with probability 0.2. To avoid confusing the left and right organs, mirroring augmentation is not used.

The model is initialized with pre-trained liver tumor weights from DiffTumor [14], then fine-tuned on our dataset for 2,000 epochs. We set the base learning rate to 0.0002 and use a batch size of 8, along with a linear warmup and a cosine annealing schedule. Training spans 2 days on eight A6000 GPUs. Additional details on the tumor synthesis process during Segmentation Model training can be found in DiffTumor [14].

For inference, a sliding window strategy with 0.75 overlap is used. Tumor predictions that fall outside their corresponding organs are removed by post-processing with organ pseudo-labels obtained from previous research†.

#### B.4. Heuristic Function

We implement a CNN-based survival analysis model as Heuristic Function. The framework adopts 3D ResNet (MC3) as the backbone and Two-way Transformer as the interaction module of pre-treatment and post-treatment CT features. A multi-

*Cardoso et al. [12]: https://monai.io/ †Liu et al. [49, 50]: https://github.com/ljwztc/CLIP-Driven-Universal-Model

instance aggregator [60] with consecutive fully connected layers is utilized for survival risk scoring. It is implemented on PyTorch using 8 NVIDIA RTX A6000 GPUs. Their intensities are truncated to the range [−175,600] to maintain the discrimination of lipiodol/necrosis/viable areas, then linearly normalized to [-1, 1]. During training, 96 × 96 × 96 patches are randomly cropped around either foreground or background voxels based on a set ratio. Each patch is subjected to a 90◦ rotation with probability 0.1 and an intensity shift of 0.1 with probability 0.2. To avoid confusing the left and right organs, mirroring augmentation is not used. We utilize the Adam optimizer with hyperparameters β1 = 0.9 and β2 = 0.999, a learning rate of 0.00002, and a batch size of 5 per GPU. For the inference of each case, we predict the survival risk scores of

- 5 patches around the foreground and average them to obtain the final score.

### C. Comparison against Multi-Modal GPTs

We carefully design prompt templates for multi-modal GPTs to generate TACE treatment protocol. The first template (Figure 11) is for our dataset with a larger action space, defining the task description, a predefined set of chemotherapy drugs and embolization materials, and an example of input-output in JSON format. The second template (Figure 12) is specifically designed for the HCC-TACE-Seg dataset, featuring a different selection of chemotherapy drugs and embolization materials. Both prompts instruct GPTs to analyze CT images and generate an appropriate TACE treatment plan, submitting results in JSON format with predefined keywords.

|You are a radiation oncologist, please **list potential TACE drug and embolism sets** based on the patient's pre-treatment CT image. Please follow the below guidelines:<br><br>--###* * Task Description**<br><br>1. Analyze the input CT images and output potential TACE chemotherapy drug and embolization material sets for treatment. You can include any drugs and embolisms that you think may be helpful for the treatment. Chemotherapy drugs and embolization materials are limited to those in the Action Base.<br>2. The TACE action set is output in JSON format, including treatment plan keywords such as chemotherapy drugs and embolization materials.<br><br><br>--### **Action Base** #### **Chemotherapy Drugs**<br><br>- Raltitrexed<br>- Epirubicin<br>- Oxaliplatin<br>- Lobaplatin<br>- Mitomycin<br>- Idarubicin<br>- Nedaplatin<br>- Pirarubicin<br>- Cisplatin<br>- Idarubicin<br>- THP<br>- Hydroxycamptothecin #### ** Embolization Materials **<br>- Lipiodol<br>- Gelatin Sponge<br>- PVA<br>- Absolute Alcohol<br>- NBCA<br>- KMG<br>--### **Example**<br><br><br>**Input** { "image": {image_property} "patient_id": 001 }<br><br>**Output** { 001: { "Chemotherapy Drugs": "Raltitrexed; Lobaplatin; Oxaliplatin; Mitomycin; THP” “Embolization Materials”: “Lipiodol; Gelatin Sponge;PVA;NBCA”} }|
|---|

###### Figure 10. Policy model prompt template for our dataset.

|You are a radiation oncologist, please design a TACE treatment plan based on the patient's pre-treatment CT image. Please follow the below guidelines: --###* * Task Description**<br><br>1. Analyze the input CT images and output the TACE treatment plan that you think is appropriate for the patient. The plan should include chemotherapy drugs and embolization materials that comply with clinical guidelines. Chemotherapy drugs and embolization materials are limited to those in the Action Set.<br>2. The TACE treatment plan is output in JSON format, including treatment plan keywords such as chemotherapy drugs and embolization materials.<br><br><br>--### **Action Set** #### **Chemotherapy Drugs**<br><br>- Raltitrexed<br>- Epirubicin<br>- Oxaliplatin<br>- Lobaplatin<br>- Mitomycin<br>- Idarubicin<br>- Nedaplatin<br>- Pirarubicin<br>- Cisplatin<br>- Idarubicin<br>- THP #### ** Embolization Materials **<br>- Lipiodol<br>- Gelatin Sponge<br>- PVA<br>- Absolute Alcohol<br>- NBCA<br>--### **Example**<br><br><br>**Input** { "image": {image_property} "patient_id": 001 }<br><br>**Output** { 001: { "keywords": "Raltitrexed; Lobaplatin; Lipiodol; Gelatin Sponge"} }|
|---|

###### Figure 11. VLM prompt template for our dataset.

|You are a radiation oncologist, please design a TACE treatment plan based on the patient's pre-treatment CT image. Please follow the below guidelines: --###* * Task Description**<br><br>1. Analyze the input CT images and output the TACE treatment plan that you think is appropriate for the patient. The plan should include chemotherapy drugs and embolization materials that comply with clinical guidelines. Chemotherapy drugs and embolization materials are limited to those in the Action Set.<br>2. The TACE treatment plan is output in JSON format, including treatment plan keywords such as chemotherapy drugs and embolization materials.<br><br><br>--### **Action Set** #### **Chemotherapy Drugs**<br><br>- Mitomycin<br>- Doxorubicin<br>- Cisplatin #### ** Embolization Materials **<br>- Lipiodol<br>- LC Beads<br>--### **Example**<br><br><br>**Input** { "image": {image_property} "patient_id": 001 }<br><br>**Output** { 001: { "keywords": " Doxorubicin; LC Beads"} }|
|---|

###### Figure 12. VLM prompt template for HCC-TACE-Seg.

