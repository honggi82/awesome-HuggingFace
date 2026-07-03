Tstars-Tryon 1.0 May 12, 2026

[Figure 1]

## Tstars-Tryon 1.0: Robust and Realistic Virtual Try-On for Diverse Fashion Items

Pailitao Team: authors are listed in alphabetical order

Mengting Chen∗, Zhengrui Chen, Yongchao Du, Zuan Gao, Taihang Hu, Jinsong Lan,

Chao Lin, Yefeng Shen, Xingjian Wang, Zhao Wang, Zhengtao Wu, Xiaoli Xu, Zhengze Xu, Hao Yan, Mingzhou Zhang, Jun Zheng, Qinye Zhou, Xiaoyong Zhu, Bo Zheng†

#### Alibaba Group

∗Project Lead †Corresponding Author

[Figure 2]

Taobao App

Tstars-VTON Bench (HuggingFace) Tstars-VTON Bench (ModelScope)

[Figure 3]

# arXiv:2604.19748v3[cs.CV]10May2026

### Abstract

Recent advances in image generation and editing have opened up new opportunities for virtual try-on applications. However, existing methods still struggle to meet the diverse and complex user demands in real-world scenarios. In this work, we present TstarsTryon 1.0, a commercial-scale virtual try-on system that is robust, realistic, versatile, and highly efficient. First, our system maintains a high success rate across challenging real-world cases, including extreme poses, severe illumination variation, motion blur, and other hard in-the-wild conditions. Second, it delivers highly photorealistic results with rich fine-grained details, faithfully preserving garment texture, material properties, and structural characteristics, while largely avoiding the synthetic artifacts often seen in AI-generated images. Third, beyond apparel try-on, our model serves as a generalpurpose framework that supports flexible multi-image composition (up to 6 reference images) across 8 categories of fashion items: tops, pants, skirts, dresses, coats, shoes, bags, and hats, together with coordinated control over person identity and background content. Fourth, to overcome the latency bottlenecks of commercial deployment, our system is heavily optimized for inference speed, delivering the near real-time generation required for a seamless and interactive user experience. These capabilities are enabled by an integrated system design spanning end-to-end model architecture, a scalable data engine, robust infrastructure, and a carefully engineered multi-stage training paradigm. Extensive evaluation and large-scale product deployment demonstrate that Tstars-Tryon 1.0 achieves leading overall performance. To support future research and development, we also plan to release a comprehensive benchmark. The model has been further extended and deployed at an industrial scale on the Taobao App, serving millions of users with tens of millions of try-on requests. It effectively addresses the long-standing cost–quality trade-off in e-commerce virtual try-on.

###### Single-Garment Try-On

###### Multi-Garment Try-On

Overall

Overall

Phys.& Struc. Logic

Identity Consist.

Phys.& Struc. Logic

Identity Consist.

Backgr. Preserv.

Garment Fidelity

Backgr. Preserv.

Garment Fidelity

Figure 1: Overall comparisons with state-of-the-art models.

[Figure 4]

[Figure 5]

[Figure 6]

[Figure 7]

[Figure 8]

[Figure 9]

|[Figure 10]|
|---|

[Figure 11]

[Figure 12]

[Figure 13]

|[Figure 14]|
|---|

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

[Figure 54]

[Figure 55]

[Figure 56]

#### Figure 2: Tstars-Tryon 1.0 supports robust and realistic virtual try-on in the wild.

|Complex lighting|
|---|

|Complex pose|
|---|

[Figure 57]

[Figure 58]

[Figure 59]

[Figure 60]

[Figure 61]

[Figure 62]

[Figure 63]

|[Figure 64]|
|---|

[Figure 65]

[Figure 66]

[Figure 67]

[Figure 68]

[Figure 69]

|Special scene|
|---|

|Multiple people|
|---|

|Pet model|
|---|

[Figure 70]

[Figure 71]

[Figure 72]

|[Figure 73]|
|---|

[Figure 74]

[Figure 75]

|Special figure|
|---|

[Figure 76]

[Figure 77]

[Figure 78]

[Figure 79]

[Figure 80]

[Figure 81]

|3D human| | |
|---|---|---|
|[Figure 82]|[Figure 83]| |

[Figure 84]

[Figure 85]

[Figure 86]

|Doll model|
|---|

[Figure 87]

[Figure 88]

[Figure 89]

|[Figure 90]<br><br>[Figure 91]|
|---|

[Figure 92]

[Figure 93]

[Figure 94]

[Figure 95]

[Figure 96]

Complex garment

[Figure 97]

[Figure 98]

|[Figure 99]|
|---|

[Figure 100]

|Anime|
|---|

#### Figure 3: Tstars-Tryon 1.0 supports multiple challenging extreme and complex scenarios virtual try-on.

### 1 Introduction

Virtual try-on has emerged as one of the most compelling applications of generative AI, promising to reshape the e-commerce experience by allowing users to visualize garments on themselves before purchase. An ideal system should handle arbitrary user photos, faithfully preserve garment details, support flexible multi-item styling, and deliver results in near real-time, setting a high bar that demands continued advancement in generative modeling.

Recent advances in diffusion-based generation (Rombach et al., 2022; Ho et al., 2020; Esser et al., 2024) have substantially accelerated progress toward an ideal virtual try-on system. On the one hand, powerful general-purpose image editing models, both proprietary ones (e.g., Nano Banana Pro (Google Blog,

- 2025), GPT-Image-1.5 (OpenAI, 2025), Seedream 5.0 lite (ByteDance, 2026)), GPT-Image-2 (OpenAI,
- 2026), and open-sourced solutions (e.g., QwenEdit-2511 (Wu et al., 2025), Flux-kelin (Black Forest Labs, 2025), FireRed-Image-Edit (Team, 2026; Cao et al., 2025), exhibits remarkable capabilities in complex semantic understanding and high-fidelity manipulation. These models can directly support virtual try-on tasks or serve as robust foundation models for task-specific fine-tuning. On the other hand, academia keeps exploring task-specific try-on models from various perspectives, such as spatial alignment and identity preservation (Chong et al., 2024; Jiang et al., 2024; Xu et al., 2025). Collectively, these synergistic explorations provide vital opportunities and a strong foundation for the commercialization of virtual try-on technologies.

However, moving toward true commercial-grade applications remains challenging. First, commercial systems demand rigorous robustness to seamlessly process diverse, in-the-wild user photos, which frequently feature extreme poses, overexposure, unconventional angles, and complex background scenes. Second, unprecedented realism is crucial, necessitating the extreme restoration and exact preservation of intricate garment details and fabrics. Third, true flexibility requires moving beyond single-item scenarios to seamlessly support multi-image inputs, cross-category generation, and complex layering or outfit combinations. Finally, real-world deployment imposes stringent demands on inference speed, requiring near real-time generation to provide instant feedback and ensure a seamless user experience. Given these demanding criteria, existing methods still exhibit a notable gap toward realizing true commercial-grade applications.

Unified Multi-Image Editing DiT

Optimized Prompt

User Prompt

Text-Encoder Output Image

Prompt Rewriter

Inference Stage

High-Quality Vertical Domain Data Supevised Fine-Tuning

Pre-training for general editing

Progressive Resolution Continuous Training

Reinforcement Learning with Multi-Reward

Few-step & CFG distillation

Training Stage

#### Figure 4: Overview of training and inference pipeline of Tstars-Tryon 1.0.

To overcome these limitations, we reformulate the full-stack pipeline for a commercial-level foundation model, from data curation, model architecture, to training strategies and inference optimization. The overall framework is shown in Figure 4:

- • Data Engine: To address the scarcity of multi-item try-on data, we built an automated pipeline for largescale, high-quality image editing datasets. This workflow integrates image element decomposition and retrieval-based recall systems to build a robust data pool. We utilize customized captioners for professional-grade descriptions, further refined by knowledge-enhanced Vision Language Model post-filtering and extensive perceptual metric screening.
- • Model Architecture: Moving away from traditional inpainting logic, we treat virtual try-on as a specialized image editing task. Tstars-Tryon 1.0 utilizes a unified MMDiT (Esser et al., 2024) architecture capable of simultaneously processing and coordinating multiple reference images, ensuring the natural fusion of full-body outfits.
- • Training Infra: Our framework natively supports variable resolutions and an arbitrary number of reference images. By leveraging Data Parallelism, Tensor Parallelism, and adapting Data Packing strategies (Dehghani et al., 2023) for Diffusion Transformers, we have eliminated the computational waste typically associated with traditional bucketing strategies.

Performance vs Latency (Single-Garment Try-On) Performance vs Latency (Multi-Garment Try-On)

| | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |

Figure 5: Performance and latency evaluation on the Tstars-VTON Benchmark. Tstars-Tryon 1.0 achieves optimal performance in the single-garment scenario with a rapid 3.92s latency. For complex multi-garment try-on (5 reference images in average), it still delivers outputs in just 6.74s. Meanwhile, top open-source models (QwenEdit-2511(Wu et al., 2025), Flux.2 dev(Black Forest Labs, 2025)) take ∼200s. Note: Tested on an H200 GPU. Closed-source model times are estimated via API calls and may include network latency (for reference only).

- • Meticulous Training Strategies: During pre-training, we utilize task-balanced and content-balanced datasets with a progressive difficulty scaling strategy to bolster the model’s world knowledge and general editing capabilities. We further apply progressive resolution continuous training to enhance high-resolution synthesis. In the high-quality SFT stage, we curate and balance vertical domain data, while guiding the training process through comprehensive metric monitoring. During reinforcement learning, we perform group-level trajectory sampling and use a multi-dimensional reward pipeline to estimate each sample’s group-relative advantage. Built upon the SFT checkpoint, the policy is further optimized with DiffusionNFT (Zheng et al., 2025) to favor positive trajectories over negative ones. This stage yields strong CFG-free inference performance and further improves garment consistency, outfit quality, and generation stability, even under complex human poses or intricate garment designs.
- • Prompt Enhancement: We introduced a tailored rewriter model to enhance semantic features. This model accurately identifies and describes complex virtual try-on editing processes, providing precise semantic guidance that enhances the detail and accuracy of the final output.
- • Fast Inference Acceleration: To meet the low-latency demands of live business environments, TstarsTryon 1.0 primary DiT model is streamlined to 5B parameters. By combining CFG (Classifier-Free Guidance) distillation and Step Distillation (Yin et al., 2024), we have achieved just 3.92 seconds for single-garment and 6.74 seconds for multi-garment try-on (5 reference images in average) without compromising visual fidelity. See Figure 5 for details.
- • Business-Centric Protocol: We developed Tstars-VTON Benchmark, a comprehensive evaluation suite to validate commercial value. This framework covers a vast array of model body types and all product categories, simulating real-world performance across a global user base and inventory.

Relying on the underlying technical breakthroughs, this model has demonstrated powerful application scalability in actual business scenarios:

- • Extreme Robustness: Breaks through the limitations of body type and style, offering flawless support for diverse human poses, extreme lighting conditions, and any combination of complex garments in the wild.
- • High-Fidelity Realism: Deeply restores the unique ID and intricate material textures of clothing, ensuring that the final rendering is physically accurate, free of synthetic artifacts, and naturally "wearable."
- • Unprecedented Flexibility: Features industry-leading multi-item generation across eight categories (tops, pants, skirts, dresses, shoes, bags, hats, and coats). Furthermore, it transcends physical boundaries by supporting non-photorealistic inputs (e.g., digital humans or anime characters) and seamlessly integrates with complex general image editing tasks.
- • High-Efficiency Inference: Designed for high-concurrency online scenarios, Tstars-Tryon 1.0 optimized architecture significantly reduces VRAM usage and latency while maintaining lossless image quality, delivering the near real-time generation required for seamless commercial deployment.

Stage 1: Data Collection Stage 2: Data Filtering, Refinement, and Anonymization

Stage 3: Try-on Pairing Strategy

[Figure 101]

[Figure 102]

Model Data Cloth Data

[Figure 103]

[Figure 104]

[Figure 105]

[Figure 106]

[Figure 107]

Quality Filtering with Hierarchical policy

[Figure 108]

[Figure 109]

[Figure 110]

[Figure 111]

[Figure 112]

[Figure 113]

[Figure 114]

[Figure 115]

[Figure 116]

[Figure 117]

[Figure 118]

[Figure 119]

[Figure 120]

[Figure 121]

[Figure 122]

[Figure 123]

[Figure 124]

Shared Polity

Blur Image Low Resolution Watermark ……

Internet Data E-commerce Data

Pairing Policy

[Figure 125]

[Figure 126]

[Figure 127]

|[Figure 128]<br><br>|Upper|
|---|
<br><br>[Figure 129]<br><br>[Figure 130]<br><br>[Figure 131]<br><br>[Figure 132]<br><br>[Figure 133]<br><br>Coat Shoes Bag Hat<br><br>|Lower|
|---|
| | |
|---|---|---|
| | | |

Model-domain Policy Cloth-domain Policy

[Figure 134]

Expert-Tagged Data Retrieval

[Figure 135]

VLM Judge

[Figure 136]

[Figure 137]

[Figure 138]

Incomplete

[Figure 139]

| |
|---|

Pose > 90°

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

Non-human … …

Multi-subject … …

[Figure 152]

[Figure 153]

[Figure 154]

Expert Check

[Figure 155]

[Figure 156]

[Figure 157]

[Figure 158]

[Figure 159]

[Figure 160]

[Figure 161]

[Figure 162]

Multi-cloth Try-on Multi-Layered Diverse Pairs

[Figure 163]

[Figure 164]

Tag List

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

Automated & Manual Retrieval

[Figure 178]

[Figure 179]

[Figure 180]

[Figure 181]

[Figure 182]

[Figure 183]

Tagging Refinement Privacy Protection

[Figure 184]

[Figure 185]

[Figure 186]

[Figure 187]

[Figure 188]

[Figure 189]

…

…

[Figure 190]

[Figure 191]

[Figure 192]

[Figure 193]

[Figure 194]

[Figure 195]

[Figure 196]

[Figure 197]

[Figure 198]

Garment and Accessory Data

Match

Human-collected Tags Taobao SKU Tags Tag Library

[Figure 199]

[Figure 200]

[Figure 201]

[Figure 202]

Model Data

[Figure 203]

[Figure 204]

|[Figure 205]<br><br>[Figure 206]|
|---|

Face Library

[Figure 207]

Switch

[Figure 208]

[Figure 209]

|[Figure 210]<br><br>[Figure 211]<br><br>[Figure 212]|
|---|

[Figure 213]

Retrieved Raw Data （>500K）

VLM

[Figure 214]

Diverse & Intricate Try-on Benchmark

Quality Check

Dense Attributes

#### Figure 6: Data curation pipeline of Tstars-VTON Benchmark.

Extensive qualitative and quantitative experiments demonstrate the superiority of Tstars-Tryon 1.0. As illustrated by the quantitative comparisons in Figure 1 and Figure 5, our model consistently outperforms top-tier commercial models(Nano Banana Pro(Google Blog, 2025), GPT-Image-1.5OpenAI (2025) and even the latest released model GPT-Image-2(OpenAI, 2026)) across key dimensions in both single- and multigarment try-on tasks, proving that state-of-the-art results can be achieved with significantly reduced computational overhead. Figure 2 showcases the model’s robust capabilities for realistic virtual try-on in the wild, maintaining high visual fidelity across diverse dynamic environments and seamlessly adapting to non-standard subjects such as statues and anime characters. Furthermore, Figure 3 highlights the exceptional versatility of Tstars-Tryon 1.0 in a variety of complex scenarios, proving its ability to handle complex lighting, extreme poses, multiple people, special figures (such as maternity wear), and even cross-domain subjects like pets and dolls. In summary, these results demonstrate that Tstars-Tryon

- 1.0 establishes a new industry-leading standard for highly adaptable and high-fidelity virtual try-on generation.

2 Tstars-VTON Benchmark

To evaluate the model under commercial standards, we introduce Tstars-VTON Benchmark. This benchmark explicitly incorporates the challenges from real applications for rigorous evaluation, such as multi-garment layering, complex background, and diverse human poses. Specifically, we collect large-scale data and refine them into 1780 paired samples across 5 garment categories and 3 accessory categories, covering 465 fine-grained subcategories and 1-6 layered try-on items.

- 2.1 Limitations of Academic Benchmarks

Despite their widespread adoption, existing academic benchmarks exhibit significant limitations that hinder the evaluation of models for real-world deployment. First, they suffer from homogeneous backgrounds and restricted garment categories. Datasets such as VITON-HD (Choi et al., 2021) and DressCode (Morelli et al., 2022) predominantly feature simplistic studio-like backgrounds and confine their scope to basic topological categories, such as upper, lower, and dresses. In contrast, authentic e-commerce and daily-life scenarios encompass complex in-the-wild environments. Furthermore, a comprehensive virtual try-on experience frequently involves a broader array of fashion items, including outerwear, footwear, bags, and accessories (e.g., hats), which are largely overlooked by current benchmarks.

Second, existing datasets fail to reflect the diversity and complexity of reference garment images provided by users in practical applications. Most academic benchmarks are strictly designed for single-garment try-on. While recent efforts, such as DressCode-MR (Chong et al., 2025), attempt to address multigarment scenarios, the reference garment images in these datasets are often artificially extracted from the source model images. More critically, existing benchmarks implicitly assume that reference garments are pristine flat-lay images on simple backgrounds. However, user-provided reference images are highly

[Figure 215]

[Figure 216]

[Figure 217]

[Figure 218]

|[Figure 219]<br><br>[Figure 220]<br><br>[Figure 221]|
|---|

|[Figure 222]<br><br>[Figure 223]|
|---|

|[Figure 224]<br><br>[Figure 225]<br><br>[Figure 226]<br><br>[Figure 227]<br><br>[Figure 228]<br><br>[Figure 229]|
|---|

[Figure 230]

[Figure 231]

[Figure 232]

|[Figure 233]<br><br>[Figure 234]<br><br>[Figure 235]|
|---|

|[Figure 236]<br><br>[Figure 237]<br><br>[Figure 238]|
|---|

[Figure 239]

[Figure 240]

[Figure 241]

[Figure 242]

|[Figure 243]<br><br>[Figure 244]|
|---|

[Figure 245]

[Figure 246]

[Figure 247]

- Figure 7: Clothing statistics of Tstars-VTON Benchmark. The distributions of garments and accessories are illustrated in two separate fan-shaped arrangements with blue and green borders, respectively, while representative images are shown around.

unconstrained in real-world commercial scenarios. The user-provided reference images frequently feature complex backgrounds or are even in-the-wild portrait photos of other individuals.

Consequently, evaluating models on these constrained datasets fails to reflect their true capability in handling the intricate garment combinations and complex reference conditions required for industrial deployment. To comprehensively assess whether a virtual try-on model is truly capable of functioning in real-world scenarios, a new highly practical benchmark is urgently needed.

#### 2.2 Principles of Our Benchmark

To address the limitations above, we construct a new benchmark that truly meets the practical demands with the following distinctive features.

- • Multi-Garment/Accessory Try-On Scenarios: Unlike most prior benchmarks that focus on singlegarment try on (Choi et al., 2021; Morelli et al., 2022), the Tstars-VTON benchmark introduces multigarment combinations with layered outfits and accessories to capture real-world dressing complexity, covering free-combination scenarios with 1-6 items. This substantially increases scenario diversity and enables a more thorough evaluation of generative models in multi-condition controllable synthesis and semantic understanding.
- • Diverse Data Coverage with Fine-Grained Attributes: To overcome data coverage limitation caused by source distribution bias (Choi et al., 2021; Morelli et al., 2022), our benchmark collects diverse data from Internet and E-commerce domains, following by a two-stage tag retrieval and refinement process involving VLM-based generation and manual check. Our benchmark adopts controllable data sampling over multiple tag attributes, including 11 tag dimensions for models and 13 tag dimensions for garments, to realize broad and balanced data coverage.
- • Privacy-Preserving Mechanism: All portraits collected from open-source data are matched to the most similar models in a licensed face database and anonymized through face swapping, following by a rigorous verification to ensure swapping quality.

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

(a) Distribution of model genders. (b) Distribution of model poses. (c) Distribution of scenarios.

- Figure 8: Model statistics of Tstars-VTON benchmark. Pose diversity and scenario variety along with basic model attributes are captured. Each subfigure is supplemented with representative images corresponding to specific attributes.

- • Flexible Unpaired Settings: Tstars-VTON benchmark supports a fully unpaired evaluation setting, decoupling the model and garment databases to maximize combinatorial diversity while maintaining control over attribute pairing process.
- • Comprehensive Evaluation Paradigm Aligned with Human Preferences: We propose a VLM-driven evaluation paradigm that decomposes virtual try-on quality into four rigorous dimensions, each evaluated on a 1-10 Likert scale. This framework supports both single-garment and multi-garment scenarios by assessing Identity Consistency, Garment Fidelity, Background Preservation, Physical and Structural Logic. By integrating human-aligned prompts and interpretable scoring rationales, this protocol bridges automated metrics with fine-grained diagnostic assessment, ensuring the evaluation paradigm meets commercial-grade standards.

#### 2.3 Benchmark Construction

Following above principles, we construct the Tstars-VTON benchmark through a three-stage pipeline to optimize the quality and diversity of paired samples under realistic try-on constraints, as shown in Figure 6.

- Stage 1. Data Collection. To achieve broad data coverage and reduce the distribution biases common in existing benchmarks, we build a comprehensive data pool consisting of two image sets, i.e., diverse human models and a wide range of garments and accessories. We retrieve large-scale data from internet and proprietary e-commerce sources. To better reflect real-world try-on distributions, human experts first design a multi-dimensional tag system. Guided by these tags, we adopt a hybrid retrieval strategy that combines automated platform extraction with targeted manual collection, covering diverse model characteristics, dressing poses, and complex garment/accessory types.
- Stage 2. Data Filtering, Refinement, and Anonymization. After collection, the raw data are rigorously filtered by fine-grained rules and refined to ensure quality. This process combines automated tools, such as a VLM-based pipeline, with human verification. Then each image is assigned rich semantic tags through a multi-step annotation process. Labels are first derived from internet or e-commerce SKU metadata and manually verified. A VLM further refines the annotations and supplements fine-grained attributes, followed by final manual checking to ensure consistency and accuracy.

For privacy protection, all model portraits undergo face swapping. Each face is matched to a licensed surrogate by attributes such as skin tone, gender, and age. Reference-guided swapping improves realism, while failed cases are iteratively corrected through automated filtering and human inspection.

- Stage 3. Try-on Pairing Strategy. The pairing process is aiming to maximize the diversity of matching selections, while strictly adhering to realistic physical and semantic rules. Beyond simple constraints such as gender-matching, the outfit combinations follow a structured layering logic, which enforces unique image utilization and establishes precise coexistence protocols. This strategy dynamically generates diverse, physically plausible layered outfits that accurately mirror real-world dressing complexity.

#### 2.4 Evaluation Metrics

To move beyond simplistic automated metrics (e.g., FID (Heusel et al., 2017)) that often fail to capture fine-grained visual defects, we introduce a specialized VLM-driven evaluation protocol. This protocol decomposes try-on quality into four semantically distinct dimensions, each scored on a scale of 1 to 10.

Shoes 534 items 48 styles

Dresses 515 items 110 styles

Skirts 360 items 32 styles

Tops 570 items 70 styles

Outerwear 914 items 103 styles

Pants 210 items 36 styles

Hats 472 items 37 styles

Bags 522 items 29 styles

Garment

Accessory

| |
|---|

| |
|---|

(a) Distribution of garment/accessory sub-categories.

Gender Age

Female 74.9%

Male 25.1%

Youth 85.4%

Child 5.5%

Teen 4.5%

Senior 4.6%

Skin Tone Body Type

Yellow 58.0%

Cool fair 36.0%

Uncertain

Dark 5.4%

- 0.6%

Athletic 3.1%

Pregnant

- 3.1%

Color filter

- 4.9%

Shadow

- 1.1%

Slim 43.4%

Normal 44.4%

Plus-size 6.1%

Pose Occlusion

Complex 29.6%

Normal 62.2%

Simple 8.2%

Foreground 38.8%

Self 48.5%

None 12.7%

Lighting Angle Background Clarity Portrait Size

Natural 84.9%

Overexposed 8.4%

Eye-level 93.3%

High 2.8%

Low 4.0%

B/W 0.7%

Regular 32.9%

Text-heavy 7.2%

Complex 41.1%

Solid 18.8%

Normal 98.9%

Motion blur 0.9%

Low-res 0.2%

Full body 93.0%

Non-full 7.0%

0 20 40 60 80 100

Share within each attribute (%)

(b) Distribution of model characteristics.

- Figure 9: Attribute statistics of Tstars-VTON benchmark. Diversity of clothing and model attributes is shown in sub-figures above.

To maximize evaluative precision and prevent the model from being distracted by irrelevant information, the protocol is executed through two independent API calls with different image inputs:

- • Stage 1: Garment-Aware Evaluation. This stage provides the VLM with the original person, the reference garment(s), and the result. It focuses on the relationship between the target items and the subject.

- – Identity Consistency: This dimension evaluates the preservation of the person’s face, pose, and body shape. It is intentionally included in Stage 1 because the reference garments provide critical context for the person’s silhouette. By seeing the clothing (e.g., a bulky jacket versus a tight top), the VLM can correctly distinguish whether a change in the person’s visible scale or shape is a natural result of the outfit’s style or an actual failure to preserve the subject’s identity.
- – Garment Fidelity: For both single and multi-garment tasks, the VLM evaluates each target item individually based on its category label, ensuring that the silhouette, texture, and complex patterns of the reference are faithfully reproduced without being influenced by other parts of the outfit.

- • Stage 2: Garment-Agnostic Evaluation. In this stage, the VLM only receives the original person and the result image, without the garment references. This isolation allows the model to focus exclusively on background integrity and structural realism.

- – Background Preservation: The VLM first performs a background type classification (Plain vs. Complex). The score is then based on color consistency (for plain backgrounds) or pixel-level content preservation and lighting consistency (for complex backgrounds).
- – Physical and Structural Logic: This dimension monitors anatomical correctness and mesh clipping. To minimize false positives in complex poses, the VLM must perform a second verification pass before flagging limb anomalies. It also checks for interpenetration, such as fabric visibly passing through skin or layers of clothing intersecting in physically impossible ways.

Overall Score Calculation. To synthesize these four dimensions into a single representative metric, we compute the Overall Score using the geometric mean of the individual scores. Unlike the standard arithmetic mean, the geometric mean is highly sensitive to "weak links." This ensures that a model must achieve balanced excellence across all categories.

#### 2.5 Benchmark Statistics

Tstars-VTON benchmark provides diverse try-on pairs whose distributions are consistent with real-world scenarios. Detailed statistics are introduced as follows.

Distribution of Garments and Accessories. To closely mirror realistic try-on complexity, as shown in Figure 7 and left part of Figure 9, the dataset incorporates 5 garment categories and 3 accessory categories, which are further subdivided into totally 465 fine-grained sub-styles. For garments, there are tops with 70 styles, dresses with 110 styles, coats with 103 styles, pants with 36 styles, and skirts with 32 styles. And for accessories, there are shoes with 48 styles, hats with 37 styles, and bags with 29 styles. This diverse collection supports complex multi-item try-on tasks ranging from 1 to 6 items per sample.

Distribution of Model Characteristics. We carefully curated the dataset to incorporate diverse human characteristics, varied poses, and in-the-wild backgrounds, as shown in Figure 8 and right part of Figure 9. In terms of demographic diversity, the dataset includes models across different genders including 74.9%

Table 1: Quantitative results on the Tstars-VTON Benchmark (Single-Garment). The best and secondbest results are demonstrated in bold and underlined, respectively.

Phys. & Struc. Logic Academic SOTA

Identity Consist.

Garment Fidelity

Backgr. Preserv.

Method Overall ↑

CatVTON (Chong et al., 2024) 6.663 9.335 4.007 9.474 7.955 Leffa (Zhou et al., 2024) 6.048 8.135 4.348 8.753 6.009 FitDiT (Jiang et al., 2024) 5.152 6.767 4.706 8.028 3.883 FastFit (Chong et al., 2025) 6.448 9.131 4.672 8.338 6.546

Proprietary / Gen-Edit

QwenEdit-2511 (Wu et al., 2025) 8.121 9.214 6.787 9.168 8.865 FLUX.2-dev (Black Forest Labs, 2025) 8.764 9.419 7.920 9.640 8.960 FLUX.2-klein-9B (Black Forest Labs, 2025) 8.797 9.442 8.183 9.504 8.902 FireRed-Image-Edit-1.1 (Team, 2026) 8.863 9.610 7.796 9.775 9.068 GPT-Image-1.5† (OpenAI, 2025) 8.892 9.381 8.563 9.075 9.219 GPT-Image-2† (OpenAI, 2026) 9.200 9.597 8.794 9.588 9.255 Nano Banana Pro (Google Blog, 2025) 9.229 9.861 8.598 9.816 9.189 Seedream5 lite (Seedream et al., 2025) 9.301 9.854 8.639 9.810 9.343

###### Tstars-Tryon 1.0 9.372 9.889 8.833 9.863 9.241

† Due to platform restrictions, GPT-Image-1.5 failed to generate results for 120 test cases and GPT-Image-2 failed to generate results for 107 test cases. The reported metrics are calculated excluding these missing instances. Both GPT models were queried via the official API with the quality parameter set to high.

female and 25.1% male, and varied age groups featuring 85% youth but also encompassing children, teenagers, and seniors. To further challenge the try-on models, the dataset incorporates 29.6% proportion of complex poses rather than focusing solely on simple poses which only accounts for 8.2%. Furthermore, more than 40% of model images feature highly complex backgrounds for intricate in-the-wild settings.

### 3 Evaluation Results

#### 3.1 Quantitative Results

To further validate the commercial utility of our framework, we conduct a comprehensive evaluation on the Tstars-VTON Benchmark, covering both single-garment and multi-garment scenarios. Unlike traditional academic benchmarks, this dataset presents significant challenges in terms of pose diversity, background complexity, and high-fidelity texture requirements. The results are summarized in Table 1 and Table 2.

Single-Garment Try-On. As shown in Table 1, there is a clear performance hierarchy between specialized academic models and general-purpose editing models. Current academic state-of-the-art (SOTA) models exhibit limited robustness in complex scenarios, often failing to produce high fidelity garments.

While leading proprietary models such as GPT-Image-1.5, Nano Banana Pro, and Seedream5 lite provide strong competition, Tstars-Tryon 1.0 consistently achieves superior or competitive performance across all dimensions. Specifically, Tstars-Tryon 1.0 demonstrates a clear advantage in Garment Fidelity, effectively capturing intricate textures, material drapes, and fine patterns that competing models sometimes blur or over-simplify. Furthermore, our model excels in Identity Consistency and Background Preservation, ensuring that the subject’s facial features and the original environment remain pixel-identical to the source, even when the new garment introduces significant changes to the person’s silhouette.

Multi-Garment Scenarios. The complexity of the try-on task escalates significantly in multi-garment scenarios, where models must coordinate multiple reference items, such as tops, bottoms, and various accessories.

A critical observation from Table 2 is the performance collapse of general-purpose image editing models (e.g., FireRed-Image-Edit-1.1 (Team, 2026), QwenEdit-2511 (Wu et al., 2025)) when transitioning from single to multi-garment tasks. This degradation is primarily attributed to two bottlenecks. First, these models struggle with multi-garment coordination. They frequently omit specific garments or fail to resolve complex layering and occlusion relationships, leading to illogical garment combinations. Second, as the number of visual conditions increases, the task often exceeds the models’ inherent capability boundaries, resulting in catastrophic generative failures where both the person’s identity and the image’s overall structure break down entirely.

#### Table 2: Quantitative results on the Tstars-VTON Benchmark (Multi-Garment).

Phys. & Struc. Logic Open-source

Identity Consist.

Garment Fidelity

Backgr. Preserv.

Method Overall ↑

FastFit (Chong et al., 2025) 6.039 8.163 4.575 8.096 5.847 QwenEdit-2511 (Wu et al., 2025) 6.441 7.274 5.638 7.256 8.235 FLUX.2-dev (Black Forest Labs, 2025) 7.775 7.964 7.797 8.508 8.458 FLUX.2-klein-9B (Black Forest Labs, 2025) 8.161 8.711 7.870 8.979 8.363 FireRed-Image-Edit-1.1 (Team, 2026) 4.822 5.393 4.837 4.879 5.139

Closed-source

GPT-Image-1.5† (OpenAI, 2025) 8.391 8.890 8.577 8.148 9.070 Nano Banana Pro (Google Blog, 2025) 8.540 8.973 8.499 8.952 8.765 Seedream5 lite (Seedream et al., 2025) 8.914 9.272 8.623 9.525 8.880 GPT-Image-2† (OpenAI, 2026) 9.111 9.554 8.823 9.478 9.052

###### Tstars-Tryon 1.0 9.171 9.619 8.955 9.620 8.883

† Due to platform restrictions, GPT-Image-1.5 failed to generate results for 168 test cases and GPT-Image-2 failed to generate results for 134 test cases. The reported metrics are calculated excluding these missing instances. Both GPT models were queried via the official API with the quality parameter set to high.

In contrast, Tstars-Tryon 1.0 maintains remarkable stability and achieves the highest overall results among all tested models. By leveraging our unified MMDiT architecture and specialized multi-garment training pipeline, Tstars-Tryon 1.0 effectively manages the "stress test" of complex multi-piece outfits. It maintains high Garment Fidelity for every individual item and preserves Physical and Structural Logic without anatomical distortion. This demonstrates that Tstars-Tryon 1.0 possesses the advanced visual reasoning required to handle industrial-grade multi-garment coordination, bridging the gap between experimental generation and professional-grade virtual fitting solutions.

Academic Benchmarks. While the results on the Tstars-VTON Benchmark unequivocally demonstrate our model’s superiority in complex real-world scenarios, we further evaluate our framework on academic benchmarks to ensure comprehensive comparability with existing methods. Specifically, we conduct experiments on the two most widely recognized public datasets: VITON-HD (Choi et al., 2021) and DressCode (Morelli et al., 2022).

VITON-HD is a high-resolution dataset specifically focused on upper-body garments, providing a standard testing set of 2,032 pairs. DressCode offers a more diverse and complex setting, featuring full-body person images categorized into upper-body, lower-body, and dresses, comprising 5,400 testing pairs.

In the literature, performance on these datasets is typically evaluated under two distinct protocols: paired and unpaired settings. The paired setting acts as a reconstruction-based assessment, where the model aims to synthesize the original garment back onto the person using a masked image. While this measures inpainting capability, it suffers from an inherent bias toward the ground-truth image and fails to represent real-world utility. Conversely, the unpaired setting requires the model to transfer a different reference garment onto the target person. Because this protocol more faithfully reflects the generalization required for practical virtual try-on applications, we predominantly focus our comparative analysis on the unpaired setting.

As shown in Table 3, Tstars-Tryon 1.0 achieves state-of-the-art quantitative results against specialized academic baselines. Notably, our training dataset does not incorporate any data from either VITON-HD or DressCode. As a result, the superior performance observed on these benchmarks clearly demonstrates the strong zero-shot generalization and robust synthesis capabilities of our model when applied to entirely unseen data distributions.

#### 3.2 Human Evaluation

While quantitative metrics provide a standardized measure of performance, they often fall short of capturing human perceptual preferences, which are the ultimate standard for commercial applications. To complement our quantitative analysis, we conducted a comprehensive human evaluation to assess the subjective visual quality of our generated images.

We benchmarked Tstars-Tryon 1.0 against three strong closed-source competitors identified in the quantitative analysis: Nano Banana Pro, Seedream5 lite, and GPT-Image-2. Evaluators were presented with randomized anonymized image pairs (ours vs. competitor) alongside the reference conditions and were

#### Table 3: Quantitative comparison on VITON-HD and DressCode benchmarks under the unpaired setting.

VITON-HD DressCode FID ↓ KID ↓ FID ↓ KID ↓

Method

Leffa (Zhou et al., 2024) 10.446 2.640 20.099 13.506 CatVTON (Chong et al., 2024) 10.552 2.272 5.872 1.606 FitDiT (Jiang et al., 2024) 9.979 1.478 4.805 0.712 FastFit (Chong et al., 2025) 8.629 0.665 4.397 0.553

#### Tstars-Tryon 1.0 8.485 0.528 4.541 0.458

Tstars-Tryon 1.0 Better

Same

Competitors Better

| |
|---|

| |
|---|

| |
|---|

###### Tstars-Tryon 1.0 vs Nano Banana Pro

###### Tstars-Tryon 1.0 vs Seedream5 lite

Tstars-Tryon 1.0 vs GPT-Image-2

| | | | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |33.6%| | |5| | | | | | |5.1%| | | | | | |11.3%|
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| |42|.2%| | | | | | | | |42.8%| | | | | |15.0%| |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | |48.4%| | | | | | | | |28.8%| | | |22.8%| | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | |47.4%| | | | |20.8%| | | | | | | |31.8%| | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | |54.8%| | | | | | | | |25.0%| | | |20.2%| | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | |52.0%| | | | | | |1| |8.4%| | | |29.6%| | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| |41|.1%| | | | | | | | |41.6%| | | | |17.3%| | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | |46.1%| | | |45.1%| | | | | |8.9%| | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | |59.7%| | | |32.5%| | | | | | |7.8%| | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | |61.5%| | | | |30.4%| | | | | |8.0%| | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | |54.9%| | | |26.0%| | | | |19.1%| | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | |70.2%| | | | | | | |25.0%| | | |4.8%| |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | |66.3%| | | | | | | |30.6%| | | | |3.1%|
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | |54.4%| | | |36.6%| | | | | |9.0%| | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | |

| | | | | | | | | | | | | | | | | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| |36.4%| | | | | | | |50.1%| | | | | |13.5%| |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | |46.1%| | | | | | |38.2%| | | | |15.7%| | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | |49.0%| | | | | | |41.7%| | | | | | |9.3%|
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| |40.|7%| |30.7%| | | | | | | | |28.7%| | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | |50.0%| | | | | |16.3%| | | | |33.7%| | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | |46.7%| | | | | | |32.0%| | | |21.3%| | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| |41|.9%| | | | | | |42.6%| | | | |15.5%| | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |
| | | | | | | | | | | | | | | | | |

- 1 Garment
- 2 Garments
- 3 Garments
- 4 Garments
- 5 Garments
- 6 Garments

Overall

0% 20% 40% 60% 80% 100%

0% 20% 40% 60% 80% 100%

0% 20% 40% 60% 80% 100%

- Figure 10: Human evaluation comparison. GSB Evaluation of Tstars-Tryon 1.0 against Nano Banana Pro, Seedream5 Lite and GPT-Image-2 grouped by the number of reference garments. Tstars-Tryon 1.0 consistently outperforms competitors overall, with its advantage becoming increasingly pronounced as the task complexity (number of garments) escalates.

asked to choose the better result or declare a tie (“Same”). The results are illustrated in Figure 10.

Overall Superiority. As shown in the “Overall” category, Tstars-Tryon 1.0 demonstrates a definitive advantage over all three top-tier commercial models. When compared to Nano Banana Pro, Tstars-Tryon 1.0 is preferred 41.1% of the time, with 41.6% rated as ties, and loses in only 17.3% of cases. Performance against GPT-Image-2 follows a very similar pattern, where Tstars-Tryon 1.0 achieves a 41.9% win rate, 42.6% ties, and only a 15.5% loss rate. Against Seedream5 lite, our advantage is even more striking, achieving a 54.4% win rate compared to a mere 9.0% preference for the competitor. The high proportion of “Same” votes in the overall metrics indicates that while proprietary models can occasionally produce acceptable baseline results, Tstars-Tryon 1.0 consistently pushes the upper bound of visual quality. Notably, although Seedream5 lite achieved slightly higher absolute scores than Nano Banana Pro in our quantitative metrics, it suffers a larger defeat margin here. This divergence occurs because GSB evaluation measures relative preference frequency rather than absolute score magnitude—a marginal visual advantage and a massive quality gap both register identically as a single “Win” in pairwise comparisons.

Robustness in High-Complexity Scenarios. A compelling trend emerges when analyzing human preference across varying numbers of garments. In relatively simple single-garment try-on tasks, competitors manage to maintain a higher rate of “Same” evaluations. However, as the task complexity escalates to coordinate more garments simultaneously, the performance gap widens drastically.

For instance, against Nano Banana Pro, our win rate surges from 33.6% (1 Garment) to a peak of 54.8% (5 Garments). This upward trajectory is consistent across all competitors: against GPT-Image-2, our win rate increases from 36.4% (1 Garment) to 50.0% (5 Garments), and against Seedream5 lite, it jumps from 46.1% (1 Garment) to an overwhelming 70.2% (5 Garments). Correspondingly, the “Same” ratios drop significantly in these complex scenarios. This user study perfectly aligns with our quantitative findings: while existing models struggle to handle the intricate spatial layering, severe occlusions, and multi-condition conditioning required for full-outfit generation, Tstars-Tryon 1.0 exhibits exceptional robustness and structural reasoning. This highly stable performance under extreme generative stress underscores the readiness of Tstars-Tryon 1.0 for industrial deployment.

- 3.3 Qualitative Results

To intuitively evaluate the performance of different models in complex multi-garment virtual try-on tasks, we present qualitative comparisons in Figures 11, 12, and 13. In these scenarios, the models are required to rationally dress the source model with multiple reference images (covering tops, pants, skirts, dresses, coats, shoes, hats, and bags) based on complex textual prompts, while strictly maintaining the original pose and background. Experimental results demonstrate that our proposed method significantly outperforms existing top open-source or proprietary models across three key dimensions:

- • Extreme Robustness: Stable Identity, Pose, and Background Consistency. A core challenge in try-on tasks is redrawing garments without disrupting the unedited regions of the source image. In Figure 12 (bottom) and Figure 13 (top), Nano Banana Pro, GPT-Image-1.5, and Seedream5 lite experience severe "Identity Degradation" during complex full-body replacements—drastically altering the model’s facial shape and body type, or hallucinating entirely incorrect backgrounds. Additionally, in Figure 11 (top), all baseline models fail to preserve the model’s original hand pose. Our method, through a more precise local control mechanism, flawlessly retains the model’s original facial features, body posture, and intricate natural/street backgrounds while achieving complex full-body garment transfers.
- • High Realism: Exceptional Fidelity of Garment Details. Observations from the generated images indicate that whether dealing with complex patterns (e.g., the kimono checkered pattern in Figure 12 (top)), specific materials (e.g., the fur coat in Figure 13 (top) and plush slippers in Figure 13 (bottom)), or non-standard accessories (e.g., the swim cap in Figure 12 (bottom)), our model highly restores the texture and structural features of the reference images. Conversely, baseline models GPT-Image-1.5 and Nano Banana Pro hallucinate the color of the bottoms (Figure 13 (top)), and Seedream5 lite generates a logo inconsistent with the reference coat (Figure 13 (bottom)). Furthermore, as shown in Figure 11 (bottom), both Seedream5 lite and Nano Banana Pro severely confuse the color and style of the skirt.
- • Unprecedented Flexibility: Superior Instruction Following and Multi-Garments Generation. As the number of reference garments increases (e.g., combinations of up to 6 items as shown in Figure 13), existing open-source and closed-source models generally suffer from severe "feature omission" or "semantic confusion." For instance, in Figure 13 (top), QwenEdit-2511 misses the crossbody bag, while GPT-Image-1.5 and Nano Banana Pro confuse the color of the bottoms. In contrast, our model demonstrates strong multimodal contextual understanding. It not only accurately places all reference items on the correct body parts but also precisely executes complex spatial logic instructions such as "keep open, revealing the inner layer" (see Figures 11 and 12).

In summary, the qualitative analysis validates that our model exhibits industry-leading flexibility, realism, and robustness in generation quality when handling highly complex, real-world "multi-garment try-on" and "multi-condition combination" scenarios.

- 4 Demonstrations

- 4.1 Single-Garment

- This Figure 14 showcases the extreme robustness and high-fidelity rendering capabilities of Tstars-Tryon 1.0 across a variety of challenging scenarios. In the top row, the system demonstrates its ability to map 2D flat-lay garments—such as black trousers (left) and a colorful knit cardigan (right)—onto complex human poses, including crouching and swinging, while maintaining accurate material textures and lighting. The second row highlights spatial adaptation and material realism, successfully fitting a fuzzy grey jacket onto a ballet dancer in motion (left) and a tailored blazer onto a subject in a sharp low-angle perspective (right) without geometric distortion. The third row emphasizes the model’s strict adherence to body shape and complex occlusions, accurately rendering a patterned pink dress on a plus-size model (left) and a flowing cream dress on a seated subject (right), ensuring realistic folds and shadows. Finally, the bottom row illustrates the meticulous preservation of identity and background; the system seamlessly integrates a green ombre shirt (left) and a long white skirt (right) into scenes with intricate outdoor lighting and architectural backgrounds, keeping the user’s facial features and the environment’s structural integrity perfectly intact.

4.2 Multi-Garment

- This Figure 15 showcases the advanced capabilities of the system to generate complex, multi-item outfits across various users and environments. In the top row, the system demonstrates precise multi-item mapping for a male model in an upscale indoor setting, seamlessly integrating an intricate white stand-up collar jacket and patterned leopard shorts with multiple accessories like a cap and sneakers, showcasing

###### Prompts

###### Input Images

[Figure 281]

2

[Figure 282]

[Figure 283]

1

将图1中模特的下装替换为图 3中的短裤，保留原有的上装。 穿上图2中的棉服外套并保持 敞开，露出里面的内搭。保 持模特的鞋子、姿势和背景 不变。

[Figure 284]

Replace the model's bottoms in

3

- [Image1] with the shorts in [Image3], keeping the original top. Put on the cotton jacket in
- [Image2] and keep it open, revealing the inner layer. Keep the model's shoes, pose, and background unchanged.

Ours

[Figure 285]

[Figure 286]

[Figure 287]

[Figure 288]

QwenEdit-2511 GPT-Image-1.5 Seedream5 lite Nano Banana Pro

[Figure 289]

[Figure 290]

[Figure 291]

将图1中模特的服装替换为图3 中的汉服风格连衣裙，并戴上 图4中的渔夫帽。将图2中的针 织背心套在连衣裙外，保持敞 开以露出内层衣物。保持模特 的姿势和背景不变。

1 2

[Figure 292]

4

[Figure 293]

Replace the model's clothing in

- [Image 1] with the Hanfu-style dress from [Image 3] and put on the bucket hat from [Image 4]. Layer the knitted vest from
- [Image 2] over the dress, keeping it open to reveal the inner garment. Keep the model’s pose and background unchanged.

3

Ours

[Figure 294]

[Figure 295]

[Figure 296]

[Figure 297]

QwenEdit-2511 GPT-Image-1.5 Seedream5 lite

Nano Banana Pro

- Figure 11: Qualitative comparison of multi-garment and accessory try-on. Compared to baseline models, Tstars-Tryon 1.0 (Ours) more accurately follows text instructions and precisely reconstructs garment details.

[Figure 298]

[Figure 299]

[Figure 300]

让图1中的模特穿上图3中的 1 连衣裙，并披上图2中的和风 外套，外套需保持敞开以露 出裙装。同时穿上图4中的高 跟鞋，戴上图5中的针织帽。 保持模特的姿势和背景不变。

[Figure 301]

[Figure 302]

Replace the clothes of the model in [Image 1] with the dress in

[Figure 303]

- [Image 3], and let her wear the high heels in [Image 4] and the knit hat in [Image 5]. The kimono-style coat in [Image 2] should be worn open over the dress. Keep the model’s pose and background unchanged.

Prompts Input Images

2 3

4

5

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

让 图 1中的模特穿上图2中的 连衣裙，背着图3中的背包， 并戴上图4中的泳帽。将图5 中的羽绒服穿在连衣裙外面， 并保持敞开。保持模特的姿 势和背景不变。

Have the model in [Image 1] wear the dress in [Image 2], carry the backpack in [Image 3], and wear the swim cap in

- [Image 4]. The puffer jacket in
- [Image 5] should be worn over the dress and kept open. Keep the model's pose and background unchanged.

Ours

[Figure 314]

[Figure 315]

[Figure 316]

[Figure 317]

Nano Banana Pro

QwenEdit-2511 GPT-Image-1.5 Seedream5 lite

1

2 3

4 5

Ours

Nano Banana Pro

QwenEdit-2511 GPT-Image-1.5 Seedream5 lite

- Figure 12: Qualitative comparison under complex layered outfits and diverse human characteristics. Our model demonstrates significant advantages in handling cross-style combinations and preserving complex backgrounds and identity.

将图1中模特的衣服替换为图2中的针织衫和图4中的运动裤，并穿上图5中的帆布鞋，背上图6中的斜挎包，戴上图7中的圆顶帽。 将图3中的皮草外套敞开穿在外面，露出内搭。模特的姿势和背景保持不变。

Replace the model's clothes in [Image1] with the knit sweater in [Image2] and the sweatpants in [Image4], and put on the canvas shoes in [Image5], carry the crossbody bag in [Image6], and wear the bowler hat in [Image7]. Wear the fur coat in [Image3] open, revealing the inner layer. Keep the model's pose and background unchanged.

[Figure 318]

[Figure 319]

[Figure 320]

[Figure 321]

[Figure 322]

3

1

4

2

[Figure 323]

[Figure 324]

[Figure 325]

5

6 7

Ours

[Figure 326]

[Figure 327]

[Figure 328]

[Figure 329]

QwenEdit-2511 GPT-Image-1.5 Seedream5 lite

Nano Banana Pro

将图1中模特的衣服替换为图2中的针织上衣和图3中的长裙。让模特穿上图4中的毛绒拖鞋，携带图5中的斜挎包，并戴上图6中的 遮阳帽。图7中的长外套应敞开穿在内搭上衣外面。保持模特的姿势和背景不变。

Replace the clothes of the model in [Image 1] with the knit top from [Image 2] and the long skirt from [Image 3]. Have the model wear the plush slippers from [Image 4], carry the crossbody bag from [Image 5], and put on the visor hat from [Image 6]. The long coat from [Image 7] should be worn open over the inner top. Keep the model's pose and background unchanged.

[Figure 330]

[Figure 331]

[Figure 332]

[Figure 333]

[Figure 334]

1

2 3

4

[Figure 335]

[Figure 336]

[Figure 337]

6 7

5

Ours

[Figure 338]

[Figure 339]

[Figure 340]

[Figure 341]

Nano Banana Pro

QwenEdit-2511 GPT-Image-1.5 Seedream5 lite

- Figure 13: Qualitative comparison of virtual try-on under extreme multi-condition scenarios (up to 6 garments). When given a massive number of reference images, baselines suffer from item omission or identity degradation, whereas our model maintains high stability and semantic alignment.

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

##### Figure 14: Qualitative demonstrations of single-garment try-on. Showcasing Tstars-Tryon 1.0 extreme robustness, precise preservation capabilities (identity, pose, background, body shape), and high-fidelity rendering of complex materials across varying perspectives and input conditions.

1

3

2

- 戴上图4中的空顶帽，挎上图7中 的编织托特包，并穿上图3中的 红色平底鞋。将图5中的浅蓝色 羽绒服套在衬衫外面并保持敞开， 露出内搭。保持模特的姿态和背 景不变。

Have the model in [Image1] wear the floral shirt in [Image2] and the plaid patchwork skirt in [Image6], the sun visor in [Image4], carry the woven tote bag in [Image7], and wear the red flat shoes in [Image3]. Place the light blue puffer jacket in [Image5] over the shirt, keeping it open to reveal the inner layer. Keep the model's pose and background unchanged.

[Figure 366]

Have the model in [Image 1] wear the long-sleeved T-shirt from [Image 2] and the wide-leg pants from [Image

- 4]. Put on the rain boots from [Image
- 5], the bucket bag from [Image 6], and the baseball cap from [Image 7]. The knitted cardigan from [Image 3] should be worn over the top and kept open. Keep the model's pose and background unchanged.

让图1中的模特穿上图2中的长 袖T恤和图4中的阔腿裤，再穿 上图5中的雨靴、图6中的水桶 包以及图7中的棒球帽。将图3 中的针织开衫套在最外面，并 保持敞开状态。保持模特的姿 势和背景不变。

[Figure 367]

[Figure 368]

[Figure 369]

[Figure 370]

[Figure 371]

[Figure 372]

[Figure 373]

[Figure 374]

Keep the model‘s pose and background in [Image 1] unchanged. Replace the top with the white stand-up collar jacket with patterns on its surface from [Image 2], and replace the bottom part with the shorts from [Image 34. Make the model wear the light brown baseball cap from [Image 5] and the silvery white sneakers from [Image 3].

[Figure 375]

[Figure 376]

[Figure 377]

[Figure 378]

[Figure 379]

[Figure 380]

保持图1中模特的姿势和背景不 变 ，上衣替换为图2中的表面具 有花纹的白色立领夹克，下半部 分替换为图4中的短裤，让模特

- 戴上图5中的浅棕色棒球帽，穿 上图3中的银白色运动鞋。

4 5

[Figure 381]

[Figure 382]

[Figure 383]

[Figure 384]

将图1中模特的服装替换为图2中 的针织连帽衫和图4中的长裙。 让模特携带图3中的迷你斜挎包， 并佩戴图5中的钟形帽，同时保 持原有的姿势和背景。

1

2 3

Replace the clothes of the model in [Image 1] with the knitted hoodie from [Image 2] and the long skirt from [Image 4]. Have the model carry the mini crossbody bag from [Image 3] and wear the cloche hat from [Image 5], while keeping the original pose and background.

[Figure 385]

[Figure 386]

4

5

1

2

4

3

6

5

7

[Figure 387]

[Figure 388]

[Figure 389]

让图1中的模特穿上图2中的碎花 衬衫和图6中的格纹拼接半身裙，

[Figure 390]

1

4

3

2

[Figure 391]

[Figure 392]

[Figure 393]

6

7

5

- Figure 15: Demonstrations of multi-garment try-on outfit composition. Highlighting the model’s capability for reasonable multi-garment layering, diverse accessory try-on, and strict preservation of user attributes including diverse body types.

robust cross-item geometric alignment. The second row features a complete style overhaul for a female model posed among numerous stuffed animals; her simple original outfit is replaced with a coordinated pink knitted hoodie and grey pleated skirt, accented by a new cloche hat and a crossbody bag. The model retains her pose and the complex, occluding background, demonstrating high-fidelity details in a dense environment. In the third row, the focus is on reasonable multi-garment layering in an outdoor setting. The model accurately layers a long-sleeved T-shirt and wide-leg pants under an open white knitted cardigan, while adding multiple accessories like a bucket bag and a different hat, ensuring realistic drape and interactions with the water background. Finally, the bottom row underscores the model’s ability to strictly preserve user attributes, specifically for a plus-size body type. A complex ensemble, including a floral shirt, a patterned skirt, and an open light blue puffer jacket, is accurately fitted without distortion or artificial slimming, demonstrating effective layering of distinct textures and patterns.

- This Figure 16 highlights the model’s versatility in handling unconventional orientations and intricate lighting interplay. In the first row, the system excels in low-light, high-contrast neon environments, accurately rendering the delicate lace of the inner top and the sheen of satin shorts while maintaining consistent global illumination. The second row showcases extreme geometric robustness, where the model successfully adapts a red coat and sneakers to a challenging lying-down perspective with significant foreshortening. The third row demonstrates a breakthrough in simultaneous multi-subject synthesis, flawlessly executing garment swaps for both an adult and a child within a shared natural setting while preserving their interactive poses and distinct body scales. Finally, the bottom row illustrates the handling of complex item ensembles and occlusion; a hat, bag, and layered jacket are realistically integrated onto a seated model holding a book, achieving exceptional perceptual consistency and physical detail across a dense combination of accessories.

Figure 17 illustrates a highly flexible application of Tstars-Tryon 1.0: the holistic “OOTD (Outfit of the Day) Swap.” This task requires the model to extract and transfer a complete stylistic ensemble from a single reference image—encompassing tops, bottoms, outerwear, footwear, and accessories, etc.—between two entirely different subjects. The model effortlessly achieves this mutual exchange while seamlessly adapting the garments to different body types, challenging postures (e.g., adapting an outfit from a standing pose in the snow to a seated pose in a library), and varying environmental lighting conditions. Notably, the left panel demonstrates exceptional cross-domain robustness, successfully swapping a complex real-world outfit with a 3D animated character. Tstars-Tryon 1.0 perfectly fits the physical garments onto the stylized 3D geometry without compromising the character’s unique aesthetic or the human subject’s photorealism. Throughout these massive holistic transformations, the user identity characteristics and highly complex background contexts are strictly preserved, underscoring the model’s superior capability in disentangling clothing semantics from spatial and identity features.

#### 4.3 Semantic expansion

To demonstrate the powerful semantic generalization capabilities of Tstars-Tryon 1.0, Figure 18 illustrates its ability to transcend the boundaries of standard human photography and adapt to highly unconventional, out-of-domain subjects. Moving from top-left to bottom-right, the model seamlessly applies modern streetwear (a zip-up hoodie and jeans) onto a 3D animated character(Row 1, left), accurately handling highly stylized, non-standard body proportions without spatial distortion. It also successfully adapts to the flat geometric domain of 2D anime, naturally fitting a patterned hat and wide-leg pants onto an illustrated character while respecting the original artistic aesthetic(Row 1, right). Furthermore, the system exhibits extraordinary cross-modality blending by replacing the iconic headwear in a classical oil painting ("Girl with a Pearl Earring") with a modern floral bucket hat, remarkably matching the historical lighting, shadows, and brushstroke textures of the original artwork(Row 2, left). Finally, the model’s structural understanding extends even to non-anthropomorphic subjects, flawlessly rendering an orange tutu skirt onto a bird(Row 2, right). These diverse cross-domain applications prove that Tstars-Tryon 1.0 has learned deep, generalizable semantic representations of garments and spatial relationships, rather than merely overfitting to real-world human pose priors.

### 5 Industrial-Scale Deployment

The proposed algorithm has been fully deployed within the Taobao App and is publicly accessible to end consumers as the "AI Try-On" service. As illustrated in Figure 19, the system supports a complete consumer-facing user journey: users can initiate a try-on request either through the in-app shopping assistant or directly from a product detail page, and, after uploading a personal portrait, can try on a wide range of apparel items as well as freely compose DIY multi-garment outfits.

To the best of our knowledge, this constitutes one of the largest production-scale deployments of virtual

[Figure 394]

[Figure 395]

将图1中模特的服装替换为图2的 蕾丝衬衫、图5的红色外套和图3 的运动裤。让模特穿上图4的拼 接运动鞋，同时保持原有的姿势、 背景、红帽子、墨镜及首饰不变。

[Figure 396]

让图1中的成年女性穿上图5中 的蓝白格子上衣和图4中的白色 长裤，并让图中的小女孩穿上

1

[Figure 397]

[Figure 398]

2 3

- 图3中的上衣和图2中的绿色裤 子。保持两人的姿态、发型、 表情和背景不变。

Keep the models' poses and background in [Image 1] unchanged. Replace the adult woman's clothes with the blue and white checkered top from

- [Image 5] and the white pants

- from [Image 4]. Replace the little girl's clothes with the top from [Image 3] and the green

- pants from [Image 2].

[Figure 399]

Have the model in [Image 1] wear the red top from [Image 2] and the skirt from [Image 3]. Put on the gray sneakers from [Image 4], the bag from [Image 5], and the hat from [Image 6]. The black and white jacket from [Image 7] should be worn over the top and kept open. Keep the model's pose, face, hair, book, and background unchanged.

让图1中的模特穿上图2中的红色 上衣和图3中的半身裙，再穿上 图4中的灰色运动鞋、图5中的包 以及图6中的帽子。将图7中的黑 白拼接夹克套在最外面，并保持 敞开状态。保持模特的姿势、面 部、发型、手中书籍及背景不变。

[Figure 400]

[Figure 401]

[Figure 402]

[Figure 403]

[Figure 404]

[Figure 405]

[Figure 406]

[Figure 407]

Keep the model's pose and background in [Image 1] unchanged. Replace the inner top with the lace top from [Image 2], layered with the open cardigan from [Image 5]. Replace the bottom part with the satin shorts from [Image 3], and wear the platform loafers from [Image 4].

[Figure 408]

[Figure 409]

[Figure 410]

[Figure 411]

保持图1中模特的姿势和背景不 变，上衣内搭替换为图2中的白 色蕾丝上衣，并外搭图5中的敞 开花纹开衫，下半部分替换为图 3中的绸缎短裤，穿上图4中的厚 底乐福鞋。

1

2

3

4

5

[Figure 412]

[Figure 413]

[Figure 414]

[Figure 415]

[Figure 416]

[Figure 417]

1

2

3

4

5

Replace the clothes of the model in [Image 1] with the lace shirt from [Image 2], the red coat from [Image 5], and the sports

- pants from [Image 3]. Have the model wear the patchwork sneakers from [Image 4], while keeping the original pose, background, red hat, sunglasses, and jewelry.

[Figure 418]

[Figure 419]

4 5

1

2

3 4

6

5

7

- Figure 16: Versatile multi-item synthesis. Advanced applications in multi-item synthesis under heterogeneous lighting, unconventional perspectives, and multi-subject interactions

try-on technology to date. The system has served several million users and fulfilled tens of millions of try-on requests, demonstrating its robustness, scalability, and commercial viability under real-world industrial workloads. We further plan to roll out the service to the entire Taobao user base, where the system is expected to handle tens of millions of try-on requests per day, driving the large-scale

[Figure 420]

[Figure 421]

[Figure 422]

[Figure 423]

OOTD Swap OOTD Swap

[Figure 424]

[Figure 425]

[Figure 426]

[Figure 427]

- Figure 17: Demonstrations of holistic OOTD (Outfit of the Day) swapping across diverse subjects, poses, and domains. Tstars-Tryon 1.0 flawlessly transfers entire ensembles between different individuals, including cross-domain transfers between real humans and 3D avatars, while strictly preserving identities and backgrounds.

adoption of virtual try-on technology in e-commerce scenarios. This large-scale deployment further validates that the proposed approach effectively resolves the long-standing trade-off between C-end serving cost and generation quality, enabling virtual try-on to transition from a research prototype to a fully commercialized consumer-facing product.

### 6 Acknowledgments

All contributors are listed in alphabetical order by their last names.

Engineering Contributors: Yichao Cai, Donglai Ge, Zhiwei Han, Shuaiqi Jia, Tao Lan, Jiacheng Li, Yi Li, Kan Liu, Xu Liu, Zhenxiao Liu, Weiyi Lu, Chong Ma, Lin Qu, Chuanli Wang, Daisong Wang, Hanlun Wang, Lujie Wang, Yi Wang, Xinjiang Wu, Jiawei Zhang, Lei Zhang, Mao Zhou, Guoxuan Zhu, Tianfu Zhu, Yongjie Zhu

Productization Contributors: Yajun Bai, Jian Ding, Zhengni Guan, Mengli Huang, Nan Liu, Yating Sheng, Xudong Wu, Xiaoyu Zhu

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

##### Figure 18: Cross-domain virtual try-on capabilities. Showcasing the model’s flexible semantic extensibility across diverse non-photorealistic styles and non-human subjects.

[Figure 442]

Taobao AI Try-On Product

Send a product to the shopping assistant Initiate try-on from the product Upload personal portrait Generated try-on result Compose multi-garment outfit Explore outfit styles Manage portraits & settings

[Figure 443]

[Figure 444]

[Figure 445]

[Figure 446]

[Figure 447]

[Figure 448]

[Figure 449]

[Figure 450]

[Figure 451]

Figure 19: Industrial application. Industrial-scale deployment of Tstars-Tryon as the "AI Try-On" service on the Taobao App, illustrating the complete consumer-facing user journey from try-on initiation, portrait upload, single-/multi-garment try-on generation, to outfit-style exploration and personal portrait management.

### References

Black Forest Labs. Flux.2-klein: Towards interactive visual intelligence, 2025. URL https://bfl.ai/blog/ flux2-klein-towards-interactive-visual-intelligence. Accessed: 2026-03-18.

ByteDance. Deeper thinking, more accurate generation: Introducing seedream 5.0 lite, 2026. URL https://seed.bytedance.com/en/blog/ deeper-thinking-more-accurate-generation-introducing-seedream-5-0-lite.

Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, et al. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025.

Seunghwan Choi, Sunghyun Park, Minsoo Lee, and Jaegul Choo. Viton-hd: High-resolution virtual try-on via misalignment-aware normalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14131–14140, 2021.

Zheng Chong, Xiao Dong, Haoxiang Li, Shiyue Zhang, Wenqing Zhang, Xujie Zhang, Hanqing Zhao, and Xiaodan Liang. Catvton: Concatenation is all you need for virtual try-on with diffusion models,

2024. URL https://arxiv.org/abs/2407.15886.

Zheng Chong, Yanwei Lei, Shiyue Zhang, Zhuandi He, Zhen Wang, Xujie Zhang, Xiao Dong, Yiling Wu, Dongmei Jiang, and Xiaodan Liang. Fastfit: Accelerating multi-reference virtual try-on via cacheable diffusion models, 2025. URL https://arxiv.org/abs/2508.20586.

Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, et al. Patch n’pack: Navit, a vision transformer for any aspect ratio and resolution. Advances in Neural Information Processing Systems, 36:2252–2274, 2023.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for highresolution image synthesis. In Forty-first international conference on machine learning, 2024.

Google Blog. Introducing nano banana pro, 2025. URL https://blog.google/innovation-and-ai/ products/nano-banana-pro/. Accessed: 2026-03-18.

Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in neural information processing systems (NIPS), volume 30, 2017.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

Boyuan Jiang, Xiaobin Hu, Donghao Luo, Qingdong He, Chengming Xu, Jinlong Peng, Jiangning Zhang, Chengjie Wang, Yunsheng Wu, and Yanwei Fu. Fitdit: Advancing the authentic garment details for high-fidelity virtual try-on. arXiv preprint arXiv:2411.10499, 2024.

Davide Morelli, Matteo Fincato, Marcella Cornia, Federico Landi, Fabio Cesari, and Rita Cucchiara. Dress code: High-resolution multi-category virtual try-on. In Computer Vision – ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part VIII, pp. 345–362, Berlin, Heidelberg, 2022. Springer-Verlag. ISBN 978-3-031-20073-1. doi: 10.1007/978-3-031-20074-8_20. URL https://doi.org/10.1007/978-3-031-20074-8_20.

- OpenAI. Gpt-image-1.5 model card, 2025. URL https://platform.openai.com/docs/models/ gpt-image-1-5.
- OpenAI. Gpt-image-2 model card, 2026. URL https://platform.openai.com/docs/models/gpt-image-2. Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution

image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10684–10695, 2022.

Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, et al. Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427, 2025.

Super Intelligence Team. Firered-image-edit-1.0 technical report, 2026. URL https://arxiv.org/abs/ 2602.13344.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025.

Yuhao Xu, Tao Gu, Weifeng Chen, and Arlene Chen. Ootdiffusion: Outfitting fusion based latent diffusion for controllable virtual try-on. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 8996–9004, 2025.

Tianwei Yin, Michaël Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6613–6623, 2024.

Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. Diffusionnft: Online diffusion reinforcement with forward process. arXiv preprint arXiv:2509.16117, 2025.

Zijian Zhou, Shikun Liu, Xiao Han, Haozhe Liu, Kam Woh Ng, Tian Xie, Yuren Cong, Hang Li, Mengmeng Xu, Juan-Manuel Pérez-Rúa, Aditya Patel, Tao Xiang, Miaojing Shi, and Sen He. Learning flow fields in attention for controllable person image generation. arXiv preprint arXiv:2412.08486, 2024.

