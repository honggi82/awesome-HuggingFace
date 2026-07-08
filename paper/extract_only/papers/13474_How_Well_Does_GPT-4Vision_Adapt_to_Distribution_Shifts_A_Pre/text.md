# arXiv:2312.07424v3[cs.LG]25Feb2024

###### How Well Does GPT-4V(ision) Adapt to Distribution Shifts? A Preliminary Investigation

Zhongyi Han1∗, Guanglin Zhou2∗, Rundong He3∗, Jindong Wang4, Tailin Wu5, Yilong Yin3, Salman Khan1,6, Lina Yao7,2,8, Tongliang Liu9,1 and Kun Zhang10,1

1Mohamed bin Zayed University of Artificial Intelligence 2The University of New South Wales 3School of Software, Shandong University 4Microsoft Research Asia 5Westlake University 6Australian National University 7Data61, CSIRO 8Macquarie University 9The University of Sydney

10Carnegie Mellon University

###### Abstract

In machine learning, generalization against distribution shifts—where deployment conditions diverge from the training scenarios—is crucial, particularly in fields like climate modeling, biomedicine, and autonomous driving. The emergence of foundation models, distinguished by their extensive pretraining and task versatility, has led to an increased interest in their adaptability to distribution shifts. GPT-4V(ision) acts as the most advanced publicly accessible multimodal foundation model, with extensive applications across various domains, including anomaly detection, video understanding, image generation, and medical diagnosis. However, its robustness against data distributions remains largely underexplored. Addressing this gap, this study rigorously evaluates GPT-4V’s adaptability and generalization capabilities in dynamic environments, benchmarking against prominent models like CLIP, LLaVA, and Gemini. We delve into GPT-4V’s zero-shot generalization across 13 diverse datasets spanning natural, medical, and molecular domains. We further investigate its adaptability to controlled data perturbations and examine the efficacy of in-context learning as a tool to enhance its adaptation. Our findings delineate GPT-4V’s capability boundaries in distribution shifts, shedding light on its strengths and limitations across various scenarios. Importantly, this investigation contributes to our understanding of how AI foundation models generalize to distribution shifts, offering pivotal insights into their adaptability and robustness. The code is publicly available at https://github.com/jameszhou-gl/gpt-4v-distribution-shift.

clip-vit-base-patch16

llava-v1.5-13b

DomainNet

gpt-4-vision-preview

Fmow

gemini-pro-vision

0.75 0.27

Ofﬁce-Home

0.68 0.24

0.92

0.89

Terra_incognita

0.58

0.78

0.70 0.15 0.37

0.52

0.49

0.46

0.16

VLCS

0.89

0.85

0.84

0.81

0.21

iWildCam 0.34

0.27

0.06

0.01

0.2 0.4 0.6 0.8

0.960.97 1.0 0.980.99

PACS

0.50

0.51

0.53

0.31

0.16

Camelyon17

0.42

0.23

0.49

0.52

0.08

0.34

COVID

0.08

0.09

0.49

0.52

0.48

0.12 0.51

HAM10000

0.52

DrugOOD_Assay DrugOOD_Scaffold

NIH_Chest

Figure 1 | Comparative analysis of zero-shot generalization performance across 13 distinct datasets, encompassing natural, medical, and molecular domains. The analysis features the performances of three advanced models: CLIP, LLaVA, GPT-4V and Gemini.

∗ These authors contributed equally to this work. Email: {hanzhongyicn;jameszhou.ustc}@gmail.com

###### Contents

- 1 Introduction 5

- 1.1 Motivation and Overview . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
- 1.2 Our Approach in Exploring GPT-4V . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 6

- 1.2.1 How Do We Treat Distribution Shifts in This Work? . . . . . . . . . . . . . . . . . . . 6
- 1.2.2 Sample Selection Guidance for GPT-4V Evaluation . . . . . . . . . . . . . . . . . . . . 7
- 1.2.3 Prompt Designs . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7

- 1.3 Contributions of This Report . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 1.4 Limitations of This Report . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 9

- 2 Observations 9
- 3 Zero-shot Generalization Across Varied Domains 11

- 3.1 Natural Images . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11

- 3.1.1 Task Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11
- 3.1.2 Comparative Accuracies Across Datasets and Domains . . . . . . . . . . . . . . . . . . 12
- 3.1.3 Case Demonstration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

- 3.2 Medical Images . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14

- 3.2.1 Task Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 14
- 3.2.2 Comparative Accuracies Across Datasets and Domains . . . . . . . . . . . . . . . . . . 16
- 3.2.3 Case Demonstration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- 3.3 Scientific Images . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16

- 3.3.1 Task Introduction . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 16
- 3.3.2 Performance Across Datasets and Domains . . . . . . . . . . . . . . . . . . . . . . . . 17
- 3.3.3 Prompt Engineering Trick . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17
- 3.3.4 Case Demonstration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- 4 Adaptability to Controlled Data Perturbations 17

- 4.1 Gaussian Noise . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 17

- 4.1.1 Comparative Accuracies Across Domains . . . . . . . . . . . . . . . . . . . . . . . . . 17
- 4.1.2 Case Demonstration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- 4.2 Style Change with ControlNet . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- 4.2.1 Comparative Accuracies Across Domains . . . . . . . . . . . . . . . . . . . . . . . . . 19
- 4.2.2 Case Demonstration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 19

- 5 Exploiting In-Context Learning for Domain Bridging 20

- 5.1 In-context Setup . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 20
- 5.2 In-context Performance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21
- 5.3 In-context Case Demonstration . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 21

- 6 Conclusion 22

###### List of Figures

- 1 Comparative Analysis of Zero-Shot Generalization Across Diverse Domains . . . . . . . . . . 1
- 2 Illustration of a structured prompt format . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 8
- 3 Illustration of a structured prompt format . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 18
- 4 In-context learning on GPT-4V for Domain Bridging . . . . . . . . . . . . . . . . . . . . . . . 21
- 5 In-context Case Demonstration: Case 1 on Camelyon17 . . . . . . . . . . . . . . . . . . . . . 23
- 6 In-context Case Demonstration: Case 2 on COVID . . . . . . . . . . . . . . . . . . . . . . . . 24
- 7 Comparative Domain Accuracies on PACS, VLCS and Office-Home . . . . . . . . . . . . . . . 25
- 8 Comparative Domain Accuracies on DomainNet, Fmow and TerraIncognita . . . . . . . . . . 26
- 9 Natural Distribution Shift: Case 1 on PACS . . . . . . . . . . . . . . . . . . . . . . . . . . . . 27
- 10 Natural Distribution Shift: Case 2 on PACS . . . . . . . . . . . . . . . . . . . . . . . . . . . . 28
- 11 Natural Distribution Shift: Case 3 on PACS . . . . . . . . . . . . . . . . . . . . . . . . . . . . 29
- 12 Natural Distribution Shift: Case 4 on PACS . . . . . . . . . . . . . . . . . . . . . . . . . . . . 30
- 13 Natural Distribution Shift: Case 5 on VLCS . . . . . . . . . . . . . . . . . . . . . . . . . . . . 31
- 14 Natural Distribution Shift: Case 6 on VLCS . . . . . . . . . . . . . . . . . . . . . . . . . . . . 32
- 15 Natural Distribution Shift: Case 7 on VLCS . . . . . . . . . . . . . . . . . . . . . . . . . . . . 33
- 16 Natural Distribution Shift: Case 8 on VLCS . . . . . . . . . . . . . . . . . . . . . . . . . . . . 34
- 17 Natural Distribution Shift: Case 9 on Office-Home . . . . . . . . . . . . . . . . . . . . . . . . 35
- 18 Natural Distribution Shift: Case 10 on Office-Home . . . . . . . . . . . . . . . . . . . . . . . . 36
- 19 Natural Distribution Shift: Case 11 on Office-Home . . . . . . . . . . . . . . . . . . . . . . . . 37
- 20 Natural Distribution Shift: Case 12 on Office-Home . . . . . . . . . . . . . . . . . . . . . . . . 38
- 21 Natural Distribution Shift: Case 13 on DomainNet . . . . . . . . . . . . . . . . . . . . . . . . 39
- 22 Natural Distribution Shift: Case 14 on DomainNet . . . . . . . . . . . . . . . . . . . . . . . . 40
- 23 Natural Distribution Shift: Case 15 on Fmow . . . . . . . . . . . . . . . . . . . . . . . . . . . 41
- 24 Natural Distribution Shift: Case 16 on Fmow . . . . . . . . . . . . . . . . . . . . . . . . . . . 42
- 25 Natural Distribution Shift: Case 17 on Fmow . . . . . . . . . . . . . . . . . . . . . . . . . . . 43
- 26 Natural Distribution Shift: Case 18 on TerraIncognita . . . . . . . . . . . . . . . . . . . . . . 44
- 27 Natural Distribution Shift: Case 19 on TerraIncognita . . . . . . . . . . . . . . . . . . . . . . 45
- 28 Natural Distribution Shift: Case 20 on TerraIncognita . . . . . . . . . . . . . . . . . . . . . . 46
- 29 Natural Distribution Shift: Case 21 on TerraIncognita . . . . . . . . . . . . . . . . . . . . . . 47
- 30 Natural Distribution Shift: Case 22 on VLCS . . . . . . . . . . . . . . . . . . . . . . . . . . . 48
- 31 Medical Distribution Shift: Case 1 on HAM10000 . . . . . . . . . . . . . . . . . . . . . . . . . 49
- 32 Medical Distribution Shift: Case 2 on COVID . . . . . . . . . . . . . . . . . . . . . . . . . . . 50
- 33 Scientific Distribution Shift: Case 1 on DrugOOD_assay . . . . . . . . . . . . . . . . . . . . . 51
- 34 Induced Distribution Shift: Case 1 on Office-Home_gaussion . . . . . . . . . . . . . . . . . . 52
- 35 Induced Distribution Shift: Case 2 on PACS_gaussion . . . . . . . . . . . . . . . . . . . . . . 53
- 36 Induced Distribution Shift: Case 3 on PACS_unseen . . . . . . . . . . . . . . . . . . . . . . . 54
- 37 Induced Distribution Shift: Case 4 on Office-Home_unseen . . . . . . . . . . . . . . . . . . . 55
- 38 Induced Distribution Shift: Case 5 on Office-Home_unseen . . . . . . . . . . . . . . . . . . . 56

- 39 Induced Distribution Shift: Case 6 on Office-Home_unseen . . . . . . . . . . . . . . . . . . . 57
- 40 Induced Distribution Shift: Case 7 on Office-Home_unseen . . . . . . . . . . . . . . . . . . . 58
- 41 Induced Distribution Shift: Case 8 on Office-Home_unseen . . . . . . . . . . . . . . . . . . . 59

###### 1 Introduction

###### 1.1 Motivation and Overview

In the evolving landscape of machine learning, the challenge of distribution shift emerges as a pivotal concern, often regarded as one of the main problems in improving the generalization ability. At the core of this issue lies the independent and identically distributed (i.i.d.) assumption, foundational yet frequently violated in practical scenarios. Mathematically, distribution shift denotes a divergence between the training data distribution and the data encountered in real-world applications, a discrepancy that can significantly impair model performance. This phenomenon is ubiquitously observed across diverse domains, from climate modeling (Knutti et al., 2010; Zwiers et al., 2013) and biomedicine (Park et al., 2021; Cascarano et al., 2023; Stacke et al., 2020; Huang et al., 2021b) to wildlife conservation (Tuia et al., 2022; Ellis, 2011; Beery et al., 2018), autonomous driving (Stocco and Tonella, 2022), and financial forecasting (Mashrur et al., 2020). The omnipresence of distribution shifts underscores a fundamental limitation: even an algorithm capable of perfectly fitting all available data lacks practical utility without the ability to generalize in a dynamic, ever-changing world. This challenge is further compounded by the significant performance degradation observed in numerous machine learning algorithms when confronted with distribution shifts, despite their otherwise notable successes. The traditional machine learning models, while adept in stable environments, struggle with evolving or varying data distributions, highlighting an urgent need for algorithms that can adaptively maintain accuracy in the face of such shifts (Wang et al., 2022a; Yue et al., 2019; Prakash et al., 2019; Huang et al., 2021a; Qiao et al., 2020; Liu et al., 2018; Li et al., 2018b; Ganin and Lempitsky, 2015; Ganin et al., 2016; Rojas-Carulla et al., 2018; Sun et al., 2021; Christiansen et al., 2021; Gulrajani and Lopez-Paz, 2020; Wiles et al., 2022).

Significant seminal studies have been made in addressing distribution shifts. In the realm of domain adaptation, numerous studies have demonstrated remarkable successes, particularly when the target distribution is accessible during the training process (Sun and Saenko, 2016; Ganin and Lempitsky, 2015; Venkateswara et al., 2017a; Peng et al., 2019a; Han et al., 2022a;b). This capability to adapt to known shifts has been a key driver in advancing models’ robustness across varied datasets. It is not an isolated case that a plethora of domain generalization methods are designed through various strategies such as domain alignment (Muandet et al., 2013; Li et al., 2018d), causal representation learning (Schölkopf et al., 2021), stable learning (Zhang et al., 2021), and invariance-based optimization (Liu et al., 2021).

Despite the proliferation of research in distribution shifts as discussed earlier, the field is experiencing a significant trend: the emergence of foundation models as robust solutions (Bommasani et al., 2021; Zhang et al., 2022; Wei et al., 2023; Zheng et al., 2023; Tu et al., 2023). This wave is driven by the capabilities derived from extensive pretraining, comprehensive data understanding, substantial training data volumes, and large-scale network parameters (Kumar et al., 2022; Li et al., 2022a; Ramé et al., 2023; Du et al., 2021; Lee et al., 2022; Li et al., 2022c). The CLIP model (Radford et al., 2021) stands out as a prominent example, showing remarkable robustness against natural distribution shifts (Recht et al., 2019; Wang et al., 2019; Barbu et al., 2019; Hendrycks et al., 2021b;a) primarily due to its extensive training on a wide range of image-text pairs. Notable in this context are models like BLIP (Li et al., 2022b; 2023b), LLaVA (Liu et al., 2023b;a), and Gemini (Team et al., 2023), each contributing unique strengths to the realm of foundation models. Concurrently, foundation models are evolving from specialized tools to versatile, general-purpose assistants, demonstrating their adaptability in various downstream tasks (Awais et al., 2023; Li et al., 2023a). In this evolving landscape, GPT-4V(ision)∗ and its successor GPT emerge as cutting-edge examples of these general-purpose foundation models (OpenAI, 2023), particularly in their ability to handle distribution shifts. Central to our research is the pivotal question: "How well does GPT-4V(ision) adapt to distribution shifts?".

Despite the growing fascination with GPT-4V and its wide-ranging applications, a significant gap is evident in current research: the assessment of its adaptability to distribution shifts remains underexplored. This gap is particularly striking, considering GPT-4V’s extensive deployment across various domains, each presenting unique data challenges. The importance of robust performance in out-of-distribution scenarios cannot be overstated; failures in these contexts can lead to critical errors, especially in high-stakes fields such as medical

∗Hereafter referred to as "GPT-4V"

diagnosis or autonomous driving, where accurate and reliable predictions are imperative. While recent studies have showcased GPT-4V’s effectiveness in anomaly detection (Cao et al., 2023), optical character recognition (Shi et al., 2023), video understanding (Lin et al., 2023), image generation (Yang et al., 2023), zero-shot visual recognition (Wu et al., 2023b), and medical diagnosis (Wu et al., 2023a), the critical question of its performance under distribution shifts has not been addressed. This study is orthogonal to these existing works, focusing on an uncharted yet crucial aspect of machine learning robustness: GPT-4V’s ability to adapt to evolving data environments.

This paper is the first to evaluate GPT-4V on distribution shifts. Through our investigation, we seek to unravel the intricacies of GPT-4V’s performance in various distribution shift scenarios and various applications, thereby providing some insights into its robustness and adaptability. This exploration is not just about probing the depths of GPT-4V’s capabilities; it also aims to broaden our understanding of the potential and limitations of multimodal foundation models in navigating the complexities of real-world data scenarios. This exploration is poised to contribute to the evolution of AI systems, paving the way for more resilient and versatile applications.

Our investigation of GPT-4V in distribution shift scenarios is guided by the following questions:

- 1. How effectively does GPT-4V manage distribution shifts across diverse domains? We seek to measure the extent of GPT-4V’s zero-shot adaptability to the unique distribution shifts inherent in diverse domains. We aim to evaluate how GPT-4V understands and reacts to changes in data distribution, benchmarking its performance against models like CLIP (Radford et al., 2021), known for robustness to natural distribution shifts, LLaVA (Liu et al., 2023b;a), an open-sourced multimodal foundation model, and Gemini (Team et al., 2023), a competitive LMM with GPT-4V.
- 2. How does GPT-4V react to deliberate alternations in data distribution? The traditional approach to distribution shifts typically considers a model’s ability to generalize from a source domain to a target domain, with an inherent shift in data distribution. However, in a zero-shot context, this adaptability may diverge from the conventional settings, as the test data could largely differ from GPT-4V’s pretraining data. Given the opacity of its pre-training data, we investigate its reaction to distribution shifts that we engineer. We first select images from domains where GPT-4V has initially exhibited promising performance, indicative of these images likely to align with or be part of its pre-training distribution. Subsequently, we intend to introduce Gaussian noise and implement stylistic transformation to these chosen images. These manipulations serve as a means to create specific distribution shifts, allowing us to assess the model’s generalization capabilities under these controlled perturbations.
- 3. Is in-context learning an effective method to augment GPT-4V’s adaptation to distribution shifts? Conventional approaches to distribution shifts in foundation models typically involve tuning model parameters, often through methods such as efficient tuning or fine-tuning (Hu et al., 2021a). Considering the impracticality of fine-tuning GPT-4V’s vast parameters, we turn to the potential of in-context learning, a technique at the heart of emergent abilities in the large language models (Brown et al., 2020), as an alternative to simulate traditional domain generalization methods. This apporach entails utilizing representative images from source domain classes as in-context examples, followed by introducing a test image from a novel target domain. This investigation centers on the capacity of in-context examples to improve GPT-4V’s performance in the face of distribution shifts.

###### 1.2 Our Approach in Exploring GPT-4V

###### 1.2.1 How Do We Treat Distribution Shifts in This Work?

In the realm of machine learning, distribution shifts pose a formidable challenge, particularly when deploying models in real-world scenarios that differ from the training environment. Traditional approaches to this issue involve fine-tuning pre-trained foundation models on source domain data to adapt them to target domains. However, when it comes to massive models like GPT-4V, this conventional approach encounters significant hurdles. The vast scale of GPT-4V’s architecture makes standard fine-tuning methods impractical, while the opacity nature of its pre-training data adds layers of uncertainty regarding its performance in novel scenarios. In response to these challenges, our study adopts a nuanced and multifaceted approach, aiming to thoroughly examine GPT-4V’s adaptability to distribution shifts. This involves employing three distinct

evaluation strategies: (1) Zero-shot Generalization: In Section 3, we evaluate GPT-4V’s inherent zero-shot generalization capabilities. Similar to models like CLIP, we assess GPT-4V’s performance across different domains without prior tuning or exposure to relevant domain data, reflecting a purview into the model’s natural adaptability. (2) Response to Data Perturbations: In Section 4, our study investigates GPT-4V’s robustness when faced with artificially induced shifts in data characteristics, focusing on domains where it shows initial high performance. (3) In-Context Learning as a Domain Bridge: In Section 5, we assess GPT-4V’s use of in-context learning to simulate conventional domain generalization paradigms, highlighting its ability to apply contextual understanding from the source domain to interpret data from the target one.

This multifaceted strategy is designed to illustrate GPT-4V’s adaptability comprehensively, from its generalization capacity in comparison with baselines to its performance under artificially induced shifts and its ability to utilize contextual learning as a means to bridge domain gaps.

###### 1.2.2 Sample Selection Guidance for GPT-4V Evaluation

To conduct a meaningful evaluation of GPT-4V within the constraints of the OpenAI API’s rate limits, we have devised a sample selection strategy that prioritizes both diversity and informative value. Our selection process is guided by the following principles.

Random Sampling for Diversity. Our process begins with a random selection of samples from each class across all domains within our 13 datasets, intending to capture the inherent diversity of each domain, reflecting varying complexities and content types. To ensure comprehensive coverage, we employ two distinct sample sizes for each dataset: 180 and 1800. This decision aligns with OpenAI’s revised rate limit policies, increasing daily requests from 100 to 500 as of December 2, 2023. Consequently, our sampling strategy, constrained by the limits of 100 and 500 requests per day for each account, strategically includes approximately 180 and 1800 random selections. Although these numbers might appear limited for certain datasets, they represent a balance between operational efficiency and practical feasibility under existing constraints. Notably, our preliminary findings indicate a consistent performance trend when comparing the two sample sizes, as shown in Tables 1 and 3. Our goal is to minimize selection bias and provide a thorough evaluation of GPT-4V’s performance across a broad spectrum of data.

Inclusion of Failure Cases From CLIP. To further enrich our evaluation, we have deliberately chosen to incorporate 180 instances for each dataset, where the CLIP model exhibits underperformance. This focused selection is driven by a specific objective: to assess how GPT-4V handles challenges that have proven difficult for a well-established model like CLIP. By analyzing GPT-4V’s performance in these particular scenarios, we aim to gain deeper insights into its relative strengths and adaptability compared to CLIP. It is noteworthy that failure cases are sourced from CLIP due to its established role as a baseline model, particularly noted for its zero-shot robustness against distribution shifts. While a similar analytical approach using LLaVa’s failure cases presents a valuable avenue for future research, it remains outside the scope of our current study.

Recognizing the continuous evolution of LMMs, the cases we have selected are designed to function as a benchmark for evaluating and tracking the adaptability of state-of-the-art foundation models to distribution shifts. This benchmark not only serves our current study but also contributes to the broader research community†.

###### 1.2.3 Prompt Designs

In transforming conventional classification tasks into a visual question answering (VQA) format, our focus has been to devise a prompt template that is straightforward yet sufficiently informative. This approach seeks to exploit the expressive capabilities of language, a strength evidenced in previous models such as GPT-2 (Radford et al., 2019) and GPT-3 (Brown et al., 2020). Crucially, our prompt design is tailored to a fair comparison of the inference abilities of GPT-4V and LLaVA. Specifically, we have developed a prompt that pairs an image with a clear, direct question, such as ‘What is in this image?’ followed by a set of answer choices. This design is intended to maintain simplicity, focusing primarily on the model’s ability to interpret and accurately respond to visual content. Moreover, GPT-4V and LLaVA are prompted not just to select an

†The test cases employed in our study have been publicly available at https://huggingface.co/datasets/jameszhou-gl/gpt4v-distribution-shift

###### Prompt:

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices: [dog, elephant, giraffe, guitar, horse, house, person]. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

Figure 2 | An illustration of a structured prompt format used in the PACS dataset, showcasing a specific approach for image-based questioning and response formatting. The format includes a question about the image’s content, a list of answer choices, and a template for answering, including an answer, confidence score, and the reasoning process.

answer option but also to provide a confidence score and a rationale for the decision, enhancing the depth of our analysis.

As exemplified in Figure 2, our structured prompt serves several key purposes:

- • Simplicity: By employing a straightforward template that contextualizes the image with basic question and answer choices, we ensure minimal complexity in the prompt design.
- • Uniformity: The approach ensures consistency and standardization in the model’s responses, which is vital for comparative analysis across diverse test scenarios.
- • Insight into Reasoning: The inclusion of confidence scoring and rationale requirements leverages GPT-4V’s ability to output the decision-making process, thereby providing valuable insights into its reasoning and improving the interpretability of its outputs.

###### 1.3 Contributions of This Report

- • First Exploration: This paper marks the first comprehensive study into the adaptability of GPT-4V to distribution shifts. Our research fills a notable gap in the existing literature, as no prior studies have systematically explored this aspect of GPT-4V. This investigation not only highlights the novelty of our approach but also sets a precedent for future research in evaluating and improving the robustness of multimodal foundation models in the face of data variability.
- • Quantitative Benchmark Results and Comparison: Before our work, explorations into the performance of models like GPT-4V were largely confined to qualitative or case-study approaches. Our study stands out as the first to provide a quantitative evaluation of GPT-4V’s adaptability to distribution shifts. This quantitative approach allows us to present concrete, measurable insights into the model’s capabilities, setting a benchmark for future research and development in this area. Our work also provides a quantitative comparison between GPT-4V and pioneering foundation models.
- • Deep Analysis and Insights: We have conducted rigorous experiments across various domains and scenarios to assess GPT-4V’s performance under distribution shifts. These experiments provide valuable insights into the model’s zero-shot adaptability, its response to engineered distribution shifts, and the effectiveness of in-context learning and style transfer in enhancing its robustness. Our empirical analysis offers a detailed understanding of how GPT-4V navigates complex, real-world data scenarios, contributing significantly to the field of AI and machine learning.

- 1.4 Limitations of This Report Here, we discuss several limitations in our evaluation of GPT-4V:

- • Sample Bias. One of the primary limitations in our assessment of GPT-4V, as well as CLIP and LLaVA, is the presence of sample bias. While we employed random sampling to enhance diversity and reduce selection bias, as detailed in Section 1.2.2, eliminating sample bias remains challenging. Our sampling process ensured a comprehensive representation of each class across domains. However, due to the rate limits imposed by the OpenAI API, our usage of GPT-4V‡ was restricted to a finite number of requests per day. Consequently, we limited our selection to 1,800 cases per dataset. This constraint might result in the derived performance metrics not fully capturing the models’ true capabilities or weaknesses, particularly in processing novel, complex, or varied inputs. Such a limitation is significant, as it potentially leads to overestimations or underestimations of the model’s practical utility and robustness in real-world scenarios.
- • Potential Challenge in VQA Format Transformation. Another limitation arises from converting conventional classification tasks into a VQA format. As illustrated in Figure 2, this approach requires inputting all class names into GPT-4V and prompting it to select one from the list. However, this method becomes impractical for datasets with a large number of classes, such as the ImageNet dataset, which contains 1000 classes (Recht et al., 2019; Wang et al., 2019; Shankar et al., 2021; Hendrycks et al., 2021b). The VQA format in such cases would necessitate an excessively high token count, posing significant challenges in terms of feasibility and efficiency.
- • Failed query may happen in GPT-4V. When verifying the robustness of GPT-4V to distributional shift via the OpenAI API, it is possible that a query failure may occur. The reasons include: 1) Using the API may lead to query failure due to reaching usage limits, API service malfunctions, or network issues. 2) The input data has a significant distributional shift, causing the model to fail to understand or respond correctly, leading to query failure. This query failure leads to a difference in the denominator when conducting quantitative assessments, between GPT-4V and other models like CLIP and LLaVA. For example, in Table 1’s random test, both CLIP and LLaVA are at 180, whereas GPT-4V is less than 180 due to query failures.

In summary, while our evaluation may not be exhaustive, we believe that this analysis offers valuable insights for both researchers and medical professionals, it sheds light on the current capabilities of the multimodal foundational model and may inspire future work towards building medical foundation models.

###### 2 Observations

In this section, we summarize our primary observations from the extensive experiments conducted on GPT-4V, addressing the adaptability of this advanced multimodal foundation model to distribution shifts.

- • General Performance Across Domains: In Section 3, across various domains, GPT-4V showcased robust performance, particularly evidencing resilience to natural image distribution shifts. Nevertheless, its proficiency waned in more specialized fields like medicine and chemistry, signaling potential areas for enhancement. This was notably apparent in datasets such as Camelyon17, NIH-Chest, DrugOOD Assay, and DrugOOD Scaffold, where GPT-4V’s classification outcomes resembled random guesses, as detailed in Table 3. This pattern suggests a need for targeted improvements in these domain-specific contexts.
- • Adaptability to Controlled Data Perturbations: The experiments in Section 4 utilizing ControlNetgenerated and random noise-induced data distributions presented GPT-4V with entirely novel domains, distinct from those involved in its pretraining phase. This setup rigorously tests the model’s generalization capabilities in handling out-of-distribution scenarios. As demonstrated in Table 5, GPT-4V almost surpassed other methods in its performance, excelling particularly with challenging samples and in situations where CLIP encountered failures. These results underscore GPT-4V’s exceptional stability

‡Namely, the gpt-4-vision-preview model via the OpenAI API.

and reliability when confronted with controlled perturbations and novel data distributions, highlighting its robust generalization abilities.

- • In-context Learning Is an Effective Method: The experiments detailed in Section 5 illuminate the efficacy of in-context learning in enhancing GPT-4V’s adaptability to distribution shifts. Notably, in the case studies depicted in Figure 5, GPT-4V demonstrates its capability to accurately identify the class of pathological images by discerning differences compared to two source images. This adaptability was consistently mirrored across four distinct datasets, reinforcing the utility of in-context learning strategies in navigating distribution shifts. Looking forward, there is a promising avenue for developing more sophisticated in-context learning methods, aiming to further bolster GPT-4V’s robustness across diverse data distributions.
- • Detail-Oriented Classification Rationale: The classification rationale provided by GPT-4V reflects a nuanced and detailed understanding of image elements, illustrating its sophisticated content comprehension. For instance, as exemplified in Figure 26, GPT-4V’s capability outshines that of LLaVA by accurately recognizing distinct characteristics such as a robust body, short tail, and tufted ears. These instances clearly demonstrate GPT-4V’s advanced ability to discern and articulate finer details in images, further reinforcing its superiority in complex image classification tasks under distribution shifts.
- • Higher Confidence in Predictions: GPT-4V consistently displayed higher and more justified confidence levels in its predictions, indicative of a confident and precise decision-making process. As illustrated in Figure 26, GPT-4V’s detail-oriented classification rationale contributes to its generating higher confidence scores compared to LLaVA. For instance, in Figure 17, GPT-4V achieves a peak confidence score with a descriptive analysis: “The image shows a metal kettle with a spout, handle, and thermometer on the top, which is a common design for a kettle used to heat water." Conversely, in medical imaging scenarios, such as depicted in Figure 31, GPT-4V’s confidence scores are more moderate, often accompanied by recommendations for further clinical testing, reflecting a prudent approach in high-stakes contexts.
- • Need for Domain-Specific Fine-Tuning: GPT-4V’s performance in fields requiring specialized knowledge, such as medicine, chemistry, and biology, highlights the need for further fine-tuning using domain-specific data. While GPT-4V often provides rational and contextually appropriate reasoning, it can still yield incorrect classifications or diagnoses. A case in point is Figure 11, where GPT-4V accurately describes an image labeled as a guitar, stating that “the image displays a stylized depiction of a guitar ... leading to high confidence in this identification,” yet it incorrectly classifies the image as a person. This example underscores the critical need for domain-specific fine-tuning, especially in areas where precision and reliability are paramount. Incorporating domain-specific knowledge and data into GPT-4V could substantially improve its accuracy, ensuring that its sophisticated reasoning consistently aligns with accurate contextual interpretations and decisions.
- • Consistency in Challenging Samples: GPT-4V showcased remarkable consistency in handling challenging samples, particularly in scenarios where CLIP encountered errors. Its performance was notably superior to that of LLaVA, exhibiting enhanced adaptability and precision. This is clearly evidenced in Tables 1 and 3, where, in instances of failure cases, GPT-4V almost outperforms both LLaVA and CLIP by a significant margin. These findings highlight GPT-4V’s robustness and efficacy in dealing with complex samples, especially those involving significant distribution shifts.
- • Limitations in Applicability for Certain Tasks: GPT-4V struggles with classification tasks when labels lack semantic information. This limitation becomes evident in scenarios such as activity identification tasks involving chemical molecular structures. In these cases, where sample labels are simply ‘active’ or ‘inactive,’ both GPT-4V and LLaVA tend to perform no better than random guessing. The provided reasoning, such as “The image shows a chemical structure, which does not have an active or inactive state in the context of physical motion or activity," as highlighted in Table 3 and Figure 33, reveals a gap in context comprehension. Similarly, tasks with numerical labels also pose a challenge for GPT-4V’s zero-shot classification capabilities. These findings underscore the need for additional adaptation or fine-tuning for downstream tasks that involve non-semantic labels.
- • Consistent Improvements as the Model Evolves: Initially, our evaluation focused on GPT-4V, a leading-edge LMM. During this phase, the introduction of Gemini presented an opportunity to broaden our analysis. Consequently, we integrated Gemini, facilitating a more comprehensive comparison. This progression, from CLIP, LLAVA, and GPT-4V to Gemini, represents a remarkable trajectory of

continuous enhancement in zero-shot generalization capabilities across a wide range of natural domains. Particularly, Gemini showcases near state-of-the-art performance, as evidenced in Tables 1 and 2. While our focus remains on GPT-4V’s performance, the inclusion of Gemini enriches our analysis. Additionally, the codes and datasets are publicly available, encouraging ongoing validation and study of the latest advancements in LMMs.

###### 3 Zero-shot Generalization Across Varied Domains

This section delineates our findings on the zero-shot generalization capabilities of GPT-4V in the context of distribution shifts, as enumerated in Table 1 and 3. We compare the performance of GPT-4V with baseline models such as CLIP§, LLaVA¶, and Gemini Pro Vision‖, highlighting its effectiveness and limitation across a variety of domains. Our investigation categorizes the datasets into three distinct groups: natural visuals, medical images, and molecular images. For each category, we first provide an overview of the collective results, showcasing GPT-4V’s generalization performance. This is followed by in-depth case studies, where we delve into specific instances to uncover nuanced insights about the model’s performance in diverse and challenging scenarios.

###### 3.1 Natural Images

###### 3.1.1 Task Introduction

The category of natural visuals encompasses an extensive array of real-world imagery, capturing the myriad facets of nature and everyday life. This domain is characterized by its inherent diversity and complexity, presenting scenes and objects that are commonly encountered in daily experiences.

In our study, we examine the following natural datasets, each with its distinct characteristics and challenges:

- • PACS (Li et al., 2017): Comprising images from four different styles - art painting, cartoon, photo, and sketch - this dataset challenges models to generalize across artistic mediums, testing their ability to recognize the same objects in vastly different visual representations.
- • VLCS (Fang et al., 2013): This dataset is a collection from four different image repositories. It poses a challenge in terms of variations in image quality, lighting, and backgrounds, requiring robust feature extraction for successful classification.
- • Office-Home (Venkateswara et al., 2017b): Featuring everyday objects from office and home environments, this dataset includes images from diverse categories such as Art, Clipart, Product, and Real World, offering a testbed for models to generalize across everyday items.
- • DomainNet (Peng et al., 2019b): Encompassing a variety of artistic styles and objects, DomainNet is a large-scale dataset that tests a model’s ability to generalize across different visual domains and a vast array of object classes.
- • Fmow (Christie et al., 2018): This dataset focuses on land use and land cover classification, presenting a challenge with its time-series satellite imagery, which includes temporal and regional variations.
- • TerraIncognita (Beery et al., 2018): Composed of wildlife images captured by camera traps in various locations, it tests models’ abilities to recognize animal species across different environmental conditions and camera settings.
- • iWildCam (Beery et al., 2021): The iWildCam dataset offers a unique challenge in the realm of wildlife conservation and ecological studies. Comprised of images captured by camera traps set up in diverse wilderness locations, it is tailored to evaluate the ability of models to identify and classify a wide range of animal species.

These datasets not only cover a wide range of natural scenes and objects but also introduce various types of distribution shifts, making them ideal for evaluating the zero-shot generalization capabilities of GPT-4V, in comparison with CLIP, LLaVA, and Gemini. Each dataset presents its unique set of challenges, from

§https://huggingface.co/openai/clip-vit-base-patch16 ¶https://huggingface.co/liuhaotian/llava-v1.5-13b ‖https://ai.google.dev/models/gemini

Table 1 | Summary of zero-shot generalization performance across various natural datasets, showcasing the comparative results of GPT-4V (gpt-4-vision-preview) with CLIP (clip-vit-base-patch16), LLaVA (llava-v1.5-13b) and Gemini (gemini-pro-vision) models.

Dataset PACS VLCS Office-Home DomainNet Fmow TerraIncognita iWildCam Category natural natural natural natural natural natural natural Prediction animal species animal species everyday items objects, creatures land use animal species animal species Domain artistic media image repositories visual categories artistic styles time, region camera trap location #domains 4 4 4 6 6 4 206 #classes 7 5 65 345 62 10 323

[Figure 1] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile1.png>)

[Figure 2] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile2.png>)

[Figure 3] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile3.png>)

[Figure 4] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile4.png>)

[Figure 5] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile5.png>)

[Figure 6] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile6.png>)

[Figure 7] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile7.png>)

Examples

random samples (180 cases) CLIP

0.967 0.833 0.800 0.572 0.111 0.194 0.061 174/180 150/180 144/180 103/180 20/180 35/180 11/180 LLaVA

0.994 0.894 0.650 0.306 0.128 0.539 0.006

179/180 161/180 117/180 55/180 23/180 97/180 1/180 GPT-4V

0.978 0.797 0.936 0.833 0.220 0.500 0.309 175/179 141/177 160/171 135/162 39/177 90/180 55/178 Gemini

0.983 0.871 0.963 0.910 0.333 0.483 0.396 173/176 148/170 155/161 142/156 56/168 87/180 71/179

random samples (1800 cases) CLIP

0.961 0.808 0.778 0.582 0.161 0.214 0.064 1730/1800 1455/1800 1400/1800 1048/1800 290/1800 385/1800 116/1800 LLaVA

0.982 0.852 0.703 0.370 0.147 0.488 0.014

1768/1800 1534/1800 1265/1800 666/1800 264/1800 879/1800 25/1800 GPT-4V

0.969 0.888 0.889 0.680 0.238 0.459 0.265 1742/1797 1455/1799 1599/1800 1162/1710 428/1800 827/1800 473/1787 Gemini

0.993 0.838 0.922 0.754 0.271 0.519 0.343 1770/1782 1445/1724 1528/1658 1214/1611 473/1743 931/1794 600/1750

failure cases CLIP

0.000 0.000 0.000 0.000 0.000 0.000 0.000 0/173 0/180 0/180 0/180 0/180 0/180 0/180

0.751 0.517 0.406 0.128 0.083 0.517 0.016

LLaVA

130/173 93/180 73/180 23/180 15/180 93/180 3/180 GPT-4V

0.732 0.651 0.774 0.523 0.192 0.411 0.285 120/164 112/172 127/164 78/149 32/167 74/180 51/179 Gemini

0.848 0.650 0.860 0.736 0.266 0.458 0.431 140/165 104/160 141/164 106/144 45/169 82/179 75/174

artistic style variations in PACS to environmental differences in TerraIncognita, providing a comprehensive testbed for assessing model robustness in natural settings. Table 1 firstly provides an overview of each natural dataset, detailing key aspects such as the type of prediction, domain characteristics, the number of domains and classes, and illustrative examples. This table serves as a quick reference to understand the diversity and scope of challenges posed by these datasets in our evaluation.

###### 3.1.2 Comparative Accuracies Across Datasets and Domains

- Table 1 outlines the accuracies and correct-to-total case ratios for four models (CLIP, LLaVA, Gemini, and GPT-4V) across seven natural datasets, incorporating both random samples and failure cases identified in CLIP. This subsection is dedicated to examining GPT-4V’s zero-shot generalization abilities within natural datasets.

GPT-4V’s Performance in Random Samples: Focusing first on datasets with a large variety of domains and classes, such as Office-Home and DomainNet, GPT-4V demonstrates a notable capacity for generalization. Its high accuracy rates in Office-Home (0.889) and DomainNet (0.680) suggest a robust understanding and adaptability to diverse natural visuals, including a broad range of everyday items and varied artistic styles.

Method Office-Home PACS DomainNet TerraIncognita VLCS Avg. MMD (Li et al., 2018c) 0.663 0.847 0.234 0.422 0.775 0.588 Mixstyle (Zhou et al., 2021) 0.604 0.852 0.340 0.440 0.779 0.603 GroupDRO (Sagawa et al., 2019) 0.660 0.844 0.333 0.432 0.767 0.607 IRM (Arjovsky et al., 2019) 0.643 0.835 0.339 0.476 0.785 0.616 CDANN (Li et al., 2018b) 0.658 0.826 0.383 0.458 0.775 0.620 DANN (Ganin et al., 2016) 0.659 0.836 0.383 0.467 0.786 0.626 MTL (Blanchard et al., 2021) 0.664 0.846 0.406 0.456 0.772 0.629 Mixup (Xu et al., 2020) 0.681 0.846 0.392 0.479 0.774 0.634 MLDG (Li et al., 2018a) 0.668 0.849 0.412 0.477 0.772 0.636 ERM (Vapnik, 1999) 0.676 0.842 0.440 0.478 0.773 0.642 SagNet (Nam et al., 2021) 0.681 0.863 0.403 0.486 0.778 0.642 SelfReg (Kim et al., 2021) 0.679 0.856 0.428 0.470 0.778 0.642 CORAL (Sun and Saenko, 2016) 0.687 0.862 0.415 0.476 0.788 0.645 mDSDI (Bui et al., 2021) 0.692 0.862 0.428 0.481 0.790 0.651 ERM + MIRO (Cha et al., 2022) 0.705 0.854 0.443 0.504 0.790 0.659 ERM + SWAD (Cha et al., 2021) 0.706 0.881 0.465 0.500 0.791 0.669 CORAL + SWAD (Cha et al., 2021) 0.713 0.883 0.468 0.510 0.789 0.673 DIWA (Rame et al., 2022) 0.728 0.890 0.477 0.519 0.786 0.680 ERM + MIRO + SWAD (Cha et al., 2021) 0.724 0.884 0.470 0.529 0.796 0.681 ERM++ (Teterwak et al., 2023) 0.747 0.898 0.508 0.512 0.780 0.689 CLIP (Radford et al., 2021) 0.778 0.961 0.582 0.214 0.808 0.669 LLaVA (Liu et al., 2023b;a) 0.703 0.982 0.370 0.488 0.852 0.679 GPT-4V (OpenAI, 2023) 0.889 0.969 0.680 0.459 0.888 0.777 Gemini (Team et al., 2023) 0.922 0.993 0.754 0.519 0.838 0.805

- Table 2 | Zero-shot Generalization Performance of GPT-4V on DomainBed: In the DomainBed benchmark for domain generalization, GPT-4V demonstrates superior zero-shot generalization capabilities, surpassing traditional approaches and marking a significant advancement in the field. The results highlight GPT-4V’s effectiveness across diverse domains, showcasing its potential for robust and versatile applications.

Additionally, in uncommon datasets like Fmow and TerraIncognita, GPT-4V significantly surpasses CLIP’s performance (0.238 vs 0.161 in Fmow and 0.459 vs 0.214 in TerraIncognita). In the PACS and VLCS datasets, all three models perform well, with accuracies exceeding 0.8. This consistency suggests that these domains may have been included in the pre-training data of these three models.

GPT-4V in Handling CLIP’s Failure Cases: To assess GPT-4V’s capabilities in more challenging scenarios, we examine its performance on CLIP’s failure cases. In datasets with a diverse range of classes, such as DomainNet and Office-Home, GPT-4V shows remarkable resilience. For instance, in Office-Home, GPT-4V achieves an accuracy of 0.774, surpassing LLaVA’s 0.406. Similarly, in DomainNet, GPT-4V records 0.523 accuracy, significantly higher than LLaVA’s 0.128. This trend is also evident in Fmow, where GPT-4V’s performance (0.192) markedly exceeds LLaVA’s (0.083). These results indicate GPT-4V’s robustness in handling complex and challenging visuals, even in scenarios where CLIP struggled.

GPT-4V’s Performance Across Individual Domains: While Table 1 provides an overall view of the accuracies for the three models across various datasets, a more granular look at their performance in specific domains is essential for a comprehensive understanding. To this end, we have detailed comparative domain accuracies for each model within the PACS, VLCS, Office-Home, DomainNet, Fmow, and TerraIncognita datasets. These comparisons are illustrated in Figures 7, 8. These figures illuminate the relative strengths and weaknesses of each model across different domains within the datasets and help to understand the extent of GPT-4V’s generalization capabilities and how it compares to CLIP and LLaVA in diverse contexts.

Highlighting GPT-4V’s Superiority in DomainBed: In the context of DomainBed (Gulrajani and Lopez-Paz, 2020), the popular benchmark for domain generalization, Table 2 provides a clear illustration of the strides made by GPT-4V. It achieves unparalleled zero-shot generalization performance, significantly outpacing traditional domain generalization methods. Its exceptional performance across the board is indicative of its sophisticated understanding and the ability to adapt to new, unseen domains. This achievement is not just a

reflection of GPT-4V’s powerful architecture but also an indicator of its potential to revolutionize how models tackle the challenge of domain generalization.

###### 3.1.3 Case Demonstration

The diverse array of case studies presented in Figures 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 23, 24, 26, 27 and 28 showcase the adeptness of GPT-4V and LLaVA in navigating the challenges posed by different datasets, including PACS, VLCS, Office-Home, DomainNet, Fmow, and TerraIncognita. These examples not only demonstrate GPT-4V’s proficiency in accurately recognizing natural distribution shifts in a zero-shot setting but also highlight its ability to adapt to various visual domains and object classifications. Additionally, Figures 12, 16, 20, 22, 25 and 29 provide insights into instances where GPT-4V does not perform optimally, shedding light on the model’s limitations and areas for improvement.

A key observation emerging from these case studies is the nuanced capability of GPT-4V to discern intricate details within images. For instance, GPT-4V exhibits its adeptness at identifying textual elements in Figure 9. Figure 17 demonstrates a keen eye for specific features, such as the metallic nature and the spout of a kettle, highlighting its attention to detail. Furthermore, in Figure 26, GPT-4V distinguishes finer characteristics like a short tail and tufted ears in identifying a bobcat, a task that poses a challenge even for human observers.

###### 3.2 Medical Images

###### 3.2.1 Task Introduction

We investigate the classification capabilities of different models in medical imaging applications under scenarios of distributional shifts. Distributional shifts are particularly common in the field of medical imaging, as changes in imaging technology, patient demographic characteristics, and disease manifestation can significantly alter the data distribution. Exploring the generalizability of the GPT-4 vision large model in medical image analysis tasks holds significant practical value.

In this part, we examine the following medical datasets, each with its distinct characteristics and challenges:

- • Camelyon17 (Bandi et al., 2018): The dataset contains 450,000 patch samples, which were derived from 50 whole-slide images (WSIs) featuring breast cancer metastases in lymph node sections. These WSIs were sourced from five different hospitals in the Netherlands, contributing 10 WSIs each. Pathologists meticulously annotated each WSI to identify tumor regions, and these annotations were used to create segmentation masks. These masks, in turn, provided the basis for assigning labels to each individual patch in the dataset.
- • HAM10000 (Tschandl et al., 2018): The dataset is a critical resource for research in skin lesion analysis, particularly focusing on generalization tasks. This dataset features a wide variety of dermatoscopic images, including numerous skin lesion types such as melanoma, basal cell carcinoma, and benign nevi. It is especially valuable for training and evaluating machine learning models on skin cancer detection and diagnosis. The diversity of images, sourced from different populations and equipment, makes HAM10000 ideal for studying and improving OOD generalization in medical imaging algorithms. This aspect is crucial for developing robust models capable of performing accurately across varied and unseen data, reflecting real-world clinical scenarios.
- • NIH-Chest (Wang et al., 2017): The NIH Chest X-ray Dataset, a substantial medical imaging collection from the National Institutes of Health, is pivotal for research in out-of-distribution (OOD) generalization and distribution shift challenges in medical imaging. Comprising over 112,000 frontal-view X-ray images from more than 30,000 patients, this dataset is annotated with 14 common thoracic pathologies, such as pneumonia and lung nodules. Its vast and diverse array of patient images, captured under various clinical settings and conditions, provides an exceptional resource for developing and testing machine learning models, particularly in assessing and improving their robustness and performance in the face of distributional shifts and OOD data, which are common obstacles in real-world medical diagnostics.
- • COVID (Han et al., 2021): This dataset serves as a resource for pneumonia detection, encompassing samples of normal cases, typical pneumonia, and COVID-19 pneumonia. The data, sourced from various

- Table 3 | Main results of zero-shot generalization performance across distribution shifts on medical and molecule datasets. Specifically, CLIP refers to clip-vit-base-patch16, LLaVA refers to llava-v1.5-13b, Gemini refers to gemini-pro-vision, GPT-4V refers to gpt-4-vision-preview.

Dataset Camelyon17 HAM10000 NIH-Chest COVID DrugOOD_Assay DrugOOD_Scaffold

Category medical medical medical medical molecule molecule Prediction tumor skin diseases lung disease pneumonia types bioassays bioassays Domain hospital hospital hospital hospital assay scaffold #domains 5 4 2 2 81 12543 #classes 2 7 15 3 2 2

[Figure 8] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile8.png>)

[Figure 9] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile9.png>)

[Figure 10] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile10.png>)

[Figure 11] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile11.png>)

[Figure 12] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile12.png>)

[Figure 13] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile13.png>)

Examples

random samples (180 cases) CLIP

0.506 0.161 0.0 78 0.360 0.517 0.533

- 91/180 29/180 14/180 36/100 93/180 96/180

LLaVA

0.508 0.100 0.044 0.450 0.517 0.533

- 92/180 18/180 8/180 45/100 93/180 96/180

0.518 0.302 0.055 0.354 0.494 0.472 72/139 49/162 6/108 28/79 89/180 68/144 Gemini

GPT-4V

0.534 0.305 0.104 0.64 0.467 0.459 94/176 53/174 17/163 64/100 84/180 79/172

random samples (1800 cases) CLIP

0.497 0.226 0.076 0.490 0.521 0.477 894/1800 406/1800 137/1800 882/1800 924/1772 858/1800 LLaVA

0.508 0.160 0.089 0.420 0.521 0.477 914/1800 288/1800 160/1800 756/1800 923/1772 859/1800 GPT-4V

0.513 0.341 0.084 0.313 0.488 0.514 923/1799 548/1606 45/535 380/1216 414/848 647/1258 Gemini

0.532 0.335 0.119 0.515 0.490 0.508 940/1766 572/1705 206/1729 926/1798 869/1772 914/1800

failure cases CLIP

0.000 0.000 0.000 0.000 0.000 0.000 0/180 0/180 0/180 0/100 0/180 0/180

0.028 0.067 0.056 0.510 0.000 0.006 5/180 12/180 10/180 51/100 0/180 1/180

LLaVA

1.000 0.308 0.102 0.543 1.000 1.000 157/157 49/159 6/59 38/70 179/179 180/180 Gemini

GPT-4V

1.000 0.28 0.093 0.93 1.000 1.000 176/176 49/175 14/150 93/100 180/180 180/180

hospitals due to collection methodologies, exhibit distributional shifts. We utilize this dataset to assess model performance in pneumonia detection tasks under conditions of distributional shift, reflecting real-world variations in medical data collection and patient demographics.

These datasets encompass a diverse array of medical scenarios and tasks, while also presenting a variety of distribution shifts. This diversity positions them as prime candidates for assessing the zero-shot generalization abilities of the GPT-4V model, with comparative analysis against CLIP, LLaVA, and Gemini. Table 3 offers a comprehensive overview of each dataset, highlighting crucial elements like prediction types, domain specifics, the range of domains and classes, along with representative examples.

###### 3.2.2 Comparative Accuracies Across Datasets and Domains

- Table 3 outlines the accuracies and correct-to-total case ratios for three models (CLIP, LLaVA, and GPT-4V) across four medical datasets, incorporating both random samples and failure cases identified in CLIP. This subsection is dedicated to examining GPT-4V’s zero-shot generalization abilities within medical datasets.

GPT-4V’s Performance in Random Samples: According to Table 3, it is observed that the performance of GPT-4V, Gemini, CLIP, and LLaVA on medical image classification tasks is quite average. For instance, on the Camelyon17 dataset, the performances of GPT-4V, Gemini, CLIP, and LLaVA are 0.513, 0.532, 0.497, and 0.508, respectively. This suggests that the data from these datasets may not have been present in the training sets of these four models, highlighting a potential gap in their pre-training data and indicating the need for further model training or adaptation to improve performance in these specific medical tasks.

GPT-4V in Handling CLIP’s Failure Cases: To assess GPT-4V’s capabilities in more challenging scenarios, we examine its performance in CLIP’s failure cases. On the HAM10000 dataset, GPT-4V achieved an accuracy of 0.308, surpassing LLaVa’s 0.067. There were also varying degrees of accuracy improvements on the NIH-Chest and COVID datasets. These results demonstrate GPT-4V’s robustness in handling complex and challenging visual tasks, maintaining stable performance even in scenarios where CLIP struggled.

###### 3.2.3 Case Demonstration

The diverse array of case studies presented in Figures 31 and 32 showcase the adeptness of GPT-4V and LLaVA in navigating the challenges posed by different datasets, including HAM10000, NIH-Chest, and COVID.

###### 3.3 Scientific Images

###### 3.3.1 Task Introduction

Our research investigates the performance of various computational models in scientific fields, with a focus on predicting molecular properties amid distributional shifts due to variations in scaffolds and assays. Such shifts, resulting from changes in molecular scaffolds and assay conditions, profoundly affect the nature of scientific datasets. Assessing how advanced models like GPT-4 can adapt to these variations is vital for enhancing their predictive accuracy and reliability in the dynamic landscape of molecular science, where the intricate interplay of molecular structure and assay environments shapes data diversity and complexity.

In this part, we examine the following scientific datasets, each with its distinct characteristics and challenges: DrugOOD (Ji et al., 2023) is a comprehensive dataset curator and benchmarking tool specifically designed for AI-aided drug discovery (AIDD). It focuses on the critical challenge of drug-target binding affinity prediction, involving both macromolecules (protein targets) and small molecules (drug compounds). Unlike traditional fixed datasets, DrugOOD offers automated data curation with customizable scripts, rich domain annotations, realistic noise annotations, and robust benchmarking of state-of-the-art OOD algorithms. It is particularly useful for testing graph-based out-of-distribution learning problems, crucial in molecular data modeled as irregular graphs. DrugOOD_Assay and DrugOOD_Scaffold can be obtained by splitting the domains with assays and scaffolds.

- • DrugOOD_Assay (Ji et al., 2023): In the DrugOOD_Assay, domains are delineated based on the assay. This means that samples generated from the same assay are classified into the same domain, reflecting the unique environmental conditions of each assay. Due to these varying conditions, activity values measured across different assays exhibit a natural distribution shift. Consequently, the model is challenged to perform on data from bioassay environments it has not previously seen, testing its ability to generalize and maintain accuracy in the face of diverse and novel assay environments.
- • DrugOOD_Scaffold (Ji et al., 2023): In the DrugOOD_Scaffold dataset, the domains are defined based on different molecular scaffolds. Molecules with the same molecular scaffold are grouped into the same domain, following the approach outlined by (Koh et al., 2021; Hu et al., 2021b). This structuring emphasizes the importance for models to have the capability to generalize effectively to unseen domains that are characterized by novel scaffolds, thereby enabling accurate predictions across a broad spectrum of molecular structures.

These datasets encompass a diverse array of scientific scenarios, while also presenting a variety of distribution shifts. This diversity positions them as prime candidates for assessing the zero-shot generalization abilities of the GPT-4V model, with comparative analysis against CLIP and LLaVA. Table 3 offers a comprehensive overview of each dataset, highlighting crucial elements like prediction types, domain specifics, the range of domains and classes, along representative examples.

###### 3.3.2 Performance Across Datasets and Domains

The results show that, in both the DrugOOD_Assay and DrugOOD_Assay datasets, GPT-4V, CLIP, and LLaVA failed. They were ineffective in accurately predicting the categories of molecules. The reasons for their failures could be attributed to three main factors: First, the complexity of the scientific task. Second, these datasets were not included in the training sets of these three models. Third, the ambiguity in data labeling, for instance, the labels ‘inactive’ and ‘active’ in scientific datasets are different from natural dataset labels like ‘elephant’ or ‘bike’. The use of ‘inactive’ and ‘active’ as class labels is more ambiguous and lacks specific meaning. In conclusion, it is understandable that the zero-shot classification capabilities of these three models are poor.

###### 3.3.3 Prompt Engineering Trick

This study explores the significant role of the Prompt Engineering Trick in enhancing performance in scientific image classification tasks. Specifically, we applied this technique in the task of chemical structure-activity classification, achieving a notable improvement in classification accuracy from 51.4% to 52.5%. This approach involves introducing meticulously designed prompts, such as instructing the model to analyze molecular structure images in the role of a chemistry expert, as shown in Figure 3. We required the model to not only identify atomic arrangements and bonding patterns in the images but also to interpret the overall configuration of the molecule to determine its chemical reactivity as either active or inactive. This method not only improved classification accuracy but also made the model’s reasoning process more logical and interpretable. This research demonstrates that carefully designed prompts can significantly enhance the performance and understanding of machine learning models in specific tasks.

###### 3.3.4 Case Demonstration

The representative case study presented in Figure 33 showcases the adeptness of GPT-4V and LLaVA in navigating the challenges. The results in Figures 33 show that GPT-4V does not perform well in predicting molecular properties. Although LLaVA can correctly predict the molecular properties, its reasoning is not convincing, suggesting that LLaVA’s correct predictions are merely guesses without any solid basis. In contrast, although GPT-4V does not make accurate predictions, it does not provide a confidence level, and its reasoning is more logical. Therefore, to some extent, GPT-4V is more reliable than LLaVA.

###### 4 Adaptability to Controlled Data Perturbations

To assess GPT-4V’s adaptability to entirely new distribution shifts, our methodology encompasses two distinct strategies: (1) Noise Injection: We introduce Gaussian noise into the PACS, VLCS, and Office-Home datasets to artificially create variations in data distribution. (2) Domain Shift Generation: Utilizing ControlNet Zhang et al. (2023), we generate datasets that exhibit domain shifts. These datasets are designed to significantly differ from those used in the pretraining phase and are not encountered by the model during its initial training. This approach allows us to systematically evaluate GPT-4V’s performance across datasets that vary substantially from the pretraining data.

###### 4.1 Gaussian Noise

###### 4.1.1 Comparative Accuracies Across Domains

- Table 4 outlines the accuracies and correct-to-total case ratios for four models (CLIP, LLaVA, Gemini, and GPT-4V) across PACS_gaussian, VLCS_gaussian, and Office-Home_gaussian, incorporating both random samples and failure cases identified in CLIP. This subsection is dedicated to examining GPT-4V’s zero-shot

###### Prompt:

As a chemistry expert specializing in molecular structures and chemical properties, your task is to examine the provided image of a chemical molecular structure. Using your profound knowledge in this field, assess the arrangement and types of atoms, bonding patterns, and overall molecular configuration. Your expertise will enable you to identify key indicators within the molecular structure that suggest whether it is 'active' or 'inactive' in terms of chemical reactivity. Please focus on aspects such as bond types, atom positioning, and molecular geometry, which are critical in determining the activity status of the molecule. Your detailed analysis should lead to a conclusion about the molecule's potential reactivity or stability based on the observable features in the structure. Question: What is in this image? \\ Choices: [active, inactive]. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE---

Do not deviate from the above format. Repeat the format template for the answer.

- Figure 3 | An illustration of a structured prompt format used in the PACS dataset, showcasing a specific approach for image-based questioning and response formatting. The format includes a question about the image’s content, a list of answer choices, and a template for answering, including an answer, confidence score, and the reasoning process.

generalization abilities within datasets with distribution shifts.

GPT-4V’s Performance in Random Samples: Focusing initially on datasets encompassing a broad range of domains and categories, like Office-Home_gausssion, GPT-4V showcases remarkable generalization capabilities. Its impressive accuracy rate of 87.4% in Office-Home_gausssion is a testament to GPT-4V’s adeptness in managing distribution shifts, especially those with Gaussian noise. In the PACS_Gaussian dataset, all three models exhibit strong performance, each surpassing an accuracy rate of 95%. This uniformity in performance hints that PACS_gausssion might have been a part of the foundational training data for these models.

GPT-4V in Handling CLIP’s Failure Cases: To evaluate GPT-4V’s performance in more challenging scenarios, we examined its response to cases where CLIP had failed. In datasets with a wide range of categories, such as Office-Home_gausssion, GPT-4V demonstrated significant resilience. For instance, in

- Table 4 | Main results of zero-shot generalization performance across distribution shifts created by adding Gaussian noise. Specifically, CLIP refers to clip-vit-base-patch16, LLaVA refers to llava-v1.5-13b, Gemini refers to gemini-pro-vision, GPT-4V refers to gpt-4-vision-preview.

Dataset PACS_gaussian VLCS_gaussian Office-Home_gaussian PACS_gaussian VLCS_gaussian Office-Home_gaussian

random samples failure cases CLIP

0.961 0.799 0.741 0.000 0.000 0.000

1729/1800 1439/1800 1334/1800 0/180 0/180 0/180 LLaVA

0.985 0.857 0.682 0.784 0.589 0.433

1773/1800 1542/1800 1229/1800 105/134 106/180 78/180 GPT-4V

0.972 0.810 0.874 0.707 0.568 0.790

1750/1800 1043/1287 1550/1773 70/99 100/176 132/167 Gemini

0.989 0.841 0.921 0.850 0.540 0.851 1729/1749 1414/1682 1480/1607 102/120 81/150 137/161

- Table 5 | Main results of zero-shot generalization performance across distribution shifts created by ControlNet. Specifically, CLIP refers to clip-vit-base-patch16, LLaVA refers to llava-v1.5-13b, Gemini refers to gemini-pro-vision, GPT-4V refers to gpt-4-vision-preview.

Dataset PACS_unseen VLCS_unseen Office-Home_unseen PACS_unseen VLCS_unseen Office-Home_unseen

random samples failure cases CLIP

0.992 0.924 0.722 0.000 0.000 0.000

1786/1800 1633/1768 1299/1800 0/16 0/135 0/180 LLaVA

0.996 0.962 0.618 0.813 0.726 0.250

1793/1800 1700/1768 1113/1800 13/16 98/135 45/180 GPT-4V

0.989 0.932 0.755 0.875 0.880 0.611 731/739 1096/1176 935/1238 14/16 117/133 110/180 Gemini

0.995 0.942 0.794 0.733 0.770 0.579 1763/1772 1627/1728 1283/1615 11/15 97/126 95/164

Office-Home_gausssion, GPT-4V achieved an accuracy rate of 79.0%, surpassing LLaVA’s 35.7%. In both PACS_gausssion and VLCS_gausssion datasets, GPT-4V consistently outperformed LLaVA. These results highlight GPT-4V’s robustness in handling complex and challenging visual scenarios, even in situations where CLIP encountered difficulties.

###### 4.1.2 Case Demonstration

The diverse array of case studies presented in Induced Distribution Shift: Cases 1 and 2 of the Appendix showcase the adeptness of GPT-4V and LLaVA in navigating the challenges posed by different datasets, including PACS_gaussian, Office-Home_gaussian, and VLCS_gaussian. These examples not only demonstrate GPT-4V’s proficiency in accurately recognizing natural distribution shifts under Gaussian noise incorporation but also highlight its ability to adapt to various visual domains and object classifications.

###### 4.2 Style Change with ControlNet

###### 4.2.1 Comparative Accuracies Across Domains

Table 5 outlines the accuracies and correct-to-total case ratios for three models (CLIP, LLaVA, and GPT-4V) across PACS_unseen, VLCS_unseen, and Office-Home_unseen, incorporating both random samples and failure cases identified in CLIP. This subsection is dedicated to examining GPT-4V’s zero-shot generalization abilities within datasets with domain shift created by ControlNet.

GPT-4V’s Performance in Random Samples: Focusing initially on datasets encompassing a broad range of domains and categories, like Office-Home_unseen, GPT-4V showcases remarkable generalization capabilities. Its impressive accuracy rate of 75.5% in Office-Home_unseen is a testament to GPT-4V’s adeptness in managing distribution shifts created by ControlNet. In the PACS_unseen and VLCS_unseen, all three models exhibit strong performance, each surpassing an accuracy rate of 90%. This uniformity in performance hints that PACS_unseen and VLCS_unseen might have been a part of the foundational training data for these models.

GPT-4V in Handling CLIP’s Failure Cases: To evaluate GPT-4V’s performance in more challenging scenarios, we examined its response to cases where CLIP had failed. In datasets with a wide range of categories, such as Office-Home_unseen, GPT-4V demonstrated significant resilience. For instance, in Office-Home_unseen, GPT-4V achieved an accuracy rate of 61.1%, surpassing LLaVA’s 25.0%. In both PACS_unseen and VLCS_unseen datasets, GPT-4V consistently outperforms LLaVA. These results highlight GPT-4V’s robustness in handling challenging visual scenarios, even in situations where CLIP encountered difficulties.

###### 4.2.2 Case Demonstration

The diverse array of case studies presented in Figure 37, 36, and 38 showcase the adeptness of GPT-4V and LLaVA in navigating the challenges posed by different datasets, including PACS_unseen, Office-Home_unseen,

and VLCS_unseen. These examples not only demonstrate GPT-4V’s proficiency in accurately recognizing natural distribution shifts created by ControlNet incorporation but also highlight its ability to adapt to various visual domains and object classifications. However, under certain complex samples, such as Figure 39, 40, and 41, both GPT-4V and LLaVA still have their limitations. They are prone to being misled by irrelevant factors in the image, leading to incorrect predictions.

###### 5 Exploiting In-Context Learning for Domain Bridging

Addressing distribution shifts traditionally involves fine-tuning pre-trained foundational models with source domain data to facilitate effective adaptation to target domains. While this approach can be effective, it often requires significant computational resources and time, especially for large foundational models (Hu et al., 2021a). Against this backdrop, our research shifts focus to the exploration of in-context learning capabilities of large multimodal models, with a specific emphasis on GPT-4V. This approach presents a novel method for simulating traditional domain generalization paradigms.

In-context learning, as defined by GPT-3 (Brown et al., 2020), involves conditioning the model on a set of natural language instructions alongside a few task demonstrations. The model is then expected to apply this learned context to complete further instances of the task, primarily through predicting subsequent sequences. This methodology leverages the model’s inherent ability to infer and apply patterns from limited information without any parameter update, a significant difference from conventional fine-tuning techniques. This ability of large foundation models to demonstrate emergent capabilities through in-context learning has been increasingly recognized and highlighted in recent studies (Wei et al., 2022b; Ouyang et al., 2022; Wei et al., 2022a; Wang et al., 2022b; Kojima et al., 2022). Our study aims to assess how effectively GPT-4V utilizes in-context learning to navigate distribution shifts across diverse domains (Ahuja and Lopez-Paz, 2023; Gupta et al., 2023).

###### 5.1 In-context Setup

For our in-context learning exploration, we focus on the Camelyon17 (Bandi et al., 2018), COVID (Han et al., 2021), DrugOOD_Assay (Ji et al., 2023) and NIH-Chest (Wang et al., 2017) datasets. These datasets were chosen due to GPT-4V’s previously observed underperformance, perhaps because the pre-training data distribution rarely includes scientific datasets like medical and protein. We wish the in-context learning that simulates conventional domain adaptation/generalization would enhance adaptability to certain tasks. In our experimental setup, we randomly select two classes within two domains of each dataset, designating them as source and target domains. From the source domain, we choose two representative examples for each class, like normal and typical pneumonia in the COVID dataset or normal and tumor in the Camelyon17 dataset, as illustrated in Figure 5. To demonstrate the potential of in-context learning as an effective approach for adapting large multimodal models to distribution shifts, we have intentionally limited our experiment to just one source domain and two examples. This decision is primarily driven by the constraints related to token cost. This setup emulates the concept of traditional out-of-distribution generalization but contrasts with it by leveraging the model’s innate ability to adapt to new contextual information while maintaining its original parameterization (Brown et al., 2020).

Below, we illustrate an example of an in-context prompt applied to the Camelyon17 dataset. This dataset is distinguished by its binary classification system, encompassing two distinct classes: ‘normal’ and ‘tumor’. In contrast to the basic prompt in Figure 2, we explicitly annotate the class labels for the two in-context examples provided to GPT-4V, i.e., ‘The first image is normal and the second image is tumor’. Furthermore, the prompt’s inquiry is subtly altered to ‘What is the third image?’, thereby aligning the model’s focus with the specific task of classification based on the provided contextual examples. The response format template is set the same as the previous basic prompt.

|Text Prompt with In-Context Examples: Given the image, answer the following question using the specified format. The first image is {class_1} and the second image is {class_2}. Question: What is the third image? Choices:[‘class_1’, ‘class_2’]. Please respond with the following format:<br><br>...|
|---|

Comparative Analysis of Accuracies with and without In-context Learning

0.8

gpt-4-vision-preview w/o in-context gpt-4-vision-preview w/ in-context

| |
|---|

0.7

+16.67%

+8.40%

0.6

+3.70%

0.5

0.4

0.3

+2.40%

0.2

0.1

Camelyon17 COVID DrugOOD_Assay NIH_Chest

Dataset

- Figure 4 | Improvements in target domain performance with in-context learning on GPT-4V across Camelyon17, COVID, DrugOOD_Assay and NIH_Chest datasets.

###### 5.2 In-context Performance

In Figure 4, we illustrate the impact of in-context learning when applied to the baseline GPT-4V model, specifically within the target domain. This approach demonstrates consistent performance enhancements across four distinct datasets. In particular, the application of in-context learning yields improvements of 3.7%, 8.4%, 2.4%, and 16.67% for the Camelyon17, COVID, DrugOOD_Assay, and NIH_Chest datasets, respectively. These results highlight the potential of in-context learning in boosting model adaptability, especially in situations characterized by distribution shifts.

The observed variability in performance gains across these datasets suggests a correlation between the inherent task complexity and the unique data distributions of each dataset. This aspect of the results prompts further investigation into the nuances of in-context learning and its differential impact based on dataset characteristics.

In our experimental setup, two examples were randomly selected from the source domain for the in-context learning process. However, a more deliberate selection of in-context examples could potentially lead to even greater improvements in model performance (Huang et al., 2023). This possibility opens avenues for future research, where the strategic choice of in-context examples could be explored as a means to optimize the efficacy of in-context learning.

###### 5.3 In-context Case Demonstration

This section showcases selected cases to demonstrate the enhancement of inference performance through in-context examples.

GPT-4V’s Interpretation of In-context Examples: Figure 5 features a case study within the Camelyon17 dataset. The procedure includes presenting GPT-4V with two annotated images from a source domain

(hospital_2): one denoted as ’normal’ and the other as ’tumor’. These are followed by a test image from a different domain (hospital_3). Conditioned with this contextual information, GPT-4V effectively discerns between the regular, uniform tissue patterns in the ’normal’ image and the abnormal, irregular cell structures in the ’tumor’ image. It then applies this discernment to precisely classify the test image from hospital_3. This case exemplifies how GPT-4V employs in-context examples to bridge different domains, enhancing its interpretive accuracy.

The Impact of In-context Examples: Figure 6 explores the influence of in-context learning on GPT-4V’s performance in classifying chest X-ray images. The figure presents a comparative analysis of the model’s accuracy with and without in-context learning. Initially, GPT-4V incorrectly classifies a test image as ‘Pneumonia’ with a confidence score of 0.85, when no contextual information is provided. However, when conditioned with two in-context examples from the source domain, one labeled ’Pneumonia’ and the other ‘Normal,’ the model’s performance shifts markedly. With in-context learning, the model compares the third image with the first ’Pneumonia’ figure and accurately categorizes the same test image as ’Normal’ with an identical confidence score. This stark difference underscores the significant role that in-context learning plays in enhancing the model’s diagnostic precision, particularly in discerning subtle distinctions in medical imaging.

###### 6 Conclusion

Our investigation into the adaptability and generalization capabilities of GPT-4V, a leading multimodal foundation model, marks a significant advancement in our understanding of AI systems’ robustness against distribution shifts. Through rigorous evaluation and comparison with models like CLIP, LLaVA, and Gemini across 13 diverse datasets in natural, medical, and molecular domains, we have delineated the capability boundaries of GPT-4V, uncovering both its strengths and limitations in various complex scenarios. Our findings reveal that while GPT-4V demonstrates notable adaptability and zero-shot generalization capabilities, its performance varies significantly across different scenarios of distribution shifts. This variation underscores the importance of continuous assessment and enhancement of foundation models to cope with evolving data landscapes. While we have made significant strides in understanding and improving the adaptability of foundation models like GPT-4V, our journey toward creating truly robust and versatile AI foundation models is ongoing.

###### In-context Case

###### Prompt:

Given the image, answer the following question using the specified format. \\ The first image is normal and the second image is tumor. Question: What is the third image? \\ Choices:[‘normal', ‘tumor']. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

[Figure 14] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile14.png>)

[Figure 15] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile15.png>)

[Figure 16] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile16.png>)

|Answer Choice: normal Confidence Score: 0.85 Reasoning:The third image presents tissue with cellular structures that appear to be regular and uniform, similar to the first image labeled as normal. There is no evident abnormal morphology typical of tumor tissue, such as irregular cell sizes, shapes, or disorganized arrangement, which is commonly observed in the second image labeled as tumor.<br><br>GPT-4V:|
|---|

- Figure 5 | Demonstration of GPT-4V’s inference process when exposed to in-context learning with examples from the Camelyon17 dataset. The experiment involves using two representative images from the source domain (hospital_2), one labeled ‘normal’ and the other ‘tumor’, followed by a test image from the target domain (hospital_3). GPT-4V, conditioned with these in-context examples, distinguishes between regular and uniform tissue patterns in the ‘normal’ image and abnormal, irregular cell sizes in the ‘tumor’ image. It then applies this contextual understanding to accurately infer the class of the test image from hospital_3. This process showcases GPT-4V’s ability to leverage in-context cues for effective domain bridging.

###### In-context Case

###### Prompt:

Given the image, answer the following question using the specified format. \\ The first image is Pneumonia and the second image is Normal. Question: What is the third image? \\ Choices:[‘Pneumonia’, ‘Normal']. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

[Figure 17] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile17.png>)

[Figure 18] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile18.png>)

[Figure 19] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile19.png>)

|Answer Choice: Pneumonia Confidence Score: 0.85 Reasoning: The X-ray image shows increased opacity in the lower lung fields which could indicate an infection or fluid, suggestive of pneumonia. The lung markings do not appear normal and there are areas that seem to be more opaque and consolidated, common indicators of pneumonia in a chest radiograph. However, a definitive diagnosis would require medical expertise and possibly additional information and testing.<br><br>GPT-4V w/o in-context:|
|---|

|Answer Choice: Normal Confidence Score: 0.85 Reasoning: The lung fields appear clear with no obvious consolidations or opacities that would suggest pneumonia. The heart size is normal, and there are no signs of pleural effusion. Compared to the first image which was identified as Pneumonia, this image lacks the characteristic signs such as lung consolidation and patchy opacities.<br><br>GPT-4V w/ in-context:|
|---|

- Figure 6 | This figure illustrates GPT-4V’s inference process under in-context learning, using selected examples from the COVID dataset. The process involves initially presenting two annotated images from the source domain: one labeled as ’Pneumonia’ and the other as ’Normal.’ Subsequently, GPT-4V is tasked with classifying a test image from the target domain. The model, having been conditioned with these specific in-context examples, evaluates and categorizes the test image by drawing comparisons to the previously presented pneumonia and normal X-ray images.

Comparative Domain Accuracies across Three Models on PACS

1.10

clip-vit-base-patch16 llava-v1.5-13b gpt-4-vision-preview

1.05

| |
|---|

| |
|---|

1.00

0.95

0.90

0.85

0.80

0.75

0.70

photo cartoon sketch art_painting

Domain

###### (a) Comparative Domain Accuracies across Three Models on PACS

Comparative Domain Accuracies across Three Models on VLCS

clip-vit-base-patch16

1.0

llava-v1.5-13b

gpt-4-vision-preview

0.9

0.8

0.7

0.6

0.5

VOC2007 Caltech101 LabelMe SUN09

Domain

###### (b) Comparative Domain Accuracies across Three Models on VLCS

Comparative Domain Accuracies across Three Models on OfficeHome

1.1

clip-vit-base-patch16

llava-v1.5-13b

1.0

gpt-4-vision-preview

0.9

0.8

0.7

0.6

0.5

Art RealWorld Clipart Product

Domain

(c) Comparative Domain Accuracies across Three Models on OfficeHome

- Figure 7 | Comparative accuracies of three models across domains in the PACS, VLCS, Office-Home datasets.

Comparative Domain Accuracies across Three Models on DomainNet

clip-vit-base-patch16 llava-v1.5-13b gpt-4-vision-preview

1.0

| |
|---|

| |
|---|

0.8

0.6

0.4

0.2

0.0

real quickdraw infograph sketch painting clipart

Domain

###### (a) Comparative Domain Accuracies across Three Models on DomainNet

Comparative Domain Accuracies across Three Models on Fmow

0.40

clip-vit-base-patch16 llava-v1.5-13b gpt-4-vision-preview

0.35

| |
|---|

| |
|---|

0.30

0.25

0.20

0.15

0.10

0.05

0.00

region_2 region_1 region_4 region_0 region_3 region_5

Domain

###### (b) Comparative Domain Accuracies across Three Models on Fmow

Comparative Domain Accuracies across Three Models on Terra_incognita

0.7

clip-vit-base-patch16 llava-v1.5-13b gpt-4-vision-preview

| |
|---|

0.6

| |
|---|

0.5

0.4

0.3

0.2

0.1

0.0

location_46 location_38 location_100 location_43

Domain

(c) Comparative Domain Accuracies across Three Models on TerraIncognita

- Figure 8 | Comparative accuracies of three models across domains in the DomainNet, Fmow, TerraIncognita datasets.

### GPT-4V: Correct | LLaVA: Correct

###### Prompt:

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices: [dog, elephant, giraffe, guitar, horse, house, person]. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 20] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile20.png>)|
|---|

|Answer Choice: elephant Confidence Score: 1 Reasoning: The image contains a cartoon illustration of an elephant, which is standing on a platform with springs attached to its feet. The caption "Elephants can't jump" also indicates that the focus of the image is an elephant.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: elephant Confidence Score: 0.9 Reasoning: The image features an elephant with a trunk and tusks, standing on a dirt field.|
|---|

- Figure 9 | Natural Distribution Shift: Case 1 - Elephant category in the Cartoon Domain of PACS Dataset. In this instance, both GPT-4V and LLaVA are provided with the same text prompt alongside an image. Both models successfully generate an answer choice, a confidence score, and their reasoning. Notably, GPT-4V demonstrates a capability for detail recognition, accurately identifying the text ’Elephants can’t jump.’ in the image. This case exemplifies GPT-4V’s advanced ability to discern and interpret finer details within visual inputs, compared to LLaVA.

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 21] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile21.png>)|
|---|

|Answer Choice: giraffe Confidence Score: 0.95 Reasoning: The image appears to be a rough line drawing of a giraffe. It is<br><br>characterized by a long neck, ossicones (horn-like structures), and facial features that resemble a giraffe.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: horse Confidence Score: 0.8 Reasoning: The image is a drawing of a horse.|
|---|

- Figure 10 | Natural Distribution Shift: Case 2 - Giraffe category in the Sketch Domain of PACS Dataset. This example showcases the performance of GPT-4V and LLaVA when given the same text prompt with an image depicting a giraffe in a sketch style. GPT-4V successfully identifies the giraffe, providing detailed reasoning and demonstrating a nuanced understanding of the image’s content, such as long neck, horn-like structures. In contrast, LLaVA fails to correctly identify the giraffe, offering limited reasoning in its response.

### GPT-4V: Incorrect | LLaVA: Correct

###### Prompt:

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices: [dog, elephant, giraffe, guitar, horse, house, person]. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 22] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile22.png>)|
|---|

|Answer Choice: person Confidence Score: 0.95 Reasoning: The image displays a stylized depiction of a guitar, predominantly in the center, being played by a skeleton character. The shape, strings, and tuning pegs are characteristic of a guitar, leading to high confidence in this identification.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: guitar Confidence Score: None Reasoning: The image shows a skeleton playing a guitar, which is the main focus of the image.|
|---|

- Figure 11 | Natural Distribution Shift: Case 3 - Guitar category in the Art_painting Domain of PACS Dataset. While LLaVA accurately classifies the image, GPT-4V fails to identify the correct class. However, an interesting observation emerges in the rationale provided by GPT-4V. Despite the incorrect classification, GPT-4V articulates a highly reasoned and contextually relevant explanation, offering a detailed and accurate description of the ground_truth class label: guitar.

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 23] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile23.png>)|
|---|

###### Ground-truth Answer: horse

|Answer Choice: giraffe Confidence Score: 1 Reasoning: The image contains illustrations of giraffes interspersed with trees and other small decorative elements, identifiable by their long necks and patterned bodies which are distinctive characteristics of giraffes.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: dog Confidence Score: None Reasoning: The image is a pattern of dogs, and the dogs are in various poses and colors.|
|---|

- Figure 12 | Natural Distribution Shift: Case 4 - Horse category in the Cartoon Domain of PACS Dataset. In this example, both GPT-4V and LLaVA incorrectly identify the subject in the image. The cartoon domain often features abstract styles where certain aspects of objects are exaggerated, as seen in the elongated necks of the horses in the image. GPT-4V incorrectly classifies the subject as a giraffe, likely influenced by the exaggerated neck feature. Compared to LLaVA, which provides limited reasoning, GPT-4V’s rationale, though leading to an incorrect conclusion, is more detailed, noting the distinctive long neck as a key characteristic for its prediction.

### GPT-4V: Correct | LLaVA: Correct

###### Prompt:

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices: ['bird', 'car', 'chair', 'dog', 'person']. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 24] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile24.png>)|
|---|

|Answer Choice: chair Confidence Score: 1 Reasoning: There are multiple chairs visible in front of the cafe in the image, arranged around tables. No birds, cars, dogs, or persons are clearly identifiable in the image.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: chair Confidence Score: 0.8 Reasoning: The image shows a cafe with several chairs outside.|
|---|

- Figure 13 | Natural Distribution Shift: Case 5 - Chair category in the LabelMe Domain of VLCS Dataset. This case illustrates the proficiency of both GPT-4V and LLaVA models in accurately identifying multiple chairs within the scene. GPT-4V, in particular, stands out for its detailed and comprehensive description, offering nuanced insights that surpass the more straightforward analysis provided by LLaVA.

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 25] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile25.png>)|
|---|

|Answer Choice: chair Confidence Score: 1.0 Reasoning: There is a striped sofa in the image, which falls under the category of a chair as it is a piece of furniture designed for sitting. No birds, cars, dogs, or people are visible in the image.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: None Confidence Score: 0.9 Reasoning: The couch is the most prominent object in the image, taking up a significant portion of the living room.|
|---|

- Figure 14 | Natural Distribution Shift: Case 6 - Chair category in the LabelMe Domain of VLCS Dataset. In this scenario, both GPT-4V and LLaVA models are presented with an image of a sofa/couch. GPT-4V demonstrates adaptability by categorizing the sofa as a type of chair, aligning with the limitations of the provided answer choices, and thus delivering an accurate classification. In contrast, LLaVA struggles to make the correct inference within the given constraints, highlighting a notable difference in their interpretative flexibility.

### GPT-4V: Incorrect | LLaVA: Correct

###### Prompt:

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices: ['bird', 'car', 'chair', 'dog', 'person']. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 26] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile26.png>)|
|---|

|Answer Choice: bird Confidence Score: 0.95 Reasoning: There are multiple people visible walking on the sidewalk in the image.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: person Confidence Score: None Reasoning: There are several people walking on the sidewalk and standing on the street corner, and some of them are carrying handbags.|
|---|

- Figure 15 | Natural Distribution Shift: Case 7 - Person category in the LabelMe Domain of VLCS Dataset. In this instance, despite GPT-4V providing a logically sound reasoning process, it paradoxically arrives at an incorrect conclusion. This case highlights an intriguing aspect of GPT-4V’s performance, where accurate analysis and reasoning do not always lead to the correct classification.

Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 27] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile27.png>)|
|---|

###### Ground-truth Answer: chair

|Answer Choice: person Confidence Score: 1 Reasoning: There is an individual seated at a table, using a device which appears to be a phone or remote, and this aligns with the 'person' answer choice.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: person Confidence Score: None Reasoning: The image shows a man sitting in a chair, eating food and drinking wine. There are also wine glasses, a bottle, and a box of chocolates on the table.|
|---|

- Figure 16 | Natural Distribution Shift: Case 8 - Chair category in the VOC2007 Domain of VLCS Dataset. This scenario illustrates the challenge faced by models like GPT-4V and LLaVA in accurately classifying images with multiple objects. Despite providing rational explanations, these models struggle to pinpoint the correct class when presented with complex scenes containing various elements.

## GPT-4V: Correct | LLaVA: Correct

##### Prompt:

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices: [‘Bottle’, ‘Exit_Sign’, ‘Lamp_Shade’, ‘Postit_Notes’, …, ‘Speaker’]. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 28] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile28.png>)|
|---|

|LLaVA:<br><br>Answer Choice: Kettle Confidence Score: 0.9 Reasoning: The image shows a silver kettle with a black handle and a thermometer on top.|
|---|

|Answer Choice: Kettle Confidence Score: 1 Reasoning: The image shows a metal kettle with a spout, handle, and a thermometer on the top, which is a common design for a kettle used to heat water.<br><br>GPT-4V:|
|---|

- Figure 17 | Natural Distribution Shift: Case 9 - Kettle category in the Product Domain of Office-Home Dataset. In this case study, both GPT-4V and LLaVA models are tasked with responding to an identical text prompt accompanied by an image. It is noteworthy that GPT-4V demonstrates a more nuanced understanding, particularly in its ability to detail specific features such as the kettle’s metallic nature and the presence of a spout. Additionally, GPT-4V enhances its answer with a summary that emphasizes typical design characteristics, thereby lending greater confidence to its response.

## GPT-4V: Correct | LLaVA: Incorrect

##### Prompt:

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices: [‘Bottle’, ‘Exit_Sign’, ‘Lamp_Shade’, ‘Postit_Notes’, …, ‘Speaker’]. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 29] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile29.png>)|
|---|

|LLaVA:<br><br>Answer Choice: Marker Confidence Score: 0.8 Reasoning: The image shows a person holding a pink object, which is a marker, and using it to write on a piece of paper.|
|---|

|Answer Choice: Eraser Confidence Score: 1 Reasoning: The image displays a hand holding a large pink eraser with the words "BIG MISTAKE" on it, and there are remnants of erased material, likely from a pencil, below it on the surface.\n<br><br>GPT-4V:|
|---|

- Figure 18 | Natural Distribution Shift: Case 10 - Analyzing the ’Eraser’ Category in the Art Domain of the Office-Home Dataset. This figure presents an intriguing instance where the depicted ’Eraser’ might be initially mistaken for a ’Marker’, a common perceptual challenge. GPT-4V remarkably identifies the correct object, utilizing cues from the text in the image, as well as the object’s size and color. Notably, GPT-4V correctly interprets the action of erasing, in contrast to LLaVA, which interprets the action as writing. This demonstrates GPT-4V’s advanced reasoning capabilities in distinguishing subtle contextual differences.

### GPT-4V: Incorrect | LLaVA: Correct

###### Prompt:

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices: [‘Bottle’, ‘Exit_Sign’, ‘Lamp_Shade’, ‘Postit_Notes’, …, ‘Speaker’]. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 30] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile30.png>)|
|---|

|Answer Choice: Table Confidence Score: 1 Reasoning: The image clearly shows a wooden chair in the foreground, which is among the choices listed. There is no ambiguity in the image regarding the object\'s identity as a chair.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: Chair Confidence Score: 0.9 Reasoning: The image shows a wooden chair with a leather seat and back.|
|---|

- Figure 19 | Natural Distribution Shift: Case 11 - Chair category in the Real World Domain of Office-Home Dataset. In this example, GPT-4V exhibits details and accuracy in its description of the image. Despite this, the model ultimately arrives at an incorrect classification.

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 31] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile31.png>)|
|---|

###### Ground-truth Answer: Couch

|Answer Choice: Chair Confidence Score: 1 Reasoning: The image displays a large chair with red upholstery and a unique design featuring skeletal and demonic features. It matches the description of a chair by providing seating for individuals, although the style is highly thematic and ornate.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: Chair Confidence Score: 0.9 Reasoning: The image shows a red chair with a black metal frame and a red velvet seat.|
|---|

- Figure 20 | Natural Distribution Shift: Case 12 - Couch category in the Art Domain of Office-Home Dataset. In this instance, both GPT-4V and LLaVA demonstrate detailed and accurate descriptions of the image, yet both models misclassify the object. This misclassification arises from the overlapping categories of ’couch’ and ’chair’ in the dataset, showcasing the challenge models face when distinct class labels share similarities. This case highlights the complexity models encounter in accurately categorizing objects within overlapping or closely related classes.

## GPT-4V: Correct | LLaVA: Incorrect

##### Prompt:

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices:['teddy-bear', 'strawberry', 'spoon', 'skull', 'school_bus', 'rain', 'pizza', 'parrot', 'ocean', 'line', 'monkey', …, 'bridge', 'camera']. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 32] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile32.png>)|
|---|

|LLaVA:<br><br>Answer Choice: hand Confidence Score: 0.9 Reasoning: The image is a white hand.|
|---|

|Answer Choice: finger Confidence Score: 1 Reasoning: The image clearly shows an illustration of a human hand with five fingers extended, which corresponds with the 'finger' choice from the provided list.<br><br>GPT-4V:|
|---|

- Figure 21 | Natural Distribution Shift: Case 13 - Analyzing the ’Finger’ Category in the Clipart Domain of the DomainNet Dataset. This figure showcases a challenging example where differentiation between ’Hand’ and ’Finger’ is crucial. GPT-4V accurately identifies the object as ’Finger’, recognizing the illustration of a human hand with all five fingers extended, which aligns with the ’Finger’ choice. In contrast, LLaVA identifies the object as ’Hand’, providing a less precise classification and a somewhat unusual description of a ’white hand’.

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 33] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile33.png>)|
|---|

###### Ground-truth Answer: cup

|Answer Choice: coffee_cup Confidence Score: 1 Reasoning: The image shows a mug typically used for coffee, with the phrase "oh what a Beautiful morning" and decorative floral patterns on its surface. This is consistent with the provided choice of "coffee_cup."<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: mug Confidence Score: 0.9 Reasoning: The image features a white coffee mug with a floral design and the words "Oh What a Beautiful Morning" written on it.|
|---|

- Figure 22 | Natural Distribution Shift: Case 14 - Analyzing the ’Cup’ Category in the Painting Domain of the DomainNet Dataset. Despite both GPT-4V and LLaVA present accurate OCR capability, neither GPT-4V nor LLaVA successfully distinguishes the correct category among these closely related concepts, leading to incorrect classifications. This scenario underscores the complexity inherent in nuanced visual recognition tasks, particularly when dealing with objects that share similar characteristics and uses.

## GPT-4V: Correct | LLaVA: Correct

##### Prompt:

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices: ['airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'aquaculture', …, 'wind_farm', 'zoo']. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 34] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile34.png>)|
|---|

|Answer Choice: airport Confidence Score: 0.9 Reasoning: The image shows a long, clear area that appears to be a dirt or unpaved runway, which is characteristic of a small airport or airfield. Structures that could be associated with aircraft hangars or terminals are adjacent to this runway. The layout is consistent with that of a small, possibly regional, airport.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: airport Confidence Score: 0.8 Reasoning: The image shows a large area of land with a road and buildings, and there is a clearing in the middle of the area. This clearing is likely an airport, as it is surrounded by buildings and infrastructure. The presence of a road and buildings also supports the idea that it is an airport, as these are common features of airport locations.|
|---|

- Figure 23 | Natural Distribution Shift: Case 15 - Evaluating the ’Airport’ Category in the Region_2 Domain of the Fmow Dataset. In this case study, both GPT-4V and LLaVA models are tasked with responding to an identical text prompt accompanied by an image. Each model correctly identifies ’airport’ as the subject, with GPT-4V exhibiting a higher confidence score of 0.9, as opposed to LLaVA’s 0.8. GPT-4V stands out for its detailed analysis, identifying key elements like runways, aircraft, and terminals, indicative of an airport. Remarkably, GPT-4V further distinguishes the airport as small or regional, showcasing its advanced reasoning and contextual interpretation abilities.

### GPT-4V: Correct | LLaVA: Incorrect

###### Prompt:

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices: ['airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'aquaculture', …, 'wind_farm', 'zoo']. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 35] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile35.png>)|
|---|

|Answer Choice: port Confidence Score: 0.9 Reasoning: The image shows large<br><br>docking areas with ships, as well as infrastructure such as cranes and storage areas typically associated with a port. The proximity to water with various piers extending into it is characteristic of ports, where cargo ships dock to load and unload goods.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: airport Confidence Score: 0.9 Reasoning: The image shows a large area with many buildings, including a runway, which is a characteristic of an airport.|
|---|

- Figure 24 | Natural Distribution Shift: Case 16 - Evaluating the ’Port’ Category in the Region_0 Domain of the Fmow Dataset. In this instance, GPT-4V accurately identifies the location as a port, citing the presence of docking areas, ships, cranes as key indicators. Its reasoning is thorough, focusing on specific port-related characteristics. Conversely, LLaVA incorrectly classifies the same image as an airport, referencing runways and buildings, but lacks the detailed analysis of maritime infrastructure present in GPT-4V’s description.

### GPT-4V: Incorrect | LLaVA: Correct

###### Prompt:

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices: ['airport', 'airport_hangar', 'airport_terminal', 'amusement_park', 'aquaculture', …, 'wind_farm', 'zoo']. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 36] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile36.png>)|
|---|

|Answer Choice: water_treatment_facility Confidence Score: 0.85 Reasoning: The structures in the image resemble large tanks or basins that are often characteristic of water treatment facilities, where water is processed in large containment areas for purification or sewage treatment purposes. There are clear geometric structures that are typical for such facilities.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: dam Confidence Score: 0.9 Reasoning: The image shows a large body of water with a dam, which is a structure built across a waterway to control the flow of water.|
|---|

- Figure 25 | Natural Distribution Shift: Case 17 - Evaluating the ’Dam’ Category in the Region_3 Domain of the Fmow Dataset. In this image, GPT-4V incorrectly identifies the scene as a water treatment facility, citing the presence of large tanks or basins typically found in such settings. Conversely, LLaVA correctly classifies the image as a dam, accurately recognizing the large body of water and the structure controlling its flow, with a confidence score of 0.9.

## GPT-4V: Correct | LLaVA: Correct

##### Prompt:

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices: ['bird', 'bobcat', 'cat', 'coyote', 'dog', 'empty', 'opossum', 'rabbit', 'raccoon‘, 'squirrel']. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 37] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile37.png>)|
|---|

|Answer Choice: bobcat Confidence Score: 0.95 Reasoning: The animal in the image has physical features consistent with a bobcat, such as a robust body, short tail, and tufted ears. The size, pattern, and color of the coat match typical bobcat characteristics.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: bobcat Confidence Score: 0.8 Reasoning: The image shows a small animal walking on a dirt road, and it is described as a bobcat.|
|---|

- Figure 26 | Natural Distribution Shift: Case 18 - Analyzing the ’Bobcat’ Category in Location_46 Domain of the TerraIncognita Dataset. In this evaluation, GPT-4V and LLaVA models respond to a uniform text prompt accompanied by a wildlife image. Both models accurately identify a ’bobcat’ as the subject. GPT-4V shows a higher confidence score of 0.95, compared to 0.8 by LLaVA. GPT-4V’s reasoning is notable for its detailed analysis, focusing on distinctive physical features of the bobcat, such as a robust body, short tail, and tufted ears, which are challenging to discern even for humans. It also augments its response by detailing the size, pattern, and color of the bobcat, which are crucial for accurate identification. In contrast, LLaVA’s identification is based on the general observation of a small animal, a criterion that could apply to multiple species, thereby lacking the specificity demonstrated by GPT-4V.

## GPT-4V: Correct | LLaVA: Incorrect

##### Prompt:

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices: ['bird', 'bobcat', 'cat', 'coyote', 'dog', 'empty', 'opossum', 'rabbit', 'raccoon‘, 'squirrel']. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 38] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile38.png>)|
|---|

|Answer Choice: coyote Confidence Score: 0.75 Reasoning: The animal in the image has a slender, elongated snout and an upright, bushy tail, which are characteristic features of a coyote. The general body shape and size in the image are also consistent with that of a coyote. However, the image is monochromatic and grainy, which might obscure finer details needed for a more definitive identification.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: cat Confidence Score: 0.8 Reasoning: The animal in the image is a cat, as it is walking in the dark and appears to be a small house cat.|
|---|

- Figure 27 | Natural Distribution Shift: Case 19 - Analyzing the ’Coyote’ Category in Location_46 Domain of the TerraIncognita Dataset. In this image, GPT-4V accurately identifies the animal as a coyote, noting its slender, elongated snout and upright, bushy tail, and assigning a confidence score of 0.75. It carefully considers the monochromatic and grainy nature of the image that may obscure finer details. In contrast, LLaVA incorrectly classifies the animal as a cat with a confidence score of 0.8, failing to recognize the distinct features of a coyote.

## GPT-4V: Incorrect | LLaVA: Correct

##### Prompt:

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices: ['bird', 'bobcat', 'cat', 'coyote', 'dog', 'empty', 'opossum', 'rabbit', 'raccoon‘, 'squirrel']. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 39] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile39.png>)|
|---|

|Answer Choice: bobcat Confidence Score: 0.80 Reasoning: The animal in the image has a physical build and coat pattern that resemble those of a bobcat, with apparent tufted ears and a short tail. The environment and the solitary nature of the subject also align with the typical habitat and behavior of a bobcat.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: cat Confidence Score: 0.9 Reasoning: The image shows a cat walking on a dirt road.|
|---|

- Figure 28 | Natural Distribution Shift: Case 20 - Analyzing the ’Cat’ Category in Location_38 Domain of the TerraIncognita Dataset. In this case, LLaVA correctly identifies the animal as a ’cat’ with a higher confidence score of 0.9, whereas GPT-4V, with a confidence score of 0.8, mistakenly identifies the animal as a ’bobcat’. The detailed reasoning of GPT-4V highlights its misclassification: it points to features such as a perceived short tail and tufted ears, typically characteristic of a bobcat, leading to its incorrect conclusion. This case illustrates the nuances and challenges of wildlife species recognition, particularly in distinguishing between visually similar animals in natural environments. The confidence score outputted by GPT-4V, despite its misclassification in this instance, can serve as a valuable metric, offering insights into the model’s decision-making process and potentially guiding reliance on its conclusions.

## GPT-4V: Incorrect | LLaVA: Incorrect

##### Prompt:

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices: ['bird', 'bobcat', 'cat', 'coyote', 'dog', 'empty', 'opossum', 'rabbit', 'raccoon‘, 'squirrel']. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 40] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile40.png>)|
|---|

###### Ground-truth Answer: rabbit

|Answer Choice: bobcat Confidence Score: 0.95 Reasoning: The animal has the distinctive physical features of a bobcat, including tufted ears, a short tail, and a stocky build. The size and physical characteristics are consistent with those of a bobcat rather than the other listed animals.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: bobcat Confidence Score: None Reasoning: The image shows two animals walking together, and the presence of a tree in the background suggests a natural environment. The animals are walking in the dirt, which is typical for bobcats. The fact that there are two animals walking together also supports the idea that they are bobcats, as they are known to be social animals.|
|---|

- Figure 29 | Natural Distribution Shift: Case 21 - Analyzing the ’Rabbit’ Category in Location_43 Domain of the TerraIncognita Dataset. This image presents a complex wildlife scenario where a bobcat is seen capturing and biting a rabbit. Both GPT-4V and LLaVA misinterpret the scene by identifying only the bobcat. This case underlines the intricacies of wildlife recognition, particularly when multiple animals interact in a single frame. The primary focus on the bobcat, while ignoring the rabbit, points to the nuanced challenges in accurately interpreting dynamic natural scenes.

## GPT-4V: Correct | Gemini: Incorrect

##### Prompt:

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices: ['bird', 'car', 'chair', 'dog', 'person']. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 41] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile41.png>)|
|---|

###### Ground-truth Answer: car

|Answer Choice: car Confidence Score: 0.85 Reasoning: The image shows a street view with buildings, and there are vehicles on the road that resemble cars. While the image is not close-up on any specific car, the context suggests that what could be identified as cars are indeed present.<br><br>GPT-4V:|
|---|

|Gemini:<br><br>Answer Choice: person Confidence Score: 0.8 Reasoning: There are people walking on the sidewalk in the image.|
|---|

- Figure 30 | Natural Distribution Shift: Case 22 - Analyzing the ‘Car’ Category in the SUN09 Domain of the VLCS Dataset. This instance illustrates a challenging scenario where GPT-4V accurately identifies the presence of cars within a street view, noting vehicles on the road amidst buildings with a confidence score of 0.85. Conversely, Gemini, with a confidence score of 0.8, incorrectly identifies a person, focusing on individuals walking on the sidewalk. This comparison highlights the nuanced differences in model perception and interpretation within complex urban environments, emphasizing the critical role of context in AI’s visual comprehension.

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices: [‘actinic keratoses and intraepithelial carcinoma’, …, ‘melanoma’]. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 42] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile42.png>)|
|---|

|Answer Choice: basal cell carcinoma Confidence Score: 0.7 Reasoning: The lesion in the image appears pearly and translucent with what may be telangiectatic vessels, which are common characteristics of basal cell carcinoma. However, without further clinical context or a diagnostic biopsy, it's not possible to be certain.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: basal cell carcinoma Confidence Score: 0.75 Reasoning: The image shows a skin lesion that is red and brown in color, which is consistent with the appearance of basal cell carcinoma. Additionally, the lesion is described as being “ugly” and “ugly looking,” which further supports the possibility that it is a skin cancer.|
|---|

- Figure 31 | Medical Distribution Shift: Case 1 - Analyzing the ‘basal cell carcinoma’ Category in the vidir_modern Domain of the HAM10000 Dataset. In this case study, both GPT-4V and LLaVA models are tasked with responding to an identical text prompt accompanied by an image. Each model correctly identifies ‘basal cell carcinoma’ as the subject, with LLaVA exhibiting a higher confidence score of 0.75, as opposed to GPT-4V’s 0.7. GPT-4V stands out for its detailed analysis, identifying key elements like pearly, translucent, and telangiectatic vessels, indicative of basal cell carcinoma. LLaVA gives an analysis mainly in terms of color appearance.

Choices: [‘COVID19’, ‘Normal’, ‘Pneumonia’]. \\ Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 43] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile43.png>)|
|---|

|Answer Choice: Normal Confidence Score: 0.7 Reasoning: The image appears to be a standard chest X-ray without obvious signs of pneumonia, such as localized opacities, or signs typically associated with COVID-19, such as bilateral peripheral ground-glass opacities. However, without clinical context or additional diagnostic information, this interpretation may not be definitive.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: Normal Confidence Score: 0.95 Reasoning: The image shows a baby's ribcage, which is a normal part of a human body.|
|---|

- Figure 32 | Medical Distribution Shift: Case 2 - Analyzing the ‘Normal’ Category in the hostial1 Domain of the COVID Dataset. In this case study, both GPT-4V and LLaVA models are tasked with responding to an identical text prompt accompanied by an image. Each model correctly identifies ‘Normal’ as the subject, with LLaVA exhibiting a higher confidence score of 0.95, as opposed to GPT-4V’s 0.7. GPT-4V stands out for its detailed analysis, identifying key elements like localized opacities, and bilateral peripheral ground-glass opacities, indicative of a normal sample. LLaVA states that the image is of a baby’s rib cage, but does not give a proper reason for why it is categorized as normal, despite the high confidence score of 0.95.

#### GPT-4V: Inorrect | LLaVA: Correct

###### Prompt:

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices: [‘active’, ‘inactive’]. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 44] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile44.png>)|
|---|

|Answer Choice: Inactive Confidence Score: null Reasoning: The image shows a chemical structure, which cannot be categorized as 'active' or 'inactive' without additional context such as biological activity, enzyme interaction, or the presence in an active/inactive state within a metabolic pathway. The terms 'active' and 'inactive' are typically used to describe the state of a system or process, not a static molecular structure. Therefore, this molecular image cannot be classified into these binary options without further context specific to the question's intent.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: Active Confidence Score: 0.9 Reasoning: The image shows a chemical structure with a red and blue bond, which indicates that it is an active compound.|
|---|

- Figure 33 | Scientific Distribution Shift: Case 1 - Analyzing the ‘active’ Category in the ID-75 Domain of the DrugOOD_assay Dataset. In this case study, both GPT-4V and LLaVA models are tasked with responding to an identical text prompt accompanied by an image. GPT-4V incorrectly predicts without confidence score while LLaVA correctly predicts with high confidence. Although the GPT-4V predicts error categories, it does not give high confidence. According to reason, GPT-4V can recognize this image as a chemical structure. The statement that it cannot be categorized as ‘active’ or ‘inactive’ without other contexts such as biological activity, enzyme interactions, or active/inactive states in metabolic pathways makes sense. The limitations of the DrugOOD dataset itself are also noted here, i.e., it is not reasonable to simply categorize the data as ‘active’ or ‘inactive’. Conversely, LLaVA, despite giving correct predictions, is very unreliable in its reasoning. It is wrong to classify them as ‘active’ or ‘inactive’ by the color of the bond.

Choices: [‘dog’, ‘elephant’, …, ‘person’]. \\ Please respond with the following format: ---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 45] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile45.png>)|
|---|

|Answer Choice: house Confidence Score: 0.75 Reasoning: The image appears to depict buildings, one of which has a spire or steeple, resembling structural architecture typically associated with a house or a church. The image is lowresolution and abstract, making it difficult to ascertain complete details, but there are clearly no animals, people, or guitars present.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: house Confidence Score: None Reasoning: The image shows a house<br><br>with a clock tower and a boat in the water. The presence of the clock tower and the boat suggest that this is a house with a unique architectural feature, making it stand out from other houses. The house is also located<br><br>near a body of water, which is a common feature in many houses.|
|---|

- Figure 34 | Induced Distribution Shift: Case 1 - Analyzing the ‘Fork’ Category in the Product Domain of the OfficeHome_gaussion Dataset. In this case study, both GPT-4V and LLaVA models are tasked with responding to an identical text prompt accompanied by an image. Both GPT-4V and LLaVA predicted correctly, but GPT-4V gave higher confidence levels as well as more detailed explanations such as tines, a handle, and several pointed prongs. These are the basic characteristics of a fork. However, the reason given by LLaVA is rough.

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 46] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile46.png>)|
|---|

|Answer Choice: Fork Confidence Score: 1.00 Reasoning: The image clearly displays a utensil with tines which is used for eating, commonly known as a fork. It matches the typical appearance of a fork with a handle and several pointed prongs.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: Fork Confidence Score: 0.90 Reasoning: The image shows a fork on a blue tablecloth.|
|---|

- Figure 35 | Induced Distribution Shift: Case 2 - Analyzing the ‘house’ Category in the art_painting Domain of the PACS_gaussion Dataset. In this case study, both GPT-4V and LLaVA models are tasked with responding to an identical text prompt accompanied

- by an image. Both GPT-4V and LLaVA predicted correctly, however, GPT-4V gave a confidence level of 0.75 and LL a VA did not give a confidence level. GPT-4V gave some more detailed information in the reason, such as spire and steeple. On the contrary, LLaVA gave a partially incorrect description in the reason, such as boat and water.

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 47] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile47.png>)|
|---|

|Answer Choice: Calendar Confidence Score: 0.95 Reasoning: The image shows a grid that resembles a monthly calendar layout, with numbers that seem to represent dates at the top, which is a common feature of calendars.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: Calendar Confidence Score: 0.90 Reasoning: The image shows a calendar with a clock on it.|
|---|

- Figure 36 | Induced Distribution Shift: Case 3 - Analyzing the ‘dog’ Category in the art_painting Domain of the PACS_unseen Dataset. In this case study, both GPT-4V and LLaVA models are tasked with responding to an identical text prompt accompanied

- by an image. Both GPT-4V and LLaVA predicted correctly, however, GPT-4V gave a confidence level of 1.00 and LLaVA did not give a confidence level. Both GPT-4V and LLaVA give a more nuanced and reliable reason.

Choices: [‘dog’, ‘elephant’, …, ‘person’].\\ Please respond with the following format: ---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|[Figure 48] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile48.png>)|
|---|

|Answer Choice: dog Confidence Score: 1.00 Reasoning: The image clearly shows a young dog with distinct facial markings and fur that is typical of a canine.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: dog Confidence Score: None Reasoning: The image is of a dog with blue and white fur.|
|---|

- Figure 37 | Natural Distribution Shift: Case 4 - Analyzing the ‘Calendar’ Category in the Clipart Domain of the OfficeHome_unseen Dataset. In this case study, both GPT-4V and LLaVA models are tasked with responding to an identical text prompt accompanied by an image. Both GPT-4V and LLaVA predicted correctly, but GPT-4V gave higher confidence. GPT-4V and LLaVA focus on different details. GPT-4V focuses on the grid and dates associated with Calendar, whereas LLaVA focuses on another object ‘clock’ in the image.

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices: [‘Alarm_Clock’, ‘Backpack’, …, ‘Webcam’]. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|img: Art/Postit_Notes/00 016.jpg<br><br>[Figure 49] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile49.png>)|
|---|

|Answer Choice: Postit_Notes Confidence Score: 0.95 Reasoning: The image appears to show a collection of Post-it notes arranged in a creative way to resemble various objects like windows, doors, and a sign. Each Post-it note has writing or drawing on it, which is characteristic of how Post-it notes are used for reminders, notes, or labels. The overall arrangement is artistic, but the primary objects depicted are clearly Post-it notes.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: Postit_Notes Confidence Score: 0.8 Reasoning: The image is a collection of post it notes with different messages on them.|
|---|

- Figure 38 | Natural Distribution Shift: Case 5 - Analyzing the ‘Postit_Notes’ Category in the Art Domain of the OfficeHome_unseen Dataset. In this case study, both GPT-4V and LLaVA models are tasked with responding to an identical text prompt accompanied by an image. Both GPT-4V and LLaVA predicted correctly, but GPT-4V gave higher confidence with more detailed description. GPT-4V focuses on the writing and drawing associated with Postit_Notes, whereas LLaVA focuses on massages on it. This also demonstrates that GPT-4V and LLaVA have a certain degree of generalization ability on unseen data with domain shifts, with GPT-4V possessing stronger explanatory capabilities.

GPT-4V: Incorrect | LLaVA: Incorrect

##### Prompt:

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices: [‘Alarm_Clock’, ‘Backpack’, …, ‘Webcam’]. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|img: Art/Postit_Notes/000 16.jpg<br><br>[Figure 50] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile50.png>)|
|---|

Ground-truth Answer: TV

|Answer Choice: Printer Confidence Score: 0.95 Reasoning: The item in the top left section of the large shelving unit appears to be a printer due to its shape, the presence of paper trays, and its position atop what seems to be a stack of paper, which is a common accessory with printers. The overall configuration of the item, with what seems like control buttons and possibly a display, is characteristic of a printer's design.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: Postit_Notes Confidence Score: 0.8 Reasoning: The image shows a display of various items, including a display of post it notes.|
|---|

- Figure 39 | Natural Distribution Shift: Case 6 - Analyzing the ‘TV’ Category in the Real World Domain of the OfficeHome_unseen Dataset. This image presents a TV displaying complex content. Due to the misleading nature of the complex content, both GPT-4V and LLaVA made mistakes. GPT-4V misidentified the TV as a Printer, while LLaVA misidentified it as Post-it Notes. This result demonstrates that both GPT-4V and LLaVA still have difficulties in predicting complex samples accurately.

GPT-4V: Incorrect | LLaVA: Correct

##### Prompt:

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices: [‘Alarm_Clock’, ‘Backpack’, …, ‘Webcam’]. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|img: Art/Postit_Notes/0 0016.jpg<br><br>[Figure 51] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile51.png>)|
|---|

|Answer Choice: Push_Pin Confidence Score: 0.95 Reasoning: The image includes what appears to be two pens on the right-hand side, next to the push pins.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: Marker Confidence Score: 0.8 Reasoning: The image shows a variety of art supplies, including a palette of colors and a paintbrush. The<br><br>presence of a palette and paintbrush suggests that the image is related to art, and markers are commonly used<br><br>in artistic projects.|
|---|

- Figure 40 | Natural Distribution Shift: Case 7 - Analyzing the ‘Marker’ Category in the Art Domain of the Office-Home_unseen Dataset. In this case, LLaVA correctly identifies the object as a ‘Marker’ with a confidence score of 0.8, whereas GPT-4V, with a confidence score of 0.95, mistakenly identifies the object as a ‘Push_Pin’. Due to the presence of an object resembling a Push_Pin in the image, GPT-4V identified the image as a Push_Pin. Meanwhile, not only did LLaVA correctly predict, but it also provided a description related to its prediction: a palette of colors and a paintbrush.

GPT-4V: Correct | LLaVA: Incorrect

##### Prompt:

Given the image, answer the following question using the specified format. \\ Question: What is in this image? \\ Choices: [‘Alarm_Clock’, ‘Backpack’, …, ‘Webcam’]. \\

Please respond with the following format:

---BEGIN FORMAT TEMPLATE--Answer Choice: [Your Answer Choice Here] Confidence Score: [Your Numerical Prediction Confidence Score Here From 0 To 1] Reasoning: [Your Reasoning Behind This Answer Here]

---END FORMAT TEMPLATE--Do not deviate from the above format. Repeat the format template for the answer.

|img: Art/Postit_Notes/0 0016.jpg<br><br>[Figure 52] Adapt to Distribution Shifts A Preliminary Investigation_images/imageFile52.png>)|
|---|

|Answer Choice: Clipboards Confidence Score: 1.0 Reasoning: The image shows a blue clipboard with a metal clip at the top, which is used to hold papers securely.<br><br>GPT-4V:|
|---|

|LLaVA:<br><br>Answer Choice: Postit_Notes Confidence Score: 0.8 Reasoning: The image is a collage of various items, and one of the items is a post it note.|
|---|

- Figure 41 | Natural Distribution Shift: Case 8 - Analyzing the ‘Clipboards’ Category in the Clipart Domain of the OfficeHome_unseen Dataset. In this image, GPT-4V accurately identifies the object as Clipboards, noting a metal clip at the top of object, and assigning a confidence score of 1.0. GPT-4V successfully captured the key element ‘clip,’ which helped in identifying the object as Clipboards. In contrast, LLaVA incorrectly classifies the object as Postit_Notes with a confidence score of 0.8, failing to recognize the key element ‘clip’ of Clipboards.

###### References

Kartik Ahuja and David Lopez-Paz. A closer look at in-context learning under distribution shifts. arXiv preprint arXiv:2305.16704, 2023.

Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.

Muhammad Awais, Muzammal Naseer, Salman Khan, Rao Muhammad Anwer, Hisham Cholakkal, Mubarak Shah, Ming-Hsuan Yang, and Fahad Shahbaz Khan. Foundational models defining a new era in vision: A survey and outlook. arXiv preprint arXiv:2307.13721, 2023.

Peter Bandi, Oscar Geessink, Quirine Manson, Marcory Van Dijk, Maschenka Balkenhol, Meyke Hermsen, Babak Ehteshami Bejnordi, Byungjae Lee, Kyunghyun Paeng, Aoxiao Zhong, et al. From detection of individual metastases to classification of lymph node status at the patient level: the camelyon17 challenge. IEEE transactions on medical imaging, 38(2):550–560, 2018.

Andrei Barbu, David Mayo, Julian Alverio, William Luo, Christopher Wang, Dan Gutfreund, Josh Tenenbaum, and Boris Katz. Objectnet: A large-scale bias-controlled dataset for pushing the limits of object recognition models. Advances in neural information processing systems, 32, 2019.

Sara Beery, Grant Van Horn, and Pietro Perona. Recognition in terra incognita. In Proceedings of the European conference on computer vision (ECCV), pages 456–473, 2018.

Sara Beery, Arushi Agarwal, Elijah Cole, and Vighnesh Birodkar. The iwildcam 2021 competition dataset. arXiv preprint arXiv:2105.03494, 2021.

Gilles Blanchard, Aniket Anand Deshmukh, Ürun Dogan, Gyemin Lee, and Clayton Scott. Domain generalization by marginal transfer learning. The Journal of Machine Learning Research, 22(1):46–100, 2021.

Rishi Bommasani, Drew A Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, et al. On the opportunities and risks of foundation models. arXiv preprint arXiv:2108.07258, 2021.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Manh-Ha Bui, Toan Tran, Anh Tran, and Dinh Phung. Exploiting domain-specific features to enhance domain generalization. Advances in Neural Information Processing Systems, 34:21189–21201, 2021.

Yunkang Cao, Xiaohao Xu, Chen Sun, Xiaonan Huang, and Weiming Shen. Towards generic anomaly detection and understanding: Large-scale visual-linguistic model (gpt-4v) takes the lead. arXiv preprint arXiv:2311.02782, 2023.

Anna Cascarano, Jordi Mur-Petit, Jerónimo Hernández-González, Marina Camacho, Nina de Toro Eadie, Polyxeni Gkontra, Marc Chadeau-Hyam, Jordi Vitrià, and Karim Lekadir. Machine and deep learning for longitudinal biomedical data: a review of methods and applications. Artificial Intelligence Review, pages 1–61, 2023.

Junbum Cha, Sanghyuk Chun, Kyungjae Lee, Han-Cheol Cho, Seunghyun Park, Yunsung Lee, and Sungrae Park. Swad: Domain generalization by seeking flat minima. Advances in Neural Information Processing Systems, 34:22405–22418, 2021.

Junbum Cha, Kyungjae Lee, Sungrae Park, and Sanghyuk Chun. Domain generalization by mutual-information regularization with pre-trained models. In Computer Vision–ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23–27, 2022, Proceedings, Part XXIII, pages 440–457. Springer, 2022.

Rune Christiansen, Niklas Pfister, Martin Emil Jakobsen, Nicola Gnecco, and Jonas Peters. A causal framework for distribution generalization. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10): 6614–6630, 2021.

Gordon Christie, Neil Fendley, James Wilson, and Ryan Mukherjee. Functional map of the world. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6172–6180, 2018.

Simon Shaolei Du, Wei Hu, Sham M. Kakade, Jason D. Lee, and Qi Lei. Few-shot learning via learning the representation, provably. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=pW2Q2xLwIMD.

Christopher J Ellis. Predicting the biodiversity response to climate change: challenges and advances. Systematics and biodiversity, 9(4):307–317, 2011.

Chen Fang, Ye Xu, and Daniel N Rockmore. Unbiased metric learning: On the utilization of multiple datasets and web images for softening bias. In Proceedings of the IEEE International Conference on Computer Vision, pages 1657–1664, 2013.

Yaroslav Ganin and Victor Lempitsky. Unsupervised domain adaptation by backpropagation. In International conference on machine learning, pages 1180–1189. PMLR, 2015.

Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, and Victor Lempitsky. Domain-adversarial training of neural networks. The journal of machine learning research, 17(1):2096–2030, 2016.

Ishaan Gulrajani and David Lopez-Paz. In search of lost domain generalization. In International Conference on Learning Representations, 2020.

Sharut Gupta, Stefanie Jegelka, David Lopez-Paz, and Kartik Ahuja. Context is environment. arXiv preprint arXiv:2309.09888, 2023.

Zhongyi Han, Rundong He, Tianyang Li, Benzheng Wei, Jian Wang, and Yilong Yin. Semi-supervised screening of covid-19 from positive and unlabeled data with constraint non-negative risk estimator. In Information Processing in Medical Imaging: 27th International Conference, IPMI 2021, Virtual Event, June 28–June 30, 2021, Proceedings 27, pages 611–623. Springer, 2021.

Zhongyi Han, Xian-Jin Gui, Haoliang Sun, Yilong Yin, and Shuo Li. Towards accurate and robust domain adaptation under multiple noisy environments. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(5):6460–6479, 2022a.

Zhongyi Han, Haoliang Sun, and Yilong Yin. Learning transferable parameters for unsupervised domain adaptation. IEEE Transactions on Image Processing, 31:6424–6439, 2022b.

Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of outof-distribution generalization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8340–8349, 2021a.

Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15262– 15271, 2021b.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021a.

Ruifeng Hu, Haodong Xu, Peilin Jia, and Zhongming Zhao. Kinasemd: kinase mutations and drug response database. Nucleic Acids Research, 49(D1):D552–D561, 2021b.

Jiaxing Huang, Dayan Guan, Aoran Xiao, and Shijian Lu. Fsdr: Frequency space domain randomization for domain generalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6891–6902, 2021a.

Kexin Huang, Cao Xiao, Lucas M Glass, Cathy W Critchlow, Greg Gibson, and Jimeng Sun. Machine learning applications for therapeutic tasks with genomics data. Patterns, 2(10), 2021b.

Zhuo Huang, Chang Liu, Yinpeng Dong, Hang Su, Shibao Zheng, and Tongliang Liu. Machine vision therapy: Multimodal large language models can enhance visual robustness via denoising in-context learning. arXiv preprint arXiv:2312.02546, 2023.

Yuanfeng Ji, Lu Zhang, Jiaxiang Wu, Bingzhe Wu, Lanqing Li, Long-Kai Huang, Tingyang Xu, Yu Rong, Jie Ren, Ding Xue, et al. Drugood: Out-of-distribution dataset curator and benchmark for ai-aided drug discovery–a focus on affinity prediction problems with noise annotations. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pages 8023–8031, 2023.

Daehee Kim, Youngjun Yoo, Seunghyun Park, Jinkyu Kim, and Jaekoo Lee. Selfreg: Self-supervised contrastive regularization for domain generalization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9619–9628, 2021.

Reto Knutti, Reinhard Furrer, Claudia Tebaldi, Jan Cermak, and Gerald A Meehl. Challenges in combining projections from multiple climate models. Journal of Climate, 23(10):2739–2758, 2010.

Pang Wei Koh, Shiori Sagawa, Henrik Marklund, Sang Michael Xie, Marvin Zhang, Akshay Balsubramani, Weihua Hu, Michihiro Yasunaga, Richard Lanas Phillips, Irena Gao, et al. Wilds: A benchmark of in-the-wild distribution shifts. In International Conference on Machine Learning, pages 5637–5664. PMLR, 2021.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022.

Ananya Kumar, Aditi Raghunathan, Robbie Matthew Jones, Tengyu Ma, and Percy Liang. Fine-tuning can distort pretrained features and underperform out-of-distribution. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=UYneFzXSJWh.

Yoonho Lee, Annie S Chen, Fahim Tajwar, Ananya Kumar, Huaxiu Yao, Percy Liang, and Chelsea Finn. Surgical fine-tuning improves adaptation to distribution shifts. In The Eleventh International Conference on Learning Representations, 2022.

Bo Li, Yifei Shen, Jingkang Yang, Yezhen Wang, Jiawei Ren, Tong Che, Jun Zhang, and Ziwei Liu. Sparse mixture-of-experts are domain generalizable learners. In The Eleventh International Conference on Learning Representations, 2022a.

Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, and Jianfeng Gao. Multimodal foundation models: From specialists to general-purpose assistants. arXiv preprint arXiv:2309.10020, 1, 2023a.

Da Li, Yongxin Yang, Yi-Zhe Song, and Timothy M Hospedales. Deeper, broader and artier domain generalization. In Proceedings of the IEEE international conference on computer vision, pages 5542–5550, 2017.

Da Li, Yongxin Yang, Yi-Zhe Song, and Timothy Hospedales. Learning to generalize: Meta-learning for

domain generalization. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018a. Haoliang Li, Sinno Jialin Pan, Shiqi Wang, and Alex C Kot. Domain generalization with adversarial feature

learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5400–5409,

- 2018b.

Haoliang Li, Sinno Jialin Pan, Shiqi Wang, and Alex C Kot. Domain generalization with adversarial feature learning. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5400–5409,

- 2018c.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 12888–12900. PMLR, 2022b.

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023b.

Ya Li, Xinmei Tian, Mingming Gong, Yajing Liu, Tongliang Liu, Kun Zhang, and Dacheng Tao. Deep domain generalization via conditional invariant adversarial networks. In ECCV, pages 624–639, 2018d.

Ziyue Li, Kan Ren, Xinyang Jiang, Yifei Shen, Haipeng Zhang, and Dongsheng Li. Simple: Specialized model-sample matching for domain generalization. In The Eleventh International Conference on Learning Representations, 2022c.

Kevin Lin, Faisal Ahmed, Linjie Li, Chung-Ching Lin, Ehsan Azarnasab, Zhengyuan Yang, Jianfeng Wang, Lin Liang, Zicheng Liu, Yumao Lu, et al. Mm-vid: Advancing video understanding with gpt-4v (ision). arXiv preprint arXiv:2310.19773, 2023.

Alexander H Liu, Yen-Cheng Liu, Yu-Ying Yeh, and Yu-Chiang Frank Wang. A unified feature disentangler for multi-domain image translation and manipulation. Advances in neural information processing systems, 31, 2018.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023a.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023b.

Jiashuo Liu, Zheyuan Hu, Peng Cui, Bo Li, and Zheyan Shen. Heterogeneous risk minimization. In ICML, pages 6804–6814. PMLR, 2021.

Akib Mashrur, Wei Luo, Nayyar A Zaidi, and Antonio Robles-Kelly. Machine learning for financial risk management: a survey. IEEE Access, 8:203203–203223, 2020.

Krikamol Muandet, David Balduzzi, and Bernhard Schölkopf. Domain generalization via invariant feature representation. In ICML, pages 10–18. PMLR, 2013.

Hyeonseob Nam, HyunJae Lee, Jongchan Park, Wonjun Yoon, and Donggeun Yoo. Reducing domain gap by reducing style bias. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8690–8699, 2021.

OpenAI. Gpt-4v(ision) system card. 2023. URL https://cdn.openai.com/papers/GPTV_System_Card.pdf. Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang,

Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744, 2022.

Chunjong Park, Anas Awadalla, Tadayoshi Kohno, and Shwetak Patel. Reliable and trustworthy machine learning for health using dataset shift detection. Advances in Neural Information Processing Systems, 34: 3043–3056, 2021.

Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. Moment matching for multi-source domain adaptation. In ICCV, 2019a.

Xingchao Peng, Qinxun Bai, Xide Xia, Zijun Huang, Kate Saenko, and Bo Wang. Moment matching for multi-source domain adaptation. In Proceedings of the IEEE/CVF international conference on computer vision, pages 1406–1415, 2019b.

Aayush Prakash, Shaad Boochoon, Mark Brophy, David Acuna, Eric Cameracci, Gavriel State, Omer Shapira, and Stan Birchfield. Structured domain randomization: Bridging the reality gap by context-aware synthetic data. In 2019 International Conference on Robotics and Automation (ICRA), pages 7249–7255. IEEE, 2019.

Fengchun Qiao, Long Zhao, and Xi Peng. Learning to learn single domain generalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12556–12565, 2020.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

Alexandre Rame, Matthieu Kirchmeyer, Thibaud Rahier, Alain Rakotomamonjy, Patrick Gallinari, and Matthieu Cord. Diverse weight averaging for out-of-distribution generalization. arXiv preprint arXiv:2205.09739, 2022.

Alexandre Ramé, Kartik Ahuja, Jianyu Zhang, Matthieu Cord, Léon Bottou, and David Lopez-Paz. Model ratatouille: Recycling diverse models for out-of-distribution generalization, 2023.

Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do imagenet classifiers generalize to imagenet? In International conference on machine learning, pages 5389–5400. PMLR, 2019.

Mateo Rojas-Carulla, Bernhard Schölkopf, Richard Turner, and Jonas Peters. Invariant models for causal transfer learning. The Journal of Machine Learning Research, 19(1):1309–1342, 2018.

Shiori Sagawa, Pang Wei Koh, Tatsunori B Hashimoto, and Percy Liang. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. arXiv preprint arXiv:1911.08731, 2019.

Bernhard Schölkopf, Francesco Locatello, Stefan Bauer, Nan Rosemary Ke, Nal Kalchbrenner, Anirudh Goyal, and Yoshua Bengio. Toward causal representation learning. Proceedings of the IEEE, 109(5):612–634, 2021.

Vaishaal Shankar, Achal Dave, Rebecca Roelofs, Deva Ramanan, Benjamin Recht, and Ludwig Schmidt. Do image classifiers generalize across time? In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9661–9669, 2021.

Yongxin Shi, Dezhi Peng, Wenhui Liao, Zening Lin, Xinhong Chen, Chongyu Liu, Yuyi Zhang, and Lianwen Jin. Exploring ocr capabilities of gpt-4v (ision): A quantitative and in-depth evaluation. arXiv preprint arXiv:2310.16809, 2023.

Karin Stacke, Gabriel Eilertsen, Jonas Unger, and Claes Lundström. Measuring domain shift for deep learning in histopathology. IEEE journal of biomedical and health informatics, 25(2):325–336, 2020.

Andrea Stocco and Paolo Tonella. Confidence-driven weighted retraining for predicting safety-critical failures in autonomous driving systems. Journal of Software: Evolution and Process, 34(10):e2386, 2022.

Baochen Sun and Kate Saenko. Deep coral: Correlation alignment for deep domain adaptation. In Computer Vision–ECCV 2016 Workshops: Amsterdam, The Netherlands, October 8-10 and 15-16, 2016, Proceedings, Part III 14, pages 443–450. Springer, 2016.

Xinwei Sun, Botong Wu, Xiangyu Zheng, Chang Liu, Wei Chen, Tao Qin, and Tie-Yan Liu. Recovering latent causal factor for generalization to distributional shifts. Advances in Neural Information Processing Systems, 34:16846–16859, 2021.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Piotr Teterwak, Kuniaki Saito, Theodoros Tsiligkaridis, Kate Saenko, and Bryan A Plummer. Erm++: An improved baseline for domain generalization. arXiv preprint arXiv:2304.01973, 2023.

Philipp Tschandl, Cliff Rosendahl, and Harald Kittler. The ham10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific data, 5(1):1–9, 2018.

Haoqin Tu, Chenhang Cui, Zijun Wang, Yiyang Zhou, Bingchen Zhao, Junlin Han, Wangchunshu Zhou, Huaxiu Yao, and Cihang Xie. How many unicorns are in this image? a safety evaluation benchmark for vision llms. arXiv preprint arXiv:2311.16101, 2023.

Devis Tuia, Benjamin Kellenberger, Sara Beery, Blair R Costelloe, Silvia Zuffi, Benjamin Risse, Alexander Mathis, Mackenzie W Mathis, Frank van Langevelde, Tilo Burghardt, et al. Perspectives in machine learning for wildlife conservation. Nature communications, 13(1):792, 2022.

Vladimir N Vapnik. An overview of statistical learning theory. IEEE transactions on neural networks, 10(5): 988–999, 1999.

Hemanth Venkateswara, Jose Eusebio, Shayok Chakraborty, and Sethuraman Panchanathan. Deep hashing network for unsupervised domain adaptation. In CVPR, 2017a.

Hemanth Venkateswara, Jose Eusebio, Shayok Chakraborty, and Sethuraman Panchanathan. Deep hashing network for unsupervised domain adaptation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5018–5027, 2017b.

Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing. Learning robust global representations by penalizing local predictive power. Advances in Neural Information Processing Systems, 32, 2019.

Jindong Wang, Cuiling Lan, Chang Liu, Yidong Ouyang, Tao Qin, Wang Lu, Yiqiang Chen, Wenjun Zeng, and Philip Yu. Generalizing to unseen domains: A survey on domain generalization. IEEE Transactions on Knowledge and Data Engineering, 2022a.

Xiaosong Wang, Yifan Peng, Le Lu, Zhiyong Lu, Mohammadhadi Bagheri, and Ronald M Summers. Chestxray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. In CVPR, pages 2097–2106, 2017.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171, 2022b.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682, 2022a.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022b.

Zhixiang Wei, Lin Chen, Yi Jin, Xiaoxiao Ma, Tianle Liu, Pengyang Lin, Ben Wang, Huaian Chen, and Jinjin Zheng. Stronger, fewer, & superior: Harnessing vision foundation models for domain generalized semantic segmentation, 2023.

Olivia Wiles, Sven Gowal, Florian Stimberg, Sylvestre-Alvise Rebuffi, Ira Ktena, Krishnamurthy Dj Dvijotham, and Ali Taylan Cemgil. A fine-grained analysis on distribution shift. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=Dl4LetuLdyK.

Chaoyi Wu, Jiayu Lei, Qiaoyu Zheng, Weike Zhao, Weixiong Lin, Xiaoman Zhang, Xiao Zhou, Ziheng Zhao, Ya Zhang, Yanfeng Wang, et al. Can gpt-4v (ision) serve medical applications? case studies on gpt-4v for multimodal medical diagnosis. arXiv preprint arXiv:2310.09909, 2023a.

Wenhao Wu, Huanjin Yao, Mengxi Zhang, Yuxin Song, Wanli Ouyang, and Jingdong Wang. Gpt4vis: What can gpt-4 do for zero-shot visual recognition?, 2023b.

Minghao Xu, Jian Zhang, Bingbing Ni, Teng Li, Chengjie Wang, Qi Tian, and Wenjun Zhang. Adversarial domain adaptation with domain mixup. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pages 6502–6509, 2020.

Zhengyuan Yang, Jianfeng Wang, Linjie Li, Kevin Lin, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. Idea2img: Iterative self-refinement with gpt-4v (ision) for automatic image design and generation. arXiv preprint arXiv:2310.08541, 2023.

Xiangyu Yue, Yang Zhang, Sicheng Zhao, Alberto Sangiovanni-Vincentelli, Kurt Keutzer, and Boqing Gong. Domain randomization and pyramid consistency: Simulation-to-real generalization without accessing target domain data. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2100–2110, 2019.

Chongzhi Zhang, Mingyuan Zhang, Shanghang Zhang, Daisheng Jin, Qiang Zhou, Zhongang Cai, Haiyu Zhao, Xianglong Liu, and Ziwei Liu. Delving deep into the generalization of vision transformers under distribution shifts. In Proceedings of the IEEE/CVF conference on Computer Vision and Pattern Recognition, pages 7277–7286, 2022.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3836–3847, 2023.

Xingxuan Zhang, Peng Cui, Renzhe Xu, Linjun Zhou, Yue He, and Zheyan Shen. Deep stable learning for out-of-distribution generalization. In CVPR, pages 5372–5382, 2021.

Zhaoheng Zheng, Jingmin Wei, Xuefeng Hu, Haidong Zhu, and Ram Nevatia. Large language models are good prompt learners for low-shot image classification, 2023.

Kaiyang Zhou, Yongxin Yang, Yu Qiao, and Tao Xiang. Domain generalization with mixstyle. arXiv preprint arXiv:2104.02008, 2021.

Francis W Zwiers, Lisa V Alexander, Gabriele C Hegerl, Thomas R Knutson, James P Kossin, Phillippe Naveau, Neville Nicholls, Christoph Schär, Sonia I Seneviratne, and Xuebin Zhang. Climate extremes: challenges in estimating and understanding recent changes in the frequency and intensity of extreme climate and weather events. Climate science for serving society: research, modeling and prediction priorities, pages 339–389, 2013.

